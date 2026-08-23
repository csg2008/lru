// Unified LRU Cache Library — Tiered Storage & Read-Through Support
// SPDX-License-Identifier: MIT
// Inspired by Facebook CacheLib's NvmCache (tiered DRAM + NVMe) design
//
// Tiered storage extends a primary in-memory cache with a slower but larger
// storage backend (e.g., SSD, remote service, database). On a primary miss,
// the tiered_cache queries the backend; if found, the value is promoted to
// the primary cache (read-through). A background promotion worker can also
// warm the primary cache from the backend based on access patterns.
//
// Architecture:
//   ┌──────────────┐
//   │  tiered_cache │  (user-facing)
//   │  get() → hit?│──→ return value
//   │       miss?  │──→ query backend → promote to primary → return
//   └──────┬───────┘
//          │
//   ┌──────▼───────┐
//   │ primary_cache│  (fast, small: e.g., safe_cache)
//   └──────┬───────┘
//          │
//   ┌──────▼───────┐
//   │storage_backend│ (slow, large: e.g., RocksDB, Redis, file)
//   └──────────────┘
//
// Key design decisions (aligned with CacheLib):
//   - Backend I/O never blocks the primary cache lock
//   - Promotion is done outside the lock, write-back under per-key lock
//   - Background worker can proactively warm the primary from the backend
//   - Evicted items from primary can optionally be written back to backend

#ifndef LRU_TIERED_STORAGE_HPP
#define LRU_TIERED_STORAGE_HPP

#include <atomic>
#include <chrono>
#include <cstddef>
#include <deque>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core.hpp"
#include "detail/distributed_mutex.hpp"
#include "detail/foundation.hpp"

namespace lru {

// ============================================================================
// Storage Backend Interface
// ============================================================================

/// Abstract interface for a persistent/remote storage backend.
///
/// Implementations must be thread-safe (multiple tiered_cache instances
/// may share the same backend). All methods should be non-blocking or
/// have bounded latency; the tiered_cache will call them outside the
/// primary cache lock.
///
/// @tparam Key    Cache key type
/// @tparam Value  Cache value type
template <typename Key, typename Value>
class storage_backend {
public:
    using key_type = Key;
    using value_type = Value;

    virtual ~storage_backend() = default;

    /// Look up a key in the backend.
    /// Returns std::nullopt if the key is not found.
    virtual std::optional<Value> get(const Key& key) = 0;

    /// Store a key-value pair in the backend.
    virtual void put(const Key& key, const Value& value) = 0;

    /// Remove a key from the backend.
    /// Returns true if the key was present and removed.
    virtual bool remove(const Key& key) = 0;

    /// Check if the backend contains a key.
    virtual bool contains(const Key& key) const = 0;

    /// Get approximate number of items in the backend.
    virtual std::size_t size() const = 0;

    /// Get a human-readable name for this backend.
    virtual std::string name() const = 0;
};

// ============================================================================
// In-Memory Storage Backend (for testing and as a simple DRAM tier)
// ============================================================================

/// A simple in-memory storage backend using an unordered_map.
/// Suitable for testing, multi-level DRAM caching, or as a base class
/// for more sophisticated backends.
///
/// Locking (P1-5): per-key striped locking via
/// `striped_mutex<distributed_shared_mutex>`. Each stripe owns its own
/// mutex AND its own unordered_map, so keys hashing to different stripes
/// are fully independent — readers and writers on different stripes
/// proceed concurrently with zero contention. This lets the backend
/// scale linearly with the number of stripes under multi-threaded
/// read-heavy workloads.
template <typename Key, typename Value, typename Hash = std::hash<Key>>
class memory_storage_backend : public storage_backend<Key, Value> {
public:
    /// Construct with the default 64 stripes.
    memory_storage_backend() : memory_storage_backend(64) {}

    /// Construct with a custom stripe count. Must be > 0. Higher counts
    /// reduce contention at the cost of more memory and per-stripe
    /// bookkeeping; 64 is a reasonable default for most workloads.
    explicit memory_storage_backend(std::size_t num_stripes)
        : stripes_(num_stripes),
          shard_maps_(num_stripes) {
        if (num_stripes == 0) {
            throw std::invalid_argument(
                "memory_storage_backend: num_stripes must be > 0");
        }
    }

    std::optional<Value> get(const Key& key) override {
        auto hash = hash_(key);
        auto stripe = stripes_.stripe_for(hash);
        auto lock = stripes_.make_shared_lock(stripe);
        auto it = shard_maps_[stripe].find(key);
        if (it != shard_maps_[stripe].end()) {
            return it->second;
        }
        return std::nullopt;
    }

    void put(const Key& key, const Value& value) override {
        auto hash = hash_(key);
        auto stripe = stripes_.stripe_for(hash);
        auto lock = stripes_.make_unique_lock(stripe);
        shard_maps_[stripe][key] = value;
    }

    bool remove(const Key& key) override {
        auto hash = hash_(key);
        auto stripe = stripes_.stripe_for(hash);
        auto lock = stripes_.make_unique_lock(stripe);
        return shard_maps_[stripe].erase(key) > 0;
    }

    bool contains(const Key& key) const override {
        auto hash = hash_(key);
        auto stripe = stripes_.stripe_for(hash);
        auto lock = stripes_.make_shared_lock(stripe);
        return shard_maps_[stripe].find(key) != shard_maps_[stripe].end();
    }

    std::size_t size() const override {
        // Aggregate across all stripes. Each stripe's map is read under
        // a shared lock; concurrent writers to other stripes are not
        // blocked. The result is a consistent point-in-time snapshot
        // modulo concurrent mutations on each individual stripe.
        std::size_t total = 0;
        for (std::size_t s = 0; s < stripes_.size(); ++s) {
            auto lock = stripes_.make_shared_lock(s);
            total += shard_maps_[s].size();
        }
        return total;
    }

    std::string name() const override {
        return "memory_storage_backend";
    }

    void clear() {
        // P2-F: Per-stripe locking instead of a global write_all lock.
        //
        // The previous implementation used `striped_mutex_write_all_guard`,
        // which acquires every stripe's exclusive lock simultaneously. This
        // made `clear()` the only operation in the backend that blocks all
        // readers and writers across every stripe — a non-trivial stall when
        // the backend is large (millions of items) or `clear()` is invoked
        // from a warm-restart / reinit path while traffic is still flowing.
        //
        // The fix acquires each stripe's exclusive lock individually and
        // swaps the map with an empty one (O(1) pointer move, deallocation
        // happens outside the lock). This allows concurrent `get`/`put`/
        // `remove` operations on every OTHER stripe to proceed while we
        // drain one stripe at a time.
        //
        // Semantics: `clear()` is no longer atomic across stripes — a
        // concurrent writer that hits stripe N after we've cleared it but
        // before we've cleared stripe N+1 will see an empty map at N and a
        // non-empty map at N+1. This is acceptable for the documented use
        // cases (warm restart, reinit, test teardown) and matches the
        // per-stripe consistency already exposed by `size()`. Callers that
        // need a strictly atomic clear can acquire an external coordinator
        // lock before invoking `clear()`.
        for (std::size_t s = 0; s < stripes_.size(); ++s) {
            ankerl::unordered_dense::map<Key, Value, Hash> empty;
            {
                auto lock = stripes_.make_unique_lock(s);
                shard_maps_[s].swap(empty);
                // `empty` now holds the old map's contents; it will be
                // deallocated when `empty` goes out of scope, AFTER the
                // stripe lock is released. This keeps the (potentially
                // expensive) deallocation off the critical path.
            }
        }
    }

    /// Returns the number of stripes (for diagnostics/tests).
    std::size_t num_stripes() const noexcept { return stripes_.size(); }

private:
    // distributed_shared_mutex avoids the MinGW pthread_rwlock_t bugs
    // described in AGENTS.md; std::shared_mutex would return EINVAL
    // under mixed read/write contention on Windows.
    mutable detail::striped_mutex<detail::distributed_shared_mutex> stripes_;
    Hash hash_;
    // One map per stripe. Indexed by stripes_.stripe_for(hash).
    // mutable because contains()/size() are const but need shared locks.
    mutable std::vector<ankerl::unordered_dense::map<Key, Value, Hash>>
        shard_maps_;
};

// ============================================================================
// Tiered Cache
// ============================================================================

// ----------------------------------------------------------------------------
// O5: Backend Circuit Breaker
// ----------------------------------------------------------------------------
//
// Protects the tiered cache against cascading backend failures. When the
// backend repeatedly throws exceptions (e.g., network errors, disk failures,
// timeouts), the circuit breaker "opens" and short-circuits subsequent
// requests — they fail fast (returning nullopt) without hitting the backend.
// This gives the backend time to recover and prevents the tiered cache from
// amplifying the failure by piling on requests.
//
// State machine:
//   CLOSED → (errors ≥ error_threshold) → OPEN
//   OPEN   → (cooldown elapsed)         → HALF_OPEN
//   HALF_OPEN → (success ≥ success_threshold) → CLOSED
//   HALF_OPEN → (any failure)            → OPEN
//
// Thread safety: all state transitions are atomic. The breaker is designed
// for high read concurrency — allow_request() is a single atomic load + branch
// on the fast path (CLOSED state).

/// Circuit breaker states.
enum class circuit_state : uint8_t {
    closed,     ///< Normal operation — requests pass through to the backend.
    open,       ///< Tripped — requests fail fast without hitting the backend.
    half_open,  ///< Probing — a limited number of requests are allowed to test
                ///< if the backend has recovered.
};

/// Configuration for the backend circuit breaker.
struct circuit_breaker_config {
    /// Number of consecutive backend failures (within the sliding window
    /// implied by `error_reset_interval`) that trips the breaker to OPEN.
    /// Set to 0 to disable the circuit breaker entirely (always CLOSED).
    std::size_t error_threshold = 5;

    /// How long to wait in the OPEN state before transitioning to HALF_OPEN.
    /// During this period, all backend requests fail fast.
    std::chrono::milliseconds cooldown{5000};

    /// Number of consecutive successes in HALF_OPEN required to close the
    /// circuit (transition back to CLOSED). A single failure in HALF_OPEN
    /// re-opens the circuit.
    std::size_t success_threshold = 2;

    /// Interval after which the error counter resets if no new failures
    /// occur. This prevents a slow drip of failures over hours from
    /// tripping the breaker. Set to 0 to disable (counter never resets).
    std::chrono::milliseconds error_reset_interval{60000};
};

/// Thread-safe circuit breaker for backend failure protection.
///
/// Usage:
///   backend_circuit_breaker breaker;
///   if (breaker.allow_request()) {
///       try {
///           auto v = backend.get(key);
///           breaker.record_success();
///           ...
///       } catch (...) {
///           breaker.record_failure();
///           throw;
///       }
///   } else {
///       // Circuit is OPEN — fail fast.
///       return std::nullopt;
///   }
class backend_circuit_breaker {
public:
    explicit backend_circuit_breaker(const circuit_breaker_config& cfg = {})
        : config_(cfg) {}

    /// Check if a backend request should be allowed to proceed.
    ///
    /// - CLOSED: always returns true.
    /// - OPEN: returns false (fail fast). If the cooldown has elapsed,
    ///   atomically transitions to HALF_OPEN and returns true (allowing
    ///   one probe request).
    /// - HALF_OPEN: returns true (allow the probe to proceed).
    ///
    /// Returns false ONLY when the circuit is OPEN and the cooldown has
    /// not elapsed. In that case the caller should short-circuit (return
    /// nullopt / skip the backend call) without hitting the backend.
    bool allow_request() {
        // Fast path: config disabled or CLOSED state.
        if (config_.error_threshold == 0) return true;

        uint8_t state = state_.load(std::memory_order_acquire);
        if (state == static_cast<uint8_t>(circuit_state::closed)) {
            return true;
        }

        if (state == static_cast<uint8_t>(circuit_state::open)) {
            // Check if cooldown has elapsed.
            const auto now = now_ms();
            const auto last = last_failure_ms_.load(std::memory_order_acquire);
            if (now - last < config_.cooldown.count()) {
                // Still cooling down — fail fast.
                return false;
            }
            // Cooldown elapsed — try to transition to HALF_OPEN.
            // Only one thread should make this transition; use CAS.
            uint8_t expected = static_cast<uint8_t>(circuit_state::open);
            if (state_.compare_exchange_strong(
                    expected, static_cast<uint8_t>(circuit_state::half_open),
                    std::memory_order_acq_rel)) {
                // This thread won the race — it becomes the probe.
                // Reset success counter for the half-open probing phase.
                half_open_successes_.store(0, std::memory_order_release);
                return true;
            }
            // Lost the race — another thread already transitioned to
            // HALF_OPEN. Fall through to the HALF_OPEN case.
            state = state_.load(std::memory_order_acquire);
        }

        if (state == static_cast<uint8_t>(circuit_state::half_open)) {
            // Allow the request — the probe is in progress.
            return true;
        }

        // Should not reach here, but fail-safe: allow the request.
        return true;
    }

    /// Record a successful backend operation. In HALF_OPEN state,
    /// consecutive successes count toward closing the circuit.
    void record_success() {
        if (config_.error_threshold == 0) return;
        const uint8_t state = state_.load(std::memory_order_acquire);
        if (state == static_cast<uint8_t>(circuit_state::half_open)) {
            std::size_t successes =
                half_open_successes_.fetch_add(1, std::memory_order_acq_rel) + 1;
            if (successes >= config_.success_threshold) {
                // Enough successes — close the circuit.
                uint8_t expected = static_cast<uint8_t>(circuit_state::half_open);
                state_.compare_exchange_strong(
                    expected, static_cast<uint8_t>(circuit_state::closed),
                    std::memory_order_acq_rel);
                // Reset error count for the next failure cycle.
                error_count_.store(0, std::memory_order_release);
            }
        } else if (state == static_cast<uint8_t>(circuit_state::closed)) {
            // A success in CLOSED state resets the error count (sliding
            // window behavior — only CONSECUTIVE failures trip the breaker).
            error_count_.store(0, std::memory_order_release);
        }
        // A success in OPEN state shouldn't happen (allow_request() returns
        // false), but if it does, ignore it.
    }

    /// Record a failed backend operation. In CLOSED state, increments
    /// the error count and trips the breaker if the threshold is reached.
    /// In HALF_OPEN state, immediately re-opens the circuit.
    void record_failure() {
        if (config_.error_threshold == 0) return;
        const auto now = now_ms();
        last_failure_ms_.store(now, std::memory_order_release);

        const uint8_t state = state_.load(std::memory_order_acquire);
        if (state == static_cast<uint8_t>(circuit_state::half_open)) {
            // A failure during probing re-opens the circuit immediately.
            state_.store(static_cast<uint8_t>(circuit_state::open),
                         std::memory_order_release);
            return;
        }
        if (state == static_cast<uint8_t>(circuit_state::closed)) {
            // Increment error count.
            std::size_t count =
                error_count_.fetch_add(1, std::memory_order_acq_rel) + 1;
            if (count >= config_.error_threshold) {
                // Trip the breaker.
                uint8_t expected = static_cast<uint8_t>(circuit_state::closed);
                state_.compare_exchange_strong(
                    expected, static_cast<uint8_t>(circuit_state::open),
                    std::memory_order_acq_rel);
            }
        }
        // A failure in OPEN state is unexpected (allow_request() should have
        // returned false), but if it happens, just refresh last_failure_ms_
        // (already done above) to extend the cooldown.
    }

    /// Query the current circuit state. For diagnostics/tests.
    circuit_state state() const {
        return static_cast<circuit_state>(
            state_.load(std::memory_order_acquire));
    }

    /// Query the current error count (for diagnostics). May be stale.
    std::size_t error_count() const {
        return error_count_.load(std::memory_order_acquire);
    }

    /// Reset the breaker to CLOSED state with zero error count.
    /// For tests and manual recovery triggers.
    void reset() {
        state_.store(static_cast<uint8_t>(circuit_state::closed),
                     std::memory_order_release);
        error_count_.store(0, std::memory_order_release);
        half_open_successes_.store(0, std::memory_order_release);
        last_failure_ms_.store(0, std::memory_order_release);
    }

    const circuit_breaker_config& config() const { return config_; }
    void set_config(const circuit_breaker_config& cfg) { config_ = cfg; }

private:
    static int64_t now_ms() noexcept {
        return static_cast<int64_t>(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count());
    }

    circuit_breaker_config config_;
    /// Stored as uint8_t (not circuit_state) for atomic compatibility.
    alignas(64) std::atomic<uint8_t> state_{
        static_cast<uint8_t>(circuit_state::closed)};
    alignas(64) std::atomic<std::size_t> error_count_{0};
    alignas(64) std::atomic<std::size_t> half_open_successes_{0};
    alignas(64) std::atomic<int64_t> last_failure_ms_{0};
};

/// A two-tier cache that combines a fast in-memory primary cache with a
/// slower storage backend. On a primary miss, it queries the backend
/// (read-through) and promotes the value to the primary cache.
///
/// Inspired by CacheLib's NvmCache architecture:
///   - Backend I/O is performed outside the primary cache lock
///   - Double-checked locking prevents thundering herd on the same key
///   - Optional write-back: evicted items can be persisted to the backend
///   - Optional background promotion worker for proactive warming
///   - O5: Backend circuit breaker protects against cascading failures
///
/// @tparam PrimaryCache  The primary in-memory cache type (e.g., safe_cache<K,V>)
/// @tparam Backend       The storage backend type (must derive from storage_backend<K,V>)
template <typename PrimaryCache, typename Backend>
class tiered_cache {
public:
    using primary_type = PrimaryCache;
    using backend_type = Backend;
    using key_type = typename primary_type::key_type;
    using mapped_type = typename primary_type::mapped_type;
    using value_type = mapped_type;
    using size_type = typename primary_type::size_type;

    /// Configuration for the tiered cache.
    struct config {
        /// When true, items evicted from the primary cache are written to
        /// the backend before being discarded (write-back / write-through).
        bool write_back_on_evict = false;

        /// When true, a primary miss triggers a backend lookup (read-through).
        bool read_through = true;

        /// Interval for the background promotion worker.
        /// Zero means no background worker.
        std::chrono::milliseconds promotion_interval{0};

        /// Maximum number of items to promote per background cycle.
        std::size_t promotion_batch_size = 64;

        /// O5: Circuit breaker configuration for the backend. When
        /// `error_threshold > 0`, repeated backend failures trip the
        /// breaker and short-circuit subsequent requests (fail fast)
        /// until the cooldown elapses. Set `error_threshold = 0` to
        /// disable (always pass through to the backend).
        circuit_breaker_config breaker;

        /// T-G5: Async writeback queue capacity. When > 0, evicted dirty
        /// items are enqueued for a background worker to persist via
        /// `backend_->put()` instead of blocking the eviction callback.
        /// When 0, writeback is synchronous (legacy behavior). When the
        /// queue is full, the oldest pending item is dropped and
        /// `writeback_dropped_count` is incremented.
        std::size_t async_writeback_queue_capacity = 4096;

        /// T-G5: Interval at which the background writeback worker drains
        /// the queue. Default 10ms balances latency vs backend batch
        /// coalescing.
        std::chrono::milliseconds async_writeback_interval{10};
    };

    // --------------------------------------------------------------------
    // Construction
    // --------------------------------------------------------------------

    /// Construct a tiered cache with a primary cache and a shared backend.
    /// The backend is not owned (must outlive this tiered_cache).
    tiered_cache(primary_type primary, Backend& backend, const config& cfg = {})
        : primary_(std::move(primary))
        , backend_(&backend)
        , config_(cfg)
        , breaker_(cfg.breaker)
        , inflight_maps_(inflight_stripes_.size())
    {
        setup_writeback();
        if (config_.promotion_interval.count() > 0) {
            start_promotion_worker(config_.promotion_interval);
        }
    }

    /// Construct with a primary cache capacity and a shared backend.
    tiered_cache(size_type max_size, Backend& backend, const config& cfg = {})
        : primary_(max_size)
        , backend_(&backend)
        , config_(cfg)
        , breaker_(cfg.breaker)
        , inflight_maps_(inflight_stripes_.size())
    {
        setup_writeback();
        if (config_.promotion_interval.count() > 0) {
            start_promotion_worker(config_.promotion_interval);
        }
    }

    ~tiered_cache() {
        stop_promotion_worker();
        stop_writeback_worker();
    }

    // Non-copyable, non-movable (contains running worker thread)
    tiered_cache(const tiered_cache&) = delete;
    tiered_cache& operator=(const tiered_cache&) = delete;
    tiered_cache(tiered_cache&&) = delete;
    tiered_cache& operator=(tiered_cache&&) = delete;

    // --------------------------------------------------------------------
    // Core API
    // --------------------------------------------------------------------

    /// Get a value from the tiered cache.
    /// If the value is in the primary cache, returns it directly.
    /// If not and read_through is enabled, queries the backend and
    /// promotes the value to the primary cache.
    ///
    /// Thundering-herd prevention: when multiple threads miss on the same
    /// key concurrently, only the leader queries the backend; followers
    /// wait on a shared_future and read the promoted value from the
    /// primary cache once the leader completes.
    read_handle<mapped_type> get(const key_type& key) {
        // Phase 1: Fast path — check primary cache
        auto handle = primary_.get(key);
        if (handle) {
            ++primary_hits_;
            return handle;
        }

        if (!config_.read_through) {
            ++primary_misses_;
            return {};
        }

        // Phase 2: In-flight deduplication. Look up the in-flight table
        // under a short lock; if a leader is already fetching this key,
        // become a follower and wait on its shared_future.
        {
            std::shared_future<std::optional<mapped_type>> follower_fut;
            {
                // O4: striped lock — only conflicts with other threads
                // fetching the SAME key (or a key hashing to the same
                // stripe), not with unrelated fetches.
                const std::size_t stripe = inflight_stripe_for(key);
                auto inflight_lock =
                    inflight_stripes_.make_unique_lock(stripe);
                auto& map = inflight_maps_[stripe];
                auto it = map.find(key);
                if (it != map.end()) {
                    follower_fut = it->second;
                }
            }
            if (follower_fut.valid()) {
                // Follower path — wait for the leader, then re-check primary.
                follower_fut.wait();
                auto handle2 = primary_.get(key);
                if (handle2) {
                    ++primary_hits_;
                    ++inflight_followers_;
                    return handle2;
                }
                // Leader observed a backend miss; propagate.
                ++primary_misses_;
                ++backend_misses_;
                ++inflight_followers_;
                return {};
            }
        }

        // Phase 3: Try to become the leader. Create our promise + shared
        // future up front, then race for the table slot under the lock.
        std::promise<std::optional<mapped_type>> leader_prom;
        std::shared_future<std::optional<mapped_type>> leader_fut =
            leader_prom.get_future().share();
        bool is_leader = false;
        {
            // O4: striped lock — race for leader slot under the per-key stripe.
            const std::size_t stripe = inflight_stripe_for(key);
            auto inflight_lock =
                inflight_stripes_.make_unique_lock(stripe);
            auto& map = inflight_maps_[stripe];
            auto it = map.find(key);
            if (it == map.end()) {
                map.emplace(key, leader_fut);
                is_leader = true;
            }
            // else: lost the race — we'll fall through and become a follower
            // via the shared_future we just retrieved from the table.
            else {
                leader_fut = it->second;
            }
        }

        if (!is_leader) {
            // Lost the leader race — follower path.
            leader_fut.wait();
            auto handle2 = primary_.get(key);
            if (handle2) {
                ++primary_hits_;
                ++inflight_followers_;
                return handle2;
            }
            ++primary_misses_;
            ++backend_misses_;
            ++inflight_followers_;
            return {};
        }

        // Phase 4: Leader — query backend outside any lock.
        //
        // O5: Circuit breaker check. If the breaker is OPEN (tripped due
        // to repeated backend failures), short-circuit: erase the in-flight
        // entry, notify followers with nullopt, and return empty without
        // hitting the backend. This protects the backend from being
        // hammered while it's recovering.
        if (!breaker_.allow_request()) {
            // Circuit is OPEN — fail fast.
            {
                const std::size_t stripe = inflight_stripe_for(key);
                auto inflight_lock =
                    inflight_stripes_.make_unique_lock(stripe);
                inflight_maps_[stripe].erase(key);
            }
            // Notify followers (they'll see nullopt and propagate the miss).
            try { leader_prom.set_value(std::nullopt); } catch (...) {}
            ++primary_misses_;
            ++backend_misses_;
            ++circuit_breaker_rejections_;
            return {};
        }

        std::optional<mapped_type> backend_value;
        bool backend_threw = false;
        std::exception_ptr backend_ex;
        try {
            backend_value = backend_->get(key);
            // O5: Record success — may close the circuit if in HALF_OPEN.
            breaker_.record_success();
        } catch (...) {
            backend_threw = true;
            backend_ex = std::current_exception();
            // O5: Record failure — may trip or re-open the circuit.
            breaker_.record_failure();
        }

        if (backend_threw) {
            // Erase from in-flight table first, then propagate exception
            // to followers so they don't wait forever.
            {
                // O4: striped lock — same stripe as the emplace above.
                const std::size_t stripe = inflight_stripe_for(key);
                auto inflight_lock =
                    inflight_stripes_.make_unique_lock(stripe);
                inflight_maps_[stripe].erase(key);
            }
            try {
                leader_prom.set_exception(backend_ex);
            } catch (...) {}
            // Re-throw on the leader path so callers observe backend errors.
            std::rethrow_exception(backend_ex);
        }

        // Phase 5: Promote to primary cache BEFORE fulfilling the promise.
        // Followers waking up from future.wait() will then find the value
        // already in the primary cache.
        if (backend_value) {
            primary_.set(key, *backend_value);
            ++backend_hits_;
            ++promotions_;
        }

        // Phase 6: Erase from in-flight table. New requests arriving after
        // this point will find the value in the primary cache (or, on a
        // backend miss, become fresh leaders themselves).
        {
            // O4: striped lock — same stripe as the emplace above.
            const std::size_t stripe = inflight_stripe_for(key);
            auto inflight_lock =
                inflight_stripes_.make_unique_lock(stripe);
            inflight_maps_[stripe].erase(key);
        }

        // Phase 7: Fulfill the promise to wake all followers.
        try {
            leader_prom.set_value(backend_value);
        } catch (...) {
            // Promise already satisfied — shouldn't happen, ignore.
        }

        if (!backend_value) {
            ++primary_misses_;
            ++backend_misses_;
            return {};
        }

        return primary_.get(key);
    }

    /// Set a value in both the primary cache and the backend.
    template <typename V>
    void set(const key_type& key, V&& value) {
        primary_.set(key, value);
        if (config_.write_back_on_evict) {
            // T-G5: write-through path persists immediately. The eviction
            // callback (async or sync) handles write-back for evictions.
            // We do NOT enqueue here because the value is already in the
            // backend — enqueuing would create redundant puts.
            backend_->put(key, value);
            ++writebacks_;
        }
    }

    /// Delete a key from both the primary cache and the backend.
    bool del(const key_type& key) {
        auto primary_result = primary_.del(key);
        backend_->remove(key);
        return primary_result;
    }

    /// Check if the key exists in either tier.
    bool contains(const key_type& key) const {
        if (primary_.contains(key)) return true;
        return backend_->contains(key);
    }

    /// Peek at the primary cache only (no backend lookup).
    auto peek(const key_type& key) const {
        return primary_.peek(key);
    }

    // --------------------------------------------------------------------
    // Capacity
    // --------------------------------------------------------------------

    bool empty() const { return primary_.empty(); }
    size_type size() const { return primary_.size(); }
    size_type max_size() const { return primary_.max_size(); }

    // --------------------------------------------------------------------
    // Promotion worker
    // --------------------------------------------------------------------

    /// Start a background promotion worker that periodically warms the
    /// primary cache from the backend. The worker scans the backend for
    /// keys and promotes the most recently accessed items.
    void start_promotion_worker(std::chrono::milliseconds interval) {
        stop_promotion_worker();
        promotion_running_.store(true, std::memory_order_release);
        promotion_thread_ = std::thread([this, interval]() {
            while (promotion_running_.load(std::memory_order_acquire)) {
                std::this_thread::sleep_for(interval);
                if (!promotion_running_.load(std::memory_order_acquire)) break;
                promote_from_backend();
            }
        });
    }

    /// Stop the background promotion worker.
    void stop_promotion_worker() {
        promotion_running_.store(false, std::memory_order_release);
        if (promotion_thread_.joinable()) {
            promotion_thread_.join();
        }
    }

    // --------------------------------------------------------------------
    // T-G5: Async writeback worker
    // --------------------------------------------------------------------

    /// Start the background writeback worker that drains the async
    /// writeback queue and persists dirty items to the backend. Called
    /// automatically by `setup_writeback()` when the queue is enabled.
    void start_writeback_worker(std::chrono::milliseconds interval) {
        stop_writeback_worker();
        writeback_running_.store(true, std::memory_order_release);
        writeback_thread_ = std::thread([this, interval]() {
            while (writeback_running_.load(std::memory_order_acquire)) {
                std::unique_lock<std::mutex> lk(writeback_mtx_);
                writeback_cv_.wait_for(lk, interval, [this] {
                    return !writeback_queue_.empty() ||
                           !writeback_running_.load(std::memory_order_acquire);
                });
                // Drain the queue under the lock, then release before
                // calling backend_->put() so a slow backend doesn't
                // block the eviction callback.
                std::deque<std::pair<key_type, mapped_type>> batch;
                batch.swap(writeback_queue_);
                lk.unlock();
                for (auto& [k, v] : batch) {
                    try {
                        backend_->put(k, v);
                        ++writebacks_;
                    } catch (...) {
                        // Backend error on a single item must not crash
                        // the writeback thread (an uncaught exception in
                        // a std::thread triggers std::terminate). Drop
                        // the offending item and keep draining.
                        ++writeback_dropped_;
                    }
                }
            }
            // Final drain after stop signal — bounded by a 5s deadline
            // so a slow or hung backend cannot block destruction
            // indefinitely. The queue is swapped out under a short lock
            // so the mutex is NOT held during backend I/O; remaining
            // items after the timeout are dropped and counted.
            std::deque<std::pair<key_type, mapped_type>> remaining;
            {
                std::lock_guard<std::mutex> lk(writeback_mtx_);
                remaining.swap(writeback_queue_);
            }
            const auto deadline = std::chrono::steady_clock::now()
                                + std::chrono::seconds(5);
            while (!remaining.empty() &&
                   std::chrono::steady_clock::now() < deadline) {
                auto item = std::move(remaining.front());
                remaining.pop_front();
                try {
                    backend_->put(item.first, item.second);
                    ++writebacks_;
                } catch (...) {
                    ++writeback_dropped_;
                }
            }
            if (!remaining.empty()) {
                writeback_dropped_.fetch_add(
                    remaining.size(), std::memory_order_relaxed);
            }
        });
    }

    /// Stop the background writeback worker. Performs a final drain so
    /// no dirty items are lost on shutdown.
    void stop_writeback_worker() {
        writeback_running_.store(false, std::memory_order_release);
        writeback_cv_.notify_all();
        if (writeback_thread_.joinable()) {
            writeback_thread_.join();
        }
    }

    /// T-G5: Number of dirty items dropped because the async writeback
    /// queue was full.
    std::size_t writeback_dropped_count() const noexcept {
        return writeback_dropped_.load(std::memory_order_relaxed);
    }

    /// T-G5: Current number of items pending in the async writeback queue.
    std::size_t writeback_queue_size() const noexcept {
        std::lock_guard<std::mutex> lk(writeback_mtx_);
        return writeback_queue_.size();
    }

private:
    /// T-G5: Wire up the eviction callback. When
    /// `async_writeback_queue_capacity > 0`, the callback enqueues the
    /// dirty item and returns immediately (O(1) amortized — a lock+copy
    /// + notify). When the queue is full, the oldest pending item is
    /// dropped (overwritten) to make room for the newer one, and
    /// `writeback_dropped_` is incremented. When the capacity is 0,
    /// the callback calls `backend_->put()` synchronously (legacy
    /// behavior, blocks the eviction path).
    void setup_writeback() {
        if (!config_.write_back_on_evict) return;
        if (config_.async_writeback_queue_capacity > 0) {
            primary_.on_evict([this](const key_type& key, const mapped_type& value) {
                {
                    std::lock_guard<std::mutex> lk(writeback_mtx_);
                    if (writeback_queue_.size() >= config_.async_writeback_queue_capacity) {
                        // Queue full — drop the oldest pending item to
                        // make room. This favors recency over age under
                        // sustained writeback pressure.
                        writeback_queue_.erase(writeback_queue_.begin());
                        writeback_dropped_.fetch_add(1, std::memory_order_relaxed);
                    }
                    writeback_queue_.emplace_back(key, value);
                }
                writeback_cv_.notify_one();
            });
            start_writeback_worker(config_.async_writeback_interval);
        } else {
            // Synchronous writeback (legacy path — blocks eviction).
            primary_.on_evict([this](const key_type& key, const mapped_type& value) {
                backend_->put(key, value);
                ++writebacks_;
            });
        }
    }

public:

    /// Manually trigger a single promotion cycle from the backend.
    std::size_t promote_from_backend() {
        // Subclasses can override the promotion strategy.
        // Default: iterate keys from primary's snapshot, check backend
        // for missing items, and promote the first N.
        std::size_t promoted = 0;
        // This base implementation relies on the backend providing a way
        // to enumerate keys. Since storage_backend doesn't require key
        // enumeration, the default promotion is a no-op.
        // Users should subclass and override promote_from_backend() or
        // provide a custom promotion strategy via the constructor.
        return promoted;
    }

    // --------------------------------------------------------------------
    // Statistics
    // --------------------------------------------------------------------

    struct stats {
        std::size_t primary_hits = 0;
        std::size_t primary_misses = 0;
        std::size_t backend_hits = 0;
        std::size_t backend_misses = 0;
        std::size_t promotions = 0;
        std::size_t writebacks = 0;
        // Number of get() calls served by waiting on an in-flight leader
        // rather than issuing a duplicate backend request.
        std::size_t inflight_followers = 0;
        // O5: Number of get() calls short-circuited by the circuit breaker.
        std::size_t circuit_breaker_rejections = 0;
        // T-G5: Dirty items dropped because the async writeback queue
        // was full. Non-zero under sustained eviction pressure with a
        // slow backend.
        std::size_t writeback_dropped = 0;

        double primary_hit_rate() const {
            auto total = primary_hits + primary_misses;
            return total > 0
                ? static_cast<double>(primary_hits) / static_cast<double>(total)
                : 0.0;
        }

        double backend_hit_rate() const {
            auto total = backend_hits + backend_misses;
            return total > 0
                ? static_cast<double>(backend_hits) / static_cast<double>(total)
                : 0.0;
        }

        double overall_hit_rate() const {
            auto total = primary_hits + primary_misses;
            return total > 0
                ? static_cast<double>(primary_hits + backend_hits) / static_cast<double>(total)
                : 0.0;
        }
    };

    stats get_stats() const {
        stats s;
        s.primary_hits = primary_hits_.load(std::memory_order_relaxed);
        s.primary_misses = primary_misses_.load(std::memory_order_relaxed);
        s.backend_hits = backend_hits_.load(std::memory_order_relaxed);
        s.backend_misses = backend_misses_.load(std::memory_order_relaxed);
        s.promotions = promotions_.load(std::memory_order_relaxed);
        s.writebacks = writebacks_.load(std::memory_order_relaxed);
        s.inflight_followers = inflight_followers_.load(std::memory_order_relaxed);
        s.circuit_breaker_rejections =
            circuit_breaker_rejections_.load(std::memory_order_relaxed);
        s.writeback_dropped = writeback_dropped_.load(std::memory_order_relaxed);
        return s;
    }

    // --------------------------------------------------------------------
    // O5: Circuit breaker access
    // --------------------------------------------------------------------

    /// Access the backend circuit breaker (for config inspection/state).
    backend_circuit_breaker& circuit_breaker() noexcept { return breaker_; }
    const backend_circuit_breaker& circuit_breaker() const noexcept {
        return breaker_;
    }

    /// Returns the number of keys currently being fetched by a leader
    /// (i.e., the size of the in-flight table). For diagnostics/tests.
    ///
    /// O4: With striped locking, there's no single lock that protects
    /// the entire map. We acquire all stripes (in ascending order to
    /// avoid deadlock) to get a consistent snapshot. This is O(N) on
    /// the number of stripes (default 64) but is only called from
    /// diagnostics paths, not the hot path.
    std::size_t inflight_count() const {
        inflight_stripes_.lock_all();
        std::size_t n = 0;
        for (const auto& m : inflight_maps_) n += m.size();
        inflight_stripes_.unlock_all();
        return n;
    }

    // --------------------------------------------------------------------
    // Direct access
    // --------------------------------------------------------------------

    primary_type& primary() noexcept { return primary_; }
    const primary_type& primary() const noexcept { return primary_; }
    Backend& backend() noexcept { return *backend_; }
    const Backend& backend() const noexcept { return *backend_; }

private:
    primary_type primary_;
    Backend* backend_;  // non-owning
    config config_;

    // Statistics (atomic for lock-free updates)
    std::atomic<std::size_t> primary_hits_{0};
    std::atomic<std::size_t> primary_misses_{0};
    std::atomic<std::size_t> backend_hits_{0};
    std::atomic<std::size_t> backend_misses_{0};
    std::atomic<std::size_t> promotions_{0};
    std::atomic<std::size_t> writebacks_{0};
    // Number of get() calls that were served by waiting on an in-flight
    // leader instead of querying the backend themselves (thundering herd
    // prevention metric).
    std::atomic<std::size_t> inflight_followers_{0};
    // O5: Number of get() calls that were short-circuited by the circuit
    // breaker (OPEN state) without hitting the backend.
    std::atomic<std::size_t> circuit_breaker_rejections_{0};

    // O5: Backend circuit breaker. Initialized from config_.breaker.
    backend_circuit_breaker breaker_;

    // In-flight table for thundering-herd prevention. Maps a key to the
    // shared_future of the leader currently fetching that key from the
    // backend. Followers find their future here and wait on it instead
    // of issuing a duplicate backend request.
    //
    // O4: striped locking — instead of a single global mutex, the
    // inflight table is sharded into N stripes (default 64), each with
    // its OWN mutex (from inflight_stripes_) and its OWN unordered_map
    // (from inflight_maps_). This is critical: concurrent emplace() on
    // a single shared std::unordered_map would race on rehash even
    // under striped locking, so each stripe MUST have its own map. With
    // per-stripe maps, operations on different keys (hashing to
    // different stripes) are truly independent — no shared data is
    // mutated, eliminating both lock contention AND rehash races. The
    // stripe is selected by Hash(key) % num_stripes, so two threads
    // racing for the SAME key still serialize (which is the desired
    // behavior — only one should become the leader).
    mutable detail::striped_mutex<std::mutex> inflight_stripes_;
    mutable std::vector<std::unordered_map<
        key_type, std::shared_future<std::optional<mapped_type>>>> inflight_maps_;

    /// O4: Compute the stripe index for a key. Uses std::hash<key_type>
    /// which is the same hash function the primary cache uses by default.
    std::size_t inflight_stripe_for(const key_type& key) const noexcept {
        return inflight_stripes_.stripe_for(
            std::hash<key_type>{}(key));
    }

    // Background promotion worker
    std::atomic<bool> promotion_running_{false};
    std::thread promotion_thread_;

    // T-G5: Async writeback queue + worker. The eviction callback enqueues
    // dirty items here under writeback_mtx_; the writeback_thread_ drains
    // the queue and calls backend_->put() outside the eviction path so a
    // slow backend cannot block cache operations.
    mutable std::mutex writeback_mtx_;
    std::condition_variable writeback_cv_;
    std::deque<std::pair<key_type, mapped_type>> writeback_queue_;
    std::atomic<bool> writeback_running_{false};
    std::thread writeback_thread_;
    std::atomic<std::size_t> writeback_dropped_{0};
};

// ============================================================================
// Key-Enumerating Storage Backend Extension
// ============================================================================

/// Extended storage backend interface that supports key enumeration.
/// Required for background promotion workers that need to scan the backend.
template <typename Key, typename Value>
class enumerable_storage_backend : public storage_backend<Key, Value> {
public:
    /// Enumerate all keys in the backend.
    /// Used by the promotion worker to warm the primary cache.
    virtual std::vector<Key> enumerate_keys() const = 0;
};

// ============================================================================
// Enumerable Memory Storage Backend
// ============================================================================

/// In-memory storage backend with key enumeration support.
template <typename Key, typename Value, typename Hash = std::hash<Key>>
class enumerable_memory_backend : public enumerable_storage_backend<Key, Value> {
public:
    std::optional<Value> get(const Key& key) override {
        std::lock_guard lock(mutex_);
        auto it = data_.find(key);
        if (it != data_.end()) {
            return it->second;
        }
        return std::nullopt;
    }

    void put(const Key& key, const Value& value) override {
        std::lock_guard lock(mutex_);
        data_[key] = value;
    }

    bool remove(const Key& key) override {
        std::lock_guard lock(mutex_);
        return data_.erase(key) > 0;
    }

    bool contains(const Key& key) const override {
        std::lock_guard lock(mutex_);
        return data_.find(key) != data_.end();
    }

    std::size_t size() const override {
        std::lock_guard lock(mutex_);
        return data_.size();
    }

    std::string name() const override {
        return "enumerable_memory_backend";
    }

    std::vector<Key> enumerate_keys() const override {
        std::shared_lock lock(mutex_);
        std::vector<Key> keys;
        keys.reserve(data_.size());
        for (const auto& [k, v] : data_) {
            keys.push_back(k);
        }
        return keys;
    }

    void clear() {
        std::unique_lock lock(mutex_);
        data_.clear();
    }

private:
    mutable std::mutex mutex_;
    ankerl::unordered_dense::map<Key, Value, Hash> data_;
};

// ============================================================================
// Tiered Cache with Background Promotion (for enumerable backends)
// ============================================================================

/// Specialization of tiered_cache for enumerable backends.
/// Provides automatic background promotion that scans the backend and
/// promotes warm items to the primary cache.
template <typename PrimaryCache, typename Key, typename Value>
class promoting_tiered_cache : public tiered_cache<PrimaryCache, enumerable_storage_backend<Key, Value>> {
public:
    using base_type = tiered_cache<PrimaryCache, enumerable_storage_backend<Key, Value>>;
    using key_type = Key;
    using value_type = Value;

    /// Construct with a primary cache, enumerable backend, and config.
    promoting_tiered_cache(PrimaryCache primary,
                           enumerable_storage_backend<Key, Value>& backend,
                           typename base_type::config cfg = {})
        : base_type(std::move(primary), backend, std::move(cfg)) {}

    /// Promote items from the backend that are not in the primary cache.
    /// Promotes up to `batch_size` items per call.
    std::size_t promote_from_backend() override {
        auto keys = this->backend().enumerate_keys();
        std::size_t promoted = 0;
        const std::size_t batch_size = this->config_.promotion_batch_size;
        for (const auto& key : keys) {
            if (promoted >= batch_size) break;
            if (!this->primary().contains(key)) {
                auto value = this->backend().get(key);
                if (value) {
                    this->primary().set(key, *value);
                    ++promoted;
                }
            }
        }
        return promoted;
    }
};

} // namespace lru

#endif // LRU_TIERED_STORAGE_HPP
