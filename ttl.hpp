// Unified LRU Cache Library - TTL Cache
// SPDX-License-Identifier: MIT

#ifndef LRU_TTL_HPP
#define LRU_TTL_HPP

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstddef>
#include <optional>
#include <ostream>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "cache_trait.hpp"
#include "core.hpp"
#include "detail/distributed_mutex.hpp"
#include "detail/foundation.hpp"

namespace lru {

// ============================================================================
// TTL Entry
// ============================================================================

/// A cache entry with an optional expiration time.
///
/// The `expired` lazy-invalidaton flag has been removed. It was
/// previously stored as a `std::atomic<bool>` inside the entry and
/// modified via `const_cast` from `peek()` / `contains()` / `get()`,
/// which broke const-correctness. Expiry is now determined purely by
/// comparing `expiry` against `clock::now()` — `steady_clock::now()` is
/// a cheap memory read, so the lazy flag was a marginal optimization
/// that did not justify the const-cast hack. `ttl_entry` is now a pure
/// value type.
template <typename Value>
struct ttl_entry {
    using clock = std::chrono::steady_clock;
    using time_point = clock::time_point;

    Value value;
    std::optional<time_point> expiry;

    ttl_entry() = default;
    ttl_entry(Value v, std::optional<time_point> exp = std::nullopt)
        : value(std::move(v)), expiry(std::move(exp)) {}

    ttl_entry(const ttl_entry&) = default;
    ttl_entry(ttl_entry&&) noexcept = default;
    ttl_entry& operator=(const ttl_entry&) = default;
    ttl_entry& operator=(ttl_entry&&) noexcept = default;

    /// Create a time_point from now + a duration.
    template <typename Rep, typename Period>
    static time_point from_now(std::chrono::duration<Rep, Period> dur) {
        return clock::now() + dur;
    }

    /// Apply ±jitter_pct randomization to a duration to prevent TTL
    /// thundering-herd avalanches: when many keys are inserted with the
    /// same TTL (e.g., bulk prewarm, scheduled refresh), identical expiry
    /// timestamps cause simultaneous expiry → downstream stampede.
    ///
    /// With jitter_pct = 0.10, the returned duration is in
    /// [dur * 0.90, dur * 1.10] uniformly distributed. jitter_pct <= 0
    /// returns dur unchanged.
    ///
    /// Implementation delegates to detail::apply_ttl_jitter (defined in
    /// cache_trait.hpp) — see there for PRNG details. P0-6 (fix.01): the
    /// return type is nanoseconds so sub-second jitter survives even when
    /// the caller passes an integer-seconds duration.
    template <typename Rep, typename Period>
    static std::chrono::nanoseconds
    apply_jitter(std::chrono::duration<Rep, Period> dur, double jitter_pct) {
        return detail::apply_ttl_jitter(dur, jitter_pct);
    }

    /// Create a time_point from now + a duration with ±jitter_pct applied.
    template <typename Rep, typename Period>
    static time_point from_now_with_jitter(std::chrono::duration<Rep, Period> dur,
                                           double jitter_pct) {
        return clock::now() + apply_jitter(dur, jitter_pct);
    }

    /// Pure read-only check — is this entry expired at `now`?
    /// Does not modify any state (no more lazy `expired` flag).
    bool is_expired_at(time_point now) const noexcept {
        return expiry.has_value() && now >= *expiry;
    }
};

// ============================================================================
// TTL Cache
// ============================================================================

/// A cache with time-to-live (TTL) support built on top of a unified_cache.
/// Entries expire and are automatically removed after a configurable duration.
///
/// @tparam Key      The key type.
/// @tparam Value    The value type.
/// @tparam Duration The duration type for TTL (default std::chrono::seconds).
/// @tparam Cache    The underlying cache type (default: single-threaded LRU).
///                  May be a thread-safe type such as safe_cache / striped_cache.
///                  The TTL layer stripes its locks by key hash with
///                  striped_mutex<distributed_shared_mutex>, so different keys proceed
/// @tparam Hash     The hash function type for mapping keys to mutex stripes.
///
/// Lock Hierarchy (MUST be followed to prevent deadlock):
///
///   Level 1: ttl_cache::mutex_           (striped_mutex<distributed_shared_mutex>, per-key or global)
///   Level 2: unified_cache::mutex_       (distributed_shared_mutex, shared/exclusive)
///   Level 3: mm_lru::update_mutex_       (detail::shared_spinlock; shared on
///                                         the promotion path, exclusive on the
///                                         eviction path, try_lock only)
///
///   All code paths MUST acquire locks in this order (1→2→3).
///   Never acquire a lower-level lock while holding a higher-level one.
///   Never acquire locks out of order.
///
///   Example valid path: set() → per-key write lock → cache_.set() → shared_mutex(write) → mm_.set() → try_lock(update_mutex_)
///   Example valid path: get() → per-key read lock → cache_.peek() → shared_mutex(read), then per-key write lock → cache_.del() → shared_mutex(write)
///   Example valid path: clear_expired() → global write lock → cache_.acquire_read_lock() → shared_mutex(read)
///   Example valid path: operator<< → global read lock → cache_.acquire_read_lock() → shared_mutex(read)
///
/// Example (thread-safe):
///   using ttl_entry_type = ttl_entry<std::string>;
///   using safe_ttl_t = ttl_cache<int, std::string, std::chrono::seconds,
///       unified_cache<lru_trait<thread_safe_policy>, int, ttl_entry_type>>;
///   safe_ttl_t cache(5s, 1000);
template <typename Key, typename Value, typename Duration = std::chrono::seconds,
          typename Cache = unified_cache<lru_trait<single_threaded_policy>, Key, ttl_entry<Value>>,
          typename Hash = std::hash<Key>>
class ttl_cache {
public:
    using key_type = Key;
    using value_type = Value;
    using duration_type = Duration;
    using clock = std::chrono::steady_clock;
    using time_point = clock::time_point;
    using size_type = std::size_t;

    using entry_type = ttl_entry<Value>;
    using entry_cache_type = Cache;
    using hash_type = Hash;

    /// Whether the underlying cache is thread-safe. When true, ttl_cache skips
    static constexpr bool is_thread_safe = entry_cache_type::is_thread_safe;

    // --------------------------------------------------------------------
    // Constructors
    // --------------------------------------------------------------------

    ttl_cache() : ttl_cache(duration_type::zero()) {}

    /// Construct with default TTL and optional max size.
    /// @tparam Rep,Period Any duration type (e.g., milliseconds, seconds).
    template <typename Rep, typename Period>
    explicit ttl_cache(std::chrono::duration<Rep, Period> default_ttl, size_type max_size = unlimited)
        : default_ttl_(std::chrono::duration_cast<std::chrono::nanoseconds>(default_ttl)),
          max_size_(max_size) {
        if (max_size != unlimited) {
            cache_.max_size(max_size);
        }
    }

    // --------------------------------------------------------------------
    // Core API
    // --------------------------------------------------------------------

    // The TTL layer uses striped_mutex<distributed_shared_mutex>: per-key
    // stripes, so operations on different keys run concurrently. Within a
    // stripe reads share the lock and writes take it exclusively, which is
    // what makes the peek-then-promote recheck in get() safe. It is
    // distributed_shared_mutex rather than std::shared_mutex because MinGW's
    // pthread_rwlock_t returns EINVAL under mixed high-contention load.

    /// Exclusive write lock for the stripe owning `key`.
    auto acquire_ttl_write_lock(const Key& key) const {
        auto hash = Hash{}(key);
        auto stripe = mutex_.stripe_for(hash);
        return mutex_.make_unique_lock(stripe);
    }

    /// Shared read lock for the stripe owning `key`.
    auto acquire_ttl_read_lock(const Key& key) const {
        auto hash = Hash{}(key);
        auto stripe = mutex_.stripe_for(hash);
        return mutex_.make_shared_lock(stripe);
    }

    /// Global write lock, for whole-cache operations such as clear_expired / flush.
    auto acquire_ttl_global_write_lock() const {
        return detail::striped_mutex_write_all_guard(mutex_);
    }

    /// Global read lock, for whole-cache queries such as size / empty.
    auto acquire_ttl_global_read_lock() const {
        return detail::striped_mutex_read_all_guard(mutex_);
    }

    /// Gracefully stop the cache. After this call:
    /// - All get/set/del/peek/contains operations become no-ops
    /// - flush() is called to clear all data
    /// - The cache cannot be restarted
    ///
    /// The dedicated `ttl_reaper` class and its `register_reaper_stop`
    /// registration mechanism have been removed. Callers that need
    /// background TTL cleanup should use a `detail::periodic_worker` (or
    /// `unified_cache::start_ttl_cleaner()` when using a `unified_cache`
    /// directly) and ensure the worker is joined before the cache is
    /// destroyed. `clear_expired()` itself no longer holds any global
    /// TTL lock — it collects expired keys under the MM read lock and
    /// deletes each key under its own per-stripe write lock.
    void stop() {
        auto lock = acquire_ttl_global_write_lock();
        if (stopped_.load(std::memory_order_acquire)) return;
        stopped_.store(true, std::memory_order_release);
        cache_.flush();
    }

    /// Check if the cache has been stopped.
    bool is_stopped() const noexcept {
        return stopped_.load(std::memory_order_acquire);
    }

    // --------------------------------------------------------------------
    // TTL jitter
    // --------------------------------------------------------------------
    //
    // Applied once, where the expiry time_point is computed, so the value
    // layer and the item-level TTL index always receive the SAME deadline
    // (see insert_locked). Without jitter, keys inserted with a shared TTL
    // (bulk prewarm, scheduled refresh) expire together and stampede the
    // origin. unified_cache::set_with_ttl has always jittered; this class did
    // not, so the two TTL entry points behaved differently.

    /// Enable/disable +/- jitter on every TTL this cache assigns.
    void set_ttl_jitter_enabled(bool enabled) noexcept {
        jitter_enabled_.store(enabled, std::memory_order_relaxed);
    }

    bool ttl_jitter_enabled() const noexcept {
        return jitter_enabled_.load(std::memory_order_relaxed);
    }

    /// Set the jitter fraction. With 0.10 the effective TTL is uniform in
    /// [ttl * 0.9, ttl * 1.1]. Must be non-negative.
    void set_ttl_jitter_pct(double pct) {
        if (pct < 0.0) {
            throw std::invalid_argument("ttl_cache: jitter pct must be non-negative");
        }
        jitter_pct_ = pct;
    }

    double ttl_jitter_pct() const noexcept { return jitter_pct_; }

    // --------------------------------------------------------------------
    // Max TTL and background cleaner
    // --------------------------------------------------------------------

    /// Cap every TTL this cache assigns; 0 disables the cap.
    ///
    /// A longer request is clamped rather than rejected, and counted in
    /// ttl_clamped_count(), so a stray `set_with_ttl(k, v, 100y)` cannot pin an
    /// entry effectively forever with no trace.
    void set_max_ttl(std::chrono::nanoseconds max_ttl) noexcept {
        max_ttl_ns_.store(
            max_ttl.count() > 0 ? static_cast<std::uint64_t>(max_ttl.count()) : 0,
            std::memory_order_relaxed);
    }

    /// The configured TTL cap (0 = none).
    std::chrono::nanoseconds max_ttl() const noexcept {
        return std::chrono::nanoseconds(max_ttl_ns_.load(std::memory_order_relaxed));
    }

    /// Number of TTLs that set_max_ttl() clamped.
    std::size_t ttl_clamped_count() const noexcept {
        return ttl_clamped_.load(std::memory_order_relaxed);
    }

    /// Start a background cleaner that removes expired entries every
    /// `interval`. Forwarded to the underlying cache's round-robin cleaner,
    /// which locks one shard at a time; a no-op for cache types without one.
    template <typename Rep, typename Period>
    void start_ttl_cleaner(std::chrono::duration<Rep, Period> interval) {
        if constexpr (requires { cache_.start_ttl_cleaner(interval); }) {
            cache_.start_ttl_cleaner(interval);
        }
    }

    /// Stop the background cleaner (no-op if none was started).
    void stop_ttl_cleaner() {
        if constexpr (requires { cache_.stop_ttl_cleaner(); }) {
            cache_.stop_ttl_cleaner();
        }
    }

    /// Maximum number of expired entries the cleaner reaps per lock
    /// acquisition (0 = drain everything under one lock).
    void set_ttl_evict_batch_size(std::size_t batch) {
        if constexpr (requires { cache_.set_ttl_evict_batch_size(std::size_t{1}); }) {
            cache_.set_ttl_evict_batch_size(batch);
        }
    }

    /// Whether the cleaner advances one shard per cycle (round-robin) instead
    /// of sweeping all shards each cycle.
    void set_ttl_cleaner_round_robin(bool round_robin) {
        if constexpr (requires { cache_.set_ttl_cleaner_round_robin(true); }) {
            cache_.set_ttl_cleaner_round_robin(round_robin);
        }
    }

    /// Insert a key-value pair with the default TTL.
    template <typename V>
    void set(const Key& key, V&& value) {
        auto lock = acquire_ttl_write_lock(key);
        if (stopped_.load(std::memory_order_acquire)) return;
        insert_locked(key, std::forward<V>(value),
                      default_ttl_ != std::chrono::nanoseconds::zero()
                          ? std::optional<time_point>(deadline_for(default_ttl_))
                          : std::nullopt);
    }

    /// Insert a key-value pair with a specific TTL.
    /// @tparam Rep,Period Any duration type.
    template <typename V, typename Rep, typename Period>
    void set(const Key& key, V&& value, std::chrono::duration<Rep, Period> ttl) {
        auto lock = acquire_ttl_write_lock(key);
        if (stopped_.load(std::memory_order_acquire)) return;
        insert_locked(key, std::forward<V>(value),
                      ttl > std::chrono::duration<Rep, Period>::zero()
                          ? std::optional<time_point>(deadline_for(ttl))
                          : std::nullopt);
    }

    /// Set with custom TTL (alias for set with explicit ttl).
    template <typename V, typename Rep, typename Period>
    void set_with_ttl(const Key& key, V&& value, std::chrono::duration<Rep, Period> ttl) {
        auto lock = acquire_ttl_write_lock(key);
        if (stopped_.load(std::memory_order_acquire)) return;
        insert_locked(key, std::forward<V>(value),
                      ttl > std::chrono::duration<Rep, Period>::zero()
                          ? std::optional<time_point>(deadline_for(ttl))
                          : std::nullopt);
    }

    /// Set without TTL (never expires).
    template <typename V>
    void set_no_ttl(const Key& key, V&& value) {
        auto lock = acquire_ttl_write_lock(key);
        if (stopped_.load(std::memory_order_acquire)) return;
        insert_locked(key, std::forward<V>(value), std::nullopt);
    }

    /// Set with absolute expiry time.
    template <typename V>
    void set_until(const Key& key, V&& value, time_point expiry) {
        auto lock = acquire_ttl_write_lock(key);
        if (stopped_.load(std::memory_order_acquire)) return;
        insert_locked(key, std::forward<V>(value), std::optional<time_point>(expiry));
    }

    /// Get a value by key. Returns std::nullopt if the key is not found
    /// or if the entry has expired (also removes expired entries).
    ///
    /// Check expiry under a read lock with peek() first, so an expired entry does
    /// not bump hit statistics or promote the LRU; only a live entry goes through
    /// get(). If it has expired, drop the read lock, take the write lock, re-check
    ///
    /// No more const_cast — expiry is checked read-only via
    /// `ttl_entry::is_expired_at()`. The lazy `expired` flag has been
    /// removed from `ttl_entry`, so `peek()` (which returns a const
    /// reference) no longer needs to mutate any state.
    std::optional<Value> get(const Key& key) {
        auto rlock = acquire_ttl_read_lock(key);
        if (stopped_.load(std::memory_order_acquire)) return std::nullopt;

        // Phase 1: read lock + peek to check expiry WITHOUT holding the
        // peek handle's reference across the delete path. The peek handle
        // is scoped so its refcount is released before del() runs —
        // mm_lru::del() refuses to remove an item with an active handle
        // (has_active_handle() == refcount > 0), so a handle still alive
        // at del() time silently leaves the expired entry in place.
        bool expired = false;
        {
            auto peek_result = cache_.peek(key);
            if (!peek_result) return std::nullopt;
            const auto& peek_entry = *peek_result;
            expired = peek_entry.is_expired_at(clock::now());
        }  // peek_result destructor releases the refcount.

        if (expired) {
            // G1 fix: lazy deletion. Release the read lock before acquiring
            // the write lock (read->write lock upgrade would deadlock).
            rlock.unlock();
            auto wlock = acquire_ttl_write_lock(key);
            // Re-check expiry under the write lock: another thread may have
            // called set() to refresh the entry between releasing the read
            // lock and acquiring the write lock; do not delete the new entry.
            // The recheck handle is likewise scoped so its reference is
            // released before del() (see Phase 1 comment).
            bool still_expired = false;
            {
                auto recheck = cache_.peek(key);
                if (recheck && recheck->is_expired_at(clock::now())) {
                    still_expired = true;
                }
            }  // recheck destructor releases the refcount.
            if (still_expired) {
                // Delete may fail (entry already removed by another thread);
                // the result is intentionally ignored.
                (void)cache_.del(key);
            }
            return std::nullopt;
        }

        // Not expired — get under the same TTL read lock (no lock upgrade needed)
        auto result = cache_.get(key);
        if (!result) return std::nullopt;
        return result->value;
    }

    /// Peek at a value without affecting LRU order.
    ///
    /// Now `const` — no longer modifies any state. The previous
    /// implementation used `const_cast` to lazily mark entries as
    /// expired; that flag has been removed and expiry is checked purely
    /// by comparing `expiry` against `clock::now()`.
    std::optional<Value> peek(const Key& key) const {
        auto rlock = acquire_ttl_read_lock(key);
        if (stopped_.load(std::memory_order_acquire)) return std::nullopt;
        auto result = cache_.peek(key);
        if (!result) return std::nullopt;
        const auto& entry = *result;
        if (entry.is_expired_at(clock::now())) {
            return std::nullopt;
        }
        return entry.value;
    }

    /// Now `const` — no longer modifies any state.
    bool contains(const Key& key) const {
        auto rlock = acquire_ttl_read_lock(key);
        if (stopped_.load(std::memory_order_acquire)) return false;
        auto result = cache_.peek(key);
        if (!result) return false;
        const auto& entry = *result;
        return !entry.is_expired_at(clock::now());
    }

    /// Check if a specific key has expired (returns false if key not found).
    bool has_expired(const Key& key) const {
        auto lock = acquire_ttl_read_lock(key);
        if (stopped_.load(std::memory_order_acquire)) return false;
        auto result = cache_.peek(key);
        if (!result) return false;
        const auto& entry = *result;
        return entry.is_expired_at(clock::now());
    }

    /// Remove an entry.
    bool del(const Key& key) {
        auto lock = acquire_ttl_write_lock(key);
        if (stopped_.load(std::memory_order_acquire)) return false;
        return cache_.del(key);
    }

    /// Remove all expired entries. Returns the number removed.
    ///
    /// Prefers the underlying cache's item-level TTL index
    /// (unified_cache::evict_expired_now), which sweeps each shard's ordered
    /// expiry heap in O(log n). The per-key scan fallback visits every entry,
    /// so on a large cache it is orders of magnitude slower — it is kept only
    /// for cache types that have no item-level index.
    size_type clear_expired() {
        if (stopped_.load(std::memory_order_acquire)) return 0;
        if constexpr (requires { cache_.evict_expired_now(); }) {
            return static_cast<size_type>(cache_.evict_expired_now());
        } else {
            return clear_expired_locked();
        }
    }

    /// Get remaining TTL for a key. Returns std::nullopt if not found, no TTL, or expired.
    std::optional<duration_type> remaining_ttl(const Key& key) const {
        auto lock = acquire_ttl_read_lock(key);
        if (stopped_.load(std::memory_order_acquire)) return std::nullopt;
        auto result = cache_.peek(key);
        if (!result) return std::nullopt;
        const auto& entry = *result;
        if (!entry.expiry) return std::nullopt; // no expiry
        auto now = clock::now();
        if (entry.is_expired_at(now)) return std::nullopt; // expired
        return std::chrono::duration_cast<duration_type>(*entry.expiry - now);
    }

    /// Access underlying cache statistics.
    auto stats() const {
        auto lock = acquire_ttl_global_read_lock();
        return cache_.stats_snapshot();
    }

    /// Remove all entries.
    void flush() {
        auto lock = acquire_ttl_global_write_lock();
        if (stopped_.load(std::memory_order_acquire)) return;
        cache_.flush();
    }

    /// Number of entries (including expired ones that haven't been cleaned up).
    size_type size() const {
        auto lock = acquire_ttl_global_read_lock();
        return cache_.size();
    }

    /// Maximum capacity.
    size_type max_size() const {
        auto lock = acquire_ttl_global_read_lock();
        return cache_.max_size();
    }

    /// Resize the cache.
    void max_size(size_type new_max) {
        auto lock = acquire_ttl_global_write_lock();
        if (stopped_.load(std::memory_order_acquire)) return;
        cache_.max_size(new_max);
    }

    /// Check if the cache is empty.
    bool empty() const {
        auto lock = acquire_ttl_global_read_lock();
        return cache_.empty();
    }

    /// Access the underlying cache (for statistics and advanced operations).
    /// WARNING: the caller is responsible for external synchronization when
    /// using the returned reference directly. Prefer the ttl_cache public API.
    entry_cache_type& underlying() { return cache_; }
    const entry_cache_type& underlying() const { return cache_; }

    /// Update the default TTL for future insertions.
    void set_default_ttl(duration_type ttl) {
        auto lock = acquire_ttl_global_write_lock();
        if (stopped_.load(std::memory_order_acquire)) return;
        // fix.01 P1-20: store in nanoseconds like the constructor.
        default_ttl_ = std::chrono::duration_cast<std::chrono::nanoseconds>(ttl);
    }

    /// Read the default TTL.
    ///
    /// Takes the global read lock because `default_ttl_` is a plain
    /// (non-atomic) member that `set_default_ttl()` writes under the
    /// global write lock. Reading it without a lock — as this accessor
    /// used to — is a data race against any concurrent
    /// `set_default_ttl()`. The lock is the same one the writer and
    /// `operator<<` already use, so the accessor stays consistent with
    /// them without introducing atomic semantics into the class.
    duration_type default_ttl() const {
        auto lock = acquire_ttl_global_read_lock();
        // fix.01 P1-20: the member is stored in nanoseconds; report in the
        // cache's Duration type.
        return std::chrono::duration_cast<duration_type>(default_ttl_);
    }

    // --------------------------------------------------------------------
    // Destructor
    // --------------------------------------------------------------------

    ~ttl_cache() {
        // No reaper stop callback — callers are responsible for
        // joining any background TTL cleaner thread before the cache is
        // destroyed (the cache itself no longer owns a reaper).
    }

    // --------------------------------------------------------------------
    // Stream output (clears expired, then prints under lock)
    // --------------------------------------------------------------------

    friend std::ostream& operator<<(std::ostream& os, const ttl_cache& c) {
        auto lock = c.acquire_ttl_global_read_lock();
        auto mm_lock = c.cache_.acquire_read_lock();
        // cache_ is `mutable`, so a const ttl_cache still yields a
        // non-const lvalue here and for_each_entry() can take a shared
        // lock on each shard while it walks it.
        auto& uc = c.cache_;

        os << "ttl_cache @" << &c;
        if (c.default_ttl_ != std::chrono::nanoseconds::zero()) {
            os << "  default_ttl="
               << std::chrono::duration_cast<std::chrono::seconds>(c.default_ttl_).count() << "s";
        } else {
            os << "  default_ttl=none";
        }
        os << "  " << uc.stats_snapshot() << "\n";

        std::size_t idx = 0;
        for_each_entry(uc, [&](const auto& item) {
            const auto& entry = item.value;
            os << "  " << idx++ << ": key=";
            if constexpr (detail::is_formattable_v<Key>) {
                os << std::format("{}", item.key);
            } else {
                os << std::format("<key at {:#x}>", reinterpret_cast<std::uintptr_t>(&item.key));
            }
            os << " value='";
            if constexpr (detail::is_formattable_v<Value>) {
                os << std::format("{}", entry.value);
            } else {
                os << std::format("<val at {:#x}>", reinterpret_cast<std::uintptr_t>(&entry.value));
            }
            os << "'";
            if (entry.expiry) {
                auto rem = std::chrono::duration_cast<duration_type>(*entry.expiry - entry_type::clock::now());
                os << " ttl=" << rem.count() << "s";
            } else {
                os << " ttl=none";
            }
            os << "\n";
        });
        return os;
    }

private:
    /// Internal insert helper (caller must hold mutex_ and must have checked
    /// `stopped_`).
    ///
    /// the expiry is now published to BOTH layers.
    ///
    /// Before this change the TTL was written only into the value-layer
    /// `ttl_entry`, while the item's own `expiry_ns` stayed 0. The cache layer
    /// therefore could not use any MM strategy's O(log n) TTL index and had to
    /// fall back to walking the entire cache under a read lock
    /// (`evict_expired_via_ttl_entry_scan`) — the one remaining O(n) TTL path
    /// in the library. Handing the SAME absolute deadline to
    /// `set_with_absolute_expiry()` (which forwards it to
    /// `mm_::set_with_expiry()`) makes the item-level index authoritative and
    /// lets that fallback be deleted.
    ///
    /// The two layers must never disagree: a divergence would mean either an
    /// entry that the item index still considers live after the value layer
    /// expired it, or the reverse. Both are avoided here because exactly one
    /// `time_point` is computed and converted, and `set_with_absolute_expiry`
    /// applies no jitter of its own (unlike `set_with_ttl`, which would have
    /// randomized only the item-level copy).
    /// Deadline for a TTL: now + ttl, with jitter applied when enabled.
    /// The single place jitter is applied, so both TTL layers agree.
    template <typename Rep, typename Period>
    time_point deadline_for(std::chrono::duration<Rep, Period> ttl) const {
        auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(ttl);
        const auto cap = max_ttl_ns_.load(std::memory_order_relaxed);
        if (cap != 0 && ns.count() > 0 &&
            static_cast<std::uint64_t>(ns.count()) > cap) {
            ttl_clamped_.fetch_add(1, std::memory_order_relaxed);
            ns = std::chrono::nanoseconds(cap);
        }
        if (jitter_enabled_.load(std::memory_order_relaxed) && jitter_pct_ > 0.0) {
            return entry_type::from_now_with_jitter(ns, jitter_pct_);
        }
        return clock::now() + ns;
    }

    template <typename V>
    void insert_locked(const Key& key, V&& value,
                       std::optional<time_point> expiry) {
        if (!expiry) {
            // fix.01 P1-17: clear the item-level TTL explicitly instead of
            // falling back to a plain set(). A plain set() left whatever
            // expiry_ns the previous value carried, so `set_no_ttl(k, v)` (and
            // any set() with no default TTL) produced an entry that the value
            // layer considered immortal while the MM still expired and deleted
            // it at the OLD deadline — `peek`/`contains` reported it present
            // and `get` reported it missing, off the same item. Writing 0 is
            // the documented "no expiration" sentinel and keeps both layers in
            // agreement.
            if constexpr (requires { cache_.set_with_absolute_expiry(key, value, std::uint64_t{0}); }) {
                cache_.set_with_absolute_expiry(
                    key, entry_type(std::forward<V>(value), std::nullopt),
                    std::uint64_t{0});
            } else {
                // Caches without native TTL: the value layer is authoritative.
                cache_.set(key, entry_type(std::forward<V>(value), std::nullopt));
            }
            return;
        }
        // Same storage convention as unified_cache::set_with_ttl():
        // nanoseconds since the steady_clock epoch, 0 = no TTL. Every
        // steady_clock deadline is >= the epoch, so a valid deadline can never
        // collide with the sentinel.
        const auto expiry_ns = static_cast<std::uint64_t>(expiry->time_since_epoch().count());
        if constexpr (requires { cache_.set_with_absolute_expiry(key, value, expiry_ns); }) {
            cache_.set_with_absolute_expiry(
                key, entry_type(std::forward<V>(value), *expiry), expiry_ns);
        } else {
            // Compact / custom Caches without native TTL: the value layer is
            // still authoritative for reads, so behaviour is unchanged.
            cache_.set(key, entry_type(std::forward<V>(value), *expiry));
        }
    }

    /// Visit every live entry of the backing cache, in whatever container
    /// the MM uses to store them.
    ///
    /// `sharded_mm_lru::begin()/end()` expose only shard 0, so iterating
    /// with them silently visits 1/N of the cache. That is how
    /// `clear_expired()` came to leave 63 of 64 shards' worth of expired
    /// entries resident forever — they kept consuming their shard's
    /// `max_size` quota and pushed live entries out through LRU, while
    /// `lru_cache_ttl_expired_total` still reported zero cleanups. Every full-scan
    /// path must go through this helper rather than calling
    /// `mm().begin()` directly.
    ///
    /// Each shard is visited under its own read lock, released before the
    /// next shard is acquired, so shards do not serialize against each
    /// other. The MM's own lock is taken here rather than relying on the
    /// caller's stripe lock, which does not protect a shard.
    template <typename Fn>
    static void for_each_entry(entry_cache_type& c, Fn&& fn) {
        auto& mm = c.mm();
        if constexpr (requires(std::size_t i) {
                          mm.num_shards();
                          mm.acquire_shard_read_lock(i);
                          mm.shard(i);
                      }) {
            for (std::size_t i = 0; i < mm.num_shards(); ++i) {
                auto shard_lock = mm.acquire_shard_read_lock(i);
                auto& shard = mm.shard(i);
                for (auto it = shard.begin(); it != shard.end(); ++it) {
                    fn(*it);
                }
            }
        } else {
            for (auto it = mm.begin(); it != mm.end(); ++it) {
                fn(*it);
            }
        }
    }

    /// Internal clear_expired helper (no global TTL lock held by caller).
    /// Collects expired keys under mm read lock, then deletes each key
    /// under its per-stripe write lock to avoid blocking unrelated stripes.
    ///
    /// No more `expired` lazy flag — just check `is_expired_at()`.
    size_type clear_expired_locked() {
        std::vector<key_type> expired_keys;
        {
            auto mm_lock = cache_.acquire_read_lock();
            const auto now = clock::now();
            for_each_entry(cache_, [&](const auto& item) {
                if (item.value.is_expired_at(now)) {
                    expired_keys.push_back(item.key);
                }
            });
        }
        size_type count = 0;
        for (const auto& key : expired_keys) {
            // Delete key by key, taking each key's stripe write lock.
            auto wlock = acquire_ttl_write_lock(key);
            // fix.01 P1-16: re-validate expiry under the write lock.
            //
            // The scan above released the MM read lock before we got here, so
            // a concurrent set() may have refreshed this key with a new value
            // and a new TTL in the meantime. Deleting it now would silently
            // drop a live entry — the exact hazard ttl_cache::get() already
            // guards against on its lazy-deletion path (see the recheck there).
            // Holding the per-key write lock makes this recheck authoritative,
            // because set() takes the same lock.
            {
                auto recheck = cache_.peek(key);
                if (!recheck || !recheck->is_expired_at(clock::now())) {
                    continue;  // gone, or refreshed by a concurrent set()
                }
            }
            if (cache_.del(key)) {
                ++count;
            }
        }
        return count;
    }

    // cache_ is mutable so const query methods (peek/contains/has_expired/
    // remaining_ttl/stats/size/empty/operator<<) can call into the
    // underlying cache's const APIs (which themselves acquire internal
    // shared/read locks). P2-5: no longer used for lazy expiry mutation.
    mutable entry_cache_type cache_;
    /// fix.01 P1-20: stored in NANOSECONDS, not in `duration_type`.
    ///
    /// The constructor used to `duration_cast` the caller's TTL into the
    /// cache's Duration type (seconds by default), so `ttl_cache<int, V>
    /// c(500ms, n)` stored 0s — and 0 is the "never expires" sentinel, meaning
    /// the cache silently had no TTL at all instead of a 500 ms one. Storing
    /// nanoseconds keeps any sub-second TTL the caller passed. `remaining_ttl`
    /// and `default_ttl()` still report in `duration_type`, which is also the
    /// unit the rest of this class computes expiry in.
    std::chrono::nanoseconds default_ttl_{0};
    /// TTL jitter, applied in deadline_for(). Defaults match unified_cache:
    /// on, +/-10%.  jitter_pct_ is a plain double: set_ttl_jitter_pct is a
    /// configuration entry point, not a hot path.
    std::atomic<bool> jitter_enabled_{true};
    double jitter_pct_ = 0.10;
    /// Hard cap on any assigned TTL, in nanoseconds (0 = no cap).
    std::atomic<std::uint64_t> max_ttl_ns_{0};
    /// mutable: deadline_for() is const (it is called from const read paths)
    /// but the clamp counter is logically-const statistics.
    mutable std::atomic<std::size_t> ttl_clamped_{0};
    size_type max_size_ = unlimited;
    /// Once stopped_, all mutating operations become no-ops and get/peek/contains return early.
    std::atomic<bool> stopped_{false};
    // The TTL layer stripes its locks by key hash with
    // striped_mutex<distributed_shared_mutex>: different keys proceed concurrently,
    // and within a key reads share while a write excludes them. It is
    // distributed_shared_mutex rather than std::shared_mutex because MinGW's
    // pthread_rwlock_t returns EINVAL when many rwlock objects mix shared and
    // exclusive acquisitions under contention; it is built on CAS + WaitOnAddress.
    //
    // Lock hierarchy. MUST be followed in order to prevent deadlock:
    //   Level 1: ttl_cache::mutex_        (striped_mutex<distributed_shared_mutex>, per-key or global)
    //   Level 2: unified_cache::mutex_    (distributed_shared_mutex, shared/exclusive)
    //   Level 3: mm_lru::update_mutex_    (shared_spinlock; shared for
    //                                      promotion, exclusive for eviction)
    //   Every code path must acquire locks in 1 -> 2 -> 3 order, never reverse.
private:
    using ttl_mutex_type = detail::striped_mutex<detail::distributed_shared_mutex>;
    mutable ttl_mutex_type mutex_;
};

// ============================================================================
// TTL Reaper (removed — P1-A)
// ============================================================================
//
// The dedicated `ttl_reaper` class has been removed. It was a thin wrapper
// around `detail::periodic_worker` that called `cache.clear_expired()`, and
// its `register_reaper_stop` callback registration mechanism added
// significant complexity to `ttl_cache` (a `std::function<void()>` member
// plus extra locking in `stop()` / `~ttl_cache()`).
//
// `ttl_cache::clear_expired()` itself holds no global TTL lock — it forwards
// to the underlying cache's item-level expiry index, which sweeps one shard at
// a time — so any external periodic worker is safe to call it concurrently with
// normal cache operations.
//
// Callers that need background TTL cleanup should use one of:
//
//   1. `ttl_cache::start_ttl_cleaner(interval)`, forwarded to the underlying
//      cache's round-robin cleaner (one shard per cycle). This works for any
//      `ttl_cache` whose underlying cache exposes a cleaner — `unified_cache`
//      does. `ttl_cache::clear_expired()` also uses that same index, so the
//      two never disagree.
//
//   2. `detail::periodic_worker` directly, for a cache type with no native
//      cleaner:
//
//        lru::ttl_cache<int, std::string> cache(5s, 1000);
//        lru::detail::periodic_worker reaper(
//            [&]{ cache.clear_expired(); },
//            std::chrono::seconds(1));
//      // `reaper` joins its thread on destruction; ensure it is destroyed
//      // before `cache`.
//
//   3. `unified_cache::start_ttl_cleaner()` when using a `unified_cache`
//      directly.

} // namespace lru

#endif // LRU_TTL_HPP
