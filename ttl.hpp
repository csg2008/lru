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
/// P2-5: The `expired` lazy-invalidaton flag has been removed. It was
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
    /// cache_trait.hpp) — see there for PRNG details.
    template <typename Rep, typename Period>
    static std::chrono::duration<Rep, Period>
    apply_jitter(std::chrono::duration<Rep, Period> dur, double jitter_pct) {
        return detail::apply_ttl_jitter(dur, jitter_pct);
    }

    /// Create a time_point from now + a duration with ±jitter_pct applied.
    template <typename Rep, typename Period>
    static time_point from_now_with_jitter(std::chrono::duration<Rep, Period> dur,
                                           double jitter_pct) {
        return clock::now() + apply_jitter(dur, jitter_pct);
    }

    /// P2-5: Pure read-only check — is this entry expired at `now`?
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
///                  可以传入 safe_cache/striped_cache 等线程安全类型。
///                  TTL 层使用 striped_mutex<distributed_shared_mutex> 实现按 key hash
///                  分段的共享/排他锁，不同 key 的操作可并发执行，读操作可共享并发。
/// @tparam Hash     The hash function type for mapping keys to mutex stripes.
///
/// Lock Hierarchy (MUST be followed to prevent deadlock):
///
///   Level 1: ttl_cache::mutex_           (striped_mutex<distributed_shared_mutex>, per-key or global)
///   Level 2: unified_cache::mutex_       (distributed_shared_mutex, shared/exclusive)
///   Level 3: mm_lru::update_mutex_       (std::mutex, exclusive only, try_lock)
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

    /// 底层缓存是否线程安全——当为 true 时跳过 ttl_cache 自身独立的 mutex。
    static constexpr bool is_thread_safe = entry_cache_type::is_thread_safe;

    // --------------------------------------------------------------------
    // Constructors
    // --------------------------------------------------------------------

    ttl_cache() : ttl_cache(duration_type::zero()) {}

    /// Construct with default TTL and optional max size.
    /// @tparam Rep,Period Any duration type (e.g., milliseconds, seconds).
    template <typename Rep, typename Period>
    explicit ttl_cache(std::chrono::duration<Rep, Period> default_ttl, size_type max_size = unlimited)
        : default_ttl_(std::chrono::duration_cast<duration_type>(default_ttl)), max_size_(max_size) {
        if (max_size != unlimited) {
            cache_.max_size(max_size);
        }
    }

    // --------------------------------------------------------------------
    // Core API
    // --------------------------------------------------------------------

    // TTL 层使用 striped_mutex<std::mutex> 实现按 key hash 分段的排他锁。
    // 不同 key 的操作可并发执行，同一 key 的操作互斥保证原子性。

    /// 获取指定 key 对应 stripe 的独占写锁
    auto acquire_ttl_write_lock(const Key& key) const {
        auto hash = Hash{}(key);
        auto stripe = mutex_.stripe_for(hash);
        return mutex_.make_unique_lock(stripe);
    }

    /// 获取指定 key 对应 stripe 的共享读锁
    auto acquire_ttl_read_lock(const Key& key) const {
        auto hash = Hash{}(key);
        auto stripe = mutex_.stripe_for(hash);
        return mutex_.make_shared_lock(stripe);
    }

    /// 获取全局写锁（用于 clear_expired、flush 等全局操作）
    auto acquire_ttl_global_write_lock() const {
        return detail::striped_mutex_write_all_guard(mutex_);
    }

    /// 获取全局读锁（用于 size、empty 等全局查询）
    auto acquire_ttl_global_read_lock() const {
        return detail::striped_mutex_read_all_guard(mutex_);
    }

    /// Gracefully stop the cache. After this call:
    /// - All get/set/del/peek/contains operations become no-ops
    /// - flush() is called to clear all data
    /// - The cache cannot be restarted
    ///
    /// P1-A: The dedicated `ttl_reaper` class and its `register_reaper_stop`
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

    /// Insert a key-value pair with the default TTL.
    template <typename V>
    void set(const Key& key, V&& value) {
        auto lock = acquire_ttl_write_lock(key);
        if (stopped_.load(std::memory_order_acquire)) return;
        set_locked(key, std::forward<V>(value), default_ttl_);
    }

    /// Insert a key-value pair with a specific TTL.
    /// @tparam Rep,Period Any duration type.
    template <typename V, typename Rep, typename Period>
    void set(const Key& key, V&& value, std::chrono::duration<Rep, Period> ttl) {
        auto lock = acquire_ttl_write_lock(key);
        if (stopped_.load(std::memory_order_acquire)) return;
        entry_type entry(std::forward<V>(value));
        if (ttl > std::chrono::duration<Rep, Period>::zero()) {
            entry.expiry = clock::now() + ttl;
        }
        cache_.set(key, std::move(entry));
    }

    /// Set with custom TTL (alias for set with explicit ttl).
    template <typename V, typename Rep, typename Period>
    void set_with_ttl(const Key& key, V&& value, std::chrono::duration<Rep, Period> ttl) {
        auto lock = acquire_ttl_write_lock(key);
        if (stopped_.load(std::memory_order_acquire)) return;
        set_locked(key, std::forward<V>(value), ttl);
    }

    /// Set without TTL (never expires).
    template <typename V>
    void set_no_ttl(const Key& key, V&& value) {
        auto lock = acquire_ttl_write_lock(key);
        if (stopped_.load(std::memory_order_acquire)) return;
        set_locked(key, std::forward<V>(value), duration_type::zero());
    }

    /// Set with absolute expiry time.
    template <typename V>
    void set_until(const Key& key, V&& value, time_point expiry) {
        auto lock = acquire_ttl_write_lock(key);
        if (stopped_.load(std::memory_order_acquire)) return;
        entry_type entry(std::forward<V>(value));
        entry.expiry = expiry;
        cache_.set(key, std::move(entry));
    }

    /// Get a value by key. Returns std::nullopt if the key is not found
    /// or if the entry has expired (also removes expired entries).
    ///
    /// 先用读锁+peek 检查是否过期，避免对已过期条目触发 hit 统计和 LRU 提升；
    /// 未过期时再用 get 触发提升。若已过期，释放读锁后获取写锁，并在写锁下
    /// 重新检查过期状态后删除条目（避免与并发 set 竞争误删新条目）。
    ///
    /// P2-5: No more const_cast — expiry is checked read-only via
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
        }  // peek_result 析构，释放引用计数

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
            }  // recheck 析构，释放引用计数
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
    /// P2-5: Now `const` — no longer modifies any state. The previous
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

    /// P2-5: Now `const` — no longer modifies any state.
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
    /// Uses per-key write locks for deletion so that unrelated keys remain
    /// accessible during cleanup. No global TTL lock is held during deletion.
    size_type clear_expired() {
        if (stopped_.load(std::memory_order_acquire)) return 0;
        return clear_expired_locked();
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
        default_ttl_ = ttl;
    }

    duration_type default_ttl() const {
        return default_ttl_;
    }

    // --------------------------------------------------------------------
    // Destructor
    // --------------------------------------------------------------------

    ~ttl_cache() {
        // P1-A: No reaper stop callback — callers are responsible for
        // joining any background TTL cleaner thread before the cache is
        // destroyed (the cache itself no longer owns a reaper).
    }

    // --------------------------------------------------------------------
    // Stream output (clears expired, then prints under lock)
    // --------------------------------------------------------------------

    friend std::ostream& operator<<(std::ostream& os, const ttl_cache& c) {
        auto lock = c.acquire_ttl_global_read_lock();
        auto mm_lock = c.cache_.acquire_read_lock();
        const auto& uc = c.cache_;

        os << "ttl_cache @" << &c;
        if (c.default_ttl_ != duration_type::zero()) {
            os << "  default_ttl="
               << std::chrono::duration_cast<std::chrono::seconds>(c.default_ttl_).count() << "s";
        } else {
            os << "  default_ttl=none";
        }
        os << "  " << uc.stats_snapshot() << "\n";

        std::size_t idx = 0;
        for (auto it = uc.mm().begin(); it != uc.mm().end(); ++it, ++idx) {
            auto& entry = it->value;
            os << "  " << idx << ": key=";
            if constexpr (detail::is_formattable_v<Key>) {
                os << std::format("{}", it->key);
            } else {
                os << std::format("<key at {:#x}>", reinterpret_cast<std::uintptr_t>(&it->key));
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
        }
        return os;
    }

private:
    /// Internal set helper (caller must hold mutex_).
    /// Templated on duration to preserve sub-second precision even when
    /// the cache's default Duration is std::chrono::seconds.
    template <typename V, typename Rep, typename Period>
    void set_locked(const Key& key, V&& value, std::chrono::duration<Rep, Period> ttl) {
        entry_type entry(std::forward<V>(value));
        if (ttl > std::chrono::duration<Rep, Period>::zero()) {
            entry.expiry = clock::now() + ttl;
        }
        cache_.set(key, std::move(entry));
    }

    /// Internal clear_expired helper (no global TTL lock held by caller).
    /// Collects expired keys under mm read lock, then deletes each key
    /// under its per-stripe write lock to avoid blocking unrelated stripes.
    ///
    /// P2-5: No more `expired` lazy flag — just check `is_expired_at()`.
    size_type clear_expired_locked() {
        std::vector<key_type> expired_keys;
        {
            auto mm_lock = cache_.acquire_read_lock();
            auto now = clock::now();
            for (auto it = cache_.mm().begin(); it != cache_.mm().end(); ++it) {
                const auto& entry = it->value;
                if (entry.is_expired_at(now)) {
                    expired_keys.push_back(it->key);
                }
            }
        }
        size_type count = 0;
        for (const auto& key : expired_keys) {
            // 逐 key 获取对应 stripe 的写锁删除
            auto wlock = acquire_ttl_write_lock(key);
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
    duration_type default_ttl_;
    size_type max_size_ = unlimited;
    /// Once stopped_, all mutating operations become no-ops and get/peek/contains return early.
    std::atomic<bool> stopped_{false};
    // TTL 层使用 striped_mutex<distributed_shared_mutex> 实现按 key hash 分段的锁，
    // 不同 key 的操作可并发执行，同一 key 的读-读并发、读写互斥保证原子性。
    // 使用 distributed_shared_mutex 代替 std::shared_mutex，因为 MinGW winpthreads
    // 的 pthread_rwlock_t 在多个 rwlock 对象高争用混合共享/排他锁时
    // 会出现 EINVAL 错误。distributed_shared_mutex 基于 CAS + WaitOnAddress/futex
    // 实现，完全绕开了 pthread_rwlock_t 的 bug。
    //
    // 锁获取顺序（Lock Hierarchy, MUST be followed to prevent deadlock）:
    //   Level 1: ttl_cache::mutex_        (striped_mutex<distributed_shared_mutex>, per-key or global)
    //   Level 2: unified_cache::mutex_    (distributed_shared_mutex, shared/exclusive)
    //   Level 3: mm_lru::update_mutex_    (std::mutex, exclusive, try_lock only)
    //   所有代码路径必须遵循 1→2→3 的顺序获取锁，严禁反序。
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
// `ttl_cache::clear_expired()` itself no longer holds any global TTL lock —
// it collects expired keys under the MM read lock and deletes each key
// under its own per-stripe write lock — so any external periodic worker
// is safe to call it concurrently with normal cache operations.
//
// Callers that need background TTL cleanup should use one of:
//
//   1. `detail::periodic_worker` directly with a `ttl_cache`:
//
//        lru::ttl_cache<int, std::string> cache(5s, 1000);
//        lru::detail::periodic_worker reaper(
//            [&]{ cache.clear_expired(); },
//            std::chrono::seconds(1));
//      // `reaper` joins its thread on destruction; ensure it is destroyed
//      // before `cache`.
//
//   2. `unified_cache::start_ttl_cleaner()` when using a `unified_cache`
//      directly (the underlying MM's native TTL support is invoked; this
//      does not work for `ttl_cache` since the TTL lives in the entry
//      value, not the MM).

} // namespace lru

#endif // LRU_TTL_HPP
