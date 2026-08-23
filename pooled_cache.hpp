// Unified LRU Cache Library — Pooled Cache (Multi-Pool Partitioning)
// SPDX-License-Identifier: MIT
// Inspired by Facebook CacheLib's PoolRebalancer / CacheAllocator multi-pool architecture
//
// Each Pool owns an independent MM strategy instance with its own size quota and
// priority. Pools share a global max_memory budget and eviction selects the victim
// pool by weighted priority. This enables:
//   - Multi-tenant isolation: different workloads don't evict each other
//   - Priority tiers: critical data in high-priority pools, ephemeral data in low-priority pools
//   - Dynamic resizing: pools grow/shrink at runtime without restart

#ifndef LRU_POOLED_CACHE_HPP
#define LRU_POOLED_CACHE_HPP

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "ankerl/unordered_dense.h"
#include "core.hpp"
#include "detail/distributed_mutex.hpp"
#include "detail/foundation.hpp"

namespace lru {

// ============================================================================
// Pool Configuration
// ============================================================================

/// Configuration for a single pool within a pooled_cache.
struct pool_config {
    /// Human-readable pool name (must be unique within a pooled_cache).
    std::string name;

    /// Maximum number of items this pool can hold.
    /// pool_auto (default) means no per-pool cap; only global limit enforced.
    static constexpr std::size_t pool_auto = unlimited;
    std::size_t max_size = pool_auto;

    /// Priority weight for eviction victim selection.
    /// Lower-priority pools (higher value) are evicted first.
    /// e.g. priority=100 (low) gets 2× more evictions than priority=200 (high).
    /// Must be >= 1.
    uint32_t priority = 100;

    /// Optional MM-level config (pool-specific delayed promotion, insertion point, etc.).
    /// When provided, overrides the global default config.
    /// Only meaningful when the underlying MM type supports per-instance config.
    struct {
        /// lru_refresh_time override (0 = use global default)
        uint32_t lru_refresh_time_override = 0;
        /// lru_insertion_point_spec override (valid range [0,7])
        uint8_t insertion_point_override = 0;
        /// update_on_read override (disabled = use global default)
        std::optional<bool> update_on_read_override;
        /// update_on_write override (disabled = use global default)
        std::optional<bool> update_on_write_override;
    } mm_overrides;

    /// Validate this config.
    void validate() const {
        if (name.empty()) {
            throw std::invalid_argument("pool_config: name must not be empty");
        }
        if (priority < 1) {
            throw std::invalid_argument("pool_config: priority must be >= 1");
        }
        if (mm_overrides.insertion_point_override > 7) {
            throw std::invalid_argument(
                "pool_config: insertion_point_override must be in [0, 7]");
        }
    }
};

// ============================================================================
// Pool Statistics
// ============================================================================

/// Per-pool statistics snapshot.
struct pool_stats {
    std::string name;
    std::size_t size = 0;
    std::size_t max_size = pool_config::pool_auto;
    std::size_t current_memory = 0;
    uint32_t priority = 100;
    double hit_rate = 0.0;
    std::size_t hits = 0;
    std::size_t misses = 0;
    std::size_t evictions = 0;
    std::size_t insertions = 0;
};

// ============================================================================
// Marginal Hits Strategy (for dynamic pool rebalancing)
// ============================================================================

/// Strategy inspired by CacheLib's MarginalHitsStrategy.
/// Analyzes per-pool hit rates and recommends capacity transfers from
/// low-hit-rate pools to high-hit-rate pools.
struct marginal_hits_strategy {
    /// Minimum hit rate difference to trigger rebalancing.
    double min_hit_rate_diff = 0.10;  // 10% difference

    /// Maximum fraction of a pool's capacity to transfer in one rebalance cycle.
    double max_transfer_fraction = 0.10;  // 10% per cycle

    /// Decide rebalancing: analyze pool stats and return a transfer recommendation.
    struct transfer {
        std::string from_pool;  // pool to shrink
        std::string to_pool;    // pool to grow
        std::size_t amount;     // items to transfer
    };

    std::optional<transfer> decide(const std::vector<pool_stats>& all_pool_stats) const {
        if (all_pool_stats.size() < 2) return std::nullopt;

        // Find best (highest hit rate) and worst (lowest hit rate) pools
        const pool_stats* best = nullptr;
        const pool_stats* worst = nullptr;
        for (const auto& ps : all_pool_stats) {
            if (!best || ps.hit_rate > best->hit_rate) best = &ps;
            if (!worst || ps.hit_rate < worst->hit_rate) worst = &ps;
        }

        if (!best || !worst || best->name == worst->name) return std::nullopt;

        double diff = best->hit_rate - worst->hit_rate;
        if (diff < min_hit_rate_diff) return std::nullopt;

        std::size_t transfer_amount = static_cast<std::size_t>(
            worst->max_size * max_transfer_fraction);
        if (transfer_amount == 0) transfer_amount = 1;

        return transfer{worst->name, best->name, transfer_amount};
    }
};

// ============================================================================
// Pool Entry (internal)
// ============================================================================

/// Internal representation of a single pool.
template <typename CacheType>
struct pool_entry {
    std::string name;
    uint32_t priority;
    std::unique_ptr<CacheType> cache;  // The underlying cache instance (owns items)

    // T6.1: Per-pool mutex. When the underlying cache is thread-safe,
    // this mutex is only acquired in shared mode by readers and in
    // exclusive mode by writers/cross-pool-eviction — concurrent ops
    // on *different* pools proceed without contending on the same
    // cache line. When the underlying cache is single-threaded, this
    // mutex serializes all access to the pool (replacing the global
    // mutex for per-pool ops).
    mutable lru::detail::distributed_shared_mutex pool_mutex;

    bool operator<(const pool_entry& other) const noexcept {
        // Higher priority = smaller sort key → evicted last.
        // Break ties by pool name for determinism.
        if (priority != other.priority) {
            return priority > other.priority; // reversed: higher priority first in sorted order
        }
        return name < other.name;
    }
};

// ============================================================================
// Forward declaration
// ============================================================================

template <typename CacheType, typename Strategy>
class pool_rebalancer;

// ============================================================================
// Pooled Cache
// ============================================================================

/// A multi-pool cache that partitions items across independent MM instances.
///
/// pooled_cache provides its own shared_mutex for thread safety. When the
/// underlying cache type is single-threaded, pooled_cache automatically
/// downgrades all operations to an exclusive lock so that no two threads can
/// touch the same pool concurrently. When the underlying cache is itself
/// thread-safe, pooled_cache uses shared locks for read operations to allow
/// concurrent reads across pools.
///
/// Usage:
///   pooled_cache<lru::cache<int, std::string>> pc(10000); // global max 10k items
///   pc.add_pool({"hot_data", .max_size=4000, .priority=200});
///   pc.add_pool({"warm_data", .max_size=4000, .priority=100});
///   pc.add_pool({"cold_data", .max_size=2000, .priority=50});
///
///   pc.set("hot_data", 1, "important");   // routed to hot_data pool
///   pc.get("warm_data", 2);               // routed to warm_data pool
///   pc.get_any(2);                        // search all pools (slower)
///
/// Eviction order: cold_data (prio 50) → warm_data (prio 100) → hot_data (prio 200)
///
/// @tparam UnifiedCache  A fully-instantiated unified_cache type, e.g. lru::cache<int, std::string>
template <typename UnifiedCache>
class pooled_cache {
public:
    using cache_type = UnifiedCache;
    using key_type = typename cache_type::key_type;
    using mapped_type = typename cache_type::mapped_type;
    using size_type = typename cache_type::size_type;
    using mm_type = typename cache_type::mm_type;
    using config_type = typename mm_type::config_type;
    using stats_type = cache_stats;

    // pooled_cache provides its own mutex for thread safety.
    // When the underlying cache is single-threaded, all operations are
    // serialized with an exclusive lock. When the underlying cache is
    // thread-safe, read operations use shared locks for concurrency.

    static constexpr size_type npos = unlimited;

    // --------------------------------------------------------------------
    // Construction
    // --------------------------------------------------------------------

    pooled_cache() = default;

    /// Construct with a global max_size limit.
    explicit pooled_cache(size_type global_max_size)
        : global_max_size_(global_max_size) {
        global_stats_.max_size.store(global_max_size);
    }

    /// Construct with per-MM config.
    explicit pooled_cache(size_type global_max_size, const config_type& default_config)
        : global_max_size_(global_max_size)
        , default_config_(default_config) {
        global_stats_.max_size.store(global_max_size);
    }

    pooled_cache(size_type global_max_size, size_type global_max_memory)
        : global_max_size_(global_max_size)
        , global_max_memory_(global_max_memory) {
        global_stats_.max_size.store(global_max_size);
        global_stats_.max_memory.store(global_max_memory);
    }

    pooled_cache(size_type global_max_size, size_type global_max_memory,
                 const config_type& default_config)
        : global_max_size_(global_max_size)
        , global_max_memory_(global_max_memory)
        , default_config_(default_config) {
        global_stats_.max_size.store(global_max_size);
        global_stats_.max_memory.store(global_max_memory);
    }

    ~pooled_cache();  // defined after pool_rebalancer

    pooled_cache(const pooled_cache&) = delete;
    pooled_cache& operator=(const pooled_cache&) = delete;
    pooled_cache(pooled_cache&&) noexcept = default;
    pooled_cache& operator=(pooled_cache&&) noexcept = default;

    // --------------------------------------------------------------------
    // Pool Management
    // --------------------------------------------------------------------

    /// Add a new pool. Throws if a pool with the same name already exists.
    /// Returns the pool index.
    std::size_t add_pool(const pool_config& cfg) {
        cfg.validate();
        std::unique_lock lock(mutex_);

        if (pool_index_.contains(cfg.name)) {
            throw std::invalid_argument(
                "pooled_cache::add_pool: pool '" + cfg.name + "' already exists");
        }

        auto idx = pools_.size();

        // Build MM-specific config from pool overrides
        config_type mm_config = default_config_;
        if (cfg.mm_overrides.lru_refresh_time_override > 0) {
            mm_config.default_lru_refresh_time = cfg.mm_overrides.lru_refresh_time_override;
        }
        if (cfg.mm_overrides.insertion_point_override > 0) {
            mm_config.lru_insertion_point_spec = cfg.mm_overrides.insertion_point_override;
        }
        if (cfg.mm_overrides.update_on_read_override.has_value()) {
            mm_config.update_on_read = *cfg.mm_overrides.update_on_read_override;
        }
        if (cfg.mm_overrides.update_on_write_override.has_value()) {
            mm_config.update_on_write = *cfg.mm_overrides.update_on_write_override;
        }

        auto entry = std::make_unique<pool_entry<cache_type>>();
        entry->name = cfg.name;
        entry->priority = cfg.priority;
        // Initialize underlying cache with pool-level limits
        if (cfg.max_size != pool_config::pool_auto) {
            entry->cache = std::make_unique<cache_type>(cfg.max_size, mm_config);
        } else {
            entry->cache = std::make_unique<cache_type>(unlimited, mm_config);
        }

        pool_index_[cfg.name] = idx;
        pools_.push_back(std::move(entry));
        mark_eviction_dirty();
        recompute_totals_unlocked();  // T6.2: refresh atomic counters
        return idx;
    }

    /// Remove a pool and all its data. Items are destroyed (eviction callbacks fire).
    /// @return true if the pool was found and removed.
    bool remove_pool(std::string_view name) {
        std::unique_lock lock(mutex_);
        auto it = pool_index_.find(std::string(name));
        if (it == pool_index_.end()) return false;

        auto idx = it->second;
        auto& entry = pools_[idx];

        // Evict all items in this pool (callbacks fire)
        entry->cache->flush();

        // Remove from index
        pool_index_.erase(it);

        // Swap-and-pop: O(1) removal from vector
        if (idx != pools_.size() - 1) {
            // Move last element to the removed position
            auto last_idx = pools_.size() - 1;
            pools_[idx] = std::move(pools_[last_idx]);
            // Update the moved pool's index entry
            pool_index_[pools_[idx]->name] = idx;
        }
        pools_.pop_back();

        mark_eviction_dirty();
        recompute_totals_unlocked();  // T6.2: refresh atomic counters
        return true;
    }

    /// Resize a pool's max_size at runtime.
    void resize_pool(std::string_view name, size_type new_max) {
        std::unique_lock lock(mutex_);
        auto& entry = get_pool(name);
        entry.cache->max_size(new_max);
        mark_eviction_dirty();
    }

    /// Change a pool's priority at runtime (affects future eviction victim selection).
    void set_pool_priority(std::string_view name, uint32_t new_priority) {
        if (new_priority < 1) {
            throw std::invalid_argument("pool priority must be >= 1");
        }
        std::unique_lock lock(mutex_);
        get_pool(name).priority = new_priority;
        mark_eviction_dirty();
    }

    /// Number of pools.
    std::size_t pool_count() const noexcept {
        auto lock = maybe_unique_lock();
        return pools_.size();
    }

    /// Check if a pool exists.
    bool has_pool(std::string_view name) const noexcept {
        auto lock = maybe_unique_lock();
        return pool_index_.contains(std::string(name));
    }

    /// Get per-pool statistics for all pools.
    std::vector<pool_stats> all_pool_stats() const {
        auto lock = maybe_unique_lock();
        std::vector<pool_stats> result;
        result.reserve(pools_.size());
        for (const auto& entry : pools_) {
            pool_stats ps;
            ps.name = entry->name;
            ps.size = entry->cache->size();
            ps.max_size = entry->cache->max_size();
            ps.current_memory = entry->cache->current_memory();
            ps.priority = entry->priority;
            ps.hit_rate = entry->cache->mm().stats().hit_rate();
            ps.hits = entry->cache->mm().stats().hits.value.load();
            ps.misses = entry->cache->mm().stats().misses.value.load();
            ps.evictions = entry->cache->mm().stats().evictions.value.load();
            ps.insertions = entry->cache->mm().stats().insertions.value.load();
            result.push_back(std::move(ps));
        }
        return result;
    }

    /// Get a pool's max_size (capacity in items).
    size_type get_pool_max_size(std::string_view name) const {
        auto lock = maybe_unique_lock();
        return get_pool(name).cache->max_size();
    }

    // --------------------------------------------------------------------
    // Core API — routed by pool name
    // --------------------------------------------------------------------

    /// Insert or update an item in the specified pool.
    /// T6.1/T6.2: Uses shared_lock on global mutex_ (for pool lookup)
    /// + unique_lock on per-pool mutex (for write serialization).
    /// For thread-safe underlying caches, concurrent sets on DIFFERENT
    /// pools proceed in parallel; concurrent sets on the SAME pool
    /// serialize on the per-pool mutex.
    ///
    /// H-2 fix (TOCTOU race on global capacity): For thread-safe underlying
    /// caches, `maybe_unique_lock()` returns a shared_lock, so two concurrent
    /// set() calls on DIFFERENT pools can both pass `ensure_capacity_locked()`
    /// seeing a stale total_size_ below global_max_size_, then both insert,
    /// pushing total_size_ to global_max_size_ + 1 (off-by-one). The post-set
    /// `ensure_capacity_locked()` recheck closes this window: after totals are
    /// updated atomically, any overshoot is evicted via evict_global() (which
    /// uses per-pool try_lock, safe under the shared global lock). For
    /// single-threaded underlying caches the recheck is a no-op (the global
    /// exclusive lock already serializes everything, so no overshoot is
    /// possible).
    template <typename V>
    void set(std::string_view pool_name, const key_type& key, V&& value) {
        auto lock = maybe_unique_lock();
        ensure_capacity_locked();
        auto& entry = get_pool(pool_name);
        // T6.1: per-pool write lock. For single-threaded underlying
        // caches, maybe_unique_lock() already holds the global mutex
        // exclusively so this is a no-op (recursive lock not needed
        // because we use a single global lock). For thread-safe
        // underlying caches, this is the actual serialization point.
        //
        // H-2 fix (counter drift): before_size/before_memory AND
        // after_size/after_memory must both be read under pool_lock so
        // concurrent ops on the same pool can't interleave and corrupt
        // the delta. Previously before_* was read before the lock and
        // after_* was read in update_totals_for_pool_delta() outside the
        // lock, causing total_size_ to drift upward under contention.
        std::size_t before_size = 0, before_memory = 0;
        std::size_t after_size = 0, after_memory = 0;
        if constexpr (cache_type::is_thread_safe) {
            std::unique_lock pool_lock(entry.pool_mutex);
            before_size = entry.cache->size();
            before_memory = entry.cache->current_memory();
            entry.cache->set(key, std::forward<V>(value));
            after_size = entry.cache->size();
            after_memory = entry.cache->current_memory();
        } else {
            before_size = entry.cache->size();
            before_memory = entry.cache->current_memory();
            entry.cache->set(key, std::forward<V>(value));
            after_size = entry.cache->size();
            after_memory = entry.cache->current_memory();
        }
        update_totals_for_pool_delta(entry, before_size, before_memory,
                                     after_size, after_memory);
        // H-2 fix: re-check global capacity AFTER totals are updated. Closes
        // the TOCTOU window where concurrent set()s on different pools both
        // passed the pre-set ensure_capacity_locked() seeing a stale total.
        ensure_capacity_locked();
        update_global_stats_after_write();
    }

    /// Insert only if not present. Returns true on insertion.
    template <typename V>
    bool add(std::string_view pool_name, const key_type& key, V&& value) {
        auto lock = maybe_unique_lock();
        ensure_capacity_locked();
        auto& entry = get_pool(pool_name);
        // H-2 fix: read before/after snapshots under pool_lock (same as set()).
        std::size_t before_size = 0, before_memory = 0;
        std::size_t after_size = 0, after_memory = 0;
        bool result;
        if constexpr (cache_type::is_thread_safe) {
            std::unique_lock pool_lock(entry.pool_mutex);
            before_size = entry.cache->size();
            before_memory = entry.cache->current_memory();
            result = entry.cache->add(key, std::forward<V>(value));
            after_size = entry.cache->size();
            after_memory = entry.cache->current_memory();
        } else {
            before_size = entry.cache->size();
            before_memory = entry.cache->current_memory();
            result = entry.cache->add(key, std::forward<V>(value));
            after_size = entry.cache->size();
            after_memory = entry.cache->current_memory();
        }
        update_totals_for_pool_delta(entry, before_size, before_memory,
                                     after_size, after_memory);
        // H-2 fix: post-add global capacity recheck (same rationale as set()).
        ensure_capacity_locked();
        update_global_stats_after_write();
        return result;
    }

    /// Get an item from the specified pool.
    /// 若底层 cache 非线程安全，则降级为独占锁，避免多个读线程同时操作
    /// 单线程底层 cache 导致的数据竞争。
    read_handle<mapped_type>
    get(std::string_view pool_name, const key_type& key) {
        auto lock = maybe_unique_lock();
        auto& entry = get_pool(pool_name);
        if constexpr (cache_type::is_thread_safe) {
            // Underlying cache is thread-safe; no per-pool lock needed.
            return entry.cache->get(key);
        } else {
            // Single-threaded cache: maybe_unique_lock() already holds
            // the global mutex exclusively.
            return entry.cache->get(key);
        }
    }

    /// Const get from the specified pool (peek semantics, no promotion).
    read_handle<const mapped_type>
    get(std::string_view pool_name, const key_type& key) const {
        auto lock = maybe_unique_lock();
        const auto& entry = get_pool(pool_name);
        return entry.cache->peek(key);
    }

    /// Peek without promoting.
    read_handle<const mapped_type>
    peek(std::string_view pool_name, const key_type& key) const {
        auto lock = maybe_unique_lock();
        const auto& entry = get_pool(pool_name);
        return entry.cache->peek(key);
    }

    /// Delete an item from the specified pool.
    /// T6.1: Uses shared_lock on global mutex_ + unique_lock on per-pool
    /// mutex. For thread-safe underlying caches, concurrent dels on
    /// DIFFERENT pools proceed in parallel.
    bool del(std::string_view pool_name, const key_type& key) {
        auto lock = maybe_unique_lock();
        auto& entry = get_pool(pool_name);
        // H-2 fix: read before/after snapshots under pool_lock (same as set()).
        std::size_t before_size = 0, before_memory = 0;
        std::size_t after_size = 0, after_memory = 0;
        bool result;
        if constexpr (cache_type::is_thread_safe) {
            std::unique_lock pool_lock(entry.pool_mutex);
            before_size = entry.cache->size();
            before_memory = entry.cache->current_memory();
            result = entry.cache->del(key);
            after_size = entry.cache->size();
            after_memory = entry.cache->current_memory();
        } else {
            before_size = entry.cache->size();
            before_memory = entry.cache->current_memory();
            result = entry.cache->del(key);
            after_size = entry.cache->size();
            after_memory = entry.cache->current_memory();
        }
        update_totals_for_pool_delta(entry, before_size, before_memory,
                                     after_size, after_memory);
        update_global_stats_after_write();
        return result;
    }

    /// Check if a key exists in the specified pool.
    bool contains(std::string_view pool_name, const key_type& key) const {
        auto lock = maybe_unique_lock();
        const auto& entry = get_pool(pool_name);
        return entry.cache->contains(key);
    }

    // --------------------------------------------------------------------
    // Global API — search across all pools
    // --------------------------------------------------------------------

    /// Get an item by searching all pools (in order).
    /// Returns an empty handle if not found in any pool.
    /// T6.4: Uses a cached first_non_empty_pool_ hint to skip the
    /// common case where the first few pools are empty (e.g., cold
    /// tier drained by rebalancer). The hint is invalidated on
    /// pool structure changes.
    read_handle<mapped_type>
    get_any(const key_type& key) {
        auto lock = maybe_unique_lock();
        auto hint = first_non_empty_pool_.load(std::memory_order_acquire);
        if (hint >= 0 && static_cast<std::size_t>(hint) < pools_.size()) {
            auto& entry = pools_[hint];
            if (!entry->cache->empty()) {
                auto result = entry->cache->get(key);
                if (result) return result;
            }
        }
        // Hint miss or pool empty — full scan, update hint.
        for (std::size_t i = 0; i < pools_.size(); ++i) {
            auto& entry = pools_[i];
            if (entry->cache->empty()) continue;
            auto result = entry->cache->get(key);
            if (result) {
                first_non_empty_pool_.store(static_cast<std::ptrdiff_t>(i),
                                            std::memory_order_release);
                return result;
            }
        }
        global_stats_.register_miss();
        return {};
    }

    /// Const get across all pools (peek semantics, no promotion).
    read_handle<const mapped_type>
    get_any(const key_type& key) const {
        auto lock = maybe_unique_lock();
        for (const auto& entry : pools_) {
            auto result = entry->cache->peek(key);
            if (result) return result;
        }
        global_stats_.register_miss();
        return {};
    }

    /// Check if a key exists in any pool.
    bool contains_any(const key_type& key) const {
        auto lock = maybe_unique_lock();
        for (const auto& entry : pools_) {
            if (entry->cache->contains(key)) return true;
        }
        return false;
    }

    /// Delete a key from whichever pool it's in.
    bool del_any(const key_type& key) {
        std::unique_lock lock(mutex_);
        for (auto& entry : pools_) {
            std::size_t before_size = entry->cache->size();
            std::size_t before_memory = entry->cache->current_memory();
            if (entry->cache->del(key)) {
                // P-FIX: del_any previously did not update total_size_ /
                // total_memory_, causing the atomic counters to drift
                // upward whenever items were removed via del_any. Under
                // mixed set/del_any workloads the counter exceeded
                // global_max_size_ even though the real item count was
                // well below the limit. Mirror del()'s accounting.
                // H-2 fix: pass after-snapshot explicitly (del_any holds
                // exclusive global lock so no per-pool race, but the 5-arg
                // signature is required).
                std::size_t after_size = entry->cache->size();
                std::size_t after_memory = entry->cache->current_memory();
                update_totals_for_pool_delta(*entry, before_size, before_memory,
                                             after_size, after_memory);
                update_global_stats_after_write();
                return true;
            }
        }
        return false;
    }

    // --------------------------------------------------------------------
    // Global Capacity
    // --------------------------------------------------------------------

    /// Total items across all pools.
    /// T6.2: O(1) via atomic counter.
    size_type size() const noexcept {
        return total_size_.load(std::memory_order_acquire);
    }

    /// Global max size.
    size_type max_size() const noexcept { return global_max_size_; }

    /// Global max memory.
    size_type max_memory() const noexcept { return global_max_memory_; }

    /// Total memory across all pools.
    /// T6.2: O(1) via atomic counter.
    size_type current_memory() const noexcept {
        return total_memory_.load(std::memory_order_acquire);
    }

    /// Change global max_size.
    void set_global_max_size(size_type new_max) {
        std::unique_lock lock(mutex_);
        global_max_size_ = new_max;
        global_stats_.max_size.store(new_max);
        // T6.2: should_evict_global() now uses atomic counters, so the
        // loop is O(1) per check; evict_global() iterates pools.
        while (should_evict_global()) {
            if (!evict_global()) break;
        }
    }

    /// Change global max_memory.
    void set_global_max_memory(size_type new_max) {
        std::unique_lock lock(mutex_);
        global_max_memory_ = new_max;
        global_stats_.max_memory.store(new_max);
        while (should_evict_global()) {
            if (!evict_global()) break;
        }
    }

    bool empty() const noexcept { return size() == 0; }

    // --------------------------------------------------------------------
    // Eviction — cross-pool victim selection by priority
    // --------------------------------------------------------------------

    /// Evict a single item from the lowest-priority pool that has items.
    /// Returns true if an item was evicted.
    /// 使用懒更新排序：仅在 pool 结构变更时重新排序，避免每次淘汰都 O(P log P)。
    /// T6.3: Uses per-pool try_lock to avoid blocking other ops. If a
    /// pool's mutex is held (another thread is writing to it), we skip
    /// it and try the next pool. This avoids deadlock and reduces
    /// contention on the cross-pool eviction path.
    bool evict_global() {
        rebuild_eviction_order();

        // Snapshot the eviction order under the order mutex so we
        // don't race with a rebuild.
        std::vector<std::size_t> order_snapshot;
        {
            std::lock_guard<std::mutex> lk(eviction_order_mutex_);
            order_snapshot = eviction_order_;
        }

        for (auto idx : order_snapshot) {
            if (idx >= pools_.size()) continue;  // pool was removed
            auto& entry = pools_[idx];
            if (entry->cache->empty()) continue;

            // T6.3: try-lock the per-pool mutex. If another thread is
            // writing to this pool, skip it and try the next one.
            // For single-threaded underlying caches, the caller already
            // holds the global mutex exclusively so this is unnecessary
            // (but harmless — try_lock will succeed since no other thread
            // can be inside a per-pool op).
            std::unique_lock<detail::distributed_shared_mutex> pool_lock(
                entry->pool_mutex, std::try_to_lock);
            if (!pool_lock.owns_lock()) continue;

            std::size_t before_size = entry->cache->size();
            std::size_t before_memory = entry->cache->current_memory();
            if (entry->cache->evict()) {
                // H-2 fix: pass after-snapshot explicitly (pool_lock held).
                std::size_t after_size = entry->cache->size();
                std::size_t after_memory = entry->cache->current_memory();
                update_totals_for_pool_delta(*entry, before_size, before_memory,
                                             after_size, after_memory);
                global_stats_.register_eviction();
                return true;
            }
        }
        return false;
    }

    /// Evict up to `count` items across pools. Returns actual count evicted.
    size_type evict_batch(size_type count) {
        // T6.3: use shared_lock on global mutex_ (not unique) so
        // concurrent reads/writes on individual pools can proceed.
        // Per-pool try_lock inside evict_global() prevents deadlock.
        std::shared_lock<std::shared_mutex> lock(mutex_);
        size_type evicted = 0;
        while (evicted < count && evict_global()) {
            ++evicted;
        }
        return evicted;
    }

    // --------------------------------------------------------------------
    // Flush
    // --------------------------------------------------------------------

    /// Remove all items from all pools.
    void flush() {
        std::unique_lock lock(mutex_);
        for (auto& entry : pools_) {
            std::unique_lock pool_lock(entry->pool_mutex);
            entry->cache->flush();
        }
        recompute_totals_unlocked();
        update_global_stats_after_write();
    }

    /// Flush a specific pool.
    void flush_pool(std::string_view name) {
        std::unique_lock lock(mutex_);
        auto& entry = get_pool(name);
        std::unique_lock pool_lock(entry.pool_mutex);
        entry.cache->flush();
        recompute_totals_unlocked();
        update_global_stats_after_write();
    }

    // --------------------------------------------------------------------
    // Statistics
    // --------------------------------------------------------------------

    /// Global statistics.
    stats_type& global_stats() noexcept { return global_stats_; }
    const stats_type& global_stats() const noexcept { return global_stats_; }

    // --------------------------------------------------------------------
    // Callbacks and Config
    // --------------------------------------------------------------------

    /// Set key size calculator for all pools.
    void set_key_size_calculator(std::function<size_type(const key_type&)> func) {
        std::unique_lock lock(mutex_);
        for (auto& entry : pools_) {
            entry->cache->set_key_size_calculator(func);
        }
        key_size_fn_ = std::move(func);
    }

    /// Set value size calculator for all pools.
    void set_value_size_calculator(std::function<size_type(const mapped_type&)> func) {
        std::unique_lock lock(mutex_);
        for (auto& entry : pools_) {
            entry->cache->set_value_size_calculator(func);
        }
        value_size_fn_ = std::move(func);
    }

    /// Set eviction predicate for all pools.
    void set_eviction_predicate(std::function<bool(const key_type&, const mapped_type&)> pred) {
        std::unique_lock lock(mutex_);
        for (auto& entry : pools_) {
            entry->cache->set_eviction_predicate(pred);
        }
    }

    /// Get direct access to a pool's MM instance (for advanced usage).
    mm_type* pool_mm(std::string_view name) {
        auto lock = maybe_unique_lock();
        auto it = pool_index_.find(std::string(name));
        if (it == pool_index_.end()) return nullptr;
        return &pools_[it->second]->cache->mm();
    }

    const mm_type* pool_mm(std::string_view name) const {
        auto lock = maybe_unique_lock();
        auto it = pool_index_.find(std::string(name));
        if (it == pool_index_.end()) return nullptr;
        return &pools_[it->second]->cache->mm();
    }

    // --------------------------------------------------------------------
    // Dynamic Rebalancing
    // --------------------------------------------------------------------

    /// Start a background rebalancer with default marginal_hits_strategy.
    /// The rebalancer periodically analyzes pool hit rates and transfers
    /// capacity from low-hit-rate pools to high-hit-rate pools.
    void start_rebalancer(std::chrono::milliseconds interval) {
        if (!rebalancer_) {
            rebalancer_ = std::make_unique<pool_rebalancer<pooled_cache, marginal_hits_strategy>>(
                *this, marginal_hits_strategy{});
        }
        rebalancer_->start(interval);
    }

    /// Stop the background rebalancer if running.
    void stop_rebalancer() {
        if (rebalancer_) rebalancer_->stop();
    }

private:
    // --------------------------------------------------------------------
    // Internal helpers
    // --------------------------------------------------------------------

    pool_entry<cache_type>& get_pool(std::string_view name) {
        auto it = pool_index_.find(std::string(name));
        if (it == pool_index_.end()) {
            throw std::out_of_range(
                std::string("pooled_cache: pool '") + std::string(name) + "' not found");
        }
        return *pools_[it->second];
    }

    const pool_entry<cache_type>& get_pool(std::string_view name) const {
        auto it = pool_index_.find(std::string(name));
        if (it == pool_index_.end()) {
            throw std::out_of_range(
                std::string("pooled_cache: pool '") + std::string(name) + "' not found");
        }
        return *pools_[it->second];
    }

    // These helpers assume the caller already holds mutex_.
    size_type size_unlocked() const {
        size_type total = 0;
        for (const auto& entry : pools_) {
            total += entry->cache->size();
        }
        return total;
    }

    size_type current_memory_unlocked() const {
        size_type total = 0;
        for (const auto& entry : pools_) {
            total += entry->cache->current_memory();
        }
        return total;
    }

    /// Choose shared vs exclusive lock based on underlying cache thread safety.
    /// When the underlying cache is single-threaded, pooled_cache must serialize
    /// all access itself; otherwise it can allow concurrent readers AND writers
    /// (the underlying cache handles its own concurrency).
    using external_lock_type = std::conditional_t<
        cache_type::is_thread_safe,
        std::shared_lock<std::shared_mutex>,
        std::unique_lock<std::shared_mutex>
    >;

    external_lock_type maybe_unique_lock() const {
        if constexpr (cache_type::is_thread_safe) {
            return std::shared_lock<std::shared_mutex>(mutex_);
        } else {
            return std::unique_lock<std::shared_mutex>(mutex_);
        }
    }

    /// T6.2: O(1) capacity check using atomic counters. Returns true if
    /// the global capacity is exceeded and eviction is needed.
    /// Safe to call without holding any lock.
    bool should_evict_global() const {
        if (global_max_size_ != unlimited &&
            total_size_.load(std::memory_order_acquire) > global_max_size_) {
            return true;
        }
        if (global_max_memory_ != unlimited &&
            total_memory_.load(std::memory_order_acquire) > global_max_memory_) {
            return true;
        }
        return false;
    }

    /// Public ensure_capacity(): acquires the global lock and evicts
    /// if the global capacity is exceeded. Safe to call from outside
    /// any pooled_cache operation.
    void ensure_capacity() {
        // T6.2: fast atomic check first; only iterate pools if we need to evict.
        if (!should_evict_global()) return;
        auto lock = maybe_unique_lock();
        ensure_capacity_locked();
    }

    /// Internal helper: assumes the caller already holds mutex_ (shared
    /// or exclusive). Does NOT acquire any lock — prevents recursive
    /// locking when called from set()/add()/del() which already hold
    /// the global lock.
    ///
    /// H-2 fix: evict_global() uses try_lock on per-pool mutexes to avoid
    /// deadlock with exclusive global lock holders (e.g. del_any). Under
    /// high concurrency all pool locks may be briefly held by set() callers,
    /// causing evict_global() to return false even though the global capacity
    /// is exceeded. We retry with a brief yield to let pool lock holders
    /// release, bounding the overshoot to at most a few items.
    void ensure_capacity_locked() {
        int retries = 0;
        while (should_evict_global()) {
            if (!evict_global()) {
                if (++retries > 64) break; // safety: avoid infinite loop
                std::this_thread::yield();
                continue;
            }
            retries = 0; // reset on successful eviction
        }
    }

    /// 重建淘汰顺序索引（按淘汰优先级升序排列）。
    /// 仅在脏标志被设置时重建，利用 eviction_gen_ 避免重复计算。
    /// T6.3: protected by eviction_order_mutex_ so concurrent
    /// evict_global() calls don't race on eviction_order_.
    /// Caller must hold either shared or exclusive lock on mutex_.
    void rebuild_eviction_order() {
        if (!eviction_dirty_.load(std::memory_order_acquire)) return;
        std::lock_guard<std::mutex> lk(eviction_order_mutex_);
        // Double-check after acquiring the mutex.
        if (!eviction_dirty_.load(std::memory_order_relaxed)) return;
        eviction_dirty_.store(false, std::memory_order_release);

        eviction_order_.resize(pools_.size());
        std::iota(eviction_order_.begin(), eviction_order_.end(), size_t(0));
        std::sort(eviction_order_.begin(), eviction_order_.end(), [this](size_t a, size_t b) {
            // 优先级低的优先淘汰
            if (pools_[a]->priority != pools_[b]->priority) {
                return pools_[a]->priority < pools_[b]->priority;
            }
            // 平局：淘汰更大 pool（比例公平）
            return pools_[a]->cache->size() > pools_[b]->cache->size();
        });
    }

    void mark_eviction_dirty() {
        eviction_dirty_.store(true, std::memory_order_release);
        // Invalidate the first_non_empty_pool cache too — pool
        // structure changed.
        first_non_empty_pool_.store(-1, std::memory_order_release);
    }

    /// T6.2: Update atomic total_size_ / total_memory_ counters based
    /// on the size/memory delta of a single pool's operation.
    /// Called by set/add/del after the underlying cache op completes.
    ///
    /// H-2 fix (counter drift): `before_size`/`before_memory` and
    /// `after_size`/`after_memory` MUST both be read under the per-pool
    /// lock. Previously, `before_*` was read before acquiring the lock and
    /// `after_*` was read here (outside the lock), so concurrent ops on the
    /// same pool could interleave and produce a wrong delta, causing
    /// total_size_ to drift upward until `pc.size()` far exceeded
    /// `global_max_size_`. Callers now read both snapshots under the
    /// per-pool lock and pass all four values here.
    void update_totals_for_pool_delta(const pool_entry<cache_type>& /*pool*/,
                                      std::size_t before_size,
                                      std::size_t before_memory,
                                      std::size_t after_size,
                                      std::size_t after_memory) {
        if (after_size > before_size) {
            total_size_.fetch_add(after_size - before_size, std::memory_order_release);
        } else if (after_size < before_size) {
            total_size_.fetch_sub(before_size - after_size, std::memory_order_release);
        }
        if (after_memory > before_memory) {
            total_memory_.fetch_add(after_memory - before_memory, std::memory_order_release);
        } else if (after_memory < before_memory) {
            total_memory_.fetch_sub(before_memory - after_memory, std::memory_order_release);
        }
    }

    /// T6.2: Recompute total_size_ / total_memory_ from scratch by
    /// iterating all pools. Used after structural changes (add/remove
    /// pool) and on first access. Caller must hold shared or exclusive
    /// lock on mutex_.
    void recompute_totals_unlocked() {
        std::size_t total_s = 0;
        std::size_t total_m = 0;
        for (const auto& entry : pools_) {
            total_s += entry->cache->size();
            total_m += entry->cache->current_memory();
        }
        total_size_.store(total_s, std::memory_order_release);
        total_memory_.store(total_m, std::memory_order_release);
    }

    void update_global_stats_after_write() {
        // T6.2: use atomic counters instead of iterating pools.
        global_stats_.current_size.store(
            total_size_.load(std::memory_order_acquire), std::memory_order_release);
        global_stats_.current_memory.store(
            total_memory_.load(std::memory_order_acquire), std::memory_order_release);
    }

    // --- Members ---

    mutable std::shared_mutex mutex_;

    std::vector<std::unique_ptr<pool_entry<cache_type>>> pools_;
    ankerl::unordered_dense::map<std::string, std::size_t> pool_index_;

    size_type global_max_size_ = unlimited;
    size_type global_max_memory_ = unlimited;
    config_type default_config_;
    mutable stats_type global_stats_;

    /// 淘汰顺序索引缓存（懒更新，由 eviction_dirty_ 控制刷新）。
    /// T6.3: protected by eviction_order_mutex_ so that concurrent
    /// evict_global() calls from multiple set() paths can rebuild
    /// safely without holding the global mutex_ exclusively.
    std::vector<std::size_t> eviction_order_;
    std::atomic<bool> eviction_dirty_{false};
    mutable std::mutex eviction_order_mutex_;

    /// T6.4: Cached index of the first non-empty pool (for get_any).
    /// Updated lazily; -1 means "needs rescan". This avoids an O(P)
    /// scan on every get_any() call when most pools are empty.
    std::atomic<std::ptrdiff_t> first_non_empty_pool_{-1};

    /// T6.2: Atomic global capacity counters. Maintained as a delta
    /// from per-op before/after size reads. Used by should_evict_global()
    /// for an O(1) capacity check without iterating pools.
    alignas(64) std::atomic<std::size_t> total_size_{0};
    alignas(64) std::atomic<std::size_t> total_memory_{0};

    std::function<size_type(const key_type&)> key_size_fn_;
    std::function<size_type(const mapped_type&)> value_size_fn_;

    /// Background pool rebalancer (lazy-initialized by start_rebalancer).
    std::unique_ptr<pool_rebalancer<pooled_cache, marginal_hits_strategy>> rebalancer_;
};

// ============================================================================
// Pool Rebalancer — out-of-line destructor (requires complete type)
// ============================================================================

template <typename UnifiedCache>
pooled_cache<UnifiedCache>::~pooled_cache() = default;

// ============================================================================
// Pool Rebalancer
// ============================================================================

/// Background rebalancer that dynamically adjusts pool sizes based on runtime
/// hit rates. Inspired by CacheLib's PoolRebalancer + MarginalHitsStrategy.
///
/// The rebalancer periodically collects per-pool hit rate statistics, applies
/// the configured Strategy to decide whether capacity should be moved between
/// pools, and executes the transfer by resizing the affected pools.
///
/// @tparam CacheType   The pooled_cache instantiation to rebalance.
/// @tparam Strategy    Decision strategy with a `decide(pool_stats) -> optional<transfer>` method.
template <typename CacheType, typename Strategy = marginal_hits_strategy>
class pool_rebalancer {
public:
    using cache_type = CacheType;
    using strategy_type = Strategy;

    pool_rebalancer(CacheType& cache, Strategy strategy = Strategy{})
        : cache_(cache), strategy_(std::move(strategy)) {}

    ~pool_rebalancer() { stop(); }

    pool_rebalancer(const pool_rebalancer&) = delete;
    pool_rebalancer& operator=(const pool_rebalancer&) = delete;

    /// Start the background rebalancer thread at the given interval.
    void start(std::chrono::milliseconds interval) {
        if (worker_) return;
        worker_ = std::make_unique<detail::periodic_worker>(
            [this] { rebalance(); }, interval);
    }

    /// Stop the background rebalancer thread.
    void stop() {
        if (worker_) {
            worker_->stop();
            worker_.reset();
        }
    }

    /// Perform a single rebalance cycle (can also be called manually).
    void rebalance() {
        auto stats = cache_.all_pool_stats();
        auto decision = strategy_.decide(stats);
        if (decision) {
            // Shrink the low-hit pool
            auto worst_size = cache_.get_pool_max_size(decision->from_pool);
            if (worst_size > decision->amount) {
                worst_size -= decision->amount;
            } else {
                return; // can't shrink further
            }
            cache_.resize_pool(decision->from_pool, worst_size);
            // Grow the high-hit pool
            auto best_size = cache_.get_pool_max_size(decision->to_pool);
            best_size += decision->amount;
            cache_.resize_pool(decision->to_pool, best_size);
        }
    }

private:
    CacheType& cache_;
    Strategy strategy_;
    std::unique_ptr<detail::periodic_worker> worker_;
};

} // namespace lru

#endif // LRU_POOLED_CACHE_HPP
