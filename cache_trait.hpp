// Unified LRU Cache Library - Cache Trait Architecture
// SPDX-License-Identifier: MIT
// Inspired by Facebook CacheLib's CacheTraits design

#ifndef LRU_CACHE_TRAIT_HPP
#define LRU_CACHE_TRAIT_HPP

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "ankerl/unordered_dense.h"
#include "core.hpp"
#include "detail/distributed_mutex.hpp"
#include "detail/foundation.hpp"
#include "detail/atomic_shared_ptr.hpp"
#include "detail/singleflight.hpp"
#include "event_tracker.hpp"
#include "memory.hpp"
#include "mm.hpp"

namespace lru {

// T15: Forward declaration so cache_trait can reference compressed_intrusive_hook
// in its uses_compressed_hook constexpr without requiring compressed_ptr.hpp
// to be included. Complete type is only needed when the trait is actually
// instantiated with Hook = compressed_intrusive_hook.
struct compressed_intrusive_hook;

// T-G1: Forward declaration of ttl_entry (defined in ttl.hpp, which includes
// cache_trait.hpp — so we cannot include ttl.hpp here). The complete type is
// only needed when value_type is ttl_entry<V> AND evict_expired_impl() is
// instantiated, which happens in user code after both headers are included.
template <typename Value>
struct ttl_entry;

// T-G1: Type trait detecting whether V is a ttl_entry<...>. Used by
// evict_expired_impl() to decide whether to fall back to value-layer TTL
// scanning when the MM does not support native TTL (e.g. mm_lru without
// ttl_heap_, mm_2q, mm_fifo). The partial specialization below matches
// ttl_entry<U>; the primary template resolves to false_type for all other
// types.
template <typename V>
struct is_ttl_entry : std::false_type {};

template <typename U>
struct is_ttl_entry<ttl_entry<U>> : std::true_type {};

template <typename V>
inline constexpr bool is_ttl_entry_v = is_ttl_entry<V>::value;

// T14: Forward declaration of compact_cache so unified_cache's
// compact_storage_helper<true> specialization can reference it. The
// complete type is only needed when is_compact=true and an actual
// compact_cache member is instantiated (which happens only in user code
// after both cache_trait.hpp and compact_cache.hpp have been included).
// Default template arguments are intentionally omitted here — they are
// only required on the actual definition in compact_cache.hpp.
template <typename Key, typename Value,
          typename Hash, typename KeyEqual,
          std::size_t kMaxItemSize, std::size_t kSlotAlignment,
          typename ThreadPolicy>
class compact_cache;

// ============================================================================
// No-op Lock Guard
// ============================================================================

/// A trivially destructible no-op lock guard for single-threaded caches.
/// Returned by acquire_write_lock() / acquire_read_lock() when thread safety
/// is disabled, so that callers can uniformly write `auto lock = acquire_...()`
/// without worrying about void return types.
struct noop_lock {
    noop_lock(...) noexcept {}
    bool owns_lock() const noexcept { return true; }
};

// ============================================================================
// Locked range helper
// ============================================================================

/// A lightweight RAII wrapper that keeps a cache read lock alive while the
/// caller iterates over an MM iterator range.
///
/// Returned by unified_cache::rbegin() / unified_cache::rend(). The lock is
/// released when the locked_range object is destroyed, so the caller must keep
/// the range object alive for the duration of iteration.
template <typename Iterator, typename Sentinel, typename Lock>
class locked_range {
public:
    locked_range(Iterator begin, Sentinel end, Lock lock)
        : begin_(std::move(begin))
        , end_(std::move(end))
        , lock_(std::move(lock)) {}

    Iterator begin() const { return begin_; }
    Sentinel end() const { return end_; }

private:
    Iterator begin_;
    Sentinel end_;
    Lock lock_;
};

// ============================================================================
// Lock Policies
// ============================================================================

/// Single-threaded lock policy (no locking).
/// All operations are direct calls with no synchronization.
struct single_threaded_policy {
    using mutex_type = void;
    using striped_mutex_type = void;
    using read_lock_type = void;
    using write_lock_type = void;

    static constexpr bool is_thread_safe = false;
    static constexpr bool is_striped = false;
};

/// Thread-safe lock policy using distributed_shared_mutex.
/// 读操作获取共享锁（允许多个读者并发），写操作获取独占锁。
///
/// Uses detail::distributed_shared_mutex instead of std::shared_mutex because
/// MinGW's winpthreads pthread_rwlock_t has a concurrency bug where
/// pthread_rwlock_rdlock()/pthread_rwlock_wrlock() returns EINVAL under
/// high contention with mixed shared/exclusive locks.
/// distributed_shared_mutex implements shared_mutex semantics via atomic CAS
/// + WaitOnAddress/futex, bypassing the buggy pthread_rwlock_t entirely.
struct thread_safe_policy {
    using mutex_type = detail::distributed_shared_mutex;
    using striped_mutex_type = void;
    using read_lock_type = std::shared_lock<mutex_type>;
    using write_lock_type = std::unique_lock<mutex_type>;

    static constexpr bool is_thread_safe = true;
    static constexpr bool is_striped = false;
};

/// Thread-safe lock policy with striped locking for per-key concurrency.
///
/// When used with sharded_mm_lru, each shard is protected by its own
/// stripe mutex, allowing concurrent access to different shards.
/// Global operations (flush, size, stats) acquire all stripes.
///
/// Uses detail::distributed_shared_mutex for stripe locks, providing
/// shared (read) / exclusive (write) semantics without relying on
/// MinGW's buggy pthread_rwlock_t. Each stripe allows concurrent
/// readers while serializing writers, maximizing read throughput.
template <std::size_t NumStripes = 64>
struct striped_thread_safe_policy {
    using mutex_type = detail::distributed_shared_mutex;
    // T-P3-5: Use lazy_striped_mutex instead of striped_mutex. This defers
    // allocation of the per-stripe distributed_shared_mutex objects until the
    // first global operation (clear, flush, snapshot, etc.) that actually
    // needs them. For sharded_mm_lru caches (which have per-shard locks),
    // per-key operations never touch striped_mutex_, so the allocation may
    // never happen — saving ~64 distributed_shared_mutex objects worth of
    // memory (each contains atomics, wait primitives, and latency histograms).
    using striped_mutex_type = detail::lazy_striped_mutex<detail::distributed_shared_mutex>;
    using read_lock_type = std::shared_lock<mutex_type>;
    using write_lock_type = std::unique_lock<mutex_type>;

    static constexpr bool is_thread_safe = true;
    static constexpr bool is_striped = true;
    static constexpr std::size_t default_num_stripes = NumStripes;
};

// ============================================================================
// Type trait: check if a policy is striped
// ============================================================================

template <typename Policy>
inline constexpr bool is_striped_policy_v = Policy::is_striped;

// ============================================================================
// P2-3 (T3.4): Type trait — check if an MM type provides per-shard locks
// (e.g. sharded_mm_lru). When true, unified_cache delegates lock
// acquisition to `mm_.acquire_shard_*_lock(shard_idx)` instead of
// `striped_mutex_`, decoupling `num_shards` from `num_stripes`.
// ============================================================================

template <typename MM, typename = void>
struct has_per_shard_lock_impl : std::false_type {};

template <typename MM>
struct has_per_shard_lock_impl<MM, std::void_t<decltype(MM::has_per_shard_lock)>>
    : std::bool_constant<bool(MM::has_per_shard_lock)> {};

template <typename MM>
inline constexpr bool has_per_shard_lock_v = has_per_shard_lock_impl<MM>::value;

// ============================================================================
// Cache Trait
// ============================================================================

/// A cache trait combines an MM (Memory Management) strategy type with
/// a lock policy type. This is inspired by CacheLib's CacheTrait design
/// which composes MMType, AccessType, and LockType at compile time.
///
/// @tparam MMType The MM strategy class template (e.g., mm_lru, mm_2q)
/// @tparam LockPolicy The lock policy (e.g., single_threaded_policy, thread_safe_policy)
/// @tparam ProbingStyle The hash table probing style (chain_probing_tag or f14_probing_tag)
/// @tparam Segmented When true, uses segmented_concurrent_hash_table for per-segment rehash
/// @tparam Hook The intrusive hook type used by cache_item (intrusive_hook or
///              compressed_intrusive_hook). T15: stored for future propagation
///              to MM strategies; current MM strategies hardcode intrusive_hook.
template <template<typename, typename, typename, typename, typename, bool> class MMType,
          typename LockPolicy = single_threaded_policy,
          typename ProbingStyle = detail::chain_probing_tag,
          bool Segmented = false,
          typename Hook = detail::intrusive_hook>
struct cache_trait {
    template <typename Key, typename Value, typename Hash = std::hash<Key>, typename KeyEqual = std::equal_to<Key>>
    using mm_type = MMType<Key, Value, Hash, KeyEqual, ProbingStyle, Segmented>;

    using lock_policy = LockPolicy;
    using probing_style = ProbingStyle;
    static constexpr bool segmented = Segmented;

    /// T15.1: The intrusive hook type. When `compressed_intrusive_hook`,
    /// callers can pair this trait with a `compressed_region` allocator
    /// to reduce per-item hook overhead from 16 to 8 bytes.
    using hook_type = Hook;
    static constexpr bool uses_compressed_hook =
        std::is_same_v<Hook, ::lru::compressed_intrusive_hook>;

    /// Task 9: When true, unified_cache auto-starts the event_drain_worker
    /// in the constructor. Production/read-heavy aliases set this to true;
    /// regular aliases (cache, safe_cache) keep the default (false).
    static constexpr bool auto_start_drain = false;

    /// P1-4: When true, init_production_features() auto-starts the
    /// metrics cache worker (1s default interval) so stats_snapshot()
    /// and prometheus_text() return cached values instead of rebuilding
    /// on every scrape. Production aliases (production_cache,
    /// segmented_*, f14_production_cache, read_heavy_*) set this to
    /// true — under high scrape rates (Prometheus every 10-15s with
    /// many caches), rebuilding on every call becomes a CPU hotspot.
    /// Aliases that may be used in benchmarks (cache, safe_cache,
    /// striped_cache) keep the default (false) so callers always see
    /// fresh metrics without surprises. Custom traits can opt in by
    /// setting auto_start_metrics_cache = true.
    static constexpr bool auto_start_metrics_cache = false;

    /// T11.1: When true, unified_cache auto-enables incremental rehash
    /// in the constructor. Segmented/production aliases set this to true
    /// to avoid blocking all writers during hash table expansion. F14
    /// probing mode also supports incremental rehash (dual-array lookup).
    static constexpr bool auto_incremental_rehash = false;

    /// P0-5: When true, unified_cache auto-starts the background rehash
    /// balancer in the constructor. The balancer periodically sweeps all
    /// segments (segmented hash table only) and advances rehash state
    /// for segments that aren't on the current write hot path. This
    /// prevents cold segments from accumulating unbounded chain length
    /// when no client writes are routing to them. Production aliases
    /// (production_*, segmented_*, read_heavy_*) set this to true.
    static constexpr bool auto_start_rehash_balancer = false;

    /// T-P1-1 (R-1): The default fairness mode applied by
    /// init_production_features() when the trait is thread-safe.
    ///   - reader_preferred: max read throughput, may starve writers under
    ///     sustained read load. Recommended for read-heavy-write-light
    ///     production workloads (the documented target of this library).
    ///   - writer_fair: prevents writer starvation at the cost of slightly
    ///     higher read latency when a writer is queued.
    /// The base trait keeps writer_fair (the historical default) to
    /// preserve behavior for low-level unified_cache instantiations.
    /// Thread-safe production aliases override this to reader_preferred
    /// via their own trait specializations.
    static constexpr detail::fairness_mode default_fairness =
        detail::fairness_mode::writer_fair;

    /// T-P1-2 (R-2): When true, init_production_features() applies
    /// Trait::default_fairness to the underlying mutex(es). The base
    /// trait leaves this false so direct unified_cache<cache_trait<...>>
    /// instantiations preserve the lock's intrinsic default. Aliases
    /// intended for thread-safe production use (safe_cache, striped_cache,
    /// production_cache, read_heavy_*) set this to true.
    static constexpr bool apply_default_fairness = false;

    /// R-3 (read-heavy trait unification): When true,
    /// init_production_features() enables defer_promotion on the
    /// underlying MM config after construction. This eliminates the
    /// historical split where `read_heavy_cache` (alias) and
    /// `make_read_heavy_cache` (factory) produced different behaviors.
    /// Only takes effect on MM types whose config has a
    /// `defer_promotion` field (mm_lru / sharded_mm_lru / mm_wtiny_lfu);
    /// on other MM types it is a SFINAE-guarded no-op.
    static constexpr bool auto_defer_promotion = false;

    /// R-3 (read-heavy trait unification): When true,
    /// init_production_features() calls
    /// `set_ebr_domain(&detail::epoch_domain::default_domain())` to
    /// switch the read path from hazptr-based reclamation to EBR.
    /// EBR's read path is faster than hazptr under sustained read
    /// contention (TLS epoch guard vs. per-node hazptr acquire/release).
    /// Only takes effect on MM types that support EBR (mm_lru /
    /// sharded_mm_lru); on other MM types (mm_2q / mm_tiny_lfu /
    /// mm_wtiny_lfu / mm_fifo) it is a guarded no-op.
    static constexpr bool auto_enable_ebr = false;

    /// T-G2: When true, init_production_features() sets a default memory
    /// limit of `default_max_memory_bytes` on the memory monitor. This
    /// prevents unbounded memory growth in production when the user forgets
    /// to call set_memory_monitor() / set_max_memory() explicitly. The
    /// user can still override via set_max_memory() after construction.
    /// Production aliases set this to true; low-level aliases keep false.
    static constexpr bool auto_set_default_memory_limit = false;

    /// T-G2: The default memory limit (bytes) applied when
    /// auto_set_default_memory_limit is true. 1 GiB is a conservative
    /// default that protects against runaway growth without cramping
    /// typical workloads. Operators should tune via set_max_memory().
    static constexpr std::size_t default_max_memory_bytes =
        static_cast<std::size_t>(1) << 30;  // 1 GiB

    /// T-G3: When true, init_production_features() enables singleflight
    /// coalescing for get_or_fetch() / try_get_or_fetch(). Concurrent
    /// misses for the same key are coalesced — only the leader calls the
    /// provider; followers block on the leader's result. This prevents
    /// cache-stampede / thundering-herd when a hot key expires under
    /// high concurrency. Production / read-heavy aliases set this to true.
    static constexpr bool auto_enable_singleflight = false;

    /// T-G16: When true, init_production_features() enables the slab
    /// allocator for item allocation. The slab allocator reduces allocation
    /// overhead and memory fragmentation for caches with many small items
    /// (10M+ items) by maintaining per-size-class free lists (lock-free
    /// Treiber stacks). Typical savings: ~30% memory reduction at 10M items
    /// vs. new/delete, plus better cache locality from packed allocation.
    ///
    /// Not enabled by default because:
    ///   - Slab allocator adds ~10MB overhead per cache (size-class tables).
    ///   - For small caches (<100K items), the overhead exceeds savings.
    ///   - Slab size classes are tuned for typical item sizes; workloads
    ///     with highly variable value sizes benefit less.
    ///
    /// Use `production_cache_with_slab` alias to opt in.
    static constexpr bool auto_enable_slab = false;

    /// T14.1: When true, unified_cache delegates storage operations to an
    /// embedded compact_cache (which uses compact_slot_allocator for
    /// dense fixed-size slot storage). This achieves 50-70% memory savings
    /// for small items (sizeof(K)+sizeof(V) <= 64) by eliminating per-item
    /// heap allocator metadata overhead.
    ///
    /// When is_compact=true:
    ///   - An embedded compact_cache member handles set/get/peek/remove/etc.
    ///   - A compile-time static_assert enforces sizeof(K)+sizeof(V) <= 64.
    ///   - get() returns a non-pinning read_handle (compact_cache uses slot
    ///     storage without refcount; the handle is valid only under external
    ///     synchronization or while the cache's read lock is held).
    ///   - Features requiring refcount pinning (try_get pinning, TTL, slab
    ///     allocator, hazptr/EBR reclamation) are not available on the
    ///     compact path; calling them is a no-op or returns a default value.
    static constexpr bool is_compact = false;
};

// ============================================================================
// Pre-defined Trait Aliases
// ============================================================================

/// LRU trait with configurable insertion point, delayed promotion, adaptive refresh
template <typename LockPolicy = single_threaded_policy>
using lru_trait = cache_trait<mm_lru, LockPolicy>;

/// T15.2: Compressed LRU trait — uses compressed_intrusive_hook for ~33%
/// per-item hook memory savings (16B → 8B). The trait exposes
/// `uses_compressed_hook` so downstream code (allocators, diagnostics)
/// can detect the intended hook type.
///
/// Note: current MM strategies hardcode `intrusive_hook` internally; the
/// `Hook` template parameter is stored on the trait and exposed via
/// `uses_compressed_hook` for downstream code (e.g. allocators, diagnostics)
/// to detect the intended hook type. Full propagation to mm_lru is a
/// future refactor.
template <typename LockPolicy = single_threaded_policy>
using compressed_lru_trait = cache_trait<mm_lru, LockPolicy,
                                         detail::chain_probing_tag, false,
                                         compressed_intrusive_hook>;

/// T15.2: Compressed sharded LRU trait — pairs compressed_intrusive_hook with
/// sharded_mm_lru so the result is compatible with striped_thread_safe_policy.
/// (striped_thread_safe_policy requires sharded_mm_lru; plain mm_lru lacks
/// shard_for/shard/num_shards and cannot be used with striped locking.)
template <typename LockPolicy = single_threaded_policy>
using compressed_sharded_lru_trait = cache_trait<sharded_mm_lru, LockPolicy,
                                                  detail::chain_probing_tag, false,
                                                  compressed_intrusive_hook>;

// ============================================================================
// T14.1/T14.2: Compact unified LRU traits
// ============================================================================
//
// These traits set `is_compact = true` on cache_trait. When unified_cache
// is instantiated with one of these traits, it embeds a compact_cache member
// that uses compact_slot_allocator (dense fixed-size slot storage) instead
// of the regular mm_lru path. This achieves 50-70% memory savings for small
// items (sizeof(K)+sizeof(V) <= 64) by eliminating per-item heap allocator
// metadata overhead.
//
// The unified_cache API remains the same as the regular unified_cache, with
// the following caveats:
//   - get() returns a non-pinning read_handle (no refcount on compact slots)
//   - TTL, slab allocator, and hazptr/EBR reclamation are not available
//   - A static_assert enforces sizeof(K)+sizeof(V) <= 64 at compile time

template <typename LockPolicy = single_threaded_policy>
struct compact_unified_lru_trait : cache_trait<mm_lru, LockPolicy> {
    static constexpr bool is_compact = true;
};

template <typename LockPolicy = single_threaded_policy>
struct compact_unified_sharded_lru_trait : cache_trait<sharded_mm_lru, LockPolicy> {
    static constexpr bool is_compact = true;
};

/// 2Q trait with Hot/Warm/Cold queues
template <typename LockPolicy = single_threaded_policy>
using two_q_trait = cache_trait<mm_2q, LockPolicy>;

/// TinyLFU trait with frequency-aware admission
template <typename LockPolicy = single_threaded_policy>
using tiny_lfu_trait = cache_trait<mm_tiny_lfu, LockPolicy>;

/// W-TinyLFU trait with Probation/Protection segments
template <typename LockPolicy = single_threaded_policy>
using w_tiny_lfu_trait = cache_trait<mm_wtiny_lfu, LockPolicy>;

/// FIFO trait (no reordering on access)
template <typename LockPolicy = single_threaded_policy>
using fifo_trait = cache_trait<mm_fifo, LockPolicy>;

/// Sharded LRU trait — uses sharded_mm_lru for partitionable concurrency
template <typename LockPolicy = single_threaded_policy>
using sharded_lru_trait = cache_trait<sharded_mm_lru, LockPolicy>;

// ============================================================================
// T-P3: Production-safe traits for safe_cache / striped_cache aliases
// ============================================================================
//
// Historically the `safe_cache` and `striped_cache` type aliases kept the
// lock's intrinsic `writer_fair` default and left incremental rehash
// disabled, while the factory functions `make_safe_cache` /
// `make_striped_cache` applied `reader_preferred` + incremental rehash.
// This split behavior was a frequent source of misuse: users who constructed
// `safe_cache<K,V>(n)` directly got a configuration that suffered periodic
// read throughput drops under read-heavy-write-light workloads (the
// documented target of this library) and global write stalls during hash
// table expansion.
//
// These traits close the gap by giving the type aliases the same
// production-safe defaults the factories applied:
//   - auto_incremental_rehash = true   (no global write stall on rehash)
//   - apply_default_fairness  = true
//   - default_fairness        = reader_preferred (max read throughput)
//
// R-1: The redundant `make_safe_cache` / `make_striped_cache` factory
// functions have been removed from lru.hpp — direct construction
// (`safe_cache<K,V>(n)` / `striped_cache<K,V>(n)`) is now the only way
// to obtain these caches and produces the same behavior.
//
// Users with mixed or write-heavy workloads who need strict writer fairness
// can still call `c.set_fairness_mode(lru::detail::fairness_mode::writer_fair)`
// after construction, or instantiate `unified_cache<lru_trait<thread_safe_policy>,
// K, V>` directly to bypass the safe defaults.

/// Production-safe LRU trait for the `safe_cache` alias. Inherits the base
/// `lru_trait` and opts in to incremental rehash + reader_preferred fairness.
template <typename LockPolicy = single_threaded_policy>
struct safe_lru_trait : cache_trait<mm_lru, LockPolicy> {
    static constexpr bool auto_incremental_rehash = true;
    static constexpr detail::fairness_mode default_fairness =
        detail::fairness_mode::reader_preferred;
    static constexpr bool apply_default_fairness = true;
};

/// Production-safe sharded LRU trait for the `striped_cache` alias.
/// Inherits the base `sharded_lru_trait` and opts in to incremental rehash
/// + reader_preferred fairness.
template <typename LockPolicy = single_threaded_policy>
struct safe_sharded_lru_trait : cache_trait<sharded_mm_lru, LockPolicy> {
    static constexpr bool auto_incremental_rehash = true;
    static constexpr detail::fairness_mode default_fairness =
        detail::fairness_mode::reader_preferred;
    static constexpr bool apply_default_fairness = true;
};

// ============================================================================
// R-3: Read-heavy traits — unify alias and factory behavior
// ============================================================================
//
// Historically `read_heavy_cache` / `read_heavy_striped_cache` /
// `read_heavy_w_tiny_lfu` aliases used the base `lru_trait` /
// `sharded_lru_trait` / `w_tiny_lfu_trait`, which do NOT enable
// `defer_promotion` or EBR. The `make_read_heavy_*` factory functions
// manually applied those settings after construction. This split caused
// the aliases to be name-misleading: `read_heavy_cache<K,V> c(n);` was
// not actually a read-heavy-optimized cache.
//
// These traits close the gap by opting in to the same settings the
// factories applied, so direct construction
// (`read_heavy_cache<K,V>(n)`, `read_heavy_striped_cache<K,V>(n)`,
// `read_heavy_w_tiny_lfu<K,V>(n)`) now produces a cache equivalent to
// the historical factory output. The factory functions themselves are
// removed (see lru.hpp).

/// Read-heavy LRU trait — single-threaded or thread-safe LRU with
/// defer_promotion + EBR + incremental rehash + reader_preferred fairness.
/// Only takes effect under `thread_safe_policy` (single-threaded caches
/// have no contention to optimize for).
template <typename LockPolicy = single_threaded_policy>
struct read_heavy_lru_trait : cache_trait<mm_lru, LockPolicy> {
    static constexpr bool auto_incremental_rehash = true;
    static constexpr detail::fairness_mode default_fairness =
        detail::fairness_mode::reader_preferred;
    static constexpr bool apply_default_fairness = true;
    // defer_promotion / EBR are no-ops on single_threaded_policy caches
    // (no concurrent readers → no contention to defer; no evictions in
    // flight → no reclamation to defer), so enabling them unconditionally
    // is safe.
    static constexpr bool auto_defer_promotion = true;
    static constexpr bool auto_enable_ebr = true;
    // P1-4: read-heavy aliases are production-oriented — auto-enable
    // metrics cache to avoid O(N) stats rebuilds under high scrape rates.
    static constexpr bool auto_start_metrics_cache = true;
    // T-G2: production safety — enforce default memory limit.
    static constexpr bool auto_set_default_memory_limit = true;
    // T-G3: production safety — auto-enable singleflight.
    static constexpr bool auto_enable_singleflight = true;
};

/// Read-heavy sharded LRU trait — striped sharded LRU with the same
/// read-heavy optimizations as `read_heavy_lru_trait`.
template <typename LockPolicy = single_threaded_policy>
struct read_heavy_sharded_lru_trait : cache_trait<sharded_mm_lru, LockPolicy> {
    static constexpr bool auto_incremental_rehash = true;
    static constexpr detail::fairness_mode default_fairness =
        detail::fairness_mode::reader_preferred;
    static constexpr bool apply_default_fairness = true;
    static constexpr bool auto_defer_promotion = true;
    static constexpr bool auto_enable_ebr = true;
    static constexpr bool auto_start_metrics_cache = true;
    // T-G2: production safety — enforce default memory limit.
    static constexpr bool auto_set_default_memory_limit = true;
    // T-G3: production safety — auto-enable singleflight.
    static constexpr bool auto_enable_singleflight = true;
};

/// Read-heavy W-TinyLFU trait — thread-safe W-TinyLFU with defer_promotion.
/// EBR is NOT enabled because mm_wtiny_lfu does not support EBR (only
/// mm_lru / sharded_mm_lru support it); the auto_enable_ebr flag is
/// SFINAE-guarded to a no-op here, so we leave it false for clarity.
template <typename LockPolicy = single_threaded_policy>
struct read_heavy_w_tiny_lfu_trait : cache_trait<mm_wtiny_lfu, LockPolicy> {
    static constexpr bool auto_incremental_rehash = true;
    static constexpr detail::fairness_mode default_fairness =
        detail::fairness_mode::reader_preferred;
    static constexpr bool apply_default_fairness = true;
    static constexpr bool auto_defer_promotion = true;
    // mm_wtiny_lfu does not support EBR; left false intentionally.
    static constexpr bool auto_enable_ebr = false;
    static constexpr bool auto_start_metrics_cache = true;
};

// ============================================================================
// Segmented Trait Aliases (per-segment rehash, no global stall)
// ============================================================================

/// Segmented LRU trait — uses segmented_concurrent_hash_table for per-segment rehash
template <typename LockPolicy = single_threaded_policy>
struct segmented_lru_trait : cache_trait<mm_lru, LockPolicy, detail::chain_probing_tag, true> {
    static constexpr bool auto_incremental_rehash = true;
    static constexpr bool auto_start_rehash_balancer = true;
    // P1-4: production-oriented — auto-enable metrics cache (only spawns
    // worker under thread-safe policy; see init_production_features).
    static constexpr bool auto_start_metrics_cache = true;
};

/// Segmented 2Q trait
template <typename LockPolicy = single_threaded_policy>
struct segmented_two_q_trait : cache_trait<mm_2q, LockPolicy, detail::chain_probing_tag, true> {
    static constexpr bool auto_incremental_rehash = true;
    static constexpr bool auto_start_rehash_balancer = true;
    static constexpr bool auto_start_metrics_cache = true;
};

/// Segmented TinyLFU trait
template <typename LockPolicy = single_threaded_policy>
struct segmented_tiny_lfu_trait : cache_trait<mm_tiny_lfu, LockPolicy, detail::chain_probing_tag, true> {
    static constexpr bool auto_incremental_rehash = true;
    static constexpr bool auto_start_rehash_balancer = true;
    static constexpr bool auto_start_metrics_cache = true;
};

/// Segmented W-TinyLFU trait
template <typename LockPolicy = single_threaded_policy>
struct segmented_w_tiny_lfu_trait : cache_trait<mm_wtiny_lfu, LockPolicy, detail::chain_probing_tag, true> {
    static constexpr bool auto_incremental_rehash = true;
    static constexpr bool auto_start_rehash_balancer = true;
    static constexpr bool auto_start_metrics_cache = true;
};

/// Segmented FIFO trait
template <typename LockPolicy = single_threaded_policy>
struct segmented_fifo_trait : cache_trait<mm_fifo, LockPolicy, detail::chain_probing_tag, true> {
    static constexpr bool auto_incremental_rehash = true;
    static constexpr bool auto_start_rehash_balancer = true;
    static constexpr bool auto_start_metrics_cache = true;
};

/// Segmented sharded LRU trait — sharded MM + segmented hash table
template <typename LockPolicy = single_threaded_policy>
struct segmented_sharded_lru_trait : cache_trait<sharded_mm_lru, LockPolicy, detail::chain_probing_tag, true> {
    static constexpr bool auto_incremental_rehash = true;
    static constexpr bool auto_start_rehash_balancer = true;
    static constexpr bool auto_start_metrics_cache = true;
};

/// Task 9/14: Production-optimized segmented sharded LRU trait.
/// Inherits all settings from segmented_sharded_lru_trait but enables:
///   - auto_start_drain: event_drain_worker starts automatically on construction
///   - auto_enable_ebr: EBR (Epoch-Based Reclamation) enabled by default.
///     EBR's read path is significantly faster than hazptr under sustained
///     read contention (32+ threads, 99%+ reads) — TLS epoch guard vs.
///     per-node hazptr acquire/release on every hash chain walk.
///   - auto_incremental_rehash: incremental rehash enabled to avoid blocking writes
///   - apply_default_fairness + default_fairness=reader_preferred (T-P1-1 / R-1):
///     production_cache defaults to reader_preferred for read-heavy workloads.
///
/// R2: EmbeddedChain enforcement. The underlying MM strategy's hash table
/// is statically asserted to use EmbeddedChain=true (see mm.hpp
/// static_assert). Non-EmbeddedChain mode degrades lock-free reads to
/// shared_lock fallback, killing read throughput under high concurrency.
/// DO NOT bypass this assertion — it is a hard safety requirement for
/// production read-heavy-write-light workloads (32+ threads, 99%+ reads).
template <typename LockPolicy = single_threaded_policy>
struct production_sharded_lru_trait : segmented_sharded_lru_trait<LockPolicy> {
    static constexpr bool auto_start_drain = true;
    // EBR enabled by default — production workloads are read-heavy (99%+ reads).
    // EBR's read path uses TLS epoch guards (no per-node hazptr acquire/release),
    // giving 30-50% lower read latency than hazptr under sustained contention.
    static constexpr bool auto_enable_ebr = true;
    // Read-heavy default: hits skip the per-shard write-lock try and defer LRU
    // promotion to the TLS access ring. Matches read_heavy_* aliases.
    static constexpr bool auto_defer_promotion = true;
    static constexpr bool auto_incremental_rehash = true;
    static constexpr bool auto_start_rehash_balancer = true;

    // T-P1-1 (R-1): production_cache defaults to reader_preferred.
    static constexpr detail::fairness_mode default_fairness =
        detail::fairness_mode::reader_preferred;
    static constexpr bool apply_default_fairness = true;

    // R2: Compile-time verification that EmbeddedChain is enforced.
    // The actual assertion lives in mm_lru's map_type static_assert.
    // This constant documents the requirement at the trait level.
    static constexpr bool requires_embedded_chain = true;

    // T-G7: production-oriented — auto-enable metrics cache. Without this,
    // high-frequency Prometheus scraping (every 10-15s) rebuilds stats on
    // every call, becoming a CPU hotspot. Aligns with f14_production_cache.
    static constexpr bool auto_start_metrics_cache = true;

    // T-G2: production safety — enforce a default memory limit so the cache
    // cannot cause OOM if the operator forgets to call set_max_memory().
    static constexpr bool auto_set_default_memory_limit = true;
    // T-G3: production safety — auto-enable singleflight to prevent
    // cache-stampede / thundering-herd on hot key expiry.
    static constexpr bool auto_enable_singleflight = true;
};

/// F14 + Production trait: combines F14 SIMD probing with all production
/// optimizations (segmented hash table, EBR, incremental rehash, reader_preferred).
template <typename LockPolicy = single_threaded_policy>
struct f14_production_sharded_lru_trait
    : cache_trait<sharded_mm_lru, LockPolicy, detail::f14_probing_tag, true> {
    static constexpr bool auto_start_drain = true;
    static constexpr bool auto_enable_ebr = true;
    static constexpr bool auto_defer_promotion = true;
    static constexpr bool auto_incremental_rehash = true;
    static constexpr bool auto_start_rehash_balancer = true;
    static constexpr detail::fairness_mode default_fairness =
        detail::fairness_mode::reader_preferred;
    static constexpr bool apply_default_fairness = true;
    static constexpr bool requires_embedded_chain = true;
    // P1-4: production-oriented — auto-enable metrics cache.
    static constexpr bool auto_start_metrics_cache = true;
    // T-G2: production safety — enforce default memory limit.
    static constexpr bool auto_set_default_memory_limit = true;
    // T-G3: production safety — auto-enable singleflight.
    static constexpr bool auto_enable_singleflight = true;
};

/// T-G16: Production trait with slab allocator enabled. Inherits all
/// production_cache optimizations (segmented hash table, EBR, incremental
/// rehash, reader_preferred fairness, singleflight, default memory limit)
/// and additionally enables the slab allocator for item allocation. Use
/// via the `production_cache_with_slab` alias.
///
/// Slab allocation reduces memory overhead for caches with many small
/// items (10M+) by maintaining per-size-class free lists. Typical savings:
/// ~30% memory reduction at 10M items vs. new/delete, plus better cache
/// locality. The slab allocator is thread-safe (lock-free Treiber stack
/// per size class) and suitable for high-concurrency workloads.
///
/// Trade-offs:
///   - ~10MB overhead per cache (size-class tables) — net savings only
///     above ~100K items.
///   - Slab size classes are tuned for typical item sizes; workloads with
///     highly variable value sizes benefit less.
///   - Slab rebalancing is not supported (was unsafe under concurrent set).
template <typename LockPolicy = single_threaded_policy>
struct production_with_slab_sharded_lru_trait : production_sharded_lru_trait<LockPolicy> {
    static constexpr bool auto_enable_slab = true;
};

// ============================================================================
// F14 SIMD Probing Trait Aliases
// ============================================================================

/// LRU trait with F14 SIMD probing for faster hash lookups
template <typename LockPolicy = single_threaded_policy>
using f14_lru_trait = cache_trait<mm_lru, LockPolicy, detail::f14_probing_tag>;

/// 2Q trait with F14 SIMD probing
template <typename LockPolicy = single_threaded_policy>
using f14_two_q_trait = cache_trait<mm_2q, LockPolicy, detail::f14_probing_tag>;

/// TinyLFU trait with F14 SIMD probing
template <typename LockPolicy = single_threaded_policy>
using f14_tiny_lfu_trait = cache_trait<mm_tiny_lfu, LockPolicy, detail::f14_probing_tag>;

/// W-TinyLFU trait with F14 SIMD probing
template <typename LockPolicy = single_threaded_policy>
using f14_w_tiny_lfu_trait = cache_trait<mm_wtiny_lfu, LockPolicy, detail::f14_probing_tag>;

/// FIFO trait with F14 SIMD probing
template <typename LockPolicy = single_threaded_policy>
using f14_fifo_trait = cache_trait<mm_fifo, LockPolicy, detail::f14_probing_tag>;

/// Sharded LRU trait with F14 SIMD probing
template <typename LockPolicy = single_threaded_policy>
using f14_sharded_lru_trait = cache_trait<sharded_mm_lru, LockPolicy, detail::f14_probing_tag>;

// ============================================================================
// SFINAE detector for mm_type::peek()
// ============================================================================

namespace detail {

template <typename MmType, typename Key, typename = void>
struct has_peek : std::false_type {};

template <typename MmType, typename Key>
struct has_peek<MmType, Key, std::void_t<decltype(std::declval<const MmType&>().peek(std::declval<const Key&>()))>> : std::true_type {};

template <typename MmType, typename Key>
inline constexpr bool has_peek_v = has_peek<MmType, Key>::value;

// ============================================================================
// TTL jitter — applied by set_with_ttl() to prevent thundering-herd
// avalanches when many keys share the same nominal TTL. With jitter_pct
// = 0.10, the effective TTL is uniformly distributed in [dur*0.9, dur*1.1].
//
// Uses a thread-local xorshift64 PRNG seeded from thread_id — ~5ns per call,
// no locks, no allocation. Quality is sufficient for jitter; do not reuse
// for cryptographic purposes. jitter_pct <= 0 returns dur unchanged.
// ============================================================================
template <typename Rep, typename Period>
std::chrono::duration<Rep, Period>
apply_ttl_jitter(std::chrono::duration<Rep, Period> dur, double jitter_pct) {
    if (jitter_pct <= 0.0) return dur;
    thread_local uint64_t state = static_cast<uint64_t>(
        std::hash<std::thread::id>{}(std::this_thread::get_id())) | 1u;
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    double u = static_cast<double>(state >> 11) *
               (1.0 / static_cast<double>(1ull << 53));
    double factor = 1.0 + jitter_pct * (2.0 * u - 1.0);
    // Operate in nanoseconds to preserve sub-second precision. Without this,
    // a 1s TTL with ±10% jitter would truncate 0.9s → 0s (immediate expiry)
    // ~50% of the time. Converting via nanoseconds keeps the full resolution.
    auto ns_count = std::chrono::duration_cast<std::chrono::nanoseconds>(dur).count();
    auto jittered_ns = static_cast<long long>(static_cast<double>(ns_count) * factor);
    if (jittered_ns < 1) jittered_ns = 1;
    return std::chrono::duration_cast<std::chrono::duration<Rep, Period>>(
        std::chrono::nanoseconds(jittered_ns));
}

// ============================================================================
// RAII latency timer — records elapsed ns to a latency_histogram on scope exit.
// Used by unified_cache::get() / set() to populate cache_stats::get_latency
// and set_latency. Cheap: one steady_clock::now() on construction + one on
// destruction + one fetch_add. The histogram is passed by reference.
// ============================================================================

class scope_latency_timer {
public:
    explicit scope_latency_timer(latency_histogram& hist, bool enabled = true) noexcept
        : hist_(enabled ? &hist : nullptr) {
        if (hist_) {
            start_ = std::chrono::steady_clock::now();
        }
    }

    /// O2: Extended constructor — also writes the measured latency (ns)
    /// to `*out_latency_ns` on destruction. Pass a non-null pointer to
    /// capture the latency for slow-query threshold checking. The output
    /// variable must outlive the timer (typically a stack variable in
    /// the caller's scope).
    scope_latency_timer(latency_histogram& hist, bool enabled,
                        std::uint64_t* out_latency_ns) noexcept
        : hist_(enabled ? &hist : nullptr),
          out_latency_ns_(out_latency_ns) {
        if (hist_) {
            start_ = std::chrono::steady_clock::now();
        }
    }

    ~scope_latency_timer() noexcept {
        if (hist_) {
            auto end = std::chrono::steady_clock::now();
            auto ns = static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(end - start_).count());
            hist_->record(ns);
            // O2: write the measured latency to the caller's variable so
            // they can check the slow-query threshold after we destruct.
            // No atomicity needed — same-thread sequenced-before relationship.
            if (out_latency_ns_) {
                *out_latency_ns_ = ns;
            }
        }
    }

    scope_latency_timer(const scope_latency_timer&) = delete;
    scope_latency_timer& operator=(const scope_latency_timer&) = delete;
    scope_latency_timer(scope_latency_timer&& other) noexcept
        : hist_(other.hist_), start_(other.start_),
          out_latency_ns_(other.out_latency_ns_) { other.hist_ = nullptr; }
    scope_latency_timer& operator=(scope_latency_timer&&) = delete;

private:
    latency_histogram* hist_;
    std::chrono::steady_clock::time_point start_{};
    std::uint64_t* out_latency_ns_ = nullptr;
};

} // namespace detail

// ============================================================================
// O2: slow_query_notifier — RAII helper that invokes the cache's slow-query
// callback on scope exit. Declared BEFORE scope_latency_timer so it destructs
// AFTER the timer (destruction is reverse of construction), allowing it to
// read the latency value written by the timer's destructor.
// ============================================================================
template <typename Cache, typename Key>
class slow_query_notifier {
public:
    slow_query_notifier(const Cache& cache, const Key& key,
                        typename Cache::slow_op_kind kind,
                        std::uint64_t& latency_ref) noexcept
        : cache_(cache), key_(key), kind_(kind), latency_ref_(latency_ref) {}
    ~slow_query_notifier() noexcept {
        cache_.notify_slow_query(kind_, key_, latency_ref_);
    }
    slow_query_notifier(const slow_query_notifier&) = delete;
    slow_query_notifier& operator=(const slow_query_notifier&) = delete;
    slow_query_notifier(slow_query_notifier&&) = delete;
    slow_query_notifier& operator=(slow_query_notifier&&) = delete;
private:
    const Cache& cache_;
    const Key& key_;
    typename Cache::slow_op_kind kind_;
    std::uint64_t& latency_ref_;
};

// ============================================================================
// O1: trace_notifier — RAII helper that invokes the cache's trace callback
// on scope exit. Declared BEFORE scope_latency_timer (and before
// slow_query_notifier) so it destructs AFTER both, allowing it to read:
//   - the latency value written by ~scope_latency_timer
//   - the hit flag set by the operation body
//   - the error status via std::uncaught_exceptions() (no explicit flag needed)
// Construction order in get()/set():
//   trace_notifier → slow_query_notifier → scope_latency_timer
// Destruction order (reverse):
//   ~scope_latency_timer (writes latency_ns)
//   ~slow_query_notifier (reads latency_ns → notify_slow_query)
//   ~trace_notifier (reads latency_ns + hit → notify_trace)
// ============================================================================
template <typename Cache, typename Key>
class trace_notifier {
public:
    trace_notifier(const Cache& cache, const Key& key,
                   typename Cache::trace_op_kind op,
                   std::uint64_t& latency_ref,
                   const bool& hit_ref) noexcept
        : cache_(cache), key_(key), op_(op), latency_ref_(latency_ref),
          hit_ref_(hit_ref) {}
    ~trace_notifier() noexcept {
        // std::uncaught_exceptions() distinguishes normal scope exit (0)
        // from stack unwinding due to an active exception (> 0). This
        // correctly reports error=true when get()/set() threw, without
        // requiring call sites to wrap their bodies in try/catch.
        const bool error = std::uncaught_exceptions() > 0;
        cache_.notify_trace(op_, key_, latency_ref_, hit_ref_, error);
    }
    trace_notifier(const trace_notifier&) = delete;
    trace_notifier& operator=(const trace_notifier&) = delete;
    trace_notifier(trace_notifier&&) = delete;
    trace_notifier& operator=(trace_notifier&&) = delete;
private:
    const Cache& cache_;
    const Key& key_;
    typename Cache::trace_op_kind op_;
    std::uint64_t& latency_ref_;
    const bool& hit_ref_;
};

// ============================================================================
// Unified Cache Template
// ============================================================================

/// Unified cache that combines an MM strategy with an optional lock policy.
/// This is the main entry point for the new trait-based architecture.
///
/// Usage:
///   // Non-thread-safe LRU cache
///   lru::unified_cache<lru::lru_trait<>, int, std::string> c(100);
///
///   // Thread-safe LRU cache
///   lru::unified_cache<lru::lru_trait<lru::thread_safe_policy>, int, std::string> c(100);
///
///   // Striped thread-safe LRU cache (currently serialized; see striped_thread_safe_policy<>)
///   lru::unified_cache<lru::lru_trait<lru::striped_thread_safe_policy<>>, int, std::string> c(100);
///
///   // Non-thread-safe 2Q cache
///   lru::unified_cache<lru::two_q_trait<>, int, std::string> c(100);
///
///   // Thread-safe W-TinyLFU cache
///   lru::unified_cache<lru::w_tiny_lfu_trait<lru::thread_safe_policy>, int, std::string> c(100);
template <typename Trait, typename Key, typename Value,
          typename Hash = std::hash<Key>,
          typename KeyEqual = std::equal_to<Key>>
class unified_cache {
public:
    using trait_type = Trait;
    using lock_policy = typename Trait::lock_policy;
    using mm_type = typename Trait::template mm_type<Key, Value, Hash, KeyEqual>;

    using key_type = Key;
    using mapped_type = Value;
    using value_type = mapped_type;
    using size_type = std::size_t;
    /// T16: Expose the hash function type so callers can compute
    /// pre-hashed keys for use with get_prehashed / set_prehashed / etc.
    using hash_type = Hash;
    using key_equal_type = KeyEqual;

    using callback_mgr = typename mm_type::callback_mgr;
    using stats_type = typename mm_type::stats_type;

    using hit_callback_type = typename callback_mgr::hit_callback_type;
    using miss_callback_type = typename callback_mgr::miss_callback_type;
    using insert_callback_type = typename callback_mgr::insert_callback_type;
    using evict_callback_type = typename callback_mgr::eviction_callback_type;
    /// O7: New event callback types exposed for user registration.
    using update_callback_type = typename callback_mgr::update_callback_type;
    using expire_callback_type = typename callback_mgr::expire_callback_type;
    using reject_callback_type = typename callback_mgr::reject_callback_type;
    /// O6: Error hook type and event-kind enum exposed for users to
    /// register observation/alerting logic when callbacks throw.
    using callback_event_kind = typename callback_mgr::callback_event_kind;
    using error_callback_type = typename callback_mgr::error_callback_type;

    /// O2: Slow query callback — invoked when get()/set() exceeds the
    /// configured slow_query_threshold. The callback receives the
    /// operation kind ("get" / "set"), the key, and the measured
    /// latency in nanoseconds. Use this to log tail-latency events
    /// to a tracing system, emit metrics, or sample for diagnostics.
    /// The callback is invoked OUTSIDE the cache lock on the calling
    /// thread; it must be non-throwing and reasonably fast (it runs
    /// on the hot path of every slow operation).
    enum class slow_op_kind : uint8_t { get, set };
    using slow_query_callback_type = std::function<void(
        slow_op_kind kind, const Key& key, std::uint64_t latency_ns)>;

    /// O1: Trace callback — invoked at the end of get()/set() with the
    /// operation kind, key, measured latency, and hit/error status. This
    /// is the integration point for distributed tracing systems
    /// (OpenTelemetry, Jaeger, Zipkin, etc.).
    ///
    /// Unlike slow_query (which is threshold-gated), the trace callback
    /// fires on EVERY get()/set() when registered — the callback itself
    /// is responsible for sampling (e.g., random 1% sampling). The fast
    /// path (no callback registered) is a single atomic load + branch.
    ///
    /// The callback is invoked OUTSIDE the cache lock on the calling
    /// thread; it must be non-throwing and reasonably fast. For expensive
    /// span creation, queue to a dedicated thread inside the callback.
    ///
    /// @param op       Operation kind (get/set)
    /// @param key      Cache key (reference is valid only during the call)
    /// @param latency_ns  Measured wall-clock latency in nanoseconds
    /// @param hit      For get: true if found in cache, false on miss.
    ///                 For set: true if inserted/updated, false if rejected.
    /// @param error    true if the operation threw an exception
    enum class trace_op_kind : uint8_t { get, set };
    using trace_callback_type = std::function<void(
        trace_op_kind op, const Key& key, std::uint64_t latency_ns,
        bool hit, bool error)>;

    // B16: 结构化 RemoveRes 枚举（对齐 CacheLib CacheAllocator.h:708-711）
    enum class RemoveRes { kSuccess, kNotFound };

    /// CAS policy controlling how the predicate is evaluated.
    /// - kLockedPredicate:  predicate evaluated under write lock (existing behavior)
    /// - kLockfreePredicate: predicate evaluated outside lock, retry on conflict
    enum class cas_policy {
        kLockedPredicate,
        kLockfreePredicate
    };

    /// P-HIGH-2 (T-H1): OOM event descriptor returned by
    /// check_memory_pressure(). Carries the snapshot of memory usage at
    /// the moment the cache entered critical mode. The event is produced
    /// inside the write lock but the handler is invoked *outside* the
    /// lock (synchronously) or by the event_drain_worker (asynchronously),
    /// so the handler is free to call cache methods (flush/remove/set)
    /// without deadlocking.
    struct oom_event {
        size_type current_memory;
        size_type max_memory;
    };

    static constexpr bool is_thread_safe = lock_policy::is_thread_safe;
    static constexpr bool is_striped = is_striped_policy_v<lock_policy>;
    static constexpr size_type npos = unlimited;

private:
    /// Conditional mutex member. For non-striped thread-safe policies this is
    /// the global mutex (std::mutex). For striped policies, per-key operations
    /// use striped_mutex_ instead; the single mutex_ is not used.
    /// For single-threaded policies it is an empty tuple that takes no space
    /// thanks to [[no_unique_address]].
    using mutex_storage = std::conditional_t<
        is_thread_safe && !is_striped,
        typename lock_policy::mutex_type,
        std::tuple<>
    >;

    /// Striped mutex storage. Only present for striped policies.
    using striped_mutex_storage = std::conditional_t<
        is_striped,
        typename lock_policy::striped_mutex_type,
        std::tuple<>
    >;

public:
    // --------------------------------------------------------------------
    // Constructors
    // --------------------------------------------------------------------

    unified_cache() {
        init_production_features();
    }

    ~unified_cache() {
        // P1-4: One-time hint for high-frequency scrapers that did not
        // enable metrics cache. If stats_snapshot() was called more than
        // 10 times and the metrics cache worker is not running, the user
        // is likely paying O(N) per scrape unnecessarily. Emit a single
        // stderr hint (best-effort, no allocation in the common case).
        // Skip for single-threaded caches — the O(N) cost is negligible
        // without lock contention.
        if constexpr (Trait::lock_policy::is_thread_safe) {
            if (!metrics_cache_enabled_.load(std::memory_order_acquire) &&
                scrape_count_.load(std::memory_order_acquire) > 10) {
                std::fprintf(stderr,
                    "[lru] hint: stats_snapshot() called %zu times without "
                    "metrics cache enabled — call start_metrics_cache_worker(1s) "
                    "to avoid O(N) rebuilds per scrape (production aliases do "
                    "this automatically).\n",
                    static_cast<std::size_t>(scrape_count_.load(std::memory_order_relaxed)));
                std::fflush(stderr);
            }
        }
        try {
            // Task 11: mark the cache closed first so concurrent get()/set()
            // reject operations while we tear down. shutdown() is idempotent
            // and also stops the TTL cleaner + drain worker + final flush.
            shutdown();
            // P-LOW-1 (T-H5): Defensive wait for outstanding read_handles
            // before tearing down the MM. shutdown() already rejects new
            // operations, but handles acquired before shutdown() may still
            // be live on other threads. Without this wait, destroying mm_
            // while a handle still references an item is a UAF. The 5s
            // timeout is the O10 default — long enough to ride out slow
            // handle release under realistic load, short enough that a
            // genuinely stuck handle doesn't hang process teardown.
            // shutdown_and_wait() is safe to call here: it re-invokes
            // shutdown() (idempotent no-op) and, if handle tracking is
            // unreliable in this build, returns false immediately without
            // blocking. No exception is thrown on timeout.
            //
            // G6: If the wait times out (drained == false), mm_ destruction
            // will UAF. Debug builds catch this via the assert below; Release
            // builds log a CRITICAL error so the operator can diagnose the
            // inevitable crash. The proper fix is for the caller to invoke
            // shutdown_and_wait() explicitly and check the return value
            // before letting the cache go out of scope.
            const bool drained = shutdown_and_wait(std::chrono::seconds(5));
            if (!drained) {
                // G6: destroying mm_ while a handle still references an item
                // is a use-after-free. Debug builds assert; Release builds
                // must not silently corrupt memory — abort loudly so the
                // leaked handle / hung thread surfaces instead of UB.
                std::fprintf(stderr,
                    "[lru] CRITICAL (G6): unified_cache destructor timed out "
                    "waiting for %zu outstanding read_handles after 5s — "
                    "destroying mm_ now would cause use-after-free. Call "
                    "shutdown_and_wait() explicitly and check the return "
                    "value before letting the cache go out of scope.\n",
                    static_cast<std::size_t>(active_handle_count()));
                std::fflush(stderr);
                std::abort();
            }
            // P-CRIT-1 (T-C1): Defensive belt-and-suspenders — in case
            // shutdown() was never called by the user (cache went out of
            // scope directly), ensure the rehash balancer is joined before
            // mm_ is destroyed. shutdown() already calls this, but it is
            // idempotent, so a second call is a no-op if already stopped.
            // If shutdown() ran, this is a no-op; if shutdown() threw or
            // the user bypassed it, this still prevents UAF.
            stop_background_rehash_balancer();
            // T14.1: Compact path — compact_cache's destructor handles its
            // own teardown (slot destruction). Skip the mm_-specific drains.
            if constexpr (Trait::is_compact) {
                stop_ttl_cleaner();
                stop_event_drain();
                return;
            }
            // 0a. Stop the TTL cleaner so it doesn't race with destruction.
            stop_ttl_cleaner();
            // 0b. Stop the event drain worker first so no more periodic
            //    flushes race with destruction. The final drain below
            //    captures any events that were queued before stop.
            stop_event_drain();
            // 1. Drain access ring — promote deferred keys (including
            //    any orphaned keys from exited threads in the backup buffer)
            drain_access_ring();
            // 2. Reset the current thread's TLS access ring singleton
            tls_access_ring<Key>::flush_all_registered();
            // 3. Drain the backup buffer one final time — a concurrent
            //    thread may have pushed keys between step 1 and now.
            if (tls_access_ring<Key>::has_backup_keys()) {
                auto backup = tls_access_ring<Key>::drain_backup();
                if (!backup.keys.empty()) {
                    auto& bk = backup.keys;
                    if (bk.size() <= 64) {
                        promote_keys(bk);
                    } else {
                        std::sort(bk.begin(), bk.end());
                        bk.erase(std::unique(bk.begin(), bk.end()), bk.end());
                        promote_keys(bk);
                    }
                }
            }
            // 4. Flush pending callbacks (drain TLS callback ring + dispatch)
            if constexpr (requires { mm_.flush_pending_all(); }) {
                mm_.flush_pending_all();
            } else {
                mm_.callbacks().flush_pending();
            }
            // 5. Drain all registered TLS callback rings for cleanup
            tls_callback_ring<Key, Value>::flush_all_registered();

            // P1-12: Final drain of the event tracker's TLS ring so all
            // events recorded by the callback dispatch above are captured
            // before destruction. If the user holds a shared_ptr to the
            // tracker, they can still query the full event history after
            // the cache is destroyed.
            std::shared_ptr<event_tracker<Key, Hash>> tracker;
            {
                std::lock_guard<std::mutex> lk(event_tracker_mutex_);
                tracker = event_tracker_;
            }
            if (tracker) {
                tracker->drain_all_threads();
            }

            // 6. Reclaim any retired items that are no longer protected so
            //    their memory is returned before destruction. We always
            //    invoke try_reclaim here, regardless of whether a slab
            //    allocator is owned — hazptr/EBR retirement is used by all
            //    MM strategies, not only slab-backed ones. Holding
            //    read_handles across cache destruction remains UB.
            detail::hazptr_domain::default_domain().try_reclaim();
            for_each_ebr_domain([](detail::epoch_domain& dom) {
                dom.try_reclaim();
            });

            // T4.3: Debug-only assertion that no handles are outstanding.
            // In release builds without tracking, this is a no-op; in
            // debug builds (or when tracking has been force-enabled), a
            // non-zero active handle count indicates a leak that would
            // cause UB if the cache is destroyed.
            if (tracking_reliable_for_wait()) {
                assert(active_handle_count() == 0 &&
                       "unified_cache destroyed with outstanding read_handles — "
                       "use shutdown_and_wait() / force_wait_handles() first");
            }
        } catch (...) {
            // G23: Swallowing all exceptions silently hides teardown failures.
            // At minimum, log to stderr so operators can diagnose issues.
            std::fprintf(stderr,
                "[lru] WARNING (G23): exception caught during unified_cache "
                "destruction — cache may be in a partially-torn-down state.\n");
            std::fflush(stderr);
        }
    }

    explicit unified_cache(size_type max_size)
        : mm_(max_size)
        , striped_mutex_(make_default_striped_mutex()) {
        if constexpr (Trait::is_compact) {
            compact_storage_.reset(max_size);
        }
        init_production_features();
    }

    /// Two-argument constructor — context-dependent based on is_striped:
    ///   - Striped caches: (max_size, num_stripes) — Task 3 runtime stripe count.
    ///   - Non-striped caches: (max_size, max_memory) — original semantics.
    /// A requires clause disambiguates so only one overload is visible per
    /// instantiation. (template <typename = enable_if_t<...>> does NOT work
    /// here: both constructors would have equivalent template-parameter lists
    /// and be rejected as redeclarations.)
    explicit unified_cache(size_type max_size, size_type num_stripes) requires (is_striped)
        : mm_(max_size, make_sharded_config_for_stripes(num_stripes))
        , striped_mutex_(make_striped_mutex(num_stripes)) {
        if constexpr (Trait::is_compact) {
            compact_storage_.reset(max_size);
        }
        init_production_features();
    }

    /// P2-3 (T3.4): Three-argument constructor for striped caches that
    /// decouples `num_shards` (LRU granularity, memory cost) from
    /// `num_stripes` (legacy stripe-lock count). Only meaningful when
    /// the MM type provides per-shard locks (e.g. sharded_mm_lru); for
    /// non-per-shard-lock MMs, `num_shards` is forced equal to
    /// `num_stripes` inside `make_sharded_config_for_stripes`.
    ///
    /// Use cases:
    ///   - More shards than stripes: finer LRU granularity with the same
    ///     stripe count (per-shard lock provides the actual concurrency).
    ///   - Fewer shards than stripes: saves memory (fewer hash tables)
    ///     while keeping the existing stripe count for compatibility.
    ///   - Equal (default): 1:1 mapping, same as the two-arg constructor.
    explicit unified_cache(size_type max_size, size_type num_stripes, size_type num_shards)
        requires (is_striped)
        : mm_(max_size, make_sharded_config_for_stripes(num_stripes, num_shards))
        , striped_mutex_(make_striped_mutex(num_stripes)) {
        if constexpr (Trait::is_compact) {
            compact_storage_.reset(max_size);
        }
        init_production_features();
    }

    unified_cache(size_type max_size, size_type max_memory) requires (!is_striped)
        : mm_(max_size, max_memory)
        , striped_mutex_(make_default_striped_mutex()) {
        if constexpr (Trait::is_compact) {
            compact_storage_.reset(max_size);
        }
        init_production_features();
    }

    /// Constructor with MM config
    template <typename MMConfig>
        requires (!std::is_arithmetic_v<std::remove_cvref_t<MMConfig>>)
    unified_cache(size_type max_size, const MMConfig& config)
        : mm_(max_size, config)
        , striped_mutex_(make_default_striped_mutex()) {
        init_production_features();
    }

    template <typename MMConfig>
    unified_cache(size_type max_size, size_type max_memory, const MMConfig& config)
        : mm_(max_size, max_memory, config)
        , striped_mutex_(make_default_striped_mutex()) {
        init_production_features();
    }

    // --------------------------------------------------------------------
    // Constructors from initializer list / range
    // --------------------------------------------------------------------

    /// Construct from initializer list.
    unified_cache(std::initializer_list<std::pair<const Key, Value>> init,
                  size_type max_size = unlimited,
                  size_type max_memory = unlimited)
        : mm_(max_size, max_memory)
        , striped_mutex_(make_default_striped_mutex()) {
        for (const auto& [k, v] : init) {
            mm_.set(k, v);
        }
        mm_.callbacks().flush_pending();
        init_production_features();
    }

    /// Construct from an input range [first, last).
    template <typename InputIt>
        requires std::input_iterator<InputIt>
    unified_cache(InputIt first, InputIt last,
                  size_type max_size = unlimited,
                  size_type max_memory = unlimited)
        : mm_(max_size, max_memory)
        , striped_mutex_(make_default_striped_mutex()) {
        for (; first != last; ++first) {
            mm_.set(first->first, first->second);
        }
        mm_.callbacks().flush_pending();
        init_production_features();
    }

    // --------------------------------------------------------------------
    // Task 3: Constructor with explicit num_stripes is integrated above
    // (see unified_cache(size_type, size_type) with is_striped SFINAE).
    // --------------------------------------------------------------------

private:
    /// Helper: construct the default striped_mutex (or empty tuple).
    static striped_mutex_storage make_default_striped_mutex() {
        if constexpr (is_striped) {
            return typename lock_policy::striped_mutex_type(lock_policy::default_num_stripes);
        } else {
            return std::tuple<>{};
        }
    }

    /// Helper: construct a striped_mutex with a custom num_stripes count.
    static striped_mutex_storage make_striped_mutex(size_type num_stripes) {
        if constexpr (is_striped) {
            if (num_stripes == 0) {
                throw std::invalid_argument("unified_cache: num_stripes must be > 0");
            }
            return typename lock_policy::striped_mutex_type(num_stripes);
        } else {
            (void)num_stripes;  // silence unused-parameter warning
            return std::tuple<>{};
        }
    }

    /// Build a sharded_mm_lru_config with num_shards = num_stripes.
    ///
    /// P2-3 (T3.4): When the MM provides per-shard locks
    /// (`has_per_shard_lock_v<mm_type>` is true), `num_shards` can be
    /// set independently of `num_stripes`. The two-arg overload defaults
    /// `num_shards = num_stripes` for backwards compatibility; the
    /// three-arg overload allows explicit decoupling.
    static auto make_sharded_config_for_stripes(size_type num_stripes) {
        return make_sharded_config_for_stripes(num_stripes, num_stripes);
    }

    /// P2-3 (T3.4): Build a sharded_mm_lru_config with decoupled
    /// `num_shards` and `num_stripes`. When the MM does not provide
    /// per-shard locks, `num_shards` is forced equal to `num_stripes`
    /// (preserving the historical safety constraint).
    static auto make_sharded_config_for_stripes(size_type num_stripes,
                                                  size_type num_shards) {
        using mm_t = typename std::decay_t<decltype(mm_)>;
        // For MMs without per-shard locks, num_shards MUST equal
        // num_stripes (otherwise concurrent access via different stripes
        // would corrupt the shard). Force the equality here.
        const size_type effective_num_shards =
            has_per_shard_lock_v<mm_t> ? num_shards : num_stripes;
        if constexpr (requires { typename mm_t::config_type; }) {
            using cfg = typename mm_t::config_type;
            cfg c;
            if constexpr (requires { c.num_shards = effective_num_shards; }) {
                c.num_shards = effective_num_shards;
            }
            return c;
        } else {
            return sharded_mm_lru_config{.num_shards = effective_num_shards};
        }
    }

    /// Task 9/14: Initialize production features based on trait flags.
    /// Called from constructors when auto_start_drain or auto_enable_slab
    /// is set in the trait.
    void init_production_features() {
        // T-P1-1 (R-1): Apply the trait's default fairness mode FIRST, before
        // any background worker (drain / rehash balancer) is started. The
        // debug-only quiescent-state assertion in set_fairness_mode() trips
        // if state_ != 0 (active writer/waiter); once start_event_drain() or
        // start_background_rehash_balancer() spawns a worker that calls
        // promote_keys()/rehash_if_needed() (both acquire write locks), the
        // worker can race with set_fairness_mode() and trip the assert.
        // Setting fairness before spawning workers guarantees state_ == 0 at
        // the call site and ensures workers observe the configured mode from
        // their first iteration. Production traits (production_sharded_lru_trait)
        // and any trait that opts in via `apply_default_fairness = true` get
        // their `default_fairness` applied here. The base cache_trait leaves
        // apply_default_fairness = false so direct
        // unified_cache<cache_trait<...>> instantiations preserve the lock's
        // intrinsic default (writer_fair).
        if constexpr (Trait::apply_default_fairness) {
            if constexpr (requires { this->set_fairness_mode(Trait::default_fairness); }) {
                this->set_fairness_mode(Trait::default_fairness);
            }
        }
        // P0-3 fix: auto-start the event drain worker for ALL thread-safe
        // caches, not just production traits. Without this, retired objects
        // (hazptr/EBR pending lists) and TLS access rings never get drained
        // in safe_cache/striped_cache/read_heavy_* and other non-production
        // thread-safe aliases, causing unbounded memory growth and eventual
        // OOM in long-running services. Single-threaded caches don't need
        // the drain (no concurrent reclamation, no cross-thread TLS flush).
        // Users can still call stop_event_drain() to disable.
        if constexpr (Trait::auto_start_drain || Trait::lock_policy::is_thread_safe) {
            start_event_drain();
        }
        // P1-4: Auto-start the metrics cache worker for production
        // aliases. Under high Prometheus scrape rates (multiple caches
        // scraped every 10-15s), rebuilding stats_snapshot() and
        // prometheus_text() on every call becomes a CPU hotspot. The
        // worker refreshes the cache every 1s (default) so scrape
        // handlers return the cached value in O(1). Aliases that may
        // be used in benchmarks keep auto_start_metrics_cache = false
        // so callers always see fresh metrics.
        //
        // Only start the worker for thread-safe caches — single-threaded
        // caches have no concurrent access, so the O(N) stats rebuild
        // is cheap and a background thread would be pure overhead.
        if constexpr (Trait::auto_start_metrics_cache &&
                      Trait::lock_policy::is_thread_safe) {
            start_metrics_cache_worker(std::chrono::seconds(1));
        }
        // H-4 fix: Configure TLS ring for read-heavy workloads.
        // 1. Set auto_drain_threshold to kRingSize/2 so the ring drains
        //    proactively at 50% capacity, instead of waiting for overflow.
        //    The default (kRingSize) effectively disables auto-drain because
        //    the condition `threshold < cap` is never true.
        // 2. Bind flush_callback to drain_access_ring() so kFlushOnFull
        //    policy actually drains instead of falling back to silent-drop.
        //    Without this, the empty std::function causes all overflow
        //    entries to be silently dropped, losing LRU access traces.
        if constexpr (Trait::lock_policy::is_thread_safe) {
            set_tls_drain_threshold(tls_access_ring<Key>::kRingSize / 2);
            // Capture 'this' — the cache outlives all threads that use it
            // (shutdown() is called in the destructor).
            this->set_tls_flush_callback([this]() {
                this->drain_access_ring();
            });
        }
        // T11.1: auto-enable incremental rehash for segmented/production
        // traits to avoid blocking all writers during hash table expansion.
        if constexpr (Trait::auto_incremental_rehash) {
            if constexpr (requires { this->set_incremental_rehash(true); }) {
                this->set_incremental_rehash(true);
            }
        }
        // R-3: enable defer_promotion on the underlying MM config. The
        // existing set_defer_promotion() helper already handles both
        // striped and non-striped MMs and is a SFINAE-guarded no-op on
        // MM types whose config has no `defer_promotion` field.
        if constexpr (Trait::auto_defer_promotion) {
            if constexpr (requires { this->set_defer_promotion(true); }) {
                this->set_defer_promotion(true);
            }
        }
        // R-3: enable EBR (Epoch-Based Reclamation). Faster read path
        // under sustained read contention than hazptr. SFINAE-guarded:
        // only mm_lru / sharded_mm_lru support EBR; other MM types no-op.
        if constexpr (Trait::auto_enable_ebr) {
            if constexpr (requires { this->set_ebr_domain(&detail::epoch_domain::default_domain()); }) {
                this->set_ebr_domain(&detail::epoch_domain::default_domain());
            }
        }
        // P0-5: auto-start the background rehash balancer for segmented/
        // production traits. The balancer periodically sweeps segments
        // that aren't on the current write hot path, ensuring cold
        // segments also get rehashed. This is a no-op for non-segmented
        // tables (rehash_if_needed() on a non-segmented table is a
        // single call), so the guard is purely a runtime optimization
        // for the segmented case.
        if constexpr (Trait::auto_start_rehash_balancer) {
            if constexpr (requires { this->start_background_rehash_balancer(); }) {
                this->start_background_rehash_balancer();
            }
        }
        // T-G2: Apply a default memory limit for production aliases. Without
        // this, the memory monitor is inactive by default (max_memory_bytes
        // == 0) and unbounded insertions can cause OOM. The default 1 GiB
        // is conservative; operators should tune via set_max_memory().
        // Only applied for thread-safe caches — single-threaded caches are
        // typically used in tests/sandbox where OOM is not a concern.
        if constexpr (Trait::auto_set_default_memory_limit &&
                      Trait::lock_policy::is_thread_safe) {
            memory_monitor_.set_max_memory(Trait::default_max_memory_bytes);
        }
        // T-G3: Auto-enable singleflight for production / read-heavy aliases.
        // Prevents cache-stampede when a hot key expires under high concurrency.
        if constexpr (Trait::auto_enable_singleflight) {
            singleflight_enabled_.store(true, std::memory_order_release);
        }
        // T-G16: Auto-enable slab allocator for production_cache_with_slab.
        // The slab allocator reduces allocation overhead for caches with
        // many small items (10M+). SFINAE-guarded: enable_slab_allocator()
        // requires the MM to support set_allocator() — all MM types do.
        if constexpr (Trait::auto_enable_slab) {
            if constexpr (requires { this->enable_slab_allocator(); }) {
                this->enable_slab_allocator();
            }
        }
        // T-G10: Wire the per-cache handle-release notifier so that
        // shutdown_and_wait() can sleep on a condition_variable instead of
        // busy-polling. read_handle::release() reaches the notifier through
        // per_cache_stats_->release_notifier and calls notify_all() when the
        // last handle is dropped during shutdown.
        per_cache_stats_.release_notifier = &active_handle_notifier_;
    }

public:

    // --------------------------------------------------------------------
    // Core API - dispatches to MM with optional locking
    // --------------------------------------------------------------------

    // --------------------------------------------------------------------
    // Value provider
    // --------------------------------------------------------------------

    /// Set a value provider for automatic fetch-on-miss.
    /// When get_or_fetch() or operator[] encounters a cache miss,
    /// this function is called to generate the value, which is then stored.
    void set_value_provider(std::function<Value(const Key&)> provider) {
        auto lock = acquire_write_lock();
        value_provider_ = std::move(provider);
    }

    // --------------------------------------------------------------------
    // O2: Slow query logging (threshold-based)
    // --------------------------------------------------------------------

    /// Set the slow operation threshold. Operations taking longer than
    /// this are reported via the registered slow_query_callback (if any).
    /// Pass 0 to disable (default). Reasonable values: 1ms-100ms.
    void set_slow_query_threshold(std::chrono::nanoseconds threshold) noexcept {
        slow_query_threshold_ns_.store(
            static_cast<std::uint64_t>(threshold.count()),
            std::memory_order_release);
    }

    /// Query the current slow operation threshold.
    std::chrono::nanoseconds slow_query_threshold() const noexcept {
        return std::chrono::nanoseconds(
            slow_query_threshold_ns_.load(std::memory_order_acquire));
    }

    /// Register a slow query callback. Pass an empty function to clear.
    /// The callback is invoked on the calling thread (NOT the cache
    /// background worker) so it should be fast and non-blocking. For
    /// expensive logging, queue to a separate thread inside the callback.
    void set_slow_query_callback(slow_query_callback_type callback) {
        // RCU-style swap: publish a new shared_ptr atomically. Concurrent
        // readers load the old pointer and finish their invocation safely
        // before the old shared_ptr is destroyed.
        std::shared_ptr<slow_query_callback_type> new_cb =
            std::make_shared<slow_query_callback_type>(std::move(callback));
        // No need for the write lock — the swap is atomic. The write lock
        // would serialize against set()/get() which we explicitly do not want.
        std::shared_ptr<slow_query_callback_type> old_cb =
            slow_query_callback_.exchange(new_cb, std::memory_order_acq_rel);
        (void)old_cb;  // released when last concurrent invoker finishes
    }

    /// O2: Total count of slow operations detected since startup.
    std::size_t slow_query_count() const noexcept {
        return slow_query_count_.load(std::memory_order_acquire);
    }

    /// O2: Notify a slow operation. Called from get()/set() after
    /// measuring latency. Fast path: if threshold is 0 or no callback
    /// is registered, this is a single relaxed load + branch.
    void notify_slow_query(slow_op_kind kind, const Key& key,
                           std::uint64_t latency_ns) const noexcept {
        const std::uint64_t threshold =
            slow_query_threshold_ns_.load(std::memory_order_relaxed);
        if (threshold == 0 || latency_ns < threshold) return;
        // Load the callback shared_ptr (RCU). The exchange in
        // set_slow_query_callback guarantees this load sees either
        // the old or the new shared_ptr, both valid.
        auto cb_ptr = slow_query_callback_.load(std::memory_order_acquire);
        if (!cb_ptr || !*cb_ptr) return;
        slow_query_count_.fetch_add(1, std::memory_order_relaxed);
        try {
            (*cb_ptr)(kind, key, latency_ns);
        } catch (...) {
            // Swallow — slow query logging must not propagate.
        }
    }

    // --------------------------------------------------------------------
    // O1: Distributed tracing callback (OpenTelemetry / Jaeger / Zipkin)
    // --------------------------------------------------------------------

    /// Register a trace callback. Pass an empty function to clear.
    /// The callback is invoked on EVERY get()/set() (unlike slow_query
    /// which is threshold-gated), so the callback itself should implement
    /// sampling if needed (e.g., random 1% sampling for production).
    ///
    /// Thread-safe: uses RCU-style atomic shared_ptr swap. Concurrent
    /// readers finish their invocation on the old callback safely before
    /// it's destroyed.
    void set_trace_callback(trace_callback_type callback) {
        std::shared_ptr<trace_callback_type> new_cb =
            std::make_shared<trace_callback_type>(std::move(callback));
        std::shared_ptr<trace_callback_type> old_cb =
            trace_callback_.exchange(new_cb, std::memory_order_acq_rel);
        (void)old_cb;  // released when last concurrent invoker finishes
    }

    /// O1: Total count of trace events fired since startup. When non-zero
    /// with a registered callback, confirms tracing is active.
    std::size_t trace_count() const noexcept {
        return trace_count_.load(std::memory_order_acquire);
    }

    /// O1: Notify a trace event. Called from ~trace_notifier() at the end
    /// of get()/set(). Fast path: if no callback is registered, this is
    /// a single atomic load + branch (no clock reads, no allocations).
    void notify_trace(trace_op_kind op, const Key& key,
                      std::uint64_t latency_ns,
                      bool hit, bool error) const noexcept {
        auto cb_ptr = trace_callback_.load(std::memory_order_acquire);
        if (!cb_ptr || !*cb_ptr) return;
        trace_count_.fetch_add(1, std::memory_order_relaxed);
        try {
            (*cb_ptr)(op, key, latency_ns, hit, error);
        } catch (...) {
            // Swallow — tracing must not propagate exceptions.
        }
    }

    // --------------------------------------------------------------------
    // Runtime defer_promotion control (Task 1)
    // --------------------------------------------------------------------

    /// Enable or disable defer_promotion at runtime.
    /// When enabled, get() always defers LRU promotion to the TLS access ring
    /// instead of attempting a non-blocking write lock for immediate promotion.
    /// This reduces lock pressure on the read path at the cost of slightly
    /// delayed LRU ordering.
    ///
    /// For striped caches (sharded_mm_lru), iterates all shards and updates
    /// each shard's config. For non-striped caches, updates the single MM
    /// config. Uses set_config() to ensure any side effects (e.g., refresh
    /// time recalculation) are applied.
    void set_defer_promotion(bool enabled) {
        auto lock = acquire_write_lock();
        if constexpr (is_striped) {
            for (std::size_t i = 0; i < mm_.num_shards(); ++i) {
                auto cfg = mm_.shard(i).config();
                cfg.defer_promotion = enabled;
                mm_.shard(i).set_config(cfg);
            }
        } else if constexpr (requires { mm_.config(); }) {
            auto cfg = mm_.config();
            cfg.defer_promotion = enabled;
            mm_.set_config(cfg);
        }
        // For MM types without defer_promotion support, this is a no-op.
    }

    /// Query the current defer_promotion setting.
    /// For striped caches, returns the value from the first shard.
    bool is_defer_promotion_enabled() const {
        if constexpr (is_striped) {
            if constexpr (requires { mm_.shard(0).config().defer_promotion; }) {
                return mm_.shard(0).config().defer_promotion;
            }
            return false;
        } else if constexpr (requires { mm_.config().defer_promotion; }) {
            return mm_.config().defer_promotion;
        } else {
            return false;
        }
    }

    // --------------------------------------------------------------------
    // TTL jitter — prevent thundering-herd avalanches on bulk TTL expiry
    // --------------------------------------------------------------------
    //
    // When many keys are inserted with identical TTL (e.g., bulk prewarm,
    // scheduled refresh), they expire simultaneously → downstream stampede.
    // TTL jitter applies ±jitter_pct randomization to each set_with_ttl()
    // call, spreading expiry across a window. Default: 10% (±10%).
    //
    // Examples (with default jitter_pct = 0.10):
    //   set_with_ttl(k, v, 100s)  →  actual TTL in [90s, 110s]
    //   set_with_ttl(k, v, 1s)    →  actual TTL in [900ms, 1.1s]
    //
    // Set jitter_pct to 0.0 to disable. The PRNG is thread-local xorshift64
    // (~5ns per call, no locks) — see ttl_entry::apply_jitter for details.
    void set_ttl_jitter_pct(double pct) {
        if (pct < 0.0) pct = 0.0;
        if (pct > 1.0) pct = 1.0;
        ttl_jitter_pct_ = pct;
    }
    double ttl_jitter_pct() const noexcept { return ttl_jitter_pct_; }
    void set_ttl_jitter_enabled(bool enabled) {
        ttl_jitter_enabled_.store(enabled, std::memory_order_release);
    }
    bool ttl_jitter_enabled() const noexcept {
        return ttl_jitter_enabled_.load(std::memory_order_acquire);
    }

    /// P1-1 (T2.2): Drain the current thread's TLS access ring before
    /// entering a write path that may trigger eviction.
    ///
    /// Problem: when defer_promotion is enabled, get() hits do not
    /// immediately promote the key in the LRU list — accesses are
    /// batched in a thread-local ring and drained later. If eviction
    /// runs while the calling thread has pending promotions in its TLS
    /// ring, find_eviction_victim() walks a stale LRU list and may
    /// select a key that was just accessed (and would have been
    /// promoted to MRU had the ring been drained). The wrong item
    /// gets evicted.
    ///
    /// Fix: drain the TLS ring *before* acquiring the write lock on
    /// the eviction-triggering path. drain_access_ring() uses
    /// try_acquire_write_lock_for_key() (non-blocking) inside
    /// promote_keys(), so it cannot deadlock with the upcoming write
    /// lock acquisition — it simply skips keys whose stripe lock is
    /// contended and re-records them for the next drain.
    ///
    /// Cost: the drain is conditional on (a) defer_promotion being
    /// enabled and (b) the current thread's TLS ring being non-empty.
    /// On the fast path (no pending promotions), this is a single
    /// atomic load + branch. The actual drain runs only when there
    /// is pending work, and its cost is amortized across all keys in
    /// the ring (drain promotes a batch, not one key at a time).
    void maybe_drain_tls_ring_pre_evict() {
        if (!is_defer_promotion_enabled()) return;
        // Fast path: TLS ring empty AND no backup keys — no work to do.
        if (tls_access_ring<Key>::instance().empty() &&
            !tls_access_ring<Key>::has_backup_keys()) return;
        // H-5 fix: drain ALL threads' pending promotions, not just the
        // calling thread's. Previously, only the calling thread's ring was
        // drained, so keys recently accessed by other threads remained
        // stale in their TLS rings and could be wrongly selected for
        // eviction. drain_all_threads() also sets needs_flush_ on other
        // threads so they drain themselves on their next record_access().
        auto drained = tls_access_ring<Key>::drain_all_threads();
        if (!drained.keys.empty()) {
            auto& keys = drained.keys;
            if (keys.size() <= 64) {
                promote_keys(keys);
            } else {
                std::sort(keys.begin(), keys.end());
                keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
                promote_keys(keys);
            }
        }
        // Also force-flush dormant threads (blocked/idle threads whose
        // rings can't be drained via needs_flush_ because they won't
        // call record_access() soon). This uses seqlock-style validation
        // for safe cross-thread reads.
        tls_access_ring<Key>::force_flush_dormant_threads();
    }

    /// T-D2 (P2-2): Record an access in this cache's TLS ring, with
    /// this cache's `tls_ring_config_` activated as the active config
    /// for the duration of the call.
    ///
    /// This is the per-cache wrapper around
    /// `tls_access_ring<Key>::instance().record_access(key)`. All
    /// `get()` / `peek()` / `try_get()` / `bulk_get()` etc. paths in
    /// `unified_cache` route through this helper so the per-cache
    /// overflow policy, auto-drain threshold, and flush callback
    /// (set via `set_tls_ring_full_policy()` /
    /// `set_tls_drain_threshold()` / `set_tls_flush_callback()`)
    /// take effect automatically.
    ///
    /// Cost: one stack-allocated RAII guard (`active_config_scope`,
    /// 2 pointer assignments + 1 destructor restore) on every call.
    /// This is negligible compared to the cost of `record_access()`
    /// itself (which writes to a TLS ring buffer slot + checks
    /// overflow/auto-drain thresholds). The guard is movable to a
    /// register pair and has no heap allocation.
    ///
    /// Reentrancy: `active_config_scope` saves the previous active
    /// config and restores it on destruction, so nested calls (e.g.,
    /// a flush callback that itself calls `record_access` on the same
    /// or a different cache) are safe — the inner call sees the inner
    /// cache's config, the outer call restores the outer cache's
    /// config when the inner returns.
    void record_access_in_ring(const Key& key) {
        typename tls_access_ring<Key>::active_config_scope scope(
            &tls_ring_config_);
        tls_access_ring<Key>::instance().record_access(key);
    }

    // --------------------------------------------------------------------
    // Runtime fairness mode control (Task 2)
    // --------------------------------------------------------------------

    /// Set the fairness mode for the underlying mutex(es).
    /// For striped caches, applies to all stripes.
    /// For non-striped thread-safe caches, applies to the global mutex.
    /// For single-threaded caches, this is a no-op (no mutex present).
    ///
    /// G14: For striped caches, this now delegates to
    /// set_fairness_mode_quiescent() (1s timeout) to avoid racing with
    /// threads waiting on stripe mutexes. If the quiescent wait times out,
    /// it falls back to a direct (non-quiescent) switch. Callers that need
    /// a guaranteed quiescent switch should use set_fairness_mode_quiescent()
    /// directly with a longer timeout.
    void set_fairness_mode(detail::fairness_mode mode) {
        if constexpr (is_striped) {
            // G14: Prefer quiescent switch to avoid racing with waiters.
            if (!set_fairness_mode_quiescent(mode, std::chrono::seconds(1))) {
                // Fallback: quiescent wait timed out — direct switch.
                if constexpr (requires { striped_mutex_.set_fairness_mode(mode); }) {
                    striped_mutex_.set_fairness_mode(mode);
                }
            }
        } else if constexpr (is_thread_safe) {
            // T-P3: Use set_fairness_mode_locked() because we hold the
            // write lock (acquired below) — the mutex's plain
            // set_fairness_mode() has a debug-only assertion that
            // state_ == 0, but acquiring the write lock sets
            // state_ = kWriterFlag. Holding the write lock IS the safe
            // condition for a quiescent switch (no other thread can be
            // in a critical section), so set_fairness_mode_locked()
            // (which skips the assertion) is the correct API.
            if constexpr (requires { mutex_.set_fairness_mode(mode); }) {
                auto lock = acquire_write_lock();
                if constexpr (requires { mutex_.set_fairness_mode_locked(mode); }) {
                    mutex_.set_fairness_mode_locked(mode);
                } else {
                    mutex_.set_fairness_mode(mode);
                }
            }
        }
        // single_threaded_policy: no mutex, no-op
    }

    /// P1-5 (T1.5): Quiescent variant of set_fairness_mode(). Drains all
    /// in-flight readers/writers before atomically switching the fairness
    /// mode, guaranteeing no in-flight operation observes a mode change
    /// mid-critical-section. For striped caches, all stripes are
    /// exclusively locked before the switch (so the switch is atomic
    /// across the entire cache).
    ///
    /// @param mode     New fairness mode.
    /// @param timeout  Maximum time to wait for quiescence. Default 5s.
    /// @return true if switched; false on timeout (mode unchanged).
    ///
    /// Cost: briefly blocks new readers/writers while waiting for
    /// in-flight operations to drain. Call during planned maintenance
    /// windows or low-traffic periods. For read-heavy workloads with
    /// long read sections, prefer a longer timeout.
    bool set_fairness_mode_quiescent(
            detail::fairness_mode mode,
            std::chrono::milliseconds timeout = std::chrono::seconds(5)) {
        if constexpr (is_striped) {
            if constexpr (requires { striped_mutex_.set_fairness_mode_quiescent(mode, timeout); }) {
                return striped_mutex_.set_fairness_mode_quiescent(mode, timeout);
            }
            return false;
        } else if constexpr (is_thread_safe) {
            if constexpr (requires { mutex_.set_fairness_mode_quiescent(mode, timeout); }) {
                return mutex_.set_fairness_mode_quiescent(mode, timeout);
            }
            return false;
        } else {
            // single_threaded_policy: no mutex, no real switch needed.
            return true;
        }
    }

    /// Query the current fairness mode.
    /// P1-2: Returns writer_fair (the actual default) for single-threaded
    /// caches and for mutexes that do not support fairness_mode. The
    /// previous fallback returned reader_preferred, which contradicted
    /// the documented default and confused operators reading
    /// diagnostics (fairness_mode_str would show "reader_preferred"
    /// for single_threaded caches even though no actual mutex exists).
    detail::fairness_mode get_fairness_mode() const noexcept {
        if constexpr (is_striped) {
            if constexpr (requires { striped_mutex_.get_fairness_mode(); }) {
                return striped_mutex_.get_fairness_mode();
            }
            return detail::fairness_mode::writer_fair;
        } else if constexpr (is_thread_safe) {
            if constexpr (requires { mutex_.get_fairness_mode(); }) {
                return mutex_.get_fairness_mode();
            }
            return detail::fairness_mode::writer_fair;
        } else {
            // single_threaded_policy: no mutex, return the documented
            // default so diagnostics are consistent.
            return detail::fairness_mode::writer_fair;
        }
    }

    // ----------------------------------------------------------------
    // P1-1: Writer starvation detector forwarding
    // ----------------------------------------------------------------
    //
    // In reader_preferred mode, writers can starve indefinitely under
    // sustained read load. The detector (in distributed_shared_mutex)
    // records when a writer first queues and, if it has been waiting
    // longer than the configured timeout, redirects new readers to the
    // writer_fair slow path. These methods forward to the underlying
    // mutex(es) so operators can configure the timeout and monitor
    // starvation events from the unified_cache API.

    /// P1-1: Set the writer starvation timeout (nanoseconds). 0 disables
    /// detection (pure reader_preferred — writers can starve). Default
    /// is 100ms. Only effective in reader_preferred mode. For striped
    /// caches, applies to all stripes.
    void set_writer_starvation_timeout(uint64_t timeout_ns) noexcept {
        if constexpr (is_striped) {
            if constexpr (requires { striped_mutex_.set_writer_starvation_timeout(timeout_ns); }) {
                striped_mutex_.set_writer_starvation_timeout(timeout_ns);
            }
        } else if constexpr (is_thread_safe) {
            if constexpr (requires { mutex_.set_writer_starvation_timeout(timeout_ns); }) {
                mutex_.set_writer_starvation_timeout(timeout_ns);
            }
        }
    }

    /// P1-1: Number of times a reader was redirected to the writer_fair
    /// slow path due to writer starvation. Aggregated across all stripes
    /// for striped caches. Returns 0 for single-threaded caches.
    std::size_t writer_starvation_events() const noexcept {
        if constexpr (is_striped) {
            if constexpr (requires { striped_mutex_.writer_starvation_events(); }) {
                return striped_mutex_.writer_starvation_events();
            }
            return 0;
        } else if constexpr (is_thread_safe) {
            if constexpr (requires { mutex_.writer_starvation_events(); }) {
                return mutex_.writer_starvation_events();
            }
            return 0;
        } else {
            return 0;
        }
    }

    /// P1-1: Maximum observed writer wait time (ns) across all stripes.
    /// Returns 0 for single-threaded caches. Reset via
    /// reset_writer_max_wait_ns().
    uint64_t writer_max_wait_ns() const noexcept {
        if constexpr (is_striped) {
            if constexpr (requires { striped_mutex_.writer_max_wait_ns(); }) {
                return striped_mutex_.writer_max_wait_ns();
            }
            return 0;
        } else if constexpr (is_thread_safe) {
            if constexpr (requires { mutex_.writer_max_wait_ns(); }) {
                return mutex_.writer_max_wait_ns();
            }
            return 0;
        } else {
            return 0;
        }
    }

    /// P1-1: Reset the maximum observed writer wait time to 0. Useful
    /// for benchmark baselines or after a transient stall.
    void reset_writer_max_wait_ns() noexcept {
        if constexpr (is_striped) {
            if constexpr (requires { striped_mutex_.reset_writer_max_wait_ns(); }) {
                striped_mutex_.reset_writer_max_wait_ns();
            }
        } else if constexpr (is_thread_safe) {
            if constexpr (requires { mutex_.reset_writer_max_wait_ns(); }) {
                mutex_.reset_writer_max_wait_ns();
            }
        }
    }

    // ----------------------------------------------------------------
    // T-B2 (P0-1-补): NUMA-aware reader counter routing
    // ----------------------------------------------------------------
    //
    // On multi-socket NUMA hardware (2+ sockets), reader counter
    // modifications on a single shared cache line cause inter-socket
    // cache coherence traffic. When `set_numa_aware(true)` is called,
    // reader counter slots are selected based on the calling thread's
    // current NUMA node (via getcpu(2) on Linux / GetNumaProcessorNode
    // on Windows), keeping modifications within a single socket's L3
    // cache.
    //
    // Default: off. On single-socket systems (the common case), enabling
    // NUMA-aware routing adds a syscall per first-touch per thread with
    // no benefit — keep it off. On 4+ socket NUMA hardware, the savings
    // from avoiding cross-socket MSI traffic can be 5-15% of read
    // throughput under heavy contention.
    //
    // Forwarded to the underlying distributed_shared_mutex (single or
    // striped). For single_threaded_policy this is a no-op (no mutex).

    /// Enable or disable NUMA-aware reader counter routing at runtime.
    /// For striped caches, applies to all stripes. For single-threaded
    /// caches, this is a no-op.
    void set_numa_aware(bool enabled) noexcept {
        if constexpr (is_striped) {
            if constexpr (requires { striped_mutex_.set_numa_aware(enabled); }) {
                striped_mutex_.set_numa_aware(enabled);
            }
        } else if constexpr (is_thread_safe) {
            if constexpr (requires { mutex_.set_numa_aware(enabled); }) {
                mutex_.set_numa_aware(enabled);
            }
        }
        // single_threaded_policy: no mutex, no-op
    }

    /// Query whether NUMA-aware reader counter routing is enabled.
    bool numa_aware() const noexcept {
        if constexpr (is_striped) {
            if constexpr (requires { striped_mutex_.numa_aware(); }) {
                return striped_mutex_.numa_aware();
            }
            return false;
        } else if constexpr (is_thread_safe) {
            if constexpr (requires { mutex_.numa_aware(); }) {
                return mutex_.numa_aware();
            }
            return false;
        } else {
            return false;
        }
    }

    /// Return the number of NUMA nodes detected on the system.
    /// Returns 1 on single-socket systems and on platforms without
    /// NUMA detection (macOS). Used by operators to validate that
    /// `set_numa_aware(true)` will actually route across multiple
    /// nodes — if this returns 1, NUMA-aware routing is a no-op.
    std::size_t num_numa_nodes() const noexcept {
        if constexpr (is_striped) {
            if constexpr (requires { striped_mutex_.num_numa_nodes(); }) {
                return striped_mutex_.num_numa_nodes();
            }
            return 1;
        } else if constexpr (is_thread_safe) {
            if constexpr (requires { mutex_.num_numa_nodes(); }) {
                return mutex_.num_numa_nodes();
            }
            return 1;
        } else {
            return 1;
        }
    }

    // ----------------------------------------------------------------
    // T-D2 (P2-2): Per-cache TLS ring config
    // ----------------------------------------------------------------
    //
    // The TLS access ring (`tls_access_ring<Key>`) historically uses
    // static members (`full_policy_`, `auto_drain_threshold_`,
    // `flush_callback_()`) shared across all cache instances of the
    // same `<Key, N>` specialization. This is fine when the process
    // hosts a single cache, but in multi-tenant deployments (multiple
    // cache instances with different workloads in one process) the
    // static-wide settings force a one-size-fits-all compromise.
    //
    // T-D2 introduces per-cache config: each `unified_cache` instance
    // owns a `tls_ring_config` that overrides the static defaults when
    // active. The cache activates its config around `record_access()`
    // calls via `active_config_scope` RAII, so callers don't need to
    // manage thread-local state themselves — every `get()` / `peek()` /
    // `try_get()` etc. on this cache automatically uses this cache's
    // config for overflow policy, drain threshold, and flush callback.
    //
    // Backward compat: by default `tls_ring_config_` matches the static
    // defaults (kFlushOnFull / N / null callback), so existing behavior
    // is unchanged unless callers explicitly customize via the setters
    // below. Calling the static `tls_access_ring<Key>::set_full_policy()`
    // etc. still works and is honored when no per-cache config is set
    // — but per-cache config takes precedence on caches that have been
    // explicitly customized.
    //
    // Tuning examples:
    //   cache1.set_tls_ring_full_policy(tls_ring_full_policy::kSilentDrop);
    //   cache1.set_tls_drain_threshold(64);  // small hot cache
    //
    //   cache2.set_tls_ring_full_policy(tls_ring_full_policy::kFlushOnFull);
    //   cache2.set_tls_drain_threshold(128); // larger, more bursty
    //   cache2.set_tls_flush_callback([&]{ cache2.drain_access_ring(); });

    /// Set the overflow policy for this cache instance's TLS ring.
    /// Overrides the static `tls_access_ring<Key>::set_full_policy()`
    /// for `record_access()` calls routed through this cache.
    void set_tls_ring_full_policy(tls_ring_full_policy policy) noexcept {
        tls_ring_config_.full_policy.store(policy, std::memory_order_relaxed);
    }

    /// Query this cache's overflow policy. Note: this returns the
    /// per-cache config value, which is only effective when this
    /// cache is the active config (auto-managed via RAII on every
    /// `get` / `peek` / `try_get` etc. call).
    tls_ring_full_policy tls_ring_full_policy_for_cache() const noexcept {
        return tls_ring_config_.full_policy.load(std::memory_order_relaxed);
    }

    /// Set the auto-drain threshold for this cache instance's TLS ring.
    /// When the ring's occupancy reaches `threshold`, the next
    /// `record_access()` call will synchronously drain. Overrides the
    /// static `tls_access_ring<Key>::set_tls_drain_threshold()`.
    ///
    /// threshold == 0 or > kRingSize disables auto-drain for this cache.
    void set_tls_drain_threshold(std::size_t threshold) noexcept {
        if (threshold == 0 || threshold > tls_access_ring<Key>::kRingSize) {
            threshold = tls_access_ring<Key>::kRingSize;
        }
        tls_ring_config_.auto_drain_threshold.store(
            threshold, std::memory_order_relaxed);
    }

    /// Query this cache's auto-drain threshold.
    std::size_t tls_drain_threshold_for_cache() const noexcept {
        return tls_ring_config_.auto_drain_threshold.load(
            std::memory_order_relaxed);
    }

    /// Set a per-cache flush callback invoked when the ring overflows
    /// under `kFlushOnFull` policy. Typical use: drain this cache's
    /// access ring to capture pending promotions before silent-drop
    /// kicks in. The callback runs on the calling thread, so it must
    /// be non-blocking or risk stalling the access path.
    void set_tls_flush_callback(std::function<void()> cb) {
        tls_ring_config_.flush_callback = std::move(cb);
    }

    /// Access the per-cache config struct (for advanced tuning or
    /// for passing to `tls_access_ring<Key>::active_config_scope` in
    /// custom scenarios outside the standard get/peek path).
    typename tls_access_ring<Key>::tls_ring_config&
    tls_ring_config() noexcept { return tls_ring_config_; }

    /// Const overload of `tls_ring_config()`.
    const typename tls_access_ring<Key>::tls_ring_config&
    tls_ring_config() const noexcept { return tls_ring_config_; }

    // ----------------------------------------------------------------
    // T-B1 (P0-2-补): synchronize_epoch() + epoch_stale_count
    // ----------------------------------------------------------------
    //
    // Forwards to the default global epoch_domain (used by EBR). This
    // is a quiescent-point API: blocks until all threads that were in
    // a critical section at call time have exited, then returns true.
    // Returns false on timeout.
    //
    // Use cases:
    //   - Shutdown: ensures all in-flight CSes complete before
    //     destruction (alternative to graceful shutdown).
    //   - Bulk evict: ensures retired objects are reclaimable before
    //     OOM-protecting eviction.
    //   - Tests: deterministic quiescent points for assertions.
    //
    // Cost: O(N_slots) per predicate check; under normal load completes
    // in <10ms because CSes are short.

    /// Block until all in-flight EBR critical sections have exited and
    /// at least one epoch advance has occurred. Returns true on
    /// quiescent-point success, false on timeout.
    bool synchronize_epoch(
            std::chrono::milliseconds timeout = std::chrono::seconds(5)) {
        // R9: synchronize every active EBR domain (global or per-shard).
        // All domains must reach a quiescent point for a full drain.
        bool all_ok = true;
        for_each_ebr_domain([&](detail::epoch_domain& dom) {
            if (!dom.synchronize_epoch(timeout)) all_ok = false;
        });
        return all_ok;
    }

    /// T-B1 (P0-2-补): Cumulative count of EBR slots that were skipped
    /// during reclaim because they were stuck in a critical section
    /// (i.e., the thread hadn't exited the CS by the reclaim pass).
    /// Fast-growing count indicates persistent blockage; slow-growing
    /// count indicates brief stalls. Combined with reclaim_pending_count,
    /// operators can distinguish "stuck thread" from "slow draining".
    std::size_t epoch_stale_count() const noexcept {
        std::size_t total = 0;
        for_each_ebr_domain([&](detail::epoch_domain& dom) {
            total += dom.epoch_stale_count();
        });
        return total;
    }

    /// T-A1 (P0-6): Current EBR slot capacity (allocated batches ×
    /// kBatchSize). Capacity grows on demand from 128 → 8192 as
    /// threads enter critical sections concurrently. Useful for sizing
    /// the workload against the upper bound.
    std::size_t epoch_slot_capacity() const noexcept {
        std::size_t total = 0;
        for_each_ebr_domain([&](detail::epoch_domain& dom) {
            total += dom.slot_capacity();
        });
        return total;
    }

    /// T-A1 (P0-6): Number of times EBR acquire_slot() exhausted all
    /// 8192 slots and had to yield. Non-zero under normal operation
    /// indicates 8000+ simultaneous critical sections — likely a
    /// runaway thread-spawn pattern rather than a normal cache workload.
    std::size_t epoch_slot_exhaustion_count() const noexcept {
        std::size_t total = 0;
        for_each_ebr_domain([&](detail::epoch_domain& dom) {
            total += dom.slot_exhaustion_count();
        });
        return total;
    }

    // ----------------------------------------------------------------
    // T-A2 (P0-7): hazptr slot exhaustion metrics
    // ----------------------------------------------------------------
    //
    // Forwarded to the default global hazptr_domain. Like EBR metrics
    // above, these surface the reclamation subsystem's health so
    // operators can detect runaway handle counts without digging into
    // the hazptr internals.

    /// T-A2 (P0-7): Number of times hazptr acquire_slot() exhausted all
    /// 8192 slots and had to yield. Non-zero indicates the workload has
    /// 8000+ simultaneous live hazard pointers — likely a runaway
    /// reader-heavy workload or a thread/fiber leak. Previously this
    /// condition caused acquire_slot() to spin forever; now it yields
    /// and bumps this counter so operators can detect the failure.
    std::size_t hazptr_slot_exhaustion_count() const noexcept {
        return detail::hazptr_domain::default_domain().slot_exhaustion_count();
    }

    /// T-A2 (P0-7): Current hazptr slot capacity (allocated batches ×
    /// kBatchSize). Capacity grows on demand from 128 → 8192 as
    /// threads acquire hazard pointers concurrently.
    std::size_t hazptr_slot_capacity() const noexcept {
        return detail::hazptr_domain::default_domain().capacity();
    }

    /// P0-3: Maximum slot count the hazptr domain can grow to. Defaults
    /// to 8192 (kMaxBatches × kBatchSize) and can be raised up to 65536
    /// via `set_hazptr_max_slots()`.
    std::size_t hazptr_max_slot_count() const noexcept {
        return detail::hazptr_domain::default_domain().max_slot_count();
    }

    /// P0-3: Currently-acquired (live) hazard pointer slots. When this
    /// approaches `hazptr_max_slot_count()`, operators should either
    /// call `set_hazptr_max_slots()` to expand capacity or investigate
    /// the cause of handle retention (e.g. long-lived read_handle).
    std::size_t hazptr_active_slot_count() const noexcept {
        return detail::hazptr_domain::default_domain().active_slot_count();
    }

    /// P0-3: Current slot usage ratio (0.0 .. 1.0). When this exceeds
    /// 0.9, operators should either raise the limit via
    /// `set_hazptr_max_slots()` or reduce reader fan-out — sustained
    /// high usage risks `acquire_slot()` returning `npos`, which
    /// degrades reads to cache misses (via empty read_handle).
    float hazptr_slot_usage_ratio() const noexcept {
        return detail::hazptr_domain::default_domain().slot_usage_ratio();
    }

    /// P0-3: Runtime capacity expansion. Raises the maximum number of
    /// hazptr slots from the default 8192 up to 65536 (512 batches ×
    /// 128 slots). Existing slots and active handles are unaffected —
    /// only the upper bound is raised, allowing `acquire_slot()` to
    /// allocate new batches on demand up to the new limit. Call this
    /// at startup (or during a quiescent period) when the workload is
    /// known to require more than 8192 concurrent live read_handle
    /// objects.
    ///
    /// @param n New maximum slot count. Silently clamped to
    ///          `[8192, 65536]`.
    void set_hazptr_max_slots(std::size_t n) noexcept {
        detail::hazptr_domain::default_domain().set_max_slots(n);
    }

    /// L-1: Number of times hazptr acquire_slot() exceeded the spin
    /// budget (kMaxSpinRetries) after slot exhaustion and fell back to
    /// a synchronous try_reclaim() to drain pending retired objects.
    /// Sustained non-zero growth means the workload has 8192+ live
    /// hazard pointers for extended periods — operators should either
    /// increase the cache's max_size (so fewer evictions create fewer
    /// concurrent handles) or reduce the reader fan-out. If this
    /// counter exceeds kMaxSyncFallbacks * (number of threads),
    /// acquire_slot() will start throwing std::runtime_error so the
    /// caller can degrade gracefully instead of deadlocking silently.
    /// Surfaced through diagnostics_info / prometheus_text() so
    /// operators can alert on sustained fallback growth without
    /// polling the hazptr domain directly.
    std::size_t hazptr_sync_fallback_count() const noexcept {
        return detail::hazptr_domain::default_domain().hazptr_sync_fallback_count();
    }

    // --------------------------------------------------------------------
    // P-HIGH-3 (T-H2): Unified pre-write validation
    // --------------------------------------------------------------------
    //
    // Every write-path function (set / set_with_ttl / add / replace /
    // get_or_fetch / set_prehashed) calls pre_write_check() at entry to
    // enforce consistent shutdown and OOM-rejection behavior. Previously
    // only set() and set_prehashed() checked both conditions; set_with_ttl,
    // add, replace, and get_or_fetch were missing the is_memory_critical()
    // guard (and add/replace/get_or_fetch were missing the shutdown guard
    // too), allowing insertions after shutdown and OOM condition to slip
    // through.
    //
    // Write paths that acquire a write lock must ALSO re-check is_shutdown()
    // inside the lock (TOCTOU: the cache may be shut down between the
    // pre_check and lock acquisition). This is done via
    // recheck_shutdown_in_lock().
    enum class pre_write_result { kProceed, kRejectInsert };

    /// Pre-write validation. Call at the entry of every write-path function.
    /// - Throws std::runtime_error if the cache is shut down.
    /// - Returns kRejectInsert if the cache is in critical memory mode
    ///   (caller should silently reject the insertion / return false).
    /// - Returns kProceed otherwise.
    pre_write_result pre_write_check(const char* fn_name) const {
        if (is_shutdown()) {
            throw cache_closed_exception(
                std::string("unified_cache::") + fn_name +
                ": cache is shut down");
        }
        if (is_memory_critical()) {
            return pre_write_result::kRejectInsert;
        }
        return pre_write_result::kProceed;
    }

    /// P-HIGH-3 (T-H2): TOCTOU re-check inside the write lock. Call after
    /// acquiring the write lock, before mutating mm_. Returns true if the
    /// cache was shut down between pre_write_check() and lock acquisition.
    /// The caller should release the lock and return without mutating.
    bool recheck_shutdown_in_lock() const noexcept {
        return is_shutdown();
    }

    template <typename V>
    void set(const Key& key, V&& value) {
        // P-HIGH-3 (T-H2): Unified pre-write validation.
        if (pre_write_check("set") == pre_write_result::kRejectInsert) {
            return;
        }
        // T14.1: Compact path — delegate to embedded compact_cache.
        if constexpr (Trait::is_compact) {
            compact().set(key, std::forward<V>(value));
            return;
        }
        // P1-1 (T2.2): Drain TLS access ring before acquiring the write
        // lock. mm_.set() may trigger evict_lru() internally; without
        // this drain, find_eviction_victim() could pick a key that was
        // just accessed on this thread but whose promotion is still
        // pending in the TLS ring.
        maybe_drain_tls_ring_pre_evict();
        // Task B: compute shard_idx once; reuse across latency timer + body.
        const std::size_t shard_idx = [&]() -> std::size_t {
            if constexpr (is_striped) return mm_.shard_for(key);
            else return 0;
        }();
        // Task 4: RAII latency timer — records to set_latency histogram.
        auto& stats_ref = [&]() -> cache_stats& {
            if constexpr (is_striped) {
                return mm_.shard(shard_idx).stats();
            } else {
                return mm_.stats();
            }
        }();
        // O2: capture latency for slow-query threshold check.
        // O1: trace_notifier declared before slow_query_notifier so it
        // destructs after, reading latency_ns written by ~scope_latency_timer.
        std::uint64_t set_latency_ns = 0;
        bool trace_hit = false;
        trace_notifier<unified_cache, Key> trace_n(
            *this, key, trace_op_kind::set, set_latency_ns, trace_hit);
        slow_query_notifier<unified_cache, Key> slow_q_notifier(
            *this, key, slow_op_kind::set, set_latency_ns);
        detail::scope_latency_timer timer_(stats_ref.set_latency,
            stats_ref.latency_tracking_enabled.load(std::memory_order_relaxed),
            &set_latency_ns);

        // P-HIGH-2 (T-H1): Capture the OOM event inside the write lock,
        // but dispatch the handler *after* the lock is released. Invoking
        // the handler inside the lock caused deadlocks when the handler
        // called back into cache methods (flush/remove/set).
        std::optional<oom_event> oom_evt;

        if constexpr (is_striped) {
            auto lock = acquire_write_lock_for_key(key);
            if (!check_memory_admission(key, value)) return;
            // Task D: overflow enforcement (skip when kAllowGrowth — default).
            if (!apply_overflow_policy_for_set(key, shard_idx)) return;
            mm_.set(key, std::forward<V>(value));
            trace_hit = true;  // O1: set succeeded
            // P1-3 / T-H1: Check memory pressure after insertion — may
            // enter critical mode if this insertion pushed us over the
            // threshold. The handler is dispatched below, outside the lock.
            oom_evt = check_memory_pressure();
            maybe_report_memory_to_monitor();
            flush_shard_pending(shard_idx);
        } else {
            flush_guard fg{mm_};
            auto lock = acquire_write_lock_for_key(key);
            if (!check_memory_admission(key, value)) return;
            // Task D: overflow enforcement (skip when kAllowGrowth — default).
            if (!apply_overflow_policy_for_set(key, 0)) return;
            mm_.set(key, std::forward<V>(value));
            trace_hit = true;  // O1: set succeeded
            // P1-3 / T-H1: Check memory pressure after insertion.
            oom_evt = check_memory_pressure();
            maybe_report_memory_to_monitor();
        }
        // P-HIGH-2 (T-H1): Dispatch OOM handler outside the write lock.
        // In sync mode this invokes the handler inline; in async mode
        // (set_async_oom_handler(true)) it enqueues for the drain worker.
        dispatch_oom_event(oom_evt);
    }

    // --------------------------------------------------------------------
    // P1-1: Native TTL set. Stores the value with an absolute expiry time
    // computed from `ttl`. The expiry is stored directly on `cache_item`
    // (no wrapper), so `peek_for_get` can check it inline on the hot path
    // without the `ttl_cache` wrapper's double-locking.
    //
    // SFINAE guards the call to `mm_.set_with_expiry()`: if the MM strategy
    // does not yet implement native TTL (e.g., mm_2q, mm_fifo), we fall back
    // to plain `set()` and the item is stored without an expiry. This keeps
    // the API uniform across all cache aliases.
    // --------------------------------------------------------------------
    template <typename V, typename Rep, typename Period>
    void set_with_ttl(const Key& key, V&& value,
                      std::chrono::duration<Rep, Period> ttl) {
        // P-HIGH-3 (T-H2): Unified pre-write validation (previously missing
        // the is_memory_critical() guard — OOM could bypass set_with_ttl).
        if (pre_write_check("set_with_ttl") == pre_write_result::kRejectInsert) {
            return;
        }
        // T14.1: Compact path — compact_cache does not support TTL natively.
        // Fall back to plain set() (item is stored without expiry).
        if constexpr (Trait::is_compact) {
            (void)ttl;  // suppress unused warning
            compact().set(key, std::forward<V>(value));
            return;
        }
        // P1-1 (T2.2): Drain TLS ring before write lock — see set().
        maybe_drain_tls_ring_pre_evict();
        const std::size_t shard_idx = [&]() -> std::size_t {
            if constexpr (is_striped) return mm_.shard_for(key);
            else return 0;
        }();
        auto& stats_ref = [&]() -> stats_type& {
            if constexpr (is_striped) {
                return mm_.shard(shard_idx).stats();
            } else {
                return mm_.stats();
            }
        }();
        // O2: capture latency for slow-query threshold check.
        // O1: trace_notifier declared before slow_query_notifier so it
        // destructs after, reading latency_ns written by ~scope_latency_timer.
        std::uint64_t set_latency_ns = 0;
        bool trace_hit = false;
        trace_notifier<unified_cache, Key> trace_n(
            *this, key, trace_op_kind::set, set_latency_ns, trace_hit);
        slow_query_notifier<unified_cache, Key> slow_q_notifier(
            *this, key, slow_op_kind::set, set_latency_ns);
        detail::scope_latency_timer timer_(stats_ref.set_latency,
            stats_ref.latency_tracking_enabled.load(std::memory_order_relaxed),
            &set_latency_ns);

        // Compute absolute expiry as nanoseconds since steady_clock epoch.
        // ttl <= 0 means "no TTL" — store 0 (the sentinel for no expiration).
        // TTL jitter (default ±10%) is applied to prevent thundering-herd
        // avalanches when many keys share the same nominal TTL.
        std::uint64_t expiry_ns = 0;
        if (ttl > std::chrono::duration<Rep, Period>::zero()) {
            auto effective_ttl = ttl;
            if (ttl_jitter_enabled_.load(std::memory_order_relaxed) &&
                ttl_jitter_pct_ > 0.0) {
                effective_ttl = detail::apply_ttl_jitter(ttl, ttl_jitter_pct_);
            }
            auto now = std::chrono::steady_clock::now();
            auto expiry = now + std::chrono::duration_cast<
                std::chrono::steady_clock::duration>(effective_ttl);
            expiry_ns = static_cast<std::uint64_t>(expiry.time_since_epoch().count());
        }

        if constexpr (is_striped) {
            auto lock = acquire_write_lock_for_key(key);
            if (!check_memory_admission(key, value)) return;
            if (!apply_overflow_policy_for_set(key, shard_idx)) return;
            if constexpr (requires { mm_.set_with_expiry(key, value, expiry_ns); }) {
                mm_.set_with_expiry(key, std::forward<V>(value), expiry_ns);
            } else {
                mm_.set(key, std::forward<V>(value));
            }
            trace_hit = true;  // O1: set succeeded
            maybe_report_memory_to_monitor();
            flush_shard_pending(shard_idx);
        } else {
            flush_guard fg{mm_};
            auto lock = acquire_write_lock_for_key(key);
            if (!check_memory_admission(key, value)) return;
            if (!apply_overflow_policy_for_set(key, 0)) return;
            if constexpr (requires { mm_.set_with_expiry(key, value, expiry_ns); }) {
                mm_.set_with_expiry(key, std::forward<V>(value), expiry_ns);
            } else {
                mm_.set(key, std::forward<V>(value));
            }
            trace_hit = true;  // O1: set succeeded
            maybe_report_memory_to_monitor();
        }
    }

    template <typename V>
    bool add(const Key& key, V&& value) {
        // P-HIGH-3 (T-H2): Unified pre-write validation (previously missing
        // both shutdown and is_memory_critical() guards).
        if (pre_write_check("add") == pre_write_result::kRejectInsert) {
            return false;
        }
        // T14.1: Compact path.
        if constexpr (Trait::is_compact) {
            return compact().add(key, std::forward<V>(value));
        }
        // P1-1 (T2.2): Drain TLS ring before write lock — see set().
        maybe_drain_tls_ring_pre_evict();
        // Task B: cache shard_idx once per call.
        const std::size_t shard_idx = [&]() -> std::size_t {
            if constexpr (is_striped) return mm_.shard_for(key);
            else return 0;
        }();
        if constexpr (is_striped) {
            auto lock = acquire_write_lock_for_key(key);
            if (!check_memory_admission(key, value)) return false;
            // Task D: overflow enforcement (skip when kAllowGrowth — default).
            if (!apply_overflow_policy_for_set(key, shard_idx)) return false;
            auto result = mm_.add(key, std::forward<V>(value));
            maybe_report_memory_to_monitor();
            flush_shard_pending(shard_idx);
            return result;
        } else {
            flush_guard fg{mm_};
            auto lock = acquire_write_lock_for_key(key);
            if (!check_memory_admission(key, value)) return false;
            // Task D: overflow enforcement (skip when kAllowGrowth — default).
            if (!apply_overflow_policy_for_set(key, 0)) return false;
            auto result = mm_.add(key, std::forward<V>(value));
            maybe_report_memory_to_monitor();
            return result;
        }
    }

    template <typename V>
    bool replace(const Key& key, V&& value) {
        // P-HIGH-3 (T-H2): Unified pre-write validation (previously missing
        // both shutdown and is_memory_critical() guards). replace() does not
        // add memory, but rejecting it in critical mode is safe and keeps
        // behavior consistent across all write paths.
        if (pre_write_check("replace") == pre_write_result::kRejectInsert) {
            return false;
        }
        // T14.1: Compact path — compact_cache lacks a replace() method,
        // so emulate it via contains() + set().
        if constexpr (Trait::is_compact) {
            if (!compact().contains(key)) return false;
            compact().set(key, std::forward<V>(value));
            return true;
        }
        // P1-1 (T2.2): Drain TLS ring before write lock — see set().
        // replace() updates an existing item in-place; it does not add a
        // new item, so eviction is not triggered. But we drain anyway
        // for consistency and to keep the LRU list fresh in case the
        // caller chains operations.
        maybe_drain_tls_ring_pre_evict();
        // Task B: cache shard_idx once per call.
        const std::size_t shard_idx = [&]() -> std::size_t {
            if constexpr (is_striped) return mm_.shard_for(key);
            else return 0;
        }();
        if constexpr (is_striped) {
            auto lock = acquire_write_lock_for_key(key);
            auto result = mm_.replace(key, std::forward<V>(value));
            maybe_report_memory_to_monitor();
            flush_shard_pending(shard_idx);
            return result;
        } else {
            flush_guard fg{mm_};
            auto lock = acquire_write_lock_for_key(key);
            auto result = mm_.replace(key, std::forward<V>(value));
            maybe_report_memory_to_monitor();
            return result;
        }
    }

    // --------------------------------------------------------------------
    // Get-or-fetch / operator[]
    // --------------------------------------------------------------------

    /// Get a value or fetch it from the provider on miss.
    /// Uses the provider set via set_value_provider().
    /// Throws std::runtime_error if no provider is set.
    ///
    /// T-G3: When singleflight is enabled (default for production aliases),
    /// concurrent misses for the same key are coalesced — only the leader
    /// calls the provider; followers block on the leader's result. This
    /// prevents cache-stampede / thundering-herd when a hot key expires.
    /// Delegates to get_or_fetch(key, provider) to reuse the singleflight
    /// integration there.
    Value get_or_fetch(const Key& key) {
        // P-HIGH-3 (T-H2): Unified pre-write validation. get_or_fetch may
        // insert on miss, so it must respect shutdown and OOM guards.
        // Note: if the key is already cached, the hit path proceeds
        // normally even in critical mode (reads are always allowed).
        // The reject only blocks the Phase-3 insertion.
        if (is_shutdown()) {
            throw cache_closed_exception(
                "unified_cache::get_or_fetch: cache is shut down");
        }
        std::function<Value(const Key&)> provider;
        // Task B: cache shard_idx once per call to avoid redundant hash.
        const std::size_t shard_idx = [&]() -> std::size_t {
            if constexpr (is_striped) return mm_.shard_for(key);
            else return 0;
        }();
        {
            if constexpr (is_striped) {
                // Phase 1: Lock-free lookup — find_and_pin() atomically
                // pins the item under the hash table's bucket lock.
                read_handle<Value> handle;
                {
                    handle = mm_.peek_for_get(key);
                    if (handle) {
                    // Hit — optimistic promotion (same pattern as get())
                    // Task C: attach per-cache stats if enabled.
                    attach_handle_stats_if_enabled(handle);
                }
                }
                if (handle) {
                    bool should_try_promote = true;
                    if constexpr (requires { mm_.shard(0).config().defer_promotion; }) {
                        if (mm_.shard(shard_idx).config().defer_promotion) {
                            should_try_promote = false;
                        }
                    }
                    if (should_try_promote) {
                        auto wlock = try_acquire_write_lock_for_key(key);
                        if (wlock.owns_lock()) {
                            mm_.promote(key);
                            mm_.shard(shard_idx).stats().register_hit();
                            mm_.shard(shard_idx).callbacks().collect_hit(key, *handle);
                            flush_shard_pending(shard_idx);
                        } else {
                            record_access_in_ring(key);
                            mm_.shard(shard_idx).stats().register_hit();
                            // R8: collect hit callback in defer path.
                            mm_.shard(shard_idx).callbacks().collect_hit(key, *handle);
                            flush_shard_pending(shard_idx);
                        }
                    } else {
                        record_access_in_ring(key);
                        mm_.shard(shard_idx).stats().register_hit();
                        // R8: collect hit callback in defer_promotion mode.
                        mm_.shard(shard_idx).callbacks().collect_hit(key, *handle);
                        flush_shard_pending(shard_idx);
                    }
                    return *handle;
                }
                // Miss — need write lock to copy provider and register miss
                {
                    auto wlock = acquire_write_lock_for_key(key);
                    // Double-check under write lock
                    auto handle2 = mm_.peek_for_get(key);
                    if (handle2) {
                        mm_.shard(shard_idx).stats().register_hit();
                        mm_.shard(shard_idx).callbacks().collect_hit(key, *handle2);
                        flush_shard_pending(shard_idx);
                        return *handle2;
                    }
                    if (!value_provider_) {
                        mm_.shard(shard_idx).stats().register_miss();
                        mm_.shard(shard_idx).callbacks().collect_miss(key);
                        flush_shard_pending(shard_idx);
                        throw cache_config_exception("get_or_fetch: no value provider set");
                    }
                    provider = value_provider_;
                    mm_.shard(shard_idx).stats().register_miss();
                    mm_.shard(shard_idx).callbacks().collect_miss(key);
                    flush_shard_pending(shard_idx);
                }
            } else {
                flush_guard fg{mm_};
                // Phase 1: Lock-free lookup — find_and_pin() atomically pins item
                read_handle<Value> handle;
                {
                    handle = mm_.peek_for_get(key);
                }
                if (handle) {
                    // Task C: attach per-cache stats if enabled (parity with striped path).
                    attach_handle_stats_if_enabled(handle);
                    bool should_try_promote = true;
                    if constexpr (requires { mm_.config().defer_promotion; }) {
                        if (mm_.config().defer_promotion) {
                            should_try_promote = false;
                        }
                    }
                    if (should_try_promote) {
                        auto wlock = try_acquire_write_lock_for_key(key);
                        if (wlock.owns_lock()) {
                            mm_.promote(key);
                            mm_.stats().register_hit();
                            mm_.callbacks().collect_hit(key, *handle);
                        } else {
                            record_access_in_ring(key);
                            mm_.stats().register_hit();
                            // R8: collect hit callback in defer path.
                            mm_.callbacks().collect_hit(key, *handle);
                        }
                    } else {
                        record_access_in_ring(key);
                        mm_.stats().register_hit();
                        // R8: collect hit callback in defer_promotion mode.
                        mm_.callbacks().collect_hit(key, *handle);
                    }
                    return *handle;
                }
                // Miss — need write lock to copy provider and register miss
                {
                    auto wlock = acquire_write_lock_for_key(key);
                    auto handle2 = mm_.peek_for_get(key);
                    if (handle2) {
                        mm_.stats().register_hit();
                        mm_.callbacks().collect_hit(key, *handle2);
                        return *handle2;
                    }
                    if (!value_provider_) {
                        mm_.stats().register_miss();
                        mm_.callbacks().collect_miss(key);
                        throw cache_config_exception("get_or_fetch: no value provider set");
                    }
                    provider = value_provider_;
                    mm_.stats().register_miss();
                    mm_.callbacks().collect_miss(key);
                }
            }
        }
        // T-G3: When singleflight is enabled, delegate to the provider-aware
        // overload so concurrent misses for the same key are coalesced.
        // The provider has already been copied out under the read lock, so
        // we can safely pass it by reference here.
        if (singleflight_enabled_.load(std::memory_order_acquire)) {
            return get_or_fetch(key,
                [&provider](const Key& k) -> Value { return provider(k); });
        }
        // Phase 2: Call provider outside any lock
        Value v = provider(key);
        // P1-1 (T2.2): Drain TLS ring before write lock — see set().
        // The provider ran outside the lock; this thread may have
        // accumulated TLS ring entries from prior get() calls.
        maybe_drain_tls_ring_pre_evict();
        // Phase 3: Insert under write lock (double-check again)
        {
            // P-HIGH-3 (T-H2): OOM guard for the insertion phase. Reads of
            // existing items are allowed even in critical mode (handled by
            // the peek_for_get hit paths above), but new insertions are
            // rejected. Also re-check shutdown (TOCTOU: cache may have been
            // shut down between the entry check and lock acquisition).
            if (is_memory_critical()) {
                // Return the fetched value without caching it. The caller
                // gets the correct result; only the cache mutation is skipped.
                return v;
            }
            if constexpr (is_striped) {
                auto lock = acquire_write_lock_for_key(key);
                // P-HIGH-3 (T-H2): TOCTOU re-check inside the lock.
                if (recheck_shutdown_in_lock()) {
                    throw cache_closed_exception(
                        "unified_cache::get_or_fetch: cache is shut down");
                }
                auto handle = mm_.peek_for_get(key);
                if (handle) {
                    mm_.shard(shard_idx).stats().register_hit();
                    mm_.shard(shard_idx).callbacks().collect_hit(key, *handle);
                    flush_shard_pending(shard_idx);
                    return *handle;
                }
                if (!check_memory_admission(key, v)) {
                    flush_shard_pending(shard_idx);
                    throw cache_oom_exception("get_or_fetch: memory monitor rejected admission");
                }
                mm_.set(key, v);
                maybe_report_memory_to_monitor();
                flush_shard_pending(shard_idx);
                return v;
            } else {
                flush_guard fg{mm_};
                auto lock = acquire_write_lock_for_key(key);
                // P-HIGH-3 (T-H2): TOCTOU re-check inside the lock.
                if (recheck_shutdown_in_lock()) {
                    throw cache_closed_exception(
                        "unified_cache::get_or_fetch: cache is shut down");
                }
                auto handle = mm_.peek_for_get(key);
                if (handle) {
                    mm_.stats().register_hit();
                    mm_.callbacks().collect_hit(key, *handle);
                    return *handle;
                }
                if (!check_memory_admission(key, v)) {
                    throw cache_oom_exception("get_or_fetch: memory monitor rejected admission");
                }
                mm_.set(key, v);
                maybe_report_memory_to_monitor();
                return v;
            }
        }
    }

    /// Get-or-fetch with an explicit provider function.
    /// @param key       The key to look up.
    /// @param provider  Callable: Value(const Key&), called on cache miss.
    ///
    /// The provider is called outside the cache lock (double-checked locking).
    template <typename Fn>
        requires std::invocable<Fn, const Key&> &&
                 std::convertible_to<std::invoke_result_t<Fn, const Key&>, Value>
    Value get_or_fetch(const Key& key, Fn&& provider) {
        // P-HIGH-3 (T-H2): Unified pre-write validation (shutdown check).
        // OOM check is deferred to Phase 3 (insertion) — see below.
        if (is_shutdown()) {
            throw cache_closed_exception(
                "unified_cache::get_or_fetch: cache is shut down");
        }
        // Task B: cache shard_idx once per call to avoid redundant hash.
        const std::size_t shard_idx = [&]() -> std::size_t {
            if constexpr (is_striped) return mm_.shard_for(key);
            else return 0;
        }();
        {
            if constexpr (is_striped) {
                // Lock-free lookup (hot path)
                read_handle<Value> handle;
                {
                    handle = mm_.peek_for_get(key);
                }
                if (handle) {
                    // Task C: attach per-cache stats if enabled.
                    attach_handle_stats_if_enabled(handle);
                    bool should_try_promote = true;
                    if constexpr (requires { mm_.shard(0).config().defer_promotion; }) {
                        if (mm_.shard(shard_idx).config().defer_promotion) {
                            should_try_promote = false;
                        }
                    }
                    if (should_try_promote) {
                        auto wlock = try_acquire_write_lock_for_key(key);
                        if (wlock.owns_lock()) {
                            mm_.promote(key);
                            mm_.shard(shard_idx).stats().register_hit();
                            mm_.shard(shard_idx).callbacks().collect_hit(key, *handle);
                            flush_shard_pending(shard_idx);
                        } else {
                            record_access_in_ring(key);
                            mm_.shard(shard_idx).stats().register_hit();
                            // R8: collect hit callback in defer path.
                            mm_.shard(shard_idx).callbacks().collect_hit(key, *handle);
                            flush_shard_pending(shard_idx);
                        }
                    } else {
                        record_access_in_ring(key);
                        mm_.shard(shard_idx).stats().register_hit();
                        // R8: collect hit callback in defer_promotion mode.
                        mm_.shard(shard_idx).callbacks().collect_hit(key, *handle);
                        flush_shard_pending(shard_idx);
                    }
                    return *handle;
                }
                mm_.shard(shard_idx).stats().register_miss();
                mm_.shard(shard_idx).callbacks().collect_miss(key);
                flush_shard_pending(shard_idx);
            } else {
                flush_guard fg{mm_};
                read_handle<Value> handle;
                {
                    handle = mm_.peek_for_get(key);
                }
                if (handle) {
                    // Task C: attach per-cache stats if enabled (parity with striped path).
                    attach_handle_stats_if_enabled(handle);
                    bool should_try_promote = true;
                    if constexpr (requires { mm_.config().defer_promotion; }) {
                        if (mm_.config().defer_promotion) {
                            should_try_promote = false;
                        }
                    }
                    if (should_try_promote) {
                        auto wlock = try_acquire_write_lock_for_key(key);
                        if (wlock.owns_lock()) {
                            mm_.promote(key);
                            mm_.stats().register_hit();
                            mm_.callbacks().collect_hit(key, *handle);
                        } else {
                            record_access_in_ring(key);
                            mm_.stats().register_hit();
                            // R8: collect hit callback in defer path.
                            mm_.callbacks().collect_hit(key, *handle);
                        }
                    } else {
                        record_access_in_ring(key);
                        mm_.stats().register_hit();
                        // R8: collect hit callback in defer_promotion mode.
                        mm_.callbacks().collect_hit(key, *handle);
                    }
                    return *handle;
                }
                mm_.stats().register_miss();
                mm_.callbacks().collect_miss(key);
            }
        }
        // T-M1: singleflight / cache stampede protection.
        // When enabled, concurrent misses on the same key are coalesced:
        // the first caller (leader) executes the provider; subsequent
        // callers (followers) block on a CV until the leader completes,
        // then receive the leader's result. This prevents thundering-herd
        // provider invocations on hot keys (especially at TTL expiry).
        typename detail::singleflight_tracker<Key, Value>::state_ptr sf_state;
        bool sf_is_leader = false;
        if (singleflight_enabled_.load(std::memory_order_acquire)) {
            auto sf_result = singleflight_.acquire(key);
            sf_state = sf_result.state;
            sf_is_leader = sf_result.is_leader;
            if (!sf_is_leader) {
                // Follower: block on leader's result. wait_and_get
                // rethrows any provider exception captured by the leader.
                singleflight_.record_coalesced();
                Value leader_v = sf_state->wait_and_get();
                // Re-check the cache: the leader should have inserted.
                // If present, return a handle-pinned value for parity
                // with the leader path. If missing (insertion failed
                // due to OOM/admission rejection), return the leader's
                // value directly.
                if constexpr (is_striped) {
                    read_handle<Value> h = mm_.peek_for_get(key);
                    if (h) {
                        attach_handle_stats_if_enabled(h);
                        mm_.shard(shard_idx).stats().register_hit();
                        return *h;
                    }
                } else {
                    flush_guard fg{mm_};
                    read_handle<Value> h = mm_.peek_for_get(key);
                    if (h) {
                        attach_handle_stats_if_enabled(h);
                        mm_.stats().register_hit();
                        return *h;
                    }
                }
                return leader_v;
            }
            // Leader: fall through to call provider + insert. After
            // completion (success or exception), signal followers via
            // singleflight_.complete() and remove the in-flight entry.
        }
        // Call provider outside lock
        try {
            Value v = std::forward<Fn>(provider)(key);
            // P1-1 (T2.2): Drain TLS ring before write lock — see set().
            // The provider ran outside the lock; this thread may have
            // accumulated TLS ring entries from prior get() calls.
            maybe_drain_tls_ring_pre_evict();
            // Insert under write lock. Consolidate all return paths into
            // a single `result` variable so the leader's singleflight
            // completion signal is emitted on exactly one path.
            Value result;
            {
                // P-HIGH-3 (T-H2): OOM guard for the insertion phase + TOCTOU
                // shutdown re-check inside the lock. See get_or_fetch(key) for
                // full rationale.
                if (is_memory_critical()) {
                    result = std::move(v);  // return fetched value without caching
                } else if constexpr (is_striped) {
                    auto lock = acquire_write_lock_for_key(key);
                    if (recheck_shutdown_in_lock()) {
                        throw cache_closed_exception(
                            "unified_cache::get_or_fetch: cache is shut down");
                    }
                    auto handle = mm_.peek_for_get(key);
                    if (handle) {
                        mm_.shard(shard_idx).stats().register_hit();
                        mm_.shard(shard_idx).callbacks().collect_hit(key, *handle);
                        flush_shard_pending(shard_idx);
                        result = *handle;
                    } else {
                        if (!check_memory_admission(key, v)) {
                            flush_shard_pending(shard_idx);
                            throw cache_oom_exception("get_or_fetch: memory monitor rejected admission");
                        }
                        mm_.set(key, v);
                        maybe_report_memory_to_monitor();
                        flush_shard_pending(shard_idx);
                        result = std::move(v);
                    }
                } else {
                    flush_guard fg{mm_};
                    auto lock = acquire_write_lock_for_key(key);
                    if (recheck_shutdown_in_lock()) {
                        throw cache_closed_exception(
                            "unified_cache::get_or_fetch: cache is shut down");
                    }
                    auto handle = mm_.peek_for_get(key);
                    if (handle) {
                        mm_.stats().register_hit();
                        mm_.callbacks().collect_hit(key, *handle);
                        result = *handle;
                    } else {
                        if (!check_memory_admission(key, v)) {
                            throw cache_oom_exception("get_or_fetch: memory monitor rejected admission");
                        }
                        mm_.set(key, v);
                        maybe_report_memory_to_monitor();
                        result = std::move(v);
                    }
                }
            }
            // T-M1: Signal singleflight completion (success path). On
            // exception, the catch block below signals with the captured
            // exception_ptr so followers rethrow it.
            if (sf_is_leader) {
                singleflight_.complete(key, sf_state, result);
            }
            return result;
        } catch (...) {
            // T-M1: Propagate provider/insertion exception to all
            // followers waiting on this key, then rethrow to caller.
            if (sf_is_leader) {
                singleflight_.complete(key, sf_state, std::nullopt,
                                        std::current_exception());
            }
            throw;
        }
    }

    /// Array subscript operator with auto-fetch semantics.
    /// Requires a value provider to be set via set_value_provider().
    /// Throws std::runtime_error on miss if no provider is set.
    Value operator[](const Key& key) {
        return get_or_fetch(key);
    }

    /// H0: Get with handle — 返回 read_handle，防止持有期被淘汰。
    /// Lock-free read path: peek_for_get() uses find_and_pin() to
    /// atomically increment refcount under the hash table's bucket lock,
    /// eliminating the need for a stripe-level read lock in Phase 1.
    /// Only Phase 2 (LRU promotion) acquires a stripe write lock.
    read_handle<Value> get(const Key& key) {
        // Task 11: reject reads after graceful shutdown.
        if (is_shutdown()) {
            throw cache_closed_exception("unified_cache::get: cache is shut down");
        }
        // T14.1: Compact path — delegate to compact_cache::get(), which
        // returns std::optional<std::reference_wrapper<Value>>. Wrap the
        // pointer in a non-pinning read_handle (refcount=nullptr). The
        // caller must ensure the value is not used after the cache is
        // destroyed or the key is evicted (use external synchronization
        // or copy the value out before releasing any locks).
        if constexpr (Trait::is_compact) {
            auto ref = compact().get(key);
            if (!ref) return {};
            // Construct a non-pinning handle: value ptr set, refcount=nullptr.
            return read_handle<Value>(&ref->get(), nullptr, nullptr);
        }
        // T-P3-1: Precompute hash ONCE and reuse it for shard dispatch,
        // hash-table lookup, and write-lock acquisition. Previously the
        // hash was computed up to 3 times per get(): once in shard_for(key),
        // once in peek_for_get(key), and once in try_acquire_write_lock_for_key(key).
        const std::size_t hash = hash_type{}(key);
        // Task B: cache shard_idx once; reused across latency timer + body.
        const std::size_t shard_idx = [&]() -> std::size_t {
            if constexpr (is_striped) return mm_.shard_for_hash(hash);
            else return 0;
        }();
        // Task 4: RAII latency timer — records to the appropriate shard's
        // get_latency histogram on function exit.
        auto& stats_ref = [&]() -> stats_type& {
            if constexpr (is_striped) {
                return mm_.shard(shard_idx).stats();
            } else {
                return mm_.stats();
            }
        }();
        // O2: capture latency for slow-query threshold check. The notifier
        // is declared BEFORE the timer so it destructs AFTER the timer
        // (destruction is reverse of construction), allowing it to read
        // the latency value written by the timer's destructor.
        // O1: trace_notifier declared before slow_query_notifier so it
        // destructs after both, reading latency + hit flag.
        std::uint64_t get_latency_ns = 0;
        bool trace_hit = false;
        trace_notifier<unified_cache, Key> trace_n(
            *this, key, trace_op_kind::get, get_latency_ns, trace_hit);
        slow_query_notifier<unified_cache, Key> slow_q_notifier(
            *this, key, slow_op_kind::get, get_latency_ns);
        detail::scope_latency_timer timer_(stats_ref.get_latency,
            stats_ref.latency_tracking_enabled.load(std::memory_order_relaxed),
            &get_latency_ns);

        if constexpr (is_striped) {
            // Phase 1: Lock-free lookup — find_and_pin() atomically
            // pins the item under the hash table's bucket shared lock.
            read_handle<Value> handle;
            {
                // T-P3-1: Reuse precomputed hash for lock-free lookup.
                // T-G11: clear overflow flag so we can attribute any
                // empty result to this call.
                detail::tls_clear_incRef_overflow_flag();
                handle = mm_.peek_for_get_with_hash(key, hash);
                if (!handle) {
                    // T-G11: refcount overflow → bump counter, yield,
                    // retry once. If the retry also overflows, throw
                    // (distinguishes overflow from a real miss).
                    if (detail::tls_last_incRef_was_overflow()) {
                        mm_.shard(shard_idx).stats().incRef_overflow_count.value
                            .fetch_add(1, std::memory_order_relaxed);
                        std::this_thread::yield();
                        detail::tls_clear_incRef_overflow_flag();
                        handle = mm_.peek_for_get_with_hash(key, hash);
                        if (!handle && detail::tls_last_incRef_was_overflow()) {
                            mm_.shard(shard_idx).stats().incRef_overflow_count.value
                                .fetch_add(1, std::memory_order_relaxed);
                            throw refcount_overflow_exception(
                                "unified_cache::get: refcount overflow on key "
                                "(access_ref saturated — likely a handle leak)");
                        }
                    }
                    if (!handle) {
                        mm_.shard(shard_idx).stats().register_miss();
                        mm_.shard(shard_idx).callbacks().collect_miss(key);
                        flush_shard_pending(shard_idx);
                        return {};
                    }
                }
                // Task C: attach per-cache stats if enabled.
                attach_handle_stats_if_enabled(handle);
                trace_hit = true;  // O1: cache hit
            }
            // Phase 2: Try non-blocking write lock for LRU promotion
            {
                bool should_try_promote = true;
                if constexpr (requires { mm_.shard(0).config().defer_promotion; }) {
                    if (mm_.shard(shard_idx).config().defer_promotion) {
                        should_try_promote = false;
                    }
                }
                if (should_try_promote) {
                    // T-P3-1: Reuse precomputed hash for write-lock acquisition.
                    auto wlock = try_acquire_write_lock_for_hash(hash);
                    if (wlock.owns_lock()) {
                        mm_.promote(key);
                        mm_.shard(shard_idx).stats().register_hit();
                        mm_.shard(shard_idx).callbacks().collect_hit(key, *handle);
                        flush_shard_pending(shard_idx);
                    } else {
                        // Write lock contended — defer promotion
                        record_access_in_ring(key);
                        mm_.shard(shard_idx).stats().register_hit();
                        // R8: defer_promotion mode previously skipped the hit
                        // callback — hit metrics/callbacks silently undercounted
                        // on the default production path. collect_hit is a
                        // zero-cost atomic check when no hit callbacks are
                        // registered, so it is safe on the hot path.
                        mm_.shard(shard_idx).callbacks().collect_hit(key, *handle);
                        flush_shard_pending(shard_idx);
                    }
                } else {
                    // defer_promotion mode: always defer to TLS ring
                    record_access_in_ring(key);
                    mm_.shard(shard_idx).stats().register_hit();
                    // R8: see note in the "Write lock contended" branch above —
                    // collect the hit callback in defer_promotion mode too.
                    mm_.shard(shard_idx).callbacks().collect_hit(key, *handle);
                    flush_shard_pending(shard_idx);
                }
            }
            return handle;
        } else {
            // Non-striped: lock-free lookup + optional write lock for promotion
            flush_guard fg{mm_};
            read_handle<Value> handle;
            {
                // Lock-free: find_and_pin() atomically pins item
                // T-P3-1: Reuse precomputed hash for lock-free lookup.
                // T-G11: clear overflow flag + retry on overflow.
                detail::tls_clear_incRef_overflow_flag();
                handle = mm_.peek_for_get_with_hash(key, hash);
                if (!handle) {
                    if (detail::tls_last_incRef_was_overflow()) {
                        mm_.stats().incRef_overflow_count.value
                            .fetch_add(1, std::memory_order_relaxed);
                        std::this_thread::yield();
                        detail::tls_clear_incRef_overflow_flag();
                        handle = mm_.peek_for_get_with_hash(key, hash);
                        if (!handle && detail::tls_last_incRef_was_overflow()) {
                            mm_.stats().incRef_overflow_count.value
                                .fetch_add(1, std::memory_order_relaxed);
                            throw refcount_overflow_exception(
                                "unified_cache::get: refcount overflow on key "
                                "(access_ref saturated — likely a handle leak)");
                        }
                    }
                    if (!handle) {
                        mm_.stats().register_miss();
                        mm_.callbacks().collect_miss(key);
                        return {};
                    }
                }
                // Task C: attach per-cache stats if enabled (parity with striped path).
                attach_handle_stats_if_enabled(handle);
                trace_hit = true;  // O1: cache hit
            }
            {
                bool should_try_promote = true;
                if constexpr (requires { mm_.config().defer_promotion; }) {
                    if (mm_.config().defer_promotion) {
                        should_try_promote = false;
                    }
                }
                if (should_try_promote) {
                    // T-P3-1: Reuse precomputed hash for write-lock acquisition.
                    auto wlock = try_acquire_write_lock_for_hash(hash);
                    if (wlock.owns_lock()) {
                        mm_.promote(key);
                        mm_.stats().register_hit();
                        mm_.callbacks().collect_hit(key, *handle);
                    } else {
                        record_access_in_ring(key);
                        mm_.stats().register_hit();
                        // R8: collect hit callback in defer mode (see striped note).
                        mm_.callbacks().collect_hit(key, *handle);
                    }
                } else {
                    // defer_promotion mode: always defer to TLS ring
                    record_access_in_ring(key);
                    mm_.stats().register_hit();
                    // R8: collect hit callback in defer_promotion mode.
                    mm_.callbacks().collect_hit(key, *handle);
                }
            }
            return handle;
        }
        // O2: slow_q_notifier destructs here (after timer_) and invokes
        // notify_slow_query if the latency exceeded the threshold.
    }

    // --------------------------------------------------------------------
    // Task 9: Production APIs — try_get, try_get_or_fetch, get_with_ttl, cas
    // --------------------------------------------------------------------

    /// Non-blocking get: returns immediately with an empty optional if the
    /// write lock for promotion is contended (would block). Useful for
    /// latency-sensitive paths where blocking is worse than skipping
    /// promotion. The item, if found, is still returned via peek_for_get()
    /// (lock-free lookup); only the LRU promotion is skipped on contention.
    ///
    /// @return std::optional<read_handle<Value>> — engaged on hit, empty on
    ///         miss OR on contention-with-skipped-promotion (use try_get
    ///         when you can tolerate occasional false negatives).
    std::optional<read_handle<Value>> try_get(const Key& key) {
        // Task 11: non-blocking variant — return nullopt after shutdown.
        if (is_shutdown()) {
            return std::nullopt;
        }
        // T14.1: Compact path — compact_cache::get() already acquires the
        // necessary lock internally; there's no separate "try" path. We
        // delegate to get() and wrap the result. (This means try_get on
        // compact caches is not truly non-blocking, but compact_cache's
        // internal lock is per-key striped, so contention is minimal.)
        if constexpr (Trait::is_compact) {
            auto ref = compact().get(key);
            if (!ref) return std::nullopt;
            read_handle<Value> handle(&ref->get(), nullptr, nullptr);
            return std::optional<read_handle<Value>>{std::move(handle)};
        }
        // T-P3-6 + T-P3-1: Precompute hash ONCE and reuse it for shard
        // dispatch, lock-free lookup, and (when needed) write-lock
        // acquisition. Previously try_get recomputed the hash up to 3x
        // (shard_for, peek_for_get, try_acquire_write_lock_for_key),
        // making it slower than get() in defer_promotion mode.
        const std::size_t hash = hash_type{}(key);
        // Task B: cache shard_idx once per call.
        const std::size_t shard_idx = [&]() -> std::size_t {
            if constexpr (is_striped) return mm_.shard_for_hash(hash);
            else return 0;
        }();
        if constexpr (is_striped) {
            // T-P3-6: Use precomputed hash for lock-free lookup.
            // T-G11: clear the thread-local overflow flag before the
            // lookup so we can attribute any failure to this call.
            detail::tls_clear_incRef_overflow_flag();
            read_handle<Value> handle = mm_.peek_for_get_with_hash(key, hash);
            if (!handle) {
                // T-G11: if the failure was due to refcount overflow,
                // bump the per-shard counter and retry once with yield.
                // Overflow may be transient (handles being released
                // concurrently), so a single retry recovers most cases.
                if (detail::tls_last_incRef_was_overflow()) {
                    mm_.shard(shard_idx).stats().incRef_overflow_count.value
                        .fetch_add(1, std::memory_order_relaxed);
                    std::this_thread::yield();
                    detail::tls_clear_incRef_overflow_flag();
                    handle = mm_.peek_for_get_with_hash(key, hash);
                    if (!handle && detail::tls_last_incRef_was_overflow()) {
                        mm_.shard(shard_idx).stats().incRef_overflow_count.value
                            .fetch_add(1, std::memory_order_relaxed);
                    }
                }
                if (!handle) {
                    mm_.shard(shard_idx).stats().register_miss();
                    mm_.shard(shard_idx).callbacks().collect_miss(key);
                    flush_shard_pending(shard_idx);
                    return std::nullopt;
                }
            }
            // Task C: attach per-cache stats if enabled.
            attach_handle_stats_if_enabled(handle);

            // T-P3-6: Fast path for defer_promotion mode — check BEFORE
            // attempting any write lock. When defer_promotion is true,
            // skip try_acquire_write_lock_for_hash entirely and just
            // record the access in the TLS ring. This makes try_get
            // FASTER than get in defer_promotion mode because:
            //   1. No write lock attempt (not even try_lock)
            //   2. No latency timer overhead (get has one, try_get doesn't)
            //   3. No shutdown throw (try_get returns nullopt, lighter)
            //   4. Early return — minimal control flow on the hot path
            if constexpr (requires { mm_.shard(0).config().defer_promotion; }) {
                if (mm_.shard(shard_idx).config().defer_promotion) {
                    // Fast path: defer promotion to TLS ring, no lock needed.
                    record_access_in_ring(key);
                    mm_.shard(shard_idx).stats().register_hit();
                    // R8: collect the hit callback in defer_promotion mode so
                    // hit metrics/callbacks are not silently undercounted.
                    mm_.shard(shard_idx).callbacks().collect_hit(key, *handle);
                    flush_shard_pending(shard_idx);
                    return std::optional<read_handle<Value>>{std::move(handle)};
                }
            }

            // Non-defer path: try non-blocking write lock for promotion.
            // T-P3-1: Reuse precomputed hash for write-lock acquisition.
            auto wlock = try_acquire_write_lock_for_hash(hash);
            if (wlock.owns_lock()) {
                mm_.promote(key);
                mm_.shard(shard_idx).stats().register_hit();
                mm_.shard(shard_idx).callbacks().collect_hit(key, *handle);
                flush_shard_pending(shard_idx);
            } else {
                // Lock contended — defer promotion via TLS ring.
                record_access_in_ring(key);
                mm_.shard(shard_idx).stats().register_hit();
                // R8: collect hit callback in defer path.
                mm_.shard(shard_idx).callbacks().collect_hit(key, *handle);
                flush_shard_pending(shard_idx);
            }
            return std::optional<read_handle<Value>>{std::move(handle)};
        } else {
            flush_guard fg{mm_};
            // T-P3-6: Use precomputed hash for lock-free lookup.
            // T-G11: clear the overflow flag and retry on overflow.
            detail::tls_clear_incRef_overflow_flag();
            read_handle<Value> handle = mm_.peek_for_get_with_hash(key, hash);
            if (!handle) {
                if (detail::tls_last_incRef_was_overflow()) {
                    mm_.stats().incRef_overflow_count.value
                        .fetch_add(1, std::memory_order_relaxed);
                    std::this_thread::yield();
                    detail::tls_clear_incRef_overflow_flag();
                    handle = mm_.peek_for_get_with_hash(key, hash);
                    if (!handle && detail::tls_last_incRef_was_overflow()) {
                        mm_.stats().incRef_overflow_count.value
                            .fetch_add(1, std::memory_order_relaxed);
                    }
                }
                if (!handle) {
                    mm_.stats().register_miss();
                    mm_.callbacks().collect_miss(key);
                    return std::nullopt;
                }
            }
            // Task C: attach per-cache stats if enabled.
            attach_handle_stats_if_enabled(handle);

            // T-P3-6: Fast path for defer_promotion mode — skip write lock
            // entirely, just record access in TLS ring.
            if constexpr (requires { mm_.config().defer_promotion; }) {
                if (mm_.config().defer_promotion) {
                    // Fast path: defer promotion to TLS ring, no lock needed.
                    record_access_in_ring(key);
                    mm_.stats().register_hit();
                    // R8: collect hit callback in defer_promotion mode.
                    mm_.callbacks().collect_hit(key, *handle);
                    return std::optional<read_handle<Value>>{std::move(handle)};
                }
            }

            // Non-defer path: try non-blocking write lock for promotion.
            // T-P3-1: Reuse precomputed hash for write-lock acquisition.
            auto wlock = try_acquire_write_lock_for_hash(hash);
            if (wlock.owns_lock()) {
                mm_.promote(key);
                mm_.stats().register_hit();
                mm_.callbacks().collect_hit(key, *handle);
            } else {
                record_access_in_ring(key);
                mm_.stats().register_hit();
                // R8: collect hit callback in defer path.
                mm_.callbacks().collect_hit(key, *handle);
            }
            return std::optional<read_handle<Value>>{std::move(handle)};
        }
    }

    /// P10: fast_get — ultra-light read path for hot loops.
    ///
    /// Trades correctness guarantees for raw throughput. Compared to
    /// `get()` / `try_get()`, fast_get skips:
    ///   - shutdown check (caller MUST ensure cache is alive)
    ///   - latency timer (no per-call histogram update)
    ///   - callback collection (deferred to next flush_shard_pending)
    ///   - per-handle stats attachment (no handle tracking)
    ///   - write-lock promotion attempt (always defer_promotion)
    ///
    /// What fast_get STILL does:
    ///   - lock-free hash-table lookup (peek_for_get_with_hash)
    ///   - atomic refcount pin (safe under concurrent eviction)
    ///   - inline TTL check (expired items report as miss)
    ///   - hit/miss counter bump (single relaxed atomic add)
    ///   - TLS ring access record (for batched LRU promotion)
    ///
    /// Returns an empty read_handle on miss or shutdown — caller must
    /// check `if (handle)` before dereferencing. Use this in hot read
    /// loops where:
    ///   - The cache is known to be alive (e.g., inside a request handler)
    ///   - Latency tracking is handled at a coarser granularity
    ///   - Callbacks are not required per-read (drained periodically)
    ///   - LRU order does not need to be exact (deferred promotion is OK)
    ///
    /// For all other use cases, prefer `get()` or `try_get()`.
    read_handle<Value> fast_get(const Key& key) {
        // T14.1: Compact path — no fast variant; fall back to get().
        // compact_cache::get() already runs a per-key striped lock with
        // no extra overhead, so the fast path is the same as the slow.
        if constexpr (Trait::is_compact) {
            return get(key);
        }
        // T-P3-1: Precompute hash ONCE for shard dispatch + lookup.
        const std::size_t hash = hash_type{}(key);
        // P10: Inline shard index lookup (no lambda indirection like get()).
        std::size_t shard_idx = 0;
        if constexpr (is_striped) {
            shard_idx = mm_.shard_for_hash(hash);
            // P10: Direct lock-free lookup — no flush_guard, no latency timer.
            auto handle = mm_.shard(shard_idx).peek_for_get_with_hash(key, hash);
            if (!handle) {
                mm_.shard(shard_idx).stats().register_miss();
                return {};
            }
            // P10: Always defer promotion. record_access_in_ring() pushes
            // to a TLS ring (lock-free, no write-lock acquisition). The
            // ring is drained later by the background worker or by the
            // next operation that needs the write lock for other reasons.
            record_access_in_ring(key);
            mm_.shard(shard_idx).stats().register_hit();
            return handle;
        } else {
            // P10: Non-striped path — same lock-free lookup, single shard.
            auto handle = mm_.peek_for_get_with_hash(key, hash);
            if (!handle) {
                mm_.stats().register_miss();
                return {};
            }
            record_access_in_ring(key);
            mm_.stats().register_hit();
            return handle;
        }
    }

    /// Bulk get: look up multiple keys in a single call.
    /// Optimized for read-heavy workloads: all lookups use lock-free
    /// peek_for_get(), and promotions are batched via the TLS ring to
    /// minimize write lock contention.
    ///
    /// @tparam InputIt  Input iterator type over keys.
    /// @param first     Start of key range.
    /// @param last      End of key range.
    /// @return Vector of optional read_handles, one per input key.
    ///         Hit positions are engaged; miss positions are nullopt.
    template <typename InputIt>
    std::vector<std::optional<read_handle<Value>>> bulk_get(InputIt first, InputIt last) {
        if (is_shutdown()) {
            throw cache_closed_exception("unified_cache::bulk_get: cache is shut down");
        }
        const auto count = static_cast<std::size_t>(std::distance(first, last));
        std::vector<std::optional<read_handle<Value>>> results(count);

        // T14.1: Compact path — delegate key-by-key to compact_cache::get().
        // (compact_cache doesn't expose a bulk API, but its per-key get()
        // is already lock-protected and fast for small items.)
        if constexpr (Trait::is_compact) {
            std::size_t idx = 0;
            for (auto it = first; it != last; ++it, ++idx) {
                auto ref = compact().get(*it);
                if (ref) {
                    read_handle<Value> handle(&ref->get(), nullptr, nullptr);
                    results[idx] = std::optional<read_handle<Value>>{std::move(handle)};
                }
            }
            return results;
        }

        // P2-1: For striped caches, group keys by shard and process each
        // shard's keys under a single lock acquisition. This reduces lock
        // contention from O(n) acquisitions to O(num_shards) acquisitions.
        if constexpr (is_striped) {
            // T16.4: Compute the hash ONCE per key and reuse it for both
            // shard dispatch (shard_for_hash) and hash-table lookup
            // (peek_for_get_with_hash). This halves the number of hash
            // computations in bulk_get — previously each key was hashed
            // twice (once in shard_for, once inside find_and_pin_lockfree).
            struct indexed_key {
                std::size_t index;
                Key key;
                std::size_t hash;
            };
            // Group keys by shard index.
            std::unordered_map<std::size_t, std::vector<indexed_key>> shard_groups;
            shard_groups.reserve(mm_.num_shards());
            Hash hash_fn{};
            std::size_t idx = 0;
            for (auto it = first; it != last; ++it, ++idx) {
                const std::size_t h = hash_fn(*it);
                const std::size_t shard_idx = mm_.shard_for_hash(h);
                shard_groups[shard_idx].push_back({idx, *it, h});
            }

            // P1-9: Process each shard group WITHOUT acquiring the shard
            // read lock. peek_for_get_with_hash() is lock-free (hazptr/EBR-
            // protected find_and_pin_lockfree), so the read lock was both
            // unnecessary and harmful:
            //   - Unnecessary: the hash table's lock-free read path handles
            //     concurrent writers safely (hazptr prevents UAF during
            //     bucket-chain traversal).
            //   - Harmful: it serialized all bulk_get reads on the shard's
            //     read lock, defeating the lock-free design and creating
            //     contention with concurrent writers (writer_fair mode
            //     blocks new readers when a writer is waiting).
            // Removing the lock turns bulk_get into a pure lock-free read
            // batch, matching the design of get_prehashed / peek_prehashed.
            //
            // record_access() is also lock-free (TLS ring push) and safe
            // to call here. The drain triggered by a full ring acquires
            // write locks, but those are try_locks (never blocking), so
            // there is no deadlock risk even if we call record_access
            // inside the loop.
            for (auto& [shard_idx, keys] : shard_groups) {
                for (const auto& ik : keys) {
                    auto h = mm_.shard(shard_idx).peek_for_get_with_hash(ik.key, ik.hash);
                    if (h) {
                        mm_.shard(shard_idx).stats().register_hit();
                        record_access_in_ring(ik.key);
                        results[ik.index] = std::optional<read_handle<Value>>{std::move(h)};
                    } else {
                        mm_.shard(shard_idx).stats().register_miss();
                    }
                }
            }
        } else {
            // Non-striped: process sequentially (single lock domain).
            std::size_t idx = 0;
            for (auto it = first; it != last; ++it, ++idx) {
                results[idx] = try_get(*it);
            }
        }
        return results;
    }

    /// Bulk get overload for initializer_list.
    std::vector<std::optional<read_handle<Value>>> bulk_get(std::initializer_list<Key> keys) {
        return bulk_get(keys.begin(), keys.end());
    }

    /// Non-blocking get-or-fetch: like get_or_fetch but uses try_get
    /// semantics for the lookup. If the lookup succeeds, returns the
    /// cached value. If it misses, invokes the provider and inserts.
    /// The provider is called outside the cache lock.
    ///
    /// @param key       Key to look up.
    /// @param provider  Callable Value(const Key&) invoked on miss.
    /// @return The cached or freshly-fetched value.
    template <typename Fn>
        requires std::invocable<Fn, const Key&> &&
                 std::convertible_to<std::invoke_result_t<Fn, const Key&>, Value>
    Value try_get_or_fetch(const Key& key, Fn&& provider) {
        auto h = try_get(key);
        if (h && h->has_value()) {
            return **h;
        }
        // P-HIGH-3 (T-H2): try_get returned nullopt — either a genuine miss
        // or shutdown. Re-check shutdown explicitly because try_get silently
        // returns nullopt on shutdown (non-throwing contract).
        if (is_shutdown()) {
            throw cache_closed_exception(
                "unified_cache::try_get_or_fetch: cache is shut down");
        }
        // Task B: cache shard_idx once per call (computed here so the
        // singleflight follower re-check below can use it).
        const std::size_t shard_idx = [&]() -> std::size_t {
            if constexpr (is_striped) return mm_.shard_for(key);
            else return 0;
        }();
        // T-M1: singleflight / cache stampede protection (see get_or_fetch
        // for full design). Same leader/follower model: leader executes the
        // provider and inserts; followers block on the leader's result.
        typename detail::singleflight_tracker<Key, Value>::state_ptr sf_state;
        bool sf_is_leader = false;
        if (singleflight_enabled_.load(std::memory_order_acquire)) {
            auto sf_result = singleflight_.acquire(key);
            sf_state = sf_result.state;
            sf_is_leader = sf_result.is_leader;
            if (!sf_is_leader) {
                singleflight_.record_coalesced();
                Value leader_v = sf_state->wait_and_get();
                // Re-check the cache for a handle-pinned value. If the
                // leader's insertion failed, return the leader's value.
                if constexpr (is_striped) {
                    read_handle<Value> h2 = mm_.peek_for_get(key);
                    if (h2) {
                        attach_handle_stats_if_enabled(h2);
                        mm_.shard(shard_idx).stats().register_hit();
                        return *h2;
                    }
                } else {
                    flush_guard fg{mm_};
                    read_handle<Value> h2 = mm_.peek_for_get(key);
                    if (h2) {
                        attach_handle_stats_if_enabled(h2);
                        mm_.stats().register_hit();
                        return *h2;
                    }
                }
                return leader_v;
            }
            // Leader: fall through to call provider + insert. After
            // completion (success or exception), signal followers.
        }
        // Miss — call provider outside lock.
        try {
            Value v = std::forward<Fn>(provider)(key);
            // P1-1 (T2.2): Drain TLS ring before write lock — see set().
            maybe_drain_tls_ring_pre_evict();
            // P-HIGH-3 (T-H2): OOM guard — skip insertion in critical mode,
            // return the fetched value without caching.
            if (is_memory_critical()) {
                if (sf_is_leader) {
                    singleflight_.complete(key, sf_state, v);
                }
                return v;
            }
            // Insert under write lock (double-checked). Consolidate all
            // return paths into `result` for singleflight completion.
            Value result;
            if constexpr (is_striped) {
                auto lock = acquire_write_lock_for_key(key);
                // P-HIGH-3 (T-H2): TOCTOU re-check inside the lock.
                if (recheck_shutdown_in_lock()) {
                    throw cache_closed_exception(
                        "unified_cache::try_get_or_fetch: cache is shut down");
                }
                auto handle = mm_.peek_for_get(key);
                if (handle) {
                    mm_.shard(shard_idx).stats().register_hit();
                    mm_.shard(shard_idx).callbacks().collect_hit(key, *handle);
                    flush_shard_pending(shard_idx);
                    result = *handle;
                } else {
                    if (!check_memory_admission(key, v)) {
                        flush_shard_pending(shard_idx);
                        throw cache_oom_exception("try_get_or_fetch: memory monitor rejected admission");
                    }
                    mm_.set(key, v);
                    maybe_report_memory_to_monitor();
                    flush_shard_pending(shard_idx);
                    result = std::move(v);
                }
            } else {
                flush_guard fg{mm_};
                auto lock = acquire_write_lock_for_key(key);
                // P-HIGH-3 (T-H2): TOCTOU re-check inside the lock.
                if (recheck_shutdown_in_lock()) {
                    throw cache_closed_exception(
                        "unified_cache::try_get_or_fetch: cache is shut down");
                }
                auto handle = mm_.peek_for_get(key);
                if (handle) {
                    mm_.stats().register_hit();
                    mm_.callbacks().collect_hit(key, *handle);
                    result = *handle;
                } else {
                    if (!check_memory_admission(key, v)) {
                        throw cache_oom_exception("try_get_or_fetch: memory monitor rejected admission");
                    }
                    mm_.set(key, v);
                    maybe_report_memory_to_monitor();
                    result = std::move(v);
                }
            }
            if (sf_is_leader) {
                singleflight_.complete(key, sf_state, result);
            }
            return result;
        } catch (...) {
            // T-M1: Propagate provider/insertion exception to followers.
            if (sf_is_leader) {
                singleflight_.complete(key, sf_state, std::nullopt,
                                        std::current_exception());
            }
            throw;
        }
    }

    /// Get a value along with its remaining TTL (in nanoseconds).
    ///
    /// For MM types without native TTL support (mm_lru, mm_2q, etc.),
    /// the returned TTL is std::nullopt (i.e., "no expiration"). For
    /// TTL-aware MMs, returns the remaining time-to-live.
    ///
    /// @return std::pair<read_handle<Value>, std::optional<std::uint64_t>>
    ///         — handle is empty on miss; ttl is the remaining TTL in ns,
    ///         or std::nullopt if the item has no expiration.
    std::pair<read_handle<Value>, std::optional<std::uint64_t>> get_with_ttl(const Key& key) {
        read_handle<Value> handle;
        std::optional<std::uint64_t> ttl_ns;
        // Task B: cache shard_idx once per call.
        const std::size_t shard_idx = [&]() -> std::size_t {
            if constexpr (is_striped) return mm_.shard_for(key);
            else return 0;
        }();

        if constexpr (is_striped) {
            handle = mm_.peek_for_get(key);
            if (!handle) {
                mm_.shard(shard_idx).stats().register_miss();
                mm_.shard(shard_idx).callbacks().collect_miss(key);
                flush_shard_pending(shard_idx);
                return {read_handle<Value>{}, std::nullopt};
            }
            // Task C: attach per-cache stats if enabled.
            attach_handle_stats_if_enabled(handle);
            // Try to get TTL if MM supports it.
            if constexpr (requires { mm_.shard(0).ttl_remaining_ns(key); }) {
                ttl_ns = mm_.shard(shard_idx).ttl_remaining_ns(key);
            }
            // Promote (best-effort).
            bool should_try_promote = true;
            if constexpr (requires { mm_.shard(0).config().defer_promotion; }) {
                if (mm_.shard(shard_idx).config().defer_promotion) {
                    should_try_promote = false;
                }
            }
            if (should_try_promote) {
                auto wlock = try_acquire_write_lock_for_key(key);
                if (wlock.owns_lock()) {
                    mm_.promote(key);
                    mm_.shard(shard_idx).stats().register_hit();
                    mm_.shard(shard_idx).callbacks().collect_hit(key, *handle);
                    flush_shard_pending(shard_idx);
                } else {
                    record_access_in_ring(key);
                    mm_.shard(shard_idx).stats().register_hit();
                }
            } else {
                record_access_in_ring(key);
                mm_.shard(shard_idx).stats().register_hit();
            }
        } else {
            flush_guard fg{mm_};
            handle = mm_.peek_for_get(key);
            if (!handle) {
                mm_.stats().register_miss();
                mm_.callbacks().collect_miss(key);
                return {read_handle<Value>{}, std::nullopt};
            }
            // Task C: attach per-cache stats if enabled (parity with striped path).
            attach_handle_stats_if_enabled(handle);
            if constexpr (requires { mm_.ttl_remaining_ns(key); }) {
                ttl_ns = mm_.ttl_remaining_ns(key);
            }
            bool should_try_promote = true;
            if constexpr (requires { mm_.config().defer_promotion; }) {
                if (mm_.config().defer_promotion) {
                    should_try_promote = false;
                }
            }
            if (should_try_promote) {
                auto wlock = try_acquire_write_lock_for_key(key);
                if (wlock.owns_lock()) {
                    mm_.promote(key);
                    mm_.stats().register_hit();
                    mm_.callbacks().collect_hit(key, *handle);
                } else {
                    record_access_in_ring(key);
                    mm_.stats().register_hit();
                }
            } else {
                record_access_in_ring(key);
                mm_.stats().register_hit();
            }
        }
        return {std::move(handle), ttl_ns};
    }

    // --------------------------------------------------------------------
    // T16.3: Pre-hashed API. Callers that already computed `Hash{}(key)`
    // can pass it in to avoid redundant re-hashing across shard dispatch,
    // stripe selection, and hash-table lookup. For string keys this can
    // save ~100ns per operation (2 saved hash computations).
    //
    // IMPORTANT: The caller is responsible for hash compatibility —
    // `hash` MUST be the result of `Hash{}(key)` for the same `Hash`
    // type the cache was instantiated with. Mismatched hashes lead to
    // undefined behavior (wrong shard, wrong stripe, silent data loss).
    // --------------------------------------------------------------------

    /// T16.3: `get` with a pre-computed hash. Equivalent to `get(key)`
    /// but reuses the caller's hash for shard dispatch and stripe locking.
    /// Returns an empty read_handle on miss (consistent with `get()`).
    /// Throws cache_closed_exception on shutdown.
    read_handle<Value> get_prehashed(const Key& key, std::size_t hash) {
        if (is_shutdown()) {
            throw cache_closed_exception("unified_cache::get_prehashed: cache is shut down");
        }
        const std::size_t shard_idx = [&]() -> std::size_t {
            if constexpr (is_striped) return mm_.shard_for_hash(hash);
            else return 0;
        }();
        auto& stats_ref = [&]() -> cache_stats& {
            if constexpr (is_striped) return mm_.shard(shard_idx).stats();
            else return mm_.stats();
        }();
        // O2: capture latency for slow-query threshold check.
        // O1: trace_notifier declared before slow_query_notifier so it
        // destructs after both, reading latency + hit flag.
        std::uint64_t get_latency_ns = 0;
        bool trace_hit = false;
        trace_notifier<unified_cache, Key> trace_n(
            *this, key, trace_op_kind::get, get_latency_ns, trace_hit);
        slow_query_notifier<unified_cache, Key> slow_q_notifier(
            *this, key, slow_op_kind::get, get_latency_ns);
        detail::scope_latency_timer timer_(stats_ref.get_latency,
            stats_ref.latency_tracking_enabled.load(std::memory_order_relaxed),
            &get_latency_ns);

        if constexpr (is_striped) {
            // P1-7: Reuse the caller-supplied hash for the hash-table lookup.
            // The previous code called peek_for_get(key), which re-computes
            // Hash{}(key) internally — defeating the purpose of the pre-hashed
            // API and adding a redundant hash computation on every get_prehashed
            // call. peek_for_get_with_hash routes the same hash through to the
            // underlying concurrent_hash_table, eliminating the double hash.
            auto handle = mm_.peek_for_get_with_hash(key, hash);
            if (!handle) {
                stats_ref.register_miss();
                mm_.shard(shard_idx).callbacks().collect_miss(key);
                flush_shard_pending(shard_idx);
                return {};
            }
            attach_handle_stats_if_enabled(handle);
            trace_hit = true;  // O1: cache hit
            bool should_try_promote = true;
            if constexpr (requires { mm_.shard(0).config().defer_promotion; }) {
                if (mm_.shard(shard_idx).config().defer_promotion) {
                    should_try_promote = false;
                }
            }
            if (should_try_promote) {
                auto wlock = try_acquire_write_lock_for_hash(hash);
                if (wlock.owns_lock()) {
                    mm_.promote(key);
                    stats_ref.register_hit();
                    mm_.shard(shard_idx).callbacks().collect_hit(key, *handle);
                    flush_shard_pending(shard_idx);
                } else {
                    record_access_in_ring(key);
                    stats_ref.register_hit();
                }
            } else {
                record_access_in_ring(key);
                stats_ref.register_hit();
            }
            return handle;
        } else {
            // Non-striped: hash reuse only saves the hash table's internal
            // hash (the unified_cache wrapper doesn't hash for non-striped).
            // Still useful for API symmetry.
            return get(key);
        }
    }

    /// T16.3: `try_get` with a pre-computed hash. Non-throwing variant.
    /// Returns std::nullopt on miss or shutdown; otherwise an engaged
    /// optional containing the read_handle.
    std::optional<read_handle<Value>> try_get_prehashed(const Key& key, std::size_t hash) noexcept {
        if (is_shutdown()) return std::nullopt;
        try {
            auto h = get_prehashed(key, hash);
            if (!h.has_value()) return std::nullopt;
            return std::optional<read_handle<Value>>{std::move(h)};
        } catch (...) {
            return std::nullopt;
        }
    }

    /// T16.3: `set` with a pre-computed hash.
    template <typename V>
    void set_prehashed(const Key& key, std::size_t hash, V&& value) {
        // P-HIGH-3 (T-H2): Unified pre-write validation.
        if (pre_write_check("set_prehashed") == pre_write_result::kRejectInsert) {
            return;
        }
        // P1-1 (T2.2): Drain TLS ring before write lock — see set().
        maybe_drain_tls_ring_pre_evict();
        const std::size_t shard_idx = [&]() -> std::size_t {
            if constexpr (is_striped) return mm_.shard_for_hash(hash);
            else return 0;
        }();
        auto& stats_ref = [&]() -> cache_stats& {
            if constexpr (is_striped) return mm_.shard(shard_idx).stats();
            else return mm_.stats();
        }();
        // O2: capture latency for slow-query threshold check.
        // O1: trace_notifier declared before slow_query_notifier so it
        // destructs after both, reading latency + hit flag.
        std::uint64_t set_latency_ns = 0;
        bool trace_hit = false;
        trace_notifier<unified_cache, Key> trace_n(
            *this, key, trace_op_kind::set, set_latency_ns, trace_hit);
        slow_query_notifier<unified_cache, Key> slow_q_notifier(
            *this, key, slow_op_kind::set, set_latency_ns);
        detail::scope_latency_timer timer_(stats_ref.set_latency,
            stats_ref.latency_tracking_enabled.load(std::memory_order_relaxed),
            &set_latency_ns);

        // P-HIGH-2 (T-H1): Capture OOM event inside the write lock,
        // dispatch the handler *after* lock release — see set().
        std::optional<oom_event> oom_evt;

        if constexpr (is_striped) {
            auto lock = acquire_write_lock_for_hash(hash);
            if (!check_memory_admission(key, value)) return;
            if (!apply_overflow_policy_for_set(key, shard_idx)) return;
            mm_.set_with_hash(key, hash, std::forward<V>(value));
            trace_hit = true;  // O1: set succeeded
            oom_evt = check_memory_pressure();
            maybe_report_memory_to_monitor();
            flush_shard_pending(shard_idx);
        } else {
            flush_guard fg{mm_};
            auto lock = acquire_write_lock_for_key(key);
            if (!check_memory_admission(key, value)) return;
            if (!apply_overflow_policy_for_set(key, 0)) return;
            mm_.set(key, std::forward<V>(value));
            trace_hit = true;  // O1: set succeeded
            oom_evt = check_memory_pressure();
            maybe_report_memory_to_monitor();
        }
        // P-HIGH-2 (T-H1): Dispatch OOM handler outside the write lock.
        dispatch_oom_event(oom_evt);
    }

    /// T16.3: `peek` with a pre-computed hash.
    /// P1-8: Lock-free — the underlying mm_.peek_with_hash() uses
    /// find_and_pin_lockfree() (hazptr/EBR-protected), which is thread-safe
    /// without external locking. The previous implementation acquired a
    /// read lock, which was both unnecessary (the hash table's lock-free
    /// read path already handles concurrent writes safely) and harmful
    /// (it added read-lock contention to a hot read-only path, contradicting
    /// the design of the non-prehashed peek() which is already lock-free).
    std::optional<std::reference_wrapper<const Value>> peek_prehashed(const Key& key, std::size_t hash) const {
        if (is_shutdown()) return std::nullopt;
        if constexpr (is_striped) {
            auto h = mm_.peek_with_hash(key, hash);
            if (!h) return std::nullopt;
            return std::cref(*h);
        } else {
            return peek(key);
        }
    }

    /// T16.3: `contains` with a pre-computed hash.
    /// P1-8: Lock-free — mm_.contains_with_hash() delegates to the hash
    /// table's contains(), which uses optimistic seqlock reads + hazptr
    /// fallback. No external lock needed. The previous read lock added
    /// contention without providing any safety guarantee (contains()
    /// returns bool, no pointer that could become dangling).
    bool contains_prehashed(const Key& key, std::size_t hash) const noexcept {
        if (is_shutdown()) return false;
        if constexpr (is_striped) {
            return mm_.contains_with_hash(key, hash);
        } else {
            return contains(key);
        }
    }

    /// Atomic compare-and-swap: replace the value for `key` only if its
    /// current value equals `expected`. Returns true on success.
    ///
    /// The comparison uses `Value::operator==`. For types without a
    /// suitable equality operator (e.g., containers), this method does
    /// not compile — callers must provide a comparator explicitly via
    /// the overloaded `cas(key, expected, desired, eq_pred)` form.
    ///
    /// @param key       Key to update.
    /// @param expected  Expected current value.
    /// @param desired   New value to install if comparison succeeds.
    /// @return true if the swap succeeded; false if the key was missing
    ///         or its value did not equal `expected`.
    bool cas(const Key& key, const Value& expected, Value desired) {
        return cas(key, expected, std::move(desired),
                   [](const Value& a, const Value& b) { return a == b; });
    }

    /// Atomic compare-and-swap with custom equality predicate.
    /// Useful for value types without operator== or for domain-specific
    /// comparison semantics (e.g., comparing only a version field).
    ///
    /// @param key       Key to update.
    /// @param expected  Expected current value.
    /// @param desired   New value to install if comparison succeeds.
    /// @param eq_pred   Callable: bool(const Value&, const Value&).
    /// @return true if the swap succeeded.
    template <typename Eq>
        requires std::invocable<Eq, const Value&, const Value&> &&
                 std::convertible_to<std::invoke_result_t<Eq, const Value&, const Value&>, bool>
    bool cas(const Key& key, const Value& expected, Value desired, Eq&& eq_pred) {
        // Task B: cache shard_idx once per call.
        const std::size_t shard_idx = [&]() -> std::size_t {
            if constexpr (is_striped) return mm_.shard_for(key);
            else return 0;
        }();
        if constexpr (is_striped) {
            auto lock = acquire_write_lock_for_key(key);
            auto handle = mm_.peek_for_get(key);
            if (!handle) {
                // Key missing — CAS fails. Don't register a miss here to
                // avoid skewing miss-rate metrics with CAS probes.
                return false;
            }
            if (!eq_pred(*handle, expected)) {
                return false;
            }
            if (!check_memory_admission(key, desired)) {
                return false;
            }
            mm_.set(key, std::move(desired));
            maybe_report_memory_to_monitor();
            flush_shard_pending(shard_idx);
            return true;
        } else {
            flush_guard fg{mm_};
            auto lock = acquire_write_lock_for_key(key);
            auto handle = mm_.peek_for_get(key);
            if (!handle) {
                return false;
            }
            if (!eq_pred(*handle, expected)) {
                return false;
            }
            if (!check_memory_admission(key, desired)) {
                return false;
            }
            mm_.set(key, std::move(desired));
            maybe_report_memory_to_monitor();
            return true;
        }
    }

    /// Task A: Atomic compare-and-swap with lock-free predicate evaluation.
    ///
    /// Predicate is invoked OUTSIDE the write lock: the value is peeked
    /// without acquiring the stripe write lock, predicate runs in user
    /// context, and only on predicate==true is the write lock acquired
    /// for final verification + set. On commit-time conflict (value
    /// changed between peek and lock), the whole read-predicate-commit
    /// flow is retried up to `max_retries` times.
    ///
    /// This variant is suited to predicates that may block (IO, RPC,
    /// expensive computation): under kLockedPredicate a slow predicate
    /// would block the entire stripe for the duration of the call,
    /// starving concurrent readers; kLockfreePredicate lets readers
    /// proceed while the predicate runs.
    ///
    /// @param key         Key to update.
    /// @param expected    Expected current value (passed to predicate).
    /// @param desired     New value to install if predicate succeeds.
    /// @param eq_pred     Callable: bool(const Value& current, const Value& expected).
    /// @param policy      Must be cas_policy::kLockfreePredicate.
    /// @param max_retries Max retry attempts on commit conflict (default 3).
    /// @return true if the swap succeeded.
    template <typename Eq>
        requires std::invocable<Eq, const Value&, const Value&> &&
                 std::convertible_to<std::invoke_result_t<Eq, const Value&, const Value&>, bool>
    bool cas(const Key& key, const Value& expected, Value desired, Eq&& eq_pred,
            cas_policy policy, std::size_t max_retries = 3) {
        // The policy parameter selects this overload; only lockfree is valid.
        // (kLockedPredicate callers use the 4-arg overload above.)
        (void)policy;
        // Task B: cache shard_idx once per call.
        const std::size_t shard_idx = [&]() -> std::size_t {
            if constexpr (is_striped) return mm_.shard_for(key);
            else return 0;
        }();

        for (std::size_t attempt = 0; attempt < max_retries; ++attempt) {
            // Phase 1: Lock-free peek. read_handle pins the item (refcount++)
            // so the value cannot be evicted from under us while predicate runs.
            read_handle<Value> peeked;
            if constexpr (is_striped) {
                peeked = mm_.peek_for_get(key);
            } else {
                flush_guard fg{mm_};
                peeked = mm_.peek_for_get(key);
            }
            if (!peeked) {
                // Key missing — CAS fails immediately. No retry can help.
                return false;
            }
            // Task C: attach per-cache stats if enabled.
            attach_handle_stats_if_enabled(peeked);
            // Phase 2: Predicate runs OUTSIDE any stripe write lock.
            // The read_handle keeps the item pinned so the Value& is stable.
            bool pred_ok = false;
            {
                // Copy the value snapshot to insulate from any concurrent
                // mutation that may occur between peek and predicate call
                // (the pinned handle prevents eviction, NOT mutation — only
                // the stripe write lock prevents concurrent writers).
                Value snapshot = *peeked;
                pred_ok = eq_pred(snapshot, expected);
            }
            if (!pred_ok) {
                // Predicate failed — value (likely) doesn't match expected.
                return false;
            }
            // Phase 3: Acquire write lock and re-verify under lock, then commit.
            if constexpr (is_striped) {
                auto lock = acquire_write_lock_for_key(key);
                auto handle = mm_.peek_for_get(key);
                if (!handle) {
                    // Item evicted between peek and lock — retry.
                    continue;
                }
                // Re-check predicate under lock to catch concurrent mutation.
                if (!eq_pred(*handle, expected)) {
                    // Value changed between peek and lock — retry.
                    continue;
                }
                if (!check_memory_admission(key, desired)) {
                    return false;
                }
                mm_.set(key, std::move(desired));
                maybe_report_memory_to_monitor();
                flush_shard_pending(shard_idx);
                return true;
            } else {
                flush_guard fg{mm_};
                auto lock = acquire_write_lock_for_key(key);
                auto handle = mm_.peek_for_get(key);
                if (!handle) {
                    continue;
                }
                if (!eq_pred(*handle, expected)) {
                    continue;
                }
                if (!check_memory_admission(key, desired)) {
                    return false;
                }
                mm_.set(key, std::move(desired));
                maybe_report_memory_to_monitor();
                return true;
            }
        }
        // Exhausted retries — CAS failed due to persistent conflict.
        return false;
    }

    // --------------------------------------------------------------------
    // Batch operations
    // --------------------------------------------------------------------

    /// Batch get: retrieve multiple keys efficiently.
    /// For striped caches, groups keys by stripe so that each stripe's
    /// write lock is acquired once for all keys in that stripe,
    /// reducing lock overhead compared to individual get() calls.
    ///
    /// Returns a vector of optional read_handles in the same order as
    /// the input keys. A disengaged optional indicates the key was not found.
    std::vector<std::optional<read_handle<Value>>> get_multi(std::span<const Key> keys) {
        std::vector<std::optional<read_handle<Value>>> results(keys.size());

        if constexpr (is_striped) {
            // P1-4: group by (stripe, shard) pair. Previously keys were
            // grouped by stripe only and the shard_idx was cached from
            // the first key — that assumed stripe_idx == shard_idx, which
            // no longer holds now that shards use the high hash bits.
            // Using a 64-bit packed key (stripe in low 32, shard in high 32)
            // avoids hash collisions between distinct (stripe, shard) pairs.
            ankerl::unordered_dense::map<std::uint64_t, std::pair<std::size_t, std::vector<std::size_t>>> groups;
            for (std::size_t i = 0; i < keys.size(); ++i) {
                const std::size_t hash = Hash{}(keys[i]);
                const std::size_t stripe = striped_mutex_.stripe_for(hash);
                const std::size_t shard = mm_.shard_for_hash(hash);
                const std::uint64_t key =
                    (static_cast<std::uint64_t>(shard) << 32) |
                    static_cast<std::uint64_t>(stripe);
                groups[key].first = shard;
                groups[key].second.push_back(i);
            }
            for (auto& [k, sv] : groups) {
                const std::size_t stripe = static_cast<std::size_t>(k & 0xFFFFFFFFu);
                const std::size_t shard_idx = sv.first;
                auto& idxs = sv.second;
                auto lock = striped_mutex_.make_unique_lock(stripe);
                for (auto i : idxs) {
                    const auto& key = keys[i];
                    auto h = mm_.peek_for_get(key);
                    if (h) {
                        mm_.promote(key);
                        mm_.shard(shard_idx).stats().register_hit();
                        mm_.shard(shard_idx).callbacks().collect_hit(key, *h);
                        // Task C: attach per-cache stats if enabled.
                        attach_handle_stats_if_enabled(h);
                    } else {
                        mm_.shard(shard_idx).stats().register_miss();
                        mm_.shard(shard_idx).callbacks().collect_miss(key);
                    }
                    if (h) {
                        results[i].emplace(std::move(h));
                    }
                }
                flush_shard_pending(shard_idx);
            }
        } else {
            for (std::size_t i = 0; i < keys.size(); ++i) {
                auto h = get(keys[i]);
                if (h) {
                    results[i].emplace(std::move(h));
                }
            }
        }

        return results;
    }

    /// Batch set: insert multiple key-value pairs efficiently.
    /// For striped caches, groups by (stripe, shard) pair to reduce
    /// lock acquisitions. See get_multi() for the P1-4 rationale on
    /// why we group by both dimensions instead of stripe alone.
    ///
    /// T3.4 bugfix (P1-7): For `sharded_mm_lru` (has_per_shard_lock_v),
    /// we MUST acquire the per-shard write lock — not the stripe lock —
    /// because `sharded_mm_lru::set()` does not internally acquire any
    /// lock; it expects the caller to hold the per-shard write lock.
    /// The previous code only acquired the stripe lock, which is a
    /// DIFFERENT lock and does not protect the MM data structures.
    /// This caused concurrent `set_multi()` and `set()` calls to race
    /// on the same shard, leading to double-retire of evicted items.
    void set_multi(std::span<const std::pair<Key, Value>> pairs) {
        if constexpr (has_per_shard_lock_v<mm_type>) {
            // T3.4 fix: group by shard only, acquire per-shard write lock.
            ankerl::unordered_dense::map<std::size_t, std::vector<std::size_t>> groups;
            for (std::size_t i = 0; i < pairs.size(); ++i) {
                const std::size_t hash = Hash{}(pairs[i].first);
                const std::size_t shard = mm_.shard_for_hash(hash);
                groups[shard].push_back(i);
            }
            for (auto& [shard_idx, idxs] : groups) {
                auto lock = mm_.acquire_shard_write_lock(shard_idx);
                for (auto i : idxs) {
                    const auto& [key, value] = pairs[i];
                    if (!check_memory_admission(key, value)) continue;
                    mm_.set(key, value);
                }
                maybe_report_memory_to_monitor();
                flush_shard_pending(shard_idx);
            }
        } else if constexpr (is_striped) {
            ankerl::unordered_dense::map<std::uint64_t, std::pair<std::size_t, std::vector<std::size_t>>> groups;
            for (std::size_t i = 0; i < pairs.size(); ++i) {
                const std::size_t hash = Hash{}(pairs[i].first);
                const std::size_t stripe = striped_mutex_.stripe_for(hash);
                const std::size_t shard = mm_.shard_for_hash(hash);
                const std::uint64_t key =
                    (static_cast<std::uint64_t>(shard) << 32) |
                    static_cast<std::uint64_t>(stripe);
                groups[key].first = shard;
                groups[key].second.push_back(i);
            }
            for (auto& [k, sv] : groups) {
                const std::size_t stripe = static_cast<std::size_t>(k & 0xFFFFFFFFu);
                const std::size_t shard_idx = sv.first;
                auto& idxs = sv.second;
                auto lock = striped_mutex_.make_unique_lock(stripe);
                for (auto i : idxs) {
                    const auto& [key, value] = pairs[i];
                    if (!check_memory_admission(key, value)) continue;
                    mm_.set(key, value);
                }
                maybe_report_memory_to_monitor();
                flush_shard_pending(shard_idx);
            }
        } else {
            for (const auto& [key, value] : pairs) {
                set(key, value);
            }
        }
    }

    /// Get a shared_ptr copy of the value.
    /// Unlike get(), returns a value copy wrapped in shared_ptr, does not
    /// depend on LRU position protection, and does not change LRU order.
    ///
    /// NOTE: Each call performs a heap allocation (std::make_shared) + value copy.
    /// For high-throughput read-heavy workloads, prefer:
    ///   - get(key): returns read_handle<Value> (zero heap allocation, pin-based safety)
    ///   - get_shared_cached(key): TLS-cached shared_ptr (avoids allocation on repeated keys)
    ///   - get_view(key): returns a lightweight view (zero allocation, but reference is
    ///     only valid while the item remains in the cache)
    ///
    /// The returned shared_ptr holds a copy of the value, safe to use even if the
    /// cache entry is later evicted. Returns nullptr on cache miss.
    std::shared_ptr<Value> get_shared(const Key& key) {
        // Task B: cache shard_idx once per call.
        const std::size_t shard_idx = [&]() -> std::size_t {
            if constexpr (is_striped) return mm_.shard_for(key);
            else return 0;
        }();
        auto h = mm_.peek_for_get(key);
        if (h) {
            // Task C: attach per-cache stats if enabled.
            attach_handle_stats_if_enabled(h);
        }
        if (!h) {
            if constexpr (is_striped) {
                mm_.shard(shard_idx).stats().register_miss();
                mm_.shard(shard_idx).callbacks().collect_miss(key);
                flush_shard_pending(shard_idx);
            } else {
                mm_.stats().register_miss();
                mm_.callbacks().collect_miss(key);
            }
            return nullptr;
        }
        if constexpr (is_striped) {
            mm_.shard(shard_idx).stats().register_hit();
            mm_.shard(shard_idx).callbacks().collect_hit(key, *h);
            flush_shard_pending(shard_idx);
        } else {
            mm_.stats().register_hit();
            mm_.callbacks().collect_hit(key, *h);
        }
        return std::make_shared<Value>(*h);
    }

    /// Get a non-owning view of the value. No heap allocation.
    /// For std::string values, returns std::optional<std::string_view>.
    /// For std::vector<T> values, returns std::optional<std::span<const T>>.
    /// For other types, returns std::optional<std::reference_wrapper<const Value>>.
    ///
    /// WARNING: The returned view/reference is only valid while the item
    /// remains in the cache. If the item is evicted, the view becomes dangling.
    /// Use get() for safe access via read_handle.
    auto get_view(const Key& key) const {
        // Lock-free: peek() uses find_and_pin() to atomically pin the item.
        // The read_handle h keeps the item alive during view construction.
        auto h = mm_.peek(key);
        if (!h) {
            if constexpr (std::is_same_v<Value, std::string>) {
                return std::optional<std::string_view>{};
            } else if constexpr (detail::is_vector_v<Value>) {
                return std::optional<std::span<const typename Value::value_type>>{};
            } else {
                return std::optional<std::reference_wrapper<const Value>>{};
            }
        }
        if constexpr (std::is_same_v<Value, std::string>) {
            return std::optional<std::string_view>(std::string_view(*h));
        } else if constexpr (detail::is_vector_v<Value>) {
            return std::optional<std::span<const typename Value::value_type>>(*h);
        } else {
            return std::optional<std::reference_wrapper<const Value>>(std::cref(*h));
        }
    }

    /// Get a shared_ptr copy of the value with TLS caching.
    /// If the same key was recently accessed via get_shared_cached(),
    /// the cached shared_ptr is returned without heap allocation.
    /// The cache is per-thread and holds at most one entry.
    ///
    /// This is the recommended shared-ptr access method for read-heavy
    /// workloads where the same key is frequently read.
    std::shared_ptr<Value> get_shared_cached(const Key& key) {
        auto& cached = detail::tls_shared_cache<Key, Value>::instance();
        if (cached.key == key && cached.ptr) {
            return cached.ptr;
        }

        auto result = get_shared(key);
        if (result) {
            cached.key = key;
            cached.ptr = result;
        }
        return result;
    }

    bool del(const Key& key) {
        if constexpr (is_striped) {
            auto lock = acquire_write_lock_for_key(key);
            auto result = mm_.del(key);
            maybe_report_memory_to_monitor();
            flush_shard_pending(key);
            return result;
        } else {
            flush_guard fg{mm_};
            auto lock = acquire_write_lock_for_key(key);
            auto result = mm_.del(key);
            maybe_report_memory_to_monitor();
            return result;
        }
    }

    /// Force delete a key even if it has active handles.
    /// The item is removed from the cache immediately; memory is
    /// freed when all outstanding handles are released.
    bool force_del(const Key& key) {
        if constexpr (is_striped) {
            auto lock = acquire_write_lock_for_key(key);
            bool result;
            if constexpr (requires { mm_.force_del(key); }) {
                result = mm_.force_del(key);
            } else {
                result = mm_.del(key);
            }
            maybe_report_memory_to_monitor();
            flush_shard_pending(key);
            return result;
        } else {
            flush_guard fg{mm_};
            auto lock = acquire_write_lock_for_key(key);
            bool result;
            if constexpr (requires { mm_.force_del(key); }) {
                result = mm_.force_del(key);
            } else {
                result = mm_.del(key);
            }
            maybe_report_memory_to_monitor();
            return result;
        }
    }

    /// Extended delete result with precise failure reason.
    enum class DelResult {
        kSuccess,   ///< Key was deleted successfully
        kNotFound,  ///< Key was not in the cache
        kPinned,    ///< Key has active handles; use force_del() to override
    };

    /// Delete with extended result code.
    DelResult del_ex(const Key& key) {
        if constexpr (is_striped) {
            auto lock = acquire_write_lock_for_key(key);
            DelResult result;
            if (!mm_.contains(key)) {
                result = DelResult::kNotFound;
            } else if (mm_.del(key)) {
                result = DelResult::kSuccess;
            } else {
                result = DelResult::kPinned;
            }
            if (result == DelResult::kSuccess) {
                maybe_report_memory_to_monitor();
            }
            flush_shard_pending(key);
            return result;
        } else {
            flush_guard fg{mm_};
            auto lock = acquire_write_lock_for_key(key);
            // First check if key exists
            if (!mm_.contains(key)) {
                return DelResult::kNotFound;
            }
            // Try normal delete
            if (mm_.del(key)) {
                maybe_report_memory_to_monitor();
                return DelResult::kSuccess;
            }
            // Key exists but del failed - must be pinned
            return DelResult::kPinned;
        }
    }

    // B16: 返回 RemoveRes 的删除操作
    RemoveRes remove(const Key& key) {
        // T14.1: Compact path.
        if constexpr (Trait::is_compact) {
            bool removed = compact().del(key);
            return removed ? RemoveRes::kSuccess : RemoveRes::kNotFound;
        }
        if constexpr (is_striped) {
            auto lock = acquire_write_lock_for_key(key);
            auto result = mm_.del(key) ? RemoveRes::kSuccess : RemoveRes::kNotFound;
            if (result == RemoveRes::kSuccess) {
                maybe_report_memory_to_monitor();
            }
            flush_shard_pending(key);
            return result;
        } else {
            flush_guard fg{mm_};
            auto lock = acquire_write_lock_for_key(key);
            auto result = mm_.del(key) ? RemoveRes::kSuccess : RemoveRes::kNotFound;
            if (result == RemoveRes::kSuccess) {
                maybe_report_memory_to_monitor();
            }
            return result;
        }
    }

    /// Pop a key from the cache, removing it and returning its value.
    /// Unlike del(), this does NOT fire eviction callbacks.
    /// Returns std::nullopt if the key is not found.
    ///
    /// All current MM types implement pop(); this function requires that method.
    std::optional<Value> pop(const Key& key) {
        if constexpr (is_striped) {
            auto lock = acquire_write_lock_for_key(key);
            std::optional<Value> result;
            if constexpr (requires { mm_.pop(key); }) {
                result = mm_.pop(key);
            } else {
                static_assert(sizeof(mm_) == 0, "mm_type does not support pop()");
            }
            if (result) {
                maybe_report_memory_to_monitor();
            }
            flush_shard_pending(key);
            return result;
        } else {
            flush_guard fg{mm_};
            auto lock = acquire_write_lock_for_key(key);
            if constexpr (requires { mm_.pop(key); }) {
                auto result = mm_.pop(key);
                if (result) {
                    maybe_report_memory_to_monitor();
                }
                return result;
            } else {
                static_assert(sizeof(mm_) == 0, "mm_type does not support pop()");
            }
        }
    }

    /// Pop the LRU item from the cache. Unlike eviction, this does NOT fire
    /// eviction callbacks. Returns std::nullopt if the cache is empty.
    std::optional<std::pair<Key, Value>> pop_lru() {
        // P1-1 (T2.2): Drain TLS ring before write lock — see set().
        maybe_drain_tls_ring_pre_evict();
        flush_guard fg{mm_};
        auto lock = acquire_write_lock();
        if constexpr (requires { mm_.pop_lru(); }) {
            auto result = mm_.pop_lru();
            if (result) {
                maybe_report_memory_to_monitor();
            }
            return result;
        } else {
            static_assert(sizeof(mm_) == 0, "mm_type does not support pop_lru()");
        }
    }

    /// Evict one item according to the MM strategy. This fires eviction callbacks.
    /// Returns true if an item was evicted.
    bool evict() {
        // P1-1 (T2.2): Drain TLS ring before write lock — see set().
        maybe_drain_tls_ring_pre_evict();
        flush_guard fg{mm_};
        auto lock = acquire_write_lock();
        if constexpr (requires { mm_.evict_lru(); }) {
            auto size_before = mm_.size();
            mm_.evict_lru();
            bool evicted = mm_.size() < size_before;
            if (evicted) {
                maybe_report_memory_to_monitor();
            }
            return evicted;
        } else if constexpr (requires { mm_.evict(); }) {
            auto size_before = mm_.size();
            mm_.evict();
            bool evicted = mm_.size() < size_before;
            if (evicted) {
                maybe_report_memory_to_monitor();
            }
            return evicted;
        } else {
            static_assert(sizeof(mm_) == 0, "mm_type does not support eviction");
        }
    }

    void flush() {
        // T14.1: Compact path.
        if constexpr (Trait::is_compact) {
            compact().flush();
            return;
        }
        drain_access_ring();
        tls_access_ring<Key>::flush_all_registered();
        tls_callback_ring<Key, Value>::flush_all_registered();
        flush_guard fg{mm_};
        auto lock = acquire_write_lock();
        // P1-8 (T2.6 bugfix, phase 3): `mm_.flush()` now refreshes hash
        // stats internally (between Pass 1 and Pass 2, before any items
        // are retired). The previous external `mm_.refresh_hash_stats()`
        // call was removed because it traversed hash chains AFTER items
        // were retired — the background `periodic_worker` could free
        // retired items via `try_reclaim_now()` while `max_chain_length()`
        // was traversing the chain → heap-use-after-free.
        mm_.flush();
        maybe_report_memory_to_monitor();
    }

    // --------------------------------------------------------------------
    // Reserve — pre-allocate hash table to avoid runtime rehash
    // --------------------------------------------------------------------

    /// Pre-allocate hash table buckets for `expected_items` entries.
    /// Call this before inserting a large number of items to prevent
    /// runtime rehash stalls (which cause latency spikes under load).
    /// No-op if the current bucket count is already sufficient.
    ///
    /// Example:
    ///   cache c(100000);
    ///   c.reserve(100000);  // Pre-allocate, no rehash during fill
    ///   for (int i = 0; i < 100000; ++i) c.set(i, value);
    ///
    /// T3.4 bugfix (P1-7): For `sharded_mm_lru`, acquire per-shard write
    /// locks before calling `reserve()` on each shard — otherwise
    /// concurrent `set()` calls on the same shard race with the hash
    /// table rehash, corrupting internal bucket state.
    void reserve(size_type expected_items) {
        if constexpr (has_per_shard_lock_v<mm_type>) {
            // T3.4 fix: lock each shard before resizing its hash table.
            const std::size_t n = mm_.num_shards();
            const std::size_t per_shard =
                std::max(size_type(1), expected_items / std::max(size_type(1), n));
            for (std::size_t i = 0; i < n; ++i) {
                auto lock = mm_.acquire_shard_write_lock(i);
                mm_.shard(i).reserve(per_shard);
            }
        } else {
            mm_.reserve(expected_items);
        }
    }

    /// Enable/disable incremental rehash for the underlying hash table.
    /// When enabled, hash table expansion migrates buckets incrementally across
    /// multiple insert/erase operations instead of blocking all writers during a
    /// single rehash. This significantly reduces write-path latency spikes under
    /// load, at the cost of a brief dual-array lookup overhead during migration.
    ///
    /// Recommended for production read-heavy-write-light workloads where
    /// write-path P99 latency matters. Should be called before the cache fills up.
    ///
    /// T19.2: Effective for ALL hash table modes:
    ///   - **Chain mode** (`chain_probing_tag`): migrates one bucket at a time
    ///     per insert/erase, draining the old bucket list into the new array.
    ///   - **F14 SIMD mode** (`f14_probing_tag`): migrates one 14-slot chunk at
    ///     a time via `rehash_step_f14`; lookups query both old and new arrays
    ///     during migration (dual-array lookup).
    ///   - **Segmented mode** (`Segmented=true`, used by `production_cache`):
    ///     applies the chain-mode incremental rehash independently per segment,
    ///     so only one of 64 segments is stalled at any moment.
    ///
    /// When disabled (the default for non-production aliases), rehash acquires
    /// all bucket locks in a single blocking pass. The `segmented_*` and
    /// `production_*` aliases auto-enable incremental rehash at construction
    /// (see `cache_trait::auto_incremental_rehash`).
    void set_incremental_rehash(bool enabled) {
        mm_.set_incremental_rehash(enabled);
    }

    /// Query whether incremental rehash is enabled.
    bool incremental_rehash_enabled() const noexcept {
        return mm_.incremental_rehash_enabled();
    }

    // --------------------------------------------------------------------
    // T2.1 / T2.4: EBR (Epoch-Based Reclamation) integration
    // --------------------------------------------------------------------

    /// T2.1: Set the EBR domain for deferred reclamation. When non-null,
    /// the read path (find_and_pin_lockfree) acquires an epoch_guard at
    /// entry, protecting all nodes from reclamation during traversal.
    /// Evicted items are retired via the domain's retire() instead of
    /// hazptr. Propagated to all shards for sharded caches.
    ///
    /// T2.4: Use is_ebr_mode() to check if EBR is active. Callers can
    /// use concurrent_hash_table::reclaim_guard for mode-agnostic code.
    ///
    /// Note: Only mm_lru and sharded_mm_lru support EBR. Calling this
    /// on other MM types (mm_2q, mm_tiny_lfu, mm_wtiny_lfu, mm_fifo)
    /// is a no-op (guarded by SFINAE).
    void set_ebr_domain(detail::epoch_domain* domain) {
        if constexpr (is_striped) {
            if constexpr (requires { mm_.shard(0).set_ebr_domain(domain); }) {
                for (std::size_t i = 0; i < mm_.num_shards(); ++i) {
                    mm_.shard(i).set_ebr_domain(domain);
                }
            }
            (void)domain;
        } else {
            if constexpr (requires { mm_.set_ebr_domain(domain); }) {
                mm_.set_ebr_domain(domain);
            }
            (void)domain;
        }
    }

    /// T2.4: Check whether EBR mode is active (domain has been set).
    /// Returns false for MM types that don't support EBR.
    bool is_ebr_mode() const noexcept {
        if constexpr (is_striped) {
            if constexpr (requires { mm_.shard(0).is_ebr_mode(); }) {
                return mm_.num_shards() > 0 ? mm_.shard(0).is_ebr_mode() : false;
            } else {
                return false;
            }
        } else {
            if constexpr (requires { mm_.is_ebr_mode(); }) {
                return mm_.is_ebr_mode();
            } else {
                return false;
            }
        }
    }

    /// R9 (per-shard EBR): Enable an independent epoch_domain per shard.
    ///
    /// Each shard gets its own EBR domain, so reclamation on one shard is
    /// scoped to that shard's readers instead of the whole cache. This is
    /// most beneficial for write-heavy / high-churn caches where the global
    /// pending list and min-epoch scan become a cross-shard serialization
    /// point.
    ///
    /// Trade-off (read-heavy): the epoch_domain keeps a SINGLE per-thread TLS
    /// slot cache. Under the default global domain, a thread that reads keys
    /// spread across many shards keeps one stable slot. With per-shard
    /// domains, that thread's TLS slot owner alternates across domains on
    /// each cross-shard read, causing slot acquire/release churn. For pure
    /// read-heavy-write-light workloads the global domain is usually the
    /// better default; enable per-shard EBR when reclamation isolation
    /// matters more than stable read-path slot caching.
    ///
    /// No-op for non-striped MMs or MMs without per-shard EBR support.
    /// Returns true if per-shard EBR was enabled.
    bool enable_per_shard_ebr() {
        if (!is_striped) return false;
        if constexpr (!requires { mm_.shard(0).set_ebr_domain(nullptr); }) {
            return false;
        }
        const std::size_t n = mm_.num_shards();
        if (n == 0) return false;
        // (Re)create one domain per shard. Existing domains are reused for
        // the already-created shards to preserve any runtime config applied
        // to them; only newly needed domains are allocated.
        per_shard_ebr_domains_.resize(n);
        for (std::size_t i = 0; i < n; ++i) {
            if (!per_shard_ebr_domains_[i]) {
                auto dom = std::make_unique<detail::epoch_domain>();
                dom->mark_drain_started();
                per_shard_ebr_domains_[i] = std::move(dom);
            }
            mm_.shard(i).set_ebr_domain(per_shard_ebr_domains_[i].get());
        }
        per_shard_ebr_enabled_ = true;
        return true;
    }

    /// R9: True when per-shard EBR domains are active.
    bool per_shard_ebr_enabled() const noexcept { return per_shard_ebr_enabled_; }

    /// R9: Number of active EBR domains (1 global default, or N per-shard).
    std::size_t ebr_domain_count() const noexcept {
        return per_shard_ebr_enabled_ ? per_shard_ebr_domains_.size() : 1;
    }

private:
    /// R9: Invoke `fn(detail::epoch_domain&)` on every active EBR domain.
    /// When per-shard EBR is enabled, this iterates the per-shard domains;
    /// otherwise it calls fn on the global default domain exactly once.
    /// Used by the reclaim worker and reclamation metrics so per-shard
    /// domains are advanced/drained/aggregated correctly.
    template <typename Fn>
        requires std::invocable<Fn, detail::epoch_domain&>
    void for_each_ebr_domain(Fn&& fn) const {
        if (per_shard_ebr_enabled_) {
            for (const auto& dom : per_shard_ebr_domains_) {
                if (dom) fn(*dom);
            }
        } else {
            fn(detail::epoch_domain::default_domain());
        }
    }

    /// R9: Aggregate pending (unreclaimed) object count across all active
    /// EBR domains.
    std::size_t ebr_pending_count() const noexcept {
        std::size_t total = 0;
        for_each_ebr_domain([&](detail::epoch_domain& dom) {
            total += dom.pending_count();
        });
        return total;
    }

    /// R9: Aggregate total objects reclaimed across all active EBR domains.
    std::size_t ebr_reclaim_total() const noexcept {
        std::size_t total = 0;
        for_each_ebr_domain([&](detail::epoch_domain& dom) {
            total += dom.reclaim_total();
        });
        return total;
    }

    /// R9: Aggregate force-advance count (stuck-slot skips) across all
    /// active EBR domains.
    std::size_t ebr_force_advance_count() const noexcept {
        std::size_t total = 0;
        for_each_ebr_domain([&](detail::epoch_domain& dom) {
            total += dom.force_advance_count();
        });
        return total;
    }

public:

    /// T-G4: Query the EBR force-advance policy of the default epoch
    /// domain. `kFailAdvance` (default) refuses to advance the epoch
    /// when a slot is stuck, preventing UAF at the cost of un-reclaimed
    /// memory. See `detail::force_advance_policy` for the full semantics.
    detail::force_advance_policy epoch_force_advance_policy() const noexcept {
        return get_force_advance_policy();
    }

    /// T-G4: Set the EBR force-advance policy on the default epoch
    /// domain. Operators can opt in to `kForceAdvanceAfter5s` (old
    /// behavior) when they accept the UAF risk in exchange for
    /// guaranteed reclamation progress. R9: propagates to every active
    /// EBR domain (global default or per-shard domains).
    void set_epoch_force_advance_policy(detail::force_advance_policy policy) noexcept {
        // P0-4: Surface the memory-safety risk of force-advance policies
        // loudly. kForceAdvanceAfter5s / kForceAdvanceAfterNs skip stuck
        // TLS slots after a timeout; a thread descheduled past the timeout
        // (GC pause, SIGSTOP, cgroup throttle) can later dereference a
        // reclaimed object. This is an explicit opt-in for guaranteed
        // reclamation progress at the cost of UAF. kFailAdvance (the
        // default) never reclaims past a stuck slot, so it is safe and
        // needs no warning.
        if (policy != detail::force_advance_policy::kFailAdvance) {
            std::fprintf(stderr,
                "[lru] WARNING: EBR force-advance policy switched away from "
                "kFailAdvance. Stuck threads may now trigger use-after-free; "
                "only use this when guaranteed reclamation progress is worth "
                "the memory-safety risk.\n");
            std::fflush(stderr);
        }
        set_force_advance_policy(policy);
    }

    /// T11.5: Set the rehash strategy by name. String-based API for
    /// configuration files / CLI flags. Equivalent to:
    ///   - "incremental": set_incremental_rehash(true)
    ///   - "blocking":    set_incremental_rehash(false)
    /// Returns false if the strategy name is unrecognized (current setting
    /// is left unchanged). Case-insensitive. Propagated to all shards.
    bool set_rehash_strategy(std::string_view strategy) {
        if constexpr (is_striped) {
            bool ok = true;
            for (std::size_t i = 0; i < mm_.num_shards(); ++i) {
                if (!mm_.shard(i).set_rehash_strategy(strategy)) ok = false;
            }
            return ok;
        } else {
            return mm_.set_rehash_strategy(strategy);
        }
    }

    /// T11.5: Query the current rehash strategy by name.
    std::string_view rehash_strategy() const noexcept {
        if constexpr (is_striped) {
            return mm_.shard(0).rehash_strategy();
        } else {
            return mm_.rehash_strategy();
        }
    }

    /// T11.3: Number of writes blocked by a non-incremental (blocking) rehash.
    /// Non-zero values indicate the user should enable incremental rehash to
    /// avoid stalling writers during hash table expansion. Aggregated across
    /// all shards for striped caches.
    std::size_t rehash_blocked_writes_count() const noexcept {
        if constexpr (is_striped) {
            std::size_t total = 0;
            for (std::size_t i = 0; i < mm_.num_shards(); ++i) {
                total += mm_.shard(i).rehash_blocked_writes_count();
            }
            return total;
        } else {
            return mm_.rehash_blocked_writes_count();
        }
    }

    /// P1-5: Number of times find_and_pin_lockfree fell back to the
    /// lock-protected path because the target segment was in incremental
    /// rehash. Aggregated across all shards for striped caches.
    std::size_t rehash_lockfree_fallback_count() const noexcept {
        if constexpr (is_striped) {
            std::size_t total = 0;
            for (std::size_t i = 0; i < mm_.num_shards(); ++i) {
                total += mm_.shard(i).rehash_lockfree_fallback_count();
            }
            return total;
        } else {
            return mm_.rehash_lockfree_fallback_count();
        }
    }

    /// P0-D: Fraction of the hash table currently in an incremental rehash.
    /// For segmented tables this is the average fraction of segments
    /// rehashing across all shards; for non-segmented tables it is 0.0
    /// (idle) or 1.0 (rehashing). Exposed as a Prometheus gauge so
    /// operators can detect sustained rehash pressure (a value near 1.0
    /// for an extended period means writes are stalling too often).
    float rehash_in_progress_ratio() const noexcept {
        if constexpr (is_striped) {
            if (mm_.num_shards() == 0) return 0.0f;
            float sum = 0.0f;
            for (std::size_t i = 0; i < mm_.num_shards(); ++i) {
                sum += mm_.shard(i).rehash_in_progress_ratio();
            }
            return sum / static_cast<float>(mm_.num_shards());
        } else {
            return mm_.rehash_in_progress_ratio();
        }
    }

    // --------------------------------------------------------------------
    // T13: Hash overload detection and callback
    // --------------------------------------------------------------------

    /// T13.1: Set the hash table load factor threshold above which the
    /// overload callback fires and `hash_overload_events` is incremented.
    /// Default: 2.0. Lower this for latency-sensitive workloads (e.g.
    /// 1.5) to get earlier warning of capacity pressure; raise it for
    /// memory-frugal workloads that tolerate longer chains.
    ///
    /// For sharded caches, the threshold is propagated to all shards.
    void set_hash_overload_threshold(float threshold) {
        if constexpr (is_striped) {
            for (std::size_t i = 0; i < mm_.num_shards(); ++i) {
                mm_.shard(i).set_hash_overload_threshold(threshold);
            }
        } else {
            mm_.set_hash_overload_threshold(threshold);
        }
    }

    /// T13.2: Register a callback invoked when the hash table load
    /// factor exceeds `hash_overload_threshold`. The callback receives
    /// (current_load_factor, threshold) and is called from the rehash
    /// hot path — it must be cheap and non-blocking. Exceptions are
    /// swallowed to protect the rehash path.
    ///
    /// Typical uses: log/alert on overload, dynamically resize the
    /// cache, throttle admissions.
    ///
    /// For sharded caches, the same callback is propagated to all
    /// shards. Each shard fires its own callback invocation when its
    /// individual load factor exceeds the threshold.
    void set_overload_callback(std::function<void(float, float)> cb) {
        if constexpr (is_striped) {
            for (std::size_t i = 0; i < mm_.num_shards(); ++i) {
                mm_.shard(i).set_overload_callback(cb);
            }
        } else {
            mm_.set_overload_callback(std::move(cb));
        }
    }

    /// P2-4 (T2.4): Toggle async mode for the overload callback on every
    /// shard's hash table. See `concurrent_hash_table::set_async_overload_callback`
    /// for semantics. When enabled, the rehash hot path enqueues overload
    /// events instead of invoking the user callback inline; a background
    /// worker (driven by `start_event_drain()`) drains the queue and
    /// dispatches the callback off the hot path.
    ///
    /// Use async mode whenever the registered callback may block (file
    /// IO, network IO, heap allocation, metrics push). Sync mode (default)
    /// is appropriate for trivially cheap callbacks (atomic counter, etc.).
    void set_async_overload_callback(bool enabled) {
        if constexpr (is_striped) {
            for (std::size_t i = 0; i < mm_.num_shards(); ++i) {
                mm_.shard(i).set_async_overload_callback(enabled);
            }
        } else {
            mm_.set_async_overload_callback(enabled);
        }
    }

    /// P2-4 (T2.4): Drain pending overload events from every shard's hash
    /// table and dispatch the registered callback for each. Returns the
    /// total number of events drained across all shards. Designed to be
    /// called from a background worker (the `event_drain_worker` invokes
    /// this automatically when `start_event_drain()` has been called).
    std::size_t drain_overload_callbacks() {
        std::size_t total = 0;
        if constexpr (is_striped) {
            for (std::size_t i = 0; i < mm_.num_shards(); ++i) {
                total += mm_.shard(i).drain_overload_callbacks();
            }
        } else {
            total = mm_.drain_overload_callbacks();
        }
        return total;
    }

    // --------------------------------------------------------------------
    // Task E: diagnostics() — consolidated runtime introspection
    // --------------------------------------------------------------------
    //
    // A single struct aggregating the lock-contention, hash-table, TLS-ring,
    // and handle counters needed to diagnose production read-heavy issues
    // (writer starvation, rehash stalls, TLS backlog overflow, handle
    // leaks). All fields are point-in-time atomic reads — call this on the
    // hot path is safe but not free; expect ~1μs for sharded caches with
    // 64 stripes. For periodic scraping, prefer `diagnostics_text()`.

    struct diagnostics_info {
        // Per-shard (per-stripe) lock contention.
        // For non-sharded caches, vectors contain a single element.
        std::vector<std::size_t> per_stripe_wait_count;       // sum of write_lock_wait_count
        std::vector<std::size_t> per_stripe_try_fail_count;  // sum of try_lock_fail_count
        std::vector<std::size_t> per_shard_size;             // items per shard
        std::vector<std::size_t> per_shard_bucket_count;     // hash table buckets per shard

        // P1-2: Per-shard fine-grained metrics for hotspot detection.
        std::vector<std::size_t> per_shard_hits;             // hits per shard
        std::vector<std::size_t> per_shard_misses;           // misses per shard
        std::vector<std::size_t> per_shard_evictions;        // evictions per shard
        std::vector<std::size_t> per_shard_memory;           // memory usage per shard (bytes)
        std::vector<std::size_t> per_shard_rehash_count;     // rehash operations per shard

        // Aggregated rehash state (sums across shards).
        bool rehash_in_progress{false};
        std::size_t rehash_progress_buckets{0};       // migrated buckets
        std::size_t rehash_old_bucket_count{0};        // total old buckets (denominator)
        std::size_t rehash_new_bucket_count{0};        // target new bucket count

        // TLS access ring (calling thread's view).
        std::size_t tls_ring_backlog{0};
        std::size_t tls_ring_dropped_promotions{0};  // cumulative
        // T-M4: Cross-thread TLS backlog aggregate. While `tls_ring_backlog`
        // reflects only the calling thread's ring (the per-thread view),
        // `tls_ring_backlog_total` is the aggregate across all threads
        // — the metric operators actually need to detect backlog overflow
        // (a non-zero value here with flat drain rate signals the drain
        // worker is under-provisioned). Read from
        // tls_access_ring<Key>::total_backlog(). As of T-P2-4 the hot
        // path updates a thread-local counter and batch-flushes to the
        // shared atomic every 64 increments, so this value has bounded
        // staleness (<= 64 per thread) — sufficient for overflow
        // detection, which only needs trend-level accuracy.
        std::size_t tls_ring_backlog_total{0};
        // T-M4: Per-shard retire-pending snapshot. Hazptr/EBR domains are
        // global by default (retired objects go to one shared domain), so
        // for the default configuration every shard will report the same
        // value (the global pending count). This vector is sized
        // num_shards and is intended for forward compatibility with
        // per-shard retire queues (already plumbed via shard_hotspot) and
        // for operators who configure per-shard domains. Reads use the
        // same global snapshot as reclaim_pending_count for consistency.
        std::vector<std::size_t> per_shard_retire_pending;

        // Handle accounting.
        std::size_t active_handle_count{0};

        // Configuration flags in effect.
        bool defer_promotion_enabled{false};
        bool incremental_rehash_enabled{false};
        std::size_t num_shards{1};

        // T19.3: Probing / hashing mode in effect. Operators reading a
        // diagnostics dump need to know whether the cache is running in
        // chain mode, F14 SIMD mode, or segmented mode, because the rehash
        // behaviour and the meaning of bucket_count differ across modes.
        bool f14_probing{false};        // true if F14 SIMD probing is active
        bool segmented_hash_table{false}; // true if 64-segment table is active
        bool compressed_hook{false};     // true if compressed_intrusive_hook is in use
        // T-P1-3 (R-3): EmbeddedChain mode. When false, the hash table
        // degrades all lock-free read paths to shared_lock fallback
        // (use-after-free prevention), killing read throughput under high
        // concurrency. production_cache and all MM strategies assert this
        // is true at compile time; surfacing it in diagnostics lets
        // operators verify at runtime that no custom trait has bypassed
        // the assertion. See AGENTS.md "Non-EmbeddedChain shared lock
        // fallback" hard constraint.
        bool embedded_chain{false};      // true if hash table uses EmbeddedChain (lock-free reads)

        // T17.3: Deferred-reclamation health (hazptr + EBR).
        // These mirror the fields already exposed via stats_snapshot() /
        // prometheus_text(); surfacing them here lets operators see reclaim
        // state alongside lock/rehash diagnostics in a single dump.
        //   - pending_count:    current retired-but-not-reclaimed objects
        //   - total:            cumulative reclaimed objects (global domain + per-cache)
        //   - freed_bytes:      estimated bytes freed (per-cache delta)
        //   - invocations:      number of try_reclaim() calls (per-cache)
        // Rising pending_count with flat total signals reclaim starvation
        // (e.g. no quiescent point reached under sustained write pressure).
        std::size_t reclaim_pending_count{0};
        std::size_t reclaim_total{0};
        std::size_t reclaim_freed_bytes{0};
        std::size_t reclaim_invocation_count{0};

        // R9: pending-deletion soft-cap observability. Items explicitly
        // force-deleted while still holding active read_handles are deferred
        // in `pending_deletion_` (per-shard). `pending_deletion_skipped_count`
        // counts force_del() calls REFUSED because the list was at/over the
        // configured cap — a growing value flags handle-holding callers.
        std::size_t pending_deletion_count{0};
        std::size_t pending_deletion_skipped_count{0};

        // R9: EBR domain layout. `per_shard_ebr_enabled` is true when each
        // shard owns an independent epoch_domain (via enable_per_shard_ebr());
        // `ebr_domain_count` is 1 (global default) or the number of shards.
        bool per_shard_ebr_enabled{false};
        std::size_t ebr_domain_count{1};

        // L-1: Hazptr slot-exhaustion diagnostics. Surface the hard-cap
        // fallback path so operators can alert before acquire_slot()
        // starts throwing std::runtime_error (which happens after
        // kMaxSyncFallbacks=64 sync-reclaim rounds all fail to free a
        // slot). Non-zero `hazptr_sync_fallback_count` indicates the
        // workload has 8192+ live hazard pointers for extended periods;
        // sustained growth signals an imminent hard-cap throw.
        //   - hazptr_slot_exhaustion_count: every retry bump (incl. yields)
        //   - hazptr_sync_fallback_count:   spin-budget-exceeded fallbacks
        //   - hazptr_slot_capacity:         current allocated slot capacity
        std::size_t hazptr_slot_exhaustion_count{0};
        std::size_t hazptr_sync_fallback_count{0};
        std::size_t hazptr_slot_capacity{0};
        // P0-3: hazptr slot usage gauge. `hazptr_active_slot_count` is
        // the current live handle count; `hazptr_max_slot_count` is the
        // runtime-configurable upper bound (default 8192, raised via
        // `set_hazptr_max_slots()`); `hazptr_slot_usage_ratio` is the
        // ratio (0.0..1.0). When usage exceeds 0.9, operators should
        // either raise the limit or reduce reader fan-out — sustained
        // high usage risks `acquire_slot()` returning `npos`, which
        // degrades reads to cache misses (via empty read_handle).
        std::size_t hazptr_active_slot_count{0};
        std::size_t hazptr_max_slot_count{0};
        float hazptr_slot_usage_ratio{0.0f};

        // P0-2: EBR force-advance policy and counters. Surface the active
        // policy and how often the safety net has fired so operators can
        // detect stuck-thread scenarios and tune the policy / timeout.
        //   - ebr_force_advance_policy: 0=kFailAdvance, 1=kForceAdvanceAfter5s,
        //     2=kForceAdvanceAfterNs, 3=kNeverForceAdvance (see
        //     `lru::detail::force_advance_policy`).
        //   - ebr_force_advance_count:  reclaim passes affected by >= 1 stuck slot
        //   - ebr_epoch_stale_count:    individual stuck-slot observations
        std::size_t ebr_force_advance_policy{0};
        std::size_t ebr_force_advance_count{0};
        std::size_t ebr_epoch_stale_count{0};

        // T-B4 (P2-10): Diagnostics cache freshness.
        // Age of the cached hash-table diagnostics snapshot, in milliseconds.
        // Returns std::numeric_limits<uint64_t>::max() if the cache has never
        // been refreshed OR the underlying hash table doesn't cache
        // (non-segmented tables). Operators should expect ~balancer_interval
        // (1s default); significantly larger values indicate a stalled or
        // missing balancer. See `diagnostics_cache_age_ms()` API for the
        // authoritative value.
        std::uint64_t diagnostics_cache_age_ms{0};

        // P1-3: Drain worker started flag. When false, the hazptr
        // domain's retire_obj() will emit a one-shot stderr warning on
        // the first retire. Operators should call start_event_drain()
        // at startup to set this to true; thread-safe aliases do this
        // automatically at construction.
        bool drain_worker_started{false};

        // P1-2: Active fairness mode. "writer_fair" (default) prevents
        // writer starvation under sustained read load; "reader_preferred"
        // maximizes read throughput at the cost of writer latency.
        // Surfaced so operators can verify the active mode without
        // calling get_fairness_mode() separately.
        std::string fairness_mode_str{"writer_fair"};

        // P1-1: Writer starvation detector metrics. Only meaningful in
        // reader_preferred mode — in writer_fair mode writers are
        // already served promptly and these counters stay 0.
        //   - writer_starvation_events: number of times a reader was
        //     redirected to the writer_fair slow path because a queued
        //     writer exceeded the starvation timeout (default 100ms).
        //   - writer_max_wait_ns: maximum observed writer wait time.
        std::size_t writer_starvation_events{0};
        std::uint64_t writer_max_wait_ns{0};
    };

    /// Snapshot of runtime diagnostics. Lock-free / atomic reads only;
    /// safe to call concurrently with cache operations. The returned
    /// struct is a point-in-time view — fields may be inconsistent with
    /// each other if the cache is being modified concurrently.
    diagnostics_info diagnostics() const {
        diagnostics_info info;
        info.defer_promotion_enabled = is_defer_promotion_enabled();
        info.incremental_rehash_enabled = incremental_rehash_enabled();
        // T19.3: Surface the probing/hashing mode so operators can tell at
        // a glance whether the cache is running chain, F14, or segmented.
        info.f14_probing = std::is_same_v<typename trait_type::probing_style,
                                         detail::f14_probing_tag>;
        info.segmented_hash_table = trait_type::segmented;
        info.compressed_hook = trait_type::uses_compressed_hook;
        // T-P1-3 (R-3): Surface EmbeddedChain mode so operators can verify
        // at runtime that the cache is using lock-free reads. All MM
        // strategies expose `map_type::uses_embedded_chain`. Detection is
        // guarded by `if constexpr (requires {...})` for forward
        // compatibility with custom MM types that don't expose this alias.
        if constexpr (requires { typename std::decay_t<decltype(mm_)>::map_type; }) {
            using map_t = typename std::decay_t<decltype(mm_)>::map_type;
            if constexpr (requires { map_t::uses_embedded_chain; }) {
                info.embedded_chain = map_t::uses_embedded_chain;
            }
        }
        info.tls_ring_backlog = tls_access_ring<Key>::instance().size();
        info.tls_ring_dropped_promotions =
            tls_access_ring<Key>::dropped_count_all_threads();
        // T-M4: Cross-thread backlog aggregate. The per-thread view above is
        // misleading for production monitoring — a single shard with high
        // backlog but only inspected from a low-traffic thread would read
        // ~0. The atomic aggregate across all threads is the right signal.
        info.tls_ring_backlog_total = tls_access_ring<Key>::total_backlog();
        // Task C: prefer per-cache count when tracking is enabled.
        info.active_handle_count = active_handle_count();

        // Per-shard introspection. We use `if constexpr (requires { ... })`
        // to support both sharded and non-sharded MM types. Non-sharded
        // caches populate the vectors with a single element.
        if constexpr (requires { mm_.num_shards(); mm_.shard(0); }) {
            info.num_shards = mm_.num_shards();
            info.per_stripe_wait_count.reserve(info.num_shards);
            info.per_stripe_try_fail_count.reserve(info.num_shards);
            info.per_shard_size.reserve(info.num_shards);
            info.per_shard_bucket_count.reserve(info.num_shards);
            // P1-2: Reserve fine-grained per-shard vectors.
            info.per_shard_hits.reserve(info.num_shards);
            info.per_shard_misses.reserve(info.num_shards);
            info.per_shard_evictions.reserve(info.num_shards);
            info.per_shard_memory.reserve(info.num_shards);
            info.per_shard_rehash_count.reserve(info.num_shards);
            for (std::size_t i = 0; i < info.num_shards; ++i) {
                const auto& s = mm_.shard(i).stats();
                info.per_stripe_wait_count.push_back(
                    s.write_lock_wait_count.load(std::memory_order_relaxed));
                info.per_stripe_try_fail_count.push_back(
                    s.try_lock_fail_count.load(std::memory_order_relaxed));
                info.per_shard_size.push_back(
                    s.current_size.load(std::memory_order_relaxed));
                info.per_shard_bucket_count.push_back(
                    mm_.shard(i).bucket_count());
                // P1-2: Fine-grained per-shard metrics.
                info.per_shard_hits.push_back(
                    s.hits.value.load(std::memory_order_relaxed));
                info.per_shard_misses.push_back(
                    s.misses.value.load(std::memory_order_relaxed));
                info.per_shard_evictions.push_back(
                    s.evictions.value.load(std::memory_order_relaxed));
                info.per_shard_memory.push_back(
                    s.current_memory.load(std::memory_order_relaxed));
                info.per_shard_rehash_count.push_back(
                    s.rehash_count.load(std::memory_order_relaxed));
            }
            // Aggregated rehash state across all shards.
            info.rehash_in_progress = mm_.is_rehashing();
            info.rehash_progress_buckets = mm_.rehash_progress();
            info.rehash_old_bucket_count = mm_.rehash_old_bucket_count();
            info.rehash_new_bucket_count = mm_.rehash_new_bucket_count();
        } else {
            // Non-sharded MM (mm_lru / mm_2q / mm_tiny_lfu / mm_wtiny_lfu / mm_fifo)
            info.num_shards = 1;
            const auto& s = mm_.stats();
            info.per_stripe_wait_count.push_back(
                s.write_lock_wait_count.load(std::memory_order_relaxed));
            info.per_stripe_try_fail_count.push_back(
                s.try_lock_fail_count.load(std::memory_order_relaxed));
            info.per_shard_size.push_back(
                s.current_size.load(std::memory_order_relaxed));
            if constexpr (requires { mm_.bucket_count(); }) {
                info.per_shard_bucket_count.push_back(mm_.bucket_count());
            } else {
                info.per_shard_bucket_count.push_back(0);
            }
            // P1-2: Fine-grained metrics for non-sharded caches.
            info.per_shard_hits.push_back(
                s.hits.value.load(std::memory_order_relaxed));
            info.per_shard_misses.push_back(
                s.misses.value.load(std::memory_order_relaxed));
            info.per_shard_evictions.push_back(
                s.evictions.value.load(std::memory_order_relaxed));
            info.per_shard_memory.push_back(
                s.current_memory.load(std::memory_order_relaxed));
            info.per_shard_rehash_count.push_back(
                s.rehash_count.load(std::memory_order_relaxed));
            if constexpr (requires { mm_.is_rehashing(); }) {
                info.rehash_in_progress = mm_.is_rehashing();
                info.rehash_progress_buckets = mm_.rehash_progress();
                info.rehash_old_bucket_count = mm_.rehash_old_bucket_count();
                info.rehash_new_bucket_count = mm_.rehash_new_bucket_count();
            }
        }
        // T17.3: Refresh deferred-reclamation counters from the global
        // hazptr/EBR domains and the per-cache accumulator. Mirrors the
        // logic in stats_snapshot() so diagnostics_text() and Prometheus
        // report consistent numbers. Lock-free reads only.
        std::size_t pending = detail::hazptr_domain::default_domain().pending_count()
                            + ebr_pending_count();
        std::size_t total = detail::hazptr_domain::default_domain().reclaim_total()
                          + ebr_reclaim_total();
        info.reclaim_pending_count = pending;
        // R9: pending-deletion soft-cap + EBR domain layout observability.
        info.pending_deletion_count = pending_deletion_count();
        info.pending_deletion_skipped_count = pending_deletion_skipped_count();
        info.per_shard_ebr_enabled = per_shard_ebr_enabled_;
        info.ebr_domain_count = ebr_domain_count();
        // T-M4: Populate per-shard retire-pending. Hazptr/EBR domains are
        // global by default, so every shard reports the same pending count
        // — we mirror `info.reclaim_pending_count` into a vector sized
        // num_shards. This keeps the field forward-compatible with
        // per-shard retire queues (already plumbed via shard_hotspot)
        // without requiring per-shard domain plumbing today.
        info.per_shard_retire_pending.assign(info.num_shards, pending);
        info.reclaim_total =
            per_cache_stats_.reclaim_total.load(std::memory_order_relaxed) + total;
        info.reclaim_invocation_count =
            per_cache_stats_.reclaim_invocation_count.load(std::memory_order_relaxed);
        info.reclaim_freed_bytes =
            per_cache_stats_.reclaim_freed_bytes.load(std::memory_order_relaxed);
        // T-B4 (P2-10): Surface the diagnostics cache age so operators
        // reading a `diagnostics_text()` dump can immediately tell whether
        // the hash-table metrics (max_chain_length, per-segment load factor)
        // are fresh or stale. For non-segmented tables this returns max
        // (no cache, age is meaningless — operators should check the
        // segmented_hash_table flag first).
        info.diagnostics_cache_age_ms = diagnostics_cache_age_ms();
        // P1-3: Surface the hazptr drain worker state so operators
        // can verify (via a single diagnostics dump) that the worker
        // is running. When false, retire_obj() emits a one-shot
        // stderr warning on first call.
        info.drain_worker_started =
            detail::hazptr_domain::default_domain().is_drain_started();
        // P1-2: Surface the active fairness mode so operators can
        // verify (via a single diagnostics dump) whether the cache is
        // running writer_fair (default, prevents writer starvation) or
        // reader_preferred (max read throughput, may starve writers).
        info.fairness_mode_str =
            detail::fairness_mode_to_string(get_fairness_mode());
        // P1-1: Writer starvation detector metrics. Aggregate across
        // all stripes for striped caches. Non-zero writer_starvation_events
        // in reader_preferred mode indicates the safety net is firing —
        // operators should consider switching to writer_fair permanently.
        info.writer_starvation_events = writer_starvation_events();
        info.writer_max_wait_ns = writer_max_wait_ns();
        // L-1: Hazptr slot-exhaustion diagnostics. Mirror the global
        // hazptr_domain counters into diagnostics so operators can alert
        // on sustained sync-fallback growth before acquire_slot() starts
        // throwing std::runtime_error (which happens after
        // kMaxSyncFallbacks=64 failed sync-reclaim rounds).
        info.hazptr_slot_exhaustion_count =
            detail::hazptr_domain::default_domain().slot_exhaustion_count();
        info.hazptr_sync_fallback_count =
            detail::hazptr_domain::default_domain().hazptr_sync_fallback_count();
        info.hazptr_slot_capacity =
            detail::hazptr_domain::default_domain().capacity();
        // P0-3: hazptr slot usage gauges — surface live handles and
        // the runtime-configurable upper bound so operators can detect
        // sustained high usage and raise the limit via
        // `set_hazptr_max_slots()` before acquire_slot() starts
        // returning `npos` (which degrades reads to cache misses).
        info.hazptr_active_slot_count =
            detail::hazptr_domain::default_domain().active_slot_count();
        info.hazptr_max_slot_count =
            detail::hazptr_domain::default_domain().max_slot_count();
        info.hazptr_slot_usage_ratio =
            detail::hazptr_domain::default_domain().slot_usage_ratio();
        // P0-2: EBR force-advance diagnostics. Surface the active policy
        // (as the underlying enum integer) and the cumulative safety-net
        // counters so operators can detect stuck-thread scenarios and
        // tune the policy / timeout via set_force_advance_policy().
        info.ebr_force_advance_policy =
            static_cast<std::size_t>(get_force_advance_policy());
        info.ebr_force_advance_count = ebr_force_advance_count();
        info.ebr_epoch_stale_count = epoch_stale_count();
        return info;
    }

    /// Multi-line human-readable diagnostics dump. Useful for ad-hoc
    /// inspection via debugger or log. For Prometheus scraping use
    /// prometheus_text() instead.
    ///
    /// T-G12: Results are cached for `diagnostics_cache_ttl_ms()`
    /// (default 500ms) — high-frequency callers (e.g. dashboards polling
    /// at 10Hz) get the same snapshot without rebuilding on every call.
    /// Call `refresh_diagnostics_cache()` to force an immediate rebuild.
    std::string diagnostics_text() const {
        auto info = *cached_diagnostics();
        std::string out;
        out.reserve(2048);

        auto append = [&](std::string_view line) {
            out.append(line);
            out.push_back('\n');
        };
        auto append_kv = [&](std::string_view key, std::size_t val) {
            out.append(key);
            out.append(": ");
            out.append(std::to_string(val));
            out.push_back('\n');
        };
        auto append_kv_str = [&](std::string_view key, std::string_view val) {
            out.append(key);
            out.append(": ");
            out.append(val);
            out.push_back('\n');
        };
        auto append_kv_u64 = [&](std::string_view key, std::uint64_t val) {
            out.append(key);
            out.append(": ");
            out.append(std::to_string(val));
            out.push_back('\n');
        };

        append("=== LRU cache diagnostics ===");

        // Aggregate.
        append_kv("num_shards", info.num_shards);
        append_kv("active_handle_count", info.active_handle_count);
        append_kv("tls_ring_backlog (calling thread)", info.tls_ring_backlog);
        append_kv("tls_ring_backlog_total (all threads)",
                  info.tls_ring_backlog_total);
        append_kv("tls_ring_dropped_promotions (cumulative, all threads)",
                  info.tls_ring_dropped_promotions);
        append_kv("defer_promotion_enabled",
                  info.defer_promotion_enabled ? 1 : 0);
        append_kv("incremental_rehash_enabled",
                  info.incremental_rehash_enabled ? 1 : 0);

        // T19.3: Hash table / probing mode in effect. This is critical for
        // operators interpreting the rest of the dump — the rehash semantics
        // and bucket_count meaning differ across chain, F14, and segmented.
        append("--- hash table mode ---");
        append_kv("f14_probing", info.f14_probing ? 1 : 0);
        append_kv("segmented_hash_table", info.segmented_hash_table ? 1 : 0);
        append_kv("compressed_hook", info.compressed_hook ? 1 : 0);
        // T-P1-3 (R-3): EmbeddedChain mode. Operators should alert if this
        // is 0 — it means the cache is using shared_lock fallback for reads
        // (use-after-free prevention), which kills throughput under high
        // concurrency. All production aliases assert this at compile time,
        // but custom traits may bypass the assertion.
        append_kv("embedded_chain", info.embedded_chain ? 1 : 0);
        // Rehash mode (incremental vs blocking) was already reported above;
        // restate it here alongside the probing mode for a single-glance view.
        append_kv("rehash_mode",
                  info.incremental_rehash_enabled ? 1 /*incremental*/ : 0 /*blocking*/);
        // T-B4 (P2-10): Diagnostics cache age — operators use this to
        // determine if max_chain_length / per-segment LF are fresh. Only
        // meaningful when segmented_hash_table == 1 (non-segmented tables
        // don't cache). A value >> balancer interval (1s default) means
        // the balancer is stalled or not started.
        append_kv_u64("diagnostics_cache_age_ms", info.diagnostics_cache_age_ms);

        append_kv("rehash_in_progress", info.rehash_in_progress ? 1 : 0);
        append_kv("rehash_progress_buckets", info.rehash_progress_buckets);
        append_kv("rehash_old_bucket_count", info.rehash_old_bucket_count);
        append_kv("rehash_new_bucket_count", info.rehash_new_bucket_count);

        // Per-shard.
        append("--- per-stripe contention ---");
        // T-M4: Iterate using the min size so a partial diagnostics
        // snapshot (e.g. MM type without per-shard retire tracking wired
        // through yet) cannot trigger OOB reads.
        const std::size_t per_shard_n = std::min({info.per_stripe_wait_count.size(),
                                                   info.per_shard_size.size(),
                                                   info.per_shard_retire_pending.size()});
        for (std::size_t i = 0; i < per_shard_n; ++i) {
            char header[64];
            std::snprintf(header, sizeof(header), "shard[%zu]", i);
            out.append(header);
            out.append(" size=");
            out.append(std::to_string(info.per_shard_size[i]));
            out.append(" buckets=");
            out.append(std::to_string(info.per_shard_bucket_count[i]));
            out.append(" wait=");
            out.append(std::to_string(info.per_stripe_wait_count[i]));
            out.append(" try_fail=");
            out.append(std::to_string(info.per_stripe_try_fail_count[i]));
            out.append(" retire_pending=");
            out.append(std::to_string(info.per_shard_retire_pending[i]));
            out.push_back('\n');
        }

        // Aggregate totals for convenience.
        std::size_t total_wait = 0, total_try_fail = 0, total_size = 0;
        for (auto v : info.per_stripe_wait_count) total_wait += v;
        for (auto v : info.per_stripe_try_fail_count) total_try_fail += v;
        for (auto v : info.per_shard_size) total_size += v;
        append("--- totals ---");
        append_kv("total_wait_count", total_wait);
        append_kv("total_try_fail_count", total_try_fail);
        append_kv("total_size", total_size);

        // Rehash completion percentage.
        if (info.rehash_in_progress && info.rehash_old_bucket_count > 0) {
            double pct = 100.0 * static_cast<double>(info.rehash_progress_buckets)
                         / static_cast<double>(info.rehash_old_bucket_count);
            char buf[64];
            std::snprintf(buf, sizeof(buf), "%.2f", pct);
            out.append("rehash_completion_percent: ");
            out.append(buf);
            out.push_back('\n');
        }

        // T17.3: Deferred-reclamation health. Useful for diagnosing
        // memory growth caused by reclaim starvation under sustained
        // write pressure (no quiescent point reached for hazptr/EBR).
        append("--- reclaim health (hazptr + EBR) ---");
        append_kv("reclaim_pending_count", info.reclaim_pending_count);
        append_kv("reclaim_total", info.reclaim_total);
        append_kv("reclaim_freed_bytes", info.reclaim_freed_bytes);
        append_kv("reclaim_invocation_count", info.reclaim_invocation_count);
        // R9: pending-deletion soft cap + EBR domain layout.
        append_kv("pending_deletion_count", info.pending_deletion_count);
        append_kv("pending_deletion_skipped_count", info.pending_deletion_skipped_count);
        append_kv("per_shard_ebr_enabled", info.per_shard_ebr_enabled);
        append_kv("ebr_domain_count", info.ebr_domain_count);

        // L-1: Hazptr slot-exhaustion / hard-cap fallback diagnostics.
        // Non-zero hazptr_sync_fallback_count indicates the workload has
        // 8192+ live hazard pointers for extended periods — sustained
        // growth signals an imminent std::runtime_error throw from
        // acquire_slot() after kMaxSyncFallbacks=64 failed sync-reclaims.
        append("--- hazptr slot exhaustion (L-1 hard-cap) ---");
        append_kv("hazptr_slot_exhaustion_count", info.hazptr_slot_exhaustion_count);
        append_kv("hazptr_sync_fallback_count", info.hazptr_sync_fallback_count);
        append_kv("hazptr_slot_capacity", info.hazptr_slot_capacity);
        // P0-3: live handle count vs runtime-configurable upper bound.
        // When usage_ratio exceeds 0.9, raise the limit via
        // set_hazptr_max_slots() or reduce reader fan-out.
        append_kv("hazptr_active_slot_count", info.hazptr_active_slot_count);
        append_kv("hazptr_max_slot_count", info.hazptr_max_slot_count);
        {
            char buf[64];
            std::snprintf(buf, sizeof(buf), "%.4f", info.hazptr_slot_usage_ratio);
            out.append("hazptr_slot_usage_ratio: ");
            out.append(buf);
            out.push_back('\n');
        }

        // P0-2: EBR force-advance policy and counters. Non-zero
        // ebr_force_advance_count indicates stuck-thread scenarios
        // triggered the safety net — operators should investigate the
        // cause (long-running CS, GC pause, descheduled thread) and
        // consider switching to kFailAdvance if UAF risk is unacceptable.
        append("--- ebr force-advance (P0-2 policy) ---");
        append_kv("ebr_force_advance_policy", info.ebr_force_advance_policy);
        append_kv("ebr_force_advance_count", info.ebr_force_advance_count);
        append_kv("ebr_epoch_stale_count", info.ebr_epoch_stale_count);

        // P1-3: Drain worker state. When false, retire_obj() emits a
        // one-shot stderr warning on first call — operators should
        // call start_event_drain() at startup (thread-safe aliases do
        // this automatically at construction).
        append("--- drain worker (P1-3) ---");
        append_kv("drain_worker_started",
            static_cast<std::size_t>(info.drain_worker_started ? 1 : 0));

        // P1-2: Active fairness mode. writer_fair (default) prevents
        // writer starvation; reader_preferred maximizes read throughput.
        append("--- fairness (P1-2) ---");
        append_kv_str("fairness_mode", info.fairness_mode_str);

        // P1-1: Writer starvation detector metrics. Only meaningful in
        // reader_preferred mode — non-zero writer_starvation_events
        // indicates the safety net fired. Operators should consider
        // switching to writer_fair permanently if events grow steadily.
        append("--- writer starvation (P1-1) ---");
        append_kv("writer_starvation_events", info.writer_starvation_events);
        append_kv("writer_max_wait_ns", info.writer_max_wait_ns);

        append("=== end diagnostics ===");
        return out;
    }

    // ----------------------------------------------------------------
    // T-G12: diagnostics_json + short-time diagnostics cache
    // ----------------------------------------------------------------
    //
    // `diagnostics_json()` produces a structured JSON view of the same
    // `diagnostics_info` consumed by `diagnostics_text()`. The JSON is
    // machine-parseable (jq / log aggregators / dashboards) and avoids
    // the parsing ambiguity of the text dump (which uses unquoted keys
    // and "key: value" lines).
    //
    // Both `diagnostics_text()` and `diagnostics_json()` read from the
    // same short-time cache (default TTL 500ms). The cache is always on
    // and lazily refreshed on read — no background worker needed. This
    // is intentionally simpler than the prometheus_text() cache (which
    // requires explicit enable + worker) because diagnostics are cheap
    // to rebuild and the cache only needs to amortize 10Hz+ polling.

    /// T-G12: Return the diagnostics cache TTL in milliseconds. The
    /// cache is always enabled; this only controls freshness. Default
    /// 500ms — callers polling faster than this get cached results.
    std::uint32_t diagnostics_cache_ttl_ms() const noexcept {
        return diagnostics_cache_ttl_ms_.load(std::memory_order_relaxed);
    }

    /// T-G12: Set the diagnostics cache TTL. Lower values trade CPU for
    /// freshness; 0 disables caching entirely (every call rebuilds).
    void set_diagnostics_cache_ttl_ms(std::uint32_t ms) noexcept {
        diagnostics_cache_ttl_ms_.store(ms, std::memory_order_relaxed);
    }

    /// T-G12: Force an immediate rebuild of the cached diagnostics
    /// snapshot. The next `diagnostics_text()` / `diagnostics_json()`
    /// call observes the fresh snapshot. Safe to call concurrently.
    void refresh_diagnostics_cache() {
        std::lock_guard<std::mutex> lk(diagnostics_cache_lock_);
        auto fresh = std::make_shared<const diagnostics_info>(diagnostics());
        diagnostics_snap_cache_.store(fresh, std::memory_order_release);
        diagnostics_snap_ts_ns_.store(
            std::chrono::steady_clock::now().time_since_epoch().count(),
            std::memory_order_release);
    }

    /// T-G12: Structured JSON diagnostics dump. Same content as
    /// `diagnostics_text()` but in a machine-parseable format. Reads
    /// from the same short-time cache as `diagnostics_text()`.
    ///
    /// The JSON uses flat key names (no nesting beyond per-shard
    /// arrays) to keep the output compact and jq-friendly:
    ///   {
    ///     "num_shards": 64,
    ///     "active_handle_count": 12,
    ///     "per_shard": [
    ///       {"index": 0, "size": 1024, "buckets": 2048, ...},
    ///       ...
    ///     ],
    ///     ...
    ///   }
    std::string diagnostics_json() const {
        auto info = *cached_diagnostics();
        std::string out;
        out.reserve(4096);
        auto append_str = [&](std::string_view s) { out.append(s); };
        auto append_kv_str_field = [&](std::string_view k, std::string_view v) {
            out.append("\""); out.append(k); out.append("\": \"");
            out.append(v); out.append("\", ");
        };
        auto append_kv_u_field = [&](std::string_view k, std::uint64_t v) {
            out.append("\""); out.append(k); out.append("\": ");
            out.append(std::to_string(v)); out.append(", ");
        };
        auto append_kv_b_field = [&](std::string_view k, bool v) {
            out.append("\""); out.append(k); out.append("\": ");
            out.append(v ? "true" : "false"); out.append(", ");
        };
        auto append_kv_f_field = [&](std::string_view k, float v) {
            char buf[64];
            std::snprintf(buf, sizeof(buf), "%.4f", v);
            out.append("\""); out.append(k); out.append("\": ");
            out.append(buf); out.append(", ");
        };

        append_str("{");

        // Aggregate scalars.
        append_kv_u_field("num_shards", info.num_shards);
        append_kv_u_field("active_handle_count", info.active_handle_count);
        append_kv_u_field("tls_ring_backlog", info.tls_ring_backlog);
        append_kv_u_field("tls_ring_backlog_total", info.tls_ring_backlog_total);
        append_kv_u_field("tls_ring_dropped_promotions", info.tls_ring_dropped_promotions);
        append_kv_b_field("defer_promotion_enabled", info.defer_promotion_enabled);
        append_kv_b_field("incremental_rehash_enabled", info.incremental_rehash_enabled);

        // Hash table mode.
        append_kv_b_field("f14_probing", info.f14_probing);
        append_kv_b_field("segmented_hash_table", info.segmented_hash_table);
        append_kv_b_field("compressed_hook", info.compressed_hook);
        append_kv_b_field("embedded_chain", info.embedded_chain);
        append_kv_u_field("diagnostics_cache_age_ms", info.diagnostics_cache_age_ms);
        append_kv_b_field("rehash_in_progress", info.rehash_in_progress);
        append_kv_u_field("rehash_progress_buckets", info.rehash_progress_buckets);
        append_kv_u_field("rehash_old_bucket_count", info.rehash_old_bucket_count);
        append_kv_u_field("rehash_new_bucket_count", info.rehash_new_bucket_count);

        // Reclaim health.
        append_kv_u_field("reclaim_pending_count", info.reclaim_pending_count);
        append_kv_u_field("reclaim_total", info.reclaim_total);
        append_kv_u_field("reclaim_freed_bytes", info.reclaim_freed_bytes);
        append_kv_u_field("reclaim_invocation_count", info.reclaim_invocation_count);
        // R9: pending-deletion soft cap + EBR domain layout.
        append_kv_u_field("pending_deletion_count", info.pending_deletion_count);
        append_kv_u_field("pending_deletion_skipped_count", info.pending_deletion_skipped_count);
        append_kv_b_field("per_shard_ebr_enabled", info.per_shard_ebr_enabled);
        append_kv_u_field("ebr_domain_count", info.ebr_domain_count);

        // Hazptr slot exhaustion.
        append_kv_u_field("hazptr_slot_exhaustion_count", info.hazptr_slot_exhaustion_count);
        append_kv_u_field("hazptr_sync_fallback_count", info.hazptr_sync_fallback_count);
        append_kv_u_field("hazptr_slot_capacity", info.hazptr_slot_capacity);
        append_kv_u_field("hazptr_active_slot_count", info.hazptr_active_slot_count);
        append_kv_u_field("hazptr_max_slot_count", info.hazptr_max_slot_count);
        append_kv_f_field("hazptr_slot_usage_ratio", info.hazptr_slot_usage_ratio);

        // EBR force-advance.
        append_kv_u_field("ebr_force_advance_policy", info.ebr_force_advance_policy);
        append_kv_u_field("ebr_force_advance_count", info.ebr_force_advance_count);
        append_kv_u_field("ebr_epoch_stale_count", info.ebr_epoch_stale_count);

        // Drain + fairness.
        append_kv_b_field("drain_worker_started", info.drain_worker_started);
        append_kv_str_field("fairness_mode", info.fairness_mode_str);
        append_kv_u_field("writer_starvation_events", info.writer_starvation_events);
        append_kv_u_field("writer_max_wait_ns", info.writer_max_wait_ns);

        // Per-shard array. Trim trailing comma from prior field.
        if (out.size() >= 2 && out[out.size() - 2] == ',') {
            out.pop_back();  // trailing space
            out.pop_back();  // trailing comma
        }
        out.append(", \"per_shard\": [");
        const std::size_t per_shard_n = std::min({info.per_stripe_wait_count.size(),
                                                   info.per_shard_size.size(),
                                                   info.per_shard_retire_pending.size()});
        for (std::size_t i = 0; i < per_shard_n; ++i) {
            if (i > 0) out.append(", ");
            out.append("{\"index\": ");
            out.append(std::to_string(i));
            out.append(", \"size\": ");
            out.append(std::to_string(info.per_shard_size[i]));
            out.append(", \"buckets\": ");
            out.append(std::to_string(info.per_shard_bucket_count[i]));
            out.append(", \"wait\": ");
            out.append(std::to_string(info.per_stripe_wait_count[i]));
            out.append(", \"try_fail\": ");
            out.append(std::to_string(info.per_stripe_try_fail_count[i]));
            out.append(", \"retire_pending\": ");
            out.append(std::to_string(info.per_shard_retire_pending[i]));
            out.append(", \"hits\": ");
            out.append(std::to_string(i < info.per_shard_hits.size()
                                       ? info.per_shard_hits[i] : 0));
            out.append(", \"misses\": ");
            out.append(std::to_string(i < info.per_shard_misses.size()
                                       ? info.per_shard_misses[i] : 0));
            out.append(", \"evictions\": ");
            out.append(std::to_string(i < info.per_shard_evictions.size()
                                       ? info.per_shard_evictions[i] : 0));
            out.append(", \"memory\": ");
            out.append(std::to_string(i < info.per_shard_memory.size()
                                       ? info.per_shard_memory[i] : 0));
            out.append("}");
        }
        out.append("]}");
        return out;
    }

private:
    /// T-G12: Return a shared_ptr to a cached diagnostics_info snapshot.
    /// Rebuilds lazily if the cache is stale (older than TTL) or empty.
    /// The returned pointer is always non-null. Safe to call concurrently.
    std::shared_ptr<const diagnostics_info> cached_diagnostics() const {
        const auto ttl_ms = diagnostics_cache_ttl_ms_.load(std::memory_order_relaxed);
        if (ttl_ms == 0) {
            // Caching disabled — rebuild on every call.
            return std::make_shared<const diagnostics_info>(diagnostics());
        }
        // Fast path: load cached snapshot + timestamp atomically.
        auto cached = diagnostics_snap_cache_.load(std::memory_order_acquire);
        const auto ts_ns = diagnostics_snap_ts_ns_.load(std::memory_order_acquire);
        if (cached) {
            const auto now_ns = std::chrono::steady_clock::now().time_since_epoch().count();
            const auto age_ms = static_cast<std::uint64_t>(now_ns - ts_ns) / 1'000'000u;
            if (age_ms < ttl_ms) {
                return cached;
            }
        }
        // Slow path: rebuild under lock. Other threads may race us;
        // the lock ensures only one rebuild happens at a time.
        std::lock_guard<std::mutex> lk(diagnostics_cache_lock_);
        // Double-check after acquiring the lock — another thread may
        // have just rebuilt the cache.
        cached = diagnostics_snap_cache_.load(std::memory_order_relaxed);
        const auto ts_ns2 = diagnostics_snap_ts_ns_.load(std::memory_order_relaxed);
        if (cached) {
            const auto now_ns = std::chrono::steady_clock::now().time_since_epoch().count();
            const auto age_ms = static_cast<std::uint64_t>(now_ns - ts_ns2) / 1'000'000u;
            if (age_ms < ttl_ms) {
                return cached;
            }
        }
        auto fresh = std::make_shared<const diagnostics_info>(diagnostics());
        diagnostics_snap_cache_.store(fresh, std::memory_order_release);
        diagnostics_snap_ts_ns_.store(
            std::chrono::steady_clock::now().time_since_epoch().count(),
            std::memory_order_release);
        return fresh;
    }

public:
    /// Returns a vector of (shard_index, total_accesses, hit_rate) sorted
    /// descending by total_accesses. Useful for detecting load imbalance
    /// and hash distribution skew in production.
    ///
    /// T18.1: The struct now also carries memory_usage, pending_reclaim
    /// (global hazptr+EBR pending count, identical across shards — per-shard
    /// retire tracking is not yet wired through cache_stats), and a per-shard
    /// rehash_in_progress flag. Use hot_shards_by_memory() to rank shards
    /// by memory consumption instead of access volume.
    ///
    /// \param top_n Maximum number of hot shards to return (default: 5).
    /// \return Sorted vector of shard statistics. Empty for non-sharded caches.
    struct shard_hotspot {
        std::size_t shard_index;
        std::size_t total_accesses;
        double hit_rate;
        std::size_t memory_bytes;
        std::size_t rehash_count;
        // T18.1: Memory-dimension and reclaim-dimension fields.
        std::size_t memory_usage;          // alias for memory_bytes (named per tasks.md)
        std::size_t pending_reclaim;       // global pending reclaim count (hazptr + EBR)
        bool rehash_in_progress;           // per-shard rehash flag
    };
    std::vector<shard_hotspot> hot_shards(std::size_t top_n = 5) const {
        std::vector<shard_hotspot> result;
        if constexpr (requires { mm_.num_shards(); mm_.shard(0); }) {
            const std::size_t n = mm_.num_shards();
            result.reserve(n);
            // T18.1: Snapshot the global pending-reclaim count once so all
            // shards report a consistent value (it is global by nature —
            // hazptr/EBR domains are not partitioned per shard).
            const std::size_t global_pending =
                detail::hazptr_domain::default_domain().pending_count()
              + ebr_pending_count();
            for (std::size_t i = 0; i < n; ++i) {
                const auto& s = mm_.shard(i).stats();
                const auto h = s.hits.value.load(std::memory_order_relaxed);
                const auto m = s.misses.value.load(std::memory_order_relaxed);
                const auto total = h + m;
                const double rate = (total == 0) ? 0.0
                    : static_cast<double>(h) / static_cast<double>(total);
                const std::size_t mem = s.current_memory.load(std::memory_order_relaxed);
                bool per_shard_rehashing = false;
                if constexpr (requires { mm_.shard(i).is_rehashing(); }) {
                    per_shard_rehashing = mm_.shard(i).is_rehashing();
                }
                result.push_back({
                    i,
                    total,
                    rate,
                    mem,
                    s.rehash_count.load(std::memory_order_relaxed),
                    /*memory_usage*/ mem,
                    /*pending_reclaim*/ global_pending,
                    /*rehash_in_progress*/ per_shard_rehashing
                });
            }
            // Sort descending by total_accesses
            std::sort(result.begin(), result.end(),
                [](const shard_hotspot& a, const shard_hotspot& b) {
                    return a.total_accesses > b.total_accesses;
                });
            if (result.size() > top_n) {
                result.resize(top_n);
            }
        }
        return result;
    }

    /// T18.2: Identify the top-N shards by memory consumption rather than
    /// access volume. Useful for detecting memory imbalance caused by a few
    /// shards holding disproportionately large values (e.g. a hot key with
    /// a large payload, or skew in value-size distribution). The returned
    /// vector carries the same fields as hot_shards() — only the sort key
    /// differs (memory_usage descending instead of total_accesses).
    ///
    /// \param top_n Maximum number of hot shards to return (default: 5).
    /// \return Sorted vector of shard statistics. Empty for non-sharded caches.
    std::vector<shard_hotspot> hot_shards_by_memory(std::size_t top_n = 5) const {
        std::vector<shard_hotspot> result;
        if constexpr (requires { mm_.num_shards(); mm_.shard(0); }) {
            const std::size_t n = mm_.num_shards();
            result.reserve(n);
            const std::size_t global_pending =
                detail::hazptr_domain::default_domain().pending_count()
              + ebr_pending_count();
            for (std::size_t i = 0; i < n; ++i) {
                const auto& s = mm_.shard(i).stats();
                const auto h = s.hits.value.load(std::memory_order_relaxed);
                const auto m = s.misses.value.load(std::memory_order_relaxed);
                const auto total = h + m;
                const double rate = (total == 0) ? 0.0
                    : static_cast<double>(h) / static_cast<double>(total);
                const std::size_t mem = s.current_memory.load(std::memory_order_relaxed);
                bool per_shard_rehashing = false;
                if constexpr (requires { mm_.shard(i).is_rehashing(); }) {
                    per_shard_rehashing = mm_.shard(i).is_rehashing();
                }
                result.push_back({
                    i,
                    total,
                    rate,
                    mem,
                    s.rehash_count.load(std::memory_order_relaxed),
                    /*memory_usage*/ mem,
                    /*pending_reclaim*/ global_pending,
                    /*rehash_in_progress*/ per_shard_rehashing
                });
            }
            // T18.2: Sort descending by memory_usage (tie-break by accesses).
            std::sort(result.begin(), result.end(),
                [](const shard_hotspot& a, const shard_hotspot& b) {
                    if (a.memory_usage != b.memory_usage) {
                        return a.memory_usage > b.memory_usage;
                    }
                    return a.total_accesses > b.total_accesses;
                });
            if (result.size() > top_n) {
                result.resize(top_n);
            }
        }
        return result;
    }

    /// T3.2 (P1-2): Hot key detection — returns the top-N hottest keys
    /// by estimated hit count. Requires event tracking to be enabled
    /// (see `enable_event_tracking()`); otherwise returns an empty vector.
    ///
    /// The returned vector contains (key_hash, estimated_hit_count) pairs
    /// sorted by count descending. The estimated count is an upper bound
    /// from the underlying Space-Saving sketch; the true count is
    /// >= estimated_count - error_bound(). Call `hot_keys_stats()` on
    /// the event tracker for the error bound.
    ///
    /// **Production guidance for hot keys** (P1-2):
    ///
    /// When a key is consistently in the top-N and its single-shard QPS
    /// exceeds the shard's sustainable throughput, consider one of:
    ///
    /// 1. **Application-layer replication** — maintain N copies of the
    ///    value in the cache under transformed keys (e.g.,
    ///    `key__replica_0`, `key__replica_1`, ..., `key__replica_{N-1}`)
    ///    that hash to different shards. Route reads randomly across
    ///    the replicas; write to all replicas. This is the recommended
    ///    approach because it keeps the cache's consistency model
    ///    simple (each replica is an independent key) and lets the
    ///    application control the replication factor per key.
    ///
    /// 2. **Use `compact_cache`** for small hot keys (≤ 64 bytes
    ///    key+value) — it has no shard lock and reads are purely
    ///    lock-free, eliminating shard contention entirely.
    ///
    /// 3. **Use `set_defer_promotion(true)`** (already the default for
    ///    `mm_lru` / `sharded_mm_lru`) — this removes write-lock
    ///    acquisition from the read path, so reads don't contend with
    ///    promotions even on a hot key.
    ///
    /// The library deliberately does **not** auto-replicate hot keys
    /// internally, because:
    /// - Replication factor is workload-dependent (a 100K-QPS key needs
    ///   more replicas than a 10K-QPS key).
    /// - Replica consistency requires CAS or 2PC across shards, which
    ///   adds latency to writes.
    /// - Application-layer routing can use domain knowledge (e.g.,
    ///   read-after-write consistency requirements, geo-distribution).
    ///
    /// \param top_n  Maximum number of hot keys to return (default: 10).
    /// \return Vector of (key_hash, estimated_hit_count) pairs,
    ///         sorted descending by hit count. Empty if event tracking
    ///         is not enabled or no keys have been tracked.
    std::vector<std::pair<uint64_t, std::size_t>> hot_keys(std::size_t top_n = 10) const {
        auto tracker = get_event_tracker();
        if (!tracker) return {};
        return tracker->top_keys(top_n);
    }

    /// T3.2 (P1-2): Hot key detection with human-readable key names.
    /// Requires `set_key_to_string()` to be set on the event tracker;
    /// otherwise the key name is the hash formatted as a hex string.
    ///
    /// \param top_n  Maximum number of hot keys to return (default: 10).
    /// \return Vector of (key_name, estimated_hit_count) pairs,
    ///         sorted descending by hit count.
    std::vector<std::pair<std::string, std::size_t>> hot_keys_with_names(
            std::size_t top_n = 10) const {
        auto tracker = get_event_tracker();
        if (!tracker) return {};
        return tracker->top_keys_with_names(top_n);
    }

    bool contains(const Key& key) const {
        // T14.1: Compact path.
        if constexpr (Trait::is_compact) {
            return compact().contains(key);
        }
        // Lock-free: hash table contains() is independently safe
        // (returns bool, no pointer that could become dangling).
        return mm_.contains(key);
    }

    // --------------------------------------------------------------------
    // Peek - read without promoting (only available if mm_type supports it)
    // --------------------------------------------------------------------

    /// H0: Peek with handle — 不提升 LRU，返回 handle 防止持有期被淘汰。
    /// Lock-free: mm_.peek() uses find_and_pin() to atomically pin the item
    /// under the hash table's bucket lock, eliminating the need for a
    /// stripe-level read lock.
    read_handle<const Value> peek(const Key& key) const {
        // T14.1: Compact path — compact_cache::peek() returns
        // std::optional<std::reference_wrapper<const Value>>. Wrap the
        // pointer in a non-pinning read_handle.
        if constexpr (Trait::is_compact) {
            auto ref = compact().peek(key);
            if (!ref) return {};
            return read_handle<const Value>(const_cast<Value*>(&ref->get()), nullptr, nullptr);
        }
        if constexpr (detail::has_peek_v<mm_type, Key>) {
            return mm_.peek(key);
        } else {
            return {};
        }
    }

    // --------------------------------------------------------------------
    // Capacity
    // --------------------------------------------------------------------

    bool empty() const {
        // T14.1: Compact path.
        if constexpr (Trait::is_compact) {
            return compact().empty();
        }
        return mm_.empty();
    }

    size_type size() const {
        // T14.1: Compact path.
        if constexpr (Trait::is_compact) {
            return compact().size();
        }
        return mm_.size();
    }

    size_type max_size() const {
        // T14.1: Compact path.
        if constexpr (Trait::is_compact) {
            return compact().max_size();
        }
        return mm_.max_size();
    }

    size_type max_memory() const {
        // T14.1: Compact path.
        if constexpr (Trait::is_compact) {
            return compact().max_memory();
        }
        return mm_.max_memory();
    }

    size_type current_memory() const {
        // T14.1: Compact path.
        if constexpr (Trait::is_compact) {
            return compact().current_memory();
        }
        return mm_.current_memory();
    }

    /// Check if any item has an active read_handle.
    /// Returns true if at least one handle is still alive, meaning destroying
    /// the cache would cause use-after-free. Call this before cache destruction
    /// in debug/test scenarios to verify no handles outlive the cache.
    ///
    /// P1-13: For striped caches, this no longer acquires the global read
    /// lock (which shared-locks all 64 stripes — an unnecessary barrier on
    /// the read path). Instead it walks each shard under that shard's
    /// shared lock, allowing concurrent reads/writes on other shards.
    /// The result is still correct: a handle is "active" iff its refcount
    /// is non-zero, and refcount is atomic, so the per-shard lock only
    /// prevents the shard's item list from being observed in an
    /// inconsistent state.
    bool has_active_handles() const {
        if constexpr (is_striped) {
            if constexpr (requires { mm_.num_shards(); }) {
                for (std::size_t i = 0; i < mm_.num_shards(); ++i) {
                    auto lock = striped_mutex_.make_shared_lock(i);
                    if (mm_.shard(i).has_active_handles()) return true;
                }
                return false;
            } else {
                auto lock = acquire_read_lock();
                return mm_.has_active_handles();
            }
        } else if constexpr (is_thread_safe) {
            auto lock = acquire_read_lock();
            return mm_.has_active_handles();
        } else {
            return mm_.has_active_handles();
        }
    }

    void max_size(size_type new_max) {
        // T14.1: Compact path.
        if constexpr (Trait::is_compact) {
            compact().max_size(new_max);
            return;
        }
        flush_guard fg{mm_};
        auto lock = acquire_write_lock();
        mm_.max_size(new_max);
        maybe_report_memory_to_monitor();
    }

    /// P2-1: Strict variant of max_size() — throws std::invalid_argument
    /// when the requested capacity would be silently amplified by the
    /// sharded MM layer (i.e. new_max < num_shards for striped caches).
    /// Use this when the caller needs hard capacity guarantees (e.g.
    /// memory-bounded caches where overshoot would cause OOM).
    ///
    /// For non-sharded caches (single-threaded, safe_cache) this is
    /// equivalent to max_size() because there is no amplification.
    ///
    /// Recommended minimum: `new_max >= num_shards * 16` so each shard
    /// has at least 16 slots of headroom for skewed distributions.
    ///
    /// @throws std::invalid_argument if new_max < num_shards (sharded
    ///         caches only) and new_max != unlimited.
    void max_size_strict(size_type new_max) {
        // T14.1: Compact path — no sharded amplification, just delegate.
        if constexpr (Trait::is_compact) {
            compact().max_size(new_max);
            return;
        }
        flush_guard fg{mm_};
        auto lock = acquire_write_lock();
        if constexpr (is_striped) {
            mm_.max_size_strict(new_max);
        } else {
            mm_.max_size(new_max);
        }
        maybe_report_memory_to_monitor();
    }

    void max_memory(size_type new_max) {
        // T14.1: Compact path — compact_cache does not support max_memory
        // directly; the limit is implicitly kMaxItemSize * max_size.
        // We ignore the call (could log a warning in debug builds).
        if constexpr (Trait::is_compact) {
            (void)new_max;
            return;
        }
        flush_guard fg{mm_};
        auto lock = acquire_write_lock();
        mm_.max_memory(new_max);
        maybe_report_memory_to_monitor();
    }

    // ----------------------------------------------------------------
    // P1-3: Memory watermark and OOM protection
    // ----------------------------------------------------------------

    /// Set memory watermarks as fractions of max_memory in [0.0, 1.0].
    /// When current_memory exceeds soft_watermark, new insertions trigger
    /// aggressive eviction. When it exceeds critical_watermark, the cache
    /// enters read-only mode (set() rejects) until memory drops below
    /// soft_watermark.
    ///
    /// Defaults: soft=0.85, critical=0.95.
    /// Set both to 1.0 to disable watermark-based OOM protection.
    ///
    /// \note soft_watermark must be <= critical_watermark.
    ///       Values are clamped to [0.0, 1.0].
    void set_memory_watermarks(double soft_watermark, double critical_watermark) {
        // Clamp and validate.
        if (soft_watermark < 0.0) soft_watermark = 0.0;
        if (soft_watermark > 1.0) soft_watermark = 1.0;
        if (critical_watermark < 0.0) critical_watermark = 0.0;
        if (critical_watermark > 1.0) critical_watermark = 1.0;
        if (soft_watermark > critical_watermark) {
            std::swap(soft_watermark, critical_watermark);
        }
        memory_soft_watermark_.store(soft_watermark, std::memory_order_relaxed);
        memory_critical_watermark_.store(critical_watermark, std::memory_order_relaxed);
    }

    /// Get the current soft memory watermark (fraction of max_memory).
    double memory_soft_watermark() const noexcept {
        return memory_soft_watermark_.load(std::memory_order_relaxed);
    }

    /// Get the current critical memory watermark (fraction of max_memory).
    double memory_critical_watermark() const noexcept {
        return memory_critical_watermark_.load(std::memory_order_relaxed);
    }

    /// Check if the cache is currently in critical memory mode (read-only).
    /// When true, set() will reject new items until memory drops below
    /// the soft watermark.
    bool is_memory_critical() const noexcept {
        return memory_critical_mode_.load(std::memory_order_acquire);
    }

    /// Register an OOM handler callback. Invoked when the cache enters
    /// critical memory mode. The handler receives (current_memory, max_memory).
    ///
    /// P-HIGH-2 (T-H1): The handler is now invoked *outside* the write lock
    /// (synchronous mode, the default) or by the event_drain_worker
    /// (asynchronous mode, see `set_async_oom_handler(true)`). This means
    /// the handler is free to call cache methods (flush/remove/set) without
    /// deadlocking — previously it was invoked inside the write lock and
    /// any reentrant cache call would deadlock.
    ///
    /// In synchronous mode the handler is called immediately after the
    /// write lock is released by the triggering set() / set_prehashed()
    /// call, on the same thread. In asynchronous mode the event is
    /// enqueued and dispatched by the event_drain_worker (default interval
    /// 1s; 500ms when live read_handles exist), reducing set() tail
    /// latency when the handler performs IO (logging, alerting, external
    /// GC triggers).
    ///
    /// The handler must not throw. Pass nullptr to unregister.
    void set_oom_handler(std::function<void(size_type, size_type)> handler) {
        oom_handler_ = std::move(handler);
    }

    /// P-HIGH-2 (T-H1): Toggle asynchronous OOM handler dispatch.
    ///
    /// When enabled, OOM events are enqueued on a lock-free MPSC queue
    /// and dispatched by the event_drain_worker (must be started via
    /// `start_event_drain()`). This keeps the set() hot path free of
    /// user-handler latency. When disabled (the default), the handler
    /// is invoked synchronously on the set() caller thread, immediately
    /// after the write lock is released.
    ///
    /// Events are coalesced: only the most-recent event is retained,
    /// so if multiple set() calls trigger critical-mode transitions
    /// between drain intervals, the handler sees the latest snapshot.
    /// This is intentional — OOM handlers are typically idempotent
    /// alert/log operations.
    void set_async_oom_handler(bool enabled) noexcept {
        async_oom_handler_.store(enabled, std::memory_order_release);
    }

    /// P-HIGH-2 (T-H1): Returns true if async OOM handler mode is enabled.
    bool is_async_oom_handler() const noexcept {
        return async_oom_handler_.load(std::memory_order_acquire);
    }

    /// P-HIGH-2 (T-H1): Drain pending OOM events and invoke the handler
    /// for each. Called by the event_drain_worker when async mode is
    /// enabled. Returns the number of events dispatched.
    /// Safe to call from any thread; no-op if no events are pending.
    std::size_t drain_oom_events() {
        // Swap out the pending event atomically. We use exchange on a
        // spinlock-protected optional rather than a lock-free queue
        // because OOM transitions are rare (at most a few per second
        // under heavy churn) and the simplicity outweighs a complex
        // MPSC queue. Coalescing means at most one event survives
        // between drains.
        oom_event evt;
        bool has_event = false;
        {
            std::lock_guard<std::mutex> lk(pending_oom_event_mutex_);
            if (pending_oom_event_.has_value()) {
                evt = *pending_oom_event_;
                pending_oom_event_.reset();
                has_event = true;
            }
        }
        if (!has_event) return 0;
        // Invoke outside the mutex to allow the handler to re-enter
        // cache methods (which may enqueue another event).
        if (oom_handler_) {
            try {
                oom_handler_(evt.current_memory, evt.max_memory);
            } catch (...) {
                // Swallow — OOM handler must not throw
            }
        }
        return 1;
    }

    /// P1-3 / P-HIGH-2 (T-H1): Check memory pressure and update critical
    /// mode. Called from set() / set_prehashed() inside the write lock.
    ///
    /// Returns std::nullopt when the insertion should proceed normally
    /// (or when no max_memory is configured). Returns an `oom_event`
    /// populated with the (current_memory, max_memory) snapshot at the
    /// moment the cache *first* entered critical mode — the caller is
    /// responsible for invoking the OOM handler with this event *after*
    /// releasing the write lock (see `dispatch_oom_event()`).
    ///
    /// This function no longer invokes the handler directly (T-H1 fix):
    /// doing so inside the write lock caused deadlocks when the handler
    /// called back into cache methods (flush, remove, etc.). The caller
    /// must call `dispatch_oom_event(evt)` outside the lock.
    ///
    /// Side effects:
    ///   - Sets `memory_critical_mode_` when ratio >= critical watermark.
    ///   - Clears it when ratio drops below the soft watermark.
    std::optional<oom_event> check_memory_pressure() {
        const auto max_mem = mm_.max_memory();
        if (max_mem == unlimited || max_mem == 0) return std::nullopt;

        const auto cur_mem = mm_.current_memory();
        const double ratio = static_cast<double>(cur_mem) / static_cast<double>(max_mem);
        const double critical = memory_critical_watermark_.load(std::memory_order_relaxed);
        const double soft = memory_soft_watermark_.load(std::memory_order_relaxed);

        if (ratio >= critical) {
            // Enter critical mode. exchange() returns the previous value;
            // only the first transition (was_critical == false) yields an
            // event, so subsequent set() calls in critical mode do not
            // re-enqueue the handler. The caller already rejects insertions
            // via the is_memory_critical() check at the top of set(), so
            // we don't need to signal rejection through the return value.
            bool was_critical = memory_critical_mode_.exchange(
                true, std::memory_order_acq_rel);
            if (!was_critical) {
                return oom_event{cur_mem, max_mem};
            }
            // Already in critical mode — no new event to dispatch.
            return std::nullopt;
        }

        if (ratio < soft && memory_critical_mode_.load(std::memory_order_relaxed)) {
            // Drop below soft watermark — exit critical mode
            memory_critical_mode_.store(false, std::memory_order_release);
        }

        return std::nullopt;
    }

    /// P-HIGH-2 (T-H1): Dispatch an OOM event produced by
    /// `check_memory_pressure()`. Must be called *outside* the write lock.
    ///
    /// - In synchronous mode (default): invokes `oom_handler_` inline.
    /// - In asynchronous mode (`set_async_oom_handler(true)`): enqueues
    ///   the event for the event_drain_worker. Events are coalesced —
    ///   only the latest event is retained.
    ///
    /// No-op if `evt` is std::nullopt or no handler is registered.
    void dispatch_oom_event(std::optional<oom_event> evt) {
        if (!evt.has_value()) return;
        if (!oom_handler_) return;
        if (async_oom_handler_.load(std::memory_order_acquire)) {
            // Async: enqueue for the drain worker. Coalesce by overwriting
            // any pending event — OOM handlers are typically idempotent
            // (log/alert), and the latest snapshot is the most actionable.
            std::lock_guard<std::mutex> lk(pending_oom_event_mutex_);
            pending_oom_event_ = evt;
        } else {
            // Sync: invoke immediately on the caller thread, outside the
            // write lock. The handler is free to call cache methods.
            try {
                oom_handler_(evt->current_memory, evt->max_memory);
            } catch (...) {
                // Swallow — OOM handler must not throw
            }
        }
    }

    // --------------------------------------------------------------------
    // Serialization estimates
    // --------------------------------------------------------------------

    /// Get number of items for serialization header.
    ///
    /// P1-13: No lock — mm_.size() reads per-shard atomic counters with
    /// relaxed ordering. Acquiring the global read lock here would
    /// shared-lock all 64 stripes just to read an atomic sum, blocking
    /// writers on every stripe during the call. The returned value is
    /// approximate (a concurrent write may have landed between shard
    /// reads), which is acceptable for a serialization header.
    size_type serialized_item_count() const {
        return mm_.size();
    }

    /// Get serialized size estimate (items only; metadata extra).
    ///
    /// P1-13: No lock — same rationale as serialized_item_count().
    size_type serialized_size_estimate() const {
        return mm_.size() * (sizeof(Key) + sizeof(Value) + sizeof(size_type) * 2);
    }

    // --------------------------------------------------------------------
    // Thread-safe serialization
    // --------------------------------------------------------------------

    /// Thread-safe serialization: acquires read lock and serializes.
    /// Preferred over the free function serialize() which is not thread-safe.
    std::vector<uint8_t> save();

    /// Thread-safe deserialization: acquires write lock and deserializes.
    /// Preferred over the free function deserialize() which is not thread-safe.
    void load(std::span<const uint8_t> data);

    // --------------------------------------------------------------------
    // Task 10: Per-shard serialization with per-shard locking
    // --------------------------------------------------------------------
    //
    // Like save()/load() but acquires only one shard's lock at a time
    // instead of locking all stripes simultaneously. This allows writes
    // to other shards to proceed concurrently with serialization, at
    // the cost of a slightly weaker consistency guarantee (the snapshot
    // may reflect writes that happened between shard snapshots).
    //
    // For non-sharded caches, these fall back to save()/load().
    std::vector<uint8_t> save_per_shard();
    void load_per_shard(std::span<const uint8_t> data);

    /// P2-3: Strictly-atomic cross-shard snapshot.
    ///
    /// Sequence:
    ///   1. shutdown_and_wait(timeout) — reject new ops, drain TLS rings,
    ///      wait for active read_handle count to drop to 0.
    ///   2. acquire_read_lock_all_shards() — hold all 64 per-shard read locks
    ///      simultaneously. Combined with shutdown, no writer can mutate any
    ///      shard, so the snapshot reflects a single instant T0 across shards.
    ///   3. Serialize every shard under the locks (same binary format as
    ///      save_per_shard()).
    ///   4. Release locks; cache remains in shutdown state. The cache
    ///      cannot be re-opened — construct a new cache and load_per_shard()
    ///      to resume service.
    ///
    /// Trade-off vs save_per_shard(): strictly consistent but blocks all
    /// writers for the duration of serialization — use only when a
    /// transactional snapshot is required (e.g. transactional warm restart,
    /// audit checkpoint). For best-effort snapshots use save_per_shard().
    ///
    /// For non-sharded caches, this is equivalent to save() (which already
    /// holds a single global read lock for the whole serialization).
    ///
    /// \param handle_drain_timeout  Maximum time to wait for outstanding
    ///        read_handles to be released after shutdown. If the timeout
    ///        elapses with handles still active, throws std::runtime_error
    ///        (the snapshot would not be safely reproducible). Inspect
    ///        active_handle_count() before calling to avoid this.
    /// \throws std::runtime_error if handle drain times out.
    template <typename Rep, typename Period>
    std::vector<uint8_t> save_atomic(
        std::chrono::duration<Rep, Period> handle_drain_timeout);

    /// T-G15: Atomic-or-per-shard snapshot with graceful degradation.
    ///
    /// Attempts `save_atomic(handle_drain_timeout)` first. If the handle
    /// drain times out (e.g. a long-lived read_handle is held by another
    /// thread), the method falls back to `save_per_shard()` instead of
    /// throwing — the caller gets a best-effort snapshot rather than no
    /// snapshot at all.
    ///
    /// The returned struct carries:
    ///   - `data`: the serialized snapshot (atomic or per-shard format).
    ///   - `atomic`: true if `save_atomic` succeeded, false if the call
    ///     degraded to `save_per_shard`.
    ///
    /// Use this when:
    ///   - You want atomic snapshots when possible but cannot tolerate
    ///     `save_atomic`'s timeout exception (e.g. scheduled backups
    ///     that must produce a snapshot every run).
    ///   - You want to log degradation events without wrapping every
    ///     call site in try/catch.
    ///
    /// The fallback snapshot is loaded via `load_per_shard()` (same as
    /// a regular per-shard snapshot). The atomic snapshot can also be
    /// loaded via `load_per_shard()` (same binary format).
    struct save_atomic_result {
        std::vector<uint8_t> data;
        bool atomic = false;  // true = save_atomic succeeded; false = save_per_shard fallback
    };
    template <typename Rep, typename Period>
    save_atomic_result save_atomic_or_per_shard(
        std::chrono::duration<Rep, Period> handle_drain_timeout);

    // --------------------------------------------------------------------
    // Statistics and callbacks
    // --------------------------------------------------------------------

    stats_type stats_snapshot() const {
        // P1-4: Track scrape frequency so the destructor can warn about
        // high-frequency scraping without metrics cache enabled. Use
        // relaxed ordering — this is a best-effort heuristic counter.
        scrape_count_.fetch_add(1, std::memory_order_relaxed);
        // P2-E: Fast path — return the cached snapshot if metrics caching
        // is enabled and the cache is still fresh. The cache is refreshed
        // lazily by the background drain worker (or by an explicit
        // refresh_metrics_cache() call) so the typical scrape path is a
        // single relaxed atomic load + copy. When the cache is stale or
        // disabled, we fall back to the original computation below.
        if (metrics_cache_enabled_.load(std::memory_order_acquire)) {
            auto cached = stats_cache_.load(std::memory_order_acquire);
            if (cached) return *cached;
            // Stale or not yet populated — rebuild and store for subsequent
            // callers. Concurrent rebuilds serialize on the
            // metrics_cache_lock_ below so only one thread pays the cost.
            std::lock_guard<std::mutex> lock(metrics_cache_lock_);
            // Double-check after acquiring the lock — another thread may
            // have already rebuilt while we were waiting.
            cached = stats_cache_.load(std::memory_order_relaxed);
            if (cached) return *cached;
            auto rebuilt = std::make_shared<stats_type>(build_stats_snapshot_uncached());
            stats_cache_.store(rebuilt, std::memory_order_release);
            return *rebuilt;
        }
        return build_stats_snapshot_uncached();
    }

private:
    /// P2-E: Build the stats snapshot without consulting the cache.
    stats_type build_stats_snapshot_uncached() const {
        // P1-1: Refresh hash table diagnostics (including rehash stats)
        // before taking the snapshot, so the returned stats reflect the
        // latest rehash activity. This is O(bucket_count) but callers
        // typically invoke stats_snapshot() infrequently (e.g. once per
        // scrape interval).
        if constexpr (is_striped) {
            for (std::size_t i = 0; i < mm_.num_shards(); ++i) {
                mm_.shard(i).refresh_hash_stats();
            }
        } else {
            mm_.refresh_hash_stats();
        }

        auto snap = mm_.stats();
        // Task 6: populate active_handle_count from the per-T static counter
        // in read_handle<Value>, and tls_ring_backlog from the calling
        // thread's TLS access ring. Note: TLS ring backlog is per-thread
        // by design — the snapshot reports the calling thread's pending
        // promotions only. Aggregating across threads would require a
        // global registry walk, which is too expensive for the snapshot
        // hot path.
        // Task C: when per-cache tracking is enabled, use the per-cache
        // counter instead of the global per-T counter.
        if (per_cache_handle_tracking_) {
            snap.active_handle_count.store(
                per_cache_stats_.active_handle_count.load(std::memory_order_relaxed),
                std::memory_order_relaxed);
        } else {
            snap.active_handle_count.store(
                read_handle<Value>::active_count(), std::memory_order_relaxed);
        }
        snap.tls_ring_backlog.store(
            tls_access_ring<Key>::instance().size(), std::memory_order_relaxed);
        // Drain the global dropped-promotion counter (atomic exchange) and
        // accumulate the delta into tls_ring_dropped_promotions. Using
        // drain_dropped_count() (not dropped_count_all_threads()) keeps the
        // tls_ring.hpp counter near zero between snapshots, bounding its
        // memory; fetch_add makes the cache_stats field cumulative across
        // snapshots, matching the semantics of hits/misses/insertions.
        snap.tls_ring_dropped_promotions.value.fetch_add(
            tls_access_ring<Key>::drain_dropped_count(), std::memory_order_relaxed);
        // P0-1: refresh hazptr/EBR reclaim diagnostics from the global
        // domains so stats consumers can monitor deferred reclamation
        // health. pending_count is a snapshot; values may change between
        // two reads, but that is acceptable for monitoring.
        std::size_t pending = detail::hazptr_domain::default_domain().pending_count()
                            + ebr_pending_count();
        std::size_t total = detail::hazptr_domain::default_domain().reclaim_total()
                          + ebr_reclaim_total();
        snap.reclaim_pending_count.store(pending, std::memory_order_relaxed);
        // reclaim_total in stats is cumulative across cache instances
        // sharing the global domain; combine with any per-cache delta
        // already accumulated (e.g. from try_reclaim_now()).
        snap.reclaim_total.store(
            per_cache_stats_.reclaim_total.load(std::memory_order_relaxed)
                + total,
            std::memory_order_relaxed);
        snap.reclaim_invocation_count.store(
            per_cache_stats_.reclaim_invocation_count.load(std::memory_order_relaxed),
            std::memory_order_relaxed);
        snap.reclaim_freed_bytes.store(
            per_cache_stats_.reclaim_freed_bytes.load(std::memory_order_relaxed),
            std::memory_order_relaxed);
        // T-M1: singleflight stampede coalescing diagnostics — cumulative
        // count of follower requests collapsed into a leader's provider
        // call. Sourced from the per-cache singleflight_tracker.
        snap.stampede_coalesced_count.store(
            singleflight_.coalesced_count(),
            std::memory_order_relaxed);
        return snap;
    }

public:
    // --------------------------------------------------------------------
    // P2-E: Metrics cache (stats_snapshot / prometheus_text)
    // --------------------------------------------------------------------
    //
    // High-frequency metrics scraping (e.g. Prometheus pulling every 5s
    // with many caches per process) can become a non-trivial CPU cost:
    //   - `stats_snapshot()` walks every shard to refresh hash-table
    //     diagnostics (O(num_shards * bucket_count) atomic reads).
    //   - `prometheus_text()` allocates and formats ~4-16KB of text on
    //     every call.
    //
    // The metrics cache lets operators decouple the scrape rate from the
    // rebuild rate: a background worker (or explicit `refresh_*` call)
    // rebuilds the snapshot and Prometheus text at a configured interval,
    // while scrape handlers simply load a `shared_ptr` to the latest
    // cached copy. This is especially valuable for:
    //   - Multi-tenant cache pools with hundreds of `unified_cache`
    //     instances, each scraped individually.
    //   - Dashboards that poll `stats_snapshot()` at sub-second intervals
    //     for live tail-latency views.
    //
    // When the cache is disabled (default), behavior is unchanged: every
    // call rebuilds the snapshot/text from scratch. When enabled, callers
    // get the most recent cached copy — which may be stale by up to the
    // refresh interval, but is consistent between `stats_snapshot()` and
    // `prometheus_text()` (both reflect the same underlying snapshot).
    //
    // The cached snapshot is a `shared_ptr<stats_type>` stored in an
    // `atomic_shared_ptr` (spinlock-based). Scrape readers load the
    // shared_ptr (incrementing its refcount) and copy it out under a
    // brief spinlock — this is far cheaper than walking the hash table.
    // Concurrent rebuilds serialize on `metrics_cache_lock_`.

    /// Enable or disable metrics caching. When enabled, `stats_snapshot()`
    /// and `prometheus_text()` return the most recent cached value (if
    /// fresh) instead of rebuilding on every call. Callers MUST arrange
    /// for `refresh_metrics_cache()` to be called periodically (e.g. via
    /// `start_metrics_cache_worker(interval)`) — otherwise the cache will
    /// be built once on first access and never refreshed.
    ///
    /// Default: disabled (preserves historical rebuild-on-every-call
    /// behavior for callers that depend on always-fresh metrics).
    void set_metrics_cache_enabled(bool enabled) noexcept {
        metrics_cache_enabled_.store(enabled, std::memory_order_release);
        if (!enabled) {
            // Drop cached values so the next call rebuilds from scratch.
            stats_cache_.store(nullptr, std::memory_order_release);
            prometheus_cache_.store(nullptr, std::memory_order_release);
        }
    }

    /// Whether metrics caching is currently enabled.
    bool metrics_cache_enabled() const noexcept {
        return metrics_cache_enabled_.load(std::memory_order_acquire);
    }

    /// Force an immediate rebuild of the cached `stats_snapshot()` and
    /// `prometheus_text()` outputs. Subsequent calls to those methods
    /// (while caching is enabled) will return this refreshed copy until
    /// the next `refresh_metrics_cache()` call.
    ///
    /// Safe to call from a background worker thread while concurrent
    /// readers call `stats_snapshot()` / `prometheus_text()`. The
    /// rebuild serializes on `metrics_cache_lock_`; readers see either
    /// the old or the new copy atomically (never a torn read).
    ///
    /// Returns the freshly built stats snapshot by value so the caller
    /// can inspect the rebuilt metrics (e.g. for logging or alerting).
    stats_type refresh_metrics_cache() const {
        std::lock_guard<std::mutex> lock(metrics_cache_lock_);
        auto snap = build_stats_snapshot_uncached();
        auto snap_ptr = std::make_shared<stats_type>(snap);
        stats_cache_.store(snap_ptr, std::memory_order_release);
        // Rebuild the Prometheus text from the same snapshot so both
        // views are consistent. We intentionally build the text here (under
        // the lock) so concurrent readers never see a half-rebuilt string.
        auto prom = build_prometheus_text_uncached();
        prometheus_cache_.store(
            std::make_shared<std::string>(std::move(prom)),
            std::memory_order_release);
        return snap;
    }

    /// Start a background worker that periodically refreshes the metrics
    /// cache. The worker calls `refresh_metrics_cache()` every
    /// `interval` (clamped to >= 50ms to avoid busy-looping). Automatically
    /// enables metrics caching if it was disabled.
    ///
    /// Calling this when a worker is already running is a no-op (the
    /// existing worker is reused with the new interval). The worker is
    /// stopped automatically on cache destruction.
    ///
    /// Recommended interval: 1-5 seconds for production Prometheus scraping.
    /// Shorter intervals increase CPU overhead; longer intervals increase
    /// staleness. With a 1s interval and 64-shard production_cache, the
    /// rebuild cost is ~50-200us per tick — negligible compared to the
    /// cost of a single scrape rebuilding on every call.
    void start_metrics_cache_worker(std::chrono::milliseconds interval) {
        if (interval < std::chrono::milliseconds(50)) {
            interval = std::chrono::milliseconds(50);
        }
        metrics_cache_enabled_.store(true, std::memory_order_release);
        // Reuse the existing worker if present — just update its interval.
        if (metrics_cache_worker_) {
            metrics_cache_worker_->set_interval(interval);
            return;
        }
        // Use a periodic_worker that captures `this`. The cache object
        // outlives the worker (declared as a unique_ptr member destroyed
        // before the cache itself), so `this` remains valid.
        metrics_cache_worker_ = std::make_unique<detail::periodic_worker>(
            [this]() { refresh_metrics_cache(); }, interval);
    }

    /// Stop the background metrics cache worker if running. The cached
    /// values remain valid (and continue to be returned by
    /// `stats_snapshot()` / `prometheus_text()`) until
    /// `set_metrics_cache_enabled(false)` or `refresh_metrics_cache()`
    /// is called. Stopping the worker only halts the periodic refresh.
    void stop_metrics_cache_worker() noexcept {
        metrics_cache_worker_.reset();
    }

    /// Whether the background metrics cache worker is currently running.
    bool metrics_cache_worker_running() const noexcept {
        return metrics_cache_worker_ != nullptr;
    }

    // --------------------------------------------------------------------
    // P1-4: Runtime latency tracking toggle
    // --------------------------------------------------------------------

    /// Enable or disable latency histogram tracking on the hot path.
    /// When disabled, scope_latency_timer skips clock reads entirely,
    /// reducing get()/set() overhead by ~10-20%.
    /// Default: enabled (for backward compatibility).
    ///
    /// T12.3: When transitioning enabled → disabled, also calls
    /// `release_memory()` on all per-shard histograms to clear
    /// accumulated data. This prevents stale latency samples from
    /// being reported via `stats_snapshot()` / `prometheus_text()`
    /// after tracking is turned off.
    void set_latency_tracking(bool enabled) {
        if constexpr (is_striped) {
            for (std::size_t i = 0; i < mm_.num_shards(); ++i) {
                auto& s = mm_.shard(i).stats();
                const bool was_enabled =
                    s.latency_tracking_enabled.load(std::memory_order_acquire);
                s.latency_tracking_enabled.store(
                    enabled, std::memory_order_release);
                // T12.3: on disable, release histogram data.
                if (was_enabled && !enabled) {
                    s.get_latency.release_memory();
                    s.set_latency.release_memory();
                    s.read_lock_wait_latency.release_memory();
                    s.write_lock_wait_latency.release_memory();
                    s.eviction_search_steps_hist.release_memory();
                }
            }
        } else {
            auto& s = mm_.stats();
            const bool was_enabled =
                s.latency_tracking_enabled.load(std::memory_order_acquire);
            s.latency_tracking_enabled.store(
                enabled, std::memory_order_release);
            if (was_enabled && !enabled) {
                s.get_latency.release_memory();
                s.set_latency.release_memory();
                s.read_lock_wait_latency.release_memory();
                s.write_lock_wait_latency.release_memory();
                s.eviction_search_steps_hist.release_memory();
            }
        }
    }

    /// Check if latency tracking is currently enabled.
    /// For striped caches, returns true if any shard has it enabled.
    bool is_latency_tracking_enabled() const {
        if constexpr (is_striped) {
            for (std::size_t i = 0; i < mm_.num_shards(); ++i) {
                if (mm_.shard(i).stats().latency_tracking_enabled.load(
                        std::memory_order_acquire)) {
                    return true;
                }
            }
            return false;
        } else {
            return mm_.stats().latency_tracking_enabled.load(
                std::memory_order_acquire);
        }
    }

    // ----------------------------------------------------------------
    // P2-2: Runtime lock order checking toggle
    // ----------------------------------------------------------------
    //
    // When enabled, lock acquisitions are validated against rank ordering
    // to detect potential deadlocks. Requires compilation with
    // -DLRU_DEBUG_LOCK_ORDER=ON; otherwise these methods are no-ops.
    // Default: disabled (even in debug builds) to avoid overhead.
    // Enable at runtime for on-demand deadlock detection.

    void set_lock_order_checking(bool enabled) noexcept {
#ifdef LRU_DEBUG_LOCK_ORDER
        if constexpr (is_striped) {
            for (std::size_t i = 0; i < striped_mutex_.size(); ++i) {
                striped_mutex_.mutex_at(i).set_lock_order_checking(enabled);
            }
        } else if constexpr (is_thread_safe) {
            mutex_.set_lock_order_checking(enabled);
        }
#else
        (void)enabled;  // no-op when LRU_DEBUG_LOCK_ORDER is not defined
#endif
    }

    bool lock_order_checking_enabled() const noexcept {
#ifdef LRU_DEBUG_LOCK_ORDER
        if constexpr (is_striped) {
            // Return true if any stripe has it enabled
            for (std::size_t i = 0; i < striped_mutex_.size(); ++i) {
                if (striped_mutex_.mutex_at(i).lock_order_checking_enabled()) {
                    return true;
                }
            }
            return false;
        } else if constexpr (is_thread_safe) {
            return mutex_.lock_order_checking_enabled();
        } else {
            return false;
        }
#else
        return false;  // always false when LRU_DEBUG_LOCK_ORDER is not defined
#endif
    }

    // --------------------------------------------------------------------
    // P1-7: Pending deletion count (items removed but pinned by read_handles)
    // --------------------------------------------------------------------

    /// Number of items in pending-deletion state. These are items that
    /// have been explicitly removed (via remove/flush/force_del) but are
    /// still pinned by active read_handles. Memory will be reclaimed once
    /// all handles are released.
    ///
    /// Best-effort count — may race with concurrent writes. For monitoring
    /// and leak detection only.
    std::size_t pending_deletion_count() const {
        return mm_.pending_deletion_count();
    }

    /// R9: Number of explicit force-delete operations refused because the
    /// pending-deletion soft cap (mm config `max_pending_deletion`) was
    /// reached. A non-zero, growing value indicates callers are holding
    /// read_handles too long (potential handle leak).
    std::size_t pending_deletion_skipped_count() const {
        if constexpr (requires { mm_.pending_deletion_skipped_count(); }) {
            return mm_.pending_deletion_skipped_count();
        }
        return 0;
    }

    // --------------------------------------------------------------------
    // Task 8: Async callback dispatch toggle
    // --------------------------------------------------------------------
    //
    // When enabled, callback events are dispatched by a background worker
    // thread instead of inline during flush_pending(). This decouples
    // callback execution from the cache hot path, reducing tail latency
    // when callbacks perform expensive work (logging, metrics, I/O).
    //
    // For sharded caches, the toggle propagates to all shards' callback
    // managers. Each shard gets its own worker thread.
    //
    // Disabling blocks until all queued events have been dispatched.
    void set_async_callbacks(bool enabled) {
        if constexpr (requires { mm_.shard(0); }) {
            mm_.callbacks().set_async_mode(enabled);
            for (std::size_t i = 0; i < mm_.num_shards(); ++i) {
                mm_.shard(i).callbacks().set_async_mode(enabled);
            }
        } else {
            mm_.callbacks().set_async_mode(enabled);
        }
    }

    /// Check if async callback dispatch is currently enabled.
    bool is_async_callbacks() const noexcept {
        return mm_.callbacks().is_async_mode();
    }

    /// P1-10: Current async callback queue size, aggregated across all
    /// shards for striped caches. Returns 0 when async callbacks are
    /// disabled. Used by prometheus_text() to export
    /// lru_cache_async_queue_size.
    std::size_t async_queue_size_aggregated() const noexcept {
        if constexpr (requires { mm_.shard(0); }) {
            std::size_t total = mm_.callbacks().async_queue_size();
            for (std::size_t i = 0; i < mm_.num_shards(); ++i) {
                total += mm_.shard(i).callbacks().async_queue_size();
            }
            return total;
        } else {
            return mm_.callbacks().async_queue_size();
        }
    }

    // --------------------------------------------------------------------
    // T-M1: singleflight / cache stampede protection
    // --------------------------------------------------------------------
    //
    // When enabled, get_or_fetch() / try_get_or_fetch() coalesce
    // concurrent misses on the same key: the first miss becomes the
    // "leader" and executes the provider; concurrent misses become
    // "followers" that block until the leader completes, then receive
    // the leader's result. This prevents thundering-herd provider
    // invocations (cache stampede) on hot keys, especially at TTL
    // expiry boundaries.
    //
    // Default: disabled, to preserve the historical semantics where
    // every miss independently calls the provider. Enable when the
    // provider is idempotent and concurrent coalescing is desired.
    //
    // Overhead when disabled: one relaxed atomic bool load on the miss
    // path. Overhead when enabled on the hit path: zero (singleflight
    // is only consulted after a confirmed miss).

    /// Enable or disable singleflight coalescing on
    /// get_or_fetch() / try_get_or_fetch(). Default: disabled.
    void set_singleflight_enabled(bool enabled) noexcept {
        singleflight_enabled_.store(enabled, std::memory_order_release);
    }

    /// Whether singleflight coalescing is currently enabled.
    bool is_singleflight_enabled() const noexcept {
        return singleflight_enabled_.load(std::memory_order_acquire);
    }

    /// T-M1: Cumulative count of follower requests that were collapsed
    /// into a leader's in-flight provider call. Aggregated across all
    /// keys and shards. Resets only on cache destruction (cumulative
    /// counter, like hits/misses).
    std::size_t stampede_coalesced_count() const noexcept {
        return singleflight_.coalesced_count();
    }

    /// T-M1: Current number of in-flight singleflight keys (keys whose
    /// leader is currently executing the provider). Diagnostic only —
    /// acquires all shard locks, so do not call on the hot path.
    std::size_t singleflight_inflight_count() const {
        return singleflight_.inflight_count();
    }

    // --------------------------------------------------------------------
    // Task 11: Graceful shutdown protocol
    // --------------------------------------------------------------------
    //
    // shutdown() transitions the cache into a closed state. After shutdown:
    //   - get() throws std::runtime_error (cache is closed).
    //   - set() throws std::runtime_error (cache is closed).
    //   - try_get() returns std::nullopt (non-blocking, no throw).
    //   - peek() and iteration continue to work (read-only, no state change).
    //   - The TTL cleaner and event drain worker are stopped.
    //   - Pending TLS access-ring promotions are drained and callbacks flushed.
    //
    // shutdown() is idempotent: calling it more than once is a no-op.
    // The cache cannot be re-opened; destroy and reconstruct instead.
    //
    // active_handle_count() reports the number of live read_handle<Value>
    // objects across all threads (via the per-T static atomic counter in
    // read_handle<Value>). A graceful drain waits for this to reach zero
    // before destroying the cache — the caller is responsible for joining
    // any threads that may hold handles, then polling active_handle_count().

    /// Shut down the cache. After this call, get()/set() throw and the
    /// background workers are stopped. Idempotent.
    void shutdown() {
        // Idempotent: use exchange to detect first shutdown.
        if (closed_.exchange(true, std::memory_order_acq_rel)) {
            return;  // already shut down
        }
        // T-G10: Signal handle-release path that shutdown is in progress,
        // so the last read_handle::release() notifies shutdown_and_wait().
        active_handle_notifier_.shutdown_in_progress.store(true, std::memory_order_release);
        // T-G10 subtask 4: stop the OS memory sampler so its background
        // thread doesn't race with teardown. Idempotent no-op if never
        // started.
        memory_monitor_.stop_os_sampling();
        // P-CRIT-1 (T-C1): Stop the rehash balancer FIRST so it cannot
        // touch mm_ members while we drain/tear down below. Previously
        // this stop() was defined but never invoked from shutdown() or
        // the destructor, causing use-after-free when the cache was
        // destroyed after start_background_rehash_balancer() was called.
        // Idempotent (no-op if never started).
        stop_background_rehash_balancer();
        // T14.1: Compact path — delegate shutdown to compact_cache, then
        // skip the mm_-specific drain/callback flush below.
        if constexpr (Trait::is_compact) {
            stop_ttl_cleaner();
            stop_event_drain();
            // P2-E: stop metrics cache worker on compact path too.
            stop_metrics_cache_worker();
            compact().shutdown();
            return;
        }
        // 1. Stop background workers so they don't race with the drain.
        stop_ttl_cleaner();
        stop_event_drain();
        // P2-E: stop the metrics cache worker so it stops referencing
        // stats_cache_ / prometheus_cache_ during teardown.
        stop_metrics_cache_worker();
        // 2. Drain deferred LRU promotions one final time.
        drain_access_ring();
        // 3. Cross-thread flush: drain ALL registered threads' TLS rings
        //    so no pending promotions are lost when the calling thread is
        //    the only one that will ever call drain_access_ring() again.
        //    The keys are batch-promoted below.
        {
            auto all_drained = tls_access_ring<Key>::drain_all_threads();
            if (!all_drained.keys.empty()) {
                promote_keys(all_drained.keys);
            }
        }
        // 4. Flush any pending callbacks collected before shutdown.
        if constexpr (requires { mm_.flush_pending_all(); }) {
            mm_.flush_pending_all();
        } else {
            mm_.callbacks().flush_pending();
        }
    }

    /// T7.3: Shut down the cache, optionally waiting for all other
    /// threads' TLS access rings to be drained into the backup buffer.
    ///
    /// When `wait_for_drain == true`, this issues a cross-thread flush
    /// request and blocks (up to `timeout`) until every other thread
    /// has serviced it, then performs a final `drain_all_threads()` to
    /// retrieve the keys they pushed into the backup buffer and
    /// promotes them. This prevents loss of pending LRU promotions
    /// during process exit if other threads are about to be killed
    /// before they get a chance to call `record_access()` again.
    ///
    /// When `wait_for_drain == false`, behaves exactly like `shutdown()`.
    ///
    /// \return true if the wait succeeded (or was not requested);
    ///         false if the timeout elapsed with pending drains still
    ///         outstanding. The cache is shut down either way.
    template <typename Rep, typename Period>
    bool shutdown(bool wait_for_drain,
                  std::chrono::duration<Rep, Period> timeout) {
        if (!wait_for_drain) {
            shutdown();
            return true;
        }
        // Idempotent check: if already shut down, just do the wait.
        const bool already_closed =
            closed_.load(std::memory_order_acquire);
        if (!already_closed) {
            shutdown();
        }
        // Issue flush requests and wait for other threads to service them.
        const bool ok = tls_access_ring<Key>::drain_all_threads_sync(timeout);
        // Pick up whatever keys other threads pushed into the backup buffer.
        auto all_drained = tls_access_ring<Key>::drain_all_threads();
        if (!all_drained.keys.empty()) {
            promote_keys(all_drained.keys);
        }
        return ok;
    }

    /// Check if the cache has been shut down.
    bool is_shutdown() const noexcept {
        return closed_.load(std::memory_order_acquire);
    }

    /// Number of live read_handle<Value> objects across all threads.
    /// Use this to wait for outstanding handles to be released before
    /// destroying a shut-down cache.
    std::size_t active_handle_count() const noexcept {
        // Task C: prefer per-cache count when tracking is enabled.
        if (per_cache_handle_tracking_) {
            return per_cache_stats_.active_handle_count.load(std::memory_order_relaxed);
        }
        return read_handle<Value>::active_count();
    }

    // --------------------------------------------------------------------
    // P1-6: Graceful shutdown with active handle wait
    // --------------------------------------------------------------------

    /// Shut down the cache and wait for all active read_handles to be
    /// released. Returns true if all handles were released within the
    /// timeout; false if the timeout expired with handles still active.
    ///
    /// After this returns true, it is safe to destroy the cache.
    /// After this returns false, the cache is shut down but some handles
    /// may still be alive — destroying the cache would be UB.
    ///
    /// Idempotent: if the cache is already shut down, just waits.
    ///
    /// O10: Per-cache handle tracking is default-on, and the per-T global
    /// counter is now also default-on in release builds, so this function
    /// is reliable out of the box. It can only return `false` (without
    /// waiting) when the caller has explicitly disabled BOTH per-cache
    /// tracking (`set_per_cache_handle_tracking(false)`) AND global
    /// tracking (`read_handle<T>::enable_global_handle_tracking(false)`);
    /// in that case use `force_wait_handles()` instead.
    ///
    /// \param timeout  Maximum time to wait for handles to drain.
    /// \param poll_interval  Reserved for API compatibility; ignored.
    ///   T-G10 replaced the busy-poll with a condition_variable wait, so
    ///   the poll interval is no longer used — wakeup is immediate on the
    ///   last handle release.
    /// \return true if all handles were released before the timeout.
    template <typename Rep, typename Period>
    bool shutdown_and_wait(
        std::chrono::duration<Rep, Period> timeout,
        std::chrono::milliseconds /*poll_interval*/ = std::chrono::milliseconds(1)) {
        shutdown();
        // T4.1: cannot reliably count handles without any tracking source.
        if (!tracking_reliable_for_wait()) {
            return false;  // caller must use force_wait_handles() instead
        }
        // T-G10: Wait on a condition_variable instead of busy-polling.
        // read_handle::release() calls notify_all() when the last handle
        // is dropped during shutdown, so this returns immediately without
        // waiting out the full timeout. A 10ms ceiling on wait_for guards
        // against lost-wakeup races (e.g. the notify firing between the
        // predicate check and the wait).
        auto& n = active_handle_notifier_;
        std::unique_lock<std::mutex> lk(n.mtx);
        auto deadline = std::chrono::steady_clock::now() + timeout;
        while (active_handle_count() > 0) {
            if (n.cv.wait_until(lk, std::min(deadline,
                    std::chrono::steady_clock::now() + std::chrono::milliseconds(10)))
                == std::cv_status::timeout) {
                if (std::chrono::steady_clock::now() >= deadline) {
                    return false;  // overall timeout elapsed
                }
            }
        }
        return true;
    }

    /// T4.1: Result of a shutdown wait operation. Distinguishes between
    /// a successful drain, a timeout, and the "tracking disabled" case
    /// where the wait could not be performed reliably.
    enum class shutdown_status {
        success,              // all handles drained within timeout
        timeout,              // handles still active after timeout
        tracking_disabled,    // per-cache tracking off AND global counter
                              // not maintained — caller must use
                              // force_wait_handles() or join handle-holding
                              // threads manually before destroying the cache
    };

    /// T4.1: Same as `shutdown_and_wait()` but returns a detailed status
    /// code so the caller can distinguish "timeout" from "tracking
    /// disabled" cases.
    template <typename Rep, typename Period>
    shutdown_status shutdown_and_wait_detailed(
        std::chrono::duration<Rep, Period> timeout,
        std::chrono::milliseconds /*poll_interval*/ = std::chrono::milliseconds(1)) {
        shutdown();
        if (!tracking_reliable_for_wait()) {
            return shutdown_status::tracking_disabled;
        }
        // T-G10: condition_variable wait — see shutdown_and_wait().
        auto& n = active_handle_notifier_;
        std::unique_lock<std::mutex> lk(n.mtx);
        auto deadline = std::chrono::steady_clock::now() + timeout;
        while (active_handle_count() > 0) {
            if (n.cv.wait_until(lk, std::min(deadline,
                    std::chrono::steady_clock::now() + std::chrono::milliseconds(10)))
                == std::cv_status::timeout) {
                if (std::chrono::steady_clock::now() >= deadline) {
                    return shutdown_status::timeout;
                }
            }
        }
        return shutdown_status::success;
    }

    /// T4.2: Force-wait for outstanding handles using the per-T global
    /// atomic counter, even when per-cache tracking is disabled.
    ///
    /// This is intended for process-exit / graceful-shutdown scenarios
    /// where the caller cannot easily join all handle-holding threads
    /// but needs to ensure no `read_handle<Value>` objects outlive the
    /// cache. The flow is:
    ///
    ///   1. `read_handle<Value>::enable_global_handle_tracking(true)` —
    ///      new handles increment the global counter.
    ///   2. `shutdown()` — reject new operations, drain TLS rings.
    ///   3. Poll `read_handle<Value>::active_count()` until 0 or timeout.
    ///
    /// T7.4: After step 2, also issues a synchronous TLS access-ring
    /// drain so that pending LRU promotions from other threads are
    /// not lost. The drain wait reuses the same `timeout` budget.
    ///
    /// Caveats:
    ///   - Only handles created AFTER step 1 are counted. Pre-existing
    ///     handles (created before enabling global tracking in a release
    ///     build) are invisible. For a clean shutdown, enable global
    ///     tracking at process start, or call this from the same thread
    ///     that has never held a handle to that Value type.
    ///   - The global counter is per-Value-type (`read_handle<T>` is
    ///     templated on T), so it counts handles across ALL cache
    ///     instances of the same T. If multiple caches share T, the wait
    ///     may not converge until all caches' handles are released.
    ///
    /// \return true if the global counter reached 0 within timeout.
    template <typename Rep, typename Period>
    bool force_wait_handles(
        std::chrono::duration<Rep, Period> timeout,
        std::chrono::milliseconds poll_interval = std::chrono::milliseconds(1)) {
        // O10: Global tracking is now default-on in release builds, so this
        // call is usually a no-op. Keep it for callers that have explicitly
        // disabled tracking via `enable_global_handle_tracking(false)` and
        // then need force_wait_handles() to work — re-enabling here ensures
        // new handles are counted even if the caller previously opted out.
        read_handle<Value>::enable_global_handle_tracking(true);
        shutdown();
        // T7.4: Synchronously drain other threads' TLS access rings so
        // pending promotions are not lost on process exit. Use half the
        // timeout budget so we leave time for the handle wait below.
        auto tls_deadline = std::chrono::steady_clock::now() + timeout / 2;
        (void)tls_access_ring<Key>::drain_all_threads_sync(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                tls_deadline - std::chrono::steady_clock::now()));
        auto all_drained = tls_access_ring<Key>::drain_all_threads();
        if (!all_drained.keys.empty()) {
            promote_keys(all_drained.keys);
        }
        auto deadline = std::chrono::steady_clock::now() + timeout;
        while (read_handle<Value>::active_count() > 0) {
            if (std::chrono::steady_clock::now() >= deadline) {
                return false;
            }
            std::this_thread::sleep_for(poll_interval);
        }
        return true;
    }

    /// T4.1: Helper — true if `active_handle_count()` is currently
    /// reliable enough to wait on. Returns false when neither per-cache
    /// tracking nor the per-T global counter is being maintained.
    bool tracking_reliable_for_wait() const noexcept {
        if (per_cache_handle_tracking_) return true;
        return read_handle<Value>::is_global_handle_tracking_enabled();
    }

    /// Task C: Enable/disable per-cache active_handle_count tracking.
    ///
    /// When enabled, every read_handle returned by get/peek/try_get/etc.
    /// is attached to this cache's per_cache_stats_ counter, isolating
    /// active_handle_count to this cache instance (rather than the global
    /// per-Value-type counter in read_handle<T>). Useful for diagnosing
    /// handle leaks or excessive pinning when multiple cache instances
    /// share the same Value type.
    ///
    /// P2-1: Default is now `true` (production behavior). The per-T
    /// global counter in read_handle<T> is only maintained under
    /// `-DLRU_DEBUG=1`; in release builds it is a no-op, so per-cache
    /// tracking is the only source of active_handle_count. Disable
    /// only for benchmarks that don't need the metric.
    ///
    /// Once enabled, existing live handles are NOT retroactively attached;
    /// only handles created after the call are tracked.
    void set_per_cache_handle_tracking(bool enabled) noexcept {
        per_cache_handle_tracking_ = enabled;
    }

    bool is_per_cache_handle_tracking_enabled() const noexcept {
        return per_cache_handle_tracking_;
    }

    // --------------------------------------------------------------------
    // Task 5: Prometheus text format export
    // --------------------------------------------------------------------
    //
    // Emits a Prometheus 0.0.4 text exposition payload describing all
    // cache counters, gauges, and latency histograms. Suitable for
    // scraping by Prometheus via a /metrics HTTP endpoint.
    //
    // Metric naming follows Prometheus conventions:
    //   - lru_cache_hits_total (counter)
    //   - lru_cache_misses_total (counter)
    //   - lru_cache_insertions_total (counter)
    //   - lru_cache_evictions_total (counter)
    //   - lru_cache_size (gauge)
    //   - lru_cache_memory_bytes (gauge)
    //   - lru_cache_max_size (gauge)
    //   - lru_cache_active_handles (gauge)
    //   - lru_cache_tls_ring_backlog (gauge)
    //   - lru_cache_get_latency_ns_bucket{le="..."} (histogram)
    //   - lru_cache_set_latency_ns_bucket{le="..."} (histogram)
    //
    // P2-4: Per-shard load-factor detail. By default only aggregate
    // gauges are emitted (worst / avg / p95 across shards) to keep the
    // scrape payload compact for caches with 64+ shards. Call
    // `set_prometheus_per_shard_detail(true)` to emit the legacy
    // `lru_hash_load_factor_per_shard{shard="N"}` series (one line per
    // shard); this is recommended only for ad-hoc diagnosis via the
    // diagnostics() dump, not for production scraping.
    //
    // The output is multi-line text terminated by '\n'. For sharded caches,
    // counters are aggregated across shards.
    //
    // P2-E: When metrics caching is enabled via `set_metrics_cache_ttl()`,
    // callers scraping at high frequency (e.g. Prometheus every 5s) hit a
    // pre-built snapshot + pre-formatted string instead of recomputing
    // hash-table diagnostics and rebuilding the Prometheus text on every
    // call. The cache is refreshed lazily on the first call after TTL
    // expiry; concurrent callers serialize on a brief spinlock so only one
    // rebuild is in flight at a time.
    //
    // P2-4: When `prometheus_per_shard_detail_` is false (default), the
    // output includes only aggregate load-factor gauges
    // (`lru_hash_load_factor_worst`, `lru_hash_load_factor_avg`,
    // `lru_hash_load_factor_p95`) — not the per-shard labelled series.
    // This keeps the scrape payload O(1) regardless of shard count.
    std::string prometheus_text() const {
        // P2-E: try the cached formatted string first.
        if (metrics_cache_enabled_.load(std::memory_order_acquire)) {
            auto cached = prometheus_cache_.load(std::memory_order_acquire);
            if (cached) return *cached;
            // Cache miss or expired — fall through to rebuild below, then
            // store the result for subsequent callers.
            std::string out = build_prometheus_text_uncached();
            prometheus_cache_.store(
                std::make_shared<std::string>(std::move(out)),
                std::memory_order_release);
            return out;
        }
        return build_prometheus_text_uncached();
    }

    /// P2-4: Toggle per-shard detail in `prometheus_text()`.
    ///
    /// When false (default), only aggregate gauges are emitted:
    ///   - `lru_hash_load_factor_worst` — max load factor across shards
    ///   - `lru_hash_load_factor_avg`  — arithmetic mean
    ///   - `lru_hash_load_factor_p95`  — 95th percentile
    /// This keeps the scrape payload O(1) regardless of shard count.
    ///
    /// When true, additionally emits one labelled series per shard:
    ///   - `lru_hash_load_factor_per_shard{shard="N"}`
    /// Useful for ad-hoc diagnosis; not recommended for production
    /// scraping on caches with 64+ shards (payload bloat + Prometheus
    /// storage cost).
    ///
    /// For non-sharded caches this flag has no effect (only the
    /// worst-shard gauge is emitted, which equals the single shard's
    /// load factor).
    void set_prometheus_per_shard_detail(bool enabled) noexcept {
        prometheus_per_shard_detail_.store(enabled, std::memory_order_release);
        // P2-E: Rebuild the prometheus cache immediately with the new flag
        // value so the next scrape reflects the change. Using rebuild
        // instead of invalidation avoids a race where a concurrent scrape
        // sees a null cache and rebuilds with the OLD flag value (having
        // loaded it before this store became visible).
        if (metrics_cache_enabled_.load(std::memory_order_acquire)) {
            auto fresh = build_prometheus_text_uncached();
            prometheus_cache_.store(
                std::make_shared<std::string>(std::move(fresh)),
                std::memory_order_release);
        }
    }

    /// P2-4: Query whether per-shard detail is currently being emitted
    /// in `prometheus_text()`.
    bool prometheus_per_shard_detail() const noexcept {
        return prometheus_per_shard_detail_.load(std::memory_order_acquire);
    }

private:
    /// P2-E: Build the Prometheus exposition text without consulting the
    /// cache. Used by `prometheus_text()` on cache miss / when caching is
    /// disabled.
    std::string build_prometheus_text_uncached() const {
        auto snap = stats_snapshot();
        std::string out;
        out.reserve(4096);

        auto append = [&](std::string_view line) {
            out.append(line);
            out.push_back('\n');
        };

        // Counters
        append("# HELP lru_cache_hits_total Total cache hits.");
        append("# TYPE lru_cache_hits_total counter");
        out.append("lru_cache_hits_total ");
        out.append(std::to_string(snap.hits.value.load(std::memory_order_relaxed)));
        out.push_back('\n');

        append("# HELP lru_cache_misses_total Total cache misses.");
        append("# TYPE lru_cache_misses_total counter");
        out.append("lru_cache_misses_total ");
        out.append(std::to_string(snap.misses.value.load(std::memory_order_relaxed)));
        out.push_back('\n');

        append("# HELP lru_cache_insertions_total Total item insertions.");
        append("# TYPE lru_cache_insertions_total counter");
        out.append("lru_cache_insertions_total ");
        out.append(std::to_string(snap.insertions.value.load(std::memory_order_relaxed)));
        out.push_back('\n');

        append("# HELP lru_cache_evictions_total Total item evictions.");
        append("# TYPE lru_cache_evictions_total counter");
        out.append("lru_cache_evictions_total ");
        out.append(std::to_string(snap.evictions.value.load(std::memory_order_relaxed)));
        out.push_back('\n');

        // Gauges
        append("# HELP lru_cache_size Current number of items in the cache.");
        append("# TYPE lru_cache_size gauge");
        out.append("lru_cache_size ");
        out.append(std::to_string(snap.current_size.load(std::memory_order_relaxed)));
        out.push_back('\n');

        append("# HELP lru_cache_memory_bytes Current memory used by cached items.");
        append("# TYPE lru_cache_memory_bytes gauge");
        out.append("lru_cache_memory_bytes ");
        out.append(std::to_string(snap.current_memory.load(std::memory_order_relaxed)));
        out.push_back('\n');

        append("# HELP lru_cache_max_size Maximum number of items allowed.");
        append("# TYPE lru_cache_max_size gauge");
        out.append("lru_cache_max_size ");
        out.append(std::to_string(snap.max_size.load(std::memory_order_relaxed)));
        out.push_back('\n');

        append("# HELP lru_cache_active_handles Currently alive read_handle objects (pinned items).");
        append("# TYPE lru_cache_active_handles gauge");
        out.append("lru_cache_active_handles ");
        out.append(std::to_string(snap.active_handle_count.load(std::memory_order_relaxed)));
        out.push_back('\n');

        append("# HELP lru_cache_tls_ring_backlog Pending deferred promotions in TLS rings.");
        append("# TYPE lru_cache_tls_ring_backlog gauge");
        out.append("lru_cache_tls_ring_backlog ");
        out.append(std::to_string(snap.tls_ring_backlog.load(std::memory_order_relaxed)));
        out.push_back('\n');

        // T-M4/T-P2-4: Cross-thread TLS backlog aggregate. Unlike
        // lru_cache_tls_ring_backlog (which reflects only the calling
        // thread's ring), this is the aggregate across all threads.
        // Non-zero values with flat drain rate signal the drain worker is
        // under-provisioned. Sourced from tls_access_ring::total_backlog():
        // the hot path updates a thread-local counter and batch-flushes to
        // the shared atomic every 64 increments (bounded staleness of
        // <= 64 per thread), so it does not require a cache_stats field
        // and needs only trend-level accuracy.
        append("# HELP lru_cache_tls_ring_backlog_total Aggregate pending deferred promotions across all threads.");
        append("# TYPE lru_cache_tls_ring_backlog_total gauge");
        out.append("lru_cache_tls_ring_backlog_total ");
        out.append(std::to_string(tls_access_ring<Key>::total_backlog()));
        out.push_back('\n');

        append("# HELP lru_cache_tls_ring_dropped_promotions Cumulative deferred promotions dropped due to TLS ring overflow.");
        append("# TYPE lru_cache_tls_ring_dropped_promotions counter");
        out.append("lru_cache_tls_ring_dropped_promotions ");
        out.append(std::to_string(snap.tls_ring_dropped_promotions.value.load(std::memory_order_relaxed)));
        out.push_back('\n');

        append("# HELP lru_cache_write_lock_wait_total Total write lock contention events (slow path entries).");
        append("# TYPE lru_cache_write_lock_wait_total counter");
        out.append("lru_cache_write_lock_wait_total ");
        out.append(std::to_string(snap.write_lock_wait_count.load(std::memory_order_relaxed)));
        out.push_back('\n');

        // P1-10: Read lock wait total — sourced from the read_lock_wait_latency
        // histogram's sample count. The histogram records one sample per
        // read-lock slow-path entry, so its count() equals the total number
        // of read lock waits. This is the read-side counterpart of
        // write_lock_wait_total and is essential for diagnosing read-path
        // contention (e.g., when writer_fair mode blocks new readers).
        append("# HELP lru_cache_read_lock_wait_total Total read lock contention events (slow path entries).");
        append("# TYPE lru_cache_read_lock_wait_total counter");
        out.append("lru_cache_read_lock_wait_total ");
        out.append(std::to_string(snap.read_lock_wait_latency.count()));
        out.push_back('\n');

        // P1-10: Async callback queue size — current backlog of callback
        // events waiting for the async dispatch worker. Non-zero values
        // indicate the worker is not keeping up (or the queue max_size is
        // too small), which can cause callback delivery delays. Sourced
        // from the cache's callback_manager, aggregated across shards.
        append("# HELP lru_cache_async_queue_size Current number of callback events in the async dispatch queue.");
        append("# TYPE lru_cache_async_queue_size gauge");
        out.append("lru_cache_async_queue_size ");
        out.append(std::to_string(async_queue_size_aggregated()));
        out.push_back('\n');

        // O6: Total callback failures observed across sync + async
        // dispatch. A non-zero / growing value indicates a registered
        // callback is throwing — operators should consult the
        // on_callback_error hook (or logs) to identify the culprit.
        // Aggregated across all shards for striped caches.
        append("# HELP lru_cache_callback_errors_total Total registered callback failures (sync + async dispatch).");
        append("# TYPE lru_cache_callback_errors_total counter");
        out.append("lru_cache_callback_errors_total ");
        out.append(std::to_string(callback_error_count()));
        out.push_back('\n');

        append("# HELP lru_cache_try_lock_fail_total Total try_lock/try_lock_shared failures.");
        append("# TYPE lru_cache_try_lock_fail_total counter");
        out.append("lru_cache_try_lock_fail_total ");
        out.append(std::to_string(snap.try_lock_fail_count.load(std::memory_order_relaxed)));
        out.push_back('\n');

        append("# HELP lru_cache_pinned_skip_total Total pinned items skipped during eviction search.");
        append("# TYPE lru_cache_pinned_skip_total counter");
        out.append("lru_cache_pinned_skip_total ");
        out.append(std::to_string(snap.pinned_skip_count.load(std::memory_order_relaxed)));
        out.push_back('\n');

        append("# HELP lru_cache_ttl_expired_total Total expired items cleaned up by TTL.");
        append("# TYPE lru_cache_ttl_expired_total counter");
        out.append("lru_cache_ttl_expired_total ");
        out.append(std::to_string(snap.ttl_expired_count.load(std::memory_order_relaxed)));
        out.push_back('\n');

        // P1-10: Total TTL checks performed on the read path. The ratio
        // ttl_expired_total / ttl_checked_total gives the expiration rate
        // observed by readers — a high ratio (>10%) indicates the background
        // TTL cleaner is not running frequently enough and expired items
        // are being discovered by readers (adding latency to reads).
        append("# HELP lru_cache_ttl_checked_total Total TTL checks performed on the read path.");
        append("# TYPE lru_cache_ttl_checked_total counter");
        out.append("lru_cache_ttl_checked_total ");
        out.append(std::to_string(snap.ttl_checked_count.load(std::memory_order_relaxed)));
        out.push_back('\n');

        append("# HELP lru_cache_ttl_cleanup_backlog Expired items still remaining at last cleanup scan.");
        append("# TYPE lru_cache_ttl_cleanup_backlog gauge");
        out.append("lru_cache_ttl_cleanup_backlog ");
        out.append(std::to_string(snap.ttl_cleanup_backlog.load(std::memory_order_relaxed)));
        out.push_back('\n');

        // Latency histograms
        append("# HELP lru_cache_get_latency_ns get() latency in nanoseconds.");
        append("# TYPE lru_cache_get_latency_ns histogram");
        emit_histogram_lines(out, "lru_cache_get_latency_ns", snap.get_latency);

        append("# HELP lru_cache_set_latency_ns set() latency in nanoseconds.");
        append("# TYPE lru_cache_set_latency_ns histogram");
        emit_histogram_lines(out, "lru_cache_set_latency_ns", snap.set_latency);

        append("# HELP lru_cache_read_lock_wait_ns Read lock wait latency in nanoseconds.");
        append("# TYPE lru_cache_read_lock_wait_ns histogram");
        emit_histogram_lines(out, "lru_cache_read_lock_wait_ns", snap.read_lock_wait_latency);

        append("# HELP lru_cache_write_lock_wait_ns Write lock wait latency in nanoseconds.");
        append("# TYPE lru_cache_write_lock_wait_ns histogram");
        emit_histogram_lines(out, "lru_cache_write_lock_wait_ns", snap.write_lock_wait_latency);

        append("# HELP lru_cache_eviction_search_steps Steps taken to find an eviction victim.");
        append("# TYPE lru_cache_eviction_search_steps histogram");
        emit_histogram_lines(out, "lru_cache_eviction_search_steps", snap.eviction_search_steps_hist);

        // P1-1: Rehash diagnostics — track hash table expansion frequency,
        // duration, and migration volume. High rehash_count or long
        // rehash_total_time_ns indicates frequent capacity growth causing
        // tail latency spikes. Use reserve() to pre-size the table.
        append("# HELP lru_cache_rehash_total Total hash table rehash operations.");
        append("# TYPE lru_cache_rehash_total counter");
        out.append("lru_cache_rehash_total ");
        out.append(std::to_string(snap.rehash_count.load(std::memory_order_relaxed)));
        out.push_back('\n');

        append("# HELP lru_cache_rehash_time_ns_total Cumulative time spent in rehash operations (nanoseconds).");
        append("# TYPE lru_cache_rehash_time_ns_total counter");
        out.append("lru_cache_rehash_time_ns_total ");
        out.append(std::to_string(snap.rehash_total_time_ns.load(std::memory_order_relaxed)));
        out.push_back('\n');

        append("# HELP lru_cache_rehash_migrated_items_total Total items migrated during rehash operations.");
        append("# TYPE lru_cache_rehash_migrated_items_total counter");
        out.append("lru_cache_rehash_migrated_items_total ");
        out.append(std::to_string(snap.rehash_migrated_items.load(std::memory_order_relaxed)));
        out.push_back('\n');

        // T13.4: Hash load factor and overload metrics.
        // Exported as gauges — current load factor of the worst shard,
        // configured threshold, and cumulative overload events.
        append("# HELP lru_hash_load_factor Current hash table load factor (worst shard).");
        append("# TYPE lru_hash_load_factor gauge");
        out.append("lru_hash_load_factor ");
        out.append(std::to_string(snap.hash_load_factor.load(std::memory_order_relaxed)));
        out.push_back('\n');

        // P2-4: Per-shard aggregate gauges — always emitted for sharded
        // caches so the scrape payload is O(1) regardless of shard count.
        // The legacy 64-line `lru_hash_load_factor_per_shard{shard="N"}`
        // series is gated behind `set_prometheus_per_shard_detail(true)`
        // and intended for ad-hoc diagnosis only.
        if constexpr (is_striped) {
            const std::size_t num_shards = mm_.num_shards();
            // Collect all per-shard load factors once — used by both the
            // aggregate gauges (always) and the labelled series (detail).
            std::vector<float> shard_lf;
            shard_lf.reserve(num_shards);
            for (std::size_t i = 0; i < num_shards; ++i) {
                shard_lf.push_back(mm_.shard(i).stats().hash_load_factor.load(
                    std::memory_order_relaxed));
            }

            // Aggregate gauges (worst / avg / p95).
            float worst = 0.0f;
            double sum = 0.0;
            for (float lf : shard_lf) {
                if (lf > worst) worst = lf;
                sum += lf;
            }
            const double avg = sum / static_cast<double>(num_shards);
            // P95: nearest-rank method on the sorted (ascending) series.
            // For 64 shards, rank = ceil(0.95 * 64) = 61 -> 61st-smallest.
            // (0-indexed: position 60). For 4 shards, rank = ceil(0.95*4)=4
            // -> largest. This matches Prometheus histogram quantile
            // convention closely enough for alerting purposes.
            std::sort(shard_lf.begin(), shard_lf.end());
            std::size_t p95_idx = static_cast<std::size_t>(
                std::ceil(0.95 * static_cast<double>(num_shards)));
            if (p95_idx == 0) p95_idx = 1;
            if (p95_idx > num_shards) p95_idx = num_shards;
            const float p95 = shard_lf[p95_idx - 1];

            append("# HELP lru_hash_load_factor_worst Worst (max) hash table load factor across shards.");
            append("# TYPE lru_hash_load_factor_worst gauge");
            out.append("lru_hash_load_factor_worst ");
            out.append(std::to_string(worst));
            out.push_back('\n');

            append("# HELP lru_hash_load_factor_avg Average hash table load factor across shards.");
            append("# TYPE lru_hash_load_factor_avg gauge");
            out.append("lru_hash_load_factor_avg ");
            out.append(std::to_string(avg));
            out.push_back('\n');

            append("# HELP lru_hash_load_factor_p95 95th-percentile hash table load factor across shards (nearest-rank).");
            append("# TYPE lru_hash_load_factor_p95 gauge");
            out.append("lru_hash_load_factor_p95 ");
            out.append(std::to_string(p95));
            out.push_back('\n');

            // P2-4: Per-shard labelled series — opt-in via
            // `set_prometheus_per_shard_detail(true)`. Skipped by default
            // to keep the scrape payload compact for caches with 64+
            // shards. The detailed series is most useful when piped
            // through `diagnostics_text()` for ad-hoc shard inspection.
            if (prometheus_per_shard_detail_.load(std::memory_order_acquire)) {
                for (std::size_t i = 0; i < num_shards; ++i) {
                    char label[64];
                    std::snprintf(label, sizeof(label), "{shard=\"%zu\"} ", i);
                    out.append("lru_hash_load_factor_per_shard");
                    out.append(label);
                    out.append(std::to_string(shard_lf[i]));
                    out.push_back('\n');
                }
            }
        }

        append("# HELP lru_hash_overload_threshold Configured load factor threshold for overload events.");
        append("# TYPE lru_hash_overload_threshold gauge");
        out.append("lru_hash_overload_threshold ");
        out.append(std::to_string(snap.hash_overload_threshold.load(std::memory_order_relaxed)));
        out.push_back('\n');

        append("# HELP lru_hash_overload_events_total Total hash table overload events (load factor exceeded threshold).");
        append("# TYPE lru_hash_overload_events_total counter");
        out.append("lru_hash_overload_events_total ");
        out.append(std::to_string(snap.hash_overload_events.load(std::memory_order_relaxed)));
        out.push_back('\n');

        // T11.3: Rehash-blocked-writes counter — non-zero values indicate the
        // user should enable incremental rehash (`set_rehash_strategy("incremental")`).
        append("# HELP lru_rehash_blocked_writes_total Writes blocked by a non-incremental (blocking) rehash.");
        append("# TYPE lru_rehash_blocked_writes_total counter");
        out.append("lru_rehash_blocked_writes_total ");
        out.append(std::to_string(rehash_blocked_writes_count()));
        out.push_back('\n');

        // P1-5: Lock-free read path fallback counter — non-zero values indicate
        // the lock-free read path is being degraded by rehash activity.
        append("# HELP lru_rehash_lockfree_fallback_total Lock-free read paths that fell back to the lock-protected path due to incremental rehash.");
        append("# TYPE lru_rehash_lockfree_fallback_total counter");
        out.append("lru_rehash_lockfree_fallback_total ");
        out.append(std::to_string(rehash_lockfree_fallback_count()));
        out.push_back('\n');

        // P0-D: Rehash-in-progress ratio — gauge of the fraction of the
        // hash table currently rehashing. Sustained values near 1.0 indicate
        // the background rehash balancer cannot keep up with write pressure.
        append("# HELP lru_rehash_in_progress_ratio Fraction of hash table segments currently in an incremental rehash.");
        append("# TYPE lru_rehash_in_progress_ratio gauge");
        out.append("lru_rehash_in_progress_ratio ");
        out.append(std::to_string(rehash_in_progress_ratio()));
        out.push_back('\n');

        append("# HELP lru_cache_tls_ring_flush_total Total TLS ring drain (flush) operations.");
        append("# TYPE lru_cache_tls_ring_flush_total counter");
        out.append("lru_cache_tls_ring_flush_total ");
        out.append(std::to_string(snap.tls_ring_flush_count.load(std::memory_order_relaxed)));
        out.push_back('\n');

        // P0-1: hazptr/EBR reclaim diagnostics — these track deferred
        // reclamation health. High pending with low reclaim growth
        // indicates the reclaim worker is not running or too many
        // read_handles are long-lived.
        append("# HELP lru_reclaim_pending_count Current number of retired objects awaiting reclamation.");
        append("# TYPE lru_reclaim_pending_count gauge");
        out.append("lru_reclaim_pending_count ");
        out.append(std::to_string(snap.reclaim_pending_count.load(std::memory_order_relaxed)));
        out.push_back('\n');

        append("# HELP lru_reclaim_total Cumulative number of retired objects reclaimed.");
        append("# TYPE lru_reclaim_total counter");
        out.append("lru_reclaim_total ");
        out.append(std::to_string(snap.reclaim_total.load(std::memory_order_relaxed)));
        out.push_back('\n');

        append("# HELP lru_reclaim_freed_bytes_total Estimated bytes freed by reclaim operations.");
        append("# TYPE lru_reclaim_freed_bytes_total counter");
        out.append("lru_reclaim_freed_bytes_total ");
        out.append(std::to_string(snap.reclaim_freed_bytes.load(std::memory_order_relaxed)));
        out.push_back('\n');

        append("# HELP lru_reclaim_invocations_total Total times try_reclaim() was invoked.");
        append("# TYPE lru_reclaim_invocations_total counter");
        out.append("lru_reclaim_invocations_total ");
        out.append(std::to_string(snap.reclaim_invocation_count.load(std::memory_order_relaxed)));
        out.push_back('\n');

        // L-1: Hazptr slot-exhaustion / hard-cap fallback diagnostics.
        // Surface the sync-reclaim fallback path so operators can alert
        // before acquire_slot() starts throwing std::runtime_error (which
        // happens after kMaxSyncFallbacks=64 sync-reclaim rounds all fail
        // to free a slot). Non-zero sync_fallback indicates the workload
        // has 8192+ live hazard pointers for extended periods.
        append("# HELP lru_hazptr_slot_exhaustion_total Cumulative count of hazptr acquire_slot() exhaustions (every retry bump, incl. yields).");
        append("# TYPE lru_hazptr_slot_exhaustion_total counter");
        out.append("lru_hazptr_slot_exhaustion_total ");
        out.append(std::to_string(hazptr_slot_exhaustion_count()));
        out.push_back('\n');

        append("# HELP lru_hazptr_sync_fallback_total Cumulative count of hazptr acquire_slot() sync-reclaim fallbacks after spin budget exhaustion. Sustained growth signals imminent hard-cap throw.");
        append("# TYPE lru_hazptr_sync_fallback_total counter");
        out.append("lru_hazptr_sync_fallback_total ");
        out.append(std::to_string(hazptr_sync_fallback_count()));
        out.push_back('\n');

        append("# HELP lru_hazptr_slot_capacity Current hazptr slot capacity (allocated batches x kBatchSize). Grows on demand from 128 to 8192.");
        append("# TYPE lru_hazptr_slot_capacity gauge");
        out.append("lru_hazptr_slot_capacity ");
        out.append(std::to_string(hazptr_slot_capacity()));
        out.push_back('\n');

        // P0-3: live handle count vs runtime-configurable upper bound.
        // Operators alert on usage_ratio > 0.9 — sustained high usage
        // risks acquire_slot() returning npos, degrading reads to cache
        // misses (via empty read_handle). Raise the limit via
        // set_hazptr_max_slots() (default 8192, max 65536).
        append("# HELP lru_hazptr_active_slot_count Currently-acquired (live) hazard pointer slots.");
        append("# TYPE lru_hazptr_active_slot_count gauge");
        out.append("lru_hazptr_active_slot_count ");
        out.append(std::to_string(hazptr_active_slot_count()));
        out.push_back('\n');

        append("# HELP lru_hazptr_max_slot_count Runtime-configurable maximum hazptr slot count (default 8192, raised via set_hazptr_max_slots up to 65536).");
        append("# TYPE lru_hazptr_max_slot_count gauge");
        out.append("lru_hazptr_max_slot_count ");
        out.append(std::to_string(hazptr_max_slot_count()));
        out.push_back('\n');

        append("# HELP lru_hazptr_slot_usage_ratio Current hazptr slot usage ratio (active/max, 0.0-1.0). Alert when > 0.9.");
        append("# TYPE lru_hazptr_slot_usage_ratio gauge");
        out.append("lru_hazptr_slot_usage_ratio ");
        out.append(std::to_string(hazptr_slot_usage_ratio()));
        out.push_back('\n');

        // P1-2: Active fairness mode as a numeric gauge (0=reader_preferred,
        // 1=writer_fair). Operators alert on unexpected switches — e.g.
        // a cache that should be writer_fair (default) but is reported as
        // 0 indicates an unintended set_fairness_mode(reader_preferred)
        // call somewhere in the codebase. The string form is also
        // available via diagnostics_text() for human-readable dumps.
        append("# HELP lru_fairness_mode Active fairness mode (0=reader_preferred, 1=writer_fair). Default is 1 (writer_fair).");
        append("# TYPE lru_fairness_mode gauge");
        out.append("lru_fairness_mode ");
        out.append(std::to_string(static_cast<std::size_t>(get_fairness_mode())));
        out.push_back('\n');

        // P1-3: Drain worker state (1=running, 0=stopped). When 0,
        // retire_obj() emits a one-shot stderr warning on first call.
        // Operators alert on 0 for any cache that should be long-running.
        append("# HELP lru_drain_worker_started Whether the hazptr drain worker has been started (1=yes, 0=no). When 0, retire_obj() emits a one-shot stderr warning.");
        append("# TYPE lru_drain_worker_started gauge");
        out.append("lru_drain_worker_started ");
        out.append(std::to_string(
            static_cast<std::size_t>(is_drain_worker_started() ? 1 : 0)));
        out.push_back('\n');

        // P1-6: Per-worker drain state. The reclaim worker handles
        // hazptr/EBR pending lists (memory safety); the callback worker
        // drains TLS rings and dispatches user callbacks. Both should
        // be 1 in production — alert on 0 for either.
        append("# HELP lru_reclaim_worker_running Whether the reclaim drain worker is running (1=yes, 0=no). When 0, retired objects are not reclaimed — alert on memory growth.");
        append("# TYPE lru_reclaim_worker_running gauge");
        out.append("lru_reclaim_worker_running ");
        out.append(std::to_string(
            static_cast<std::size_t>(is_reclaim_worker_running() ? 1 : 0)));
        out.push_back('\n');
        append("# HELP lru_callback_worker_running Whether the callback drain worker is running (1=yes, 0=no). When 0, TLS rings and async event queues are not drained — alert on stale LRU ordering or delayed callbacks.");
        append("# TYPE lru_callback_worker_running gauge");
        out.append("lru_callback_worker_running ");
        out.append(std::to_string(
            static_cast<std::size_t>(is_callback_worker_running() ? 1 : 0)));
        out.push_back('\n');

        // P1-1: Writer starvation detector metrics. Non-zero
        // writer_starvation_events in reader_preferred mode indicates
        // the safety net fired — operators should consider switching
        // to writer_fair permanently. writer_max_wait_ns tracks the
        // worst observed writer latency for alerting.
        append("# HELP lru_writer_starvation_events_total Cumulative count of readers redirected to writer_fair slow path due to writer starvation (reader_preferred mode only).");
        append("# TYPE lru_writer_starvation_events_total counter");
        out.append("lru_writer_starvation_events_total ");
        out.append(std::to_string(writer_starvation_events()));
        out.push_back('\n');

        append("# HELP lru_writer_max_wait_ns Maximum observed writer wait time in nanoseconds. Reset via reset_writer_max_wait_ns().");
        append("# TYPE lru_writer_max_wait_ns gauge");
        out.append("lru_writer_max_wait_ns ");
        out.append(std::to_string(writer_max_wait_ns()));
        out.push_back('\n');

        // T-M1: singleflight / cache stampede coalescing diagnostics.
        // Non-zero values indicate the stampede protection is actively
        // preventing thundering-herd provider invocations on hot keys.
        append("# HELP lru_stampede_coalesced_total Cumulative count of follower requests collapsed into a leader's in-flight provider call.");
        append("# TYPE lru_stampede_coalesced_total counter");
        out.append("lru_stampede_coalesced_total ");
        out.append(std::to_string(snap.stampede_coalesced_count.load(std::memory_order_relaxed)));
        out.push_back('\n');

        // T-G11: refcount overflow diagnostics. Non-zero values almost
        // certainly indicate a handle leak in user code — under normal
        // operation, access_ref (32 bits, max ~4B) is unreachable.
        append("# HELP lru_incRef_overflow_count_total Cumulative count of incRef() calls that returned kIncFailedOverflow (access_ref saturated at 2^32-1). Non-zero values indicate a read_handle leak.");
        append("# TYPE lru_incRef_overflow_count_total counter");
        out.append("lru_incRef_overflow_count_total ");
        out.append(std::to_string(snap.incRef_overflow_count.value.load(std::memory_order_relaxed)));
        out.push_back('\n');

        // T-B4 (P2-10): Diagnostics cache freshness — operators use this
        // to detect a stalled or missing balancer. Expected value ≈ balancer
        // interval (1s default). Values >> interval indicate the balancer
        // thread died or `start_background_rehash_balancer()` was never
        // called, in which case max_chain_length and per-segment load
        // factor metrics may be stale.
        append("# HELP lru_diagnostics_cache_age_ms Age of the cached diagnostics snapshot in milliseconds. Returns max value if cache never refreshed or underlying table doesn't cache.");
        append("# TYPE lru_diagnostics_cache_age_ms gauge");
        out.append("lru_diagnostics_cache_age_ms ");
        out.append(std::to_string(diagnostics_cache_age_ms()));
        out.push_back('\n');

        // P1-3: Memory watermark / OOM protection status
        append("# HELP lru_cache_memory_critical 1 if cache is in critical memory mode (read-only), 0 otherwise.");
        append("# TYPE lru_cache_memory_critical gauge");
        out.append("lru_cache_memory_critical ");
        out.append(is_memory_critical() ? "1" : "0");
        out.push_back('\n');

        append("# HELP lru_cache_memory_soft_watermark Soft memory watermark (fraction of max_memory).");
        append("# TYPE lru_cache_memory_soft_watermark gauge");
        out.append("lru_cache_memory_soft_watermark ");
        out.append(std::to_string(memory_soft_watermark()));
        out.push_back('\n');

        append("# HELP lru_cache_memory_critical_watermark Critical memory watermark (fraction of max_memory).");
        append("# TYPE lru_cache_memory_critical_watermark gauge");
        out.append("lru_cache_memory_critical_watermark ");
        out.append(std::to_string(memory_critical_watermark()));
        out.push_back('\n');

        // P-HIGH-2 (T-H1): Async OOM handler dispatch mode.
        append("# HELP lru_cache_async_oom_handler 1 if OOM handler is dispatched asynchronously by the drain worker, 0 if synchronous.");
        append("# TYPE lru_cache_async_oom_handler gauge");
        out.append("lru_cache_async_oom_handler ");
        out.append(is_async_oom_handler() ? "1" : "0");
        out.push_back('\n');

        return out;
    }

private:
    /// Helper: emit Prometheus histogram bucket lines + sum/count.
    ///
    /// P2-2: the histogram now uses 512 log-linear buckets (16 sub-buckets
    /// per power-of-2 octave). Bucket upper bounds are computed via
    /// latency_histogram::bucket_upper_bound(i) instead of the old
    /// power-of-2 formula. The overflow bucket (last index) is emitted
    /// with le="+Inf".
    static void emit_histogram_lines(std::string& out,
                                     std::string_view name,
                                     const detail::latency_histogram& hist) {
        auto append_bucket = [&](std::size_t idx, std::string_view le_str) {
            out.append(name);
            out.append("_bucket{le=\"");
            out.append(le_str);
            out.append("\"} ");
            out.append(std::to_string(hist.bucket(idx)));
            out.push_back('\n');
        };

        // buckets 0..bucket_count-2: le = bucket_upper_bound(i)
        // (exclusive upper bound; Prometheus histograms are cumulative
        // and use le="<upper bound>" semantics).
        char buf[32];
        constexpr std::size_t last = detail::latency_histogram::bucket_count - 1;
        for (std::size_t i = 0; i < last; ++i) {
            std::uint64_t le = detail::latency_histogram::bucket_upper_bound(i);
            // Skip buckets with zero width (cannot happen with the log-linear
            // layout, but guard anyway for safety).
            if (le == 0) continue;
            std::snprintf(buf, sizeof(buf), "%llu",
                          static_cast<unsigned long long>(le));
            append_bucket(i, buf);
        }
        // overflow bucket: le="+Inf"
        append_bucket(last, "+Inf");

        // Use the true accumulated sum (atomic uint64, fetch_add on every
        // record()). This replaces the previous (min+max)/2 * count
        // approximation which distorted SLO reporting on skewed latencies.
        std::uint64_t total = hist.count();
        std::uint64_t true_sum = hist.sum();
        out.append(name);
        out.append("_sum ");
        std::snprintf(buf, sizeof(buf), "%llu", static_cast<unsigned long long>(true_sum));
        out.append(buf);
        out.push_back('\n');

        out.append(name);
        out.append("_count ");
        out.append(std::to_string(total));
        out.push_back('\n');
    }

public:

    /// Register a hit callback. The callback is invoked outside the cache lock.
    /// For sharded caches, the callback is propagated to all shards.
    void on_hit(hit_callback_type callback) {
        auto lock = acquire_write_lock();
        mm_.callbacks().on_hit(callback);
        if constexpr (requires { mm_.shard(0); }) {
            for (std::size_t i = 0; i < mm_.num_shards(); ++i) {
                mm_.shard(i).callbacks().on_hit(callback);
            }
        }
    }

    /// Register a miss callback. The callback is invoked outside the cache lock.
    /// For sharded caches, the callback is propagated to all shards.
    void on_miss(miss_callback_type callback) {
        auto lock = acquire_write_lock();
        mm_.callbacks().on_miss(callback);
        if constexpr (requires { mm_.shard(0); }) {
            for (std::size_t i = 0; i < mm_.num_shards(); ++i) {
                mm_.shard(i).callbacks().on_miss(callback);
            }
        }
    }

    /// Register an insert callback. The callback is invoked outside the cache lock.
    /// For sharded caches, the callback is propagated to all shards.
    void on_insert(insert_callback_type callback) {
        auto lock = acquire_write_lock();
        mm_.callbacks().on_insert(callback);
        if constexpr (requires { mm_.shard(0); }) {
            for (std::size_t i = 0; i < mm_.num_shards(); ++i) {
                mm_.shard(i).callbacks().on_insert(callback);
            }
        }
    }

    /// Register an eviction callback. The callback is invoked outside the cache lock.
    /// For sharded caches, the callback is propagated to all shards.
    void on_evict(evict_callback_type callback) {
        auto lock = acquire_write_lock();
        mm_.callbacks().on_evict(callback);
        if constexpr (requires { mm_.shard(0); }) {
            for (std::size_t i = 0; i < mm_.num_shards(); ++i) {
                mm_.shard(i).callbacks().on_evict(callback);
            }
        }
    }

    /// O7: Register an update callback — fires when set() overwrites an
    /// existing key's value (distinct from on_insert which fires for
    /// fresh inserts). For sharded caches, the callback is propagated
    /// to all shards.
    void on_update(update_callback_type callback) {
        auto lock = acquire_write_lock();
        mm_.callbacks().on_update(callback);
        if constexpr (requires { mm_.shard(0); }) {
            for (std::size_t i = 0; i < mm_.num_shards(); ++i) {
                mm_.shard(i).callbacks().on_update(callback);
            }
        }
    }

    /// O7: Register an expire callback — fires when an item is evicted
    /// due to TTL expiry (distinct from on_evict which fires for
    /// capacity eviction). For sharded caches, propagated to all shards.
    void on_expire(expire_callback_type callback) {
        auto lock = acquire_write_lock();
        mm_.callbacks().on_expire(callback);
        if constexpr (requires { mm_.shard(0); }) {
            for (std::size_t i = 0; i < mm_.num_shards(); ++i) {
                mm_.shard(i).callbacks().on_expire(callback);
            }
        }
    }

    /// O7: Register a reject callback — fires when an insert is rejected
    /// by the overflow policy (cache full, OOM, admission denial).
    /// For sharded caches, propagated to all shards.
    void on_reject(reject_callback_type callback) {
        auto lock = acquire_write_lock();
        mm_.callbacks().on_reject(callback);
        if constexpr (requires { mm_.shard(0); }) {
            for (std::size_t i = 0; i < mm_.num_shards(); ++i) {
                mm_.shard(i).callbacks().on_reject(callback);
            }
        }
    }

    /// O6: Register an error hook invoked when any registered callback
    /// throws. The hook receives the captured exception, the event kind,
    /// the key, and (for non-miss events) a pointer to the value. Pass
    /// an empty `error_callback_type` to clear. For sharded caches, the
    /// hook is propagated to all shards.
    ///
    /// The hook is invoked on the dispatching thread (async worker when
    /// async mode is active, otherwise the caller of flush_pending).
    /// Exceptions thrown by the hook itself are swallowed.
    void on_callback_error(error_callback_type callback) {
        auto lock = acquire_write_lock();
        mm_.callbacks().on_callback_error(callback);
        if constexpr (requires { mm_.shard(0); }) {
            for (std::size_t i = 0; i < mm_.num_shards(); ++i) {
                mm_.shard(i).callbacks().on_callback_error(callback);
            }
        }
    }

    /// O6: Total callback failures observed across all shards since
    /// construction or the last reset. Aggregates sync + async dispatch
    /// failures. For sharded caches, sums counts across all shards.
    std::size_t callback_error_count() const {
        std::size_t total = mm_.callbacks().callback_error_count();
        if constexpr (requires { mm_.shard(0); }) {
            for (std::size_t i = 0; i < mm_.num_shards(); ++i) {
                total += mm_.shard(i).callbacks().callback_error_count();
            }
        }
        return total;
    }

    /// O6: Reset the callback failure counter across all shards.
    void reset_callback_error_count() {
        mm_.callbacks().reset_callback_error_count();
        if constexpr (requires { mm_.shard(0); }) {
            for (std::size_t i = 0; i < mm_.num_shards(); ++i) {
                mm_.shard(i).callbacks().reset_callback_error_count();
            }
        }
    }

    /// Remove all registered callbacks and pending deferred events.
    /// For sharded caches, clears callbacks from all shards.
    void clear_callbacks() {
        auto lock = acquire_write_lock();
        mm_.callbacks().clear_all();
        if constexpr (requires { mm_.shard(0); }) {
            for (std::size_t i = 0; i < mm_.num_shards(); ++i) {
                mm_.shard(i).callbacks().clear_all();
            }
        }
    }

    /// Direct access to the callback manager (primarily for registration).
    /// NOTE: the returned reference is valid for the lifetime of the cache.
    /// Concurrent mutation should be done through the on_* helpers above.
    callback_mgr& callbacks() { return mm_.callbacks(); }
    const callback_mgr& callbacks() const { return mm_.callbacks(); }

    /// Direct access (for advanced use, with external locking)
    mm_type& mm() { return mm_; }
    const mm_type& mm() const { return mm_; }

    /// Promote an item by key without triggering hit statistics or callbacks.
    /// This performs LRU position promotion only, making it suitable for
    /// batch promotion from TLS ring flush where side effects (hit counts,
    /// hit callbacks) are undesirable.
    /// Returns true if the key was found and the item was eligible for promotion.
    bool promote(const Key& key) {
        if constexpr (is_striped) {
            auto lock = acquire_write_lock_for_key(key);
            bool result = false;
            if constexpr (requires { mm_.promote(key); }) {
                result = mm_.promote(key);
            }
            flush_shard_pending(key);
            return result;
        } else {
            flush_guard fg{mm_};
            auto lock = acquire_write_lock_for_key(key);
            if constexpr (requires { mm_.promote(key); }) {
                return mm_.promote(key);
            } else {
                return false;
            }
        }
    }

    // --------------------------------------------------------------------
    // Memory policy
    // --------------------------------------------------------------------

    void set_key_size_calculator(std::function<size_type(const Key&)> func) {
        auto lock = acquire_write_lock();
        mm_.set_key_size_calculator(std::move(func));
    }

    void set_value_size_calculator(std::function<size_type(const Value&)> func) {
        auto lock = acquire_write_lock();
        mm_.set_value_size_calculator(std::move(func));
    }

    void set_eviction_predicate(std::function<bool(const Key&, const Value&)> pred) {
        auto lock = acquire_write_lock();
        mm_.set_eviction_predicate(std::move(pred));
    }

    // --------------------------------------------------------------------
    // Memory monitor (admission control)
    // --------------------------------------------------------------------

    /// Configure the memory monitor for admission control. When active
    /// (max_memory_bytes > 0 or max_growth_rate_bytes > 0), new insertions
    /// via set()/add() are admitted only if the monitor allows them. Updates
    /// to existing keys are always allowed and trigger eviction if needed.
    void set_memory_monitor(const memory_monitor::config& cfg) {
        auto lock = acquire_write_lock();
        memory_monitor_.configure(cfg);
    }

    /// Direct access to the memory monitor (for stats and dynamic tuning).
    /// The caller must not mutate the monitor in a way that races with cache
    /// operations; prefer set_memory_monitor() for configuration changes.
    lru::memory_monitor& monitor() { return memory_monitor_; }
    const lru::memory_monitor& monitor() const { return memory_monitor_; }

    /// Get a snapshot of memory monitor statistics.
    ///
    /// P1-13: No lock — memory_monitor_.get_stats() reads its own atomic
    /// counters, and mm_.current_memory() is a sum of per-shard atomic
    /// counters. The global read lock would shared-lock all 64 stripes
    /// for no correctness benefit (no shard state is read here).
    memory_monitor::stats memory_monitor_stats() const {
        auto s = memory_monitor_.get_stats();
        // Ensure the snapshot reflects the cache's current memory, even if the
        // monitor has not been reported to recently.
        s.current_memory_bytes = mm_.current_memory();
        if (s.max_memory_bytes > 0) {
            s.occupancy_fraction = static_cast<double>(s.current_memory_bytes) /
                                   static_cast<double>(s.max_memory_bytes);
        }
        s.current_state = memory_monitor_.current_state();
        return s;
    }

    // --------------------------------------------------------------------
    // Slab allocator & rebalancer
    // --------------------------------------------------------------------

    /// Set custom hash table node allocation/deallocation functions.
    /// When non-null, hash table node_type objects in non-EmbeddedChain mode
    /// are allocated via alloc_fn instead of ::operator new, enabling slab
    /// allocator integration for the hash table's internal nodes.
    ///
    /// For F14 cache aliases (f14_cache, f14_safe_cache, etc.), this allows
    /// the hash table's overflow chain nodes to be allocated from a slab
    /// allocator instead of the default heap, reducing allocation overhead
    /// and memory fragmentation in high-throughput scenarios.
    ///
    /// Note: Only effective when the hash table is in non-EmbeddedChain mode
    /// (EmbeddedChain = false). In the default EmbeddedChain mode, the hash
    /// table does not allocate separate node_type objects, so these functions
    /// are ignored.
    void set_hash_alloc_fns(void* (*alloc_fn)(std::size_t), void (*dealloc_fn)(void*)) {
        auto lock = acquire_write_lock();
        mm_.set_hash_alloc_fns(alloc_fn, dealloc_fn);
    }

    /// Enable the slab allocator for this cache, creating one with the given
    /// configuration. When enabled, the MM layer will use the slab allocator
    /// for item allocation instead of ::operator new.
    ///
    /// Note: slab rebalancing (`try_move_slab`) has been removed — it was
    /// unsafe under concurrent `set()` because the eviction_check callback
    /// dereferenced item pointers that could be freed and reallocated by
    /// another thread. The slab allocator itself remains thread-safe
    /// (lock-free Treiber stack per size class) and is suitable for
    /// read-heavy workloads where allocation/free patterns are stable.
    void enable_slab_allocator(const slab_allocator::config& cfg = slab_allocator::config{}) {
        auto lock = acquire_write_lock();
        if (owned_allocator_) return;  // already enabled
        owned_allocator_ = std::make_unique<slab_allocator>(cfg);
        mm_.set_allocator(owned_allocator_.get());
    }

    /// Check whether the slab allocator is enabled.
    bool has_slab_allocator() const noexcept {
        return owned_allocator_ != nullptr;
    }

    /// Get a reference to the owned slab allocator.
    /// Throws cache_config_exception if the slab allocator is not enabled.
    slab_allocator& slab_alloc() {
        if (!owned_allocator_)
            throw cache_config_exception("slab allocator not enabled; call enable_slab_allocator() first");
        return *owned_allocator_;
    }

    const slab_allocator& slab_alloc() const {
        if (!owned_allocator_)
            throw cache_config_exception("slab allocator not enabled; call enable_slab_allocator() first");
        return *owned_allocator_;
    }

    // --------------------------------------------------------------------
    // Event drain workers (P1-6: split into reclaim + callback workers)
    // --------------------------------------------------------------------
    //
    // P1-6: The drain worker was previously a single thread that did both
    //   (a) memory reclamation (hazptr/EBR pending drain, epoch advance,
    //       try_reclaim_now) and
    //   (b) user-visible side-effect dispatch (TLS access/callback ring
    //       flush, async overload/OOM event drain, event tracker drain).
    //
    // This coupled memory safety to user-callback latency: a slow user
    // callback (e.g. logging to a remote service) blocked reclamation,
    // which under high churn could cause unbounded memory growth. The
    // split into two workers decouples these concerns:
    //
    //   * reclaim_drain_worker_  — memory-safety critical. Runs at a
    //                               short, fixed interval (default 250ms,
    //                               100ms when read handles are alive)
    //                               and only does hazptr/EBR reclamation
    //                               plus epoch advancement. User callbacks
    //                               never run on this thread, so its tick
    //                               latency stays bounded (<1ms typical).
    //
    //   * callback_drain_worker_ — user-visible. Runs at the caller-
    //                               supplied interval (default 1s, 500ms
    //                               when handles are alive) and drains
    //                               TLS access/callback rings, overload
    //                               and OOM event queues, and the event
    //                               tracker. Slow user callbacks may
    //                               stretch this worker's tick latency
    //                               without affecting reclamation.
    //
    // Both workers self-adjust their interval based on active_handle_count:
    // when read handles are alive (active_handle_count > 0), items being
    // released by ~read_handle() cannot be retired until the reclaim
    // worker flushes the pending list; the shorter interval bounds the
    // worst-case delay between handle release and reclamation.

    /// Start the background drain workers. Two workers are launched:
    ///   1. reclaim_drain_worker_   — frequent, memory-safety critical
    ///   2. callback_drain_worker_  — normal cadence, user-visible
    ///
    /// @param interval  Callback worker drain interval (default: 1 second).
    ///                  The reclaim worker uses min(interval, 250ms) so
    ///                  memory reclamation always runs at least every
    ///                  250ms. Shorter intervals reduce staleness but
    ///                  increase CPU wake-ups. Use 100ms for latency-
    ///                  sensitive scenarios.
    void start_event_drain(std::chrono::milliseconds interval = std::chrono::seconds(1)) {
        if (reclaim_drain_worker_ || callback_drain_worker_) return;  // Already running
        // P1-3: Mark the hazptr domain's drain worker as started so
        // retire_obj() stops emitting the one-shot stderr warning.
        // Without this, operators using hazptr-mode caches (any cache
        // that did not call set_ebr_domain()) would see the warning on
        // the first eviction, even though the drain worker is about to
        // start. Set the flag BEFORE creating the worker so the window
        // between the first retire and the worker startup is closed.
        detail::hazptr_domain::default_domain().set_drain_started(true);

        const std::chrono::milliseconds callback_normal = interval;
        const std::chrono::milliseconds callback_aggressive =
            std::min<std::chrono::milliseconds>(callback_normal,
                std::chrono::milliseconds(500));
        // Reclaim worker uses a shorter baseline — memory safety must
        // not wait on user-callback cadence. Capped at 250ms so even
        // long caller-supplied intervals (e.g. 5s) still reclaim within
        // 250ms; aggressive mode (handles alive) drops to 100ms.
        const std::chrono::milliseconds reclaim_normal =
            std::min<std::chrono::milliseconds>(callback_normal,
                std::chrono::milliseconds(250));
        const std::chrono::milliseconds reclaim_aggressive =
            std::min<std::chrono::milliseconds>(reclaim_normal,
                std::chrono::milliseconds(100));

        // ----- Reclaim worker (memory safety) -----
        reclaim_drain_worker_ = std::make_unique<detail::periodic_worker>(
            [this, reclaim_normal, reclaim_aggressive] {
                // C-2 fix: also call maybe_time_advance() on the EBR
                // domain(s) to force epoch advancement in read-heavy-write-
                // light workloads where the TLS retire buffer rarely
                // fills. Without this, retired objects can sit in the
                // EBR pending list indefinitely because no thread is
                // entering/leaving critical sections fast enough to
                // advance the epoch naturally.
                for_each_ebr_domain([](detail::epoch_domain& dom) {
                    dom.maybe_time_advance();
                });
                // T-P2-3 (R-7): incremental reclaim — process at most
                // kIncrementalReclaimBatch objects per domain per tick
                // to bound worst-case tick latency. If the backlog is
                // large (pending > 4 * batch), process 4x the default
                // batch this tick to accelerate drain.
                const std::size_t pending =
                    detail::hazptr_domain::default_domain().pending_count()
                    + ebr_pending_count();
                std::size_t batch = kIncrementalReclaimBatch;
                if (pending > kIncrementalReclaimBatch * 4) {
                    batch = kIncrementalReclaimBatch * 4;
                }
                try_reclaim_now(batch);
                // P1-7 (T2.6): Self-adjust interval based on active
                // handle count. When handles are alive, drop to the
                // aggressive interval so items released by ~read_handle
                // are reclaimed promptly; otherwise use the normal
                // interval to save CPU.
                if (reclaim_drain_worker_) {
                    const std::size_t live_handles = active_handle_count();
                    const std::chrono::milliseconds next_interval =
                        (live_handles > 0) ? reclaim_aggressive : reclaim_normal;
                    reclaim_drain_worker_->set_interval(next_interval);
                }
            },
            reclaim_normal);

        // ----- Callback worker (user-visible dispatch) -----
        callback_drain_worker_ = std::make_unique<detail::periodic_worker>(
            [this, callback_normal, callback_aggressive] {
                drain_access_ring();
                tls_callback_ring<Key, Value>::flush_all_registered();
                // P1-4 (T3.3): Force-flush dormant threads' TLS rings.
                // Threads that have not called record_access() in the
                // last 2 seconds (R6: lowered from 5s) are considered
                // dormant (e.g., blocked on I/O, descheduled by the OS,
                // or simply idle). Their pending TLS ring entries are
                // at risk of being lost because they cannot service a
                // needs_flush_ request. This call safely reads their
                // rings cross-thread using seqlock-style validation and
                // pushes the drained keys into the global backup buffer,
                // which drain_access_ring() will pick up on the next
                // iteration.
                tls_access_ring<Key>::force_flush_dormant_threads();
                // P2-4 (T2.4): Drain async overload-callback queue.
                // When `set_async_overload_callback(true)` is enabled,
                // the rehash hot path enqueues overload events instead
                // of invoking the user callback inline. Draining them
                // here keeps the callback off the rehash path while
                // still dispatching it promptly (≤1s, 500ms aggressive).
                drain_overload_callbacks();
                // P-HIGH-2 (T-H1): Drain async OOM handler event queue.
                // Safe no-op when async mode is disabled or no event
                // pending.
                drain_oom_events();
                // P1-12: drain the event tracker's TLS ring (if enabled)
                // AFTER tls_callback_ring flush, because callback dispatch
                // may record events into the tracker's TLS ring. Using
                // drain_all_threads() ensures all threads' events are
                // captured, not just the drain worker's.
                std::shared_ptr<event_tracker<Key, Hash>> tracker;
                {
                    std::lock_guard<std::mutex> lk(event_tracker_mutex_);
                    tracker = event_tracker_;
                }
                if (tracker) {
                    tracker->drain_all_threads();
                }
                // P1-7 (T2.6): Self-adjust interval based on active
                // handle count. When handles are alive, drop to the
                // aggressive interval so deferred promotions are
                // applied sooner (reducing stale LRU ordering window);
                // otherwise use the normal interval to save CPU.
                if (callback_drain_worker_) {
                    const std::size_t live_handles = active_handle_count();
                    const std::chrono::milliseconds next_interval =
                        (live_handles > 0) ? callback_aggressive : callback_normal;
                    callback_drain_worker_->set_interval(next_interval);
                }
            },
            callback_normal);
    }

    /// Stop both background drain workers (reclaim + callback) if running.
    /// P1-3: Also clears the hazptr domain's drain_started flag so that
    /// subsequent retire_obj() calls (e.g. from another cache sharing
    /// the default domain) emit the warning again — but the warning's
    /// CAS flag is NOT reset, so the message fires at most once per
    /// process lifetime (operators who saw it once will not see it
    /// again even after stop/start cycles).
    ///
    /// Stop order: callback worker first, then reclaim worker. The
    /// callback worker may trigger evictions during its final tick
    /// (TLS access ring drain causes LRU promotions); those evicted
    /// items are retired into the hazptr/EBR pending list, which the
    /// reclaim worker must still be alive to drain. Stopping in this
    /// order bounds the residual pending list to one reclaim tick's
    /// worth (≤250ms).
    ///
    /// P1-8: Only reset the global drain_started flag when THIS cache
    /// actually started a drain worker. The flag is process-global
    /// (default hazptr domain), but ordinary single-threaded caches
    /// (e.g. block_cache_layer's mm_lru) never call start_event_drain()
    /// — they only call stop_event_drain() during shutdown. Unconditionally
    /// clearing the flag here would wipe out the drain state of another
    /// cache whose worker is still running, causing spurious
    /// "retire_obj() called before start_event_drain()" warnings.
    void stop_event_drain() {
        const bool had_drain_worker =
            reclaim_drain_worker_ != nullptr || callback_drain_worker_ != nullptr;
        if (callback_drain_worker_) {
            callback_drain_worker_->stop();
            callback_drain_worker_.reset();
        }
        if (reclaim_drain_worker_) {
            reclaim_drain_worker_->stop();
            reclaim_drain_worker_.reset();
        }
        // P1-3: clear drain_started only if this cache started a worker;
        // otherwise a fresh start_event_drain() call from another cache
        // re-arms the state correctly. The drain_warn_emitted_ CAS flag
        // in hazptr_domain is NOT reset, so the warning never fires twice
        // per process.
        if (had_drain_worker) {
            detail::hazptr_domain::default_domain().set_drain_started(false);
        }
    }

    /// Check if either background drain worker (reclaim or callback) is
    /// running. Returns true if at least one is active.
    bool is_event_drain_running() const noexcept {
        return reclaim_drain_worker_ != nullptr || callback_drain_worker_ != nullptr;
    }

    /// P1-6: Query whether the reclaim worker specifically is running.
    /// The reclaim worker drains hazptr/EBR pending lists and advances
    /// the EBR epoch. If this returns false but is_event_drain_running()
    /// returns true, only the callback worker is active and memory
    /// reclamation is not happening — operators should call
    /// start_event_drain() to start both workers.
    bool is_reclaim_worker_running() const noexcept {
        return reclaim_drain_worker_ != nullptr;
    }

    /// P1-6: Query whether the callback worker specifically is running.
    /// The callback worker drains TLS access/callback rings and
    /// dispatches async overload/OOM events. If this returns false but
    /// is_event_drain_running() returns true, only the reclaim worker
    /// is active and deferred promotions/callbacks are not being
    /// processed — operators should call start_event_drain() to start
    /// both workers.
    bool is_callback_worker_running() const noexcept {
        return callback_drain_worker_ != nullptr;
    }

    /// P1-3: Query whether the hazptr drain worker has been started.
    /// Mirrors `hazptr_domain::is_drain_started()`. Returns true after
    /// `start_event_drain()` and false after `stop_event_drain()`. When
    /// false, the first `retire_obj()` call emits a one-shot stderr
    /// warning so operators notice the missing worker before memory
    /// grows unboundedly.
    bool is_drain_worker_started() const noexcept {
        return detail::hazptr_domain::default_domain().is_drain_started();
    }

    // --------------------------------------------------------------------
    // P0-5 (T1.3): Background rehash balancer
    // --------------------------------------------------------------------
    //
    // The hot-path rehash_if_needed(hash) only touches the segment owning
    // `hash`. Segments that aren't currently being written to can fall
    // behind — their load factor grows as inserts land elsewhere, but
    // no insert triggers their rehash. The background rehash balancer
    // periodically sweeps all segments and advances any in-progress
    // incremental rehashes, plus invokes rehash_if_needed() on segments
    // whose load factor has drifted above the threshold.
    //
    // Recommended cadence: 500ms–2s. Faster = less rehash backlog but
    // more wake-ups; slower = lower CPU overhead but bigger backlog.
    // The worker uses try_reclaim-friendly semantics (no locks held
    // across the sweep).

    /// Start a background worker that periodically sweeps all segments
    /// and advances rehash state. Idempotent: if the worker is already
    /// running, the new interval is applied to the live worker (via
    /// `periodic_worker::set_interval`), so callers can tighten or loosen
    /// the cadence at runtime.
    /// The worker is automatically stopped on cache destruction.
    /// @param interval  Sweep interval (default: 1 second).
    void start_background_rehash_balancer(
            std::chrono::milliseconds interval = std::chrono::seconds(1)) {
        if (rehash_balancer_worker_) {
            // Already running: restart the worker with the new interval
            // so the new cadence takes effect immediately. `set_interval`
            // alone is insufficient because `periodic_worker::wait_for`
            // doesn't reset its deadline on notify — the worker would
            // continue waiting for the remainder of the original interval
            // before picking up the new one. Stopping + restarting ensures
            // the next sweep happens within `interval` from now, which
            // tests like F14DiagnosticsTest.DiagnosticsCacheServesMaxChainLength
            // rely on (they call start(100ms) after the auto-started 1s
            // default and expect a sweep within 300ms).
            rehash_balancer_worker_->stop();
            rehash_balancer_worker_.reset();
        }
        rehash_balancer_worker_ = std::make_unique<detail::periodic_worker>(
            [this] {
                // T-B4 (P2-10): Refresh the diagnostics cache BEFORE
                // refresh_hash_stats() reads max_chain_length(). Without
                // this ordering, refresh_hash_stats() would call
                // max_chain_length() on a cold cache, triggering a live
                // O(buckets) scan and defeating the whole point of the
                // cache. By refreshing first, the subsequent read in
                // refresh_hash_stats() hits the fast atomic-load path.
                // For non-segmented tables refresh_diagnostics_cache()
                // is a no-op via `if constexpr`, so this is zero-cost
                // for caches that don't use the segmented hash table.
                mm_.refresh_diagnostics_cache();
                // Sweep all segments and trigger rehash_if_needed().
                // For segmented tables this is the catch-all path; for
                // non-segmented tables it's equivalent to a single
                // rehash_if_needed() call.
                mm_.refresh_hash_stats();
                // Also nudge any stalled incremental rehash to make
                // progress (rehash_finish respects the per-call budget
                // and returns early if more work remains — repeated
                // nudges eventually complete the migration).
                mm_.advance_incremental_rehash();
            },
            interval);
    }

    /// Stop the background rehash balancer if running.
    void stop_background_rehash_balancer() {
        if (rehash_balancer_worker_) {
            rehash_balancer_worker_->stop();
            rehash_balancer_worker_.reset();
        }
    }

    /// Check if the background rehash balancer is running.
    bool is_background_rehash_balancer_running() const noexcept {
        return rehash_balancer_worker_ != nullptr;
    }

    /// T-B4 (P2-10): Age of the diagnostics snapshot in milliseconds.
    ///
    /// Returns the worst-case age across all shards. Returns
    /// `std::numeric_limits<std::uint64_t>::max()` if the cache has
    /// never been refreshed or the underlying hash table doesn't cache
    /// (non-segmented tables — operators should check the
    /// `segmented_hash_table` flag in `diagnostics()` first).
    ///
    /// Operators should expect this value to be roughly equal to the
    /// balancer interval (1s default) under normal operation. Values
    /// significantly larger than the interval indicate either:
    ///   - The balancer was never started (call
    ///     `start_background_rehash_balancer()` at cache setup).
    ///   - The balancer thread died or is starved (check
    ///     `is_background_rehash_balancer_running()`).
    ///   - The cache was just constructed and the balancer hasn't run
    ///     its first sweep yet (age will drop to ~interval after the
    ///     first sweep).
    ///
    /// Exposed via Prometheus as `lru_diagnostics_cache_age_ms`.
    std::uint64_t diagnostics_cache_age_ms() const noexcept {
        return mm_.diagnostics_cache_age_ms();
    }

    // --------------------------------------------------------------------
    // P1-12: Event tracker integration
    // --------------------------------------------------------------------
    //
    // event_tracker records item lifecycle events (insert/hit/evict) into
    // a lock-free TLS ring buffer, drained periodically by the cache's
    // event_drain_worker. Enable tracking during cache setup, then query
    // the tracker (top_keys, generate_report, etc.) at runtime.
    //
    // The tracker is stored as std::atomic<std::shared_ptr<...>> so the
    // drain worker can read it race-free. Callbacks capture a weak_ptr to
    // avoid preventing tracker destruction if the user replaces it.

    /// Enable event tracking with the given configuration.
    /// Creates an event_tracker, registers on_hit/on_insert/on_evict
    /// callbacks that record events, and stores the tracker.
    ///
    /// Should be called during cache setup, before concurrent access.
    /// Calling it again replaces the previous tracker (old callbacks
    /// remain registered — call clear_callbacks() first if you want a
    /// clean slate).
    ///
    /// @param cfg  Event tracker configuration (capacity, sampling, etc.)
    /// @return     Shared pointer to the created tracker. The cache
    ///             retains a copy; the user may retain one for querying.
    std::shared_ptr<event_tracker<Key, Hash>> enable_event_tracking(
        typename event_tracker<Key, Hash>::config cfg = {}) {
        auto tracker = std::make_shared<event_tracker<Key, Hash>>(cfg);
        // Capture a weak_ptr to avoid extending the tracker's lifetime
        // beyond the user's reference. If the user releases their
        // reference and the cache is destroyed, the tracker is freed;
        // the callbacks' weak_ptr::lock() returns null and skips.
        std::weak_ptr<event_tracker<Key, Hash>> wp = tracker;
        on_hit([wp](const Key& k, const Value&) {
            if (auto t = wp.lock()) t->record_hit(k);
        });
        on_insert([wp](const Key& k, const Value&) {
            if (auto t = wp.lock()) t->record_insert(k);
        });
        on_evict([wp](const Key& k, const Value&) {
            if (auto t = wp.lock()) t->record_evict(k);
        });
        std::lock_guard<std::mutex> lock(event_tracker_mutex_);
        event_tracker_ = tracker;
        return tracker;
    }

    /// Disable event tracking by clearing the stored tracker reference.
    /// Also calls `tracker->disable()` to stop the drain worker and clear
    /// the tracker's event callback (P2-D), ensuring a clean shutdown even
    /// if the user retains a shared_ptr to the tracker.
    /// Note: previously registered on_hit/on_insert/on_evict callbacks remain
    /// registered (they become no-ops once the tracker's shared_ptr count
    /// drops to zero). Use clear_callbacks() to remove all callbacks.
    void disable_event_tracking() noexcept {
        std::shared_ptr<event_tracker<Key, Hash>> tracker;
        {
            std::lock_guard<std::mutex> lock(event_tracker_mutex_);
            tracker = event_tracker_;
            event_tracker_.reset();
        }
        if (tracker) {
            // P2-D: stop the drain worker, clear the event callback, and
            // gate all record_* methods to no-ops. Safe to call outside
            // the lock — disable() is thread-safe. Wrapped in try/catch
            // to preserve the noexcept guarantee.
            try {
                tracker->disable();
            } catch (...) {
                // Suppress — best-effort cleanup during shutdown.
            }
        }
    }

    /// Access the event tracker (if enabled), or nullptr.
    /// The returned shared_ptr keeps the tracker alive for the duration
    /// of use, even if disable_event_tracking() is called concurrently.
    ///
    /// Named get_event_tracker() (not event_tracker()) to avoid shadowing
    /// the event_tracker<Key, Hash> type name within this class template.
    std::shared_ptr<event_tracker<Key, Hash>> get_event_tracker() const noexcept {
        std::lock_guard<std::mutex> lock(event_tracker_mutex_);
        return event_tracker_;
    }

    /// Check if event tracking is enabled.
    bool event_tracking_enabled() const noexcept {
        std::lock_guard<std::mutex> lock(event_tracker_mutex_);
        return event_tracker_ != nullptr;
    }

    // --------------------------------------------------------------------
    // P0-1: hazptr / EBR reclamation control
    // --------------------------------------------------------------------
    //
    // Hazard pointer and epoch-based reclamation defer the actual deletion
    // of retired cache items until no reader can observe them. Without
    // periodic invocation of try_reclaim(), retired items accumulate in
    // the global pending list, causing unbounded memory growth under
    // high-churn workloads. The background event_drain_worker invokes
    // try_reclaim() on each tick; users may also trigger it manually.

    /// Trigger an immediate reclamation pass on both the hazptr and EBR
    /// default domains. Returns the total number of objects reclaimed.
    /// Safe to call concurrently with cache operations — try_reclaim()
    /// is lock-free.
    ///
    /// @param batch_size T-P2-3 (R-7): Maximum objects to process per
    ///        domain per call. 0 = drain all (original behavior, used
    ///        by explicit user-initiated full reclaim). When > 0, at
    ///        most `batch_size` objects are inspected per domain; any
    ///        unprocessed or still-protected objects remain pending for
    ///        the next call. The background `event_drain_worker` uses
    ///        `try_reclaim_now(kIncrementalReclaimBatch)` to bound the
    ///        worst-case latency of each tick — eliminating the spikes
    ///        seen when a large pending list (e.g., after a burst of
    ///        evictions) was fully traversed in a single pass. Users
    ///        may inspect `reclaim_pending_count` via `stats_snapshot()`
    ///        and call `try_reclaim_now(0)` for a full drain if needed.
    std::size_t try_reclaim_now(std::size_t batch_size = 0) noexcept {
        std::size_t reclaimed = 0;
        try {
            reclaimed += detail::hazptr_domain::default_domain().try_reclaim(batch_size);
            // R9: reclaim every active EBR domain (global default, or each
            // per-shard domain when per-shard EBR is enabled).
            for_each_ebr_domain([&](detail::epoch_domain& dom) {
                reclaimed += dom.try_reclaim(batch_size);
            });
        } catch (...) {
            // Reclaim must be noexcept — swallow to avoid breaking callers.
        }
        // Update per-cache stats so users can observe reclaim progress.
        // pending_count snapshots (best-effort, may change concurrently).
        per_cache_stats_.reclaim_total.fetch_add(reclaimed, std::memory_order_relaxed);
        per_cache_stats_.reclaim_invocation_count.fetch_add(1, std::memory_order_relaxed);
        // P1-11: Estimate freed bytes as reclaimed * sizeof(item_type).
        // The hazptr/EBR domain tracks only object counts, not per-object
        // sizes, so we approximate using the cache's item_type. This is
        // accurate for the common case (all retired objects are
        // cache_items); it overestimates slightly when wrapper objects
        // are also retired, but the comment in cache_stats already
        // declares this field as "estimated bytes freed (approx)".
        // Without this increment the metric was always 0, making it
        // useless for monitoring reclamation effectiveness.
        constexpr std::size_t kEstimatedItemBytes = sizeof(typename mm_type::item_type);
        per_cache_stats_.reclaim_freed_bytes.fetch_add(
            reclaimed * kEstimatedItemBytes, std::memory_order_relaxed);
        std::size_t pending = detail::hazptr_domain::default_domain().pending_count()
                            + ebr_pending_count();
        per_cache_stats_.reclaim_pending_count.store(pending, std::memory_order_relaxed);
        return reclaimed;
    }

    /// T-P2-3 (R-7): Default batch size for incremental reclaim used by
    /// the background `event_drain_worker`. Each tick processes at most
    /// this many objects per reclamation domain (hazptr + EBR), bounding
    /// tick latency. With 1024 objects per tick and a 500ms–1s tick
    /// interval, steady-state reclaim throughput is 1024–2048 objects/s
    /// per domain — sufficient to keep up with typical eviction rates
    /// while ensuring no single tick blocks for a full-list scan.
    /// Spikes from bursty evictions are absorbed across multiple ticks
    /// instead of one long pass.
    static constexpr std::size_t kIncrementalReclaimBatch = 1024;

    // --------------------------------------------------------------------
    // P1-3 (T1.4): Auto-reclaim threshold
    // --------------------------------------------------------------------
    //
    // Under bursty retire workloads, retired items accumulate in the
    // hazptr/EBR pending list faster than the background worker can
    // drain them. Setting an auto-reclaim threshold causes the retire
    // path itself to synchronously trigger try_reclaim() when the
    // pending count exceeds the threshold — bounding worst-case memory
    // pressure without adding lock contention to the steady-state hot
    // path. A CAS flag prevents stampede (at most one thread reclaims
    // at a time).
    //
    // Default threshold is 65536. Set to
    // std::numeric_limits<std::size_t>::max() to disable auto-reclaim
    // (only background / explicit reclaims will run).
    //
    // The threshold applies to BOTH the hazptr default domain and the
    // EBR default domain. Per-domain thresholds can be set directly on
    // detail::hazptr_domain::default_domain() / epoch_domain::default_domain()
    // if asymmetric sizing is required.

    /// Set the auto-reclaim threshold on both default domains.
    void set_reclaim_threshold(std::size_t threshold) noexcept {
        detail::hazptr_domain::default_domain().set_reclaim_threshold(threshold);
        // R9: propagate to every active EBR domain (global or per-shard).
        for_each_ebr_domain([threshold](detail::epoch_domain& dom) {
            dom.set_reclaim_threshold(threshold);
        });
    }

    /// Return the lower of the hazptr / EBR reclaim thresholds (i.e.
    /// the effective bound on pending backlog before auto-reclaim fires).
    std::size_t reclaim_threshold() const noexcept {
        std::size_t h = detail::hazptr_domain::default_domain().reclaim_threshold();
        // R9: per-shard domains are configured uniformly; sample the min.
        std::size_t e = std::numeric_limits<std::size_t>::max();
        for_each_ebr_domain([&](detail::epoch_domain& dom) {
            e = std::min(e, dom.reclaim_threshold());
        });
        if (e == std::numeric_limits<std::size_t>::max()) e = 0;
        return std::min(h, e);
    }

    /// Return the cumulative number of auto-reclaim triggers across both
    /// domains since process start. Useful for sizing the threshold
    /// against actual retire pressure: a steadily-increasing count
    /// indicates the threshold is too low for the workload's retire rate.
    std::size_t reclaim_auto_triggered_count() const noexcept {
        std::size_t total = detail::hazptr_domain::default_domain().reclaim_auto_triggered_count();
        // R9: aggregate across all active EBR domains.
        for_each_ebr_domain([&](detail::epoch_domain& dom) {
            total += dom.reclaim_auto_triggered_count();
        });
        return total;
    }

    // --------------------------------------------------------------------
    // P0-2 (T2.1): EBR epoch force-advance timeout
    // --------------------------------------------------------------------
    //
    // EBR uses a global epoch counter; reclamation is safe when retired
    // objects' epoch < min(active slot epochs). If a thread is stuck in
    // a long critical section (descheduled by the OS, deadlocked, or in
    // a long-running operation), the global epoch can't advance, and
    // reclamation is blocked indefinitely. Setting a force-advance
    // timeout causes compute_min_epoch() to treat slots older than the
    // timeout as "stuck" and skip them, allowing reclamation of older
    // objects despite the stuck thread.
    //
    // The trade-off: if a stuck thread resumes after the reclaim and
    // dereferences a retired pointer, UAF may result. The default 30s
    // timeout is well beyond normal CS duration; only lower it when
    // the alternative (unbounded reclaim backlog) is worse.

    /// Set the EBR force-advance timeout on the default domain.
    void set_epoch_force_advance_timeout(std::chrono::nanoseconds timeout) noexcept {
        // R9: propagate to every active EBR domain (global or per-shard).
        for_each_ebr_domain([timeout](detail::epoch_domain& dom) {
            dom.set_epoch_force_advance_timeout(timeout);
        });
    }

    /// P0-2: Alias for `set_epoch_force_advance_timeout` — shorter name
    /// aligned with the `force_advance_policy` API.
    void set_force_advance_timeout(std::chrono::nanoseconds timeout) noexcept {
        set_epoch_force_advance_timeout(timeout);
    }

    /// P0-2: Select the EBR force-advance policy on the default domain.
    /// See `lru::detail::force_advance_policy` for the semantics of each
    /// value. Default is `kForceAdvanceAfter5s` (preserves pre-P0-2
    /// behavior). Use `kFailAdvance` for memory-safety-sensitive
    /// scenarios where UAF risk is unacceptable; use `kNeverForceAdvance`
    /// only when the workload guarantees no long-running critical sections.
    void set_force_advance_policy(detail::force_advance_policy policy) noexcept {
        // R9: propagate to every active EBR domain (global or per-shard).
        for_each_ebr_domain([policy](detail::epoch_domain& dom) {
            dom.set_force_advance_policy(policy);
        });
    }

    /// P0-2: Query the active EBR force-advance policy.
    detail::force_advance_policy get_force_advance_policy() const noexcept {
        // R9: domains are configured uniformly; read the first active one.
        detail::force_advance_policy result =
            detail::force_advance_policy::kFailAdvance;
        for_each_ebr_domain([&](detail::epoch_domain& dom) {
            result = dom.get_force_advance_policy();
        });
        return result;
    }

    /// Query the current EBR force-advance timeout.
    std::chrono::nanoseconds epoch_force_advance_timeout() const noexcept {
        std::chrono::nanoseconds result{0};
        for_each_ebr_domain([&](detail::epoch_domain& dom) {
            result = dom.epoch_force_advance_timeout();
        });
        return result;
    }

    /// Return the number of times compute_min_epoch() skipped a stuck
    /// slot. A steadily-increasing count indicates either the timeout
    /// is too low, or the workload has genuine long-running operations
    /// that should be split into smaller CSes.
    std::size_t force_advance_count() const noexcept {
        return ebr_force_advance_count();
    }

    // --------------------------------------------------------------------
    // Task 7: TTL background cleaner
    // --------------------------------------------------------------------
    //
    // Spawns a background thread that periodically calls evict_expired()
    // on the underlying MM. For MM types without native TTL support
    // (mm_lru, mm_2q, mm_fifo, etc.), evict_expired() is a no-op
    // returning 0 — the cleaner still runs but does nothing, allowing
    // user code to call start_ttl_cleaner() unconditionally without
    // SFINAE gymnastics at the call site.
    //
    // For sharded MM types (sharded_mm_lru), the cleaner iterates all
    // shards under per-shard locks, evicting expired items from each.
    // The eviction count is aggregated and added to cache_stats::evictions.

    /// Start the TTL background cleaner thread.
    /// @param interval  Polling interval. Default 1 second. Lower values
    ///                  reduce stale-item memory at the cost of CPU.
    ///                  100ms is reasonable for tight TTL budgets;
    ///                  30s is fine for lax budgets.
    /// @throws std::logic_error if the cleaner is already running.
    void start_ttl_cleaner(std::chrono::milliseconds interval = std::chrono::seconds(1)) {
        if (ttl_cleaner_worker_) {
            throw std::logic_error("start_ttl_cleaner: cleaner already running");
        }
        // T-G17: Background cleaner uses round-robin (one shard per cycle)
        // to amortize CPU across cycles. Callers wanting immediate full
        // eviction should use evict_expired_now() instead.
        ttl_cleaner_worker_ = std::make_unique<detail::periodic_worker>(
            [this] { evict_expired_impl(/*round_robin=*/true); },
            interval);
    }

    /// Stop the TTL background cleaner thread if running.
    /// Blocks until the worker thread has joined.
    void stop_ttl_cleaner() {
        if (ttl_cleaner_worker_) {
            ttl_cleaner_worker_->stop();
            ttl_cleaner_worker_.reset();
        }
    }

    /// Check if the TTL cleaner is currently running.
    bool is_ttl_cleaner_running() const noexcept {
        return ttl_cleaner_worker_ != nullptr;
    }

    // ----------------------------------------------------------------
    // T-G17: True per-shard round-robin TTL cleanup
    // ----------------------------------------------------------------
    //
    // The legacy `evict_expired_impl()` walks ALL shards on every cleaner
    // cycle. For caches with 64 shards × 15K items each, that is ~1M
    // iterator comparisons per cycle — easily >10% CPU under tight
    // polling intervals (100ms).
    //
    // T-G17 introduces a round-robin counter so each cleaner cycle
    // processes exactly ONE shard (the next one in rotation). This cuts
    // per-cycle work by 1/num_shards (≈1/64 for production_cache),
    // bringing cleaner CPU under 1% for 1M TTL entries.
    //
    // The trade-off is that a freshly-expired item may wait up to
    // `interval × num_shards` before being reaped (e.g. 64s with 1s
    // interval). For typical TTL budgets (minutes to hours) this lag
    // is negligible; for sub-second TTLs callers should reduce the
    // interval proportionally (e.g. 10ms × 64 = 640ms worst-case).
    //
    // Per-shard expiration min-heap (sub-task 1 of T-G17) is NOT
    // implemented — it would require invasive changes to mm.hpp to
    // maintain a heap per shard, and the round-robin optimization
    // alone meets the <1% CPU target on 1M entries. The heap can be
    // added later as a drop-in optimization if sub-millisecond TTL
    // reaping becomes a requirement.

    /// T-G17: Is per-shard round-robin TTL cleanup enabled? Default
    /// true for sharded caches (cuts cleaner CPU by ~1/num_shards).
    /// Set to false to restore legacy "scan all shards per cycle"
    /// behavior.
    bool is_ttl_cleaner_round_robin_enabled() const noexcept {
        return ttl_cleaner_round_robin_.load(std::memory_order_relaxed);
    }

    /// T-G17: Enable/disable per-shard round-robin TTL cleanup.
    void set_ttl_cleaner_round_robin(bool enabled) noexcept {
        ttl_cleaner_round_robin_.store(enabled, std::memory_order_relaxed);
    }

    /// T-G17: Return the shard index that the next cleaner cycle will
    /// process. Exposed for testing and diagnostics.
    std::size_t ttl_cleaner_next_shard() const noexcept {
        return ttl_cleaner_next_shard_.load(std::memory_order_relaxed);
    }

    /// Synchronously evict expired items once and return the count.
    /// This is the same operation performed by the background cleaner,
    /// but invoked manually (e.g., from a custom scheduler).
    ///
    /// T-G17: Unlike the background cleaner (which uses round-robin to
    /// process ONE shard per cycle), `evict_expired_now()` processes ALL
    /// shards in a single call — callers explicitly invoke this when they
    /// want immediate full eviction, not the deferred round-robin behavior.
    std::size_t evict_expired_now() {
        refresh_cached_now();
        return evict_expired_impl(/*round_robin=*/false);
    }

    /// Refresh the internal cached time used for TTL checks.
    /// This forces the next TTL check to use the current real time,
    /// which is useful when the cache has been idle for a while and
    /// the cached time may be stale.
    void refresh_cached_now() {
        if constexpr (is_striped) {
            for (std::size_t i = 0; i < mm_.num_shards(); ++i) {
                mm_.shard(i).refresh_cached_now();
            }
        } else {
            if constexpr (requires { mm_.refresh_cached_now(); }) {
                mm_.refresh_cached_now();
            }
        }
    }

private:
    /// T-G1: Value-layer TTL scanner for caches whose value_type is
    /// `ttl_entry<U>`. Walks the MM under a read lock to collect expired
    /// keys, then deletes each key under its per-shard write lock (or
    /// global write lock for non-sharded caches). This makes
    /// `start_ttl_cleaner()` effective for `ttl_cache<U>` users without
    /// requiring them to wire up an external periodic worker.
    ///
    /// Design notes:
    ///   - Read-lock walk + per-key write-lock delete mirrors the pattern
    ///     in `ttl_cache::clear_expired_locked()`, but lives in
    ///     unified_cache so it works for direct
    ///     `unified_cache<..., ttl_entry<U>>` usage too.
    ///   - For sharded caches, the read lock is acquired per-shard via
    ///     `try_acquire_shard_*_lock` to avoid blocking writers across
    ///     the entire cache. A shard that is contended is skipped this
    ///     cycle and retried next cycle.
    ///   - Batch size is honored (config.ttl_evict_batch_size) to bound
    ///     the per-cycle work; remaining expired items are picked up in
    ///     subsequent cleaner cycles.
    std::size_t evict_expired_via_ttl_entry_scan(bool round_robin = false) {
        using entry_t = value_type;
        using clock_t = typename entry_t::clock;
        const auto now = clock_t::now();

        // Determine batch size (0 = unlimited per cycle).
        const std::size_t batch_size = [this]() -> std::size_t {
            if constexpr (requires { mm_.config().ttl_evict_batch_size; }) {
                auto bs = mm_.config().ttl_evict_batch_size;
                if (bs > 0) return bs;
                if constexpr (requires { mm_.config().lru_config.ttl_evict_batch_size; }) {
                    return mm_.config().lru_config.ttl_evict_batch_size;
                }
            }
            if constexpr (requires { mm_.config().lru_config.ttl_evict_batch_size; }) {
                return mm_.config().lru_config.ttl_evict_batch_size;
            }
            return 0;
        }();

        std::size_t total = 0;

        if constexpr (is_striped) {
            // T-G17: True per-shard round-robin. When enabled (default for
            // background cleaner), each cycle processes exactly ONE shard
            // (the next in rotation). This cuts per-cycle work by
            // ~1/num_shards for production_cache (64 shards). When disabled,
            // falls back to legacy "scan all shards per cycle" behavior.
            const bool use_round_robin = round_robin
                && ttl_cleaner_round_robin_.load(std::memory_order_relaxed);
            const std::size_t num_shards = mm_.num_shards();
            const std::size_t start_shard = use_round_robin
                ? (ttl_cleaner_next_shard_.fetch_add(1, std::memory_order_relaxed) % num_shards)
                : 0;
            const std::size_t shard_limit = use_round_robin
                ? (start_shard + 1)  // process exactly one shard
                : num_shards;        // process all shards

            for (std::size_t shard_idx = start_shard; shard_idx < shard_limit; ++shard_idx) {
                const std::size_t i = shard_idx % num_shards;
                if (total > 0 && batch_size > 0 && total >= batch_size) break;

                // Phase 1: collect expired keys under a read lock (try-lock
                // to avoid blocking writers).
                std::vector<key_type> expired_keys;
                {
                    if constexpr (has_per_shard_lock_v<mm_type>) {
                        auto lock = mm_.try_acquire_shard_read_lock(i);
                        if (!lock.owns_lock()) continue;
                        for (auto it = mm_.shard(i).begin();
                             it != mm_.shard(i).end(); ++it) {
                            if (it->value.is_expired_at(now)) {
                                expired_keys.push_back(it->key);
                                if (batch_size > 0 && (total + expired_keys.size()) >= batch_size) break;
                            }
                        }
                    } else {
                        auto stripe = i % striped_mutex_.size();
                        auto lock = striped_mutex_.try_make_shared_lock(stripe);
                        if (!lock.owns_lock()) continue;
                        for (auto it = mm_.shard(i).begin();
                             it != mm_.shard(i).end(); ++it) {
                            if (it->value.is_expired_at(now)) {
                                expired_keys.push_back(it->key);
                                if (batch_size > 0 && (total + expired_keys.size()) >= batch_size) break;
                            }
                        }
                    }
                }
                // Phase 2: delete each expired key under a write lock.
                for (const auto& k : expired_keys) {
                    if constexpr (has_per_shard_lock_v<mm_type>) {
                        auto wlock = mm_.try_acquire_shard_write_lock(i);
                        if (!wlock.owns_lock()) continue;
                        if (mm_.shard(i).del(k)) ++total;
                    } else {
                        auto stripe = i % striped_mutex_.size();
                        auto wlock = striped_mutex_.try_make_unique_lock(stripe);
                        if (!wlock.owns_lock()) continue;
                        if (mm_.shard(i).del(k)) ++total;
                    }
                    if (batch_size > 0 && total >= batch_size) break;
                }
            }
        } else {
            // Non-sharded: single read-lock walk + single write-lock delete.
            std::vector<key_type> expired_keys;
            {
                auto lock = acquire_read_lock();
                for (auto it = mm_.begin(); it != mm_.end(); ++it) {
                    const auto& entry = it->value;
                    if (entry.is_expired_at(now)) {
                        expired_keys.push_back(it->key);
                        if (batch_size > 0 && expired_keys.size() >= batch_size) break;
                    }
                }
            }
            auto wlock = acquire_write_lock();
            for (const auto& k : expired_keys) {
                if (mm_.del(k)) ++total;
                if (batch_size > 0 && total >= batch_size) break;
            }
        }
        return total;
    }

    /// SFINAE-aware evict_expired implementation. Dispatches to the MM's
    /// evict_expired() if it exists; otherwise returns 0 (no-op).
    ///
    /// T-G1: When value_type is `ttl_entry<U>` (i.e. the cache was
    /// constructed as `unified_cache<..., ttl_entry<U>>` or via
    /// `ttl_cache<U>`), TTL information lives in the value layer
    /// (`ttl_entry::expiry`), NOT in the MM's native TTL heap. The MM's
    /// `evict_expired()` returns 0 because its ttl_heap_ is never
    /// populated. To fix this, when `is_ttl_entry_v<value_type>` is true
    /// we dispatch to `evict_expired_via_ttl_entry_scan()` which walks
    /// the MM under a read lock, collects expired keys, and deletes them
    /// per-shard under write locks — the same pattern as
    /// `ttl_cache::clear_expired()`, but integrated into unified_cache so
    /// `start_ttl_cleaner()` works out of the box for ttl_cache users.
    ///
    /// When ttl_evict_batch_size > 0 (configured via mm config), the
    /// implementation uses try_lock to avoid blocking concurrent readers
    /// and limits eviction to `batch_size` items per shard/lock acquisition.
    /// Remaining expired items are processed in subsequent cleaner cycles.
    std::size_t evict_expired_impl(bool round_robin = false) {
        // T-G1: value-layer TTL scan for ttl_entry-wrapped caches.
        if constexpr (is_ttl_entry_v<value_type>) {
            return evict_expired_via_ttl_entry_scan(round_robin);
        }

        if constexpr (is_striped) {
            std::size_t total = 0;
            if constexpr (requires { mm_.shard(0).evict_expired(); }) {
                // Determine batch size from sharded config or per-shard config.
                const std::size_t batch_size = [this]() -> std::size_t {
                    if constexpr (requires { mm_.config().ttl_evict_batch_size; }) {
                        auto bs = mm_.config().ttl_evict_batch_size;
                        return bs > 0 ? bs : mm_.config().lru_config.ttl_evict_batch_size;
                    }
                    return 0;
                }();

                // T-G17: True per-shard round-robin — process exactly ONE
                // shard per cleaner cycle when enabled (default for
                // background cleaner). This cuts per-cycle work by
                // ~1/num_shards. `evict_expired_now()` passes
                // round_robin=false to process ALL shards in one call.
                const bool use_round_robin = round_robin
                    && ttl_cleaner_round_robin_.load(std::memory_order_relaxed);
                const std::size_t num_shards = mm_.num_shards();
                const std::size_t start_shard = use_round_robin
                    ? (ttl_cleaner_next_shard_.fetch_add(1, std::memory_order_relaxed) % num_shards)
                    : 0;
                const std::size_t shard_limit = use_round_robin
                    ? (start_shard + 1)
                    : num_shards;

                for (std::size_t shard_idx = start_shard; shard_idx < shard_limit; ++shard_idx) {
                    const std::size_t i = shard_idx % num_shards;
                    if (batch_size > 0) {
                        // Non-blocking: try to acquire the write lock; skip
                        // this shard if the lock is held by a reader/writer.
                        // T3.4 bugfix (P1-7): for sharded_mm_lru, acquire
                        // the per-shard write lock (not the stripe lock) —
                        // the stripe lock does not protect the MM data
                        // structures and would race with concurrent set().
                        if constexpr (has_per_shard_lock_v<mm_type>) {
                            auto lock = mm_.try_acquire_shard_write_lock(i);
                            if (!lock.owns_lock()) continue;
                            total += mm_.shard(i).evict_expired_n(batch_size);
                        } else {
                            auto stripe = i % striped_mutex_.size();
                            auto lock = striped_mutex_.try_make_unique_lock(stripe);
                            if (!lock.owns_lock()) continue;
                            // Evict at most batch_size expired items from this shard.
                            total += mm_.shard(i).evict_expired_n(batch_size);
                        }
                    } else {
                        // Legacy behavior: blocking lock, evict all expired.
                        // T3.4 bugfix (P1-7): per-shard write lock for sharded_mm_lru.
                        if constexpr (has_per_shard_lock_v<mm_type>) {
                            auto lock = mm_.acquire_shard_write_lock(i);
                            total += mm_.shard(i).evict_expired();
                        } else {
                            auto lock = striped_mutex_.make_unique_lock(i % striped_mutex_.size());
                            total += mm_.shard(i).evict_expired();
                        }
                    }
                }
            } else if constexpr (requires { mm_.evict_expired(); }) {
                // Non-sharded fallback (should not normally be reached for
                // is_striped caches, but handled for completeness).
                if constexpr (requires { mm_.config().lru_config.ttl_evict_batch_size; }) {
                    const std::size_t batch_size = mm_.config().lru_config.ttl_evict_batch_size;
                    if (batch_size > 0) {
                        // P1-1 fix: single_threaded_policy has mutex_type=void,
                        // so std::unique_lock<void> would not compile. Use the
                        // policy's acquire_write_lock() helper, which returns a
                        // no-op lock for single-threaded caches.
                        auto lock = acquire_write_lock();
                        if constexpr (requires { mm_.evict_expired_n(batch_size); }) {
                            total += mm_.evict_expired_n(batch_size);
                        } else {
                            total += mm_.evict_expired();
                        }
                    } else {
                        auto lock = acquire_write_lock();
                        total += mm_.evict_expired();
                    }
                } else {
                    auto lock = acquire_write_lock();
                    total += mm_.evict_expired();
                }
            }
            return total;
        } else {
            if constexpr (requires { mm_.evict_expired(); }) {
                // Determine batch size from the mm config.
                const std::size_t batch_size = [this]() -> std::size_t {
                    if constexpr (requires { mm_.config().ttl_evict_batch_size; }) {
                        return mm_.config().ttl_evict_batch_size;
                    }
                    return 0;
                }();

                if (batch_size > 0) {
                    // P1-1 fix: single_threaded_policy has mutex_type=void,
                    // so std::unique_lock<void> would not compile. Use the
                    // policy's acquire_write_lock() helper, which returns a
                    // no-op lock for single-threaded caches. The try_lock
                    // optimization is meaningless here anyway (only one shard).
                    auto lock = acquire_write_lock();
                    if constexpr (requires { mm_.evict_expired_n(batch_size); }) {
                        return mm_.evict_expired_n(batch_size);
                    } else {
                        // Fallback: evict all but only hold lock briefly.
                        return mm_.evict_expired();
                    }
                } else {
                    auto lock = acquire_write_lock();
                    return mm_.evict_expired();
                }
            } else {
                return 0;
            }
        }
    }

public:

    // --------------------------------------------------------------------
    // Iterators (reverse: LRU → MRU)
    // --------------------------------------------------------------------

    /// Returns a locked range over the MM reverse iterator pair.
    /// The global read lock is held until the returned locked_range is destroyed.
    auto rbegin() {
        // P-HIGH-1 (T-C3): For sharded_mm_lru, acquire per-shard read locks
        // (the stripe lock does NOT protect MM data — see T3.4 bugfix).
        auto lock = acquire_read_lock_all_shards();
        if constexpr (requires { std::declval<mm_type&>().rbegin(); }) {
            return locked_range(mm_.rbegin(), mm_.rend(), std::move(lock));
        } else {
            return locked_range(mm_.begin(), mm_.end(), std::move(lock));
        }
    }

    /// Returns an empty locked range marking the end of the reverse range.
    auto rend() {
        auto lock = acquire_read_lock_all_shards();
        if constexpr (requires { std::declval<mm_type&>().rend(); }) {
            auto end = mm_.rend();
            return locked_range(end, end, std::move(lock));
        } else {
            auto end = mm_.end();
            return locked_range(end, end, std::move(lock));
        }
    }

    /// Const locked range over the MM reverse iterator pair.
    auto rbegin() const {
        auto lock = acquire_read_lock_all_shards();
        if constexpr (requires { std::declval<const mm_type&>().rbegin(); }) {
            return locked_range(mm_.rbegin(), mm_.rend(), std::move(lock));
        } else {
            return locked_range(mm_.begin(), mm_.end(), std::move(lock));
        }
    }

    /// Const empty locked range marking the end of the reverse range.
    auto rend() const {
        auto lock = acquire_read_lock_all_shards();
        if constexpr (requires { std::declval<const mm_type&>().rend(); }) {
            auto end = mm_.rend();
            return locked_range(end, end, std::move(lock));
        } else {
            auto end = mm_.end();
            return locked_range(end, end, std::move(lock));
        }
    }

    // --------------------------------------------------------------------
    // Per-shard reverse iterators (striped cache only)
    // --------------------------------------------------------------------

    /// Returns a locked range over a single shard's reverse iterator pair.
    /// Only the stripe for the given shard is locked, allowing concurrent
    /// access to other shards during iteration.
    ///
    /// @param shard_id The shard index (0 to num_shards()-1)
    /// @throws std::out_of_range if shard_id is invalid
    auto shard_rbegin(std::size_t shard_id) {
        // P-HIGH-1 (T-C3): For sharded_mm_lru, acquire the per-shard read
        // lock (NOT the stripe lock — stripe lock does not protect MM data).
        if constexpr (has_per_shard_lock_v<mm_type>) {
            if (shard_id >= mm_.num_shards()) {
                throw std::out_of_range("shard_rbegin: shard_id out of range");
            }
            auto lock = mm_.acquire_shard_read_lock(shard_id);
            auto& shard = mm_.shard(shard_id);
            if constexpr (requires { shard.rbegin(); }) {
                return locked_range(shard.rbegin(), shard.rend(), std::move(lock));
            } else {
                return locked_range(shard.begin(), shard.end(), std::move(lock));
            }
        } else if constexpr (is_striped) {
            if (shard_id >= mm_.num_shards()) {
                throw std::out_of_range("shard_rbegin: shard_id out of range");
            }
            auto lock = striped_mutex_.make_shared_lock(shard_id);
            auto& shard = mm_.shard(shard_id);
            if constexpr (requires { shard.rbegin(); }) {
                return locked_range(shard.rbegin(), shard.rend(), std::move(lock));
            } else {
                return locked_range(shard.begin(), shard.end(), std::move(lock));
            }
        } else {
            // Non-striped: fall back to global rbegin
            auto lock = acquire_read_lock();
            if constexpr (requires { mm_.rbegin(); }) {
                return locked_range(mm_.rbegin(), mm_.rend(), std::move(lock));
            } else {
                return locked_range(mm_.begin(), mm_.end(), std::move(lock));
            }
        }
    }

    /// Const overload of shard_rbegin.
    auto shard_rbegin(std::size_t shard_id) const {
        if constexpr (has_per_shard_lock_v<mm_type>) {
            if (shard_id >= mm_.num_shards()) {
                throw std::out_of_range("shard_rbegin: shard_id out of range");
            }
            auto lock = mm_.acquire_shard_read_lock(shard_id);
            const auto& shard = mm_.shard(shard_id);
            if constexpr (requires { shard.rbegin(); }) {
                return locked_range(shard.rbegin(), shard.rend(), std::move(lock));
            } else {
                return locked_range(shard.begin(), shard.end(), std::move(lock));
            }
        } else if constexpr (is_striped) {
            if (shard_id >= mm_.num_shards()) {
                throw std::out_of_range("shard_rbegin: shard_id out of range");
            }
            auto lock = striped_mutex_.make_shared_lock(shard_id);
            const auto& shard = mm_.shard(shard_id);
            if constexpr (requires { shard.rbegin(); }) {
                return locked_range(shard.rbegin(), shard.rend(), std::move(lock));
            } else {
                return locked_range(shard.begin(), shard.end(), std::move(lock));
            }
        } else {
            auto lock = acquire_read_lock();
            if constexpr (requires { mm_.rbegin(); }) {
                return locked_range(mm_.rbegin(), mm_.rend(), std::move(lock));
            } else {
                return locked_range(mm_.begin(), mm_.end(), std::move(lock));
            }
        }
    }

    /// Returns an empty locked range marking the end of a shard's reverse range.
    /// Only the stripe for the given shard is locked.
    ///
    /// @param shard_id The shard index (0 to num_shards()-1)
    /// @throws std::out_of_range if shard_id is invalid
    auto shard_rend(std::size_t shard_id) {
        if constexpr (is_striped) {
            if (shard_id >= mm_.num_shards()) {
                throw std::out_of_range("shard_rend: shard_id out of range");
            }
            auto lock = striped_mutex_.make_shared_lock(shard_id);
            auto& shard = mm_.shard(shard_id);
            if constexpr (requires { shard.rend(); }) {
                auto end = shard.rend();
                return locked_range(end, end, std::move(lock));
            } else {
                auto end = shard.end();
                return locked_range(end, end, std::move(lock));
            }
        } else {
            auto lock = acquire_read_lock();
            if constexpr (requires { mm_.rend(); }) {
                auto end = mm_.rend();
                return locked_range(end, end, std::move(lock));
            } else {
                auto end = mm_.end();
                return locked_range(end, end, std::move(lock));
            }
        }
    }

    /// Const overload of shard_rend.
    auto shard_rend(std::size_t shard_id) const {
        if constexpr (is_striped) {
            if (shard_id >= mm_.num_shards()) {
                throw std::out_of_range("shard_rend: shard_id out of range");
            }
            auto lock = striped_mutex_.make_shared_lock(shard_id);
            const auto& shard = mm_.shard(shard_id);
            if constexpr (requires { shard.rend(); }) {
                auto end = shard.rend();
                return locked_range(end, end, std::move(lock));
            } else {
                auto end = shard.end();
                return locked_range(end, end, std::move(lock));
            }
        } else {
            auto lock = acquire_read_lock();
            if constexpr (requires { mm_.rend(); }) {
                auto end = mm_.rend();
                return locked_range(end, end, std::move(lock));
            } else {
                auto end = mm_.end();
                return locked_range(end, end, std::move(lock));
            }
        }
    }

    // --------------------------------------------------------------------
    // Key snapshot (lock-free iteration after brief read lock)
    // --------------------------------------------------------------------

    /// Take a snapshot of all keys in the cache.
    /// Briefly acquires the read lock to collect keys, then releases it.
    /// Safe for iteration without blocking writes for the duration.
    std::vector<Key> snapshot_keys() const {
        auto lock = acquire_read_lock();
        std::vector<Key> keys;
        keys.reserve(mm_.size());
        if constexpr (requires { mm_.shard(0); }) {
            // Sharded MM: iterate each shard individually
            for (std::size_t i = 0; i < mm_.num_shards(); ++i) {
                for (auto it = mm_.shard(i).begin(); it != mm_.shard(i).end(); ++it) {
                    keys.push_back(it->key);
                }
            }
        } else {
            for (auto it = mm_.begin(); it != mm_.end(); ++it) {
                keys.push_back(it->key);
            }
        }
        return keys;
    }

    // --------------------------------------------------------------------
    // Lock management
    // --------------------------------------------------------------------

    /// RAII guard that exclusively locks all stripes of a striped_mutex.
    /// Used for global write operations (flush, resize, etc.).
    struct striped_write_lock_all {
        striped_mutex_storage& sm;
        explicit striped_write_lock_all(striped_mutex_storage& sm) : sm(sm) { sm.lock_all(); }
        ~striped_write_lock_all() { sm.unlock_all(); }
        striped_write_lock_all(const striped_write_lock_all&) = delete;
        striped_write_lock_all& operator=(const striped_write_lock_all&) = delete;
    };

    /// RAII guard that shared-locks all stripes of a striped_mutex.
    /// Used for global read operations (size, stats, etc.).
    struct striped_read_lock_all {
        striped_mutex_storage& sm;
        explicit striped_read_lock_all(striped_mutex_storage& sm) : sm(sm) { sm.lock_shared_all(); }
        ~striped_read_lock_all() { sm.unlock_shared_all(); }
        striped_read_lock_all(const striped_read_lock_all&) = delete;
        striped_read_lock_all& operator=(const striped_read_lock_all&) = delete;
    };

    /// Acquire a global write lock. For striped policies, locks all stripes.
    /// For non-striped thread-safe policies, acquires the global mutex.
    /// For single-threaded policies returns a noop_lock.
    auto acquire_write_lock() const {
        if constexpr (is_striped) {
            return striped_write_lock_all(striped_mutex_);
        } else if constexpr (is_thread_safe) {
            return typename lock_policy::write_lock_type(mutex_);
        } else {
            return noop_lock{};
        }
    }

    /// Acquire a global read lock. For striped policies, shared-locks all stripes.
    /// For non-striped thread-safe policies, acquires the global shared mutex.
    /// For single-threaded policies returns a noop_lock.
    auto acquire_read_lock() const {
        if constexpr (is_striped) {
            return striped_read_lock_all(striped_mutex_);
        } else if constexpr (is_thread_safe) {
            return typename lock_policy::read_lock_type(mutex_);
        } else {
            return noop_lock{};
        }
    }

    /// Acquire a write lock for operations targeting a specific key.
    /// For striped policies, locks only the stripe that the key hashes to,
    /// enabling concurrent access to different stripes.
    /// For non-striped thread-safe policies, acquires the global write lock.
    /// For single-threaded policies, returns a noop_lock.
    auto acquire_write_lock_for_key(const Key& key) const {
        // P2-3 (T3.4): Prefer per-shard lock when the MM provides one
        // (e.g. sharded_mm_lru). This decouples num_shards from
        // num_stripes and removes the historical constraint that
        // shard_idx == stripe_idx.
        if constexpr (has_per_shard_lock_v<mm_type>) {
            auto hash = Hash{}(key);
            auto shard = mm_.shard_for_hash(hash);
            return mm_.acquire_shard_write_lock(shard);
        } else if constexpr (is_striped) {
            auto hash = Hash{}(key);
            auto stripe = striped_mutex_.stripe_for(hash);
            return striped_mutex_.make_unique_lock(stripe);
        } else if constexpr (is_thread_safe) {
            return typename lock_policy::write_lock_type(mutex_);
        } else {
            return noop_lock{};
        }
    }

    /// Acquire a read lock for operations targeting a specific key.
    /// For striped policies, shared-locks only the stripe that the key hashes to.
    /// For non-striped thread-safe policies, acquires the global read lock.
    /// For single-threaded policies, returns a noop_lock.
    auto acquire_read_lock_for_key(const Key& key) const {
        if constexpr (has_per_shard_lock_v<mm_type>) {
            auto hash = Hash{}(key);
            auto shard = mm_.shard_for_hash(hash);
            return mm_.acquire_shard_read_lock(shard);
        } else if constexpr (is_striped) {
            auto hash = Hash{}(key);
            auto stripe = striped_mutex_.stripe_for(hash);
            return striped_mutex_.make_shared_lock(stripe);
        } else if constexpr (is_thread_safe) {
            return typename lock_policy::read_lock_type(mutex_);
        } else {
            return noop_lock{};
        }
    }

    /// P2-1 / P-CRIT-2 (T-C2): Acquire a read lock for a specific shard index.
    /// For sharded_mm_lru, uses the per-shard mutex (the stripe lock does
    /// NOT protect MM data — see T3.4 bugfix). For non-sharded MM with
    /// striped locking, uses the stripe lock keyed by shard_idx (modulo
    /// stripe count, since num_shards may differ from num_stripes — T3.4
    /// decoupled them). Used by bulk_get and save_per_shard().
    auto acquire_read_lock_for_shard(std::size_t shard_idx) const {
        if constexpr (has_per_shard_lock_v<mm_type>) {
            return mm_.acquire_shard_read_lock(shard_idx);
        } else if constexpr (is_striped) {
            return striped_mutex_.make_shared_lock(shard_idx % striped_mutex_.size());
        } else if constexpr (is_thread_safe) {
            return typename lock_policy::read_lock_type(mutex_);
        } else {
            return noop_lock{};
        }
    }

    /// P-CRIT-2 (T-C2): Acquire a write lock for a specific shard index.
    /// Used by load_per_shard() to rebuild a single shard under its
    /// per-shard write lock (the stripe lock does NOT protect MM data
    /// for sharded_mm_lru — see T3.4 bugfix comment in evict_expired_impl).
    auto acquire_write_lock_for_shard(std::size_t shard_idx) const {
        if constexpr (has_per_shard_lock_v<mm_type>) {
            return mm_.acquire_shard_write_lock(shard_idx);
        } else if constexpr (is_striped) {
            return striped_mutex_.make_unique_lock(shard_idx % striped_mutex_.size());
        } else if constexpr (is_thread_safe) {
            return typename lock_policy::write_lock_type(mutex_);
        } else {
            return noop_lock{};
        }
    }

    /// P-CRIT-2 / P-HIGH-1 (T-C2/T-C3): Acquire read locks for ALL shards.
    ///
    /// For sharded_mm_lru, the MM data is protected by per-shard mutexes,
    /// NOT by the stripe mutexes. Global operations that iterate the MM
    /// (rbegin(), save()) must hold per-shard read locks to prevent
    /// concurrent set() from modifying the intrusive list / hash table
    /// during iteration. Returns a vector of shared_lock objects whose
    /// destructors release all locks.
    auto acquire_read_lock_all_shards() const {
        if constexpr (has_per_shard_lock_v<mm_type>) {
            std::vector<std::shared_lock<detail::distributed_shared_mutex>> locks;
            locks.reserve(mm_.num_shards());
            for (std::size_t i = 0; i < mm_.num_shards(); ++i) {
                locks.emplace_back(mm_.acquire_shard_read_lock(i));
            }
            return locks;
        } else if constexpr (is_striped) {
            return striped_read_lock_all(striped_mutex_);
        } else if constexpr (is_thread_safe) {
            return typename lock_policy::read_lock_type(mutex_);
        } else {
            return noop_lock{};
        }
    }

    /// P-CRIT-2 (T-C2): Acquire write locks for ALL shards.
    ///
    /// Used by load() to rebuild the entire MM under per-shard write locks.
    /// Returns a vector of unique_lock objects whose destructors release
    /// all locks.
    auto acquire_write_lock_all_shards() const {
        if constexpr (has_per_shard_lock_v<mm_type>) {
            std::vector<std::unique_lock<detail::distributed_shared_mutex>> locks;
            locks.reserve(mm_.num_shards());
            for (std::size_t i = 0; i < mm_.num_shards(); ++i) {
                locks.emplace_back(mm_.acquire_shard_write_lock(i));
            }
            return locks;
        } else if constexpr (is_striped) {
            return striped_write_lock_all(striped_mutex_);
        } else if constexpr (is_thread_safe) {
            return typename lock_policy::write_lock_type(mutex_);
        } else {
            return noop_lock{};
        }
    }


    /// Try to acquire a write lock for a specific key without blocking.
    /// Returns the lock if successful, or an empty (not-owned) lock if the
    /// lock cannot be acquired immediately. Used for optimistic read path
    /// where we attempt a deferred promotion under write lock but don't
    /// want to block if the lock is contended.
    auto try_acquire_write_lock_for_key(const Key& key) const {
        if constexpr (has_per_shard_lock_v<mm_type>) {
            auto hash = Hash{}(key);
            auto shard = mm_.shard_for_hash(hash);
            return mm_.try_acquire_shard_write_lock(shard);
        } else if constexpr (is_striped) {
            auto hash = Hash{}(key);
            auto stripe = striped_mutex_.stripe_for(hash);
            return striped_mutex_.try_make_unique_lock(stripe);
        } else if constexpr (is_thread_safe) {
            return std::unique_lock<typename lock_policy::mutex_type>(mutex_, std::try_to_lock);
        } else {
            return noop_lock{};
        }
    }

    // --------------------------------------------------------------------
    // T16.2: Hash-reuse lock acquisition. Accepts a pre-computed hash so
    // callers that already computed Hash{}(key) for shard dispatch can
    // reuse it for stripe selection, avoiding a second hash computation.
    // --------------------------------------------------------------------
    auto acquire_write_lock_for_hash(std::size_t hash) const {
        if constexpr (has_per_shard_lock_v<mm_type>) {
            auto shard = mm_.shard_for_hash(hash);
            return mm_.acquire_shard_write_lock(shard);
        } else if constexpr (is_striped) {
            auto stripe = striped_mutex_.stripe_for(hash);
            return striped_mutex_.make_unique_lock(stripe);
        } else if constexpr (is_thread_safe) {
            return typename lock_policy::write_lock_type(mutex_);
        } else {
            return noop_lock{};
        }
    }

    auto acquire_read_lock_for_hash(std::size_t hash) const {
        if constexpr (has_per_shard_lock_v<mm_type>) {
            auto shard = mm_.shard_for_hash(hash);
            return mm_.acquire_shard_read_lock(shard);
        } else if constexpr (is_striped) {
            auto stripe = striped_mutex_.stripe_for(hash);
            return striped_mutex_.make_shared_lock(stripe);
        } else if constexpr (is_thread_safe) {
            return typename lock_policy::read_lock_type(mutex_);
        } else {
            return noop_lock{};
        }
    }

    auto try_acquire_write_lock_for_hash(std::size_t hash) const {
        if constexpr (has_per_shard_lock_v<mm_type>) {
            auto shard = mm_.shard_for_hash(hash);
            return mm_.try_acquire_shard_write_lock(shard);
        } else if constexpr (is_striped) {
            auto stripe = striped_mutex_.stripe_for(hash);
            return striped_mutex_.try_make_unique_lock(stripe);
        } else if constexpr (is_thread_safe) {
            return std::unique_lock<typename lock_policy::mutex_type>(mutex_, std::try_to_lock);
        } else {
            return noop_lock{};
        }
    }

    /// Get the number of stripes (only meaningful for striped policies).
    /// For striped caches, returns the actual stripe count (which may differ
    /// from default_num_stripes if a custom count was supplied at construction).
    /// For non-striped policies, returns 1.
    std::size_t num_stripes() const noexcept {
        if constexpr (is_striped) {
            return striped_mutex_.size();
        } else {
            return 1;
        }
    }

    /// Compile-time default stripe count (for backwards compatibility and
    /// static_assert checks). Use num_stripes() for the runtime value.
    static constexpr std::size_t default_num_stripes() {
        if constexpr (is_striped) {
            return lock_policy::default_num_stripes;
        } else {
            return 1;
        }
    }

private:
    // ------------------------------------------------------------------
    // Members declared BEFORE mm_ are destroyed AFTER mm_ (C++ destroys
    // members in reverse declaration order). Teardown order matters:
    //
    //   1. mm_  — mm_lru's destructor flushes remaining items and
    //             synchronously reclaims them (so item memory is freed
    //             NOW, not deferred to a reclamation domain).
    //   2. per_shard_ebr_domains_ — reclaim whatever per-shard retirement
    //             is still pending.
    //   3. owned_allocator_ — the slab allocator that backs item memory.
    //             Must outlive mm_ and the EBR domains, because item
    //             deletion routes back to it. If it were destroyed first
    //             (as it would be if declared after mm_), mm_lru's flush
    //             would traverse already-freed slab memory → use-after-free.
    //   4. memory_monitor_ — owns no item memory, freed last.
    // ------------------------------------------------------------------

    /// Optional memory monitor for admission control and memory growth
    /// throttling. Default-constructed (inactive) unless configured.
    memory_monitor memory_monitor_;

    /// Owned slab allocator (created by enable_slab_allocator()).
    std::unique_ptr<slab_allocator> owned_allocator_;

    // R9 (per-shard EBR): Optional per-shard epoch domains. When enabled via
    // enable_per_shard_ebr(), each shard owns an independent epoch_domain so
    // reclamation on one shard is not blocked by readers on another. Off by
    // default: the global default_domain() is used, which keeps the per-thread
    // TLS slot cache stable for threads that read across many shards (a
    // single-slot TLS cache would thrash if the owner domain alternated).
    //
    // Declared BEFORE mm_ so destruction order destroys mm_ first (mm_lru's
    // destructor calls flush(), which retires remaining items to the shard's
    // EBR domain) and the domains LAST (their destructor reclaims whatever
    // was retired during mm_ teardown).
    std::vector<std::unique_ptr<detail::epoch_domain>> per_shard_ebr_domains_;
    bool per_shard_ebr_enabled_ = false;

    mm_type mm_;
    [[no_unique_address]] mutable mutex_storage mutex_;
    // T-P3-5: striped_mutex_ uses lazy_striped_mutex (unique_ptr-based).
    // The per-stripe distributed_shared_mutex objects are NOT allocated at
    // construction — they are deferred until the first global operation
    // (lock_all/lock_shared_all/mutex_at/etc.) that actually needs them.
    // For sharded_mm_lru caches (which have per-shard locks), per-key
    // operations never touch striped_mutex_, so the ~64 mutex objects may
    // never be allocated, saving significant memory. stripe_for() and
    // size() are O(1) non-allocating (computed from stored num_stripes_).
    [[no_unique_address]] mutable striped_mutex_storage striped_mutex_;

    // T14.1: Embedded compact_cache storage. Only present when
    // Trait::is_compact == true; otherwise this is an empty tuple (zero
    // overhead thanks to [[no_unique_address]]). When is_compact=true,
    // all storage operations (set/get/peek/remove/etc.) are delegated to
    // this member instead of mm_.
    //
    // We cannot directly condition the member on `is_compact` via
    // std::conditional_t<compact, compact_cache_t, std::monostate>
    // because compact_cache_t requires K,V,Hash,KeyEqual template params
    // that are only available here (not in cache_trait). So we use a
    // nested helper that only instantiates compact_cache when needed.
    //
    // T14.3: Compile-time size check is enforced in the compact_cache
    // itself (compact_cache.hpp line 214-215) — instantiating
    // compact_cache<K,V,...> when sizeof(K)+sizeof(V) > 64 is a
    // compile error with a clear message.
    template <bool Enabled, typename = void>
    struct compact_storage_helper {
        // Disabled (default): empty, zero-overhead.
        // Cannot use std::monostate because we need an explicit
        // operator bool / has_value() pattern below.
        compact_storage_helper() = default;
        explicit compact_storage_helper(size_type) {}
        void reset(size_type) {}
        bool has_value() const noexcept { return false; }
        template <typename... Args>
        auto& get(Args&&...) { 
            // Should never be called — guarded by `if constexpr (Trait::is_compact)`.
            // Provide a stub to satisfy the linker when is_compact=false.
            static_assert(!Enabled, "compact_storage_helper::get() called on disabled storage");
            throw std::logic_error("compact storage not enabled");
        }
    };
    template <typename Void>
    struct compact_storage_helper<true, Void> {
        // Enabled: holds a compact_cache<K,V,Hash,KeyEqual,64,alignof(max_align_t),LockPolicy>.
        using compact_cache_type = compact_cache<
            Key, Value, Hash, KeyEqual,
            64,  // T14.3: kMaxItemSize = 64
            alignof(std::max_align_t),
            lock_policy>;
        std::optional<compact_cache_type> storage_;
        compact_storage_helper() = default;
        explicit compact_storage_helper(size_type max_size) {
            storage_.emplace(max_size);
        }
        void reset(size_type max_size) {
            storage_.emplace(max_size);
        }
        bool has_value() const noexcept { return storage_.has_value(); }
        compact_cache_type& get() { return *storage_; }
        const compact_cache_type& get() const { return *storage_; }
    };
    using compact_storage = compact_storage_helper<Trait::is_compact>;
    [[no_unique_address]] compact_storage compact_storage_;

    /// T14.1: Convenience accessor for the embedded compact_cache.
    /// Only valid when Trait::is_compact == true; calling this on a
    /// non-compact cache is a compile error (guarded by the helper's
    /// static_assert).
    auto& compact() {
        static_assert(Trait::is_compact,
                      "compact() called on a non-compact unified_cache");
        return compact_storage_.get();
    }
    const auto& compact() const {
        static_assert(Trait::is_compact,
                      "compact() called on a non-compact unified_cache");
        return compact_storage_.get();
    }

    /// Optional value provider for get_or_fetch / operator[].
    std::function<Value(const Key&)> value_provider_;

    /// O2: Slow query logger — invoked when get()/set() latency exceeds
    /// `slow_query_threshold_ns_`. Stored as detail::atomic_shared_ptr
    /// (spinlock-based, since libc++ lacks std::atomic<std::shared_ptr<T>>).
    /// Set via set_slow_query_callback() — load() is the read path.
    /// When nullptr or threshold is 0, the slow-query check is a no-op
    /// (single relaxed load + branch on the fast path).
    detail::atomic_shared_ptr<slow_query_callback_type> slow_query_callback_{};
    /// O2: Threshold in nanoseconds. 0 = disabled (default). Set via
    /// set_slow_query_threshold(). Aligned to its own cache line to
    /// avoid false sharing with the value_provider_ above (which is
    /// read on every get_or_fetch miss).
    alignas(64) std::atomic<std::uint64_t> slow_query_threshold_ns_{0};
    /// O2: Cumulative count of slow operations detected since startup.
    /// Useful for alerting on tail-latency regressions. Padded to its
    /// own cache line because it's incremented on every slow op (which
    /// can happen concurrently from many threads). Mutable because
    /// notify_slow_query() is const (called from slow_query_notifier
    /// which holds a const cache reference) but the counter must be
    /// modifiable.
    alignas(64) mutable std::atomic<std::size_t> slow_query_count_{0};

    /// O1: Trace callback — invoked at the end of every get()/set() when
    /// registered. Stored as detail::atomic_shared_ptr for RCU-style
    /// lock-free reads on the hot path. Set via set_trace_callback().
    /// When nullptr, the trace check is a no-op (single relaxed load +
    /// branch on the fast path). Padded to its own cache line to avoid
    /// false sharing with the slow-query counter (which is incremented
    /// on a different code path).
    detail::atomic_shared_ptr<trace_callback_type> trace_callback_{};
    /// O1: Cumulative count of trace events fired since startup. Useful
    /// for verifying tracing is active and monitoring sampling rates.
    /// Mutable because notify_trace() is const (called from trace_notifier
    /// which holds a const cache reference).
    alignas(64) mutable std::atomic<std::size_t> trace_count_{0};

    /// T-M1: singleflight / cache stampede protection tracker.
    ///
    /// Prevents thundering-herd provider calls when multiple threads
    /// concurrently miss the same key in get_or_fetch()/try_get_or_fetch().
    /// The first miss becomes the "leader" and executes the provider;
    /// concurrent misses on the same key become "followers" that block
    /// on a condition variable until the leader completes, then receive
    /// the leader's result.
    ///
    /// Disabled by default (singleflight_enabled_ = false) to preserve
    /// the historical "every miss calls the provider" semantics for
    /// callers that rely on per-thread provider side effects. Enable
    /// explicitly via `set_singleflight_enabled(true)` when the provider
    /// is idempotent and concurrent coalescing is desired.
    ///
    /// Overhead when disabled: one bool load on the miss path. Overhead
    /// when enabled on the hit path: zero (singleflight is only consulted
    /// after a confirmed miss). On the miss path: one sharded map lookup
    /// + one mutex lock — negligible compared to the provider call.
    detail::singleflight_tracker<Key, Value> singleflight_;
    std::atomic<bool> singleflight_enabled_{false};

    /// T-D2 (P2-2): Per-cache TLS access ring config.
    ///
    /// Holds the per-cache overflow policy, auto-drain threshold, and
    /// flush callback that take precedence over the static
    /// `tls_access_ring<Key>` defaults when this cache is the active
    /// config. Activated automatically around `record_access()` calls
    /// by `record_access_in_ring()` (via `active_config_scope` RAII),
    /// so callers don't need to manage thread-local state themselves.
    ///
    /// Default-constructed to mirror the static defaults
    /// (kFlushOnFull / kRingSize / null callback), so existing behavior
    /// is unchanged unless callers explicitly customize via the public
    /// `set_tls_ring_full_policy()` / `set_tls_drain_threshold()` /
    /// `set_tls_flush_callback()` setters.
    typename tls_access_ring<Key>::tls_ring_config tls_ring_config_;

    /// Task C (P2-1 default-on): Per-cache active-handle tracking.
    /// When `per_cache_handle_tracking_` is true, every read_handle returned
    /// from get/peek/etc. is attached to `per_cache_stats_` so that
    /// active_handle_count is isolated to this cache instance (rather than
    /// the global per-Value-type counter in read_handle<T>).
    ///
    /// P2-1: This is now DEFAULT-ON. The per-T global counter in
    /// read_handle<T> is only maintained under `-DLRU_DEBUG=1` (it is
    /// a no-op in release builds to avoid cross-cache cache-line
    /// ping-pong under high read QPS). Production observability relies
    /// on this per-cache counter, which is per-instance (no cross-cache
    /// contention) and lives in a cache-line-aligned cache_stats struct.
    ///
    /// Call `set_per_cache_handle_tracking(false)` to opt out (e.g., for
    /// benchmarks that want the absolute minimal hot-path overhead and
    /// do not need active_handle_count reporting).
    cache_stats per_cache_stats_;
    bool per_cache_handle_tracking_ = true;

    // TTL jitter configuration — see set_ttl_jitter_pct() / set_ttl_jitter_enabled().
    // Default ±10% applied to every set_with_ttl() to prevent thundering-herd
    // avalanches on bulk TTL expiry. Operators may set to 0.0 to disable.
    double ttl_jitter_pct_ = 0.10;
    std::atomic<bool> ttl_jitter_enabled_{true};

    /// P1-6: Optional reclaim drain worker (created by start_event_drain()).
    /// Periodically drains hazptr/EBR pending lists and advances the EBR
    /// epoch so retired objects are reclaimed within bounded time. Runs
    /// at a short, fixed interval (default 250ms; 100ms when read
    /// handles are alive) independent of user-callback latency.
    std::unique_ptr<detail::periodic_worker> reclaim_drain_worker_;

    /// P1-6: Optional callback drain worker (created by start_event_drain()).
    /// Periodically drains TLS access/callback rings, async overload/OOM
    /// event queues, and the event tracker. Runs at the caller-supplied
    /// interval (default 1s; 500ms when read handles are alive). Slow
    /// user callbacks may stretch this worker's tick latency without
    /// affecting reclamation (which runs on reclaim_drain_worker_).
    std::unique_ptr<detail::periodic_worker> callback_drain_worker_;

    /// P0-5 (T1.3): Optional background rehash balancer. Periodically
    /// sweeps all hash-table segments to advance incremental rehash and
    /// trigger rehash_if_needed() on segments that aren't on the write
    /// hot path. Declared after the drain workers so it is destroyed
    /// first during cache destruction (the balancer calls into mm_,
    /// which must remain valid while the drain workers are still running).
    std::unique_ptr<detail::periodic_worker> rehash_balancer_worker_;

    /// P1-12: Optional event tracker for item lifecycle analysis.
    /// Created by enable_event_tracking(). Records insert/hit/evict events
    /// via callbacks registered on the cache's callback_manager. The
    /// callback_drain_worker_ periodically drains the tracker's TLS ring
    /// (via drain_all_threads()) so events are available for reporting
    /// without manual flush.
    ///
    /// Thread safety: protected by event_tracker_mutex_. The drain worker
    /// takes the mutex once per cycle (~1/sec) to load a copy of the
    /// shared_ptr; this is negligible contention compared to the cost of
    /// drain_all_threads().
    std::shared_ptr<event_tracker<Key, Hash>> event_tracker_;
    mutable std::mutex event_tracker_mutex_;

    /// Optional TTL background cleaner (created by start_ttl_cleaner()).
    /// Periodically calls evict_expired() on the MM (or each shard) to
    /// reap items whose TTL has elapsed. For MM types without native
    /// TTL support, the cleaner runs but evict_expired() is a no-op.
    std::unique_ptr<detail::periodic_worker> ttl_cleaner_worker_;

    /// T-G17: Round-robin counter for per-shard TTL cleanup. Each cleaner
    /// cycle fetches the current value, processes the corresponding shard,
    /// and increments the counter. This spreads cleaner CPU across cycles
    /// (1/num_shards per cycle) instead of scanning all shards every cycle.
    /// Default enabled; disable via `set_ttl_cleaner_round_robin(false)`.
    std::atomic<bool> ttl_cleaner_round_robin_{true};
    std::atomic<std::size_t> ttl_cleaner_next_shard_{0};

    /// P2-E: Metrics cache fields. When `metrics_cache_enabled_` is true,
    /// `stats_snapshot()` and `prometheus_text()` return the cached copy
    /// instead of rebuilding on every call. The cache is refreshed
    /// lazily by `refresh_metrics_cache()` (called periodically by
    /// `metrics_cache_worker_`, or explicitly by the operator).
    ///
    /// The snapshot is stored as a shared_ptr<stats_type> in an
    /// atomic_shared_ptr (spinlock-based) so concurrent readers can load
    /// it without taking a mutex. Concurrent rebuilds serialize on
    /// `metrics_cache_lock_`. The Prometheus text cache is built from the
    /// same snapshot so the two views are always consistent.
    ///
    /// Member declaration order matters for destruction: the worker must
    /// stop before the cached shared_ptrs are destroyed (the worker's
    /// task lambda captures `this`, so it must not access the cached
    /// fields after the cache destructor begins tearing them down). The
    /// worker is declared AFTER the cache fields so C++ destroys it
    /// first (reverse declaration order).
    mutable std::atomic<bool> metrics_cache_enabled_{false};
    mutable detail::atomic_shared_ptr<stats_type> stats_cache_;
    mutable detail::atomic_shared_ptr<std::string> prometheus_cache_;
    mutable std::mutex metrics_cache_lock_;
    std::unique_ptr<detail::periodic_worker> metrics_cache_worker_;

    // T-G12: Short-time cache for diagnostics_text() / diagnostics_json().
    // The cache amortizes 10Hz+ polling (dashboards, log aggregators) —
    // rebuilding diagnostics_info is O(num_shards) and would otherwise
    // dominate CPU under fast polling. The snapshot is a shared_ptr so
    // readers never block; rebuilds serialize on diagnostics_cache_lock_.
    // The timestamp is stored as nanoseconds since steady_clock epoch so
    // std::atomic works portably (std::atomic<time_point> is not
    // guaranteed by the standard even though time_point is trivially
    // copyable on most implementations).
    mutable std::atomic<std::uint32_t> diagnostics_cache_ttl_ms_{500};
    mutable std::atomic<std::int64_t> diagnostics_snap_ts_ns_{0};
    mutable detail::atomic_shared_ptr<const diagnostics_info> diagnostics_snap_cache_;
    mutable std::mutex diagnostics_cache_lock_;

    /// P2-4: When false (default), `prometheus_text()` emits only
    /// aggregate load-factor gauges (worst/avg/p95) — not the
    /// per-shard labelled series. Set to true via
    /// `set_prometheus_per_shard_detail(true)` for ad-hoc diagnosis.
    mutable std::atomic<bool> prometheus_per_shard_detail_{false};

    /// P1-4: Scrape counter — incremented on every stats_snapshot() call.
    /// Used by the destructor to detect high-frequency scraping without
    /// metrics cache enabled, and emit a one-time stderr hint suggesting
    /// the user call start_metrics_cache_worker() to avoid per-scrape
    /// O(N) rebuilds.
    mutable std::atomic<std::size_t> scrape_count_{0};

    /// Task 11: Graceful shutdown flag. Once set, get()/set() reject
    /// operations. Uses acq/rel ordering so shutdown() drains are
    /// visible to all threads.
    std::atomic<bool> closed_{false};

    /// T-G10: Notifier for shutdown_and_wait(). read_handle::release()
    /// calls notify_all() on this cv when the last handle is dropped
    /// during shutdown, so shutdown_and_wait() returns immediately
    /// instead of busy-polling. Wired in init_production_features().
    handle_release_notifier active_handle_notifier_;

    // ----------------------------------------------------------------
    // P1-3: Memory watermark and OOM protection
    // ----------------------------------------------------------------
    // Soft/critical watermarks as fractions of max_memory in [0.0, 1.0].
    // When current_memory exceeds the soft watermark, new insertions
    // trigger aggressive eviction. When it exceeds the critical
    // watermark, the cache enters read-only mode (set() rejects new
    // items) until memory drops below the soft watermark.
    //
    // Defaults: soft=0.85, critical=0.95. Set to 1.0 to effectively
    // disable (backwards-compatible behavior).
    std::atomic<double> memory_soft_watermark_{0.85};
    std::atomic<double> memory_critical_watermark_{0.95};
    std::atomic<bool> memory_critical_mode_{false};

    /// P1-3: OOM handler callback. Invoked when the cache enters
    /// critical memory mode. The handler can log alerts, trigger
    /// external GC, or initiate graceful degradation. Must not throw
    /// or block — it is called from the set() hot path.
    std::function<void(size_type current, size_type max)> oom_handler_;

    /// P-HIGH-2 (T-H1): When true, OOM events are enqueued on
    /// pending_oom_event_ and dispatched by the event_drain_worker
    /// (see drain_oom_events()). When false (default), the handler
    /// is invoked synchronously outside the write lock.
    std::atomic<bool> async_oom_handler_{false};

    /// P-HIGH-2 (T-H1): Pending OOM event for async dispatch.
    /// Protected by pending_oom_event_mutex_. At most one event is
    /// retained (coalesced) — see dispatch_oom_event() rationale.
    std::optional<oom_event> pending_oom_event_;
    mutable std::mutex pending_oom_event_mutex_;

    // RAII guard that calls flush_pending() on destruction.
    // For sharded_mm_lru, used ONLY with global write locks (lock_all),
    // where all stripes are held exclusively so iterating shards is safe.
    // For per-key operations on striped caches, use flush_shard_pending()
    // inside the stripe lock instead.
    // Construct before the lock so that destruction order is:
    //   1) lock released ( unlocks mutex / stripes )
    //   2) flush_guard destroyed ( flushes pending callbacks outside the lock )
    struct flush_guard {
        mm_type& mm;
        ~flush_guard() {
            try {
                if constexpr (requires { mm.flush_pending_all(); }) {
                    mm.flush_pending_all();
                } else {
                    mm.callbacks().flush_pending();
                }
            } catch (...) {
                // Swallow exceptions during stack unwinding to prevent
                // std::terminate. Callback exceptions are non-fatal; the
                // cache state remains consistent regardless.
            }
        }
    };

    /// Flush the pending callbacks for the shard that owns `key`.
    /// Must be called while holding the stripe lock for that key.
    /// For non-sharded MM types, this is a no-op (flush_guard handles it).
    void flush_shard_pending(const Key& key) {
        if constexpr (requires { mm_.shard(0); }) {
            mm_.shard(mm_.shard_for(key)).callbacks().flush_pending();
        }
    }

    /// Task 8: Overload that accepts a pre-computed shard_idx to avoid
    /// redundant shard_for(key) calls when the caller already has the index.
    void flush_shard_pending(std::size_t shard_idx) {
        if constexpr (requires { mm_.shard(0); }) {
            mm_.shard(shard_idx).callbacks().flush_pending();
        }
    }

    /// Task C: Attach per-cache stats pointer to a read_handle when
    /// per_cache_handle_tracking_ is enabled. Called on every get/peek
    /// path returning a handle. When disabled, this is a single bool
    /// load + branch (essentially free).
    template <typename H>
    void attach_handle_stats_if_enabled(H& handle) noexcept {
        if (per_cache_handle_tracking_ && handle) {
            handle.attach_per_cache_stats(&per_cache_stats_);
        }
    }

    /// Task D: Apply overflow policy before inserting a new item.
    /// Must be called while holding the write lock for `key`.
    /// Returns true if the insert should proceed, false if it should be
    /// rejected (kRejectInsert policy hit). For kForceEvict, evicts the
    /// LRU tail item (bypassing pin check) to make room and returns true.
    /// For kAllowGrowth (default), always returns true.
    ///
    /// Updates to existing keys always proceed (no overflow check) —
    /// only fresh inserts are subject to the policy.
    bool apply_overflow_policy_for_set(const Key& key, std::size_t shard_idx) {
        // Read config; if kAllowGrowth, fast path (no work).
        // Config access is read-only under the write lock we hold.
        // Use `requires` to detect whether the MM config exposes the
        // overflow_policy_value field — only mm_lru_config does. Other
        // MM types (mm_2q, mm_tiny_lfu, mm_wtiny_lfu, mm_fifo) fall back
        // to kAllowGrowth, which is the existing behavior.
        overflow_policy policy = overflow_policy::kAllowGrowth;
        double tolerance = 0.0;
        if constexpr (is_striped) {
            if constexpr (requires { mm_.shard(0).config().overflow_policy_value; }) {
                const auto& cfg = mm_.shard(shard_idx).config();
                policy = cfg.overflow_policy_value;
                tolerance = cfg.overflow_tolerance;
            }
        } else {
            if constexpr (requires { mm_.config().overflow_policy_value; }) {
                const auto& cfg = mm_.config();
                policy = cfg.overflow_policy_value;
                tolerance = cfg.overflow_tolerance;
            }
        }
        if (policy == overflow_policy::kAllowGrowth) return true;
        // Existing key (update) bypasses overflow check.
        if constexpr (is_striped) {
            if (mm_.shard(shard_idx).contains(key)) return true;
        } else {
            if (mm_.contains(key)) return true;
        }
        // Current size and effective cap.
        std::size_t cur_size, max_sz;
        if constexpr (is_striped) {
            cur_size = mm_.shard(shard_idx).size();
            max_sz   = mm_.shard(shard_idx).max_size();
        } else {
            cur_size = mm_.size();
            max_sz   = mm_.max_size();
        }
        // max_size == unlimited means no cap.
        if (max_sz == unlimited) return true;
        std::size_t effective_cap = static_cast<std::size_t>(
            static_cast<double>(max_sz) * (1.0 + tolerance));
        if (cur_size < effective_cap) return true;
        // Over the soft cap — apply policy.
        if (policy == overflow_policy::kRejectInsert) {
            return false;
        }
        if (policy == overflow_policy::kForceEvict) {
            // Force-evict the LRU tail item, bypassing pin check.
            // The item's memory is reclaimed when outstanding handles release.
            // Use the public peek_lru_tail_key() + force_del() API to avoid
            // exposing internal item_type pointers across the MM boundary.
            if constexpr (is_striped) {
                if constexpr (requires { mm_.peek_lru_tail_key(shard_idx); }) {
                    auto victim_key = mm_.peek_lru_tail_key(shard_idx);
                    if (victim_key) {
                        mm_.shard(shard_idx).force_del(*victim_key);
                    }
                }
            } else {
                if constexpr (requires { mm_.peek_lru_tail_key(); mm_.force_del(key); }) {
                    auto victim_key = mm_.peek_lru_tail_key();
                    if (victim_key) {
                        mm_.force_del(*victim_key);
                    }
                }
            }
            return true;
        }
        return true;
    }

public:
    /// Drain the TLS access ring: batch-promote all deferred keys.
    /// Should be called periodically (e.g., by periodic_worker or manually
    /// via flush()). Uses try_lock to avoid blocking: if the write lock for a
    /// key's stripe is contended, the key is re-recorded to the TLS ring and
    /// will be retried on the next drain cycle. This aligns with CacheLib's
    /// non-blocking drain path, reducing tail latency under contention.
    ///
    /// Public so users can invoke it from a `set_tls_flush_callback()` handler
    /// (e.g., `cache.set_tls_flush_callback([&]{ cache.drain_access_ring(); })`)
    /// or call it manually after a burst of `get()` traffic to amortize
    /// promotion cost.
    ///
    /// When the lock IS available, each key is promoted immediately and
    /// callbacks are collected. When the lock is NOT available, the key
    /// stays in the TLS ring (re-recorded) for the next drain attempt.
    ///
    /// No hazard pointer protection is needed in this path because the
    /// read_handle returned by peek() provides reference-count-based
    /// protection. While a read_handle is alive (h), the pointed-to item
    /// cannot be evicted — its refcount is incremented, causing
    /// has_active_handle() to return true and find_eviction_victim() to
    /// skip it.
    ///
    /// Dedup is performed via sort + unique to avoid redundant promote()
    /// calls, especially important with defer_promotion where every get()
    /// records to the ring. This avoids the heap allocation that
    /// ankerl::unordered_dense::set would incur while still being efficient.
    ///
    /// Optimization: when the drained key count is small (≤ 64), we skip the
    /// sort+unique dedup pass entirely and call promote_keys() directly —
    /// promote() is idempotent so redundant calls are harmless and the
    /// overhead of sort+unique exceeds the cost of a few duplicate promotes
    /// for small batches. The threshold of 64 matches the default TLS ring
    /// size, so a single full ring drains without dedup.
    void drain_access_ring() {
        // P1-1: Track flush frequency for observability.
        if constexpr (is_striped) {
            for (std::size_t i = 0; i < mm_.num_shards(); ++i) {
                mm_.shard(i).stats().tls_ring_flush_count.fetch_add(
                    1, std::memory_order_relaxed);
            }
        } else {
            mm_.stats().tls_ring_flush_count.fetch_add(
                1, std::memory_order_relaxed);
        }

        // Phase 1: Drain the global backup buffer first — these are keys
        // orphaned by threads that have already exited. Any surviving
        // thread must pick them up to avoid lost promotions.
        if (tls_access_ring<Key>::has_backup_keys()) {
            auto backup = tls_access_ring<Key>::drain_backup();
            if (!backup.keys.empty()) {
                auto& bk = backup.keys;
                if (bk.size() <= 64) {
                    // Small batch: skip dedup, promote() is idempotent.
                    promote_keys(bk);
                } else {
                    std::sort(bk.begin(), bk.end());
                    bk.erase(std::unique(bk.begin(), bk.end()), bk.end());
                    promote_keys(bk);
                }
            }
        }

        // Phase 2: Drain the current thread's TLS ring
        auto& ring = tls_access_ring<Key>::instance();
        if (ring.empty()) return;
        auto drained = ring.drain();
        if (drained.keys.empty()) return;
        // Small batch optimization: skip dedup for ≤ 64 keys.
        auto& keys = drained.keys;
        if (keys.size() <= 64) {
            promote_keys(keys);
        } else {
            std::sort(keys.begin(), keys.end());
            keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
            promote_keys(keys);
        }
    }

private:
    /// Promote a deduplicated set of keys. For each key, tries a
    /// non-blocking write lock; if contended, re-records to the TLS
    /// ring for later retry. Shared by both backup-drain and TLS-drain.
    ///
    /// P1-6: Bounded spin-then-yield backoff before re-recording. Without
    /// backoff, a hot key under sustained write contention could be
    /// re-recorded on every drain attempt, causing:
    ///   - Promotion starvation (hot key never reaches MRU)
    ///   - TLS ring pressure (each retry occupies a slot)
    ///   - Wasted CPU on repeated failed try_lock
    /// The backoff gives the write lock holder a brief window to release
    /// before deferring to the next drain, dramatically reducing
    /// re-record rate under moderate contention. Under heavy contention
    /// the backoff is bounded (≤ kMaxBackoffIters yields) so the drain
    /// path never blocks for long.
    void promote_keys(std::vector<Key>& keys) {
        // P1-6: Bounded backoff iterations. Tuned for read-heavy-write-light
        // workloads: enough to absorb brief lock hold times (typical write
        // critical section is <1µs), small enough to never stall the drain
        // path. Each iteration is a single yield(), so 8 iters ≈ 8 scheduler
        // quanta ≈ tens of microseconds worst case.
        constexpr int kMaxBackoffIters = 8;

        // T-P2-2 (R-5): Dedup input keys to avoid retrying the same hot key
        // multiple times in one drain cycle. Without dedup, a hot key
        // appearing K times in `keys` (because it was accessed K times
        // since the last drain) would cause:
        //   - K × (1 + kMaxBackoffIters) try_lock attempts — wasted CPU
        //     under write contention (each attempt is a CAS + yield)
        //   - K re-records on failure — ring pollution, starving other keys
        //     (each re-record occupies a ring slot; a hot key under
        //     sustained contention could fill the ring with its own
        //     retries, evicting other keys' access records and causing
        //     `tls_ring_dropped_promotions` to grow)
        // Dedup merges K appearances into 1 attempt + 1 re-record,
        // preserving the access signal (the key IS hot) without the
        // overhead. The sort is O(N log N) — for typical N ≤ 64, this is
        // ~microseconds, far less than the K × 9 try_lock attempts it
        // saves under contention.
        //
        // Note: callers may have already dedup'd for batches > 64; the
        // double dedup here is a fast no-op on already-sorted+unique data.
        if (keys.size() > 1) {
            std::sort(keys.begin(), keys.end());
            keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
        }

        // P1-C: Shard indices whose TLS callback ring was touched during
        // this drain cycle. `flush_pending()` may invoke user callbacks
        // (on_hit etc.) which can perform IO/logging — running them inside
        // the per-key write lock extends the critical section for no
        // benefit (callback dispatch is independent of MM state). We
        // defer the flush until after the lock is released, batching all
        // pending shards together.
        std::vector<std::size_t> shards_to_flush;
        shards_to_flush.reserve(keys.size());

        for (auto& key : keys) {
            bool promoted = false;
            for (int attempt = 0; attempt <= kMaxBackoffIters; ++attempt) {
                auto lock = try_acquire_write_lock_for_key(key);
                if (lock.owns_lock()) {
                    mm_.promote(key);
                    auto h = mm_.peek(key);
                    if constexpr (is_striped) {
                        // Task B: cache shard_idx once per key (avoid double hash).
                        const std::size_t shard_idx = mm_.shard_for(key);
                        if (h) mm_.shard(shard_idx).callbacks().collect_hit(key, *h);
                        // P1-C: do NOT flush_shard_pending() here — defer
                        // until after lock release. collect_hit() is a
                        // TLS ring push (no cross-thread sync), so it is
                        // safe and cheap to do under the lock; only the
                        // subsequent flush_pending() may invoke user
                        // callbacks and must be done outside.
                        shards_to_flush.push_back(shard_idx);
                    } else {
                        if (h) mm_.callbacks().collect_hit(key, *h);
                    }
                    promoted = true;
                    break;
                }
                // P1-6: Backoff before retrying. Use yield() rather than
                // raw pause — promotes fairness across cores and lets the
                // lock holder make progress on the same physical core.
                std::this_thread::yield();
            }
            if (!promoted) {
                // Lock still contended after bounded backoff — re-record to
                // TLS ring for the next drain. This preserves correctness
                // (promotion eventually happens once contention subsides)
                // while bounding drain-path latency under heavy contention.
                //
                // T-P2-2 (R-5): Because input was dedup'd above, each
                // unique failed key is re-recorded at most once per
                // promote_keys call — no ring pollution.
                record_access_in_ring(key);
            }
        }

        // P1-C: Flush pending callbacks outside any write lock. User
        // callbacks (on_hit) may perform IO or logging and must not extend
        // the per-key write critical section. Dedup the shard indices
        // first so we don't flush the same shard's TLS ring multiple
        // times in one drain cycle (flush_pending() drains the entire
        // ring, so subsequent calls on the same shard are no-ops anyway,
        // but dedup keeps the loop tight under high key counts).
        if constexpr (is_striped) {
            if (shards_to_flush.size() > 1) {
                std::sort(shards_to_flush.begin(), shards_to_flush.end());
                shards_to_flush.erase(
                    std::unique(shards_to_flush.begin(), shards_to_flush.end()),
                    shards_to_flush.end());
            }
            for (std::size_t idx : shards_to_flush) {
                flush_shard_pending(idx);
            }
        }
        // For non-sharded MM types, the global callback ring is flushed by
        // the drain worker's existing post-promote path (see drain_event_*).
    }

    // --------------------------------------------------------------------
    // Memory monitor helpers
    // --------------------------------------------------------------------

    /// Check if a new item with `key`/`value` should be admitted by the memory
    /// monitor. Existing keys always pass (updates are permitted; eviction
    /// handles overflow). Must be called while holding the appropriate cache
    /// lock so that mm_.contains() and mm_.estimate_item_memory() are stable.
    ///
    /// T-G2: Updates no longer bypass admission. Previously, an existing
    /// key could be updated with a much larger value without any memory
    /// check, causing `current_memory` to silently exceed
    /// `max_memory_bytes`. Now we conservatively check
    /// `current_memory + new_value_size` (an upper bound — the old value
    /// will be freed by the upcoming set(), so actual post-set memory is
    /// lower). This may over-reject updates when the new value is much
    /// larger than the old, but it guarantees the memory budget is never
    /// exceeded, which is the safer default for production.
    template <typename V>
    bool check_memory_admission(const Key& key, const V& value) {
        if (!memory_monitor_.active()) return true;
        const std::size_t new_item_size = mm_.estimate_item_memory(key, value);
        if (mm_.contains(key)) {
            // G2 fix: Update path — conservatively check current_memory +
            // new_item_size (upper bound). The old value will be freed
            // during set(), so actual post-set memory is
            // current_memory - old_item_size + new_item_size <=
            // current_memory + new_item_size. We check the upper bound to
            // avoid requiring a peek-of-old-value-size API. Previously this
            // branch called should_admit(new_item_size) — identical to the
            // insert path — which silently let updates bypass the memory
            // budget (a large value replacing a small one could push
            // current_memory far past max_memory_bytes).
            const std::size_t current_mem = mm_.current_memory();
            return memory_monitor_.should_admit(current_mem + new_item_size);
        }
        return memory_monitor_.should_admit(new_item_size);
    }

    /// Report current cache memory to the monitor if it is active.
    /// Must be called while holding the appropriate cache lock so that
    /// mm_.current_memory() is stable.
    void maybe_report_memory_to_monitor() {
        if (memory_monitor_.active()) {
            memory_monitor_.report_memory(mm_.current_memory());
        }
    }

    // --------------------------------------------------------------------
    // Equality comparison (read locks on both sides)
    // --------------------------------------------------------------------

    friend bool operator==(const unified_cache& a, const unified_cache& b) {
        if (&a == &b) return true;
        // Acquire read locks in address order to prevent AB-BA deadlock.
        const unified_cache* first = std::addressof(a) < std::addressof(b) ? &a : &b;
        const unified_cache* second = std::addressof(a) < std::addressof(b) ? &b : &a;
        auto l1 = first->acquire_read_lock();
        auto l2 = second->acquire_read_lock();
        return a.mm_.size() == b.mm_.size() && a.mm_ == b.mm_;
    }
    friend bool operator!=(const unified_cache& a, const unified_cache& b) { return !(a == b); }

    // --------------------------------------------------------------------
    // Stream output (acquires read lock)
    // --------------------------------------------------------------------

    friend std::ostream& operator<<(std::ostream& os, const unified_cache& c) {
        auto lock = c.acquire_read_lock();
        os << "unified_cache @" << &c << "  type=";
        if constexpr (std::is_same_v<mm_type, mm_lru<Key, Value, Hash, KeyEqual>>) os << "mm_lru";
        else if constexpr (std::is_same_v<mm_type, mm_lru<Key, Value, Hash, KeyEqual, detail::chain_probing_tag, true>>) os << "mm_lru[segmented]";
        else if constexpr (std::is_same_v<mm_type, mm_fifo<Key, Value, Hash, KeyEqual>>) os << "mm_fifo";
        else if constexpr (std::is_same_v<mm_type, mm_fifo<Key, Value, Hash, KeyEqual, detail::chain_probing_tag, true>>) os << "mm_fifo[segmented]";
        else if constexpr (std::is_same_v<mm_type, mm_2q<Key, Value, Hash, KeyEqual>>) os << "mm_2q";
        else if constexpr (std::is_same_v<mm_type, mm_2q<Key, Value, Hash, KeyEqual, detail::chain_probing_tag, true>>) os << "mm_2q[segmented]";
        else if constexpr (std::is_same_v<mm_type, mm_tiny_lfu<Key, Value, Hash, KeyEqual>>) os << "mm_tiny_lfu";
        else if constexpr (std::is_same_v<mm_type, mm_tiny_lfu<Key, Value, Hash, KeyEqual, detail::chain_probing_tag, true>>) os << "mm_tiny_lfu[segmented]";
        else if constexpr (std::is_same_v<mm_type, mm_wtiny_lfu<Key, Value, Hash, KeyEqual>>) os << "mm_wtiny_lfu";
        else if constexpr (std::is_same_v<mm_type, mm_wtiny_lfu<Key, Value, Hash, KeyEqual, detail::chain_probing_tag, true>>) os << "mm_wtiny_lfu[segmented]";
        else if constexpr (std::is_same_v<mm_type, sharded_mm_lru<Key, Value, Hash, KeyEqual>>) os << "sharded_mm_lru";
        else if constexpr (std::is_same_v<mm_type, sharded_mm_lru<Key, Value, Hash, KeyEqual, detail::chain_probing_tag, true>>) os << "sharded_mm_lru[segmented]";
        else os << "unknown";
        os << "  " << c.mm_.stats() << "\n";
        if constexpr (requires { c.mm_.shard(0); }) {
            // Sharded: print per-shard summary
            for (std::size_t i = 0; i < c.mm_.num_shards(); ++i) {
                const auto& s = c.mm_.shard(i);
                if (!s.empty()) {
                    os << "  [shard " << i << "] size=" << s.size() << "\n";
                }
            }
        } else {
            std::size_t idx = 0;
            for (auto it = c.mm_.begin(); it != c.mm_.end(); ++it, ++idx) {
                os << "  [" << idx << "] key='";
                if constexpr (lru::detail::is_formattable_v<Key>) {
                    os << std::format("{}", it->key);
                } else {
                    os << std::format("<key at {:#x}>", reinterpret_cast<std::uintptr_t>(&it->key));
                }
                os << "' value='";
                if constexpr (lru::detail::is_formattable_v<Value>) {
                    os << std::format("{}", it->value);
                } else {
                    os << std::format("<val at {:#x}>", reinterpret_cast<std::uintptr_t>(&it->value));
                }
                os << "'\n";
            }
        }
        return os;
    }
};

} // namespace lru

#endif // LRU_CACHE_TRAIT_HPP
// test
