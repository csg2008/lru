// Unified LRU Cache Library
// A high-performance, feature-rich LRU cache implementation for C++20
//
// Architecture migrated from Facebook's CacheLib:
//   - Intrusive doubly-linked list (eliminates per-node heap allocation)
//   - Per-node updateTime for delayed promotion
//   - Adaptive refresh time based on tail age
//   - Incremental insertion point maintenance (O(1) amortized)
//   - tryLockUpdate optimization
//   - peek() and updateOnWrite/updateOnRead fine-grained control
//   - CountMinSketch with dynamic growth and step decay
//   - Striped locking for better read concurrency
//   - Async callback system (collect inside lock, flush outside)
//   - Trait-based compile-time strategy architecture
//
// SPDX-License-Identifier: MIT

#ifndef LRU_LRU_HPP
#define LRU_LRU_HPP

// Core foundation (concepts + traits + callbacks + statistics + iterator)
#include "core.hpp"

// Internal utilities (utils + periodic_worker + striped_mutex)
#include "detail/foundation.hpp"

// P0-A: well-mixed default hash.
// ankerl::unordered_dense::hash provides high-quality mixing for integer keys
// (splitmix64-based), eliminating the "identity hash + no shard mixing" hot
// shard problem documented in P0-A. It is already a hard dependency of the
// library (find_package(unordered_dense) in CMakeLists). When unavailable,
// we fall back to std::hash<Key> (preserving build but degrading shard
// distribution for sequential integer keys — diagnostics_text() surfaces
// this via hot_shards()).
#if __has_include(<ankerl/unordered_dense.h>)
    #include <ankerl/unordered_dense.h>
    #define LRU_HAS_WELL_MIXED_HASH 1
#endif

// Infrastructure
#include "detail/count_min_sketch.hpp"
#include "detail/intrusive_list.hpp"

// Eviction strategies (all in one header)
#include "mm.hpp"

// Unified trait-based architecture
#include "cache_trait.hpp"

// TTL support
#include "ttl.hpp"

// CacheLib-migrated advanced features
#include "admission.hpp"
#include "chained_item.hpp"
#include "compact_cache.hpp"
#include "compressed_ptr.hpp"
#include "event_tracker.hpp"
#include "hot_key_replica.hpp"
#include "memory.hpp"
#include "pooled_cache.hpp"
#include "reshard_utils.hpp"
#include "serialization.hpp"
#include "shared_memory_backend.hpp"
#include "tls_ring.hpp"
#include "tiered_storage.hpp"
#include "warm_cache.hpp"

// Version info
#define LRU_CACHE_VERSION_MAJOR 4
#define LRU_CACHE_VERSION_MINOR 0
#define LRU_CACHE_PATCH 0

namespace lru {

/// Library version string.
inline constexpr char version_string[] = "4.0.0";

/// Library version number (MAJOR * 10000 + MINOR * 100 + PATCH).
inline constexpr int version_number = 40000;

// ============================================================================
// P0-A: Default hash — well-mixed to prevent hot shards.
// ============================================================================
// Previously every alias used `std::hash<Key>`, which for integer keys is the
// identity function (`std::hash<int>{}(i) == i`). Combined with
// `sharded_mm_lru::shard_for(key) = Hash{}(key) % num_shards_` (no extra
// mixing — see R-9), sequential integer keys (auto-increment IDs, timestamps,
// round-robin counters) all landed on the first N shards, producing severe
// hot-shard skew under 32+ thread read contention.
//
// `default_hash` selects `ankerl::unordered_dense::hash` when available
// (high-quality splitmix64-based mixing, already a library dependency) and
// falls back to `std::hash<Key>` otherwise. Callers can always override by
// passing an explicit Hash template parameter.
#if defined(LRU_HAS_WELL_MIXED_HASH)
template <typename Key>
using default_hash = ankerl::unordered_dense::hash<Key>;
#else
template <typename Key>
using default_hash = std::hash<Key>;
#endif

// ============================================================================
// Convenience type aliases
// ============================================================================

/// Enhanced LRU cache with CacheLib features (insertion point, delayed promotion, adaptive refresh)
template <typename Key, typename Value,
          typename Hash = default_hash<Key>,
          typename KeyEqual = std::equal_to<Key>>
using enhanced_lru_cache = mm_lru<Key, Value, Hash, KeyEqual>;

/// 2Q cache
template <typename Key, typename Value,
          typename Hash = default_hash<Key>,
          typename KeyEqual = std::equal_to<Key>>
using two_q_cache = mm_2q<Key, Value, Hash, KeyEqual>;

/// TinyLFU cache
template <typename Key, typename Value,
          typename Hash = default_hash<Key>,
          typename KeyEqual = std::equal_to<Key>>
using tiny_lfu_cache = mm_tiny_lfu<Key, Value, Hash, KeyEqual>;

/// W-TinyLFU cache
template <typename Key, typename Value,
          typename Hash = default_hash<Key>,
          typename KeyEqual = std::equal_to<Key>>
using w_tiny_lfu_cache = mm_wtiny_lfu<Key, Value, Hash, KeyEqual>;

/// FIFO cache (raw mm_fifo, no unified_cache wrapper)
template <typename Key, typename Value,
          typename Hash = default_hash<Key>,
          typename KeyEqual = std::equal_to<Key>>
using fifo_cache_raw = mm_fifo<Key, Value, Hash, KeyEqual>;

// ============================================================================
// Unified cache convenience aliases
// ============================================================================

/// Single-threaded LRU cache (unified_cache + lru_trait).
/// 用法：lru::cache<int, std::string> c(100);
template <typename Key, typename Value,
          typename Hash = default_hash<Key>,
          typename KeyEqual = std::equal_to<Key>>
using cache = unified_cache<lru_trait<single_threaded_policy>, Key, Value, Hash, KeyEqual>;

/// Thread-safe LRU cache (unified_cache + safe_lru_trait + thread_safe_policy).
/// WARNING: Uses a single global distributed_shared_mutex for all operations.
/// In high-concurrency scenarios, occasional writes (set/remove) block ALL
/// concurrent reads. For production read-heavy workloads, prefer
/// production_cache, striped_cache, or segmented_striped_cache.
///
/// T-P3 (R-1 / R-2): This alias now defaults to production-safe settings:
///   - `reader_preferred` fairness (max read throughput, may starve writers
///     under sustained read load — acceptable for read-heavy-write-light,
///     the documented target of this library). Call
///     `c.set_fairness_mode(lru::detail::fairness_mode::writer_fair)` after
///     construction for mixed/write-heavy workloads that need strict writer
///     fairness.
///   - Incremental rehash enabled (no global write stall when the hash table
///     grows past its load factor).
///
/// Users who need the historical `writer_fair` + non-incremental rehash
/// defaults can instantiate
/// `lru::unified_cache<lru::lru_trait<lru::thread_safe_policy>, K, V>` directly.
///
/// 用法：lru::safe_cache<int, std::string> c(100);
template <typename Key, typename Value,
          typename Hash = default_hash<Key>,
          typename KeyEqual = std::equal_to<Key>>
using safe_cache = unified_cache<safe_lru_trait<thread_safe_policy>, Key, Value, Hash, KeyEqual>;

/// Striped thread-safe LRU cache (unified_cache + safe_sharded_lru_trait + striped_thread_safe_policy<>).
/// Uses sharded_mm_lru with per-shard striped locking for high concurrency.
///
/// T-P3 (R-1 / R-2): This alias now defaults to production-safe settings:
///   - `reader_preferred` fairness (max read throughput, may starve writers
///     under sustained read load — acceptable for read-heavy-write-light).
///     Call `c.set_fairness_mode(lru::detail::fairness_mode::writer_fair)`
///     after construction for mixed/write-heavy workloads.
///   - Incremental rehash enabled (no global write stall during rehash).
///
/// Users who need the historical `writer_fair` + non-incremental rehash
/// defaults can instantiate
/// `lru::unified_cache<lru::sharded_lru_trait<lru::striped_thread_safe_policy<>>, K, V>` directly.
template <typename Key, typename Value,
          typename Hash = default_hash<Key>,
          typename KeyEqual = std::equal_to<Key>>
using striped_cache = unified_cache<safe_sharded_lru_trait<striped_thread_safe_policy<>>, Key, Value, Hash, KeyEqual>;

/// Single-threaded TinyLFU cache (unified_cache + tiny_lfu_trait).
template <typename Key, typename Value,
          typename Hash = default_hash<Key>,
          typename KeyEqual = std::equal_to<Key>>
using lfu_cache = unified_cache<tiny_lfu_trait<single_threaded_policy>, Key, Value, Hash, KeyEqual>;

/// Thread-safe TinyLFU cache (unified_cache + tiny_lfu_trait + thread_safe_policy).
/// Uses a single distributed_shared_mutex for shared reads / exclusive writes.
/// For high-concurrency read-heavy workloads, prefer production_cache (LRU)
/// — only mm_lru / sharded_mm_lru support striped locking and EBR.
template <typename Key, typename Value,
          typename Hash = default_hash<Key>,
          typename KeyEqual = std::equal_to<Key>>
using safe_lfu_cache = unified_cache<tiny_lfu_trait<thread_safe_policy>, Key, Value, Hash, KeyEqual>;

/// Single-threaded W-TinyLFU cache (unified_cache + w_tiny_lfu_trait).
template <typename Key, typename Value,
          typename Hash = default_hash<Key>,
          typename KeyEqual = std::equal_to<Key>>
using w_tiny_lfu = unified_cache<w_tiny_lfu_trait<single_threaded_policy>, Key, Value, Hash, KeyEqual>;

/// Thread-safe W-TinyLFU cache (unified_cache + w_tiny_lfu_trait + thread_safe_policy).
/// Uses a single distributed_shared_mutex. EBR is not supported by mm_wtiny_lfu.
/// For read-heavy W-TinyLFU workloads, prefer `read_heavy_w_tiny_lfu`
/// (defer_promotion + reader_preferred fairness).
template <typename Key, typename Value,
          typename Hash = default_hash<Key>,
          typename KeyEqual = std::equal_to<Key>>
using safe_w_tiny_lfu = unified_cache<w_tiny_lfu_trait<thread_safe_policy>, Key, Value, Hash, KeyEqual>;

/// Single-threaded 2Q cache (unified_cache + two_q_trait).
template <typename Key, typename Value,
          typename Hash = default_hash<Key>,
          typename KeyEqual = std::equal_to<Key>>
using two_q = unified_cache<two_q_trait<single_threaded_policy>, Key, Value, Hash, KeyEqual>;

/// Thread-safe 2Q cache (unified_cache + two_q_trait + thread_safe_policy).
/// Uses a single distributed_shared_mutex.
template <typename Key, typename Value,
          typename Hash = default_hash<Key>,
          typename KeyEqual = std::equal_to<Key>>
using safe_two_q = unified_cache<two_q_trait<thread_safe_policy>, Key, Value, Hash, KeyEqual>;

/// Single-threaded FIFO cache (unified_cache + fifo_trait).
/// FIFO: first-in-first-out eviction. Accessing an item does NOT
/// change its position in the eviction queue.
template <typename Key, typename Value,
          typename Hash = default_hash<Key>,
          typename KeyEqual = std::equal_to<Key>>
using fifo_cache = unified_cache<fifo_trait<single_threaded_policy>, Key, Value, Hash, KeyEqual>;

/// Thread-safe FIFO cache (unified_cache + fifo_trait + thread_safe_policy).
/// Uses a single distributed_shared_mutex.
template <typename Key, typename Value,
          typename Hash = default_hash<Key>,
          typename KeyEqual = std::equal_to<Key>>
using safe_fifo_cache = unified_cache<fifo_trait<thread_safe_policy>, Key, Value, Hash, KeyEqual>;

// ============================================================================
// F14 SIMD probing convenience aliases
// ============================================================================
//
// F14 aliases use 14-slot chunks with 8-bit hash tags for faster lookups
// via SIMD parallel tag comparison (SSE2/NEON/scalar fallback).
//
// Slab allocator integration:
//   For non-EmbeddedChain hash tables, custom node allocation can be
//   configured via the MM config's alloc_fn/dealloc_fn fields, or by
//   calling set_hash_alloc_fns() on the cache instance. This allows
//   hash table overflow chain nodes to be allocated from a slab
//   allocator instead of the default heap, reducing allocation overhead
//   and memory fragmentation in high-throughput scenarios.
//
//   Example:
//     lru::mm_lru_config cfg;
//     cfg.alloc_fn = my_slab_alloc;
//     cfg.dealloc_fn = my_slab_dealloc;
//     lru::f14_cache<int, std::string> c(1000, cfg);
//
//   Or after construction:
//     lru::f14_cache<int, std::string> c(1000);
//     c.set_hash_alloc_fns(my_slab_alloc, my_slab_dealloc);

/// Single-threaded LRU cache with F14 SIMD hash probing.
/// Uses 14-slot chunks with 8-bit hash tags for faster lookups via
/// SIMD parallel tag comparison (SSE2/NEON/scalar fallback).
/// Usage: lru::f14_cache<int, std::string> c(100);
template <typename Key, typename Value,
          typename Hash = default_hash<Key>,
          typename KeyEqual = std::equal_to<Key>>
using f14_cache = unified_cache<f14_lru_trait<single_threaded_policy>, Key, Value, Hash, KeyEqual>;

/// Thread-safe LRU cache with F14 SIMD hash probing.
/// Usage: lru::f14_safe_cache<int, std::string> c(100);
template <typename Key, typename Value,
          typename Hash = default_hash<Key>,
          typename KeyEqual = std::equal_to<Key>>
using f14_safe_cache = unified_cache<f14_lru_trait<thread_safe_policy>, Key, Value, Hash, KeyEqual>;

/// Striped thread-safe LRU cache with F14 SIMD hash probing.
/// Combines per-shard striped locking with F14 SIMD probing for maximum
/// read throughput under high concurrency.
template <typename Key, typename Value,
          typename Hash = default_hash<Key>,
          typename KeyEqual = std::equal_to<Key>>
using f14_striped_cache = unified_cache<f14_sharded_lru_trait<striped_thread_safe_policy<>>, Key, Value, Hash, KeyEqual>;

/// Single-threaded TinyLFU cache with F14 SIMD hash probing.
template <typename Key, typename Value,
          typename Hash = default_hash<Key>,
          typename KeyEqual = std::equal_to<Key>>
using f14_lfu_cache = unified_cache<f14_tiny_lfu_trait<single_threaded_policy>, Key, Value, Hash, KeyEqual>;

/// Single-threaded W-TinyLFU cache with F14 SIMD hash probing.
template <typename Key, typename Value,
          typename Hash = default_hash<Key>,
          typename KeyEqual = std::equal_to<Key>>
using f14_w_tiny_lfu = unified_cache<f14_w_tiny_lfu_trait<single_threaded_policy>, Key, Value, Hash, KeyEqual>;

/// Single-threaded 2Q cache with F14 SIMD hash probing.
template <typename Key, typename Value,
          typename Hash = default_hash<Key>,
          typename KeyEqual = std::equal_to<Key>>
using f14_two_q = unified_cache<f14_two_q_trait<single_threaded_policy>, Key, Value, Hash, KeyEqual>;

/// Single-threaded FIFO cache with F14 SIMD hash probing.
template <typename Key, typename Value,
          typename Hash = default_hash<Key>,
          typename KeyEqual = std::equal_to<Key>>
using f14_fifo_cache = unified_cache<f14_fifo_trait<single_threaded_policy>, Key, Value, Hash, KeyEqual>;

/// Production-optimized LRU cache with F14 SIMD hash probing.
/// Combines F14-sharded MM + striped thread-safe policy for maximum
/// read throughput under high concurrency with SIMD-accelerated lookups.
template <typename Key, typename Value,
          typename Hash = default_hash<Key>,
          typename KeyEqual = std::equal_to<Key>>
using f14_production_cache = unified_cache<f14_production_sharded_lru_trait<striped_thread_safe_policy<>>, Key, Value, Hash, KeyEqual>;

// ============================================================================
// Segmented hash table convenience aliases (per-segment rehash, no global stall)
// ============================================================================
//
// These aliases use segmented_concurrent_hash_table (64 segments) instead of
// a single concurrent_hash_table. Rehash only locks one segment, not the
// entire table, eliminating the global stall during hash table expansion.
// Each segment has its own bucket array, locks, and rehash state.

/// Single-threaded LRU cache with segmented hash table (per-segment rehash).
/// Usage: lru::segmented_cache<int, std::string> c(100);
template <typename Key, typename Value,
          typename Hash = default_hash<Key>,
          typename KeyEqual = std::equal_to<Key>>
using segmented_cache = unified_cache<segmented_lru_trait<single_threaded_policy>, Key, Value, Hash, KeyEqual>;

/// Thread-safe LRU cache with segmented hash table.
/// Usage: lru::segmented_safe_cache<int, std::string> c(100);
template <typename Key, typename Value,
          typename Hash = default_hash<Key>,
          typename KeyEqual = std::equal_to<Key>>
using segmented_safe_cache = unified_cache<segmented_lru_trait<thread_safe_policy>, Key, Value, Hash, KeyEqual>;

/// Striped thread-safe LRU cache with segmented hash table.
/// Combines sharded MM + striped locking + segmented hash table for maximum
/// concurrency under rehash.
template <typename Key, typename Value,
          typename Hash = default_hash<Key>,
          typename KeyEqual = std::equal_to<Key>>
using segmented_striped_cache = unified_cache<segmented_sharded_lru_trait<striped_thread_safe_policy<>>, Key, Value, Hash, KeyEqual>;

/// Production-ready LRU cache for high-concurrency read-heavy workloads.
/// Combines segmented hash table (per-segment rehash, no global stall),
/// sharded MM LRU (independent shard locks), and striped thread-safe policy
/// (64-stripe distributed_shared_mutex) for maximum read throughput.
///
/// This is the recommended alias for production deployment. It provides:
///   - Per-segment rehash: rehash only locks one segment (1/64 of the table),
///     not the entire table, eliminating global stalls during hash table growth.
///   - Sharded LRU: different shards can be accessed concurrently.
///   - Striped locking: 64-stripe mutex allows concurrent reads to different keys.
///   - Lock-free read path: get() uses find_and_pin() with optimistic read fallback.
///
/// R2: EmbeddedChain enforcement. production_cache uses EmbeddedChain=true
/// (compile-time asserted in mm.hpp). This is a hard requirement for
/// lock-free reads — non-EmbeddedChain mode degrades to shared_lock reads
/// (use-after-free prevention), killing throughput under 32+ thread read
/// contention. DO NOT attempt to bypass this via custom traits.
///
/// Usage: lru::production_cache<int, std::string> cache(1000000);
template <typename Key, typename Value,
          typename Hash = default_hash<Key>,
          typename KeyEqual = std::equal_to<Key>>
using production_cache = unified_cache<production_sharded_lru_trait<striped_thread_safe_policy<>>, Key, Value, Hash, KeyEqual>;

/// T-G16: Production cache with slab allocator enabled.
///
/// Same as production_cache but additionally enables the slab allocator
/// for item allocation. Reduces memory overhead for caches with many
/// small items (10M+) by ~30% vs. new/delete, plus better cache locality.
///
/// Use this when:
///   - Cache holds >= 100K items (slab overhead is amortized).
///   - Value sizes are bounded and similar (slab size-class efficiency).
///   - Memory savings matter (10M items × 30% = significant).
///
/// For smaller or variable-size workloads, prefer production_cache.
///
/// Usage: lru::production_cache_with_slab<int, std::string> cache(1000000);
template <typename Key, typename Value,
          typename Hash = default_hash<Key>,
          typename KeyEqual = std::equal_to<Key>>
using production_cache_with_slab = unified_cache<production_with_slab_sharded_lru_trait<striped_thread_safe_policy<>>, Key, Value, Hash, KeyEqual>;

// ============================================================================
// Read-heavy convenience aliases
// ============================================================================
//
// R-3: These aliases now use the dedicated `read_heavy_*_trait` types
// (defined in cache_trait.hpp), which opt in to:
//   - defer_promotion=true (TLS-batched LRU promotion, no write-lock on hit)
//   - auto_enable_ebr=true (LRU/sharded variants only — faster read path
//     than hazptr under sustained read contention)
//   - incremental_rehash=true (no global write stall during expansion)
//   - reader_preferred fairness (max read throughput)
//
// Direct construction is now equivalent to the historical
// `make_read_heavy_*` factory output, and the factory functions have
// been removed (R-4).
//
// Usage:
//   lru::read_heavy_cache<int, std::string> cache(100000);
//   lru::read_heavy_striped_cache<int, std::string> striped(100000);
//   lru::read_heavy_w_tiny_lfu<int, std::string> wtiny(100000);

/// Thread-safe LRU cache pre-configured for read-heavy workloads.
/// Uses read_heavy_lru_trait<thread_safe_policy>:
/// defer_promotion + EBR + incremental rehash + reader_preferred fairness.
/// get() always defers LRU promotion to the TLS ring, eliminating
/// write-lock CAS on the hot read path.
template <typename Key, typename Value,
          typename Hash = default_hash<Key>,
          typename KeyEqual = std::equal_to<Key>>
using read_heavy_cache = unified_cache<read_heavy_lru_trait<thread_safe_policy>, Key, Value, Hash, KeyEqual>;

/// Striped thread-safe LRU cache pre-configured for read-heavy workloads.
/// Uses read_heavy_sharded_lru_trait<striped_thread_safe_policy<>>:
/// sharded MM + striped locking + defer_promotion + EBR + incremental rehash.
template <typename Key, typename Value,
          typename Hash = default_hash<Key>,
          typename KeyEqual = std::equal_to<Key>>
using read_heavy_striped_cache = unified_cache<read_heavy_sharded_lru_trait<striped_thread_safe_policy<>>, Key, Value, Hash, KeyEqual>;

/// Thread-safe W-TinyLFU cache pre-configured for read-heavy workloads.
/// Uses read_heavy_w_tiny_lfu_trait<thread_safe_policy>:
/// defer_promotion + incremental rehash + reader_preferred fairness.
/// EBR is not enabled (mm_wtiny_lfu does not support EBR).
template <typename Key, typename Value,
          typename Hash = default_hash<Key>,
          typename KeyEqual = std::equal_to<Key>>
using read_heavy_w_tiny_lfu = unified_cache<read_heavy_w_tiny_lfu_trait<thread_safe_policy>, Key, Value, Hash, KeyEqual>;

// ============================================================================
// R-1 / R-4: Removed factory functions
// ============================================================================
//
// The following factory functions have been removed because their behavior
// is now provided by the corresponding type aliases via trait-layer opt-in:
//
//   make_safe_cache                  → safe_cache                (via safe_lru_trait)
//   make_striped_cache               → striped_cache             (via safe_sharded_lru_trait)
//   make_read_heavy_cache            → read_heavy_cache          (via read_heavy_lru_trait)
//   make_read_heavy_striped_cache    → read_heavy_striped_cache  (via read_heavy_sharded_lru_trait)
//   make_read_heavy_w_tiny_lfu       → read_heavy_w_tiny_lfu     (via read_heavy_w_tiny_lfu_trait)
//
// Historical rationale (preserved for context):
//   - T-P2-7: All thread-safe variants default to incremental rehash to
//     avoid blocking all writers during hash table expansion.
//   - T-P1-1 (R-1): All thread-safe factories applied `reader_preferred`
//     fairness for read-heavy-write-light workloads.
//   - T-P1-4 (R-6): `make_read_heavy_*` factories enabled EBR (Epoch-Based
//     Reclamation) for faster read-path under sustained read contention.
//
// Direct construction is now the only way to obtain these caches:
//   lru::safe_cache<int, std::string> c(10000);
//   lru::read_heavy_striped_cache<int, std::string> rh(100000);

/// Single-threaded 2Q cache with segmented hash table.
template <typename Key, typename Value,
          typename Hash = default_hash<Key>,
          typename KeyEqual = std::equal_to<Key>>
using segmented_two_q = unified_cache<segmented_two_q_trait<single_threaded_policy>, Key, Value, Hash, KeyEqual>;

/// Single-threaded TinyLFU cache with segmented hash table.
template <typename Key, typename Value,
          typename Hash = default_hash<Key>,
          typename KeyEqual = std::equal_to<Key>>
using segmented_lfu_cache = unified_cache<segmented_tiny_lfu_trait<single_threaded_policy>, Key, Value, Hash, KeyEqual>;

/// Single-threaded W-TinyLFU cache with segmented hash table.
template <typename Key, typename Value,
          typename Hash = default_hash<Key>,
          typename KeyEqual = std::equal_to<Key>>
using segmented_w_tiny_lfu = unified_cache<segmented_w_tiny_lfu_trait<single_threaded_policy>, Key, Value, Hash, KeyEqual>;

/// Single-threaded FIFO cache with segmented hash table.
template <typename Key, typename Value,
          typename Hash = default_hash<Key>,
          typename KeyEqual = std::equal_to<Key>>
using segmented_fifo_cache = unified_cache<segmented_fifo_trait<single_threaded_policy>, Key, Value, Hash, KeyEqual>;

// ============================================================================
// Compact cache convenience aliases
// ============================================================================

/// Single-threaded compact cache for small key-value pairs.
/// 用法：lru::compact_cache<int, int> c(10000);
template <typename Key, typename Value,
          typename Hash = default_hash<Key>,
          typename KeyEqual = std::equal_to<Key>,
          std::size_t kMaxItemSize = 64>
using compact_cache_default = compact_cache<Key, Value, Hash, KeyEqual, kMaxItemSize>;

/// Thread-safe compact cache — alias defined in compact_cache.hpp.
/// Includes kSlotAlignment parameter for full customization.
/// 用法：lru::safe_compact_cache<int, int> c(10000);

/// T14: Striped thread-safe compact cache — alias defined in compact_cache.hpp.
/// API-symmetric with lru::striped_cache for code that swaps between
/// unified_cache and compact_cache behind a common interface.
/// 用法：lru::striped_compact_cache<int, int> c(10000);

// ============================================================================
// T15: Compressed pointer cache convenience aliases
// ============================================================================
//
// compressed_cache uses the compressed_lru_trait, which sets
// Hook = compressed_intrusive_hook on the cache_trait. The trait exposes
// `uses_compressed_hook` so downstream code (allocators, diagnostics)
// can detect the intended hook type. Note: current MM strategies hardcode
// `intrusive_hook` internally; full propagation of the Hook parameter
// to mm_lru is a future refactor — the savings documented here are
// hook-level only until that lands.
//
// Example:
//   lru::compressed_cache<int, std::string> c(1'000'000);

/// Single-threaded LRU cache with compressed_intrusive_hook trait.
template <typename Key, typename Value,
          typename Hash = default_hash<Key>,
          typename KeyEqual = std::equal_to<Key>>
using compressed_cache = unified_cache<compressed_lru_trait<single_threaded_policy>, Key, Value, Hash, KeyEqual>;

/// Thread-safe LRU cache with compressed_intrusive_hook trait.
template <typename Key, typename Value,
          typename Hash = default_hash<Key>,
          typename KeyEqual = std::equal_to<Key>>
using safe_compressed_cache = unified_cache<compressed_lru_trait<thread_safe_policy>, Key, Value, Hash, KeyEqual>;

/// Striped thread-safe LRU cache with compressed_intrusive_hook trait.
/// Uses sharded_mm_lru (required by striped_thread_safe_policy).
template <typename Key, typename Value,
          typename Hash = default_hash<Key>,
          typename KeyEqual = std::equal_to<Key>>
using striped_compressed_cache = unified_cache<compressed_sharded_lru_trait<striped_thread_safe_policy<>>, Key, Value, Hash, KeyEqual>;

// ============================================================================
// T14.2: Unified compact cache aliases
// ============================================================================
//
// These aliases instantiate unified_cache with the compact_unified_*_trait,
// which sets `is_compact = true`. The resulting cache embeds a compact_cache
// member that uses compact_slot_allocator (dense fixed-size slot storage)
// instead of the regular mm_lru path, achieving 50-70% memory savings for
// small items (sizeof(K)+sizeof(V) <= 64).
//
// The API is identical to unified_cache (set/get/peek/try_get/bulk_get/
// remove/contains/size/empty/flush/shutdown), with these caveats:
//   - get() returns a non-pinning read_handle (no refcount on compact slots)
//   - TTL, slab allocator, and hazptr/EBR reclamation are not available
//   - A static_assert enforces sizeof(K)+sizeof(V) <= 64 at compile time
//
// Example:
//   lru::unified_compact_cache<int, int> c(10000);
//   c.set(1, 42);
//   auto h = c.get(1);
//   if (h) std::cout << *h << "\n";

/// Single-threaded unified compact LRU cache.
template <typename Key, typename Value,
          typename Hash = default_hash<Key>,
          typename KeyEqual = std::equal_to<Key>>
using unified_compact_cache = unified_cache<compact_unified_lru_trait<single_threaded_policy>, Key, Value, Hash, KeyEqual>;

/// Thread-safe unified compact LRU cache.
template <typename Key, typename Value,
          typename Hash = default_hash<Key>,
          typename KeyEqual = std::equal_to<Key>>
using safe_unified_compact_cache = unified_cache<compact_unified_lru_trait<thread_safe_policy>, Key, Value, Hash, KeyEqual>;

/// Striped thread-safe unified compact LRU cache.
template <typename Key, typename Value,
          typename Hash = default_hash<Key>,
          typename KeyEqual = std::equal_to<Key>>
using striped_unified_compact_cache = unified_cache<compact_unified_sharded_lru_trait<striped_thread_safe_policy<>>, Key, Value, Hash, KeyEqual>;

// ============================================================================
// Out-of-line definitions for unified_cache::save() / load()
// Must appear after serialization.hpp is included so that the
// free functions serialize() / deserialize() are visible.
// ============================================================================

namespace detail {

/// Helper: determine the mm_type_id for a given MM type at compile time.
template <typename MM> struct mm_type_id_for;
template <typename K, typename V, typename H, typename E, typename P, bool S>
struct mm_type_id_for<mm_lru<K, V, H, E, P, S>> {
    static constexpr mm_type_id value = mm_type_id::lru;
};
template <typename K, typename V, typename H, typename E, typename P, bool S>
struct mm_type_id_for<mm_2q<K, V, H, E, P, S>> {
    static constexpr mm_type_id value = mm_type_id::two_q;
};
template <typename K, typename V, typename H, typename E, typename P, bool S>
struct mm_type_id_for<mm_tiny_lfu<K, V, H, E, P, S>> {
    static constexpr mm_type_id value = mm_type_id::tiny_lfu;
};
template <typename K, typename V, typename H, typename E, typename P, bool S>
struct mm_type_id_for<mm_wtiny_lfu<K, V, H, E, P, S>> {
    static constexpr mm_type_id value = mm_type_id::w_tiny_lfu;
};
template <typename K, typename V, typename H, typename E, typename P, bool S>
struct mm_type_id_for<mm_fifo<K, V, H, E, P, S>> {
    static constexpr mm_type_id value = mm_type_id::fifo;
};
template <typename K, typename V, typename H, typename E, typename P, bool S>
struct mm_type_id_for<sharded_mm_lru<K, V, H, E, P, S>> {
    static constexpr mm_type_id value = mm_type_id::sharded_lru;
};

template <typename MM>
inline constexpr mm_type_id mm_type_id_v = mm_type_id_for<MM>::value;

/// Collect a cache_snapshot from an MM object (to be called under read lock).
template <typename MM>
auto collect_snapshot(const MM& mm) {
    using Key = typename MM::key_type;
    using Value = typename MM::mapped_type;
    constexpr auto type_id = mm_type_id_v<MM>;

    cache_snapshot<Key, Value> snap;
    snap.type = type_id;

    // Collect items (in iteration order = MRU→LRU / multi-queue order)
    snap.items.reserve(mm.size());
    for (auto it = mm.begin(); it != mm.end(); ++it) {
        serialized_item<Key, Value> item;
        item.key = it->key;
        item.value = it->value;
        item.update_time = it->hook.update_time;
        item.flags = it->hook.flags;
        item.queue_id = it->queue_id;
        snap.items.push_back(std::move(item));
    }

    // Collect config and extra state per MM type
    if constexpr (type_id == mm_type_id::lru) {
        const auto& c = mm.config();
        snap.config.lru_refresh_time = mm.refresh_time();
        snap.config.lru_insertion_point_spec = c.lru_insertion_point_spec;
        snap.config.update_on_read = c.update_on_read;
        snap.config.update_on_write = c.update_on_write;
        snap.config.try_lock_update = c.try_lock_update;
        snap.config.lru_refresh_ratio = c.lru_refresh_ratio;
        snap.config.mm_reconfigure_interval_secs = c.mm_reconfigure_interval_secs;
        auto ins_pos = mm.insertion_point_position();
        if (ins_pos != MM::npos) {
            snap.list_state.insertion_point_pos = static_cast<uint32_t>(ins_pos);
            snap.list_state.tail_size = static_cast<uint32_t>(mm.tail_size());
        }
    } else if constexpr (type_id == mm_type_id::two_q) {
        const auto& c = mm.config();
        snap.config.lru_refresh_time = mm.refresh_time();
        snap.config.update_on_read = c.update_on_read;
        snap.config.update_on_write = c.update_on_write;
        snap.config.try_lock_update = c.try_lock_update;
        snap.config.lru_refresh_ratio = c.lru_refresh_ratio;
        snap.config.mm_reconfigure_interval_secs = c.mm_reconfigure_interval_secs;
        snap.config.hot_ratio = c.hot_ratio;
        snap.config.warm_ratio = c.warm_ratio;
        snap.config.rebalance_on_record_access = c.rebalance_on_record_access;
    } else if constexpr (type_id == mm_type_id::fifo) {
        // FIFO has minimal config
    } else if constexpr (type_id == mm_type_id::tiny_lfu) {
        const auto& c = mm.config();
        snap.config.lru_refresh_time = mm.refresh_time();
        snap.config.lru_refresh_ratio = c.lru_refresh_ratio;
        snap.config.cms_error_rate = c.cms_error_rate;
        snap.config.try_lock_update = c.try_lock_update;
        auto cms_words = static_cast<uint32_t>(mm.sketch().serialized_state_words());
        snap.cms_state.resize(cms_words);
        mm.sketch().save_state(snap.cms_state.begin());
    } else if constexpr (type_id == mm_type_id::w_tiny_lfu) {
        const auto& c = mm.config();
        snap.config.lru_refresh_time = mm.refresh_time();
        snap.config.lru_refresh_ratio = c.lru_refresh_ratio;
        snap.config.cms_error_rate = c.cms_error_rate;
        snap.config.try_lock_update = c.try_lock_update;
        auto cms_words = static_cast<uint32_t>(mm.sketch().serialized_state_words());
        snap.cms_state.resize(cms_words);
        mm.sketch().save_state(snap.cms_state.begin());
    }

    return snap;
}

/// Collect a snapshot from a sharded_mm_lru (per-shard snapshots).
template <typename MM>
auto collect_sharded_snapshot(const MM& mm) {
    using Key = typename MM::key_type;
    using Value = typename MM::mapped_type;
    using shard_type = typename MM::shard_type;

    // Collect per-shard serialized data
    std::vector<std::vector<uint8_t>> shard_data;
    shard_data.reserve(mm.num_shards());

    std::size_t total_size = kV5HeaderSize;
    for (std::size_t i = 0; i < mm.num_shards(); ++i) {
        auto snap = collect_snapshot(static_cast<const shard_type&>(mm.shard(i)));
        auto data = serialize_from_snapshot(snap);
        total_size += 4; // shard_data_length
        total_size += data.size();
        shard_data.push_back(std::move(data));
    }

    // Build the sharded header + concatenated shard data
    binary_writer w;
    w.reserve(total_size);

    w.write(kSerializationMagic);
    w.write(kSerializationVersion);
    w.write(static_cast<uint32_t>(mm.num_shards()));
    w.write(static_cast<uint32_t>(mm_type_id::sharded_lru));
    w.write(kV5HeaderSize);
    w.write(static_cast<uint32_t>(0)); // flags
    w.write(static_cast<uint64_t>(0)); // feature_flags
    auto checksum_offset = w.size();
    w.write(static_cast<uint32_t>(0)); // checksum placeholder
    for (auto& data : shard_data) {
        w.write(static_cast<uint32_t>(data.size()));
        w.write_bytes(data.data(), data.size());
    }

    uint32_t checksum = detail::crc32(
        w.data().data() + checksum_offset + sizeof(uint32_t),
        w.size() - checksum_offset - sizeof(uint32_t));
    w.patch_at(checksum_offset, checksum);

    return w.release();
}

/// Rebuild an MM object from parsed_deserialization_data (to be called under
/// write lock). The parsed data was obtained from parse_serialized_data()
/// without any lock.
template <typename MM>
void rebuild_from_parsed(MM& mm, const parsed_deserialization_data<typename MM::key_type, typename MM::mapped_type>& parsed) {
    using Key = typename MM::key_type;
    using Value = typename MM::mapped_type;
    constexpr auto type_id = mm_type_id_v<MM>;

    mm.flush();

    if constexpr (type_id == mm_type_id::lru) {
        auto new_cfg = mm.config();
        new_cfg.lru_insertion_point_spec = parsed.config.lru_insertion_point_spec;
        new_cfg.update_on_read = parsed.config.update_on_read;
        new_cfg.update_on_write = parsed.config.update_on_write;
        new_cfg.try_lock_update = parsed.config.try_lock_update;
        new_cfg.lru_refresh_ratio = parsed.config.lru_refresh_ratio;
        new_cfg.mm_reconfigure_interval_secs = parsed.config.mm_reconfigure_interval_secs;
        new_cfg.default_lru_refresh_time = parsed.config.lru_refresh_time;

        uint32_t ins_pos = parsed.list_state.insertion_point_pos;
        uint32_t tail_sz = parsed.list_state.tail_size;

        mm.rebuild_from_serialized(parsed.items.begin(), parsed.items.end(), ins_pos, tail_sz);
        mm.set_config(new_cfg);
    } else if constexpr (type_id == mm_type_id::two_q) {
        auto new_cfg = mm.config();
        new_cfg.update_on_read = parsed.config.update_on_read;
        new_cfg.update_on_write = parsed.config.update_on_write;
        new_cfg.try_lock_update = parsed.config.try_lock_update;
        new_cfg.lru_refresh_ratio = parsed.config.lru_refresh_ratio;
        new_cfg.mm_reconfigure_interval_secs = parsed.config.mm_reconfigure_interval_secs;
        new_cfg.hot_ratio = parsed.config.hot_ratio;
        new_cfg.warm_ratio = parsed.config.warm_ratio;
        new_cfg.rebalance_on_record_access = parsed.config.rebalance_on_record_access;
        new_cfg.default_lru_refresh_time = parsed.config.lru_refresh_time;
        mm.set_config(new_cfg);

        mm.rebuild_from_serialized(parsed.items.begin(), parsed.items.end());
    } else if constexpr (type_id == mm_type_id::fifo) {
        for (const auto& item : parsed.items) {
            mm.set(item.key, std::move(item.value));
        }
    } else if constexpr (type_id == mm_type_id::tiny_lfu) {
        auto new_cfg = mm.config();
        new_cfg.lru_refresh_ratio = parsed.config.lru_refresh_ratio;
        new_cfg.cms_error_rate = parsed.config.cms_error_rate;
        new_cfg.try_lock_update = parsed.config.try_lock_update;
        new_cfg.default_lru_refresh_time = parsed.config.lru_refresh_time;
        mm.set_config(new_cfg);

        if (!parsed.cms_state.empty()) {
            auto it = parsed.cms_state.begin();
            mm.sketch_mut().load_state(it);
        }

        mm.rebuild_from_serialized(parsed.items.begin(), parsed.items.end());
    } else if constexpr (type_id == mm_type_id::w_tiny_lfu) {
        auto new_cfg = mm.config();
        new_cfg.lru_refresh_ratio = parsed.config.lru_refresh_ratio;
        new_cfg.cms_error_rate = parsed.config.cms_error_rate;
        new_cfg.try_lock_update = parsed.config.try_lock_update;
        new_cfg.default_lru_refresh_time = parsed.config.lru_refresh_time;
        mm.set_config(new_cfg);

        if (!parsed.cms_state.empty()) {
            auto it = parsed.cms_state.begin();
            mm.sketch_mut().load_state(it);
        }

        mm.rebuild_from_serialized(parsed.items.begin(), parsed.items.end());
    }
}

/// Parse sharded LRU binary data into per-shard parsed data.
/// Only v5 format is supported.
template <typename Key, typename Value>
std::vector<parsed_deserialization_data<Key, Value>>
parse_sharded_serialized_data(std::span<const uint8_t> data) {
    detail::binary_reader r(data.data(), data.size());

    auto magic = r.read<uint32_t>();
    if (magic != kSerializationMagic) {
        throw std::runtime_error("sharded_mm_lru deserialize: invalid magic number");
    }
    auto version = r.read<uint32_t>();
    if (version != kSerializationVersion) {
        throw std::runtime_error("sharded_mm_lru deserialize: unsupported version " +
            std::to_string(version) +
            " (only version " + std::to_string(kSerializationVersion) + " is supported)");
    }
    auto num_shards = r.read<uint32_t>();
    [[maybe_unused]] auto mm_type = r.read<uint32_t>(); // mm_type_id (sharded_lru)
    auto header_size = r.read<uint32_t>();
    if (header_size < kV5HeaderSize || header_size > data.size()) {
        throw std::runtime_error("sharded_mm_lru deserialize: invalid header_size");
    }
    [[maybe_unused]] auto flags = r.read<uint32_t>();

    // Read feature_flags (v5 header)
    [[maybe_unused]] auto feature_flags = r.read<uint64_t>();

    auto stored_checksum = r.read<uint32_t>();
    std::size_t payload_offset = header_size;
    uint32_t computed_checksum = detail::crc32(
        data.data() + payload_offset,
        data.size() - payload_offset);
    if (computed_checksum != stored_checksum) {
        throw std::runtime_error("sharded_mm_lru deserialize: header checksum mismatch");
    }

    std::vector<parsed_deserialization_data<Key, Value>> result;
    result.reserve(num_shards);
    for (uint32_t i = 0; i < num_shards; ++i) {
        auto shard_span = r.read_span();
        auto parsed = detail::parse_serialized_data<Key, Value>(
            mm_type_id::lru, shard_span);
        result.push_back(std::move(parsed));
    }
    return result;
}

/// Rebuild a sharded_mm_lru from per-shard parsed data (under write lock).
template <typename MM>
void rebuild_sharded_from_parsed(
    MM& mm,
    const std::vector<parsed_deserialization_data<typename MM::key_type, typename MM::mapped_type>>& shard_parsed)
{
    using shard_type = typename MM::shard_type;
    if (shard_parsed.size() != mm.num_shards()) {
        throw std::runtime_error("deserialization: num_shards mismatch");
    }
    for (std::size_t i = 0; i < mm.num_shards(); ++i) {
        rebuild_from_parsed(static_cast<shard_type&>(mm.shard(i)), shard_parsed[i]);
    }
}

} // namespace detail

template <typename Trait, typename Key, typename Value,
          typename Hash, typename KeyEqual>
std::vector<uint8_t>
unified_cache<Trait, Key, Value, Hash, KeyEqual>::save() {
    // COW serialization: collect a snapshot under a brief read lock, then
    // serialize without holding any lock. This minimizes lock contention
    // for large caches where serialization may take seconds.
    //
    // The snapshot may not reflect the most recent state (concurrent writes
    // after the lock is released are not captured), but this is acceptable
    // for serialization — it provides a consistent point-in-time view.
    if constexpr (is_striped) {
        // C-2 fix: Delegate to save_per_shard() which acquires per-shard read
        // locks one at a time (max 1 held at any moment), avoiding the
        // deadlock with the drain worker. The drain worker's promote_keys()
        // can hold a shard write lock (via try_lock) and reenter the cache
        // via user callbacks (on_hit/on_insert/on_evict) that hash to a
        // different shard — if save() holds all 64 read locks simultaneously,
        // a deadlock cycle forms: save() holds shard J read lock, waits for
        // shard K read lock (drain worker holds K write lock); drain worker
        // holds K write lock, waits for shard J lock (via callback reentry,
        // blocked by save()'s read lock). save_per_shard() breaks this cycle
        // by never holding more than one shard lock at a time. The trade-off
        // is a non-atomic cross-shard snapshot (shard 0 may reflect T0 state,
        // shard 63 reflects T0+Δ), which is acceptable for serialization.
        // save_per_shard() produces the same sharded binary format.
        return save_per_shard();
    } else {
        // Phase 1: snapshot under brief read lock
        auto snap = [this]() {
            auto lock = acquire_read_lock();
            return detail::collect_snapshot(mm_);
        }();
        // Phase 2: serialize without lock
        return detail::serialize_from_snapshot(snap);
    }
}

template <typename Trait, typename Key, typename Value,
          typename Hash, typename KeyEqual>
void
unified_cache<Trait, Key, Value, Hash, KeyEqual>::load(
        std::span<const uint8_t> data) {
    // Two-phase deserialization to minimize write lock hold time:
    //   Phase 1 (no lock): parse binary data into pre-validated structures
    //   Phase 2 (write lock): flush + rebuild from parsed data
    //
    // The parsing phase (header validation, checksum verification, item
    // deserialization) is the expensive part; rebuilding is O(n) memory
    // operations that are much faster than parsing.
    if constexpr (is_striped) {
        // Sharded: parse all shard data without lock, then rebuild under lock
        auto shard_parsed = detail::parse_sharded_serialized_data<Key, Value>(data);
        flush_guard fg{mm_};
        // P-CRIT-2 (T-C2): For sharded_mm_lru, acquire per-shard write locks
        // (the stripe lock does NOT protect MM data — see T3.4 bugfix).
        auto lock = acquire_write_lock_all_shards();
        detail::rebuild_sharded_from_parsed(mm_, shard_parsed);
        maybe_report_memory_to_monitor();
    } else {
        constexpr auto type_id = detail::mm_type_id_v<mm_type>;
        // Phase 1: parse without lock (heavy work: validation + deserialization)
        auto parsed = detail::parse_serialized_data<Key, Value>(type_id, data);
        // Phase 2: brief write lock for cache mutation (fast: flush + rebuild)
        flush_guard fg{mm_};
        auto lock = acquire_write_lock();
        detail::rebuild_from_parsed(mm_, parsed);
        maybe_report_memory_to_monitor();
    }
}

// ============================================================================
// Task 10: Per-shard serialization with per-shard locking
// ============================================================================

template <typename Trait, typename Key, typename Value,
          typename Hash, typename KeyEqual>
std::vector<uint8_t>
unified_cache<Trait, Key, Value, Hash, KeyEqual>::save_per_shard() {
    if constexpr (!is_striped) {
        // Non-sharded: fall back to regular save().
        return save();
    } else {
        // Per-shard: lock each shard in shared mode, snapshot, release.
        // This allows writes to other shards to proceed.
        //
        // P-CRIT-2 (T-C2): For sharded_mm_lru, acquire the per-shard read
        // lock (NOT the stripe lock — stripe lock does not protect MM data,
        // see T3.4 bugfix). For non-sharded MM with striped locking,
        // acquire_read_lock_for_shard() falls back to the stripe lock.
        using shard_type = typename mm_type::shard_type;

        std::vector<std::vector<uint8_t>> shard_data;
        shard_data.reserve(mm_.num_shards());

        for (std::size_t i = 0; i < mm_.num_shards(); ++i) {
            // Acquire only this shard's read lock (per-shard or stripe).
            auto lock = acquire_read_lock_for_shard(i);
            auto snap = detail::collect_snapshot(
                static_cast<const shard_type&>(mm_.shard(i)));
            shard_data.push_back(detail::serialize_from_snapshot(snap));
        }

        // Build the sharded header + concatenated shard data (no lock needed).
        detail::binary_writer w;
        w.reserve(detail::kV5HeaderSize);
        w.write(detail::kSerializationMagic);
        w.write(detail::kSerializationVersion);
        w.write(static_cast<uint32_t>(mm_.num_shards()));
        w.write(static_cast<uint32_t>(detail::mm_type_id::sharded_lru));
        w.write(detail::kV5HeaderSize);
        w.write(static_cast<uint32_t>(0)); // flags
        w.write(static_cast<uint64_t>(0)); // feature_flags
        auto checksum_offset = w.size();
        w.write(static_cast<uint32_t>(0)); // checksum placeholder
        for (auto& data : shard_data) {
            w.write(static_cast<uint32_t>(data.size()));
            w.write_bytes(data.data(), data.size());
        }

        uint32_t checksum = detail::crc32(
            w.data().data() + checksum_offset + sizeof(uint32_t),
            w.size() - checksum_offset - sizeof(uint32_t));
        w.patch_at(checksum_offset, checksum);

        return w.release();
    }
}

template <typename Trait, typename Key, typename Value,
          typename Hash, typename KeyEqual>
void
unified_cache<Trait, Key, Value, Hash, KeyEqual>::load_per_shard(
        std::span<const uint8_t> data) {
    if constexpr (!is_striped) {
        // Non-sharded: fall back to regular load().
        load(data);
    } else {
        // Phase 1 (no lock): parse all shard data.
        auto shard_parsed = detail::parse_sharded_serialized_data<Key, Value>(data);
        if (shard_parsed.size() != mm_.num_shards()) {
            throw std::runtime_error("load_per_shard: num_shards mismatch");
        }
        // Phase 2: rebuild each shard under only that shard's write lock.
        // P-CRIT-2 (T-C2): For sharded_mm_lru, acquire the per-shard write
        // lock (NOT the stripe lock — see T3.4 bugfix). For non-sharded MM
        // with striped locking, acquire_write_lock_for_shard() falls back
        // to the stripe lock.
        using shard_type = typename mm_type::shard_type;
        for (std::size_t i = 0; i < mm_.num_shards(); ++i) {
            auto lock = acquire_write_lock_for_shard(i);
            detail::rebuild_from_parsed(
                static_cast<shard_type&>(mm_.shard(i)), shard_parsed[i]);
        }
        maybe_report_memory_to_monitor();
    }
}

// ============================================================================
// P2-3: save_atomic — strictly-atomic cross-shard snapshot
// ============================================================================

template <typename Trait, typename Key, typename Value,
          typename Hash, typename KeyEqual>
template <typename Rep, typename Period>
std::vector<uint8_t>
unified_cache<Trait, Key, Value, Hash, KeyEqual>::save_atomic(
        std::chrono::duration<Rep, Period> handle_drain_timeout) {
    if constexpr (!is_striped) {
        // Non-sharded: save() already holds a single global read lock for
        // the entire serialization, which provides cross-shard atomicity
        // trivially (there is only one shard). The shutdown+drain dance
        // would add cost without strengthening the guarantee.
        return save();
    } else {
        // P2-3: Strictly-atomic cross-shard snapshot.
        //
        // 1. shutdown_and_wait() — rejects new operations and waits for
        //    outstanding read_handles to be released. Without this, a
        //    handle-holding thread could observe a half-serialized state.
        //    If the timeout elapses with handles still active, abort: the
        //    snapshot would not be safely reproducible and continuing
        //    could deadlock if the handle holder is blocked on this
        //    thread.
        const bool drained = shutdown_and_wait(handle_drain_timeout);
        if (!drained) {
            throw std::runtime_error(
                "save_atomic: timeout waiting for active read_handles "
                "to drain. Inspect active_handle_count() before calling "
                "save_atomic(), or increase handle_drain_timeout.");
        }

        // 2. Acquire all per-shard read locks simultaneously. Combined
        //    with shutdown (which rejects new writes and has drained all
        //    TLS rings), no writer can mutate any shard — the snapshot
        //    reflects a single instant T0 across all 64 shards.
        using shard_type = typename mm_type::shard_type;
        auto locks = acquire_read_lock_all_shards();

        // 3. Serialize every shard under the locks. Same binary format
        //    as save_per_shard() so the result can be loaded via
        //    load_per_shard() on a fresh cache.
        std::vector<std::vector<uint8_t>> shard_data;
        shard_data.reserve(mm_.num_shards());
        for (std::size_t i = 0; i < mm_.num_shards(); ++i) {
            auto snap = detail::collect_snapshot(
                static_cast<const shard_type&>(mm_.shard(i)));
            shard_data.push_back(detail::serialize_from_snapshot(snap));
        }

        // 4. Build the sharded header + concatenated shard data. Locks
        //    are released when `locks` goes out of scope below.
        detail::binary_writer w;
        w.reserve(detail::kV5HeaderSize);
        w.write(detail::kSerializationMagic);
        w.write(detail::kSerializationVersion);
        w.write(static_cast<uint32_t>(mm_.num_shards()));
        w.write(static_cast<uint32_t>(detail::mm_type_id::sharded_lru));
        w.write(detail::kV5HeaderSize);
        w.write(static_cast<uint32_t>(0)); // flags
        w.write(static_cast<uint64_t>(0)); // feature_flags
        auto checksum_offset = w.size();
        w.write(static_cast<uint32_t>(0)); // checksum placeholder
        for (auto& data : shard_data) {
            w.write(static_cast<uint32_t>(data.size()));
            w.write_bytes(data.data(), data.size());
        }

        uint32_t checksum = detail::crc32(
            w.data().data() + checksum_offset + sizeof(uint32_t),
            w.size() - checksum_offset - sizeof(uint32_t));
        w.patch_at(checksum_offset, checksum);

        return w.release();
    }
}

// ============================================================================
// T-G15: save_atomic_or_per_shard — atomic-or-per-shard with degradation
// ============================================================================

template <typename Trait, typename Key, typename Value,
          typename Hash, typename KeyEqual>
template <typename Rep, typename Period>
typename unified_cache<Trait, Key, Value, Hash, KeyEqual>::save_atomic_result
unified_cache<Trait, Key, Value, Hash, KeyEqual>::save_atomic_or_per_shard(
        std::chrono::duration<Rep, Period> handle_drain_timeout) {
    save_atomic_result result;
    try {
        result.data = save_atomic(handle_drain_timeout);
        result.atomic = true;
        return result;
    } catch (const std::runtime_error&) {
        // save_atomic timed out waiting for handle drain. Fall back to
        // save_per_shard() which does not require shutdown — the caller
        // gets a best-effort snapshot instead of an exception. The
        // fallback snapshot may reflect writes that happened between
        // shard snapshots, but is safe to load via load_per_shard().
        result.data = save_per_shard();
        result.atomic = false;
        return result;
    }
}

// ============================================================================
// TLS-Aware Cache Wrapper (requires both core.hpp and tls_ring.hpp)
// ============================================================================

/// Wraps a cache with TLS-based promotion deferral.
/// Automatically records accesses and flushes when the ring is full.
///
/// Usage:
///   lru::safe_cache<int, std::string> underlying(10000);
///   tls_cache_adapter adapter(underlying);
///   auto value = adapter.get(42); // automatically records + flushes as needed
template <typename CacheType, std::size_t RingSize = 64>
class tls_cache_adapter {
public:
    using cache_type = CacheType;
    using key_type = typename cache_type::key_type;
    using mapped_type = typename cache_type::mapped_type;
    using size_type = typename cache_type::size_type;

    static constexpr std::size_t kDefaultRingSize = RingSize;

    explicit tls_cache_adapter(cache_type& cache)
        : cache_(cache) {}

    ~tls_cache_adapter() { flush_promotions(); }

    // --------------------------------------------------------------------
    // Core API (get is wrapped with TLS ring; set/del pass through)
    // --------------------------------------------------------------------

    /// Get with automatic TLS recording + batch flush.
    read_handle<mapped_type> get(const key_type& key) {
        auto result = cache_.get(key);
        if (result) {
            // Record access in TLS ring
            if (ring_.record(key) || ring_.should_flush()) {
                flush_promotions();
            }
        }
        return result;
    }

    template <typename V>
    void set(const key_type& key, V&& value) {
        cache_.set(key, std::forward<V>(value));
    }

    bool del(const key_type& key) {
        return cache_.del(key);
    }

    // --------------------------------------------------------------------
    // Explicit flush
    // --------------------------------------------------------------------

    /// Manually flush all pending promotions.
    void flush_promotions() {
        ring_.flush_to([this](const key_type& k) {
            // Promote the item's LRU position without triggering hit statistics
            // or hit callbacks. Uses cache_.promote() which only performs
            // record_access on the underlying MM, avoiding the side effects
            // of cache_.get().
            (void)cache_.promote(k);
        });
    }

    /// Flush all pending promotions (must be called before thread exit).
    void flush() {
        flush_promotions();
    }

    // --------------------------------------------------------------------
    // Pass-through
    // --------------------------------------------------------------------

    bool empty() const { return cache_.empty(); }
    size_type size() const { return cache_.size(); }
    size_type max_size() const { return cache_.max_size(); }

    cache_type& cache() noexcept { return cache_; }
    const cache_type& cache() const noexcept { return cache_; }

    const tls_active_item_ring<key_type, RingSize>& ring() const noexcept {
        return ring_;
    }

private:
    cache_type& cache_;
    tls_active_item_ring<key_type, RingSize> ring_;
};

} // namespace lru

#endif // LRU_LRU_HPP
