// SPDX-License-Identifier: MIT
// T-G13: clone_and_reshard — stop-the-world shard-count expansion utility.
//
// Online resharding (in-place shard count change while serving traffic)
// is NOT supported because:
//   - Per-shard hash tables would need to be migrated under concurrent
//     access, requiring complex two-phase commit or stop-the-world pauses
//     anyway.
//   - The sharded_mm_lru config is immutable after construction (shard
//     count affects memory layout, hash distribution, and lock granularity).
//   - Most production workloads pick a shard count up front (64 is the
//     default and is sufficient for 32-256 core machines).
//
// This header provides `clone_and_reshard(old_cache, new_num_shards)`,
// a stop-the-world utility that:
//   1. Constructs a new cache with the same type but `new_num_shards`.
//   2. Iterates `old_cache` under read locks (LRU → MRU order).
//   3. Inserts each item into the new cache.
//   4. Returns the new cache by value (move).
//
// The caller is responsible for:
//   - Quiescing the old cache (no concurrent writes during clone).
//     The function acquires read locks, so concurrent reads are safe,
//     but concurrent writes may produce an inconsistent clone.
//   - Atomically swapping the old cache for the new one in the
//     application's cache reference (e.g. via std::shared_ptr<cache>
//     or std::atomic<cache*> indirection).
//   - Destroying the old cache after the swap.
//
// For zero-downtime expansion, pair this with warm_cache: snapshot the
// old cache via save_per_shard(), construct the new cache with the
// desired shard count, load_per_shard() the snapshot, then atomically
// swap. See warm_cache.hpp for the snapshot/restart primitives.
//
// Complexity: O(N) where N = old_cache.size(). Memory: 2× peak (old +
// new coexist during clone). For 10M items this is typically < 1s on
// modern hardware.

#ifndef LRU_RESHARD_UTILS_HPP
#define LRU_RESHARD_UTILS_HPP

#include "cache_trait.hpp"

#include <utility>

namespace lru {

/// T-G13: Clone a cache with a different shard count.
///
/// Constructs a new cache of the same type with `new_num_shards` shards,
/// copies all items from `old_cache`, and returns the new cache. The old
/// cache is not modified — the caller swaps it out after this function
/// returns.
///
/// Requirements:
///   - `Cache` must be a `unified_cache` instantiation with `is_striped`
///     (i.e. backed by sharded_mm_lru). Non-sharded caches have no shard
///     count to change.
///   - The caller MUST quiesce concurrent writes to `old_cache` before
///     calling. Concurrent reads are safe (the function acquires read
///     locks), but concurrent writes may produce an inconsistent clone.
///
/// Iteration order: LRU → MRU (least-recently-used first). This preserves
/// the LRU eviction order in the new cache — items that were close to
/// eviction in the old cache remain close to eviction in the new cache.
///
/// Memory: 2× peak during clone (old + new coexist). Destroy the old
/// cache promptly after swapping.
///
/// Example:
///   lru::production_cache<int, std::string> old_cache(1'000'000);
///   // ... fill old_cache ...
///   // Quiesce writes, then reshard:
///   auto new_cache = lru::clone_and_reshard(old_cache, 128);
///   // Swap old_cache for new_cache in your application's reference.
template <typename Cache>
auto clone_and_reshard(const Cache& old_cache,
                       typename Cache::size_type new_num_shards)
    -> Cache
{
    static_assert(Cache::trait_type::is_striped,
        "clone_and_reshard requires a striped (sharded_mm_lru) cache");

    // Construct the new cache with the same max_size but new shard count.
    // Use the 3-argument constructor (max_size, num_stripes, num_shards).
    // We set num_stripes = new_num_shards for 1:1 stripe:shard mapping
    // (the default for production_cache).
    Cache new_cache(old_cache.max_size(),
                    new_num_shards,   // num_stripes
                    new_num_shards);  // num_shards

    // Iterate old cache in LRU → MRU order under read locks. Each item
    // is inserted into the new cache. Insertion order determines the
    // initial LRU position — the last-inserted item becomes MRU.
    //
    // rbegin() returns a locked_range that bundles BOTH the begin and end
    // iterators together with the read lock. We iterate using range.begin()
    // and range.end() so the lock is held for the entire iteration.
    //
    // Note: rbegin() iteration does not promote items (peek-style), so the
    // old cache's LRU order is not mutated during the clone.
    auto range = old_cache.rbegin();
    for (auto it = range.begin(), e = range.end(); it != e; ++it) {
        new_cache.set(it->key, it->value);
    }

    return new_cache;  // NRVO/move
}

} // namespace lru

#endif // LRU_RESHARD_UTILS_HPP
