// SPDX-License-Identifier: MIT
// P2-1: max_size silent amplification warning + set_max_size_strict API.
//
// Validates:
//   1. striped_cache(max_size=10, num_stripes=64) silently amplifies
//      max_size to num_shards (64) — verified via size() capacity.
//   2. max_size_strict(N) with N < num_shards throws std::invalid_argument.
//   3. max_size_strict(N) with N >= num_shards succeeds and applies.
//   4. max_size_strict(unlimited) succeeds (no amplification check).
//   5. Non-sharded caches (safe_cache) accept any max_size via
//      max_size_strict (no amplification possible).

#include <gtest/gtest.h>

#include <stdexcept>

#include "../lru.hpp"

using namespace lru;

// ============================================================================
// TC-P2-1a: Silent amplification of undersized max_size
// ============================================================================
TEST(MaxSizeStrict, SilentlyAmplifiesUndersizedMaxSize) {
    // striped_cache(10, 64) — requested max_size=10, num_stripes=64.
    // distribute_max_size() raises max_size_ to 64 so every shard has
    // at least one slot. This is the "lenient" behavior; the warning
    // goes to stderr (not testable here without capturing stderr).
    striped_cache<int, int> c{10, 64};
    // The amplification raises effective capacity to num_shards (64).
    EXPECT_EQ(c.max_size(), 64u);

    // Inserting 10 keys must not be *silently rejected globally* — the
    // P0-A fix guarantees every shard can hold at least one item, so a
    // well-mixed hash can always place each key somewhere. However, the
    // per-shard quota is 1, so when two keys hash to the same shard one
    // of them is evicted (10 keys over 64 shards collide with ~30%
    // probability under a well-mixed hash). Assert the lenient behavior:
    // inserts succeed, at least one item survives, and every surviving
    // item holds the correct value.
    for (int i = 0; i < 10; ++i) {
        c.set(i, i * 10);
    }
    // At least one insert survived (no silent global rejection).
    EXPECT_GT(c.size(), 0u);
    // Every surviving key must hold the correct value.
    for (int i = 0; i < 10; ++i) {
        auto h = c.try_get(i);
        if (h) {
            EXPECT_EQ(**h, i * 10);
        }
    }
}

// ============================================================================
// TC-P2-1b: max_size_strict throws on undersized capacity
// ============================================================================
TEST(MaxSizeStrict, ThrowsOnUndersizedCapacity) {
    striped_cache<int, int> c{1024, 64};
    EXPECT_THROW(c.max_size_strict(10), std::invalid_argument);
    EXPECT_THROW(c.max_size_strict(63), std::invalid_argument);  // just under
    // Capacity must be unchanged after a throw (strong exception guarantee).
    EXPECT_EQ(c.max_size(), 1024);
}

// ============================================================================
// TC-P2-1c: max_size_strict accepts valid capacity
// ============================================================================
TEST(MaxSizeStrict, AcceptsValidCapacity) {
    striped_cache<int, int> c{1024, 64};
    // num_shards = 64, so 64 is the minimum accepted value.
    c.max_size_strict(64);
    EXPECT_EQ(c.max_size(), 64);
    // Larger values are accepted.
    c.max_size_strict(256);
    EXPECT_EQ(c.max_size(), 256);
    c.max_size_strict(4096);
    EXPECT_EQ(c.max_size(), 4096);
}

// ============================================================================
// TC-P2-1d: max_size_strict accepts unlimited
// ============================================================================
TEST(MaxSizeStrict, AcceptsUnlimited) {
    striped_cache<int, int> c{1024, 64};
    // unlimited is a sentinel; the strict check must skip it.
    c.max_size_strict(static_cast<std::size_t>(-1));
    // Verify the cache still works (no eviction for unlimited).
    for (int i = 0; i < 10000; ++i) {
        c.set(i, i);
    }
    EXPECT_EQ(c.size(), 10000);
}

// ============================================================================
// TC-P2-1e: Non-sharded caches accept any max_size via strict API
// ============================================================================
TEST(MaxSizeStrict, NonShardedAcceptsAnyCapacity) {
    // safe_cache uses mm_lru (no sharding), so max_size_strict is
    // equivalent to max_size — no amplification possible.
    safe_cache<int, int> c{1024};
    c.max_size_strict(1);  // would throw for striped, but safe_cache is fine
    EXPECT_EQ(c.max_size(), 1);
    c.max_size_strict(0);  // zero is also fine for non-sharded
    EXPECT_EQ(c.max_size(), 0);
}

// ============================================================================
// TC-P2-1f: max_size_strict enforces capacity after switch
// ============================================================================
TEST(MaxSizeStrict, EnforcesCapacityAfterSwitch) {
    // Start with a large cache, then shrink via max_size_strict.
    striped_cache<int, int> c{4096, 64};
    // Fill beyond the target strict capacity.
    for (int i = 0; i < 200; ++i) {
        c.set(i, i);
    }
    ASSERT_GT(c.size(), 0);
    // Shrink to 64 (the minimum allowed for 64 shards). Existing items
    // beyond the new capacity are evicted lazily on subsequent inserts.
    c.max_size_strict(64);
    EXPECT_EQ(c.max_size(), 64);
    // Insert enough new keys to force eviction of old ones. The cache
    // must not exceed max_size after the dust settles.
    for (int i = 1000; i < 1100; ++i) {
        c.set(i, i);
    }
    EXPECT_LE(c.size(), 64u + 1u)  // +1 for rounding across shards
        << "cache size must not exceed max_size after strict shrink";
    // The most-recently-inserted key must be present (LRU).
    auto h = c.try_get(1099);
    ASSERT_TRUE(h.has_value());
    EXPECT_EQ(**h, 1099);
}
