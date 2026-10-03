// SPDX-License-Identifier: MIT
// capacity amplification policy + set_max_size_strict API.
//
// Validates:
//   1. P1-30: an undersized striped capacity is never silently amplified —
//      the shard/stripe count is derived from the requested capacity instead
//      (option B), and an EXPLICIT shard layout that cannot be honoured throws
//      (option A).
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
// TC-P2-1a: P1-30 — undersized capacity is never silently amplified
// ============================================================================
TEST(MaxSizeStrict, ShrinksShardCountInsteadOfAmplifying) {
    // `striped_cache<int,int> c{10}` used to be
    // laid out over `default_num_stripes` (64) shards, which silently raised
    // the effective capacity to 64 (6.4x the request) and made max_size()
    // report a value the caller never asked for.
    //
    // Option A makes silent amplification a hard error; option B removes the need
    // for it in the common one-argument case by deriving the SHARD count from
    // the requested capacity (stripes stay as requested — they are the lock
    // granularity, not a capacity bound).
    striped_cache<int, int> c{10};
    EXPECT_EQ(c.max_size(), 10u);
    EXPECT_EQ(c.requested_max_size(), 10u);
    EXPECT_LE(c.num_shards(), 10u);
    EXPECT_EQ(c.num_stripes(), 64u);

    // A request that matches the shard count is accepted unchanged.
    striped_cache<int, int> ok{64, 64};
    EXPECT_EQ(ok.max_size(), 64u);

    // Largest requests are accepted and honoured verbatim.
    striped_cache<int, int> big{1024, 64};
    EXPECT_EQ(big.max_size(), 1024u);
    EXPECT_EQ(big.requested_max_size(), 1024u);
}

// ============================================================================
// TC-P2-1a2: P1-30 — an explicit shard layout still refuses amplification
// ============================================================================
TEST(MaxSizeStrict, ThrowsWhenExplicitShardLayoutCannotBeHonoured) {
    // When the shard count is supplied explicitly through the MM config there
    // is nothing to shrink: asking for 8 slots across 64 shards must fail
    // loudly rather than being multiplied into 64 slots.
    sharded_mm_lru_config cfg;
    cfg.num_shards = 64;
    EXPECT_THROW((striped_cache<int, int>(8, cfg)), cache_config_exception);

    // ...unless the caller explicitly accepts the per-shard floor.
    cfg.allow_amplification = true;
    EXPECT_NO_THROW((striped_cache<int, int>(8, cfg)));
}

// ============================================================================
// TC-P2-1a3: P1-30 — explicit opt-in restores the documented amplification
// ============================================================================
TEST(MaxSizeStrict, AmplifiesWhenExplicitlyAllowed) {
    // With allow_amplification = true the per-shard floor of one slot applies
    // and the effective capacity is raised to num_shards. The two numbers stay
    // separately reportable: max_size() = effective, requested_max_size() =
    // what the caller asked for.
    sharded_mm_lru_config cfg;
    cfg.num_shards = 64;
    cfg.allow_amplification = true;
    striped_cache<int, int> c(10, cfg);
    EXPECT_EQ(c.requested_max_size(), 10u);
    EXPECT_EQ(c.max_size(), 64u);

    // Every shard has at least one slot, so no insert is silently dropped
    // for landing on a zero-quota shard.
    for (int i = 0; i < 10; ++i) {
        c.set(i, i * 10);
    }
    EXPECT_GT(c.size(), 0u);
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
