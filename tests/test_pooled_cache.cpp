// SPDX-License-Identifier: MIT
// G22 regression test: pooled_cache basic functionality.
//
// pooled_cache partitions items across independent MM instances (pools),
// each with its own size quota and priority. Pools share a global
// max_size budget with weighted-priority cross-pool eviction.
//
// This test covers:
//   - Basic set/get/del/contains per pool
//   - Pool isolation (a key in pool A is invisible to pool B)
//   - get_any / contains_any cross-pool search
//   - Global max_size enforcement (cross-pool eviction)
//   - Per-pool stats and pool management (add/remove/resize)

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "../lru.hpp"

using namespace lru;

// ============================================================================
// Basic set/get/contains per pool
// ============================================================================
TEST(PooledCacheTest, BasicSetGetPerPool) {
    pooled_cache<cache<int, std::string>> pc(1000);
    pc.add_pool({.name = "alpha", .max_size = 500, .priority = 200});
    pc.add_pool({.name = "beta", .max_size = 500, .priority = 100});

    pc.set("alpha", 1, "one");
    pc.set("beta", 2, "two");

    auto h1 = pc.get("alpha", 1);
    ASSERT_TRUE(h1);
    EXPECT_EQ(*h1, "one");

    auto h2 = pc.get("beta", 2);
    ASSERT_TRUE(h2);
    EXPECT_EQ(*h2, "two");

    EXPECT_TRUE(pc.contains("alpha", 1));
    EXPECT_TRUE(pc.contains("beta", 2));
    EXPECT_FALSE(pc.contains("alpha", 2));
    EXPECT_FALSE(pc.contains("beta", 1));

    EXPECT_EQ(pc.size(), 2u);
}

// ============================================================================
// Pool isolation: a key in one pool is invisible to another
// ============================================================================
TEST(PooledCacheTest, PoolIsolation) {
    pooled_cache<cache<int, std::string>> pc(1000);
    pc.add_pool({.name = "A", .max_size = 500, .priority = 200});
    pc.add_pool({.name = "B", .max_size = 500, .priority = 100});

    // Same key in different pools with different values.
    pc.set("A", 42, "from_A");
    pc.set("B", 42, "from_B");

    {
        auto ha = pc.get("A", 42);
        ASSERT_TRUE(ha);
        EXPECT_EQ(*ha, "from_A");
    }
    {
        auto hb = pc.get("B", 42);
        ASSERT_TRUE(hb);
        EXPECT_EQ(*hb, "from_B");
    }
    // ha/hb have been released (read_handle is RAII — destructor decRefs).

    // Deleting from pool A must not affect pool B.
    EXPECT_TRUE(pc.del("A", 42));
    EXPECT_FALSE(pc.contains("A", 42));
    EXPECT_TRUE(pc.contains("B", 42));
    EXPECT_EQ(pc.size(), 1u);
}

// ============================================================================
// Cross-pool search: get_any / contains_any / del_any
// ============================================================================
TEST(PooledCacheTest, CrossPoolSearch) {
    pooled_cache<cache<int, std::string>> pc(1000);
    pc.add_pool({.name = "hot", .max_size = 500, .priority = 200});
    pc.add_pool({.name = "cold", .max_size = 500, .priority = 50});

    pc.set("hot", 1, "hot_val");
    pc.set("cold", 2, "cold_val");

    // contains_any searches all pools.
    EXPECT_TRUE(pc.contains_any(1));
    EXPECT_TRUE(pc.contains_any(2));
    EXPECT_FALSE(pc.contains_any(99));

    // get_any searches all pools.
    {
        auto h1 = pc.get_any(1);
        ASSERT_TRUE(h1);
        EXPECT_EQ(*h1, "hot_val");
    }
    {
        auto h2 = pc.get_any(2);
        ASSERT_TRUE(h2);
        EXPECT_EQ(*h2, "cold_val");
    }
    // h1/h2 released (RAII) before del_any — see PoolIsolation comment:
    // del_any() routes to mm_lru::del which refuses items with active
    // handles.

    auto h99 = pc.get_any(99);
    EXPECT_FALSE(h99);

    // del_any removes from whichever pool holds the key.
    EXPECT_TRUE(pc.del_any(1));
    EXPECT_FALSE(pc.contains_any(1));
    EXPECT_EQ(pc.size(), 1u);
}

// ============================================================================
// Global max_size enforcement triggers cross-pool eviction
// ============================================================================
TEST(PooledCacheTest, GlobalMaxSizeEviction) {
    // Global limit of 4 items across all pools.
    pooled_cache<cache<int, std::string>> pc(4);
    pc.add_pool({.name = "high", .max_size = pool_config::pool_auto, .priority = 200});
    pc.add_pool({.name = "low", .max_size = pool_config::pool_auto, .priority = 50});

    // Insert 4 items — fills the global budget.
    pc.set("high", 1, "h1");
    pc.set("high", 2, "h2");
    pc.set("low", 3, "l3");
    pc.set("low", 4, "l4");
    EXPECT_EQ(pc.size(), 4u);

    // Insert a 5th item — must trigger cross-pool eviction.
    // The low-priority pool (priority=50) should be evicted first.
    pc.set("low", 5, "l5");
    EXPECT_LE(pc.size(), 4u)
        << "Global max_size exceeded — cross-pool eviction failed";
}

// ============================================================================
// Pool management: add_pool / remove_pool / has_pool / pool_count
// ============================================================================
TEST(PooledCacheTest, PoolManagement) {
    pooled_cache<cache<int, std::string>> pc(1000);

    EXPECT_EQ(pc.pool_count(), 0u);
    EXPECT_FALSE(pc.has_pool("x"));

    pc.add_pool({.name = "x", .max_size = 100, .priority = 100});
    pc.add_pool({.name = "y", .max_size = 100, .priority = 200});

    EXPECT_EQ(pc.pool_count(), 2u);
    EXPECT_TRUE(pc.has_pool("x"));
    EXPECT_TRUE(pc.has_pool("y"));

    // Duplicate pool name throws.
    EXPECT_THROW(pc.add_pool({.name = "x", .max_size = 50, .priority = 100}),
                 std::invalid_argument);

    // Set data, then remove pool — data is gone.
    pc.set("x", 1, "a");
    pc.set("y", 2, "b");
    EXPECT_EQ(pc.size(), 2u);

    EXPECT_TRUE(pc.remove_pool("x"));
    EXPECT_FALSE(pc.has_pool("x"));
    EXPECT_EQ(pc.pool_count(), 1u);
    EXPECT_EQ(pc.size(), 1u);  // only pool "y" remains
    EXPECT_TRUE(pc.contains("y", 2));
}

// ============================================================================
// Per-pool statistics
// ============================================================================
TEST(PooledCacheTest, PerPoolStats) {
    pooled_cache<cache<int, std::string>> pc(1000);
    pc.add_pool({.name = "A", .max_size = 100, .priority = 200});
    pc.add_pool({.name = "B", .max_size = 100, .priority = 100});

    pc.set("A", 1, "a1");
    pc.set("A", 2, "a2");
    pc.set("B", 3, "b3");

    auto stats = pc.all_pool_stats();
    ASSERT_EQ(stats.size(), 2u);

    // Find stats for pool A and B (order may vary after eviction-order sort).
    const pool_stats* sa = nullptr;
    const pool_stats* sb = nullptr;
    for (const auto& s : stats) {
        if (s.name == "A") sa = &s;
        if (s.name == "B") sb = &s;
    }
    ASSERT_NE(sa, nullptr);
    ASSERT_NE(sb, nullptr);

    EXPECT_EQ(sa->size, 2u);
    EXPECT_EQ(sb->size, 1u);
    EXPECT_EQ(sa->priority, 200u);
    EXPECT_EQ(sb->priority, 100u);
    EXPECT_EQ(sa->insertions, 2u);
    EXPECT_EQ(sb->insertions, 1u);
}

// ============================================================================
// Flush clears all pools; flush_pool clears a single pool
// ============================================================================
TEST(PooledCacheTest, Flush) {
    pooled_cache<cache<int, std::string>> pc(1000);
    pc.add_pool({.name = "A", .max_size = 100, .priority = 200});
    pc.add_pool({.name = "B", .max_size = 100, .priority = 100});

    pc.set("A", 1, "a");
    pc.set("B", 2, "b");
    EXPECT_EQ(pc.size(), 2u);

    pc.flush_pool("A");
    EXPECT_EQ(pc.size(), 1u);
    EXPECT_FALSE(pc.contains("A", 1));
    EXPECT_TRUE(pc.contains("B", 2));

    pc.flush();
    EXPECT_EQ(pc.size(), 0u);
    EXPECT_TRUE(pc.empty());
}
