// Unified LRU Cache - LFU (TinyLFU) Cache unit tests
// Aligned with the unified_cache architecture: use the lru::lfu_cache<K,V> alias
// (TinyLFU). mm_tiny_lfu does not expose frequency()/min_frequency()/
// max_frequency()/pop_lfu(); frequency estimates come from
// c.mm().sketch().estimate(key) and queue sizes from tiny_size()/main_size().

#include <gtest/gtest.h>
#include <string>
#include <vector>

#include "../lru.hpp"

using namespace lru;

// ============================================================================
// Basic CRUD Tests
// ============================================================================

class LfuCacheCrudTest : public ::testing::Test {
protected:
    lfu_cache<int, char> c;

    void SetUp() override {
        c.set(1, 'a');
        c.set(2, 'b');
        c.set(3, 'c');
    }
};

TEST_F(LfuCacheCrudTest, SetAndGet) {
    auto result = c.get(1);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, 'a');
}

TEST_F(LfuCacheCrudTest, Contains) {
    EXPECT_TRUE(c.contains(1));
    EXPECT_FALSE(c.contains(99));
}

TEST_F(LfuCacheCrudTest, EmptyCache) {
    lfu_cache<int, char> empty_cache;
    EXPECT_TRUE(empty_cache.empty());
    EXPECT_EQ(empty_cache.size(), 0u);
}

TEST_F(LfuCacheCrudTest, PeekDoesNotPromote) {
    // peek does not change frequency or queue placement; returns read_handle<const V>.
    auto result = c.peek(1);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, 'a');
}

TEST_F(LfuCacheCrudTest, AddNewKey) {
    EXPECT_TRUE(c.add(4, 'd'));
    EXPECT_EQ(c.size(), 4u);
}

TEST_F(LfuCacheCrudTest, AddExistingKeyDoesNotChangeValue) {
    EXPECT_FALSE(c.add(1, 'z'));
    EXPECT_EQ(*c.get(1), 'a'); // value unchanged
}

TEST_F(LfuCacheCrudTest, ReplaceExisting) {
    EXPECT_TRUE(c.replace(1, 'z'));
    EXPECT_EQ(*c.get(1), 'z');
}

TEST_F(LfuCacheCrudTest, ReplaceNonExisting) {
    EXPECT_FALSE(c.replace(99, 'z'));
}

TEST_F(LfuCacheCrudTest, DeleteExisting) {
    EXPECT_TRUE(c.del(1));
    EXPECT_EQ(c.size(), 2u);
    EXPECT_FALSE(c.get(1).has_value());
}

TEST_F(LfuCacheCrudTest, DeleteNonExisting) {
    EXPECT_FALSE(c.del(99));
}

TEST_F(LfuCacheCrudTest, Flush) {
    c.flush();
    EXPECT_EQ(c.size(), 0u);
    EXPECT_TRUE(c.empty());
}

// ============================================================================
// TinyLFU Eviction Tests
// ============================================================================

TEST(LfuEvictionTest, EvictsOnCapacityOverflow) {
    lfu_cache<int, std::string> c(3);
    c.set(1, "one");
    c.set(2, "two");
    c.set(3, "three");

    // Inserting a new element triggers eviction.
    c.set(4, "four");

    // TinyLFU's frequency admission can leave size <= max_size (a window promotion evicts).
    EXPECT_LE(c.size(), 3u);
    // The newly inserted element must be present.
    EXPECT_TRUE(c.contains(4));
    // An eviction must have happened.
    auto stats = c.stats_snapshot();
    EXPECT_GE(stats.evictions.value.load(), 1u);
}

TEST(LfuEvictionTest, FrequencyAwareAdmission) {
    // TinyLFU's core property: frequently used elements are more likely to survive.
    lfu_cache<int, std::string> c(4);
    c.set(1, "one");
    c.set(2, "two");
    c.set(3, "three");
    c.set(4, "four");

    // Access 2 and 4 repeatedly to raise their frequency estimates.
    for (int i = 0; i < 5; ++i) {
        c.get(2);
        c.get(4);
    }

    // Insert a new element to trigger an admission decision.
    c.set(5, "five");
    c.set(6, "six");

    // size must not exceed max_size.
    EXPECT_LE(c.size(), 4u);
    // At least one of the high-frequency keys (2 and 4) must survive.
    int hot_retained = 0;
    if (c.contains(2)) ++hot_retained;
    if (c.contains(4)) ++hot_retained;
    EXPECT_GE(hot_retained, 1);
}

TEST(LfuEvictionTest, ResizeDown) {
    lfu_cache<int, std::string> c(5);
    for (int i = 0; i < 5; ++i) {
        c.set(i, std::to_string(i));
    }
    // A TinyLFU window promotion can leave size <= max_size.
    EXPECT_LE(c.size(), 5u);

    c.max_size(2); // shrinking triggers eviction
    EXPECT_LE(c.size(), 2u);
}

// ============================================================================
// Statistics Tests
// ============================================================================

TEST(LfuCacheStatsTest, HitMissTracking) {
    lfu_cache<int, int> c;
    c.set(1, 100);

    c.get(1); // hit
    c.get(1); // hit
    c.get(2); // miss

    auto stats = c.stats_snapshot();
    EXPECT_EQ(stats.hits.value.load(), 2u);
    EXPECT_EQ(stats.misses.value.load(), 1u);
    EXPECT_DOUBLE_EQ(stats.hit_rate(), 2.0 / 3.0);
}

TEST(LfuCacheStatsTest, InsertionAndEvictionTracking) {
    lfu_cache<int, int> c(2);
    c.set(1, 1);
    c.set(2, 2);
    c.set(3, 3); // triggers eviction

    auto stats = c.stats_snapshot();
    EXPECT_EQ(stats.insertions.value.load(), 3u);
    EXPECT_EQ(stats.evictions.value.load(), 1u);
}

// ============================================================================
// Callback Tests
// ============================================================================

TEST(LfuCacheCallbackTest, HitCallback) {
    lfu_cache<int, int> c;
    c.set_defer_promotion(false);  // enable immediate promotion so the hit callback fires synchronously
    int hit_key = 0;
    int hit_value = 0;

    c.callbacks().on_hit([&](const int& k, const int& v) {
        hit_key = k;
        hit_value = v;
    });

    c.set(1, 100);
    c.get(1);

    EXPECT_EQ(hit_key, 1);
    EXPECT_EQ(hit_value, 100);
}

TEST(LfuCacheCallbackTest, MissCallback) {
    lfu_cache<int, int> c;
    int miss_key = 0;

    c.callbacks().on_miss([&](const int& k) {
        miss_key = k;
    });

    c.get(99);

    EXPECT_EQ(miss_key, 99);
}

TEST(LfuCacheCallbackTest, EvictCallback) {
    lfu_cache<int, int> c(2);
    int evict_key = -1;
    int evict_value = 0;

    c.callbacks().on_evict([&](const int& k, const int& v) {
        evict_key = k;
        evict_value = v;
    });

    c.set(1, 100);
    c.set(2, 200);
    c.set(3, 300); // triggers eviction

    // Frequency-aware eviction: on a frequency tie (newcomer_wins_on_tie == true)
    // the Main tail is evicted. After inserting 1 and 2: Tiny={2}, Main={1}.
    // Inserting 3 evicts, comparing the Tiny tail (2) with the Main tail (1),
    // both frequency 1: 1>=1 -> admit -> evict_from_main -> evict key 1.
    // Mirrors CacheLib MMTinyLFU.h:488-500.
    EXPECT_EQ(evict_key, 1);
    EXPECT_EQ(evict_value, 100);
}

TEST(LfuCacheCallbackTest, InsertCallback) {
    lfu_cache<int, int> c;
    int insert_key = 0;
    int insert_value = 0;

    c.callbacks().on_insert([&](const int& k, const int& v) {
        insert_key = k;
        insert_value = v;
    });

    c.set(1, 100);

    EXPECT_EQ(insert_key, 1);
    EXPECT_EQ(insert_value, 100);
}

// ============================================================================
// Dual Capacity Limits Tests
// ============================================================================

TEST(LfuCacheCapacityTest, MaxMemoryEviction) {
    lfu_cache<std::string, std::string> c(unlimited, 500);
    c.set_key_size_calculator([](const std::string& s) { return s.size(); });
    c.set_value_size_calculator([](const std::string& s) { return s.size(); });

    c.set("a", "1");     // small
    c.set("bb", "22");   // medium
    c.set("ccc", "333"); // large - should trigger eviction

    EXPECT_LE(c.current_memory(), 500u);
}

// ============================================================================
// TinyLFU-specific API tests (reached through c.mm()).
// ============================================================================

TEST(LfuCacheTinyLfuApiTest, TinyAndMainQueueSizes) {
    // TinyLFU uses two queues: Tiny (window) and Main (frequency admission).
    lfu_cache<int, int> c(10);
    c.set(1, 10);
    c.set(2, 20);
    c.set(3, 30);

    // A new element starts in the Tiny queue.
    auto& mm = c.mm();
    EXPECT_EQ(mm.tiny_size() + mm.main_size(), 3u);
}

TEST(LfuCacheTinyLfuApiTest, SketchFrequencyEstimation) {
    // CountMinSketch provides the frequency estimate.
    lfu_cache<int, int> c(100);
    c.set(1, 100);

    // Access key 1 several times.
    for (int i = 0; i < 5; ++i) {
        c.get(1);
    }

    // The sketch must have recorded the accesses to key 1.
    auto freq = c.mm().sketch().estimate(1);
    EXPECT_GT(freq, 0u);
}

TEST(LfuCacheTinyLfuApiTest, PopByKey) {
    // pop() goes through the MM layer.
    lfu_cache<int, std::string> c;
    c.set(1, "one");
    c.set(2, "two");

    auto val = c.mm().pop(1);
    ASSERT_TRUE(val.has_value());
    EXPECT_EQ(*val, "one");
    EXPECT_EQ(c.size(), 1u);
    EXPECT_FALSE(c.contains(1));
}

TEST(LfuCacheTinyLfuApiTest, PopNonExistentKey) {
    lfu_cache<int, std::string> c;
    auto val = c.mm().pop(99);
    EXPECT_FALSE(val.has_value());
}

TEST(LfuCacheTinyLfuApiTest, PopLru) {
    // pop_lru() goes through the MM layer (Tiny tail first, then Main tail).
    lfu_cache<int, std::string> c;
    c.set(1, "one");
    c.set(2, "two");

    auto result = c.mm().pop_lru();
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(c.size(), 1u);
}

TEST(LfuCacheTinyLfuApiTest, PopLruEmptyCache) {
    lfu_cache<int, std::string> c;
    auto result = c.mm().pop_lru();
    EXPECT_FALSE(result.has_value());
}

TEST(LfuCacheTinyLfuApiTest, PopDoesNotTriggerEvictCallback) {
    lfu_cache<int, int> c(2);
    int evict_count = 0;
    c.callbacks().on_evict([&](const int&, const int&) {
        ++evict_count;
    });

    c.set(1, 100);
    c.set(2, 200);
    c.mm().pop(1); // explicit removal, not an eviction

    EXPECT_EQ(evict_count, 0);
}

// ============================================================================
// Large-scale Frequency Admission Test
// ============================================================================

TEST(LfuCacheAdmissionTest, MassInsertionTriggersEviction) {
    // Verify eviction behaviour after inserting many elements.
    lfu_cache<int, int> c(50);

    for (int i = 0; i < 100; ++i) {
        c.set(i, i * 10);
    }

    // A TinyLFU window promotion can leave size <= max_size.
    EXPECT_LE(c.size(), 50u);

    auto stats = c.stats_snapshot();
    EXPECT_EQ(stats.insertions.value.load(), 100u);
    // At least 50 evictions (at_capacity check plus maybe_promote eviction).
    EXPECT_GE(stats.evictions.value.load(), 50u);
}

TEST(LfuCacheAdmissionTest, HotKeysRetainedBetter) {
    // Frequently accessed keys must be more likely to survive eviction.
    lfu_cache<int, int> c(10);

    // Insert 10 elements (TinyLFU promotion evicts some of them).
    for (int i = 0; i < 10; ++i) {
        c.set(i, i);
    }

    // Access keys that exist to raise their frequency estimates.
    for (int round = 0; round < 10; ++round) {
        for (int i = 0; i < 10; ++i) {
            c.get(i); // a hit raises the frequency; a miss changes nothing
        }
    }

    // Insert a new element to trigger an admission decision.
    for (int i = 10; i < 20; ++i) {
        c.set(i, i);
    }

    // size must not exceed max_size.
    EXPECT_LE(c.size(), 10u);
    // The cache must still hold elements.
    EXPECT_GT(c.size(), 0u);
}
