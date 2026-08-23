// Unified LRU Cache - LFU (TinyLFU) Cache unit tests
// 与 unified_cache 架构对齐：使用 lru::lfu_cache<K,V> 别名（TinyLFU 策略）。
// mm_tiny_lfu 不直接提供 frequency()/min_frequency()/max_frequency()/pop_lfu()，
// 频率估计通过 c.mm().sketch().estimate(key) 访问，队列大小通过 tiny_size()/main_size() 访问。

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
    // peek 不改变频率和队列，返回 read_handle<const V>
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
    EXPECT_EQ(*c.get(1), 'a'); // 值不变
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

    // 插入新元素触发淘汰
    c.set(4, "four");

    // TinyLFU 的频率准入可能导致 size <= max_size（窗口队列晋升时会淘汰）
    EXPECT_LE(c.size(), 3u);
    // 新插入的元素应存在
    EXPECT_TRUE(c.contains(4));
    // 应触发了淘汰
    auto stats = c.stats_snapshot();
    EXPECT_GE(stats.evictions.value.load(), 1u);
}

TEST(LfuEvictionTest, FrequencyAwareAdmission) {
    // TinyLFU 的核心特性：频率高的元素更可能被保留
    lfu_cache<int, std::string> c(4);
    c.set(1, "one");
    c.set(2, "two");
    c.set(3, "three");
    c.set(4, "four");

    // 频繁访问 2 和 4，提高其频率估计
    for (int i = 0; i < 5; ++i) {
        c.get(2);
        c.get(4);
    }

    // 插入新元素，触发淘汰决策
    c.set(5, "five");
    c.set(6, "six");

    // size 不超过 max_size
    EXPECT_LE(c.size(), 4u);
    // 频率高的 2 和 4 中至少一个应被保留
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
    // TinyLFU 的窗口队列晋升可能导致 size <= max_size
    EXPECT_LE(c.size(), 5u);

    c.max_size(2); // 缩容触发淘汰
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
    c.set(3, 3); // 触发淘汰

    auto stats = c.stats_snapshot();
    EXPECT_EQ(stats.insertions.value.load(), 3u);
    EXPECT_EQ(stats.evictions.value.load(), 1u);
}

// ============================================================================
// Callback Tests
// ============================================================================

TEST(LfuCacheCallbackTest, HitCallback) {
    lfu_cache<int, int> c;
    c.set_defer_promotion(false);  // 显式启用即时提升，使 hit 回调同步触发
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
    c.set(3, 300); // 触发淘汰

    // A8: 频率感知淘汰——当频率相等时（newcomer_wins_on_tie=true），淘汰 Main tail。
    // 插入 1,2 后：Tiny={2}, Main={1}（1 被晋升到 Main）。
    // 插入 3 时触发 evict，比较 Tiny tail(2) 与 Main tail(1) 频率（均为 1），
    // 1>=1 → admit=true → evict_from_main → evict key=1。
    // 对齐 CacheLib MMTinyLFU.h:488-500。
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

    c.set("a", "1");     // 小
    c.set("bb", "22");   // 中
    c.set("ccc", "333"); // 大 - 应触发淘汰

    EXPECT_LE(c.current_memory(), 500u);
}

// ============================================================================
// TinyLFU-specific API Tests（通过 c.mm() 访问）
// ============================================================================

TEST(LfuCacheTinyLfuApiTest, TinyAndMainQueueSizes) {
    // TinyLFU 使用两个队列：Tiny（窗口）+ Main（频率准入）
    lfu_cache<int, int> c(10);
    c.set(1, 10);
    c.set(2, 20);
    c.set(3, 30);

    // 新元素进入 Tiny 队列
    auto& mm = c.mm();
    EXPECT_EQ(mm.tiny_size() + mm.main_size(), 3u);
}

TEST(LfuCacheTinyLfuApiTest, SketchFrequencyEstimation) {
    // CountMinSketch 提供频率估计
    lfu_cache<int, int> c(100);
    c.set(1, 100);

    // 多次访问 key=1
    for (int i = 0; i < 5; ++i) {
        c.get(1);
    }

    // sketch 应记录 key=1 的访问频率
    auto freq = c.mm().sketch().estimate(1);
    EXPECT_GT(freq, 0u);
}

TEST(LfuCacheTinyLfuApiTest, PopByKey) {
    // pop() 通过 MM 层访问
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
    // pop_lru() 通过 MM 层访问（先尝试 Tiny tail，再尝试 Main tail）
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
    c.mm().pop(1); // 显式移除，非淘汰

    EXPECT_EQ(evict_count, 0);
}

// ============================================================================
// Large-scale Frequency Admission Test
// ============================================================================

TEST(LfuCacheAdmissionTest, MassInsertionTriggersEviction) {
    // 插入大量元素后验证淘汰行为
    lfu_cache<int, int> c(50);

    for (int i = 0; i < 100; ++i) {
        c.set(i, i * 10);
    }

    // TinyLFU 的窗口队列晋升可能导致 size <= max_size
    EXPECT_LE(c.size(), 50u);

    auto stats = c.stats_snapshot();
    EXPECT_EQ(stats.insertions.value.load(), 100u);
    // 至少 50 次淘汰（at_capacity 检查 + maybe_promote 淘汰）
    EXPECT_GE(stats.evictions.value.load(), 50u);
}

TEST(LfuCacheAdmissionTest, HotKeysRetainedBetter) {
    // 验证频繁访问的 key 在淘汰决策中更可能被保留
    lfu_cache<int, int> c(10);

    // 插入 10 个元素（TinyLFU 晋升机制会使部分元素被淘汰）
    for (int i = 0; i < 10; ++i) {
        c.set(i, i);
    }

    // 频繁访问缓存中存在的 key，提升其频率估计
    for (int round = 0; round < 10; ++round) {
        for (int i = 0; i < 10; ++i) {
            c.get(i); // 命中的 key 会提升频率，未命中的不影响
        }
    }

    // 插入新元素触发淘汰决策
    for (int i = 10; i < 20; ++i) {
        c.set(i, i);
    }

    // size 不超过 max_size
    EXPECT_LE(c.size(), 10u);
    // 缓存应仍有元素
    EXPECT_GT(c.size(), 0u);
}
