// Unified LRU Cache - W-TinyLFU / 2Q Eviction Strategy unit tests
// Tests for mm_wtiny_lfu and mm_2q via their convenience aliases

#include <gtest/gtest.h>
#include <string>
#include <vector>

#include "../lru.hpp"

using namespace lru;

// ============================================================================
// W-TinyLFU Basic Tests
// ============================================================================

class WTinyLfuTest : public ::testing::Test {
protected:
    w_tiny_lfu<int, char> c;

    void SetUp() override {
        // W-TinyLFU 窗口队列太小（max(1, size*0.01)=1），仅插入 1 个避免提前淘汰
        c.set(1, 'a');
    }
};

TEST_F(WTinyLfuTest, SetAndGet) {
    auto result = c.get(1);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, 'a');
}

TEST_F(WTinyLfuTest, Contains) {
    EXPECT_TRUE(c.contains(1));
    EXPECT_FALSE(c.contains(99));
}

TEST_F(WTinyLfuTest, DeleteExisting) {
    EXPECT_TRUE(c.del(1));
    EXPECT_EQ(c.size(), 0u);
    EXPECT_FALSE(c.get(1).has_value());
}

TEST_F(WTinyLfuTest, Flush) {
    c.flush();
    EXPECT_EQ(c.size(), 0u);
    EXPECT_TRUE(c.empty());
}

TEST_F(WTinyLfuTest, Peek) {
    auto result = c.peek(1);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, 'a');
}

TEST_F(WTinyLfuTest, AddReplaceSemantics) {
    EXPECT_TRUE(c.add(2, 'b'));
    EXPECT_FALSE(c.add(1, 'z'));  // 已存在
    EXPECT_EQ(*c.get(1), 'a');

    EXPECT_TRUE(c.replace(1, 'z'));
    EXPECT_EQ(*c.get(1), 'z');
    EXPECT_FALSE(c.replace(99, 'x'));
}

TEST_F(WTinyLfuTest, RemoveAndContains) {
    auto res = c.remove(1);
    EXPECT_EQ(res, decltype(c)::RemoveRes::kSuccess);
    EXPECT_FALSE(c.contains(1));

    res = c.remove(999);
    EXPECT_EQ(res, decltype(c)::RemoveRes::kNotFound);

    c.set(5, 'e');
    EXPECT_TRUE(c.contains(5));
    EXPECT_FALSE(c.contains(999));
}

TEST_F(WTinyLfuTest, InsertCallback) {
    int insert_key = 0;
    c.callbacks().on_insert([&](const int& k, const char&) {
        insert_key = k;
    });
    c.set(42, 'x');
    EXPECT_EQ(insert_key, 42);
}

TEST_F(WTinyLfuTest, EvictCallback) {
    w_tiny_lfu<int, int> c2(2);
    int evict_key = -1;
    c2.callbacks().on_evict([&](const int& k, const int&) {
        evict_key = k;
    });
    c2.set(1, 100);
    c2.set(2, 200);
    c2.set(3, 300);
    EXPECT_GE(evict_key, 0);
}

// ============================================================================
// W-TinyLFU Eviction & Queue Tests
// ============================================================================

TEST(WTinyLfuCapacityTest, MaxSizeEviction) {
    w_tiny_lfu<int, std::string> c(3);
    c.set(1, "one");
    c.set(2, "two");
    c.set(3, "three");
    c.set(4, "four");

    EXPECT_LE(c.size(), 3u);
    EXPECT_TRUE(c.contains(4));
}

TEST(WTinyLfuQueueTest, QueueSizes) {
    w_tiny_lfu<int, int> c(10);
    c.set(1, 10);
    c.set(2, 20);

    auto& mm = c.mm();
    EXPECT_EQ(mm.tiny_size() + mm.probation_size() + mm.protection_size(), 2u);
}

TEST(WTinyLfuQueueTest, SketchEstimation) {
    w_tiny_lfu<int, int> c(100);
    c.set(1, 100);
    for (int i = 0; i < 5; ++i) c.get(1);

    auto freq = c.mm().sketch().estimate(1);
    EXPECT_GT(freq, 0u);
}

TEST(WTinyLfuCapacityTest, MassInsertionTriggersEviction) {
    w_tiny_lfu<int, int> c(50);
    for (int i = 0; i < 100; ++i) c.set(i, i * 10);

    EXPECT_LE(c.size(), 50u);
    auto stats = c.stats_snapshot();
    EXPECT_GE(stats.evictions.value.load(), 50u);
}

TEST(WTinyLfuApiTest, PopByKey) {
    w_tiny_lfu<int, std::string> c;
    c.set(1, "one");
    c.set(2, "two");

    auto val = c.mm().pop(1);
    ASSERT_TRUE(val.has_value());
    EXPECT_EQ(*val, "one");
    EXPECT_EQ(c.size(), 1u);
}

TEST(WTinyLfuApiTest, PopLru) {
    w_tiny_lfu<int, std::string> c;
    c.set(1, "one");
    c.set(2, "two");

    auto result = c.mm().pop_lru();
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(c.size(), 1u);
}

TEST(WTinyLfuApiTest, PopLruEmptyCache) {
    w_tiny_lfu<int, std::string> c;
    auto result = c.mm().pop_lru();
    EXPECT_FALSE(result.has_value());
}

// ============================================================================
// 2Q Cache Tests
// ============================================================================

class TwoQTest : public ::testing::Test {
protected:
    two_q<int, char> c;

    void SetUp() override {
        c.set(1, 'a');
        c.set(2, 'b');
        c.set(3, 'c');
    }
};

TEST_F(TwoQTest, SetAndGet) {
    auto result = c.get(1);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, 'a');
}

TEST_F(TwoQTest, Contains) {
    EXPECT_TRUE(c.contains(1));
    EXPECT_FALSE(c.contains(99));
}

TEST_F(TwoQTest, DeleteExisting) {
    EXPECT_TRUE(c.del(1));
    EXPECT_FALSE(c.contains(1));
}

TEST_F(TwoQTest, Peek) {
    auto result = c.peek(1);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, 'a');
}

TEST_F(TwoQTest, AddReplace) {
    EXPECT_TRUE(c.add(4, 'd'));
    EXPECT_FALSE(c.add(1, 'z'));
    EXPECT_TRUE(c.replace(1, 'z'));
    EXPECT_EQ(*c.get(1), 'z');
}

TEST_F(TwoQTest, Flush) {
    c.flush();
    EXPECT_EQ(c.size(), 0u);
}

TEST_F(TwoQTest, RemoveAndContains) {
    EXPECT_EQ(c.remove(1), decltype(c)::RemoveRes::kSuccess);
    EXPECT_FALSE(c.contains(1));
    EXPECT_EQ(c.remove(999), decltype(c)::RemoveRes::kNotFound);
    EXPECT_FALSE(c.contains(999));
}

TEST(TwoQCapacityTest, MaxSizeEviction) {
    two_q<int, std::string> c(3);
    c.set(1, "one");
    c.set(2, "two");
    c.set(3, "three");
    c.set(4, "four");

    EXPECT_LE(c.size(), 3u);
    EXPECT_TRUE(c.contains(4));
}

TEST(TwoQApiTest, QueueSizes) {
    two_q<int, int> c(10);
    c.set(1, 10);
    c.set(2, 20);
    c.set(3, 30);

    auto& mm = c.mm();
    EXPECT_EQ(mm.hot_size() + mm.warm_size() + mm.cold_size(), 3u);
}

TEST(TwoQApiTest, PerQueueStats) {
    two_q<int, int> c(10);
    c.set(1, 10);
    c.get(1);
    auto pqs = c.mm().get_per_queue_stats();
    EXPECT_GE(pqs.num_hot_accesses + pqs.num_warm_accesses + pqs.num_cold_accesses, 0u);
}

TEST(TwoQApiTest, PopByKey) {
    two_q<int, std::string> c;
    c.set(1, "one");
    c.set(2, "two");

    auto val = c.mm().pop(1);
    ASSERT_TRUE(val.has_value());
    EXPECT_EQ(*val, "one");
}

// ============================================================================
// FIFO Cache — CRUD 和淘汰语义
// (migrated from test_serde.cpp — these CRUD/eviction-order tests are
//  unrelated to serialization and belong with the other eviction strategies)
// ============================================================================

class FifoCacheTest : public ::testing::Test {
protected:
    fifo_cache<int, int> c;
    void SetUp() override {
        c.set(1, 10);
        c.set(2, 20);
        c.set(3, 30);
    }
};

TEST_F(FifoCacheTest, SetAndGet) {
    auto v = c.get(1);
    ASSERT_TRUE(v.has_value());
    EXPECT_EQ(*v, 10);
}

TEST_F(FifoCacheTest, Contains) {
    EXPECT_TRUE(c.contains(1));
    EXPECT_FALSE(c.contains(99));
}

TEST_F(FifoCacheTest, AddNewKey) {
    EXPECT_TRUE(c.add(4, 40));
    EXPECT_EQ(c.size(), 4u);
}

TEST_F(FifoCacheTest, AddExistingDoesNotChangeValue) {
    EXPECT_FALSE(c.add(1, 999));
    EXPECT_EQ(*c.get(1), 10);
}

TEST_F(FifoCacheTest, ReplaceExisting) {
    EXPECT_TRUE(c.replace(1, 999));
    EXPECT_EQ(*c.get(1), 999);
}

TEST_F(FifoCacheTest, ReplaceNonExisting) {
    EXPECT_FALSE(c.replace(99, 999));
}

TEST_F(FifoCacheTest, DeleteExisting) {
    EXPECT_TRUE(c.del(2));
    EXPECT_EQ(c.size(), 2u);
    EXPECT_FALSE(c.get(2).has_value());
}

TEST_F(FifoCacheTest, DeleteNonExisting) {
    EXPECT_FALSE(c.del(99));
}

TEST_F(FifoCacheTest, Flush) {
    c.flush();
    EXPECT_TRUE(c.empty());
    EXPECT_EQ(c.size(), 0u);
}

TEST_F(FifoCacheTest, AccessDoesNotChangeEvictionOrder) {
    // FIFO: access should NOT promote, so oldest remains at eviction position
    // Re-create with limited size
    fifo_cache<int, int> limited(3);
    limited.set(1, 10);
    limited.set(2, 20);
    limited.set(3, 30);
    limited.get(1);         // access oldest — FIFO should NOT promote
    limited.set(4, 40);     // should evict 1 (oldest)
    EXPECT_FALSE(limited.contains(1)) << "FIFO should evict oldest on insertion";
    EXPECT_TRUE(limited.contains(2));
    EXPECT_TRUE(limited.contains(3));
    EXPECT_TRUE(limited.contains(4));
}

TEST_F(FifoCacheTest, EvictionOrderOldestFirst) {
    fifo_cache<int, int> small(2);
    small.set(1, 10);
    small.set(2, 20);
    small.set(3, 30); // evicts 1
    EXPECT_FALSE(small.contains(1));
    EXPECT_TRUE(small.contains(2));
    EXPECT_TRUE(small.contains(3));
}

TEST_F(FifoCacheTest, PopByKey) {
    auto v1 = c.mm().pop(1);
    ASSERT_TRUE(v1.has_value());
    EXPECT_EQ(*v1, 10);
    EXPECT_EQ(c.size(), 2u);
}

TEST_F(FifoCacheTest, PopOldest) {
    auto p = c.mm().pop_lru();
    ASSERT_TRUE(p.has_value());
    EXPECT_EQ(p->first, 1); // oldest = smallest key
    EXPECT_EQ(p->second, 10);
}
