// Unified LRU Cache — Cache statistics, key tracker, operator==/<<  unit tests
//
// Split from test_serde.cpp (2026-07-26) so test_serde.cpp focuses purely on
// serialization round-trips. These tests cover:
//   - cache_stats::to_string() and ostream operator
//   - key_statistics_tracker monitor/query/unmonitor/clear/hit_rate
//   - operator== across all cache aliases (LRU/FIFO/2Q/TinyLFU/W-TinyLFU)
//   - operator<< across all cache aliases and raw MM strategies

#include <gtest/gtest.h>

#include <chrono>
#include <sstream>
#include <string>

#include "../lru.hpp"

using namespace lru;

// ============================================================================
// cache_stats::to_string()
// ============================================================================

TEST(CacheStatsTest, ToStringFormat) {
    cache_stats s;
    s.register_hit();
    s.register_hit();
    s.register_hit();
    s.register_miss();

    auto str = s.to_string();
    EXPECT_NE(str.find("hits=3"), std::string::npos);
    EXPECT_NE(str.find("misses=1"), std::string::npos);
    EXPECT_NE(str.find("hit_rate=75.00%"), std::string::npos);
}

TEST(CacheStatsTest, ToStringEmpty) {
    cache_stats s;
    auto str = s.to_string();
    EXPECT_NE(str.find("hits=0"), std::string::npos);
    EXPECT_NE(str.find("misses=0"), std::string::npos);
    EXPECT_NE(str.find("hit_rate=0.00%"), std::string::npos);
}

TEST(CacheStatsTest, OstreamOperator) {
    cache_stats s;
    s.register_hit();
    std::ostringstream oss;
    oss << s;
    EXPECT_NE(oss.str().find("hits=1"), std::string::npos);
}

// ============================================================================
// key_statistics_tracker
// ============================================================================

TEST(KeyStatsTest, MonitorAndQuery) {
    key_statistics_tracker<int> tracker;
    tracker.monitor(42);
    tracker.register_hit(42);
    tracker.register_hit(42);
    tracker.register_miss(42);

    auto ks = tracker.stats_for(42);
    ASSERT_TRUE(ks.has_value());
    EXPECT_EQ(ks->hits.load(), 2u);
    EXPECT_EQ(ks->misses.load(), 1u);
}

TEST(KeyStatsTest, UnmonitoredKeyIgnored) {
    key_statistics_tracker<int> tracker;
    tracker.register_hit(42);   // not monitored
    EXPECT_FALSE(tracker.stats_for(42).has_value());
}

TEST(KeyStatsTest, Unmonitor) {
    key_statistics_tracker<int> tracker;
    tracker.monitor(42);
    tracker.register_hit(42);
    tracker.unmonitor(42);
    EXPECT_FALSE(tracker.stats_for(42).has_value());
}

TEST(KeyStatsTest, Clear) {
    key_statistics_tracker<int> tracker;
    tracker.monitor(1);
    tracker.monitor(2);
    tracker.register_hit(1);
    tracker.clear();
    EXPECT_FALSE(tracker.stats_for(1).has_value());
    EXPECT_FALSE(tracker.stats_for(2).has_value());
    EXPECT_EQ(tracker.monitored_count(), 0u);
}

TEST(KeyStatsTest, HitRate) {
    key_statistics_tracker<int> tracker;
    tracker.monitor(42);
    tracker.register_hit(42);
    tracker.register_hit(42);
    tracker.register_miss(42);
    auto ks = tracker.stats_for(42);
    ASSERT_TRUE(ks.has_value());
    EXPECT_DOUBLE_EQ(ks->hit_rate(), 2.0 / 3.0);
}

TEST(KeyStatsTest, IsMonitoring) {
    key_statistics_tracker<int> tracker;
    tracker.monitor(42);
    EXPECT_TRUE(tracker.is_monitoring(42));
    EXPECT_FALSE(tracker.is_monitoring(99));
    tracker.unmonitor(42);
    EXPECT_FALSE(tracker.is_monitoring(42));
}

// ============================================================================
// operator== Tests
// ============================================================================

TEST(OperatorEqTest, LruCacheEqual) {
    cache<int, int> a(10), b(10);
    a.set(1, 10); a.set(2, 20);
    b.set(1, 10); b.set(2, 20);
    EXPECT_EQ(a, b);
}

TEST(OperatorEqTest, LruCacheNotEqual) {
    cache<int, int> a(10), b(10);
    a.set(1, 10); a.set(2, 20);
    b.set(1, 10); b.set(2, 99);
    EXPECT_NE(a, b);
}

TEST(OperatorEqTest, LruCacheDifferentOrderNotEqual) {
    cache<int, int> a(10), b(10);
    a.set(1, 10); a.set(2, 20);
    b.set(2, 20); b.set(1, 10); // different order
    EXPECT_NE(a, b);
}

TEST(OperatorEqTest, FifoCacheEqual) {
    fifo_cache<int, int> a(10), b(10);
    a.set(1, 10); a.set(2, 20);
    b.set(1, 10); b.set(2, 20);
    EXPECT_EQ(a, b);
}

TEST(OperatorEqTest, FifoCacheNotEqual) {
    fifo_cache<int, int> a(10), b(10);
    a.set(1, 10); a.set(2, 20);
    b.set(1, 10); b.set(3, 30);
    EXPECT_NE(a, b);
}

TEST(OperatorEqTest, TwoQCacheEqual) {
    // Use raw mm_2q to ensure strict order comparison
    two_q<int, int> a(10), b(10);
    a.set(1, 10); a.set(2, 20);
    b.set(1, 10); b.set(2, 20);
    EXPECT_EQ(a, b);
}

TEST(OperatorEqTest, TwoQCacheNotEqual) {
    two_q<int, int> a(10), b(10);
    a.set(1, 10); a.set(2, 20);
    b.set(1, 99); b.set(2, 20);
    EXPECT_NE(a, b);
}

TEST(OperatorEqTest, TinyLfuCacheEqual) {
    // Use lfu_cache alias (= unified_cache + tiny_lfu_trait)
    lfu_cache<int, int> a(100), b(100);
    a.set(1, 10); a.set(2, 20);
    b.set(1, 10); b.set(2, 20);
    EXPECT_EQ(a, b);
}

TEST(OperatorEqTest, TinyLfuCacheNotEqual) {
    lfu_cache<int, int> a(100), b(100);
    a.set(1, 10); a.set(2, 20);
    b.set(1, 10); b.set(3, 30);
    EXPECT_NE(a, b);
}

TEST(OperatorEqTest, WTinyLfuCacheEqual) {
    w_tiny_lfu<int, int> a(100), b(100);
    a.set(1, 10); a.set(2, 20);
    b.set(1, 10); b.set(2, 20);
    EXPECT_EQ(a, b);
}

TEST(OperatorEqTest, WTinyLfuCacheNotEqual) {
    w_tiny_lfu<int, int> a(100), b(100);
    a.set(1, 10); a.set(2, 20);
    b.set(1, 10); b.set(3, 30);
    EXPECT_NE(a, b);
}

TEST(OperatorEqTest, EmptyCacheIsEqual) {
    cache<int, int> a(10), b(10);
    EXPECT_EQ(a, b);
}

TEST(OperatorEqTest, DifferentSizesNotEqual) {
    cache<int, int> a(10), b(10);
    a.set(1, 10);
    EXPECT_NE(a, b);
}

// ============================================================================
// operator<< Tests
// ============================================================================

TEST(StreamOutputTest, LruCacheContainsTypeName) {
    cache<int, int> c(10);
    c.set(1, 10);
    std::ostringstream oss;
    oss << c;
    auto s = oss.str();
    EXPECT_NE(s.find("unified_cache"), std::string::npos);
    EXPECT_NE(s.find("mm_lru"), std::string::npos);
    EXPECT_NE(s.find("1"), std::string::npos); // key or value
}

TEST(StreamOutputTest, FifoCacheContainsTypeName) {
    fifo_cache<int, int> c(10);
    c.set(1, 10);
    std::ostringstream oss;
    oss << c;
    auto s = oss.str();
    EXPECT_NE(s.find("unified_cache"), std::string::npos);
    EXPECT_NE(s.find("mm_fifo"), std::string::npos);
}

TEST(StreamOutputTest, TwoQCacheContainsTypeName) {
    two_q<int, int> c(10);
    c.set(1, 10);
    std::ostringstream oss;
    oss << c;
    auto s = oss.str();
    EXPECT_NE(s.find("mm_2q"), std::string::npos);
}

TEST(StreamOutputTest, TinyLfuCacheContainsTypeName) {
    lfu_cache<int, int> c(100);
    c.set(1, 10);
    std::ostringstream oss;
    oss << c;
    auto s = oss.str();
    EXPECT_NE(s.find("mm_tiny_lfu"), std::string::npos);
}

TEST(StreamOutputTest, WTinyLfuCacheContainsTypeName) {
    w_tiny_lfu<int, int> c(100);
    c.set(1, 10);
    std::ostringstream oss;
    oss << c;
    auto s = oss.str();
    EXPECT_NE(s.find("mm_wtiny_lfu"), std::string::npos);
}

TEST(StreamOutputTest, TtlCacheContainsDefaultTtl) {
    ttl_cache<int, int> c(std::chrono::seconds(30), 100);
    c.set(1, 10);
    std::ostringstream oss;
    oss << c;
    auto s = oss.str();
    EXPECT_NE(s.find("ttl_cache"), std::string::npos);
    EXPECT_NE(s.find("30s"), std::string::npos);
}

TEST(StreamOutputTest, TtlCacheOutputContainsTtlNone) {
    ttl_cache<int, int> c(std::chrono::seconds(30), 100);
    c.set_no_ttl(1, 10);
    std::ostringstream oss;
    oss << c;
    auto s = oss.str();
    EXPECT_NE(s.find("ttl=none"), std::string::npos);
}

TEST(StreamOutputTest, StreamOutputOnRawMmLru) {
    mm_lru<int, int> c(10);
    c.set(1, 10);
    std::ostringstream oss;
    oss << c;
    auto s = oss.str();
    EXPECT_NE(s.find("mm_lru"), std::string::npos);
    EXPECT_NE(s.find("hits=0"), std::string::npos);
}

TEST(StreamOutputTest, StreamOutputOnRawMmFifo) {
    mm_fifo<int, int> c(10);
    c.set(1, 10);
    std::ostringstream oss;
    oss << c;
    auto s = oss.str();
    EXPECT_NE(s.find("mm_fifo"), std::string::npos);
}

TEST(StreamOutputTest, StreamOutputOnRawMm2q) {
    mm_2q<int, int> c(10);
    c.set(1, 10);
    std::ostringstream oss;
    oss << c;
    auto s = oss.str();
    EXPECT_NE(s.find("mm_2q"), std::string::npos);
}

TEST(StreamOutputTest, StreamOutputOnRawMmTinyLfu) {
    mm_tiny_lfu<int, int> c(100);
    c.set(1, 10);
    std::ostringstream oss;
    oss << c;
    auto s = oss.str();
    EXPECT_NE(s.find("mm_tiny_lfu"), std::string::npos);
}

TEST(StreamOutputTest, StreamOutputOnRawMmWtinyLfu) {
    mm_wtiny_lfu<int, int> c(100);
    c.set(1, 10);
    std::ostringstream oss;
    oss << c;
    auto s = oss.str();
    EXPECT_NE(s.find("mm_wtiny_lfu"), std::string::npos);
}
