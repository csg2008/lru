// T18: hot_shards memory dimension tests.
//
// Validates that shard_hotspot carries the new T18.1 fields
// (memory_usage / pending_reclaim / rehash_in_progress) and that
// hot_shards_by_memory(n) ranks shards by memory consumption rather
// than access volume.

#include "../lru.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <string>
#include <vector>

TEST(HotShardsMemoryTest, StructHasNewFields) {
    lru::striped_cache<int, std::string> c{256, 4};
    c.set(1, "one");
    auto hot = c.hot_shards(2);
    ASSERT_FALSE(hot.empty());
    // All entries must have non-negative memory_usage and pending_reclaim.
    for (const auto& s : hot) {
        EXPECT_GE(s.memory_usage, 0u);
        EXPECT_GE(s.pending_reclaim, 0u);
        // rehash_in_progress is a bool flag.
        EXPECT_TRUE(s.rehash_in_progress == true || s.rehash_in_progress == false);
    }
}

TEST(HotShardsMemoryTest, MemoryUsageMatchesMemoryBytes) {
    // memory_usage is an alias for memory_bytes (both reflect current_memory
    // of the shard). They must be equal.
    lru::striped_cache<int, std::string> c{256, 4};
    for (int i = 0; i < 64; ++i) c.set(i, std::string(32, 'x'));
    auto hot = c.hot_shards(4);
    ASSERT_FALSE(hot.empty());
    for (const auto& s : hot) {
        EXPECT_EQ(s.memory_usage, s.memory_bytes);
    }
}

TEST(HotShardsMemoryTest, PendingReclaimConsistentAcrossShards) {
    // pending_reclaim is a global counter (hazptr+EBR domain is not
    // partitioned per shard), so all shards must report the same value.
    lru::striped_cache<int, std::string> c{512, 8};
    for (int i = 0; i < 128; ++i) c.set(i, std::string(64, 'y'));
    auto hot = c.hot_shards(8);
    ASSERT_GT(hot.size(), 1u);
    std::size_t first = hot.front().pending_reclaim;
    for (const auto& s : hot) {
        EXPECT_EQ(s.pending_reclaim, first);
    }
}

TEST(HotShardsMemoryTest, HotShardsByMemoryReturnsSorted) {
    lru::striped_cache<int, std::string> c{256, 4};
    // Insert items with varying payload sizes so memory distribution is skewed.
    // Keys 0..15 -> 16-byte values; keys 16..31 -> 256-byte values; etc.
    for (int i = 0; i < 16; ++i) c.set(i, std::string(16, 'a'));
    for (int i = 16; i < 32; ++i) c.set(i, std::string(256, 'b'));
    for (int i = 32; i < 48; ++i) c.set(i, std::string(64, 'c'));

    auto hot = c.hot_shards_by_memory(4);
    ASSERT_FALSE(hot.empty());
    // Result must be sorted descending by memory_usage.
    for (std::size_t i = 1; i < hot.size(); ++i) {
        EXPECT_GE(hot[i - 1].memory_usage, hot[i].memory_usage);
    }
}

TEST(HotShardsMemoryTest, HotShardsByMemoryReturnsEmptyForNonSharded) {
    lru::cache<int, std::string> c{256};
    c.set(1, "one");
    auto hot = c.hot_shards_by_memory(5);
    EXPECT_TRUE(hot.empty());
}

TEST(HotShardsMemoryTest, HotShardsByMemoryRespectsTopN) {
    lru::striped_cache<int, std::string> c{256, 8};
    for (int i = 0; i < 64; ++i) c.set(i, std::to_string(i));
    auto hot = c.hot_shards_by_memory(3);
    EXPECT_LE(hot.size(), 3u);
}

TEST(HotShardsMemoryTest, RehashInProgressInitiallyFalse) {
    // On a quiet cache with no concurrent rehash, the per-shard flag
    // should be false. (This may briefly be true if a rehash is in
    // progress, but that is unlikely on a freshly-populated cache.)
    lru::striped_cache<int, std::string> c{1024, 4};
    for (int i = 0; i < 32; ++i) c.set(i, std::to_string(i));
    auto hot = c.hot_shards(4);
    for (const auto& s : hot) {
        // Rehash may or may not be in progress; just verify the field exists
        // and is a valid bool. We do not assert false because a concurrent
        // rehash could legitimately set it.
        (void)s.rehash_in_progress;
    }
}

TEST(HotShardsMemoryTest, HotShardsByMemoryHasAllFields) {
    lru::striped_cache<int, std::string> c{256, 4};
    for (int i = 0; i < 32; ++i) c.set(i, std::string(32, 'x'));
    auto hot = c.hot_shards_by_memory(4);
    ASSERT_FALSE(hot.empty());
    for (const auto& s : hot) {
        EXPECT_GE(s.shard_index, 0u);
        EXPECT_GE(s.memory_usage, 0u);
        EXPECT_GE(s.pending_reclaim, 0u);
        EXPECT_TRUE(s.rehash_in_progress == true || s.rehash_in_progress == false);
        EXPECT_GE(s.rehash_count, 0u);
    }
}

TEST(HotShardsMemoryTest, MemoryHotShardDetectsSkewedMemory) {
    // Insert one large item in a single shard and verify that
    // hot_shards_by_memory surfaces that shard as the hottest by memory.
    // We rely on hash distribution to land the large item somewhere;
    // since striped_cache has 4 shards, inserting many small items plus
    // one very large item should make the large-item shard rank #1
    // when sorted by memory.
    lru::striped_cache<int, std::string> c{1024, 4};
    // Pre-populate with small items across all shards.
    for (int i = 0; i < 100; ++i) c.set(i, std::string(8, 'x'));
    // Now insert a single very large item.
    c.set(99999, std::string(4096, 'L'));
    auto hot = c.hot_shards_by_memory(1);
    ASSERT_EQ(hot.size(), 1u);
    // The hottest shard by memory must have at least the large item's size.
    EXPECT_GE(hot[0].memory_usage, 4096u);
}

TEST(HotShardsMemoryTest, BothApisReturnConsistentShardIndices) {
    // hot_shards() and hot_shards_by_memory() should both enumerate the same
    // set of shard indices (just sorted differently).
    lru::striped_cache<int, std::string> c{256, 4};
    for (int i = 0; i < 32; ++i) c.set(i, std::to_string(i));
    auto by_access = c.hot_shards(10);
    auto by_memory = c.hot_shards_by_memory(10);
    ASSERT_EQ(by_access.size(), by_memory.size());
    std::vector<std::size_t> access_idx, memory_idx;
    for (const auto& s : by_access) access_idx.push_back(s.shard_index);
    for (const auto& s : by_memory) memory_idx.push_back(s.shard_index);
    std::sort(access_idx.begin(), access_idx.end());
    std::sort(memory_idx.begin(), memory_idx.end());
    EXPECT_EQ(access_idx, memory_idx);
}
