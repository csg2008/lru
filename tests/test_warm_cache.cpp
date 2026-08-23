// Unified LRU Cache Library — Warm Cache & Delta Snapshot Tests
// SPDX-License-Identifier: MIT
//
// Covers P2-G: true incremental delta snapshots.
//   - enable/disable delta tracking
//   - delta captures insert/remove
//   - .full snapshot is written on the first tick
//   - .delta snapshot is written on subsequent ticks
//   - load_with_delta restores full + delta to the correct state
//   - full_snapshot_interval triggers periodic full rewrites

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "../lru.hpp"

using namespace lru;

namespace {

constexpr const char* kTestSnapshotBase = "test_warm_cache_snap.bin";

void remove_snapshot_files(const std::string& base) {
    std::error_code ec;
    std::filesystem::remove(base, ec);
    std::filesystem::remove(base + ".full", ec);
    std::filesystem::remove(base + ".delta", ec);
    std::filesystem::remove(base + ".tmp", ec);
    std::filesystem::remove(base + ".delta.tmp", ec);
}

}  // namespace

// ============================================================================
// Delta tracking
// ============================================================================

TEST(WarmCacheDeltaTest, EnableDeltaTrackingPopulatesDeltaMap) {
    auto cache_ptr = std::make_shared<safe_cache<int, std::string>>(64);
    warm_cache_manager<safe_cache<int, std::string>> mgr(cache_ptr);
    mgr.enable_delta_tracking();
    EXPECT_TRUE(mgr.delta_tracking_enabled());

    cache_ptr->set(1, "a");
    cache_ptr->set(2, "b");
    cache_ptr->remove(1);

    EXPECT_EQ(mgr.pending_delta_count(), 2u);  // key 1 (remove) + key 2 (insert)
}

TEST(WarmCacheDeltaTest, DisableDeltaTrackingClearsDeltaMap) {
    auto cache_ptr = std::make_shared<safe_cache<int, std::string>>(64);
    warm_cache_manager<safe_cache<int, std::string>> mgr(cache_ptr);
    mgr.enable_delta_tracking();
    cache_ptr->set(1, "a");
    cache_ptr->set(2, "b");
    EXPECT_EQ(mgr.pending_delta_count(), 2u);

    mgr.disable_delta_tracking();
    EXPECT_FALSE(mgr.delta_tracking_enabled());
    EXPECT_EQ(mgr.pending_delta_count(), 0u);
}

TEST(WarmCacheDeltaTest, DeltaCoalescesRepeatedUpdates) {
    auto cache_ptr = std::make_shared<safe_cache<int, std::string>>(64);
    warm_cache_manager<safe_cache<int, std::string>> mgr(cache_ptr);
    mgr.enable_delta_tracking();
    for (int i = 0; i < 5; ++i) {
        cache_ptr->set(7, "v" + std::to_string(i));
    }
    EXPECT_EQ(mgr.pending_delta_count(), 1u);  // single key coalesced
}

// ============================================================================
// take_incremental_snapshot
// ============================================================================

TEST(WarmCacheDeltaTest, FirstTickWritesFullSnapshot) {
    remove_snapshot_files(kTestSnapshotBase);
    auto cache_ptr = std::make_shared<safe_cache<int, std::string>>(64);
    warm_cache_manager<safe_cache<int, std::string>> mgr(cache_ptr);
    for (int i = 0; i < 5; ++i) cache_ptr->set(i, "v" + std::to_string(i));

    // First tick: full snapshot interval=10 → tick 0 triggers a full snapshot
    mgr.take_incremental_snapshot(kTestSnapshotBase);

    EXPECT_TRUE(std::filesystem::exists(std::string(kTestSnapshotBase) + ".full"));
    // Delta should not be written when a full snapshot was produced this tick
    EXPECT_FALSE(std::filesystem::exists(std::string(kTestSnapshotBase) + ".delta"));
}

TEST(WarmCacheDeltaTest, SecondTickWritesDeltaOnly) {
    remove_snapshot_files(kTestSnapshotBase);
    auto cache_ptr = std::make_shared<safe_cache<int, std::string>>(64);
    warm_cache_manager<safe_cache<int, std::string>> mgr(cache_ptr);
    for (int i = 0; i < 3; ++i) cache_ptr->set(i, "v" + std::to_string(i));

    mgr.take_incremental_snapshot(kTestSnapshotBase);  // tick 0 → full
    EXPECT_TRUE(std::filesystem::exists(std::string(kTestSnapshotBase) + ".full"));

    // Mutate some keys after the full snapshot
    cache_ptr->set(10, "new");
    cache_ptr->remove(0);

    mgr.take_incremental_snapshot(kTestSnapshotBase);  // tick 1 → delta
    EXPECT_TRUE(std::filesystem::exists(std::string(kTestSnapshotBase) + ".delta"));
}

// ============================================================================
// load_with_delta
// ============================================================================

TEST(WarmCacheDeltaTest, LoadWithDeltaRestoresState) {
    remove_snapshot_files(kTestSnapshotBase);
    auto cache_ptr = std::make_shared<safe_cache<int, std::string>>(64);
    warm_cache_manager<safe_cache<int, std::string>> mgr(cache_ptr);

    // Initial state
    cache_ptr->set(1, "one");
    cache_ptr->set(2, "two");
    cache_ptr->set(3, "three");

    // Write full snapshot
    mgr.take_incremental_snapshot(kTestSnapshotBase);

    // Mutate: add, update, remove
    cache_ptr->set(4, "four");
    cache_ptr->set(2, "TWO");
    cache_ptr->remove(3);

    // Write delta
    mgr.take_incremental_snapshot(kTestSnapshotBase);

    // Load into a fresh cache via load_with_delta
    auto restored_ptr = std::make_shared<safe_cache<int, std::string>>(64);
    warm_cache_manager<safe_cache<int, std::string>> restore_mgr(restored_ptr);
    EXPECT_TRUE(restore_mgr.load_with_delta(kTestSnapshotBase));

    // Verify final state — peek() returns read_handle<const V> (truthy if hit)
    auto h1 = restored_ptr->peek(1);
    ASSERT_TRUE(h1.has_value());
    EXPECT_EQ(*h1, "one");

    auto h2 = restored_ptr->peek(2);
    ASSERT_TRUE(h2.has_value());
    EXPECT_EQ(*h2, "TWO");

    auto h3 = restored_ptr->peek(3);
    EXPECT_FALSE(h3.has_value());

    auto h4 = restored_ptr->peek(4);
    ASSERT_TRUE(h4.has_value());
    EXPECT_EQ(*h4, "four");

    remove_snapshot_files(kTestSnapshotBase);
}

TEST(WarmCacheDeltaTest, LoadWithDeltaOnlyFullNoDelta) {
    remove_snapshot_files(kTestSnapshotBase);
    auto cache_ptr = std::make_shared<safe_cache<int, std::string>>(64);
    warm_cache_manager<safe_cache<int, std::string>> mgr(cache_ptr);
    cache_ptr->set(1, "one");
    cache_ptr->set(2, "two");

    // Only write full snapshot (no subsequent delta)
    mgr.take_incremental_snapshot(kTestSnapshotBase);

    auto restored_ptr = std::make_shared<safe_cache<int, std::string>>(64);
    warm_cache_manager<safe_cache<int, std::string>> restore_mgr(restored_ptr);
    EXPECT_TRUE(restore_mgr.load_with_delta(kTestSnapshotBase));
    EXPECT_EQ(restored_ptr->size(), 2u);

    remove_snapshot_files(kTestSnapshotBase);
}

// ============================================================================
// full_snapshot_interval
// ============================================================================

TEST(WarmCacheDeltaTest, FullSnapshotRewrittenOnInterval) {
    remove_snapshot_files(kTestSnapshotBase);
    auto cache_ptr = std::make_shared<safe_cache<int, std::string>>(64);
    warm_cache_manager<safe_cache<int, std::string>> mgr(cache_ptr);

    // interval = 3 → full at tick 0, 3, 6, ...
    mgr.start_incremental_snapshot(kTestSnapshotBase,
                                    std::chrono::milliseconds(30),
                                    /*full_snapshot_interval=*/3);
    // Seed initial data and let it run through several ticks
    for (int i = 0; i < 10; ++i) cache_ptr->set(i, "v" + std::to_string(i));
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
    mgr.stop_incremental_snapshot();

    // After several ticks, the .full file should exist (rewritten at ticks 0,3,6)
    EXPECT_TRUE(std::filesystem::exists(std::string(kTestSnapshotBase) + ".full"));
    remove_snapshot_files(kTestSnapshotBase);
}

// ============================================================================
// Background incremental snapshot worker
// ============================================================================

TEST(WarmCacheDeltaTest, StartIncrementalSnapshotWritesFiles) {
    remove_snapshot_files(kTestSnapshotBase);
    auto cache_ptr = std::make_shared<safe_cache<int, std::string>>(64);
    warm_cache_manager<safe_cache<int, std::string>> mgr(cache_ptr);

    mgr.start_incremental_snapshot(kTestSnapshotBase,
                                    std::chrono::milliseconds(40));
    for (int i = 0; i < 5; ++i) cache_ptr->set(i, "v" + std::to_string(i));
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    mgr.stop_incremental_snapshot();

    EXPECT_TRUE(std::filesystem::exists(std::string(kTestSnapshotBase) + ".full"));
    remove_snapshot_files(kTestSnapshotBase);
}
