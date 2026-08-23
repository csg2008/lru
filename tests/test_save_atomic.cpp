// P2-3: save_atomic — strictly-atomic cross-shard snapshot tests.
//
// Validates that save_atomic():
//   - Produces a binary payload loadable via load_per_shard().
//   - Round-trips all key/value pairs across all shards.
//   - Leaves the source cache in shutdown state.
//   - Throws on handle-drain timeout when a handle is held across the call.
//   - Falls back to save() semantics for non-sharded caches.

#include <gtest/gtest.h>
#include "../lru.hpp"

#include <atomic>
#include <chrono>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace lru;
using namespace std::chrono_literals;

// ============================================================================
// Round-trip on a sharded cache
// ============================================================================
TEST(SaveAtomicTest, RoundTripsAllShards) {
    striped_cache<int, std::string> c(1024, 8);
    for (int i = 0; i < 200; ++i) {
        c.set(i, "v" + std::to_string(i));
    }
    EXPECT_EQ(c.size(), 200u);

    // save_atomic drains handles, shuts down, then snapshots.
    auto data = c.save_atomic(5s);
    EXPECT_FALSE(data.empty());
    EXPECT_TRUE(c.is_shutdown());

    // Load into a fresh cache.
    striped_cache<int, std::string> c2(1024, 8);
    c2.load_per_shard(data);
    EXPECT_EQ(c2.size(), 200u);
    for (int i = 0; i < 200; ++i) {
        auto h = c2.get(i);
        ASSERT_TRUE(h.has_value());
        EXPECT_EQ(*h, "v" + std::to_string(i));
    }
}

// ============================================================================
// Empty cache round-trip
// ============================================================================
TEST(SaveAtomicTest, EmptyCacheRoundTrip) {
    striped_cache<int, std::string> c(256, 4);
    auto data = c.save_atomic(5s);
    EXPECT_FALSE(data.empty());
    EXPECT_TRUE(c.is_shutdown());

    striped_cache<int, std::string> c2(256, 4);
    c2.load_per_shard(data);
    EXPECT_EQ(c2.size(), 0u);
}

// ============================================================================
// Leaves source in shutdown state
// ============================================================================
TEST(SaveAtomicTest, LeavesSourceShutdown) {
    striped_cache<int, int> c(256, 4);
    for (int i = 0; i < 32; ++i) c.set(i, i);
    ASSERT_FALSE(c.is_shutdown());
    (void)c.save_atomic(5s);
    EXPECT_TRUE(c.is_shutdown());
}

// ============================================================================
// Throws on handle-drain timeout
// ============================================================================
TEST(SaveAtomicTest, ThrowsOnHandleDrainTimeout) {
    striped_cache<int, std::string> c(256, 4);
    for (int i = 0; i < 16; ++i) c.set(i, "v" + std::to_string(i));

    // Hold a read_handle in another thread so shutdown_and_wait cannot
    // drain within the short timeout. The thread releases the handle
    // only after save_atomic has thrown, otherwise the destructor of the
    // handle would touch freed state.
    std::atomic<bool> hold_release{false};
    std::atomic<bool> handle_acquired{false};
    std::thread holder([&] {
        auto h = c.get(0);
        ASSERT_TRUE(h.has_value());
        handle_acquired.store(true, std::memory_order_release);
        while (!hold_release.load(std::memory_order_acquire)) {
            std::this_thread::sleep_for(2ms);
        }
        // h released on thread exit
    });

    // Wait for the holder to actually acquire the handle before timing
    // the save_atomic call.
    while (!handle_acquired.load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(1ms);
    }

    // 50ms timeout is far shorter than the holder will keep the handle.
    EXPECT_THROW(c.save_atomic(50ms), std::runtime_error);
    EXPECT_TRUE(c.is_shutdown());

    // Let the holder release its handle so the thread can join cleanly.
    hold_release.store(true, std::memory_order_release);
    holder.join();
}

// ============================================================================
// Non-sharded cache falls back to save()
// ============================================================================
TEST(SaveAtomicTest, NonShardedFallsBackToSave) {
    safe_cache<int, std::string> c(64);
    for (int i = 0; i < 32; ++i) c.set(i, "v" + std::to_string(i));

    auto atomic_data = c.save_atomic(5s);
    EXPECT_FALSE(atomic_data.empty());

    // Non-sharded save_atomic does NOT shut down the cache (it just
    // delegates to save(), which holds a brief read lock).
    // Document this explicitly so the test pins the contract.
    EXPECT_FALSE(c.is_shutdown());

    // Round-trip via load() into a fresh cache.
    safe_cache<int, std::string> c2(64);
    c2.load(atomic_data);
    EXPECT_EQ(c2.size(), 32u);
    for (int i = 0; i < 32; ++i) {
        auto h = c2.get(i);
        ASSERT_TRUE(h.has_value());
        EXPECT_EQ(*h, "v" + std::to_string(i));
    }
}

// ============================================================================
// production_cache alias also works
// ============================================================================
TEST(SaveAtomicTest, ProductionCacheAlias) {
    production_cache<int, std::string> c(2048);
    for (int i = 0; i < 256; ++i) c.set(i, "p" + std::to_string(i));

    auto data = c.save_atomic(5s);
    EXPECT_FALSE(data.empty());
    EXPECT_TRUE(c.is_shutdown());

    production_cache<int, std::string> c2(2048);
    c2.load_per_shard(data);
    EXPECT_EQ(c2.size(), 256u);
    for (int i = 0; i < 256; ++i) {
        auto h = c2.get(i);
        ASSERT_TRUE(h.has_value());
        EXPECT_EQ(*h, "p" + std::to_string(i));
    }
}
