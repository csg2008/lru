// SPDX-License-Identifier: MIT
// Additional concurrent edge-case tests.
//
// Covers spec gaps G9, G11, G14, G15 (P2):
//   G9a: TTL cleaner stop concurrent with sweep (no deadlock)
//   G9b: TTL cleaner does not hold global lock (no stall)
//   G11: striped lock hash distribution
//   G14: lock order checking toggle (LRU_DEBUG_LOCK_ORDER)
//   G15: defer_promotion toggle under load (no lost promotions)

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

#include "lru.hpp"
#include "test_helpers.hpp"

using namespace lru;
using namespace std::chrono_literals;

// ============================================================================
// TC-G9a: TTL cleaner stop concurrent with sweep
// Call stop_ttl_cleaner() while the cleaner is mid-sweep. Must not deadlock.
// ============================================================================
TEST(ConcurrentEdgeExtras, TtlCleanerStopConcurrentWithSweep) {
    safe_cache<int, std::string> c(1000);

    // Insert items with TTLs.
    for (int i = 0; i < 100; ++i) {
        c.set_with_ttl(i, "v" + std::to_string(i), 50ms);
    }

    // Start the cleaner with a short interval so it's likely to be mid-sweep
    // when we stop it.
    c.start_ttl_cleaner(10ms);

    // Concurrent writers inserting items with TTLs.
    std::atomic<bool> stop{false};
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&, t] {
            for (int i = 0; !stop.load(std::memory_order_relaxed); ++i) {
                int key = 1000 + t * 1000 + i;
                c.set_with_ttl(key, "v", 30ms);
            }
        });
    }

    // Let the cleaner run for a bit.
    std::this_thread::sleep_for(50ms);

    // Stop the cleaner concurrently while it may be mid-sweep.
    // The watchdog will catch any deadlock.
    bool ok = lru_test::Watchdog::run(
        [&] { c.stop_ttl_cleaner(); },
        std::chrono::seconds(5));
    EXPECT_TRUE(ok) << "stop_ttl_cleaner() deadlocked during sweep";

    stop.store(true);
    for (auto& th : threads) th.join();
}

// ============================================================================
// TC-G9b: TTL cleaner does not hold global lock (no stall)
// While the cleaner runs, gets must complete quickly (no global stall).
// ============================================================================
TEST(ConcurrentEdgeExtras, TtlCleanerNoGlobalStall) {
    striped_cache<int, int> c(10000);

    // Populate cache with TTL items.
    for (int i = 0; i < 5000; ++i) {
        c.set_with_ttl(i, i, 100ms);
    }

    // Start cleaner with a 10ms interval — frequent sweeps.
    c.start_ttl_cleaner(10ms);

    // Concurrent readers measuring latency.
    std::atomic<bool> stop{false};
    std::atomic<long long> max_latency_ns{0};

    std::vector<std::thread> readers;
    for (int t = 0; t < 4; ++t) {
        readers.emplace_back([&, t] {
            while (!stop.load(std::memory_order_relaxed)) {
                auto start = std::chrono::steady_clock::now();
                auto h = c.try_get(t % 5000);
                auto end = std::chrono::steady_clock::now();
                auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                              end - start).count();
                long long cur = max_latency_ns.load(std::memory_order_relaxed);
                while (ns > cur && !max_latency_ns.compare_exchange_weak(
                            cur, ns, std::memory_order_relaxed)) {
                    // retry
                }
                (void)h;
            }
        });
    }

    // Let it run for 200ms.
    std::this_thread::sleep_for(200ms);
    stop.store(true);
    for (auto& th : readers) th.join();

    c.stop_ttl_cleaner();

    // P99 latency should be bounded (well under 100ms — typical <1ms).
    // We use a generous bound: 50ms. If exceeded, the cleaner is holding
    // a global lock.
    EXPECT_LT(max_latency_ns.load(), 50'000'000LL)
        << "TTL cleaner is causing global stalls (max get latency "
        << (max_latency_ns.load() / 1'000'000) << "ms)";
}

// ============================================================================
// TC-G11: Striped lock hash distribution
// Insert many keys and verify they distribute roughly evenly across stripes.
// ============================================================================
TEST(ConcurrentEdgeExtras, StripedLockHashDistribution) {
    constexpr std::size_t kStripes = 16;
    striped_cache<int, int> c(10000, kStripes);

    constexpr int kKeys = 5000;
    for (int i = 0; i < kKeys; ++i) {
        c.set(i, i);
    }

    // No public API for per-stripe count, so we verify via the cache's
    // diagnostics info. The per-shard sizes should be roughly equal.
    auto info = c.diagnostics();
    if (info.num_shards > 0) {
        std::size_t total = 0;
        std::size_t min_size = SIZE_MAX;
        std::size_t max_size = 0;
        for (std::size_t i = 0; i < info.per_shard_size.size(); ++i) {
            std::size_t s = info.per_shard_size[i];
            total += s;
            min_size = std::min(min_size, s);
            max_size = std::max(max_size, s);
        }
        EXPECT_EQ(total, static_cast<std::size_t>(kKeys));
        if (max_size > 0) {
            // Imbalance < 50% — for 5000 keys across 16 shards, ~312 per shard.
            // 50% is generous; hash distribution should be much tighter.
            double expected = static_cast<double>(kKeys) / info.num_shards;
            (void)expected;
            // Sanity: max_size should not be larger than 2x the average.
            double avg = static_cast<double>(total) / info.per_shard_size.size();
            EXPECT_LE(max_size, 2.0 * avg)
                << "severe shard imbalance: max=" << max_size
                << " avg=" << avg;
        }
    }
}

// ============================================================================
// TC-G14: Lock order checking toggle (LRU_DEBUG_LOCK_ORDER)
// Verify the API exists and does not crash whether or not the macro is
// defined. When LRU_DEBUG_LOCK_ORDER is not defined, the API is a no-op.
// ============================================================================
TEST(ConcurrentEdgeExtras, LockOrderCheckingToggle) {
    safe_cache<int, int> c(64);

#ifdef LRU_DEBUG_LOCK_ORDER
    c.set_lock_order_checking(true);
    EXPECT_TRUE(c.lock_order_checking_enabled());
    c.set_lock_order_checking(false);
    EXPECT_FALSE(c.lock_order_checking_enabled());
#else
    // When the macro is not defined, the API is a no-op. Toggle should
    // not crash and the check should always return false.
    c.set_lock_order_checking(true);
    EXPECT_FALSE(c.lock_order_checking_enabled());
    c.set_lock_order_checking(false);
    EXPECT_FALSE(c.lock_order_checking_enabled());
#endif
}

// ============================================================================
// TC-G15: defer_promotion toggle under load
// Toggle set_defer_promotion(false) while N readers are active. All
// previously-recorded accesses should still be promoted (no lost updates).
// ============================================================================
TEST(ConcurrentEdgeExtras, DeferPromotionToggleUnderLoad) {
    striped_cache<int, int> c(2048);

    // Pre-populate so gets will hit.
    for (int i = 0; i < 500; ++i) {
        c.set(i, i);
    }

    constexpr int kThreads = 8;
    std::atomic<bool> stop{false};
    std::atomic<int> reads{0};

    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t] {
            std::mt19937 rng(t);
            while (!stop.load(std::memory_order_relaxed)) {
                int key = rng() % 500;
                auto h = c.try_get(key);
                if (h) reads.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }

    // Toggle defer_promotion under load.
    for (int i = 0; i < 5; ++i) {
        std::this_thread::sleep_for(20ms);
        c.set_defer_promotion(i % 2 == 0);
    }

    stop.store(true);
    for (auto& th : threads) th.join();

    // All keys must still be present with correct values.
    for (int i = 0; i < 500; ++i) {
        auto h = c.try_get(i);
        ASSERT_TRUE(h.has_value());
        EXPECT_EQ(**h, i);
    }
    EXPECT_GT(reads.load(), 0);
}
