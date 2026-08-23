// SPDX-License-Identifier: MIT
// T-G14: ttl_cache + external cleaner concurrent correctness.
//
// Verifies that the TTL cleaner (either background or manual
// evict_expired_now()) runs correctly under concurrent set/get
// traffic without races, missed evictions, or crashes.

#include "../lru.hpp"

#include <atomic>
#include <chrono>
#include <random>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "test_helpers.hpp"

// Concurrent set + TTL expiry + background cleaner: no crash, no race.
TEST(TtlCacheConcurrentCleanup, BackgroundCleanerWithConcurrentTraffic) {
    using namespace std::chrono_literals;
    // striped_cache with native TTL support (sharded_mm_lru).
    lru::striped_cache<int, std::string> cache(10'000, 4);
    cache.start_ttl_cleaner(50ms);

    std::atomic<bool> stop{false};
    std::atomic<std::uint64_t> errors{0};

    auto worker = [&](unsigned int seed) {
        std::mt19937 rng(seed);
        std::uniform_int_distribution<int> key_dist(0, 999);

        while (!stop.load(std::memory_order_relaxed)) {
            int key = key_dist(rng);
            // Mix of set-with-TTL, set-no-TTL, and get.
            int op = rng() % 3;
            if (op == 0) {
                cache.set_with_ttl(key, std::to_string(key), 30ms);
            } else if (op == 1) {
                cache.set(key, std::to_string(key));
            } else {
                auto h = cache.try_get(key);
                if (h.has_value()) {
                    if (**h != std::to_string(key)) {
                        errors.fetch_add(1, std::memory_order_relaxed);
                    }
                }
            }
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(8);
    for (unsigned int i = 0; i < 8; ++i) {
        threads.emplace_back(worker, i + 1);
    }

    // Run for 500ms — enough for many TTL expiry cycles at 50ms interval.
    std::this_thread::sleep_for(500ms);
    stop.store(true, std::memory_order_relaxed);
    for (auto& t : threads) t.join();

    cache.stop_ttl_cleaner();
    EXPECT_EQ(errors.load(), 0u);
}

// Manual evict_expired_now() under concurrent traffic: processes ALL
// shards in one call (T-G17: round-robin is for background cleaner only).
TEST(TtlCacheConcurrentCleanup, ManualEvictExpiredNowProcessesAllShards) {
    using namespace std::chrono_literals;
    lru::striped_cache<int, std::string> cache(10'000, 8);

    // Set items with short TTL across all shards.
    for (int i = 0; i < 1'000; ++i) {
        cache.set_with_ttl(i, std::to_string(i), 50ms);
    }
    EXPECT_EQ(cache.size(), 1'000u);

    // Wait for expiry.
    std::this_thread::sleep_for(100ms);
    cache.refresh_cached_now();

    // evict_expired_now() should process ALL shards (not round-robin),
    // evicting every expired item in a single call.
    std::size_t evicted = cache.evict_expired_now();
    EXPECT_EQ(evicted, 1'000u);
    EXPECT_EQ(cache.size(), 0u);
}

// T-G17: Background cleaner round-robin processes ONE shard per cycle.
// Over N cycles (N = num_shards), all expired items are eventually evicted.
TEST(TtlCacheConcurrentCleanup, BackgroundRoundRobinEventuallyEvictsAll) {
    using namespace std::chrono_literals;
    lru::striped_cache<int, std::string> cache(10'000, 4);

    // Verify round-robin is enabled by default.
    EXPECT_TRUE(cache.is_ttl_cleaner_round_robin_enabled());

    // Set items with TTL across all shards.
    for (int i = 0; i < 1'000; ++i) {
        cache.set_with_ttl(i, std::to_string(i), 50ms);
    }

    // Wait for expiry.
    std::this_thread::sleep_for(100ms);

    // Start background cleaner with short interval so all shards are
    // covered quickly. With 4 shards and 10ms interval, all shards
    // are processed within ~40ms.
    cache.start_ttl_cleaner(10ms);
    std::this_thread::sleep_for(200ms);
    cache.stop_ttl_cleaner();

    // All items should have been evicted by the round-robin cleaner.
    EXPECT_EQ(cache.size(), 0u);
}

// T-G17: Disabling round-robin restores "scan all shards per cycle".
TEST(TtlCacheConcurrentCleanup, DisableRoundRobinScansAllShardsPerCycle) {
    using namespace std::chrono_literals;
    lru::striped_cache<int, std::string> cache(10'000, 8);
    cache.set_ttl_cleaner_round_robin(false);
    EXPECT_FALSE(cache.is_ttl_cleaner_round_robin_enabled());

    for (int i = 0; i < 1'000; ++i) {
        cache.set_with_ttl(i, std::to_string(i), 50ms);
    }
    std::this_thread::sleep_for(100ms);
    cache.refresh_cached_now();

    // With round-robin disabled, a single evict_expired_now() (which
    // passes round_robin=false) processes all shards. But we want to
    // test the background cleaner path — start it with short interval
    // and verify all items are evicted in one cycle.
    cache.start_ttl_cleaner(10ms);
    std::this_thread::sleep_for(50ms);
    cache.stop_ttl_cleaner();

    EXPECT_EQ(cache.size(), 0u);
}
