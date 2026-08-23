// SPDX-License-Identifier: MIT
// T-G14: EBR force-advance UAF scenario test.
//
// Verifies that the default kFailAdvance policy does NOT cause UAF
// when an EBR slot is stuck (thread parked indefinitely). Under the
// old kForceAdvanceAfter5s default, a stuck slot would be force-
// advanced, potentially reclaiming memory still referenced by the
// stuck thread → UAF. The new kFailAdvance default refuses to
// advance, leaving the pending objects un-reclaimed (memory grows
// but no UAF).
//
// We cannot actually park a thread indefinitely in a unit test, so
// this test verifies the policy configuration and the reclaim
// behavior under simulated slot contention. A true UAF test requires
// TSan + signal-based thread parking (covered by the stress suite).

#include "../lru.hpp"

#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

// Default policy for read_heavy_cache is kFailAdvance (T-G4).
TEST(EbrForceAdvanceUaf, ReadHeavyCacheDefaultsToFailAdvance) {
    lru::read_heavy_cache<int, std::string> cache(1'000);
    // The default force-advance policy should be kFailAdvance, NOT
    // kForceAdvanceAfter5s. kFailAdvance refuses to advance the epoch
    // when a slot is stuck, preventing UAF at the cost of un-reclaimed
    // memory.
    EXPECT_EQ(cache.epoch_force_advance_policy(),
              lru::detail::force_advance_policy::kFailAdvance);
}

// Default policy for production_cache is also kFailAdvance.
TEST(EbrForceAdvanceUaf, ProductionCacheDefaultsToFailAdvance) {
    lru::production_cache<int, std::string> cache(1'000);
    EXPECT_EQ(cache.epoch_force_advance_policy(),
              lru::detail::force_advance_policy::kFailAdvance);
}

// Explicit opt-in to kForceAdvanceAfter5s restores the old behavior.
TEST(EbrForceAdvanceUaf, CanOptInToForceAdvanceAfter5s) {
    lru::read_heavy_cache<int, std::string> cache(1'000);
    // Save original policy and restore on exit (policy lives on the
    // global default epoch_domain, so leakage would break later tests).
    const auto original = cache.epoch_force_advance_policy();
    cache.set_epoch_force_advance_policy(
        lru::detail::force_advance_policy::kForceAdvanceAfter5s);
    EXPECT_EQ(cache.epoch_force_advance_policy(),
              lru::detail::force_advance_policy::kForceAdvanceAfter5s);
    // Restore so subsequent tests see the default kFailAdvance.
    cache.set_epoch_force_advance_policy(original);
}

// Under kFailAdvance, sustained set+evict does not crash (no UAF).
// This is a smoke test — the real UAF scenario requires a stuck slot.
TEST(EbrForceAdvanceUaf, FailAdvanceNoCrashUnderEviction) {
    using namespace std::chrono_literals;
    lru::read_heavy_cache<int, std::string> cache(500);
    // Confirm kFailAdvance is active.
    ASSERT_EQ(cache.epoch_force_advance_policy(),
              lru::detail::force_advance_policy::kFailAdvance);

    // Fill and evict repeatedly — exercises the EBR retire/reclaim path.
    // If kFailAdvance had a UAF bug, this would crash under ASan/TSan.
    for (int round = 0; round < 50; ++round) {
        for (int i = 0; i < 1'000; ++i) {
            cache.set(round * 1'000 + i, std::to_string(i));
        }
        // cache evicts older items as it overflows (max_size=500).
    }

    // Verify the cache is still functional post-eviction.
    auto h = cache.try_get(49 * 1'000 + 999);
    EXPECT_TRUE(h.has_value());
    EXPECT_EQ(**h, std::to_string(999));
}

// Concurrent set+get under kFailAdvance: no crash, no corruption.
TEST(EbrForceAdvanceUaf, ConcurrentAccessUnderFailAdvance) {
    using namespace std::chrono_literals;
    lru::read_heavy_cache<int, std::string> cache(2'000);
    ASSERT_EQ(cache.epoch_force_advance_policy(),
              lru::detail::force_advance_policy::kFailAdvance);

    // Pre-populate.
    for (int i = 0; i < 1'000; ++i) {
        cache.set(i, std::to_string(i));
    }

    std::atomic<bool> stop{false};
    std::atomic<std::uint64_t> errors{0};

    auto reader = [&](unsigned int seed) {
        std::mt19937 rng(seed);
        std::uniform_int_distribution<int> dist(0, 1'999);
        while (!stop.load(std::memory_order_relaxed)) {
            int key = dist(rng);
            auto h = cache.try_get(key);
            if (h.has_value()) {
                // Value should match key (for keys < 1000) or be a
                // recently-set value (for keys >= 1000, set by writers).
                // We only check keys < 1000 for exact match.
                if (key < 1'000 && **h != std::to_string(key)) {
                    errors.fetch_add(1, std::memory_order_relaxed);
                }
            }
        }
    };

    auto writer = [&](unsigned int seed) {
        std::mt19937 rng(seed);
        std::uniform_int_distribution<int> dist(1'000, 2'999);
        while (!stop.load(std::memory_order_relaxed)) {
            int key = dist(rng);
            cache.set(key, std::to_string(key));
        }
    };

    std::vector<std::thread> threads;
    for (unsigned int i = 0; i < 4; ++i) threads.emplace_back(reader, i + 1);
    for (unsigned int i = 0; i < 2; ++i) threads.emplace_back(writer, i + 100);

    std::this_thread::sleep_for(300ms);
    stop.store(true, std::memory_order_relaxed);
    for (auto& t : threads) t.join();

    EXPECT_EQ(errors.load(), 0u);
}
