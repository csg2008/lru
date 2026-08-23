// SPDX-License-Identifier: MIT
// T-G14: 128-thread read-heavy soak test.
//
// Verifies the cache remains correct (no race, no crash, no data
// corruption) under sustained 99% read / 1% write load from 128
// threads. The test auto-SKIPs on machines with fewer than 16
// logical cores (128 threads on <16 cores would thrash the scheduler
// rather than stress the cache).
//
// Duration is configurable via LRU_STRESS_DURATION_MS (default 5min
// for CI soak jobs, 5s for default test runs). The CI workflow can
// set LRU_STRESS_DURATION_MS=300000 for the long soak job.

#include "../lru.hpp"

#include <atomic>
#include <chrono>
#include <random>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "test_helpers.hpp"

namespace {

unsigned int hardware_concurrency() {
    unsigned int n = std::thread::hardware_concurrency();
    return n == 0 ? 1 : n;
}

} // namespace

// 128 threads, 99% reads, 1% writes, 5min default (configurable).
// SKIP if <16 logical cores — 128 threads on fewer cores thrashes
// the scheduler rather than stressing the cache.
TEST(OneTwentyEightThreadsSoak, ReadHeavyNoCorruption) {
    constexpr unsigned int kMinCores = 16;
    constexpr unsigned int kNumThreads = 128;
    if (hardware_concurrency() < kMinCores) {
        GTEST_SKIP() << "Skipping 128-thread soak: only "
                     << hardware_concurrency()
                     << " cores available (need >= " << kMinCores << ")";
    }

    // Default 5s for regular test runs; CI soak job sets 300000 (5min).
    const auto duration = lru_test::read_stress_duration_ms(
        std::chrono::milliseconds(5000));

    lru::production_cache<int, std::string> cache(100'000);
    // Pre-populate so reads have something to hit.
    for (int i = 0; i < 10'000; ++i) {
        cache.set(i, std::to_string(i));
    }

    std::atomic<bool> stop{false};
    std::atomic<std::uint64_t> total_reads{0};
    std::atomic<std::uint64_t> total_writes{0};
    std::atomic<std::uint64_t> errors{0};

    auto worker = [&](unsigned int seed) {
        std::mt19937 rng(seed);
        std::uniform_int_distribution<int> key_dist(0, 9'999);
        std::uniform_int_distribution<int> op_dist(0, 99);  // 99% read

        std::uint64_t local_reads = 0;
        std::uint64_t local_writes = 0;

        while (!stop.load(std::memory_order_relaxed)) {
            int key = key_dist(rng);
            if (op_dist(rng) < 99) {
                // Read path.
                auto h = cache.try_get(key);
                if (h.has_value()) {
                    // Verify value matches key (no corruption).
                    // **h: first * dereferences the optional, second *
                    // dereferences the read_handle to get the value.
                    if (**h != std::to_string(key)) {
                        errors.fetch_add(1, std::memory_order_relaxed);
                    }
                }
                ++local_reads;
            } else {
                // Write path.
                cache.set(key, std::to_string(key));
                ++local_writes;
            }
        }

        total_reads.fetch_add(local_reads, std::memory_order_relaxed);
        total_writes.fetch_add(local_writes, std::memory_order_relaxed);
    };

    std::vector<std::thread> threads;
    threads.reserve(kNumThreads);
    for (unsigned int i = 0; i < kNumThreads; ++i) {
        threads.emplace_back(worker, i + 1);
    }

    std::this_thread::sleep_for(duration);
    stop.store(true, std::memory_order_relaxed);

    for (auto& t : threads) t.join();

    // Sanity: at least some operations should have completed.
    EXPECT_GT(total_reads.load(), 0u);
    EXPECT_GT(total_writes.load(), 0u);
    // No data corruption tolerated.
    EXPECT_EQ(errors.load(), 0u);

    // Verify cache integrity post-soak: every key that was set should
    // be retrievable with the correct value.
    for (int i = 0; i < 10'000; ++i) {
        auto h = cache.try_get(i);
        ASSERT_TRUE(h.has_value()) << "key " << i << " missing post-soak";
        EXPECT_EQ(**h, std::to_string(i)) << "key " << i << " corrupted post-soak";
    }
}
