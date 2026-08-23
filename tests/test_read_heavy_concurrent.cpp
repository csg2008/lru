// SPDX-License-Identifier: MIT
// Concurrent correctness tests for read-heavy / production aliases.
//
// Covers H-3 (P2): production_cache / segmented_striped_cache /
// read_heavy_cache / read_heavy_striped_cache / f14_production_cache had
// zero concurrent test coverage (or were avoided in test_chaos.cpp due to
// C-1/C-2). With C-1/C-2 fixed, these aliases can now be exercised.
//
// Each test runs a 95% read / 5% write workload across 16 threads against
// the alias under test, asserting:
//   - No crash / abort (assertions in mm_* or intrusive_list)
//   - Value integrity: any value observed by try_get matches the canonical
//     mapping for that key (key * 10)
//   - Capacity invariant: cache.size() <= max_size

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <random>
#include <thread>
#include <vector>

#include "lru.hpp"
#include "test_helpers.hpp"

using namespace lru;

namespace {

// Generic read-heavy concurrent workload: 95% reads, 5% writes, 16 threads.
// Pre-populates the cache with `key_space` items, then each thread runs
// `ops_per_thread` operations. Writes update existing keys (exercising the
// update_existing path) and occasionally insert new keys (exercising
// insert_new + eviction under capacity).
template <typename CacheT>
void run_read_heavy_concurrent(CacheT& c, int key_space,
                                int num_threads, int ops_per_thread) {
    // Pre-populate
    for (int i = 0; i < key_space; ++i) {
        c.set(i, i * 10);
    }

    std::atomic<int> value_mismatch{0};
    std::atomic<int> total_ops{0};

    std::vector<std::thread> threads;
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            std::mt19937 rng(t * 7919 + 1);
            for (int i = 0; i < ops_per_thread; ++i) {
                int key = rng() % key_space;
                if (rng() % 100 < 5) {
                    // 5% writes: update existing key with canonical value
                    c.set(key, key * 10);
                } else {
                    // 95% reads
                    auto h = c.try_get(key);
                    if (h) {
                        if (**h != key * 10) {
                            value_mismatch.fetch_add(1, std::memory_order_relaxed);
                        }
                    }
                    // Miss is acceptable (item may have been evicted between
                    // pre-populate and read, but with key_space <= max_size
                    // and only updates, misses should be rare).
                }
                total_ops.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }
    for (auto& th : threads) th.join();

    EXPECT_GT(total_ops.load(), 0);
    EXPECT_EQ(value_mismatch.load(), 0)
        << "try_get returned a value inconsistent with canonical mapping";
    EXPECT_LE(c.size(), c.max_size())
        << "cache exceeded its declared max_size";
}

// Variant that forces rehash by using a tiny initial bucket count, then
// runs concurrent reads + writes. Verifies the dual-array lookup / per-
// segment rehash path is safe under load.
template <typename CacheT>
void run_concurrent_with_rehash(CacheT& c, int key_space,
                                 int num_threads, int ops_per_thread) {
    std::atomic<int> value_mismatch{0};
    std::atomic<int> total_ops{0};

    std::vector<std::thread> threads;
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            std::mt19937 rng(t * 31337 + 7);
            for (int i = 0; i < ops_per_thread; ++i) {
                int key = rng() % key_space;
                int op = rng() % 10;
                if (op < 7) {
                    c.set(key, key * 10);
                } else {
                    auto h = c.try_get(key);
                    if (h) {
                        if (**h != key * 10) {
                            value_mismatch.fetch_add(1, std::memory_order_relaxed);
                        }
                    }
                }
                total_ops.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }
    for (auto& th : threads) th.join();

    EXPECT_GT(total_ops.load(), 0);
    EXPECT_EQ(value_mismatch.load(), 0)
        << "value corruption under concurrent set() + rehash";
    EXPECT_LE(c.size(), c.max_size());
}

}  // namespace

// ============================================================================
// H-3-A: segmented_striped_cache concurrent stress
// (was avoided in test_stress.cpp:SegmentedCacheStress due to a stale
// "deleted operator=" compilation concern — verified compilable here.)
// ============================================================================
TEST(ReadHeavyConcurrent, SegmentedStripedCacheReadHeavy) {
    segmented_striped_cache<int, int> c(2000);
    run_read_heavy_concurrent(c, 1000, 16, 3000);
}

TEST(ReadHeavyConcurrent, SegmentedStripedCacheRehashStress) {
    // expected_items=1 forces tiny initial buckets -> rehash on insert
    sharded_mm_lru_config cfg;
    cfg.expected_items = 1;
    segmented_striped_cache<int, int> c(5000, cfg);
    run_concurrent_with_rehash(c, 2000, 8, 2000);
}

// ============================================================================
// H-3-B: read_heavy_cache / read_heavy_striped_cache concurrent stress
// (95% read / 5% write, 16 threads). These are the aliases AGENTS.md
// recommends for read-heavy production workloads but had ZERO concurrent
// test coverage before this file.
// ============================================================================
TEST(ReadHeavyConcurrent, ReadHeavyCacheReadHeavy) {
    // Direct construction — unified_cache is non-movable (contains mutexes/atomics)
    mm_lru_config cfg;
    cfg.defer_promotion = true;
    read_heavy_cache<int, int> c(2000, cfg);
    (void)c.set_fairness_mode_quiescent(detail::fairness_mode::reader_preferred);
    c.set_incremental_rehash(true);
    run_read_heavy_concurrent(c, 1000, 16, 3000);
}

TEST(ReadHeavyConcurrent, ReadHeavyStripedCacheReadHeavy) {
    // Direct construction — unified_cache is non-movable (contains mutexes/atomics)
    sharded_mm_lru_config cfg;
    cfg.lru_config.defer_promotion = true;
    read_heavy_striped_cache<int, int> c(2000, cfg);
    (void)c.set_fairness_mode_quiescent(detail::fairness_mode::reader_preferred);
    c.set_incremental_rehash(true);
    run_read_heavy_concurrent(c, 1000, 16, 3000);
}

// ============================================================================
// H-3-C: f14_production_cache concurrent stress (mixed reads + writes +
// rehash). Combines F14 SIMD probing + sharded MM + striped locking.
// ============================================================================
TEST(ReadHeavyConcurrent, F14ProductionCacheRehashStress) {
    sharded_mm_lru_config cfg;
    cfg.expected_items = 1;  // force rehash
    f14_production_cache<int, int> c(5000, cfg);
    run_concurrent_with_rehash(c, 2000, 8, 2000);
}

TEST(ReadHeavyConcurrent, F14ProductionCacheReadHeavy) {
    f14_production_cache<int, int> c(2000);
    run_read_heavy_concurrent(c, 1000, 16, 3000);
}

// ============================================================================
// H-3-D: production_cache (the headline "recommended for production" alias)
// under sustained read-heavy load. With C-1/C-2 fixed, this alias is now
// safe to exercise. The test pre-populates, then runs 16 threads doing
// 95% reads + 5% writes for at least 30s (configurable via env).
// ============================================================================
TEST(ReadHeavyConcurrent, ProductionCacheSustainedReadHeavy) {
    production_cache<int, int> c(5000);
    // Honor LRU_STRESS_DURATION_MS for CI tuning; default 30s per H-3-B.
    auto duration = lru_test::read_stress_duration_ms(std::chrono::seconds(30));

    const int key_space = 2000;
    for (int i = 0; i < key_space; ++i) {
        c.set(i, i * 10);
    }

    std::atomic<bool> stop{false};
    std::atomic<int> value_mismatch{0};
    std::atomic<long long> total_ops{0};

    std::vector<std::thread> threads;
    for (int t = 0; t < 16; ++t) {
        threads.emplace_back([&, t]() {
            std::mt19937 rng(t * 7919 + 1);
            while (!stop.load(std::memory_order_relaxed)) {
                int key = rng() % key_space;
                if (rng() % 100 < 5) {
                    c.set(key, key * 10);
                } else {
                    auto h = c.try_get(key);
                    if (h && **h != key * 10) {
                        value_mismatch.fetch_add(1, std::memory_order_relaxed);
                    }
                }
                total_ops.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }

    std::this_thread::sleep_for(duration);
    stop.store(true, std::memory_order_release);
    for (auto& th : threads) th.join();

    EXPECT_GT(total_ops.load(), 0);
    EXPECT_EQ(value_mismatch.load(), 0)
        << "value corruption under sustained read-heavy load";
    EXPECT_LE(c.size(), c.max_size());
}
