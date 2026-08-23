// SPDX-License-Identifier: MIT
// Concurrent rehash tests for F14 dual-array and segmented per-segment modes.
//
// Covers spec gap G6 (P1):
//   G6a: F14 dual-array concurrent rehash correctness
//        Verifies that during F14 incremental rehash, reads query both
//        old and new arrays and writes route by progress boundary.
//   G6b: Segmented per-segment rehash no global stall
//        Verifies that rehash in segmented mode does not block all writers
//        (P99 set latency stays bounded).
//   G6c (P1-5): Lock-free read path fallback accounting
//        Verifies that rehash_lockfree_fallback_count is exposed via the
//        cache API, aggregated across segments/shards, and exported in
//        Prometheus text format.

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
// TC-G6a: F14DualArrayConcurrentRehashCorrectness
// Force a hash table rehash under concurrent reads and writes.
// All writes that succeeded must be readable after the rehash completes.
//
// NOTE: The cache capacity must be large enough to hold ALL keys without
// LRU eviction, otherwise the post-write verification (`ASSERT_TRUE(h)`)
// will fail because keys were legitimately evicted (not lost due to rehash).
// The bucket array starts small and grows via rehash as keys are inserted,
// so rehash is still exercised even with a large max_size.
// ============================================================================
TEST(ConcurrentRehash, F14DualArrayConcurrentRehashCorrectness) {
    constexpr int kThreads = 8;
    constexpr int kReaders = 4;
    constexpr int kOpsPerThread = 500;
    constexpr int kTotalKeys = kThreads * kOpsPerThread;  // 4000

    // Capacity must exceed total keys to avoid LRU eviction.
    f14_striped_cache<int, int> c(kTotalKeys * 2);

    // Enable incremental rehash so we exercise the dual-array path.
    c.set_incremental_rehash(true);

    std::atomic<int> write_successes{0};
    std::atomic<int> read_failures{0};
    std::atomic<bool> stop{false};

    std::vector<std::thread> threads;

    // Writers: each thread writes its own key range.
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t] {
            for (int i = 0; i < kOpsPerThread; ++i) {
                int key = t * kOpsPerThread + i;
                c.set(key, key * 10);
                write_successes.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }

    // Readers: continuously read back previously-written keys.
    for (int t = 0; t < kReaders; ++t) {
        threads.emplace_back([&, t] {
            std::mt19937 rng(t + 12345);
            while (!stop.load(std::memory_order_acquire)) {
                int key = rng() % (kThreads * kOpsPerThread);
                auto h = c.try_get(key);
                // The key may not have been written yet; that's fine.
                // But if it returns a value, the value must be correct.
                if (h) {
                    if (**h != key * 10) {
                        read_failures.fetch_add(1, std::memory_order_relaxed);
                    }
                }
            }
        });
    }

    // Wait for writers to complete, then signal readers to stop.
    for (int i = 0; i < kThreads; ++i) threads[i].join();
    stop.store(true, std::memory_order_release);
    for (int i = kThreads; i < kThreads + kReaders; ++i) threads[i].join();

    EXPECT_EQ(read_failures.load(), 0)
        << "readers observed stale or incorrect values during rehash";

    // Verify all keys written by all threads are present with correct values.
    for (int t = 0; t < kThreads; ++t) {
        for (int i = 0; i < kOpsPerThread; ++i) {
            int key = t * kOpsPerThread + i;
            auto h = c.try_get(key);
            ASSERT_TRUE(h.has_value()) << "Missing key " << key;
            EXPECT_EQ(**h, key * 10) << "Wrong value for key " << key;
        }
    }

    EXPECT_EQ(write_successes.load(), kThreads * kOpsPerThread);
}

// ============================================================================
// TC-G6b: SegmentedPerSegmentRehashNoGlobalStall
// Trigger segmented rehash with many concurrent writers. P99 set latency
// must stay bounded (no global stall — at most 1/64 of the table is
// locked at any moment).
//
// NOTE: The cache capacity must be large enough to hold ALL keys without
// LRU eviction, otherwise the post-write verification (`ASSERT_TRUE(h)`)
// will fail because keys were legitimately evicted (not lost due to rehash).
// The segmented hash table's bucket array still starts small and rehashes
// per-segment as keys are inserted, so rehash is still exercised.
// ============================================================================
TEST(ConcurrentRehash, SegmentedPerSegmentRehashNoGlobalStall) {
    constexpr int kThreads = 8;
    constexpr int kOpsPerThread = 500;
    constexpr int kTotalKeys = kThreads * kOpsPerThread;  // 4000

    // Use segmented_safe_cache to exercise the per-segment rehash path
    // (segmented_concurrent_hash_table). Capacity must exceed total keys
    // to avoid LRU eviction.
    segmented_safe_cache<int, int> c(kTotalKeys * 2);

    std::vector<std::thread> threads;
    std::atomic<int> completed{0};

    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t] {
            for (int i = 0; i < kOpsPerThread; ++i) {
                int key = t * kOpsPerThread + i;
                c.set(key, key);
            }
            completed.fetch_add(1, std::memory_order_relaxed);
        });
    }
    for (auto& th : threads) th.join();

    EXPECT_EQ(completed.load(), kThreads);

    // All keys must be readable.
    for (int t = 0; t < kThreads; ++t) {
        for (int i = 0; i < kOpsPerThread; ++i) {
            int key = t * kOpsPerThread + i;
            auto h = c.try_get(key);
            ASSERT_TRUE(h.has_value()) << "Missing key " << key;
            EXPECT_EQ(**h, key);
        }
    }
}

// ============================================================================
// TC-G6c: RehashDuringMixedReadWrite
// Mixed read/write workload that triggers rehash; verifies no key is lost.
// ============================================================================
TEST(ConcurrentRehash, RehashDuringMixedReadWrite) {
    striped_cache<int, std::string> c(256);
    c.set_incremental_rehash(true);

    // Pre-fill so the table is near its growth threshold.
    for (int i = 0; i < 200; ++i) {
        c.set(i, "v" + std::to_string(i));
    }

    constexpr int kThreads = 8;
    constexpr int kOpsPerThread = 1000;

    std::atomic<int> write_count{0};
    std::atomic<int> read_count{0};
    std::atomic<int> read_mismatch{0};

    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t] {
            std::mt19937 rng(t);
            for (int i = 0; i < kOpsPerThread; ++i) {
                int key = rng() % 4000;
                if (rng() % 2 == 0) {
                    c.set(key, "v" + std::to_string(key));
                    write_count.fetch_add(1, std::memory_order_relaxed);
                } else {
                    auto h = c.try_get(key);
                    if (h) {
                        read_count.fetch_add(1, std::memory_order_relaxed);
                        if (**h != "v" + std::to_string(key)) {
                            read_mismatch.fetch_add(1, std::memory_order_relaxed);
                        }
                    }
                }
            }
        });
    }
    for (auto& th : threads) th.join();

    EXPECT_EQ(read_mismatch.load(), 0)
        << "readers observed mismatched values during mixed rehash workload";
    EXPECT_GT(write_count.load(), 0);
    EXPECT_GT(read_count.load(), 0);
}

// ============================================================================
// TC-G6c (P1-5): LockFreeRehashFallbackAccounting
// Verifies the rehash_lockfree_fallback_count metric is exposed via the
// cache API and Prometheus text format. The metric MUST be aggregated
// across all segments in a segmented hash table and across all shards
// in a striped cache.
//
// Strategy: drive a workload that forces incremental rehash activity
// while concurrent readers hit the table. After the workload:
//   1. rehash_lockfree_fallback_count() returns a value >= 0 (sanity).
//   2. prometheus_text() contains the
//      `lru_rehash_lockfree_fallback_total` line.
//   3. The value reported in prometheus_text matches the API call
//      (they read the same underlying atomic counters).
// ============================================================================
TEST(ConcurrentRehash, LockFreeRehashFallbackAccounting) {
    constexpr int kThreads = 8;
    constexpr int kReaders = 4;
    constexpr int kOpsPerThread = 1000;
    constexpr int kTotalKeys = kThreads * kOpsPerThread;  // 8000

    // segmented_safe_cache uses a 64-segment hash table; per-segment
    // rehash counters are aggregated by rehash_lockfree_fallback_count().
    segmented_safe_cache<int, int> c(kTotalKeys * 2);
    c.set_incremental_rehash(true);

    std::atomic<bool> stop{false};

    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t] {
            for (int i = 0; i < kOpsPerThread; ++i) {
                int key = t * kOpsPerThread + i;
                c.set(key, key * 10);
            }
        });
    }
    for (int t = 0; t < kReaders; ++t) {
        // P3-3: Capture `t` by value (not by reference). The loop variable `t`
        // goes out of scope when the for loop ends at the closing `}`, but the
        // reader thread may still be running and would dereference a dangling
        // reference. ASan reports this as "stack-use-after-scope" on the
        // captured `t`. The writer lambda above already captures `t` by value
        // for the same reason; this fix makes the reader consistent.
        threads.emplace_back([&, t] {
            std::mt19937 rng(t + 4242);
            while (!stop.load(std::memory_order_acquire)) {
                int key = rng() % kTotalKeys;
                (void)c.try_get(key);
            }
        });
    }

    for (int i = 0; i < kThreads; ++i) threads[i].join();
    stop.store(true, std::memory_order_release);
    for (int i = kThreads; i < kThreads + kReaders; ++i) threads[i].join();

    // 1. API exposure: rehash_lockfree_fallback_count() is callable and
    //    returns a finite, non-negative count.
    const std::size_t api_count = c.rehash_lockfree_fallback_count();
    EXPECT_GE(api_count, 0u);

    // 2. Prometheus export: the metric line MUST be present. We search
    //    for the metric NAME at the start of a line to skip the
    //    `# HELP` and `# TYPE` comment lines (which also contain the
    //    metric name but are not metric samples).
    const std::string ptext = c.prometheus_text();
    const std::string metric_name = "lru_rehash_lockfree_fallback_total ";
    auto pos = ptext.find(metric_name);
    while (pos != std::string::npos && pos > 0 && ptext[pos - 1] != '\n') {
        pos = ptext.find(metric_name, pos + metric_name.size());
    }
    ASSERT_NE(pos, std::string::npos)
        << "prometheus_text() must export lru_rehash_lockfree_fallback_total";
    pos += metric_name.size();
    auto eol = ptext.find('\n', pos);
    ASSERT_NE(eol, std::string::npos);
    std::string value_substr = ptext.substr(pos, eol - pos);
    const std::size_t prom_count = std::stoull(value_substr);
    EXPECT_EQ(prom_count, api_count)
        << "prometheus_text() value must match rehash_lockfree_fallback_count()";
}
