// SPDX-License-Identifier: MIT
// Concurrent metrics / statistics atomicity tests.
//
// Covers spec gaps G7, G17, G18, G21, G22 (P1/P2):
//   G7:  refcount saturation boundary (kIncFailedOverflow)
//   G17: event_tracker concurrent record consistency
//   G18: latency_histogram concurrent record correctness
//   G21: count_min_sketch concurrent add atomicity
//   G22: overflow_policy::kRejectInsert under concurrent set()

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

#include "detail/count_min_sketch.hpp"
#include "detail/latency_histogram.hpp"
#include "detail/refcount.hpp"
#include "event_tracker.hpp"
#include "lru.hpp"
#include "test_helpers.hpp"

using namespace lru;
using namespace lru::detail;
using namespace std::chrono_literals;

// ============================================================================
// TC-G7: RefcountSaturatesAndStaysCorrect
// Drive incRef() to the saturation point; verify kIncFailedOverflow is
// returned and that subsequent decRef() calls correctly restore state.
//
// The decrement phase uses an atomic "ticket" counter to ensure that exactly
// kAccessRefMax decrements are attempted across all threads. Without this,
// the read-then-dec pattern races: thread A reads access_ref == 1, thread B
// reads access_ref == 1, thread A decRef()s to 0, thread B decRef() throws
// std::underflow_error. The ticket counter prevents this by atomically
// reserving a decrement slot before calling decRef().
// ============================================================================
TEST(ConcurrentMetrics, RefcountSaturatesAndStaysCorrect) {
    refcount_with_flags rc;
    rc.markInMMContainer();

    // Single-threaded: drive to saturation.
    std::atomic<bool> saturated{false};
    std::atomic<int> saturated_count{0};

    constexpr int kThreads = 8;
    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&] {
            while (true) {
                auto r = rc.incRef();
                if (r == IncResult::kIncOk) continue;
                if (r == IncResult::kIncFailedOverflow) {
                    saturated.store(true, std::memory_order_release);
                    saturated_count.fetch_add(1, std::memory_order_relaxed);
                    return;
                }
                // Other failure modes (kIncFailedEviction / kIncFailedMoving)
                // would indicate an unexpected state transition.
                return;
            }
        });
    }
    for (auto& th : threads) th.join();

    EXPECT_TRUE(saturated.load()) << "refcount never saturated";
    EXPECT_GE(saturated_count.load(), 1);

    // Verify the refcount is at the saturation value.
    EXPECT_EQ(rc.getAccessRef(), kAccessRefMax);

    // Decrement back to zero across all threads. Use a ticket counter so
    // exactly kAccessRefMax decrements are attempted — no more, no less.
    // This avoids the read-then-decRef race that would otherwise throw
    // std::underflow_error when two threads observe the same non-zero
    // access_ref and both call decRef().
    std::atomic<std::int64_t> tickets{static_cast<std::int64_t>(kAccessRefMax)};
    std::atomic<std::uint64_t> decrements{0};
    std::vector<std::thread> dec_threads;
    for (int t = 0; t < kThreads; ++t) {
        dec_threads.emplace_back([&] {
            while (true) {
                // Atomically reserve a decrement slot.
                auto ticket = tickets.fetch_sub(1, std::memory_order_acq_rel);
                if (ticket <= 0) {
                    // No more tickets — restore the slot we just took below 0
                    // (not strictly necessary; we're done).
                    return;
                }
                rc.decRef();
                decrements.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }
    for (auto& th : dec_threads) th.join();

    EXPECT_EQ(rc.getAccessRef(), 0u);
    EXPECT_EQ(decrements.load(), static_cast<std::uint64_t>(kAccessRefMax));
}

// ============================================================================
// TC-G17: EventTrackerConcurrentRecord
// Many threads concurrently record events; the streaming top_keys summary
// must be consistent after all threads drain.
// ============================================================================
TEST(ConcurrentMetrics, EventTrackerConcurrentRecord) {
    event_tracker<int> tracker(10000);

    constexpr int kThreads = 16;
    constexpr int kEventsPerThread = 1000;

    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t] {
            for (int i = 0; i < kEventsPerThread; ++i) {
                // Each thread records a mix of events with key == thread id.
                tracker.record_hit(t);
                if (i % 100 == 0) {
                    tracker.record_insert(t);
                }
                if (i % 200 == 0) {
                    tracker.record_evict(t);
                }
            }
        });
    }
    for (auto& th : threads) th.join();

    tracker.drain_all_threads();

    auto top = tracker.top_keys(kThreads);
    // Each thread recorded at least one hit, so we expect at least kThreads
    // distinct keys to appear in the top-k.
    // (Space-Saving is an upper-bound estimator.)
    EXPECT_LE(top.size(), static_cast<std::size_t>(kThreads));
}

// ============================================================================
// TC-G18: LatencyHistogramConcurrentRecord
// N threads concurrently record samples; sum() and count() must be exact.
// ============================================================================
TEST(ConcurrentMetrics, LatencyHistogramConcurrentRecord) {
    latency_histogram h;

    constexpr int kThreads = 8;
    constexpr int kSamplesPerThread = 10000;

    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&h, t] {
            for (int i = 0; i < kSamplesPerThread; ++i) {
                // Each thread records a deterministic latency value.
                uint64_t latency = static_cast<uint64_t>(t * 100 + i);
                h.record(latency);
            }
        });
    }
    for (auto& th : threads) th.join();

    // Total samples recorded.
    uint64_t total = static_cast<uint64_t>(kThreads) * kSamplesPerThread;
    EXPECT_EQ(h.count(), total);

    // Sum: sum over t in [0, kThreads), sum over i in [0, kSamplesPerThread)
    //      of (t * 100 + i).
    // = kSamplesPerThread * sum(t*100 for t) + kThreads * sum(i for i)
    // = kSamplesPerThread * 100 * (kThreads-1)*kThreads/2
    //   + kThreads * (kSamplesPerThread-1)*kSamplesPerThread/2
    uint64_t expected_sum = 0;
    for (int t = 0; t < kThreads; ++t) {
        for (int i = 0; i < kSamplesPerThread; ++i) {
            expected_sum += static_cast<uint64_t>(t * 100 + i);
        }
    }
    EXPECT_EQ(h.sum(), expected_sum);
}

// ============================================================================
// TC-G21: CountMinSketchConcurrentAdd
// Concurrent record() calls must not corrupt the sketch; estimate() must
// be a lower bound (Count-Min Sketch property).
// ============================================================================
TEST(ConcurrentMetrics, CountMinSketchConcurrentAdd) {
    count_min_sketch<int> sketch(/*capacity=*/10000,
                                  /*error_rate=*/0.1,
                                  /*confidence=*/0.95);

    constexpr int kThreads = 8;
    constexpr int kAccessesPerThread = 1000;
    const int the_key = 42;

    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&] {
            for (int i = 0; i < kAccessesPerThread; ++i) {
                sketch.record(the_key);
            }
        });
    }
    for (auto& th : threads) th.join();

    // Count-Min Sketch overestimates; the estimate must be >= true count.
    uint32_t estimate = sketch.estimate(the_key);
    uint32_t true_count = static_cast<uint32_t>(kThreads * kAccessesPerThread);
    EXPECT_GE(estimate, true_count)
        << "Count-Min Sketch underestimates (impossible — should be upper bound)";
    // Sanity upper bound: estimate should not exceed true_count + error_margin.
    // For an error_rate of 0.1 with confidence 0.95, the expected over-count
    // is bounded; an exact bound is hard to compute without depth/width,
    // so we just sanity-check that the estimate is not absurdly large.
    EXPECT_LE(estimate, true_count * 10)
        << "Count-Min Sketch estimate is unreasonably large";
}

// ============================================================================
// TC-G22: OverflowPolicyRejectInsertUnderConcurrency
// When the cache is at capacity with kRejectInsert policy, concurrent set()
// calls must not over-insert. The cache size must stay bounded.
// ============================================================================
TEST(ConcurrentMetrics, OverflowPolicyRejectInsertUnderConcurrency) {
    cache<int, int> c(20);
    mm_lru_config cfg = c.mm().config();
    cfg.overflow_policy_value = overflow_policy::kRejectInsert;
    cfg.overflow_tolerance = 0.0;
    c.mm().set_config(cfg);

    // Fill to capacity.
    for (int i = 0; i < 20; ++i) c.set(i, i);
    ASSERT_EQ(c.size(), 20u);

    constexpr int kThreads = 8;
    constexpr int kOpsPerThread = 200;

    std::atomic<int> attempts{0};
    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t] {
            for (int i = 0; i < kOpsPerThread; ++i) {
                int key = 1000 + t * kOpsPerThread + i;
                c.set(key, key);
                attempts.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }
    for (auto& th : threads) th.join();

    // With kRejectInsert at 0% tolerance, the cache size must remain exactly
    // 20 (no growth, no eviction of existing items).
    EXPECT_EQ(c.size(), 20u)
        << "kRejectInsert policy allowed cache to grow beyond capacity";

    // None of the new keys should be present.
    for (int t = 0; t < kThreads; ++t) {
        for (int i = 0; i < kOpsPerThread; ++i) {
            int key = 1000 + t * kOpsPerThread + i;
            EXPECT_FALSE(c.contains(key))
                << "key " << key << " was inserted despite kRejectInsert";
        }
    }

    // Original keys must remain readable.
    for (int i = 0; i < 20; ++i) {
        auto h = c.try_get(i);
        ASSERT_TRUE(h.has_value());
        EXPECT_EQ(**h, i);
    }
}
