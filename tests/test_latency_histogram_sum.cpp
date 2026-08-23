// Unit tests for latency_histogram::sum() — the exact accumulated latency
// counter that replaces the previous (min+max)/2 * count approximation used
// for the Prometheus histogram `_sum` line.
//
// Coverage:
//   - sum() accuracy with known inputs
//   - reset() zeroes sum
//   - copy/move construction and assignment preserve sum_
//   - merge_from() is additively correct
//   - concurrent record() from multiple threads yields exact sum

#include <gtest/gtest.h>

#include "../lru.hpp"

#include <atomic>
#include <cstdint>
#include <numeric>
#include <thread>
#include <vector>

using lru::detail::latency_histogram;

// ============================================================================
// sum() accuracy
// ============================================================================
TEST(LatencyHistogramSum, KnownValuesSumExactly) {
    latency_histogram h;
    const std::vector<uint64_t> samples{100, 200, 300, 400, 500};
    for (auto ns : samples) {
        h.record(ns);
    }
    EXPECT_EQ(h.count(), 5u);
    EXPECT_EQ(h.sum(), 1500u);
}

TEST(LatencyHistogramSum, EmptyHistogramSumIsZero) {
    latency_histogram h;
    EXPECT_EQ(h.sum(), 0u);
    EXPECT_EQ(h.count(), 0u);
}

TEST(LatencyHistogramSum, SingleSample) {
    latency_histogram h;
    h.record(12345u);
    EXPECT_EQ(h.sum(), 12345u);
    EXPECT_EQ(h.count(), 1u);
}

TEST(LatencyHistogramSum, ZeroLatencyRecorded) {
    latency_histogram h;
    h.record(0u);  // bucket 0, but sum contribution is 0
    h.record(0u);
    EXPECT_EQ(h.count(), 2u);
    EXPECT_EQ(h.sum(), 0u);
}

TEST(LatencyHistogramSum, LargeValuesDoNotOverflow) {
    latency_histogram h;
    // 2^32 ns ~ 4.29s — well below uint64_t range. Recording it twice must
    // produce exactly 2^33, proving no truncation in the fetch_add path.
    const uint64_t big = uint64_t{1} << 32;
    h.record(big);
    h.record(big);
    EXPECT_EQ(h.sum(), big * 2);
}

TEST(LatencyHistogramSum, SumConsistentWithCountAcrossBuckets) {
    latency_histogram h;
    // Record values that land in different buckets (powers of two).
    uint64_t expected_sum = 0;
    for (int i = 0; i < 20; ++i) {
        uint64_t v = uint64_t{1} << i;  // 1, 2, 4, ... 2^19
        h.record(v);
        expected_sum += v;
    }
    EXPECT_EQ(h.count(), 20u);
    EXPECT_EQ(h.sum(), expected_sum);
}

// ============================================================================
// reset()
// ============================================================================
TEST(LatencyHistogramSum, ResetZeroesSum) {
    latency_histogram h;
    h.record(100);
    h.record(200);
    h.record(300);
    ASSERT_GT(h.sum(), 0u);
    ASSERT_GT(h.count(), 0u);

    h.reset();

    EXPECT_EQ(h.sum(), 0u);
    EXPECT_EQ(h.count(), 0u);
    // min/max reset to sentinel → reported as 0.
    EXPECT_EQ(h.min(), 0u);
    EXPECT_EQ(h.max(), 0u);
}

TEST(LatencyHistogramSum, RecordAfterResetReaccumulates) {
    latency_histogram h;
    h.record(1000);
    h.reset();
    h.record(50);
    EXPECT_EQ(h.sum(), 50u);
    EXPECT_EQ(h.count(), 1u);
}

// ============================================================================
// Copy / move semantics preserve sum_
// ============================================================================
TEST(LatencyHistogramSum, CopyConstructorPreservesSum) {
    latency_histogram h;
    for (uint64_t v : {10u, 20u, 30u, 40u}) {
        h.record(v);
    }
    ASSERT_EQ(h.sum(), 100u);

    latency_histogram copy(h);
    EXPECT_EQ(copy.sum(), 100u);
    EXPECT_EQ(copy.count(), 4u);
}

TEST(LatencyHistogramSum, MoveConstructorPreservesSum) {
    latency_histogram h;
    for (uint64_t v : {7u, 14u, 21u}) {
        h.record(v);
    }
    ASSERT_EQ(h.sum(), 42u);

    latency_histogram moved(std::move(h));
    EXPECT_EQ(moved.sum(), 42u);
    EXPECT_EQ(moved.count(), 3u);
}

TEST(LatencyHistogramSum, CopyAssignmentPreservesSum) {
    latency_histogram h;
    for (uint64_t v : {1u, 2u, 3u, 4u, 5u}) {
        h.record(v);
    }
    ASSERT_EQ(h.sum(), 15u);

    latency_histogram target;
    target.record(999u);  // pre-existing data must be overwritten
    target = h;
    EXPECT_EQ(target.sum(), 15u);
    EXPECT_EQ(target.count(), 5u);
}

TEST(LatencyHistogramSum, MoveAssignmentPreservesSum) {
    latency_histogram h;
    for (uint64_t v : {100u, 200u, 300u}) {
        h.record(v);
    }
    ASSERT_EQ(h.sum(), 600u);

    latency_histogram target;
    target.record(7u);
    target = std::move(h);
    EXPECT_EQ(target.sum(), 600u);
    EXPECT_EQ(target.count(), 3u);
}

TEST(LatencyHistogramSum, SelfAssignmentSafe) {
    latency_histogram h;
    h.record(11);
    h.record(22);
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wself-assign-overloaded"
#endif
    h = h;  // NOLINT(bugprone-unhandled-self-assignment)
#ifdef __clang__
#pragma clang diagnostic pop
#endif
    EXPECT_EQ(h.sum(), 33u);
    EXPECT_EQ(h.count(), 2u);
}

// ============================================================================
// merge_from() is additively correct for sum
// ============================================================================
TEST(LatencyHistogramSum, MergeFromAddsSums) {
    latency_histogram a;
    latency_histogram b;
    for (uint64_t v : {1u, 2u, 3u}) a.record(v);  // sum = 6
    for (uint64_t v : {10u, 20u}) b.record(v);    // sum = 30

    a.merge_from(b);

    EXPECT_EQ(a.sum(), 36u);
    EXPECT_EQ(a.count(), 5u);
}

TEST(LatencyHistogramSum, MergeFromEmptyOtherLeavesSumUnchanged) {
    latency_histogram a;
    latency_histogram empty;
    a.record(42);
    ASSERT_EQ(a.sum(), 42u);

    a.merge_from(empty);
    EXPECT_EQ(a.sum(), 42u);
    EXPECT_EQ(a.count(), 1u);
}

TEST(LatencyHistogramSum, MergeFromIntoEmptyCopiesSum) {
    latency_histogram a;
    latency_histogram b;
    for (uint64_t v : {5u, 10u, 15u}) b.record(v);  // sum = 30

    a.merge_from(b);
    EXPECT_EQ(a.sum(), 30u);
    EXPECT_EQ(a.count(), 3u);
}

// ============================================================================
// Concurrent record(): final sum must equal the exact arithmetic sum
// ============================================================================
TEST(LatencyHistogramSum, ConcurrentRecordProducesExactSum) {
    latency_histogram h;
    const int num_threads = 8;
    const int per_thread = 1000;

    // Each thread records 1..per_thread. Per-thread sum = N(N+1)/2.
    const uint64_t per_thread_sum =
        static_cast<uint64_t>(per_thread) * (per_thread + 1) / 2;
    const uint64_t expected_sum = per_thread_sum * num_threads;
    const uint64_t expected_count =
        static_cast<uint64_t>(num_threads) * per_thread;

    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&h, per_thread]() {
            for (int i = 1; i <= per_thread; ++i) {
                h.record(static_cast<uint64_t>(i));
            }
        });
    }
    for (auto& th : threads) {
        th.join();
    }

    EXPECT_EQ(h.count(), expected_count);
    EXPECT_EQ(h.sum(), expected_sum);
}

// ============================================================================
// T12.1 / T12.2: set_latency_sample_rate() — deterministic sampling
// ============================================================================
TEST(LatencyHistogramSampleRate, DefaultRateIsOne) {
    latency_histogram h;
    EXPECT_EQ(h.sample_rate(), 1u);
}

TEST(LatencyHistogramSampleRate, RateZeroIsTreatedAsOne) {
    latency_histogram h;
    h.set_latency_sample_rate(0);
    EXPECT_EQ(h.sample_rate(), 1u);
}

TEST(LatencyHistogramSampleRate, RateOneRecordsAll) {
    latency_histogram h;
    h.set_latency_sample_rate(1);
    for (uint64_t i = 1; i <= 100; ++i) {
        h.record(i * 100);
    }
    EXPECT_EQ(h.count(), 100u);
    EXPECT_EQ(h.sum(), 505000u);
}

TEST(LatencyHistogramSampleRate, RateGreaterThanOneSamplesSubset) {
    latency_histogram h;
    h.set_latency_sample_rate(10);
    for (uint64_t i = 1; i <= 1000; ++i) {
        h.record(i * 100);
    }
    const uint64_t c = h.count();
    EXPECT_GE(c, 50u);
    EXPECT_LE(c, 200u);
}

TEST(LatencyHistogramSampleRate, MinMaxStillAccurateUnderSampling) {
    latency_histogram h;
    h.set_latency_sample_rate(5);
    for (int i = 0; i < 1000; ++i) {
        h.record(10);
        h.record(99999);
    }
    EXPECT_EQ(h.min(), 10u);
    EXPECT_EQ(h.max(), 99999u);
}

// ============================================================================
// T12.3: release_memory() — clear all data on disable
// ============================================================================
TEST(LatencyHistogramReleaseMemory, ClearsAllCounters) {
    latency_histogram h;
    for (uint64_t v : {100u, 200u, 300u, 400u, 500u}) {
        h.record(v);
    }
    ASSERT_EQ(h.count(), 5u);
    ASSERT_EQ(h.sum(), 1500u);
    ASSERT_GT(h.max(), 0u);

    h.release_memory();

    EXPECT_EQ(h.count(), 0u);
    EXPECT_EQ(h.sum(), 0u);
    EXPECT_EQ(h.min(), 0u);
    EXPECT_EQ(h.max(), 0u);
    for (std::size_t i = 0; i < latency_histogram::bucket_count; ++i) {
        EXPECT_EQ(h.bucket(i), 0u);
    }
}

TEST(LatencyHistogramReleaseMemory, RecordingResumesAfterRelease) {
    latency_histogram h;
    h.record(123u);
    h.release_memory();
    EXPECT_EQ(h.count(), 0u);

    h.record(456u);
    EXPECT_EQ(h.count(), 1u);
    EXPECT_EQ(h.sum(), 456u);
    EXPECT_EQ(h.min(), 456u);
    EXPECT_EQ(h.max(), 456u);
}

TEST(LatencyHistogramReleaseMemory, SampleRatePreservedAcrossRelease) {
    latency_histogram h;
    h.set_latency_sample_rate(7);
    h.record(100u);
    h.release_memory();
    EXPECT_EQ(h.sample_rate(), 7u);
}

// ============================================================================
// Concurrent record() with distinct values per thread — final sum is exact
// ============================================================================
TEST(LatencyHistogramSum, ConcurrentRecordDistinctValuesExactSum) {
    // Each thread records a distinct, non-overlapping range so the expected
    // sum is unambiguous and easy to verify. Thread t records
    // [t*base, t*base + per_thread) for a known arithmetic series.
    latency_histogram h;
    const int num_threads = 4;
    const int per_thread = 500;
    const uint64_t base = 1000;

    // Thread t records values base*t + 1 .. base*t + per_thread.
    // Total sum = sum over t of [ per_thread*(base*t) + N(N+1)/2 ].
    const uint64_t tri = static_cast<uint64_t>(per_thread) * (per_thread + 1) / 2;
    uint64_t expected_sum = 0;
    for (int t = 0; t < num_threads; ++t) {
        expected_sum += static_cast<uint64_t>(per_thread) * (base * t) + tri;
    }
    const uint64_t expected_count =
        static_cast<uint64_t>(num_threads) * per_thread;

    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&h, t, base, per_thread]() {
            for (int i = 1; i <= per_thread; ++i) {
                h.record(base * static_cast<uint64_t>(t) +
                         static_cast<uint64_t>(i));
            }
        });
    }
    for (auto& th : threads) {
        th.join();
    }

    EXPECT_EQ(h.count(), expected_count);
    EXPECT_EQ(h.sum(), expected_sum);
}
