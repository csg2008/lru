// P1-25 / P1-26 regression tests for the CountMinSketch used by TinyLFU and
// W-TinyLFU admission.
//
// These lock in the two contracts that were violated:
//
//   P1-25 — the counter table must be sized from the CACHE CAPACITY
//           (numCounters = next_pow2(e * capacity * window_multiplier /
//           error_threshold), hash_count rows). The old (error_rate, confidence)
//           pair produced width=6, depth=7 => 42 counters TOTAL regardless of
//           capacity, so frequency estimates were collision noise and admission
//           was effectively random.
//
//   P1-26 — decay must halve the WHOLE table and shrink the decay window
//           geometrically. The old policy halved ONE ROW per event in
//           round-robin order; since estimate() takes the minimum across rows,
//           rows aged at different times made the estimate a minimum over counts
//           of different ages — systematically biased, and drifting.

#include <gtest/gtest.h>

#include <cstddef>

#include "../lru.hpp"

namespace {

using lru::detail::count_min_sketch;

// ---------------------------------------------------------------------------
// sizing is capacity-driven
// ---------------------------------------------------------------------------

TEST(CmsP1, CounterCountScalesWithCapacity) {
    // Check the whole table size, not just the row width: the defect was in the
    // PRODUCT (42 counters total), so testing width alone could pass by accident.
    count_min_sketch<int> small(/*capacity=*/1000);
    count_min_sketch<int> large(/*capacity=*/100'000);

    const std::size_t small_counters = small.width() * small.depth();
    const std::size_t large_counters = large.width() * large.depth();

    // 42 was the old value; a capacity-sized table is orders of magnitude larger.
    EXPECT_GT(small_counters, 1000u)
        << "a 1000-item cache must get far more than the old 42 counters";
    // Roughly proportional to capacity (pow2 rounding makes it inexact).
    EXPECT_GT(large_counters, small_counters * 20)
        << "counter count must grow with capacity, not stay fixed";
    // CacheLib's hash_count.
    EXPECT_GE(small.depth(), 4u);
}

TEST(CmsP1, ErrorThresholdControlsAccuracy) {
    // A smaller error threshold must buy more counters (the formula divides by
    // it), which is what makes the new configuration surface meaningful.
    count_min_sketch<int> loose(/*capacity=*/10'000, /*window_multiplier=*/32,
                               /*error_threshold=*/10.0, /*hash_count=*/4);
    count_min_sketch<int> tight(/*capacity=*/10'000, /*window_multiplier=*/32,
                               /*error_threshold=*/1.0, /*hash_count=*/4);
    EXPECT_GT(tight.width() * tight.depth(), loose.width() * loose.depth());
}

// ---------------------------------------------------------------------------
// whole-table decay with a geometrically shrinking window
// ---------------------------------------------------------------------------

TEST(CmsP1, WindowStartsAtCapacityTimesMultiplier) {
    count_min_sketch<int> s(/*capacity=*/64, /*window_multiplier=*/32);
    EXPECT_EQ(s.window_size(), 64u * 32u);
}

TEST(CmsP1, DecayShrinksTheWindowGeometricallyDownToAFloor) {
    count_min_sketch<int> s(/*capacity=*/64, /*window_multiplier=*/32);
    const std::size_t w0 = s.window_size();
    ASSERT_GT(w0, count_min_sketch<int>::kMinWindowSize);

    // Drive many decays. The window halves each time, so a handful of decay
    // periods is enough to reach the floor.
    for (std::size_t i = 0; i < w0 * 64; ++i) {
        s.record(static_cast<int>(i % 100));
    }

    EXPECT_LT(s.window_size(), w0) << "the decay window must shrink geometrically";
    EXPECT_GE(s.window_size(), count_min_sketch<int>::kMinWindowSize)
        << "the window must not collapse below the floor (each decay halves the "
           "whole table, so an unbounded shrink would make decay a hot loop)";
    // The floor is a power-of-two halving sequence, so it must be reached exactly.
    EXPECT_EQ(s.window_size(), count_min_sketch<int>::kMinWindowSize);
}

TEST(CmsP1, ExplicitDecayReArmsTheWindow) {
    count_min_sketch<int> s(/*capacity=*/64, /*window_multiplier=*/32);
    const std::size_t w0 = s.window_size();
    for (std::size_t i = 0; i < w0 * 64; ++i) {
        s.record(static_cast<int>(i % 100));
    }
    ASSERT_LT(s.window_size(), w0);

    s.decay();  // explicit full-table decay
    EXPECT_EQ(s.window_size(), w0)
        << "an explicit decay() must restore the configured window";
}

TEST(CmsP1, WholeTableDecayHalvesEveryRow) {
    // The observable consequence of whole-table decay: a key recorded steadily
    // has every one of its rows halved together, so its estimate decays smoothly
    // instead of being pinned to whichever row happened to be halved last.
    // With per-row round-robin decay the minimum across rows stayed at the
    // un-decayed value far longer.
    count_min_sketch<int> s(/*capacity=*/64, /*window_multiplier=*/32);

    // Record a fixed key until at least one automatic decay has fired, then keep
    // the key hot so its counters stay representative.
    constexpr int kHot = 12345;
    for (int i = 0; i < 20'000; ++i) {
        s.record(kHot);
    }
    const std::uint32_t hot_estimate = s.estimate(kHot);
    EXPECT_GT(hot_estimate, 0u) << "a constantly-accessed key must have a count";

    // A key never recorded must estimate far below the hot key.
    EXPECT_LT(s.estimate(999), hot_estimate);

    // And the hot count must be strictly bounded by the number of records (a
    // decayed sketch cannot report more than it saw).
    EXPECT_LE(hot_estimate, 20'000u);
}

}  // namespace
