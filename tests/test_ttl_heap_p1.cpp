// P1-31 regression tests for detail::ttl_heap — the strategy-agnostic TTL index.
//
// The component is the mechanism P1-31 方案 A relies on: one O(log n) expiry
// index shared by every eviction strategy, instead of an LRU-only heap plus an
// O(n) full-cache scan for everybody else. These tests pin the component's
// contracts directly, so the strategies that adopt it cannot silently regress
// them. They need no cache and no clock — the "current time" is a parameter,
// which keeps the tests deterministic and fast.

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

#include "../detail/ttl_heap.hpp"

namespace {

using lru::detail::ttl_heap;
using probe = ttl_heap<std::string>::probe_result;

// Always-ready probe: every popped entry is live and unpinned.
auto ready_probe() {
    return [](const std::string&, std::uint64_t) { return probe::ready; };
}

TEST(TtlHeapP1, SweepsOnlyExpiredEntries) {
    ttl_heap<std::string> h;
    h.push("a", 100);
    h.push("b", 200);
    h.push("c", 300);

    std::vector<std::string> erased;
    const std::size_t n = h.sweep(150, 0, ready_probe(),
        [&](const std::string& k) { erased.push_back(k); });

    EXPECT_EQ(n, 1u);
    ASSERT_EQ(erased.size(), 1u);
    EXPECT_EQ(erased[0], "a") << "the earliest-expiring key must come out first";
    EXPECT_FALSE(h.empty()) << "future expiries must stay indexed";
}

TEST(TtlHeapP1, BatchBudgetIsHonoured) {
    ttl_heap<std::string> h;
    h.push("a", 10);
    h.push("b", 20);
    h.push("c", 30);

    std::size_t erased = 0;
    const std::size_t n = h.sweep(100, /*budget=*/2, ready_probe(),
        [&](const std::string&) { ++erased; });

    EXPECT_EQ(n, 2u);
    EXPECT_EQ(erased, 2u);
    EXPECT_FALSE(h.empty()) << "the third entry must remain for the next round";
}

TEST(TtlHeapP1, StaleAndAbsentEntriesAreDiscardedWithoutErase) {
    ttl_heap<std::string> h;
    h.push("removed", 10);  // probe says absent
    h.push("refreshed", 20);  // probe says stale (expiry changed)
    h.push("live", 30);  // probe says ready

    std::size_t erased = 0;
    const std::size_t n = h.sweep(100, 0,
        [](const std::string& k, std::uint64_t) {
            if (k == "removed") return probe::absent;
            if (k == "refreshed") return probe::stale;
            return probe::ready;
        },
        [&](const std::string&) { ++erased; });

    EXPECT_EQ(n, 1u) << "only the genuinely expired, current entry counts";
    EXPECT_EQ(erased, 1u) << "absent/stale entries must not be erased";
}

TEST(TtlHeapP1, PinnedEntryDoesNotBlockLaterExpiries) {
    // This is the P1-31 sub-fix A1 contract. The pinned key has the EARLIEST
    // expiry, so it is always at the top of the min-heap. The old sweep pushed it
    // back and `break`ed, which blocked every later expiry behind it forever.
    ttl_heap<std::string> h;
    h.push("pinned", 10);
    h.push("later", 20);
    h.push("latest", 30);

    std::vector<std::string> erased;
    const std::size_t n = h.sweep(100, 0,
        [](const std::string& k, std::uint64_t) {
            return k == "pinned" ? probe::pinned : probe::ready;
        },
        [&](const std::string& k) { erased.push_back(k); });

    EXPECT_EQ(n, 2u) << "both unblocked expired entries must be reclaimed";
    ASSERT_EQ(erased.size(), 2u);
    EXPECT_EQ(erased[0], "later");
    EXPECT_EQ(erased[1], "latest");
    EXPECT_FALSE(h.empty()) << "the pinned entry must stay indexed for a retry";
}

TEST(TtlHeapP1, DeferredEntryIsRetriedOnceUnpinned) {
    ttl_heap<std::string> h;
    h.push("k", 10);

    // First sweep: still pinned, so nothing is erased but the entry survives.
    std::size_t erased = 0;
    EXPECT_EQ(h.sweep(100, 0,
                  [](const std::string&, std::uint64_t) { return probe::pinned; },
                  [&](const std::string&) { ++erased; }),
              0u);
    EXPECT_EQ(erased, 0u);

    // Second sweep: the handle was released.
    const std::size_t n = h.sweep(100, 0, ready_probe(),
        [&](const std::string&) { ++erased; });
    EXPECT_EQ(n, 1u) << "the deferred entry must be retried, not lost";
    EXPECT_EQ(erased, 1u);
    EXPECT_TRUE(h.empty());
}

TEST(TtlHeapP1, ZeroExpiryIsNeverIndexed) {
    ttl_heap<std::string> h;
    h.push("no_ttl", 0);
    EXPECT_TRUE(h.empty()) << "expiry_ns == 0 means 'no TTL'";
}

TEST(TtlHeapP1, RebuildBoundsHeapGrowth) {
    ttl_heap<std::string> h;
    // Simulate many refreshes of the SAME live key: each pushes a new entry and
    // leaves the previous one stale.
    for (std::uint64_t i = 1; i <= 100; ++i) {
        h.push("hot", 1000 + i);
    }
    ASSERT_GT(h.size(), 4u);
    EXPECT_TRUE(h.needs_rebuild(/*live_items=*/1, /*multiplier=*/4));

    h.rebuild([](auto&& emit) { emit("hot", 2000); });
    EXPECT_EQ(h.size(), 1u) << "rebuild must re-derive the index from live items";
    EXPECT_FALSE(h.needs_rebuild(1, 4));
}

}  // namespace
