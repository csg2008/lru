// SPDX-License-Identifier: MIT
// Tests for refcount_with_flags

#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>

#include "detail/refcount.hpp"

using namespace lru::detail;

// ============================================================================
// Basic incRef / decRef
// ============================================================================

TEST(RefcountTest, BasicIncDecRef) {
    refcount_with_flags rc;
    EXPECT_EQ(rc.getAccessRef(), 0u);
    EXPECT_TRUE(rc.isDrained());

    auto r1 = rc.incRef();
    EXPECT_EQ(r1, IncResult::kIncOk);
    EXPECT_EQ(rc.getAccessRef(), 1u);
    EXPECT_FALSE(rc.isDrained());

    auto r2 = rc.incRef();
    EXPECT_EQ(r2, IncResult::kIncOk);
    EXPECT_EQ(rc.getAccessRef(), 2u);

    auto v = rc.decRef();
    EXPECT_EQ(rc.getAccessRef(), 1u);
    v = rc.decRef();
    EXPECT_EQ(rc.getAccessRef(), 0u);
    EXPECT_TRUE(rc.isDrained());
}

TEST(RefcountTest, IncRefMultipleTimes) {
    refcount_with_flags rc;
    for (int i = 0; i < 100; ++i) {
        auto r = rc.incRef();
        EXPECT_EQ(r, IncResult::kIncOk);
    }
    EXPECT_EQ(rc.getAccessRef(), 100u);

    for (int i = 0; i < 100; ++i) {
        rc.decRef();
    }
    EXPECT_EQ(rc.getAccessRef(), 0u);
    EXPECT_TRUE(rc.isDrained());
}

// ============================================================================
// markForEviction blocks incRef
// ============================================================================

TEST(RefcountTest, MarkForEvictionBlocksIncRef) {
    refcount_with_flags rc;
    rc.markInMMContainer();  // Set kLinked so markForEviction can succeed

    auto result = rc.markForEviction();
    EXPECT_EQ(result, MarkForEvictionResult::kSuccess);
    EXPECT_TRUE(rc.isMarkedForEviction());

    // incRef should fail now
    auto inc_result = rc.incRef();
    EXPECT_EQ(inc_result, IncResult::kIncFailedEviction);
    EXPECT_EQ(rc.getAccessRef(), 0u);
}

// ============================================================================
// incRef blocks markForEviction
// ============================================================================

TEST(RefcountTest, IncRefBlocksMarkForEviction) {
    refcount_with_flags rc;
    rc.markInMMContainer();

    rc.incRef();
    EXPECT_EQ(rc.getAccessRef(), 1u);

    auto result = rc.markForEviction();
    EXPECT_EQ(result, MarkForEvictionResult::kRefHeld);
    EXPECT_FALSE(rc.isMarkedForEviction());

    rc.decRef();
    EXPECT_EQ(rc.getAccessRef(), 0u);

    // Now markForEviction should succeed
    auto result2 = rc.markForEviction();
    EXPECT_EQ(result2, MarkForEvictionResult::kSuccess);
}

// ============================================================================
// markMoving vs markForEviction distinction
// ============================================================================

TEST(RefcountTest, MarkMovingVsEviction) {
    refcount_with_flags rc;
    rc.markInMMContainer();

    // markMoving: sets kExclusive + increments access_ref
    bool moving = rc.markMoving();
    EXPECT_TRUE(moving);
    EXPECT_TRUE(rc.isMoving());
    EXPECT_FALSE(rc.isMarkedForEviction());
    // Raw access_ref should be 1, but getAccessRef subtracts the moving ref
    EXPECT_EQ(rc.getAccessRef(), 0u);

    // unmarkMoving: decrements access_ref and clears kExclusive
    rc.unmarkMoving();
    EXPECT_EQ(rc.getAccessRef(), 0u);
    EXPECT_FALSE(rc.isMoving());
    EXPECT_FALSE(rc.isMarkedForEviction());
}

TEST(RefcountTest, MarkForEvictionThenUnmark) {
    refcount_with_flags rc;
    rc.markInMMContainer();

    auto result = rc.markForEviction();
    EXPECT_EQ(result, MarkForEvictionResult::kSuccess);
    EXPECT_TRUE(rc.isMarkedForEviction());

    rc.unmarkForEviction();
    EXPECT_FALSE(rc.isMarkedForEviction());
    EXPECT_FALSE(rc.isMoving());
}

TEST(RefcountTest, MovingBlocksEviction) {
    refcount_with_flags rc;
    rc.markInMMContainer();

    rc.markMoving();
    auto result = rc.markForEviction();
    EXPECT_EQ(result, MarkForEvictionResult::kExclusive);
}

TEST(RefcountTest, EvictionBlocksMoving) {
    refcount_with_flags rc;
    rc.markInMMContainer();

    rc.markForEviction();
    bool moving = rc.markMoving();
    EXPECT_FALSE(moving);
}

// ============================================================================
// kLinked (MM container membership)
// ============================================================================

TEST(RefcountTest, MMContainerMembership) {
    refcount_with_flags rc;
    EXPECT_FALSE(rc.isInMMContainer());

    rc.markInMMContainer();
    EXPECT_TRUE(rc.isInMMContainer());

    rc.unmarkInMMContainer();
    EXPECT_FALSE(rc.isInMMContainer());
}

TEST(RefcountTest, MarkForEvictionRequiresLinked) {
    refcount_with_flags rc;
    // kLinked not set → markForEviction returns kUnlinked
    auto result = rc.markForEviction();
    EXPECT_EQ(result, MarkForEvictionResult::kUnlinked);
}

// ============================================================================
// kAccessible
// ============================================================================

TEST(RefcountTest, AccessibleBit) {
    refcount_with_flags rc;
    EXPECT_FALSE(rc.isAccessible());

    rc.markAccessible();
    EXPECT_TRUE(rc.isAccessible());

    rc.unmarkAccessible();
    EXPECT_FALSE(rc.isAccessible());
}

// ============================================================================
// User flags
// ============================================================================

TEST(RefcountTest, UserFlags) {
    refcount_with_flags rc;

    rc.setFlag<Flags::kMMFlag0>();
    EXPECT_TRUE(rc.isFlagSet<Flags::kMMFlag0>());
    EXPECT_FALSE(rc.isFlagSet<Flags::kMMFlag1>());

    rc.setFlag<Flags::kMMFlag1>();
    EXPECT_TRUE(rc.isFlagSet<Flags::kMMFlag0>());
    EXPECT_TRUE(rc.isFlagSet<Flags::kMMFlag1>());

    rc.unSetFlag<Flags::kMMFlag0>();
    EXPECT_FALSE(rc.isFlagSet<Flags::kMMFlag0>());
    EXPECT_TRUE(rc.isFlagSet<Flags::kMMFlag1>());

    rc.setFlag<Flags::kIsChainedItem>();
    EXPECT_TRUE(rc.isFlagSet<Flags::kIsChainedItem>());
    rc.unSetFlag<Flags::kIsChainedItem>();
    EXPECT_FALSE(rc.isFlagSet<Flags::kIsChainedItem>());
}

// ============================================================================
// isDrained
// ============================================================================

TEST(RefcountTest, IsDrained) {
    refcount_with_flags rc;
    EXPECT_TRUE(rc.isDrained());

    rc.markInMMContainer();
    EXPECT_FALSE(rc.isDrained());

    rc.unmarkInMMContainer();
    EXPECT_TRUE(rc.isDrained());

    rc.incRef();
    EXPECT_FALSE(rc.isDrained());
    rc.decRef();
    EXPECT_TRUE(rc.isDrained());
}

// ============================================================================
// Concurrent incRef / decRef
// ============================================================================

TEST(RefcountTest, ConcurrentIncDecRef) {
    refcount_with_flags rc;
    rc.markInMMContainer();

    constexpr int kThreads = 8;
    constexpr int kOpsPerThread = 10000;

    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&rc]() {
            for (int i = 0; i < kOpsPerThread; ++i) {
                auto r = rc.incRef();
                ASSERT_EQ(r, IncResult::kIncOk);
                rc.decRef();
            }
        });
    }
    for (auto& th : threads) {
        th.join();
    }

    EXPECT_EQ(rc.getAccessRef(), 0u);
    // isDrained() is false because kLinked is still set (item is still in MM container)
    EXPECT_FALSE(rc.isDrained());
    rc.unmarkInMMContainer();
    EXPECT_TRUE(rc.isDrained());
}

// ============================================================================
// Concurrent markForEviction with incRef
// ============================================================================

TEST(RefcountTest, ConcurrentMarkForEvictionWithIncRef) {
    refcount_with_flags rc;
    rc.markInMMContainer();

    constexpr int kIterations = 10000;
    std::atomic<int> evict_successes{0};
    std::atomic<int> evict_failures{0};

    std::thread evictor([&]() {
        for (int i = 0; i < kIterations; ++i) {
            auto r = rc.markForEviction();
            if (r == MarkForEvictionResult::kSuccess) {
                evict_successes.fetch_add(1, std::memory_order_relaxed);
                rc.unmarkForEviction();
            } else {
                evict_failures.fetch_add(1, std::memory_order_relaxed);
            }
        }
    });

    std::thread accessor([&]() {
        for (int i = 0; i < kIterations; ++i) {
            auto r = rc.incRef();
            if (r == IncResult::kIncOk) {
                rc.decRef();
            }
        }
    });

    evictor.join();
    accessor.join();

    EXPECT_EQ(rc.getAccessRef(), 0u);
    // At least some operations should have completed
    EXPECT_GT(evict_successes.load() + evict_failures.load(), 0);
}

// ============================================================================
// getRaw and layout
// ============================================================================

TEST(RefcountTest, RawValueLayout) {
    refcount_with_flags rc;
    EXPECT_EQ(rc.getRaw(), 0u);

    rc.incRef();
    // access_ref should be 1 → bit 0 is set
    EXPECT_EQ(rc.getRaw() & kAccessRefMask, 1u);

    rc.markInMMContainer();
    // kLinked bit (bit 32) should be set  [R3: shifted from 24 to 32]
    EXPECT_TRUE(rc.getRaw() & (1ULL << kLinkedBit));

    rc.setFlag<Flags::kMMFlag0>();
    // kMMFlag0 bit (bit 35) should be set  [R3: shifted from 27 to 35]
    EXPECT_TRUE(rc.getRaw() & (1ULL << Flags::kMMFlag0));
}
