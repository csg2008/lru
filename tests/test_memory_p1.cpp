// Unified LRU Cache Library - slab allocator and memory monitor regression tests.
// (slab allocator / memory monitor fixes P1-13 … P1-18)
// SPDX-License-Identifier: MIT
//
// Covered here:
//   P1-13  Treiber-stack ABA tag must advance on the stack → empty transition
//   P1-14  deallocate() must not trust the caller-supplied size (RAII handle +
//          header-verified internal path; wrong size = rejected + counted)
//   P1-15  the owning NUMA node is recorded per block and deallocate routes by
//          it instead of by current_allocator()
//   P1-16  report_memory() is a pure gauge; the rate limiter exposes an
//          explicit `exceeded` verdict; negative deltas cannot become ~2^64
//   P1-17  the memory cap is a hard limit enforced by CAS reservation
//   P1-18  per-class live/peak bytes + failure counters, stable snapshot, and a
//          utilization that stays inside [0, 1] under concurrency

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <limits>
#include <thread>
#include <utility>
#include <vector>

#include "../lru.hpp"

using namespace lru;

// ============================================================================
// P1-13 — ABA tag on the Treiber stack
// ============================================================================

TEST(MemoryP1Test, AbaTagAdvancesWhenFreeListEmpties) {
    slab_allocator::config cfg;
    cfg.initial_slabs_per_class = 1;
    cfg.max_slabs_per_class = 1;   // exactly one slab, so the list can be drained
    slab_allocator alloc(cfg);

    // 48 payload bytes + 16 header bytes == the 64-byte class (index 0).
    const std::uint32_t idx = slab_allocator::class_index_for_payload(48);
    ASSERT_EQ(idx, 0u);
    const std::uint32_t items = alloc.class_at(idx).items_per_slab();
    ASSERT_GT(items, 0u);

    std::vector<void*> blocks;
    blocks.reserve(items);
    while (void* p = alloc.allocate(48)) {
        blocks.push_back(p);
    }
    // With max_slabs_per_class == 1 the drain must stop exactly at capacity.
    ASSERT_EQ(blocks.size(), static_cast<std::size_t>(items));
    const std::uint32_t tag_empty_1 = alloc.class_at(idx).free_list_tag();

    // Push one block back and pop it again: this drives the free list through
    // the stack → empty transition that used to reset the tag to 0.
    alloc.deallocate(blocks.back(), 48);
    blocks.pop_back();
    const std::uint32_t tag_after_push = alloc.class_at(idx).free_list_tag();
    void* again = alloc.allocate(48);
    ASSERT_NE(again, nullptr);
    const std::uint32_t tag_empty_2 = alloc.class_at(idx).free_list_tag();

    // Every push/pop bumps the tag by exactly one.
    EXPECT_EQ(tag_after_push, tag_empty_1 + 1u);
    EXPECT_EQ(tag_empty_2, tag_after_push + 1u);
    // The empty state must be distinguishable from the previous empty state —
    // with the P1-13 bug both were tag 0 and a stale {ptr=0, tag=0} snapshot
    // could CAS successfully, handing the same block to two threads.
    EXPECT_NE(tag_empty_2, tag_empty_1);
    EXPECT_NE(tag_empty_2, 0u);
}

TEST(MemoryP1Test, AbaNoDoubleHandoutUnderThreads) {
    // Drain one size class concurrently from every thread. Because the class is
    // capped at a single slab, the total number of successful allocations must
    // be exactly the class capacity, and no block may be handed out twice.
    slab_allocator::config cfg;
    cfg.initial_slabs_per_class = 1;
    cfg.max_slabs_per_class = 1;
    slab_allocator alloc(cfg);

    constexpr std::uint32_t kPayload = 48;
    const std::uint32_t idx = slab_allocator::class_index_for_payload(kPayload);
    const std::size_t capacity = alloc.class_at(idx).items_per_slab();

    constexpr std::size_t kThreads = 8;
    std::vector<std::vector<void*>> per_thread(kThreads);
    std::atomic<std::size_t> total{0};

    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    for (std::size_t t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t] {
            auto& mine = per_thread[t];
            while (void* p = alloc.allocate(kPayload)) {
                mine.push_back(p);
                total.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }
    for (auto& th : threads) th.join();

    EXPECT_EQ(total.load(), capacity);

    // No duplicates: a block handed out twice while both owners still hold it
    // is exactly the ABA failure mode. Compare as integers so the sort does not
    // depend on pointer ordering of unrelated objects.
    std::vector<std::uintptr_t> sorted;
    sorted.reserve(capacity);
    std::size_t total_checked = 0;
    for (const auto& mine : per_thread) {
        total_checked += mine.size();
        for (void* p : mine) sorted.push_back(reinterpret_cast<std::uintptr_t>(p));
    }
    ASSERT_EQ(total_checked, capacity);
    ASSERT_EQ(sorted.size(), capacity);
    std::sort(sorted.begin(), sorted.end());
    EXPECT_EQ(std::adjacent_find(sorted.begin(), sorted.end()), sorted.end());

    for (auto& mine : per_thread) {
        for (void* p : mine) alloc.deallocate(p, kPayload);
    }
}

// ============================================================================
// P1-14 — deallocation must not trust the caller-supplied size
// ============================================================================

TEST(MemoryP1Test, WrongSizeDeallocationIsRejectedAndCounted) {
    slab_allocator alloc;
    const std::uint32_t cls = slab_allocator::class_index_for_payload(128);

    void* p = alloc.allocate(128);
    ASSERT_NE(p, nullptr);
    std::memset(p, 0x5A, 128);

    const std::uint32_t free_before = alloc.class_at(cls).free_count();
    const std::uint64_t rejects_before = alloc.dealloc_rejects_total();

    // A claim far larger than the block used to be routed to
    // ::operator delete (heap corruption) or onto a bigger class's free list.
    alloc.deallocate(p, 100000);
    EXPECT_EQ(alloc.dealloc_rejects_total(), rejects_before + 1);
    EXPECT_EQ(static_cast<int>(alloc.snapshot().last_reject_reason),
              static_cast<int>(slab_dealloc_reject::size_class_mismatch));
    // Nothing was freed and nothing was corrupted: the block still belongs to
    // its class.
    EXPECT_EQ(alloc.class_at(cls).free_count(), free_before);

    // A claim that fits the block is still accepted and routed by the
    // *recorded* class, so the honest free keeps working.
    alloc.deallocate(p, 128);
    EXPECT_EQ(alloc.class_at(cls).free_count(), free_before + 1);
    EXPECT_EQ(alloc.dealloc_rejects_total(), rejects_before + 1);
}

TEST(MemoryP1Test, LargeBlockRejectsSlabSizedClaim) {
    slab_allocator alloc;
    void* big = alloc.allocate(100000);   // > kMaxClassSize → global heap
    ASSERT_NE(big, nullptr);
    EXPECT_EQ(slab_allocator::block_capacity(big), 100000u);

    const std::uint64_t before = alloc.dealloc_rejects_total();
    alloc.deallocate(big, 128);            // slab-sized claim on a large block
    EXPECT_EQ(alloc.dealloc_rejects_total(), before + 1);

    alloc.deallocate(big, 100000);         // correct claim frees it
    EXPECT_EQ(alloc.dealloc_rejects_total(), before + 1);
}

TEST(MemoryP1Test, SlabPtrCannotPassAWrongSize) {
    slab_allocator alloc;
    slab_ptr empty;
    EXPECT_FALSE(empty);
    EXPECT_EQ(empty.get(), nullptr);

    const std::uint32_t cls = slab_allocator::class_index_for_payload(128);
    const std::uint32_t free_before = alloc.class_at(cls).free_count();
    const std::uint64_t rejects_before = alloc.dealloc_rejects_total();

    {
        slab_ptr h = alloc.allocate_owned(128);
        ASSERT_TRUE(h);
        EXPECT_EQ(h.size(), 128u);
        EXPECT_EQ(h.owner(), &alloc);
        // The header — not the caller — states the block capacity.
        EXPECT_GE(slab_allocator::block_capacity(h.get()), 128u);
        std::memset(h.get(), 0x11, h.size());

        // Ownership transfer must not double-free.
        slab_ptr moved = std::move(h);
        EXPECT_TRUE(moved);
        EXPECT_FALSE(h);
    }   // moved released here

    // A full allocate/release cycle through the handle restores the free list,
    // and no size was ever handed to the allocator by the caller.
    EXPECT_EQ(alloc.class_at(cls).free_count(), free_before);
    EXPECT_EQ(alloc.class_at(cls).live_blocks(), 0u);
    EXPECT_EQ(alloc.dealloc_rejects_total(), rejects_before);

    // release() hands the raw block back without freeing it.
    slab_ptr h2 = alloc.allocate_owned(128);
    ASSERT_TRUE(h2);
    void* raw = h2.release();
    EXPECT_FALSE(h2);
    EXPECT_NE(raw, nullptr);
    alloc.deallocate(raw, 128);
    EXPECT_EQ(alloc.dealloc_rejects_total(), rejects_before);
}

TEST(MemoryP1Test, PayloadsStayAlignedAndWithinCapacity) {
    slab_allocator alloc;
    for (std::size_t sz : {std::size_t{1}, std::size_t{64}, std::size_t{1024},
                           std::size_t{65520}, std::size_t{65537}}) {
        void* p = alloc.allocate(sz);
        ASSERT_NE(p, nullptr);
        EXPECT_EQ(reinterpret_cast<std::uintptr_t>(p) % alignof(std::max_align_t), 0u);
        EXPECT_GE(slab_allocator::block_capacity(p), sz);
        std::memset(p, 0xC3, sz);
        alloc.deallocate(p, sz);
        EXPECT_EQ(alloc.dealloc_rejects_total(), 0u);
    }
}

// ============================================================================
// P1-15 — owner NUMA node recorded per block and used for routing
// ============================================================================

TEST(MemoryP1Test, NumaOwnerNodeIsRecordedAndUsedForRouting) {
    numa_aware_slab_allocator::config cfg;
    cfg.max_nodes = 2;
    numa_aware_slab_allocator alloc(cfg);
    ASSERT_GE(alloc.num_nodes(), 1);

    const std::uint32_t cls = slab_allocator::class_index_for_payload(128);

    // Allocate straight from node 0's allocator, then free through the facade.
    // The facade must route by the header's owner node rather than by
    // current_allocator() (the comment above the old code said as much while
    // the code did the opposite).
    slab_allocator& node0 = alloc.node_allocator(0);
    void* p = node0.allocate(128);
    ASSERT_NE(p, nullptr);
    EXPECT_EQ(slab_allocator::owner_node_of(p), 0);

    const std::uint32_t free_before = node0.class_at(cls).free_count();
    alloc.deallocate(p, 128);
    EXPECT_EQ(node0.class_at(cls).free_count(), free_before + 1);

    // The RAII path routes identically and records a valid owner for the
    // current thread's node.
    slab_ptr h = alloc.allocate_owned(128);
    ASSERT_TRUE(h);
    const int owner = slab_allocator::owner_node_of(h.get());
    ASSERT_GE(owner, 0);
    ASSERT_LT(owner, alloc.num_nodes());
    const std::uint32_t owner_free_before =
        alloc.node_allocator(owner).class_at(cls).free_count();
    h.reset();
    EXPECT_EQ(alloc.node_allocator(owner).class_at(cls).free_count(),
              owner_free_before + 1);

    // A non-NUMA allocator records "no owner".
    slab_allocator plain;
    void* q = plain.allocate(64);
    ASSERT_NE(q, nullptr);
    EXPECT_EQ(slab_allocator::owner_node_of(q), -1);
    plain.deallocate(q, 64);
}

// ============================================================================
// P1-16 — gauge vs delta accounting, explicit rate verdict, checked magnitudes
// ============================================================================

TEST(MemoryP1Test, MixedReportCallsKeepGaugeEqualToTrueValue) {
    memory_monitor::config cfg;
    cfg.max_memory_bytes.store(1000000);
    memory_monitor mon(cfg);

    // report_memory() is the absolute gauge; report_insert/report_evict are
    // deltas. Interleaving them must never drift: the gauge is authoritative.
    mon.report_memory(1000);
    EXPECT_EQ(mon.get_stats().current_memory_bytes, 1000u);

    mon.report_insert(500);
    EXPECT_EQ(mon.get_stats().current_memory_bytes, 1500u);

    mon.report_evict(200);
    EXPECT_EQ(mon.get_stats().current_memory_bytes, 1300u);

    mon.report_memory(4096);
    EXPECT_EQ(mon.get_stats().current_memory_bytes, 4096u);

    mon.report_insert(96);
    EXPECT_EQ(mon.get_stats().current_memory_bytes, 4192u);

    mon.report_evict(4192);
    EXPECT_EQ(mon.get_stats().current_memory_bytes, 0u);
}

TEST(MemoryP1Test, NegativeDeltaMagnitudeIsRangeChecked) {
    rate_limiter limiter(/*detect_increase=*/true);
    limiter.set_window_size(8);
    limiter.add_value(0);
    limiter.add_value(-100);   // declining window

    // Magnitudes are magnitudes: a negative delta returns its absolute value.
    EXPECT_EQ(limiter.throttle(-5), 5u);
    EXPECT_EQ(limiter.throttle(5), 5u);

    // INT64_MIN: std::abs() is undefined and the old cast produced ~2^64.
    // The magnitude is 2^63, which is representable as size_t.
    EXPECT_EQ(limiter.throttled_magnitude(std::numeric_limits<std::int64_t>::min()),
              std::uint64_t{1} << 63);
    EXPECT_EQ(limiter.throttle(std::numeric_limits<std::int64_t>::min()),
              static_cast<std::size_t>(std::uint64_t{1} << 63));
}

TEST(MemoryP1Test, RateLimiterExposesExplicitExceededVerdict) {
    rate_limiter limiter(/*detect_increase=*/true);
    limiter.set_window_size(8);

    // Fewer than two samples → no verdict, no throttle.
    EXPECT_FALSE(limiter.exceeded(1));

    limiter.add_value(0);
    limiter.add_value(10000);   // +10000 bytes per sample
    EXPECT_TRUE(limiter.exceeded(1000));
    EXPECT_FALSE(limiter.exceeded(100000));
    EXPECT_FALSE(limiter.exceeded(0));       // 0 disables the check
    EXPECT_FALSE(limiter.exceeded(-5));

    // detect_increase == false watches decline instead.
    rate_limiter declining(/*detect_increase=*/false);
    declining.set_window_size(8);
    declining.add_value(10000);
    declining.add_value(0);     // -10000 per sample
    EXPECT_TRUE(declining.exceeded(1000));
}

TEST(MemoryP1Test, GrowthRateFlagUsesTheExplicitVerdict) {
    memory_monitor::config cfg;
    cfg.max_memory_bytes.store(1000000);      // occupancy stays low
    cfg.max_growth_rate_bytes.store(1000);    // bytes per sample budget
    cfg.rate_window_size.store(4);
    memory_monitor mon(cfg);

    mon.report_memory(0);
    EXPECT_EQ(mon.current_state(), memory_monitor::state::normal);

    mon.report_memory(10000);                 // 10000 > 1000 → exceeded
    EXPECT_EQ(mon.current_state(), memory_monitor::state::throttled);

    // A flat window must clear the latch again. The old code compared a rate
    // against a byte budget and latched permanently on any growth.
    for (int i = 0; i < 8; ++i) mon.report_memory(10000);
    EXPECT_EQ(mon.current_state(), memory_monitor::state::normal);
}

// ============================================================================
// P1-17 — the memory cap is a hard limit
// ============================================================================

TEST(MemoryP1Test, ConcurrentGuardsCannotExceedTheCap) {
    constexpr std::size_t kCap = std::size_t{1} << 20;   // 1 MiB
    constexpr std::size_t kChunk = 4096;
    constexpr std::size_t kThreads = 8;
    constexpr std::size_t kIterations = 400;

    memory_monitor::config cfg;
    cfg.max_memory_bytes.store(kCap);
    memory_monitor mon(cfg);

    std::atomic<std::size_t> peak{0};
    std::atomic<std::size_t> committed{0};
    std::atomic<std::size_t> rejected{0};

    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    for (std::size_t t = 0; t < kThreads; ++t) {
        threads.emplace_back([&] {
            for (std::size_t i = 0; i < kIterations; ++i) {
                memory_guard guard(mon, kChunk);
                if (!guard) {
                    rejected.fetch_add(1, std::memory_order_relaxed);
                    continue;
                }
                // Hold the reservation across a scheduling point so the limit
                // has to be enforced by the reservation, not by timing luck.
                std::this_thread::yield();
                const std::size_t now =
                    mon.get_stats().current_memory_bytes + mon.reserved_bytes();
                std::size_t prev = peak.load(std::memory_order_relaxed);
                while (now > prev &&
                       !peak.compare_exchange_weak(prev, now, std::memory_order_relaxed)) {
                }
                guard.commit();
                committed.fetch_add(kChunk, std::memory_order_relaxed);
            }
        });
    }
    for (auto& th : threads) th.join();

    EXPECT_GT(committed.load(), 0u);
    EXPECT_GT(rejected.load(), 0u);
    // Committed + reserved never exceeded the cap, even transiently.
    EXPECT_LE(peak.load(), kCap);
    EXPECT_LE(mon.get_stats().current_memory_bytes, kCap);
    // Every reservation was returned.
    EXPECT_EQ(mon.reserved_bytes(), 0u);
    EXPECT_EQ(mon.get_stats().reserved_memory_bytes, 0u);
}

TEST(MemoryP1Test, GuardReleasesReservationOnDestruction) {
    memory_monitor::config cfg;
    cfg.max_memory_bytes.store(10000);
    memory_monitor mon(cfg);

    {
        memory_guard guard(mon, 4000);
        ASSERT_TRUE(guard);
        EXPECT_EQ(mon.reserved_bytes(), 4000u);
        EXPECT_TRUE(guard.admitted());
    }
    EXPECT_EQ(mon.reserved_bytes(), 0u);
    EXPECT_EQ(mon.get_stats().current_memory_bytes, 0u);

    // A moved guard keeps the reservation until the new owner destroys it.
    {
        memory_guard first(mon, 4000);
        ASSERT_TRUE(first);
        memory_guard second(std::move(first));
        EXPECT_FALSE(first);
        EXPECT_EQ(mon.reserved_bytes(), 4000u);
        second.commit();
    }
    EXPECT_EQ(mon.reserved_bytes(), 0u);
    EXPECT_EQ(mon.get_stats().current_memory_bytes, 4000u);

    // Requests that cannot fit are refused by the reservation, not by a
    // best-effort comparison.
    memory_guard too_big(mon, 100000);
    EXPECT_FALSE(too_big);
    EXPECT_EQ(mon.reserved_bytes(), 0u);

    // An unlimited monitor (max_memory_bytes == 0) still keeps the reservation
    // ledger symmetric, so the gauge cannot underflow.
    memory_monitor::config unbounded_cfg;   // max_memory_bytes stays 0
    memory_monitor unbounded(unbounded_cfg);
    {
        memory_guard g(unbounded, 1234);
        ASSERT_TRUE(g);
        EXPECT_EQ(unbounded.reserved_bytes(), 1234u);
    }
    EXPECT_EQ(unbounded.reserved_bytes(), 0u);
    EXPECT_EQ(unbounded.get_stats().reserved_memory_bytes, 0u);
}

TEST(MemoryP1Test, ShouldAdmitHonoursTheHardCap) {
    memory_monitor::config cfg;
    cfg.max_memory_bytes.store(1000);
    // Keep the critical band above the projected occupancy used below so the
    // assertions isolate the cap rather than the probabilistic tier.
    cfg.critical_fraction.store(0.99);
    cfg.throttle_fraction.store(0.9);
    cfg.critical_fraction.store(0.99);
    cfg.throttle_fraction.store(0.9);
    memory_monitor mon(cfg);

    mon.report_memory(500);
    EXPECT_TRUE(mon.should_admit(400));   // 90% of the cap: admitted
    EXPECT_FALSE(mon.should_admit(501));  // would exceed the cap: refused

    // Exactly at the cap: nothing more fits, not even a zero-byte insert.
    mon.report_memory(1000);
    EXPECT_FALSE(mon.should_admit(0));
    EXPECT_FALSE(mon.should_admit(1));

    // Over the cap: everything is refused.
    mon.report_memory(1200);
    EXPECT_FALSE(mon.should_admit(0));

    // Going back under the cap restores admission.
    mon.report_memory(100);
    EXPECT_TRUE(mon.should_admit(0));
}

// ============================================================================
// P1-18 — per-class live/peak/failure counters and a stable snapshot
// ============================================================================

TEST(MemoryP1Test, PerClassLivePeakAndFailureCounters) {
    slab_allocator::config cfg;
    cfg.initial_slabs_per_class = 1;
    cfg.max_slabs_per_class = 1;
    slab_allocator alloc(cfg);

    const std::uint32_t cls = slab_allocator::class_index_for_payload(128);
    std::vector<void*> held;
    for (int i = 0; i < 4; ++i) {
        void* p = alloc.allocate(128);
        ASSERT_NE(p, nullptr);
        held.push_back(p);
    }

    auto& c = alloc.class_at(cls);
    EXPECT_EQ(c.live_blocks(), 4u);
    EXPECT_EQ(c.live_bytes(), 4ull * c.class_size());
    EXPECT_GE(c.peak_bytes(), c.live_bytes());
    EXPECT_EQ(c.total_allocations(), 4u);
    EXPECT_EQ(c.total_frees(), 0u);

    // Exhaust the class: with one slab the next request cannot be satisfied and
    // must be counted with a reason instead of silently returning nullptr.
    while (void* p = alloc.allocate(128)) {
        held.push_back(p);
    }
    const auto snap = alloc.snapshot();
    EXPECT_GT(snap.alloc_failures_total, 0u);
    ASSERT_EQ(snap.classes.size(), static_cast<std::size_t>(slab_allocator::num_classes()));
    EXPECT_GT(snap.classes[cls].alloc_failures, 0u);
    EXPECT_EQ(static_cast<int>(snap.classes[cls].last_failure_reason),
              static_cast<int>(slab_alloc_failure::max_slabs_reached));
    EXPECT_EQ(snap.alloc_failures_total, alloc.alloc_failures_total());

    const std::size_t held_count = held.size();
    for (void* p : held) alloc.deallocate(p, 128);
    EXPECT_EQ(alloc.class_at(cls).live_blocks(), 0u);
    EXPECT_EQ(alloc.class_at(cls).live_bytes(), 0u);
    EXPECT_EQ(alloc.class_at(cls).total_frees(), held_count);
    // The peak is a high-water mark: it must not come back down.
    EXPECT_GE(alloc.class_at(cls).peak_bytes(), alloc.class_at(cls).live_bytes());
    EXPECT_GT(alloc.class_at(cls).peak_bytes(), 0u);
}

TEST(MemoryP1Test, UtilizationAndLiveBytesStayConsistentUnderConcurrency) {
    slab_allocator alloc;

    constexpr std::size_t kThreads = 8;
    constexpr std::size_t kIterations = 20000;

    std::atomic<bool> stop{false};
    std::atomic<int> out_of_range{0};

    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    for (std::size_t t = 0; t < kThreads; ++t) {
        threads.emplace_back([&alloc, t] {
            for (std::size_t i = 0; i < kIterations; ++i) {
                const std::size_t sz = 64 + ((i + t) % 7) * 64;
                void* p = alloc.allocate(sz);
                if (p) alloc.deallocate(p, sz);
            }
        });
    }

    std::thread reader([&] {
        while (!stop.load(std::memory_order_relaxed)) {
            const auto snap = alloc.snapshot();
            for (const auto& cs : snap.classes) {
                // P1-18 acceptance criterion: the utilization an operator sees
                // is always inside [0, 1], no matter how the two monotonic
                // counters interleave. (The raw counters themselves may
                // transiently invert under concurrency — the derived live count
                // and the utilization are clamped, which is why only these are
                // asserted here.)
                if (!(cs.utilization >= 0.0 && cs.utilization <= 1.0)) {
                    out_of_range.fetch_add(1, std::memory_order_relaxed);
                }
                if (cs.live_bytes !=
                    cs.live_blocks * static_cast<std::uint64_t>(cs.class_size)) {
                    out_of_range.fetch_add(1, std::memory_order_relaxed);
                }
            }
        }
    });

    for (auto& th : threads) th.join();
    stop.store(true, std::memory_order_relaxed);
    reader.join();

    EXPECT_EQ(out_of_range.load(), 0) << "utilization must stay inside [0, 1]";
    EXPECT_EQ(alloc.dealloc_rejects_total(), 0u);
    EXPECT_EQ(alloc.snapshot().alloc_failures_total, 0u);

    // Once the churn has quiesced every class must be exactly empty and the
    // monotonic counters must agree.
    for (const auto& cs : alloc.get_stats()) {
        EXPECT_EQ(cs.live_blocks, 0u);
        EXPECT_EQ(cs.live_bytes, 0u);
        EXPECT_EQ(cs.total_frees, cs.total_allocations);
        EXPECT_GE(cs.utilization, 0.0);
        EXPECT_LE(cs.utilization, 1.0);
    }
}

TEST(MemoryP1Test, SnapshotAggregatesNumaNodes) {
    numa_aware_slab_allocator::config cfg;
    cfg.max_nodes = 2;
    numa_aware_slab_allocator alloc(cfg);

    std::vector<slab_ptr> held;
    for (int i = 0; i < 8; ++i) {
        slab_ptr h = alloc.allocate_owned(128);
        ASSERT_TRUE(h);
        held.push_back(std::move(h));
    }

    const auto snap = alloc.snapshot();
    EXPECT_EQ(snap.classes.size(), static_cast<std::size_t>(alloc.kNumClasses));
    EXPECT_GT(snap.live_bytes_total, 0u);
    EXPECT_GE(snap.peak_bytes_total, snap.live_bytes_total);
    for (const auto& cs : snap.classes) {
        EXPECT_GE(cs.utilization, 0.0);
        EXPECT_LE(cs.utilization, 1.0);
    }
    held.clear();
    EXPECT_EQ(alloc.snapshot().live_bytes_total, 0u);
}

// ---------------------------------------------------------------------------
// Real payload billing (max_memory as a byte budget, not an item-count proxy)
// ---------------------------------------------------------------------------
//
// Without a size calculator the cache bills a compile-time constant per item,
// so a 1 KB std::string and a 10-byte one are billed identically and max_memory
// is really an item count. These tests pin the API that fixes the accounting.

TEST(MemoryPayloadSizes, DeepSizeReportsContainerFootprint) {
    // deep_size reports the CONTAINER OBJECT plus its heap payload, which is
    // what a byte budget has to account for. It uses capacity(), not size():
    // the allocated block is what occupies memory. Note that capacity() is
    // never 0 for std::string -- the small-string buffer counts -- so the
    // assertions are >= rather than exact.
    EXPECT_GE(lru::detail::deep_size(std::string{}), sizeof(std::string));
    EXPECT_GE(lru::detail::deep_size(std::string(1000, 'x')),
              sizeof(std::string) + 1000u);

    std::vector<int> v;
    v.reserve(64);
    EXPECT_EQ(v.capacity(), 64u);
    EXPECT_EQ(lru::detail::deep_size(v),
              sizeof(std::vector<int>) + 64u * sizeof(int));

    // Trivially-copyable types fall back to sizeof(T).
    EXPECT_EQ(lru::detail::deep_size(int{7}), sizeof(int));
}

TEST(MemoryPayloadSizes, ValueSizeCalculatorIsHonoured) {
    lru::safe_cache<int, std::string> c(64);

    // Baseline: the default billing ignores the payload entirely.
    c.set(0, std::string(4000, 'z'));
    const std::size_t default_billed = c.current_memory();

    lru::safe_cache<int, std::string> c2(64);
    c2.use_real_payload_sizes();
    c2.set(0, std::string(4000, 'z'));
    const std::size_t real_billed = c2.current_memory();

    EXPECT_GT(real_billed, default_billed)
        << "billing the real payload must exceed the constant-per-item default";

    // Two items whose payloads differ by 100x must be billed differently under
    // real accounting; under the default they are billed identically.
    lru::safe_cache<int, std::string> a(64), b(64);
    a.use_real_payload_sizes();
    b.use_real_payload_sizes();

    a.set(1, std::string(10, 'p'));
    const std::size_t one_small = a.current_memory();
    a.set(2, std::string(1000, 'p'));
    const std::size_t small_plus_large = a.current_memory();

    b.set(1, std::string(10, 'p'));
    b.set(2, std::string(10, 'p'));
    const std::size_t two_small = b.current_memory();

    EXPECT_GT(small_plus_large, two_small)
        << "a 1000-byte payload must cost more than a 10-byte one";
    EXPECT_GT(small_plus_large - one_small, 900u);
}

TEST(MemoryPayloadSizes, KeySizeCalculatorIsHonoured) {
    lru::safe_cache<std::string, int> c(64);
    c.set_key_size_calculator(
        [](const std::string& k) { return lru::detail::deep_size(k); });

    c.set(std::string(2000, 'k'), 1);
    EXPECT_GT(c.current_memory(), 2000u);
}
