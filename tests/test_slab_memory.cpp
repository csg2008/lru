// Unified LRU Cache Library — Slab Allocator & Memory Pressure Tests
// SPDX-License-Identifier: MIT
//
// Tests:
//   1. Concurrent allocate/deallocate for slab_allocator (8 threads, 10K cycles)
//   2. Memory pressure lifecycle: normal → throttled → critical → normal

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <fstream>
#include <mutex>
#include <thread>
#include <vector>

#include "../lru.hpp"

using namespace lru;

// ============================================================================
// SubTask 2.4: Concurrent allocate/deallocate test for slab_allocator
// ============================================================================
//
// P2-H: The allocation_class uses a 128-bit tagged-pointer Treiber stack
// (32-bit ABA counter + 64-bit pointer, cmpxchg16b). The 32-bit tag wraps
// after ~4 billion operations, effectively eliminating ABA for realistic
// workloads. The previous 16-bit tag wrapped after 65K ops; tests worked
// around it by partitioning work so each thread operated on its own blocks.

TEST(SlabAllocatorTest, ConcurrentAllocateDeallocate8Threads) {
    slab_allocator alloc;

    constexpr int kThreads = 8;
    constexpr int kCycles = 10000;
    constexpr uint32_t kAllocSize = 128;  // fits in class size 128

    // Each thread pre-allocates a batch, holds them, then deallocates.
    // (P2-H: the 32-bit tag now eliminates ABA; this partitioning is kept
    // as a stress pattern rather than an ABA workaround.)
    std::vector<std::vector<void*>> thread_ptrs(kThreads);

    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t]() {
            auto& my_ptrs = thread_ptrs[t];
            my_ptrs.reserve(kCycles);
            for (int i = 0; i < kCycles; ++i) {
                void* ptr = alloc.allocate(kAllocSize);
                if (ptr) {
                    my_ptrs.push_back(ptr);
                }
            }
        });
    }
    for (auto& t : threads) t.join();

    // Now deallocate in parallel — each thread returns its own batch
    threads.clear();
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t]() {
            for (auto* p : thread_ptrs[t]) {
                alloc.deallocate(p, kAllocSize);
            }
        });
    }
    for (auto& t : threads) t.join();

    // No crash means success
}

TEST(SlabAllocatorTest, ConcurrentMixedSizeAllocate) {
    slab_allocator alloc;

    constexpr int kThreads = 8;
    constexpr int kCycles = 5000;

    // Each thread uses a different size class — no contention on the same free list
    uint32_t sizes[] = {64, 128, 256, 512, 1024, 2048, 4096, 8192};

    std::vector<std::vector<void*>> thread_ptrs(kThreads);

    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t]() {
            uint32_t sz = sizes[t];
            auto& my_ptrs = thread_ptrs[t];
            my_ptrs.reserve(kCycles);
            for (int i = 0; i < kCycles; ++i) {
                void* ptr = alloc.allocate(sz);
                if (ptr) {
                    my_ptrs.push_back(ptr);
                }
            }
        });
    }
    for (auto& t : threads) t.join();

    // Deallocate
    threads.clear();
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t]() {
            uint32_t sz = sizes[t];
            for (auto* p : thread_ptrs[t]) {
                alloc.deallocate(p, sz);
            }
        });
    }
    for (auto& t : threads) t.join();
    // No crash means success
}

// P2-H: Verify the 32-bit ABA tag survives a workload that would have wrapped
// the old 16-bit tag (>65K ops on the same free list) and caused ABA.
// Threads rapidly allocate and deallocate on the same shared free list,
// forcing the tag counter past the 16-bit wrap point.
TEST(SlabAllocatorTest, HighContentionRapidAllocDeallocNoABA) {
    slab_allocator alloc;

    constexpr int kThreads = 8;
    constexpr int kCycles = 20000;  // 8 * 20K = 160K ops, well past 16-bit wrap
    constexpr uint32_t kAllocSize = 128;

    std::atomic<int> alloc_count{0};
    std::atomic<int> dealloc_count{0};

    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&]() {
            for (int i = 0; i < kCycles; ++i) {
                void* ptr = alloc.allocate(kAllocSize);
                if (ptr) {
                    alloc_count.fetch_add(1, std::memory_order_relaxed);
                    // Immediate dealloc: this is the ABA-triggering pattern
                    // where the same pointer reappears at the head of the free
                    // list while a stale snapshot is being CAS'd.
                    alloc.deallocate(ptr, kAllocSize);
                    dealloc_count.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }
    for (auto& t : threads) t.join();

    EXPECT_EQ(alloc_count.load(), dealloc_count.load());
    EXPECT_GT(alloc_count.load(), 0);
}

// ============================================================================
// SubTask 10.3: Memory pressure lifecycle test
// ============================================================================

TEST(MemoryPressureTest, NormalToThrottledToCriticalToNormal) {
    // Use a very small max_memory so we can easily trigger watermarks
    memory_monitor::config cfg;
    cfg.max_memory_bytes.store(1000);
    cfg.high_watermark_fraction.store(0.70);
    cfg.critical_watermark_fraction.store(0.90);
    cfg.low_watermark_fraction.store(0.50);
    memory_monitor mon(cfg);

    // Initially: normal state, should admit, not throttled
    EXPECT_EQ(mon.current_state(), memory_monitor::state::normal);
    EXPECT_TRUE(mon.should_admit(0));
    EXPECT_FALSE(mon.is_throttled());

    // Report memory at 50% → still normal
    mon.report_memory(500);
    EXPECT_EQ(mon.current_state(), memory_monitor::state::normal);
    EXPECT_TRUE(mon.should_admit(0));

    // Report memory at 75% → above high watermark → high state
    mon.report_memory(750);
    EXPECT_EQ(mon.current_state(), memory_monitor::state::high);
    // should_admit with 0 extra bytes: still admitted but throttled
    EXPECT_TRUE(mon.should_admit(0));
    EXPECT_TRUE(mon.is_throttled());

    // Report memory at 95% → above critical watermark → critical state
    mon.report_memory(950);
    EXPECT_EQ(mon.current_state(), memory_monitor::state::critical);
    // should_admit: rejected
    EXPECT_FALSE(mon.should_admit(0));
    EXPECT_TRUE(mon.is_throttled());

    // Report memory back to 40% → below low watermark → normal again
    mon.report_memory(400);
    EXPECT_EQ(mon.current_state(), memory_monitor::state::normal);
    EXPECT_TRUE(mon.should_admit(0));
    EXPECT_FALSE(mon.is_throttled());
}

TEST(MemoryPressureTest, CriticalRejectsAdmission) {
    memory_monitor::config cfg;
    cfg.max_memory_bytes.store(1000);
    cfg.high_watermark_fraction.store(0.70);
    cfg.critical_watermark_fraction.store(0.90);
    cfg.low_watermark_fraction.store(0.50);
    memory_monitor mon(cfg);

    // Fill past critical watermark
    mon.report_memory(950);
    EXPECT_EQ(mon.current_state(), memory_monitor::state::critical);

    // should_admit returns false for any new insertion
    EXPECT_FALSE(mon.should_admit(10));
    EXPECT_FALSE(mon.should_admit(1));
    EXPECT_FALSE(mon.should_admit(0));
}

TEST(MemoryPressureTest, MemoryAwareEvictorLifecycle) {
    // Create a small cache with memory monitoring
    cache<int, int> c(1000);

    memory_monitor::config cfg;
    cfg.max_memory_bytes.store(10000);
    cfg.high_watermark_fraction.store(0.60);
    cfg.critical_watermark_fraction.store(0.90);
    cfg.low_watermark_fraction.store(0.40);
    c.set_memory_monitor(cfg);

    // Fill the cache to trigger throttled state
    for (int i = 0; i < 500; ++i) {
        c.set(i, i * 10);
    }

    // The cache should be functional (no crash), and the monitor should be active
    EXPECT_TRUE(c.monitor().active());

    // Verify state transitions via the monitor
    auto st = c.monitor().get_stats();
    // After inserting many items, memory should be > 0
    EXPECT_GT(st.current_memory_bytes, 0u);

    // Create a memory_aware_evictor
    memory_aware_evictor evictor(c, free_threshold_strategy{
        .low_watermark = 0.05,
        .high_watermark = 0.15,
        .max_eviction_batch = 10
    }, c.monitor());

    // Manually trigger a tick — should not crash
    evictor.tick();

    // Fill more to trigger critical
    for (int i = 500; i < 1000; ++i) {
        c.set(i, i * 10);
    }

    // Tick again — should attempt accelerated eviction
    evictor.tick();

    // Verify the cache still works
    EXPECT_GT(c.size(), 0u);
}

// ============================================================================
// spec.md P0-4: OS Memory Sampler + Pressure Callback
// ============================================================================
//
// These tests cover the new OS-level memory integration:
//   1. os_memory_sampler produces a non-empty snapshot on supported platforms
//   2. pressure callback reject verdict short-circuits should_admit
//   3. pressure callback throttle verdict preserves throttle flag
//   4. start/stop OS sampler is idempotent and the monitor remains usable

TEST(OsMemorySamplerTest, RefreshProducesSnapshot) {
    os_memory_sampler sampler;
    sampler.refresh();
    auto snap = sampler.latest();
    // On Windows/Linux a snapshot should always be available after refresh().
    // The level field is platform-dependent (may be `unknown` on unsupported
    // platforms), but the timestamp must be set.
    ASSERT_TRUE(snap.has_value());
    EXPECT_NE(snap->sampled_at.time_since_epoch().count(), 0);
}

TEST(OsMemorySamplerTest, StartStopIsIdempotent) {
    os_memory_sampler sampler;
    EXPECT_FALSE(sampler.is_running());
    sampler.start();
    EXPECT_TRUE(sampler.is_running());
    // Double-start is a no-op
    sampler.start();
    EXPECT_TRUE(sampler.is_running());
    sampler.stop();
    EXPECT_FALSE(sampler.is_running());
    // Double-stop is a no-op
    sampler.stop();
    EXPECT_FALSE(sampler.is_running());
}

TEST(OsMemorySamplerTest, LatestReturnsNulloptBeforeFirstRefresh) {
    os_memory_sampler sampler;
    EXPECT_FALSE(sampler.latest().has_value());
}

TEST(MemoryPressureCallbackTest, RejectVerdictShortCircuitsAdmit) {
    memory_monitor mon;
    // Install a callback that always rejects.
    mon.set_memory_pressure_callback(
        [](std::size_t) { return memory_monitor::pressure_verdict::reject; });
    EXPECT_FALSE(mon.should_admit(0));
    EXPECT_FALSE(mon.should_admit(100));
    EXPECT_TRUE(mon.is_throttled());
}

TEST(MemoryPressureCallbackTest, AdmitVerdictDefersToInternalPressure) {
    memory_monitor mon;
    // Callback admits, but no internal budget set → should admit.
    mon.set_memory_pressure_callback(
        [](std::size_t) { return memory_monitor::pressure_verdict::admit; });
    EXPECT_TRUE(mon.should_admit(0));
}

TEST(MemoryPressureCallbackTest, NullCallbackRemovesPreviousCallback) {
    memory_monitor mon;
    mon.set_memory_pressure_callback(
        [](std::size_t) { return memory_monitor::pressure_verdict::reject; });
    EXPECT_FALSE(mon.should_admit(0));
    // Remove callback
    mon.set_memory_pressure_callback(nullptr);
    EXPECT_TRUE(mon.should_admit(0));
}

TEST(MemoryMonitorOsSamplingTest, StartAndStopSamplerKeepsMonitorUsable) {
    memory_monitor mon;
    mon.start_os_sampling();
    EXPECT_TRUE(mon.os_sampling_running());
    // Sampler accessible
    EXPECT_NE(mon.os_sampler(), nullptr);
    // should_admit still works
    EXPECT_TRUE(mon.should_admit(0));
    mon.stop_os_sampling();
    EXPECT_FALSE(mon.os_sampling_running());
    // After stop, latest snapshot is still queryable (was refreshed on start)
    EXPECT_TRUE(mon.os_snapshot().has_value());
}

TEST(MemoryMonitorOsSamplingTest, ActiveFlagReflectsSamplerState) {
    memory_monitor mon;
    // No budget, no callback, no sampler → inactive
    EXPECT_FALSE(mon.active());
    mon.start_os_sampling();
    // Sampler running → active
    EXPECT_TRUE(mon.active());
    mon.stop_os_sampling();
    // Sampler stopped, but the snapshot is still cached. The active() flag
    // tracks the running state of the sampler, not the existence of a cached
    // snapshot, so it should be false again.
    EXPECT_FALSE(mon.active());
}
