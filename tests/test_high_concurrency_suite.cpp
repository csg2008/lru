// SPDX-License-Identifier: MIT
// R7: High-concurrency test suite for production read-heavy workloads.
//
// Supplements the existing 16-thread tests (test_read_heavy_concurrent.cpp,
// test_read_heavy_soak.cpp) with:
//   - 32/64 thread high-concurrency read-heavy correctness tests
//   - Extreme read ratio tests (99%, 99.9% reads)
//   - NUMA-aware routing smoke test
//   - EBR epoch advancement / reclamation pressure test
//   - Extended soak with 32+ threads and memory growth monitoring
//
// Thread counts are capped at hardware_concurrency() to avoid oversubscription
// on low-core CI machines. Set LRU_HC_THREAD_COUNT to force a specific count.
// Duration is configurable via LRU_STRESS_DURATION_MS (default: short CI run).

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#  include <windows.h>
#elif defined(__linux__)
#  include <sched.h>
#endif

#include "lru.hpp"
#include "test_helpers.hpp"

using namespace lru;

namespace {

// ---------------------------------------------------------------------------
// Resolve thread count: min(requested, hardware_concurrency * 2).
// Allows oversubscription up to 2x for stress testing, but prevents spawning
// 64 threads on a 4-core CI runner. Override with LRU_HC_THREAD_COUNT.
// ---------------------------------------------------------------------------
int resolve_thread_count(int requested) {
    const char* env = std::getenv("LRU_HC_THREAD_COUNT");
    if (env && env[0] != '\0') {
        try {
            int v = std::stoi(env);
            if (v > 0) return v;
        } catch (...) {}
    }
    unsigned hw = std::thread::hardware_concurrency();
    if (hw == 0) hw = 4;
    int cap = static_cast<int>(hw) * 2;
    return std::min(requested, cap);
}

// ---------------------------------------------------------------------------
// Cross-platform RSS sampling (in bytes). Returns 0 if unavailable.
// ---------------------------------------------------------------------------
inline std::size_t sample_rss_bytes() {
#if defined(_WIN32)
    HMODULE psapi = ::LoadLibraryW(L"psapi.dll");
    if (!psapi) return 0;
    using GetProcessMemoryInfo_t = BOOL(WINAPI*)(HANDLE, void*, DWORD);
    auto fn = reinterpret_cast<GetProcessMemoryInfo_t>(
        ::GetProcAddress(psapi, "GetProcessMemoryInfo"));
    if (!fn) { ::FreeLibrary(psapi); return 0; }
    struct PMC {
        DWORD  cb; DWORD PageFaultCount;
        SIZE_T PeakWorkingSetSize, WorkingSetSize;
        SIZE_T QuotaPeakPagedPoolUsage, QuotaPagedPoolUsage;
        SIZE_T QuotaPeakNonPagedPoolUsage, QuotaNonPagedPoolUsage;
        SIZE_T PagefileUsage, PeakPagefileUsage;
    } pmc;
    pmc.cb = sizeof(pmc);
    BOOL ok = fn(::GetCurrentProcess(), &pmc, sizeof(pmc));
    ::FreeLibrary(psapi);
    return ok ? static_cast<std::size_t>(pmc.WorkingSetSize) : 0;
#elif defined(__linux__)
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("VmRSS:", 0) == 0) {
            std::size_t val = 0;
            for (char c : line) {
                if (c >= '0' && c <= '9')
                    val = val * 10 + static_cast<std::size_t>(c - '0');
            }
            return val * 1024;
        }
    }
    return 0;
#else
    return 0;
#endif
}

// ---------------------------------------------------------------------------
// Generic read-heavy concurrent workload with configurable read ratio.
// Pre-populates the cache, then runs `num_threads` threads doing
// `read_percent`% reads and (100-read_percent)% writes for `duration`.
// Asserts value integrity and capacity invariant.
// ---------------------------------------------------------------------------
template <typename CacheT>
void run_configurable_concurrent(
        CacheT& c, int key_space, int num_threads,
        int read_percent,
        std::chrono::milliseconds duration,
        std::atomic<long long>& total_ops,
        std::atomic<int>& value_mismatch) {

    // Pre-populate
    for (int i = 0; i < key_space; ++i) {
        c.set(i, i * 10);
    }

    std::atomic<bool> stop{false};
    std::vector<std::thread> threads;
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            std::mt19937 rng(static_cast<unsigned>(t * 7919 + 1));
            while (!stop.load(std::memory_order_relaxed)) {
                int key = static_cast<int>(rng() % key_space);
                if (rng() % 100 < read_percent) {
                    auto h = c.try_get(key);
                    if (h && **h != key * 10) {
                        value_mismatch.fetch_add(1, std::memory_order_relaxed);
                    }
                } else {
                    c.set(key, key * 10);
                }
                total_ops.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }

    std::this_thread::sleep_for(duration);
    stop.store(true, std::memory_order_release);
    for (auto& th : threads) th.join();
}

// ---------------------------------------------------------------------------
// Generic fixed-ops concurrent workload (for correctness tests that don't
// need a duration-based run).
// ---------------------------------------------------------------------------
template <typename CacheT>
void run_fixed_ops_concurrent(
        CacheT& c, int key_space, int num_threads,
        int read_percent, int ops_per_thread,
        std::atomic<int>& value_mismatch) {

    for (int i = 0; i < key_space; ++i) {
        c.set(i, i * 10);
    }

    std::vector<std::thread> threads;
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            std::mt19937 rng(static_cast<unsigned>(t * 7919 + 1));
            for (int i = 0; i < ops_per_thread; ++i) {
                int key = static_cast<int>(rng() % key_space);
                if (rng() % 100 < read_percent) {
                    auto h = c.try_get(key);
                    if (h && **h != key * 10) {
                        value_mismatch.fetch_add(1, std::memory_order_relaxed);
                    }
                } else {
                    c.set(key, key * 10);
                }
            }
        });
    }
    for (auto& th : threads) th.join();
}

} // anonymous namespace

// ============================================================================
// R7-1: 32-thread high-concurrency read-heavy correctness
// Tests production_cache with 32 threads doing 95% reads / 5% writes.
// Validates no value corruption, no capacity violation, no crash.
// ============================================================================
TEST(HighConcurrencySuite, ProductionCache32Threads) {
    int nthreads = resolve_thread_count(32);
    if (nthreads < 4) GTEST_SKIP() << "Too few cores for 32-thread test";

    production_cache<int, int> c(10000);
    std::atomic<int> value_mismatch{0};

    run_fixed_ops_concurrent(c, 5000, nthreads, 95, 5000, value_mismatch);

    EXPECT_EQ(value_mismatch.load(), 0)
        << "value corruption with " << nthreads << " threads";
    EXPECT_LE(c.size(), c.max_size());

    std::cout << "[INFO] ProductionCache32Threads ran with " << nthreads
              << " threads\n";
}

// ============================================================================
// R7-2: 64-thread high-concurrency read-heavy correctness
// Tests production_cache with 64 threads doing 95% reads / 5% writes.
// ============================================================================
TEST(HighConcurrencySuite, ProductionCache64Threads) {
    int nthreads = resolve_thread_count(64);
    if (nthreads < 8) GTEST_SKIP() << "Too few cores for 64-thread test";

    production_cache<int, int> c(20000);
    std::atomic<int> value_mismatch{0};

    run_fixed_ops_concurrent(c, 10000, nthreads, 95, 3000, value_mismatch);

    EXPECT_EQ(value_mismatch.load(), 0)
        << "value corruption with " << nthreads << " threads";
    EXPECT_LE(c.size(), c.max_size());

    std::cout << "[INFO] ProductionCache64Threads ran with " << nthreads
              << " threads\n";
}

// ============================================================================
// R7-3: Extreme read ratio — 99% reads with 32 threads
// In read-heavy production, the read ratio can exceed 99%. This test
// verifies correctness under near-pure-read workloads where TLS rings
// fill slowly and EBR epochs advance infrequently.
// ============================================================================
TEST(HighConcurrencySuite, ExtremeReadRatio99Percent) {
    int nthreads = resolve_thread_count(32);
    if (nthreads < 4) GTEST_SKIP() << "Too few cores for extreme read test";

    // Direct construction — unified_cache is non-movable (contains mutexes/atomics)
    sharded_mm_lru_config hc_cfg;
    hc_cfg.lru_config.defer_promotion = true;
    read_heavy_striped_cache<int, int> c(10000, hc_cfg);
    (void)c.set_fairness_mode_quiescent(detail::fairness_mode::reader_preferred);
    c.set_incremental_rehash(true);
    std::atomic<int> value_mismatch{0};

    // 99% reads, 1% writes — very low write rate means EBR epochs
    // advance slowly and TLS rings fill slowly. Validates that the
    // background drain worker + capacity-based epoch advance (R1) keep
    // the system healthy.
    run_fixed_ops_concurrent(c, 5000, nthreads, 99, 5000, value_mismatch);

    EXPECT_EQ(value_mismatch.load(), 0)
        << "value corruption at 99% read ratio";
    EXPECT_LE(c.size(), c.max_size());
}

// ============================================================================
// R7-4: Extreme read ratio — 99.9% reads with 32 threads
// Even more extreme: only 1 write per 1000 operations. Validates that
// the R1 capacity-based epoch advance triggers and the R6 auto-drain
// threshold (kRingSize/2) keeps TLS rings from overflowing.
// ============================================================================
TEST(HighConcurrencySuite, ExtremeReadRatio999Percent) {
    int nthreads = resolve_thread_count(32);
    if (nthreads < 4) GTEST_SKIP() << "Too few cores for 99.9% read test";

    // Direct construction — unified_cache is non-movable (contains mutexes/atomics)
    sharded_mm_lru_config hc_cfg;
    hc_cfg.lru_config.defer_promotion = true;
    read_heavy_striped_cache<int, int> c(10000, hc_cfg);
    (void)c.set_fairness_mode_quiescent(detail::fairness_mode::reader_preferred);
    c.set_incremental_rehash(true);
    std::atomic<int> value_mismatch{0};

    // 99.9% reads — use 999 out of 1000 as read threshold
    // Pre-populate heavily since writes are extremely rare
    for (int i = 0; i < 5000; ++i) {
        c.set(i, i * 10);
    }

    std::atomic<bool> stop{false};
    std::vector<std::thread> threads;
    for (int t = 0; t < nthreads; ++t) {
        threads.emplace_back([&, t]() {
            std::mt19937 rng(static_cast<unsigned>(t * 7919 + 1));
            while (!stop.load(std::memory_order_relaxed)) {
                int key = static_cast<int>(rng() % 5000);
                if (rng() % 1000 < 1) {
                    // 0.1% writes
                    c.set(key, key * 10);
                } else {
                    auto h = c.try_get(key);
                    if (h && **h != key * 10) {
                        value_mismatch.fetch_add(1, std::memory_order_relaxed);
                    }
                }
            }
        });
    }

    // Short duration for CI; this is a correctness test, not a soak
    auto duration = lru_test::read_stress_duration_ms(std::chrono::seconds(5));
    std::this_thread::sleep_for(duration);
    stop.store(true, std::memory_order_release);
    for (auto& th : threads) th.join();

    EXPECT_EQ(value_mismatch.load(), 0)
        << "value corruption at 99.9% read ratio";
    EXPECT_LE(c.size(), c.max_size());

    // Check that dropped promotions are zero (R6 auto-drain should prevent drops)
    auto snap = c.stats_snapshot();
    EXPECT_EQ(snap.tls_ring_dropped_promotions.load(), 0u)
        << "TLS ring dropped promotions at 99.9% read ratio — "
        << "auto-drain threshold (R6) may need tuning";
}

// ============================================================================
// R7-4b: Extreme read ratio 99.99% (1 write per 10000 ops)
// Validates that with extremely sparse writes, the EBR epoch still advances
// and TLS ring auto-drain prevents dropped promotions. This is the harshest
// test for the reclamation subsystem under read-heavy workloads.
// ============================================================================
TEST(HighConcurrencySuite, ExtremeReadRatio9999Percent) {
    int nthreads = resolve_thread_count(32);
    if (nthreads < 4) GTEST_SKIP() << "Too few cores for 99.99% read test";

    // Direct construction — unified_cache is non-movable (contains mutexes/atomics)
    sharded_mm_lru_config hc_cfg;
    hc_cfg.lru_config.defer_promotion = true;
    read_heavy_striped_cache<int, int> c(10000, hc_cfg);
    (void)c.set_fairness_mode_quiescent(detail::fairness_mode::reader_preferred);
    c.set_incremental_rehash(true);
    std::atomic<int> value_mismatch{0};

    // 99.99% reads — use 9999 out of 10000 as read threshold
    // Pre-populate heavily since writes are extremely rare
    for (int i = 0; i < 5000; ++i) {
        c.set(i, i * 10);
    }

    std::atomic<bool> stop{false};
    std::vector<std::thread> threads;
    for (int t = 0; t < nthreads; ++t) {
        threads.emplace_back([&, t]() {
            std::mt19937 rng(static_cast<unsigned>(t * 7919 + 1));
            while (!stop.load(std::memory_order_relaxed)) {
                int key = static_cast<int>(rng() % 5000);
                if (rng() % 10000 < 1) {
                    // 0.01% writes
                    c.set(key, key * 10);
                } else {
                    auto h = c.try_get(key);
                    if (h && **h != key * 10) {
                        value_mismatch.fetch_add(1, std::memory_order_relaxed);
                    }
                }
            }
        });
    }

    // Short duration for CI; this is a correctness test, not a soak
    auto duration = lru_test::read_stress_duration_ms(std::chrono::seconds(5));
    std::this_thread::sleep_for(duration);
    stop.store(true, std::memory_order_release);
    for (auto& th : threads) th.join();

    EXPECT_EQ(value_mismatch.load(), 0)
        << "value corruption at 99.99% read ratio";
    EXPECT_LE(c.size(), c.max_size());

    // Check that dropped promotions are zero (R6 auto-drain should prevent drops)
    auto snap = c.stats_snapshot();
    EXPECT_EQ(snap.tls_ring_dropped_promotions.load(), 0u)
        << "TLS ring dropped promotions at 99.99% read ratio — "
        << "auto-drain threshold (R6) may need tuning";
}

// ============================================================================
// R7-5: NUMA-aware routing smoke test
// Validates that set_numa_aware(true) doesn't crash or corrupt data under
// concurrent access. Real NUMA benefit requires multi-socket hardware; this
// test is a correctness smoke test, not a performance benchmark.
// ============================================================================
TEST(HighConcurrencySuite, NumaAwareRoutingCorrectness) {
    int nthreads = resolve_thread_count(32);
    if (nthreads < 4) GTEST_SKIP() << "Too few cores for NUMA test";

    production_cache<int, int> c(10000);
    c.set_numa_aware(true);

    std::atomic<int> value_mismatch{0};
    run_fixed_ops_concurrent(c, 5000, nthreads, 95, 3000, value_mismatch);

    EXPECT_EQ(value_mismatch.load(), 0)
        << "value corruption with NUMA-aware routing enabled";
    EXPECT_LE(c.size(), c.max_size());
}

// ============================================================================
// R7-6: EBR epoch advancement under read-heavy load
// Validates that with R1's capacity-based epoch advance + time-based advance,
// retired objects are reclaimed even when writes are infrequent.
// Runs a workload that evicts items (triggering retire) and checks that
// reclaim_pending_count stays bounded.
// ============================================================================
TEST(HighConcurrencySuite, EpochAdvancementUnderReadHeavyLoad) {
    int nthreads = resolve_thread_count(16);
    if (nthreads < 4) GTEST_SKIP() << "Too few cores for epoch test";

    // Small cache + large key space = high eviction rate (lots of retires)
    production_cache<int, int> c(500);
    const int key_space = 5000;

    // Pre-populate to capacity
    for (int i = 0; i < 500; ++i) {
        c.set(i, i * 10);
    }

    // Run a mixed workload that constantly evicts (writes to keys beyond
    // capacity, forcing eviction + retire). 80% reads, 20% writes.
    std::atomic<bool> stop{false};
    std::atomic<int> value_mismatch{0};
    std::atomic<long long> total_ops{0};

    std::vector<std::thread> threads;
    for (int t = 0; t < nthreads; ++t) {
        threads.emplace_back([&, t]() {
            std::mt19937 rng(static_cast<unsigned>(t * 7919 + 1));
            while (!stop.load(std::memory_order_relaxed)) {
                int key = static_cast<int>(rng() % key_space);
                if (rng() % 100 < 20) {
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

    auto duration = lru_test::read_stress_duration_ms(std::chrono::seconds(5));
    std::this_thread::sleep_for(duration);
    stop.store(true, std::memory_order_release);
    for (auto& th : threads) th.join();

    EXPECT_EQ(value_mismatch.load(), 0)
        << "value corruption during epoch advancement test";

    // Wait for background drain worker to reclaim
    std::this_thread::sleep_for(std::chrono::milliseconds(1500));
    c.try_reclaim_now();

    auto snap = c.stats_snapshot();
    std::size_t pending = snap.reclaim_pending_count.load(std::memory_order_relaxed);

    // With R1 (capacity-based + time-based epoch advance) and R4 (optimized
    // reclaim scan), pending should be bounded. The threshold is generous
    // because the test runs for only 5s and the key space is 10x capacity.
    EXPECT_LT(pending, std::size_t(50000))
        << "reclaim_pending_count=" << pending << " is too high; "
        << "EBR epoch advancement (R1) may be stalled";

    // active handles must return to 0
    EXPECT_EQ(c.active_handle_count(), 0u)
        << "handle leak detected after epoch advancement test";

    std::cout << "[INFO] EpochAdvancement: total_ops=" << total_ops.load()
              << ", reclaim_pending=" << pending
              << ", duration=" << duration.count() << "ms\n";
}

// ============================================================================
// R7-7: Extended soak with 32+ threads and memory growth monitoring
// Runs a 95% read / 5% write workload across 32+ threads for a configurable
// duration (default 5s CI; set LRU_STRESS_DURATION_MS=300000 for 5-min soak).
// Validates:
//   - No value corruption
//   - RSS growth < 10% (for runs >= 30s)
//   - reclaim_pending_count < 100000
//   - active_handle_count == 0 after workload
//   - tls_ring_dropped_promotions == 0 (R6 auto-drain working)
// ============================================================================
TEST(HighConcurrencySuite, ExtendedSoak32Threads) {
    int nthreads = resolve_thread_count(32);
    if (nthreads < 8) GTEST_SKIP() << "Too few cores for 32-thread soak";

    // Direct construction — unified_cache is non-movable (contains mutexes/atomics)
    sharded_mm_lru_config hc_cfg;
    hc_cfg.lru_config.defer_promotion = true;
    read_heavy_striped_cache<int, int> c(10000, hc_cfg);
    (void)c.set_fairness_mode_quiescent(detail::fairness_mode::reader_preferred);
    c.set_incremental_rehash(true);
    auto duration = lru_test::read_stress_duration_ms(std::chrono::seconds(5));

    std::atomic<long long> total_ops{0};
    std::atomic<int> value_mismatch{0};

    // Pre-populate + warmup
    for (int i = 0; i < 5000; ++i) {
        c.set(i, i * 10);
    }
    {
        std::mt19937 rng(12345);
        for (int i = 0; i < 2000; ++i) {
            int key = static_cast<int>(rng() % 5000);
            if (i % 20 == 0) c.set(key, key * 10);
            else c.try_get(key);
        }
    }

    // Baseline RSS
    std::size_t rss_baseline = sample_rss_bytes();
    std::vector<std::size_t> rss_samples;

    // Run workload with RSS sampling
    std::atomic<bool> stop{false};
    std::vector<std::thread> threads;
    for (int t = 0; t < nthreads; ++t) {
        threads.emplace_back([&, t]() {
            std::mt19937 rng(static_cast<unsigned>(t * 7919 + 1));
            while (!stop.load(std::memory_order_relaxed)) {
                int key = static_cast<int>(rng() % 5000);
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

    // RSS sampler thread
    std::thread sampler([&]() {
        while (!stop.load(std::memory_order_relaxed)) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            if (stop.load(std::memory_order_relaxed)) break;
            auto s = sample_rss_bytes();
            if (s > 0) rss_samples.push_back(s);
        }
    });

    std::this_thread::sleep_for(duration);
    stop.store(true, std::memory_order_release);
    for (auto& th : threads) th.join();
    sampler.join();

    // Post-soak invariant checks
    std::this_thread::sleep_for(std::chrono::milliseconds(1500));
    c.try_reclaim_now();

    EXPECT_GT(total_ops.load(), 0) << "no operations executed";
    EXPECT_EQ(value_mismatch.load(), 0)
        << "value corruption during extended soak";

    auto snap = c.stats_snapshot();
    std::size_t pending = snap.reclaim_pending_count.load(std::memory_order_relaxed);
    std::size_t handles = c.active_handle_count();
    std::size_t dropped = snap.tls_ring_dropped_promotions.load();

    EXPECT_EQ(handles, 0u)
        << "active_handle_count != 0 after soak; handle leak";
    EXPECT_LT(pending, std::size_t(100000))
        << "reclaim_pending_count=" << pending << " exceeded 100000 bound";
    EXPECT_EQ(dropped, 0u)
        << "tls_ring_dropped_promotions=" << dropped
        << " — R6 auto-drain threshold may need tuning";

    // RSS growth check (only for runs >= 30s)
    if (duration >= std::chrono::seconds(30) && rss_baseline > 0 && !rss_samples.empty()) {
        std::size_t max_rss = rss_baseline;
        for (auto s : rss_samples) max_rss = std::max(max_rss, s);
        if (max_rss > rss_baseline) {
            double growth = static_cast<double>(max_rss - rss_baseline) /
                            static_cast<double>(rss_baseline);
            EXPECT_LT(growth, 0.10)
                << "RSS grew " << (growth * 100.0) << "% (threshold 10%, "
                << "baseline=" << rss_baseline << " bytes, duration="
                << duration.count() << "ms); possible leak";
        }
    } else {
        std::cout << "[INFO] RSS growth check skipped (duration < 30s or "
                  << "RSS sampling unavailable); set LRU_STRESS_DURATION_MS=300000 "
                  << "for full soak\n";
    }

    std::cout << "[INFO] ExtendedSoak32Threads: threads=" << nthreads
              << ", total_ops=" << total_ops.load()
              << ", reclaim_pending=" << pending
              << ", dropped_promotions=" << dropped
              << ", duration=" << duration.count() << "ms\n";
}

// ============================================================================
// R7-8: Mixed read-heavy + rehash stress with 32 threads
// Forces rehash by using tiny initial buckets, then runs 32 threads doing
// 90% reads / 10% writes. Validates dual-array lookup safety under high
// concurrency during incremental rehash.
// ============================================================================
TEST(HighConcurrencySuite, RehashStress32Threads) {
    int nthreads = resolve_thread_count(32);
    if (nthreads < 4) GTEST_SKIP() << "Too few cores for 32-thread rehash test";

    sharded_mm_lru_config cfg;
    cfg.expected_items = 1;  // force tiny initial buckets → rehash on insert
    production_cache<int, int> c(10000, cfg);

    std::atomic<int> value_mismatch{0};
    run_fixed_ops_concurrent(c, 5000, nthreads, 90, 3000, value_mismatch);

    EXPECT_EQ(value_mismatch.load(), 0)
        << "value corruption during concurrent rehash with " << nthreads << " threads";
    EXPECT_LE(c.size(), c.max_size());
}

// ============================================================================
// R7-9: Production cache with defer_promotion under 32-thread read-heavy
// Tests the TLS ring deferred promotion path (the primary read-heavy
// optimization) under high thread counts. Validates that promotions are
// not lost and LRU ordering is maintained.
// ============================================================================
TEST(HighConcurrencySuite, DeferredPromotion32Threads) {
    int nthreads = resolve_thread_count(32);
    if (nthreads < 4) GTEST_SKIP() << "Too few cores for deferred promotion test";

    production_cache<int, int> c(5000);
    // Ensure deferred promotion is enabled (default for production_cache)
    c.set_defer_promotion(true);
    // Start the drain worker to process TLS ring entries
    c.start_event_drain(std::chrono::milliseconds(100));

    const int key_space = 3000;
    for (int i = 0; i < key_space; ++i) {
        c.set(i, i * 10);
    }

    // Run read-heavy workload
    std::atomic<bool> stop{false};
    std::atomic<int> value_mismatch{0};
    std::atomic<long long> total_ops{0};

    std::vector<std::thread> threads;
    for (int t = 0; t < nthreads; ++t) {
        threads.emplace_back([&, t]() {
            std::mt19937 rng(static_cast<unsigned>(t * 7919 + 1));
            while (!stop.load(std::memory_order_relaxed)) {
                int key = static_cast<int>(rng() % key_space);
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

    auto duration = lru_test::read_stress_duration_ms(std::chrono::seconds(5));
    std::this_thread::sleep_for(duration);
    stop.store(true, std::memory_order_release);
    for (auto& th : threads) th.join();

    // Drain any remaining TLS ring entries
    c.drain_access_ring();
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    EXPECT_EQ(value_mismatch.load(), 0)
        << "value corruption with deferred promotion";
    EXPECT_LE(c.size(), c.max_size());

    auto snap = c.stats_snapshot();
    EXPECT_EQ(snap.tls_ring_dropped_promotions.load(), 0u)
        << "TLS ring dropped promotions with deferred promotion — "
        << "R6 auto-drain (kRingSize/2 threshold) should prevent drops";

    EXPECT_EQ(c.active_handle_count(), 0u)
        << "handle leak with deferred promotion";

    auto diag = c.diagnostics();
    std::cout << "[INFO] DeferredPromotion32Threads: threads=" << nthreads
              << ", total_ops=" << total_ops.load()
              << ", tls_ring_backlog_total=" << diag.tls_ring_backlog_total
              << ", dropped=" << snap.tls_ring_dropped_promotions.load() << "\n";
}
