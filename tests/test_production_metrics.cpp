// Production metrics & diagnostics tests for unified_cache.
//
// Split from test_production.cpp (2026-07-26): this file focuses on
// observability — latency histogram, Prometheus export, metrics cache,
// hash overload tracking, active-handle counter, incremental rehash,
// diagnostics dump, per-cache handle tracking, and streaming hot-key
// detection (Space-Saving). Runtime API tests live in
// test_production_api.cpp.
//
// Covers Tasks 4-6, 12, E plus P2-1 (per-cache handle tracking),
// P2-2 (log-linear histogram precision), P2-3 (streaming hot-keys),
// M-4-B (per-cache counter in release builds), T-M4 (TLS backlog
// aggregate), T13.1-13.4 (hash overload metrics).

#include <gtest/gtest.h>
#include "../lru.hpp"
#include <atomic>
#include <chrono>
#include <cstdint>
#include <limits>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace lru;
using namespace std::chrono_literals;

// ============================================================================
// Task 4: latency histogram
// ============================================================================
TEST(ProductionLatencyHistogram, RecordAndPercentile) {
    detail::latency_histogram h;
    for (int i = 1; i <= 100; ++i) {
        h.record(static_cast<uint64_t>(i * 10));
    }
    EXPECT_EQ(h.count(), 100u);
    // percentile() returns the bucket lower bound; verify monotonicity.
    auto p0 = h.percentile(0.0);
    auto p50 = h.percentile(0.5);
    auto p100 = h.percentile(1.0);
    EXPECT_LE(p0, p50);
    EXPECT_LE(p50, p100);
    EXPECT_GT(p50, 0u);
}

TEST(ProductionLatencyHistogram, EmptyHistogram) {
    detail::latency_histogram h;
    EXPECT_EQ(h.count(), 0u);
    EXPECT_EQ(h.percentile(0.5), 0u);
}

TEST(ProductionLatencyHistogram, GetRecordsLatency) {
    cache<int, std::string> c(10);
    c.set(1, "a");
    c.get(1);
    auto stats = c.stats_snapshot();
    EXPECT_GE(stats.get_latency.count(), 1u);
}

TEST(ProductionLatencyHistogram, SetRecordsLatency) {
    cache<int, std::string> c(10);
    c.set(1, "a");
    auto stats = c.stats_snapshot();
    EXPECT_GE(stats.set_latency.count(), 1u);
}

TEST(LatencyHistogram, SumAccuracy) {
    lru::detail::latency_histogram h;
    h.record(100);
    h.record(200);
    h.record(300);
    EXPECT_EQ(h.sum(), 600u);
    EXPECT_EQ(h.count(), 3u);
    h.reset();
    EXPECT_EQ(h.sum(), 0u);
}

// ============================================================================
// P2-2: log-linear histogram precision.
// 512 buckets, 16 sub-buckets per power-of-2 octave. Within-bucket width is
// 1/16 of the octave lower bound (for L ≥ 128 ns), so the worst-case relative
// error of percentile() (which returns the bucket lower bound) is ≤ 6.25%.
// In the µs-to-ms range the precision is well under 5%.
// ============================================================================
TEST(LatencyHistogramP2, BucketCountIs512) {
    EXPECT_EQ(lru::detail::latency_histogram::bucket_count, 512u);
    EXPECT_EQ(lru::detail::latency_histogram::sub_buckets_per_octave, 16u);
}

TEST(LatencyHistogramP2, LowLatencyLinearBuckets) {
    // Latencies 1..15 ns map to buckets 1..15 (linear 1 ns each).
    // Recording N samples of latency L should place all N in bucket L.
    lru::detail::latency_histogram h;
    h.record(5);
    h.record(5);
    h.record(10);
    EXPECT_EQ(h.bucket(5), 2u);
    EXPECT_EQ(h.bucket(10), 1u);
}

TEST(LatencyHistogramP2, OctaveSubBuckets) {
    // L=16..31 should map to buckets 16..31 (linear 1 ns each within octave).
    lru::detail::latency_histogram h;
    h.record(16);
    h.record(31);
    EXPECT_EQ(h.bucket(16), 1u);
    EXPECT_EQ(h.bucket(31), 1u);

    // L=32..63 should map to buckets 32..47 (linear 2 ns each).
    // bucket 32 lower bound = 32, covers [32, 34).
    // bucket 33 lower bound = 34, covers [34, 36).
    h.record(32);
    h.record(33);  // same bucket as 32 (both in [32, 34))
    h.record(34);  // next bucket (in [34, 36))
    EXPECT_EQ(h.bucket(32), 2u);  // 32 and 33 both go here
    EXPECT_EQ(h.bucket(33), 1u);  // 34 goes here
    EXPECT_EQ(h.bucket(34), 0u);  // no samples in [36, 38)
}

TEST(LatencyHistogramP2, PercentilePrecisionWithinFivePercent) {
    // P2-2 acceptance: P99 reported value vs. true value relative error < 5%.
    //
    // We record 1000 samples all with latency = 100µs (100000 ns). The true
    // P99 is 100000 ns. The histogram bucket containing 100000 ns has a
    // lower bound that must be within 5% of 100000 ns (i.e., ≥ 95000 ns).
    lru::detail::latency_histogram h;
    for (int i = 0; i < 1000; ++i) {
        h.record(100000);  // 100 µs
    }
    uint64_t p99 = h.percentile(0.99);
    EXPECT_GE(p99, 95000u) << "P99 reported value " << p99
                           << " is more than 5% below true 100000";

    // Also verify at 1 ms — typical SLO boundary.
    lru::detail::latency_histogram h2;
    for (int i = 0; i < 1000; ++i) {
        h2.record(1000000);  // 1 ms
    }
    uint64_t p99_ms = h2.percentile(0.99);
    EXPECT_GE(p99_ms, 950000u) << "P99 reported value " << p99_ms
                               << " is more than 5% below true 1000000";
}

TEST(LatencyHistogramP2, BucketLowerBoundMonotonic) {
    // Lower bounds must be strictly monotonically increasing.
    uint64_t prev = 0;
    for (std::size_t i = 0; i < lru::detail::latency_histogram::bucket_count; ++i) {
        uint64_t lb = lru::detail::latency_histogram::bucket_lower_bound(i);
        EXPECT_GE(lb, prev);
        prev = lb;
    }
    // The last bucket's lower bound should be a large but finite value
    // (not UINT64_MAX, which is reserved for upper_bound of overflow bucket).
    uint64_t last_lb = lru::detail::latency_histogram::bucket_lower_bound(
        lru::detail::latency_histogram::bucket_count - 1);
    EXPECT_GT(last_lb, 0u);
    EXPECT_LT(last_lb, std::numeric_limits<uint64_t>::max());
}

TEST(LatencyHistogramP2, BucketUpperBoundOverflowIsMaxValue) {
    // The overflow bucket (last index) has upper bound = UINT64_MAX.
    EXPECT_EQ(lru::detail::latency_histogram::bucket_upper_bound(
                  lru::detail::latency_histogram::bucket_count - 1),
              std::numeric_limits<uint64_t>::max());
}

// ============================================================================
// Task 5: Prometheus text export
// ============================================================================
TEST(ProductionPrometheus, BasicExport) {
    cache<int, std::string> c(100);
    c.set(1, "a");
    c.get(1);   // hit
    c.get(2);   // miss
    std::string text = c.prometheus_text();
    EXPECT_NE(text.find("lru_cache_hits_total"), std::string::npos);
    EXPECT_NE(text.find("lru_cache_misses_total"), std::string::npos);
    EXPECT_NE(text.find("lru_cache_size"), std::string::npos);
    EXPECT_NE(text.find("lru_cache_evictions_total"), std::string::npos);
}

TEST(ProductionPrometheus, ContainsLatencyHistogram) {
    cache<int, std::string> c(100);
    c.set(1, "a");
    c.get(1);
    std::string text = c.prometheus_text();
    EXPECT_NE(text.find("lru_cache_get_latency_ns"), std::string::npos);
    EXPECT_NE(text.find("lru_cache_set_latency_ns"), std::string::npos);
}

// T13.4: Prometheus export for hash load factor and overload metrics.
TEST(ProductionPrometheus, ContainsHashLoadFactorMetrics) {
    cache<int, std::string> c(100);
    c.set(1, "a");
    c.get(1);
    std::string text = c.prometheus_text();
    EXPECT_NE(text.find("lru_hash_load_factor"), std::string::npos);
    EXPECT_NE(text.find("lru_hash_overload_threshold"), std::string::npos);
    EXPECT_NE(text.find("lru_hash_overload_events_total"), std::string::npos);
}

// P2-E: Metrics cache — verify that enabling the cache returns the same
// snapshot and Prometheus text as the uncached path, and that the
// background worker refreshes it.
TEST(ProductionMetricsCache, DisabledByDefault) {
    cache<int, std::string> c(100);
    EXPECT_FALSE(c.metrics_cache_enabled());
    EXPECT_FALSE(c.metrics_cache_worker_running());
}

TEST(ProductionMetricsCache, EnableReturnsCachedSnapshot) {
    cache<int, std::string> c(100);
    c.set(1, "a");
    c.get(1);
    c.set_metrics_cache_enabled(true);
    EXPECT_TRUE(c.metrics_cache_enabled());
    // First call populates the cache.
    auto snap1 = c.stats_snapshot();
    EXPECT_GT(snap1.hits.value.load(), 0u);
    // Second call returns the cached copy (same hits count).
    auto snap2 = c.stats_snapshot();
    EXPECT_EQ(snap1.hits.value.load(), snap2.hits.value.load());
}

TEST(ProductionMetricsCache, RefreshUpdatesCache) {
    cache<int, std::string> c(100);
    c.set_metrics_cache_enabled(true);
    c.set(1, "a");
    c.refresh_metrics_cache();
    auto snap1 = c.stats_snapshot();
    auto hits1 = snap1.hits.value.load();
    EXPECT_EQ(hits1, 0u);
    // Activity that should bump hits — refresh and verify.
    c.get(1);
    c.refresh_metrics_cache();
    auto snap2 = c.stats_snapshot();
    EXPECT_GT(snap2.hits.value.load(), hits1);
}

TEST(ProductionMetricsCache, PrometheusTextUsesCache) {
    cache<int, std::string> c(100);
    c.set(1, "a");
    c.set_metrics_cache_enabled(true);
    // Prime the cache.
    c.refresh_metrics_cache();
    std::string text1 = c.prometheus_text();
    EXPECT_NE(text1.find("lru_cache_hits_total"), std::string::npos);
    // Subsequent call returns the cached string.
    std::string text2 = c.prometheus_text();
    EXPECT_EQ(text1, text2);
}

TEST(ProductionMetricsCache, DisableDropsCache) {
    cache<int, std::string> c(100);
    c.set_metrics_cache_enabled(true);
    c.set(1, "a");
    c.refresh_metrics_cache();
    EXPECT_TRUE(c.metrics_cache_enabled());
    c.set_metrics_cache_enabled(false);
    EXPECT_FALSE(c.metrics_cache_enabled());
    // After disabling, prometheus_text() rebuilds on every call (still works).
    std::string text = c.prometheus_text();
    EXPECT_NE(text.find("lru_cache_hits_total"), std::string::npos);
}

TEST(ProductionMetricsCache, BackgroundWorkerRefreshes) {
    cache<int, std::string> c(100);
    c.start_metrics_cache_worker(std::chrono::milliseconds(50));
    EXPECT_TRUE(c.metrics_cache_worker_running());
    c.set(1, "a");
    c.get(1);
    // Wait long enough for at least one tick.
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    auto snap = c.stats_snapshot();
    EXPECT_GT(snap.hits.value.load(), 0u);
    c.stop_metrics_cache_worker();
    EXPECT_FALSE(c.metrics_cache_worker_running());
}

// T13.1: set_hash_overload_threshold() configures the threshold.
TEST(HashOverloadThreshold, DefaultIsTwo) {
    cache<int, std::string> c(100);
    auto snap = c.stats_snapshot();
    EXPECT_FLOAT_EQ(snap.hash_overload_threshold.load(), 2.0f);
}

TEST(HashOverloadThreshold, SetThresholdPropagatesToHashStats) {
    cache<int, std::string> c(100);
    c.set_hash_overload_threshold(1.5f);
    // Force a refresh of hash stats via stats_snapshot().
    auto snap = c.stats_snapshot();
    EXPECT_FLOAT_EQ(snap.hash_overload_threshold.load(), 1.5f);
}

TEST(HashOverloadThreshold, ZeroThresholdFallsBackToDefault) {
    cache<int, std::string> c(100);
    c.set_hash_overload_threshold(0.0f);
    auto snap = c.stats_snapshot();
    EXPECT_FLOAT_EQ(snap.hash_overload_threshold.load(), 2.0f);
}

// T13.2: set_overload_callback() — the callback fires when load_factor
// exceeds the threshold. We use a very small threshold and a single-shard
// cache to make the test deterministic.
TEST(HashOverloadCallback, CallbackFiresOnOverload) {
    safe_cache<int, int> c(1000);
    std::atomic<int> fire_count{0};
    std::atomic<float> observed_lf{0.0f};
    c.set_hash_overload_threshold(0.01f);  // very low — any insert overloads
    c.set_overload_callback([&](float current_lf, float /*threshold*/) {
        fire_count.fetch_add(1, std::memory_order_relaxed);
        observed_lf.store(current_lf, std::memory_order_relaxed);
    });
    // Insert enough items to trigger rehash and fire the callback.
    for (int i = 0; i < 50; ++i) {
        c.set(i, i);
    }
    EXPECT_GE(fire_count.load(), 1);
    EXPECT_GT(observed_lf.load(), 0.0f);
}

TEST(HashOverloadCallback, CallbackExceptionIsSwallowed) {
    safe_cache<int, int> c(1000);
    c.set_hash_overload_threshold(0.01f);
    c.set_overload_callback([](float, float) {
        throw std::runtime_error("user callback error");
    });
    // Insertions must not propagate the exception.
    for (int i = 0; i < 50; ++i) {
        c.set(i, i);
    }
    SUCCEED();
}

// T13.3: Overload events counter increments when threshold is exceeded.
TEST(HashOverloadEvents, CounterIncrementsOnOverload) {
    safe_cache<int, int> c(1000);
    c.set_hash_overload_threshold(0.01f);
    auto snap_before = c.stats_snapshot();
    ASSERT_EQ(snap_before.hash_overload_events.load(), 0u);
    for (int i = 0; i < 50; ++i) {
        c.set(i, i);
    }
    auto snap_after = c.stats_snapshot();
    EXPECT_GT(snap_after.hash_overload_events.load(), 0u);
}

// ============================================================================
// Task 6: active_handle_count
//
// P2-1/P2-2: per-T global counter (sharded across 64 cache lines) is
// only maintained under -DLRU_DEBUG=1 (or when global tracking is
// explicitly enabled). In release builds, read_handle<T>::active_count()
// is 0 and observability is sourced from per-cache active_handle_count()
// via the per_cache_stats_ pointer (per_cache_handle_tracking_ defaults
// to true). The tests below use c.active_handle_count() which works in
// both debug and release builds.
// ============================================================================
TEST(ProductionActiveHandle, BasicCount) {
    cache<int, std::string> c(10);
    c.set(1, "a");
    {
        auto h = c.get(1);
        ASSERT_TRUE(h.has_value());
        // Per-cache tracking is default-on (P2-1), so the cache-level
        // counter reflects the live handle. The per-T global counter is
        // only maintained under LRU_DEBUG.
        EXPECT_GE(c.active_handle_count(), 1u);
    }
    EXPECT_EQ(c.active_handle_count(), 0u);
}

TEST(ProductionActiveHandle, CacheMethodMatchesStatic) {
    // P2-1: in release builds, read_handle<T>::active_count() is always 0
    // (the per-T global counter is only maintained under LRU_DEBUG). The
    // per-cache c.active_handle_count() is the authoritative source in
    // production. Under LRU_DEBUG the per-T global counter is maintained
    // and should be >= the per-cache count (it counts handles across all
    // caches of the same value type).
    cache<int, std::string> c(10);
    c.set(1, "a");
    {
        auto h = c.get(1);
        ASSERT_TRUE(h.has_value());
        EXPECT_GE(c.active_handle_count(), 1u);
#ifdef LRU_DEBUG
        // Under LRU_DEBUG the per-T global counter is maintained and must
        // be >= the per-cache count (which is a subset in a multi-cache
        // process; in this single-cache test they are equal).
        EXPECT_GE(read_handle<std::string>::active_count(),
                  c.active_handle_count());
#endif
    }
    EXPECT_EQ(c.active_handle_count(), 0u);
#ifdef LRU_DEBUG
    // The per-T global counter also drops to 0 once all handles release.
    EXPECT_EQ(read_handle<std::string>::active_count(), 0u);
#endif
}

// M-4-B: Per-cache sharded handle counter is maintained and exposed in
// RELEASE builds (not just -DLRU_DEBUG=1). This guards against regressions
// where the per-cache counter accidentally becomes DEBUG-only, which would
// make production observability (stats_snapshot / prometheus_text) lose
// active-handle visibility in deployed binaries.
TEST(ProductionActiveHandle, PerCacheCounterMaintainedInReleaseBuilds) {
    cache<int, std::string> c(64);
    ASSERT_TRUE(c.is_per_cache_handle_tracking_enabled())
        << "per_cache_handle_tracking_ must default to true in release builds";

    c.set(7, "v7");
    c.set(8, "v8");

    std::string prom_before = c.prometheus_text();
    EXPECT_NE(prom_before.find("lru_cache_active_handles"), std::string::npos)
        << "prometheus_text must export lru_cache_active_handles";

    // Hold two live handles across separate scopes.
    std::optional<read_handle<std::string>> h1 = c.get(7);
    std::optional<read_handle<std::string>> h2 = c.get(8);
    ASSERT_TRUE(h1.has_value() && h2.has_value());

    const std::size_t live = c.active_handle_count();
    EXPECT_GE(live, 2u)
        << "per-cache sharded_handle_counter must reflect live handles in release builds";

    auto snap = c.stats_snapshot();
    EXPECT_GE(snap.active_handle_count.load(std::memory_order_relaxed), 2u)
        << "stats_snapshot must surface per-cache active_handle_count in release builds";

    // Release one handle — count must drop by exactly one (counter is
    // maintained on both inc and dec paths).
    h1.reset();
    const std::size_t after_one = c.active_handle_count();
    EXPECT_EQ(after_one + 1, live)
        << "dec path must update per-cache counter; expected " << (live - 1)
        << " got " << after_one;

    h2.reset();
    EXPECT_EQ(c.active_handle_count(), 0u)
        << "per-cache counter must return to 0 once all handles are released";

    // Confirm prometheus_text reflects the post-release state (0). Skip the
    // HELP/TYPE comment lines and locate the actual metric line, which is
    // `\nlru_cache_active_handles <value>\n`.
    std::string prom_after = c.prometheus_text();
    const std::string metric_prefix = "\nlru_cache_active_handles ";
    auto pos = prom_after.find(metric_prefix);
    ASSERT_NE(pos, std::string::npos)
        << "lru_cache_active_handles metric line missing in:\n" << prom_after;
    auto value_start = pos + metric_prefix.size();
    auto line_end = prom_after.find('\n', value_start);
    std::string value_str = prom_after.substr(value_start, line_end - value_start);
    EXPECT_EQ(value_str, "0")
        << "expected active_handles=0 after release, got: '" << value_str << "'";
}

// ============================================================================
// Task 12: Incremental rehash
// ============================================================================
TEST(ProductionIncrementalRehash, DefaultDisabled) {
    cache<int, std::string> c(1000);
    EXPECT_FALSE(c.incremental_rehash_enabled());
}

// T-P3: safe_cache / striped_cache now default to incremental rehash enabled
// via safe_lru_trait / safe_sharded_lru_trait. This avoids global write stalls
// when the hash table grows past its load factor.
TEST(ProductionIncrementalRehash, SafeCacheDefaultEnabled) {
    safe_cache<int, std::string> c(1000);
    EXPECT_TRUE(c.incremental_rehash_enabled());
}

TEST(ProductionIncrementalRehash, StripedCacheDefaultEnabled) {
    striped_cache<int, std::string> c(1000);
    EXPECT_TRUE(c.incremental_rehash_enabled());
}

TEST(ProductionIncrementalRehash, EnableDisable) {
    cache<int, std::string> c(1000);
    c.set_incremental_rehash(true);
    EXPECT_TRUE(c.incremental_rehash_enabled());
    c.set_incremental_rehash(false);
    EXPECT_FALSE(c.incremental_rehash_enabled());
}

TEST(ProductionIncrementalRehash, CorrectnessDuringRehash) {
    // Insert enough items to trigger rehash with incremental rehash enabled.
    // Verify all items remain accessible after rehash completes.
    cache<int, std::string> c(10000);
    c.set_incremental_rehash(true);

    // Insert many items to trigger rehash (default bucket count is small).
    // Use keys that spread across buckets.
    for (int i = 0; i < 5000; ++i) {
        c.set(i, "value_" + std::to_string(i));
    }

    // All items must be readable.
    for (int i = 0; i < 5000; ++i) {
        auto h = c.get(i);
        ASSERT_TRUE(h.has_value()) << "Missing key " << i;
        EXPECT_EQ(*h, "value_" + std::to_string(i));
    }
}

TEST(ProductionIncrementalRehash, MixedOpsDuringRehash) {
    // Interleave inserts, gets, and removes while rehash may be in progress.
    cache<int, std::string> c(5000);
    c.set_incremental_rehash(true);

    // Phase 1: insert
    for (int i = 0; i < 2000; ++i) {
        c.set(i, "v" + std::to_string(i));
    }
    // Phase 2: read + insert more (triggers rehash)
    for (int i = 0; i < 2000; ++i) {
        auto h = c.get(i);
        ASSERT_TRUE(h.has_value());
    }
    for (int i = 2000; i < 4000; ++i) {
        c.set(i, "v" + std::to_string(i));
    }
    // Phase 3: remove some, then verify remaining
    for (int i = 0; i < 1000; ++i) {
        c.remove(i);
    }
    for (int i = 1000; i < 4000; ++i) {
        auto h = c.get(i);
        ASSERT_TRUE(h.has_value()) << "Missing key " << i;
    }
}

TEST(ProductionIncrementalRehash, ConcurrentAccessDuringRehash) {
    // Multiple threads reading and writing while rehash is in progress.
    striped_cache<int, std::string> c(20000);
    c.set_incremental_rehash(true);

    // Pre-fill
    for (int i = 0; i < 2000; ++i) {
        c.set(i, "init_" + std::to_string(i));
    }

    constexpr int kThreads = 4;
    constexpr int kOpsPerThread = 2000;

    auto worker = [&](int tid) {
        for (int i = 0; i < kOpsPerThread; ++i) {
            int key = tid * kOpsPerThread + i;
            c.set(key, "t" + std::to_string(tid) + "_" + std::to_string(i));
            auto h = c.get(key);
            if (h.has_value()) {
                // Read is fine
            }
        }
    };

    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back(worker, t);
    }
    for (auto& th : threads) {
        th.join();
    }

    // Verify all keys written by all threads are present.
    for (int t = 0; t < kThreads; ++t) {
        for (int i = 0; i < kOpsPerThread; ++i) {
            int key = t * kOpsPerThread + i;
            auto h = c.get(key);
            EXPECT_TRUE(h.has_value()) << "Missing key " << key;
        }
    }
}

// ============================================================================
// Task E: diagnostics() API
// ============================================================================
TEST(ProductionDiagnostics, BasicFieldsShardedCache) {
    striped_cache<int, std::string> c(1000, 8);
    c.set_incremental_rehash(true);
    for (int i = 0; i < 100; ++i) {
        c.set(i, "v" + std::to_string(i));
    }
    auto info = c.diagnostics();
    EXPECT_EQ(info.num_shards, 8u);
    EXPECT_EQ(info.per_stripe_wait_count.size(), 8u);
    EXPECT_EQ(info.per_stripe_try_fail_count.size(), 8u);
    EXPECT_EQ(info.per_shard_size.size(), 8u);
    EXPECT_EQ(info.per_shard_bucket_count.size(), 8u);
    // Sum of per-shard sizes should equal total inserted (100).
    std::size_t total = 0;
    for (auto s : info.per_shard_size) total += s;
    EXPECT_EQ(total, 100u);
    // incremental_rehash should be enabled.
    EXPECT_TRUE(info.incremental_rehash_enabled);
    // defer_promotion defaults to true for sharded mm_lru.
    EXPECT_TRUE(info.defer_promotion_enabled);
}

TEST(ProductionDiagnostics, NonShardedCache) {
    cache<int, std::string> c(100);
    c.set(1, "a");
    c.set(2, "b");
    auto info = c.diagnostics();
    EXPECT_EQ(info.num_shards, 1u);
    EXPECT_EQ(info.per_stripe_wait_count.size(), 1u);
    EXPECT_EQ(info.per_shard_size.size(), 1u);
    EXPECT_EQ(info.per_shard_size[0], 2u);
}

TEST(ProductionDiagnostics, TextOutputContainsFields) {
    striped_cache<int, int> c(100, 4);
    for (int i = 0; i < 10; ++i) c.set(i, i * 10);
    auto text = c.diagnostics_text();
    EXPECT_NE(text.find("num_shards"), std::string::npos);
    EXPECT_NE(text.find("active_handle_count"), std::string::npos);
    EXPECT_NE(text.find("tls_ring_backlog"), std::string::npos);
    EXPECT_NE(text.find("defer_promotion_enabled"), std::string::npos);
    EXPECT_NE(text.find("incremental_rehash_enabled"), std::string::npos);
    EXPECT_NE(text.find("per-stripe contention"), std::string::npos);
    EXPECT_NE(text.find("shard[0]"), std::string::npos);
    EXPECT_NE(text.find("=== end diagnostics ==="), std::string::npos);
}

// T-M4: Verify the new cross-thread TLS backlog aggregate and per-shard
// retire_pending fields are populated and exposed through diagnostics(),
// diagnostics_text(), and prometheus_text(). These are the metrics
// operators actually need to detect drain-worker starvation (a non-zero
// tls_ring_backlog_total with flat drain rate is the smoking gun).
TEST(ProductionDiagnostics, TlsRingBacklogTotalAndPerShardRetirePendingExposed) {
    striped_cache<int, int> c(100, 4);
    for (int i = 0; i < 10; ++i) c.set(i, i * 10);

    // diagnostics_info struct fields.
    auto info = c.diagnostics();
    // tls_ring_backlog_total is the cross-thread aggregate; it must be
    // >= the calling-thread view (tls_ring_backlog) since the latter is
    // a subset of the former.
    EXPECT_GE(info.tls_ring_backlog_total, info.tls_ring_backlog);
    // per_shard_retire_pending must be sized to num_shards and every
    // entry must mirror the global pending count (hazptr/EBR domains
    // are global by default).
    EXPECT_EQ(info.per_shard_retire_pending.size(), info.num_shards);
    for (std::size_t i = 0; i < info.per_shard_retire_pending.size(); ++i) {
        EXPECT_EQ(info.per_shard_retire_pending[i], info.reclaim_pending_count);
    }

    // diagnostics_text() must surface both new metrics.
    auto text = c.diagnostics_text();
    EXPECT_NE(text.find("tls_ring_backlog_total"), std::string::npos);
    EXPECT_NE(text.find("retire_pending="), std::string::npos);

    // prometheus_text() must export the cross-thread backlog gauge.
    auto prom = c.prometheus_text();
    EXPECT_NE(prom.find("lru_cache_tls_ring_backlog_total"), std::string::npos);
}

TEST(ProductionDiagnostics, ActiveHandleCountReflected) {
    striped_cache<int, std::string> c(100, 4);
    c.set(1, "v1");
    auto info_before = c.diagnostics();
    {
        auto h = c.get(1);
        ASSERT_TRUE(h.has_value());
        auto info_during = c.diagnostics();
        // Active handle count while the handle is alive should be >= 1.
        // Note: per-cache tracking is off by default, so this checks the
        // global per-T counter (which may include handles from other tests
        // on the same thread, so we just check it didn't decrease).
        EXPECT_GE(info_during.active_handle_count, info_before.active_handle_count);
    }
}

// ============================================================================
// Task C: per-cache handle tracking
//
// P2-1: per_cache_handle_tracking_ now defaults to true. The per-T global
// counter (sharded across 64 cache lines via P2-2) is only maintained
// under LRU_DEBUG; in release builds the per-cache counter is the sole
// source of active_handle_count.
// ============================================================================
TEST(ProductionPerCacheHandleTracking, DefaultOn) {
    cache<int, std::string> c(100);
    // P2-1: default is now true (production behavior).
    EXPECT_TRUE(c.is_per_cache_handle_tracking_enabled());
}

TEST(ProductionPerCacheHandleTracking, ToggleAndCount) {
    cache<int, std::string> c(100);
    c.set_per_cache_handle_tracking(true);
    EXPECT_TRUE(c.is_per_cache_handle_tracking_enabled());
    c.set(1, "a");
    auto before = c.active_handle_count();
    {
        auto h = c.get(1);
        ASSERT_TRUE(h.has_value());
        auto during = c.active_handle_count();
        EXPECT_GT(during, before);
    }
    auto after = c.active_handle_count();
    EXPECT_EQ(after, before);
}

TEST(ProductionPerCacheHandleTracking, SnapshotReflectsPerCacheCount) {
    striped_cache<int, std::string> c(100, 4);
    c.set_per_cache_handle_tracking(true);
    c.set(1, "v1");
    {
        auto h = c.get(1);
        ASSERT_TRUE(h.has_value());
        auto snap = c.stats_snapshot();
        EXPECT_GE(snap.active_handle_count.load(), 1u);
    }
}

// ============================================================================
// P2-3: Streaming hot-key detection (Space-Saving algorithm)
//
// Acceptance criteria from spec.md:
//   1. top-100 hot Key 召回率 > 95% (recall > 95%)
//   2. 查询耗时 < 1ms (query latency < 1ms)
// ============================================================================
TEST(ProductionStreamingHotKeys, TopKRecallHighSkew) {
    // Single-threaded deterministic workload with a clear hot-key
    // distribution: 100 hot keys (0..99) each accessed 1000 times, plus
    // 10,000 cold keys (100..10099) each accessed once.
    //
    // With K=200 capacity Space-Saving summary, the top-100 should be
    // recalled with ~100% accuracy since the hot keys are 1000x more
    // frequent than the cold keys.
    //
    // NOTE: The TLS event ring is 64 entries; we drain periodically to
    // avoid overflow losing events before they reach the streaming
    // summary. This mirrors realistic usage (background drain worker or
    // periodic manual drain).
    event_tracker<int>::config cfg;
    cfg.hot_keys_capacity = 200;
    cfg.record_hits = true;
    cfg.hit_sampling_rate = 1.0;
    event_tracker<int> tracker(cfg);

    // Record 100 hot keys × 1000 hits each = 100,000 hot hits.
    // Drain every 32 records to stay well below the 64-entry TLS ring.
    int since_drain = 0;
    for (int i = 0; i < 100; ++i) {
        for (int j = 0; j < 1000; ++j) {
            tracker.record_hit(i);
            if (++since_drain >= 32) {
                tracker.drain_tls();
                since_drain = 0;
            }
        }
    }
    // Record 10,000 cold keys × 1 hit each = 10,000 cold hits.
    for (int i = 100; i < 10100; ++i) {
        tracker.record_hit(i);
        if (++since_drain >= 32) {
            tracker.drain_tls();
            since_drain = 0;
        }
    }
    tracker.drain_tls();  // Final drain.

    auto top = tracker.top_keys(100);
    ASSERT_EQ(top.size(), 100u);

    // Build a set of the true top-100 hot keys (0..99).
    std::unordered_set<uint64_t> true_top;
    for (int i = 0; i < 100; ++i) {
        true_top.insert(std::hash<int>{}(i));
    }

    // Count how many of the reported top-100 are actually in the true top-100.
    std::size_t hits = 0;
    for (const auto& [hash, count] : top) {
        if (true_top.count(hash) > 0) ++hits;
    }

    // Recall must be > 95%.
    double recall = static_cast<double>(hits) / 100.0;
    EXPECT_GT(recall, 0.95)
        << "Recall=" << recall << " hits=" << hits << "/100";
}

TEST(ProductionStreamingHotKeys, QueryLatencyUnder1ms) {
    // Pre-populate the streaming summary with a realistic workload:
    // 50,000 keys with skewed access pattern.
    event_tracker<int>::config cfg;
    cfg.hot_keys_capacity = 1024;
    cfg.record_hits = true;
    cfg.hit_sampling_rate = 1.0;
    event_tracker<int> tracker(cfg);

    // 500 hot keys × 200 hits each, draining periodically.
    int since_drain = 0;
    for (int i = 0; i < 500; ++i) {
        for (int j = 0; j < 200; ++j) {
            tracker.record_hit(i);
            if (++since_drain >= 32) {
                tracker.drain_tls();
                since_drain = 0;
            }
        }
    }
    // 49,500 cold keys × 1 hit each.
    for (int i = 500; i < 50000; ++i) {
        tracker.record_hit(i);
        if (++since_drain >= 32) {
            tracker.drain_tls();
            since_drain = 0;
        }
    }
    tracker.drain_tls();

    // Measure top-100 query latency (averaged over several runs to
    // smooth out scheduling jitter).
    using namespace std::chrono;
    constexpr int kIterations = 50;
    nanoseconds total_dur{0};
    for (int i = 0; i < kIterations; ++i) {
        auto start = steady_clock::now();
        auto top = tracker.top_keys(100);
        auto end = steady_clock::now();
        total_dur += end - start;
        // Prevent the compiler from optimizing away the call.
        EXPECT_FALSE(top.empty());
    }
    double avg_us =
        static_cast<double>(total_dur.count()) / kIterations / 1000.0;
    // Spec requires < 1ms = 1000us.
    EXPECT_LT(avg_us, 1000.0)
        << "Avg top_keys(100) latency=" << avg_us << "us";
}

TEST(ProductionStreamingHotKeys, KeyToStringCallback) {
    // The `set_key_to_string` callback should enable human-readable key
    // names in `top_keys_with_names()`.
    event_tracker<int>::config cfg;
    cfg.hot_keys_capacity = 64;
    // R6/T-P1-6 changed the default hit_sampling_rate from 1.0 to 0.01.
    // This test verifies key-to-string mapping with known hit counts,
    // so disable sampling to ensure every hit is recorded.
    cfg.hit_sampling_rate = 1.0;
    event_tracker<int> tracker(cfg);

    // Register a callback mapping hash → "key_<N>" style name.
    // For test determinism, we maintain our own hash → name map.
    std::unordered_map<uint64_t, std::string> hash_to_name;
    for (int i = 0; i < 10; ++i) {
        auto h = std::hash<int>{}(i);
        hash_to_name[h] = "key_" + std::to_string(i);
    }
    tracker.set_key_to_string([&hash_to_name](uint64_t h) {
        auto it = hash_to_name.find(h);
        return it != hash_to_name.end() ? it->second : "unknown";
    });

    // Record some hits.
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 100; ++j) {
            tracker.record_hit(i);
        }
    }

    // Drain the TLS ring buffer so events reach the Space-Saving summary.
    // Without this call, top_keys_with_names() returns empty because the
    // hits are still in the per-thread ring and haven't been processed.
    tracker.drain_tls();

    auto named = tracker.top_keys_with_names(5);
    ASSERT_LE(named.size(), 5u);
    ASSERT_FALSE(named.empty());
    // Every name should be "key_<N>" (no "unknown" entries for the keys we recorded).
    for (const auto& [name, count] : named) {
        EXPECT_EQ(name.substr(0, 4), "key_")
            << "Unexpected name: " << name;
    }
    // Counts should be in descending order.
    for (std::size_t i = 1; i < named.size(); ++i) {
        EXPECT_GE(named[i - 1].second, named[i].second);
    }
}

TEST(ProductionStreamingHotKeys, SummaryStatsConsistency) {
    // `hot_keys_summary()` should return consistent stats after a known
    // workload, and `reset_hot_keys()` should clear the summary.
    event_tracker<int>::config cfg;
    cfg.hot_keys_capacity = 100;
    // R6/T-P1-6 changed the default hit_sampling_rate from 1.0 to 0.01.
    // This test expects exact hit counts (total_hits == 50), so disable
    // sampling to ensure every hit is recorded.
    cfg.hit_sampling_rate = 1.0;
    event_tracker<int> tracker(cfg);

    // 50 unique keys × 1 hit each. Drain periodically so all events
    // reach the streaming summary (TLS ring holds only 64 entries).
    int since_drain = 0;
    for (int i = 0; i < 50; ++i) {
        tracker.record_hit(i);
        if (++since_drain >= 32) {
            tracker.drain_tls();
            since_drain = 0;
        }
    }
    tracker.drain_tls();

    auto s = tracker.hot_keys_summary();
    EXPECT_EQ(s.capacity, 100u);
    EXPECT_LE(s.tracked, 100u);
    EXPECT_EQ(s.total_hits, 50u);

    tracker.reset_hot_keys();
    auto s2 = tracker.hot_keys_summary();
    EXPECT_EQ(s2.tracked, 0u);
    EXPECT_EQ(s2.total_hits, 0u);
    EXPECT_EQ(s2.error_bound, 0u);
}

// main() is provided by GTest::gtest_main (linked via CMakeLists.txt).
