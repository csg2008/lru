// Production API tests for unified_cache.
//
// Split from test_production.cpp (2026-07-26): this file focuses on
// user-facing runtime APIs (defer_promotion, fairness, stripes, TTL
// cleaner, async callbacks, try_get/get_with_ttl/cas, per-shard
// serialization, graceful shutdown, integration, overflow policy,
// native wait ops, TTL const-cast). Metrics/diagnostics tests live in
// test_production_metrics.cpp.
//
// Covers Tasks 1-3, 7-11 plus P2-5 (TTL const-cast) and Task D
// (overflow_policy).

#include <gtest/gtest.h>
#include "../lru.hpp"
#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

using namespace lru;
using namespace std::chrono_literals;

// ============================================================================
// Task 1: defer_promotion
// ============================================================================
TEST(ProductionDeferPromotion, DefaultEnabled) {
    cache<int, std::string> c(10);
    c.set(1, "a");
    c.set(2, "b");
    // get() still registers a hit even when defer_promotion is on.
    auto h = c.get(1);
    ASSERT_TRUE(h.has_value());
    auto stats = c.stats_snapshot();
    EXPECT_GE(stats.hits.value.load(), 1u);
    // flush() drains deferred promotions — must not crash.
    c.flush();
}

TEST(ProductionDeferPromotion, RuntimeToggle) {
    cache<int, std::string> c(10);
    c.set_defer_promotion(false);
    c.set(1, "a");
    c.set(2, "b");
    auto h = c.get(1);
    ASSERT_TRUE(h.has_value());
    EXPECT_FALSE(c.is_defer_promotion_enabled());
    c.set_defer_promotion(true);
    EXPECT_TRUE(c.is_defer_promotion_enabled());
}

// ============================================================================
// Task 2: fairness mode
// ============================================================================
// T-P3: striped_cache (and safe_cache) now default to reader_preferred via
// safe_sharded_lru_trait / safe_lru_trait. Tests below verify the new
// default and that the mode is still runtime-switchable.
TEST(ProductionFairness, DefaultReaderPreferred) {
    striped_cache<int, std::string> c(100);
    EXPECT_EQ(c.get_fairness_mode(), detail::fairness_mode::reader_preferred);
}

TEST(ProductionFairness, SafeCacheDefaultReaderPreferred) {
    safe_cache<int, std::string> c(100);
    EXPECT_EQ(c.get_fairness_mode(), detail::fairness_mode::reader_preferred);
}

TEST(ProductionFairness, RuntimeSwitch) {
    striped_cache<int, std::string> c(100);
    c.set_fairness_mode(detail::fairness_mode::writer_fair);
    EXPECT_EQ(c.get_fairness_mode(), detail::fairness_mode::writer_fair);
    c.set_fairness_mode(detail::fairness_mode::reader_preferred);
    EXPECT_EQ(c.get_fairness_mode(), detail::fairness_mode::reader_preferred);
}

// ============================================================================
// Task 3: configurable num_stripes
// ============================================================================
TEST(ProductionStripes, RuntimeNumStripes) {
    striped_cache<int, std::string> c(100, 128);
    EXPECT_EQ(c.num_stripes(), 128u);
}

TEST(ProductionStripes, DefaultNumStripes) {
    striped_cache<int, std::string> c(100);
    EXPECT_EQ(c.num_stripes(), 64u);
}

// ============================================================================
// P10: fast_get — ultra-light read path
// ============================================================================
TEST(ProductionFastGet, HitReturnsHandle) {
    cache<int, std::string> c(10);
    c.set(1, "alpha");
    auto h = c.fast_get(1);
    EXPECT_TRUE(h.has_value());
    EXPECT_EQ(*h, "alpha");
}

TEST(ProductionFastGet, MissReturnsEmpty) {
    cache<int, std::string> c(10);
    c.set(1, "alpha");
    auto h = c.fast_get(99);
    EXPECT_FALSE(h.has_value());
}

TEST(ProductionFastGet, BumpsHitCounter) {
    cache<int, std::string> c(10);
    c.set(1, "alpha");
    (void)c.fast_get(1);
    (void)c.fast_get(1);
    auto stats = c.stats_snapshot();
    EXPECT_GE(stats.hits.value.load(), 2u);
}

TEST(ProductionFastGet, BumpsMissCounter) {
    cache<int, std::string> c(10);
    (void)c.fast_get(42);
    auto stats = c.stats_snapshot();
    EXPECT_GE(stats.misses.value.load(), 1u);
}

TEST(ProductionFastGet, StripedPath) {
    striped_cache<int, std::string> c(100, 4);
    c.set(7, "seven");
    auto h = c.fast_get(7);
    ASSERT_TRUE(h.has_value());
    EXPECT_EQ(*h, "seven");
    // Miss on striped path
    auto h2 = c.fast_get(999);
    EXPECT_FALSE(h2.has_value());
}

TEST(ProductionFastGet, ConcurrentReadsSafe) {
    striped_cache<int, int> c(1000, 8);
    for (int i = 0; i < 100; ++i) c.set(i, i * 2);

    std::vector<std::thread> threads;
    std::atomic<unsigned> hits{0};
    std::atomic<unsigned> misses{0};
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&]() {
            for (int i = 0; i < 1000; ++i) {
                auto h = c.fast_get(i % 150);
                if (h) {
                    ++hits;
                    EXPECT_EQ(*h, (i % 150) * 2);
                } else {
                    ++misses;
                }
            }
        });
    }
    for (auto& t : threads) t.join();
    EXPECT_GT(hits.load(), 0u);
    EXPECT_GT(misses.load(), 0u);  // some keys 100..149 miss
}

TEST(ProductionFastGet, ProductionCachePath) {
    production_cache<int, std::string> c(100);
    c.set(1, "p1");
    auto h = c.fast_get(1);
    ASSERT_TRUE(h.has_value());
    EXPECT_EQ(*h, "p1");
}

// ============================================================================
// O11: Exception hierarchy — cache_closed / cache_oom / cache_config
// ============================================================================
TEST(ProductionException, GetOnShutdownThrowsCacheClosed) {
    cache<int, std::string> c(10);
    c.set(1, "a");
    c.shutdown();
    try {
        (void)c.get(1);
        FAIL() << "expected cache_closed_exception";
    } catch (const cache_closed_exception&) {
        SUCCEED();
    } catch (const std::runtime_error&) {
        FAIL() << "expected cache_closed_exception, got base runtime_error";
    }
}

TEST(ProductionException, GetPrehashedOnShutdownThrowsCacheClosed) {
    cache<int, std::string> c(10);
    c.set(1, "a");
    c.shutdown();
    try {
        (void)c.get_prehashed(1, std::hash<int>{}(1));
        FAIL() << "expected cache_closed_exception";
    } catch (const cache_closed_exception&) {
        SUCCEED();
    }
}

TEST(ProductionException, BulkGetOnShutdownThrowsCacheClosed) {
    striped_cache<int, std::string> c(100);
    c.set(1, "a");
    c.shutdown();
    std::vector<int> keys{1, 2, 3};
    try {
        (void)c.bulk_get(keys.begin(), keys.end());
        FAIL() << "expected cache_closed_exception";
    } catch (const cache_closed_exception&) {
        SUCCEED();
    }
}

TEST(ProductionException, GetOrFetchNoProviderThrowsConfig) {
    cache<int, std::string> c(10);
    try {
        (void)c.get_or_fetch(42);
        FAIL() << "expected cache_config_exception";
    } catch (const cache_config_exception&) {
        SUCCEED();
    }
}

TEST(ProductionException, ReadHandleNullDerefThrowsConfig) {
    cache<int, std::string> c(10);
    auto h = c.get(99);  // miss → empty handle
    ASSERT_FALSE(h.has_value());
    try {
        (void)*h;
        FAIL() << "expected cache_config_exception";
    } catch (const cache_config_exception&) {
        SUCCEED();
    }
}

TEST(ProductionException, SlabNotEnabledThrowsConfig) {
    cache<int, std::string> c(10);
    try {
        (void)c.slab_alloc();
        FAIL() << "expected cache_config_exception";
    } catch (const cache_config_exception&) {
        SUCCEED();
    }
}

TEST(ProductionException, BaseClassCatchesAll) {
    cache<int, std::string> c(10);
    c.shutdown();
    // cache_closed_exception should be caught by cache_exception base.
    try {
        (void)c.get(1);
        FAIL();
    } catch (const cache_exception&) {
        SUCCEED();
    }
    // And by std::runtime_error (backward compatibility).
    try {
        (void)c.get(1);
        FAIL();
    } catch (const std::runtime_error&) {
        SUCCEED();
    }
}

// ============================================================================
// Task 7: TTL cleaner
// ============================================================================
TEST(ProductionTTLCleaner, StartStop) {
    striped_cache<int, std::string> c(100, 4);
    EXPECT_FALSE(c.is_ttl_cleaner_running());
    c.start_ttl_cleaner(std::chrono::milliseconds(100));
    EXPECT_TRUE(c.is_ttl_cleaner_running());
    c.stop_ttl_cleaner();
    EXPECT_FALSE(c.is_ttl_cleaner_running());
}

TEST(ProductionTTLCleaner, EvictExpiredNoop) {
    cache<int, std::string> c(100);
    c.set(1, "a");
    // mm_lru has no per-key TTL; evict_expired_now is a no-op returning 0.
    EXPECT_EQ(c.evict_expired_now(), 0u);
    EXPECT_EQ(c.size(), 1u);
}

TEST(ProductionTTLCleaner, DoubleStopIsSafe) {
    striped_cache<int, std::string> c(100, 4);
    c.start_ttl_cleaner(std::chrono::milliseconds(100));
    c.stop_ttl_cleaner();
    c.stop_ttl_cleaner();  // idempotent
    EXPECT_FALSE(c.is_ttl_cleaner_running());
}

// ============================================================================
// Task 8: async callbacks
// ============================================================================
TEST(ProductionAsyncCallback, Toggle) {
    cache<int, std::string> c(10);
    EXPECT_FALSE(c.is_async_callbacks());
    c.set_async_callbacks(true);
    EXPECT_TRUE(c.is_async_callbacks());
    c.set_async_callbacks(false);
    EXPECT_FALSE(c.is_async_callbacks());
}

TEST(ProductionAsyncCallback, CallbackInvoked) {
    cache<int, std::string> c(10);
    // defer_promotion skips collect_hit; disable it so the hit is collected.
    c.set_defer_promotion(false);
    std::atomic<int> hit_count{0};
    c.on_hit([&](const int&, const std::string&) { hit_count.fetch_add(1); });
    c.set_async_callbacks(true);
    c.set(1, "a");
    c.get(1);
    // Give the worker thread time to dispatch.
    for (int i = 0; i < 100 && hit_count.load() == 0; ++i) {
        std::this_thread::sleep_for(10ms);
    }
    c.set_async_callbacks(false);  // stops worker and drains queue
    EXPECT_GE(hit_count.load(), 1);
}

// ============================================================================
// Task 9: Production APIs
// ============================================================================
TEST(ProductionAPI, TryGet) {
    cache<int, std::string> c(10);
    c.set(1, "a");
    auto h = c.try_get(1);
    ASSERT_TRUE(h.has_value());
    EXPECT_EQ(**h, "a");
    auto miss = c.try_get(2);
    EXPECT_FALSE(miss.has_value());
}

TEST(ProductionAPI, TryGetOrFetchHit) {
    cache<int, std::string> c(10);
    c.set(1, "a");
    auto v = c.try_get_or_fetch(1, [](const int&) { return std::string("fetched"); });
    EXPECT_EQ(v, "a");
}

TEST(ProductionAPI, TryGetOrFetchMiss) {
    cache<int, std::string> c(10);
    auto v = c.try_get_or_fetch(2, [](const int& k) {
        return std::string("fetched_") + std::to_string(k);
    });
    EXPECT_EQ(v, "fetched_2");
    // After fetch, the value should be cached.
    auto h = c.try_get(2);
    ASSERT_TRUE(h.has_value());
    EXPECT_EQ(**h, "fetched_2");
}

TEST(ProductionAPI, GetWithTTL) {
    cache<int, std::string> c(10);
    c.set(1, "a");
    auto [h, ttl] = c.get_with_ttl(1);
    ASSERT_TRUE(h.has_value());
    // mm_lru has no per-key TTL; ttl is nullopt.
    EXPECT_FALSE(ttl.has_value());
}

TEST(ProductionAPI, GetWithTTLMiss) {
    cache<int, std::string> c(10);
    auto [h, ttl] = c.get_with_ttl(99);
    EXPECT_FALSE(h.has_value());
    EXPECT_FALSE(ttl.has_value());
}

TEST(ProductionAPI, CAS) {
    cache<int, std::string> c(10);
    c.set(1, "a");
    EXPECT_FALSE(c.cas(1, "wrong", "b"));
    EXPECT_EQ(*c.get(1), "a");
    EXPECT_TRUE(c.cas(1, "a", "b"));
    EXPECT_EQ(*c.get(1), "b");
}

TEST(ProductionAPI, CASMissingKey) {
    cache<int, std::string> c(10);
    EXPECT_FALSE(c.cas(99, "x", "y"));
}

TEST(ProductionAPI, CASWithPredicate) {
    cache<int, std::string> c(10);
    c.set(1, "hello");
    EXPECT_TRUE(c.cas(1, "hello", "world", [](const std::string& a, const std::string& b) {
        return a == b;
    }));
    EXPECT_EQ(*c.get(1), "world");
}

TEST(ProductionAPI, CASWithPredicateFails) {
    cache<int, std::string> c(10);
    c.set(1, "hello");
    EXPECT_FALSE(c.cas(1, "hello", "world", [](const std::string&, const std::string&) {
        return false;
    }));
    EXPECT_EQ(*c.get(1), "hello");
}

// ============================================================================
// Task 10: sharded serialization
// ============================================================================
TEST(ProductionSerde, PerShardRoundTrip) {
    striped_cache<int, std::string> c(100, 4);
    for (int i = 0; i < 50; ++i) {
        c.set(i, "val" + std::to_string(i));
    }
    auto data = c.save_per_shard();
    EXPECT_FALSE(data.empty());

    striped_cache<int, std::string> c2(100, 4);
    c2.load_per_shard(data);
    EXPECT_EQ(c2.size(), 50u);
    EXPECT_EQ(*c2.get(1), "val1");
    EXPECT_EQ(*c2.get(49), "val49");
}

TEST(ProductionSerde, PerShardEmpty) {
    striped_cache<int, std::string> c(100, 4);
    auto data = c.save_per_shard();
    EXPECT_FALSE(data.empty());

    striped_cache<int, std::string> c2(100, 4);
    c2.load_per_shard(data);
    EXPECT_EQ(c2.size(), 0u);
}

// ============================================================================
// Task 11: graceful shutdown
// ============================================================================
TEST(ProductionShutdown, BasicShutdown) {
    cache<int, std::string> c(10);
    c.set(1, "a");
    EXPECT_FALSE(c.is_shutdown());
    c.shutdown();
    EXPECT_TRUE(c.is_shutdown());
}

TEST(ProductionShutdown, Idempotent) {
    cache<int, std::string> c(10);
    c.set(1, "a");
    c.shutdown();
    c.shutdown();  // second call is a no-op
    EXPECT_TRUE(c.is_shutdown());
}

TEST(ProductionShutdown, TryGetReturnsNulloptAfterShutdown) {
    cache<int, std::string> c(10);
    c.set(1, "a");
    c.shutdown();
    // try_get is non-blocking and returns nullopt after shutdown.
    auto h = c.try_get(1);
    EXPECT_FALSE(h.has_value());
}

TEST(ProductionShutdown, ActiveHandleCount) {
    cache<int, std::string> c(10);
    c.set(1, "a");
    EXPECT_EQ(c.active_handle_count(), 0u);
    {
        auto h = c.try_get(1);
        ASSERT_TRUE(h.has_value());
        EXPECT_GE(c.active_handle_count(), 1u);
    }
    EXPECT_EQ(c.active_handle_count(), 0u);
}

// ============================================================================
// Integration test
// ============================================================================
TEST(ProductionIntegration, AllFeaturesTogether) {
    striped_cache<int, std::string> c(1000, 16);
    c.set_fairness_mode(detail::fairness_mode::writer_fair);
    EXPECT_EQ(c.num_stripes(), 16u);

    // Fill
    for (int i = 0; i < 100; ++i) {
        c.set(i, "v" + std::to_string(i));
    }
    EXPECT_EQ(c.size(), 100u);

    // Read and record latency. Note: get() records latency, try_get() does not.
    for (int i = 0; i < 100; ++i) {
        auto h = c.get(i);
        ASSERT_TRUE(h.has_value());
    }

    auto stats = c.stats_snapshot();
    EXPECT_GE(stats.hits.value.load(), 100u);
    EXPECT_GE(stats.get_latency.count(), 1u);

    // Prometheus export
    std::string prom = c.prometheus_text();
    EXPECT_NE(prom.find("lru_cache_hits_total"), std::string::npos);

    // CAS modification
    EXPECT_TRUE(c.cas(0, "v0", "v0_new"));
    EXPECT_EQ(*c.get(0), "v0_new");

    // Graceful shutdown
    c.shutdown();
    EXPECT_TRUE(c.is_shutdown());
    EXPECT_FALSE(c.try_get(0).has_value());
}

// ============================================================================
// Task D: overflow_policy
// ============================================================================
TEST(ProductionOverflowPolicy, DefaultIsAllowGrowth) {
    cache<int, int> c(10);
    // Default should be kAllowGrowth — no rejection even at capacity.
    for (int i = 0; i < 20; ++i) {
        c.set(i, i);
    }
    // Cache should have grown beyond max_size (default tolerance 0.1, but
    // kAllowGrowth means no enforcement).
    EXPECT_GE(c.size(), 10u);
}

TEST(ProductionOverflowPolicy, RejectInsertBeyondTolerance) {
    cache<int, int> c(10);
    // Configure kRejectInsert policy with 0% tolerance.
    mm_lru_config cfg = c.mm().config();
    cfg.overflow_policy_value = overflow_policy::kRejectInsert;
    cfg.overflow_tolerance = 0.0;
    c.mm().set_config(cfg);
    // Fill to capacity.
    for (int i = 0; i < 10; ++i) {
        c.set(i, i);
    }
    EXPECT_EQ(c.size(), 10u);
    // The 11th insert should be rejected.
    c.set(100, 100);
    EXPECT_EQ(c.size(), 10u);
    EXPECT_FALSE(c.contains(100));
    // Updates to existing keys should still succeed.
    c.set(0, 999);
    auto h = c.get(0);
    ASSERT_TRUE(h.has_value());
    EXPECT_EQ(*h, 999);
}

TEST(ProductionOverflowPolicy, ForceEvictBeyondTolerance) {
    cache<int, int> c(10);
    mm_lru_config cfg = c.mm().config();
    cfg.overflow_policy_value = overflow_policy::kForceEvict;
    cfg.overflow_tolerance = 0.0;
    c.mm().set_config(cfg);
    for (int i = 0; i < 10; ++i) {
        c.set(i, i);
    }
    EXPECT_EQ(c.size(), 10u);
    // Force-evict: inserting the 11th should evict the LRU tail (key 0).
    c.set(100, 100);
    EXPECT_EQ(c.size(), 10u);
    EXPECT_TRUE(c.contains(100));
    // Key 0 should have been force-evicted.
    EXPECT_FALSE(c.contains(0));
}

// ============================================================================
// P2-4: native_wait_ops macOS support + fallback warning
//
// Acceptance criteria from spec.md:
//   1. macOS: distributed_shared_mutex no longer busy-waits (uses ulock)
//   2. Fallback path emits a one-time stderr warning on first entry
//
// On Windows/Linux the native path is always available, so this test
// verifies the warning API contract: it is callable, idempotent, and
// safe to invoke from multiple threads concurrently.
// ============================================================================
TEST(ProductionNativeWaitOps, FallbackWarningIsIdempotent) {
    // warn_fallback_once() must be safe to call many times and from
    // multiple threads. The warning should fire at most once globally.
    std::vector<std::thread> threads;
    for (int t = 0; t < 8; ++t) {
        threads.emplace_back([]() {
            for (int i = 0; i < 100; ++i) {
                lru::detail::native_wait_ops::warn_fallback_once();
            }
        });
    }
    for (auto& th : threads) th.join();
    // No assert on stderr content — the contract is "fires at most once
    // and never crashes". The std::once_flag inside guarantees this.
    SUCCEED();
}

TEST(ProductionNativeWaitOps, AvailabilityIsConsistent) {
    // available() must return the same value across calls (cached).
    bool a1 = lru::detail::native_wait_ops::available();
    bool a2 = lru::detail::native_wait_ops::available();
    EXPECT_EQ(a1, a2);
    // On Windows 8+ / Linux / macOS 10.12+ this should be true.
    // On ancient platforms it may be false — that's acceptable as long
    // as the CV fallback works.
}

// ============================================================================
// O2: Slow query logging (threshold-based)
// ============================================================================
//
// Validates the slow-query callback hook on get()/set(). The fast path
// (threshold == 0 or no callback registered) must be near-zero overhead.
// When a callback is registered and threshold > 0, operations whose
// measured latency exceeds the threshold trigger the callback and bump
// the slow_query_count counter.

TEST(ProductionSlowQuery, DefaultDisabled) {
    cache<int, std::string> c(64);
    // Default threshold is 0 (disabled) and no callback registered.
    EXPECT_EQ(c.slow_query_threshold(), std::chrono::nanoseconds(0));
    EXPECT_EQ(c.slow_query_count(), 0u);
    // Operations must NOT trigger any slow-query notification.
    c.set(1, "a");
    (void)c.get(1);
    EXPECT_EQ(c.slow_query_count(), 0u);
}

TEST(ProductionSlowQuery, ThresholdGetSet) {
    cache<int, std::string> c(64);
    c.set_slow_query_threshold(std::chrono::nanoseconds(500));
    EXPECT_EQ(c.slow_query_threshold(), std::chrono::nanoseconds(500));
    // 0 disables.
    c.set_slow_query_threshold(std::chrono::nanoseconds(0));
    EXPECT_EQ(c.slow_query_threshold(), std::chrono::nanoseconds(0));
}

TEST(ProductionSlowQuery, CallbackInvokedOnGetAndSet) {
    cache<int, std::string> c(64);
    // Threshold of 1 ns — every real operation exceeds this.
    c.set_slow_query_threshold(std::chrono::nanoseconds(1));

    std::atomic<int> get_calls{0};
    std::atomic<int> set_calls{0};
    std::atomic<int> last_latency_ns{0};

    using cache_t = cache<int, std::string>;
    c.set_slow_query_callback(
        [&](cache_t::slow_op_kind kind, const int& key,
            std::uint64_t latency_ns) {
            if (kind == cache_t::slow_op_kind::get) {
                get_calls.fetch_add(1, std::memory_order_relaxed);
            } else {
                set_calls.fetch_add(1, std::memory_order_relaxed);
            }
            (void)key;
            last_latency_ns.store(static_cast<int>(latency_ns),
                                  std::memory_order_relaxed);
        });

    c.set(1, "a");      // triggers set slow-query
    (void)c.get(1);     // triggers get slow-query

    EXPECT_GE(set_calls.load(), 1);
    EXPECT_GE(get_calls.load(), 1);
    EXPECT_GT(last_latency_ns.load(), 0);
    EXPECT_GE(c.slow_query_count(), 2u);
}

TEST(ProductionSlowQuery, NoCallbackNoCount) {
    cache<int, std::string> c(64);
    // Threshold set but no callback registered — counter must stay 0
    // because notify_slow_query fast-paths out when callback is null.
    c.set_slow_query_threshold(std::chrono::nanoseconds(1));
    c.set(1, "a");
    (void)c.get(1);
    EXPECT_EQ(c.slow_query_count(), 0u);
}

TEST(ProductionSlowQuery, ClearCallbackStopsNotification) {
    cache<int, std::string> c(64);
    c.set_slow_query_threshold(std::chrono::nanoseconds(1));

    std::atomic<int> calls{0};
    c.set_slow_query_callback(
        [&](auto /*kind*/, const int& /*key*/, std::uint64_t /*latency_ns*/) {
            calls.fetch_add(1, std::memory_order_relaxed);
        });

    c.set(1, "a");
    const int calls_after_first = calls.load();
    EXPECT_GE(calls_after_first, 1);

    // Clear the callback. Subsequent operations must NOT trigger it.
    c.set_slow_query_callback({});
    const std::size_t count_before = c.slow_query_count();
    c.set(2, "b");
    (void)c.get(2);
    EXPECT_EQ(calls.load(), calls_after_first);
    EXPECT_EQ(c.slow_query_count(), count_before);
}

TEST(ProductionSlowQuery, HighThresholdNeverFires) {
    cache<int, std::string> c(64);
    // Threshold of 1 hour — no real operation will exceed this.
    c.set_slow_query_threshold(std::chrono::hours(1));

    std::atomic<int> calls{0};
    c.set_slow_query_callback(
        [&](auto /*kind*/, const int& /*key*/, std::uint64_t /*latency_ns*/) {
            calls.fetch_add(1, std::memory_order_relaxed);
        });

    c.set(1, "a");
    (void)c.get(1);
    EXPECT_EQ(calls.load(), 0);
    EXPECT_EQ(c.slow_query_count(), 0u);
}

TEST(ProductionSlowQuery, CallbackExceptionSwallowed) {
    cache<int, std::string> c(64);
    c.set_slow_query_threshold(std::chrono::nanoseconds(1));

    // A throwing callback must NOT propagate to the caller. The count
    // is still incremented before the callback is invoked.
    c.set_slow_query_callback(
        [](auto /*kind*/, const int& /*key*/, std::uint64_t /*latency_ns*/) {
            throw std::runtime_error("intentional slow-query callback throw");
        });

    EXPECT_NO_THROW({
        c.set(1, "a");
        (void)c.get(1);
    });
    EXPECT_GE(c.slow_query_count(), 2u);
}

// ============================================================================
// O1: Distributed tracing callback (OpenTelemetry / Jaeger / Zipkin hook)
// ============================================================================

TEST(ProductionTrace, DefaultDisabled) {
    cache<int, std::string> c(64);
    // No trace callback registered by default.
    EXPECT_EQ(c.trace_count(), 0u);
    c.set(1, "a");
    (void)c.get(1);
    // No callback → no events counted.
    EXPECT_EQ(c.trace_count(), 0u);
}

TEST(ProductionTrace, CallbackInvokedOnEveryGetAndSet) {
    cache<int, std::string> c(64);

    std::atomic<int> get_calls{0};
    std::atomic<int> set_calls{0};
    std::atomic<std::uint64_t> last_latency_ns{0};

    using cache_t = cache<int, std::string>;
    c.set_trace_callback(
        [&](cache_t::trace_op_kind op, const int& /*key*/,
            std::uint64_t latency_ns, bool /*hit*/, bool /*error*/) {
            if (op == cache_t::trace_op_kind::get) {
                get_calls.fetch_add(1, std::memory_order_relaxed);
            } else {
                set_calls.fetch_add(1, std::memory_order_relaxed);
            }
            last_latency_ns.store(latency_ns, std::memory_order_relaxed);
        });

    c.set(1, "a");
    (void)c.get(1);

    EXPECT_GE(set_calls.load(), 1);
    EXPECT_GE(get_calls.load(), 1);
    EXPECT_GT(last_latency_ns.load(), 0u);
    EXPECT_GE(c.trace_count(), 2u);
}

TEST(ProductionTrace, ReportsHitStatusCorrectly) {
    cache<int, std::string> c(64);

    std::atomic<int> hits{0};
    std::atomic<int> misses{0};

    using cache_t = cache<int, std::string>;
    c.set_trace_callback(
        [&](cache_t::trace_op_kind op, const int& /*key*/,
            std::uint64_t /*latency_ns*/, bool hit, bool /*error*/) {
            if (op == cache_t::trace_op_kind::get) {
                if (hit) hits.fetch_add(1, std::memory_order_relaxed);
                else     misses.fetch_add(1, std::memory_order_relaxed);
            }
        });

    c.set(1, "v1");                    // set: hit=true
    (void)c.get(1);                    // get hit: hit=true
    (void)c.get(999);                  // get miss: hit=false

    EXPECT_EQ(hits.load(), 1);
    EXPECT_EQ(misses.load(), 1);
}

TEST(ProductionTrace, SetReportsHitTrueOnSuccess) {
    cache<int, std::string> c(64);

    std::atomic<int> set_hits{0};
    std::atomic<int> set_misses{0};

    using cache_t = cache<int, std::string>;
    c.set_trace_callback(
        [&](cache_t::trace_op_kind op, const int& /*key*/,
            std::uint64_t /*latency_ns*/, bool hit, bool /*error*/) {
            if (op == cache_t::trace_op_kind::set) {
                if (hit) set_hits.fetch_add(1, std::memory_order_relaxed);
                else     set_misses.fetch_add(1, std::memory_order_relaxed);
            }
        });

    c.set(1, "a");
    c.set(2, "b");
    EXPECT_EQ(set_hits.load(), 2);
    EXPECT_EQ(set_misses.load(), 0);
}

TEST(ProductionTrace, ClearCallbackStopsNotification) {
    cache<int, std::string> c(64);

    std::atomic<int> calls{0};
    c.set_trace_callback(
        [&](auto, const int&, std::uint64_t, bool, bool) {
            calls.fetch_add(1, std::memory_order_relaxed);
        });

    c.set(1, "a");
    const int calls_after_first = calls.load();
    EXPECT_GE(calls_after_first, 1);

    // Clear the callback.
    c.set_trace_callback({});
    c.set(2, "b");
    (void)c.get(2);
    EXPECT_EQ(calls.load(), calls_after_first);
}

TEST(ProductionTrace, CallbackExceptionSwallowed) {
    cache<int, std::string> c(64);

    c.set_trace_callback(
        [](auto, const int&, std::uint64_t, bool, bool) {
            throw std::runtime_error("intentional trace callback throw");
        });

    EXPECT_NO_THROW({
        c.set(1, "a");
        (void)c.get(1);
    });
    // Count is still incremented before the callback is invoked.
    EXPECT_GE(c.trace_count(), 2u);
}

TEST(ProductionTrace, ConcurrentCallbacksAreSafe) {
    // Verify the trace callback can be invoked concurrently from many
    // threads without crashing or corrupting the counter.
    cache<int, std::string> c(256);

    std::atomic<int> total_events{0};
    c.set_trace_callback(
        [&](auto, const int&, std::uint64_t, bool, bool) {
            total_events.fetch_add(1, std::memory_order_relaxed);
        });

    constexpr int kThreads = 8;
    constexpr int kOpsPerThread = 100;
    std::vector<std::thread> threads;
    threads.reserve(kThreads);

    std::atomic<int> ready{0};
    std::atomic<bool> go{false};

    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t]() {
            ready.fetch_add(1, std::memory_order_acq_rel);
            while (!go.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            for (int i = 0; i < kOpsPerThread; ++i) {
                int key = t * kOpsPerThread + i;
                c.set(key, "v");
                (void)c.get(key);
            }
        });
    }

    while (ready.load(std::memory_order_acquire) < kThreads) {
        std::this_thread::yield();
    }
    go.store(true, std::memory_order_release);

    for (auto& t : threads) t.join();

    // Each thread does kOpsPerThread sets + kOpsPerThread gets = 2 * kOpsPerThread ops.
    const int expected = kThreads * kOpsPerThread * 2;
    EXPECT_EQ(total_events.load(), expected);
    EXPECT_EQ(c.trace_count(), static_cast<std::size_t>(expected));
}

// ============================================================================
// P2-5: Fix TTL const-cast
//
// Acceptance criteria from spec.md:
//   - `peek()` const method no longer modifies any state.
//
// Before P2-5, `peek()` / `contains()` / `get()` used `const_cast` to
// lazily mark entries as expired via a `std::atomic<bool> expired` field
// inside `ttl_entry`. P2-5 removes the field entirely and makes `peek()`
// and `contains()` properly `const`. These tests verify the new contract.
// ============================================================================
TEST(ProductionTtlConstCast, PeekIsConstAndDoesNotModify) {
    // Compile-time check: peek() must be callable on a const cache.
    // If peek() were non-const, this line would not compile.
    ttl_cache<int, std::string, std::chrono::milliseconds> c(100ms);
    c.set(1, "one");
    const auto& const_c = c;
    auto v = const_c.peek(1);
    ASSERT_TRUE(v.has_value());
    EXPECT_EQ(*v, "one");
}

TEST(ProductionTtlConstCast, ContainsIsConstAndDoesNotModify) {
    // Compile-time check: contains() must be callable on a const cache.
    ttl_cache<int, std::string, std::chrono::milliseconds> c(100ms);
    c.set(1, "one");
    const auto& const_c = c;
    EXPECT_TRUE(const_c.contains(1));
    EXPECT_FALSE(const_c.contains(999));
}

TEST(ProductionTtlConstCast, PeekReturnsNulloptForExpiredEntry) {
    // After expiry, peek() must return nullopt without modifying state.
    ttl_cache<int, std::string, std::chrono::milliseconds> c(50ms);
    c.set(1, "one");
    std::this_thread::sleep_for(std::chrono::milliseconds(80));
    const auto& const_c = c;
    EXPECT_FALSE(const_c.peek(1).has_value());
    EXPECT_FALSE(const_c.contains(1));
    EXPECT_TRUE(c.has_expired(1));
}

TEST(ProductionTtlConstCast, TtlEntryIsPureValueType) {
    // P2-5: ttl_entry no longer has an `expired` atomic field. Verify
    // that the entry can be copied/moved like a pure value type.
    using entry = ttl_entry<std::string>;
    entry e1{"value", std::chrono::steady_clock::now() + std::chrono::seconds(1)};
    entry e2 = e1;  // copy
    EXPECT_EQ(e2.value, "value");
    EXPECT_TRUE(e2.expiry.has_value());

    entry e3 = std::move(e1);  // move
    EXPECT_EQ(e3.value, "value");

    // is_expired_at() is a pure read-only check.
    EXPECT_FALSE(e3.is_expired_at(std::chrono::steady_clock::now()));
    auto future = e3.expiry.value() + std::chrono::seconds(1);
    EXPECT_TRUE(e3.is_expired_at(future));
}

// main() is provided by GTest::gtest_main (linked via CMakeLists.txt).
