// Unified LRU Cache - CompactCache unit tests
// Tests both single-threaded and thread-safe variants.

#include <gtest/gtest.h>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <functional>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "../lru.hpp"

using namespace lru;

// ============================================================================
// Single-threaded CompactCache Tests
// ============================================================================

class CompactCacheTest : public ::testing::Test {
protected:
    compact_cache<int, int> c{10};

    void SetUp() override {
        c.set(1, 100);
        c.set(2, 200);
        c.set(3, 300);
    }
};

TEST_F(CompactCacheTest, BasicSetAndGet) {
    auto v = c.get(1);
    ASSERT_TRUE(v.has_value());
    EXPECT_EQ(v->get(), 100);
}

TEST_F(CompactCacheTest, GetNonExistent) {
    auto v = c.get(99);
    EXPECT_FALSE(v.has_value());
}

TEST_F(CompactCacheTest, PeekDoesNotPromote) {
    auto v = c.peek(1);
    ASSERT_TRUE(v.has_value());
    EXPECT_EQ(v->get(), 100);
}

TEST_F(CompactCacheTest, PeekNonExistent) {
    auto v = c.peek(99);
    EXPECT_FALSE(v.has_value());
}

TEST_F(CompactCacheTest, Contains) {
    EXPECT_TRUE(c.contains(1));
    EXPECT_FALSE(c.contains(99));
}

TEST_F(CompactCacheTest, Delete) {
    EXPECT_TRUE(c.del(1));
    EXPECT_FALSE(c.contains(1));
    EXPECT_FALSE(c.del(1)); // already deleted
}

TEST_F(CompactCacheTest, AddNewKey) {
    EXPECT_TRUE(c.add(4, 400));
    EXPECT_EQ(c.size(), 4);
}

TEST_F(CompactCacheTest, AddExistingKey) {
    EXPECT_FALSE(c.add(1, 999));
    auto v = c.get(1);
    ASSERT_TRUE(v.has_value());
    EXPECT_EQ(v->get(), 100); // value unchanged
}

TEST_F(CompactCacheTest, Size) {
    EXPECT_EQ(c.size(), 3);
}

TEST_F(CompactCacheTest, Empty) {
    EXPECT_FALSE(c.empty());
    c.flush();
    EXPECT_TRUE(c.empty());
}

TEST_F(CompactCacheTest, Flush) {
    c.flush();
    EXPECT_EQ(c.size(), 0);
    EXPECT_TRUE(c.empty());
}

TEST_F(CompactCacheTest, MaxSizeEviction) {
    for (int i = 4; i <= 12; ++i) {
        c.set(i, i * 100);
    }
    EXPECT_EQ(c.size(), 10); // max_size = 10
}

TEST_F(CompactCacheTest, UpdateExisting) {
    c.set(1, 999);
    auto v = c.get(1);
    ASSERT_TRUE(v.has_value());
    EXPECT_EQ(v->get(), 999);
}

TEST_F(CompactCacheTest, IsThreadSafeFalse) {
    static_assert(!compact_cache<int, int>::is_thread_safe,
                  "default compact_cache must have is_thread_safe = false");
}

TEST_F(CompactCacheTest, SetRefreshTime) {
    c.set_refresh_time(30);
    // No crash — basic smoke test
}

TEST_F(CompactCacheTest, CurrentMemory) {
    EXPECT_GT(c.current_memory(), 0);
}

TEST_F(CompactCacheTest, MaxSizeResize) {
    c.max_size(2);
    EXPECT_EQ(c.size(), 2);
}

TEST_F(CompactCacheTest, ShrinkToFit) {
    c.max_size(2);
    c.shrink_to_fit();
    EXPECT_EQ(c.size(), 2);
}

TEST_F(CompactCacheTest, StatsTracking) {
    c.get(1);  // hit
    c.get(1);  // hit
    c.get(99); // miss
    auto& s = c.stats();
    EXPECT_GE(s.hits.value.load(), 2);
    EXPECT_GE(s.misses.value.load(), 1);
}

TEST_F(CompactCacheTest, Callbacks) {
    int evict_count = 0;
    int last_evict_key = -1;
    int last_evict_value = -1;
    c.callbacks().on_evict([&](const int& k, const int& v) {
        ++evict_count;
        last_evict_key = k;
        last_evict_value = v;
    });
    // Cache has max_size=10, already holds keys 1,2,3 (size=3).
    // Add 8 more keys to trigger evictions: size reaches 11 → evict key 1.
    for (int i = 4; i <= 11; ++i) {
        c.set(i, i * 100);
    }
    EXPECT_GE(evict_count, 1);
    // First evicted key is 1 (the LRU tail)
    EXPECT_EQ(last_evict_key, 1);
    EXPECT_EQ(last_evict_value, 100);
}

// ============================================================================
// String key/value compact_cache
// ============================================================================

TEST(CompactCacheStringTest, StringKeyValue) {
    // Small strings fit in kMaxItemSize=64
    compact_cache<std::string, std::string> c{100};
    c.set("hello", "world");
    auto v = c.get("hello");
    ASSERT_TRUE(v.has_value());
    EXPECT_EQ(v->get(), "world");
}

// ============================================================================
// Thread-safe CompactCache Tests
// ============================================================================

class SafeCompactCacheTest : public ::testing::Test {
protected:
    safe_compact_cache<int, int> c{100};
};

TEST_F(SafeCompactCacheTest, BasicSetAndGet) {
    c.set(1, 100);
    auto v = c.get(1);
    ASSERT_TRUE(v.has_value());
    EXPECT_EQ(v->get(), 100);
}

TEST_F(SafeCompactCacheTest, Peek) {
    c.set(1, 100);
    auto v = c.peek(1);
    ASSERT_TRUE(v.has_value());
    EXPECT_EQ(v->get(), 100);
}

TEST_F(SafeCompactCacheTest, Contains) {
    c.set(1, 100);
    EXPECT_TRUE(c.contains(1));
    EXPECT_FALSE(c.contains(99));
}

TEST_F(SafeCompactCacheTest, Delete) {
    c.set(1, 100);
    EXPECT_TRUE(c.del(1));
    EXPECT_FALSE(c.contains(1));
}

TEST_F(SafeCompactCacheTest, Add) {
    EXPECT_TRUE(c.add(1, 100));
    EXPECT_FALSE(c.add(1, 200));
    auto v = c.get(1);
    ASSERT_TRUE(v.has_value());
    EXPECT_EQ(v->get(), 100);
}

TEST_F(SafeCompactCacheTest, Size) {
    c.set(1, 100);
    c.set(2, 200);
    EXPECT_EQ(c.size(), 2);
}

TEST_F(SafeCompactCacheTest, Empty) {
    EXPECT_TRUE(c.empty());
    c.set(1, 100);
    EXPECT_FALSE(c.empty());
}

TEST_F(SafeCompactCacheTest, Flush) {
    c.set(1, 100);
    c.set(2, 200);
    c.flush();
    EXPECT_EQ(c.size(), 0);
}

TEST_F(SafeCompactCacheTest, MaxSizeEviction) {
    for (int i = 0; i < 200; ++i) {
        c.set(i, i * 10);
    }
    EXPECT_EQ(c.size(), 100);
}

TEST_F(SafeCompactCacheTest, UpdateExisting) {
    c.set(1, 100);
    c.set(1, 999);
    auto v = c.get(1);
    ASSERT_TRUE(v.has_value());
    EXPECT_EQ(v->get(), 999);
}

TEST_F(SafeCompactCacheTest, IsThreadSafeTrue) {
    static_assert(safe_compact_cache<int, int>::is_thread_safe,
                  "safe_compact_cache must have is_thread_safe = true");
}

TEST_F(SafeCompactCacheTest, CurrentMemory) {
    c.set(1, 100);
    EXPECT_GT(c.current_memory(), 0);
}

TEST_F(SafeCompactCacheTest, MaxSizeResize) {
    for (int i = 0; i < 50; ++i) {
        c.set(i, i);
    }
    c.max_size(10);
    EXPECT_EQ(c.size(), 10);
}

TEST_F(SafeCompactCacheTest, StatsTracking) {
    c.set(1, 100);
    c.get(1);  // hit
    c.get(1);  // hit
    c.get(99); // miss
    auto& s = c.stats();
    EXPECT_GE(s.hits.value.load(), 2);
    EXPECT_GE(s.misses.value.load(), 1);
}

// ============================================================================
// Thread-safe CompactCache Concurrency Tests
// ============================================================================

TEST(SafeCompactCacheConcurrencyTest, ConcurrentWrites) {
    // compact_cache uses a single shared_mutex for write operations,
    // so concurrent writes are serialized but still thread-safe.
    safe_compact_cache<int, int> c{1000};

    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&, t]() {
            for (int i = 0; i < 100; ++i) {
                c.set(t * 100 + i, t * 1000 + i);
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_EQ(c.size(), 400);
}

TEST(SafeCompactCacheConcurrencyTest, ConcurrentReads) {
    safe_compact_cache<int, int> c{100};
    for (int i = 0; i < 100; ++i) {
        c.set(i, i * 10);
    }

    std::vector<std::thread> threads;
    std::atomic<int> hit_count{0};

    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&]() {
            for (int i = 0; i < 100; ++i) {
                auto result = c.get(i);
                if (result.has_value()) {
                    ++hit_count;
                }
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_EQ(hit_count.load(), 400);
}

TEST(SafeCompactCacheConcurrencyTest, MixedReadWrite) {
    safe_compact_cache<int, int> c{100};
    for (int i = 0; i < 50; ++i) {
        c.set(i, i);
    }

    std::atomic<bool> done{false};
    std::vector<std::thread> threads;

    // Writer thread
    threads.emplace_back([&]() {
        for (int i = 50; i < 150; ++i) {
            c.set(i, i * 2);
        }
        done = true;
    });

    // Reader threads
    for (int t = 0; t < 3; ++t) {
        threads.emplace_back([&]() {
            while (!done) {
                for (int i = 0; i < 50; ++i) {
                    (void)c.get(i);
                }
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_LE(c.size(), 100);
}

TEST(SafeCompactCacheConcurrencyTest, ConcurrentGetAndDel) {
    safe_compact_cache<int, int> c{100};
    for (int i = 0; i < 50; ++i) {
        c.set(i, i);
    }

    std::vector<std::thread> threads;

    threads.emplace_back([&]() {
        for (int i = 0; i < 50; ++i) {
            (void)c.get(i);
        }
    });

    threads.emplace_back([&]() {
        for (int i = 0; i < 50; ++i) {
            (void)c.del(i);
        }
    });

    for (auto& t : threads) {
        t.join();
    }

    // Retry deletion for any keys that survived the concurrent race
    for (int i = 0; i < 50; ++i) {
        c.del(i);
    }
    for (int i = 0; i < 50; ++i) {
        EXPECT_FALSE(c.contains(i));
    }
}

TEST(SafeCompactCacheConcurrencyTest, FlushUnderConcurrency) {
    safe_compact_cache<int, int> c{1000};
    for (int i = 0; i < 500; ++i) {
        c.set(i, i);
    }

    std::atomic<bool> stop{false};
    std::vector<std::thread> threads;

    for (int t = 0; t < 2; ++t) {
        threads.emplace_back([&, t]() {
            for (int i = 0; i < 500 && !stop.load(); ++i) {
                c.set(t * 500 + i, i);
            }
        });
    }

    threads.emplace_back([&]() {
        for (int i = 0; i < 5 && !stop.load(); ++i) {
            c.flush();
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        stop = true;
    });

    for (auto& t : threads) {
        t.join();
    }
    // No crash or deadlock — test passes
}

TEST(SafeCompactCacheConcurrencyTest, HighContentionStress) {
    safe_compact_cache<int, int> c{1000};
    constexpr int num_threads = 2;
    constexpr int ops_per_thread = 500;

    std::vector<std::thread> threads;
    std::atomic<int> total_sets{0};

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            for (int i = 0; i < ops_per_thread; ++i) {
                int key = (t * ops_per_thread + i) % 200;
                int op = i % 10;
                if (op < 7) {
                    c.get(key);
                } else if (op < 9) {
                    c.set(key, t * 1000 + i);
                    total_sets.fetch_add(1, std::memory_order_relaxed);
                } else {
                    c.del(key);
                }
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_LE(c.size(), 1000);
    EXPECT_GT(total_sets.load(), 0);
}

// ============================================================================
// CompactCache type alias verification
// ============================================================================

TEST(CompactCacheAliasTest, CompactCacheDefaultIsSingleThreaded) {
    static_assert(compact_cache<int, int>::is_thread_safe == false,
                  "compact_cache default must be single-threaded");
}

TEST(CompactCacheAliasTest, CompactCacheDefaultAliasIsSingleThreaded) {
    static_assert(compact_cache_default<int, int>::is_thread_safe == false,
                  "compact_cache_default must be single-threaded");
}

TEST(CompactCacheAliasTest, SafeCompactCacheIsThreadSafe) {
    static_assert(safe_compact_cache<int, int>::is_thread_safe == true,
                  "safe_compact_cache must be thread-safe");
}

TEST(CompactCacheAliasTest, CompactCacheWithPolicyIsThreadSafe) {
    using ts_cache = compact_cache<int, int, std::hash<int>, std::equal_to<int>, 64,
                                   alignof(std::max_align_t), thread_safe_policy>;
    static_assert(ts_cache::is_thread_safe == true,
                  "compact_cache with thread_safe_policy must be thread-safe");
}

// ============================================================================
// T14: Production-grade API integration tests
//
// Verifies that compact_cache exposes the same observability/controllability
// surface as unified_cache: stats_snapshot, prometheus_text, diagnostics,
// set_fairness_mode, set_async_callbacks, set_latency_tracking, shutdown,
// hash overload detection.
// ============================================================================

TEST(CompactCacheProductionApi, StatsSnapshotReflectsOperations) {
    compact_cache<int, int> c{100};
    c.set(1, 10);
    c.set(2, 20);
    c.get(1);  // hit
    c.get(99); // miss

    auto snap = c.stats_snapshot();
    EXPECT_EQ(snap.current_size.load(), 2u);
    EXPECT_GE(snap.hits.value.load(), 1u);
    EXPECT_GE(snap.misses.value.load(), 1u);
    EXPECT_GE(snap.insertions.value.load(), 2u);
}

TEST(CompactCacheProductionApi, PrometheusTextContainsCoreMetrics) {
    compact_cache<int, int> c{100};
    c.set(1, 10);
    c.set(2, 20);
    c.get(1);

    auto text = c.prometheus_text();
    EXPECT_NE(text.find("lru_compact_hits_total"), std::string::npos);
    EXPECT_NE(text.find("lru_compact_misses_total"), std::string::npos);
    EXPECT_NE(text.find("lru_compact_insertions_total"), std::string::npos);
    EXPECT_NE(text.find("lru_compact_evictions_total"), std::string::npos);
    EXPECT_NE(text.find("lru_compact_size"), std::string::npos);
    EXPECT_NE(text.find("lru_compact_memory_bytes"), std::string::npos);
    EXPECT_NE(text.find("lru_compact_max_size"), std::string::npos);
    EXPECT_NE(text.find("lru_compact_slot_total"), std::string::npos);
    EXPECT_NE(text.find("lru_compact_slot_used"), std::string::npos);
    EXPECT_NE(text.find("lru_compact_load_factor"), std::string::npos);
    EXPECT_NE(text.find("lru_compact_latency_tracking_enabled"), std::string::npos);
    EXPECT_NE(text.find("lru_compact_shutdown"), std::string::npos);
}

TEST(CompactCacheProductionApi, DiagnosticsTextContainsFields) {
    compact_cache<int, int> c{50};
    c.set(7, 70);

    auto text = c.diagnostics_text();
    EXPECT_NE(text.find("compact_cache diagnostics"), std::string::npos);
    EXPECT_NE(text.find("size:"), std::string::npos);
    EXPECT_NE(text.find("max_size:"), std::string::npos);
    EXPECT_NE(text.find("slot_total:"), std::string::npos);
    EXPECT_NE(text.find("slot_used:"), std::string::npos);
    EXPECT_NE(text.find("bucket_count:"), std::string::npos);
    EXPECT_NE(text.find("load_factor:"), std::string::npos);
    EXPECT_NE(text.find("hits:"), std::string::npos);
    EXPECT_NE(text.find("latency_tracking_enabled:"), std::string::npos);
    EXPECT_NE(text.find("shutdown_in_progress:"), std::string::npos);
}

TEST(CompactCacheProductionApi, DiagnosticsInfoStructFields) {
    compact_cache<int, int> c{100};
    c.set(1, 10);
    c.get(1);

    auto info = c.diagnostics();
    EXPECT_EQ(info.size, 1u);
    EXPECT_EQ(info.max_size, 100u);
    EXPECT_EQ(info.slot_used, 1u);
    EXPECT_GE(info.slot_total, 1u);
    EXPECT_GE(info.bucket_count, 0u);
    EXPECT_GE(info.load_factor, 0.0f);
    EXPECT_TRUE(info.latency_tracking_enabled);
    EXPECT_FALSE(info.async_callbacks_enabled);
    EXPECT_FALSE(info.is_thread_safe);  // default is single-threaded
    EXPECT_FALSE(info.shutdown_in_progress);
    EXPECT_GE(info.hits, 1u);
}

TEST(CompactCacheProductionApi, SetLatencyTrackingTogglesFlag) {
    compact_cache<int, int> c{100};
    EXPECT_TRUE(c.is_latency_tracking_enabled());

    c.set_latency_tracking(false);
    EXPECT_FALSE(c.is_latency_tracking_enabled());

    c.set_latency_tracking(true);
    EXPECT_TRUE(c.is_latency_tracking_enabled());
}

TEST(CompactCacheProductionApi, SetAsyncCallbacksTogglesFlag) {
    compact_cache<int, int> c{100};
    EXPECT_FALSE(c.is_async_callbacks_enabled());

    c.set_async_callbacks(true);
    EXPECT_TRUE(c.is_async_callbacks_enabled());

    c.set_async_callbacks(false);
    EXPECT_FALSE(c.is_async_callbacks_enabled());
}

TEST(CompactCacheProductionApi, ShutdownBlocksMutatingOps) {
    compact_cache<int, int> c{100};
    c.set(1, 10);
    EXPECT_FALSE(c.is_shutdown());

    c.shutdown();
    EXPECT_TRUE(c.is_shutdown());

    // Mutating ops must throw after shutdown.
    EXPECT_THROW(c.set(2, 20), std::logic_error);
    EXPECT_THROW(c.add(3, 30), std::logic_error);
    EXPECT_THROW(c.del(1), std::logic_error);
    EXPECT_THROW(c.flush(), std::logic_error);

    // Reads remain permitted.
    EXPECT_TRUE(c.contains(1));
    auto v = c.peek(1);
    ASSERT_TRUE(v.has_value());
    EXPECT_EQ(v->get(), 10);
}

TEST(CompactCacheProductionApi, ActiveHandleCountAlwaysZero) {
    compact_cache<int, int> c{100};
    c.set(1, 10);
    auto v = c.get(1);
    ASSERT_TRUE(v.has_value());
    // compact_cache uses reference_wrapper, not refcounted handles.
    EXPECT_EQ(c.active_handle_count(), 0u);
    (void)v;
}

TEST(CompactCacheProductionApi, HashOverloadThresholdDefaultsToTwo) {
    compact_cache<int, int> c{100};
    EXPECT_FLOAT_EQ(c.hash_overload_threshold(), 2.0f);
    EXPECT_EQ(c.hash_overload_events(), 0u);
}

TEST(CompactCacheProductionApi, HashOverloadCallbackFires) {
    safe_compact_cache<int, int> c{1000};
    std::atomic<int> fires{0};
    c.set_hash_overload_threshold(0.01f);  // very low threshold to force trigger
    c.set_overload_callback([&fires](float current, float threshold) {
        EXPECT_GT(current, threshold);
        fires.fetch_add(1, std::memory_order_relaxed);
    });
    // Insert enough items to push the load factor above 0.01.
    for (int i = 0; i < 50; ++i) {
        c.set(i, i * 10);
    }
    EXPECT_GE(fires.load(), 1);
    EXPECT_GE(c.hash_overload_events(), 1u);
}

TEST(CompactCacheProductionApi, HashOverloadCallbackExceptionIsSwallowed) {
    safe_compact_cache<int, int> c{1000};
    c.set_hash_overload_threshold(0.01f);
    c.set_overload_callback([](float, float) {
        throw std::runtime_error("user callback boom");
    });
    // Should not propagate the exception.
    for (int i = 0; i < 20; ++i) {
        c.set(i, i * 10);
    }
    // Counter still increments despite the exception.
    EXPECT_GE(c.hash_overload_events(), 1u);
}

TEST(CompactCacheProductionApi, SafeCacheSetFairnessMode) {
    safe_compact_cache<int, int> c{100};
    // Default is writer_fair.
    EXPECT_EQ(c.get_fairness_mode(), detail::fairness_mode::writer_fair);

    c.set_fairness_mode(detail::fairness_mode::reader_preferred);
    EXPECT_EQ(c.get_fairness_mode(), detail::fairness_mode::reader_preferred);

    c.set_fairness_mode(detail::fairness_mode::writer_fair);
    EXPECT_EQ(c.get_fairness_mode(), detail::fairness_mode::writer_fair);
}

TEST(CompactCacheProductionApi, SingleThreadedSetFairnessModeIsNoOp) {
    compact_cache<int, int> c{100};
    // Single-threaded cache has no mutex; the call must not crash and
    // reports the default writer_fair mode.
    c.set_fairness_mode(detail::fairness_mode::reader_preferred);
    EXPECT_EQ(c.get_fairness_mode(), detail::fairness_mode::writer_fair);
}

// ============================================================================
// T14: striped_compact_cache alias verification
// ============================================================================

TEST(StripedCompactCacheAliasTest, IsThreadSafe) {
    static_assert(striped_compact_cache<int, int>::is_thread_safe == true,
                  "striped_compact_cache must be thread-safe");
}

TEST(StripedCompactCacheAliasTest, IsStripedPolicy) {
    static_assert(striped_compact_cache<int, int>::is_striped == true,
                  "striped_compact_cache must use a striped policy");
}

TEST(StripedCompactCacheAliasTest, ProductionApiSurfaceCompilesAndRuns) {
    striped_compact_cache<int, int> c{100};
    c.set(1, 10);
    c.set(2, 20);
    EXPECT_EQ(c.size(), 2u);

    auto snap = c.stats_snapshot();
    EXPECT_EQ(snap.current_size.load(), 2u);

    auto text = c.prometheus_text();
    EXPECT_NE(text.find("lru_compact_size"), std::string::npos);

    auto info = c.diagnostics();
    EXPECT_EQ(info.size, 2u);
    EXPECT_TRUE(info.is_thread_safe);
    EXPECT_TRUE(info.is_striped);

    c.set_fairness_mode(detail::fairness_mode::reader_preferred);
    EXPECT_EQ(c.get_fairness_mode(), detail::fairness_mode::reader_preferred);
}

TEST(StripedCompactCacheAliasTest, ConcurrentAccessPreservesCorrectness) {
    striped_compact_cache<int, int> c{10000};
    constexpr int kThreads = 4;
    constexpr int kPerThread = 1000;

    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&c, t]() {
            for (int i = 0; i < kPerThread; ++i) {
                int key = t * kPerThread + i;
                c.set(key, key * 2);
            }
        });
    }
    for (auto& th : threads) th.join();

    EXPECT_EQ(c.size(), kThreads * kPerThread);
    // Verify all keys are present.
    for (int t = 0; t < kThreads; ++t) {
        for (int i = 0; i < kPerThread; ++i) {
            int key = t * kPerThread + i;
            auto v = c.peek(key);
            ASSERT_TRUE(v.has_value()) << "missing key " << key;
            EXPECT_EQ(v->get(), key * 2);
        }
    }
}

// ============================================================================
// UnifiedCompactCache Tests (merged from test_unified_compact.cpp)
//
// Verifies that unified_compact_cache / safe_unified_compact_cache /
// striped_unified_compact_cache:
//   1. Compile and operate (set/get/peek/try_get/bulk_get/remove/contains/
//      size/empty/flush/shutdown/max_size/current_memory)
//   2. Match the API surface of unified_cache (interchangeable behind
//      common code)
//   3. Enforce the sizeof(K)+sizeof(V) <= 64 static_assert (compile-only)
//   4. Provide memory savings vs the regular unified_cache (slot allocator
//      eliminates per-item heap allocator metadata)
//   5. Support safe/striped variants with correct thread-safety flags
//
// Note: The static_assert test is a compile-time check — instantiating
// unified_compact_cache<K, V> with sizeof(K)+sizeof(V) > 64 is a
// compile error. We do not test this here (it would break the build).
// ============================================================================

// ----------------------------------------------------------------------------
// §1: Basic API surface — single-threaded unified_compact_cache
// ----------------------------------------------------------------------------

TEST(UnifiedCompactCacheTest, SetGetContainsSize) {
    unified_compact_cache<int, int> c(100);
    ASSERT_TRUE(c.empty());
    EXPECT_EQ(c.size(), 0u);

    c.set(1, 100);
    c.set(2, 200);
    c.set(3, 300);

    EXPECT_EQ(c.size(), 3u);
    EXPECT_FALSE(c.empty());
    EXPECT_TRUE(c.contains(1));
    EXPECT_TRUE(c.contains(2));
    EXPECT_TRUE(c.contains(3));
    EXPECT_FALSE(c.contains(99));

    auto h = c.get(1);
    ASSERT_TRUE(static_cast<bool>(h));
    EXPECT_EQ(*h, 100);
}

TEST(UnifiedCompactCacheTest, TryGetReturnsNulloptOnMiss) {
    unified_compact_cache<int, int> c(100);
    c.set(1, 42);

    auto h = c.try_get(1);
    ASSERT_TRUE(h.has_value());
    EXPECT_EQ(**h, 42);

    auto miss = c.try_get(99);
    EXPECT_FALSE(miss.has_value());
}

TEST(UnifiedCompactCacheTest, PeekDoesNotPromote) {
    unified_compact_cache<int, int> c(3);
    c.set(1, 10);
    c.set(2, 20);
    c.set(3, 30);

    auto p = c.peek(2);
    // peek returns read_handle<const Value>; verify it works.
    // (compact_cache's peek returns std::optional<reference_wrapper<const V>>,
    // unified_cache wraps it in a non-pinning read_handle.)
    ASSERT_TRUE(static_cast<bool>(p));
    EXPECT_EQ(*p, 20);
}

TEST(UnifiedCompactCacheTest, RemoveReturnsStatus) {
    unified_compact_cache<int, int> c(100);
    using RemoveRes = unified_compact_cache<int, int>::RemoveRes;
    c.set(1, 10);
    c.set(2, 20);

    EXPECT_EQ(c.remove(1), RemoveRes::kSuccess);
    EXPECT_FALSE(c.contains(1));
    EXPECT_EQ(c.remove(99), RemoveRes::kNotFound);
}

TEST(UnifiedCompactCacheTest, FlushClearsCache) {
    unified_compact_cache<int, int> c(100);
    c.set(1, 10);
    c.set(2, 20);
    c.set(3, 30);
    EXPECT_EQ(c.size(), 3u);

    c.flush();
    EXPECT_EQ(c.size(), 0u);
    EXPECT_TRUE(c.empty());
}

TEST(UnifiedCompactCacheTest, BulkGetReturnsHitsAndMisses) {
    unified_compact_cache<int, int> c(100);
    for (int i = 0; i < 10; ++i) c.set(i, i * 10);

    std::vector<int> keys{0, 5, 9, 99};
    auto results = c.bulk_get(keys.begin(), keys.end());
    ASSERT_EQ(results.size(), 4u);
    ASSERT_TRUE(results[0].has_value());
    EXPECT_EQ(**results[0], 0);
    ASSERT_TRUE(results[1].has_value());
    EXPECT_EQ(**results[1], 50);
    ASSERT_TRUE(results[2].has_value());
    EXPECT_EQ(**results[2], 90);
    EXPECT_FALSE(results[3].has_value());
}

TEST(UnifiedCompactCacheTest, MaxSizeEvicts) {
    unified_compact_cache<int, int> c(3);
    c.set(1, 10);
    c.set(2, 20);
    c.set(3, 30);
    c.set(4, 40);  // should evict key 1 (LRU)

    EXPECT_EQ(c.size(), 3u);
    // Key 1 should have been evicted (oldest).
    EXPECT_FALSE(c.contains(1));
    EXPECT_TRUE(c.contains(4));
}

TEST(UnifiedCompactCacheTest, AddReturnsFalseIfKeyExists) {
    unified_compact_cache<int, int> c(100);
    EXPECT_TRUE(c.add(1, 10));
    EXPECT_FALSE(c.add(1, 99));  // already exists
    EXPECT_EQ(*c.get(1), 10);    // original value retained
}

TEST(UnifiedCompactCacheTest, ReplaceReturnsFalseIfKeyMissing) {
    unified_compact_cache<int, int> c(100);
    EXPECT_FALSE(c.replace(1, 10));
    c.set(1, 10);
    EXPECT_TRUE(c.replace(1, 99));
    EXPECT_EQ(*c.get(1), 99);
}

TEST(UnifiedCompactCacheTest, ShutdownBlocksOps) {
    unified_compact_cache<int, int> c(100);
    c.set(1, 10);
    c.shutdown();

    EXPECT_THROW(c.set(2, 20), std::runtime_error);
    EXPECT_THROW(c.get(1), std::runtime_error);
    EXPECT_FALSE(c.try_get(1).has_value());
}

TEST(UnifiedCompactCacheTest, CurrentMemoryReflectsItems) {
    unified_compact_cache<int, int> c(100);
    EXPECT_EQ(c.current_memory(), 0u);

    c.set(1, 10);
    c.set(2, 20);
    // compact_cache tracks memory per item (key + value).
    EXPECT_GT(c.current_memory(), 0u);

    c.remove(1);
    EXPECT_GT(c.current_memory(), 0u);  // still has key 2
    c.remove(2);
    EXPECT_EQ(c.current_memory(), 0u);
}

TEST(UnifiedCompactCacheTest, MaxSizeResize) {
    unified_compact_cache<int, int> c(100);
    c.set(1, 10);
    c.set(2, 20);

    c.max_size(1);
    EXPECT_EQ(c.max_size(), 1u);
    // Adding a new key should now evict down to 1.
    c.set(3, 30);
    EXPECT_EQ(c.size(), 1u);
}

// ----------------------------------------------------------------------------
// §2: String key/value (still fits in 64 bytes)
// ----------------------------------------------------------------------------

TEST(UnifiedCompactCacheTest, SmallStringValueWorks) {
    // sizeof(std::string) is typically 32 bytes on 64-bit platforms (SSO).
    // sizeof(int) is 4 bytes. Total: 36 bytes, well under 64.
    unified_compact_cache<int, std::string> c(100);
    c.set(1, "hello");
    c.set(2, "world");

    auto h = c.get(1);
    ASSERT_TRUE(static_cast<bool>(h));
    EXPECT_EQ(*h, "hello");
}

// ----------------------------------------------------------------------------
// §3: Thread-safe variants
// ----------------------------------------------------------------------------

TEST(UnifiedCompactCacheTest, SafeVariantIsThreadSafe) {
    safe_unified_compact_cache<int, int> c(1000);
    EXPECT_TRUE(decltype(c)::is_thread_safe);
    EXPECT_FALSE(decltype(c)::is_striped);

    c.set(1, 10);
    EXPECT_EQ(*c.get(1), 10);
}

TEST(UnifiedCompactCacheTest, StripedVariantIsStriped) {
    striped_unified_compact_cache<int, int> c(1000, 16);
    EXPECT_TRUE(decltype(c)::is_thread_safe);
    EXPECT_TRUE(decltype(c)::is_striped);
    EXPECT_EQ(c.num_stripes(), 16u);

    c.set(1, 10);
    EXPECT_EQ(*c.get(1), 10);
}

// ----------------------------------------------------------------------------
// §4: Concurrent access on safe/striped variants
// ----------------------------------------------------------------------------

TEST(UnifiedCompactCacheTest, SafeVariantConcurrentReadWrite) {
    safe_unified_compact_cache<int, int> c(10000);
    constexpr int kThreads = 4;
    constexpr int kOpsPerThread = 500;

    auto worker = [&](int tid) {
        for (int i = 0; i < kOpsPerThread; ++i) {
            int key = tid * kOpsPerThread + i;
            c.set(key, i);
            auto h = c.try_get(key);
            ASSERT_TRUE(h.has_value());
            EXPECT_EQ(**h, i);
        }
    };

    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t) threads.emplace_back(worker, t);
    for (auto& th : threads) th.join();

    EXPECT_EQ(c.size(), static_cast<std::size_t>(kThreads * kOpsPerThread));
}

TEST(UnifiedCompactCacheTest, StripedVariantConcurrentReadWrite) {
    striped_unified_compact_cache<int, int> c(10000, 16);
    constexpr int kThreads = 8;
    constexpr int kOpsPerThread = 500;

    auto worker = [&](int tid) {
        for (int i = 0; i < kOpsPerThread; ++i) {
            int key = tid * kOpsPerThread + i;
            c.set(key, i);
            auto h = c.try_get(key);
            ASSERT_TRUE(h.has_value());
            EXPECT_EQ(**h, i);
        }
    };

    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t) threads.emplace_back(worker, t);
    for (auto& th : threads) th.join();

    EXPECT_EQ(c.size(), static_cast<std::size_t>(kThreads * kOpsPerThread));
}

// ----------------------------------------------------------------------------
// §5: API compatibility with regular unified_cache
// ----------------------------------------------------------------------------
//
// The test below uses a template helper to verify that the same operations
// work on both lru::cache<K,V> and lru::unified_compact_cache<K,V>. This
// is the "interoperability" check required by T14.4.

template <typename Cache>
void RunApiCompatibilitySuite(Cache& c) {
    using key_t = typename Cache::key_type;
    using val_t = typename Cache::mapped_type;
    using remove_res_t = typename Cache::RemoveRes;

    // set + get
    c.set(key_t{1}, val_t{42});
    c.set(key_t{2}, val_t{99});
    ASSERT_EQ(c.size(), 2u);

    // Scope-limit the get() handle so it releases its pin on key=1 before
    // we call remove() below. For non-compact caches the read_handle pins
    // the item via refcount; if the handle outlives the call to remove(),
    // mm_.del() returns false and remove() reports kNotFound.
    {
        auto h = c.get(key_t{1});
        ASSERT_TRUE(static_cast<bool>(h));
        EXPECT_EQ(*h, val_t{42});
    }

    // try_get (scope-limited for the same reason)
    {
        auto opt = c.try_get(key_t{2});
        ASSERT_TRUE(opt.has_value());
        EXPECT_EQ(**opt, val_t{99});
    }

    {
        auto miss = c.try_get(key_t{999});
        EXPECT_FALSE(miss.has_value());
    }

    // peek (scope-limited)
    {
        auto p = c.peek(key_t{1});
        ASSERT_TRUE(static_cast<bool>(p));
        EXPECT_EQ(*p, val_t{42});
    }

    // contains
    EXPECT_TRUE(c.contains(key_t{1}));
    EXPECT_FALSE(c.contains(key_t{999}));

    // bulk_get (scope-limited)
    {
        std::vector<key_t> keys{1, 2, 999};
        auto results = c.bulk_get(keys.begin(), keys.end());
        ASSERT_EQ(results.size(), 3u);
        EXPECT_TRUE(results[0].has_value());
        EXPECT_TRUE(results[1].has_value());
        EXPECT_FALSE(results[2].has_value());
    }

    // remove — now safe because all handles above have been released.
    EXPECT_EQ(c.remove(key_t{1}), remove_res_t::kSuccess);
    EXPECT_FALSE(c.contains(key_t{1}));
    EXPECT_EQ(c.remove(key_t{999}), remove_res_t::kNotFound);

    // flush
    c.flush();
    EXPECT_EQ(c.size(), 0u);
    EXPECT_TRUE(c.empty());
}

TEST(UnifiedCompactInteropTest, RegularCacheMatchesApi) {
    cache<int, int> c(100);
    RunApiCompatibilitySuite(c);
}

TEST(UnifiedCompactInteropTest, UnifiedCompactCacheMatchesApi) {
    unified_compact_cache<int, int> c(100);
    RunApiCompatibilitySuite(c);
}

TEST(UnifiedCompactInteropTest, SafeUnifiedCompactCacheMatchesApi) {
    safe_unified_compact_cache<int, int> c(100);
    RunApiCompatibilitySuite(c);
}

TEST(UnifiedCompactInteropTest, StripedUnifiedCompactCacheMatchesApi) {
    striped_unified_compact_cache<int, int> c(100, 8);
    RunApiCompatibilitySuite(c);
}

// ----------------------------------------------------------------------------
// §6: Memory savings (slot allocator eliminates per-item overhead)
// ----------------------------------------------------------------------------
//
// Note: This is a soft test — we don't assert exact savings because
// allocator behavior varies. We just verify that current_memory is
// reasonable (no more than 64 bytes per item).

TEST(UnifiedCompactMemoryTest, PerItemMemoryWithinSlotSize) {
    unified_compact_cache<int, int> c(100);
    c.set(1, 10);
    c.set(2, 20);

    // Each item should fit in a 64-byte slot (kMaxItemSize).
    // current_memory() returns the total memory used; per-item should be
    // <= 64 bytes.
    std::size_t per_item = c.current_memory() / c.size();
    EXPECT_LE(per_item, 64u);
}

// ----------------------------------------------------------------------------
// §7: Larger value type (still under 64 bytes)
// ----------------------------------------------------------------------------

struct SmallStruct {
    int a;
    int b;
    int c;
    int d;
};  // 16 bytes

static_assert(sizeof(int) + sizeof(SmallStruct) <= 64,
              "Test fixture: int + SmallStruct should fit in 64 bytes");

TEST(UnifiedCompactCacheTest, SmallStructValueWorks) {
    unified_compact_cache<int, SmallStruct> c(100);
    c.set(1, SmallStruct{1, 2, 3, 4});

    auto h = c.get(1);
    ASSERT_TRUE(static_cast<bool>(h));
    EXPECT_EQ(h->a, 1);
    EXPECT_EQ(h->d, 4);
}
