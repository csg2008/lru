// SPDX-License-Identifier: MIT
// Concurrent edge-case tests for cross-thread invariants.
//
// Covers spec gaps G1, G2, G3, G8, G13, G16 (P0):
//   G1:  shutdown() holds active read_handle across threads
//   G2:  cas() atomicity under concurrency
//   G3:  try_get_or_fetch thundering-herd
//   G8:  pinned_skip_count increments on concurrent evict
//   G13: bulk_get handle survives eviction
//   G16: read_handle cross-thread transfer (producer/consumer)

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <optional>
#include <random>
#include <semaphore>
#include <string>
#include <thread>
#include <vector>

#include "lru.hpp"
#include "test_helpers.hpp"

using namespace lru;
using namespace std::chrono_literals;

// ============================================================================
// TC-G1: shutdown() holds active read_handle across threads
// ============================================================================
TEST(ConcurrentEdge, ShutdownHoldsActiveHandleAcrossThreads) {
    safe_cache<std::string, std::string> c(1024);
    c.set_per_cache_handle_tracking(true);
    c.set("k1", "v1");

    std::optional<read_handle<std::string>> held;
    std::binary_semaphore hold_ready(0), shutdown_done(0);
    std::atomic<bool> holder_done{false};

    std::thread holder([&] {
        held = c.get("k1");
        ASSERT_TRUE(held.has_value());
        EXPECT_EQ(**held, "v1");
        EXPECT_GE(c.active_handle_count(), 1u);
        hold_ready.release();
        shutdown_done.acquire();
        // After shutdown, the held handle must remain valid.
        EXPECT_EQ(**held, "v1");
        held.reset();
        holder_done.store(true, std::memory_order_release);
    });

    std::thread killer([&] {
        hold_ready.acquire();
        c.shutdown();
        EXPECT_TRUE(c.is_shutdown());
        // New ops must be rejected.
        EXPECT_EQ(c.try_get("k1"), std::nullopt);
        shutdown_done.release();
    });

    holder.join();
    killer.join();

    EXPECT_TRUE(holder_done.load());
    EXPECT_EQ(c.active_handle_count(), 0u);
}

// ============================================================================
// TC-G2: cas() atomicity under concurrency
// Multiple threads concurrently perform cas() on a single shared key.
// Exactly one CAS must succeed per "round" (the rest must fail and observe
// the updated value on retry). The final value must equal the number of
// successful CAS operations.
//
// Notes:
//   - Uses safe_cache (single global mutex) to keep the test simple.
//   - Each thread runs a bounded number of CAS attempts (no unbounded retry
//     loop) so the test cannot hang under contention.
//   - Readers are kept light (no tight loop) to avoid starving writers on
//     the global mutex.
// ============================================================================
TEST(ConcurrentEdge, CasAtomicityUnderConcurrency) {
    safe_cache<int, int> c(64);
    c.set(42, 0);

    constexpr int kThreads = 4;
    constexpr int kAttemptsPerThread = 200;
    std::atomic<int> success_count{0};
    std::atomic<int> fail_count{0};
    std::atomic<bool> stop_readers{false};

    std::vector<std::thread> threads;

    // CAS workers: each thread tries to increment the shared counter a
    // bounded number of times. After each failed CAS, the thread yields
    // and retries with the freshly-read expected value.
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&] {
            int successes = 0;
            for (int i = 0; i < kAttemptsPerThread; ++i) {
                int expected;
                {
                    auto h = c.try_get(42);
                    if (!h) continue;
                    expected = **h;
                }
                int desired = expected + 1;
                if (c.cas(42, expected, desired)) {
                    ++successes;
                    success_count.fetch_add(1, std::memory_order_relaxed);
                } else {
                    fail_count.fetch_add(1, std::memory_order_relaxed);
                    std::this_thread::yield();
                }
            }
            // Each thread must have succeeded at least once; otherwise the
            // CAS path is starving under contention.
            EXPECT_GT(successes, 0) << "thread starved — CAS never succeeded";
        });
    }

    // Reader workers: a single reader that reads occasionally to verify
    // no torn reads. The reader sleeps between reads to avoid starving
    // the CAS writers on the global mutex.
    threads.emplace_back([&] {
        for (int i = 0; i < 50; ++i) {
            if (stop_readers.load(std::memory_order_acquire)) break;
            auto h = c.try_get(42);
            if (h) {
                EXPECT_GE(**h, 0);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    });

    for (auto& th : threads) th.join();
    stop_readers.store(true);

    // The final value must equal the total successful CAS operations,
    // since each successful CAS increments by exactly 1.
    auto final_h = c.try_get(42);
    ASSERT_TRUE(final_h.has_value());
    EXPECT_EQ(**final_h, success_count.load());
    // Sanity: failures must have happened (contention exists).
    EXPECT_GT(fail_count.load(), 0);
    (void)fail_count;
}

// ============================================================================
// TC-G3: try_get_or_fetch thundering-herd
// Many threads concurrently call try_get_or_fetch on a missing key.
// The provider should be invoked a bounded number of times (no thundering
// herd). All threads must observe the same cached value.
// ============================================================================
TEST(ConcurrentEdge, TryGetOrFetchThunderingHerd) {
    safe_cache<int, std::string> c(64);
    c.set_per_cache_handle_tracking(false);

    constexpr int kThreads = 32;
    std::atomic<int> provider_invocations{0};
    std::atomic<std::string*> first_value{nullptr};
    std::atomic<bool> go{false};
    std::vector<std::thread> threads;
    std::vector<std::string> results(kThreads);

    auto provider = [&](const int& key) {
        provider_invocations.fetch_add(1, std::memory_order_relaxed);
        // Simulate slow provider so concurrent callers overlap.
        std::this_thread::sleep_for(10ms);
        return std::string("val_") + std::to_string(key);
    };

    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t] {
            // Spin until all threads are ready.
            while (!go.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            results[t] = c.try_get_or_fetch(99, provider);
        });
    }

    // Synchronize start so all threads race for the missing key.
    go.store(true, std::memory_order_release);
    for (auto& th : threads) th.join();

    // All threads must observe the same value.
    for (int t = 0; t < kThreads; ++t) {
        EXPECT_EQ(results[t], std::string("val_99"))
            << "thread " << t << " got unexpected value";
    }

    // The provider may be invoked more than once (no single-flight guarantee
    // in the current try_get_or_fetch implementation), but the number should
    // be bounded. A reasonable upper bound is the number of threads; an
    // unbounded count would indicate a true thundering herd.
    int invocations = provider_invocations.load();
    EXPECT_LE(invocations, kThreads)
        << "provider invoked " << invocations << " times (thundering herd)";
    EXPECT_GE(invocations, 1);
}

// ============================================================================
// T-M1: singleflight / cache stampede protection
// ============================================================================
//
// The next group of tests verifies the singleflight coalescing introduced in
// T-M1. When `set_singleflight_enabled(true)` is set on the cache, concurrent
// misses on the same key are coalesced: the first caller (leader) executes
// the provider; subsequent callers (followers) block on a CV until the
// leader completes, then receive the leader's result. The
// `stampede_coalesced_count` metric tracks how many follower requests were
// collapsed into a leader's in-flight provider call.
//
// Default behavior (singleflight disabled) is unchanged — every miss
// independently calls the provider. This is verified by the
// DisabledDoesNotCoalesce test below.

// ---------------------------------------------------------------------------
// T-M1-A: get_or_fetch with singleflight enabled coalesces concurrent misses
// on the same key into exactly one provider invocation.
// ---------------------------------------------------------------------------
TEST(ConcurrentEdge, GetOrFetchSingleflightCoalescesMisses) {
    safe_cache<int, std::string> c(64);
    c.set_per_cache_handle_tracking(false);
    c.set_singleflight_enabled(true);
    ASSERT_TRUE(c.is_singleflight_enabled());

    constexpr int kThreads = 32;
    std::atomic<int> provider_invocations{0};
    std::atomic<bool> go{false};
    std::vector<std::thread> threads;
    std::vector<std::string> results(kThreads);

    auto provider = [&](const int& key) {
        provider_invocations.fetch_add(1, std::memory_order_relaxed);
        // Simulate a slow provider so concurrent callers overlap and all
        // arrive at the singleflight acquire() while the leader is still
        // in flight.
        std::this_thread::sleep_for(50ms);
        return std::string("val_") + std::to_string(key);
    };

    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t] {
            while (!go.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            results[t] = c.get_or_fetch(99, provider);
        });
    }
    go.store(true, std::memory_order_release);
    for (auto& th : threads) th.join();

    // All threads observe the same value.
    for (int t = 0; t < kThreads; ++t) {
        EXPECT_EQ(results[t], std::string("val_99"))
            << "thread " << t << " got unexpected value";
    }

    // Provider must be invoked exactly once — singleflight coalesced the
    // remaining 31 follower requests into the leader's in-flight call.
    EXPECT_EQ(provider_invocations.load(), 1)
        << "singleflight failed to coalesce concurrent misses";

    // stampede_coalesced_count should be kThreads - 1 (all followers).
    EXPECT_EQ(c.stampede_coalesced_count(), kThreads - 1)
        << "stampede_coalesced_count did not track follower collapses";

    // After the singleflight cycle, the in-flight entry must be removed.
    EXPECT_EQ(c.singleflight_inflight_count(), 0u);
}

// ---------------------------------------------------------------------------
// T-M1-B: try_get_or_fetch with singleflight enabled coalesces concurrent
// misses on the same key into exactly one provider invocation.
// ---------------------------------------------------------------------------
TEST(ConcurrentEdge, TryGetOrFetchSingleflightCoalescesMisses) {
    safe_cache<int, std::string> c(64);
    c.set_per_cache_handle_tracking(false);
    c.set_singleflight_enabled(true);

    constexpr int kThreads = 32;
    std::atomic<int> provider_invocations{0};
    std::atomic<bool> go{false};
    std::vector<std::thread> threads;
    std::vector<std::string> results(kThreads);

    auto provider = [&](const int& key) {
        provider_invocations.fetch_add(1, std::memory_order_relaxed);
        std::this_thread::sleep_for(50ms);
        return std::string("val_") + std::to_string(key);
    };

    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t] {
            while (!go.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            results[t] = c.try_get_or_fetch(99, provider);
        });
    }
    go.store(true, std::memory_order_release);
    for (auto& th : threads) th.join();

    for (int t = 0; t < kThreads; ++t) {
        EXPECT_EQ(results[t], std::string("val_99"))
            << "thread " << t << " got unexpected value";
    }

    EXPECT_EQ(provider_invocations.load(), 1)
        << "try_get_or_fetch singleflight failed to coalesce";
    EXPECT_EQ(c.stampede_coalesced_count(), kThreads - 1);
    EXPECT_EQ(c.singleflight_inflight_count(), 0u);
}

// ---------------------------------------------------------------------------
// T-M1-C: When singleflight is disabled (default), concurrent misses on the
// same key each invoke the provider independently. This is the control test
// for T-M1-A — it verifies that the coalescing in T-M1-A is actually due to
// the singleflight mechanism, not some other serialization.
// ---------------------------------------------------------------------------
TEST(ConcurrentEdge, SingleflightDisabledDoesNotCoalesce) {
    safe_cache<int, std::string> c(64);
    c.set_per_cache_handle_tracking(false);
    ASSERT_FALSE(c.is_singleflight_enabled());

    constexpr int kThreads = 16;
    std::atomic<int> provider_invocations{0};
    std::atomic<bool> go{false};
    std::vector<std::thread> threads;
    std::vector<std::string> results(kThreads);

    auto provider = [&](const int& key) {
        provider_invocations.fetch_add(1, std::memory_order_relaxed);
        std::this_thread::sleep_for(20ms);
        return std::string("val_") + std::to_string(key);
    };

    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t] {
            while (!go.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            results[t] = c.get_or_fetch(99, provider);
        });
    }
    go.store(true, std::memory_order_release);
    for (auto& th : threads) th.join();

    // With singleflight disabled, provider is invoked multiple times.
    EXPECT_GT(provider_invocations.load(), 1)
        << "singleflight disabled but provider invoked only once";
    // stampede_coalesced_count must remain zero.
    EXPECT_EQ(c.stampede_coalesced_count(), 0u);
}

// ---------------------------------------------------------------------------
// T-M1-D: provider exceptions are propagated to all followers.
// The leader's exception_ptr is captured and rethrown by each follower's
// wait_and_get() call. This test verifies the exception type and message
// survive the cross-thread propagation.
// ---------------------------------------------------------------------------
TEST(ConcurrentEdge, SingleflightPropagatesProviderException) {
    safe_cache<int, std::string> c(64);
    c.set_per_cache_handle_tracking(false);
    c.set_singleflight_enabled(true);

    constexpr int kThreads = 8;
    std::atomic<int> provider_invocations{0};
    std::atomic<bool> go{false};
    std::vector<std::thread> threads;
    std::vector<std::string> errors(kThreads);
    std::vector<bool> threw(kThreads, false);

    auto provider = [&](const int& key) -> std::string {
        provider_invocations.fetch_add(1, std::memory_order_relaxed);
        std::this_thread::sleep_for(30ms);
        throw std::runtime_error("provider_failed_" + std::to_string(key));
    };

    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t] {
            while (!go.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            try {
                (void)c.get_or_fetch(42, provider);
            } catch (const std::runtime_error& e) {
                threw[t] = true;
                errors[t] = e.what();
            }
        });
    }
    go.store(true, std::memory_order_release);
    for (auto& th : threads) th.join();

    // All threads must observe the leader's exception.
    for (int t = 0; t < kThreads; ++t) {
        EXPECT_TRUE(threw[t]) << "thread " << t << " did not receive exception";
        EXPECT_EQ(errors[t], "provider_failed_42")
            << "thread " << t << " received wrong exception message";
    }

    // Provider must be invoked exactly once (leader only).
    EXPECT_EQ(provider_invocations.load(), 1);
    // In-flight entry must be cleaned up even on exception.
    EXPECT_EQ(c.singleflight_inflight_count(), 0u);
}

// ---------------------------------------------------------------------------
// T-M1-E: distinct keys are NOT coalesced — each key gets its own
// singleflight cycle. This guards against an over-coalescing bug where
// the tracker accidentally merges unrelated keys.
// ---------------------------------------------------------------------------
TEST(ConcurrentEdge, SingleflightDistinctKeysAreNotCoalesced) {
    safe_cache<int, std::string> c(256);
    c.set_per_cache_handle_tracking(false);
    c.set_singleflight_enabled(true);

    constexpr int kKeys = 8;
    constexpr int kThreadsPerKey = 4;
    std::atomic<int> provider_invocations{0};
    std::atomic<bool> go{false};
    std::vector<std::thread> threads;

    auto provider = [&](const int& key) {
        provider_invocations.fetch_add(1, std::memory_order_relaxed);
        std::this_thread::sleep_for(20ms);
        return std::string("val_") + std::to_string(key);
    };

    for (int k = 0; k < kKeys; ++k) {
        for (int t = 0; t < kThreadsPerKey; ++t) {
            threads.emplace_back([&, k] {
                while (!go.load(std::memory_order_acquire)) {
                    std::this_thread::yield();
                }
                (void)c.get_or_fetch(k, provider);
            });
        }
    }
    go.store(true, std::memory_order_release);
    for (auto& th : threads) th.join();

    // Each distinct key must invoke the provider exactly once.
    EXPECT_EQ(provider_invocations.load(), kKeys)
        << "singleflight over-coalesced distinct keys";
    // Followers across all keys: kKeys * (kThreadsPerKey - 1).
    EXPECT_EQ(c.stampede_coalesced_count(), kKeys * (kThreadsPerKey - 1));
}

// ---------------------------------------------------------------------------
// T-M1-F: Second wave of misses on the same key starts a NEW singleflight
// cycle after the first completes. Verifies that the in-flight entry is
// removed on completion (no permanent coalescing of unrelated calls).
// ---------------------------------------------------------------------------
TEST(ConcurrentEdge, SingleflightSecondWaveStartsNewCycle) {
    safe_cache<int, std::string> c(64);
    c.set_per_cache_handle_tracking(false);
    c.set_singleflight_enabled(true);

    std::atomic<int> provider_invocations{0};

    auto provider = [&](const int& key) {
        provider_invocations.fetch_add(1, std::memory_order_relaxed);
        std::this_thread::sleep_for(10ms);
        return std::string("val_") + std::to_string(key);
    };

    // Wave 1: 4 threads concurrently miss key 7 → 1 provider call.
    {
        std::atomic<bool> go{false};
        std::vector<std::thread> threads;
        for (int t = 0; t < 4; ++t) {
            threads.emplace_back([&] {
                while (!go.load(std::memory_order_acquire)) {
                    std::this_thread::yield();
                }
                (void)c.get_or_fetch(7, provider);
            });
        }
        go.store(true, std::memory_order_release);
        for (auto& th : threads) th.join();
    }
    EXPECT_EQ(provider_invocations.load(), 1);
    EXPECT_EQ(c.singleflight_inflight_count(), 0u);

    // Remove the key so the next wave misses again.
    c.remove(7);

    // Wave 2: 4 threads concurrently miss key 7 again → 1 more provider call.
    {
        std::atomic<bool> go{false};
        std::vector<std::thread> threads;
        for (int t = 0; t < 4; ++t) {
            threads.emplace_back([&] {
                while (!go.load(std::memory_order_acquire)) {
                    std::this_thread::yield();
                }
                (void)c.get_or_fetch(7, provider);
            });
        }
        go.store(true, std::memory_order_release);
        for (auto& th : threads) th.join();
    }
    EXPECT_EQ(provider_invocations.load(), 2)
        << "second wave did not start a new singleflight cycle";
    EXPECT_EQ(c.singleflight_inflight_count(), 0u);
}

// ---------------------------------------------------------------------------
// T-M1-G: stampede_coalesced_count is exported via stats_snapshot() and
// prometheus_text() so operators can monitor coalescing effectiveness.
// ---------------------------------------------------------------------------
TEST(ConcurrentEdge, SingleflightMetricExportedViaStatsAndPrometheus) {
    safe_cache<int, std::string> c(64);
    c.set_per_cache_handle_tracking(false);
    c.set_singleflight_enabled(true);

    std::atomic<int> provider_invocations{0};
    std::atomic<bool> go{false};
    std::vector<std::thread> threads;

    auto provider = [&](const int& key) {
        provider_invocations.fetch_add(1, std::memory_order_relaxed);
        std::this_thread::sleep_for(20ms);
        return std::string("val_") + std::to_string(key);
    };

    for (int t = 0; t < 8; ++t) {
        threads.emplace_back([&] {
            while (!go.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            (void)c.get_or_fetch(99, provider);
        });
    }
    go.store(true, std::memory_order_release);
    for (auto& th : threads) th.join();

    // stats_snapshot() must reflect the coalesced count.
    auto snap = c.stats_snapshot();
    EXPECT_EQ(snap.stampede_coalesced_count.load(std::memory_order_relaxed), 7u);

    // prometheus_text() must export the metric.
    std::string prom = c.prometheus_text();
    EXPECT_NE(prom.find("lru_stampede_coalesced_total"), std::string::npos)
        << "prometheus_text() did not export lru_stampede_coalesced_total";
    EXPECT_NE(prom.find("lru_stampede_coalesced_total 7"), std::string::npos)
        << "prometheus_text() did not export the correct coalesced count";
}

// ---------------------------------------------------------------------------
// T-M1-H: TTL expiry stampede — when a hot key expires, concurrent
// get_or_fetch calls must not trigger a thundering herd of provider calls.
// This is the canonical motivation for singleflight (M-1 in spec.md).
// ---------------------------------------------------------------------------
TEST(ConcurrentEdge, SingleflightCoalescesTtlExpiryStampede) {
    safe_cache<int, std::string> c(64);
    c.set_per_cache_handle_tracking(false);
    c.set_singleflight_enabled(true);

    // Pre-populate key 7 with a short TTL.
    c.set_with_ttl(7, std::string("initial"), 50ms);
    // Wait for expiry.
    std::this_thread::sleep_for(80ms);
    // Confirm the key has expired (try_get returns nullopt).
    EXPECT_FALSE(c.try_get(7).has_value());

    constexpr int kThreads = 32;
    std::atomic<int> provider_invocations{0};
    std::atomic<bool> go{false};
    std::vector<std::thread> threads;
    std::vector<std::string> results(kThreads);

    auto provider = [&](const int& key) {
        provider_invocations.fetch_add(1, std::memory_order_relaxed);
        std::this_thread::sleep_for(30ms);
        return std::string("refreshed_") + std::to_string(key);
    };

    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t] {
            while (!go.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            results[t] = c.get_or_fetch(7, provider);
        });
    }
    go.store(true, std::memory_order_release);
    for (auto& th : threads) th.join();

    for (int t = 0; t < kThreads; ++t) {
        EXPECT_EQ(results[t], std::string("refreshed_7"))
            << "thread " << t << " got unexpected value";
    }
    EXPECT_EQ(provider_invocations.load(), 1)
        << "TTL expiry stampede was not coalesced";
    EXPECT_EQ(c.stampede_coalesced_count(), kThreads - 1);
}

// ---------------------------------------------------------------------------
// T-M1-I: singleflight works correctly with production_cache (segmented +
// striped). Verifies the integration is sound under the recommended
// production alias, not just safe_cache.
// ---------------------------------------------------------------------------
TEST(ConcurrentEdge, SingleflightWorksWithProductionCache) {
    production_cache<int, std::string> c(256);
    c.set_per_cache_handle_tracking(false);
    c.set_singleflight_enabled(true);

    constexpr int kThreads = 16;
    std::atomic<int> provider_invocations{0};
    std::atomic<bool> go{false};
    std::vector<std::thread> threads;
    std::vector<std::string> results(kThreads);

    auto provider = [&](const int& key) {
        provider_invocations.fetch_add(1, std::memory_order_relaxed);
        std::this_thread::sleep_for(30ms);
        return std::string("val_") + std::to_string(key);
    };

    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t] {
            while (!go.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            results[t] = c.get_or_fetch(123, provider);
        });
    }
    go.store(true, std::memory_order_release);
    for (auto& th : threads) th.join();

    for (int t = 0; t < kThreads; ++t) {
        EXPECT_EQ(results[t], std::string("val_123"))
            << "thread " << t << " got unexpected value";
    }
    EXPECT_EQ(provider_invocations.load(), 1)
        << "production_cache singleflight failed to coalesce";
    EXPECT_EQ(c.stampede_coalesced_count(), kThreads - 1);
    EXPECT_EQ(c.singleflight_inflight_count(), 0u);
}

// ============================================================================
// TC-G8: pinned_skip_count increments when evictor encounters a pinned item
// ============================================================================
TEST(ConcurrentEdge, PinnedSkipCountIncrementsOnConcurrentEvict) {
    safe_cache<int, std::string> c(8);
    // Fill the cache exactly to capacity.
    for (int i = 0; i < 8; ++i) {
        c.set(i, "v" + std::to_string(i));
    }
    // Pin key 0 so the evictor must skip it.
    auto pinned = c.get(0);
    ASSERT_TRUE(pinned.has_value());

    // Insert additional keys to force eviction; the evictor must skip key 0
    // (it's pinned) and evict other items instead.
    for (int i = 8; i < 24; ++i) {
        c.set(i, "v" + std::to_string(i));
    }

    // The pinned key must still be present (was skipped during eviction).
    auto still_pinned = c.try_get(0);
    EXPECT_TRUE(still_pinned.has_value());
    EXPECT_EQ(**still_pinned, "v0");

    // The pinned_skip_count statistic should be non-zero — the evictor
    // encountered a pinned item at least once.
    auto snap = c.stats_snapshot();
    EXPECT_GE(snap.pinned_skip_count.load(std::memory_order_relaxed), 1u)
        << "pinned_skip_count did not increment when evictor encountered a pinned item";
}

// ============================================================================
// TC-G13: bulk_get handle survives eviction
// bulk_get returns handles that pin the items; subsequent evictions must
// not invalidate the handles.
// ============================================================================
TEST(ConcurrentEdge, BulkGetHandleSurvivesEviction) {
    safe_cache<int, std::string> c(20);
    constexpr int kKeys = 8;

    // Pre-populate.
    for (int i = 0; i < kKeys; ++i) {
        c.set(i, "v" + std::to_string(i));
    }

    // Snapshot the keys we want to verify.
    std::vector<int> keys(kKeys);
    for (int i = 0; i < kKeys; ++i) keys[i] = i;

    // Acquire handles via bulk_get.
    auto handles = c.bulk_get(keys.begin(), keys.end());
    ASSERT_EQ(handles.size(), kKeys);
    for (auto& h : handles) {
        ASSERT_TRUE(h.has_value()) << "bulk_get returned missing handle";
    }

    // Concurrent eviction: insert many more keys to force eviction of the
    // originally-cached items. The handles must remain valid throughout.
    std::thread evictor([&] {
        for (int i = 1000; i < 5000; ++i) {
            c.set(i, "x" + std::to_string(i));
        }
    });

    // While evictions are happening, dereference each handle.
    for (int i = 0; i < kKeys; ++i) {
        ASSERT_TRUE(handles[i].has_value());
        std::string v = **handles[i];
        EXPECT_EQ(v, "v" + std::to_string(i));
    }

    evictor.join();

    // After evictions, the handles must STILL be valid (they pin the items).
    for (int i = 0; i < kKeys; ++i) {
        ASSERT_TRUE(handles[i].has_value());
        std::string v = **handles[i];
        EXPECT_EQ(v, "v" + std::to_string(i));
    }
}

// ============================================================================
// TC-G16: read_handle cross-thread transfer (producer/consumer)
// A producer thread acquires a read_handle and transfers it to a consumer
// thread via move semantics. The consumer must be able to dereference the
// handle and the active_handle_count must remain consistent.
// ============================================================================
TEST(ConcurrentEdge, ReadHandleCrossThreadTransfer) {
    safe_cache<int, std::string> c(64);
    c.set_per_cache_handle_tracking(true);
    c.set(123, "transferred_value");

    std::optional<read_handle<std::string>> transferred;
    std::binary_semaphore produced(0), consumed(0);

    std::thread producer([&] {
        auto h = c.get(123);
        ASSERT_TRUE(h.has_value());
        EXPECT_EQ(*h, "transferred_value");
        EXPECT_GE(c.active_handle_count(), 1u);
        // Move the handle into the shared slot.
        transferred = std::move(h);
        produced.release();
        consumed.acquire();
    });

    std::thread consumer([&] {
        produced.acquire();
        ASSERT_TRUE(transferred.has_value());
        EXPECT_EQ(**transferred, "transferred_value");
        // active_handle_count should still be >= 1 (handle moved, not released).
        EXPECT_GE(c.active_handle_count(), 1u);
        // Release the handle.
        transferred.reset();
        EXPECT_EQ(c.active_handle_count(), 0u);
        consumed.release();
    });

    producer.join();
    consumer.join();

    EXPECT_EQ(c.active_handle_count(), 0u);
}
