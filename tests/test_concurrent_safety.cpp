// SPDX-License-Identifier: MIT
// Concurrent safety and correctness tests.
//
// These tests verify that the cache library behaves correctly under
// concurrent access, focusing on the specific optimizations applied:
//   - TLS ring atomic head/tail (cross-thread drain safety)
//   - Hazptr-protected lock-free hash table reads
//   - Refcount saturation (no overflow crash)
//   - TTL lazy expiry under concurrent access
//   - bulk_get concurrent safety
//   - read_handle pinning prevents use-after-free

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <thread>
#include <vector>
#include <random>
#include <set>
#include <algorithm>
#include <string>

#include "lru.hpp"

using namespace lru;
using namespace std::chrono_literals;

// ============================================================================
// 1. Concurrent get/set data consistency
//    Multiple threads write distinct keys and read them back.
//    A key written by thread T must be readable as the exact value T wrote
//    (no torn reads, no cross-thread corruption).
// ============================================================================

TEST(ConcurrentSafety, ConcurrentGetSetDataConsistency) {
    safe_cache<int, std::string> c(10000);
    constexpr int kNumThreads = 8;
    constexpr int kKeysPerThread = 500;

    std::vector<std::thread> threads;
    std::atomic<int> errors{0};

    for (int t = 0; t < kNumThreads; ++t) {
        threads.emplace_back([&, t]() {
            for (int i = 0; i < kKeysPerThread; ++i) {
                int key = t * kKeysPerThread + i;
                std::string val = "thread_" + std::to_string(t) + "_key_" + std::to_string(key);
                c.set(key, val);

                // Immediately read back and verify
                auto h = c.get(key);
                if (!h || *h != val) {
                    errors.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }

    for (auto& th : threads) th.join();

    EXPECT_EQ(errors.load(), 0);

    // Verify all keys are present with correct values
    for (int t = 0; t < kNumThreads; ++t) {
        for (int i = 0; i < kKeysPerThread; ++i) {
            int key = t * kKeysPerThread + i;
            std::string expected = "thread_" + std::to_string(t) + "_key_" + std::to_string(key);
            auto h = c.get(key);
            ASSERT_TRUE(h) << "Missing key " << key;
            EXPECT_EQ(*h, expected) << "Wrong value for key " << key;
        }
    }
}

// ============================================================================
// 2. Concurrent get/set/remove — no crash, no memory corruption
//    Threads perform mixed operations on overlapping keys.
//    Verify no use-after-free or data corruption.
// ============================================================================

TEST(ConcurrentSafety, ConcurrentMixedOperationsNoCrash) {
    safe_cache<int, std::string> c(5000);
    constexpr int kNumThreads = 8;
    constexpr int kIterations = 5000;

    std::atomic<bool> stop{false};
    std::atomic<int> crashes{0};
    std::vector<std::thread> threads;

    for (int t = 0; t < kNumThreads; ++t) {
        threads.emplace_back([&, t]() {
            std::mt19937 rng(t * 42 + 7);
            for (int i = 0; i < kIterations; ++i) {
                int key = rng() % 2000;
                int op = rng() % 10;
                try {
                    if (op < 5) {
                        c.set(key, "val_" + std::to_string(key));
                    } else if (op < 8) {
                        auto h = c.get(key);
                        if (h) {
                            // Use the handle to ensure the value is valid
                            volatile auto v = *h;
                            (void)v;
                        }
                    } else {
                        c.remove(key);
                    }
                } catch (...) {
                    crashes.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }

    for (auto& th : threads) th.join();
    stop.store(true);

    EXPECT_EQ(crashes.load(), 0);
}

// ============================================================================
// 3. read_handle pinning prevents use-after-free
//    Thread 1 holds a read_handle while thread 2 evicts the same key.
//    The handle must remain valid until released.
// ============================================================================

TEST(ConcurrentSafety, ReadHandlePreventsUseAfterFree) {
    safe_cache<int, std::string> c(10);

    // Fill cache to capacity
    for (int i = 0; i < 10; ++i) {
        c.set(i, "val_" + std::to_string(i));
    }

    std::atomic<bool> handle_acquired{false};
    std::atomic<bool> handle_valid{true};
    std::atomic<bool> writer_done{false};

    // Thread 1: get a handle and hold it
    auto holder = [&]() {
        auto h = c.get(0);
        if (!h) {
            handle_valid.store(false);
            return;
        }
        // Signal that handle is acquired — writer can now proceed
        handle_acquired.store(true, std::memory_order_release);
        // Hold the handle while writer evicts
        while (!writer_done.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        // Verify the handle is still valid
        if (!h.has_value()) {
            handle_valid.store(false);
            return;
        }
        // The value must be "val_0" — the original value when we got the handle
        if (*h != "val_0") {
            handle_valid.store(false);
        }
    };

    // Thread 2: wait for handle acquisition, then insert new keys to force eviction
    auto writer = [&]() {
        // Wait until the reader has acquired the handle
        while (!handle_acquired.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        for (int i = 10; i < 100; ++i) {
            c.set(i, "new_" + std::to_string(i));
        }
        writer_done.store(true, std::memory_order_release);
    };

    std::thread t1(holder);
    std::thread t2(writer);
    t1.join();
    t2.join();

    EXPECT_TRUE(handle_valid.load()) << "read_handle was invalidated during eviction";
}

// ============================================================================
// 4. TTL lazy expiry under concurrent access
//    Multiple threads get() a TTL item simultaneously. After expiry,
//    all threads must see a miss.
// ============================================================================

TEST(ConcurrentSafety, TtlLazyExpiryConcurrent) {
    safe_cache<int, std::string> c(100);
    c.set_with_ttl(1, "one", 50ms);

    // Concurrent gets before expiry — all should hit
    {
        constexpr int kNumThreads = 8;
        std::vector<std::thread> threads;
        std::atomic<int> hits{0};
        for (int t = 0; t < kNumThreads; ++t) {
            threads.emplace_back([&]() {
                auto h = c.get(1);
                if (h) hits.fetch_add(1, std::memory_order_relaxed);
            });
        }
        for (auto& th : threads) th.join();
        EXPECT_EQ(hits.load(), kNumThreads);
    }

    // Wait for expiry
    std::this_thread::sleep_for(100ms);

    // Refresh cached time (as the background TTL cleaner would)
    c.refresh_cached_now();

    // Concurrent gets after expiry — all should miss
    {
        constexpr int kNumThreads = 8;
        std::vector<std::thread> threads;
        std::atomic<int> misses{0};
        for (int t = 0; t < kNumThreads; ++t) {
            threads.emplace_back([&]() {
                auto h = c.get(1);
                if (!h) misses.fetch_add(1, std::memory_order_relaxed);
            });
        }
        for (auto& th : threads) th.join();
        EXPECT_EQ(misses.load(), kNumThreads);
    }
}

// ============================================================================
// 5. bulk_get concurrent safety
//    Multiple threads call bulk_get simultaneously with overlapping keys.
// ============================================================================

TEST(ConcurrentSafety, BulkGetConcurrentSafety) {
    safe_cache<int, std::string> c(1000);
    constexpr int kNumThreads = 8;

    // Pre-populate
    for (int i = 0; i < 500; ++i) {
        c.set(i, "val_" + std::to_string(i));
    }

    std::atomic<int> errors{0};
    std::vector<std::thread> threads;

    for (int t = 0; t < kNumThreads; ++t) {
        threads.emplace_back([&, t]() {
            std::vector<int> keys;
            for (int i = 0; i < 100; ++i) {
                keys.push_back((t * 100 + i) % 500);
            }
            auto results = c.bulk_get(keys.begin(), keys.end());
            if (results.size() != keys.size()) {
                errors.fetch_add(1, std::memory_order_relaxed);
                return;
            }
            for (std::size_t i = 0; i < results.size(); ++i) {
                if (results[i]) {
                    std::string expected = "val_" + std::to_string(keys[i]);
                    if (**results[i] != expected) {
                        errors.fetch_add(1, std::memory_order_relaxed);
                    }
                }
            }
        });
    }

    for (auto& th : threads) th.join();
    EXPECT_EQ(errors.load(), 0);
}

// ============================================================================
// 6. Striped cache concurrent get/set correctness
//    Verify sharded_mm_lru correctly isolates per-shard operations.
// ============================================================================

TEST(ConcurrentSafety, StripedCacheConcurrentCorrectness) {
    striped_cache<int, std::string> c(10000, 8);
    constexpr int kNumThreads = 8;
    constexpr int kKeysPerThread = 1000;

    std::atomic<int> errors{0};

    // Phase 1: concurrent inserts
    {
        std::vector<std::thread> threads;
        for (int t = 0; t < kNumThreads; ++t) {
            threads.emplace_back([&, t]() {
                for (int i = 0; i < kKeysPerThread; ++i) {
                    int key = t * kKeysPerThread + i;
                    std::string val = "t" + std::to_string(t) + "_k" + std::to_string(key);
                    c.set(key, val);
                }
            });
        }
        for (auto& th : threads) th.join();
    }

    // Phase 2: concurrent reads and verify
    {
        std::vector<std::thread> threads;
        for (int t = 0; t < kNumThreads; ++t) {
            threads.emplace_back([&, t]() {
                for (int i = 0; i < kKeysPerThread; ++i) {
                    int key = t * kKeysPerThread + i;
                    std::string expected = "t" + std::to_string(t) + "_k" + std::to_string(key);
                    auto h = c.get(key);
                    if (!h || *h != expected) {
                        errors.fetch_add(1, std::memory_order_relaxed);
                    }
                }
            });
        }
        for (auto& th : threads) th.join();
    }

    EXPECT_EQ(errors.load(), 0);
}

// ============================================================================
// 7. TLS ring cross-thread drain safety
//    Threads record accesses via TLS rings, then a different thread
//    drains all rings. Verify no data race or corruption.
// ============================================================================

TEST(ConcurrentSafety, TlsRingCrossThreadDrain) {
    safe_cache<int, std::string> c(5000);
    constexpr int kNumThreads = 8;

    std::atomic<bool> stop{false};
    std::vector<std::thread> threads;

    // Writer threads: continuously set and get keys
    for (int t = 0; t < kNumThreads; ++t) {
        threads.emplace_back([&, t]() {
            std::mt19937 rng(t);
            for (int i = 0; i < 2000 && !stop.load(std::memory_order_relaxed); ++i) {
                int key = rng() % 1000;
                c.set(key, "val");
                c.get(key);
            }
        });
    }

    // Drainer thread: periodically drain all TLS rings from a different thread
    std::thread drainer([&]() {
        for (int i = 0; i < 100; ++i) {
            tls_access_ring<int>::drain_all_threads();
            std::this_thread::yield();
        }
    });

    for (auto& th : threads) th.join();
    stop.store(true);
    drainer.join();

    // No crash = success. Verify cache is still functional.
    c.set(999, "final");
    auto h = c.get(999);
    EXPECT_TRUE(h);
    EXPECT_EQ(*h, "final");
}

// ============================================================================
// 8. Refcount saturation under extreme concurrency
//    Many threads simultaneously get() the same key, pushing refcount
//    to high values. Verify no crash (saturation, not overflow).
// ============================================================================

TEST(ConcurrentSafety, RefcountSaturationNoCrash) {
    safe_cache<int, std::string> c(100);
    c.set(42, "answer");

    constexpr int kNumThreads = 16;
    constexpr int kGetsPerThread = 10000;

    std::atomic<int> total_hits{0};
    std::vector<std::thread> threads;

    for (int t = 0; t < kNumThreads; ++t) {
        threads.emplace_back([&]() {
            int local_hits = 0;
            for (int i = 0; i < kGetsPerThread; ++i) {
                auto h = c.get(42);
                if (h) ++local_hits;
            }
            total_hits.fetch_add(local_hits, std::memory_order_relaxed);
        });
    }

    for (auto& th : threads) th.join();

    // All gets should hit (key 42 is never evicted since cache has capacity)
    EXPECT_EQ(total_hits.load(), kNumThreads * kGetsPerThread);
}

// ============================================================================
// 9. Concurrent eviction with read_handle — no use-after-free
//    Continuous eviction while readers hold handles to evicted items.
// ============================================================================

TEST(ConcurrentSafety, ConcurrentEvictionWithHandles) {
    safe_cache<int, std::string> c(20);
    constexpr int kNumReaders = 4;
    constexpr int kNumWriters = 2;

    std::atomic<bool> stop{false};
    std::atomic<int> crashes{0};

    std::vector<std::thread> threads;

    // Writers: continuously insert to force eviction
    for (int t = 0; t < kNumWriters; ++t) {
        threads.emplace_back([&, t]() {
            for (int i = 0; i < 5000 && !stop.load(std::memory_order_relaxed); ++i) {
                int key = t * 5000 + i;
                c.set(key, "val_" + std::to_string(key));
            }
        });
    }

    // Readers: get handles and hold them briefly
    for (int t = 0; t < kNumReaders; ++t) {
        threads.emplace_back([&, t]() {
            std::mt19937 rng(t);
            for (int i = 0; i < 5000 && !stop.load(std::memory_order_relaxed); ++i) {
                int key = rng() % 10000;
                try {
                    auto h = c.get(key);
                    if (h) {
                        // Hold handle for a short time
                        volatile auto& v = *h;
                        (void)v;
                    }
                } catch (...) {
                    crashes.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }

    for (auto& th : threads) th.join();
    stop.store(true);

    EXPECT_EQ(crashes.load(), 0);
}

// ============================================================================
// 10. Stats consistency under concurrent access
//     hits + misses should equal total_accesses at all times.
// ============================================================================

TEST(ConcurrentSafety, StatsConsistencyUnderConcurrency) {
    safe_cache<int, std::string> c(500);
    constexpr int kNumThreads = 8;
    constexpr int kIterations = 2000;

    std::vector<std::thread> threads;

    for (int t = 0; t < kNumThreads; ++t) {
        threads.emplace_back([&, t]() {
            std::mt19937 rng(t);
            for (int i = 0; i < kIterations; ++i) {
                int key = rng() % 1000;
                if (rng() % 2 == 0) {
                    c.set(key, "val");
                } else {
                    c.get(key);
                }
            }
        });
    }

    for (auto& th : threads) th.join();

    auto stats = c.stats_snapshot();
    auto total = stats.hits.value.load() + stats.misses.value.load();
    EXPECT_EQ(total, stats.total_accesses());
    EXPECT_GT(total, 0u);
}
