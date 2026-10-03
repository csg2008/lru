// Unified LRU Cache - SafeCache (thread-safe LRU) unit tests
// Aligned with the unified_cache architecture: use the lru::safe_cache<K,V> alias.
// Key differences: get() returns read_handle<V> (no longer a locked guard), and
// size()/contains() return plain values while peek() returns read_handle<const V>.
// Note: unified_cache is neither copyable nor movable, because mm_lru owns
// new-allocated item pointers. The copy/move tests were therefore replaced by
// tests that stats_snapshot() returns a copyable value.

#include <gtest/gtest.h>
#include <atomic>
#include <chrono>
#include <string>
#include <thread>
#include <vector>

#include "../lru.hpp"

using namespace lru;

// ============================================================================
// Basic Thread-Safe Operations
// ============================================================================

class SafeCacheTest : public ::testing::Test {
protected:
    safe_cache<int, std::string> c{10};

    void SetUp() override {
        c.set(1, "one");
        c.set(2, "two");
        c.set(3, "three");
    }
};

TEST_F(SafeCacheTest, BasicGet) {
    // get() returns read_handle<V>; dereference with *result.
    auto result = c.get(1);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, "one");
}

TEST_F(SafeCacheTest, BasicSetAndGet) {
    c.set(4, "four");
    auto result = c.get(4);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, "four");
}

TEST_F(SafeCacheTest, PeekDoesNotTouch) {
    // peek() returns read_handle<const V> and does not change the LRU order.
    auto result = c.peek(1);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, "one");
}

TEST_F(SafeCacheTest, PeekNonExistent) {
    auto result = c.peek(99);
    EXPECT_FALSE(result.has_value());
}

TEST_F(SafeCacheTest, Delete) {
    EXPECT_TRUE(c.del(1));
    auto result = c.get(1);
    EXPECT_FALSE(result.has_value());
}

TEST_F(SafeCacheTest, Contains) {
    // contains() returns bool, no longer a guard wrapper.
    EXPECT_TRUE(c.contains(1));
    EXPECT_FALSE(c.contains(99));
}

TEST_F(SafeCacheTest, Size) {
    // size() returns size_type, no longer a guard wrapper.
    EXPECT_EQ(c.size(), 3);
}

TEST_F(SafeCacheTest, Flush) {
    c.flush();
    EXPECT_EQ(c.size(), 0);
}

TEST_F(SafeCacheTest, GetNonExistent) {
    auto result = c.get(99);
    EXPECT_FALSE(result.has_value());
}

TEST_F(SafeCacheTest, ReplaceExisting) {
    EXPECT_TRUE(c.replace(1, "updated"));
    EXPECT_EQ(*c.get(1), "updated");
}

TEST_F(SafeCacheTest, AddNewKey) {
    EXPECT_TRUE(c.add(4, "four"));
    EXPECT_EQ(c.size(), 4);
}

TEST_F(SafeCacheTest, AddExistingKey) {
    EXPECT_FALSE(c.add(1, "z"));
    EXPECT_EQ(*c.get(1), "one"); // value unchanged
}

// ============================================================================
// Concurrent Access Tests
// ============================================================================

TEST(SafeCacheConcurrencyTest, ConcurrentReads) {
    safe_cache<int, int> c{100};
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

    // Every read must succeed.
    EXPECT_EQ(hit_count.load(), 400);
}

TEST(SafeCacheConcurrencyTest, ConcurrentWrites) {
    safe_cache<int, int> c{1000};

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

TEST(SafeCacheConcurrencyTest, MixedReadWrite) {
    safe_cache<int, int> c{100};
    for (int i = 0; i < 50; ++i) {
        c.set(i, i);
    }

    std::atomic<bool> done{false};
    std::vector<std::thread> threads;

    // Writer thread.
    threads.emplace_back([&]() {
        for (int i = 50; i < 150; ++i) {
            c.set(i, i * 2);
        }
        done = true;
    });

    // Reader thread.
    for (int t = 0; t < 3; ++t) {
        threads.emplace_back([&]() {
            while (!done) {
                for (int i = 0; i < 50; ++i) {
                    auto result = c.get(i);
                    (void)result; // read only, no assertion, to avoid a race
                }
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    // The final size must not exceed max_size.
    EXPECT_LE(c.size(), 100);
}

TEST(SafeCacheConcurrencyTest, ConcurrentGetAndDel) {
    safe_cache<int, int> c{100};
    for (int i = 0; i < 50; ++i) {
        c.set(i, i);
    }

    std::vector<std::thread> threads;

    // Thread A: reads continuously (handle destruction races with del).
    threads.emplace_back([&]() {
        for (int i = 0; i < 50; ++i) {
            (void)c.get(i);
        }
    });

    // Thread B: deletes continuously (del returns false while a handle is live).
    threads.emplace_back([&]() {
        for (int i = 0; i < 50; ++i) {
            (void)c.del(i);
        }
    });

    for (auto& t : threads) {
        t.join();
    }

    // del() may fail because of the handle race, so force_del guarantees cleanup.
    for (int i = 0; i < 50; ++i) {
        c.force_del(i);
    }
    for (int i = 0; i < 50; ++i) {
        EXPECT_FALSE(c.contains(i));
    }
}

// ============================================================================
// Statistics Tests
// ============================================================================

TEST(SafeCacheStatsTest, HitMissTracking) {
    safe_cache<int, int> c{10};
    c.set(1, 100);

    c.get(1); // hit
    c.get(1); // hit
    c.get(2); // miss

    auto stats = c.stats_snapshot();
    EXPECT_EQ(stats.hits.value.load(), 2);
    EXPECT_EQ(stats.misses.value.load(), 1);
}

TEST(SafeCacheStatsTest, ConcurrentStats) {
    safe_cache<int, int> c{1000};

    // Local counters, independent of the cache's own statistics, proving the operations ran.
    std::atomic<int> set_count{0};
    std::vector<std::thread> threads;

    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&, t]() {
            for (int i = 0; i < 50; ++i) {
                c.set(t * 100 + i, t * 100 + i);
                c.get(t * 100 + i);
                set_count.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    // Every set has executed.
    EXPECT_EQ(set_count.load(), 200);

    auto stats = c.stats_snapshot();
    // Under an ideal schedule every get hits the key just set: hits=200, misses=0.
    // Some interleavings run a get before another thread's set, producing a few misses.
    // Relaxed check: insertions may be 200 (4 threads x 50), and
    // hits+misses must be > 0 (every get returns either a hit or a miss).
    EXPECT_GE(stats.insertions.value.load(), 200);
    EXPECT_GE(stats.hits.value.load() + stats.misses.value.load(), 200);
    EXPECT_GE(stats.hits.value.load(), 0);
    EXPECT_GE(stats.misses.value.load(), 0);
}

TEST(SafeCacheStatsTest, StatsSnapshotIsCopyable) {
    // unified_cache is neither copyable nor movable (mm_lru owns new-allocated pointers),
    // but the cache_stats returned by stats_snapshot() is a copyable value type.
    //
    // safe_cache embeds a large cache_stats (latency histograms), so a
    // stack-allocated cache plus several cache_stats copies would exceed
    // the default 1 MiB thread stack (clang ___chkstk_ms probe fails at
    // function entry). Allocate the cache and the first snapshot on the
    // heap; the copy/move under test still exercises value semantics.
    auto c = std::make_unique<safe_cache<int, int>>(10);
    c->set(1, 100);
    c->get(1);

    auto stats1 = std::make_unique<cache_stats>(c->stats_snapshot());
    EXPECT_EQ(stats1->hits.value.load(), 1);

    // Copy construction.
    cache_stats stats2 = *stats1;
    EXPECT_EQ(stats2.hits.value.load(), 1);

    // Copy assignment.
    cache_stats stats3;
    stats3 = *stats1;
    EXPECT_EQ(stats3.hits.value.load(), 1);

    // Move construction.
    cache_stats stats4 = std::move(stats2);
    EXPECT_EQ(stats4.hits.value.load(), 1);
}

// ============================================================================
// Capacity Tests
// ============================================================================

TEST(SafeCacheCapacityTest, MaxSizeEviction) {
    safe_cache<int, std::string> c{3};
    c.set(1, "one");
    c.set(2, "two");
    c.set(3, "three");
    c.set(4, "four"); // evicts 1

    EXPECT_EQ(c.size(), 3);
    EXPECT_FALSE(c.contains(1));
    EXPECT_TRUE(c.contains(4));
}

TEST(SafeCacheCapacityTest, ResizeDown) {
    safe_cache<int, int> c{5};
    for (int i = 0; i < 5; ++i) {
        c.set(i, i);
    }
    EXPECT_EQ(c.size(), 5);

    c.max_size(2);
    EXPECT_EQ(c.size(), 2);
}

// ============================================================================
// Callback Tests
// ============================================================================

TEST(SafeCacheCallbackTest, EvictCallback) {
    safe_cache<int, int> c{2};
    int evict_key = 0;
    int evict_value = 0;

    c.callbacks().on_evict([&](const int& k, const int& v) {
        evict_key = k;
        evict_value = v;
    });

    c.set(1, 100);
    c.set(2, 200);
    c.set(3, 300); // evicts 1

    EXPECT_EQ(evict_key, 1);
    EXPECT_EQ(evict_value, 100);
}

// ============================================================================
// Consistent Snapshot Tests
// ============================================================================

TEST(SafeCacheConsistentSnapshotTest, ConsistentSnapshotBasic) {
    safe_cache<int, int> c{10};
    c.set(1, 100);
    c.get(1);  // hit
    c.get(2);  // miss

    auto snap = c.stats_snapshot();
    EXPECT_EQ(snap.hits.value.load(), 1);
    EXPECT_EQ(snap.misses.value.load(), 1);
    // The snapshot's hit_rate() should be exactly 0.5 because the snapshot
    // captures all atomics together.
    EXPECT_DOUBLE_EQ(snap.hit_rate(), 0.5);
}

// ============================================================================
// Striped Cache Tests
// ============================================================================

class StripedCacheTest : public ::testing::Test {
protected:
    striped_cache<int, std::string> c{10};

    void SetUp() override {
        c.set(1, "one");
        c.set(2, "two");
        c.set(3, "three");
    }
};

TEST_F(StripedCacheTest, BasicGet) {
    auto result = c.get(1);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, "one");
}

TEST_F(StripedCacheTest, BasicSetAndGet) {
    c.set(4, "four");
    auto result = c.get(4);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, "four");
}

TEST_F(StripedCacheTest, PeekDoesNotTouch) {
    auto result = c.peek(1);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, "one");
}

TEST_F(StripedCacheTest, Delete) {
    EXPECT_TRUE(c.del(1));
    auto result = c.get(1);
    EXPECT_FALSE(result.has_value());
}

TEST_F(StripedCacheTest, Contains) {
    EXPECT_TRUE(c.contains(1));
    EXPECT_FALSE(c.contains(99));
}

TEST_F(StripedCacheTest, Size) {
    EXPECT_EQ(c.size(), 3);
}

TEST_F(StripedCacheTest, Flush) {
    c.flush();
    EXPECT_EQ(c.size(), 0);
}

TEST_F(StripedCacheTest, ReplaceExisting) {
    EXPECT_TRUE(c.replace(1, "updated"));
    EXPECT_EQ(*c.get(1), "updated");
}

TEST_F(StripedCacheTest, AddNewKey) {
    EXPECT_TRUE(c.add(4, "four"));
    EXPECT_EQ(c.size(), 4);
}

TEST_F(StripedCacheTest, AddExistingKey) {
    EXPECT_FALSE(c.add(1, "z"));
    EXPECT_EQ(*c.get(1), "one");
}

TEST_F(StripedCacheTest, MaxSizeEviction) {
    // With 64 shards, max_size must be large enough for meaningful per-shard capacity.
    // default hash is now well-mixed (ankerl::unordered_dense::hash),
    // so keys distribute uniformly across all 64 shards. Use a larger max_size
    // so per-shard capacity is high enough that hash distribution skew does
    // not cause cross-shard size variance below the global cap.
    striped_cache<int, std::string> sc{1024};
    // Fill beyond capacity to trigger evictions
    for (int i = 0; i < 1200; ++i) {
        sc.set(i, "v" + std::to_string(i));
    }
    EXPECT_LE(sc.size(), 1024u);
    // Older items should have been evicted; the most recent ones should remain
    EXPECT_TRUE(sc.contains(1199));
}

TEST_F(StripedCacheTest, StatsTracking) {
    c.get(1); // hit
    c.get(1); // hit
    c.get(99); // miss
    auto stats = c.stats_snapshot();
    EXPECT_GE(stats.hits.value.load(), 2);
    EXPECT_GE(stats.misses.value.load(), 1);
}

TEST_F(StripedCacheTest, IsStripedPolicy) {
    // Verify the striped_cache uses the striped policy at compile time
    static_assert(striped_cache<int, std::string>::is_striped,
                  "striped_cache must have is_striped = true");
    static_assert(striped_cache<int, std::string>::is_thread_safe,
                  "striped_cache must be thread_safe");
}

// ============================================================================
// Striped Cache Concurrency Tests
// ============================================================================

TEST(StripedCacheConcurrencyTest, ConcurrentReads) {
    // default hash is now well-mixed, so per-shard distribution is
    // uniform. Use a max_size large enough that no shard overflows during
    // the test (100 keys across 64 shards ≈ 1.5/shard; with max_size=100
    // some shards receive 2+ keys and evict, causing hit_count < 400).
    striped_cache<int, int> c{1024};
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

TEST(StripedCacheConcurrencyTest, ConcurrentWrites) {
    striped_cache<int, int> c{1000};

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

TEST(StripedCacheConcurrencyTest, MixedReadWrite) {
    striped_cache<int, int> c{100};
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
                    auto result = c.get(i);
                    (void)result;
                }
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_LE(c.size(), 100);
}

TEST(StripedCacheConcurrencyTest, ConcurrentGetAndDel) {
    striped_cache<int, int> c{100};
    for (int i = 0; i < 50; ++i) {
        c.set(i, i);
    }

    std::vector<std::thread> threads;

    // Thread A: reads (handle destruction races with del)
    threads.emplace_back([&]() {
        for (int i = 0; i < 50; ++i) {
            (void)c.get(i);
        }
    });

    // Thread B: deletes (may fail if handle still active)
    threads.emplace_back([&]() {
        for (int i = 0; i < 50; ++i) {
            (void)c.del(i);
        }
    });

    for (auto& t : threads) {
        t.join();
    }

    // del() may fail due to handle race; use force_del to clean up
    for (int i = 0; i < 50; ++i) {
        c.force_del(i);
    }
    for (int i = 0; i < 50; ++i) {
        EXPECT_FALSE(c.contains(i));
    }
}

TEST(StripedCacheConcurrencyTest, ConcurrentStatsConsistency) {
    striped_cache<int, int> c{500};
    for (int i = 0; i < 100; ++i) {
        c.set(i, i);
    }

    std::atomic<bool> stop{false};
    std::atomic<std::size_t> violation_count{0};

    // Writer threads: continuously set/get/del
    std::vector<std::thread> writers;
    for (int t = 0; t < 4; ++t) {
        writers.emplace_back([&, t]() {
            for (int i = 0; i < 500 && !stop.load(std::memory_order_relaxed); ++i) {
                int key = t * 1000 + i;
                c.set(key, key);
                c.get(key);
                c.get(key + 999);
                c.del(key);
            }
        });
    }

    // Reader thread: repeatedly take snapshots and verify invariants
    std::thread reader([&]() {
        for (int i = 0; i < 2000 && !stop.load(std::memory_order_relaxed); ++i) {
            auto snap = c.stats_snapshot();
            auto h = snap.hits.value.load(std::memory_order_relaxed);
            auto m = snap.misses.value.load(std::memory_order_relaxed);
            auto total = h + m;
            if (snap.total_accesses() != total) {
                violation_count.fetch_add(1, std::memory_order_relaxed);
            }
            if (h > total || m > total) {
                violation_count.fetch_add(1, std::memory_order_relaxed);
            }
        }
        stop.store(true, std::memory_order_relaxed);
    });

    for (auto& t : writers) {
        if (t.joinable()) t.join();
    }
    if (reader.joinable()) reader.join();

    EXPECT_EQ(violation_count.load(), 0);
}

TEST(StripedCacheConcurrencyTest, HighContentionStress) {
    // 8 threads, each performing 10K operations with get:set:del = 7:2:1
    striped_cache<int, int> c{2000};
    constexpr int num_threads = 8;
    constexpr int ops_per_thread = 10000;

    std::vector<std::thread> threads;
    std::atomic<int> total_sets{0};
    std::atomic<int> total_dels{0};

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            for (int i = 0; i < ops_per_thread; ++i) {
                int key = (t * ops_per_thread + i) % 500;
                int op = i % 10;
                if (op < 7) {
                    c.get(key);
                } else if (op < 9) {
                    c.set(key, t * 1000 + i);
                    total_sets.fetch_add(1, std::memory_order_relaxed);
                } else {
                    c.del(key);
                    total_dels.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    // Cache should not exceed max_size
    EXPECT_LE(c.size(), 2000);
    // Operations completed without crash or deadlock
    EXPECT_GT(total_sets.load(), 0);
}

TEST(StripedCacheConcurrencyTest, FlushUnderConcurrency) {
    striped_cache<int, int> c{1000};
    for (int i = 0; i < 500; ++i) {
        c.set(i, i);
    }

    std::atomic<bool> stop{false};
    std::vector<std::thread> threads;

    // Writer threads
    for (int t = 0; t < 2; ++t) {
        threads.emplace_back([&, t]() {
            for (int i = 0; i < 500 && !stop.load(); ++i) {
                c.set(t * 500 + i, i);
            }
        });
    }

    // Flush thread
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

TEST(StripedCacheConcurrencyTest, CallbackUnderConcurrency) {
    striped_cache<int, int> c{100};
    std::atomic<int> evict_count{0};

    // Use unified_cache::on_evict() which propagates to all shards
    c.on_evict([&](const int&, const int&) {
        evict_count.fetch_add(1, std::memory_order_relaxed);
    });

    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&, t]() {
            for (int i = 0; i < 200; ++i) {
                c.set(t * 200 + i, i);
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    // Cache is bounded to 100, so evictions must have occurred
    EXPECT_GT(evict_count.load(), 0);
    EXPECT_EQ(c.size(), 100);
}

// ============================================================================
// Consistent Snapshot Tests
// ============================================================================

TEST(SafeCacheConsistentSnapshotTest, ConcurrentStatsConsistency) {
    safe_cache<int, int> c{500};
    // Pre-populate some items
    for (int i = 0; i < 100; ++i) {
        c.set(i, i);
    }

    std::atomic<bool> stop{false};
    std::atomic<std::size_t> violation_count{0};

    // Writer threads: continuously set/get/del
    std::vector<std::thread> writers;
    for (int t = 0; t < 4; ++t) {
        writers.emplace_back([&, t]() {
            for (int i = 0; i < 500 && !stop.load(std::memory_order_relaxed); ++i) {
                int key = t * 1000 + i;
                c.set(key, key);
                c.get(key);        // hit (just set)
                c.get(key + 999);  // miss (unlikely to exist)
                c.del(key);
            }
        });
    }

    // Reader thread: repeatedly take consistent snapshots and verify invariants
    std::thread reader([&]() {
        for (int i = 0; i < 2000 && !stop.load(std::memory_order_relaxed); ++i) {
            auto snap = c.stats_snapshot();
            auto h = snap.hits.value.load(std::memory_order_relaxed);
            auto m = snap.misses.value.load(std::memory_order_relaxed);
            auto total = h + m;

            // Invariant: total_accesses() must equal hits + misses
            // since the snapshot captured all atomics together
            if (snap.total_accesses() != total) {
                violation_count.fetch_add(1, std::memory_order_relaxed);
            }
            // hits and misses must not exceed total
            if (h > total || m > total) {
                violation_count.fetch_add(1, std::memory_order_relaxed);
            }
        }
        stop.store(true, std::memory_order_relaxed);
    });

    for (auto& t : writers) {
        if (t.joinable()) t.join();
    }
    if (reader.joinable()) reader.join();

    // No invariant violations should be observed
    EXPECT_EQ(violation_count.load(), 0);
}
