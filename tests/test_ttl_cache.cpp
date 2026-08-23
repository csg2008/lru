// Unified LRU Cache - TTL Cache unit tests
// Updated for merged ttl.hpp API

#include <gtest/gtest.h>
#include <atomic>
#include <chrono>
#include <string>
#include <thread>
#include <vector>

#include "../lru.hpp"

using namespace lru;
using namespace std::chrono_literals;

// ============================================================================
// Basic TTL Tests
// ============================================================================

class TtlCacheTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

TEST_F(TtlCacheTest, DefaultConstructorNoTtl) {
    ttl_cache<int, std::string> c;
    c.set(1, "one");
    EXPECT_TRUE(c.contains(1));
    EXPECT_FALSE(c.has_expired(1));
}

TEST_F(TtlCacheTest, SetWithDefaultTtl) {
    ttl_cache<int, std::string, std::chrono::milliseconds> c(100ms);
    c.set(1, "one");
    EXPECT_TRUE(c.contains(1));

    std::this_thread::sleep_for(150ms);
    EXPECT_FALSE(c.contains(1));  // lazy expiration on access
}

TEST_F(TtlCacheTest, SetNoTtl) {
    ttl_cache<int, std::string, std::chrono::milliseconds> c(100ms);
    c.set_no_ttl(1, "permanent");

    std::this_thread::sleep_for(150ms);
    EXPECT_TRUE(c.contains(1));   // never expires
}

TEST_F(TtlCacheTest, SetWithCustomTtl) {
    ttl_cache<int, std::string> c;
    c.set_with_ttl(1, "short", 50ms);
    c.set_with_ttl(2, "long", 500ms);

    EXPECT_TRUE(c.contains(1));
    EXPECT_TRUE(c.contains(2));

    std::this_thread::sleep_for(100ms);
    EXPECT_FALSE(c.contains(1));  // expired
    EXPECT_TRUE(c.contains(2));   // still alive
}

TEST_F(TtlCacheTest, SetUntilAbsoluteTime) {
    ttl_cache<int, std::string> c;
    auto expiry = ttl_entry<std::string>::from_now(50ms);
    c.set_until(1, "one", expiry);

    EXPECT_TRUE(c.contains(1));
    std::this_thread::sleep_for(100ms);
    EXPECT_FALSE(c.contains(1));
}

// ============================================================================
// Lazy Expiration Tests
// ============================================================================

TEST_F(TtlCacheTest, GetExpiresLazily) {
    ttl_cache<int, std::string, std::chrono::milliseconds> c(50ms);
    c.set(1, "one");

    std::this_thread::sleep_for(100ms);
    auto result = c.get(1);  // should trigger lazy expiration
    EXPECT_FALSE(result.has_value());
}

TEST_F(TtlCacheTest, PeekDoesNotTouchExpired) {
    ttl_cache<int, std::string, std::chrono::milliseconds> c(50ms);
    c.set(1, "one");
    // peek() does not trigger lazy removal, so stale entries remain
    // has_expired() uses peek() and correctly reports expiry
    std::this_thread::sleep_for(100ms);
    EXPECT_TRUE(c.has_expired(1));
}

TEST_F(TtlCacheTest, GetBeforeExpiryReturnsValue) {
    ttl_cache<int, std::string, std::chrono::milliseconds> c(500ms);
    c.set(1, "one");

    auto result = c.get(1);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, "one");
}

// ============================================================================
// Eager Expiration Tests
// ============================================================================

TEST_F(TtlCacheTest, ClearExpired) {
    ttl_cache<int, std::string> c;
    c.set_with_ttl(1, "one", 50ms);
    c.set_with_ttl(2, "two", 500ms);
    c.set_no_ttl(3, "three");

    std::this_thread::sleep_for(100ms);

    auto cleared = c.clear_expired();
    EXPECT_EQ(cleared, 1);  // only key 1 expired
    EXPECT_FALSE(c.contains(1));
    EXPECT_TRUE(c.contains(2));
    EXPECT_TRUE(c.contains(3));
}

TEST_F(TtlCacheTest, ClearExpiredEmptyCache) {
    ttl_cache<int, std::string> c;
    EXPECT_EQ(c.clear_expired(), 0);
}

TEST_F(TtlCacheTest, ClearExpiredNoneExpired) {
    ttl_cache<int, std::string> c;
    c.set_with_ttl(1, "one", 500ms);
    EXPECT_EQ(c.clear_expired(), 0);
    EXPECT_TRUE(c.contains(1));
}

// ============================================================================
// Remaining TTL Tests
// ============================================================================

TEST_F(TtlCacheTest, RemainingTtl) {
    ttl_cache<int, std::string> c;
    c.set_with_ttl(1, "one", 500ms);

    auto remaining = c.remaining_ttl(1);
    ASSERT_TRUE(remaining.has_value());
    EXPECT_GE(remaining->count(), 0);
    EXPECT_LE(remaining->count(), 500);
}

TEST_F(TtlCacheTest, RemainingTtlNoTtl) {
    ttl_cache<int, std::string> c;
    c.set_no_ttl(1, "one");

    auto remaining = c.remaining_ttl(1);
    EXPECT_FALSE(remaining.has_value());
}

TEST_F(TtlCacheTest, RemainingTtlNotFound) {
    ttl_cache<int, std::string> c;
    auto remaining = c.remaining_ttl(99);
    EXPECT_FALSE(remaining.has_value());
}

TEST_F(TtlCacheTest, RemainingTtlAfterExpiry) {
    ttl_cache<int, std::string> c;
    c.set_with_ttl(1, "one", 50ms);

    std::this_thread::sleep_for(100ms);
    auto remaining = c.remaining_ttl(1);
    EXPECT_FALSE(remaining.has_value());  // expired
}

// ============================================================================
// Default TTL Configuration Tests
// ============================================================================

TEST_F(TtlCacheTest, DefaultTtlSetter) {
    ttl_cache<int, std::string> c;
    EXPECT_EQ(c.default_ttl(), std::chrono::seconds::zero());

    c.set_default_ttl(5s);
    EXPECT_EQ(c.default_ttl().count(), 5);
}

TEST_F(TtlCacheTest, ConstructorWithDefaultTtl) {
    ttl_cache<int, std::string, std::chrono::milliseconds> c(100ms, 10);
    c.set(1, "one");

    std::this_thread::sleep_for(150ms);
    EXPECT_FALSE(c.contains(1));
}

TEST_F(TtlCacheTest, MixedTtlAndNoTtl) {
    ttl_cache<int, std::string, std::chrono::milliseconds> c(50ms);
    c.set(1, "default_ttl");        // 50ms TTL
    c.set_no_ttl(2, "no_ttl");      // never expires
    c.set_with_ttl(3, "custom", 200ms);  // 200ms TTL

    std::this_thread::sleep_for(100ms);

    EXPECT_FALSE(c.contains(1));  // default 50ms expired
    EXPECT_TRUE(c.contains(2));   // no TTL, alive
    EXPECT_TRUE(c.contains(3));   // custom 200ms, alive
}

// ============================================================================
// Capacity with TTL Tests
// ============================================================================

TEST_F(TtlCacheTest, MaxSizeEvictionWithTtl) {
    ttl_cache<int, std::string> c(500ms, 2);  // TTL 500ms, max 2 items
    c.set(1, "one");
    c.set(2, "two");
    c.set(3, "three");  // should evict key 1 (LRU)

    EXPECT_EQ(c.size(), 2);
    EXPECT_FALSE(c.contains(1));
    EXPECT_TRUE(c.contains(2));
    EXPECT_TRUE(c.contains(3));
}

// ============================================================================
// Statistics Tests
// ============================================================================

TEST_F(TtlCacheTest, EvictionStatsOnLazyExpire) {
    ttl_cache<int, std::string, std::chrono::milliseconds> c(50ms);
    c.set(1, "one");

    std::this_thread::sleep_for(100ms);
    c.contains(1);  // triggers lazy expiration (del, not eviction)

    auto stats = c.stats();
    EXPECT_EQ(stats.evictions.value.load(), 0);  // del is not an eviction
}

TEST_F(TtlCacheTest, EvictionStatsOnClearExpired) {
    ttl_cache<int, std::string> c;
    c.set_with_ttl(1, "one", 50ms);
    c.set_with_ttl(2, "two", 50ms);

    std::this_thread::sleep_for(100ms);
    auto cleared = c.clear_expired();

    EXPECT_EQ(cleared, 2);
    auto stats = c.stats();
    EXPECT_EQ(stats.evictions.value.load(), 0);  // erase is not an eviction
}

TEST_F(TtlCacheTest, InsertionStats) {
    ttl_cache<int, std::string> c;
    c.set(1, "one");
    c.set(2, "two");
    c.set(1, "updated");  // update, not new insertion

    auto stats = c.stats();
    EXPECT_EQ(stats.insertions.value.load(), 2);
}

// ============================================================================
// Callback Tests
// ============================================================================

TEST_F(TtlCacheTest, EvictCallbackOnTtlExpiry) {
    ttl_cache<int, std::string, std::chrono::milliseconds> c(50ms);
    int evict_count = 0;

    c.underlying().callbacks().on_evict([&](const int& k, const ttl_entry<std::string>& v) {
        (void)k; (void)v;
        ++evict_count;
    });

    c.set(1, "one");
    std::this_thread::sleep_for(100ms);
    c.clear_expired();

    EXPECT_EQ(evict_count, 1);
}

TEST_F(TtlCacheTest, InsertCallback) {
    ttl_cache<int, std::string> c;
    int insert_key = 0;

    c.underlying().callbacks().on_insert([&](const int& k, const ttl_entry<std::string>& v) {
        (void)v;
        insert_key = k;
    });

    c.set(42, "answer");
    EXPECT_EQ(insert_key, 42);
}

// ============================================================================
// Del and Flush Tests
// ============================================================================

TEST_F(TtlCacheTest, DelExisting) {
    ttl_cache<int, std::string> c;
    c.set(1, "one");
    EXPECT_TRUE(c.del(1));
    EXPECT_FALSE(c.contains(1));
}

TEST_F(TtlCacheTest, DelNonExisting) {
    ttl_cache<int, std::string> c;
    EXPECT_FALSE(c.del(99));
}

TEST_F(TtlCacheTest, Flush) {
    ttl_cache<int, std::string> c;
    c.set(1, "one");
    c.set(2, "two");
    c.flush();
    EXPECT_TRUE(c.empty());
    EXPECT_EQ(c.size(), 0);
}

// ============================================================================
// HasExpired Tests
// ============================================================================

TEST_F(TtlCacheTest, HasExpiredBeforeAndAfter) {
    ttl_cache<int, std::string> c;
    c.set_with_ttl(1, "one", 50ms);

    EXPECT_FALSE(c.has_expired(1));
    std::this_thread::sleep_for(100ms);
    EXPECT_TRUE(c.has_expired(1));
}

TEST_F(TtlCacheTest, HasExpiredNotFound) {
    ttl_cache<int, std::string> c;
    EXPECT_FALSE(c.has_expired(99));  // not found = not expired
}

// ============================================================================
// Concurrency Tests (thread-safe TTL cache)
// ============================================================================

using ttl_entry_type = ttl_entry<int>;
using safe_ttl_t = ttl_cache<int, int, std::chrono::seconds,
    unified_cache<lru_trait<thread_safe_policy>, int, ttl_entry_type>>;
using safe_ttl_ms_t = ttl_cache<int, int, std::chrono::milliseconds,
    unified_cache<lru_trait<thread_safe_policy>, int, ttl_entry_type>>;

TEST(TtlCacheConcurrencyTest, ConcurrentGetAndSet) {
    safe_ttl_t cache(60s, 1000);

    // Pre-populate
    for (int i = 0; i < 100; ++i) {
        cache.set(i, i * 10);
    }

    std::vector<std::thread> threads;
    std::atomic<int> ops{0};

    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&, t]() {
            for (int i = 0; i < 500; ++i) {
                int key = (t * 500 + i) % 200;
                if (i % 3 == 0) {
                    cache.set(key, key * 10);
                } else {
                    (void)cache.get(key);
                }
                ops.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_EQ(ops.load(), 2000);
}

TEST(TtlCacheConcurrencyTest, ConcurrentExpiredCleanup) {
    safe_ttl_ms_t cache(20ms, 1000);

    std::vector<std::thread> threads;
    std::atomic<int> ops{0};

    // Writer threads
    for (int t = 0; t < 2; ++t) {
        threads.emplace_back([&, t]() {
            for (int i = 0; i < 500; ++i) {
                cache.set(t * 500 + i, i);
                ops.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }

    // Cleanup thread — runs a fixed number of cleanup cycles
    threads.emplace_back([&]() {
        for (int i = 0; i < 100; ++i) {
            cache.clear_expired();
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    });

    // Reader threads
    for (int t = 0; t < 2; ++t) {
        threads.emplace_back([&, t]() {
            for (int i = 0; i < 500; ++i) {
                (void)cache.get((t * 500 + i) % 1000);
                ops.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    // All writer + reader ops completed (cleanup thread ops not counted)
    EXPECT_GE(ops.load(), 2000);
}

TEST(TtlCacheConcurrencyTest, ConcurrentMixedOperations) {
    safe_ttl_t cache(60s, 500);

    // Pre-populate
    for (int i = 0; i < 100; ++i) {
        cache.set(i, i);
    }

    std::vector<std::thread> threads;
    std::atomic<int> ops{0};

    // Thread group: get + peek + contains
    for (int t = 0; t < 2; ++t) {
        threads.emplace_back([&]() {
            for (int i = 0; i < 500; ++i) {
                int key = i % 150;
                (void)cache.get(key);
                (void)cache.peek(key);
                (void)cache.contains(key);
                ops.fetch_add(3, std::memory_order_relaxed);
            }
        });
    }

    // Thread group: set + del
    for (int t = 0; t < 2; ++t) {
        threads.emplace_back([&, t]() {
            for (int i = 0; i < 500; ++i) {
                int key = (t * 500 + i) % 150;
                cache.set(key, key * 10);
                if (i % 5 == 0) {
                    (void)cache.del(key);
                }
                ops.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    // Verify no deadlock occurred (we reached here) and all ops completed
    EXPECT_EQ(ops.load(), 4000);
}

// ============================================================================
// Graceful Shutdown Tests
// ============================================================================

TEST_F(TtlCacheTest, StopMarksCacheAsStopped) {
    ttl_cache<int, std::string> c;
    EXPECT_FALSE(c.is_stopped());
    c.stop();
    EXPECT_TRUE(c.is_stopped());
}

TEST_F(TtlCacheTest, StopClearsAllData) {
    ttl_cache<int, std::string> c;
    c.set(1, "one");
    c.set(2, "two");
    EXPECT_EQ(c.size(), 2);
    c.stop();
    EXPECT_TRUE(c.empty());
    EXPECT_EQ(c.size(), 0);
}

TEST_F(TtlCacheTest, StopMakesSetNoop) {
    ttl_cache<int, std::string> c;
    c.stop();
    c.set(1, "one");
    EXPECT_TRUE(c.empty());
}

TEST_F(TtlCacheTest, StopMakesGetReturnNullopt) {
    ttl_cache<int, std::string> c;
    c.set(1, "one");
    c.stop();
    auto result = c.get(1);
    EXPECT_FALSE(result.has_value());
}

TEST_F(TtlCacheTest, StopMakesPeekReturnNullopt) {
    ttl_cache<int, std::string> c;
    c.set(1, "one");
    c.stop();
    auto result = c.peek(1);
    EXPECT_FALSE(result.has_value());
}

TEST_F(TtlCacheTest, StopMakesContainsReturnFalse) {
    ttl_cache<int, std::string> c;
    c.set(1, "one");
    c.stop();
    EXPECT_FALSE(c.contains(1));
}

TEST_F(TtlCacheTest, StopMakesDelReturnFalse) {
    ttl_cache<int, std::string> c;
    c.set(1, "one");
    c.stop();
    EXPECT_FALSE(c.del(1));
}

TEST_F(TtlCacheTest, StopMakesSetWithTtlNoop) {
    ttl_cache<int, std::string> c;
    c.stop();
    c.set_with_ttl(1, "one", 5s);
    EXPECT_TRUE(c.empty());
}

TEST_F(TtlCacheTest, StopMakesSetNoTtlNoop) {
    ttl_cache<int, std::string> c;
    c.stop();
    c.set_no_ttl(1, "one");
    EXPECT_TRUE(c.empty());
}

TEST_F(TtlCacheTest, StopMakesSetUntilNoop) {
    ttl_cache<int, std::string> c;
    c.stop();
    c.set_until(1, "one", ttl_entry<std::string>::from_now(5s));
    EXPECT_TRUE(c.empty());
}

TEST_F(TtlCacheTest, StopMakesClearExpiredReturnZero) {
    ttl_cache<int, std::string> c;
    c.set_with_ttl(1, "one", 10ms);
    std::this_thread::sleep_for(50ms);
    c.stop();
    // After stop, clear_expired is a no-op
    EXPECT_EQ(c.clear_expired(), 0);
}

TEST_F(TtlCacheTest, StopMakesFlushNoop) {
    ttl_cache<int, std::string> c;
    c.set(1, "one");
    c.stop();
    // Already empty after stop(), calling flush again is a no-op
    c.flush();
    EXPECT_TRUE(c.empty());
}

TEST_F(TtlCacheTest, StopMakesSetDefaultTtlNoop) {
    ttl_cache<int, std::string> c;
    c.stop();
    c.set_default_ttl(5s);
    // default_ttl() still returns the old value since set_default_ttl was a no-op
    EXPECT_EQ(c.default_ttl(), std::chrono::seconds::zero());
}

TEST_F(TtlCacheTest, StopIdempotent) {
    ttl_cache<int, std::string> c;
    c.set(1, "one");
    c.stop();
    c.stop();  // second stop should be a no-op
    EXPECT_TRUE(c.is_stopped());
    EXPECT_TRUE(c.empty());
}

TEST_F(TtlCacheTest, SizeAfterStopReflectsEmpty) {
    ttl_cache<int, std::string> c;
    c.set(1, "one");
    c.set(2, "two");
    c.set(3, "three");
    EXPECT_EQ(c.size(), 3);
    c.stop();
    EXPECT_EQ(c.size(), 0);
}

TEST_F(TtlCacheTest, HasExpiredAfterStopReturnsFalse) {
    ttl_cache<int, std::string> c;
    c.set_with_ttl(1, "one", 10ms);
    c.stop();
    EXPECT_FALSE(c.has_expired(1));
}

TEST_F(TtlCacheTest, RemainingTtlAfterStopReturnsNullopt) {
    ttl_cache<int, std::string> c;
    c.set_with_ttl(1, "one", 500ms);
    c.stop();
    EXPECT_FALSE(c.remaining_ttl(1).has_value());
}

// ============================================================================
// Periodic Worker Destructor noexcept Tests
// ============================================================================

TEST(PeriodicWorkerNoexceptTest, DestructorDoesNotThrow) {
    // Create and destroy periodic_worker instances in a loop.
    // If the destructor throws, the program would terminate (noexcept violation).
    for (int i = 0; i < 5; ++i) {
        lru::detail::periodic_worker worker(
            []() {}, std::chrono::milliseconds(100));
        worker.stop();
    }
}

TEST(PeriodicWorkerNoexceptTest, DestructorNoexceptWithoutExplicitStop) {
    // Let the destructor call stop() implicitly — it must not throw.
    for (int i = 0; i < 5; ++i) {
        lru::detail::periodic_worker worker(
            []() {}, std::chrono::milliseconds(100));
        // Intentionally not calling stop() — destructor handles it
    }
}

// ============================================================================
// TTL Periodic Worker Shutdown Order Tests
// ============================================================================
//
// P1-A: `ttl_reaper` has been removed. These tests verify that a
// `detail::periodic_worker` driving `ttl_cache::clear_expired()` can
// be safely stopped before / after the cache, and that concurrent
// `clear_expired()` calls on a stopped cache are no-ops.

TEST(TtlPeriodicWorkerShutdownTest, StopWorkerBeforeCacheDestruction) {
    auto cache = std::make_unique<ttl_cache<int, std::string>>(5s, 100);
    cache->set(1, "one");
    cache->set(2, "two");

    {
        detail::periodic_worker worker(
            [&] { cache->clear_expired(); },
            std::chrono::milliseconds(50));
        EXPECT_TRUE(worker.is_running());
        worker.stop();
        EXPECT_FALSE(worker.is_running());
    }
    // worker destroyed; cache still valid
    EXPECT_TRUE(cache->contains(1));
    EXPECT_TRUE(cache->contains(2));
    cache.reset();
}

TEST(TtlPeriodicWorkerShutdownTest, WorkerDestructorStopsThread) {
    auto cache = std::make_unique<ttl_cache<int, std::string>>(5s, 100);
    cache->set(1, "one");
    {
        detail::periodic_worker worker(
            [&] { cache->clear_expired(); },
            std::chrono::milliseconds(50));
        EXPECT_TRUE(worker.is_running());
        // Let worker destructor call stop()
    }
    // Cache should still be usable after worker is gone
    EXPECT_TRUE(cache->contains(1));
    cache->set(3, "three");
    EXPECT_TRUE(cache->contains(3));
}

TEST(TtlPeriodicWorkerShutdownTest, WorkerWithStoppedCache) {
    ttl_cache<int, std::string> cache(5s, 100);
    cache.set(1, "one");

    detail::periodic_worker worker(
        [&] { cache.clear_expired(); },
        std::chrono::milliseconds(50));
    // Stop the cache first, then the worker
    cache.stop();
    EXPECT_TRUE(cache.is_stopped());
    worker.stop();
    // The worker's clear_expired() calls on a stopped cache are no-ops
    EXPECT_FALSE(worker.is_running());
}

TEST(TtlPeriodicWorkerShutdownTest, MultipleWorkersSequentialShutdown) {
    auto cache = std::make_unique<ttl_cache<int, std::string>>(5s, 100);
    cache->set(1, "one");

    {
        detail::periodic_worker worker1(
            [&] { cache->clear_expired(); },
            std::chrono::milliseconds(100));
        detail::periodic_worker worker2(
            [&] { cache->clear_expired(); },
            std::chrono::milliseconds(150));
        worker1.stop();
        worker2.stop();
    }
    EXPECT_TRUE(cache->contains(1));
}

// ============================================================================
// Background Evictor Shutdown Tests
// ============================================================================

TEST(BackgroundEvictorShutdownTest, DestructorNoexcept) {
    using cache_t = lru::cache<int, std::string>;
    cache_t c(100);
    for (int i = 0; i < 5; ++i) {
        background_evictor<cache_t> evictor(c, [](cache_t&) {});
        evictor.start(std::chrono::milliseconds(100));
        // Destructor must not throw
    }
}

TEST(BackgroundEvictorShutdownTest, StopBeforeCacheDestruction) {
    using cache_t = lru::cache<int, std::string>;
    auto c = std::make_unique<cache_t>(100);
    c->set(1, "one");

    {
        background_evictor<cache_t> evictor(*c, [](cache_t&) {});
        evictor.start(std::chrono::milliseconds(100));
        evictor.stop();
    }
    // Cache still usable after evictor is gone
    EXPECT_TRUE(c->contains(1));
}

// ============================================================================
// spec.md P1-1: Native TTL integration on unified_cache (no wrapper)
// ============================================================================
//
// These tests verify that `unified_cache` (specifically `lru::cache` and
// `lru::striped_cache`) natively supports TTL via `set_with_ttl()` and that
// `peek_for_get` inline-checks expiry without the `ttl_cache` wrapper's
// double-locking. The background TTL cleaner (`start_ttl_cleaner`) and the
// synchronous `evict_expired_now()` should also work.

TEST(NativeTtlTest, SetWithTtlExpiresLazilyOnGet) {
    lru::cache<int, std::string> c(100);
    c.set_with_ttl(1, "one", 50ms);
    // Immediately readable.
    EXPECT_TRUE(c.get(1).has_value());
    // Wait for expiry.
    std::this_thread::sleep_for(100ms);
    // Refresh the cached time (as the background TTL cleaner would),
    // then get() should report a miss via lazy expiry.
    c.refresh_cached_now();
    EXPECT_FALSE(c.get(1).has_value());
}

TEST(NativeTtlTest, SetWithZeroTtlMeansNoExpiry) {
    lru::cache<int, std::string> c(100);
    c.set_with_ttl(1, "one", std::chrono::milliseconds(0));
    std::this_thread::sleep_for(50ms);
    // Zero TTL = no expiry.
    EXPECT_TRUE(c.get(1).has_value());
}

TEST(NativeTtlTest, SetWithoutTtlNeverExpires) {
    lru::cache<int, std::string> c(100);
    // Plain set() (no TTL) — item should never expire.
    c.set(1, "one");
    std::this_thread::sleep_for(50ms);
    EXPECT_TRUE(c.get(1).has_value());
}

TEST(NativeTtlTest, GetWithTtlReturnsRemainingNanoseconds) {
    lru::cache<int, std::string> c(100);
    // Disable jitter so the TTL bounds are deterministic.
    c.set_ttl_jitter_enabled(false);
    c.set_with_ttl(1, "one", 1s);
    auto [handle, ttl_ns] = c.get_with_ttl(1);
    EXPECT_TRUE(handle.has_value());
    // Should have ~1s remaining (allow some slack for timing).
    ASSERT_TRUE(ttl_ns.has_value());
    EXPECT_GT(*ttl_ns, 500u * 1000u * 1000u);  // > 500ms
    EXPECT_LE(*ttl_ns, 1000u * 1000u * 1000u); // <= 1s
}

TEST(NativeTtlTest, GetWithTtlReturnsNulloptTtlForNonTtlItem) {
    lru::cache<int, std::string> c(100);
    c.set(1, "one");  // no TTL
    auto [handle, ttl_ns] = c.get_with_ttl(1);
    EXPECT_TRUE(handle.has_value());
    // Plain set() leaves expiry_ns == 0, so ttl_remaining_ns returns nullopt.
    EXPECT_FALSE(ttl_ns.has_value());
}

TEST(NativeTtlTest, EvictExpiredNowRemovesExpiredItems) {
    lru::cache<int, std::string> c(100);
    // Disable jitter so the TTL is deterministic.
    c.set_ttl_jitter_enabled(false);
    c.set_with_ttl(1, "one", 50ms);
    c.set_with_ttl(2, "two", 1s);
    c.set(3, "three");  // no TTL
    EXPECT_EQ(c.size(), 3u);

    // Wait for key 1 to expire.
    std::this_thread::sleep_for(100ms);

    // Synchronous eviction: should remove key 1 only.
    std::size_t evicted = c.evict_expired_now();
    EXPECT_EQ(evicted, 1u);
    EXPECT_FALSE(c.contains(1));
    EXPECT_TRUE(c.contains(2));
    EXPECT_TRUE(c.contains(3));
}

TEST(NativeTtlTest, BackgroundCleanerEvictsExpiredItems) {
    lru::cache<int, std::string> c(100);
    c.set_with_ttl(1, "one", 50ms);
    c.set_with_ttl(2, "two", 50ms);
    EXPECT_EQ(c.size(), 2u);

    // Start background cleaner at 50ms interval.
    c.start_ttl_cleaner(std::chrono::milliseconds(50));

    // Wait long enough for the items to expire AND the cleaner to run.
    std::this_thread::sleep_for(300ms);
    c.stop_ttl_cleaner();

    // Both expired items should have been removed by the background cleaner.
    EXPECT_EQ(c.size(), 0u);
}

TEST(NativeTtlTest, StripedCacheSupportsNativeTtl) {
    // striped_cache uses sharded_mm_lru which delegates to per-shard mm_lru.
    // This test verifies the delegation chain works end-to-end.
    lru::striped_cache<int, std::string> c(1000, 4);
    c.set_with_ttl(1, "one", 50ms);
    c.set(2, "two");
    EXPECT_TRUE(c.get(1).has_value());
    EXPECT_TRUE(c.get(2).has_value());

    std::this_thread::sleep_for(100ms);
    // Refresh cached time (as the background TTL cleaner would),
    // then get() should report a miss for the expired key.
    c.refresh_cached_now();
    EXPECT_FALSE(c.get(1).has_value());  // expired
    EXPECT_TRUE(c.get(2).has_value());   // no TTL

    // evict_expired_now should clean up the expired entry.
    std::size_t evicted = c.evict_expired_now();
    EXPECT_GE(evicted, 1u);
}

TEST(NativeTtlTest, UpdateExistingItemResetsExpiry) {
    lru::cache<int, std::string> c(100);
    // Disable jitter so the TTL is deterministic.
    c.set_ttl_jitter_enabled(false);
    // Set with 50ms TTL.
    c.set_with_ttl(1, "one", 50ms);
    std::this_thread::sleep_for(30ms);
    // Update with 1s TTL — should reset the expiry.
    c.set_with_ttl(1, "ONE", 1s);
    std::this_thread::sleep_for(50ms);
    // Original 50ms TTL would have expired; new 1s TTL should still be valid.
    auto h = c.get(1);
    ASSERT_TRUE(h.has_value());
    EXPECT_EQ(*h, "ONE");
}
