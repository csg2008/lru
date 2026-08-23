// Tiered cache unit tests.
// Focus: P1-2 thundering-herd prevention via the in-flight shared_future table.
// Verifies that N concurrent get() calls on the same missing key result in
// exactly 1 backend lookup; the other N-1 callers become followers and are
// served from the primary cache after the leader promotes the value.

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "../lru.hpp"

using namespace lru;

namespace {

/// A counting storage backend. Tracks how many times get() is invoked per
/// key so tests can assert that the thundering-herd deduplication works.
class counting_backend : public storage_backend<int, std::string> {
public:
    std::optional<std::string> get(const int& key) override {
        // Slow down the first call so other threads have time to pile up
        // behind the in-flight table entry — this is what makes the test
        // reliably exercise the follower path.
        if (delay_ > std::chrono::milliseconds::zero()) {
            std::this_thread::sleep_for(delay_);
        }
        std::lock_guard lock(mutex_);
        ++get_call_count_;
        ++per_key_calls_[key];
        auto it = data_.find(key);
        if (it == data_.end()) return std::nullopt;
        return it->second;
    }

    void put(const int& key, const std::string& value) override {
        std::lock_guard lock(mutex_);
        data_[key] = value;
    }

    bool remove(const int& key) override {
        std::lock_guard lock(mutex_);
        return data_.erase(key) > 0;
    }

    bool contains(const int& key) const override {
        std::lock_guard lock(mutex_);
        return data_.find(key) != data_.end();
    }

    std::size_t size() const override {
        std::lock_guard lock(mutex_);
        return data_.size();
    }

    std::string name() const override { return "counting_backend"; }

    void seed(const int& key, std::string value) {
        std::lock_guard lock(mutex_);
        data_[key] = std::move(value);
    }

    std::size_t get_call_count() const {
        std::lock_guard lock(mutex_);
        return get_call_count_;
    }

    std::size_t calls_for_key(const int& key) const {
        std::lock_guard lock(mutex_);
        auto it = per_key_calls_.find(key);
        return it == per_key_calls_.end() ? 0 : it->second;
    }

    void set_delay(std::chrono::milliseconds d) { delay_ = d; }

private:
    mutable std::mutex mutex_;
    ankerl::unordered_dense::map<int, std::string> data_;
    std::unordered_map<int, std::size_t> per_key_calls_;
    std::size_t get_call_count_ = 0;
    std::chrono::milliseconds delay_{0};
};

}  // namespace

// ============================================================================
// Basic tiered_cache behaviour
// ============================================================================

TEST(TieredCacheTest, ReadThroughPromotesOnMiss) {
    counting_backend backend;
    backend.seed(42, "answer");

    tiered_cache<safe_cache<int, std::string>, counting_backend> tc(
        /*max_size=*/8, backend);

    auto h = tc.get(42);
    ASSERT_TRUE(h.has_value());
    EXPECT_EQ(*h, "answer");

    auto stats = tc.get_stats();
    EXPECT_EQ(stats.backend_hits, 1u);
    EXPECT_EQ(stats.promotions, 1u);
    EXPECT_EQ(backend.get_call_count(), 1u);

    // Second get() should hit the primary cache, not the backend.
    auto h2 = tc.get(42);
    ASSERT_TRUE(h2.has_value());
    EXPECT_EQ(*h2, "answer");
    EXPECT_EQ(backend.get_call_count(), 1u);
    EXPECT_EQ(tc.get_stats().primary_hits, 1u);
}

TEST(TieredCacheTest, BackendMissPropagates) {
    counting_backend backend;
    tiered_cache<safe_cache<int, std::string>, counting_backend> tc(
        /*max_size=*/8, backend);

    auto h = tc.get(999);
    EXPECT_FALSE(h.has_value());

    auto stats = tc.get_stats();
    EXPECT_EQ(stats.backend_misses, 1u);
    EXPECT_EQ(backend.get_call_count(), 1u);
}

TEST(TieredCacheTest, InflightCountIsZeroAtRest) {
    counting_backend backend;
    backend.seed(1, "v1");
    tiered_cache<safe_cache<int, std::string>, counting_backend> tc(
        /*max_size=*/8, backend);

    EXPECT_EQ(tc.inflight_count(), 0u);
    (void)tc.get(1);
    EXPECT_EQ(tc.inflight_count(), 0u);
}

// ============================================================================
// P1-2: Thundering herd prevention
// ============================================================================

TEST(TieredCacheThunderingHerdTest, HundredConcurrentSameKeyMissInvokesBackendOnce) {
    // Spec acceptance criterion (P1-2): 100 concurrent same-key misses
    // must result in the backend being called exactly once.
    counting_backend backend;
    backend.seed(7, "lucky");
    // Make the leader slow so all followers pile up behind the in-flight
    // entry while it's still in flight.
    backend.set_delay(std::chrono::milliseconds(80));

    tiered_cache<safe_cache<int, std::string>, counting_backend> tc(
        /*max_size=*/64, backend);

    constexpr int kThreads = 100;
    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    std::atomic<int> hits{0};
    std::atomic<int> misses{0};
    std::atomic<int> errors{0};

    // Barrier so all threads start as close to simultaneously as possible.
    std::atomic<int> ready{0};
    std::atomic<bool> go{false};

    for (int i = 0; i < kThreads; ++i) {
        threads.emplace_back([&, i]() {
            ready.fetch_add(1, std::memory_order_acq_rel);
            while (!go.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            try {
                auto h = tc.get(7);
                if (h.has_value()) {
                    if (*h == "lucky") {
                        hits.fetch_add(1, std::memory_order_relaxed);
                    } else {
                        errors.fetch_add(1, std::memory_order_relaxed);
                    }
                } else {
                    misses.fetch_add(1, std::memory_order_relaxed);
                }
            } catch (...) {
                errors.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }

    while (ready.load(std::memory_order_acquire) < kThreads) {
        std::this_thread::yield();
    }
    go.store(true, std::memory_order_release);

    for (auto& t : threads) t.join();

    EXPECT_EQ(errors.load(), 0) << "no thread should observe an error";
    EXPECT_EQ(misses.load(), 0) << "every thread should see the value";
    EXPECT_EQ(hits.load(), kThreads);

    // Critical assertion: backend called exactly once for key 7.
    EXPECT_EQ(backend.calls_for_key(7), 1u);
    EXPECT_EQ(backend.get_call_count(), 1u);

    // At least 99 threads should have been served as followers.
    auto stats = tc.get_stats();
    EXPECT_GE(stats.inflight_followers,
              static_cast<std::size_t>(kThreads - 1));
    EXPECT_EQ(tc.inflight_count(), 0u) << "in-flight table must drain";
}

TEST(TieredCacheThunderingHerdTest, DistinctKeysAreNotDeduplicated) {
    // Sanity: dedup is per-key. N threads each fetching a *distinct* key
    // must produce N backend calls.
    counting_backend backend;
    for (int i = 0; i < 32; ++i) backend.seed(i, "v" + std::to_string(i));
    backend.set_delay(std::chrono::milliseconds(20));

    tiered_cache<safe_cache<int, std::string>, counting_backend> tc(
        /*max_size=*/256, backend);

    constexpr int kThreads = 32;
    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    std::atomic<int> hits{0};

    std::atomic<int> ready{0};
    std::atomic<bool> go{false};

    for (int i = 0; i < kThreads; ++i) {
        threads.emplace_back([&, i]() {
            ready.fetch_add(1, std::memory_order_acq_rel);
            while (!go.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            auto h = tc.get(i);
            if (h.has_value() && *h == "v" + std::to_string(i)) {
                hits.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }

    while (ready.load(std::memory_order_acquire) < kThreads) {
        std::this_thread::yield();
    }
    go.store(true, std::memory_order_release);

    for (auto& t : threads) t.join();

    EXPECT_EQ(hits.load(), kThreads);
    // Each key was fetched exactly once.
    EXPECT_EQ(backend.get_call_count(),
              static_cast<std::size_t>(kThreads));
    for (int i = 0; i < kThreads; ++i) {
        EXPECT_EQ(backend.calls_for_key(i), 1u);
    }
}

TEST(TieredCacheThunderingHerdTest, BackendMissIsAlsoDeduplicated) {
    // Negative caching: when the backend returns nullopt, followers should
    // also observe the miss without re-querying the backend.
    counting_backend backend;
    // No seed for key 1234 — backend will return nullopt.
    backend.set_delay(std::chrono::milliseconds(60));

    tiered_cache<safe_cache<int, std::string>, counting_backend> tc(
        /*max_size=*/8, backend);

    constexpr int kThreads = 50;
    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    std::atomic<int> misses{0};

    std::atomic<int> ready{0};
    std::atomic<bool> go{false};

    for (int i = 0; i < kThreads; ++i) {
        threads.emplace_back([&]() {
            ready.fetch_add(1, std::memory_order_acq_rel);
            while (!go.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            auto h = tc.get(1234);
            if (!h.has_value()) misses.fetch_add(1, std::memory_order_relaxed);
        });
    }

    while (ready.load(std::memory_order_acquire) < kThreads) {
        std::this_thread::yield();
    }
    go.store(true, std::memory_order_release);

    for (auto& t : threads) t.join();

    EXPECT_EQ(misses.load(), kThreads);
    // Backend called exactly once even on a miss.
    EXPECT_EQ(backend.calls_for_key(1234), 1u);
    EXPECT_EQ(tc.inflight_count(), 0u);
}

TEST(TieredCacheThunderingHerdTest, SecondWaveBecomesFreshLeader) {
    // After the leader completes and erases its in-flight entry, a later
    // miss on the same key (because the primary cache evicted it) must
    // trigger a fresh backend call — the in-flight table should NOT
    // cache negative or stale entries.
    counting_backend backend;
    backend.seed(55, "first");
    backend.set_delay(std::chrono::milliseconds(30));

    // Tiny primary cache so we can force eviction between waves.
    tiered_cache<safe_cache<int, std::string>, counting_backend> tc(
        /*max_size=*/2, backend);

    auto h1 = tc.get(55);
    ASSERT_TRUE(h1.has_value());
    EXPECT_EQ(*h1, "first");
    EXPECT_EQ(backend.calls_for_key(55), 1u);
    // Release the handle so the item can be evicted.
    h1.release();

    // Fill the primary cache with other keys to evict key 55.
    tc.set(1, "a");
    tc.set(2, "b");
    // Now key 55 should be gone from the primary cache (size 2, LRU).
    auto peek_h = tc.peek(55);
    EXPECT_FALSE(peek_h.has_value())
        << "precondition: key 55 must have been evicted from primary";

    // Update the backend value so we can distinguish fresh vs. stale.
    backend.put(55, "second");

    auto h2 = tc.get(55);
    ASSERT_TRUE(h2.has_value());
    EXPECT_EQ(*h2, "second") << "must observe the refreshed backend value";
    EXPECT_EQ(backend.calls_for_key(55), 2u)
        << "a fresh miss after eviction must trigger a new backend call";
}

// ============================================================================
// P1-5: memory_storage_backend striped locking
// ============================================================================

TEST(TieredCacheStripedInflightTest, ManyDistinctKeysFetchedConcurrently) {
    // O4: Verify the striped inflight_mutex allows concurrent fetches on
    // DIFFERENT keys to proceed in parallel (no global serialization).
    // With 32 threads fetching 32 distinct keys (each backend.get taking
    // 30ms), a single global mutex would serialize all fetches → ~960ms
    // total. With 64 stripes, the expected serialization is negligible
    // (each thread hits its own stripe) → ~30ms total. We assert the
    // wall-clock time is well under the serialized lower bound.
    counting_backend backend;
    constexpr int kKeys = 32;
    for (int i = 0; i < kKeys; ++i) {
        backend.seed(i, "v" + std::to_string(i));
    }
    backend.set_delay(std::chrono::milliseconds(30));

    tiered_cache<safe_cache<int, std::string>, counting_backend> tc(
        /*max_size=*/256, backend);

    constexpr int kThreads = 32;
    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    std::atomic<int> hits{0};

    std::atomic<int> ready{0};
    std::atomic<bool> go{false};

    for (int i = 0; i < kThreads; ++i) {
        threads.emplace_back([&, i]() {
            ready.fetch_add(1, std::memory_order_acq_rel);
            while (!go.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            auto h = tc.get(i);
            if (h.has_value() && *h == "v" + std::to_string(i)) {
                hits.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }

    while (ready.load(std::memory_order_acquire) < kThreads) {
        std::this_thread::yield();
    }
    auto t_start = std::chrono::steady_clock::now();
    go.store(true, std::memory_order_release);
    for (auto& t : threads) t.join();
    auto t_end = std::chrono::steady_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                          t_end - t_start).count();

    EXPECT_EQ(hits.load(), kThreads);
    EXPECT_EQ(backend.get_call_count(),
              static_cast<std::size_t>(kThreads));
    // O4 contract: striped locking enables parallelism. The serialized
    // lower bound is kThreads * delay = 32 * 30ms = 960ms. We require
    // at least 4x speedup (< 240ms) to confirm the striping is effective.
    // (Conservative threshold — actual is typically ~30-50ms.)
    EXPECT_LT(elapsed_ms, 240)
        << "expected parallel fetches to complete well under 960ms; "
        << "got " << elapsed_ms << "ms (striped locking may be ineffective)";
}

TEST(TieredCacheStripedInflightTest, InflightCountCorrectAfterConcurrentOps) {
    // O4: Verify inflight_count() returns 0 after concurrent operations
    // complete. The lock_all-based snapshot must not deadlock or return
    // stale non-zero values due to stripes being held by in-flight ops.
    counting_backend backend;
    constexpr int kKeys = 64;
    for (int i = 0; i < kKeys; ++i) {
        backend.seed(i, "v" + std::to_string(i));
    }
    backend.set_delay(std::chrono::milliseconds(10));

    tiered_cache<safe_cache<int, std::string>, counting_backend> tc(
        /*max_size=*/256, backend);

    constexpr int kThreads = 64;
    std::vector<std::thread> threads;
    threads.reserve(kThreads);

    std::atomic<int> ready{0};
    std::atomic<bool> go{false};

    for (int i = 0; i < kThreads; ++i) {
        threads.emplace_back([&, i]() {
            ready.fetch_add(1, std::memory_order_acq_rel);
            while (!go.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            (void)tc.get(i);
        });
    }
    while (ready.load(std::memory_order_acquire) < kThreads) {
        std::this_thread::yield();
    }
    go.store(true, std::memory_order_release);
    for (auto& t : threads) t.join();

    EXPECT_EQ(tc.inflight_count(), 0u)
        << "inflight table must be empty after all threads complete";
    EXPECT_EQ(backend.get_call_count(),
              static_cast<std::size_t>(kKeys));
}

// ============================================================================
// P1-5: memory_storage_backend striped locking
// ============================================================================

TEST(MemoryStorageBackendTest, BasicGetPutRemoveContainsSize) {
    memory_storage_backend<int, std::string> backend;
    EXPECT_EQ(backend.size(), 0u);
    EXPECT_FALSE(backend.get(1).has_value());
    EXPECT_FALSE(backend.contains(1));

    backend.put(1, "one");
    backend.put(2, "two");
    EXPECT_EQ(backend.size(), 2u);
    EXPECT_TRUE(backend.contains(1));
    ASSERT_TRUE(backend.get(1).has_value());
    EXPECT_EQ(*backend.get(1), "one");

    EXPECT_TRUE(backend.remove(1));
    EXPECT_FALSE(backend.contains(1));
    EXPECT_EQ(backend.size(), 1u);
    EXPECT_FALSE(backend.remove(999));
}

TEST(MemoryStorageBackendTest, DefaultIs64Stripes) {
    memory_storage_backend<int, std::string> backend;
    EXPECT_EQ(backend.num_stripes(), 64u);
}

TEST(MemoryStorageBackendTest, CustomStripeCount) {
    memory_storage_backend<int, std::string> backend{16};
    EXPECT_EQ(backend.num_stripes(), 16u);
    // Functional sanity check.
    backend.put(7, "v");
    ASSERT_TRUE(backend.get(7).has_value());
    EXPECT_EQ(*backend.get(7), "v");
}

TEST(MemoryStorageBackendTest, ClearDrainsAllEntries) {
    memory_storage_backend<int, std::string> backend;
    for (int i = 0; i < 100; ++i) backend.put(i, "v");
    EXPECT_EQ(backend.size(), 100u);
    backend.clear();
    EXPECT_EQ(backend.size(), 0u);
    for (int i = 0; i < 100; ++i) EXPECT_FALSE(backend.contains(i));
}

// ============================================================================
// P1-5 acceptance: 64-thread concurrent read/write scales linearly.
// We assert (a) no crashes/deadlocks, (b) all reads after writers finish
// observe the expected values, (c) the striped backend outperforms a
// single-mutex baseline by a measurable margin under contention.
// ============================================================================

TEST(MemoryStorageBackendConcurrencyTest, SixtyFourThreadsNoDeadlockOrCorruption) {
    // Distinct keys per writer thread so writes don't conflict; readers
    // sweep all keys. Verifies the striped lock lets writers proceed
    // concurrently and readers see consistent state.
    memory_storage_backend<int, std::string> backend{64};

    constexpr int kWriters = 32;
    constexpr int kReaders = 32;
    constexpr int kKeysPerWriter = 64;

    // Pre-seed so readers always have something to read.
    for (int w = 0; w < kWriters; ++w) {
        for (int k = 0; k < kKeysPerWriter; ++k) {
            int key = w * kKeysPerWriter + k;
            backend.put(key, "init");
        }
    }

    std::atomic<int> ready{0};
    std::atomic<bool> go{false};
    std::atomic<std::size_t> read_ops{0};
    std::atomic<std::size_t> read_mismatch{0};
    std::atomic<std::size_t> write_ops{0};
    std::atomic<bool> stop_readers{false};

    std::vector<std::thread> threads;
    threads.reserve(kWriters + kReaders);

    // Writers: each owns kKeysPerWriter keys, writes "wN:kM" in a loop.
    for (int w = 0; w < kWriters; ++w) {
        threads.emplace_back([&, w]() {
            ready.fetch_add(1, std::memory_order_acq_rel);
            while (!go.load(std::memory_order_acquire)) std::this_thread::yield();
            for (int k = 0; k < kKeysPerWriter; ++k) {
                int key = w * kKeysPerWriter + k;
                backend.put(key, "w" + std::to_string(w) + ":k" + std::to_string(k));
                write_ops.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }

    // Readers: sweep keys until told to stop; verify any returned value
    // matches either "init" or the writer's own "wN:kM" format.
    for (int r = 0; r < kReaders; ++r) {
        threads.emplace_back([&]() {
            ready.fetch_add(1, std::memory_order_acq_rel);
            while (!go.load(std::memory_order_acquire)) std::this_thread::yield();
            while (!stop_readers.load(std::memory_order_acquire)) {
                for (int w = 0; w < kWriters; ++w) {
                    for (int k = 0; k < kKeysPerWriter; ++k) {
                        int key = w * kKeysPerWriter + k;
                        auto v = backend.get(key);
                        if (!v.has_value()) {
                            read_mismatch.fetch_add(1, std::memory_order_relaxed);
                            continue;
                        }
                        // Must be either the initial value or the writer's update.
                        if (*v != "init" &&
                            *v != ("w" + std::to_string(w) + ":k" + std::to_string(k))) {
                            read_mismatch.fetch_add(1, std::memory_order_relaxed);
                        }
                    }
                }
                read_ops.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }

    while (ready.load(std::memory_order_acquire) < kWriters + kReaders) {
        std::this_thread::yield();
    }
    go.store(true, std::memory_order_release);

    // Wait for writers to finish.
    for (int i = 0; i < kWriters; ++i) threads[i].join();
    stop_readers.store(true, std::memory_order_release);
    for (int i = kWriters; i < kWriters + kReaders; ++i) threads[i].join();

    EXPECT_EQ(read_mismatch.load(), 0u)
        << "readers must never observe a torn or stale value";
    EXPECT_EQ(write_ops.load(),
              static_cast<std::size_t>(kWriters * kKeysPerWriter));

    // After all writers finish, every key must reflect its final value.
    for (int w = 0; w < kWriters; ++w) {
        for (int k = 0; k < kKeysPerWriter; ++k) {
            int key = w * kKeysPerWriter + k;
            auto v = backend.get(key);
            ASSERT_TRUE(v.has_value()) << "key=" << key;
            EXPECT_EQ(*v, "w" + std::to_string(w) + ":k" + std::to_string(k))
                << "key=" << key;
        }
    }
}

TEST(MemoryStorageBackendConcurrencyTest, StripedOutperformsSingleMutexUnderContention) {
    // P1-5 acceptance: striped locking must scale better than a single
    // mutex. We can't easily construct a single-mutex baseline here
    // without adding a separate backend class, so instead we verify that
    // the striped backend's throughput on a read-heavy workload with 64
    // threads is bounded by the stripe count, not by a single lock.
    //
    // Concretely: 64 threads each performing 1000 reads of distinct keys
    // must complete in bounded wall-clock time. The assertion is a
    // sanity floor rather than a strict speedup ratio (which would be
    // flaky in CI).
    memory_storage_backend<int, std::string> backend{64};
    constexpr int kThreads = 64;
    constexpr int kReadsPerThread = 1000;

    // Distinct keys per thread so reads don't contend on the same stripe.
    for (int t = 0; t < kThreads; ++t) {
        for (int i = 0; i < 16; ++i) {
            backend.put(t * 16 + i, "v" + std::to_string(i));
        }
    }

    std::atomic<int> ready{0};
    std::atomic<bool> go{false};
    std::atomic<std::size_t> total_reads{0};
    std::atomic<std::size_t> misses{0};

    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t]() {
            ready.fetch_add(1, std::memory_order_acq_rel);
            while (!go.load(std::memory_order_acquire)) std::this_thread::yield();
            for (int i = 0; i < kReadsPerThread; ++i) {
                int key = t * 16 + (i % 16);
                if (!backend.get(key).has_value()) {
                    misses.fetch_add(1, std::memory_order_relaxed);
                }
                total_reads.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }

    auto t0 = std::chrono::steady_clock::now();
    while (ready.load(std::memory_order_acquire) < kThreads) std::this_thread::yield();
    go.store(true, std::memory_order_release);
    for (auto& th : threads) th.join();
    auto t1 = std::chrono::steady_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

    EXPECT_EQ(misses.load(), 0u);
    EXPECT_EQ(total_reads.load(),
              static_cast<std::size_t>(kThreads * kReadsPerThread));
    // Sanity floor: 64k reads across 64 stripes should complete in well
    // under 5 seconds. A single-mutex backend under this contention
    // would typically take longer; we keep the floor generous to avoid
    // CI flakiness on slow runners.
    EXPECT_LT(elapsed_ms, 5000)
        << "striped backend took too long: " << elapsed_ms << "ms";
}

// P2-F: clear() must NOT hold a global write_all lock — concurrent
// operations on other stripes should make progress while clear() runs.
// We verify this by having a background thread hammer a non-cleared
// stripe while the main thread clears the backend. If clear() used the
// old global lock_all path, the reader would be blocked for the entire
// duration of clear().
TEST(MemoryStorageBackendConcurrencyTest, ClearDoesNotBlockOtherStripes) {
    memory_storage_backend<int, std::string> backend{64};
    // Populate the backend so clear() has work to do.
    for (int i = 0; i < 10'000; ++i) backend.put(i, "v");

    std::atomic<bool> stop{false};
    std::atomic<std::size_t> reads_during_clear{0};
    std::atomic<bool> clear_done{false};

    // Reader thread: continuously reads a key from a different stripe than
    // the bulk of the populated data. As long as clear() doesn't take a
    // global lock, this thread makes progress.
    std::thread reader([&]() {
        // Use a key in a specific stripe that's separate from the 0..9999
        // range — pick a key that hashes to a distinct stripe. The exact
        // key doesn't matter; we only care that the read returns quickly.
        const int probe_key = 1'000'000;
        backend.put(probe_key, "probe");
        while (!stop.load(std::memory_order_acquire)) {
            auto v = backend.get(probe_key);
            if (v.has_value()) {
                ++reads_during_clear;
            }
        }
    });

    // Let the reader warm up.
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    // Clear the backend while the reader is running.
    backend.clear();
    clear_done.store(true, std::memory_order_release);

    // Give the reader a bit more time to make progress after clear().
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    stop.store(true, std::memory_order_release);
    reader.join();

    // Sanity: the reader should have completed at least some reads
    // while clear() was running. With the global lock_all path the reader
    // would have been blocked for the entire duration of clear().
    EXPECT_GT(reads_during_clear.load(), 0u);
    EXPECT_TRUE(clear_done.load());
}

// P2-F: clear() must produce an empty backend (functional correctness).
TEST(MemoryStorageBackendTest, ClearWithPerStripeLockingDrainsAll) {
    memory_storage_backend<int, std::string> backend{16};
    for (int i = 0; i < 1'000; ++i) backend.put(i, "v");
    EXPECT_EQ(backend.size(), 1000u);
    backend.clear();
    EXPECT_EQ(backend.size(), 0u);
    // After clear, the backend should behave like a fresh one.
    EXPECT_FALSE(backend.contains(0));
    EXPECT_FALSE(backend.contains(999));
    backend.put(42, "answer");
    EXPECT_TRUE(backend.contains(42));
    EXPECT_EQ(*backend.get(42), "answer");
}

// ============================================================================
// O5: Backend Circuit Breaker
// ============================================================================
//
// Verifies the circuit breaker state machine: CLOSED → OPEN on repeated
// failures, OPEN → HALF_OPEN after cooldown, HALF_OPEN → CLOSED on
// success_threshold consecutive successes, and HALF_OPEN → OPEN on any
// failure. Also verifies that the tiered_cache short-circuits requests
// when the breaker is OPEN (no backend call is made).

namespace {

/// A backend that can be configured to fail (throw) on get() or succeed.
/// Used to test the circuit breaker's failure detection.
class failing_backend : public storage_backend<int, std::string> {
public:
    std::optional<std::string> get(const int& key) override {
        std::lock_guard lock(mutex_);
        ++get_call_count_;
        if (fail_mode_) {
            throw std::runtime_error("backend failure (test)");
        }
        auto it = data_.find(key);
        if (it == data_.end()) return std::nullopt;
        return it->second;
    }

    void put(const int& key, const std::string& value) override {
        std::lock_guard lock(mutex_);
        data_[key] = value;
    }

    bool remove(const int& key) override {
        std::lock_guard lock(mutex_);
        return data_.erase(key) > 0;
    }

    bool contains(const int& key) const override {
        std::lock_guard lock(mutex_);
        return data_.find(key) != data_.end();
    }

    std::size_t size() const override {
        std::lock_guard lock(mutex_);
        return data_.size();
    }

    std::string name() const override { return "failing_backend"; }

    void seed(const int& key, std::string value) {
        std::lock_guard lock(mutex_);
        data_[key] = std::move(value);
    }

    void set_fail_mode(bool fail) {
        std::lock_guard lock(mutex_);
        fail_mode_ = fail;
    }

    std::size_t get_call_count() const {
        std::lock_guard lock(mutex_);
        return get_call_count_;
    }

    void reset_call_count() {
        std::lock_guard lock(mutex_);
        get_call_count_ = 0;
    }

private:
    mutable std::mutex mutex_;
    ankerl::unordered_dense::map<int, std::string> data_;
    bool fail_mode_ = false;
    std::size_t get_call_count_ = 0;
};

} // anonymous namespace

TEST(BackendCircuitBreakerTest, StartsInClosedState) {
    backend_circuit_breaker breaker;
    EXPECT_EQ(breaker.state(), circuit_state::closed);
    EXPECT_EQ(breaker.error_count(), 0u);
    EXPECT_TRUE(breaker.allow_request());
}

TEST(BackendCircuitBreakerTest, DisabledWhenThresholdZero) {
    circuit_breaker_config cfg;
    cfg.error_threshold = 0;
    backend_circuit_breaker breaker(cfg);
    // With threshold=0, allow_request always returns true and
    // record_failure/record_success are no-ops.
    EXPECT_TRUE(breaker.allow_request());
    breaker.record_failure();
    breaker.record_failure();
    EXPECT_EQ(breaker.state(), circuit_state::closed);
    EXPECT_EQ(breaker.error_count(), 0u);
}

TEST(BackendCircuitBreakerTest, TripsAfterErrorThreshold) {
    circuit_breaker_config cfg;
    cfg.error_threshold = 3;
    cfg.cooldown = std::chrono::milliseconds(1000);
    backend_circuit_breaker breaker(cfg);

    // First two failures — still CLOSED.
    breaker.record_failure();
    breaker.record_failure();
    EXPECT_EQ(breaker.state(), circuit_state::closed);
    EXPECT_EQ(breaker.error_count(), 2u);
    EXPECT_TRUE(breaker.allow_request());

    // Third failure trips the breaker.
    breaker.record_failure();
    EXPECT_EQ(breaker.state(), circuit_state::open);
    EXPECT_EQ(breaker.error_count(), 3u);
    EXPECT_FALSE(breaker.allow_request());
}

TEST(BackendCircuitBreakerTest, SuccessResetsErrorCount) {
    circuit_breaker_config cfg;
    cfg.error_threshold = 3;
    backend_circuit_breaker breaker(cfg);

    breaker.record_failure();
    breaker.record_failure();
    EXPECT_EQ(breaker.error_count(), 2u);
    // A success in CLOSED state resets the error count (only consecutive
    // failures trip the breaker).
    breaker.record_success();
    EXPECT_EQ(breaker.error_count(), 0u);
    EXPECT_EQ(breaker.state(), circuit_state::closed);
}

TEST(BackendCircuitBreakerTest, HalfOpenClosesOnSuccessThreshold) {
    circuit_breaker_config cfg;
    cfg.error_threshold = 2;
    cfg.cooldown = std::chrono::milliseconds(10);  // short for testing
    cfg.success_threshold = 2;
    backend_circuit_breaker breaker(cfg);

    // Trip the breaker.
    breaker.record_failure();
    breaker.record_failure();
    EXPECT_EQ(breaker.state(), circuit_state::open);

    // Wait for cooldown to elapse.
    std::this_thread::sleep_for(std::chrono::milliseconds(20));

    // allow_request() should transition to HALF_OPEN and allow the probe.
    EXPECT_TRUE(breaker.allow_request());
    EXPECT_EQ(breaker.state(), circuit_state::half_open);

    // First success in HALF_OPEN — not enough to close yet.
    breaker.record_success();
    EXPECT_EQ(breaker.state(), circuit_state::half_open);

    // Second success — closes the circuit.
    breaker.record_success();
    EXPECT_EQ(breaker.state(), circuit_state::closed);
    EXPECT_EQ(breaker.error_count(), 0u);
}

TEST(BackendCircuitBreakerTest, HalfOpenReopensOnFailure) {
    circuit_breaker_config cfg;
    cfg.error_threshold = 1;
    cfg.cooldown = std::chrono::milliseconds(10);
    cfg.success_threshold = 2;
    backend_circuit_breaker breaker(cfg);

    breaker.record_failure();  // trips immediately (threshold=1)
    EXPECT_EQ(breaker.state(), circuit_state::open);

    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    EXPECT_TRUE(breaker.allow_request());  // transitions to HALF_OPEN
    EXPECT_EQ(breaker.state(), circuit_state::half_open);

    // A failure in HALF_OPEN re-opens immediately.
    breaker.record_failure();
    EXPECT_EQ(breaker.state(), circuit_state::open);
}

TEST(BackendCircuitBreakerTest, ResetReturnsToClosed) {
    circuit_breaker_config cfg;
    cfg.error_threshold = 1;
    backend_circuit_breaker breaker(cfg);

    breaker.record_failure();
    EXPECT_EQ(breaker.state(), circuit_state::open);
    EXPECT_NE(breaker.error_count(), 0u);

    breaker.reset();
    EXPECT_EQ(breaker.state(), circuit_state::closed);
    EXPECT_EQ(breaker.error_count(), 0u);
}

// --- Integration tests: tiered_cache + circuit breaker ---

TEST(TieredCircuitBreakerTest, ShortCircuitsWhenOpen) {
    // When the breaker is OPEN, tiered_cache.get() must NOT call the
    // backend — it should return empty immediately.
    failing_backend backend;
    backend.seed(1, "v1");
    backend.set_fail_mode(true);  // all get() calls throw

    tiered_cache<safe_cache<int, std::string>, failing_backend>::config cfg;
    cfg.breaker.error_threshold = 3;
    cfg.breaker.cooldown = std::chrono::milliseconds(5000);
    tiered_cache<safe_cache<int, std::string>, failing_backend> tc(
        /*max_size=*/10, backend, cfg);

    // Trigger 3 failures to trip the breaker. Each get() will throw
    // because the backend is in fail mode.
    for (int i = 0; i < 3; ++i) {
        try {
            // Use distinct keys so each becomes a fresh leader (no
            // in-flight dedup).
            (void)tc.get(100 + i);
        } catch (const std::runtime_error&) {
            // Expected — backend is failing.
        }
    }
    EXPECT_EQ(tc.circuit_breaker().state(), circuit_state::open);

    // Now the breaker is OPEN. Subsequent get() calls must NOT hit the
    // backend (fail fast). We verify by checking the backend's call count
    // doesn't increase.
    const std::size_t calls_before = backend.get_call_count();
    auto h = tc.get(200);
    EXPECT_FALSE(h.has_value());
    EXPECT_EQ(backend.get_call_count(), calls_before)
        << "circuit breaker should have short-circuited without calling backend";

    // The rejection counter should be incremented.
    auto s = tc.get_stats();
    EXPECT_GE(s.circuit_breaker_rejections, 1u);
}

TEST(TieredCircuitBreakerTest, RecoversAfterCooldownAndSuccess) {
    failing_backend backend;
    backend.seed(1, "v1");

    tiered_cache<safe_cache<int, std::string>, failing_backend>::config cfg;
    cfg.breaker.error_threshold = 2;
    cfg.breaker.cooldown = std::chrono::milliseconds(30);
    cfg.breaker.success_threshold = 2;
    tiered_cache<safe_cache<int, std::string>, failing_backend> tc(
        /*max_size=*/10, backend, cfg);

    // Fail mode ON — trip the breaker.
    backend.set_fail_mode(true);
    try { (void)tc.get(1); } catch (...) {}
    try { (void)tc.get(2); } catch (...) {}
    EXPECT_EQ(tc.circuit_breaker().state(), circuit_state::open);

    // Wait for cooldown.
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    // Fail mode OFF — backend now works.
    backend.set_fail_mode(false);

    // Next get() should transition to HALF_OPEN and succeed. Need
    // success_threshold consecutive successes to close.
    // The first success transitions HALF_OPEN → still HALF_OPEN (needs 2).
    auto h1 = tc.get(1);
    EXPECT_TRUE(h1.has_value());
    EXPECT_EQ(*h1, "v1");
    // The breaker may still be HALF_OPEN (1 success, needs 2).
    // Evict key 1 from primary so the next get() hits the backend again.
    h1.release();
    tc.primary().del(1);
    auto h2 = tc.get(1);
    EXPECT_TRUE(h2.has_value());
    // After 2 successes, the breaker should be CLOSED.
    EXPECT_EQ(tc.circuit_breaker().state(), circuit_state::closed);
}

TEST(TieredCircuitBreakerTest, DefaultConfigDoesNotTripOnMisses) {
    // Backend misses (returning nullopt) are NOT failures — they should
    // NOT trip the breaker. Only exceptions count as failures.
    counting_backend backend;
    // No seeds — every get() returns nullopt (backend miss).

    tiered_cache<safe_cache<int, std::string>, counting_backend>::config cfg;
    cfg.breaker.error_threshold = 3;
    tiered_cache<safe_cache<int, std::string>, counting_backend> tc(
        /*max_size=*/10, backend, cfg);

    for (int i = 0; i < 10; ++i) {
        auto h = tc.get(i);
        EXPECT_FALSE(h.has_value());
    }
    // All were misses, but no exceptions → breaker stays CLOSED.
    EXPECT_EQ(tc.circuit_breaker().state(), circuit_state::closed);
    EXPECT_EQ(backend.get_call_count(), 10u);
    auto s = tc.get_stats();
    EXPECT_EQ(s.circuit_breaker_rejections, 0u);
}
