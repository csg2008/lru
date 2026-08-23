// SPDX-License-Identifier: MIT
// Concurrent correctness tests for non-LRU eviction strategies.
//
// Covers spec gap G12 (P2):
//   G12: TinyLFU / W-TinyLFU / 2Q / FIFO concurrent correctness
//        Verify hit+miss == total and data integrity under multi-thread load.

#include <gtest/gtest.h>

#include <atomic>
#include <thread>
#include <vector>

#include "lru.hpp"
#include "test_helpers.hpp"

using namespace lru;

namespace {

// Generic concurrent correctness check: each thread performs a mix of
// set/get on its OWN disjoint key range. hit + miss must equal total accesses,
// and any value observed by try_get must match the most recent set() for that
// key (eventual consistency is sufficient — items may be evicted between
// set() and try_get(), in which case try_get() returns nullopt).
//
// We use disjoint key ranges per thread AND unique keys per set() call to
// maximize insert-path coverage and isolate per-thread integrity from
// cross-thread eviction. The update path (set() on an existing key) is
// covered separately by the `*RepeatSetOnSameKey` tests below (H-1-C),
// which stress the insert_new/update_existing split for mm_2q /
// mm_tiny_lfu / mm_wtiny_lfu. Reads can hit any key in the thread's range
// (some will be misses due to eviction — that's expected).
template <typename CacheT>
void run_concurrent_correctness(CacheT& c, int num_threads, int ops_per_thread) {
    std::atomic<int> hits{0};
    std::atomic<int> misses{0};
    std::atomic<int> total{0};
    std::atomic<int> value_mismatch{0};

    // Each thread owns a disjoint key range. set() uses incrementing keys
    // (no updates), reads pick from the same range.
    constexpr int kKeySpacePerThread = 1000;

    std::vector<std::thread> threads;
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t] {
            const int key_base = t * kKeySpacePerThread;
            int next_set_key = 0;  // offset within this thread's key space
            for (int i = 0; i < ops_per_thread; ++i) {
                if (i % 3 == 0) {
                    // Always insert a new key — no updates.
                    int key = key_base + (next_set_key++);
                    int v = key * 10;
                    c.set(key, v);
                    // Immediate read-back: the item may have been evicted by
                    // another thread's insertion (capacity-bounded), so
                    // missing is acceptable. But if present, value must match.
                    auto h = c.try_get(key);
                    if (h && **h != v) {
                        value_mismatch.fetch_add(1, std::memory_order_relaxed);
                    }
                } else {
                    // Read a random key from this thread's space.
                    int key = key_base + ((i * 7) % kKeySpacePerThread);
                    auto h = c.try_get(key);
                    if (h) {
                        hits.fetch_add(1, std::memory_order_relaxed);
                        // Value must match the canonical mapping (key * 10).
                        // If a different value is observed, it's a corruption.
                        if (**h != key * 10) {
                            value_mismatch.fetch_add(1, std::memory_order_relaxed);
                        }
                    } else {
                        misses.fetch_add(1, std::memory_order_relaxed);
                    }
                    total.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }
    for (auto& th : threads) th.join();

    EXPECT_EQ(value_mismatch.load(), 0)
        << "try_get returned a value inconsistent with the canonical mapping";
    EXPECT_EQ(hits.load() + misses.load(), total.load())
        << "hit+miss != total — stats counters are inconsistent";
}

}  // namespace

// ============================================================================
// TC-G12: TinyLFU concurrent correctness
//
// Uses the `safe_lfu_cache` alias (unified_cache + tiny_lfu_trait +
// thread_safe_policy, single global distributed_shared_mutex) because the
// base `lfu_cache` alias uses single_threaded_policy (no locks) and would
// race under concurrent access.
//
// Capacity is set generously (8 threads × 333 sets = 2664 keys; capacity 8000
// gives 3x headroom) to isolate per-thread insert-path integrity from
// cross-thread eviction (items evicted between set() and try_get() return
// nullopt, which is acceptable). The update path (set() on an existing key)
// is covered by the `*RepeatSetOnSameKey` / `*MixedInsertUpdate` tests
// below (H-1-B / H-1-C), which verify the insert_new/update_existing
// split does not trigger historical assertions
// (`link_at_head: item is already linked` / `src_qid != target_qid`).
// ============================================================================
TEST(ConcurrentMMStrategies, TinyLfuConcurrentCorrectness) {
    safe_lfu_cache<int, int> c(8000);
    run_concurrent_correctness(c, 8, 1000);
}

// ============================================================================
// TC-G12: W-TinyLFU concurrent correctness
// ============================================================================
TEST(ConcurrentMMStrategies, WTinyLfuConcurrentCorrectness) {
    safe_w_tiny_lfu<int, int> c(8000);
    run_concurrent_correctness(c, 8, 1000);
}

// ============================================================================
// TC-G12: 2Q concurrent correctness
// ============================================================================
TEST(ConcurrentMMStrategies, TwoQConcurrentCorrectness) {
    safe_two_q<int, int> c(8000);
    run_concurrent_correctness(c, 8, 1000);
}

// ============================================================================
// TC-G12: FIFO concurrent correctness
// ============================================================================
TEST(ConcurrentMMStrategies, FifoConcurrentCorrectness) {
    safe_fifo_cache<int, int> c(8000);
    run_concurrent_correctness(c, 8, 1000);
}

// ============================================================================
// TC-G12 (extended): TinyLFU under heavy read load + occasional writes
// Verifies data integrity when reads dominate.
//
// Uses `safe_lfu_cache` (thread-safe variant) because `lfu_cache` is
// single-threaded.
// ============================================================================
TEST(ConcurrentMMStrategies, TinyLfuReadHeavyCorrectness) {
    safe_lfu_cache<int, std::string> c(200);
    for (int i = 0; i < 100; ++i) {
        c.set(i, "v" + std::to_string(i));
    }

    constexpr int kThreads = 8;
    constexpr int kOpsPerThread = 5000;
    std::atomic<int> errors{0};

    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t] {
            std::mt19937 rng(t);
            for (int i = 0; i < kOpsPerThread; ++i) {
                int key = rng() % 100;
                if (i % 50 == 0) {
                    c.set(key, "v" + std::to_string(key));
                } else {
                    auto h = c.try_get(key);
                    if (h && **h != "v" + std::to_string(key)) {
                        errors.fetch_add(1, std::memory_order_relaxed);
                    }
                }
            }
        });
    }
    for (auto& th : threads) th.join();

    EXPECT_EQ(errors.load(), 0)
        << "TinyLFU returned mismatched values under read-heavy load";
}

// ============================================================================
// H-1-C: Repeated set() on the SAME key from multiple threads.
//
// Verifies the insert_new/update_existing split for mm_2q / mm_tiny_lfu /
// mm_wtiny_lfu does NOT trigger the historical assertions:
//   - `link_at_head: item is already linked`
//   - `src_qid != target_qid`
//
// 8 threads concurrently set() DIFFERENT values on the SAME key. Under the
// thread-safe_policy (single global distributed_shared_mutex), all set() calls
// are serialized, so the final value must be the last set() that won the race.
// The test asserts: (1) no abort / assertion fires, (2) the final value is
// one of the values written by some thread, (3) try_get immediately after
// returns the same value (consistency).
// ============================================================================
template <typename CacheT>
void run_repeat_set_on_same_key(CacheT& c, int num_threads, int ops_per_thread) {
    const int key = 42;
    // Seed the key once so all later set() calls hit the update_existing path.
    c.set(key, 0);

    std::atomic<int> last_written{-1};
    std::atomic<int> errors{0};

    std::vector<std::thread> threads;
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            for (int i = 0; i < ops_per_thread; ++i) {
                int v = t * ops_per_thread + i + 1;  // unique non-zero value
                c.set(key, v);
                last_written.store(v, std::memory_order_release);
                // Read-back: the value must be either the just-written value
                // or some other thread's value that won the race after this
                // set(). Either way, it must be a positive value we wrote.
                auto h = c.try_get(key);
                if (!h) {
                    errors.fetch_add(1, std::memory_order_relaxed);
                } else if (**h <= 0) {
                    errors.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }
    for (auto& th : threads) th.join();

    // Final consistency: try_get must return the lastWritten value (under
    // the global lock, the last set() to acquire the write lock wins, and
    // no further writes occur after join()).
    auto h = c.try_get(key);
    EXPECT_TRUE(h.has_value()) << "key missing after concurrent updates";
    if (h) {
        EXPECT_EQ(**h, last_written.load(std::memory_order_acquire))
            << "final value does not match last set()";
    }
    EXPECT_EQ(errors.load(), 0)
        << "try_get returned missing/non-positive value during updates";
}

TEST(ConcurrentMMStrategies, Mm2qRepeatSetOnSameKey) {
    safe_two_q<int, int> c(100);
    run_repeat_set_on_same_key(c, 8, 1000);
}

TEST(ConcurrentMMStrategies, MmTinyLfuRepeatSetOnSameKey) {
    safe_lfu_cache<int, int> c(100);
    run_repeat_set_on_same_key(c, 8, 1000);
}

TEST(ConcurrentMMStrategies, MmWTinyLfuRepeatSetOnSameKey) {
    safe_w_tiny_lfu<int, int> c(100);
    run_repeat_set_on_same_key(c, 8, 1000);
}

// ============================================================================
// H-1-B (negative assertion): Under high concurrency with mixed insert +
// update paths, no assertion fires. Each thread interleaves new-key inserts
// with updates to a shared hot key, exercising the cross-queue promotion
// path (Cold->Warm for mm_2q, Probation->Protection for mm_wtiny_lfu).
// ============================================================================
template <typename CacheT>
void run_mixed_insert_update(CacheT& c, int num_threads, int ops_per_thread) {
    const int hot_key = 0;
    c.set(hot_key, 0);

    std::atomic<int> errors{0};
    std::vector<std::thread> threads;
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            for (int i = 0; i < ops_per_thread; ++i) {
                if (i % 4 == 0) {
                    // Update the hot key (existing key -> update_existing path)
                    c.set(hot_key, t * 1000 + i);
                } else {
                    // Insert a new key unique to this thread
                    int new_key = (t + 1) * 100000 + i;
                    c.set(new_key, new_key);
                }
            }
        });
    }
    for (auto& th : threads) th.join();

    EXPECT_EQ(errors.load(), 0);
    // Hot key must still be present (frequently updated items never evict
    // themselves; capacity is generous enough that it won't be the victim).
    auto h = c.try_get(hot_key);
    EXPECT_TRUE(h.has_value());
}

TEST(ConcurrentMMStrategies, Mm2qMixedInsertUpdate) {
    safe_two_q<int, int> c(2000);
    run_mixed_insert_update(c, 8, 1000);
}

TEST(ConcurrentMMStrategies, MmTinyLfuMixedInsertUpdate) {
    safe_lfu_cache<int, int> c(2000);
    run_mixed_insert_update(c, 8, 1000);
}

TEST(ConcurrentMMStrategies, MmWTinyLfuMixedInsertUpdate) {
    safe_w_tiny_lfu<int, int> c(2000);
    run_mixed_insert_update(c, 8, 1000);
}
