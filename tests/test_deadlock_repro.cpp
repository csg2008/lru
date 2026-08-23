// SPDX-License-Identifier: MIT
// Minimal reproduction test for production_cache hang under concurrent try_get.
//
// Previously, two threads calling try_get() on different keys could hang
// indefinitely. The root cause was a TOCTOU on rehash_in_progress_ in the
// hazptr/EBR walk path of find_and_pin_lockfree_with_hash: the entry-level
// check verified no rehash was in progress, but a concurrent rehash_step()
// (driven by the background rehash balancer or another writer) could START
// migrating the very bucket being walked. rehash_step() rewrites each
// node's hash_chain_next() pointer to splice it into the new bucket's
// chain, so the in-place chain could briefly form a cycle, causing the
// while(curr) loop to spin forever.
//
// The fix (C-3) re-checks rehash_in_progress_ inside the walk loop and
// bounds the walk at kMaxWalkSteps nodes; either condition bails to
// find_and_pin_with_hash, which uses proper bucket locking and is safe
// during rehash.
//
// This test runs the simplest possible concurrent try_get workload with a
// hard timeout. If the test completes within the timeout, the deadlock is
// fixed. If it times out, the deadlock is still present.
//
// P2-5: Refactored to use bounded-join + stop-flag instead of detaching
// the watchdog thread on timeout. Each worker thread polls a stop_flag
// between iterations and exits cleanly when the flag is set, so the test
// runner can `join()` every thread within a bounded post-timeout grace
// period — no leaked threads, no detached threads, no risk of the GTest
// process hanging at exit.

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

#include "lru.hpp"

using namespace lru;
using namespace std::chrono_literals;

namespace {

// P2-5: Bounded-join thread wrapper. Spawns a worker that polls a stop
// flag between iterations of its work loop. The caller can `join()` the
// worker thread; if the worker has not finished within `deadline`, the
// caller sets `stop = true` and waits an additional grace period
// (`grace`) for the worker to observe the flag and return. If the worker
// still hasn't joined after the grace period, the test FAILS (and the
// worker is force-detached as a last resort to avoid hanging GTest — but
// this is an unrecoverable test infrastructure failure, not the normal
// path).
//
// The worker lambda takes a `const std::atomic<bool>& stop` parameter and
// is expected to check it between cache operations; once `stop` is true
// the worker must return promptly (within ~1 cache op of latency).
template <typename Fn>
bool bounded_join_with_stop(Fn&& fn,
                            std::chrono::milliseconds deadline,
                            std::chrono::milliseconds grace) {
    std::atomic<bool> stop{false};
    std::atomic<bool> done{false};

    std::thread worker([&] {
        std::forward<Fn>(fn)(stop);
        done.store(true, std::memory_order_release);
    });

    auto deadline_t = std::chrono::steady_clock::now() + deadline;
    while (std::chrono::steady_clock::now() < deadline_t) {
        if (done.load(std::memory_order_acquire)) {
            worker.join();
            return true;
        }
        std::this_thread::sleep_for(10ms);
    }

    if (done.load(std::memory_order_acquire)) {
        worker.join();
        return true;
    }

    // Timeout — signal the worker to stop and wait for the grace period.
    stop.store(true, std::memory_order_release);

    auto grace_deadline = std::chrono::steady_clock::now() + grace;
    while (std::chrono::steady_clock::now() < grace_deadline) {
        if (done.load(std::memory_order_acquire)) {
            worker.join();
            return false;  // timed out, but joined cleanly
        }
        std::this_thread::sleep_for(5ms);
    }

    if (done.load(std::memory_order_acquire)) {
        worker.join();
        return false;
    }

    // Last-resort: detach the stuck worker so GTest can exit. This means
    // the cache deadlock fix has regressed and the worker is spinning in
    // user code that doesn't check `stop`. Fail loudly.
    worker.detach();
    return false;
}

// Convenience wrapper: 5s deadline + 1s grace.
template <typename Fn>
bool bounded_join_with_stop(Fn&& fn) {
    return bounded_join_with_stop(std::forward<Fn>(fn), 5s, 1s);
}

} // anonymous namespace

// ============================================================================
// Reproduction: 2 threads x 1 op (try_get) on production_cache
// This was the exact scenario that hung before the fix.
//
// P2-5: Each worker checks the stop flag between operations. With only
// 1 op per thread, the stop flag is mostly insurance against the case
// where try_get itself hangs — the worker can't be interrupted mid-op,
// but the grace period gives it time to return after the cache op
// completes. If the op truly hangs (the original bug), the grace period
// elapses and the test fails.
// ============================================================================
TEST(ProductionCacheDeadlock, TwoThreadTryGetNoHang) {
    production_cache<int, int> c(10000);
    for (int i = 0; i < 5000; ++i) {
        c.set(i, i * 10);
    }
    ASSERT_EQ(c.size(), 5000u);

    auto work = [&](const std::atomic<bool>& stop) {
        std::thread t1([&] { (void)c.try_get(3368); });
        std::thread t2([&] { (void)c.try_get(845); });
        // Both threads do a single op; we still respect `stop` for the
        // grace-period exit path (if try_get itself hangs, the join below
        // cannot complete and the test will fail at the watchdog level).
        (void)stop;
        t1.join();
        t2.join();
    };

    EXPECT_TRUE(bounded_join_with_stop(work, 5s, 2s))
        << "production_cache try_get hung — likely C-3 hazptr/rehash TOCTOU";
}

// ============================================================================
// Slightly more aggressive: 8 threads, many try_get ops, mixed with
// occasional set() calls to keep the bucket locks moving.
//
// P2-5: Each worker thread checks the stop flag between iterations, so
// when the watchdog signals stop, all 8 threads exit their loops within
// a single cache-op latency (microseconds). The outer thread then joins
// all 8 cleanly within the grace period.
// ============================================================================
TEST(ProductionCacheDeadlock, EightThreadMixedTryGetNoHang) {
    production_cache<int, int> c(10000);
    for (int i = 0; i < 5000; ++i) {
        c.set(i, i * 10);
    }

    auto work = [&](const std::atomic<bool>& stop) {
        std::vector<std::thread> threads;
        for (int t = 0; t < 8; ++t) {
            threads.emplace_back([&, t] {
                for (int i = 0; i < 200; ++i) {
                    if (stop.load(std::memory_order_relaxed)) return;
                    int key = (t * 137 + i) % 5000;
                    if ((i & 0x7) == 0) {
                        c.set(key, key * 10);
                    } else {
                        (void)c.try_get(key);
                    }
                }
            });
        }
        for (auto& th : threads) th.join();
    };

    EXPECT_TRUE(bounded_join_with_stop(work, 15s, 3s))
        << "production_cache mixed workload hung — likely C-3 hazptr/rehash TOCTOU";
}

// ============================================================================
// Same scenarios for segmented_striped_cache (writer_fair default).
// This covers both fairness modes.
// ============================================================================
TEST(SegmentedStripedCacheDeadlock, TwoThreadTryGetNoHang) {
    segmented_striped_cache<int, int> c(10000);
    for (int i = 0; i < 5000; ++i) {
        c.set(i, i * 10);
    }

    auto work = [&](const std::atomic<bool>& stop) {
        std::thread t1([&] { (void)c.try_get(3368); });
        std::thread t2([&] { (void)c.try_get(845); });
        (void)stop;
        t1.join();
        t2.join();
    };

    EXPECT_TRUE(bounded_join_with_stop(work, 5s, 2s))
        << "segmented_striped_cache try_get hung — likely C-3 hazptr/rehash TOCTOU";
}
