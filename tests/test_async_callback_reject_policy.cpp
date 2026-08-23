// SPDX-License-Identifier: MIT
// G22 regression test: G7 fix — async callback kReject preserves prior events.
//
// Before the G7 fix, when the async callback queue was full and the overflow
// policy was kReject, callback_manager::flush_pending() called
// async_queue_.clear(), which destroyed ALL previously-enqueued events that
// the background worker had not yet dispatched. The rejected batch itself
// was still dispatched synchronously, but the prior events were silently
// lost.
//
// The fix (core.hpp lines 694-699) records the queue size before enqueuing
// the current batch and, on kReject, resizes back to that size
// (async_queue_.resize(queue_size_before)), preserving prior events for the
// worker to dispatch.
//
// Test strategy (deterministic):
//   1. Block the async worker's first callback dispatch so it cannot drain
//      the queue while we fill it.
//   2. Enqueue 2 events (filling the queue to max_size=2).
//   3. Trigger kReject with a 3rd event — this falls through to synchronous
//      dispatch for the 3rd event.
//   4. Release the worker. It drains the 2 preserved events.
//   5. Verify all 3 callbacks fired (count == 3).
//
// Without the fix, step 4 would find an empty queue (clear() destroyed the
// 2 prior events), so only 2 callbacks would fire (1 from the worker's
// initial batch + 1 from sync dispatch), not 3.

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include "../lru.hpp"

using namespace lru;
using namespace std::chrono_literals;

// ============================================================================
// G7: kReject preserves previously-enqueued events
// ============================================================================
TEST(AsyncCallbackRejectPolicy, kRejectPreservesPriorEvents) {
    cache<int, std::string> c(100);
    c.set_defer_promotion(false);  // ensure collect_hit/insert fires

    // Latch to block the async worker after it dequeues the first event.
    std::mutex block_mtx;
    std::condition_variable block_cv;
    std::atomic<bool> release_worker{false};
    std::atomic<bool> worker_blocked{false};

    std::atomic<int> insert_count{0};

    c.on_insert([&](const int&, const std::string&) {
        int n = insert_count.fetch_add(1, std::memory_order_acq_rel);
        if (n == 0) {
            // First callback invocation — this is the worker dispatching
            // the first event. Block here so the worker cannot return to
            // drain the queue.
            std::unique_lock<std::mutex> lk(block_mtx);
            worker_blocked.store(true, std::memory_order_release);
            block_cv.wait(lk, [&release_worker] {
                return release_worker.load(std::memory_order_acquire);
            });
        }
    });

    // Configure async queue: max_size=2, kReject policy.
    auto& mgr = c.callbacks();
    mgr.set_async_queue_max_size(2);
    mgr.set_async_overflow_policy(
        cache<int, std::string>::callback_mgr::overflow_policy::kReject);
    c.set_async_callbacks(true);

    // Insert key 1 → flush_pending enqueues event [1] → worker wakes,
    // dequeues [1], and blocks in the callback (first invocation).
    c.set(1, "a");

    // Wait until the worker has entered the blocked state.
    for (int i = 0; i < 200 && !worker_blocked.load(std::memory_order_acquire); ++i) {
        std::this_thread::sleep_for(10ms);
    }
    ASSERT_TRUE(worker_blocked.load(std::memory_order_acquire))
        << "Worker did not start dispatching within 2s";

    // Confirm the worker is actually inside cv.wait (mutex released) by
    // briefly acquiring block_mtx. If we get the lock, the worker released
    // it in cv.wait — guaranteed blocked.
    {
        std::unique_lock<std::mutex> lk(block_mtx);
        // Worker is now blocked in cv.wait; release immediately.
    }

    // The worker has dequeued [1] into its local batch, so the async queue
    // is empty. Now rapidly fill the queue to max_size and trigger kReject.
    // Each set() calls flush_pending() (via flush_guard on non-striped cache).
    c.set(2, "b");  // enqueue [2] → queue: [2] (size 1, max 2)
    c.set(3, "c");  // enqueue [3] → queue: [2, 3] (size 2, max 2 — full)
    c.set(4, "d");  // try enqueue [4] → queue full → kReject.
                    // FIX: resize(2) → queue stays [2, 3]; [4] sync-dispatched.
                    // BUG: clear()     → queue [];      [4] sync-dispatched.
                    // Sync dispatch of [4] calls on_insert (2nd invocation,
                    // does not block).

    // At this point insert_count should be 2: [1] from worker (blocked but
    // already incremented) + [4] from sync dispatch.
    EXPECT_EQ(insert_count.load(), 2);

    // Release the blocked worker so it can finish [1] and drain the queue.
    release_worker.store(true, std::memory_order_release);
    block_cv.notify_one();

    // set_async_callbacks(false) stops the worker and synchronously drains
    // any remaining events in the async queue.
    c.set_async_callbacks(false);

    // G7 key assertion: all 4 insert callbacks must have fired.
    //   [1] — worker's first batch (was blocked, now released)
    //   [2] — worker's second batch (preserved by resize, not cleared)
    //   [3] — worker's second batch (preserved by resize, not cleared)
    //   [4] — synchronous dispatch (kReject fallback)
    // With the bug (clear), [2] and [3] would be lost → count == 2.
    EXPECT_EQ(insert_count.load(), 4)
        << "Prior events were lost on kReject (G7 regression)";
}

// ============================================================================
// G7: kReject with empty queue does not lose the rejected batch (sync fallback)
// ============================================================================
TEST(AsyncCallbackRejectPolicy, kRejectEmptyQueueSyncFallback) {
    cache<int, std::string> c(100);
    c.set_defer_promotion(false);

    std::atomic<int> insert_count{0};
    c.on_insert([&](const int&, const std::string&) {
        insert_count.fetch_add(1, std::memory_order_relaxed);
    });

    // max_size=1, kReject — even a single event fills the queue, and a
    // second event triggers kReject with sync fallback.
    auto& mgr = c.callbacks();
    mgr.set_async_queue_max_size(1);
    mgr.set_async_overflow_policy(
        cache<int, std::string>::callback_mgr::overflow_policy::kReject);
    c.set_async_callbacks(true);

    c.set(1, "a");  // enqueue [1] (queue full at size 1)
    c.set(2, "b");  // kReject → [2] sync-dispatched, [1] preserved
    c.set(3, "c");  // kReject → [3] sync-dispatched, [1] preserved

    c.set_async_callbacks(false);  // drain remaining

    // [1] was preserved throughout (never cleared by kReject).
    // [2] and [3] were sync-dispatched.
    // Total: 3 callbacks.
    EXPECT_EQ(insert_count.load(), 3);
}

// ============================================================================
// G7: kDropNewest and kDropOldest do not lose prior events either
// (sanity check — these policies were already correct, but verify the
//  overall event-delivery guarantee holds under overflow.)
// ============================================================================
TEST(AsyncCallbackRejectPolicy, kDropNewestDeliversPriorEvents) {
    // Large capacity so the event loop never triggers eviction callbacks.
    cache<int, std::string> c(1000000);
    c.set_defer_promotion(false);

    std::atomic<int> insert_count{0};
    c.on_insert([&](const int&, const std::string&) {
        insert_count.fetch_add(1, std::memory_order_relaxed);
    });

    auto& mgr = c.callbacks();
    mgr.set_async_queue_max_size(2);
    mgr.set_async_overflow_policy(
        cache<int, std::string>::callback_mgr::overflow_policy::kDropNewest);
    c.set_async_callbacks(true);

    // Produce enough events that the producer deterministically outruns the
    // async worker, so the capacity-2 queue reliably overflows and kDropNewest
    // drops events. With small event counts the worker sometimes keeps up and
    // nothing is dropped (the queue never fills), making dropped_count flaky.
    constexpr int kNumEvents = 100000;
    for (int i = 0; i < kNumEvents; ++i) {
        c.set(i, "v");
    }

    c.set_async_callbacks(false);  // stop worker + drain remaining

    // The worker dispatches the enqueued events asynchronously, so wait
    // (bounded) for it to deliver the first 2 events before asserting.
    for (int waited = 0; waited < 500 && insert_count.load() < 2; ++waited) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    // With kDropNewest, the first 2 events are enqueued and dispatched by
    // the worker; the overflow is dropped (tracked). The 2 enqueued events
    // must be delivered.
    EXPECT_GE(insert_count.load(), 2)
        << "kDropNewest lost the initial enqueued events";
    EXPECT_GT(mgr.async_queue_dropped_count(), 0u)
        << "Expected some events to be dropped under kDropNewest overflow";
}
