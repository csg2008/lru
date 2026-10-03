// SPDX-License-Identifier: MIT
// Regression tests for the tiered cache's asynchronous write-back ordering.
//
// A queued write-back is a snapshot of a value the primary cache evicted. By
// the time the worker persists it, the caller may already have set() or del()
// that key through the write-through path. Replaying the stale snapshot then
// reverts the value (set) or resurrects the key (del).
//
// Both tests pin the interleaving deterministically: the backend blocks the
// worker while it persists an unrelated "blocker" key, so the victim entry is
// still pending when the test mutates it. Without the generation/in-flight
// bookkeeping in tiered_cache, the worker's put lands afterwards and the
// assertion fails.

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>

#include "../lru.hpp"

using namespace lru;

namespace {

/// Backend whose put() can be parked for one designated key, so a test can
/// hold the write-back worker inside backend I/O while it mutates the cache.
class latch_backend : public storage_backend<int, std::string> {
public:
    std::optional<std::string> get(const int& key) override {
        std::lock_guard lock_guard(mutex_);
        auto it = data_.find(key);
        if (it == data_.end()) return std::nullopt;
        return it->second;
    }

    void put(const int& key, const std::string& value) override {
        if (block_puts_.load(std::memory_order_acquire) && key == blocker_key_) {
            blocked_.store(true, std::memory_order_release);
            std::unique_lock<std::mutex> lk(gate_mutex_);
            gate_cv_.wait(lk, [this] { return gate_open_; });
        }
        std::lock_guard lock_guard(mutex_);
        data_[key] = value;
    }

    bool remove(const int& key) override {
        std::lock_guard lock_guard(mutex_);
        return data_.erase(key) > 0;
    }

    bool contains(const int& key) const override {
        std::lock_guard lock_guard(mutex_);
        return data_.find(key) != data_.end();
    }

    std::size_t size() const override {
        std::lock_guard lock_guard(mutex_);
        return data_.size();
    }

    std::string name() const override { return "latch_backend"; }

    void arm(int blocker_key) {
        blocker_key_ = blocker_key;
        blocked_.store(false, std::memory_order_release);
        block_puts_.store(true, std::memory_order_release);
    }

    /// True once the worker has entered the parked put().
    bool wait_until_blocked(std::chrono::milliseconds limit) {
        const auto deadline = std::chrono::steady_clock::now() + limit;
        while (std::chrono::steady_clock::now() < deadline) {
            if (blocked_.load(std::memory_order_acquire)) return true;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        return blocked_.load(std::memory_order_acquire);
    }

    void release_gate() {
        {
            std::lock_guard<std::mutex> lk(gate_mutex_);
            gate_open_ = true;
        }
        gate_cv_.notify_all();
    }

private:
    mutable std::mutex mutex_;
    std::unordered_map<int, std::string> data_;

    std::atomic<bool> block_puts_{false};
    std::atomic<bool> blocked_{false};
    int blocker_key_ = -1;
    std::mutex gate_mutex_;
    std::condition_variable gate_cv_;
    bool gate_open_ = false;
};

using test_cache = tiered_cache<safe_cache<int, std::string>, latch_backend>;

test_cache::config async_writeback_config() {
    test_cache::config cfg;
    cfg.write_back_on_evict = true;
    cfg.async_writeback_queue_capacity = 64;
    cfg.async_writeback_interval = std::chrono::milliseconds(1);
    cfg.promotion_interval = std::chrono::milliseconds(0);
    return cfg;
}

/// Spin until the write-back queue drains (or the deadline expires).
bool wait_for_writeback_idle(test_cache& tc, std::chrono::milliseconds limit) {
    const auto deadline = std::chrono::steady_clock::now() + limit;
    while (std::chrono::steady_clock::now() < deadline) {
        if (tc.writeback_queue_size() == 0) {
            // The worker may still hold a batch; give it a moment to finish.
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            if (tc.writeback_queue_size() == 0) return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return false;
}

}  // namespace

// del() must win over a write-back that was enqueued before the delete.
TEST(TieredWriteback, DeletedKeyIsNotResurrectedByPendingWriteback) {
    latch_backend backend;
    test_cache tc(/*max_size=*/2, backend, async_writeback_config());

    constexpr int kBlocker = 1;
    constexpr int kVictim = 2;

    tc.set(kBlocker, "blocker");   // write-through (worker not parked yet)
    tc.set(kVictim, "victim");     // write-through

    // Park the worker inside put(kBlocker) so the victim's write-back stays
    // pending while the test deletes it.
    backend.arm(kBlocker);
    tc.set(3, "third");            // evicts kBlocker (LRU) -> enqueue -> worker parks
    ASSERT_TRUE(backend.wait_until_blocked(std::chrono::seconds(5)))
        << "write-back worker never reached the parked put()";

    tc.set(4, "fourth");           // evicts kVictim -> enqueue (still pending)
    // kVictim is no longer in the primary, so del() reports false; the point
    // is that the backend must end up without it.
    (void)tc.del(kVictim);         // explicit delete

    backend.release_gate();
    ASSERT_TRUE(wait_for_writeback_idle(tc, std::chrono::seconds(5)));

    EXPECT_FALSE(backend.contains(kVictim))
        << "a pending write-back resurrected a key that was deleted";
}

// set() must win over a write-back that was enqueued before the set.
TEST(TieredWriteback, StaleWritebackDoesNotOverwriteNewerValue) {
    latch_backend backend;
    test_cache tc(/*max_size=*/2, backend, async_writeback_config());

    constexpr int kBlocker = 1;
    constexpr int kVictim = 2;

    tc.set(kBlocker, "blocker");
    tc.set(kVictim, "old");

    backend.arm(kBlocker);
    tc.set(3, "third");            // evicts kBlocker -> worker parks
    ASSERT_TRUE(backend.wait_until_blocked(std::chrono::seconds(5)));

    tc.set(4, "fourth");           // evicts kVictim -> enqueue "old"
    tc.set(kVictim, "new");        // write-through: the newest value

    backend.release_gate();
    ASSERT_TRUE(wait_for_writeback_idle(tc, std::chrono::seconds(5)));

    auto got = backend.get(kVictim);
    ASSERT_TRUE(got.has_value());
    EXPECT_EQ(*got, "new")
        << "a stale write-back overwrote a newer value";
}
