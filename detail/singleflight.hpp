// Unified LRU Cache Library — Singleflight / Cache Stampede Protection
// SPDX-License-Identifier: MIT
//
// Prevents cache stampede (thundering herd) when multiple threads concurrently
// miss the same key in get_or_fetch() / try_get_or_fetch(). The first miss
// becomes the "leader" and executes the provider; concurrent misses on the
// same key become "followers" that block on a condition variable until the
// leader completes, then receive the leader's result.
//
// Design:
//   - Sharded tracker (16 shards by default) keyed by std::hash<Key> to avoid
//     single-point contention. Shard count is independent of the cache's stripe
//     count — singleflight is only exercised on the miss path (infrequent),
//     so a smaller shard count suffices.
//   - Per-key state is held in a shared_ptr so the leader can safely remove
//     the map entry while followers still hold a reference to the state.
//   - Leader signals completion via state->complete(); followers wake via CV.
//   - Provider exceptions are propagated to all followers via exception_ptr.
//     Insertion failures (OOM/admission rejection) are also propagated: the
//     leader's get_or_fetch throws cache_oom_exception on admission failure,
//     which the catch-all path forwards to followers so they rethrow it
//     instead of silently using a value that was never cached. The only
//     non-throwing no-cache case is is_memory_critical(), where the value is
//     intentionally returned uncached to all callers alike.
//
// Overhead:
//   - Hit path: zero (singleflight is only consulted after a confirmed miss).
//   - Miss path: one map lookup + one mutex lock per shard, plus shared_ptr
//     refcount ops. Negligible compared to the provider call.

#ifndef LRU_DETAIL_SINGLEFLIGHT_HPP
#define LRU_DETAIL_SINGLEFLIGHT_HPP

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <utility>

#include "ankerl/unordered_dense.h"

namespace lru::detail {

/// Per-key singleflight state, shared between leader and followers via
/// shared_ptr. The leader calls complete(); followers call wait_and_get().
template <typename Value>
struct singleflight_state {
    std::mutex mtx;
    std::condition_variable cv;
    bool done = false;
    std::optional<Value> value;       // Set by leader on provider success
    std::exception_ptr exception;     // Set by leader on provider throw

    /// Leader: store result (value and/or exception) and wake all followers.
    /// Called exactly once per singleflight cycle.
    void complete(std::optional<Value> v, std::exception_ptr exc) {
        {
            std::lock_guard<std::mutex> lock(mtx);
            value = std::move(v);
            exception = exc;
            done = true;
        }
        cv.notify_all();
    }

    /// Follower: block until the leader completes. If the leader's provider
    /// threw, rethrow the captured exception. Otherwise return the value.
    ///
    /// Note: the returned value is a copy of the leader's result. If the
    /// leader also inserted into the cache, the caller may prefer to re-check
    /// the cache for a handle-pinned value; this method does not do that.
    Value wait_and_get() {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [this] { return done; });
        if (exception) {
            std::rethrow_exception(exception);
        }
        return *value;
    }
};

/// Sharded in-flight tracker for singleflight / cache stampede protection.
///
/// Thread-safety: all public methods are thread-safe.
/// Template params:
///   - Key:       cache key type (must be hashable via std::hash)
///   - Value:     cache value type (must be copyable/movable)
///   - NumShards: power-of-2 shard count (default 16)
template <typename Key, typename Value, std::size_t NumShards = 16>
class singleflight_tracker {
public:
    using state_type = singleflight_state<Value>;
    using state_ptr = std::shared_ptr<state_type>;

    /// Result of acquire(): the shared state + whether the caller is leader.
    struct acquire_result {
        state_ptr state;
        bool is_leader;
    };

    // Default constructor
    singleflight_tracker() = default;

    // Move constructor — needed because shards_ contains std::mutex
    // (non-movable). Each shard's mutex is default-constructed; the
    // inflight map is moved under the source shard's lock.
    // Safe because moves only occur when no other thread accesses the source.
    singleflight_tracker(singleflight_tracker&& other) noexcept {
        for (std::size_t i = 0; i < NumShards; ++i) {
            std::lock_guard<std::mutex> lk(other.shards_[i].mtx);
            shards_[i].inflight = std::move(other.shards_[i].inflight);
        }
        coalesced_.store(other.coalesced_.load(std::memory_order_relaxed),
                         std::memory_order_relaxed);
    }

    singleflight_tracker& operator=(singleflight_tracker&& other) noexcept {
        if (this != &other) {
            for (std::size_t i = 0; i < NumShards; ++i) {
                std::lock_guard<std::mutex> lk1(shards_[i].mtx);
                std::lock_guard<std::mutex> lk2(other.shards_[i].mtx);
                shards_[i].inflight = std::move(other.shards_[i].inflight);
            }
            coalesced_.store(other.coalesced_.load(std::memory_order_relaxed),
                             std::memory_order_relaxed);
        }
        return *this;
    }

    /// Acquire a singleflight slot for `key`.
    /// - Returns {state, true}  → caller is the leader; must call complete().
    /// - Returns {state, false} → caller is a follower; must call wait_and_get().
    acquire_result acquire(const Key& key) {
        auto& s = shard_for(key);
        std::lock_guard<std::mutex> lock(s.mtx);
        auto it = s.inflight.find(key);
        if (it != s.inflight.end()) {
            return {it->second, false};
        }
        auto state = std::make_shared<state_type>();
        s.inflight.emplace(key, state);
        return {state, true};
    }

    /// Leader: signal completion (value or exception) and remove the in-flight
    /// entry so future misses on this key start a new singleflight cycle.
    ///
    /// Safe to call even if the entry was already removed (defensive —
    /// the shared_ptr keeps the state alive for any in-flight followers).
    void complete(const Key& key, const state_ptr& state,
                  std::optional<Value> v,
                  std::exception_ptr exc = nullptr) {
        // Signal followers FIRST, then remove from map. If we removed first,
        // a concurrent miss could start a redundant provider call before
        // followers wake. Signaling first ensures all current followers wake
        // before a new cycle can begin.
        state->complete(std::move(v), exc);
        auto& s = shard_for(key);
        std::lock_guard<std::mutex> lock(s.mtx);
        s.inflight.erase(key);
    }

    /// Cumulative count of coalesced (follower) requests across all keys.
    std::size_t coalesced_count() const noexcept {
        return coalesced_.load(std::memory_order_relaxed);
    }

    /// Increment the coalesced counter (called when a follower is collapsed).
    void record_coalesced() noexcept {
        coalesced_.fetch_add(1, std::memory_order_relaxed);
    }

    /// Number of currently in-flight keys (diagnostic; acquires all shard locks).
    std::size_t inflight_count() const {
        std::size_t total = 0;
        for (const auto& s : shards_) {
            std::lock_guard<std::mutex> lock(s.mtx);
            total += s.inflight.size();
        }
        return total;
    }

private:
    struct shard {
        mutable std::mutex mtx;
        ankerl::unordered_dense::map<Key, state_ptr> inflight;
    };

    std::array<shard, NumShards> shards_;
    std::atomic<std::size_t> coalesced_{0};

    shard& shard_for(const Key& key) noexcept {
        const std::size_t h = std::hash<Key>{}(key);
        return shards_[h % NumShards];
    }
    const shard& shard_for(const Key& key) const noexcept {
        const std::size_t h = std::hash<Key>{}(key);
        return shards_[h % NumShards];
    }
};

}  // namespace lru::detail

#endif  // LRU_DETAIL_SINGLEFLIGHT_HPP
