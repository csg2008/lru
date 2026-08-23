// SPDX-License-Identifier: MIT
// Thread-Local Active Item Ring — Inspired by CacheLib's TlsActiveItemRing
//
// Under high concurrency, the LRU lock (even when striped) becomes a bottleneck
// because every read access must move the item to the MRU position. The TLS
// ring defers these promotions: each thread maintains a small ring buffer of
// recently-accessed items. When the buffer fills, the thread flushes the
// promotions to the global LRU in a single batch under the lock.
//
// This reduces lock acquisitions from O(reads) to O(reads / ring_size), which
// for a ring of 64 entries reduces lock contention by ~98% under heavy load.
//
// Key design:
//   - Per-thread ring buffer (thread_local storage, no allocation on hot path)
//   - Deduplication: if the same key is accessed twice before flush, only record once
//   - Flush-on-fill: automatically flush when ring is full
//   - Manual flush: user can flush explicitly (e.g., before lock release)
//   - Configurable ring size per thread
//
// Usage:
//   tls_active_item_ring<Key> ring(64);  // 64-entry ring per thread
//
//   // On read:
//   ring.record(key);
//   if (ring.should_flush()) {
//       ring.flush_to([&](const Key& k) { cache.promote(k); });
//   }

#ifndef LRU_TLS_RING_HPP
#define LRU_TLS_RING_HPP

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <thread>
#include <type_traits>
#include <vector>

#include "ankerl/unordered_dense.h"
#include "event_types.hpp"

namespace lru {

// ============================================================================
// TLS Callback Ring — Zero-allocation callback event collector
// ============================================================================

/// Thread-local circular buffer for collecting cache callback events
/// (hit, miss, insert, evict) without heap allocation or mutex locking.
///
/// Each `tls_callback_ring` instance manages per-thread ring buffers via
/// an instance-ID-to-ring map (same pattern as `tls_active_item_ring`).
/// The `collect_*` methods write to the calling thread's ring; `drain()`
/// reads back all events for the current thread and resets the ring.
///
/// Design:
///   - Zero-allocation hot path: stores Key/Value directly (no shared_ptr)
///   - Zero-lock hot path: writes to thread-local ring (no mutex)
///   - Per-instance TLS: each tls_callback_ring has its own rings per thread
///   - Overflow: oldest entries are silently dropped when ring is full
///   - Separate sub-rings per event type for compact storage
///
/// @tparam Key    Cache key type
/// @tparam Value  Cache value type
/// @tparam N      Ring size per event type (must be power of 2, default 256)
///
/// T10.1: Default raised from 64 to 256 to reduce the frequency of
/// deferred-promotion drops under read-heavy workloads. With N=64 a
/// single thread doing 1M ops/s fills the ring in ~64us, forcing the
/// drain worker to run at >15K Hz just to keep up. With N=256 the
/// same workload allows a 250us drain interval with no drops.
template <typename Key, typename Value, std::size_t N = 256>
class tls_callback_ring {
    static_assert(N > 0, "Ring size must be positive");
    static_assert((N & (N - 1)) == 0, "Ring size must be a power of 2");

public:
    using key_type = Key;
    using value_type = Value;

    static constexpr std::size_t kRingSize = N;
    static constexpr std::size_t kRingMask = N - 1;

    // --------------------------------------------------------------------
    // Event types
    // --------------------------------------------------------------------

    /// Hit event: key + value copy (callback receives const Value&)
    struct hit_event {
        Key key;
        Value value;
    };

    /// Miss event: key only
    struct miss_event {
        Key key;
    };

    /// Insert event: key + value copy (callback receives const Value&)
    struct insert_event {
        Key key;
        Value value;
    };

    /// Evict event: key + value (moved from the evicted item)
    struct evict_event {
        Key key;
        Value value;
    };

    /// O7: Update event — fired when set() overwrites an existing key's
    /// value (not a fresh insert). Same payload as insert_event but
    /// distinct kind so users can audit write-amplification separately
    /// from new inserts.
    struct update_event {
        Key key;
        Value value;
    };

    /// O7: Expire event — fired when an item is evicted due to TTL
    /// expiry (not capacity eviction). Same payload as evict_event but
    /// distinct kind so users can monitor TTL effectiveness and tune
    /// refresh windows separately from capacity pressure.
    struct expire_event {
        Key key;
        Value value;
    };

    /// O7: Reject event — fired when an insert is rejected by the
    /// overflow policy (cache full, OOM, or admission denial). Carries
    /// the would-be key + value so callers can route rejected items to
    /// a fallback store or trigger back-pressure signalling.
    struct reject_event {
        Key key;
        Value value;
    };

    // --------------------------------------------------------------------
    // Drain result
    // --------------------------------------------------------------------

    /// Result of draining all event rings for the current thread.
    struct drain_result {
        std::vector<hit_event> hits;
        std::vector<miss_event> misses;
        std::vector<insert_event> inserts;
        std::vector<evict_event> evicts;
        std::vector<update_event> updates;
        std::vector<expire_event> expires;
        std::vector<reject_event> rejects;

        bool empty() const noexcept {
            return hits.empty() && misses.empty() && inserts.empty()
                && evicts.empty() && updates.empty() && expires.empty()
                && rejects.empty();
        }
    };

    // --------------------------------------------------------------------
    // Construction — non-copyable, non-movable
    // --------------------------------------------------------------------

    tls_callback_ring()
        : instance_id_(next_instance_id_.fetch_add(1, std::memory_order_relaxed)) {
        std::lock_guard<std::mutex> lock(registry_mutex_);
        registry_.push_back({instance_id_, this});
    }

    ~tls_callback_ring() {
        // Unregister from global registry
        {
            std::lock_guard<std::mutex> lock(registry_mutex_);
            registry_.erase(
                std::remove_if(registry_.begin(), registry_.end(),
                    [this](const auto& e) { return e.instance_ptr == this; }),
                registry_.end());
        }
        // Remove this instance's ring_data from the current thread's TLS map.
        // Other threads' entries will be cleaned up lazily on their next access.
        get_thread_data().rings.erase(instance_id_);
    }

    tls_callback_ring(const tls_callback_ring&) = delete;
    tls_callback_ring& operator=(const tls_callback_ring&) = delete;
    tls_callback_ring(tls_callback_ring&&) = delete;
    tls_callback_ring& operator=(tls_callback_ring&&) = delete;

    // --------------------------------------------------------------------
    // Collect methods — lock-free, zero-allocation hot path
    // --------------------------------------------------------------------

    /// Collect a hit event (lvalue key + value).
    void collect_hit(const Key& key, const Value& value) {
        auto& rd = get_ring();
        if (rd.hit_head - rd.hit_tail >= kRingSize) {
            rd.hit_tail = rd.hit_head - kRingSize + 1;
        }
        auto pos = rd.hit_head;
        rd.hits[pos & kRingMask].key = key;
        rd.hits[pos & kRingMask].value = value;
        rd.hit_head = pos + 1;
    }

    /// Collect a hit event (rvalue key + lvalue value).
    void collect_hit(Key&& key, const Value& value) {
        auto& rd = get_ring();
        if (rd.hit_head - rd.hit_tail >= kRingSize) {
            rd.hit_tail = rd.hit_head - kRingSize + 1;
        }
        auto pos = rd.hit_head;
        rd.hits[pos & kRingMask].key = std::move(key);
        rd.hits[pos & kRingMask].value = value;
        rd.hit_head = pos + 1;
    }

    /// Collect a miss event (lvalue key).
    void collect_miss(const Key& key) {
        auto& rd = get_ring();
        if (rd.miss_head - rd.miss_tail >= kRingSize) {
            rd.miss_tail = rd.miss_head - kRingSize + 1;
        }
        auto pos = rd.miss_head;
        rd.misses[pos & kRingMask].key = key;
        rd.miss_head = pos + 1;
    }

    /// Collect a miss event (rvalue key).
    void collect_miss(Key&& key) {
        auto& rd = get_ring();
        if (rd.miss_head - rd.miss_tail >= kRingSize) {
            rd.miss_tail = rd.miss_head - kRingSize + 1;
        }
        auto pos = rd.miss_head;
        rd.misses[pos & kRingMask].key = std::move(key);
        rd.miss_head = pos + 1;
    }

    /// Collect an insert event (lvalue key + lvalue value).
    void collect_insert(const Key& key, const Value& value) {
        auto& rd = get_ring();
        if (rd.insert_head - rd.insert_tail >= kRingSize) {
            rd.insert_tail = rd.insert_head - kRingSize + 1;
        }
        auto pos = rd.insert_head;
        rd.inserts[pos & kRingMask].key = key;
        rd.inserts[pos & kRingMask].value = value;
        rd.insert_head = pos + 1;
    }

    /// Collect an insert event (rvalue key + lvalue value).
    void collect_insert(Key&& key, const Value& value) {
        auto& rd = get_ring();
        if (rd.insert_head - rd.insert_tail >= kRingSize) {
            rd.insert_tail = rd.insert_head - kRingSize + 1;
        }
        auto pos = rd.insert_head;
        rd.inserts[pos & kRingMask].key = std::move(key);
        rd.inserts[pos & kRingMask].value = value;
        rd.insert_head = pos + 1;
    }

    /// Collect an insert event (rvalue key + rvalue value).
    void collect_insert(Key&& key, Value&& value) {
        auto& rd = get_ring();
        if (rd.insert_head - rd.insert_tail >= kRingSize) {
            rd.insert_tail = rd.insert_head - kRingSize + 1;
        }
        auto pos = rd.insert_head;
        rd.inserts[pos & kRingMask].key = std::move(key);
        rd.inserts[pos & kRingMask].value = std::move(value);
        rd.insert_head = pos + 1;
    }

    /// Collect an evict event (lvalue key + rvalue value).
    void collect_evict(const Key& key, Value&& value) {
        auto& rd = get_ring();
        if (rd.evict_head - rd.evict_tail >= kRingSize) {
            rd.evict_tail = rd.evict_head - kRingSize + 1;
        }
        auto pos = rd.evict_head;
        rd.evicts[pos & kRingMask].key = key;
        rd.evicts[pos & kRingMask].value = std::move(value);
        rd.evict_head = pos + 1;
    }

    /// Collect an evict event (rvalue key + rvalue value).
    void collect_evict(Key&& key, Value&& value) {
        auto& rd = get_ring();
        if (rd.evict_head - rd.evict_tail >= kRingSize) {
            rd.evict_tail = rd.evict_head - kRingSize + 1;
        }
        auto pos = rd.evict_head;
        rd.evicts[pos & kRingMask].key = std::move(key);
        rd.evicts[pos & kRingMask].value = std::move(value);
        rd.evict_head = pos + 1;
    }

    // --------------------------------------------------------------------
    // O7: New event collect methods — update / expire / reject.
    // These mirror the insert/evict pattern. Each event kind keeps its
    // own sub-ring so dispatch can stay branch-free per kind.
    // --------------------------------------------------------------------

    /// Collect an update event (lvalue key + lvalue value).
    void collect_update(const Key& key, const Value& value) {
        auto& rd = get_ring();
        if (rd.update_head - rd.update_tail >= kRingSize) {
            rd.update_tail = rd.update_head - kRingSize + 1;
        }
        auto pos = rd.update_head;
        rd.updates[pos & kRingMask].key = key;
        rd.updates[pos & kRingMask].value = value;
        rd.update_head = pos + 1;
    }

    /// Collect an update event (rvalue key + rvalue value).
    void collect_update(Key&& key, Value&& value) {
        auto& rd = get_ring();
        if (rd.update_head - rd.update_tail >= kRingSize) {
            rd.update_tail = rd.update_head - kRingSize + 1;
        }
        auto pos = rd.update_head;
        rd.updates[pos & kRingMask].key = std::move(key);
        rd.updates[pos & kRingMask].value = std::move(value);
        rd.update_head = pos + 1;
    }

    /// Collect an expire event (lvalue key + rvalue value).
    void collect_expire(const Key& key, Value&& value) {
        auto& rd = get_ring();
        if (rd.expire_head - rd.expire_tail >= kRingSize) {
            rd.expire_tail = rd.expire_head - kRingSize + 1;
        }
        auto pos = rd.expire_head;
        rd.expires[pos & kRingMask].key = key;
        rd.expires[pos & kRingMask].value = std::move(value);
        rd.expire_head = pos + 1;
    }

    /// Collect an expire event (rvalue key + rvalue value).
    void collect_expire(Key&& key, Value&& value) {
        auto& rd = get_ring();
        if (rd.expire_head - rd.expire_tail >= kRingSize) {
            rd.expire_tail = rd.expire_head - kRingSize + 1;
        }
        auto pos = rd.expire_head;
        rd.expires[pos & kRingMask].key = std::move(key);
        rd.expires[pos & kRingMask].value = std::move(value);
        rd.expire_head = pos + 1;
    }

    /// Collect a reject event (lvalue key + lvalue value).
    void collect_reject(const Key& key, const Value& value) {
        auto& rd = get_ring();
        if (rd.reject_head - rd.reject_tail >= kRingSize) {
            rd.reject_tail = rd.reject_head - kRingSize + 1;
        }
        auto pos = rd.reject_head;
        rd.rejects[pos & kRingMask].key = key;
        rd.rejects[pos & kRingMask].value = value;
        rd.reject_head = pos + 1;
    }

    /// Collect a reject event (rvalue key + rvalue value).
    void collect_reject(Key&& key, Value&& value) {
        auto& rd = get_ring();
        if (rd.reject_head - rd.reject_tail >= kRingSize) {
            rd.reject_tail = rd.reject_head - kRingSize + 1;
        }
        auto pos = rd.reject_head;
        rd.rejects[pos & kRingMask].key = std::move(key);
        rd.rejects[pos & kRingMask].value = std::move(value);
        rd.reject_head = pos + 1;
    }

    // --------------------------------------------------------------------
    // Drain — read all events and reset the ring
    // --------------------------------------------------------------------

    /// Drain all event rings for the current thread, returning events by value.
    /// Resets the rings to empty state.
    drain_result drain() {
        drain_result result;
        auto& rd = get_ring();

        // Drain hits
        {
            auto count = rd.hit_head - rd.hit_tail;
            if (count > 0) {
                result.hits.reserve(count);
                auto start = rd.hit_tail & kRingMask;
                for (std::size_t i = 0; i < count; ++i) {
                    result.hits.push_back(std::move(rd.hits[(start + i) & kRingMask]));
                }
                rd.hit_tail = rd.hit_head;
            }
        }

        // Drain misses
        {
            auto count = rd.miss_head - rd.miss_tail;
            if (count > 0) {
                result.misses.reserve(count);
                auto start = rd.miss_tail & kRingMask;
                for (std::size_t i = 0; i < count; ++i) {
                    result.misses.push_back(std::move(rd.misses[(start + i) & kRingMask]));
                }
                rd.miss_tail = rd.miss_head;
            }
        }

        // Drain inserts
        {
            auto count = rd.insert_head - rd.insert_tail;
            if (count > 0) {
                result.inserts.reserve(count);
                auto start = rd.insert_tail & kRingMask;
                for (std::size_t i = 0; i < count; ++i) {
                    result.inserts.push_back(std::move(rd.inserts[(start + i) & kRingMask]));
                }
                rd.insert_tail = rd.insert_head;
            }
        }

        // Drain evicts
        {
            auto count = rd.evict_head - rd.evict_tail;
            if (count > 0) {
                result.evicts.reserve(count);
                auto start = rd.evict_tail & kRingMask;
                for (std::size_t i = 0; i < count; ++i) {
                    result.evicts.push_back(std::move(rd.evicts[(start + i) & kRingMask]));
                }
                rd.evict_tail = rd.evict_head;
            }
        }

        // O7: Drain updates
        {
            auto count = rd.update_head - rd.update_tail;
            if (count > 0) {
                result.updates.reserve(count);
                auto start = rd.update_tail & kRingMask;
                for (std::size_t i = 0; i < count; ++i) {
                    result.updates.push_back(std::move(rd.updates[(start + i) & kRingMask]));
                }
                rd.update_tail = rd.update_head;
            }
        }

        // O7: Drain expires
        {
            auto count = rd.expire_head - rd.expire_tail;
            if (count > 0) {
                result.expires.reserve(count);
                auto start = rd.expire_tail & kRingMask;
                for (std::size_t i = 0; i < count; ++i) {
                    result.expires.push_back(std::move(rd.expires[(start + i) & kRingMask]));
                }
                rd.expire_tail = rd.expire_head;
            }
        }

        // O7: Drain rejects
        {
            auto count = rd.reject_head - rd.reject_tail;
            if (count > 0) {
                result.rejects.reserve(count);
                auto start = rd.reject_tail & kRingMask;
                for (std::size_t i = 0; i < count; ++i) {
                    result.rejects.push_back(std::move(rd.rejects[(start + i) & kRingMask]));
                }
                rd.reject_tail = rd.reject_head;
            }
        }

        return result;
    }

    // --------------------------------------------------------------------
    // Observers
    // --------------------------------------------------------------------

    /// Whether any events are pending for the current thread.
    bool has_pending() const noexcept {
        auto& rd = get_ring();
        return (rd.hit_head != rd.hit_tail)
            || (rd.miss_head != rd.miss_tail)
            || (rd.insert_head != rd.insert_tail)
            || (rd.evict_head != rd.evict_tail)
            || (rd.update_head != rd.update_tail)
            || (rd.expire_head != rd.expire_tail)
            || (rd.reject_head != rd.reject_tail);
    }

    /// Number of pending events for the current thread.
    std::size_t pending_count() const noexcept {
        auto& rd = get_ring();
        return (rd.hit_head - rd.hit_tail)
             + (rd.miss_head - rd.miss_tail)
             + (rd.insert_head - rd.insert_tail)
             + (rd.evict_head - rd.evict_tail)
             + (rd.update_head - rd.update_tail)
             + (rd.expire_head - rd.expire_tail)
             + (rd.reject_head - rd.reject_tail);
    }

    /// Reset (discard) all pending events for the current thread.
    void reset() {
        auto& rd = get_ring();
        rd.hit_tail = rd.hit_head;
        rd.miss_tail = rd.miss_head;
        rd.insert_tail = rd.insert_head;
        rd.evict_tail = rd.evict_head;
        rd.update_tail = rd.update_head;
        rd.expire_tail = rd.expire_head;
        rd.reject_tail = rd.reject_head;
    }

    // --------------------------------------------------------------------
    // Cross-thread drain support
    // --------------------------------------------------------------------

    /// Drain all registered instances' current-thread data.
    /// Iterates the global registry and calls drain() on each live instance,
    /// ensuring the calling thread's pending events are processed.
    /// Typically called during cache destruction to minimize event loss.
    ///
    /// Note: This only drains the calling thread's TLS data. Other threads'
    /// data remains in their TLS and cannot be accessed from here.
    static void flush_all_registered() {
        std::vector<tls_callback_ring*> instances;
        {
            std::lock_guard<std::mutex> lock(registry_mutex_);
            instances.reserve(registry_.size());
            for (auto& e : registry_) {
                instances.push_back(e.instance_ptr);
            }
        }
        for (auto* inst : instances) {
            (void)inst->drain();
        }
    }

    // --------------------------------------------------------------------
    // Thread-local singleton
    // --------------------------------------------------------------------

    /// Get the thread-local singleton instance for this Key/Value/N combination.
    /// Note: Most usage is through per-instance objects (with per-instance TLS
    /// routing). The singleton is provided for lightweight standalone use.
    static tls_callback_ring& instance() {
        thread_local tls_callback_ring ring;
        return ring;
    }

private:
    // --------------------------------------------------------------------
    // Per-thread, per-instance ring data
    // --------------------------------------------------------------------

    /// Ring data for one event type: fixed-size circular buffer with
    /// monotonic head/tail indices. Overflow silently drops oldest entries.
    struct ring_data {
        // Hit events: Key + Value
        std::array<hit_event, kRingSize> hits{};
        std::size_t hit_head{0}, hit_tail{0};

        // Miss events: Key only
        std::array<miss_event, kRingSize> misses{};
        std::size_t miss_head{0}, miss_tail{0};

        // Insert events: Key + Value
        std::array<insert_event, kRingSize> inserts{};
        std::size_t insert_head{0}, insert_tail{0};

        // Evict events: Key + Value
        std::array<evict_event, kRingSize> evicts{};
        std::size_t evict_head{0}, evict_tail{0};

        // O7: Update events — same payload as inserts but distinct kind
        std::array<update_event, kRingSize> updates{};
        std::size_t update_head{0}, update_tail{0};

        // O7: Expire events — TTL expiry, distinct from capacity evict
        std::array<expire_event, kRingSize> expires{};
        std::size_t expire_head{0}, expire_tail{0};

        // O7: Reject events — insert rejected (overflow / OOM / admission)
        std::array<reject_event, kRingSize> rejects{};
        std::size_t reject_head{0}, reject_tail{0};
    };

    /// Per-thread map: instance_id → ring_data.
    /// Each tls_callback_ring instance gets its own ring_data per thread.
    struct thread_data {
        ankerl::unordered_dense::map<uint64_t, std::unique_ptr<ring_data>> rings;
    };

    static thread_data& get_thread_data() {
        thread_local thread_data td;
        return td;
    }

    ring_data& get_ring() const {
        auto& td = get_thread_data();
        auto it = td.rings.find(instance_id_);
        if (it == td.rings.end()) {
            auto [insert_it, _] = td.rings.emplace(instance_id_, std::make_unique<ring_data>());
            return *insert_it->second;
        }
        return *it->second;
    }

    static inline std::atomic<uint64_t> next_instance_id_{0};
    uint64_t instance_id_;

    // Global registry for cross-thread drain support
    struct instance_registry_entry {
        uint64_t instance_id;
        tls_callback_ring* instance_ptr;
    };

    static inline std::mutex registry_mutex_;
    static inline std::vector<instance_registry_entry> registry_;
};

// ============================================================================
// TLS Event Ring — Lock-free per-thread event recorder for event_tracker
// ============================================================================

/// Thread-local circular buffer for recording cache lifecycle events
/// (insert, promote, demote, evict, hit) without mutex locking.
///
/// Each `tls_event_ring` instance manages per-thread ring buffers via
/// an instance-ID-to-ring map (same pattern as `tls_callback_ring`).
/// The `record()` method writes to the calling thread's ring; `drain()`
/// reads back all events for the current thread and resets the ring.
///
/// Design:
///   - Zero-lock hot path: writes to thread-local ring (no mutex)
///   - Per-instance TLS: each tls_event_ring has its own rings per thread
///   - Overflow: oldest entries are silently dropped when ring is full
///   - Stores key_hash (uint64_t) + timestamp + event_type + queue_id
///
/// @tparam Key    Cache key type (used only for Hash computation)
/// @tparam Hash   Hash function for Key
/// @tparam N      Ring size (must be power of 2, default 256)
///
/// T10.1: Default raised from 64 to 256 (see tls_callback_ring docs).
template <typename Key, typename Hash = std::hash<Key>, std::size_t N = 256>
class tls_event_ring {
    static_assert(N > 0, "Ring size must be positive");
    static_assert((N & (N - 1)) == 0, "Ring size must be a power of 2");

public:
    using key_type = Key;
    using hash_type = Hash;

    static constexpr std::size_t kRingSize = N;
    static constexpr std::size_t kRingMask = N - 1;

    // --------------------------------------------------------------------
    // Event entry stored in the ring
    // --------------------------------------------------------------------

    struct event_entry {
        uint64_t key_hash = 0;
        uint64_t timestamp_ms = 0;
        event_type type = event_type::insert;
        uint8_t queue_id = 0;
    };

    // --------------------------------------------------------------------
    // Drain result
    // --------------------------------------------------------------------

    struct drain_result {
        std::vector<event_entry> entries;

        bool empty() const noexcept { return entries.empty(); }
        std::size_t size() const noexcept { return entries.size(); }
    };

    // --------------------------------------------------------------------
    // Construction — non-copyable, non-movable
    // --------------------------------------------------------------------

    tls_event_ring()
        : instance_id_(next_instance_id_.fetch_add(1, std::memory_order_relaxed)) {
        std::lock_guard<std::mutex> lock(registry_mutex_);
        registry_.push_back({instance_id_, this});
    }

    ~tls_event_ring() {
        // Unregister from global registry
        {
            std::lock_guard<std::mutex> lock(registry_mutex_);
            registry_.erase(
                std::remove_if(registry_.begin(), registry_.end(),
                    [this](const auto& e) { return e.instance_ptr == this; }),
                registry_.end());
        }
        // P2-3: Remove all per-thread ring_data entries for this instance
        // from the cross-thread registry. This prevents drain_all_threads()
        // from observing a dangling ring_ptr after this instance is destroyed.
        // The ring_data objects themselves are owned by each thread's
        // thread_data map and will be cleaned up when the thread exits
        // (or lazily on next access).
        {
            std::lock_guard<std::mutex> lock(thread_ring_registry_mutex_);
            thread_ring_registry_.erase(
                std::remove_if(thread_ring_registry_.begin(),
                               thread_ring_registry_.end(),
                               [this](const thread_ring_registry_entry& e) {
                                   return e.instance_id == instance_id_;
                               }),
                thread_ring_registry_.end());
        }
        // Remove this instance's ring_data from the current thread's TLS map.
        // Other threads' entries will be cleaned up lazily on their next access.
        get_thread_data().rings.erase(instance_id_);
    }

    tls_event_ring(const tls_event_ring&) = delete;
    tls_event_ring& operator=(const tls_event_ring&) = delete;
    tls_event_ring(tls_event_ring&&) = delete;
    tls_event_ring& operator=(tls_event_ring&&) = delete;

    // --------------------------------------------------------------------
    // Record — lock-free, zero-allocation hot path
    // --------------------------------------------------------------------

    /// Record an event for the given key.
    void record(event_type type, const key_type& key, uint8_t queue_id = 0) {
        auto& rd = get_ring();
        std::size_t head = rd.head.load(std::memory_order_relaxed);
        std::size_t tail = rd.tail.load(std::memory_order_relaxed);
        if (head - tail >= kRingSize) {
            rd.tail.store(head - kRingSize + 1, std::memory_order_relaxed);
        }
        auto pos = head;
        auto& entry = rd.entries[pos & kRingMask];
        entry.key_hash = Hash{}(key);
        entry.timestamp_ms = now_ms();
        entry.type = type;
        entry.queue_id = queue_id;
        rd.head.store(pos + 1, std::memory_order_relaxed);
    }

    /// Record an event with a pre-computed key hash (avoids re-hashing).
    void record_hash(event_type type, uint64_t key_hash, uint8_t queue_id = 0) {
        auto& rd = get_ring();
        std::size_t head = rd.head.load(std::memory_order_relaxed);
        std::size_t tail = rd.tail.load(std::memory_order_relaxed);
        if (head - tail >= kRingSize) {
            rd.tail.store(head - kRingSize + 1, std::memory_order_relaxed);
        }
        auto pos = head;
        auto& entry = rd.entries[pos & kRingMask];
        entry.key_hash = key_hash;
        entry.timestamp_ms = now_ms();
        entry.type = type;
        entry.queue_id = queue_id;
        rd.head.store(pos + 1, std::memory_order_relaxed);
    }

    // --------------------------------------------------------------------
    // Drain — read all events and reset the ring
    // --------------------------------------------------------------------

    /// Drain all events for the current thread, returning them in FIFO order.
    /// Resets the ring to empty state.
    drain_result drain() {
        drain_result result;
        auto& rd = get_ring();

        std::size_t head = rd.head.load(std::memory_order_relaxed);
        std::size_t tail = rd.tail.load(std::memory_order_relaxed);
        auto count = head - tail;
        if (count == 0) {
            return result;
        }
        if (count > kRingSize) {
            count = kRingSize;
            tail = head - kRingSize;
        }

        result.entries.reserve(count);
        auto start = tail & kRingMask;
        for (std::size_t i = 0; i < count; ++i) {
            result.entries.push_back(rd.entries[(start + i) & kRingMask]);
        }

        rd.tail.store(head, std::memory_order_relaxed);
        return result;
    }

    // --------------------------------------------------------------------
    // Steal — lock-free atomic buffer exchange (zero-copy drain)
    // --------------------------------------------------------------------

    /// Result of stealing the ring buffer contents.
    /// Holds a pointer into the TLS ring array and the captured head/tail.
    /// The caller must process entries before the ring wraps around and
    /// overwrites them (safe if processed immediately after steal).
    struct steal_result {
        const event_entry* entries;
        std::size_t head;
        std::size_t tail;

        std::size_t count() const noexcept { return head - tail; }
        bool empty() const noexcept { return head == tail; }
    };

    /// Atomically capture and reset the ring buffer for the current thread.
    /// Returns a lightweight view (steal_result) into the ring entries with
    /// the captured head/tail positions. After this call, the ring is empty
    /// and new records will write to positions beyond the captured head.
    ///
    /// This is the lock-free equivalent of drain() — it avoids the vector
    /// allocation/copy by returning a pointer into the ring array. The
    /// caller processes entries directly from the stolen view.
    ///
    /// Thread safety: Only the calling thread's TLS ring is affected.
    /// Since ring_data is per-thread, no synchronization is needed for
    /// the exchange itself. The "atomic" semantics ensure that subsequent
    /// record() calls see an empty ring.
    steal_result steal() {
        auto& rd = get_ring();
        auto h = rd.head.load(std::memory_order_relaxed);
        auto t = rd.tail.load(std::memory_order_relaxed);
        rd.tail.store(h, std::memory_order_relaxed);  // Reset: atomically mark all entries as consumed
        return steal_result{rd.entries.data(), h, t};
    }

    // --------------------------------------------------------------------
    // Observers
    // --------------------------------------------------------------------

    /// Whether any events are pending for the current thread.
    bool has_pending() const noexcept {
        auto& rd = get_ring();
        return rd.head.load(std::memory_order_relaxed) != rd.tail.load(std::memory_order_relaxed);
    }

    /// Number of pending events for the current thread.
    std::size_t pending_count() const noexcept {
        auto& rd = get_ring();
        return rd.head.load(std::memory_order_relaxed) - rd.tail.load(std::memory_order_relaxed);
    }

    /// Reset (discard) all pending events for the current thread.
    void reset() {
        auto& rd = get_ring();
        rd.tail.store(rd.head.load(std::memory_order_relaxed), std::memory_order_relaxed);
    }

    // --------------------------------------------------------------------
    // Cross-thread drain support
    // --------------------------------------------------------------------

    /// Drain all registered instances' current-thread data.
    /// Iterates the global registry and calls drain() on each live instance,
    /// ensuring the calling thread's pending events are processed.
    /// Typically called during cache destruction to minimize event loss.
    ///
    /// T20.3: Also drains the global backup buffer (events from exited
    /// threads) and returns them via the returned drain_result. Callers
    /// that want to process those events should consume the returned
    /// result; callers that only care about the side-effect (clearing
    /// the TLS rings) can ignore the return value.
    ///
    /// Note: This only drains the calling thread's TLS data plus the
    /// backup buffer. Other live threads' data remains in their TLS and
    /// will be picked up by `drain_all_threads()` (which also drains
    /// the backup).
    static drain_result flush_all_registered() {
        drain_result result;
        std::vector<tls_event_ring*> instances;
        {
            std::lock_guard<std::mutex> lock(registry_mutex_);
            instances.reserve(registry_.size());
            for (auto& e : registry_) {
                instances.push_back(e.instance_ptr);
            }
        }
        for (auto* inst : instances) {
            auto drained = inst->drain();
            if (!drained.entries.empty()) {
                result.entries.insert(result.entries.end(),
                                      std::make_move_iterator(drained.entries.begin()),
                                      std::make_move_iterator(drained.entries.end()));
            }
        }
        // T20.3: Drain the backup buffer (events from exited threads).
        auto backup = drain_backup();
        if (!backup.entries.empty()) {
            result.entries.insert(result.entries.end(),
                                  std::make_move_iterator(backup.entries.begin()),
                                  std::make_move_iterator(backup.entries.end()));
        }
        return result;
    }

    /// P2-3: Drain ALL threads' TLS data for this instance.
    ///
    /// Walks the global per-thread ring registry and drains every thread's
    /// ring_data for this instance. This ensures that `top_keys()` and
    /// `generate_report()` reflect events recorded on all threads, not just
    /// the calling thread.
    ///
    /// T20.3: Also drains the global backup buffer first, so events pushed
    /// there by dying threads (via `thread_exit_sentinel`) are retrieved
    /// alongside live threads' data. The pattern mirrors
    /// `tls_access_ring::drain_all_threads()`.
    ///
    /// The drain is best-effort with respect to concurrent `record()` calls
    /// on other threads: a concurrent record may either be included in this
    /// batch (if it wrote before the drain read head) or deferred to the next
    /// drain (if it wrote after). This is acceptable for event tracking,
    /// which is idempotent and approximate. The pattern mirrors
    /// `tls_access_ring::drain_all_threads()`.
    ///
    /// \return All drained events for this instance, aggregated across every
    ///         registered thread ring AND the global backup buffer, in
    ///         registry order (backup first).
    drain_result drain_all_threads() {
        drain_result result;
        // T20.3: Drain the global backup buffer first (events from exited
        // threads). This ensures dying-thread events are surfaced before
        // any newly-recorded events on live threads, preserving temporal
        // order to the extent possible.
        auto backup = drain_backup();
        if (!backup.entries.empty()) {
            result.entries = std::move(backup.entries);
        }
        std::vector<ring_data*> rings_to_drain;
        {
            std::lock_guard<std::mutex> lock(thread_ring_registry_mutex_);
            rings_to_drain.reserve(thread_ring_registry_.size());
            for (const auto& e : thread_ring_registry_) {
                if (e.instance_id == instance_id_) {
                    rings_to_drain.push_back(e.ring_ptr);
                }
            }
        }
        for (auto* rd : rings_to_drain) {
            if (!rd) continue;
            std::size_t head = rd->head.load(std::memory_order_relaxed);
            std::size_t tail = rd->tail.load(std::memory_order_relaxed);
            auto count = head - tail;
            if (count == 0) continue;
            if (count > kRingSize) {
                count = kRingSize;
                tail = head - kRingSize;
            }
            auto start = tail & kRingMask;
            for (std::size_t i = 0; i < count; ++i) {
                result.entries.push_back(rd->entries[(start + i) & kRingMask]);
            }
            // Reset: mark all consumed entries as read.
            rd->tail.store(head, std::memory_order_relaxed);
        }
        return result;
    }

    // --------------------------------------------------------------------
    // T20: Global backup buffer — collects events from exited threads
    // --------------------------------------------------------------------
    //
    // When a thread exits, its thread_exit_sentinel drains the remaining
    // events from its TLS ring and pushes them into this process-global
    // backup buffer (mutex-protected). A subsequent drain_all_threads()
    // (or flush_all_registered() + drain_backup()) retrieves them.
    //
    // This mirrors the tls_access_ring backup buffer pattern (T7) and
    // prevents event loss when a thread exits before the next drain cycle.
    // Without this, events recorded by short-lived threads (e.g. request
    // handler threads in a thread pool) would be silently dropped.

    /// Push events from a dying thread's TLS ring into the global backup.
    /// Called by `thread_exit_sentinel` when the thread is about to exit.
    /// Safe to call from any thread; the backup is mutex-protected.
    static void push_to_backup(std::vector<event_entry>&& entries) {
        if (entries.empty()) return;
        auto& bk = backup_buffer();
        std::lock_guard<std::mutex> lock(bk.mutex);
        bk.entries.insert(bk.entries.end(),
                          std::make_move_iterator(entries.begin()),
                          std::make_move_iterator(entries.end()));
    }

    /// Drain the global backup buffer, returning all events accumulated
    /// from exited threads. After this call the backup buffer is empty.
    /// Safe to call from any thread.
    static drain_result drain_backup() {
        drain_result result;
        auto& bk = backup_buffer();
        std::lock_guard<std::mutex> lock(bk.mutex);
        result.entries = std::move(bk.entries);
        bk.entries.clear();
        return result;
    }

    /// Whether the global backup buffer has any pending events.
    static bool has_backup_entries() {
        auto& bk = backup_buffer();
        std::lock_guard<std::mutex> lock(bk.mutex);
        return !bk.entries.empty();
    }

private:
    // --------------------------------------------------------------------
    // Per-thread, per-instance ring data
    // --------------------------------------------------------------------

    struct ring_data {
        std::array<event_entry, kRingSize> entries{};
        alignas(64) std::atomic<std::size_t> head{0};
        alignas(64) std::atomic<std::size_t> tail{0};
    };

    struct thread_data {
        ankerl::unordered_dense::map<uint64_t, std::unique_ptr<ring_data>> rings;
    };

    static thread_data& get_thread_data() {
        thread_local thread_data td;
        return td;
    }

    ring_data& get_ring() const {
        auto& td = get_thread_data();
        auto it = td.rings.find(instance_id_);
        if (it == td.rings.end()) {
            auto [insert_it, _] = td.rings.emplace(instance_id_, std::make_unique<ring_data>());
            // P2-3: Register this thread's ring in the cross-thread registry
            // so drain_all_threads() can find and drain it.
            ring_data* ptr = insert_it->second.get();
            {
                std::lock_guard<std::mutex> lock(thread_ring_registry_mutex_);
                thread_ring_registry_.push_back(
                    thread_ring_registry_entry{instance_id_, std::this_thread::get_id(), ptr});
            }
            // Install the thread-exit sentinel on first access. The sentinel
            // is thread_local: constructed once per thread per template
            // instantiation, destroyed (unregistering this thread's entries)
            // when the thread exits. It must run before `thread_data td` is
            // destroyed (td is constructed before the sentinel since get_thread_data()
            // is called first), ensuring ring_data stays alive while the sentinel
            // holds a pointer to it in the registry.
            thread_local thread_exit_sentinel sentinel;
            (void)sentinel;
            return *insert_it->second;
        }
        return *it->second;
    }

    static uint64_t now_ms() {
        // R6: Use steady_clock instead of system_clock for monotonic
        // timestamps. system_clock is subject to NTP adjustments and can
        // jump backwards, causing negative TTL values in event_tracker.
        // steady_clock is monotonically increasing — correct for all
        // elapsed-time calculations (TTL, churn detection, etc.).
        return static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count());
    }

    static inline std::atomic<uint64_t> next_instance_id_{0};
    uint64_t instance_id_;

    // Global registry for cross-thread drain support (instance-level)
    struct instance_registry_entry {
        uint64_t instance_id;
        tls_event_ring* instance_ptr;
    };

    static inline std::mutex registry_mutex_;
    static inline std::vector<instance_registry_entry> registry_;

    // --------------------------------------------------------------------
    // P2-3: Per-thread ring registry — tracks every (instance, thread) pair
    // so drain_all_threads() can find and drain every thread's ring_data
    // for a given instance.
    // --------------------------------------------------------------------

    struct thread_ring_registry_entry {
        uint64_t instance_id;
        std::thread::id tid;
        ring_data* ring_ptr;
    };

    static inline std::mutex thread_ring_registry_mutex_;
    static inline std::vector<thread_ring_registry_entry> thread_ring_registry_;

    // --------------------------------------------------------------------
    // T20: Global backup buffer storage — one per (Key, Hash, N) template
    // instantiation. Shared by all instances of this tls_event_ring
    // specialization. Mutex-protected; push/drain are safe from any thread.
    // --------------------------------------------------------------------
    struct backup_storage {
        std::mutex mutex;
        std::vector<event_entry> entries;
    };

    /// Meyers singleton for the backup buffer — one per Key/Hash/N
    /// specialization. Returned by reference so it lives across calls
    /// and is destroyed at program exit.
    static backup_storage& backup_buffer() {
        static backup_storage storage;
        return storage;
    }

    /// Thread-exit sentinel: unregisters this thread's ring_data entries
    /// from the cross-thread registry when the thread exits. One sentinel
    /// per thread per template instantiation; constructed on first call to
    /// get_ring(), destroyed before the thread's `thread_data` (ensuring
    /// the registry doesn't hold a dangling pointer to destroyed ring_data).
    ///
    /// T20.2: Before unregistering, the sentinel drains the current thread's
    /// ring_data for every registered instance and pushes the events into
    /// the corresponding instance's global backup buffer. This prevents
    /// event loss when a thread exits before the next drain cycle (e.g.
    /// short-lived worker threads in a thread pool).
    struct thread_exit_sentinel {
        ~thread_exit_sentinel() {
            const auto tid = std::this_thread::get_id();
            // T20.2: Collect (instance_id, ring_ptr) pairs for this thread
            // BEFORE removing them from the registry. We need to hold the
            // lock during both the snapshot and the erase to prevent
            // drain_all_threads() from observing a half-removed state.
            std::vector<std::pair<uint64_t, ring_data*>> to_flush;
            {
                std::lock_guard<std::mutex> lock(thread_ring_registry_mutex_);
                for (const auto& e : thread_ring_registry_) {
                    if (e.tid == tid) {
                        to_flush.emplace_back(e.instance_id, e.ring_ptr);
                    }
                }
                thread_ring_registry_.erase(
                    std::remove_if(thread_ring_registry_.begin(),
                                   thread_ring_registry_.end(),
                                   [tid](const thread_ring_registry_entry& e) {
                                       return e.tid == tid;
                                   }),
                    thread_ring_registry_.end());
            }
            // T20.2: Drain each ring_data and push events into the backup.
            // We do this OUTSIDE the registry mutex to minimize its hold
            // time and avoid potential deadlock with drain_all_threads()
            // (which takes the same mutex then reads ring_data). The
            // ring_data is still alive here because thread_data (which
            // owns it) is destroyed after the sentinel — see get_ring().
            for (auto [inst_id, rd] : to_flush) {
                if (!rd) continue;
                std::size_t head = rd->head.load(std::memory_order_relaxed);
                std::size_t tail = rd->tail.load(std::memory_order_relaxed);
                auto count = head - tail;
                if (count == 0) continue;
                if (count > kRingSize) {
                    count = kRingSize;
                    tail = head - kRingSize;
                }
                std::vector<event_entry> entries;
                entries.reserve(count);
                auto start = tail & kRingMask;
                for (std::size_t i = 0; i < count; ++i) {
                    entries.push_back(rd->entries[(start + i) & kRingMask]);
                }
                // Mark all consumed entries as read so a concurrent drain
                // does not double-count them (best-effort; the ring_data
                // is about to be destroyed anyway).
                rd->tail.store(head, std::memory_order_relaxed);
                // T20.2: Push into the backup buffer. We must find the
                // instance whose backup buffer to use. The backup is
                // per-instance (per template specialization), so we look
                // up the instance pointer from the instance registry.
                // However, the backup is static (one per template
                // instantiation), so any instance of the same type shares
                // the same backup. We can therefore push directly.
                push_to_backup(std::move(entries));
            }
        }
    };
};

// ============================================================================
// TLS Access Ring — Lightweight lock-free per-thread access event recorder
// ============================================================================

/// A thread-local fixed-size circular buffer for recording cache access events.
/// Inspired by CacheLib's TlsActiveItemRing, but simpler and lighter-weight:
///   - Lock-free write path (single-writer per thread by construction)
///   - Stores key values (not hashes) for deferred LRU promotion
///   - Power-of-2 size for efficient modulo via bitmask
///   - drain() returns all buffered keys and resets the ring
///   - Configurable ring size at compile time (template param) or at
///     construction time (via tls_access_ring_config)
///
/// Usage:
///   // Default: 64-entry ring, thread-local instance
///   auto& ring = tls_access_ring<int>::instance();
///   ring.record_access(42);
///   ring.record_miss(99);
///   auto keys = ring.drain();  // returns vector<int>{42, 99}
///
///   // Custom size (must be power of 2):
///   tls_access_ring<int, 128> ring;
///   ring.record_access(key);
///   auto entries = ring.drain();

/// Configuration for tls_access_ring when runtime size selection is needed.
struct tls_access_ring_config {
    std::size_t ring_size = 64;

    /// Validate and canonicalize: round up to next power of 2, minimum 1.
    static constexpr std::size_t next_pow2(std::size_t n) noexcept {
        if (n <= 1) return 1;
        --n;
        n |= n >> 1;
        n |= n >> 2;
        n |= n >> 4;
        n |= n >> 8;
        n |= n >> 16;
        if constexpr (sizeof(std::size_t) > 4) n |= n >> 32;
        return n + 1;
    }

    constexpr tls_access_ring_config() = default;
    constexpr explicit tls_access_ring_config(std::size_t size)
        : ring_size(next_pow2(size)) {}
};

/// Overflow policy for `tls_access_ring`. Selected per `<Key, N>` template
/// specialization via `tls_access_ring::set_full_policy()`.
enum class tls_ring_full_policy {
    /// Silently overwrite the oldest entry and increment the dropped
    /// counters (`dropped_count_` and `total_dropped_`). This is the
    /// default and preserves the historical wrap-around behavior.
    kSilentDrop,
    /// Invoke the registered flush callback (`set_flush_callback`) when
    /// the ring overflows. The callback is expected to drain the ring
    /// and promote the keys. If the callback does not drain the ring,
    /// the implementation falls back to silent-drop semantics to
    /// guarantee forward progress.
    kFlushOnFull,
    /// Trigger an assertion on overflow (debugging aid). In release
    /// builds (NDEBUG) the assertion is a no-op and the ring recovers
    /// by dropping the oldest entry, matching kSilentDrop semantics.
    kAssertOnFull,
};

/// Result type returned by drain(). Contains the keys in FIFO order.
template <typename Key>
struct access_ring_drain_result {
    std::vector<Key> keys;

    access_ring_drain_result() = default;
    explicit access_ring_drain_result(std::vector<Key>&& k) : keys(std::move(k)) {}

    bool empty() const noexcept { return keys.empty(); }
    std::size_t size() const noexcept { return keys.size(); }
};

/// T10.1: Default raised from 64 to 256 (see tls_callback_ring docs).
///
/// T-D1 (P2-1): Runtime-configurable capacity. The compile-time template
/// parameter `N` is now the *upper bound* on the ring capacity. The
/// effective capacity is read from `runtime_capacity_` (a static atomic,
/// default N) on each thread's first `record_access()` call and cached
/// in `per_thread_cap_` (a per-thread atomic). This matches the spec's
/// "thread_local 首次访问时动态分配" intent: each thread's ring uses
/// the capacity that was configured at the time of its first access.
///
///   - `set_tls_ring_capacity(cap)` — global API, affects new threads only
///   - `tls_ring_capacity()` — query the configured global capacity
///   - Existing threads keep their snapshot until the ring is reset
///   - Lowering capacity uses fewer slots of the pre-allocated `buf_`
///     (no reallocation); raising it back up to N uses more slots
///   - Cross-thread reads (`force_flush_dormant_threads`) read the
///     target thread's `per_thread_cap_` for correct mask computation
template <typename Key, std::size_t N = 256>
class tls_access_ring {
    static_assert(N > 0, "Ring size must be positive");
    static_assert((N & (N - 1)) == 0, "Ring size must be a power of 2");

public:
    using key_type = Key;
    using drain_result = access_ring_drain_result<Key>;

    static constexpr std::size_t kRingSize = N;

    // --------------------------------------------------------------------
    // T-D2 (P2-2): Per-cache config struct
    // --------------------------------------------------------------------
    //
    // Historically `full_policy_`, `auto_drain_threshold_`, and the flush
    // callback were `static inline` members — shared across every cache
    // instance that used the same `<Key, N>` specialization. This made it
    // impossible for two caches with the same key type to have different
    // overflow policies or drain thresholds.
    //
    // T-D2 introduces `tls_ring_config`: a per-cache config struct that
    // the cache sets as the "active config" (via `set_active_config()`)
    // before calling `record_access()`. The active config is stored in a
    // `thread_local` pointer, so each thread can have a different active
    // cache (e.g. in a request handler that touches multiple caches).
    //
    // Backward compat: if `active_config_` is null (no config set), the
    // static defaults are used — preserving the historical behavior.
    struct tls_ring_config {
        /// Overflow policy for this cache. Default: kFlushOnFull.
        std::atomic<tls_ring_full_policy> full_policy{
            tls_ring_full_policy::kFlushOnFull};

        /// Auto-drain threshold for this cache. Default: kRingSize / 2
        /// (R6: lowered from kRingSize to enable proactive draining
        /// before the ring fills up, bounding drain latency to ~N/2
        /// record_access() calls instead of N).
        std::atomic<std::size_t> auto_drain_threshold{kRingSize / 2};

        /// Flush callback for this cache. Default: empty (no-op).
        /// When non-empty, invoked under kFlushOnFull policy on overflow.
        std::function<void()> flush_callback;

        tls_ring_config() = default;
        tls_ring_config(const tls_ring_config& other) {
            full_policy.store(other.full_policy.load(std::memory_order_relaxed),
                              std::memory_order_relaxed);
            auto_drain_threshold.store(
                other.auto_drain_threshold.load(std::memory_order_relaxed),
                std::memory_order_relaxed);
            flush_callback = other.flush_callback;  // copy (not atomic)
        }
        tls_ring_config& operator=(const tls_ring_config& other) {
            if (this != &other) {
                full_policy.store(other.full_policy.load(std::memory_order_relaxed),
                                  std::memory_order_relaxed);
                auto_drain_threshold.store(
                    other.auto_drain_threshold.load(std::memory_order_relaxed),
                    std::memory_order_relaxed);
                flush_callback = other.flush_callback;
            }
            return *this;
        }
        // Move constructor/assignment — needed for unified_cache movability
        tls_ring_config(tls_ring_config&& other) noexcept {
            full_policy.store(other.full_policy.load(std::memory_order_relaxed),
                              std::memory_order_relaxed);
            auto_drain_threshold.store(
                other.auto_drain_threshold.load(std::memory_order_relaxed),
                std::memory_order_relaxed);
            flush_callback = std::move(other.flush_callback);
        }
        tls_ring_config& operator=(tls_ring_config&& other) noexcept {
            if (this != &other) {
                full_policy.store(other.full_policy.load(std::memory_order_relaxed),
                                  std::memory_order_relaxed);
                auto_drain_threshold.store(
                    other.auto_drain_threshold.load(std::memory_order_relaxed),
                    std::memory_order_relaxed);
                flush_callback = std::move(other.flush_callback);
            }
            return *this;
        }
    };

    // --------------------------------------------------------------------
    // T-D2 (P2-2): Active config management
    // --------------------------------------------------------------------
    //
    // The active config is a thread_local pointer — each thread can have
    // a different active cache (e.g. a request handler thread that
    // alternates between caches). The RAII guard `active_config_scope`
    // saves/restores the previous active config for exception safety.
    //
    // Usage:
    //   tls_ring_config cfg;
    //   cfg.auto_drain_threshold.store(128, std::memory_order_relaxed);
    //   {
    //       tls_access_ring<K>::active_config_scope scope(&cfg);
    //       cache.get(key);  // record_access uses cfg, not static defaults
    //   }  // previous active config restored

    /// Set the active config for the calling thread. Pass nullptr to
    /// revert to static defaults (historical behavior).
    static void set_active_config(tls_ring_config* cfg) noexcept {
        active_config_ = cfg;
    }

    /// Get the active config for the calling thread, or nullptr if none
    /// is set (falls back to static defaults).
    static tls_ring_config* get_active_config() noexcept {
        return active_config_;
    }

    /// RAII guard that sets the active config on construction and
    /// restores the previous config on destruction. Exception-safe.
    struct active_config_scope {
        tls_ring_config* prev;
        explicit active_config_scope(tls_ring_config* cfg) noexcept
            : prev(active_config_) {
            active_config_ = cfg;
        }
        ~active_config_scope() noexcept {
            active_config_ = prev;
        }
        active_config_scope(const active_config_scope&) = delete;
        active_config_scope& operator=(const active_config_scope&) = delete;
    };

    // --------------------------------------------------------------------
    // Construction — non-copyable, non-movable
    // --------------------------------------------------------------------

    tls_access_ring() = default;

    tls_access_ring(const tls_access_ring&) = delete;
    tls_access_ring& operator=(const tls_access_ring&) = delete;
    tls_access_ring(tls_access_ring&&) = delete;
    tls_access_ring& operator=(tls_access_ring&&) = delete;

    // --------------------------------------------------------------------
    // Access recording (lock-free, single-writer per thread)
    // --------------------------------------------------------------------

    /// Record a cache access (hit) for the given key.
    /// Overflow behavior is governed by the configured `tls_ring_full_policy`
    /// (see `set_full_policy`). The default (`kSilentDrop`) preserves the
    /// historical wrap-around semantics.
    ///
    /// T-D1 (P2-1): On first access, snapshots the global
    /// `runtime_capacity_` into this thread's `per_thread_cap_`. The
    /// snapshot is used as the effective ring capacity and mask for all
    /// subsequent accesses until `reset()` clears it.
    void record_access(const Key& key) {
        // T-D1: Snapshot the runtime capacity on first access (or after
        // reset). This matches the spec's "thread_local 首次访问时动态
        // 分配" intent: each thread uses the capacity configured at the
        // time of its first access.
        const std::size_t cap = effective_capacity();
        const std::size_t mask = cap - 1;

        // P1-4 (T3.3): Update heartbeat timestamp so the background
        // drain worker can detect dormant threads. Use release
        // semantics so that a cross-thread reader observing this
        // timestamp via acquire also observes all prior writes to
        // buf_[] (head_ is monotonically increasing, so any prior
        // write to buf_[pos & mask] happens before this store).
        //
        // Batched: the clock read is skipped on most accesses. The
        // heartbeat only answers "did this thread do anything recently",
        // so refreshing at batch boundaries keeps it accurate while
        // removing a steady_clock::now() from every hit. (buf_[] writes
        // remain visible to cross-thread drainers via the head_ release
        // store below, independent of this heartbeat.)
        if ((heartbeat_skip_++ & (kHeartbeatInterval - 1)) == 0) {
            last_activity_ns_.store(
                static_cast<std::uint64_t>(
                    std::chrono::steady_clock::now().time_since_epoch().count()),
                std::memory_order_release);
        }

        // Cross-thread flush request: if another thread asked us to drain
        // (see drain_all_threads()), service it now on our own thread to
        // keep thread_local mutations on their owning thread.
        if (needs_flush_.load(std::memory_order_acquire)) {
            // T7.1: Clear the flag BEFORE draining so a concurrent
            // drain_all_threads_sync() that just set the flag again
            // observes a consistent state (we'll service that next time).
            needs_flush_.store(false, std::memory_order_release);
            auto drained = drain();
            if (!drained.keys.empty()) {
                push_to_backup(std::move(drained.keys));
            }
            // T7.1: Decrement the pending-drain counter so that
            // drain_all_threads_sync() can detect completion.
            if (pending_drain_count_.load(std::memory_order_relaxed) > 0) {
                pending_drain_count_.fetch_sub(1, std::memory_order_release);
            }
        }

        std::size_t pos = head_.load(std::memory_order_relaxed);
        buf_[pos & mask] = key;
        // G15: release (not relaxed) so the buf_[pos & mask] write above
        // is visible to cross-thread readers (force_flush_dormant_threads)
        // that acquire-load head_. last_activity_ns_ release above does NOT
        // cover this buf_[] write (it runs before this write in source
        // order, so it synchronizes only prior iterations). With a relaxed
        // store the compiler/CPU could reorder buf_[] past head_, letting a
        // dormant-thread drainer observe new head_ but stale buf_[] and miss
        // data. head_ is single-writer (this TLS thread) / multi-reader
        // (steal path), so release-acquire is the standard pairing. Owning-
        // thread loads of head_ below stay relaxed - they do not cross threads.
        head_.store(pos + 1, std::memory_order_release);

        // T-P2-4: Maintain the cross-thread backlog aggregate for
        // diagnostics without contending on the shared atomic. Increment
        // a pure thread-local counter (zero cache-line contention) and,
        // every `kBacklogFlushBatch` increments, batch-flush the accumulated
        // delta to `total_backlog_` with a single relaxed `fetch_add`. This
        // removes the per-`record_access()` cache-line bounce that limited
        // scalability under high read concurrency.
        ++tls_pending_backlog_;
        if ((tls_pending_backlog_ & (kBacklogFlushBatch - 1)) == 0) {
            total_backlog_.fetch_add(tls_pending_backlog_,
                std::memory_order_relaxed);
            tls_pending_backlog_ = 0;
        }

        // Overflow handling: an entry was just overwritten. Apply the
        // configured policy so callers can react (drop, flush, or assert).
        if (head_.load(std::memory_order_relaxed) - tail_.load(std::memory_order_relaxed) > cap) {
            apply_overflow_policy();
        }

        // T10.2: Auto-drain when the ring fills past the configured
        // threshold. This caps the worst-case drain latency without
        // requiring a background worker. Default threshold is N (i.e.
        // only auto-drain on overflow, which is already handled by
        // apply_overflow_policy above); set it lower (e.g. N/2) for
        // smoother drain behavior under burst traffic.
        //
        // T-D1: threshold is clamped to per_thread_cap_ so auto-drain
        // triggers no later than ring-full (which is handled above).
        //
        // T-D2 (P2-2): Prefer the per-cache active config's threshold
        // when set (via `set_active_config()` / `active_config_scope`).
        // This allows two cache instances in the same process to use
        // different drain thresholds without falling back to the
        // `<Key, N>`-specialization-wide static default. When the
        // active config is null, fall back to the static default.
        tls_ring_config* cfg = active_config_;
        const std::size_t threshold_raw = cfg
            ? cfg->auto_drain_threshold.load(std::memory_order_relaxed)
            : auto_drain_threshold_.load(std::memory_order_relaxed);
        const std::size_t threshold = threshold_raw > cap ? cap : threshold_raw;
        if (threshold < cap &&
            (head_.load(std::memory_order_relaxed) - tail_.load(std::memory_order_relaxed)) >= threshold) {
            // T10.3: count this auto-drain for the flush_per_sec metric.
            tls_ring_flush_count_.fetch_add(1, std::memory_order_relaxed);
            auto drained = drain();
            if (!drained.keys.empty()) {
                push_to_backup(std::move(drained.keys));
            }
        }
    }

    /// Record a cache miss for the given key.
    /// Same as record_access — the caller decides whether the event
    /// represents a hit or miss; the ring just records keys.
    void record_miss(const Key& key) {
        record_access(key);
    }

    // --------------------------------------------------------------------
    // Overflow policy and dropped-access accounting (Task 2)
    // --------------------------------------------------------------------

    /// Set the overflow policy for this `<Key, N>` specialization. The
    /// policy is shared across all threads because it is a deployment-level
    /// decision (typically set once during cache initialization).
    static void set_full_policy(tls_ring_full_policy policy) noexcept {
        full_policy_.store(policy, std::memory_order_relaxed);
    }

    /// Get the currently configured overflow policy.
    static tls_ring_full_policy get_full_policy() noexcept {
        return full_policy_.load(std::memory_order_relaxed);
    }

    /// T10.2: Set the auto-drain threshold for this `<Key, N>` specialization.
    /// When the ring's occupancy reaches `threshold`, the next `record_access()`
    /// call will synchronously drain the ring into the global backup buffer.
    ///
    /// - threshold == N: auto-drain is disabled; the ring is only
    ///   drained by the background worker or by explicit `drain_all_threads()`.
    /// - threshold < N (default N/2): auto-drain triggers when occupancy
    ///   reaches threshold. Recommended value is N/2 (128 for the default
    ///   N=256) — this bounds the worst-case drain latency to ~threshold
    ///   record_access() calls while keeping the drain frequency reasonable.
    ///
    /// The threshold is shared across all threads (deployment-level setting).
    static void set_tls_drain_threshold(std::size_t threshold) noexcept {
        if (threshold == 0 || threshold > kRingSize) {
            threshold = kRingSize / 2;  // R6: default to N/2 instead of disabling
        }
        auto_drain_threshold_.store(threshold, std::memory_order_relaxed);
    }

    /// T10.2: Query the current auto-drain threshold.
    static std::size_t tls_drain_threshold() noexcept {
        return auto_drain_threshold_.load(std::memory_order_relaxed);
    }

    /// T10.2: Runtime configuration of the effective ring capacity.
    ///
    /// The physical ring size `N` is a compile-time template parameter
    /// (default 256, see T10.1). At runtime, callers can shrink the
    /// *effective* capacity by lowering the auto-drain threshold — once
    /// occupancy reaches `effective_size`, the ring auto-drains. This is
    /// functionally equivalent to a smaller ring: lower effective size →
    /// more frequent drains → lower worst-case drain latency, at the cost
    /// of higher drain frequency.
    ///
    /// Use `effective_size > N` or `effective_size == 0` to disable
    /// auto-drain (the ring fills fully before draining).
    ///
    /// This API is the runtime-configurable counterpart to the
    /// compile-time `N` parameter. It does NOT change `sizeof(tls_ring)`
    /// — the physical storage is fixed at compile time.
    ///
    /// T-D1 (P2-1): For true physical-capacity changes (mask used for
    /// indexing), use `set_tls_ring_capacity()` instead. The two APIs
    /// are complementary: `set_tls_ring_size` controls auto-drain
    /// frequency; `set_tls_ring_capacity` controls the actual ring size.
    static void set_tls_ring_size(std::size_t effective_size) noexcept {
        set_tls_drain_threshold(effective_size);
    }

    /// T-D1 (P2-1): Set the runtime ring capacity for NEW thread_local
    /// rings. Existing threads keep their snapshot until their ring is
    /// reset (e.g., via `reset()` followed by a new `record_access()`).
    ///
    /// This is the true runtime counterpart to the compile-time template
    /// parameter `N`. Unlike `set_tls_ring_size()` (which only adjusts
    /// the auto-drain threshold), this API changes the mask used for
    /// indexing into `buf_[]`, effectively shrinking or growing the
    /// ring's physical capacity (clamped to the compile-time upper
    /// bound `N`).
    ///
    /// Capacity must be a power of 2 in [1, N]. Values outside this
    /// range are clamped: 0 → N (default), > N → N. Non-power-of-2
    /// values are rounded up to the next power of 2 (then clamped to N).
    ///
    /// When a thread first calls `record_access()`, it snapshots the
    /// current `runtime_capacity_` into its per-thread `per_thread_cap_`
    /// field. Subsequent capacity changes do not affect this thread
    /// until its ring is reset (which clears `per_thread_cap_` so the
    /// next `record_access()` re-snapshots).
    ///
    /// Lowering capacity:
    ///   - New threads use fewer slots of `buf_[]` (smaller mask).
    ///   - Existing threads are unaffected until reset.
    ///   - May cause overflow on next `record_access()` if existing
    ///     occupancy exceeds the new capacity; overflow handling
    ///     (silent-drop / flush-callback / assert) kicks in normally.
    ///
    /// Raising capacity (up to N):
    ///   - New threads use more of the pre-allocated `buf_[]`.
    ///   - No reallocation — `buf_` is always N-sized.
    static void set_tls_ring_capacity(std::size_t cap) noexcept {
        if (cap == 0 || cap > kRingSize) {
            cap = kRingSize;
        } else {
            cap = tls_access_ring_config::next_pow2(cap);
            if (cap > kRingSize) cap = kRingSize;
        }
        runtime_capacity_.store(cap, std::memory_order_relaxed);
        // T-D1: also clamp auto_drain_threshold_ to the new capacity.
        // Existing threads' per_thread_cap_ is unaffected (snapshot
        // semantics), but the global auto-drain threshold should not
        // exceed the new global capacity.
        std::size_t cur_threshold = auto_drain_threshold_.load(std::memory_order_relaxed);
        if (cur_threshold > cap) {
            auto_drain_threshold_.store(cap, std::memory_order_relaxed);
        }
    }

    /// T-D1 (P2-1): Query the configured runtime capacity (the value
    /// that will be snapshot by new threads on first access). Existing
    /// threads may have a different capacity if `set_tls_ring_capacity`
    /// was called after their first access.
    static std::size_t tls_ring_capacity() noexcept {
        return runtime_capacity_.load(std::memory_order_relaxed);
    }

    /// T10.3: Returns the global total of auto-drain invocations across
    /// all threads for this `<Key, N>` specialization. Monotonic counter
    /// that only resets when `drain_flush_count()` is called.
    static std::size_t tls_ring_flush_count() noexcept {
        return tls_ring_flush_count_.load(std::memory_order_relaxed);
    }

    /// T10.3: Atomically read and reset the flush counter. Intended for
    /// periodic stats aggregation (e.g. computing flushes/sec).
    static std::size_t drain_flush_count() noexcept {
        return tls_ring_flush_count_.exchange(0, std::memory_order_acq_rel);
    }

    /// Returns the global total of dropped access events across all threads
    /// for this `<Key, N>` specialization. This is a monotonic counter that
    /// only resets when `drain_dropped_count()` is called.
    static std::size_t dropped_count() noexcept {
        return total_dropped_.load(std::memory_order_relaxed);
    }

    /// Atomically read and reset the global dropped counter. Intended for
    /// periodic stats aggregation: the caller incorporates the returned
    /// value into a snapshot (e.g. `cache_stats::tls_ring_dropped_promotions`)
    /// and the counter restarts from zero.
    static std::size_t drain_dropped_count() noexcept {
        return total_dropped_.exchange(0, std::memory_order_relaxed);
    }

    /// Register a flush callback invoked under the `kFlushOnFull` policy
    /// when the ring overflows. The callback is responsible for draining
    /// the ring (e.g. via `drain()`) and promoting the keys; otherwise the
    /// ring falls back to silent-drop behavior to preserve forward progress.
    /// The callback is invoked on the thread that triggered the overflow.
    static void set_flush_callback(std::function<void()> cb) {
        flush_callback_() = std::move(cb);
    }

    // --------------------------------------------------------------------
    // Drain (SubTask 2.2)
    // --------------------------------------------------------------------

    /// Drain all buffered entries, returning them in FIFO order.
    /// Resets the ring to empty state.
    ///
    /// T-D1 (P2-1): Uses `per_thread_cap_` for the count clamp and mask.
    /// If `per_thread_cap_` is 0 (first call before any record_access),
    /// falls back to `runtime_capacity_` then `kRingSize`.
    drain_result drain() {
        const std::size_t cap = effective_capacity();
        const std::size_t mask = cap - 1;
        std::size_t head = head_.load(std::memory_order_relaxed);
        std::size_t tail = tail_.load(std::memory_order_relaxed);
        std::size_t count = head - tail;
        if (count == 0) {
            return drain_result{};
        }
        if (count > cap) {
            // P2-A: Clamp path. Entries beyond `cap` are silently dropped
            // because the ring was overflowed by record_access() without
            // an intervening apply_overflow_policy() call (e.g. head was
            // advanced by record_access but the overflow policy path was
            // preempted before adjusting tail_). Account the dropped count
            // so `tls_ring_dropped_promotions` stays accurate; without
            // this, the drain path would silently lose entries that the
            // overflow policy path thought it had already accounted for.
            const std::size_t dropped = count - cap;
            ++dropped_count_;
            total_dropped_.fetch_add(dropped, std::memory_order_relaxed);
            count = cap;
            tail = head - cap;
        }

        std::vector<Key> result;
        result.reserve(count);

        std::size_t start = tail & mask;
        for (std::size_t i = 0; i < count; ++i) {
            result.push_back(std::move(buf_[(start + i) & mask]));
        }

        tail_.store(head, std::memory_order_relaxed);

        // T-P2-4: Maintain the cross-thread backlog aggregate. Drain is
        // per-thread and far less frequent than `record_access()`, so we
        // can afford a relaxed atomic subtract here. First flush this
        // thread's pending TLS counter to the shared atomic so the
        // subtract operates on a consistent aggregate (some of the drained
        // entries may still be sitting in `tls_pending_backlog_` un-flushed).
        // Then apply a saturated subtract to avoid underflow if record/drain
        // ordering races (shouldn't happen since drain is per-thread, but
        // defensive).
        if (tls_pending_backlog_ > 0) {
            total_backlog_.fetch_add(tls_pending_backlog_,
                std::memory_order_relaxed);
            tls_pending_backlog_ = 0;
        }
        std::size_t current = total_backlog_.load(std::memory_order_relaxed);
        while (current >= count) {
            if (total_backlog_.compare_exchange_weak(current, current - count,
                    std::memory_order_relaxed)) {
                break;
            }
        }
        // If current < count (shouldn't happen), clamp to zero.
        if (current < count) {
            total_backlog_.store(0, std::memory_order_relaxed);
        }

        return drain_result{std::move(result)};
    }

    // --------------------------------------------------------------------
    // Observers
    // --------------------------------------------------------------------

    /// Number of entries currently in the ring.
    std::size_t size() const noexcept {
        return head_.load(std::memory_order_relaxed) - tail_.load(std::memory_order_relaxed);
    }

    /// Whether the ring is empty.
    bool empty() const noexcept {
        return head_.load(std::memory_order_relaxed) == tail_.load(std::memory_order_relaxed);
    }

    /// Whether the ring is ≥ 75% full (heuristic for early flush).
    ///
    /// T-D1 (P2-1): Uses `per_thread_cap_` instead of compile-time `kRingSize`.
    bool should_flush() const noexcept {
        const std::size_t cap = effective_capacity();
        return size() >= cap * 3 / 4;
    }

    /// Discard all entries without returning them.
    ///
    /// T-D1 (P2-1): Also clears `per_thread_cap_` so the next
    /// `record_access()` re-snapshots the global `runtime_capacity_`.
    /// This allows runtime capacity changes to take effect on existing
    /// threads after an explicit `reset()`.
    void reset() {
        tail_.store(head_.load(std::memory_order_relaxed), std::memory_order_relaxed);
        // Clear the per-thread capacity snapshot so the next record_access
        // picks up the current runtime_capacity_. Use relaxed ordering —
        // the snapshot is only read by this thread on the next access.
        per_thread_cap_.store(0, std::memory_order_relaxed);
    }

    // --------------------------------------------------------------------
    // Cross-thread drain support
    // --------------------------------------------------------------------

    /// Reset the thread-local singleton instance for this Key/N combination.
    /// Provides a consistent API with tls_active_item_ring and tls_callback_ring.
    /// Typically called during cache destruction/flush to clean up TLS data.
    ///
    /// Note: This only resets the calling thread's TLS data. Other threads'
    /// data remains in their TLS and cannot be accessed from here.
    static void flush_all_registered() {
        instance().reset();
    }

    /// Drain the TLS rings of *all* registered threads from the calling
    /// thread (Task 3).
    ///
    /// P0-2 Safety Fix: This method NO LONGER directly drains other threads'
    /// ring buffers. Direct cross-thread access to `buf_[]` (which is not
    /// atomic) constituted a data race (undefined behavior). Instead, this
    /// method now:
    ///   1. Drains the calling thread's own ring (safe — same thread).
    ///   2. Drains the global backup buffer (mutex-protected, safe).
    ///   3. Sets `needs_flush_` on every other thread's ring, requesting
    ///      them to drain themselves into the backup buffer on their next
    ///      `record_access()` call.
    ///
    /// This means the returned keys may not include other threads' pending
    /// promotions immediately — they will appear in the backup buffer and
    /// be returned on the next `drain_all_threads()` call after those
    /// threads service the flush request. This is an acceptable trade-off
    /// for memory safety: LRU promotion is idempotent and best-effort,
    /// and deferred promotion has negligible impact on cache hit rate.
    ///
    /// For scenarios requiring synchronous drain of all threads (e.g.,
    /// cache destruction), call this method repeatedly until the result
    /// is empty and `has_backup_keys()` returns false.
    ///
    /// \return All drained keys: the calling thread's ring + backup buffer.
    static drain_result drain_all_threads() {
        std::vector<Key> all_keys;

        // 1. Drain the global backup buffer first (keys from exited threads
        //    and from other threads that serviced a previous flush request).
        auto backup = drain_backup();
        if (!backup.keys.empty()) {
            all_keys = std::move(backup.keys);
        }

        // 2. Drain the calling thread's own ring (safe — same thread owns it).
        //    Use instance() to get the calling thread's ring.
        auto& self = instance();
        if (!self.empty()) {
            auto drained = self.drain();
            if (!drained.keys.empty()) {
                all_keys.insert(all_keys.end(),
                    std::make_move_iterator(drained.keys.begin()),
                    std::make_move_iterator(drained.keys.end()));
            }
        }

        // 3. Request other threads to flush their rings into the backup
        //    buffer on their next record_access() call. We only set the
        //    flag — we do NOT touch their buf_[] array (which would be
        //    a data race since buf_[] is not atomic).
        {
            std::lock_guard<std::mutex> lock(ring_registry_mutex_);
            for (auto& entry : ring_registry_) {
                // Skip the calling thread — already drained above.
                if (entry.tid == std::this_thread::get_id()) continue;
                // Set the flush request flag. The owning thread will
                // service it on its next record_access() call, draining
                // its ring into the backup buffer.
                // T7.1: Only increment pending_drain_count_ when we
                // transition the flag from false to true. This prevents
                // double-counting when drain_all_threads() is called
                // repeatedly while a previous request is still pending.
                bool expected = false;
                if (entry.ring_ptr->needs_flush_.compare_exchange_strong(
                        expected, true, std::memory_order_acq_rel)) {
                    pending_drain_count_.fetch_add(1, std::memory_order_release);
                }
            }
        }

        return drain_result{std::move(all_keys)};
    }

    /// T7.2: Synchronous version of `drain_all_threads()`.
    ///
    /// Issues a flush request to every other thread (exactly like
    /// `drain_all_threads()`) and then blocks until either:
    ///   - every other thread has serviced the flush request
    ///     (pending_drain_count_ drops to zero), OR
    ///   - `timeout` elapses.
    ///
    /// After this function returns (success or timeout), the caller
    /// should call `drain_all_threads()` once more to pick up the keys
    /// that other threads pushed into the backup buffer.
    ///
    /// \tparam Rep   Rep type of std::chrono::duration.
    /// \tparam Period Period type of std::chrono::duration.
    /// \param timeout Maximum time to wait for other threads to drain.
    /// \param poll_interval Sleep interval between polls of
    ///                      pending_drain_count_.
    /// \return true if every other thread serviced the flush request
    ///         before the timeout; false if the timeout elapsed.
    ///
    /// \note This function does NOT itself drain — it only waits for
    ///       other threads to push their keys into the backup buffer.
    ///       Call `drain_all_threads()` afterwards to retrieve them.
    /// \note If a target thread is dormant (never calls record_access()
    ///       again), it cannot service the request — this function will
    ///       time out in that case. The pending request remains set and
    ///       will be serviced if the thread wakes up later.
    template <typename Rep, typename Period>
    static bool drain_all_threads_sync(
        std::chrono::duration<Rep, Period> timeout,
        std::chrono::milliseconds poll_interval = std::chrono::milliseconds(1)) {
        // Issue flush requests to every other thread (increments
        // pending_drain_count_ for each newly-set flag).
        (void)drain_all_threads();

        // If there are no pending requests (no other threads, or they
        // all already serviced the request inline), we're done.
        if (pending_drain_count_.load(std::memory_order_acquire) == 0) {
            return true;
        }

        const auto deadline = std::chrono::steady_clock::now() + timeout;
        while (pending_drain_count_.load(std::memory_order_acquire) > 0) {
            if (std::chrono::steady_clock::now() >= deadline) {
                return false;
            }
            std::this_thread::sleep_for(poll_interval);
        }
        return true;
    }

    /// T7.1: Returns the current number of pending cross-thread drain
    /// requests (i.e., threads that have been asked to flush their TLS
    /// ring into the backup buffer but have not yet done so).
    static std::size_t pending_drain_count() noexcept {
        return pending_drain_count_.load(std::memory_order_acquire);
    }

    /// P1-4 (T3.3): Force-flush dormant threads' TLS rings.
    ///
    /// Scans all registered threads and identifies those whose
    /// `last_activity_ns_` is older than `idle_threshold` (i.e., the
    /// thread has not called `record_access()` in the last
    /// `idle_threshold`). For each dormant thread, safely reads its
    /// ring cross-thread using a seqlock-style validation on
    /// `head_`/`tail_` and pushes the drained keys into the global
    /// backup buffer.
    ///
    /// **Safety guarantee**: The cross-thread read of `buf_[]` is safe
    /// because:
    /// 1. The owning thread has been idle for at least `idle_threshold`
    ///    (default 2s, R6: lowered from 5s for faster detection of
    ///    dormant threads in production read-heavy workloads), so it is
    ///    not currently writing to `buf_[]`.
    /// 2. We validate `head_` and `tail_` before and after the read;
    ///    if they changed (indicating the thread woke up), we abort the
    ///    drain for that ring and skip it this round.
    /// 3. The drained keys are moved (Key move constructor), and the
    ///    owning thread's `tail_` is advanced via atomic CAS to
    ///    reflect the drain. If the owning thread wakes up concurrently
    ///    and tries to drain, the CAS will fail (head_/tail_ changed)
    ///    and the owning thread will simply re-read the new state.
    ///
    /// **Idempotency**: LRU promotion is idempotent, so even if both
    /// the dormant-flush and the owning thread's own drain flush the
    /// same keys, the second drain will see an empty ring (tail_ has
    /// advanced) and produce no duplicate keys.
    ///
    /// \param idle_threshold  Threads idle for longer than this are
    ///                        flushed. Default: 2 seconds (R6: lowered
    ///                        from 5s to detect dormant threads faster
    ///                        and reduce LRU ordering lag in production
    ///                        read-heavy workloads).
    /// \return Number of dormant threads whose rings were drained.
    static std::size_t force_flush_dormant_threads(
            std::chrono::milliseconds idle_threshold = std::chrono::seconds(2)) {
        const auto now_ns = static_cast<std::uint64_t>(
            std::chrono::steady_clock::now().time_since_epoch().count());
        const std::uint64_t threshold_ns =
            static_cast<std::uint64_t>(idle_threshold.count() * 1000000ULL);  // ms → ns

        std::size_t drained_threads = 0;
        std::vector<Key> all_drained_keys;

        {
            std::lock_guard<std::mutex> lock(ring_registry_mutex_);
            for (auto& entry : ring_registry_) {
                // Skip the calling thread — it drains its own ring
                // via the normal path.
                if (entry.tid == std::this_thread::get_id()) continue;

                auto* ring = entry.ring_ptr;
                const std::uint64_t last_activity =
                    ring->last_activity_ns_.load(std::memory_order_acquire);

                // Not dormant — skip.
                if (last_activity == 0) continue;
                if (now_ns - last_activity < threshold_ns) continue;

                // Seqlock-style read: snapshot head_/tail_ before
                // reading buf_[].
                //
                // C-4 fix: Load tail_ FIRST, then head_. head_ only
                // increases and tail_ only increases, with the invariant
                // tail_ <= head_. If we load head first then tail, the
                // owning thread can advance tail_ (via drain/auto-drain)
                // past our stale head between the two loads, making
                // count = head - tail wrap to ~SIZE_MAX. This corrupted
                // `dropped = count - cap` (also ~SIZE_MAX), and the
                // subsequent `total_dropped_.fetch_add(dropped)` overflowed
                // the global dropped counter — surfacing as
                // `tls_ring_dropped_promotions` near UINT64_MAX in
                // ExtendedSoak32Threads.
                //
                // Loading tail first guarantees count >= 0: at the
                // instant we load tail1, tail1 <= head_at_that_instant;
                // head1 (loaded later) >= head_at_that_instant >= tail1.
                std::size_t tail1 = ring->tail_.load(std::memory_order_acquire);
                std::size_t head1 = ring->head_.load(std::memory_order_acquire);
                std::size_t count = head1 - tail1;
                if (count == 0) continue;
                // T-D1 (P2-1): Use the target thread's per-thread capacity
                // (snapshotted at its first record_access). If the thread
                // has not yet snapshotted (per_thread_cap_ == 0), fall back
                // to the global runtime_capacity_, then to kRingSize.
                // Acquire ordering on per_thread_cap_ ensures we observe
                // the owning thread's prior writes to buf_[] (since the
                // snapshot store used release ordering in effective_capacity()).
                std::size_t target_cap = ring->per_thread_cap_.load(std::memory_order_acquire);
                if (target_cap == 0 || target_cap > kRingSize) {
                    target_cap = runtime_capacity_.load(std::memory_order_relaxed);
                    if (target_cap == 0 || target_cap > kRingSize) {
                        target_cap = kRingSize;
                    }
                }
                std::size_t dropped = 0;
                if (count > target_cap) {
                    // P2-A: Cross-thread clamp path. Same situation as
                    // drain(): the dormant thread overflowed its ring
                    // without an intervening apply_overflow_policy()
                    // adjustment, so count exceeds the physical capacity.
                    // We can only see this ring cross-thread via seqlock,
                    // so we cannot safely touch the owning thread's
                    // thread_local `dropped_count_`; only the global
                    // atomic `total_dropped_` is updated here. The owning
                    // thread will not double-count because it is dormant
                    // (no further record_access / drain calls).
                    //
                    // C-4: Defer total_dropped_.fetch_add until AFTER
                    // seqlock validation passes. Computing `dropped` here
                    // is safe (count can no longer wrap thanks to the
                    // tail-first load order above), but charging it to
                    // total_dropped_ before validation would corrupt the
                    // counter if the owning thread wakes up mid-read and
                    // the snapshot is discarded. We only account drops
                    // for snapshots that pass validation AND the CAS.
                    dropped = count - target_cap;
                    count = target_cap;
                    tail1 = head1 - target_cap;
                }

                // Read buf_[] using the snapshot. The mask is derived
                // from the target thread's capacity — NOT kMask — so we
                // correctly index the slots the writer used.
                std::vector<Key> batch;
                batch.reserve(count);
                const std::size_t target_mask = target_cap - 1;
                const std::size_t start = tail1 & target_mask;
                for (std::size_t i = 0; i < count; ++i) {
                    // Copy (not move) — the owning thread may still
                    // wake up and read these slots. Move would leave
                    // the slot in a moved-from state, which is safe
                    // for std::vector but not for arbitrary Key types.
                    batch.push_back(ring->buf_[(start + i) & target_mask]);
                }

                // Seqlock validation: re-read head_/tail_. If they
                // changed, the thread woke up and wrote to buf_[] —
                // abort this drain to avoid using inconsistent data.
                // C-4: same tail-first ordering as the initial snapshot.
                std::size_t tail2 = ring->tail_.load(std::memory_order_acquire);
                std::size_t head2 = ring->head_.load(std::memory_order_acquire);
                if (head1 != head2 || tail1 != tail2) {
                    // The thread woke up — skip this round. The
                    // owning thread will drain itself on its next
                    // record_access() call (or via needs_flush_ flag).
                    //
                    // C-4: We also skip the total_dropped_.fetch_add
                    // below for the overflow case — the dropped
                    // accounting is only meaningful for snapshots that
                    // pass validation. Aborted snapshots might have
                    // stale or inconsistent counts, and charging them
                    // to total_dropped_ would corrupt the counter
                    // (the owning thread will account drops itself
                    // when it next drains).
                    continue;
                }

                // Validation passed — the thread is still dormant
                // and the snapshot is consistent. Advance tail_ via
                // CAS to mark these entries as drained. If the CAS
                // fails, the thread woke up and changed head_/tail_;
                // we skip this round (the keys are in our batch but
                // also still in the ring — the owning thread will
                // drain them itself).
                if (!ring->tail_.compare_exchange_strong(
                        tail1, head1,
                        std::memory_order_acq_rel,
                        std::memory_order_acquire)) {
                    continue;
                }

                // C-4: Now that the CAS succeeded (the snapshot was
                // consistent and we've claimed the entries), account
                // any overflow drops. This is the ONLY place we charge
                // total_dropped_ for the cross-thread path — deferred
                // from the count > target_cap check above to ensure
                // we only account drops for validated, claimed drains.
                if (dropped > 0) {
                    total_dropped_.fetch_add(dropped, std::memory_order_relaxed);
                }

                // Successfully drained — append the batch to the
                // global result.
                all_drained_keys.insert(all_drained_keys.end(),
                    std::make_move_iterator(batch.begin()),
                    std::make_move_iterator(batch.end()));
                ++drained_threads;
            }
        }

        // Push all drained keys to the backup buffer (mutex-protected).
        if (!all_drained_keys.empty()) {
            push_to_backup(std::move(all_drained_keys));
        }

        return drained_threads;
    }

    /// Aggregate dropped-promotion count across all threads (Task 2).
    ///
    /// Returns the total number of record_access() calls that were
    /// silently dropped (or, under kFlushOnFull, dropped after the
    /// flush callback declined to consume them) across every thread
    /// since the last drain_dropped_count() reset. The counter is a
    /// single std::atomic accumulated from each thread
    /// dropped_count_ on every drop, so it reflects a globally
    /// consistent total without requiring a registry walk.
    ///
    /// \note This reads the global atomic total_dropped_, *not* the
    ///       calling thread dropped_count_. The per-thread counter
    ///       is static thread_local, so reading it from this thread
    ///       would only ever return this thread local view - useless
    ///       for cross-thread aggregation. The global atomic is the
    ///       correct source of truth for the aggregate.
    static std::size_t dropped_count_all_threads() noexcept {
        return total_dropped_.load(std::memory_order_relaxed);
    }

    /// T-P2-4: Cross-thread aggregate of the live backlog (pending
    /// un-drained access entries) across all threads. The hot path
    /// (`record_access()`) updates a thread-local counter and only
    /// batch-flushes to `total_backlog_` every `kBacklogFlushBatch`
    /// increments, so this query returns the shared atomic plus the
    /// *calling* thread's not-yet-flushed pending count. Other threads'
    /// pending increments are not visible until their next batch flush
    /// (bounded staleness of <= `kBacklogFlushBatch` increments per thread),
    /// which is acceptable for diagnostics / `stats_snapshot()`. Cheaper
    /// than walking the thread registry.
    static std::size_t total_backlog() noexcept {
        return total_backlog_.load(std::memory_order_relaxed)
             + tls_pending_backlog_;
    }

    // --------------------------------------------------------------------
    // Backup ring — thread-exit safety for deferred promotions
    // --------------------------------------------------------------------

    /// Push keys from a dying thread's TLS ring into the global backup buffer.
    /// Called automatically by the thread-local sentinel when a thread exits.
    /// Safe to call from any thread; uses a mutex for exclusive access.
    ///
    /// P1-B: Sets `has_backup_keys_` to true under the lock so that
    /// `has_backup_keys()` can fast-path on a single atomic load without
    /// acquiring the mutex on the common (empty) path.
    static void push_to_backup(std::vector<Key>&& keys) {
        if (keys.empty()) return;
        auto& bk = backup_buffer();
        std::lock_guard<std::mutex> lock(bk.mutex);
        bk.keys.insert(bk.keys.end(),
            std::make_move_iterator(keys.begin()),
            std::make_move_iterator(keys.end()));
        // Release-store pairs with the acquire-load in has_backup_keys().
        // Under the mutex so the store is serialization-consistent: any
        // thread that observes the flag set will also observe the keys.
        has_backup_keys_.store(true, std::memory_order_release);
    }

    /// Drain the global backup buffer, returning all keys accumulated from
    /// exited threads. After this call the backup buffer is empty.
    /// Should be called by drain_access_ring() before draining the TLS ring
    /// so that orphaned keys from exited threads get promoted.
    ///
    /// P1-B: Clears `has_backup_keys_` under the lock so subsequent
    /// fast-path checks skip the mutex. New pushes after unlock will
    /// re-set the flag.
    static drain_result drain_backup() {
        auto& bk = backup_buffer();
        std::vector<Key> result;
        {
            std::lock_guard<std::mutex> lock(bk.mutex);
            result.swap(bk.keys);
            // After swap, bk.keys is empty (swapped with result's initial
            // empty state). Clear the fast-path flag under the lock so
            // that a concurrent push that's waiting on the mutex will
            // re-set it after we unlock.
            has_backup_keys_.store(false, std::memory_order_release);
        }
        return drain_result{std::move(result)};
    }

    /// Whether the global backup buffer has any keys.
    ///
    /// P1-B: Fast-path on a single atomic load. When the flag is false
    /// (the common case — no thread has exited with pending promotions),
    /// this avoids the mutex acquisition entirely. When the flag is true,
    /// falls back to a mutex-protected check (the flag may be stale if
    /// another thread already drained the buffer).
    static bool has_backup_keys() {
        if (!has_backup_keys_.load(std::memory_order_acquire)) return false;
        auto& bk = backup_buffer();
        std::lock_guard<std::mutex> lock(bk.mutex);
        return !bk.keys.empty();
    }

    // --------------------------------------------------------------------
    // Configuration (SubTask 2.3)
    // --------------------------------------------------------------------

    /// Get the compile-time ring size (must be power of 2).
    static constexpr std::size_t ring_size() noexcept {
        return kRingSize;
    }

    /// Validate that a given size is a valid ring size (power of 2, > 0).
    static constexpr bool is_valid_ring_size(std::size_t s) noexcept {
        return s > 0 && (s & (s - 1)) == 0;
    }

    // --------------------------------------------------------------------
    // Thread-local instance (SubTask 2.1)
    // --------------------------------------------------------------------

    /// Get the thread-local singleton instance for this Key/N combination.
    /// A thread-local sentinel is installed alongside the ring to push
    /// remaining keys to the global backup buffer when the thread exits,
    /// and to unregister the ring from the cross-thread registry.
    static tls_access_ring& instance() {
        auto& ring = get_tl_ring();
        // Install the thread-exit sentinel on first access.
        // The sentinel's destructor fires when this thread exits,
        // draining remaining keys into the global backup buffer and
        // unregistering from ring_registry_.
        thread_local thread_exit_sentinel sentinel;
        (void)sentinel;  // sole purpose is its destructor
        // Register this thread's ring in the cross-thread registry on
        // first access. drain_all_threads() walks the registry to request
        // lazy flushes on other threads without touching their thread_local
        // state directly.
        thread_local bool registered = []() {
            std::lock_guard<std::mutex> lock(ring_registry_mutex_);
            ring_registry_.push_back(
                ring_registry_entry{std::this_thread::get_id(), &get_tl_ring()});
            return true;
        }();
        (void)registered;
        return ring;
    }

private:
    static constexpr std::size_t kMask = N - 1;

    std::array<Key, N> buf_{};
    alignas(64) std::atomic<std::size_t> head_{0};  // write position (monotonically increasing)
    alignas(64) std::atomic<std::size_t> tail_{0};  // oldest unread position

    /// T-D1 (P2-1): Per-thread snapshot of the global `runtime_capacity_`,
    /// taken on first `record_access()` (or after `reset()`). Used as the
    /// effective ring capacity for this thread's mask and overflow checks.
    ///
    /// Semantics: 0 means "not yet snapshotted" — `effective_capacity()`
    /// will populate it on the next call. Once populated, it stays stable
    /// until `reset()` clears it (allowing the next access to re-snapshot
    /// a potentially updated `runtime_capacity_`).
    ///
    /// Atomic for cross-thread reads in `force_flush_dormant_threads`:
    /// the reader uses acquire ordering to establish happens-before with
    /// the owning thread's prior writes to `buf_[]` (the snapshot store
    /// in `effective_capacity()` uses release ordering).
    alignas(64) std::atomic<std::size_t> per_thread_cap_{0};

    /// T-D1 (P2-1): Helper that returns the effective capacity for the
    /// calling thread. On first call (or after `reset()`), snapshots the
    /// global `runtime_capacity_` into `per_thread_cap_` and returns it.
    /// Subsequent calls return the cached snapshot.
    std::size_t effective_capacity() noexcept {
        std::size_t cap = per_thread_cap_.load(std::memory_order_relaxed);
        if (cap != 0) return cap;
        // Snapshot the global runtime capacity. Validate and clamp to
        // [1, kRingSize]; if runtime_capacity_ was never set or is
        // invalid, fall back to kRingSize.
        cap = runtime_capacity_.load(std::memory_order_relaxed);
        if (cap == 0 || cap > kRingSize) {
            cap = kRingSize;
        }
        // Cache the snapshot. Release ordering so cross-thread readers
        // (force_flush_dormant_threads) observing per_thread_cap_ via
        // acquire also observe all prior writes to buf_[] (head_ is
        // monotonically increasing, so any subsequent write to
        // buf_[pos & mask] happens after this store).
        per_thread_cap_.store(cap, std::memory_order_release);
        return cap;
    }

    /// T-D1 (P2-1): const-overload of effective_capacity() for use in
    /// const member functions (should_flush, drain). Uses relaxed load
    /// because const methods are read-only; if per_thread_cap_ is 0
    /// (not yet snapshotted), the call site should be a record_access()
    /// which would have snapshotted first. As a fallback, we read
    /// runtime_capacity_ and return it without caching (caching would
    /// require mutable).
    std::size_t effective_capacity() const noexcept {
        std::size_t cap = per_thread_cap_.load(std::memory_order_relaxed);
        if (cap != 0) return cap;
        cap = runtime_capacity_.load(std::memory_order_relaxed);
        if (cap == 0 || cap > kRingSize) {
            cap = kRingSize;
        }
        return cap;
    }

    /// Flag set by `drain_all_threads()` to request that this ring drain
    /// itself on the owning thread's next `record_access()` call. Stored
    /// per-instance (each thread_local ring has its own flag). Cross-thread
    /// reads/writes are safe because the flag is atomic.
    std::atomic<bool> needs_flush_{false};

    /// P1-4 (T3.3): Heartbeat timestamp — last time this thread called
    /// record_access(). Updated with release semantics so that a
    /// cross-thread reader can use acquire semantics to establish a
    /// happens-before relationship with the owning thread's writes to
    /// buf_[].
    ///
    /// Used by `force_flush_dormant_threads(timeout)` to detect threads
    /// that have gone idle (e.g., a worker thread blocked on I/O or
    /// descheduled by the OS). When a thread is idle for longer than
    /// the threshold, its pending TLS ring entries are at risk of being
    /// lost (the thread may never call record_access() again to service
    /// a flush request). The dormant-flush mechanism safely reads the
    /// ring cross-thread using a seqlock-style validation on head_/tail_.
    alignas(64) std::atomic<std::uint64_t> last_activity_ns_{0};

    // ----------------------------------------------------------------
    // Overflow accounting (per <Key, N> specialization)
    // ----------------------------------------------------------------

    /// Per-thread count of dropped access events. Incremented under
    /// `kSilentDrop` / `kAssertOnFull` policies, or when `kFlushOnFull`
    /// falls back to dropping. Useful for per-thread diagnostics; the
    /// global aggregate lives in `total_dropped_`.
    static inline thread_local std::size_t dropped_count_{0};

    /// Global aggregate of dropped access events across all threads for
    /// this `<Key, N>` specialization. Read via `dropped_count()` and
    /// atomically drained via `drain_dropped_count()`.
    static inline std::atomic<std::size_t> total_dropped_{0};

    /// T-P2-4: Global aggregate of the live backlog (pending un-drained
    /// access entries) across all threads for this `<Key, N>` specialization.
    /// Now updated only by periodic batch-flushes from each thread's
    /// `tls_pending_backlog_` (every `kBacklogFlushBatch` increments in
    /// `record_access()`) and by `drain()` (saturated subtract). Read via
    /// `total_backlog()`, which adds the calling thread's pending TLS
    /// counter. Kept on its own cache line to avoid false sharing. This
    /// enables `diagnostics()` to report the cross-thread TLS ring backlog
    /// without walking the thread registry, while removing the per-
    /// `record_access()` cache-line contention of the previous design.
    alignas(64) static inline std::atomic<std::size_t> total_backlog_{0};

    /// T-P2-4: Per-thread (thread-local) pending backlog counter. The hot
    /// path (`record_access()`) increments this pure-TLS counter instead of
    /// contending on the shared `total_backlog_` atomic, eliminating
    /// cache-line contention under high read concurrency. Every
    /// `kBacklogFlushBatch` increments the accumulated count is batch-flushed
    /// to `total_backlog_` with a single relaxed `fetch_add` (and the TLS
    /// counter reset to 0). On `drain()` the pending TLS counter is first
    /// flushed to the atomic and then the subtract is applied to the atomic,
    /// keeping the aggregate consistent. `total_backlog()` returns
    /// `total_backlog_ + tls_pending_backlog_`, reflecting the calling
    /// thread's pending count; other threads' not-yet-flushed increments
    /// become visible only on their next batch flush (bounded staleness).
    static inline thread_local std::size_t tls_pending_backlog_{0};

    /// T-P2-4: Number of `record_access()` increments accumulated in
    /// `tls_pending_backlog_` before a batch flush to the shared
    /// `total_backlog_` atomic. A power of two so the modulo is a cheap
    /// bitmask. 64 keeps the per-thread footprint tiny while bounding the
    /// aggregate's staleness to <= 64 increments per thread.
    static constexpr std::size_t kBacklogFlushBatch = 64;

    /// Number of `record_access()` calls since the last heartbeat update.
    /// `last_activity_ns_` is only refreshed every `kHeartbeatInterval`
    /// accesses — dormancy detection only needs to know "this thread did
    /// something recently", and a batch-boundary update keeps that true
    /// while eliminating the `steady_clock::now()` read from every hit.
    /// Per-instance (each thread_local ring owns one), single-writer.
    std::uint32_t heartbeat_skip_{0};
    static constexpr std::uint32_t kHeartbeatInterval = 64;

    /// T7.1: Global counter tracking the number of pending cross-thread
    /// drain requests. Incremented in `drain_all_threads()` when we
    /// transition a thread's `needs_flush_` flag from false to true;
    /// decremented in `record_access()` when the owning thread services
    /// the request. `drain_all_threads_sync()` blocks until this counter
    /// reaches zero or the timeout elapses.
    ///
    /// Placed on its own cache line to avoid false sharing with the
    /// per-thread `needs_flush_` flags written by other threads.
    alignas(64) static inline std::atomic<std::size_t> pending_drain_count_{0};

    /// T10.2: Auto-drain threshold. When the ring's occupancy reaches
    /// this value, the next `record_access()` synchronously drains.
    /// Default is kRingSize / 2 (R6: lowered from kRingSize to enable
    /// proactive draining before overflow, bounding worst-case drain
    /// latency). Set lower via `set_tls_drain_threshold()` to cap drain
    /// latency further; set to kRingSize to disable auto-drain.
    static inline std::atomic<std::size_t> auto_drain_threshold_{kRingSize / 2};

    /// T-D1 (P2-1): Global runtime ring capacity. Read by new threads
    /// on their first `record_access()` (via `effective_capacity()`) and
    /// cached in `per_thread_cap_`. Changes only affect threads whose
    /// `per_thread_cap_` is 0 (i.e., never accessed or post-`reset()`).
    ///
    /// Default is `kRingSize` (N) — equivalent to the historical
    /// compile-time-only behavior. Lower it via `set_tls_ring_capacity()`
    /// to use fewer slots of `buf_[]` (smaller mask). Raise it back up
    /// to N to use more of the pre-allocated buffer (no reallocation).
    ///
    /// Placed on its own cache line: read by every `effective_capacity()`
    /// on cache miss (first access per thread) and by every
    /// `force_flush_dormant_threads()` call; write rarely
    /// (`set_tls_ring_capacity()`).
    alignas(64) static inline std::atomic<std::size_t> runtime_capacity_{kRingSize};

    /// T-D2 (P2-2): Thread-local pointer to the active per-cache config.
    /// Set via `set_active_config()` (typically through `active_config_scope`
    /// RAII guard). When null, `record_access()` and `apply_overflow_policy()`
    /// fall back to the static defaults (`full_policy_`, `auto_drain_threshold_`,
    /// `flush_callback_()`), preserving historical behavior.
    ///
    /// Thread-local so each thread can have a different active cache (e.g.
    /// a request handler thread alternating between caches in the same
    /// process). Set/unset on the calling thread only — no cross-thread
    /// synchronization needed.
    ///
    /// `inline` (C++17) is required because this is a header-only library:
    /// without it, multiple translation units including this header would
    /// cause ODR violations / linker duplicate-symbol errors. `static inline`
    /// thread_local members are merged by the linker into a single per-thread
    /// instance per `<Key, N>` specialization.
    inline static thread_local tls_ring_config* active_config_ = nullptr;

    /// T10.3: Global counter of auto-drain invocations. Read via
    /// `tls_ring_flush_count()` and atomically drained via
    /// `drain_flush_count()`. Used to compute flushes/sec for the
    /// `tls_ring_flush_per_sec` monitoring metric.
    alignas(64) static inline std::atomic<std::size_t> tls_ring_flush_count_{0};

    /// P1-B: Fast-path flag for `has_backup_keys()`. Set to true under
    /// the backup buffer mutex in `push_to_backup()`, cleared under the
    /// same mutex in `drain_backup()`. Read with acquire ordering by
    /// `has_backup_keys()` to skip the mutex on the common empty path.
    ///
    /// Placed on its own cache line: written by every thread exit
    /// (`push_to_backup`) and read by every `drain_access_ring()` call,
    /// so separating it from `tls_ring_flush_count_` avoids false sharing
    /// between the (rare) push path and the (frequent) drain fast-path.
    alignas(64) static inline std::atomic<bool> has_backup_keys_{false};

    /// Current overflow policy (atomic for cross-thread reads). Per
    /// `<Key, N>` specialization, shared across all threads.
    ///
    /// P1-5: Default changed from `kSilentDrop` to `kFlushOnFull` so that
    /// production workloads don't silently drop access traces (which would
    /// degrade LRU accuracy under burst traffic). Falls back to silent-drop
    /// if the flush callback fails to drain the ring.
    static inline std::atomic<tls_ring_full_policy> full_policy_{
        tls_ring_full_policy::kFlushOnFull};

    /// Meyers-singleton accessor for the flush callback. Avoids
    /// static-initialization-order issues across translation units that
    /// a `static inline std::function` member would face.
    static std::function<void()>& flush_callback_() {
        static std::function<void()> cb;
        return cb;
    }

    /// Apply the configured overflow policy. Called from `record_access()`
    /// when the ring has overflowed (`head_ - tail_ > cap`).
    ///
    /// T-D1 (P2-1): Uses `effective_capacity()` instead of compile-time
    /// `kRingSize` for the tail reset. This ensures overflow handling
    /// respects the runtime-configured capacity.
    ///
    /// T-D2 (P2-2): Prefers the per-cache active config's `full_policy`
    /// and `flush_callback` when set (via `set_active_config()` /
    /// `active_config_scope`). This allows two cache instances to use
    /// different overflow policies (e.g. one kSilentDrop, one kFlushOnFull
    /// with a different drain target) without falling back to the
    /// `<Key, N>`-specialization-wide static default. When the active
    /// config is null, fall back to the static default (`full_policy_`
    /// and `flush_callback_()`).
    void apply_overflow_policy() {
        tls_ring_config* cfg = active_config_;
        const auto policy = cfg
            ? cfg->full_policy.load(std::memory_order_relaxed)
            : full_policy_.load(std::memory_order_relaxed);
        const std::size_t cap = effective_capacity();
        switch (policy) {
            case tls_ring_full_policy::kSilentDrop:
                ++dropped_count_;
                total_dropped_.fetch_add(1, std::memory_order_relaxed);
                tail_.store(head_.load(std::memory_order_relaxed) - cap, std::memory_order_relaxed);
                break;
            case tls_ring_full_policy::kFlushOnFull: {
                // T-D2: prefer the per-cache flush_callback from the active
                // config; fall back to the static default when no config
                // is set.
                if (cfg) {
                    auto& cb = cfg->flush_callback;
                    if (cb) cb();
                } else {
                    auto& cb = flush_callback_();
                    if (cb) cb();
                }
                // If the callback did not drain (still overflowing), fall
                // back to silent-drop semantics to preserve forward progress.
                if (head_.load(std::memory_order_relaxed) - tail_.load(std::memory_order_relaxed) > cap) {
                    ++dropped_count_;
                    total_dropped_.fetch_add(1, std::memory_order_relaxed);
                    tail_.store(head_.load(std::memory_order_relaxed) - cap, std::memory_order_relaxed);
                }
                break;
            }
            case tls_ring_full_policy::kAssertOnFull:
                assert(false && "tls_access_ring overflow under kAssertOnFull policy");
                // In release builds (NDEBUG) the assertion is a no-op;
                // recover by dropping the oldest entry, matching kSilentDrop.
                ++dropped_count_;
                total_dropped_.fetch_add(1, std::memory_order_relaxed);
                tail_.store(head_.load(std::memory_order_relaxed) - cap, std::memory_order_relaxed);
                break;
        }
    }

    // ----------------------------------------------------------------
    // Global backup buffer — collects keys from exited threads
    // ----------------------------------------------------------------

    /// Shared backup storage protected by a mutex.
    /// T7.5: Pre-reserves capacity to avoid heap allocations on the
    /// thread-exit path (which runs in the sentinel destructor and
    /// should be wait-free in the common case). The default reserve
    /// covers ~16 thread exits × ring size N each without growing.
    struct backup_storage {
        backup_storage() {
            // T7.5: reserve upfront for the common case. The vector
            // still grows on demand if more threads exit than expected.
            keys.reserve(16 * kRingSize);
        }
        std::mutex mutex;
        std::vector<Key> keys;
    };

    /// Meyers singleton for the backup buffer — one per Key/N specialization.
    static backup_storage& backup_buffer() {
        static backup_storage storage;
        return storage;
    }

    // ----------------------------------------------------------------
    // Cross-thread registry — tracks all live thread_local rings for
    // drain_all_threads() (Task 3).
    // ----------------------------------------------------------------

    struct ring_registry_entry {
        std::thread::id tid;
        tls_access_ring* ring_ptr;
    };

    static inline std::mutex ring_registry_mutex_;
    static inline std::vector<ring_registry_entry> ring_registry_;

    // ----------------------------------------------------------------
    // Thread-exit sentinel — pushes remaining TLS keys to backup and
    // unregisters from the cross-thread registry.
    // ----------------------------------------------------------------

    /// A lightweight thread-local object whose sole purpose is to detect
    /// thread exit. When a thread that has used tls_access_ring exits,
    /// the sentinel's destructor drains the TLS ring and pushes any
    /// remaining keys into the global backup buffer so they can be
    /// promoted by any surviving thread's drain_access_ring(). It also
    /// removes the ring from `ring_registry_` so `drain_all_threads()`
    /// never observes a dangling pointer.
    struct thread_exit_sentinel {
        ~thread_exit_sentinel() {
            // Access the thread-local ring directly (same thread, safe).
            // We cannot call instance() here because it would create
            // a new sentinel recursively. Instead, reach into the
            // thread_local ring directly via get_tl_ring().
            auto& ring = get_tl_ring();
            // T7.1: If another thread is currently blocked in
            // drain_all_threads_sync() waiting on pending_drain_count_,
            // service the flush request here so it doesn't time out
            // waiting for a thread that's about to exit.
            if (ring.needs_flush_.load(std::memory_order_acquire)) {
                ring.needs_flush_.store(false, std::memory_order_release);
                if (pending_drain_count_.load(std::memory_order_relaxed) > 0) {
                    pending_drain_count_.fetch_sub(1, std::memory_order_release);
                }
            }
            if (!ring.empty()) {
                auto drained = ring.drain();
                if (!drained.keys.empty()) {
                    push_to_backup(std::move(drained.keys));
                }
            }
            // Unregister from the cross-thread registry so that
            // drain_all_threads() does not observe a dangling pointer.
            // The ring object itself stays alive until the thread_local
            // storage is torn down (immediately after sentinel destruction),
            // but we cannot rely on that from other threads.
            std::lock_guard<std::mutex> lock(ring_registry_mutex_);
            const auto tid = std::this_thread::get_id();
            ring_registry_.erase(
                std::remove_if(ring_registry_.begin(), ring_registry_.end(),
                    [tid](const ring_registry_entry& e) { return e.tid == tid; }),
                ring_registry_.end());
        }
    };

    /// The single source of truth for the thread-local ring.
    /// Both instance() and thread_exit_sentinel use this to ensure
    /// they reference the same ring object.
    static tls_access_ring& get_tl_ring() {
        thread_local tls_access_ring ring;
        return ring;
    }
};

// ============================================================================
// TLS Active Item Ring
// ============================================================================

/// A thread-local ring buffer that accumulates recently-accessed keys and
/// flushes them in batch to the global LRU under a single lock acquisition.
///
/// @tparam Key      The cache key type.
/// @tparam RingSize Number of entries per ring (must be a power of 2 for
///                  efficient modulo operations).
template <typename Key, std::size_t RingSize = 64>
class tls_active_item_ring {
    static_assert(RingSize > 0, "RingSize must be positive");
    static_assert((RingSize & (RingSize - 1)) == 0, "RingSize must be a power of 2");

public:
    using key_type = Key;

    static constexpr std::size_t kRingSize = RingSize;

    tls_active_item_ring() : instance_id_(next_instance_id_.fetch_add(1, std::memory_order_relaxed)) {
        std::lock_guard<std::mutex> lock(registry_mutex_);
        registry_.push_back({instance_id_, this});
    }

    ~tls_active_item_ring() {
        // Unregister from global registry
        {
            std::lock_guard<std::mutex> lock(registry_mutex_);
            registry_.erase(
                std::remove_if(registry_.begin(), registry_.end(),
                    [this](const auto& e) { return e.instance_ptr == this; }),
                registry_.end());
        }
        // Remove this instance's ring_data from all threads that have accessed it.
        // Since we can't enumerate all threads, we mark the entry for lazy cleanup
        // by erasing from the current thread's map. Other threads' entries will be
        // cleaned up lazily on their next access (see get_ring()).
        get_thread_data().rings.erase(instance_id_);
    }

    // Non-copyable, non-movable (instance_id_ is identity)
    tls_active_item_ring(const tls_active_item_ring&) = delete;
    tls_active_item_ring& operator=(const tls_active_item_ring&) = delete;
    tls_active_item_ring(tls_active_item_ring&&) = delete;
    tls_active_item_ring& operator=(tls_active_item_ring&&) = delete;

    // --------------------------------------------------------------------
    // Access tracking
    // --------------------------------------------------------------------

    /// Record that a key was accessed (hit or write).
    /// This is fast: just stores the key in the ring buffer (no lock, no hash lookup).
    ///
    /// @return true if the ring is now full and should be flushed.
    bool record(const key_type& key) {
        auto& rd = this->get_ring();
        // If ring is full (unflushed data would be overwritten), advance tail
        if (rd.head - rd.tail >= kRingSize) {
            rd.tail = rd.head - kRingSize + 1;
        }
        auto pos = rd.head;
        rd.keys[pos & kMask] = key;
        rd.head = pos + 1;
        return (pos + 1) % kRingSize == 0;
    }

    /// Record with move semantics (avoids a copy for large keys).
    bool record(key_type&& key) {
        auto& rd = this->get_ring();
        if (rd.head - rd.tail >= kRingSize) {
            rd.tail = rd.head - kRingSize + 1;
        }
        auto pos = rd.head;
        rd.keys[pos & kMask] = std::move(key);
        rd.head = pos + 1;
        return (pos + 1) % kRingSize == 0;
    }

    // --------------------------------------------------------------------
    // Flush control
    // --------------------------------------------------------------------

    /// Check if the ring should be flushed (≥ 75% full).
    bool should_flush() const noexcept {
        auto& rd = this->get_ring();
        auto head = rd.head;
        auto tail = rd.tail;
        std::size_t used = (head >= tail) ? (head - tail) : 0;
        return used >= kRingSize * 3 / 4;
    }

    /// Number of entries currently in the ring.
    std::size_t size() const noexcept {
        auto& rd = this->get_ring();
        auto head = rd.head;
        auto tail = rd.tail;
        return head > tail ? head - tail : 0;
    }

    /// Whether the ring is completely empty.
    bool empty() const noexcept {
        return size() == 0;
    }

    /// Force a flush (useful before releasing a lock or at thread shutdown).
    template <typename Func>
    void flush_to(Func&& func) {
        auto& rd = this->get_ring();
        auto head = rd.head;
        auto tail = rd.tail;

        if (head == tail) return; // empty

        // Deduplicate: use a small hash set for O(n) dedup per batch
        std::size_t count = (head >= tail) ? (head - tail) : 0;
        std::size_t start = tail & kMask;
        ankerl::unordered_dense::set<key_type> seen;
        seen.reserve(count);

        for (std::size_t i = 0; i < count; ++i) {
            auto idx = (start + i) & kMask;
            const auto& k = rd.keys[idx];
            if (seen.insert(k).second) {
                func(k);
            }
        }

        rd.tail = head;
    }

    /// Flush to a batch callback (receives a span of unique keys).
    /// The callback is invoked once with all unique keys — suitable for
    /// bulk-promoting under a single lock acquisition.
    template <typename Func>
    void flush_batch(Func&& func) {
        auto& rd = this->get_ring();
        auto head = rd.head;
        auto tail = rd.tail;

        if (head == tail) return;

        std::size_t count = (head >= tail) ? (head - tail) : 0;
        std::size_t start = tail & kMask;

        // Collect unique keys via hash set (O(n) vs previously O(n²))
        ankerl::unordered_dense::set<key_type> seen;
        seen.reserve(count);
        for (std::size_t i = 0; i < count; ++i) {
            auto idx = (start + i) & kMask;
            seen.insert(rd.keys[idx]);
        }

        std::vector<key_type> unique_keys;
        unique_keys.reserve(seen.size());
        for (auto& k : seen) {
            unique_keys.push_back(std::move(k));
        }

        func(unique_keys);

        rd.tail = head;
    }

    /// Get raw count without deduplication (for metrics).
    std::size_t raw_count() const noexcept {
        auto& rd = this->get_ring();
        auto head = rd.head;
        auto tail = rd.tail;
        return head > tail ? head - tail : 0;
    }

    /// Reset the ring (discard all entries).
    void reset() {
        auto& rd = this->get_ring();
        auto head = rd.head;
        rd.tail = head;
    }

    // --------------------------------------------------------------------
    // Global statistics (across all threads)
    // --------------------------------------------------------------------

    /// Increment a shared counter for total accesses (useful for global hit counting).
    static void increment_global(std::atomic<std::size_t>& counter) {
        counter.fetch_add(1, std::memory_order_relaxed);
    }

    // --------------------------------------------------------------------
    // Cross-thread drain support
    // --------------------------------------------------------------------

    /// Reset all registered instances' current-thread ring data.
    /// Iterates the global registry and calls reset() on each live instance,
    /// discarding the calling thread's pending keys.
    /// Typically called during cache destruction to clean up TLS data.
    ///
    /// Note: This only resets the calling thread's TLS data. Other threads'
    /// data remains in their TLS and cannot be accessed from here.
    static void flush_all_registered() {
        std::vector<tls_active_item_ring*> instances;
        {
            std::lock_guard<std::mutex> lock(registry_mutex_);
            instances.reserve(registry_.size());
            for (auto& e : registry_) {
                instances.push_back(e.instance_ptr);
            }
        }
        for (auto* inst : instances) {
            inst->reset();
        }
    }

private:
    static constexpr std::size_t kMask = RingSize - 1;

    /// Per-thread ring data (thread_local to avoid false sharing across threads).
    struct ring_data {
        std::array<key_type, RingSize> keys{};
        std::size_t head{0};  // producer index (write position)
        std::size_t tail{0};  // consumer index (read position after flush)

        ring_data() = default;
    };

    /// Per-thread, per-instance ring data map.
    /// Each tls_active_item_ring instance gets its own ring_data per thread,
    /// ensuring that multiple instances on the same thread don't share state.
    struct thread_data {
        ankerl::unordered_dense::map<uint64_t, std::unique_ptr<ring_data>> rings;
    };

    static thread_data& get_thread_data() {
        thread_local thread_data td;
        return td;
    }

    ring_data& get_ring() const {
        auto& td = get_thread_data();
        auto it = td.rings.find(instance_id_);
        if (it == td.rings.end()) {
            auto [insert_it, _] = td.rings.emplace(instance_id_, std::make_unique<ring_data>());
            return *insert_it->second;
        }
        return *it->second;
    }

    static inline std::atomic<uint64_t> next_instance_id_{0};
    uint64_t instance_id_;

    // Global registry for cross-thread drain support
    struct instance_registry_entry {
        uint64_t instance_id;
        tls_active_item_ring* instance_ptr;
    };

    static inline std::mutex registry_mutex_;
    static inline std::vector<instance_registry_entry> registry_;
};

// ============================================================================
// Concurrent Ring (larger, for high-throughput scenarios)
// ============================================================================

/// For scenarios with > 1M ops/s per thread, a larger ring helps.
/// Same API, different default ring size.
template <typename Key, std::size_t RingSize = 256>
using large_tls_ring = tls_active_item_ring<Key, RingSize>;

} // namespace lru

#endif // LRU_TLS_RING_HPP
