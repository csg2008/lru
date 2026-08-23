// SPDX-License-Identifier: MIT
// Space-Saving Top-K — Streaming heavy-hitter detection
//
// Implements the Space-Saving algorithm (Metwally et al., 2005) for
// approximate tracking of the most frequently observed items in a stream.
// This replaces the O(N) batch scan in event_tracker::top_keys() with
// an O(1) amortized update and O(K log K) query.
//
// Properties:
//   - Fixed capacity K (default 1024)
//   - O(1) update for existing keys (hash map lookup + increment)
//   - O(log K) amortized update when replacing the min entry
//     (T-P2-4 / R-8: was O(K) via std::min_element scan; now uses a
//     min-heap with lazy deletion to find the true minimum in O(log K))
//   - O(K log K) top-k query (partial_sort of K entries)
//   - Guaranteed error bound:
//       estimated_count >= true_count >= estimated_count - error
//     where error <= total_observations / K (for items in the summary)
//   - Recall = 100% for items whose true frequency exceeds total / K
//     (i.e., items that are guaranteed to be in the summary)
//
// NOT thread-safe. Caller must synchronize (e.g., via a mutex).
//
// Reference:
//   Metwally, A., Agrawal, D., El Abbadi, A. (2005).
//   "Efficient Computation of Frequent and Top-k Elements in Data Streams."
//   ICDT 2005.

#ifndef LRU_SPACE_SAVING_HPP
#define LRU_SPACE_SAVING_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <queue>
#include <utility>
#include <vector>

#include "ankerl/unordered_dense.h"

namespace lru::detail {

/// Streaming top-K heavy-hitter tracker using the Space-Saving algorithm.
///
/// @tparam Key  The key type (defaults to uint64_t for key hashes).
template <typename Key = uint64_t>
class space_saving_top_k {
public:
    /// Construct a tracker with the given capacity.
    /// @param capacity  Maximum number of distinct keys tracked (default 1024).
    ///                  Larger capacity → better accuracy, more memory.
    explicit space_saving_top_k(std::size_t capacity = 1024)
        : capacity_(capacity) {
        if (capacity_ == 0) capacity_ = 1;
        map_.reserve(capacity_ * 2);
    }

    /// Observe a key occurrence (count = 1).
    void add(const Key& key) { add(key, 1); }

    /// Observe a key with a given count increment.
    ///
    /// Complexity:
    ///   - O(1) if the key is already tracked (map lookup + increment; the
    ///     heap is NOT updated on this path — stale heap entries are
    ///     filtered lazily during the eviction path).
    ///   - O(log K) amortized if a new key must evict the current minimum.
    ///     The min-heap top is popped until a non-stale entry is found
    ///     (count in heap == count in map). Each stale entry encountered
    ///     is refreshed by pushing the current count back onto the heap,
    ///     ensuring the key retains a heap presence. Amortized O(log K)
    ///     because each (key, count) pair is pushed at most twice: once
    ///     on initial insert, once on refresh.
    void add(const Key& key, std::size_t count) {
        total_ += count;
        auto it = map_.find(key);
        if (it != map_.end()) {
            // Existing key: increment count in map. The heap entry for
            // this key is now stale (count in heap < count in map). It
            // will be refreshed lazily when we next need the minimum.
            it->second += count;
            return;
        }
        if (map_.size() < capacity_) {
            map_.emplace(key, count);
            heap_.push(Entry{count, key});
            return;
        }
        // Capacity full: find the true minimum via lazy deletion.
        // T-P2-4 (R-8): replaced O(K) std::min_element scan with O(log K)
        // amortized min-heap pop. Stale entries (where the key's count in
        // map_ has increased since it was pushed) are filtered out by
        // comparing heap count to map count. Each stale entry is refreshed
        // by pushing the current count, so the key always has a heap
        // presence for future eviction queries.
        std::size_t min_count = 0;
        Key min_key{};
        bool found = false;
        while (!heap_.empty()) {
            Entry top = heap_.top();
            const std::size_t& top_count = top.first;
            const Key& top_key = top.second;
            auto map_it = map_.find(top_key);
            if (map_it == map_.end()) {
                // Key was evicted by a previous add() call. Discard the
                // stale heap entry.
                heap_.pop();
                continue;
            }
            if (map_it->second != top_count) {
                // Stale count: the key's count has increased since this
                // heap entry was pushed. Refresh by pushing the current
                // count and discarding the stale entry. This ensures the
                // key retains a heap presence with its correct count.
                heap_.pop();
                heap_.push(Entry{map_it->second, top_key});
                continue;
            }
            // Valid minimum found — heap count matches map count.
            min_count = top_count;
            min_key = top_key;
            found = true;
            break;
        }

        if (!found) {
            // Should not happen if capacity > 0 and map is non-empty,
            // but defensive: if heap was somehow drained without finding
            // a valid min, fall back to scanning the map.
            auto min_it = std::min_element(
                map_.begin(), map_.end(),
                [](const auto& a, const auto& b) { return a.second < b.second; });
            min_count = min_it->second;
            min_key = min_it->first;
        } else {
            heap_.pop();  // Remove the valid min entry we found
        }

        // Replace the min key with the new key, inheriting min_count
        // (per Space-Saving algorithm: new key's estimate is at least
        // min_count, the over-estimate from the evicted key).
        std::size_t new_count = min_count + count;
        map_.erase(min_key);
        map_.emplace(key, new_count);
        heap_.push(Entry{new_count, key});

        // T-P2-4 (R-8): Bound heap growth. Each add() that triggers
        // eviction pushes at most 2 entries (1 refresh + 1 new). Over
        // many operations, the heap could grow to O(updates) if we
        // never trimmed. When heap size exceeds 4x capacity, rebuild
        // from the map to remove all stale entries. This is O(K log K)
        // but runs at most once every O(K) evictions, so amortized
        // cost is O(log K) per add().
        if (heap_.size() > capacity_ * 4) {
            rebuild_heap();
        }
    }

    /// Return the top-k entries sorted by count (descending).
    /// O(K log K) where K = min(k, capacity).
    std::vector<std::pair<Key, std::size_t>> top_k(std::size_t k) const {
        std::vector<std::pair<Key, std::size_t>> result(map_.begin(), map_.end());
        std::size_t limit = std::min(k, result.size());
        std::partial_sort(result.begin(), result.begin() + limit,
                          result.end(),
                          [](const auto& a, const auto& b) {
                              return a.second > b.second;
                          });
        if (result.size() > k) result.resize(k);
        return result;
    }

    /// Total number of observations (sum of all add() counts).
    /// Used for the error bound: error <= total() / capacity().
    std::size_t total() const noexcept { return total_; }

    /// Maximum number of distinct keys tracked.
    std::size_t capacity() const noexcept { return capacity_; }

    /// Current number of tracked keys (<= capacity).
    std::size_t size() const noexcept { return map_.size(); }

    /// Reset the structure to empty state.
    void reset() {
        map_.clear();
        // Clearing a priority_queue is done by assignment from an empty one.
        heap_ = MinHeap{};
        total_ = 0;
    }

    /// Estimated error bound for counts of items in the summary.
    /// The true count of any tracked item is >= estimated_count - error_bound().
    std::size_t error_bound() const noexcept {
        return capacity_ > 0 ? total_ / capacity_ : 0;
    }

private:
    using Entry = std::pair<std::size_t, Key>;  // (count, key)

    // Min-heap comparator: smaller count = higher priority (top).
    // std::priority_queue is a max-heap by default, so we use greater<>
    // to make it a min-heap on the first element (count).
    struct EntryGreater {
        bool operator()(const Entry& a, const Entry& b) const noexcept {
            return a.first > b.first;  // smaller count → higher priority
        }
    };
    using MinHeap = std::priority_queue<Entry, std::vector<Entry>, EntryGreater>;

    /// Rebuild the heap from the map, discarding all stale entries.
    /// O(K log K) but runs infrequently (only when heap_ exceeds 4x capacity).
    void rebuild_heap() {
        MinHeap fresh;
        for (const auto& [k, c] : map_) {
            fresh.push(Entry{c, k});
        }
        heap_ = std::move(fresh);
    }

    std::size_t capacity_;
    std::size_t total_{0};
    ankerl::unordered_dense::map<Key, std::size_t> map_;
    MinHeap heap_;
};

}  // namespace lru::detail

#endif  // LRU_SPACE_SAVING_HPP
