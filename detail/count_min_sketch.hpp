// SPDX-License-Identifier: MIT

#ifndef LRU_DETAIL_COUNT_MIN_SKETCH_HPP
#define LRU_DETAIL_COUNT_MIN_SKETCH_HPP

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include "distributed_mutex.hpp"
#include <stdexcept>
#include <vector>

namespace lru::detail {

// CountMinSketch: probabilistic frequency estimation data structure
// Used by TinyLFU and W-TinyLFU for frequency-aware admission decisions.
// Design inspired by CacheLib's CountMinSketch implementation.
template <typename Key, typename Hash = std::hash<Key>>
class count_min_sketch {
public:
    // width = number of counters per hash row
    // depth = number of hash functions (rows)
    // Default: width ~6*e/epsilon, depth ~ln(1/delta)/ln(2)
    explicit count_min_sketch(std::size_t capacity = 1000,
                              double error_rate = 0.5,
                              double confidence = 0.99)
        : max_window_size_(capacity) {
        auto width = static_cast<std::size_t>(std::ceil(std::exp(1) / error_rate));
        auto depth = static_cast<std::size_t>(std::ceil(std::log(1.0 / (1.0 - confidence)) / std::log(2.0)));
        width = std::max(width, std::size_t(1));
        depth = std::max(depth, std::size_t(1));
        width_ = width;
        depth_ = depth;
        resize_table(depth_ * width_);
        init_hash_seeds();
        recount_saturated();
    }

    // Record an access to the given key. The cell-level CAS loop is lock-free;
    // a shared lock protects the table dimensions and hash seeds from concurrent
    // structural changes (grow/reset/decay/load).
    //
    // Decay trigger: when total_accesses_ reaches max_window_size_, ONE thread
    // atomically claims the decay duty via CAS on total_accesses_ (swapping
    // the value to 0). Other threads see total_accesses_ reset and skip decay
    // entirely — no exclusive lock contention. The decay itself executes
    // under a shared lock (not exclusive), because step_decay only modifies
    // atomic cells within a single row (no structural change).
    void record(const Key& key) noexcept {
        {
            std::shared_lock<detail::distributed_shared_mutex> read_lock(mutex_);
            for (std::size_t i = 0; i < depth_; ++i) {
                auto idx = index_for(key, i);
                auto& cell = table_[idx];
                auto old = cell.load(std::memory_order_relaxed);
                while (old < std::numeric_limits<uint32_t>::max() &&
                       !cell.compare_exchange_weak(old, old + 1,
                           std::memory_order_relaxed, std::memory_order_relaxed)) {
                    // CAS loop: retry on contention
                }
                if (old == std::numeric_limits<uint32_t>::max() - 1u) {
                    saturated_.fetch_add(1, std::memory_order_relaxed);
                }
            }
            // Atomically increment total_accesses_. If the result exceeds
            // max_window_size_, attempt to claim the decay duty via CAS
            // (reset total_accesses_ to 0). Only the winning thread proceeds.
            auto old_total = total_accesses_.fetch_add(1, std::memory_order_relaxed) + 1;
            if (old_total < max_window_size_.load(std::memory_order_relaxed)) {
                return;
            }
            // Try to claim the decay duty: atomically swap total_accesses_ to 0.
            // Only one thread wins; others see total_accesses_ already reset.
            auto expected = old_total;
            if (!total_accesses_.compare_exchange_strong(expected, 0,
                    std::memory_order_acq_rel, std::memory_order_relaxed)) {
                return;  // Another thread already claimed decay
            }
        }
        // We claimed the decay duty. Execute step_decay under a shared lock
        // (NOT exclusive). step_decay only modifies atomic cells in one row
        // and increments decay_step_ — no structural change, so concurrent
        // record() and estimate() can proceed.
        step_decay_shared();
    }

    // Estimate the frequency of a key. Uses an optimistic lock-free path:
    // read structure_version_ before and after accessing the table; if it
    // changed (or was odd, meaning a structural change was in progress),
    // fall back to the shared-lock path.
    uint32_t estimate(const Key& key) const noexcept {
        // Optimistic lock-free path: check structure_version before/after
        auto v1 = structure_version_.load(std::memory_order_acquire);
        if ((v1 & 1u) == 0u) {  // Even = no structural change in progress
            if (depth_ == 0) return 0;
            auto w = width_;
            auto d = depth_;
            uint32_t min_val = std::numeric_limits<uint32_t>::max();
            for (std::size_t i = 0; i < d; ++i) {
                auto idx = index_for(key, i, w);
                min_val = std::min(min_val, table_[idx].load(std::memory_order_relaxed));
            }
            auto v2 = structure_version_.load(std::memory_order_acquire);
            if (v1 == v2) return min_val;  // No structural change occurred
        }
        // Fallback: use shared lock
        std::shared_lock<detail::distributed_shared_mutex> read_lock(mutex_);
        if (depth_ == 0) return 0;
        uint32_t min_val = std::numeric_limits<uint32_t>::max();
        for (std::size_t i = 0; i < depth_; ++i) {
            auto idx = index_for(key, i);
            min_val = std::min(min_val, table_[idx].load(std::memory_order_relaxed));
        }
        return min_val;
    }

    // Decay all counters by factor (e.g., 0.5 = halve)
    void decay(double factor = 0.5) noexcept {
        std::unique_lock<detail::distributed_shared_mutex> write_lock(mutex_);
        structure_change_guard guard(structure_version_);
        for (std::size_t i = 0; i < table_size_; ++i) {
            table_[i].store(
                static_cast<uint32_t>(table_[i].load(std::memory_order_relaxed) * factor),
                std::memory_order_relaxed);
        }
        total_accesses_.store(0, std::memory_order_relaxed);
        decay_step_.store(0, std::memory_order_relaxed);
        recount_saturated();
    }

    // Step decay: instead of halving all counters at once, only halve one
    // row per event. This spreads the work across multiple decay events.
    void step_decay() noexcept {
        std::unique_lock<detail::distributed_shared_mutex> write_lock(mutex_);
        step_decay_locked();
    }

    // Dynamically grow the counter table when cache size increases
    // significantly. Called when total_accesses_ exceeds 2x max_window_size.
    // Doubles the width and re-seeds hash functions.
    void maybe_grow_access_counters() {
        std::unique_lock<detail::distributed_shared_mutex> write_lock(mutex_);
        if (total_accesses_.load(std::memory_order_relaxed) <= 2 * max_window_size_.load(std::memory_order_relaxed)) {
            return;
        }
        grow(width_ * 2);
    }

    // Resize the table to a new width. This is an approximation of
    // key-preserving growth because the original keys are not stored. We
    // redistribute the old counters by re-hashing (row, old_column) pairs,
    // which preserves aggregate counts but not per-key locations.
    void grow(std::size_t new_capacity) {
        std::unique_lock<detail::distributed_shared_mutex> write_lock(mutex_);
        structure_change_guard guard(structure_version_);
        if (new_capacity <= width_) {
            return;
        }
        // Copy old table values (atomic -> plain)
        std::vector<uint32_t> old_table;
        old_table.reserve(table_size_);
        for (std::size_t i = 0; i < table_size_; ++i) {
            old_table.push_back(table_[i].load(std::memory_order_relaxed));
        }
        auto old_width = width_;

        width_ = new_capacity;
        resize_table(depth_ * width_);

        // Redistribute old counters into the new, wider table.
        for (std::size_t i = 0; i < depth_; ++i) {
            for (std::size_t j = 0; j < old_width; ++j) {
                uint32_t val = old_table[i * old_width + j];
                if (val > 0) {
                    // Approximate re-hash of the old (row, column) pair.
                    std::size_t h = hash_seeds_[i];
                    h ^= static_cast<std::size_t>(j + 0x9e3779b9 + (h << 6) + (h >> 2));
                    std::size_t new_col = h % width_;
                    std::size_t new_idx = i * width_ + new_col;
                    auto& cell = table_[new_idx];
                    auto cur = cell.load(std::memory_order_relaxed);
                    uint32_t next;
                    do {
                        next = cur + val;
                        if (next < cur) {
                            // Overflow: saturate
                            next = std::numeric_limits<uint32_t>::max();
                        }
                    } while (!cell.compare_exchange_weak(cur, next,
                                 std::memory_order_relaxed, std::memory_order_relaxed));
                }
            }
        }

        // Re-initialize hash seeds for better independence with new width
        init_hash_seeds();
        total_accesses_.store(0, std::memory_order_relaxed);
        decay_step_.store(0, std::memory_order_relaxed);
        recount_saturated();
    }

    void reset() noexcept {
        std::unique_lock<detail::distributed_shared_mutex> write_lock(mutex_);
        structure_change_guard guard(structure_version_);
        for (std::size_t i = 0; i < table_size_; ++i) {
            table_[i].store(0, std::memory_order_relaxed);
        }
        total_accesses_.store(0, std::memory_order_relaxed);
        decay_step_.store(0, std::memory_order_relaxed);
        recount_saturated();
    }

    std::size_t total_accesses() const noexcept { return total_accesses_.load(std::memory_order_relaxed); }
    std::size_t width() const noexcept { return width_; }
    std::size_t depth() const noexcept { return depth_; }

    /// Number of cells currently saturated at UINT32_MAX. Used to detect
    /// sketch distortion and guide decay/rebuild decisions.
    std::uint64_t get_saturated_count() const noexcept { return saturated_.load(std::memory_order_relaxed); }

    void set_max_window_size(std::size_t size) noexcept {
        max_window_size_.store(size, std::memory_order_relaxed);
    }
    std::size_t max_window_size() const noexcept { return max_window_size_.load(std::memory_order_relaxed); }

    // ====================================================================
    // S3: Serialization support — save/restore CMS internal state
    // ====================================================================

    /// Serialize CMS internal state to an output iterator.
    /// Format: width + depth + total_accesses + decay_step + saturated +
    ///         table (depth*width uint32_t values).
    /// Throws std::overflow_error if the table dimensions do not fit in
    /// uint32_t, preventing silent truncation.
    template <typename OutputIt>
    void save_state(OutputIt out) const {
        std::unique_lock<detail::distributed_shared_mutex> write_lock(mutex_);
        structure_change_guard guard(structure_version_);
        constexpr auto u32max = std::numeric_limits<uint32_t>::max();
        if (width_ > u32max || depth_ > u32max ||
            serialized_state_words() > u32max) {
            throw std::overflow_error(
                "count_min_sketch: dimensions exceed uint32_t serialization limit");
        }
        *out++ = static_cast<uint32_t>(width_);
        *out++ = static_cast<uint32_t>(depth_);
        *out++ = static_cast<uint32_t>(total_accesses_.load(std::memory_order_relaxed));
        *out++ = static_cast<uint32_t>(decay_step_.load(std::memory_order_relaxed));
        *out++ = static_cast<uint32_t>(saturated_.load(std::memory_order_relaxed));
        for (std::size_t i = 0; i < table_size_; ++i) {
            *out++ = table_[i].load(std::memory_order_relaxed);
        }
    }

    /// Restore CMS internal state from an input iterator.
    /// Returns the number of uint32_t values consumed.
    /// Throws std::runtime_error if the saved dimensions are malformed.
    template <typename InputIt>
    std::size_t load_state(InputIt& it) {
        std::unique_lock<detail::distributed_shared_mutex> write_lock(mutex_);
        structure_change_guard guard(structure_version_);
        auto new_width = static_cast<std::size_t>(*it++);
        auto new_depth = static_cast<std::size_t>(*it++);
        auto new_total_accesses = static_cast<std::size_t>(*it++);
        auto new_decay_step = static_cast<std::size_t>(*it++);
        auto new_saturated = static_cast<std::uint64_t>(*it++);

        // Guard against malformed / malicious state before allocating a huge
        // table. The limits are generous for any realistic CMS usage.
        static constexpr std::size_t k_max_width = 1'000'000;
        static constexpr std::size_t k_max_depth = 64;
        if (new_width == 0 || new_depth == 0 ||
            new_width > k_max_width || new_depth > k_max_depth) {
            throw std::runtime_error(
                "count_min_sketch: malformed dimensions in load_state");
        }

        width_ = new_width;
        depth_ = new_depth;
        total_accesses_.store(new_total_accesses, std::memory_order_relaxed);
        decay_step_.store(new_decay_step, std::memory_order_relaxed);
        saturated_.store(new_saturated, std::memory_order_relaxed);

        resize_table(depth_ * width_);
        for (std::size_t i = 0; i < table_size_; ++i) {
            table_[i].store(*it++, std::memory_order_relaxed);
        }
        init_hash_seeds();
        recount_saturated();
        // 返回消耗的 uint32_t 数：5(header) + depth_ * width_(table)
        return 5 + depth_ * width_;
    }

    /// Return the number of uint32_t values required for serialization.
    std::size_t serialized_state_words() const noexcept {
        return 5 + table_size_;
    }

private:
    // Initialize independent hash seeds for better hash independence
    void init_hash_seeds() {
        // Golden ratio derived constants for independent hashing
        static constexpr std::size_t seeds[] = {
            0xc3a7c8e5ULL,
            0x5bd1e995ULL,
            0x1b873593ULL,
            0x27d4eb2fULL,
        };
        hash_seeds_.clear();
        for (std::size_t i = 0; i < depth_; ++i) {
            hash_seeds_.push_back(seeds[i % 4] ^ (i * 0x9e3779b9ULL));
        }
    }

    // Allocate and zero-fill the atomic table.
    // Uses unique_ptr<atomic[]> because vector<atomic> cannot resize
    // (atomic is neither copyable nor movable).
    void resize_table(std::size_t new_size) {
        table_.reset(new std::atomic<uint32_t>[new_size]);
        table_size_ = new_size;
        // Zero-fill: new[] value-initializes trivial types, but atomic
        // default construction may leave values indeterminate on some
        // implementations, so store explicitly.
        for (std::size_t i = 0; i < new_size; ++i) {
            table_[i].store(0, std::memory_order_relaxed);
        }
    }

    // Recompute saturated_ from the table. Called after any operation that
    // may decrease counters (decay, reset, grow) to keep the accounting exact.
    void recount_saturated() noexcept {
        std::uint64_t count = 0;
        for (std::size_t i = 0; i < table_size_; ++i) {
            if (table_[i].load(std::memory_order_relaxed) == std::numeric_limits<uint32_t>::max()) {
                ++count;
            }
        }
        saturated_.store(count, std::memory_order_relaxed);
    }

    // Internal step_decay assuming mutex_ is already held exclusively.
    // Called from decay(), step_decay(), and grow() which already hold
    // the exclusive lock and use structure_change_guard.
    void step_decay_locked() noexcept {
        structure_change_guard guard(structure_version_);
        // Each step processes one row (depth partitions)
        std::size_t step = decay_step_.load(std::memory_order_relaxed);
        std::size_t row = step % depth_;
        std::size_t base = row * width_;
        for (std::size_t j = 0; j < width_; ++j) {
            auto& cell = table_[base + j];
            cell.store(cell.load(std::memory_order_relaxed) >> 1,
                       std::memory_order_relaxed);
        }
        decay_step_.store(step + 1, std::memory_order_relaxed);
        // Reset total_accesses after each decay step so that the next
        // cycle of record() calls can proceed lock-free until the threshold
        // is hit again.
        total_accesses_.store(0, std::memory_order_relaxed);
        recount_saturated();
    }

    // Step decay under a shared lock (NOT exclusive).
    // Used by record() when it claims the decay duty. Since step_decay only
    // modifies atomic cells in one row (no structural change to width_ or
    // depth_), a shared lock is sufficient — concurrent record() calls can
    // proceed, and estimate() readers are safe because cells are atomic.
    //
    // No structure_change_guard is needed here because the table dimensions
    // and hash seeds don't change. The row-halving only touches atomic
    // cells, which estimate() already reads atomically.
    void step_decay_shared() noexcept {
        std::shared_lock<detail::distributed_shared_mutex> read_lock(mutex_);
        // Each step processes one row (depth partitions)
        std::size_t step = decay_step_.fetch_add(1, std::memory_order_relaxed);
        std::size_t row = step % depth_;
        std::size_t base = row * width_;
        for (std::size_t j = 0; j < width_; ++j) {
            auto& cell = table_[base + j];
            cell.store(cell.load(std::memory_order_relaxed) >> 1,
                       std::memory_order_relaxed);
        }
        // total_accesses_ was already reset to 0 by the caller (record())
        // before entering this method, so no need to reset again.
        recount_saturated();
    }

    // Stronger per-row hash mix: combine the key hash with the row seed and
    // row index, then apply splitmix64 to reduce correlation between rows.
    std::size_t index_for(const Key& key, std::size_t hash_num) const noexcept {
        auto h = Hash{}(key);
        std::size_t x = h + hash_seeds_[hash_num] + hash_num;

        // splitmix64
        x += 0x9e3779b97f4a7c15ULL;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
        x = x ^ (x >> 31);

        return hash_num * width_ + (x % width_);
    }

    // Overload that accepts an explicit width parameter for lock-free reads
    // where width_ might change concurrently.
    std::size_t index_for(const Key& key, std::size_t hash_num, std::size_t w) const noexcept {
        auto h = Hash{}(key);
        std::size_t x = h + hash_seeds_[hash_num] + hash_num;

        // splitmix64
        x += 0x9e3779b97f4a7c15ULL;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
        x = x ^ (x >> 31);

        return hash_num * w + (x % w);
    }

    // Table storage: unique_ptr<atomic[]> avoids vector<atomic> which
    // cannot resize (atomic is neither copyable nor movable).
    std::unique_ptr<std::atomic<uint32_t>[]> table_;
    std::size_t table_size_ = 0;
    std::vector<std::size_t> hash_seeds_;
    std::size_t width_ = 0;
    std::size_t depth_ = 0;
    alignas(64) std::atomic<std::size_t> total_accesses_{0};
    std::atomic<std::size_t> max_window_size_{1000};
    std::atomic<std::size_t> decay_step_{0};
    // Number of cells that have reached UINT32_MAX.
    std::atomic<std::uint64_t> saturated_{0};
    // Version counter for optimistic lock-free reads in estimate().
    // Incremented before and after structural changes; odd values indicate
    // a structural change is in progress.
    mutable std::atomic<uint64_t> structure_version_{0};

    // RAII guard that increments structure_version_ on entry and exit,
    // marking the region as a structural change.  Must be used after
    // acquiring the exclusive lock.
    struct structure_change_guard {
        std::atomic<uint64_t>& version;
        explicit structure_change_guard(std::atomic<uint64_t>& v) : version(v) {
            version.fetch_add(1, std::memory_order_release);  // "write lock begin"
        }
        ~structure_change_guard() {
            version.fetch_add(1, std::memory_order_release);  // "write lock end"
        }
    };

    // Mutex for operations that modify table structure (decay, grow, reset,
    // save/load_state) and step_decay serialization in record().
    // All table-modifying cold-path operations share this mutex so they are
    // mutually exclusive with each other and with the step_decay path in
    // record(). The hot-path CAS loop in record() uses a shared lock so it
    // remains concurrent with other readers.
    mutable detail::distributed_shared_mutex mutex_;
};

} // namespace lru::detail

#endif
