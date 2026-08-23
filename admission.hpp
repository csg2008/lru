// Unified LRU Cache Library — Admission Policy (RejectFirst + pluggable filters)
// SPDX-License-Identifier: MIT
// Inspired by Facebook CacheLib's NvmAdmissionPolicy and RejectFirstAP
//
// Admission policies filter items before they enter the cache, rejecting
// "one-hit-wonder" items that would waste cache space. This is especially
// effective against:
//   - Full-table scans that flood the cache with cold data
//   - Bulk-load operations that bypass normal access patterns
//   - Workloads with heavy-tailed key distributions
//
// RejectFirstAP: only admits items whose key has been seen at least twice.
// This alone can improve hit rates by 20–40% in scan-heavy workloads.

#ifndef LRU_ADMISSION_HPP
#define LRU_ADMISSION_HPP

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <type_traits>
#include <vector>

#include "ankerl/unordered_dense.h"

namespace lru {

// ============================================================================
// Approximate Split Set (lightweight "seen-before" tracker)
// ============================================================================

/// A space-efficient probabilistic set for tracking "have we seen this key before?".
///
/// Uses per-split hash sets (unordered_dense) for O(1) lookup per key,
/// with bounded per-split capacity and random eviction within each split.
/// Not a Bloom filter — it stores exact hashes, but may have false positives
/// due to hash collisions (configurable via num_splits).
///
/// Memory usage: ~24 bytes per tracked entry (hash set node overhead).
/// E.g., tracking 1M keys = ~24 MB.
/// NOTE: This class is NOT thread-safe. All access must be externally synchronized.
class approx_split_set {
public:
    /// @param max_entries  Maximum number of unique keys to track.
    /// @param num_splits   Number of hash partitions (more splits = lower false positive rate).
    ///                     Default 4 is good for most use cases.
    approx_split_set(std::size_t max_entries, uint32_t num_splits = 4)
        : num_splits_(num_splits)
        , max_per_split_(std::max(std::size_t(1), max_entries / num_splits)) {
        splits_.resize(num_splits_);
        for (auto& split : splits_) {
            split.entries.reserve(max_per_split_);
        }
    }

    /// Check if key_hash was seen before, and record it for future checks.
    /// @return true if the key was already in the set (seen before).
    /// @return false if this is the first time seeing the key.
    bool insert_and_check(uint64_t key_hash) {
        auto split_idx = key_hash % num_splits_;
        auto& split = splits_[split_idx];

        // O(1) lookup via hash set
        auto [it, inserted] = split.entries.insert(key_hash);
        if (!inserted) {
            return true; // seen before
        }

        // New insertion — maintain capacity
        if (split.entries.size() > max_per_split_) {
            // Arbitrary eviction: begin() on unordered_dense::set has no defined
            // ordering, so the evicted key is non-deterministic. This is
            // intentional — we only need to bound the window, not pick a
            // specific victim.
            split.entries.erase(split.entries.begin());
        }
        ++total_inserts_;
        return false; // first time seeing this key
    }

    /// Insert without checking (for pre-seeding).
    void insert(uint64_t key_hash) {
        (void)insert_and_check(key_hash);
    }

    /// Check if present without inserting.
    bool contains(uint64_t key_hash) const {
        auto split_idx = key_hash % num_splits_;
        const auto& split = splits_[split_idx];
        return split.entries.find(key_hash) != split.entries.end();
    }

    /// Remove a key hash from the set.
    void remove(uint64_t key_hash) {
        auto split_idx = key_hash % num_splits_;
        auto& split = splits_[split_idx];
        split.entries.erase(key_hash);
    }

    /// Total number of keys currently tracked.
    std::size_t size() const noexcept {
        std::size_t total = 0;
        for (const auto& split : splits_) {
            total += split.entries.size();
        }
        return total;
    }

    /// Total number of unique insertions since creation.
    std::size_t total_inserts() const noexcept {
        return total_inserts_;
    }

    /// Clear all tracked keys.
    void clear() {
        for (auto& split : splits_) {
            split.entries.clear();
        }
        total_inserts_ = 0;
    }

    /// Maximum capacity.
    std::size_t max_entries() const noexcept {
        return max_per_split_ * num_splits_;
    }

private:
    struct split {
        ankerl::unordered_dense::set<uint64_t> entries;
    };

    uint32_t num_splits_;
    std::size_t max_per_split_;
    std::vector<split> splits_;
    std::size_t total_inserts_ = 0;
};

// ============================================================================
// Abstract Admission Policy
// ============================================================================

/// Base class for admission policies.
/// Subclass and override should_admit() to implement custom logic.
template <typename Key>
class admission_policy {
public:
    using key_type = Key;

    virtual ~admission_policy() = default;

    /// Decide whether a key should be admitted to the cache.
    /// @param key  The key being considered for cache entry.
    /// @return     true = admit, false = reject (skip cache insertion).
    virtual bool should_admit(const key_type& key) = 0;

    /// Called when an item is accessed (hit). Override for policies that
    /// need visibility into access patterns.
    virtual void on_access(const key_type& /*key*/) {}

    /// Called when an item is evicted. Override to clean up per-key state.
    virtual void on_evict(const key_type& /*key*/) {}

    /// Called when an item is explicitly removed (del/erase), as opposed to
    /// on_evict() which fires on LRU eviction. Some policies treat removal
    /// differently from eviction — e.g., a removed key may still be tracked
    /// for future admission decisions, while an evicted key should be
    /// untracked. By default, falls through to on_evict() for backward
    /// compatibility.
    virtual void on_remove(const key_type& key) { on_evict(key); }

    /// Called when an item is inserted.
    virtual void on_insert(const key_type& /*key*/) {}

    /// Get a human-readable name for this policy.
    virtual std::string name() const = 0;

    /// Reset internal state.
    virtual void reset() = 0;
};

// ============================================================================
// RejectFirst Admission Policy
// ============================================================================

/// Only admits a key if it has been seen at least once before.
/// This is the classic "reject first access" pattern, extremely effective
/// against scan workloads and one-hit-wonder keys.
///
/// Uses approx_split_set internally to track recently-seen key hashes.
///
/// Usage:
///   RejectFirstAP<std::string> ap(1'000'000);  // track 1M keys
///   if (ap.should_admit("user:42")) {
///       cache.set("user:42", data);
///   }
template <typename Key, typename Hash = std::hash<Key>>
class reject_first_ap : public admission_policy<Key> {
public:
    using key_type = Key;
    using base_type = admission_policy<Key>;

    /// @param max_entries     Number of unique keys to track.
    /// @param num_splits      Hash partitions for the internal split set.
    /// @param use_hit_signal  If true, admit on first access IF the item had a hit
    ///                        in DRAM (item was accessed more than once). Requires
    ///                        on_access() to be called on hits.
    reject_first_ap(std::size_t max_entries,
                    uint32_t num_splits = 4,
                    bool use_hit_signal = false)
        : tracker_(max_entries, num_splits)
        , use_hit_signal_(use_hit_signal) {}

    /// Decide admission.
    /// Without hit_signal: admits only if seen before.
    /// With hit_signal: admits if seen before OR the key had a hit signal set.
    bool should_admit(const key_type& key) override {
        auto key_hash = hash_key(key);

        bool seen_before = tracker_.insert_and_check(key_hash);
        if (seen_before) {
            ++admitted_seen_;
            return true;
        }

        // Check if DRAM hit signal allows bypass
        if (use_hit_signal_) {
            std::lock_guard lock(hit_mutex_);
            auto it = hit_signals_.find(key_hash);
            if (it != hit_signals_.end() && it->second) {
                hit_signals_.erase(it);
                ++admitted_by_dram_hit_;
                return true;
            }
        }

        ++rejected_;
        return false;
    }

    /// Record that a key was accessed (hit). Sets the hit signal for next admission check.
    void on_access(const key_type& key) override {
        if (use_hit_signal_) {
            auto key_hash = hash_key(key);
            std::lock_guard lock(hit_mutex_);
            hit_signals_[key_hash] = true;
        }
    }

    /// Clean up tracking state for evicted keys.
    void on_evict(const key_type& key) override {
        tracker_.remove(hash_key(key));
    }

    /// Explicit removal: same as eviction for RejectFirst — untrack the key
    /// so it can be re-evaluated on next access.
    void on_remove(const key_type& key) override {
        tracker_.remove(hash_key(key));
    }

    void on_insert(const key_type& /*key*/) override {
        ++total_admitted_;
    }

    /// Statistics
    struct stats {
        std::size_t admitted_seen = 0;        // Admitted because seen before
        std::size_t admitted_by_dram_hit = 0; // Admitted via DRAM hit signal
        std::size_t rejected = 0;             // Rejected (first-time access)
        std::size_t total_admitted = 0;       // Total admitted to cache
        std::size_t keys_tracked = 0;         // Currently tracked unique keys
        double admit_rate() const {
            auto total = admitted_seen + admitted_by_dram_hit + rejected;
            return total > 0
                ? static_cast<double>(admitted_seen + admitted_by_dram_hit) / static_cast<double>(total)
                : 0.0;
        }
    };

    stats get_stats() const {
        stats s;
        s.admitted_seen = admitted_seen_.load(std::memory_order_relaxed);
        s.admitted_by_dram_hit = admitted_by_dram_hit_.load(std::memory_order_relaxed);
        s.rejected = rejected_.load(std::memory_order_relaxed);
        s.total_admitted = total_admitted_.load(std::memory_order_relaxed);
        s.keys_tracked = tracker_.size();
        return s;
    }

    std::string name() const override {
        return "reject_first_ap";
    }

    void reset() override {
        std::lock_guard lock(hit_mutex_);
        tracker_.clear();
        hit_signals_.clear();
        admitted_seen_.store(0);
        admitted_by_dram_hit_.store(0);
        rejected_.store(0);
        total_admitted_.store(0);
    }

    /// Get access to the underlying tracker (for advanced usage).
    approx_split_set& tracker() noexcept { return tracker_; }
    const approx_split_set& tracker() const noexcept { return tracker_; }

private:
    std::size_t hash_key(const key_type& key) const {
        if constexpr (std::is_integral_v<key_type>) {
            return static_cast<std::size_t>(key);
        } else {
            return Hash{}(key);
        }
    }

    approx_split_set tracker_;
    bool use_hit_signal_;

    // Per-hash hit signal: set by on_access(), consumed by should_admit()
    ankerl::unordered_dense::map<std::size_t, bool> hit_signals_;
    mutable std::mutex hit_mutex_;

    std::atomic<std::size_t> admitted_seen_{0};
    std::atomic<std::size_t> admitted_by_dram_hit_{0};
    std::atomic<std::size_t> rejected_{0};
    std::atomic<std::size_t> total_admitted_{0};
};

// ============================================================================
// Always-Admit Policy (no-op / default)
// ============================================================================

/// A pass-through admission policy that admits everything.
/// Use this when you don't need admission filtering.
template <typename Key>
class always_admit_ap : public admission_policy<Key> {
public:
    bool should_admit(const Key&) override { return true; }
    std::string name() const override { return "always_admit"; }
    void reset() override {}
};

// ============================================================================
// Admission Filter Cache (Decorator)
// ============================================================================

/// Wraps any cache type with an admission policy filter.
/// All insertions go through should_admit() first; rejected items are skipped.
///
/// Usage:
///   using filtered_cache = admission_filter_cache<
///       lru::cache<int, std::string>,
///       reject_first_ap<int>
///   >;
///
///   auto ap = std::make_unique<reject_first_ap<int>>(1'000'000);
///   filtered_cache fc(std::move(ap), std::move(my_cache));
///   fc.set(42, "hello");  // may be rejected by the admission policy
template <typename CacheType, typename AdmissionPolicy>
class admission_filter_cache {
public:
    using cache_type = CacheType;
    using policy_type = AdmissionPolicy;
    using key_type = typename cache_type::key_type;
    using mapped_type = typename cache_type::mapped_type;
    using size_type = typename cache_type::size_type;

    /// Construct with an admission policy and an existing cache.
    admission_filter_cache(std::unique_ptr<policy_type> policy, cache_type cache)
        : policy_(std::move(policy))
        , cache_(std::move(cache)) {}

    /// Construct with policy only (cache is default-constructed).
    explicit admission_filter_cache(std::unique_ptr<policy_type> policy)
        : policy_(std::move(policy)) {}

    /// Construct with policy and cache capacity.
    admission_filter_cache(std::unique_ptr<policy_type> policy, size_type max_size)
        : policy_(std::move(policy))
        , cache_(max_size) {}

    // --------------------------------------------------------------------
    // Filtered API
    // --------------------------------------------------------------------

    template <typename V>
    bool set(const key_type& key, V&& value) {
        {
            std::lock_guard<std::mutex> lock(policy_mutex_);
            if (!policy_->should_admit(key)) {
                // O7: Fire on_reject so consumers can monitor admission
                // rejections (e.g. reject_first_ap filtering first-access
                // keys). Value is forwarded by const reference here — we
                // must NOT move it, since the caller still owns it.
                cache_.callbacks().collect_reject(key, value);
                return false; // rejected
            }
        }
        cache_.set(key, std::forward<V>(value));
        {
            std::lock_guard<std::mutex> lock(policy_mutex_);
            policy_->on_insert(key);
        }
        return true;
    }

    template <typename V>
    bool add(const key_type& key, V&& value) {
        {
            std::lock_guard<std::mutex> lock(policy_mutex_);
            if (!policy_->should_admit(key)) {
                // O7: Fire on_reject for admission rejection.
                cache_.callbacks().collect_reject(key, value);
                return false;
            }
        }
        bool added = cache_.add(key, std::forward<V>(value));
        {
            std::lock_guard<std::mutex> lock(policy_mutex_);
            if (added) {
                policy_->on_insert(key);
            } else {
                policy_->on_access(key);
            }
        }
        return added;
    }

    auto get(const key_type& key) -> decltype(std::declval<cache_type&>().get(key)) {
        auto result = cache_.get(key);
        if (static_cast<bool>(result)) {
            std::lock_guard<std::mutex> lock(policy_mutex_);
            policy_->on_access(key);
            ++hits_;
        } else {
            ++misses_;
        }
        return result;
    }

    auto get(const key_type& key) const -> decltype(std::declval<const cache_type&>().get(key)) {
        auto result = cache_.get(key);
        if (static_cast<bool>(result)) {
            std::lock_guard<std::mutex> lock(policy_mutex_);
            policy_->on_access(key);
            ++hits_;
        } else {
            ++misses_;
        }
        return result;
    }

    bool del(const key_type& key) {
        bool result = cache_.del(key);
        if (result) {
            std::lock_guard<std::mutex> lock(policy_mutex_);
            policy_->on_remove(key);
        }
        return result;
    }

    bool contains(const key_type& key) const {
        return cache_.contains(key);
    }

    auto peek(const key_type& key) const {
        return cache_.peek(key);
    }

    // --------------------------------------------------------------------
    // Passthrough API
    // --------------------------------------------------------------------

    bool empty() const { return cache_.empty(); }
    size_type size() const { return cache_.size(); }
    size_type max_size() const { return cache_.max_size(); }
    void max_size(size_type n) { cache_.max_size(n); }
    void flush() {
        cache_.flush();
        std::lock_guard<std::mutex> lock(policy_mutex_);
        policy_->reset();
    }

    auto& cache() noexcept { return cache_; }
    const auto& cache() const noexcept { return cache_; }
    auto& policy() noexcept { return *policy_; }
    const auto& policy() const noexcept { return *policy_; }

    /// Per-tier stats.
    double hit_rate() const {
        auto total = hits_.load() + misses_.load();
        return total > 0
            ? static_cast<double>(hits_.load()) / static_cast<double>(total)
            : 0.0;
    }

    std::size_t hits() const noexcept { return hits_.load(); }
    std::size_t misses() const noexcept { return misses_.load(); }

private:
    std::unique_ptr<policy_type> policy_;
    cache_type cache_;
    mutable std::mutex policy_mutex_;  // guards all policy_ calls (approx_split_set is not thread-safe)

    mutable std::atomic<std::size_t> hits_{0};
    mutable std::atomic<std::size_t> misses_{0};
};

} // namespace lru

#endif // LRU_ADMISSION_HPP
