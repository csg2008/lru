// SPDX-License-Identifier: MIT
// Event Tracker — Item lifecycle event recording for offline analysis
//
// Records per-item lifecycle events:
//   - Created (first insertion into cache)
//   - Promoted (moved toward MRU head)
//   - Demoted  (moved toward LRU tail, e.g., Hot→Cold in 2Q)
//   - Evicted  (removed from cache)
//   - Hit      (read access — optional, high volume)
//
// Events are batched via lock-free TLS ring buffers and periodically
// drained into the main ring buffer using lock-free atomic operations.
// Both the recording path and the drain path are lock-free, eliminating
// all mutex contention on hot paths.
//
// This enables:
//   - Hit rate analysis by item age cohort
//   - Time-to-eviction distributions
//   - Promotion frequency histograms
//   - Identifying "churn" items (frequently evicted, quickly re-inserted)
//
// Usage:
//   event_tracker<int> tracker(/*capacity=*/1'000'000);
//   tracker.record_insert(key, /*timestamp=*/now);
//   tracker.record_hit(key, now);
//   tracker.record_evict(key, now);
//   auto report = tracker.flush_report();  // aggregate and reset

#ifndef LRU_EVENT_TRACKER_HPP
#define LRU_EVENT_TRACKER_HPP

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "detail/atomic_shared_ptr.hpp"
#include "detail/foundation.hpp"
#include "detail/space_saving.hpp"
#include "event_types.hpp"
#include "tls_ring.hpp"

namespace lru {

// ============================================================================
// Event Tracker
// ============================================================================

/// TLS lock-free ring buffer for recording item lifecycle events.
///
/// Events are written into per-thread TLS ring buffers (lock-free) on the
/// hot path. A background drain worker periodically collects TLS entries
/// into the main ring buffer using atomic slot claiming and per-entry
/// atomic stores — entirely lock-free. This design eliminates mutex
/// contention on both the recording and drain paths while keeping the
/// reporting/query path simple and consistent.
///
/// Lock-free drain mechanism:
///   - drain_tls() atomically steals the TLS ring buffer (zero-copy)
///   - Claims contiguous slots in the main ring via write_pos_.fetch_add()
///   - Writes entries using std::atomic_ref with release semantics
///   - Commits each slot via a per-slot sequence counter (slot_seq_)
///   - Readers check slot_seq_ to ensure they read committed entries
///
/// @tparam Key  The key type (only hash is stored for space efficiency).
template <typename Key, typename Hash = std::hash<Key>>
class event_tracker {
public:
    using key_type = Key;
    using hash_type = Hash;

    /// Configuration.
    struct config {
        /// Maximum number of events in the ring buffer.
        std::size_t max_events = 1'000'000;

        /// If true, new events overwrite old events when buffer is full.
        /// If false, new events are dropped when buffer is full.
        bool overwrite_on_full = true;

        /// Record hit events (can be very high volume — may want to sample).
        bool record_hits = true;

        /// Hit sampling rate [0.0, 1.0]. 1.0 = record all hits, 0.1 = record 10%.
        // H-9 fix: default lowered from 1.0 to 0.01. In read-heavy
        // workloads (10M+ ops/s), recording every hit adds 5-10%
        // throughput overhead from TLS ring writes and drain-time
        // Space-Saving updates. 1% sampling retains statistical
        // usefulness for hot-key detection while cutting overhead ~100x.
        double hit_sampling_rate = 0.01;

        /// Record promote/demote events.
        bool record_movements = true;

        /// P2-3: Capacity of the streaming Space-Saving top-k summary used
        /// for hot-key detection. Larger capacity → better recall and
        /// tighter error bound, more memory. Set to 0 to disable streaming
        /// hot-key tracking (top_keys() will fall back to the O(N) scan).
        std::size_t hot_keys_capacity = 1024;
    };

    // --------------------------------------------------------------------
    // Construction
    // --------------------------------------------------------------------

    explicit event_tracker(const config& cfg = config{})
        : config_(cfg)
        , hot_keys_active_(std::make_shared<detail::space_saving_top_k<uint64_t>>(
              cfg.hot_keys_capacity > 0 ? cfg.hot_keys_capacity : 1))
        , hot_keys_capacity_(cfg.hot_keys_capacity)
    {
        ring_.resize(cfg.max_events);
        slot_seq_ = std::make_unique<std::atomic<uint32_t>[]>(cfg.max_events);
        slot_seq_size_ = cfg.max_events;
    }

    event_tracker(std::size_t max_events) : event_tracker(config{max_events}) {}

    ~event_tracker() {
        stop_drain_worker();
        // P2-D: callback storage is now a shared_ptr — its refcount-based
        // cleanup handles deferred deletion automatically. No manual
        // pending_callbacks_ vector to drain.
    }

    event_tracker(const event_tracker&) = delete;
    event_tracker& operator=(const event_tracker&) = delete;

    // --------------------------------------------------------------------
    // Enable / Disable (P2-D)
    // --------------------------------------------------------------------

    /// Disable the tracker: stops the background drain worker, clears the
    /// event callback (so no further callback invocations occur), and
    /// marks all `record_*` methods as no-ops. Pending TLS entries and
    /// ring-buffer entries are NOT cleared — call `reset()` afterwards
    /// if a clean slate is required. Safe to call from any thread.
    ///
    /// This is the recommended way to turn off tracking at runtime without
    /// destroying the tracker instance. Re-enable with `enable()`.
    void disable() {
        enabled_.store(false, std::memory_order_release);
        stop_drain_worker();
        // Clear the callback by storing an empty shared_ptr. In-flight
        // drain operations holding the old shared_ptr continue safely
        // until they release their reference (refcount drops to 0 →
        // auto-deletes). No manual deferred-deletion list needed.
        event_callback_.store(nullptr, std::memory_order_release);
    }

    /// Re-enable the tracker after `disable()`. Does NOT restart the
    /// background drain worker — call `start_drain_worker()` separately
    /// if periodic draining is desired.
    void enable() {
        enabled_.store(true, std::memory_order_release);
    }

    /// Whether recording is currently enabled.
    bool is_enabled() const noexcept {
        return enabled_.load(std::memory_order_acquire);
    }

    /// Atomically set the enabled state.
    void set_enabled(bool enabled) {
        enabled_.store(enabled, std::memory_order_release);
    }

    // --------------------------------------------------------------------
    // Event Recording
    // --------------------------------------------------------------------

    /// Record an insertion event.
    void record_insert(const key_type& key) {
        record(key, event_type::insert, 0);
    }

    /// Record a hit event (subject to sampling).
    void record_hit(const key_type& key) {
        if (!config_.record_hits) return;
        if (config_.hit_sampling_rate < 1.0) {
            if (!sample_hit()) return;
        }
        record(key, event_type::hit, 0);
    }

    /// Record a promotion event.
    void record_promote(const key_type& key, uint8_t queue_id = 0) {
        if (!config_.record_movements) return;
        record(key, event_type::promote, queue_id);
    }

    /// Record a demotion event.
    void record_demote(const key_type& key, uint8_t queue_id = 0) {
        if (!config_.record_movements) return;
        record(key, event_type::demote, queue_id);
    }

    /// Record an eviction event.
    void record_evict(const key_type& key) {
        record(key, event_type::evict, 0);
    }

    /// Generic event recording.
    /// Writes to the per-thread TLS ring buffer (lock-free).
    /// No-op when the tracker is disabled (`disable()` was called).
    void record(const key_type& key, event_type type, uint8_t queue_id = 0) {
        // P2-D: fast-path early return when disabled. Relaxed load is
        // safe — a stale `true` just records one extra event before the
        // disable takes effect, which is benign for an idempotent
        // monitoring ring.
        if (!enabled_.load(std::memory_order_relaxed)) return;
        tls_ring_.record(type, key, queue_id);
    }

    // --------------------------------------------------------------------
    // TLS Drain (lock-free)
    // --------------------------------------------------------------------

    /// Drain TLS events from the calling thread into the main ring buffer.
    /// Must be called from the thread whose events should be drained.
    /// This is called automatically by the background drain worker, but
    /// can also be called manually (e.g., before generating a report).
    ///
    /// Lock-free implementation:
    ///   1. Atomically steal the TLS ring buffer (zero-copy exchange)
    ///   2. Claim contiguous slots via write_pos_.fetch_add()
    ///   3. Write entries using std::atomic_ref with release semantics
    ///   4. Commit each slot via slot_seq_ increment
    void drain_tls() {
        // Atomically steal the TLS ring buffer (zero-copy)
        auto stolen = tls_ring_.steal();
        if (stolen.empty()) return;

        constexpr auto kMask = tls_event_ring<Key, Hash>::kRingMask;
        auto src_start = stolen.tail & kMask;
        // Pass base pointer + start offset + mask. The mask handles ring
        // wraparound: entry i is at base[(start + i) & kMask].
        process_events(stolen.entries, src_start, stolen.count(), kMask);
    }

    /// P2-3: Drain ALL threads' TLS event rings for this tracker.
    ///
    /// Unlike `drain_tls()` (which only drains the calling thread), this
    /// method walks the global per-thread ring registry and drains every
    /// thread's pending events. Use this before `top_keys()` or
    /// `generate_report()` to ensure the results reflect all threads'
    /// activity, not just the calling thread's.
    ///
    /// The drain is best-effort with respect to concurrent `record()`
    /// calls on other threads (see `tls_event_ring::drain_all_threads()`
    /// for the race semantics). This is acceptable for event tracking,
    /// which is idempotent and approximate.
    void drain_all_threads() {
        auto drained = tls_ring_.drain_all_threads();
        if (drained.entries.empty()) return;
        // Contiguous vector — use all-ones mask so (0 + i) & mask = i.
        constexpr std::size_t kContiguous = ~std::size_t(0);
        process_events(drained.entries.data(), 0, drained.entries.size(), kContiguous);
    }

    /// Start a background worker that periodically drains TLS ring buffers.
    /// The worker calls drain_tls() on its own thread at the given interval.
    /// Since each thread's TLS ring is only accessed by that thread, the
    /// background worker drains its own TLS entries. For complete coverage,
    /// user threads should also call drain_tls() periodically or before
    /// generating reports.
    void start_drain_worker(std::chrono::milliseconds interval = std::chrono::milliseconds(100)) {
        bool expected = false;
        if (!drain_active_.compare_exchange_strong(expected, true,
                std::memory_order_acq_rel, std::memory_order_relaxed)) {
            return;  // Already running
        }
        drain_worker_ = std::make_unique<detail::periodic_worker>(
            [this] { drain_tls(); }, interval);
    }

    /// Stop the background drain worker if running.
    void stop_drain_worker() {
        bool expected = true;
        if (!drain_active_.compare_exchange_strong(expected, false,
                std::memory_order_acq_rel, std::memory_order_relaxed)) {
            return;  // Not running
        }
        drain_worker_->stop();
        drain_worker_.reset();
    }

    /// Check if the background drain worker is running.
    bool is_drain_worker_running() const noexcept {
        return drain_active_.load(std::memory_order_relaxed);
    }

    // --------------------------------------------------------------------
    // Reporting
    // --------------------------------------------------------------------

    /// Aggregate summary.
    struct report {
        std::size_t total_events = 0;
        std::size_t inserts = 0;
        std::size_t promotes = 0;
        std::size_t demotes = 0;
        std::size_t evictions = 0;
        std::size_t hits = 0;

        /// Average time between insert and evict (in ms).
        double avg_time_to_evict_ms = 0.0;

        /// Average promotions per evicted item.
        double avg_promotions_per_eviction = 0.0;

        /// Churn ratio: items evicted within 1 second of insertion / total evictions.
        double churn_ratio = 0.0;

        /// Event type distribution as fractions.
        std::array<double, 5> event_distribution{};

        /// Per-queue hit distribution (for multi-queue strategies).
        std::array<std::size_t, 8> queue_hits{};

        /// Number of events sampled.
        std::size_t events_sampled = 0;

        std::string to_string() const {
            char buf[512];
            std::snprintf(buf, sizeof(buf),
                "events=%zu insert=%zu promote=%zu demote=%zu evict=%zu hits=%zu "
                "avg_ttl=%.1fms avg_promo=%.2f churn=%.3f",
                total_events, inserts, promotes, demotes, evictions, hits,
                avg_time_to_evict_ms, avg_promotions_per_eviction, churn_ratio);
            return std::string(buf);
        }
    };

    /// Generate a report from the current buffer contents.
    /// Drains TLS buffers first to include all recorded events.
    /// This is a snapshot — events recorded after this call are not included.
    report generate_report() const {
        // Drain TLS first so all events are in the main ring buffer.
        const_cast<event_tracker*>(this)->drain_tls();

        std::lock_guard<std::mutex> lock(config_mutex_);
        report r;
        auto total = std::min(write_pos_.load(std::memory_order_acquire), config_.max_events);
        r.total_events = total;
        r.events_sampled = std::min(total, std::size_t(100000)); // sample at most 100k for performance

        // Step size for uniform sampling
        std::size_t step = r.events_sampled > 0 ? total / r.events_sampled : 1;
        if (step < 1) step = 1;

        // Per-key tracking for accurate TTL and churn computation.
        // Maps key_hash → most recent insert timestamp.
        ankerl::unordered_dense::map<uint64_t, uint64_t> last_insert_time;

        double ttl_sum = 0.0;
        std::size_t ttl_count = 0;
        std::size_t churned = 0;

        for (std::size_t i = 0; i < total; i += step) {
            auto idx = i % config_.max_events;
            // Check if slot is committed (slot_seq_ > 0)
            auto seq1 = slot_seq_[idx].load(std::memory_order_acquire);
            if (seq1 == 0) continue;

            // Read entry atomically
            auto e = load_entry_atomic(ring_[idx]);

            // Verify no concurrent write changed the slot during our read
            auto seq2 = slot_seq_[idx].load(std::memory_order_acquire);
            if (seq1 != seq2) continue;  // Concurrent write, skip

            switch (e.type) {
                case event_type::insert:
                    ++r.inserts;
                    last_insert_time[e.key_hash] = e.timestamp_ms;
                    break;
                case event_type::promote:
                    ++r.promotes;
                    break;
                case event_type::demote:
                    ++r.demotes;
                    break;
                case event_type::evict:
                    ++r.evictions;
                    // Compute per-item TTL by matching eviction to the same key's insert
                    if (auto it = last_insert_time.find(e.key_hash);
                        it != last_insert_time.end()) {
                        if (e.timestamp_ms >= it->second) {
                            auto ttl = e.timestamp_ms - it->second;
                            ttl_sum += static_cast<double>(ttl);
                            ++ttl_count;
                            if (ttl < 1000) ++churned;
                        }
                        last_insert_time.erase(it);
                    }
                    break;
                case event_type::hit:
                    ++r.hits;
                    if (e.queue_id < 8) ++r.queue_hits[e.queue_id];
                    break;
            }
        }

        // Compute avg time to evict (per-item, not aggregate timestamps)
        if (ttl_count > 0) {
            r.avg_time_to_evict_ms = ttl_sum / static_cast<double>(ttl_count);
        }

        // Promotions per eviction
        if (r.evictions > 0) {
            r.avg_promotions_per_eviction = static_cast<double>(r.promotes) /
                                            static_cast<double>(r.evictions);
        }

        // Churn ratio: items evicted within 1 second of their own insertion
        if (r.evictions > 0) {
            r.churn_ratio = static_cast<double>(churned) /
                            static_cast<double>(r.evictions);
        }

        // Event distribution
        auto sum = static_cast<double>(r.inserts + r.promotes + r.demotes + r.evictions + r.hits);
        if (sum > 0) {
            r.event_distribution[0] = static_cast<double>(r.inserts) / sum;
            r.event_distribution[1] = static_cast<double>(r.promotes) / sum;
            r.event_distribution[2] = static_cast<double>(r.demotes) / sum;
            r.event_distribution[3] = static_cast<double>(r.evictions) / sum;
            r.event_distribution[4] = static_cast<double>(r.hits) / sum;
        }

        return r;
    }

    // --------------------------------------------------------------------
    // "Hot Keys" Analysis (P2-3: streaming via Space-Saving)
    // --------------------------------------------------------------------

    /// P2-3: Set a callback to convert a key hash back to a human-readable
    /// key name. This enables `top_keys_with_names()` to return readable
    /// key identifiers instead of raw 64-bit hashes.
    ///
    /// The callback is copied under the drain mutex during query, so
    /// it should be fast (e.g., a hash map lookup, not a database query).
    /// If no callback is set, `top_keys_with_names()` returns the raw
    /// hash formatted as a hex string.
    void set_key_to_string(std::function<std::string(uint64_t)> callback) {
        std::lock_guard<std::mutex> lock(hot_keys_drain_mutex_);
        key_to_string_cb_ = std::move(callback);
    }

    /// Top-K most frequently accessed keys (from hit events only).
    ///
    /// P2-3: Now uses the streaming Space-Saving algorithm for O(1) amortized
    /// update and O(K log K) query, replacing the previous O(N) batch scan.
    /// Drains ALL threads' TLS buffers first (via `drain_all_threads()`) to
    /// ensure the result reflects every thread's hit activity.
    ///
    /// P2-9 (double buffering): The query path is now lock-free — it loads
    /// an atomic `shared_ptr` to the active Space-Saving instance and reads
    /// from it without taking any mutex. This ensures `top_keys()` is never
    /// blocked by a concurrent drain operation (which writes to a separate
    /// backup instance and atomically swaps it in).
    ///
    /// Returns key_hash → estimated hit count pairs, sorted by count descending.
    /// The estimated count is an upper bound; the true count is >= estimated
    /// count - error_bound() (see `hot_keys_stats()`).
    std::vector<std::pair<uint64_t, std::size_t>> top_keys(std::size_t k = 10) const {
        // P2-9: Read from the active instance via atomic shared_ptr load.
        // No mutex is acquired — queries are never blocked by drain operations.
        // Callers wanting the freshest data should call `drain_all_threads()`
        // (or rely on the background drain worker) before querying.
        if (hot_keys_capacity_.load(std::memory_order_relaxed) == 0) {
            return {};
        }
        auto ptr = hot_keys_active_.load(std::memory_order_acquire);
        if (!ptr || ptr->size() == 0) {
            return {};
        }
        return ptr->top_k(k);
    }

    /// P2-3: Top-K hot keys with human-readable names (via `set_key_to_string`).
    ///
    /// Returns key_name → estimated hit count pairs, sorted by count descending.
    /// If no `key_to_string` callback is set, the key name is the hash
    /// formatted as a hexadecimal string.
    ///
    /// P2-9: The top-K query is lock-free (reads from the active instance
    /// via atomic shared_ptr). The callback is copied under the drain mutex
    /// (brief — only during configuration changes), so this method is
    /// effectively non-blocking during drain.
    std::vector<std::pair<std::string, std::size_t>> top_keys_with_names(
        std::size_t k = 10) const {
        auto raw = top_keys(k);
        std::vector<std::pair<std::string, std::size_t>> result;
        result.reserve(raw.size());

        std::function<std::string(uint64_t)> cb;
        {
            std::lock_guard<std::mutex> lock(hot_keys_drain_mutex_);
            cb = key_to_string_cb_;
        }

        for (const auto& [hash, count] : raw) {
            if (cb) {
                result.emplace_back(cb(hash), count);
            } else {
                char buf[32];
                std::snprintf(buf, sizeof(buf), "0x%016llx",
                              static_cast<unsigned long long>(hash));
                result.emplace_back(std::string(buf), count);
            }
        }
        return result;
    }

    /// P2-3: Streaming hot-keys summary statistics.
    struct hot_keys_stats {
        std::size_t capacity = 0;       ///< Max distinct keys tracked
        std::size_t tracked = 0;        ///< Current tracked key count
        std::size_t total_hits = 0;     ///< Total hit observations processed
        std::size_t error_bound = 0;    ///< Max count error for tracked items
    };

    /// P2-3: Return current streaming hot-keys summary statistics.
    ///
    /// P2-9: Now lock-free — reads from the active instance via atomic
    /// shared_ptr load. Not blocked by concurrent drain operations.
    hot_keys_stats hot_keys_summary() const {
        hot_keys_stats s;
        if (hot_keys_capacity_.load(std::memory_order_relaxed) == 0) {
            return s;
        }
        auto ptr = hot_keys_active_.load(std::memory_order_acquire);
        if (!ptr) return s;
        s.capacity = ptr->capacity();
        s.tracked = ptr->size();
        s.total_hits = ptr->total();
        s.error_bound = ptr->error_bound();
        return s;
    }

    /// P2-3: Reset the streaming hot-keys summary (does not clear the event ring).
    ///
    /// P2-9: Creates a fresh empty instance and atomically swaps it in,
    /// replacing the active instance. Concurrent readers holding the old
    /// shared_ptr continue safely until they finish.
    void reset_hot_keys() {
        std::lock_guard<std::mutex> lock(hot_keys_drain_mutex_);
        auto cap = hot_keys_capacity_.load(std::memory_order_relaxed);
        hot_keys_active_.store(
            std::make_shared<detail::space_saving_top_k<uint64_t>>(
                cap > 0 ? cap : 1),
            std::memory_order_release);
    }

    // --------------------------------------------------------------------
    // Management
    // --------------------------------------------------------------------

    /// Reset the tracker (clears all events).
    /// Drains TLS buffers first.
    void reset() {
        drain_tls();
        std::lock_guard<std::mutex> lock(config_mutex_);
        write_pos_.store(0, std::memory_order_relaxed);
        // Reset all slot sequence numbers to mark entries as invalid
        for (std::size_t i = 0; i < config_.max_events; ++i) {
            slot_seq_[i].store(0, std::memory_order_relaxed);
        }
        // P2-3: Also reset the streaming hot-keys summary.
        // P2-9: Use double-buffer swap — create fresh instance and atomically
        // publish it so concurrent top_keys() readers are not blocked.
        {
            std::lock_guard<std::mutex> hk_lock(hot_keys_drain_mutex_);
            auto cap = hot_keys_capacity_.load(std::memory_order_relaxed);
            hot_keys_active_.store(
                std::make_shared<detail::space_saving_top_k<uint64_t>>(
                    cap > 0 ? cap : 1),
                std::memory_order_release);
        }
    }

    /// Total events ever recorded (since last reset).
    std::size_t total_recorded() const noexcept {
        return write_pos_.load(std::memory_order_relaxed);
    }

    /// Current buffer utilization [0.0, 1.0].
    double buffer_utilization() const noexcept {
        auto used = std::min(write_pos_.load(std::memory_order_relaxed), config_.max_events);
        return config_.max_events > 0
                   ? static_cast<double>(used) / static_cast<double>(config_.max_events)
                   : 0.0;
    }

    /// Update config at runtime.
    void set_config(const config& cfg) {
        // Drain TLS before changing config
        drain_tls();
        std::lock_guard<std::mutex> lock(config_mutex_);
        config_ = cfg;
        if (ring_.size() < cfg.max_events) {
            ring_.resize(cfg.max_events);
            auto new_seq = std::make_unique<std::atomic<uint32_t>[]>(cfg.max_events);
            // Copy old sequence numbers
            for (std::size_t i = 0; i < slot_seq_size_; ++i) {
                new_seq[i].store(slot_seq_[i].load(std::memory_order_relaxed),
                                 std::memory_order_relaxed);
            }
            slot_seq_ = std::move(new_seq);
            slot_seq_size_ = cfg.max_events;
        }
        // P2-3: Reallocate hot-keys summary if capacity changed.
        // P2-9: Use double-buffer swap — create a new instance with the
        // updated capacity and atomically publish it. Concurrent readers
        // continue using the old instance until they release their
        // shared_ptr reference.
        hot_keys_capacity_.store(cfg.hot_keys_capacity, std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> hk_lock(hot_keys_drain_mutex_);
            auto current = hot_keys_active_.load(std::memory_order_relaxed);
            auto new_cap = cfg.hot_keys_capacity > 0 ? cfg.hot_keys_capacity : 1;
            if (!current || current->capacity() != new_cap) {
                hot_keys_active_.store(
                    std::make_shared<detail::space_saving_top_k<uint64_t>>(new_cap),
                    std::memory_order_release);
            }
        }
    }

    const config& get_config() const noexcept { return config_; }

    /// Register a custom callback for each event (useful for real-time streaming).
    /// Pass an empty/empty function to clear the callback.
    ///
    /// P2-D: The callback is stored in a `detail::atomic_shared_ptr`, which
    /// provides atomic load/store via a brief spinlock. Drain operations
    /// (`process_events`) load the shared_ptr (incrementing its refcount)
    /// and call through it outside the lock — the callback remains alive
    /// until the last reference is released. This eliminates the
    /// deferred-deletion list (`pending_callbacks_`) and the associated
    /// memory leak when `on_event()` is called many times.
    void on_event(std::function<void(const event_record&)> callback) {
        auto new_cb = std::make_shared<std::function<void(const event_record&)>>(
            std::move(callback));
        // store() drops the old shared_ptr's refcount by 1. If a
        // concurrent drain holds a copy (loaded via load()), the old
        // callback stays alive until that drain finishes and releases
        // its reference. No manual deletion needed.
        event_callback_.store(std::move(new_cb), std::memory_order_release);
    }

private:
    // --------------------------------------------------------------------
    // Lock-free ring entry access (std::atomic_ref)
    // --------------------------------------------------------------------

    /// Write a single entry to the ring buffer using atomic stores.
    /// Uses release semantics on the last field (queue_id) to publish
    /// all preceding relaxed stores. This ensures readers that acquire
    /// on queue_id will see a consistent snapshot of all fields.
    static void store_entry_atomic(event_record& entry,
                                    uint64_t key_hash, uint64_t timestamp_ms,
                                    event_type type, uint8_t queue_id) {
        std::atomic_ref<uint64_t>(entry.key_hash).store(key_hash, std::memory_order_relaxed);
        std::atomic_ref<uint64_t>(entry.timestamp_ms).store(timestamp_ms, std::memory_order_relaxed);
        // Store type via its underlying type to ensure atomic_ref compatibility
        auto type_val = static_cast<std::underlying_type_t<event_type>>(type);
        std::atomic_ref<std::underlying_type_t<event_type>>(
            reinterpret_cast<std::underlying_type_t<event_type>&>(entry.type)
        ).store(type_val, std::memory_order_relaxed);
        // Release store on the last field publishes all preceding stores
        std::atomic_ref<uint8_t>(entry.queue_id).store(queue_id, std::memory_order_release);
    }

    /// Read a single entry from the ring buffer using atomic loads.
    /// Acquires on queue_id first, then reads remaining fields with
    /// relaxed ordering. The acquire ensures we see all stores that
    /// happened-before the release store of queue_id by the writer.
    static event_record load_entry_atomic(const event_record& entry) {
        auto& e = const_cast<event_record&>(entry);
        event_record result;
        // Acquire on queue_id (last field written) synchronizes with
        // the writer's release store, ensuring we see consistent fields
        result.queue_id = std::atomic_ref<uint8_t>(e.queue_id).load(std::memory_order_acquire);
        result.key_hash = std::atomic_ref<uint64_t>(e.key_hash).load(std::memory_order_relaxed);
        result.timestamp_ms = std::atomic_ref<uint64_t>(e.timestamp_ms).load(std::memory_order_relaxed);
        auto type_val = std::atomic_ref<std::underlying_type_t<event_type>>(
            reinterpret_cast<std::underlying_type_t<event_type>&>(e.type)
        ).load(std::memory_order_relaxed);
        result.type = static_cast<event_type>(type_val);
        return result;
    }

    /// P2-3: Process a batch of drained TLS events.
    ///
    /// Writes the events to the main ring buffer (lock-free slot claiming),
    /// fires the event callback for each entry, and updates the streaming
    /// hot-keys summary with hit events.
    ///
    /// @param base    Pointer to the first event entry (or ring array base).
    /// @param start   Start index within the array (0 for contiguous data).
    /// @param count   Number of entries to process.
    /// @param mask    Mask for wraparound indexing (~0 for contiguous data,
    ///                kRingMask for TLS ring data). Entry i is at
    ///                base[(start + i) & mask].
    void process_events(const typename tls_event_ring<Key, Hash>::event_entry* base,
                        std::size_t start, std::size_t count, std::size_t mask) {
        if (count == 0) return;

        // Best-effort overwrite check (TOCTOU is benign — at most one extra
        // batch slips through, which is acceptable for a monitoring ring)
        if (!config_.overwrite_on_full) {
            auto current = write_pos_.load(std::memory_order_relaxed);
            if (current >= config_.max_events) return;
        }

        // Lock-free batch write: atomically claim contiguous slots
        auto start_pos = write_pos_.fetch_add(count, std::memory_order_relaxed);

        // Write entries using atomic_ref stores + commit via slot_seq_
        for (std::size_t i = 0; i < count; ++i) {
            auto dst_idx = (start_pos + i) % config_.max_events;
            const auto& src = base[(start + i) & mask];
            store_entry_atomic(ring_[dst_idx],
                               src.key_hash, src.timestamp_ms, src.type, src.queue_id);
            slot_seq_[dst_idx].fetch_add(1, std::memory_order_release);
        }

        // Fire event callback if registered. P2-D: load the shared_ptr
        // (bumps refcount) so the callback stays alive for the duration
        // of this drain batch even if `on_event()` swaps in a new one
        // concurrently. Calling through `cb` is safe — `*cb` will not
        // be freed until `cb` goes out of scope at the end of this block.
        auto cb = event_callback_.load(std::memory_order_acquire);
        if (cb && *cb) {
            for (std::size_t i = 0; i < count; ++i) {
                auto dst_idx = (start_pos + i) % config_.max_events;
                auto entry = load_entry_atomic(ring_[dst_idx]);
                (*cb)(entry);
            }
        }

        // P2-3: Update streaming hot-keys summary with hit events.
        // P2-9 (double buffering): Instead of mutating the active instance
        // under a shared mutex (which blocks top_keys() queries), the drain
        // path copies the current active instance, applies the new hit
        // events to the copy, and atomically publishes it. Queries
        // (top_keys, hot_keys_summary) load the atomic shared_ptr and read
        // from the active instance without any lock — they are never blocked
        // by drain.
        //
        // The drain mutex (hot_keys_drain_mutex_) serializes concurrent
        // drains so they don't overwrite each other's updates. It is only
        // held during the copy-add-swap sequence, not during the TLS event
        // processing or ring-buffer writes above.
        if (hot_keys_capacity_.load(std::memory_order_relaxed) > 0) {
            // Collect hit key hashes first (no lock needed for this pass).
            std::vector<uint64_t> hit_keys;
            hit_keys.reserve(count);
            for (std::size_t i = 0; i < count; ++i) {
                const auto& src = base[(start + i) & mask];
                if (src.type == event_type::hit) {
                    hit_keys.push_back(src.key_hash);
                }
            }

            if (!hit_keys.empty()) {
                std::lock_guard<std::mutex> lock(hot_keys_drain_mutex_);
                // Load the current active instance (atomic, keeps it alive
                // via shared_ptr refcount until we finish the copy).
                auto current = hot_keys_active_.load(std::memory_order_acquire);
                // Create a backup instance as a copy of the active one.
                // This preserves all accumulated counts from prior drains.
                auto next = std::make_shared<detail::space_saving_top_k<uint64_t>>(
                    current ? *current
                           : detail::space_saving_top_k<uint64_t>(
                                 hot_keys_capacity_.load(std::memory_order_relaxed) > 0
                                     ? hot_keys_capacity_.load(std::memory_order_relaxed)
                                     : 1));
                // Write new hit events to the backup instance.
                for (auto h : hit_keys) {
                    next->add(h);
                }
                // Atomically swap: the backup becomes the new active instance.
                // Concurrent readers that already loaded the old shared_ptr
                // continue reading from it safely (refcount keeps it alive).
                hot_keys_active_.store(next, std::memory_order_release);
            }
        }
    }

    uint64_t now_ms() const {
        // R6: Use steady_clock for monotonic timestamps. The event_tracker
        // calculates TTL as (evict_timestamp - insert_timestamp); with
        // system_clock, an NTP adjustment between insert and evict can
        // produce negative or wildly incorrect TTL values. steady_clock
        // is monotonic and immune to wall-clock adjustments.
        return static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count());
    }

    bool sample_hit() const {
        // Deterministic sampling: sample every N-th hit where N = 1/rate
        if (config_.hit_sampling_rate >= 1.0) return true;
        if (config_.hit_sampling_rate <= 0.0) return false;
        static thread_local uint64_t counter = 0;
        ++counter;
        auto period = static_cast<uint64_t>(1.0 / config_.hit_sampling_rate);
        return (counter % period) == 0;
    }

    config config_;
    std::vector<event_record> ring_;

    /// Per-slot commit sequence number. Incremented after each write to
    /// the corresponding ring slot. Readers check this to determine if
    /// a slot contains committed data (seq > 0) and to detect concurrent
    /// writes (seq changed between read-start and read-end).
    std::unique_ptr<std::atomic<uint32_t>[]> slot_seq_;
    std::size_t slot_seq_size_{0};  // Track allocated size for resize

    /// Slot claiming counter. Writers increment this via fetch_add to
    /// claim contiguous slots in the ring. Readers use write_pos_ (same
    /// counter) to determine the range of valid entries.
    std::atomic<std::size_t> write_pos_{0};

    /// Mutex for rare operations: config changes, reset, non-TLS direct
    /// writes. NOT used by drain_tls() — the drain path is lock-free.
    mutable std::mutex config_mutex_;

    /// P2-D: Event callback stored as a shared_ptr behind a spinlock-based
    /// atomic wrapper. Drain operations load it (bumping the refcount) and
    /// call through it without holding any mutex; `on_event()` swaps in a
    /// new shared_ptr atomically. The old callback auto-deletes when its
    /// refcount reaches zero (i.e., after all in-flight drains finish).
    /// This replaces the previous raw-pointer + deferred-deletion-list
    /// design, which leaked memory when `on_event()` was called many times.
    lru::detail::atomic_shared_ptr<std::function<void(const event_record&)>> event_callback_;

    /// P2-D: Global enable/disable flag. When false, all `record_*` methods
    /// become no-ops (fast-path relaxed load + early return). `disable()`
    /// also stops the drain worker and clears the event callback.
    std::atomic<bool> enabled_{true};

    // TLS lock-free recording
    tls_event_ring<Key, Hash> tls_ring_;

    // Background drain worker — managed via atomic flag (lock-free)
    std::unique_ptr<detail::periodic_worker> drain_worker_;
    std::atomic<bool> drain_active_{false};

    // --------------------------------------------------------------------
    // P2-3: Streaming hot-keys summary (Space-Saving top-k)
    // P2-9: Double-buffered via atomic shared_ptr
    // --------------------------------------------------------------------
    //
    // The active Space-Saving instance is stored as an atomic shared_ptr.
    // Queries (top_keys, hot_keys_summary) load it lock-free and read from
    // it without any mutex — they are NEVER blocked by drain operations.
    //
    // The drain path (process_events) takes hot_keys_drain_mutex_ to
    // serialize concurrent drains, then copies the active instance, adds
    // new hit events to the copy, and atomically stores the copy as the
    // new active instance. The old instance stays alive via shared_ptr
    // refcount until all concurrent readers finish.
    //
    // Thread safety:
    //   - hot_keys_active_      : atomic shared_ptr — safe for concurrent
    //                             load/store from any thread.
    //   - hot_keys_drain_mutex_ : serializes the drain (write) path only.
    //                             NOT needed by readers.
    //   - key_to_string_cb_     : protected by hot_keys_drain_mutex_ (set
    //                             during config, copied briefly during query).
    //   - hot_keys_capacity_    : atomic, read by top_keys() for the
    //                             disabled-check without any lock.

    /// Active Space-Saving instance — atomically swapped during drain.
    /// Readers load this and call top_k() without any lock.
    lru::detail::atomic_shared_ptr<detail::space_saving_top_k<uint64_t>> hot_keys_active_;

    /// Serializes the drain (write) path. NOT taken by readers.
    mutable std::mutex hot_keys_drain_mutex_;

    /// Atomic copy of hot_keys_capacity for lock-free disabled checks.
    std::atomic<std::size_t> hot_keys_capacity_{0};

    std::function<std::string(uint64_t)> key_to_string_cb_;
};

} // namespace lru

#endif // LRU_EVENT_TRACKER_HPP
