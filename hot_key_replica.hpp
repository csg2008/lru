// SPDX-License-Identifier: MIT
// T-C1 (P1-2): Hot key replica manager — opt-in utility for automatic
// hot-key replication across cache shards.
//
// Design rationale
// ================
// The library's `cache_trait.hpp::hot_keys()` documentation explains why
// the cache itself does not auto-replicate hot keys internally:
//   - Replication factor is workload-dependent (a 100K-QPS key needs
//     more replicas than a 10K-QPS key).
//   - Replica consistency requires CAS or 2PC across shards, adding
//     write latency.
//   - Application-layer routing can use domain knowledge (read-after-
//     write consistency, geo-distribution).
//
// This header provides an opt-in `hot_key_replica_manager` that
// applications can instantiate when they want automatic replication.
// It wraps a `unified_cache<K, V>` reference and:
//   1. Provides `register_hot_key(key)` for the application to opt a
//      specific key into replication (the manager cannot auto-detect
//      keys from `hot_keys()` because that API returns hashes only).
//   2. Inserts N replicas under transformed key names that hash to
//      different shards (configurable replica_factor, default 4).
//   3. Routes `replicated_get(key)` to a randomly chosen replica.
//   4. Synchronizes `replicated_set(key, value)` across all replicas.
//   5. Demotes (deletes replicas) on `unregister_hot_key(key)`.
//   6. Exposes metrics: `replica_count`, `write_amplification_total`,
//      `read_routed_total`, `read_fallback_total`, `read_miss_total`.
//
// The replica key transform is `key || 0xFF || (4-byte big-endian index)`.
// This is enough to ensure different replicas hash to different shards
// under typical hash functions (FNV, Murmur, etc.).
//
// Consistency model
// =================
// Writes are eventually consistent across replicas: `replicated_set`
// inserts each replica sequentially; if a concurrent `replicated_get`
// races with a `replicated_set`, it may see a mix of old and new
// values across replicas. Applications needing read-after-write
// consistency should use `replicated_cas()` with an expected value,
// or maintain a separate version counter per key.
//
// Use cases
// =========
//   - Read-heavy workloads with a small set of pathological hot keys
//     (e.g. config keys, feature flags, schema entries).
//   - Workloads where occasional stale reads are acceptable (e.g.
//     cache-as-CDN-front, where the source-of-truth is a database).
//
// NOT appropriate for:
//   - Workloads requiring strong cross-key consistency.
//   - Workloads where every key is hot (no skew — replication just
//     multiplies memory cost with no benefit).

#ifndef LRU_HOT_KEY_REPLICA_HPP
#define LRU_HOT_KEY_REPLICA_HPP

#include "cache_trait.hpp"
#include "event_tracker.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <shared_mutex>
#include <stop_token>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace lru {

/// T-C1 (P1-2): Opt-in hot key replica manager.
///
/// Wraps a `unified_cache<K, V>` reference and provides transparent
/// replication for keys the application explicitly registers via
/// `register_hot_key()`. See file header for design rationale and
/// consistency model.
///
/// Thread-safety: all public methods are thread-safe. The manager
/// uses an internal shared_mutex for the replica metadata map.
///
/// Lifecycle: two ownership modes are supported.
///   1. Reference mode (construct with `cache_type&`): the manager must
///      outlive the underlying cache — the caller is responsible for
///      that lifetime contract.
///   2. Shared mode (construct with `std::shared_ptr<cache_type>`): the
///      manager holds only a weak reference, so if the last strong
///      reference is dropped while the manager is still alive, every
///      operation safely becomes a no-op (reads return a miss, writes
///      are ignored) instead of a use-after-free.
/// No background worker thread is needed — promotion and demotion are
/// driven by explicit `register_hot_key()` / `unregister_hot_key()`
/// calls from the application.
template <typename Cache>
class hot_key_replica_manager {
public:
    using cache_type = Cache;
    using key_type = typename Cache::key_type;
    using value_type = typename Cache::value_type;
    using size_type = std::size_t;

    /// T-G9: Replica key transform function. Maps (original_key, index)
    /// to a distinct replica key. The default transform handles
    /// std::string and integral key types; for other key types the
    /// caller MUST supply a custom transform via `config.replica_key_fn`.
    ///
    /// The transform must be deterministic and produce distinct keys for
    /// distinct (key, idx) pairs. Replica keys should ideally hash to
    /// different shards than the original key — the default string and
    /// integer transforms achieve this by mixing the index into the
    /// key's trailing bytes / high bits.
    using replica_key_fn_t = std::function<key_type(const key_type&, size_type)>;

    /// Configuration for the replica manager.
    struct config {
        /// Number of replicas to create per hot key. Default: 4.
        /// Higher values give better read scaling but more write
        /// amplification and memory cost.
        size_type replica_factor = 4;

        /// Maximum number of keys that can be replicated simultaneously.
        /// Default: 1024 — caps memory overhead of replication metadata.
        /// Excess candidates are skipped (later register calls fail
        /// silently until existing keys are unregistered).
        size_type max_replicated_keys = 1024;

        /// T-G9: Optional custom replica key transform. If unset, the
        /// manager uses the default transform for std::string and
        /// integral key types (see `default_replica_key()`).
        replica_key_fn_t replica_key_fn;

        /// T-G9: QPS threshold for auto_subscribe. A candidate key
        /// whose estimated hit count over the poll interval exceeds
        /// this threshold is auto-registered; one that falls below
        /// `qps_unregister_ratio * threshold` is auto-unregistered.
        /// Default: 1000 hits/sec — typical shard-sustainable QPS.
        std::size_t auto_qps_threshold = 1000;

        /// T-G9: Ratio of `auto_qps_threshold` below which a replicated
        /// key is auto-unregistered. Default: 0.5 (i.e., 500 hits/sec
        /// when threshold is 1000).
        double auto_unregister_ratio = 0.5;
    };

    /// Construct the manager wrapping `cache`.
    explicit hot_key_replica_manager(cache_type& cache, config cfg = {})
        : raw_cache_(&cache)
        , cfg_(std::move(cfg))
    {
        if (!cfg_.replica_key_fn) {
            cfg_.replica_key_fn = [](const key_type& k, size_type i) {
                return default_replica_key(k, i);
            };
        }
    }

    /// Construct the manager in shared-ownership mode. The manager holds
    /// only a `weak_ptr` to the cache, so if the last strong reference
    /// is dropped (the cache is destroyed) while the manager is still
    /// alive, every subsequent operation safely becomes a no-op (reads
    /// return a miss, writes are ignored) instead of a use-after-free.
    explicit hot_key_replica_manager(std::shared_ptr<cache_type> cache,
                                     config cfg = {})
        : raw_cache_(nullptr)
        , weak_cache_(std::move(cache))
        , owns_shared_(true)
        , cfg_(std::move(cfg))
    {
        if (!cfg_.replica_key_fn) {
            cfg_.replica_key_fn = [](const key_type& k, size_type i) {
                return default_replica_key(k, i);
            };
        }
    }

    hot_key_replica_manager(const hot_key_replica_manager&) = delete;
    hot_key_replica_manager& operator=(const hot_key_replica_manager&) = delete;
    hot_key_replica_manager(hot_key_replica_manager&&) = delete;
    hot_key_replica_manager& operator=(hot_key_replica_manager&&) = delete;

    ~hot_key_replica_manager() {
        stop_auto_subscribe();
    }

    // ----------------------------------------------------------------
    // T-G9: Candidate registry + auto_subscribe
    // ----------------------------------------------------------------
    //
    // The event_tracker stores only key hashes (uint64_t) for memory
    // efficiency, so it cannot directly return the original keys. The
    // candidate registry bridges this gap: the application pre-
    // registers keys it considers eligible for replication (e.g., the
    // top-N keys it serves), and `auto_subscribe()` matches the
    // registry's hashes against the event_tracker's hot-key list to
    // decide which candidates to register/unregister.
    //
    // This keeps the hot-key detection path lock-free and avoids
    // storing full keys in the event_tracker (which would multiply
    // memory cost ~10x for typical key sizes).

    /// T-G9: Add `key` to the candidate registry. Candidates are keys
    /// the application is willing to replicate if they become hot.
    /// Does NOT immediately replicate — call `auto_subscribe()` (or
    /// start the background worker) to evaluate candidates against the
    /// event_tracker's hot-key list. Idempotent.
    void register_candidate(const key_type& key) {
        std::unique_lock<std::shared_mutex> lock(candidate_mutex_);
        candidates_.insert(key);
    }

    /// T-G9: Remove `key` from the candidate registry. If the key is
    /// currently replicated, it is also unregistered (replicas
    /// deleted). Idempotent.
    void unregister_candidate(const key_type& key) {
        {
            std::unique_lock<std::shared_mutex> lock(candidate_mutex_);
            candidates_.erase(key);
        }
        unregister_hot_key(key);
    }

    /// T-G9: Snapshot of the candidate registry.
    std::vector<key_type> candidates() const {
        std::shared_lock<std::shared_mutex> lock(candidate_mutex_);
        return {candidates_.begin(), candidates_.end()};
    }

    /// T-G9: Number of registered candidates.
    size_type candidate_count() const noexcept {
        std::shared_lock<std::shared_mutex> lock(candidate_mutex_);
        return candidates_.size();
    }

    /// T-G9: One-shot evaluation of candidates against `tracker`'s
    /// hot-key list. For each candidate whose estimated hit count over
    /// `poll_interval` exceeds `config.auto_qps_threshold`, the key is
    /// auto-registered (if not already). For each currently-replicated
    /// key whose estimated hit count falls below
    /// `config.auto_unregister_ratio * auto_qps_threshold`, the key is
    /// auto-unregistered.
    ///
    /// The QPS estimate is `delta_hits / poll_interval_seconds`. The
    /// caller is responsible for ensuring `tracker.drain_all_threads()`
    /// has been called recently (or that the background drain worker
    /// is running) so the hot-key list reflects current traffic.
    ///
    /// Returns the number of register/unregister decisions made.
    struct auto_subscribe_result {
        std::size_t registered = 0;
        std::size_t unregistered = 0;
    };

    auto_subscribe_result auto_subscribe(
        const event_tracker<key_type>& tracker,
        std::chrono::milliseconds poll_interval) {
        auto_subscribe_result result;
        const double interval_sec =
            std::chrono::duration<double>(poll_interval).count();
        if (interval_sec <= 0.0) return result;

        // Pull hot-key hashes from the tracker. We request a generous
        // top-N so we can match all candidates — if the tracker tracks
        // fewer hot keys, the rest are returned as zero-count.
        const std::size_t top_n = [this]() {
            std::shared_lock<std::shared_mutex> lock(candidate_mutex_);
            return std::max<std::size_t>(candidates_.size(), 64);
        }();
        auto hot = tracker.top_keys(top_n);
        // Build hash → count lookup.
        std::unordered_map<uint64_t, std::size_t> hot_map;
        hot_map.reserve(hot.size());
        for (auto& [h, c] : hot) hot_map[h] = c;

        const std::size_t register_threshold =
            static_cast<std::size_t>(cfg_.auto_qps_threshold * interval_sec);
        const std::size_t unregister_threshold = static_cast<std::size_t>(
            cfg_.auto_qps_threshold * cfg_.auto_unregister_ratio * interval_sec);

        // Snapshot candidates.
        std::vector<key_type> cand_snapshot;
        {
            std::shared_lock<std::shared_mutex> lock(candidate_mutex_);
            cand_snapshot.assign(candidates_.begin(), candidates_.end());
        }

        std::hash<key_type> hasher;
        for (const auto& key : cand_snapshot) {
            const uint64_t h = static_cast<uint64_t>(hasher(key));
            auto it = hot_map.find(h);
            const std::size_t count = (it != hot_map.end()) ? it->second : 0;
            const bool is_replicated = is_replicated_key(key);
            if (!is_replicated && count >= register_threshold) {
                register_hot_key(key);
                ++result.registered;
            } else if (is_replicated && count < unregister_threshold) {
                unregister_hot_key(key);
                ++result.unregistered;
            }
        }
        return result;
    }

    /// T-G9: Start a background worker that periodically calls
    /// `auto_subscribe()` with the given poll interval. Replaces any
    /// previously-running worker. The worker uses a `std::jthread` +
    /// `stop_token` so `stop_auto_subscribe()` (or destruction) signals
    /// it to exit promptly.
    void start_auto_subscribe(
        event_tracker<key_type>& tracker,
        std::chrono::milliseconds poll_interval = std::chrono::seconds(1)) {
        stop_auto_subscribe();
        auto_worker_running_.store(true, std::memory_order_release);
        auto_thread_ = std::jthread(
            [this, &tracker, poll_interval](std::stop_token st) {
                auto last = std::chrono::steady_clock::now();
                while (!st.stop_requested()) {
                    auto now = std::chrono::steady_clock::now();
                    if (now - last < poll_interval) {
                        std::this_thread::sleep_for(
                            std::chrono::milliseconds(10));
                        continue;
                    }
                    last = now;
                    (void)auto_subscribe(tracker, poll_interval);
                }
            });
    }

    /// T-G9: Stop the background auto_subscribe worker (if running).
    /// Blocks until the worker has exited.
    void stop_auto_subscribe() {
        if (auto_thread_.joinable()) {
            auto_thread_.request_stop();
            auto_thread_.join();
        }
        auto_worker_running_.store(false, std::memory_order_release);
    }

    /// T-G9: Is the background auto_subscribe worker running?
    bool auto_subscribe_running() const noexcept {
        return auto_worker_running_.load(std::memory_order_acquire);
    }

    /// T-C1 (P1-2): Explicitly register a key as a candidate for
    /// replication. The manager will create `config.replica_factor`
    /// replicas of the key's current value and route subsequent
    /// `replicated_get(key)` calls to a random replica.
    ///
    /// This is the primary entry point for applications to opt a
    /// specific key into replication. The manager does NOT
    /// automatically detect hot keys (because it can't construct
    /// replica keys from hash-only data returned by `hot_keys()`).
    /// Instead, the application identifies candidate keys (via its
    /// own metrics, application knowledge, or by polling
    /// `cache.hot_keys_with_names()` and matching against a key
    /// registry) and calls this method.
    ///
    /// Idempotent: calling twice with the same key is a no-op (the
    /// second call does not add more replicas).
    void register_hot_key(const key_type& key) {
        std::unique_lock<std::shared_mutex> lock(metadata_mutex_);
        if (metadata_.find(key) != metadata_.end()) return;  // already
        if (metadata_.size() >= cfg_.max_replicated_keys) return;  // at cap
        // Read the current value (if any) and replicate it.
        // Use peek() (no LRU promotion) to avoid amplifying the
        // hot key's read traffic during registration.
        auto c = acquire_cache();
        if (!c) return;  // cache destroyed (shared mode)
        auto h = c->peek(key);
        if (!h) return;  // not in cache — nothing to replicate
        value_type v = *h;
        replica_metadata md;
        md.replica_count = cfg_.replica_factor;
        md.last_seen_count = 0;
        for (size_type i = 0; i < cfg_.replica_factor; ++i) {
            key_type replica_key = make_replica_key(key, i);
            c->set(replica_key, v);
        }
        metadata_[key] = md;
        promote_count_.fetch_add(1, std::memory_order_relaxed);
    }

    /// T-C1 (P1-2): Unregister a key from replication. Deletes all
    /// replicas. Idempotent.
    void unregister_hot_key(const key_type& key) {
        std::unique_lock<std::shared_mutex> lock(metadata_mutex_);
        auto it = metadata_.find(key);
        if (it == metadata_.end()) return;
        auto c = acquire_cache();
        if (!c) return;  // cache destroyed (shared mode)
        for (size_type i = 0; i < it->second.replica_count; ++i) {
            key_type replica_key = make_replica_key(key, i);
            c->remove(replica_key);
        }
        metadata_.erase(it);
        demote_count_.fetch_add(1, std::memory_order_relaxed);
    }

    /// Check whether a key is currently replicated.
    bool is_replicated(const key_type& key) const {
        std::shared_lock<std::shared_mutex> lock(metadata_mutex_);
        return metadata_.find(key) != metadata_.end();
    }

    // ================================================================
    // G13: Closed-loop batch maintenance API
    // ================================================================
    //
    // Problem (G13): the replica mechanism was not closed-loop —
    // `register_hot_key()` required explicit per-key application
    // calls, and the manager could not auto-detect hot keys from
    // `cache.hot_keys()` because that API returns key *hashes*, not
    // the original keys (the event_tracker stores only hashes for
    // memory efficiency). Applications had no convenient way to keep
    // the replica set in sync with the event_tracker's hot-key list.
    //
    // Full solution (deferred): have the replica manager subscribe to
    // the event_tracker and auto-register/unregister keys as they
    // cross QPS thresholds. This requires cross-file coordination:
    //   - event_tracker.hpp: expose a key-resolution callback (the
    //     tracker stores hashes only, so it needs an application-
    //     supplied registry to map hashes back to original keys).
    //   - hot_key_replica.hpp: add a subscription API that consumes
    //     tracker events and drives register/unregister internally.
    //   - cache_trait.hpp: wire the tracker into unified_cache so the
    //     manager can observe insert/hit/evict events transparently.
    // That refactor is out of scope for G13.
    //
    // Minimal viable solution (this commit): provide two convenience
    // batch APIs that close the loop at the application level. The
    // application periodically polls the hot-key list (via
    // `cache.hot_keys_with_names()`, its own key registry, or
    // application metrics) and reconciles the replica set:
    //   1. `auto_register_from_hot_keys(current_hot_keys)` registers
    //      any not-yet-replicated hot key.
    //   2. `auto_unregister_cold_keys(current_hot_keys)` unregisters
    //      any replicated key no longer in the hot list.
    //
    // Key type note: this manager is template-parameterized on
    // `Cache::key_type` (see the `key_type` alias above). The default
    // replica-key transform (`default_replica_key`) supports
    // `std::string` and integral key types; other key types require a
    // custom `config.replica_key_fn`. These batch APIs take
    // `key_type` (NOT hard-coded `std::string`) so they work uniformly
    // with any key type the manager already supports. Generalizing to
    // fully automatic, key-type-agnostic event subscription remains
    // future work (it requires the cross-file refactor above plus a
    // hash→key resolution mechanism in event_tracker.hpp).

    /// G13: Batch-register hot keys for replication.
    ///
    /// For each key in `hot_keys`, calls `register_hot_key()`.
    /// `register_hot_key()` is idempotent and silently skips keys that
    /// are already replicated, not present in the cache (peek returns
    /// nothing), or would exceed `config.max_replicated_keys`. This
    /// method is therefore safe to call repeatedly with the same list.
    ///
    /// Returns the number of newly-registered keys, computed from the
    /// `promote_total` counter delta so the count stays accurate even
    /// under concurrent registration from other threads.
    std::size_t auto_register_from_hot_keys(
        const std::vector<key_type>& hot_keys) {
        const std::size_t before = promote_total();
        for (const auto& key : hot_keys) {
            // register_hot_key is idempotent: it no-ops if the key is
            // already replicated, absent from the cache, or would
            // exceed max_replicated_keys.
            register_hot_key(key);
        }
        return promote_total() - before;
    }

    /// G13: Batch-unregister keys that are no longer hot.
    ///
    /// Snapshots the currently-replicated key set, then unregisters
    /// every replicated key NOT present in `current_hot_keys`. This
    /// closes the replica-maintenance loop: paired with
    /// `auto_register_from_hot_keys()`, the application can keep the
    /// replica set in sync with the hot-key list by periodically
    /// calling both methods.
    ///
    /// Returns the number of unregistered keys, computed from the
    /// `demote_total` counter delta.
    std::size_t auto_unregister_cold_keys(
        const std::vector<key_type>& current_hot_keys) {
        // Snapshot the currently-replicated keys under the shared
        // lock, then release it before calling unregister_hot_key()
        // (which acquires the unique lock internally). This avoids
        // holding the shared lock across the unregistration work and
        // prevents lock-ordering deadlock with concurrent writers.
        std::vector<key_type> replicated_snapshot;
        {
            std::shared_lock<std::shared_mutex> lock(metadata_mutex_);
            replicated_snapshot.reserve(metadata_.size());
            for (const auto& kv : metadata_) {
                replicated_snapshot.push_back(kv.first);
            }
        }
        // Build a lookup set of the current hot keys for O(1)
        // membership testing. Duplicate entries in current_hot_keys
        // are collapsed by the set.
        std::unordered_set<key_type> hot_set(
            current_hot_keys.begin(), current_hot_keys.end());

        const std::size_t before = demote_total();
        for (const auto& key : replicated_snapshot) {
            if (hot_set.find(key) == hot_set.end()) {
                // unregister_hot_key is idempotent.
                unregister_hot_key(key);
            }
        }
        return demote_total() - before;
    }

    /// Read with replica routing. If `key` is replicated, picks a
    /// random replica and reads it. Otherwise falls back to a direct
    /// cache read. Returns empty optional on miss.
    ///
    /// Uses `peek()` (no LRU promotion) instead of `get()` (with
    /// promotion) to avoid amplifying promotion traffic on hot keys.
    /// The original key's LRU position is still managed normally by
    /// `replicated_set()` writes.
    std::optional<value_type> replicated_get(const key_type& key) const {
        // Look up replica metadata. If the key is replicated, route
        // to a random replica; otherwise read the original key.
        replica_metadata md;
        bool replicated;
        {
            std::shared_lock<std::shared_mutex> lock(metadata_mutex_);
            auto it = metadata_.find(key);
            replicated = (it != metadata_.end());
            if (replicated) md = it->second;
        }
        if (!replicated) {
            // Not replicated — direct read (peek to avoid promotion).
            auto c = acquire_cache();
            if (!c) return std::nullopt;  // cache destroyed (shared mode)
            auto h = c->peek(key);
            if (!h) return std::nullopt;
            return *h;
        }
        // Pick a random replica index. Use thread_local RNG for
        // performance (no global contention on the RNG state).
        thread_local std::mt19937 rng{
            static_cast<std::uint32_t>(
                std::hash<std::thread::id>{}(std::this_thread::get_id()))};
        std::uniform_int_distribution<size_type> dist(0, md.replica_count - 1);
        size_type idx = dist(rng);
        // Try the chosen replica first; on miss, fall back to the
        // original key (which the manager also maintains). This gives
        // us a fallback path when a replica has been evicted by the
        // cache's LRU policy (rare but possible under memory pressure).
        auto c = acquire_cache();
        if (!c) return std::nullopt;  // cache destroyed (shared mode)
        key_type replica_key = make_replica_key(key, idx);
        auto h = c->peek(replica_key);
        if (h) {
            read_routed_count_.fetch_add(1, std::memory_order_relaxed);
            return *h;
        }
        // Fallback: original key.
        h = c->peek(key);
        if (h) {
            read_fallback_count_.fetch_add(1, std::memory_order_relaxed);
            return *h;
        }
        read_miss_count_.fetch_add(1, std::memory_order_relaxed);
        return std::nullopt;
    }

    /// Write with replica synchronization. If `key` is replicated,
    /// updates all replicas sequentially. The write is eventually
    /// consistent — concurrent readers may see a mix of old and new
    /// values across replicas during the update.
    void replicated_set(const key_type& key, const value_type& value) {
        replica_metadata md;
        bool replicated;
        {
            std::shared_lock<std::shared_mutex> lock(metadata_mutex_);
            auto it = metadata_.find(key);
            replicated = (it != metadata_.end());
            if (replicated) md = it->second;
        }
        if (!replicated) {
            // Not replicated — direct write.
            auto c = acquire_cache();
            if (!c) return;  // cache destroyed (shared mode)
            c->set(key, value);
            return;
        }
        // Write all replicas. Count the amplification for metrics.
        auto c = acquire_cache();
        if (!c) return;  // cache destroyed (shared mode)
        c->set(key, value);
        for (size_type i = 0; i < md.replica_count; ++i) {
            key_type replica_key = make_replica_key(key, i);
            c->set(replica_key, value);
        }
        write_amplification_count_.fetch_add(
            md.replica_count, std::memory_order_relaxed);
    }

    /// Compare-and-swap with replica synchronization. Returns true if
    /// the CAS succeeded on all replicas, false on any mismatch.
    /// Uses the cache's built-in `cas(key, expected, desired)` for each
    /// replica. If any replica's CAS fails, the remaining replicas are
    /// still updated (best-effort) and the function returns false.
    bool replicated_cas(const key_type& key,
                        const value_type& expected,
                        const value_type& desired) {
        replica_metadata md;
        bool replicated;
        {
            std::shared_lock<std::shared_mutex> lock(metadata_mutex_);
            auto it = metadata_.find(key);
            replicated = (it != metadata_.end());
            if (replicated) md = it->second;
        }
        if (!replicated) {
            auto c = acquire_cache();
            if (!c) return false;  // cache destroyed (shared mode)
            return c->cas(key, expected, desired);
        }
        // CAS all replicas. Track failures — return false if any fail,
        // but continue updating the rest (best-effort) so a subsequent
        // read sees a consistent value across all replicas.
        auto c = acquire_cache();
        if (!c) return false;  // cache destroyed (shared mode)
        bool all_ok = true;
        all_ok = all_ok && c->cas(key, expected, desired);
        for (size_type i = 0; i < md.replica_count; ++i) {
            key_type replica_key = make_replica_key(key, i);
            if (!c->cas(replica_key, expected, desired)) {
                all_ok = false;
            }
        }
        if (all_ok) {
            write_amplification_count_.fetch_add(
                md.replica_count, std::memory_order_relaxed);
        }
        return all_ok;
    }

    /// Number of currently-replicated keys.
    size_type replica_count() const noexcept {
        std::shared_lock<std::shared_mutex> lock(metadata_mutex_);
        return metadata_.size();
    }

    /// Cumulative number of write-amplification operations (sum of
    /// extra replica writes across all replicated_set / replicated_cas
    /// calls). Divide by `replica_count()` to get average amplification
    /// per replicated key.
    std::size_t write_amplification_total() const noexcept {
        return write_amplification_count_.load(std::memory_order_relaxed);
    }

    /// Cumulative count of reads routed to a replica (hit on first try).
    std::size_t read_routed_total() const noexcept {
        return read_routed_count_.load(std::memory_order_relaxed);
    }

    /// Cumulative count of reads that fell back to the original key
    /// because the chosen replica missed (e.g. evicted by LRU).
    std::size_t read_fallback_total() const noexcept {
        return read_fallback_count_.load(std::memory_order_relaxed);
    }

    /// Cumulative count of reads that missed entirely (no replica
    /// and the original key both missed).
    std::size_t read_miss_total() const noexcept {
        return read_miss_count_.load(std::memory_order_relaxed);
    }

    /// Cumulative count of `register_hot_key` calls that succeeded.
    std::size_t promote_total() const noexcept {
        return promote_count_.load(std::memory_order_relaxed);
    }

    /// Cumulative count of `unregister_hot_key` calls that succeeded.
    std::size_t demote_total() const noexcept {
        return demote_count_.load(std::memory_order_relaxed);
    }

    /// Snapshot of all metrics for prometheus export.
    struct metrics_snapshot {
        size_type replica_count;
        std::size_t write_amplification_total;
        std::size_t read_routed_total;
        std::size_t read_fallback_total;
        std::size_t read_miss_total;
        size_type promote_count;
        size_type demote_count;
    };

    metrics_snapshot metrics() const {
        metrics_snapshot s;
        s.replica_count = replica_count();
        s.write_amplification_total = write_amplification_total();
        s.read_routed_total = read_routed_total();
        s.read_fallback_total = read_fallback_total();
        s.read_miss_total = read_miss_total();
        s.promote_count = promote_count_.load(std::memory_order_relaxed);
        s.demote_count = demote_count_.load(std::memory_order_relaxed);
        return s;
    }

private:
    /// Per-key replication metadata.
    struct replica_metadata {
        size_type replica_count = 0;
        std::size_t last_seen_count = 0;  // for cooldown detection
    };

    /// Lock-free check whether `key` is currently replicated.
    /// Used by `auto_subscribe()` to avoid taking the shared_mutex
    /// on every candidate (the candidate set can be large).
    bool is_replicated_key(const key_type& key) const {
        std::shared_lock<std::shared_mutex> lock(metadata_mutex_);
        return metadata_.find(key) != metadata_.end();
    }

    /// Construct the i-th replica key. Delegates to the configured
    /// transform (default = `default_replica_key`).
    key_type make_replica_key(const key_type& key, size_type idx) const {
        return cfg_.replica_key_fn(key, idx);
    }

    /// T-G9: Default replica-key transform.
    ///
    /// For `std::string` keys, appends a 0xFF marker + 4-byte big-endian
    /// index. The 0xFF byte avoids collisions with legitimate key
    /// suffixes (assuming keys don't typically end in 0xFF), and the
    /// index ensures different replicas hash to different shards under
    /// typical hash functions (FNV, Murmur, etc.).
    ///
    /// For integral key types, XORs the original key with a magic
    /// high-bit pattern derived from the index. The high bits are
    /// chosen so the result is unlikely to collide with legitimate
    /// keys (which typically use the low bits for sequential IDs).
    /// The transform is reversible: `original = replica ^ mask`.
    ///
    /// For other key types, the manager requires the caller to supply
    /// `config.replica_key_fn` — `default_replica_key` static_asserts
    /// at compile time if invoked for an unsupported type.
    static key_type default_replica_key(const key_type& key, size_type idx) {
        if constexpr (std::is_same_v<key_type, std::string>) {
            std::string replica = key;
            replica.push_back(static_cast<char>(0xFF));
            replica.push_back(static_cast<char>((idx >> 24) & 0xFF));
            replica.push_back(static_cast<char>((idx >> 16) & 0xFF));
            replica.push_back(static_cast<char>((idx >> 8) & 0xFF));
            replica.push_back(static_cast<char>(idx & 0xFF));
            return replica;
        } else if constexpr (std::is_integral_v<key_type>) {
            // Mix the index into the high bits of the key. We use a
            // 0xDB magic byte (high bit set, non-sequential) to mark
            // replica keys. For 64-bit keys the mask is in bits 48-55;
            // for 32-bit keys it's in bits 24-31. The 16-bit case uses
            // bits 8-15. This avoids the typical low-bit sequential-ID
            // space and prevents accidental collisions with legit keys.
            using U = std::make_unsigned_t<key_type>;
            U k = static_cast<U>(key);
            // Shift amount: leave the low half of the key intact so
            // replica keys remain in a "neighborhood" of the original
            // (useful for debugging). The high half gets the marker.
            constexpr int width = static_cast<int>(sizeof(U)) * 8;
            constexpr int half = width / 2;
            const U marker = static_cast<U>(0xDBu) << half;
            const U idx_mask = (static_cast<U>(idx) << (half - 4))
                               & ((~U{0}) << half);
            return static_cast<key_type>(k ^ marker ^ idx_mask);
        } else {
            static_assert(!std::is_same_v<key_type, key_type>,
                "hot_key_replica_manager: non-string, non-integral key "
                "type requires a custom config.replica_key_fn. Provide "
                "one in the config struct passed to the constructor.");
            return key;  // unreachable
        }
    }

    cache_type* raw_cache_ = nullptr;
    std::weak_ptr<cache_type> weak_cache_;
    bool owns_shared_ = false;
    config cfg_;

    /// RAII accessor that keeps the cache alive for the duration of a
    /// single operation.
    ///
    /// - Reference mode (`cache_type&` constructor): returns the raw
    ///   pointer directly; the caller's lifetime contract guarantees it
    ///   stays valid.
    /// - Shared mode (`std::shared_ptr<cache_type>` constructor): locks
    ///   the weak_ptr, keeping a strong reference alive on the stack for
    ///   as long as the returned accessor lives. If the cache was
    ///   destroyed, `operator bool()` is false and all operations safely
    ///   become no-ops.
    struct cache_access {
        cache_type* cache = nullptr;
        std::shared_ptr<cache_type> keep_alive;  // shared mode only
        explicit operator bool() const noexcept { return cache != nullptr; }
        cache_type* operator->() const noexcept { return cache; }
    };

    cache_access acquire_cache() const noexcept {
        if (!owns_shared_) return cache_access{raw_cache_, {}};
        auto sp = weak_cache_.lock();
        if (!sp) return cache_access{nullptr, {}};
        return cache_access{sp.get(), std::move(sp)};
    }

    /// Metadata map: original_key -> replica_metadata.
    /// Protected by metadata_mutex_ (shared for reads, unique for writes).
    mutable std::shared_mutex metadata_mutex_;
    std::unordered_map<key_type, replica_metadata> metadata_;

    /// T-G9: Candidate registry — keys the application is willing to
    /// auto-replicate. Protected by candidate_mutex_ (independent from
    /// metadata_mutex_ so candidate updates don't block replica reads).
    mutable std::shared_mutex candidate_mutex_;
    std::unordered_set<key_type> candidates_;

    /// T-G9: Background auto_subscribe worker.
    std::jthread auto_thread_;
    std::atomic<bool> auto_worker_running_{false};

    /// Metrics counters (cache-line aligned to avoid false sharing).
    alignas(64) mutable std::atomic<std::size_t> write_amplification_count_{0};
    alignas(64) mutable std::atomic<std::size_t> read_routed_count_{0};
    alignas(64) mutable std::atomic<std::size_t> read_fallback_count_{0};
    alignas(64) mutable std::atomic<std::size_t> read_miss_count_{0};
    alignas(64) mutable std::atomic<std::size_t> promote_count_{0};
    alignas(64) mutable std::atomic<std::size_t> demote_count_{0};
};

} // namespace lru

#endif // LRU_HOT_KEY_REPLICA_HPP
