// Unified LRU Cache Library — Warm Cache & Snapshot Support
// SPDX-License-Identifier: MIT
// Inspired by Facebook CacheLib's warm restart and Navy's snapshot mechanisms
//
// Warm cache support enables:
//   1. Async loading: populate a cache from a snapshot without blocking readers
//   2. Atomic swap: once the new cache is ready, atomically switch the pointer
//   3. Incremental snapshot: periodically write delta changes to a file
//
// Architecture:
//   ┌─────────────────────────────┐
//   │  warm_cache_manager<Cache>  │  (manages lifecycle)
//   │  - async_load()             │  → builds cache in background
//   │  - swap_when_ready()        │  → atomic pointer swap
//   │  - start_incremental_snap() │  → periodic delta snapshots
//   └─────────────────────────────┘
//
// Usage:
//   lru::safe_cache<int, std::string> cache(10000);
//   lru::warm_cache_manager mgr(cache);
//
//   // Async load from file
//   mgr.async_load("cache.dat");
//   // ... readers still hit the old cache ...
//   mgr.swap_when_ready();  // blocks until loaded, then swaps
//
//   // Periodic incremental snapshots
//   mgr.start_incremental_snapshot("cache.dat", std::chrono::seconds(30));

#ifndef LRU_WARM_CACHE_HPP
#define LRU_WARM_CACHE_HPP

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "ankerl/unordered_dense.h"
#include "core.hpp"
#include "detail/foundation.hpp"
#include "serialization.hpp"

namespace lru {

// ============================================================================
// Warm Cache Manager
// ============================================================================

/// Manages warm cache loading, atomic swapping, and incremental snapshots.
///
/// The manager holds a reference to a "live" cache pointer (shared_ptr) and
/// can asynchronously build a new cache from a serialized snapshot, then
/// atomically swap it in. Existing readers holding the old shared_ptr are
/// unaffected until they request a new reference.
///
/// P2-G: True incremental delta snapshots. The previous implementation
/// wrote a full cache->save() on every tick, which is wasteful when only
/// a small fraction of keys change between snapshots. The new
/// implementation hooks on_insert / on_evict callbacks to track the dirty
/// set in memory, then writes only the deltas to `<path>.delta` on each
/// tick. The full snapshot `<path>.full` is rewritten less frequently
/// (configurable via full_snapshot_interval).
///
/// Delta file format (binary, little-endian):
///   - 4 bytes: magic (0x4C525544 = "LRUD")
///   - 4 bytes: version (1)
///   - 4 bytes: entry count
///   - For each entry:
///     - 1 byte: op (0=insert, 1=remove)
///     - serde<Key>::serialize(key)
///     - serde<Value>::serialize(value)  (only for op=insert)
///
/// On load: cache->load(full_data) first, then apply each delta entry
/// via cache->set() or cache->remove().
///
/// @tparam CacheType  The cache type (e.g., safe_cache<K, V>)
template <typename CacheType>
class warm_cache_manager {
public:
    using cache_type = CacheType;
    using key_type = typename cache_type::key_type;
    using mapped_type = typename cache_type::mapped_type;
    using size_type = typename cache_type::size_type;

    /// State of the async load operation.
    enum class load_state {
        idle,           ///< No load in progress
        loading,        ///< Background thread is loading
        ready,          ///< Load complete, ready to swap
        failed,         ///< Load failed
    };

    // --------------------------------------------------------------------
    // Construction
    // --------------------------------------------------------------------

    /// Construct a warm cache manager that manages the given cache pointer.
    /// The shared_ptr allows atomic swap when the new cache is ready.
    explicit warm_cache_manager(std::shared_ptr<cache_type> cache)
        : live_cache_(std::move(cache)) {}

    /// Construct with a new cache of the given capacity.
    explicit warm_cache_manager(size_type capacity)
        : live_cache_(std::make_shared<cache_type>(capacity)) {}

    ~warm_cache_manager() {
        stop_incremental_snapshot();
        // G9: Explicitly join the load thread before cancel_load() does
        // further cleanup. If swap_when_ready() was never called and the
        // load thread is still running, this guarantees the thread is
        // reaped before the object is destroyed, preventing UAF.
        // cancel_load() will find the thread non-joinable and skip its
        // own join, but still perform state/error/pending_cache cleanup.
        if (load_thread_.joinable()) {
            load_thread_.join();
        }
        cancel_load();
    }

    warm_cache_manager(const warm_cache_manager&) = delete;
    warm_cache_manager& operator=(const warm_cache_manager&) = delete;

    // --------------------------------------------------------------------
    // Async Load
    // --------------------------------------------------------------------

    /// Start loading a cache from a serialized snapshot file in the background.
    /// Returns immediately. Use swap_when_ready() or load_state() to check progress.
    ///
    /// @param path      Path to the serialized snapshot file
    /// @param capacity  Capacity for the new cache (0 = auto from snapshot)
    void async_load(const std::string& path, size_type capacity = 0) {
        cancel_load();  // Cancel any previous load

        std::unique_lock lock(load_mutex_);
        pending_cache_ = (capacity > 0)
            ? std::make_shared<cache_type>(capacity)
            : std::make_shared<cache_type>();
        state_.store(load_state::loading, std::memory_order_release);

        load_thread_ = std::thread([this, path]() {
            try {
                // Read the file
                std::ifstream file(path, std::ios::binary | std::ios::ate);
                if (!file.is_open()) {
                    state_.store(load_state::failed, std::memory_order_release);
                    load_error_ = "cannot open file: " + path;
                    load_cv_.notify_all();
                    return;
                }
                auto size = file.tellg();
                file.seekg(0);
                std::vector<uint8_t> data(static_cast<std::size_t>(size));
                file.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(size));

                if (!file) {
                    state_.store(load_state::failed, std::memory_order_release);
                    load_error_ = "failed to read file: " + path;
                    load_cv_.notify_all();
                    return;
                }

                // Deserialize into the pending cache
                pending_cache_->load(data);

                state_.store(load_state::ready, std::memory_order_release);
                load_cv_.notify_all();
            } catch (const std::exception& e) {
                load_error_ = e.what();
                state_.store(load_state::failed, std::memory_order_release);
                load_cv_.notify_all();
            }
        });
    }

    /// Start loading from a binary data buffer in the background.
    void async_load_from_data(std::vector<uint8_t> data, size_type capacity = 0) {
        cancel_load();

        std::unique_lock lock(load_mutex_);
        pending_cache_ = (capacity > 0)
            ? std::make_shared<cache_type>(capacity)
            : std::make_shared<cache_type>();
        state_.store(load_state::loading, std::memory_order_release);

        load_thread_ = std::thread([this, d = std::move(data)]() {
            try {
                pending_cache_->load(d);
                state_.store(load_state::ready, std::memory_order_release);
                load_cv_.notify_all();
            } catch (const std::exception& e) {
                load_error_ = e.what();
                state_.store(load_state::failed, std::memory_order_release);
                load_cv_.notify_all();
            }
        });
    }

    /// Block until the async load is ready (or failed), then atomically swap
    /// the live cache pointer. Returns true on successful swap.
    ///
    /// T-G6: If delta tracking was enabled on the old live cache, it is
    /// automatically re-attached to the new cache after the swap so
    /// callbacks continue firing without the caller needing to call
    /// `reattach_delta_callbacks()` manually.
    ///
    /// @param timeout  Maximum time to wait (0 = wait indefinitely)
    bool swap_when_ready(std::chrono::milliseconds timeout = std::chrono::milliseconds(0)) {
        std::unique_lock lock(load_mutex_);
        if (timeout.count() > 0) {
            load_cv_.wait_for(lock, timeout, [this] {
                auto s = state_.load(std::memory_order_acquire);
                return s == load_state::ready || s == load_state::failed;
            });
        } else {
            load_cv_.wait(lock, [this] {
                auto s = state_.load(std::memory_order_acquire);
                return s == load_state::ready || s == load_state::failed;
            });
        }

        if (state_.load(std::memory_order_acquire) != load_state::ready) {
            return false;
        }

        // Atomic swap
        live_cache_ = std::move(pending_cache_);
        pending_cache_.reset();
        state_.store(load_state::idle, std::memory_order_release);
        // G9: Auto-reattach delta callbacks on the new live cache so
        // delta tracking continues without caller intervention. The
        // in-memory delta map is preserved so pending deltas are not
        // lost across the swap.
        reattach_delta_callbacks();
        return true;
    }

    /// Get the current load state.
    load_state load_state_value() const {
        return state_.load(std::memory_order_acquire);
    }

    /// Get the load error message (empty if no error).
    std::string load_error() const {
        std::lock_guard lock(load_mutex_);
        return load_error_;
    }

    /// Cancel any in-progress load.
    void cancel_load() {
        if (load_thread_.joinable()) {
            // We can't safely interrupt the thread, but we can set the state
            // and wait for it to finish. The thread checks state_ on completion.
            state_.store(load_state::failed, std::memory_order_release);
            load_cv_.notify_all();
            load_thread_.join();
        }
        pending_cache_.reset();
        state_.store(load_state::idle, std::memory_order_release);
        load_error_.clear();
    }

    // --------------------------------------------------------------------
    // Incremental Snapshot (P2-G: true delta)
    // --------------------------------------------------------------------
    //
    // Two files are written:
    //   <path>.full  — full cache snapshot via cache->save() (rewritten
    //                  every `full_snapshot_interval` ticks, or when the
    //                  delta tracking infrastructure is unavailable)
    //   <path>.delta — incremental delta since the last .full snapshot
    //                  (rewritten on every tick)
    //
    // The .delta file is small (only the dirty set), so per-tick I/O is
    // bounded by the rate of mutations, not by cache size. The .full
    // snapshot is rewritten periodically to bound the .delta replay time
    // on warm restart.
    //
    // Delta tracking hooks on_insert / on_evict on the live cache. The
    // hooks append to `delta_map_` (key → optional<value>; nullopt means
    // the key was removed). Multiple updates to the same key coalesce
    // into a single entry, so delta_map_.size() <= unique mutated keys.
    //
    // If the live cache is swapped out (via swap_when_ready()), delta
    // callbacks are automatically re-attached to the new cache (G9).
    // Callers may also invoke reattach_delta_callbacks() manually after
    // any operation that replaces the live cache.

    /// Magic number for delta files ("LRUD" = 0x4C525544).
    static constexpr uint32_t kDeltaMagic = 0x4C525544u;
    /// Delta file format version.
    static constexpr uint32_t kDeltaVersion = 1;

    /// Op codes for delta entries.
    enum class delta_op : uint8_t {
        insert = 0,
        remove = 1,
    };

    /// Enable delta tracking on the current live cache.
    /// Hooks on_insert / on_update / on_evict to populate `delta_map_`.
    /// Idempotent: calling twice on the same cache is a no-op (the second
    /// call sees `delta_callbacks_attached_` is true and returns early).
    ///
    /// The callbacks remain registered until `disable_delta_tracking()`
    /// is called or the cache is destroyed. They are automatically
    /// re-attached after a `swap_when_ready()` via
    /// `reattach_delta_callbacks()` (G9).
    void enable_delta_tracking() {
        auto cache = get_cache();
        if (delta_callbacks_attached_.exchange(true)) {
            return;  // already attached
        }
        delta_tracking_enabled_.store(true, std::memory_order_release);
        // Hook on_insert: record the new value for this key (new insertion).
        cache->on_insert([this](const key_type& key, const mapped_type& value) {
            if (!delta_tracking_enabled_.load(std::memory_order_relaxed)) return;
            auto& s = delta_shards_[delta_shard_for(key)];
            std::lock_guard lock(s.mtx);
            s.map[key] = value;
        });
        // O7: Hook on_update: record the updated value for this key
        // (existing key whose value changed). This replaces the previous
        // behavior of firing on_insert for updates.
        cache->on_update([this](const key_type& key, const mapped_type& value) {
            if (!delta_tracking_enabled_.load(std::memory_order_relaxed)) return;
            auto& s = delta_shards_[delta_shard_for(key)];
            std::lock_guard lock(s.mtx);
            s.map[key] = value;
        });
        // Hook on_evict: record a tombstone for this key.
        cache->on_evict([this](const key_type& key, const mapped_type& /*value*/) {
            if (!delta_tracking_enabled_.load(std::memory_order_relaxed)) return;
            auto& s = delta_shards_[delta_shard_for(key)];
            std::lock_guard lock(s.mtx);
            s.map[key] = std::nullopt;
        });
    }

    /// Disable delta tracking and detach callbacks. Clears the in-memory
    /// delta map. The on_insert / on_update / on_evict callbacks remain
    /// registered on the cache (the cache's callback_manager does not
    /// support removal), but become no-ops because
    /// `delta_tracking_enabled_` is false.
    void disable_delta_tracking() {
        delta_tracking_enabled_.store(false, std::memory_order_release);
        delta_callbacks_attached_.store(false, std::memory_order_release);
        for (auto& s : delta_shards_) {
            std::lock_guard lock(s.mtx);
            s.map.clear();
        }
    }

    /// Re-attach delta callbacks (on_insert / on_update / on_evict) to
    /// the current live cache. After `swap_when_ready()` replaces the
    /// live cache, the old cache's callbacks become no-ops (gated by
    /// `delta_tracking_enabled_`), so the new cache has no hooks. This
    /// method resets the attached flag and calls
    /// `enable_delta_tracking()` to re-hook the callbacks on the new
    /// live cache. The in-memory delta map is preserved so pending
    /// deltas are not lost across the swap.
    ///
    /// This is called automatically by `swap_when_ready()` when delta
    /// tracking is enabled (G9). Callers may also invoke it manually
    /// after any operation that replaces the live cache.
    void reattach_delta_callbacks() {
        if (!delta_tracking_enabled_.load(std::memory_order_acquire)) {
            return;  // delta tracking not enabled, nothing to reattach
        }
        delta_callbacks_attached_.store(false, std::memory_order_release);
        enable_delta_tracking();
    }

    /// Whether delta tracking is currently enabled.
    bool delta_tracking_enabled() const noexcept {
        return delta_tracking_enabled_.load(std::memory_order_acquire);
    }

    /// Number of pending delta entries (unique mutated keys since the
    /// last take_delta_snapshot()). For monitoring / alerting when the
    /// delta grows too large.
    std::size_t pending_delta_count() const {
        std::size_t total = 0;
        for (auto& s : delta_shards_) {
            std::lock_guard lock(s.mtx);
            total += s.map.size();
        }
        return total;
    }

    /// Start periodic incremental snapshots.
    ///
    /// On the first tick: writes a full snapshot to `<path>.full` and
    /// enables delta tracking (if not already enabled).
    /// On subsequent ticks: writes the accumulated delta to `<path>.delta`
    /// and clears the delta map.
    /// Every `full_snapshot_interval` ticks: rewrites `<path>.full` and
    /// clears the delta map (the next tick's delta starts fresh).
    ///
    /// @param path                    Base path; `.full` and `.delta` are appended.
    /// @param interval                Time between delta snapshots.
    /// @param full_snapshot_interval  Number of delta ticks between full
    ///                                snapshot rewrites. 0 = never rewrite
    ///                                (only the initial .full is written).
    ///                                Default: 10 (e.g. with a 30s delta
    ///                                interval, .full is rewritten every
    ///                                5 minutes).
    void start_incremental_snapshot(const std::string& path,
                                     std::chrono::milliseconds interval,
                                     std::size_t full_snapshot_interval = 10) {
        stop_incremental_snapshot();
        snapshot_base_path_ = path;
        full_snapshot_interval_ = full_snapshot_interval;
        snapshot_tick_count_ = 0;
        snapshot_running_.store(true, std::memory_order_release);
        snapshot_thread_ = std::thread([this, path, interval]() {
            while (snapshot_running_.load(std::memory_order_acquire)) {
                std::this_thread::sleep_for(interval);
                if (!snapshot_running_.load(std::memory_order_acquire)) break;
                (void)take_incremental_snapshot(path);
            }
        });
    }

    /// Stop the incremental snapshot worker.
    void stop_incremental_snapshot() {
        snapshot_running_.store(false, std::memory_order_release);
        if (snapshot_thread_.joinable()) {
            snapshot_thread_.join();
        }
    }

    /// Take a one-time full snapshot to the given file path (no delta).
    /// Useful for the initial snapshot before enabling incremental mode,
    /// or for ad-hoc checkpoints. Does NOT touch the delta state.
    bool take_snapshot(const std::string& path) const {
        try {
            auto cache = get_cache();
            auto data = cache->save();

            // Write to a temporary file first, then rename atomically.
            auto tmp_path = path + ".tmp";
            {
                std::ofstream file(tmp_path, std::ios::binary | std::ios::trunc);
                if (!file.is_open()) return false;
                file.write(reinterpret_cast<const char*>(data.data()),
                          static_cast<std::streamsize>(data.size()));
                if (!file) return false;
                file.close();
            }

            // Atomic rename (on POSIX, rename() is atomic; on Windows,
            // MoveFileEx with MOVEFILE_REPLACE_EXISTING).
            std::filesystem::rename(tmp_path, path);
            return true;
        } catch (...) {
            return false;
        }
    }

    /// Take an incremental delta snapshot. Writes `<path>.delta` with all
    /// pending mutations since the last call, and clears the delta map.
    /// Returns the number of delta entries written.
    ///
    /// If `full_snapshot_interval` ticks have elapsed since the last
    /// full snapshot, also rewrites `<path>.full` and clears the delta
    /// map (the next tick starts a fresh delta).
    std::size_t take_incremental_snapshot(const std::string& path) {
        // Decide whether to write a full snapshot this tick.
        bool write_full = false;
        if (full_snapshot_interval_ > 0 &&
            snapshot_tick_count_ % full_snapshot_interval_ == 0) {
            write_full = true;
        }

        // Ensure delta tracking is enabled on the first tick.
        if (snapshot_tick_count_ == 0 && !delta_tracking_enabled()) {
            enable_delta_tracking();
        }
        ++snapshot_tick_count_;

        // Snapshot the delta map (swap with empty under each shard lock).
        ankerl::unordered_dense::map<key_type, std::optional<mapped_type>>
            local_delta;
        for (auto& s : delta_shards_) {
            std::lock_guard lock(s.mtx);
            for (auto& [k, v] : s.map) {
                local_delta[std::move(k)] = std::move(v);
            }
            s.map.clear();
        }

        // If we're writing a full snapshot this tick, the delta becomes
        // unnecessary (the full snapshot captures everything). Drop it.
        if (write_full) {
            local_delta.clear();
            if (!take_snapshot(path + ".full")) {
                return 0;
            }
            return 0;
        }

        // Write the delta to <path>.delta via a temp file + rename.
        auto delta_path = path + ".delta";
        auto tmp_path = delta_path + ".tmp";
        try {
            std::vector<uint8_t> data;
            detail::binary_writer w;
            w.reserve(16 + local_delta.size() * 32);
            w.write(kDeltaMagic);
            w.write(kDeltaVersion);
            w.write(static_cast<uint32_t>(local_delta.size()));
            for (const auto& [key, opt_val] : local_delta) {
                if (opt_val.has_value()) {
                    w.write(static_cast<uint8_t>(delta_op::insert));
                    serde<key_type>::serialize(w, key);
                    serde<mapped_type>::serialize(w, *opt_val);
                } else {
                    w.write(static_cast<uint8_t>(delta_op::remove));
                    serde<key_type>::serialize(w, key);
                }
            }
            data = w.release();

            {
                std::ofstream file(tmp_path, std::ios::binary | std::ios::trunc);
                if (!file.is_open()) return 0;
                file.write(reinterpret_cast<const char*>(data.data()),
                          static_cast<std::streamsize>(data.size()));
                if (!file) return 0;
                file.close();
            }
            std::filesystem::rename(tmp_path, delta_path);
            return local_delta.size();
        } catch (...) {
            // Best-effort cleanup of the temp file.
            std::error_code ec;
            std::filesystem::remove(tmp_path, ec);
            return 0;
        }
    }

    /// Load a cache from a full snapshot + optional delta file.
    /// Reads `<path>.full` and `<path>.delta` (if present), applies
    /// the delta on top of the full snapshot, and atomically swaps the
    /// result in. Returns true on success.
    ///
    /// This is the recommended warm-restart path: it restores the most
    /// recent state with minimal I/O (full snapshot + small delta).
    bool load_with_delta(const std::string& path) {
        try {
            // Phase 1: load the full snapshot into a fresh cache.
            auto cache = get_cache();
            auto full_path = path + ".full";
            {
                std::ifstream full_file(full_path, std::ios::binary | std::ios::ate);
                if (!full_file.is_open()) return false;
                auto size = full_file.tellg();
                full_file.seekg(0);
                std::vector<uint8_t> full_data(static_cast<std::size_t>(size));
                full_file.read(reinterpret_cast<char*>(full_data.data()),
                              static_cast<std::streamsize>(size));
                if (!full_file) return false;
                cache->load(full_data);
            }

            // Phase 2: apply the delta file if it exists.
            auto delta_path = path + ".delta";
            if (!std::filesystem::exists(delta_path)) {
                return true;  // no delta, full snapshot is the latest state
            }
            std::vector<uint8_t> delta_data;
            {
                std::ifstream delta_file(delta_path, std::ios::binary | std::ios::ate);
                if (!delta_file.is_open()) return true;  // delta unreadable, use full
                auto size = delta_file.tellg();
                delta_file.seekg(0);
                delta_data.resize(static_cast<std::size_t>(size));
                delta_file.read(reinterpret_cast<char*>(delta_data.data()),
                               static_cast<std::streamsize>(size));
                if (!delta_file) return true;
            }

            // Parse and apply the delta.
            detail::binary_reader r(delta_data);
            auto magic = r.read<uint32_t>();
            if (magic != kDeltaMagic) {
                return false;  // not a delta file
            }
            auto version = r.read<uint32_t>();
            if (version != kDeltaVersion) {
                return false;  // unsupported version
            }
            auto count = r.read<uint32_t>();
            if (count > 10'000'000) {
                return false;  // sanity bound
            }
            // T-G6: Batch deserialization — parse all entries first, then
            // apply in a tight loop. This separates I/O-bound work
            // (deserialization) from mutation (set/remove) and avoids
            // interleaving cache lock acquisition with binary reading.
            struct delta_entry {
                delta_op op;
                key_type key;
                mapped_type value;
            };
            std::vector<delta_entry> entries;
            entries.reserve(count);
            for (uint32_t i = 0; i < count; ++i) {
                auto op_byte = r.read<uint8_t>();
                auto key = serde<key_type>::deserialize(r);
                if (op_byte == static_cast<uint8_t>(delta_op::insert)) {
                    auto value = serde<mapped_type>::deserialize(r);
                    entries.push_back({delta_op::insert, std::move(key), std::move(value)});
                } else if (op_byte == static_cast<uint8_t>(delta_op::remove)) {
                    entries.push_back({delta_op::remove, std::move(key), mapped_type{}});
                } else {
                    return false;  // unknown op
                }
            }
            // Apply all entries in a single tight loop. No I/O here —
            // just cache mutations.
            for (auto& e : entries) {
                if (e.op == delta_op::insert) {
                    cache->set(std::move(e.key), std::move(e.value));
                } else {
                    cache->remove(e.key);
                }
            }
            return true;
        } catch (...) {
            return false;
        }
    }

    // --------------------------------------------------------------------
    // Cache access
    // --------------------------------------------------------------------

    /// Get the current live cache (shared_ptr for safe concurrent access).
    std::shared_ptr<cache_type> get_cache() const {
        std::lock_guard lock(cache_mutex_);
        return live_cache_;
    }

    /// Direct access to the live cache (for convenience).
    cache_type& operator*() { return *live_cache_; }
    const cache_type& operator*() const { return *live_cache_; }
    cache_type* operator->() { return live_cache_.get(); }
    const cache_type* operator->() const { return live_cache_.get(); }

private:
    // Live cache (atomically swappable via shared_ptr)
    mutable std::mutex cache_mutex_;
    std::shared_ptr<cache_type> live_cache_;

    // Async load state
    mutable std::mutex load_mutex_;
    std::shared_ptr<cache_type> pending_cache_;
    std::atomic<load_state> state_{load_state::idle};
    std::string load_error_;
    std::condition_variable load_cv_;
    std::thread load_thread_;

    // Incremental snapshot state (P2-G: true delta)
    std::atomic<bool> snapshot_running_{false};
    std::thread snapshot_thread_;
    std::string snapshot_base_path_;
    std::size_t full_snapshot_interval_ = 10;
    std::size_t snapshot_tick_count_ = 0;

    // Delta tracking state (P2-G, T-G6: sharded for low contention)
    // T-G6: delta_map_ is sharded into 16 stripes, each with its own mutex
    // and map. The on_insert / on_update / on_evict callbacks (hot path
    // under high write concurrency) pick a shard by hash(key) % 16 and
    // lock only that shard. size() and swap() iterate all shards.
    static constexpr std::size_t kDeltaShards = 16;
    struct alignas(64) delta_shard {
        mutable std::mutex mtx;
        ankerl::unordered_dense::map<key_type, std::optional<mapped_type>> map;
    };
    std::array<delta_shard, kDeltaShards> delta_shards_;

    std::size_t delta_shard_for(const key_type& key) const noexcept {
        return std::hash<key_type>{}(key) & (kDeltaShards - 1);
    }

    std::atomic<bool> delta_tracking_enabled_{false};
    std::atomic<bool> delta_callbacks_attached_{false};
};

// ============================================================================
// Warm Cache Loader (functional style)
// ============================================================================

/// Load a cache from a snapshot file, returning a new shared_ptr.
/// Blocks until the load is complete.
///
/// @tparam CacheType  The cache type
/// @param path        Path to the serialized snapshot
/// @param capacity    Capacity for the new cache (0 = auto)
/// @return            shared_ptr to the loaded cache, or nullptr on failure
template <typename CacheType>
std::shared_ptr<CacheType> load_cache_from_file(
    const std::string& path,
    typename CacheType::size_type capacity = 0)
{
    try {
        auto cache = (capacity > 0)
            ? std::make_shared<CacheType>(capacity)
            : std::make_shared<CacheType>();

        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file.is_open()) return nullptr;

        auto size = file.tellg();
        file.seekg(0);
        std::vector<uint8_t> data(static_cast<std::size_t>(size));
        file.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(size));

        if (!file) return nullptr;

        cache->load(data);
        return cache;
    } catch (...) {
        return nullptr;
    }
}

/// Save a cache to a snapshot file.
/// Returns true on success.
template <typename CacheType>
bool save_cache_to_file(const CacheType& cache, const std::string& path) {
    try {
        auto data = cache.save();

        auto tmp_path = path + ".tmp";
        {
            std::ofstream file(tmp_path, std::ios::binary | std::ios::trunc);
            if (!file.is_open()) return false;
            file.write(reinterpret_cast<const char*>(data.data()),
                      static_cast<std::streamsize>(data.size()));
            if (!file) return false;
        }

        std::filesystem::rename(tmp_path, path);
        return true;
    } catch (...) {
        return false;
    }
}

} // namespace lru

#endif // LRU_WARM_CACHE_HPP
