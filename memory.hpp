// Unified LRU Cache Library - Memory Monitor & Background Mover
// Merged from: memory_monitor.hpp, background_mover.hpp
// SPDX-License-Identifier: MIT

#ifndef LRU_MEMORY_HPP
#define LRU_MEMORY_HPP

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <concepts>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <string>
#include <thread>
#include <utility>
#include <unordered_set>
#include <vector>

// Platform-specific headers (NUMA + shared memory + OS memory sampling)
#if defined(_WIN32)
#  if !defined(_WINDOWS_)
#    define WIN32_LEAN_AND_MEAN
#    define NOMINMAX
#  endif
#  include <windows.h>
#  include <psapi.h>  // GetProcessMemoryInfo for RSS sampling
#  define LRU_HAS_WIN32_NUMA 1
#  define LRU_HAS_WIN32_MEM_NOTIF 1  // QueryMemoryResourceNotification
#elif defined(__linux__)
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <fcntl.h>
#  include <unistd.h>
#  if !defined(LRU_NO_NUMA)
#    include <sys/syscall.h>
#    ifndef MPOL_BIND
#      define MPOL_BIND 2
#    endif
#    define LRU_HAS_LINUX_NUMA 1
#  endif
#  define LRU_HAS_LINUX_PROC_STATM 1  // /proc/self/statm RSS sampling
#endif

#include "core.hpp"
#include "detail/atomic_shared_ptr.hpp"
#include "detail/foundation.hpp"

namespace lru {

// ============================================================================
// Rate Limiter
// ============================================================================

/// Tracks rate of change of a value over a sliding window.
/// Detects whether a value is growing too fast and throttles accordingly.
///
/// Lock-free implementation: add_value() and throttle() use only atomic
/// operations on a power-of-2 ring buffer, eliminating mutex contention
/// on the hot path (every insert/evict calls report_memory() → add_value()).
class rate_limiter {
public:
    /// @param detect_increase  If true, detect rate of increase (memory growth).
    ///                         If false, detect rate of decrease.
    explicit rate_limiter(bool detect_increase = true)
        : detect_increase_(detect_increase) {
        // Default window: 60 samples, rounded up to power-of-2 by set_window_size
        set_window_size(60);
    }

    // Move constructor — needed because std::atomic members are non-movable.
    // Safe because moves only occur when no other thread accesses the source.
    rate_limiter(rate_limiter&& other) noexcept
        : detect_increase_(other.detect_increase_)
        , buffer_(std::move(other.buffer_))
        , window_size_(other.window_size_.load(std::memory_order_relaxed))
        , mask_(other.mask_.load(std::memory_order_relaxed))
        , head_(other.head_.load(std::memory_order_relaxed))
        , count_(other.count_.load(std::memory_order_relaxed)) {}

    rate_limiter& operator=(rate_limiter&& other) noexcept {
        if (this != &other) {
            detect_increase_ = other.detect_increase_;
            buffer_ = std::move(other.buffer_);
            window_size_.store(other.window_size_.load(std::memory_order_relaxed), std::memory_order_relaxed);
            mask_.store(other.mask_.load(std::memory_order_relaxed), std::memory_order_relaxed);
            head_.store(other.head_.load(std::memory_order_relaxed), std::memory_order_relaxed);
            count_.store(other.count_.load(std::memory_order_relaxed), std::memory_order_relaxed);
        }
        return *this;
    }

    /// Set the size of the sliding window (in samples).
    /// Not thread-safe — same guarantee as the original mutex-based version
    /// (the original also held a mutex here, but set_window_size is only
    /// called during configuration, not on the hot path).
    void set_window_size(std::size_t window_size) {
        // Round up to next power of 2 (at least 2)
        std::size_t pow2 = 2;
        while (pow2 < window_size + 1) pow2 <<= 1;
        init_buffer(pow2);
    }

    /// Add a new sample to the window. Lock-free.
    void add_value(int64_t value) {
        auto pos = head_.fetch_add(1, std::memory_order_acq_rel);
        auto mask = mask_.load(std::memory_order_acquire);
        buffer_[pos & mask].store(value, std::memory_order_release);
        count_.fetch_add(1, std::memory_order_relaxed);
    }

    /// Throttle a proposed change based on current rate of change. Lock-free.
    /// @param delta  The proposed amount to increase (positive) or decrease (negative).
    /// @return       The throttled delta (may be 0 if fully throttled).
    size_t throttle(int64_t delta) {
        auto cnt = count_.load(std::memory_order_acquire);
        if (cnt < 2) return static_cast<size_t>(std::abs(delta));

        auto h = head_.load(std::memory_order_acquire);
        auto mask = mask_.load(std::memory_order_acquire);

        // Read newest (head-1) and oldest (head-cnt) values from the ring buffer.
        // These loads are not atomic with respect to head_, but the ring buffer
        // semantics guarantee that the slots are valid (written to) — the only
        // race is with concurrent add_value(), which may shift head_ forward.
        // In the worst case we read slightly stale values, which is acceptable
        // for a rate estimator.
        int64_t newest = buffer_[(h - 1) & mask].load(std::memory_order_acquire);
        int64_t oldest = buffer_[(h - cnt) & mask].load(std::memory_order_acquire);

        int64_t total_change = newest - oldest;
        double avg_rate = static_cast<double>(total_change) /
                          static_cast<double>(cnt - 1);

        // Normalize avg_rate: for detect_increase_, we care about rate in the
        // "increasing" direction. If detect_increase_==true, positive rate = growth.
        // If detect_increase_==false, negative rate = decline (the "increasing" direction).
        double normalized_rate = detect_increase_ ? avg_rate : -avg_rate;

        if (normalized_rate <= 0.0) {
            // Rate is in the "decreasing" direction or zero — allow full delta
            return static_cast<size_t>(std::abs(delta));
        }

        if (normalized_rate < 1.0) {
            // Rate is very low — allow full delta
            return static_cast<size_t>(std::abs(delta));
        }

        // Scale delta proportionally to how much it exceeds the average rate
        double ratio = std::abs(static_cast<double>(delta) / normalized_rate);
        if (ratio <= 1.0) {
            return static_cast<size_t>(std::abs(delta));
        }

        // Throttle: reduce to the average rate
        return static_cast<size_t>(normalized_rate);
    }

    /// Reset all state. Not thread-safe with concurrent add_value().
    void reset() {
        head_.store(0, std::memory_order_release);
        count_.store(0, std::memory_order_release);
    }

    /// Current number of samples in the window.
    std::size_t size() const {
        return count_.load(std::memory_order_acquire);
    }

private:
    /// Allocate and zero-initialize the ring buffer for the given capacity.
    /// Capacity must be a power of 2.
    void init_buffer(std::size_t capacity) {
        buffer_ = std::make_unique<std::atomic<int64_t>[]>(capacity);
        // std::atomic<int64_t> is trivially default-constructible (value-initialized to 0)
        // but make_unique default-initializes; explicitly zero for clarity.
        for (std::size_t i = 0; i < capacity; ++i)
            buffer_[i].store(0, std::memory_order_relaxed);
        mask_.store(capacity - 1, std::memory_order_release);
        window_size_.store(capacity, std::memory_order_release);
        head_.store(0, std::memory_order_release);
        count_.store(0, std::memory_order_release);
    }

    bool detect_increase_;
    std::unique_ptr<std::atomic<int64_t>[]> buffer_;
    std::atomic<std::size_t> window_size_{0};
    std::atomic<std::size_t> mask_{0};
    std::atomic<std::size_t> head_{0};
    std::atomic<std::size_t> count_{0};
};

// ============================================================================
// OS Memory Sampler (spec.md P0-4)
// ============================================================================
//
// Periodically samples process RSS, cgroup memory usage, and system available
// memory from the operating system, and derives a coarse pressure level that
// `memory_monitor::should_admit()` can consult before the application-defined
// pressure callback and the internal tiered model.
//
// Design goals:
//   * Cheap: sampling runs on a background thread at a configurable interval
//     (default 1s). The hot path only reads an atomic snapshot pointer.
//   * Portable: uses Win32 `QueryMemoryResourceNotification` + `GetProcessMemoryInfo`
//     on Windows and `/proc/self/statm` + `/proc/meminfo` + cgroup v1/v2 files
//     on Linux. Unsupported platforms return `level::unknown`.
//   * Non-blocking: `latest()` returns the most recent snapshot without
//     triggering a fresh sample. Call `refresh()` to force a synchronous sample.
//   * Self-contained: does not depend on `memory_monitor`, so it can be unit
//     tested in isolation.

/// Coarse OS-level memory pressure level.
enum class os_pressure_level : uint8_t {
    unknown  = 0,  // Sampling unsupported or not yet performed
    normal   = 1,  // Plenty of headroom
    warning  = 2,  // Approaching limits — prefer to throttle
    critical = 3,  // At or beyond limits — reject new work
};

/// A single OS memory sample. All sizes are in bytes. `sampled_at` is the
/// steady_clock timepoint when the sample was taken.
struct os_memory_snapshot {
    std::chrono::steady_clock::time_point sampled_at;
    std::size_t rss_bytes = 0;             // Process resident set size
    std::size_t cgroup_usage_bytes = 0;    // cgroup memory.current (Linux only)
    std::size_t cgroup_limit_bytes = 0;    // cgroup memory.max (Linux only)
    std::size_t system_available_bytes = 0;// System-wide available memory
    std::size_t system_total_bytes = 0;    // System-wide total physical memory
    os_pressure_level level = os_pressure_level::unknown;
};

class os_memory_sampler {
public:
    struct config {
        /// Sampling interval. Defaults to 1 second.
        std::chrono::milliseconds interval{1000};

        /// RSS fraction (rss / system_total) at which `warning` kicks in.
        double rss_warning_fraction = 0.75;
        /// RSS fraction at which `critical` kicks in.
        double rss_critical_fraction = 0.90;

        /// cgroup usage fraction (usage / limit) at which `warning` kicks in.
        /// Ignored when no cgroup limit is reported.
        double cgroup_warning_fraction = 0.80;
        /// cgroup usage fraction at which `critical` kicks in.
        double cgroup_critical_fraction = 0.95;

        /// System available fraction (available / total) below which `warning`
        /// kicks in.
        double system_available_warning_fraction = 0.15;
        /// System available fraction below which `critical` kicks in.
        double system_available_critical_fraction = 0.05;
    };

    os_memory_sampler() = default;
    ~os_memory_sampler() { stop(); }

    os_memory_sampler(const os_memory_sampler&) = delete;
    os_memory_sampler& operator=(const os_memory_sampler&) = delete;
    os_memory_sampler(os_memory_sampler&&) = delete;
    os_memory_sampler& operator=(os_memory_sampler&&) = delete;

    /// Start the background sampling thread. Idempotent.
    void start() {
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            if (running_) return;
            running_ = true;
        }
        // Refresh immediately so callers see a snapshot without waiting for
        // the first interval to elapse.
        refresh();
#if defined(LRU_HAS_WIN32_MEM_NOTIF)
        setup_win32_notification();
#endif
        worker_ = std::thread([this] { run_loop(); });
    }

    /// Stop the background sampling thread. Idempotent.
    void stop() {
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            if (!running_) return;
            running_ = false;
        }
        cv_.notify_all();
#if defined(LRU_HAS_WIN32_MEM_NOTIF)
        teardown_win32_notification();
#endif
        if (worker_.joinable()) worker_.join();
    }

    /// Force a synchronous sample and update the cached snapshot.
    void refresh() {
        auto snap = sample_once();
        // T-P3-8: Store the snapshot atomically via shared_ptr.
        // Readers (should_admit via latest()) load without any lock.
        latest_.store(
            std::make_shared<os_memory_snapshot>(std::move(snap)),
            std::memory_order_release);
    }

    /// Return the most recent snapshot, or `std::nullopt` if no sample has
    /// been taken yet.
    ///
    /// T-P3-8: Lock-free — loads the atomic shared_ptr without any mutex.
    /// The returned optional is a copy of the snapshot, so it remains valid
    /// even if the background thread publishes a newer snapshot concurrently.
    std::optional<os_memory_snapshot> latest() const {
        auto ptr = latest_.load(std::memory_order_acquire);
        if (!ptr) return std::nullopt;
        return *ptr;
    }

    /// Update sampler configuration. Takes effect on the next interval.
    void configure(const config& cfg) {
        std::lock_guard<std::mutex> lock(state_mutex_);
        config_ = cfg;
    }

    bool is_running() const noexcept {
        return running_.load(std::memory_order_acquire);
    }

    const config& configuration() const noexcept { return config_; }

private:
    void run_loop() {
        while (true) {
            std::unique_lock<std::mutex> lock(state_mutex_);
            cv_.wait_for(lock, config_.interval, [this] {
                return !running_.load(std::memory_order_acquire);
            });
            if (!running_.load(std::memory_order_acquire)) return;
            // Drop the state lock before sampling (sample_once may block on
            // OS calls / file reads).
            lock.unlock();
            refresh();
        }
    }

    /// Perform a single platform-specific sample.
    os_memory_snapshot sample_once() {
        os_memory_snapshot snap;
        snap.sampled_at = std::chrono::steady_clock::now();

#if defined(_WIN32)
        sample_win32(snap);
#elif defined(__linux__)
        sample_linux(snap);
#else
        snap.level = os_pressure_level::unknown;
#endif

        snap.level = derive_level(snap);
        return snap;
    }

#if defined(_WIN32)
    void sample_win32(os_memory_snapshot& snap) {
        // Process RSS via GetProcessMemoryInfo (psapi.h).
        PROCESS_MEMORY_COUNTERS pmc{};
        if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
            snap.rss_bytes = pmc.WorkingSetSize;
        }

        // System-wide memory via GlobalMemoryStatusEx.
        MEMORYSTATUSEX msex{};
        msex.dwLength = sizeof(msex);
        if (GlobalMemoryStatusEx(&msex)) {
            snap.system_total_bytes = msex.ullTotalPhys;
            snap.system_available_bytes = msex.ullAvailPhys;
        }
    }

    void setup_win32_notification() {
#if defined(LRU_HAS_WIN32_MEM_NOTIF)
        // Create handles for both low and high memory pressure so the worker
        // thread can wake up promptly when the OS signals a transition.
        low_mem_handle_ = CreateMemoryResourceNotification(LowMemoryResourceNotification);
        high_mem_handle_ = CreateMemoryResourceNotification(HighMemoryResourceNotification);
#endif
    }

    void teardown_win32_notification() {
#if defined(LRU_HAS_WIN32_MEM_NOTIF)
        if (low_mem_handle_) { CloseHandle(low_mem_handle_); low_mem_handle_ = nullptr; }
        if (high_mem_handle_) { CloseHandle(high_mem_handle_); high_mem_handle_ = nullptr; }
#endif
    }

    void* low_mem_handle_ = nullptr;
    void* high_mem_handle_ = nullptr;
#endif // _WIN32

#if defined(__linux__)
    void sample_linux(os_memory_snapshot& snap) {
        // Process RSS from /proc/self/statm (field 1, in pages).
        std::ifstream statm("/proc/self/statm");
        if (statm) {
            unsigned long size = 0, resident = 0, shared = 0, text = 0, lib = 0, data = 0, dt = 0;
            statm >> size >> resident >> shared >> text >> lib >> data >> dt;
            if (statm) {
                long page_size = sysconf(_SC_PAGESIZE);
                if (page_size > 0) {
                    snap.rss_bytes = static_cast<std::size_t>(resident) * static_cast<std::size_t>(page_size);
                }
            }
        }

        // System-wide memory from /proc/meminfo.
        std::ifstream meminfo("/proc/meminfo");
        if (meminfo) {
            std::string key;
            std::size_t value = 0;
            std::string unit;
            while (meminfo >> key >> value >> unit) {
                if (key == "MemTotal:") {
                    snap.system_total_bytes = value * 1024ULL;
                } else if (key == "MemAvailable:") {
                    snap.system_available_bytes = value * 1024ULL;
                }
            }
        }

        // cgroup v2: /sys/fs/cgroup/memory.current and memory.max
        std::ifstream cg_current("/sys/fs/cgroup/memory.current");
        if (cg_current) {
            std::size_t v = 0;
            if (cg_current >> v) snap.cgroup_usage_bytes = v;
        }
        std::ifstream cg_max("/sys/fs/cgroup/memory.max");
        if (cg_max) {
            std::string token;
            if (cg_max >> token) {
                if (token != "max") {
                    try { snap.cgroup_limit_bytes = std::stoull(token); }
                    catch (...) { /* invalid — leave at 0 */ }
                }
            }
        }

        // cgroup v1 fallback: /sys/fs/cgroup/memory/memory.usage_in_bytes
        if (snap.cgroup_usage_bytes == 0) {
            std::ifstream v1_usage("/sys/fs/cgroup/memory/memory.usage_in_bytes");
            if (v1_usage) {
                std::size_t v = 0;
                if (v1_usage >> v) snap.cgroup_usage_bytes = v;
            }
        }
        if (snap.cgroup_limit_bytes == 0) {
            std::ifstream v1_limit("/sys/fs/cgroup/memory/memory.limit_in_bytes");
            if (v1_limit) {
                std::size_t v = 0;
                if (v1_limit >> v && v != static_cast<std::size_t>(-1)) {
                    snap.cgroup_limit_bytes = v;
                }
            }
        }
    }
#endif // __linux__

    /// Compute the pressure level from a snapshot. The most pessimistic signal
    /// wins: if any of RSS / cgroup / system available crosses the `critical`
    /// threshold, the level is `critical`; otherwise the most pessimistic
    /// `warning` wins; otherwise `normal`.
    os_pressure_level derive_level(const os_memory_snapshot& snap) const {
        bool warning = false;
        bool critical = false;

        // RSS fraction of system total.
        if (snap.system_total_bytes > 0 && snap.rss_bytes > 0) {
            double frac = static_cast<double>(snap.rss_bytes) /
                          static_cast<double>(snap.system_total_bytes);
            if (frac >= config_.rss_critical_fraction) critical = true;
            else if (frac >= config_.rss_warning_fraction) warning = true;
        }

        // cgroup usage fraction of cgroup limit.
        if (snap.cgroup_limit_bytes > 0 && snap.cgroup_usage_bytes > 0) {
            double frac = static_cast<double>(snap.cgroup_usage_bytes) /
                          static_cast<double>(snap.cgroup_limit_bytes);
            if (frac >= config_.cgroup_critical_fraction) critical = true;
            else if (frac >= config_.cgroup_warning_fraction) warning = true;
        }

        // System available fraction of system total.
        if (snap.system_total_bytes > 0) {
            double avail_frac = static_cast<double>(snap.system_available_bytes) /
                                static_cast<double>(snap.system_total_bytes);
            if (avail_frac <= config_.system_available_critical_fraction) critical = true;
            else if (avail_frac <= config_.system_available_warning_fraction) warning = true;
        }

        if (critical) return os_pressure_level::critical;
        if (warning)  return os_pressure_level::warning;
        return os_pressure_level::normal;
    }

    // T-P3-8: Lock-free snapshot storage.
    //
    // The old `std::optional<os_memory_snapshot> latest_` + `snapshot_mutex_`
    // has been replaced by an atomic shared_ptr. The background sampler thread
    // stores a new snapshot via `latest_.store(new_ptr, release)`. Readers
    // (should_admit, os_snapshot) load via `latest_.load(acquire)` without any
    // lock. The old shared_ptr stays alive via refcount until all concurrent
    // readers finish, providing lock-free, wait-free reads.
    lru::detail::atomic_shared_ptr<os_memory_snapshot> latest_;

    mutable std::mutex state_mutex_;
    std::condition_variable cv_;
    std::atomic<bool> running_{false};
    config config_;
    std::thread worker_;
};

// ============================================================================
// Memory Monitor
// ============================================================================

/// Monitors memory usage of a cache and provides admission control
/// to prevent runaway memory consumption.
///
/// Operates in two modes:
///   1. Passive monitoring: tracks current_memory and exposes rate-of-change stats
///   2. Active throttling: rejects new insertions when memory grows too fast
///
/// Tiered pressure model (aligned with CacheLib):
///   - Normal    (< throttle_fraction): accept all insertions
///   - Throttled (throttle_fraction .. critical_fraction): accept insertions,
///     but signal background eviction to free memory
///   - Critical  (>= critical_fraction): reject new insertions + aggressive
///     background eviction
class memory_monitor {
public:
    /// Tiered memory pressure level, aligned with CacheLib's approach.
    enum class pressure_level : uint8_t {
        normal    = 0,  // < throttle_fraction occupancy
        throttled = 1,  // throttle_fraction .. critical_fraction occupancy
        critical  = 2,  // >= critical_fraction occupancy
    };

    /// Configuration for the memory monitor.
    struct config {
        /// Maximum memory budget in bytes (0 = unlimited).
        std::atomic<std::size_t> max_memory_bytes{0};

        /// Window size for rate-of-change calculation (in samples).
        /// Larger values smooth out short-term spikes.
        std::atomic<std::size_t> rate_window_size{60};

        /// Memory growth rate threshold (bytes per sample) above which
        /// throttling activates. 0 = never throttle based on rate.
        std::atomic<std::size_t> max_growth_rate_bytes{0};

        /// High watermark: when current_memory exceeds this fraction of max,
        /// aggressive throttling kicks in [0.0, 1.0].
        std::atomic<double> high_watermark_fraction{0.90};

        /// Critical watermark: when exceeded, all new insertions are rejected [0.0, 1.0].
        std::atomic<double> critical_watermark_fraction{0.98};

        /// Low watermark: throttling is released when memory drops below this [0.0, 1.0].
        std::atomic<double> low_watermark_fraction{0.75};

        /// Start throttling at this occupancy fraction [0.0, 1.0].
        /// Below this, all insertions are accepted (pressure_level::normal).
        std::atomic<double> throttle_fraction{0.8};

        /// Critical pressure at this occupancy fraction [0.0, 1.0].
        /// At or above this, new insertions are rejected and aggressive
        /// background eviction is triggered (pressure_level::critical).
        std::atomic<double> critical_fraction{0.95};

        config() = default;

        // Move constructor — needed because std::atomic members are non-movable.
        config(config&& other) noexcept {
            max_memory_bytes.store(other.max_memory_bytes.load(std::memory_order_relaxed), std::memory_order_relaxed);
            rate_window_size.store(other.rate_window_size.load(std::memory_order_relaxed), std::memory_order_relaxed);
            max_growth_rate_bytes.store(other.max_growth_rate_bytes.load(std::memory_order_relaxed), std::memory_order_relaxed);
            high_watermark_fraction.store(other.high_watermark_fraction.load(std::memory_order_relaxed), std::memory_order_relaxed);
            critical_watermark_fraction.store(other.critical_watermark_fraction.load(std::memory_order_relaxed), std::memory_order_relaxed);
            low_watermark_fraction.store(other.low_watermark_fraction.load(std::memory_order_relaxed), std::memory_order_relaxed);
            throttle_fraction.store(other.throttle_fraction.load(std::memory_order_relaxed), std::memory_order_relaxed);
            critical_fraction.store(other.critical_fraction.load(std::memory_order_relaxed), std::memory_order_relaxed);
        }

        config& operator=(config&& other) noexcept {
            if (this != &other) {
                max_memory_bytes.store(other.max_memory_bytes.load(std::memory_order_relaxed), std::memory_order_relaxed);
                rate_window_size.store(other.rate_window_size.load(std::memory_order_relaxed), std::memory_order_relaxed);
                max_growth_rate_bytes.store(other.max_growth_rate_bytes.load(std::memory_order_relaxed), std::memory_order_relaxed);
                high_watermark_fraction.store(other.high_watermark_fraction.load(std::memory_order_relaxed), std::memory_order_relaxed);
                critical_watermark_fraction.store(other.critical_watermark_fraction.load(std::memory_order_relaxed), std::memory_order_relaxed);
                low_watermark_fraction.store(other.low_watermark_fraction.load(std::memory_order_relaxed), std::memory_order_relaxed);
                throttle_fraction.store(other.throttle_fraction.load(std::memory_order_relaxed), std::memory_order_relaxed);
                critical_fraction.store(other.critical_fraction.load(std::memory_order_relaxed), std::memory_order_relaxed);
            }
            return *this;
        }
    };

    memory_monitor()
        : memory_monitor(config{}) {}

    explicit memory_monitor(const config& cfg)
        : growth_limiter_(/*detect increase*/ true) {
        config_.max_memory_bytes.store(cfg.max_memory_bytes.load(std::memory_order_relaxed),
                                       std::memory_order_relaxed);
        config_.rate_window_size.store(cfg.rate_window_size.load(std::memory_order_relaxed),
                                       std::memory_order_relaxed);
        config_.max_growth_rate_bytes.store(cfg.max_growth_rate_bytes.load(std::memory_order_relaxed),
                                            std::memory_order_relaxed);
        config_.high_watermark_fraction.store(cfg.high_watermark_fraction.load(std::memory_order_relaxed),
                                              std::memory_order_relaxed);
        config_.critical_watermark_fraction.store(cfg.critical_watermark_fraction.load(std::memory_order_relaxed),
                                                  std::memory_order_relaxed);
        config_.low_watermark_fraction.store(cfg.low_watermark_fraction.load(std::memory_order_relaxed),
                                             std::memory_order_relaxed);
        config_.throttle_fraction.store(cfg.throttle_fraction.load(std::memory_order_relaxed),
                                        std::memory_order_relaxed);
        config_.critical_fraction.store(cfg.critical_fraction.load(std::memory_order_relaxed),
                                        std::memory_order_relaxed);
        growth_limiter_.set_window_size(cfg.rate_window_size.load(std::memory_order_relaxed));
    }

    // Move constructor — needed because memory_monitor has non-movable
    // members (std::atomic, rate_limiter). Safe because moves only occur
    // when no other thread accesses the source (e.g., NRVO of returned
    // unified_cache instances from factory-style helpers).
    memory_monitor(memory_monitor&& other) noexcept
        : config_(std::move(other.config_))
        , current_memory_(other.current_memory_.load(std::memory_order_relaxed))
        , growth_rate_exceeded_(other.growth_rate_exceeded_.load(std::memory_order_relaxed))
        , is_throttled_(other.is_throttled_.load(std::memory_order_relaxed))
        , bg_eviction_requested_(other.bg_eviction_requested_.load(std::memory_order_relaxed))
        , pressure_normal_count_(other.pressure_normal_count_.load(std::memory_order_relaxed))
        , pressure_throttled_count_(other.pressure_throttled_count_.load(std::memory_order_relaxed))
        , pressure_critical_count_(other.pressure_critical_count_.load(std::memory_order_relaxed))
        , growth_limiter_(std::move(other.growth_limiter_))
        , pressure_cb_(std::move(other.pressure_cb_))
        , os_sampler_(std::move(other.os_sampler_)) {}

    memory_monitor& operator=(memory_monitor&& other) noexcept {
        if (this != &other) {
            config_ = std::move(other.config_);
            current_memory_.store(other.current_memory_.load(std::memory_order_relaxed), std::memory_order_relaxed);
            growth_rate_exceeded_.store(other.growth_rate_exceeded_.load(std::memory_order_relaxed), std::memory_order_relaxed);
            is_throttled_.store(other.is_throttled_.load(std::memory_order_relaxed), std::memory_order_relaxed);
            bg_eviction_requested_.store(other.bg_eviction_requested_.load(std::memory_order_relaxed), std::memory_order_relaxed);
            pressure_normal_count_.store(other.pressure_normal_count_.load(std::memory_order_relaxed), std::memory_order_relaxed);
            pressure_throttled_count_.store(other.pressure_throttled_count_.load(std::memory_order_relaxed), std::memory_order_relaxed);
            pressure_critical_count_.store(other.pressure_critical_count_.load(std::memory_order_relaxed), std::memory_order_relaxed);
            growth_limiter_ = std::move(other.growth_limiter_);
            pressure_cb_ = std::move(other.pressure_cb_);
            os_sampler_ = std::move(other.os_sampler_);
        }
        return *this;
    }

    // --------------------------------------------------------------------
    // State queries
    // --------------------------------------------------------------------

    /// Current throttling state.
    enum class state {
        normal,     // No throttling — all insertions allowed
        throttled,  // Growth rate exceeds threshold — some insertions may be rejected
        high,       // Above high watermark — aggressive throttling
        critical,   // Above critical watermark — all insertions rejected
    };

    /// Get current state based on last reported memory.
    state current_state() const noexcept {
        auto mem = current_memory_.load(std::memory_order_acquire);
        auto max = config_.max_memory_bytes.load(std::memory_order_relaxed);

        if (max == 0) return state::normal;

        double fraction = static_cast<double>(mem) / static_cast<double>(max);

        if (fraction >= config_.critical_watermark_fraction.load(std::memory_order_relaxed)) return state::critical;
        if (fraction >= config_.high_watermark_fraction.load(std::memory_order_relaxed)) return state::high;
        if (growth_rate_exceeded_.load(std::memory_order_relaxed)) return state::throttled;
        return state::normal;
    }

    /// Human-readable state name.
    static std::string state_name(state s) {
        switch (s) {
            case state::normal:    return "normal";
            case state::throttled: return "throttled";
            case state::high:      return "high";
            case state::critical:  return "critical";
        }
        return "unknown";
    }

    // --------------------------------------------------------------------
    // Tiered pressure model
    // --------------------------------------------------------------------

    /// Compute the current memory pressure level based on occupancy fraction.
    /// Uses throttle_fraction and critical_fraction from config.
    pressure_level current_pressure() const noexcept {
        auto mem = current_memory_.load(std::memory_order_acquire);
        auto max = config_.max_memory_bytes.load(std::memory_order_relaxed);

        if (max == 0) return pressure_level::normal;

        double fraction = static_cast<double>(mem) / static_cast<double>(max);

        if (fraction >= config_.critical_fraction.load(std::memory_order_relaxed))
            return pressure_level::critical;
        if (fraction >= config_.throttle_fraction.load(std::memory_order_relaxed))
            return pressure_level::throttled;
        return pressure_level::normal;
    }

    /// Human-readable pressure level name.
    static std::string pressure_level_name(pressure_level p) {
        switch (p) {
            case pressure_level::normal:    return "normal";
            case pressure_level::throttled: return "throttled";
            case pressure_level::critical:  return "critical";
        }
        return "unknown";
    }

    /// Whether the background evictor should be actively evicting.
    /// Returns true when pressure is throttled or critical.
    bool should_trigger_background_eviction() const noexcept {
        auto p = current_pressure();
        return p == pressure_level::throttled || p == pressure_level::critical;
    }

    // --------------------------------------------------------------------
    // Admission control
    // --------------------------------------------------------------------
    //
    // Verdict returned by an application-provided pressure callback. The
    // callback is invoked at the top of `should_admit()` and may short-circuit
    // the entire admission decision.
    enum class pressure_verdict : uint8_t {
        admit    = 0,  // Defer to the internal tiered model
        throttle = 1,  // Defer to the internal tiered model (sets throttle flag if warranted)
        reject   = 2,  // Hard reject — caller must not insert
    };

    using pressure_callback = std::function<pressure_verdict(std::size_t delta_bytes)>;

    /// Check if an insertion of `delta_bytes` should be allowed.
    /// @param delta_bytes  The expected increase in memory if the insertion succeeds.
    /// @return            true if the insertion should proceed.
    ///
    /// Behavior by pressure level:
    ///   normal    — admit, clear throttle flag, clear background eviction flag
    ///   throttled — admit, set throttle flag, set background eviction flag
    ///   critical  — reject with probability proportional to how far above
    ///               critical threshold; set throttle + background eviction flags
    bool should_admit(std::size_t delta_bytes = 0) {
        // -----------------------------------------------------------------
        // Tier 1: Application-provided pressure callback takes precedence.
        // This lets production callers integrate external signals (K8s
        // pressure, custom cgroup readers, downstream back-pressure, etc.)
        // before any internal logic runs.
        //
        // T-P3-8: The callback is loaded atomically via shared_ptr — no
        // mutex is acquired. The shared_ptr keeps the callback alive until
        // the call completes, even if another thread replaces the callback
        // concurrently.
        // -----------------------------------------------------------------
        auto cb_ptr = pressure_cb_.load(std::memory_order_acquire);
        if (cb_ptr && *cb_ptr) {
            auto verdict = (*cb_ptr)(delta_bytes);
            if (verdict == pressure_verdict::reject) {
                is_throttled_.store(true, std::memory_order_release);
                bg_eviction_requested_.store(true, std::memory_order_release);
                track_pressure(pressure_level::critical);
                return false;
            }
            // `admit` and `throttle` both fall through to the internal model
            // so the monitor can still set throttle/background-eviction flags
            // based on its own observations. `throttle` is treated identically
            // to `admit` here because the internal tiered model already sets
            // the throttle flag when warranted; the callback's job is to veto
            // (reject) or defer (admit/throttle).
        }

        // -----------------------------------------------------------------
        // Tier 2: OS-level pressure check (spec.md P0-4).
        // The OS sampler runs on a background thread and exposes the most
        // recent snapshot. T-P3-8: the snapshot is now loaded lock-free via
        // an atomic shared_ptr (no mutex on the admission hot path).
        // When the OS reports `critical` pressure, reject almost all new
        // insertions (0.1% admit probability to avoid complete starvation
        // under sustained pressure). When it reports `warning`, set the
        // throttle and background-eviction flags but still admit.
        // -----------------------------------------------------------------
        os_memory_sampler* sampler = os_sampler_.get();
        if (sampler) {
            auto snap = sampler->latest();
            if (snap && snap->level == os_pressure_level::critical) {
                is_throttled_.store(true, std::memory_order_release);
                bg_eviction_requested_.store(true, std::memory_order_release);
                track_pressure(pressure_level::critical);
                // 0.1% admit probability under OS-level critical pressure.
                return accept_with_probability(0.001);
            }
            if (snap && snap->level == os_pressure_level::warning) {
                is_throttled_.store(true, std::memory_order_release);
                bg_eviction_requested_.store(true, std::memory_order_release);
                // Fall through to the internal model; it may still admit.
            }
        }

        // -----------------------------------------------------------------
        // Tier 3: Internal tiered pressure model (legacy behavior).
        // -----------------------------------------------------------------
        auto mem = current_memory_.load(std::memory_order_acquire);
        auto max = config_.max_memory_bytes.load(std::memory_order_relaxed);

        if (max == 0) {
            // Even with no internal budget, preserve any throttle flag set
            // by Tier 2 (warning) rather than clobbering it. The throttle
            // flag is only cleared when we actually transition to normal.
            if (!is_throttled_.load(std::memory_order_acquire)) {
                bg_eviction_requested_.store(false, std::memory_order_release);
                track_pressure(pressure_level::normal);
            }
            return true; // unlimited internal budget
        }

        double fraction = static_cast<double>(mem + delta_bytes) / static_cast<double>(max);
        auto crit_frac = config_.critical_fraction.load(std::memory_order_relaxed);
        auto throttle_frac = config_.throttle_fraction.load(std::memory_order_relaxed);
        auto crit_wm = config_.critical_watermark_fraction.load(std::memory_order_relaxed);
        auto high_wm = config_.high_watermark_fraction.load(std::memory_order_relaxed);

        // Use the lower of critical_fraction and critical_watermark_fraction
        // as the effective critical threshold, so that both the tiered model
        // and the legacy watermark model are respected.
        double effective_crit = std::min(crit_frac, crit_wm);
        // Similarly, use the lower of throttle_fraction and high_watermark_fraction
        // as the effective throttle threshold.
        double effective_throttle = std::min(throttle_frac, high_wm);

        // Critical: reject new insertions + aggressive background eviction.
        // Uses probabilistic rejection: the further above the critical threshold,
        // the higher the rejection probability. At exactly effective_crit, the
        // acceptance probability is zero (hard reject). As overflow increases
        // toward full capacity, rejection remains total.
        if (fraction >= effective_crit) {
            is_throttled_.store(true, std::memory_order_release);
            bg_eviction_requested_.store(true, std::memory_order_release);
            track_pressure(pressure_level::critical);
            // Acceptance probability: starts at 0 at effective_crit, increases
            // slightly as we move past critical (allowing a tiny fraction of
            // insertions to avoid complete starvation), then drops back to 0
            // as we approach full capacity. This models CacheLib's approach of
            // probabilistic admission under pressure.
            // overflow: 0.0 at effective_crit, 1.0 at full capacity
            double overflow = (fraction - effective_crit) / (1.0 - effective_crit);
            // Small hump: accept_prob peaks at overflow=0.5 with value
            // 0.25*(1-effective_crit)^2, then decays to 0.
            // For effective_crit=0.90: peak ≈ 0.0025 (0.25%), effectively
            // rejecting nearly all insertions.
            double accept_prob = 4.0 * overflow * (1.0 - overflow) * (1.0 - effective_crit) * (1.0 - effective_crit);
            return accept_with_probability(accept_prob);
        }

        // Throttled: admit but signal throttle + background eviction
        if (fraction >= effective_throttle) {
            is_throttled_.store(true, std::memory_order_release);
            bg_eviction_requested_.store(true, std::memory_order_release);
            track_pressure(pressure_level::throttled);
            return true;
        }

        // Rate-based throttling: admit but signal throttle (accelerate eviction)
        if (config_.max_growth_rate_bytes.load(std::memory_order_relaxed) > 0 &&
            growth_rate_exceeded_.load(std::memory_order_relaxed)) {
            is_throttled_.store(true, std::memory_order_release);
            bg_eviction_requested_.store(true, std::memory_order_release);
            track_pressure(pressure_level::throttled);
            return true;
        }

        // Normal: clear throttle + background eviction flags
        is_throttled_.store(false, std::memory_order_release);
        bg_eviction_requested_.store(false, std::memory_order_release);
        track_pressure(pressure_level::normal);
        return true;
    }

    /// Whether the monitor is in a throttled state, signalling that
    /// accelerated eviction should be active.
    bool is_throttled() const noexcept {
        return is_throttled_.load(std::memory_order_acquire);
    }

    // --------------------------------------------------------------------
    // Application-provided pressure callback + OS sampler API (spec.md P0-4)
    // --------------------------------------------------------------------
    //
    // `pressure_verdict` and `pressure_callback` are declared above, next to
    // `should_admit()`, because the admission path references them directly.

    /// Install (or replace) an application-provided pressure callback. Pass
    /// `nullptr` to remove a previously-installed callback. The callback is
    /// invoked synchronously on the thread that calls `should_admit()`, so it
    /// must be cheap (no IO, no locks on hot paths).
    ///
    /// T-P3-8: The callback is stored atomically via shared_ptr. This method
    /// is lock-free — it atomically swaps in the new shared_ptr. Concurrent
    /// callers of should_admit() that already loaded the old shared_ptr
    /// continue invoking the old callback until they finish.
    void set_memory_pressure_callback(pressure_callback cb) {
        // An empty std::function maps to a null shared_ptr (no callback).
        pressure_cb_.store(
            cb ? std::make_shared<pressure_callback>(std::move(cb)) : nullptr,
            std::memory_order_release);
    }

    /// T-P3-8: Return the current callback as a shared_ptr (lock-free).
    /// Returns nullptr if no callback is installed.
    std::shared_ptr<pressure_callback> memory_pressure_callback() const noexcept {
        return pressure_cb_.load(std::memory_order_acquire);
    }

    /// Start the OS memory sampler with the given configuration. The sampler
    /// runs on its own background thread and is automatically stopped in the
    /// monitor's destructor.
    void start_os_sampling(const os_memory_sampler::config& cfg = os_memory_sampler::config{}) {
        if (!os_sampler_) os_sampler_ = std::make_unique<os_memory_sampler>();
        os_sampler_->configure(cfg);
        os_sampler_->start();
    }

    /// Stop the OS memory sampler if it is running. The cached snapshot
    /// remains queryable via `os_snapshot()` until the monitor is destroyed.
    void stop_os_sampling() {
        if (os_sampler_) os_sampler_->stop();
    }

    bool os_sampling_running() const noexcept {
        return os_sampler_ && os_sampler_->is_running();
    }

    os_memory_sampler* os_sampler() noexcept { return os_sampler_.get(); }
    const os_memory_sampler* os_sampler() const noexcept { return os_sampler_.get(); }

    std::optional<os_memory_snapshot> os_snapshot() const {
        return os_sampler_ ? os_sampler_->latest() : std::nullopt;
    }

    // --------------------------------------------------------------------
    // Reporting
    // --------------------------------------------------------------------

    /// Report the current memory usage (call periodically, e.g., after each insert/evict).
    /// @param current_memory_bytes  Total memory used by the cache right now.
    void report_memory(std::size_t current_memory_bytes) {
        current_memory_.store(current_memory_bytes, std::memory_order_release);

        // Feed into rate limiter
        growth_limiter_.add_value(static_cast<int64_t>(current_memory_bytes));

        // Check if growth rate exceeds threshold
        if (config_.max_growth_rate_bytes.load(std::memory_order_relaxed) > 0) {
            auto throttled_delta = growth_limiter_.throttle(
                static_cast<int64_t>(config_.max_growth_rate_bytes.load(std::memory_order_relaxed)));
            growth_rate_exceeded_.store(
                throttled_delta < config_.max_growth_rate_bytes.load(std::memory_order_relaxed),
                std::memory_order_release);
        }
    }

    /// Report a successful insertion of `delta_bytes`.
    void report_insert(std::size_t delta_bytes) {
        current_memory_.fetch_add(delta_bytes, std::memory_order_release);
    }

    /// Report an eviction freeing `delta_bytes`.
    void report_evict(std::size_t delta_bytes) {
        current_memory_.fetch_sub(delta_bytes, std::memory_order_release);
    }

    // --------------------------------------------------------------------
    // Statistics
    // --------------------------------------------------------------------

    struct stats {
        std::size_t current_memory_bytes = 0;
        std::size_t max_memory_bytes = 0;
        double occupancy_fraction = 0.0;
        double growth_rate_bytes_per_sample = 0.0;
        state current_state = state::normal;
        pressure_level current_pressure = pressure_level::normal;
        std::size_t pressure_normal_count = 0;
        std::size_t pressure_throttled_count = 0;
        std::size_t pressure_critical_count = 0;
    };

    stats get_stats() const {
        stats s;
        s.current_memory_bytes = current_memory_.load(std::memory_order_acquire);
        s.max_memory_bytes = config_.max_memory_bytes.load(std::memory_order_relaxed);
        if (s.max_memory_bytes > 0) {
            s.occupancy_fraction = static_cast<double>(s.current_memory_bytes) /
                                   static_cast<double>(s.max_memory_bytes);
        }
        s.current_state = current_state();
        s.current_pressure = current_pressure();
        s.pressure_normal_count = pressure_normal_count_.load(std::memory_order_relaxed);
        s.pressure_throttled_count = pressure_throttled_count_.load(std::memory_order_relaxed);
        s.pressure_critical_count = pressure_critical_count_.load(std::memory_order_relaxed);
        return s;
    }

    // --------------------------------------------------------------------
    // Configuration
    // --------------------------------------------------------------------

    void set_max_memory(std::size_t bytes) {
        config_.max_memory_bytes.store(bytes, std::memory_order_relaxed);
    }

    void set_high_watermark(double fraction) {
        config_.high_watermark_fraction.store(std::clamp(fraction, 0.0, 1.0), std::memory_order_relaxed);
    }

    void set_critical_watermark(double fraction) {
        config_.critical_watermark_fraction.store(std::clamp(fraction, 0.0, 1.0), std::memory_order_relaxed);
    }

    void reset() {
        current_memory_.store(0, std::memory_order_release);
        growth_rate_exceeded_.store(false, std::memory_order_release);
        is_throttled_.store(false, std::memory_order_release);
        bg_eviction_requested_.store(false, std::memory_order_release);
        pressure_normal_count_.store(0, std::memory_order_release);
        pressure_throttled_count_.store(0, std::memory_order_release);
        pressure_critical_count_.store(0, std::memory_order_release);
        growth_limiter_.reset();
    }

    /// Reconfigure the monitor from a new config and reset its internal state.
    /// Existing reports are discarded; the monitor starts fresh with the new
    /// configuration.
    void configure(const config& cfg) {
        config_.max_memory_bytes.store(cfg.max_memory_bytes.load(std::memory_order_relaxed),
                                       std::memory_order_relaxed);
        config_.rate_window_size.store(cfg.rate_window_size.load(std::memory_order_relaxed),
                                       std::memory_order_relaxed);
        config_.max_growth_rate_bytes.store(cfg.max_growth_rate_bytes.load(std::memory_order_relaxed),
                                            std::memory_order_relaxed);
        config_.high_watermark_fraction.store(cfg.high_watermark_fraction.load(std::memory_order_relaxed),
                                              std::memory_order_relaxed);
        config_.critical_watermark_fraction.store(cfg.critical_watermark_fraction.load(std::memory_order_relaxed),
                                                  std::memory_order_relaxed);
        config_.low_watermark_fraction.store(cfg.low_watermark_fraction.load(std::memory_order_relaxed),
                                             std::memory_order_relaxed);
        config_.throttle_fraction.store(cfg.throttle_fraction.load(std::memory_order_relaxed),
                                        std::memory_order_relaxed);
        config_.critical_fraction.store(cfg.critical_fraction.load(std::memory_order_relaxed),
                                        std::memory_order_relaxed);
        growth_limiter_.set_window_size(cfg.rate_window_size.load(std::memory_order_relaxed));
        reset();
    }

    /// Whether the monitor is configured to do any work.
    /// If false, should_admit() always returns true and report_memory() is a
    /// no-op, so callers can skip the overhead entirely.
    ///
    /// T-P3-8: The callback check is now a lock-free atomic load.
    bool active() const noexcept {
        return config_.max_memory_bytes.load(std::memory_order_relaxed) != 0 ||
               config_.max_growth_rate_bytes.load(std::memory_order_relaxed) != 0 ||
               pressure_cb_.load(std::memory_order_relaxed) != nullptr ||
               (os_sampler_ != nullptr && os_sampler_->is_running());
    }

private:
    /// Accept with a given probability (0.0 = never, 1.0 = always).
    /// Uses xoshiro128** PRNG — better statistical quality than LCG with
    /// comparable speed. Each thread gets its own state (no contention).
    bool accept_with_probability(double probability) {
        if (probability >= 1.0) return true;
        if (probability <= 0.0) return false;

        static thread_local auto s_init = [] {
            std::array<uint32_t, 4> seed;
            std::random_device rd;
            seed[0] = rd();
            seed[1] = rd();
            seed[2] = rd();
            seed[3] = rd();
            // Mix in thread-local address for additional diversity
            auto addr = reinterpret_cast<std::uintptr_t>(&seed);
            seed[0] ^= static_cast<uint32_t>(addr);
            seed[1] ^= static_cast<uint32_t>(addr >> 32);
            // Ensure no zero state (xoshiro requires non-zero)
            for (auto& v : seed) {
                if (v == 0) v = 0xdeadbeef;
            }
            return seed;
        }();
        static thread_local uint32_t* s = s_init.data();

        auto rotl = [](uint32_t x, int k) -> uint32_t { return (x << k) | (x >> (32 - k)); };

        uint32_t result = rotl(s[1] * 5, 7) * 9;
        uint32_t t = s[1] << 9;

        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];

        s[2] ^= t;
        s[3] = rotl(s[3], 11);

        double r = static_cast<double>(result >> 8) / static_cast<double>(1U << 24);
        return r < probability;
    }

    /// Track pressure level transitions for statistics.
    void track_pressure(pressure_level level) noexcept {
        switch (level) {
            case pressure_level::normal:
                pressure_normal_count_.fetch_add(1, std::memory_order_relaxed);
                break;
            case pressure_level::throttled:
                pressure_throttled_count_.fetch_add(1, std::memory_order_relaxed);
                break;
            case pressure_level::critical:
                pressure_critical_count_.fetch_add(1, std::memory_order_relaxed);
                break;
        }
    }

    config config_;
    std::atomic<std::size_t> current_memory_{0};
    std::atomic<bool> growth_rate_exceeded_{false};
    std::atomic<bool> is_throttled_{false};
    std::atomic<bool> bg_eviction_requested_{false};
    std::atomic<std::size_t> pressure_normal_count_{0};
    std::atomic<std::size_t> pressure_throttled_count_{0};
    std::atomic<std::size_t> pressure_critical_count_{0};
    // low_watermark_fraction reserved for future hysteresis implementation
    rate_limiter growth_limiter_;

    // P0-4: Application-provided pressure callback + OS memory sampler.
    //
    // T-P3-8: Lock-free callback storage.
    //
    // std::function is not atomically copyable, so the callback is wrapped in
    // a shared_ptr and stored atomically. Readers (should_admit) load the
    // shared_ptr without any mutex — the admission path is now fully
    // lock-free on Tier 1. Writers (set_memory_pressure_callback) store a
    // new shared_ptr atomically. The old shared_ptr stays alive via refcount
    // until all concurrent readers finish invoking the old callback.
    lru::detail::atomic_shared_ptr<pressure_callback> pressure_cb_;
    std::unique_ptr<os_memory_sampler> os_sampler_;
};

// ============================================================================
// Memory Guard (RAII)
// ============================================================================

/// RAII guard for a throttled insertion attempt.
/// On construction, checks if insertion should be allowed.
/// On successful commit(), the memory is accounted for.
/// On destruction without commit(), the attempt is discarded.
class memory_guard {
public:
    memory_guard(memory_monitor& monitor, std::size_t delta_bytes)
        : monitor_(&monitor)
        , delta_bytes_(delta_bytes)
        , admitted_(monitor.should_admit(delta_bytes)) {}

    memory_guard(memory_guard&& other) noexcept
        : monitor_(other.monitor_)
        , delta_bytes_(other.delta_bytes_)
        , admitted_(other.admitted_)
        , committed_(other.committed_) {
        other.monitor_ = nullptr;
        other.committed_ = true; // prevent double-commit
    }

    memory_guard& operator=(memory_guard&&) = delete;
    memory_guard(const memory_guard&) = delete;
    memory_guard& operator=(const memory_guard&) = delete;

    ~memory_guard() {
        if (admitted_ && !committed_ && monitor_) {
            // Rollback: the insert was allowed but didn't happen
            // Nothing to do — we only account on commit
        }
    }

    /// Whether the insertion was admitted.
    explicit operator bool() const noexcept { return admitted_; }
    bool admitted() const noexcept { return admitted_; }

    /// Commit the insertion — the memory is now accounted for.
    void commit() {
        if (admitted_ && !committed_ && monitor_) {
            monitor_->report_insert(delta_bytes_);
            committed_ = true;
        }
    }

private:
    memory_monitor* monitor_;
    std::size_t delta_bytes_;
    bool admitted_;
    bool committed_ = false;
};

// ============================================================================
// Eviction Strategy Concept
// ============================================================================

/// Concept defining the unified interface for eviction strategies.
///
/// Any type satisfying this concept can be used with background_evictor
/// and memory_aware_evictor. Strategies are simple structs with a
/// templated execute() method — no virtual dispatch needed.
///
/// Required interface:
///   - template <typename Cache> void execute(Cache& cache) const
///   - std::string name() const
///
/// Optionally, provide operator()(Cache&) that delegates to execute()
/// for seamless use with std::function<void(Cache&)>.
template <typename S, typename Cache>
concept eviction_strategy = requires(const S& s, Cache& cache) {
    { s.execute(cache) } -> std::same_as<void>;
    { s.name() } -> std::same_as<std::string>;
};

// ============================================================================
// Background Eviction Mover
// ============================================================================

/// Background evictor that periodically pre-evicts items to maintain a free
/// memory buffer, reducing synchronous eviction on the write path.
///
/// Usage:
///   lru::background_evictor evictor(cache, lru::free_threshold_strategy{
///       .low_watermark = 0.05,
///       .high_watermark = 0.15,
///       .max_eviction_batch = 100,
///   });
///   evictor.start(std::chrono::seconds(5));
template <typename Cache>
class background_evictor {
public:
    using cache_type = Cache;
    using size_type = typename Cache::size_type;

    /// Create a background evictor with an optional strategy.
    /// The strategy is a callable: void(Cache&) that performs one eviction cycle.
    template <typename Fn>
    background_evictor(Cache& cache, Fn&& strategy)
        : cache_(cache), strategy_(std::forward<Fn>(strategy)) {}

    ~background_evictor() noexcept {
        try {
            stop();
        } catch (...) {
            // Suppress exceptions in destructor — periodic_worker destructor
            // is already noexcept, but be defensive.
        }
    }

    /// Start the background worker with the given interval.
    void start(std::chrono::milliseconds interval) {
        if (worker_) return;
        worker_ = std::make_unique<detail::periodic_worker>(
            [this] { if (strategy_) strategy_(cache_); }, interval);
    }

    /// Stop the background worker.
    void stop() {
        if (worker_) {
            worker_->stop();
            worker_.reset();
        }
    }

    /// Manually trigger one eviction cycle.
    void evict_once() {
        if (strategy_) strategy_(cache_);
    }

private:
    Cache& cache_;
    std::function<void(Cache&)> strategy_;
    std::unique_ptr<detail::periodic_worker> worker_;
};

/// Free threshold strategy: evict when free memory drops below a watermark.
struct free_threshold_strategy {
    double low_watermark = 0.05;   // Start evicting when < 5% free
    double high_watermark = 0.15;  // Stop when > 15% free
    std::size_t max_eviction_batch = 100;

    template <typename Cache>
    void execute(Cache& cache) const {
        if (cache.size() == 0) return;
        auto max_mem = cache.max_memory();
        auto cur_mem = cache.current_memory();
        double free_ratio = (max_mem == unlimited || max_mem == 0)
                          ? 1.0 : 1.0 - static_cast<double>(cur_mem) / static_cast<double>(max_mem);

        if (free_ratio < low_watermark) {
            // Evict until we reach the high watermark or batch limit
            for (std::size_t i = 0; i < max_eviction_batch; ++i) {
                if (cache.size() == 0) break;
                free_ratio = (max_mem == unlimited || max_mem == 0)
                           ? 1.0 : 1.0 - static_cast<double>(cache.current_memory()) / static_cast<double>(max_mem);
                if (free_ratio >= high_watermark) break;
                cache.evict();
            }
        }
    }

    template <typename Cache>
    void operator()(Cache& cache) const {
        execute(cache);
    }

    std::string name() const {
        return "free_threshold";
    }
};

// ============================================================================
// LRU Tail Age Strategy
// ============================================================================

/// Evicts items from the LRU tail when the tail item's age exceeds a threshold,
/// or when the cache exceeds a size ratio relative to max_size.
///
/// This strategy is most effective for workloads where stale items accumulate
/// at the tail of the LRU and should be proactively removed to make room for
/// fresher data.
///
/// Since not all cache types expose per-item timestamps, the strategy uses a
/// pragmatic approach: when the cache occupancy exceeds `size_ratio_threshold`,
/// it performs batch eviction of LRU tail items. If the cache type supports
/// tail age inspection via `tail_update_time()`, items older than
/// `max_tail_age_seconds` are evicted preferentially.
struct lru_tail_age_strategy {
    /// Maximum age (in seconds) of the LRU tail item before triggering eviction.
    uint32_t max_tail_age_seconds = 3600;  // 1 hour default

    /// Fraction of max_size at which eviction activates [0.0, 1.0].
    /// When cache occupancy exceeds this ratio, batch eviction begins.
    double size_ratio_threshold = 0.80;

    /// Maximum number of items to evict in one cycle.
    std::size_t max_eviction_batch = 100;

    template <typename Cache>
    void execute(Cache& cache) const {
        if (cache.size() == 0) return;

        auto now = static_cast<uint32_t>(
            std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch()).count());

        // Check if the cache supports tail_update_time() for age-based eviction
        if constexpr (requires { cache.tail_update_time(); }) {
            auto tail_time = cache.tail_update_time();
            if (tail_time == 0 || (now - tail_time) < max_tail_age_seconds) {
                return;  // Tail item is fresh enough
            }
            // Evict items older than the threshold
            for (std::size_t i = 0; i < max_eviction_batch; ++i) {
                if (cache.size() == 0) break;
                auto t = cache.tail_update_time();
                if (t == 0 || (now - t) < max_tail_age_seconds) break;
                cache.evict();
            }
        } else {
            // Fallback: use size-ratio-based batch eviction
            auto max_sz = cache.max_size();
            if (max_sz == 0 || max_sz == unlimited) return;

            double occupancy = static_cast<double>(cache.size()) /
                               static_cast<double>(max_sz);
            if (occupancy < size_ratio_threshold) return;

            // Evict a proportional batch based on how far above threshold
            std::size_t to_evict = static_cast<std::size_t>(
                (occupancy - size_ratio_threshold) * static_cast<double>(max_sz));
            to_evict = std::min(to_evict, max_eviction_batch);

            for (std::size_t i = 0; i < to_evict; ++i) {
                if (cache.size() == 0) break;
                cache.evict();
            }
        }
    }

    template <typename Cache>
    void operator()(Cache& cache) const {
        execute(cache);
    }

    std::string name() const {
        return "lru_tail_age";
    }
};

// ============================================================================
// Hits Per Slab Strategy
// ============================================================================

/// Evicts items when the overall cache hit rate drops below a threshold,
/// making room for potentially more useful items.
///
/// Inspired by CacheLib's hits-per-slab strategy. Since this library does not
/// maintain per-slab hit statistics, the strategy degrades to an overall
/// hit-rate check: if hit rate is below the threshold, it evicts a batch of
/// LRU items to churn the cache and make room for potentially hotter items.
struct hits_per_slab_strategy {
    /// Hit rate below which aggressive eviction begins [0.0, 1.0].
    double low_hit_rate_threshold = 0.05;  // 5% hit rate

    /// Maximum number of items to evict in one cycle.
    std::size_t max_eviction_batch = 100;

    template <typename Cache>
    void execute(Cache& cache) const {
        if (cache.size() == 0) return;

        // Retrieve hit rate — try stats_snapshot() first (unified_cache),
        // then stats() (compact_cache / raw mm types)
        double hr = 0.0;
        if constexpr (requires { cache.stats_snapshot().hit_rate(); }) {
            hr = cache.stats_snapshot().hit_rate();
        } else if constexpr (requires { cache.stats().hit_rate(); }) {
            hr = cache.stats().hit_rate();
        } else {
            return;  // No way to get hit rate — skip
        }

        if (hr >= low_hit_rate_threshold) return;

        // Low hit rate: evict a batch to churn the cache
        for (std::size_t i = 0; i < max_eviction_batch; ++i) {
            if (cache.size() == 0) break;
            cache.evict();
        }
    }

    template <typename Cache>
    void operator()(Cache& cache) const {
        execute(cache);
    }

    std::string name() const {
        return "hits_per_slab";
    }
};

// ============================================================================
// Memory-Aware Evictor
// ============================================================================

/// A background evictor that connects memory_monitor with an eviction strategy,
/// accelerating eviction when the monitor signals a throttled state.
///
/// Unlike `background_evictor` which runs at a fixed interval and batch size,
/// `memory_aware_evictor` dynamically adjusts:
///   - Normal state: runs the strategy once per tick at `normal_interval`
///   - Throttled state: runs the strategy `throttle_batch_` times per tick
///     at `throttled_interval` (typically much faster)
///
/// Usage:
///   lru::memory_monitor mon(cfg);
///   lru::memory_aware_evictor evictor(cache,
///       lru::free_threshold_strategy{.low_watermark = 0.05, .high_watermark = 0.15},
///       mon);
///   evictor.start(std::chrono::seconds(5), std::chrono::milliseconds(500));
template <typename Cache>
class memory_aware_evictor {
public:
    using cache_type = Cache;

    /// Create a memory-aware evictor.
    /// @param cache     The cache to evict from.
    /// @param strategy  A callable: void(Cache&) that performs one eviction cycle.
    /// @param monitor   The memory monitor to observe for throttle signals.
    template <typename EvictionStrategy, typename Monitor>
    memory_aware_evictor(Cache& cache, EvictionStrategy&& strategy, Monitor& monitor)
        : cache_(cache)
        , strategy_(std::forward<EvictionStrategy>(strategy))
        , monitor_(&monitor) {}

    ~memory_aware_evictor() noexcept {
        try {
            stop();
        } catch (...) {
            // Suppress exceptions in destructor
        }
    }

    /// Start the background worker.
    /// @param normal_interval    Interval between ticks in normal state.
    /// @param throttled_interval Interval between ticks when throttled.
    void start(std::chrono::milliseconds normal_interval,
               std::chrono::milliseconds throttled_interval) {
        if (worker_) return;
        normal_interval_ = normal_interval;
        throttled_interval_ = throttled_interval;
        // Start with normal interval; tick() will reschedule if needed
        worker_ = std::make_unique<detail::periodic_worker>(
            [this] { tick(); }, normal_interval);
    }

    /// Stop the background worker.
    void stop() {
        if (worker_) {
            worker_->stop();
            worker_.reset();
        }
    }

    /// Manually trigger one eviction tick.
    void tick() {
        if (monitor_ && monitor_->is_throttled()) {
            // Accelerated eviction: run strategy multiple times
            for (std::size_t i = 0; i < throttle_batch_; ++i) {
                if (strategy_) strategy_(cache_);
            }
        } else {
            // Normal eviction: run strategy once
            if (strategy_) strategy_(cache_);
        }
    }

    /// Set the number of strategy invocations per tick when throttled.
    void set_throttle_batch(std::size_t batch) noexcept {
        throttle_batch_ = batch;
    }

    /// Get the current throttle batch size.
    std::size_t throttle_batch() const noexcept {
        return throttle_batch_;
    }

private:
    Cache& cache_;
    std::function<void(Cache&)> strategy_;
    memory_monitor* monitor_;
    std::unique_ptr<detail::periodic_worker> worker_;
    std::chrono::milliseconds normal_interval_{5000};
    std::chrono::milliseconds throttled_interval_{500};
    std::size_t throttle_batch_ = 10;
};

// ============================================================================
// Allocation Class (lock-free free list per power-of-2 size class)
// ============================================================================

/// A single size class within the slab allocator.
/// Manages an ABA-safe lock-free free list (Treiber stack with tagged pointer)
/// for blocks of a fixed size.
///
/// P2-H: The tagged pointer now uses a 32-bit ABA counter packed alongside a
/// full 64-bit pointer in a 128-bit atomic word (cmpxchg16b on x86-64). The
/// previous 16-bit tag packed into the upper 16 bits of a 64-bit word wrapped
/// after 65,536 operations, allowing ABA under high contention. The 32-bit tag
/// wraps after ~4 billion operations, effectively eliminating ABA for
/// realistic workloads.
///
/// Requires cmpxchg16b support (x86-64 v2 baseline, available on all modern
/// x86-64 CPUs since ~2006). Compile with -mcx16 on GCC/Clang. MSVC emits
/// cmpxchg16b by default on x86-64.
class allocation_class {
public:
    /// @param class_size   Power-of-2 block size (e.g., 64, 128, 256, …)
    /// @param slab_size    Size of a single slab in bytes (default 64 KiB)
    ///
    /// T-P2-10: slab_size is now stored per-class so that each size class
    /// can have a different slab granularity (e.g., larger slabs for the
    /// 64-byte class to amortise the per-sab overhead, smaller slabs for
    /// the 65536-byte class to reduce wasted memory).
    explicit allocation_class(uint32_t class_size, uint32_t slab_size = 65536) noexcept
        : class_size_(class_size)
        , slab_size_(slab_size)
        , items_per_slab_(slab_size / class_size) {
        // 一次性提示：16 字节原子非编译期锁自由时（如 MSYS2 MinGW GCC），
        // free-list CAS 经 libatomic 锁池回退仍正确工作；此提示引导启用 cmpxchg16b。
        static const bool s_warned = []() noexcept {
            if (!std::atomic<tagged_ptr>::is_always_lock_free) {
                std::fprintf(stderr,
                    "[lru] WARNING: allocation_class free_list uses std::atomic<16-byte> "
                    "tagged_ptr which is not compile-time lock-free on this toolchain "
                    "(libatomic fallback in use). Compile with -mcx16 on GCC/Clang for "
                    "lock-free cmpxchg16b.\n");
                std::fflush(stderr);
            }
            return true;
        }();
        (void)s_warned;
    }

    allocation_class(allocation_class&& other) noexcept
        : class_size_(other.class_size_)
        , slab_size_(other.slab_size_)
        , items_per_slab_(other.items_per_slab_)
        , num_slabs_(other.num_slabs_)
        , free_count_(other.free_count_.load(std::memory_order_relaxed))
        , free_list_(other.free_list_.load(std::memory_order_relaxed)) {
        other.free_list_.store(tagged_ptr{}, std::memory_order_relaxed);
        other.num_slabs_ = 0;
        other.free_count_.store(0, std::memory_order_relaxed);
    }

    allocation_class(const allocation_class&) = delete;
    allocation_class& operator=(const allocation_class&) = delete;
    allocation_class& operator=(allocation_class&&) = delete;

    /// Lock-free allocate with ABA-safe tagged pointer.
    /// @return Pointer to a free block, or nullptr if the free list is exhausted.
    void* allocate() noexcept {
        tagged_ptr expected = free_list_.load(std::memory_order_acquire);
        while (expected.ptr != 0) {
            node* head = reinterpret_cast<node*>(expected.ptr);
            node* next = head->next.load(std::memory_order_acquire);
            tagged_ptr desired;
            if (next) {
                desired.ptr = reinterpret_cast<uint64_t>(next);
                desired.tag = expected.tag + 1;
            }
            // When next is nullptr (last element), desired remains {0, 0} (empty).
            if (free_list_.compare_exchange_weak(expected, desired,
                    std::memory_order_acq_rel, std::memory_order_acquire)) {
                free_count_.fetch_sub(1, std::memory_order_relaxed);
                return head;
            }
            // CAS failed — expected is updated with current value, retry
        }
        return nullptr;
    }

    /// Lock-free deallocate with ABA-safe tagged pointer.
    void deallocate(void* ptr) noexcept {
        auto* node_ptr = static_cast<node*>(ptr);
        tagged_ptr expected = free_list_.load(std::memory_order_acquire);
        tagged_ptr desired;
        do {
            node_ptr->next.store(reinterpret_cast<node*>(expected.ptr),
                                  std::memory_order_release);
            desired.ptr = reinterpret_cast<uint64_t>(node_ptr);
            desired.tag = expected.tag + 1;
        } while (!free_list_.compare_exchange_weak(expected, desired,
                    std::memory_order_acq_rel, std::memory_order_acquire));
        free_count_.fetch_add(1, std::memory_order_relaxed);
    }

    /// Populate the free list by carving a slab into class-sized blocks.
    void add_slab(void* slab_memory) noexcept {
        auto* ptr = static_cast<char*>(slab_memory);
        for (uint32_t i = 0; i < items_per_slab_; ++i) {
            deallocate(ptr + i * class_size_);
        }
        ++num_slabs_;
    }

    // -- Accessors --

    uint32_t class_size() const noexcept { return class_size_; }
    /// T-P2-10: Return the per-class slab size in bytes.
    uint32_t slab_size() const noexcept { return slab_size_; }
    uint32_t items_per_slab() const noexcept { return items_per_slab_; }
    uint32_t num_slabs() const noexcept { return num_slabs_; }
    uint32_t free_count() const noexcept { return free_count_.load(std::memory_order_relaxed); }
    bool empty() const noexcept { return free_list_.load(std::memory_order_acquire).ptr == 0; }

    /// Decrement slab count after a slab has been transferred out.
    void release_slab() noexcept { --num_slabs_; }

private:
    /// Intrusive link stored at the head of each free block.
    /// Because every free block is at least class_size_ bytes (≥ 64),
    /// there is always room for a single pointer.
    struct node {
        std::atomic<node*> next{nullptr};
    };

    /// 128-bit tagged pointer: 64-bit pointer + 32-bit ABA counter.
    /// P2-H: Extended from 16-bit tag (which wrapped after 65K ops) to
    /// 32-bit tag (wraps after ~4B ops). Uses cmpxchg16b for atomic CAS.
    struct alignas(16) tagged_ptr {
        uint64_t ptr{0};      ///< Raw pointer (full 64 bits, no bit-stealing)
        uint32_t tag{0};      ///< 32-bit ABA counter
        uint32_t reserved{0}; ///< Padding to 16 bytes for cmpxchg16b
    };
    static_assert(sizeof(tagged_ptr) == 16, "tagged_ptr must be 16 bytes");

    uint32_t class_size_;
    /// T-P2-10: Per-class slab size (bytes per slab for this class).
    uint32_t slab_size_;
    uint32_t items_per_slab_;
    uint32_t num_slabs_{0};
    std::atomic<uint32_t> free_count_{0};
    std::atomic<tagged_ptr> free_list_{};  ///< 128-bit tagged pointer + ABA counter
    // P2-H: 16-byte atomics should be lock-free (cmpxchg16b) for the free-list
    // Treiber stack to scale under contention. Compile with -mcx16 on GCC/Clang
    // (x86-64 v2 baseline); MSVC emits cmpxchg16b by default.
    //
    // UCRT64 GCC 兼容性（2026-08）: MSYS2 MinGW GCC 的 libstdc++ 对 16 字节类型
    // 无条件报告 is_always_lock_free == false（即使 -mcx16 / -march=native 也不改变），
    // 但 std::atomic<16 字节> 仍可经 libatomic 锁池回退正确工作（较慢但正确）。
    // 因此这里不再编译期硬性断言，改为构造时一次性运行时提示。
    static_assert(sizeof(tagged_ptr) == 16, "tagged_ptr must be 16 bytes");
};

// ============================================================================
// Shared Memory Header (for warm restart)
// ============================================================================

/// Header written at the beginning of the shared memory file.
/// Used to validate the file on warm restart — if the header doesn't
/// match expected config, the allocator falls back to fresh allocation.
///
/// Layout (88 bytes):
///   magic[4]               "LRUS" identifier
///   version (uint32_t)     Format version (currently 1)
///   slab_size (uint64_t)   Bytes per slab
///   num_slabs (uint64_t)   Total number of slabs in the file
///   total_size (uint64_t)  Total mapped region size (header + data)
///   class_slab_counts[11]  Per-class slab count
///   reserved_[3]           Reserved for future use
struct shared_memory_header {
    char magic[4] = {'L', 'R', 'U', 'S'};
    uint32_t version = 1;
    uint64_t slab_size = 0;
    uint64_t num_slabs = 0;
    uint64_t total_size = 0;
    uint32_t class_slab_counts[11] = {};
    uint32_t reserved_[3] = {};
};

static_assert(sizeof(shared_memory_header) == 88,
              "shared_memory_header must be exactly 88 bytes");

// ============================================================================
// Slab Allocator (size-class based, lock-free fast path)
// ============================================================================

/// Multi-size-class slab allocator inspired by Facebook CacheLib.
///
/// Allocation fast path (per-class Treiber stack) is lock-free.
/// Only the slow path (adding a new slab) acquires a per-class mutex.
///
/// T-P2-10: The allocator now uses per-class locks (slab_mutexes_[i])
/// instead of a single global mutex. This eliminates contention between
/// different size classes on the slow path. Per-class slab_size and
/// max_slabs_per_class are also configurable and dynamically growable.
///
/// Size classes: 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384,
///               32768, 65536  (11 classes)
/// Requests larger than 65536 bytes fall back to ::operator new/delete.
class slab_allocator {
public:
    /// Configuration for the slab allocator.
    struct config {
        uint32_t slab_size = 65536;        // bytes per slab (default for all classes)
        uint32_t initial_slabs_per_class = 1;
        uint32_t max_slabs_per_class = 64;

        /// T-P2-10: Per-class slab size overrides.
        /// Entry [i] sets the slab size (bytes) for size class i.
        /// If 0, the default `slab_size` is used for that class.
        /// Indexed by class index (0 = 64-byte class, …, 10 = 65536-byte class).
        /// Example: set per_class_slab_sizes[0] = 262144 to give the 64-byte
        /// class larger slabs (more items per slab → fewer slow-path misses).
        std::array<uint32_t, 11> per_class_slab_sizes{};

        /// T-P2-10: Per-class max slabs overrides.
        /// Entry [i] sets the maximum number of slabs for size class i.
        /// If 0, the default `max_slabs_per_class` is used for that class.
        /// The per-class limit can also be grown at runtime via
        /// set_max_slabs_for_class().
        std::array<uint32_t, 11> per_class_max_slabs{};

        /// NUMA node for slab allocation.
        /// -1 (default): no NUMA binding, use default allocation.
        /// >= 0: bind slab memory to the specified NUMA node.
        ///
        /// Platform behavior:
        ///   Linux  : uses mbind() syscall with MPOL_BIND to set the memory
        ///            policy for the allocated region. Physical pages will be
        ///            placed on the specified node on first touch.
        ///   Windows: uses VirtualAllocExNuma() (Win8+) for NUMA-aware
        ///            allocation. Falls back to regular allocation if the
        ///            API is unavailable.
        ///
        /// If the NUMA API fails or is unavailable, regular allocation is
        /// used — the slab is still valid, just not NUMA-bound.
        int numa_node = -1;

        /// Path for shared memory file (warm restart / nil-restart support).
        /// Empty string (default) means normal allocation.
        /// When non-empty, slabs are allocated in a memory-mapped file at
        /// this path.
        ///
        /// On first run: creates the file and allocates slabs within it.
        /// On subsequent runs: mmaps the existing file and reuses the slab
        /// data directly for near-zero startup time (warm restart).
        ///
        /// Platform behavior:
        ///   Windows: CreateFileMappingW + MapViewOfFile
        ///   Linux  : mmap with MAP_SHARED on a file descriptor
        ///
        /// If the shared memory APIs fail, falls back to regular allocation.
        /// The shared memory file is NOT automatically deleted — the user
        /// manages its lifecycle.
        std::string shared_memory_path;
    };

    static constexpr uint32_t kMinClassSize = 64;
    static constexpr uint32_t kMaxClassSize = 65536;
    static constexpr uint32_t kNumClasses = 11; // 2^6 … 2^16

    // ----------------------------------------------------------------
    // Shared Memory / Warm Restart
    // ----------------------------------------------------------------

    /// Returns true if the allocator was initialized from an existing
    /// shared memory file (warm restart). Returns false for a fresh
    /// start or when using regular allocation.
    bool is_warm_restart() const noexcept { return is_warm_restart_; }

    /// Returns a pointer to the shared memory data region (after the header).
    /// Returns nullptr if shared memory is not active.
    /// On warm restart, this region contains slab data from the previous run.
    void* shared_memory_data() const noexcept {
        if (!shared_mem_base_) return nullptr;
        return static_cast<char*>(shared_mem_base_) + kSharedMemHeaderSize;
    }

    /// Returns the size of the shared memory data region in bytes
    /// (total mapped size minus the header).
    /// Returns 0 if shared memory is not active.
    std::size_t shared_memory_data_size() const noexcept {
        if (!shared_mem_base_) return 0;
        return shared_mem_size_ > kSharedMemHeaderSize
             ? shared_mem_size_ - kSharedMemHeaderSize : 0;
    }

    slab_allocator()
        : slab_allocator(config{}) {}

    explicit slab_allocator(const config& cfg)
        : config_(cfg)
    {
        // T-P2-10: Initialize per-class slab sizes and max-slabs from config.
        // If per_class_slab_sizes[i] == 0, fall back to the default slab_size.
        // If per_class_max_slabs[i] == 0, fall back to the default max_slabs_per_class.
        for (uint32_t i = 0; i < kNumClasses; ++i) {
            slab_sizes_[i] = (cfg.per_class_slab_sizes[i] != 0)
                ? cfg.per_class_slab_sizes[i]
                : cfg.slab_size;
            uint32_t max_slabs = (cfg.per_class_max_slabs[i] != 0)
                ? cfg.per_class_max_slabs[i]
                : cfg.max_slabs_per_class;
            max_slabs_per_class_[i].store(max_slabs, std::memory_order_relaxed);
        }

        // Initialise allocation classes: 64, 128, 256, …, 65536
        // T-P2-10: each class gets its own slab_size.
        uint32_t cs = kMinClassSize;
        for (uint32_t i = 0; i < kNumClasses; ++i) {
            classes_.emplace_back(cs, slab_sizes_[i]);
            cs <<= 1;
        }

        // Initialize shared memory if configured
        if (!config_.shared_memory_path.empty()) {
            init_shared_memory();
        }

        // If shared memory is not active, pre-allocate using regular allocation
        if (!shared_mem_base_) {
            for (uint32_t i = 0; i < kNumClasses; ++i) {
                for (uint32_t s = 0; s < config_.initial_slabs_per_class; ++s) {
                    add_slab_to_class(i);
                }
            }
        }
    }

    ~slab_allocator() {
        // Clean up shared memory mapping and handles first
        cleanup_shared_memory();

        for (auto* slab : all_slabs_) {
            auto addr = reinterpret_cast<std::uintptr_t>(slab);
            // Slabs in shared memory are freed by cleanup_shared_memory()
            if (shared_mem_slabs_.count(addr)) {
                continue;
            }
            if (numa_slabs_.count(addr)) {
                // NUMA-allocated via VirtualAllocExNuma → must use VirtualFree
#if defined(LRU_HAS_WIN32_NUMA)
                VirtualFree(slab, 0, MEM_RELEASE);
#else
                std::free(slab);
#endif
            } else if (aligned_slabs_.count(addr)) {
#if defined(_WIN32)
                _aligned_free(slab);
#else
                std::free(slab);
#endif
            } else {
                ::operator delete(slab);
            }
        }
    }

    slab_allocator(const slab_allocator&) = delete;
    slab_allocator& operator=(const slab_allocator&) = delete;

    // ----------------------------------------------------------------
    // Core API
    // ----------------------------------------------------------------

    /// Allocate a block of at least `size` bytes.
    /// Uses the appropriate size class; falls back to ::operator new for
    /// sizes larger than the maximum class.
    void* allocate(std::size_t size) {
        if (size > kMaxClassSize) {
            return ::operator new(size);
        }
        uint32_t idx = class_index_for(static_cast<uint32_t>(size));
        // Fast path: lock-free pop from the per-class free list
        if (void* ptr = classes_[idx].allocate()) {
            return ptr;
        }
        // Slow path: need a new slab for this class.
        // T-P2-10: acquire the per-class lock instead of the global mutex.
        // This avoids blocking allocations in other size classes.
        {
            std::lock_guard<std::mutex> lock(slab_mutexes_[idx]);
            // Double-check after acquiring lock (another thread may have added a slab)
            if (void* ptr = classes_[idx].allocate()) {
                return ptr;
            }
            add_slab_to_class(idx);
            return classes_[idx].allocate();
        }
    }

    /// Deallocate a block previously returned by `allocate(size)`.
    /// The `size` parameter must match the size passed to allocate.
    void deallocate(void* ptr, std::size_t size) noexcept {
        if (!ptr) return;
        if (size > kMaxClassSize) {
            ::operator delete(ptr);
            return;
        }
        uint32_t idx = class_index_for(static_cast<uint32_t>(size));
        classes_[idx].deallocate(ptr);
    }

    // ----------------------------------------------------------------
    // Size-class helpers
    // ----------------------------------------------------------------

    /// Return the index into classes_ for the smallest class ≥ size.
    /// `size` must be in [1, kMaxClassSize].
    static uint32_t class_index_for(uint32_t size) {
        if (size <= kMinClassSize) return 0;
        // floor(log2(size-1)) clamped to [0, kNumClasses-1]
        // Equivalent: find the smallest power-of-2 >= size
        uint32_t cls = kMinClassSize;
        uint32_t idx = 0;
        while (cls < size && idx < kNumClasses - 1) {
            cls <<= 1;
            ++idx;
        }
        return idx;
    }

    /// Return the allocation_class for a given index (0..kNumClasses-1).
    allocation_class& class_at(uint32_t idx) { return classes_[idx]; }
    const allocation_class& class_at(uint32_t idx) const { return classes_[idx]; }

    /// Number of size classes.
    static constexpr uint32_t num_classes() noexcept { return kNumClasses; }

    // ----------------------------------------------------------------
    // T-P2-10: Per-class configuration
    // ----------------------------------------------------------------

    /// T-P2-10: Get the slab size (bytes) for a specific class.
    uint32_t slab_size_for_class(uint32_t class_idx) const {
        if (class_idx >= kNumClasses) return config_.slab_size;
        return slab_sizes_[class_idx];
    }

    /// T-P2-10: Get the max slabs for a specific class.
    /// Returns the current dynamically-adjustable limit.
    uint32_t max_slabs_for_class(uint32_t class_idx) const {
        if (class_idx >= kNumClasses) return config_.max_slabs_per_class;
        return max_slabs_per_class_[class_idx].load(std::memory_order_acquire);
    }

    /// T-P2-10: Grow (or shrink) the max-slabs limit for a specific class.
    /// Safe to call at any time — new allocations will respect the updated
    /// limit immediately. Growing the limit allows add_slab_to_class() to
    /// allocate more slabs for the class.
    void set_max_slabs_for_class(uint32_t class_idx, uint32_t max_slabs) {
        if (class_idx >= kNumClasses) return;
        max_slabs_per_class_[class_idx].store(max_slabs, std::memory_order_release);
    }

    // ----------------------------------------------------------------
    // Statistics
    // ----------------------------------------------------------------

    /// Per-class utilization statistics used by the slab rebalancer.
    struct allocation_class_stats {
        uint32_t class_size;
        uint32_t num_slabs;
        uint32_t items_per_slab;
        double utilization;  // allocated / total capacity
    };

    /// Get per-class utilization statistics.
    std::vector<allocation_class_stats> get_stats() const {
        std::vector<allocation_class_stats> result;
        result.reserve(kNumClasses);
        for (uint32_t i = 0; i < kNumClasses; ++i) {
            const auto& cls = classes_[i];
            allocation_class_stats cs{};
            cs.class_size = cls.class_size();
            cs.num_slabs = cls.num_slabs();
            cs.items_per_slab = cls.items_per_slab();
            uint32_t total_capacity = cs.num_slabs * cs.items_per_slab;
            uint32_t free_cnt = cls.free_count();
            cs.utilization = (total_capacity > 0)
                ? 1.0 - static_cast<double>(free_cnt) / static_cast<double>(total_capacity)
                : 0.0;
            result.push_back(cs);
        }
        return result;
    }

private:
    /// Allocate a raw slab from the OS and add it to class `idx`.
    /// Must be called while holding slab_mutexes_[idx].
    ///
    /// T-P2-10: Uses per-class slab_size (slab_sizes_[idx]) and per-class
    /// max-slabs limit (max_slabs_per_class_[idx]).
    ///
    /// When shared memory is active, allocates from the mapped region.
    /// When config_.numa_node >= 0, attempts NUMA-aware allocation:
    ///   Linux  : mbind() with MPOL_BIND after aligned_alloc (zero libnuma dependency)
    ///   Windows: VirtualAllocExNuma() (Win8+), dynamically resolved
    /// Falls back to regular aligned allocation if NUMA APIs are unavailable or fail.
    void add_slab_to_class(uint32_t idx) {
        auto& cls = classes_[idx];
        // T-P2-10: use the per-class (dynamically growable) max-slabs limit.
        if (cls.num_slabs() >= max_slabs_per_class_[idx].load(std::memory_order_relaxed)) return;
        // T-P2-10: use the per-class slab size.
        auto slab_size = slab_sizes_[idx];

        void* slab = nullptr;
        bool numa_allocated = false;
        bool from_shared_mem = false;

        // ---- Shared memory allocation ----
        if (shared_mem_base_) {
            std::size_t offset = shared_mem_next_slab_offset_;
            if (kSharedMemHeaderSize + offset + slab_size <= shared_mem_size_) {
                slab = static_cast<char*>(shared_mem_base_) + kSharedMemHeaderSize + offset;
                shared_mem_next_slab_offset_ += slab_size;
                from_shared_mem = true;
            }
            // If not enough space in shared memory, fall through to regular allocation
        }

        // ---- NUMA-aware allocation (numa_node >= 0) ----
        if (config_.numa_node >= 0) {
#if defined(LRU_HAS_WIN32_NUMA)
            // Windows: dynamically resolve VirtualAllocExNuma (available on Win8+).
            // VirtualAllocExNuma allocates pages on the preferred NUMA node at
            // allocation time, so no separate binding step is needed.
            using alloc_numa_fn_t = LPVOID(WINAPI*)(HANDLE, LPVOID, SIZE_T, DWORD, DWORD, DWORD);
            static const alloc_numa_fn_t pVirtualAllocExNuma =
                reinterpret_cast<alloc_numa_fn_t>(
                    GetProcAddress(GetModuleHandleW(L"kernel32.dll"),
                                   "VirtualAllocExNuma"));
            if (pVirtualAllocExNuma) {
                slab = pVirtualAllocExNuma(
                    GetCurrentProcess(), nullptr, slab_size,
                    MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE,
                    static_cast<DWORD>(config_.numa_node));
                if (slab) numa_allocated = true;
            }
#endif
        }

        // ---- Regular aligned allocation (fallback or non-NUMA) ----
        if (!slab) {
#if defined(_WIN32)
            slab = _aligned_malloc(slab_size, slab_size);
#elif defined(__APPLE__)
            if (posix_memalign(&slab, slab_size, slab_size) != 0) slab = nullptr;
#else
            slab = std::aligned_alloc(slab_size, slab_size);
#endif
        }

        if (!slab) {
            // Aligned allocation failed — fall back to ordinary allocation
            slab = ::operator new(slab_size);
        }

        // ---- Linux NUMA binding via mbind() syscall ----
        // Must be called BEFORE the slab memory is touched (first write),
        // because mbind() with MPOL_BIND only affects pages that haven't
        // been faulted in yet. The kernel's first-touch policy will place
        // physical pages on the specified NUMA node on subsequent faults.
#if defined(LRU_HAS_LINUX_NUMA)
        if (config_.numa_node >= 0 && slab) {
            int node = config_.numa_node;
            if (node >= 0 &&
                node < static_cast<int>(sizeof(unsigned long) * 8)) {
                unsigned long nodemask = 1UL << node;
                unsigned long maxnode = static_cast<unsigned long>(node) + 1;
                // Direct syscall — no dependency on libnuma.
                long ret = syscall(__NR_mbind, slab, slab_size, MPOL_BIND,
                                   &nodemask, maxnode, 0);
                // Non-fatal: slab is valid regardless of mbind result.
                // Cross-NUMA access works, just with higher latency.
                (void)ret;
            }
        }
#endif

        // T-P2-10 fix: Protect cross-class bookkeeping containers with a
        // dedicated mutex. The per-class slab_mutexes_[idx] only protects
        // the allocation_class operations; the shared containers below are
        // accessed from all classes concurrently.
        {
            std::lock_guard<std::mutex> bk_lock(bookkeeping_mutex_);
            all_slabs_.push_back(slab);
            auto addr = reinterpret_cast<std::uintptr_t>(slab);
            if (from_shared_mem) {
                // Shared memory slabs are not necessarily aligned to slab_size,
                // so they won't use the fast path in find_slab_base().
                shared_mem_slabs_.insert(addr);
            } else {
                // Record whether this slab was aligned (addr is a multiple of slab_size)
                if ((addr % static_cast<std::uintptr_t>(slab_size)) == 0) {
                    aligned_slabs_.insert(addr);
                }
                if (numa_allocated) {
                    numa_slabs_.insert(addr);
                }
            }
        }
        cls.add_slab(slab);

        // Update shared memory header with current slab counts
        if (shared_mem_base_ && from_shared_mem) {
            update_shared_memory_header();
        }
    }

    /// Find the slab base address that contains `ptr`.
    /// Returns nullptr if the pointer doesn't belong to any known slab.
    ///
    /// Fast path (O(1)): for aligned slabs, compute the candidate base via
    /// bit-mask and verify with aligned_slabs_ hash set lookup.
    /// Slow path (O(n)): linear scan for non-aligned slabs (rare fallback)
    /// and shared memory slabs.
    ///
    /// T-P2-10: An optional `slab_size_hint` overrides the default config
    /// slab_size, allowing callers that know the class index to use the
    /// correct per-class slab size. If 0, falls back to config_.slab_size.
    void* find_slab_base(void* ptr, uint32_t slab_size_hint = 0) const {
        auto addr = reinterpret_cast<std::uintptr_t>(ptr);
        auto slab_size = (slab_size_hint != 0) ? slab_size_hint : config_.slab_size;

        // T-P2-10 fix: Lock bookkeeping_mutex_ to prevent concurrent writes
        // from add_slab_to_class() in another thread. This is only called
        // from the rare try_move_slab() path, so the lock doesn't affect
        // hot-path performance.
        std::lock_guard<std::mutex> lock(bookkeeping_mutex_);

        // Fast path: if the slab was aligned, compute base via alignment mask
        auto candidate = addr & ~(static_cast<std::uintptr_t>(slab_size) - 1);

        // Verify candidate is actually a known aligned slab
        if (aligned_slabs_.count(candidate)) {
            return reinterpret_cast<void*>(candidate);
        }

        // Check if the pointer falls within the shared memory region
        if (shared_mem_base_) {
            auto sm_start = reinterpret_cast<std::uintptr_t>(shared_mem_base_) + kSharedMemHeaderSize;
            auto sm_end = sm_start + (shared_mem_size_ - kSharedMemHeaderSize);
            if (addr >= sm_start && addr < sm_end) {
                // Compute slab base within shared memory
                auto offset = addr - sm_start;
                auto slab_offset = (offset / slab_size) * slab_size;
                return reinterpret_cast<void*>(sm_start + slab_offset);
            }
        }

        // Slow path: linear scan for non-aligned slabs
        for (auto* slab : all_slabs_) {
            auto slab_addr = reinterpret_cast<std::uintptr_t>(slab);
            if (addr >= slab_addr && addr < slab_addr + slab_size) {
                return slab;
            }
        }
        return nullptr;
    }

    config config_;
    std::vector<allocation_class> classes_;
    std::vector<void*> all_slabs_;
    std::unordered_set<std::uintptr_t> aligned_slabs_;
    /// Slabs allocated via Windows NUMA API (VirtualAllocExNuma).
    /// Tracked separately because they require VirtualFree for deallocation.
    /// Always empty on non-Windows platforms.
    std::unordered_set<std::uintptr_t> numa_slabs_;

    /// T-P2-10 fix: Mutex protecting the cross-class bookkeeping containers
    /// (`all_slabs_`, `aligned_slabs_`, `numa_slabs_`, `shared_mem_slabs_`).
    ///
    /// The per-class `slab_mutexes_[idx]` only protects the allocation_class
    /// free-list and slab-count operations for a single class. However,
    /// `add_slab_to_class()` also writes to the shared containers above, which
    /// are cross-class. Without this mutex, two threads adding slabs to
    /// different classes concurrently would race on `all_slabs_.push_back()`
    /// and `aligned_slabs_.insert()`, causing heap corruption (0xc0000374 on
    /// Windows).
    ///
    /// This mutex is only held on the slow path (new slab creation) and in
    /// `find_slab_base()` (rare slab-move path), so it does not affect the
    /// lock-free fast path of `allocate()` / `deallocate()`.
    mutable std::mutex bookkeeping_mutex_;

    // ----------------------------------------------------------------
    // T-P2-10: Per-class locks and per-class configuration
    // ----------------------------------------------------------------
    //
    // The old single global `slab_mutex_` has been replaced by a per-class
    // lock array (`slab_mutexes_`). The slow path in allocate() only locks
    // the mutex for the specific class being grown, so allocations in
    // different size classes no longer contend with each other.
    //
    // `slab_sizes_[i]` holds the slab size (bytes) for class i, initialised
    // from config (default or per-class override).
    //
    // `max_slabs_per_class_[i]` is an atomic, dynamically growable limit
    // on the number of slabs for class i. It can be grown at runtime via
    // set_max_slabs_for_class().

    /// T-P2-10: Per-class slab sizes (bytes per slab for each class).
    std::array<uint32_t, kNumClasses> slab_sizes_{};

    /// T-P2-10: Per-class maximum slab counts (dynamically growable).
    std::array<std::atomic<uint32_t>, kNumClasses> max_slabs_per_class_{};

    /// T-P2-10: Per-class mutexes — replaces the old global slab_mutex_.
    /// Only taken on the slow path (adding a new slab). Each class has its
    /// own lock so that growing one class does not block another.
    std::array<std::mutex, kNumClasses> slab_mutexes_;

    // ----------------------------------------------------------------
    // Shared memory state
    // ----------------------------------------------------------------
    static constexpr std::size_t kSharedMemHeaderSize = sizeof(shared_memory_header);

    /// Base address of the mapped shared memory region (or nullptr).
    void* shared_mem_base_ = nullptr;
    /// Total size of the mapped region in bytes.
    std::size_t shared_mem_size_ = 0;
    /// Next slab offset within the data region (after header).
    /// Always accessed under the relevant slab_mutexes_[idx].
    std::size_t shared_mem_next_slab_offset_ = 0;
    /// True if the allocator was initialized from an existing shared memory file.
    bool is_warm_restart_ = false;
    /// Set of slab addresses that belong to the shared memory region.
    std::unordered_set<std::uintptr_t> shared_mem_slabs_;

#if defined(_WIN32)
    /// Windows: file handle returned by CreateFileW.
    HANDLE shared_mem_file_handle_ = nullptr;
    /// Windows: file mapping handle returned by CreateFileMappingW.
    HANDLE shared_mem_mapping_handle_ = nullptr;
#else
    /// Linux/POSIX: file descriptor returned by open().
    int shared_mem_fd_ = -1;
#endif

    /// Initialize shared memory mapping.
    /// Called from the constructor when shared_memory_path is set.
    /// On success, shared_mem_base_ is non-null.
    /// On failure, shared_mem_base_ remains null and the constructor
    /// falls back to regular allocation.
    void init_shared_memory() {
        const auto& path = config_.shared_memory_path;
        auto default_slab_size = config_.slab_size;

        // T-P2-10: Calculate the file size using per-class slab sizes and
        // per-class max-slabs limits. Each class i contributes
        // max_slabs_per_class_[i] * slab_sizes_[i] bytes.
        std::size_t data_size = 0;
        std::size_t max_total_slabs = 0;
        for (uint32_t i = 0; i < kNumClasses; ++i) {
            std::size_t mc = max_slabs_per_class_[i].load(std::memory_order_relaxed);
            max_total_slabs += mc;
            data_size += mc * slab_sizes_[i];
        }
        std::size_t total_file_size = kSharedMemHeaderSize + data_size;

        bool warm_restart = false;
        std::size_t existing_file_size = 0;

#if defined(_WIN32)
        // ---- Windows: CreateFileMappingW + MapViewOfFile ----
        // Convert UTF-8 path to wide string
        int wlen = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
        if (wlen <= 0) {
            std::fprintf(stderr, "[slab_allocator] shared_memory_path: "
                         "MultiByteToWideChar failed for '%s'\n", path.c_str());
            return;
        }
        std::wstring wpath(static_cast<std::size_t>(wlen), L'\0');
        MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, &wpath[0], wlen);

        // Try to open existing file first (potential warm restart)
        HANDLE file = CreateFileW(wpath.c_str(),
            GENERIC_READ | GENERIC_WRITE,
            FILE_SHARE_READ | FILE_SHARE_WRITE,
            nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);

        if (file != INVALID_HANDLE_VALUE) {
            // File exists — check if it's large enough to contain a header
            LARGE_INTEGER file_size_li;
            if (GetFileSizeEx(file, &file_size_li) &&
                file_size_li.QuadPart >= static_cast<LONGLONG>(kSharedMemHeaderSize)) {
                warm_restart = true;
                existing_file_size = static_cast<std::size_t>(file_size_li.QuadPart);
            } else {
                CloseHandle(file);
                file = INVALID_HANDLE_VALUE;
            }
        }

        if (!warm_restart) {
            // Create new file (overwrite if exists)
            file = CreateFileW(wpath.c_str(),
                GENERIC_READ | GENERIC_WRITE,
                FILE_SHARE_READ | FILE_SHARE_WRITE,
                nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
            if (file == INVALID_HANDLE_VALUE) {
                std::fprintf(stderr, "[slab_allocator] shared_memory_path: "
                             "CreateFileW failed for '%s' (error %lu)\n",
                             path.c_str(), GetLastError());
                return;
            }
            // Pre-allocate the file to the required size
            LARGE_INTEGER li;
            li.QuadPart = static_cast<LONGLONG>(total_file_size);
            if (!SetFilePointerEx(file, li, nullptr, FILE_BEGIN) || !SetEndOfFile(file)) {
                std::fprintf(stderr, "[slab_allocator] shared_memory_path: "
                             "SetFilePointerEx/SetEndOfFile failed (error %lu)\n",
                             GetLastError());
                CloseHandle(file);
                return;
            }
        }

        // Use the actual file size for mapping (existing file may differ)
        std::size_t map_size = warm_restart ? existing_file_size : total_file_size;

        // Create file mapping
        HANDLE mapping = CreateFileMappingW(file, nullptr, PAGE_READWRITE,
            static_cast<DWORD>(map_size >> 32),
            static_cast<DWORD>(map_size & 0xFFFFFFFFu),
            nullptr);
        if (!mapping) {
            std::fprintf(stderr, "[slab_allocator] shared_memory_path: "
                         "CreateFileMappingW failed (error %lu)\n", GetLastError());
            CloseHandle(file);
            return;
        }

        // Map view of the entire file
        void* base = MapViewOfFile(mapping, FILE_MAP_ALL_ACCESS, 0, 0, 0);
        if (!base) {
            std::fprintf(stderr, "[slab_allocator] shared_memory_path: "
                         "MapViewOfFile failed (error %lu)\n", GetLastError());
            CloseHandle(mapping);
            CloseHandle(file);
            return;
        }

        shared_mem_file_handle_ = file;
        shared_mem_mapping_handle_ = mapping;
        shared_mem_base_ = base;
        shared_mem_size_ = map_size;

#else
        // ---- Linux/POSIX: mmap with MAP_SHARED ----

        // Try to open existing file first (potential warm restart)
        int fd = open(path.c_str(), O_RDWR, 0666);

        if (fd >= 0) {
            // File exists — check size
            struct stat st;
            if (fstat(fd, &st) == 0 &&
                static_cast<std::size_t>(st.st_size) >= kSharedMemHeaderSize) {
                warm_restart = true;
                existing_file_size = static_cast<std::size_t>(st.st_size);
            } else {
                close(fd);
                fd = -1;
            }
        }

        if (!warm_restart) {
            // Create new file
            fd = open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0666);
            if (fd < 0) {
                std::fprintf(stderr, "[slab_allocator] shared_memory_path: "
                             "open(O_CREAT) failed for '%s' (errno %d)\n",
                             path.c_str(), errno);
                return;
            }
            // Pre-allocate the file to the required size
            if (ftruncate(fd, static_cast<off_t>(total_file_size)) != 0) {
                std::fprintf(stderr, "[slab_allocator] shared_memory_path: "
                             "ftruncate failed (errno %d)\n", errno);
                close(fd);
                return;
            }
        }

        // Use the actual file size for mapping
        std::size_t map_size = warm_restart ? existing_file_size : total_file_size;

        // Map the file with MAP_SHARED so writes are visible to other processes
        void* base = mmap(nullptr, map_size, PROT_READ | PROT_WRITE,
                          MAP_SHARED, fd, 0);
        if (base == MAP_FAILED) {
            std::fprintf(stderr, "[slab_allocator] shared_memory_path: "
                         "mmap failed (errno %d)\n", errno);
            close(fd);
            return;
        }

        shared_mem_fd_ = fd;
        shared_mem_base_ = base;
        shared_mem_size_ = map_size;
#endif

        // ---- Validate header on warm restart ----
        if (warm_restart) {
            auto* header = static_cast<shared_memory_header*>(shared_mem_base_);
            // T-P2-10: The header stores a single default slab_size. We
            // validate against the config default. Per-class sizes are an
            // advanced feature and a mismatch in per-class sizes does not
            // invalidate the warm restart (the slab layout is reconstructed
            // from the header's per-class counts).
            if (std::memcmp(header->magic, "LRUS", 4) != 0 ||
                header->version != 1 ||
                header->slab_size != static_cast<uint64_t>(default_slab_size)) {
                // Header mismatch — treat as fresh start
                std::fprintf(stderr, "[slab_allocator] shared_memory_path: "
                             "header validation failed — treating as fresh start\n");
                warm_restart = false;
            }
        }

        if (warm_restart) {
            // ---- Warm restart: rebuild allocator state from shared memory ----
            auto* header = static_cast<shared_memory_header*>(shared_mem_base_);
            is_warm_restart_ = true;

            // Set up slabs for each class from the mapped region.
            // The header stores per-class slab counts; slabs are laid out
            // sequentially: class 0's slabs first, then class 1's, etc.
            // T-P2-10: each class uses its own slab_size for the offset.
            std::size_t slab_offset = 0;
            for (uint32_t i = 0; i < kNumClasses; ++i) {
                uint32_t count = header->class_slab_counts[i];
                uint32_t cls_slab_size = slab_sizes_[i];
                for (uint32_t s = 0; s < count; ++s) {
                    void* slab_ptr = static_cast<char*>(shared_mem_base_)
                                   + kSharedMemHeaderSize + slab_offset;
                    all_slabs_.push_back(slab_ptr);
                    auto addr = reinterpret_cast<std::uintptr_t>(slab_ptr);
                    shared_mem_slabs_.insert(addr);
                    classes_[i].add_slab(slab_ptr);
                    slab_offset += cls_slab_size;
                }
            }
            shared_mem_next_slab_offset_ = slab_offset;
        } else {
            // ---- Fresh start: write header and allocate initial slabs ----
            // Zero the header area, then populate fields
            auto* header = static_cast<shared_memory_header*>(shared_mem_base_);
            std::memset(header, 0, kSharedMemHeaderSize);
            std::memcpy(header->magic, "LRUS", 4);
            header->version = 1;
            // T-P2-10: store the default slab_size in the header for
            // warm-restart validation. Per-class sizes are implicit.
            header->slab_size = static_cast<uint64_t>(default_slab_size);
            header->num_slabs = 0;
            header->total_size = static_cast<uint64_t>(shared_mem_size_);

            // Allocate initial slabs from the mapped region via add_slab_to_class
            for (uint32_t i = 0; i < kNumClasses; ++i) {
                for (uint32_t s = 0; s < config_.initial_slabs_per_class; ++s) {
                    add_slab_to_class(i);
                }
            }

            // update_shared_memory_header() is already called by add_slab_to_class
        }
    }

    /// Update the shared memory header with current slab counts.
    /// T-P2-10: Must be called under the relevant slab_mutexes_[idx]
    /// (or from the constructor).
    void update_shared_memory_header() {
        if (!shared_mem_base_) return;
        auto* header = static_cast<shared_memory_header*>(shared_mem_base_);
        uint64_t total = 0;
        for (uint32_t i = 0; i < kNumClasses; ++i) {
            header->class_slab_counts[i] = classes_[i].num_slabs();
            total += classes_[i].num_slabs();
        }
        header->num_slabs = total;
    }

    /// Clean up shared memory mapping and platform-specific handles.
    /// Called from the destructor.
    void cleanup_shared_memory() {
        if (!shared_mem_base_) return;

#if defined(_WIN32)
        UnmapViewOfFile(shared_mem_base_);
        if (shared_mem_mapping_handle_) {
            CloseHandle(shared_mem_mapping_handle_);
            shared_mem_mapping_handle_ = nullptr;
        }
        if (shared_mem_file_handle_ && shared_mem_file_handle_ != INVALID_HANDLE_VALUE) {
            CloseHandle(shared_mem_file_handle_);
            shared_mem_file_handle_ = nullptr;
        }
#else
        munmap(shared_mem_base_, shared_mem_size_);
        if (shared_mem_fd_ >= 0) {
            close(shared_mem_fd_);
            shared_mem_fd_ = -1;
        }
#endif
        shared_mem_base_ = nullptr;
        shared_mem_size_ = 0;
    }
};

// ============================================================================
// NUMA-Aware Slab Allocator
// ============================================================================

/// NUMA-aware slab allocator that routes allocations to the NUMA node
/// of the calling thread for reduced cross-node memory access latency.
///
/// This is a separate class from slab_allocator (not a replacement) so that
/// existing code remains unaffected. Enable via the LRU_HAS_NUMA compile macro
/// or by using this class directly.
///
/// Architecture:
///   - Maintains a per-NUMA-node slab_allocator instance
///   - On allocation, detects the calling thread's NUMA node and routes
///     to the corresponding allocator
///   - Falls back to a default allocator when NUMA detection is unavailable
///
/// Platform support:
///   - Windows: Uses GetCurrentProcessorNumberEx() + GetNumaProcessorNode()
///     to determine the current thread's NUMA node
///   - Linux: Uses getcpu() syscall (no libnuma dependency)
///   - Other platforms: Falls back to regular slab_allocator behavior
///
/// Usage:
///   lru::numa_aware_slab_allocator alloc;
///   void* ptr = alloc.allocate(128);
///   alloc.deallocate(ptr, 128);
class numa_aware_slab_allocator {
public:
    /// Configuration for the NUMA-aware slab allocator.
    struct config {
        slab_allocator::config slab_config;
        /// Maximum number of NUMA nodes to support.
        /// Defaults to 8, which covers most x86 servers.
        /// Nodes beyond this count fall back to node 0's allocator.
        int max_nodes = 8;
    };

    /// Default constructor — auto-detects NUMA topology.
    numa_aware_slab_allocator()
        : numa_aware_slab_allocator(config{}) {}

    /// Construct with the given configuration.
    explicit numa_aware_slab_allocator(const config& cfg)
        : config_(cfg)
        , num_nodes_(detect_num_nodes(cfg.max_nodes))
    {
        // Create per-node allocators with NUMA binding
        // Using unique_ptr because slab_allocator is non-copyable/non-movable
        auto node_cfg = cfg.slab_config;
        for (int node = 0; node < num_nodes_; ++node) {
            node_cfg.numa_node = node;
            node_allocators_.push_back(
                std::make_unique<slab_allocator>(node_cfg));
        }
    }

    // Non-copyable, non-movable
    numa_aware_slab_allocator(const numa_aware_slab_allocator&) = delete;
    numa_aware_slab_allocator& operator=(const numa_aware_slab_allocator&) = delete;

    // ----------------------------------------------------------------
    // Core API (matches slab_allocator)
    // ----------------------------------------------------------------

    /// Allocate a block of at least `size` bytes from the current
    /// thread's NUMA node allocator.
    void* allocate(std::size_t size) {
        return current_allocator().allocate(size);
    }

    /// Deallocate a block previously returned by `allocate(size)`.
    void deallocate(void* ptr, std::size_t size) noexcept {
        // Deallocate through the allocator that owns the slab containing ptr.
        // We try the current thread's allocator first (most common case),
        // then fall back to checking all nodes.
        auto& alloc = current_allocator();
        alloc.deallocate(ptr, size);
    }

    // ----------------------------------------------------------------
    // NUMA access
    // ----------------------------------------------------------------

    /// Return the NUMA node of the calling thread.
    /// Returns 0 if NUMA detection is unavailable.
    static int get_current_node() {
#if defined(LRU_HAS_WIN32_NUMA)
        // Windows: get the current processor number, then look up its NUMA node
        PROCESSOR_NUMBER proc_number;
        GetCurrentProcessorNumberEx(&proc_number);
        USHORT node_number = 0;
        if (GetNumaProcessorNodeEx(&proc_number, &node_number)) {
            return static_cast<int>(node_number);
        }
        return 0;
#elif defined(__linux__)
        // Linux: use getcpu() syscall (no libnuma dependency)
        unsigned int cpu = 0, node = 0;
        if (syscall(__NR_getcpu, &cpu, &node, nullptr) == 0) {
            return static_cast<int>(node);
        }
        return 0;
#else
        return 0;
#endif
    }

    /// Return the number of NUMA nodes detected.
    int num_nodes() const noexcept { return num_nodes_; }

    /// Access the per-node slab allocator for a given NUMA node.
    /// Falls back to node 0 if the node index is out of range.
    slab_allocator& node_allocator(int node) {
        if (node < 0 || node >= num_nodes_) node = 0;
        return *node_allocators_[static_cast<std::size_t>(node)];
    }

    const slab_allocator& node_allocator(int node) const {
        if (node < 0 || node >= num_nodes_) node = 0;
        return *node_allocators_[static_cast<std::size_t>(node)];
    }

    // ----------------------------------------------------------------
    // Size-class helpers (delegated to node 0's allocator)
    // ----------------------------------------------------------------

    static constexpr uint32_t kMinClassSize = slab_allocator::kMinClassSize;
    static constexpr uint32_t kMaxClassSize = slab_allocator::kMaxClassSize;
    static constexpr uint32_t kNumClasses = slab_allocator::kNumClasses;

    static uint32_t class_index_for(uint32_t size) {
        return slab_allocator::class_index_for(size);
    }

    // ----------------------------------------------------------------
    // Statistics
    // ----------------------------------------------------------------

    /// Per-class utilization statistics (aggregated across all NUMA nodes).
    std::vector<slab_allocator::allocation_class_stats> get_stats() const {
        // Aggregate stats from all nodes
        std::vector<slab_allocator::allocation_class_stats> result;
        result.reserve(kNumClasses);

        for (uint32_t i = 0; i < kNumClasses; ++i) {
            slab_allocator::allocation_class_stats agg{};
            agg.class_size = 0;
            agg.num_slabs = 0;
            agg.items_per_slab = 0;
            agg.utilization = 0.0;

            for (int node = 0; node < num_nodes_; ++node) {
                auto node_stats = node_allocators_[static_cast<std::size_t>(node)]->get_stats();
                const auto& cs = node_stats[i];
                if (agg.class_size == 0) {
                    agg.class_size = cs.class_size;
                    agg.items_per_slab = cs.items_per_slab;
                }
                agg.num_slabs += cs.num_slabs;
                // Weight utilization by number of slabs
                double weight = static_cast<double>(cs.num_slabs);
                agg.utilization += cs.utilization * weight;
            }
            if (agg.num_slabs > 0) {
                agg.utilization /= static_cast<double>(agg.num_slabs);
            }
            result.push_back(agg);
        }
        return result;
    }

    /// Returns true if any per-node allocator was initialized from an
    /// existing shared memory file (warm restart).
    bool is_warm_restart() const {
        for (const auto& alloc : node_allocators_) {
            if (alloc->is_warm_restart()) return true;
        }
        return false;
    }

private:
    config config_;
    int num_nodes_;
    std::vector<std::unique_ptr<slab_allocator>> node_allocators_;

    /// Get the slab allocator for the current thread's NUMA node.
    slab_allocator& current_allocator() {
        return node_allocator(get_current_node());
    }

    const slab_allocator& current_allocator() const {
        return node_allocator(get_current_node());
    }

    /// Detect the number of NUMA nodes on the system.
    /// Returns at least 1. Clamps to max_nodes.
    static int detect_num_nodes(int max_nodes) {
        int detected = 1;

#if defined(LRU_HAS_WIN32_NUMA)
        // Windows: use GetNumaHighestNodeNumber()
        ULONG highest_node = 0;
        if (GetNumaHighestNodeNumber(&highest_node)) {
            detected = static_cast<int>(highest_node) + 1;
        }
#elif defined(__linux__)
        // Linux: read from /sys/devices/system/node/possible
        // or fall back to counting online nodes
        FILE* f = std::fopen("/sys/devices/system/node/possible", "r");
        if (f) {
            char buf[128];
            if (std::fgets(buf, sizeof(buf), f)) {
                // Parse "0-N" format
                int lo = 0, hi = 0;
                if (std::sscanf(buf, "%d-%d", &lo, &hi) == 2 && hi >= lo) {
                    detected = hi - lo + 1;
                }
            }
            std::fclose(f);
        }
#endif

        // Clamp to max_nodes
        if (detected > max_nodes) detected = max_nodes;
        if (detected < 1) detected = 1;
        return detected;
    }
};

// ============================================================================
// Compile-time NUMA allocator selection
// ============================================================================

/// Type alias that selects NUMA-aware allocator when LRU_HAS_NUMA is defined,
/// or falls back to regular slab_allocator otherwise.
///
/// Usage:
///   lru::default_slab_allocator<> alloc;
///   void* ptr = alloc.allocate(128);
#if defined(LRU_HAS_NUMA)
using default_slab_allocator = numa_aware_slab_allocator;
#else
using default_slab_allocator = slab_allocator;
#endif

} // namespace lru

// Definitions of cache_item's slab-aware operator new/delete.  They live here
// because they need the full definition of slab_allocator, while
// detail/intrusive_list.hpp only forward-declares it.
#include "detail/intrusive_list.hpp"

namespace lru::detail {

template <typename Key, typename Value, typename Hook>
void* cache_item<Key, Value, Hook>::operator new(std::size_t sz) {
    // cache_item is over-aligned (refcount_with_flags has alignas(64) to
    // prevent false sharing on the hot refcount word). The default global
    // ::operator new(size_t) only guarantees alignof(std::max_align_t)
    // (typically 8 or 16 bytes), which is insufficient — constructing a
    // 64-byte-aligned object on a misaligned address is undefined behavior
    // (flagged by UBSan's alignment check). Use the C++17 sized-aligned
    // operator new so the runtime returns a properly aligned block.
    return ::operator new(sz, std::align_val_t(alignof(cache_item)));
}

template <typename Key, typename Value, typename Hook>
void* cache_item<Key, Value, Hook>::operator new(std::size_t sz, slab_allocator* alloc) {
    if (!alloc) throw std::bad_alloc{};
    void* mem = alloc->allocate(sz);
    if (!mem) throw std::bad_alloc{};
    return mem;
}

template <typename Key, typename Value, typename Hook>
void cache_item<Key, Value, Hook>::operator delete(void* p, std::size_t sz) {
    if (!p) return;
    auto* self = static_cast<cache_item*>(p);
    if (self->allocator_) {
        self->allocator_->deallocate(p, sz);
    } else {
        // Matching aligned deallocation for the C++17 aligned operator new
        // used above. Without std::align_val_t, MSVC/UCRT may route the
        // free through the wrong pool (or abort on mismatched alignment).
        ::operator delete(p, std::align_val_t(alignof(cache_item)));
    }
}

template <typename Key, typename Value, typename Hook>
void cache_item<Key, Value, Hook>::operator delete(void* p) {
    if (!p) return;
    auto* self = static_cast<cache_item*>(p);
    if (self->allocator_) {
        self->allocator_->deallocate(p, sizeof(cache_item));
    } else {
        ::operator delete(p, std::align_val_t(alignof(cache_item)));
    }
}

template <typename Key, typename Value, typename Hook>
void cache_item<Key, Value, Hook>::operator delete(void* p, std::size_t sz, slab_allocator* alloc) {
    if (p) alloc->deallocate(p, sz);
}

template <typename Key, typename Value, typename Hook>
void cache_item<Key, Value, Hook>::operator delete(void* p, slab_allocator* alloc) {
    if (p) alloc->deallocate(p, sizeof(cache_item));
}

} // namespace lru::detail

#endif // LRU_MEMORY_HPP
