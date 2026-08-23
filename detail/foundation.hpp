// Unified LRU Cache Library - Internal Utilities & Infrastructure
// Merged from: utils.hpp, periodic_worker.hpp, striped_mutex.hpp
// SPDX-License-Identifier: MIT

#ifndef LRU_DETAIL_FOUNDATION_HPP
#define LRU_DETAIL_FOUNDATION_HPP

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <exception>
#include <format>
#include <functional>
#include <memory>
#include <mutex>
#include <new>
#include <shared_mutex>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

namespace lru::detail {

// ============================================================================
// Forward declaration: fairness_mode (defined in detail/distributed_mutex.hpp)
// ============================================================================
//
// Forward-declared here so that striped_mutex<> can reference fairness_mode
// in its set_fairness_mode()/get_fairness_mode() forwarding methods without
// requiring foundation.hpp to include distributed_mutex.hpp (which would
// create a header dependency cycle: distributed_mutex.hpp includes
// native_wait_ops.hpp only, and cache_trait.hpp includes both headers).
enum class fairness_mode;

// ============================================================================
// Type trait helpers
// ============================================================================

/// Check if type is formattable with std::format.
template <typename T, typename = void>
struct is_formattable : std::false_type {};

template <typename T>
struct is_formattable<T, std::void_t<decltype(std::formatter<std::remove_cv_t<T>, char>{})>> : std::true_type {};

template <typename T>
inline constexpr bool is_formattable_v = is_formattable<T>::value;

// ============================================================================
// Type trait: is_vector
// ============================================================================

/// Check if a type is std::vector<T, Alloc>.
template <typename T>
struct is_vector : std::false_type {};

template <typename T, typename Alloc>
struct is_vector<std::vector<T, Alloc>> : std::true_type {};

template <typename T>
inline constexpr bool is_vector_v = is_vector<T>::value;

// ============================================================================
// String formatting helpers
// ============================================================================

/// Format a key-value pair for debugging output.
template <typename Key, typename Value>
std::string format_item(const std::pair<Key, Value>& item, std::size_t index) {
    std::string key_str;
    std::string val_str;

    if constexpr (is_formattable_v<Key>) {
        key_str = std::format("{}", item.first);
    } else {
        key_str = std::format("<key at {:#x}>", reinterpret_cast<std::uintptr_t>(&item.first));
    }

    if constexpr (is_formattable_v<Value>) {
        val_str = std::format("{}", item.second);
    } else {
        val_str = std::format("<val at {:#x}>", reinterpret_cast<std::uintptr_t>(&item.second));
    }

    return std::format("{}: [{}] = '{}'", index, key_str, val_str);
}

// ============================================================================
// Integer sequence helpers
// ============================================================================

/// Construct an object from a tuple using index sequence.
template <typename T, typename Tuple, std::size_t... Is>
T construct_from_tuple_impl(Tuple&& tuple, std::index_sequence<Is...>) {
    return T(std::get<Is>(std::forward<Tuple>(tuple))...);
}

template <typename T, typename Tuple>
T construct_from_tuple(Tuple&& tuple) {
    return construct_from_tuple_impl<T>(
        std::forward<Tuple>(tuple),
        std::make_index_sequence<std::tuple_size_v<std::remove_reference_t<Tuple>>>{});
}

// ============================================================================
// Memory helpers
// ============================================================================

/// RAII helper to execute a function on scope exit.
template <typename F>
class scope_exit {
public:
    explicit scope_exit(F&& f) : func_(std::move(f)), active_(true) {}
    explicit scope_exit(const F& f) : func_(f), active_(true) {}

    scope_exit(scope_exit&& other) noexcept
        : func_(std::move(other.func_)), active_(other.active_) {
        other.active_ = false;
    }

    scope_exit(const scope_exit&) = delete;
    scope_exit& operator=(const scope_exit&) = delete;
    scope_exit& operator=(scope_exit&&) = delete;

    ~scope_exit() {
        if (active_) {
            func_();
        }
    }

    void release() noexcept { active_ = false; }

private:
    F func_;
    bool active_;
};

template <typename F>
scope_exit<F> make_scope_exit(F&& f) {
    return scope_exit<F>(std::forward<F>(f));
}

// ============================================================================
// SeqLock — Sequence lock for read-heavy scenarios
// ============================================================================

/// A sequence lock optimized for read-heavy workloads where writers are rare.
/// Readers retry if a write was in progress during their read.
/// Writers acquire exclusive access by incrementing the sequence to an odd
/// value, performing the write, then incrementing to the next even value.
///
/// This is ideal for cache_stats::consistent_snapshot() where:
///   - All counter updates are atomic (lock-free writes)
///   - Snapshots need consistent reads across multiple counters
///   - Writes to the snapshot lock are rare (only during reset_counters)
///
/// Based on the Linux kernel seqlock design.
class seqlock {
public:
    seqlock() : seq_(0) {}

    /// Begin a read section. Returns the current sequence number.
    /// If the sequence is odd, a write is in progress — spin until even.
    std::uint32_t read_begin() const noexcept {
        std::uint32_t s;
        do {
            s = seq_.load(std::memory_order_acquire);
        } while (s & 1);  // Wait until no writer
        return s;
    }

    /// Check if a read section needs to be retried.
    /// Returns true if a write occurred during the read section.
    bool read_retry(std::uint32_t start_seq) const noexcept {
        std::atomic_thread_fence(std::memory_order_acquire);
        return seq_.load(std::memory_order_relaxed) != start_seq;
    }

    /// Acquire write access. Increments sequence to odd value.
    void write_lock() noexcept {
        std::uint32_t s = seq_.fetch_add(1, std::memory_order_acq_rel);
        // s must be even — no nested writes allowed
        (void)s;
    }

    /// Release write access. Increments sequence to even value.
    void write_unlock() noexcept {
        seq_.fetch_add(1, std::memory_order_release);
    }

    /// RAII write guard
    class write_guard {
    public:
        explicit write_guard(seqlock& sl) : sl_(sl) { sl_.write_lock(); }
        ~write_guard() { sl_.write_unlock(); }
        write_guard(const write_guard&) = delete;
        write_guard& operator=(const write_guard&) = delete;
    private:
        seqlock& sl_;
    };

private:
    std::atomic<std::uint32_t> seq_;
};

// ============================================================================
// Striped Mutex
// ============================================================================

// cacheline 大小：固定为 64 以避免 std::hardware_destructive_interference_size 的 ABI
// 警告（其值可能随编译器版本或 -mtune 变化）。x86/ARM 常规 cacheline 均为 64。
inline constexpr std::size_t kCachelineSize = 64;

// 每个 stripe 的 mutex 独占一个 cacheline，消除多核 false sharing。
// 对齐 CacheLib 的 folly::cacheline_aligned<Mutex> 用法（MMLru.h:474）。
template <typename MutexType>
struct alignas(kCachelineSize) AlignedMutex {
    MutexType mutex;
};

// StripedMutex: divides a shared resource into stripes (buckets),
// each with its own mutex, allowing concurrent access to different stripes.
// Inspired by CacheLib's ChainedHashTable bucket mutexes.
template <typename MutexType = std::shared_mutex>
class striped_mutex {
public:
    explicit striped_mutex(std::size_t num_stripes = 64)
        : stripes_(num_stripes) {
        if (num_stripes == 0) {
            throw std::invalid_argument("striped_mutex: num_stripes must be > 0");
        }
    }

    // Get the stripe index for a key
    std::size_t stripe_for(std::size_t hash) const noexcept {
        return hash % stripes_.size();
    }

    // Lock a specific stripe for exclusive access
    void lock(std::size_t stripe) {
        stripes_[stripe].mutex.lock();
    }

    void unlock(std::size_t stripe) {
        stripes_[stripe].mutex.unlock();
    }

    // Lock a specific stripe for shared access (for shared_mutex)
    void lock_shared(std::size_t stripe) {
        stripes_[stripe].mutex.lock_shared();
    }

    void unlock_shared(std::size_t stripe) {
        stripes_[stripe].mutex.unlock_shared();
    }

    // Try to lock a stripe (non-blocking)
    bool try_lock(std::size_t stripe) {
        return stripes_[stripe].mutex.try_lock();
    }

    bool try_lock_shared(std::size_t stripe) {
        return stripes_[stripe].mutex.try_lock_shared();
    }

    // RAII lock guards for a specific stripe
    auto make_unique_lock(std::size_t stripe) {
        return std::unique_lock<MutexType>(stripes_[stripe].mutex);
    }

    auto make_shared_lock(std::size_t stripe) {
        return std::shared_lock<MutexType>(stripes_[stripe].mutex);
    }

    // Try to exclusively lock a stripe (non-blocking). Returns the lock if successful.
    auto try_make_unique_lock(std::size_t stripe) {
        return std::unique_lock<MutexType>(stripes_[stripe].mutex, std::try_to_lock);
    }

    // T-G1: Try to shared-lock a stripe (non-blocking). Used by the
    // value-layer TTL scanner to avoid blocking writers.
    auto try_make_shared_lock(std::size_t stripe) {
        return std::shared_lock<MutexType>(stripes_[stripe].mutex, std::try_to_lock);
    }

    std::size_t size() const noexcept { return stripes_.size(); }

    // P2-2: Access a specific stripe's mutex by index. Used for runtime
    // configuration (e.g., set_lock_order_checking, set_fairness_mode).
    MutexType& mutex_at(std::size_t stripe) { return stripes_[stripe].mutex; }
    const MutexType& mutex_at(std::size_t stripe) const { return stripes_[stripe].mutex; }

    // Lock all stripes (for operations that need exclusive global access).
    // 异常安全：若某个 stripe 的 lock 抛异常，回滚所有已锁定的 stripe，避免死锁。
    // 仅异常路径有额外开销，正常路径无性能损失。
    void lock_all() {
        std::size_t locked = 0;
        try {
            for (; locked < stripes_.size(); ++locked) {
                stripes_[locked].mutex.lock();
            }
        } catch (...) {
            for (std::size_t i = 0; i < locked; ++i) {
                stripes_[i].mutex.unlock();
            }
            throw;
        }
    }

    void unlock_all() {
        for (auto& m : stripes_) m.mutex.unlock();
    }

    // Shared lock all stripes (for global read operations).
    // Exception-safe: rolls back on failure.
    void lock_shared_all() {
        std::size_t locked = 0;
        try {
            for (; locked < stripes_.size(); ++locked) {
                stripes_[locked].mutex.lock_shared();
            }
        } catch (...) {
            for (std::size_t i = 0; i < locked; ++i) {
                stripes_[i].mutex.unlock_shared();
            }
            throw;
        }
    }

    void unlock_shared_all() {
        for (auto& m : stripes_) m.mutex.unlock_shared();
    }

    // ----------------------------------------------------------------
    // Fairness mode forwarding (only for distributed_shared_mutex)
    // ----------------------------------------------------------------

    /// Set the fairness mode on every stripe (no-op for mutex types
    /// that do not support fairness_mode, e.g., std::mutex).
    template <typename M = MutexType>
    auto set_fairness_mode(fairness_mode mode)
        -> decltype(std::declval<M&>().set_fairness_mode(mode), void())
    {
        for (auto& a : stripes_) a.mutex.set_fairness_mode(mode);
    }

    /// Query the fairness mode of the first stripe. Returns
    /// reader_preferred for mutex types that do not support fairness.
    template <typename M = MutexType>
    auto get_fairness_mode() const
        -> decltype(std::declval<const M&>().get_fairness_mode())
    {
        return stripes_[0].mutex.get_fairness_mode();
    }

    // ----------------------------------------------------------------
    // P1-1: Writer starvation detector forwarding
    // ----------------------------------------------------------------

    /// P1-1: Set the writer starvation timeout on every stripe.
    template <typename M = MutexType>
    auto set_writer_starvation_timeout(uint64_t timeout_ns)
        -> decltype(std::declval<M&>().set_writer_starvation_timeout(timeout_ns), void())
    {
        for (auto& a : stripes_) a.mutex.set_writer_starvation_timeout(timeout_ns);
    }

    /// P1-1: Aggregate writer_starvation_events across all stripes.
    template <typename M = MutexType>
    auto writer_starvation_events() const
        -> decltype(std::declval<const M&>().writer_starvation_events())
    {
        std::size_t total = 0;
        for (auto& a : stripes_) total += a.mutex.writer_starvation_events();
        return total;
    }

    /// P1-1: Maximum writer_max_wait_ns across all stripes.
    template <typename M = MutexType>
    auto writer_max_wait_ns() const
        -> decltype(std::declval<const M&>().writer_max_wait_ns())
    {
        uint64_t max_ns = 0;
        for (auto& a : stripes_) {
            uint64_t v = a.mutex.writer_max_wait_ns();
            if (v > max_ns) max_ns = v;
        }
        return max_ns;
    }

    /// P1-1: Reset writer_max_wait_ns on every stripe.
    template <typename M = MutexType>
    auto reset_writer_max_wait_ns()
        -> decltype(std::declval<M&>().reset_writer_max_wait_ns(), void())
    {
        for (auto& a : stripes_) a.mutex.reset_writer_max_wait_ns();
    }

    /// P1-5 (T1.5): Quiescent variant — acquires all stripes exclusively
    /// (draining all in-flight readers), then atomically switches every
    /// stripe's fairness mode, then releases. Guarantees no in-flight
    /// operation observes a mode change mid-critical-section.
    /// Returns true if all stripes were switched; false if the timeout
    /// expired before all stripes could be acquired (in which case the
    /// mode is unchanged on all stripes).
    template <typename M = MutexType>
    auto set_fairness_mode_quiescent(
            fairness_mode mode,
            std::chrono::milliseconds timeout = std::chrono::seconds(5))
        -> decltype(std::declval<M&>().try_lock(), bool())
    {
        // Polling-based acquisition with timeout. We try to acquire all
        // stripes via try_lock(); if any stripe is busy, we release all
        // acquired ones and retry after a short yield. This bounds the
        // worst-case wait and avoids holding partial state indefinitely.
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        while (std::chrono::steady_clock::now() < deadline) {
            // Try to acquire all stripes.
            std::size_t acquired = 0;
            bool all_acquired = true;
            try {
                for (; acquired < stripes_.size(); ++acquired) {
                    if (!stripes_[acquired].mutex.try_lock()) {
                        all_acquired = false;
                        break;
                    }
                }
            } catch (...) {
                all_acquired = false;
            }
            if (all_acquired) {
                // We hold all stripes exclusively — no readers or
                // writers in flight. Safe to switch all fairness modes
                // atomically. The stores are release-ordered so that
                // subsequent lock acquisitions in other threads observe
                // the new mode before they observe state_ == 0.
                // Use set_fairness_mode_locked() (no state_ assertion)
                // because we DO hold the write lock — state_ == kWriterFlag.
                for (auto& a : stripes_) a.mutex.set_fairness_mode_locked(mode);
                for (auto& a : stripes_) a.mutex.unlock();
                return true;
            }
            // Release any partially-acquired stripes and retry.
            for (std::size_t i = 0; i < acquired; ++i) {
                stripes_[i].mutex.unlock();
            }
            std::this_thread::yield();
        }
        return false;
    }

    // ----------------------------------------------------------------
    // T-B2 (P0-1-补): NUMA-aware reader counter forwarding
    // ----------------------------------------------------------------
    //
    // Forward set_numa_aware / numa_aware / num_numa_nodes to every
    // stripe's mutex. SFINAE-gated via decltype so plain std::mutex
    // (no NUMA support) silently skips these — the std::mutex
    // specialization of striped_mutex doesn't even instantiate these
    // templates.
    //
    // Multi-socket NUMA routing benefit: each stripe's reader counter
    // stays within a single socket's L3 cache, eliminating cross-socket
    // MSI traffic. See distributed_shared_mutex::set_numa_aware for
    // the per-stripe implementation.

    /// Enable or disable NUMA-aware reader counter routing on every
    /// stripe. No-op for mutex types that don't support NUMA routing.
    template <typename M = MutexType>
    auto set_numa_aware(bool enabled)
        -> decltype(std::declval<M&>().set_numa_aware(enabled), void())
    {
        for (auto& a : stripes_) a.mutex.set_numa_aware(enabled);
    }

    /// Query whether NUMA-aware routing is enabled (returns the first
    /// stripe's state — all stripes are toggled together via the setter
    /// above, so they should always agree).
    template <typename M = MutexType>
    auto numa_aware() const
        -> decltype(std::declval<const M&>().numa_aware())
    {
        return stripes_[0].mutex.numa_aware();
    }

    /// Return the number of NUMA nodes detected on the system.
    /// Delegates to the first stripe's static probe — the result is
    /// system-wide, not per-stripe.
    template <typename M = MutexType>
    static auto num_numa_nodes()
        -> decltype(M::num_numa_nodes())
    {
        return M::num_numa_nodes();
    }

private:
    std::vector<AlignedMutex<MutexType>> stripes_;
};

// Specialization for plain mutex (no shared lock support)
template <>
class striped_mutex<std::mutex> {
public:
    explicit striped_mutex(std::size_t num_stripes = 64)
        : stripes_(num_stripes) {
        if (num_stripes == 0) {
            throw std::invalid_argument("striped_mutex: num_stripes must be > 0");
        }
    }

    std::size_t stripe_for(std::size_t hash) const noexcept {
        return hash % stripes_.size();
    }

    void lock(std::size_t stripe) { stripes_[stripe].mutex.lock(); }
    void unlock(std::size_t stripe) { stripes_[stripe].mutex.unlock(); }
    bool try_lock(std::size_t stripe) { return stripes_[stripe].mutex.try_lock(); }

    auto make_unique_lock(std::size_t stripe) {
        return std::unique_lock<std::mutex>(stripes_[stripe].mutex);
    }

    // Try to exclusively lock a stripe (non-blocking). Returns the lock if successful.
    auto try_make_unique_lock(std::size_t stripe) {
        return std::unique_lock<std::mutex>(stripes_[stripe].mutex, std::try_to_lock);
    }

    /// 对于 std::mutex 特化，共享锁等价于独占锁（因为没有读-读并发需求）。
    auto make_shared_lock(std::size_t stripe) {
        return make_unique_lock(stripe);
    }

    std::size_t size() const noexcept { return stripes_.size(); }

    // P2-2: Access a specific stripe's mutex by index.
    std::mutex& mutex_at(std::size_t stripe) { return stripes_[stripe].mutex; }
    const std::mutex& mutex_at(std::size_t stripe) const { return stripes_[stripe].mutex; }

    // 异常安全：若某个 stripe 的 lock 抛异常，回滚所有已锁定的 stripe，避免死锁。
    // 仅异常路径有额外开销，正常路径无性能损失。
    void lock_all() {
        std::size_t locked = 0;
        try {
            for (; locked < stripes_.size(); ++locked) {
                stripes_[locked].mutex.lock();
            }
        } catch (...) {
            for (std::size_t i = 0; i < locked; ++i) {
                stripes_[i].mutex.unlock();
            }
            throw;
        }
    }

    void unlock_all() { for (auto& m : stripes_) m.mutex.unlock(); }

    // 对于 std::mutex 特化，shared lock 等价于 exclusive lock（无读-读并发语义）。
    // 提供 lock_shared_all/unlock_shared_all 以便模板代码（striped_read_lock_all、
    // striped_mutex_read_all_guard 等）无需特化即可统一调用。
    void lock_shared_all() { lock_all(); }
    void unlock_shared_all() { unlock_all(); }

private:
    std::vector<AlignedMutex<std::mutex>> stripes_;
};

// ============================================================================
// T-P3-5: Lazy-allocated striped_mutex wrapper
// ============================================================================
//
// For striped caches backed by sharded_mm_lru (which has its own per-shard
// locks), the striped_mutex_ is ONLY used for global operations (clear,
// snapshot, flush, etc.) — never for per-key operations.  Allocating 64
// distributed_shared_mutex objects eagerly in every constructor wastes
// memory (each distributed_shared_mutex has internal atomics, wait
// primitives, and latency histograms) for caches that may never perform a
// global operation.
//
// lazy_striped_mutex wraps a std::unique_ptr<striped_mutex<MutexType>> and
// defers construction until the first call to a method that actually needs
// the underlying mutexes.  The lightweight methods stripe_for() and size()
// are computed directly from the stored num_stripes_ without allocation.
//
// Thread safety: lazy initialization uses std::call_once, guaranteeing
// exactly one allocation even under concurrent first-access from multiple
// threads.  After initialization, all calls are lock-free (just a pointer
// dereference).
//
// Interface: mirrors striped_mutex<MutexType> exactly, so it can be used as
// a drop-in replacement via the lock_policy::striped_mutex_type alias.

template <typename MutexType>
class lazy_striped_mutex {
public:
    explicit lazy_striped_mutex(std::size_t num_stripes = 64)
        : num_stripes_(num_stripes) {
        if (num_stripes == 0) {
            throw std::invalid_argument("lazy_striped_mutex: num_stripes must be > 0");
        }
    }

    // ----------------------------------------------------------------
    // Non-allocating methods: computed from num_stripes_ directly.
    // These are the hot-path methods used by stripe_for() lookups in
    // per-key operations — no allocation, no atomic, no once_flag.
    // ----------------------------------------------------------------

    std::size_t stripe_for(std::size_t hash) const noexcept {
        return hash % num_stripes_;
    }

    std::size_t size() const noexcept { return num_stripes_; }

    // ----------------------------------------------------------------
    // Allocating methods: lazily construct the inner striped_mutex on
    // first call.  After the first call, subsequent calls just
    // dereference the unique_ptr (lock-free).
    // ----------------------------------------------------------------

    void lock(std::size_t stripe) { ensure().lock(stripe); }
    void unlock(std::size_t stripe) { ensure().unlock(stripe); }

    void lock_shared(std::size_t stripe) { ensure().lock_shared(stripe); }
    void unlock_shared(std::size_t stripe) { ensure().unlock_shared(stripe); }

    bool try_lock(std::size_t stripe) { return ensure().try_lock(stripe); }
    bool try_lock_shared(std::size_t stripe) { return ensure().try_lock_shared(stripe); }

    auto make_unique_lock(std::size_t stripe) {
        return ensure().make_unique_lock(stripe);
    }

    auto make_shared_lock(std::size_t stripe) {
        return ensure().make_shared_lock(stripe);
    }

    auto try_make_unique_lock(std::size_t stripe) {
        return ensure().try_make_unique_lock(stripe);
    }

    MutexType& mutex_at(std::size_t stripe) { return ensure().mutex_at(stripe); }
    const MutexType& mutex_at(std::size_t stripe) const { return ensure().mutex_at(stripe); }

    // Global lock/unlock — these are the primary triggers for lazy
    // allocation, since global operations (clear, flush, snapshot) are
    // the main consumers of striped_mutex_ when per-shard locks exist.
    void lock_all() { ensure().lock_all(); }
    void unlock_all() { ensure().unlock_all(); }
    void lock_shared_all() { ensure().lock_shared_all(); }
    void unlock_shared_all() { ensure().unlock_shared_all(); }

    // ----------------------------------------------------------------
    // SFINAE-gated forwarding for fairness / NUMA configuration.
    // These are rare administrative operations; lazy allocation here
    // is acceptable.  The SFINAE pattern mirrors striped_mutex<> so
    // that `requires` checks in cache_trait.hpp work unchanged.
    // ----------------------------------------------------------------

    template <typename M = MutexType>
    auto set_fairness_mode(fairness_mode mode)
        -> decltype(std::declval<M&>().set_fairness_mode(mode), void())
    {
        ensure().set_fairness_mode(mode);
    }

    template <typename M = MutexType>
    auto set_fairness_mode_quiescent(
            fairness_mode mode,
            std::chrono::milliseconds timeout = std::chrono::seconds(5))
        -> decltype(std::declval<M&>().try_lock(), bool())
    {
        return ensure().set_fairness_mode_quiescent(mode, timeout);
    }

    template <typename M = MutexType>
    auto get_fairness_mode() const
        -> decltype(std::declval<const M&>().get_fairness_mode())
    {
        return ensure().get_fairness_mode();
    }

    // P1-1: Writer starvation detector forwarding (lazy variant)
    template <typename M = MutexType>
    auto set_writer_starvation_timeout(uint64_t timeout_ns)
        -> decltype(std::declval<M&>().set_writer_starvation_timeout(timeout_ns), void())
    {
        ensure().set_writer_starvation_timeout(timeout_ns);
    }

    template <typename M = MutexType>
    auto writer_starvation_events() const
        -> decltype(std::declval<const M&>().writer_starvation_events())
    {
        return ensure().writer_starvation_events();
    }

    template <typename M = MutexType>
    auto writer_max_wait_ns() const
        -> decltype(std::declval<const M&>().writer_max_wait_ns())
    {
        return ensure().writer_max_wait_ns();
    }

    template <typename M = MutexType>
    auto reset_writer_max_wait_ns()
        -> decltype(std::declval<M&>().reset_writer_max_wait_ns(), void())
    {
        ensure().reset_writer_max_wait_ns();
    }

    template <typename M = MutexType>
    auto set_numa_aware(bool enabled)
        -> decltype(std::declval<M&>().set_numa_aware(enabled), void())
    {
        ensure().set_numa_aware(enabled);
    }

    template <typename M = MutexType>
    auto numa_aware() const
        -> decltype(std::declval<const M&>().numa_aware())
    {
        return ensure().numa_aware();
    }

    template <typename M = MutexType>
    static auto num_numa_nodes()
        -> decltype(M::num_numa_nodes())
    {
        return MutexType::num_numa_nodes();
    }

private:
    /// Lazily allocate the inner striped_mutex on first access.
    /// Uses std::call_once for thread-safe initialization.
    striped_mutex<MutexType>& ensure() const {
        std::call_once(once_, [this] {
            impl_ = std::make_unique<striped_mutex<MutexType>>(num_stripes_);
        });
        return *impl_;
    }

    std::size_t num_stripes_;
    mutable std::once_flag once_;
    mutable std::unique_ptr<striped_mutex<MutexType>> impl_;
};

// ============================================================================
// Striped Mutex Global Lock Guards
// ============================================================================

/// RAII guard that exclusively locks all stripes of a striped_mutex.
/// Used for global write operations (clear_expired, flush, etc.).
template <typename MutexType>
struct striped_mutex_write_all_guard {
    striped_mutex<MutexType>& sm;
    explicit striped_mutex_write_all_guard(striped_mutex<MutexType>& sm) : sm(sm) { sm.lock_all(); }
    ~striped_mutex_write_all_guard() { sm.unlock_all(); }
    striped_mutex_write_all_guard(const striped_mutex_write_all_guard&) = delete;
    striped_mutex_write_all_guard& operator=(const striped_mutex_write_all_guard&) = delete;
};

/// RAII guard that shared-locks all stripes of a striped_mutex.
/// Used for global read operations (size, empty, stats, etc.).
template <typename MutexType>
struct striped_mutex_read_all_guard {
    striped_mutex<MutexType>& sm;
    explicit striped_mutex_read_all_guard(striped_mutex<MutexType>& sm) : sm(sm) { sm.lock_shared_all(); }
    ~striped_mutex_read_all_guard() { sm.unlock_shared_all(); }
    striped_mutex_read_all_guard(const striped_mutex_read_all_guard&) = delete;
    striped_mutex_read_all_guard& operator=(const striped_mutex_read_all_guard&) = delete;
};

// ============================================================================
// Periodic Worker
// ============================================================================

// PeriodicWorker: base class for background tasks that run at regular intervals.
// Inspired by CacheLib's PeriodicWorker used for Reaper, PoolRebalancer, etc.
// Provides graceful stop with condition variable wake-up.
class periodic_worker {
public:
    explicit periodic_worker(std::function<void()> task,
                             std::chrono::milliseconds interval)
        : task_(std::move(task))
        , interval_(interval)
        , running_(true) {
        thread_ = std::thread([this] { run(); });
    }

    virtual ~periodic_worker() noexcept {
        try {
            stop();
        } catch (...) {
            // Suppress all exceptions in destructor — the run() loop already
            // catches task exceptions via error_handler_.  Exceptions from
            // stop() (e.g., thread join failure) must not propagate during
            // stack unwinding.
        }
    }

    void stop() {
        bool expected = true;
        if (running_.compare_exchange_strong(expected, false)) {
            cv_.notify_all();
            if (thread_.joinable()) {
                thread_.join();
            }
        }
    }

    bool is_running() const noexcept { return running_.load(); }

    // Change the interval dynamically
    void set_interval(std::chrono::milliseconds new_interval) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            interval_ = new_interval;
        }
        cv_.notify_all();
    }

    void on_error(std::function<void(std::exception_ptr)> handler) {
        error_handler_ = std::move(handler);
    }

private:
    void run() {
        std::unique_lock<std::mutex> lock(mutex_);
        while (running_.load()) {
            if (cv_.wait_for(lock, interval_, [this] { return !running_.load(); })) {
                break; // stopped
            }
            if (running_.load()) {
                lock.unlock();
                try {
                    task_();
                } catch (...) {
                    if (error_handler_) {
                        error_handler_(std::current_exception());
                    }
                }
                lock.lock();
            }
        }
    }

    std::function<void()> task_;
    std::function<void(std::exception_ptr)> error_handler_;
    std::chrono::milliseconds interval_;
    std::atomic<bool> running_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::thread thread_;
};

// ============================================================================
// Locked Iterator Guard — 消除四个 MM 类型中 LockedIterator 的重复代码
// ============================================================================

/// 管理 LockedIterator 的锁生命周期和 active flag。
/// 每个 MM 类型的 LockedIterator 通过组合此守卫 + 队列遍历逻辑实现。
///
/// 用法（在 MM type 的 LockedIterator 中）：
///   class LockedIterator {
///       locked_iterator_guard guard_;
///       // ... 专有遍历逻辑
///   public:
///       LockedIterator(MMType& mm)
///           : guard_(mm.update_mutex_.m, mm.iterator_active_) {}
///       void destroy() { guard_.destroy(); }
///       // ...
///   };
template <typename Mutex = std::mutex>
class locked_iterator_guard {
public:
    /// 构造时锁定 mutex 并检查 active flag。
    /// 若 iterator_active 已为 true，抛出 runtime_error。
    /// @param m             MM 层的 update_mutex
    /// @param active_flag   MM 层的 iterator_active_ 原子标记
    locked_iterator_guard(Mutex& m, std::atomic<bool>& active_flag)
        : lock_(m), active_flag_(&active_flag) {
        if (active_flag_->exchange(true)) {
            lock_.unlock();
            throw std::runtime_error("LockedIterator already active");
        }
    }

    ~locked_iterator_guard() { destroy(); }

    locked_iterator_guard(const locked_iterator_guard&) = delete;
    locked_iterator_guard& operator=(const locked_iterator_guard&) = delete;

    locked_iterator_guard(locked_iterator_guard&& other) noexcept
        : lock_(std::move(other.lock_))
        , active_flag_(other.active_flag_)
        , valid_(other.valid_) {
        other.valid_ = false;
    }

    /// 释放锁并清除 active flag。
    void destroy() noexcept {
        if (valid_) {
            if (active_flag_) active_flag_->store(false, std::memory_order_release);
            valid_ = false;
            if (lock_.owns_lock()) lock_.unlock();
        }
    }

    /// 返回底层的 unique_lock 引用（允许手动 unlock 等操作）。
    std::unique_lock<Mutex>& lock() noexcept { return lock_; }

private:
    std::unique_lock<Mutex> lock_;
    std::atomic<bool>* active_flag_ = nullptr;
    bool valid_ = true;
};

// ============================================================================
// TLS shared_ptr cache — per-thread single-entry cache for get_shared_cached()
// ============================================================================

/// Thread-local cache holding at most one (key, shared_ptr) pair.
/// Used by unified_cache::get_shared_cached() to avoid repeated heap
/// allocations when the same key is accessed consecutively from the
/// same thread.
template <typename Key, typename Value>
struct tls_shared_cache {
    static tls_shared_cache& instance() {
        thread_local tls_shared_cache cache;
        return cache;
    }

    Key key{};
    std::shared_ptr<Value> ptr;
};

} // namespace lru::detail

#endif // LRU_DETAIL_FOUNDATION_HPP
