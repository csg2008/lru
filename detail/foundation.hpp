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

#include "atomic_shared_ptr.hpp"

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
///
/// The sequence word packs a writer-active bit (the top bit) with a
/// monotonically increasing sequence in the low bits. A writer must claim
/// the writer bit before bumping the sequence. Without that claim two
/// concurrent writers can leave the low bits even (0 -> 1 -> 2) while one
/// of them is still mutating the protected data, so a reader that sampled
/// the even value would accept torn data as consistent. The Linux kernel
/// seqlock this is modelled on relies on an outer spinlock for exactly this
/// reason; folding the claim into the same atomic keeps the type at one
/// 4-byte word.
///
/// This is ideal for cache_stats::consistent_snapshot() where:
///   - All counter updates are atomic (lock-free writes)
///   - Snapshots need consistent reads across multiple counters
///   - Writes to the snapshot lock are rare (only during reset_counters)
///
/// The writer claim spins rather than blocking. A blocking writer lock
/// would deadlock `cache_stats::operator=` for two threads running
/// `a = b` and `b = a` concurrently, which take the two snapshot locks in
/// opposite orders; use dual_seqlock_write_guard() for that case.
class seqlock {
public:
    static constexpr std::uint32_t kWriterBit = 0x8000'0000u;
    static constexpr std::uint32_t kSeqMask = ~kWriterBit;

    seqlock() : seq_(0) {}

    /// Begin a read section. Returns the current sequence number.
    ///
    /// this is a plain load — it deliberately does NOT
    /// wait for an in-progress writer, and never sleeps. Waiting here inverted
    /// the entire point of a seqlock: one preempted writer put *every* reader
    /// to sleep for microseconds-to-milliseconds, so a rare writer could
    /// wreck P99 for all readers.
    ///
    /// The standard optimistic protocol is used instead:
    ///     do { seq = read_begin(); ...read payload...; } while (read_retry(seq));
    ///
    /// This is sound because `read_retry()` compares the entire sequence word,
    /// and a writer advances it *twice* per critical section — once when it
    /// claims the writer bit (low bits go odd) and once on release (write_unlock
    /// stores `cur + 1`, clearing the bit). A sequence sampled while a writer is
    /// active therefore can never compare equal afterwards:
    ///   - sampled `kWriterBit | (S+1)` -> on release the word becomes `S+2`
    ///   - sampled `S` (writer not yet claimed) -> after a full cycle it is `S+2`
    std::uint32_t read_begin() const noexcept {
        return seq_.load(std::memory_order_acquire);
    }

    /// Check if a read section needs to be retried.
    /// Returns true if a write occurred during the read section.
    bool read_retry(std::uint32_t start_seq) const noexcept {
        std::atomic_thread_fence(std::memory_order_acquire);
        return seq_.load(std::memory_order_relaxed) != start_seq;
    }

    /// Claim exclusive write access and enter the critical section
    /// (sequence low bits become odd). Only one writer can hold the claim.
    void write_lock() noexcept {
        std::uint32_t cur = seq_.load(std::memory_order_relaxed);
        for (;;) {
            int spins = 0;
            while (cur & kWriterBit) {
                if (++spins < 64) {
                    std::this_thread::yield();
                } else {
                    std::this_thread::sleep_for(std::chrono::microseconds(1));
                }
                cur = seq_.load(std::memory_order_relaxed);
            }
            // Set the writer bit and bump the low bits to odd in one step.
            const std::uint32_t desired =
                kWriterBit | ((cur & kSeqMask) + 1u);
            if (seq_.compare_exchange_weak(cur, desired,
                    std::memory_order_acquire, std::memory_order_relaxed)) {
                return;
            }
            // cur was refreshed by the failed CAS — re-check the writer bit.
        }
    }

    /// Leave the critical section: sequence low bits become even and the
    /// writer claim is released, in a single store so no reader can observe
    /// "sequence even but writer still active".
    void write_unlock() noexcept {
        const std::uint32_t cur = seq_.load(std::memory_order_relaxed);
        seq_.store((cur + 1u) & kSeqMask, std::memory_order_release);
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

/// Acquire write guards on two seqlocks in address order.
///
/// `cache_stats::operator=` guards both `this` and `other`. Two threads
/// running `a = b` and `b = a` concurrently would take those two claims in
/// opposite orders, so any exclusive writer lock — spinning or blocking —
/// would hang both threads. Ordering the acquisition by address makes the
/// order a total order, which removes the cycle. Handles the self-assignment
/// case (identical address) by taking the claim only once.
class dual_seqlock_write_guard {
public:
    dual_seqlock_write_guard(seqlock& a, seqlock& b) noexcept
        : first_(&a), second_(&b) {
        if (first_ == second_) {
            second_ = nullptr;
        } else if (std::less<seqlock*>{}(second_, first_)) {
            std::swap(first_, second_);
        }
        first_->write_lock();
        if (second_) second_->write_lock();
    }
    ~dual_seqlock_write_guard() {
        if (second_) second_->write_unlock();
        first_->write_unlock();
    }
    dual_seqlock_write_guard(const dual_seqlock_write_guard&) = delete;
    dual_seqlock_write_guard& operator=(const dual_seqlock_write_guard&) = delete;

private:
    seqlock* first_;
    seqlock* second_;
};

// ============================================================================
// Striped Mutex
// ============================================================================

// Cacheline size is fixed at 64 rather than using
// std::hardware_destructive_interference_size, whose value can change with the
// compiler version or -mtune and would therefore be an ABI hazard. x86 and ARM
// both use 64-byte cachelines in practice.
inline constexpr std::size_t kCachelineSize = 64;

// Each stripe's mutex owns a whole cacheline, eliminating false sharing between
// cores. Mirrors CacheLib's folly::cacheline_aligned<Mutex> (MMLru.h:474).
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

    // Try to shared-lock a stripe (non-blocking). Used by the
    // value-layer TTL scanner to avoid blocking writers.
    auto try_make_shared_lock(std::size_t stripe) {
        return std::shared_lock<MutexType>(stripes_[stripe].mutex, std::try_to_lock);
    }

    std::size_t size() const noexcept { return stripes_.size(); }

    // Access a specific stripe's mutex by index. Used for runtime
    // configuration (e.g., set_lock_order_checking, set_fairness_mode).
    MutexType& mutex_at(std::size_t stripe) { return stripes_[stripe].mutex; }
    const MutexType& mutex_at(std::size_t stripe) const { return stripes_[stripe].mutex; }

    // Lock all stripes (for operations that need exclusive global access).
    // Exception safety: if locking a stripe throws, unlock every stripe already
    // held, so the caller cannot deadlock against itself. The rollback costs
    // nothing on the normal path.
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
    // Writer starvation detector forwarding
    // ----------------------------------------------------------------

    /// Set the writer starvation timeout on every stripe.
    template <typename M = MutexType>
    auto set_writer_starvation_timeout(uint64_t timeout_ns)
        -> decltype(std::declval<M&>().set_writer_starvation_timeout(timeout_ns), void())
    {
        for (auto& a : stripes_) a.mutex.set_writer_starvation_timeout(timeout_ns);
    }

    /// Aggregate writer_starvation_events across all stripes.
    template <typename M = MutexType>
    auto writer_starvation_events() const
        -> decltype(std::declval<const M&>().writer_starvation_events())
    {
        std::size_t total = 0;
        for (auto& a : stripes_) total += a.mutex.writer_starvation_events();
        return total;
    }

    /// Maximum writer_max_wait_ns across all stripes.
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

    /// Reset writer_max_wait_ns on every stripe.
    template <typename M = MutexType>
    auto reset_writer_max_wait_ns()
        -> decltype(std::declval<M&>().reset_writer_max_wait_ns(), void())
    {
        for (auto& a : stripes_) a.mutex.reset_writer_max_wait_ns();
    }

    /// Quiescent variant — acquires all stripes exclusively
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
    // NUMA-aware reader counter forwarding
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

    /// In the std::mutex specialisation, a shared lock is an exclusive lock:
    /// there is no reader-reader concurrency to preserve.
    auto make_shared_lock(std::size_t stripe) {
        return make_unique_lock(stripe);
    }

    std::size_t size() const noexcept { return stripes_.size(); }

    // Access a specific stripe's mutex by index.
    std::mutex& mutex_at(std::size_t stripe) { return stripes_[stripe].mutex; }
    const std::mutex& mutex_at(std::size_t stripe) const { return stripes_[stripe].mutex; }

    // Exception safety: if locking a stripe throws, unlock every stripe already
    // held, so the caller cannot deadlock against itself. The rollback costs
    // nothing on the normal path.
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

    // In the std::mutex specialisation, a shared lock is an exclusive lock (there
    // is no reader-reader concurrency). lock_shared_all / unlock_shared_all are
    // still provided so template code (striped_read_lock_all,
    // striped_mutex_read_all_guard, ...) can call one uniform interface without
    // specialising on the mutex type.
    void lock_shared_all() { lock_all(); }
    void unlock_shared_all() { unlock_all(); }

private:
    std::vector<AlignedMutex<std::mutex>> stripes_;
};

// ============================================================================
// Non-striped storage placeholder for striped_mutex_storage
// ============================================================================
//
// unified_cache<Trait, K, V> stores its striped lock array in a member whose
// type is `std::conditional_t<is_striped, lock_policy::striped_mutex_type, X>`.
//
// Historically X was `std::tuple<>`, which forced every constructor to
// initialise the member from a static factory that returned the member type by
// value:
//
//     , striped_mutex_(make_default_striped_mutex())
//
// That works only when the compiler applies guaranteed copy elision to a
// `[[no_unique_address]]` member.  GCC does not (long-standing PR98995, still
// open for GCC 14/15/16/17 — "copy elision not applied to members declared with
// [[no_unique_address]]"), so the deleted move constructor of
// `lazy_striped_mutex` (which holds a non-movable std::once_flag) made the whole
// library fail to compile with g++ for every striped alias
// (production_cache / striped_cache / segmented_* / f14_striped_cache /
// read_heavy_striped_cache).
//
// The fix is to make the storage type uniformly constructible from a stripe
// count, so the member can be *directly initialised* — no factory, no prvalue
// of the member type, and therefore no reliance on copy elision at all:
//
//     , striped_mutex_(lock_policy::default_num_stripes)   // striped
//     , striped_mutex_(num_stripes)                        // striped
//     , striped_mutex_(any_count)                          // non-striped -> no-op
//
// `no_striped_mutex` is deliberately an *empty* class with no member functions,
// so every `if constexpr (requires { striped_mutex_.foo(); })` probe in
// cache_trait.hpp keeps exactly the same answer it had with `std::tuple<>`.
// Combined with [[no_unique_address]] it occupies no storage.
struct no_striped_mutex {
    explicit constexpr no_striped_mutex(std::size_t = 0) noexcept {}
};

// ============================================================================
// Lazy-allocated striped_mutex wrapper
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

    // Writer starvation detector forwarding (lazy variant)
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
    // Parameter named `m` (not `sm`) to avoid shadowing the member.
    explicit striped_mutex_write_all_guard(striped_mutex<MutexType>& m) : sm(m) { sm.lock_all(); }
    ~striped_mutex_write_all_guard() { sm.unlock_all(); }
    striped_mutex_write_all_guard(const striped_mutex_write_all_guard&) = delete;
    striped_mutex_write_all_guard& operator=(const striped_mutex_write_all_guard&) = delete;
};

/// RAII guard that shared-locks all stripes of a striped_mutex.
/// Used for global read operations (size, empty, stats, etc.).
template <typename MutexType>
struct striped_mutex_read_all_guard {
    striped_mutex<MutexType>& sm;
    // Parameter named `m` (not `sm`) to avoid shadowing the member.
    explicit striped_mutex_read_all_guard(striped_mutex<MutexType>& m) : sm(m) { sm.lock_shared_all(); }
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
    /// Tag selecting the self-referencing task overload below. A distinct tag
    /// (rather than relying on overload resolution) keeps the choice explicit
    /// and avoids ambiguity with generic task lambdas.
    struct self_task_tag {};

    explicit periodic_worker(std::function<void()> task,
                             std::chrono::milliseconds interval)
        : task_(std::move(task))
        , interval_(interval)
        , running_(true)
        , interval_changed_at_(std::chrono::steady_clock::now()) {
        thread_ = std::thread([this] { run(); });
    }

    /// Construct with a task that receives a reference to the worker itself.
    ///
    /// A task that wants to re-anchor its own cadence (e.g. an adaptive drain
    /// interval) would otherwise have to reach back through the owning
    /// object's worker handle, which other threads may be resetting
    /// concurrently — a data race on that handle. Passing the worker by
    /// reference removes that access entirely.
    periodic_worker(self_task_tag, std::function<void(periodic_worker&)> task,
                    std::chrono::milliseconds interval)
        : task_([this, task = std::move(task)] { task(*this); })
        , interval_(interval)
        , running_(true)
        , interval_changed_at_(std::chrono::steady_clock::now()) {
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

    // Change the interval dynamically.
    //
    // the notification is only a hint to *re-evaluate the deadline*; it
    // must not be interpreted by run() as "the interval elapsed", otherwise
    // every reconfiguration runs the task one extra time (for a TTL cleaner or
    // drain worker that means a full extra pass over every shard).
    void set_interval(std::chrono::milliseconds new_interval) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            interval_ = new_interval;
            // Anchor the deadline at the moment of the change so that
            // *shortening* the interval takes effect immediately (the deadline
            // may already be in the past) while *lengthening* it does not
            // trigger an early run.
            interval_changed_at_ = std::chrono::steady_clock::now();
        }
        cv_.notify_all();
    }

    /// Install the exception handler invoked when the task throws.
    ///
    /// the handler is published through an atomic
    /// shared_ptr, because the worker thread reads it (in `run()`) while any
    /// thread may replace it here. The previous code read and wrote a plain
    /// `std::function` with no synchronization — a data race that could let the
    /// worker observe a half-constructed `std::function` (UB). The reader takes
    /// one RCU-style snapshot per tick, so an already-running tick may still
    /// invoke the previous handler, which is harmless.
    void on_error(std::function<void(std::exception_ptr)> handler) {
        error_handler_.store(
            std::make_shared<std::function<void(std::exception_ptr)>>(std::move(handler)));
    }

private:
    void run() {
        std::unique_lock<std::mutex> lock(mutex_);
        while (running_.load()) {
            // wait on an absolute DEADLINE rather than on "a
            // notification arrived". A notification from set_interval() (or a
            // spurious wake-up) merely causes a re-evaluation of the deadline;
            // the task runs only once the deadline has genuinely passed.
            auto deadline = interval_changed_at_ + interval_;
            while (running_.load()) {
                // Deliberately the 2-argument wait_until. The 3-argument
                // overload with a predicate does NOT return on notify_all()
                // unless the predicate is already true — it keeps re-waiting
                // until the absolute deadline. That made a notification from
                // set_interval() a no-op whenever the worker was already
                // parked (the common case: the worker wins the race to the
                // mutex right after construction), so shortening the interval
                // only took effect once the *old* deadline expired. Waiting
                // without a predicate lets any notification fall through to
                // the deadline re-evaluation below, which is exactly the
                // documented contract.
                cv_.wait_until(lock, deadline);
                if (!running_.load()) {
                    return;  // stopped
                }
                if (std::chrono::steady_clock::now() >= deadline) {
                    break;  // the interval genuinely elapsed
                }
                // Early wake-up: either spurious or an interval change. Re-anchor
                // to the (possibly new) interval; if the new deadline is already
                // in the past the next wait_until returns immediately.
                deadline = interval_changed_at_ + interval_;
            }
            if (!running_.load()) return;
            lock.unlock();
            try {
                task_();
            } catch (...) {
                auto handler = error_handler_.load();
                if (handler && *handler) {
                    (*handler)(std::current_exception());
                }
            }
            lock.lock();
            // The next period starts when this run finished.
            interval_changed_at_ = std::chrono::steady_clock::now();
        }
    }

    std::function<void()> task_;
    atomic_shared_ptr<std::function<void(std::exception_ptr)>> error_handler_;
    std::chrono::milliseconds interval_;
    std::atomic<bool> running_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::thread thread_;
    /// Time origin for the current wait: construction, the last set_interval(),
    /// or the end of the previous run. Guarded by mutex_.
    std::chrono::steady_clock::time_point interval_changed_at_;
};

// ============================================================================
// Payload size estimation
// ============================================================================

/// Real in-memory size of a key or value, in bytes.
///
/// The cache's default accounting bills a compile-time constant per item, so
/// `max_memory` degrades into an item-count proxy: a 1 KB std::string and a
/// 10-byte one are billed identically, and the memory watermarks then trigger
/// on the wrong signal. These overloads give the common owning containers their
/// real footprint (capacity plus the container object itself), so a caller can
/// opt into byte-accurate accounting:
///
///     c.set_value_size_calculator([](const V& v) { return detail::deep_size(v); });
///
/// It is deliberately not the default: changing the billed size changes when
/// eviction fires, which is observable behaviour.
template <typename T>
std::size_t deep_size(const T&) noexcept {
    return sizeof(T);
}

inline std::size_t deep_size(const std::string& s) noexcept {
    return sizeof(std::string) + s.capacity();
}

template <typename T, typename Alloc>
std::size_t deep_size(const std::vector<T, Alloc>& v) noexcept {
    return sizeof(std::vector<T, Alloc>) + v.capacity() * sizeof(T);
}

// ============================================================================
// Lock wait accounting
// ============================================================================

/// RAII: add the number of slow-path entries a mutex took while we acquired it
/// to a statistics counter.
///
/// The lock primitives (distributed_shared_mutex, shared_spinlock) expose a
/// monotonic per-instance `wait_count()`, incremented once per slow-path entry.
/// Sampling it around one acquisition gives exactly the number of times this
/// thread blocked — no guessing from a latency threshold — and costs two
/// relaxed loads when uncontended.
///
/// Construct it *before* the lock is taken; the destructor runs after the
/// acquisition completes (the returned lock is constructed first, then locals
/// are destroyed).
template <typename Mutex>
class lock_wait_counter {
public:
    lock_wait_counter(std::atomic<std::size_t>& counter,
                      const Mutex* mutex) noexcept
        : counter_(&counter), mutex_(mutex) {
        if constexpr (has_wait_count) {
            before_ = mutex ? mutex->wait_count() : 0;
        }
    }

    ~lock_wait_counter() {
        if constexpr (has_wait_count) {
            if (!mutex_) return;
            const std::size_t now = mutex_->wait_count();
            if (now != before_) {
                counter_->fetch_add(now - before_, std::memory_order_relaxed);
            }
        }
    }

    lock_wait_counter(const lock_wait_counter&) = delete;
    lock_wait_counter& operator=(const lock_wait_counter&) = delete;

private:
    static constexpr bool has_wait_count =
        requires(const Mutex& m) { m.wait_count(); };

    std::atomic<std::size_t>* counter_;
    const Mutex* mutex_;
    std::size_t before_ = 0;
};

// ============================================================================
// Locked Iterator Guard - removes the duplicated LockedIterator code from the four
// MM strategy types.
// ============================================================================

/// Manages a LockedIterator's lock lifetime and active flag.
/// Each MM type's LockedIterator composes this guard with its own queue-walk logic.
///
/// Usage, inside an MM type's LockedIterator:
///   class LockedIterator {
///       locked_iterator_guard guard_;
///       // ... strategy-specific walk
///   public:
///       LockedIterator(MMType& mm)
///           : guard_(mm.update_mutex_.m, mm.iterator_active_) {}
///       void destroy() { guard_.destroy(); }
///       // ...
///   };
template <typename Mutex = std::mutex>
class locked_iterator_guard {
public:
    /// Locks the mutex and checks the active flag.
    /// Throws runtime_error if iterator_active was already true.
    /// @param m             The MM layer's update_mutex.
    /// @param active_flag   The MM layer's iterator_active_ atomic flag.
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

    /// Releases the lock and clears the active flag.
    void destroy() noexcept {
        if (valid_) {
            if (active_flag_) active_flag_->store(false, std::memory_order_release);
            valid_ = false;
            if (lock_.owns_lock()) lock_.unlock();
        }
    }

    /// The underlying unique_lock, so callers can unlock manually if needed.
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
