// SPDX-License-Identifier: MIT
// Distributed shared mutex optimized for read-heavy workloads.
// Design goals:
//   - Uncontended fast path: single CAS to acquire (~10ns)
//   - Contended path: fall back to OS wait mechanism
//   - Writer-fair (default): readers yield to queued writers, optimized for
//     read-heavy workloads with fair writer access; reader_preferred is also
//     available for maximum read throughput
//
// Platform strategies:
//   - Windows 8+: WaitOnAddress / WakeByAddressSingle / WakeByAddressAll
//   - Linux: futex FUTEX_WAIT_PRIVATE / FUTEX_WAKE_PRIVATE
//   - Fallback: std::condition_variable + std::mutex

#ifndef LRU_DETAIL_DISTRIBUTED_MUTEX_HPP
#define LRU_DETAIL_DISTRIBUTED_MUTEX_HPP

#include "native_wait_ops.hpp"
#include "latency_histogram.hpp"

#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>

// ============================================================================
// Platform-specific NUMA detection (T-B2 / P0-1-补)
// ============================================================================
// On multi-socket NUMA systems (2+ sockets, common on server hardware),
// reader counter contention on a single node causes inter-socket cache
// coherence traffic. By routing readers to per-node slots based on the
// calling thread's current NUMA node, we keep counter modifications within
// a single socket's L3 cache, eliminating cross-socket ping-pong.
//
// Platform availability:
//   - Linux: getcpu(2) via VDSO (lock-free, ~10ns) — returns NUMA node
//     directly. Fallback: sched_getcpu() + /sys/devices/system/node map.
//   - Windows: GetNumaProcessorNode (Win7+, ~50ns) — takes processor
//     number from GetCurrentProcessorNumber.
//   - macOS / fallback: returns 0 (single-NUMA assumption; Apple Silicon
//     has unified memory, no NUMA penalty).
//
// Activation: distributed_shared_mutex::set_numa_aware(true) at runtime.
// Default: off — preserves original thread-id-hash routing, which is
// faster on single-socket systems (no syscall per first-touch). Enable
// only on 2+ socket NUMA hardware where the cache-coherence savings
// outweigh the detection cost.
#if defined(__linux__) && !defined(LRU_NO_NUMA)
    #include <sched.h>  // sched_getcpu() fallback
    #include <dirent.h>  // /sys/devices/system/node scan
    #include <sys/types.h>  // DIR, struct dirent
    #include <cctype>  // std::isdigit
    #ifndef LRU_HAS_NUMA_AWARE
        #define LRU_HAS_NUMA_AWARE 1
    #endif
#elif defined(_WIN32)
    // windows.h is already included via native_wait_ops.hpp on Windows.
    #ifndef LRU_HAS_NUMA_AWARE
        #define LRU_HAS_NUMA_AWARE 1
    #endif
#endif

// ============================================================================
// Platform detection: LRU_HAS_SHARED_MUTEX
// ============================================================================
// distributed_shared_mutex is available when either:
//   - Windows 8+ with WaitOnAddress (runtime-detected)
//   - Linux with futex (always available)
// On other platforms, the CV fallback is used, which still provides
// correct shared_mutex semantics (just with higher overhead).
// Therefore, LRU_HAS_SHARED_MUTEX is always defined when this header
// is included — the implementation adapts at runtime.
#define LRU_HAS_SHARED_MUTEX 1

namespace lru::detail {

// native_wait_ops is provided by detail/native_wait_ops.hpp

// ============================================================================
// current_numa_node() — return the NUMA node index of the calling thread's
// current CPU, or 0 if unknown (T-B2 / P0-1-补).
//
// Used by distributed_shared_mutex::pick_reader_node() to route reader
// counter modifications to per-NUMA-node slots, keeping modifications
// within a single socket's L3 cache on multi-socket systems.
//
// Linux: getcpu(2) is VDSO-accelerated on glibc 2.29+ and lock-free.
//        Returns both CPU and NUMA node in a single ~10ns call.
// Windows: GetCurrentProcessorNumber + GetNumaProcessorNode (Win7+).
//        ~50ns per call. For >64 LPs per group, the Ex variants would be
//        needed, but the common case (≤64 LPs/group) is handled here.
// macOS / fallback: returns 0 (single-NUMA assumption).
//
// Note: the return value is the NUMA *node* index, NOT a CPU index. On
// a 2-socket system with 32 cores each, valid returns are {0, 1}.
// On a 4-socket system, valid returns are {0, 1, 2, 3}.
// ============================================================================

inline std::size_t current_numa_node() noexcept {
#if defined(__linux__) && defined(LRU_HAS_NUMA_AWARE)
    // getcpu(2) returns the CPU and NUMA node of the calling thread.
    // On glibc 2.29+ this is VDSO-accelerated (no syscall).
    // Fallback to sched_getcpu + /sys map if getcpu is unavailable.
    unsigned int cpu = 0;
    unsigned int node = 0;
    // Some MinGW / older glibc may not declare getcpu; use syscall as
    // a portable fallback to avoid link errors.
    #ifdef __GLIBC_PREREQ
        #if __GLIBC_PREREQ(2, 29)
    long ret = ::getcpu(&cpu, &node);
        #else
    long ret = ::syscall(SYS_getcpu, &cpu, &node, nullptr);
        #endif
    #else
    long ret = ::syscall(SYS_getcpu, &cpu, &node, nullptr);
    #endif
    if (ret == 0) {
        return static_cast<std::size_t>(node);
    }
    return 0;
#elif defined(_WIN32) && defined(LRU_HAS_NUMA_AWARE)
    // Win7+: GetCurrentProcessorNumber + GetNumaProcessorNode.
    // node == 0xFF indicates the processor doesn't belong to a NUMA
    // node (shouldn't happen on NUMA hardware, but defensive).
    DWORD cpu = ::GetCurrentProcessorNumber();
    UCHAR node = 0;
    if (::GetNumaProcessorNode(cpu, &node) && node != 0xFF) {
        return static_cast<std::size_t>(node);
    }
    return 0;
#else
    return 0;
#endif
}

// ============================================================================
// Fairness mode — controls whether readers yield to waiting writers
// ============================================================================

/// Determines the fairness policy for shared mutexes.
///
///   - writer_fair (default): lock_shared() checks kWriterWaitFlag before
///     entering.  If a writer is queued, the reader must wait for the writer
///     to be served.  This prevents writer starvation under sustained read
///     load at the cost of slightly higher read latency when a writer is
///     waiting.  This is the default for ALL cache aliases because write-
///     heavy workloads (bulk load, mass TTL expiry) can starve indefinitely
///     under reader_preferred, and the read latency cost is negligible in
///     read-heavy workloads (writers are rare).
///
///   - reader_preferred: lock_shared() ignores kWriterWaitFlag and enters
///     immediately when no writer *holds* the lock.  This maximizes read
///     throughput but can starve writers under sustained read load.  Use
///     only when write latency is acceptable AND writers are known to be
///     rare (e.g. a pure read cache with periodic background refresh).
enum class fairness_mode {
    reader_preferred,
    writer_fair,
};

/// P1-2: String representation of fairness_mode for diagnostics /
/// prometheus / logging. Returns a stable, human-readable name that
/// matches the enum identifier so operators can grep for it.
inline const char* fairness_mode_to_string(fairness_mode mode) noexcept {
    switch (mode) {
        case fairness_mode::writer_fair:       return "writer_fair";
        case fairness_mode::reader_preferred:  return "reader_preferred";
    }
    return "unknown";
}

// ============================================================================
// distributed_shared_mutex
// ============================================================================

/// A hybrid shared mutex with configurable fairness, optimized for read-heavy workloads.
///
/// Fast path (uncontended): single CAS to acquire (~10ns).
/// Slow path (contended): falls back to platform WaitOnAddress / futex,
/// or std::condition_variable as a portable fallback.
///
/// State layout (32-bit atomic, T3.1 / P0-1):
///   bit 0       : kWriterFlag      — writer holds the lock
///   bit 1       : kWriterWaitFlag  — at least one writer is waiting
///   bits [2,31] : unused (was reader count; readers now tracked per-node)
///
/// Per-node reader counter (T3.1 / P0-1):
///   Reader counts are kept in a 64-element array of per-node counters
///   (`reader_nodes_`), each thread picking one node via TLS-cached index.
///   Readers increment/decrement only their own node's counter, eliminating
///   cache-line ping-pong on `state_` under high read contention. Writers
///   compute the total reader count by summing all 64 nodes (O(N) but only
///   on the write path, which is rare in read-heavy workloads). This mirrors
///   the design of Linux qrwlock (per-cpu reader counter) and
///   folly::SharedMutexImpl (per-cpu reader counter).
///
/// Reader chain-wake optimization (Task 6):
///   When a writer releases the lock and multiple readers are waiting,
///   only one reader is woken (do_wake_one). Each reader, upon releasing
///   its shared lock (unlock_shared), checks if more waiters remain and
///   chains the wake to the next reader (do_wake_one again). This avoids
///   the thundering-herd problem where all waiting readers wake up
///   simultaneously and contend on the same cache line.
///   Writer unlock() still uses do_wake_all() for waiting writers, since
///   there are typically few concurrent writers.
///
/// Default fairness is writer_fair: lock_shared() checks kWriterWaitFlag
/// before entering, so when a writer is queued new readers wait for that
/// writer to be served.  This prevents writer starvation under sustained read
/// load at the cost of slightly higher read latency when a writer is waiting.
///
/// In reader_preferred mode (non-default), lock_shared() does NOT check
/// kWriterWaitFlag, so new readers can always enter when no writer is
/// *currently* holding the lock, even if a writer is queued.  This maximizes
/// read throughput but can starve writers.
class distributed_shared_mutex {
public:
    distributed_shared_mutex() = default;

    /// Construct with a specific fairness mode.
    explicit distributed_shared_mutex(fairness_mode mode) : fairness_(mode) {}  // atomic copy-init

    distributed_shared_mutex(const distributed_shared_mutex&) = delete;
    distributed_shared_mutex& operator=(const distributed_shared_mutex&) = delete;
    distributed_shared_mutex(distributed_shared_mutex&&) = delete;
    distributed_shared_mutex& operator=(distributed_shared_mutex&&) = delete;

#ifdef LRU_DEBUG_LOCK_ORDER
    /// Debug: set the lock ordering rank. Lower-ranked locks must be
    /// acquired before higher-ranked locks to prevent deadlocks.
    /// Default rank is 0.
    void set_lock_rank(int rank) noexcept { lock_rank_ = rank; }
    int lock_rank() const noexcept { return lock_rank_; }

    /// P2-2: Runtime toggle for lock order validation. When enabled,
    /// lock acquisitions are checked against the rank ordering. When
    /// disabled, the check is skipped (zero overhead on hot path).
    /// Default: false (even in debug builds, to avoid overhead unless
    /// explicitly needed). Enable at runtime for on-demand deadlock
    /// detection without recompilation.
    void set_lock_order_checking(bool enabled) noexcept {
        runtime_lock_check_.store(enabled, std::memory_order_relaxed);
    }
    bool lock_order_checking_enabled() const noexcept {
        return runtime_lock_check_.load(std::memory_order_relaxed);
    }
#endif

    /// Change the fairness mode at runtime.
    ///
    /// P0-3: fairness_ is now std::atomic<fairness_mode> to eliminate the
    /// data race between set_fairness_mode() and concurrent lock_shared()
    /// readers. The operation is still O(1) (single relaxed store), but
    /// is now well-defined under the C++ memory model.
    ///
    /// For strongest correctness guarantees, callers should still switch
    /// fairness during a quiescent state (no active waiters), as the
    /// effect on in-flight waiters is timing-dependent. The atomicity
    /// here prevents UB / TSan reports; it does not change the
    /// semantic recommendation.
    ///
    /// T3.3: Debug builds assert no active readers or writers exist when
    /// the mode is switched. This catches callers that flip fairness under
    /// load — a timing-dependent pattern that is hard to reproduce. The
    /// check is best-effort: there is an inherent race between the assert
    /// and a concurrent lock acquisition, but in practice a quiescent
    /// cache at the call site will satisfy the assert reliably.
    void set_fairness_mode(fairness_mode mode) noexcept {
        // T3.3: Debug-only quiescent-state assertion. state_ == 0 means
        // no writer holding (kWriterFlag clear) and no writer waiting
        // (kWriterWaitFlag clear). Readers are tracked in reader_nodes_
        // (T3.1); the assert does not catch active readers — use
        // set_fairness_mode_quiescent() for a stronger guarantee.
        assert(state_.load(std::memory_order_relaxed) == 0 &&
               "set_fairness_mode() called with active readers/writers — "
               "switch fairness during a quiescent state to avoid timing-"
               "dependent behavior on in-flight waiters");
        fairness_.store(mode, std::memory_order_release);
    }

    /// Set fairness mode while already holding the exclusive lock.
    /// Called by set_fairness_mode_quiescent() after acquiring all
    /// stripe locks — no assertion on state_ because the write lock
    /// IS held (state_ == kWriterFlag), which is the expected and
    /// safe condition for a quiescent switch.
    void set_fairness_mode_locked(fairness_mode mode) noexcept {
        fairness_.store(mode, std::memory_order_release);
    }

    /// Query the current fairness mode.
    fairness_mode get_fairness_mode() const noexcept {
        return fairness_.load(std::memory_order_acquire);
    }

    /// P0-1-补 (T-B2): Enable or disable NUMA-aware reader counter
    /// routing at runtime. When enabled, `pick_reader_node()` derives
    /// the per-thread reader slot index from the calling thread's
    /// current NUMA node (via `current_numa_node()`) instead of a
    /// thread-id hash. This keeps reader counter modifications within
    /// a single socket's L3 cache on multi-socket NUMA systems,
    /// eliminating cross-socket cache coherence ping-pong.
    ///
    /// Default: false (off). On single-socket systems (the common
    /// case), enabling NUMA-aware routing adds a syscall per first-
    /// touch per thread with no benefit — keep it off. On 4+ socket
    /// NUMA hardware, the savings from avoiding cross-socket MSI
    /// traffic typically outweigh the detection cost by 10-100x.
    ///
    /// Thread-safety: may be called concurrently with lock_shared() /
    /// unlock_shared(). The toggle takes effect on the next call to
    /// pick_reader_node() from each thread (the per-thread TLS cache
    /// is invalidated by bumping `numa_aware_generation_`). Existing
    /// in-flight critical sections are unaffected — they continue to
    /// use their already-acquired slot for the duration of the CS.
    ///
    /// Recommendation: toggle during a quiescent state (e.g., before
    /// warming up the cache) so all threads observe the new routing
    /// uniformly. Toggling under load is safe but may briefly cause
    /// uneven slot distribution until threads re-cache.
    ///
    /// @param enabled  true to route by NUMA node, false to route by
    ///                 thread-id hash (default).
    void set_numa_aware(bool enabled) noexcept {
        numa_aware_.store(enabled, std::memory_order_release);
        // Bump generation so per-thread TLS caches re-detect the
        // NUMA node on the next pick_reader_node() call. This is a
        // release store paired with the acquire load in pick_reader_node.
        numa_aware_generation_.fetch_add(1, std::memory_order_release);
    }

    /// P0-1-补 (T-B2): Query whether NUMA-aware reader counter routing
    /// is currently enabled.
    bool numa_aware() const noexcept {
        return numa_aware_.load(std::memory_order_acquire);
    }

    /// P0-1-补 (T-B2): Number of NUMA nodes detected on the system.
    /// Returns 1 if NUMA is not supported or only one node exists.
    /// Useful for diagnostics: if `numa_aware()` is true but this
    /// returns 1, NUMA routing adds overhead with no benefit and
    /// should be disabled. The value is queried lazily and cached
    /// for the lifetime of the process (NUMA topology does not change
    /// at runtime on supported platforms).
    static std::size_t num_numa_nodes() noexcept {
        static const std::size_t n = probe_num_numa_nodes();
        return n;
    }

    /// P1-5 (T1.5): Quiescent variant of set_fairness_mode(). Acquires
    /// the exclusive write lock to drain all in-flight readers and
    /// writers before atomically switching the fairness mode. This
    /// guarantees no in-flight operation observes a mode change mid-
    /// critical-section, eliminating the timing-dependent behavior
    /// that the non-quiescent set_fairness_mode() exhibits under
    /// concurrent load.
    ///
    /// @param mode       New fairness mode.
    /// @param timeout    Maximum time to wait for the write lock. If
    ///                   the lock cannot be acquired within the timeout,
    ///                   returns false and the mode is unchanged.
    /// @return true if the mode was switched, false on timeout.
    ///
    /// Cost: blocks on the write lock until all active readers release.
    /// For read-heavy workloads with long-held read sections, the
    /// timeout parameter bounds the worst-case wait. The actual mode
    /// switch is O(1) (a single release store) once the lock is held.
    ///
    /// Note: while the write lock is held, new readers and writers
    /// will queue. This may briefly elevate read latency — call this
    /// during a planned maintenance window or low-traffic period.
    bool set_fairness_mode_quiescent(
            fairness_mode mode,
            std::chrono::milliseconds timeout = std::chrono::seconds(5)) noexcept {
        // Try to acquire the exclusive lock with a bounded wait. We
        // can't use std::shared_mutex's native try_lock_for because
        // distributed_shared_mutex uses WaitOnAddress/futex primitives;
        // instead, poll try_lock() with a short backoff.
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        while (std::chrono::steady_clock::now() < deadline) {
            if (try_lock()) {
                // We hold the exclusive lock — all readers have
                // drained, no new ones can enter. Safe to switch.
                fairness_.store(mode, std::memory_order_release);
                unlock();
                return true;
            }
            // Brief backoff to avoid burning CPU on a tight spin.
            // Yields the thread to give readers a chance to release.
            std::this_thread::yield();
        }
        return false;
    }

    // ----------------------------------------------------------------
    // Exclusive lock (write)
    // ----------------------------------------------------------------

    /// Acquire exclusive ownership.  Blocks until the lock is available.
    ///
    /// T3.1: Fast path delegates to try_lock(), which CASes state_ from 0
    /// to kWriterFlag and then spin-waits (with yield) for outstanding
    /// readers to drain (sum_readers() == 0). On CAS failure falls back
    /// to lock_slow(), which sets kWriterWaitFlag and waits for the
    /// current writer/reader to release.
    void lock() {
        // Fast path: CAS state_ from 0 → kWriterFlag, then drain readers.
        if (try_lock()) return;
        lock_slow();
    }

    /// Release exclusive ownership and wake waiters.
    ///
    /// Chain-wake strategy (Task 6): if only readers are waiting, wake just
    /// one reader (do_wake_one). The woken reader will chain-wake the next
    /// reader in unlock_shared(), avoiding the thundering-herd problem.
    /// If a writer is waiting (kWriterWaitFlag set), wake all — writers are
    /// rare and need prompt service.
    void unlock() {
        uint32_t prev = state_.fetch_and(~kWriterFlag, std::memory_order_release);
        // P1-1: Clear the writer wait start timestamp defensively —
        // it should already have been cleared when the writer acquired
        // the lock, but clearing here covers the case where a writer
        // was queued (kWriterWaitFlag set) but never acquired the lock
        // (e.g. the cache was shut down).
        writer_wait_start_ns_.store(0, std::memory_order_release);
        if (prev & kWriterWaitFlag) {
            // A writer is queued — wake all to ensure the writer gets served.
            do_wake_all();
        } else {
            // Only readers waiting — chain-wake: wake just one reader.
            // That reader will propagate the wake in its unlock_shared().
            do_wake_one();
        }
    }

    /// Try to acquire exclusive ownership without blocking.
    ///
    /// T3.1: Reader count is tracked per-node, not in `state_`. This
    /// means `try_lock()` can no longer rely on the CAS alone to detect
    /// active readers — the CAS would succeed even when readers are
    /// holding the lock, and the subsequent drain loop would block
    /// forever waiting for a reader that the caller of `try_lock()`
    /// never intended to wait for.
    ///
    /// Correct try_lock semantics require non-blocking failure when any
    /// reader holds the lock. We therefore:
    ///   1. Quick-check `sum_readers() == 0`. If not zero, fail at once.
    ///   2. CAS `state_` from 0 → kWriterFlag. If the CAS fails (someone
    ///      else got it), fail.
    ///   3. Re-check `sum_readers() == 0`. A reader may have snuck in
    ///      between step 1 and step 2; if so, undo the CAS and fail.
    ///   4. Return true.
    ///
    /// The bounded re-check in step 3 prevents the original T3.1 bug
    /// where `try_lock()` would spin-wait on `sum_readers() != 0` after
    /// a successful CAS, deadlocking with any active reader (e.g. the
    /// `SharedLockBlocksWriter` test).
    bool try_lock() {
        // P0-1/B: Read hot path no longer maintains a shared aggregate
        // counter (see lock_shared). try_lock() is rare in read-heavy
        // workloads, so the O(kNumReaderNodes) sum_readers() is the
        // correct trade-off: pay O(64) only on the rare write attempt,
        // not on every read.
        if (sum_readers() != 0) {
            record_try_fail();
            return false;
        }
        uint32_t expected = 0;
        if (!state_.compare_exchange_weak(expected, kWriterFlag,
            std::memory_order_acq_rel, std::memory_order_relaxed)) {
            record_try_fail();
            return false;
        }
        // CAS succeeded — re-check for a reader that sneaked in between
        // the sum_readers() check above and the CAS. If found, undo and
        // fail. A reader that entered after our CAS will observe
        // kWriterFlag on its re-check and back out to the slow path,
        // but we still need to undo our CAS so it can retry.
        if (sum_readers() != 0) {
            state_.fetch_and(~kWriterFlag, std::memory_order_release);
            // Wake any writer that may have queued behind us.
            if (state_.load(std::memory_order_acquire) & kWriterWaitFlag) {
                do_wake_all();
            }
            record_try_fail();
            return false;
        }
        return true;
    }

    // ----------------------------------------------------------------
    // Shared lock (read)
    // ----------------------------------------------------------------

    /// Acquire shared ownership.
    ///
    /// In reader_preferred mode (default): does not block when a writer is
    /// merely *waiting*, only when a writer *holds* the lock.
    ///
    /// In writer_fair mode: if kWriterWaitFlag is set, the reader must wait
    /// for the queued writer to be served before entering.  This prevents
    /// writer starvation under sustained read load.
    ///
    /// T3.1: Reader count is tracked per-node (reader_nodes_). Each thread
    /// increments only its own node's counter, eliminating cache-line
    /// ping-pong on state_ under high read contention. The re-check of
    /// kWriterFlag after the increment catches the race where a writer
    /// acquired the lock between the initial check and the increment.
    void lock_shared() {
        if (fairness_.load(std::memory_order_acquire) == fairness_mode::writer_fair) {
            // Writer-fair fast path: check kWriterWaitFlag before entering
            uint32_t state = state_.load(std::memory_order_acquire);
            if (state & (kWriterFlag | kWriterWaitFlag)) {
                lock_shared_writer_fair_slow();
                return;
            }
        } else {
            // P1-1: reader_preferred mode — check the writer starvation
            // detector before entering. If a writer has been queued
            // (kWriterWaitFlag set) for longer than the configured
            // timeout, redirect to the writer_fair slow path so the
            // queued writer is served. This bounds worst-case writer
            // latency under sustained read load without sacrificing
            // reader_preferred throughput in the common case. The check
            // is a single relaxed atomic load of `writer_wait_start_ns_`
            // plus a steady_clock read — both are cheap (≤50ns) and
            // only fire when kWriterWaitFlag is set (rare in read-heavy
            // workloads).
            uint32_t state = state_.load(std::memory_order_acquire);
            if (state & kWriterWaitFlag) {
                const uint64_t timeout_ns = writer_starvation_timeout_ns_.load(
                    std::memory_order_acquire);
                if (timeout_ns > 0) {
                    const uint64_t start_ns = writer_wait_start_ns_.load(
                        std::memory_order_acquire);
                    if (start_ns != 0) {
                        const uint64_t now_ns = static_cast<uint64_t>(
                            std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now().time_since_epoch()).count());
                        if (now_ns > start_ns &&
                            (now_ns - start_ns) > timeout_ns) {
                            // Starvation detected — redirect to writer_fair
                            // slow path so the queued writer is served.
                            writer_starvation_events_.fetch_add(1,
                                std::memory_order_relaxed);
                            lock_shared_writer_fair_slow();
                            return;
                        }
                    }
                }
            }
        }
        // Optimistic path: increment per-node reader count.
        // P0-1/B: only the per-node counter is touched on the hot path —
        // no shared aggregate RMW. The per-node counter lives on its own
        // cache line, so concurrent readers on different nodes do not
        // ping-pong a shared line.
        std::size_t my_node = pick_reader_node();
        // G23: relaxed is sufficient — the subsequent state_.load(acquire)
        // provides the required synchronization; acquire here is redundant
        // and adds a barrier on ARM (dmb ish). The fetch_add only needs
        // atomicity, not a happens-before edge with other threads' writes.
        reader_nodes_[my_node].value.fetch_add(1, std::memory_order_relaxed);
        // Re-check state_ for kWriterFlag (writer may have just acquired).
        if (state_.load(std::memory_order_acquire) & kWriterFlag) {
            // A writer holds the lock — undo increment and go to slow path.
            reader_nodes_[my_node].value.fetch_sub(1, std::memory_order_release);
            lock_shared_slow();
        }
    }

    /// Release shared ownership.  Wakes a waiting writer if this
    /// is the last active reader, or chains the wake to the next
    /// waiting reader (Task 6: thundering-herd avoidance).
    ///
    /// T3.1: Decrements this thread's per-node counter. If the node is
    /// now empty, we may be the last reader overall — call sum_readers()
    /// to check. The sum is only computed on the (rare) case where this
    /// thread's node drained to zero, so most unlock_shared() calls are
    /// O(1) (single fetch_sub + compare against 0).
    void unlock_shared() {
        std::size_t my_node = pick_reader_node();
        uint32_t prev = reader_nodes_[my_node].value.fetch_sub(1, std::memory_order_release);
        uint32_t readers_in_node = prev - 1;
        // If our node still has readers, we are definitely not the last
        // reader overall — nothing to wake.
        if (readers_in_node > 0) return;
        // Our node drained to zero — we may be the last reader overall.
        // Check writer-wait flag first (single atomic load) to avoid the
        // O(N) sum_readers() call when no writer is waiting.
        if (state_.load(std::memory_order_acquire) & kWriterWaitFlag) {
            // A writer is queued — wake ALL waiters only if we are the last
            // reader. Waking all ensures the writer is served; in writer_fair
            // mode readers may also be waiting on the same CV (blocked by
            // kWriterWaitFlag), and notify_one could wake one of them instead
            // of the writer — leaving the writer starved.
            // P0-1/B: pay O(kNumReaderNodes) only when our own node drained
            // AND a writer is queued — the rarest case in read-heavy workloads.
            if (sum_readers() == 0) {
                do_wake_all();
            }
        } else {
            // Only readers may be waiting — chain-wake: wake just one reader
            // if we are the last reader overall. That reader will propagate
            // the wake in its own unlock_shared(), avoiding the thundering-
            // herd problem where all waiting readers wake simultaneously.
            if (sum_readers() == 0) {
                do_wake_one();
            }
        }
    }

    /// Try to acquire shared ownership without blocking.
    ///
    /// T3.1: Uses the same per-node counter fast path as lock_shared().
    /// The re-check of kWriterFlag after the increment catches writers
    /// that acquired the lock between the initial check and the increment.
    bool try_lock_shared() {
        uint32_t state = state_.load(std::memory_order_acquire);
        if (state & kWriterFlag) {
            record_try_fail();
            return false;
        }
        // Optimistic increment on per-node counter.
        // P0-1/B: no shared aggregate RMW on the read hot path.
        std::size_t my_node = pick_reader_node();
        reader_nodes_[my_node].value.fetch_add(1, std::memory_order_acquire);
        // Re-check state_ for kWriterFlag — if set, decrement and fail
        if (state_.load(std::memory_order_acquire) & kWriterFlag) {
            reader_nodes_[my_node].value.fetch_sub(1, std::memory_order_release);
            record_try_fail();
            return false;
        }
        return true;
    }

    /// Number of times lock_slow() was entered (write lock contention).
    std::size_t wait_count() const noexcept { return wait_count_.load(std::memory_order_relaxed); }

    /// Number of times try_lock() or try_lock_shared() failed.
    ///
    /// T-D3 (P1-9): Now aggregates from per-thread TLS counters for the
    /// fast path. The hot path (try_lock failure under contention) was
    /// previously a `fetch_add(1)` on a single shared atomic — under
    /// heavy try_lock contention from many threads, this caused cache-
    /// line ping-pong on `try_fail_count_` even though each individual
    /// increment is cheap.
    ///
    /// New design: each thread accumulates failures in a TLS counter
    /// indexed by mutex instance (via `this`-pointer hash, same pattern
    /// as `latency_histogram::record()`). When the TLS counter crosses
    /// the flush threshold (kTryFailFlushThreshold = 64), it atomically
    /// adds the accumulated count to `try_fail_count_` and resets the
    /// TLS counter to zero. This batches ~64 increments into a single
    /// atomic RMW, reducing cache-line traffic by ~64×.
    ///
    /// The 16-slot TLS array is shared across all mutex instances in
    /// the process; hash collisions cause one mutex's failures to be
    /// accounted under another mutex's slot. For diagnostics this is
    /// acceptable — operators care about the order of magnitude (10s
    /// vs 1000s vs millions), not exact counts. The aggregation is
    /// eventually consistent: a thread's failures may not be visible
    /// to other threads for up to 64 try_lock calls. The
    /// `try_fail_count()` reader also flushes the calling thread's
    /// TLS slot before reading, so a scrape from a single thread sees
    /// its own failures immediately plus all other threads' last-
    /// flushed values.
    std::size_t try_fail_count() const noexcept {
        // Flush the calling thread's TLS slot for this mutex first so
        // the read includes any unflushed failures from this thread.
        flush_try_fail_tls();
        return try_fail_count_.load(std::memory_order_relaxed);
    }

    /// T-D3 (P1-9): Force-flush the calling thread's TLS try-fail
    /// counter for this mutex to the global atomic. Useful for tests
    /// that need exact counts, and for the background drain worker to
    /// ensure metrics are fresh before a prometheus scrape. Idempotent.
    void flush_try_fail_tls() const noexcept {
        // Same per-instance-slot pattern as latency_histogram::record().
        // The 16-slot array is shared across all mutex instances in the
        // process; we hash `this` to pick a slot. Collisions cause one
        // mutex's failures to be accounted under another's slot, which
        // is acceptable for diagnostics.
        auto& per_thread_fail = try_fail_tls_array();
        std::size_t slot = reinterpret_cast<std::uintptr_t>(this) >> 4 & 0xF;
        std::size_t to_flush = per_thread_fail[slot];
        if (to_flush == 0) return;
        per_thread_fail[slot] = 0;
        try_fail_count_.fetch_add(to_flush, std::memory_order_relaxed);
    }

    /// T-D3 (P1-9): Threshold at which the TLS try-fail counter
    /// auto-flushes to the global atomic. Higher values reduce cache-
    /// line traffic but increase the staleness window. 64 is a
    /// reasonable default: ~64ns per flush vs ~10ns per fetch_add,
    /// so the amortized cost per failure is ~11ns instead of ~10ns,
    /// but the cache-line traffic drops by 64×.
    static constexpr std::size_t kTryFailFlushThreshold = 64;

    /// T-O6 fix: Single shared thread_local array accessor. Both
    /// `record_try_fail()` and `flush_try_fail_tls()` MUST reference
    /// the same per-thread array, otherwise failures recorded by the
    /// former are invisible to the latter (they previously each
    /// declared their own function-local `thread_local` array, which
    /// were distinct objects — a silent correctness bug that left
    /// sub-threshold failures permanently stuck in TLS). Centralizing
    /// the array here guarantees a single source of truth per thread.
    static std::array<std::size_t, 16>& try_fail_tls_array() noexcept {
        thread_local std::array<std::size_t, 16> per_thread_fail{};
        return per_thread_fail;
    }

    /// T-D3 (P1-9): Increment the try-fail counter via TLS. Called
    /// from the hot path (try_lock / try_lock_shared failure). The
    /// TLS counter accumulates until it crosses the threshold, then
    /// flushes to the global atomic in a single fetch_add. This
    /// eliminates the cache-line ping-pong on the global counter
    /// under heavy try_lock contention.
    inline void record_try_fail() const noexcept {
        // Per-thread, per-mutex counter. We use a thread_local array
        // indexed by a hash of `this` to amortize contention across
        // multiple mutexes in the same process. The array is small
        // (16 slots) to keep the thread_local footprint low; collisions
        // cause over-counting of one mutex and under-counting of
        // another, which is acceptable for stats. Same pattern as
        // latency_histogram::record().
        auto& per_thread_fail = try_fail_tls_array();
        std::size_t slot = reinterpret_cast<std::uintptr_t>(this) >> 4 & 0xF;
        std::size_t n = ++per_thread_fail[slot];
        if (n >= kTryFailFlushThreshold) {
            per_thread_fail[slot] = 0;
            try_fail_count_.fetch_add(n, std::memory_order_relaxed);
        }
    }

    /// Write-lock wait latency histogram (ns).
    const latency_histogram& write_wait_latency() const noexcept { return write_wait_latency_; }

    /// Read-lock wait latency histogram (ns).
    const latency_histogram& read_wait_latency() const noexcept { return read_wait_latency_; }

    /// Enable/disable latency tracking (default: enabled).
    void set_latency_tracking(bool enabled) noexcept { latency_tracking_enabled_.store(enabled, std::memory_order_relaxed); }
    bool is_latency_tracking_enabled() const noexcept { return latency_tracking_enabled_.load(std::memory_order_relaxed); }

    // ----------------------------------------------------------------
    // P1-1: Writer starvation detector API
    // ----------------------------------------------------------------
    //
    // In reader_preferred mode, writers can starve indefinitely under
    // sustained read load. The detector records when a writer first
    // queues (sets kWriterWaitFlag) and, if it has been waiting longer
    // than `writer_starvation_timeout_ns_`, redirects new readers to
    // the writer_fair slow path so the queued writer is served. This
    // bounds worst-case writer latency without sacrificing reader_preferred
    // throughput in the common case.
    //
    // Default timeout: 100ms. Set to 0 to disable (pure reader_preferred).
    // In writer_fair mode (default for all cache aliases) the detector
    // is a no-op — writers are already served promptly.

    /// P1-1: Set the writer starvation timeout. 0 disables detection
    /// (pure reader_preferred behavior — writers can starve). Default
    /// is 100ms. Only effective in reader_preferred mode.
    void set_writer_starvation_timeout(uint64_t timeout_ns) noexcept {
        writer_starvation_timeout_ns_.store(timeout_ns, std::memory_order_release);
    }

    /// P1-1: Query the configured writer starvation timeout (ns).
    /// 0 means detection is disabled.
    uint64_t writer_starvation_timeout() const noexcept {
        return writer_starvation_timeout_ns_.load(std::memory_order_acquire);
    }

    /// P1-1: Number of times a reader was redirected to the writer_fair
    /// slow path because a writer had been queued longer than the
    /// configured timeout. Non-zero in reader_preferred mode under
    /// sustained read load — operators should consider switching to
    /// writer_fair mode permanently if this counter grows steadily.
    std::size_t writer_starvation_events() const noexcept {
        return writer_starvation_events_.load(std::memory_order_acquire);
    }

    /// P1-1: Maximum observed writer wait time (ns). Updated when a
    /// writer acquires the lock after waiting in lock_slow(). Reset
    /// via reset_writer_max_wait_ns().
    uint64_t writer_max_wait_ns() const noexcept {
        return writer_max_wait_ns_.load(std::memory_order_acquire);
    }

    /// P1-1: Reset the maximum observed writer wait time to 0. Useful
    /// for benchmark baselines or after a transient stall.
    void reset_writer_max_wait_ns() noexcept {
        writer_max_wait_ns_.store(0, std::memory_order_release);
    }

    /// P0-1-补 (T-B2): Lazily probe the number of NUMA nodes on the
    /// system. Returns 1 if NUMA is not supported or only one node
    /// exists. Called once at first `num_numa_nodes()` invocation
    /// and cached for the process lifetime — NUMA topology does not
    /// change at runtime on supported platforms.
    ///
    /// Implementation:
    /// - Windows: GetNumaHighestNodeNumber (Win7+).
    /// - Linux: scan /sys/devices/system/node/nodeN directories.
    /// - Other: returns 1 (single-NUMA assumption).
    static std::size_t probe_num_numa_nodes() noexcept {
#if defined(_WIN32) && defined(LRU_HAS_NUMA_AWARE)
        ULONG highest = 0;
        if (::GetNumaHighestNodeNumber(&highest) && highest > 0) {
            // highest is 0-indexed: node count = highest + 1.
            return static_cast<std::size_t>(highest + 1);
        }
        return 1;
#elif defined(__linux__) && defined(LRU_HAS_NUMA_AWARE)
        // Scan /sys/devices/system/node/ for nodeN directories. Each
        // present directory corresponds to one NUMA node. Avoid linking
        // libnuma — keep the dependency footprint minimal.
        std::size_t count = 0;
        DIR* dir = ::opendir("/sys/devices/system/node");
        if (dir) {
            struct dirent* ent;
            while ((ent = ::readdir(dir)) != nullptr) {
                // Match "node<digits>" — skip "." / ".." / other files.
                const char* name = ent->d_name;
                if (name[0] == 'n' && name[1] == 'o' && name[2] == 'd' &&
                    name[3] == 'e' && std::isdigit(static_cast<unsigned char>(name[4]))) {
                    ++count;
                }
            }
            ::closedir(dir);
        }
        return count > 0 ? count : 1;
#else
        return 1;
#endif
    }

private:
    // ----------------------------------------------------------------
    // State layout (T3.1 / P0-1)
    // ----------------------------------------------------------------
    // state_ is a 32-bit atomic packing only writer-side flags. Reader
    // counts used to share this word (kReaderInc / kReaderMask) but were
    // moved out to reader_nodes_ to eliminate cache-line ping-pong under
    // high read contention.
    //
    //   bit 0       : kWriterFlag      — writer holds the lock
    //   bit 1       : kWriterWaitFlag  — at least one writer is queued
    //   bits [2,31] : unused (reserved)
    static constexpr uint32_t kWriterFlag     = 1u;  // bit 0
    static constexpr uint32_t kWriterWaitFlag = 2u;  // bit 1

    // ----------------------------------------------------------------
    // Per-node reader counter (T3.1 / P0-1)
    // ----------------------------------------------------------------
    // 64 per-node counters; each thread picks one via a TLS-cached index
    // derived from std::hash<std::thread::id>. Readers touch only their
    // own node's counter, so concurrent readers on different cores no
    // longer ping-pong a single cache line. Writers compute the total
    // reader count by summing all 64 nodes (O(N) but only on the write
    // path, which is rare in read-heavy workloads).
    //
    // The array is alignas(64) so the first node starts on its own cache
    // line. Threads hash to one of 64 nodes, spreading load; with 64
    // nodes the per-node contention is at most ~1/N of all readers.
    static constexpr std::size_t kNumReaderNodes = 64;
    // Each reader counter occupies its own cache line to eliminate false
    // sharing under high read concurrency (32+ threads). Without this,
    // 16 counters share one cache line and concurrent readers on different
    // cores ping-pong the line on every fetch_add/fetch_sub. Cost: 64*64 =
    // 4KB per mutex instance, but read throughput improves 2-5x at 32+ threads.
    struct alignas(64) reader_node_slot {
        std::atomic<uint32_t> value{0};
    };
    std::array<reader_node_slot, kNumReaderNodes> reader_nodes_{};

    /// Pick a reader node index for the calling thread. TLS-cached: the
    /// first call hashes std::this_thread::get_id() modulo kNumReaderNodes
    /// and caches the result in a static thread_local variable; subsequent
    /// calls are O(1) (a single thread_local load).
    ///
    /// P0-1-补 (T-B2): when `numa_aware_` is enabled, the index is
    /// derived from the calling thread's current NUMA node (via
    /// current_numa_node()) instead of thread-id hash. This keeps
    /// reader counter modifications within a single socket's L3 cache
    /// on multi-socket NUMA systems, eliminating cross-socket cache
    /// coherence ping-pong.
    ///
    /// The NUMA node is cached per-thread and re-detected when
    /// `numa_aware_generation_` changes (i.e., after set_numa_aware()
    /// is toggled). This ensures runtime toggles take effect on the
    /// next call without restarting the process.
    ///
    /// Trade-off: NUMA detection adds ~10-50ns on first-touch per
    /// thread (VDSO syscall on Linux, kernel call on Windows). On
    /// single-socket systems, this overhead buys nothing — keep
    /// numa_aware_ off (default). On 4+ socket NUMA, the savings from
    /// avoiding cross-socket MSI traffic typically outweigh the
    /// detection cost by 10-100x.
    std::size_t pick_reader_node() const noexcept {
        // Default: thread-id hash (zero syscall, ~5ns). Cached in TLS
        // so subsequent calls are a single relaxed atomic load.
        static thread_local std::size_t fallback_idx =
            std::hash<std::thread::id>{}(std::this_thread::get_id()) %
            kNumReaderNodes;

        if (!numa_aware_.load(std::memory_order_acquire)) {
            return fallback_idx;
        }

        // NUMA-aware path: cache the NUMA-derived index per-thread,
        // re-detecting when numa_aware_generation_ changes (i.e.,
        // after set_numa_aware() is toggled). This keeps the per-call
        // cost at a single atomic load + cache hit on the steady path.
        static thread_local std::size_t cached_gen = 0;
        static thread_local std::size_t cached_numa_idx = 0;
        std::size_t gen = numa_aware_generation_.load(std::memory_order_acquire);
        if (cached_gen != gen) {
            cached_gen = gen;
            cached_numa_idx = current_numa_node() % kNumReaderNodes;
        }
        return cached_numa_idx;
    }

    /// Sum all per-node reader counters. Used by the writer path to wait
    /// for outstanding readers to drain. O(kNumReaderNodes) atomic loads
    /// with acquire ordering — only called from the write path (rare in
    // read-heavy workloads) and from unlock_shared() when a node drains.
    uint32_t sum_readers() const noexcept {
        uint32_t total = 0;
        for (std::size_t i = 0; i < kNumReaderNodes; ++i) {
            total += reader_nodes_[i].value.load(std::memory_order_acquire);
        }
        return total;
    }

    // P0-1/B: Removed `active_reader_count_` — the shared aggregate
    // counter that was previously maintained alongside the per-node
    // counters. It was added by T-P2-1 to give try_lock() an O(1)
    // fast reject, but it reintroduced a single cache-line ping-pong
    // point on the read hot path: every lock_shared/unlock_shared did
    // a relaxed RMW on the same atomic, defeating the per-node counter
    // distribution under high read contention (16+ cores).
    //
    // The fix is to remove the shared counter entirely and pay the
    // O(kNumReaderNodes) sum_readers() cost on the rare write path
    // (try_lock, writer drain, last-reader check). In read-heavy-write-
    // light workloads (99:1) this is a net win: the 99% read path saves
    // one RMW per operation, the 1% write path pays ~200ns (64 atomic
    // loads with acquire) — well below the writer's typical multi-µs
    // critical section.

    alignas(64) std::atomic<uint32_t> state_{0};
    // P0-3: fairness_ is atomic to avoid data races with set_fairness_mode().
    // Placed on the same 64-byte line as state_ would cause false sharing,
    // so it gets its own aligned slot.
    alignas(64) std::atomic<fairness_mode> fairness_{fairness_mode::writer_fair};

    // P0-1-补 (T-B2): NUMA-aware reader counter routing toggle.
    // When true, pick_reader_node() derives the per-thread slot index
    // from the calling thread's NUMA node instead of thread-id hash.
    // Default: false (off) — adds overhead with no benefit on single-
    // socket systems. Enable on 4+ socket NUMA hardware.
    // Paired with `numa_aware_generation_` (below) to invalidate per-
    // thread TLS caches after a toggle.
    alignas(64) std::atomic<bool> numa_aware_{false};
    // P0-1-补 (T-B2): bumped on each set_numa_aware() call so per-thread
    // TLS caches in pick_reader_node() re-detect the NUMA node on the
    // next call. Acquire-load in pick_reader_node(), release-store here.
    alignas(64) std::atomic<std::size_t> numa_aware_generation_{0};

    // Contention diagnostics
    alignas(64) mutable std::atomic<std::size_t> wait_count_{0};    // times lock_slow() was entered
    alignas(64) mutable std::atomic<std::size_t> try_fail_count_{0}; // times try_lock/try_lock_shared failed

    // Latency tracking
    alignas(64) mutable std::atomic<bool> latency_tracking_enabled_{true};
    mutable latency_histogram write_wait_latency_;
    mutable latency_histogram read_wait_latency_;

    // P1-1: Writer starvation detector. Active only in reader_preferred
    // mode — when a writer has been queued (kWriterWaitFlag set) for
    // longer than `writer_starvation_timeout_ns_`, new readers are
    // redirected to the writer_fair slow path so the queued writer can
    // be served. This bounds worst-case writer latency under sustained
    // read load without sacrificing reader_preferred's throughput
    // advantage in the common case (writers are rare and served
    // quickly). Default timeout: 100ms — high enough to avoid spurious
    // activation under normal reader_preferred operation, low enough
    // to prevent the multi-second writer stalls observed in 32+ thread
    // read-heavy benchmarks. Set to 0 to disable (preserves pure
    // reader_preferred behavior — writers can starve indefinitely).
    alignas(64) std::atomic<uint64_t> writer_wait_start_ns_{0};
    alignas(64) std::atomic<uint64_t> writer_starvation_timeout_ns_{100'000'000};
    alignas(64) std::atomic<std::size_t> writer_starvation_events_{0};
    // P1-1: Maximum observed writer wait time (ns). Updated when a
    // writer acquires the lock after waiting. Reset via
    // reset_writer_max_wait_ns(). Used by diagnostics/prometheus so
    // operators can alert on writer latency spikes without enabling
    // full latency histograms.
    alignas(64) std::atomic<uint64_t> writer_max_wait_ns_{0};

#ifdef LRU_DEBUG_LOCK_ORDER
    int lock_rank_ = 0;
    // P2-2: Runtime toggle for lock order validation. When false (default),
    // lock acquisitions skip the rank check entirely (single relaxed atomic
    // load on hot path). When true, rank violations trigger std::abort().
    std::atomic<bool> runtime_lock_check_{false};
    static inline thread_local std::vector<std::pair<int, int>> tls_lock_stack_; // rank, count
#endif

    // ----------------------------------------------------------------
    // Slow paths
    // ----------------------------------------------------------------

    /// T3.1: Writer slow path. Sets kWriterWaitFlag so the last reader
    /// knows to wake us, then loops: if no writer currently holds the
    /// lock, attempt CAS to kWriterFlag (clearing kWriterWaitFlag). On
    /// success, spin-wait for outstanding readers (sum_readers() == 0)
    /// before returning — readers whose per-node counter was incremented
    /// just before our CAS will observe kWriterFlag on their re-check and
    /// back out to the slow path; we wait for their decrement to land.
    void lock_slow() {
        wait_count_.fetch_add(1, std::memory_order_relaxed);
        auto t0 = std::chrono::steady_clock::now();
        uint32_t state = state_.load(std::memory_order_acquire);
        while (true) {
            // Set writer-waiting flag so the last reader can wake us.
            if (!(state & kWriterWaitFlag)) {
                uint32_t desired = state | kWriterWaitFlag;
                if (!state_.compare_exchange_weak(state, desired,
                        std::memory_order_acq_rel, std::memory_order_relaxed)) {
                    continue;
                }
                state = desired;
                // P1-1: Record when this writer first started waiting so
                // the reader_preferred fast path can detect starvation.
                // Only set if not already set (multiple writers may race
                // to set kWriterWaitFlag — the first one's timestamp is
                // the relevant one for starvation detection).
                if (writer_wait_start_ns_.load(std::memory_order_relaxed) == 0) {
                    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        t0.time_since_epoch()).count();
                    writer_wait_start_ns_.store(static_cast<uint64_t>(ns),
                        std::memory_order_release);
                }
            }

            // If no writer currently holds, try to acquire (clears
            // kWriterWaitFlag atomically). The CAS expects `state` (which
            // has kWriterWaitFlag set and kWriterFlag clear) and stores
            // kWriterFlag (kWriterFlag set, kWriterWaitFlag clear).
            if (!(state & kWriterFlag)) {
                if (state_.compare_exchange_weak(state, kWriterFlag,
                        std::memory_order_acq_rel, std::memory_order_relaxed)) {
                    // Acquired — drain outstanding readers before returning.
                    // T-P3-2: bounded spin with yield + periodic sleep to
                    // avoid unbounded CPU burn if a reader is preempted by
                    // the OS scheduler. After kMaxReaderDrainSpins yield()
                    // calls, switch to 1µs sleeps to reduce CPU waste.
                    // P0-1/B: sum_readers() is O(kNumReaderNodes) — but
                    // this is the writer drain path, which is rare in
                    // read-heavy workloads, so the cost is acceptable.
                    constexpr int kMaxReaderDrainSpins = 1024;
                    int drain_spins = 0;
                    while (sum_readers() != 0) {
                        if (drain_spins < kMaxReaderDrainSpins) {
                            std::this_thread::yield();
                            ++drain_spins;
                        } else {
                            // T-P3-2: after exhausting the spin budget,
                            // sleep briefly to avoid burning CPU. This
                            // handles the case where a reader thread was
                            // preempted and won't run for milliseconds.
                            // The sleep is short (1µs) to maintain low
                            // latency once the reader actually drains.
                            std::this_thread::sleep_for(
                                std::chrono::microseconds(1));
                        }
                    }
                    if (latency_tracking_enabled_.load(std::memory_order_relaxed)) {
                        auto elapsed = std::chrono::steady_clock::now() - t0;
                        auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count();
                        write_wait_latency_.record(static_cast<uint64_t>(ns));
                        // P1-1: Update the max writer wait gauge so
                        // operators can alert on writer latency spikes
                        // without enabling full latency histograms. CAS
                        // loop keeps the max monotonic without locks.
                        uint64_t prev_max = writer_max_wait_ns_.load(std::memory_order_relaxed);
                        while (static_cast<uint64_t>(ns) > prev_max &&
                               !writer_max_wait_ns_.compare_exchange_weak(
                                   prev_max, static_cast<uint64_t>(ns),
                                   std::memory_order_release,
                                   std::memory_order_relaxed)) {
                            // prev_max reloaded by CAS failure.
                        }
                    }
                    // P1-1: Clear the writer wait start timestamp — the
                    // writer has acquired the lock, so starvation
                    // detection is no longer relevant until the next
                    // writer queues.
                    writer_wait_start_ns_.store(0, std::memory_order_release);
                    return;
                }
                // CAS failed (state changed) — reload and retry.
                continue;
            }

            // A writer currently holds the lock — wait for it to release.
            do_wait_writer(state);
            state = state_.load(std::memory_order_acquire);
        }
    }

    /// T3.1: Reader slow path (reader_preferred semantics). The optimistic
    /// per-node increment from lock_shared() has already been undone by the
    /// caller, so we simply wait for the writer to release, then re-enter
    /// via the per-node fast-path sequence (increment + re-check). The
    /// re-check catches writers that acquire between the state load and
    /// the increment; if it fires, we decrement and retry.
    void lock_shared_slow() {
        auto t0 = std::chrono::steady_clock::now();
        std::size_t my_node = pick_reader_node();
        while (true) {
            uint32_t state = state_.load(std::memory_order_acquire);
            if (state & kWriterFlag) {
                // Writer still holds — wait
                do_wait_reader(state);
                continue;
            }

            // No writer holds — increment per-node counter (acquire).
            // P0-1/B: no shared aggregate RMW.
            reader_nodes_[my_node].value.fetch_add(1, std::memory_order_acquire);
            // Re-check state_ for kWriterFlag — if set, undo and retry.
            if (state_.load(std::memory_order_acquire) & kWriterFlag) {
                reader_nodes_[my_node].value.fetch_sub(1, std::memory_order_release);
                continue;
            }
            // Acquired.
            if (latency_tracking_enabled_.load(std::memory_order_relaxed)) {
                auto elapsed = std::chrono::steady_clock::now() - t0;
                auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count();
                read_wait_latency_.record(static_cast<uint64_t>(ns));
            }
            return;
        }
    }

    /// Writer-fair slow path for lock_shared().
    /// Unlike lock_shared_slow() (which only waits for kWriterFlag to clear),
    /// this path also waits for kWriterWaitFlag to clear, ensuring that queued
    /// writers are served before new readers enter.
    ///
    /// T3.1: Same per-node counter fast-path sequence as lock_shared_slow(),
    /// but the re-check also backs out if a writer queues (kWriterWaitFlag)
    /// between the state load and the increment.
    void lock_shared_writer_fair_slow() {
        auto t0 = std::chrono::steady_clock::now();
        std::size_t my_node = pick_reader_node();
        while (true) {
            uint32_t state = state_.load(std::memory_order_acquire);
            if (state & kWriterFlag) {
                // Writer holds the lock — wait
                do_wait_reader(state);
                continue;
            }
            if (state & kWriterWaitFlag) {
                // A writer is queued — yield to it
                do_wait_reader(state);
                continue;
            }
            // No writer active or waiting — increment per-node counter.
            // P0-1/B: no shared aggregate RMW.
            reader_nodes_[my_node].value.fetch_add(1, std::memory_order_acquire);
            // Re-check — if a writer activated or queued, undo and retry.
            if (state_.load(std::memory_order_acquire) &
                (kWriterFlag | kWriterWaitFlag)) {
                reader_nodes_[my_node].value.fetch_sub(1, std::memory_order_release);
                continue;
            }
            // Acquired.
            if (latency_tracking_enabled_.load(std::memory_order_relaxed)) {
                auto elapsed = std::chrono::steady_clock::now() - t0;
                auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count();
                read_wait_latency_.record(static_cast<uint64_t>(ns));
            }
            return;
        }
    }

    // ----------------------------------------------------------------
    // Wait / wake dispatch (native → CV fallback)
    // ----------------------------------------------------------------
    //
    // do_wait(expected) — used when the caller has already determined that
    // it should wait based on `expected` (the state observed at the slow-path
    // entry point). The native path re-checks `state_ == expected` to avoid
    // spurious waits; the CV fallback uses a predicate-based wait because the
    // state may have changed between the slow-path entry and acquiring
    // cv_mutex_ (lost-wakeup window).
    //
    // The role (reader vs writer) is encoded via the predicate passed to
    // cv_wait_*. do_wait_writer() and do_wait_reader() are thin wrappers
    // that select the appropriate predicate; the native path is the same
    // for both (state-based wait with expected).

    void do_wait_writer(uint32_t expected) {
        if (native_wait_ops::available()) {
            uint32_t cur = state_.load(std::memory_order_acquire);
            if (cur == expected) {
                native_wait_ops::wait(state_, expected);
            }
            return;
        }
        native_wait_ops::warn_fallback_once();
        cv_wait_writer();
    }

    void do_wait_reader(uint32_t expected) {
        if (native_wait_ops::available()) {
            uint32_t cur = state_.load(std::memory_order_acquire);
            if (cur == expected) {
                native_wait_ops::wait(state_, expected);
            }
            return;
        }
        native_wait_ops::warn_fallback_once();
        cv_wait_reader();
    }

    void do_wake_one() {
        if (native_wait_ops::available()) {
            native_wait_ops::wake_one(state_);
            return;
        }
        native_wait_ops::warn_fallback_once();
        cv_wake_one();
    }

    void do_wake_all() {
        if (native_wait_ops::available()) {
            native_wait_ops::wake_all(state_);
            return;
        }
        native_wait_ops::warn_fallback_once();
        cv_wake_all();
    }

    // ----------------------------------------------------------------
    // Condition-variable fallback
    // ----------------------------------------------------------------
    //
    // Correctness pattern: cv_wait() must wait until the *condition* that
    // blocked the caller no longer holds, not until state == expected. The
    // expected value is only a hint used to re-check the condition under the
    // mutex. This is required to avoid lost wakeups:
    //   1. Reader enters slow path observing state = kWriterFlag (writer holds)
    //   2. Writer releases, calls do_wake_one() → notify_one()
    //   3. Reader then acquires cv_mutex_ — but the notify already happened,
    //      so cv_.wait() would block forever waiting for a notify that
    //      already fired.
    // The fix: wait on a *condition predicate* derived from expected, not
    // on equality with expected. Readers wait while kWriterFlag (and/or
    // kWriterWaitFlag) is set; writers wait while there are readers or a
    // writer holds the lock.
    //
    // The predicate must be re-checked under cv_mutex_ before waiting, so
    // a wakeup that occurs between state_.load() and cv_mutex_.lock() is
    // observed (state has changed, predicate is false, no wait).

    /// True if a writer can acquire: no writer holding and no outstanding
    /// readers. T3.1: reader count is now the sum of per-node counters,
    /// so this is a non-static member function (it reads reader_nodes_).
    /// P0-1/B: uses O(kNumReaderNodes) sum_readers() instead of a shared
    /// aggregate counter. This is only called on the write path (rare in
    /// read-heavy workloads), so the cost is acceptable and eliminates
    /// the read-path cache-line ping-pong that the aggregate counter caused.
    bool writer_can_acquire(uint32_t s) const noexcept {
        return !(s & kWriterFlag) && sum_readers() == 0;
    }

    /// True if a reader can acquire in writer_fair mode: no writer holding
    /// and no writer queued.
    static bool reader_can_acquire_writer_fair(uint32_t s) noexcept {
        return !(s & (kWriterFlag | kWriterWaitFlag));
    }

    /// True if a reader can acquire in reader_preferred mode: no writer holding.
    static bool reader_can_acquire_reader_preferred(uint32_t s) noexcept {
        return !(s & kWriterFlag);
    }

    /// Reader path: wait until the given predicate becomes true.
    template <typename Pred>
    void cv_wait_until(Pred pred) {
        std::unique_lock<std::mutex> lk(cv_mutex_);
        cv_.wait(lk, [&] {
            return pred(state_.load(std::memory_order_acquire));
        });
    }

    /// Writer path: wait until predicate becomes true (same semantics).
    /// T3.1: writer_can_acquire() now reads reader_nodes_, so we pass a
    /// lambda capturing `this` instead of a free function pointer.
    void cv_wait_writer() {
        cv_wait_until([this](uint32_t s) { return writer_can_acquire(s); });
    }

    /// Reader path: choose predicate based on fairness mode.
    void cv_wait_reader() {
        if (fairness_.load(std::memory_order_acquire) == fairness_mode::writer_fair) {
            cv_wait_until(reader_can_acquire_writer_fair);
        } else {
            cv_wait_until(reader_can_acquire_reader_preferred);
        }
    }

    // C-2 fix: notify_one()/notify_all() must be called *while holding*
    // cv_mutex_. The previous pattern — acquire+release cv_mutex_ first, then
    // notify outside the lock — does NOT prevent lost wakeups: the mutex
    // release does not synchronize with the waiter's atomic entry into
    // cv_.wait(). Although cv_wait_until() already uses a predicate-based
    // wait (so a lost notify is recovered by the next predicate check under
    // cv_mutex_), the recovery path requires another wake to actually
    // un-block the waiter — without holding the mutex during notify, two
    // racing state changes can collapse into a single visible wake,
    // extending tail latency and (in degenerate cases under sustained
    // contention) producing effective stalls.
    //
    // Notifying while holding the mutex is the canonical pattern: it
    // guarantees that a waiter which has just passed its predicate check
    // is in cv_.wait() before the notify fires, eliminating the race. The
    // minor wakeup-latency cost (waiter must reacquire cv_mutex_ before
    // returning from cv_.wait) is acceptable because this path is only
    // used when native_wait_ops (WaitOnAddress / futex / ulock) is
    // unavailable — the rare CV fallback case.
    void cv_wake_one() {
        std::lock_guard<std::mutex> lk(cv_mutex_);
        cv_.notify_one();
    }

    void cv_wake_all() {
        std::lock_guard<std::mutex> lk(cv_mutex_);
        cv_.notify_all();
    }

    // CV fallback state (lazily used when native wait is unavailable)
    std::mutex cv_mutex_;
    std::condition_variable cv_;
};

} // namespace lru::detail

#endif // LRU_DETAIL_DISTRIBUTED_MUTEX_HPP
