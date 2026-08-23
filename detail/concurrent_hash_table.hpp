// SPDX-License-Identifier: MIT
// Per-bucket locked concurrent hash table.
// Inspired by CacheLib's ChainedHashTable and folly::ConcurrentHashMap.
//
// Design choices:
//   - Bucket count is always a power of two → hash-to-bucket via bitwise AND
//   - Each bucket owns a singly-linked chain of nodes + a shared spinlock
//   - Spinlocks are cacheline-aligned (64 bytes) to eliminate false sharing
//   - find()/contains() use version-based optimistic read by default:
//     reads only do atomic loads (no RMW), falling back to shared lock on
//     version mismatch; disable via set_optimistic_read(false)
//   - insert()/erase() acquire exclusive locks and bump bucket version
//   - RAII bucket_lock / shared_bucket_lock / exclusive_bucket_lock guards
//     ensure exception-safe lock lifetime
//   - size() is tracked via a relaxed atomic — cheap but eventually consistent
//   - Seqlock-protected rehash: a global seqlock coordinates concurrent readers
//     with bucket array expansion. When load_factor() exceeds max_load_factor_,
//     rehash_if_needed() doubles the bucket count. Readers check the seqlock
//     and retry if a rehash is in progress, ensuring safe bucket array access
//     without blocking writers during the common (no-rehash) path.
//   - max_chain_length() scans all buckets for diagnostics
//
// EmbeddedChain mode (CacheLib-style):
//   When EmbeddedChain = true, the Value type itself participates as the hash
//   chain node (must have hash_chain_next/set_hash_chain_next/cached_hash/
//   set_cached_hash accessors and a .key member). This eliminates the separate
//   node_type allocation per entry, saving one allocation and one pointer
//   indirection per find().
//
// F14 SIMD probing mode (ProbingStyle = f14_probing_tag):
//   Each "chunk" stores 14 inline slots with 8-bit hash tags and a 16-bit
//   occupied bitmask. SIMD instructions (SSE2 / NEON / scalar fallback) can
//   check all 14 tags in one operation, eliminating pointer chasing within
//   the chunk. Items that don't fit in the inline slots overflow to a
//   traditional chain attached to the chunk. The same per-chunk locking
//   model (shared_spinlock + version counter) and seqlock-protected rehash
//   apply. Tag encoding: 0x00 = empty, 0x01 = tombstone, 0x80-0xFE = occupied.
//
// H-2 note: Non-EmbeddedChain mode and read-heavy-write-light workloads
//   When EmbeddedChain = false, nodes are allocated/deallocated directly
//   (not via hazptr/EBR reclamation). This means find_and_pin_lockfree()
//   CANNOT safely do lock-free pin — a node could be deleted between the
//   version check and the pin call (use-after-free). All lock-free read
//   paths fall back to shared_lock in this mode.
//
//   For read-heavy-write-light high-concurrency workloads (32+ threads,
//   99%+ reads), prefer EmbeddedChain = true to get true lock-free reads.
//   Non-EmbeddedChain mode is acceptable for write-heavy workloads or
//   low-concurrency scenarios where shared lock contention is negligible.

#ifndef LRU_DETAIL_CONCURRENT_HASH_TABLE_HPP
#define LRU_DETAIL_CONCURRENT_HASH_TABLE_HPP

#include "native_wait_ops.hpp"
#include "distributed_mutex.hpp"
#include "hazptr.hpp"
#include "epoch_reclamation.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

// Platform-specific pause intrinsic for spin-wait loops.
#if defined(_MSC_VER)
#include <intrin.h>
#define LRU_SPIN_PAUSE() _mm_pause()
#elif defined(__x86_64__) || defined(__i386__)
#include <xmmintrin.h>
#define LRU_SPIN_PAUSE() _mm_pause()
#elif defined(__aarch64__)
#define LRU_SPIN_PAUSE() __asm__ __volatile__("yield" ::: "memory")
#else
#define LRU_SPIN_PAUSE() ((void)0)
#endif

namespace lru::detail {

// ============================================================================
// Spinlock — lightweight exclusive lock for per-bucket synchronization
// ============================================================================

class spinlock {
public:
    spinlock() noexcept = default;
    spinlock(const spinlock&) = delete;
    spinlock& operator=(const spinlock&) = delete;

    void lock() noexcept {
        while (state_.exchange(1, std::memory_order_acquire) != 0) {
            LRU_SPIN_PAUSE();
        }
    }

    void unlock() noexcept {
        state_.store(0, std::memory_order_release);
    }

    bool try_lock() noexcept {
        uint32_t expected = 0;
        return state_.compare_exchange_strong(
            expected, 1, std::memory_order_acquire, std::memory_order_relaxed);
    }

    bool is_locked() const noexcept {
        return state_.load(std::memory_order_relaxed) != 0;
    }

private:
    std::atomic<uint32_t> state_{0};
};

struct alignas(64) aligned_spinlock : spinlock {};

// ============================================================================
// Shared spinlock — lightweight shared/exclusive lock for per-bucket
// synchronization, allowing concurrent readers on the same bucket.
// ============================================================================

class shared_spinlock {
public:
    shared_spinlock() noexcept = default;

    /// Construct with a specific fairness mode.
    explicit shared_spinlock(fairness_mode mode) noexcept : fairness_(mode) {}

    shared_spinlock(const shared_spinlock&) = delete;
    shared_spinlock& operator=(const shared_spinlock&) = delete;

    /// Change the fairness mode at runtime.
    /// Thread-safe: uses atomic store so concurrent lock_shared() callers
    /// observe a consistent value. C-1 fix: previously a plain (non-atomic)
    /// assignment, which was a data race under the C++ memory model.
    void set_fairness_mode(fairness_mode mode) noexcept {
        fairness_.store(mode, std::memory_order_relaxed);
    }

    /// Query the current fairness mode.
    fairness_mode get_fairness_mode() const noexcept {
        return fairness_.load(std::memory_order_relaxed);
    }

    /// Set the writer-starvation timeout in nanoseconds. When in
    /// reader_preferred mode, a reader that observes a queued writer
    /// (kWriterWaitFlag set) for longer than this timeout redirects to
    /// the writer-fair slow path so the queued writer is served. This
    /// bounds worst-case writer latency under sustained read load
    /// without sacrificing reader_preferred throughput in the common
    /// case. Set to 0 to disable starvation detection (writers can be
    /// starved indefinitely by continuous reads). Mirrors
    /// distributed_shared_mutex::set_writer_starvation_timeout().
    /// Default: 100ms.
    void set_writer_starvation_timeout(uint64_t timeout_ns) noexcept {
        writer_starvation_timeout_ns_.store(timeout_ns, std::memory_order_release);
    }

    /// Query the configured writer-starvation timeout (ns).
    uint64_t writer_starvation_timeout() const noexcept {
        return writer_starvation_timeout_ns_.load(std::memory_order_acquire);
    }

    /// Number of times a reader was redirected to the writer-fair slow
    /// path because a queued writer exceeded the starvation timeout.
    std::size_t writer_starvation_events() const noexcept {
        return writer_starvation_events_.load(std::memory_order_relaxed);
    }

    void lock() {
        uint32_t expected = 0;
        if (state_.compare_exchange_strong(
                expected, kWriterFlag,
                std::memory_order_acquire, std::memory_order_relaxed)) {
            return;
        }
        lock_slow();
    }

    void unlock() {
        state_.fetch_and(~(kWriterFlag | kWriterWaitFlag), std::memory_order_release);
        // P0-1: Clear the starvation timer so the next queued writer
        // records a fresh start time. Without this, the stale timestamp
        // from a previous queueing episode would make the next queued
        // writer appear "already starved" and flip the lock to
        // effectively writer-fair.
        writer_wait_start_ns_.store(0, std::memory_order_release);
        do_wake_all();
    }

    bool try_lock() noexcept {
        uint32_t expected = 0;
        if (state_.compare_exchange_strong(
            expected, kWriterFlag, std::memory_order_acquire, std::memory_order_relaxed)) {
            return true;
        }
        try_fail_count_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }

    void lock_shared() {
        // C-1 fix: atomic load to prevent data race with set_fairness_mode().
        if (fairness_.load(std::memory_order_relaxed) == fairness_mode::writer_fair) {
            // Writer-fair fast path: check kWriterWaitFlag before entering
            uint32_t s = state_.load(std::memory_order_acquire);
            if (s & (kWriterFlag | kWriterWaitFlag)) {
                lock_shared_writer_fair_slow();
                return;
            }
        }
        // Optimistic path: increment reader count
        uint32_t old = state_.fetch_add(kReaderInc, std::memory_order_acquire);
        if ((old & kWriterFlag) == 0) {
            // P0-1: Writer-starvation detection under reader_preferred mode.
            // If a writer has been queued (kWriterWaitFlag set) for longer
            // than the configured timeout, redirect to the writer_fair slow
            // path so the queued writer is served. Mirrors
            // distributed_shared_mutex's reader_preferred anti-starvation.
            // The kWriterWaitFlag check is a single branch on a bit already
            // loaded in `old` — no extra atomic operation in the common case
            // (no writer queued).
            if ((old & kWriterWaitFlag) && writer_starved()) {
                // Undo this reader's increment, then wait for the queued
                // writer to be served before entering.
                state_.fetch_sub(kReaderInc, std::memory_order_release);
                lock_shared_writer_fair_slow();
            }
            return;
        }
        lock_shared_slow();
    }

    void unlock_shared() {
        uint32_t state = state_.fetch_sub(kReaderInc, std::memory_order_release);
        if ((state & kReaderMask) == kReaderInc && (state & kWriterWaitFlag)) {
            do_wake_one();
        }
    }

    bool try_lock_shared() noexcept {
        // T-P3-4: Use fetch_add instead of CAS for better scalability under
        // contention. The previous CAS-based approach could spuriously fail
        // when concurrent readers updated state_ between the initial load and
        // the compare_exchange, even though no writer was active. fetch_add
        // always succeeds atomically; we then check the writer flag in the
        // returned old value and undo the increment if a writer holds the lock.
        // This eliminates spurious try-lock failures caused by reader-reader
        // contention and matches the pattern used in lock_shared()'s fast path.
        uint32_t old = state_.fetch_add(kReaderInc, std::memory_order_acq_rel);
        if ((old & kWriterFlag) == 0) {
            // P0-1: Writer-starvation cooperation. A non-blocking
            // try_lock_shared() must still refuse to admit new readers
            // when a queued writer has exceeded the starvation timeout —
            // otherwise the anti-starvation mechanism in lock_shared()
            // is defeated by callers using try_lock_shared(), which is
            // the fallback path for optimistic reads. Undo the increment
            // and fail so the caller retries (or falls back to a locked
            // path) instead of feeding the reader-sustained starvation.
            if ((old & kWriterWaitFlag) && writer_starved()) {
                state_.fetch_sub(kReaderInc, std::memory_order_release);
                try_fail_count_.fetch_add(1, std::memory_order_relaxed);
                return false;
            }
            return true;  // No writer held the lock — read lock acquired.
        }
        // Writer holds the lock — undo the increment and fail.
        state_.fetch_sub(kReaderInc, std::memory_order_release);
        try_fail_count_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }

    bool is_locked() const noexcept {
        return (state_.load(std::memory_order_relaxed) & kWriterFlag) != 0;
    }

    bool is_lock_shared() const noexcept {
        return (state_.load(std::memory_order_relaxed) & kReaderMask) != 0;
    }

    /// Number of times lock_slow() was entered (write lock contention).
    std::size_t wait_count() const noexcept { return wait_count_.load(std::memory_order_relaxed); }

    /// Number of times try_lock() or try_lock_shared() failed.
    std::size_t try_fail_count() const noexcept { return try_fail_count_.load(std::memory_order_relaxed); }

private:
    static constexpr uint32_t kWriterFlag     = 1u;
    static constexpr uint32_t kReaderInc      = 2u;
    static constexpr uint32_t kReaderMask     = 0x7FFFFFFEu;
    static constexpr uint32_t kWriterWaitFlag = 0x80000000u;
    static constexpr int kSpinThreshold = 100;

    std::atomic<uint32_t> state_{0};
    // C-1 fix: atomic to prevent data race when set_fairness_mode() is
    // called concurrently with lock_shared(). Previously a plain
    // (non-atomic) member, which was UB under the C++ memory model.
    std::atomic<fairness_mode> fairness_{fairness_mode::reader_preferred};

    // P0-1: Writer-starvation detection under reader_preferred mode.
    // A queued writer records its start time in writer_wait_start_ns_;
    // a reader that observes kWriterWaitFlag set for longer than
    // writer_starvation_timeout_ns_ redirects to the writer-fair slow
    // path so the queued writer is served. Mirrors
    // distributed_shared_mutex. Default timeout 100ms. These fields are
    // written only on the writer-queueing / release paths and read only
    // when kWriterWaitFlag is observed (rare in read-heavy workloads),
    // so they stay off the reader hot path.
    std::atomic<uint64_t> writer_wait_start_ns_{0};
    std::atomic<uint64_t> writer_starvation_timeout_ns_{100'000'000};
    mutable std::atomic<std::size_t> writer_starvation_events_{0};

    // Contention diagnostics
    mutable std::atomic<std::size_t> wait_count_{0};    // times lock_slow() was entered
    mutable std::atomic<std::size_t> try_fail_count_{0}; // times try_lock/try_lock_shared failed

    /// P0-1: True when a queued writer has been waiting longer than the
    /// configured starvation timeout. Only meaningful when kWriterWaitFlag
    /// is set. Costs one relaxed load of the timer plus a steady_clock
    /// read — cheap, and only fires when a writer is actually queued.
    bool writer_starved() const noexcept {
        const uint64_t timeout_ns = writer_starvation_timeout_ns_.load(
            std::memory_order_acquire);
        if (timeout_ns == 0) return false;  // detection disabled
        const uint64_t start_ns = writer_wait_start_ns_.load(
            std::memory_order_acquire);
        if (start_ns == 0) return false;  // writer queued but timer not set yet
        const uint64_t now_ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count());
        if (now_ns <= start_ns) return false;  // clock went backwards
        if ((now_ns - start_ns) <= timeout_ns) return false;
        writer_starvation_events_.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    void lock_slow() {
        wait_count_.fetch_add(1, std::memory_order_relaxed);
        int spins = 0;
        for (;;) {
            uint32_t s = state_.load(std::memory_order_acquire);
            if ((s & kReaderMask) == 0 && !(s & kWriterFlag)) {
                if (state_.compare_exchange_weak(s, kWriterFlag,
                        std::memory_order_acq_rel, std::memory_order_relaxed))
                    return;
                continue;
            }
            if (!(s & kWriterWaitFlag)) {
                uint32_t desired = s | kWriterWaitFlag;
                if (!state_.compare_exchange_weak(s, desired,
                        std::memory_order_acq_rel, std::memory_order_relaxed))
                    continue;
                s = desired;
                // P0-1: Record when this writer first started waiting so
                // the reader_preferred fast path can detect starvation.
                // Only set if not already set — multiple writers may race
                // to set kWriterWaitFlag; the first one's timestamp is the
                // relevant one (it measures how long the oldest queued
                // writer has been waiting).
                if (writer_wait_start_ns_.load(std::memory_order_relaxed) == 0) {
                    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now().time_since_epoch()).count();
                    writer_wait_start_ns_.store(static_cast<uint64_t>(ns),
                        std::memory_order_release);
                }
            }
            if (spins < kSpinThreshold) { LRU_SPIN_PAUSE(); ++spins; continue; }
            do_wait(s);
            spins = 0;
        }
    }

    void lock_shared_slow() {
        state_.fetch_sub(kReaderInc, std::memory_order_relaxed);
        int spins = 0;
        for (;;) {
            uint32_t s = state_.load(std::memory_order_acquire);
            if (s & kWriterFlag) {
                if (spins < kSpinThreshold) { LRU_SPIN_PAUSE(); ++spins; continue; }
                do_wait(s); spins = 0; continue;
            }
            uint32_t desired = s + kReaderInc;
            if (state_.compare_exchange_weak(s, desired,
                    std::memory_order_acq_rel, std::memory_order_relaxed))
                return;
            spins = 0;
        }
    }

    /// Writer-fair slow path for lock_shared().
    /// Unlike lock_shared_slow() (which only waits for kWriterFlag to clear),
    /// this path also waits for kWriterWaitFlag to clear, ensuring that queued
    /// writers are served before new readers enter.
    void lock_shared_writer_fair_slow() {
        int spins = 0;
        for (;;) {
            uint32_t s = state_.load(std::memory_order_acquire);
            if (s & kWriterFlag) {
                if (spins < kSpinThreshold) { LRU_SPIN_PAUSE(); ++spins; continue; }
                do_wait(s); spins = 0; continue;
            }
            if (s & kWriterWaitFlag) {
                // A writer is queued — yield to it
                if (spins < kSpinThreshold) { LRU_SPIN_PAUSE(); ++spins; continue; }
                do_wait(s); spins = 0; continue;
            }
            // No writer active or waiting — try to acquire shared lock
            uint32_t desired = s + kReaderInc;
            if (state_.compare_exchange_weak(s, desired,
                    std::memory_order_acq_rel, std::memory_order_relaxed))
                return;
            spins = 0;
        }
    }

    void do_wait(uint32_t expected) {
        if (native_wait_ops::available()) {
            uint32_t cur = state_.load(std::memory_order_acquire);
            if (cur == expected) native_wait_ops::wait(state_, expected);
            return;
        }
        native_wait_ops::warn_fallback_once();
        cv_wait(expected);
    }

    void do_wake_one() {
        if (native_wait_ops::available()) { native_wait_ops::wake_one(state_); return; }
        native_wait_ops::warn_fallback_once();
        cv_wake_one();
    }

    void do_wake_all() {
        if (native_wait_ops::available()) { native_wait_ops::wake_all(state_); return; }
        native_wait_ops::warn_fallback_once();
        cv_wake_all();
    }

    // C-2 fix: previously cv_wait() used a bare cv_.wait(lk) after a single
    // state_ equality check. That check is racy: a concurrent cv_wake_*()
    // can fire its notify in the gap between the state_.load() here and the
    // actual entry into cv_.wait(), causing the notify to be lost and the
    // waiter to hang forever. This is the lost-wakeup pattern documented
    // for shared_spinlock's CV fallback path under high contention.
    //
    // The fix uses cv_.wait(lock, predicate) instead of the bare cv_.wait().
    // The predicate is "state_ != expected" — i.e. wait until the state has
    // changed from what the caller observed before deciding to wait. The
    // predicate is re-evaluated atomically (under cv_mutex_) both before
    // sleeping and on every spurious wakeup, so any concurrent state change
    // observed under cv_mutex_ causes the predicate to return true and the
    // waiter to return promptly without missing the wake.
    //
    // Mirror of distributed_shared_mutex::cv_wait_until(). See also the
    // matching fix to cv_wake_one()/cv_wake_all() below: notify must be
    // called *while holding* cv_mutex_ so that a waiter which has just
    // passed its predicate check is guaranteed to be in cv_.wait() before
    // the notify fires.
    void cv_wait(uint32_t expected) {
        std::unique_lock<std::mutex> lk(cv_mutex_);
        cv_.wait(lk, [&] {
            return state_.load(std::memory_order_acquire) != expected;
        });
    }

    // C-2 fix: notify_one() must be called *while holding* cv_mutex_. The
    // previous pattern — acquire+release cv_mutex_ first, then notify_one()
    // outside the lock — does NOT prevent lost wakeups: the mutex release
    // does not synchronize with the waiter's atomic entry into cv_.wait().
    // Holding the mutex during notify guarantees that any waiter that has
    // just passed its predicate check is in cv_.wait() when the notify
    // fires. (Notifying while holding the mutex is a permitted and standard
    // pattern; it does add minor wakeup latency but eliminates the race.)
    void cv_wake_one() {
        std::lock_guard<std::mutex> lk(cv_mutex_);
        cv_.notify_one();
    }

    void cv_wake_all() {
        std::lock_guard<std::mutex> lk(cv_mutex_);
        cv_.notify_all();
    }

    std::mutex cv_mutex_;
    std::condition_variable cv_;
};

/// RAII scoped guard for shared (reader) locking of shared_spinlock.
/// Supports both blocking lock_shared() and try_lock_shared() modes.
class shared_scoped_lock {
public:
    /// Blocking shared lock: calls lock_shared() unconditionally.
    explicit shared_scoped_lock(shared_spinlock& m) noexcept
        : m_(&m), locked_(true) { m_->lock_shared(); }

    /// Try-shared lock: calls try_lock_shared(); check operator bool().
    shared_scoped_lock(shared_spinlock& m, std::try_to_lock_t) noexcept
        : m_(&m), locked_(m_->try_lock_shared()) {}

    ~shared_scoped_lock() { if (locked_ && m_) m_->unlock_shared(); }

    shared_scoped_lock(const shared_scoped_lock&) = delete;
    shared_scoped_lock& operator=(const shared_scoped_lock&) = delete;

    explicit operator bool() const noexcept { return locked_; }

private:
    shared_spinlock* m_ = nullptr;
    bool locked_ = false;
};

struct alignas(64) aligned_shared_spinlock : shared_spinlock {};

// ============================================================================
// Probing style tags for concurrent_hash_table
// ============================================================================

/// Tag for the default chained-bucket collision resolution (linked lists).
struct chain_probing_tag {};

/// Tag for F14-style SIMD tag-based probing (inline chunk slots with
/// overflow chain). Each chunk holds 14 inline slots with 8-bit hash tags;
/// SIMD instructions can match all 14 tags in one operation, avoiding
/// pointer chasing within the chunk.
struct f14_probing_tag {};

// ============================================================================
// F14 SIMD tag matching — checks all 14 tags in a single SIMD operation
// ============================================================================

namespace f14_detail {

/// Compute the F14 tag from a hash value.
/// Tag encoding: 0x00 = empty slot, 0x01 = tombstone (deleted),
/// 0x80-0xFE = occupied. This reserves 0x00 and 0x01 for special meanings
/// and maps 7 bits of the hash to the occupied range.
inline uint8_t f14_tag(std::size_t h) noexcept {
    return static_cast<uint8_t>(((h >> 57) & 0x7E) | 0x80);
}

static constexpr int kChunkCapacity = 14;
static constexpr uint16_t kFullMask = (1u << kChunkCapacity) - 1;
static constexpr uint8_t kTagEmpty     = 0x00;
static constexpr uint8_t kTagTombstone = 0x01;

// --------------------------------------------------------------------------
// SIMD tag matching — platform dispatch
// --------------------------------------------------------------------------
//
// T-O2: Memory-ordering safety model for the F14 tag array.
//
// `f14_match_tags()` performs a single 16-byte SIMD load of the `tags[]`
// array (SSE2 `_mm_loadu_si128` / NEON `vld1q_u8` / scalar fallback).
// This load is NOT atomic — a concurrent writer could update one tag
// while the reader is mid-load, producing a "torn" read that mixes old
// and new tag values.
//
// Correctness is guaranteed by the seqlock version-recheck pattern in
// the caller (find_f14_* / find_and_pin_lockfree_*):
//   1. Reader loads `chunk.version` (odd = writer in progress).
//   2. Reader does the SIMD tag match + slot pointer dereference.
//   3. Reader re-checks `chunk.version`; if it changed, the entire
//      optimistic read is discarded and retried.
//
// Platform guarantees for the 16-byte SIMD load:
//   - **x86-64 / x86 (SSE2)**: Intel SDM Vol 1 §8.1.1 guarantees that
//     aligned 16-byte accesses are atomic; `_mm_loadu_si128` on
//     naturally-aligned data (which `tags[]` is, being the first member
//     of an aligned struct) is atomic. Even if a concurrent write tears
//     the load, the seqlock re-check catches it. x86's TSO memory model
//     also prevents the version-load from being reordered after the
//     SIMD load. **Fully safe.**
//   - **AArch64 (NEON)**: `vld1q_u8` is a 128-bit load. ARMv8.4+ with
//     LSE2 (Large System Extensions 2) guarantees single-copy atomicity
//     for aligned 128-bit load/store. On ARMv8.0–8.3 without LSE2, the
//     load may tear across 64-bit halves — but the seqlock version
//     re-check still detects the inconsistency because the writer bumps
//     `version` with release ordering after writing tags. The only
//     residual risk is reordering of the version load relative to the
//     SIMD load on ARM's weak memory model; this is mitigated by the
//     `memory_order_acquire` on the version load and `memory_order_release`
//     on the version store. **Safe with seqlock; LSE2 recommended for
//     zero-torn-load overhead.**
//   - **Scalar fallback**: Individual `uint8_t` loads are always atomic
//     on all platforms (C++17 [intro.memory] §4.7). **Fully safe.**
//
// The compile-time `static_assert` in `concurrent_hash_table` (search
// `kF14TagsPlatformSafe`) verifies that at least one of these
// platform paths is active — if none is selected, F14 mode falls back
// to the scalar path which is always safe.
// --------------------------------------------------------------------------

#if defined(__SSE2__) || (defined(_M_X64) && !defined(__aarch64__))
#include <emmintrin.h>

inline uint16_t f14_match_tags(const uint8_t* tags, uint8_t target_tag) {
    __m128i tag_vec = _mm_set1_epi8(static_cast<char>(target_tag));
    __m128i chunk_tags = _mm_loadu_si128(reinterpret_cast<const __m128i*>(tags));
    __m128i cmp = _mm_cmpeq_epi8(tag_vec, chunk_tags);
    return static_cast<uint16_t>(_mm_movemask_epi8(cmp));
}

#elif defined(__aarch64__)
#include <arm_neon.h>

inline uint16_t f14_match_tags(const uint8_t* tags, uint8_t target_tag) {
    uint8x16_t tag_vec = vdupq_n_u8(target_tag);
    uint8x16_t chunk_tags = vld1q_u8(tags);
    uint8x16_t cmp = vceqq_u8(tag_vec, chunk_tags);
    uint8_t tmp[16];
    vst1q_u8(tmp, cmp);
    uint16_t mask = 0;
    for (int i = 0; i < kChunkCapacity; ++i) {
        if (tmp[i]) mask |= static_cast<uint16_t>(1u << i);
    }
    return mask;
}

#else

inline uint16_t f14_match_tags(const uint8_t* tags, uint8_t target_tag) {
    uint16_t mask = 0;
    for (int i = 0; i < kChunkCapacity; ++i) {
        if (tags[i] == target_tag) mask |= static_cast<uint16_t>(1u << i);
    }
    return mask;
}

#endif

/// Count trailing zeros (find first set bit) in a uint16_t.
inline int ctz16(uint16_t mask) noexcept {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_ctz(mask);
#elif defined(_MSC_VER)
    unsigned long idx;
    _BitScanForward(&idx, static_cast<unsigned long>(mask));
    return static_cast<int>(idx);
#else
    int idx = 0;
    while (!(mask & (1u << idx))) ++idx;
    return idx;
#endif
}

/// Population count (count set bits) in a uint16_t.
inline int popcount16(uint16_t mask) noexcept {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_popcount(static_cast<unsigned int>(mask));
#elif defined(_MSC_VER)
    return static_cast<int>(__popcnt(static_cast<unsigned long>(mask)));
#else
    int count = 0;
    while (mask) { mask &= static_cast<uint16_t>(mask - 1); ++count; }
    return count;
#endif
}

} // namespace f14_detail

// ============================================================================
// Concurrent Hash Table
// ============================================================================

template <
    typename Key,
    typename Value,
    typename Hash = std::hash<Key>,
    typename KeyEqual = std::equal_to<Key>,
    bool EmbeddedChain = false,
    typename ProbingStyle = chain_probing_tag>
class concurrent_hash_table {
public:
    using key_type        = Key;
    using mapped_type     = Value;
    using value_type      = std::pair<const Key, Value>;
    using size_type       = std::size_t;
    using difference_type = std::ptrdiff_t;
    using hasher          = Hash;
    using key_equal       = KeyEqual;
    using reference       = value_type&;
    using const_reference = const value_type&;

    static constexpr bool kIsF14 = std::is_same_v<ProbingStyle, f14_probing_tag>;

    // R2: Expose EmbeddedChain setting for compile-time verification by
    // MM strategies and cache traits. production_sharded_lru_trait asserts
    // this is true to prevent accidental regression to non-EmbeddedChain
    // mode, which degrades to shared-lock reads (use-after-free prevention)
    // and kills read throughput under high concurrency.
    static constexpr bool uses_embedded_chain = EmbeddedChain;

    // T-O2: Compile-time verification that the active platform provides
    // the memory-ordering guarantees required for the F14 SIMD tag read.
    // See the safety-model comment block above `f14_match_tags()`.
    //
    // When F14 mode is active (kIsF14), at least one of these must hold:
    //   - SSE2 available (x86-64 / x86): 16-byte aligned load is atomic
    //     (Intel SDM Vol 1 §8.1.1) + TSO prevents version-load reordering.
    //   - NEON available (AArch64): 128-bit load + seqlock version re-check
    //     detects any torn read. LSE2 (ARMv8.4+) provides zero-overhead
    //     single-copy atomicity but is not required for correctness.
    //   - Scalar fallback: individual uint8_t loads are always atomic
    //     (C++17 [intro.memory] §4.7).
    //
    // If none of these paths is compiled in, F14 mode would silently use
    // an undefined tag-match implementation. The static_assert catches
    // this at compile time. (In practice the `#if/#elif/#else` chain in
    // f14_match_tags always selects one path, so this assert never fires
    // — it is a documentation + regression-prevention device.)
#if defined(__SSE2__) || (defined(_M_X64) && !defined(__aarch64__))
    static constexpr bool kF14TagsPlatformSafe = true;   // SSE2: aligned 16B atomic
#elif defined(__aarch64__)
    static constexpr bool kF14TagsPlatformSafe = true;   // NEON: seqlock-safe
#else
    static constexpr bool kF14TagsPlatformSafe = true;   // scalar: always atomic
#endif
    static_assert(!kIsF14 || kF14TagsPlatformSafe,
                  "F14 mode requires SSE2 (x86-64/x86), NEON (AArch64), "
                  "or the scalar fallback for safe tag matching. "
                  "If you see this, the platform dispatch in "
                  "f14_match_tags() is broken.");

    // P2-6 (T2.5): Upper bound on optimistic-read retry attempts before
    // falling back to the hazptr mid path or shared-lock slow path. Each
    // retry re-issues the seqlock load + version-stamped validation, so
    // an unbounded retry loop could spin indefinitely under sustained
    // write contention (e.g. concurrent inserts causing continuous
    // bucket version bumps). 16 retries × ~1µs per retry bounds the
    // worst-case optimistic-read latency to ~16µs before degrading to
    // the (correctness-guaranteed) slow path. Each retry after the first
    // also executes `LRU_SPIN_PAUSE()` to reduce cache-line contention
    // with the writer holding the seqlock.
    static constexpr int kOptimisticReadMaxRetries = 16;

    // C-3: Upper bound on the number of chain nodes a single hazptr/EBR
    // walk will visit before bailing to the shared-lock path. This is a
    // safety net against the rare case where a concurrent rehash_step()
    // mutates hash_chain_next() pointers mid-walk and briefly forms a
    // cycle in the chain we are traversing — without this bound the
    // while(curr) loop would hang. 128 is 2x the bound used by
    // is_reachable() (which uses 64), comfortably covering any
    // legitimate chain length while still catching pathological cases
    // quickly. Real cache chains are typically <8 nodes (the hash table
    // grows to keep load factor low), so 128 is far above any expected
    // chain length.
    static constexpr int kMaxWalkSteps = 128;

    // G18: Upper bound on the number of dual-array lookup retries in
    // find_f14_dual_array() during incremental rehash. Each retry must
    // acquire a per-bucket shared_bucket_lock and re-validate
    // rehash_progress_; if the rehash thread migrates the target bucket
    // between the initial progress read and the in-lock re-check, the
    // reader retries. Under high-frequency rehash advancement this loop
    // would spin indefinitely, wasting CPU on lock acquisitions that
    // never converge. 8 retries bounds the worst-case dual-array lookup
    // latency before falling back to a deterministic both-arrays search
    // (see find_f14_dual_array). Each retry is heavier than an
    // optimistic-read retry (which is pure atomic loads), so 8 (vs 16
    // for kOptimisticReadMaxRetries) is the right ceiling.
    static constexpr int kDualArrayMaxRetries = 8;

    // P0-4 (T1.2): Upper bound on the number of buckets/chunks that
    // rehash_finish() will migrate synchronously in a single call. Any
    // remaining work is left for subsequent calls or the background
    // rehash balancer. 64 buckets/chunks per call keeps the worst-case
    // rehash_finish duration bounded to a few microseconds on modern
    // hardware, while still completing small rehash (≤64 buckets
    // remaining) inline without scheduler delay. Tune higher (e.g. 256)
    // if your workload tolerates larger tail latency but you want
    // fewer stall events; tune lower (e.g. 16) for stricter latency SLAs.
    static constexpr size_type kRehashFinishMaxBucketsPerCall = 64;

    using allocate_fn   = void*(*)(std::size_t);
    using deallocate_fn = void(*)(void*);

    static constexpr size_type entry_overhead = EmbeddedChain ? 0 : sizeof(value_type);

    // ========================================================================
    // Bucket lock — RAII guard for a single bucket's spinlock (exclusive)
    // ========================================================================
    class bucket_lock {
    public:
        explicit bucket_lock(aligned_spinlock& spin) noexcept
            : spin_(&spin), owns_(true) { spin_->lock(); }
        bucket_lock() noexcept : spin_(nullptr), owns_(false) {}
        ~bucket_lock() { if (owns_ && spin_) spin_->unlock(); }

        bucket_lock(bucket_lock&& other) noexcept
            : spin_(other.spin_), owns_(other.owns_) {
            other.owns_ = false; other.spin_ = nullptr;
        }
        bucket_lock& operator=(bucket_lock&& other) noexcept {
            if (this != &other) {
                if (owns_ && spin_) spin_->unlock();
                spin_ = other.spin_; owns_ = other.owns_;
                other.owns_ = false; other.spin_ = nullptr;
            }
            return *this;
        }
        bucket_lock(const bucket_lock&) = delete;
        bucket_lock& operator=(const bucket_lock&) = delete;

        void unlock() noexcept { if (owns_ && spin_) { spin_->unlock(); owns_ = false; } }
        bool owns_lock() const noexcept { return owns_; }
    private:
        aligned_spinlock* spin_;
        bool owns_;
    };

    // ========================================================================
    // Shared bucket lock — RAII guard for shared (reader) access
    // ========================================================================
    class shared_bucket_lock {
    public:
        explicit shared_bucket_lock(aligned_shared_spinlock& spin) noexcept
            : spin_(&spin), owns_(true) { spin_->lock_shared(); }
        shared_bucket_lock() noexcept : spin_(nullptr), owns_(false) {}
        ~shared_bucket_lock() { if (owns_ && spin_) spin_->unlock_shared(); }

        shared_bucket_lock(shared_bucket_lock&& other) noexcept
            : spin_(other.spin_), owns_(other.owns_) {
            other.owns_ = false; other.spin_ = nullptr;
        }
        shared_bucket_lock& operator=(shared_bucket_lock&& other) noexcept {
            if (this != &other) {
                if (owns_ && spin_) spin_->unlock_shared();
                spin_ = other.spin_; owns_ = other.owns_;
                other.owns_ = false; other.spin_ = nullptr;
            }
            return *this;
        }
        shared_bucket_lock(const shared_bucket_lock&) = delete;
        shared_bucket_lock& operator=(const shared_bucket_lock&) = delete;

        void unlock() noexcept { if (owns_ && spin_) { spin_->unlock_shared(); owns_ = false; } }
        bool owns_lock() const noexcept { return owns_; }
    private:
        aligned_shared_spinlock* spin_;
        bool owns_;
    };

    // ========================================================================
    // Exclusive bucket lock — RAII guard for exclusive (writer) access
    // ========================================================================
    class exclusive_bucket_lock {
    public:
        explicit exclusive_bucket_lock(aligned_shared_spinlock& spin) noexcept
            : spin_(&spin), owns_(true) { spin_->lock(); }
        exclusive_bucket_lock() noexcept : spin_(nullptr), owns_(false) {}
        ~exclusive_bucket_lock() { if (owns_ && spin_) spin_->unlock(); }

        exclusive_bucket_lock(exclusive_bucket_lock&& other) noexcept
            : spin_(other.spin_), owns_(other.owns_) {
            other.owns_ = false; other.spin_ = nullptr;
        }
        exclusive_bucket_lock& operator=(exclusive_bucket_lock&& other) noexcept {
            if (this != &other) {
                if (owns_ && spin_) spin_->unlock();
                spin_ = other.spin_; owns_ = other.owns_;
                other.owns_ = false; other.spin_ = nullptr;
            }
            return *this;
        }
        exclusive_bucket_lock(const exclusive_bucket_lock&) = delete;
        exclusive_bucket_lock& operator=(const exclusive_bucket_lock&) = delete;

        void unlock() noexcept { if (owns_ && spin_) { spin_->unlock(); owns_ = false; } }
        bool owns_lock() const noexcept { return owns_; }
    private:
        aligned_shared_spinlock* spin_;
        bool owns_;
    };

    using scoped_lock = exclusive_bucket_lock;

    // ========================================================================
    // Automatic bucket sizing
    // ========================================================================

    struct from_expected_items_t { explicit from_expected_items_t() = default; };
    static constexpr from_expected_items_t from_expected_items{};

    static constexpr size_type buckets_for_items(size_type expected_items) noexcept {
        if constexpr (kIsF14) {
            size_type chunks = expected_items / 10;
            return next_power_of_two(chunks > 0 ? chunks : 1);
        } else {
            return next_power_of_two(expected_items * 4);
        }
    }

    // ========================================================================
    // Constructor / Destructor
    // ========================================================================

    explicit concurrent_hash_table(
        size_type num_buckets = 1024,
        const Hash& hash = Hash(),
        const KeyEqual& equal = KeyEqual())
        : hash_(hash), equal_(equal)
        , bucket_mask_(next_power_of_two(num_buckets) - 1), size_(0)
    {
        auto count = next_power_of_two(num_buckets);
        if constexpr (kIsF14) {
            chunks_ = std::make_unique<f14_chunk_type[]>(count);
        } else {
            buckets_ = std::make_unique<bucket_type[]>(count);
        }
    }

    concurrent_hash_table(
        from_expected_items_t tag,
        size_type expected_items,
        const Hash& hash = Hash(),
        const KeyEqual& equal = KeyEqual())
        : hash_(hash), equal_(equal)
        , bucket_mask_(buckets_for_items(expected_items) - 1), size_(0)
    {
        (void)tag;
        auto count = buckets_for_items(expected_items);
        if constexpr (kIsF14) {
            chunks_ = std::make_unique<f14_chunk_type[]>(count);
        } else {
            buckets_ = std::make_unique<bucket_type[]>(count);
        }
    }

    concurrent_hash_table(
        size_type num_buckets,
        allocate_fn alloc_fn,
        deallocate_fn dealloc_fn,
        const Hash& hash = Hash(),
        const KeyEqual& equal = KeyEqual())
        : hash_(hash), equal_(equal)
        , bucket_mask_(next_power_of_two(num_buckets) - 1), size_(0)
        , alloc_fn_(alloc_fn), dealloc_fn_(dealloc_fn)
    {
        auto count = next_power_of_two(num_buckets);
        if constexpr (kIsF14) {
            chunks_ = std::make_unique<f14_chunk_type[]>(count);
        } else {
            buckets_ = std::make_unique<bucket_type[]>(count);
        }
    }

    concurrent_hash_table(
        from_expected_items_t tag,
        size_type expected_items,
        allocate_fn alloc_fn,
        deallocate_fn dealloc_fn,
        const Hash& hash = Hash(),
        const KeyEqual& equal = KeyEqual())
        : hash_(hash), equal_(equal)
        , bucket_mask_(buckets_for_items(expected_items) - 1), size_(0)
        , alloc_fn_(alloc_fn), dealloc_fn_(dealloc_fn)
    {
        (void)tag;
        auto count = buckets_for_items(expected_items);
        if constexpr (kIsF14) {
            chunks_ = std::make_unique<f14_chunk_type[]>(count);
        } else {
            buckets_ = std::make_unique<bucket_type[]>(count);
        }
    }

    ~concurrent_hash_table() { clear(); }

    concurrent_hash_table(const concurrent_hash_table&) = delete;
    concurrent_hash_table& operator=(const concurrent_hash_table&) = delete;

    concurrent_hash_table(concurrent_hash_table&& other) noexcept
        : hash_(std::move(other.hash_))
        , equal_(std::move(other.equal_))
        , alloc_fn_(std::move(other.alloc_fn_))
        , dealloc_fn_(std::move(other.dealloc_fn_))
        , bucket_mask_(other.bucket_mask_.load(std::memory_order_relaxed))
        , buckets_(std::move(other.buckets_))
        , chunks_(std::move(other.chunks_))
        , size_(other.size_.load(std::memory_order_relaxed))
        , seqlock_(other.seqlock_.load(std::memory_order_relaxed))
        , max_load_factor_(other.max_load_factor_)
        , enable_optimistic_read_(other.enable_optimistic_read_)
        , enable_hazptr_read_(other.enable_hazptr_read_)
        , retired_buckets_(std::move(other.retired_buckets_))
        , retired_chunks_(std::move(other.retired_chunks_))
        , incremental_rehash_(other.incremental_rehash_)
        , rehash_in_progress_(other.rehash_in_progress_.load(std::memory_order_relaxed))
        , rehash_new_buckets_(other.rehash_new_buckets_.load(std::memory_order_relaxed))
        , rehash_new_bucket_count_(other.rehash_new_bucket_count_.load(std::memory_order_relaxed))
        , rehash_progress_(other.rehash_progress_.load(std::memory_order_relaxed))
        , rehash_migrating_(other.rehash_migrating_.load(std::memory_order_relaxed))
        , rehash_new_buckets_owner_(std::move(other.rehash_new_buckets_owner_))
        , rehash_new_chunks_(other.rehash_new_chunks_.load(std::memory_order_relaxed))
        , rehash_new_chunks_owner_(std::move(other.rehash_new_chunks_owner_))
    {
        other.bucket_mask_.store(0, std::memory_order_relaxed);
        other.size_.store(0, std::memory_order_relaxed);
        other.rehash_in_progress_.store(false, std::memory_order_relaxed);
        other.rehash_new_buckets_.store(nullptr, std::memory_order_relaxed);
        other.rehash_new_bucket_count_.store(0, std::memory_order_relaxed);
        other.rehash_progress_.store(0, std::memory_order_relaxed);
        other.rehash_migrating_.store(false, std::memory_order_relaxed);
        other.rehash_new_chunks_.store(nullptr, std::memory_order_relaxed);
    }

    concurrent_hash_table& operator=(concurrent_hash_table&& other) noexcept {
        if (this != &other) {
            clear();
            hash_ = std::move(other.hash_);
            equal_ = std::move(other.equal_);
            alloc_fn_ = std::move(other.alloc_fn_);
            dealloc_fn_ = std::move(other.dealloc_fn_);
            bucket_mask_.store(other.bucket_mask_.load(std::memory_order_relaxed), std::memory_order_relaxed);
            buckets_ = std::move(other.buckets_);
            chunks_ = std::move(other.chunks_);
            size_.store(other.size_.load(std::memory_order_relaxed), std::memory_order_relaxed);
            seqlock_.store(other.seqlock_.load(std::memory_order_relaxed), std::memory_order_relaxed);
            max_load_factor_ = other.max_load_factor_;
            enable_optimistic_read_ = other.enable_optimistic_read_;
            enable_hazptr_read_ = other.enable_hazptr_read_;
            retired_buckets_ = std::move(other.retired_buckets_);
            retired_chunks_ = std::move(other.retired_chunks_);
            incremental_rehash_ = other.incremental_rehash_;
            rehash_in_progress_.store(other.rehash_in_progress_.load(std::memory_order_relaxed), std::memory_order_relaxed);
            rehash_new_buckets_.store(other.rehash_new_buckets_.load(std::memory_order_relaxed), std::memory_order_relaxed);
            rehash_new_bucket_count_.store(other.rehash_new_bucket_count_.load(std::memory_order_relaxed), std::memory_order_relaxed);
            rehash_progress_.store(other.rehash_progress_.load(std::memory_order_relaxed), std::memory_order_relaxed);
            rehash_migrating_.store(other.rehash_migrating_.load(std::memory_order_relaxed), std::memory_order_relaxed);
            rehash_new_buckets_owner_ = std::move(other.rehash_new_buckets_owner_);
            rehash_new_chunks_.store(other.rehash_new_chunks_.load(std::memory_order_relaxed), std::memory_order_relaxed);
            rehash_new_chunks_owner_ = std::move(other.rehash_new_chunks_owner_);
            other.bucket_mask_.store(0, std::memory_order_relaxed);
            other.size_.store(0, std::memory_order_relaxed);
            other.rehash_in_progress_.store(false, std::memory_order_relaxed);
            other.rehash_new_buckets_.store(nullptr, std::memory_order_relaxed);
            other.rehash_new_bucket_count_.store(0, std::memory_order_relaxed);
            other.rehash_progress_.store(0, std::memory_order_relaxed);
            other.rehash_migrating_.store(false, std::memory_order_relaxed);
            other.rehash_new_chunks_.store(nullptr, std::memory_order_relaxed);
        }
        return *this;
    }

    // ========================================================================
    // Lookup
    // ========================================================================

    auto find(const Key& key) {
        const size_type h = hash_(key);
        if constexpr (kIsF14) {
            // F14 incremental rehash: dual-array aware lookup
            if (incremental_rehash_ && rehash_in_progress_.load(std::memory_order_acquire)) {
                if constexpr (EmbeddedChain) {
                    return find_f14_dual_array(key, h);
                } else {
                    node_type* node = find_f14_dual_array(key, h);
                    return node ? &(node->value) : nullptr;
                }
            }
            if constexpr (EmbeddedChain) {
                // Fast path: optimistic read (atomic loads only, no RMW)
                if (enable_optimistic_read_) {
                    for (int _retry = 0; _retry < kOptimisticReadMaxRetries; ++_retry) { if (_retry) LRU_SPIN_PAUSE();
                        auto sl1 = seqlock_.load(std::memory_order_acquire);
                        if (sl1 & 1u) continue;
                        size_type idx = h & bucket_mask_.load(std::memory_order_acquire);
                        auto v1 = chunks_[idx].version.load(std::memory_order_acquire);
                        Value node = find_f14_embedded(key, idx);
                        auto v2 = chunks_[idx].version.load(std::memory_order_acquire);
                        auto sl2 = seqlock_.load(std::memory_order_acquire);
                        if (sl1 == sl2 && (v1 == v2) && (v1 & 1u) == 0) return node;
                    }
                }
                // Mid path: hazptr traversal (wait-free, no RMW)
                // P0-2 fix: validate against blocking rehash using seqlock.
                // Without this, find_f14_hazptr_embedded would race with
                // blocking rehash's non-atomic chunks_ swap (data race on
                // the unique_ptr's stored pointer, which is UB even on x86-64).
                if (enable_hazptr_read_) {
                    auto sl1_haz = seqlock_.load(std::memory_order_acquire);
                    if ((sl1_haz & 1u) == 0) {  // Even — no rehash in progress
                        size_type idx = h & bucket_mask_.load(std::memory_order_acquire);
                        Value node = find_f14_hazptr_embedded(key, idx);
                        auto sl2_haz = seqlock_.load(std::memory_order_acquire);
                        if (sl1_haz == sl2_haz && node) return node;
                    }
                    // Rehash in progress or key not found — fall through to
                    // the shared-lock slow path, which is rehash-safe.
                }
                // Slow path: shared lock (RMW, cache line bouncing)
                size_type idx = h & bucket_mask_.load(std::memory_order_relaxed);
                auto guard = lock_bucket_shared(idx);
                return find_f14_embedded(key, idx);
            } else {
                if (enable_optimistic_read_) {
                    for (int _retry = 0; _retry < kOptimisticReadMaxRetries; ++_retry) { if (_retry) LRU_SPIN_PAUSE();
                        auto sl1 = seqlock_.load(std::memory_order_acquire);
                        if (sl1 & 1u) continue;
                        size_type idx = h & bucket_mask_.load(std::memory_order_acquire);
                        auto v1 = chunks_[idx].version.load(std::memory_order_acquire);
                        node_type* node = find_f14_node(key, idx);
                        auto v2 = chunks_[idx].version.load(std::memory_order_acquire);
                        auto sl2 = seqlock_.load(std::memory_order_acquire);
                        if (sl1 == sl2 && (v1 == v2) && (v1 & 1u) == 0)
                            return node ? &(node->value) : nullptr;
                    }
                }
                size_type idx = h & bucket_mask_.load(std::memory_order_relaxed);
                auto guard = lock_bucket_shared(idx);
                return find_locked(key, idx);
            }
        } else {
            // Incremental rehash path for chain mode: use shared lock with
            // dual-array awareness. Optimistic/hazptr paths are skipped during
            // incremental rehash because items may be split across two arrays.
            if (incremental_rehash_ && rehash_in_progress_.load(std::memory_order_acquire)) {
                for (;;) {
                    size_type old_idx = h & bucket_mask_.load(std::memory_order_acquire);
                    size_type progress = rehash_progress_.load(std::memory_order_acquire);
                    if (old_idx < progress) {
                        // Bucket has been migrated — look in new array
                        size_type new_mask = rehash_new_bucket_count_.load(std::memory_order_acquire) - 1;
                        size_type new_idx = h & new_mask;
                        bucket_type* new_bkts = rehash_new_buckets_.load(std::memory_order_acquire);
                        auto guard = shared_bucket_lock(new_bkts[new_idx].spin);
                        if constexpr (EmbeddedChain) {
                            return find_node_embedded_in(key, new_bkts, new_idx);
                        } else {
                            node_type* node = find_node_in(key, new_bkts, new_idx);
                            return node ? &(node->value) : nullptr;
                        }
                    } else {
                        // Not yet migrated — look in old array
                        auto guard = lock_bucket_shared(old_idx);
                        // Re-check: bucket might have been migrated while we waited
                        if (rehash_progress_.load(std::memory_order_acquire) <= old_idx) {
                            if constexpr (EmbeddedChain) {
                                return find_node_embedded(key, old_idx);
                            } else {
                                return find_locked(key, old_idx);
                            }
                        }
                        // Bucket was migrated, retry
                    }
                }
            }
            if constexpr (EmbeddedChain) {
                // Fast path: optimistic read (atomic loads only, no RMW)
                if (enable_optimistic_read_) {
                    for (int _retry = 0; _retry < kOptimisticReadMaxRetries; ++_retry) { if (_retry) LRU_SPIN_PAUSE();
                        auto sl1 = seqlock_.load(std::memory_order_acquire);
                        if (sl1 & 1u) continue;
                        size_type idx = h & bucket_mask_.load(std::memory_order_acquire);
                        auto v1 = buckets_[idx].version.load(std::memory_order_acquire);
                        Value node = find_node_embedded(key, idx);
                        auto v2 = buckets_[idx].version.load(std::memory_order_acquire);
                        auto sl2 = seqlock_.load(std::memory_order_acquire);
                        if (sl1 == sl2 && (v1 == v2) && (v1 & 1u) == 0) return node;
                    }
                }
                // Mid path: hazptr traversal (wait-free, no RMW)
                // P0-2 fix: validate against blocking rehash using seqlock.
                // Without this, find_hazptr_embedded would race with blocking
                // rehash's non-atomic buckets_ swap (data race on the
                // unique_ptr's stored pointer, which is UB even on x86-64).
                // Pattern mirrors the optimistic-read validation above.
                if (enable_hazptr_read_) {
                    auto sl1_haz = seqlock_.load(std::memory_order_acquire);
                    if ((sl1_haz & 1u) == 0) {  // Even — no rehash in progress
                        size_type idx = h & bucket_mask_.load(std::memory_order_acquire);
                        Value node = find_hazptr_embedded(key, idx);
                        auto sl2_haz = seqlock_.load(std::memory_order_acquire);
                        if (sl1_haz == sl2_haz && node) return node;
                    }
                    // Rehash in progress or key not found — fall through to
                    // the shared-lock slow path, which is rehash-safe.
                }
                // Slow path: shared lock (RMW, cache line bouncing)
                auto sl = seqlock_.load(std::memory_order_acquire);
                size_type idx = h & bucket_mask_.load(std::memory_order_relaxed);
                auto guard = lock_bucket_shared(idx);
                return find_node_embedded(key, idx);
            } else {
                if (enable_optimistic_read_) {
                    for (int _retry = 0; _retry < kOptimisticReadMaxRetries; ++_retry) { if (_retry) LRU_SPIN_PAUSE();
                        auto sl1 = seqlock_.load(std::memory_order_acquire);
                        if (sl1 & 1u) continue;
                        size_type idx = h & bucket_mask_.load(std::memory_order_acquire);
                        auto v1 = buckets_[idx].version.load(std::memory_order_acquire);
                        node_type* node = find_node(key, idx);
                        auto v2 = buckets_[idx].version.load(std::memory_order_acquire);
                        auto sl2 = seqlock_.load(std::memory_order_acquire);
                        if (sl1 == sl2 && (v1 == v2) && (v1 & 1u) == 0)
                            return node ? &(node->value) : nullptr;
                    }
                }
                auto sl = seqlock_.load(std::memory_order_acquire);
                size_type idx = h & bucket_mask_.load(std::memory_order_relaxed);
                auto guard = lock_bucket_shared(idx);
                return find_locked(key, idx);
            }
        }
    }

    auto find(const Key& key) const {
        const size_type h = hash_(key);
        if constexpr (kIsF14) {
            // F14 incremental rehash: dual-array aware lookup
            if (incremental_rehash_ && rehash_in_progress_.load(std::memory_order_acquire)) {
                if constexpr (EmbeddedChain) {
                    return find_f14_dual_array(key, h);
                } else {
                    node_type* node = find_f14_dual_array(key, h);
                    return node ? &(node->value) : nullptr;
                }
            }
            if constexpr (EmbeddedChain) {
                // Fast path: optimistic read (atomic loads only, no RMW)
                if (enable_optimistic_read_) {
                    for (int _retry = 0; _retry < kOptimisticReadMaxRetries; ++_retry) { if (_retry) LRU_SPIN_PAUSE();
                        auto sl1 = seqlock_.load(std::memory_order_acquire);
                        if (sl1 & 1u) continue;
                        size_type idx = h & bucket_mask_.load(std::memory_order_acquire);
                        auto v1 = chunks_[idx].version.load(std::memory_order_acquire);
                        Value node = find_f14_embedded(key, idx);
                        auto v2 = chunks_[idx].version.load(std::memory_order_acquire);
                        auto sl2 = seqlock_.load(std::memory_order_acquire);
                        if (sl1 == sl2 && (v1 == v2) && (v1 & 1u) == 0) return node;
                    }
                }
                // Mid path: hazptr traversal (wait-free, no RMW)
                // P0-2 fix: validate against blocking rehash using seqlock.
                // Without this, find_f14_hazptr_embedded would race with
                // blocking rehash's non-atomic chunks_ swap (data race on
                // the unique_ptr's stored pointer, which is UB even on x86-64).
                if (enable_hazptr_read_) {
                    auto sl1_haz = seqlock_.load(std::memory_order_acquire);
                    if ((sl1_haz & 1u) == 0) {  // Even — no rehash in progress
                        size_type idx = h & bucket_mask_.load(std::memory_order_acquire);
                        Value node = find_f14_hazptr_embedded(key, idx);
                        auto sl2_haz = seqlock_.load(std::memory_order_acquire);
                        if (sl1_haz == sl2_haz && node) return node;
                    }
                    // Rehash in progress or key not found — fall through to
                    // the shared-lock slow path, which is rehash-safe.
                }
                // Slow path: shared lock (RMW, cache line bouncing)
                size_type idx = h & bucket_mask_.load(std::memory_order_relaxed);
                auto guard = const_cast<concurrent_hash_table*>(this)->lock_bucket_shared(idx);
                return find_f14_embedded(key, idx);
            } else {
                if (enable_optimistic_read_) {
                    for (int _retry = 0; _retry < kOptimisticReadMaxRetries; ++_retry) { if (_retry) LRU_SPIN_PAUSE();
                        auto sl1 = seqlock_.load(std::memory_order_acquire);
                        if (sl1 & 1u) continue;
                        size_type idx = h & bucket_mask_.load(std::memory_order_acquire);
                        auto v1 = chunks_[idx].version.load(std::memory_order_acquire);
                        node_type* node = find_f14_node(key, idx);
                        auto v2 = chunks_[idx].version.load(std::memory_order_acquire);
                        auto sl2 = seqlock_.load(std::memory_order_acquire);
                        if (sl1 == sl2 && (v1 == v2) && (v1 & 1u) == 0)
                            return node ? &(node->value) : nullptr;
                    }
                }
                size_type idx = h & bucket_mask_.load(std::memory_order_relaxed);
                auto guard = const_cast<concurrent_hash_table*>(this)->lock_bucket_shared(idx);
                return find_locked(key, idx);
            }
        } else {
            // Incremental rehash path for chain mode (const)
            if (incremental_rehash_ && rehash_in_progress_.load(std::memory_order_acquire)) {
                for (;;) {
                    size_type old_idx = h & bucket_mask_.load(std::memory_order_acquire);
                    size_type progress = rehash_progress_.load(std::memory_order_acquire);
                    if (old_idx < progress) {
                        size_type new_mask = rehash_new_bucket_count_.load(std::memory_order_acquire) - 1;
                        size_type new_idx = h & new_mask;
                        bucket_type* new_bkts = rehash_new_buckets_.load(std::memory_order_acquire);
                        auto guard = shared_bucket_lock(new_bkts[new_idx].spin);
                        if constexpr (EmbeddedChain) {
                            return find_node_embedded_in(key, new_bkts, new_idx);
                        } else {
                            node_type* node = find_node_in(key, new_bkts, new_idx);
                            return node ? &(node->value) : nullptr;
                        }
                    } else {
                        auto guard = const_cast<concurrent_hash_table*>(this)->lock_bucket_shared(old_idx);
                        if (rehash_progress_.load(std::memory_order_acquire) <= old_idx) {
                            if constexpr (EmbeddedChain) {
                                return find_node_embedded(key, old_idx);
                            } else {
                                return find_locked(key, old_idx);
                            }
                        }
                    }
                }
            }
            if constexpr (EmbeddedChain) {
                // Fast path: optimistic read (atomic loads only, no RMW)
                if (enable_optimistic_read_) {
                    for (int _retry = 0; _retry < kOptimisticReadMaxRetries; ++_retry) { if (_retry) LRU_SPIN_PAUSE();
                        auto sl1 = seqlock_.load(std::memory_order_acquire);
                        if (sl1 & 1u) continue;
                        size_type idx = h & bucket_mask_.load(std::memory_order_acquire);
                        auto v1 = buckets_[idx].version.load(std::memory_order_acquire);
                        Value node = find_node_embedded(key, idx);
                        auto v2 = buckets_[idx].version.load(std::memory_order_acquire);
                        auto sl2 = seqlock_.load(std::memory_order_acquire);
                        if (sl1 == sl2 && (v1 == v2) && (v1 & 1u) == 0) return node;
                    }
                }
                // Mid path: hazptr traversal (wait-free, no RMW)
                // P0-2 fix: validate against blocking rehash using seqlock.
                // Without this, find_hazptr_embedded would race with blocking
                // rehash's non-atomic buckets_ swap (data race on the
                // unique_ptr's stored pointer, which is UB even on x86-64).
                // Pattern mirrors the optimistic-read validation above.
                if (enable_hazptr_read_) {
                    auto sl1_haz = seqlock_.load(std::memory_order_acquire);
                    if ((sl1_haz & 1u) == 0) {  // Even — no rehash in progress
                        size_type idx = h & bucket_mask_.load(std::memory_order_acquire);
                        Value node = find_hazptr_embedded(key, idx);
                        auto sl2_haz = seqlock_.load(std::memory_order_acquire);
                        if (sl1_haz == sl2_haz && node) return node;
                    }
                    // Rehash in progress or key not found — fall through to
                    // the shared-lock slow path, which is rehash-safe.
                }
                // Slow path: shared lock (RMW, cache line bouncing)
                auto sl = seqlock_.load(std::memory_order_acquire);
                size_type idx = h & bucket_mask_.load(std::memory_order_relaxed);
                auto guard = const_cast<concurrent_hash_table*>(this)->lock_bucket_shared(idx);
                return find_node_embedded(key, idx);
            } else {
                if (enable_optimistic_read_) {
                    for (int _retry = 0; _retry < kOptimisticReadMaxRetries; ++_retry) { if (_retry) LRU_SPIN_PAUSE();
                        auto sl1 = seqlock_.load(std::memory_order_acquire);
                        if (sl1 & 1u) continue;
                        size_type idx = h & bucket_mask_.load(std::memory_order_acquire);
                        auto v1 = buckets_[idx].version.load(std::memory_order_acquire);
                        node_type* node = find_node(key, idx);
                        auto v2 = buckets_[idx].version.load(std::memory_order_acquire);
                        auto sl2 = seqlock_.load(std::memory_order_acquire);
                        if (sl1 == sl2 && (v1 == v2) && (v1 & 1u) == 0)
                            return node ? &(node->value) : nullptr;
                    }
                }
                auto sl = seqlock_.load(std::memory_order_acquire);
                size_type idx = h & bucket_mask_.load(std::memory_order_relaxed);
                auto guard = const_cast<concurrent_hash_table*>(this)->lock_bucket_shared(idx);
                return find_locked(key, idx);
            }
        }
    }

    auto find_locked(const Key& key, size_type bucket_idx) {
        if constexpr (kIsF14) {
            if constexpr (EmbeddedChain) {
                return find_f14_embedded(key, bucket_idx);
            } else {
                node_type* node = find_f14_node(key, bucket_idx);
                return node ? &(node->value) : nullptr;
            }
        } else {
            if constexpr (EmbeddedChain) {
                return find_node_embedded(key, bucket_idx);
            } else {
                node_type* node = find_node(key, bucket_idx);
                return node ? &(node->value) : nullptr;
            }
        }
    }

    auto find_locked(const Key& key, size_type bucket_idx) const {
        if constexpr (kIsF14) {
            if constexpr (EmbeddedChain) {
                return find_f14_embedded(key, bucket_idx);
            } else {
                node_type* node = find_f14_node(key, bucket_idx);
                return node ? &(node->value) : nullptr;
            }
        } else {
            if constexpr (EmbeddedChain) {
                return find_node_embedded(key, bucket_idx);
            } else {
                node_type* node = find_node(key, bucket_idx);
                return node ? &(node->value) : nullptr;
            }
        }
    }

    // ========================================================================
    // Find-and-pin: atomic lookup + pin under bucket shared lock
    // ========================================================================

    /// Find a key and atomically pin the found item before releasing the
    /// bucket shared lock. This eliminates the TOCTOU window between a
    /// plain find() returning and the caller incrementing the item's
    /// refcount, which is critical for lock-free read paths.
    ///
    /// PinFn is invoked with the found Value while the bucket shared lock
    /// is still held. If pin_fn returns true, the item is considered pinned
    /// and this function returns the Value. If pin_fn returns false (or the
    /// key is not found), this function returns nullptr.
    ///
    /// This is the fallback path using bucket shared lock (RMW on spinlock).
    /// For the optimistic read path, see find_and_pin_optimistic().
    template <typename PinFn>
    Value find_and_pin(const Key& key, PinFn&& pin_fn) {
        return find_and_pin_with_hash(key, hash_(key), std::forward<PinFn>(pin_fn));
    }

    /// T16.4: find_and_pin with a pre-computed hash. The hash MUST be the
    /// result of `hash_(key)` (i.e., the same `Hash` instance the table
    /// uses). Mismatched hashes lead to undefined behavior (wrong bucket,
    /// silent data loss). Used by bulk_get to avoid re-hashing the same
    /// key for shard dispatch + stripe lock + hash-table lookup.
    template <typename PinFn>
    Value find_and_pin_with_hash(const Key& key, size_type h, PinFn&& pin_fn) {
        if constexpr (kIsF14) {
            // F14 incremental rehash: dual-array aware find-and-pin
            if (incremental_rehash_ && rehash_in_progress_.load(std::memory_order_acquire)) {
                if constexpr (EmbeddedChain) {
                    Value node = find_f14_dual_array(key, h);
                    if (node && pin_fn(node)) return node;
                } else {
                    node_type* node = find_f14_dual_array(key, h);
                    if (node && pin_fn(&(node->value))) return &(node->value);
                }
                return nullptr;
            }
            // P3-1: Use acquire (not relaxed) to pair with the writer's
            // release store on bucket_mask_. This guarantees that if we see
            // the new mask, we also see the new chunks_ array installed by
            // the rehash. Without acquire, the relaxed load could observe
            // the new bucket_mask_ while chunks_ is still the old (smaller)
            // array — or, in the pre-P3-1 swap-fix code, while chunks_ is
            // briefly null — leading to an out-of-bounds access or null
            // dereference.
            size_type idx = h & bucket_mask_.load(std::memory_order_acquire);
            auto guard = lock_bucket_shared(idx);
            if constexpr (EmbeddedChain) {
                Value node = find_f14_embedded(key, idx);
                if (node && pin_fn(node)) return node;
            } else {
                node_type* node = find_f14_node(key, idx);
                if (node && pin_fn(&(node->value))) return &(node->value);
            }
        } else {
            // Incremental rehash path for chain mode find_and_pin
            if (incremental_rehash_ && rehash_in_progress_.load(std::memory_order_acquire)) {
                for (;;) {
                    size_type old_idx = h & bucket_mask_.load(std::memory_order_acquire);
                    size_type progress = rehash_progress_.load(std::memory_order_acquire);
                    if (old_idx < progress) {
                        size_type new_mask = rehash_new_bucket_count_.load(std::memory_order_acquire) - 1;
                        size_type new_idx = h & new_mask;
                        bucket_type* new_bkts = rehash_new_buckets_.load(std::memory_order_acquire);
                        auto guard = shared_bucket_lock(new_bkts[new_idx].spin);
                        if (rehash_progress_.load(std::memory_order_acquire) >= progress) {
                            if constexpr (EmbeddedChain) {
                                Value node = find_node_embedded_in(key, new_bkts, new_idx);
                                if (node && pin_fn(node)) return node;
                            } else {
                                node_type* node = find_node_in(key, new_bkts, new_idx);
                                if (node && pin_fn(&(node->value))) return &(node->value);
                            }
                            return nullptr;
                        }
                    } else {
                        // T-P2-1 (R-4): For strictly unmigrated buckets
                        // (old_idx > progress), attempt an optimistic
                        // (lock-free) read to detect misses BEFORE acquiring
                        // the bucket shared lock — mirrors the
                        // find_and_pin_optimistic T-P2-6 incremental-rehash
                        // path. A consistent read with found==false is a
                        // confirmed miss (return nullptr without any lock);
                        // a consistent read with found==true falls through
                        // to the shared-lock path for safe pinning (pinning
                        // has side effects that cannot be rolled back if the
                        // node is concurrently retired). The equality case
                        // (old_idx == progress) is the migration boundary
                        // and conservatively skips optimistic read.
                        //
                        // Safety against concurrent migration of this bucket:
                        //   (1) per-bucket version check (v1 == v2 && even)
                        //       detects any completed modification
                        //       (insert/remove/migrate bumps the version);
                        //   (2) post-read progress re-check
                        //       (progress >= old_idx) closes the brief
                        //       window between head-modification and
                        //       version-bump where the bucket is being
                        //       emptied but the version is still unchanged.
                        // If either check fails, fall through to the
                        // existing shared-lock path.
                        if (enable_optimistic_read_ && old_idx > progress) {
                            for (int _retry = 0;
                                 _retry < kOptimisticReadMaxRetries;
                                 ++_retry) {
                                if (_retry) LRU_SPIN_PAUSE();
                                auto sl1 = seqlock_.load(std::memory_order_acquire);
                                if (sl1 & 1u) continue;  // global rehash mutation in progress
                                auto v1 = buckets_[old_idx].version.load(std::memory_order_acquire);
                                bool found;
                                if constexpr (EmbeddedChain) {
                                    found = find_node_embedded(key, old_idx) != nullptr;
                                } else {
                                    found = find_node(key, old_idx) != nullptr;
                                }
                                auto v2 = buckets_[old_idx].version.load(std::memory_order_acquire);
                                auto sl2 = seqlock_.load(std::memory_order_acquire);
                                if (sl1 == sl2 && (v1 == v2) && (v1 & 1u) == 0) {
                                    // Consistent read — re-check progress
                                    // hasn't moved past this bucket.
                                    if (rehash_progress_.load(std::memory_order_acquire) >= old_idx) {
                                        break;  // bucket reached by migration — fall back to shared lock
                                    }
                                    if (!found) return nullptr;  // Confirmed miss — skip lock entirely
                                    break;  // Confirmed hit — fall through to shared lock for safe pin
                                }
                                // Inconsistent read — retry
                            }
                        }

                        auto guard = lock_bucket_shared(old_idx);

                        if (rehash_progress_.load(std::memory_order_acquire) <= old_idx) {
                            if constexpr (EmbeddedChain) {
                                Value node = find_node_embedded(key, old_idx);
                                if (node && pin_fn(node)) return node;
                            } else {
                                node_type* node = find_node(key, old_idx);
                                if (node && pin_fn(&(node->value))) return &(node->value);
                            }
                            return nullptr;
                        }
                    }
                }
            }
            // P3-1: Use acquire (not relaxed) — same rationale as the F14
            // path above. Pairs with the writer's release store on
            // bucket_mask_ to ensure the new buckets_ array is visible when
            // we see the new mask.
            size_type idx = h & bucket_mask_.load(std::memory_order_acquire);
            auto guard = lock_bucket_shared(idx);
            if constexpr (EmbeddedChain) {
                Value node = find_node_embedded(key, idx);
                if (node && pin_fn(node)) return node;
            } else {
                node_type* node = find_node(key, idx);
                if (node && pin_fn(&(node->value))) return &(node->value);
            }
        }
        return nullptr;
    }

    /// Optimistic find-and-pin: uses optimistic read to quickly determine
    /// if the key is NOT present (avoiding the RMW on bucket spinlock for
    /// misses), then falls back to the shared lock path for actual pinning.
    ///
    /// This is safer than attempting to pin (incRef) under optimistic read,
    /// because pinning has side effects that cannot be safely rolled back
    /// if the node is being concurrently removed and retired — the memory
    /// could be reclaimed before we can unpin. By keeping the optimistic
    /// phase read-only, we avoid this hazard while still gaining the
    /// benefit of lock-free miss detection.
    ///
    /// Supports both EmbeddedChain and node-based chain modes.
    template <typename PinFn, typename UnpinFn>
    Value find_and_pin_optimistic(const Key& key, PinFn&& pin_fn, UnpinFn&& /*unpin_fn*/) {
        const size_type h = hash_(key);

        // T-P2-6: During incremental rehash, allow optimistic (lock-free)
        // reads for buckets NOT in the current migration window instead of
        // unconditionally falling back to the shared-lock find_and_pin path.
        // For chain mode, a bucket with old_idx > rehash_progress_ has not
        // yet been reached by migration and still lives entirely in the old
        // array, so the optimistic read on buckets_[old_idx] is safe. Only
        // buckets already migrated (old_idx <= progress) — or F14 chunks,
        // whose dual-array lookup is handled by find_and_pin — need the
        // shared-lock fallback.
        //
        // Safety against concurrent migration of the target bucket relies on
        // two complementary checks inside the optimistic loop below:
        //   (1) per-bucket version check (v1 == v2 && even) detects any
        //       completed modification — migration bumps the version by 2;
        //   (2) a post-read progress re-check (progress < idx) closes the
        //       brief window between head-modification and version-bump where
        //       the version is unchanged but the bucket is being emptied.
        // `rehash_progress_guard` enables check (2) only on this rehash path.
        bool rehash_progress_guard = false;
        if (incremental_rehash_ && rehash_in_progress_.load(std::memory_order_acquire)) {
            if constexpr (kIsF14) {
                // F14 chunk-based dual-array lookup is handled by find_and_pin.
                return find_and_pin(key, std::forward<PinFn>(pin_fn));
            } else {
                const size_type old_idx = h & bucket_mask_.load(std::memory_order_acquire);
                const size_type progress = rehash_progress_.load(std::memory_order_acquire);
                if (old_idx <= progress) {
                    // Bucket is being migrated or has already been migrated to
                    // the new array — an optimistic read of the old bucket
                    // would observe an empty list (false miss). Fall back to
                    // shared-lock find_and_pin, which has dual-array awareness.
                    return find_and_pin(key, std::forward<PinFn>(pin_fn));
                }
                // old_idx > progress: bucket NOT in the migration window.
                // Fall through to the optimistic read; the per-bucket version
                // check plus the progress re-check below guarantee safety.
                rehash_progress_guard = true;
            }
        }

        if (enable_optimistic_read_) {
            if constexpr (kIsF14) {
                if constexpr (EmbeddedChain) {
                    for (int _retry = 0; _retry < kOptimisticReadMaxRetries; ++_retry) { if (_retry) LRU_SPIN_PAUSE();
                        auto sl1 = seqlock_.load(std::memory_order_acquire);
                        if (sl1 & 1u) continue; // rehash in progress, retry
                        size_type idx = h & bucket_mask_.load(std::memory_order_acquire);
                        auto v1 = chunks_[idx].version.load(std::memory_order_acquire);
                        bool found = find_f14_embedded(key, idx) != nullptr;
                        auto v2 = chunks_[idx].version.load(std::memory_order_acquire);
                        auto sl2 = seqlock_.load(std::memory_order_acquire);
                        if (sl1 == sl2 && (v1 == v2) && (v1 & 1u) == 0) {
                            // T-P2-6: during incremental rehash, re-verify the
                            // target bucket wasn't migrated between v1 and v2.
                            // No-op when rehash_progress_guard is false.
                            if (rehash_progress_guard &&
                                rehash_progress_.load(std::memory_order_acquire) >= idx) {
                                break; // bucket reached by migration — fall back to find_and_pin
                            }
                            if (!found) return nullptr; // Confirmed miss — skip lock
                            break; // Key exists — fall through to shared lock path for safe pin
                        }
                    }
                } else {
                    // Non-EmbeddedChain F14: optimistic read to detect misses
                    for (int _retry = 0; _retry < kOptimisticReadMaxRetries; ++_retry) { if (_retry) LRU_SPIN_PAUSE();
                        auto sl1 = seqlock_.load(std::memory_order_acquire);
                        if (sl1 & 1u) continue;
                        size_type idx = h & bucket_mask_.load(std::memory_order_acquire);
                        auto v1 = chunks_[idx].version.load(std::memory_order_acquire);
                        bool found = find_f14_node(key, idx) != nullptr;
                        auto v2 = chunks_[idx].version.load(std::memory_order_acquire);
                        auto sl2 = seqlock_.load(std::memory_order_acquire);
                        if (sl1 == sl2 && (v1 == v2) && (v1 & 1u) == 0) {
                            if (rehash_progress_guard &&
                                rehash_progress_.load(std::memory_order_acquire) >= idx) {
                                break;
                            }
                            if (!found) return nullptr;
                            break;
                        }
                    }
                }
            } else {
                if constexpr (EmbeddedChain) {
                    for (int _retry = 0; _retry < kOptimisticReadMaxRetries; ++_retry) { if (_retry) LRU_SPIN_PAUSE();
                        auto sl1 = seqlock_.load(std::memory_order_acquire);
                        if (sl1 & 1u) continue;
                        size_type idx = h & bucket_mask_.load(std::memory_order_acquire);
                        auto v1 = buckets_[idx].version.load(std::memory_order_acquire);
                        bool found = find_node_embedded(key, idx) != nullptr;
                        auto v2 = buckets_[idx].version.load(std::memory_order_acquire);
                        auto sl2 = seqlock_.load(std::memory_order_acquire);
                        if (sl1 == sl2 && (v1 == v2) && (v1 & 1u) == 0) {
                            // T-P2-6: progress re-check for the chain-mode
                            // incremental-rehash optimistic read path.
                            if (rehash_progress_guard &&
                                rehash_progress_.load(std::memory_order_acquire) >= idx) {
                                break; // bucket migrated — use dual-array find_and_pin
                            }
                            if (!found) return nullptr;
                            break;
                        }
                    }
                } else {
                    // Non-EmbeddedChain: optimistic read to detect misses
                    for (int _retry = 0; _retry < kOptimisticReadMaxRetries; ++_retry) { if (_retry) LRU_SPIN_PAUSE();
                        auto sl1 = seqlock_.load(std::memory_order_acquire);
                        if (sl1 & 1u) continue;
                        size_type idx = h & bucket_mask_.load(std::memory_order_acquire);
                        auto v1 = buckets_[idx].version.load(std::memory_order_acquire);
                        bool found = find_node(key, idx) != nullptr;
                        auto v2 = buckets_[idx].version.load(std::memory_order_acquire);
                        auto sl2 = seqlock_.load(std::memory_order_acquire);
                        if (sl1 == sl2 && (v1 == v2) && (v1 & 1u) == 0) {
                            if (rehash_progress_guard &&
                                rehash_progress_.load(std::memory_order_acquire) >= idx) {
                                break;
                            }
                            if (!found) return nullptr;
                            break;
                        }
                    }
                }
            }
        }

        // Fallback: shared lock path for safe pinning
        return find_and_pin(key, std::forward<PinFn>(pin_fn));
    }

    /// Const overload of find_and_pin_optimistic.
    /// Supports both EmbeddedChain and node-based chain modes.
    template <typename PinFn, typename UnpinFn>
    Value find_and_pin_optimistic(const Key& key, PinFn&& pin_fn, UnpinFn&& /*unpin_fn*/) const {
        const size_type h = hash_(key);

        // T-P2-6: mirror the non-const overload — during incremental rehash,
        // allow optimistic reads for chain-mode buckets NOT in the current
        // migration window (old_idx > rehash_progress_). See the non-const
        // overload for the full safety argument (version check + progress
        // re-check). F14 and already-migrated buckets fall back to find_and_pin.
        bool rehash_progress_guard = false;
        if (incremental_rehash_ && rehash_in_progress_.load(std::memory_order_acquire)) {
            if constexpr (kIsF14) {
                return find_and_pin(key, std::forward<PinFn>(pin_fn));
            } else {
                const size_type old_idx = h & bucket_mask_.load(std::memory_order_acquire);
                const size_type progress = rehash_progress_.load(std::memory_order_acquire);
                if (old_idx <= progress) {
                    return find_and_pin(key, std::forward<PinFn>(pin_fn));
                }
                rehash_progress_guard = true;
            }
        }

        if (enable_optimistic_read_) {
            if constexpr (kIsF14) {
                if constexpr (EmbeddedChain) {
                    for (int _retry = 0; _retry < kOptimisticReadMaxRetries; ++_retry) { if (_retry) LRU_SPIN_PAUSE();
                        auto sl1 = seqlock_.load(std::memory_order_acquire);
                        if (sl1 & 1u) continue;
                        size_type idx = h & bucket_mask_.load(std::memory_order_acquire);
                        auto v1 = chunks_[idx].version.load(std::memory_order_acquire);
                        bool found = find_f14_embedded(key, idx) != nullptr;
                        auto v2 = chunks_[idx].version.load(std::memory_order_acquire);
                        auto sl2 = seqlock_.load(std::memory_order_acquire);
                        if (sl1 == sl2 && (v1 == v2) && (v1 & 1u) == 0) {
                            // T-P2-6: progress re-check (no-op outside rehash path).
                            if (rehash_progress_guard &&
                                rehash_progress_.load(std::memory_order_acquire) >= idx) {
                                break;
                            }
                            if (!found) return nullptr;
                            break;
                        }
                    }
                } else {
                    // Non-EmbeddedChain F14: optimistic read to detect misses
                    for (int _retry = 0; _retry < kOptimisticReadMaxRetries; ++_retry) { if (_retry) LRU_SPIN_PAUSE();
                        auto sl1 = seqlock_.load(std::memory_order_acquire);
                        if (sl1 & 1u) continue;
                        size_type idx = h & bucket_mask_.load(std::memory_order_acquire);
                        auto v1 = chunks_[idx].version.load(std::memory_order_acquire);
                        bool found = find_f14_node(key, idx) != nullptr;
                        auto v2 = chunks_[idx].version.load(std::memory_order_acquire);
                        auto sl2 = seqlock_.load(std::memory_order_acquire);
                        if (sl1 == sl2 && (v1 == v2) && (v1 & 1u) == 0) {
                            if (rehash_progress_guard &&
                                rehash_progress_.load(std::memory_order_acquire) >= idx) {
                                break;
                            }
                            if (!found) return nullptr;
                            break;
                        }
                    }
                }
            } else {
                if constexpr (EmbeddedChain) {
                    for (int _retry = 0; _retry < kOptimisticReadMaxRetries; ++_retry) { if (_retry) LRU_SPIN_PAUSE();
                        auto sl1 = seqlock_.load(std::memory_order_acquire);
                        if (sl1 & 1u) continue;
                        size_type idx = h & bucket_mask_.load(std::memory_order_acquire);
                        auto v1 = buckets_[idx].version.load(std::memory_order_acquire);
                        bool found = find_node_embedded(key, idx) != nullptr;
                        auto v2 = buckets_[idx].version.load(std::memory_order_acquire);
                        auto sl2 = seqlock_.load(std::memory_order_acquire);
                        if (sl1 == sl2 && (v1 == v2) && (v1 & 1u) == 0) {
                            // T-P2-6: progress re-check for chain-mode rehash path.
                            if (rehash_progress_guard &&
                                rehash_progress_.load(std::memory_order_acquire) >= idx) {
                                break;
                            }
                            if (!found) return nullptr;
                            break;
                        }
                    }
                } else {
                    // Non-EmbeddedChain: optimistic read to detect misses
                    for (int _retry = 0; _retry < kOptimisticReadMaxRetries; ++_retry) { if (_retry) LRU_SPIN_PAUSE();
                        auto sl1 = seqlock_.load(std::memory_order_acquire);
                        if (sl1 & 1u) continue;
                        size_type idx = h & bucket_mask_.load(std::memory_order_acquire);
                        auto v1 = buckets_[idx].version.load(std::memory_order_acquire);
                        bool found = find_node(key, idx) != nullptr;
                        auto v2 = buckets_[idx].version.load(std::memory_order_acquire);
                        auto sl2 = seqlock_.load(std::memory_order_acquire);
                        if (sl1 == sl2 && (v1 == v2) && (v1 & 1u) == 0) {
                            if (rehash_progress_guard &&
                                rehash_progress_.load(std::memory_order_acquire) >= idx) {
                                break;
                            }
                            if (!found) return nullptr;
                            break;
                        }
                    }
                }
            }
        }

        return find_and_pin(key, std::forward<PinFn>(pin_fn));
    }

    // ========================================================================
    // Lock-free find-and-pin: optimistic read + lock-free incRef
    // ========================================================================

    /// Lock-free find-and-pin: attempts to pin an item WITHOUT acquiring any
    /// bucket lock at all. This is the fastest possible read path for hits:
    ///
    ///   1. Optimistic read to find the item and get its pointer
    ///   2. Attempt incRef() on the item outside any lock
    ///   3. If incRef succeeds (returns kIncOk), the item is pinned — return it
    ///   4. If incRef fails (item being evicted/moved), fall back to
    ///      find_and_pin() (shared lock path)
    ///
    /// Safety argument:
    ///   - If incRef succeeds, we hold a valid reference that prevents eviction
    ///   - If incRef fails, the item is being removed, so we fall back safely
    ///   - The version check ensures the item pointer we got is valid at the
    ///     time of the optimistic read
    ///   - Between the optimistic read and incRef, the item could be removed,
    ///     but incRef would fail in that case (kExclusive flag set by
    ///     markForEviction)
    ///
    /// PinFn is invoked with the found Value to attempt the pin (e.g. incRef).
    /// It should return true if the pin succeeded, false otherwise.
    template <typename PinFn>
    Value find_and_pin_lockfree(const Key& key, PinFn&& pin_fn) {
        return find_and_pin_lockfree_with_hash(key, hash_(key), std::forward<PinFn>(pin_fn));
    }

    /// T16.4: find_and_pin_lockfree with a pre-computed hash. The hash
    /// MUST be the result of `hash_(key)`. Used by bulk_get to avoid
    /// re-hashing each key for both shard dispatch and hash-table lookup.
    template <typename PinFn>
    Value find_and_pin_lockfree_with_hash(const Key& key, size_type h, PinFn&& pin_fn) {
        // T2.1: Acquire epoch_guard at entry when EBR is enabled. This
        // protects ALL nodes in the hash table from reclamation during
        // the traversal — concurrent retire() calls defer deletion until
        // all active critical sections exit. In hazptr mode (ebr_domain_
        // == nullptr), the guard is a no-op; per-pointer protection is
        // provided by hazptr_holder inside the traversal.
        std::optional<detail::epoch_domain::epoch_guard> ebr_guard;
        if (ebr_domain_) {
            ebr_guard.emplace(*ebr_domain_);
        }

        if (!enable_optimistic_read_) {
            return find_and_pin_with_hash(key, h, std::forward<PinFn>(pin_fn));
        }

        // During incremental rehash, lockfree/optimistic paths are unsafe for
        // chain mode — items may be split across two arrays.
        if (incremental_rehash_ && rehash_in_progress_.load(std::memory_order_acquire)) {
            rehash_lockfree_fallback_count_.fetch_add(1, std::memory_order_relaxed);
            return find_and_pin_with_hash(key, h, std::forward<PinFn>(pin_fn));
        }

        if constexpr (kIsF14) {
            if constexpr (EmbeddedChain) {
                // T-P2-8: Read path prefers EBR. When an EBR domain is set,
                // the epoch_guard at entry protects all nodes from reclamation,
                // allowing us to skip per-pointer hazptr_holder protection.
                // The F14 chunk version checks remain for slot validation
                // (detecting concurrent slot mutations), and the seqlock
                // re-validation catches chain modifications.
                const bool use_ebr = (ebr_domain_ != nullptr);
                if (use_ebr) {
                    // EBR fast path: raw pointer access, no hazptr overhead.
                    for (int _retry = 0; _retry < kOptimisticReadMaxRetries; ++_retry) { if (_retry) LRU_SPIN_PAUSE();
                        auto sl1 = seqlock_.load(std::memory_order_acquire);
                        if (sl1 & 1u) continue;
                        size_type idx = h & bucket_mask_.load(std::memory_order_acquire);
                        const size_type h2 = h;
                        uint8_t tag = f14_detail::f14_tag(h2);
                        auto& chunk = chunks_[idx];
                        bool need_retry = false;

                        // ---- Inline-slot scan (version-stamped, no hazptr) ----
                        auto v1 = chunk.version.load(std::memory_order_acquire);
                        if ((v1 & 1u) == 0) {
                            uint16_t match_mask = f14_detail::f14_match_tags(chunk.tags, tag)
                                                & chunk.load_occupied_mask_acquire();
                            while (match_mask) {
                                int slot = f14_detail::ctz16(match_mask);
                                match_mask &= static_cast<uint16_t>(match_mask - 1);
                                Value node = static_cast<Value>(
                                    std::atomic_ref<void*>(chunk.slots[slot]).load(std::memory_order_acquire));
                                if (!node) continue;
                                // Re-read version (no hazptr needed — EBR
                                // protects all nodes from reclamation).
                                auto v2 = chunk.version.load(std::memory_order_acquire);
                                if (v1 != v2 || (v2 & 1u) != 0) {
                                    need_retry = true;
                                    break;
                                }
                                // Slot still references `node` (slot mutations bump version).
                                Value node2 = static_cast<Value>(
                                    std::atomic_ref<void*>(chunk.slots[slot]).load(std::memory_order_acquire));
                                if (node2 == node && equal_(node->key, key)) {
                                    if (pin_fn(node)) {
                                        return node;
                                    }
                                    // Pin failed — fall to locked path.
                                    return find_and_pin_with_hash(key, h, std::forward<PinFn>(pin_fn));
                                }
                            }
                        }

                        if (!need_retry) {
                            // ---- Overflow chain traversal (raw pointer, no hazptr) ----
                            Value curr = chunk.embed_head.load(std::memory_order_acquire);
                            while (curr) {
                                if (equal_(curr->key, key)) {
                                    if (pin_fn(curr)) {
                                        return curr;
                                    }
                                    return find_and_pin_with_hash(key, h, std::forward<PinFn>(pin_fn));
                                }
                                curr = static_cast<Value>(curr->hash_chain_next());
                            }

                            // Confirmed miss — re-validate seqlock.
                            auto sl2 = seqlock_.load(std::memory_order_acquire);
                            if (sl1 == sl2 && (sl1 & 1u) == 0) {
                                return nullptr;
                            }
                        }
                    }
                    // Exhausted retries — fall to locked path.
                    return find_and_pin_with_hash(key, h, std::forward<PinFn>(pin_fn));
                } else {
                    // Hazptr path: version-stamped hazptr pin.
                    // The node is hazptr-protected *before* dereferencing its key,
                    // and the chunk version is re-read *after* publishing the hazptr.
                    // If the version is unchanged and even, the slot still owns
                    // the node — any concurrent retire will observe our hazptr and
                    // defer reclamation, making the subsequent pin_fn (incRef) safe.
                    hazptr_holder hazslot;
                    for (int _retry = 0; _retry < kOptimisticReadMaxRetries; ++_retry) { if (_retry) LRU_SPIN_PAUSE();
                        auto sl1 = seqlock_.load(std::memory_order_acquire);
                        if (sl1 & 1u) continue;
                        size_type idx = h & bucket_mask_.load(std::memory_order_acquire);
                        const size_type h2 = h; // already hashed above
                        uint8_t tag = f14_detail::f14_tag(h2);
                        auto& chunk = chunks_[idx];

                        // ---- Inline-slot scan with version-stamped hazptr ----
                        auto v1 = chunk.version.load(std::memory_order_acquire);
                        if ((v1 & 1u) == 0) {
                            uint16_t match_mask = f14_detail::f14_match_tags(chunk.tags, tag)
                                                & chunk.load_occupied_mask_acquire();
                            bool found_in_inline = false;
                            Value found_node = nullptr;
                            while (match_mask) {
                                int slot = f14_detail::ctz16(match_mask);
                                match_mask &= static_cast<uint16_t>(match_mask - 1);
                                Value node = static_cast<Value>(
                                    std::atomic_ref<void*>(chunk.slots[slot]).load(std::memory_order_acquire));
                                if (!node) continue;
                                hazslot.protect(node);
                                // Re-read version after publishing hazptr.
                                auto v2 = chunk.version.load(std::memory_order_acquire);
                                if (v1 != v2 || (v2 & 1u) != 0) {
                                    goto retry_f14_embedded;  // chunk changed — restart
                                }
                                // Slot still references `node` (slot mutations bump version).
                                Value node2 = static_cast<Value>(
                                    std::atomic_ref<void*>(chunk.slots[slot]).load(std::memory_order_acquire));
                                if (node2 == node && equal_(node->key, key)) {
                                    // Found — attempt pin *before* releasing hazptr.
                                    if (pin_fn(node)) {
                                        return node;
                                    }
                                    // Pin failed (item being evicted) — fall to locked path.
                                    goto fallback_f14_embedded;
                                }
                                // Mismatch — clear slot hazptr and continue scanning.
                                hazslot.clear();
                            }
                        }

                        // ---- Overflow chain traversal (hazptr-protected) ----
                        {
                            Value curr = chunk.embed_head.load(std::memory_order_acquire);
                            hazslot.protect(curr);
                            while (curr) {
                                // Verify curr still reachable from head (handles
                                // concurrent unlink during traversal).
                                Value verify = chunk.embed_head.load(std::memory_order_acquire);
                                if (verify != curr && !is_reachable(verify, curr)) {
                                    hazslot.protect(verify);
                                    curr = verify;
                                    continue;
                                }
                                if (equal_(curr->key, key)) {
                                    // Found — attempt pin *before* releasing hazptr.
                                    if (pin_fn(curr)) {
                                        return curr;
                                    }
                                    goto fallback_f14_embedded;
                                }
                                Value next = static_cast<Value>(curr->hash_chain_next());
                                if (!next) break;
                                // Move hazptr to next, then re-verify reachability
                                // of `next` from the current head.
                                hazptr_holder haztmp;
                                haztmp.protect(next);
                                Value v2 = chunk.embed_head.load(std::memory_order_acquire);
                                if (v2 != curr && !is_reachable(v2, curr)) {
                                    // curr was unlinked — restart from head
                                    hazslot.protect(v2);
                                    curr = v2;
                                    continue;
                                }
                                hazslot.swap(haztmp);
                                curr = next;
                            }
                            hazslot.clear();
                        }

                        // Confirmed miss — re-validate seqlock.
                        {
                            auto sl2 = seqlock_.load(std::memory_order_acquire);
                            if (sl1 == sl2 && (sl1 & 1u) == 0) {
                                return nullptr;
                            }
                        }
                    retry_f14_embedded:
                        continue;
                    }
                    // Unreachable: while(true) only exits via return or goto.
                    // Label target for pin-failure fallthrough — return the locked path.
                fallback_f14_embedded:
                    return find_and_pin_with_hash(key, h, std::forward<PinFn>(pin_fn));
                }
            } else {
                // F14 non-EmbeddedChain: nodes are allocated/deallocated directly
                // (not via hazptr reclamation), so lock-free pin after optimistic
                // read would be unsafe — node could be deleted between the version
                // check and the pin_fn call. Fall back to the shared-lock path.
                return find_and_pin_with_hash(key, h, std::forward<PinFn>(pin_fn));
            }
        } else {
            if constexpr (EmbeddedChain) {
                // T-P2-8: Read path prefers EBR. When an EBR domain is set,
                // the epoch_guard acquired at function entry protects ALL nodes
                // from reclamation during the traversal. This lets us skip
                // per-pointer hazptr_holder protection (slot acquire/release +
                // atomic stores on every node visit) and use raw pointer access.
                // The seqlock re-validation at the end ensures we retry if the
                // chain was modified concurrently. Hazptr is still used for
                // long-held scenarios (iterators, read_handle) via find_and_pin.
                const bool use_ebr = (ebr_domain_ != nullptr);
                if (use_ebr) {
                    // EBR fast path: raw pointer traversal, no hazptr overhead.
                    // Nodes cannot be freed during the critical section, so we
                    // can safely dereference curr->key and curr->hash_chain_next()
                    // without per-pointer hazard protection. Reachability checks
                    // are also unnecessary — if the chain changes, the seqlock
                    // bumps and we retry.
                    //
                    // C-3 fix: same TOCTOU on rehash_in_progress_ as the hazptr
                    // path. The seqlock re-validation at the end only catches
                    // the FINAL rehash_finish() (which bumps seqlock_); it does
                    // NOT catch rehash_step() invocations that mutate
                    // hash_chain_next() pointers mid-walk. A concurrent
                    // migration can therefore introduce a cycle into the chain
                    // we are traversing, hanging the while(curr) loop. Bail to
                    // the locked path if a rehash starts, and bound the walk.
                    for (int _retry = 0; _retry < kOptimisticReadMaxRetries; ++_retry) { if (_retry) LRU_SPIN_PAUSE();
                        if (incremental_rehash_ && rehash_in_progress_.load(std::memory_order_acquire)) {
                            return find_and_pin_with_hash(key, h, std::forward<PinFn>(pin_fn));
                        }
                        auto sl1 = seqlock_.load(std::memory_order_acquire);
                        if (sl1 & 1u) continue;
                        size_type idx = h & bucket_mask_.load(std::memory_order_acquire);
                        Value curr = buckets_[idx].embed_head.load(std::memory_order_acquire);
                        Value found_node = nullptr;
                        int walk_steps = 0;
                        while (curr) {
                            if (incremental_rehash_ && rehash_in_progress_.load(std::memory_order_acquire)) {
                                return find_and_pin_with_hash(key, h, std::forward<PinFn>(pin_fn));
                            }
                            if (++walk_steps > kMaxWalkSteps) {
                                return find_and_pin_with_hash(key, h, std::forward<PinFn>(pin_fn));
                            }
                            if (equal_(curr->key, key)) {
                                found_node = curr;
                                break;
                            }
                            curr = static_cast<Value>(curr->hash_chain_next());
                        }
                        auto sl2 = seqlock_.load(std::memory_order_acquire);
                        if (sl1 == sl2 && (sl1 & 1u) == 0) {
                            if (!found_node) return nullptr;
                            if (pin_fn(found_node)) {
                                return found_node;
                            }
                            break;
                        }
                    }
                } else {
                    // Hazptr path: per-pointer protection via hazptr_holder.
                    //
                    // C-3 fix: TOCTOU on rehash_in_progress_. The entry-level
                    // check at the top of find_and_pin_lockfree_with_hash only
                    // verifies that no rehash is in progress AT ENTRY. A
                    // concurrent rehash_step() (e.g. driven by the background
                    // rehash balancer) can START migrating the very bucket we
                    // are walking. rehash_step() rewrites each node's
                    // hash_chain_next() pointer to splice it into the new
                    // bucket's chain, so the in-place chain we are traversing
                    // can briefly form a cycle (e.g. node A's next gets pointed
                    // at B, and B's next — previously A — was already updated
                    // to point at the old new-bucket head). Without a bound or
                    // a re-check, the while(curr) loop below never terminates
                    // and the reader hangs — the production_cache try_get
                    // deadlock observed under concurrent reads.
                    //
                    // Fix: (a) re-check rehash_in_progress_ inside the loop
                    //          and bail to the shared-lock path if a rehash
                    //          started during our walk;
                    //      (b) bound the walk at 2x the longest expected chain
                    //          (kMaxWalkSteps, see is_reachable's 64-step
                    //          bound) so even a degenerate cyclic chain cannot
                    //          spin forever — we fall back to find_and_pin_with_hash
                    //          which uses proper bucket locking and is safe
                    //          during rehash.
                    hazptr_holder hazslot;
                    for (int _retry = 0; _retry < kOptimisticReadMaxRetries; ++_retry) { if (_retry) LRU_SPIN_PAUSE();
                        // C-3: re-check at retry boundary — a rehash that
                        // started since our entry check must route to the
                        // shared-lock path; the hazptr walk is no longer safe.
                        if (incremental_rehash_ && rehash_in_progress_.load(std::memory_order_acquire)) {
                            return find_and_pin_with_hash(key, h, std::forward<PinFn>(pin_fn));
                        }
                        auto sl1 = seqlock_.load(std::memory_order_acquire);
                        if (sl1 & 1u) continue;
                        size_type idx = h & bucket_mask_.load(std::memory_order_acquire);
                        Value curr = buckets_[idx].embed_head.load(std::memory_order_acquire);
                        hazslot.protect(curr);
                        bool found = false;
                        Value found_node = nullptr;
                        int walk_steps = 0;
                        while (curr) {
                            // C-3 (a): bail if a rehash started mid-walk.
                            if (incremental_rehash_ && rehash_in_progress_.load(std::memory_order_acquire)) {
                                return find_and_pin_with_hash(key, h, std::forward<PinFn>(pin_fn));
                            }
                            // C-3 (b): bound the walk. A chain longer than
                            // kMaxWalkSteps under EmbeddedChain+chain mode is
                            // either a degenerate hash collision (rare) or, more
                            // likely, a cycle introduced by concurrent migration.
                            // Either way, fall back to the locked path.
                            if (++walk_steps > kMaxWalkSteps) {
                                return find_and_pin_with_hash(key, h, std::forward<PinFn>(pin_fn));
                            }
                            Value verify = buckets_[idx].embed_head.load(std::memory_order_acquire);
                            if (verify != curr && !is_reachable(verify, curr)) {
                                hazslot.protect(verify);
                                curr = verify;
                                continue;
                            }
                            if (equal_(curr->key, key)) {
                                found = true;
                                found_node = curr;
                                break;
                            }
                            Value next = static_cast<Value>(curr->hash_chain_next());
                            if (!next) break;
                            hazptr_holder haztmp;
                            haztmp.protect(next);
                            Value v2 = buckets_[idx].embed_head.load(std::memory_order_acquire);
                            if (v2 != curr && !is_reachable(v2, curr)) {
                                hazslot.protect(v2);
                                curr = v2;
                                continue;
                            }
                            hazslot.swap(haztmp);
                            curr = next;
                        }
                        auto sl2 = seqlock_.load(std::memory_order_acquire);
                        if (sl1 == sl2 && (sl1 & 1u) == 0) {
                            if (!found_node) return nullptr;
                            if (pin_fn(found_node)) {
                                return found_node;
                            }
                            break;
                        }
                    }
                }
            } else {
                // Non-F14 non-EmbeddedChain: nodes are allocated/deallocated
                // directly (not via hazptr reclamation), so lock-free pin after
                // optimistic read would be unsafe. Fall back to shared-lock path.
                return find_and_pin_with_hash(key, h, std::forward<PinFn>(pin_fn));
            }
        }

        // Fallback: shared lock path for safe pinning
        return find_and_pin_with_hash(key, h, std::forward<PinFn>(pin_fn));
    }

    /// Const overload of find_and_pin_lockfree.
    template <typename PinFn>
    Value find_and_pin_lockfree(const Key& key, PinFn&& pin_fn) const {
        const size_type h = hash_(key);

        // T-P2-8: Acquire epoch_guard at entry when EBR is enabled, mirroring
        // the non-const find_and_pin_lockfree_with_hash. This protects ALL
        // nodes from reclamation during the traversal, allowing the EBR fast
        // paths below to skip per-pointer hazptr_holder protection.
        std::optional<detail::epoch_domain::epoch_guard> ebr_guard;
        if (ebr_domain_) {
            ebr_guard.emplace(*ebr_domain_);
        }

        if (!enable_optimistic_read_) {
            return find_and_pin(key, std::forward<PinFn>(pin_fn));
        }

        if (incremental_rehash_ && rehash_in_progress_.load(std::memory_order_acquire)) {
            rehash_lockfree_fallback_count_.fetch_add(1, std::memory_order_relaxed);
            return find_and_pin(key, std::forward<PinFn>(pin_fn));
        }

        if constexpr (kIsF14) {
            if constexpr (EmbeddedChain) {
                // T-P2-8: Read path prefers EBR. When EBR is active, the
                // epoch_guard at entry protects all nodes, allowing raw pointer
                // access without per-pointer hazptr_holder protection.
                const bool use_ebr = (ebr_domain_ != nullptr);
                if (use_ebr) {
                    // EBR fast path: raw pointer access, no hazptr overhead.
                    for (int _retry = 0; _retry < kOptimisticReadMaxRetries; ++_retry) { if (_retry) LRU_SPIN_PAUSE();
                        auto sl1 = seqlock_.load(std::memory_order_acquire);
                        if (sl1 & 1u) continue;
                        size_type idx = h & bucket_mask_.load(std::memory_order_acquire);
                        const size_type h2 = h;
                        uint8_t tag = f14_detail::f14_tag(h2);
                        auto& chunk = chunks_[idx];
                        bool need_retry = false;

                        auto v1 = chunk.version.load(std::memory_order_acquire);
                        if ((v1 & 1u) == 0) {
                            uint16_t match_mask = f14_detail::f14_match_tags(chunk.tags, tag)
                                                & chunk.load_occupied_mask_acquire();
                            while (match_mask) {
                                int slot = f14_detail::ctz16(match_mask);
                                match_mask &= static_cast<uint16_t>(match_mask - 1);
                                void* raw = const_cast<void*>(chunk.slots[slot]);
                                Value node = static_cast<Value>(
                                    std::atomic_ref<void*>(raw).load(std::memory_order_acquire));
                                if (!node) continue;
                                // Re-read version (no hazptr needed — EBR
                                // protects all nodes from reclamation).
                                auto v2 = chunk.version.load(std::memory_order_acquire);
                                if (v1 != v2 || (v2 & 1u) != 0) {
                                    need_retry = true;
                                    break;
                                }
                                void* raw2 = const_cast<void*>(chunk.slots[slot]);
                                Value node2 = static_cast<Value>(
                                    std::atomic_ref<void*>(raw2).load(std::memory_order_acquire));
                                if (node2 == node && equal_(node->key, key)) {
                                    if (pin_fn(node)) {
                                        return node;
                                    }
                                    return find_and_pin(key, std::forward<PinFn>(pin_fn));
                                }
                            }
                        }

                        if (!need_retry) {
                            // Overflow chain traversal (raw pointer, no hazptr)
                            Value curr = chunk.embed_head.load(std::memory_order_acquire);
                            while (curr) {
                                if (equal_(curr->key, key)) {
                                    if (pin_fn(curr)) {
                                        return curr;
                                    }
                                    return find_and_pin(key, std::forward<PinFn>(pin_fn));
                                }
                                curr = static_cast<Value>(curr->hash_chain_next());
                            }

                            auto sl2 = seqlock_.load(std::memory_order_acquire);
                            if (sl1 == sl2 && (sl1 & 1u) == 0) {
                                return nullptr;
                            }
                        }
                    }
                    return find_and_pin(key, std::forward<PinFn>(pin_fn));
                } else {
                    // Hazptr path: hazptr-protected traversal for safety.
                    hazptr_holder hazslot;
                    for (int _retry = 0; _retry < kOptimisticReadMaxRetries; ++_retry) { if (_retry) LRU_SPIN_PAUSE();
                        auto sl1 = seqlock_.load(std::memory_order_acquire);
                        if (sl1 & 1u) continue;
                        size_type idx = h & bucket_mask_.load(std::memory_order_acquire);
                        const size_type h2 = h;
                        uint8_t tag = f14_detail::f14_tag(h2);
                        auto& chunk = chunks_[idx];
                        bool retry = false;

                        auto v1 = chunk.version.load(std::memory_order_acquire);
                        if ((v1 & 1u) == 0) {
                            uint16_t match_mask = f14_detail::f14_match_tags(chunk.tags, tag)
                                                & chunk.load_occupied_mask_acquire();
                            while (match_mask) {
                                int slot = f14_detail::ctz16(match_mask);
                                match_mask &= static_cast<uint16_t>(match_mask - 1);
                                void* raw = const_cast<void*>(chunk.slots[slot]);
                                Value node = static_cast<Value>(
                                    std::atomic_ref<void*>(raw).load(std::memory_order_acquire));
                                if (!node) continue;
                                hazslot.protect(node);
                                auto v2 = chunk.version.load(std::memory_order_acquire);
                                if (v1 != v2 || (v2 & 1u) != 0) {
                                    retry = true;
                                    break;
                                }
                                void* raw2 = const_cast<void*>(chunk.slots[slot]);
                                Value node2 = static_cast<Value>(
                                    std::atomic_ref<void*>(raw2).load(std::memory_order_acquire));
                                if (node2 == node && equal_(node->key, key)) {
                                    if (pin_fn(node)) {
                                        return node;
                                    }
                                    goto fallback_f14_embedded_const;
                                }
                                hazslot.clear();
                            }
                        }

                        if (!retry) {
                            // Overflow chain traversal (hazptr-protected)
                            Value curr = chunk.embed_head.load(std::memory_order_acquire);
                            hazslot.protect(curr);
                            while (curr) {
                                Value verify = chunk.embed_head.load(std::memory_order_acquire);
                                if (verify != curr && !is_reachable(verify, curr)) {
                                    hazslot.protect(verify);
                                    curr = verify;
                                    continue;
                                }
                                if (equal_(curr->key, key)) {
                                    if (pin_fn(curr)) {
                                        return curr;
                                    }
                                    goto fallback_f14_embedded_const;
                                }
                                Value next = static_cast<Value>(curr->hash_chain_next());
                                if (!next) break;
                                hazptr_holder haztmp;
                                haztmp.protect(next);
                                Value v2 = chunk.embed_head.load(std::memory_order_acquire);
                                if (v2 != curr && !is_reachable(v2, curr)) {
                                    hazslot.protect(v2);
                                    curr = v2;
                                    continue;
                                }
                                hazslot.swap(haztmp);
                                curr = next;
                            }
                            hazslot.clear();
                        }

                        auto sl2 = seqlock_.load(std::memory_order_acquire);
                        if (!retry && sl1 == sl2 && (sl1 & 1u) == 0) {
                            return nullptr;
                        }
                    }
                fallback_f14_embedded_const:
                    return find_and_pin(key, std::forward<PinFn>(pin_fn));
                }
            } else {
                // F14 non-EmbeddedChain: nodes are directly allocated/deallocated,
                // lock-free pin after optimistic read is unsafe. Fall back to shared lock.
                return find_and_pin(key, std::forward<PinFn>(pin_fn));
            }
        } else {
            if constexpr (EmbeddedChain) {
                // T-P2-8: Read path prefers EBR. When EBR is active, the
                // epoch_guard at entry protects all nodes, allowing raw pointer
                // access without per-pointer hazptr_holder protection.
                const bool use_ebr = (ebr_domain_ != nullptr);
                if (use_ebr) {
                    // EBR fast path: raw pointer traversal, no hazptr overhead.
                    // C-3 fix: see non-const overload for rationale.
                    for (int _retry = 0; _retry < kOptimisticReadMaxRetries; ++_retry) { if (_retry) LRU_SPIN_PAUSE();
                        if (incremental_rehash_ && rehash_in_progress_.load(std::memory_order_acquire)) {
                            return find_and_pin(key, std::forward<PinFn>(pin_fn));
                        }
                        auto sl1 = seqlock_.load(std::memory_order_acquire);
                        if (sl1 & 1u) continue;
                        size_type idx = h & bucket_mask_.load(std::memory_order_acquire);
                        Value curr = buckets_[idx].embed_head.load(std::memory_order_acquire);
                        Value found_node = nullptr;
                        int walk_steps = 0;
                        while (curr) {
                            if (incremental_rehash_ && rehash_in_progress_.load(std::memory_order_acquire)) {
                                return find_and_pin(key, std::forward<PinFn>(pin_fn));
                            }
                            if (++walk_steps > kMaxWalkSteps) {
                                return find_and_pin(key, std::forward<PinFn>(pin_fn));
                            }
                            if (equal_(curr->key, key)) {
                                found_node = curr;
                                break;
                            }
                            curr = static_cast<Value>(curr->hash_chain_next());
                        }
                        auto sl2 = seqlock_.load(std::memory_order_acquire);
                        if (sl1 == sl2 && (sl1 & 1u) == 0) {
                            if (!found_node) return nullptr;
                            if (pin_fn(found_node)) {
                                return found_node;
                            }
                            break;
                        }
                    }
                } else {
                    // Hazptr path: per-pointer protection via hazptr_holder.
                    // C-3 fix: see non-const overload for rationale.
                    hazptr_holder hazslot;
                    for (int _retry = 0; _retry < kOptimisticReadMaxRetries; ++_retry) { if (_retry) LRU_SPIN_PAUSE();
                        if (incremental_rehash_ && rehash_in_progress_.load(std::memory_order_acquire)) {
                            return find_and_pin(key, std::forward<PinFn>(pin_fn));
                        }
                        auto sl1 = seqlock_.load(std::memory_order_acquire);
                        if (sl1 & 1u) continue;
                        size_type idx = h & bucket_mask_.load(std::memory_order_acquire);
                        Value curr = buckets_[idx].embed_head.load(std::memory_order_acquire);
                        hazslot.protect(curr);
                        bool found = false;
                        Value found_node = nullptr;
                        int walk_steps = 0;
                        while (curr) {
                            if (incremental_rehash_ && rehash_in_progress_.load(std::memory_order_acquire)) {
                                return find_and_pin(key, std::forward<PinFn>(pin_fn));
                            }
                            if (++walk_steps > kMaxWalkSteps) {
                                return find_and_pin(key, std::forward<PinFn>(pin_fn));
                            }
                            Value verify = buckets_[idx].embed_head.load(std::memory_order_acquire);
                            if (verify != curr && !is_reachable(verify, curr)) {
                                hazslot.protect(verify);
                                curr = verify;
                                continue;
                            }
                            if (equal_(curr->key, key)) {
                                found = true;
                                found_node = curr;
                                break;
                            }
                            Value next = static_cast<Value>(curr->hash_chain_next());
                            if (!next) break;
                            hazptr_holder haztmp;
                            haztmp.protect(next);
                            Value v2 = buckets_[idx].embed_head.load(std::memory_order_acquire);
                            if (v2 != curr && !is_reachable(v2, curr)) {
                                hazslot.protect(v2);
                                curr = v2;
                                continue;
                            }
                            hazslot.swap(haztmp);
                            curr = next;
                        }
                        auto sl2 = seqlock_.load(std::memory_order_acquire);
                        if (sl1 == sl2 && (sl1 & 1u) == 0) {
                            if (!found_node) return nullptr;
                            if (pin_fn(found_node)) {
                                return found_node;
                            }
                            break;
                        }
                    }
                }
            } else {
                // Non-F14 non-EmbeddedChain: nodes are directly allocated/deallocated,
                // lock-free pin after optimistic read is unsafe. Fall back to shared lock.
                return find_and_pin(key, std::forward<PinFn>(pin_fn));
            }
        }

        return find_and_pin(key, std::forward<PinFn>(pin_fn));
    }

    /// Const overload of find_and_pin.
    template <typename PinFn>
    Value find_and_pin(const Key& key, PinFn&& pin_fn) const {
        const size_type h = hash_(key);
        if constexpr (kIsF14) {
            // F14 incremental rehash: dual-array aware find-and-pin
            if (incremental_rehash_ && rehash_in_progress_.load(std::memory_order_acquire)) {
                if constexpr (EmbeddedChain) {
                    Value node = find_f14_dual_array(key, h);
                    if (node && pin_fn(node)) return node;
                } else {
                    node_type* node = find_f14_dual_array(key, h);
                    if (node && pin_fn(&(node->value))) return &(node->value);
                }
                return nullptr;
            }
            size_type idx = h & bucket_mask_.load(std::memory_order_relaxed);
            auto guard = const_cast<concurrent_hash_table*>(this)->lock_bucket_shared(idx);
            if constexpr (EmbeddedChain) {
                Value node = find_f14_embedded(key, idx);
                if (node && pin_fn(node)) return node;
            } else {
                node_type* node = find_f14_node(key, idx);
                if (node && pin_fn(&(node->value))) return &(node->value);
            }
        } else {
            // Incremental rehash path for chain mode find_and_pin (const)
            if (incremental_rehash_ && rehash_in_progress_.load(std::memory_order_acquire)) {
                for (;;) {
                    size_type old_idx = h & bucket_mask_.load(std::memory_order_acquire);
                    size_type progress = rehash_progress_.load(std::memory_order_acquire);
                    if (old_idx < progress) {
                        size_type new_mask = rehash_new_bucket_count_.load(std::memory_order_acquire) - 1;
                        size_type new_idx = h & new_mask;
                        bucket_type* new_bkts = rehash_new_buckets_.load(std::memory_order_acquire);
                        shared_bucket_lock guard(new_bkts[new_idx].spin);
                        if (rehash_progress_.load(std::memory_order_acquire) >= progress) {
                            if constexpr (EmbeddedChain) {
                                Value node = find_node_embedded_in(key, new_bkts, new_idx);
                                if (node && pin_fn(node)) return node;
                            } else {
                                node_type* node = find_node_in(key, new_bkts, new_idx);
                                if (node && pin_fn(&(node->value))) return &(node->value);
                            }
                            return nullptr;
                        }
                    } else {
                        auto guard = const_cast<concurrent_hash_table*>(this)->lock_bucket_shared(old_idx);
                        if (rehash_progress_.load(std::memory_order_acquire) <= old_idx) {
                            if constexpr (EmbeddedChain) {
                                Value node = find_node_embedded(key, old_idx);
                                if (node && pin_fn(node)) return node;
                            } else {
                                node_type* node = find_node(key, old_idx);
                                if (node && pin_fn(&(node->value))) return &(node->value);
                            }
                            return nullptr;
                        }
                    }
                }
            }
            size_type idx = h & bucket_mask_.load(std::memory_order_relaxed);
            auto guard = const_cast<concurrent_hash_table*>(this)->lock_bucket_shared(idx);
            if constexpr (EmbeddedChain) {
                Value node = find_node_embedded(key, idx);
                if (node && pin_fn(node)) return node;
            } else {
                node_type* node = find_node(key, idx);
                if (node && pin_fn(&(node->value))) return &(node->value);
            }
        }
        return nullptr;
    }

    bool contains(const Key& key) const {
        const size_type h = hash_(key);
        if constexpr (kIsF14) {
            // F14 incremental rehash: dual-array aware lookup
            if (incremental_rehash_ && rehash_in_progress_.load(std::memory_order_acquire)) {
                return find_f14_dual_array(key, h) != nullptr;
            }
            if constexpr (EmbeddedChain) {
                if (enable_optimistic_read_) {
                    for (int _retry = 0; _retry < kOptimisticReadMaxRetries; ++_retry) { if (_retry) LRU_SPIN_PAUSE();
                        auto sl1 = seqlock_.load(std::memory_order_acquire);
                        if (sl1 & 1u) continue;
                        size_type idx = h & bucket_mask_.load(std::memory_order_acquire);
                        auto v1 = chunks_[idx].version.load(std::memory_order_acquire);
                        bool found = find_f14_embedded(key, idx) != nullptr;
                        auto v2 = chunks_[idx].version.load(std::memory_order_acquire);
                        auto sl2 = seqlock_.load(std::memory_order_acquire);
                        if (sl1 == sl2 && (v1 == v2) && (v1 & 1u) == 0) return found;
                    }
                }
                size_type idx = h & bucket_mask_.load(std::memory_order_relaxed);
                auto guard = const_cast<concurrent_hash_table*>(this)->lock_bucket_shared(idx);
                return find_f14_embedded(key, idx) != nullptr;
            } else {
                if (enable_optimistic_read_) {
                    for (int _retry = 0; _retry < kOptimisticReadMaxRetries; ++_retry) { if (_retry) LRU_SPIN_PAUSE();
                        auto sl1 = seqlock_.load(std::memory_order_acquire);
                        if (sl1 & 1u) continue;
                        size_type idx = h & bucket_mask_.load(std::memory_order_acquire);
                        auto v1 = chunks_[idx].version.load(std::memory_order_acquire);
                        bool found = find_f14_node(key, idx) != nullptr;
                        auto v2 = chunks_[idx].version.load(std::memory_order_acquire);
                        auto sl2 = seqlock_.load(std::memory_order_acquire);
                        if (sl1 == sl2 && (v1 == v2) && (v1 & 1u) == 0) return found;
                    }
                }
                size_type idx = h & bucket_mask_.load(std::memory_order_relaxed);
                auto guard = const_cast<concurrent_hash_table*>(this)->lock_bucket_shared(idx);
                return find_f14_node(key, idx) != nullptr;
            }
        } else {
            // Incremental rehash path for chain mode contains
            if (incremental_rehash_ && rehash_in_progress_.load(std::memory_order_acquire)) {
                for (;;) {
                    size_type old_idx = h & bucket_mask_.load(std::memory_order_acquire);
                    size_type progress = rehash_progress_.load(std::memory_order_acquire);
                    if (old_idx < progress) {
                        // Bucket has been migrated — look in new array
                        size_type new_mask = rehash_new_bucket_count_.load(std::memory_order_acquire) - 1;
                        size_type new_idx = h & new_mask;
                        bucket_type* new_bkts = rehash_new_buckets_.load(std::memory_order_acquire);
                        shared_bucket_lock guard(new_bkts[new_idx].spin);
                        if constexpr (EmbeddedChain) {
                            return find_node_embedded_in(key, new_bkts, new_idx) != nullptr;
                        } else {
                            return find_node_in(key, new_bkts, new_idx) != nullptr;
                        }
                    } else {
                        // Not yet migrated — look in old array
                        auto guard = const_cast<concurrent_hash_table*>(this)->lock_bucket_shared(old_idx);
                        if (rehash_progress_.load(std::memory_order_acquire) <= old_idx) {
                            if constexpr (EmbeddedChain) {
                                return find_node_embedded(key, old_idx) != nullptr;
                            } else {
                                return find_node(key, old_idx) != nullptr;
                            }
                        }
                        // Bucket was migrated, retry
                    }
                }
            }
            if constexpr (EmbeddedChain) {
                if (enable_optimistic_read_) {
                    for (int _retry = 0; _retry < kOptimisticReadMaxRetries; ++_retry) { if (_retry) LRU_SPIN_PAUSE();
                        auto sl1 = seqlock_.load(std::memory_order_acquire);
                        if (sl1 & 1u) continue;
                        size_type idx = h & bucket_mask_.load(std::memory_order_acquire);
                        auto v1 = buckets_[idx].version.load(std::memory_order_acquire);
                        bool found = find_node_embedded(key, idx) != nullptr;
                        auto v2 = buckets_[idx].version.load(std::memory_order_acquire);
                        auto sl2 = seqlock_.load(std::memory_order_acquire);
                        if (sl1 == sl2 && (v1 == v2) && (v1 & 1u) == 0) return found;
                    }
                }
                auto sl = seqlock_.load(std::memory_order_acquire);
                size_type idx = h & bucket_mask_.load(std::memory_order_relaxed);
                auto guard = const_cast<concurrent_hash_table*>(this)->lock_bucket_shared(idx);
                return find_node_embedded(key, idx) != nullptr;
            } else {
                if (enable_optimistic_read_) {
                    for (int _retry = 0; _retry < kOptimisticReadMaxRetries; ++_retry) { if (_retry) LRU_SPIN_PAUSE();
                        auto sl1 = seqlock_.load(std::memory_order_acquire);
                        if (sl1 & 1u) continue;
                        size_type idx = h & bucket_mask_.load(std::memory_order_acquire);
                        auto v1 = buckets_[idx].version.load(std::memory_order_acquire);
                        bool found = find_node(key, idx) != nullptr;
                        auto v2 = buckets_[idx].version.load(std::memory_order_acquire);
                        auto sl2 = seqlock_.load(std::memory_order_acquire);
                        if (sl1 == sl2 && (v1 == v2) && (v1 & 1u) == 0) return found;
                    }
                }
                auto sl = seqlock_.load(std::memory_order_acquire);
                size_type idx = h & bucket_mask_.load(std::memory_order_relaxed);
                auto guard = const_cast<concurrent_hash_table*>(this)->lock_bucket_shared(idx);
                return find_node(key, idx) != nullptr;
            }
        }
    }

    /// Shared-lock lookup returning the embedded node pointer (no pinning).
    /// Used by TTL-aware contains(): the caller must NOT dereference the
    /// returned pointer after this method returns (the lock is released).
    /// Returns nullptr if the key is absent. Handles incremental rehash via
    /// the same dual-array logic as contains().
    template <typename K>
    Value find_embedded_shared(const K& key) const {
        const size_type h = hash_(key);
        // During incremental rehash F14 chunks migrate differently than
        // chain buckets — mirror contains() by delegating to the dual-array
        // lookup, which is self-locking.
        if (incremental_rehash_ && rehash_in_progress_.load(std::memory_order_acquire)) {
            if constexpr (kIsF14) {
                return find_f14_dual_array(key, h);
            }
            for (;;) {
                size_type old_idx = h & bucket_mask_.load(std::memory_order_acquire);
                size_type progress = rehash_progress_.load(std::memory_order_acquire);
                if (old_idx < progress) {
                    size_type new_mask = rehash_new_bucket_count_.load(std::memory_order_acquire) - 1;
                    size_type new_idx = h & new_mask;
                    bucket_type* new_bkts = rehash_new_buckets_.load(std::memory_order_acquire);
                    shared_bucket_lock guard(new_bkts[new_idx].spin);
                    if constexpr (EmbeddedChain) {
                        return find_node_embedded_in(key, new_bkts, new_idx);
                    } else {
                        return find_node_in(key, new_bkts, new_idx);
                    }
                } else {
                    auto guard = const_cast<concurrent_hash_table*>(this)->lock_bucket_shared(old_idx);
                    if (rehash_progress_.load(std::memory_order_acquire) <= old_idx) {
                        if constexpr (EmbeddedChain) {
                            return find_node_embedded(key, old_idx);
                        } else {
                            return find_node(key, old_idx);
                        }
                    }
                    // Bucket was migrated — retry.
                }
            }
        }
        const size_type idx = h & bucket_mask_.load(std::memory_order_acquire);
        auto guard = const_cast<concurrent_hash_table*>(this)->lock_bucket_shared(idx);
        if constexpr (kIsF14) {
            return find_f14_embedded(key, idx);
        } else if constexpr (EmbeddedChain) {
            return find_node_embedded(key, idx);
        } else {
            return find_node(key, idx);
        }
    }

    // ========================================================================
    // Bucket locking
    // ========================================================================

    exclusive_bucket_lock lock_bucket(const Key& key) {
        const size_type h = hash_(key);
        return lock_bucket(bucket_for_hash(h));
    }

    exclusive_bucket_lock lock_bucket_for_hash(size_type hash_val) {
        return lock_bucket(bucket_for_hash(hash_val));
    }

    exclusive_bucket_lock lock_bucket(size_type idx) {
        if constexpr (kIsF14) {
            return exclusive_bucket_lock(chunks_[idx].spin);
        } else {
            return exclusive_bucket_lock(buckets_[idx].spin);
        }
    }

    shared_bucket_lock lock_bucket_shared(const Key& key) {
        const size_type h = hash_(key);
        return lock_bucket_shared(bucket_for_hash(h));
    }

    shared_bucket_lock lock_bucket_shared_for_hash(size_type hash_val) {
        return lock_bucket_shared(bucket_for_hash(hash_val));
    }

    shared_bucket_lock lock_bucket_shared(size_type idx) {
        if constexpr (kIsF14) {
            return shared_bucket_lock(chunks_[idx].spin);
        } else {
            return shared_bucket_lock(buckets_[idx].spin);
        }
    }

    // ========================================================================
    // Insertion
    // ========================================================================

    auto insert(const Key& key, Value value) {
        // P0-5 (T1.3): hash is computed next line anyway; rehash only the
        // owning segment for segmented tables (no-op for non-segmented).
        const size_type h = hash_(key);
        rehash_if_needed(h);
        const size_type idx = bucket_for_hash(h);
        auto guard = lock_bucket(idx);

        if constexpr (kIsF14) {
            // F14 incremental rehash: dual-array aware insert
            if (incremental_rehash_ && rehash_in_progress_.load(std::memory_order_acquire)) {
                guard.unlock();
                for (;;) {
                    size_type old_idx = h & bucket_mask_.load(std::memory_order_acquire);
                    size_type progress = rehash_progress_.load(std::memory_order_acquire);
                    if (old_idx < progress) {
                        size_type new_mask = rehash_new_bucket_count_.load(std::memory_order_acquire) - 1;
                        size_type new_idx = h & new_mask;
                        f14_chunk_type* new_chunks = rehash_new_chunks_.load(std::memory_order_acquire);
                        exclusive_bucket_lock eguard(new_chunks[new_idx].spin);
                        if (rehash_progress_.load(std::memory_order_acquire) >= progress) {
                            if constexpr (EmbeddedChain) {
                                Value existing = find_f14_embedded_in(key, new_chunks, new_idx);
                                if (existing) return std::pair<Value, bool>{existing, false};
                                new_chunks[new_idx].version.fetch_add(1, std::memory_order_release);
                                value->set_cached_hash(h);
                                int slot = find_empty_slot_in_chunk(new_chunks[new_idx]);
                                if (slot >= 0) {
                                    new_chunks[new_idx].tags[slot] = f14_detail::f14_tag(h);
                                    new_chunks[new_idx].slots[slot] = value;
                                    new_chunks[new_idx].occupied_mask |= static_cast<uint16_t>(1u << slot);
                                } else {
                                    value->set_hash_chain_next(new_chunks[new_idx].embed_head.load(std::memory_order_acquire));
                                    new_chunks[new_idx].embed_head.store(value, std::memory_order_release);
                                }
                                size_.fetch_add(1, std::memory_order_relaxed);
                                new_chunks[new_idx].version.fetch_add(1, std::memory_order_release);
                                eguard.unlock();
                                rehash_step(1);
                                return std::pair<Value, bool>{value, true};
                            } else {
                                node_type* existing = find_f14_node_in(key, new_chunks, new_idx);
                                if (existing) return std::pair<Value*, bool>{&(existing->value), false};
                                new_chunks[new_idx].version.fetch_add(1, std::memory_order_release);
                                int slot = find_empty_slot_in_chunk(new_chunks[new_idx]);
                                node_type* new_node;
                                if (slot >= 0) {
                                    new_node = allocate_node(key, std::move(value), h, nullptr);
                                    new_chunks[new_idx].tags[slot] = f14_detail::f14_tag(h);
                                    new_chunks[new_idx].slots[slot] = new_node;
                                    new_chunks[new_idx].occupied_mask |= static_cast<uint16_t>(1u << slot);
                                } else {
                                    new_node = allocate_node(key, std::move(value), h, new_chunks[new_idx].node_head.load(std::memory_order_acquire));
                                    new_chunks[new_idx].node_head.store(new_node, std::memory_order_release);
                                }
                                size_.fetch_add(1, std::memory_order_relaxed);
                                new_chunks[new_idx].version.fetch_add(1, std::memory_order_release);
                                eguard.unlock();
                                rehash_step(1);
                                return std::pair<Value*, bool>{&(new_node->value), true};
                            }
                        }
                    } else {
                        exclusive_bucket_lock eguard(chunks_[old_idx].spin);
                        if (rehash_progress_.load(std::memory_order_acquire) <= old_idx) {
                            if constexpr (EmbeddedChain) {
                                Value existing = find_f14_embedded(key, old_idx);
                                if (existing) return std::pair<Value, bool>{existing, false};
                                chunks_[old_idx].version.fetch_add(1, std::memory_order_release);
                                value->set_cached_hash(h);
                                int slot = find_f14_empty_slot(old_idx);
                                if (slot >= 0) {
                                    chunks_[old_idx].tags[slot] = f14_detail::f14_tag(h);
                                    chunks_[old_idx].slots[slot] = value;
                                    chunks_[old_idx].occupied_mask |= static_cast<uint16_t>(1u << slot);
                                } else {
                                    value->set_hash_chain_next(chunks_[old_idx].embed_head.load(std::memory_order_acquire));
                                    chunks_[old_idx].embed_head.store(value, std::memory_order_release);
                                }
                                size_.fetch_add(1, std::memory_order_relaxed);
                                chunks_[old_idx].version.fetch_add(1, std::memory_order_release);
                                eguard.unlock();
                                rehash_step(1);
                                return std::pair<Value, bool>{value, true};
                            } else {
                                node_type* existing = find_f14_node(key, old_idx);
                                if (existing) return std::pair<Value*, bool>{&(existing->value), false};
                                chunks_[old_idx].version.fetch_add(1, std::memory_order_release);
                                int slot = find_f14_empty_slot(old_idx);
                                node_type* new_node;
                                if (slot >= 0) {
                                    new_node = allocate_node(key, std::move(value), h, nullptr);
                                    chunks_[old_idx].tags[slot] = f14_detail::f14_tag(h);
                                    chunks_[old_idx].slots[slot] = new_node;
                                    chunks_[old_idx].occupied_mask |= static_cast<uint16_t>(1u << slot);
                                } else {
                                    new_node = allocate_node(key, std::move(value), h, chunks_[old_idx].node_head.load(std::memory_order_acquire));
                                    chunks_[old_idx].node_head.store(new_node, std::memory_order_release);
                                }
                                size_.fetch_add(1, std::memory_order_relaxed);
                                chunks_[old_idx].version.fetch_add(1, std::memory_order_release);
                                eguard.unlock();
                                rehash_step(1);
                                return std::pair<Value*, bool>{&(new_node->value), true};
                            }
                        }
                    }
                }
            }
            if constexpr (EmbeddedChain) {
                Value existing = find_f14_embedded(key, idx);
                if (existing) return std::pair<Value, bool>{existing, false};

                chunks_[idx].version.fetch_add(1, std::memory_order_release);

                uint8_t tag = f14_detail::f14_tag(h);
                int slot = find_f14_empty_slot(idx);
                if (slot >= 0) {
                    chunks_[idx].tags[slot] = tag;
                    chunks_[idx].slots[slot] = value;
                    chunks_[idx].occupied_mask |= static_cast<uint16_t>(1u << slot);
                } else {
                    value->set_hash_chain_next(chunks_[idx].embed_head.load(std::memory_order_acquire));
                    chunks_[idx].embed_head.store(value, std::memory_order_release);
                }
                value->set_cached_hash(h);
                size_.fetch_add(1, std::memory_order_relaxed);

                chunks_[idx].version.fetch_add(1, std::memory_order_release);
                return std::pair<Value, bool>{value, true};
            } else {
                node_type* existing = find_f14_node(key, idx);
                if (existing) return std::pair<Value*, bool>{&(existing->value), false};

                chunks_[idx].version.fetch_add(1, std::memory_order_release);

                uint8_t tag = f14_detail::f14_tag(h);
                int slot = find_f14_empty_slot(idx);
                if (slot >= 0) {
                    node_type* new_node = allocate_node(key, std::move(value), h, nullptr);
                    chunks_[idx].tags[slot] = tag;
                    chunks_[idx].slots[slot] = new_node;
                    chunks_[idx].occupied_mask |= static_cast<uint16_t>(1u << slot);
                } else {
                    node_type* new_node = allocate_node(key, std::move(value), h, chunks_[idx].node_head.load(std::memory_order_acquire));
                    chunks_[idx].node_head.store(new_node, std::memory_order_release);
                }
                size_.fetch_add(1, std::memory_order_relaxed);

                chunks_[idx].version.fetch_add(1, std::memory_order_release);
                node_type* inserted = find_f14_node(key, idx);
                return std::pair<Value*, bool>{&(inserted->value), true};
            }
        } else {
            // Incremental rehash path for chain mode insert
            if (incremental_rehash_ && rehash_in_progress_.load(std::memory_order_acquire)) {
                // P-FIX (deadlock): Release the bucket lock acquired above before
                // entering the dual-array for(;;) loop. The loop re-acquires either
                // the old-array or new-array bucket lock depending on migration
                // progress. Without this unlock, the old-array lock acquired at
                // line `auto guard = lock_bucket(idx)` remains held, and the loop's
                // `lock_bucket(old_idx)` branch deadlocks when old_idx == idx
                // (shared_spinlock is non-reentrant). The F14 path above already
                // does this via `guard.unlock()` — the chain-mode path was missing
                // it, causing a self-deadlock under sharded_mm_lru + segmented
                // hash table + incremental rehash when per-segment load factor
                // exceeded the overload threshold and triggered rehash mid-insert.
                guard.unlock();
                for (;;) {
                    size_type old_idx = h & bucket_mask_.load(std::memory_order_acquire);
                    size_type progress = rehash_progress_.load(std::memory_order_acquire);
                    if (old_idx < progress) {
                        // Migrated — insert into new array
                        size_type new_mask = rehash_new_bucket_count_.load(std::memory_order_acquire) - 1;
                        size_type new_idx = h & new_mask;
                        bucket_type* new_bkts = rehash_new_buckets_.load(std::memory_order_acquire);
                        auto guard = exclusive_bucket_lock(new_bkts[new_idx].spin);
                        if (rehash_progress_.load(std::memory_order_acquire) >= progress) {
                            if constexpr (EmbeddedChain) {
                                Value existing = find_node_embedded_in(key, new_bkts, new_idx);
                                if (existing) return std::pair<Value, bool>{existing, false};
                                new_bkts[new_idx].version.fetch_add(1, std::memory_order_release);
                                value->set_cached_hash(h);
                                value->set_hash_chain_next(new_bkts[new_idx].embed_head.load(std::memory_order_acquire));
                                new_bkts[new_idx].embed_head.store(value, std::memory_order_release);
                                size_.fetch_add(1, std::memory_order_relaxed);
                                new_bkts[new_idx].version.fetch_add(1, std::memory_order_release);
                            } else {
                                node_type* existing = find_node_in(key, new_bkts, new_idx);
                                if (existing) return std::pair<Value*, bool>{&(existing->value), false};
                                new_bkts[new_idx].version.fetch_add(1, std::memory_order_release);
                                node_type* new_node = allocate_node(key, std::move(value), h, new_bkts[new_idx].node_head.load(std::memory_order_acquire));
                                new_bkts[new_idx].node_head.store(new_node, std::memory_order_release);
                                size_.fetch_add(1, std::memory_order_relaxed);
                                new_bkts[new_idx].version.fetch_add(1, std::memory_order_release);
                                guard.unlock();
                                rehash_step(1);
                                return std::pair<Value*, bool>{&(new_node->value), true};
                            }
                            guard.unlock();
                            rehash_step(1);
                            return std::pair<Value, bool>{value, true};
                        }
                        // Progress changed, retry
                    } else {
                        // Not yet migrated — insert into old array
                        auto guard = lock_bucket(old_idx);
                        if (rehash_progress_.load(std::memory_order_acquire) <= old_idx) {
                            if constexpr (EmbeddedChain) {
                                Value existing = find_node_embedded(key, old_idx);
                                if (existing) return std::pair<Value, bool>{existing, false};
                                buckets_[old_idx].version.fetch_add(1, std::memory_order_release);
                                value->set_cached_hash(h);
                                value->set_hash_chain_next(buckets_[old_idx].embed_head.load(std::memory_order_acquire));
                                buckets_[old_idx].embed_head.store(value, std::memory_order_release);
                                size_.fetch_add(1, std::memory_order_relaxed);
                                buckets_[old_idx].version.fetch_add(1, std::memory_order_release);
                            } else {
                                node_type* existing = find_node(key, old_idx);
                                if (existing) return std::pair<Value*, bool>{&(existing->value), false};
                                buckets_[old_idx].version.fetch_add(1, std::memory_order_release);
                                node_type* new_node = allocate_node(key, std::move(value), h, buckets_[old_idx].node_head.load(std::memory_order_acquire));
                                buckets_[old_idx].node_head.store(new_node, std::memory_order_release);
                                size_.fetch_add(1, std::memory_order_relaxed);
                                buckets_[old_idx].version.fetch_add(1, std::memory_order_release);
                                guard.unlock();
                                rehash_step(1);
                                return std::pair<Value*, bool>{&(new_node->value), true};
                            }
                            guard.unlock();
                            rehash_step(1);
                            return std::pair<Value, bool>{value, true};
                        }
                        // Bucket was migrated, retry
                    }
                }
            }
            // Normal (non-incremental) chain mode insert
            if constexpr (EmbeddedChain) {
                Value existing = find_node_embedded(key, idx);
                if (existing) return std::pair<Value, bool>{existing, false};

                buckets_[idx].version.fetch_add(1, std::memory_order_release);
                value->set_cached_hash(h);
                value->set_hash_chain_next(buckets_[idx].embed_head.load(std::memory_order_acquire));
                buckets_[idx].embed_head.store(value, std::memory_order_release);
                size_.fetch_add(1, std::memory_order_relaxed);
                buckets_[idx].version.fetch_add(1, std::memory_order_release);
                return std::pair<Value, bool>{value, true};
            } else {
                node_type* existing = find_node(key, idx);
                if (existing) return std::pair<Value*, bool>{&(existing->value), false};

                buckets_[idx].version.fetch_add(1, std::memory_order_release);
                node_type* new_node = allocate_node(key, std::move(value), h, buckets_[idx].node_head.load(std::memory_order_acquire));
                buckets_[idx].node_head.store(new_node, std::memory_order_release);
                size_.fetch_add(1, std::memory_order_relaxed);
                buckets_[idx].version.fetch_add(1, std::memory_order_release);
                return std::pair<Value*, bool>{&(new_node->value), true};
            }
        }
    }

    auto insert_or_assign(const Key& key, Value value) {
        // P0-5 (T1.3): hash is computed next line anyway; rehash only the
        // owning segment for segmented tables (no-op for non-segmented).
        const size_type h = hash_(key);
        rehash_if_needed(h);
        const size_type idx = bucket_for_hash(h);
        auto guard = lock_bucket(idx);

        if constexpr (kIsF14) {
            // F14 incremental rehash: dual-array aware insert_or_assign
            if (incremental_rehash_ && rehash_in_progress_.load(std::memory_order_acquire)) {
                guard.unlock();
                for (;;) {
                    size_type old_idx = h & bucket_mask_.load(std::memory_order_acquire);
                    size_type progress = rehash_progress_.load(std::memory_order_acquire);
                    if (old_idx < progress) {
                        size_type new_mask = rehash_new_bucket_count_.load(std::memory_order_acquire) - 1;
                        size_type new_idx = h & new_mask;
                        f14_chunk_type* new_chunks = rehash_new_chunks_.load(std::memory_order_acquire);
                        exclusive_bucket_lock eguard(new_chunks[new_idx].spin);
                        if (rehash_progress_.load(std::memory_order_acquire) >= progress) {
                            if constexpr (EmbeddedChain) {
                                Value existing = find_f14_embedded_in(key, new_chunks, new_idx);
                                if (existing) {
                                    new_chunks[new_idx].version.fetch_add(1, std::memory_order_release);
                                    value->set_cached_hash(h);
                                    value->set_hash_chain_next(existing->hash_chain_next());
                                    new_chunks[new_idx].version.fetch_add(1, std::memory_order_release);
                                    return std::pair<Value, bool>{value, false};
                                }
                                new_chunks[new_idx].version.fetch_add(1, std::memory_order_release);
                                value->set_cached_hash(h);
                                int slot = find_empty_slot_in_chunk(new_chunks[new_idx]);
                                if (slot >= 0) {
                                    new_chunks[new_idx].tags[slot] = f14_detail::f14_tag(h);
                                    new_chunks[new_idx].slots[slot] = value;
                                    new_chunks[new_idx].occupied_mask |= static_cast<uint16_t>(1u << slot);
                                } else {
                                    value->set_hash_chain_next(new_chunks[new_idx].embed_head.load(std::memory_order_acquire));
                                    new_chunks[new_idx].embed_head.store(value, std::memory_order_release);
                                }
                                size_.fetch_add(1, std::memory_order_relaxed);
                                new_chunks[new_idx].version.fetch_add(1, std::memory_order_release);
                                eguard.unlock();
                                rehash_step(1);
                                return std::pair<Value, bool>{value, true};
                            } else {
                                node_type* existing = find_f14_node_in(key, new_chunks, new_idx);
                                if (existing) {
                                    new_chunks[new_idx].version.fetch_add(1, std::memory_order_release);
                                    existing->value = std::move(value);
                                    new_chunks[new_idx].version.fetch_add(1, std::memory_order_release);
                                    return std::pair<Value*, bool>{&(existing->value), false};
                                }
                                new_chunks[new_idx].version.fetch_add(1, std::memory_order_release);
                                int slot = find_empty_slot_in_chunk(new_chunks[new_idx]);
                                node_type* new_node;
                                if (slot >= 0) {
                                    new_node = allocate_node(key, std::move(value), h, nullptr);
                                    new_chunks[new_idx].tags[slot] = f14_detail::f14_tag(h);
                                    new_chunks[new_idx].slots[slot] = new_node;
                                    new_chunks[new_idx].occupied_mask |= static_cast<uint16_t>(1u << slot);
                                } else {
                                    new_node = allocate_node(key, std::move(value), h, new_chunks[new_idx].node_head.load(std::memory_order_acquire));
                                    new_chunks[new_idx].node_head.store(new_node, std::memory_order_release);
                                }
                                size_.fetch_add(1, std::memory_order_relaxed);
                                new_chunks[new_idx].version.fetch_add(1, std::memory_order_release);
                                eguard.unlock();
                                rehash_step(1);
                                return std::pair<Value*, bool>{&(new_node->value), true};
                            }
                        }
                    } else {
                        exclusive_bucket_lock eguard(chunks_[old_idx].spin);
                        if (rehash_progress_.load(std::memory_order_acquire) <= old_idx) {
                            if constexpr (EmbeddedChain) {
                                Value existing = find_f14_embedded(key, old_idx);
                                if (existing) {
                                    chunks_[old_idx].version.fetch_add(1, std::memory_order_release);
                                    value->set_cached_hash(h);
                                    value->set_hash_chain_next(existing->hash_chain_next());
                                    chunks_[old_idx].version.fetch_add(1, std::memory_order_release);
                                    return std::pair<Value, bool>{value, false};
                                }
                                chunks_[old_idx].version.fetch_add(1, std::memory_order_release);
                                value->set_cached_hash(h);
                                int slot = find_f14_empty_slot(old_idx);
                                if (slot >= 0) {
                                    chunks_[old_idx].tags[slot] = f14_detail::f14_tag(h);
                                    chunks_[old_idx].slots[slot] = value;
                                    chunks_[old_idx].occupied_mask |= static_cast<uint16_t>(1u << slot);
                                } else {
                                    value->set_hash_chain_next(chunks_[old_idx].embed_head.load(std::memory_order_acquire));
                                    chunks_[old_idx].embed_head.store(value, std::memory_order_release);
                                }
                                size_.fetch_add(1, std::memory_order_relaxed);
                                chunks_[old_idx].version.fetch_add(1, std::memory_order_release);
                                eguard.unlock();
                                rehash_step(1);
                                return std::pair<Value, bool>{value, true};
                            } else {
                                node_type* existing = find_f14_node(key, old_idx);
                                if (existing) {
                                    chunks_[old_idx].version.fetch_add(1, std::memory_order_release);
                                    existing->value = std::move(value);
                                    chunks_[old_idx].version.fetch_add(1, std::memory_order_release);
                                    return std::pair<Value*, bool>{&(existing->value), false};
                                }
                                chunks_[old_idx].version.fetch_add(1, std::memory_order_release);
                                int slot = find_f14_empty_slot(old_idx);
                                node_type* new_node;
                                if (slot >= 0) {
                                    new_node = allocate_node(key, std::move(value), h, nullptr);
                                    chunks_[old_idx].tags[slot] = f14_detail::f14_tag(h);
                                    chunks_[old_idx].slots[slot] = new_node;
                                    chunks_[old_idx].occupied_mask |= static_cast<uint16_t>(1u << slot);
                                } else {
                                    new_node = allocate_node(key, std::move(value), h, chunks_[old_idx].node_head.load(std::memory_order_acquire));
                                    chunks_[old_idx].node_head.store(new_node, std::memory_order_release);
                                }
                                size_.fetch_add(1, std::memory_order_relaxed);
                                chunks_[old_idx].version.fetch_add(1, std::memory_order_release);
                                eguard.unlock();
                                rehash_step(1);
                                return std::pair<Value*, bool>{&(new_node->value), true};
                            }
                        }
                    }
                }
            }
            if constexpr (EmbeddedChain) {
                // Search inline slots for existing key
                uint8_t tag = f14_detail::f14_tag(h);
                uint16_t match_mask = f14_detail::f14_match_tags(chunks_[idx].tags, tag)
                                    & chunks_[idx].occupied_mask;
                while (match_mask) {
                    int slot = f14_detail::ctz16(match_mask);
                    match_mask &= static_cast<uint16_t>(match_mask - 1);
                    Value node = static_cast<Value>(chunks_[idx].slots[slot]);
                    if (equal_(node->key, key)) {
                        chunks_[idx].version.fetch_add(1, std::memory_order_release);
                        value->set_cached_hash(h);
                        value->set_hash_chain_next(node->hash_chain_next());
                        chunks_[idx].slots[slot] = value;
                        chunks_[idx].tags[slot] = tag;
                        node->set_hash_chain_next(nullptr);
                        chunks_[idx].version.fetch_add(1, std::memory_order_release);
                        return std::pair<Value, bool>{value, false};
                    }
                }
                // Search overflow chain
                {
                    Value prev = nullptr;
                    Value curr = chunks_[idx].embed_head.load(std::memory_order_acquire);
                    while (curr) {
                        if (equal_(curr->key, key)) {
                            chunks_[idx].version.fetch_add(1, std::memory_order_release);
                            value->set_cached_hash(h);
                            value->set_hash_chain_next(curr->hash_chain_next());
                            if (prev) prev->set_hash_chain_next(value);
                            else chunks_[idx].embed_head.store(value, std::memory_order_release);
                            curr->set_hash_chain_next(nullptr);
                            chunks_[idx].version.fetch_add(1, std::memory_order_release);
                            return std::pair<Value, bool>{value, false};
                        }
                        prev = curr;
                        curr = static_cast<Value>(curr->hash_chain_next());
                    }
                }
                // Not found: insert
                chunks_[idx].version.fetch_add(1, std::memory_order_release);
                value->set_cached_hash(h);
                int slot = find_f14_empty_slot(idx);
                if (slot >= 0) {
                    chunks_[idx].tags[slot] = tag;
                    chunks_[idx].slots[slot] = value;
                    chunks_[idx].occupied_mask |= static_cast<uint16_t>(1u << slot);
                } else {
                    value->set_hash_chain_next(chunks_[idx].embed_head.load(std::memory_order_acquire));
                    chunks_[idx].embed_head.store(value, std::memory_order_release);
                }
                size_.fetch_add(1, std::memory_order_relaxed);
                chunks_[idx].version.fetch_add(1, std::memory_order_release);
                return std::pair<Value, bool>{value, true};
            } else {
                node_type* existing = find_f14_node(key, idx);
                if (existing) {
                    chunks_[idx].version.fetch_add(1, std::memory_order_release);
                    existing->value = std::move(value);
                    chunks_[idx].version.fetch_add(1, std::memory_order_release);
                    return std::pair<Value*, bool>{&(existing->value), false};
                }

                chunks_[idx].version.fetch_add(1, std::memory_order_release);
                uint8_t tag = f14_detail::f14_tag(h);
                int slot = find_f14_empty_slot(idx);
                if (slot >= 0) {
                    node_type* new_node = allocate_node(key, std::move(value), h, nullptr);
                    chunks_[idx].tags[slot] = tag;
                    chunks_[idx].slots[slot] = new_node;
                    chunks_[idx].occupied_mask |= static_cast<uint16_t>(1u << slot);
                } else {
                    node_type* new_node = allocate_node(key, std::move(value), h, chunks_[idx].node_head.load(std::memory_order_acquire));
                    chunks_[idx].node_head.store(new_node, std::memory_order_release);
                }
                size_.fetch_add(1, std::memory_order_relaxed);
                chunks_[idx].version.fetch_add(1, std::memory_order_release);
                node_type* inserted = find_f14_node(key, idx);
                return std::pair<Value*, bool>{&(inserted->value), true};
            }
        } else {
            // Incremental rehash path for chain mode insert_or_assign
            if (incremental_rehash_ && rehash_in_progress_.load(std::memory_order_acquire)) {
                // P-FIX (deadlock): Release the bucket lock acquired above before
                // entering the dual-array for(;;) loop — same fix as insert()/erase().
                guard.unlock();
                for (;;) {
                    size_type old_idx = h & bucket_mask_.load(std::memory_order_acquire);
                    size_type progress = rehash_progress_.load(std::memory_order_acquire);
                    if (old_idx < progress) {
                        // Migrated — operate on new array
                        size_type new_mask = rehash_new_bucket_count_.load(std::memory_order_acquire) - 1;
                        size_type new_idx = h & new_mask;
                        bucket_type* new_bkts = rehash_new_buckets_.load(std::memory_order_acquire);
                        auto guard = exclusive_bucket_lock(new_bkts[new_idx].spin);
                        if (rehash_progress_.load(std::memory_order_acquire) >= progress) {
                            if constexpr (EmbeddedChain) {
                                Value prev = nullptr;
                                Value curr = new_bkts[new_idx].embed_head.load(std::memory_order_acquire);
                                while (curr) {
                                    if (equal_(curr->key, key)) {
                                        new_bkts[new_idx].version.fetch_add(1, std::memory_order_release);
                                        value->set_cached_hash(h);
                                        value->set_hash_chain_next(curr->hash_chain_next());
                                        if (prev) prev->set_hash_chain_next(value);
                                        else new_bkts[new_idx].embed_head.store(value, std::memory_order_release);
                                        curr->set_hash_chain_next(nullptr);
                                        new_bkts[new_idx].version.fetch_add(1, std::memory_order_release);
                                        return std::pair<Value, bool>{value, false};
                                    }
                                    prev = curr;
                                    curr = static_cast<Value>(curr->hash_chain_next());
                                }
                                // Not found: insert
                                new_bkts[new_idx].version.fetch_add(1, std::memory_order_release);
                                value->set_cached_hash(h);
                                value->set_hash_chain_next(new_bkts[new_idx].embed_head.load(std::memory_order_acquire));
                                new_bkts[new_idx].embed_head.store(value, std::memory_order_release);
                                size_.fetch_add(1, std::memory_order_relaxed);
                                new_bkts[new_idx].version.fetch_add(1, std::memory_order_release);
                                guard.unlock();
                                rehash_step(1);
                                return std::pair<Value, bool>{value, true};
                            } else {
                                node_type* existing = find_node_in(key, new_bkts, new_idx);
                                if (existing) {
                                    new_bkts[new_idx].version.fetch_add(1, std::memory_order_release);
                                    existing->value = std::move(value);
                                    new_bkts[new_idx].version.fetch_add(1, std::memory_order_release);
                                    return std::pair<Value*, bool>{&(existing->value), false};
                                }
                                new_bkts[new_idx].version.fetch_add(1, std::memory_order_release);
                                node_type* new_node = allocate_node(key, std::move(value), h, new_bkts[new_idx].node_head.load(std::memory_order_acquire));
                                new_bkts[new_idx].node_head.store(new_node, std::memory_order_release);
                                size_.fetch_add(1, std::memory_order_relaxed);
                                new_bkts[new_idx].version.fetch_add(1, std::memory_order_release);
                                guard.unlock();
                                rehash_step(1);
                                return std::pair<Value*, bool>{&(new_node->value), true};
                            }
                        }
                        // Progress changed, retry
                    } else {
                        // Not yet migrated — operate on old array
                        auto guard = lock_bucket(old_idx);
                        if (rehash_progress_.load(std::memory_order_acquire) <= old_idx) {
                            if constexpr (EmbeddedChain) {
                                Value prev = nullptr;
                                Value curr = buckets_[old_idx].embed_head.load(std::memory_order_acquire);
                                while (curr) {
                                    if (equal_(curr->key, key)) {
                                        buckets_[old_idx].version.fetch_add(1, std::memory_order_release);
                                        value->set_cached_hash(h);
                                        value->set_hash_chain_next(curr->hash_chain_next());
                                        if (prev) prev->set_hash_chain_next(value);
                                        else buckets_[old_idx].embed_head.store(value, std::memory_order_release);
                                        curr->set_hash_chain_next(nullptr);
                                        buckets_[old_idx].version.fetch_add(1, std::memory_order_release);
                                        return std::pair<Value, bool>{value, false};
                                    }
                                    prev = curr;
                                    curr = static_cast<Value>(curr->hash_chain_next());
                                }
                                // Not found: insert
                                buckets_[old_idx].version.fetch_add(1, std::memory_order_release);
                                value->set_cached_hash(h);
                                value->set_hash_chain_next(buckets_[old_idx].embed_head.load(std::memory_order_acquire));
                                buckets_[old_idx].embed_head.store(value, std::memory_order_release);
                                size_.fetch_add(1, std::memory_order_relaxed);
                                buckets_[old_idx].version.fetch_add(1, std::memory_order_release);
                                guard.unlock();
                                rehash_step(1);
                                return std::pair<Value, bool>{value, true};
                            } else {
                                node_type* existing = find_node(key, old_idx);
                                if (existing) {
                                    buckets_[old_idx].version.fetch_add(1, std::memory_order_release);
                                    existing->value = std::move(value);
                                    buckets_[old_idx].version.fetch_add(1, std::memory_order_release);
                                    return std::pair<Value*, bool>{&(existing->value), false};
                                }
                                buckets_[old_idx].version.fetch_add(1, std::memory_order_release);
                                node_type* new_node = allocate_node(key, std::move(value), h, buckets_[old_idx].node_head.load(std::memory_order_acquire));
                                buckets_[old_idx].node_head.store(new_node, std::memory_order_release);
                                size_.fetch_add(1, std::memory_order_relaxed);
                                buckets_[old_idx].version.fetch_add(1, std::memory_order_release);
                                guard.unlock();
                                rehash_step(1);
                                return std::pair<Value*, bool>{&(new_node->value), true};
                            }
                        }
                        // Bucket was migrated, retry
                    }
                }
            }
            if constexpr (EmbeddedChain) {
                Value prev = nullptr;
                Value curr = buckets_[idx].embed_head.load(std::memory_order_acquire);
                while (curr) {
                    if (equal_(curr->key, key)) {
                        buckets_[idx].version.fetch_add(1, std::memory_order_release);
                        value->set_cached_hash(h);
                        value->set_hash_chain_next(curr->hash_chain_next());
                        if (prev) prev->set_hash_chain_next(value);
                        else buckets_[idx].embed_head.store(value, std::memory_order_release);
                        curr->set_hash_chain_next(nullptr);
                        buckets_[idx].version.fetch_add(1, std::memory_order_release);
                        return std::pair<Value, bool>{value, false};
                    }
                    prev = curr;
                    curr = static_cast<Value>(curr->hash_chain_next());
                }

                buckets_[idx].version.fetch_add(1, std::memory_order_release);
                value->set_cached_hash(h);
                value->set_hash_chain_next(buckets_[idx].embed_head.load(std::memory_order_acquire));
                buckets_[idx].embed_head.store(value, std::memory_order_release);
                size_.fetch_add(1, std::memory_order_relaxed);
                buckets_[idx].version.fetch_add(1, std::memory_order_release);
                return std::pair<Value, bool>{value, true};
            } else {
                node_type* existing = find_node(key, idx);
                if (existing) {
                    buckets_[idx].version.fetch_add(1, std::memory_order_release);
                    existing->value = std::move(value);
                    buckets_[idx].version.fetch_add(1, std::memory_order_release);
                    return std::pair<Value*, bool>{&(existing->value), false};
                }

                buckets_[idx].version.fetch_add(1, std::memory_order_release);
                node_type* new_node = allocate_node(key, std::move(value), h, buckets_[idx].node_head.load(std::memory_order_acquire));
                buckets_[idx].node_head.store(new_node, std::memory_order_release);
                size_.fetch_add(1, std::memory_order_relaxed);
                buckets_[idx].version.fetch_add(1, std::memory_order_release);
                return std::pair<Value*, bool>{&(new_node->value), true};
            }
        }
    }

    // ========================================================================
    // Erasure
    // ========================================================================

    bool erase(const Key& key) {
        const size_type h = hash_(key);
        const size_type idx = bucket_for_hash(h);
        auto guard = lock_bucket(idx);

        if constexpr (kIsF14) {
            // F14 incremental rehash: dual-array aware erase
            if (incremental_rehash_ && rehash_in_progress_.load(std::memory_order_acquire)) {
                guard.unlock();
                for (;;) {
                    size_type old_idx = h & bucket_mask_.load(std::memory_order_acquire);
                    size_type progress = rehash_progress_.load(std::memory_order_acquire);
                    if (old_idx < progress) {
                        size_type new_mask = rehash_new_bucket_count_.load(std::memory_order_acquire) - 1;
                        size_type new_idx = h & new_mask;
                        f14_chunk_type* new_chunks = rehash_new_chunks_.load(std::memory_order_acquire);
                        exclusive_bucket_lock eguard(new_chunks[new_idx].spin);
                        if (rehash_progress_.load(std::memory_order_acquire) >= progress) {
                            if constexpr (EmbeddedChain) {
                                uint8_t tag = f14_detail::f14_tag(h);
                                uint16_t match_mask = f14_detail::f14_match_tags(new_chunks[new_idx].tags, tag)
                                                    & new_chunks[new_idx].occupied_mask;
                                while (match_mask) {
                                    int slot = f14_detail::ctz16(match_mask);
                                    match_mask &= static_cast<uint16_t>(match_mask - 1);
                                    Value node = static_cast<Value>(new_chunks[new_idx].slots[slot]);
                                    if (equal_(node->key, key)) {
                                        new_chunks[new_idx].version.fetch_add(1, std::memory_order_release);
                                        new_chunks[new_idx].tags[slot] = f14_detail::kTagTombstone;
                                        new_chunks[new_idx].slots[slot] = nullptr;
                                        new_chunks[new_idx].occupied_mask &= static_cast<uint16_t>(~(1u << slot));
                                        node->set_hash_chain_next(nullptr);
                                        size_.fetch_sub(1, std::memory_order_relaxed);
                                        new_chunks[new_idx].version.fetch_add(1, std::memory_order_release);
                                        eguard.unlock();
                                        rehash_step(1);
                                        return true;
                                    }
                                }
                                Value prev = nullptr;
                                Value curr = new_chunks[new_idx].embed_head.load(std::memory_order_acquire);
                                while (curr) {
                                    if (equal_(curr->key, key)) {
                                        new_chunks[new_idx].version.fetch_add(1, std::memory_order_release);
                                        Value next = static_cast<Value>(curr->hash_chain_next());
                                        if (prev) prev->set_hash_chain_next(next);
                                        else new_chunks[new_idx].embed_head.store(next, std::memory_order_release);
                                        curr->set_hash_chain_next(nullptr);
                                        size_.fetch_sub(1, std::memory_order_relaxed);
                                        new_chunks[new_idx].version.fetch_add(1, std::memory_order_release);
                                        eguard.unlock();
                                        rehash_step(1);
                                        return true;
                                    }
                                    prev = curr;
                                    curr = static_cast<Value>(curr->hash_chain_next());
                                }
                            } else {
                                uint8_t tag = f14_detail::f14_tag(h);
                                uint16_t match_mask = f14_detail::f14_match_tags(new_chunks[new_idx].tags, tag)
                                                    & new_chunks[new_idx].occupied_mask;
                                while (match_mask) {
                                    int slot = f14_detail::ctz16(match_mask);
                                    match_mask &= static_cast<uint16_t>(match_mask - 1);
                                    node_type* node = static_cast<node_type*>(new_chunks[new_idx].slots[slot]);
                                    if (equal_(node->key, key)) {
                                        new_chunks[new_idx].version.fetch_add(1, std::memory_order_release);
                                        new_chunks[new_idx].tags[slot] = f14_detail::kTagTombstone;
                                        new_chunks[new_idx].slots[slot] = nullptr;
                                        new_chunks[new_idx].occupied_mask &= static_cast<uint16_t>(~(1u << slot));
                                        deallocate_node(node);
                                        size_.fetch_sub(1, std::memory_order_relaxed);
                                        new_chunks[new_idx].version.fetch_add(1, std::memory_order_release);
                                        eguard.unlock();
                                        rehash_step(1);
                                        return true;
                                    }
                                }
                                node_type* prev = nullptr;
                                node_type* curr = new_chunks[new_idx].node_head.load(std::memory_order_acquire);
                                while (curr) {
                                    if (equal_(curr->key, key)) {
                                        new_chunks[new_idx].version.fetch_add(1, std::memory_order_release);
                                        node_type* next = curr->next;
                                        if (prev) prev->next = next;
                                        else new_chunks[new_idx].node_head.store(next, std::memory_order_release);
                                        deallocate_node(curr);
                                        size_.fetch_sub(1, std::memory_order_relaxed);
                                        new_chunks[new_idx].version.fetch_add(1, std::memory_order_release);
                                        eguard.unlock();
                                        rehash_step(1);
                                        return true;
                                    }
                                    prev = curr;
                                    curr = curr->next;
                                }
                            }
                            eguard.unlock();
                            rehash_step(1);
                            return false;
                        }
                    } else {
                        exclusive_bucket_lock eguard(chunks_[old_idx].spin);
                        if (rehash_progress_.load(std::memory_order_acquire) <= old_idx) {
                            if constexpr (EmbeddedChain) {
                                uint8_t tag = f14_detail::f14_tag(h);
                                uint16_t match_mask = f14_detail::f14_match_tags(chunks_[old_idx].tags, tag)
                                                    & chunks_[old_idx].occupied_mask;
                                while (match_mask) {
                                    int slot = f14_detail::ctz16(match_mask);
                                    match_mask &= static_cast<uint16_t>(match_mask - 1);
                                    Value node = static_cast<Value>(chunks_[old_idx].slots[slot]);
                                    if (equal_(node->key, key)) {
                                        chunks_[old_idx].version.fetch_add(1, std::memory_order_release);
                                        chunks_[old_idx].tags[slot] = f14_detail::kTagTombstone;
                                        chunks_[old_idx].slots[slot] = nullptr;
                                        chunks_[old_idx].occupied_mask &= static_cast<uint16_t>(~(1u << slot));
                                        node->set_hash_chain_next(nullptr);
                                        size_.fetch_sub(1, std::memory_order_relaxed);
                                        chunks_[old_idx].version.fetch_add(1, std::memory_order_release);
                                        eguard.unlock();
                                        rehash_step(1);
                                        return true;
                                    }
                                }
                                Value prev = nullptr;
                                Value curr = chunks_[old_idx].embed_head.load(std::memory_order_acquire);
                                while (curr) {
                                    if (equal_(curr->key, key)) {
                                        chunks_[old_idx].version.fetch_add(1, std::memory_order_release);
                                        Value next = static_cast<Value>(curr->hash_chain_next());
                                        if (prev) prev->set_hash_chain_next(next);
                                        else chunks_[old_idx].embed_head.store(next, std::memory_order_release);
                                        curr->set_hash_chain_next(nullptr);
                                        size_.fetch_sub(1, std::memory_order_relaxed);
                                        chunks_[old_idx].version.fetch_add(1, std::memory_order_release);
                                        eguard.unlock();
                                        rehash_step(1);
                                        return true;
                                    }
                                    prev = curr;
                                    curr = static_cast<Value>(curr->hash_chain_next());
                                }
                            } else {
                                uint8_t tag = f14_detail::f14_tag(h);
                                uint16_t match_mask = f14_detail::f14_match_tags(chunks_[old_idx].tags, tag)
                                                    & chunks_[old_idx].occupied_mask;
                                while (match_mask) {
                                    int slot = f14_detail::ctz16(match_mask);
                                    match_mask &= static_cast<uint16_t>(match_mask - 1);
                                    node_type* node = static_cast<node_type*>(chunks_[old_idx].slots[slot]);
                                    if (equal_(node->key, key)) {
                                        chunks_[old_idx].version.fetch_add(1, std::memory_order_release);
                                        chunks_[old_idx].tags[slot] = f14_detail::kTagTombstone;
                                        chunks_[old_idx].slots[slot] = nullptr;
                                        chunks_[old_idx].occupied_mask &= static_cast<uint16_t>(~(1u << slot));
                                        deallocate_node(node);
                                        size_.fetch_sub(1, std::memory_order_relaxed);
                                        chunks_[old_idx].version.fetch_add(1, std::memory_order_release);
                                        eguard.unlock();
                                        rehash_step(1);
                                        return true;
                                    }
                                }
                                node_type* prev = nullptr;
                                node_type* curr = chunks_[old_idx].node_head.load(std::memory_order_acquire);
                                while (curr) {
                                    if (equal_(curr->key, key)) {
                                        chunks_[old_idx].version.fetch_add(1, std::memory_order_release);
                                        node_type* next = curr->next;
                                        if (prev) prev->next = next;
                                        else chunks_[old_idx].node_head.store(next, std::memory_order_release);
                                        deallocate_node(curr);
                                        size_.fetch_sub(1, std::memory_order_relaxed);
                                        chunks_[old_idx].version.fetch_add(1, std::memory_order_release);
                                        eguard.unlock();
                                        rehash_step(1);
                                        return true;
                                    }
                                    prev = curr;
                                    curr = curr->next;
                                }
                            }
                            eguard.unlock();
                            rehash_step(1);
                            return false;
                        }
                    }
                }
            }
            if constexpr (EmbeddedChain) {
                // Search inline slots
                uint8_t tag = f14_detail::f14_tag(h);
                uint16_t match_mask = f14_detail::f14_match_tags(chunks_[idx].tags, tag)
                                    & chunks_[idx].load_occupied_mask_acquire();
                while (match_mask) {
                    int slot = f14_detail::ctz16(match_mask);
                    match_mask &= static_cast<uint16_t>(match_mask - 1);
                    Value node = static_cast<Value>(chunks_[idx].slots[slot]);
                    if (equal_(node->key, key)) {
                        chunks_[idx].version.fetch_add(1, std::memory_order_release);
                        chunks_[idx].tags[slot] = f14_detail::kTagTombstone;
                        chunks_[idx].slots[slot] = nullptr;
                        chunks_[idx].occupied_mask &= static_cast<uint16_t>(~(1u << slot));
                        node->set_hash_chain_next(nullptr);
                        size_.fetch_sub(1, std::memory_order_relaxed);
                        chunks_[idx].version.fetch_add(1, std::memory_order_release);
                        return true;
                    }
                }
                // Search overflow chain
                Value prev = nullptr;
                Value curr = chunks_[idx].embed_head.load(std::memory_order_acquire);
                while (curr) {
                    if (equal_(curr->key, key)) {
                        chunks_[idx].version.fetch_add(1, std::memory_order_release);
                        Value next = static_cast<Value>(curr->hash_chain_next());
                        if (prev) prev->set_hash_chain_next(next);
                        else chunks_[idx].embed_head.store(next, std::memory_order_release);
                        curr->set_hash_chain_next(nullptr);
                        size_.fetch_sub(1, std::memory_order_relaxed);
                        chunks_[idx].version.fetch_add(1, std::memory_order_release);
                        return true;
                    }
                    prev = curr;
                    curr = static_cast<Value>(curr->hash_chain_next());
                }
                return false;
            } else {
                // Non-embedded F14
                uint8_t tag = f14_detail::f14_tag(h);
                uint16_t match_mask = f14_detail::f14_match_tags(chunks_[idx].tags, tag)
                                    & chunks_[idx].load_occupied_mask_acquire();
                while (match_mask) {
                    int slot = f14_detail::ctz16(match_mask);
                    match_mask &= static_cast<uint16_t>(match_mask - 1);
                    node_type* node = static_cast<node_type*>(chunks_[idx].slots[slot]);
                    if (equal_(node->key, key)) {
                        chunks_[idx].version.fetch_add(1, std::memory_order_release);
                        chunks_[idx].tags[slot] = f14_detail::kTagTombstone;
                        chunks_[idx].slots[slot] = nullptr;
                        chunks_[idx].occupied_mask &= static_cast<uint16_t>(~(1u << slot));
                        deallocate_node(node);
                        size_.fetch_sub(1, std::memory_order_relaxed);
                        chunks_[idx].version.fetch_add(1, std::memory_order_release);
                        return true;
                    }
                }
                // Search overflow chain
                {
                    node_type* prev = nullptr;
                    node_type* curr = chunks_[idx].node_head.load(std::memory_order_acquire);
                    while (curr) {
                        if (equal_(curr->key, key)) {
                            chunks_[idx].version.fetch_add(1, std::memory_order_release);
                            node_type* next = curr->next;
                            if (prev) prev->next = next;
                            else chunks_[idx].node_head.store(next, std::memory_order_release);
                            deallocate_node(curr);
                            size_.fetch_sub(1, std::memory_order_relaxed);
                            chunks_[idx].version.fetch_add(1, std::memory_order_release);
                            return true;
                        }
                        prev = curr;
                        curr = curr->next;
                    }
                }
                return false;
            }
        } else {
            // Incremental rehash path for chain mode erase
            if (incremental_rehash_ && rehash_in_progress_.load(std::memory_order_acquire)) {
                // P-FIX (deadlock): Release the bucket lock acquired above before
                // entering the dual-array for(;;) loop — same fix as insert().
                // Without this, lock_bucket(old_idx) in the not-yet-migrated
                // branch deadlocks on the already-held bucket lock.
                guard.unlock();
                for (;;) {
                    size_type old_idx = h & bucket_mask_.load(std::memory_order_acquire);
                    size_type progress = rehash_progress_.load(std::memory_order_acquire);
                    if (old_idx < progress) {
                        // Migrated — erase from new array
                        size_type new_mask = rehash_new_bucket_count_.load(std::memory_order_acquire) - 1;
                        size_type new_idx = h & new_mask;
                        bucket_type* new_bkts = rehash_new_buckets_.load(std::memory_order_acquire);
                        auto guard = exclusive_bucket_lock(new_bkts[new_idx].spin);
                        if (rehash_progress_.load(std::memory_order_acquire) >= progress) {
                            if constexpr (EmbeddedChain) {
                                Value prev = nullptr;
                                Value curr = new_bkts[new_idx].embed_head.load(std::memory_order_acquire);
                                while (curr) {
                                    if (equal_(curr->key, key)) {
                                        new_bkts[new_idx].version.fetch_add(1, std::memory_order_release);
                                        Value next = static_cast<Value>(curr->hash_chain_next());
                                        if (prev) prev->set_hash_chain_next(next);
                                        else new_bkts[new_idx].embed_head.store(next, std::memory_order_release);
                                        curr->set_hash_chain_next(nullptr);
                                        size_.fetch_sub(1, std::memory_order_relaxed);
                                        new_bkts[new_idx].version.fetch_add(1, std::memory_order_release);
                                        guard.unlock();
                                        rehash_step(1);
                                        return true;
                                    }
                                    prev = curr;
                                    curr = static_cast<Value>(curr->hash_chain_next());
                                }
                            } else {
                                node_type* prev = nullptr;
                                node_type* curr = new_bkts[new_idx].node_head.load(std::memory_order_acquire);
                                while (curr) {
                                    if (equal_(curr->key, key)) {
                                        new_bkts[new_idx].version.fetch_add(1, std::memory_order_release);
                                        node_type* next = curr->next;
                                        if (prev) prev->next = next;
                                        else new_bkts[new_idx].node_head.store(next, std::memory_order_release);
                                        deallocate_node(curr);
                                        size_.fetch_sub(1, std::memory_order_relaxed);
                                        new_bkts[new_idx].version.fetch_add(1, std::memory_order_release);
                                        guard.unlock();
                                        rehash_step(1);
                                        return true;
                                    }
                                    prev = curr;
                                    curr = curr->next;
                                }
                            }
                            return false;
                        }
                        // Progress changed, retry
                    } else {
                        // Not yet migrated — erase from old array
                        auto guard = lock_bucket(old_idx);
                        if (rehash_progress_.load(std::memory_order_acquire) <= old_idx) {
                            if constexpr (EmbeddedChain) {
                                Value prev = nullptr;
                                Value curr = buckets_[old_idx].embed_head.load(std::memory_order_acquire);
                                while (curr) {
                                    if (equal_(curr->key, key)) {
                                        buckets_[old_idx].version.fetch_add(1, std::memory_order_release);
                                        Value next = static_cast<Value>(curr->hash_chain_next());
                                        if (prev) prev->set_hash_chain_next(next);
                                        else buckets_[old_idx].embed_head.store(next, std::memory_order_release);
                                        curr->set_hash_chain_next(nullptr);
                                        size_.fetch_sub(1, std::memory_order_relaxed);
                                        buckets_[old_idx].version.fetch_add(1, std::memory_order_release);
                                        guard.unlock();
                                        rehash_step(1);
                                        return true;
                                    }
                                    prev = curr;
                                    curr = static_cast<Value>(curr->hash_chain_next());
                                }
                            } else {
                                node_type* prev = nullptr;
                                node_type* curr = buckets_[old_idx].node_head.load(std::memory_order_acquire);
                                while (curr) {
                                    if (equal_(curr->key, key)) {
                                        buckets_[old_idx].version.fetch_add(1, std::memory_order_release);
                                        node_type* next = curr->next;
                                        if (prev) prev->next = next;
                                        else buckets_[old_idx].node_head.store(next, std::memory_order_release);
                                        deallocate_node(curr);
                                        size_.fetch_sub(1, std::memory_order_relaxed);
                                        buckets_[old_idx].version.fetch_add(1, std::memory_order_release);
                                        guard.unlock();
                                        rehash_step(1);
                                        return true;
                                    }
                                    prev = curr;
                                    curr = curr->next;
                                }
                            }
                            return false;
                        }
                        // Bucket was migrated, retry
                    }
                }
            }
            if constexpr (EmbeddedChain) {
                Value prev = nullptr;
                Value curr = buckets_[idx].embed_head.load(std::memory_order_acquire);
                while (curr) {
                    if (equal_(curr->key, key)) {
                        buckets_[idx].version.fetch_add(1, std::memory_order_release);
                        Value next = static_cast<Value>(curr->hash_chain_next());
                        if (prev) prev->set_hash_chain_next(next);
                        else buckets_[idx].embed_head.store(next, std::memory_order_release);
                        curr->set_hash_chain_next(nullptr);
                        size_.fetch_sub(1, std::memory_order_relaxed);
                        buckets_[idx].version.fetch_add(1, std::memory_order_release);
                        return true;
                    }
                    prev = curr;
                    curr = static_cast<Value>(curr->hash_chain_next());
                }
                return false;
            } else {
                node_type* prev = nullptr;
                node_type* curr = buckets_[idx].node_head.load(std::memory_order_acquire);
                while (curr) {
                    if (equal_(curr->key, key)) {
                        buckets_[idx].version.fetch_add(1, std::memory_order_release);
                        node_type* next = curr->next;
                        if (prev) prev->next = next;
                        else buckets_[idx].node_head.store(next, std::memory_order_release);
                        deallocate_node(curr);
                        size_.fetch_sub(1, std::memory_order_relaxed);
                        buckets_[idx].version.fetch_add(1, std::memory_order_release);
                        return true;
                    }
                    prev = curr;
                    curr = curr->next;
                }
                return false;
            }
        }
    }

    // ========================================================================
    // Capacity
    // ========================================================================

    size_type size() const noexcept { return size_.load(std::memory_order_relaxed); }
    bool empty() const noexcept { return size() == 0; }
    size_type bucket_count() const noexcept {
        return bucket_mask_.load(std::memory_order_relaxed) + 1;
    }

    // ========================================================================
    // Bucket mapping
    // ========================================================================

    size_type bucket_for(const Key& key) const { return bucket_for_hash(hash_(key)); }
    size_type bucket_for_hash(size_type hash_val) const noexcept {
        return hash_val & bucket_mask_.load(std::memory_order_relaxed);
    }

    // ========================================================================
    // Bulk operations
    // ========================================================================

    void clear() noexcept {
        // If incremental rehash is in progress, clean up the new array too
        if (rehash_in_progress_.load(std::memory_order_acquire)) {
            if constexpr (kIsF14) {
                const size_type new_count = rehash_new_bucket_count_.load(std::memory_order_relaxed);
                f14_chunk_type* new_chunks = rehash_new_chunks_.load(std::memory_order_relaxed);
                if (new_chunks && new_count > 0) {
                    const size_type new_mask = new_count - 1;
                    for (size_type i = 0; i <= new_mask; ++i) {
                        if constexpr (EmbeddedChain) {
                            Value curr = new_chunks[i].embed_head.load(std::memory_order_acquire);
                            while (curr) {
                                Value next = static_cast<Value>(curr->hash_chain_next());
                                curr->set_hash_chain_next(nullptr);
                                curr = next;
                            }
                            new_chunks[i].embed_head.store(nullptr, std::memory_order_release);
                        } else {
                            node_type* node = new_chunks[i].node_head.load(std::memory_order_acquire);
                            while (node) {
                                node_type* next = node->next;
                                deallocate_node(node);
                                node = next;
                            }
                            new_chunks[i].node_head.store(nullptr, std::memory_order_release);
                        }
                    }
                }
                rehash_new_chunks_owner_.reset();
                rehash_new_chunks_.store(nullptr, std::memory_order_release);
            } else {
            const size_type new_count = rehash_new_bucket_count_.load(std::memory_order_relaxed);
            bucket_type* new_bkts = rehash_new_buckets_.load(std::memory_order_relaxed);
            if (new_bkts && new_count > 0) {
                const size_type new_mask = new_count - 1;
                for (size_type i = 0; i <= new_mask; ++i) {
                    if constexpr (EmbeddedChain) {
                        Value curr = new_bkts[i].embed_head.load(std::memory_order_acquire);
                        while (curr) {
                            Value next = static_cast<Value>(curr->hash_chain_next());
                            curr->set_hash_chain_next(nullptr);
                            curr = next;
                        }
                        new_bkts[i].embed_head.store(nullptr, std::memory_order_release);
                    } else {
                        node_type* node = new_bkts[i].node_head.load(std::memory_order_acquire);
                        while (node) {
                            node_type* next = node->next;
                            deallocate_node(node);
                            node = next;
                        }
                        new_bkts[i].node_head.store(nullptr, std::memory_order_release);
                    }
                }
            }
                rehash_new_buckets_owner_.reset();
            }
            rehash_in_progress_.store(false, std::memory_order_release);
            rehash_new_buckets_.store(nullptr, std::memory_order_release);
            rehash_new_chunks_.store(nullptr, std::memory_order_release);
            rehash_new_bucket_count_.store(0, std::memory_order_release);
            rehash_progress_.store(0, std::memory_order_release);
            rehash_migrating_.store(false, std::memory_order_release);
        }
        const size_type mask = bucket_mask_.load(std::memory_order_relaxed);
        if constexpr (kIsF14) {
            for (size_type i = 0; i <= mask; ++i) {
                chunks_[i].version.fetch_add(1, std::memory_order_release);
                for (int s = 0; s < f14_detail::kChunkCapacity; ++s) {
                    if (chunks_[i].occupied_mask & (1u << s)) {
                        if constexpr (!EmbeddedChain) {
                            node_type* node = static_cast<node_type*>(chunks_[i].slots[s]);
                            deallocate_node(node);
                        }
                        chunks_[i].tags[s] = f14_detail::kTagEmpty;
                        chunks_[i].slots[s] = nullptr;
                    }
                }
                chunks_[i].occupied_mask = 0;
                // Clear overflow chain
                if constexpr (EmbeddedChain) {
                    Value curr = chunks_[i].embed_head.load(std::memory_order_acquire);
                    while (curr) {
                        Value next = static_cast<Value>(curr->hash_chain_next());
                        curr->set_hash_chain_next(nullptr);
                        curr = next;
                    }
                    chunks_[i].embed_head.store(nullptr, std::memory_order_release);
                } else {
                    node_type* node = chunks_[i].node_head.load(std::memory_order_acquire);
                    while (node) {
                        node_type* next = node->next;
                        deallocate_node(node);
                        node = next;
                    }
                    chunks_[i].node_head.store(nullptr, std::memory_order_release);
                }
                chunks_[i].version.fetch_add(1, std::memory_order_release);
            }
        } else {
            for (size_type i = 0; i <= mask; ++i) {
                buckets_[i].version.fetch_add(1, std::memory_order_release);
                if constexpr (EmbeddedChain) {
                    Value curr = buckets_[i].embed_head.load(std::memory_order_acquire);
                    while (curr) {
                        Value next = static_cast<Value>(curr->hash_chain_next());
                        curr->set_hash_chain_next(nullptr);
                        curr = next;
                    }
                    buckets_[i].embed_head.store(nullptr, std::memory_order_release);
                } else {
                    node_type* node = buckets_[i].node_head.load(std::memory_order_acquire);
                    while (node) {
                        node_type* next = node->next;
                        deallocate_node(node);
                        node = next;
                    }
                    buckets_[i].node_head.store(nullptr, std::memory_order_release);
                }
                buckets_[i].version.fetch_add(1, std::memory_order_release);
            }
        }
        size_.store(0, std::memory_order_relaxed);
        retired_buckets_.clear();
        retired_chunks_.clear();
    }

    void swap(concurrent_hash_table& other) noexcept {
        if constexpr (kIsF14) {
            chunks_.swap(other.chunks_);
        } else {
            buckets_.swap(other.buckets_);
        }
        size_type tmp_mask = bucket_mask_.load(std::memory_order_relaxed);
        bucket_mask_.store(other.bucket_mask_.load(std::memory_order_relaxed), std::memory_order_relaxed);
        other.bucket_mask_.store(tmp_mask, std::memory_order_relaxed);
        size_type tmp_size = size_.exchange(other.size_.load(std::memory_order_relaxed),
                                             std::memory_order_relaxed);
        other.size_.store(tmp_size, std::memory_order_relaxed);
        retired_buckets_.swap(other.retired_buckets_);
        retired_chunks_.swap(other.retired_chunks_);
        // Swap incremental rehash state
        std::swap(incremental_rehash_, other.incremental_rehash_);
        bool tmp_rip = rehash_in_progress_.load(std::memory_order_relaxed);
        rehash_in_progress_.store(other.rehash_in_progress_.load(std::memory_order_relaxed), std::memory_order_relaxed);
        other.rehash_in_progress_.store(tmp_rip, std::memory_order_relaxed);
        bucket_type* tmp_nb = rehash_new_buckets_.load(std::memory_order_relaxed);
        rehash_new_buckets_.store(other.rehash_new_buckets_.load(std::memory_order_relaxed), std::memory_order_relaxed);
        other.rehash_new_buckets_.store(tmp_nb, std::memory_order_relaxed);
        size_type tmp_nbc = rehash_new_bucket_count_.load(std::memory_order_relaxed);
        rehash_new_bucket_count_.store(other.rehash_new_bucket_count_.load(std::memory_order_relaxed), std::memory_order_relaxed);
        other.rehash_new_bucket_count_.store(tmp_nbc, std::memory_order_relaxed);
        size_type tmp_prog = rehash_progress_.load(std::memory_order_relaxed);
        rehash_progress_.store(other.rehash_progress_.load(std::memory_order_relaxed), std::memory_order_relaxed);
        other.rehash_progress_.store(tmp_prog, std::memory_order_relaxed);
        bool tmp_mig = rehash_migrating_.load(std::memory_order_relaxed);
        rehash_migrating_.store(other.rehash_migrating_.load(std::memory_order_relaxed), std::memory_order_relaxed);
        other.rehash_migrating_.store(tmp_mig, std::memory_order_relaxed);
        rehash_new_buckets_owner_.swap(other.rehash_new_buckets_owner_);

        // Swap F14 incremental rehash state
        f14_chunk_type* tmp_nc = rehash_new_chunks_.load(std::memory_order_relaxed);
        rehash_new_chunks_.store(other.rehash_new_chunks_.load(std::memory_order_relaxed), std::memory_order_relaxed);
        other.rehash_new_chunks_.store(tmp_nc, std::memory_order_relaxed);
        rehash_new_chunks_owner_.swap(other.rehash_new_chunks_owner_);
    }

    // ========================================================================
    // Load factor
    // ========================================================================

    float load_factor() const noexcept {
        auto bc = bucket_count();
        return bc == 0 ? 0.0f : static_cast<float>(size()) / static_cast<float>(bc);
    }

    float max_load_factor() const noexcept { return max_load_factor_; }
    void max_load_factor(float ml) noexcept { max_load_factor_ = ml; }

    // ========================================================================
    // Rehash
    // ========================================================================

    /// P0-5 (T1.3): hash-aware rehash entry point. For non-segmented
    /// tables this is equivalent to `rehash_if_needed()` — the hash is
    /// ignored. Provided so callers (insert/erase) can use a uniform
    /// `rehash_if_needed(hash)` call site for both segmented and
    /// non-segmented tables.
    void rehash_if_needed(size_type /*hash*/) {
        rehash_if_needed();
    }

    void rehash_if_needed() {
        // T13.3: Configurable overload threshold. Check this BEFORE the
        // max_load_factor_ early return so the application gets early
        // warning *before* rehash is strictly required. The default 2.0
        // matches the historical hardcoded warning threshold; users can
        // lower it (e.g. 1.5) for latency-sensitive workloads.
        //
        // The callback fires on every insert that observes overload —
        // callers should deduplicate inside the callback if they only
        // want one notification per overload episode.
        //
        // T13.4: The stderr warning is rate-limited to once per overload
        // episode (via hash_overload_warned_) to prevent log spam when
        // the table is persistently overloaded. The flag is cleared when
        // load_factor drops back below the threshold.
        //
        // T13.3 (revised): When load_factor exceeds the overload threshold,
        // trigger an emergency rehash immediately (rather than waiting until
        // max_load_factor_ is reached). This bounds the worst-case chain
        // length and tail latency under sustained insert pressure. The
        // emergency rehash respects the incremental_rehash_ setting — if
        // incremental rehash is enabled, the expansion proceeds incrementally
        // (no writer stall); otherwise it falls back to a blocking rehash.
        const float current_lf = load_factor();
        const float threshold = hash_overload_threshold_.load(std::memory_order_acquire);
        const bool overloaded = current_lf > threshold;
        if (overloaded) {
            hash_overload_events_.fetch_add(1, std::memory_order_relaxed);
            // P2-4 (T2.4): Async mode enqueues {current_lf, threshold} for
            // later draining by the event_drain_worker, so user callbacks
            // performing IO/logging cannot block the rehash hot path.
            // Sync mode preserves the original inline-invocation semantics.
            if (overload_callback_async_.load(std::memory_order_acquire)) {
                enqueue_overload_event(current_lf, threshold);
            } else if (hash_overload_callback_) {
                try {
                    hash_overload_callback_(current_lf, threshold);
                } catch (...) {
                    // Swallow user callback exceptions — they must not
                    // break the rehash path.
                }
            }
            // T13.4: Rate-limited stderr warning — only print once per
            // overload episode to avoid log spam under sustained overload.
            bool expected = false;
            if (hash_overload_warned_.compare_exchange_strong(
                    expected, true, std::memory_order_relaxed)) {
                std::cerr << "[LRU WARNING] Hash table load factor " << current_lf
                          << " exceeds overload threshold " << threshold
                          << ". Consider increasing expected_items for better performance.\n";
            }
        } else {
            // Clear the flag so the next overload episode warns again.
            hash_overload_warned_.store(false, std::memory_order_relaxed);
        }

        // If already rehashing incrementally, just make progress.
        if (incremental_rehash_ && rehash_in_progress_.load(std::memory_order_acquire)) {
            if constexpr (kIsF14) {
                rehash_step_f14(1);
            } else {
                rehash_step(1);
            }
            return;
        }

        // T13.3: Trigger emergency rehash when load_factor exceeds the
        // overload threshold (even if max_load_factor_ hasn't been reached
        // yet). This bounds chain length and tail latency.
        // Also trigger when load_factor exceeds max_load_factor_ (the
        // historical mandatory-rehash trigger).
        const bool mandatory_rehash = load_factor() > max_load_factor_;
        if (!overloaded && !mandatory_rehash) return;

        // T13.3: For emergency rehash, double the bucket count. For
        // mandatory rehash, the same doubling applies.
        rehash(bucket_count() * 2);
    }

    /// P2-4 (T2.4): Hot-path helper — pushes {current_lf, threshold} onto
    /// the async overload queue. Mutex hold time is bounded by a single
    /// `push_back` (amortized O(1)). To avoid unbounded memory growth
    /// under sustained overload (when the drain worker cannot keep up),
    /// we drop the oldest event if the queue is at capacity — the
    /// callback receives at most `kMaxOverloadQueue` events per drain
    /// cycle, which is sufficient for observability (rate-limited
    /// warning at line 3287 already deduplicates per-episode logging).
    void enqueue_overload_event(float current_lf, float threshold) {
        constexpr std::size_t kMaxOverloadQueue = 1024;
        std::lock_guard<std::mutex> lock(overload_queue_mutex_);
        if (overload_queue_.size() >= kMaxOverloadQueue) {
            overload_queue_.erase(overload_queue_.begin());
        }
        overload_queue_.emplace_back(current_lf, threshold);
    }

    /// T13.2: Register a callback invoked when load_factor exceeds the
    /// overload threshold. The callback receives (current_load_factor,
    /// threshold) and is invoked from the rehash hot path — it must be
    /// cheap and non-blocking. Exceptions thrown by the callback are
    /// swallowed to protect the rehash path.
    void set_overload_callback(std::function<void(float, float)> cb) {
        hash_overload_callback_ = std::move(cb);
    }

    /// P2-4 (T2.4): Toggle async mode for the overload callback.
    ///
    /// When enabled, the rehash hot path enqueues `{current_lf, threshold}`
    /// into a mutex-protected queue instead of invoking the callback inline.
    /// A separate worker (driven by `drain_overload_callbacks()`) consumes
    /// the queue and dispatches the callback off the hot path, so user
    /// callbacks performing IO/logging cannot stall inserts.
    ///
    /// When disabled (default), the callback fires inline on the rehash
    /// hot path — preserving the original T13.2 semantics. Use async mode
    /// whenever the registered callback may block (file/network IO,
    /// heap allocation, metrics push).
    void set_async_overload_callback(bool enabled) noexcept {
        overload_callback_async_.store(enabled, std::memory_order_release);
    }

    bool async_overload_callback_enabled() const noexcept {
        return overload_callback_async_.load(std::memory_order_acquire);
    }

    /// P2-4 (T2.4): Drain pending overload events and dispatch the
    /// registered callback for each. Designed to be called from a
    /// background worker (e.g. the event_drain_worker in `unified_cache`)
    /// rather than from a rehash hot path.
    ///
    /// Returns the number of events drained. Exceptions thrown by the
    /// user callback are swallowed to protect the drain worker.
    std::size_t drain_overload_callbacks() {
        std::vector<std::pair<float, float>> local;
        {
            std::lock_guard<std::mutex> lock(overload_queue_mutex_);
            if (overload_queue_.empty()) return 0;
            local.swap(overload_queue_);
        }
        if (!hash_overload_callback_) return local.size();
        for (const auto& [lf, th] : local) {
            try {
                hash_overload_callback_(lf, th);
            } catch (...) {
                // Swallow — drain worker must survive bad callbacks.
            }
        }
        return local.size();
    }

    /// P2-4 (T2.4): Number of overload events currently buffered in the
    /// async queue (best-effort snapshot, no lock).
    std::size_t pending_overload_events() const {
        std::lock_guard<std::mutex> lock(overload_queue_mutex_);
        return overload_queue_.size();
    }

    /// T13.1: Set the load factor threshold above which the overload
    /// callback fires and hash_overload_events is incremented.
    /// Default: 2.0 (matches historical hardcoded warning threshold).
    /// Lower for latency-sensitive workloads, raise for memory-frugal.
    void set_hash_overload_threshold(float threshold) noexcept {
        if (threshold <= 0.0f) threshold = 2.0f;  // sanity guard
        hash_overload_threshold_.store(threshold, std::memory_order_release);
    }

    float hash_overload_threshold() const noexcept {
        return hash_overload_threshold_.load(std::memory_order_acquire);
    }

    std::size_t hash_overload_events() const noexcept {
        return hash_overload_events_.load(std::memory_order_relaxed);
    }

    void rehash(size_type new_bucket_count) {
        new_bucket_count = next_power_of_two(new_bucket_count);
        if (new_bucket_count <= bucket_count()) return;

        // P1-1: Record rehash start time and item count for diagnostics.
        const auto rehash_t0 = std::chrono::steady_clock::now();
        const std::size_t items_before = size_.load(std::memory_order_relaxed);
        rehash_count_.fetch_add(1, std::memory_order_relaxed);

        if constexpr (kIsF14) {
            if (incremental_rehash_) {
                // Incremental rehash: allocate new array, set state, don't block.
                // Progress is made incrementally by subsequent operations via
                // rehash_step_f14() (invoked from rehash_if_needed, insert, erase).
                rehash_begin_f14(new_bucket_count);
                // For incremental rehash, record the setup time (actual migration
                // happens incrementally; total migration time is amortized).
                rehash_migrated_items_.fetch_add(items_before, std::memory_order_relaxed);
                const auto elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - rehash_t0).count();
                rehash_total_time_ns_.fetch_add(static_cast<std::uint64_t>(elapsed_ns),
                                                std::memory_order_relaxed);
                return;
            }
            // T11.3: Blocking rehash — every concurrent writer will stall on
            // the chunk locks acquired below. Count this so users can detect
            // that they should enable incremental rehash.
            rehash_blocked_writes_count_.fetch_add(1, std::memory_order_relaxed);
            // Blocking rehash: acquire ALL chunk exclusive locks
            std::vector<exclusive_bucket_lock> locks;
            locks.reserve(bucket_count());
            const size_type old_mask = bucket_mask_.load(std::memory_order_relaxed);
            for (size_type i = 0; i <= old_mask; ++i) {
                locks.emplace_back(chunks_[i].spin);
            }

            seqlock_.fetch_add(1, std::memory_order_release);

            auto new_chunks = std::make_unique<f14_chunk_type[]>(new_bucket_count);
            size_type new_mask = new_bucket_count - 1;

            if constexpr (EmbeddedChain) {
                for (size_type i = 0; i <= old_mask; ++i) {
                    auto& old_chunk = chunks_[i];
                    // Redistribute inline slots
                    for (int s = 0; s < f14_detail::kChunkCapacity; ++s) {
                        if (old_chunk.occupied_mask & (1u << s)) {
                            Value node = static_cast<Value>(old_chunk.slots[s]);
                            size_type new_idx = node->cached_hash() & new_mask;
                            auto& nc = new_chunks[new_idx];
                            int new_slot = find_empty_slot_in_chunk(nc);
                            if (new_slot >= 0) {
                                nc.tags[new_slot] = old_chunk.tags[s];
                                nc.slots[new_slot] = node;
                                nc.occupied_mask |= static_cast<uint16_t>(1u << new_slot);
                            } else {
                                node->set_hash_chain_next(nc.embed_head.load(std::memory_order_acquire));
                                nc.embed_head.store(node, std::memory_order_release);
                            }
                            old_chunk.slots[s] = nullptr;
                        }
                        old_chunk.tags[s] = f14_detail::kTagEmpty;
                    }
                    old_chunk.occupied_mask = 0;
                    // Redistribute overflow chain
                    Value curr = old_chunk.embed_head.load(std::memory_order_acquire);
                    while (curr) {
                        Value next = static_cast<Value>(curr->hash_chain_next());
                        size_type new_idx = curr->cached_hash() & new_mask;
                        auto& nc = new_chunks[new_idx];
                        int new_slot = find_empty_slot_in_chunk(nc);
                        if (new_slot >= 0) {
                            nc.tags[new_slot] = f14_detail::f14_tag(curr->cached_hash());
                            nc.slots[new_slot] = curr;
                            nc.occupied_mask |= static_cast<uint16_t>(1u << new_slot);
                            curr->set_hash_chain_next(nullptr);
                        } else {
                            curr->set_hash_chain_next(nc.embed_head.load(std::memory_order_acquire));
                            nc.embed_head.store(curr, std::memory_order_release);
                        }
                        curr = next;
                    }
                    old_chunk.embed_head.store(nullptr, std::memory_order_release);
                }
            } else {
                for (size_type i = 0; i <= old_mask; ++i) {
                    auto& old_chunk = chunks_[i];
                    for (int s = 0; s < f14_detail::kChunkCapacity; ++s) {
                        if (old_chunk.occupied_mask & (1u << s)) {
                            node_type* node = static_cast<node_type*>(old_chunk.slots[s]);
                            size_type new_idx = node->hash & new_mask;
                            auto& nc = new_chunks[new_idx];
                            int new_slot = find_empty_slot_in_chunk(nc);
                            if (new_slot >= 0) {
                                nc.tags[new_slot] = old_chunk.tags[s];
                                nc.slots[new_slot] = node;
                                nc.occupied_mask |= static_cast<uint16_t>(1u << new_slot);
                            } else {
                                node->next = nc.node_head.load(std::memory_order_acquire);
                                nc.node_head.store(node, std::memory_order_release);
                            }
                            old_chunk.slots[s] = nullptr;
                        }
                        old_chunk.tags[s] = f14_detail::kTagEmpty;
                    }
                    old_chunk.occupied_mask = 0;
                    node_type* node = old_chunk.node_head.load(std::memory_order_acquire);
                    while (node) {
                        node_type* next = node->next;
                        size_type new_idx = node->hash & new_mask;
                        auto& nc = new_chunks[new_idx];
                        int new_slot = find_empty_slot_in_chunk(nc);
                        if (new_slot >= 0) {
                            nc.tags[new_slot] = f14_detail::f14_tag(node->hash);
                            nc.slots[new_slot] = node;
                            nc.occupied_mask |= static_cast<uint16_t>(1u << new_slot);
                            node->next = nullptr;
                        } else {
                            node->next = nc.node_head.load(std::memory_order_acquire);
                            nc.node_head.store(node, std::memory_order_release);
                        }
                        node = next;
                    }
                    old_chunk.node_head.store(nullptr, std::memory_order_release);
                }
            }

            // P3-1: Swap chunks_ atomically (never null) before publishing the
            // new bucket_mask_. The previous code did:
            //   retired_chunks_.push_back(std::move(chunks_));  // chunks_ = null
            //   bucket_mask_.store(new_mask, release);
            //   chunks_ = std::move(new_chunks);
            // This left a window where chunks_ was null AND bucket_mask_ had
            // already been updated, so a concurrent reader in
            // find_and_pin_with_hash (which loads bucket_mask_ then accesses
            // chunks_[idx]) would dereference a null unique_ptr, triggering
            // UBSan "applying non-zero offset to null pointer".
            //
            // The fix:
            //   1. swap chunks_ with new_chunks (chunks_ is never null)
            //   2. publish new bucket_mask_ (release — pairs with reader's acquire)
            //   3. retire the old array (now held in new_chunks)
            //
            // Safety for readers: the writer updates chunks_ BEFORE bucket_mask_,
            // and uses release on bucket_mask_. A reader using acquire on
            // bucket_mask_ that sees the new mask is guaranteed to see the new
            // chunks_. A reader that sees the old mask computes an idx in the
            // old (smaller) range — also valid in the new (larger) chunks_, so
            // chunks_[old_idx] is in bounds. Rehash only ever grows the array.
            chunks_.swap(new_chunks);
            bucket_mask_.store(new_mask, std::memory_order_release);
            retired_chunks_.clear();
            retired_chunks_.push_back(std::move(new_chunks));
            seqlock_.fetch_add(1, std::memory_order_release);
        } else {
            // Chain mode rehash
            if (incremental_rehash_) {
                // C-1 / Defect C fix: Atomically claim the rehash start via
                // CAS to prevent two threads from both allocating a new array
                // and racing on rehash_new_buckets_owner_ (the second would
                // std::move over the first, leaving rehash_new_buckets_
                // dangling). Under segmented+sharded MM the per-shard lock
                // serializes same-segment writers, but defensive CAS is
                // correct for all configurations.
                bool exp_rip = false;
                if (!rehash_in_progress_.compare_exchange_strong(exp_rip, true,
                        std::memory_order_acq_rel)) {
                    return; // Another thread already started rehash
                }
                // Incremental rehash: allocate new array, set state, don't block
                auto new_buckets = std::make_unique<bucket_type[]>(new_bucket_count);

                rehash_new_buckets_owner_ = std::move(new_buckets);
                rehash_new_buckets_.store(rehash_new_buckets_owner_.get(), std::memory_order_release);
                rehash_new_bucket_count_.store(new_bucket_count, std::memory_order_release);
                rehash_progress_.store(0, std::memory_order_release);
                // rehash_in_progress_ already set true by CAS above
                seqlock_.fetch_add(1, std::memory_order_release);
            } else {
                // T11.3: Blocking rehash — every concurrent writer will stall
                // on the bucket locks acquired below. Count this so users can
                // detect that they should enable incremental rehash.
                rehash_blocked_writes_count_.fetch_add(1, std::memory_order_relaxed);
                // Blocking rehash (original behavior)
                std::vector<exclusive_bucket_lock> locks;
                locks.reserve(bucket_count());
                const size_type old_mask = bucket_mask_.load(std::memory_order_relaxed);
                for (size_type i = 0; i <= old_mask; ++i) {
                    locks.emplace_back(buckets_[i].spin);
                }

                seqlock_.fetch_add(1, std::memory_order_release);

                auto new_buckets = std::make_unique<bucket_type[]>(new_bucket_count);
                size_type new_mask = new_bucket_count - 1;

                if constexpr (EmbeddedChain) {
                    for (size_type i = 0; i <= old_mask; ++i) {
                        Value curr = buckets_[i].embed_head.load(std::memory_order_acquire);
                        while (curr) {
                            Value next = static_cast<Value>(curr->hash_chain_next());
                            size_type new_idx = curr->cached_hash() & new_mask;
                            curr->set_hash_chain_next(new_buckets[new_idx].embed_head.load(std::memory_order_acquire));
                            new_buckets[new_idx].embed_head.store(curr, std::memory_order_release);
                            curr = next;
                        }
                        buckets_[i].embed_head.store(nullptr, std::memory_order_release);
                    }
                } else {
                    for (size_type i = 0; i <= old_mask; ++i) {
                        node_type* node = buckets_[i].node_head.load(std::memory_order_acquire);
                        while (node) {
                            node_type* next = node->next;
                            size_type new_idx = node->hash & new_mask;
                            node->next = new_buckets[new_idx].node_head.load(std::memory_order_acquire);
                            new_buckets[new_idx].node_head.store(node, std::memory_order_release);
                            node = next;
                        }
                        buckets_[i].node_head.store(nullptr, std::memory_order_release);
                    }
                }

                // P3-1: Swap buckets_ atomically (never null) before publishing
                // the new bucket_mask_. See the F14 path above for the full
                // rationale — the same TOCTOU on `buckets_ == null` between
                // `std::move(buckets_)` and `buckets_ = std::move(new_buckets)`
                // would let a concurrent reader dereference a null unique_ptr.
                buckets_.swap(new_buckets);
                bucket_mask_.store(new_mask, std::memory_order_release);
                retired_buckets_.clear();
                retired_buckets_.push_back(std::move(new_buckets));
                seqlock_.fetch_add(1, std::memory_order_release);
            }
        }

        // P1-1: Record rehash diagnostics for blocking rehash paths.
        // (Incremental rehash paths record stats at their begin_* entry
        // points above.) We record migrated items and total duration here
        // to capture the full blocking cost.
        rehash_migrated_items_.fetch_add(items_before, std::memory_order_relaxed);
        const auto elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - rehash_t0).count();
        rehash_total_time_ns_.fetch_add(static_cast<std::uint64_t>(elapsed_ns),
                                        std::memory_order_relaxed);
    }

    // ========================================================================
    // Chain length diagnostics
    // ========================================================================

    size_type max_chain_length() const noexcept {
        size_type max_len = 0;
        const size_type mask = bucket_mask_.load(std::memory_order_relaxed);
        if constexpr (kIsF14) {
            for (size_type i = 0; i <= mask; ++i) {
                size_type len = 0;
                // Count inline slots
                len += static_cast<size_type>(
                    f14_detail::popcount16(chunks_[i].occupied_mask));
                // Count overflow chain
                if constexpr (EmbeddedChain) {
                    Value curr = chunks_[i].embed_head.load(std::memory_order_acquire);
                    while (curr) { ++len; curr = static_cast<Value>(curr->hash_chain_next()); }
                } else {
                    node_type* node = chunks_[i].node_head.load(std::memory_order_acquire);
                    while (node) { ++len; node = node->next; }
                }
                if (len > max_len) max_len = len;
            }
        } else {
            for (size_type i = 0; i <= mask; ++i) {
                size_type len = 0;
                if constexpr (EmbeddedChain) {
                    Value curr = buckets_[i].embed_head.load(std::memory_order_acquire);
                    while (curr) { ++len; curr = static_cast<Value>(curr->hash_chain_next()); }
                } else {
                    node_type* node = buckets_[i].node_head.load(std::memory_order_acquire);
                    while (node) { ++len; node = node->next; }
                }
                if (len > max_len) max_len = len;
            }
        }
        return max_len;
    }

    // ========================================================================
    // Custom allocation
    // ========================================================================

    void set_alloc_fns(allocate_fn alloc_fn, deallocate_fn dealloc_fn) noexcept {
        alloc_fn_ = alloc_fn; dealloc_fn_ = dealloc_fn;
    }
    allocate_fn get_alloc_fn() const noexcept { return alloc_fn_; }
    deallocate_fn get_dealloc_fn() const noexcept { return dealloc_fn_; }

    // ========================================================================
    // Optimistic read configuration
    // ========================================================================

    /// Enable/disable optimistic read (lock-free, version-based).
    ///
    /// P0-1 Safety Note:
    ///   - EmbeddedChain = true: 乐观读是安全的（节点由 refcount + hazptr 保护）。
    ///     默认启用，可获得最高读吞吐量。
    ///   - EmbeddedChain = false: 乐观读存在 use-after-free 风险。node_type 独立
    ///     分配且无 refcount 保护，另一个线程可能在 TOCTOU 窗口内删除节点。
    ///     默认禁用。仅在以下所有条件满足时才可安全启用：
    ///       1. 调用方保证不会在 find() 返回后持有指针跨线程切换
    ///       2. 调用方使用 find_and_pin() 在锁内完成 refcount 递增
    ///       3. 或调用方接受 UAF 风险（如只读工作负载，无并发 erase）
    void set_optimistic_read(bool enabled) noexcept {
        // P0-1: 非 EmbeddedChain 模式下启用乐观读需要用户显式确认风险。
        // 这里不阻止启用（保持向后兼容），但在文档中明确警告。
        enable_optimistic_read_ = enabled;
    }
    bool optimistic_read_enabled() const noexcept { return enable_optimistic_read_; }

    /// P0-1: 查询当前模式是否为安全乐观读模式（EmbeddedChain + 乐观读）。
    /// 生产环境推荐在 safe_optimistic_read() == true 时才依赖乐观读路径。
    bool safe_optimistic_read() const noexcept {
        return enable_optimistic_read_ && EmbeddedChain;
    }

    // --------------------------------------------------------------------
    // T2.1 / T2.4: EBR (Epoch-Based Reclamation) integration
    // --------------------------------------------------------------------

    /// T2.1: Set the EBR domain for this hash table. When non-null,
    /// find_and_pin_lockfree*() acquires an epoch_guard at entry,
    /// protecting all nodes from reclamation during traversal. When
    /// null (default), the hash table operates in hazptr mode.
    ///
    /// The domain must outlive this hash table. The caller is also
    /// responsible for ensuring the domain's retire() is used for
    /// evicted items (the MM layer handles this via its own
    /// set_ebr_domain()).
    void set_ebr_domain(detail::epoch_domain* domain) noexcept {
        ebr_domain_ = domain;
    }

    detail::epoch_domain* get_ebr_domain() const noexcept { return ebr_domain_; }

    /// T2.4: Check whether this hash table is operating in EBR mode
    /// (i.e., an EBR domain has been set). In EBR mode, the read path
    /// is protected by an epoch_guard acquired at find_and_pin_lockfree
    /// entry; the per-pointer hazptr_holder calls inside the traversal
    /// become redundant (but harmless — they just add a small overhead).
    bool is_ebr_mode() const noexcept { return ebr_domain_ != nullptr; }

    /// T2.4: Unified guard type for read-path protection. Resolves to
    /// `epoch_guard` in EBR mode (protects all nodes in the critical
    /// section) and to a no-op guard in hazptr mode (per-pointer
    /// protection is handled by hazptr_holder inside the traversal).
    ///
    /// Usage:
    ///   detail::epoch_domain* d = table.get_ebr_domain();
    ///   typename concurrent_hash_table::reclaim_guard guard(d);
    ///   // ... traverse hash table safely ...
    ///
    /// This allows callers to write mode-agnostic code without #ifdefs.
    class reclaim_guard {
    public:
        /// Constructs a guard. If `domain` is non-null, acquires an
        /// epoch critical section; otherwise is a no-op.
        explicit reclaim_guard(detail::epoch_domain* domain)
            : domain_(domain) {
            if (domain_) {
                domain_->enter_critical();
                active_ = true;
            }
        }

        ~reclaim_guard() {
            if (active_ && domain_) {
                domain_->exit_critical();
            }
        }

        // Non-copyable
        reclaim_guard(const reclaim_guard&) = delete;
        reclaim_guard& operator=(const reclaim_guard&) = delete;

        // Movable
        reclaim_guard(reclaim_guard&& other) noexcept
            : domain_(other.domain_), active_(other.active_) {
            other.active_ = false;
            other.domain_ = nullptr;
        }
        reclaim_guard& operator=(reclaim_guard&& other) noexcept {
            if (this != &other) {
                if (active_ && domain_) domain_->exit_critical();
                domain_ = other.domain_;
                active_ = other.active_;
                other.active_ = false;
                other.domain_ = nullptr;
            }
            return *this;
        }

        bool is_active() const noexcept { return active_; }

    private:
        detail::epoch_domain* domain_;
        bool active_ = false;
    };

    void set_hazptr_read(bool enabled) noexcept { enable_hazptr_read_ = enabled; }
    bool hazptr_read_enabled() const noexcept { return enable_hazptr_read_; }

    // ========================================================================
    // P1-1: Rehash diagnostics accessors
    // ========================================================================

    std::size_t rehash_count() const noexcept {
        return rehash_count_.load(std::memory_order_relaxed);
    }
    std::uint64_t rehash_total_time_ns() const noexcept {
        return rehash_total_time_ns_.load(std::memory_order_relaxed);
    }
    std::size_t rehash_migrated_items() const noexcept {
        return rehash_migrated_items_.load(std::memory_order_relaxed);
    }
    /// T11.3: Number of writes blocked by a non-incremental (blocking) rehash.
    /// Non-zero values indicate the user should enable incremental rehash to
    /// avoid stalling writers during hash table expansion.
    std::size_t rehash_blocked_writes_count() const noexcept {
        return rehash_blocked_writes_count_.load(std::memory_order_relaxed);
    }

    /// P0-4 (T1.2): Number of times rehash_finish() returned early because
    /// the per-call migration budget (kRehashFinishMaxBucketsPerCall) was
    /// exhausted before all buckets/chunks could be migrated. Non-zero
    /// values indicate the workload is producing large rehash bursts that
    /// outpace inline migration; in this case, rely on the background
    /// rehash balancer (or raise the per-call budget for larger batches).
    std::size_t rehash_finish_stall_count() const noexcept {
        return rehash_finish_stall_count_.load(std::memory_order_relaxed);
    }
    /// P0-4 (T1.2): High-water mark of remaining buckets/chunks at the
    /// moment a stalled rehash_finish() returned. Useful for sizing the
    /// per-call budget: if this stays above zero under load, the budget
    /// is too small for the workload's rehash volume.
    std::size_t rehash_finish_max_backlog() const noexcept {
        return rehash_finish_max_backlog_.load(std::memory_order_relaxed);
    }

    /// P1-5: Number of times find_and_pin_lockfree fell back to the
    /// lock-protected path because the segment was in incremental rehash.
    /// Non-zero values indicate the lock-free read path is being degraded
    /// by rehash activity — operators should consider pre-reserving capacity
    /// (`reserve()`) or raising the per-call rehash budget.
    std::size_t rehash_lockfree_fallback_count() const noexcept {
        return rehash_lockfree_fallback_count_.load(std::memory_order_relaxed);
    }

    // ========================================================================
    // Reserve — pre-allocate buckets to avoid runtime rehash
    // ========================================================================

    /// Pre-allocate enough buckets for `expected_items` entries without
    /// triggering a runtime rehash.  If the current bucket count is already
    /// sufficient, this is a no-op.  Otherwise it performs a one-time
    /// rehash to the target bucket count (computed via buckets_for_items).
    ///
    /// Thread-safety: safe to call concurrently with reads; the internal
    /// rehash uses the same seqlock-based protocol as automatic rehash.
    void reserve(size_type expected_items) {
        auto target = buckets_for_items(expected_items);
        auto current = bucket_count();
        if (target <= current) return;
        rehash(target);
    }

    // ========================================================================
    // Incremental rehash configuration (chain mode only)
    // ========================================================================

    /// Enable or disable incremental (progressive) rehash.
    /// When enabled, rehash() starts an incremental migration instead of
    /// blocking all readers. Only effective for chain mode (non-F14).
    ///
    /// T19.2: Effective for ALL hash table modes:
    ///   - **Chain mode** (`chain_probing_tag`): migrates one bucket at a time
    ///     via rehash_step().
    ///   - **F14 SIMD mode** (`f14_probing_tag`): migrates one 14-slot chunk
    ///     at a time via rehash_step_f14() (dual-array lookup).
    ///   - **Segmented mode** (`Segmented=true`): applies chain-mode incremental
    ///     rehash independently per segment (1/64 stall at any moment).
    void set_incremental_rehash(bool enabled) noexcept { incremental_rehash_ = enabled; }
    bool incremental_rehash_enabled() const noexcept { return incremental_rehash_; }

    /// T11.5: Set the rehash strategy by name. String-based API for
    /// configuration files / CLI flags. Equivalent to:
    ///   - "incremental": set_incremental_rehash(true)
    ///   - "blocking":    set_incremental_rehash(false)
    /// Any other value falls back to "blocking" (the historical default)
    /// and is reported via the returned bool (false = unrecognized).
    /// Case-insensitive.
    bool set_rehash_strategy(std::string_view strategy) noexcept {
        // Case-insensitive comparison.
        auto to_lower = [](char c) noexcept -> char {
            return (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c;
        };
        std::string lower;
        lower.reserve(strategy.size());
        for (char c : strategy) lower.push_back(to_lower(c));

        if (lower == "incremental") {
            set_incremental_rehash(true);
            return true;
        }
        if (lower == "blocking") {
            set_incremental_rehash(false);
            return true;
        }
        // Unknown strategy — leave the current setting untouched.
        return false;
    }

    /// T11.5: Query the current rehash strategy by name.
    std::string_view rehash_strategy() const noexcept {
        return incremental_rehash_ ? std::string_view{"incremental"}
                                   : std::string_view{"blocking"};
    }

    /// Returns true if an incremental rehash is currently in progress.
    bool is_rehashing() const noexcept {
        return rehash_in_progress_.load(std::memory_order_acquire);
    }

    /// P0-D: Ratio of the hash table currently in an incremental rehash.
    /// For non-segmented tables: 0.0f (idle) or 1.0f (rehashing). The
    /// segmented_concurrent_hash_table overrides this to return the
    /// fraction of its 64 segments currently rehashing.
    float rehash_in_progress_ratio() const noexcept {
        return rehash_in_progress_.load(std::memory_order_acquire) ? 1.0f : 0.0f;
    }

    /// Number of old buckets fully migrated so far during an incremental
    /// rehash. Returns 0 when no rehash is in progress. Always 0 for F14
    /// mode (F14 uses chunk-based migration; see rehash_progress_f14()).
    size_type rehash_progress() const noexcept {
        if (!rehash_in_progress_.load(std::memory_order_acquire)) return 0;
        return rehash_progress_.load(std::memory_order_acquire);
    }

    /// Number of buckets in the new array being migrated to. Returns 0
    /// when no incremental rehash is in progress.
    size_type rehash_new_bucket_count() const noexcept {
        if (!rehash_in_progress_.load(std::memory_order_acquire)) return 0;
        return rehash_new_bucket_count_.load(std::memory_order_acquire);
    }

    /// Total number of buckets in the old (current) array. Combined with
    /// rehash_progress(), this gives the migration completion fraction.
    size_type rehash_old_bucket_count() const noexcept {
        if (!rehash_in_progress_.load(std::memory_order_acquire)) return 0;
        return bucket_mask_.load(std::memory_order_relaxed) + 1;
    }

    /// Perform one step of incremental rehash, migrating up to batch_size
    /// buckets from the old array to the new array. Returns the number of
    /// buckets actually migrated. Only meaningful for chain mode (non-F14).
    size_type rehash_step(size_type batch_size = 16) {
        if constexpr (kIsF14) {
            return rehash_step_f14(batch_size);
        }
        if (!rehash_in_progress_.load(std::memory_order_acquire)) return 0;

        // Only one thread migrates at a time to avoid out-of-order progress
        bool expected = false;
        if (!rehash_migrating_.compare_exchange_strong(expected, true,
                std::memory_order_acq_rel)) {
            return 0;
        }

        const size_type old_bucket_count = bucket_mask_.load(std::memory_order_relaxed) + 1;
        const size_type new_mask = rehash_new_bucket_count_.load(std::memory_order_relaxed) - 1;
        bucket_type* new_bkts = rehash_new_buckets_.load(std::memory_order_acquire);
        size_type migrated = 0;

        while (migrated < batch_size) {
            size_type current = rehash_progress_.load(std::memory_order_acquire);
            if (current >= old_bucket_count) break;

            // Migrate bucket 'current' under exclusive lock
            exclusive_bucket_lock old_guard(buckets_[current].spin);

            if constexpr (EmbeddedChain) {
                Value curr = buckets_[current].embed_head.load(std::memory_order_acquire);
                while (curr) {
                    Value next = static_cast<Value>(curr->hash_chain_next());
                    size_type new_idx = curr->cached_hash() & new_mask;
                    exclusive_bucket_lock new_guard(new_bkts[new_idx].spin);
                    curr->set_hash_chain_next(new_bkts[new_idx].embed_head.load(std::memory_order_acquire));
                    new_bkts[new_idx].embed_head.store(curr, std::memory_order_release);
                    new_guard.unlock();
                    curr = next;
                }
                buckets_[current].embed_head.store(nullptr, std::memory_order_release);
            } else {
                node_type* node = buckets_[current].node_head.load(std::memory_order_acquire);
                while (node) {
                    node_type* next_node = node->next;
                    size_type new_idx = node->hash & new_mask;
                    exclusive_bucket_lock new_guard(new_bkts[new_idx].spin);
                    node->next = new_bkts[new_idx].node_head.load(std::memory_order_acquire);
                    new_bkts[new_idx].node_head.store(node, std::memory_order_release);
                    new_guard.unlock();
                    node = next_node;
                }
                buckets_[current].node_head.store(nullptr, std::memory_order_release);
            }

            // Increment version to invalidate optimistic reads of old bucket
            buckets_[current].version.fetch_add(2, std::memory_order_release);

            // Advance progress inside the lock so find() sees consistent state
            rehash_progress_.store(current + 1, std::memory_order_release);

            ++migrated;
        }

        rehash_migrating_.store(false, std::memory_order_release);

        // Auto-complete if all buckets are migrated
        if (rehash_progress_.load(std::memory_order_acquire) >= old_bucket_count) {
            rehash_finish();
        }

        return migrated;
    }

    /// Complete any ongoing incremental rehash, migrating all remaining
    /// buckets and installing the new array.
    ///
    /// P0-4 (T1.2): To bound the worst-case stall, we cap the number of
    /// buckets migrated in a single rehash_finish() call at
    /// `kRehashFinishMaxBucketsPerCall`. If more work remains, we leave
    /// the rehash in progress and return without installing the new
    /// array; subsequent writes (or the background rehash balancer)
    /// will call rehash_finish() again to advance migration. This
    /// eliminates the P99 disaster scenario where one unlucky insert
    /// synchronously migrates thousands of buckets while holding the
    /// rehash_migrating_ exclusive flag.
    void rehash_finish() {
        if constexpr (kIsF14) {
            rehash_finish_f14();
            return;
        }
        if (!rehash_in_progress_.load(std::memory_order_acquire)) return;

        const size_type old_bucket_count = bucket_mask_.load(std::memory_order_relaxed) + 1;

        // P0-4 (T1.2): Cap synchronous migration to bound tail latency.
        // Migrate at most kRehashFinishMaxBucketsPerCall buckets per call;
        // if more remain, leave rehash in progress and let subsequent
        // writes / the background rehash balancer make further progress.
        size_type migrated_this_call = 0;
        while (rehash_progress_.load(std::memory_order_acquire) < old_bucket_count) {
            if (migrated_this_call >= kRehashFinishMaxBucketsPerCall) {
                // Out of budget — record stall and return without finishing.
                rehash_finish_stall_count_.fetch_add(1, std::memory_order_relaxed);
                std::size_t backlog = old_bucket_count - rehash_progress_.load(std::memory_order_relaxed);
                std::size_t prev_max = rehash_finish_max_backlog_.load(std::memory_order_relaxed);
                while (backlog > prev_max && !rehash_finish_max_backlog_.compare_exchange_weak(
                    prev_max, backlog, std::memory_order_relaxed)) {
                    // retry with updated prev_max
                }
                return;
            }
            rehash_step(16);
            migrated_this_call += 16;
        }

        // All buckets migrated — atomically claim the finish. Only the first
        // thread past this point proceeds to install the new array; concurrent
        // callers (e.g. the background rehash balancer calling
        // advance_incremental_rehash() -> rehash_finish() without holding the
        // per-shard lock) return early. This mirrors the F14 path's CAS claim
        // in rehash_finish_f14(). Without this guard, two threads could
        // simultaneously execute std::move(buckets_) below, causing
        // double-free / use-after-free. (C-1 / Defect B fix.)
        bool expected = true;
        if (!rehash_in_progress_.compare_exchange_strong(expected, false,
                std::memory_order_acq_rel)) {
            return; // Another thread already finished
        }

        // We've claimed the finish. Install the new array.
        seqlock_.fetch_add(1, std::memory_order_release);

        const size_type new_mask = rehash_new_bucket_count_.load(std::memory_order_relaxed) - 1;

        // P3-1: Swap buckets_ atomically (never null) before publishing the
        // new bucket_mask_. See the blocking-rehash path above for rationale.
        buckets_.swap(rehash_new_buckets_owner_);
        bucket_mask_.store(new_mask, std::memory_order_release);
        retired_buckets_.clear();
        retired_buckets_.push_back(std::move(rehash_new_buckets_owner_));

        // Clear remaining incremental rehash state (in_progress already cleared by CAS)
        rehash_new_buckets_.store(nullptr, std::memory_order_release);
        rehash_new_bucket_count_.store(0, std::memory_order_release);
        rehash_progress_.store(0, std::memory_order_release);

        seqlock_.fetch_add(1, std::memory_order_release);
    }

    // ========================================================================
    // Incremental rehash (F14 mode)
    // ========================================================================
    //
    // Mirrors the chain-mode incremental rehash but operates on f14_chunk_type
    // arrays. Migration proceeds one old chunk at a time: the migrator acquires
    // the exclusive lock on the source old chunk and, for each item, the
    // exclusive lock on the destination new chunk. This bounds the critical
    // section to two chunks, eliminating the global stall of blocking rehash.
    //
    // Concurrent find()/insert()/erase() during migration use a dual-array
    // protocol keyed on rehash_progress_:
    //   - old_idx < progress  -> item lives in the new array
    //   - old_idx >= progress -> item lives in the old array (under its lock)
    // Insertions during migration target the array owning the key's bucket.

    /// Begin incremental rehash for F14 mode.
    /// Allocates the new chunks array and sets up rehash state. Returns true
    /// if the rehash was started, false if it was skipped (already in progress
    /// or no growth needed).
    bool rehash_begin_f14(size_type new_chunk_count) {
        if constexpr (!kIsF14) return false;
        new_chunk_count = next_power_of_two(new_chunk_count);
        if (new_chunk_count <= bucket_count()) return false;
        // C-1 / Defect C fix: CAS guard (was load-check, which had a TOCTOU
        // window between the load and the store at the end).
        bool expected = false;
        if (!rehash_in_progress_.compare_exchange_strong(expected, true,
                std::memory_order_acq_rel)) {
            return false;
        }

        auto new_chunks = std::make_unique<f14_chunk_type[]>(new_chunk_count);

        seqlock_.fetch_add(1, std::memory_order_release);

        rehash_new_chunks_owner_ = std::move(new_chunks);
        rehash_new_chunks_.store(rehash_new_chunks_owner_.get(), std::memory_order_release);
        rehash_new_bucket_count_.store(new_chunk_count, std::memory_order_release);
        rehash_progress_.store(0, std::memory_order_release);
        // rehash_in_progress_ already set true by CAS above

        seqlock_.fetch_add(1, std::memory_order_release);
        return true;
    }

    /// Perform one step of F14 incremental rehash, migrating up to num_chunks
    /// chunks from the old array to the new array. Returns the number of
    /// chunks actually migrated. Each migration holds at most two chunk
    /// exclusive locks (source + destination), bounding the critical section.
    size_type rehash_step_f14(size_type num_chunks) {
        if constexpr (!kIsF14) return 0;
        if (!rehash_in_progress_.load(std::memory_order_acquire)) return 0;

        // Only one thread migrates at a time to avoid out-of-order progress
        bool expected = false;
        if (!rehash_migrating_.compare_exchange_strong(expected, true,
                std::memory_order_acq_rel)) {
            return 0;
        }

        const size_type old_chunk_count = bucket_mask_.load(std::memory_order_relaxed) + 1;
        const size_type new_mask = rehash_new_bucket_count_.load(std::memory_order_relaxed) - 1;
        f14_chunk_type* new_chunks = rehash_new_chunks_.load(std::memory_order_acquire);
        size_type migrated = 0;

        while (migrated < num_chunks) {
            size_type current = rehash_progress_.load(std::memory_order_acquire);
            if (current >= old_chunk_count) break;

            // Migrate chunk 'current' under exclusive lock
            exclusive_bucket_lock old_guard(chunks_[current].spin);

            if constexpr (EmbeddedChain) {
                // Redistribute inline slots
                for (int s = 0; s < f14_detail::kChunkCapacity; ++s) {
                    if (chunks_[current].occupied_mask & (1u << s)) {
                        Value node = static_cast<Value>(chunks_[current].slots[s]);
                        size_type new_idx = node->cached_hash() & new_mask;
                        exclusive_bucket_lock new_guard(new_chunks[new_idx].spin);
                        int new_slot = find_empty_slot_in_chunk(new_chunks[new_idx]);
                        if (new_slot >= 0) {
                            new_chunks[new_idx].tags[new_slot] = chunks_[current].tags[s];
                            new_chunks[new_idx].slots[new_slot] = node;
                            new_chunks[new_idx].occupied_mask |= static_cast<uint16_t>(1u << new_slot);
                        } else {
                            node->set_hash_chain_next(new_chunks[new_idx].embed_head.load(std::memory_order_acquire));
                            new_chunks[new_idx].embed_head.store(node, std::memory_order_release);
                        }
                        chunks_[current].slots[s] = nullptr;
                    }
                    chunks_[current].tags[s] = f14_detail::kTagEmpty;
                }
                chunks_[current].occupied_mask = 0;
                // Redistribute overflow chain
                Value curr = chunks_[current].embed_head.load(std::memory_order_acquire);
                while (curr) {
                    Value next = static_cast<Value>(curr->hash_chain_next());
                    size_type new_idx = curr->cached_hash() & new_mask;
                    exclusive_bucket_lock new_guard(new_chunks[new_idx].spin);
                    int new_slot = find_empty_slot_in_chunk(new_chunks[new_idx]);
                    if (new_slot >= 0) {
                        new_chunks[new_idx].tags[new_slot] = f14_detail::f14_tag(curr->cached_hash());
                        new_chunks[new_idx].slots[new_slot] = curr;
                        new_chunks[new_idx].occupied_mask |= static_cast<uint16_t>(1u << new_slot);
                        curr->set_hash_chain_next(nullptr);
                    } else {
                        curr->set_hash_chain_next(new_chunks[new_idx].embed_head.load(std::memory_order_acquire));
                        new_chunks[new_idx].embed_head.store(curr, std::memory_order_release);
                    }
                    curr = next;
                }
                chunks_[current].embed_head.store(nullptr, std::memory_order_release);
            } else {
                // Non-EmbeddedChain: redistribute inline slots
                for (int s = 0; s < f14_detail::kChunkCapacity; ++s) {
                    if (chunks_[current].occupied_mask & (1u << s)) {
                        node_type* node = static_cast<node_type*>(chunks_[current].slots[s]);
                        size_type new_idx = node->hash & new_mask;
                        exclusive_bucket_lock new_guard(new_chunks[new_idx].spin);
                        int new_slot = find_empty_slot_in_chunk(new_chunks[new_idx]);
                        if (new_slot >= 0) {
                            new_chunks[new_idx].tags[new_slot] = chunks_[current].tags[s];
                            new_chunks[new_idx].slots[new_slot] = node;
                            new_chunks[new_idx].occupied_mask |= static_cast<uint16_t>(1u << new_slot);
                        } else {
                            node->next = new_chunks[new_idx].node_head.load(std::memory_order_acquire);
                            new_chunks[new_idx].node_head.store(node, std::memory_order_release);
                        }
                        chunks_[current].slots[s] = nullptr;
                    }
                    chunks_[current].tags[s] = f14_detail::kTagEmpty;
                }
                chunks_[current].occupied_mask = 0;
                // Redistribute overflow chain
                node_type* node = chunks_[current].node_head.load(std::memory_order_acquire);
                while (node) {
                    node_type* next_node = node->next;
                    size_type new_idx = node->hash & new_mask;
                    exclusive_bucket_lock new_guard(new_chunks[new_idx].spin);
                    int new_slot = find_empty_slot_in_chunk(new_chunks[new_idx]);
                    if (new_slot >= 0) {
                        new_chunks[new_idx].tags[new_slot] = f14_detail::f14_tag(node->hash);
                        new_chunks[new_idx].slots[new_slot] = node;
                        new_chunks[new_idx].occupied_mask |= static_cast<uint16_t>(1u << new_slot);
                        node->next = nullptr;
                    } else {
                        node->next = new_chunks[new_idx].node_head.load(std::memory_order_acquire);
                        new_chunks[new_idx].node_head.store(node, std::memory_order_release);
                    }
                    node = next_node;
                }
                chunks_[current].node_head.store(nullptr, std::memory_order_release);
            }

            // Invalidate optimistic reads of the old chunk
            chunks_[current].version.fetch_add(2, std::memory_order_release);

            // Advance progress inside the lock so find() sees consistent state
            rehash_progress_.store(current + 1, std::memory_order_release);

            ++migrated;
        }

        rehash_migrating_.store(false, std::memory_order_release);

        // Auto-complete if all chunks migrated
        if (rehash_progress_.load(std::memory_order_acquire) >= old_chunk_count) {
            rehash_finish_f14();
        }

        return migrated;
    }

    /// Complete F14 incremental rehash: migrate any remaining chunks and
    /// install the new chunks array as the primary. Uses a CAS on
    /// rehash_in_progress_ to ensure only one thread performs the installation.
    ///
    /// P0-4 (T1.2): As in chain mode, we cap the synchronous chunk
    /// migration at kRehashFinishMaxBucketsPerCall chunks per call to
    /// bound the worst-case tail latency. If more work remains, we leave
    /// the rehash in progress and return; subsequent writes or the
    /// background rehash balancer will advance the migration.
    void rehash_finish_f14() {
        if constexpr (!kIsF14) return;
        if (!rehash_in_progress_.load(std::memory_order_acquire)) return;

        const size_type old_chunk_count = bucket_mask_.load(std::memory_order_relaxed) + 1;

        // P0-4 (T1.2): Cap synchronous migration to bound tail latency.
        size_type migrated_this_call = 0;
        while (rehash_progress_.load(std::memory_order_acquire) < old_chunk_count) {
            if (migrated_this_call >= kRehashFinishMaxBucketsPerCall) {
                // Out of budget — record stall and return without finishing.
                rehash_finish_stall_count_.fetch_add(1, std::memory_order_relaxed);
                std::size_t backlog = old_chunk_count - rehash_progress_.load(std::memory_order_relaxed);
                std::size_t prev_max = rehash_finish_max_backlog_.load(std::memory_order_relaxed);
                while (backlog > prev_max && !rehash_finish_max_backlog_.compare_exchange_weak(
                    prev_max, backlog, std::memory_order_relaxed)) {
                    // retry with updated prev_max
                }
                return;
            }
            if (rehash_step_f14(16) == 0) {
                // Another thread holds the migrating flag; spin briefly.
                LRU_SPIN_PAUSE();
            } else {
                migrated_this_call += 16;
            }
        }

        // Atomically claim the finish. Only the first thread past this point
        // proceeds to install the new array; concurrent callers return early.
        bool expected = true;
        if (!rehash_in_progress_.compare_exchange_strong(expected, false,
                std::memory_order_acq_rel)) {
            return; // Another thread already finished
        }

        // We've claimed the finish. Install the new array.
        seqlock_.fetch_add(1, std::memory_order_release);

        const size_type new_mask = rehash_new_bucket_count_.load(std::memory_order_relaxed) - 1;

        // P3-1: Swap chunks_ atomically (never null) before publishing the
        // new bucket_mask_. See the blocking-rehash path above for rationale.
        chunks_.swap(rehash_new_chunks_owner_);
        bucket_mask_.store(new_mask, std::memory_order_release);
        retired_chunks_.clear();
        retired_chunks_.push_back(std::move(rehash_new_chunks_owner_));

        // Clear remaining incremental rehash state (in_progress already cleared)
        rehash_new_chunks_.store(nullptr, std::memory_order_release);
        rehash_new_bucket_count_.store(0, std::memory_order_release);
        rehash_progress_.store(0, std::memory_order_release);

        seqlock_.fetch_add(1, std::memory_order_release);
    }

private:
    // ========================================================================
    // Internal node type (used only when EmbeddedChain = false)
    // ========================================================================

    struct node_type {
        Key        key;
        Value      value;
        size_type  hash;
        node_type* next;

        node_type(Key k, Value v, size_type h, node_type* n)
            : key(std::move(k)), value(std::move(v)), hash(h), next(n) {}
    };

    // ========================================================================
    // Bucket type — chain mode (head pointer + cacheline-aligned shared spinlock)
    // ========================================================================

    // P2-C: `version` co-located with the chain head atomics so the
    // optimistic-read path (v1 → embed_head/node_head → v2) touches a
    // single cache line. Shrinks the bucket from 192 → 128 bytes.
    struct bucket_type {
        std::atomic<Value>      embed_head{nullptr};
        std::atomic<node_type*> node_head{nullptr};
        alignas(8) std::atomic<uint64_t> version{0};
        aligned_shared_spinlock spin;
    };

    // ========================================================================
    // F14 chunk type — 14 inline slots with tag-based SIMD filtering
    // ========================================================================
    //
    // Layout:
    //   tags[0..13]  — 8-bit hash tags (0x00=empty, 0x01=tombstone, 0x80+=occupied)
    //   occupied_mask — bit i set if slot i is occupied
    //   slots[0..13] — void* slot storage (Value* or node_type* depending on EmbeddedChain)
    //   embed_head / node_head — overflow chain for items that don't fit inline
    //   spin        — per-chunk shared spinlock
    //   version     — optimistic read version counter
    //
    // The tags[] and occupied_mask are placed contiguously so that a 16-byte
    // SIMD load starting at tags[0] covers all 14 tags plus the 2 bytes of
    // occupied_mask. The SIMD match result is ANDed with occupied_mask to
    // filter out false positives from the mask bytes and tombstone tags.

    struct f14_chunk_type {
        static constexpr int kCapacity = f14_detail::kChunkCapacity;

        // ----------------------------------------------------------------
        // P2-C: Cache-line aware layout for the F14 chunk.
        //
        // The seqlock `version` is bumped twice per writer critical section
        // (odd = write in progress, even = write published) and loaded twice
        // per optimistic read (v1 before data, v2 after data). It is the
        // hottest atomic in the chunk.
        //
        // Placing `version` immediately after the 16-byte SIMD block
        // (tags[14] + occupied_mask) co-locates it with the data the
        // reader must load anyway, so:
        //   - Reader hot path: 2 cache lines (line 0: tags+mask+version+
        //     slots[0..4]; line 1: slots[5..12]) instead of 3.
        //   - Writer hot path: 1 cache line touched per modification
        //     (tag write + version bump on the same line) instead of 2.
        //   - Struct size: 256 bytes (4 cache lines) instead of 320 bytes
        //     (5 cache lines) — 20% memory reduction for the chunk array.
        //
        // The 16-byte SIMD load (`_mm_loadu_si128` / `vld1q_u8`) reads
        // exactly `tags[14] + occupied_mask` and does NOT include `version`,
        // so the SIMD match result is unaffected. `version` is 8-byte
        // aligned (offset 16), satisfying the alignment requirement of
        // `std::atomic<uint64_t>` on x86-64 / ARM.
        // ----------------------------------------------------------------
        uint8_t  tags[kCapacity] = {};
        uint16_t occupied_mask = 0;
        alignas(8) std::atomic<uint64_t> version{0};
        void*    slots[kCapacity] = {};

        // Overflow chain (same structure as chain mode's bucket_type).
        // Cold path — only touched when the chunk's inline slots are full
        // and the lookup must follow the spill chain.
        std::atomic<Value>      embed_head{nullptr};
        std::atomic<node_type*> node_head{nullptr};

        aligned_shared_spinlock spin;

        // ------------------------------------------------------------
        // P1-8 (T-B3): Atomic access helpers for tags[] and occupied_mask.
        // ------------------------------------------------------------
        // The optimistic-read path (find_f14_*) reads `tags[]` via SIMD
        // and `occupied_mask` via a plain load, then re-checks `version`
        // to detect concurrent writes (seqlock pattern). On x86-64 (TSO)
        // plain loads are atomic for naturally-sized accesses, and the
        // seqlock guarantees correctness. On weak-memory architectures
        // (ARMv8 without LSE, RISC-V relaxed), plain loads could be
        // reordered with the version check, breaking the seqlock.
        //
        // These helpers wrap std::atomic_ref to provide acquire/release
        // ordering without changing the layout (preserves SIMD match).
        // The SIMD load of `tags[]` remains a plain load — on x86-64
        // aligned 16-byte loads are atomic; on ARM the version check
        // is the safety net (any torn SIMD read will be detected when
        // the version is re-checked). The occupied_mask read is the
        // most exposed (16-bit RMW on weak architectures), so it gets
        // the atomic treatment.
        //
        // Callers holding the per-chunk spinlock may use the plain
        // (non-atomic) accessors — the spinlock provides the necessary
        // synchronization. Use these atomic helpers ONLY on the
        // optimistic-read path (without the spinlock).
        // ------------------------------------------------------------

        /// Atomically load `occupied_mask` with acquire ordering.
        /// Use this on the optimistic-read path; under the spinlock,
        /// a plain read is sufficient (and faster).
        uint16_t load_occupied_mask_acquire() const noexcept {
            return std::atomic_ref<uint16_t>(
                const_cast<uint16_t&>(occupied_mask))
                .load(std::memory_order_acquire);
        }

        /// Atomically load a single tag with acquire ordering.
        /// Use this when reading a specific slot's tag outside the
        /// spinlock (e.g., for individual slot validation).
        uint8_t load_tag_acquire(int slot) const noexcept {
            return std::atomic_ref<uint8_t>(
                const_cast<uint8_t&>(tags[slot]))
                .load(std::memory_order_acquire);
        }

        /// Atomically store a tag with release ordering. Callers MUST
        /// hold the spinlock — the release ordering ensures the slot
        /// pointer write (which happens after this) is not reordered
        /// before the tag write, and that the tag write is visible to
        /// optimistic readers before the version bump.
        void store_tag_release(int slot, uint8_t tag) noexcept {
            std::atomic_ref<uint8_t>(tags[slot])
                .store(tag, std::memory_order_release);
        }

        /// Atomically set a bit in `occupied_mask` with acq_rel
        /// ordering. RMW operation — uses fetch_or to ensure the bit
        /// is set atomically even if another thread is concurrently
        /// modifying a different bit.
        void set_occupied_bit_acq_rel(int slot) noexcept {
            std::atomic_ref<uint16_t>(occupied_mask)
                .fetch_or(static_cast<uint16_t>(1u << slot),
                          std::memory_order_acq_rel);
        }

        /// Atomically clear a bit in `occupied_mask` with acq_rel
        /// ordering. Used when a slot is marked as tombstone.
        void clear_occupied_bit_acq_rel(int slot) noexcept {
            std::atomic_ref<uint16_t>(occupied_mask)
                .fetch_and(static_cast<uint16_t>(~(1u << slot)),
                           std::memory_order_acq_rel);
        }

        /// Atomically store `occupied_mask` with release ordering.
        /// Used during rehash/clear when the entire mask is replaced.
        void store_occupied_mask_release(uint16_t mask) noexcept {
            std::atomic_ref<uint16_t>(occupied_mask)
                .store(mask, std::memory_order_release);
        }
    };

    // ========================================================================
    // Internal helpers — chain mode (unchanged)
    // ========================================================================

    node_type* find_node(const Key& key, size_type idx) const {
        node_type* node = buckets_[idx].node_head.load(std::memory_order_acquire);
        while (node) {
            if (equal_(node->key, key)) return node;
            node = node->next;
        }
        return nullptr;
    }

    Value find_node_embedded(const Key& key, size_type idx) const {
        Value node = buckets_[idx].embed_head.load(std::memory_order_acquire);
        while (node) {
            if (equal_(node->key, key)) return node;
            node = static_cast<Value>(node->hash_chain_next());
        }
        return nullptr;
    }

    // ========================================================================
    // Internal helpers — incremental rehash (chain mode)
    // ========================================================================

    /// Find an embedded-chain node in a specific bucket array (not necessarily buckets_).
    Value find_node_embedded_in(const Key& key, bucket_type* bkts, size_type idx) const {
        Value node = bkts[idx].embed_head.load(std::memory_order_acquire);
        while (node) {
            if (equal_(node->key, key)) return node;
            node = static_cast<Value>(node->hash_chain_next());
        }
        return nullptr;
    }

    /// Find a non-embedded node in a specific bucket array.
    node_type* find_node_in(const Key& key, bucket_type* bkts, size_type idx) const {
        node_type* node = bkts[idx].node_head.load(std::memory_order_acquire);
        while (node) {
            if (equal_(node->key, key)) return node;
            node = node->next;
        }
        return nullptr;
    }

    // ========================================================================
    // Internal helpers — F14 mode
    // ========================================================================

    /// Find in an F14 chunk (EmbeddedChain mode).
    /// Searches inline slots using SIMD tag matching, then overflow chain.
    Value find_f14_embedded(const Key& key, size_type chunk_idx) const {
        const size_type h = hash_(key);
        uint8_t tag = f14_detail::f14_tag(h);
        auto& chunk = chunks_[chunk_idx];

        // SIMD tag match in inline slots
        uint16_t match_mask = f14_detail::f14_match_tags(chunk.tags, tag)
                            & chunk.load_occupied_mask_acquire();

        while (match_mask) {
            int slot = f14_detail::ctz16(match_mask);
            match_mask &= static_cast<uint16_t>(match_mask - 1);
            Value node = static_cast<Value>(chunk.slots[slot]);
            if (equal_(node->key, key)) return node;
        }

        // Search overflow chain
        Value curr = chunk.embed_head.load(std::memory_order_acquire);
        while (curr) {
            if (equal_(curr->key, key)) return curr;
            curr = static_cast<Value>(curr->hash_chain_next());
        }

        return nullptr;
    }

    /// Find in an F14 chunk (non-EmbeddedChain mode).
    node_type* find_f14_node(const Key& key, size_type chunk_idx) const {
        const size_type h = hash_(key);
        uint8_t tag = f14_detail::f14_tag(h);
        auto& chunk = chunks_[chunk_idx];

        uint16_t match_mask = f14_detail::f14_match_tags(chunk.tags, tag)
                            & chunk.load_occupied_mask_acquire();

        while (match_mask) {
            int slot = f14_detail::ctz16(match_mask);
            match_mask &= static_cast<uint16_t>(match_mask - 1);
            node_type* node = static_cast<node_type*>(chunk.slots[slot]);
            if (equal_(node->key, key)) return node;
        }

        // Search overflow chain
        node_type* node = chunk.node_head.load(std::memory_order_acquire);
        while (node) {
            if (equal_(node->key, key)) return node;
            node = node->next;
        }

        return nullptr;
    }

    /// Find an empty slot (empty or tombstone) in a chunk.
    /// Returns slot index [0, kCapacity), or -1 if all slots are occupied.
    int find_f14_empty_slot(size_type chunk_idx) const {
        auto& chunk = chunks_[chunk_idx];
        // Prefer empty slots over tombstones for better SIMD filtering
        uint16_t empty_mask = static_cast<uint16_t>(~chunk.occupied_mask & f14_detail::kFullMask);
        // Check for slots with kTagEmpty first (clean empty, not tombstone)
        for (int i = 0; i < f14_detail::kChunkCapacity; ++i) {
            if (chunk.tags[i] == f14_detail::kTagEmpty && !(chunk.occupied_mask & (1u << i))) {
                return i;
            }
        }
        // Fall back to tombstone slots
        if (empty_mask) {
            return f14_detail::ctz16(empty_mask);
        }
        return -1; // All slots occupied
    }

    /// Find an empty slot in a new (empty) chunk during rehash.
    static int find_empty_slot_in_chunk(const f14_chunk_type& chunk) {
        for (int i = 0; i < f14_detail::kChunkCapacity; ++i) {
            if (chunk.tags[i] == f14_detail::kTagEmpty) return i;
        }
        return -1;
    }

    // ========================================================================
    // Internal helpers — F14 incremental rehash (dual-array lookup)
    // ========================================================================

    /// Find in a specific F14 chunk array (EmbeddedChain mode).
    /// Used during incremental rehash to search the new (or old) chunks array.
    Value find_f14_embedded_in(const Key& key, f14_chunk_type* chs, size_type idx) const {
        const size_type h = hash_(key);
        uint8_t tag = f14_detail::f14_tag(h);
        auto& chunk = chs[idx];

        uint16_t match_mask = f14_detail::f14_match_tags(chunk.tags, tag)
                            & chunk.load_occupied_mask_acquire();
        while (match_mask) {
            int slot = f14_detail::ctz16(match_mask);
            match_mask &= static_cast<uint16_t>(match_mask - 1);
            Value node = static_cast<Value>(chunk.slots[slot]);
            if (equal_(node->key, key)) return node;
        }

        Value curr = chunk.embed_head.load(std::memory_order_acquire);
        while (curr) {
            if (equal_(curr->key, key)) return curr;
            curr = static_cast<Value>(curr->hash_chain_next());
        }
        return nullptr;
    }

    /// Find in a specific F14 chunk array (non-EmbeddedChain mode).
    node_type* find_f14_node_in(const Key& key, f14_chunk_type* chs, size_type idx) const {
        const size_type h = hash_(key);
        uint8_t tag = f14_detail::f14_tag(h);
        auto& chunk = chs[idx];

        uint16_t match_mask = f14_detail::f14_match_tags(chunk.tags, tag)
                            & chunk.load_occupied_mask_acquire();
        while (match_mask) {
            int slot = f14_detail::ctz16(match_mask);
            match_mask &= static_cast<uint16_t>(match_mask - 1);
            node_type* node = static_cast<node_type*>(chunk.slots[slot]);
            if (equal_(node->key, key)) return node;
        }

        node_type* node = chunk.node_head.load(std::memory_order_acquire);
        while (node) {
            if (equal_(node->key, key)) return node;
            node = node->next;
        }
        return nullptr;
    }

    /// F14 dual-array aware lookup. Called only when incremental_rehash_
    /// && rehash_in_progress_. Returns the found node (Value for
    /// EmbeddedChain, node_type* for non-EmbeddedChain) or nullptr.
    /// Uses shared_bucket_lock on whichever array owns the key's bucket.
    ///
    /// G18: The retry loop is bounded by kDualArrayMaxRetries. If the
    /// rehash thread migrates the target bucket between the initial
    /// progress read and the in-lock re-check on every attempt, an
    /// unbounded loop would spin indefinitely — wasting CPU on lock
    /// acquisitions that never converge. When retries are exhausted, we
    /// fall back to a deterministic both-arrays search (see the fallback
    /// block below), mirroring the shared-lock fallback used by
    /// find_and_pin_lockfree_with_hash during incremental rehash.
    auto find_f14_dual_array(const Key& key, size_type h) const {
        for (int retry = 0; retry < kDualArrayMaxRetries; ++retry) {
            size_type old_idx = h & bucket_mask_.load(std::memory_order_acquire);
            size_type progress = rehash_progress_.load(std::memory_order_acquire);
            if (old_idx < progress) {
                size_type new_mask = rehash_new_bucket_count_.load(std::memory_order_acquire) - 1;
                size_type new_idx = h & new_mask;
                f14_chunk_type* new_chunks = rehash_new_chunks_.load(std::memory_order_acquire);
                shared_bucket_lock guard(new_chunks[new_idx].spin);
                if (rehash_progress_.load(std::memory_order_acquire) >= progress) {
                    if constexpr (EmbeddedChain) {
                        return find_f14_embedded_in(key, new_chunks, new_idx);
                    } else {
                        return find_f14_node_in(key, new_chunks, new_idx);
                    }
                }
            } else {
                shared_bucket_lock guard(chunks_[old_idx].spin);
                if (rehash_progress_.load(std::memory_order_acquire) <= old_idx) {
                    if constexpr (EmbeddedChain) {
                        return find_f14_embedded(key, old_idx);
                    } else {
                        return find_f14_node(key, old_idx);
                    }
                }
            }
        }

        // G18: Bounded retries exhausted — rehash progress is advancing
        // faster than the per-bucket shared-lock retry loop can converge.
        // Continuing to retry would spin indefinitely, wasting CPU on
        // lock acquisitions that never settle. Fall back to a
        // deterministic both-arrays search that does NOT depend on
        // rehash_progress_ stability: acquire the shared lock on the
        // old-array bucket and the new-array bucket in turn (at most 2
        // lock acquisitions). Each shared lock blocks migration of that
        // bucket, so the item (if present) is stable under the lock. The
        // item resides in exactly one array at any moment (migration is
        // atomic per bucket under exclusive lock), so searching both is
        // guaranteed to find it without retrying. This is the shared-lock
        // fallback path, consistent with find_and_pin_lockfree_with_hash
        // (line ~1759) falling back to find_and_pin_with_hash during
        // incremental rehash; the rehash_lockfree_fallback_count_ counter
        // is incremented here too for operator visibility.
        rehash_lockfree_fallback_count_.fetch_add(1, std::memory_order_relaxed);

        // Rehash may have completed during the retries — if so, the
        // primary array (chunks_) is now the sole source of truth and
        // bucket_mask_ reflects the new (larger) array.
        if (!rehash_in_progress_.load(std::memory_order_acquire)) {
            size_type idx = h & bucket_mask_.load(std::memory_order_acquire);
            shared_bucket_lock guard(chunks_[idx].spin);
            if constexpr (EmbeddedChain) {
                return find_f14_embedded(key, idx);
            } else {
                return find_f14_node(key, idx);
            }
        }

        // Rehash still in progress — search the old array first. The
        // shared lock blocks concurrent migration of this bucket, so a
        // consistent miss here means the bucket was already migrated.
        {
            size_type old_idx = h & bucket_mask_.load(std::memory_order_acquire);
            shared_bucket_lock guard(chunks_[old_idx].spin);
            if constexpr (EmbeddedChain) {
                Value node = find_f14_embedded(key, old_idx);
                if (node) return node;
            } else {
                node_type* node = find_f14_node(key, old_idx);
                if (node) return node;
            }
        }

        // Search the new array (if rehash hasn't completed and freed it).
        const size_type new_count = rehash_new_bucket_count_.load(std::memory_order_acquire);
        f14_chunk_type* new_chunks = rehash_new_chunks_.load(std::memory_order_acquire);
        if (new_chunks && new_count) {
            size_type new_mask = new_count - 1;
            size_type new_idx = h & new_mask;
            shared_bucket_lock guard(new_chunks[new_idx].spin);
            if constexpr (EmbeddedChain) {
                return find_f14_embedded_in(key, new_chunks, new_idx);
            } else {
                return find_f14_node_in(key, new_chunks, new_idx);
            }
        }

        // Rehash completed between the in-progress check and the new-array
        // load (new_chunks is null) — re-check the primary array, which is
        // now the sole source of truth with the updated bucket_mask_.
        {
            size_type idx = h & bucket_mask_.load(std::memory_order_acquire);
            shared_bucket_lock guard(chunks_[idx].spin);
            if constexpr (EmbeddedChain) {
                return find_f14_embedded(key, idx);
            } else {
                return find_f14_node(key, idx);
            }
        }
    }

    // ========================================================================
    // Node allocation / deallocation (slab-aware, non-embedded only)
    // ========================================================================

    node_type* allocate_node(const Key& key, Value value, size_type h, node_type* next) {
        if (alloc_fn_) {
            void* raw = alloc_fn_(sizeof(node_type));
            return new (raw) node_type{key, std::move(value), h, next};
        }
        return new node_type{key, std::move(value), h, next};
    }

    void deallocate_node(node_type* node) noexcept {
        if (dealloc_fn_) {
            node->~node_type();
            dealloc_fn_(node);
        } else {
            delete node;
        }
    }

    static constexpr size_type next_power_of_two(size_type n) noexcept {
        if (n == 0) return 1;
        --n;
        n |= n >> 1;  n |= n >> 2;
        n |= n >> 4;  n |= n >> 8;
        n |= n >> 16;
#if SIZE_MAX > 0xFFFFFFFF
        n |= n >> 32;
#endif
        return n + 1;
    }

    // ========================================================================
    // Hazptr-protected wait-free find (no RMW, no cache line bouncing)
    // ========================================================================

    /// Hazptr-protected traversal for EmbeddedChain (non-F14) mode.
    /// Uses two hazptr holders to walk the chain without any RMW operations.
    /// This is wait-free: no fetch_add, no lock acquisition.
    /// Falls back to nullptr if hazptr slots are exhausted (caller should
    /// then fall back to shared lock).
    Value find_hazptr_embedded(const Key& key, size_type idx) const {
        hazptr_holder hazcurr;
        hazptr_holder haznext;

        Value curr = buckets_[idx].embed_head.load(std::memory_order_acquire);
        hazcurr.protect(curr);

        while (curr) {
            // Check key while curr is protected from reclamation
            if (equal_(curr->key, key)) {
                return curr;
            }

            // Read next pointer and protect it
            Value next = static_cast<Value>(curr->hash_chain_next());
            if (!next) break;

            haznext.protect(next);
            // Re-verify that curr is still reachable: reload embed_head
            // and walk to see if curr is still in the chain.
            // This handles the case where another thread unlinked curr
            // between our protect(curr) and reading curr's next pointer.
            Value verify = buckets_[idx].embed_head.load(std::memory_order_acquire);
            if (verify != curr && !is_reachable(verify, curr)) {
                // curr was unlinked — restart from head
                hazcurr.protect(verify);
                curr = verify;
                continue;
            }

            // Advance: swap holders so hazcurr now protects next
            hazcurr.swap(haznext);
            curr = next;
        }

        return nullptr;
    }

    /// Check if `target` is reachable from `head` by following hash_chain_next.
    /// Used to verify that a hazptr-protected node is still in the chain.
    bool is_reachable(Value head, Value target) const {
        int steps = 0;
        while (head && steps < 64) {  // Bound walk to avoid unbounded scan
            if (head == target) return true;
            head = static_cast<Value>(head->hash_chain_next());
            ++steps;
        }
        return false;
    }

    /// Hazptr-protected traversal for F14 EmbeddedChain mode.
    ///
    /// Inline slots: version-stamped hazptr. The node pointer is read from
    /// the slot, immediately hazptr-protected, then the chunk version is
    /// re-read. If the version is unchanged and even, the slot still owns
    /// the node (no concurrent erase has unlinked it), so the hazptr
    /// protection is effective — any concurrent retire will be deferred
    /// until we release the hazptr. Only then is `node->key` dereferenced.
    ///
    /// Overflow chain: standard hazptr-protected traversal with re-reachability
    /// verification to handle concurrent unlink.
    Value find_f14_hazptr_embedded(const Key& key, size_type chunk_idx) const {
        const size_type h = hash_(key);
        uint8_t tag = f14_detail::f14_tag(h);
        auto& chunk = chunks_[chunk_idx];

        // Inline slots: version-stamped hazptr (safe against concurrent erase+retire)
        {
            hazptr_holder hazslot;
            auto v1 = chunk.version.load(std::memory_order_acquire);
            if ((v1 & 1u) == 0) {
                uint16_t match_mask = f14_detail::f14_match_tags(chunk.tags, tag)
                                    & chunk.load_occupied_mask_acquire();
                while (match_mask) {
                    int slot = f14_detail::ctz16(match_mask);
                    match_mask &= static_cast<uint16_t>(match_mask - 1);
                    // Atomic load of slot pointer (slots[] is void*, not atomic)
                    Value node = static_cast<Value>(
                        std::atomic_ref<void*>(chunk.slots[slot]).load(std::memory_order_acquire));
                    if (!node) continue;
                    hazslot.protect(node);
                    // Re-read version after publishing hazptr: if unchanged and
                    // even, the slot still references `node`, so retire (if any
                    // is in flight on another thread) will observe our hazptr
                    // and defer reclamation.
                    auto v2 = chunk.version.load(std::memory_order_acquire);
                    if (v1 != v2 || (v2 & 1u) != 0) {
                        break;  // chunk modified concurrently — fall to overflow
                    }
                    // Re-confirm slot still points to `node` (any slot mutation
                    // bumps version, so v1==v2 is sufficient; the explicit
                    // re-check is a belt-and-suspenders against ABA on the
                    // slot pointer itself).
                    Value node2 = static_cast<Value>(
                        std::atomic_ref<void*>(chunk.slots[slot]).load(std::memory_order_acquire));
                    if (node2 == node && equal_(node->key, key)) {
                        return node;
                    }
                }
            }
        }

        // Hazptr traversal of overflow chain
        hazptr_holder hazcurr;
        hazptr_holder haznext;

        Value curr = chunk.embed_head.load(std::memory_order_acquire);
        hazcurr.protect(curr);

        while (curr) {
            if (equal_(curr->key, key)) {
                return curr;
            }

            Value next = static_cast<Value>(curr->hash_chain_next());
            if (!next) break;

            haznext.protect(next);
            Value verify = chunk.embed_head.load(std::memory_order_acquire);
            if (verify != curr && !is_reachable(verify, curr)) {
                hazcurr.protect(verify);
                curr = verify;
                continue;
            }

            hazcurr.swap(haznext);
            curr = next;
        }

        return nullptr;
    }

    // ========================================================================
    // Data members
    // ========================================================================

    Hash        hash_;
    KeyEqual    equal_;

    std::atomic<size_type> bucket_mask_{0};

    // Chain mode storage
    std::unique_ptr<bucket_type[]> buckets_;

    // F14 mode storage
    std::unique_ptr<f14_chunk_type[]> chunks_;

    // P0-3 (T1.1): Hot-path atomic counter padded to its own cache line.
    // Every insert/erase performs fetch_add/fetch_sub here; on 64+ core
    // machines an unpadded `size_` would false-share with adjacent members
    // and concentrate RMW traffic on a single cache line. Padding isolates
    // the contention to its own line, eliminating false sharing with
    // neighboring state. (For per-shard caches the contention is already
    // bounded by the sharded design, but the padding is still beneficial
    // for non-segmented configurations.)
    alignas(64) std::atomic<size_type> size_{0};

    allocate_fn    alloc_fn_ = nullptr;
    deallocate_fn  dealloc_fn_ = nullptr;

    // P0-1: 乐观读默认值取决于 EmbeddedChain 模式。
    //  - EmbeddedChain = true: 节点即 Value，由 MM 的 refcount + hazptr 保护，
    //    乐观读返回的指针在被 refcount 递增前可能被驱逐，但 hazptr 保护
    //    确保节点内存不会在乐观读窗口内被释放。默认启用。
    //  - EmbeddedChain = false: node_type 独立分配，其生命周期由哈希表
    //    管理（无 refcount）。乐观读返回 node->value 指针后，另一个线程
    //    可能在 TOCTOU 窗口内删除该节点，导致 use-after-free。默认禁用，
    //    仅在用户显式调用 set_optimistic_read(true) 且接受 UAF 风险时启用。
    bool enable_optimistic_read_ = EmbeddedChain;
    bool enable_hazptr_read_ = true;

    std::atomic<uint64_t> seqlock_{0};

    // P1-1: Rehash diagnostics — atomically track rehash frequency,
    // duration, and migration volume. Read by unified_cache::stats_snapshot()
    // and exported via Prometheus. These are local to the hash table to
    // preserve layering (hash table does not depend on cache_stats).
    alignas(64) std::atomic<std::size_t> rehash_count_{0};
    alignas(64) std::atomic<std::uint64_t> rehash_total_time_ns_{0};
    alignas(64) std::atomic<std::size_t> rehash_migrated_items_{0};

    // T11.3: Counter of write operations that arrived while a blocking
    // (non-incremental) rehash was in progress and had to wait for it
    // to finish before acquiring the bucket lock. Incremented once per
    // affected write. Non-zero values indicate the user should enable
    // `set_incremental_rehash(true)` to avoid blocking writes during
    // hash table expansion. Padded to its own cache line because the
    // rehash path mutates it while concurrent readers poll it.
    alignas(64) std::atomic<std::size_t> rehash_blocked_writes_count_{0};

    // P0-4 (T1.2): Rehash-finish stall metrics. rehash_finish_stall_count
    // is incremented each time rehash_finish() returns early because it hit
    // the kRehashFinishMaxBucketsPerCall budget — i.e., the rehash could
    // not be completed synchronously. rehash_finish_max_backlog tracks the
    // high-water mark of remaining buckets/chunks across all stalled
    // finishes. Non-zero stall_count means the background rehash balancer
    // is needed (or the workload is outpacing incremental migration).
    alignas(64) std::atomic<std::size_t> rehash_finish_stall_count_{0};
    alignas(64) std::atomic<std::size_t> rehash_finish_max_backlog_{0};

    // P1-5: Count of times find_and_pin_lockfree fell back to the
    // lock-protected path because the segment was in incremental rehash.
    // Per-segment counter — aggregated by segmented_concurrent_hash_table.
    // Operators monitor this to detect sustained rehash activity that
    // degrades the lock-free read path. Mutable because the const
    // find_and_pin_lockfree() overload must still be able to record a
    // fallback event (the const-ness is about the table structure, not
    // the diagnostic counters).
    alignas(64) mutable std::atomic<std::size_t> rehash_lockfree_fallback_count_{0};

    // F14: ~10 items per 14-slot chunk ≈ 71% inline utilization
    float max_load_factor_ = kIsF14 ? 10.0f : 4.0f;

    // T13.1: Configurable overload threshold and counter. Padded to
    // avoid false sharing with the rehash counters above (which are
    // updated on every rehash, while these are read on every insert).
    alignas(64) std::atomic<float> hash_overload_threshold_{2.0f};
    alignas(64) std::atomic<std::size_t> hash_overload_events_{0};
    // T13.2: User-registered callback invoked when load_factor exceeds
    // the overload threshold. Invoked from the rehash hot path, so it
    // must be cheap. Exceptions are swallowed by the caller.
    std::function<void(float, float)> hash_overload_callback_;
    // T13.4: Rate-limit flag for the stderr warning. Set when the warning
    // is printed; cleared when load_factor drops back below the threshold.
    // This prevents log spam when the table is persistently overloaded
    // (e.g. during stress tests). The callback above still fires on every
    // overloaded insert — only the stderr message is deduplicated.
    alignas(64) std::atomic<bool> hash_overload_warned_{false};

    // P2-4 (T2.4): Async overload-callback queue. When
    // `overload_callback_async_` is true, the rehash hot path enqueues
    // {current_lf, threshold} pairs into `overload_queue_` (guarded by
    // `overload_queue_mutex_`) instead of invoking the user callback
    // inline. A background worker (driven by `drain_overload_callbacks()`)
    // later drains the queue and dispatches the callback off the hot
    // path. This bounds rehash-path latency when the user callback may
    // block (IO, metrics push, heap allocation).
    //
    // `overload_queue_mutex_` is `mutable` so `pending_overload_events()`
    // can acquire it from a const-qualified observer.
    alignas(64) std::atomic<bool> overload_callback_async_{false};
    mutable std::mutex overload_queue_mutex_;
    std::vector<std::pair<float, float>> overload_queue_;

    // Retired storage arrays from previous rehashes
    std::vector<std::unique_ptr<bucket_type[]>> retired_buckets_;
    std::vector<std::unique_ptr<f14_chunk_type[]>> retired_chunks_;

    // T2.1: EBR (Epoch-Based Reclamation) domain pointer. When non-null,
    // find_and_pin_lockfree_with_hash() acquires an epoch_guard at entry,
    // protecting all nodes from reclamation during the traversal. This is
    // the correct place for the guard (at the hash table entry, per the
    // T2.1 spec) rather than at the MM wrapper (peek_for_get), because it
    // covers ALL callers of find_and_pin_lockfree, current and future.
    //
    // When null (default), the hash table operates in hazptr mode: each
    // traversal uses hazptr_holder for per-pointer protection.
    //
    // T2.4: Callers can use the `reclaim_guard` type alias below to write
    // mode-agnostic code — it resolves to epoch_guard in EBR mode and to
    // a no-op guard in hazptr mode.
    detail::epoch_domain* ebr_domain_ = nullptr;

    // Incremental rehash state (chain mode only)
    bool incremental_rehash_{false};
    std::atomic<bool> rehash_in_progress_{false};
    std::atomic<bucket_type*> rehash_new_buckets_{nullptr};
    std::atomic<size_type> rehash_new_bucket_count_{0};
    std::atomic<size_type> rehash_progress_{0};     // number of old buckets fully migrated
    std::atomic<bool> rehash_migrating_{false};     // only one thread migrates at a time
    std::unique_ptr<bucket_type[]> rehash_new_buckets_owner_;  // owns the new bucket array memory

    // Incremental rehash state (F14 mode). These fields mirror the chain-mode
    // state above but are typed for f14_chunk_type. They are only used when
    // kIsF14 is true. rehash_progress_/rehash_in_progress_/rehash_migrating_/
    // rehash_new_bucket_count_ are shared between the two modes (only one mode
    // is active per instance, selected at compile time via ProbingStyle).
    std::atomic<f14_chunk_type*> rehash_new_chunks_{nullptr};
    std::unique_ptr<f14_chunk_type[]> rehash_new_chunks_owner_;
};

// ============================================================================
// Segmented Concurrent Hash Table
// ============================================================================
//
// Splits the hash table into N independent segments (default 64), each with
// its own bucket array, locks, and rehash state. Rehash only locks one
// segment, not the entire table, eliminating the global stall during rehash.
//
// The public API is identical to concurrent_hash_table. All operations are
// delegated to the appropriate segment based on the hash of the key.

template <
    typename Key,
    typename Value,
    typename Hash = std::hash<Key>,
    typename KeyEqual = std::equal_to<Key>,
    bool EmbeddedChain = false,
    typename ProbingStyle = chain_probing_tag,
    std::size_t NumSegments = 64>
class segmented_concurrent_hash_table {
public:
    using size_type = std::size_t;
    using hash_type = Hash;
    using key_equal = KeyEqual;
    using value_type = Value;
    using segment_type = concurrent_hash_table<Key, Value, Hash, KeyEqual, EmbeddedChain, ProbingStyle>;

    // R2: Expose EmbeddedChain setting for compile-time verification.
    static constexpr bool uses_embedded_chain = EmbeddedChain;

    // Same type aliases as concurrent_hash_table for compatibility
    using allocate_fn = typename segment_type::allocate_fn;
    using deallocate_fn = typename segment_type::deallocate_fn;
    using from_expected_items_t = typename segment_type::from_expected_items_t;
    static constexpr from_expected_items_t from_expected_items{};

private:
    std::vector<std::unique_ptr<segment_type>> segments_;
    std::size_t num_segments_;  // always a power of 2
    std::size_t segment_mask_;  // num_segments_ - 1
    Hash hash_;

    // P0-3 (T1.1): Aggregate size counter — padded to its own cache line to
    // avoid false sharing with hash_ and the segments_ vector. Every
    // per-segment insert/erase does fetch_add/fetch_sub here; on 64+ core
    // machines the unpadded layout causes measurable contention.
    alignas(64) std::atomic<size_type> total_size_{0};

    // T-B4 (P2-10): Diagnostics snapshot cache — periodically refreshed by
    // the background rehash balancer (1s cadence by default). Without this
    // cache, every prometheus_text() / diagnostics() scrape triggers an
    // O(total_buckets) scan across all 64 segments, which dominates scrape
    // latency on large caches (e.g. 1M buckets × 64 segments = 64M bucket
    // reads per scrape). With the cache, scrapes read a single atomic per
    // metric (~10ns), and the balancer's single 1s sweep absorbs the cost.
    //
    // Consistency: the snapshot is best-effort. A scrape may see metrics
    // up to ~1s stale; operators can check `diagnostics_cache_age_ms()` to
    // detect a stalled balancer (age >> interval). The cached values are
    // always <= the live values for max_chain_length (chain lengths only
    // grow between refreshes unless an erase shrinks them, which would
    // make the cached value a conservative upper bound).
    //
    // Layout: each padded to its own cache line to prevent false sharing
    // between the writer (balancer thread) and reader (scrape thread).
    alignas(64) mutable std::atomic<size_type> cached_max_chain_length_{0};
    alignas(64) mutable std::atomic<std::uint64_t> cached_snapshot_ns_{0};
    // Per-segment load factors are stored as bit-cast uint32_t (atomic<float>
    // is not guaranteed lock-free on all platforms). num_segments_ is bounded
    // by NumSegments (compile-time constant, typically 64); we size the array
    // to a compile-time upper bound so no dynamic allocation is needed.
    static constexpr std::size_t kMaxCachedSegments = 64;
    alignas(64) mutable std::array<std::atomic<std::uint32_t>, kMaxCachedSegments> cached_per_segment_lf_{};
    alignas(64) mutable std::atomic<std::size_t> cached_num_segments_{0};

    std::size_t segment_for_hash(size_type h) const noexcept {
        return h & segment_mask_;
    }

    std::size_t segment_for_key(const Key& key) const noexcept {
        return segment_for_hash(hash_(key));
    }

    static constexpr size_type next_power_of_two(size_type n) noexcept {
        if (n == 0) return 1;
        --n;
        n |= n >> 1;  n |= n >> 2;
        n |= n >> 4;  n |= n >> 8;
        n |= n >> 16;
#if SIZE_MAX > 0xFFFFFFFF
        n |= n >> 32;
#endif
        return n + 1;
    }

public:
    // ========================================================================
    // Constructor: each segment gets num_buckets/NumSegments buckets
    // ========================================================================

    explicit segmented_concurrent_hash_table(
        size_type num_buckets = 1024,
        const Hash& hash = Hash(),
        const KeyEqual& equal = KeyEqual())
        : hash_(hash)
    {
        num_segments_ = next_power_of_two(NumSegments);
        segment_mask_ = num_segments_ - 1;
        size_type per_segment = std::max(size_type(1),
            next_power_of_two(num_buckets) / num_segments_);
        segments_.reserve(num_segments_);
        for (std::size_t i = 0; i < num_segments_; ++i) {
            segments_.emplace_back(std::make_unique<segment_type>(per_segment, hash, equal));
        }
    }

    // from_expected_items constructor
    segmented_concurrent_hash_table(
        from_expected_items_t,
        size_type expected_items,
        const Hash& hash = Hash(),
        const KeyEqual& equal = KeyEqual())
        : hash_(hash)
    {
        num_segments_ = next_power_of_two(NumSegments);
        segment_mask_ = num_segments_ - 1;
        size_type per_segment_items = std::max(size_type(1), expected_items / num_segments_);
        segments_.reserve(num_segments_);
        for (std::size_t i = 0; i < num_segments_; ++i) {
            segments_.emplace_back(std::make_unique<segment_type>(from_expected_items, per_segment_items, hash, equal));
        }
    }

    // With alloc/dealloc fns
    segmented_concurrent_hash_table(
        size_type num_buckets,
        allocate_fn alloc_fn,
        deallocate_fn dealloc_fn,
        const Hash& hash = Hash(),
        const KeyEqual& equal = KeyEqual())
        : hash_(hash)
    {
        num_segments_ = next_power_of_two(NumSegments);
        segment_mask_ = num_segments_ - 1;
        size_type per_segment = std::max(size_type(1),
            next_power_of_two(num_buckets) / num_segments_);
        segments_.reserve(num_segments_);
        for (std::size_t i = 0; i < num_segments_; ++i) {
            segments_.emplace_back(std::make_unique<segment_type>(per_segment, alloc_fn, dealloc_fn, hash, equal));
        }
    }

    // from_expected_items + alloc/dealloc
    segmented_concurrent_hash_table(
        from_expected_items_t,
        size_type expected_items,
        allocate_fn alloc_fn,
        deallocate_fn dealloc_fn,
        const Hash& hash = Hash(),
        const KeyEqual& equal = KeyEqual())
        : hash_(hash)
    {
        num_segments_ = next_power_of_two(NumSegments);
        segment_mask_ = num_segments_ - 1;
        size_type per_segment_items = std::max(size_type(1), expected_items / num_segments_);
        segments_.reserve(num_segments_);
        for (std::size_t i = 0; i < num_segments_; ++i) {
            segments_.emplace_back(std::make_unique<segment_type>(from_expected_items, per_segment_items, alloc_fn, dealloc_fn, hash, equal));
        }
    }

    // ========================================================================
    // Move operations — std::atomic is not movable, handle explicitly
    // ========================================================================

    segmented_concurrent_hash_table(segmented_concurrent_hash_table&& other) noexcept
        : segments_(std::move(other.segments_))
        , num_segments_(other.num_segments_)
        , segment_mask_(other.segment_mask_)
        , hash_(std::move(other.hash_))
        , total_size_(other.total_size_.load(std::memory_order_relaxed))
        // T-B4: move the diagnostics cache snapshot. Snapshot is best-effort;
        // the moved-from cache will be empty, the moved-to cache will see the
        // pre-move snapshot (which may be stale if the balancer hasn't run
        // since the move). The next balancer sweep will refresh it.
        , cached_max_chain_length_(other.cached_max_chain_length_.load(std::memory_order_relaxed))
        , cached_snapshot_ns_(other.cached_snapshot_ns_.load(std::memory_order_relaxed))
        , cached_num_segments_(other.cached_num_segments_.load(std::memory_order_relaxed))
    {
        for (std::size_t i = 0; i < kMaxCachedSegments; ++i) {
            cached_per_segment_lf_[i].store(
                other.cached_per_segment_lf_[i].load(std::memory_order_relaxed),
                std::memory_order_relaxed);
        }
        other.num_segments_ = 0;
        other.segment_mask_ = 0;
        other.total_size_.store(0, std::memory_order_relaxed);
        other.cached_max_chain_length_.store(0, std::memory_order_relaxed);
        other.cached_snapshot_ns_.store(0, std::memory_order_relaxed);
        other.cached_num_segments_.store(0, std::memory_order_relaxed);
        for (std::size_t i = 0; i < kMaxCachedSegments; ++i) {
            other.cached_per_segment_lf_[i].store(0, std::memory_order_relaxed);
        }
    }

    segmented_concurrent_hash_table& operator=(segmented_concurrent_hash_table&& other) noexcept {
        if (this != &other) {
            segments_ = std::move(other.segments_);
            num_segments_ = other.num_segments_;
            segment_mask_ = other.segment_mask_;
            hash_ = std::move(other.hash_);
            total_size_.store(other.total_size_.load(std::memory_order_relaxed), std::memory_order_relaxed);
            cached_max_chain_length_.store(
                other.cached_max_chain_length_.load(std::memory_order_relaxed),
                std::memory_order_relaxed);
            cached_snapshot_ns_.store(
                other.cached_snapshot_ns_.load(std::memory_order_relaxed),
                std::memory_order_relaxed);
            cached_num_segments_.store(
                other.cached_num_segments_.load(std::memory_order_relaxed),
                std::memory_order_relaxed);
            for (std::size_t i = 0; i < kMaxCachedSegments; ++i) {
                cached_per_segment_lf_[i].store(
                    other.cached_per_segment_lf_[i].load(std::memory_order_relaxed),
                    std::memory_order_relaxed);
            }
            other.num_segments_ = 0;
            other.segment_mask_ = 0;
            other.total_size_.store(0, std::memory_order_relaxed);
            other.cached_max_chain_length_.store(0, std::memory_order_relaxed);
            other.cached_snapshot_ns_.store(0, std::memory_order_relaxed);
            other.cached_num_segments_.store(0, std::memory_order_relaxed);
            for (std::size_t i = 0; i < kMaxCachedSegments; ++i) {
                other.cached_per_segment_lf_[i].store(0, std::memory_order_relaxed);
            }
        }
        return *this;
    }

    segmented_concurrent_hash_table(const segmented_concurrent_hash_table&) = delete;
    segmented_concurrent_hash_table& operator=(const segmented_concurrent_hash_table&) = delete;

    // ========================================================================
    // Lookup — delegate to the appropriate segment
    // ========================================================================

    auto find(const Key& key) {
        return segments_[segment_for_key(key)]->find(key);
    }

    auto find(const Key& key) const {
        return segments_[segment_for_key(key)]->find(key);
    }

    /// Shared-lock embedded-node lookup (no pinning). Delegates to the
    /// appropriate segment. See concurrent_hash_table::find_embedded_shared.
    Value find_embedded_shared(const Key& key) const {
        return segments_[segment_for_key(key)]->find_embedded_shared(key);
    }

    /// Find-and-pin: delegates to the appropriate segment.
    /// See concurrent_hash_table::find_and_pin for details.
    template <typename PinFn>
    auto find_and_pin(const Key& key, PinFn&& pin_fn) {
        return segments_[segment_for_key(key)]->find_and_pin(key, std::forward<PinFn>(pin_fn));
    }

    template <typename PinFn>
    auto find_and_pin(const Key& key, PinFn&& pin_fn) const {
        return segments_[segment_for_key(key)]->find_and_pin(key, std::forward<PinFn>(pin_fn));
    }

    /// T16.4: find_and_pin with a pre-computed hash. Delegates to the
    /// appropriate segment using the hash (avoids re-hashing for segment
    /// dispatch). The hash MUST be from the same Hash function the table
    /// was constructed with.
    template <typename PinFn>
    auto find_and_pin_with_hash(const Key& key, size_type h, PinFn&& pin_fn) {
        return segments_[segment_for_hash(h)]->find_and_pin_with_hash(key, h, std::forward<PinFn>(pin_fn));
    }

    template <typename PinFn>
    auto find_and_pin_with_hash(const Key& key, size_type h, PinFn&& pin_fn) const {
        return segments_[segment_for_hash(h)]->find_and_pin_with_hash(key, h, std::forward<PinFn>(pin_fn));
    }

    /// Optimistic find-and-pin: delegates to the appropriate segment.
    template <typename PinFn, typename UnpinFn>
    auto find_and_pin_optimistic(const Key& key, PinFn&& pin_fn, UnpinFn&& unpin_fn) {
        return segments_[segment_for_key(key)]->find_and_pin_optimistic(key, std::forward<PinFn>(pin_fn), std::forward<UnpinFn>(unpin_fn));
    }

    template <typename PinFn, typename UnpinFn>
    auto find_and_pin_optimistic(const Key& key, PinFn&& pin_fn, UnpinFn&& unpin_fn) const {
        return segments_[segment_for_key(key)]->find_and_pin_optimistic(key, std::forward<PinFn>(pin_fn), std::forward<UnpinFn>(unpin_fn));
    }

    /// Lockfree find-and-pin: delegates to the appropriate segment.
    /// See concurrent_hash_table::find_and_pin_lockfree for details.
    template <typename PinFn>
    auto find_and_pin_lockfree(const Key& key, PinFn&& pin_fn) {
        return segments_[segment_for_key(key)]->find_and_pin_lockfree(key, std::forward<PinFn>(pin_fn));
    }

    template <typename PinFn>
    auto find_and_pin_lockfree(const Key& key, PinFn&& pin_fn) const {
        return segments_[segment_for_key(key)]->find_and_pin_lockfree(key, std::forward<PinFn>(pin_fn));
    }

    /// T16.4: Lockfree find-and-pin with a pre-computed hash. Delegates
    /// to the appropriate segment using the hash. Used by bulk_get to
    /// avoid re-hashing for segment dispatch + per-segment hash-table lookup.
    template <typename PinFn>
    auto find_and_pin_lockfree_with_hash(const Key& key, size_type h, PinFn&& pin_fn) {
        return segments_[segment_for_hash(h)]->find_and_pin_lockfree_with_hash(key, h, std::forward<PinFn>(pin_fn));
    }

    template <typename PinFn>
    auto find_and_pin_lockfree_with_hash(const Key& key, size_type h, PinFn&& pin_fn) const {
        return segments_[segment_for_hash(h)]->find_and_pin_lockfree_with_hash(key, h, std::forward<PinFn>(pin_fn));
    }

    bool contains(const Key& key) const {
        return segments_[segment_for_key(key)]->contains(key);
    }

    // ========================================================================
    // Insertion
    // ========================================================================

    auto insert(const Key& key, Value value) {
        auto result = segments_[segment_for_key(key)]->insert(key, std::move(value));
        if (result.second) total_size_.fetch_add(1, std::memory_order_relaxed);
        return result;
    }

    auto insert_or_assign(const Key& key, Value value) {
        auto result = segments_[segment_for_key(key)]->insert_or_assign(key, std::move(value));
        if (result.second) total_size_.fetch_add(1, std::memory_order_relaxed);
        return result;
    }

    // ========================================================================
    // Erasure
    // ========================================================================

    bool erase(const Key& key) {
        bool removed = segments_[segment_for_key(key)]->erase(key);
        if (removed) total_size_.fetch_sub(1, std::memory_order_relaxed);
        return removed;
    }

    // ========================================================================
    // Capacity
    // ========================================================================

    size_type size() const noexcept {
        return total_size_.load(std::memory_order_relaxed);
    }

    bool empty() const noexcept { return size() == 0; }

    size_type bucket_count() const noexcept {
        size_type total = 0;
        for (std::size_t i = 0; i < num_segments_; ++i) {
            total += segments_[i]->bucket_count();
        }
        return total;
    }

    // ========================================================================
    // Load factor
    // ========================================================================

    float load_factor() const noexcept {
        auto bc = bucket_count();
        return bc == 0 ? 0.0f : static_cast<float>(size()) / static_cast<float>(bc);
    }

    float max_load_factor() const noexcept { return segments_[0]->max_load_factor(); }
    void max_load_factor(float ml) noexcept {
        for (std::size_t i = 0; i < num_segments_; ++i) segments_[i]->max_load_factor(ml);
    }

    // ========================================================================
    // Rehash — per-segment, no global stall
    // ========================================================================

    /// P0-5 (T1.3): Sweep all segments and rehash any that need it.
    /// This is the legacy "scan-all-segments" entry point and is kept
    /// for compatibility with code paths that have no specific hash
    /// available (e.g. `flush()`, periodic maintenance). On the write
    /// hot path, prefer `rehash_if_needed(hash)` instead — it touches
    /// only the segment owning `hash`, eliminating the 64× single-segment
    /// rehash stall the legacy loop exhibits under high write concurrency.
    void rehash_if_needed() {
        for (std::size_t i = 0; i < num_segments_; ++i) segments_[i]->rehash_if_needed();
    }

    /// P0-5 (T1.3): Rehash only the segment owning `hash`. This is the
    /// hot-path entry point used by insert/erase — those operations
    /// already know the key's hash, so they can route directly to the
    /// owning segment and avoid stalling on the other 63 segments.
    /// Background rehash balancing (see `start_background_rehash_balancer`)
    /// is responsible for catching up segments that aren't currently
    /// being written to.
    void rehash_if_needed(size_type hash) {
        segments_[segment_for_hash(hash)]->rehash_if_needed();
    }

    /// P0-5 (T1.3): Rehash only the segment owning `key`. Convenience
    /// overload for callers that don't have a precomputed hash.
    void rehash_if_needed_for_key(const Key& key) {
        segments_[segment_for_key(key)]->rehash_if_needed();
    }

    // ========================================================================
    // Reserve — pre-allocate buckets per segment to avoid runtime rehash
    // ========================================================================

    /// Pre-allocate enough buckets for `expected_items` entries across all
    /// segments.  Distributes items evenly across segments and calls
    /// reserve() on each.  No-op if already sufficient.
    void reserve(size_type expected_items) {
        auto per_segment_items = std::max(size_type(1), expected_items / num_segments_);
        for (std::size_t i = 0; i < num_segments_; ++i) {
            segments_[i]->reserve(per_segment_items);
        }
    }

    // ========================================================================
    // Chain length diagnostics
    // ========================================================================
    //
    // T-B4 (P2-10): max_chain_length() and per_segment_load_factors() now
    // serve from the cached snapshot when available, falling back to a live
    // O(buckets) scan on the first call (before the balancer has run) or
    // when the cache has never been refreshed. Callers that require the
    // freshest possible value should call `refresh_diagnostics_cache()`
    // first, then read. The background rehash balancer (started via
    // `cache.start_background_rehash_balancer()`) refreshes the cache every
    // 1s by default.
    //
    // Live scan path (fallback) is `max_chain_length_live()`; cached path
    // is `max_chain_length()`. Both are safe to call concurrently.

    /// Live (uncached) max chain length — O(total_buckets) scan across all
    /// segments. Used as the underlying computation for the cache refresh
    /// and as a fallback when the cache is cold. Public so callers that
    /// need the freshest possible value can bypass the cache.
    size_type max_chain_length_live() const {
        size_type max_cl = 0;
        for (std::size_t i = 0; i < num_segments_; ++i) {
            max_cl = std::max(max_cl, segments_[i]->max_chain_length());
        }
        return max_cl;
    }

    /// Cached max chain length — returns the snapshot most recently
    /// computed by `refresh_diagnostics_cache()`. If the cache has never
    /// been refreshed (cached_snapshot_ns_ == 0), this falls back to a
    /// live scan and updates the cache as a side effect so subsequent
    /// calls are fast.
    size_type max_chain_length() const {
        if (cached_snapshot_ns_.load(std::memory_order_acquire) != 0) {
            return cached_max_chain_length_.load(std::memory_order_acquire);
        }
        // Cold cache — fall back to live scan and warm the cache so the
        // next scrape is fast. This is a benign race: concurrent cold
        // callers may all do the scan once, but only the last store wins
        // and subsequent calls are served from cache.
        size_type live = max_chain_length_live();
        cached_max_chain_length_.store(live, std::memory_order_relaxed);
        cached_num_segments_.store(num_segments_, std::memory_order_relaxed);
        // Also populate per-segment LF while we're at it.
        for (std::size_t i = 0; i < num_segments_; ++i) {
            float lf = segments_[i]->load_factor();
            std::uint32_t bits;
            std::memcpy(&bits, &lf, sizeof(float));
            cached_per_segment_lf_[i].store(bits, std::memory_order_relaxed);
        }
        auto now = std::chrono::steady_clock::now().time_since_epoch();
        auto ns = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
        cached_snapshot_ns_.store(ns, std::memory_order_release);
        return live;
    }

    /// T-B4 (P2-10): Compute fresh diagnostics and update the cached
    /// snapshot. O(total_buckets) scan — call only from the background
    /// rehash balancer or other low-frequency paths. After this call,
    /// `max_chain_length()` and `per_segment_load_factors()` return the
    /// freshly computed values without re-scanning. Safe to call
    /// concurrently with readers (writes are atomic).
    void refresh_diagnostics_cache() const noexcept {
        size_type max_cl = 0;
        for (std::size_t i = 0; i < num_segments_; ++i) {
            max_cl = std::max(max_cl, segments_[i]->max_chain_length());
            float lf = segments_[i]->load_factor();
            std::uint32_t bits;
            std::memcpy(&bits, &lf, sizeof(float));
            cached_per_segment_lf_[i].store(bits, std::memory_order_relaxed);
        }
        cached_max_chain_length_.store(max_cl, std::memory_order_relaxed);
        cached_num_segments_.store(num_segments_, std::memory_order_relaxed);
        auto now = std::chrono::steady_clock::now().time_since_epoch();
        auto ns = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
        // Release-store the timestamp last so readers that see a non-zero
        // snapshot_ns also see the matching cached_max_chain_length and
        // per-segment LF values (publish ordering).
        cached_snapshot_ns_.store(ns, std::memory_order_release);
    }

    /// T-B4 (P2-10): Age of the cached snapshot in milliseconds. Returns
    /// `std::numeric_limits<std::uint64_t>::max()` if the cache has never
    /// been refreshed. Operators can use this to detect a stalled balancer
    /// (age >> balancer interval typically indicates the worker thread
    /// died or `start_background_rehash_balancer()` was never called).
    std::uint64_t diagnostics_cache_age_ms() const noexcept {
        std::uint64_t cached_ns = cached_snapshot_ns_.load(std::memory_order_acquire);
        if (cached_ns == 0) {
            return std::numeric_limits<std::uint64_t>::max();
        }
        auto now = std::chrono::steady_clock::now().time_since_epoch();
        auto now_ns = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
        // Steady clock is monotonic — subtraction cannot underflow unless
        // the system clock was adjusted backwards, which steady_clock
        // ignores by design.
        return (now_ns - cached_ns) / 1'000'000u;
    }

    /// T-B4 (P2-10): Number of segments captured in the last snapshot.
    /// Useful for sanity-checking per_segment_load_factors() output size
    /// against the live num_segments_.
    std::size_t cached_segment_count() const noexcept {
        return cached_num_segments_.load(std::memory_order_acquire);
    }

    // ========================================================================
    // P1-1: Rehash diagnostics — aggregate across all segments
    // ========================================================================

    std::size_t rehash_count() const noexcept {
        std::size_t total = 0;
        for (std::size_t i = 0; i < num_segments_; ++i) {
            total += segments_[i]->rehash_count();
        }
        return total;
    }
    std::uint64_t rehash_total_time_ns() const noexcept {
        std::uint64_t total = 0;
        for (std::size_t i = 0; i < num_segments_; ++i) {
            total += segments_[i]->rehash_total_time_ns();
        }
        return total;
    }
    std::size_t rehash_migrated_items() const noexcept {
        std::size_t total = 0;
        for (std::size_t i = 0; i < num_segments_; ++i) {
            total += segments_[i]->rehash_migrated_items();
        }
        return total;
    }
    /// T11.3: Aggregate blocked-writes count across all segments.
    std::size_t rehash_blocked_writes_count() const noexcept {
        std::size_t total = 0;
        for (std::size_t i = 0; i < num_segments_; ++i) {
            total += segments_[i]->rehash_blocked_writes_count();
        }
        return total;
    }
    /// P0-4 (T1.2): Aggregate rehash_finish stall count across all segments.
    std::size_t rehash_finish_stall_count() const noexcept {
        std::size_t total = 0;
        for (std::size_t i = 0; i < num_segments_; ++i) {
            total += segments_[i]->rehash_finish_stall_count();
        }
        return total;
    }
    /// P1-5: Aggregate rehash_lockfree_fallback_count across all segments.
    /// Non-zero values indicate the lock-free read path is being degraded
    /// by rehash activity in one or more segments.
    std::size_t rehash_lockfree_fallback_count() const noexcept {
        std::size_t total = 0;
        for (std::size_t i = 0; i < num_segments_; ++i) {
            total += segments_[i]->rehash_lockfree_fallback_count();
        }
        return total;
    }
    /// P0-D: Ratio of segments with an in-progress incremental rehash,
    /// in [0.0, 1.0]. Returns 0.0 when no segments are rehashing; 1.0
    /// when every segment is rehashing. Useful as a Prometheus gauge
    /// (`lru_rehash_in_progress_ratio`) to detect sustained rehash
    /// pressure: a value near 1.0 for an extended period indicates the
    /// hash table is growing faster than the background balancer can
    /// drain migrations, which signals that writes are stalling on
    /// rehash_migrating_ too often.
    ///
    /// Works for both chain and F14 probing modes — `is_rehashing()` is
    /// the per-segment flag set by `start_incremental_rehash()` and
    /// cleared by `rehash_finish()` regardless of probing style.
    float rehash_in_progress_ratio() const noexcept {
        if (num_segments_ == 0) return 0.0f;
        std::size_t in_progress = 0;
        for (std::size_t i = 0; i < num_segments_; ++i) {
            if (segments_[i]->is_rehashing()) ++in_progress;
        }
        return static_cast<float>(in_progress) /
               static_cast<float>(num_segments_);
    }
    /// P0-4 (T1.2): Maximum backlog across segments — the worst-case
    /// high-water mark for tail-latency analysis.
    std::size_t rehash_finish_max_backlog() const noexcept {
        std::size_t max_b = 0;
        for (std::size_t i = 0; i < num_segments_; ++i) {
            max_b = std::max(max_b, segments_[i]->rehash_finish_max_backlog());
        }
        return max_b;
    }
    /// P0-5 (T1.3): Advance any in-progress incremental rehash in every
    /// segment. Called by the background rehash balancer via
    /// `mm_lru::advance_incremental_rehash()`. Each segment's
    /// `rehash_finish()` migrates at most `kRehashFinishMaxBucketsPerCall`
    /// buckets per call, so repeated invocations eventually drain the
    /// entire migration backlog without blocking writers. No-op when no
    /// rehash is in progress in any segment.
    void rehash_finish() noexcept {
        for (std::size_t i = 0; i < num_segments_; ++i) {
            segments_[i]->rehash_finish();
        }
    }
    /// T13.4: Per-segment load factor for prometheus shard label export.
    /// Returns a vector of (segment_index, load_factor) pairs.
    ///
    /// T-B4 (P2-10): Now serves from the cached snapshot (populated by
    /// `refresh_diagnostics_cache()`). If the cache is cold, falls back to
    /// a live per-segment `load_factor()` scan and warms the cache as a
    /// side effect. Use `refresh_diagnostics_cache()` first if you need
    /// the freshest values for a one-off diagnostic dump.
    std::vector<std::pair<std::size_t, float>> per_segment_load_factors() const {
        std::vector<std::pair<std::size_t, float>> result;
        result.reserve(num_segments_);
        if (cached_snapshot_ns_.load(std::memory_order_acquire) != 0) {
            // Serve from cached snapshot — O(num_segments_) atomic reads,
            // no bucket scan. This is the fast path used by prometheus
            // scrapes and diagnostics() dumps.
            for (std::size_t i = 0; i < num_segments_; ++i) {
                std::uint32_t bits =
                    cached_per_segment_lf_[i].load(std::memory_order_acquire);
                float lf;
                std::memcpy(&lf, &bits, sizeof(float));
                result.emplace_back(i, lf);
            }
        } else {
            // Cold cache — fall back to live scan. This also warms the
            // cache for subsequent reads (benign race with concurrent
            // cold callers).
            for (std::size_t i = 0; i < num_segments_; ++i) {
                float lf = segments_[i]->load_factor();
                result.emplace_back(i, lf);
                std::uint32_t bits;
                std::memcpy(&bits, &lf, sizeof(float));
                cached_per_segment_lf_[i].store(bits, std::memory_order_relaxed);
            }
            // Warm the rest of the cache so max_chain_length() also hits
            // the fast path on its next call.
            size_type max_cl = 0;
            for (std::size_t i = 0; i < num_segments_; ++i) {
                max_cl = std::max(max_cl, segments_[i]->max_chain_length());
            }
            cached_max_chain_length_.store(max_cl, std::memory_order_relaxed);
            cached_num_segments_.store(num_segments_, std::memory_order_relaxed);
            auto now = std::chrono::steady_clock::now().time_since_epoch();
            auto ns = static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
            cached_snapshot_ns_.store(ns, std::memory_order_release);
        }
        return result;
    }

    // ========================================================================
    // T13.1: Hash overload threshold / events — aggregate across segments.
    //
    // P1-13: These wrappers were missing on segmented_concurrent_hash_table,
    // which caused mm_lru::refresh_hash_stats() (and therefore
    // stats_snapshot() / prometheus_text()) to fail to compile for any
    // Segmented=true cache alias (production_cache, segmented_striped_cache,
    // f14_production_cache, etc.). The wrappers sum per-segment event
    // counts and return the maximum threshold across segments (segments
    // share the same default, but per-segment set_hash_overload_threshold
    // can diverge; reporting the max surfaces the most permissive setting).
    // ========================================================================

    /// Aggregate hash overload event count across all segments.
    std::size_t hash_overload_events() const noexcept {
        std::size_t total = 0;
        for (std::size_t i = 0; i < num_segments_; ++i) {
            total += segments_[i]->hash_overload_events();
        }
        return total;
    }

    /// Return the max per-segment overload threshold. This is informational;
    /// per-segment thresholds can diverge if the user calls
    /// segment(i).set_hash_overload_threshold() individually. The default
    /// path (cache-wide set_hash_overload_threshold below) keeps them in sync.
    float hash_overload_threshold() const noexcept {
        float max_t = 0.0f;
        for (std::size_t i = 0; i < num_segments_; ++i) {
            float t = segments_[i]->hash_overload_threshold();
            if (t > max_t) max_t = t;
        }
        return max_t;
    }

    /// Set the overload threshold on every segment. Keeps per-segment
    /// state consistent so hash_overload_threshold() reliably reports the
    /// configured value.
    void set_hash_overload_threshold(float threshold) noexcept {
        for (std::size_t i = 0; i < num_segments_; ++i) {
            segments_[i]->set_hash_overload_threshold(threshold);
        }
    }

    // ========================================================================
    // P2-4 (T2.4): Overload-callback forwarding — propagate to every
    // segment so that a cache-wide `set_async_overload_callback(true)`
    // covers the entire segmented table without requiring per-segment
    // setup. `drain_overload_callbacks()` aggregates across segments so
    // the event_drain_worker can drain the whole table in one call.
    // ========================================================================

    void set_overload_callback(std::function<void(float, float)> cb) {
        for (std::size_t i = 0; i < num_segments_; ++i) {
            segments_[i]->set_overload_callback(cb);
        }
    }

    void set_async_overload_callback(bool enabled) noexcept {
        for (std::size_t i = 0; i < num_segments_; ++i) {
            segments_[i]->set_async_overload_callback(enabled);
        }
    }

    bool async_overload_callback_enabled() const noexcept {
        return segments_[0]->async_overload_callback_enabled();
    }

    std::size_t drain_overload_callbacks() {
        std::size_t total = 0;
        for (std::size_t i = 0; i < num_segments_; ++i) {
            total += segments_[i]->drain_overload_callbacks();
        }
        return total;
    }

    std::size_t pending_overload_events() const {
        std::size_t total = 0;
        for (std::size_t i = 0; i < num_segments_; ++i) {
            total += segments_[i]->pending_overload_events();
        }
        return total;
    }

    // ========================================================================
    // Bulk operations
    // ========================================================================

    void clear() noexcept {
        for (std::size_t i = 0; i < num_segments_; ++i) segments_[i]->clear();
        total_size_.store(0, std::memory_order_relaxed);
    }

    // ========================================================================
    // Per-segment access
    // ========================================================================

    segment_type& segment(std::size_t idx) noexcept { return *segments_[idx]; }
    const segment_type& segment(std::size_t idx) const noexcept { return *segments_[idx]; }
    std::size_t num_segments() const noexcept { return num_segments_; }

    // ========================================================================
    // Static metadata (same as concurrent_hash_table for compatibility)
    // ========================================================================

    static constexpr size_type entry_overhead = segment_type::entry_overhead;

    static constexpr size_type buckets_for_items(size_type expected_items) noexcept {
        return segment_type::buckets_for_items(expected_items);
    }

    // ========================================================================
    // Hazptr / optimistic read configuration — propagate to all segments
    // ========================================================================

    void set_hazptr_read(bool enabled) noexcept {
        for (std::size_t i = 0; i < num_segments_; ++i) segments_[i]->set_hazptr_read(enabled);
    }
    bool hazptr_read_enabled() const noexcept { return segments_[0]->hazptr_read_enabled(); }

    void set_optimistic_read(bool enabled) noexcept {
        for (std::size_t i = 0; i < num_segments_; ++i) segments_[i]->set_optimistic_read(enabled);
    }
    bool optimistic_read_enabled() const noexcept { return segments_[0]->optimistic_read_enabled(); }

    // --------------------------------------------------------------------
    // T2.1 / T2.4: EBR integration — propagate to all segments
    // --------------------------------------------------------------------

    /// T2.1: Set the EBR domain for all segments. See
    /// concurrent_hash_table::set_ebr_domain for details.
    void set_ebr_domain(detail::epoch_domain* domain) noexcept {
        for (std::size_t i = 0; i < num_segments_; ++i) {
            segments_[i]->set_ebr_domain(domain);
        }
    }

    bool is_ebr_mode() const noexcept {
        return segments_[0]->is_ebr_mode();
    }

    // ========================================================================
    // Incremental rehash — propagate to all segments
    // ========================================================================

    void set_incremental_rehash(bool enabled) noexcept {
        for (std::size_t i = 0; i < num_segments_; ++i) {
            segments_[i]->set_incremental_rehash(enabled);
        }
    }
    bool incremental_rehash_enabled() const noexcept {
        // true only when all segments have it enabled
        for (std::size_t i = 0; i < num_segments_; ++i) {
            if (!segments_[i]->incremental_rehash_enabled()) return false;
        }
        return true;
    }
    /// T11.5: String-based strategy setter — propagates to all segments.
    bool set_rehash_strategy(std::string_view strategy) noexcept {
        bool ok = true;
        for (std::size_t i = 0; i < num_segments_; ++i) {
            if (!segments_[i]->set_rehash_strategy(strategy)) ok = false;
        }
        return ok;
    }
    std::string_view rehash_strategy() const noexcept {
        return segments_[0]->rehash_strategy();
    }

    /// Returns true if any segment is currently in an incremental rehash.
    bool is_rehashing() const noexcept {
        for (std::size_t i = 0; i < num_segments_; ++i) {
            if (segments_[i]->is_rehashing()) return true;
        }
        return false;
    }

    /// Sum of rehash progress across all segments currently rehashing.
    size_type rehash_progress() const noexcept {
        size_type total = 0;
        for (std::size_t i = 0; i < num_segments_; ++i) {
            total += segments_[i]->rehash_progress();
        }
        return total;
    }

    /// Sum of new bucket counts across all segments currently rehashing.
    size_type rehash_new_bucket_count() const noexcept {
        size_type total = 0;
        for (std::size_t i = 0; i < num_segments_; ++i) {
            total += segments_[i]->rehash_new_bucket_count();
        }
        return total;
    }

    /// Sum of old bucket counts across all segments currently rehashing.
    /// Used as the denominator for the migration completion fraction.
    size_type rehash_old_bucket_count() const noexcept {
        size_type total = 0;
        for (std::size_t i = 0; i < num_segments_; ++i) {
            total += segments_[i]->rehash_old_bucket_count();
        }
        return total;
    }

    // ========================================================================
    // Custom allocation — propagate to all segments
    // ========================================================================

    void set_alloc_fns(allocate_fn alloc_fn, deallocate_fn dealloc_fn) noexcept {
        for (std::size_t i = 0; i < num_segments_; ++i) segments_[i]->set_alloc_fns(alloc_fn, dealloc_fn);
    }
};

} // namespace lru::detail

// Clean up platform macro
#undef LRU_SPIN_PAUSE

#endif // LRU_DETAIL_CONCURRENT_HASH_TABLE_HPP
