// SPDX-License-Identifier: MIT
// CAS-lockfree reference counting with embedded flags and admin bits.
// Inspired by Facebook CacheLib's RefcountWithFlags.
//
// Layout (single 64-bit atomic word):
//   |-- flags (5 bits) --|-- admin_ref (3 bits) --|-- access_ref (32 bits) --|
//   bit positions (from LSB):
//     access_ref : bits [0, 31]   — 32 bits, max 4,294,967,295 concurrent handles
//     admin_ref  : bits [32, 34]  — kLinked(32), kAccessible(33), kExclusive(34)
//     flags      : bits [35, 39]  — user flags (5 bits)
//
// Design decision (single packed word, NOT split):
//   access_ref and admin_ref/flags share a single 64-bit atomic word.
//   Splitting them into two `alignas(64)` atomics was evaluated and
//   rejected because:
//     1. On x86-64/ARM64 a 64-bit CAS touches a single cache line, so
//        splitting does not reduce traffic on the dominant incRef/decRef
//        path (access_ref-only).
//     2. Splitting would double per-node memory (two 64-byte aligned
//        atomics vs. one 8-byte word). For 10M-item caches this costs
//        ~80 MB and worsens cache locality for the cold flag word.
//     3. CAS contention on hot keys is mitigated by exponential backoff
//        (refcount_cas_backoff) — the cacheline migrates between
//        contenders instead of live-spinning.
//     4. access_ref is 32 bits (4 billion handles) — overflow is
//        impossible under realistic read-heavy-write-light loads.
//
// All CAS loops use refcount_cas_backoff() to reduce cache-line
// ping-pong under 32+ thread hot-key contention.

#ifndef LRU_DETAIL_REFCOUNT_HPP
#define LRU_DETAIL_REFCOUNT_HPP

#include <atomic>
#include <cstdint>
#include <stdexcept>

// T-P2-2: CPU pause macro for CAS backoff. Reduces cache-line contention
// by inserting a hint that the CPU is in a spin-wait loop, allowing the
// hyper-threaded sibling to use more execution resources.
#if defined(_MSC_VER)
#include <intrin.h>
#define LRU_REFCOUNT_PAUSE() _mm_pause()
#elif defined(__x86_64__) || defined(__i386__)
#define LRU_REFCOUNT_PAUSE() __builtin_ia32_pause()
#elif defined(__aarch64__)
#define LRU_REFCOUNT_PAUSE() __asm__ __volatile__("yield" ::: "memory")
#else
#define LRU_REFCOUNT_PAUSE() ((void)0)
#endif

namespace lru::detail {

// ============================================================================
// Bit positions and masks
// ============================================================================

// access_ref: bits [0, 31] — 32 bits, max 4294967295
static constexpr uint64_t kAccessRefBits   = 32;
static constexpr uint64_t kAccessRefMask   = (1ULL << kAccessRefBits) - 1ULL;  // 0xFFFFFFFF
static constexpr uint64_t kAccessRefMax    = kAccessRefMask;                    // 4294967295

// admin_ref: bits [32, 34]
static constexpr uint64_t kLinkedBit       = 32;
static constexpr uint64_t kAccessibleBit   = 33;
static constexpr uint64_t kExclusiveBit     = 34;
static constexpr uint64_t kAdminRefMask     = (1ULL << kLinkedBit) | (1ULL << kAccessibleBit) | (1ULL << kExclusiveBit);

// flags: bits [35, 39] — 5 bits (kMMFlag0-2, kIsChainedItem, kHasChainedItem)
static constexpr uint64_t kFlagsShift      = 35;
static constexpr uint64_t kFlagsMask       = ((1ULL << 5) - 1ULL) << kFlagsShift;

// ============================================================================
// User flag bit positions
// ============================================================================

enum Flags : uint64_t {
    kMMFlag0        = 35,
    kMMFlag1        = 36,
    kMMFlag2        = 37,
    kIsChainedItem  = 38,
    kHasChainedItem = 39,
};

// ============================================================================
// CAS backoff helper
// ============================================================================
// After kCasBackoffThreshold consecutive CAS failures, inserts
// LRU_REFCOUNT_PAUSE() with exponential backoff (capped at
// kCasBackoffMaxPause) to reduce cache-line ping-pong on hot keys
// where many threads contend on the same refcount word.
//
// Usage in CAS loops:
//   uint32_t spins = 0;
//   while (true) {
//       // ... CAS attempt ...
//       refcount_cas_backoff(spins);
//   }
static constexpr uint32_t kCasBackoffThreshold = 4;
static constexpr uint32_t kCasBackoffMaxPause  = 64;

inline void refcount_cas_backoff(uint32_t& spin_count) {
    if (spin_count < kCasBackoffThreshold) {
        ++spin_count;
        return;
    }
    // P3-2: Clamp the shift exponent to avoid undefined behavior.
    // `1u << n` is UB when n >= width(unsigned int) (typically 32). Under
    // sustained contention spin_count can grow without bound (the only thing
    // that resets it is a successful CAS), so eventually the shift would
    // exceed 31 and trigger UBSan's "shift exponent N is too large for
    // 32-bit type 'unsigned int'" runtime error. Cap the exponent at 31
    // (the maximum valid shift for uint32_t); the resulting `pauses` value
    // (2^31) is then clamped to kCasBackoffMaxPause anyway, so the clamp
    // has no behavioral effect beyond silencing the UB.
    uint32_t shift = spin_count - kCasBackoffThreshold;
    if (shift > 31u) shift = 31u;
    uint32_t pauses = 1u << shift;
    if (pauses > kCasBackoffMaxPause) pauses = kCasBackoffMaxPause;
    for (uint32_t i = 0; i < pauses; ++i) {
        LRU_REFCOUNT_PAUSE();
    }
    ++spin_count;
}

// ============================================================================
// Result enums
// ============================================================================

enum class MarkForEvictionResult {
    kSuccess,      // exclusively marked for eviction
    kUnlinked,     // item is not in MM container (kLinked not set)
    kExclusive,    // already exclusively held (moving or eviction)
    kRefHeld,      // access_ref > 0, cannot evict
};

enum class IncResult {
    kIncOk,              // increment succeeded
    kIncFailedMoving,    // item is being moved (kExclusive + access_ref > 0)
    kIncFailedEviction,  // item is being evicted (kExclusive + access_ref == 0)
    kIncFailedOverflow,  // access_ref would overflow (saturated at max)
};

// ============================================================================
// T-G11: thread-local last-incRef result + overflow diagnostics
// ============================================================================
//
// `incRef()` sets this thread-local on every call so callers can inspect
// the most recent failure reason. The flag is reliable for the specific
// pattern in `peek_for_get_with_hash`: when the function returns empty,
// the last `incRef()` call on this thread happened during the failed pin
// attempt inside that function (no other incRef calls intervene between
// the pin attempt and the return). Callers that perform other incRef
// calls between the lookup and the check must NOT rely on this flag.
//
// `try_get()` uses it to detect overflow and retry once with yield.
// `get()` uses it to throw `refcount_overflow_exception` on overflow.
// The cache bumps its `incRef_overflow_count` counter whenever the flag
// indicates an overflow caused an empty result.
inline IncResult& tls_last_incRef_result() {
    thread_local IncResult r = IncResult::kIncOk;
    return r;
}

/// T-G11: convenience accessor — was the most recent incRef on this
/// thread an overflow?
inline bool tls_last_incRef_was_overflow() noexcept {
    return tls_last_incRef_result() == IncResult::kIncFailedOverflow;
}

/// T-G11: Reset the thread-local overflow flag. Call before a lookup
/// if you intend to check `tls_last_incRef_was_overflow()` afterwards
/// and want to avoid stale positives from prior calls.
inline void tls_clear_incRef_overflow_flag() noexcept {
    tls_last_incRef_result() = IncResult::kIncOk;
}

/// T-G11: Process-wide cumulative overflow counter. Bumped by
/// `incRef()` on every kIncFailedOverflow. Per-cache counters are
/// maintained in `cache_stats::incRef_overflow_count`, but this global
/// is exposed for runtime monitoring (e.g. crash dump attribution)
/// without holding a cache reference.
inline std::atomic<std::uint64_t>& global_incRef_overflow_count() noexcept {
    alignas(64) static std::atomic<std::uint64_t> counter{0};
    return counter;
}

// ============================================================================
// refcount_with_flags
// ============================================================================

class refcount_with_flags {
public:
    using Value = uint64_t;

    refcount_with_flags() noexcept : value_(0) {}
    explicit refcount_with_flags(Value v) noexcept : value_(v) {}

    // Non-copyable, non-movable (atomic semantics)
    refcount_with_flags(const refcount_with_flags&) = delete;
    refcount_with_flags& operator=(const refcount_with_flags&) = delete;

    // ----------------------------------------------------------------
    // G19: overflow diagnostics
    // ----------------------------------------------------------------
    // `incRef()` returns kIncFailedOverflow when access_ref saturates at
    // kAccessRefMax. Callers (e.g. read_handle ctor, try_get) produce an
    // empty handle on overflow, which is indistinguishable from a genuine
    // cache miss at the call site — risking a thundering-herd back-source
    // storm. These accessors expose the cumulative overflow count so
    // operators can detect "miss was actually an overflow" and act
    // accordingly (e.g. shed load, circuit-break the provider).
    //
    // The counter is class-wide (process-wide) rather than per-instance:
    // refcount_with_flags nodes are per-item and short-lived; an overflow
    // is a process-level anomaly (4 billion concurrent handles), so a
    // single global counter is the right granularity for diagnostics.
    // Per-cache counters live in cache_stats::incRef_overflow_count.

    /// G19: Cumulative count of incRef() overflow failures across all
    /// instances. Monotonically increasing; use reset_overflow_count()
    /// to zero it (testing only).
    static uint64_t overflow_count() noexcept {
        return overflow_count_.load(std::memory_order_relaxed);
    }

    /// G19: Reset the overflow counter to zero. Intended for tests that
    /// need a clean baseline; not for production use.
    static void reset_overflow_count() noexcept {
        overflow_count_.store(0, std::memory_order_relaxed);
    }

    // ----------------------------------------------------------------
    // access_ref operations
    // ----------------------------------------------------------------

    /// Increment access_ref. Fails if kExclusive is set or access_ref would overflow.
    /// - If kExclusive is not set and access_ref < max: increments access_ref, returns kIncOk.
    /// - If kExclusive + access_ref == 0: returns kIncFailedEviction (should not happen normally).
    /// - If kExclusive + access_ref > 0: returns kIncFailedMoving.
    /// - If access_ref >= kAccessRefMax: returns kIncFailedOverflow (saturated, no increment).
    IncResult incRef() {
        // G23: relaxed initial load — the CAS failure case below already
        // uses memory_order_acquire, which provides the required acquire
        // semantics on every retry. The initial load only seeds the loop
        // value, so acquire here is redundant (matters on ARM where acquire
        // implies a barrier). On x86 acquire is free, but relaxed is
        // strictly weaker-or-equal and correct.
        Value old = value_.load(std::memory_order_relaxed);
        uint32_t spins = 0;  // T-P2-2: CAS backoff counter
        while (true) {
            bool exclusive = (old >> kExclusiveBit) & 1ULL;
            if (exclusive) {
                Value access = old & kAccessRefMask;
                IncResult r = (access == 0) ? IncResult::kIncFailedEviction
                                            : IncResult::kIncFailedMoving;
                // T-G11: record result for caller inspection.
                tls_last_incRef_result() = r;
                return r;
            }
            Value access = old & kAccessRefMask;
            if (access >= kAccessRefMax) {
                // T-G11: record overflow + bump process-wide counter.
                tls_last_incRef_result() = IncResult::kIncFailedOverflow;
                global_incRef_overflow_count().fetch_add(1, std::memory_order_relaxed);
                // G19: bump class-level overflow counter so callers can
                // disambiguate overflow-induced empty handles (which look
                // like a miss) from genuine misses — preventing thundering-
                // herd back-source storms. Exposed via overflow_count().
                overflow_count_.fetch_add(1, std::memory_order_relaxed);
                return IncResult::kIncFailedOverflow;
            }
            Value desired = old + 1ULL;
            if (value_.compare_exchange_weak(old, desired,
                    std::memory_order_acq_rel, std::memory_order_acquire)) {
                // T-G11: record success for caller inspection.
                tls_last_incRef_result() = IncResult::kIncOk;
                return IncResult::kIncOk;
            }
            // T-P2-2: backoff to reduce cache-line ping-pong on hot keys
            refcount_cas_backoff(spins);
        }
    }

    /// Decrement access_ref. Throws on underflow. Returns new value with admin bits.
    Value decRef() {
        Value old = value_.load(std::memory_order_acquire);
        uint32_t spins = 0;  // T-P2-2: CAS backoff counter
        while (true) {
            Value access = old & kAccessRefMask;
            if (access == 0) {
                throw std::underflow_error("refcount_with_flags: access_ref underflow");
            }
            Value desired = old - 1ULL;
            if (value_.compare_exchange_weak(old, desired,
                    std::memory_order_acq_rel, std::memory_order_acquire)) {
                return desired;
            }
            // T-P2-2: backoff to reduce cache-line ping-pong on hot keys
            refcount_cas_backoff(spins);
        }
    }

    // ----------------------------------------------------------------
    // Eviction / Moving operations
    // ----------------------------------------------------------------

    /// Mark item for eviction. Only succeeds when:
    ///   - access_ref == 0 (no outstanding handles)
    ///   - kLinked is set (item is in MM container)
    ///   - kExclusive is not set (not already being moved/evicted)
    /// On success, sets kExclusive.
    MarkForEvictionResult markForEviction() {
        Value old = value_.load(std::memory_order_acquire);
        uint32_t spins = 0;  // T-P2-2: CAS backoff counter
        while (true) {
            bool linked     = (old >> kLinkedBit) & 1ULL;
            bool exclusive  = (old >> kExclusiveBit) & 1ULL;
            Value access    = old & kAccessRefMask;

            if (!linked)    return MarkForEvictionResult::kUnlinked;
            if (exclusive)  return MarkForEvictionResult::kExclusive;
            if (access > 0) return MarkForEvictionResult::kRefHeld;

            Value desired = old | (1ULL << kExclusiveBit);
            if (value_.compare_exchange_weak(old, desired,
                    std::memory_order_acq_rel, std::memory_order_acquire)) {
                return MarkForEvictionResult::kSuccess;
            }
            refcount_cas_backoff(spins);
        }
    }

    /// Mark item as moving. Only succeeds when:
    ///   - kLinked is set
    ///   - kExclusive is not set
    ///   - access_ref == 0
    /// On success, sets kExclusive AND increments access_ref (to distinguish
    /// from eviction: moving has kExclusive + access_ref >= 1).
    bool markMoving() {
        Value old = value_.load(std::memory_order_acquire);
        uint32_t spins = 0;  // T-P2-2: CAS backoff counter
        while (true) {
            bool linked    = (old >> kLinkedBit) & 1ULL;
            bool exclusive = (old >> kExclusiveBit) & 1ULL;
            Value access   = old & kAccessRefMask;

            if (!linked || exclusive || access > 0) return false;

            Value desired = (old | (1ULL << kExclusiveBit)) + 1ULL;
            if (value_.compare_exchange_weak(old, desired,
                    std::memory_order_acq_rel, std::memory_order_acquire)) {
                return true;
            }
            refcount_cas_backoff(spins);
        }
    }

    /// Clear kExclusive (used after eviction is confirmed or cancelled).
    Value unmarkForEviction() {
        Value old = value_.load(std::memory_order_acquire);
        uint32_t spins = 0;
        while (true) {
            Value desired = old & ~(1ULL << kExclusiveBit);
            if (value_.compare_exchange_weak(old, desired,
                    std::memory_order_acq_rel, std::memory_order_acquire)) {
                return desired;
            }
            refcount_cas_backoff(spins);
        }
    }

    /// Undo markMoving: decrement access_ref AND clear kExclusive.
    Value unmarkMoving() {
        Value old = value_.load(std::memory_order_acquire);
        uint32_t spins = 0;
        while (true) {
            Value desired = (old & ~(1ULL << kExclusiveBit)) - 1ULL;
            if (value_.compare_exchange_weak(old, desired,
                    std::memory_order_acq_rel, std::memory_order_acquire)) {
                return desired;
            }
            refcount_cas_backoff(spins);
        }
    }

    // ----------------------------------------------------------------
    // State queries
    // ----------------------------------------------------------------

    /// Item is marked for eviction (kExclusive set, access_ref == 0).
    bool isMarkedForEviction() const noexcept {
        Value v = value_.load(std::memory_order_acquire);
        return ((v >> kExclusiveBit) & 1ULL) && ((v & kAccessRefMask) == 0);
    }

    /// Item is being moved (kExclusive set, access_ref > 0).
    bool isMoving() const noexcept {
        Value v = value_.load(std::memory_order_acquire);
        return ((v >> kExclusiveBit) & 1ULL) && ((v & kAccessRefMask) > 0);
    }

    /// Get access_ref portion. If moving, subtract 1 (the extra ref from markMoving).
    Value getAccessRef() const noexcept {
        Value v = value_.load(std::memory_order_acquire);
        Value access = v & kAccessRefMask;
        bool exclusive = (v >> kExclusiveBit) & 1ULL;
        return (exclusive && access > 0) ? access - 1ULL : access;
    }

    /// Raw value (relaxed load).
    Value getRaw() const noexcept {
        return value_.load(std::memory_order_relaxed);
    }

    /// All refs are zero (access_ref == 0 && admin_ref == 0).
    bool isDrained() const noexcept {
        Value v = value_.load(std::memory_order_acquire);
        return (v & (kAccessRefMask | kAdminRefMask)) == 0;
    }

    // ----------------------------------------------------------------
    // kLinked (MM container membership)
    // ----------------------------------------------------------------

    void markInMMContainer() {
        Value old = value_.load(std::memory_order_acquire);
        uint32_t spins = 0;
        while (true) {
            Value desired = old | (1ULL << kLinkedBit);
            if (value_.compare_exchange_weak(old, desired,
                    std::memory_order_acq_rel, std::memory_order_acquire)) {
                return;
            }
            refcount_cas_backoff(spins);
        }
    }

    void unmarkInMMContainer() {
        Value old = value_.load(std::memory_order_acquire);
        uint32_t spins = 0;
        while (true) {
            Value desired = old & ~(1ULL << kLinkedBit);
            if (value_.compare_exchange_weak(old, desired,
                    std::memory_order_acq_rel, std::memory_order_acquire)) {
                return;
            }
            refcount_cas_backoff(spins);
        }
    }

    bool isInMMContainer() const noexcept {
        return (value_.load(std::memory_order_acquire) >> kLinkedBit) & 1ULL;
    }

    // ----------------------------------------------------------------
    // kAccessible
    // ----------------------------------------------------------------

    void markAccessible() {
        Value old = value_.load(std::memory_order_acquire);
        uint32_t spins = 0;
        while (true) {
            Value desired = old | (1ULL << kAccessibleBit);
            if (value_.compare_exchange_weak(old, desired,
                    std::memory_order_acq_rel, std::memory_order_acquire)) {
                return;
            }
            refcount_cas_backoff(spins);
        }
    }

    void unmarkAccessible() {
        Value old = value_.load(std::memory_order_acquire);
        uint32_t spins = 0;
        while (true) {
            Value desired = old & ~(1ULL << kAccessibleBit);
            if (value_.compare_exchange_weak(old, desired,
                    std::memory_order_acq_rel, std::memory_order_acquire)) {
                return;
            }
            refcount_cas_backoff(spins);
        }
    }

    bool isAccessible() const noexcept {
        return (value_.load(std::memory_order_acquire) >> kAccessibleBit) & 1ULL;
    }

    // ----------------------------------------------------------------
    // User flag operations
    // ----------------------------------------------------------------

    template <Flags flagBit>
    void setFlag() {
        static_assert(static_cast<uint64_t>(flagBit) >= kFlagsShift,
                      "flagBit must be in the flags region [35, 39]");
        Value old = value_.load(std::memory_order_acquire);
        uint32_t spins = 0;
        while (true) {
            Value desired = old | (1ULL << static_cast<uint64_t>(flagBit));
            if (value_.compare_exchange_weak(old, desired,
                    std::memory_order_acq_rel, std::memory_order_acquire)) {
                return;
            }
            refcount_cas_backoff(spins);
        }
    }

    template <Flags flagBit>
    void unSetFlag() {
        static_assert(static_cast<uint64_t>(flagBit) >= kFlagsShift,
                      "flagBit must be in the flags region [35, 39]");
        Value old = value_.load(std::memory_order_acquire);
        uint32_t spins = 0;
        while (true) {
            Value desired = old & ~(1ULL << static_cast<uint64_t>(flagBit));
            if (value_.compare_exchange_weak(old, desired,
                    std::memory_order_acq_rel, std::memory_order_acquire)) {
                return;
            }
            refcount_cas_backoff(spins);
        }
    }

    template <Flags flagBit>
    bool isFlagSet() const noexcept {
        static_assert(static_cast<uint64_t>(flagBit) >= kFlagsShift,
                      "flagBit must be in the flags region [35, 39]");
        return (value_.load(std::memory_order_acquire) >> static_cast<uint64_t>(flagBit)) & 1ULL;
    }

private:
    alignas(64) std::atomic<Value> value_;

    // G19: class-wide overflow counter. Bumped on every kIncFailedOverflow
    // return from incRef(), exposed via overflow_count() / reset_overflow_count().
    // Uses C++17 inline static so no separate out-of-class definition is needed.
    // Relaxed atomic is sufficient: the counter is a diagnostic best-effort
    // tally, not a synchronization primitive.
    static inline std::atomic<uint64_t> overflow_count_{0};
};

} // namespace lru::detail

#endif // LRU_DETAIL_REFCOUNT_HPP
