// SPDX-License-Identifier: MIT
// Epoch-Based Reclamation (EBR) mechanism for deferred memory reclamation.
//
// Design inspired by folly::EpochBasedReclamation and crossbeam-epoch (Rust).
// EBR provides faster read-path overhead than hazard pointers: only an atomic
// load + branch on the read path, vs acquire_slot + store_slot for hazptr.
//
// How it works:
//   - A global epoch counter advances periodically.
//   - Each thread entering a critical section records the epoch it observed
//     in a per-thread slot.
//   - When retiring an object, it is tagged with the current global epoch.
//   - Reclamation is safe when all retired objects have an epoch strictly
//     less than the minimum epoch observed by any active thread — this
//     guarantees no thread holds a reference to those objects.
//
// v4.2: Lock-free retire path with thread-local buffering, fixed-size slot
//       array, TLS slot caching for O(1) enter/exit, and compatibility with
//       hazptr_obj_base so items can be retired via either mechanism.

#ifndef LRU_DETAIL_EPOCH_RECLAMATION_HPP
#define LRU_DETAIL_EPOCH_RECLAMATION_HPP

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <vector>

#include "hazptr.hpp"  // hazptr_obj_base, is_hazptr_obj_v

namespace lru::detail {

// P2: Count trailing zeros in a 64-bit unsigned integer (portable).
// Used by the active-slot bitmap scan in compute_min_epoch() etc.
// Returns the index of the lowest set bit (0-based). Behavior is
// undefined when x == 0 — callers must guard with `if (bits == 0)`.
inline std::size_t lru_ctzll(uint64_t x) noexcept {
#if defined(__GNUC__) || defined(__clang__)
    return static_cast<std::size_t>(__builtin_ctzll(x));
#elif defined(_MSC_VER)
    unsigned long idx;
    _BitScanForward64(&idx, x);
    return static_cast<std::size_t>(idx);
#else
    // Fallback: simple loop (rarely used — all target compilers have
    // intrinsics above).
    std::size_t n = 0;
    while ((x & 1) == 0) { x >>= 1; ++n; }
    return n;
#endif
}

// P0-2: Force-advance policy for handling "stuck" critical sections
// (threads descheduled by the OS, GC pauses, long-running operations).
//
// When a slot's entry_time_ns is older than the configured timeout,
// compute_min_epoch() must decide what to do with the stuck slot.
// The policy trades off memory safety against reclaim progress:
//
//   kFailAdvance         — Detect stuck slots using the configured
//                          timeout, but abort the reclaim pass instead
//                          of skipping them. Safest: never causes UAF,
//                          but stuck slots block reclamation until the
//                          caller explicitly intervenes (e.g. via
//                          synchronize_epoch()). Operators observe via
//                          force_advance_count() and handle externally.
//
//   kForceAdvanceAfter5s — Skip stuck slots after a hardcoded 5s
//                          timeout. NOT the default — must be
//                          explicitly enabled via
//                          set_force_advance_policy() when the operator
//                          accepts the UAF risk in exchange for reclaim
//                          progress under stuck threads (high-throughput
//                          scenarios). May cause UAF if a stuck thread
//                          resumes and dereferences a retired pointer
//                          after reclamation.
//
//   kForceAdvanceAfterNs — Skip stuck slots after a configurable
//                          timeout (set via set_force_advance_timeout).
//                          Same UAF risk as kForceAdvanceAfter5s.
//
//   kNeverForceAdvance   — Do not detect or skip stuck slots at all.
//                          Stuck threads block reclamation indefinitely.
//                          Use only when the workload guarantees no
//                          long-running critical sections.
enum class force_advance_policy {
    kFailAdvance,
    kForceAdvanceAfter5s,
    kForceAdvanceAfterNs,
    kNeverForceAdvance,
};

// ============================================================================
// epoch_domain — manages epoch-based reclamation with a dynamically-sized
// slot array.
//
// Slots: organized in batches of kBatchSize (128). New batches are added on
// demand when all existing slots are occupied, so acquire_slot() never
// blocks — the domain grows dynamically up to kMaxBatches * kBatchSize
// (8192) slots.
//
// Each slot stores the epoch observed by the thread when it entered a critical
// section, or kInactiveEpoch when not in a critical section.
//
// Retirement is lock-free:
//   - retire() pushes to a thread-local buffer (no mutex, no CAS)
//   - When the TLS buffer is full (64 entries), it is flushed to a global
//     lock-free stack via CAS (same pattern as hazptr_domain)
//   - try_reclaim() atomically swaps the global stack, computes the minimum
//     active epoch, and batch-reclaims all entries with epoch < min_epoch
//
// Thread-local slot caching:
//   - When a thread enters a critical section, it caches its assigned slot
//     index in TLS so subsequent enter/exit from the same thread is O(1)
//
// P0-6 (T-A1): Previously slots_ was a fixed std::array<epoch_slot, 128>.
// Under 4K+ thread workloads (large HTTP servers, fiber runtimes), all 128
// slots could be occupied simultaneously, causing acquire_slot() to spin-wait
// indefinitely — a silent deadlock. The slot array now grows dynamically in
// batches (mirroring hazptr_domain's batch expansion), with a configurable
// upper bound (kMaxBatches * kBatchSize = 8192 by default). When all batches
// are exhausted, acquire_slot() falls back to bounded spin with metrics
// reporting (instead of infinite retry).
// ============================================================================

class epoch_domain {
public:
    static constexpr std::size_t kBatchSize  = 128;
    static constexpr std::size_t kMaxBatches = 64;  // Up to 8192 slots
    // Kept for backward compatibility with code that references kMaxSlots.
    // The runtime capacity is now num_batches * kBatchSize.
    static constexpr std::size_t kMaxSlots = kMaxBatches * kBatchSize;
    static constexpr uint64_t kInactiveEpoch = std::numeric_limits<uint64_t>::max();

    // Bounded retry budget before falling back to synchronize_epoch() when
    // all 8192 slots are exhausted. Mirrors hazptr's kMaxSpinRetries so EBR
    // and hazptr exhibit consistent latency under slot pressure.
    static constexpr std::size_t kMaxSpinRetries = 1024;
    // Hard cap on synchronize_epoch() fallbacks before throwing. Mirrors
    // hazptr's kMaxSyncFallbacks — beyond this the workload genuinely has
    // too many concurrent critical sections and the caller must degrade.
    static constexpr std::size_t kMaxSyncFallbacks = 64;

    // ----------------------------------------------------------------
    // Per-thread epoch slot
    // ----------------------------------------------------------------

    /// Each slot holds the epoch observed by the owning thread, or
    /// kInactiveEpoch if the thread is not in a critical section.
    /// P0-2 (T2.1): also tracks `entry_time_ns` — the steady-clock
    /// timestamp (ns since epoch) when the slot entered its current
    /// critical section. Used by `compute_min_epoch()` to detect
    /// "stuck" threads (long-running CS) and force-advance the epoch
    /// past them, preventing unbounded reclaim backlog when one
    /// thread is descheduled by the OS or stuck in a long operation.
    struct alignas(64) epoch_slot {
        std::atomic<uint64_t> local_epoch{kInactiveEpoch};
        std::atomic<bool> occupied{false};
        /// P0-2 (T2.1): Steady-clock nanoseconds when the slot entered
        /// its current critical section. 0 means "not in CS".
        /// Read by compute_min_epoch() to detect stuck slots.
        std::atomic<uint64_t> entry_time_ns{0};
    };

    // ----------------------------------------------------------------
    // Thread-local slot cache — O(1) enter/exit on the fast path
    // ----------------------------------------------------------------

    struct tls_slot_cache {
        std::size_t slot_index = static_cast<std::size_t>(-1);
        const epoch_domain* owner = nullptr;
        /// Nested critical-section depth for this thread. The slot is
        /// stamped on the outermost enter_critical() and cleared on the
        /// outermost exit_critical(); nested guards only bump this counter.
        /// This keeps the CS protected for the entire nesting scope and
        /// avoids re-reading the steady clock on every nested guard.
        std::size_t cs_depth = 0;

        ~tls_slot_cache() {
            // Return the cached slot to the global pool on thread exit.
            // Without this, the slot would remain marked as "occupied"
            // forever, eventually exhausting the 128-slot pool under
            // thread churn (which would cause acquire_slot() to spin).
            if (owner != nullptr && slot_index != static_cast<std::size_t>(-1)) {
                const_cast<epoch_domain*>(owner)->release_slot_global(slot_index);
                slot_index = static_cast<std::size_t>(-1);
                owner = nullptr;
            }
        }
    };

    static tls_slot_cache& tls_cache() {
        static thread_local tls_slot_cache instance;
        return instance;
    }

    // ----------------------------------------------------------------
    // Thread-local retire buffer — lock-free retirement
    // ----------------------------------------------------------------

    struct tls_retire_buffer {
        // R1-3: Reduced from 64 to 16. In read-heavy-write-light workloads,
        // writes are sparse so the 64-entry buffer rarely fills, meaning
        // flush_tls_buffer() (and the epoch advance it triggers) almost never
        // fires from the retire path. With 16 entries, the buffer fills 4x
        // sooner, ensuring retired objects are pushed to the global pending
        // list and become eligible for reclamation much faster. The trade-off
        // — slightly more CAS operations on the global pending list — is
        // negligible because the global push is lock-free and batched.
        static constexpr std::size_t kCapacity = 16;

        /// P1-6 (T2.3): retired_entry no longer carries the epoch — it
        /// is stored on the object itself (hazptr_obj_base::epoch_).
        /// This shrinks the entry from 16 bytes (obj + epoch) to 8
        /// bytes (obj only), halving the TLS buffer's memory footprint
        /// and improving cache locality during flush.
        struct retired_entry {
            hazptr_obj_base* obj;
        };

        retired_entry entries[kCapacity];
        std::size_t count = 0;
        epoch_domain* owner_domain = nullptr;

        bool full() const { return count >= kCapacity; }
        void clear() { count = 0; }

        ~tls_retire_buffer() {
            // Flush remaining entries to their owning domain on thread exit.
            // Without this, retired-but-unflushed objects would be silently
            // leaked when the thread terminates — the buffer is destroyed
            // without ever being pushed to the global pending list, so
            // try_reclaim() would never see them.
            if (count > 0 && owner_domain != nullptr) {
                try {
                    owner_domain->flush_tls_buffer();
                } catch (...) {
                    // Swallow exceptions during thread-local destruction:
                    // throwing from a destructor would terminate the program.
                }
            }
        }
    };

    static tls_retire_buffer& tls_retire_buf() {
        static thread_local tls_retire_buffer buf;
        return buf;
    }

    // ----------------------------------------------------------------
    // epoch_guard — RAII wrapper for critical section enter/exit
    // ----------------------------------------------------------------

    class epoch_guard {
    public:
        epoch_guard() : domain_(nullptr), active_(false) {}

        explicit epoch_guard(epoch_domain& domain)
            : domain_(&domain), active_(true)
        {
            domain_->enter_critical();
        }

        ~epoch_guard() {
            if (active_ && domain_) {
                domain_->exit_critical();
            }
        }

        // Non-copyable
        epoch_guard(const epoch_guard&) = delete;
        epoch_guard& operator=(const epoch_guard&) = delete;

        // Movable
        epoch_guard(epoch_guard&& other) noexcept
            : domain_(other.domain_), active_(other.active_)
        {
            other.active_ = false;
            other.domain_ = nullptr;
        }

        epoch_guard& operator=(epoch_guard&& other) noexcept {
            if (this != &other) {
                if (active_ && domain_) {
                    domain_->exit_critical();
                }
                domain_ = other.domain_;
                active_ = other.active_;
                other.active_ = false;
                other.domain_ = nullptr;
            }
            return *this;
        }

    private:
        epoch_domain* domain_;
        bool active_;
    };

    // ----------------------------------------------------------------
    // Constructor / Destructor
    // ----------------------------------------------------------------

    /// P0-6 (T-A1): Default constructor pre-allocates one batch (128 slots)
    /// to keep the common low-thread-count case zero-overhead. Additional
    /// batches are added on demand by acquire_slot() when all existing
    /// slots are occupied.
    epoch_domain() {
        slot_batches_.reserve(kMaxBatches);
        used_batches_.reserve(kMaxBatches);
        add_batch_unchecked();
        num_batches_.store(1, std::memory_order_release);
    }

    ~epoch_domain() {
        invalidate_tls_cache();
        reclaim_all_pending();
    }

    // Non-copyable, non-movable
    epoch_domain(const epoch_domain&) = delete;
    epoch_domain& operator=(const epoch_domain&) = delete;

    // ----------------------------------------------------------------
    // Critical section enter/exit
    // ----------------------------------------------------------------

    /// Enter a critical section: atomically load the global epoch and
    /// record it in the calling thread's epoch slot.
    /// Returns the observed epoch value.
    /// P0-2 (T2.1): also records the entry time so that stuck slots
    /// (long-running CS) can be detected and skipped by compute_min_epoch().
    uint64_t enter_critical() {
        auto& cache = tls_cache();

        // Fast path: reuse cached slot. Nested critical sections on the
        // same thread only bump the TLS depth counter — the slot stays
        // stamped until the OUTERMOST exit. This is both a correctness fix
        // (the previous code cleared the slot on the inner exit, briefly
        // exposing the outer CS to reclamation) and a performance win
        // (no steady_clock read on nested entry).
        if (cache.owner == this && cache.slot_index != static_cast<std::size_t>(-1)) {
            if (cache.cs_depth > 0) {
                ++cache.cs_depth;
                return global_epoch_.load(std::memory_order_acquire);
            }
            const uint64_t now_ns = steady_clock_now_ns();
            std::size_t idx = cache.slot_index;
            std::size_t batch  = idx / kBatchSize;
            std::size_t offset = idx % kBatchSize;
            uint64_t observed = global_epoch_.load(std::memory_order_acquire);
            slot_batches_[batch][offset].local_epoch.store(observed, std::memory_order_release);
            // P0-2 (T2.1): record entry time after epoch so the
            // force-advance check sees a consistent (epoch, time) pair.
            slot_batches_[batch][offset].entry_time_ns.store(now_ns, std::memory_order_release);
            // P2: mark this slot as active in the bitmap + counter so
            // compute_min_epoch can find it without a full O(N) scan.
            mark_slot_active(idx);
            active_cs_count_.fetch_add(1, std::memory_order_relaxed);
            cache.cs_depth = 1;
            return observed;
        }

        // Slow path: acquire a slot
        // If TLS cache belongs to a different domain, release the old slot
        if (cache.owner != nullptr && cache.owner != this) {
            const_cast<epoch_domain*>(cache.owner)->release_slot_global(cache.slot_index);
            cache.slot_index = static_cast<std::size_t>(-1);
        }
        cache.owner = this;

        std::size_t idx = acquire_slot();
        cache.slot_index = idx;
        std::size_t batch  = idx / kBatchSize;
        std::size_t offset = idx % kBatchSize;

        const uint64_t now_ns = steady_clock_now_ns();
        uint64_t observed = global_epoch_.load(std::memory_order_acquire);
        slot_batches_[batch][offset].local_epoch.store(observed, std::memory_order_release);
        slot_batches_[batch][offset].entry_time_ns.store(now_ns, std::memory_order_release);
        // P2: mark slot active (same as fast path).
        mark_slot_active(idx);
        active_cs_count_.fetch_add(1, std::memory_order_relaxed);
        cache.cs_depth = 1;
        return observed;
    }

    /// Exit a critical section: mark the calling thread's epoch slot as inactive.
    /// P0-2 (T2.1): also clears entry_time_ns so the slot is no longer
    /// considered "stuck" if it gets reused.
    /// P0-2 (T-B1): notifies any thread blocked in synchronize_epoch().
    void exit_critical() {
        auto& cache = tls_cache();
        if (cache.owner == this && cache.slot_index != static_cast<std::size_t>(-1)) {
            // Nested exit: only the outermost exit clears the slot, keeping
            // the CS protected until the outermost guard is destroyed.
            if (cache.cs_depth > 1) {
                --cache.cs_depth;
                return;
            }
            cache.cs_depth = 0;
            std::size_t idx = cache.slot_index;
            std::size_t batch  = idx / kBatchSize;
            std::size_t offset = idx % kBatchSize;
            slot_batches_[batch][offset].entry_time_ns.store(0, std::memory_order_release);
            slot_batches_[batch][offset].local_epoch.store(kInactiveEpoch, std::memory_order_release);
            // P2: clear the active-bitmap bit + decrement counter so
            // compute_min_epoch can short-circuit when no CS is active.
            clear_slot_active(idx);
            active_cs_count_.fetch_sub(1, std::memory_order_relaxed);
            notify_epoch_progress();
        }
    }

    /// Create an RAII epoch_guard for a critical section.
    epoch_guard make_guard() {
        return epoch_guard(*this);
    }

    // ----------------------------------------------------------------
    // Retire / reclaim
    // ----------------------------------------------------------------

    /// Retire a hazptr_obj_base-derived object — zero-allocation path.
    /// The object is tagged with the current global epoch (stored on
    /// the object itself via hazptr_obj_base::epoch_, see T2.3) and
    /// buffered in TLS.
    ///
    /// P1-6 (T2.3): Previously this method allocated a retired_node
    /// wrapper per retire to carry the epoch tag. Now the epoch is
    /// stored directly on the object, eliminating the allocation.
    ///
    /// P1-7 (T2.6 bugfix): Idempotent via the `retired_` atomic flag
    /// on hazptr_obj_base. Same defense as hazptr_domain::retire_obj().
    void retire_obj(hazptr_obj_base* obj) {
        if (!obj) return;

        // P1-7: Idempotent guard — only the first caller proceeds.
        // See hazptr_domain::retire_obj() for the full rationale.
        bool expected = false;
        bool already_retired = !obj->retired_.compare_exchange_strong(
            expected, true, std::memory_order_acq_rel, std::memory_order_acquire);
        if (already_retired) {
            return;
        }

        // R1-4: Warn once if the drain worker was never started.
        assert_drain_started();

        auto& buf = tls_retire_buf();
        // If the buffer currently belongs to a different domain, flush it
        // to that domain first so entries are routed to the correct pending
        // list. This preserves correctness when a thread interacts with
        // multiple domains (rare in practice; default_domain is the norm).
        if (buf.count > 0 && buf.owner_domain != nullptr && buf.owner_domain != this) {
            buf.owner_domain->flush_tls_buffer();
        }
        buf.owner_domain = this;
        // P1-6 (T2.3): Tag the object with the current epoch. The tag
        // is read by try_reclaim() to decide if the object is safe to
        // reclaim (epoch < min_epoch). Set exactly once at retire time.
        uint64_t current_epoch = global_epoch_.load(std::memory_order_acquire);
        // G23: epoch_ is a non-atomic write. This is safe because:
        // 1. retired_ CAS (above) guarantees we are the only thread reaching here.
        // 2. push_pending()'s release CAS on pending_head_ ensures this write is
        //    visible to reclaim threads that acquire-load pending_head_.
        // No atomic needed, but the happens-before chain must be preserved.
        obj->epoch_ = current_epoch;
        buf.entries[buf.count++] = {obj};

        if (buf.full()) {
            flush_tls_buffer();
        }
    }

    /// Retire a pointer for deferred deletion (template interface).
    /// For types inheriting from hazptr_obj_base, uses zero-allocation path.
    /// For other types, wraps with a heap-allocated hazptr_obj_base.
    template <typename T>
    void retire(T* ptr) {
        if (!ptr) return;

        if constexpr (is_hazptr_obj_v<T>) {
            // Zero-allocation path: T inherits from hazptr_obj_base
            auto* obj = static_cast<hazptr_obj_base*>(ptr);
            obj->reclaim_ = [](hazptr_obj_base* p) { delete static_cast<T*>(p); };
            retire_obj(obj);
        } else {
            // Heap-allocating wrapper for backward compatibility
            struct retire_wrapper : hazptr_obj_base {
                T* wrapped_ptr;
                explicit retire_wrapper(T* p) : wrapped_ptr(p) {
                    reclaim_ = [](hazptr_obj_base* p) {
                        auto* self = static_cast<retire_wrapper*>(p);
                        delete self->wrapped_ptr;
                        delete self;
                    };
                }
            };
            retire_obj(new retire_wrapper(ptr));
        }
    }

    /// Flush the TLS retire buffer to the global pending list.
    /// P1-6 (T2.3): No longer allocates retired_node wrappers. The
    /// hazptr_obj_base objects are chained directly via their embedded
    /// `next_` pointer, and the epoch tag is read from the object's
    /// `epoch_` field. This eliminates one heap allocation + free per
    /// retired object, halving allocator pressure on the evict path.
    void flush_tls_buffer() {
        auto& buf = tls_retire_buf();
        if (buf.count == 0) return;

        // Build a chain of hazptr_obj_base objects directly — no
        // retired_node wrapper allocation. The chain is built in
        // reverse order (LIFO) so that entries[0] is at the tail;
        // order within a single flush batch does not matter for
        // correctness, only the epoch tags matter.
        hazptr_obj_base* chain_head = nullptr;
        for (std::size_t i = 0; i < buf.count; ++i) {
            hazptr_obj_base* obj = buf.entries[i].obj;
            obj->next_ = chain_head;
            chain_head = obj;
        }

        buf.clear();

        // CAS-loop prepend to global pending list (same pattern as hazptr_domain)
        push_pending(chain_head);

        // Advance global epoch periodically
        maybe_advance_epoch();

        // R1-2: Capacity-based epoch advance. When the pending list
        // exceeds 25% of the reclaim threshold, force-advance the epoch
        // so that older retired objects become eligible for reclamation
        // sooner. Without this, in read-heavy-write-light workloads the
        // epoch may not advance until the time-based trigger fires (200ms),
        // during which the pending list can grow significantly under bursty
        // eviction. This trigger ensures the epoch advances promptly when
        // memory pressure starts building, not just on a timer.
        maybe_capacity_advance();

        // P1-3 (T1.4): Auto-reclaim when pending_count exceeds threshold.
        // Bound worst-case memory pressure under bursty retire workloads
        // without adding lock contention to the steady-state hot path.
        // The CAS flag inside maybe_auto_reclaim prevents stampede — at
        // most one thread performs the synchronous try_reclaim; others
        // observe the flag and bail out, leaving reclaim to the background
        // worker or the next flush_tls_buffer() call.
        maybe_auto_reclaim();
    }

    /// Attempt to reclaim pending retired pointers whose epoch
    /// is guaranteed to be no longer observable by any active thread.
    ///
    /// @param batch_size T-P2-3 (R-7): Maximum number of objects to
    ///        process per call. 0 means unlimited (drain all — original
    ///        behavior, used by destructor / explicit full-drain paths).
    ///        When > 0, at most `batch_size` objects are inspected; any
    ///        unprocessed or still-observable objects are pushed back to
    ///        the pending list for the next call. This bounds the
    ///        worst-case latency of a single reclaim pass, eliminating
    ///        the spikes seen when the pending list is large (e.g.,
    ///        after a burst of evictions). The list is reversed before
    ///        processing so oldest objects (smallest epoch, most likely
    ///        to be reclaimable) are inspected first — this prevents
    ///        starvation of old objects under continuous retire pressure.
    /// @return Number of objects actually reclaimed.
    ///
    /// P1-6 (T2.3): Walks hazptr_obj_base chain directly (via next_)
    /// and reads epoch from hazptr_obj_base::epoch_. No retired_node
    /// wrappers to delete — reclamation invokes obj->reclaim_(obj)
    /// which both destroys the object and frees its memory.
    std::size_t try_reclaim(std::size_t batch_size = 0) {
        // Flush TLS buffer first
        flush_tls_buffer();

        // Atomically swap the pending list
        hazptr_obj_base* head = pending_head_.exchange(nullptr, std::memory_order_acq_rel);
        if (!head) return 0;

        // T-P2-3 (R-7): For incremental reclaim (batch_size > 0), reverse
        // the list so oldest objects (smallest epoch) are processed first.
        // The pending list is LIFO (push_pending prepends), so without
        // reversal the newest objects would be inspected first and oldest
        // objects could starve under continuous retire pressure.
        if (batch_size > 0) {
            head = reverse_pending_list(head);
        }

        // Compute the minimum epoch observed by all active threads
        uint64_t min_epoch = compute_min_epoch();

        // Walk the list and reclaim safe entries
        hazptr_obj_base* new_head = nullptr;
        hazptr_obj_base* new_tail = nullptr;  // T-P2-3 (R-7): track tail for append
        std::size_t reclaimed_count = 0;
        std::size_t remaining_count = 0;
        std::size_t processed = 0;  // T-P2-3 (R-7): batch counter

        hazptr_obj_base* curr = head;
        while (curr) {
            // T-P2-3 (R-7): Stop after inspecting batch_size objects.
            if (batch_size > 0 && processed >= batch_size) {
                break;
            }

            hazptr_obj_base* next = curr->next_;
            curr->next_ = nullptr;

            if (curr->epoch_ < min_epoch) {
                // Safe to reclaim — no active thread can hold a reference.
                // reclaim_ destroys the object AND frees its memory (via
                // delete or slab allocator).
                if (curr->reclaim_) {
                    curr->reclaim_(curr);
                }
                ++reclaimed_count;
            } else {
                // Still potentially observable — re-chain for later reclamation
                // T-P2-3 (R-7): append to tail (instead of prepend to head)
                // to preserve epoch ordering within the re-chained list.
                if (!new_head) {
                    new_head = curr;
                    new_tail = curr;
                } else {
                    new_tail->next_ = curr;
                    new_tail = curr;
                }
                ++remaining_count;
            }
            curr = next;
            ++processed;
        }

        // T-P2-3 (R-7): Re-chain any unprocessed objects (the `curr`
        // pointer is at the first unprocessed object, or null if we
        // processed the entire list). Append them to the re-chained
        // protected entries so a single push_pending() call returns
        // everything to the global pending list.
        if (curr) {
            if (!new_head) {
                new_head = curr;
                new_tail = curr;
                while (new_tail->next_) {
                    new_tail = new_tail->next_;
                }
            } else {
                new_tail->next_ = curr;
                while (new_tail->next_) {
                    new_tail = new_tail->next_;
                }
            }
            for (hazptr_obj_base* p = curr; p; p = p->next_) {
                ++remaining_count;
            }
        }

        // Push remaining entries back to the global list
        if (new_head) {
            push_pending(new_head);
        }

        // Update statistics
        reclaim_total_.fetch_add(reclaimed_count, std::memory_order_relaxed);
        pending_count_.store(remaining_count, std::memory_order_release);

        return reclaimed_count;
    }

    /// Return the current number of pending (unreclaimed) retired objects.
    std::size_t pending_count() const noexcept {
        return pending_count_.load(std::memory_order_acquire);
    }

    /// Return the cumulative number of objects reclaimed since the domain
    /// was created.
    std::size_t reclaim_total() const noexcept {
        return reclaim_total_.load(std::memory_order_acquire);
    }

    /// P1-3 (T1.4): Return the current auto-reclaim threshold. When
    /// `pending_count()` exceeds this value, the next `flush_tls_buffer()`
    /// synchronously invokes `try_reclaim()` to drain the pending list.
    /// Default is 65536 — high enough to avoid reclaim thrashing under
    /// steady-state retire rates, low enough to bound worst-case memory
    /// pressure under bursty retire spikes.
    std::size_t reclaim_threshold() const noexcept {
        return reclaim_threshold_.load(std::memory_order_acquire);
    }

    /// P1-3 (T1.4): Set the auto-reclaim threshold. Set to
    /// `std::numeric_limits<std::size_t>::max()` to effectively disable
    /// auto-reclaim (only background / explicit reclaims will run).
    /// Thread-safe: may be called concurrently with retire().
    void set_reclaim_threshold(std::size_t threshold) noexcept {
        reclaim_threshold_.store(threshold, std::memory_order_release);
    }

    /// P1-3 (T1.4): Number of times auto-reclaim was triggered because
    /// `pending_count` exceeded the threshold. Useful for sizing the
    /// threshold against actual retire pressure.
    std::size_t reclaim_auto_triggered_count() const noexcept {
        return reclaim_auto_triggered_count_.load(std::memory_order_acquire);
    }

    /// P0-2 (T2.1): Set the maximum critical-section duration before
    /// a slot is considered "stuck" and force-advanced past. Default
    /// is 30 seconds. Set to 0 to disable force-advance (preserves
    /// original behavior — stuck threads block reclaim indefinitely).
    ///
    /// When a slot is skipped due to timeout, compute_min_epoch()
    /// excludes its local_epoch from the min computation, allowing
    /// objects in older epochs to be reclaimed despite the stuck slot.
    /// The trade-off: if a stuck thread resumes after the reclaim and
    /// dereferences a retired pointer, UAF may result. Only enable this
    /// when the alternative (unbounded reclaim backlog) is worse.
    ///
    /// P0-2 (refactor): The policy is now governed by `force_advance_policy_`.
    /// This timeout is consulted only when the policy is
    /// `kForceAdvanceAfterNs` (or `kFailAdvance` for detection). Use
    /// `set_force_advance_policy()` to select the active policy.
    void set_epoch_force_advance_timeout(std::chrono::nanoseconds timeout) noexcept {
        epoch_force_advance_timeout_ns_.store(
            static_cast<uint64_t>(timeout.count()),
            std::memory_order_release);
    }

    /// P0-2: Alias for `set_epoch_force_advance_timeout` — shorter name
    /// aligned with the `force_advance_policy` API.
    void set_force_advance_timeout(std::chrono::nanoseconds timeout) noexcept {
        set_epoch_force_advance_timeout(timeout);
    }

    /// T-G4: Select the force-advance policy. See `force_advance_policy`
    /// for the semantics of each value. Default is `kFailAdvance` (safe;
    /// never UAF). Switch to `kForceAdvanceAfter5s` only when the operator
    /// accepts the UAF risk in exchange for reclaim progress under stuck
    /// threads.
    void set_force_advance_policy(force_advance_policy policy) noexcept {
        force_advance_policy_.store(
            static_cast<uint32_t>(policy), std::memory_order_release);
    }

    /// P0-2: Query the active force-advance policy.
    force_advance_policy get_force_advance_policy() const noexcept {
        return static_cast<force_advance_policy>(
            force_advance_policy_.load(std::memory_order_acquire));
    }

    /// P0-2 (T2.1): Query the current force-advance timeout.
    std::chrono::nanoseconds epoch_force_advance_timeout() const noexcept {
        return std::chrono::nanoseconds(
            epoch_force_advance_timeout_ns_.load(std::memory_order_acquire));
    }

    /// P0-2 (T2.1): Number of times compute_min_epoch() skipped a
    /// stuck slot. A steadily-increasing count indicates threads are
    /// entering CSes longer than the configured timeout — either the
    /// timeout is too low, or the workload has genuine long-running
    /// operations that should be split.
    std::size_t force_advance_count() const noexcept {
        return force_advance_count_.load(std::memory_order_acquire);
    }

    /// P0-2 (T-B1): Number of individual slot observations skipped by
    /// `compute_min_epoch()` because `entry_time_ns` was older than
    /// the configured force-advance timeout. Distinct from
    /// `force_advance_count()` — the latter counts *reclaim passes*
    /// affected by at least one stuck slot; this counter counts
    /// *individual stuck slots* across all reclaim passes, giving
    /// better visibility into how many distinct threads are stuck.
    /// A fast-growing count indicates persistent blockage; a slowly-
    /// growing count indicates brief stalls.
    std::size_t epoch_stale_count() const noexcept {
        return epoch_stale_count_.load(std::memory_order_acquire);
    }

    /// P0-6 (T-A1): Current slot capacity (number of allocated batches
    /// × kBatchSize). Capacity grows on demand from 128 → 8192 as
    /// threads enter critical sections concurrently. Useful for sizing
    /// the workload against the upper bound.
    std::size_t slot_capacity() const noexcept {
        return num_batches_.load(std::memory_order_acquire) * kBatchSize;
    }

    /// P0-6 (T-A1): Number of times acquire_slot() exhausted all
    /// 8192 slots and had to yield. Non-zero under normal operation
    /// indicates the workload has 8000+ simultaneous critical sections
    /// — likely a runaway thread-spawn pattern rather than a normal
    /// cache workload.
    std::size_t slot_exhaustion_count() const noexcept {
        return slot_exhaustion_count_.load(std::memory_order_acquire);
    }

    /// Number of times acquire_slot() exceeded kMaxSpinRetries and fell
    /// back to synchronize_epoch() to wait for a quiescent point. Sustained
    /// non-zero values indicate the workload has 8000+ concurrent critical
    /// sections for extended periods.
    std::size_t sync_fallback_count() const noexcept {
        return sync_fallback_count_.load(std::memory_order_acquire);
    }

    /// Obtain the default global epoch domain.
    static epoch_domain& default_domain() {
        static epoch_domain domain;
        return domain;
    }

    /// R1-4: Mark that the background drain worker has been started.
    /// Called by unified_cache::start_event_drain(). If never called,
    /// assert_drain_started() will log a warning on the first retire,
    /// alerting operators that retired objects will accumulate without
    /// reclamation (eventually causing OOM in long-running processes).
    void mark_drain_started() noexcept {
        drain_worker_started_.store(true, std::memory_order_release);
    }

    /// R1-4: Check whether the drain worker has been started.
    bool is_drain_started() const noexcept {
        return drain_worker_started_.load(std::memory_order_acquire);
    }

    /// R1-4: Assert the drain worker is running. Called on the retire
    /// path to warn (via stderr) if the drain worker was never started.
    /// Uses an atomic flag to ensure the warning prints at most once.
    void assert_drain_started() noexcept {
        if (drain_warn_printed_.exchange(true, std::memory_order_acq_rel)) {
            return;
        }
        if (!drain_worker_started_.load(std::memory_order_acquire)) {
            std::fprintf(stderr,
                "[lru::epoch_domain] WARNING: background drain worker not started. "
                "Retired objects will accumulate without reclamation, eventually "
                "causing OOM. Call cache.start_event_drain() at startup, or use "
                "production_cache / striped_cache / safe_cache which auto-start it.\n");
        }
    }

    /// Return the current global epoch.
    uint64_t current_epoch() const noexcept {
        return global_epoch_.load(std::memory_order_acquire);
    }

    /// P0-2 (T-B1): Block the calling thread until all threads that
    /// were in a critical section at call time have exited, and at
    /// least one epoch advance has occurred. Returns true on success
    /// (a quiescent point was reached), false on timeout.
    ///
    /// Implementation: bumps `sync_target_epoch_` to `current+1`, then
    /// waits on a condition variable. Each `exit_critical()` call
    /// notifies the CV; the predicate checks whether all active slots
    /// have a local_epoch >= sync_target_epoch_ (meaning they entered
    /// CS *after* our call) OR are inactive.
    ///
    /// Use cases:
    ///   - Shutdown: ensures all in-flight CSes complete before destruction
    ///   - Bulk evict: ensures retired objects are reclaimable before OOM
    ///   - Tests: deterministic quiescent points for assertions
    ///
    /// Cost: O(N_slots) per predicate check. Under heavy load this may
    /// take up to the configured timeout; under normal operation it
    /// completes in <10ms because CSes are short.
    ///
    /// Note: does NOT advance the epoch itself — it waits for natural
    /// advancement. Use `try_reclaim()` to drain pending objects after.
    bool synchronize_epoch(std::chrono::milliseconds timeout = std::chrono::seconds(5)) {
        // Snapshot the target epoch: any thread that enters a CS *after*
        // this point will observe an epoch >= target. We then wait
        // until no thread holds a local_epoch < target.
        const uint64_t target_epoch = global_epoch_.load(std::memory_order_acquire) + 1;

        // Optimistic check: maybe all threads already moved past target.
        if (all_slots_at_or_past_epoch(target_epoch)) {
            return true;
        }

        // Slow path: wait on the CV. exit_critical() notifies after
        // updating its slot's local_epoch.
        std::unique_lock<std::mutex> lock(sync_mutex_);
        auto deadline = std::chrono::steady_clock::now() + timeout;
        sync_target_epoch_.store(target_epoch, std::memory_order_release);
        bool ok = sync_cv_.wait_until(lock, deadline, [this, target_epoch] {
            return all_slots_at_or_past_epoch(target_epoch);
        });
        // Reset the target so future callers can reuse the CV with a
        // fresh target. If multiple callers race here, the last writer
        // wins — but the wait predicate captures target_epoch by value,
        // so each caller still observes its own quiescent point.
        sync_target_epoch_.store(0, std::memory_order_release);
        return ok;
    }

private:
    /// P0-2 (T-B1): Used by `exit_critical()` to notify any thread
    /// blocked in `synchronize_epoch()` that a slot has been released.
    /// Called without holding sync_mutex_ — notify_one is safe to call
    /// without the lock (it may be a spurious wakeup, which is filtered
    /// by the predicate).
    void notify_epoch_progress() {
        // Only notify if someone is actually waiting. Reading
        // sync_target_epoch_ is sufficient — non-zero means a sync call
        // is in progress. This avoids the wakeup cost on the common
        // exit_critical() path where no one is waiting.
        if (sync_target_epoch_.load(std::memory_order_acquire) != 0) {
            std::lock_guard<std::mutex> lock(sync_mutex_);
            sync_cv_.notify_all();
        }
    }

    /// P0-2 (T-B1): Predicate used by synchronize_epoch(): returns
    /// true iff all active slots have local_epoch >= target_epoch
    /// (meaning they entered their CS after our sync call) or are
    /// inactive (local_epoch == kInactiveEpoch).
    bool all_slots_at_or_past_epoch(uint64_t target_epoch) const {
        // P2: Fast path — no active CS means no slot can hold back
        // the target epoch.
        if (active_cs_count_.load(std::memory_order_acquire) == 0) {
            return true;
        }
        // P2: Slow path — bitmap scan.
        const std::size_t nb = num_batches_.load(std::memory_order_acquire);
        const std::size_t total_slots = nb * kBatchSize;
        const std::size_t words_to_scan = (total_slots + 63) / 64;

        for (std::size_t w = 0; w < words_to_scan; ++w) {
            uint64_t bits = active_bitmap_[w].load(std::memory_order_acquire);
            if (bits == 0) continue;

            while (bits) {
                std::size_t bit = lru_ctzll(bits);
                std::size_t idx = w * 64 + bit;
                bits &= bits - 1;

                std::size_t b = idx / kBatchSize;
                std::size_t i = idx % kBatchSize;
                if (!used_batches_[b][i].load(std::memory_order_acquire)) continue;
                uint64_t local = slot_batches_[b][i].local_epoch.load(std::memory_order_acquire);
                if (local == kInactiveEpoch) continue;
                if (local < target_epoch) return false;
            }
        }
        return true;
    }

    // ---- Retired node type ------------------------------------------------
    // P1-6 (T2.3): retired_node wrapper removed. hazptr_obj_base now
    // carries the epoch directly (hazptr_obj_base::epoch_), and the
    // objects are chained via their embedded next_ pointer. This
    // eliminates one heap allocation + free per retired object.

    // ---- Slot management ---------------------------------------------------

    std::size_t acquire_slot() {
        // P0-6 (T-A1): Dynamic batch expansion. Scan existing slots; if all
        // are occupied, expand under the mutex and retry. Capacity grows
        // from 128 → 8192 across 64 batches, eliminating the infinite-spin
        // failure mode under 4K+ concurrent threads.
        std::size_t spin_retries = 0;
        std::size_t sync_fallbacks = 0;
        while (true) {
            std::size_t nb = num_batches_.load(std::memory_order_acquire);
            std::size_t total = nb * kBatchSize;

            for (std::size_t i = 0; i < total; ++i) {
                std::size_t batch  = i / kBatchSize;
                std::size_t offset = i % kBatchSize;
                bool expected = false;
                if (used_batches_[batch][offset].compare_exchange_strong(
                        expected, true,
                        std::memory_order_acquire)) {
                    return i;
                }
            }

            // All existing slots occupied — expand under the mutex.
            std::lock_guard<std::mutex> lock(expand_mutex_);

            std::size_t nb2 = num_batches_.load(std::memory_order_acquire);
            std::size_t total2 = nb2 * kBatchSize;

            // Pick up any slots added by another thread while we waited.
            for (std::size_t i = total; i < total2; ++i) {
                std::size_t batch  = i / kBatchSize;
                std::size_t offset = i % kBatchSize;
                bool expected = false;
                if (used_batches_[batch][offset].compare_exchange_strong(
                        expected, true,
                        std::memory_order_acquire)) {
                    return i;
                }
            }

            if (nb2 >= kMaxBatches) {
                // All 8192 slots exhausted — extreme scenario.
                // Bump exhaustion metric and apply bounded yield-and-retry,
                // then fall back to synchronize_epoch() to wait for a
                // quiescent point (which lets threads release their slots).
                // After exhausting sync fallbacks, throw so the caller can
                // degrade instead of hanging forever — matches hazptr
                // behavior (see hazptr.hpp acquire_slot).
                slot_exhaustion_count_.fetch_add(1, std::memory_order_relaxed);
                if (++spin_retries <= kMaxSpinRetries) {
                    std::this_thread::yield();
                    continue;
                }
                sync_fallback_count_.fetch_add(1, std::memory_order_relaxed);
                if (++sync_fallbacks > kMaxSyncFallbacks) {
                    throw std::runtime_error(
                        "epoch_domain::acquire_slot: exhausted all " +
                        std::to_string(kMaxBatches * kBatchSize) +
                        " slots after " + std::to_string(kMaxSyncFallbacks) +
                        " synchronize_epoch fallbacks; workload has too many "
                        "concurrent live critical sections");
                }
                // Wait briefly for a quiescent point so other threads can
                // release their slots. Short timeout keeps the caller from
                // blocking too long before retrying the slot scan.
                (void)synchronize_epoch(std::chrono::milliseconds(100));
                spin_retries = 0;
                std::this_thread::yield();
                continue;
            }

            add_batch_unchecked();
            num_batches_.store(nb2 + 1, std::memory_order_release);
        }
    }

    void release_slot_global(std::size_t idx) {
        if (idx == static_cast<std::size_t>(-1)) return;
        std::size_t batch  = idx / kBatchSize;
        std::size_t offset = idx % kBatchSize;
        // P2: If the slot was active (thread exited without calling
        // exit_critical), clear the bitmap bit and decrement the counter.
        // The local_epoch store below makes the slot inactive for
        // compute_min_epoch, but the bitmap bit would remain set,
        // causing the slow path to unnecessarily visit this slot.
        // Use a CAS to only decrement if we actually clear the bit.
        std::size_t word = idx / 64;
        std::size_t bit  = idx % 64;
        uint64_t mask = ~(uint64_t{1} << bit);
        uint64_t old = active_bitmap_[word].fetch_and(mask, std::memory_order_relaxed);
        if (old & (uint64_t{1} << bit)) {
            // Bit was set — the slot was active. Decrement the counter
            // to keep active_cs_count_ consistent.
            active_cs_count_.fetch_sub(1, std::memory_order_relaxed);
        }
        slot_batches_[batch][offset].local_epoch.store(kInactiveEpoch, std::memory_order_release);
        slot_batches_[batch][offset].entry_time_ns.store(0, std::memory_order_release);
        used_batches_[batch][offset].store(false, std::memory_order_release);
    }

    void invalidate_tls_cache() {
        auto& cache = tls_cache();
        if (cache.owner == this) {
            release_slot_global(cache.slot_index);
            cache.slot_index = static_cast<std::size_t>(-1);
            cache.owner = nullptr;
            cache.cs_depth = 0;
        }
    }

    // ---- Epoch computation -------------------------------------------------

    /// Compute the minimum epoch observed by all currently active threads.
    /// Returns global_epoch + 1 if no threads are active (meaning all
    /// retired objects can be reclaimed).
    ///
    /// P0-2 (T2.1): slots whose entry_time_ns is older than the
    /// configured force-advance timeout are treated as "stuck" and
    /// skipped. This unblocks reclamation of objects in older epochs
    /// even when a thread is descheduled by the OS or stuck in a long
    /// operation. The trade-off: if a stuck thread resumes after the
    /// reclaim and dereferences a retired pointer, UAF may result. The
    /// default timeout is 30 seconds — well beyond normal CS duration,
    /// so this only triggers in genuine stuck-thread scenarios.
    ///
    /// P0-2 (policy refactor): Behavior is now governed by
    /// `force_advance_policy_`. See `force_advance_policy` enum doc.
    /// - `kFailAdvance`: returns 0 (no object reclaimable) if any stuck
    ///   slot is detected, so the caller can recover explicitly.
    /// - `kForceAdvanceAfter5s` / `kForceAdvanceAfterNs`: skips stuck
    ///   slots so reclamation proceeds (UAF risk if stuck thread resumes).
    /// - `kNeverForceAdvance`: includes all active slots in min_epoch.
    uint64_t compute_min_epoch() const {
        uint64_t min_epoch = std::numeric_limits<uint64_t>::max();
        const auto policy = get_force_advance_policy();

        // Determine the timeout to apply based on the active policy.
        uint64_t timeout_ns = 0;
        switch (policy) {
            case force_advance_policy::kForceAdvanceAfter5s:
                timeout_ns = 5000000000ULL;  // 5s hardcoded
                break;
            case force_advance_policy::kForceAdvanceAfterNs:
            case force_advance_policy::kFailAdvance:
                // kFailAdvance uses the same timeout for *detection*,
                // but aborts (returns 0) instead of skipping the slot.
                timeout_ns = epoch_force_advance_timeout_ns_.load(std::memory_order_acquire);
                break;
            case force_advance_policy::kNeverForceAdvance:
            default:
                timeout_ns = 0;
                break;
        }
        const uint64_t now_ns = (timeout_ns != 0) ? steady_clock_now_ns() : 0;
        bool skipped_stuck = false;

        // P2: Fast path — no thread is in a critical section, so all
        // retired objects are safe to reclaim. This is the common case
        // in read-heavy-write-light workloads where reclaim runs on the
        // write path after readers have exited their CS.
        if (active_cs_count_.load(std::memory_order_acquire) == 0) {
            return global_epoch_.load(std::memory_order_acquire) + 1;
        }

        // P2: Slow path — iterate the active-slot bitmap word by word.
        // Each 64-bit word covers 64 consecutive slots; zero words are
        // skipped entirely. This reduces the worst-case scan from
        // O(num_batches * kBatchSize) = O(8192) atomic loads to
        // O(kBitmapWords) = O(128) word loads + O(active_cs_count) slot
        // loads. In typical read-heavy workloads with few concurrent
        // CS, the bitmap is very sparse, so most words are zero.
        const std::size_t nb = num_batches_.load(std::memory_order_acquire);
        const std::size_t total_slots = nb * kBatchSize;
        const std::size_t words_to_scan = (total_slots + 63) / 64;

        for (std::size_t w = 0; w < words_to_scan; ++w) {
            uint64_t bits = active_bitmap_[w].load(std::memory_order_acquire);
            if (bits == 0) continue;  // entire 64-slot span is inactive

            // Iterate over set bits in this word.
            while (bits) {
                std::size_t bit = lru_ctzll(bits);
                std::size_t idx = w * 64 + bit;
                bits &= bits - 1;  // clear lowest set bit

                std::size_t b = idx / kBatchSize;
                std::size_t i = idx % kBatchSize;
                // Double-check used flag — the bitmap may lag slightly
                // behind used_batches_ due to relaxed ordering, but the
                // local_epoch load below is the authoritative check.
                if (!used_batches_[b][i].load(std::memory_order_acquire)) continue;
                uint64_t local = slot_batches_[b][i].local_epoch.load(std::memory_order_acquire);
                if (local == kInactiveEpoch) continue;
                // P0-2 (T2.1): skip stuck slots — entry_time_ns older
                // than the force-advance timeout.
                if (timeout_ns != 0 && now_ns != 0) {
                    uint64_t entry = slot_batches_[b][i].entry_time_ns.load(std::memory_order_acquire);
                    if (entry != 0 && (now_ns - entry) > timeout_ns) {
                        epoch_stale_count_.fetch_add(
                            1, std::memory_order_relaxed);
                        // P0-2 (policy) / G4: kFailAdvance aborts the
                        // reclaim pass on the FIRST stuck slot. Early-
                        // return 0 immediately instead of continuing to
                        // scan remaining slots: the result is determin-
                        // istically 0 regardless, and scanning wastes
                        // CPU under cgroup CPU limits / GC / SIGSTOP
                        // where many slots may appear stuck at once.
                        if (policy == force_advance_policy::kFailAdvance) {
                            force_advance_count_.fetch_add(
                                1, std::memory_order_relaxed);
                            return 0;
                        }
                        skipped_stuck = true;
                        continue;
                    }
                }
                if (local < min_epoch) {
                    min_epoch = local;
                }
            }
        }

        // kFailAdvance is handled by the early-return inside the scan
        // loop above, so no post-loop check is needed here. Branches
        // below handle the remaining policies (kForceAdvanceAfter5s /
        // kForceAdvanceAfterNs / kNeverForceAdvance).

        // If no threads are active (or all active threads were skipped
        // as stuck), all objects are safe to reclaim.
        if (min_epoch == std::numeric_limits<uint64_t>::max()) {
            if (skipped_stuck) {
                // P0-2 (T2.1): we skipped at least one stuck slot —
                // bump the force-advance counter so users can observe
                // how often this safety net fires.
                // T-P3-7: force_advance_count_ is now mutable.
                force_advance_count_.fetch_add(
                    1, std::memory_order_relaxed);
            }
            return global_epoch_.load(std::memory_order_acquire) + 1;
        }
        if (skipped_stuck) {
            // T-P3-7: force_advance_count_ is now mutable.
            force_advance_count_.fetch_add(
                1, std::memory_order_relaxed);
        }
        return min_epoch;
    }

    // ---- Lock-free pending list --------------------------------------------

    /// T-P2-3 (R-7): Reverse a singly-linked pending list in place.
    /// Used by incremental try_reclaim(batch_size > 0) so that oldest
    /// objects (at the tail of the LIFO pending list) are processed
    /// first, preventing starvation under continuous retire pressure.
    /// Only touches next_ pointers — no callbacks, no allocations.
    /// Returns the new head (previously the last node).
    static hazptr_obj_base* reverse_pending_list(hazptr_obj_base* head) {
        hazptr_obj_base* prev = nullptr;
        hazptr_obj_base* curr = head;
        while (curr) {
            hazptr_obj_base* next = curr->next_;
            curr->next_ = prev;
            prev = curr;
            curr = next;
        }
        return prev;
    }

    /// CAS-loop prepend: atomically insert a chain at the front of the
    /// global pending list. Finds the tail of the chain and links it to
    /// the current head, then CAS-swaps the head pointer. No mutex required.
    /// P1-3 (T1.4): also bumps `pending_count_` by the chain length so
    /// `maybe_auto_reclaim()` can detect threshold overflow without
    /// walking the pending list.
    ///
    /// P1-6 (T2.3): chain_head is a hazptr_obj_base* chain (linked via
    /// next_), not a retired_node* chain. No wrapper allocations.
    void push_pending(hazptr_obj_base* chain_head) {
        // Find the tail of the chain and count its length.
        hazptr_obj_base* chain_tail = chain_head;
        std::size_t chain_len = 1;
        while (chain_tail->next_) {
            chain_tail = chain_tail->next_;
            ++chain_len;
        }

        // CAS-loop: prepend chain to the global pending list
        chain_tail->next_ = pending_head_.load(std::memory_order_acquire);
        while (!pending_head_.compare_exchange_weak(
                    chain_tail->next_, chain_head,
                    std::memory_order_release,
                    std::memory_order_acquire)) {
            // CAS failed — chain_tail->next_ was updated with the current head.
            // Retry with the new pending_head_ value.
        }

        pending_count_.fetch_add(chain_len, std::memory_order_release);
    }

    // ---- Epoch advancement -------------------------------------------------

    /// Periodically advance the global epoch.
    /// Called after flushing the TLS retire buffer.
    void maybe_advance_epoch() {
        uint64_t current = global_epoch_.load(std::memory_order_acquire);

        // P2: Fast path — no active CS means all threads have trivially
        // "caught up", so we can advance the epoch immediately.
        if (active_cs_count_.load(std::memory_order_acquire) == 0) {
            if (global_epoch_.compare_exchange_strong(
                    current, current + 1,
                    std::memory_order_acq_rel)) {
                last_advance_time_ns_.store(steady_clock_now_ns(),
                                            std::memory_order_release);
            }
            return;
        }

        // P2: Slow path — use the active-slot bitmap to find slots that
        // are both used and in a CS, then check their local_epoch.
        bool all_caught_up = true;
        const std::size_t nb = num_batches_.load(std::memory_order_acquire);
        const std::size_t total_slots = nb * kBatchSize;
        const std::size_t words_to_scan = (total_slots + 63) / 64;

        for (std::size_t w = 0; w < words_to_scan && all_caught_up; ++w) {
            uint64_t bits = active_bitmap_[w].load(std::memory_order_acquire);
            if (bits == 0) continue;

            while (bits) {
                std::size_t bit = lru_ctzll(bits);
                std::size_t idx = w * 64 + bit;
                bits &= bits - 1;

                std::size_t b = idx / kBatchSize;
                std::size_t i = idx % kBatchSize;
                if (!used_batches_[b][i].load(std::memory_order_acquire)) continue;
                uint64_t local = slot_batches_[b][i].local_epoch.load(std::memory_order_acquire);
                if (local != kInactiveEpoch && local < current) {
                    all_caught_up = false;
                    break;
                }
            }
        }

        if (all_caught_up) {
            // Try to advance. CAS avoids lost updates if another thread
            // is also calling maybe_advance_epoch concurrently.
            if (global_epoch_.compare_exchange_strong(
                    current, current + 1,
                    std::memory_order_acq_rel)) {
                last_advance_time_ns_.store(steady_clock_now_ns(),
                                            std::memory_order_release);
            }
        }
    }

    /// R1-2: Capacity-based epoch advance. When the pending retire list
    /// exceeds 25% of the reclaim threshold, force-advances the epoch
    /// so that older retired objects become eligible for reclamation.
    /// This complements maybe_time_advance() (timer-based) and
    /// maybe_advance_epoch() (flush-based) by adding a pressure-based
    /// trigger that fires immediately when memory starts accumulating,
    /// rather than waiting for the next timer tick.
    void maybe_capacity_advance() {
        const std::size_t threshold = reclaim_threshold_.load(std::memory_order_acquire);
        if (threshold == std::numeric_limits<std::size_t>::max()) return;
        const std::size_t pending = pending_count_.load(std::memory_order_acquire);
        // Trigger at 25% of threshold — early enough to prevent buildup,
        // late enough to avoid epoch thrashing under normal retire rates.
        const std::size_t trigger = threshold / 4;
        if (pending <= trigger) return;

        // Force-advance: compute_min_epoch() will handle stuck slots
        // via the force-advance timeout, so this is safe.
        uint64_t current = global_epoch_.load(std::memory_order_acquire);
        if (global_epoch_.compare_exchange_strong(
                current, current + 1,
                std::memory_order_acq_rel)) {
            last_advance_time_ns_.store(steady_clock_now_ns(),
                                        std::memory_order_release);
        }
    }

public:
    /// C-2 fix: Time-based epoch advance. Called by the background drain
    /// worker on each tick. If enough time has elapsed since the last epoch
    /// advance, force-advances the epoch even if not all threads have caught
    /// up — stuck threads are handled by compute_min_epoch()'s force-advance
    /// timeout. This unblocks reclamation in read-heavy-write-light workloads
    /// where the TLS retire buffer rarely fills (so maybe_advance_epoch() is
    /// never called from the retire path).
    void maybe_time_advance() {
        const uint64_t interval = time_advance_interval_ns_.load(std::memory_order_acquire);
        if (interval == 0) return;  // disabled

        const uint64_t now = steady_clock_now_ns();
        const uint64_t last = last_advance_time_ns_.load(std::memory_order_acquire);
        if (last == 0) {
            // First call — just record the timestamp.
            last_advance_time_ns_.store(now, std::memory_order_release);
            return;
        }
        if ((now - last) < interval) return;  // not enough time yet

        // Try to advance the epoch. We use a CAS loop: if all threads have
        // caught up, a normal advance suffices. If not, we still advance —
        // compute_min_epoch() will skip stuck slots via the force-advance
        // timeout, so reclamation can proceed safely for objects in older
        // epochs.
        uint64_t current = global_epoch_.load(std::memory_order_acquire);
        if (global_epoch_.compare_exchange_strong(
                current, current + 1,
                std::memory_order_acq_rel)) {
            last_advance_time_ns_.store(now, std::memory_order_release);
        }
    }

    /// C-2 fix: Set the time-based epoch advance interval. Set to 0 to
    /// disable time-based advance (reverts to flush-only behavior).
    void set_time_advance_interval(std::chrono::nanoseconds interval) noexcept {
        time_advance_interval_ns_.store(
            static_cast<uint64_t>(interval.count()),
            std::memory_order_release);
    }

    /// C-2 fix: Query the current time-based advance interval.
    std::chrono::nanoseconds time_advance_interval() const noexcept {
        return std::chrono::nanoseconds(
            time_advance_interval_ns_.load(std::memory_order_acquire));
    }

    // ---- Reclaim all remaining (destructor) --------------------------------

    void reclaim_all_pending() {
        flush_tls_buffer();

        hazptr_obj_base* head = pending_head_.exchange(nullptr, std::memory_order_acq_rel);
        std::size_t reclaimed = 0;
        while (head) {
            hazptr_obj_base* next = head->next_;
            if (head->reclaim_) {
                head->reclaim_(head);
            }
            ++reclaimed;
            head = next;
        }
        if (reclaimed > 0) {
            reclaim_total_.fetch_add(reclaimed, std::memory_order_relaxed);
            pending_count_.store(0, std::memory_order_release);
        }
    }

    /// P1-3 (T1.4): If pending_count exceeds the configured threshold,
    /// synchronously trigger try_reclaim() to drain the pending list.
    /// Uses a CAS flag to prevent stampede — at most one thread at a
    /// time performs the synchronous reclaim; concurrent flush_tls_buffer()
    /// calls observe the flag and skip. Recursion from within try_reclaim
    /// is safe: try_reclaim() calls flush_tls_buffer() first, but the
    /// buffer is empty by the time it runs (we just flushed it), so the
    /// recursive call is a no-op.
    void maybe_auto_reclaim() {
        const std::size_t threshold = reclaim_threshold_.load(std::memory_order_acquire);
        if (threshold == std::numeric_limits<std::size_t>::max()) return;
        const std::size_t pending = pending_count_.load(std::memory_order_acquire);
        if (pending <= threshold) return;

        bool expected = false;
        if (!reclaim_in_progress_.compare_exchange_strong(
                expected, true, std::memory_order_acq_rel)) {
            return;
        }

        reclaim_auto_triggered_count_.fetch_add(1, std::memory_order_relaxed);
        try_reclaim();

        reclaim_in_progress_.store(false, std::memory_order_release);
    }

    /// P0-2 (T2.1): Read the steady clock and return nanoseconds since
    /// the clock's epoch. Used to timestamp slot entry so stuck slots
    /// can be detected by compute_min_epoch(). Inline + thread-safe.
    static uint64_t steady_clock_now_ns() noexcept {
        auto now = std::chrono::steady_clock::now();
        return static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                now.time_since_epoch()).count());
    }

    /// P0-6 (T-A1): Append a new batch of kBatchSize slots without
    /// synchronization. Caller must hold `expand_mutex_` (or be the
    /// sole thread during construction). Mirrors the same pattern in
    /// `hazptr_domain::add_batch_unchecked()`. Each batch is a unique_ptr
    /// to keep the slot array stable in memory (reallocation would
    /// break the alignment guarantees of `epoch_slot`).
    void add_batch_unchecked() {
        slot_batches_.push_back(std::make_unique<epoch_slot[]>(kBatchSize));
        // Mark the new batch's slots as kInactiveEpoch and entry_time_ns=0
        // (epoch_slot's default constructor already does this, but being
        // explicit guards against future changes to epoch_slot's defaults).
        auto& batch = slot_batches_.back();
        for (std::size_t i = 0; i < kBatchSize; ++i) {
            batch[i].local_epoch.store(kInactiveEpoch, std::memory_order_relaxed);
            batch[i].entry_time_ns.store(0, std::memory_order_relaxed);
        }
        // Add a parallel array of "used" flags for this batch. Each flag
        // is an atomic<bool> in its own cache line to avoid false sharing
        // between slots — same trick hazptr_domain uses for slot occupancy.
        used_batches_.push_back(std::make_unique<std::atomic<bool>[]>(kBatchSize));
        auto& used = used_batches_.back();
        for (std::size_t i = 0; i < kBatchSize; ++i) {
            used[i].store(false, std::memory_order_relaxed);
        }
    }

    // ---- Data members ------------------------------------------------------

    // Global epoch counter
    alignas(64) std::atomic<uint64_t> global_epoch_{0};

    // P0-6 (T-A1): Dynamic slot batches (128 slots per batch, up to 64
    // batches = 8192 slots). Slot pointers are stored as unique_ptr arrays
    // so the slot addresses remain stable when new batches are appended —
    // concurrent readers (e.g. compute_min_epoch) can safely traverse the
    // vector's current contents while another thread appends a batch.
    std::vector<std::unique_ptr<epoch_slot[]>> slot_batches_;
    std::vector<std::unique_ptr<std::atomic<bool>[]>> used_batches_;
    alignas(64) std::atomic<std::size_t> num_batches_{0};
    alignas(64) std::mutex expand_mutex_;

    // P2: Active-CS tracking — avoids O(N) full slot scan in
    // compute_min_epoch / maybe_advance_epoch / all_slots_at_or_past_epoch
    // when no thread is in a critical section (the common case in
    // read-heavy-write-light workloads where reclaim runs on the write
    // path after readers have exited their CS).
    //
    // active_cs_count_: total number of threads currently in a CS.
    //   0  → all scan functions short-circuit to "no active threads".
    //   >0 → fall back to the word-level bitmap below.
    //
    // active_bitmap_: 1 bit per slot, packed 64-per-word.
    //   Set in enter_critical(), cleared in exit_critical() /
    //   release_slot_global(). Scan functions iterate only words with
    //   non-zero bits (O(N/64) worst case instead of O(N)).
    static constexpr std::size_t kBitmapWords =
        kMaxBatches * kBatchSize / 64;  // 128 words for 8192 slots
    alignas(64) std::atomic<std::size_t> active_cs_count_{0};
    alignas(64) std::array<std::atomic<uint64_t>, kBitmapWords> active_bitmap_{};

    /// P2: Set the bit for slot `idx` (called by enter_critical).
    void mark_slot_active(std::size_t idx) noexcept {
        std::size_t word = idx / 64;
        std::size_t bit  = idx % 64;
        active_bitmap_[word].fetch_or(uint64_t{1} << bit,
                                      std::memory_order_relaxed);
    }

    /// P2: Clear the bit for slot `idx` (called by exit_critical /
    /// release_slot_global).
    void clear_slot_active(std::size_t idx) noexcept {
        std::size_t word = idx / 64;
        std::size_t bit  = idx % 64;
        active_bitmap_[word].fetch_and(~(uint64_t{1} << bit),
                                       std::memory_order_relaxed);
    }

    // Lock-free global pending retire list.
    // P1-6 (T2.3): Points to hazptr_obj_base chain (linked via next_),
    // not retired_node wrappers.
    std::atomic<hazptr_obj_base*> pending_head_{nullptr};

    // Statistics counters (alignas to avoid false sharing)
    alignas(64) std::atomic<std::size_t> pending_count_{0};
    alignas(64) std::atomic<std::size_t> reclaim_total_{0};

    // P1-3 (T1.4): Auto-reclaim threshold and stampede guard.
    // When `pending_count_` exceeds `reclaim_threshold_`, the next
    // flush_tls_buffer() synchronously invokes try_reclaim(). The
    // `reclaim_in_progress_` CAS flag ensures at most one thread at a
    // time performs the synchronous reclaim. `reclaim_auto_triggered_count_`
    // tracks how many times this auto-reclaim fired.
    // C-2 fix: lowered default from 65536 to 4096. In read-heavy-write-light
    // workloads, writes are sparse so the TLS retire buffer rarely fills,
    // and the threshold was almost never hit — retired objects accumulated
    // to 65536 before any reclaim fired, causing memory bloat.
    alignas(64) std::atomic<std::size_t> reclaim_threshold_{4096};
    alignas(64) std::atomic<std::size_t> reclaim_auto_triggered_count_{0};
    alignas(64) std::atomic<bool> reclaim_in_progress_{false};

    // P0-2 (T2.1): Force-advance timeout and counter. When a slot's
    // entry_time_ns is older than `epoch_force_advance_timeout_ns_`,
    // compute_min_epoch() treats the slot as "stuck" and skips it,
    // allowing reclamation of objects in older epochs despite the
    // stuck thread.
    // C-2 fix: lowered default from 30s to 5s. 30s was too long for
    // read-heavy workloads where retired objects accumulated for 30s
    // before the safety net fired, causing sustained memory pressure.
    alignas(64) std::atomic<uint64_t> epoch_force_advance_timeout_ns_{5000000000ULL};
    alignas(64) mutable std::atomic<std::size_t> force_advance_count_{0};

    // T-G4: Active force-advance policy. Default is `kFailAdvance` to
    // guarantee safety (never UAF); `kForceAdvanceAfter5s` requires
    // explicit opt-in via set_force_advance_policy() and is intended
    // for high-throughput scenarios that accept UAF risk in exchange
    // for reclaim progress under stuck threads. Stuck slots block
    // reclamation under the default rather than risk UAF. Operators
    // observing `force_advance_count_` growth should switch to
    // `kForceAdvanceAfter5s` ONLY when they accept the UAF risk.
    alignas(64) std::atomic<uint32_t> force_advance_policy_{
        static_cast<uint32_t>(force_advance_policy::kFailAdvance)};

    // C-2 fix: Time-based epoch advancement. In read-heavy-write-light
    // workloads, maybe_advance_epoch() is only called after flush_tls_buffer()
    // (which fires when the TLS retire buffer fills). With sparse writes,
    // the buffer rarely fills, so the epoch never advances and retired
    // objects can never be reclaimed (their epoch == global_epoch).
    // The background drain worker calls maybe_time_advance() on each tick.
    // R1-1: Lowered default from 1s to 200ms. In read-heavy workloads,
    // retired objects accumulated for up to 1s before the epoch advanced,
    // causing sustained memory pressure under bursty eviction patterns.
    // 200ms bounds worst-case reclamation latency to ~200ms while keeping
    // CPU overhead negligible (one atomic CAS per tick).
    alignas(64) std::atomic<uint64_t> last_advance_time_ns_{0};
    alignas(64) std::atomic<uint64_t> time_advance_interval_ns_{200000000ULL}; // 200ms default (R1-1)

    // P0-2 (T-B1): Per-slot stale observation counter. Increments
    // each time compute_min_epoch() observes a slot whose entry_time_ns
    // exceeds the force-advance timeout. Higher granularity than
    // `force_advance_count_` (which counts reclaim passes, not slots).
    alignas(64) mutable std::atomic<std::size_t> epoch_stale_count_{0};

    // P0-6 (T-A1): Bumped each time acquire_slot() finds all 8192 slots
    // occupied. Non-zero indicates the workload has outgrown the upper
    // bound (8000+ simultaneous critical sections).
    alignas(64) std::atomic<std::size_t> slot_exhaustion_count_{0};

    // Bumped each time acquire_slot() exhausts kMaxSpinRetries and falls
    // back to synchronize_epoch() to wait for a quiescent point. Mirrors
    // hazptr's sync_fallback_count_ for behavioral parity.
    alignas(64) std::atomic<std::size_t> sync_fallback_count_{0};

    // R1-4: Drain worker started flag. Set by mark_drain_started() when
    // the cache's background drain worker is launched. Checked by
    // assert_drain_started() on the retire path to warn operators if
    // the drain worker was never started (retired objects will leak).
    alignas(64) std::atomic<bool> drain_worker_started_{false};
    alignas(64) std::atomic<bool> drain_warn_printed_{false};

    // P0-2 (T-B1): synchronize_epoch() support — condition variable
    // notified by exit_critical() when a slot is released. The target
    // epoch is set by synchronize_epoch() and cleared after the wait
    // completes; non-zero means a sync call is in progress.
    // Aligned to 64 to avoid false sharing with the slot vectors above.
    alignas(64) std::mutex sync_mutex_;
    alignas(64) std::condition_variable sync_cv_;
    alignas(64) std::atomic<uint64_t> sync_target_epoch_{0};
};

} // namespace lru::detail

#endif // LRU_DETAIL_EPOCH_RECLAMATION_HPP
