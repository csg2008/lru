// SPDX-License-Identifier: MIT
// Lightweight Hazard Pointer mechanism for protecting raw pointer accesses
// in concurrent cache iterators and drain_access_ring operations.
//
// Design inspired by Maged Michael's hazard pointer methodology and
// Facebook CacheLib's reclamation infrastructure.
//
// v4.2: Lock-free retire path with thread-local buffering (no mutex),
//       hazptr_obj_base for zero-allocation retirement, and thread-local
//       slot cache for O(1) acquire/release on the fast path.

#ifndef LRU_DETAIL_HAZPTR_HPP
#define LRU_DETAIL_HAZPTR_HPP

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

namespace lru::detail {

// ============================================================================
// hazptr_obj_base — base class for objects that can be retired without
// heap allocation. Objects inheriting from this class use the zero-alloc
// retirement path: the embedded next_ pointer forms the retire list chain.
//
// P1-6 (T2.3): Embedded epoch field for EBR. Previously, epoch_domain
// allocated a separate `retired_node` wrapper per retired object to
// carry the epoch tag — this added a heap allocation + free to every
// retire path, doubling allocator pressure under high eviction rates.
// Embedding `epoch_` directly on hazptr_obj_base eliminates the
// wrapper allocation entirely: epoch_domain now chains hazptr_obj_base
// objects via the existing `next_` pointer and reads the epoch from
// `epoch_`. The 8-byte-per-item memory overhead is negligible relative
// to the typical Key+Value payload (often 50-200+ bytes per item), and
// is recovered many times over by eliminating the 32-48 byte
// retired_node wrapper allocation per retire.
//
// The `epoch_` field is only read/written by the EBR retire/reclaim
// path; it is not used by hazptr_domain (which uses only `next_` and
// `reclaim_`). An object is retired via either hazptr OR EBR, never
// both simultaneously, so there is no aliasing concern.
//
// P1-7 (T2.6 bugfix): `retired_` atomic flag for idempotent retire.
// Under high-concurrency eviction workloads, the same cache_item can
// be observed by two threads racing on `evict_lru()`: Thread A wins
// `markForEviction()` and proceeds to retire, but `evict_lru()` calls
// `unmarkForEviction()` BEFORE `retire()`, briefly reopening the
// window for Thread B to also `markForEviction()` and `retire()` the
// same object. Without an idempotency guard, both threads push the
// same pointer to their TLS buffers — `try_reclaim()` then frees the
// object twice (double-free) or dereferences `next_` after free
// (heap-use-after-free). The `retired_` flag is set atomically via
// CAS (false → true); the first caller wins and the second caller
// becomes a no-op. The 1-byte flag is padded to 8 bytes by the
// struct's natural alignment, adding 8 bytes per cache_item —
// negligible relative to Key+Value payload.
// ============================================================================

struct hazptr_obj_base {
    void (*reclaim_)(hazptr_obj_base* obj) = nullptr;
    hazptr_obj_base* next_ = nullptr;
    /// P1-6 (T2.3): Epoch tag assigned by epoch_domain::retire_obj().
    /// 0 means "not retired via EBR" (default). Read by
    /// epoch_domain::try_reclaim() to decide if the object is safe
    /// to reclaim (epoch < min_epoch). Set exactly once at retire
    /// time, never mutated afterward.
    std::uint64_t epoch_ = 0;
    /// P1-7 (T2.6 bugfix): Idempotent retire guard. Set atomically
    /// via CAS (false → true) inside `hazptr_domain::retire_obj()` and
    /// `epoch_domain::retire_obj()` before pushing to the TLS buffer.
    /// If already true, retire_obj() is a no-op — preventing the
    /// double-retire / heap-use-after-free observed under high-
    /// concurrency eviction workloads.
    std::atomic<bool> retired_{false};
};

// ============================================================================
// Type trait: detect whether T inherits from hazptr_obj_base
// ============================================================================

template <typename T, typename = void>
struct is_hazptr_obj : std::false_type {};

template <typename T>
struct is_hazptr_obj<T, std::void_t<decltype(static_cast<const T*>(nullptr)->reclaim_)>> 
    : std::is_base_of<hazptr_obj_base, T> {};

template <typename T>
inline constexpr bool is_hazptr_obj_v = is_hazptr_obj<T>::value;

// ============================================================================
// hazptr_domain — manages a hazard pointer registry with deferred reclamation
//
// Slots are organized in batches of kBatchSize (128). New batches are added
// on demand when all existing slots are occupied, so acquire_slot() never
// throws — the domain grows dynamically up to kMaxBatches * kBatchSize slots.
//
// Retirement is lock-free:
//   - retire() pushes to a thread-local buffer (no mutex, no CAS)
//   - When the TLS buffer is full (64 entries), it is flushed to a global
//     lock-free stack via CAS
//   - try_reclaim() atomically swaps the global stack, builds a sorted
//     protected-pointer vector, and batch-reclaims unprotected entries
//   - Objects inheriting from hazptr_obj_base use zero-allocation retirement
//     (the embedded next_ pointer is the list link)
//
// Thread-local slot cache:
//   - When a thread releases a slot, it is cached thread-locally so the next
//     acquire from the same thread is O(1) instead of requiring a linear scan.
// ============================================================================

class hazptr_domain {
public:
    static constexpr std::size_t kBatchSize  = 128;
    static constexpr std::size_t kMaxBatches = 64;  // Up to 8192 slots

    // T-O1: Bounded retry policy for acquire_slot() when all 8192 slots
    // are exhausted. Previously the loop yielded forever (silent
    // deadlock under 10K+ live handles). Now:
    //   - kMaxSpinRetries: yield-and-retry attempts before falling back
    //     to a synchronous try_reclaim() to drain pending retired
    //     objects (frees memory pressure; may indirectly release slots
    //     held by reclaim-related paths).
    //   - kMaxSyncFallbacks: hard cap on sync-reclaim fallbacks. After
    //     this many fallbacks ALL fail to free a slot, acquire_slot()
    //     returns `npos` (P0-3: previously threw std::runtime_error,
    //     which terminated the program because read_handle's constructor
    //     is noexcept). Callers must check the return value against
    //     `npos` and degrade gracefully (e.g. read_handle produces an
    //     empty handle, matching the refcount-overflow behavior).
    static constexpr std::size_t kMaxSpinRetries  = 1024;
    static constexpr std::size_t kMaxSyncFallbacks = 64;

    // P0-3: Sentinel value returned by acquire_slot() when all slots
    // are exhausted and the bounded retry budget is spent. Callers
    // must check the return value against `npos` and degrade
    // gracefully — typically by producing an empty read_handle so the
    // caller sees a cache miss instead of a program termination.
    static constexpr std::size_t npos = static_cast<std::size_t>(-1);

    // P0-3: Upper bound on the number of slot batches. Defaults to
    // `kMaxBatches` (64 batches × 128 slots = 8192 slots) but can be
    // raised at runtime via `set_max_slots()` up to `kAbsoluteMaxBatches`
    // (512 batches × 128 slots = 65536 slots) for workloads with
    // exceptionally high reader fan-out.
    static constexpr std::size_t kAbsoluteMaxBatches = 512;  // 65536 slots

    // ----------------------------------------------------------------
    // Thread-local slot cache — avoids linear scan on the fast path
    // ----------------------------------------------------------------

    /// Per-thread cache of recently released hazard pointer slots.
    /// On thread exit the destructor returns cached slots to the global pool,
    /// preventing slot leakage in applications with high thread churn.
    struct tls_slot_cache {
        // H-8 fix: increased from 2 to 8. With only 2 slots, nested
        // hazptr_holder usage (e.g., curr/next traversal) immediately
        // exhausted the cache and triggered the O(N) slow path in
        // acquire_slot(). 8 slots cover common nesting depths while
        // keeping per-thread memory negligible (8 * sizeof(size_t) = 64B).
        static constexpr int kCacheSize = 8;
        std::size_t slots[kCacheSize];
        int count = 0;
        const hazptr_domain* owner = nullptr;

        ~tls_slot_cache() {
            // Return any cached slots to the global pool on thread exit.
            // Without this, slots would remain marked as "used" forever,
            // eventually exhausting the 8192-slot pool under thread churn.
            if (owner != nullptr && count > 0) {
                for (int i = 0; i < count; ++i) {
                    const_cast<hazptr_domain*>(owner)->release_slot_global(slots[i]);
                }
                count = 0;
                owner = nullptr;
            }
        }
    };

    /// Return the thread-local slot cache instance.
    static tls_slot_cache& tls_cache() {
        static thread_local tls_slot_cache instance;
        return instance;
    }

    /// Invalidate the TLS cache for the calling thread.
    void invalidate_tls_cache() {
        auto& cache = tls_cache();
        if (cache.owner == this) {
            for (int i = 0; i < cache.count; ++i) {
                release_slot_global(cache.slots[i]);
            }
            cache.count = 0;
            cache.owner = nullptr;
        }
    }

    // ----------------------------------------------------------------
    // Thread-local retire buffer — lock-free retirement
    // ----------------------------------------------------------------

    /// Per-thread buffer for retiring pointers without mutex contention.
    /// When full, the entire buffer is flushed to the global pending list
    /// via a CAS operation (lock-free).
    /// On thread exit the destructor flushes any remaining entries to the
    /// owning domain, preventing silent memory leaks in applications with
    /// high thread churn (each short-lived thread could otherwise leak up
    /// to kCapacity-1 retired objects).
    struct tls_retire_buffer {
        static constexpr std::size_t kCapacity = 64;
        hazptr_obj_base* entries[kCapacity];
        std::size_t count = 0;
        hazptr_domain* owner_domain = nullptr;

        bool full() const { return count >= kCapacity; }
        void clear() { count = 0; }

        ~tls_retire_buffer() {
            // Flush remaining entries to their owning domain on thread exit.
            // Without this, retired-but-unflushed objects would be silently
            // leaked when the thread terminates (the buffer is destroyed
            // without ever being pushed to the global pending list).
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

    /// Return the thread-local retire buffer instance.
    static tls_retire_buffer& tls_retire_buf() {
        static thread_local tls_retire_buffer buf;
        return buf;
    }

    // ----------------------------------------------------------------
    // Constructor / Destructor
    // ----------------------------------------------------------------

    explicit hazptr_domain(std::size_t initial_slots = kBatchSize) {
        std::size_t initial_batches = (initial_slots + kBatchSize - 1) / kBatchSize;
        if (initial_batches == 0) initial_batches = 1;
        if (initial_batches > kAbsoluteMaxBatches) initial_batches = kAbsoluteMaxBatches;

        // P0-3: Reserve capacity for the absolute max so add_batch_unchecked()
        // never needs to grow the vectors (which would invalidate references
        // held by concurrent readers on the slow path).
        slot_batches_.reserve(kAbsoluteMaxBatches);
        used_batches_.reserve(kAbsoluteMaxBatches);
        // G16: reserve for the free-list "next" link batches (same lifetime
        // as slot_batches_/used_batches_; never reallocated after this).
        next_free_batches_.reserve(kAbsoluteMaxBatches);
        for (std::size_t b = 0; b < initial_batches; ++b) {
            add_batch_unchecked();
        }
        num_batches_.store(initial_batches, std::memory_order_release);
        // P0-3: default batch limit preserves pre-P0-3 capacity (8192 slots).
        max_batches_limit_.store(kMaxBatches, std::memory_order_release);
    }

    ~hazptr_domain() {
        invalidate_tls_cache();
        // Reclaim any remaining retired objects
        reclaim_all_pending();
    }

    // Non-copyable, non-movable
    hazptr_domain(const hazptr_domain&) = delete;
    hazptr_domain& operator=(const hazptr_domain&) = delete;

    // ----------------------------------------------------------------
    // Slot acquisition / release
    // ----------------------------------------------------------------

    /// Acquire a hazard pointer slot.
    /// Fast path: check the thread-local slot cache for O(1) acquisition.
    /// Slow path: linear scan of all slots, then domain expansion.
    ///
    /// T-O1: When all 8192 slots are exhausted, the loop is now bounded:
    /// after kMaxSpinRetries (1024) yield-and-retry cycles, it falls back
    /// to a synchronous try_reclaim() to drain pending retired objects
    /// (bumps hazptr_sync_fallback_count_). After kMaxSyncFallbacks (64)
    /// such fallbacks all fail to free a slot, it throws
    /// std::runtime_error so the caller can degrade gracefully instead of
    /// deadlocking silently. Operators detect the condition via
    /// hazptr_sync_fallback_count() / slot_exhaustion_count().
    std::size_t acquire_slot() {
        // Fast path: check TLS cache
        auto& cache = tls_cache();
        if (cache.owner == this && cache.count > 0) {
            --cache.count;
            return cache.slots[cache.count];
        }

        // If the TLS cache belongs to a different domain, flush stale entries
        if (cache.owner != nullptr && cache.owner != this) {
            auto* old_domain = const_cast<hazptr_domain*>(cache.owner);
            for (int i = 0; i < cache.count; ++i) {
                old_domain->release_slot_global(cache.slots[i]);
            }
            cache.count = 0;
        }
        cache.owner = this;

        // T-O1: bounded retry counters for the slot-exhaustion path.
        std::size_t spin_retries  = 0;
        std::size_t sync_fallbacks = 0;

        // Slow path: free-list (O(1)) → linear scan → expansion.
        // The loop retries the free-list on every iteration because a
        // concurrent release may push a slot between iterations.
        while (true) {
            // G16: Try the lock-free free-list first. This is O(1) and
            // avoids the O(N) linear scan when slots have been previously
            // released (the common case after warmup). Only when the
            // free-list is empty do we fall through to the linear scan
            // for never-used slots or concurrent-release races.
            std::size_t fl_slot = free_list_pop_and_claim();
            if (fl_slot != static_cast<std::size_t>(kFreeSlotNil)) {
                return fl_slot;
            }

            // Free-list empty — linear scan for a never-used slot or one
            // released concurrently with the pop above.
            std::size_t nb = num_batches_.load(std::memory_order_acquire);
            std::size_t total = nb * kBatchSize;

            for (std::size_t i = 0; i < total; ++i) {
                std::size_t batch  = i / kBatchSize;
                std::size_t offset = i % kBatchSize;
                bool expected = false;
                if (used_batches_[batch][offset].compare_exchange_strong(
                        expected, true, std::memory_order_acquire)) {
                    // R4: track active slot count for try_reclaim() fast-path
                    active_slot_count_.fetch_add(1, std::memory_order_relaxed);
                    return i;
                }
            }

            // All existing slots occupied — expand under the mutex.
            std::lock_guard<std::mutex> lock(expand_mutex_);

            std::size_t nb2   = num_batches_.load(std::memory_order_acquire);
            std::size_t total2 = nb2 * kBatchSize;

            for (std::size_t i = total; i < total2; ++i) {
                std::size_t batch  = i / kBatchSize;
                std::size_t offset = i % kBatchSize;
                bool expected = false;
                if (used_batches_[batch][offset].compare_exchange_strong(
                        expected, true, std::memory_order_acquire)) {
                    // R4: track active slot count for try_reclaim() fast-path
                    active_slot_count_.fetch_add(1, std::memory_order_relaxed);
                    return i;
                }
            }

            if (nb2 >= max_batches_limit_.load(std::memory_order_acquire)) {
                // All slots exhausted — extreme scenario.
                slot_exhaustion_count_.fetch_add(1, std::memory_order_relaxed);
                if (++spin_retries <= kMaxSpinRetries) {
                    // P0-7 (T-A2): bounded yield-and-retry.
                    std::this_thread::yield();
                    continue;
                }
                // T-O1: exceeded spin budget — fall back to synchronous
                // reclaim to drain pending retired objects. This frees
                // memory pressure and may indirectly release slots held
                // by reclaim-related paths. Bump the fallback metric so
                // operators can detect sustained slot exhaustion.
                sync_fallback_count_.fetch_add(1, std::memory_order_relaxed);
                if (++sync_fallbacks > kMaxSyncFallbacks) {
                    // P0-3: Hard cap reached — 8192+ genuinely-live handles
                    // for an extended period. Return `npos` instead of
                    // throwing so the caller (e.g. hazptr_holder →
                    // read_handle ctor, which is noexcept) can degrade
                    // gracefully by producing an empty handle. This
                    // matches the refcount-overflow behavior in
                    // read_handle and prevents std::terminate.
                    return npos;
                }
                (void)try_reclaim();
                // Reset spin budget for the next fallback round.
                spin_retries = 0;
                std::this_thread::yield();
                continue;
            }

            add_batch_unchecked();
            num_batches_.store(nb2 + 1, std::memory_order_release);
        }
    }

    /// Release a previously acquired hazard pointer slot.
    /// If the TLS cache has room, the slot is cached thread-locally for
    /// O(1) re-acquisition by the same thread.
    void release_slot(std::size_t slot) {
        std::size_t batch  = slot / kBatchSize;
        std::size_t offset = slot % kBatchSize;
        slot_batches_[batch][offset].store(nullptr, std::memory_order_release);

        // Try to cache thread-locally for O(1) re-acquisition
        auto& cache = tls_cache();
        if ((cache.owner == nullptr || cache.owner == this) &&
            cache.count < tls_slot_cache::kCacheSize) {
            cache.slots[cache.count] = slot;
            ++cache.count;
            cache.owner = this;
            return;
        }

        // TLS cache full or different owner — release globally.
        // G16: push the slot index onto the lock-free free-list so the
        // next acquire_slot() (from any thread) can find it in O(1)
        // instead of doing the O(N) linear scan.
        used_batches_[batch][offset].store(false, std::memory_order_release);
        // R4: track active slot count for try_reclaim() fast-path
        active_slot_count_.fetch_sub(1, std::memory_order_relaxed);
        free_list_push(slot);
    }

    /// Release a slot globally (bypassing the TLS cache).
    void release_slot_global(std::size_t slot) {
        std::size_t batch  = slot / kBatchSize;
        std::size_t offset = slot % kBatchSize;
        slot_batches_[batch][offset].store(nullptr, std::memory_order_release);
        used_batches_[batch][offset].store(false, std::memory_order_release);
        // R4: track active slot count for try_reclaim() fast-path
        active_slot_count_.fetch_sub(1, std::memory_order_relaxed);
        // G16: push onto the lock-free free-list for O(1) re-acquisition
        // by any thread (used by TLS-cache destructor on thread exit and
        // invalidate_tls_cache()).
        free_list_push(slot);
    }

    /// Store a value to a hazard pointer slot (used by hazptr_holder).
    void store_slot(std::size_t slot, void* ptr) {
        std::size_t batch  = slot / kBatchSize;
        std::size_t offset = slot % kBatchSize;
        slot_batches_[batch][offset].store(ptr, std::memory_order_release);
    }

    /// Load a value from a hazard pointer slot (used by hazptr_holder).
    void* load_slot(std::size_t slot) const {
        std::size_t batch  = slot / kBatchSize;
        std::size_t offset = slot % kBatchSize;
        return slot_batches_[batch][offset].load(std::memory_order_acquire);
    }

    /// Return the current total slot capacity.
    std::size_t capacity() const noexcept {
        return num_batches_.load(std::memory_order_acquire) * kBatchSize;
    }

    /// P0-3: Number of currently-acquired (live) hazard pointer slots.
    /// Useful for monitoring slot pressure — when this approaches
    /// `max_slot_count()`, operators should either call `set_max_slots()`
    /// to expand capacity or investigate the cause of handle retention
    /// (e.g. long-lived read_handle objects).
    std::size_t active_slot_count() const noexcept {
        return active_slot_count_.load(std::memory_order_acquire);
    }

    /// P0-3: Maximum number of slots the domain can hold. Defaults to
    /// `kMaxBatches * kBatchSize` (8192) but can be raised at runtime
    /// via `set_max_slots()` up to `kAbsoluteMaxBatches * kBatchSize`
    /// (65536).
    std::size_t max_slot_count() const noexcept {
        return max_batches_limit_.load(std::memory_order_acquire) * kBatchSize;
    }

    /// P0-3: Runtime capacity expansion. Raises the maximum number of
    /// slots from the default 8192 up to 65536 (512 batches × 128 slots).
    /// Existing slots and active handles are unaffected — only the upper
    /// bound is raised, allowing `acquire_slot()` to allocate new batches
    /// on demand up to the new limit. Call this at startup (or during a
    /// quiescent period) when the workload is known to require more than
    /// 8192 concurrent live read_handle objects.
    ///
    /// @param n New maximum slot count. Silently clamped to
    ///          `[kMaxBatches * kBatchSize, kAbsoluteMaxBatches * kBatchSize]`.
    void set_max_slots(std::size_t n) noexcept {
        std::size_t batches = (n + kBatchSize - 1) / kBatchSize;
        if (batches < kMaxBatches) batches = kMaxBatches;
        if (batches > kAbsoluteMaxBatches) batches = kAbsoluteMaxBatches;
        max_batches_limit_.store(batches, std::memory_order_release);
    }

    /// P0-3: Current slot usage ratio (0.0 .. 1.0). When this exceeds
    /// 0.9, operators should either raise the limit via `set_max_slots()`
    /// or reduce reader fan-out — sustained high usage risks
    /// `acquire_slot()` returning `npos`, which degrades reads to cache
    /// misses (via empty read_handle).
    float slot_usage_ratio() const noexcept {
        std::size_t cap = max_slot_count();
        if (cap == 0) return 0.0f;
        std::size_t active = active_slot_count();
        return static_cast<float>(static_cast<double>(active) /
                                  static_cast<double>(cap));
    }

    /// P0-7 (T-A2): Number of times acquire_slot() exhausted all 8192
    /// slots and had to yield. Non-zero indicates the workload has
    /// 8000+ simultaneous live hazard pointers — likely a runaway
    /// reader-heavy workload or a thread/fiber leak. Previously this
    /// condition caused acquire_slot() to spin forever; now it yields
    /// and bumps this counter so operators can detect the failure.
    std::size_t slot_exhaustion_count() const noexcept {
        return slot_exhaustion_count_.load(std::memory_order_acquire);
    }

    /// T-O1: Number of times acquire_slot() exceeded the spin budget
    /// (kMaxSpinRetries) and fell back to a synchronous try_reclaim()
    /// to drain pending retired objects. Sustained non-zero growth
    /// means the workload has 8192+ live handles for extended periods
    /// — operators should either increase the cache's max_size (so fewer
    /// evictions create fewer concurrent handles) or reduce the reader
    /// fan-out. If this counter exceeds kMaxSyncFallbacks * (number of
    /// threads), acquire_slot() will start throwing std::runtime_error.
    std::size_t hazptr_sync_fallback_count() const noexcept {
        return sync_fallback_count_.load(std::memory_order_acquire);
    }

    // ----------------------------------------------------------------
    // Retire / reclaim (lock-free)
    // ----------------------------------------------------------------

    /// Retire a hazptr_obj_base-derived object — zero-allocation path.
    /// The object's embedded next_ pointer is used as the list link.
    /// The reclaim_ function pointer must be set before calling retire.
    ///
    /// P1-7 (T2.6 bugfix): Idempotent via the `retired_` atomic flag.
    /// `test_and_set` is atomic and ensures only one caller wins the
    /// race; subsequent callers become no-ops. This is the primary
    /// defense against double-retire under high-concurrency eviction
    /// workloads (the secondary defense is the per-shard write lock
    /// added by T3.4 in cache_trait.hpp).
    void retire_obj(hazptr_obj_base* obj) {
        if (!obj) return;

        // P1-7: Idempotent guard — only the first caller proceeds.
        // CAS on `retired_` from false → true; if it was already true,
        // the object is already in a TLS retire buffer (or pending list)
        // and a second push would corrupt the chain via `next_`.
        bool expected = false;
        bool already_retired = !obj->retired_.compare_exchange_strong(
            expected, true, std::memory_order_acq_rel, std::memory_order_acquire);
        if (already_retired) {
            // Double-retire attempted — drop silently. This happens
            // when two threads race on evict_lru() for the same
            // victim after unmarkForEviction() clears kExclusive
            // but before retire() runs. The race is benign because
            // only one thread can possibly have a live pointer to
            // the object's slot in the MM list (markForEviction is
            // exclusive), but the second thread may still hold the
            // raw `victim` pointer from find_eviction_victim().
            return;
        }

        // P1-3: Warn once if the drain worker has not been started.
        // Without the background drain worker, retired objects
        // accumulate in the global pending list and are only reclaimed
        // when maybe_auto_reclaim() fires (after pending_count exceeds
        // 4096) or when a thread explicitly calls try_reclaim(). In
        // read-heavy-write-light workloads this causes unbounded memory
        // growth — the worker is REQUIRED for production deployments.
        // The warning fires at most once per domain via a CAS flag; it
        // does not block retire, just alerts the operator.
        if (!drain_started_.load(std::memory_order_acquire)) {
            warn_drain_not_started_once();
        }

        // Push to TLS buffer (no mutex, no CAS)
        auto& buf = tls_retire_buf();
        // If the buffer currently belongs to a different domain, flush it
        // to that domain first so entries are routed to the correct pending
        // list. This preserves correctness when a thread interacts with
        // multiple domains (rare in practice; default_domain is the norm).
        if (buf.count > 0 && buf.owner_domain != nullptr && buf.owner_domain != this) {
            buf.owner_domain->flush_tls_buffer();
        }
        buf.owner_domain = this;
        buf.entries[buf.count++] = obj;

        if (buf.full()) {
            flush_tls_buffer();
        }
    }

    /// P1-3: Mark the drain worker as started. Called by
    /// `unified_cache::start_event_drain()` so that subsequent
    /// `retire_obj()` calls no longer emit the stderr warning. Has no
    /// effect on correctness — only suppresses the warning.
    void set_drain_started(bool started) noexcept {
        drain_started_.store(started, std::memory_order_release);
    }

    /// P1-3: Query whether the drain worker has been started. Used by
    /// `unified_cache::is_drain_worker_started()` and diagnostics to
    /// surface the state to operators.
    bool is_drain_started() const noexcept {
        return drain_started_.load(std::memory_order_acquire);
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
    /// Links all buffered entries into a chain and pushes via CAS.
    ///
    /// P1-7 (T2.6 bugfix): `buf.clear()` MUST run BEFORE `push_pending()`.
    /// `push_pending()` → `maybe_auto_reclaim()` → `try_reclaim()` →
    /// `flush_tls_buffer()` is recursive. If `buf` is not yet cleared when
    /// the recursive call enters, the recursive call re-chains the SAME
    /// entries (overwriting their `next_` pointers) and calls
    /// `push_pending()` a SECOND time — pushing the same chain to
    /// `pending_head_` twice. This creates a cycle (same object appears
    /// twice in the pending list), and `try_reclaim()` then frees the
    /// object on first encounter and dereferences `next_` on the second
    /// → heap-use-after-free (exit code 0xc0000374). Clearing before
    /// pushing makes the recursive `flush_tls_buffer()` a no-op.
    void flush_tls_buffer() {
        auto& buf = tls_retire_buf();
        if (buf.count == 0) return;

        // Link entries into a chain: entries[0] -> entries[1] -> ... -> entries[count-1]
        for (std::size_t i = 0; i + 1 < buf.count; ++i) {
            buf.entries[i]->next_ = buf.entries[i + 1];
        }
        buf.entries[buf.count - 1]->next_ = nullptr;

        // Save head/tail into locals BEFORE clearing the buffer.
        hazptr_obj_base* head = buf.entries[0];
        hazptr_obj_base* tail = buf.entries[buf.count - 1];
        buf.clear();

        // CAS-loop prepend to global pending list. Safe to recurse now:
        // buf is empty, so any recursive flush_tls_buffer() is a no-op.
        push_pending(head, tail);
    }

    /// Attempt to reclaim pending retired pointers (lock-free).
    /// Atomically swaps the entire pending list, builds a sorted protected-
    /// pointer vector, and batch-reclaims unprotected entries.
    ///
    /// @param batch_size T-P2-3 (R-7): Maximum number of objects to
    ///        process per call. 0 means unlimited (drain all — original
    ///        behavior, used by destructor / explicit full-drain paths).
    ///        When > 0, at most `batch_size` objects are inspected; any
    ///        unprocessed or still-protected objects are pushed back to
    ///        the pending list for the next call. This bounds the
    ///        worst-case latency of a single reclaim pass, eliminating
    ///        the spikes seen when the pending list is large (e.g.,
    ///        after a burst of evictions). The list is reversed before
    ///        processing so oldest objects are inspected first — this
    ///        prevents starvation of old objects under continuous retire
    ///        pressure (newest objects are more likely to still be
    ///        protected, so they yield fewer reclaims per inspection).
    /// @return Number of objects actually reclaimed.
    std::size_t try_reclaim(std::size_t batch_size = 0) {
        // Flush TLS buffer first
        flush_tls_buffer();

        // Atomically swap the pending list
        hazptr_obj_base* head = pending_head_.exchange(nullptr, std::memory_order_acq_rel);
        if (!head) return 0;

        // T-P2-3 (R-7): For incremental reclaim (batch_size > 0), reverse
        // the list so we process oldest objects first. The pending list is
        // LIFO (push_pending prepends), so without reversal the newest
        // objects would be inspected first and oldest objects could starve
        // under continuous retire pressure. Reversal is O(n) but only
        // touches next_ pointers — no reclaim_ callbacks, no allocations.
        if (batch_size > 0) {
            head = reverse_pending_list(head);
        }

        // R4: Fast-path — if no hazard pointers are active, all pending
        // objects are safe to reclaim. Skip the expensive slot scan and
        // protected-vector build entirely. In read-heavy-write-light
        // workloads, reclaims often happen when no readers hold handles
        // (e.g., background worker tick between read bursts), so this
        // fast-path fires frequently.
        const std::vector<void*>* protected_vec = nullptr;
        if (active_slot_count_.load(std::memory_order_acquire) > 0) {
            // Slow path: some hazard pointers are active — build the
            // protected vector to check which objects are safe to reclaim.
            protected_vec = &build_protected_vector();
        }

        // Walk the list and reclaim unprotected entries
        std::size_t reclaimed_count = 0;
        hazptr_obj_base* new_head = nullptr;  // re-chain protected entries
        hazptr_obj_base* new_tail = nullptr;
        std::size_t remaining_count = 0;
        std::size_t processed = 0;  // T-P2-3 (R-7): batch counter

        hazptr_obj_base* curr = head;
        while (curr) {
            // T-P2-3 (R-7): Stop after inspecting batch_size objects.
            // The unprocessed tail (newest objects when batch_size > 0,
            // since we reversed) is pushed back to the pending list.
            if (batch_size > 0 && processed >= batch_size) {
                break;
            }

            hazptr_obj_base* next = curr->next_;
            curr->next_ = nullptr;

            // R4: If protected_vec is null (fast-path), all objects are safe.
            if (protected_vec == nullptr ||
                !is_in_protected_vector(*protected_vec, curr)) {
                // Reclaim this object
                if (curr->reclaim_) {
                    curr->reclaim_(curr);
                }
                ++reclaimed_count;
            } else {
                // Still protected — re-chain for later reclamation
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
                // Walk to the end of the unprocessed tail to find new_tail.
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
            // Count the unprocessed objects for pending_count_ accounting.
            for (hazptr_obj_base* p = curr; p; p = p->next_) {
                ++remaining_count;
            }
        }

        // Push remaining protected entries back to the global list
        if (new_head) {
            push_pending(new_head, new_tail);
        }

        // Update statistics (relaxed, hot path is try_reclaim itself)
        reclaim_total_.fetch_add(reclaimed_count, std::memory_order_relaxed);
        pending_count_.store(remaining_count, std::memory_order_release);

        return reclaimed_count;
    }

    /// Return the current number of pending (unreclaimed) retired objects.
    /// This is a snapshot — the actual count may change concurrently.
    std::size_t pending_count() const noexcept {
        return pending_count_.load(std::memory_order_acquire);
    }

    /// Return the cumulative number of objects reclaimed since the domain
    /// was created.
    std::size_t reclaim_total() const noexcept {
        return reclaim_total_.load(std::memory_order_acquire);
    }

    /// P1-3 (T1.4): Return the current auto-reclaim threshold. When
    /// `pending_count()` exceeds this value, the next `retire()` /
    /// `push_pending()` call synchronously invokes `try_reclaim()` to
    /// drain the pending list. Default is 65536 — high enough to avoid
    /// reclaim thrashing under steady-state retire rates, low enough to
    /// bound worst-case memory pressure under bursty retire spikes.
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

    /// Obtain the default global hazard pointer domain.
    static hazptr_domain& default_domain() {
        static hazptr_domain domain;
        return domain;
    }

    // Allow hazptr_holder to access internal helpers.
    friend class hazptr_holder;

private:
    // ----------------------------------------------------------------
    // G16: Lock-free free-list (Treiber stack) for O(1) slot reuse.
    //
    // Problem: acquire_slot()'s TLS cache miss path linearly scanned up
    // to 8192 slots (kMaxBatches * kBatchSize). Under high concurrency
    // with frequent read_handle churn, the TLS cache hit rate drops and
    // the O(N) scan dominates acquire latency.
    //
    // Solution: maintain a Treiber stack of released slot indices.
    //   - release_slot (TLS cache full) / release_slot_global push the
    //     slot index onto the stack (O(1), lock-free CAS).
    //   - acquire_slot pops from the stack (O(1)) before falling back
    //     to the linear scan. The TLS cache remains the first priority.
    //
    // Node memory: the "next" link for each slot is embedded in the
    // `next_free_batches_` array (indexed by global slot index), so no
    // heap allocation is needed per push/pop — only the slot index is
    // threaded through the stack.
    //
    // ABA safety: the 64-bit head packs a 32-bit tag (high) with a
    // 32-bit slot index (low). The tag is incremented on every
    // successful push/pop, so a stale CAS attempt always observes a
    // different tag and fails — eliminating the classic Treiber-stack
    // ABA problem without requiring double-word CAS.
    //
    // Dead-entry handling: a slot may appear on the free-list AND be
    // claimed concurrently by the linear-scan fallback (which CASes
    // used_batches_ false→true without consulting the free-list). When
    // pop unlinks such a slot, the used_batches_ CAS fails and the slot
    // is discarded (NOT re-pushed) — it is removed from the free-list
    // and re-added later via a normal release. This prevents stale
    // entries from accumulating.
    // ----------------------------------------------------------------

    // Sentinel slot index meaning "end of free-list". Valid slot indices
    // are always < kAbsoluteMaxBatches * kBatchSize (65536), so uint32_t
    // max is a safe nil.
    static constexpr std::uint32_t kFreeSlotNil = static_cast<std::uint32_t>(-1);

    static constexpr std::uint64_t pack_free(std::uint32_t tag, std::uint32_t slot) noexcept {
        return (static_cast<std::uint64_t>(tag) << 32) | static_cast<std::uint64_t>(slot);
    }
    static constexpr std::uint32_t free_slot_of(std::uint64_t packed) noexcept {
        return static_cast<std::uint32_t>(packed & 0xFFFFFFFFULL);
    }
    static constexpr std::uint32_t free_tag_of(std::uint64_t packed) noexcept {
        return static_cast<std::uint32_t>(packed >> 32);
    }

    // Read/write the embedded "next" link for a slot. Relaxed ordering
    // suffices because the acquire-release operations on free_list_head_
    // (CAS in push, load+CAS in pop) establish the necessary
    // happens-before relationships between a pushed link and the pop
    // that reads it.
    std::uint32_t load_next_free(std::size_t slot) const noexcept {
        std::size_t batch  = slot / kBatchSize;
        std::size_t offset = slot % kBatchSize;
        return next_free_batches_[batch][offset].load(std::memory_order_relaxed);
    }

    void store_next_free(std::size_t slot, std::uint32_t next) noexcept {
        std::size_t batch  = slot / kBatchSize;
        std::size_t offset = slot % kBatchSize;
        next_free_batches_[batch][offset].store(next, std::memory_order_relaxed);
    }

    /// Push a slot index onto the free-list (lock-free, O(1)).
    void free_list_push(std::size_t slot) noexcept {
        std::uint32_t idx = static_cast<std::uint32_t>(slot);
        std::uint64_t head = free_list_head_.load(std::memory_order_acquire);
        while (true) {
            // Link this slot to the current head. This relaxed store is
            // ordered before the release CAS on head below; a concurrent
            // pop that acquires the new head is therefore guaranteed to
            // observe this link.
            store_next_free(slot, free_slot_of(head));
            std::uint64_t new_head = pack_free(free_tag_of(head) + 1, idx);
            if (free_list_head_.compare_exchange_weak(
                    head, new_head,
                    std::memory_order_release,
                    std::memory_order_acquire)) {
                return;
            }
            // CAS failed — head changed; retry with the updated value.
        }
    }

    /// Pop a slot from the free-list and claim it (used_batches_ CAS).
    /// Returns the slot index, or kFreeSlotNil if the free-list is empty.
    /// Handles dead entries (slot already claimed by the linear-scan
    /// fallback) by discarding them and retrying.
    std::size_t free_list_pop_and_claim() noexcept {
        while (true) {
            std::uint64_t head = free_list_head_.load(std::memory_order_acquire);
            std::uint32_t idx = free_slot_of(head);
            if (idx == kFreeSlotNil) {
                return static_cast<std::size_t>(kFreeSlotNil);
            }
            std::uint32_t next = load_next_free(idx);
            std::uint64_t new_head = pack_free(free_tag_of(head) + 1, next);
            if (free_list_head_.compare_exchange_weak(
                    head, new_head,
                    std::memory_order_acquire,
                    std::memory_order_acquire)) {
                // Successfully unlinked idx from the free-list. Claim it
                // via used_batches_ so the slot is not handed out twice
                // (the linear-scan fallback may have grabbed it first).
                std::size_t batch  = idx / kBatchSize;
                std::size_t offset = idx % kBatchSize;
                bool expected = false;
                if (used_batches_[batch][offset].compare_exchange_strong(
                        expected, true, std::memory_order_acquire)) {
                    active_slot_count_.fetch_add(1, std::memory_order_relaxed);
                    return static_cast<std::size_t>(idx);
                }
                // Dead entry — slot already claimed by the linear-scan
                // fallback. Discard (do NOT re-push); continue popping.
            }
            // CAS failed or dead entry — retry with the new head.
        }
    }

    // ---- Batch management helpers -----------------------------------------

    void add_batch_unchecked() {
        auto slots = std::make_unique<std::atomic<void*>[]>(kBatchSize);
        auto used  = std::make_unique<std::atomic<bool>[]>(kBatchSize);
        // G16: Per-slot "next" link for the lock-free free-list.
        // Initialized to kFreeSlotNil (end-of-list sentinel); only
        // written by free_list_push when the slot is released.
        auto next_free = std::make_unique<std::atomic<std::uint32_t>[]>(kBatchSize);
        for (std::size_t j = 0; j < kBatchSize; ++j) {
            slots[j].store(nullptr, std::memory_order_relaxed);
            used[j].store(false, std::memory_order_relaxed);
            next_free[j].store(kFreeSlotNil, std::memory_order_relaxed);
        }
        slot_batches_.push_back(std::move(slots));
        used_batches_.push_back(std::move(used));
        next_free_batches_.push_back(std::move(next_free));
    }

    // ---- Protected pointer utilities --------------------------------------

    /// Check whether a pointer is currently protected by any hazard slot.
    /// O(max_slots) scan — used only in the fast-path of single-pointer checks.
    bool is_protected(void* ptr) const {
        std::size_t nb = num_batches_.load(std::memory_order_acquire);
        for (std::size_t b = 0; b < nb; ++b) {
            for (std::size_t j = 0; j < kBatchSize; ++j) {
                if (used_batches_[b][j].load(std::memory_order_acquire)) {
                    if (slot_batches_[b][j].load(std::memory_order_acquire) == ptr) {
                        return true;
                    }
                }
            }
        }
        return false;
    }

    /// Build a sorted vector of all currently protected pointers.
    /// Replaces the old std::unordered_set — avoids hash computation and
    /// per-bucket heap allocation. Binary search gives O(log n) lookups.
    ///
    /// R4: Uses a thread_local cached vector to avoid heap allocation on
    /// every try_reclaim() call. The vector's capacity is reused across
    /// calls — only the size changes. This eliminates the malloc/free
    /// pair per reclaim under high eviction rates.
    const std::vector<void*>& build_protected_vector() const {
        thread_local std::vector<void*> tls_vec;
        tls_vec.clear();

        std::size_t nb = num_batches_.load(std::memory_order_acquire);
        for (std::size_t b = 0; b < nb; ++b) {
            for (std::size_t j = 0; j < kBatchSize; ++j) {
                if (used_batches_[b][j].load(std::memory_order_acquire)) {
                    void* ptr = slot_batches_[b][j].load(std::memory_order_acquire);
                    if (ptr != nullptr) {
                        tls_vec.push_back(ptr);
                    }
                }
            }
        }
        std::sort(tls_vec.begin(), tls_vec.end());
        return tls_vec;
    }

    /// Binary search for a pointer in a sorted protected-pointer vector.
    static bool is_in_protected_vector(const std::vector<void*>& vec, void* ptr) {
        return std::binary_search(vec.begin(), vec.end(), ptr);
    }

    // ---- Lock-free pending list -------------------------------------------

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

    /// CAS-loop prepend: atomically insert [head..tail] at the front of
    /// the global pending list. No mutex required.
    /// Returns the number of entries pushed (chain length).
    std::size_t push_pending(hazptr_obj_base* head, hazptr_obj_base* tail) {
        // Count the chain length first for stats accounting.
        std::size_t chain_len = 1;
        for (hazptr_obj_base* p = head; p != tail; p = p->next_) {
            ++chain_len;
        }

        tail->next_ = pending_head_.load(std::memory_order_acquire);
        while (!pending_head_.compare_exchange_weak(
                    tail->next_, head,
                    std::memory_order_release,
                    std::memory_order_acquire)) {
            // CAS failed — tail->next_ was updated with the current head.
            // Retry with the new pending_head_ value.
        }

        pending_count_.fetch_add(chain_len, std::memory_order_release);

        // P1-3 (T1.4): Auto-reclaim when pending_count exceeds threshold.
        // The check is on every push_pending() (called from flush_tls_buffer
        // when a TLS retire buffer fills, or from explicit retire_obj
        // chains) — it bounds worst-case pending backlog under bursty
        // retire pressure even if no background worker is running. A CAS
        // flag prevents stampede: only one thread at a time performs the
        // synchronous try_reclaim; concurrent threads observe the flag is
        // set and skip, leaving reclaim to the background worker or the
        // next push_pending after the in-progress one completes.
        maybe_auto_reclaim();
        return chain_len;
    }

    /// Reclaim all remaining pending objects (called in destructor).
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
    /// Uses a CAS flag to prevent stampede — at most one thread performs
    /// the synchronous reclaim; concurrent push_pending() calls observe
    /// the flag and skip, leaving the next push_pending() to retry after
    /// the in-progress reclaim completes. This bounds worst-case memory
    /// pressure under bursty retire workloads without adding lock
    /// contention to the steady-state hot path.
    void maybe_auto_reclaim() {
        const std::size_t threshold = reclaim_threshold_.load(std::memory_order_acquire);
        // threshold == max() effectively disables auto-reclaim.
        if (threshold == std::numeric_limits<std::size_t>::max()) return;
        const std::size_t pending = pending_count_.load(std::memory_order_acquire);
        if (pending <= threshold) return;

        // Try to claim the reclaim slot. CAS from false->true; if it
        // fails, another thread is already reclaiming — bail out.
        bool expected = false;
        if (!reclaim_in_progress_.compare_exchange_strong(
                expected, true, std::memory_order_acq_rel)) {
            return;
        }

        reclaim_auto_triggered_count_.fetch_add(1, std::memory_order_relaxed);
        try_reclaim();

        reclaim_in_progress_.store(false, std::memory_order_release);
    }

    // ---- Data members -----------------------------------------------------

    // Hazard pointer slots (batch-based)
    std::vector<std::unique_ptr<std::atomic<void*>[]>> slot_batches_;
    std::vector<std::unique_ptr<std::atomic<bool>[]>>  used_batches_;
    // G16: Per-slot "next" link for the lock-free free-list (Treiber
    // stack), organized in batches alongside slot_batches_/used_batches_.
    // next_free_batches_[b][o] holds the next free slot index in the
    // stack chain (kFreeSlotNil = end of list). Only meaningful while
    // the slot is on the free-list; written by free_list_push.
    std::vector<std::unique_ptr<std::atomic<std::uint32_t>[]>> next_free_batches_;
    std::atomic<std::size_t> num_batches_{0};
    std::mutex expand_mutex_;  // Only used for batch expansion

    // G16: Head of the lock-free free-list (Treiber stack). Packs a
    // 32-bit ABA tag (high word) with a 32-bit slot index (low word).
    // kFreeSlotNil in the low bits means the list is empty. The tag is
    // incremented on every successful push/pop, eliminating ABA without
    // double-word CAS. Initialized to (tag=0, slot=kFreeSlotNil) = empty.
    alignas(64) std::atomic<std::uint64_t> free_list_head_{kFreeSlotNil};

    // P0-3: Runtime-configurable upper bound on the number of batches.
    // Defaults to `kMaxBatches` (64 → 8192 slots) and can be raised via
    // `set_max_slots()` up to `kAbsoluteMaxBatches` (512 → 65536 slots).
    // Stored as atomic so `acquire_slot()` can read it lock-free on the
    // slow path; updates happen at startup or during quiescent periods.
    alignas(64) std::atomic<std::size_t> max_batches_limit_{kMaxBatches};

    // P0-7 (T-A2): bumped each time acquire_slot() exhausts all 8192
    // slots. Distinct from reclaim counters — this measures slot
    // pressure, not retire pressure.
    alignas(64) std::atomic<std::size_t> slot_exhaustion_count_{0};

    // R4: Active hazard pointer count. Incremented on acquire_slot(),
    // decremented on release. When 0, try_reclaim() can skip the
    // expensive slot scan and protected-vector build — all pending
    // objects are guaranteed safe to reclaim. Uses relaxed atomics
    // (no ordering requirement — it's an optimization hint, not a
    // correctness constraint).
    alignas(64) std::atomic<std::size_t> active_slot_count_{0};

    // T-O1: bumped each time acquire_slot() exceeds kMaxSpinRetries and
    // falls back to a synchronous try_reclaim(). Distinct from
    // slot_exhaustion_count_ (which counts every exhausted retry) — this
    // only counts the sync-reclaim fallback invocations, giving operators
    // a clearer signal of sustained slot pressure.
    alignas(64) std::atomic<std::size_t> sync_fallback_count_{0};

    // Lock-free global pending retire list
    std::atomic<hazptr_obj_base*> pending_head_{nullptr};

    // Statistics counters (alignas to avoid false sharing with pending_head_)
    alignas(64) std::atomic<std::size_t> pending_count_{0};
    alignas(64) std::atomic<std::size_t> reclaim_total_{0};

    // P1-3 (T1.4): Auto-reclaim threshold and stampede guard.
    // When `pending_count_` exceeds `reclaim_threshold_`, the next
    // push_pending() synchronously invokes try_reclaim(). The
    // `reclaim_in_progress_` CAS flag ensures at most one thread at a
    // time performs the synchronous reclaim — concurrent callers
    // observe the flag and skip. `reclaim_auto_triggered_count_`
    // tracks how many times this auto-reclaim fired (useful for
    // sizing the threshold against actual retire pressure).
    // H-7 fix: lowered default from 65536 to 4096. In read-heavy-write-light
    // workloads, writes are sparse so the threshold was rarely hit — retired
    // objects accumulated to 65536 before any reclaim fired.
    alignas(64) std::atomic<std::size_t> reclaim_threshold_{4096};
    alignas(64) std::atomic<std::size_t> reclaim_auto_triggered_count_{0};
    alignas(64) std::atomic<bool> reclaim_in_progress_{false};

    // P1-3: Drain worker started flag. Set by
    // `unified_cache::start_event_drain()` so that `retire_obj()` can
    // detect the missing-worker condition and warn operators once.
    // Without the worker, retired objects accumulate until auto-reclaim
    // fires (pending > 4096) — causing unbounded memory growth in
    // read-heavy-write-light workloads.
    alignas(64) std::atomic<bool> drain_started_{false};

    /// P1-3: Emit a one-shot stderr warning when `retire_obj()` is
    /// called before `set_drain_started(true)`. Uses a CAS flag so the
    /// warning fires at most once per domain lifetime — operators see
    /// the message, start the worker, and subsequent retires are silent.
    void warn_drain_not_started_once() noexcept {
        bool expected = false;
        if (drain_warn_emitted_.compare_exchange_strong(
                expected, true, std::memory_order_acq_rel)) {
            std::fprintf(stderr,
                "[lru::hazptr] WARNING: retire_obj() called before "
                "start_event_drain(). Retired objects will accumulate "
                "until pending_count > 4096 (auto-reclaim threshold). "
                "Call unified_cache::start_event_drain() at startup to "
                "drain the pending list periodically. This warning "
                "fires once per hazptr_domain.\n");
            std::fflush(stderr);
        }
    }

    // P1-3: CAS flag ensuring the drain-not-started warning fires at
    // most once per domain. Stored separately from `drain_started_` so
    // that calling `set_drain_started(false)` (e.g. on shutdown) does
    // not re-arm the warning.
    alignas(64) std::atomic<bool> drain_warn_emitted_{false};
};

// ============================================================================
// hazptr_holder — RAII wrapper around a single hazard pointer slot
// ============================================================================

class hazptr_holder {
public:
    /// Acquire a slot from the default domain.
    /// P0-3: If acquire_slot() returns `npos` (all slots exhausted), the
    /// holder enters an empty state (`valid() == false`). Callers that
    /// need a valid holder must check `valid()` and degrade gracefully
    /// (e.g. produce an empty read_handle). This matches the refcount-
    /// overflow behavior and prevents std::terminate in noexcept callers.
    hazptr_holder()
        : domain_(hazptr_domain::default_domain())
        , slot_(domain_.acquire_slot())
    {}

    /// Acquire a slot from a specific domain.
    /// P0-3: Same npos-handling as the default-domain constructor.
    explicit hazptr_holder(hazptr_domain& domain)
        : domain_(domain)
        , slot_(domain_.acquire_slot())
    {}

    /// Release the slot on destruction.
    /// P0-3: Only release if the slot is valid (not npos). An empty
    /// holder (acquire failed) has no slot to release.
    ~hazptr_holder() {
        if (valid()) {
            domain_.release_slot(slot_);
        }
    }

    // Non-copyable
    hazptr_holder(const hazptr_holder&) = delete;
    hazptr_holder& operator=(const hazptr_holder&) = delete;

    // Movable
    hazptr_holder(hazptr_holder&& other) noexcept
        : domain_(other.domain_)
        , slot_(other.slot_)
    {
        other.slot_ = hazptr_domain::npos;
    }

    hazptr_holder& operator=(hazptr_holder&& other) noexcept {
        if (this != &other) {
            if (valid()) {
                domain_.release_slot(slot_);
            }
            slot_ = other.slot_;
            other.slot_ = hazptr_domain::npos;
        }
        return *this;
    }

    /// Publish protection for a pointer.
    /// P0-3: No-op if the holder is empty (slot acquisition failed).
    template <typename T>
    void protect(T* ptr) {
        if (!valid()) return;
        domain_.store_slot(slot_,
            const_cast<void*>(static_cast<const volatile void*>(ptr)));
    }

    /// Clear the protected pointer (e.g. when the iterator moves away).
    /// P0-3: No-op if the holder is empty.
    void clear() {
        if (!valid()) return;
        domain_.store_slot(slot_, nullptr);
    }

    /// Retrieve the currently protected pointer.
    template <typename T>
    T* get() const {
        return static_cast<T*>(domain_.load_slot(slot_));
    }

    /// Whether this holder owns a valid slot.
    /// P0-3: Returns false if the slot acquisition failed (npos). Callers
    /// must check this before relying on protection.
    bool valid() const noexcept {
        return slot_ != hazptr_domain::npos;
    }

    /// Swap the protected pointers between two holders (not the slots).
    /// Used by hazptr linked-list traversal: one holder protects curr,
    /// the other protects next; after advancing, swap them so the
    /// former-curr holder now protects next (the new curr).
    /// P0-3: No-op if either holder is empty.
    void swap(hazptr_holder& other) noexcept {
        if (!valid() || !other.valid()) return;
        // Both holders must be from the same domain
        void* my_ptr = domain_.load_slot(slot_);
        void* other_ptr = other.domain_.load_slot(other.slot_);
        domain_.store_slot(slot_, other_ptr);
        other.domain_.store_slot(other.slot_, my_ptr);
    }

private:
    hazptr_domain& domain_;
    std::size_t    slot_;
};

} // namespace lru::detail

#endif // LRU_DETAIL_HAZPTR_HPP
