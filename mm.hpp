// Unified LRU Cache Library - Eviction Strategies
// Merged from: mm_lru.hpp, mm_2q.hpp, mm_tiny_lfu.hpp, mm_wtiny_lfu.hpp
// SPDX-License-Identifier: MIT

#ifndef LRU_MM_HPP
#define LRU_MM_HPP

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

#include "detail/concurrent_hash_table.hpp"
#include "core.hpp"
#include "detail/count_min_sketch.hpp"
#include "detail/hazptr.hpp"
#include "detail/epoch_reclamation.hpp"
#include "detail/intrusive_list.hpp"

namespace lru {

// Forward declaration (defined in memory.hpp)
class slab_allocator;

// ============================================================================
// Access Mode
// ============================================================================

/// Controls whether an access is a read or write operation.
/// Used with updateOnWrite/updateOnRead to decide if promotion should occur.
enum class access_mode : uint8_t {
    read = 0,
    write = 1,
};

/// Task D: Overflow policy for handling set() when cache size exceeds
/// `max_size * (1 + overflow_tolerance)` due to items pinned by
/// outstanding read_handle objects (which cannot be evicted normally).
///
/// - kAllowGrowth: Default. Cache grows beyond max_size; pin release
///   triggers normal eviction back to max_size. Existing behavior.
/// - kRejectInsert: New set() calls are silently dropped when the
///   soft cap is exceeded. Cache size never exceeds the cap.
/// - kForceEvict: Force-evict the LRU item (bypassing pin check) to
///   make room. Pinned items' memory is reclaimed when handles release.
enum class overflow_policy : uint8_t {
    kAllowGrowth = 0,
    kRejectInsert = 1,
    kForceEvict = 2,
};

namespace detail {
/// Validate capacity limits.
/// max_size == 0 means zero capacity (no insertions allowed).
/// Use lru::unlimited for no limit.
/// max_memory == 0 means zero memory budget (no insertions allowed).
/// Use lru::unlimited for no memory limit.
inline void validate_capacity(std::size_t /*max_size*/, std::size_t /*max_memory*/) {
    // Zero capacity is allowed: it means no insertions are permitted.
    // Use lru::unlimited for no limit.
}

/// Mixin that gives an MM strategy a slab_allocator integration point.
/// It stores the allocator pointer and provides allocate_item(), which uses
/// the slab allocator when one is attached and falls back to the global heap
/// otherwise.  Item destruction still uses `delete item`; cache_item's class-
/// specific operator delete routes the memory back to the correct allocator.
template <typename Item>
class mm_allocator_mixin {
public:
    /// Attach an external slab allocator for item allocation.
    void set_allocator(slab_allocator* alloc) noexcept { allocator_ = alloc; }

    /// Get the attached slab allocator (may be nullptr).
    slab_allocator* get_allocator() const noexcept { return allocator_; }

protected:
    slab_allocator* allocator_ = nullptr;

    /// Allocate and construct a new item.
    /// When a slab allocator is attached, the item memory comes from the slab
    /// allocator and the item's allocator_ member is marked so that deletion
    /// routes back to the same allocator.
    template <typename... Args>
    Item* allocate_item(Args&&... args) {
        if (allocator_) {
            // Class-specific placement new: if the constructor throws, the
            // matching operator delete frees the slab block.
            auto* item = new (allocator_) Item(std::forward<Args>(args)...);
            item->allocator_ = allocator_;
            return item;
        }
        return new Item(std::forward<Args>(args)...);
    }
};

} // namespace detail

// ============================================================================
// Enhanced LRU Strategy - mm_lru
// ============================================================================


// ============================================================================
// Enhanced LRU Strategy Configuration
// ============================================================================

struct mm_lru_config {
    /// Insertion point specification:
    /// 0 = insert at head (traditional LRU, default)
    /// 1 = insert at 1/2 from tail
    /// 2 = insert at 1/4 from tail
    uint8_t lru_insertion_point_spec = 0;

    /// Default time between promotions for the same item (seconds).
    /// 0 = no delay (traditional LRU behavior).
    uint32_t default_lru_refresh_time = 60;

    /// Ratio for adaptive refresh time adjustment.
    /// newRefreshTime = max(default, tail_age * ratio), capped at kLruRefreshTimeCap.
    double lru_refresh_ratio = 0.0;

    /// Whether to promote the item on write access.
    bool update_on_write = false;

    /// Whether to promote the item on read access.
    bool update_on_read = true;

    /// Enable tryLock mode: skip promotion if lock unavailable.
    /// Default true since unified_cache already provides outer concurrency
    /// protection; blocking on update_mutex_ would create double-lock overhead.
    bool try_lock_update = true;

    /// Use combined lock for eviction iterators.
    bool use_combined_lock_for_iterators = false;

    /// B15: 淘汰搜索次数上限——当 EvictionPredicate 否决时最多继续搜索的项数。
    size_t eviction_search_tries = 3;

    /// Use segmented LRU list with per-segment spinlocks for higher
    /// read-concurrency. When true, record_access() locks only the source
    /// segment + segment 0 (MRU) instead of a global update_mutex_.
    /// Default false — use the traditional single intrusive_list.
    bool use_segmented_lru = false;

    /// Number of segments for the segmented LRU list (only when use_segmented_lru=true).
    /// Must be in [1, 256]. Default 64.
    uint8_t segmented_lru_num_segments = 64;

    /// Use epoch-based reclamation (EBR) instead of hazard pointers for
    /// deferred deletion of evicted items. EBR has faster read-path overhead
    /// (only an atomic load + branch) compared to hazptr (acquire_slot +
    /// store_slot). When true, the mm_lru uses epoch_domain for retire();
    /// when false (default), it uses hazptr_domain.
    bool use_ebr = false;

    /// In read-heavy mode, skip the try_acquire_write_lock_for_key() step in
    /// get() and always defer LRU promotion to the TLS access ring. This
    /// eliminates the atomic CAS operation on the write lock for every get(),
    /// reducing read-path latency at the cost of slightly delayed LRU ordering.
    /// Default true — production-recommended: defer promotion to TLS ring for
    /// lower read-path latency. The original `false` value forced a CAS on
    /// every get(); the new default reduces lock pressure significantly.
    bool defer_promotion = true;

    /// R9: Soft cap on the number of items deferred in `pending_deletion_`
    /// (items explicitly evicted while still holding active read_handles).
    /// 0 = unlimited (default, preserves legacy behavior). When non-zero and
    /// the pending list is at/over the cap, further evictions of pinned items
    /// are REFUSED — the item stays in the cache — bounding memory retained by
    /// handle-holding callers. Observable via `pending_deletion_count()` and
    /// `pending_deletion_skipped_count()`.
    ///
    /// Note: the main LRU capacity-eviction path (`evict_lru`) already skips
    /// pinned items entirely, so this cap only guards the explicit force-delete
    /// paths (force_del). It is a safety valve against handle leaks, not a
    /// normal-path limiter.
    std::size_t max_pending_deletion = 0;

    /// Interval in seconds for reconfiguring the adaptive refresh time.
    uint32_t mm_reconfigure_interval_secs = 0;

    /// Expected number of items for automatic bucket count sizing.
    /// 0 = use default bucket count (1024). When > 0, the internal hash
    /// table is pre-sized via concurrent_hash_table::buckets_for_items()
    /// to keep average chain length ≤ 0.25 at the expected load.
    size_t expected_items = 0;

    /// Custom node allocation function for non-EmbeddedChain hash table nodes.
    /// When non-null, hash table node_type objects are allocated via this
    /// function instead of ::operator new, enabling slab allocator integration.
    /// nullptr (default) = standard new/delete allocation.
    /// Only effective when the hash table is in non-EmbeddedChain mode
    /// (EmbeddedChain = false); in EmbeddedChain mode this is ignored.
    void* (*alloc_fn)(std::size_t) = nullptr;

    /// Custom node deallocation function (must pair with alloc_fn).
    /// nullptr (default) = standard delete deallocation.
    void  (*dealloc_fn)(void*) = nullptr;

    /// TTL eviction batch size: maximum number of expired items to evict per
    /// shard per lock acquisition in evict_expired_impl(). When 0 (default),
    /// all expired items are evicted in a single lock hold (legacy behavior).
    /// Setting this to a positive value (e.g. 64) causes the TTL cleaner to
    /// release the lock after evicting `ttl_evict_batch_size` items, allowing
    /// concurrent readers/writers to proceed. Remaining expired items are
    /// processed in subsequent cleaner cycles.
    std::size_t ttl_evict_batch_size = 0;

    /// Task D: Overflow tolerance — fraction of max_size beyond which the
    /// overflow_policy kicks in. Default 0.1 (10%). Only effective when
    /// overflow_policy_value != kAllowGrowth.
    double overflow_tolerance = 0.1;

    /// Task D: Overflow policy applied when cache size exceeds
    /// max_size * (1 + overflow_tolerance). Default kAllowGrowth
    /// (existing behavior — no enforcement).
    overflow_policy overflow_policy_value = overflow_policy::kAllowGrowth;

    // Max lruRefreshTime cap (same as CacheLib's 900s)
    static constexpr uint32_t k_lru_refresh_time_cap = 900;

    // B4: 配置校验——lru_insertion_point_spec 必须在 [0, 7] 范围内
    mm_lru_config() noexcept = default;

    void validate() const {
        if (lru_insertion_point_spec > 7) {
            throw std::invalid_argument(
                "mm_lru_config: lru_insertion_point_spec must be in [0, 7]");
        }
        if (!(lru_refresh_ratio >= 0.0)) {
            throw std::invalid_argument(
                "mm_lru_config: lru_refresh_ratio must be non-negative");
        }
    }
};

// ============================================================================
// Enhanced LRU Strategy
// ============================================================================

namespace detail {

/// Cached epoch time (seconds) — samples real clock at ~6.25% rate.
/// Uses thread_local uint32_t with constant initialization (plain primitive
/// to avoid MinGW TLS struct initialization issues in gtest context).
inline uint32_t cached_epoch_sec() noexcept {
    thread_local uint32_t tl_cached = 0;
    thread_local uint32_t tl_counter = 0;
    thread_local const uint32_t tl_epoch_base = static_cast<uint32_t>(
        std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());

    if ((tl_counter++ & 0xF) == 0) [[unlikely]] {
        auto now = static_cast<uint32_t>(
            std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count());
        auto elapsed = now - tl_epoch_base;
        // Monotonicity: never go backwards (defensive, steady_clock guarantees this)
        if (elapsed > tl_cached) {
            tl_cached = elapsed;
        }
        return tl_cached;
    }

    // Fast path: plain uint32_t load (~1-2 ns)
    return tl_cached;
}

} // namespace detail

template <
    typename Key,
    typename Value,
    typename Hash = std::hash<Key>,
    typename KeyEqual = std::equal_to<Key>,
    typename ProbingStyle = detail::chain_probing_tag,
    bool Segmented = false
>
/// A4: 线程安全契约——此类非线程安全，调用方必须确保在外层 unified_cache 锁内访问。
/// 内部 update_mutex_ 仅用于 try_lock_update 路径的解耦优化，不保证 MM 层独立线程安全。
class mm_lru : public detail::mm_allocator_mixin<detail::cache_item<Key, Value>> {
public:
    using key_type = Key;
    using mapped_type = Value;
    using size_type = std::size_t;
    using config_type = mm_lru_config;

    // Item type: cache_item with embedded intrusive hook
    using item_type = detail::cache_item<Key, Value>;
    using item_ptr = item_type*;

    // Intrusive list type
    using item_list = detail::intrusive_list<item_type, detail::intrusive_hook, detail::default_get_hook<item_type>>;

    // Segmented intrusive list type (per-segment spinlocks for read-heavy concurrency)
    using segmented_item_list = detail::segmented_intrusive_list<item_type, detail::intrusive_hook, detail::default_get_hook<item_type>, 64>;

    using iterator = typename item_list::iterator;
    using const_iterator = typename item_list::const_iterator;
    /// Reverse iterator: same as iterator type, but rbegin() starts from
    /// the LRU tail and moves toward the MRU head.
    using reverse_iterator = typename item_list::reverse_iterator;
    using const_reverse_iterator = typename item_list::const_reverse_iterator;

    // Map: Key -> item pointer
    // When Segmented=true, uses segmented_concurrent_hash_table (per-segment rehash, no global stall)
    using map_type = std::conditional_t<
        Segmented,
        detail::segmented_concurrent_hash_table<Key, item_ptr, Hash, KeyEqual, true, ProbingStyle, 64>,
        detail::concurrent_hash_table<Key, item_ptr, Hash, KeyEqual, true, ProbingStyle>
    >;

    // R2: Compile-time enforcement that the hash table uses EmbeddedChain.
    // Non-EmbeddedChain mode degrades all lock-free read paths to shared_lock
    // fallback (to prevent use-after-free), which kills read throughput under
    // high concurrency. This assert prevents accidental regression.
    static_assert(map_type::uses_embedded_chain,
        "MM strategies must use EmbeddedChain=true for lock-free reads. "
        "Non-EmbeddedChain mode is unsafe for read-heavy-write-light production workloads.");

    using callback_mgr = callback_manager<Key, Value>;
    using stats_type = cache_stats;

    static constexpr size_type npos = unlimited;
    // Item overhead: hook(2 ptrs + uint32_t + uint8_t + uint8_t) + key + value + handle + map entry
    static constexpr size_type item_overhead = sizeof(item_type) + map_type::entry_overhead;

    // --------------------------------------------------------------------
    // Constructors / Destructor
    // --------------------------------------------------------------------

    mm_lru() : mm_lru(mm_lru_config{}) {}

    explicit mm_lru(const mm_lru_config& config)
        : config_(config)
        , map_(config.expected_items > 0
            ? map_type::buckets_for_items(config.expected_items)
            : 1024,
            config.alloc_fn, config.dealloc_fn)
        , lru_refresh_time_(config.default_lru_refresh_time)
        , next_reconfigure_time_(config.mm_reconfigure_interval_secs == 0
            ? std::numeric_limits<uint32_t>::max()
            : current_time_sec() + config.mm_reconfigure_interval_secs) {}

    mm_lru(size_type max_size, const mm_lru_config& config = mm_lru_config{})
        : mm_lru([&] {
              auto cfg = config;
              if (cfg.expected_items == 0 && max_size > 0 && max_size != unlimited) {
                  cfg.expected_items = max_size;
              }
              return cfg;
          }()) {
        detail::validate_capacity(max_size, unlimited);
        max_size_ = max_size;
        stats_.max_size.store(max_size);
    }

    mm_lru(size_type max_size, size_type max_memory,
            const mm_lru_config& config = mm_lru_config{})
        : mm_lru([&] {
              auto cfg = config;
              if (cfg.expected_items == 0 && max_size > 0 && max_size != unlimited) {
                  cfg.expected_items = max_size;
              }
              return cfg;
          }()) {
        detail::validate_capacity(max_size, max_memory);
        max_size_ = max_size;
        max_memory_ = max_memory;
        stats_.max_size.store(max_size);
        stats_.max_memory.store(max_memory);
    }

    ~mm_lru() {
        // Clean up pending deletions first (items with active handles that were force_del'd)
        for (auto* item : pending_deletion_) {
            delete item;
        }
        pending_deletion_.clear();
        flush();
        // flush() retires the surviving items to the EBR/hazptr domain, which
        // defers their destruction. Force synchronous reclamation NOW so the
        // item memory is returned to its allocator (slab or heap) while the
        // allocator is still alive. If we left the items pending, slab-backed
        // items would be freed after the slab allocator is destroyed during
        // cache teardown → use-after-free. After shutdown() there are no
        // active critical sections, so the reclaim is immediate and safe.
        if (config_.use_ebr) {
            auto* domain = ebr_domain_ ? ebr_domain_ : &detail::epoch_domain::default_domain();
            domain->try_reclaim();
        } else {
            detail::hazptr_domain::default_domain().try_reclaim();
        }
    }

    // Non-copyable, non-movable (owns item pointers)
    mm_lru(const mm_lru&) = delete;
    mm_lru& operator=(const mm_lru&) = delete;

    // --------------------------------------------------------------------
    // Core cache API
    // --------------------------------------------------------------------

    template <typename V>
    void set(const Key& key, V&& value) {
        auto ptr = map_.find(key);
        if (!ptr) {
            insert_new(key, std::forward<V>(value));
        } else {
            update_existing(ptr, std::forward<V>(value), access_mode::write);
        }
    }

    // --------------------------------------------------------------------
    // P1-1: Native TTL integration. These methods satisfy the SFINAE checks
    // in `unified_cache::get_with_ttl()` and `unified_cache::evict_expired_impl()`,
    // so a `unified_cache<lru_trait<...>>` can be used directly with TTL
    // without the `ttl_cache` wrapper (and its double-locking).
    // --------------------------------------------------------------------

    /// Set with an explicit absolute expiry. `expiry_ns` is nanoseconds since
    /// the steady_clock epoch; 0 means no TTL.
    ///
    /// P0-3: Single hash lookup — finds the key once and branches to
    /// insert or update, setting expiry inline. No extra `map_.find` after set.
    template <typename V>
    void set_with_expiry(const Key& key, V&& value, std::uint64_t expiry_ns) {
        auto ptr = map_.find(key);
        if (!ptr) {
            insert_new(key, std::forward<V>(value), expiry_ns);
        } else {
            update_existing(ptr, std::forward<V>(value), access_mode::write);
            ptr->expiry_ns = expiry_ns;
            // P0-2: push updated entry into TTL heap (lazy; old entry stays stale)
            if (expiry_ns != 0) {
                ttl_heap_push(key, expiry_ns);
            }
        }
    }

    /// Return remaining TTL in nanoseconds, or `nullopt` if the item has no
    /// TTL, is not found, or has already expired. Satisfies the SFINAE check
    /// in `unified_cache::get_with_ttl()`.
    std::optional<std::uint64_t> ttl_remaining_ns(const Key& key) const {
        auto ptr = map_.find(key);
        if (!ptr || ptr->expiry_ns == 0) return std::nullopt;
        auto now_ns = static_cast<std::uint64_t>(
            std::chrono::steady_clock::now().time_since_epoch().count());
        if (now_ns >= ptr->expiry_ns) return std::nullopt;
        return ptr->expiry_ns - now_ns;
    }

    /// Evict all expired items in a single pass. Returns the count of items
    /// removed. Items with active handles (pinned by a `read_handle`) are
    /// skipped — they will be evicted lazily on the next pass once the handle
    /// is released. Satisfies the SFINAE check in
    /// `unified_cache::evict_expired_impl()`.
    ///
    /// Force-refresh the cached now_ns value used by the TTL hot path.
    /// Called by background TTL cleaner to keep the cache reasonably fresh,
    /// or manually before TTL-sensitive operations.
    void refresh_cached_now() const {
        auto now_ns = static_cast<std::uint64_t>(
            std::chrono::steady_clock::now().time_since_epoch().count());
        cached_now_ns_.store(now_ns, std::memory_order_relaxed);
    }

    /// Evict expired items synchronously.
    /// Locking: callers must hold the MM write lock (this is the same contract
    /// as `flush()`). The background TTL cleaner in `unified_cache` acquires
    /// the write lock before calling this.
    std::size_t evict_expired() {
        cleanup_pending_deletion();
        auto now_ns = static_cast<std::uint64_t>(
            std::chrono::steady_clock::now().time_since_epoch().count());

        // P0-2: Use TTL min-heap for O(log n) expired-item lookup.
        // Lazy: pop from heap and validate against the current map entry.
        // Stale entries (key removed, or expiry updated to later time) are
        // simply discarded. Pinned items are pushed back so they are retried
        // on the next eviction pass.
        std::size_t evicted = 0;
        while (!ttl_heap_.empty()) {
            const auto& top = ttl_heap_.front();
            if (top.expiry_ns > now_ns) {
                break;  // smallest expiry is in the future — done
            }
            Key key = top.key;
            std::uint64_t heap_expiry = top.expiry_ns;
            std::pop_heap(ttl_heap_.begin(), ttl_heap_.end(),
                          std::greater<ttl_heap_entry>{});
            ttl_heap_.pop_back();

            auto ptr = map_.find(key);
            if (!ptr) continue;  // key already removed
            if (ptr->expiry_ns != heap_expiry) continue;  // stale heap entry
            if (ptr->has_active_handle()) {
                // Pinned — push back and try again next time
                ttl_heap_.push_back(ttl_heap_entry{heap_expiry, std::move(key)});
                std::push_heap(ttl_heap_.begin(), ttl_heap_.end(),
                               std::greater<ttl_heap_entry>{});
                // Stop early to avoid cycling through many pinned items
                // under sustained handle pressure.
                break;
            }
            // O7: TTL expiration — fire on_expire (not on_evict).
            erase_expired_impl(key);
            ++evicted;
        }
        return evicted;
    }

    /// Batched variant: evict at most `batch_size` expired items. Used by
    /// the non-blocking TTL cleaner path to avoid holding the write lock for
    /// too long under heavy expiry load. If `batch_size == 0`, delegates to
    /// `evict_expired()` (evict all).
    std::size_t evict_expired_n(std::size_t batch_size) {
        if (batch_size == 0) return evict_expired();
        cleanup_pending_deletion();
        auto now_ns = static_cast<std::uint64_t>(
            std::chrono::steady_clock::now().time_since_epoch().count());

        // P0-2: Batched TTL min-heap eviction.
        std::size_t evicted = 0;
        while (evicted < batch_size && !ttl_heap_.empty()) {
            const auto& top = ttl_heap_.front();
            if (top.expiry_ns > now_ns) {
                break;
            }
            Key key = top.key;
            std::uint64_t heap_expiry = top.expiry_ns;
            std::pop_heap(ttl_heap_.begin(), ttl_heap_.end(),
                          std::greater<ttl_heap_entry>{});
            ttl_heap_.pop_back();

            auto ptr = map_.find(key);
            if (!ptr) continue;
            if (ptr->expiry_ns != heap_expiry) continue;
            if (ptr->has_active_handle()) {
                // Pinned — push back and stop this batch
                ttl_heap_.push_back(ttl_heap_entry{heap_expiry, std::move(key)});
                std::push_heap(ttl_heap_.begin(), ttl_heap_.end(),
                               std::greater<ttl_heap_entry>{});
                break;
            }
            // O7: TTL expiration — fire on_expire (not on_evict).
            erase_expired_impl(key);
            ++evicted;
        }
        return evicted;
    }

    template <typename V>
    bool add(const Key& key, V&& value) {
        auto ptr = map_.find(key);
        if (!ptr) {
            insert_new(key, std::forward<V>(value));
            return true;
        }
        record_access(ptr, access_mode::read);
        return false;
    }

    template <typename V>
    bool replace(const Key& key, V&& value) {
        cleanup_pending_deletion();
        auto ptr = map_.find(key);
        if (!ptr) return false;
        update_existing(ptr, std::forward<V>(value), access_mode::write);
        return true;
    }

    /// H0: Get with handle — 返回 read_handle，防止持有期被淘汰。
    read_handle<Value> get(const Key& key) {
        auto ptr = map_.find(key);
        if (!ptr) {
            stats_.register_miss();
            callbacks_.collect_miss(key);
            return {};
        }
        auto* item = ptr;
        record_access(item, access_mode::read);
        stats_.register_hit();
        callbacks_.collect_hit(key, item->value);
        return read_handle<Value>{&item->value, &item->refcount};
    }

    /// Const get — 不提升 LRU，返回 const handle（适用于只读场景）。
    read_handle<const Value> get(const Key& key) const {
        auto ptr = map_.find(key);
        if (!ptr) {
            stats_.register_miss();
            callbacks_.collect_miss(key);
            return {};
        }
        auto* item = ptr;
        stats_.register_hit();
        callbacks_.collect_hit(key, item->value);
        return read_handle<const Value>{&item->value, &item->refcount};
    }

    /// H0: Peek with handle — 不提升 LRU，但返回 handle 防止持有期被淘汰。
    /// Uses find_and_pin_lockfree() to attempt lock-free pinning first
    /// (optimistic read + incRef without bucket lock), falling back to
    /// find_and_pin() (shared lock path) if the lock-free pin fails.
    /// No stripe-level read lock is needed.
    read_handle<const Value> peek(const Key& key) const {
        auto pin_fn = [](auto* item) {
            return item->refcount.incRef() == detail::IncResult::kIncOk;
        };
        auto ptr = map_.find_and_pin_lockfree(key, pin_fn);
        if (!ptr) return {};
        return read_handle<const Value>{&ptr->value, &ptr->refcount, pre_pinned};
    }

    read_handle<Value> peek_for_get(const Key& key) {
        return peek_for_get_with_hash(key, Hash{}(key));
    }

    /// T16.4: peek_for_get with a pre-computed hash. The hash MUST be
    /// the result of Hash{}(key) — callers are responsible for hash
    /// compatibility. Used by bulk_get to avoid re-hashing each key
    /// for both shard dispatch and hash-table lookup.
    read_handle<Value> peek_for_get_with_hash(const Key& key, std::size_t hash) {
        // P0-2 / T2.2: When EBR is enabled (config_.use_ebr == true), the
        // read path must hold an epoch_guard so that concurrently retired
        // objects are not reclaimed while readers traverse the hash table.
        //
        // T2.1: The hash table's find_and_pin_lockfree_with_hash() also
        // acquires an epoch_guard at entry when its ebr_domain_ is set
        // (via set_ebr_domain()). This MM-level guard handles the case
        // where config_.use_ebr == true but no explicit domain was set —
        // it falls back to the default global domain. Both guards are
        // idempotent (nested epoch_guards on the same thread are safe:
        // enter_critical/exit_critical use a per-thread nesting counter),
        // so defense-in-depth here is correct and low-overhead.
        //
        // In hazptr mode (use_ebr=false), the guard is a no-op; hazptr
        // uses per-pointer protection via hazptr_holder inside the hash
        // table's find_and_pin_lockfree() implementation.
        std::optional<detail::epoch_domain::epoch_guard> ebr_guard;
        if (config_.use_ebr) {
            auto* domain = ebr_domain_ ? ebr_domain_ : &detail::epoch_domain::default_domain();
            ebr_guard.emplace(*domain);
        }
        auto pin_fn = [](auto* item) {
            return item->refcount.incRef() == detail::IncResult::kIncOk;
        };
        auto ptr = map_.find_and_pin_lockfree_with_hash(key, hash, pin_fn);
        if (!ptr) return {};
        // P1-1: Inline TTL check. Fast path: expiry_ns == 0 (no TTL) skips
        // steady_clock::now() entirely. Only items with a TTL set pay the
        // clock read, and only on cache hits. Expired items are unpinned
        // (decRef) and reported as a miss; the actual eviction is handled
        // lazily by evict_expired() / the background TTL cleaner, NOT here,
        // so we don't need a write lock on the hot path.
        if (ptr->expiry_ns != 0) {
            // P1-10: Track TTL check frequency on the read path.
            // ttl_expired_count / ttl_checked_count gives the expiration
            // ratio, useful for sizing the TTL cleaner interval.
            stats_.ttl_checked_count.fetch_add(1, std::memory_order_relaxed);
            // P1-8: Three-tier TTL check using cached_now_ns_.
            //
            // Key invariant: cached_now_ns_ <= real_now (the cached value
            // is only ever updated to a real clock reading, and real_now
            // is monotonically increasing). This gives us:
            //
            //   cached_now >= expiry_ns  =>  real_now >= expiry_ns  (definitely expired)
            //   cached_now < expiry_ns   =>  unknown (cache may be stale)
            //
            // Tier 1 (fast, no clock read): If the cached time already
            // exceeds expiry_ns, the item is definitely expired. This
            // handles the common case of repeatedly accessing items that
            // expired long ago but haven't been cleaned up yet.
            //
            // Tier 2 (fast, no clock read): If cached_now is far from
            // expiry_ns (beyond the drift window), the item is treated as
            // fresh. This is safe from false-expires (we never report an
            // unexpired item as expired), but may delay expiry detection
            // if cached_now is very stale. The delay is bounded by the
            // TTL cleaner's refresh interval (refresh_cached_now() is
            // called periodically by the background cleaner).
            //
            // Tier 3 (slow, clock read): If cached_now is within the drift
            // window of expiry_ns, we need a precise reading to determine
            // whether the item has actually expired. This is the only
            // path that reads steady_clock::now().
            std::uint64_t now_ns = cached_now_ns_.load(std::memory_order_relaxed);

            // Tier 1 (cheap, no clock): if the cached time already passed
            // expiry, the item is definitely expired — cached_now_ns_ is a
            // lower bound on real time (it only ever moves forward), so
            // cached_now >= expiry  ⇒  real_now >= expiry. This catches
            // repeatedly-accessed long-expired items without a clock read.
            if (now_ns != 0 && now_ns >= ptr->expiry_ns) {
                stats_.ttl_expired_count.fetch_add(1, std::memory_order_relaxed);
                ptr->refcount.decRef();
                stats_.register_miss();
                callbacks_.collect_miss(key);
                return {};
            }

            // Tier 2 (precise): cached_now_ns_ is only refreshed by the
            // background TTL cleaner, so without one it can go arbitrarily
            // stale. "cached_now is far before expiry" therefore does NOT
            // imply the item is still fresh — a stale cached time would
            // serve already-expired items forever. Always read the real
            // clock here for a sound expiry check. (Non-TTL items skip this
            // whole block via the `expiry_ns != 0` guard above.)
            now_ns = static_cast<std::uint64_t>(
                std::chrono::steady_clock::now().time_since_epoch().count());
            cached_now_ns_.store(now_ns, std::memory_order_relaxed);

            if (now_ns >= ptr->expiry_ns) {
                stats_.ttl_expired_count.fetch_add(1, std::memory_order_relaxed);
                ptr->refcount.decRef();
                stats_.register_miss();
                callbacks_.collect_miss(key);
                return {};
            }
        }
        return read_handle<Value>{&ptr->value, &ptr->refcount, pre_pinned};
    }

    /// Get a shared_ptr copy of the value.
    /// Returns a heap-copied value wrapped in shared_ptr, safe to use even
    /// if the cache entry is later evicted. Returns nullptr on cache miss.
    /// Does NOT modify LRU order (peek-like semantics).
    std::shared_ptr<Value> get_shared(const Key& key) {
        auto ptr = map_.find(key);
        if (!ptr) {
            stats_.register_miss();
            callbacks_.collect_miss(key);
            return nullptr;
        }
        stats_.register_hit();
        callbacks_.collect_hit(key, ptr->value);
        return std::make_shared<Value>(ptr->value);
    }

    bool del(const Key& key) {
        cleanup_pending_deletion();
        auto ptr = map_.find(key);
        if (!ptr) return false;
        if (ptr->has_active_handle()) return false;
        erase_impl(key);
        return true;
    }

    /// Peek the LRU tail item's key (the next eviction victim).
    /// Returns std::nullopt if the cache is empty. Does not promote or
    /// modify the item — pure read. Used by overflow_policy::kForceEvict
    /// to identify which key to force_del.
    std::optional<Key> peek_lru_tail_key() const {
        auto* tail = list_tail();
        if (!tail) return std::nullopt;
        return tail->key;
    }

    /// Force delete a key even if it has active handles.
    /// The item is immediately removed from the map and LRU list,
    /// but memory is not freed until all handles are released.
    /// Returns true if the key was found (and removed from the map).
    bool force_del(const Key& key) {
        cleanup_pending_deletion();
        auto ptr = map_.find(key);
        if (!ptr) return false;
        auto* item = ptr;
        // R9: pending_deletion soft cap — if this item currently holds an
        // active handle and the deferred-deletion list is at/over the cap,
        // REFUSE the deletion (the item stays in the cache). Bounds memory
        // retained by handle-holding callers and surfaces the situation via
        // pending_deletion_skipped_count(). The check runs before any state
        // is mutated, so a refused delete leaves the cache fully intact.
        if (item->has_active_handle() && pending_deletion_at_cap()) {
            pending_deletion_skipped_count_.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        size_type mem = calc_item_memory(item->key, item->value);

        // Remove from map first
        map_.erase(key);

        // Remove from LRU list
        remove_from_list(item);
        unpinned_tail_ = nullptr;

        // Update stats
        stats_.current_memory.fetch_sub(mem);
        stats_.current_size.store(list_size());

        if (!item->has_active_handle()) {
            // No active handles - safe to move value and retire for deferred deletion.
            // Use hazptr retire() in case a concurrent iterator holds a hazard
            // pointer protecting this node. When EBR is enabled, use
            // epoch_domain::retire() instead for faster read-path overhead.
            callbacks_.collect_evict(item->key, std::move(item->value));
            if (config_.use_ebr) {
                auto* domain = ebr_domain_ ? ebr_domain_ : &detail::epoch_domain::default_domain();
                domain->retire(item);
            } else {
                detail::hazptr_domain::default_domain().retire(item);
            }
        } else {
            // Has active handles - add to pending, do NOT move or invalidate value.
            // Eviction callback and deletion deferred to cleanup_pending_deletion()
            // after all handles are released.
            pending_deletion_.push_back(item);
        }
        return true;
    }

    std::optional<Value> pop(const Key& key) {
        cleanup_pending_deletion();
        auto ptr = map_.find(key);
        if (!ptr) return std::nullopt;
        auto* item = ptr;
        if (item->has_active_handle()) return std::nullopt;
        size_type mem = calc_item_memory(item->key, item->value);
        Value value = std::move(item->value);
        stats_.current_memory.fetch_sub(mem);
        remove_from_list(item);
        unpinned_tail_ = nullptr;
        map_.erase(key);
        if (config_.use_ebr) {
            auto* domain = ebr_domain_ ? ebr_domain_ : &detail::epoch_domain::default_domain();
            domain->retire(item);
        } else {
            detail::hazptr_domain::default_domain().retire(item);
        }
        stats_.current_size.store(list_size());
        return value;
    }

    std::optional<std::pair<Key, Value>> pop_lru() {
        cleanup_pending_deletion();
        auto* tail = list_tail();
        if (!tail || tail->has_active_handle()) return std::nullopt;
        if (!map_.contains(tail->key)) return std::nullopt;
        size_type mem = calc_item_memory(tail->key, tail->value);
        std::pair<Key, Value> result{std::move(tail->key), std::move(tail->value)};
        stats_.current_memory.fetch_sub(mem);
        remove_from_list(tail);
        unpinned_tail_ = nullptr;
        map_.erase(result.first);
        if (config_.use_ebr) {
            auto* domain = ebr_domain_ ? ebr_domain_ : &detail::epoch_domain::default_domain();
            domain->retire(tail);
        } else {
            detail::hazptr_domain::default_domain().retire(tail);
        }
        stats_.current_size.store(list_size());
        return result;
    }

    /// Flush the cache. Items pinned by an active read_handle are left in place.
    ///
    /// P1-8 (T2.6 bugfix): TOCTOU race fix + two-pass deferred retirement.
    ///
    /// Phase 1 (TOCTOU fix): The original implementation checked
    /// `has_active_handle()` then directly `delete`'d the item. A concurrent
    /// `peek()` (lockfree) could `incRef()` between check and delete, then
    /// return a `read_handle` to freed memory → heap-use-after-free.
    /// Fixed by mirroring `evict_lru()`: `markForEviction()` atomically
    /// claims the item (sets kExclusive), preventing subsequent `incRef()`.
    ///
    /// Phase 2 (two-pass deferred retirement): The single-pass approach
    /// retired items inside the iteration loop via `hazptr::retire()`.
    /// `retire()` may trigger `maybe_auto_reclaim()` synchronously, and the
    /// background `periodic_worker` calls `try_reclaim_now()` concurrently.
    /// Both paths can free retired items (which are already removed from the
    /// list) while `flush()` is still iterating. Under high load this caused
    /// heap-use-after-free when reading `curr->hook.next_` if the allocator
    /// reused the freed block. The two-pass approach collects items to
    /// retire in Pass 1 and retires them all in Pass 2 after iteration
    /// completes — guaranteeing no item is added to the retire pending
    /// list during iteration.
    void flush() {
        // Pass 1: iterate the list, collect items to retire.
        // Do NOT call retire() inside the loop — deferred to Pass 2.
        std::vector<item_ptr> to_retire;
        auto* curr = list_head();
        while (curr) {
            // Save next before any mutation. remove_from_list() only clears
            // curr's hook pointers; the next node remains valid and linked.
            auto* next = list_get_next(*curr);
            if (!curr->has_active_handle()) {
                // Atomically claim for eviction. If another thread's peek()
                // incRef'd between has_active_handle() and here, markForEviction
                // returns kRefHeld and we skip — the item will be retired via
                // cleanup_pending_deletion() when the handle is released.
                auto evict_result = curr->refcount.markForEviction();
                if (evict_result != detail::MarkForEvictionResult::kSuccess) {
                    curr = next;
                    continue;
                }
                size_type mem = calc_item_memory(curr->key, curr->value);
                stats_.current_memory.fetch_sub(mem);
                callbacks_.collect_evict(curr->key, std::move(curr->value));
                map_.erase(curr->key);
                remove_from_list(curr);
                // Clear the exclusive mark before retiring — the item is
                // about to be reclaimed, no concurrent incRef can succeed.
                curr->refcount.unmarkForEviction();
                // Defer retirement to Pass 2 — do NOT retire here.
                to_retire.push_back(curr);
            } else {
                // Clear tail flag on survivors so tail_size_ can be reset cleanly.
                if (curr->hook.is_tail()) {
                    curr->hook.clear_tail();
                }
            }
            curr = next;
        }
        // P1-8 (T2.6 bugfix, phase 3): Refresh hash stats BEFORE retiring
        // any items. `refresh_hash_stats()` calls `map_.max_chain_length()`,
        // which does a lock-free traversal of hash table bucket chains via
        // `hash_chain_next()` pointers. If we retire items first (Pass 2)
        // and then refresh stats, the background `periodic_worker` could
        // free retired items via `try_reclaim_now()` while `max_chain_length()`
        // is traversing the chain → heap-use-after-free. By refreshing stats
        // here (after `map_.erase()` in Pass 1 but before retirement in
        // Pass 2), items are still alive in `to_retire` — even if
        // `max_chain_length()` somehow encounters a stale pointer, the
        // memory is still valid.
        refresh_hash_stats();
        // Pass 2: retire all collected items after iteration is complete.
        // Now the background reclaimer (or maybe_auto_reclaim) can free them
        // without risking UAF during list traversal or hash chain traversal.
        for (auto* item : to_retire) {
            if (config_.use_ebr) {
                auto* domain = ebr_domain_ ? ebr_domain_ : &detail::epoch_domain::default_domain();
                domain->retire(item);
            } else {
                detail::hazptr_domain::default_domain().retire(item);
            }
        }
        insertion_point_ = nullptr;
        unpinned_tail_ = nullptr;
        tail_size_ = 0;
        insertion_point_pos_ = npos;
        stats_.current_size.store(list_size());
        cleanup_pending_deletion();
    }

    bool contains(const Key& key) const {
        // Physical presence first (lock-free optimistic read).
        if (!map_.contains(key)) return false;
        // TTL-aware: an expired-but-not-yet-reaped entry is reported absent,
        // matching get()/peek_for_get() semantics. Probe the expiry under a
        // short shared bucket lock — contains() is not a hot-path call, so a
        // simple locked probe is clearer than a lock-free one, and the lock
        // also makes the expiry read race-free.
        auto* item = map_.find_embedded_shared(key);
        if (!item) return false;
        if (item->expiry_ns != 0) {
            // Same cached-now fast path as peek_for_get(): cached_now_ns_ is
            // a lower bound on real time, so cached_now >= expiry ⇒ expired.
            std::uint64_t now_ns = cached_now_ns_.load(std::memory_order_relaxed);
            if (now_ns != 0 && now_ns >= item->expiry_ns) return false;
            now_ns = static_cast<std::uint64_t>(
                std::chrono::steady_clock::now().time_since_epoch().count());
            cached_now_ns_.store(now_ns, std::memory_order_relaxed);
            if (now_ns >= item->expiry_ns) return false;
        }
        return true;
    }

    // --------------------------------------------------------------------
    // Capacity
    // --------------------------------------------------------------------

    bool empty() const noexcept { return list_empty(); }
    size_type size() const noexcept { return list_size(); }
    size_type max_size() const noexcept { return max_size_; }
    size_type max_memory() const noexcept { return max_memory_; }
    size_type current_memory() const noexcept { return stats_.current_memory.load(); }

    /// Estimate memory that would be accounted for an item with the given key
    /// and value, including the fixed item overhead and any custom size
    /// calculators registered via set_key/value_size_calculator().
    size_type estimate_item_memory(const Key& key, const Value& value) const {
        return calc_item_memory(key, value);
    }

    /// Check if any item (in the list or pending deletion) has an active handle.
    /// Useful for verifying that no read_handles outlive the cache.
    bool has_active_handles() const noexcept {
        if (use_segmented_lru()) {
            for (auto it = segmented_items_.begin(); it != segmented_items_.end(); ++it) {
                if (it->has_active_handle()) return true;
            }
        } else {
            for (auto it = items_.begin(); it != items_.end(); ++it) {
                if (it->has_active_handle()) return true;
            }
        }
        for (auto* item : pending_deletion_) {
            if (item->has_active_handle()) return true;
        }
        return false;
    }

    void max_size(size_type new_max) {
        max_size_ = new_max;
        stats_.max_size.store(new_max);
        if (new_max != npos) shrink_to_fit();
    }

    void max_memory(size_type new_max) {
        max_memory_ = new_max;
        stats_.max_memory.store(new_max);
        if (new_max != npos) shrink_to_fit();
    }

    void shrink_to_fit() {
        while (should_evict()) {
            auto old_size = size();
            evict_lru();
            if (size() == old_size) break; // all remaining items pinned/rejected
        }
    }

    // --------------------------------------------------------------------
    // Serialization estimates
    // --------------------------------------------------------------------

    /// Get number of items for serialization header.
    size_type serialized_item_count() const { return size(); }

    /// Get serialized size estimate (items only; metadata extra).
    size_type serialized_size_estimate() const {
        return size() * (sizeof(Key) + sizeof(Value) + sizeof(size_type) * 2);
    }

    // --------------------------------------------------------------------
    // Iterators
    // --------------------------------------------------------------------

    iterator begin() noexcept { return items_.begin(); }
    iterator end() noexcept { return items_.end(); }
    const_iterator begin() const noexcept { return items_.begin(); }
    const_iterator end() const noexcept { return items_.end(); }

    /// Segmented list iterators (use when use_segmented_lru is true)
    using segmented_iterator = typename segmented_item_list::iterator;
    using segmented_const_iterator = typename segmented_item_list::const_iterator;
    using segmented_reverse_iterator = typename segmented_item_list::reverse_iterator;
    using segmented_const_reverse_iterator = typename segmented_item_list::const_reverse_iterator;

    segmented_iterator segmented_begin() noexcept { return segmented_items_.begin(); }
    segmented_iterator segmented_end() noexcept { return segmented_items_.end(); }
    segmented_const_iterator segmented_begin() const noexcept { return segmented_items_.begin(); }
    segmented_const_iterator segmented_end() const noexcept { return segmented_items_.end(); }

    /// Reverse iterators: traverse from LRU tail to MRU head.
    /// rbegin() 指向 LRU 端, rend() 指向 MRU 端之前。
    reverse_iterator rbegin() noexcept { return items_.rbegin(); }
    reverse_iterator rend() noexcept { return items_.rend(); }
    const_reverse_iterator rbegin() const noexcept { return items_.rbegin(); }
    const_reverse_iterator rend() const noexcept { return items_.rend(); }

    // --------------------------------------------------------------------
    // LockedIterator (B7)
    // --------------------------------------------------------------------

    /// B7: 持有锁的迭代器，同一时刻仅一个活跃实例（对齐 CacheLib MMLru.h:278-315）。
    /// 从 LRU tail（淘汰端）开始遍历，向 MRU head 方向移动。
    /// 使用 locked_iterator_guard<shared_spinlock> 消除锁生命周期与 active flag 管理代码。
    /// LockedIterator 持有 exclusive lock（用于淘汰遍历）。
    class LockedIterator {
    public:
        LockedIterator(mm_lru& mm)
            : guard_(mm.update_mutex_.m, mm.iterator_active_), mm_(&mm) {
            // B7: 从 LRU tail（淘汰端）开始 → 对齐 CacheLib MMLru.h:704
            curr_ = mm_->list_tail();
        }

        ~LockedIterator() = default;

        LockedIterator(const LockedIterator&) = delete;
        LockedIterator& operator=(const LockedIterator&) = delete;
        LockedIterator(LockedIterator&& other) noexcept
            : guard_(std::move(other.guard_)), mm_(other.mm_), curr_(other.curr_) {}

        void destroy() { guard_.destroy(); }

        /// 重置到 LRU tail（淘汰端）
        void resetToBegin() {
            curr_ = mm_->list_tail();
        }

        /// 向 MRU head 移动一步（通过 get_prev 从 tail→head 遍历）
        bool next() {
            if (!curr_) return false;
            curr_ = mm_->list_get_prev(*curr_);
            return curr_ != nullptr;
        }

        auto& operator*() { return *curr_; }
        auto* operator->() { return curr_; }
        explicit operator bool() const { return curr_ != nullptr; }

    private:
        detail::locked_iterator_guard<detail::shared_spinlock> guard_;
        typename mm_lru::item_ptr curr_;
        mm_lru* mm_;
        bool valid_ = true;
    };

    // --------------------------------------------------------------------
    // Statistics and callbacks
    // --------------------------------------------------------------------

    stats_type& stats() noexcept { return stats_; }
    const stats_type& stats() const noexcept { return stats_; }
    callback_mgr& callbacks() noexcept { return callbacks_; }
    const callback_mgr& callbacks() const noexcept { return callbacks_; }

    // P1-7: Number of items in pending-deletion state (removed from cache
    // but still pinned by active read_handles). Best-effort count — may
    // race with concurrent writes. For monitoring only.
    std::size_t pending_deletion_count() const noexcept {
        return pending_deletion_.size();
    }

    /// Refresh hash table diagnostic stats (load factor, max chain length).
    /// O(bucket_count) scan — call periodically, not on every operation.
    void refresh_hash_stats() const noexcept {
        stats_.hash_load_factor.store(map_.load_factor(), std::memory_order_relaxed);
        stats_.max_chain_length.store(map_.max_chain_length(), std::memory_order_relaxed);
        // P1-1: Refresh rehash diagnostics from the hash table.
        stats_.rehash_count.store(map_.rehash_count(), std::memory_order_relaxed);
        stats_.rehash_total_time_ns.store(map_.rehash_total_time_ns(), std::memory_order_relaxed);
        stats_.rehash_migrated_items.store(map_.rehash_migrated_items(), std::memory_order_relaxed);
        // T13.1: Refresh overload threshold and event counter from the
        // hash table. These mirror the live state in concurrent_hash_table.
        stats_.hash_overload_threshold.store(map_.hash_overload_threshold(), std::memory_order_relaxed);
        stats_.hash_overload_events.store(map_.hash_overload_events(), std::memory_order_relaxed);
    }

    /// T-B4 (P2-10): Forward to the underlying hash table's diagnostics
    /// cache refresh. Only segmented_concurrent_hash_table implements
    /// this (regular concurrent_hash_table doesn't cache — its
    /// `max_chain_length()` is already a single-table scan, cheap enough
    /// to not warrant caching). For non-segmented tables this is a no-op
    /// via `if constexpr` (zero-cost). The background rehash balancer
    /// invokes this unconditionally.
    void refresh_diagnostics_cache() const noexcept {
        if constexpr (requires { map_.refresh_diagnostics_cache(); }) {
            map_.refresh_diagnostics_cache();
        }
    }

    /// T-B4 (P2-10): Forward to the underlying hash table's age metric.
    /// Returns `std::numeric_limits<std::uint64_t>::max()` if the cache
    /// has never been refreshed or the underlying table doesn't cache.
    /// Operators should check the `segmented_hash_table` flag in
    /// diagnostics() before relying on this value — non-segmented tables
    /// always report max (no cache, so age is meaningless).
    std::uint64_t diagnostics_cache_age_ms() const noexcept {
        if constexpr (requires { map_.diagnostics_cache_age_ms(); }) {
            return map_.diagnostics_cache_age_ms();
        }
        return std::numeric_limits<std::uint64_t>::max();
    }

    // --------------------------------------------------------------------
    // Memory policy
    // --------------------------------------------------------------------

    void set_key_size_calculator(std::function<size_type(const Key&)> func) {
        key_size_fn_ = std::move(func);
    }

    void set_value_size_calculator(std::function<size_type(const Value&)> func) {
        value_size_fn_ = std::move(func);
    }

    // --------------------------------------------------------------------
    // EBR (Epoch-Based Reclamation) integration
    // --------------------------------------------------------------------

    /// T2.1: Set an external epoch domain for EBR-based deferred deletion.
    /// When use_ebr is true in config and ebr_domain is set, evict_lru() and
    /// force_del() use EBR retire instead of hazptr retire. If ebr_domain is
    /// nullptr but use_ebr is true, the default global epoch_domain is used.
    ///
    /// T2.1: The domain is also propagated to the underlying hash table
    /// (map_) so that find_and_pin_lockfree() acquires an epoch_guard at
    /// entry, protecting the read path from concurrent reclamation.
    void set_ebr_domain(detail::epoch_domain* domain) noexcept {
        ebr_domain_ = domain;
        map_.set_ebr_domain(domain);
    }

    /// Get the currently associated EBR domain (may be nullptr).
    detail::epoch_domain* get_ebr_domain() const noexcept { return ebr_domain_; }

    /// T2.4: Check whether EBR mode is active on the underlying hash table.
    /// Returns true only when an EBR domain has been set via set_ebr_domain
    /// (i.e., the hash table's find_and_pin_lockfree acquires epoch_guard).
    /// Note: config_.use_ebr alone (without set_ebr_domain) does NOT make
    /// this return true — in that case, the MM-level peek_for_get guard
    /// still protects reads using the default domain.
    bool is_ebr_mode() const noexcept { return map_.is_ebr_mode(); }

    // --------------------------------------------------------------------
    // Enhanced LRU-specific API
    // --------------------------------------------------------------------

    const mm_lru_config& config() const noexcept { return config_; }

    /// Set custom hash table node allocation/deallocation functions.
    /// When non-null, hash table node_type objects are allocated via alloc_fn
    /// instead of ::operator new, enabling slab allocator integration.
    /// Only effective when EmbeddedChain = false; in EmbeddedChain mode this is ignored.
    void set_hash_alloc_fns(void* (*alloc_fn)(std::size_t), void (*dealloc_fn)(void*)) {
        config_.alloc_fn = alloc_fn;
        config_.dealloc_fn = dealloc_fn;
        map_.set_alloc_fns(alloc_fn, dealloc_fn);
    }

    void set_config(const mm_lru_config& config) {
        config_ = config;
        map_.set_alloc_fns(config.alloc_fn, config.dealloc_fn);
        // A4: 当 lru_insertion_point_spec 从非 0 改为 0 时，清理 tail 段所有项的 kTailFlag。
        // 遍历从 insertion_point_ 向 tail 方向，对齐 CacheLib MMLru.h:614-623。
        if (config_.lru_insertion_point_spec == 0 && insertion_point_ != nullptr) {
            auto* curr = insertion_point_;
            while (tail_size_ != 0) {
                assert(curr != nullptr);
                curr->hook.clear_tail();
                --tail_size_;
                curr = list_get_next(*curr);
            }
            insertion_point_ = nullptr;
        }
        // Recalculate insertion point if spec is non-zero and cache is not empty.
        // Handles both zero→non-zero and non-zero→different-non-zero transitions.
        if (config_.lru_insertion_point_spec != 0 && !list_empty()) {
            update_lru_insertion_point();
        }
        // F1: 不变量——spec 非 0 时必须已有插入点，或刚启动时 tail_size_ 为 0
        assert(config_.lru_insertion_point_spec == 0 ||
               insertion_point_ != nullptr ||
               tail_size_ == 0);
        lru_refresh_time_ = config.default_lru_refresh_time;
        next_reconfigure_time_ = config.mm_reconfigure_interval_secs == 0
            ? std::numeric_limits<uint32_t>::max()
            : current_time_sec() + config.mm_reconfigure_interval_secs;
    }

    /// Pre-allocate hash table buckets for `expected_items` entries.
    /// Prevents runtime rehash stalls when the eventual item count is known
    /// in advance (e.g., during cache warm-up).  No-op if already sufficient.
    void reserve(size_type expected_items) {
        map_.reserve(expected_items);
    }

    /// Enable/disable incremental rehash for the hash table.
    /// When enabled, rehash migrates buckets incrementally across multiple
    /// operations instead of blocking all writers during a single rehash.
    /// This reduces write-path latency spikes under load.
    void set_incremental_rehash(bool enabled) {
        map_.set_incremental_rehash(enabled);
    }

    /// Query whether incremental rehash is enabled.
    bool incremental_rehash_enabled() const noexcept {
        return map_.incremental_rehash_enabled();
    }

    /// P0-5 (T1.3): Advance any in-progress incremental rehash by one
    /// per-call migration budget (kRehashFinishMaxBucketsPerCall).
    /// Called by the background rehash balancer to ensure stalled
    /// rehashes eventually complete without requiring writes to the
    /// affected hash table. No-op when no rehash is in progress.
    void advance_incremental_rehash() noexcept {
        map_.rehash_finish();
    }

    /// T11.5: String-based strategy setter (see concurrent_hash_table::set_rehash_strategy).
    bool set_rehash_strategy(std::string_view strategy) noexcept {
        return map_.set_rehash_strategy(strategy);
    }
    std::string_view rehash_strategy() const noexcept {
        return map_.rehash_strategy();
    }

    /// T11.3: Number of writes blocked by a non-incremental (blocking) rehash.
    std::size_t rehash_blocked_writes_count() const noexcept {
        return map_.rehash_blocked_writes_count();
    }

    /// P1-5: Number of times find_and_pin_lockfree fell back to the
    /// lock-protected path because the target segment was in incremental
    /// rehash. Non-zero values indicate the lock-free read path is being
    /// degraded by rehash activity.
    std::size_t rehash_lockfree_fallback_count() const noexcept {
        return map_.rehash_lockfree_fallback_count();
    }

    /// P0-D: Ratio of the hash table currently in an incremental rehash.
    /// For non-segmented tables: 0.0 or 1.0 (whole table rehashing or not).
    /// For segmented tables: fraction of segments currently rehashing.
    /// Exposed as a Prometheus gauge to detect sustained rehash pressure.
    float rehash_in_progress_ratio() const noexcept {
        return map_.rehash_in_progress_ratio();
    }

    /// T13.1: Set the hash table load factor overload threshold.
    /// See concurrent_hash_table::set_hash_overload_threshold.
    void set_hash_overload_threshold(float threshold) noexcept {
        map_.set_hash_overload_threshold(threshold);
    }

    float hash_overload_threshold() const noexcept {
        return map_.hash_overload_threshold();
    }

    std::size_t hash_overload_events() const noexcept {
        return map_.hash_overload_events();
    }

    /// T13.2: Register an overload callback on the underlying hash table.
    void set_overload_callback(std::function<void(float, float)> cb) {
        map_.set_overload_callback(std::move(cb));
    }

    /// P2-4 (T2.4): Toggle async mode for the overload callback.
    /// Forwarded to the underlying hash table. See
    /// `concurrent_hash_table::set_async_overload_callback` for semantics.
    void set_async_overload_callback(bool enabled) noexcept {
        map_.set_async_overload_callback(enabled);
    }

    /// P2-4 (T2.4): Drain pending overload events from the underlying
    /// hash table and dispatch the registered callback for each. Returns
    /// the number of events drained. Designed to be called from a
    /// background worker (e.g. the `event_drain_worker` in `unified_cache`).
    std::size_t drain_overload_callbacks() {
        return map_.drain_overload_callbacks();
    }

    /// Whether an incremental rehash is currently in progress.
    bool is_rehashing() const noexcept { return map_.is_rehashing(); }
    /// Buckets fully migrated so far during the in-progress rehash (0 if none).
    size_type rehash_progress() const noexcept { return map_.rehash_progress(); }
    /// New bucket count target for the in-progress rehash (0 if none).
    size_type rehash_new_bucket_count() const noexcept { return map_.rehash_new_bucket_count(); }
    /// Old bucket count for the in-progress rehash (0 if none).
    size_type rehash_old_bucket_count() const noexcept { return map_.rehash_old_bucket_count(); }
    /// Total number of hash table buckets currently allocated.
    size_type bucket_count() const noexcept { return map_.bucket_count(); }

    uint32_t refresh_time() const noexcept { return lru_refresh_time_; }

    /// A5: 返回 try_lock_update 配置，用于 record_access 内的 try_to_lock 优化
    bool try_lock_update_enabled() const noexcept { return config_.try_lock_update; }

    /// E1: 淘汰年龄统计结构（对齐 CacheLib EvictionAgeStat 的 warmQueueStat）
    struct eviction_age_stat {
        uint64_t oldest_element_age{0};  // tail 节点的年龄（当前时间 - update_time）
        uint64_t projected_age{0};       // 前瞻 projected_length 个新项后的 oldest_element_age
    };

    /// E1: 获取淘汰年龄统计。
    /// projected_length > 0 时，前瞻计算插入 projected_length 个新项后的 oldest_element_age。
    /// 对齐 CacheLib MMLru.h:582-608。
    eviction_age_stat get_eviction_age_stat(std::size_t projected_length = 0) const noexcept {
        eviction_age_stat stat;
        const auto curr_time = current_time_sec();
        const auto* node = list_tail();
        stat.oldest_element_age = node ? (curr_time - node->hook.update_time) : 0;
        // 前瞻：从 tail 向 head 方向走 projected_length 步，模拟插入新项后 tail 的位置
        for (std::size_t seen = 0; seen < projected_length && node != nullptr; ++seen) {
            node = list_get_prev(*node);
        }
        stat.projected_age = node ? (curr_time - node->hook.update_time)
                                  : stat.oldest_element_age;
        return stat;
    }

    /// Peek at LRU tail item's age (seconds since last update)
    std::optional<uint32_t> tail_age() const {
        auto* tail = list_tail();
        if (!tail) return std::nullopt;
        auto curr = current_time_sec();
        return curr - tail->hook.update_time;
    }

    size_type insertion_point_position() const {
        return insertion_point_pos_;
    }

    /// S0: 获取 tail 段大小（用于序列化 faithful restore）。
    size_type tail_size() const noexcept { return tail_size_; }

    /// Promote an item by key without triggering hit statistics or callbacks.
    /// This is a side-effect-free alternative to get() for batch promotion
    /// (e.g., from TLS ring flush). Returns true if the key was found and
    /// the item was eligible for promotion.
    bool promote(const Key& key) {
        auto ptr = map_.find(key);
        if (!ptr) return false;
        return record_access(ptr, access_mode::read);
    }

    // --------------------------------------------------------------------
    // Eviction (public for pooled_cache / unified_cache::evict())
    // --------------------------------------------------------------------

    /// Evict the LRU item from the cache. Fires eviction callbacks.
    /// Uses markForEviction() to atomically claim the item for eviction;
    /// if another thread holds a reference, the item is skipped.
    ///
    /// Chained items: When the victim's refcount_with_flags has
    /// kHasChainedItem set, the victim's value is a chained_value whose
    /// destructor automatically deallocates all chain chunks via
    /// chained_value::clear(). No separate traversal of the chain is
    /// needed — destroying the parent item (via `delete victim`) invokes
    /// the value destructor, which in turn frees every chunk in the chain.
    void evict_lru() {
        cleanup_pending_deletion();
        auto* victim = find_eviction_victim();
        if (victim) {
            // Atomically mark for eviction. If another thread grabbed a ref
            // between find_eviction_victim() and here, markForEviction will
            // fail and we skip this item.
            auto evict_result = victim->refcount.markForEviction();
            if (evict_result != detail::MarkForEvictionResult::kSuccess) {
                // Could not claim — skip (another thread holds a ref or item
                // is already exclusive). No state change was made.
                // R5: Don't reset unpinned_tail_ to nullptr — find_eviction_victim()
                // already set it to the victim's previous item, which is still
                // a valid starting point for the next eviction. Resetting to
                // nullptr would force a full scan from list_tail() next time.
                stats_.current_size.store(list_size());
                return;
            }

            const auto& key = victim->key;
            size_type mem = calc_item_memory(key, victim->value);

            stats_.current_memory.fetch_sub(mem);
            stats_.register_eviction();
            if (callbacks_.has_eviction_callbacks()) {
                Value value = std::move(victim->value);
                callbacks_.collect_evict(key, std::move(value));
            }

            map_.erase(key);
            remove_from_list(victim);

            // R5: find_eviction_victim() already set unpinned_tail_ to the
            // victim's previous item. After remove_from_list(), that item is
            // now the new tail (or closer to it). Only fall back to list_tail()
            // if unpinned_tail_ was not set (e.g., victim was the list head).
            if (!unpinned_tail_) {
                unpinned_tail_ = list_tail();
            }

            // Clear the exclusive mark before retiring the item.
            // Use hazard pointer retire() instead of delete: if a concurrent
            // iterator holds a hazptr protecting this node, deletion is
            // deferred until the hazard is released. When EBR is enabled,
            // use epoch_domain::retire() instead for faster read-path overhead.
            victim->refcount.unmarkForEviction();
            if (config_.use_ebr) {
                auto* domain = ebr_domain_ ? ebr_domain_ : &detail::epoch_domain::default_domain();
                domain->retire(victim);
            } else {
                detail::hazptr_domain::default_domain().retire(victim);
            }
        }
        stats_.current_size.store(list_size());
    }

    /// B15: 找到可淘汰的节点，考虑 EvictionPredicate 和活跃句柄。
    /// H0: 跳过 has_active_handle() 的节点，防止淘汰正在被引用的 item。
    /// Optimized: starts from unpinned_tail_ when valid, skipping the entire
    /// pinned tail region in O(1). Falls back to list_tail() if stale.
    ///
    /// R5: Improved unpinned_tail_ management:
    /// - When walking past pinned items, update unpinned_tail_ to the first
    ///   unpinned item found (even if not a valid victim), so subsequent
    ///   evictions skip the pinned prefix immediately.
    /// - After finding a victim, set unpinned_tail_ to the victim's previous
    ///   item (the new tail candidate after eviction), not the victim itself.
    /// - Don't reset to nullptr when no victim found — keep the last known
    ///   unpinned position for the next eviction attempt.
    item_ptr find_eviction_victim() {
        // Start from unpinned_tail_ if valid, otherwise from list_tail()
        auto* curr = unpinned_tail_;
        if (!curr || curr->has_active_handle()) {
            curr = list_tail();
        }
        if (!curr) return curr;

        const bool has_pred = static_cast<bool>(eviction_predicate_);
        size_t tries = 0;
        size_t steps = 0;
        // R5: Track the first unpinned item found (for delayed update)
        item_ptr first_unpinned = nullptr;
        while (curr) {
            stats_.eviction_search_steps.fetch_add(1, std::memory_order_relaxed);
            ++steps;
            // H0: 跳过有活跃句柄的节点（不计入 tries，这些节点绝对不能淘汰）
            if (curr->has_active_handle()) {
                stats_.pinned_skip_count.fetch_add(1, std::memory_order_relaxed);
                curr = list_get_prev(*curr);
                continue;
            }
            // R5: Remember the first unpinned item — update unpinned_tail_
            // lazily so the next eviction starts from here, skipping the
            // pinned prefix that we already walked past.
            if (!first_unpinned) {
                first_unpinned = curr;
            }
            // B15: EvictionPredicate 否决（计入 tries）
            if (has_pred && !eviction_predicate_(curr->key, curr->value)) {
                curr = list_get_prev(*curr);
                if (++tries >= config_.eviction_search_tries) break;
                continue;
            }
            // Found a victim — update unpinned_tail_ to the victim's
            // previous item (the new tail candidate after this eviction).
            // R5: This is the delayed update — instead of pointing at the
            // victim (which will be removed), point at what will be the
            // new tail after removal.
            unpinned_tail_ = list_get_prev(*curr);
            stats_.eviction_search_steps_hist.record(steps);
            return curr;
        }
        // R5: No victim found — but update unpinned_tail_ to the first
        // unpinned item we found (if any), so the next eviction doesn't
        // re-walk the pinned prefix. Only reset to nullptr if we found
        // no unpinned items at all.
        unpinned_tail_ = first_unpinned;
        if (steps > 0) stats_.eviction_search_steps_hist.record(steps);
        return nullptr;
    }

protected:
    item_list items_;
    segmented_item_list segmented_items_;  // Used when config_.use_segmented_lru == true
    map_type map_;

    size_type max_size_ = unlimited;
    size_type max_memory_ = unlimited;
    mm_lru_config config_;

    // Insertion point tracking (CacheLib's insertionPoint_ + tailSize_)
    item_ptr insertion_point_ = nullptr;
    size_type tail_size_ = 0;
    size_type insertion_point_pos_ = npos;

    /// Optimized eviction: tracks the most-recently-evictable (closest to MRU)
    /// item that has no active handles. find_eviction_victim() starts here
    /// instead of list_tail(), skipping the entire pinned tail region in O(1).
    /// Updated lazily: on eviction and on find_eviction_victim() success.
    /// May become stale (point to a now-pinned item or past the list end),
    /// in which case find_eviction_victim() falls back to list_tail().
    item_ptr unpinned_tail_ = nullptr;

    // Delayed promotion / adaptive refresh
    uint32_t lru_refresh_time_ = 0;
    uint32_t next_reconfigure_time_ = std::numeric_limits<uint32_t>::max();

    // A5: try_lock_update 优化使用的独立内部锁，与统一缓存层锁解耦
    // B10: 缓存行对齐以避免 false sharing（对齐 CacheLib MMLru.h:474）
    // Promotion (read) paths use shared locking; eviction (write) paths use exclusive locking.
    struct alignas(64) aligned_shared_mutex_t { detail::shared_spinlock m; };
    mutable aligned_shared_mutex_t update_mutex_;

    mutable stats_type stats_;
    mutable callback_mgr callbacks_;

    std::function<size_type(const Key&)> key_size_fn_;
    std::function<size_type(const Value&)> value_size_fn_;

    // B15: EvictionPredicate——返回 false 阻止淘汰
    std::function<bool(const Key&, const Value&)> eviction_predicate_;

    // EBR domain (optional; when config_.use_ebr is true, used for retire)
    detail::epoch_domain* ebr_domain_ = nullptr;

    // B7: LockedIterator 活跃标记
    std::atomic<bool> iterator_active_{false};

    // Items removed by force_del() that still have active handles.
    // Memory is freed when all handles are released and cleanup runs.
    std::vector<item_ptr> pending_deletion_;

    // R9: Number of force_del() calls refused because pending_deletion_ was
    // at/over config_.max_pending_deletion. A steadily-increasing count
    // indicates callers are holding read_handles too long (potential leak).
    alignas(64) std::atomic<std::size_t> pending_deletion_skipped_count_{0};

    /// R9: True when the pending_deletion soft cap is configured and reached.
    bool pending_deletion_at_cap() const noexcept {
        return config_.max_pending_deletion > 0 &&
               pending_deletion_.size() >= config_.max_pending_deletion;
    }

public:
    /// R9: Number of force_del() calls refused because the pending-deletion
    /// soft cap was reached. See config_.max_pending_deletion.
    std::size_t pending_deletion_skipped_count() const noexcept {
        return pending_deletion_skipped_count_.load(std::memory_order_relaxed);
    }

private:

    // P0-2: TTL min-heap for O(log n) expired-item lookup.
    // Lazy heap: entries may be stale (key updated to a later expiry or
    // removed). evict_expired() validates entries when popping.
    // Stored as a min-heap via std::push_heap / std::pop_heap with greater<>.
    struct ttl_heap_entry {
        std::uint64_t expiry_ns;
        Key key;
        bool operator>(const ttl_heap_entry& other) const noexcept {
            return expiry_ns > other.expiry_ns;
        }
    };
    std::vector<ttl_heap_entry> ttl_heap_;

    // P1-8: Cached current time for TTL hot path. Avoids steady_clock::now()
    // on every TTL get. Updated lazily when the drift exceeds max_drift,
    // or when refresh_cached_now() is called (e.g. from background cleaner).
    //
    // P2-B: `cached_now_ns_` is on the TTL get hot path (read on every
    // TTL-aware lookup) and is written by both the get path (lazy refresh)
    // and the background cleaner. Without isolation, it shares a cache
    // line with `ttl_heap_` (whose data pointer / size / capacity mutate
    // on every TTL insert / heap pop) — every TTL get that refreshes the
    // cached time would invalidate the cache line for threads reading
    // `ttl_heap_`, and vice versa. Pin it to its own 64-byte line to
    // eliminate the false sharing.
    alignas(64) mutable std::atomic<std::uint64_t> cached_now_ns_{0};

    // ====================================================================
    // Segmented LRU dispatch helpers
    // ====================================================================

    /// Whether segmented LRU is active.
    bool use_segmented_lru() const noexcept { return config_.use_segmented_lru; }

    /// Dispatch: link_at_head
    void list_link_at_head(item_type& item) {
        if (use_segmented_lru()) {
            segmented_items_.link_at_head(item);
        } else {
            items_.link_at_head(item);
        }
    }

    /// Dispatch: link_at_tail
    void list_link_at_tail(item_type& item) {
        if (use_segmented_lru()) {
            segmented_items_.link_at_tail(item);
        } else {
            items_.link_at_tail(item);
        }
    }

    /// Dispatch: insert_before
    void list_insert_before(item_type& next_node, item_type& item) {
        if (use_segmented_lru()) {
            segmented_items_.insert_before(next_node, item);
        } else {
            items_.insert_before(next_node, item);
        }
    }

    /// Dispatch: remove
    void list_remove(item_type& item) {
        if (use_segmented_lru()) {
            segmented_items_.remove(item);
        } else {
            items_.remove(item);
        }
    }

    /// Dispatch: replace
    void list_replace(item_type& old_node, item_type& new_node) {
        if (use_segmented_lru()) {
            segmented_items_.replace(old_node, new_node);
        } else {
            items_.replace(old_node, new_node);
        }
    }

    /// Dispatch: head
    item_type* list_head() const {
        if (use_segmented_lru()) {
            return segmented_items_.head();
        }
        return items_.head();
    }

    /// Dispatch: tail
    item_type* list_tail() const {
        if (use_segmented_lru()) {
            return segmented_items_.tail();
        }
        return items_.tail();
    }

    /// Dispatch: size
    size_type list_size() const {
        if (use_segmented_lru()) {
            return segmented_items_.size();
        }
        return items_.size();
    }

    /// Dispatch: empty
    bool list_empty() const {
        if (use_segmented_lru()) {
            return segmented_items_.empty();
        }
        return items_.empty();
    }

    /// Dispatch: get_next
    item_type* list_get_next(const item_type& node) const {
        if (use_segmented_lru()) {
            return segmented_items_.get_next(node);
        }
        return items_.get_next(node);
    }

    /// Dispatch: get_prev
    item_type* list_get_prev(const item_type& node) const {
        if (use_segmented_lru()) {
            return segmented_items_.get_prev(node);
        }
        return items_.get_prev(node);
    }

    /// Dispatch: pop_tail
    item_type* list_pop_tail() {
        if (use_segmented_lru()) {
            return segmented_items_.pop_tail();
        }
        return items_.pop_tail();
    }

    /// Dispatch: clear
    void list_clear() {
        if (use_segmented_lru()) {
            segmented_items_.clear();
        } else {
            items_.clear();
        }
    }

    /// Clean up pending-deletion items whose handles have all been released.
    /// Now also fires the deferred eviction callback (with value move) before
    /// deleting the item, which was skipped in force_del() when handles were
    /// still active.
    void cleanup_pending_deletion() {
        auto it = pending_deletion_.begin();
        while (it != pending_deletion_.end()) {
            if (!(*it)->has_active_handle()) {
                auto* item = *it;
                callbacks_.collect_evict(item->key, std::move(item->value));
                // P1-5: Route through hazptr/EBR retire instead of raw delete.
                // A concurrent hazptr-protected reader (iterator or hash-table
                // traversal) may still hold a hazard pointer to this item even
                // though its refcount is 0 — raw delete would cause UAF.
                // Use the same EBR/hazptr selection as the non-pending path.
                if (config_.use_ebr) {
                    auto* domain = ebr_domain_ ? ebr_domain_ : &detail::epoch_domain::default_domain();
                    domain->retire(item);
                } else {
                    detail::hazptr_domain::default_domain().retire(item);
                }
                it = pending_deletion_.erase(it);
            } else {
                ++it;
            }
        }
    }

    // P0-2: Push a TTL entry into the min-heap. Used on insert and update.
    // Each TTL update pushes a new entry; stale ones are only removed lazily
    // when evict_expired() pops them after expiry. When the heap grows past
    // `kMaxTtlHeapMultiplier * live_items`, rebuild it from the list to
    // discard stale entries and bound heap memory.
    static constexpr std::size_t kMaxTtlHeapMultiplier = 4;

    void ttl_heap_push(const Key& key, std::uint64_t expiry_ns) {
        if (expiry_ns == 0) return;
        ttl_heap_.push_back(ttl_heap_entry{expiry_ns, key});
        std::push_heap(ttl_heap_.begin(), ttl_heap_.end(),
                       std::greater<ttl_heap_entry>{});
        // Bound heap growth: repeated set_with_expiry() updates push one new
        // entry each while the stale one stays until its (old) expiry is
        // popped. When the heap far exceeds the live item count, most entries
        // are stale — rebuild from the list. Amortized O(1) per push (the
        // O(list_size) rebuild fires only once per multiplier worth of pushes).
        const std::size_t live = list_size();
        if (ttl_heap_.size() > kMaxTtlHeapMultiplier * live) {
            ttl_heap_rebuild();
        }
    }

    /// Rebuild the TTL min-heap from the live items on the LRU list,
    /// discarding entries for keys that were removed or whose expiry was
    /// updated. Must be called under the write lock (list mutation would
    /// otherwise race); the iteration is read-only.
    void ttl_heap_rebuild() {
        ttl_heap_.clear();
        for (auto* curr = list_head(); curr != nullptr; curr = list_get_next(*curr)) {
            if (curr->expiry_ns != 0) {
                ttl_heap_.push_back(ttl_heap_entry{curr->expiry_ns, curr->key});
            }
        }
        std::make_heap(ttl_heap_.begin(), ttl_heap_.end(),
                       std::greater<ttl_heap_entry>{});
    }

    // --------------------------------------------------------------------
    // Time utilities
    // --------------------------------------------------------------------

    static uint32_t current_time_sec() {
        return detail::cached_epoch_sec();
    }

    // --------------------------------------------------------------------
    // Internal helpers
    // --------------------------------------------------------------------

    bool should_evict() const {
        if (max_size_ != unlimited && size() > max_size_) return true;
        if (max_memory_ != unlimited && current_memory() > max_memory_) return true;
        return false;
    }

    bool is_at_capacity() const {
        return max_size_ != unlimited && size() >= max_size_;
    }

    size_type calc_item_memory(const Key& key, const Value& value) const {
        size_type mem = item_overhead;
        if (key_size_fn_) [[unlikely]] mem += key_size_fn_(key) * 2;
        if (value_size_fn_) [[unlikely]] mem += value_size_fn_(value);
        return mem;
    }

    // ====================================================================
    // Delayed Promotion (CacheLib's recordAccess)
    // ====================================================================

    /// Record access to an item, with delayed promotion support.
    /// Returns true if the item was actually promoted.
    /// D5: 增加 is_linked() 运行时守卫（对齐 CacheLib MMLru.h:539 node.isInMMContainer）。
    bool record_access(item_ptr item, access_mode mode) {
        return record_access_at(item, mode, current_time_sec());
    }

    /// Record access with a pre-computed current time.
    bool record_access_at(item_ptr item, access_mode mode, uint32_t curr) {
        if ((mode == access_mode::write && !config_.update_on_write) ||
            (mode == access_mode::read && !config_.update_on_read)) [[unlikely]] {
            return false;
        }
        if (!item->refcount.isInMMContainer()) [[unlikely]] {
            return false;
        }

        if (!item->hook.is_accessed()) {
            item->hook.set_accessed();
        } else if (curr < item->hook.update_time + lru_refresh_time_) [[likely]] {
            return false;
        }

        auto promote = [this, item, curr]() {
            reconfigure_locked(curr);
            ensure_not_insertion_point(item);
            if (config_.use_segmented_lru) {
                segmented_items_.move_to_head(*item);
            } else {
                items_.move_to_head(*item);
            }
            item->hook.update_time = curr;
            if (item->hook.is_tail()) {
                item->hook.clear_tail();
                --tail_size_;
                update_lru_insertion_point();
            }
        };

        if (config_.use_segmented_lru) {
            // Segmented LRU: use per-segment spinlocks instead of global update_mutex_.
            // move_to_head() locks only the source segment + segment 0 (MRU),
            // enabling concurrent promotions of items in different segments.
            auto src_idx = item->hook.segment_idx;
            if (try_lock_update_enabled()) {
                // Try-lock mode: attempt to acquire segment locks without blocking.
                auto result = segmented_items_.try_lock_two_segments(src_idx, 0);
                if (!result.success) {
                    stats_.try_lock_fail_count.fetch_add(1, std::memory_order_relaxed);
                    return false;
                }
                promote();
            } else {
                // Blocking mode: acquire segment locks in address order (handled by move_to_head).
                // But we still need to coordinate with reconfigure_locked and insertion point.
                // Use the global update_mutex_ as a fallback for non-try_lock path
                // to avoid complex lock ordering with segment locks.
                // G5: promote() calls move_to_head() which mutates the intrusive list
                // (prev/next pointers), so it is a WRITE operation and requires an
                // EXCLUSIVE lock — not a shared lock. Previously this used
                // shared_scoped_lock under the mistaken assumption that "promotion
                // is a read path"; that caused a data race on list pointers when
                // two threads held the shared lock concurrently. Aligned with
                // mm_2q / mm_tiny_lfu / mm_fifo which all use exclusive locking.
                std::unique_lock<detail::shared_spinlock> lock(update_mutex_.m);
                promote();
            }
        } else if (try_lock_update_enabled()) {
            // G5: promote() mutates the list (move_to_head) — use EXCLUSIVE lock,
            // not shared. Aligned with mm_2q (unique_lock<std::mutex>).
            std::unique_lock<detail::shared_spinlock> lock(update_mutex_.m, std::try_to_lock);
            if (!lock) {
                stats_.try_lock_fail_count.fetch_add(1, std::memory_order_relaxed);
                return false;
            }
            promote();
        } else {
            // G5: Callers entering this branch (try_lock_update=false) MUST hold
            // an external exclusive lock (e.g. unified_cache's per-key write lock)
            // that serializes all list mutations. This matches mm_2q's else branch
            // which also calls promote() lock-free under the same contract.
            promote();
        }
        return true;
    }

    // ====================================================================
    // Adaptive Refresh Time (CacheLib's reconfigureLocked)
    // ====================================================================

    void reconfigure_locked(uint32_t curr_time) {
        if (curr_time < next_reconfigure_time_) return;

        next_reconfigure_time_ = curr_time + config_.mm_reconfigure_interval_secs;

        // E1: 通过 get_eviction_age_stat 获取 oldest_element_age，对齐 CacheLib MMLru.h:855。
        // 当 ratio=0 或 tail 为空时，结果退化为 default_lru_refresh_time，行为与原实现等价。
        auto stat = get_eviction_age_stat(0);
        auto new_refresh = std::min(
            std::max(config_.default_lru_refresh_time,
                     static_cast<uint32_t>(static_cast<double>(stat.oldest_element_age) *
                                           config_.lru_refresh_ratio)),
            config_type::k_lru_refresh_time_cap);
        lru_refresh_time_ = new_refresh;
    }

    // ====================================================================
    // Incremental Insertion Point (CacheLib's updateLruInsertionPoint)
    // ====================================================================

    /// Update the insertion point incrementally (O(1) amortized).
    /// Instead of recalculating from scratch, we grow/shrink the tail
    /// section by marking/unmarking items.
    void update_lru_insertion_point() {
        if (config_.lru_insertion_point_spec == 0) { insertion_point_pos_ = npos; return; }

        // F1: 不变量——tail 段大小不能超过当前缓存大小
        assert(tail_size_ <= list_size());

        // Initialize insertion point to tail if null
        if (!insertion_point_) {
            insertion_point_ = list_tail();
            tail_size_ = 0;
            if (insertion_point_) {
                insertion_point_->hook.set_tail();
                ++tail_size_;
            }
        }

        if (list_size() <= 1) {
            if (insertion_point_) {
                insertion_point_pos_ = list_size() - tail_size_;
            } else {
                insertion_point_pos_ = npos;
            }
            return;
        }

        // F1: 初始化后插入点必须有效（items 非空时）
        assert(insertion_point_ != nullptr);

        auto expected_size = list_size() >> config_.lru_insertion_point_spec;
        auto* curr = insertion_point_;

        // Grow tail section: move insertion point toward head
        while (tail_size_ < expected_size && curr != list_head()) {
            curr = list_get_prev(*curr);
            if (curr) {
                curr->hook.set_tail();
                ++tail_size_;
            }
        }

        // Shrink tail section: move insertion point toward tail
        while (tail_size_ > expected_size && curr != list_tail()) {
            curr->hook.clear_tail();
            --tail_size_;
            curr = list_get_next(*curr);
        }

        insertion_point_ = curr;

        // Update cached position: insertion point position = total size - tail_size
        if (insertion_point_) {
            insertion_point_pos_ = list_size() - tail_size_;
        } else {
            insertion_point_pos_ = npos;
        }
    }

    /// Ensure the item being moved/removed is not the insertion point.
    /// If it is, grow the tail section first so insertion_point_ remains valid.
    void ensure_not_insertion_point(item_ptr item) {
        // F1: 节点指针必须有效
        assert(item != nullptr);
        if (item == insertion_point_) {
            // F1: 若 item 是插入点，则插入点本身必须有效
            assert(insertion_point_ != nullptr);
            insertion_point_ = list_get_prev(*insertion_point_);
            if (insertion_point_) {
                ++tail_size_;
                insertion_point_->hook.set_tail();
            }
            // D1: 插入点变为 nullptr 时断言列表中仅剩一个节点
            // （对齐 CacheLib MMLru.h:735 ensureNotInsertionPoint）
            assert(insertion_point_ != nullptr || list_size() == 1);
        }
    }

    // ====================================================================
    // Insert / Update / Evict
    // ====================================================================

    template <typename V>
    void insert_new(const Key& key, V&& value, std::uint64_t expiry_ns = 0) {
        // Zero max_size or max_memory means no insert capacity (updates to existing keys still allowed).
        if (max_size_ == 0 || max_memory_ == 0) return;

        // Clean up any deferred deletions whose handles have been released.
        cleanup_pending_deletion();

        // Make room under the size cap; if every candidate is pinned/rejected,
        // do not insert the new item.
        if (max_size_ != unlimited && size() >= max_size_) {
            evict_lru();
            if (size() >= max_size_) return;
        }
        while (should_evict()) {
            auto old_size = size();
            evict_lru();
            if (size() == old_size) return;
        }

        auto mem = calc_item_memory(key, value);
        if (max_memory_ != unlimited) {
            while (!list_empty() &&
                   stats_.current_memory.load() + mem > max_memory_) {
                auto old_size = size();
                evict_lru();
                if (size() == old_size) return;
            }
        }

        // Allocate new item
        auto* item = this->allocate_item(key, std::forward<V>(value));
        auto curr = current_time_sec();
        item->hook.update_time = curr;
        item->hook.clear_accessed();  // Not yet accessed
        // P0-3: set TTL expiry if provided (set_with_expiry path). The
        // corresponding ttl_heap_push() happens AFTER the item is fully
        // linked (list + map) below, so a rebuild triggered by the push
        // sees the new item.
        if (expiry_ns != 0) {
            item->expiry_ns = expiry_ns;
        }

        // P-MED-1 (T-H3): RAII rollback guard. If map_.insert throws
        // (e.g. bad_alloc), the item would be linked in the intrusive list
        // but unreachable from the hash table, corrupting list_size() and
        // leaking memory. The guard undoes the list link and deallocates
        // the item on exception, then is dismissed once map_.insert succeeds.
        struct insert_rollback_guard {
            mm_lru* self;
            item_type* item;
            bool linked = false;
            bool committed = false;
            ~insert_rollback_guard() {
                if (committed) return;
                if (linked) {
                    self->list_remove(*item);
                }
                delete item;
            }
        } guard{this, item, false, false};

        // Insert into list based on insertion point config
        if (config_.lru_insertion_point_spec == 0 || !insertion_point_) {
            list_link_at_head(*item);
        } else {
            list_insert_before(*insertion_point_, *item);
            // A1: 不在 insert_new 中递增 tail_size_，由 update_lru_insertion_point 统一维护
        }
        guard.linked = true;

        // Sync refcount kLinked bit — intrusive_list sets the hook flag,
        // we set the refcount flag for markForEviction/markMoving consistency.
        item->refcount.markInMMContainer();

        map_.insert(key, item);  // may throw bad_alloc — guard undoes list link

        // P-MED-1 (T-H3): Commit — item is now fully in both list and map.
        guard.committed = true;

        // P0-2: insert into TTL min-heap AFTER successful map_.insert, so
        // that a throw from map_.insert does not leave a dangling entry.
        // (Previously this was before the list link, risking heap entries
        // pointing to items that were never fully inserted.)
        if (expiry_ns != 0) {
            ttl_heap_push(key, expiry_ns);
        }

        stats_.current_size.store(list_size());
        stats_.current_memory.fetch_add(mem);
        stats_.register_insertion();
        callbacks_.collect_insert(key, item->value);

        update_lru_insertion_point();
    }

    template <typename V>
    void update_existing(item_ptr item, V&& value, access_mode mode) {
        auto curr = current_time_sec();
        size_type old_mem = calc_item_memory(item->key, item->value);
        // P-MED-2 (T-H4): Strong exception guarantee via copy-then-swap.
        // Construct tmp first (may throw — item->value unchanged on failure),
        // then noexcept swap commits the update atomically. Falls back to
        // direct assignment for Value types that are not nothrow-swappable
        // (e.g. types with throwing swap); those retain the weak guarantee.
        if constexpr (std::is_nothrow_swappable_v<Value> &&
                      std::is_constructible_v<Value, V>) {
            Value tmp(std::forward<V>(value));
            using std::swap;
            swap(item->value, tmp);
        } else {
            item->value = std::forward<V>(value);
        }
        record_access_at(item, mode, curr);
        size_type new_mem = calc_item_memory(item->key, item->value);
        if (new_mem > old_mem) {
            stats_.current_memory.fetch_add(new_mem - old_mem);
        } else if (new_mem < old_mem) {
            stats_.current_memory.fetch_sub(old_mem - new_mem);
        }
        // A value can grow (e.g. vector/string append). If the cache is now
        // over its memory or size budget, evict least-recently-used items.
        // record_access_at promotes the updated item toward MRU, so it will
        // not be the first victim.
        if (should_evict()) {
            shrink_to_fit();
        }
        // O7: Fire on_update for value changes on existing keys (distinct
        // from on_insert, which fires only for new key insertions).
        // warm_cache delta tracking subscribes to on_update to capture
        // value changes on existing keys.
        callbacks_.collect_update(item->key, item->value);
    }

    /// B14: 原位替换节点，保留 update_time 与位置（对齐 CacheLib MMLru.h:773-800）。
    /// 与 replace(key, value) 不同——后者只改值，本方法替换整个节点。
    /// 替换时保留 old_node 的 is_accessed 和 tail 状态（对齐 CacheLib MMLru.h:783-787）。
    /// D5: 增加 is_linked() 运行时守卫（对齐 CacheLib MMLru.h:775）。
    void replace_node(item_ptr old_node, item_ptr new_node) {
        assert(old_node != nullptr && new_node != nullptr);
        // D5: 运行时检查——old_node 必须在容器中，new_node 不能在容器中（对齐 CacheLib MMLru.h:775）
        if (!old_node->refcount.isInMMContainer() || new_node->refcount.isInMMContainer()) {
            return;
        }
        ensure_not_insertion_point(old_node);
        bool was_tail = old_node->hook.is_tail();
        if (was_tail) {
            old_node->hook.clear_tail();
            --tail_size_;
        }
        // D3: 保留 old_node 的 update_time 和 is_accessed 状态（对齐 CacheLib MMLru.h:778-787）
        new_node->hook.update_time = old_node->hook.update_time;
        if (old_node->hook.is_accessed()) {
            new_node->hook.set_accessed();
        } else {
            new_node->hook.clear_accessed();
        }
        list_replace(*old_node, *new_node);
        // 更新 map
        map_.insert_or_assign(old_node->key, new_node);
        // 更新 tail 跟踪
        if (tail_size_ > 0 && insertion_point_ == old_node) {
            insertion_point_ = new_node;
        }
        if (was_tail) {
            new_node->hook.set_tail();
            ++tail_size_;
        }
        update_lru_insertion_point();
    }

    /// B15: 设置淘汰谓词——返回 false 表示该 item 不可淘汰。
    void set_eviction_predicate(std::function<bool(const Key&, const Value&)> pred) {
        eviction_predicate_ = std::move(pred);
    }

    void erase_impl(const Key& key) {
        auto ptr = map_.find(key);
        if (!ptr) return;
        auto* item = ptr;
        if (item->has_active_handle()) return;
        size_type mem = calc_item_memory(item->key, item->value);

        stats_.current_memory.fetch_sub(mem);
        callbacks_.collect_evict(item->key, std::move(item->value));

        map_.erase(key);
        remove_from_list(item);
        unpinned_tail_ = nullptr;
        detail::hazptr_domain::default_domain().retire(item);
        stats_.current_size.store(list_size());
    }

    /// O7: TTL expiration variant of erase_impl — fires on_expire instead
    /// of on_evict so consumers can distinguish capacity-driven evictions
    /// from TTL-driven expirations. The retirement path is identical.
    void erase_expired_impl(const Key& key) {
        auto ptr = map_.find(key);
        if (!ptr) return;
        auto* item = ptr;
        if (item->has_active_handle()) return;
        size_type mem = calc_item_memory(item->key, item->value);

        stats_.current_memory.fetch_sub(mem);
        callbacks_.collect_expire(item->key, std::move(item->value));

        map_.erase(key);
        remove_from_list(item);
        unpinned_tail_ = nullptr;
        detail::hazptr_domain::default_domain().retire(item);
        stats_.current_size.store(list_size());
    }

    /// Remove item from the list and adjust insertion point tracking.
    /// D5: 增加 isInMMContainer() 运行时守卫，防止重复移除。
    void remove_from_list(item_ptr item) {
        // F1: 节点指针必须有效
        assert(item != nullptr);
        // D5: 运行时检查节点仍在容器中（对齐 CacheLib MMLru.h:756 node.isInMMContainer）
        if (!item->refcount.isInMMContainer()) {
            return;
        }
        // F1: tail 段大小不能超过当前缓存大小
        assert(tail_size_ <= list_size());

        ensure_not_insertion_point(item);

        // If item is in the tail section, adjust count
        if (item->hook.is_tail()) {
            item->hook.clear_tail();
            --tail_size_;
        }

        // A2: 清除被移除节点的 accessed 标志，对齐 CacheLib MMLru.h:744
        item->hook.clear_accessed();

        // Sync refcount kLinked bit BEFORE remove() poisons the item's memory.
        item->refcount.unmarkInMMContainer();
        list_remove(*item);
        update_lru_insertion_point();
    }

    // ====================================================================
    // S0: Faithful serialization rebuild (public for deserialization)
    // ====================================================================
public:
    /// Deserialize helper — 按 MRU→LRU 顺序重建缓存，保留链表结构和插入点。
    /// 在调用前需通过 flush() 清空缓存。
    /// @param first         输入迭代器到序列化 items 起始
    /// @param last          输入迭代器到序列化 items 末尾
    /// @param ins_pos       插入点的位置（0=MRU head, n=LruTail, UINT32_MAX=未设置）
    /// @param tail_sz       tail 段大小（spec>0 时有效）
    template <typename InputIt>
    void rebuild_from_serialized(InputIt first, InputIt last,
                                 uint32_t ins_pos, size_type tail_sz) {
        flush();

        // 逐项重建链表（按 MRU→LRU 顺序 link_at_tail）
        for (auto it = first; it != last; ++it) {
            auto* item = this->allocate_item(it->key, it->value);
            item->hook.update_time = it->update_time;
            if (it->flags & detail::intrusive_hook::kAccessedFlag) {
                item->hook.set_accessed();
            }
            item->queue_id = it->queue_id;
            list_link_at_tail(*item);
            item->refcount.markInMMContainer();
            map_.insert(item->key, item);
            // Memory estimate: use calc_item_memory for custom size functions
            stats_.current_memory.fetch_add(calc_item_memory(item->key, item->value));
        }

        stats_.current_size.store(list_size());

        // 重建插入点
        tail_size_ = 0;
        insertion_point_ = nullptr;
        if (ins_pos != std::numeric_limits<uint32_t>::max() && ins_pos < list_size()) {
            auto* curr = list_head();
            for (uint32_t i = 0; i < ins_pos; ++i) {
                curr = list_get_next(*curr);
                if (!curr) break;
            }
            if (curr) {
                insertion_point_ = curr;
                tail_size_ = tail_sz;
                // 从插入点开始向 tail 标记 kTailFlag
                auto* mark = insertion_point_;
                for (size_type i = 0; i < tail_size_ && mark; ++i) {
                    mark->hook.set_tail();
                    mark = list_get_next(*mark);
                }
            }
        }
    }

    // --------------------------------------------------------------------
    // Equality comparison (content and order must match)
    // --------------------------------------------------------------------

    friend bool operator==(const mm_lru& a, const mm_lru& b) {
        if (a.size() != b.size()) return false;
        auto ai = a.begin(), bi = b.begin();
        for (; ai != a.end(); ++ai, ++bi) {
            if (ai->key != bi->key || ai->value != bi->value) return false;
        }
        return true;
    }

    friend bool operator!=(const mm_lru& a, const mm_lru& b) { return !(a == b); }

    // --------------------------------------------------------------------
    // Stream output
    // --------------------------------------------------------------------

    friend std::ostream& operator<<(std::ostream& os, const mm_lru& c) {
        os << "mm_lru @" << &c << "  " << c.stats_ << "\n";
        std::size_t idx = 0;
        for (auto it = c.begin(); it != c.end(); ++it, ++idx) {
            os << "  [" << idx << "] key='";
            if constexpr (lru::detail::is_formattable_v<Key>) {
                os << std::format("{}", it->key);
            } else {
                os << std::format("<key at {:#x}>", reinterpret_cast<std::uintptr_t>(&it->key));
            }
            os << "' value='";
            if constexpr (lru::detail::is_formattable_v<Value>) {
                os << std::format("{}", it->value);
            } else {
                os << std::format("<val at {:#x}>", reinterpret_cast<std::uintptr_t>(&it->value));
            }
            os << "'\n";
        }
        return os;
    }
};


// ============================================================================
// FIFO Strategy - mm_fifo
// ============================================================================

/// Configuration for FIFO eviction.
struct mm_fifo_config {
    /// B15: 淘汰搜索次数上限——当 EvictionPredicate 否决时最多继续搜索的项数。
    size_t eviction_search_tries = 3;

    /// Expected number of items for automatic bucket count sizing.
    /// 0 = use default bucket count (1024). When > 0, the internal hash
    /// table is pre-sized via concurrent_hash_table::buckets_for_items()
    /// to keep average chain length ≤ 0.25 at the expected load.
    size_t expected_items = 0;

    /// Custom node allocation function for non-EmbeddedChain hash table nodes.
    /// nullptr (default) = standard new/delete allocation.
    void* (*alloc_fn)(std::size_t) = nullptr;

    /// Custom node deallocation function (must pair with alloc_fn).
    void  (*dealloc_fn)(void*) = nullptr;

    void validate() const {
        // FIFO config is always valid
    }
};

template <
    typename Key,
    typename Value,
    typename Hash = std::hash<Key>,
    typename KeyEqual = std::equal_to<Key>,
    typename ProbingStyle = detail::chain_probing_tag,
    bool Segmented = false
>
/// FIFO eviction strategy: first-in-first-out. Accessing an item does NOT
/// change its position in the eviction queue. The oldest inserted item
/// (tail of the list) is evicted first.
class mm_fifo : public detail::mm_allocator_mixin<detail::cache_item<Key, Value>> {
public:
    using key_type = Key;
    using mapped_type = Value;
    using size_type = std::size_t;
    using config_type = mm_fifo_config;

    // Item type: cache_item with embedded intrusive hook
    using item_type = detail::cache_item<Key, Value>;
    using item_ptr = item_type*;

    // Intrusive list type
    using item_list = detail::intrusive_list<item_type, detail::intrusive_hook, detail::default_get_hook<item_type>>;
    using iterator = typename item_list::iterator;
    using const_iterator = typename item_list::const_iterator;
    /// Reverse iterator: same as iterator type, but rbegin() starts from
    /// the LRU tail and moves toward the MRU head.
    using reverse_iterator = typename item_list::reverse_iterator;
    using const_reverse_iterator = typename item_list::const_reverse_iterator;

    // Map: Key -> item pointer
    using map_type = std::conditional_t<
        Segmented,
        detail::segmented_concurrent_hash_table<Key, item_ptr, Hash, KeyEqual, true, ProbingStyle, 64>,
        detail::concurrent_hash_table<Key, item_ptr, Hash, KeyEqual, true, ProbingStyle>
    >;

    using callback_mgr = callback_manager<Key, Value>;
    using stats_type = cache_stats;

    static constexpr size_type npos = unlimited;
    static constexpr size_type item_overhead = sizeof(item_type) + map_type::entry_overhead;

    // --------------------------------------------------------------------
    // Constructors / Destructor
    // --------------------------------------------------------------------

    mm_fifo() = default;

    explicit mm_fifo(const mm_fifo_config& cfg)
        : config_(cfg)
        , map_(cfg.expected_items > 0
            ? map_type::buckets_for_items(cfg.expected_items)
            : 1024,
            cfg.alloc_fn, cfg.dealloc_fn) {}

    mm_fifo(size_type max_size)
        : mm_fifo() {
        detail::validate_capacity(max_size, unlimited);
        max_size_ = max_size;
        stats_.max_size.store(max_size);
    }

    mm_fifo(size_type max_size, size_type max_memory)
        : mm_fifo(max_size) {
        max_memory_ = max_memory;
        stats_.max_memory.store(max_memory);
    }

    mm_fifo(size_type max_size, const mm_fifo_config& cfg)
        : mm_fifo(cfg) {
        detail::validate_capacity(max_size, unlimited);
        max_size_ = max_size;
        stats_.max_size.store(max_size);
    }

    mm_fifo(size_type max_size, size_type max_memory, const mm_fifo_config& cfg)
        : mm_fifo(cfg) {
        detail::validate_capacity(max_size, unlimited);
        max_size_ = max_size;
        stats_.max_size.store(max_size);
        max_memory_ = max_memory;
        stats_.max_memory.store(max_memory);
    }

    ~mm_fifo() {
        // Clean up pending deletions first (items with active handles that were force_del'd)
        for (auto* item : pending_deletion_) {
            delete item;
        }
        pending_deletion_.clear();
        // Use flush() to properly erase items from both map and list before deletion,
        // preventing use-after-free when the hash table destructor iterates its chains.
        flush();
    }

    // Non-copyable, non-movable (owns item pointers)
    mm_fifo(const mm_fifo&) = delete;
    mm_fifo& operator=(const mm_fifo&) = delete;

    // --------------------------------------------------------------------
    // Core cache API
    // --------------------------------------------------------------------

    template <typename V>
    void set(const Key& key, V&& value) {
        auto ptr = map_.find(key);
        if (!ptr) {
            insert_new(key, std::forward<V>(value));
        } else {
            update_existing(ptr, std::forward<V>(value));
        }
    }

    template <typename V>
    bool add(const Key& key, V&& value) {
        auto ptr = map_.find(key);
        if (!ptr) {
            insert_new(key, std::forward<V>(value));
            return true;
        }
        return false;
    }

    template <typename V>
    bool replace(const Key& key, V&& value) {
        auto ptr = map_.find(key);
        if (!ptr) return false;
        update_existing(ptr, std::forward<V>(value));
        return true;
    }

    read_handle<Value> get(const Key& key) {
        auto ptr = map_.find(key);
        if (!ptr) {
            stats_.register_miss();
            callbacks_.collect_miss(key);
            return {};
        }
        // FIFO: no promotion on access
        auto* item = ptr;
        stats_.register_hit();
        callbacks_.collect_hit(key, item->value);
        return read_handle<Value>{&item->value, &item->refcount};
    }

    read_handle<const Value> get(const Key& key) const {
        auto ptr = map_.find(key);
        if (!ptr) {
            stats_.register_miss();
            callbacks_.collect_miss(key);
            return {};
        }
        auto* item = ptr;
        stats_.register_hit();
        callbacks_.collect_hit(key, item->value);
        return read_handle<const Value>{&item->value, &item->refcount};
    }

    /// H0: Peek with handle — 不提升 LRU，返回 handle 防止持有期被淘汰。
    /// Uses find_and_pin_lockfree() to attempt lock-free pinning first
    /// (optimistic read + incRef without bucket lock), falling back to
    /// find_and_pin() (shared lock path) if the lock-free pin fails.
    /// No stripe-level read lock is needed.
    read_handle<const Value> peek(const Key& key) const {
        auto pin_fn = [](auto* item) {
            return item->refcount.incRef() == detail::IncResult::kIncOk;
        };
        auto ptr = map_.find_and_pin_lockfree(key, pin_fn);
        if (!ptr) return {};
        return read_handle<const Value>{&ptr->value, &ptr->refcount, pre_pinned};
    }

    /// Internal: peek that returns mutable handle (for optimistic get path).
    read_handle<Value> peek_for_get(const Key& key) {
        return peek_for_get_with_hash(key, Hash{}(key));
    }

    /// T16.4: peek_for_get with a pre-computed hash. The hash MUST be
    /// the result of Hash{}(key) — callers are responsible for hash
    /// compatibility. Used by bulk_get to avoid re-hashing each key
    /// for both shard dispatch and hash-table lookup.
    read_handle<Value> peek_for_get_with_hash(const Key& key, std::size_t hash) {
        auto pin_fn = [](auto* item) {
            return item->refcount.incRef() == detail::IncResult::kIncOk;
        };
        auto ptr = map_.find_and_pin_lockfree_with_hash(key, hash, pin_fn);
        if (!ptr) return {};
        // P1-1: Inline TTL check. Fast path: expiry_ns == 0 (no TTL) skips
        // steady_clock::now() entirely. Only items with a TTL set pay the
        // clock read, and only on cache hits. Expired items are unpinned
        // (decRef) and reported as a miss; the actual eviction is handled
        // lazily by evict_expired() / the background TTL cleaner, NOT here,
        // so we don't need a write lock on the hot path.
        if (ptr->expiry_ns != 0) {
            // P1-10: Track TTL check frequency on the read path.
            stats_.ttl_checked_count.fetch_add(1, std::memory_order_relaxed);
            auto now_ns = static_cast<std::uint64_t>(
                std::chrono::steady_clock::now().time_since_epoch().count());
            if (now_ns >= ptr->expiry_ns) {
                stats_.ttl_expired_count.fetch_add(1, std::memory_order_relaxed);
                ptr->refcount.decRef();
                stats_.register_miss();
                callbacks_.collect_miss(key);
                return {};
            }
        }
        return read_handle<Value>{&ptr->value, &ptr->refcount, pre_pinned};
    }

    bool del(const Key& key) {
        cleanup_pending_deletion();
        auto ptr = map_.find(key);
        if (!ptr) return false;
        if (ptr->has_active_handle()) return false;
        erase_impl(key);
        return true;
    }

    /// Force delete a key even if it has active handles.
    /// The item is immediately removed from the map and list,
    /// but memory is not freed until all handles are released.
    /// Returns true if the key was found (and removed from the map).
    bool force_del(const Key& key) {
        cleanup_pending_deletion();
        auto ptr = map_.find(key);
        if (!ptr) return false;
        auto* item = ptr;
        size_type mem = calc_item_memory(item->key, item->value);

        map_.erase(key);
        // Capture item state BEFORE remove() poisons the item's memory.
        bool has_handle = item->has_active_handle();
        if (!has_handle) {
            callbacks_.collect_evict(item->key, std::move(item->value));
        }
        items_.remove(*item);

        stats_.current_memory.fetch_sub(mem);
        stats_.current_size.store(total_size());

        if (!has_handle) {
            detail::hazptr_domain::default_domain().retire(item);
        } else {
            pending_deletion_.push_back(item);
        }
        return true;
    }

    std::optional<Value> pop(const Key& key) {
        auto ptr = map_.find(key);
        if (!ptr) return std::nullopt;
        auto* item = ptr;
        if (item->has_active_handle()) return std::nullopt;
        size_type mem = calc_item_memory(item->key, item->value);
        Value value = std::move(item->value);
        stats_.current_memory.fetch_sub(mem);
        map_.erase(key);
        items_.remove(*item);
        detail::hazptr_domain::default_domain().retire(item);
        stats_.current_size.store(total_size());
        return value;
    }

    std::optional<std::pair<Key, Value>> pop_lru() {
        cleanup_pending_deletion();
        auto* tail = items_.tail();
        if (!tail || tail->has_active_handle()) return std::nullopt;
        if (!map_.contains(tail->key)) return std::nullopt;
        size_type mem = calc_item_memory(tail->key, tail->value);
        std::pair<Key, Value> result{std::move(tail->key), std::move(tail->value)};
        stats_.current_memory.fetch_sub(mem);
        map_.erase(result.first);
        items_.pop_tail();
        detail::hazptr_domain::default_domain().retire(tail);
        stats_.current_size.store(total_size());
        return result;
    }

    /// Flush the cache. Items pinned by an active read_handle are left in place.
    ///
    /// P1-8 (T2.6 bugfix): Two-pass deferred retirement — see mm_lru::flush()
    /// for the full rationale. Pass 1 collects items to retire (markForEviction,
    /// collect_evict, map_.erase, items_.remove, unmarkForEviction). Pass 2
    /// retires all collected items after iteration completes, preventing the
    /// background reclaimer from freeing items during list traversal.
    void flush() {
        std::vector<item_ptr> to_retire;
        auto* curr = items_.head();
        while (curr) {
            // Save next before any mutation. remove() only clears curr's hook
            // pointers; the next node remains valid and linked.
            auto* next = items_.get_next(*curr);
            if (!curr->has_active_handle()) {
                auto evict_result = curr->refcount.markForEviction();
                if (evict_result != detail::MarkForEvictionResult::kSuccess) {
                    curr = next;
                    continue;
                }
                size_type mem = calc_item_memory(curr->key, curr->value);
                stats_.current_memory.fetch_sub(mem);
                callbacks_.collect_evict(curr->key, std::move(curr->value));
                map_.erase(curr->key);
                items_.remove(*curr);
                curr->refcount.unmarkForEviction();
                to_retire.push_back(curr);
            }
            curr = next;
        }
        // P1-8 (T2.6 bugfix, phase 3): Refresh hash stats BEFORE retiring
        // any items — see mm_lru::flush() for the full rationale. The
        // background `periodic_worker` could free retired items while
        // `max_chain_length()` traverses the hash chain → UAF. Refreshing
        // here (after `map_.erase()` in Pass 1 but before retirement in
        // Pass 2) keeps items alive during the traversal.
        refresh_hash_stats();
        // Pass 2: retire all collected items after iteration is complete.
        // mm_fifo currently has no use_ebr config; always use hazptr.
        for (auto* item : to_retire) {
            detail::hazptr_domain::default_domain().retire(item);
        }
        stats_.current_size.store(total_size());
        cleanup_pending_deletion();
    }

    bool contains(const Key& key) const {
        return map_.contains(key);
    }

    // --------------------------------------------------------------------
    // Iterators (MRU->LRU / head->tail)
    // --------------------------------------------------------------------

    iterator begin() noexcept { return items_.begin(); }
    iterator end() noexcept { return items_.end(); }
    const_iterator begin() const noexcept { return items_.begin(); }
    const_iterator end() const noexcept { return items_.end(); }

    /// Reverse iterators: traverse from LRU tail to MRU head.
    reverse_iterator rbegin() noexcept { return items_.rbegin(); }
    reverse_iterator rend() noexcept { return items_.rend(); }
    const_reverse_iterator rbegin() const noexcept { return items_.rbegin(); }
    const_reverse_iterator rend() const noexcept { return items_.rend(); }

    // --------------------------------------------------------------------
    // Capacity
    // --------------------------------------------------------------------

    bool empty() const noexcept { return map_.empty(); }
    size_type size() const noexcept { return map_.size(); }
    size_type max_size() const noexcept { return max_size_; }
    size_type max_memory() const noexcept { return max_memory_; }
    size_type current_memory() const noexcept { return stats_.current_memory.load(); }

    /// Estimate memory that would be accounted for an item with the given key
    /// and value, including the fixed item overhead and any custom size
    /// calculators registered via set_key/value_size_calculator().
    size_type estimate_item_memory(const Key& key, const Value& value) const {
        return calc_item_memory(key, value);
    }

    /// Check if any item (in the list or pending deletion) has an active handle.
    bool has_active_handles() const noexcept {
        for (auto it = items_.begin(); it != items_.end(); ++it) {
            if (it->has_active_handle()) return true;
        }
        for (auto* item : pending_deletion_) {
            if (item->has_active_handle()) return true;
        }
        return false;
    }

    void max_size(size_type new_max) {
        max_size_ = new_max;
        stats_.max_size.store(new_max);
        if (new_max != npos) shrink_to_fit();
    }

    void max_memory(size_type new_max) {
        max_memory_ = new_max;
        stats_.max_memory.store(new_max);
        if (new_max != npos) shrink_to_fit();
    }

    void shrink_to_fit() {
        while (should_evict()) {
            auto old_size = size();
            evict();
            if (size() == old_size) break;
        }
    }

    // --------------------------------------------------------------------
    // Statistics and callbacks
    // --------------------------------------------------------------------

    stats_type& stats() noexcept { return stats_; }
    const stats_type& stats() const noexcept { return stats_; }
    callback_mgr& callbacks() noexcept { return callbacks_; }
    const callback_mgr& callbacks() const noexcept { return callbacks_; }

    // P1-7: Number of items in pending-deletion state (removed from cache
    // but still pinned by active read_handles). Best-effort count — may
    // race with concurrent writes. For monitoring only.
    std::size_t pending_deletion_count() const noexcept {
        return pending_deletion_.size();
    }

    /// Refresh hash table diagnostic stats (load factor, max chain length).
    /// O(bucket_count) scan — call periodically, not on every operation.
    void refresh_hash_stats() const noexcept {
        stats_.hash_load_factor.store(map_.load_factor(), std::memory_order_relaxed);
        stats_.max_chain_length.store(map_.max_chain_length(), std::memory_order_relaxed);
        // P1-1: Refresh rehash diagnostics from the hash table.
        stats_.rehash_count.store(map_.rehash_count(), std::memory_order_relaxed);
        stats_.rehash_total_time_ns.store(map_.rehash_total_time_ns(), std::memory_order_relaxed);
        stats_.rehash_migrated_items.store(map_.rehash_migrated_items(), std::memory_order_relaxed);
        // T13.1: Refresh overload threshold and event counter from the
        // hash table. These mirror the live state in concurrent_hash_table.
        stats_.hash_overload_threshold.store(map_.hash_overload_threshold(), std::memory_order_relaxed);
        stats_.hash_overload_events.store(map_.hash_overload_events(), std::memory_order_relaxed);
    }

    /// T-B4 (P2-10): Forward to the underlying hash table's diagnostics
    /// cache refresh. Only segmented_concurrent_hash_table implements
    /// this (regular concurrent_hash_table doesn't cache — its
    /// `max_chain_length()` is already a single-table scan, cheap enough
    /// to not warrant caching). For non-segmented tables this is a no-op
    /// via `if constexpr` (zero-cost). The background rehash balancer
    /// invokes this unconditionally.
    void refresh_diagnostics_cache() const noexcept {
        if constexpr (requires { map_.refresh_diagnostics_cache(); }) {
            map_.refresh_diagnostics_cache();
        }
    }

    /// T-B4 (P2-10): Forward to the underlying hash table's age metric.
    /// Returns `std::numeric_limits<std::uint64_t>::max()` if the cache
    /// has never been refreshed or the underlying table doesn't cache.
    /// Operators should check the `segmented_hash_table` flag in
    /// diagnostics() before relying on this value — non-segmented tables
    /// always report max (no cache, so age is meaningless).
    std::uint64_t diagnostics_cache_age_ms() const noexcept {
        if constexpr (requires { map_.diagnostics_cache_age_ms(); }) {
            return map_.diagnostics_cache_age_ms();
        }
        return std::numeric_limits<std::uint64_t>::max();
    }

    // --------------------------------------------------------------------
    // Config (stubs for unified_cache compatibility)
    // --------------------------------------------------------------------

    mm_fifo_config config() const noexcept { return config_; }
    void set_config(const mm_fifo_config& cfg) {
        config_ = cfg;
        map_.set_alloc_fns(cfg.alloc_fn, cfg.dealloc_fn);
    }

    /// Pre-allocate hash table buckets for `expected_items` entries.
    void reserve(size_type expected_items) {
        map_.reserve(expected_items);
    }

    /// Enable/disable incremental rehash for the hash table.
    /// When enabled, rehash migrates buckets incrementally across multiple
    /// operations instead of blocking all writers during a single rehash.
    /// This reduces write-path latency spikes under load.
    void set_incremental_rehash(bool enabled) {
        map_.set_incremental_rehash(enabled);
    }

    /// Query whether incremental rehash is enabled.
    bool incremental_rehash_enabled() const noexcept {
        return map_.incremental_rehash_enabled();
    }

    /// P0-5 (T1.3): Advance any in-progress incremental rehash by one
    /// per-call migration budget (kRehashFinishMaxBucketsPerCall).
    /// Called by the background rehash balancer to ensure stalled
    /// rehashes eventually complete without requiring writes to the
    /// affected hash table. No-op when no rehash is in progress.
    void advance_incremental_rehash() noexcept {
        map_.rehash_finish();
    }

    /// T11.5: String-based strategy setter (see concurrent_hash_table::set_rehash_strategy).
    bool set_rehash_strategy(std::string_view strategy) noexcept {
        return map_.set_rehash_strategy(strategy);
    }
    std::string_view rehash_strategy() const noexcept {
        return map_.rehash_strategy();
    }

    /// T11.3: Number of writes blocked by a non-incremental (blocking) rehash.
    std::size_t rehash_blocked_writes_count() const noexcept {
        return map_.rehash_blocked_writes_count();
    }

    /// P1-5: Number of times find_and_pin_lockfree fell back to the
    /// lock-protected path because the target segment was in incremental
    /// rehash. Non-zero values indicate the lock-free read path is being
    /// degraded by rehash activity.
    std::size_t rehash_lockfree_fallback_count() const noexcept {
        return map_.rehash_lockfree_fallback_count();
    }

    /// P0-D: Ratio of the hash table currently in an incremental rehash.
    /// For non-segmented tables: 0.0 or 1.0 (whole table rehashing or not).
    /// For segmented tables: fraction of segments currently rehashing.
    /// Exposed as a Prometheus gauge to detect sustained rehash pressure.
    float rehash_in_progress_ratio() const noexcept {
        return map_.rehash_in_progress_ratio();
    }

    /// T13.1: Set the hash table load factor overload threshold.
    /// See concurrent_hash_table::set_hash_overload_threshold.
    void set_hash_overload_threshold(float threshold) noexcept {
        map_.set_hash_overload_threshold(threshold);
    }

    float hash_overload_threshold() const noexcept {
        return map_.hash_overload_threshold();
    }

    std::size_t hash_overload_events() const noexcept {
        return map_.hash_overload_events();
    }

    /// T13.2: Register an overload callback on the underlying hash table.
    void set_overload_callback(std::function<void(float, float)> cb) {
        map_.set_overload_callback(std::move(cb));
    }

    /// P2-4 (T2.4): Toggle async mode for the overload callback.
    /// Forwarded to the underlying hash table. See
    /// `concurrent_hash_table::set_async_overload_callback` for semantics.
    void set_async_overload_callback(bool enabled) noexcept {
        map_.set_async_overload_callback(enabled);
    }

    /// P2-4 (T2.4): Drain pending overload events from the underlying
    /// hash table and dispatch the registered callback for each. Returns
    /// the number of events drained. Designed to be called from a
    /// background worker (e.g. the `event_drain_worker` in `unified_cache`).
    std::size_t drain_overload_callbacks() {
        return map_.drain_overload_callbacks();
    }

    /// Whether an incremental rehash is currently in progress.
    bool is_rehashing() const noexcept { return map_.is_rehashing(); }
    /// Buckets fully migrated so far during the in-progress rehash (0 if none).
    size_type rehash_progress() const noexcept { return map_.rehash_progress(); }
    /// New bucket count target for the in-progress rehash (0 if none).
    size_type rehash_new_bucket_count() const noexcept { return map_.rehash_new_bucket_count(); }
    /// Old bucket count for the in-progress rehash (0 if none).
    size_type rehash_old_bucket_count() const noexcept { return map_.rehash_old_bucket_count(); }
    /// Total number of hash table buckets currently allocated.
    size_type bucket_count() const noexcept { return map_.bucket_count(); }

    /// Set custom hash table node allocation/deallocation functions.
    void set_hash_alloc_fns(void* (*alloc_fn)(std::size_t), void (*dealloc_fn)(void*)) {
        config_.alloc_fn = alloc_fn;
        config_.dealloc_fn = dealloc_fn;
        map_.set_alloc_fns(alloc_fn, dealloc_fn);
    }
    uint32_t refresh_time() const noexcept { return 0; }

    /// FIFO does not promote on access; promote() is a no-op for API compatibility.
    bool promote(const Key& /*key*/) { return false; }

    // --------------------------------------------------------------------
    // Eviction (public for pooled_cache / unified_cache::evict())
    // --------------------------------------------------------------------

    void evict() {
        cleanup_pending_deletion();
        auto* victim = find_eviction_victim();
        if (victim) {
            auto evict_result = victim->refcount.markForEviction();
            if (evict_result != detail::MarkForEvictionResult::kSuccess) {
                stats_.current_size.store(total_size());
                return;
            }

            const auto& key = victim->key;
            size_type mem = calc_item_memory(key, victim->value);

            stats_.current_memory.fetch_sub(mem);
            stats_.register_eviction();
            if (callbacks_.has_eviction_callbacks()) {
                Value value = std::move(victim->value);
                callbacks_.collect_evict(key, std::move(value));
            }

            map_.erase(key);
            victim->refcount.unmarkInMMContainer();
            victim->refcount.unmarkForEviction();
            items_.remove(*victim);
            detail::hazptr_domain::default_domain().retire(victim);
        }
        stats_.current_size.store(total_size());
    }

    /// Find eviction victim from tail, respecting EvictionPredicate and active handles.
    item_ptr find_eviction_victim() {
        auto* curr = items_.tail();
        if (!curr) return curr;

        const bool has_pred = static_cast<bool>(eviction_predicate_);
        size_t tries = 0;
        while (curr) {
            stats_.eviction_search_steps.fetch_add(1, std::memory_order_relaxed);
            // H0: Skip nodes with active handles (not counted toward tries)
            if (curr->has_active_handle()) {
                stats_.pinned_skip_count.fetch_add(1, std::memory_order_relaxed);
                curr = items_.get_prev(*curr);
                continue;
            }
            // B15: EvictionPredicate veto (counted toward tries)
            if (has_pred && !eviction_predicate_(curr->key, curr->value)) {
                curr = items_.get_prev(*curr);
                if (++tries >= config_.eviction_search_tries) break;
                continue;
            }
            return curr;
        }
        return nullptr;
    }

    // --------------------------------------------------------------------
    // Memory policy
    // --------------------------------------------------------------------

    void set_key_size_calculator(std::function<size_type(const Key&)> func) {
        key_size_fn_ = std::move(func);
    }

    void set_value_size_calculator(std::function<size_type(const Value&)> func) {
        value_size_fn_ = std::move(func);
    }

    // --------------------------------------------------------------------
    // Eviction predicate (B15)
    // --------------------------------------------------------------------

    void set_eviction_predicate(std::function<bool(const Key&, const Value&)> pred) {
        eviction_predicate_ = std::move(pred);
    }

private:
    item_list items_;
    map_type map_;

    size_type max_size_ = unlimited;
    size_type max_memory_ = unlimited;
    mm_fifo_config config_;

    mutable stats_type stats_;
    mutable callback_mgr callbacks_;

    std::function<size_type(const Key&)> key_size_fn_;
    std::function<size_type(const Value&)> value_size_fn_;
    std::function<bool(const Key&, const Value&)> eviction_predicate_;

    // Items removed by force_del() that still have active handles.
    std::vector<item_ptr> pending_deletion_;

    /// Clean up pending-deletion items whose handles have all been released.
    /// Fires the deferred eviction callback (with value move) before deleting.
    void cleanup_pending_deletion() {
        auto it = pending_deletion_.begin();
        while (it != pending_deletion_.end()) {
            if (!(*it)->has_active_handle()) {
                auto* item = *it;
                callbacks_.collect_evict(item->key, std::move(item->value));
                // P1-5: Route through hazptr retire instead of raw delete —
                // a concurrent hazptr-protected reader may still hold a
                // hazard pointer to this item even with refcount=0.
                detail::hazptr_domain::default_domain().retire(item);
                it = pending_deletion_.erase(it);
            } else {
                ++it;
            }
        }
    }

    // --------------------------------------------------------------------
    // Internal helpers
    // --------------------------------------------------------------------

    bool should_evict() const {
        if (max_size_ != unlimited && size() > max_size_) return true;
        if (max_memory_ != unlimited && current_memory() > max_memory_) return true;
        return false;
    }

    bool is_at_capacity() const {
        return max_size_ != unlimited && size() >= max_size_;
    }

    size_type calc_item_memory(const Key& key, const Value& value) const {
        size_type mem = item_overhead;
        if (key_size_fn_) mem += key_size_fn_(key) * 2;
        if (value_size_fn_) mem += value_size_fn_(value);
        return mem;
    }

    size_type total_size() const noexcept { return map_.size(); }

    // --------------------------------------------------------------------
    // Insert / Update / Evict
    // --------------------------------------------------------------------

    template <typename V>
    void insert_new(const Key& key, V&& value) {
        if (max_size_ == 0 || max_memory_ == 0) return;
        cleanup_pending_deletion();
        if (is_at_capacity()) evict();
        while (should_evict()) {
            auto old_size = size();
            evict();
            if (size() == old_size) break;
        }

        auto mem = calc_item_memory(key, value);
        if (max_memory_ != unlimited) {
            while (!items_.empty() &&
                   stats_.current_memory.load() + mem > max_memory_) {
                auto old_size = size();
                evict();
                if (size() == old_size) break;
            }
        }

        // Allocate new item and link at head (MRU)
        auto* item = this->allocate_item(key, std::forward<V>(value));
        items_.link_at_head(*item);
        item->refcount.markInMMContainer();
        map_.insert(key, item);

        stats_.current_size.store(total_size());
        stats_.current_memory.fetch_add(mem);
        stats_.register_insertion();
        callbacks_.collect_insert(key, item->value);
    }

    template <typename V>
    void update_existing(item_ptr item, V&& value) {
        size_type old_mem = calc_item_memory(item->key, item->value);
        // P-MED-2 (T-H4): Strong exception guarantee via copy-then-swap.
        if constexpr (std::is_nothrow_swappable_v<Value> &&
                      std::is_constructible_v<Value, V>) {
            Value tmp(std::forward<V>(value));
            using std::swap;
            swap(item->value, tmp);
        } else {
            item->value = std::forward<V>(value);
        }
        // FIFO: do NOT move to head on update
        size_type new_mem = calc_item_memory(item->key, item->value);
        if (new_mem > old_mem) {
            stats_.current_memory.fetch_add(new_mem - old_mem);
        } else if (new_mem < old_mem) {
            stats_.current_memory.fetch_sub(old_mem - new_mem);
        }
        // O7: Fire on_update for value changes on existing keys (distinct
        // from on_insert, which fires only for new key insertions).
        callbacks_.collect_update(item->key, item->value);
        if (should_evict()) {
            shrink_to_fit();
        }
    }

    void erase_impl(const Key& key) {
        auto ptr = map_.find(key);
        if (!ptr) return;
        auto* item = ptr;
        size_type mem = calc_item_memory(item->key, item->value);

        stats_.current_memory.fetch_sub(mem);
        callbacks_.collect_evict(item->key, std::move(item->value));

        map_.erase(key);
        item->refcount.unmarkInMMContainer();
        items_.remove(*item);
        detail::hazptr_domain::default_domain().retire(item);
        stats_.current_size.store(total_size());
    }

    friend bool operator==(const mm_fifo& a, const mm_fifo& b) {
        if (a.size() != b.size()) return false;
        auto ai = a.begin(), bi = b.begin();
        for (; ai != a.end(); ++ai, ++bi) {
            if (ai->key != bi->key || ai->value != bi->value) return false;
        }
        return true;
    }

    friend bool operator!=(const mm_fifo& a, const mm_fifo& b) { return !(a == b); }

    // --------------------------------------------------------------------
    // Stream output
    // --------------------------------------------------------------------

    friend std::ostream& operator<<(std::ostream& os, const mm_fifo& c) {
        os << "mm_fifo @" << &c << "  " << c.stats_ << "\n";
        std::size_t idx = 0;
        for (auto it = c.begin(); it != c.end(); ++it, ++idx) {
            os << "  [" << idx << "] key='";
            if constexpr (lru::detail::is_formattable_v<Key>) {
                os << std::format("{}", it->key);
            } else {
                os << std::format("<key at {:#x}>", reinterpret_cast<std::uintptr_t>(&it->key));
            }
            os << "' value='";
            if constexpr (lru::detail::is_formattable_v<Value>) {
                os << std::format("{}", it->value);
            } else {
                os << std::format("<val at {:#x}>", reinterpret_cast<std::uintptr_t>(&it->value));
            }
            os << "'\n";
        }
        return os;
    }
};


// ============================================================================
// 2Q Strategy - mm_2q
// ============================================================================


// ============================================================================
// 2Q Strategy Configuration
// ============================================================================

struct mm_2q_config {
    /// Percentage of cache for Hot queue (default 30%)
    double hot_ratio = 0.3;
    /// Percentage of cache for Warm queue (default 40%)
    /// Remaining goes to Cold queue (default 30%)
    double warm_ratio = 0.4;
    /// Default time between promotions for the same item (seconds).
    /// 0 = no delay (traditional 2Q behavior).
    uint32_t default_lru_refresh_time = 60;
    /// Whether to promote the item on write access.
    bool update_on_write = false;
    /// Whether to promote the item on read access.
    bool update_on_read = true;
    /// 是否在 record_access 的 Cold→Warm 晋升后触发 rebalance。
    /// 对齐 CacheLib MM2Q.h:338 的 rebalanceOnRecordAccess。
    /// 读路径对延迟敏感时可关闭，交由 insert/evict 路径周期性修正 Hot/Warm 配额。
    bool rebalance_on_record_access = true;

    /// A5: 是否在 record_access 中使用 try_to_lock 跳过提升。
    /// 对齐 CacheLib MMLru.h:567-577 的 tryLockUpdate。
    /// 默认 true，因为 unified_cache 已提供外层并发保护，
    /// 阻塞等待 update_mutex_ 会造成双重锁开销。
    bool try_lock_update = true;

    /// Use combined lock for eviction iterators.
    bool use_combined_lock_for_iterators = false;

    /// B15: 淘汰搜索次数上限。
    size_t eviction_search_tries = 3;

    /// Ratio for adaptive refresh time adjustment.
    /// newRefreshTime = max(default, warm_tail_age * ratio), capped at k_lru_refresh_time_cap.
    double lru_refresh_ratio = 0.0;

    /// Interval in seconds for reconfiguring the adaptive refresh time.
    /// 0 = disabled (default).
    uint32_t mm_reconfigure_interval_secs = 0;

    /// Expected number of items for automatic bucket count sizing.
    /// 0 = use default bucket count (1024). When > 0, the internal hash
    /// table is pre-sized via concurrent_hash_table::buckets_for_items()
    /// to keep average chain length ≤ 0.25 at the expected load.
    size_t expected_items = 0;

    /// Custom node allocation function for non-EmbeddedChain hash table nodes.
    /// nullptr (default) = standard new/delete allocation.
    void* (*alloc_fn)(std::size_t) = nullptr;

    /// Custom node deallocation function (must pair with alloc_fn).
    void  (*dealloc_fn)(void*) = nullptr;

    // Max lruRefreshTime cap
    static constexpr uint32_t k_lru_refresh_time_cap = 900;

    // B4: 配置校验——各 ratio 范围合法
    mm_2q_config() noexcept = default;

    void validate() const {
        if (!(hot_ratio >= 0.0) || !(warm_ratio >= 0.0)) {
            throw std::invalid_argument(
                "mm_2q_config: hot_ratio and warm_ratio must be non-negative and finite");
        }
        if (!(lru_refresh_ratio >= 0.0)) {
            throw std::invalid_argument(
                "mm_2q_config: lru_refresh_ratio must be non-negative");
        }
        if (hot_ratio + warm_ratio > 1.0) {
            throw std::invalid_argument(
                "mm_2q_config: hot_ratio + warm_ratio must not exceed 1.0");
        }
    }
};

// ============================================================================
// Queue ID constants
// ============================================================================

namespace mm2q {
    static constexpr uint8_t kQueueHot  = 0;
    static constexpr uint8_t kQueueWarm = 1;
    static constexpr uint8_t kQueueCold = 2;
    static constexpr uint8_t kQueueCount = 3;
} // namespace mm2q

// ============================================================================
// 2Q Strategy
// ============================================================================

/// 2Q (Two Queues) eviction strategy.
/// Divides the cache into Hot, Warm, and Cold queues:
/// - New items enter Hot queue (head)
/// - When Hot overflows, tail items move to Cold
/// - Cold items that get accessed are promoted to Warm (with delayed promotion)
/// - When Warm overflows, tail items move to Cold
/// - Cold items are evicted first
template <
    typename Key,
    typename Value,
    typename Hash = std::hash<Key>,
    typename KeyEqual = std::equal_to<Key>,
    typename ProbingStyle = detail::chain_probing_tag,
    bool Segmented = false
>
/// A4: 线程安全契约——此类非线程安全，调用方必须确保在外层 unified_cache 锁内访问。
/// 内部 update_mutex_ 仅用于 try_lock_update 路径的解耦优化，不保证 MM 层独立线程安全。
class mm_2q : public detail::mm_allocator_mixin<detail::cache_item<Key, Value>> {
public:
    using key_type = Key;
    using mapped_type = Value;
    using size_type = std::size_t;
    using config_type = mm_2q_config;

    // Item type: cache_item with embedded intrusive hook
    using item_type = detail::cache_item<Key, Value>;
    using item_ptr = item_type*;

    // Intrusive list type
    using item_list = detail::intrusive_list<item_type, detail::intrusive_hook, detail::default_get_hook<item_type>>;

    // Map: Key -> item pointer
    using map_type = std::conditional_t<
        Segmented,
        detail::segmented_concurrent_hash_table<Key, item_ptr, Hash, KeyEqual, true, ProbingStyle, 64>,
        detail::concurrent_hash_table<Key, item_ptr, Hash, KeyEqual, true, ProbingStyle>
    >;

    using callback_mgr = callback_manager<Key, Value>;
    using stats_type = cache_stats;

    static constexpr size_type npos = unlimited;
    static constexpr size_type item_overhead = sizeof(item_type) + map_type::entry_overhead;

    // --------------------------------------------------------------------
    // Constructors / Destructor
    // --------------------------------------------------------------------

    mm_2q() : mm_2q(mm_2q_config{}) {}

    explicit mm_2q(const mm_2q_config& config)
        : config_(config)
        , map_(config.expected_items > 0
            ? map_type::buckets_for_items(config.expected_items)
            : 1024,
            config.alloc_fn, config.dealloc_fn)
        , lru_refresh_time_(config.default_lru_refresh_time)
        , next_reconfigure_time_(config.mm_reconfigure_interval_secs == 0
            ? std::numeric_limits<uint32_t>::max()
            : current_time_sec() + config.mm_reconfigure_interval_secs) {}

    mm_2q(size_type max_size, const mm_2q_config& config = mm_2q_config{})
        : mm_2q(config) {
        detail::validate_capacity(max_size, unlimited);
        max_size_ = max_size;
        stats_.max_size.store(max_size);
    }

    mm_2q(size_type max_size, size_type max_memory,
            const mm_2q_config& config = mm_2q_config{})
        : mm_2q(config) {
        detail::validate_capacity(max_size, max_memory);
        max_size_ = max_size;
        max_memory_ = max_memory;
        stats_.max_size.store(max_size);
        stats_.max_memory.store(max_memory);
    }

    ~mm_2q() {
        // Clean up pending deletions first (items with active handles that were force_del'd)
        for (auto* item : pending_deletion_) {
            delete item;
        }
        pending_deletion_.clear();
        // Use flush() to properly erase items from both map and list before deletion,
        // preventing use-after-free when the hash table destructor iterates its chains.
        flush();
    }

    // Non-copyable, non-movable (owns item pointers)
    mm_2q(const mm_2q&) = delete;
    mm_2q& operator=(const mm_2q&) = delete;

    // --------------------------------------------------------------------
    // Core cache API
    // --------------------------------------------------------------------

    template <typename V>
    void set(const Key& key, V&& value) {
        cleanup_pending_deletion();
        auto ptr = map_.find(key);
        if (!ptr) {
            insert_new(key, std::forward<V>(value));
        } else {
            update_existing(ptr, std::forward<V>(value), access_mode::write);
        }
    }

    template <typename V>
    bool add(const Key& key, V&& value) {
        cleanup_pending_deletion();
        auto ptr = map_.find(key);
        if (!ptr) {
            insert_new(key, std::forward<V>(value));
            return true;
        }
        record_access(ptr, access_mode::read);
        return false;
    }

    template <typename V>
    bool replace(const Key& key, V&& value) {
        cleanup_pending_deletion();
        auto ptr = map_.find(key);
        if (!ptr) return false;
        update_existing(ptr, std::forward<V>(value), access_mode::write);
        return true;
    }

    read_handle<Value> get(const Key& key) {
        auto ptr = map_.find(key);
        if (!ptr) {
            stats_.register_miss();
            callbacks_.collect_miss(key);
            return {};
        }
        auto* item = ptr;
        record_access(item, access_mode::read);
        stats_.register_hit();
        callbacks_.collect_hit(key, item->value);
        return read_handle<Value>{&item->value, &item->refcount};
    }

    read_handle<const Value> get(const Key& key) const {
        auto ptr = map_.find(key);
        if (!ptr) {
            stats_.register_miss();
            callbacks_.collect_miss(key);
            return {};
        }
        auto* item = ptr;
        stats_.register_hit();
        callbacks_.collect_hit(key, item->value);
        return read_handle<const Value>{&item->value, &item->refcount};
    }

    /// H0: Peek with handle — 不提升 LRU，返回 handle 防止持有期被淘汰。
    /// Uses find_and_pin_lockfree() to attempt lock-free pinning first
    /// (optimistic read + incRef without bucket lock), falling back to
    /// find_and_pin() (shared lock path) if the lock-free pin fails.
    /// No stripe-level read lock is needed.
    read_handle<const Value> peek(const Key& key) const {
        auto pin_fn = [](auto* item) {
            return item->refcount.incRef() == detail::IncResult::kIncOk;
        };
        auto ptr = map_.find_and_pin_lockfree(key, pin_fn);
        if (!ptr) return {};
        return read_handle<const Value>{&ptr->value, &ptr->refcount, pre_pinned};
    }

    /// Internal: peek that returns mutable handle (for optimistic get path).
    read_handle<Value> peek_for_get(const Key& key) {
        return peek_for_get_with_hash(key, Hash{}(key));
    }

    /// T16.4: peek_for_get with a pre-computed hash. The hash MUST be
    /// the result of Hash{}(key) — callers are responsible for hash
    /// compatibility. Used by bulk_get to avoid re-hashing each key
    /// for both shard dispatch and hash-table lookup.
    read_handle<Value> peek_for_get_with_hash(const Key& key, std::size_t hash) {
        auto pin_fn = [](auto* item) {
            return item->refcount.incRef() == detail::IncResult::kIncOk;
        };
        auto ptr = map_.find_and_pin_lockfree_with_hash(key, hash, pin_fn);
        if (!ptr) return {};
        // P1-1: Inline TTL check. Fast path: expiry_ns == 0 (no TTL) skips
        // steady_clock::now() entirely. Only items with a TTL set pay the
        // clock read, and only on cache hits. Expired items are unpinned
        // (decRef) and reported as a miss; the actual eviction is handled
        // lazily by evict_expired() / the background TTL cleaner, NOT here,
        // so we don't need a write lock on the hot path.
        if (ptr->expiry_ns != 0) {
            // P1-10: Track TTL check frequency on the read path.
            stats_.ttl_checked_count.fetch_add(1, std::memory_order_relaxed);
            auto now_ns = static_cast<std::uint64_t>(
                std::chrono::steady_clock::now().time_since_epoch().count());
            if (now_ns >= ptr->expiry_ns) {
                stats_.ttl_expired_count.fetch_add(1, std::memory_order_relaxed);
                ptr->refcount.decRef();
                stats_.register_miss();
                callbacks_.collect_miss(key);
                return {};
            }
        }
        return read_handle<Value>{&ptr->value, &ptr->refcount, pre_pinned};
    }

    bool del(const Key& key) {
        cleanup_pending_deletion();
        auto ptr = map_.find(key);
        if (!ptr) return false;
        if (ptr->has_active_handle()) return false;
        erase_impl(key);
        return true;
    }

    /// Force delete a key even if it has active handles.
    /// The item is immediately removed from the map and queue,
    /// but memory is not freed until all handles are released.
    /// Returns true if the key was found (and removed from the map).
    bool force_del(const Key& key) {
        cleanup_pending_deletion();
        auto ptr = map_.find(key);
        if (!ptr) return false;
        auto* item = ptr;
        size_type mem = calc_item_memory(item->key, item->value);

        map_.erase(key);
        // Capture item state BEFORE remove_from_queue() poisons the item's memory.
        bool has_handle = item->has_active_handle();
        if (!has_handle) {
            callbacks_.collect_evict(item->key, std::move(item->value));
        }
        remove_from_queue(item);

        stats_.current_memory.fetch_sub(mem);
        stats_.current_size.store(total_size());

        if (!has_handle) {
            detail::hazptr_domain::default_domain().retire(item);
        } else {
            pending_deletion_.push_back(item);
        }
        return true;
    }

    std::optional<Value> pop(const Key& key) {
        auto ptr = map_.find(key);
        if (!ptr) return std::nullopt;
        auto* item = ptr;
        if (item->has_active_handle()) return std::nullopt;
        size_type mem = calc_item_memory(item->key, item->value);
        Value value = std::move(item->value);
        stats_.current_memory.fetch_sub(mem);
        remove_from_queue(item);
        map_.erase(key);
        detail::hazptr_domain::default_domain().retire(item);
        stats_.current_size.store(total_size());
        return value;
    }

    /// Flush the cache. Items pinned by an active read_handle are left in place.
    ///
    /// P1-8 (T2.6 bugfix): Two-pass deferred retirement — see mm_lru::flush()
    /// for the full rationale. Bypasses erase_impl() to apply markForEviction()
    /// directly on the iterated item. All retirements are deferred to Pass 2
    /// after all queues have been traversed.
    void flush() {
        // Pass 1: iterate all queues, collect items to retire.
        std::vector<item_ptr> to_retire;
        for (uint8_t q = 0; q < mm2q::kQueueCount; ++q) {
            auto* curr = queues_[q].head();
            while (curr) {
                // Save next before any mutation. remove() only clears curr's hook
                // pointers; the next node remains valid and linked.
                auto* next = queues_[q].get_next(*curr);
                if (!curr->has_active_handle()) {
                    auto evict_result = curr->refcount.markForEviction();
                    if (evict_result != detail::MarkForEvictionResult::kSuccess) {
                        curr = next;
                        continue;
                    }
                    size_type mem = calc_item_memory(curr->key, curr->value);
                    stats_.current_memory.fetch_sub(mem);
                    callbacks_.collect_evict(curr->key, std::move(curr->value));
                    // Item may or may not be in the map (transient queue state).
                    map_.erase(curr->key);
                    queues_[q].remove(*curr);
                    curr->refcount.unmarkForEviction();
                    to_retire.push_back(curr);
                }
                curr = next;
            }
        }
        // P1-8 (T2.6 bugfix, phase 3): Refresh hash stats BEFORE retiring
        // any items — see mm_lru::flush() for the full rationale.
        refresh_hash_stats();
        // Pass 2: retire all collected items after iteration is complete.
        // mm_2q currently has no use_ebr config; always use hazptr.
        for (auto* item : to_retire) {
            detail::hazptr_domain::default_domain().retire(item);
        }
        stats_.current_size.store(total_size());
        cleanup_pending_deletion();
    }

    bool contains(const Key& key) const {
        return map_.contains(key);
    }

    // --------------------------------------------------------------------
    // Capacity
    // --------------------------------------------------------------------

    bool empty() const noexcept { return map_.empty(); }
    size_type size() const noexcept { return map_.size(); }
    size_type max_size() const noexcept { return max_size_; }
    size_type max_memory() const noexcept { return max_memory_; }
    size_type current_memory() const noexcept { return stats_.current_memory.load(); }

    /// Estimate memory that would be accounted for an item with the given key
    /// and value, including the fixed item overhead and any custom size
    /// calculators registered via set_key/value_size_calculator().
    size_type estimate_item_memory(const Key& key, const Value& value) const {
        return calc_item_memory(key, value);
    }

    /// Check if any item (in any queue or pending deletion) has an active handle.
    bool has_active_handles() const noexcept {
        for (std::size_t q = 0; q < mm2q::kQueueCount; ++q) {
            for (auto it = queues_[q].begin(); it != queues_[q].end(); ++it) {
                if (it->has_active_handle()) return true;
            }
        }
        for (auto* item : pending_deletion_) {
            if (item->has_active_handle()) return true;
        }
        return false;
    }

    void max_size(size_type new_max) {
        max_size_ = new_max;
        stats_.max_size.store(new_max);
        if (new_max != npos) {
            while (should_evict()) {
                auto old_size = size();
                evict();
                if (size() == old_size) break;
            }
        }
    }

    void max_memory(size_type new_max) {
        max_memory_ = new_max;
        stats_.max_memory.store(new_max);
        if (new_max != npos) {
            while (should_evict()) {
                auto old_size = size();
                evict();
                if (size() == old_size) break;
            }
        }
    }

    // --------------------------------------------------------------------
    // Statistics and callbacks
    // --------------------------------------------------------------------

    stats_type& stats() noexcept { return stats_; }
    const stats_type& stats() const noexcept { return stats_; }
    callback_mgr& callbacks() noexcept { return callbacks_; }
    const callback_mgr& callbacks() const noexcept { return callbacks_; }

    // P1-7: Number of items in pending-deletion state (removed from cache
    // but still pinned by active read_handles). Best-effort count — may
    // race with concurrent writes. For monitoring only.
    std::size_t pending_deletion_count() const noexcept {
        return pending_deletion_.size();
    }

    /// Refresh hash table diagnostic stats (load factor, max chain length).
    /// O(bucket_count) scan — call periodically, not on every operation.
    void refresh_hash_stats() const noexcept {
        stats_.hash_load_factor.store(map_.load_factor(), std::memory_order_relaxed);
        stats_.max_chain_length.store(map_.max_chain_length(), std::memory_order_relaxed);
        // P1-1: Refresh rehash diagnostics from the hash table.
        stats_.rehash_count.store(map_.rehash_count(), std::memory_order_relaxed);
        stats_.rehash_total_time_ns.store(map_.rehash_total_time_ns(), std::memory_order_relaxed);
        stats_.rehash_migrated_items.store(map_.rehash_migrated_items(), std::memory_order_relaxed);
        // T13.1: Refresh overload threshold and event counter from the
        // hash table. These mirror the live state in concurrent_hash_table.
        stats_.hash_overload_threshold.store(map_.hash_overload_threshold(), std::memory_order_relaxed);
        stats_.hash_overload_events.store(map_.hash_overload_events(), std::memory_order_relaxed);
    }

    /// T-B4 (P2-10): Forward to the underlying hash table's diagnostics
    /// cache refresh. Only segmented_concurrent_hash_table implements
    /// this (regular concurrent_hash_table doesn't cache — its
    /// `max_chain_length()` is already a single-table scan, cheap enough
    /// to not warrant caching). For non-segmented tables this is a no-op
    /// via `if constexpr` (zero-cost). The background rehash balancer
    /// invokes this unconditionally.
    void refresh_diagnostics_cache() const noexcept {
        if constexpr (requires { map_.refresh_diagnostics_cache(); }) {
            map_.refresh_diagnostics_cache();
        }
    }

    /// T-B4 (P2-10): Forward to the underlying hash table's age metric.
    /// Returns `std::numeric_limits<std::uint64_t>::max()` if the cache
    /// has never been refreshed or the underlying table doesn't cache.
    /// Operators should check the `segmented_hash_table` flag in
    /// diagnostics() before relying on this value — non-segmented tables
    /// always report max (no cache, so age is meaningless).
    std::uint64_t diagnostics_cache_age_ms() const noexcept {
        if constexpr (requires { map_.diagnostics_cache_age_ms(); }) {
            return map_.diagnostics_cache_age_ms();
        }
        return std::numeric_limits<std::uint64_t>::max();
    }

    // --------------------------------------------------------------------
    // Per-queue access statistics（对齐 CacheLib MM2Q.h:1096-1104）
    // --------------------------------------------------------------------

    /// Per-queue 访问明细，反映 Hot/Warm/Cold 的命中分布。
    struct per_queue_stats {
        std::size_t num_hot_accesses;
        std::size_t num_warm_accesses;
        std::size_t num_cold_accesses;
    };

    /// 返回 per-queue 访问统计快照（不影响原有 stats() 接口）。
    per_queue_stats get_per_queue_stats() const noexcept {
        return per_queue_stats{
            num_hot_accesses_.load(std::memory_order_relaxed),
            num_warm_accesses_.load(std::memory_order_relaxed),
            num_cold_accesses_.load(std::memory_order_relaxed),
        };
    }

    void set_key_size_calculator(std::function<size_type(const Key&)> func) {
        key_size_fn_ = std::move(func);
    }
    void set_value_size_calculator(std::function<size_type(const Value&)> func) {
        value_size_fn_ = std::move(func);
    }

    // --------------------------------------------------------------------
    // Queue size queries
    // --------------------------------------------------------------------

    size_type hot_size() const noexcept { return queues_[mm2q::kQueueHot].size(); }
    size_type warm_size() const noexcept { return queues_[mm2q::kQueueWarm].size(); }
    size_type cold_size() const noexcept { return queues_[mm2q::kQueueCold].size(); }

    /// B8: Eviction age statistics for mm_2q (对齐 CacheLib MM2Q.h:774-821).
    /// Returns per-queue and overall statistics.
    struct eviction_age_stat_2q {
        uint32_t hot_oldest_age = 0;
        uint32_t warm_oldest_age = 0;
        uint32_t cold_oldest_age = 0;
        size_type hot_size = 0;
        size_type warm_size = 0;
        size_type cold_size = 0;
    };

    eviction_age_stat_2q get_eviction_age_stat(std::size_t projected_length = 0) const noexcept {
        eviction_age_stat_2q stat;
        const auto curr_time = current_time_sec();
        stat.hot_size = queues_[mm2q::kQueueHot].size();
        stat.warm_size = queues_[mm2q::kQueueWarm].size();
        stat.cold_size = queues_[mm2q::kQueueCold].size();

        // Hot queue tail age
        auto* hot_tail = queues_[mm2q::kQueueHot].tail();
        stat.hot_oldest_age = hot_tail ? (curr_time - hot_tail->hook.update_time) : 0;

        // Warm queue tail age
        auto* warm_tail = queues_[mm2q::kQueueWarm].tail();
        stat.warm_oldest_age = warm_tail ? (curr_time - warm_tail->hook.update_time) : 0;

        // Cold queue tail age (primary eviction target)
        auto* cold_tail = queues_[mm2q::kQueueCold].tail();
        stat.cold_oldest_age = cold_tail ? (curr_time - cold_tail->hook.update_time) : 0;

        return stat;
    }

    // --------------------------------------------------------------------
    // 2Q-specific API
    // --------------------------------------------------------------------

    const mm_2q_config& config() const noexcept { return config_; }

    /// Set custom hash table node allocation/deallocation functions.
    void set_hash_alloc_fns(void* (*alloc_fn)(std::size_t), void (*dealloc_fn)(void*)) {
        config_.alloc_fn = alloc_fn;
        config_.dealloc_fn = dealloc_fn;
        map_.set_alloc_fns(alloc_fn, dealloc_fn);
    }

    void set_config(const mm_2q_config& config) {
        config_ = config;
        map_.set_alloc_fns(config.alloc_fn, config.dealloc_fn);
        lru_refresh_time_ = config.default_lru_refresh_time;
        next_reconfigure_time_ = config.mm_reconfigure_interval_secs == 0
            ? std::numeric_limits<uint32_t>::max()
            : current_time_sec() + config.mm_reconfigure_interval_secs;
        rebalance();
    }

    /// Pre-allocate hash table buckets for `expected_items` entries.
    void reserve(size_type expected_items) {
        map_.reserve(expected_items);
    }

    /// Enable/disable incremental rehash for the hash table.
    /// When enabled, rehash migrates buckets incrementally across multiple
    /// operations instead of blocking all writers during a single rehash.
    /// This reduces write-path latency spikes under load.
    void set_incremental_rehash(bool enabled) {
        map_.set_incremental_rehash(enabled);
    }

    /// Query whether incremental rehash is enabled.
    bool incremental_rehash_enabled() const noexcept {
        return map_.incremental_rehash_enabled();
    }

    /// P0-5 (T1.3): Advance any in-progress incremental rehash by one
    /// per-call migration budget (kRehashFinishMaxBucketsPerCall).
    /// Called by the background rehash balancer to ensure stalled
    /// rehashes eventually complete without requiring writes to the
    /// affected hash table. No-op when no rehash is in progress.
    void advance_incremental_rehash() noexcept {
        map_.rehash_finish();
    }

    /// T11.5: String-based strategy setter (see concurrent_hash_table::set_rehash_strategy).
    bool set_rehash_strategy(std::string_view strategy) noexcept {
        return map_.set_rehash_strategy(strategy);
    }
    std::string_view rehash_strategy() const noexcept {
        return map_.rehash_strategy();
    }

    /// T11.3: Number of writes blocked by a non-incremental (blocking) rehash.
    std::size_t rehash_blocked_writes_count() const noexcept {
        return map_.rehash_blocked_writes_count();
    }

    /// P1-5: Number of times find_and_pin_lockfree fell back to the
    /// lock-protected path because the target segment was in incremental
    /// rehash. Non-zero values indicate the lock-free read path is being
    /// degraded by rehash activity.
    std::size_t rehash_lockfree_fallback_count() const noexcept {
        return map_.rehash_lockfree_fallback_count();
    }

    /// P0-D: Ratio of the hash table currently in an incremental rehash.
    /// For non-segmented tables: 0.0 or 1.0 (whole table rehashing or not).
    /// For segmented tables: fraction of segments currently rehashing.
    /// Exposed as a Prometheus gauge to detect sustained rehash pressure.
    float rehash_in_progress_ratio() const noexcept {
        return map_.rehash_in_progress_ratio();
    }

    /// T13.1: Set the hash table load factor overload threshold.
    /// See concurrent_hash_table::set_hash_overload_threshold.
    void set_hash_overload_threshold(float threshold) noexcept {
        map_.set_hash_overload_threshold(threshold);
    }

    float hash_overload_threshold() const noexcept {
        return map_.hash_overload_threshold();
    }

    std::size_t hash_overload_events() const noexcept {
        return map_.hash_overload_events();
    }

    /// T13.2: Register an overload callback on the underlying hash table.
    void set_overload_callback(std::function<void(float, float)> cb) {
        map_.set_overload_callback(std::move(cb));
    }

    /// P2-4 (T2.4): Toggle async mode for the overload callback.
    /// Forwarded to the underlying hash table. See
    /// `concurrent_hash_table::set_async_overload_callback` for semantics.
    void set_async_overload_callback(bool enabled) noexcept {
        map_.set_async_overload_callback(enabled);
    }

    /// P2-4 (T2.4): Drain pending overload events from the underlying
    /// hash table and dispatch the registered callback for each. Returns
    /// the number of events drained. Designed to be called from a
    /// background worker (e.g. the `event_drain_worker` in `unified_cache`).
    std::size_t drain_overload_callbacks() {
        return map_.drain_overload_callbacks();
    }

    /// Whether an incremental rehash is currently in progress.
    bool is_rehashing() const noexcept { return map_.is_rehashing(); }
    /// Buckets fully migrated so far during the in-progress rehash (0 if none).
    size_type rehash_progress() const noexcept { return map_.rehash_progress(); }
    /// New bucket count target for the in-progress rehash (0 if none).
    size_type rehash_new_bucket_count() const noexcept { return map_.rehash_new_bucket_count(); }
    /// Old bucket count for the in-progress rehash (0 if none).
    size_type rehash_old_bucket_count() const noexcept { return map_.rehash_old_bucket_count(); }
    /// Total number of hash table buckets currently allocated.
    size_type bucket_count() const noexcept { return map_.bucket_count(); }

    uint32_t refresh_time() const noexcept { return lru_refresh_time_; }

    /// A5: 返回 try_lock_update 配置，用于 record_access 内的 try_to_lock 优化
    bool try_lock_update_enabled() const noexcept { return config_.try_lock_update; }

    /// Promote an item by key without triggering hit statistics or callbacks.
    /// This is a side-effect-free alternative to get() for batch promotion
    /// (e.g., from TLS ring flush). Returns true if the key was found and
    /// the item was eligible for promotion.
    bool promote(const Key& key) {
        auto ptr = map_.find(key);
        if (!ptr) return false;
        return record_access(ptr, access_mode::read);
    }

    // --------------------------------------------------------------------
    // LockedIterator (B7)
    // --------------------------------------------------------------------

    /// B7: 持有 mm_2q 锁的迭代器。
    /// 使用 locked_iterator_guard 管理锁生命周期。
    class LockedIterator {
    public:
        LockedIterator(mm_2q& mm)
            : guard_(mm.update_mutex_.m, mm.iterator_active_), mm_(&mm) {
            // 初始化到第一个非空队列的 tail
            init_queue(0);
        }

        ~LockedIterator() = default;
        LockedIterator(const LockedIterator&) = delete;
        LockedIterator& operator=(const LockedIterator&) = delete;
        LockedIterator(LockedIterator&& other) noexcept
            : guard_(std::move(other.guard_)), mm_(other.mm_),
              qid_(other.qid_), curr_(other.curr_) {}

        void destroy() { guard_.destroy(); }

        void resetToBegin() {
            qid_ = 0;
            init_queue(0);
        }

        bool next() {
            // 尝试前进
            auto* next = mm_->queues_[qid_].get_prev(*curr_);
            if (next) {
                curr_ = next;
                return true;
            }
            // 当前队列到头了，尝试下一个非空队列
            for (uint8_t q = qid_ + 1; q < mm2q::kQueueCount; ++q) {
                if (mm_->queues_[q].size() > 0) {
                    qid_ = q;
                    curr_ = mm_->queues_[q].tail();
                    return true;
                }
            }
            return false;
        }

        auto& operator*() { return *curr_; }
        auto* operator->() { return curr_; }
        explicit operator bool() const { return curr_ != nullptr; }

    private:
        void init_queue(uint8_t start_qid) {
            qid_ = start_qid;
            curr_ = nullptr;
            for (uint8_t q = start_qid; q < mm2q::kQueueCount; ++q) {
                curr_ = mm_->queues_[q].tail();
                if (curr_) { qid_ = q; return; }
            }
        }

        detail::locked_iterator_guard<> guard_;
        mm_2q* mm_;
        item_ptr curr_ = nullptr;
        uint8_t qid_ = 0;
    };

    // --------------------------------------------------------------------
    // UnifiedIterator (B6) — 跨队列淘汰顺序遍历
    // --------------------------------------------------------------------

    /// B6: 按 Cold→Hot→Warm 淘汰优先级顺序遍历，自动跳过空队列。
    /// 对齐 CacheLib MultiDList.h:98-178 的跨队列统一迭代器设计。
    class UnifiedIterator {
    public:
        // Eviction order: Cold(2) → Hot(0) → Warm(1)
        static constexpr uint8_t kEvictionOrder[] = {
            mm2q::kQueueCold, mm2q::kQueueHot, mm2q::kQueueWarm
        };
        static constexpr uint8_t kEvictionOrderSize = 3;

        UnifiedIterator(mm_2q& mm) : mm_(&mm) {
            step_ = 0;
            init_from_step();
        }

        bool next() {
            auto* next = mm_->queues_[qid_].get_prev(*curr_);
            if (next) {
                curr_ = next;
                return true;
            }
            // Current queue exhausted, advance to next in eviction order
            for (++step_; step_ < kEvictionOrderSize; ++step_) {
                uint8_t q = kEvictionOrder[step_];
                if (mm_->queues_[q].size() > 0) {
                    qid_ = q;
                    curr_ = mm_->queues_[q].tail();
                    return true;
                }
            }
            return false;
        }

        auto& operator*() { return *curr_; }
        auto* operator->() { return curr_; }
        explicit operator bool() const { return curr_ != nullptr; }

        /// Current queue ID (useful for caller to know which queue the item is in)
        uint8_t queue_id() const { return qid_; }

    private:
        void init_from_step() {
            curr_ = nullptr;
            for (; step_ < kEvictionOrderSize; ++step_) {
                uint8_t q = kEvictionOrder[step_];
                curr_ = mm_->queues_[q].tail();
                if (curr_) { qid_ = q; return; }
            }
        }

        mm_2q* mm_;
        item_ptr curr_ = nullptr;
        uint8_t qid_ = 0;
        uint8_t step_ = 0;
    };

    // --------------------------------------------------------------------
    // Const iterator: traverses Hot→Warm→Cold, each queue MRU→LRU
    // --------------------------------------------------------------------

    class const_iterator {
    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = const item_type;
        using reference = const value_type&;
        using pointer = value_type*;
        using difference_type = std::ptrdiff_t;

        const_iterator() : mm_(nullptr), qid_(mm2q::kQueueCount) {}

        const_iterator(const mm_2q* mm, uint8_t qid, typename item_list::const_iterator it)
            : mm_(mm), qid_(qid), it_(it) {}

        reference operator*() const { return *it_; }
        pointer operator->() const { return &*it_; }

        const_iterator& operator++() {
            ++it_;
            advance_to_nonempty();
            return *this;
        }
        const_iterator operator++(int) { auto t = *this; ++*this; return t; }

        bool operator==(const const_iterator& o) const {
            if (mm_ == nullptr && o.mm_ == nullptr) return true;
            if (qid_ >= mm2q::kQueueCount && o.qid_ >= mm2q::kQueueCount) return true;
            return qid_ == o.qid_ && it_ == o.it_;
        }
        bool operator!=(const const_iterator& o) const { return !(*this == o); }

    private:
        void advance_to_nonempty() {
            while (qid_ < mm2q::kQueueCount && it_ == mm_->queues_[qid_].end()) {
                ++qid_;
                if (qid_ < mm2q::kQueueCount) it_ = mm_->queues_[qid_].begin();
            }
        }
        const mm_2q* mm_;
        uint8_t qid_;
        typename item_list::const_iterator it_;
    };

    const_iterator begin() const {
        for (uint8_t q = 0; q < mm2q::kQueueCount; ++q) {
            if (!queues_[q].empty()) {
                return const_iterator(this, q, queues_[q].begin());
            }
        }
        return end();
    }
    const_iterator end() const {
        return const_iterator();
    }

    // --------------------------------------------------------------------
    // Equality comparison (content and order must match)
    // --------------------------------------------------------------------

    friend bool operator==(const mm_2q& a, const mm_2q& b) {
        if (a.size() != b.size()) return false;
        auto ai = a.begin(), bi = b.begin();
        for (; ai != a.end(); ++ai, ++bi) {
            if (ai->key != bi->key || ai->value != bi->value) return false;
        }
        return true;
    }
    friend bool operator!=(const mm_2q& a, const mm_2q& b) { return !(a == b); }

    // --------------------------------------------------------------------
    // Stream output
    // --------------------------------------------------------------------

    friend std::ostream& operator<<(std::ostream& os, const mm_2q& c) {
        os << "mm_2q @" << &c << "  " << c.stats_ << "\n";
        std::size_t idx = 0;
        for (auto it = c.begin(); it != c.end(); ++it, ++idx) {
            os << "  [" << idx << "] key='";
            if constexpr (lru::detail::is_formattable_v<Key>) {
                os << std::format("{}", it->key);
            } else {
                os << std::format("<key at {:#x}>", reinterpret_cast<std::uintptr_t>(&it->key));
            }
            os << "' value='";
            if constexpr (lru::detail::is_formattable_v<Value>) {
                os << std::format("{}", it->value);
            } else {
                os << std::format("<val at {:#x}>", reinterpret_cast<std::uintptr_t>(&it->value));
            }
            os << "'\n";
        }
        return os;
    }

    // --------------------------------------------------------------------
    // Eviction (public for pooled_cache / unified_cache::evict())
    // --------------------------------------------------------------------

    void evict() {
        cleanup_pending_deletion();
        // Eviction priority: Cold > Hot > Warm
        static const uint8_t eviction_order[] = {
            mm2q::kQueueCold,
            mm2q::kQueueHot,
            mm2q::kQueueWarm
        };

        for (auto qid : eviction_order) {
            auto* victim = find_eviction_victim(qid);
            if (victim) {
                auto evict_result = victim->refcount.markForEviction();
                if (evict_result != detail::MarkForEvictionResult::kSuccess) {
                    continue;  // try next victim in next queue
                }

                const auto& key = victim->key;
                size_type mem = calc_item_memory(key, victim->value);

                stats_.current_memory.fetch_sub(mem);
                stats_.register_eviction();
                if (callbacks_.has_eviction_callbacks()) {
                    Value value = std::move(victim->value);
                    callbacks_.collect_evict(key, std::move(value));
                }

                map_.erase(key);
                remove_from_queue(victim);
                victim->refcount.unmarkForEviction();
                detail::hazptr_domain::default_domain().retire(victim);
                break;
            }
        }
        stats_.current_size.store(total_size());
    }

    /// B15: 在指定队列中找到可淘汰的节点，考虑 EvictionPredicate 和活跃句柄。
    item_ptr find_eviction_victim(uint8_t qid) {
        auto* curr = queues_[qid].tail();
        if (!curr) return curr;

        const bool has_pred = static_cast<bool>(eviction_predicate_);
        size_t tries = 0;
        while (curr) {
            stats_.eviction_search_steps.fetch_add(1, std::memory_order_relaxed);
            // H0: 跳过有活跃句柄的节点（不计入 tries，这些节点绝对不能淘汰）
            if (curr->has_active_handle()) {
                stats_.pinned_skip_count.fetch_add(1, std::memory_order_relaxed);
                curr = static_cast<item_type*>(queues_[qid].get_prev(*curr));
                continue;
            }
            // B15: EvictionPredicate 否决（计入 tries）
            if (has_pred && !eviction_predicate_(curr->key, curr->value)) {
                curr = static_cast<item_type*>(queues_[qid].get_prev(*curr));
                if (++tries >= config_.eviction_search_tries) break;
                continue;
            }
            return curr;
        }
        return nullptr;
    }

private:
    // 3 separate intrusive lists for Hot/Warm/Cold queues
    item_list queues_[mm2q::kQueueCount];
    map_type map_;

    size_type max_size_ = unlimited;
    size_type max_memory_ = unlimited;
    mm_2q_config config_;

    // A5: try_lock_update 优化使用的独立内部锁，与统一缓存层锁解耦
    // B10: 缓存行对齐以避免 false sharing（对齐 CacheLib MMLru.h:474）
    struct alignas(64) aligned_mutex_t { std::mutex m; };
    mutable aligned_mutex_t update_mutex_;

    // Delayed promotion
    uint32_t lru_refresh_time_ = 0;
    uint32_t next_reconfigure_time_ = std::numeric_limits<uint32_t>::max();

    mutable stats_type stats_;
    mutable callback_mgr callbacks_;
    std::function<size_type(const Key&)> key_size_fn_;
    std::function<size_type(const Value&)> value_size_fn_;

    // B15: EvictionPredicate
    std::function<bool(const Key&, const Value&)> eviction_predicate_;
    // B7: LockedIterator 活跃标记
    std::atomic<bool> iterator_active_{false};

    // Items removed by force_del() that still have active handles.
    std::vector<item_ptr> pending_deletion_;

    /// Clean up pending-deletion items whose handles have all been released.
    /// Fires the deferred eviction callback (with value move) before deleting.
    void cleanup_pending_deletion() {
        auto it = pending_deletion_.begin();
        while (it != pending_deletion_.end()) {
            if (!(*it)->has_active_handle()) {
                auto* item = *it;
                callbacks_.collect_evict(item->key, std::move(item->value));
                // P1-5: Route through hazptr retire instead of raw delete —
                // a concurrent hazptr-protected reader may still hold a
                // hazard pointer to this item even with refcount=0.
                detail::hazptr_domain::default_domain().retire(item);
                it = pending_deletion_.erase(it);
            } else {
                ++it;
            }
        }
    }

    // Per-queue 访问计数器（对齐 CacheLib MM2Q.h:644 numHotAccesses_ 等）。
    // 使用 relaxed 内存序：计数仅用于可观测性，无发布/同步语义。
    std::atomic<std::size_t> num_hot_accesses_{0};
    std::atomic<std::size_t> num_warm_accesses_{0};
    std::atomic<std::size_t> num_cold_accesses_{0};

    // --------------------------------------------------------------------
    // Time utilities
    // --------------------------------------------------------------------

    static uint32_t current_time_sec() {
        return detail::cached_epoch_sec();
    }

    // --------------------------------------------------------------------
    // Internal helpers
    // --------------------------------------------------------------------

    size_type total_size() const {
        return queues_[0].size() + queues_[1].size() + queues_[2].size();
    }

    bool should_evict() const {
        if (max_size_ != unlimited && size() > max_size_) return true;
        if (max_memory_ != unlimited && current_memory() > max_memory_) return true;
        return false;
    }

    size_type calc_item_memory(const Key& key, const Value& value) const {
        size_type mem = item_overhead;
        if (key_size_fn_) mem += key_size_fn_(key) * 2;
        if (value_size_fn_) mem += value_size_fn_(value);
        return mem;
    }

    size_type expected_queue_size(uint8_t qid) const {
        // A7: 始终基于当前实际 size()，而非 max_size_（对齐 CacheLib MM2Q.h:843-883）
        auto total = size();
        switch (qid) {
            case mm2q::kQueueHot:
                return static_cast<size_type>(total * config_.hot_ratio);
            case mm2q::kQueueWarm:
                return static_cast<size_type>(total * config_.warm_ratio);
            case mm2q::kQueueCold:
                return total - expected_queue_size(mm2q::kQueueHot)
                            - expected_queue_size(mm2q::kQueueWarm);
            default:
                return 0;
        }
    }

    // ====================================================================
    // Record Access with delayed promotion
    // ====================================================================

    /// Record access to an item, with delayed promotion support.
    /// - Cold items: promoted to Warm (after delayed promotion check)
    /// - Hot/Warm items: moved to front of current queue (after delayed promotion check)
    /// Returns true if the item was actually promoted/moved.
    bool record_access(item_ptr item, access_mode mode) {
        return record_access_at(item, mode, current_time_sec());
    }

    /// Record access with a pre-computed current time.
    bool record_access_at(item_ptr item, access_mode mode, uint32_t curr) {
        assert(item != nullptr);

        // Check updateOnWrite/updateOnRead
        if ((mode == access_mode::write && !config_.update_on_write) ||
            (mode == access_mode::read && !config_.update_on_read)) {
            return false;
        }

        // A3 修正: CacheLib MM2Q (MM2Q.h:715-717) 没有 || !isAccessed(node) 分支。
        // mm_lru 和 mm_tiny_lfu 保留该分支（对齐 MMLru.h:540-542, MMTinyLFU.h:748-751），
        // 但 mm_2q 不做首次访问必提升——新插入节点需等待 refresh time 后才能提升。
        if (curr < item->hook.update_time + lru_refresh_time_) {
            return false;  // Not enough time since last promotion
        }

        // A5: try_lock_update 优化——若启用，则尝试加锁；失败则返回 false 跳过提升（不阻塞）。
        // 对齐 CacheLib MMLru.h:567-577。成功获取锁后正常执行提升逻辑。
        auto promote = [this, item, curr]() {
            // B1: 在 promote 路径上定期调整 lru_refresh_time_
            reconfigure_locked(curr);

            // 记录访问前所在队列，按队列类型递增 per-queue 计数器
            // （对齐 CacheLib MM2Q.h:725,736,751 的递增时机）
            uint8_t qid = item->queue_id;
            if (qid == mm2q::kQueueHot) {
                num_hot_accesses_.fetch_add(1);
            } else if (qid == mm2q::kQueueWarm) {
                num_warm_accesses_.fetch_add(1);
            } else if (qid == mm2q::kQueueCold) {
                num_cold_accesses_.fetch_add(1);
            }

            if (qid == mm2q::kQueueCold) {
                // Cold -> Warm promotion on access
                move_to_queue(item, mm2q::kQueueWarm);
                // 仅当配置开启时才在 Cold→Warm 晋升后触发 rebalance
                // （对齐 CacheLib MM2Q.h:739-741）
                if (config_.rebalance_on_record_access) {
                    rebalance();
                }
            } else if (qid == mm2q::kQueueHot || qid == mm2q::kQueueWarm) {
                // Move to front of current queue
                queues_[qid].move_to_head(*item);
            }

            item->hook.update_time = curr;
        };

        if (try_lock_update_enabled()) {
            std::unique_lock<std::mutex> lock(update_mutex_.m, std::try_to_lock);
            if (!lock) {
                stats_.try_lock_fail_count.fetch_add(1, std::memory_order_relaxed);
                return false;
            }
            promote();
        } else {
            promote();
        }

        return true;
    }

    // ====================================================================
    // Queue management
    // ====================================================================

    void move_to_queue(item_ptr item, uint8_t target_qid, bool to_tail = false) {
        assert(item != nullptr);
        uint8_t src_qid = item->queue_id;
        // 不变量：源队列与目标队列必须不同
        assert(src_qid != target_qid);
        // 不变量：队列 ID 合法
        assert(src_qid < mm2q::kQueueCount && target_qid < mm2q::kQueueCount);
        item->queue_id = target_qid;
        queues_[src_qid].remove(*item);
        if (to_tail) {
            queues_[target_qid].link_at_tail(*item);
        } else {
            queues_[target_qid].link_at_head(*item);
        }
        // kLinked stays set since item moves between queues within the same MM container
    }

    void remove_from_queue(item_ptr item) {
        uint8_t qid = item->queue_id;
        // A2: 清除 accessed 标志，对齐 CacheLib MMLru.h:744
        item->hook.clear_accessed();
        item->refcount.unmarkInMMContainer();
        queues_[qid].remove(*item);
    }

    // ====================================================================
    // Adaptive Refresh Time (CacheLib's reconfigureLocked)
    // ====================================================================

    /// B1: 基于 Warm 队列尾部年龄动态调整 lru_refresh_time_。
    /// 对齐 CacheLib MM2Q.h:1108-1122。
    void reconfigure_locked(uint32_t curr_time) {
        if (curr_time < next_reconfigure_time_) return;
        if (config_.mm_reconfigure_interval_secs == 0) return;

        next_reconfigure_time_ = curr_time + config_.mm_reconfigure_interval_secs;

        auto* warm_tail = queues_[mm2q::kQueueWarm].tail();
        uint32_t tail_age = warm_tail ? (curr_time - warm_tail->hook.update_time) : 0;
        auto new_refresh = std::min(
            std::max(config_.default_lru_refresh_time,
                     static_cast<uint32_t>(static_cast<double>(tail_age) *
                                           config_.lru_refresh_ratio)),
            config_type::k_lru_refresh_time_cap);
        lru_refresh_time_ = new_refresh;
    }

    /// B8: 获取淘汰年龄统计（对齐 CacheLib MM2Q.h:774-821）。
    struct eviction_age_stat {
        uint64_t oldest_element_age{0};
        uint64_t projected_age{0};
        uint64_t queue_size{0};
    };

    eviction_age_stat get_hot_age_stat(std::size_t projected_length = 0) const noexcept {
        eviction_age_stat stat;
        stat.queue_size = queues_[mm2q::kQueueHot].size();
        const auto curr_time = current_time_sec();
        const auto* node = queues_[mm2q::kQueueHot].tail();
        stat.oldest_element_age = node ? (curr_time - node->hook.update_time) : 0;
        for (std::size_t seen = 0; seen < projected_length && node != nullptr; ++seen) {
            node = static_cast<const item_type*>(queues_[mm2q::kQueueHot].get_prev(*node));
        }
        stat.projected_age = node ? (curr_time - node->hook.update_time) : stat.oldest_element_age;
        return stat;
    }

    eviction_age_stat get_warm_age_stat(std::size_t projected_length = 0) const noexcept {
        eviction_age_stat stat;
        stat.queue_size = queues_[mm2q::kQueueWarm].size();
        const auto curr_time = current_time_sec();
        const auto* node = queues_[mm2q::kQueueWarm].tail();
        stat.oldest_element_age = node ? (curr_time - node->hook.update_time) : 0;
        for (std::size_t seen = 0; seen < projected_length && node != nullptr; ++seen) {
            node = static_cast<const item_type*>(queues_[mm2q::kQueueWarm].get_prev(*node));
        }
        stat.projected_age = node ? (curr_time - node->hook.update_time) : stat.oldest_element_age;
        return stat;
    }

    eviction_age_stat get_cold_age_stat(std::size_t projected_length = 0) const noexcept {
        eviction_age_stat stat;
        stat.queue_size = queues_[mm2q::kQueueCold].size();
        const auto curr_time = current_time_sec();
        const auto* node = queues_[mm2q::kQueueCold].tail();
        stat.oldest_element_age = node ? (curr_time - node->hook.update_time) : 0;
        for (std::size_t seen = 0; seen < projected_length && node != nullptr; ++seen) {
            node = static_cast<const item_type*>(queues_[mm2q::kQueueCold].get_prev(*node));
        }
        stat.projected_age = node ? (curr_time - node->hook.update_time) : stat.oldest_element_age;
        return stat;
    }

    // ====================================================================
    // Insert / Update / Evict
    // ====================================================================

    template <typename V>
    void insert_new(const Key& key, V&& value) {
        if (max_size_ == 0) return;
        while (should_evict()) {
            auto old_size = size();
            evict();
            if (size() == old_size) break;
        }
        if (max_size_ != unlimited && size() >= max_size_) evict();

        auto mem = calc_item_memory(key, value);
        if (max_memory_ != unlimited) {
            while (!map_.empty() && max_memory_ - stats_.current_memory.load() < mem) {
                auto old_size = size();
                evict();
                if (size() == old_size) break;
            }
        }

        // Allocate new item — new items go to Hot queue
        auto* item = this->allocate_item(key, std::forward<V>(value));
        auto curr = current_time_sec();
        item->hook.update_time = curr;
        item->hook.clear_accessed();
        item->queue_id = mm2q::kQueueHot;

        queues_[mm2q::kQueueHot].link_at_head(*item);
        item->refcount.markInMMContainer();
        map_.insert(key, item);

        stats_.current_size.store(total_size());
        stats_.current_memory.fetch_add(mem);
        stats_.register_insertion();
        callbacks_.collect_insert(key, item->value);

        rebalance();
    }

    template <typename V>
    void update_existing(item_ptr item, V&& value, access_mode mode) {
        auto curr = current_time_sec();
        size_type old_mem = calc_item_memory(item->key, item->value);
        // P-MED-2 (T-H4): Strong exception guarantee via copy-then-swap.
        if constexpr (std::is_nothrow_swappable_v<Value> &&
                      std::is_constructible_v<Value, V>) {
            Value tmp(std::forward<V>(value));
            using std::swap;
            swap(item->value, tmp);
        } else {
            item->value = std::forward<V>(value);
        }
        record_access_at(item, mode, curr);
        size_type new_mem = calc_item_memory(item->key, item->value);
        if (new_mem > old_mem) {
            stats_.current_memory.fetch_add(new_mem - old_mem);
        } else if (new_mem < old_mem) {
            stats_.current_memory.fetch_sub(old_mem - new_mem);
        }
        if (should_evict()) {
            while (should_evict()) {
                auto old_size = size();
                evict();
                if (size() == old_size) break;
            }
        }
        // O7: Fire on_update for value changes on existing keys (distinct
        // from on_insert, which fires only for new key insertions).
        callbacks_.collect_update(item->key, item->value);
    }

    /// B14: 原位替换节点，保留 queue_id、update_time 与 accessed 状态（对齐 CacheLib MM2Q.h:991-1025）。
    void replace_node(item_ptr old_node, item_ptr new_node) {
        assert(old_node != nullptr && new_node != nullptr);
        uint8_t qid = old_node->queue_id;
        new_node->hook.update_time = old_node->hook.update_time;
        if (old_node->hook.is_accessed()) {
            new_node->hook.set_accessed();
        } else {
            new_node->hook.clear_accessed();
        }
        new_node->queue_id = qid;
        queues_[qid].replace(*old_node, *new_node);
        // 更新 map
        map_.insert_or_assign(old_node->key, new_node);
    }

    /// B15: 设置淘汰谓词。
    void set_eviction_predicate(std::function<bool(const Key&, const Value&)> pred) {
        eviction_predicate_ = std::move(pred);
    }

    void erase_impl(const Key& key) {
        auto ptr = map_.find(key);
        if (!ptr) return;
        auto* item = ptr;
        if (item->has_active_handle()) return;
        size_type mem = calc_item_memory(item->key, item->value);

        stats_.current_memory.fetch_sub(mem);
        callbacks_.collect_evict(item->key, std::move(item->value));

        map_.erase(key);
        remove_from_queue(item);
        detail::hazptr_domain::default_domain().retire(item);
        stats_.current_size.store(total_size());
    }

    void rebalance() {
        if (max_size_ == unlimited) return;

        // 不变量：三队列 size 之和等于 map 实际持有量
        assert(hot_size() + warm_size() + cold_size() == size());

        // A5: rebalance 顺序为 Warm→Hot→Cold（对齐 CacheLib MM2Q.h:843-883）
        // Move overflow from Warm to Cold first
        auto expected_warm = expected_queue_size(mm2q::kQueueWarm);
        while (queues_[mm2q::kQueueWarm].size() > expected_warm) {
            auto* tail = queues_[mm2q::kQueueWarm].tail();
            assert(tail != nullptr);
            if (!tail) break;
            if (map_.contains(tail->key)) {
                move_to_queue(tail, mm2q::kQueueCold, /*to_tail=*/true);
            } else {
                break;
            }
        }

        // Move overflow from Hot to Cold
        auto expected_hot = expected_queue_size(mm2q::kQueueHot);
        while (queues_[mm2q::kQueueHot].size() > expected_hot) {
            auto* tail = queues_[mm2q::kQueueHot].tail();
            assert(tail != nullptr);
            if (!tail) break;
            if (map_.contains(tail->key)) {
                move_to_queue(tail, mm2q::kQueueCold, /*to_tail=*/true);
            } else {
                break;
            }
        }

        // 不变量：rebalance 后三队列 size 之和仍等于 map 实际持有量
        assert(hot_size() + warm_size() + cold_size() == size());
    }

    // ====================================================================
    // S0: Faithful serialization rebuild（public for deserialization）
    // ====================================================================
public:
    /// 从序列化数据重建缓存，恢复 item→queue 映射。
    /// 输入 items 按队列连续排列（Hot→Warm→Cold，队列内 MRU→LRU）。
    template <typename InputIt>
    void rebuild_from_serialized(InputIt first, InputIt last) {
        flush();
        for (auto it = first; it != last; ++it) {
            auto* item = this->allocate_item(it->key, it->value);
            item->hook.update_time = it->update_time;
            if (it->flags & detail::intrusive_hook::kAccessedFlag) {
                item->hook.set_accessed();
            }
            item->queue_id = it->queue_id < mm2q::kQueueCount ? it->queue_id : mm2q::kQueueHot;
            queues_[item->queue_id].link_at_tail(*item);
            item->refcount.markInMMContainer();
            map_.insert(item->key, item);
            stats_.current_memory.fetch_add(calc_item_memory(item->key, item->value));
        }
        stats_.current_size.store(total_size());
        stats_.register_insertion();
        rebalance();
    }
};


// ============================================================================
// TinyLFU Strategy - mm_tiny_lfu
// ============================================================================


// ============================================================================
// TinyLFU Strategy Configuration
// ============================================================================

struct mm_tiny_lfu_config {
    /// Ratio of Tiny (window) cache to total cache (default 1%)
    double window_to_cache_size_ratio = 0.01;

    /// CountMinSketch error rate (default 0.5)
    double cms_error_rate = 0.5;

    /// CountMinSketch confidence (default 0.99)
    double cms_confidence = 0.99;

    /// Default time between promotions for the same item (seconds).
    /// 0 = no delay (traditional behavior).
    uint32_t default_lru_refresh_time = 60;

    /// Whether to promote the item on write access.
    bool update_on_write = false;

    /// Whether to promote the item on read access.
    bool update_on_read = true;

    /// 平局时新元素（Tiny tail）是否胜出。
    /// true=平局时 Tiny 晋升（>=，CacheLib 默认）；
    /// false=平局时老元素保留（>）。
    /// 对齐 MMTinyLFU.h:603-607 的 admitToMain。
    bool newcomer_wins_on_tie = true;

    /// A5: 是否在 record_access 中使用 try_to_lock 跳过提升。
    /// 对齐 CacheLib MMLru.h:567-577 的 tryLockUpdate。
    /// 默认 true，因为 unified_cache 已提供外层并发保护，
    /// 阻塞等待 update_mutex_ 会造成双重锁开销。
    bool try_lock_update = true;

    /// Use combined lock for eviction iterators.
    bool use_combined_lock_for_iterators = false;

    /// B15: 淘汰搜索次数上限。
    size_t eviction_search_tries = 3;

    /// Ratio for adaptive refresh time adjustment.
    double lru_refresh_ratio = 0.0;

    /// Interval in seconds for reconfiguring the adaptive refresh time.
    /// 0 = disabled (default).
    uint32_t mm_reconfigure_interval_secs = 0;

    /// Expected number of items for automatic bucket count sizing.
    /// 0 = use default bucket count (1024). When > 0, the internal hash
    /// table is pre-sized via concurrent_hash_table::buckets_for_items()
    /// to keep average chain length ≤ 0.25 at the expected load.
    size_t expected_items = 0;

    /// Custom node allocation function for non-EmbeddedChain hash table nodes.
    /// nullptr (default) = standard new/delete allocation.
    void* (*alloc_fn)(std::size_t) = nullptr;

    /// Custom node deallocation function (must pair with alloc_fn).
    void  (*dealloc_fn)(void*) = nullptr;

    /// In read-heavy mode, skip the try_acquire_write_lock_for_key() step in
    /// get() and always defer LRU promotion to the TLS access ring.
    /// Default true — production-recommended for lower read-path latency.
    bool defer_promotion = true;

    // Max lruRefreshTime cap
    static constexpr uint32_t k_lru_refresh_time_cap = 900;

    // B4: 配置校验——各 ratio/rate 范围合法
    mm_tiny_lfu_config() noexcept = default;

    void validate() const {
        if (window_to_cache_size_ratio <= 0.0 || window_to_cache_size_ratio > 0.5) {
            throw std::invalid_argument(
                "mm_tiny_lfu_config: window_to_cache_size_ratio must be in (0, 0.5]");
        }
        if (cms_error_rate <= 0.0 || cms_error_rate >= 1.0) {
            throw std::invalid_argument(
                "mm_tiny_lfu_config: cms_error_rate must be in (0, 1)");
        }
        if (cms_confidence <= 0.0 || cms_confidence >= 1.0) {
            throw std::invalid_argument(
                "mm_tiny_lfu_config: cms_confidence must be in (0, 1)");
        }
        if (!(lru_refresh_ratio >= 0.0)) {
            throw std::invalid_argument(
                "mm_tiny_lfu_config: lru_refresh_ratio must be non-negative");
        }
    }
};

// ============================================================================
// TinyLFU Strategy
// ============================================================================

/// TinyLFU eviction strategy with frequency-aware admission.
/// Uses a small window cache + main cache, with CountMinSketch
/// frequency estimation to make admission decisions.
/// When eviction is needed, the Tiny tail and Main tail frequencies
/// are compared; the higher-frequency item stays.
template <
    typename Key,
    typename Value,
    typename Hash = std::hash<Key>,
    typename KeyEqual = std::equal_to<Key>,
    typename ProbingStyle = detail::chain_probing_tag,
    bool Segmented = false
>
/// A4: 线程安全契约——此类非线程安全，调用方必须确保在外层 unified_cache 锁内访问。
/// 内部 update_mutex_ 仅用于 try_lock_update 路径的解耦优化，不保证 MM 层独立线程安全。
class mm_tiny_lfu : public detail::mm_allocator_mixin<detail::cache_item<Key, Value>> {
public:
    using key_type = Key;
    using mapped_type = Value;
    using size_type = std::size_t;
    using config_type = mm_tiny_lfu_config;

    // Item type: cache_item with embedded intrusive hook
    using item_type = detail::cache_item<Key, Value>;
    using item_ptr = item_type*;

    // Intrusive list type
    using item_list = detail::intrusive_list<item_type, detail::intrusive_hook, detail::default_get_hook<item_type>>;

    // Map: Key -> item pointer
    using map_type = std::conditional_t<
        Segmented,
        detail::segmented_concurrent_hash_table<Key, item_ptr, Hash, KeyEqual, true, ProbingStyle, 64>,
        detail::concurrent_hash_table<Key, item_ptr, Hash, KeyEqual, true, ProbingStyle>
    >;

    using callback_mgr = callback_manager<Key, Value>;
    using stats_type = cache_stats;
    using sketch_type = detail::count_min_sketch<Key, Hash>;

    static constexpr size_type npos = unlimited;
    // Item overhead: hook + key + value + handle + map entry
    static constexpr size_type item_overhead = sizeof(item_type) + map_type::entry_overhead;

    // Queue IDs
    static constexpr uint8_t kTinyQueue = 0;
    static constexpr uint8_t kMainQueue = 1;

    // --------------------------------------------------------------------
    // Constructors / Destructor
    // --------------------------------------------------------------------

    mm_tiny_lfu() : mm_tiny_lfu(mm_tiny_lfu_config{}) {}

    explicit mm_tiny_lfu(const mm_tiny_lfu_config& config)
        : config_(config)
        , map_(config.expected_items > 0
            ? map_type::buckets_for_items(config.expected_items)
            : 1024,
            config.alloc_fn, config.dealloc_fn)
        , lru_refresh_time_(config.default_lru_refresh_time)
        , next_reconfigure_time_(config.mm_reconfigure_interval_secs == 0
            ? std::numeric_limits<uint32_t>::max()
            : current_time_sec() + config.mm_reconfigure_interval_secs)
        , sketch_(1000, config.cms_error_rate, config.cms_confidence) {}

    mm_tiny_lfu(size_type max_size, const mm_tiny_lfu_config& config = mm_tiny_lfu_config{})
        : mm_tiny_lfu(config) {
        detail::validate_capacity(max_size, unlimited);
        max_size_ = max_size;
        stats_.max_size.store(max_size);
        sketch_.set_max_window_size(std::max(max_size, size_type(100)));
    }

    mm_tiny_lfu(size_type max_size, size_type max_memory,
                 const mm_tiny_lfu_config& config = mm_tiny_lfu_config{})
        : mm_tiny_lfu(config) {
        detail::validate_capacity(max_size, max_memory);
        max_size_ = max_size;
        max_memory_ = max_memory;
        stats_.max_size.store(max_size);
        stats_.max_memory.store(max_memory);
        sketch_.set_max_window_size(std::max(max_size, size_type(100)));
    }

    ~mm_tiny_lfu() {
        // Clean up pending deletions first (items with active handles that were force_del'd)
        for (auto* item : pending_deletion_) {
            delete item;
        }
        pending_deletion_.clear();
        // Use flush() to properly erase items from both map and list before deletion,
        // preventing use-after-free when the hash table destructor iterates its chains.
        flush();
    }

    // Non-copyable, non-movable (owns item pointers)
    mm_tiny_lfu(const mm_tiny_lfu&) = delete;
    mm_tiny_lfu& operator=(const mm_tiny_lfu&) = delete;

    // --------------------------------------------------------------------
    // Core cache API
    // --------------------------------------------------------------------

    template <typename V>
    void set(const Key& key, V&& value) {
        cleanup_pending_deletion();
        auto ptr = map_.find(key);
        if (ptr) {
            update_existing(ptr, std::forward<V>(value), access_mode::write);
        } else {
            insert_new(key, std::forward<V>(value));
        }
    }

    template <typename V>
    bool add(const Key& key, V&& value) {
        cleanup_pending_deletion();
        auto ptr = map_.find(key);
        if (ptr) {
            // 频率计数由 record_access 统一更新，此处不再重复记录
            record_access(ptr, access_mode::read);
            return false;
        }
        insert_new(key, std::forward<V>(value));
        return true;
    }

    template <typename V>
    bool replace(const Key& key, V&& value) {
        cleanup_pending_deletion();
        auto ptr = map_.find(key);
        if (!ptr) return false;
        update_existing(ptr, std::forward<V>(value), access_mode::write);
        return true;
    }

    read_handle<Value> get(const Key& key) {
        auto ptr = map_.find(key);
        if (!ptr) {
            stats_.register_miss();
            callbacks_.collect_miss(key);
            return {};
        }
        auto* item = ptr;
        // 频率计数由 record_access 统一更新，此处不再重复记录
        record_access(item, access_mode::read);

        // Tiny -> Main promotion on access
        if (item->queue_id == kTinyQueue) {
            maybe_promote_from_tiny();
        }

        stats_.register_hit();
        callbacks_.collect_hit(key, item->value);
        return read_handle<Value>{&item->value, &item->refcount};
    }

    read_handle<const Value> get(const Key& key) const {
        auto ptr = map_.find(key);
        if (!ptr) {
            stats_.register_miss();
            callbacks_.collect_miss(key);
            return {};
        }
        auto* item = ptr;
        stats_.register_hit();
        callbacks_.collect_hit(key, item->value);
        return read_handle<const Value>{&item->value, &item->refcount};
    }

    /// H0: Peek with handle — 不提升 LFU，返回 handle 防止持有期被淘汰。
    /// Uses find_and_pin_lockfree() to attempt lock-free pinning first
    /// (optimistic read + incRef without bucket lock), falling back to
    /// find_and_pin() (shared lock path) if the lock-free pin fails.
    /// No stripe-level read lock is needed.
    read_handle<const Value> peek(const Key& key) const {
        auto pin_fn = [](auto* item) {
            return item->refcount.incRef() == detail::IncResult::kIncOk;
        };
        auto ptr = map_.find_and_pin_lockfree(key, pin_fn);
        if (!ptr) return {};
        return read_handle<const Value>{&ptr->value, &ptr->refcount, pre_pinned};
    }

    /// Internal: peek that returns mutable handle (for optimistic get path).
    read_handle<Value> peek_for_get(const Key& key) {
        return peek_for_get_with_hash(key, Hash{}(key));
    }

    /// T16.4: peek_for_get with a pre-computed hash. The hash MUST be
    /// the result of Hash{}(key) — callers are responsible for hash
    /// compatibility. Used by bulk_get to avoid re-hashing each key
    /// for both shard dispatch and hash-table lookup.
    read_handle<Value> peek_for_get_with_hash(const Key& key, std::size_t hash) {
        auto pin_fn = [](auto* item) {
            return item->refcount.incRef() == detail::IncResult::kIncOk;
        };
        auto ptr = map_.find_and_pin_lockfree_with_hash(key, hash, pin_fn);
        if (!ptr) return {};
        // P1-1: Inline TTL check. Fast path: expiry_ns == 0 (no TTL) skips
        // steady_clock::now() entirely. Only items with a TTL set pay the
        // clock read, and only on cache hits. Expired items are unpinned
        // (decRef) and reported as a miss; the actual eviction is handled
        // lazily by evict_expired() / the background TTL cleaner, NOT here,
        // so we don't need a write lock on the hot path.
        if (ptr->expiry_ns != 0) {
            // P1-10: Track TTL check frequency on the read path.
            stats_.ttl_checked_count.fetch_add(1, std::memory_order_relaxed);
            auto now_ns = static_cast<std::uint64_t>(
                std::chrono::steady_clock::now().time_since_epoch().count());
            if (now_ns >= ptr->expiry_ns) {
                stats_.ttl_expired_count.fetch_add(1, std::memory_order_relaxed);
                ptr->refcount.decRef();
                stats_.register_miss();
                callbacks_.collect_miss(key);
                return {};
            }
        }
        return read_handle<Value>{&ptr->value, &ptr->refcount, pre_pinned};
    }

    bool del(const Key& key) {
        cleanup_pending_deletion();
        auto ptr = map_.find(key);
        if (!ptr) return false;
        if (ptr->has_active_handle()) return false;
        erase_impl(key);
        return true;
    }

    /// Force delete a key even if it has active handles.
    /// The item is immediately removed from the map and queue,
    /// but memory is not freed until all handles are released.
    /// Returns true if the key was found (and removed from the map).
    bool force_del(const Key& key) {
        cleanup_pending_deletion();
        auto ptr = map_.find(key);
        if (!ptr) return false;
        auto* item = ptr;
        size_type mem = calc_item_memory(item->key, item->value);

        map_.erase(key);
        // Capture item state BEFORE remove_from_queue() poisons the item's memory.
        bool has_handle = item->has_active_handle();
        if (!has_handle) {
            callbacks_.collect_evict(item->key, std::move(item->value));
        }
        remove_from_queue(item);

        stats_.current_memory.fetch_sub(mem);
        stats_.current_size.store(total_size());

        if (!has_handle) {
            detail::hazptr_domain::default_domain().retire(item);
        } else {
            pending_deletion_.push_back(item);
        }
        return true;
    }

    std::optional<Value> pop(const Key& key) {
        auto ptr = map_.find(key);
        if (!ptr) return std::nullopt;
        auto* item = ptr;
        if (item->has_active_handle()) return std::nullopt;
        size_type mem = calc_item_memory(item->key, item->value);
        Value value = std::move(item->value);
        stats_.current_memory.fetch_sub(mem);
        remove_from_queue(item);
        map_.erase(key);
        detail::hazptr_domain::default_domain().retire(item);
        stats_.current_size.store(total_size());
        return value;
    }

    std::optional<std::pair<Key, Value>> pop_lru() {
        cleanup_pending_deletion();
        // Try Tiny tail first, then Main tail; skip items pinned by handles.
        auto* tail = tiny_queue_.tail();
        size_t tries = 0;
        while (tail && tail->has_active_handle()) {
            tail = static_cast<item_type*>(tiny_queue_.get_prev(*tail));
            if (++tries >= config_.eviction_search_tries) { tail = nullptr; break; }
        }
        if (!tail) {
            tail = main_queue_.tail();
            tries = 0;
            while (tail && tail->has_active_handle()) {
                tail = static_cast<item_type*>(main_queue_.get_prev(*tail));
                if (++tries >= config_.eviction_search_tries) { tail = nullptr; break; }
            }
        }
        if (!tail) return std::nullopt;
        if (!map_.contains(tail->key)) return std::nullopt;
        std::pair<Key, Value> result{std::move(tail->key), std::move(tail->value)};
        size_type mem = calc_item_memory(result.first, result.second);
        stats_.current_memory.fetch_sub(mem);
        remove_from_queue(tail);
        map_.erase(result.first);
        detail::hazptr_domain::default_domain().retire(tail);
        stats_.current_size.store(total_size());
        return result;
    }

    /// Flush the cache. Items pinned by an active read_handle are left in place.
    ///
    /// P1-8 (T2.6 bugfix): Two-pass deferred retirement — see mm_lru::flush()
    /// for the full rationale. Bypasses erase_impl() to apply markForEviction()
    /// directly on the iterated item. All retirements are deferred to Pass 2
    /// after both queues have been traversed.
    void flush() {
        // Pass 1: iterate all queues, collect items to retire.
        std::vector<item_ptr> to_retire;
        auto flush_queue = [this, &to_retire](item_list& queue) {
            auto* curr = queue.head();
            while (curr) {
                // Save next before any mutation. remove() only clears curr's hook
                // pointers; the next node remains valid and linked.
                auto* next = queue.get_next(*curr);
                if (!curr->has_active_handle()) {
                    auto evict_result = curr->refcount.markForEviction();
                    if (evict_result != detail::MarkForEvictionResult::kSuccess) {
                        curr = next;
                        continue;
                    }
                    size_type mem = calc_item_memory(curr->key, curr->value);
                    stats_.current_memory.fetch_sub(mem);
                    callbacks_.collect_evict(curr->key, std::move(curr->value));
                    // Item may or may not be in the map (transient queue state).
                    map_.erase(curr->key);
                    queue.remove(*curr);
                    curr->refcount.unmarkForEviction();
                    to_retire.push_back(curr);
                }
                curr = next;
            }
        };
        flush_queue(tiny_queue_);
        flush_queue(main_queue_);
        // P1-8 (T2.6 bugfix, phase 3): Refresh hash stats BEFORE retiring
        // any items — see mm_lru::flush() for the full rationale.
        refresh_hash_stats();
        // Pass 2: retire all collected items after iteration is complete.
        // mm_tiny_lfu currently has no use_ebr config; always use hazptr.
        for (auto* item : to_retire) {
            detail::hazptr_domain::default_domain().retire(item);
        }
        sketch_.reset();
        stats_.current_size.store(total_size());
        cleanup_pending_deletion();
    }

    bool contains(const Key& key) const {
        return map_.contains(key);
    }

    // --------------------------------------------------------------------
    // Capacity
    // --------------------------------------------------------------------

    bool empty() const noexcept { return map_.empty(); }
    size_type size() const noexcept { return map_.size(); }
    size_type max_size() const noexcept { return max_size_; }
    size_type max_memory() const noexcept { return max_memory_; }
    size_type current_memory() const noexcept { return stats_.current_memory.load(); }

    /// Estimate memory that would be accounted for an item with the given key
    /// and value, including the fixed item overhead and any custom size
    /// calculators registered via set_key/value_size_calculator().
    size_type estimate_item_memory(const Key& key, const Value& value) const {
        return calc_item_memory(key, value);
    }
    size_type tiny_size() const noexcept { return tiny_queue_.size(); }
    size_type main_size() const noexcept { return main_queue_.size(); }

    /// Check if any item (in any queue or pending deletion) has an active handle.
    bool has_active_handles() const noexcept {
        for (auto it = tiny_queue_.begin(); it != tiny_queue_.end(); ++it) {
            if (it->has_active_handle()) return true;
        }
        for (auto it = main_queue_.begin(); it != main_queue_.end(); ++it) {
            if (it->has_active_handle()) return true;
        }
        for (auto* item : pending_deletion_) {
            if (item->has_active_handle()) return true;
        }
        return false;
    }

    void max_size(size_type new_max) {
        max_size_ = new_max;
        stats_.max_size.store(new_max);
        sketch_.set_max_window_size(std::max(new_max, size_type(100)));
        if (new_max != npos) shrink_to_fit();
    }

    void max_memory(size_type new_max) {
        max_memory_ = new_max;
        stats_.max_memory.store(new_max);
        if (new_max != npos) shrink_to_fit();
    }

    void shrink_to_fit() {
        while (should_evict()) {
            auto old_size = size();
            evict();
            if (size() == old_size) break;
        }
    }

    // --------------------------------------------------------------------
    // Statistics and callbacks
    // --------------------------------------------------------------------

    stats_type& stats() noexcept { return stats_; }
    const stats_type& stats() const noexcept { return stats_; }
    callback_mgr& callbacks() noexcept { return callbacks_; }
    const callback_mgr& callbacks() const noexcept { return callbacks_; }

    // P1-7: Number of items in pending-deletion state (removed from cache
    // but still pinned by active read_handles). Best-effort count — may
    // race with concurrent writes. For monitoring only.
    std::size_t pending_deletion_count() const noexcept {
        return pending_deletion_.size();
    }

    /// Refresh hash table diagnostic stats (load factor, max chain length).
    /// O(bucket_count) scan — call periodically, not on every operation.
    void refresh_hash_stats() const noexcept {
        stats_.hash_load_factor.store(map_.load_factor(), std::memory_order_relaxed);
        stats_.max_chain_length.store(map_.max_chain_length(), std::memory_order_relaxed);
        // P1-1: Refresh rehash diagnostics from the hash table.
        stats_.rehash_count.store(map_.rehash_count(), std::memory_order_relaxed);
        stats_.rehash_total_time_ns.store(map_.rehash_total_time_ns(), std::memory_order_relaxed);
        stats_.rehash_migrated_items.store(map_.rehash_migrated_items(), std::memory_order_relaxed);
        // T13.1: Refresh overload threshold and event counter from the
        // hash table. These mirror the live state in concurrent_hash_table.
        stats_.hash_overload_threshold.store(map_.hash_overload_threshold(), std::memory_order_relaxed);
        stats_.hash_overload_events.store(map_.hash_overload_events(), std::memory_order_relaxed);
    }

    /// T-B4 (P2-10): Forward to the underlying hash table's diagnostics
    /// cache refresh. Only segmented_concurrent_hash_table implements
    /// this (regular concurrent_hash_table doesn't cache — its
    /// `max_chain_length()` is already a single-table scan, cheap enough
    /// to not warrant caching). For non-segmented tables this is a no-op
    /// via `if constexpr` (zero-cost). The background rehash balancer
    /// invokes this unconditionally.
    void refresh_diagnostics_cache() const noexcept {
        if constexpr (requires { map_.refresh_diagnostics_cache(); }) {
            map_.refresh_diagnostics_cache();
        }
    }

    /// T-B4 (P2-10): Forward to the underlying hash table's age metric.
    /// Returns `std::numeric_limits<std::uint64_t>::max()` if the cache
    /// has never been refreshed or the underlying table doesn't cache.
    /// Operators should check the `segmented_hash_table` flag in
    /// diagnostics() before relying on this value — non-segmented tables
    /// always report max (no cache, so age is meaningless).
    std::uint64_t diagnostics_cache_age_ms() const noexcept {
        if constexpr (requires { map_.diagnostics_cache_age_ms(); }) {
            return map_.diagnostics_cache_age_ms();
        }
        return std::numeric_limits<std::uint64_t>::max();
    }

    // --------------------------------------------------------------------
    // Memory policy
    // --------------------------------------------------------------------

    void set_key_size_calculator(std::function<size_type(const Key&)> func) {
        key_size_fn_ = std::move(func);
    }

    void set_value_size_calculator(std::function<size_type(const Value&)> func) {
        value_size_fn_ = std::move(func);
    }

    // --------------------------------------------------------------------
    // TinyLFU-specific API
    // --------------------------------------------------------------------

    const sketch_type& sketch() const noexcept { return sketch_; }
    /// S3: 非 const CMS 访问（用于反序列化恢复 CMS 状态）。
    sketch_type& sketch_mut() noexcept { return sketch_; }

    const mm_tiny_lfu_config& config() const noexcept { return config_; }

    /// Set custom hash table node allocation/deallocation functions.
    void set_hash_alloc_fns(void* (*alloc_fn)(std::size_t), void (*dealloc_fn)(void*)) {
        config_.alloc_fn = alloc_fn;
        config_.dealloc_fn = dealloc_fn;
        map_.set_alloc_fns(alloc_fn, dealloc_fn);
    }

    void set_config(const mm_tiny_lfu_config& config) {
        config_ = config;
        map_.set_alloc_fns(config.alloc_fn, config.dealloc_fn);
        lru_refresh_time_ = config.default_lru_refresh_time;
        next_reconfigure_time_ = config.mm_reconfigure_interval_secs == 0
            ? std::numeric_limits<uint32_t>::max()
            : current_time_sec() + config.mm_reconfigure_interval_secs;
        maybe_promote_from_tiny();
    }

    /// Pre-allocate hash table buckets for `expected_items` entries.
    void reserve(size_type expected_items) {
        map_.reserve(expected_items);
    }

    /// Enable/disable incremental rehash for the hash table.
    /// When enabled, rehash migrates buckets incrementally across multiple
    /// operations instead of blocking all writers during a single rehash.
    /// This reduces write-path latency spikes under load.
    void set_incremental_rehash(bool enabled) {
        map_.set_incremental_rehash(enabled);
    }

    /// Query whether incremental rehash is enabled.
    bool incremental_rehash_enabled() const noexcept {
        return map_.incremental_rehash_enabled();
    }

    /// P0-5 (T1.3): Advance any in-progress incremental rehash by one
    /// per-call migration budget (kRehashFinishMaxBucketsPerCall).
    /// Called by the background rehash balancer to ensure stalled
    /// rehashes eventually complete without requiring writes to the
    /// affected hash table. No-op when no rehash is in progress.
    void advance_incremental_rehash() noexcept {
        map_.rehash_finish();
    }

    /// T11.5: String-based strategy setter (see concurrent_hash_table::set_rehash_strategy).
    bool set_rehash_strategy(std::string_view strategy) noexcept {
        return map_.set_rehash_strategy(strategy);
    }
    std::string_view rehash_strategy() const noexcept {
        return map_.rehash_strategy();
    }

    /// T11.3: Number of writes blocked by a non-incremental (blocking) rehash.
    std::size_t rehash_blocked_writes_count() const noexcept {
        return map_.rehash_blocked_writes_count();
    }

    /// P1-5: Number of times find_and_pin_lockfree fell back to the
    /// lock-protected path because the target segment was in incremental
    /// rehash. Non-zero values indicate the lock-free read path is being
    /// degraded by rehash activity.
    std::size_t rehash_lockfree_fallback_count() const noexcept {
        return map_.rehash_lockfree_fallback_count();
    }

    /// P0-D: Ratio of the hash table currently in an incremental rehash.
    /// For non-segmented tables: 0.0 or 1.0 (whole table rehashing or not).
    /// For segmented tables: fraction of segments currently rehashing.
    /// Exposed as a Prometheus gauge to detect sustained rehash pressure.
    float rehash_in_progress_ratio() const noexcept {
        return map_.rehash_in_progress_ratio();
    }

    /// T13.1: Set the hash table load factor overload threshold.
    /// See concurrent_hash_table::set_hash_overload_threshold.
    void set_hash_overload_threshold(float threshold) noexcept {
        map_.set_hash_overload_threshold(threshold);
    }

    float hash_overload_threshold() const noexcept {
        return map_.hash_overload_threshold();
    }

    std::size_t hash_overload_events() const noexcept {
        return map_.hash_overload_events();
    }

    /// T13.2: Register an overload callback on the underlying hash table.
    void set_overload_callback(std::function<void(float, float)> cb) {
        map_.set_overload_callback(std::move(cb));
    }

    /// P2-4 (T2.4): Toggle async mode for the overload callback.
    /// Forwarded to the underlying hash table. See
    /// `concurrent_hash_table::set_async_overload_callback` for semantics.
    void set_async_overload_callback(bool enabled) noexcept {
        map_.set_async_overload_callback(enabled);
    }

    /// P2-4 (T2.4): Drain pending overload events from the underlying
    /// hash table and dispatch the registered callback for each. Returns
    /// the number of events drained. Designed to be called from a
    /// background worker (e.g. the `event_drain_worker` in `unified_cache`).
    std::size_t drain_overload_callbacks() {
        return map_.drain_overload_callbacks();
    }

    /// Whether an incremental rehash is currently in progress.
    bool is_rehashing() const noexcept { return map_.is_rehashing(); }
    /// Buckets fully migrated so far during the in-progress rehash (0 if none).
    size_type rehash_progress() const noexcept { return map_.rehash_progress(); }
    /// New bucket count target for the in-progress rehash (0 if none).
    size_type rehash_new_bucket_count() const noexcept { return map_.rehash_new_bucket_count(); }
    /// Old bucket count for the in-progress rehash (0 if none).
    size_type rehash_old_bucket_count() const noexcept { return map_.rehash_old_bucket_count(); }
    /// Total number of hash table buckets currently allocated.
    size_type bucket_count() const noexcept { return map_.bucket_count(); }

    uint32_t refresh_time() const noexcept { return lru_refresh_time_; }

    /// A5: 返回 try_lock_update 配置，用于 record_access 内的 try_to_lock 优化
    bool try_lock_update_enabled() const noexcept { return config_.try_lock_update; }

    /// Promote an item by key without triggering hit statistics or callbacks.
    /// This is a side-effect-free alternative to get() for batch promotion
    /// (e.g., from TLS ring flush). Returns true if the key was found and
    /// the item was eligible for promotion.
    bool promote(const Key& key) {
        auto ptr = map_.find(key);
        if (!ptr) return false;
        return record_access(ptr, access_mode::read);
    }

    /// Serialization support: access all items via the internal map.
    const map_type& internal_map() const noexcept { return map_; }

    // --------------------------------------------------------------------
    // LockedIterator (B7)
    // --------------------------------------------------------------------

    /// B7: 持有锁的迭代器。使用 locked_iterator_guard 管理锁生命周期。
    class LockedIterator {
    public:
        LockedIterator(mm_tiny_lfu& mm)
            : guard_(mm.update_mutex_.m, mm.iterator_active_), mm_(&mm) {
            curr_tiny_ = mm_->tiny_queue_.tail();
            curr_main_ = mm_->main_queue_.tail();
            // 初始化到有效位置
            if (!curr_tiny_ && curr_main_) { use_main_ = true; }
            else { use_main_ = false; }
        }

        ~LockedIterator() = default;
        LockedIterator(const LockedIterator&) = delete;
        LockedIterator& operator=(const LockedIterator&) = delete;
        LockedIterator(LockedIterator&& other) noexcept
            : guard_(std::move(other.guard_)), mm_(other.mm_),
              curr_tiny_(other.curr_tiny_), curr_main_(other.curr_main_),
              use_main_(other.use_main_) {}

        void destroy() { guard_.destroy(); }

        void resetToBegin() {
            curr_tiny_ = mm_->tiny_queue_.tail();
            curr_main_ = mm_->main_queue_.tail();
            use_main_ = (!curr_tiny_ && curr_main_);
        }

        bool next() {
            if (!use_main_ && curr_tiny_) {
                auto* next = mm_->tiny_queue_.get_prev(*curr_tiny_);
                if (next) { curr_tiny_ = next; return true; }
                // Tiny 到头，切到 Main
                use_main_ = true;
                if (curr_main_) return true;
            }
            if (use_main_ && curr_main_) {
                auto* next = mm_->main_queue_.get_prev(*curr_main_);
                if (next) { curr_main_ = next; return true; }
            }
            return false;
        }

        item_ptr operator->() { return use_main_ ? curr_main_ : curr_tiny_; }
        explicit operator bool() const { return curr_tiny_ || curr_main_; }

    private:
        detail::locked_iterator_guard<> guard_;
        mm_tiny_lfu* mm_;
        item_ptr curr_tiny_ = nullptr;
        item_ptr curr_main_ = nullptr;
        bool use_main_ = false;
    };

    // --------------------------------------------------------------------
    // Const iterator: traverses Tiny→Main, each queue MRU→LRU
    // --------------------------------------------------------------------

    class const_iterator {
    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = const item_type;
        using reference = const value_type&;
        using pointer = value_type*;
        using difference_type = std::ptrdiff_t;

        const_iterator() : mm_(nullptr), qid_(2) {}

        const_iterator(const mm_tiny_lfu* mm, uint8_t qid, typename item_list::const_iterator it)
            : mm_(mm), qid_(qid), it_(it) {}

        reference operator*() const { return *it_; }
        pointer operator->() const { return &*it_; }

        const_iterator& operator++() {
            ++it_;
            advance_to_nonempty();
            return *this;
        }
        const_iterator operator++(int) { auto t = *this; ++*this; return t; }

        bool operator==(const const_iterator& o) const {
            if (mm_ == nullptr && o.mm_ == nullptr) return true;
            if (qid_ >= 2 && o.qid_ >= 2) return true;
            return qid_ == o.qid_ && it_ == o.it_;
        }
        bool operator!=(const const_iterator& o) const { return !(*this == o); }

    private:
        void advance_to_nonempty() {
            while (qid_ < 2 && it_ == mm_->get_queue(qid_).end()) {
                ++qid_;
                if (qid_ < 2) it_ = mm_->get_queue(qid_).begin();
            }
        }
        const mm_tiny_lfu* mm_;
        uint8_t qid_;
        typename item_list::const_iterator it_;
    };

    // Const overload of get_queue for const_iterator
    const item_list& get_queue(uint8_t qid) const {
        return (qid == kMainQueue) ? main_queue_ : tiny_queue_;
    }

    const_iterator begin() const {
        for (uint8_t q = 0; q < 2; ++q) {
            auto& qref = (q == kMainQueue) ? main_queue_ : tiny_queue_;
            if (!qref.empty()) return const_iterator(this, q, qref.begin());
        }
        return end();
    }
    const_iterator end() const { return const_iterator(); }

    // --------------------------------------------------------------------
    // Equality comparison (content and order must match)
    // --------------------------------------------------------------------

    friend bool operator==(const mm_tiny_lfu& a, const mm_tiny_lfu& b) {
        if (a.size() != b.size()) return false;
        auto ai = a.begin(), bi = b.begin();
        for (; ai != a.end(); ++ai, ++bi) {
            if (ai->key != bi->key || ai->value != bi->value) return false;
        }
        return true;
    }
    friend bool operator!=(const mm_tiny_lfu& a, const mm_tiny_lfu& b) { return !(a == b); }

    // --------------------------------------------------------------------
    // Stream output
    // --------------------------------------------------------------------

    friend std::ostream& operator<<(std::ostream& os, const mm_tiny_lfu& c) {
        os << "mm_tiny_lfu @" << &c << "  " << c.stats_ << "\n";
        std::size_t idx = 0;
        for (auto it = c.begin(); it != c.end(); ++it, ++idx) {
            os << "  [" << idx << "] key='";
            if constexpr (lru::detail::is_formattable_v<Key>) {
                os << std::format("{}", it->key);
            } else {
                os << std::format("<key at {:#x}>", reinterpret_cast<std::uintptr_t>(&it->key));
            }
            os << "' value='";
            if constexpr (lru::detail::is_formattable_v<Value>) {
                os << std::format("{}", it->value);
            } else {
                os << std::format("<val at {:#x}>", reinterpret_cast<std::uintptr_t>(&it->value));
            }
            os << "'\n";
        }
        return os;
    }

    // --------------------------------------------------------------------
    // Eviction (public for pooled_cache / unified_cache::evict())
    // --------------------------------------------------------------------

    /// A8: 频率感知淘汰——比较 Tiny tail 与 Main tail 频率，淘汰低频者。
    /// 对齐 CacheLib MMTinyLFU.h:488-500。
    void evict() {
        cleanup_pending_deletion();
        assert(!tiny_queue_.empty() || !main_queue_.empty());

        // B15: 用 predicate 辅助查找
        auto* tiny_victim = find_eviction_victim_in_queue(tiny_queue_);
        auto* main_victim = find_eviction_victim_in_queue(main_queue_);

        // 当两个队列均非空时，比较频率决定淘汰对象
        if (tiny_victim && main_victim) {
            auto tiny_freq = sketch_.estimate(tiny_victim->key);
            auto main_freq = sketch_.estimate(main_victim->key);

            bool admit = config_.newcomer_wins_on_tie
                             ? (tiny_freq >= main_freq)
                             : (tiny_freq > main_freq);
            if (admit) {
                evict_generic(main_victim, main_queue_);
            } else {
                evict_generic(tiny_victim, tiny_queue_);
            }
            return;
        }

        // 单队列非空时直接淘汰
        if (tiny_victim) { evict_generic(tiny_victim, tiny_queue_); return; }
        if (main_victim) { evict_generic(main_victim, main_queue_); return; }
    }

    item_ptr find_eviction_victim_in_queue(item_list& queue) {
        auto* curr = queue.tail();
        if (!curr) return curr;

        const bool has_pred = static_cast<bool>(eviction_predicate_);
        size_t tries = 0;
        while (curr) {
            stats_.eviction_search_steps.fetch_add(1, std::memory_order_relaxed);
            if (curr->has_active_handle()) {
                stats_.pinned_skip_count.fetch_add(1, std::memory_order_relaxed);
                curr = static_cast<item_type*>(queue.get_prev(*curr));
                continue;
            }
            if (has_pred && !eviction_predicate_(curr->key, curr->value)) {
                curr = static_cast<item_type*>(queue.get_prev(*curr));
                if (++tries >= config_.eviction_search_tries) break;
                continue;
            }
            return curr;
        }
        return nullptr;
    }

protected:
    item_list tiny_queue_;   // Queue 0: Window cache (new items)
    item_list main_queue_;   // Queue 1: Main cache (frequency-accepted)
    map_type map_;
    sketch_type sketch_;

    size_type max_size_ = unlimited;
    size_type max_memory_ = unlimited;
    mm_tiny_lfu_config config_;
    // A5: try_lock_update 优化使用的独立内部锁，与统一缓存层锁解耦
    // B10: 缓存行对齐以避免 false sharing（对齐 CacheLib MMLru.h:474）
    struct alignas(64) aligned_mutex_t { std::mutex m; };
    mutable aligned_mutex_t update_mutex_;
    uint32_t lru_refresh_time_ = 0;
    uint32_t next_reconfigure_time_ = std::numeric_limits<uint32_t>::max();
    // B15: EvictionPredicate
    std::function<bool(const Key&, const Value&)> eviction_predicate_;
    // B7: LockedIterator 活跃标记
    std::atomic<bool> iterator_active_{false};

    // Items removed by force_del() that still have active handles.
    std::vector<item_ptr> pending_deletion_;

    /// Clean up pending-deletion items whose handles have all been released.
    /// Fires the deferred eviction callback (with value move) before deleting.
    void cleanup_pending_deletion() {
        auto it = pending_deletion_.begin();
        while (it != pending_deletion_.end()) {
            if (!(*it)->has_active_handle()) {
                auto* item = *it;
                callbacks_.collect_evict(item->key, std::move(item->value));
                // P1-5: Route through hazptr retire instead of raw delete —
                // a concurrent hazptr-protected reader may still hold a
                // hazard pointer to this item even with refcount=0.
                detail::hazptr_domain::default_domain().retire(item);
                it = pending_deletion_.erase(it);
            } else {
                ++it;
            }
        }
    }

    mutable stats_type stats_;
    mutable callback_mgr callbacks_;

    std::function<size_type(const Key&)> key_size_fn_;
    std::function<size_type(const Value&)> value_size_fn_;

    // --------------------------------------------------------------------
    // Time utilities
    // --------------------------------------------------------------------

    static uint32_t current_time_sec() {
        return detail::cached_epoch_sec();
    }

    // ====================================================================
    // Adaptive Refresh Time (CacheLib's reconfigureLocked)
    // ====================================================================

    /// B1: 基于 Main 队列尾部年龄动态调整 lru_refresh_time_。
    void reconfigure_locked(uint32_t curr_time) {
        if (curr_time < next_reconfigure_time_) return;
        if (config_.mm_reconfigure_interval_secs == 0) return;

        next_reconfigure_time_ = curr_time + config_.mm_reconfigure_interval_secs;

        auto* tail = main_queue_.tail();
        uint32_t tail_age = tail ? (curr_time - tail->hook.update_time) : 0;
        auto new_refresh = std::min(
            std::max(config_.default_lru_refresh_time,
                     static_cast<uint32_t>(static_cast<double>(tail_age) *
                                           config_.lru_refresh_ratio)),
            config_type::k_lru_refresh_time_cap);
        lru_refresh_time_ = new_refresh;
    }

    /// B8: 获取淘汰年龄统计。
    struct eviction_age_stat {
        uint64_t oldest_element_age{0};
        uint64_t projected_age{0};
        uint64_t queue_size{0};
    };

    eviction_age_stat get_eviction_age_stat(std::size_t projected_length = 0) const noexcept {
        eviction_age_stat stat;
        stat.queue_size = main_queue_.size();
        const auto curr_time = current_time_sec();
        const auto* node = main_queue_.tail();
        const auto* current_queue = static_cast<const item_list*>(&main_queue_);
        // main 队列空时退化为 tiny 队列
        if (!node) {
            node = tiny_queue_.tail();
            current_queue = &tiny_queue_;
        }
        stat.oldest_element_age = node ? (curr_time - node->hook.update_time) : 0;
        for (std::size_t seen = 0; seen < projected_length && node != nullptr; ++seen) {
            node = static_cast<const item_type*>(current_queue->get_prev(*node));
        }
        stat.projected_age = node ? (curr_time - node->hook.update_time) : stat.oldest_element_age;
        return stat;
    }

    // --------------------------------------------------------------------
    // Internal helpers
    // --------------------------------------------------------------------

    item_list& get_queue(uint8_t qid) {
        return (qid == kMainQueue) ? main_queue_ : tiny_queue_;
    }

    size_type total_size() const {
        return tiny_queue_.size() + main_queue_.size();
    }

    bool should_evict() const {
        if (max_size_ != unlimited && size() > max_size_) return true;
        if (max_memory_ != unlimited && current_memory() > max_memory_) return true;
        return false;
    }

    size_type calc_item_memory(const Key& key, const Value& value) const {
        size_type mem = item_overhead;
        if (key_size_fn_) mem += key_size_fn_(key) * 2;
        if (value_size_fn_) mem += value_size_fn_(value);
        return mem;
    }

    size_type expected_tiny_size() const {
        auto total = max_size_ != unlimited ? max_size_ : size();
        return std::max(size_type(1), static_cast<size_type>(total * config_.window_to_cache_size_ratio));
    }

    void remove_from_queue(item_ptr item) {
        auto& queue = get_queue(item->queue_id);
        // A2: 清除 accessed 标志，对齐 CacheLib MMLru.h:744
        item->hook.clear_accessed();
        queue.remove(*item);
    }

    // ====================================================================
    // Delayed Promotion (CacheLib's recordAccess)
    // ====================================================================

    /// Record access to an item, with delayed promotion support.
    /// Returns true if the item was actually promoted (moved to head).
    bool record_access(item_ptr item, access_mode mode) {
        assert(item != nullptr);
        // Check updateOnWrite/updateOnRead
        if ((mode == access_mode::write && !config_.update_on_write) ||
            (mode == access_mode::read && !config_.update_on_read)) {
            return false;
        }

        auto curr = current_time_sec();

        // D3+A3: 对齐 CacheLib——首次访问(is_accessed==false)必提升；
        // 已访问节点仅在超 refresh time 后提升。移除 update_time>0 守卫。
        if (!item->hook.is_accessed()) {
            item->hook.set_accessed();
        } else if (curr < item->hook.update_time + lru_refresh_time_) {
            return false;  // Not enough time since last promotion
        }

        // A5: try_lock_update 优化——若启用，则尝试加锁；失败则返回 false 跳过提升（不阻塞）。
        // 对齐 CacheLib MMLru.h:567-577。成功获取锁后正常执行提升逻辑。
        auto promote = [this, item, curr]() {
            // B1: 在 promote 路径上定期调整 lru_refresh_time_
            reconfigure_locked(curr);

            // Move to head in the current queue
            auto& queue = get_queue(item->queue_id);
            queue.move_to_head(*item);
            item->hook.update_time = curr;

            // 提升成功时更新频率计数（统一入口，对齐 MMTinyLFU.h:771 updateFrequenciesLocked）。
            // 读/写访问路径均由此更新 CMS，避免上层 get/set 重复记录。
            sketch_.record(item->key);
        };

        if (try_lock_update_enabled()) {
            std::unique_lock<std::mutex> lock(update_mutex_.m, std::try_to_lock);
            if (!lock) {
                stats_.try_lock_fail_count.fetch_add(1, std::memory_order_relaxed);
                return false;
            }
            promote();
        } else {
            promote();
        }

        return true;
    }

    /// Record access with a pre-computed current time.
    bool record_access_at(item_ptr item, access_mode mode, uint32_t curr) {
        assert(item != nullptr);
        if ((mode == access_mode::write && !config_.update_on_write) ||
            (mode == access_mode::read && !config_.update_on_read)) {
            return false;
        }

        if (!item->hook.is_accessed()) {
            item->hook.set_accessed();
        } else if (curr < item->hook.update_time + lru_refresh_time_) {
            return false;
        }

        auto promote = [this, item, curr]() {
            reconfigure_locked(curr);
            auto& queue = get_queue(item->queue_id);
            queue.move_to_head(*item);
            item->hook.update_time = curr;
            sketch_.record(item->key);
        };

        if (try_lock_update_enabled()) {
            std::unique_lock<std::mutex> lock(update_mutex_.m, std::try_to_lock);
            if (!lock) {
                stats_.try_lock_fail_count.fetch_add(1, std::memory_order_relaxed);
                return false;
            }
            promote();
        } else {
            promote();
        }

        return true;
    }

    // ====================================================================
    // Insert / Update / Evict
    // ====================================================================

    template <typename V>
    void insert_new(const Key& key, V&& value) {
        if (max_size_ == 0) return;
        while (should_evict()) {
            auto old_size = size();
            evict();
            if (size() == old_size) break;
        }
        if (max_size_ != unlimited && size() >= max_size_) evict();

        auto mem = calc_item_memory(key, value);
        if (max_memory_ != unlimited) {
            while (!map_.empty() && max_memory_ - stats_.current_memory.load() < mem) {
                auto old_size = size();
                evict();
                if (size() == old_size) break;
            }
        }

        // New items always enter Tiny queue
        auto* item = this->allocate_item(key, std::forward<V>(value));
        assert(item != nullptr);
        auto curr = current_time_sec();
        item->hook.update_time = curr;
        item->hook.clear_accessed();  // Not yet accessed
        item->queue_id = kTinyQueue;

        tiny_queue_.link_at_head(*item);
        item->refcount.markInMMContainer();
        map_.insert(key, item);

        sketch_.record(key);
        stats_.current_size.store(total_size());
        stats_.current_memory.fetch_add(mem);
        stats_.register_insertion();
        callbacks_.collect_insert(key, item->value);

        maybe_promote_from_tiny();

        // 随缓存增长自动扩容 CMS，避免频率估计失真。
        // 对齐 MMTinyLFU.h 的 add() 末尾扩容语义。
        sketch_.maybe_grow_access_counters();
    }

    template <typename V>
    void update_existing(item_ptr item, V&& value, access_mode mode) {
        auto curr = current_time_sec();
        size_type old_mem = calc_item_memory(item->key, item->value);
        // P-MED-2 (T-H4): Strong exception guarantee via copy-then-swap.
        if constexpr (std::is_nothrow_swappable_v<Value> &&
                      std::is_constructible_v<Value, V>) {
            Value tmp(std::forward<V>(value));
            using std::swap;
            swap(item->value, tmp);
        } else {
            item->value = std::forward<V>(value);
        }
        // 频率计数由 record_access 统一更新，此处不再重复记录
        record_access_at(item, mode, curr);
        size_type new_mem = calc_item_memory(item->key, item->value);
        if (new_mem > old_mem) {
            stats_.current_memory.fetch_add(new_mem - old_mem);
        } else if (new_mem < old_mem) {
            stats_.current_memory.fetch_sub(old_mem - new_mem);
        }
        if (should_evict()) {
            shrink_to_fit();
        }
        // O7: Fire on_update for value changes on existing keys (distinct
        // from on_insert, which fires only for new key insertions).
        callbacks_.collect_update(item->key, item->value);
    }

    /// A9 修正: 对齐 CacheLib MMTinyLFU.h:866-878。
    /// Tiny 超容时无条件晋升 Tiny tail 到 Main head，不淘汰。
    /// 频率比较仅在 evict() 路径中进行（对齐 LockedIterator::evictTiny 语义）。
    void maybe_promote_from_tiny() {
        auto expected_tiny = expected_tiny_size();
        while (tiny_queue_.size() > expected_tiny) {
            auto old_size = tiny_queue_.size();
            auto* tiny_tail = tiny_queue_.tail();
            if (!tiny_tail) break;

            // Unconditional promotion: move Tiny tail to Main head
            // (CacheLib: maybePromoteTailLocked in add() always promotes)
            auto main_capacity = (max_size_ != unlimited)
                ? (max_size_ > expected_tiny ? max_size_ - expected_tiny : 0)
                : unlimited;
            if (main_capacity != unlimited && main_queue_.size() >= main_capacity) {
                evict_from_main();
            }
            promote_tiny_tail_to_main();
            if (tiny_queue_.size() == old_size) break;  // No progress, exit
        }
    }

    void promote_tiny_tail_to_main() {
        auto* item = tiny_queue_.tail();
        if (!item) return;
        // Move from Tiny tail to Main head
        item->queue_id = kMainQueue;
        tiny_queue_.remove(*item);
        main_queue_.link_at_head(*item);
    }

    /// 从指定队列中淘汰一个通用节点。
    void evict_generic(item_ptr item, item_list& queue) {
        auto evict_result = item->refcount.markForEviction();
        if (evict_result != detail::MarkForEvictionResult::kSuccess) {
            return;  // Cannot evict — item has active handles or is already exclusive
        }
        const auto& key = item->key;
        size_type mem = calc_item_memory(key, item->value);
        stats_.current_memory.fetch_sub(mem);
        stats_.register_eviction();
        if (callbacks_.has_eviction_callbacks()) {
            Value value = std::move(item->value);
            callbacks_.collect_evict(key, std::move(value));
        }
        map_.erase(key);
        item->refcount.unmarkInMMContainer();
        item->refcount.unmarkForEviction();
        queue.remove(*item);
        detail::hazptr_domain::default_domain().retire(item);
        stats_.current_size.store(total_size());
    }

    void evict_from_tiny() {
        auto* item = tiny_queue_.tail();
        // Skip items with active handles
        size_t tries = 0;
        while (item && item->has_active_handle()) {
            item = tiny_queue_.get_prev(*item);
            if (++tries >= config_.eviction_search_tries) { item = nullptr; break; }
        }
        if (!item) return;
        evict_generic(item, tiny_queue_);
    }

    void evict_from_main() {
        auto* item = main_queue_.tail();
        // Skip items with active handles
        size_t tries = 0;
        while (item && item->has_active_handle()) {
            item = main_queue_.get_prev(*item);
            if (++tries >= config_.eviction_search_tries) { item = nullptr; break; }
        }
        if (!item) return;
        evict_generic(item, main_queue_);
    }

    /// B14: 原位替换节点，保留 queue_id、update_time 与 accessed 状态。
    void replace_node(item_ptr old_node, item_ptr new_node) {
        assert(old_node != nullptr && new_node != nullptr);
        new_node->hook.update_time = old_node->hook.update_time;
        if (old_node->hook.is_accessed()) {
            new_node->hook.set_accessed();
        } else {
            new_node->hook.clear_accessed();
        }
        new_node->queue_id = old_node->queue_id;
        auto& queue = get_queue(old_node->queue_id);
        queue.replace(*old_node, *new_node);
        map_.insert_or_assign(old_node->key, new_node);
    }

    /// B15: 设置淘汰谓词。
    void set_eviction_predicate(std::function<bool(const Key&, const Value&)> pred) {
        eviction_predicate_ = std::move(pred);
    }

    void erase_impl(const Key& key) {
        auto ptr = map_.find(key);
        if (!ptr) return;
        auto* item = ptr;
        if (item->has_active_handle()) return;
        size_type mem = calc_item_memory(item->key, item->value);
        stats_.current_memory.fetch_sub(mem);
        callbacks_.collect_evict(item->key, std::move(item->value));
        remove_from_queue(item);
        map_.erase(key);
        detail::hazptr_domain::default_domain().retire(item);
        stats_.current_size.store(total_size());
    }

    // ====================================================================
    // S0: Faithful serialization rebuild
    // ====================================================================
public:

    template <typename InputIt>
    void rebuild_from_serialized(InputIt first, InputIt last) {
        flush();
        for (auto it = first; it != last; ++it) {
            auto* item = this->allocate_item(it->key, it->value);
            item->hook.update_time = it->update_time;
            if (it->flags & detail::intrusive_hook::kAccessedFlag) {
                item->hook.set_accessed();
            }
            item->queue_id = (it->queue_id <= kMainQueue) ? it->queue_id : kTinyQueue;
            auto& queue = (item->queue_id == kMainQueue) ? main_queue_ : tiny_queue_;
            queue.link_at_tail(*item);
            item->refcount.markInMMContainer();
            map_.insert(item->key, item);
            stats_.current_memory.fetch_add(calc_item_memory(item->key, item->value));
        }
        stats_.current_size.store(total_size());
    }
};



// ============================================================================
// W-TinyLFU Strategy - mm_wtiny_lfu
// ============================================================================


// ============================================================================
// W-TinyLFU Strategy Configuration
// ============================================================================

struct mm_wtiny_lfu_config {
    /// Ratio of Tiny (window) cache to total cache (default 1%)
    double window_to_cache_size_ratio = 0.01;

    /// Ratio of Protection to Main cache (default 80%)
    double protection_ratio = 0.8;

    /// Minimum frequency to be promoted from Probation to Protection (default 3)
    uint32_t protection_freq = 3;

    /// CountMinSketch error rate
    double cms_error_rate = 0.5;

    /// CountMinSketch confidence
    double cms_confidence = 0.99;

    /// Default time between promotions for the same item (seconds).
    /// 0 = no delay (traditional behavior).
    uint32_t default_lru_refresh_time = 60;

    /// Whether to promote the item on write access.
    /// W-TinyLFU defaults to TRUE (unlike the other MM strategies), aligning
    /// with CacheLib MMWTinyLFU's default behavior: writes are accesses too.
    /// The W-TinyLFU admission policy compares Tiny-tail (newcomer) frequency
    /// against Probation-tail frequency; if writes do not bump the frequency
    /// sketch or refresh LRU position, frequently-updated hot keys can be
    /// evicted by newcomers with zero access history (test_concurrent_mm_strategies
    /// MmWTinyLfuMixedInsertUpdate exercises exactly this scenario).
    /// Performance impact is mitigated by try_lock_update (default true): the
    /// write path uses try_lock on update_mutex_ instead of blocking, so
    /// contention under high write concurrency stays bounded.
    bool update_on_write = true;

    /// Whether to promote the item on read access.
    bool update_on_read = true;

    /// A2: Tie-breaking policy for Tiny vs Probation admission.
    /// Aligns with CacheLib MMWTinyLFU.h:685-693 admitToProbation.
    /// - true (default): newcomer (Tiny tail) wins on tie, uses >= comparison.
    /// - false: existing Probation tail is retained on tie, uses > comparison.
    bool newcomer_wins_on_tie = true;

    /// A5: 是否在 record_access 中使用 try_to_lock 跳过提升。
    /// 对齐 CacheLib MMLru.h:567-577 的 tryLockUpdate。
    /// 默认 true，因为 unified_cache 已提供外层并发保护，
    /// 阻塞等待 update_mutex_ 会造成双重锁开销。
    bool try_lock_update = true;

    /// Use combined lock for eviction iterators.
    bool use_combined_lock_for_iterators = false;

    /// B15: 淘汰搜索次数上限。
    size_t eviction_search_tries = 3;

    /// Ratio for adaptive refresh time adjustment.
    double lru_refresh_ratio = 0.0;

    /// Interval in seconds for reconfiguring the adaptive refresh time.
    /// 0 = disabled (default).
    uint32_t mm_reconfigure_interval_secs = 0;

    /// Expected number of items for automatic bucket count sizing.
    /// 0 = use default bucket count (1024). When > 0, the internal hash
    /// table is pre-sized via concurrent_hash_table::buckets_for_items()
    /// to keep average chain length ≤ 0.25 at the expected load.
    size_t expected_items = 0;

    /// Custom node allocation function for non-EmbeddedChain hash table nodes.
    /// nullptr (default) = standard new/delete allocation.
    void* (*alloc_fn)(std::size_t) = nullptr;

    /// Custom node deallocation function (must pair with alloc_fn).
    void  (*dealloc_fn)(void*) = nullptr;

    /// In read-heavy mode, skip the try_acquire_write_lock_for_key() step in
    /// get() and always defer LRU promotion to the TLS access ring.
    /// Default true — production-recommended for lower read-path latency.
    bool defer_promotion = true;

    // Max lruRefreshTime cap
    static constexpr uint32_t k_lru_refresh_time_cap = 900;

    // B4: 配置校验——各 ratio/rate 范围合法
    mm_wtiny_lfu_config() noexcept = default;

    void validate() const {
        if (window_to_cache_size_ratio <= 0.0 || window_to_cache_size_ratio > 0.5) {
            throw std::invalid_argument(
                "mm_wtiny_lfu_config: window_to_cache_size_ratio must be in (0, 0.5]");
        }
        if (protection_ratio < 0.0 || protection_ratio > 1.0) {
            throw std::invalid_argument(
                "mm_wtiny_lfu_config: protection_ratio must be in [0, 1]");
        }
        if (cms_error_rate <= 0.0 || cms_error_rate >= 1.0) {
            throw std::invalid_argument(
                "mm_wtiny_lfu_config: cms_error_rate must be in (0, 1)");
        }
        if (cms_confidence <= 0.0 || cms_confidence >= 1.0) {
            throw std::invalid_argument(
                "mm_wtiny_lfu_config: cms_confidence must be in (0, 1)");
        }
        if (!(lru_refresh_ratio >= 0.0)) {
            throw std::invalid_argument(
                "mm_wtiny_lfu_config: lru_refresh_ratio must be non-negative");
        }
    }
};

// ============================================================================
// W-TinyLFU Strategy
// ============================================================================

/// W-TinyLFU eviction strategy with segmented Main cache.
/// Extends TinyLFU by splitting Main into Probation and Protection:
/// - New items enter Tiny (window)
/// - Frequency-accepted items enter Probation
/// - High-frequency Probation items get promoted to Protection
/// - When Protection overflows, tail degrades to Probation TAIL (not head)
///   This preserves the item's survival chance.
template <
    typename Key,
    typename Value,
    typename Hash = std::hash<Key>,
    typename KeyEqual = std::equal_to<Key>,
    typename ProbingStyle = detail::chain_probing_tag,
    bool Segmented = false
>
/// A4: 线程安全契约——此类非线程安全，调用方必须确保在外层 unified_cache 锁内访问。
/// 内部 update_mutex_ 仅用于 try_lock_update 路径的解耦优化，不保证 MM 层独立线程安全。
class mm_wtiny_lfu : public detail::mm_allocator_mixin<detail::cache_item<Key, Value>> {
public:
    using key_type = Key;
    using mapped_type = Value;
    using size_type = std::size_t;
    using config_type = mm_wtiny_lfu_config;

    // Item type: cache_item with embedded intrusive hook
    using item_type = detail::cache_item<Key, Value>;
    using item_ptr = item_type*;

    // Intrusive list type
    using item_list = detail::intrusive_list<item_type, detail::intrusive_hook, detail::default_get_hook<item_type>>;

    // Map: Key -> item pointer
    using map_type = std::conditional_t<
        Segmented,
        detail::segmented_concurrent_hash_table<Key, item_ptr, Hash, KeyEqual, true, ProbingStyle, 64>,
        detail::concurrent_hash_table<Key, item_ptr, Hash, KeyEqual, true, ProbingStyle>
    >;

    using callback_mgr = callback_manager<Key, Value>;
    using stats_type = cache_stats;
    using sketch_type = detail::count_min_sketch<Key, Hash>;

    static constexpr size_type npos = unlimited;
    // Item overhead: hook + key + value + handle + map entry
    static constexpr size_type item_overhead = sizeof(item_type) + map_type::entry_overhead;

    // Queue IDs
    static constexpr uint8_t kTinyQueue = 0;
    static constexpr uint8_t kProbationQueue = 1;
    static constexpr uint8_t kProtectionQueue = 2;

    // --------------------------------------------------------------------
    // Constructors / Destructor
    // --------------------------------------------------------------------

    mm_wtiny_lfu() : mm_wtiny_lfu(mm_wtiny_lfu_config{}) {}

    explicit mm_wtiny_lfu(const mm_wtiny_lfu_config& config)
        : config_(config)
        , map_(config.expected_items > 0
            ? map_type::buckets_for_items(config.expected_items)
            : 1024,
            config.alloc_fn, config.dealloc_fn)
        , protection_freq_(config.protection_freq)
        , lru_refresh_time_(config.default_lru_refresh_time)
        , next_reconfigure_time_(config.mm_reconfigure_interval_secs == 0
            ? std::numeric_limits<uint32_t>::max()
            : current_time_sec() + config.mm_reconfigure_interval_secs)
        , sketch_(1000, config.cms_error_rate, config.cms_confidence) {}

    mm_wtiny_lfu(size_type max_size, const mm_wtiny_lfu_config& config = mm_wtiny_lfu_config{})
        : mm_wtiny_lfu(config) {
        detail::validate_capacity(max_size, unlimited);
        max_size_ = max_size;
        stats_.max_size.store(max_size);
        sketch_.set_max_window_size(std::max(max_size, size_type(100)));
    }

    mm_wtiny_lfu(size_type max_size, size_type max_memory,
                  const mm_wtiny_lfu_config& config = mm_wtiny_lfu_config{})
        : mm_wtiny_lfu(config) {
        detail::validate_capacity(max_size, max_memory);
        max_size_ = max_size;
        max_memory_ = max_memory;
        stats_.max_size.store(max_size);
        stats_.max_memory.store(max_memory);
        sketch_.set_max_window_size(std::max(max_size, size_type(100)));
    }

    ~mm_wtiny_lfu() {
        // Clean up pending deletions first (items with active handles that were force_del'd)
        for (auto* item : pending_deletion_) {
            delete item;
        }
        pending_deletion_.clear();
        // Use flush() to properly erase items from both map and list before deletion,
        // preventing use-after-free when the hash table destructor iterates its chains.
        flush();
    }

    // Non-copyable, non-movable (owns item pointers)
    mm_wtiny_lfu(const mm_wtiny_lfu&) = delete;
    mm_wtiny_lfu& operator=(const mm_wtiny_lfu&) = delete;

    // --------------------------------------------------------------------
    // Core cache API
    // --------------------------------------------------------------------

    template <typename V>
    void set(const Key& key, V&& value) {
        cleanup_pending_deletion();
        auto ptr = map_.find(key);
        if (ptr) {
            update_existing(ptr, std::forward<V>(value), access_mode::write);
        } else {
            insert_new(key, std::forward<V>(value));
        }
    }

    template <typename V>
    bool add(const Key& key, V&& value) {
        cleanup_pending_deletion();
        auto ptr = map_.find(key);
        if (ptr) {
            // B4: frequency update is now unified inside record_access.
            record_access(ptr, access_mode::read);
            return false;
        }
        insert_new(key, std::forward<V>(value));
        return true;
    }

    template <typename V>
    bool replace(const Key& key, V&& value) {
        auto ptr = map_.find(key);
        if (!ptr) return false;
        update_existing(ptr, std::forward<V>(value), access_mode::write);
        return true;
    }

    read_handle<Value> get(const Key& key) {
        auto ptr = map_.find(key);
        if (!ptr) {
            stats_.register_miss();
            callbacks_.collect_miss(key);
            return {};
        }

        auto* item = ptr;
        // B4: frequency update is now unified inside record_access.
        // B5: Probation->Protection promotion is now handled inside
        // record_access via try_promote_to_protection.
        record_access(item, access_mode::read);

        // Tiny promotion check (Tiny->Probation admission)
        if (item->queue_id == kTinyQueue) {
            maybe_promote_from_tiny();
        }

        stats_.register_hit();
        callbacks_.collect_hit(key, item->value);
        return read_handle<Value>{&item->value, &item->refcount};
    }

    read_handle<const Value> get(const Key& key) const {
        auto ptr = map_.find(key);
        if (!ptr) {
            stats_.register_miss();
            callbacks_.collect_miss(key);
            return {};
        }
        auto* item = ptr;
        stats_.register_hit();
        callbacks_.collect_hit(key, item->value);
        return read_handle<const Value>{&item->value, &item->refcount};
    }

    /// H0: Peek with handle — 不提升 LRU，返回 handle 防止持有期被淘汰。
    /// Uses find_and_pin_lockfree() to attempt lock-free pinning first
    /// (optimistic read + incRef without bucket lock), falling back to
    /// find_and_pin() (shared lock path) if the lock-free pin fails.
    /// No stripe-level read lock is needed.
    read_handle<const Value> peek(const Key& key) const {
        auto pin_fn = [](auto* item) {
            return item->refcount.incRef() == detail::IncResult::kIncOk;
        };
        auto ptr = map_.find_and_pin_lockfree(key, pin_fn);
        if (!ptr) return {};
        return read_handle<const Value>{&ptr->value, &ptr->refcount, pre_pinned};
    }

    /// Internal: peek that returns mutable handle (for optimistic get path).
    read_handle<Value> peek_for_get(const Key& key) {
        return peek_for_get_with_hash(key, Hash{}(key));
    }

    /// T16.4: peek_for_get with a pre-computed hash. The hash MUST be
    /// the result of Hash{}(key) — callers are responsible for hash
    /// compatibility. Used by bulk_get to avoid re-hashing each key
    /// for both shard dispatch and hash-table lookup.
    read_handle<Value> peek_for_get_with_hash(const Key& key, std::size_t hash) {
        auto pin_fn = [](auto* item) {
            return item->refcount.incRef() == detail::IncResult::kIncOk;
        };
        auto ptr = map_.find_and_pin_lockfree_with_hash(key, hash, pin_fn);
        if (!ptr) return {};
        // P1-1: Inline TTL check. Fast path: expiry_ns == 0 (no TTL) skips
        // steady_clock::now() entirely. Only items with a TTL set pay the
        // clock read, and only on cache hits. Expired items are unpinned
        // (decRef) and reported as a miss; the actual eviction is handled
        // lazily by evict_expired() / the background TTL cleaner, NOT here,
        // so we don't need a write lock on the hot path.
        if (ptr->expiry_ns != 0) {
            // P1-10: Track TTL check frequency on the read path.
            stats_.ttl_checked_count.fetch_add(1, std::memory_order_relaxed);
            auto now_ns = static_cast<std::uint64_t>(
                std::chrono::steady_clock::now().time_since_epoch().count());
            if (now_ns >= ptr->expiry_ns) {
                stats_.ttl_expired_count.fetch_add(1, std::memory_order_relaxed);
                ptr->refcount.decRef();
                stats_.register_miss();
                callbacks_.collect_miss(key);
                return {};
            }
        }
        return read_handle<Value>{&ptr->value, &ptr->refcount, pre_pinned};
    }

    bool del(const Key& key) {
        cleanup_pending_deletion();
        auto ptr = map_.find(key);
        if (!ptr) return false;
        if (ptr->has_active_handle()) return false;
        erase_impl(key);
        return true;
    }

    /// Force delete a key even if it has active handles.
    /// The item is immediately removed from the map and queue,
    /// but memory is not freed until all handles are released.
    /// Returns true if the key was found (and removed from the map).
    bool force_del(const Key& key) {
        cleanup_pending_deletion();
        auto ptr = map_.find(key);
        if (!ptr) return false;
        auto* item = ptr;
        size_type mem = calc_item_memory(item->key, item->value);

        map_.erase(key);
        // Capture item state BEFORE remove_from_queue() poisons the item's memory.
        bool has_handle = item->has_active_handle();
        if (!has_handle) {
            callbacks_.collect_evict(item->key, std::move(item->value));
        }
        remove_from_queue(item);

        stats_.current_memory.fetch_sub(mem);
        stats_.current_size.store(total_size());

        if (!has_handle) {
            detail::hazptr_domain::default_domain().retire(item);
        } else {
            pending_deletion_.push_back(item);
        }
        return true;
    }

    std::optional<Value> pop(const Key& key) {
        auto ptr = map_.find(key);
        if (!ptr) return std::nullopt;
        auto* item = ptr;
        if (item->has_active_handle()) return std::nullopt;
        size_type mem = calc_item_memory(item->key, item->value);
        Value value = std::move(item->value);
        stats_.current_memory.fetch_sub(mem);
        remove_from_queue(item);
        map_.erase(key);
        detail::hazptr_domain::default_domain().retire(item);
        stats_.current_size.store(total_size());
        return value;
    }

    std::optional<std::pair<Key, Value>> pop_lru() {
        cleanup_pending_deletion();
        // Priority: Tiny tail > Probation tail; Protection is never popped directly.
        auto find_unpinned_tail = [this](item_list& queue) -> item_ptr {
            auto* tail = queue.tail();
            size_t tries = 0;
            while (tail && tail->has_active_handle()) {
                tail = static_cast<item_type*>(queue.get_prev(*tail));
                if (++tries >= config_.eviction_search_tries) { tail = nullptr; break; }
            }
            return tail;
        };
        auto* tail = find_unpinned_tail(tiny_queue_);
        if (!tail) tail = find_unpinned_tail(probation_queue_);
        if (!tail) return std::nullopt;
        if (!map_.contains(tail->key)) return std::nullopt;
        std::pair<Key, Value> result{std::move(tail->key), std::move(tail->value)};
        size_type mem = calc_item_memory(result.first, result.second);
        stats_.current_memory.fetch_sub(mem);
        remove_from_queue(tail);
        map_.erase(result.first);
        detail::hazptr_domain::default_domain().retire(tail);
        stats_.current_size.store(total_size());
        return result;
    }

    /// Flush the cache. Items pinned by an active read_handle are left in place.
    ///
    /// P1-8 (T2.6 bugfix): Two-pass deferred retirement — see mm_lru::flush()
    /// for the full rationale. Bypasses erase_impl() to apply markForEviction()
    /// directly on the iterated item. All retirements are deferred to Pass 2
    /// after all three queues have been traversed.
    void flush() {
        // Pass 1: iterate all queues, collect items to retire.
        std::vector<item_ptr> to_retire;
        auto flush_queue = [this, &to_retire](item_list& queue) {
            auto* curr = queue.head();
            while (curr) {
                // Save next before any mutation. remove() only clears curr's hook
                // pointers; the next node remains valid and linked.
                auto* next = queue.get_next(*curr);
                if (!curr->has_active_handle()) {
                    auto evict_result = curr->refcount.markForEviction();
                    if (evict_result != detail::MarkForEvictionResult::kSuccess) {
                        curr = next;
                        continue;
                    }
                    size_type mem = calc_item_memory(curr->key, curr->value);
                    stats_.current_memory.fetch_sub(mem);
                    callbacks_.collect_evict(curr->key, std::move(curr->value));
                    // Item may or may not be in the map (transient queue state).
                    map_.erase(curr->key);
                    queue.remove(*curr);
                    curr->refcount.unmarkForEviction();
                    to_retire.push_back(curr);
                }
                curr = next;
            }
        };
        flush_queue(tiny_queue_);
        flush_queue(probation_queue_);
        flush_queue(protection_queue_);
        // P1-8 (T2.6 bugfix, phase 3): Refresh hash stats BEFORE retiring
        // any items — see mm_lru::flush() for the full rationale.
        refresh_hash_stats();
        // Pass 2: retire all collected items after iteration is complete.
        // mm_wtiny_lfu currently has no use_ebr config; always use hazptr.
        for (auto* item : to_retire) {
            detail::hazptr_domain::default_domain().retire(item);
        }
        sketch_.reset();
        stats_.current_size.store(total_size());
        cleanup_pending_deletion();
    }

    bool contains(const Key& key) const {
        return map_.contains(key);
    }

    // --------------------------------------------------------------------
    // Capacity
    // --------------------------------------------------------------------

    bool empty() const noexcept { return map_.empty(); }
    size_type size() const noexcept { return map_.size(); }
    size_type max_size() const noexcept { return max_size_; }
    size_type max_memory() const noexcept { return max_memory_; }
    size_type current_memory() const noexcept { return stats_.current_memory.load(); }

    /// Estimate memory that would be accounted for an item with the given key
    /// and value, including the fixed item overhead and any custom size
    /// calculators registered via set_key/value_size_calculator().
    size_type estimate_item_memory(const Key& key, const Value& value) const {
        return calc_item_memory(key, value);
    }
    size_type tiny_size() const noexcept { return tiny_queue_.size(); }
    size_type probation_size() const noexcept { return probation_queue_.size(); }
    size_type protection_size() const noexcept { return protection_queue_.size(); }

    /// Check if any item (in any queue or pending deletion) has an active handle.
    bool has_active_handles() const noexcept {
        for (auto it = tiny_queue_.begin(); it != tiny_queue_.end(); ++it) {
            if (it->has_active_handle()) return true;
        }
        for (auto it = probation_queue_.begin(); it != probation_queue_.end(); ++it) {
            if (it->has_active_handle()) return true;
        }
        for (auto it = protection_queue_.begin(); it != protection_queue_.end(); ++it) {
            if (it->has_active_handle()) return true;
        }
        for (auto* item : pending_deletion_) {
            if (item->has_active_handle()) return true;
        }
        return false;
    }

    void max_size(size_type new_max) {
        max_size_ = new_max;
        stats_.max_size.store(new_max);
        sketch_.set_max_window_size(std::max(new_max, size_type(100)));
        if (new_max != npos) shrink_to_fit();
    }

    void max_memory(size_type new_max) {
        max_memory_ = new_max;
        stats_.max_memory.store(new_max);
        if (new_max != npos) shrink_to_fit();
    }

    void shrink_to_fit() {
        while (should_evict()) {
            auto old_size = size();
            evict();
            if (size() == old_size) break;
        }
    }

    // --------------------------------------------------------------------
    // Statistics and callbacks
    // --------------------------------------------------------------------

    stats_type& stats() noexcept { return stats_; }
    const stats_type& stats() const noexcept { return stats_; }
    callback_mgr& callbacks() noexcept { return callbacks_; }
    const callback_mgr& callbacks() const noexcept { return callbacks_; }

    // P1-7: Number of items in pending-deletion state (removed from cache
    // but still pinned by active read_handles). Best-effort count — may
    // race with concurrent writes. For monitoring only.
    std::size_t pending_deletion_count() const noexcept {
        return pending_deletion_.size();
    }

    /// Refresh hash table diagnostic stats (load factor, max chain length).
    /// O(bucket_count) scan — call periodically, not on every operation.
    void refresh_hash_stats() const noexcept {
        stats_.hash_load_factor.store(map_.load_factor(), std::memory_order_relaxed);
        stats_.max_chain_length.store(map_.max_chain_length(), std::memory_order_relaxed);
        // P1-1: Refresh rehash diagnostics from the hash table.
        stats_.rehash_count.store(map_.rehash_count(), std::memory_order_relaxed);
        stats_.rehash_total_time_ns.store(map_.rehash_total_time_ns(), std::memory_order_relaxed);
        stats_.rehash_migrated_items.store(map_.rehash_migrated_items(), std::memory_order_relaxed);
        // T13.1: Refresh overload threshold and event counter from the
        // hash table. These mirror the live state in concurrent_hash_table.
        stats_.hash_overload_threshold.store(map_.hash_overload_threshold(), std::memory_order_relaxed);
        stats_.hash_overload_events.store(map_.hash_overload_events(), std::memory_order_relaxed);
    }

    /// T-B4 (P2-10): Forward to the underlying hash table's diagnostics
    /// cache refresh. Only segmented_concurrent_hash_table implements
    /// this (regular concurrent_hash_table doesn't cache — its
    /// `max_chain_length()` is already a single-table scan, cheap enough
    /// to not warrant caching). For non-segmented tables this is a no-op
    /// via `if constexpr` (zero-cost). The background rehash balancer
    /// invokes this unconditionally.
    void refresh_diagnostics_cache() const noexcept {
        if constexpr (requires { map_.refresh_diagnostics_cache(); }) {
            map_.refresh_diagnostics_cache();
        }
    }

    /// T-B4 (P2-10): Forward to the underlying hash table's age metric.
    /// Returns `std::numeric_limits<std::uint64_t>::max()` if the cache
    /// has never been refreshed or the underlying table doesn't cache.
    /// Operators should check the `segmented_hash_table` flag in
    /// diagnostics() before relying on this value — non-segmented tables
    /// always report max (no cache, so age is meaningless).
    std::uint64_t diagnostics_cache_age_ms() const noexcept {
        if constexpr (requires { map_.diagnostics_cache_age_ms(); }) {
            return map_.diagnostics_cache_age_ms();
        }
        return std::numeric_limits<std::uint64_t>::max();
    }

    // --------------------------------------------------------------------
    // Memory policy
    // --------------------------------------------------------------------

    void set_key_size_calculator(std::function<size_type(const Key&)> func) {
        key_size_fn_ = std::move(func);
    }

    void set_value_size_calculator(std::function<size_type(const Value&)> func) {
        value_size_fn_ = std::move(func);
    }

    // --------------------------------------------------------------------
    // W-TinyLFU-specific API
    // --------------------------------------------------------------------

    const sketch_type& sketch() const noexcept { return sketch_; }
    /// S3: 非 const CMS 访问（用于反序列化恢复 CMS 状态）。
    sketch_type& sketch_mut() noexcept { return sketch_; }

    const mm_wtiny_lfu_config& config() const noexcept { return config_; }

    /// Set custom hash table node allocation/deallocation functions.
    void set_hash_alloc_fns(void* (*alloc_fn)(std::size_t), void (*dealloc_fn)(void*)) {
        config_.alloc_fn = alloc_fn;
        config_.dealloc_fn = dealloc_fn;
        map_.set_alloc_fns(alloc_fn, dealloc_fn);
    }

    void set_config(const mm_wtiny_lfu_config& config) {
        config_ = config;
        map_.set_alloc_fns(config.alloc_fn, config.dealloc_fn);
        protection_freq_ = config.protection_freq;
        lru_refresh_time_ = config.default_lru_refresh_time;
        next_reconfigure_time_ = config.mm_reconfigure_interval_secs == 0
            ? std::numeric_limits<uint32_t>::max()
            : current_time_sec() + config.mm_reconfigure_interval_secs;
        maybe_promote_from_tiny();
    }

    /// Pre-allocate hash table buckets for `expected_items` entries.
    void reserve(size_type expected_items) {
        map_.reserve(expected_items);
    }

    /// Enable/disable incremental rehash for the hash table.
    /// When enabled, rehash migrates buckets incrementally across multiple
    /// operations instead of blocking all writers during a single rehash.
    /// This reduces write-path latency spikes under load.
    void set_incremental_rehash(bool enabled) {
        map_.set_incremental_rehash(enabled);
    }

    /// Query whether incremental rehash is enabled.
    bool incremental_rehash_enabled() const noexcept {
        return map_.incremental_rehash_enabled();
    }

    /// P0-5 (T1.3): Advance any in-progress incremental rehash by one
    /// per-call migration budget (kRehashFinishMaxBucketsPerCall).
    /// Called by the background rehash balancer to ensure stalled
    /// rehashes eventually complete without requiring writes to the
    /// affected hash table. No-op when no rehash is in progress.
    void advance_incremental_rehash() noexcept {
        map_.rehash_finish();
    }

    /// T11.5: String-based strategy setter (see concurrent_hash_table::set_rehash_strategy).
    bool set_rehash_strategy(std::string_view strategy) noexcept {
        return map_.set_rehash_strategy(strategy);
    }
    std::string_view rehash_strategy() const noexcept {
        return map_.rehash_strategy();
    }

    /// T11.3: Number of writes blocked by a non-incremental (blocking) rehash.
    std::size_t rehash_blocked_writes_count() const noexcept {
        return map_.rehash_blocked_writes_count();
    }

    /// P1-5: Number of times find_and_pin_lockfree fell back to the
    /// lock-protected path because the target segment was in incremental
    /// rehash. Non-zero values indicate the lock-free read path is being
    /// degraded by rehash activity.
    std::size_t rehash_lockfree_fallback_count() const noexcept {
        return map_.rehash_lockfree_fallback_count();
    }

    /// P0-D: Ratio of the hash table currently in an incremental rehash.
    /// For non-segmented tables: 0.0 or 1.0 (whole table rehashing or not).
    /// For segmented tables: fraction of segments currently rehashing.
    /// Exposed as a Prometheus gauge to detect sustained rehash pressure.
    float rehash_in_progress_ratio() const noexcept {
        return map_.rehash_in_progress_ratio();
    }

    /// T13.1: Set the hash table load factor overload threshold.
    /// See concurrent_hash_table::set_hash_overload_threshold.
    void set_hash_overload_threshold(float threshold) noexcept {
        map_.set_hash_overload_threshold(threshold);
    }

    float hash_overload_threshold() const noexcept {
        return map_.hash_overload_threshold();
    }

    std::size_t hash_overload_events() const noexcept {
        return map_.hash_overload_events();
    }

    /// T13.2: Register an overload callback on the underlying hash table.
    void set_overload_callback(std::function<void(float, float)> cb) {
        map_.set_overload_callback(std::move(cb));
    }

    /// P2-4 (T2.4): Toggle async mode for the overload callback.
    /// Forwarded to the underlying hash table. See
    /// `concurrent_hash_table::set_async_overload_callback` for semantics.
    void set_async_overload_callback(bool enabled) noexcept {
        map_.set_async_overload_callback(enabled);
    }

    /// P2-4 (T2.4): Drain pending overload events from the underlying
    /// hash table and dispatch the registered callback for each. Returns
    /// the number of events drained. Designed to be called from a
    /// background worker (e.g. the `event_drain_worker` in `unified_cache`).
    std::size_t drain_overload_callbacks() {
        return map_.drain_overload_callbacks();
    }

    /// Whether an incremental rehash is currently in progress.
    bool is_rehashing() const noexcept { return map_.is_rehashing(); }
    /// Buckets fully migrated so far during the in-progress rehash (0 if none).
    size_type rehash_progress() const noexcept { return map_.rehash_progress(); }
    /// New bucket count target for the in-progress rehash (0 if none).
    size_type rehash_new_bucket_count() const noexcept { return map_.rehash_new_bucket_count(); }
    /// Old bucket count for the in-progress rehash (0 if none).
    size_type rehash_old_bucket_count() const noexcept { return map_.rehash_old_bucket_count(); }
    /// Total number of hash table buckets currently allocated.
    size_type bucket_count() const noexcept { return map_.bucket_count(); }

    uint32_t refresh_time() const noexcept { return lru_refresh_time_; }

    /// A5: 返回 try_lock_update 配置，用于 record_access 内的 try_to_lock 优化
    bool try_lock_update_enabled() const noexcept { return config_.try_lock_update; }

    /// Promote an item by key without triggering hit statistics or callbacks.
    /// This is a side-effect-free alternative to get() for batch promotion
    /// (e.g., from TLS ring flush). Returns true if the key was found and
    /// the item was eligible for promotion.
    bool promote(const Key& key) {
        auto ptr = map_.find(key);
        if (!ptr) return false;
        return record_access(ptr, access_mode::read);
    }

    uint32_t protection_freq() const noexcept { return protection_freq_; }

    /// Serialization support: access all items via the internal map.
    const map_type& internal_map() const noexcept { return map_; }

    // --------------------------------------------------------------------
    // LockedIterator (B7)
    // --------------------------------------------------------------------

    /// B7: 持有锁的迭代器，遍历 Tiny→Probation→Protection。
    /// 使用 locked_iterator_guard 管理锁生命周期。
    class LockedIterator {
    public:
        LockedIterator(mm_wtiny_lfu& mm)
            : guard_(mm.update_mutex_.m, mm.iterator_active_), mm_(&mm) {
            init_queues();
        }

        ~LockedIterator() = default;
        LockedIterator(const LockedIterator&) = delete;
        LockedIterator& operator=(const LockedIterator&) = delete;
        LockedIterator(LockedIterator&& other) noexcept
            : guard_(std::move(other.guard_)), mm_(other.mm_),
              qid_(other.qid_), curr_(other.curr_) {}

        void destroy() { guard_.destroy(); }

        void resetToBegin() {
            qid_ = 0;
            init_queues();
        }

        bool next() {
            auto* next = mm_->get_queue(qid_).get_prev(*curr_);
            if (next) { curr_ = next; return true; }
            for (uint8_t q = qid_ + 1; q < 3; ++q) {
                auto& q_ref = mm_->get_queue(q);
                if (!q_ref.empty()) { qid_ = q; curr_ = q_ref.tail(); return true; }
            }
            return false;
        }

        item_ptr operator->() { return curr_; }
        explicit operator bool() const { return curr_ != nullptr; }

    private:
        void init_queues() {
            curr_ = nullptr;
            for (uint8_t q = 0; q < 3; ++q) {
                auto& q_ref = mm_->get_queue(q);
                if (!q_ref.empty()) { qid_ = q; curr_ = q_ref.tail(); return; }
            }
        }

        detail::locked_iterator_guard<> guard_;
        mm_wtiny_lfu* mm_;
        item_ptr curr_ = nullptr;
        uint8_t qid_ = 0;
    };

    // --------------------------------------------------------------------
    // UnifiedIterator (B6) — 淘汰优先级顺序遍历
    // --------------------------------------------------------------------

    /// B6: 跨队列统一迭代器，按 Protection→Probation→Tiny 顺序遍历，自动跳过空队列。
    /// 对齐 CacheLib MultiDList.h:98-178 的跨队列统一迭代器设计。
    class UnifiedIterator {
    public:
        // Eviction order: Protection(2) → Probation(1) → Tiny(0)
        static constexpr uint8_t kEvictionOrder[] = {
            kProtectionQueue, kProbationQueue, kTinyQueue
        };
        static constexpr uint8_t kEvictionOrderSize = 3;

        UnifiedIterator(mm_wtiny_lfu& mm) : mm_(&mm) {
            step_ = 0;
            init_from_step();
        }

        bool next() {
            auto* next = mm_->get_queue(qid_).get_prev(*curr_);
            if (next) { curr_ = next; return true; }
            // Current queue exhausted, advance to next in eviction order
            for (++step_; step_ < kEvictionOrderSize; ++step_) {
                uint8_t q = kEvictionOrder[step_];
                auto& queue = mm_->get_queue(q);
                if (queue.tail()) {
                    qid_ = q;
                    curr_ = queue.tail();
                    return true;
                }
            }
            return false;
        }

        item_ptr operator->() { return curr_; }
        explicit operator bool() const { return curr_ != nullptr; }

        /// Current queue ID (useful for caller to know which queue the item is in)
        uint8_t queue_id() const { return qid_; }

    private:
        void init_from_step() {
            curr_ = nullptr;
            for (; step_ < kEvictionOrderSize; ++step_) {
                uint8_t q = kEvictionOrder[step_];
                auto& q_ref = mm_->get_queue(q);
                if (!q_ref.empty()) { qid_ = q; curr_ = q_ref.tail(); return; }
            }
        }

        mm_wtiny_lfu* mm_;
        item_ptr curr_ = nullptr;
        uint8_t qid_ = 0;
        uint8_t step_ = 0;
    };

    // --------------------------------------------------------------------
    // Const iterator: traverses Tiny→Probation→Protection, MRU→LRU per queue
    // --------------------------------------------------------------------

    class const_iterator {
    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = const item_type;
        using reference = const value_type&;
        using pointer = value_type*;
        using difference_type = std::ptrdiff_t;

        const_iterator() : mm_(nullptr), qid_(3) {}

        const_iterator(const mm_wtiny_lfu* mm, uint8_t qid, typename item_list::const_iterator it)
            : mm_(mm), qid_(qid), it_(it) {}

        reference operator*() const { return *it_; }
        pointer operator->() const { return &*it_; }

        const_iterator& operator++() {
            ++it_;
            advance_to_nonempty();
            return *this;
        }
        const_iterator operator++(int) { auto t = *this; ++*this; return t; }

        bool operator==(const const_iterator& o) const {
            if (mm_ == nullptr && o.mm_ == nullptr) return true;
            if (qid_ >= 3 && o.qid_ >= 3) return true;
            return qid_ == o.qid_ && it_ == o.it_;
        }
        bool operator!=(const const_iterator& o) const { return !(*this == o); }

    private:
        void advance_to_nonempty() {
            while (qid_ < 3 && it_ == mm_->get_queue(qid_).end()) {
                ++qid_;
                if (qid_ < 3) it_ = mm_->get_queue(qid_).begin();
            }
        }
        const mm_wtiny_lfu* mm_;
        uint8_t qid_;
        typename item_list::const_iterator it_;
    };

    // Const overload for const_iterator
    const item_list& get_queue(uint8_t qid) const {
        if (qid == kProtectionQueue) return protection_queue_;
        if (qid == kProbationQueue) return probation_queue_;
        return tiny_queue_;
    }

    const_iterator begin() const {
        for (uint8_t q = 0; q < 3; ++q) {
            if (!get_queue(q).empty()) return const_iterator(this, q, get_queue(q).begin());
        }
        return end();
    }
    const_iterator end() const { return const_iterator(); }

    // --------------------------------------------------------------------
    // Equality comparison (content and order must match)
    // --------------------------------------------------------------------

    friend bool operator==(const mm_wtiny_lfu& a, const mm_wtiny_lfu& b) {
        if (a.size() != b.size()) return false;
        auto ai = a.begin(), bi = b.begin();
        for (; ai != a.end(); ++ai, ++bi) {
            if (ai->key != bi->key || ai->value != bi->value) return false;
        }
        return true;
    }
    friend bool operator!=(const mm_wtiny_lfu& a, const mm_wtiny_lfu& b) { return !(a == b); }

    // --------------------------------------------------------------------
    // Stream output
    // --------------------------------------------------------------------

    friend std::ostream& operator<<(std::ostream& os, const mm_wtiny_lfu& c) {
        os << "mm_wtiny_lfu @" << &c << "  " << c.stats_ << "\n";
        std::size_t idx = 0;
        for (auto it = c.begin(); it != c.end(); ++it, ++idx) {
            os << "  [" << idx << "] key='";
            if constexpr (lru::detail::is_formattable_v<Key>) {
                os << std::format("{}", it->key);
            } else {
                os << std::format("<key at {:#x}>", reinterpret_cast<std::uintptr_t>(&it->key));
            }
            os << "' value='";
            if constexpr (lru::detail::is_formattable_v<Value>) {
                os << std::format("{}", it->value);
            } else {
                os << std::format("<val at {:#x}>", reinterpret_cast<std::uintptr_t>(&it->value));
            }
            os << "'\n";
        }
        return os;
    }

    // --------------------------------------------------------------------
    // Eviction (public for pooled_cache / unified_cache::evict())
    // --------------------------------------------------------------------

    /// B15: 在队列中找可淘汰节点。
    item_ptr find_eviction_victim_in_queue(item_list& queue) {
        auto* curr = queue.tail();
        if (!curr) return curr;
        const bool has_pred = static_cast<bool>(eviction_predicate_);
        size_t tries = 0;
        while (curr) {
            stats_.eviction_search_steps.fetch_add(1, std::memory_order_relaxed);
            // H0: 跳过有活跃句柄的节点（不计入 tries，这些节点绝对不能淘汰）
            if (curr->has_active_handle()) {
                stats_.pinned_skip_count.fetch_add(1, std::memory_order_relaxed);
                curr = static_cast<item_type*>(queue.get_prev(*curr));
                continue;
            }
            // B15: EvictionPredicate 否决（计入 tries）
            if (has_pred && !eviction_predicate_(curr->key, curr->value)) {
                curr = static_cast<item_type*>(queue.get_prev(*curr));
                if (++tries >= config_.eviction_search_tries) break;
                continue;
            }
            return curr;
        }
        return nullptr;
    }

    /// 从指定队列中淘汰一个通用节点。
    void evict_generic(item_ptr item, item_list& queue) {
        auto evict_result = item->refcount.markForEviction();
        if (evict_result != detail::MarkForEvictionResult::kSuccess) {
            return;  // Cannot evict — item has active handles or is already exclusive
        }
        const auto& key = item->key;
        size_type mem = calc_item_memory(key, item->value);
        stats_.current_memory.fetch_sub(mem);
        stats_.register_eviction();
        if (callbacks_.has_eviction_callbacks()) {
            Value value = std::move(item->value);
            callbacks_.collect_evict(key, std::move(value));
        }
        map_.erase(key);
        item->refcount.unmarkInMMContainer();
        item->refcount.unmarkForEviction();
        queue.remove(*item);
        detail::hazptr_domain::default_domain().retire(item);
        stats_.current_size.store(total_size());
    }

    void evict() {
        cleanup_pending_deletion();
        // B15: 使用 predicate 辅助查找
        auto* tiny_victim = find_eviction_victim_in_queue(tiny_queue_);
        auto* prob_victim = find_eviction_victim_in_queue(probation_queue_);

        // W-TinyLFU admission policy (A2): when both Tiny (window) and
        // Probation (main) have eviction candidates, compare their frequencies
        // and evict the lower-frequency item. This is the core W-TinyLFU
        // admission decision: a newcomer in Tiny displaces a Main item only
        // if its frequency is >= the Main tail's frequency (configurable via
        // newcomer_wins_on_tie). Aligns with CacheLib MMWTinyLFU.h:685-693
        // admitToProbation and the W-TinyLFU paper (Einziger et al., 2017).
        // Without this check, frequently-accessed hot keys resting at the
        // Probation tail could be evicted by newcomers with zero history,
        // defeating the purpose of frequency-aware admission.
        if (tiny_victim && prob_victim) {
            auto tiny_freq = sketch_.estimate(tiny_victim->key);
            auto prob_freq = sketch_.estimate(prob_victim->key);
            bool admit = config_.newcomer_wins_on_tie
                             ? (tiny_freq >= prob_freq)
                             : (tiny_freq > prob_freq);
            if (admit) {
                evict_generic(prob_victim, probation_queue_);
            } else {
                evict_generic(tiny_victim, tiny_queue_);
            }
            return;
        }

        // Single-queue fallbacks: evict whichever queue has a candidate.
        if (tiny_victim) { evict_generic(tiny_victim, tiny_queue_); return; }
        if (prob_victim) { evict_generic(prob_victim, probation_queue_); return; }

        // Final fallback: if Tiny and Probation are empty (or all pinned),
        // demote the LRU item from Protection to Probation, then evict it.
        // This prevents eviction deadlock when shrink_to_fit() needs to
        // reduce capacity below the current Protection queue size.
        if (!protection_queue_.empty()) {
            auto* prot_tail = protection_queue_.tail();
            if (prot_tail && !prot_tail->has_active_handle()) {
                prot_tail->queue_id = kProbationQueue;
                protection_queue_.remove(*prot_tail);
                probation_queue_.link_at_tail(*prot_tail);
                evict_generic(prot_tail, probation_queue_);
                return;
            }
        }
    }

protected:
    item_list tiny_queue_;       // Queue 0: Window cache (new items)
    item_list probation_queue_;  // Queue 1: Trial area in Main
    item_list protection_queue_; // Queue 2: Protected area in Main
    map_type map_;
    sketch_type sketch_;
    uint32_t protection_freq_;

    size_type max_size_ = unlimited;
    size_type max_memory_ = unlimited;
    mm_wtiny_lfu_config config_;
    // A5: try_lock_update 优化使用的独立内部锁，与统一缓存层锁解耦
    // B10: 缓存行对齐以避免 false sharing（对齐 CacheLib MMLru.h:474）
    struct alignas(64) aligned_mutex_t { std::mutex m; };
    mutable aligned_mutex_t update_mutex_;
    uint32_t lru_refresh_time_ = 0;
    uint32_t next_reconfigure_time_ = std::numeric_limits<uint32_t>::max();

    mutable stats_type stats_;
    mutable callback_mgr callbacks_;
    // B15: EvictionPredicate
    std::function<bool(const Key&, const Value&)> eviction_predicate_;
    // B7: LockedIterator 活跃标记
    std::atomic<bool> iterator_active_{false};

    // Items removed by force_del() that still have active handles.
    std::vector<item_ptr> pending_deletion_;

    /// Clean up pending-deletion items whose handles have all been released.
    /// Fires the deferred eviction callback (with value move) before deleting.
    void cleanup_pending_deletion() {
        auto it = pending_deletion_.begin();
        while (it != pending_deletion_.end()) {
            if (!(*it)->has_active_handle()) {
                auto* item = *it;
                callbacks_.collect_evict(item->key, std::move(item->value));
                // P1-5: Route through hazptr retire instead of raw delete —
                // a concurrent hazptr-protected reader may still hold a
                // hazard pointer to this item even with refcount=0.
                detail::hazptr_domain::default_domain().retire(item);
                it = pending_deletion_.erase(it);
            } else {
                ++it;
            }
        }
    }

    std::function<size_type(const Key&)> key_size_fn_;
    std::function<size_type(const Value&)> value_size_fn_;

    // --------------------------------------------------------------------
    // Time utilities
    // --------------------------------------------------------------------

    static uint32_t current_time_sec() {
        return detail::cached_epoch_sec();
    }

    // ====================================================================
    // Adaptive Refresh Time (CacheLib's reconfigureLocked)
    // ====================================================================

    /// B1: 基于 Protection 队列尾部年龄动态调整 lru_refresh_time_。
    void reconfigure_locked(uint32_t curr_time) {
        if (curr_time < next_reconfigure_time_) return;
        if (config_.mm_reconfigure_interval_secs == 0) return;

        next_reconfigure_time_ = curr_time + config_.mm_reconfigure_interval_secs;

        auto* tail = protection_queue_.tail();
        if (!tail) tail = probation_queue_.tail();
        uint32_t tail_age = tail ? (curr_time - tail->hook.update_time) : 0;
        auto new_refresh = std::min(
            std::max(config_.default_lru_refresh_time,
                     static_cast<uint32_t>(static_cast<double>(tail_age) *
                                           config_.lru_refresh_ratio)),
            config_type::k_lru_refresh_time_cap);
        lru_refresh_time_ = new_refresh;
    }

    /// B8: 获取淘汰年龄统计。
    struct eviction_age_stat {
        uint64_t oldest_element_age{0};
        uint64_t projected_age{0};
        uint64_t queue_size{0};
    };

    eviction_age_stat get_eviction_age_stat(std::size_t projected_length = 0) const noexcept {
        eviction_age_stat stat;
        stat.queue_size = protection_queue_.size();
        const auto curr_time = current_time_sec();
        const auto* node = protection_queue_.tail();
        const auto* current_queue = static_cast<const item_list*>(&protection_queue_);
        if (!node) {
            node = probation_queue_.tail();
            current_queue = &probation_queue_;
        }
        if (!node) {
            node = tiny_queue_.tail();
            current_queue = &tiny_queue_;
        }
        stat.oldest_element_age = node ? (curr_time - node->hook.update_time) : 0;
        for (std::size_t seen = 0; seen < projected_length && node != nullptr; ++seen) {
            node = static_cast<const item_type*>(current_queue->get_prev(*node));
        }
        stat.projected_age = node ? (curr_time - node->hook.update_time) : stat.oldest_element_age;
        return stat;
    }

    // --------------------------------------------------------------------
    // Internal helpers
    // --------------------------------------------------------------------

    item_list& get_queue(uint8_t qid) {
        if (qid == kProtectionQueue) return protection_queue_;
        if (qid == kProbationQueue) return probation_queue_;
        return tiny_queue_;
    }

    size_type total_size() const {
        return tiny_queue_.size() + probation_queue_.size() + protection_queue_.size();
    }

    bool should_evict() const {
        if (max_size_ != unlimited && size() > max_size_) return true;
        if (max_memory_ != unlimited && current_memory() > max_memory_) return true;
        return false;
    }

    size_type calc_item_memory(const Key& key, const Value& value) const {
        size_type mem = item_overhead;
        if (key_size_fn_) mem += key_size_fn_(key) * 2;
        if (value_size_fn_) mem += value_size_fn_(value);
        return mem;
    }

    size_type expected_tiny_size() const {
        auto total = max_size_ != unlimited ? max_size_ : size();
        return std::max(size_type(1), static_cast<size_type>(total * config_.window_to_cache_size_ratio));
    }

    size_type expected_protection_size() const {
        auto main_size = size() - tiny_queue_.size();
        return static_cast<size_type>(main_size * config_.protection_ratio);
    }

    size_type expected_probation_size() const {
        auto main_size = size() - tiny_queue_.size();
        auto prot_size = expected_protection_size();
        return main_size > prot_size ? main_size - prot_size : 0;
    }

    void remove_from_queue(item_ptr item) {
        auto& queue = get_queue(item->queue_id);
        // A2: 清除 accessed 标志，对齐 CacheLib MMLru.h:744
        item->hook.clear_accessed();
        queue.remove(*item);
    }

    // ====================================================================
    // Delayed Promotion (CacheLib's recordAccess)
    // ====================================================================

    /// Record access to an item, with delayed promotion support.
    /// Returns true if the item was actually promoted (moved to head).
    bool record_access(item_ptr item, access_mode mode) {
        assert(item != nullptr);  // F1

        // Check updateOnWrite/updateOnRead
        if ((mode == access_mode::write && !config_.update_on_write) ||
            (mode == access_mode::read && !config_.update_on_read)) {
            return false;
        }

        auto curr = current_time_sec();

        // D3+A3: 对齐 CacheLib——首次访问(is_accessed==false)必提升；
        // 已访问节点仅在超 refresh time 后提升。移除 update_time>0 守卫。
        if (!item->hook.is_accessed()) {
            item->hook.set_accessed();
        } else if (curr < item->hook.update_time + lru_refresh_time_) {
            return false;  // Not enough time since last promotion
        }

        // A5: try_lock_update 优化——若启用，则尝试加锁；失败则返回 false 跳过提升（不阻塞）。
        // 对齐 CacheLib MMLru.h:567-577。成功获取锁后正常执行提升逻辑。
        auto promote = [this, item, curr]() {
            // B1: 在 promote 路径上定期调整 lru_refresh_time_
            reconfigure_locked(curr);

            // Move to head in the current queue
            auto& queue = get_queue(item->queue_id);
            queue.move_to_head(*item);
            item->hook.update_time = curr;

            // B5: Probation->Protection promotion. The frequency check uses the
            // count BEFORE this access's increment, aligning with CacheLib's
            // recordAccess (MMWTinyLFU.h:868-893) which checks freq before
            // calling updateFrequenciesLocked.
            if (item->queue_id == kProbationQueue) {
                try_promote_to_protection(item);
            }

            // B4: Update frequency count (equivalent to CacheLib's
            // updateFrequenciesLocked). Called after the promotion check so the
            // check uses the pre-increment frequency.
            sketch_.record(item->key);
        };

        if (try_lock_update_enabled()) {
            std::unique_lock<std::mutex> lock(update_mutex_.m, std::try_to_lock);
            if (!lock) {
                stats_.try_lock_fail_count.fetch_add(1, std::memory_order_relaxed);
                return false;
            }
            promote();
        } else {
            promote();
        }

        return true;
    }

    /// Record access with a pre-computed current time.
    bool record_access_at(item_ptr item, access_mode mode, uint32_t curr) {
        assert(item != nullptr);

        if ((mode == access_mode::write && !config_.update_on_write) ||
            (mode == access_mode::read && !config_.update_on_read)) {
            return false;
        }

        if (!item->hook.is_accessed()) {
            item->hook.set_accessed();
        } else if (curr < item->hook.update_time + lru_refresh_time_) {
            return false;
        }

        auto promote = [this, item, curr]() {
            reconfigure_locked(curr);
            auto& queue = get_queue(item->queue_id);
            queue.move_to_head(*item);
            item->hook.update_time = curr;

            if (item->queue_id == kProbationQueue) {
                try_promote_to_protection(item);
            }

            sketch_.record(item->key);
        };

        if (try_lock_update_enabled()) {
            std::unique_lock<std::mutex> lock(update_mutex_.m, std::try_to_lock);
            if (!lock) {
                stats_.try_lock_fail_count.fetch_add(1, std::memory_order_relaxed);
                return false;
            }
            promote();
        } else {
            promote();
        }

        return true;
    }

    /// B5: Promote a Probation item to Protection if its frequency exceeds
    /// protection_freq_. If Protection overflows, degrade its tail to the
    /// Probation TAIL (not head) to preserve the degraded item's survival.
    /// Aligns with CacheLib MMWTinyLFU.h:834-893 recordAccess promotion logic.
    void try_promote_to_protection(item_ptr node) {
        assert(node != nullptr);                  // F1
        assert(node->queue_id == kProbationQueue);  // F1: caller guarantees Probation

        auto freq = sketch_.estimate(node->key);
        // A1: strict greater-than (aligns with MMWTinyLFU.h:870).
        // protection_freq_=3 means frequency 3 does NOT promote; frequency 4 does.
        if (freq > protection_freq_) {
            // Promote to Protection head
            node->queue_id = kProtectionQueue;
            probation_queue_.remove(*node);
            protection_queue_.link_at_head(*node);

            // If Protection is over capacity, degrade tail to Probation TAIL.
            auto expected_prot = expected_protection_size();
            // A3: strict > (no +1 tolerance, aligns with MMWTinyLFU.h:885).
            if (protection_queue_.size() > expected_prot) {
                assert(protection_queue_.size() > expected_prot);  // F1
                auto* prot_tail = protection_queue_.tail();
                if (prot_tail) {
                    prot_tail->queue_id = kProbationQueue;
                    protection_queue_.remove(*prot_tail);
                    // B5 key correction: Degrade to Probation TAIL (not head!),
                    // aligning with MMWTinyLFU.h:889. This preserves the
                    // degraded item's survival chance.
                    probation_queue_.link_at_tail(*prot_tail);
                }
            }
        }
    }

    // ====================================================================
    // Insert / Update / Evict
    // ====================================================================

    template <typename V>
    void insert_new(const Key& key, V&& value) {
        if (max_size_ == 0) return;
        while (should_evict()) {
            auto old_size = size();
            evict();
            if (size() == old_size) break;
        }
        if (max_size_ != unlimited && size() >= max_size_) evict();

        auto mem = calc_item_memory(key, value);
        if (max_memory_ != unlimited) {
            while (!map_.empty() && max_memory_ - stats_.current_memory.load() < mem) {
                auto old_size = size();
                evict();
                if (size() == old_size) break;
            }
        }

        // New items always enter Tiny queue
        auto* item = this->allocate_item(key, std::forward<V>(value));
        assert(item != nullptr);  // F1
        auto curr = current_time_sec();
        item->hook.update_time = curr;
        item->hook.clear_accessed();  // Not yet accessed
        item->queue_id = kTinyQueue;

        // F1: item must not be linked into any list before insertion.
        assert(item->hook.prev == nullptr && item->hook.next == nullptr);
        tiny_queue_.link_at_head(*item);
        item->refcount.markInMMContainer();
        map_.insert(key, item);

        sketch_.record(key);
        // B3: auto-grow CMS counters when cache size increases significantly.
        // Aligns with CacheLib MMWTinyLFU add() calling maybeGrowAccessCountersLocked.
        sketch_.maybe_grow_access_counters();
        stats_.current_size.store(total_size());
        stats_.current_memory.fetch_add(mem);
        stats_.register_insertion();
        callbacks_.collect_insert(key, item->value);

        maybe_promote_from_tiny();
    }

    template <typename V>
    void update_existing(item_ptr item, V&& value, access_mode mode) {
        auto curr = current_time_sec();
        size_type old_mem = calc_item_memory(item->key, item->value);
        // P-MED-2 (T-H4): Strong exception guarantee via copy-then-swap.
        if constexpr (std::is_nothrow_swappable_v<Value> &&
                      std::is_constructible_v<Value, V>) {
            Value tmp(std::forward<V>(value));
            using std::swap;
            swap(item->value, tmp);
        } else {
            item->value = std::forward<V>(value);
        }
        // B4: frequency update is now unified inside record_access.
        record_access_at(item, mode, curr);
        size_type new_mem = calc_item_memory(item->key, item->value);
        if (new_mem > old_mem) {
            stats_.current_memory.fetch_add(new_mem - old_mem);
        } else if (new_mem < old_mem) {
            stats_.current_memory.fetch_sub(old_mem - new_mem);
        }
        if (should_evict()) {
            shrink_to_fit();
        }
        // O7: Fire on_update for value changes on existing keys (distinct
        // from on_insert, which fires only for new key insertions).
        callbacks_.collect_update(item->key, item->value);
    }

    /// A9 修正: 对齐 CacheLib W-TinyLFU Tiny 超容时的频率感知晋升语义。
    /// Tiny 超容时，Tiny tail 是"候选者"（candidate）。
    /// - 若 Probation 未满：直接晋升候选者到 Probation head。
    /// - 若 Probation 已满：比较候选者频率 vs Probation tail 频率：
    ///   * 候选者胜（>= with tie, configurable via newcomer_wins_on_tie）：
    ///     淘汰 Probation tail，晋升候选者到 Probation head。
    ///   * 候选者败：淘汰候选者（不晋升），保留 Probation tail。
    /// 这才是真正的 W-TinyLFU admission policy（对齐 CacheLib
    /// MMWTinyLFU.h:685-693 admitToProbation 与论文 Einziger et al., 2017）。
    /// 之前的实现无条件晋升 + 无条件淘汰 Probation tail，会导致
    /// 高频热键被新来者挤出缓存（test_concurrent_mm_strategies.cpp
    /// 的 MmWTinyLfuMixedInsertUpdate 在某些调度下失败）。
    void maybe_promote_from_tiny() {
        auto expected_tiny = expected_tiny_size();
        while (tiny_queue_.size() > expected_tiny) {
            auto* candidate = tiny_queue_.tail();
            if (!candidate) break;

            auto probation_capacity = expected_probation_size();
            if (probation_capacity != unlimited &&
                probation_queue_.size() >= probation_capacity) {
                // Probation 满了：找 Probation tail 作为对照
                auto* prob_tail = probation_queue_.tail();
                size_t tries = 0;
                while (prob_tail && prob_tail->has_active_handle()) {
                    prob_tail = probation_queue_.get_prev(*prob_tail);
                    if (++tries >= config_.eviction_search_tries) {
                        prob_tail = nullptr;
                        break;
                    }
                }
                if (prob_tail) {
                    auto cand_freq = sketch_.estimate(candidate->key);
                    auto prob_freq = sketch_.estimate(prob_tail->key);
                    bool admit = config_.newcomer_wins_on_tie
                                     ? (cand_freq >= prob_freq)
                                     : (cand_freq > prob_freq);
                    if (admit) {
                        // 候选者胜出：淘汰 Probation tail，晋升候选者
                        evict_generic(prob_tail, probation_queue_);
                    } else {
                        // 候选者失败：淘汰候选者，不晋升
                        evict_generic(candidate, tiny_queue_);
                        continue;
                    }
                }
                // 若找不到可淘汰的 Probation tail（全部被 pin），
                // 仍走无条件晋升路径以避免死锁。
            }
            // 晋升 Tiny tail 到 Probation head
            promote_tiny_tail_to_probation();
        }
    }

    void promote_tiny_tail_to_probation() {
        auto* item = tiny_queue_.tail();
        if (!item) return;
        // Move from Tiny tail to Probation head
        item->queue_id = kProbationQueue;
        tiny_queue_.remove(*item);
        probation_queue_.link_at_head(*item);
    }

    void evict_from_tiny() {
        auto* item = tiny_queue_.tail();
        size_t tries = 0;
        while (item && item->has_active_handle()) {
            item = tiny_queue_.get_prev(*item);
            if (++tries >= config_.eviction_search_tries) { item = nullptr; break; }
        }
        if (!item) return;
        evict_generic(item, tiny_queue_);
    }

    void evict_from_probation() {
        auto* item = probation_queue_.tail();
        size_t tries = 0;
        while (item && item->has_active_handle()) {
            item = probation_queue_.get_prev(*item);
            if (++tries >= config_.eviction_search_tries) { item = nullptr; break; }
        }
        if (!item) return;
        evict_generic(item, probation_queue_);
    }

    void evict_from_protection() {
        auto* item = protection_queue_.tail();
        size_t tries = 0;
        while (item && item->has_active_handle()) {
            item = protection_queue_.get_prev(*item);
            if (++tries >= config_.eviction_search_tries) { item = nullptr; break; }
        }
        if (!item) return;
        evict_generic(item, protection_queue_);
    }

    /// B14: 原位替换节点，保留 queue_id、update_time 与 accessed 状态。
    void replace_node(item_ptr old_node, item_ptr new_node) {
        assert(old_node != nullptr && new_node != nullptr);
        new_node->hook.update_time = old_node->hook.update_time;
        if (old_node->hook.is_accessed()) {
            new_node->hook.set_accessed();
        } else {
            new_node->hook.clear_accessed();
        }
        new_node->queue_id = old_node->queue_id;
        auto& queue = get_queue(old_node->queue_id);
        queue.replace(*old_node, *new_node);
        map_.insert_or_assign(old_node->key, new_node);
    }

    /// B15: 设置淘汰谓词。
    void set_eviction_predicate(std::function<bool(const Key&, const Value&)> pred) {
        eviction_predicate_ = std::move(pred);
    }

    void erase_impl(const Key& key) {
        auto ptr = map_.find(key);
        if (!ptr) return;
        auto* item = ptr;
        if (item->has_active_handle()) return;
        size_type mem = calc_item_memory(item->key, item->value);
        stats_.current_memory.fetch_sub(mem);
        callbacks_.collect_evict(item->key, std::move(item->value));
        remove_from_queue(item);
        map_.erase(key);
        detail::hazptr_domain::default_domain().retire(item);
        stats_.current_size.store(total_size());
    }

    // ====================================================================
    // S0: Faithful serialization rebuild
    // ====================================================================
public:

    template <typename InputIt>
    void rebuild_from_serialized(InputIt first, InputIt last) {
        flush();
        for (auto it = first; it != last; ++it) {
            auto* item = this->allocate_item(it->key, it->value);
            item->hook.update_time = it->update_time;
            if (it->flags & detail::intrusive_hook::kAccessedFlag) {
                item->hook.set_accessed();
            }
            item->queue_id = (it->queue_id <= kProtectionQueue) ? it->queue_id : kTinyQueue;
            auto& queue = get_queue(item->queue_id);
            queue.link_at_tail(*item);
            item->refcount.markInMMContainer();
            map_.insert(item->key, item);
            stats_.current_memory.fetch_add(calc_item_memory(item->key, item->value));
        }
        stats_.current_size.store(total_size());
    }
};



// ============================================================================
// Sharded LRU Strategy - sharded_mm_lru
// ============================================================================

/// Configuration for sharded LRU eviction.
struct sharded_mm_lru_config {
    /// Number of shards (default 64, must be > 0).
    ///
    /// P2-3 (T3.4): The shard count is now DECOUPLED from the stripe
    /// count of `striped_thread_safe_policy`. The sharded_mm_lru itself
    /// owns per-shard `distributed_shared_mutex` instances, so each shard
    /// is safely protected by its own lock — independent of how many
    /// stripes the caller holds. This means you can tune `num_shards`
    /// and `num_stripes` independently:
    ///   - num_shards > num_stripes: finer LRU granularity with fewer
    ///     stripe mutexes (each shard gets its own per-shard lock, so
    ///     concurrency is num_shards, not num_stripes).
    ///   - num_shards < num_stripes: fewer shards for memory savings
    ///     with more stripes (the extra stripes become redundant since
    ///     per-shard lock already serializes access).
    ///   - num_shards == num_stripes (default): 1:1 mapping.
    ///
    /// Historical note: Before T3.4, num_shards HAD to equal num_stripes
    /// because the stripe lock was the only protection for the shard.
    /// With per-shard locks added in T3.4, this constraint is lifted.
    ///
    /// Note: When max_size < num_shards, distribute_max_size() gives
    /// shards [0, max_size) max_size=1 and shards [max_size, num_shards)
    /// max_size=0. Keys that hash to the zero-capacity shards are silently
    /// rejected. This is the existing behavior and is acceptable because
    /// small caches don't benefit from sharding anyway. Tests that use
    /// small max_size (e.g., striped_cache{10}) are designed to work with
    /// this behavior by using keys that hash to live shards.
    std::size_t num_shards = 64;

    /// Underlying mm_lru config applied to each shard.
    mm_lru_config lru_config;

    /// Expected total number of items across all shards for automatic bucket
    /// count sizing. 0 = use default bucket count (1024 per shard).
    /// When > 0, each shard's hash table is pre-sized via
    /// concurrent_hash_table::buckets_for_items(expected_items / num_shards)
    /// to keep average chain length ≤ 0.25 at the expected load.
    /// If lru_config.expected_items is also > 0, this field takes precedence.
    std::size_t expected_items = 0;

    /// Custom node allocation function for non-EmbeddedChain hash table nodes.
    /// Propagated to all shards. nullptr (default) = standard new/delete.
    void* (*alloc_fn)(std::size_t) = nullptr;

    /// Custom node deallocation function (must pair with alloc_fn).
    void  (*dealloc_fn)(void*) = nullptr;

    /// TTL eviction batch size: maximum number of expired items to evict per
    /// shard per lock acquisition. When 0 (default), delegates to
    /// lru_config.ttl_evict_batch_size. This field allows overriding the
    /// per-shard config for the sharded case. Setting to a positive value
    /// (e.g. 64) makes the TTL cleaner release the lock after evicting
    /// that many items, allowing concurrent readers/writers to proceed.
    std::size_t ttl_evict_batch_size = 0;

    /// T-G8: When true, `shard_for_hash()` applies a splitmix64 hash
    /// mixing on top of the caller-supplied hash before taking
    /// `hash % num_shards`. This spreads poorly-distributed keys
    /// (sequential integers, monotonic IDs, low-entropy hashes) evenly
    /// across shards, eliminating hot shards under skewed access.
    ///
    /// Default is `true` (T-G8 / G12): production safety default. The
    /// `distribute_max_size()` P0-A fix grants every shard a minimum
    /// quota of 1, so mixing is safe even when `max_size < num_shards`.
    /// The default_hash (ankerl::unordered_dense::hash) is already
    /// well-mixed, so mixing is technically redundant for it — but
    /// keeping it on as a defense-in-depth against callers who supply
    /// identity hashes (std::hash<int>) costs only a few
    /// shifts/multiplies per shard lookup, which is negligible vs. the
    /// cost of a hot shard under 32+ thread read contention. For
    /// already-mixed hashes (e.g. ankerl::unordered_dense::hash),
    /// overlaying splitmix64 does not harm distribution — it is
    /// idempotent in the sense that well-mixed input remains well-mixed.
    ///
    /// Set to `false` only when you have benchmarked your workload and
    /// confirmed the splitmix64 cost is measurable AND your hash is
    /// already well-mixed.
    bool mix_shard_hash = true;

    void validate() const {
        if (num_shards == 0) {
            throw std::invalid_argument("sharded_mm_lru_config: num_shards must be > 0");
        }
        lru_config.validate();
    }
};

/// Sharded LRU: distributes keys across N independent mm_lru shards.
/// Each shard has its own map + intrusive list + callback_manager + stats.
/// Key-to-shard mapping uses Hash(key) % num_shards.
///
/// Thread safety is NOT provided by this class itself — the caller
/// (unified_cache) is responsible for locking. With striped locking,
/// each shard is protected by its own stripe mutex, enabling concurrent
/// access to different shards.
template <
    typename Key,
    typename Value,
    typename Hash = std::hash<Key>,
    typename KeyEqual = std::equal_to<Key>,
    typename ProbingStyle = detail::chain_probing_tag,
    bool Segmented = false
>
class sharded_mm_lru {
public:
    using key_type = Key;
    using mapped_type = Value;
    using size_type = std::size_t;
    using config_type = sharded_mm_lru_config;

    using shard_type = mm_lru<Key, Value, Hash, KeyEqual, ProbingStyle, Segmented>;
    using item_type = typename shard_type::item_type;
    using item_ptr = typename shard_type::item_ptr;
    using item_list = typename shard_type::item_list;
    using iterator = typename shard_type::iterator;
    using const_iterator = typename shard_type::const_iterator;
    using reverse_iterator = typename shard_type::reverse_iterator;
    using const_reverse_iterator = typename shard_type::const_reverse_iterator;
    using map_type = typename shard_type::map_type;

    using callback_mgr = callback_manager<Key, Value>;
    using stats_type = cache_stats;

    static constexpr size_type npos = unlimited;
    static constexpr size_type item_overhead = shard_type::item_overhead;

    /// P2-3 (T3.4): Static flag indicating that this MM type provides
    /// per-shard locking. When true, `unified_cache` delegates lock
    /// acquisition to `acquire_shard_*_lock(shard_idx)` instead of
    /// `striped_thread_safe_policy`, allowing `num_shards` to be set
    /// independently of `num_stripes` (decoupling shard count from
    /// stripe count for independent tuning).
    static constexpr bool has_per_shard_lock = true;

    // --------------------------------------------------------------------
    // Constructors / Destructor
    // --------------------------------------------------------------------

    sharded_mm_lru() : sharded_mm_lru(sharded_mm_lru_config{}) {}

    explicit sharded_mm_lru(const sharded_mm_lru_config& config)
        : num_shards_(config.num_shards)
        , config_(config)
        , mix_shard_hash_(config.mix_shard_hash) {
        config.validate();
        // P2-3 (T3.4): Allocate one mutex per shard in a single
        // heap allocation. `distributed_shared_mutex` is neither
        // copyable nor movable (it owns atomics and wait queues),
        // so it cannot live in a `std::vector`. A `unique_ptr<T[]>`
        // default-constructs each mutex in place — no move needed —
        // and gives contiguous storage for cache-friendly access.
        per_shard_mutexes_ = std::make_unique<detail::distributed_shared_mutex[]>(num_shards_);
        // Compute per-shard expected_items if the top-level expected_items is set.
        // This takes precedence over lru_config.expected_items.
        mm_lru_config shard_config = config.lru_config;
        if (config.expected_items > 0) {
            shard_config.expected_items = config.expected_items / num_shards_;
            if (shard_config.expected_items == 0) shard_config.expected_items = 1;
        }
        // Propagate custom alloc/dealloc functions to each shard.
        shard_config.alloc_fn = config.alloc_fn;
        shard_config.dealloc_fn = config.dealloc_fn;
        shards_.reserve(num_shards_);
        for (std::size_t i = 0; i < num_shards_; ++i) {
            shards_.push_back(std::make_unique<shard_type>(shard_config));
        }
    }

    sharded_mm_lru(size_type max_size, const sharded_mm_lru_config& config = sharded_mm_lru_config{})
        : sharded_mm_lru([&] {
              auto cfg = config;
              if (cfg.expected_items == 0 && cfg.lru_config.expected_items == 0
                  && max_size > 0 && max_size != unlimited) {
                  cfg.expected_items = max_size;
              }
              return cfg;
          }()) {
        max_size_ = max_size;
        distribute_max_size();
    }

    sharded_mm_lru(size_type max_size, size_type max_memory,
                    const sharded_mm_lru_config& config = sharded_mm_lru_config{})
        : sharded_mm_lru([&] {
              auto cfg = config;
              if (cfg.expected_items == 0 && cfg.lru_config.expected_items == 0
                  && max_size > 0 && max_size != unlimited) {
                  cfg.expected_items = max_size;
              }
              return cfg;
          }()) {
        max_size_ = max_size;
        max_memory_ = max_memory;
        distribute_max_size();
        distribute_max_memory();
    }

    ~sharded_mm_lru() = default;

    // Non-copyable, non-movable
    sharded_mm_lru(const sharded_mm_lru&) = delete;
    sharded_mm_lru& operator=(const sharded_mm_lru&) = delete;

    // --------------------------------------------------------------------
    // Shard lookup
    // --------------------------------------------------------------------

    // P2-3 (T3.4): shard selection uses `hash % num_shards_`. This is
    // now independent of stripe selection (`hash % num_stripes` in
    // striped_mutex) — the two can use different N values. This is
    // safe because sharded_mm_lru owns per-shard distributed_shared_mutex
    // instances (see per_shard_mutexes_ below), so each shard is
    // protected by its own lock regardless of which stripe the caller
    // holds. The caller's stripe lock and the shard's per-shard lock
    // are both acquired (the stripe lock is now redundant for safety,
    // but kept for backwards compatibility and to avoid disturbing
    // the existing is_striped dispatch in unified_cache).
    //
    // Historical note: Before T3.4, shard_idx HAD to equal stripe_idx
    // because the stripe lock was the only protection for the shard.
    // With per-shard locks, this constraint is lifted.
    //
    // P5 note: The R-9 zero-capacity shard problem (mixing routes keys
    // to shards with quota 0, silently dropping inserts) is now fully
    // resolved by the P0-A fix in `distribute_max_size()`: every shard
    // is granted a minimum quota of 1, even when `max_size < num_shards`.
    // Hash mixing is therefore safe to enable via `config.mix_shard_hash`
    // for any workload. The flag defaults to true (G12) to prevent hot
    // shards when callers supply identity hashes (e.g. std::hash<int>);
    // for already-mixed hashes (e.g. ankerl::unordered_dense::hash),
    // overlaying splitmix64 does not harm distribution.

    /// Compute which shard a key belongs to.
    std::size_t shard_for(const Key& key) const noexcept {
        return shard_for_hash(Hash{}(key));
    }

    /// Compute which shard a pre-computed hash belongs to.
    /// When `mix_shard_hash` is enabled in config, applies splitmix64
    /// mixing to spread poorly-distributed keys (sequential integers,
    /// low-entropy hashes) evenly across shards.
    std::size_t shard_for_hash(std::size_t hash) const noexcept {
        if (mix_shard_hash_) {
            // G12: splitmix64 (the Murmur3 64-bit finalizer / Mix13) —
            // 3 rounds of xor-shift-multiply with good avalanche properties.
            // Default mix_shard_hash=true prevents hot shards when callers
            // supply identity hashes (std::hash<int>); for already-mixed
            // hashes (ankerl::unordered_dense::hash) overlaying splitmix64
            // does not harm distribution. The cost is a few
            // shifts/multiplies per shard lookup, negligible vs. a hot
            // shard under 32+ thread read contention.
            std::uint64_t z = static_cast<std::uint64_t>(hash)
                              + 0x9e3779b97f4a7c15ULL;
            z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
            z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
            hash = static_cast<std::size_t>(z ^ (z >> 31));
        }
        return hash % num_shards_;
    }

    /// P5: query whether hash mixing is enabled.
    bool mix_shard_hash() const noexcept { return mix_shard_hash_; }

    std::size_t num_shards() const noexcept { return num_shards_; }

    // --------------------------------------------------------------------
    // P2-3 (T3.4): Per-shard lock acquisition — replaces the role of
    // striped_thread_safe_policy for sharded_mm_lru. Each shard owns its
    // own distributed_shared_mutex, so callers can hold ANY stripe (or
    // no stripe) and still safely access the shard via its per-shard
    // lock. This decouples `num_shards` from `num_stripes`.
    //
    // Lock semantics:
    //   - acquire_shard_write_lock: blocking unique_lock (exclusive).
    //   - acquire_shard_read_lock:  blocking shared_lock (shared, multiple readers).
    //   - try_acquire_shard_write_lock: non-blocking; returns unowned
    //     unique_lock if the lock is contended (used by defer_promotion).
    // --------------------------------------------------------------------

    /// Acquire an exclusive (write) lock on the shard at `shard_idx`.
    /// Blocks until the lock is acquired.
    auto acquire_shard_write_lock(std::size_t shard_idx) const {
        return std::unique_lock<detail::distributed_shared_mutex>(
            per_shard_mutexes_[shard_idx]);
    }

    /// Acquire a shared (read) lock on the shard at `shard_idx`.
    /// Blocks until the lock is acquired. Multiple readers can hold the
    /// shared lock concurrently.
    auto acquire_shard_read_lock(std::size_t shard_idx) const {
        return std::shared_lock<detail::distributed_shared_mutex>(
            per_shard_mutexes_[shard_idx]);
    }

    /// Try to acquire an exclusive (write) lock on the shard at
    /// `shard_idx` without blocking. Returns the lock if successful,
    /// or an empty (not-owned) lock if the lock is contended.
    auto try_acquire_shard_write_lock(std::size_t shard_idx) const {
        return std::unique_lock<detail::distributed_shared_mutex>(
            per_shard_mutexes_[shard_idx], std::try_to_lock);
    }

    /// T-G1: Try to acquire a shared (read) lock on the shard at
    /// `shard_idx` without blocking. Used by the value-layer TTL scanner
    /// to avoid blocking writers during expired-key collection.
    auto try_acquire_shard_read_lock(std::size_t shard_idx) const {
        return std::shared_lock<detail::distributed_shared_mutex>(
            per_shard_mutexes_[shard_idx], std::try_to_lock);
    }

    // --------------------------------------------------------------------
    // Core cache API — dispatches to the appropriate shard
    // --------------------------------------------------------------------

    template <typename V>
    void set(const Key& key, V&& value) {
        shards_[shard_for(key)]->set(key, std::forward<V>(value));
    }

    // --------------------------------------------------------------------
    // P1-1: Native TTL integration — delegate to the per-shard mm_lru.
    // --------------------------------------------------------------------
    template <typename V>
    void set_with_expiry(const Key& key, V&& value, std::uint64_t expiry_ns) {
        shards_[shard_for(key)]->set_with_expiry(key, std::forward<V>(value), expiry_ns);
    }

    std::optional<std::uint64_t> ttl_remaining_ns(const Key& key) const {
        return shards_[shard_for(key)]->ttl_remaining_ns(key);
    }

    /// Evict expired items across all shards. Callers must hold the global
    /// write lock (or call from the background TTL cleaner which acquires
    /// per-shard write locks in turn).
    std::size_t evict_expired() {
        std::size_t total = 0;
        for (auto& shard : shards_) {
            total += shard->evict_expired();
        }
        return total;
    }

    /// Batched variant: evict at most `batch_size` expired items per shard.
    std::size_t evict_expired_n(std::size_t batch_size) {
        std::size_t total = 0;
        for (auto& shard : shards_) {
            total += shard->evict_expired_n(batch_size);
        }
        return total;
    }

    template <typename V>
    bool add(const Key& key, V&& value) {
        return shards_[shard_for(key)]->add(key, std::forward<V>(value));
    }

    template <typename V>
    bool replace(const Key& key, V&& value) {
        return shards_[shard_for(key)]->replace(key, std::forward<V>(value));
    }

    read_handle<Value> get(const Key& key) {
        return shards_[shard_for(key)]->get(key);
    }

    read_handle<const Value> get(const Key& key) const {
        return shards_[shard_for(key)]->get(key);
    }

    read_handle<const Value> peek(const Key& key) const {
        return shards_[shard_for(key)]->peek(key);
    }

    /// Internal: peek that returns mutable handle (for optimistic get path).
    read_handle<Value> peek_for_get(const Key& key) {
        return peek_for_get_with_hash(key, Hash{}(key));
    }

    /// T16.4: peek_for_get with a pre-computed hash. The hash MUST be
    /// the result of Hash{}(key) — callers are responsible for hash
    /// compatibility. Used by bulk_get to avoid re-hashing each key
    /// for both shard dispatch and hash-table lookup.
    read_handle<Value> peek_for_get_with_hash(const Key& key, std::size_t hash) {
        return shards_[shard_for_hash(hash)]->peek_for_get_with_hash(key, hash);
    }

    std::shared_ptr<Value> get_shared(const Key& key) {
        return shards_[shard_for(key)]->get_shared(key);
    }

    bool del(const Key& key) {
        return shards_[shard_for(key)]->del(key);
    }

    bool force_del(const Key& key) {
        return shards_[shard_for(key)]->force_del(key);
    }

    /// Peek the LRU tail key for a specific shard. Used by overflow_policy
    /// ::kForceEvict to identify the victim key without exposing internal
    /// item_type pointers. Returns std::nullopt if the shard is empty.
    std::optional<Key> peek_lru_tail_key(std::size_t shard_idx) const {
        return shards_[shard_idx]->peek_lru_tail_key();
    }

    std::optional<Value> pop(const Key& key) {
        return shards_[shard_for(key)]->pop(key);
    }

    /// Pop the LRU item from the first non-empty shard.
    /// Does NOT fire eviction callbacks. Does not lose data from other shards.
    /// Uses atomic counter to check for non-empty shards without acquiring locks.
    std::optional<std::pair<Key, Value>> pop_lru() {
        for (auto& shard : shards_) {
            if (shard->stats().current_size.load(std::memory_order_relaxed) != 0) {
                return shard->pop_lru();
            }
        }
        return std::nullopt;
    }

    bool contains(const Key& key) const {
        return shards_[shard_for(key)]->contains(key);
    }

    // --------------------------------------------------------------------
    // T16: Hash-reuse overloads. Accept a pre-computed hash so the caller
    // can avoid re-hashing the same key across shard dispatch, stripe
    // selection, and hash-table lookup. The hash MUST be the result of
    // Hash{}(key) — callers are responsible for hash compatibility.
    // --------------------------------------------------------------------

    /// T16.1: peek_for_get that also returns the hash it would use for
    /// shard dispatch. The returned hash can be passed to subsequent
    /// `*_with_hash` calls to avoid re-hashing.
    ///
    /// T16.4 note: This method is retained for API compatibility with
    /// callers that want to discover the hash from a single call. It
    /// delegates to the T16.4 `peek_for_get_with_hash(key, hash)` and
    /// returns the hash alongside the handle.
    std::pair<read_handle<Value>, std::size_t>
    peek_for_get_with_hash_and_return_hash(const Key& key, std::size_t hash) {
        return {peek_for_get_with_hash(key, hash), hash};
    }

    template <typename V>
    void set_with_hash(const Key& key, std::size_t hash, V&& value) {
        shards_[shard_for_hash(hash)]->set(key, std::forward<V>(value));
    }

    read_handle<Value> get_with_hash(const Key& key, std::size_t hash) {
        return shards_[shard_for_hash(hash)]->get(key);
    }

    read_handle<const Value> peek_with_hash(const Key& key, std::size_t hash) const {
        return shards_[shard_for_hash(hash)]->peek(key);
    }

    bool contains_with_hash(const Key& key, std::size_t hash) const {
        return shards_[shard_for_hash(hash)]->contains(key);
    }

    bool del_with_hash(const Key& key, std::size_t hash) {
        return shards_[shard_for_hash(hash)]->del(key);
    }

    /// Flush all shards.
    ///
    /// P1-8 (T2.6 bugfix, phase 3): After delegating to per-shard `flush()`
    /// (which refreshes per-shard hash stats internally between Pass 1 and
    /// Pass 2), aggregate the per-shard stats into the first shard's stats
    /// for unified_cache to read. Do NOT call `shard->refresh_hash_stats()`
    /// here — that would traverse hash chains again, and items retired by
    /// Pass 2 may have been freed by the background reclaimer by now.
    ///
    /// P3-1 (UAF fix): Acquire each shard's exclusive write lock before
    /// calling `shard->flush()`. This is CRITICAL because `unified_cache::
    /// flush()` acquires the STRIPE locks (via `acquire_write_lock()` ->
    /// `striped_write_lock_all`), NOT the per-shard locks. Without the
    /// per-shard lock, concurrent `set()`/`get()` operations (which acquire
    /// the SHARD lock via `acquire_write_lock_for_key()`) can run in
    /// parallel with `flush()` and modify the hash chain via `map_.erase()`
    /// while `shard->flush()` -> `refresh_hash_stats()` ->
    /// `max_chain_length()` is traversing it. The background reclaimer can
    /// then free the just-retired item, causing heap-use-after-free in
    /// `max_chain_length()`. Holding the shard write lock serializes
    /// `shard->flush()` with concurrent per-key operations on the same
    /// shard, eliminating the race window.
    void flush() {
        for (std::size_t i = 0; i < num_shards_; ++i) {
            auto lock = acquire_shard_write_lock(i);
            shards_[i]->flush();
        }
        aggregate_hash_stats_internal();
    }

    /// Evict from a shard with the most items (best-effort load balancing).
    /// Uses atomic counters for shard selection — avoids acquiring shard locks.
    void evict_lru() {
        // Find the shard with the largest size and evict from it
        std::size_t best_shard = 0;
        std::size_t best_size = 0;
        for (std::size_t i = 0; i < num_shards_; ++i) {
            auto sz = shards_[i]->stats().current_size.load(std::memory_order_relaxed);
            if (sz > best_size) {
                best_size = sz;
                best_shard = i;
            }
        }
        if (best_size > 0) {
            // Acquire the per-shard write lock before evicting. A concurrent
            // set()/flush() on the same shard holds the shard lock and can
            // mutate the LRU list / hash chain; calling shard->evict_lru()
            // without the lock races with it. We cannot rely on the caller's
            // global stripe lock here (unified_cache::evict()) because per-key
            // operations use per-shard mutexes, which are a separate lock
            // domain. Use try_lock so this best-effort eviction never blocks:
            // if the shard is contended, skip this round — a later call will
            // retry. (Note: shards_[best_shard]->evict_lru() is mm_lru's
            // per-shard eviction, which expects the caller to hold the shard
            // write lock — the callers in insert_new() already do.)
            auto lock = try_acquire_shard_write_lock(best_shard);
            if (!lock) return;
            shards_[best_shard]->evict_lru();
        }
    }

    // --------------------------------------------------------------------
    // Capacity
    // --------------------------------------------------------------------

    bool empty() const noexcept {
        for (const auto& shard : shards_) {
            if (shard->stats().current_size.load(std::memory_order_relaxed) != 0)
                return false;
        }
        return true;
    }

    /// O(num_shards) size query using per-shard atomic counters.
    /// Reads stats().current_size with relaxed ordering — approximate but
    /// avoids traversing the intrusive list and is safe for concurrent reads.
    size_type size() const noexcept {
        size_type total = 0;
        for (const auto& shard : shards_) {
            total += static_cast<size_type>(
                shard->stats().current_size.load(std::memory_order_relaxed));
        }
        return total;
    }

    size_type max_size() const noexcept { return max_size_; }

    size_type max_memory() const noexcept { return max_memory_; }

    /// O(num_shards) memory query using per-shard atomic counters.
    /// Reads stats().current_memory with relaxed ordering for consistency
    /// with size() and concurrent-safety.
    size_type current_memory() const noexcept {
        size_type total = 0;
        for (const auto& shard : shards_) {
            total += static_cast<size_type>(
                shard->stats().current_memory.load(std::memory_order_relaxed));
        }
        return total;
    }

    /// Estimate memory that would be accounted for an item with the given key
    /// and value, including the fixed item overhead and any custom size
    /// calculators registered on the shard that owns `key`.
    size_type estimate_item_memory(const Key& key, const Value& value) const {
        return shards_[shard_for(key)]->estimate_item_memory(key, value);
    }

    /// Check if any item across all shards has an active handle.
    bool has_active_handles() const noexcept {
        for (const auto& shard : shards_) {
            if (shard->has_active_handles()) return true;
        }
        return false;
    }

    void max_size(size_type new_max) {
        max_size_ = new_max;
        distribute_max_size();
    }

    /// P2-1: Strict variant of max_size() — throws std::invalid_argument
    /// when `new_max < num_shards_`. This prevents the silent capacity
    /// amplification that the lenient `max_size()` performs to keep
    /// every shard reachable. Use this when the caller wants hard
    /// capacity guarantees (e.g. memory-bounded caches where 6.4x
    /// overshoot would cause OOM).
    ///
    /// Recommended minimum: `new_max >= num_shards_ * 16` so each shard
    /// has at least 16 slots of headroom for skewed distributions.
    void max_size_strict(size_type new_max) {
        if (new_max != unlimited && new_max < num_shards_) {
            throw std::invalid_argument(
                "sharded_mm_lru::max_size_strict(): new_max (" +
                std::to_string(new_max) + ") < num_shards (" +
                std::to_string(num_shards_) + "); refusing to silently "
                "amplify capacity. Use max_size() for lenient behavior "
                "or raise new_max to >= num_shards.");
        }
        max_size_ = new_max;
        distribute_max_size();
    }

    void max_memory(size_type new_max) {
        max_memory_ = new_max;
        distribute_max_memory();
    }

    void shrink_to_fit() {
        for (auto& shard : shards_) {
            shard->shrink_to_fit();
        }
    }

    // --------------------------------------------------------------------
    // Serialization estimates
    // --------------------------------------------------------------------

    size_type serialized_item_count() const { return size(); }

    size_type serialized_size_estimate() const {
        return size() * (sizeof(Key) + sizeof(Value) + sizeof(size_type) * 2);
    }

    // --------------------------------------------------------------------
    // Iterators — iterate over the first shard only (for compatibility).
    // Full iteration across shards should be done via rbegin()/rend().
    // --------------------------------------------------------------------

    iterator begin() noexcept { return shards_[0]->begin(); }
    iterator end() noexcept { return shards_[0]->end(); }
    const_iterator begin() const noexcept { return shards_[0]->begin(); }
    const_iterator end() const noexcept { return shards_[0]->end(); }

    // --------------------------------------------------------------------
    // Statistics and callbacks
    // --------------------------------------------------------------------

    /// Aggregate stats across all shards.
    stats_type stats() const {
        stats_type total;
        total.max_size.store(max_size_);
        total.max_memory.store(max_memory_);
        std::size_t total_hits = 0, total_misses = 0;
        std::size_t total_insertions = 0, total_evictions = 0;
        std::size_t total_current_size = 0, total_current_memory = 0;
        std::size_t total_write_lock_wait = 0, total_try_lock_fail = 0;
        std::size_t total_eviction_search = 0, total_pinned_skip = 0;
        // P1-1: Rehash and TLS ring flush aggregation.
        std::size_t total_rehash_count = 0, total_rehash_migrated = 0;
        std::uint64_t total_rehash_time_ns = 0;
        std::size_t total_tls_flush = 0;
        for (const auto& shard : shards_) {
            const auto& s = shard->stats();
            total_hits += s.hits.value.load(std::memory_order_relaxed);
            total_misses += s.misses.value.load(std::memory_order_relaxed);
            total_insertions += s.insertions.value.load(std::memory_order_relaxed);
            total_evictions += s.evictions.value.load(std::memory_order_relaxed);
            total_current_size += s.current_size.load(std::memory_order_relaxed);
            total_current_memory += s.current_memory.load(std::memory_order_relaxed);
            total_write_lock_wait += s.write_lock_wait_count.load(std::memory_order_relaxed);
            total_try_lock_fail += s.try_lock_fail_count.load(std::memory_order_relaxed);
            total_eviction_search += s.eviction_search_steps.load(std::memory_order_relaxed);
            total_pinned_skip += s.pinned_skip_count.load(std::memory_order_relaxed);
            total_rehash_count += s.rehash_count.load(std::memory_order_relaxed);
            total_rehash_time_ns += s.rehash_total_time_ns.load(std::memory_order_relaxed);
            total_rehash_migrated += s.rehash_migrated_items.load(std::memory_order_relaxed);
            total_tls_flush += s.tls_ring_flush_count.load(std::memory_order_relaxed);
            // Aggregate latency histograms bucket-by-bucket (get/set).
            total.get_latency.merge_from(s.get_latency);
            total.set_latency.merge_from(s.set_latency);
        }
        total.hits.value.store(total_hits, std::memory_order_relaxed);
        total.misses.value.store(total_misses, std::memory_order_relaxed);
        total.insertions.value.store(total_insertions, std::memory_order_relaxed);
        total.evictions.value.store(total_evictions, std::memory_order_relaxed);
        total.current_size.store(total_current_size, std::memory_order_relaxed);
        total.current_memory.store(total_current_memory, std::memory_order_relaxed);
        total.write_lock_wait_count.store(total_write_lock_wait, std::memory_order_relaxed);
        total.try_lock_fail_count.store(total_try_lock_fail, std::memory_order_relaxed);
        total.eviction_search_steps.store(total_eviction_search, std::memory_order_relaxed);
        total.pinned_skip_count.store(total_pinned_skip, std::memory_order_relaxed);
        total.rehash_count.store(total_rehash_count, std::memory_order_relaxed);
        total.rehash_total_time_ns.store(total_rehash_time_ns, std::memory_order_relaxed);
        total.rehash_migrated_items.store(total_rehash_migrated, std::memory_order_relaxed);
        total.tls_ring_flush_count.store(total_tls_flush, std::memory_order_relaxed);
        return total;
    }

    // P1-7: Aggregate pending-deletion count across all shards.
    std::size_t pending_deletion_count() const noexcept {
        std::size_t total = 0;
        for (const auto& shard : shards_) {
            total += shard->pending_deletion_count();
        }
        return total;
    }

    /// R9: Aggregate count of force_del() calls refused across all shards
    /// because the pending-deletion soft cap was reached.
    std::size_t pending_deletion_skipped_count() const noexcept {
        std::size_t total = 0;
        for (const auto& shard : shards_) {
            total += shard->pending_deletion_skipped_count();
        }
        return total;
    }

    /// Unified callback manager that dispatches to all shards.
    /// We store a single unified callback_mgr that mirrors registrations
    /// to all per-shard managers. collect_* is called per-shard internally
    /// by the mm_lru instances. flush_pending() must be called on the
    /// unified manager to flush all shards.
    ///
    /// NOTE: The per-shard mm_lru instances each have their own callback_mgr.
    /// The unified callbacks() returns the first shard's callback_mgr for
    /// registration (on_hit/on_miss/on_insert/on_evict). When registering
    /// callbacks through unified_cache, they must be propagated to all shards.
    /// This is handled by the unified_cache layer.
    callback_mgr& callbacks() noexcept { return unified_callbacks_; }
    const callback_mgr& callbacks() const noexcept { return unified_callbacks_; }

    /// Refresh hash table diagnostic stats across all shards.
    /// O(total_bucket_count) scan — call periodically, not on every operation.
    ///
    /// P1-8 (T2.6 bugfix, phase 3): This method is UNSAFE to call right
    /// after `flush()` because `shard->refresh_hash_stats()` traverses
    /// hash chains via `max_chain_length()`, and items retired by
    /// `flush()` Pass 2 may have been freed by the background reclaimer.
    /// `flush()` now refreshes per-shard stats internally (between Pass 1
    /// and Pass 2) and aggregates via `aggregate_hash_stats_internal()`.
    /// This method remains for `stats_snapshot()` and the background
    /// rehash balancer, which don't have the post-flush UAF window.
    void refresh_hash_stats() const noexcept {
        for (auto& shard : shards_) {
            shard->refresh_hash_stats();
        }
        aggregate_hash_stats_internal();
    }

    /// P1-8 (T2.6 bugfix, phase 3): Aggregate already-refreshed per-shard
    /// hash stats into the first shard's stats for unified_cache to read.
    /// Does NOT call `shard->refresh_hash_stats()` — assumes per-shard
    /// stats are already fresh. Used by `flush()` after per-shard `flush()`
    /// has refreshed stats internally (between Pass 1 and Pass 2).
    void aggregate_hash_stats_internal() const noexcept {
        float max_lf = 0.0f;
        std::size_t max_cl = 0;
        // T13.1: aggregate overload events across shards (counter),
        // and pick the strictest (smallest) threshold (config).
        std::size_t total_overload_events = 0;
        float strictest_threshold = std::numeric_limits<float>::max();
        bool any_threshold_set = false;
        for (auto& shard : shards_) {
            auto lf = shard->stats().hash_load_factor.load(std::memory_order_relaxed);
            auto cl = shard->stats().max_chain_length.load(std::memory_order_relaxed);
            if (lf > max_lf) max_lf = lf;
            if (cl > max_cl) max_cl = cl;
            total_overload_events += shard->stats().hash_overload_events.load(std::memory_order_relaxed);
            float th = shard->stats().hash_overload_threshold.load(std::memory_order_relaxed);
            if (th > 0.0f) {
                any_threshold_set = true;
                if (th < strictest_threshold) strictest_threshold = th;
            }
        }
        // Store the worst-case values in the first shard's stats for
        // unified_cache to read via mm_.stats().
        shards_[0]->stats().hash_load_factor.store(max_lf, std::memory_order_relaxed);
        shards_[0]->stats().max_chain_length.store(max_cl, std::memory_order_relaxed);
        shards_[0]->stats().hash_overload_events.store(total_overload_events, std::memory_order_relaxed);
        if (any_threshold_set) {
            shards_[0]->stats().hash_overload_threshold.store(strictest_threshold, std::memory_order_relaxed);
        }
    }

    /// P0-5 (T1.3): Nudge any in-progress incremental rehash on every
    /// shard to advance migration. Called by the background rehash
    /// balancer to ensure stalled rehashes eventually complete without
    /// requiring writes to the affected shard. Each call advances the
    /// per-call migration budget (kRehashFinishMaxBucketsPerCall) on
    /// each shard that has a pending rehash.
    void advance_incremental_rehash() noexcept {
        for (auto& shard : shards_) {
            shard->advance_incremental_rehash();
        }
    }

    /// T-B4 (P2-10): Refresh diagnostics cache on every shard. Each shard
    /// forwards to its underlying hash table's `refresh_diagnostics_cache()`
    /// (which is a no-op for non-segmented tables). Called by the
    /// background rehash balancer before `refresh_hash_stats()` so that
    /// the subsequent `max_chain_length()` read in `refresh_hash_stats()`
    /// hits the fast cached path instead of triggering a full bucket scan.
    /// After this returns, `diagnostics_cache_age_ms()` reports the
    /// maximum age across all shards (worst-case freshness).
    void refresh_diagnostics_cache() const noexcept {
        for (auto& shard : shards_) {
            shard->refresh_diagnostics_cache();
        }
    }

    /// T-B4 (P2-10): Maximum diagnostics cache age across all shards.
    /// Returns the worst-case age (most-stale shard). Returns
    /// `std::numeric_limits<std::uint64_t>::max()` if any shard's cache
    /// has never been refreshed. Operators should expect this value to
    /// be roughly equal to the balancer interval (1s default) under
    /// normal operation; significantly larger values indicate a stalled
    /// or missing balancer.
    std::uint64_t diagnostics_cache_age_ms() const noexcept {
        std::uint64_t max_age = 0;
        for (auto& shard : shards_) {
            std::uint64_t age = shard->diagnostics_cache_age_ms();
            if (age > max_age) max_age = age;
        }
        return max_age;
    }

    /// Propagate registered callbacks to all shards.
    /// Called after registering callbacks via the unified_callbacks_ manager.
    void propagate_callbacks_to_shards() {
        for (auto& shard : shards_) {
            shard->callbacks().clear_all();
            // Copy hit callbacks
            for (const auto& cb : unified_callbacks_.hit_callbacks()) {
                shard->callbacks().on_hit(cb);
            }
            // Copy miss callbacks
            for (const auto& cb : unified_callbacks_.miss_callbacks()) {
                shard->callbacks().on_miss(cb);
            }
            // Copy insert callbacks
            for (const auto& cb : unified_callbacks_.insert_callbacks()) {
                shard->callbacks().on_insert(cb);
            }
            // Copy eviction callbacks
            for (const auto& cb : unified_callbacks_.eviction_callbacks()) {
                shard->callbacks().on_evict(cb);
            }
            // O7: Copy update/expire/reject callbacks
            for (const auto& cb : unified_callbacks_.update_callbacks()) {
                shard->callbacks().on_update(cb);
            }
            for (const auto& cb : unified_callbacks_.expire_callbacks()) {
                shard->callbacks().on_expire(cb);
            }
            for (const auto& cb : unified_callbacks_.reject_callbacks()) {
                shard->callbacks().on_reject(cb);
            }
        }
    }

    /// Flush pending callbacks from all shards.
    void flush_pending_all() {
        for (auto& shard : shards_) {
            shard->callbacks().flush_pending();
        }
        unified_callbacks_.flush_pending();
    }

    /// Direct access to a specific shard (for advanced use).
    shard_type& shard(std::size_t idx) { return *shards_[idx]; }
    const shard_type& shard(std::size_t idx) const { return *shards_[idx]; }

    /// Pre-allocate hash table buckets across all shards.
    /// Distributes `expected_items` evenly and calls reserve() on each shard.
    void reserve(size_type expected_items) {
        auto per_shard = std::max(size_type(1), expected_items / num_shards_);
        for (auto& s : shards_) {
            s->reserve(per_shard);
        }
    }

    /// Enable/disable incremental rehash across all shards.
    void set_incremental_rehash(bool enabled) {
        for (auto& s : shards_) {
            s->set_incremental_rehash(enabled);
        }
    }

    /// T13.1: Set the hash overload threshold across all shards.
    void set_hash_overload_threshold(float threshold) noexcept {
        for (auto& s : shards_) {
            s->set_hash_overload_threshold(threshold);
        }
    }

    /// T13.2: Register an overload callback on all shards.
    void set_overload_callback(std::function<void(float, float)> cb) {
        for (auto& s : shards_) {
            s->set_overload_callback(cb);
        }
    }

    /// P2-4 (T2.4): Toggle async mode for the overload callback on all
    /// shards. See `concurrent_hash_table::set_async_overload_callback`
    /// for semantics. When enabled, each shard's rehash hot path enqueues
    /// overload events instead of invoking the callback inline.
    void set_async_overload_callback(bool enabled) noexcept {
        for (auto& s : shards_) {
            s->set_async_overload_callback(enabled);
        }
    }

    /// P2-4 (T2.4): Drain pending overload events from all shards and
    /// dispatch the registered callback for each. Returns the total
    /// number of events drained across all shards.
    std::size_t drain_overload_callbacks() {
        std::size_t total = 0;
        for (auto& s : shards_) {
            total += s->drain_overload_callbacks();
        }
        return total;
    }

    /// Query whether incremental rehash is enabled (true only if all shards have it enabled).
    bool incremental_rehash_enabled() const noexcept {
        for (const auto& s : shards_) {
            if (!s->incremental_rehash_enabled()) return false;
        }
        return true;
    }

    /// T11.5: String-based strategy setter — propagates to all shards.
    bool set_rehash_strategy(std::string_view strategy) noexcept {
        bool ok = true;
        for (auto& s : shards_) {
            if (!s->set_rehash_strategy(strategy)) ok = false;
        }
        return ok;
    }
    std::string_view rehash_strategy() const noexcept {
        return shards_.empty() ? std::string_view{"blocking"}
                               : shards_[0]->rehash_strategy();
    }

    /// T2.1: Set the EBR domain for all shards. Propagates to each
    /// shard's mm_lru, which in turn propagates to the shard's hash
    /// table (so find_and_pin_lockfree acquires epoch_guard at entry).
    void set_ebr_domain(detail::epoch_domain* domain) noexcept {
        for (auto& s : shards_) {
            s->set_ebr_domain(domain);
        }
    }

    bool is_ebr_mode() const noexcept {
        return shards_.empty() ? false : shards_[0]->is_ebr_mode();
    }

    /// T11.3: Aggregate blocked-writes count across all shards.
    std::size_t rehash_blocked_writes_count() const noexcept {
        std::size_t total = 0;
        for (const auto& s : shards_) {
            total += s->rehash_blocked_writes_count();
        }
        return total;
    }

    /// P1-5: Aggregate rehash_lockfree_fallback_count across all shards.
    /// Non-zero values indicate the lock-free read path is being degraded
    /// by rehash activity in one or more shards.
    std::size_t rehash_lockfree_fallback_count() const noexcept {
        std::size_t total = 0;
        for (const auto& s : shards_) {
            total += s->rehash_lockfree_fallback_count();
        }
        return total;
    }

    /// P0-D: Average rehash_in_progress_ratio across all shards. Each
    /// shard's ratio is itself the fraction of segments currently
    /// rehashing (or 0/1 for non-segmented shards). The average gives
    /// a single gauge for the whole cache: 0.0 means no shard is
    /// rehashing; 1.0 means every segment of every shard is rehashing
    /// (severe write-pressure / under-provisioned capacity).
    float rehash_in_progress_ratio() const noexcept {
        if (shards_.empty()) return 0.0f;
        float sum = 0.0f;
        for (const auto& s : shards_) {
            sum += s->rehash_in_progress_ratio();
        }
        return sum / static_cast<float>(shards_.size());
    }

    /// Whether any shard is currently in an incremental rehash.
    bool is_rehashing() const noexcept {
        for (const auto& s : shards_) {
            if (s->is_rehashing()) return true;
        }
        return false;
    }

    /// Sum of rehash_progress across all shards currently rehashing.
    size_type rehash_progress() const noexcept {
        size_type total = 0;
        for (const auto& s : shards_) total += s->rehash_progress();
        return total;
    }

    /// Sum of new bucket counts across all shards currently rehashing.
    size_type rehash_new_bucket_count() const noexcept {
        size_type total = 0;
        for (const auto& s : shards_) total += s->rehash_new_bucket_count();
        return total;
    }

    /// Sum of old bucket counts across all shards currently rehashing.
    size_type rehash_old_bucket_count() const noexcept {
        size_type total = 0;
        for (const auto& s : shards_) total += s->rehash_old_bucket_count();
        return total;
    }

    /// Total hash table buckets across all shards.
    size_type bucket_count() const noexcept {
        size_type total = 0;
        for (const auto& s : shards_) total += s->bucket_count();
        return total;
    }

    // --------------------------------------------------------------------
    // Memory policy
    // --------------------------------------------------------------------

    void set_key_size_calculator(std::function<size_type(const Key&)> func) {
        key_size_fn_ = func;
        for (auto& s : shards_) {
            s->set_key_size_calculator(func);
        }
    }

    void set_value_size_calculator(std::function<size_type(const Value&)> func) {
        value_size_fn_ = func;
        for (auto& s : shards_) {
            s->set_value_size_calculator(func);
        }
    }

    void set_eviction_predicate(std::function<bool(const Key&, const Value&)> pred) {
        eviction_predicate_ = pred;
        for (auto& s : shards_) {
            s->set_eviction_predicate(pred);
        }
    }

    // --------------------------------------------------------------------
    // Config access
    // --------------------------------------------------------------------

    const sharded_mm_lru_config& config() const noexcept { return config_; }

    /// Set custom hash table node allocation/deallocation functions.
    /// Propagated to all shards.
    void set_hash_alloc_fns(void* (*alloc_fn)(std::size_t), void (*dealloc_fn)(void*)) {
        config_.alloc_fn = alloc_fn;
        config_.dealloc_fn = dealloc_fn;
        for (std::size_t i = 0; i < num_shards_; ++i) {
            shards_[i]->set_hash_alloc_fns(alloc_fn, dealloc_fn);
        }
    }

    /// Set an external slab allocator for item allocation.
    /// Propagated to all shards.
    void set_allocator(slab_allocator* alloc) noexcept {
        allocator_ = alloc;
        for (std::size_t i = 0; i < num_shards_; ++i) {
            shards_[i]->set_allocator(alloc);
        }
    }

    /// Get the currently associated slab allocator (may be nullptr).
    slab_allocator* get_allocator() const noexcept { return allocator_; }

    // --------------------------------------------------------------------
    // Promote — for TLS ring flush
    // --------------------------------------------------------------------

    bool promote(const Key& key) {
        return shards_[shard_for(key)]->promote(key);
    }

    // --------------------------------------------------------------------
    // Equality comparison
    // --------------------------------------------------------------------

    friend bool operator==(const sharded_mm_lru& a, const sharded_mm_lru& b) {
        if (a.num_shards_ != b.num_shards_) return false;
        for (std::size_t i = 0; i < a.num_shards_; ++i) {
            if (*a.shards_[i] != *b.shards_[i]) return false;
        }
        return true;
    }

    friend bool operator!=(const sharded_mm_lru& a, const sharded_mm_lru& b) {
        return !(a == b);
    }

    // --------------------------------------------------------------------
    // Stream output
    // --------------------------------------------------------------------

    friend std::ostream& operator<<(std::ostream& os, const sharded_mm_lru& c) {
        os << "sharded_mm_lru @" << &c << " shards=" << c.num_shards_
           << " " << c.stats() << "\n";
        for (std::size_t i = 0; i < c.num_shards_; ++i) {
            if (!c.shards_[i]->empty()) {
                os << "  [shard " << i << "] " << *c.shards_[i] << "\n";
            }
        }
        return os;
    }

private:
    std::size_t num_shards_;
    std::vector<std::unique_ptr<shard_type>> shards_;
    sharded_mm_lru_config config_;
    slab_allocator* allocator_ = nullptr;
    /// P5: when true, shard_for_hash() applies splitmix64 hash mixing to
    /// spread poorly-distributed keys across shards. Initialized from
    /// config_.mix_shard_hash. See sharded_mm_lru_config::mix_shard_hash.
    bool mix_shard_hash_ = true;

    /// P2-3 (T3.4): Per-shard reader/writer mutex. Each shard owns its
    /// own distributed_shared_mutex, so the shard can be safely accessed
    /// regardless of which stripe (or how many stripes) the caller holds.
    /// This decouples `num_shards_` from `num_stripes`, allowing the user
    /// to tune them independently (e.g. more shards for finer LRU
    /// granularity, fewer stripes for less lock overhead, or vice versa).
    /// The mutexes are `mutable` so const-qualified observers
    /// (`peek`, `contains`, etc.) can acquire shared locks.
    mutable std::unique_ptr<detail::distributed_shared_mutex[]> per_shard_mutexes_;

    size_type max_size_ = unlimited;
    size_type max_memory_ = unlimited;

    /// P2-1: one-shot warning flag for silent max_size amplification.
    /// When `max_size_ < num_shards_`, distribute_max_size() silently
    /// raises max_size_ to num_shards_ so every shard has at least one
    /// slot. This flag ensures the stderr warning fires at most once
    /// per sharded_mm_lru instance (CAS-protected).
    alignas(64) std::atomic<bool> max_size_amplified_warned_{false};

    /// P2-1: Emit a one-shot stderr warning when max_size is silently
    /// amplified to num_shards. Uses CAS so the warning fires at most
    /// once per instance — operators see the message once, raise
    /// max_size, and subsequent calls are silent.
    void warn_max_size_amplified_once(size_type requested,
                                       std::size_t num_shards) noexcept {
        bool expected = false;
        if (!max_size_amplified_warned_.compare_exchange_strong(
                expected, true, std::memory_order_acq_rel)) {
            return;
        }
        std::fprintf(stderr,
            "[lru::sharded_mm_lru] WARNING: max_size (%zu) < num_shards (%zu); "
            "silently amplifying max_size to %zu so every shard has at least "
            "one slot. Effective capacity is %.1fx the requested value. "
            "Use max_size_strict() to make this a hard error, or raise "
            "max_size to >= num_shards * 16 for production workloads. "
            "This warning fires once per sharded_mm_lru instance.\n",
            static_cast<std::size_t>(requested),
            num_shards, num_shards,
            static_cast<double>(num_shards) /
                static_cast<double>(requested));
        std::fflush(stderr);
    }

    /// Unified callback manager — registrations are stored here and
    /// propagated to per-shard managers via propagate_callbacks_to_shards().
    /// Per-shard collect_* calls accumulate events in per-shard managers.
    /// flush_pending_all() flushes all per-shard managers.
    callback_mgr unified_callbacks_;

    std::function<size_type(const Key&)> key_size_fn_;
    std::function<size_type(const Value&)> value_size_fn_;
    std::function<bool(const Key&, const Value&)> eviction_predicate_;

    /// Distribute max_size across shards. Each shard gets max_size / num_shards_.
    /// If max_size is unlimited, each shard also gets unlimited.
    ///
    /// P0-A fix: when `max_size < num_shards_`, every shard still receives
    /// a minimum quota of 1 (and `max_size_` is raised to `num_shards_`).
    /// Without this floor, a well-mixed hash would route keys to 0-quota
    /// shards and silently reject every insert — the dual of the R-9 hot
    /// shard problem. The original R-9 design relied on `std::hash<int>`
    /// being the identity function so keys 0..N-1 landed on the first N
    /// shards (the only ones with capacity); changing the default hash to
    /// a well-mixed function (P0-A) broke that hidden assumption. Lifting
    /// every shard to quota 1 makes the cache usable with any hash, at
    /// the cost of a slightly larger effective capacity when the caller
    /// requested `max_size < num_shards_` — an acceptable trade for
    /// correctness (silently dropping inserts is never acceptable).
    void distribute_max_size() {
        if (max_size_ == unlimited) {
            for (auto& shard : shards_) {
                shard->max_size(unlimited);
            }
            return;
        }
        // P0-A: enforce per-shard minimum quota of 1.
        if (max_size_ < num_shards_) {
            // P2-1: Warn once when max_size is silently amplified to
            // num_shards_. Operators asking for e.g. max_size=10 with
            // 64 shards end up with 64 effective slots — 6.4x the
            // requested capacity. The warning fires at most once per
            // sharded_mm_lru instance so repeated set_max_size() calls
            // don't spam stderr. Use the strict API (set_max_size_strict)
            // to make this a hard error instead of a silent amplification.
            warn_max_size_amplified_once(max_size_, num_shards_);
            max_size_ = num_shards_;
        }
        size_type per_shard = max_size_ / num_shards_;
        size_type remainder = max_size_ % num_shards_;
        for (std::size_t i = 0; i < num_shards_; ++i) {
            size_type quota = per_shard + (i < remainder ? 1 : 0);
            // per_shard may be 0 only when num_shards_ > max_size_, which
            // the floor above prevents. Keep the max(per_shard, 1) guard
            // as a defensive invariant.
            if (quota == 0) quota = 1;
            shards_[i]->max_size(quota);
        }
    }

    void distribute_max_memory() {
        if (max_memory_ == unlimited) {
            for (auto& shard : shards_) {
                shard->max_memory(unlimited);
            }
            return;
        }
        // P0-A: enforce per-shard minimum memory quota of 1 byte so a
        // well-mixed hash does not route keys to 0-byte shards and
        // silently reject every insert. See distribute_max_size() above.
        if (max_memory_ < num_shards_) {
            max_memory_ = num_shards_;
        }
        size_type per_shard = max_memory_ / num_shards_;
        size_type remainder = max_memory_ % num_shards_;
        for (std::size_t i = 0; i < num_shards_; ++i) {
            size_type quota = per_shard + (i < remainder ? 1 : 0);
            if (quota == 0) quota = 1;
            shards_[i]->max_memory(quota);
        }
    }
};

} // namespace lru

#endif // LRU_MM_HPP
