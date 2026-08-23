// SPDX-License-Identifier: MIT
// Compact Cache — Inspired by Facebook CacheLib's CCacheAllocator / ICompactCache
//
// For extremely small items (key+value <= 64 bytes), the per-item overhead of
// heap allocation, hash map entries, and linked-list pointers can dominate memory
// usage. CompactCache stores items in dense, fixed-size slots within contiguous
// memory blocks, eliminating:
//   - Heap allocator metadata (16+ bytes per allocation on modern allocators)
//   - Hash map node overhead (key-value pair malloc overhead)
//   - Pointer chasing and cache misses from fragmented allocations
//
// Memory savings: up to 50-70% for small items (int→int, small strings)
//
// Tradeoffs:
//   + Dramatically less memory for small items
//   + Better cache locality (contiguous slots)
//   + No per-item heap allocation/free overhead
//   - Fixed max item size (configured at construction)
//   - Items are copied (no move semantics for in-place storage)
//   - Slower for large items (use regular cache in parallel for mixed workloads)

#ifndef LRU_COMPACT_CACHE_HPP
#define LRU_COMPACT_CACHE_HPP

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <shared_mutex>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "ankerl/unordered_dense.h"
#include "cache_trait.hpp"
#include "core.hpp"
#include "detail/distributed_mutex.hpp"
#include "detail/intrusive_list.hpp"

namespace lru {

// ============================================================================
// Compact Slot (fixed-size storage)
// ============================================================================

/// A single slot in the compact cache. Stores one key-value pair.
/// The slot is pre-sized; key and value are stored in-place via placement new.
/// @tparam kAlignment 对齐粒度，默认 alignof(std::max_align_t) 适合大多数场景。
///                    对 NUMA/高争用场景可设为 std::hardware_destructive_interference_size。
template <typename Key, typename Value,
          std::size_t kAlignment = alignof(std::max_align_t)>
struct alignas(kAlignment) compact_slot {
    // 嵌入侵入式钩子，取代手动的 prev/next/update_time/flags
    detail::intrusive_hook hook;

    // Inline storage for key and value (placement-new'd)
    alignas(Key) char key_storage[sizeof(Key)];
    alignas(Value) char value_storage[sizeof(Value)];

    // 仅标志 slot 是否被占用（hook.is_linked() = 是否在 LRU 链表中）
    bool occupied = false;

    // Accessors
    Key* key_ptr() noexcept { return reinterpret_cast<Key*>(key_storage); }
    const Key* key_ptr() const noexcept { return reinterpret_cast<const Key*>(key_storage); }
    Value* value_ptr() noexcept { return reinterpret_cast<Value*>(value_storage); }
    const Value* value_ptr() const noexcept { return reinterpret_cast<const Value*>(value_storage); }

    // Hook 访问器，供 intrusive_list 使用
    auto& get_hook() noexcept { return hook; }
    const auto& get_hook() const noexcept { return hook; }

    /// Estimated memory overhead per slot (excluding key+value).
    static constexpr std::size_t slot_overhead =
        sizeof(compact_slot) - sizeof(Key) - sizeof(Value);
};

// ============================================================================
// Compact Slot Allocator
// ============================================================================

/// Manages a pool of pre-allocated compact_slots.
/// Slots are allocated in blocks for contiguous memory access.
template <typename Key, typename Value, std::size_t kAlignment = alignof(std::max_align_t)>
class compact_slot_allocator {
public:
    using slot_type = compact_slot<Key, Value, kAlignment>;
    static constexpr std::size_t kSlotsPerBlock = 4096; // ~256 KB per block

    compact_slot_allocator() = default;

    ~compact_slot_allocator() {
        for (auto* block : blocks_) {
            for (std::size_t i = 0; i < kSlotsPerBlock; ++i) {
                auto* slot = &block[i];
                if (slot->occupied) {
                    slot->key_ptr()->~Key();
                    slot->value_ptr()->~Value();
                }
            }
            ::operator delete(block);
        }
    }

    /// Pre-allocate slots for expected capacity.
    void reserve(std::size_t count) {
        while (total_slots_ < count) {
            allocate_block();
        }
    }

    /// Allocate a single slot.
    slot_type* allocate() {
        if (free_list_.empty()) {
            allocate_block();
        }
        auto* slot = free_list_.back();
        free_list_.pop_back();
        slot->occupied = false;
        slot->hook.prev = nullptr;
        slot->hook.next = nullptr;
        slot->hook.update_time = 0;
        slot->hook.flags = 0;
        used_slots_++;
        return slot;
    }

    /// Deallocate a slot (return to free list).
    void deallocate(slot_type* slot) {
        assert(slot != nullptr);
        if (slot->occupied) {
            slot->key_ptr()->~Key();
            slot->value_ptr()->~Value();
            slot->occupied = false;
        }
        free_list_.push_back(slot);
        used_slots_--;
    }

    std::size_t used() const noexcept { return used_slots_; }
    std::size_t total() const noexcept { return total_slots_; }

private:
    void allocate_block() {
        auto* block = static_cast<slot_type*>(
            ::operator new(kSlotsPerBlock * sizeof(slot_type)));
        blocks_.push_back(block);

        for (std::size_t i = 0; i < kSlotsPerBlock; ++i) {
            auto* slot = &block[i];
            // Must initialize occupied=false so the allocator's destructor
            // does not try to call ~Key()/~Value() on uninitialized storage.
            slot->occupied = false;
            free_list_.push_back(slot);
        }
        total_slots_ += kSlotsPerBlock;
    }

    std::vector<slot_type*> blocks_;
    std::vector<slot_type*> free_list_;
    std::size_t total_slots_ = 0;
    std::size_t used_slots_ = 0;
};

// ============================================================================
// Compact Cache
// ============================================================================

/// A dense, memory-efficient cache for small key-value pairs.
///
/// Thread safety: Controlled by ThreadPolicy template parameter.
///   - single_threaded_policy (default): NOT thread-safe. All access must be
///     externally synchronized.
///   - thread_safe_policy: Thread-safe using distributed_shared_mutex for
///     writes and striped_mutex<distributed_shared_mutex> for per-key reads.
///     Write operations (set, add, get, del, flush, etc.) acquire an exclusive
///     lock on a single distributed_shared_mutex, serializing all mutations.
///     Per-key read operations (peek, contains) acquire a shared lock on
///     the key's stripe via the striped_mutex, allowing concurrent reads across
///     different stripes.
///     Global read operations (size, empty, etc.) acquire a shared lock on
///     the distributed_shared_mutex, allowing concurrent reads with each other
///     but blocking during writes.
///
/// Items must satisfy: sizeof(Key) + sizeof(Value) <= kMaxItemSize
/// Default max item size is 64 bytes (configurable via template parameter).
///
/// Usage:
///   compact_cache<int, int> cache(/*max_size=*/10'000'000);
///   cache.set(42, 123);
///   auto v = cache.get(42); // std::optional<std::reference_wrapper<int>>
///
///   // Thread-safe variant:
///   compact_cache<int, int, std::hash<int>, std::equal_to<int>, 64,
///                 alignof(std::max_align_t), thread_safe_policy> safe_cache(1000);
template <
    typename Key,
    typename Value,
    typename Hash = std::hash<Key>,
    typename KeyEqual = std::equal_to<Key>,
    std::size_t kMaxItemSize = 64,
    std::size_t kSlotAlignment = alignof(std::max_align_t),
    typename ThreadPolicy = single_threaded_policy
>
class compact_cache {
    static_assert(sizeof(Key) + sizeof(Value) <= kMaxItemSize,
                  "compact_cache: key + value exceeds kMaxItemSize");

public:
    using key_type = Key;
    using mapped_type = Value;
    using size_type = std::size_t;
    using slot_type = compact_slot<Key, Value, kSlotAlignment>;
    using thread_policy = ThreadPolicy;
    // 使用 get_hook 函数指针（隐式转换自 default_get_hook<slot_type>）
    static detail::intrusive_hook& get_slot_hook(slot_type& s) noexcept { return s.get_hook(); }
    using slot_list = detail::intrusive_list<slot_type, detail::intrusive_hook, get_slot_hook>;

    using map_type = ankerl::unordered_dense::map<Key, slot_type*, Hash, KeyEqual>;
    using stats_type = cache_stats;
    using callback_mgr = callback_manager<Key, Value>;

    static constexpr size_type npos = unlimited;
    static constexpr bool is_thread_safe = ThreadPolicy::is_thread_safe;
    static constexpr bool is_striped = is_striped_policy_v<ThreadPolicy>;

private:
    // Conditional mutex storage: only present for thread-safe policies.
    // Uses [[no_unique_address]] so that single-threaded caches pay zero overhead.
    // Uses distributed_shared_mutex for stripe locks, providing shared (read) /
    // exclusive (write) semantics without relying on MinGW's buggy pthread_rwlock_t.
    using striped_mutex_storage = std::conditional_t<
        is_thread_safe,
        detail::striped_mutex<detail::distributed_shared_mutex>,
        std::tuple<>
    >;


public:

    // --------------------------------------------------------------------
    // Construction
    // --------------------------------------------------------------------

    compact_cache() = default;

    explicit compact_cache(size_type max_size)
        : max_size_(max_size) {
        allocator_.reserve(std::min(max_size, size_type(4096)));
        stats_.max_size.store(max_size);
    }

    compact_cache(size_type max_size, size_type max_memory)
        : max_size_(max_size), max_memory_(max_memory) {
        allocator_.reserve(std::min(max_size, size_type(4096)));
        stats_.max_size.store(max_size);
        stats_.max_memory.store(max_memory);
    }

    ~compact_cache() {
        // 使用 pop_tail 安全销毁所有 slot（复用 intrusive_list 的 ASAN/TSAN 支持）
        while (auto* slot = lru_list_.pop_tail()) {
            destroy_slot(slot);
        }
    }

    compact_cache(const compact_cache&) = delete;
    compact_cache& operator=(const compact_cache&) = delete;

    // --------------------------------------------------------------------
    // Core API
    // --------------------------------------------------------------------

    template <typename V>
    void set(const key_type& key, V&& value) {
        if (shutdown_.load(std::memory_order_acquire)) {
            throw std::logic_error("compact_cache::set: cache is shutdown");
        }
        auto lock = acquire_write_lock();
        auto it = map_.find(key);
        if (it != map_.end()) {
            // Update existing
            *it->second->value_ptr() = std::forward<V>(value);
            record_access(it->second);
            return;
        }

        // Evict if at capacity
        while (size_unlocked() >= max_size_ && max_size_ != npos) {
            evict_lru();
        }
        while (should_evict_memory() && !map_.empty()) {
            evict_lru();
        }

        // Allocate new slot
        auto* slot = allocator_.allocate();
        ::new (slot->key_ptr()) Key(key);
        ::new (slot->value_ptr()) Value(std::forward<V>(value));
        slot->occupied = true;
        slot->hook.update_time = current_time_sec();
        slot->hook.clear_accessed();

        // Link at head (MRU) via intrusive_list
        lru_list_.link_at_head(*slot);

        map_[key] = slot;

        stats_.current_size.store(size_unlocked());
        current_memory_ += calc_item_memory(key, *slot->value_ptr());
        stats_.current_memory.store(current_memory_);
        stats_.register_insertion();
        callbacks_.collect_insert(key, *slot->value_ptr());
        callbacks_.flush_pending();
        // T13/T14: monitor hash table load factor for overload events.
        check_hash_overload_locked();
    }

    template <typename V>
    bool add(const key_type& key, V&& value) {
        if (shutdown_.load(std::memory_order_acquire)) {
            throw std::logic_error("compact_cache::add: cache is shutdown");
        }
        auto lock = acquire_write_lock();
        if (map_.contains(key)) return false;
        // Already holding write lock; call unlocked implementation.
        set_unlocked(key, std::forward<V>(value));
        return true;
    }

    /// Get a value by key. Returns a reference to the in-cache value.
    /// WARNING: The returned reference is valid only as long as no mutating
    /// operation (set, del, flush, etc.) is performed on the cache. For
    /// thread-safe access or longer-lived references, use a different cache type.
    std::optional<std::reference_wrapper<Value>> get(const key_type& key) {
        auto lock = acquire_write_lock();
        auto it = map_.find(key);
        if (it == map_.end()) {
            stats_.register_miss();
            callbacks_.collect_miss(key);
            callbacks_.flush_pending();
            return std::nullopt;
        }
        record_access(it->second);
        stats_.register_hit();
        callbacks_.collect_hit(key, *it->second->value_ptr());
        callbacks_.flush_pending();
        return std::ref(*it->second->value_ptr());
    }

    /// Peek without promoting.
    std::optional<std::reference_wrapper<const Value>> peek(const key_type& key) const {
        auto lock = acquire_read_lock_for_key(key);
        auto it = map_.find(key);
        if (it == map_.end()) return std::nullopt;
        return std::cref(*it->second->value_ptr());
    }

    bool del(const key_type& key) {
        if (shutdown_.load(std::memory_order_acquire)) {
            throw std::logic_error("compact_cache::del: cache is shutdown");
        }
        auto lock = acquire_write_lock();
        auto it = map_.find(key);
        if (it == map_.end()) return false;
        erase_slot(it);
        callbacks_.flush_pending();
        return true;
    }

    bool contains(const key_type& key) const {
        auto lock = acquire_read_lock_for_key(key);
        return map_.contains(key);
    }

    void flush() {
        if (shutdown_.load(std::memory_order_acquire)) {
            throw std::logic_error("compact_cache::flush: cache is shutdown");
        }
        auto lock = acquire_write_lock();
        while (auto* slot = lru_list_.pop_tail()) {
            destroy_slot(slot);
        }
        map_.clear();
        current_memory_ = 0;
        stats_.current_size.store(0);
        stats_.current_memory.store(0);
        callbacks_.flush_pending();
    }

    // --------------------------------------------------------------------
    // Capacity
    // --------------------------------------------------------------------

    bool empty() const {
        auto lock = acquire_read_lock();
        return map_.empty();
    }
    size_type size() const {
        auto lock = acquire_read_lock();
        return map_.size();
    }
    size_type max_size() const {
        auto lock = acquire_read_lock();
        return max_size_;
    }
    size_type max_memory() const {
        auto lock = acquire_read_lock();
        return max_memory_;
    }
    size_type current_memory() const {
        auto lock = acquire_read_lock();
        return current_memory_;
    }

    void max_size(size_type new_max) {
        auto lock = acquire_write_lock();
        max_size_ = new_max;
        stats_.max_size.store(new_max);
        shrink_to_fit_unlocked();
    }

    void shrink_to_fit() {
        auto lock = acquire_write_lock();
        shrink_to_fit_unlocked();
    }

    // --------------------------------------------------------------------
    // Statistics and callbacks
    // --------------------------------------------------------------------

    stats_type& stats() noexcept { return stats_; }
    const stats_type& stats() const noexcept { return stats_; }
    callback_mgr& callbacks() noexcept { return callbacks_; }
    const callback_mgr& callbacks() const noexcept { return callbacks_; }

    /// 设置访问记录的刷新间隔（秒），默认 60 秒
    void set_refresh_time(uint32_t seconds) {
        auto lock = acquire_write_lock();
        refresh_time_ = seconds;
    }

    // ========================================================================
    // T14: Production-grade observability & controllability APIs
    //
    // These methods mirror the unified_cache production API surface so that
    // compact_cache can be dropped into the same monitoring/control pipeline
    // as unified_cache (stats scraping, Prometheus export, runtime tuning).
    // The implementations are intentionally lightweight — compact_cache uses
    // a single hash map + intrusive list, so per-shard aggregation is trivial
    // (single shard).
    // ========================================================================

    /// Consistent point-in-time snapshot of cache statistics.
    /// Safe to call concurrently with cache operations (lock-free atomic reads
    /// via cache_stats' SeqLock). Callers should invoke this infrequently
    /// (e.g. once per scrape interval) since it performs a SeqLock read.
    stats_type stats_snapshot() const {
        return stats_.consistent_snapshot();
    }

    /// Multi-line human-readable diagnostics dump. Useful for ad-hoc
    /// inspection via debugger or log. For Prometheus scraping use
    /// prometheus_text() instead.
    struct diagnostics_info {
        std::size_t size{0};
        std::size_t max_size{0};
        std::size_t current_memory{0};
        std::size_t max_memory{0};
        std::size_t slot_total{0};        // allocator total slots
        std::size_t slot_used{0};         // allocator used slots
        std::size_t bucket_count{0};      // hash table bucket count
        float load_factor{0.0f};
        bool latency_tracking_enabled{true};
        bool async_callbacks_enabled{false};
        bool is_thread_safe{false};
        bool is_striped{false};
        bool shutdown_in_progress{false};
        std::size_t hits{0};
        std::size_t misses{0};
        std::size_t insertions{0};
        std::size_t evictions{0};
    };

    diagnostics_info diagnostics() const {
        diagnostics_info info;
        auto snap = stats_.consistent_snapshot();
        info.size = snap.current_size.load(std::memory_order_relaxed);
        info.max_size = snap.max_size.load(std::memory_order_relaxed);
        info.current_memory = snap.current_memory.load(std::memory_order_relaxed);
        info.max_memory = snap.max_memory.load(std::memory_order_relaxed);
        info.slot_total = allocator_.total();
        info.slot_used = allocator_.used();
        info.bucket_count = map_.bucket_count();
        info.load_factor = map_.load_factor();
        info.latency_tracking_enabled =
            snap.latency_tracking_enabled.load(std::memory_order_relaxed);
        info.async_callbacks_enabled = callbacks_.is_async_mode();
        info.is_thread_safe = is_thread_safe;
        info.is_striped = is_striped;
        info.shutdown_in_progress = shutdown_.load(std::memory_order_acquire);
        info.hits = snap.hits.value.load(std::memory_order_relaxed);
        info.misses = snap.misses.value.load(std::memory_order_relaxed);
        info.insertions = snap.insertions.value.load(std::memory_order_relaxed);
        info.evictions = snap.evictions.value.load(std::memory_order_relaxed);
        return info;
    }

    std::string diagnostics_text() const {
        auto info = diagnostics();
        std::string out;
        out.reserve(1024);

        auto append = [&](std::string_view line) {
            out.append(line);
            out.push_back('\n');
        };
        auto append_kv = [&](std::string_view key, std::size_t val) {
            out.append(key);
            out.append(": ");
            out.append(std::to_string(val));
            out.push_back('\n');
        };

        append("=== compact_cache diagnostics ===");
        append_kv("size", info.size);
        append_kv("max_size", info.max_size);
        append_kv("current_memory_bytes", info.current_memory);
        append_kv("max_memory_bytes", info.max_memory);
        append_kv("slot_total", info.slot_total);
        append_kv("slot_used", info.slot_used);
        append_kv("bucket_count", info.bucket_count);

        char lf_buf[32];
        std::snprintf(lf_buf, sizeof(lf_buf), "load_factor: %.4f\n", info.load_factor);
        out.append(lf_buf);

        append_kv("hits", info.hits);
        append_kv("misses", info.misses);
        append_kv("insertions", info.insertions);
        append_kv("evictions", info.evictions);
        append_kv("latency_tracking_enabled", info.latency_tracking_enabled ? 1 : 0);
        append_kv("async_callbacks_enabled", info.async_callbacks_enabled ? 1 : 0);
        append_kv("is_thread_safe", info.is_thread_safe ? 1 : 0);
        append_kv("is_striped", info.is_striped ? 1 : 0);
        append_kv("shutdown_in_progress", info.shutdown_in_progress ? 1 : 0);
        append("=== end compact_cache diagnostics ===");
        return out;
    }

    /// Prometheus exposition format. Mirrors a subset of the unified_cache
    /// metrics, prefixed with `lru_compact_` to distinguish from regular
    /// caches when both are scraped from the same process.
    std::string prometheus_text() const {
        auto snap = stats_snapshot();
        std::string out;
        out.reserve(2048);

        auto append = [&](std::string_view line) {
            out.append(line);
            out.push_back('\n');
        };

        // Counters
        append("# HELP lru_compact_hits_total Total cache hits.");
        append("# TYPE lru_compact_hits_total counter");
        out.append("lru_compact_hits_total ");
        out.append(std::to_string(snap.hits.value.load(std::memory_order_relaxed)));
        out.push_back('\n');

        append("# HELP lru_compact_misses_total Total cache misses.");
        append("# TYPE lru_compact_misses_total counter");
        out.append("lru_compact_misses_total ");
        out.append(std::to_string(snap.misses.value.load(std::memory_order_relaxed)));
        out.push_back('\n');

        append("# HELP lru_compact_insertions_total Total item insertions.");
        append("# TYPE lru_compact_insertions_total counter");
        out.append("lru_compact_insertions_total ");
        out.append(std::to_string(snap.insertions.value.load(std::memory_order_relaxed)));
        out.push_back('\n');

        append("# HELP lru_compact_evictions_total Total item evictions.");
        append("# TYPE lru_compact_evictions_total counter");
        out.append("lru_compact_evictions_total ");
        out.append(std::to_string(snap.evictions.value.load(std::memory_order_relaxed)));
        out.push_back('\n');

        // Gauges
        append("# HELP lru_compact_size Current number of items in the cache.");
        append("# TYPE lru_compact_size gauge");
        out.append("lru_compact_size ");
        out.append(std::to_string(snap.current_size.load(std::memory_order_relaxed)));
        out.push_back('\n');

        append("# HELP lru_compact_memory_bytes Current memory used by cached items.");
        append("# TYPE lru_compact_memory_bytes gauge");
        out.append("lru_compact_memory_bytes ");
        out.append(std::to_string(snap.current_memory.load(std::memory_order_relaxed)));
        out.push_back('\n');

        append("# HELP lru_compact_max_size Maximum number of items allowed.");
        append("# TYPE lru_compact_max_size gauge");
        out.append("lru_compact_max_size ");
        out.append(std::to_string(snap.max_size.load(std::memory_order_relaxed)));
        out.push_back('\n');

        append("# HELP lru_compact_max_memory_bytes Maximum memory budget (bytes).");
        append("# TYPE lru_compact_max_memory_bytes gauge");
        out.append("lru_compact_max_memory_bytes ");
        out.append(std::to_string(snap.max_memory.load(std::memory_order_relaxed)));
        out.push_back('\n');

        // Slot allocator gauges (compact_cache specific)
        {
            auto lock = acquire_read_lock();
            append("# HELP lru_compact_slot_total Total slots allocated by the slab allocator.");
            append("# TYPE lru_compact_slot_total gauge");
            out.append("lru_compact_slot_total ");
            out.append(std::to_string(allocator_.total()));
            out.push_back('\n');

            append("# HELP lru_compact_slot_used Slots currently holding live items.");
            append("# TYPE lru_compact_slot_used gauge");
            out.append("lru_compact_slot_used ");
            out.append(std::to_string(allocator_.used()));
            out.push_back('\n');

            append("# HELP lru_compact_bucket_count Hash table bucket count.");
            append("# TYPE lru_compact_bucket_count gauge");
            out.append("lru_compact_bucket_count ");
            out.append(std::to_string(map_.bucket_count()));
            out.push_back('\n');

            char lf_help[64];
            std::snprintf(lf_help, sizeof(lf_help), "lru_compact_load_factor %.4f\n",
                          static_cast<double>(map_.load_factor()));
            append("# HELP lru_compact_load_factor Hash table load factor.");
            append("# TYPE lru_compact_load_factor gauge");
            out.append(lf_help);
        }

        // Configuration flags
        append("# HELP lru_compact_latency_tracking_enabled 1 if latency histogram tracking is on.");
        append("# TYPE lru_compact_latency_tracking_enabled gauge");
        out.append("lru_compact_latency_tracking_enabled ");
        out.append(std::to_string(
            snap.latency_tracking_enabled.load(std::memory_order_relaxed) ? 1 : 0));
        out.push_back('\n');

        append("# HELP lru_compact_shutdown 1 if the cache is shutting down (rejecting new ops).");
        append("# TYPE lru_compact_shutdown gauge");
        out.append("lru_compact_shutdown ");
        out.append(std::to_string(
            shutdown_.load(std::memory_order_acquire) ? 1 : 0));
        out.push_back('\n');

        return out;
    }

    // --------------------------------------------------------------------
    // Runtime configuration (mirrors unified_cache production API)
    // --------------------------------------------------------------------

    /// Switch the writer/reader fairness policy at runtime.
    /// Only meaningful for thread-safe variants; single-threaded variants
    /// have no mutex and silently ignore the call.
    void set_fairness_mode(detail::fairness_mode mode) {
        if constexpr (is_thread_safe) {
            write_mutex_.set_fairness_mode(mode);
            // striped_mutex<>::set_fairness_mode is a SFINAEd template that
            // is a no-op for mutex types without fairness support.
            if constexpr (requires { striped_mutex_.set_fairness_mode(mode); }) {
                striped_mutex_.set_fairness_mode(mode);
            }
        }
        (void)mode;
    }

    detail::fairness_mode get_fairness_mode() const noexcept {
        if constexpr (is_thread_safe) {
            return write_mutex_.get_fairness_mode();
        } else {
            return detail::fairness_mode::writer_fair;
        }
    }

    /// Enable or disable latency histogram tracking on the hot path.
    /// When disabled, scope_latency_timer skips clock reads entirely.
    void set_latency_tracking(bool enabled) {
        const bool was_enabled =
            stats_.latency_tracking_enabled.load(std::memory_order_acquire);
        stats_.latency_tracking_enabled.store(enabled, std::memory_order_release);
        // T12.3 parity: on disable, release histogram data.
        if (was_enabled && !enabled) {
            stats_.get_latency.release_memory();
            stats_.set_latency.release_memory();
            stats_.read_lock_wait_latency.release_memory();
            stats_.write_lock_wait_latency.release_memory();
            stats_.eviction_search_steps_hist.release_memory();
        }
    }

    bool is_latency_tracking_enabled() const noexcept {
        return stats_.latency_tracking_enabled.load(std::memory_order_acquire);
    }

    /// Toggle async callback mode. When enabled, drained callbacks are
    /// enqueued to a dedicated worker thread instead of being invoked
    /// inline on the caller. This prevents user callbacks (IO/logging)
    /// from blocking the drain path.
    void set_async_callbacks(bool enabled) {
        callbacks_.set_async_mode(enabled);
    }

    bool is_async_callbacks_enabled() const noexcept {
        return callbacks_.is_async_mode();
    }

    // --------------------------------------------------------------------
    // Graceful shutdown (mirrors unified_cache production API)
    // --------------------------------------------------------------------

    /// Mark the cache as shutting down. Subsequent mutating operations
    /// (set/add/del/flush) will throw std::logic_error. Reads (get/peek/
    /// contains) remain permitted so in-flight work can drain.
    ///
    /// Note: compact_cache returns references (not read_handle) from get(),
    /// so callers must ensure no outstanding references are used after
    /// shutdown completes. active_handle_count() is always 0 for
    /// compact_cache (no refcounted handles).
    void shutdown() noexcept {
        shutdown_.store(true, std::memory_order_release);
    }

    bool is_shutdown() const noexcept {
        return shutdown_.load(std::memory_order_acquire);
    }

    /// Number of outstanding read_handle objects pinning items.
    /// compact_cache does not use refcounted handles (get() returns a
    /// reference_wrapper), so this is always 0. Provided for API
    /// compatibility with unified_cache.
    std::size_t active_handle_count() const noexcept { return 0; }

    // --------------------------------------------------------------------
    // Hash overload configuration (T13 parity)
    //
    // compact_cache uses ankerl::unordered_dense::map (not our
    // concurrent_hash_table), so the threshold and callback are stored
    // but only consulted on set()/add() to fire user callbacks when the
    // load factor crosses the configured threshold. This mirrors the
    // T13 API surface so monitoring pipelines can subscribe to overload
    // events uniformly across cache types.
    // --------------------------------------------------------------------

    void set_hash_overload_threshold(float threshold) noexcept {
        hash_overload_threshold_.store(threshold, std::memory_order_release);
    }

    float hash_overload_threshold() const noexcept {
        return hash_overload_threshold_.load(std::memory_order_acquire);
    }

    std::size_t hash_overload_events() const noexcept {
        return hash_overload_events_.load(std::memory_order_acquire);
    }

    void set_overload_callback(std::function<void(float, float)> cb) {
        auto lock = acquire_write_lock();
        hash_overload_callback_ = std::move(cb);
    }

private:
    /// T13: invoked under the write lock on each set()/add() to detect
    /// load-factor overloads. Returns true if an overload event fired.
    bool check_hash_overload_locked() {
        const float threshold = hash_overload_threshold_.load(std::memory_order_acquire);
        if (threshold <= 0.0f) return false;
        const float current_lf = map_.load_factor();
        if (current_lf <= threshold) return false;
        hash_overload_events_.fetch_add(1, std::memory_order_relaxed);
        if (hash_overload_callback_) {
            try {
                hash_overload_callback_(current_lf, threshold);
            } catch (...) {
                // Swallow user callback exceptions to preserve cache integrity.
            }
        }
        return true;
    }

public:
    // ========================================================================
    // Memory savings estimation
    // ========================================================================

    /// Compare memory usage between compact_cache and regular mm_lru.
    static constexpr std::size_t per_item_memory_savings() {
        // Regular LRU: map entry (~40) + cache_item (key+value+hook+ptrs ~sizeof(KV)+40)
        // Compact:     map entry (~24) + slot_overhead (no separate allocation)
        std::size_t regular_overhead = sizeof(typename map_type::value_type) +
                                       slot_type::slot_overhead + 16 /* heap overhead */;
        std::size_t compact_overhead = sizeof(typename map_type::value_type) +
                                       slot_type::slot_overhead;
        return regular_overhead > compact_overhead
                   ? regular_overhead - compact_overhead : 0;
    }

private:
    // --------------------------------------------------------------------
    // Helpers
    // --------------------------------------------------------------------

    /// 延迟提升（通过 intrusive_hook 的 update_time）
    void record_access(slot_type* slot) {
        auto curr = current_time_sec();
        if (curr < slot->hook.update_time + refresh_time_) return;
        // 使用 intrusive_list 的 move_to_head 代替手动链表操作
        lru_list_.move_to_head(*slot);
        slot->hook.update_time = curr;
        slot->hook.set_accessed();
    }

    void evict_lru() {
        auto* victim = lru_list_.tail();
        if (!victim) return;
        auto map_it = map_.find(*victim->key_ptr());
        assert(map_it != map_.end());
        erase_slot(map_it);
        stats_.register_eviction();
    }

    void erase_slot(typename map_type::iterator map_it) {
        auto* slot = map_it->second;
        Value value = std::move(*slot->value_ptr());
        auto key = *slot->key_ptr();

        current_memory_ -= calc_item_memory(key, value);
        stats_.current_memory.store(current_memory_);
        callbacks_.collect_evict(key, std::move(value));

        // Destroy slot contents BEFORE remove() poisons the slot's memory.
        slot->key_ptr()->~Key();
        slot->value_ptr()->~Value();
        slot->occupied = false;

        lru_list_.remove(*slot);
        map_.erase(map_it);
        allocator_.deallocate(slot);
        stats_.current_size.store(size_unlocked());
    }

    void destroy_slot(slot_type* slot) {
        slot->key_ptr()->~Key();
        slot->value_ptr()->~Value();
        slot->occupied = false;
        allocator_.deallocate(slot);
    }

    bool should_evict_memory() const {
        return max_memory_ != npos && current_memory_ > max_memory_;
    }

    size_type calc_item_memory(const key_type& key, const mapped_type& value) const {
        (void)key; (void)value;
        return sizeof(slot_type);
    }

    static uint32_t current_time_sec() {
        return static_cast<uint32_t>(
            std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch()).count());
    }

    // Unlocked variants for internal use (caller must already hold the lock)

    /// Size without acquiring lock (caller must hold a lock).
    size_type size_unlocked() const noexcept { return map_.size(); }

    /// set() implementation without lock acquisition (caller must hold write lock).
    template <typename V>
    void set_unlocked(const key_type& key, V&& value) {
        auto it = map_.find(key);
        if (it != map_.end()) {
            *it->second->value_ptr() = std::forward<V>(value);
            record_access(it->second);
            return;
        }

        while (size_unlocked() >= max_size_ && max_size_ != npos) {
            evict_lru();
        }
        while (should_evict_memory() && !map_.empty()) {
            evict_lru();
        }

        auto* slot = allocator_.allocate();
        ::new (slot->key_ptr()) Key(key);
        ::new (slot->value_ptr()) Value(std::forward<V>(value));
        slot->occupied = true;
        slot->hook.update_time = current_time_sec();
        slot->hook.clear_accessed();

        lru_list_.link_at_head(*slot);
        map_[key] = slot;

        stats_.current_size.store(size_unlocked());
        current_memory_ += calc_item_memory(key, *slot->value_ptr());
        stats_.current_memory.store(current_memory_);
        stats_.register_insertion();
        callbacks_.collect_insert(key, *slot->value_ptr());
        callbacks_.flush_pending();
        // T13/T14: monitor hash table load factor for overload events.
        check_hash_overload_locked();
    }

    /// shrink_to_fit without lock acquisition (caller must hold write lock).
    void shrink_to_fit_unlocked() {
        while (size_unlocked() > max_size_ && max_size_ != npos) {
            evict_lru();
        }
        callbacks_.flush_pending();
    }

    // --------------------------------------------------------------------
    // Lock infrastructure (thread-safe variant)
    // --------------------------------------------------------------------

    /// Write mutex: a distributed_shared_mutex for all operations.
    /// This serializes mutations (set, add, get, del, flush, etc.) with
    /// exclusive locks, while global reads use shared locks.
    /// Per-key reads use the striped_mutex for concurrency.
    /// For single-threaded policies, this is an empty tuple (zero overhead).
    /// Uses distributed_shared_mutex instead of std::shared_mutex to avoid
    /// MinGW winpthreads pthread_rwlock_t concurrency bug.
    using write_mutex_storage = std::conditional_t<
        is_thread_safe,
        detail::distributed_shared_mutex,
        std::tuple<>
    >;

    /// Acquire an exclusive write lock. For thread-safe policies, locks the
    /// mutex exclusively. For single-threaded policies, returns noop_lock.
    auto acquire_write_lock() const {
        if constexpr (is_thread_safe) {
            return std::unique_lock<detail::distributed_shared_mutex>(write_mutex_);
        } else {
            return noop_lock{};
        }
    }

    /// Acquire a shared read lock (global). For thread-safe policies, acquires
    /// a shared lock on the distributed_shared_mutex, allowing concurrent reads.
    /// For single-threaded policies, returns noop_lock.
    auto acquire_read_lock() const {
        if constexpr (is_thread_safe) {
            return std::shared_lock<detail::distributed_shared_mutex>(write_mutex_);
        } else {
            return noop_lock{};
        }
    }

    /// Acquire a shared read lock for a specific key's stripe.
    /// This allows concurrent per-key reads across different stripes.
    /// For single-threaded policies, returns a noop_lock.
    auto acquire_read_lock_for_key(const key_type& key) const {
        if constexpr (is_thread_safe) {
            auto hash = Hash{}(key);
            auto stripe = striped_mutex_.stripe_for(hash);
            return striped_mutex_.make_shared_lock(stripe);
        } else {
            return noop_lock{};
        }
    }

    // --- Members ---
    // 使用 intrusive_list 接管 LRU 链表管理，复用其 ASAN/TSAN 集成
    slot_list lru_list_;

    map_type map_;
    compact_slot_allocator<Key, Value, kSlotAlignment> allocator_;

    /// Write mutex for thread safety. Empty tuple for single-threaded policies.
    [[no_unique_address]] mutable write_mutex_storage write_mutex_{};

    /// Striped mutex for per-key read operations. Empty tuple for single-threaded
    /// policies. Default-constructed with 64 stripes for thread-safe variants.
    [[no_unique_address]] mutable striped_mutex_storage striped_mutex_{};

    size_type max_size_ = npos;
    size_type max_memory_ = npos;
    size_type current_memory_ = 0;

    stats_type stats_;
    callback_mgr callbacks_;
    uint32_t refresh_time_ = 60;

    // T14: graceful shutdown flag. When true, mutating operations reject
    // new work. Reads remain permitted so in-flight handles can drain.
    alignas(64) std::atomic<bool> shutdown_{false};

    // T13/T14 parity: hash overload detection. compact_cache uses
    // ankerl::unordered_dense::map; we monitor its load factor on writes.
    alignas(64) std::atomic<float> hash_overload_threshold_{2.0f};
    alignas(64) std::atomic<std::size_t> hash_overload_events_{0};
    std::function<void(float, float)> hash_overload_callback_;
};

// ============================================================================
// Thread-safe convenience alias
// ============================================================================

/// Thread-safe compact cache convenience alias.
/// Uses compact_cache with thread_safe_policy, which provides:
///   - distributed_shared_mutex for write serialization and global read concurrency
///   - striped_mutex<distributed_shared_mutex> for per-key read concurrency
///
/// Read operations (peek, contains) acquire a shared lock on the key's stripe,
/// allowing concurrent reads across different stripes.
/// Write operations (set, add, get, del, flush) acquire an exclusive lock on
/// the distributed_shared_mutex, serializing all mutations.
/// Global read operations (size, empty, max_size, etc.) acquire a shared lock
/// on the distributed_shared_mutex, allowing concurrent reads with each other
/// but blocking during writes.
///
/// 用法：lru::safe_compact_cache<int, int> c(10000);
template <typename Key, typename Value,
          typename Hash = std::hash<Key>,
          typename KeyEqual = std::equal_to<Key>,
          std::size_t kMaxItemSize = 64,
          std::size_t kSlotAlignment = alignof(std::max_align_t)>
using safe_compact_cache = compact_cache<Key, Value, Hash, KeyEqual,
                                         kMaxItemSize, kSlotAlignment,
                                         thread_safe_policy>;

/// T14: Striped thread-safe compact cache convenience alias.
///
/// Uses compact_cache with striped_thread_safe_policy, which inherits the
/// same distributed_shared_mutex + striped_mutex layout as safe_compact_cache
/// (compact_cache's lock acquisition is identical for both thread-safe
/// policies). The striped policy is provided for API symmetry with
/// unified_cache's striped_cache / production_cache aliases so that
/// monitoring code can treat compact_cache as a drop-in replacement.
///
/// 用法：lru::striped_compact_cache<int, int> c(10000);
template <typename Key, typename Value,
          typename Hash = std::hash<Key>,
          typename KeyEqual = std::equal_to<Key>,
          std::size_t kMaxItemSize = 64,
          std::size_t kSlotAlignment = alignof(std::max_align_t)>
using striped_compact_cache = compact_cache<Key, Value, Hash, KeyEqual,
                                            kMaxItemSize, kSlotAlignment,
                                            striped_thread_safe_policy<>>;

} // namespace lru

#endif // LRU_COMPACT_CACHE_HPP
