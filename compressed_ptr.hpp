// SPDX-License-Identifier: MIT
// Compressed Pointer — Inspired by CacheLib's CompressedPtr
//
// In a 64-bit address space, cache item hooks carry two pointers (prev/next)
// consuming 16 bytes per item. For caches with millions of items, this adds
// significant overhead. Compressed pointers store a 32-bit offset relative to
// a base address, reducing pointer overhead by 50%.
//
// This implementation provides:
//   - compressed_intrusive_hook: intrusive hook using 32-bit offsets
//   - hook_pointer_traits<compressed_intrusive_hook>: bridges to intrusive_list
//   - compressed_region: contiguous memory allocator ensuring 4GB addressability
//   - compressed_ptr<T>: standalone compressed pointer utility
//
// Memory savings vs intrusive_hook:
//   - Raw hook:      16 bytes (prev + next as void*)
//   - Compressed hook: 8 bytes (prev + next as uint32_t offsets)
//   - For 10M items:  saves ~76 MB + better cache locality
//
// Usage (manual template selection):
//   lru::compressed_region region(1'000'000, sizeof(my_item));
//   using my_item = lru::detail::cache_item<K, V, lru::compressed_intrusive_hook>;
//   using my_list = lru::detail::intrusive_list<my_item, lru::compressed_intrusive_hook, ...>;
//   my_list list(region.base());
//   auto* item = region.allocate<my_item>(key, value);
//   list.link_at_head(*item);

#ifndef LRU_COMPRESSED_PTR_HPP
#define LRU_COMPRESSED_PTR_HPP

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "detail/intrusive_list.hpp"  // cache_item, used by the allocator below

namespace lru {

// Forward declaration: detail::intrusive_hook is used by the traits specialisations.
namespace detail { struct intrusive_hook; }

// ============================================================================
// Compressed Pointer
// ============================================================================

/// A 32-bit compressed pointer storing an offset from a base address.
/// Reduces pointer size from 8 bytes to 4 bytes on 64-bit systems.
///
/// @tparam T          The pointed-to type.
/// @tparam BasePtr    Type of the base address pointer (defaults to void*).
template <typename T, typename BasePtr = void*>
class compressed_ptr {
public:
    using value_type = T;
    using base_type = BasePtr;
    using offset_type = uint32_t;

    static constexpr offset_type kNullOffset = std::numeric_limits<offset_type>::max();
    static constexpr std::size_t kMaxOffset = kNullOffset - 1; // 4 GB - 1

    // --------------------------------------------------------------------
    // Construction
    // --------------------------------------------------------------------

    compressed_ptr() noexcept : offset_(kNullOffset) {}

    /// Construct from a raw pointer and base address.
    /// @param ptr       The raw pointer.
    /// @param base      The base address. All pointers must be >= base and within 4 GB.
    explicit compressed_ptr(T* ptr, base_type base)
        : offset_(compress(ptr, base)) {}

    // --------------------------------------------------------------------
    // Access
    // --------------------------------------------------------------------

    /// Get the raw pointer given a base address.
    T* get(base_type base) const noexcept {
        if (is_null()) return nullptr;
        auto base_addr = reinterpret_cast<uintptr_t>(base);
        return reinterpret_cast<T*>(base_addr + offset_);
    }

    /// Check if this pointer is null.
    bool is_null() const noexcept {
        return offset_ == kNullOffset;
    }

    /// Set to null.
    void set_null() noexcept {
        offset_ = kNullOffset;
    }

    /// Get the raw offset value.
    offset_type offset() const noexcept { return offset_; }

    /// Set from a raw pointer and base address.
    void set(T* ptr, base_type base) {
        offset_ = compress(ptr, base);
    }

    bool operator==(const compressed_ptr& other) const noexcept {
        return offset_ == other.offset_;
    }
    bool operator!=(const compressed_ptr& other) const noexcept {
        return offset_ != other.offset_;
    }

    /// Get the offset for serialization.
    offset_type save_state() const noexcept { return offset_; }

private:
    static offset_type compress(T* ptr, base_type base) {
        if (ptr == nullptr) return kNullOffset;
        auto base_addr = reinterpret_cast<uintptr_t>(base);
        auto ptr_addr = reinterpret_cast<uintptr_t>(ptr);
        if (ptr_addr < base_addr) {
            throw std::overflow_error("compressed_ptr: pointer is below base address");
        }
        auto diff = ptr_addr - base_addr;
        if (diff > kNullOffset - 1) {
            throw std::overflow_error("compressed_ptr: offset exceeds 4GB limit");
        }
        return static_cast<offset_type>(diff);
    }

    offset_type offset_;
};

// ============================================================================
// Compressed Intrusive Hook
// ============================================================================

/// An alternative to intrusive_hook that uses compressed pointers.
/// Same interface but uses 32-bit offsets for prev/next instead of 64-bit pointers.
///
/// Total size: 14 bytes (prev:4 + next:4 + update_time:4 + flags:1 + queue_id:1)
/// vs. original intrusive_hook: 24 bytes (prev:8 + next:8 + update_time:4 + flags:1 + pad:3)
///
/// To use it with intrusive_list, set Hook = compressed_intrusive_hook and point
/// intrusive_list's base at compressed_region::base().
struct alignas(8) compressed_intrusive_hook {
    // Marker used by has_compressed_hook trait detection.
    static constexpr bool hook_is_compressed = true;

    // Bit positions for flags (same as intrusive_hook)
    static constexpr uint8_t kTailFlag     = 1 << 0;
    static constexpr uint8_t kAccessedFlag = 1 << 1;
    static constexpr uint8_t kLinkedFlag   = 1 << 2;
    // pre-computed clear masks — see the note in detail/intrusive_list.hpp.
    static constexpr uint8_t kTailClearMask     = static_cast<uint8_t>(~kTailFlag);
    static constexpr uint8_t kAccessedClearMask = static_cast<uint8_t>(~kAccessedFlag);
    static constexpr uint8_t kLinkedClearMask   = static_cast<uint8_t>(~kLinkedFlag);

    /// Compressed offsets from the item's allocation base.
    uint32_t prev_offset = 0;
    uint32_t next_offset = 0;

    /// Seconds since epoch (for delayed promotion).
    uint32_t update_time = 0;

    /// Bit flags for tail/accessed tracking.
    uint8_t flags = 0;

    /// Queue ID (same role as intrusive_hook::queue_id).
    uint8_t queue_id = 0;

    // --- Null sentinel ---

    static constexpr uint32_t kNullOffset = std::numeric_limits<uint32_t>::max();

    bool is_prev_null() const noexcept { return prev_offset == kNullOffset; }
    bool is_next_null() const noexcept { return next_offset == kNullOffset; }
    void set_prev_null() noexcept { prev_offset = kNullOffset; }
    void set_next_null() noexcept { next_offset = kNullOffset; }

    // --- Pointer resolution ---

    /// Resolve a prev pointer given a base address.
    template <typename T = void>
    T* prev(void* base) noexcept {
        if (is_prev_null()) return nullptr;
        return reinterpret_cast<T*>(
            reinterpret_cast<uintptr_t>(base) + prev_offset);
    }

    template <typename T = void>
    const T* prev(void* base) const noexcept {
        if (is_prev_null()) return nullptr;
        return reinterpret_cast<const T*>(
            reinterpret_cast<uintptr_t>(base) + prev_offset);
    }

    /// Resolve a next pointer given a base address.
    template <typename T = void>
    T* next(void* base) noexcept {
        if (is_next_null()) return nullptr;
        return reinterpret_cast<T*>(
            reinterpret_cast<uintptr_t>(base) + next_offset);
    }

    template <typename T = void>
    const T* next(void* base) const noexcept {
        if (is_next_null()) return nullptr;
        return reinterpret_cast<const T*>(
            reinterpret_cast<uintptr_t>(base) + next_offset);
    }

    /// Set prev pointer relative to base.
    void set_prev(void* ptr, void* base) {
        if (ptr == nullptr) {
            prev_offset = kNullOffset;
            return;
        }
        auto ptr_addr = reinterpret_cast<uintptr_t>(ptr);
        auto base_addr = reinterpret_cast<uintptr_t>(base);
        if (ptr_addr < base_addr) {
            throw std::overflow_error("compressed_intrusive_hook::set_prev: pointer below base address");
        }
        auto diff = ptr_addr - base_addr;
        // fix.01 P2-22: the bound must be kNullOffset - 1, not UINT32_MAX.
        // kNullOffset is the "null" sentinel, so accepting diff == UINT32_MAX
        // stored a value that every reader decodes as NULL — the node silently
        // lost its predecessor. compressed_ptr (the non-hook variant) already
        // used kMaxOffset = kNullOffset - 1.
        if (diff > kNullOffset - 1) {
            throw std::overflow_error("compressed_intrusive_hook::set_prev: offset exceeds 32-bit range");
        }
        prev_offset = static_cast<uint32_t>(diff);
    }

    /// Set next pointer relative to base.
    void set_next(void* ptr, void* base) {
        if (ptr == nullptr) {
            next_offset = kNullOffset;
            return;
        }
        auto ptr_addr = reinterpret_cast<uintptr_t>(ptr);
        auto base_addr = reinterpret_cast<uintptr_t>(base);
        if (ptr_addr < base_addr) {
            throw std::overflow_error("compressed_intrusive_hook::set_next: pointer below base address");
        }
        auto diff = ptr_addr - base_addr;
        // fix.01 P2-22: see set_prev — UINT32_MAX is the null sentinel.
        if (diff > kNullOffset - 1) {
            throw std::overflow_error("compressed_intrusive_hook::set_next: offset exceeds 32-bit range");
        }
        next_offset = static_cast<uint32_t>(diff);
    }

    // --- Flag operations (same as intrusive_hook) ---

    bool is_tail() const noexcept { return flags & kTailFlag; }
    void set_tail() noexcept { flags |= kTailFlag; }
    void clear_tail() noexcept { flags &= kTailClearMask; }

    bool is_accessed() const noexcept { return flags & kAccessedFlag; }
    void set_accessed() noexcept { flags |= kAccessedFlag; }
    void clear_accessed() noexcept { flags &= kAccessedClearMask; }

    bool is_linked() const noexcept { return flags & kLinkedFlag; }
    void set_linked() noexcept { flags |= kLinkedFlag; }
    void clear_linked() noexcept { flags &= kLinkedClearMask; }

    uint32_t get_update_time() const noexcept { return update_time; }
    void set_update_time(uint32_t t) noexcept { update_time = t; }
};

// ============================================================================
// Specialisation of hook_pointer_traits for compressed_intrusive_hook.
// ============================================================================

namespace detail {

template <typename Hook>
struct hook_pointer_traits;

} // namespace detail

/// Compressed-pointer specialisation of hook_pointer_traits:
/// every pointer operation goes through encode/decode to a base-relative offset.
/// is_end_prev / is_end_next true means the node sits at a list boundary.
template <>
struct detail::hook_pointer_traits<compressed_intrusive_hook> {
    static void set_prev(compressed_intrusive_hook& h, void* p, void* base) {
        h.set_prev(p, base);
    }
    static void set_next(compressed_intrusive_hook& h, void* p, void* base) {
        h.set_next(p, base);
    }
    static void* get_prev(compressed_intrusive_hook& h, void* base) noexcept {
        return h.template prev<void>(base);
    }
    static void* get_next(compressed_intrusive_hook& h, void* base) noexcept {
        return h.template next<void>(base);
    }
    static const void* get_prev(const compressed_intrusive_hook& h, void* base) noexcept {
        return h.template prev<const void>(base);
    }
    static const void* get_next(const compressed_intrusive_hook& h, void* base) noexcept {
        return h.template next<const void>(base);
    }
    static bool is_end_prev(const compressed_intrusive_hook& h) noexcept {
        return h.is_prev_null();
    }
    static bool is_end_next(const compressed_intrusive_hook& h) noexcept {
        return h.is_next_null();
    }
};

// ============================================================================
// Compressed Region Allocator
// ============================================================================

/// Lightweight type-erased destructor (16 bytes vs std::function's 32-48).
/// Stores a function pointer and a void* context, avoiding heap allocation
/// and SSO overhead of std::function.
struct trivial_destructor {
    using destructor_fn = void(*)(void*) noexcept;
    destructor_fn fn = nullptr;
    void* ctx = nullptr;

    void operator()() const noexcept {
        if (fn) fn(ctx);
    }
};

/// Contiguous-memory allocator: reserves one large block and records its base
///
/// address for the compressed pointers. Giving `compressed_region` and
/// `intrusive_list` the same base is what makes `compressed_intrusive_hook`'s
///
///   compressed_region region(num_items, sizeof(my_item));
///   intrusive_list<my_item, compressed_intrusive_hook> list(region.base());
///
/// Constraint: every item must be allocated by region.allocate(), i.e. inside the
/// reserved block. Plain `new` gives no contiguity guarantee and cannot be used.
class compressed_region {
public:
    /// @param item_count  Expected number of items.
    /// @param item_size   Size of one item (sizeof(ItemType)).
    compressed_region(std::size_t item_count, std::size_t item_size)
        : item_size_(item_size), capacity_(item_count) {
        if (item_size_ == 0) {
            throw std::invalid_argument("compressed_region: item_size must be > 0");
        }
        if (item_count > std::numeric_limits<std::size_t>::max() / item_size_) {
            throw std::bad_alloc();
        }
        auto total = item_count * item_size_;
        memory_ = static_cast<char*>(::operator new(total, std::nothrow));
        if (!memory_) {
            throw std::bad_alloc();
        }
        base_ = memory_;
    }

    ~compressed_region() {
        // Call destructors in reverse allocation order before releasing memory.
        for (auto it = destructors_.rbegin(); it != destructors_.rend(); ++it) {
            (*it)();
        }
        ::operator delete(memory_);
    }

    compressed_region(const compressed_region&) = delete;
    compressed_region& operator=(const compressed_region&) = delete;

    /// Base address; every allocate() result lies in [base, base + total_bytes).
    void* base() const noexcept { return base_; }

    /// Placement-new a T inside the reserved region.
    /// The returned pointer lies in [base, base + total_bytes) and is encodable
    /// as a uint32_t offset. Returns nullptr if the region is exhausted.
    template <typename T, typename... Args>
    T* allocate(Args&&... args) {
        std::lock_guard<std::mutex> lock(allocate_mutex_);
        // Align used_ to alignof(T) before allocation
        std::size_t alignment = alignof(T);
        std::size_t aligned_used = (used_ + alignment - 1) & ~(alignment - 1);
        if (aligned_used + sizeof(T) > capacity()) {
            return nullptr;
        }
        auto* ptr = ::new (memory_ + aligned_used) T(std::forward<Args>(args)...);
        used_ = aligned_used + sizeof(T);
        destructors_.push_back(trivial_destructor{
            [](void* p) noexcept { static_cast<T*>(p)->~T(); },
            ptr
        });
        return ptr;
    }

    /// Bytes currently in use.
    std::size_t used() const noexcept { return used_; }

    /// Total bytes available.
    std::size_t capacity() const noexcept { return capacity_ * item_size_; }

    /// Bytes still available.
    std::size_t remaining() const noexcept { return capacity_ * item_size_ - used_; }

private:
    char* memory_;
    void* base_;
    std::size_t item_size_;
    std::size_t capacity_;
    std::size_t used_ = 0;
    std::vector<trivial_destructor> destructors_;
    std::mutex allocate_mutex_;
};

// ============================================================================
// Utility: Detect if a type uses compressed hooks
// ============================================================================

template <typename T, typename = void>
struct has_compressed_hook : std::false_type {};

template <typename T>
struct has_compressed_hook<T, std::void_t<decltype(T::hook_is_compressed)>>
    : std::true_type {};

template <typename T>
inline constexpr bool has_compressed_hook_v = has_compressed_hook<T>::value;

/// Estimate per-item memory savings when using compressed_intrusive_hook
/// instead of intrusive_hook.
/// @tparam Key    Cache key type
/// @tparam Value  Cache value type
/// @return Bytes saved per item (positive) or 0 if no savings.
template <typename Key, typename Value>
constexpr std::size_t compressed_hook_memory_savings() {
    using regular_item = detail::cache_item<Key, Value, detail::intrusive_hook>;
    using compressed_item = detail::cache_item<Key, Value, compressed_intrusive_hook>;
    return sizeof(regular_item) > sizeof(compressed_item)
               ? sizeof(regular_item) - sizeof(compressed_item)
               : 0;
}

/// Estimate total memory savings for a cache of `n` items.
template <typename Key, typename Value>
constexpr std::size_t compressed_hook_total_savings(std::size_t n) {
    return n * compressed_hook_memory_savings<Key, Value>();
}

} // namespace lru

#endif // LRU_COMPRESSED_PTR_HPP
