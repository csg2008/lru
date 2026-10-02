// SPDX-License-Identifier: MIT
// Enhanced intrusive doubly-linked list inspired by CacheLib's DList.
// Key improvements over std::list:
//   - Embedded hook with updateTime and bit flags (no separate allocation)
//   - O(1) pointer-based insertBefore/replace/moveToHead
//   - Per-node metadata for delayed promotion and insertion point tracking
//   - Null-terminated design (no embedded sentinel objects)
//   - Hook pointer traits support both raw and compressed_intrusive_hook

#ifndef LRU_DETAIL_INTRUSIVE_LIST_HPP
#define LRU_DETAIL_INTRUSIVE_LIST_HPP

#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <new>
#include <type_traits>

#include "refcount.hpp"

namespace lru {
class slab_allocator;
}

// B11: ASAN 毒化（条件编译，仅在 ASAN 启用且头文件可用时生效）

// B11: ASAN 毒化（条件编译，仅在 ASAN 启用且头文件可用时生效）
#if defined(__SANITIZE_ADDRESS__) || (defined(__has_feature) && __has_feature(address_sanitizer))
#if __has_include(<sanitizer/asan_interface.h>)
#include <sanitizer/asan_interface.h>
#define LRU_HAS_ASAN 1
#endif
#endif

// B12: TSan 注解（条件编译）
#if defined(__SANITIZE_THREAD__) || (defined(__has_feature) && __has_feature(thread_sanitizer))
#if __has_include(<sanitizer/tsan_interface.h>)
#include <sanitizer/tsan_interface.h>
#define LRU_HAS_TSAN 1
#endif
#endif

namespace lru::detail {

// Forward declaration for hazptr_obj_base (defined in hazptr.hpp).
// cache_item inherits from it to enable zero-allocation retirement in the
// lock-free hazptr retire path. The full definition is not needed here
// because hazptr_obj_base only has two pointer-sized members (reclaim_ and
// next_) with default initializers, so the base class layout is trivially
// constructible.
struct hazptr_obj_base;

// ============================================================================
// Hook Pointer Traits - 桥接 raw 指针钩子和压缩指针钩子的统一接口
// ============================================================================

struct intrusive_hook;

/// 默认的 raw 指针钩子特质（使用 intrusive_hook）。
/// compressed_intrusive_hook 的特化在 compressed_ptr.hpp 中定义。
/// set_prev/set_next/get_prev/get_next 都忽略 base 参数（raw 指针不需要基地址）。
template <typename Hook>
struct hook_pointer_traits {
    static void set_prev(Hook& h, void* p, void*) noexcept { h.prev = p; }
    static void set_next(Hook& h, void* p, void*) noexcept { h.next = p; }
    static void* get_prev(Hook& h, void*) noexcept { return h.prev; }
    static void* get_next(Hook& h, void*) noexcept { return h.next; }
    static const void* get_prev(const Hook& h, void*) noexcept { return h.prev; }
    static const void* get_next(const Hook& h, void*) noexcept { return h.next; }
    /// prev 为空 → 此节点是链表头
    static bool is_end_prev(const Hook& h) noexcept { return h.prev == nullptr; }
    /// next 为空 → 此节点是链表尾
    static bool is_end_next(const Hook& h) noexcept { return h.next == nullptr; }
};

// ============================================================================
// Intrusive Hook
// ============================================================================

/// Embed this in your item class to make it linkable.
/// Inspired by CacheLib's DListHook<T> with updateTime and flag bits.
struct intrusive_hook {
    // Bit positions for flags
    static constexpr uint8_t kTailFlag     = 1 << 0;  // Item is in tail section (insertion point tracking)
    static constexpr uint8_t kAccessedFlag = 1 << 1;  // Item has been accessed since insertion
    static constexpr uint8_t kLinkedFlag   = 1 << 2;  // Item is currently linked in a list
    // P0-3: pre-computed clear masks. `flags &= ~kXxxFlag` performs the ~ on a
    // promoted int, producing a negative constant that -Wsign-conversion flags
    // when masked back into a uint8_t. Using a uint8_t mask keeps the compound
    // assignment value-preserving and warning-free.
    static constexpr uint8_t kTailClearMask     = static_cast<uint8_t>(~kTailFlag);
    static constexpr uint8_t kAccessedClearMask = static_cast<uint8_t>(~kAccessedFlag);
    static constexpr uint8_t kLinkedClearMask   = static_cast<uint8_t>(~kLinkedFlag);

    void* prev = nullptr;
    void* next = nullptr;

    // Seconds since epoch (like CacheLib's DListHook::updateTime).
    // Used for delayed promotion: currTime - updateTime >= lruRefreshTime.
    uint32_t update_time = 0;

    // Bit flags for tail/accessed tracking
    uint8_t flags = 0;

    // --- Flag operations ---
    bool is_tail() const noexcept { return flags & kTailFlag; }
    void set_tail() noexcept { flags |= kTailFlag; }
    void clear_tail() noexcept { flags &= kTailClearMask; }

    bool is_accessed() const noexcept { return flags & kAccessedFlag; }
    void set_accessed() noexcept { flags |= kAccessedFlag; }
    void clear_accessed() noexcept { flags &= kAccessedClearMask; }

    // Linked flag (node liveness tracking)
    bool is_linked() const noexcept { return flags & kLinkedFlag; }
    void set_linked() noexcept { flags |= kLinkedFlag; }
    void clear_linked() noexcept { flags &= kLinkedClearMask; }

    // B12: TSan-safe update_time 读取（对齐 CacheLib DList.h:69-75）
    uint32_t get_update_time() const noexcept {
#if defined(LRU_HAS_TSAN)
        AnnotateIgnoreReadsBegin(__FILE__, __LINE__);
        auto t = update_time;
        AnnotateIgnoreReadsEnd(__FILE__, __LINE__);
        return t;
#else
        return update_time;
#endif
    }
};

// ============================================================================
// Cache Item
// ============================================================================

/// Generic cache item that embeds the intrusive hook.
/// All MM strategies use this as the list node type.
/// @tparam Hook  Hook type (intrusive_hook, or compressed_intrusive_hook for ~33% less memory).
template <typename Key, typename Value, typename Hook = intrusive_hook>
struct cache_item : hazptr_obj_base {
    Hook hook;
    Key key;
    Value value;

    // Queue ID for multi-queue strategies (2Q, TinyLFU, W-TinyLFU)
    uint8_t queue_id = 0;

    // P1-1: Native TTL support.
    // 0 means "no TTL" (item never expires). Otherwise, this is the absolute
    // expiry time as nanoseconds since the steady_clock epoch. We use a packed
    // uint64 instead of std::optional<time_point> to keep the per-item overhead
    // at 8 bytes (well under the 10% size budget) and to keep the fast path
    // branch-free: `if (expiry_ns != 0)` is predicted not-taken for non-TTL
    // items, and only then do we pay for `steady_clock::now()`.
    //
    // The field is mutable through atomic stores in the MM layer (set_with_expiry)
    // but is read without atomicity on the hot path — this is safe because
    // expiry is set once at insertion time and never updated; the only race
    // is between a reader seeing 0 vs. a non-zero value, which is harmless
    // (a reader acting on stale 0 just treats the item as non-expiring for
    // that one access; the next access sees the correct value).
    std::uint64_t expiry_ns = 0;

    /// Check whether this item has expired at time `now_ns` (nanoseconds since
    /// the steady_clock epoch). Cheap: two loads + one comparison, branch
    /// predicted to not-taken for non-TTL items.
    bool is_expired(std::uint64_t now_ns) const noexcept {
        return expiry_ns != 0 && now_ns >= expiry_ns;
    }

    /// Whether this item has a TTL set at all.
    bool has_ttl() const noexcept { return expiry_ns != 0; }

    // H0: CAS-lockfree reference counting with embedded flags (CacheLib-style)
    // Tracks access references + admin bits (kLinked, kAccessible, kExclusive)
    // and user flags in a single atomic word.
    refcount_with_flags refcount;

    // Slab allocator that owns this item's memory.  nullptr means the item was
    // allocated with the global operator new/delete.  Stored in the item so
    // that every delete path (direct, hazptr retire, epoch retire) routes back
    // to the correct allocator without requiring every caller to know it.
    slab_allocator* allocator_ = nullptr;

    // Embedded hash chain link for concurrent_hash_table (CacheLib-style).
    // When the hash table operates in EmbeddedChain mode, bucket chains link
    // directly through cache_item objects instead of allocating separate
    // node_type wrappers. This eliminates one allocation per entry and one
    // pointer indirection per find().
    // Atomic to support hazptr-protected wait-free traversal: readers load
    // the next pointer with acquire, writers store with release.
    std::atomic<void*> hash_chain_next_{nullptr};
    std::size_t cached_hash_ = 0;

    // Hash chain accessors (atomic for concurrent hazptr traversal)
    void* hash_chain_next() const noexcept { return hash_chain_next_.load(std::memory_order_acquire); }
    void set_hash_chain_next(void* p) noexcept { hash_chain_next_.store(p, std::memory_order_release); }
    std::size_t cached_hash() const noexcept { return cached_hash_; }
    void set_cached_hash(std::size_t h) noexcept { cached_hash_ = h; }

    template <typename K, typename V>
    cache_item(K&& k, V&& v) noexcept(std::is_nothrow_constructible_v<Key, K> &&
                                       std::is_nothrow_constructible_v<Value, V>)
        : key(std::forward<K>(k)), value(std::forward<V>(v)) {
        // Newly allocated items are not yet linked into any list.
    }

    // Default allocation path: use the global heap.  The allocator_ member is
    // left nullptr, so operator delete below will also use the global heap.
    static void* operator new(std::size_t sz);

    // Slab-aware allocation path.  The MM layer uses this when a slab allocator
    // is attached; the matching operator delete is invoked automatically if the
    // constructor throws.  Defined in memory.hpp after slab_allocator is complete.
    static void* operator new(std::size_t sz, slab_allocator* alloc);

    static void operator delete(void* p, std::size_t sz);
    static void operator delete(void* p);
    static void operator delete(void* p, std::size_t sz, slab_allocator* alloc);
    static void operator delete(void* p, slab_allocator* alloc);

    // D5: 节点存活性查询（对齐 CacheLib isInMMContainer）
    // Both the hook's linked flag (used by intrusive_list internally) and the
    // refcount's kLinked bit (used by markForEviction/markMoving) are kept in
    // sync. is_in_container() queries the refcount since that is the canonical
    // source for eviction logic.
    bool is_in_container() const noexcept { return refcount.isInMMContainer(); }
    void mark_in_container() noexcept { hook.set_linked(); refcount.markInMMContainer(); }
    void unmark_in_container() noexcept { hook.clear_linked(); refcount.unmarkInMMContainer(); }

    // H0: 句柄计数操作 (via refcount_with_flags)
    bool has_active_handle() const noexcept { return refcount.getAccessRef() > 0; }

    // Access the hook
    Hook& get_hook() noexcept { return hook; }
    const Hook& get_hook() const noexcept { return hook; }
};

// ============================================================================
// Hook Accessor (default: item.get_hook())
// ============================================================================

/// Default hook accessor for types with get_hook() member.
template <typename T>
auto& default_get_hook(T& item) noexcept { return item.get_hook(); }

template <typename T>
const auto& default_get_hook(const T& item) noexcept { return item.get_hook(); }

// ============================================================================
// Intrusive List
// ============================================================================

/// Intrusive doubly-linked list using embedded hooks.
/// Items are NOT allocated by the list; they must already exist
/// and have a hook embedded in them.
///
/// 设计特点（相对于使用哨兵节点的传统实现）：
///   - Null-terminated: 头项 prev=nullptr, 尾项 next=nullptr
///   - 使用 hook_pointer_traits<Hook> 抽象所有指鈭操作
///   - 支持 intrusive_hook（raw void*）和 compressed_intrusive_hook（uint32_t offset）
///   - 当使用压缩指针时，base_ 提供基地址用于偏移解析
///
/// @tparam T           Item type (must have get_hook() or provide GetHook)
/// @tparam Hook        Hook type (intrusive_hook or compressed_intrusive_hook)
/// @tparam GetHook     Function to extract hook from item (default: default_get_hook<T>)
template <typename T, typename Hook = intrusive_hook, Hook& (*GetHook)(T&) = default_get_hook<T>>
class intrusive_list {
public:
    using value_type = T;
    using size_type = std::size_t;
    using reference = T&;
    using pointer = T*;

    using hook_traits = hook_pointer_traits<Hook>;

    // ========================================================================
    // Iterator
    // ========================================================================

    template <typename ItemT>
    class iterator_impl {
    public:
        using iterator_category = std::bidirectional_iterator_tag;
        using value_type = ItemT;
        using difference_type = std::ptrdiff_t;
        using pointer = ItemT*;
        using reference = ItemT&;

        // B9: 方向枚举（对齐 CacheLib DList.h:185）
        enum class Direction { FROM_HEAD, FROM_TAIL };

        iterator_impl() : node_(nullptr), list_(nullptr) {}
        iterator_impl(ItemT* node, const intrusive_list* list)
            : node_(node), list_(list) {}

        reference operator*() const { return *node_; }
        pointer operator->() const { return node_; }

        iterator_impl& operator++() {
            if (node_ && list_) {
                // 通过 hook_traits 解析 next 指针（raw 直接取值，compressed 需要 decode）
                auto& hook = list_->get_hook_const(*node_);
                auto* next_ptr = static_cast<ItemT*>(hook_traits::get_next(const_cast<Hook&>(hook), list_->base_));
                // B13: 预取下一个节点（对齐 CacheLib AllocationClass.h 预取优化）
                // 在遍历大缓存时减少 cache miss 延迟
#if defined(__GNUC__) || defined(__clang__)
                if (next_ptr) {
                    __builtin_prefetch(next_ptr, 0, 1);
                }
#endif
                node_ = next_ptr;
            }
            return *this;
        }

        iterator_impl operator++(int) {
            auto tmp = *this;
            ++*this;
            return tmp;
        }

        iterator_impl& operator--() {
            if (node_ && list_) {
                auto& hook = list_->get_hook_const(*node_);
                auto* prev_ptr = static_cast<ItemT*>(hook_traits::get_prev(const_cast<Hook&>(hook), list_->base_));
#if defined(__GNUC__) || defined(__clang__)
                if (prev_ptr) {
                    __builtin_prefetch(prev_ptr, 0, 1);
                }
#endif
                node_ = prev_ptr;
            }
            return *this;
        }

        iterator_impl operator--(int) {
            auto tmp = *this;
            --*this;
            return tmp;
        }

        bool operator==(const iterator_impl& other) const { return node_ == other.node_; }
        bool operator!=(const iterator_impl& other) const { return node_ != other.node_; }

        ItemT* node() const { return node_; }

        // D5: reset/resetToBegin（对齐 CacheLib DList.h:221-226）
        void reset() {
            node_ = nullptr;
        }

        void reset_to_begin(bool from_tail = false) {
            if (list_) {
                node_ = from_tail ? static_cast<ItemT*>(list_->tail_ptr_) : static_cast<ItemT*>(list_->head_ptr_);
            } else {
                node_ = nullptr;
            }
        }

    private:
        ItemT* node_;
        const intrusive_list* list_;
    };

    using iterator = iterator_impl<T>;
    using const_iterator = iterator_impl<const T>;

    /// 反向迭代器：++ 向 head 方向（get_prev），-- 向 tail 方向（get_next）。
    /// 不从 std::reverse_iterator 派生——因为 end() = nullptr 无法前置递减。
    template <typename ItemT>
    class reverse_iterator_impl : public iterator_impl<ItemT> {
        using base = iterator_impl<ItemT>;
    public:
        reverse_iterator_impl() : base() {}
        reverse_iterator_impl(ItemT* node, const intrusive_list* list) : base(node, list) {}

        // ++ 反向 → 向 head 移动（与 base 的 -- 相同）
        reverse_iterator_impl& operator++() {
            base::operator--();
            return *this;
        }
        reverse_iterator_impl operator++(int) { auto t = *this; ++*this; return t; }

        // -- 反向 → 向 tail 移动（与 base 的 ++ 相同）
        reverse_iterator_impl& operator--() {
            base::operator++();
            return *this;
        }
        reverse_iterator_impl operator--(int) { auto t = *this; --*this; return t; }
    };

    using reverse_iterator = reverse_iterator_impl<T>;
    using const_reverse_iterator = reverse_iterator_impl<const T>;

    // ========================================================================
    // Construction
    // ========================================================================

    intrusive_list() : head_ptr_(nullptr), tail_ptr_(nullptr), base_(nullptr), size_(0) {}

    explicit intrusive_list(void* base_address)
        : head_ptr_(nullptr), tail_ptr_(nullptr), base_(base_address), size_(0) {}

    ~intrusive_list() = default;

    intrusive_list(const intrusive_list&) = delete;
    intrusive_list& operator=(const intrusive_list&) = delete;

    /// 设置压缩指针基地址（仅对 compressed_intrusive_hook 有效，raw 指针忽略）。
    void set_base(void* base) noexcept { base_ = base; }
    void* base() const noexcept { return base_; }

    // ========================================================================
    // Link operations
    // ========================================================================

    /// Link item at head (MRU position)
    void link_at_head(T& item) {
#if defined(LRU_HAS_ASAN)
        ASAN_UNPOISON_MEMORY_REGION(&item, sizeof(T));
#endif
        auto& hook = GetHook(item);
        assert(!hook.is_linked() && "intrusive_list::link_at_head: item is already linked");
        // 新项的 prev=nullptr（头标记），next=当前头节点
        hook_traits::set_prev(hook, nullptr, base_);
        hook_traits::set_next(hook, head_ptr_, base_);

        // 原头节点的 prev 指向新节点
        if (head_ptr_) {
            auto& head_hook = GetHook(*static_cast<T*>(head_ptr_));
            hook_traits::set_prev(head_hook, &item, base_);
        } else {
            // 空链表，新节点同时也是尾节点
            tail_ptr_ = &item;
        }
        head_ptr_ = &item;
        hook.set_linked();
        ++size_;
    }

    /// Link item at tail (LRU position)
    void link_at_tail(T& item) {
#if defined(LRU_HAS_ASAN)
        ASAN_UNPOISON_MEMORY_REGION(&item, sizeof(T));
#endif
        auto& hook = GetHook(item);
        assert(!hook.is_linked() && "intrusive_list::link_at_tail: item is already linked");
        // 新项的 next=nullptr（尾标记），prev=当前尾节点
        hook_traits::set_next(hook, nullptr, base_);
        hook_traits::set_prev(hook, tail_ptr_, base_);

        // 原尾节点的 next 指向新节点
        if (tail_ptr_) {
            auto& tail_hook = GetHook(*static_cast<T*>(tail_ptr_));
            hook_traits::set_next(tail_hook, &item, base_);
        } else {
            // 空链表，新节点同时也是头节点
            head_ptr_ = &item;
        }
        tail_ptr_ = &item;
        hook.set_linked();
        ++size_;
    }

    /// Insert item before next_node in the list.
    /// next_node must be in the list, item must NOT be in the list.
    void insert_before(T& next_node, T& item) {
#if defined(LRU_HAS_ASAN)
        ASAN_UNPOISON_MEMORY_REGION(&item, sizeof(T));
#endif
        auto& next_hook = GetHook(next_node);
        auto& item_hook = GetHook(item);
        assert(!item_hook.is_linked() && "intrusive_list::insert_before: item is already linked");
        assert(next_hook.is_linked() && "intrusive_list::insert_before: next_node is not linked");

        // 获取 next_node 的前驱
        auto* prev_ptr = hook_traits::get_prev(next_hook, base_);

        // 将 item 链接在 prev_ptr 与 next_node 之间
        hook_traits::set_prev(item_hook, prev_ptr, base_);
        hook_traits::set_next(item_hook, &next_node, base_);

        // 更新前驱的 next
        if (prev_ptr) {
            auto& prev_hook = GetHook(*static_cast<T*>(prev_ptr));
            hook_traits::set_next(prev_hook, &item, base_);
        } else {
            // 插入到链表头
            head_ptr_ = &item;
        }
        // 更新 next_node 的 prev
        hook_traits::set_prev(next_hook, &item, base_);

        item_hook.set_linked();
        ++size_;
    }

    /// Unlink item from the list (does NOT clear item's hook pointers).
    void unlink(T& item) {
        auto& hook = GetHook(item);
        assert(hook.is_linked() && "intrusive_list::unlink: item is not linked");
        auto* prev_ptr = hook_traits::get_prev(hook, base_);
        auto* next_ptr = hook_traits::get_next(hook, base_);

        if (prev_ptr) {
            auto& prev_hook = GetHook(*static_cast<T*>(prev_ptr));
            hook_traits::set_next(prev_hook, next_ptr, base_);
        } else {
            // 被移除的是头节点
            head_ptr_ = next_ptr;
        }

        if (next_ptr) {
            auto& next_hook = GetHook(*static_cast<T*>(next_ptr));
            hook_traits::set_prev(next_hook, prev_ptr, base_);
        } else {
            // 被移除的是尾节点
            tail_ptr_ = prev_ptr;
        }

        hook.clear_linked();
        --size_;
        // Note: ASAN poisoning deferred to remove() / caller, because
        // hook pointers may still need to be accessed after unlink (e.g. in remove()).
    }

    /// Remove item from the list AND clear its hook pointers.
    /// Does NOT poison memory — callers that need ASan poisoning should call
    /// poison_removed() explicitly after all field accesses are complete.
    void remove(T& item) {
        unlink(item);
        auto& hook = GetHook(item);
        hook_traits::set_prev(hook, nullptr, base_);
        hook_traits::set_next(hook, nullptr, base_);
    }

    /// Explicitly poison a removed item's memory for ASan.
    /// Call this after remove() when no further field access on the item is needed.
    void poison_removed(T& item) {
#if defined(LRU_HAS_ASAN)
        ASAN_POISON_MEMORY_REGION(&item, sizeof(T));
#else
        (void)item;  // only meaningful under ASan
#endif
    }

    /// Move an existing item to head (optimized: direct pointer manipulation
    /// without unlink+link overhead — avoids redundant size/flag/ASAN updates).
    void move_to_head(T& item) {
        auto& hook = GetHook(item);
        // Already at head?
        if (hook_traits::is_end_prev(hook)) return;

        auto* prev_ptr = hook_traits::get_prev(hook, base_);
        auto* next_ptr = hook_traits::get_next(hook, base_);

        // Unlink from current position (no size/flag changes)
        if (prev_ptr) {
            auto& prev_hook = GetHook(*static_cast<T*>(prev_ptr));
            hook_traits::set_next(prev_hook, next_ptr, base_);
        } else {
            head_ptr_ = next_ptr;
        }
        if (next_ptr) {
            auto& next_hook = GetHook(*static_cast<T*>(next_ptr));
            hook_traits::set_prev(next_hook, prev_ptr, base_);
        } else {
            tail_ptr_ = prev_ptr;
        }

        // Relink at head (no size/linked-flag changes)
        hook_traits::set_prev(hook, nullptr, base_);
        hook_traits::set_next(hook, head_ptr_, base_);
        if (head_ptr_) {
            auto& head_hook = GetHook(*static_cast<T*>(head_ptr_));
            hook_traits::set_prev(head_hook, &item, base_);
        } else {
            tail_ptr_ = &item;
        }
        head_ptr_ = &item;
    }

    /// Replace old_node with new_node at the same position.
    void replace(T& old_node, T& new_node) {
        assert(&old_node != &new_node && "intrusive_list::replace: old_node and new_node must be different");
        auto& old_hook = GetHook(old_node);
        auto& new_hook = GetHook(new_node);
        assert(old_hook.is_linked() && "intrusive_list::replace: old_node must be linked");
        assert(!new_hook.is_linked() && "intrusive_list::replace: new_node must not be linked");

        auto* prev_ptr = hook_traits::get_prev(old_hook, base_);
        auto* next_ptr = hook_traits::get_next(old_hook, base_);

        // 新节点继承位置
        hook_traits::set_prev(new_hook, prev_ptr, base_);
        hook_traits::set_next(new_hook, next_ptr, base_);

        // 更新前驱的 next
        if (prev_ptr) {
            auto& prev_hook = GetHook(*static_cast<T*>(prev_ptr));
            hook_traits::set_next(prev_hook, &new_node, base_);
        } else {
            head_ptr_ = &new_node;
        }
        // 更新后继的 prev
        if (next_ptr) {
            auto& next_hook = GetHook(*static_cast<T*>(next_ptr));
            hook_traits::set_prev(next_hook, &new_node, base_);
        } else {
            tail_ptr_ = &new_node;
        }

        // 清除旧节点
        hook_traits::set_prev(old_hook, nullptr, base_);
        hook_traits::set_next(old_hook, nullptr, base_);
        old_hook.clear_linked();
        new_hook.set_linked();
    }

    // ========================================================================
    // Accessors
    // ========================================================================

    bool empty() const noexcept { return size_ == 0; }
    size_type size() const noexcept { return size_; }

    T& front() { assert(head_ptr_); return *static_cast<T*>(head_ptr_); }
    T& back()  { assert(tail_ptr_); return *static_cast<T*>(tail_ptr_); }
    T* front_ptr() const { return static_cast<T*>(head_ptr_); }
    T* back_ptr() const  { return static_cast<T*>(tail_ptr_); }

    T* head() const { return front_ptr(); }
    T* tail() const { return back_ptr(); }

    T* get_next(const T& node) const {
        auto& hook = GetHook(const_cast<T&>(node));
        return static_cast<T*>(hook_traits::get_next(const_cast<Hook&>(hook), base_));
    }

    T* get_prev(const T& node) const {
        auto& hook = GetHook(const_cast<T&>(node));
        return static_cast<T*>(hook_traits::get_prev(const_cast<Hook&>(hook), base_));
    }

    // ========================================================================
    // Iterators
    // ========================================================================

    iterator begin() { return iterator(static_cast<T*>(head_ptr_), this); }
    iterator end()   { return iterator(nullptr, this); }
    const_iterator begin() const { return const_iterator(static_cast<const T*>(head_ptr_), this); }
    const_iterator end() const   { return const_iterator(nullptr, this); }

    /// Reverse iterator starting from tail (++ 向 head 方向移动)
    reverse_iterator rbegin() { return reverse_iterator(static_cast<T*>(tail_ptr_), this); }
    reverse_iterator rend()   { return reverse_iterator(nullptr, this); }
    const_reverse_iterator rbegin() const {
        return const_reverse_iterator(static_cast<const T*>(tail_ptr_), this);
    }
    const_reverse_iterator rend() const {
        return const_reverse_iterator(nullptr, this);
    }

    // ========================================================================
    // Bulk operations
    // ========================================================================

    void clear() {
        // Unlink all items (doesn't destroy them - intrusive list doesn't own)
        auto* curr = head_ptr_;
        while (curr) {
            auto& hook = GetHook(*static_cast<T*>(curr));
            auto* next_ptr = hook_traits::get_next(hook, base_);
            hook_traits::set_prev(hook, nullptr, base_);
            hook_traits::set_next(hook, nullptr, base_);
            hook.clear_linked();
            curr = next_ptr;
        }
        head_ptr_ = nullptr;
        tail_ptr_ = nullptr;
        size_ = 0;
    }

    /// Pop the tail item and return it (caller is responsible for deletion)
    T* pop_tail() {
        if (empty()) return nullptr;
        auto* item = static_cast<T*>(tail_ptr_);
        // Use unlink + clear hooks (NOT remove) so the item remains unpoisoned
        // for the caller to access its fields (e.g. key/value destructors).
        unlink(*item);
        auto& hook = GetHook(*item);
        hook_traits::set_prev(hook, nullptr, base_);
        hook_traits::set_next(hook, nullptr, base_);
        return item;
    }

    /// Pop the head item and return it
    T* pop_head() {
        if (empty()) return nullptr;
        auto* item = static_cast<T*>(head_ptr_);
        // Use unlink + clear hooks (NOT remove) so the item remains unpoisoned
        // for the caller to access its fields.
        unlink(*item);
        auto& hook = GetHook(*item);
        hook_traits::set_prev(hook, nullptr, base_);
        hook_traits::set_next(hook, nullptr, base_);
        return item;
    }

private:
    // Helper: strip const for intrusive hook access (used by const_iterator).
    static Hook& get_hook_const(const T& item) {
        return GetHook(const_cast<T&>(item));
    }

    void* head_ptr_ = nullptr;  // 指向第一个 item，空表时为 nullptr
    void* tail_ptr_ = nullptr;  // 指向最后一个 item，空表时为 nullptr
    void* base_     = nullptr;  // 压缩指针基地址（raw 指针时忽略）
    size_type size_;
};


} // namespace lru::detail

#endif // LRU_DETAIL_INTRUSIVE_LIST_HPP
