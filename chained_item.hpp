// SPDX-License-Identifier: MIT
// Chained Items — Inspired by Facebook CacheLib's ChainedAllocs / ChainedHashTable
//
// Standard cache items are limited to a single allocation. When values exceed
// the typical cache item size (or the slab size in a slab-based allocator),
// memory fragmentation hurts utilization. Chained items split large values
// across multiple linked allocation units, each fitting within the allocator's
// preferred size class.
//
// This implementation provides:
//   - chained_value<T>: a sequence of fixed-size chunks storing a single value
//   - Automatic chunking: split large values; re-assemble on access
//   - Memory estimation: accurate per-item memory accounting
//   - Iterator: transparent traversal of all chunks
//   - Custom allocation integration: optional function pointers for chunk allocation
//   - chained_allocs: RAII wrapper that holds a parent handle to prevent eviction
//
// Usage:
//   // Large string cache
//   lru::cache<int, chained_value<std::string>> cache(10000);
//   cache.set(42, make_chained("very long string exceeding 4KB..."));
//   auto result = cache.get(42);
//   if (result) {
//       std::string& full = result->get().assemble();  // transparent re-assembly
//   }
//
//   // With custom chunk allocation
//   chunk_allocate_fn my_alloc = ...;
//   chunk_deallocate_fn my_dealloc = ...;
//   auto cv = make_chained_with_allocator("large data...", my_alloc, my_dealloc);
//
//   // With chained_allocs RAII wrapper
//   auto handle = cache.get(42);
//   chained_allocs<decltype(cache), decltype(handle)> chain(std::move(handle));

#ifndef LRU_CHAINED_ITEM_HPP
#define LRU_CHAINED_ITEM_HPP

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "detail/refcount.hpp"

namespace lru {

// Function pointer types for chunk allocation/deallocation.
// Avoids depending on the full slab_allocator definition (prevents
// "incomplete type" warnings when the template is instantiated before
// memory.hpp is included). Same pattern as concurrent_hash_table.hpp.
using chunk_allocate_fn   = void*(*)(std::size_t);
using chunk_deallocate_fn = void(*)(void*, std::size_t);

// ============================================================================
// Chained Value
// ============================================================================

/// A value split across multiple fixed-size chunks.
/// When stored in a cache, each chunk fits in one allocator slot.
///
/// Optionally uses function pointers for chunk allocation/deallocation.
/// When no alloc/dealloc functions are set, falls back to new/delete.
///
/// @tparam T        The underlying value type.
/// @tparam ChunkSize Size of each chunk in bytes (should be < allocator max).
template <typename T, std::size_t ChunkSize = 4096>
class chained_value {
public:
    using value_type = T;
    static constexpr std::size_t kChunkSize = ChunkSize;
    static constexpr std::size_t kChunkHeaderSize = 2 * sizeof(void*) + sizeof(uint32_t) + sizeof(detail::refcount_with_flags);
    static constexpr std::size_t kDataPerChunk = ChunkSize - kChunkHeaderSize;

    // --------------------------------------------------------------------
    // Chunk (internal)
    // --------------------------------------------------------------------

    struct chunk {
        chunk* next = nullptr;    // Linked list: next chunk
        chunk* prev = nullptr;    // Linked list: prev chunk (for insertion/deletion)
        uint32_t data_size = 0;   // Bytes of actual data in this chunk
        detail::refcount_with_flags refcount_{};  // Per-chunk refcount + flags
        char data[kDataPerChunk];

        /// Write data into this chunk. Returns bytes written (may be less if chunk is full).
        std::size_t write(const char* src, std::size_t len, std::size_t offset = 0) {
            auto capacity = kDataPerChunk - offset;
            auto to_copy = std::min(len, capacity);
            std::memcpy(data + offset, src, to_copy);
            data_size = static_cast<uint32_t>(offset + to_copy);
            return to_copy;
        }

        /// Read data from this chunk.
        std::size_t read(char* dst, std::size_t max_len, std::size_t offset = 0) const {
            auto available = data_size > offset ? data_size - offset : 0;
            auto to_copy = std::min(max_len, available);
            std::memcpy(dst, data + offset, to_copy);
            return to_copy;
        }

        /// Bytes of data currently stored in this chunk.
        std::size_t size() const noexcept { return data_size; }

        /// Remaining free space in this chunk.
        std::size_t free() const noexcept { return kDataPerChunk > data_size ? kDataPerChunk - data_size : 0; }

        // ---------------------------------------------------------------
        // Refcount / pin operations (each chunk independently pinnable)
        // ---------------------------------------------------------------

        /// Pin this chunk: increment access ref. Returns true on success.
        bool pin() noexcept {
            return refcount_.incRef() == detail::IncResult::kIncOk;
        }

        /// Unpin this chunk: decrement access ref.
        void unpin() {
            refcount_.decRef();
        }

        /// Check if this chunk has active handles (access_ref > 0).
        bool is_pinned() const noexcept {
            return refcount_.getAccessRef() > 0;
        }

        /// Access the underlying refcount_with_flags (for flag operations).
        detail::refcount_with_flags& refcount() noexcept { return refcount_; }
        const detail::refcount_with_flags& refcount() const noexcept { return refcount_; }
    };

    static_assert(sizeof(chunk) <= ChunkSize,
                  "chunk size exceeds ChunkSize — increase ChunkSize");

    // --------------------------------------------------------------------
    // Construction
    // --------------------------------------------------------------------

    chained_value() = default;

    /// Construct from a value, automatically chunking it.
    explicit chained_value(const T& value) {
        assign(value);
    }

    explicit chained_value(T&& value) {
        assign(std::move(value));
    }

    /// Construct from raw bytes (deserialization path).
    chained_value(const char* data, std::size_t size) {
        assign_raw(data, size);
    }

    ~chained_value() {
        clear();
    }

    chained_value(const chained_value& other)
        : total_size_(other.total_size_), alloc_fn_(other.alloc_fn_), dealloc_fn_(other.dealloc_fn_) {
        copy_chunks_from(other);
    }

    chained_value& operator=(const chained_value& other) {
        if (this != &other) {
            clear();
            total_size_ = other.total_size_;
            alloc_fn_ = other.alloc_fn_;
            dealloc_fn_ = other.dealloc_fn_;
            copy_chunks_from(other);
        }
        return *this;
    }

    chained_value(chained_value&& other) noexcept
        : head_(other.head_), total_size_(other.total_size_), alloc_fn_(other.alloc_fn_), dealloc_fn_(other.dealloc_fn_) {
        other.head_ = nullptr;
        other.total_size_ = 0;
        other.alloc_fn_ = nullptr;
        other.dealloc_fn_ = nullptr;
    }

    chained_value& operator=(chained_value&& other) noexcept {
        if (this != &other) {
            clear();
            head_ = other.head_;
            total_size_ = other.total_size_;
            alloc_fn_ = other.alloc_fn_;
            dealloc_fn_ = other.dealloc_fn_;
            other.head_ = nullptr;
            other.total_size_ = 0;
            other.alloc_fn_ = nullptr;
            other.dealloc_fn_ = nullptr;
        }
        return *this;
    }

    // --------------------------------------------------------------------
    // Allocator integration
    // --------------------------------------------------------------------

    /// Set chunk allocation/deallocation function pointers.
    /// When set, chunks are allocated via these functions; otherwise
    /// new/delete is used.
    void set_alloc_fns(chunk_allocate_fn alloc_fn, chunk_deallocate_fn dealloc_fn) noexcept {
        alloc_fn_ = alloc_fn;
        dealloc_fn_ = dealloc_fn;
    }

    /// Get the current chunk allocate function (may be nullptr).
    chunk_allocate_fn get_alloc_fn() const noexcept {
        return alloc_fn_;
    }

    /// Get the current chunk deallocate function (may be nullptr).
    chunk_deallocate_fn get_dealloc_fn() const noexcept {
        return dealloc_fn_;
    }

    // --------------------------------------------------------------------
    // Parent refcount / kHasChainedItem flag management
    // --------------------------------------------------------------------

    /// Access the parent-level refcount_with_flags.
    /// This is the refcount for the entire chained_value (parent item).
    detail::refcount_with_flags& parent_refcount() noexcept { return parent_refcount_; }
    const detail::refcount_with_flags& parent_refcount() const noexcept { return parent_refcount_; }

    /// Check whether the parent has the kHasChainedItem flag set.
    bool has_chained_items() const noexcept {
        return parent_refcount_.isFlagSet<detail::Flags::kHasChainedItem>();
    }

    // --------------------------------------------------------------------
    // Cascading eviction
    // --------------------------------------------------------------------

    /// Attempt to evict the parent and all its chained items.
    /// Returns true if eviction succeeded (no chunks are pinned).
    /// Returns false if any chunk has active handles (access_ref > 0),
    /// in which case nothing is evicted — the caller should defer.
    bool try_cascading_evict() {
        // First pass: check if any chunk is pinned
        for (auto* c = head_; c; c = c->next) {
            if (c->is_pinned()) {
                return false;
            }
        }
        // Also check if parent itself has active access refs
        if (parent_refcount_.getAccessRef() > 0) {
            return false;
        }
        // All chunks are unpinned — safe to evict the entire chain
        clear();
        parent_refcount_.unSetFlag<detail::Flags::kHasChainedItem>();
        return true;
    }

    /// Check whether cascading eviction is possible (no pinned chunks).
    bool can_cascading_evict() const noexcept {
        if (parent_refcount_.getAccessRef() > 0) return false;
        for (auto* c = head_; c; c = c->next) {
            if (c->is_pinned()) return false;
        }
        return true;
    }

    // --------------------------------------------------------------------
    // Assignment (with chunking)
    // --------------------------------------------------------------------

    void assign(const T& value) {
        static_assert(std::is_trivially_copyable_v<T>,
                      "chained_value<T> requires a trivially copyable type; "
                      "use the std::string specialization or provide a custom serializer");
        clear();
        assign_raw(reinterpret_cast<const char*>(&value), sizeof(T));
    }

    void assign(T&& value) {
        static_assert(std::is_trivially_copyable_v<T>,
                      "chained_value<T> requires a trivially copyable type; "
                      "use the std::string specialization or provide a custom serializer");
        clear();
        assign_raw(reinterpret_cast<const char*>(&value), sizeof(T));
    }

    void assign_raw(const char* data, std::size_t size) {
        clear();
        total_size_ = size;

        const char* src = data;
        std::size_t remaining = size;
        chunk* tail = nullptr;

        while (remaining > 0) {
            chunk* c = allocate_chunk();
            auto written = c->write(src, remaining);
            src += written;
            remaining -= written;

            c->prev = tail;
            if (tail) tail->next = c;
            if (!head_) head_ = c;
            tail = c;
        }

        // Mark parent as having chained items
        if (head_) {
            parent_refcount_.setFlag<detail::Flags::kHasChainedItem>();
        }
    }

    /// Assemble all chunks back into a single T value.
    T assemble() const {
        static_assert(std::is_trivially_copyable_v<T>,
                      "chained_value<T> requires a trivially copyable type");
        T result{};
        auto assembled = assemble_raw();
        if (assembled.has_value() && assembled->size() >= sizeof(T)) {
            std::memcpy(&result, assembled->data(), sizeof(T));
        }
        return result;
    }

    /// Assemble all chunks into a byte vector.
    std::optional<std::vector<char>> assemble_raw() const {
        if (!head_) return std::nullopt;

        std::vector<char> result;
        result.reserve(total_size_);

        for (auto* c = head_; c; c = c->next) {
            result.insert(result.end(), c->data, c->data + c->data_size);
        }
        return result;
    }

    /// Assemble into a string (convenience — calls assemble_raw).
    std::string to_string() const {
        auto raw = assemble_raw();
        if (!raw) return {};
        return std::string(raw->data(), raw->size());
    }

    // --------------------------------------------------------------------
    // Queries
    // --------------------------------------------------------------------

    /// Total bytes across all chunks.
    std::size_t size() const noexcept { return total_size_; }

    /// Number of chunks.
    std::size_t chunk_count() const noexcept {
        std::size_t count = 0;
        for (auto* c = head_; c; c = c->next) ++count;
        return count;
    }

    /// Total memory overhead (chunk headers + unused space).
    std::size_t overhead() const noexcept {
        std::size_t header_overhead = chunk_count() * (sizeof(chunk) - kDataPerChunk);
        std::size_t unused = 0;
        for (auto* c = head_; c; c = c->next) {
            unused += c->free();
        }
        return header_overhead + unused;
    }

    bool empty() const noexcept { return head_ == nullptr; }

    // --------------------------------------------------------------------
    // Iteration
    // --------------------------------------------------------------------

    class const_iterator {
    public:
        using value_type = std::string_view;
        using iterator_category = std::forward_iterator_tag;

        const_iterator() = default;
        explicit const_iterator(const chunk* c) : current_(c) {}

        std::string_view operator*() const {
            return current_ ? std::string_view(current_->data, current_->data_size)
                            : std::string_view{};
        }

        const_iterator& operator++() {
            if (current_) current_ = current_->next;
            return *this;
        }

        const_iterator operator++(int) {
            auto tmp = *this;
            ++*this;
            return tmp;
        }

        bool operator==(const const_iterator& other) const noexcept {
            return current_ == other.current_;
        }
        bool operator!=(const const_iterator& other) const noexcept {
            return !(*this == other);
        }

        const chunk* chunk_ptr() const noexcept { return current_; }

    private:
        const chunk* current_ = nullptr;
    };

    const_iterator begin() const noexcept { return const_iterator(head_); }
    const_iterator end() const noexcept { return const_iterator(nullptr); }

    // --------------------------------------------------------------------
    // Memory management
    // --------------------------------------------------------------------

    void clear() {
        auto* c = head_;
        while (c) {
            auto* next = c->next;
            deallocate_chunk(c);
            c = next;
        }
        head_ = nullptr;
        total_size_ = 0;
        parent_refcount_.unSetFlag<detail::Flags::kHasChainedItem>();
    }

private:
    chunk* head_ = nullptr;
    std::size_t total_size_ = 0;
    // Optional custom allocation via function pointers (avoids circular
    // dependency on slab_allocator definition).
    chunk_allocate_fn   alloc_fn_   = nullptr;
    chunk_deallocate_fn dealloc_fn_ = nullptr;
    detail::refcount_with_flags parent_refcount_{};  // Parent-level refcount + flags

    /// Allocate a single chunk using the alloc function if available,
    /// otherwise fall back to new.
    chunk* allocate_chunk() {
        if (alloc_fn_) {
            void* mem = alloc_fn_(sizeof(chunk));
            if (mem) {
                return new (mem) chunk();
            }
        }
        return new chunk();
    }

    /// Deallocate a single chunk. If a dealloc function is set, destruct the
    /// chunk in-place and return the memory via the function; otherwise delete.
    void deallocate_chunk(chunk* c) {
        if (dealloc_fn_) {
            c->~chunk();
            dealloc_fn_(c, sizeof(chunk));
        } else {
            delete c;
        }
    }

    void copy_chunks_from(const chained_value& other) {
        chunk* tail = nullptr;
        for (auto* c = other.head_; c; c = c->next) {
            auto* new_c = allocate_chunk();
            std::memcpy(new_c->data, c->data, c->data_size);
            new_c->data_size = c->data_size;
            new_c->prev = tail;
            new_c->next = nullptr;
            if (tail) tail->next = new_c;
            if (!head_) head_ = new_c;
            tail = new_c;
        }
    }
};

// ============================================================================
// Chained Value (std::string specialization)
// ============================================================================

/// Specialization for std::string: stores the string directly in chunks
/// without the intermediate serialization step.
///
/// Optionally uses function pointers for chunk allocation/deallocation.
template <std::size_t ChunkSize>
class chained_value<std::string, ChunkSize> {
public:
    using value_type = std::string;
    static constexpr std::size_t kChunkSize = ChunkSize;
    static constexpr std::size_t kChunkHeaderSize = 2 * sizeof(void*) + sizeof(uint32_t) + sizeof(detail::refcount_with_flags);
    static constexpr std::size_t kDataPerChunk = ChunkSize - kChunkHeaderSize;

    struct chunk {
        chunk* next = nullptr;
        chunk* prev = nullptr;
        uint32_t data_size = 0;
        detail::refcount_with_flags refcount_{};  // Per-chunk refcount + flags
        char data[kDataPerChunk];

        std::size_t write(const char* src, std::size_t len, std::size_t offset = 0) {
            auto capacity = kDataPerChunk - offset;
            auto to_copy = std::min(len, capacity);
            std::memcpy(data + offset, src, to_copy);
            data_size = static_cast<uint32_t>(offset + to_copy);
            return to_copy;
        }

        std::size_t read(char* dst, std::size_t max_len, std::size_t offset = 0) const {
            auto available = data_size > offset ? data_size - offset : 0;
            auto to_copy = std::min(max_len, available);
            std::memcpy(dst, data + offset, to_copy);
            return to_copy;
        }

        std::size_t size() const noexcept { return data_size; }
        std::size_t free() const noexcept { return kDataPerChunk > data_size ? kDataPerChunk - data_size : 0; }

        // ---------------------------------------------------------------
        // Refcount / pin operations (each chunk independently pinnable)
        // ---------------------------------------------------------------

        /// Pin this chunk: increment access ref. Returns true on success.
        bool pin() noexcept {
            return refcount_.incRef() == detail::IncResult::kIncOk;
        }

        /// Unpin this chunk: decrement access ref.
        void unpin() {
            refcount_.decRef();
        }

        /// Check if this chunk has active handles (access_ref > 0).
        bool is_pinned() const noexcept {
            return refcount_.getAccessRef() > 0;
        }

        /// Access the underlying refcount_with_flags (for flag operations).
        detail::refcount_with_flags& refcount() noexcept { return refcount_; }
        const detail::refcount_with_flags& refcount() const noexcept { return refcount_; }
    };

    chained_value() = default;

    explicit chained_value(const std::string& s) {
        assign_raw(s.data(), s.size());
    }

    explicit chained_value(std::string_view sv) {
        assign_raw(sv.data(), sv.size());
    }

    chained_value(const char* data, std::size_t size) {
        assign_raw(data, size);
    }

    ~chained_value() { clear(); }

    chained_value(const chained_value& other)
        : total_size_(other.total_size_), alloc_fn_(other.alloc_fn_), dealloc_fn_(other.dealloc_fn_) {
        copy_chunks_from(other);
    }
    chained_value& operator=(const chained_value& other) {
        if (this != &other) {
            clear();
            total_size_ = other.total_size_;
            alloc_fn_ = other.alloc_fn_;
            dealloc_fn_ = other.dealloc_fn_;
            copy_chunks_from(other);
        }
        return *this;
    }

    chained_value(chained_value&& other) noexcept
        : head_(other.head_), total_size_(other.total_size_), alloc_fn_(other.alloc_fn_), dealloc_fn_(other.dealloc_fn_) {
        other.head_ = nullptr;
        other.total_size_ = 0;
        other.alloc_fn_ = nullptr;
        other.dealloc_fn_ = nullptr;
    }

    chained_value& operator=(chained_value&& other) noexcept {
        if (this != &other) {
            clear();
            head_ = other.head_;
            total_size_ = other.total_size_;
            alloc_fn_ = other.alloc_fn_;
            dealloc_fn_ = other.dealloc_fn_;
            other.head_ = nullptr;
            other.total_size_ = 0;
            other.alloc_fn_ = nullptr;
            other.dealloc_fn_ = nullptr;
        }
        return *this;
    }

    // --------------------------------------------------------------------
    // Allocator integration
    // --------------------------------------------------------------------

    /// Set chunk allocation/deallocation function pointers.
    void set_alloc_fns(chunk_allocate_fn alloc_fn, chunk_deallocate_fn dealloc_fn) noexcept {
        alloc_fn_ = alloc_fn;
        dealloc_fn_ = dealloc_fn;
    }

    /// Get the current chunk allocate function (may be nullptr).
    chunk_allocate_fn get_alloc_fn() const noexcept {
        return alloc_fn_;
    }

    /// Get the current chunk deallocate function (may be nullptr).
    chunk_deallocate_fn get_dealloc_fn() const noexcept {
        return dealloc_fn_;
    }

    // --------------------------------------------------------------------
    // Parent refcount / kHasChainedItem flag management
    // --------------------------------------------------------------------

    /// Access the parent-level refcount_with_flags.
    /// This is the refcount for the entire chained_value (parent item).
    detail::refcount_with_flags& parent_refcount() noexcept { return parent_refcount_; }
    const detail::refcount_with_flags& parent_refcount() const noexcept { return parent_refcount_; }

    /// Check whether the parent has the kHasChainedItem flag set.
    bool has_chained_items() const noexcept {
        return parent_refcount_.isFlagSet<detail::Flags::kHasChainedItem>();
    }

    // --------------------------------------------------------------------
    // Cascading eviction
    // --------------------------------------------------------------------

    /// Attempt to evict the parent and all its chained items.
    /// Returns true if eviction succeeded (no chunks are pinned).
    /// Returns false if any chunk has active handles (access_ref > 0),
    /// in which case nothing is evicted — the caller should defer.
    bool try_cascading_evict() {
        for (auto* c = head_; c; c = c->next) {
            if (c->is_pinned()) return false;
        }
        if (parent_refcount_.getAccessRef() > 0) return false;
        clear();
        parent_refcount_.unSetFlag<detail::Flags::kHasChainedItem>();
        return true;
    }

    /// Check whether cascading eviction is possible (no pinned chunks).
    bool can_cascading_evict() const noexcept {
        if (parent_refcount_.getAccessRef() > 0) return false;
        for (auto* c = head_; c; c = c->next) {
            if (c->is_pinned()) return false;
        }
        return true;
    }

    // --------------------------------------------------------------------
    // Assignment / assembly
    // --------------------------------------------------------------------

    void assign_raw(const char* data, std::size_t size) {
        clear();
        total_size_ = size;
        const char* src = data;
        std::size_t remaining = size;
        chunk* tail = nullptr;
        while (remaining > 0) {
            chunk* c = allocate_chunk();
            auto written = c->write(src, remaining);
            src += written;
            remaining -= written;
            c->prev = tail;
            if (tail) tail->next = c;
            if (!head_) head_ = c;
            tail = c;
        }
        // Mark parent as having chained items
        if (head_) {
            parent_refcount_.setFlag<detail::Flags::kHasChainedItem>();
        }
    }

    std::string assemble() const {
        std::string result;
        result.reserve(total_size_);
        for (auto* c = head_; c; c = c->next) {
            result.append(c->data, c->data_size);
        }
        return result;
    }

    void assign(const std::string& s) { assign_raw(s.data(), s.size()); }
    void assign(std::string&& s) { assign_raw(s.data(), s.size()); }
    void assign(std::string_view sv) { assign_raw(sv.data(), sv.size()); }

    std::string_view peek_chunk(std::size_t index) const {
        auto* c = head_;
        for (std::size_t i = 0; i < index && c; ++i) {
            c = c->next;
        }
        return c ? std::string_view(c->data, c->data_size) : std::string_view{};
    }

    // --------------------------------------------------------------------
    // Queries
    // --------------------------------------------------------------------

    std::size_t size() const noexcept { return total_size_; }
    std::size_t chunk_count() const noexcept {
        std::size_t count = 0;
        for (auto* c = head_; c; c = c->next) ++count;
        return count;
    }
    std::size_t overhead() const noexcept {
        return chunk_count() * (sizeof(chunk) - kDataPerChunk);
    }
    bool empty() const noexcept { return !head_; }

    // --------------------------------------------------------------------
    // Memory management
    // --------------------------------------------------------------------

    void clear() {
        auto* c = head_;
        while (c) { auto* next = c->next; deallocate_chunk(c); c = next; }
        head_ = nullptr;
        total_size_ = 0;
        parent_refcount_.unSetFlag<detail::Flags::kHasChainedItem>();
    }

private:
    chunk* head_ = nullptr;
    std::size_t total_size_ = 0;
    // Optional custom allocation via function pointers (avoids circular
    // dependency on slab_allocator definition).
    chunk_allocate_fn   alloc_fn_   = nullptr;
    chunk_deallocate_fn dealloc_fn_ = nullptr;
    detail::refcount_with_flags parent_refcount_{};  // Parent-level refcount + flags

    /// Allocate a single chunk using the alloc function if available,
    /// otherwise fall back to new.
    chunk* allocate_chunk() {
        if (alloc_fn_) {
            void* mem = alloc_fn_(sizeof(chunk));
            if (mem) {
                return new (mem) chunk();
            }
        }
        return new chunk();
    }

    /// Deallocate a single chunk. If a dealloc function is set, destruct the
    /// chunk in-place and return the memory via the function; otherwise delete.
    void deallocate_chunk(chunk* c) {
        if (dealloc_fn_) {
            c->~chunk();
            dealloc_fn_(c, sizeof(chunk));
        } else {
            delete c;
        }
    }

    void copy_chunks_from(const chained_value& other) {
        chunk* tail = nullptr;
        for (auto* c = other.head_; c; c = c->next) {
            auto* new_c = allocate_chunk();
            std::memcpy(new_c->data, c->data, c->data_size);
            new_c->data_size = c->data_size;
            new_c->prev = tail;
            new_c->next = nullptr;
            if (tail) tail->next = new_c;
            if (!head_) head_ = new_c;
            tail = new_c;
        }
    }
};

// ============================================================================
// Convenience factory
// ============================================================================

/// Create a chained_value from any compatible source.
template <std::size_t ChunkSize = 4096, typename T>
chained_value<T, ChunkSize> make_chained(const T& value) {
    return chained_value<T, ChunkSize>(value);
}

template <std::size_t ChunkSize = 4096, typename T>
chained_value<T, ChunkSize> make_chained(T&& value) {
    chained_value<T, ChunkSize> result;
    result.assign(std::move(value));
    return result;
}

template <std::size_t ChunkSize = 4096>
chained_value<std::string, ChunkSize> make_chained(const char* data, std::size_t size) {
    return chained_value<std::string, ChunkSize>(data, size);
}

/// Create a chained_value with custom chunk allocation/deallocation functions.
template <std::size_t ChunkSize = 4096, typename T>
chained_value<T, ChunkSize> make_chained_with_allocator(
    const T& value, chunk_allocate_fn alloc_fn, chunk_deallocate_fn dealloc_fn) {
    chained_value<T, ChunkSize> result(value);
    result.set_alloc_fns(alloc_fn, dealloc_fn);
    return result;
}

/// Create a chained_value<std::string> with custom chunk allocation/deallocation from raw data.
template <std::size_t ChunkSize = 4096>
chained_value<std::string, ChunkSize> make_chained_with_allocator(
    const char* data, std::size_t size, chunk_allocate_fn alloc_fn, chunk_deallocate_fn dealloc_fn) {
    chained_value<std::string, ChunkSize> result(data, size);
    result.set_alloc_fns(alloc_fn, dealloc_fn);
    return result;
}

// ============================================================================
// Chained Allocs — RAII container for chained items (CacheLib-style)
// ============================================================================

/// RAII wrapper that holds a parent handle to prevent the chain from being
/// evicted. Inspired by Facebook CacheLib's ChainedAllocs.
///
/// The key guarantee: as long as the parent handle is held, the parent item
/// (and its chained values) cannot be evicted by the cache, because the
/// read_handle increments the parent's reference count.
///
/// When the parent's refcount_with_flags has kHasChainedItem set, the
/// eviction path should traverse the chain and mark all chained items for
/// eviction before evicting the parent.
///
/// @tparam Cache  The cache type (used for SFINAE on iterator support).
/// @tparam Handle The handle type (typically read_handle<Value>).
template <typename Cache, typename Handle>
class chained_allocs {
public:
    using parent_handle_type = Handle;

    /// Construct from a parent handle. Holding this handle prevents
    /// the parent (and thus the chain) from being evicted.
    explicit chained_allocs(Handle parent) noexcept
        : parent_(std::move(parent)) {}

    chained_allocs(const chained_allocs&) = delete;
    chained_allocs& operator=(const chained_allocs&) = delete;

    chained_allocs(chained_allocs&& other) noexcept
        : parent_(std::move(other.parent_)) {}

    chained_allocs& operator=(chained_allocs&& other) noexcept {
        if (this != &other) {
            parent_ = std::move(other.parent_);
        }
        return *this;
    }

    ~chained_allocs() = default;

    /// Access the parent handle.
    const Handle& parent_handle() const noexcept { return parent_; }
    Handle& parent_handle() noexcept { return parent_; }

    /// Whether the parent handle is valid (non-empty).
    explicit operator bool() const noexcept { return static_cast<bool>(parent_); }

private:
    Handle parent_;
};

} // namespace lru

#endif // LRU_CHAINED_ITEM_HPP
