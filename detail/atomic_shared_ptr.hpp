// SPDX-License-Identifier: MIT
// atomic_shared_ptr — spinlock-based replacement for std::atomic<std::shared_ptr<T>>
//
// Clang 22's libc++ does not provide the C++20/23 partial specialization of
// std::atomic for std::shared_ptr<T> (it requires T to be trivially copyable).
// This wrapper provides the same load/store interface using a lightweight
// spinlock, which is sufficient for the library's usage patterns:
//   - Infrequent writes (background sampler, config changes, drain operations)
//   - Frequent lock-free-style reads (admission control, hot-key queries)
//
// The spinlock is held for only a few instructions (copy the control block
// pointer), so contention is negligible even under high read concurrency.

#ifndef LRU_DETAIL_ATOMIC_SHARED_PTR_HPP
#define LRU_DETAIL_ATOMIC_SHARED_PTR_HPP

#include <atomic>
#include <memory>
#include <thread>

namespace lru::detail {

/// Spinlock-based atomic shared_ptr wrapper.
/// Provides load() and store() with the same signatures as
/// std::atomic<std::shared_ptr<T>>.
template <typename T>
class atomic_shared_ptr {
public:
    atomic_shared_ptr() = default;
    explicit atomic_shared_ptr(std::shared_ptr<T> ptr)
        : ptr_(std::move(ptr)) {}

    // Non-copyable (matches std::atomic semantics)
    atomic_shared_ptr(const atomic_shared_ptr&) = delete;
    atomic_shared_ptr& operator=(const atomic_shared_ptr&) = delete;

    // Movable — safe because moves only occur when no other thread
    // accesses the source (standard C++ move contract). This is needed
    // because unified_cache (which contains memory_monitor as a direct
    // member) must be movable for factory-style helpers and move
    // construction to compile.
    atomic_shared_ptr(atomic_shared_ptr&& other) noexcept
        : lock_(ATOMIC_FLAG_INIT), ptr_(nullptr)
    {
        while (other.lock_.test_and_set(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        ptr_ = std::move(other.ptr_);
        other.lock_.clear(std::memory_order_release);
    }

    atomic_shared_ptr& operator=(atomic_shared_ptr&& other) noexcept {
        if (this != &other) {
            while (lock_.test_and_set(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            while (other.lock_.test_and_set(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            ptr_ = std::move(other.ptr_);
            other.lock_.clear(std::memory_order_release);
            lock_.clear(std::memory_order_release);
        }
        return *this;
    }

    /// Atomically load the shared_ptr. The memory_order parameter is
    /// respected at the spinlock level (acquire fence on load).
    std::shared_ptr<T> load(std::memory_order order = std::memory_order_seq_cst) const {
        (void)order;  // spinlock provides acq_rel semantics regardless
        while (lock_.test_and_set(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        std::shared_ptr<T> result = ptr_;  // copy under lock
        lock_.clear(std::memory_order_release);
        return result;
    }

    /// Atomically store a new shared_ptr. The memory_order parameter is
    /// respected at the spinlock level (release fence on store).
    void store(std::shared_ptr<T> new_ptr,
               std::memory_order order = std::memory_order_seq_cst) {
        (void)order;
        while (lock_.test_and_set(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        ptr_ = std::move(new_ptr);  // swap under lock
        lock_.clear(std::memory_order_release);
    }

    /// Atomically exchange the stored shared_ptr with a new one.
    /// Returns the previous value.
    std::shared_ptr<T> exchange(std::shared_ptr<T> new_ptr,
                                std::memory_order order = std::memory_order_seq_cst) {
        (void)order;
        while (lock_.test_and_set(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        std::shared_ptr<T> old = std::move(ptr_);
        ptr_ = std::move(new_ptr);
        lock_.clear(std::memory_order_release);
        return old;
    }

private:
    mutable std::atomic_flag lock_ = ATOMIC_FLAG_INIT;
    std::shared_ptr<T> ptr_;
};

}  // namespace lru::detail

#endif  // LRU_DETAIL_ATOMIC_SHARED_PTR_HPP
