// Unified LRU Cache Library - Core Foundation
// Merged from: concepts.hpp, traits.hpp, callbacks.hpp, statistics.hpp, iterator.hpp
// SPDX-License-Identifier: MIT

#ifndef LRU_CORE_HPP
#define LRU_CORE_HPP

#include "ankerl/unordered_dense.h"
#include "detail/foundation.hpp"
#include "detail/hazptr.hpp"
#include "detail/latency_histogram.hpp"
#include "detail/refcount.hpp"
#include "tls_ring.hpp"

#include <atomic>
#include <array>
#include <chrono>
#include <concepts>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <exception>
#include <format>
#include <functional>
#include <iterator>
#include <limits>
#include <list>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace lru {

// ============================================================================
// C++23 feature detection
// ============================================================================

#if __cplusplus >= 202300L || (__cpp_lib_expected >= 202211L)
#define LRU_HAS_CPP23_EXPECTED 1
#endif

#if __cplusplus >= 202300L || (__cpp_lib_mdspan >= 202207L)
#define LRU_HAS_CPP23_MDSPAN 1
#endif

#if __cplusplus >= 202300L || (__cpp_lib_print >= 202207L)
#define LRU_HAS_CPP23_PRINT 1
#endif

// ============================================================================
// Byte concepts
// ============================================================================

/// Byte-like type concept.
template <typename T>
concept byte_like = std::same_as<T, char>
                 || std::same_as<T, unsigned char>
                 || std::same_as<T, signed char>
                 || std::same_as<T, std::byte>
                 || std::same_as<T, uint8_t>
                 || std::same_as<T, int8_t>;

/// Concept for input iterator with byte value type.
template <typename I>
concept byte_input_iterator = std::input_iterator<I> && byte_like<std::iter_value_t<I>>;

/// Concept for output iterator accepting bytes.
template <typename I>
concept byte_output_iterator = std::output_iterator<I, std::byte>;

/// Concept for a container holding byte-like elements.
template <typename C>
concept byte_container = requires(C c) {
    typename C::value_type;
    requires byte_like<typename C::value_type>;
    { c.begin() } -> std::input_or_output_iterator;
    { c.end() } -> std::input_or_output_iterator;
};

// ============================================================================
// O11: Exception hierarchy
// ============================================================================
// Differentiated exception types so callers can catch specific failure modes
// (cache shutdown, OOM rejection, etc.) without parsing error strings.
//
// Hierarchy:
//   cache_exception              (base — catches all cache-originated errors)
//     cache_closed_exception     (graceful shutdown rejected the operation)
//     cache_oom_exception        (memory monitor / slab rejected admission)
//     cache_config_exception     (precondition failure: provider not set,
//                                 slab not enabled, invalid state, etc.)
//
// All inherit std::runtime_error so existing catch (std::runtime_error&)
// handlers continue to work. New code should catch the specific types.

class cache_exception : public std::runtime_error {
public:
    explicit cache_exception(const std::string& msg) : std::runtime_error(msg) {}
    explicit cache_exception(const char* msg) : std::runtime_error(msg) {}
};

/// Thrown when an operation is rejected because the cache has been
/// shut down via shutdown(). Caller should treat this as a permanent
/// condition — the cache cannot be revived.
class cache_closed_exception : public cache_exception {
public:
    explicit cache_closed_exception(const std::string& msg)
        : cache_exception(msg) {}
    explicit cache_closed_exception(const char* msg)
        : cache_exception(msg) {}
};

/// Thrown when the memory monitor or slab allocator rejects an insert
/// because the configured memory budget would be exceeded. Caller may
/// retry after evicting or raising the budget.
class cache_oom_exception : public cache_exception {
public:
    explicit cache_oom_exception(const std::string& msg)
        : cache_exception(msg) {}
    explicit cache_oom_exception(const char* msg)
        : cache_exception(msg) {}
};

/// Thrown when an operation cannot proceed due to a configuration or
/// state precondition (e.g., get_or_fetch without a value provider,
/// slab operations without enabling the slab allocator, read_handle
/// null dereference). Distinct from cache_closed (operational state)
/// and cache_oom (resource exhaustion).
class cache_config_exception : public cache_exception {
public:
    explicit cache_config_exception(const std::string& msg)
        : cache_exception(msg) {}
    explicit cache_config_exception(const char* msg)
        : cache_exception(msg) {}
};

/// T-G11: Thrown by `get()` when an item is found in the hash table
/// but `incRef()` returns `kIncFailedOverflow` even after a yield +
/// single retry. This indicates the item's access_ref counter (32
/// bits, max ~4B) is saturated — almost certainly a read_handle leak
/// in user code. Distinct from a cache miss (which returns an empty
/// handle, not an exception) so callers can differentiate.
class refcount_overflow_exception : public cache_exception {
public:
    explicit refcount_overflow_exception(const std::string& msg)
        : cache_exception(msg) {}
    explicit refcount_overflow_exception(const char* msg)
        : cache_exception(msg) {}
};

// ============================================================================
// Hash and comparison concepts
// ============================================================================

/// Concept for types that can be hashed.
template <typename T>
concept hashable = requires(T t) {
    { std::hash<T>{}(t) } -> std::convertible_to<std::size_t>;
};

/// Concept for key types suitable for cache use.
template <typename K>
concept cache_key = std::copy_constructible<K>
                 && std::equality_comparable<K>
                 && hashable<K>;

// ============================================================================
// Memory policy concepts
// ============================================================================

/// Concept for a callable that computes memory size of a value.
template <typename F, typename T>
concept memory_calculator = std::invocable<F, const T&>
                         && std::convertible_to<std::invoke_result_t<F, const T&>, std::size_t>;

/// Concept for a cache eviction callback.
template <typename F, typename K, typename V>
concept eviction_callback = std::invocable<F, const K&, const V&>;

/// Concept for a cache hit callback.
template <typename F, typename K, typename V>
concept hit_callback = std::invocable<F, const K&, const V&>;

/// Concept for a cache miss callback.
template <typename F, typename K>
concept miss_callback = std::invocable<F, const K&>;

/// Concept for a value provider (auto-fetch on miss).
template <typename F, typename K, typename V>
concept value_provider = std::invocable<F, const K&>
                      && std::convertible_to<std::invoke_result_t<F, const K&>, V>;

// ============================================================================
// Serialization concepts
// ============================================================================

/// Concept for a type with Serialize/Deserialize interface.
template <typename S, typename T>
concept serde_type = requires(S s, const T& ct, const std::byte* data, std::size_t size) {
    { S::serialize(ct) } -> byte_container;
    { S::deserialize(data, size) } -> std::same_as<T>;
};

// ============================================================================
// Constants
// ============================================================================

/// Sentinel value representing "unlimited".
inline constexpr std::size_t unlimited = std::numeric_limits<std::size_t>::max();

// ============================================================================
// Default policies
// ============================================================================

/// Default memory calculator: returns sizeof(T).
struct default_memory_calculator {
    template <typename T>
    std::size_t operator()(const T&) const {
        return sizeof(T);
    }
};

/// No-op eviction callback.
struct no_op_eviction_callback {
    template <typename K, typename V>
    void operator()(const K&, const V&) const {}
};

/// No-op hit callback.
struct no_op_hit_callback {
    template <typename K, typename V>
    void operator()(const K&, const V&) const {}
};

/// No-op miss callback.
struct no_op_miss_callback {
    template <typename K>
    void operator()(const K&) const {}
};

/// Throwing value provider.
template <typename Key, typename Value>
struct throwing_value_provider {
    Value operator()(const Key& key) const {
        (void)key;
        throw cache_config_exception("Key not found in cache");
    }
};

// ============================================================================
// Helper type traits
// ============================================================================

namespace detail {

/// Detect if a type has a reserve() method.
template <typename T, typename = void>
struct has_reserve : std::false_type {};

template <typename T>
struct has_reserve<T, std::void_t<decltype(std::declval<T>().reserve(std::declval<std::size_t>()))>>
    : std::true_type {};

template <typename T>
inline constexpr bool has_reserve_v = has_reserve<T>::value;

/// Detect if a type has an insert(end, first, last) method.
template <typename C, typename It, typename = void>
struct has_bulk_insert : std::false_type {};

template <typename C, typename It>
struct has_bulk_insert<C, It, std::void_t<
    decltype(std::declval<C>().insert(
        std::declval<typename C::iterator>(),
        std::declval<It>(),
        std::declval<It>()
    ))
>> : std::true_type {};

template <typename C, typename It>
inline constexpr bool has_bulk_insert_v = has_bulk_insert<C, It>::value;

} // namespace detail

// ============================================================================
// Callback Manager
// ============================================================================

/// Manages lifecycle callbacks for cache operations.
/// Supports hit, miss, insert, and eviction callbacks.
///
/// Callback events are collected via TLS ring buffers (zero-allocation,
/// zero-lock on the hot path) and flushed outside the cache lock for
/// reduced contention. When no callbacks are registered for a given
/// event type, collect_*() returns immediately with zero overhead.
template <typename Key, typename Value>
class callback_manager {
public:
    using hit_callback_type = std::function<void(const Key&, const Value&)>;
    using miss_callback_type = std::function<void(const Key&)>;
    using insert_callback_type = std::function<void(const Key&, const Value&)>;
    /// Eviction callback receives the value by const reference to avoid
    /// unnecessary copies when multiple callbacks are registered and to
    /// support move-only value types.
    using eviction_callback_type = std::function<void(const Key&, const Value&)>;
    /// O7: Update callback — fired when set() overwrites an existing
    /// key's value (not a fresh insert). Same signature as insert.
    using update_callback_type = std::function<void(const Key&, const Value&)>;
    /// O7: Expire callback — fired when an item is evicted due to TTL
    /// expiry (not capacity eviction). Same signature as evict.
    using expire_callback_type = std::function<void(const Key&, const Value&)>;
    /// O7: Reject callback — fired when an insert is rejected by the
    /// overflow policy (cache full, OOM, admission denial). Same
    /// signature as insert.
    using reject_callback_type = std::function<void(const Key&, const Value&)>;

    /// Identifies which callback event kind failed — passed to the
    /// error hook so operators can correlate failures with the operation
    /// that triggered them.
    enum class callback_event_kind {
        hit,
        miss,
        insert,
        evict,
        update,   ///< O7: set() overwrote an existing key
        expire,   ///< O7: TTL expiry (distinct from capacity evict)
        reject,   ///< O7: insert rejected by overflow policy
    };

    /// Error hook invoked when a registered callback throws. The hook
    /// receives:
    ///   - `eptr`        : the captured exception (std::current_exception())
    ///   - `kind`        : which event type triggered the failure
    ///   - `key`         : the key involved in the failed event
    ///   - `value`       : pointer to the value (nullptr for miss events,
    ///                     which carry no value)
    ///
    /// The hook is invoked on the dispatching thread (the async worker
    /// when async mode is active, otherwise the caller of flush_pending).
    /// Exceptions thrown by the error hook itself are swallowed to
    /// guarantee the dispatch loop is never aborted by error-handling
    /// code. At most one hook may be registered; registering a new hook
    /// replaces the previous one.
    using error_callback_type = std::function<void(
        std::exception_ptr eptr,
        callback_event_kind kind,
        const Key& key,
        const Value* value)>;

    /// Destructor: stop the async worker if running and drain remaining events.
    ~callback_manager() {
        try {
            stop_async_worker();
        } catch (...) {
            // Suppress exceptions in destructor
        }
    }

    // ------------------------------------------------------------------------
    // Registration
    // ------------------------------------------------------------------------

    /// Register a hit callback.
    void on_hit(hit_callback_type callback) {
        std::lock_guard<std::mutex> lock(mutex_);
        hit_callbacks_.push_back(std::move(callback));
        has_hit_callbacks_.store(!hit_callbacks_.empty(), std::memory_order_release);
    }

    /// Register a miss callback.
    void on_miss(miss_callback_type callback) {
        std::lock_guard<std::mutex> lock(mutex_);
        miss_callbacks_.push_back(std::move(callback));
        has_miss_callbacks_.store(!miss_callbacks_.empty(), std::memory_order_release);
    }

    /// Register an insert callback.
    void on_insert(insert_callback_type callback) {
        std::lock_guard<std::mutex> lock(mutex_);
        insert_callbacks_.push_back(std::move(callback));
        has_insert_callbacks_.store(!insert_callbacks_.empty(), std::memory_order_release);
    }

    /// Register an eviction callback.
    void on_evict(eviction_callback_type callback) {
        std::lock_guard<std::mutex> lock(mutex_);
        eviction_callbacks_.push_back(std::move(callback));
        has_eviction_callbacks_.store(!eviction_callbacks_.empty(), std::memory_order_release);
    }

    /// O7: Register an update callback — fires when set() overwrites
    /// an existing key's value (distinct from a fresh insert, which
    /// fires on_insert). Multiple callbacks may be registered; they
    /// fire in registration order.
    void on_update(update_callback_type callback) {
        std::lock_guard<std::mutex> lock(mutex_);
        update_callbacks_.push_back(std::move(callback));
        has_update_callbacks_.store(!update_callbacks_.empty(),
                                     std::memory_order_release);
    }

    /// O7: Register an expire callback — fires when an item is evicted
    /// due to TTL expiry (distinct from capacity eviction, which fires
    /// on_evict). Useful for monitoring TTL effectiveness and refresh
    /// window tuning. Multiple callbacks may be registered.
    void on_expire(expire_callback_type callback) {
        std::lock_guard<std::mutex> lock(mutex_);
        expire_callbacks_.push_back(std::move(callback));
        has_expire_callbacks_.store(!expire_callbacks_.empty(),
                                      std::memory_order_release);
    }

    /// O7: Register a reject callback — fires when an insert is rejected
    /// by the overflow policy (cache full, OOM, admission denial).
    /// Useful for routing rejected items to a fallback store or
    /// triggering back-pressure signalling. Multiple callbacks may be
    /// registered.
    void on_reject(reject_callback_type callback) {
        std::lock_guard<std::mutex> lock(mutex_);
        reject_callbacks_.push_back(std::move(callback));
        has_reject_callbacks_.store(!reject_callbacks_.empty(),
                                      std::memory_order_release);
    }

    /// Register an error hook invoked when any registered callback throws.
    /// Replaces any previously registered error hook. Pass a default-
    /// constructed `error_callback_type` (or `nullptr`) to clear.
    ///
    /// The hook is invoked at most once per failed event, after the
    /// exception has been captured. In synchronous mode, the first
    /// exception is still re-thrown by `flush_pending()` after all
    /// events have been dispatched — the hook is for observation only
    /// and does not alter control flow. In async mode, exceptions are
    /// swallowed by the worker (existing behavior) but the hook still
    /// fires, so operators can log metrics, emit alerts, or trigger
    /// circuit breakers without losing visibility.
    void on_callback_error(error_callback_type callback) {
        std::lock_guard<std::mutex> lock(mutex_);
        error_callback_ = std::move(callback);
        has_error_callback_.store(error_callback_ != nullptr,
                                  std::memory_order_release);
    }

    /// Total number of callback failures observed since construction or
    /// the last `reset_callback_error_count()`. Counts both sync and
    /// async dispatch failures.
    std::size_t callback_error_count() const noexcept {
        return callback_error_count_.load(std::memory_order_acquire);
    }

    /// Reset the callback failure counter.
    void reset_callback_error_count() noexcept {
        callback_error_count_.store(0, std::memory_order_release);
    }

    // ------------------------------------------------------------------------
    // Deferred (async) collection — TLS ring buffer (zero-allocation path)
    // ------------------------------------------------------------------------

    /// Collect a hit event into the TLS ring buffer (does not fire callbacks).
    /// Zero-allocation: stores Value directly (no shared_ptr).
    /// Zero-lock: writes to thread-local ring buffer.
    /// Fast-path: returns immediately when no hit callbacks are registered.
    void collect_hit(const Key& key, const Value& value) {
        if (!has_hit_callbacks_.load(std::memory_order_acquire)) return;
        ring_.collect_hit(key, value);
    }

    /// Collect a miss event into the TLS ring buffer.
    /// Fast-path: returns immediately when no miss callbacks are registered.
    void collect_miss(const Key& key) {
        if (!has_miss_callbacks_.load(std::memory_order_acquire)) return;
        ring_.collect_miss(key);
    }

    /// Collect an insert event into the TLS ring buffer.
    /// Zero-allocation: stores Value directly (no shared_ptr).
    /// Fast-path: returns immediately when no insert callbacks are registered.
    void collect_insert(const Key& key, const Value& value) {
        if (!has_insert_callbacks_.load(std::memory_order_acquire)) return;
        ring_.collect_insert(key, value);
    }

    /// Collect an evict event into the TLS ring buffer.
    /// Fast-path: returns immediately when no eviction callbacks are registered.
    void collect_evict(const Key& key, Value&& value) {
        if (!has_eviction_callbacks_.load(std::memory_order_acquire)) return;
        ring_.collect_evict(key, std::move(value));
    }

    // ------------------------------------------------------------------------
    // Move-semantic overloads – allow callers to avoid deep copies when the
    // key and/or value are no longer needed after the collect call.
    // ------------------------------------------------------------------------

    /// Collect a hit event with move semantics for key.
    void collect_hit(Key&& key, const Value& value) {
        if (!has_hit_callbacks_.load(std::memory_order_acquire)) return;
        ring_.collect_hit(std::move(key), value);
    }

    /// Collect an insert event with move semantics for key.
    void collect_insert(Key&& key, const Value& value) {
        if (!has_insert_callbacks_.load(std::memory_order_acquire)) return;
        ring_.collect_insert(std::move(key), value);
    }

    /// Collect an insert event with move semantics for both key and value.
    void collect_insert(Key&& key, Value&& value) {
        if (!has_insert_callbacks_.load(std::memory_order_acquire)) return;
        ring_.collect_insert(std::move(key), std::move(value));
    }

    /// Collect an evict event with move semantics for key too.
    void collect_evict(Key&& key, Value&& value) {
        if (!has_eviction_callbacks_.load(std::memory_order_acquire)) return;
        ring_.collect_evict(std::move(key), std::move(value));
    }

    // ------------------------------------------------------------------------
    // O7: New event collect methods — update / expire / reject.
    // Fast-path: returns immediately when no callbacks of the relevant
    // kind are registered.
    // ------------------------------------------------------------------------

    /// Collect an update event (lvalue key + value).
    void collect_update(const Key& key, const Value& value) {
        if (!has_update_callbacks_.load(std::memory_order_acquire)) return;
        ring_.collect_update(key, value);
    }

    /// Collect an update event (rvalue key + rvalue value).
    void collect_update(Key&& key, Value&& value) {
        if (!has_update_callbacks_.load(std::memory_order_acquire)) return;
        ring_.collect_update(std::move(key), std::move(value));
    }

    /// Collect an expire event (lvalue key + rvalue value).
    void collect_expire(const Key& key, Value&& value) {
        if (!has_expire_callbacks_.load(std::memory_order_acquire)) return;
        ring_.collect_expire(key, std::move(value));
    }

    /// Collect an expire event (rvalue key + rvalue value).
    void collect_expire(Key&& key, Value&& value) {
        if (!has_expire_callbacks_.load(std::memory_order_acquire)) return;
        ring_.collect_expire(std::move(key), std::move(value));
    }

    /// Collect a reject event (lvalue key + value).
    void collect_reject(const Key& key, const Value& value) {
        if (!has_reject_callbacks_.load(std::memory_order_acquire)) return;
        ring_.collect_reject(key, value);
    }

    /// Collect a reject event (rvalue key + rvalue value).
    void collect_reject(Key&& key, Value&& value) {
        if (!has_reject_callbacks_.load(std::memory_order_acquire)) return;
        ring_.collect_reject(std::move(key), std::move(value));
    }

    // ------------------------------------------------------------------------
    // Flush pending events
    // ------------------------------------------------------------------------

    /// Execute all pending callbacks and clear the TLS ring buffer.
    /// Safe to call outside the cache lock because events store their own
    /// value state (by-value, not shared_ptr).
    ///
    /// If async mode is enabled (set_async_mode(true)), drained events are
    /// pushed to a thread-safe queue and dispatched by a background worker
    /// thread. This method returns immediately after enqueuing.
    ///
    /// \throws The first exception thrown by a registered callback, re-thrown
    /// after all events have been dispatched (synchronous mode only). In
    /// async mode, exceptions are swallowed by the worker.
    void flush_pending() {
        // Drain TLS ring into local vectors (lock-free for the ring itself)
        auto drained = ring_.drain();
        if (drained.empty()) return;

        // Task 8: If async mode is enabled, push events to the async queue
        // and return immediately. The worker thread will dispatch them.
        if (async_mode_.load(std::memory_order_acquire)) {
            std::lock_guard<std::mutex> lock(async_mutex_);
            const std::size_t max_sz = async_queue_max_size_;
            const overflow_policy policy = async_overflow_policy_;
            // Record the queue size before enqueuing this batch so that on
            // kReject we can roll back only the events added here, preserving
            // events enqueued by previous flush_pending() calls that the
            // worker thread has not yet dispatched.
            const std::size_t queue_size_before = async_queue_.size();

            auto push_event = [&](async_event&& evt) {
                if (max_sz == 0 || async_queue_.size() < max_sz) {
                    async_queue_.push_back(std::move(evt));
                    return true;
                }
                // Queue full — apply overflow policy
                switch (policy) {
                    case overflow_policy::kDropOldest:
                        async_queue_.pop_front();
                        async_queue_.push_back(std::move(evt));
                        async_dropped_count_.fetch_add(1, std::memory_order_relaxed);
                        return true;
                    case overflow_policy::kDropNewest:
                        async_dropped_count_.fetch_add(1, std::memory_order_relaxed);
                        return true;  // drop the incoming event
                    case overflow_policy::kReject:
                        return false;  // signal caller to fall back to sync
                }
                return false;  // unreachable
            };

            bool rejected = false;
            for (auto& e : drained.hits) {
                if (!push_event(async_event{callback_event_kind::hit,
                                            std::move(e.key),
                                            std::move(e.value)})) {
                    rejected = true;
                    break;
                }
            }
            if (!rejected) {
                for (auto& e : drained.misses) {
                    if (!push_event(async_event{callback_event_kind::miss,
                                                std::move(e.key),
                                                Value{}})) {
                        rejected = true;
                        break;
                    }
                }
            }
            if (!rejected) {
                for (auto& e : drained.inserts) {
                    if (!push_event(async_event{callback_event_kind::insert,
                                                std::move(e.key),
                                                std::move(e.value)})) {
                        rejected = true;
                        break;
                    }
                }
            }
            if (!rejected) {
                for (auto& e : drained.evicts) {
                    if (!push_event(async_event{callback_event_kind::evict,
                                                std::move(e.key),
                                                std::move(e.value)})) {
                        rejected = true;
                        break;
                    }
                }
            }
            if (!rejected) {
                for (auto& e : drained.updates) {
                    if (!push_event(async_event{callback_event_kind::update,
                                                std::move(e.key),
                                                std::move(e.value)})) {
                        rejected = true;
                        break;
                    }
                }
            }
            if (!rejected) {
                for (auto& e : drained.expires) {
                    if (!push_event(async_event{callback_event_kind::expire,
                                                std::move(e.key),
                                                std::move(e.value)})) {
                        rejected = true;
                        break;
                    }
                }
            }
            if (!rejected) {
                for (auto& e : drained.rejects) {
                    if (!push_event(async_event{callback_event_kind::reject,
                                                std::move(e.key),
                                                std::move(e.value)})) {
                        rejected = true;
                        break;
                    }
                }
            }
            if (rejected) {
                // kReject policy: roll back only the events we enqueued in
                // this batch (preserving previously-enqueued events for the
                // worker to dispatch), then fall through to synchronous
                // dispatch for the full batch.
                async_queue_.resize(queue_size_before);
            } else {
                async_cv_.notify_one();
                return;
            }
        }

        // Synchronous dispatch path (original behavior).
        // Copy callback lists under mutex (only for dispatch)
        std::vector<hit_callback_type> local_hit_cbs;
        std::vector<miss_callback_type> local_miss_cbs;
        std::vector<insert_callback_type> local_insert_cbs;
        std::vector<eviction_callback_type> local_evict_cbs;
        std::vector<update_callback_type> local_update_cbs;
        std::vector<expire_callback_type> local_expire_cbs;
        std::vector<reject_callback_type> local_reject_cbs;
        error_callback_type local_error_cb;

        {
            std::lock_guard<std::mutex> lock(mutex_);
            local_hit_cbs = hit_callbacks_;
            local_miss_cbs = miss_callbacks_;
            local_insert_cbs = insert_callbacks_;
            local_evict_cbs = eviction_callbacks_;
            local_update_cbs = update_callbacks_;
            local_expire_cbs = expire_callbacks_;
            local_reject_cbs = reject_callbacks_;
            local_error_cb = error_callback_;
        }

        std::exception_ptr first_exception;

        // Dispatch hit events
        for (auto& e : drained.hits) {
            try {
                for (const auto& cb : local_hit_cbs) {
                    cb(e.key, e.value);
                }
            } catch (...) {
                auto eptr = std::current_exception();
                callback_error_count_.fetch_add(1, std::memory_order_relaxed);
                report_callback_error(local_error_cb, eptr,
                                      callback_event_kind::hit,
                                      e.key, &e.value);
                if (!first_exception) first_exception = eptr;
            }
        }

        // Dispatch miss events
        for (auto& e : drained.misses) {
            try {
                for (const auto& cb : local_miss_cbs) {
                    cb(e.key);
                }
            } catch (...) {
                auto eptr = std::current_exception();
                callback_error_count_.fetch_add(1, std::memory_order_relaxed);
                report_callback_error(local_error_cb, eptr,
                                      callback_event_kind::miss,
                                      e.key, nullptr);
                if (!first_exception) first_exception = eptr;
            }
        }

        // Dispatch insert events
        for (auto& e : drained.inserts) {
            try {
                for (const auto& cb : local_insert_cbs) {
                    cb(e.key, e.value);
                }
            } catch (...) {
                auto eptr = std::current_exception();
                callback_error_count_.fetch_add(1, std::memory_order_relaxed);
                report_callback_error(local_error_cb, eptr,
                                      callback_event_kind::insert,
                                      e.key, &e.value);
                if (!first_exception) first_exception = eptr;
            }
        }

        // Dispatch evict events
        for (auto& e : drained.evicts) {
            try {
                for (const auto& cb : local_evict_cbs) {
                    cb(e.key, e.value);
                }
            } catch (...) {
                auto eptr = std::current_exception();
                callback_error_count_.fetch_add(1, std::memory_order_relaxed);
                report_callback_error(local_error_cb, eptr,
                                      callback_event_kind::evict,
                                      e.key, &e.value);
                if (!first_exception) first_exception = eptr;
            }
        }

        // O7: Dispatch update events
        for (auto& e : drained.updates) {
            try {
                for (const auto& cb : local_update_cbs) {
                    cb(e.key, e.value);
                }
            } catch (...) {
                auto eptr = std::current_exception();
                callback_error_count_.fetch_add(1, std::memory_order_relaxed);
                report_callback_error(local_error_cb, eptr,
                                      callback_event_kind::update,
                                      e.key, &e.value);
                if (!first_exception) first_exception = eptr;
            }
        }

        // O7: Dispatch expire events
        for (auto& e : drained.expires) {
            try {
                for (const auto& cb : local_expire_cbs) {
                    cb(e.key, e.value);
                }
            } catch (...) {
                auto eptr = std::current_exception();
                callback_error_count_.fetch_add(1, std::memory_order_relaxed);
                report_callback_error(local_error_cb, eptr,
                                      callback_event_kind::expire,
                                      e.key, &e.value);
                if (!first_exception) first_exception = eptr;
            }
        }

        // O7: Dispatch reject events
        for (auto& e : drained.rejects) {
            try {
                for (const auto& cb : local_reject_cbs) {
                    cb(e.key, e.value);
                }
            } catch (...) {
                auto eptr = std::current_exception();
                callback_error_count_.fetch_add(1, std::memory_order_relaxed);
                report_callback_error(local_error_cb, eptr,
                                      callback_event_kind::reject,
                                      e.key, &e.value);
                if (!first_exception) first_exception = eptr;
            }
        }

        if (first_exception) {
            std::rethrow_exception(first_exception);
        }
    }

    /// Check if there are pending events waiting to be flushed.
    bool has_pending() const noexcept {
        return ring_.has_pending();
    }

    /// Number of pending events.
    std::size_t pending_count() const noexcept {
        return ring_.pending_count();
    }

    // ------------------------------------------------------------------------
    // Task 8: Async callback dispatch mode
    // ------------------------------------------------------------------------
    //
    // When async mode is enabled, flush_pending() drains the TLS ring into a
    // thread-safe queue (instead of dispatching callbacks inline). A dedicated
    // worker thread pops events from the queue and dispatches them outside
    // the calling thread's context.
    //
    // This decouples callback execution from the cache hot path, useful when
    // callbacks perform expensive work (e.g., logging, metric updates, network
    // I/O) that would otherwise inflate get()/set() tail latency.
    //
    // The worker thread is joined on disable or destructor, draining any
    // remaining events before exiting.

    /// Enable or disable async callback dispatch.
    /// @param enabled  true to spawn the worker and route events through the
    ///                 async queue; false to stop the worker and revert to
    ///                 synchronous dispatch in flush_pending().
    /// @throws nothing. Disabling blocks until the worker has drained the
    ///         queue and joined.
    void set_async_mode(bool enabled) {
        if (enabled) {
            start_async_worker();
        } else {
            stop_async_worker();
        }
    }

    /// Check if async dispatch mode is currently active.
    bool is_async_mode() const noexcept {
        return async_mode_.load(std::memory_order_acquire);
    }

    /// Synchronously drain the async queue, dispatching all pending events
    /// on the calling thread. Useful for shutdown or testing.
    void flush_async_queue() {
        drain_async_queue();
    }

    /// Number of events currently in the async dispatch queue.
    std::size_t async_queue_size() const {
        std::lock_guard<std::mutex> lock(async_mutex_);
        return async_queue_.size();
    }

    // --------------------------------------------------------------------
    // P0-1: Async queue backpressure
    // --------------------------------------------------------------------

    /// Overflow policy when the async queue reaches max_size.
    enum class overflow_policy {
        kDropOldest,   ///< Drop the oldest event to make room (default)
        kDropNewest,   ///< Drop the incoming event (newest)
        kReject        ///< Reject: fall back to synchronous dispatch
    };

    /// Set the maximum size of the async callback queue.
    /// 0 means unbounded (default — production use should set a limit).
    void set_async_queue_max_size(std::size_t max_size) {
        std::lock_guard<std::mutex> lock(async_mutex_);
        async_queue_max_size_ = max_size;
    }

    /// Get the current max size of the async callback queue.
    std::size_t async_queue_max_size() const {
        std::lock_guard<std::mutex> lock(async_mutex_);
        return async_queue_max_size_;
    }

    /// Set the overflow policy for when the async queue is full.
    void set_async_overflow_policy(overflow_policy policy) {
        std::lock_guard<std::mutex> lock(async_mutex_);
        async_overflow_policy_ = policy;
    }

    /// Get the current overflow policy.
    overflow_policy async_overflow_policy() const {
        std::lock_guard<std::mutex> lock(async_mutex_);
        return async_overflow_policy_;
    }

    /// Number of events dropped due to async queue overflow.
    std::size_t async_queue_dropped_count() const {
        return async_dropped_count_.load(std::memory_order_acquire);
    }

    /// Reset the dropped event counter.
    void reset_async_queue_dropped_count() {
        async_dropped_count_.store(0, std::memory_order_release);
    }

private:
    /// Variant event type for the async queue. `type` reuses the public
    /// `callback_event_kind` enum so error reporting can dispatch on a
    /// single enum without casting.
    struct async_event {
        callback_event_kind type;
        Key key;
        Value value;  // unused for miss events
    };

    /// Start the async dispatch worker thread.
    void start_async_worker() {
        if (async_mode_.exchange(true, std::memory_order_acq_rel)) {
            return;  // already running
        }
        async_worker_ = std::make_unique<std::thread>([this] { async_worker_loop(); });
    }

    /// Stop the async dispatch worker thread and drain remaining events.
    void stop_async_worker() {
        if (!async_mode_.exchange(false, std::memory_order_acq_rel)) {
            return;  // not running
        }
        async_cv_.notify_all();
        if (async_worker_ && async_worker_->joinable()) {
            async_worker_->join();
        }
        async_worker_.reset();
        // Drain any remaining events synchronously after the worker has stopped.
        drain_async_queue();
    }

    /// Worker thread loop: pop events from the queue and dispatch them.
    void async_worker_loop() {
        while (true) {
            std::vector<async_event> batch;
            {
                std::unique_lock<std::mutex> lock(async_mutex_);
                async_cv_.wait_for(lock, std::chrono::milliseconds(50),
                    [this] { return !async_mode_.load(std::memory_order_acquire)
                                          || !async_queue_.empty(); });
                if (!async_mode_.load(std::memory_order_acquire) && async_queue_.empty()) {
                    return;
                }
                // Move up to 256 events into a local batch to minimize
                // lock hold time.
                std::size_t n = std::min<std::size_t>(async_queue_.size(), 256);
                batch.reserve(n);
                for (std::size_t i = 0; i < n; ++i) {
                    batch.push_back(std::move(async_queue_.front()));
                    async_queue_.pop_front();
                }
            }
            dispatch_batch(batch);
        }
    }

    /// Drain the entire async queue and dispatch synchronously.
    void drain_async_queue() {
        std::vector<async_event> batch;
        {
            std::lock_guard<std::mutex> lock(async_mutex_);
            batch.reserve(async_queue_.size());
            while (!async_queue_.empty()) {
                batch.push_back(std::move(async_queue_.front()));
                async_queue_.pop_front();
            }
        }
        dispatch_batch(batch);
    }

    /// Dispatch a batch of events. Catches exceptions per-event to ensure
    /// one bad callback doesn't abort the worker. Failures increment
    /// `callback_error_count_` and invoke the error hook (if registered).
    void dispatch_batch(const std::vector<async_event>& batch) {
        if (batch.empty()) return;
        // Snapshot callback lists under mutex to avoid races with registration.
        std::vector<hit_callback_type> local_hit;
        std::vector<miss_callback_type> local_miss;
        std::vector<insert_callback_type> local_insert;
        std::vector<eviction_callback_type> local_evict;
        std::vector<update_callback_type> local_update;
        std::vector<expire_callback_type> local_expire;
        std::vector<reject_callback_type> local_reject;
        error_callback_type local_error_cb;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            local_hit = hit_callbacks_;
            local_miss = miss_callbacks_;
            local_insert = insert_callbacks_;
            local_evict = eviction_callbacks_;
            local_update = update_callbacks_;
            local_expire = expire_callbacks_;
            local_reject = reject_callbacks_;
            local_error_cb = error_callback_;
        }
        for (const auto& e : batch) {
            try {
                switch (e.type) {
                    case callback_event_kind::hit:
                        for (const auto& cb : local_hit) cb(e.key, e.value);
                        break;
                    case callback_event_kind::miss:
                        for (const auto& cb : local_miss) cb(e.key);
                        break;
                    case callback_event_kind::insert:
                        for (const auto& cb : local_insert) cb(e.key, e.value);
                        break;
                    case callback_event_kind::evict:
                        for (const auto& cb : local_evict) cb(e.key, e.value);
                        break;
                    case callback_event_kind::update:
                        for (const auto& cb : local_update) cb(e.key, e.value);
                        break;
                    case callback_event_kind::expire:
                        for (const auto& cb : local_expire) cb(e.key, e.value);
                        break;
                    case callback_event_kind::reject:
                        for (const auto& cb : local_reject) cb(e.key, e.value);
                        break;
                }
            } catch (...) {
                // Swallow per-event exceptions to keep the worker alive,
                // but record the failure and notify the error hook so
                // operators don't lose visibility into callback breakage.
                auto eptr = std::current_exception();
                callback_error_count_.fetch_add(1, std::memory_order_relaxed);
                const Value* vptr = (e.type == callback_event_kind::miss)
                                        ? nullptr
                                        : &e.value;
                report_callback_error(local_error_cb, eptr, e.type,
                                      e.key, vptr);
            }
        }
    }

    /// Invoke the error hook (if any) with a captured exception. Swallows
    /// any exception thrown by the hook itself so error-handling code can
    /// never abort the dispatch loop. The hook is snapshotted by the
    /// caller under `mutex_` to avoid races with `on_callback_error()`.
    static void report_callback_error(const error_callback_type& hook,
                                      std::exception_ptr eptr,
                                      callback_event_kind kind,
                                      const Key& key,
                                      const Value* value) {
        if (!hook) return;
        try {
            hook(eptr, kind, key, value);
        } catch (...) {
            // Swallow — error handlers must not throw.
        }
    }

public:

    // ------------------------------------------------------------------------
    // Clearing
    // ------------------------------------------------------------------------

    void clear_hit_callbacks() {
        std::lock_guard<std::mutex> lock(mutex_);
        hit_callbacks_.clear();
        has_hit_callbacks_.store(false, std::memory_order_release);
    }

    void clear_miss_callbacks() {
        std::lock_guard<std::mutex> lock(mutex_);
        miss_callbacks_.clear();
        has_miss_callbacks_.store(false, std::memory_order_release);
    }

    void clear_insert_callbacks() {
        std::lock_guard<std::mutex> lock(mutex_);
        insert_callbacks_.clear();
        has_insert_callbacks_.store(false, std::memory_order_release);
    }

    void clear_eviction_callbacks() {
        std::lock_guard<std::mutex> lock(mutex_);
        eviction_callbacks_.clear();
        has_eviction_callbacks_.store(false, std::memory_order_release);
    }

    /// O7: Clear update/expire/reject callbacks.
    void clear_update_callbacks() {
        std::lock_guard<std::mutex> lock(mutex_);
        update_callbacks_.clear();
        has_update_callbacks_.store(false, std::memory_order_release);
    }

    void clear_expire_callbacks() {
        std::lock_guard<std::mutex> lock(mutex_);
        expire_callbacks_.clear();
        has_expire_callbacks_.store(false, std::memory_order_release);
    }

    void clear_reject_callbacks() {
        std::lock_guard<std::mutex> lock(mutex_);
        reject_callbacks_.clear();
        has_reject_callbacks_.store(false, std::memory_order_release);
    }

    void clear_all() {
        std::lock_guard<std::mutex> lock(mutex_);
        hit_callbacks_.clear();
        miss_callbacks_.clear();
        insert_callbacks_.clear();
        eviction_callbacks_.clear();
        update_callbacks_.clear();
        expire_callbacks_.clear();
        reject_callbacks_.clear();
        error_callback_ = nullptr;
        has_hit_callbacks_.store(false, std::memory_order_release);
        has_miss_callbacks_.store(false, std::memory_order_release);
        has_insert_callbacks_.store(false, std::memory_order_release);
        has_eviction_callbacks_.store(false, std::memory_order_release);
        has_update_callbacks_.store(false, std::memory_order_release);
        has_expire_callbacks_.store(false, std::memory_order_release);
        has_reject_callbacks_.store(false, std::memory_order_release);
        has_error_callback_.store(false, std::memory_order_release);
        // Reset TLS ring for this instance
        ring_.reset();
    }

    // ------------------------------------------------------------------------
    // Queries
    // ------------------------------------------------------------------------

    bool has_hit_callbacks() const noexcept { return has_hit_callbacks_.load(std::memory_order_acquire); }
    bool has_miss_callbacks() const noexcept { return has_miss_callbacks_.load(std::memory_order_acquire); }
    bool has_insert_callbacks() const noexcept { return has_insert_callbacks_.load(std::memory_order_acquire); }
    bool has_eviction_callbacks() const noexcept { return has_eviction_callbacks_.load(std::memory_order_acquire); }
    /// O7: Queries for the new event kinds.
    bool has_update_callbacks() const noexcept { return has_update_callbacks_.load(std::memory_order_acquire); }
    bool has_expire_callbacks() const noexcept { return has_expire_callbacks_.load(std::memory_order_acquire); }
    bool has_reject_callbacks() const noexcept { return has_reject_callbacks_.load(std::memory_order_acquire); }

    std::size_t hit_callback_count() const noexcept { std::lock_guard<std::mutex> lock(mutex_); return hit_callbacks_.size(); }
    std::size_t miss_callback_count() const noexcept { std::lock_guard<std::mutex> lock(mutex_); return miss_callbacks_.size(); }
    std::size_t insert_callback_count() const noexcept { std::lock_guard<std::mutex> lock(mutex_); return insert_callbacks_.size(); }
    std::size_t eviction_callback_count() const noexcept { std::lock_guard<std::mutex> lock(mutex_); return eviction_callbacks_.size(); }
    std::size_t update_callback_count() const noexcept { std::lock_guard<std::mutex> lock(mutex_); return update_callbacks_.size(); }
    std::size_t expire_callback_count() const noexcept { std::lock_guard<std::mutex> lock(mutex_); return expire_callbacks_.size(); }
    std::size_t reject_callback_count() const noexcept { std::lock_guard<std::mutex> lock(mutex_); return reject_callbacks_.size(); }

    // ------------------------------------------------------------------------
    // Read-only access to callback vectors (for sharded propagation)
    // ------------------------------------------------------------------------

    const std::vector<hit_callback_type> hit_callbacks() const noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        return hit_callbacks_;
    }
    const std::vector<miss_callback_type> miss_callbacks() const noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        return miss_callbacks_;
    }
    const std::vector<insert_callback_type> insert_callbacks() const noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        return insert_callbacks_;
    }
    const std::vector<eviction_callback_type> eviction_callbacks() const noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        return eviction_callbacks_;
    }
    /// O7: Read-only access to the new callback vectors (for sharded propagation).
    const std::vector<update_callback_type> update_callbacks() const noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        return update_callbacks_;
    }
    const std::vector<expire_callback_type> expire_callbacks() const noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        return expire_callbacks_;
    }
    const std::vector<reject_callback_type> reject_callbacks() const noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        return reject_callbacks_;
    }

private:
    // --------------------------------------------------------------------
    // Data members
    // --------------------------------------------------------------------

    std::vector<hit_callback_type> hit_callbacks_;
    std::vector<miss_callback_type> miss_callbacks_;
    std::vector<insert_callback_type> insert_callbacks_;
    std::vector<eviction_callback_type> eviction_callbacks_;
    /// O7: New event callback vectors.
    std::vector<update_callback_type> update_callbacks_;
    std::vector<expire_callback_type> expire_callbacks_;
    std::vector<reject_callback_type> reject_callbacks_;

    /// O6: Optional error hook invoked when any registered callback throws.
    /// Snapshotted under `mutex_` before dispatch to avoid races.
    error_callback_type error_callback_;

    // Atomic flags for zero-overhead short-circuit when no callbacks registered.
    // Updated on registration/clear under mutex_; read lock-free in collect_*.
    std::atomic<bool> has_hit_callbacks_{false};
    std::atomic<bool> has_miss_callbacks_{false};
    std::atomic<bool> has_insert_callbacks_{false};
    std::atomic<bool> has_eviction_callbacks_{false};
    std::atomic<bool> has_update_callbacks_{false};
    std::atomic<bool> has_expire_callbacks_{false};
    std::atomic<bool> has_reject_callbacks_{false};
    std::atomic<bool> has_error_callback_{false};

    /// O6: Total callback failures observed across sync + async dispatch.
    /// Monotonically increasing; reset via `reset_callback_error_count()`.
    std::atomic<std::size_t> callback_error_count_{0};

    // Per-instance TLS ring for zero-allocation, zero-lock event collection.
    tls_callback_ring<Key, Value> ring_;

    // Protects callback registration and queries (NOT the collect/flush path)
    mutable std::mutex mutex_;

    // --------------------------------------------------------------------
    // Task 8: Async dispatch queue + worker thread
    // --------------------------------------------------------------------
    std::atomic<bool> async_mode_{false};
    mutable std::mutex async_mutex_;
    std::condition_variable async_cv_;
    std::deque<async_event> async_queue_;
    std::unique_ptr<std::thread> async_worker_;

    // P0-1: async queue backpressure
    std::size_t async_queue_max_size_{0};  // 0 = unbounded
    overflow_policy async_overflow_policy_{overflow_policy::kDropOldest};
    std::atomic<std::size_t> async_dropped_count_{0};
};

// ============================================================================
// Cache Iterator
// ============================================================================

/// Iterator for traversing cache items in LRU order (MRU to LRU).
/// Holds a hazard pointer to protect the currently pointed-to node from
/// concurrent reclamation. The hazard slot is acquired lazily (only when
/// the iterator becomes non-default) and released automatically on
/// destruction or when the iterator moves to a new position.
template <typename BaseIterator>
class cache_iterator {
public:
    using iterator_category = std::bidirectional_iterator_tag;
    using value_type = typename std::iterator_traits<BaseIterator>::value_type;
    using difference_type = typename std::iterator_traits<BaseIterator>::difference_type;
    using pointer = typename std::iterator_traits<BaseIterator>::pointer;
    using reference = typename std::iterator_traits<BaseIterator>::reference;

    /// Default constructor — creates a sentinel iterator with no hazard protection.
    cache_iterator() = default;

    /// Construct from a base iterator — acquires a hazard slot and protects
    /// the pointed-to node so that concurrent eviction cannot reclaim it
    /// while this iterator holds the position.
    explicit cache_iterator(BaseIterator it) : it_(it) {
        if (it_ != BaseIterator{}) {
            hazptr_.emplace();
            hazptr_->protect(&*it_);
        }
    }

    // Allow conversion from iterator to const_iterator.
    // The new iterator gets its own hazard slot protecting the same node.
    template <typename OtherIt>
    requires std::is_convertible_v<OtherIt, BaseIterator>
    cache_iterator(const cache_iterator<OtherIt>& other)
        : it_(other.base()) {
        if (it_ != BaseIterator{}) {
            hazptr_.emplace();
            hazptr_->protect(&*it_);
        }
    }

    // Copyable — each copy gets its own hazard slot protecting the same node.
    cache_iterator(const cache_iterator& other)
        : it_(other.it_) {
        if (it_ != BaseIterator{}) {
            hazptr_.emplace();
            hazptr_->protect(&*it_);
        }
    }

    cache_iterator& operator=(const cache_iterator& other) {
        if (this != &other) {
            it_ = other.it_;
            if (it_ != BaseIterator{}) {
                if (!hazptr_.has_value()) {
                    hazptr_.emplace();
                }
                hazptr_->protect(&*it_);
            } else {
                hazptr_.reset();
            }
        }
        return *this;
    }

    // Movable — transfer the hazard slot ownership.
    cache_iterator(cache_iterator&& other) noexcept
        : it_(std::move(other.it_))
        , hazptr_(std::move(other.hazptr_)) {}

    cache_iterator& operator=(cache_iterator&& other) noexcept {
        if (this != &other) {
            it_ = std::move(other.it_);
            hazptr_ = std::move(other.hazptr_);
        }
        return *this;
    }

    reference operator*() const { return *it_; }
    pointer operator->() const { return &*it_; }

    cache_iterator& operator++() {
        ++it_;
        protect_current();
        return *this;
    }

    cache_iterator operator++(int) {
        auto tmp = *this;
        ++it_;
        protect_current();
        return tmp;
    }

    cache_iterator& operator--() {
        --it_;
        protect_current();
        return *this;
    }

    cache_iterator operator--(int) {
        auto tmp = *this;
        --it_;
        protect_current();
        return tmp;
    }

    bool operator==(const cache_iterator& other) const { return it_ == other.it_; }
    bool operator!=(const cache_iterator& other) const { return it_ != other.it_; }

    BaseIterator base() const { return it_; }

private:
    /// Update the hazard pointer to protect the current iterator position.
    /// If the iterator is at the sentinel position, release the slot;
    /// otherwise ensure a slot is held and publish protection.
    void protect_current() {
        if (it_ == BaseIterator{}) {
            hazptr_.reset();
        } else {
            if (!hazptr_.has_value()) {
                hazptr_.emplace();
            }
            hazptr_->protect(&*it_);
        }
    }

    BaseIterator it_;
    std::optional<detail::hazptr_holder> hazptr_;
};

// ============================================================================
// Unordered Cache Iterator
// ============================================================================

/// Iterator providing direct access to the underlying hash map.
template <typename MapIterator>
class unordered_cache_iterator {
public:
    using map_iterator_type = MapIterator;
    using list_iterator_type = typename MapIterator::value_type::second_type;
    using value_type = typename std::iterator_traits<list_iterator_type>::value_type;
    using difference_type = std::ptrdiff_t;
    using pointer = value_type*;
    using reference = value_type&;

    unordered_cache_iterator() = default;
    explicit unordered_cache_iterator(MapIterator it) : it_(it) {}

    // The map stores Key -> list_iterator, so dereferencing gives the list node
    reference operator*() const {
        return *it_->second;
    }

    pointer operator->() const {
        return &*it_->second;
    }

    unordered_cache_iterator& operator++() {
        ++it_;
        return *this;
    }

    unordered_cache_iterator operator++(int) {
        auto tmp = *this;
        ++*this;
        return tmp;
    }

    bool operator==(const unordered_cache_iterator& other) const { return it_ == other.it_; }
    bool operator!=(const unordered_cache_iterator& other) const { return it_ != other.it_; }

    const auto& key() const { return it_->first; }
    auto map_iterator() const { return it_; }

private:
    MapIterator it_;
};

// ============================================================================
// Cache Statistics
// ============================================================================

/// Cache-line-padded atomic to eliminate false sharing on hot counters.
struct alignas(64) padded_atomic_size {
    std::atomic<std::size_t> value{0};
    // Padding is implicit: alignas(64) ensures the struct occupies a full
    // cache line, and sizeof(atomic<size_t>) < 64 on all mainstream platforms.

    // T8.1: Forwarding methods so callers can use padded_atomic_size
    // interchangeably with std::atomic<size_t> without sprinkling
    // `.value.` everywhere. Both spellings are supported: existing
    // code that uses `.value.load()` continues to work, while new
    // code can simply call `.load()`.
    std::size_t load(std::memory_order mo = std::memory_order_seq_cst) const noexcept {
        return value.load(mo);
    }
    void store(std::size_t v, std::memory_order mo = std::memory_order_seq_cst) noexcept {
        value.store(v, mo);
    }
    std::size_t fetch_add(std::size_t v,
                          std::memory_order mo = std::memory_order_seq_cst) noexcept {
        return value.fetch_add(v, mo);
    }
    std::size_t fetch_sub(std::size_t v,
                          std::memory_order mo = std::memory_order_seq_cst) noexcept {
        return value.fetch_sub(v, mo);
    }
    std::size_t exchange(std::size_t v,
                         std::memory_order mo = std::memory_order_seq_cst) noexcept {
        return value.exchange(v, mo);
    }
    bool compare_exchange_strong(std::size_t& expected, std::size_t desired,
                                 std::memory_order success,
                                 std::memory_order failure) noexcept {
        return value.compare_exchange_strong(expected, desired, success, failure);
    }
    bool compare_exchange_strong(std::size_t& expected, std::size_t desired,
                                 std::memory_order mo = std::memory_order_seq_cst) noexcept {
        return value.compare_exchange_strong(expected, desired, mo);
    }
};

/// P0-4: Sharded counter that distributes increments/decrements across
/// multiple cache lines to eliminate cache-line bouncing on hot counters
/// that are updated on every read_handle create/destroy (active_handle_count).
///
/// Each thread picks a shard based on a thread-local hash of its thread::id
/// (computed once per thread, cached in TLS). Concurrent threads therefore
/// write to different cache lines, eliminating the contention that a single
/// padded_atomic_size would suffer under high read concurrency (64+ cores
/// all fetch_add on the same line).
///
/// API is compatible with padded_atomic_size (load/store/fetch_add/fetch_sub)
/// so existing call sites need no changes. The cost is that load() must sum
/// all shards (O(kNumShards) atomic loads), which is acceptable because
/// active_handle_count is read infrequently (stats_snapshot, shutdown checks).
struct alignas(64) sharded_handle_counter {
    // 64 shards — one per core on typical 64-core / 2-socket NUMA
    // machines, so under a 64-thread read-heavy workload each thread
    // writes to its own cache line with high probability. Raised from
    // 16 (which left ~4:1 shard contention on 64 cores). Memory cost is
    // 64 * 64 B = 4 KiB per cache_stats instance, which is negligible
    // for a production cache.
    static constexpr std::size_t kNumShards = 64;  // power of 2 for fast modulo

    struct alignas(64) shard {
        std::atomic<std::size_t> value{0};
    };

    std::array<shard, kNumShards> shards{};

    /// Pick a shard for the current thread. Cached in TLS so the cost is
    /// a single relaxed load after the first call.
    static std::size_t shard_idx_for_thread() noexcept {
        static thread_local std::size_t idx = compute_shard_idx();
        return idx;
    }

    static std::size_t compute_shard_idx() noexcept {
        try {
            return std::hash<std::thread::id>{}(std::this_thread::get_id()) & (kNumShards - 1);
        } catch (...) {
            return 0;
        }
    }

    sharded_handle_counter() noexcept = default;

    /// Single-value constructor (used by copy constructors of cache_stats
    /// which initialize from a loaded snapshot value).
    explicit sharded_handle_counter(std::size_t v) noexcept {
        store(v, std::memory_order_relaxed);
    }

    std::size_t fetch_add(std::size_t v,
                          std::memory_order mo = std::memory_order_seq_cst) noexcept {
        return shards[shard_idx_for_thread()].value.fetch_add(v, mo);
    }

    std::size_t fetch_sub(std::size_t v,
                          std::memory_order mo = std::memory_order_seq_cst) noexcept {
        return shards[shard_idx_for_thread()].value.fetch_sub(v, mo);
    }

    std::size_t load(std::memory_order mo = std::memory_order_seq_cst) const noexcept {
        std::size_t sum = 0;
        for (std::size_t i = 0; i < kNumShards; ++i) {
            sum += shards[i].value.load(mo);
        }
        return sum;
    }

    void store(std::size_t v, std::memory_order mo = std::memory_order_seq_cst) noexcept {
        // Zero all shards except the first, then set the first.
        // (Used by copy/move/reset; the next fetch_add from any thread
        // will land on that thread's own shard, repopulating as needed.)
        for (std::size_t i = 1; i < kNumShards; ++i) {
            shards[i].value.store(0, mo);
        }
        shards[0].value.store(v, mo);
    }

    std::size_t exchange(std::size_t v,
                         std::memory_order mo = std::memory_order_seq_cst) noexcept {
        std::size_t old = load(mo);
        store(v, mo);
        return old;
    }
};
static_assert(sizeof(sharded_handle_counter) == 64 * sharded_handle_counter::kNumShards,
              "sharded_handle_counter must be fully cache-line padded per shard");

/// Hot-counters pair: groups hits and misses on a single cache line so that
/// a get() — which always updates exactly one of them — touches only one line
/// instead of two.  This is an alternative layout for workloads where
/// hits-updaters and misses-updaters run on the same core (e.g. single-threaded
/// or low-contention scenarios).  The default cache_stats keeps hits and misses
/// on separate padded lines to avoid false sharing between independent updaters.
///
/// T8.2: Engineering decision — keep separate padded lines as the default.
/// Rationale: the production target is "多线程高并发读多写少" (multi-threaded,
/// high-concurrency, read-heavy). Under that workload, hits and misses are
/// updated by *different* threads on *different* cores — a thread that has
/// warmed up its working set updates hits, while a thread experiencing a cold
/// start or capacity miss updates misses. Pairing them on a single cache line
/// would cause cache-line ping-pong between those cores (false sharing),
/// degrading throughput. The cost of pairing (one extra cache line per
/// cache_stats instance, 64 bytes) is negligible compared to the gain of
/// eliminating false sharing under contention.
///
/// To opt into the paired layout (e.g. for single-threaded caches or unit
/// tests), use `cache_stats_paired` defined below. The paired layout reduces
/// `sizeof(cache_stats)` by 64 bytes and may improve locality for
/// single-threaded access patterns.
struct alignas(64) hot_counters_pair {
    std::atomic<std::size_t> hits{0};
    std::atomic<std::size_t> misses{0};
    // Remaining space in 64-byte cache line is padding (implicit)
};

/// T-G10: Notifier used by `shutdown_and_wait()` to wake up immediately when
/// the last active `read_handle` is released, instead of busy-polling.
///
/// Owned by `unified_cache` (`active_handle_notifier_` member). A pointer to
/// it is stored in `cache_stats::release_notifier` so that `read_handle`
/// (which already carries a `cache_stats*`) can reach it without an extra
/// per-handle field. The pointer is NOT propagated to `cache_stats` copies /
/// snapshots (it defaults to `nullptr` there), so only the cache's own
/// `per_cache_stats_` instance carries a live notifier — exactly what we want.
///
/// Hot-path cost when no shutdown is in progress: one relaxed atomic load of
/// `shutdown_in_progress` (false) plus a branch — essentially free. The
/// expensive O(kNumShards) `active_handle_count.load()` and the
/// `notify_all()` only run once the cache is shutting down.
struct handle_release_notifier {
    std::mutex mtx;
    std::condition_variable cv;
    std::atomic<bool> shutdown_in_progress{false};
};

/// Thread-safe cache statistics (lock-free using atomics).
struct cache_stats {
    padded_atomic_size hits{};
    padded_atomic_size misses{};

    padded_atomic_size insertions{};
    padded_atomic_size evictions{};

    // T8.1: hot counters promoted to padded_atomic_size to eliminate
    // false sharing. current_size / current_memory are updated on every
    // insert/evict path; without padding they share a cache line with
    // each other and with max_size/max_memory (read-mostly), causing
    // unnecessary invalidations under striped / sharded workloads.
    padded_atomic_size current_size{};
    padded_atomic_size current_memory{};
    std::atomic<std::size_t> max_size{unlimited};
    std::atomic<std::size_t> max_memory{unlimited};

    // Hash table diagnostics
    std::atomic<float> hash_load_factor{0.0f};
    std::atomic<std::size_t> max_chain_length{0};

    // T13.1: Hash overload threshold — when load_factor exceeds this
    // value, an emergency rehash is triggered and the overload
    // callback (if set) is invoked. Default 2.0 matches the
    // historical hardcoded warning threshold in rehash_if_needed().
    // Lower it (e.g. 1.5) for latency-sensitive workloads; raise it
    // (e.g. 3.0) for memory-frugal workloads that tolerate longer
    // chains.
    std::atomic<float> hash_overload_threshold{2.0f};
    // T13.1: Cumulative count of overload events (load_factor
    // exceeded threshold). Useful for alerting on chronic under-
    // provisioning.
    std::atomic<std::size_t> hash_overload_events{0};

    // Lock contention diagnostics
    // P1-D: write-hot counters promoted to padded_atomic_size. Under
    // read-heavy-write-light workloads with frequent try_lock failures
    // (defer_promotion drain path) and TTL-checked gets, the unpadded
    // counters shared a cache line and caused false sharing across cores.
    padded_atomic_size write_lock_wait_count{};   // times lock_slow() was entered
    padded_atomic_size try_lock_fail_count{};      // times try_lock/try_lock_shared failed

    // Eviction pressure diagnostics
    padded_atomic_size eviction_search_steps{};    // total steps in find_eviction_victim()
    padded_atomic_size pinned_skip_count{};        // times pinned items were skipped during eviction

    // TTL diagnostics
    padded_atomic_size ttl_expired_count{};        // total expired items cleaned up
    std::atomic<std::size_t> ttl_cleanup_backlog{0};      // items still expired at last cleanup scan (read-mostly)
    // P1-10: Total TTL checks performed on the read path (peek_for_get).
    // Increments every time an item with expiry_ns != 0 is accessed —
    // whether or not it is expired. ttl_expired_count / ttl_checked_count
    // gives the expiration ratio, useful for sizing the TTL cleaner
    // interval: a high ratio (>10%) means the cleaner is not running
    // frequently enough and expired items are being discovered by readers
    // (adding latency to the read path).
    padded_atomic_size ttl_checked_count{};

    // Production: active read_handle count (refcount-pinned items) and TLS
    // ring backlog (deferred promotions not yet drained to MM).
    // T8.1: padded to their own cache lines — active_handle_count is
    // updated by every get()/peek() returning a handle (read-heavy
    // workloads), and tls_ring_backlog is updated by every deferred
    // promotion drain. Without padding they would share a line with
    // the read-mostly TTL counters above, causing cache-line ping-pong
    // between readers and the drain worker.
    //
    // P0-4: active_handle_count uses sharded_handle_counter (64 padded
    // shards) instead of a single padded_atomic_size. This eliminates
    // cache-line bouncing under high read concurrency: each thread writes
    // to its own shard (selected by hashing thread::id), so 64+ cores
    // doing get() no longer contend on one atomic. load() sums all 64
    // shards — acceptable because it's only called from stats_snapshot
    // and shutdown checks (infrequent), not from the read hot path.
    sharded_handle_counter active_handle_count{};
    padded_atomic_size tls_ring_backlog{};

    // Production: TLS ring overflow accounting — counts access events
    // dropped because the per-thread tls_access_ring was full and could
    // not defer the promotion. Padded to its own cache line because
    // overflow bursts from multiple threads can otherwise contend on
    // the same line as tls_ring_backlog. Populated by the cache layer
    // via drain_dropped_count() (see tls_ring.hpp).
    padded_atomic_size tls_ring_dropped_promotions{};

    // P1-1: Rehash diagnostics — track hash table expansion frequency,
    // duration, and migration volume. High rehash_count or long
    // rehash_total_time_ns indicates frequent capacity growth, which
    // causes tail latency spikes. Use reserve() to pre-size the table.
    std::atomic<std::size_t> rehash_count{0};              // number of rehash operations
    std::atomic<std::uint64_t> rehash_total_time_ns{0};    // cumulative rehash duration (ns)
    std::atomic<std::size_t> rehash_migrated_items{0};     // total items moved during rehash

    // T11.3: Number of writes blocked by a non-incremental (blocking) rehash.
    // Non-zero values indicate the user should enable incremental rehash
    // (`set_rehash_strategy("incremental")`) to avoid stalling writers
    // during hash table expansion.
    std::atomic<std::size_t> rehash_blocked_writes_count{0};

    // P1-5: Number of times find_and_pin_lockfree fell back to the
    // lock-protected path because the target segment was in incremental
    // rehash. Non-zero values indicate the lock-free read path is being
    // degraded by rehash activity — operators should consider pre-reserving
    // capacity (`reserve()`) to avoid runtime rehash.
    std::atomic<std::size_t> rehash_lockfree_fallback_count{0};

    // P1-1: TLS ring flush accounting — tracks how often the deferred
    // promotion ring is drained. High flush_count with low backlog
    // indicates healthy drain behavior; low flush_count with high
    // backlog indicates the drain worker is not running frequently
    // enough.
    std::atomic<std::size_t> tls_ring_flush_count{0};      // times drain_access_ring() was called

    // P0-1: hazptr/EBR reclaim diagnostics — track deferred reclamation
    // health. High pending_reclaim_count with low reclaim_total growth
    // indicates the reclaim worker is not running or too many handles
    // are long-lived. These are snapshots read from hazptr_domain /
    // epoch_domain at stats_snapshot() time.
    std::atomic<std::size_t> reclaim_pending_count{0};     // current pending retire count
    std::atomic<std::size_t> reclaim_total{0};             // cumulative reclaimed objects
    std::atomic<std::size_t> reclaim_freed_bytes{0};       // estimated bytes freed (approx)
    std::atomic<std::size_t> reclaim_invocation_count{0};  // times try_reclaim() was invoked

    // T-M1: singleflight / cache stampede diagnostics — counts follower
    // requests that were collapsed into a leader's in-flight provider
    // call. Non-zero values indicate the stampede protection is actively
    // preventing thundering-herd provider invocations on hot keys.
    std::atomic<std::size_t> stampede_coalesced_count{0};  // followers collapsed into leader

    // T-G11: refcount overflow diagnostics — bumped every time `incRef()`
    // returns kIncFailedOverflow and the caller (try_get / get /
    // get_or_fetch) observes an empty handle as a result. Non-zero values
    // indicate a key is being pinned by ~4 billion concurrent handles,
    // which is unreachable in practice but possible in pathological
    // handle-leak scenarios. Operators should alert on any non-zero
    // value: it almost certainly indicates a handle leak in user code.
    padded_atomic_size incRef_overflow_count{};

    // Latency histograms (ns) for get() and set() hot paths.
    // mutable: record() is logically const (atomic counters).
    mutable detail::latency_histogram get_latency{};
    mutable detail::latency_histogram set_latency{};

    // Lock wait latency histograms (ns) — read and write lock contention.
    mutable detail::latency_histogram read_lock_wait_latency{};
    mutable detail::latency_histogram write_lock_wait_latency{};

    // Eviction search steps histogram — distribution of steps taken
    // to find an eviction victim (indicates pinned-item pressure).
    mutable detail::latency_histogram eviction_search_steps_hist{};

    // P1-4: Runtime toggle for latency tracking. When disabled,
    // scope_latency_timer skips clock reads entirely, saving ~10-20%
    // on hot-path overhead. Default: enabled for backward compatibility.
    std::atomic<bool> latency_tracking_enabled{true};

    /// SeqLock for consistent_snapshot() — allows lock-free reads of
    /// multiple atomic counters while writers (reset) are rare.
    mutable detail::seqlock snapshot_lock_;

    /// T-G10: Pointer to the cache's handle-release notifier. Non-null only
    /// on the cache's own `per_cache_stats_` instance (set during
    /// `init_production_features()`). Snapshots / copies default to nullptr
    /// because they are not wired to a live cache's condition_variable.
    /// NOT copied by the copy/move constructors or assignment operators.
    handle_release_notifier* release_notifier = nullptr;

    cache_stats() = default;

    cache_stats(const cache_stats& other)
        : hits{padded_atomic_size{other.hits.value.load(std::memory_order_relaxed)}}
        , misses{padded_atomic_size{other.misses.value.load(std::memory_order_relaxed)}}
        , insertions{padded_atomic_size{other.insertions.value.load(std::memory_order_relaxed)}}
        , evictions{padded_atomic_size{other.evictions.value.load(std::memory_order_relaxed)}}
        , current_size{padded_atomic_size{other.current_size.value.load(std::memory_order_relaxed)}}
        , current_memory{padded_atomic_size{other.current_memory.value.load(std::memory_order_relaxed)}}
        , max_size(other.max_size.load(std::memory_order_relaxed))
        , max_memory(other.max_memory.load(std::memory_order_relaxed))
        , hash_load_factor(other.hash_load_factor.load(std::memory_order_relaxed))
        , max_chain_length(other.max_chain_length.load(std::memory_order_relaxed))
        , hash_overload_threshold(other.hash_overload_threshold.load(std::memory_order_relaxed))
        , hash_overload_events(other.hash_overload_events.load(std::memory_order_relaxed))
        , write_lock_wait_count(other.write_lock_wait_count.load(std::memory_order_relaxed))
        , try_lock_fail_count(other.try_lock_fail_count.load(std::memory_order_relaxed))
        , eviction_search_steps(other.eviction_search_steps.load(std::memory_order_relaxed))
        , pinned_skip_count(other.pinned_skip_count.load(std::memory_order_relaxed))
        , ttl_expired_count(other.ttl_expired_count.load(std::memory_order_relaxed))
        , ttl_cleanup_backlog(other.ttl_cleanup_backlog.load(std::memory_order_relaxed))
        , ttl_checked_count(other.ttl_checked_count.load(std::memory_order_relaxed))
        , active_handle_count{sharded_handle_counter{other.active_handle_count.load(std::memory_order_relaxed)}}
        , tls_ring_backlog{padded_atomic_size{other.tls_ring_backlog.value.load(std::memory_order_relaxed)}}
        , tls_ring_dropped_promotions{padded_atomic_size{other.tls_ring_dropped_promotions.value.load(std::memory_order_relaxed)}}
        , rehash_count(other.rehash_count.load(std::memory_order_relaxed))
        , rehash_total_time_ns(other.rehash_total_time_ns.load(std::memory_order_relaxed))
        , rehash_migrated_items(other.rehash_migrated_items.load(std::memory_order_relaxed))
        , tls_ring_flush_count(other.tls_ring_flush_count.load(std::memory_order_relaxed))
        , reclaim_pending_count(other.reclaim_pending_count.load(std::memory_order_relaxed))
        , reclaim_total(other.reclaim_total.load(std::memory_order_relaxed))
        , reclaim_freed_bytes(other.reclaim_freed_bytes.load(std::memory_order_relaxed))
        , reclaim_invocation_count(other.reclaim_invocation_count.load(std::memory_order_relaxed))
        , stampede_coalesced_count(other.stampede_coalesced_count.load(std::memory_order_relaxed))
        , incRef_overflow_count{padded_atomic_size{other.incRef_overflow_count.value.load(std::memory_order_relaxed)}}
        , get_latency(other.get_latency)
        , set_latency(other.set_latency)
        , read_lock_wait_latency(other.read_lock_wait_latency)
        , write_lock_wait_latency(other.write_lock_wait_latency)
        , eviction_search_steps_hist(other.eviction_search_steps_hist)
        , latency_tracking_enabled(other.latency_tracking_enabled.load(std::memory_order_relaxed))
        , snapshot_lock_() {}  // default-constructed, not copied

    cache_stats(cache_stats&& other) noexcept
        : hits{padded_atomic_size{other.hits.value.load(std::memory_order_relaxed)}}
        , misses{padded_atomic_size{other.misses.value.load(std::memory_order_relaxed)}}
        , insertions{padded_atomic_size{other.insertions.value.load(std::memory_order_relaxed)}}
        , evictions{padded_atomic_size{other.evictions.value.load(std::memory_order_relaxed)}}
        , current_size{padded_atomic_size{other.current_size.value.load(std::memory_order_relaxed)}}
        , current_memory{padded_atomic_size{other.current_memory.value.load(std::memory_order_relaxed)}}
        , max_size(other.max_size.load(std::memory_order_relaxed))
        , max_memory(other.max_memory.load(std::memory_order_relaxed))
        , hash_load_factor(other.hash_load_factor.load(std::memory_order_relaxed))
        , max_chain_length(other.max_chain_length.load(std::memory_order_relaxed))
        , hash_overload_threshold(other.hash_overload_threshold.load(std::memory_order_relaxed))
        , hash_overload_events(other.hash_overload_events.load(std::memory_order_relaxed))
        , write_lock_wait_count(other.write_lock_wait_count.load(std::memory_order_relaxed))
        , try_lock_fail_count(other.try_lock_fail_count.load(std::memory_order_relaxed))
        , eviction_search_steps(other.eviction_search_steps.load(std::memory_order_relaxed))
        , pinned_skip_count(other.pinned_skip_count.load(std::memory_order_relaxed))
        , ttl_expired_count(other.ttl_expired_count.load(std::memory_order_relaxed))
        , ttl_cleanup_backlog(other.ttl_cleanup_backlog.load(std::memory_order_relaxed))
        , ttl_checked_count(other.ttl_checked_count.load(std::memory_order_relaxed))
        , active_handle_count{sharded_handle_counter{other.active_handle_count.load(std::memory_order_relaxed)}}
        , tls_ring_backlog{padded_atomic_size{other.tls_ring_backlog.value.load(std::memory_order_relaxed)}}
        , tls_ring_dropped_promotions{padded_atomic_size{other.tls_ring_dropped_promotions.value.load(std::memory_order_relaxed)}}
        , rehash_count(other.rehash_count.load(std::memory_order_relaxed))
        , rehash_total_time_ns(other.rehash_total_time_ns.load(std::memory_order_relaxed))
        , rehash_migrated_items(other.rehash_migrated_items.load(std::memory_order_relaxed))
        , tls_ring_flush_count(other.tls_ring_flush_count.load(std::memory_order_relaxed))
        , reclaim_pending_count(other.reclaim_pending_count.load(std::memory_order_relaxed))
        , reclaim_total(other.reclaim_total.load(std::memory_order_relaxed))
        , reclaim_freed_bytes(other.reclaim_freed_bytes.load(std::memory_order_relaxed))
        , reclaim_invocation_count(other.reclaim_invocation_count.load(std::memory_order_relaxed))
        , stampede_coalesced_count(other.stampede_coalesced_count.load(std::memory_order_relaxed))
        , incRef_overflow_count{padded_atomic_size{other.incRef_overflow_count.value.load(std::memory_order_relaxed)}}
        , get_latency(std::move(other.get_latency))
        , set_latency(std::move(other.set_latency))
        , read_lock_wait_latency(std::move(other.read_lock_wait_latency))
        , write_lock_wait_latency(std::move(other.write_lock_wait_latency))
        , eviction_search_steps_hist(std::move(other.eviction_search_steps_hist))
        , latency_tracking_enabled(other.latency_tracking_enabled.load(std::memory_order_relaxed))
        , snapshot_lock_() {}  // default-constructed, not moved

    cache_stats& operator=(const cache_stats& other) {
        if (this != &other) {
            // Use seqlock write guards on both sides to prevent torn reads
            // from consistent_snapshot() during assignment.
            detail::seqlock::write_guard lhs_guard(snapshot_lock_);
            detail::seqlock::write_guard rhs_guard(other.snapshot_lock_);
            hits.value.store(other.hits.value.load(std::memory_order_relaxed), std::memory_order_relaxed);
            misses.value.store(other.misses.value.load(std::memory_order_relaxed), std::memory_order_relaxed);
            insertions.value.store(other.insertions.value.load(std::memory_order_relaxed), std::memory_order_relaxed);
            evictions.value.store(other.evictions.value.load(std::memory_order_relaxed), std::memory_order_relaxed);
            current_size.store(other.current_size.load(std::memory_order_relaxed), std::memory_order_relaxed);
            current_memory.store(other.current_memory.load(std::memory_order_relaxed), std::memory_order_relaxed);
            max_size.store(other.max_size.load(std::memory_order_relaxed), std::memory_order_relaxed);
            max_memory.store(other.max_memory.load(std::memory_order_relaxed), std::memory_order_relaxed);
            hash_load_factor.store(other.hash_load_factor.load(std::memory_order_relaxed), std::memory_order_relaxed);
            max_chain_length.store(other.max_chain_length.load(std::memory_order_relaxed), std::memory_order_relaxed);
            hash_overload_threshold.store(other.hash_overload_threshold.load(std::memory_order_relaxed), std::memory_order_relaxed);
            hash_overload_events.store(other.hash_overload_events.load(std::memory_order_relaxed), std::memory_order_relaxed);
            write_lock_wait_count.store(other.write_lock_wait_count.load(std::memory_order_relaxed), std::memory_order_relaxed);
            try_lock_fail_count.store(other.try_lock_fail_count.load(std::memory_order_relaxed), std::memory_order_relaxed);
            eviction_search_steps.store(other.eviction_search_steps.load(std::memory_order_relaxed), std::memory_order_relaxed);
            pinned_skip_count.store(other.pinned_skip_count.load(std::memory_order_relaxed), std::memory_order_relaxed);
            ttl_expired_count.store(other.ttl_expired_count.load(std::memory_order_relaxed), std::memory_order_relaxed);
            ttl_cleanup_backlog.store(other.ttl_cleanup_backlog.load(std::memory_order_relaxed), std::memory_order_relaxed);
            ttl_checked_count.store(other.ttl_checked_count.load(std::memory_order_relaxed), std::memory_order_relaxed);
            active_handle_count.store(other.active_handle_count.load(std::memory_order_relaxed), std::memory_order_relaxed);
            tls_ring_backlog.store(other.tls_ring_backlog.load(std::memory_order_relaxed), std::memory_order_relaxed);
            tls_ring_dropped_promotions.value.store(other.tls_ring_dropped_promotions.value.load(std::memory_order_relaxed), std::memory_order_relaxed);
            rehash_count.store(other.rehash_count.load(std::memory_order_relaxed), std::memory_order_relaxed);
            rehash_total_time_ns.store(other.rehash_total_time_ns.load(std::memory_order_relaxed), std::memory_order_relaxed);
            rehash_migrated_items.store(other.rehash_migrated_items.load(std::memory_order_relaxed), std::memory_order_relaxed);
            tls_ring_flush_count.store(other.tls_ring_flush_count.load(std::memory_order_relaxed), std::memory_order_relaxed);
            reclaim_pending_count.store(other.reclaim_pending_count.load(std::memory_order_relaxed), std::memory_order_relaxed);
            reclaim_total.store(other.reclaim_total.load(std::memory_order_relaxed), std::memory_order_relaxed);
            reclaim_freed_bytes.store(other.reclaim_freed_bytes.load(std::memory_order_relaxed), std::memory_order_relaxed);
            reclaim_invocation_count.store(other.reclaim_invocation_count.load(std::memory_order_relaxed), std::memory_order_relaxed);
            stampede_coalesced_count.store(other.stampede_coalesced_count.load(std::memory_order_relaxed), std::memory_order_relaxed);
            incRef_overflow_count.value.store(other.incRef_overflow_count.value.load(std::memory_order_relaxed), std::memory_order_relaxed);
            get_latency = other.get_latency;
            set_latency = other.set_latency;
            read_lock_wait_latency = other.read_lock_wait_latency;
            write_lock_wait_latency = other.write_lock_wait_latency;
            eviction_search_steps_hist = other.eviction_search_steps_hist;
            latency_tracking_enabled.store(other.latency_tracking_enabled.load(std::memory_order_relaxed), std::memory_order_relaxed);
        }
        return *this;
    }

    cache_stats& operator=(cache_stats&& other) {
        if (this != &other) {
            // Use seqlock write guards on both sides to prevent torn reads
            // from consistent_snapshot() during assignment.
            detail::seqlock::write_guard lhs_guard(snapshot_lock_);
            detail::seqlock::write_guard rhs_guard(other.snapshot_lock_);
            hits.value.store(other.hits.value.load(std::memory_order_relaxed), std::memory_order_relaxed);
            misses.value.store(other.misses.value.load(std::memory_order_relaxed), std::memory_order_relaxed);
            insertions.value.store(other.insertions.value.load(std::memory_order_relaxed), std::memory_order_relaxed);
            evictions.value.store(other.evictions.value.load(std::memory_order_relaxed), std::memory_order_relaxed);
            current_size.store(other.current_size.load(std::memory_order_relaxed), std::memory_order_relaxed);
            current_memory.store(other.current_memory.load(std::memory_order_relaxed), std::memory_order_relaxed);
            max_size.store(other.max_size.load(std::memory_order_relaxed), std::memory_order_relaxed);
            max_memory.store(other.max_memory.load(std::memory_order_relaxed), std::memory_order_relaxed);
            hash_load_factor.store(other.hash_load_factor.load(std::memory_order_relaxed), std::memory_order_relaxed);
            max_chain_length.store(other.max_chain_length.load(std::memory_order_relaxed), std::memory_order_relaxed);
            hash_overload_threshold.store(other.hash_overload_threshold.load(std::memory_order_relaxed), std::memory_order_relaxed);
            hash_overload_events.store(other.hash_overload_events.load(std::memory_order_relaxed), std::memory_order_relaxed);
            write_lock_wait_count.store(other.write_lock_wait_count.load(std::memory_order_relaxed), std::memory_order_relaxed);
            try_lock_fail_count.store(other.try_lock_fail_count.load(std::memory_order_relaxed), std::memory_order_relaxed);
            eviction_search_steps.store(other.eviction_search_steps.load(std::memory_order_relaxed), std::memory_order_relaxed);
            pinned_skip_count.store(other.pinned_skip_count.load(std::memory_order_relaxed), std::memory_order_relaxed);
            ttl_expired_count.store(other.ttl_expired_count.load(std::memory_order_relaxed), std::memory_order_relaxed);
            ttl_cleanup_backlog.store(other.ttl_cleanup_backlog.load(std::memory_order_relaxed), std::memory_order_relaxed);
            ttl_checked_count.store(other.ttl_checked_count.load(std::memory_order_relaxed), std::memory_order_relaxed);
            active_handle_count.store(other.active_handle_count.load(std::memory_order_relaxed), std::memory_order_relaxed);
            tls_ring_backlog.store(other.tls_ring_backlog.load(std::memory_order_relaxed), std::memory_order_relaxed);
            tls_ring_dropped_promotions.value.store(other.tls_ring_dropped_promotions.value.load(std::memory_order_relaxed), std::memory_order_relaxed);
            rehash_count.store(other.rehash_count.load(std::memory_order_relaxed), std::memory_order_relaxed);
            rehash_total_time_ns.store(other.rehash_total_time_ns.load(std::memory_order_relaxed), std::memory_order_relaxed);
            rehash_migrated_items.store(other.rehash_migrated_items.load(std::memory_order_relaxed), std::memory_order_relaxed);
            tls_ring_flush_count.store(other.tls_ring_flush_count.load(std::memory_order_relaxed), std::memory_order_relaxed);
            reclaim_pending_count.store(other.reclaim_pending_count.load(std::memory_order_relaxed), std::memory_order_relaxed);
            reclaim_total.store(other.reclaim_total.load(std::memory_order_relaxed), std::memory_order_relaxed);
            reclaim_freed_bytes.store(other.reclaim_freed_bytes.load(std::memory_order_relaxed), std::memory_order_relaxed);
            reclaim_invocation_count.store(other.reclaim_invocation_count.load(std::memory_order_relaxed), std::memory_order_relaxed);
            stampede_coalesced_count.store(other.stampede_coalesced_count.load(std::memory_order_relaxed), std::memory_order_relaxed);
            incRef_overflow_count.value.store(other.incRef_overflow_count.value.load(std::memory_order_relaxed), std::memory_order_relaxed);
            get_latency = std::move(other.get_latency);
            set_latency = std::move(other.set_latency);
            read_lock_wait_latency = std::move(other.read_lock_wait_latency);
            write_lock_wait_latency = std::move(other.write_lock_wait_latency);
            eviction_search_steps_hist = std::move(other.eviction_search_steps_hist);
            latency_tracking_enabled.store(other.latency_tracking_enabled.load(std::memory_order_relaxed), std::memory_order_relaxed);
        }
        return *this;
    }

    /// Register a cache hit.
    void register_hit() noexcept { hits.value.fetch_add(1, std::memory_order_relaxed); }

    /// Register a cache miss.
    void register_miss() noexcept { misses.value.fetch_add(1, std::memory_order_relaxed); }

    /// Register an insertion.
    void register_insertion() noexcept { insertions.value.fetch_add(1, std::memory_order_relaxed); }

    /// Register an eviction.
    void register_eviction() noexcept { evictions.value.fetch_add(1, std::memory_order_relaxed); }

    /// Bulk-register cache hits (e.g. batched from a shard or TLS buffer).
    void bulk_register_hits(std::size_t count) noexcept { hits.value.fetch_add(count, std::memory_order_relaxed); }

    /// Bulk-register cache misses (e.g. batched from a shard or TLS buffer).
    void bulk_register_misses(std::size_t count) noexcept { misses.value.fetch_add(count, std::memory_order_relaxed); }

    /// Bulk-register insertions.
    void bulk_register_insertions(std::size_t count) noexcept { insertions.value.fetch_add(count, std::memory_order_relaxed); }

    /// Bulk-register evictions.
    void bulk_register_evictions(std::size_t count) noexcept { evictions.value.fetch_add(count, std::memory_order_relaxed); }

    /// Get total accesses (hits + misses).
    std::size_t total_accesses() const noexcept {
        return hits.value.load(std::memory_order_relaxed) + misses.value.load(std::memory_order_relaxed);
    }

    /// Returns a consistent snapshot of all counters.
    /// Uses a seqlock to ensure all counter values come from the same
    /// logical point in time — lock-free for readers, only writers
    /// (reset) incur the lock overhead.
    cache_stats consistent_snapshot() const {
        std::uint32_t seq;
        cache_stats snap;
        do {
            seq = snapshot_lock_.read_begin();
            snap.hits.value.store(hits.value.load(std::memory_order_relaxed), std::memory_order_relaxed);
            snap.misses.value.store(misses.value.load(std::memory_order_relaxed), std::memory_order_relaxed);
            snap.insertions.value.store(insertions.value.load(std::memory_order_relaxed), std::memory_order_relaxed);
            snap.evictions.value.store(evictions.value.load(std::memory_order_relaxed), std::memory_order_relaxed);
            snap.current_size.store(current_size.load(std::memory_order_relaxed), std::memory_order_relaxed);
            snap.current_memory.store(current_memory.load(std::memory_order_relaxed), std::memory_order_relaxed);
            snap.max_size.store(max_size.load(std::memory_order_relaxed), std::memory_order_relaxed);
            snap.max_memory.store(max_memory.load(std::memory_order_relaxed), std::memory_order_relaxed);
            snap.hash_load_factor.store(hash_load_factor.load(std::memory_order_relaxed), std::memory_order_relaxed);
            snap.max_chain_length.store(max_chain_length.load(std::memory_order_relaxed), std::memory_order_relaxed);
            snap.hash_overload_threshold.store(hash_overload_threshold.load(std::memory_order_relaxed), std::memory_order_relaxed);
            snap.hash_overload_events.store(hash_overload_events.load(std::memory_order_relaxed), std::memory_order_relaxed);
            snap.write_lock_wait_count.store(write_lock_wait_count.load(std::memory_order_relaxed), std::memory_order_relaxed);
            snap.try_lock_fail_count.store(try_lock_fail_count.load(std::memory_order_relaxed), std::memory_order_relaxed);
            snap.eviction_search_steps.store(eviction_search_steps.load(std::memory_order_relaxed), std::memory_order_relaxed);
            snap.pinned_skip_count.store(pinned_skip_count.load(std::memory_order_relaxed), std::memory_order_relaxed);
            snap.ttl_expired_count.store(ttl_expired_count.load(std::memory_order_relaxed), std::memory_order_relaxed);
            snap.ttl_cleanup_backlog.store(ttl_cleanup_backlog.load(std::memory_order_relaxed), std::memory_order_relaxed);
            snap.ttl_checked_count.store(ttl_checked_count.load(std::memory_order_relaxed), std::memory_order_relaxed);
            snap.active_handle_count.store(active_handle_count.load(std::memory_order_relaxed), std::memory_order_relaxed);
            snap.tls_ring_backlog.store(tls_ring_backlog.load(std::memory_order_relaxed), std::memory_order_relaxed);
            snap.tls_ring_dropped_promotions.value.store(tls_ring_dropped_promotions.value.load(std::memory_order_relaxed), std::memory_order_relaxed);
            // P1-1: Rehash and TLS ring flush diagnostics.
            snap.rehash_count.store(rehash_count.load(std::memory_order_relaxed), std::memory_order_relaxed);
            snap.rehash_total_time_ns.store(rehash_total_time_ns.load(std::memory_order_relaxed), std::memory_order_relaxed);
            snap.rehash_migrated_items.store(rehash_migrated_items.load(std::memory_order_relaxed), std::memory_order_relaxed);
            snap.tls_ring_flush_count.store(tls_ring_flush_count.load(std::memory_order_relaxed), std::memory_order_relaxed);
            // P1-5: Lock-free read path fallback count (per-segment rehash)
            snap.rehash_lockfree_fallback_count.store(rehash_lockfree_fallback_count.load(std::memory_order_relaxed), std::memory_order_relaxed);
            // P0-1: hazptr/EBR reclaim diagnostics
            snap.reclaim_pending_count.store(reclaim_pending_count.load(std::memory_order_relaxed), std::memory_order_relaxed);
            snap.reclaim_total.store(reclaim_total.load(std::memory_order_relaxed), std::memory_order_relaxed);
            snap.reclaim_freed_bytes.store(reclaim_freed_bytes.load(std::memory_order_relaxed), std::memory_order_relaxed);
            snap.reclaim_invocation_count.store(reclaim_invocation_count.load(std::memory_order_relaxed), std::memory_order_relaxed);
            // T-M1: singleflight stampede coalescing diagnostics
            snap.stampede_coalesced_count.store(stampede_coalesced_count.load(std::memory_order_relaxed), std::memory_order_relaxed);
            // Histograms use relaxed atomic loads internally; copy them out.
            snap.get_latency = get_latency;
            snap.set_latency = set_latency;
            snap.read_lock_wait_latency = read_lock_wait_latency;
            snap.write_lock_wait_latency = write_lock_wait_latency;
            snap.eviction_search_steps_hist = eviction_search_steps_hist;
            snap.latency_tracking_enabled.store(latency_tracking_enabled.load(std::memory_order_relaxed), std::memory_order_relaxed);
        } while (snapshot_lock_.read_retry(seq));
        return snap;
    }

    /// Get hit rate [0.0, 1.0].
    /// Approximate: reads hits and misses in separate atomic loads,
    /// so a TOCTOU window exists between the two loads.
    /// For a consistent snapshot, use consistent_snapshot().hit_rate() instead.
    double hit_rate() const noexcept {
        auto h = hits.value.load(std::memory_order_relaxed);
        auto m = misses.value.load(std::memory_order_relaxed);
        auto total = h + m;
        if (total == 0) return 0.0;
        auto rate = static_cast<double>(h) / static_cast<double>(total);
        // Clamp to [0,1] — TOCTOU between two atomic loads may cause minor inconsistency
        return (rate > 1.0) ? 1.0 : ((rate < 0.0) ? 0.0 : rate);
    }

    /// Get miss rate [0.0, 1.0].
    /// Approximate: inherits the same TOCTOU caveat as hit_rate().
    /// For a consistent snapshot, use consistent_snapshot().miss_rate() instead.
    double miss_rate() const noexcept { return 1.0 - hit_rate(); }

    /// Get a consistent hit rate using the seqlock snapshot.
    /// Unlike hit_rate(), this guarantees hits and misses are read atomically
    /// with respect to reset operations, so the ratio is always in [0,1].
    double consistent_hit_rate() const {
        auto snap = consistent_snapshot();
        return snap.hit_rate();
    }

    /// Reset all counters (but not limits).
    void reset_counters() {
        detail::seqlock::write_guard guard(snapshot_lock_);
        hits.value.store(0, std::memory_order_relaxed);
        misses.value.store(0, std::memory_order_relaxed);
        insertions.value.store(0, std::memory_order_relaxed);
        evictions.value.store(0, std::memory_order_relaxed);
        write_lock_wait_count.store(0, std::memory_order_relaxed);
        try_lock_fail_count.store(0, std::memory_order_relaxed);
        eviction_search_steps.store(0, std::memory_order_relaxed);
        pinned_skip_count.store(0, std::memory_order_relaxed);
        ttl_expired_count.store(0, std::memory_order_relaxed);
        ttl_cleanup_backlog.store(0, std::memory_order_relaxed);
        ttl_checked_count.store(0, std::memory_order_relaxed);
        active_handle_count.store(0, std::memory_order_relaxed);
        tls_ring_backlog.store(0, std::memory_order_relaxed);
        tls_ring_dropped_promotions.value.store(0, std::memory_order_relaxed);
        // T13.1: hash_overload_events is a counter — reset on reset_counters().
        // hash_overload_threshold is configuration — preserved across resets.
        hash_overload_events.store(0, std::memory_order_relaxed);
        get_latency.reset();
        set_latency.reset();
        read_lock_wait_latency.reset();
        write_lock_wait_latency.reset();
        eviction_search_steps_hist.reset();
    }

    /// Reset everything including limits.
    void reset_all() {
        detail::seqlock::write_guard guard(snapshot_lock_);
        hits.value.store(0, std::memory_order_relaxed);
        misses.value.store(0, std::memory_order_relaxed);
        insertions.value.store(0, std::memory_order_relaxed);
        evictions.value.store(0, std::memory_order_relaxed);
        current_size.store(0, std::memory_order_relaxed);
        current_memory.store(0, std::memory_order_relaxed);
        max_size.store(unlimited, std::memory_order_relaxed);
        max_memory.store(unlimited, std::memory_order_relaxed);
        hash_load_factor.store(0.0f, std::memory_order_relaxed);
        max_chain_length.store(0, std::memory_order_relaxed);
        // T13.1: reset_all() resets counters AND configuration.
        hash_overload_threshold.store(2.0f, std::memory_order_relaxed);
        hash_overload_events.store(0, std::memory_order_relaxed);
        write_lock_wait_count.store(0, std::memory_order_relaxed);
        try_lock_fail_count.store(0, std::memory_order_relaxed);
        eviction_search_steps.store(0, std::memory_order_relaxed);
        pinned_skip_count.store(0, std::memory_order_relaxed);
        active_handle_count.store(0, std::memory_order_relaxed);
        tls_ring_backlog.store(0, std::memory_order_relaxed);
        tls_ring_dropped_promotions.value.store(0, std::memory_order_relaxed);
        reclaim_pending_count.store(0, std::memory_order_relaxed);
        reclaim_total.store(0, std::memory_order_relaxed);
        reclaim_freed_bytes.store(0, std::memory_order_relaxed);
        reclaim_invocation_count.store(0, std::memory_order_relaxed);
        stampede_coalesced_count.store(0, std::memory_order_relaxed);
        get_latency.reset();
        set_latency.reset();
    }

    /// Convert to string representation.
    std::string to_string() const {
        auto fmt_size = [](std::size_t s) {
            return s == unlimited ? "inf" : std::to_string(s);
        };
        return std::format(
            "hits={} misses={} insertions={} evictions={} "
            "maxsize={} currsize={} maxmem={} currmem={} "
            "hash_lf={:.2f} max_chain={} "
            "write_lock_waits={} try_lock_fails={} "
            "eviction_steps={} pinned_skips={} "
            "tls_dropped_promos={} "
            "reclaim_pending={} reclaim_total={} reclaim_invocations={} "
            "stampede_coalesced={} "
            "hit_rate={:.2f}%",
            hits.value.load(std::memory_order_relaxed),
            misses.value.load(std::memory_order_relaxed),
            insertions.value.load(std::memory_order_relaxed),
            evictions.value.load(std::memory_order_relaxed),
            fmt_size(max_size.load(std::memory_order_relaxed)),
            current_size.load(std::memory_order_relaxed),
            fmt_size(max_memory.load(std::memory_order_relaxed)),
            current_memory.load(std::memory_order_relaxed),
            hash_load_factor.load(std::memory_order_relaxed),
            max_chain_length.load(std::memory_order_relaxed),
            write_lock_wait_count.load(std::memory_order_relaxed),
            try_lock_fail_count.load(std::memory_order_relaxed),
            eviction_search_steps.load(std::memory_order_relaxed),
            pinned_skip_count.load(std::memory_order_relaxed),
            tls_ring_dropped_promotions.value.load(std::memory_order_relaxed),
            reclaim_pending_count.load(std::memory_order_relaxed),
            reclaim_total.load(std::memory_order_relaxed),
            reclaim_invocation_count.load(std::memory_order_relaxed),
            stampede_coalesced_count.load(std::memory_order_relaxed),
            hit_rate() * 100.0);
    }

    bool operator==(const cache_stats& other) const noexcept {
        return hits.value.load(std::memory_order_relaxed) == other.hits.value.load(std::memory_order_relaxed)
            && misses.value.load(std::memory_order_relaxed) == other.misses.value.load(std::memory_order_relaxed)
            && insertions.value.load(std::memory_order_relaxed) == other.insertions.value.load(std::memory_order_relaxed)
            && evictions.value.load(std::memory_order_relaxed) == other.evictions.value.load(std::memory_order_relaxed)
            && current_size.load(std::memory_order_relaxed) == other.current_size.load(std::memory_order_relaxed)
            && current_memory.load(std::memory_order_relaxed) == other.current_memory.load(std::memory_order_relaxed)
            && max_size.load(std::memory_order_relaxed) == other.max_size.load(std::memory_order_relaxed)
            && max_memory.load(std::memory_order_relaxed) == other.max_memory.load(std::memory_order_relaxed)
            && hash_load_factor.load(std::memory_order_relaxed) == other.hash_load_factor.load(std::memory_order_relaxed)
            && max_chain_length.load(std::memory_order_relaxed) == other.max_chain_length.load(std::memory_order_relaxed)
            && write_lock_wait_count.load(std::memory_order_relaxed) == other.write_lock_wait_count.load(std::memory_order_relaxed)
            && try_lock_fail_count.load(std::memory_order_relaxed) == other.try_lock_fail_count.load(std::memory_order_relaxed)
            && eviction_search_steps.load(std::memory_order_relaxed) == other.eviction_search_steps.load(std::memory_order_relaxed)
            && pinned_skip_count.load(std::memory_order_relaxed) == other.pinned_skip_count.load(std::memory_order_relaxed);
    }

    bool operator!=(const cache_stats& other) const noexcept {
        return !(*this == other);
    }

    /// Combine statistics from multiple shards into a single aggregate.
    cache_stats operator+(const cache_stats& other) const {
        cache_stats result;
        result.hits.value.store(
            hits.value.load(std::memory_order_relaxed) + other.hits.value.load(std::memory_order_relaxed),
            std::memory_order_relaxed);
        result.misses.value.store(
            misses.value.load(std::memory_order_relaxed) + other.misses.value.load(std::memory_order_relaxed),
            std::memory_order_relaxed);
        result.insertions.value.store(
            insertions.value.load(std::memory_order_relaxed) + other.insertions.value.load(std::memory_order_relaxed),
            std::memory_order_relaxed);
        result.evictions.value.store(
            evictions.value.load(std::memory_order_relaxed) + other.evictions.value.load(std::memory_order_relaxed),
            std::memory_order_relaxed);
        result.current_size.store(
            current_size.load(std::memory_order_relaxed) + other.current_size.load(std::memory_order_relaxed),
            std::memory_order_relaxed);
        result.current_memory.store(
            current_memory.load(std::memory_order_relaxed) + other.current_memory.load(std::memory_order_relaxed),
            std::memory_order_relaxed);
        // max_size and max_memory are limits, not accumulators — keep the larger.
        result.max_size.store(
            std::max(max_size.load(std::memory_order_relaxed), other.max_size.load(std::memory_order_relaxed)),
            std::memory_order_relaxed);
        result.max_memory.store(
            std::max(max_memory.load(std::memory_order_relaxed), other.max_memory.load(std::memory_order_relaxed)),
            std::memory_order_relaxed);
        result.hash_load_factor.store(
            std::max(hash_load_factor.load(std::memory_order_relaxed), other.hash_load_factor.load(std::memory_order_relaxed)),
            std::memory_order_relaxed);
        result.max_chain_length.store(
            std::max(max_chain_length.load(std::memory_order_relaxed), other.max_chain_length.load(std::memory_order_relaxed)),
            std::memory_order_relaxed);
        // T13.1: hash_overload_threshold is configuration — keep the
        // smaller (stricter) of the two so aggregated stats don't
        // understate overload pressure. hash_overload_events is a
        // counter — accumulate.
        result.hash_overload_threshold.store(
            std::min(hash_overload_threshold.load(std::memory_order_relaxed),
                     other.hash_overload_threshold.load(std::memory_order_relaxed)),
            std::memory_order_relaxed);
        result.hash_overload_events.store(
            hash_overload_events.load(std::memory_order_relaxed) +
            other.hash_overload_events.load(std::memory_order_relaxed),
            std::memory_order_relaxed);
        // Observability counters are accumulators across shards
        result.write_lock_wait_count.store(
            write_lock_wait_count.load(std::memory_order_relaxed) + other.write_lock_wait_count.load(std::memory_order_relaxed),
            std::memory_order_relaxed);
        result.try_lock_fail_count.store(
            try_lock_fail_count.load(std::memory_order_relaxed) + other.try_lock_fail_count.load(std::memory_order_relaxed),
            std::memory_order_relaxed);
        result.eviction_search_steps.store(
            eviction_search_steps.load(std::memory_order_relaxed) + other.eviction_search_steps.load(std::memory_order_relaxed),
            std::memory_order_relaxed);
        result.pinned_skip_count.store(
            pinned_skip_count.load(std::memory_order_relaxed) + other.pinned_skip_count.load(std::memory_order_relaxed),
            std::memory_order_relaxed);
        // P1-10: TTL counters aggregated across shards.
        result.ttl_expired_count.store(
            ttl_expired_count.load(std::memory_order_relaxed) + other.ttl_expired_count.load(std::memory_order_relaxed),
            std::memory_order_relaxed);
        result.ttl_checked_count.store(
            ttl_checked_count.load(std::memory_order_relaxed) + other.ttl_checked_count.load(std::memory_order_relaxed),
            std::memory_order_relaxed);
        result.active_handle_count.store(
            active_handle_count.load(std::memory_order_relaxed) + other.active_handle_count.load(std::memory_order_relaxed),
            std::memory_order_relaxed);
        result.tls_ring_backlog.store(
            tls_ring_backlog.load(std::memory_order_relaxed) + other.tls_ring_backlog.load(std::memory_order_relaxed),
            std::memory_order_relaxed);
        result.tls_ring_dropped_promotions.value.store(
            tls_ring_dropped_promotions.value.load(std::memory_order_relaxed) + other.tls_ring_dropped_promotions.value.load(std::memory_order_relaxed),
            std::memory_order_relaxed);
        // P0-1: hazptr/EBR reclaim diagnostics aggregation
        result.reclaim_pending_count.store(
            reclaim_pending_count.load(std::memory_order_relaxed) + other.reclaim_pending_count.load(std::memory_order_relaxed),
            std::memory_order_relaxed);
        result.reclaim_total.store(
            reclaim_total.load(std::memory_order_relaxed) + other.reclaim_total.load(std::memory_order_relaxed),
            std::memory_order_relaxed);
        result.reclaim_freed_bytes.store(
            reclaim_freed_bytes.load(std::memory_order_relaxed) + other.reclaim_freed_bytes.load(std::memory_order_relaxed),
            std::memory_order_relaxed);
        result.reclaim_invocation_count.store(
            reclaim_invocation_count.load(std::memory_order_relaxed) + other.reclaim_invocation_count.load(std::memory_order_relaxed),
            std::memory_order_relaxed);
        // T-M1: singleflight stampede coalescing aggregated across shards
        result.stampede_coalesced_count.store(
            stampede_coalesced_count.load(std::memory_order_relaxed) + other.stampede_coalesced_count.load(std::memory_order_relaxed),
            std::memory_order_relaxed);
        // Aggregate latency histograms bucket-by-bucket.
        result.get_latency = get_latency;
        result.get_latency.merge_from(other.get_latency);
        result.set_latency = set_latency;
        result.set_latency.merge_from(other.set_latency);
        // Latency tracking: enabled if either side has it enabled
        result.latency_tracking_enabled.store(
            latency_tracking_enabled.load(std::memory_order_relaxed) ||
            other.latency_tracking_enabled.load(std::memory_order_relaxed),
            std::memory_order_relaxed);
        return result;
    }

    /// Approximate load factor based on current_size / bucket_count.
    /// Unlike hash_load_factor (which requires refresh_hash_stats()),
    /// this is O(1) and always available.
    double approximate_load_factor(std::size_t bucket_count) const noexcept {
        if (bucket_count == 0) return 0.0;
        return static_cast<double>(current_size.load(std::memory_order_relaxed)) /
               static_cast<double>(bucket_count);
    }
};

// T8.2: Opt-in paired-layout variant for single-threaded or low-contention
// scenarios. Trades false-sharing resistance for a smaller struct footprint
// (saves 64 bytes per instance). Use this when profiling shows the separate
// layout wastes too much L1 capacity (e.g. many small caches per thread).
// Production multi-threaded caches should keep the default `cache_stats`.
//
// Implementation note: this is a typedef rather than a full alias because
// the paired layout replaces two `padded_atomic_size` members with a single
// `hot_counters_pair`. Callers using `stats.hits.load()` must change to
// `stats.hits_misses.hits.load()`. The default `cache_stats` API is unchanged.
struct cache_stats_paired : cache_stats {
    // Inheriting from cache_stats preserves all the other counters. The
    // paired hits/misses live in this->hits_misses_ (added below); the
    // base class's separate hits/misses members still exist but are
    // unused — to keep the struct size benefit, callers should use
    // cache_stats_paired directly and access hits_misses_ instead of
    // hits/misses. This is intentionally a documentation-level opt-in
    // rather than a transparent swap, to avoid silent API breakage.
    hot_counters_pair hits_misses_{};
};

inline std::ostream& operator<<(std::ostream& out, const cache_stats& stats) {
    return out << stats.to_string();
}

// ============================================================================
// Per-Key Statistics
// ============================================================================

/// Statistics for a single monitored key.
struct key_stats {
    std::atomic<std::size_t> hits{0};
    std::atomic<std::size_t> misses{0};

    key_stats() = default;
    key_stats(const key_stats& other)
        : hits(other.hits.load(std::memory_order_relaxed)), misses(other.misses.load(std::memory_order_relaxed)) {}

    void register_hit() noexcept { hits.fetch_add(1, std::memory_order_relaxed); }
    void register_miss() noexcept { misses.fetch_add(1, std::memory_order_relaxed); }

    std::size_t total_accesses() const noexcept {
        return hits.load(std::memory_order_relaxed) + misses.load(std::memory_order_relaxed);
    }

    double hit_rate() const noexcept {
        auto total = total_accesses();
        if (total == 0) return 0.0;
        auto rate = static_cast<double>(hits.load(std::memory_order_relaxed)) / static_cast<double>(total);
        return (rate > 1.0) ? 1.0 : ((rate < 0.0) ? 0.0 : rate);
    }
};

/// Per-key statistics tracker.
/// Optimized for the common case where no keys are monitored:
/// register_hit()/register_miss() check an atomic flag first and return
/// immediately if no keys are being tracked, avoiding lock acquisition
/// on the hot path.
template <typename Key>
class key_statistics_tracker {
public:
    using key_type = Key;

    /// Register a key for monitoring.
    void monitor(const Key& key) {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        stats_.try_emplace(key);
        has_monitored_keys_.store(!stats_.empty(), std::memory_order_release);
    }

    /// Unregister a key from monitoring.
    void unmonitor(const Key& key) {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        stats_.erase(key);
        has_monitored_keys_.store(!stats_.empty(), std::memory_order_release);
    }

    /// Register a hit for the given key (if monitored).
    /// Fast path: returns immediately if no keys are being monitored.
    void register_hit(const Key& key) {
        if (!has_monitored_keys_.load(std::memory_order_acquire)) return;
        std::shared_lock<std::shared_mutex> lock(mutex_);
        if (auto it = stats_.find(key); it != stats_.end()) {
            it->second.register_hit();
        }
    }

    /// Register a miss for the given key (if monitored).
    /// Fast path: returns immediately if no keys are being monitored.
    void register_miss(const Key& key) {
        if (!has_monitored_keys_.load(std::memory_order_acquire)) return;
        std::shared_lock<std::shared_mutex> lock(mutex_);
        if (auto it = stats_.find(key); it != stats_.end()) {
            it->second.register_miss();
        }
    }

    /// Get statistics for a key.
    std::optional<key_stats> stats_for(const Key& key) const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        if (auto it = stats_.find(key); it != stats_.end()) {
            return it->second;
        }
        return std::nullopt;
    }

    /// Check if a key is being monitored.
    bool is_monitoring(const Key& key) const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        return stats_.contains(key);
    }

    /// Number of monitored keys.
    std::size_t monitored_count() const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        return stats_.size();
    }

    /// Clear all monitored keys.
    void clear() {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        stats_.clear();
        has_monitored_keys_.store(false, std::memory_order_release);
    }

private:
    mutable std::shared_mutex mutex_;
    std::unordered_map<Key, key_stats> stats_;
    std::atomic<bool> has_monitored_keys_{false};
};

/// Tag type to indicate a read_handle is constructed from an already-pinned item.
/// When this tag is passed, the read_handle constructor skips incRef() because
/// the caller has already incremented the reference count (e.g., via find_and_pin).
struct pre_pinned_t {
    explicit pre_pinned_t() = default;
};
inline constexpr pre_pinned_t pre_pinned{};

/// Cache Handle — prevents the referenced item from being evicted while held.
///
/// When get() returns a handle, the underlying cache_item's reference count is
/// incremented via refcount_with_flags::incRef(). As long as the handle is alive,
/// the item will not be evicted by evict_lru() (the eviction algorithm uses
/// markForEviction() which fails when access_ref > 0). The reference count is
/// decremented automatically when the handle is destroyed.
///
/// IMPORTANT PRODUCTION CONTRACT:
///   - read_handle only guards against ORDINARY EVICTION. It does NOT prevent
///     explicit del()/pop()/flush() on the same key from invalidating the value.
///   - Holding a read_handle across cache destruction is UNDEFINED BEHAVIOR.
///     The cache must outlive all outstanding handles.
///   - For a value that remains valid even after eviction or cache destruction,
///     use get_shared() which returns a heap-copied std::shared_ptr.
///
/// @tparam T Value type (may be const Value)
template <typename T>
class read_handle {
public:
    using value_type = T;

    // ----------------------------------------------------------------
    // Task 6 (P2-1 revised): Global active-handle counter.
    //
    // Each read_handle<T> instantiation maintains a single process-wide
    // atomic counter tracking the number of currently-alive handles with
    // a non-null refcount (i.e., handles that pin an item against
    // eviction).
    //
    // P2-1: This per-T global counter is a cache-line ping-pong hotspot
    // under high read QPS (>10M ops/s) when multiple caches share the
    // same Value type. It is now ONLY maintained under `-DLRU_DEBUG=1`.
    // In release builds the counter stays at 0 and active_handle_count
    // is sourced from per-cache cache_stats::active_handle_count via
    // the per_cache_stats_ pointer (see attach_per_cache_stats()).
    //
    // P2-2: The counter is now sharded (64 cache-line-aligned atomics)
    // with TLS-based shard assignment. Each thread increments/decrements
    // its own shard, eliminating cross-core cache-line bounce on the
    // hot read path. The 5-20 cycles of ping-pong per handle is gone;
    // the cost is a 64-way sum in `active_count()` (called rarely, e.g.
    // during shutdown or metrics scrape).
    //
    // Per-T (rather than per-cache) tracking is still useful in debug
    // builds for diagnosing handle leaks or excessive pinning across
    // the entire process. Production observability uses per-cache
    // tracking (now default-on in unified_cache).
    // ----------------------------------------------------------------
    static std::size_t active_count() noexcept {
#ifdef LRU_DEBUG
        return sum_sharded_active_count();
#else
        // Release builds: counter is maintained when global tracking is
        // enabled (O10 default: true). Callers that have disabled it via
        // `enable_global_handle_tracking(false)` should use per-cache
        // `active_handle_count()` on unified_cache instead.
        if (s_global_tracking_enabled_.load(std::memory_order_acquire)) {
            return sum_sharded_active_count();
        }
        return 0;
#endif
    }

    /// P2-1: Increment the per-T global counter. In release builds the
    /// counter is maintained when global tracking is enabled (O10 default:
    /// true) — pass `enable_global_handle_tracking(false)` to opt out.
    /// Per-cache tracking via per_cache_stats_ is always maintained when
    /// a per_cache_stats pointer is provided.
    ///
    /// T4.2: When `enable_global_handle_tracking(false)` has been called,
    /// this function becomes a no-op in release builds. Only handles
    /// created after the flag change are affected.
    ///
    /// P2-2: Uses TLS-based shard assignment so each thread increments
    /// its own cache-line-aligned shard — no cross-core ping-pong.
    static void inc_active_count_debug() noexcept {
#ifdef LRU_DEBUG
        sharded_active_count_inc();
#else
        if (s_global_tracking_enabled_.load(std::memory_order_relaxed)) {
            sharded_active_count_inc();
        }
#endif
    }

    /// P2-1: Decrement the per-T global counter. No-op in release builds.
    static void dec_active_count_debug() noexcept {
#ifdef LRU_DEBUG
        sharded_active_count_dec();
#else
        if (s_global_tracking_enabled_.load(std::memory_order_relaxed)) {
            sharded_active_count_dec();
        }
#endif
    }

    /// T4.2: Force-enable or disable the per-T global handle counter.
    ///
    /// O10: As of this revision the default is `true` in both debug and
    /// release builds — the global counter is always maintained so that
    /// `force_wait_handles()` and the destructor's `shutdown_and_wait()`
    /// work reliably without callers having to opt in at process start.
    /// Call `enable_global_handle_tracking(false)` to opt out when a
    /// benchmark is sensitive to the per-handle cache-line ping-pong
    /// and per-cache tracking (`set_per_cache_handle_tracking(true)`)
    /// is sufficient for observability.
    ///
    /// Once disabled, handles created after the call stop incrementing
    /// the global counter (existing live handles are not retroactively
    /// decremented). In debug builds (`-DLRU_DEBUG=1`) the counter is
    /// always maintained regardless of this flag.
    static void enable_global_handle_tracking(bool enabled) noexcept {
        s_global_tracking_enabled_.store(enabled, std::memory_order_release);
    }

    /// T4.2: Query whether the per-T global handle counter is currently
    /// being maintained at runtime.
    static bool is_global_handle_tracking_enabled() noexcept {
#ifdef LRU_DEBUG
        return true;
#else
        return s_global_tracking_enabled_.load(std::memory_order_acquire);
#endif
    }

    read_handle() noexcept = default;

    /// Primary constructor using refcount_with_flags (CAS-lockfree path).
    /// Calls incRef(); if it fails (item being evicted/moved), produces an
    /// empty handle.
    /// \param per_cache_stats Optional per-cache stats pointer for
    ///   active_handle_count isolation (Task 11). When non-null, the
    ///   per-cache counter is incremented/decremented in addition to the
    ///   global per-T counter.
    read_handle(T* value, detail::refcount_with_flags* refcount,
                cache_stats* per_cache_stats = nullptr) noexcept
        : value_(value), refcount_(refcount), per_cache_stats_(per_cache_stats) {
        if (refcount_) {
            auto result = refcount_->incRef();
            if (result != detail::IncResult::kIncOk) {
                // Item is being evicted or moved — produce empty handle
                value_ = nullptr;
                refcount_ = nullptr;
                per_cache_stats_ = nullptr;
            } else {
                inc_active_count_debug();
                if (per_cache_stats_) {
                    per_cache_stats_->active_handle_count.fetch_add(1, std::memory_order_relaxed);
                }
            }
        }
    }

    /// Pre-pinned constructor: item refcount already incremented by caller.
    /// Skips incRef() because the item was pinned atomically during
    /// concurrent_hash_table::find_and_pin(), which eliminates the
    /// TOCTOU window between find() and pinning.
    read_handle(T* value, detail::refcount_with_flags* refcount, pre_pinned_t,
                cache_stats* per_cache_stats = nullptr) noexcept
        : value_(value), refcount_(refcount), per_cache_stats_(per_cache_stats) {
        // refcount already incremented by caller — no incRef() needed
        if (refcount_) {
            inc_active_count_debug();
            if (per_cache_stats_) {
                per_cache_stats_->active_handle_count.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }

    /// Destructor: decrement the reference count.
    ~read_handle() noexcept { release(); }

    /// Copy constructor: shares the reference and increments the count.
    read_handle(const read_handle& other) noexcept
        : value_(other.value_),
          refcount_(other.refcount_),
          per_cache_stats_(other.per_cache_stats_) {
        if (refcount_) {
            auto result = refcount_->incRef();
            if (result != detail::IncResult::kIncOk) {
                value_ = nullptr;
                refcount_ = nullptr;
                per_cache_stats_ = nullptr;
            } else {
                inc_active_count_debug();
                if (per_cache_stats_) {
                    per_cache_stats_->active_handle_count.fetch_add(1, std::memory_order_relaxed);
                }
            }
        }
    }

    /// Conversion from read_handle<U> where U* is convertible to T*.
    template <typename U>
        requires std::convertible_to<U*, T*>
    read_handle(const read_handle<U>& other) noexcept
        : value_(other.get()),
          refcount_(other.refcount_ptr()),
          per_cache_stats_(other.per_cache_stats_ptr()) {
        if (refcount_) {
            auto result = refcount_->incRef();
            if (result != detail::IncResult::kIncOk) {
                value_ = nullptr;
                refcount_ = nullptr;
                per_cache_stats_ = nullptr;
            } else {
                inc_active_count_debug();
                if (per_cache_stats_) {
                    per_cache_stats_->active_handle_count.fetch_add(1, std::memory_order_relaxed);
                }
            }
        }
    }

    /// Copy assignment.
    read_handle& operator=(const read_handle& other) noexcept {
        if (this != &other) {
            // Increment new refcount first to prevent window where refcount=0
            cache_stats* new_stats = other.per_cache_stats_;
            if (other.refcount_) {
                auto result = other.refcount_->incRef();
                if (result != detail::IncResult::kIncOk) {
                    release();
                    value_ = nullptr;
                    refcount_ = nullptr;
                    per_cache_stats_ = nullptr;
                    return *this;
                }
                inc_active_count_debug();
                if (new_stats) {
                    new_stats->active_handle_count.fetch_add(1, std::memory_order_relaxed);
                }
            }
            release();
            value_ = other.value_;
            refcount_ = other.refcount_;
            per_cache_stats_ = new_stats;
        }
        return *this;
    }

    /// Conversion copy assignment.
    template <typename U>
        requires std::convertible_to<U*, T*>
    read_handle& operator=(const read_handle<U>& other) noexcept {
        if (other.refcount_) {
            auto result = other.refcount_->incRef();
            if (result != detail::IncResult::kIncOk) {
                release();
                value_ = nullptr;
                refcount_ = nullptr;
                return *this;
            }
            inc_active_count_debug();
        }
        release();
        value_ = other.get();
        refcount_ = other.refcount_ptr();
        return *this;
    }

    /// Move constructor: transfers ownership; the source is not decremented.
    read_handle(read_handle&& other) noexcept
        : value_(other.value_),
          refcount_(other.refcount_),
          per_cache_stats_(other.per_cache_stats_) {
        other.value_ = nullptr;
        other.refcount_ = nullptr;
        other.per_cache_stats_ = nullptr;
    }

    /// Move assignment.
    read_handle& operator=(read_handle&& other) noexcept {
        if (this != &other) {
            release();
            value_ = other.value_;
            refcount_ = other.refcount_;
            per_cache_stats_ = other.per_cache_stats_;
            other.value_ = nullptr;
            other.refcount_ = nullptr;
            other.per_cache_stats_ = nullptr;
        }
        return *this;
    }

    // ---- Accessors ----

    /// Release the handle (decrement the reference count).
    ///
    /// P1-7 (T2.6) — Evaluation: decRef归零时主动 retire 的可行性.
    ///
    /// We deliberately do NOT retire the item here when refcount hits
    /// zero. Reasons:
    ///
    /// 1. **Item is still in the LRU list.** decRef归零只表示没有
    ///    active handle，不代表 item 应该被回收。Item 仍然在 LRU
    ///    list 中，可能被后续 get() 命中（重新 incRef）。如果这里
    ///    主动 retire，item 会被 hazptr/EBR 延迟释放，但 LRU list
    ///    仍持有该指针 — 后续 list 遍历（evict_lru, pop_lru, 遍历
    ///    迭代器）会触发 UAF。
    ///
    /// 2. **Evict 路径已正确协调.** evict_lru() 先调用
    ///    markForEviction() 设置 kExclusive 标志，使后续 incRef
    ///    失败（防止新 handle pin 住正在被淘汰的 item）。然后才
    ///    retire + reclaim。decRef 路径绕过这个协调会破坏不变量。
    ///
    /// 3. **替代方案已实施.** 缩短 drain interval 到 500ms（当
    ///    active_handle_count > 0 时），见 cache_trait.hpp 的
    ///    start_event_drain()。这保证了即使 handle 频繁创建/释放，
    ///    pending list 也不会累积太久。
    ///
    /// 4. **生产部署必须调用 start_event_drain().** 见 CLAUDE.md /
    ///    AGENTS.md 中的明确要求。不调用会导致 retired 对象不
    ///    被回收，最终 OOM。
    void release() noexcept {
        if (refcount_) {
            refcount_->decRef();
            dec_active_count_debug();
            if (per_cache_stats_) {
                per_cache_stats_->active_handle_count.fetch_sub(1, std::memory_order_relaxed);
                // T-G10: If the owning cache is shutting down and this was
                // the last active handle, wake up `shutdown_and_wait()` so
                // it returns immediately instead of waiting out its poll
                // interval. The shutdown_in_progress load is a single
                // relaxed atomic — essentially free on the hot path when
                // no shutdown is in progress (the common case).
                auto* notifier = per_cache_stats_->release_notifier;
                if (notifier &&
                    notifier->shutdown_in_progress.load(std::memory_order_acquire) &&
                    per_cache_stats_->active_handle_count.load(std::memory_order_acquire) == 0) {
                    // G23: Acquire and immediately release the mutex before notify.
                    // This follows the condvar pattern: if a waiter is about to wait,
                    // holding the mutex here ensures its predicate check happens-before
                    // our notify_all, preventing lost wakeup.
                    { std::lock_guard<std::mutex> lk(notifier->mtx); }
                    notifier->cv.notify_all();
                }
            }
            refcount_ = nullptr;
            value_ = nullptr;
            per_cache_stats_ = nullptr;
        }
    }

    /// Get the value pointer (may be nullptr).
    T* get() noexcept { return value_; }
    const T* get() const noexcept { return value_; }

    T* operator->() {
        if (!value_) throw cache_config_exception("read_handle: null dereference");
        return value_;
    }
    const T* operator->() const {
        if (!value_) throw cache_config_exception("read_handle: null dereference");
        return value_;
    }
    T& operator*() {
        if (!value_) throw cache_config_exception("read_handle: null dereference");
        return *value_;
    }
    const T& operator*() const {
        if (!value_) throw cache_config_exception("read_handle: null dereference");
        return *value_;
    }

    explicit operator bool() const noexcept { return value_ != nullptr; }

    bool has_value() const noexcept { return value_ != nullptr; }

    /// Internal: expose the refcount_with_flags pointer for cross-instantiation conversions.
    detail::refcount_with_flags* refcount_ptr() const noexcept { return refcount_; }

    /// Internal: expose the per-cache stats pointer for cross-instantiation conversions.
    cache_stats* per_cache_stats_ptr() const noexcept { return per_cache_stats_; }

    /// Task C: Attach (or re-attach) a per-cache stats pointer to a handle
    /// that was constructed with nullptr (e.g., returned by peek_for_get).
    ///
    /// If `stats` is non-null and the handle currently pins an item
    /// (refcount_ != nullptr), the per-cache active_handle_count counter
    /// is incremented exactly once. If the handle is empty or `stats`
    /// equals the current per_cache_stats_, this is a no-op.
    ///
    /// Idempotent w.r.t. the same stats pointer: calling attach_per_cache_stats
    /// twice with the same pointer does NOT double-count.
    void attach_per_cache_stats(cache_stats* stats) noexcept {
        if (per_cache_stats_ == stats) return;  // already attached (or both null)
        // Detach from previous stats (decrement if we were counting).
        if (per_cache_stats_ && refcount_) {
            per_cache_stats_->active_handle_count.fetch_sub(1, std::memory_order_relaxed);
        }
        per_cache_stats_ = stats;
        // Attach to new stats (increment if non-null and handle is live).
        if (per_cache_stats_ && refcount_) {
            per_cache_stats_->active_handle_count.fetch_add(1, std::memory_order_relaxed);
        }
    }

private:
    T* value_ = nullptr;
    detail::refcount_with_flags* refcount_ = nullptr;
    cache_stats* per_cache_stats_ = nullptr;  // Task 11: optional per-cache stats

    // P2-2: Sharded active-handle counter. 64 cache-line-aligned atomics
    // keyed by TLS shard index. Each thread picks a shard on first use
    // and reuses it for the thread's lifetime, so inc/dec hit only the
    // thread's own cache line — no cross-core bounce. `active_count()`
    // sums all 64 shards (called rarely: shutdown, metrics scrape).
    //
    // The previous single-atomic counter (s_active_count_) caused 5-20
    // cycles of cache-line ping-pong per read_handle create/destroy
    // under high read QPS. The sharded layout eliminates that cost;
    // the only downside is a 64-way sum on the cold `active_count()`
    // path, which is negligible (64 relaxed loads + adds, <50ns).
    static constexpr std::size_t kActiveCountShards = 64;
    struct alignas(64) active_count_shard {
        std::atomic<std::size_t> count{0};
        char pad_[64 - sizeof(std::atomic<std::size_t>)];
    };
    static inline active_count_shard s_active_count_shards_[kActiveCountShards];

    /// P2-2: Pick a shard index for the current thread. Uses a TLS
    /// counter that increments per thread (wraps at kActiveCountShards)
    /// so threads spread across shards. The assignment is sticky for
    /// the thread's lifetime, ensuring inc/dec always hit the same
    /// cache line — no migration, no bounce.
    static std::size_t active_count_shard_index() noexcept {
        static thread_local std::size_t tls_shard = kActiveCountShards;
        if (tls_shard == kActiveCountShards) {
            // First call from this thread: assign the next shard using
            // a relaxed atomic counter. The wrap-around at kActiveCountShards
            // ensures even distribution across shards; collisions (two
            // threads on the same shard) are rare and acceptable — they
            // only cause a small bounce on that one shard, not all 64.
            static std::atomic<std::size_t> s_next_shard{0};
            tls_shard = s_next_shard.fetch_add(1, std::memory_order_relaxed)
                        % kActiveCountShards;
        }
        return tls_shard;
    }

    static void sharded_active_count_inc() noexcept {
        const std::size_t idx = active_count_shard_index();
        s_active_count_shards_[idx].count.fetch_add(1, std::memory_order_relaxed);
    }

    static void sharded_active_count_dec() noexcept {
        const std::size_t idx = active_count_shard_index();
        s_active_count_shards_[idx].count.fetch_sub(1, std::memory_order_relaxed);
    }

    static std::size_t sum_sharded_active_count() noexcept {
        std::size_t total = 0;
        for (std::size_t i = 0; i < kActiveCountShards; ++i) {
            total += s_active_count_shards_[i].count.load(std::memory_order_relaxed);
        }
        return total;
    }

    // T4.2: Runtime opt-in flag for the per-T global counter in release
    // builds. In debug builds the counter is always maintained and this
    // flag is ignored.
    //
    // O10: Default is now `true` in release builds as well. The previous
    // default (`false`) made `force_wait_handles()` and the destructor's
    // `shutdown_and_wait()` unreliable unless the caller remembered to
    // opt in at process start — a footgun that silently degraded to
    // "no wait" in production. The cost is one relaxed atomic
    // fetch_add/fetch_sub per read_handle create/destroy (on top of the
    // per-cache counter, which is always maintained when per-cache
    // tracking is on). With P2-2's sharded layout this cost is now
    // cache-line-local (no ping-pong), so the flag can stay on by
    // default without perf concerns.
    static inline std::atomic<bool> s_global_tracking_enabled_{true};
};

} // namespace lru

#endif // LRU_CORE_HPP
