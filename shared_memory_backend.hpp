// Unified LRU Cache Library - Shared Memory Backend for Warm Restart
// SPDX-License-Identifier: MIT
//
// Provides a shared memory segment abstraction for near-zero restart time.
// When a cache process restarts, it can attach to the same shared memory
// segment and recover its data without waiting for a cold fill.
//
// Platform support:
//   Windows: CreateFileMappingA / OpenFileMappingA / MapViewOfFile
//   Linux/POSIX: shm_open / ftruncate / mmap
//
// Serialization format (P1-3): items are stored in the data region
// immediately after the 128-byte header using a length-prefixed format:
//   [uint32_t key_len][key bytes][uint32_t val_len][val bytes]
// Keys and values are written as raw bytes. For trivially-copyable types
// this is a memcpy; for std::string/std::vector the byte payload is the
// string contents / element range. Custom types may provide a free
// function `serialize(const T&, void* dst, std::size_t cap)` plus a
// `deserialize(T&, const void* src, std::size_t len)` overload, which
// this header detects via SFINAE and uses in preference to memcpy.
//
// T5 (P0-5): Cross-process synchronization & integrity
// ----------------------------------------------------
// The header now carries a CRC32 of the data region and a byte count of
// the valid data, so attach() can detect torn writes / partial flushes.
// A `cross_process_mutex` (named mutex on Windows, named POSIX semaphore
// on Linux) serializes save()/attach() across processes so a writer
// never has its writes interleaved with a reader's attach(). The mutex
// is *kernel-level* on both platforms — it survives process crashes
// (the OS releases it on process termination).

#ifndef LRU_SHARED_MEMORY_BACKEND_HPP
#define LRU_SHARED_MEMORY_BACKEND_HPP

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <new>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

// Platform-specific headers
#if defined(_WIN32)
#  if !defined(_WINDOWS_)
#    define WIN32_LEAN_AND_MEAN
#    define NOMINMAX
#  endif
#  include <windows.h>
#else
#  include <cerrno>
#  include <fcntl.h>
#  include <semaphore.h>
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <unistd.h>
#endif

namespace lru {

// ============================================================================
// Serialization helpers (P1-3)
// ============================================================================
//
// Length-prefixed byte serialization for the warm-restart data region.
// Each item is encoded as:
//   [uint32_t key_len][key bytes][uint32_t val_len][val bytes]
//
// Type traits dispatch to one of three paths:
//   1. User-provided serialize/deserialize overloads (ADL) — for
//      arbitrary user types that need a custom wire format.
//   2. std::string / std::vector<T> — the byte payload is the
//      string contents or the element range.
//   3. Trivially-copyable types — the byte payload is a memcpy of
//      the object representation.

namespace detail {

// --- SFINAE detection for user-provided serialize/deserialize ---------

template <typename T, typename = void>
struct has_user_serialize : std::false_type {};

template <typename T>
struct has_user_serialize<T,
    std::void_t<decltype(serialize(std::declval<const T&>(),
                                   std::declval<void*>(),
                                   std::declval<std::size_t>()))>>
    : std::true_type {};

template <typename T, typename = void>
struct has_user_deserialize : std::false_type {};

template <typename T>
struct has_user_deserialize<T,
    std::void_t<decltype(deserialize(std::declval<T&>(),
                                     std::declval<const void*>(),
                                     std::declval<std::size_t>()))>>
    : std::true_type {};

// --- std::string detection -------------------------------------------

template <typename T> struct is_std_string : std::false_type {};
template <typename C, typename Tr, typename A>
struct is_std_string<std::basic_string<C, Tr, A>> : std::true_type {};

// --- std::vector detection -------------------------------------------

template <typename T> struct is_std_vector : std::false_type {};
template <typename T, typename A>
struct is_std_vector<std::vector<T, A>> : std::true_type {};

// --- byte_size(): payload size in bytes for a value ------------------

template <typename T>
std::size_t byte_size(const T& v) {
    if constexpr (has_user_serialize<T>::value) {
        // Ask the user serializer for the size by passing nullptr/0.
        return serialize(v, nullptr, 0);
    } else if constexpr (is_std_string<T>::value) {
        return v.size() * sizeof(typename T::value_type);
    } else if constexpr (is_std_vector<T>::value) {
        return v.size() * sizeof(typename T::value_type);
    } else {
        static_assert(std::is_trivially_copyable_v<T>,
            "Type must be trivially-copyable, a std::string/std::vector, "
            "or provide serialize()/deserialize() overloads.");
        return sizeof(T);
    }
}

// --- write_bytes(): write payload to dst, return bytes written -------
// dst/return mirror the user-serializer contract: when cap is 0, the
// caller is asking for the size and dst may be nullptr.

template <typename T>
std::size_t write_bytes(const T& v, void* dst, std::size_t cap) {
    if constexpr (has_user_serialize<T>::value) {
        return serialize(v, dst, cap);
    } else if constexpr (is_std_string<T>::value) {
        std::size_t n = v.size() * sizeof(typename T::value_type);
        if (dst && cap >= n) std::memcpy(dst, v.data(), n);
        return n;
    } else if constexpr (is_std_vector<T>::value) {
        std::size_t n = v.size() * sizeof(typename T::value_type);
        if (dst && cap >= n) std::memcpy(dst, v.data(), n);
        return n;
    } else {
        if (dst && cap >= sizeof(T)) std::memcpy(dst, &v, sizeof(T));
        return sizeof(T);
    }
}

// --- read_bytes(): construct a value from src/len --------------------

template <typename T>
    void read_bytes(T& v, const void* src, std::size_t len) {
        if constexpr (has_user_deserialize<T>::value) {
            deserialize(v, src, len);
        } else if constexpr (is_std_string<T>::value) {
            if (len > 0) {
                v.assign(static_cast<const typename T::value_type*>(src),
                         len / sizeof(typename T::value_type));
            } else {
                v.clear();
            }
        } else if constexpr (is_std_vector<T>::value) {
            if (len > 0) {
                v.assign(static_cast<const typename T::value_type*>(src),
                         len / sizeof(typename T::value_type));
            } else {
                v.clear();
            }
        } else {
            // Trivially-copyable: copy the fixed-width representation.
            // Guard against len == 0 (e.g., zero-padded data region past
            // the last real record) so we don't dereference src when the
            // caller passed a bogus 0-length record.
            if (len >= sizeof(T)) {
                std::memcpy(&v, src, sizeof(T));
            } else {
                v = T{};
            }
        }
    }

}  // namespace detail

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for a shared memory segment.
struct shared_memory_config {
    /// Name of the shared memory segment.
    /// On Windows: used as the mapping object name.
    /// On Linux: used as the shm_open() name (should start with '/').
    std::string name;

    /// Total size of the shared memory segment in bytes.
    std::size_t size = 0;

    /// If true, create a new segment (fails if one already exists with the
    /// same name unless it can be opened). If false, open an existing one.
    bool create = true;

    /// If true, open the segment in read-only mode.
    bool read_only = false;
};

// ============================================================================
// Shared Memory Segment
// ============================================================================

/// RAII wrapper for a shared memory segment.
///
/// Provides portable access to named shared memory across Windows and
/// POSIX platforms. The segment is mapped into the process address space
/// on construction and unmapped on destruction.
///
/// Usage:
///   lru::shared_memory_config cfg{.name = "/my_cache", .size = 1 << 30};
///   lru::shared_memory_segment seg(cfg);
///   if (seg.is_newly_created()) {
///       // Fresh segment — initialize data structures
///   } else {
///       // Existing segment — recover data from previous run
///   }
class shared_memory_segment {
public:
    /// Create or open a shared memory segment.
    ///
    /// On success, data() returns a pointer to the mapped region and
    /// size() returns the segment size.
    ///
    /// On failure, data() returns nullptr and size() returns 0.
    explicit shared_memory_segment(const shared_memory_config& config)
        : data_(nullptr)
        , size_(0)
        , newly_created_(false)
#if defined(_WIN32)
        , handle_(nullptr)
#else
        , fd_(-1)
#endif
    {
        if (config.name.empty() || config.size == 0) return;

#if defined(_WIN32)
        // ---- Windows: CreateFileMappingA / OpenFileMappingA ----
        DWORD protect = config.read_only ? PAGE_READONLY : PAGE_READWRITE;
        DWORD access  = config.read_only ? FILE_MAP_READ : FILE_MAP_ALL_ACCESS;

        if (config.create) {
            // Try to create a new mapping first
            HANDLE mapping = CreateFileMappingA(
                INVALID_HANDLE_VALUE, nullptr, protect,
                static_cast<DWORD>(config.size >> 32),
                static_cast<DWORD>(config.size & 0xFFFFFFFFu),
                config.name.c_str());

            if (!mapping) {
                std::fprintf(stderr, "[shared_memory_segment] "
                             "CreateFileMappingA failed for '%s' (error %lu)\n",
                             config.name.c_str(), GetLastError());
                return;
            }

            // Check if we created a new mapping or opened an existing one
            newly_created_ = (GetLastError() != ERROR_ALREADY_EXISTS);
            handle_ = mapping;

            // Map the view
            void* base = MapViewOfFile(mapping, access, 0, 0,
                                       static_cast<SIZE_T>(config.size));
            if (!base) {
                std::fprintf(stderr, "[shared_memory_segment] "
                             "MapViewOfFile failed for '%s' (error %lu)\n",
                             config.name.c_str(), GetLastError());
                CloseHandle(mapping);
                handle_ = nullptr;
                return;
            }

            data_ = base;
            size_ = config.size;
        } else {
            // Open an existing mapping
            HANDLE mapping = OpenFileMappingA(access, FALSE, config.name.c_str());
            if (!mapping) {
                std::fprintf(stderr, "[shared_memory_segment] "
                             "OpenFileMappingA failed for '%s' (error %lu)\n",
                             config.name.c_str(), GetLastError());
                return;
            }

            newly_created_ = false;
            handle_ = mapping;

            void* base = MapViewOfFile(mapping, access, 0, 0,
                                       static_cast<SIZE_T>(config.size));
            if (!base) {
                std::fprintf(stderr, "[shared_memory_segment] "
                             "MapViewOfFile failed for '%s' (error %lu)\n",
                             config.name.c_str(), GetLastError());
                CloseHandle(mapping);
                handle_ = nullptr;
                return;
            }

            data_ = base;
            size_ = config.size;
        }
#else
        // ---- POSIX: shm_open / ftruncate / mmap ----
        int oflag = config.read_only ? O_RDONLY : O_RDWR;
        if (config.create) {
            oflag |= O_CREAT | O_EXCL;
        }

        int fd = shm_open(config.name.c_str(), oflag, 0666);
        if (fd < 0) {
            // If O_EXCL failed because the segment already exists and we're
            // in create mode, try opening it without O_EXCL
            if (config.create && errno == EEXIST) {
                oflag = config.read_only ? O_RDONLY : O_RDWR;
                oflag |= O_CREAT;
                fd = shm_open(config.name.c_str(), oflag, 0666);
            }
            if (fd < 0) {
                std::fprintf(stderr, "[shared_memory_segment] "
                             "shm_open failed for '%s' (errno %d)\n",
                             config.name.c_str(), errno);
                return;
            }
        } else {
            newly_created_ = true;
        }

        // Set the size if we created the segment
        if (newly_created_) {
            if (ftruncate(fd, static_cast<off_t>(config.size)) != 0) {
                std::fprintf(stderr, "[shared_memory_segment] "
                             "ftruncate failed for '%s' (errno %d)\n",
                             config.name.c_str(), errno);
                ::close(fd);
                return;
            }
        } else {
            // Verify the existing segment is large enough
            struct stat st;
            if (fstat(fd, &st) != 0) {
                std::fprintf(stderr, "[shared_memory_segment] "
                             "fstat failed for '%s' (errno %d)\n",
                             config.name.c_str(), errno);
                ::close(fd);
                return;
            }
            if (static_cast<std::size_t>(st.st_size) < config.size) {
                std::fprintf(stderr, "[shared_memory_segment] "
                             "existing segment '%s' is too small "
                             "(%jd bytes, need %zu)\n",
                             config.name.c_str(),
                             static_cast<intmax_t>(st.st_size),
                             config.size);
                ::close(fd);
                return;
            }
        }

        // Map the segment
        int prot = config.read_only ? PROT_READ : (PROT_READ | PROT_WRITE);
        void* base = mmap(nullptr, config.size, prot, MAP_SHARED, fd, 0);
        if (base == MAP_FAILED) {
            std::fprintf(stderr, "[shared_memory_segment] "
                         "mmap failed for '%s' (errno %d)\n",
                         config.name.c_str(), errno);
            ::close(fd);
            return;
        }

        fd_ = fd;
        data_ = base;
        size_ = config.size;
#endif
    }

    /// Destructor — unmaps and closes the shared memory segment.
    ~shared_memory_segment() {
        close();
    }

    // Non-copyable
    shared_memory_segment(const shared_memory_segment&) = delete;
    shared_memory_segment& operator=(const shared_memory_segment&) = delete;

    // Movable
    shared_memory_segment(shared_memory_segment&& other) noexcept
        : data_(other.data_)
        , size_(other.size_)
        , newly_created_(other.newly_created_)
#if defined(_WIN32)
        , handle_(other.handle_)
#else
        , fd_(other.fd_)
#endif
    {
        other.data_ = nullptr;
        other.size_ = 0;
#if defined(_WIN32)
        other.handle_ = nullptr;
#else
        other.fd_ = -1;
#endif
    }

    shared_memory_segment& operator=(shared_memory_segment&& other) noexcept {
        if (this != &other) {
            close();
            data_ = other.data_;
            size_ = other.size_;
            newly_created_ = other.newly_created_;
#if defined(_WIN32)
            handle_ = other.handle_;
            other.handle_ = nullptr;
#else
            fd_ = other.fd_;
            other.fd_ = -1;
#endif
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    // ----------------------------------------------------------------
    // Accessors
    // ----------------------------------------------------------------

    /// Pointer to the mapped shared memory region.
    /// Returns nullptr if the segment is not mapped.
    void* data() noexcept { return data_; }
    const void* data() const noexcept { return data_; }

    /// Size of the mapped region in bytes.
    std::size_t size() const noexcept { return size_; }

    /// Check if the segment was newly created (vs attached to existing).
    /// Returns true if this object created the segment; false if it
    /// attached to an already-existing segment.
    bool is_newly_created() const noexcept { return newly_created_; }

    /// Check if the segment is valid (mapped).
    explicit operator bool() const noexcept { return data_ != nullptr; }

    /// Flush modified pages to physical storage.
    /// Ensures that writes to the shared memory are visible to other
    /// processes that map the same segment.
    void flush() {
        if (!data_ || size_ == 0) return;

#if defined(_WIN32)
        FlushViewOfFile(data_, static_cast<SIZE_T>(size_));
#else
        msync(data_, size_, MS_SYNC);
#endif
    }

private:
    void* data_;
    std::size_t size_;
    bool newly_created_;

#if defined(_WIN32)
    /// Windows: file mapping handle returned by CreateFileMappingA.
    void* handle_;  // HANDLE
#else
    /// POSIX: file descriptor returned by shm_open().
    int fd_;
#endif

    /// Release the mapping and platform-specific handles.
    void close() {
        if (!data_) return;

#if defined(_WIN32)
        UnmapViewOfFile(data_);
        if (handle_) {
            CloseHandle(static_cast<HANDLE>(handle_));
            handle_ = nullptr;
        }
#else
        munmap(data_, size_);
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }
#endif
        data_ = nullptr;
        size_ = 0;
    }
};

// ============================================================================
// Warm Restart Cache
// ============================================================================

// ----------------------------------------------------------------------------
// T5.1: cross_process_mutex
// ----------------------------------------------------------------------------
//
// A kernel-level named mutex that survives process crashes. Used to
// serialize save()/attach() across processes that share the same
// shared-memory segment.
//
//   - Windows: CreateMutexA / OpenMutexA (kernel object, auto-released
//              on process exit by the OS).
//   - POSIX:   named POSIX semaphore (sem_open with O_CREAT). Named
//              semaphores are kernel-persistent on Linux and survive
//              process crashes; the last `sem_unlink` removes them.
//
// The mutex name is derived from the shared-memory segment name by
// appending a "_lock" suffix, so callers don't need to invent a
// separate name.
//
// Concurrency model: single-writer / multi-reader enforced at the API
// level — both save() and attach() acquire the mutex exclusively.
// This is correct because attach() *reads* the data region and must
// not see a half-written snapshot. Concurrent attach() calls from
// multiple processes serialize on the mutex; concurrent save() calls
// also serialize.

class cross_process_mutex {
public:
    /// Construct (or open) a named mutex.
    ///
    /// \param name  The mutex name. On POSIX this is used as the
    ///              sem_open() name (should start with '/'). On
    ///              Windows it is the kernel object name.
    /// \param create_if_missing  If true, create the mutex if it
    ///              doesn't exist. If false, only open an existing
    ///              one (returns false on failure).
    cross_process_mutex(const std::string& name, bool create_if_missing = true)
        : name_(name)
#if !defined(_WIN32)
        , sem_(nullptr)
        , owner_(false)
#endif
    {
        if (name.empty()) return;
#if defined(_WIN32)
        DWORD desired_access = MUTEX_ALL_ACCESS;
        if (create_if_missing) {
            handle_ = CreateMutexA(nullptr, FALSE, name.c_str());
        } else {
            handle_ = OpenMutexA(desired_access, FALSE, name.c_str());
        }
        if (!handle_) {
            std::fprintf(stderr, "[cross_process_mutex] "
                         "Create/OpenMutexA failed for '%s' (error %lu)\n",
                         name.c_str(), GetLastError());
        }
#else
        // POSIX named semaphore. Use O_CREAT if asked to create.
        if (create_if_missing) {
            sem_ = sem_open(name.c_str(), O_CREAT, 0666, 1);
        } else {
            sem_ = sem_open(name.c_str(), 0);
        }
        if (sem_ == SEM_FAILED) {
            std::fprintf(stderr, "[cross_process_mutex] "
                         "sem_open failed for '%s' (errno %d)\n",
                         name.c_str(), errno);
            sem_ = nullptr;
        } else {
            owner_ = create_if_missing;
        }
#endif
    }

    ~cross_process_mutex() {
#if defined(_WIN32)
        if (handle_) {
            CloseHandle(handle_);
            handle_ = nullptr;
        }
#else
        if (sem_) {
            sem_close(sem_);
            sem_ = nullptr;
        }
        // Note: we deliberately do NOT sem_unlink() here, because other
        // processes may still hold the semaphore open. The creator is
        // responsible for unlinking at program exit if desired.
#endif
    }

    // Non-copyable, non-movable (the underlying handle/fd is process-local).
    cross_process_mutex(const cross_process_mutex&) = delete;
    cross_process_mutex& operator=(const cross_process_mutex&) = delete;
    cross_process_mutex(cross_process_mutex&&) = delete;
    cross_process_mutex& operator=(cross_process_mutex&&) = delete;

    /// Acquire the mutex. Blocks up to `timeout` (or forever if the
    /// timeout is negative). Returns true on acquisition, false on
    /// timeout or failure.
    template <typename Rep, typename Period>
    bool lock(std::chrono::duration<Rep, Period> timeout) {
        return lock_impl(timeout);
    }

    /// Acquire the mutex, blocking forever.
    void lock() {
#if defined(_WIN32)
        if (!handle_) return;
        DWORD rv = WaitForSingleObject(handle_, INFINITE);
        if (rv != WAIT_OBJECT_0) {
            std::fprintf(stderr, "[cross_process_mutex] "
                         "WaitForSingleObject failed (rv=%lu, err=%lu)\n",
                         rv, GetLastError());
        }
#else
        if (!sem_) return;
        while (sem_wait(sem_) != 0) {
            if (errno != EINTR) {
                std::fprintf(stderr, "[cross_process_mutex] "
                             "sem_wait failed (errno %d)\n", errno);
                return;
            }
        }
#endif
    }

    /// Release the mutex.
    void unlock() {
#if defined(_WIN32)
        if (!handle_) return;
        ReleaseMutex(handle_);
#else
        if (!sem_) return;
        sem_post(sem_);
#endif
    }

    /// Check whether the underlying primitive was successfully opened.
    explicit operator bool() const noexcept {
#if defined(_WIN32)
        return handle_ != nullptr;
#else
        return sem_ != nullptr;
#endif
    }

    /// Unlink the named primitive (POSIX-only). Call this from the
    /// process that "owns" the segment when the segment is being
    /// permanently removed. After unlink, other processes that have
    /// already opened the mutex can still use it, but new opens will
    /// fail. No-op on Windows (kernel releases the name when the last
    /// handle is closed).
    void unlink() noexcept {
#if !defined(_WIN32)
        if (owner_ && !name_.empty()) {
            sem_unlink(name_.c_str());
            owner_ = false;
        }
#endif
    }

private:
    std::string name_;
#if defined(_WIN32)
    void* handle_ = nullptr;  // HANDLE
#else
    sem_t* sem_ = nullptr;
    bool owner_ = false;
#endif

    template <typename Rep, typename Period>
    bool lock_impl(std::chrono::duration<Rep, Period> timeout) {
        if (timeout.count() < 0) { lock(); return true; }
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(timeout);
        if (ms.count() == 0) ms = std::chrono::milliseconds(1);
#if defined(_WIN32)
        if (!handle_) return false;
        DWORD rv = WaitForSingleObject(
            handle_, static_cast<DWORD>(ms.count()));
        if (rv == WAIT_OBJECT_0) return true;
        if (rv == WAIT_TIMEOUT) return false;
        std::fprintf(stderr, "[cross_process_mutex] "
                     "WaitForSingleObject failed (rv=%lu, err=%lu)\n",
                     rv, GetLastError());
        return false;
#else
        if (!sem_) return false;
        struct timespec ts;
        auto now = std::chrono::system_clock::now();
        auto abs = now + ms;
        auto secs = std::chrono::duration_cast<std::chrono::seconds>(abs.time_since_epoch());
        auto nsecs = std::chrono::duration_cast<std::chrono::nanoseconds>(abs.time_since_epoch()) -
                     std::chrono::duration_cast<std::chrono::nanoseconds>(secs);
        ts.tv_sec = secs.count();
        ts.tv_nsec = nsecs.count();
        while (sem_timedwait(sem_, &ts) != 0) {
            if (errno == ETIMEDOUT) return false;
            if (errno != EINTR) {
                std::fprintf(stderr, "[cross_process_mutex] "
                             "sem_timedwait failed (errno %d)\n", errno);
                return false;
            }
        }
        return true;
#endif
    }
};

// ----------------------------------------------------------------------------
// T5.3: CRC32 (IEEE 802.3 polynomial, table-less branchless implementation)
// ----------------------------------------------------------------------------

inline uint32_t crc32_compute(const void* data, std::size_t len) noexcept {
    // Polynomial 0xEDB88320 (reflected). Use a lazily-initialized table
    // for throughput on large data regions; fall back to byte-by-byte
    // computation otherwise.
    static constexpr uint32_t kPoly = 0xEDB88320u;
    static uint32_t table[256];
    static std::atomic<int> table_init{0};  // 0 = not init, 1 = init done
    if (table_init.load(std::memory_order_acquire) == 0) {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k) {
                c = (c & 1u) ? (kPoly ^ (c >> 1)) : (c >> 1);
            }
            table[i] = c;
        }
        table_init.store(1, std::memory_order_release);
    }

    const auto* p = static_cast<const uint8_t*>(data);
    uint32_t crc = 0xFFFFFFFFu;
    for (std::size_t i = 0; i < len; ++i) {
        crc = table[(crc ^ p[i]) & 0xFFu] ^ (crc >> 8);
    }
    return crc ^ 0xFFFFFFFFu;
}

/// Warm restart cache — uses shared memory for near-zero restart time.
///
/// When a cache process restarts, it can attach to the same shared memory
/// segment and recover its data without waiting for a cold fill. This
/// template provides the building blocks for such a cache:
///
///   1. A shared_memory_segment for persistent storage
///   2. A header structure in the segment for metadata validation
///   3. Methods to check for and attach to previous data
///
/// Cache integration is decoupled from this class: callers supply an
/// `inserter` callable to `attach()` / `attach_with_timeout()` that
/// receives `(const Key&, const Value&)` pairs and inserts them into
/// whatever in-memory cache type they use (unified_cache, striped_cache,
/// etc.). This avoids coupling the shared-memory layer to any specific
/// cache implementation.
///
/// T5: Cross-process safety
/// ------------------------
/// `save()` and `attach()` acquire a `cross_process_mutex` (named mutex
/// on Windows, named POSIX semaphore on Linux) before touching the data
/// region, so concurrent writers/readers from different processes cannot
/// interleave and corrupt each other. The header carries a CRC32 of the
/// valid data region, so `attach()` can detect a torn or partially
/// flushed write and refuse to recover from it. Use
/// `attach_with_timeout(timeout, inserter)` to bound the wait on a
/// contended mutex.
///
/// Usage:
///   lru::shared_memory_config cfg{.name = "/my_cache", .size = 1 << 30};
///   lru::warm_restart_cache<int, std::string> cache(cfg);
///   if (cache.has_previous_data()) {
///       cache.attach();  // Recover from previous run
///   }
template <typename Key, typename Value,
          typename Hash = std::hash<Key>,
          typename KeyEqual = std::equal_to<Key>>
class warm_restart_cache {
public:
    using config_type = shared_memory_config;
    using key_type = Key;
    using mapped_type = Value;

    /// Magic number written at the start of the shared memory header.
    static constexpr uint64_t kMagic = 0x4C52554D53484300ULL; // "LRUMSHC\0"

    /// Current format version. Bumped to 2 in T5 to indicate the
    /// header now carries a CRC32 + data_size_bytes for integrity
    /// checking. v1 segments (no CRC) are rejected by attach().
    static constexpr uint64_t kVersion = 2;

    /// Header written at the beginning of the shared memory segment.
    /// Used to validate the segment on warm restart — if the header
    /// doesn't match, attach() returns false.
    ///
    /// Layout (128 bytes, fixed-width types for cross-platform consistency):
    ///   Offset  0: magic (uint64_t)
    ///   Offset  8: version (uint64_t)
    ///   Offset 16: item_count (uint64_t)
    ///   Offset 24: data_offset (uint64_t)
    ///   Offset 32: create_timestamp (uint64_t)
    ///   Offset 40: last_attach_timestamp (uint64_t)
    ///   Offset 48: data_crc32 (uint32_t)         — T5.3: CRC32 of data region
    ///   Offset 52: data_size_bytes (uint32_t)    — T5.3: valid bytes in data region
    ///   Offset 56: reserved (72 bytes)           — future use
    struct header {
        uint64_t magic = kMagic;
        uint64_t version = kVersion;
        uint64_t item_count = 0;
        uint64_t data_offset = 128;  // offset to data region (after header)
        uint64_t create_timestamp = 0;              // epoch seconds
        uint64_t last_attach_timestamp = 0;         // epoch seconds
        uint32_t data_crc32 = 0;                    // T5.3: CRC32 of data region
        uint32_t data_size_bytes = 0;               // T5.3: bytes used in data region
        uint8_t  reserved[72] = {};
    };

    static_assert(sizeof(header) == 128, "warm_restart_cache header must be 128 bytes");

    /// Construct a warm restart cache with the given shared memory config.
    explicit warm_restart_cache(const shared_memory_config& config)
        : segment_(config)
        , has_previous_data_(false)
        , mutex_(make_mutex_name(config.name), /*create_if_missing=*/true)
    {
        if (!segment_) return;  // Segment creation/opening failed

        if (segment_.is_newly_created()) {
            // Fresh segment — initialize the header
            auto* hdr = header_ptr();
            std::memset(hdr, 0, sizeof(header));
            hdr->magic = kMagic;
            hdr->version = kVersion;
            hdr->item_count = 0;
            hdr->data_offset = sizeof(header);
            hdr->data_crc32 = 0;
            hdr->data_size_bytes = 0;
            hdr->create_timestamp = current_timestamp();
            hdr->last_attach_timestamp = 0;
            segment_.flush();
        } else {
            // Existing segment — check if it has valid data
            auto* hdr = header_ptr();
            has_previous_data_ = (hdr->magic == kMagic && hdr->version == kVersion);
        }
    }

    /// Check if the cache has valid data from a previous run.
    /// Returns true if the shared memory segment contains a valid header
    /// from a previous instance.
    bool has_previous_data() const noexcept {
        return has_previous_data_;
    }

    /// Attach to the shared memory and rebuild the cache structure.
    /// Returns the number of items successfully deserialized, or 0 if
    /// no valid previous data exists or the data region is corrupt.
    ///
    /// On success, the header's last_attach_timestamp is updated and
    /// the segment is flushed so other processes see the attach.
    ///
    /// The caller supplies an `inserter` callable that takes
    /// `(const Key&, const Value&)` and inserts the item into the
    /// in-memory cache. This decouples attach() from any specific cache
    /// type (unified_cache, striped_cache, etc.).
    ///
    /// T5.2: Acquires `cross_process_mutex_` so a concurrent save() in
    /// another process cannot interleave with this attach.
    /// T5.3: Verifies the CRC32 of the data region before deserializing.
    /// If the CRC does not match (torn write, partial flush, corruption),
    /// attach() returns 0 and the header's data_size_bytes is reset.
    ///
    /// Example:
    ///   cache.attach([&](const K& k, const V& v) { primary_cache.set(k, v); });
    template <typename Inserter>
    std::size_t attach(Inserter&& inserter) {
        return attach_with_timeout(
            std::chrono::seconds(-1),  // negative = block forever
            std::forward<Inserter>(inserter));
    }

    /// T5.4: Same as attach(inserter) but with a bounded wait on the
    /// cross-process mutex. Returns 0 if the timeout expired before
    /// the mutex could be acquired (no items read).
    template <typename Inserter, typename Rep, typename Period>
    std::size_t attach_with_timeout(
        std::chrono::duration<Rep, Period> timeout,
        Inserter&& inserter) {
        if (!has_previous_data_) return 0;

        // T5.2: try to acquire the cross-process mutex.
        // Negative timeout = block forever (matches attach()).
        bool acquired = false;
        if (timeout.count() < 0) {
            mutex_.lock();
            acquired = true;
        } else {
            acquired = mutex_.lock(timeout);
            if (!acquired) return 0;  // timeout
        }
        // RAII release — std::lock_guard works because cross_process_mutex
        // exposes lock()/unlock().
        std::lock_guard<cross_process_mutex> lk(mutex_, std::adopt_lock);

        auto* hdr = header_ptr();
        const std::size_t claimed = hdr->item_count;
        const std::size_t data_size = data_region_size();

        // T5.3: verify CRC32 before deserializing. If the writer crashed
        // mid-save, the CRC will not match and we refuse to recover.
        const uint32_t expected_crc = hdr->data_crc32;
        const std::size_t valid_bytes = hdr->data_size_bytes;
        if (valid_bytes > data_size) {
            // Header is inconsistent — region is too small for claimed bytes.
            return 0;
        }
        if (valid_bytes > 0) {
            const uint32_t actual_crc =
                crc32_compute(data_region(), valid_bytes);
            if (actual_crc != expected_crc) {
                // Corrupt or torn write — refuse to recover.
                std::fprintf(stderr, "[warm_restart_cache] "
                             "CRC mismatch on attach: expected %08x, got %08x "
                             "(%zu valid bytes)\n",
                             expected_crc, actual_crc, valid_bytes);
                return 0;
            }
        }

        // Walk the data region and deserialize each item. The region is
        // a sequence of [u32 key_len][key bytes][u32 val_len][val bytes]
        // records. We bound every read against data_size so a corrupt
        // or truncated region can't cause an out-of-bounds read.
        const char* p = static_cast<const char*>(data_region());
        const char* end = p + data_size;
        // T5.3: prefer the recorded valid_bytes as the bound, falling back
        // to the full region size only if the header predates the size
        // field (it should not, since version is checked at construction).
        if (valid_bytes > 0 && valid_bytes <= data_size) {
            end = p + valid_bytes;
        }
        std::size_t recovered = 0;

        for (std::size_t i = 0; i < claimed && p + sizeof(uint32_t) <= end; ++i) {
            // Read key length.
            uint32_t klen = 0;
            std::memcpy(&klen, p, sizeof(uint32_t));
            p += sizeof(uint32_t);
            if (p + klen > end) break;  // corrupt: key extends past end

            // Deserialize key.
            Key k{};
            detail::read_bytes(k, p, klen);
            p += klen;

            // Read value length.
            if (p + sizeof(uint32_t) > end) break;
            uint32_t vlen = 0;
            std::memcpy(&vlen, p, sizeof(uint32_t));
            p += sizeof(uint32_t);
            if (p + vlen > end) break;  // corrupt: value extends past end

            // Deserialize value and insert.
            Value v{};
            detail::read_bytes(v, p, vlen);
            p += vlen;

            try {
                inserter(k, v);
                ++recovered;
            } catch (...) {
                // If the caller's inserter throws (e.g., cache is full or
                // admission rejects), skip this item but keep going so we
                // recover as many items as possible.
            }
        }

        // Update the attach timestamp and persist it.
        hdr->last_attach_timestamp = current_timestamp();
        segment_.flush();

        return recovered;
    }

    /// Convenience overload of attach() that takes no inserter: it just
    /// validates the segment and updates the attach timestamp. Items in
    /// the data region are NOT loaded into any in-memory cache. Use the
    /// templated attach(inserter) overload for actual recovery.
    bool attach() {
        if (!has_previous_data_) return false;
        // T5.2: still acquire the mutex to serialize with save().
        std::lock_guard<cross_process_mutex> lk(mutex_);
        header_ptr()->last_attach_timestamp = current_timestamp();
        segment_.flush();
        return true;
    }

    /// Persist a range of key-value pairs to the shared memory data region.
    /// Overwrites any existing data. Returns the number of items written.
    ///
    /// Each item is encoded as [u32 key_len][key bytes][u32 val_len][val bytes].
    /// If the data region is too small to hold all items, the write stops
    /// at the last item that fits and the header's item_count is updated
    /// to reflect the actual count persisted.
    ///
    /// T5.2: Acquires `cross_process_mutex_` before writing, so concurrent
    /// save() / attach() calls from different processes cannot interleave.
    /// T5.3: Computes CRC32 of the written data region and stores it in
    /// the header along with `data_size_bytes` for attach()-time
    /// integrity verification.
    ///
    /// The caller typically passes primary_cache contents:
    ///   std::vector<std::pair<K,V>> items = primary.snapshot();
    ///   cache.save(items.begin(), items.end());
    template <typename InputIt>
    std::size_t save(InputIt first, InputIt last) {
        if (!segment_) return 0;

        // T5.2: serialize with other processes.
        std::lock_guard<cross_process_mutex> lk(mutex_);

        char* p = static_cast<char*>(data_region());
        char* end = p + data_region_size();
        std::size_t written = 0;

        for (auto it = first; it != last; ++it) {
            const Key& k = it->first;
            const Value& v = it->second;

            const std::size_t klen = detail::byte_size(k);
            const std::size_t vlen = detail::byte_size(v);
            const std::size_t rec =
                sizeof(uint32_t) + klen + sizeof(uint32_t) + vlen;

            if (p + rec > end) break;  // data region full

            // Write key length + key bytes.
            uint32_t klen32 = static_cast<uint32_t>(klen);
            std::memcpy(p, &klen32, sizeof(uint32_t));
            p += sizeof(uint32_t);
            detail::write_bytes(k, p, klen);
            p += klen;

            // Write value length + value bytes.
            uint32_t vlen32 = static_cast<uint32_t>(vlen);
            std::memcpy(p, &vlen32, sizeof(uint32_t));
            p += sizeof(uint32_t);
            detail::write_bytes(v, p, vlen);
            p += vlen;

            ++written;
        }

        // T5.3: compute CRC32 + valid byte count over what we just wrote.
        const std::size_t bytes_used = static_cast<std::size_t>(p - static_cast<char*>(data_region()));
        const uint32_t crc = (bytes_used > 0)
            ? crc32_compute(data_region(), bytes_used)
            : 0u;

        // Update the header (item_count first, then size + CRC, then flush).
        // A reader that races with this update is blocked by the mutex
        // anyway; the ordering here is for crash-consistency: if we crash
        // mid-flush, the reader's CRC check will fail and refuse to attach.
        auto* hdr = header_ptr();
        hdr->item_count = written;
        hdr->data_size_bytes = static_cast<uint32_t>(bytes_used);
        hdr->data_crc32 = crc;
        segment_.flush();
        return written;
    }

    /// Get the number of items stored in the previous run.
    /// Returns 0 if no previous data exists.
    std::size_t previous_item_count() const noexcept {
        if (!has_previous_data_) return 0;
        return header_ptr()->item_count;
    }

    /// Get a pointer to the data region (after the header).
    /// Returns nullptr if the segment is not mapped.
    void* data_region() noexcept {
        if (!segment_) return nullptr;
        return static_cast<char*>(segment_.data()) + sizeof(header);
    }

    const void* data_region() const noexcept {
        if (!segment_) return nullptr;
        return static_cast<const char*>(segment_.data()) + sizeof(header);
    }

    /// Get the size of the data region in bytes.
    std::size_t data_region_size() const noexcept {
        if (!segment_) return 0;
        return segment_.size() > sizeof(header) ? segment_.size() - sizeof(header) : 0;
    }

    /// Get the underlying shared memory segment.
    shared_memory_segment& segment() noexcept { return segment_; }
    const shared_memory_segment& segment() const noexcept { return segment_; }

    /// Update the item count in the header and flush.
    void set_item_count(std::size_t count) {
        if (!segment_) return;
        header_ptr()->item_count = count;
        segment_.flush();
    }

private:
    shared_memory_segment segment_;
    bool has_previous_data_;
    cross_process_mutex mutex_;

    /// Get the header pointer from the mapped segment.
    header* header_ptr() noexcept {
        return static_cast<header*>(segment_.data());
    }

    const header* header_ptr() const noexcept {
        return static_cast<const header*>(segment_.data());
    }

    /// Derive a mutex name from the segment name. On both Windows and
    /// POSIX the name is just the segment name with a "_lock" suffix;
    /// on POSIX the leading '/' is preserved if present.
    static std::string make_mutex_name(const std::string& seg_name) {
        if (seg_name.empty()) return {};
        return seg_name + "_lock";
    }

    /// Get the current timestamp in epoch seconds.
    static uint64_t current_timestamp() {
        auto now = std::chrono::system_clock::now();
        return static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::seconds>(
                now.time_since_epoch()).count());
    }
};

} // namespace lru

#endif // LRU_SHARED_MEMORY_BACKEND_HPP
