// SPDX-License-Identifier: MIT
// Native wait/wake wrappers (runtime-selected).
//
// Platform strategies:
//   - Windows 8+: WaitOnAddress / WakeByAddressSingle / WakeByAddressAll
//   - Linux: futex FUTEX_WAIT_PRIVATE / FUTEX_WAKE_PRIVATE
//   - macOS: ulock_wait (UL_COMPARE_AND_WAIT) / ulock_wake
//   - Fallback: no-op (callers provide a CV fallback, which emits a
//                one-time stderr warning via warn_fallback_once()).
//
// This header is shared by distributed_mutex.hpp and concurrent_hash_table.hpp
// to avoid code duplication of the OS wait/wake primitives.

#ifndef LRU_DETAIL_NATIVE_WAIT_OPS_HPP
#define LRU_DETAIL_NATIVE_WAIT_OPS_HPP

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <mutex>

// ============================================================================
// Platform-specific includes — must appear OUTSIDE any namespace
// ============================================================================

#if defined(__linux__)
    #include <climits>
    #include <sys/syscall.h>
    #include <unistd.h>
    #ifndef FUTEX_WAIT_PRIVATE
        #define FUTEX_WAIT_PRIVATE 128
    #endif
    #ifndef FUTEX_WAKE_PRIVATE
        #define FUTEX_WAKE_PRIVATE 129
    #endif
    #define LRU_NATIVE_HAS_FUTEX 1
#elif defined(__APPLE__) && defined(__MACH__)
    // macOS ulock API — private kernel interface available on macOS 10.12+.
    // Not declared in any public header; declare manually with weak_import
    // so older SDKs / runtime versions degrade gracefully to the CV fallback.
    #include <mach/vm_param.h>
    #include <sys/sysctl.h>

    // ulock operation codes (from xnu osfmk/kern/syscall_ulock.c).
    #define LRU_UL_COMPARE_AND_WAIT        1
    #define LRU_UL_COMPARE_AND_WAIT_SHARED 3
    #define LRU_ULF_WAKE_ALL               0x00000100
    #define LRU_ULF_NO_ERRNO               0x01000000

    extern "C" {
        // Returns >0 on woken, 0 on timeout, -1 on error (errno set).
        // `value` is the expected 32-bit value at `addr`; the wait blocks
        // while *(uint32_t*)addr == value.
        int __ulock_wait(uint32_t operation, void* addr, uint64_t value,
                         uint32_t timeout_us) __attribute__((weak_import));
        // Returns 0 on success, -1 on error. `wake_value` is unused for
        // UL_COMPARE_AND_WAIT but required by the signature.
        int __ulock_wake(uint32_t operation, void* addr, uint64_t wake_value)
                         __attribute__((weak_import));
    }
    #define LRU_NATIVE_HAS_ULOCK 1
#endif

#if defined(_WIN32)
    // Runtime detection of WaitOnAddress (may not exist on older Windows / MinGW).
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #include <windows.h>
    #define LRU_NATIVE_HAS_WIN32 1
#endif

namespace lru::detail {

// ============================================================================
// Native wait/wake wrappers (runtime-selected)
// ============================================================================

class native_wait_ops {
public:
    /// Returns true if native wait/wake is available on this system.
    static bool available() noexcept {
        static const bool ok = probe();
        return ok;
    }

    /// Wait on addr if its value still equals expected.
    static void wait(const std::atomic<uint32_t>& addr, uint32_t expected) {
#if LRU_NATIVE_HAS_WIN32
        if (s_WaitOnAddress) {
            s_WaitOnAddress(
                const_cast<volatile void*>(
                    static_cast<const volatile void*>(&addr)),
                const_cast<uint32_t*>(&expected),
                sizeof(uint32_t), 0xFFFFFFFF /* INFINITE */);
            return;
        }
#endif
#if LRU_NATIVE_HAS_FUTEX
        syscall(__NR_futex,
                reinterpret_cast<const uint32_t*>(std::addressof(addr)),
                FUTEX_WAIT_PRIVATE,
                expected, nullptr, nullptr, 0);
        return;
#elif LRU_NATIVE_HAS_ULOCK
        if (__ulock_wait) {
            // Block indefinitely (timeout=0 means infinite for ulock_wait).
            __ulock_wait(LRU_UL_COMPARE_AND_WAIT | LRU_ULF_NO_ERRNO,
                         const_cast<void*>(static_cast<const void*>(&addr)),
                         static_cast<uint64_t>(expected),
                         0);
            return;
        }
#else
        (void)addr; (void)expected;
#endif
    }

    /// Wake one waiter on addr.
    static void wake_one(std::atomic<uint32_t>& addr) {
#if LRU_NATIVE_HAS_WIN32
        if (s_WakeByAddressSingle) {
            s_WakeByAddressSingle(static_cast<void*>(&addr));
            return;
        }
#endif
#if LRU_NATIVE_HAS_FUTEX
        syscall(__NR_futex,
                reinterpret_cast<uint32_t*>(std::addressof(addr)),
                FUTEX_WAKE_PRIVATE, 1, nullptr, nullptr, 0);
        return;
#elif LRU_NATIVE_HAS_ULOCK
        if (__ulock_wake) {
            __ulock_wake(LRU_UL_COMPARE_AND_WAIT | LRU_ULF_NO_ERRNO,
                         static_cast<void*>(&addr), 0);
            return;
        }
#else
        (void)addr;
#endif
    }

    /// Wake all waiters on addr.
    static void wake_all(std::atomic<uint32_t>& addr) {
#if LRU_NATIVE_HAS_WIN32
        if (s_WakeByAddressAll) {
            s_WakeByAddressAll(static_cast<void*>(&addr));
            return;
        }
#endif
#if LRU_NATIVE_HAS_FUTEX
        syscall(__NR_futex,
                reinterpret_cast<uint32_t*>(std::addressof(addr)),
                FUTEX_WAKE_PRIVATE, INT_MAX, nullptr, nullptr, 0);
        return;
#elif LRU_NATIVE_HAS_ULOCK
        if (__ulock_wake) {
            __ulock_wake(LRU_UL_COMPARE_AND_WAIT | LRU_ULF_WAKE_ALL |
                             LRU_ULF_NO_ERRNO,
                         static_cast<void*>(&addr), 0);
            return;
        }
#else
        (void)addr;
#endif
    }

    /// Emit a one-time warning to stderr when the CV fallback path is used.
    /// Callers should invoke this once before relying on the CV fallback
    /// so users notice they are running without native wait/wake support
    /// (which means the distributed_shared_mutex / shared_spinlock will
    /// fall back to std::condition_variable — higher latency under
    /// contention).
    static void warn_fallback_once() {
        std::call_once(s_warn_flag_, []() {
            std::fprintf(stderr,
                "[lru] WARNING: native wait/wake unavailable — "
                "distributed_shared_mutex and shared_spinlock are using "
                "the std::condition_variable fallback. This increases "
                "wake latency under contention. Consider running on "
                "Linux (futex), Windows 8+ (WaitOnAddress), or macOS "
                "10.12+ (ulock).\n");
        });
    }

private:
    static bool probe() {
#if LRU_NATIVE_HAS_WIN32
        // On Windows, dynamically resolve WaitOnAddress family.
        // These live in kernel32.dll on Windows 8/8.1, but moved to
        // KERNELBASE.dll (via the api-ms-win-core-synch-l1-2-0.dll API set)
        // on modern Windows 10/11 — kernel32 no longer exports them there.
        // Try kernel32 first, then KERNELBASE, then the API-set DLL, so the
        // first module resolving the full family wins.
        const wchar_t* kModules[] = {
            L"kernel32.dll",
            L"KERNELBASE.dll",
            L"api-ms-win-core-synch-l1-2-0.dll",
        };
        for (const wchar_t* name : kModules) {
            HMODULE mod = GetModuleHandleW(name);
            if (!mod) mod = LoadLibraryW(name);
            if (!mod) continue;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-function-type"
            auto pWait = reinterpret_cast<WaitOnAddress_fn>(
                GetProcAddress(mod, "WaitOnAddress"));
            auto pWakeOne = reinterpret_cast<WakeByAddressSingle_fn>(
                GetProcAddress(mod, "WakeByAddressSingle"));
            auto pWakeAll = reinterpret_cast<WakeByAddressAll_fn>(
                GetProcAddress(mod, "WakeByAddressAll"));
#pragma GCC diagnostic pop

            if (!pWait || !pWakeOne || !pWakeAll) continue;

            s_WaitOnAddress      = pWait;
            s_WakeByAddressSingle = pWakeOne;
            s_WakeByAddressAll   = pWakeAll;
            return true;
        }
        return false;
#elif LRU_NATIVE_HAS_FUTEX
        return true; // futex is always available on Linux
#elif LRU_NATIVE_HAS_ULOCK
        // ulock_wait/ulock_wake exist on macOS 10.12+. With weak_import,
        // the symbols are nullptr on older runtimes — detect at runtime.
        return __ulock_wait != nullptr && __ulock_wake != nullptr;
#else
        return false;
#endif
    }

    /// One-time warning flag for the CV fallback path.
    static inline std::once_flag s_warn_flag_{};

#if LRU_NATIVE_HAS_WIN32
    // Function pointer types matching the Windows API signatures.
    using WaitOnAddress_fn      = int(__stdcall*)(volatile void*, void*, unsigned long, unsigned long);
    using WakeByAddressSingle_fn = void(__stdcall*)(void*);
    using WakeByAddressAll_fn   = void(__stdcall*)(void*);

    static inline WaitOnAddress_fn       s_WaitOnAddress{};
    static inline WakeByAddressSingle_fn s_WakeByAddressSingle{};
    static inline WakeByAddressAll_fn    s_WakeByAddressAll{};
#endif
};

} // namespace lru::detail

#endif // LRU_DETAIL_NATIVE_WAIT_OPS_HPP
