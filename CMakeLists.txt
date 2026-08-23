cmake_minimum_required(VERSION 3.20)

# Auto-detect ccache and wire it up as the compiler launcher.
find_program(LRU_CCACHE_PROGRAM ccache)
if(LRU_CCACHE_PROGRAM)
    # Do not FORCE, so a user-provided launcher can still override this.
    set(CMAKE_CXX_COMPILER_LAUNCHER "${LRU_CCACHE_PROGRAM}" CACHE FILEPATH "C++ compiler launcher")
    set(CMAKE_C_COMPILER_LAUNCHER "${LRU_CCACHE_PROGRAM}" CACHE FILEPATH "C compiler launcher")
    message(STATUS "LRU Cache: ccache enabled (${LRU_CCACHE_PROGRAM})")
else()
    message(STATUS "LRU Cache: ccache not found; builds will not be cached")
endif()

# Default-enable compile_commands.json for IDE integration.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE BOOL "Generate compile_commands.json")

project(lru_cache VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_EXTENSIONS OFF)

# Header-only library
add_library(lru_cache INTERFACE)
add_library(lru::cache ALIAS lru_cache)
set_target_properties(lru_cache PROPERTIES EXPORT_NAME cache)

# C++20 is required; C++23 is optional and controlled by LRU_ENABLE_CPP23.
target_compile_features(lru_cache INTERFACE cxx_std_20)

# P2-H: Enable cmpxchg16b on x86-64 for 128-bit lock-free CAS in the slab
# allocator's tagged-pointer free list. This is required for the 32-bit ABA
# tag extension (the 16-bit tag wrapped after 65K ops, allowing ABA under
# high contention). All modern x86-64 CPUs (since ~2006) support cmpxchg16b.
# MSVC emits cmpxchg16b by default on x86-64, so the flag is only needed for
# GCC/Clang.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64|AMD64|i686|i386|x86")
    if(NOT MSVC)
        target_compile_options(lru_cache INTERFACE -mcx16)
        message(STATUS "LRU Cache: -mcx16 enabled for 128-bit CAS (slab allocator ABA tag)")
    endif()
endif()

option(LRU_ENABLE_CPP23 "Enable C++23 features" OFF)
if(LRU_ENABLE_CPP23)
    target_compile_features(lru_cache INTERFACE cxx_std_23)
    target_compile_definitions(lru_cache INTERFACE LRU_CACHE_CPP23=1)
    message(STATUS "LRU Cache: C++23 features enabled")
endif()

# Find ankerl/unordered_dense.
# 1. Prefer a system-installed or user-provided package.
# 2. Fall back to the sibling directory layout used by this repo.
# 3. If neither is available, fail with a helpful message.
find_package(unordered_dense QUIET)

if(unordered_dense_FOUND)
    target_link_libraries(lru_cache INTERFACE unordered_dense::unordered_dense)
    message(STATUS "LRU Cache: using found unordered_dense package")
else()
    set(LRU_UNORDERED_DENSE_SIBLING "${CMAKE_CURRENT_SOURCE_DIR}/../unordered_dense/include")
    if(EXISTS "${LRU_UNORDERED_DENSE_SIBLING}/ankerl/unordered_dense.h")
        target_include_directories(lru_cache INTERFACE
            $<BUILD_INTERFACE:${LRU_UNORDERED_DENSE_SIBLING}>
        )
        message(STATUS "LRU Cache: using sibling unordered_dense at ${LRU_UNORDERED_DENSE_SIBLING}")
    else()
        message(FATAL_ERROR
            "LRU Cache requires ankerl/unordered_dense.\n"
            "  - Install it so that find_package(unordered_dense) succeeds, OR\n"
            "  - Place the unordered_dense sources in ${CMAKE_CURRENT_SOURCE_DIR}/../unordered_dense\n"
            "    so that ${LRU_UNORDERED_DENSE_SIBLING}/ankerl/unordered_dense.h exists.")
    endif()
endif()

target_include_directories(lru_cache INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

# Installation
include(GNUInstallDirs)

install(TARGETS lru_cache
    EXPORT lru_cache-targets
)

install(EXPORT lru_cache-targets
    FILE lru_cache-targets.cmake
    NAMESPACE lru::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/lru_cache
)

include(CMakePackageConfigHelpers)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/lru_cache-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/lru_cache-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/lru_cache
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/lru_cache-config-version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/lru_cache-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/lru_cache-config-version.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/lru_cache
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/lru
    FILES_MATCHING PATTERN "*.hpp"
    PATTERN "tests" EXCLUDE
    PATTERN "examples" EXCLUDE
    PATTERN "benchmarks" EXCLUDE
    PATTERN "build" EXCLUDE
    PATTERN ".git" EXCLUDE
    PATTERN "cmake" EXCLUDE
)

# Sanitizer options
option(LRU_ENABLE_ASAN  "Enable Address Sanitizer" OFF)
option(LRU_ENABLE_TSAN  "Enable Thread Sanitizer" OFF)
option(LRU_ENABLE_UBSAN "Enable Undefined Behavior Sanitizer" OFF)
option(LRU_ENABLE_LSAN  "Enable Leak Sanitizer (standalone)" OFF)

if(LRU_ENABLE_TSAN AND (WIN32 OR MINGW))
    message(FATAL_ERROR
        "LRU_ENABLE_TSAN is not supported on MinGW/Windows. "
        "Use Linux or WSL for ThreadSanitizer builds.")
endif()

# Hard constraint: ASan and TSan are mutually exclusive (linking both
# produces a runtime error: "ASan/TSan are mutually exclusive"). LSan
# (standalone -fsanitize=leak) is a strict subset of ASan and must NOT
# be combined with either ASan or TSan.
if(LRU_ENABLE_ASAN AND LRU_ENABLE_TSAN)
    message(FATAL_ERROR
        "LRU_ENABLE_ASAN and LRU_ENABLE_TSAN are mutually exclusive. "
        "AddressSanitizer and ThreadSanitizer cannot be linked into the "
        "same binary (the ASan runtime intercepts TSan's interceptor hooks).")
endif()
if(LRU_ENABLE_LSAN AND LRU_ENABLE_ASAN)
    message(WARNING
        "LRU_ENABLE_LSAN is redundant with LRU_ENABLE_ASAN: ASan already "
        "includes leak detection. Disabling standalone LSan to avoid "
        "duplicate runtime symbol definitions.")
    set(LRU_ENABLE_LSAN OFF CACHE BOOL "Enable Leak Sanitizer (standalone)" FORCE)
endif()
if(LRU_ENABLE_LSAN AND LRU_ENABLE_TSAN)
    message(FATAL_ERROR
        "LRU_ENABLE_LSAN and LRU_ENABLE_TSAN are mutually exclusive. "
        "LeakSanitizer is not compatible with ThreadSanitizer; TSan has "
        "no leak tracking and the LSan runtime conflicts with TSan's.")
endif()
# UBSan composes safely with ASan and LSan on Clang/GCC. It does NOT
# compose with TSan when recoverability semantics differ (TSan requires
# -fno-sanitize-recover=thread; mixing recover policies is undefined).
if(LRU_ENABLE_UBSAN AND LRU_ENABLE_TSAN)
    message(WARNING
        "LRU_ENABLE_UBSAN combined with LRU_ENABLE_TSAN may produce "
        "conflicting -fno-sanitize-recover policies. The build will "
        "proceed but consider running them separately for clearest reports.")
endif()

# Emit a one-line summary of the active sanitizer configuration so the
# configure log shows at a glance which sanitizers are in effect. This
# is important because the LRU_ENABLE_* flags are sticky in the CMake
# cache — a stale ON value from a prior configure silently carries over
# to a subsequent "plain" build in the same BUILD_DIR unless the cache
# is wiped. The summary makes such carryover visible.
set(_lru_active_sanitizers "")
if(LRU_ENABLE_ASAN)
    list(APPEND _lru_active_sanitizers "ASan")
endif()
if(LRU_ENABLE_TSAN)
    list(APPEND _lru_active_sanitizers "TSan")
endif()
if(LRU_ENABLE_UBSAN)
    list(APPEND _lru_active_sanitizers "UBSan")
endif()
if(LRU_ENABLE_LSAN)
    list(APPEND _lru_active_sanitizers "LSan")
endif()
if(_lru_active_sanitizers)
    string(REPLACE ";" " + " _lru_san_summary "${_lru_active_sanitizers}")
    message(STATUS "LRU Cache: sanitizers enabled — ${_lru_san_summary}")
else()
    message(STATUS "LRU Cache: no sanitizers enabled")
endif()
unset(_lru_active_sanitizers)
unset(_lru_san_summary)

# Sub-project options
option(LRU_BUILD_TESTS "Build tests" ON)
option(LRU_BUILD_EXAMPLES "Build examples" ON)
option(LRU_BUILD_BENCHMARKS "Build benchmarks" OFF)

# Debug instrumentation (default OFF in Release, ON for Debug builds of tests).
# When enabled, read_handle<T> maintains a per-T global atomic counter
# (s_active_count_) tracking the number of live handles. In release builds
# this counter is not maintained (always 0) — active_handle_count() is
# sourced from per-cache cache_stats instead, eliminating the cross-cache
# cache-line ping-pong that this per-T global caused under high read QPS.
# See P2-1 in spec.md.
if(LRU_BUILD_TESTS AND CMAKE_BUILD_TYPE STREQUAL "Debug")
    option(LRU_DEBUG "Enable LRU debug instrumentation (per-T global handle counter)" ON)
else()
    option(LRU_DEBUG "Enable LRU debug instrumentation (per-T global handle counter)" OFF)
endif()

if(LRU_DEBUG)
    target_compile_definitions(lru_cache INTERFACE LRU_DEBUG=1)
    message(STATUS "LRU Cache: LRU_DEBUG instrumentation enabled")
endif()

# Internal warning target for tests/examples/benchmarks; not exported.
add_library(lru_cache_warnings INTERFACE)

if(MSVC)
    target_compile_options(lru_cache_warnings INTERFACE
        /W4
        /permissive-
        /w14640
        /w14242
        /w14254
        /w14263
        /w14265
        /w14287
        /we4289
        /w14296
        /w14311
        /w14545
        /w14547
        /w14549
        /w14619
        /w14826
        /w14905
        /w14906
        /w14928
    )
else()
    target_compile_options(lru_cache_warnings INTERFACE
        -Wall
        -Wextra
        -Wpedantic
        -Wshadow
        -Wnon-virtual-dtor
        -Wold-style-cast
        -Wcast-align
        -Wunused
        -Woverloaded-virtual
        -Wconversion
        -Wsign-conversion
        -Wnull-dereference
        -Wdouble-promotion
        -Wformat=2
        -Wimplicit-fallthrough
    )
endif()

option(LRU_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)
if(LRU_WARNINGS_AS_ERRORS)
    if(MSVC)
        target_compile_options(lru_cache_warnings INTERFACE /WX)
    else()
        target_compile_options(lru_cache_warnings INTERFACE -Werror)
    endif()
    message(STATUS "LRU Cache: warnings treated as errors")
endif()

# Helper to apply sanitizer flags consistently.
#
# Composition rules (enforced by the option guards above):
#   - ASan + UBSan: allowed (Clang/GCC compose cleanly; UBSan's
#     -fsanitize-recover=undefined keeps UBSan reports non-fatal while
#     ASan aborts on the first error).
#   - ASan alone: implicitly includes LeakSanitizer on Linux (ASan's
#     runtime links LSan in). On Windows/MinGW LSan is not available,
#     so use -fsanitize=address with detect_leaks=0 in ASAN_OPTIONS.
#   - LSan standalone: only meaningful on Linux where you want leak
#     detection WITHOUT the overhead of the full ASan shadow memory.
#   - TSan: mutually exclusive with both ASan and LSan.
#   - UBSan + TSan: warned above; the recover policies conflict.
function(lru_add_sanitizers target)
    if(LRU_ENABLE_ASAN)
        target_compile_options(${target} PRIVATE -fsanitize=address -g)
        target_link_options(${target} PRIVATE -fsanitize=address)
    endif()
    if(LRU_ENABLE_TSAN)
        target_compile_options(${target} PRIVATE -fsanitize=thread -g)
        target_link_options(${target} PRIVATE -fsanitize=thread)
    endif()
    if(LRU_ENABLE_LSAN)
        # Standalone LeakSanitizer. On Linux this is -fsanitize=leak;
        # on other platforms LSan is unavailable (ASan embeds it).
        target_compile_options(${target} PRIVATE -fsanitize=leak -g)
        target_link_options(${target} PRIVATE -fsanitize=leak)
    endif()
    if(LRU_ENABLE_UBSAN)
        # UBSan: full undefined-behavior set. The project is header-only
        # with no third-party TUs compiled here, so we enable the full
        # set (alignment, bool, builtin, enum, integer-overflow, null,
        # return, shift, signed-integer-overflow, unreachable, vla-bound,
        # vptr). -fno-sanitize-recover=undefined makes UBSan fatal so a
        # test hits a clear abort on the first UB instead of silently
        # continuing with an indeterminate value — this is what we want
        # for CI: a single UB must fail the run, not get lost in noise.
        # To allow ASan+UBSan composition, ASan remains the abort path
        # for memory errors while UBSan aborts on UB; both are fatal.
        target_compile_options(${target} PRIVATE
            -fsanitize=undefined
            -fno-sanitize-recover=undefined
            -g)
        target_link_options(${target} PRIVATE
            -fsanitize=undefined
            -fno-sanitize-recover=undefined)
    endif()
endfunction()

# Helper for internal executables (tests/examples/benchmarks).
function(lru_add_internal_executable target)
    add_executable(${target} ${ARGN})
    target_link_libraries(${target} PRIVATE lru::cache)
    target_compile_features(${target} PRIVATE cxx_std_20)
    target_link_libraries(${target} PRIVATE lru_cache_warnings)
    lru_add_sanitizers(${target})
endfunction()

# Tests
if(LRU_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Examples
if(LRU_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Benchmarks (requires Google Benchmark)
if(LRU_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()
