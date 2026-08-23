# SPDX-License-Identifier: MIT
# Benchmarks CMakeLists.txt

include(FetchContent)

# Prefer a system-installed Google Benchmark, fall back to a local source
# directory, and finally fetch from GitHub if neither is available.
find_package(benchmark QUIET)

if(NOT benchmark_FOUND)
    set(LRU_BENCHMARK_SOURCE_DIR "" CACHE PATH "Path to a local Google Benchmark source directory")
    mark_as_advanced(LRU_BENCHMARK_SOURCE_DIR)
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    # Google Benchmark v1.9.1 enables -Werror for clang by default
    # (BENCHMARK_ENABLE_WERROR=ON), which breaks the build on newer clang
    # versions that flag `__COUNTER__` as a C2y extension. Disable it —
    # warnings in the third-party library are not actionable from this repo.
    set(BENCHMARK_ENABLE_WERROR OFF CACHE BOOL "" FORCE)

    if(LRU_BENCHMARK_SOURCE_DIR AND EXISTS "${LRU_BENCHMARK_SOURCE_DIR}/CMakeLists.txt")
        message(STATUS "LRU Cache benchmarks: using local Google Benchmark at ${LRU_BENCHMARK_SOURCE_DIR}")
        FetchContent_Declare(
            googlebenchmark
            SOURCE_DIR ${LRU_BENCHMARK_SOURCE_DIR}
        )
    else()
        message(STATUS "LRU Cache benchmarks: fetching Google Benchmark from GitHub")
        set(BENCHMARK_DOWNLOAD_DEPENDENCIES ON CACHE BOOL "" FORCE)
        FetchContent_Declare(
            googlebenchmark
            GIT_REPOSITORY https://github.com/google/benchmark.git
            GIT_TAG v1.9.1
        )
    endif()

    # Avoid repeated network access on reconfigure once sources are present.
    set(FETCHCONTENT_UPDATES_DISCONNECTED_GOOGLEBENCHMARK ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googlebenchmark)

    # Google Benchmark v1.9.1 enables -Werror for the `benchmark` target by
    # default, which breaks the build on newer clang that flags `__COUNTER__`
    # as a C2y extension. Strip -Werror from the third-party target so this
    # repo's code is the only thing treated strictly, and disable the specific
    # warning outright (transitive consumers include benchmark headers).
    foreach(_gb_target benchmark benchmark_main)
        if(TARGET ${_gb_target})
            get_target_property(_opts ${_gb_target} COMPILE_OPTIONS)
            if(_opts)
                list(FILTER _opts EXCLUDE REGEX "^-Werror$")
                set_target_properties(${_gb_target} PROPERTIES COMPILE_OPTIONS "${_opts}")
            endif()
            target_compile_options(${_gb_target} PRIVATE -Wno-c2y-extensions)
        endif()
    endforeach()
endif()

set(BENCHMARK_LIB benchmark::benchmark)

lru_add_internal_executable(lru_cache_benchmark
    benchmark_lru.cpp
)
target_link_libraries(lru_cache_benchmark PRIVATE ${BENCHMARK_LIB})
# Suppress clang's -Wc2y-extensions warning triggered by Google Benchmark's
# use of `__COUNTER__` in its public headers (benchmark.h:1534).
target_compile_options(lru_cache_benchmark PRIVATE -Wno-c2y-extensions)

lru_add_internal_executable(lru_concurrent_read_benchmark
    benchmark_concurrent_read.cpp
)
target_link_libraries(lru_concurrent_read_benchmark PRIVATE ${BENCHMARK_LIB})
target_compile_options(lru_concurrent_read_benchmark PRIVATE -Wno-c2y-extensions)
