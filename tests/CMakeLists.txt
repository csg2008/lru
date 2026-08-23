# Tests CMakeLists.txt

include(FetchContent)

# Prefer a system-installed GTest, fall back to a local source directory,
# and finally fetch from GitHub if neither is available.
find_package(GTest QUIET)

if(NOT GTest_FOUND)
    set(LRU_GTEST_SOURCE_DIR "" CACHE PATH "Path to a local GoogleTest source directory")
    mark_as_advanced(LRU_GTEST_SOURCE_DIR)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

    if(LRU_GTEST_SOURCE_DIR AND EXISTS "${LRU_GTEST_SOURCE_DIR}/CMakeLists.txt")
        message(STATUS "LRU Cache tests: using local GTest at ${LRU_GTEST_SOURCE_DIR}")
        FetchContent_Declare(
            googletest
            SOURCE_DIR ${LRU_GTEST_SOURCE_DIR}
        )
    else()
        message(STATUS "LRU Cache tests: fetching GoogleTest from GitHub")
        FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG v1.14.0
        )
    endif()

    # Avoid repeated network access on reconfigure once sources are present.
    set(FETCHCONTENT_UPDATES_DISCONNECTED_GOOGLETEST ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

lru_add_internal_executable(lru_cache_test
    test_cache.cpp
    test_cache_stats.cpp
    test_compact_cache.cpp
    test_lfu_cache.cpp
    test_safe_cache.cpp
    test_serde.cpp
    test_ttl_cache.cpp
    test_eviction.cpp
    test_stress.cpp
    test_slab_memory.cpp
    test_hazptr.cpp
    test_production_api.cpp
    test_production_metrics.cpp
    test_tiered.cpp
    test_shared_memory.cpp
    test_mutex.cpp
    test_concurrent_safety.cpp
    test_chaos.cpp
    test_latency_histogram_sum.cpp
    test_compressed_cache.cpp
    test_prehashed.cpp
    test_reclaim_metrics.cpp
    test_hot_shards_memory.cpp
    test_f14_diagnostics.cpp
    test_tls_event_backup.cpp
    test_ebr_integration.cpp
    # Concurrency test supplement (specs/concurrency-tests/spec.md)
    test_concurrent_edge.cpp
    test_concurrent_callback.cpp
    test_concurrent_rehash.cpp
    test_concurrent_metrics.cpp
    test_concurrent_mm_strategies.cpp
    test_concurrent_mutex_runtime.cpp
    test_concurrent_edge_extras.cpp
    test_tls_ring_improvements.cpp
    test_read_heavy_concurrent.cpp
    test_read_heavy_soak.cpp
    test_high_concurrency_suite.cpp
    test_warm_cache.cpp
    test_deadlock_repro.cpp
    test_max_size_strict.cpp
    test_save_atomic.cpp
    test_prometheus_per_shard_detail.cpp
    # T-G14: production-read-heavy-eval test coverage
    test_128_threads_soak.cpp
    test_ttl_cache_concurrent_cleanup.cpp
    test_ebr_force_advance_uaf.cpp
    # G22: regression tests for G1 (TTL lazy delete), G7 (async kReject),
    # and pooled_cache basic functionality.
    test_ttl_lazy_delete.cpp
    test_async_callback_reject_policy.cpp
    test_pooled_cache.cpp
)
target_link_libraries(lru_cache_test PRIVATE GTest::gtest_main)

lru_add_internal_executable(lru_refcount_test
    test_refcount.cpp
)
target_link_libraries(lru_refcount_test PRIVATE GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(lru_cache_test)
gtest_discover_tests(lru_refcount_test)
