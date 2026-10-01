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
    # P0 regression: API-surface instantiation. Keeps P0-1 (striped-alias
    # constructibility on GCC) and P0-2 (formerly uncompilable public API
    # methods) from silently regressing.
    test_api_surface.cpp
    test_compact_cache.cpp
    test_lfu_cache.cpp
    test_safe_cache.cpp
    test_serde.cpp
    test_ttl_cache.cpp
    test_eviction.cpp
    test_stress.cpp
    test_slab_memory.cpp
    # spec/fix.01.md §3.3 (P1-13 … P1-18): slab allocator + memory monitor
    # regression tests (ABA tag, wrong-size deallocation, NUMA owner routing,
    # gauge accounting, hard memory cap, per-class live/peak/failure counters).
    test_memory_p1.cpp
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
    # fix.01 §3.4/§3.5 regression batch (P1-19 … P1-44): single admission gate,
    # record_access guards, replace_node coherence, W-TinyLFU window/main swap,
    # strict capacity, eviction-cursor LRU order, pop_lru accounting, TTL.
    test_mm_p1.cpp
    # fix.01 §3.1/§3.2 regression batch: reclamation, seqlock and background
    # worker contracts (P1-1, P1-4, P1-7, P1-9 … P1-12, P1-35, P1-36, P1-39).
    test_reclaim_p1.cpp
    # fix.01 §3.4 regression batch: CountMinSketch sizing from capacity and
    # whole-table decay with a geometrically shrinking window (P1-25, P1-26).
    test_cms_p1.cpp
    # fix.01 §3.5 regression batch: the strategy-agnostic TTL expiry index used
    # by P1-31 (detail::ttl_heap) — sweep/budget/stale/pinned-deferral contracts.
    test_ttl_heap_p1.cpp
    # fix.01 §3.4/§3.6 completion batch: item-level TTL for EVERY eviction
    # strategy with a single O(log n) sweep path (P1-31), and the 2Q A1out ghost
    # queue (P1-28).
    test_mm_2q_ghost_p1.cpp
)
target_link_libraries(lru_cache_test PRIVATE GTest::gtest_main)

lru_add_internal_executable(lru_refcount_test
    test_refcount.cpp
)
target_link_libraries(lru_refcount_test PRIVATE GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(lru_cache_test)
gtest_discover_tests(lru_refcount_test)
