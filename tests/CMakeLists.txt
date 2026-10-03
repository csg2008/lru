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
    # API-surface instantiation. Keeps striped-alias constructibility on GCC
    # and the formerly uncompilable public API methods from regressing.
    test_api_surface.cpp
    test_compact_cache.cpp
    test_lfu_cache.cpp
    test_safe_cache.cpp
    test_serde.cpp
    test_ttl_cache.cpp
    test_eviction.cpp
    test_stress.cpp
    test_slab_memory.cpp
    # Slab allocator and memory monitor:
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
    # Concurrency test supplement.
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
    # production-read-heavy-eval test coverage
    test_128_threads_soak.cpp
    test_ttl_cache_concurrent_cleanup.cpp
    test_ebr_force_advance_uaf.cpp
    # TTL lazy delete, the async rejection policy, and pooled_cache basics.
    test_ttl_lazy_delete.cpp
    test_async_callback_reject_policy.cpp
    test_pooled_cache.cpp
    # Admission and capacity regressions: the single admission gate, the
    # record_access guards, replace_node coherence, the W-TinyLFU window/main
    # swap, strict capacity, eviction-cursor LRU order, pop_lru accounting, TTL.
    test_mm_p1.cpp
    # Reclamation, seqlock and background-worker contracts.
    test_reclaim_p1.cpp
    # CountMinSketch sizing from capacity, and whole-table decay with a
    # geometrically shrinking window.
    test_cms_p1.cpp
    # The strategy-agnostic TTL expiry index (detail::ttl_heap): sweep, budget,
    # stale-entry and pinned-deferral contracts.
    test_ttl_heap_p1.cpp
    # Item-level TTL for EVERY eviction strategy through one O(log n) sweep, and
    # the 2Q A1out ghost queue.
    test_mm_2q_ghost_p1.cpp
    # The per-bucket shared_spinlock wake protocol: a parked reader MUST be
    # released by the writer's unlock(). Dropping that wake (which an earlier
    # revision did when the state word showed no waiter) strands every queued
    # reader, and this lock guards every hash bucket.
    test_spinlock_wakeup.cpp
    # Asynchronous write-back must not replay an evicted value after the caller
    # has set()/del()ed that key.
    test_tiered_writeback.cpp
)
target_link_libraries(lru_cache_test PRIVATE GTest::gtest_main)

lru_add_internal_executable(lru_refcount_test
    test_refcount.cpp
)
target_link_libraries(lru_refcount_test PRIVATE GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(lru_cache_test)
gtest_discover_tests(lru_refcount_test)
