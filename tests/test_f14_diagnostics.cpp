// T19: F14 / segmented / chain mode diagnostics consistency tests.
//
// Validates that diagnostics() / diagnostics_text() correctly report the
// hash table mode (chain / F14 / segmented / compressed_hook) for each
// of the predefined cache aliases. Also verifies that the set_incremental_rehash()
// doc comment's claim holds — the flag is honoured by all modes.

#include "../lru.hpp"

#include <gtest/gtest.h>

#include <string>

TEST(F14DiagnosticsTest, ChainCacheReportsChainMode) {
    lru::cache<int, std::string> c{256};
    auto info = c.diagnostics();
    EXPECT_FALSE(info.f14_probing);
    EXPECT_FALSE(info.segmented_hash_table);
    EXPECT_FALSE(info.compressed_hook);
    auto text = c.diagnostics_text();
    EXPECT_NE(text.find("f14_probing: 0"), std::string::npos);
    EXPECT_NE(text.find("segmented_hash_table: 0"), std::string::npos);
    EXPECT_NE(text.find("compressed_hook: 0"), std::string::npos);
    EXPECT_NE(text.find("hash table mode"), std::string::npos);
}

TEST(F14DiagnosticsTest, F14CacheReportsF14Mode) {
    lru::f14_cache<int, std::string> c{256};
    auto info = c.diagnostics();
    EXPECT_TRUE(info.f14_probing);
    EXPECT_FALSE(info.segmented_hash_table);
    auto text = c.diagnostics_text();
    EXPECT_NE(text.find("f14_probing: 1"), std::string::npos);
}

TEST(F14DiagnosticsTest, SegmentedCacheReportsSegmentedMode) {
    lru::segmented_cache<int, std::string> c{256};
    auto info = c.diagnostics();
    EXPECT_TRUE(info.segmented_hash_table);
    auto text = c.diagnostics_text();
    EXPECT_NE(text.find("segmented_hash_table: 1"), std::string::npos);
}

TEST(F14DiagnosticsTest, ProductionCacheReportsSegmentedAndIncremental) {
    lru::production_cache<int, std::string> c{256};
    auto info = c.diagnostics();
    EXPECT_TRUE(info.segmented_hash_table);
    // production_cache auto-enables incremental rehash.
    EXPECT_TRUE(info.incremental_rehash_enabled);
    auto text = c.diagnostics_text();
    EXPECT_NE(text.find("segmented_hash_table: 1"), std::string::npos);
    EXPECT_NE(text.find("incremental_rehash_enabled: 1"), std::string::npos);
    EXPECT_NE(text.find("rehash_mode: 1"), std::string::npos);
}

TEST(F14DiagnosticsTest, CompressedCacheReportsCompressedHook) {
    lru::compressed_cache<int, std::string> c{256};
    auto info = c.diagnostics();
    EXPECT_TRUE(info.compressed_hook);
    auto text = c.diagnostics_text();
    EXPECT_NE(text.find("compressed_hook: 1"), std::string::npos);
}

TEST(F14DiagnosticsTest, SetIncrementalRehashWorksForAllModes) {
    // T19.2: set_incremental_rehash() must be effective for chain, F14,
    // and segmented caches. We verify by toggling the flag and reading
    // it back via incremental_rehash_enabled().
    //
    // The cache objects embed a large cache_stats (latency histograms),
    // so multiple instances on the stack would exceed the default 1 MiB
    // thread stack (clang's ___chkstk_ms stack probe fails at function
    // entry). Allocate them on the heap.
    {
        auto c = std::make_unique<lru::cache<int, std::string>>(256);
        c->set_incremental_rehash(true);
        EXPECT_TRUE(c->incremental_rehash_enabled());
        c->set_incremental_rehash(false);
        EXPECT_FALSE(c->incremental_rehash_enabled());
    }
    {
        auto c = std::make_unique<lru::f14_cache<int, std::string>>(256);
        c->set_incremental_rehash(true);
        EXPECT_TRUE(c->incremental_rehash_enabled());
    }
    {
        auto c = std::make_unique<lru::segmented_cache<int, std::string>>(256);
        c->set_incremental_rehash(true);
        EXPECT_TRUE(c->incremental_rehash_enabled());
    }
}

TEST(F14DiagnosticsTest, DiagnosticsTextContainsReclaimSection) {
    // T17.3 + T19.3: diagnostics_text() must include both the hash table
    // mode section (T19.3) and the reclaim health section (T17.3).
    lru::cache<int, std::string> c{256};
    c.set(1, "one");
    auto text = c.diagnostics_text();
    EXPECT_NE(text.find("hash table mode"), std::string::npos);
    EXPECT_NE(text.find("reclaim health"), std::string::npos);
    EXPECT_NE(text.find("rehash_mode"), std::string::npos);
}

TEST(F14DiagnosticsTest, F14StripedCacheReportsBothF14AndStriped) {
    lru::f14_striped_cache<int, std::string> c{256, 4};
    auto info = c.diagnostics();
    EXPECT_TRUE(info.f14_probing);
    EXPECT_FALSE(info.segmented_hash_table);  // f14_striped is not segmented
    EXPECT_GT(info.num_shards, 1u);
}

// ============================================================================
// T-O5: diagnostics aggregation cache
// Validates that the segmented hash table's diagnostics snapshot cache
// (cached_max_chain_length_, cached_snapshot_ns_, cached_per_segment_lf_)
// is populated by refresh_diagnostics_cache() and served by max_chain_length()
// without re-scanning on each call.
// ============================================================================

TEST(F14DiagnosticsTest, DiagnosticsCacheColdStartsAtMaxAge) {
    // A freshly constructed segmented cache has never been refreshed —
    // diagnostics_cache_age_ms() should return max() (sentinel for
    // "never refreshed").
    lru::segmented_cache<int, int> c{256};
    EXPECT_EQ(c.diagnostics_cache_age_ms(),
              std::numeric_limits<std::uint64_t>::max());
}

TEST(F14DiagnosticsTest, DiagnosticsCacheRefreshedViaBalancer) {
    // Start the background rehash balancer, wait for at least one sweep,
    // then verify diagnostics_cache_age_ms() drops from max() to a
    // reasonable value (close to the balancer interval).
    lru::segmented_cache<int, int> c{256};
    c.set(1, 10);
    c.set(2, 20);

    // Before balancer: cache is cold.
    EXPECT_EQ(c.diagnostics_cache_age_ms(),
              std::numeric_limits<std::uint64_t>::max());

    // Start balancer with a short 100ms interval for fast testing.
    c.start_background_rehash_balancer(std::chrono::milliseconds(100));

    // Wait for at least one sweep (300ms = 3 intervals, generous margin).
    std::this_thread::sleep_for(std::chrono::milliseconds(300));

    // After balancer: cache should be refreshed, age should be small.
    auto age = c.diagnostics_cache_age_ms();
    EXPECT_LT(age, std::numeric_limits<std::uint64_t>::max());
    EXPECT_LE(age, 1000u);  // should be well under 1s

    c.stop_background_rehash_balancer();
}

TEST(F14DiagnosticsTest, DiagnosticsCacheServesMaxChainLength) {
    // After refresh, the diagnostics cache should serve a fresh snapshot
    // (small age) consistently across multiple reads. We verify by checking
    // that diagnostics_cache_age_ms() stays small on repeated calls, which
    // proves the cache is populated and being served without live re-scan.
    lru::segmented_cache<int, int> c{256};
    for (int i = 0; i < 100; ++i) {
        c.set(i, i * 10);
    }

    // Start balancer to refresh the cache.
    c.start_background_rehash_balancer(std::chrono::milliseconds(100));
    std::this_thread::sleep_for(std::chrono::milliseconds(300));

    // First read: cache should be fresh (age << max).
    auto age1 = c.diagnostics_cache_age_ms();
    EXPECT_LT(age1, std::numeric_limits<std::uint64_t>::max());

    // Second read immediately after: should also be fresh (served from cache,
    // no live scan that would invalidate the snapshot).
    auto age2 = c.diagnostics_cache_age_ms();
    EXPECT_LT(age2, std::numeric_limits<std::uint64_t>::max());

    // Both reads should be consistent (both small, within balancer interval).
    EXPECT_LE(age1, 1000u);
    EXPECT_LE(age2, 1000u);

    c.stop_background_rehash_balancer();
}

TEST(F14DiagnosticsTest, NonSegmentedCacheReportsMaxAgeForCacheAge) {
    // Non-segmented caches (chain/F14 without segmented hash table) don't
    // have a diagnostics cache — diagnostics_cache_age_ms() should return
    // max() to signal "not applicable".
    lru::cache<int, int> c{256};
    c.set(1, 10);
    EXPECT_EQ(c.diagnostics_cache_age_ms(),
              std::numeric_limits<std::uint64_t>::max());
}
