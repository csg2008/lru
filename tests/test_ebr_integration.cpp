// Unified LRU Cache - EBR (Epoch-Based Reclamation) Integration Tests
// T2.1 / T2.2 / T2.4: Validates that:
//   - set_ebr_domain() propagates to the hash table
//   - is_ebr_mode() reports the correct state
//   - find_and_pin_lockfree acquires epoch_guard at entry (no UAF)
//   - The reclaim_guard unified type works in both modes
//   - Non-sharded MM types without EBR support gracefully ignore set_ebr_domain

#include <gtest/gtest.h>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

#include "../lru.hpp"

using namespace lru;

// ============================================================================
// T2.1 / T2.4: EBR domain setup and mode query
// ============================================================================

TEST(EbrIntegration, DefaultIsNotEbrMode) {
    cache<int, std::string> c(100);
    EXPECT_FALSE(c.is_ebr_mode());
}

TEST(EbrIntegration, SetEbrDomainActivatesEbrMode) {
    cache<int, std::string> c(100);
    auto& domain = detail::epoch_domain::default_domain();
    c.set_ebr_domain(&domain);
    EXPECT_TRUE(c.is_ebr_mode());
}

TEST(EbrIntegration, SetEbrDomainNullDeactivatesEbrMode) {
    cache<int, std::string> c(100);
    auto& domain = detail::epoch_domain::default_domain();
    c.set_ebr_domain(&domain);
    EXPECT_TRUE(c.is_ebr_mode());
    c.set_ebr_domain(nullptr);
    EXPECT_FALSE(c.is_ebr_mode());
}

TEST(EbrIntegration, StripedCachePropagatesEbrDomain) {
    striped_cache<int, std::string> c(1000);
    EXPECT_FALSE(c.is_ebr_mode());
    auto& domain = detail::epoch_domain::default_domain();
    c.set_ebr_domain(&domain);
    EXPECT_TRUE(c.is_ebr_mode());
    c.set_ebr_domain(nullptr);
    EXPECT_FALSE(c.is_ebr_mode());
}

TEST(EbrIntegration, ProductionCachePropagatesEbrDomain) {
    production_cache<int, std::string> c(1000);
    // production_sharded_lru_trait sets auto_enable_ebr = true (R-6 / O10
    // production default), so EBR is active immediately after construction.
    EXPECT_TRUE(c.is_ebr_mode());
    // Setting the domain explicitly is still supported and remains a no-op
    // when EBR is already enabled.
    auto& domain = detail::epoch_domain::default_domain();
    c.set_ebr_domain(&domain);
    EXPECT_TRUE(c.is_ebr_mode());
}

// ============================================================================
// T2.2: Read paths work correctly under EBR mode
// ============================================================================

TEST(EbrIntegration, GetWorksUnderEbrMode) {
    cache<int, std::string> c(100);
    auto& domain = detail::epoch_domain::default_domain();
    c.set_ebr_domain(&domain);
    c.set(1, "one");
    auto h = c.get(1);
    ASSERT_TRUE(h.has_value());
    EXPECT_EQ(*h, "one");
}

TEST(EbrIntegration, TryGetWorksUnderEbrMode) {
    cache<int, std::string> c(100);
    auto& domain = detail::epoch_domain::default_domain();
    c.set_ebr_domain(&domain);
    c.set(1, "one");
    auto h = c.try_get(1);
    ASSERT_TRUE(h.has_value());
    EXPECT_EQ(**h, "one");
}

TEST(EbrIntegration, BulkGetWorksUnderEbrMode) {
    striped_cache<int, std::string> c(1000);
    auto& domain = detail::epoch_domain::default_domain();
    c.set_ebr_domain(&domain);
    for (int i = 0; i < 10; ++i) {
        c.set(i, std::to_string(i));
    }
    std::vector<int> keys = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto results = c.bulk_get(keys.begin(), keys.end());
    EXPECT_EQ(results.size(), 10u);
    for (std::size_t i = 0; i < results.size(); ++i) {
        ASSERT_TRUE(results[i].has_value()) << "key " << i << " missing";
        EXPECT_EQ(**results[i], std::to_string(i));
    }
}

TEST(EbrIntegration, PeekWorksUnderEbrMode) {
    cache<int, std::string> c(100);
    auto& domain = detail::epoch_domain::default_domain();
    c.set_ebr_domain(&domain);
    c.set(1, "one");
    auto h = c.peek(1);
    ASSERT_TRUE(h.has_value());
    EXPECT_EQ(*h, "one");
}

// ============================================================================
// T2.1: Concurrent read + retire under EBR mode (no UAF)
// ============================================================================

TEST(EbrIntegration, ConcurrentReadAndEvictNoUaf) {
    safe_cache<int, std::string> c(100);
    auto& domain = detail::epoch_domain::default_domain();
    c.set_ebr_domain(&domain);

    // Populate
    for (int i = 0; i < 100; ++i) {
        c.set(i, std::to_string(i));
    }

    // Reader thread: continuously reads random keys
    std::atomic<bool> stop{false};
    std::thread reader([&]() {
        while (!stop.load(std::memory_order_relaxed)) {
            int key = std::rand() % 100;
            auto h = c.try_get(key);
            if (h && h->has_value()) {
                // Dereference the value — would crash on UAF
                volatile auto& v = **h;
                (void)v;
            }
        }
    });

    // Writer thread: continuously inserts new keys (triggering evictions)
    std::thread writer([&]() {
        for (int i = 100; i < 1000 && !stop.load(std::memory_order_relaxed); ++i) {
            c.set(i, std::to_string(i));
        }
    });

    writer.join();
    stop.store(true, std::memory_order_relaxed);
    reader.join();
    // If we get here without crashing, EBR protected the readers.
    SUCCEED();
}

TEST(EbrIntegration, StripedConcurrentReadAndEvictNoUaf) {
    striped_cache<int, std::string> c(500);
    auto& domain = detail::epoch_domain::default_domain();
    c.set_ebr_domain(&domain);

    for (int i = 0; i < 500; ++i) {
        c.set(i, std::to_string(i));
    }

    std::atomic<bool> stop{false};
    std::vector<std::thread> readers;
    for (int t = 0; t < 4; ++t) {
        readers.emplace_back([&]() {
            while (!stop.load(std::memory_order_relaxed)) {
                int key = std::rand() % 500;
                auto h = c.try_get(key);
                if (h && h->has_value()) {
                    volatile auto& v = **h;
                    (void)v;
                }
            }
        });
    }

    std::thread writer([&]() {
        for (int i = 500; i < 5000 && !stop.load(std::memory_order_relaxed); ++i) {
            c.set(i, std::to_string(i));
        }
    });

    writer.join();
    stop.store(true, std::memory_order_relaxed);
    for (auto& t : readers) t.join();
    SUCCEED();
}

// ============================================================================
// T2.4: reclaim_guard unified type (tested indirectly via is_ebr_mode)
// ============================================================================

TEST(EbrIntegration, EbrModeCanBeToggledOff) {
    cache<int, std::string> c(100);
    auto& domain = detail::epoch_domain::default_domain();

    // Enable EBR
    c.set_ebr_domain(&domain);
    EXPECT_TRUE(c.is_ebr_mode());
    c.set(1, "one");
    EXPECT_EQ(*c.get(1), "one");

    // Disable EBR (back to hazptr mode)
    c.set_ebr_domain(nullptr);
    EXPECT_FALSE(c.is_ebr_mode());
    EXPECT_EQ(*c.get(1), "one");
}

// ============================================================================
// Non-mm_lru types: set_ebr_domain should be a no-op (SFINAE-guarded)
// ============================================================================

TEST(EbrIntegration, LfuCacheSetEbrDomainIsNoOp) {
    // mm_tiny_lfu doesn't support EBR — set_ebr_domain should be a no-op
    lfu_cache<int, std::string> c(100);
    auto& domain = detail::epoch_domain::default_domain();
    c.set_ebr_domain(&domain);  // Should not crash
    EXPECT_FALSE(c.is_ebr_mode());  // EBR not active
    // Cache still works
    c.set(1, "one");
    EXPECT_EQ(*c.get(1), "one");
}

TEST(EbrIntegration, TwoQCacheSetEbrDomainIsNoOp) {
    two_q<int, std::string> c(100);
    auto& domain = detail::epoch_domain::default_domain();
    c.set_ebr_domain(&domain);
    EXPECT_FALSE(c.is_ebr_mode());
    c.set(1, "one");
    EXPECT_EQ(*c.get(1), "one");
}

TEST(EbrIntegration, FifoCacheSetEbrDomainIsNoOp) {
    fifo_cache<int, std::string> c(100);
    auto& domain = detail::epoch_domain::default_domain();
    c.set_ebr_domain(&domain);
    EXPECT_FALSE(c.is_ebr_mode());
    c.set(1, "one");
    EXPECT_EQ(*c.get(1), "one");
}

// ============================================================================
// R9: per-shard EBR domains (opt-in)
// ============================================================================

TEST(EbrIntegration, EnablePerShardEbrActivatesAndWorks) {
    striped_cache<int, std::string> c(1000);
    EXPECT_FALSE(c.per_shard_ebr_enabled());
    EXPECT_EQ(c.ebr_domain_count(), 1u);

    EXPECT_TRUE(c.enable_per_shard_ebr());
    EXPECT_TRUE(c.per_shard_ebr_enabled());
    EXPECT_GT(c.ebr_domain_count(), 1u);
    EXPECT_TRUE(c.is_ebr_mode());

    // reads/writes still work under per-shard domains
    c.set(1, "one");
    auto h = c.get(1);
    ASSERT_TRUE(h);
    EXPECT_EQ(*h, "one");
}

TEST(EbrIntegration, PerShardEbrPropagatesReclaimConfig) {
    striped_cache<int, int> c(1000);
    c.enable_per_shard_ebr();

    // Force-advance policy propagates to every per-shard domain.
    c.set_force_advance_policy(detail::force_advance_policy::kFailAdvance);
    EXPECT_EQ(c.get_force_advance_policy(),
              detail::force_advance_policy::kFailAdvance);
    // Timeout propagates too.
    c.set_epoch_force_advance_timeout(std::chrono::seconds(3));
    EXPECT_EQ(c.epoch_force_advance_timeout(), std::chrono::seconds(3));
    // Reclaim threshold propagates.
    c.set_reclaim_threshold(1024);
    EXPECT_LE(c.reclaim_threshold(), 1024u);
}

TEST(EbrIntegration, PerShardEbrReclaimsAfterEviction) {
    // Small capacity forces eviction + retirement traffic.
    striped_cache<int, int> c(8);
    c.enable_per_shard_ebr();
    for (int i = 0; i < 200; ++i) {
        c.set(i, i);
    }
    // Full drain across all per-shard domains (no UAF expected).
    c.try_reclaim_now(0);
    // Reads still work after eviction + reclamation.
    EXPECT_FALSE(c.get(0));
    EXPECT_TRUE(c.get(199));
}

// ============================================================================
// R8: defer_promotion mode must still collect hit callbacks
// ============================================================================

TEST(EbrIntegration, DeferPromotionCollectsHitCallbacks) {
    // read_heavy_striped_cache uses defer_promotion=true (TLS-batched LRU
    // promotion). R8: hit callbacks must still fire — previously the defer
    // branch skipped collect_hit entirely, silently undercounting hits.
    read_heavy_striped_cache<int, int> c(1000);
    std::atomic<int> hits{0};
    c.on_hit([&](const int&, const int&) { hits.fetch_add(1); });

    c.set(1, 10);
    c.set(2, 20);
    c.get(1);
    c.get(2);
    c.get(1);

    // Force-drain the TLS callback ring so collected hits dispatch.
    c.flush();
    EXPECT_GE(hits.load(), 3);
}

// ============================================================================
// R9: pending_deletion soft cap — refuse force_del of pinned items when the
// deferred-deletion list is at/over the cap, to bound memory retention.
// ============================================================================

TEST(EbrIntegration, PendingDeletionSoftCapRefusesForceDel) {
    // Non-sharded MM: a single pending_deletion_ list, so the cap is
    // observable deterministically (sharded caches cap per-shard).
    mm_lru_config cfg;
    cfg.max_pending_deletion = 1;
    safe_cache<int, int> c(100, cfg);

    c.set(1, 10);
    c.set(2, 20);

    // Pin key 1, then force_del it -> goes to pending_deletion_ (size 1 == cap).
    auto h1 = c.get(1);
    ASSERT_TRUE(h1);
    EXPECT_TRUE(c.force_del(1));

    // Pin key 2, then force_del it while pending is at cap -> refused.
    auto h2 = c.get(2);
    ASSERT_TRUE(h2);
    EXPECT_FALSE(c.force_del(2));
    EXPECT_GE(c.pending_deletion_skipped_count(), 1u);

    // The refused item is still intact and readable.
    EXPECT_TRUE(c.get(2));
}

