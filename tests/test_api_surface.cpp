// P0 regression tests — API surface instantiation.
//
// This translation unit exists to make a whole class of defects fail the build
// (or the test run) instead of lurking until a user calls the API:
//
//   * P0-1 — every striped alias must be constructible. GCC previously rejected
//     all of them ("use of deleted function lazy_striped_mutex(lazy_striped_mutex&&)")
//     because unified_cache initialised its [[no_unique_address]] striped-mutex
//     member from a factory prvalue, which requires guaranteed copy elision that
//     GCC does not implement for such members (GCC PR98995). Any reintroduction
//     of that pattern makes this file fail to compile under g++.
//
//   * P0-2 — three public API methods did not compile at all, because the test
//     suite never instantiated them:
//       - sharded_mm_lru::begin()/end() const  (unique_ptr::operator-> returns a
//         non-const T*, so the const overloads needed std::as_const)
//       - mm_lru / mm_2q / mm_tiny_lfu / mm_wtiny_lfu ::set_eviction_predicate()
//         were non-public, so unified_cache::set_eviction_predicate() and
//         sharded_mm_lru::set_eviction_predicate() could not forward to them
//       - segmented_intrusive_list::clear() assigned a void* to a T*
//
// NOTE on allocation: a unified_cache instance is large — measured on
// MSYS2/Clang64, sizeof(production_cache<int,std::string>) == 30144 B and
// sizeof(safe_cache<int,std::string>) == 80128 B, of which cache_stats alone is
// 27136 B (five inline 512-bucket latency histograms plus padded atomics).
// Constructing ~23 such objects in a single stack frame overflows the default
// 1 MB stack (observed as STATUS_STACK_OVERFLOW), so every alias below is
// constructed on the heap, one at a time.

#include <gtest/gtest.h>

#include <cstddef>
#include <functional>
#include <memory>
#include <array>
#include <span>
#include <string>

#include "../lru.hpp"

namespace {

constexpr std::size_t kSmall = 64;

/// Construct one alias on the heap, exercise the documented API surface, and
/// destroy it. Peak stack use is a single cache object.
template <typename Cache>
void exercise_alias(std::size_t max_size = kSmall) {
    auto c = std::make_unique<Cache>(max_size);

    c->set(1, std::string("one"));
    EXPECT_TRUE(static_cast<bool>(c->get(1)));

    // P0-2 (a): set_eviction_predicate() must be reachable on every alias.
    c->set_eviction_predicate([](const typename Cache::key_type&,
                                 const typename Cache::mapped_type&) { return true; });

    // P0-2 (b): const and non-const iteration over the MM layer.
    auto& mm = c->mm();
    for (auto it = mm.begin(); it != mm.end(); ++it) {
        (void)it;
    }
    const auto& cmm = c->mm();
    for (auto it = cmm.begin(); it != cmm.end(); ++it) {
        (void)it;
    }

    c->flush();
}

}  // namespace

// ---------------------------------------------------------------------------
// P0-1: every striped alias must be constructible on every supported compiler
// ---------------------------------------------------------------------------

TEST(P0ApiSurface, StripedAliasesConstructAndOperate) {
    // Each alias below goes through unified_cache's striped constructor, i.e.
    // the code path that was uncompilable under GCC before P0-1.
    auto production = std::make_unique<lru::production_cache<int, std::string>>(256);
    auto striped = std::make_unique<lru::striped_cache<int, std::string>>(256);
    auto segmented_striped =
        std::make_unique<lru::segmented_striped_cache<int, std::string>>(256);
    auto f14_striped = std::make_unique<lru::f14_striped_cache<int, std::string>>(256);
    auto read_heavy_striped =
        std::make_unique<lru::read_heavy_striped_cache<int, std::string>>(256);
    auto f14_production =
        std::make_unique<lru::f14_production_cache<int, std::string>>(256);

    production->set(1, "one");
    striped->set(1, "one");
    segmented_striped->set(1, "one");
    f14_striped->set(1, "one");
    read_heavy_striped->set(1, "one");
    f14_production->set(1, "one");

    EXPECT_TRUE(static_cast<bool>(production->get(1)));
    EXPECT_TRUE(static_cast<bool>(striped->get(1)));
    EXPECT_TRUE(static_cast<bool>(segmented_striped->get(1)));
    EXPECT_TRUE(static_cast<bool>(f14_striped->get(1)));
    EXPECT_TRUE(static_cast<bool>(read_heavy_striped->get(1)));
    EXPECT_TRUE(static_cast<bool>(f14_production->get(1)));

    // Global operations exercise the lazily-allocated striped mutex storage
    // (flush -> lock_all/unlock_all), which is the member whose initialisation
    // P0-1 changed.
    production->flush();
    striped->flush();
    segmented_striped->flush();
    f14_striped->flush();
    read_heavy_striped->flush();
    f14_production->flush();

    EXPECT_EQ(production->size(), 0u);
    EXPECT_EQ(striped->size(), 0u);
}

TEST(P0ApiSurface, RuntimeStripeCountConstructorWorks) {
    // The two-argument striped constructor formerly routed through the removed
    // factory (which also validated num_stripes > 0).
    auto with_stripes =
        std::make_unique<lru::production_cache<int, std::string>>(256, 8);
    auto striped_explicit =
        std::make_unique<lru::striped_cache<int, std::string>>(256, 4);

    with_stripes->set(1, "one");
    striped_explicit->set(1, "one");
    EXPECT_TRUE(static_cast<bool>(with_stripes->get(1)));
    EXPECT_TRUE(static_cast<bool>(striped_explicit->get(1)));
    with_stripes->flush();
    striped_explicit->flush();

    // num_stripes == 0 must still be rejected (validation moved from the
    // deleted factory into lazy_striped_mutex's constructor).
    EXPECT_THROW((lru::production_cache<int, std::string>(64, 0)),
                 std::invalid_argument);
}

TEST(P0ApiSurface, NonStripedAliasesStillConstruct) {
    // The non-striped branch now uses detail::no_striped_mutex instead of
    // std::tuple<>; verify it still constructs and operates.
    auto single = std::make_unique<lru::cache<int, std::string>>(64);
    auto safe = std::make_unique<lru::safe_cache<int, std::string>>(64);
    single->set(1, "one");
    safe->set(1, "one");
    EXPECT_TRUE(static_cast<bool>(single->get(1)));
    EXPECT_TRUE(static_cast<bool>(safe->get(1)));
    single->flush();
    safe->flush();
}

// ---------------------------------------------------------------------------
// P0-2 (a): set_eviction_predicate must be reachable on every public alias
// ---------------------------------------------------------------------------

TEST(P0ApiSurface, SetEvictionPredicateOnEveryAlias) {
    exercise_alias<lru::cache<int, std::string>>();
    exercise_alias<lru::safe_cache<int, std::string>>();
    exercise_alias<lru::striped_cache<int, std::string>>();
    exercise_alias<lru::production_cache<int, std::string>>();
    exercise_alias<lru::segmented_cache<int, std::string>>();
    exercise_alias<lru::segmented_striped_cache<int, std::string>>();
    exercise_alias<lru::f14_cache<int, std::string>>();
    exercise_alias<lru::f14_striped_cache<int, std::string>>();
    exercise_alias<lru::f14_production_cache<int, std::string>>();
    exercise_alias<lru::lfu_cache<int, std::string>>();
    exercise_alias<lru::safe_lfu_cache<int, std::string>>();
    exercise_alias<lru::segmented_lfu_cache<int, std::string>>();
    exercise_alias<lru::w_tiny_lfu<int, std::string>>();
    exercise_alias<lru::safe_w_tiny_lfu<int, std::string>>();
    exercise_alias<lru::segmented_w_tiny_lfu<int, std::string>>();
    exercise_alias<lru::two_q<int, std::string>>();
    exercise_alias<lru::safe_two_q<int, std::string>>();
    exercise_alias<lru::segmented_two_q<int, std::string>>();
    exercise_alias<lru::fifo_cache<int, std::string>>();
    exercise_alias<lru::safe_fifo_cache<int, std::string>>();
    exercise_alias<lru::segmented_fifo_cache<int, std::string>>();
    exercise_alias<lru::read_heavy_cache<int, std::string>>();
    exercise_alias<lru::read_heavy_striped_cache<int, std::string>>();
    SUCCEED();
}

TEST(P0ApiSurface, EvictionPredicateIsActuallyHonoured) {
    // The predicate must be functional, not merely compilable: a cache with a
    // predicate that rejects everything must not evict protected items.
    auto c = std::make_unique<lru::cache<int, std::string>>(4);
    c->set_defer_promotion(false);
    c->set_eviction_predicate([](const int&, const std::string&) { return false; });
    for (int i = 0; i < 4; ++i) {
        c->set(i, "v");
    }
    ASSERT_EQ(c->size(), 4u);
    // Every resident item is protected -> the insert must be rejected, not
    // satisfied by evicting a protected item.
    c->set(99, "v");
    EXPECT_LE(c->size(), 4u);
    EXPECT_TRUE(static_cast<bool>(c->get(0)));
}

TEST(P0ApiSurface, ShardedEvictionPredicateIsForwardedToShards) {
    auto c = std::make_unique<lru::production_cache<int, std::string>>(256);
    c->set_defer_promotion(false);
    c->set_eviction_predicate([](const int&, const std::string&) { return false; });
    for (int i = 0; i < 8; ++i) {
        c->set(i, "v");
    }
    EXPECT_EQ(c->size(), 8u);
    EXPECT_TRUE(static_cast<bool>(c->get(3)));
}

// ---------------------------------------------------------------------------
// P0-2 (b): const iteration over the MM layer
// ---------------------------------------------------------------------------

TEST(P0ApiSurface, ConstAndNonConstIterationOverSingleShardMm) {
    // For a single-shard MM (mm_lru), iteration must cover the whole cache.
    auto c = std::make_unique<lru::cache<int, std::string>>(64);
    c->set_defer_promotion(false);
    c->set(1, "a");
    c->set(2, "b");
    c->set(3, "c");

    const auto& mm = c->mm();
    std::size_t const_count = 0;
    for (auto it = mm.begin(); it != mm.end(); ++it) {
        ++const_count;
    }
    EXPECT_EQ(const_count, 3u);

    auto& mmm = c->mm();
    std::size_t nonconst_count = 0;
    for (auto it = mmm.begin(); it != mmm.end(); ++it) {
        ++nonconst_count;
    }
    EXPECT_EQ(nonconst_count, 3u);
}

TEST(P0ApiSurface, ConstIterationOverShardedMmIsConsistent) {
    // sharded_mm_lru::begin()/end() iterate the FIRST SHARD ONLY by design
    // (documented at the definition). The P0-2 defect was that the const
    // overloads did not compile at all; this test locks in that both const and
    // non-const iteration compile and are bounded by size().
    auto c = std::make_unique<lru::production_cache<int, std::string>>(256);
    c->set_defer_promotion(false);
    for (int i = 0; i < 64; ++i) {
        c->set(i, "v");
    }

    const auto& mm = c->mm();
    std::size_t const_count = 0;
    for (auto it = mm.begin(); it != mm.end(); ++it) {
        ++const_count;
    }
    auto& mmm = c->mm();
    std::size_t nonconst_count = 0;
    for (auto it = mmm.begin(); it != mmm.end(); ++it) {
        ++nonconst_count;
    }

    EXPECT_EQ(const_count, nonconst_count)
        << "const and non-const iteration must visit the same nodes";
    EXPECT_LE(const_count, c->size())
        << "first-shard iteration cannot exceed the whole-cache size";
}


// ---------------------------------------------------------------------------
// fix.01 P1-38: lru::config + apply_config()
// ---------------------------------------------------------------------------

TEST(P0ApiSurface, UnifiedConfigAppliesAndValidates) {
    // The aggregate config must reach the cache's real setters.
    lru::production_cache<int, std::string> c(512);

    lru::config cfg;
    cfg.max_size = 512;
    cfg.memory_soft_watermark = 0.70;
    cfg.memory_critical_watermark = 0.90;
    cfg.num_stripes = 128;
    cfg.num_shards = 64;
    cfg.defer_promotion = false;
    cfg.fairness = lru::detail::fairness_mode::writer_fair;
    cfg.latency_tracking = true;
    cfg.ttl_cleaner_interval = std::chrono::milliseconds(0);  // don't start a thread

    lru::apply_config(c, cfg);

    EXPECT_FALSE(c.is_defer_promotion_enabled())
        << "apply_config must push defer_promotion";
    EXPECT_EQ(c.get_fairness_mode(), lru::detail::fairness_mode::writer_fair);
    EXPECT_EQ(c.max_size(), 512u);

    // to_string() must mention the values, so it is usable in a log line.
    const std::string text = cfg.to_string();
    EXPECT_NE(text.find("defer_promotion=0"), std::string::npos) << text;
    EXPECT_NE(text.find("stripes=128"), std::string::npos) << text;

    // validate() must reject nonsense instead of accepting it silently.
    lru::config bad = cfg;
    bad.memory_soft_watermark = 0.99;
    bad.memory_critical_watermark = 0.10;  // soft > critical
    EXPECT_THROW(bad.validate(), std::invalid_argument);
    EXPECT_THROW(lru::apply_config(c, bad), std::invalid_argument);

    lru::config zero_stripes = cfg;
    zero_stripes.num_stripes = 0;
    EXPECT_THROW(zero_stripes.validate(), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// fix.01: bulk_get accepts a contiguous key range, like get_multi's span.
// ---------------------------------------------------------------------------

TEST(P0ApiSurface, BulkGetAcceptsContainerRange) {
    auto c = std::make_unique<lru::safe_cache<int, std::string>>(16);
    c->set(1, "one");
    c->set(2, "two");

    const std::vector<int> keys{1, 2, 3};

    // The container form must compile and behave exactly like the iterator
    // pair (it forwards to it).
    auto via_span = c->bulk_get(keys);
    auto via_iters = c->bulk_get(keys.begin(), keys.end());
    ASSERT_EQ(via_span.size(), 3u);
    ASSERT_EQ(via_iters.size(), 3u);
    EXPECT_TRUE(via_span[0].has_value());
    EXPECT_TRUE(via_span[1].has_value());
    EXPECT_FALSE(via_span[2].has_value());  // key 3 absent
    for (std::size_t i = 0; i < via_span.size(); ++i) {
        EXPECT_EQ(via_span[i].has_value(), via_iters[i].has_value());
    }

    // std::array and C arrays are contiguous too.
    const std::array<int, 2> arr{1, 2};
    EXPECT_EQ(c->bulk_get(arr).size(), 2u);
    const int raw[2] = {1, 2};
    EXPECT_EQ(c->bulk_get(std::span<const int>(raw)).size(), 2u);

    // The initializer_list overload still resolves for a braced list.
    EXPECT_EQ(c->bulk_get({1, 2}).size(), 2u);
}

// ---------------------------------------------------------------------------
// fix.01: bulk_try_get — the non-throwing bulk read.
// ---------------------------------------------------------------------------

TEST(P0ApiSurface, BulkTryGetIsNonThrowingOnShutdown) {
    auto c = std::make_unique<lru::safe_cache<int, std::string>>(16);
    c->set(1, "one");
    c->set(2, "two");

    const std::vector<int> keys{1, 2, 3};

    // On a live cache the two bulk reads agree element-for-element: bulk_get
    // already resolves every element through try_get, so a miss is a nullopt
    // in both.
    auto try_res = c->bulk_try_get(keys);
    auto get_res = c->bulk_get(keys);
    ASSERT_EQ(try_res.size(), keys.size());
    ASSERT_EQ(get_res.size(), keys.size());
    for (std::size_t i = 0; i < try_res.size(); ++i) {
        EXPECT_EQ(try_res[i].has_value(), get_res[i].has_value()) << "index " << i;
    }
    EXPECT_TRUE(try_res[0].has_value());
    EXPECT_TRUE(try_res[1].has_value());
    EXPECT_FALSE(try_res[2].has_value());

    // The convenience overloads mirror bulk_get's.
    EXPECT_EQ(c->bulk_try_get({1, 2}).size(), 2u);
    EXPECT_EQ(c->bulk_try_get(std::span<const int>(keys)).size(), 3u);

    // After shutdown this is the point of the name: one nullopt per key and no
    // exception, where bulk_get throws cache_closed_exception.
    c->shutdown();
    const auto after = c->bulk_try_get(keys);
    ASSERT_EQ(after.size(), keys.size());
    for (const auto& entry : after) {
        EXPECT_FALSE(entry.has_value());
    }
    EXPECT_THROW((void)c->bulk_get(keys), lru::cache_closed_exception);
}
