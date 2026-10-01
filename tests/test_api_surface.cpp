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

/// Instantiates segmented_intrusive_list::clear() (P0-2: void* -> T*).
void touch_segmented_list_clear() {
    using mm_t = lru::mm_lru<int, std::string>;
    using seg_list_t = typename mm_t::segmented_item_list;
    seg_list_t list;
    list.clear();
    EXPECT_TRUE(list.empty());
    EXPECT_EQ(list.size(), 0u);
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

TEST(P0ApiSurface, SegmentedIntrusiveListClearCompilesAndRuns) {
    // P0-2 (c): this function used to be uncompilable and was reachable only
    // through dead code (mm_lru::list_clear(), since deleted). Instantiating it
    // here keeps the fix locked in.
    touch_segmented_list_clear();
}
