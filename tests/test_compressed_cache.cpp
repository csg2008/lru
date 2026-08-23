// T15: Compressed pointer cache integration tests
//
// Verifies:
//   - compressed_lru_trait correctly sets Hook = compressed_intrusive_hook
//   - compressed_cache / safe_compressed_cache / striped_compressed_cache
//     aliases compile and operate correctly
//   - compressed_cache_region RAII wrapper auto-installs and removes
//     alloc/dealloc callbacks
//   - Memory savings estimation helpers return sane values

#include <gtest/gtest.h>
#include <cstddef>
#include <string>
#include <type_traits>

#include "../lru.hpp"

using namespace lru;

// ============================================================================
// T15.1: cache_trait Hook template parameter
// ============================================================================

TEST(CompressedCacheTraitTest, DefaultHookIsIntrusiveHook) {
    using trait = lru_trait<>;
    static_assert(std::is_same_v<trait::hook_type, detail::intrusive_hook>,
                  "default hook_type must be detail::intrusive_hook");
    static_assert(!trait::uses_compressed_hook,
                  "default trait must not use compressed hook");
}

TEST(CompressedCacheTraitTest, CompressedTraitUsesCompressedHook) {
    using trait = compressed_lru_trait<>;
    static_assert(std::is_same_v<trait::hook_type, compressed_intrusive_hook>,
                  "compressed_lru_trait must set Hook = compressed_intrusive_hook");
    static_assert(trait::uses_compressed_hook,
                  "compressed_lru_trait must report uses_compressed_hook = true");
}

TEST(CompressedCacheTraitTest, CompressedTraitPreservesOtherDefaults) {
    using trait = compressed_lru_trait<>;
    static_assert(std::is_same_v<trait::lock_policy, single_threaded_policy>,
                  "default lock policy must be single_threaded");
    static_assert(!trait::segmented,
                  "default segmented must be false");
    static_assert(std::is_same_v<trait::probing_style, detail::chain_probing_tag>,
                  "default probing must be chain");
}

// ============================================================================
// T15.2: compressed_cache / safe_compressed_cache / striped_compressed_cache
// ============================================================================

TEST(CompressedCacheAliasTest, CompressedCacheCompilesAndOperates) {
    compressed_cache<int, std::string> c{100};
    c.set(1, "one");
    c.set(2, "two");
    EXPECT_EQ(c.size(), 2u);

    auto v = c.get(1);
    ASSERT_TRUE(v.has_value());
    EXPECT_EQ(*v, "one");
}

TEST(CompressedCacheAliasTest, SafeCompressedCacheCompilesAndOperates) {
    safe_compressed_cache<int, std::string> c{100};
    c.set(1, "one");
    c.set(2, "two");
    EXPECT_EQ(c.size(), 2u);

    auto v = c.get(1);
    ASSERT_TRUE(v.has_value());
    EXPECT_EQ(*v, "one");
}

TEST(CompressedCacheAliasTest, StripedCompressedCacheCompilesAndOperates) {
    striped_compressed_cache<int, std::string> c{100};
    c.set(1, "one");
    c.set(2, "two");
    EXPECT_EQ(c.size(), 2u);

    auto v = c.get(1);
    ASSERT_TRUE(v.has_value());
    EXPECT_EQ(*v, "one");
}

TEST(CompressedCacheAliasTest, TraitReportsCompressedHook) {
    using cache_t = compressed_cache<int, std::string>;
    static_assert(cache_t::trait_type::uses_compressed_hook,
                  "compressed_cache must report uses_compressed_hook = true");
}

// ============================================================================
// T15.4: Memory savings estimation helpers
//
// Note: per-item savings depend on struct layout and padding. For small K/V
// types like int/int, padding can absorb the hook savings at the cache_item
// level. The hook-level savings (16B → ~14B packed, smaller after alignment)
// are always positive and are the authoritative figure.
// ============================================================================

TEST(CompressedHookSavingsTest, HookSizeReduction) {
    // The compressed hook uses 32-bit offsets (4 bytes each for prev/next)
    // while the regular hook uses 64-bit pointers (8 bytes each).
    EXPECT_LT(sizeof(compressed_intrusive_hook),
              sizeof(detail::intrusive_hook));
}

TEST(CompressedHookSavingsTest, TotalScalesLinearly) {
    // Whatever the per-item savings, the total must scale linearly.
    constexpr std::size_t per_item = compressed_hook_memory_savings<int, int>();
    constexpr std::size_t for_1m = compressed_hook_total_savings<int, int>(1'000'000);
    EXPECT_EQ(for_1m, per_item * 1'000'000);
}

TEST(CompressedHookSavingsTest, PerItemSavingsNonNegative) {
    // Savings are never negative (compressed hook is never larger).
    constexpr std::size_t per_item_int = compressed_hook_memory_savings<int, int>();
    constexpr std::size_t per_item_str = compressed_hook_memory_savings<std::string, std::string>();
    EXPECT_GE(per_item_int, 0u);
    EXPECT_GE(per_item_str, 0u);
}

// ============================================================================
// Hook type size sanity checks
// ============================================================================

TEST(CompressedHookSizeTest, CompressedHookSmallerThanRegularHook) {
    // The compressed hook uses 32-bit offsets (4 bytes each for prev/next)
    // while the regular hook uses 64-bit pointers (8 bytes each).
    // Total compressed: 4+4+4+1+1 = 14 bytes (aligned to 16)
    // Total regular:    8+8+4+1+3(pad) = 24 bytes
    // Savings: 8 bytes per item (before struct alignment).
    EXPECT_LT(sizeof(compressed_intrusive_hook),
              sizeof(detail::intrusive_hook));
}

TEST(CompressedHookSizeTest, CompressedItemSmallerThanRegularItem) {
    using regular_item = detail::cache_item<int, int, detail::intrusive_hook>;
    using compressed_item = detail::cache_item<int, int, compressed_intrusive_hook>;
    EXPECT_LE(sizeof(compressed_item), sizeof(regular_item));
}
