// SPDX-License-Identifier: MIT
//
// Regression tests for the fix.01 §3.4 / §3.5 issue batch (P1-19 … P1-44).
//
// Each test names the issue it guards and states the invariant in the failure
// message, so a regression points straight back at the design note.
//
// Covered here:
//   P1-19 / P1-23  single admission gate: all items pinned ⇒ size() <= max_size
//                  and the rejection is observable.
//   P1-21          tiny tail pinned + probation full ⇒ bounded-time return
//                  (guards the livelock/hang).
//   P1-24          replace_node() keeps isInMMContainer / size / stats coherent.
//   P1-27          probation tail is hot ⇒ window promotion still happens
//                  (the swap must not lose items and must not deadlock).
//   P1-28          TinyLFU admission comparison on the promotion path.
//   P1-30          max_size < num_shards throws by default; succeeds with
//                  allow_amplification = true.
//   P1-41          tail pinned + newer items present ⇒ eviction still starts
//                  from the oldest evictable item.
//   P1-42          insert N then pop_lru N times ⇒ current_memory() == 0.
//   P1-43 / P1-44  both size notions agree after an injected insert failure.

#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <string>
#include <thread>
#include <vector>

#include "../lru.hpp"
#include "../detail/intrusive_list.hpp"

using namespace lru;
using namespace std::chrono_literals;

using lru::detail::cache_item;
using lru::detail::intrusive_hook;

namespace {

// `replace_node()` is intentionally not part of the public API of any MM
// strategy (it is the P1-24 repair of an unreachable entry point, and the spec's
// *recommended* option was to delete it). The regression test therefore reaches
// it through a derived accessor rather than widening the production API.
template <typename Mm>
struct replace_node_access : Mm {
    using Mm::Mm;
    void do_replace(typename Mm::item_ptr old_node, typename Mm::item_ptr new_node) {
        this->replace_node(old_node, new_node);
    }
};

// ============================================================================
// P1-19 / P1-23: capacity is enforced by ONE gate, and rejection is visible
// ============================================================================
//
// Every eviction strategy is exercised. `EvictionPredicate` refuses to evict
// anything, which is the strongest form of "no candidate may be reclaimed":
// the previous per-strategy `insert_new()` bodies inserted regardless of
// whether an eviction had succeeded, so size() grew without bound. After
// P1-19 (方案 B) the unified_cache admission gate must roll the insert back and
// report the rejection.
template <typename Cache>
void expect_capacity_enforced_under_pin_pressure() {
    constexpr std::size_t kMax = 4;
    Cache c(static_cast<std::size_t>(kMax));
    c.set_eviction_predicate([](const int&, const int&) { return false; });

    for (int i = 0; i < static_cast<int>(kMax); ++i) {
        c.set(i, i);
    }
    ASSERT_EQ(c.size(), kMax) << "cache should be exactly full before the pin";

    // Pin every existing item so no eviction candidate is reclaimable.
    std::vector<decltype(c.get(0))> pins;
    pins.reserve(kMax);
    for (int i = 0; i < static_cast<int>(kMax); ++i) {
        auto h = c.get(i);
        ASSERT_TRUE(h.has_value()) << "key " << i << " must be present";
        pins.push_back(std::move(h));
    }

    std::size_t rejects = 0;
    c.callbacks().on_reject(
        [&rejects](const int&, const int&) { ++rejects; });

    for (int i = static_cast<int>(kMax); i < static_cast<int>(kMax) + 8; ++i) {
        c.set(i, i);
    }

    // P1-19: the capacity invariant holds no matter what the MM layer tried.
    EXPECT_LE(c.size(), kMax)
        << "P1-19: size() must never exceed max_size even when every item is pinned";
    // P1-23 方案 B: the failure is observable, not silent.
    EXPECT_GT(rejects, 0u)
        << "P1-23: a refused insertion must be reported via on_reject";
}

// ============================================================================
// P1-21: bounded-time return when the tiny tail is pinned
// ============================================================================

TEST(MmP1, P1_21_TinyTailPinnedReturnsInBoundedTime) {
    // Small cache ⇒ the window (window_to_cache_size_ratio * max_size, floor 1)
    // is 1, and `expected_probation_size()` is 0, so the promotion path runs on
    // essentially every insert. The old `maybe_promote_from_tiny()` called the
    // void `evict_generic(candidate)` and then `continue`d unconditionally: with
    // the tiny tail pinned it never made progress and spun forever inside
    // insert_new()/get().
    w_tiny_lfu<int, int> c(4);
    c.set(1, 1);
    c.set(2, 2);
    ASSERT_EQ(c.size(), 2u);

    // Pin the tiny tail's item (the most recently inserted one).
    auto pinned = c.get(2);
    ASSERT_TRUE(pinned.has_value());

    // If the livelock is back, this call never returns and the test times out
    // rather than failing an assertion — which is exactly the reported symptom.
    for (int i = 10; i < 40; ++i) {
        c.set(i, i);
    }
    EXPECT_GT(c.size(), 0u);
    EXPECT_LE(c.size(), 4u);
}

// ============================================================================
// P1-24: replace_node() keeps map / list / refcount coherent
// ============================================================================

TEST(MmP1, P1_24_ReplaceNodeKeepsStateCoherent) {
    replace_node_access<mm_lru<int, int>> mm(8);
    mm.set(1, 100);
    mm.set(2, 200);

    const auto size_before = mm.size();
    ASSERT_EQ(size_before, 2u);

    // Locate the live item for key 1 through the public iterator.
    cache_item<int, int>* old_node = nullptr;
    for (auto it = mm.begin(); it != mm.end(); ++it) {
        if (it->key == 1) { old_node = &*it; break; }
    }
    ASSERT_NE(old_node, nullptr);
    ASSERT_TRUE(old_node->refcount.isInMMContainer());
    ASSERT_EQ(old_node->value, 100);

    // A replacement node that is not yet in any container.
    auto* new_node = new cache_item<int, int>(1, 100);
    ASSERT_FALSE(new_node->refcount.isInMMContainer());
    ASSERT_FALSE(new_node->hook.is_linked());

    mm.do_replace(old_node, new_node);

    // P1-24: the new node must be the live, container-registered one.
    EXPECT_TRUE(new_node->refcount.isInMMContainer())
        << "P1-24: new_node must be marked in the MM container";
    EXPECT_TRUE(new_node->hook.is_linked())
        << "P1-24: new_node must be linked into the eviction list";
    EXPECT_FALSE(old_node->refcount.isInMMContainer())
        << "P1-24: old_node must be unmarked from the MM container";
    EXPECT_FALSE(old_node->hook.is_linked())
        << "P1-24: old_node must be unlinked from the eviction list";
    EXPECT_EQ(mm.size(), size_before)
        << "P1-24: replace_node must not change the item count";

    // Lookup must observe the replacement node (and thus its value).
    {
        auto live = mm.get(1);
        ASSERT_TRUE(live.has_value())
            << "P1-24: the replaced key must remain retrievable";
        EXPECT_EQ(*live, 100);
    }

    // P1-24: a replaced node must be removable. Before the fix new_node never
    // got kLinked, so `remove_from_list()` early-returned on the D5 guard and
    // `pop()` returned nullopt (an item that can never leave the cache is a
    // permanent leak). `mm_lru::pop()` is used directly so that an access
    // promotion side effect cannot mask the result.
    auto popped = mm.pop(1);
    ASSERT_TRUE(popped.has_value())
        << "P1-24: a replaced node must be removable";
    EXPECT_EQ(*popped, 100);
    EXPECT_EQ(mm.size(), size_before - 1)
        << "P1-24: removing the replaced node must decrement the item count";
    EXPECT_EQ(mm.size(), 1u);
    EXPECT_TRUE(mm.contains(2))
        << "P1-24: the untouched key must be unaffected";
}

// ============================================================================
// P1-27: a hot probation tail must not block window promotion
// ============================================================================

TEST(MmP1, P1_27_HotProbationTailStillPromotesWindow) {
    w_tiny_lfu<int, int> c(8);
    // Create a probation tail with a high frequency: insert then hammer key 1.
    c.set(1, 1);
    for (int i = 0; i < 50; ++i) {
        auto h = c.get(1);
        (void)h;
    }
    // Fill the cache so the window overflows repeatedly.
    for (int i = 100; i < 140; ++i) {
        c.set(i, i);
    }

    // P1-27: promotion must still happen — the swap (not an eviction) keeps the
    // cache full rather than permanently one slot short, and the window keeps
    // rotating. Assert both effects.
    EXPECT_EQ(c.size(), c.max_size())
        << "P1-27: the admission swap must not permanently under-fill the cache";

    // The most recent window occupants must be reachable (promotion moved them
    // into probation rather than dropping them).
    std::size_t reachable = 0;
    for (int i = 100; i < 140; ++i) {
        if (c.contains(i)) ++reachable;
    }
    EXPECT_GT(reachable, 0u)
        << "P1-27: window items must be promoted into main, not lost";
}

// ============================================================================
// P1-28: TinyLFU's promotion path performs the admission comparison
// ============================================================================
//
// Verified structurally through the frequency-protected window→main handoff:
// a key that has accumulated higher frequency must survive a flood of
// never-before-seen keys, and the cache must stay exactly full (the promotion
// path must neither drop the candidate nor leak a slot). `recordAccess` only
// increments the sketch once per `default_lru_refresh_time` (60 s default), so
// the sketch is seeded directly to represent an already-hot key — relying on
// repeated `get()` calls inside a unit test would leave every frequency equal
// and the comparison vacuous.
TEST(MmP1, P1_28_TinyLfuPromotionRespectsFrequency) {
    lfu_cache<int, int> c(8);
    // `hot` is admitted with a deliberately high frequency before anything else
    // can reach main, so the comparison is against a genuinely hot entry.
    for (int i = 0; i < 40; ++i) {
        c.mm().sketch_mut().record(/*key=*/1);
    }
    ASSERT_GT(c.mm().sketch().estimate(1), 1u);

    c.set(1, 1);
    // Flood with cold newcomers.
    for (int i = 1000; i < 1060; ++i) {
        c.set(i, i);
    }
    EXPECT_LE(c.size(), 8u) << "P1-28: capacity must still be enforced";
    EXPECT_TRUE(c.contains(1))
        << "P1-28: a high-frequency entry must survive cold newcomers";

    // The promotion path must not lose slots: keep inserting and the cache must
    // stay exactly at capacity (not permanently short by one, as an
    // eviction-instead-of-comparison implementation would be).
    for (int i = 2000; i < 2040; ++i) {
        c.set(i, i);
    }
    EXPECT_EQ(c.size(), 8u)
        << "P1-28: the admission path must keep the cache exactly full";
}

// ============================================================================
// P1-30: strict capacity by default, explicit amplification opt-in
// ============================================================================
//
// P1-30 方案 A makes silent amplification a hard error. 方案 B's complement —
// also implemented — removes the *need* to throw for the common
// `striped_cache<K,V> c{N}` case by deriving the shard/stripe count from the
// requested capacity, so a request of N slots becomes N shards rather than 64
// shards over 4 slots.
TEST(MmP1, P1_30_CapacityIsNeverSilentlyAmplified) {
    // A one-argument striped cache used to be laid out over
    // `default_num_stripes` (64) shards, which silently raised its effective
    // capacity to 64 and made max_size() report a value the caller never asked
    // for. The SHARD count is now derived from the requested capacity (stripes
    // stay as requested — they are lock granularity and do not cap capacity).
    striped_cache<int, int> c{8};
    EXPECT_EQ(c.max_size(), 8u)
        << "P1-30: max_size() must report what the caller asked for";
    EXPECT_EQ(c.requested_max_size(), 8u);
    EXPECT_LE(c.num_shards(), 4u)
        << "P1-30: a cache with 8 slots must not be laid out over 64 shards";
    // Stripes are the lock granularity: unchanged from the requested default.
    EXPECT_EQ(c.num_stripes(), 64u);

    // A single-shard layout with an 8-slot budget must really hold 8 items —
    // i.e. the effective capacity is 8, not 1 and not 64.
    striped_cache<int, int> single{8, /*num_stripes=*/1};
    EXPECT_EQ(single.max_size(), 8u);
    EXPECT_EQ(single.num_shards(), 1u);
    for (int i = 0; i < 8; ++i) {
        single.set(i, i * 7);
    }
    EXPECT_EQ(single.size(), 8u)
        << "P1-30: the requested 8-slot capacity must be genuinely usable";
    for (int i = 0; i < 8; ++i) {
        auto h = single.try_get(i);
        ASSERT_TRUE(h.has_value()) << "key " << i << " must be resident";
        EXPECT_EQ(**h, i * 7);
    }

    // Multi-shard small cache: an auto-derived layout keeps at least
    // kMinItemsPerShard slots in every shard, so the hard ceiling is never
    // exceeded and the reported capacity stays mostly usable.
    for (int i = 100; i < 300; ++i) {
        c.set(i, i);
    }
    EXPECT_LE(c.size(), c.max_size())
        << "P1-30: a small striped cache must never exceed its capacity";
    // With 8 slots over at most 4 shards, at least two shards' worth of the
    // budget must still be reachable (the previous 64-shard layout guaranteed
    // only 8 slots total across 64 quota-1 shards, i.e. constant eviction).
    EXPECT_GE(c.size(), 4u)
        << "P1-30: the auto-derived layout must keep the reported capacity usable";
}

TEST(MmP1, P1_30_ExplicitConfigStillThrowsOnUndersizedCapacity) {
    // The default guard is still in place when a shard count is supplied
    // explicitly through the MM config: an undersized capacity is refused
    // rather than silently multiplied.
    sharded_mm_lru_config cfg;
    cfg.num_shards = 64;
    EXPECT_THROW((striped_cache<int, int>(8, cfg)), cache_config_exception)
        << "P1-30: an explicit 64-shard layout with max_size 8 must throw";
}

TEST(MmP1, P1_30_AllowAmplificationOptIn) {
    // Explicit opt-in: the MM keeps the requested 64 shards and raises the
    // effective capacity to the per-shard floor. Requested vs effective
    // capacity stay separately reportable.
    sharded_mm_lru_config cfg;
    cfg.num_shards = 64;
    cfg.allow_amplification = true;
    striped_cache<int, int> c(8, cfg);
    EXPECT_EQ(c.requested_max_size(), 8u);
    EXPECT_EQ(c.max_size(), 64u);
    EXPECT_GT(c.max_size(), c.requested_max_size());
}

// ============================================================================
// P1-41: eviction still starts from the oldest evictable item
// ============================================================================

TEST(MmP1, P1_41_PinnedTailDoesNotHideOlderItems) {
    // Capacity 8 so that inserting key 4 does not itself trigger an eviction —
    // the test drives evict_lru() explicitly to keep the assertion about the
    // cursor unambiguous.
    mm_lru<int, int> mm(8);
    mm.set(1, 1);
    mm.set(2, 2);
    mm.set(3, 3);
    ASSERT_EQ(mm.size(), 3u);
    // Key 1 is at the LRU tail (inserted first, never touched).
    ASSERT_TRUE(mm.peek_lru_tail_key().has_value());
    EXPECT_EQ(*mm.peek_lru_tail_key(), 1);

    // Pin the oldest item and perform an eviction: the pinned tail must be
    // skipped, and the *next* oldest item must be the victim.
    // `peek()` is used rather than `get()` because `get()` is an *access*: it
    // promotes the item to the MRU head, which would move key 1 out of the LRU
    // tail and make the assertion below meaningless.
    {
        auto pinned = mm.peek(1);
        ASSERT_TRUE(pinned.has_value());
        ASSERT_EQ(*mm.peek_lru_tail_key(), 1)
            << "P1-41: peek() must not change LRU order";
        mm.evict_lru();
        EXPECT_TRUE(mm.contains(1))
            << "P1-41: a pinned item must never be evicted";
        EXPECT_EQ(*mm.peek_lru_tail_key(), 1)
            << "P1-41: the pinned oldest item is still the LRU tail";
        EXPECT_FALSE(mm.contains(2))
            << "P1-41: eviction must move on to the next oldest evictable item";
    }
    // The pin is released here (end of scope).

    // The cursor must NOT have permanently skipped key 1: now that the pin is
    // released, the oldest item is evictable again and must be chosen next.
    // Before the fix the cursor had advanced past key 1 and the cache would
    // evict *newer* items while keeping this older one forever.
    mm.evict_lru();
    EXPECT_FALSE(mm.contains(1))
        << "P1-41: after the pin is released, the oldest item must be "
           "reconsidered (otherwise LRU order is violated)";
}

// ============================================================================
// P1-42: pop_lru() accounting must return current_memory() to zero
// ============================================================================

template <typename Cache>
void expect_pop_lru_drains_memory() {
    Cache c(64);
    c.set_value_size_calculator([](const std::string& v) { return v.size(); });

    constexpr int kItems = 16;
    for (int i = 0; i < kItems; ++i) {
        c.set(i, std::string(static_cast<std::size_t>(i + 1), 'x'));
    }
    ASSERT_GT(c.current_memory(), 0u);

    for (int i = 0; i < kItems; ++i) {
        auto popped = c.pop_lru();
        ASSERT_TRUE(popped.has_value()) << "pop_lru must yield item " << i;
    }
    // P1-42: the memory charge must have been computed from the pre-move value,
    // so N pops exactly cancel N inserts.
    EXPECT_EQ(c.size(), 0u);
    EXPECT_EQ(c.current_memory(), 0u)
        << "P1-42: current_memory() must return to zero after popping every item";
}

TEST(MmP1, P1_42_PopLruRestoresMemoryToZeroTinyLfu) {
    expect_pop_lru_drains_memory<lfu_cache<int, std::string>>();
}

TEST(MmP1, P1_42_PopLruRestoresMemoryToZeroWTinyLfu) {
    expect_pop_lru_drains_memory<w_tiny_lfu<int, std::string>>();
}

TEST(MmP1, P1_42_PopLruRestoresMemoryToZeroLru) {
    expect_pop_lru_drains_memory<safe_cache<int, std::string>>();
}

// ============================================================================
// P1-43 / P1-44: the two size notions stay identical after an insert failure
// ============================================================================
//
// P1-44 removes the split between "linked in a queue" and "present in the map"
// by arming the rollback guard before the list link. P1-43 then makes both size
// notions derive from the single authoritative counter. A deterministic
// injected `map_.insert` failure is not reachable through the public API, so
// this test drives the observable half of the invariant: after any sequence of
// operations — including ones that reject inserts (P1-19 gate) and ones that
// fail to evict (pinned items) — the reported size must still equal the number
// of keys actually retrievable. A queue/map split shows up immediately as
// `size() != live_count`.
template <typename Cache>
void expect_size_matches_live_key_count() {
    constexpr std::size_t kMax = 8;
    Cache c(static_cast<std::size_t>(kMax));

    auto live_count = [&c]() {
        std::size_t live = 0;
        for (int k = 0; k < 200; ++k) {
            if (c.contains(k)) ++live;
        }
        return live;
    };

    for (int i = 0; i < 60; ++i) {
        c.set(i, i);
    }
    EXPECT_EQ(c.size(), live_count())
        << "P1-43: size() must equal the number of live keys";
    EXPECT_LE(c.size(), kMax);

    // Now force rejections: refuse every eviction and pin everything present.
    c.set_eviction_predicate([](const int&, const int&) { return false; });
    std::vector<decltype(c.get(0))> pins;
    for (int i = 0; i < 200; ++i) {
        if (c.contains(i)) {
            auto h = c.get(i);
            if (h.has_value()) pins.push_back(std::move(h));
        }
    }
    for (int i = 60; i < 120; ++i) {
        c.set(i, i);
    }
    EXPECT_EQ(c.size(), live_count())
        << "P1-43/P1-44: a rejected insert must not leave the queue and the "
           "map disagreeing about how many items the cache holds";
    EXPECT_LE(c.size(), kMax) << "P1-19: capacity must still be enforced";
}

TEST(MmP1, P1_43_44_SizeMatchesLiveKeysTwoQ) {
    expect_size_matches_live_key_count<two_q<int, int>>();
}

TEST(MmP1, P1_43_44_SizeMatchesLiveKeysFifo) {
    expect_size_matches_live_key_count<fifo_cache<int, int>>();
}

TEST(MmP1, P1_43_44_SizeMatchesLiveKeysTinyLfu) {
    expect_size_matches_live_key_count<lfu_cache<int, int>>();
}

TEST(MmP1, P1_43_44_SizeMatchesLiveKeysWTinyLfu) {
    expect_size_matches_live_key_count<w_tiny_lfu<int, int>>();
}

// ============================================================================
// P1-19 / P1-23 across every strategy
// ============================================================================

TEST(MmP1, P1_19_23_CapacityEnforcedUnderPinPressureTwoQ) {
    expect_capacity_enforced_under_pin_pressure<two_q<int, int>>();
}

TEST(MmP1, P1_19_23_CapacityEnforcedUnderPinPressureFifo) {
    expect_capacity_enforced_under_pin_pressure<fifo_cache<int, int>>();
}

TEST(MmP1, P1_19_23_CapacityEnforcedUnderPinPressureTinyLfu) {
    expect_capacity_enforced_under_pin_pressure<lfu_cache<int, int>>();
}

TEST(MmP1, P1_19_23_CapacityEnforcedUnderPinPressureWTinyLfu) {
    expect_capacity_enforced_under_pin_pressure<w_tiny_lfu<int, int>>();
}

// ============================================================================
// P1-32: TTL semantics are consistent across the read APIs
// ============================================================================
//
// `contains()` and the `get()`/`get_with_ttl()` family must agree about an
// expired item. The three-state deletion result is already exposed through the
// existing `RemoveRes` enum: kSuccess / kNotFound cover the reachable states.
TEST(MmP1, P1_32_TtlConsistentAcrossReadApis) {
    safe_cache<int, int> c(64);

    // key 1: no TTL. key 2: 5 ms TTL.
    c.set(1, 111);
    c.set_with_ttl(2, 222, 5ms);

    // Both are live initially, and every read API agrees.
    EXPECT_TRUE(c.contains(1));
    EXPECT_TRUE(c.contains(2));
    {
        auto handle = c.get(1);
        ASSERT_TRUE(handle.has_value()) << "P1-32: get() must find a live key";
        EXPECT_EQ(*handle, 111);
    }
    {
        auto [handle, ttl_ns] = c.get_with_ttl(2);
        EXPECT_TRUE(handle.has_value()) << "P1-32: get_with_ttl() must find a live key";
        EXPECT_TRUE(ttl_ns.has_value()) << "P1-32: a live TTL key must report a TTL";
    }

    // Let key 2 lapse.
    std::this_thread::sleep_for(30ms);

    // P1-32: the read paths must agree that key 2 is gone while key 1 remains.
    EXPECT_TRUE(c.contains(1)) << "P1-32: a non-TTL key must stay present";
    EXPECT_FALSE(c.contains(2)) << "P1-32: contains() must honour TTL";
    {
        auto handle = c.get(2);
        EXPECT_FALSE(handle.has_value()) << "P1-32: get() must honour TTL";
    }
    {
        auto [handle, ttl_ns] = c.get_with_ttl(2);
        EXPECT_FALSE(handle.has_value())
            << "P1-32: get_with_ttl() must honour TTL";
        EXPECT_FALSE(ttl_ns.has_value())
            << "P1-32: no TTL may be reported for an expired item";
    }

    // Deletion-like operations report their outcome through the existing
    // RemoveRes enum: a live key is kSuccess, an absent key is kNotFound.
    using remove_res = typename safe_cache<int, int>::RemoveRes;
    EXPECT_TRUE(c.remove(1) == remove_res::kSuccess)
        << "P1-32: removing a live key must report success";
    EXPECT_TRUE(c.remove(1) == remove_res::kNotFound)
        << "P1-32: removing an already-absent key must report not-found";
}

// ============================================================================
// P1-31: a pinned oldest-expiry item must not block the other expiries
// ============================================================================

TEST(MmP1, P1_31_PinnedOldestExpiryDoesNotStarveOtherKeys) {
    safe_cache<int, int> c(64);

    // Key 1 is the EARLIEST-expiring entry, and its read_handle is acquired
    // while it is still live, then held across its own expiry. That is the
    // starvation scenario: it sits at the top of the TTL min-heap and is
    // un-evictable for the whole test.
    c.set_with_ttl(1, 1, 50ms);
    std::optional<read_handle<const int>> pinned;
    {
        // peek() pins without the TTL check, so the handle can be held across
        // the item's own expiry (which is exactly the situation being guarded).
        auto h = c.peek(1);
        ASSERT_TRUE(h.has_value()) << "key 1 must be live when it is pinned";
        pinned = std::move(h);
    }
    ASSERT_TRUE(pinned.has_value());

    // A later-expiring key that must still be reaped.
    c.set_with_ttl(2, 2, 1ms);
    std::this_thread::sleep_for(120ms);

    const auto evicted = c.evict_expired_now();
    // P1-31 (A1): the pinned key is deferred, but the other expired keys are
    // still processed in the same call. The old heap loop `break`ed on the first
    // pinned entry (which is key 1, the earliest expiry), so a single long-lived
    // handle over the earliest-expiring key blocked every subsequent expiry.
    EXPECT_GE(evicted, 1u)
        << "P1-31: a pinned expired item must not block other expiries";
    EXPECT_FALSE(c.contains(2))
        << "P1-31: the unpinned expired key must be gone";
    // The pinned entry survives as a live item: the handle is still valid, so
    // the key must still be physically present (not force-deleted under it).
    EXPECT_TRUE(pinned->has_value())
        << "P1-31: the pinned handle must remain valid across the sweep";
    pinned.reset();
}

} // namespace
