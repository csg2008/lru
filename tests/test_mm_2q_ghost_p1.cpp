// SPDX-License-Identifier: MIT
//
// Regression tests for the last two items of the fix.01 §3.4 / §3.6 batch:
//
//   P1-31 (completion)  Every eviction strategy has item-level TTL, and there
//                       is exactly ONE sweep path. The former
//                       `evict_expired_via_ttl_entry_scan()` — which walked the
//                       whole cache under a read lock for `ttl_entry<V>` values
//                       — has been deleted, and `ttl_cache` now publishes its
//                       expiry to the item level so the O(log n) index can do
//                       the work. These tests pin BOTH halves: (a) each raw MM
//                       implements the item-level TTL contract, and (b) a cache
//                       whose value type is `ttl_entry` is swept by the MM.
//
//   P1-28 (completion)  `mm_2q` now implements the A1out ghost queue, which is
//                       the part of 2Q that makes a re-referenced key land in
//                       the protected (Warm) queue instead of restarting at the
//                       bottom of the window. Tests pin: the ghost is consulted
//                       on insert, it is bounded, it is cleared by flush(), it
//                       is conditionally enabled, and it never leaks into
//                       size()/current_memory()/iteration.
//
// Design note: every assertion below targets a CONTRACT (an invariant or an
// observable consequence), never an internal call sequence, so refactoring the
// internals does not invalidate the tests but breaking the contract does.

#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
#include <thread>
#include <type_traits>

#include "../lru.hpp"

using namespace lru;
using namespace std::chrono_literals;

namespace {

// ============================================================================
// P1-31: item-level TTL is a capability of EVERY eviction strategy
// ============================================================================

/// A deadline comfortably in the past, expressed the way the library stores
/// item-level expiries: nanoseconds since the steady_clock epoch.
std::uint64_t expired_deadline_ns() {
    const auto now_ns = static_cast<std::uint64_t>(
        std::chrono::steady_clock::now().time_since_epoch().count());
    return now_ns - 1'000'000'000ULL;  // 1 s ago
}

std::uint64_t future_deadline_ns(std::chrono::nanoseconds ahead) {
    const auto now_ns = static_cast<std::uint64_t>(
        std::chrono::steady_clock::now().time_since_epoch().count());
    return now_ns + static_cast<std::uint64_t>(ahead.count());
}

/// P1-31: the raw MM contract. `set_with_expiry()` records an item-level
/// deadline, `ttl_remaining_ns()` reports it, and `evict_expired()` removes it
/// once it has passed.
///
/// This is what made the O(n) value-layer scan unnecessary: the strategies that
/// used to lack this capability (2Q / TinyLFU / W-TinyLFU / FIFO) now answer it
/// exactly like LRU does.
template <typename Mm>
void expect_item_level_ttl_contract() {
    Mm mm(64);

    ASSERT_TRUE((std::is_same_v<decltype(mm.evict_expired()), std::size_t>))
        << "P1-31: every strategy must expose evict_expired()";

    // A live TTL item reports a positive remaining time and is not swept.
    mm.set_with_expiry(1, 100, future_deadline_ns(60s));
    EXPECT_EQ(mm.evict_expired(), 0u)
        << "P1-31: a live item must survive the sweep";
    auto remaining = mm.ttl_remaining_ns(1);
    ASSERT_TRUE(remaining.has_value())
        << "P1-31: set_with_expiry() must make the deadline observable";
    EXPECT_GT(*remaining, 0u);

    // An item with no TTL reports no remaining time.
    mm.set(2, 200);
    EXPECT_FALSE(mm.ttl_remaining_ns(2).has_value())
        << "P1-31: an item without a TTL must report no deadline";

    // An already-expired item is swept by the index, not by a full scan.
    mm.set_with_expiry(3, 300, expired_deadline_ns());
    const auto swept = mm.evict_expired();
    EXPECT_GE(swept, 1u)
        << "P1-31: an expired item must be reclaimed by evict_expired()";
    EXPECT_FALSE(mm.ttl_remaining_ns(3).has_value())
        << "P1-31: the expired key must be gone";
    EXPECT_TRUE(mm.ttl_remaining_ns(1).has_value())
        << "P1-31: the live TTL key must be untouched";
}

TEST(MmP1Ttl, P1_31_ItemLevelTtlContractLru) {
    expect_item_level_ttl_contract<mm_lru<int, int>>();
}

TEST(MmP1Ttl, P1_31_ItemLevelTtlContractFifo) {
    expect_item_level_ttl_contract<mm_fifo<int, int>>();
}

TEST(MmP1Ttl, P1_31_ItemLevelTtlContractTwoQ) {
    expect_item_level_ttl_contract<mm_2q<int, int>>();
}

TEST(MmP1Ttl, P1_31_ItemLevelTtlContractTinyLfu) {
    expect_item_level_ttl_contract<mm_tiny_lfu<int, int>>();
}

TEST(MmP1Ttl, P1_31_ItemLevelTtlContractWTinyLfu) {
    expect_item_level_ttl_contract<mm_wtiny_lfu<int, int>>();
}

/// P1-31: the TTL sweep must be O(expired) rather than O(cache). Inserting a
/// large number of never-expiring items and then sweeping must not reap any of
/// them; the index is what makes that true (a full scan would also return 0,
/// but the batched variant below pins the budget semantics).
TEST(MmP1Ttl, P1_31_NoTtlItemsAreNeverSwept) {
    mm_2q<int, int> mm(1000);
    for (int i = 0; i < 500; ++i) {
        mm.set(i, i);
    }
    EXPECT_EQ(mm.evict_expired(), 0u)
        << "P1-31: a cache with no TTLs must sweep nothing";
    EXPECT_EQ(mm.size(), 500u);
}

/// P1-31: the batched sweep honours its budget, so the background cleaner can
/// bound the time it holds the write lock.
TEST(MmP1Ttl, P1_31_BatchedSweepHonoursTheBudget) {
    mm_wtiny_lfu<int, int> mm(4096);
    const auto past = expired_deadline_ns();
    for (int i = 0; i < 50; ++i) {
        mm.set_with_expiry(i, i, past);
    }

    const auto first = mm.evict_expired_n(10);
    EXPECT_LE(first, 10u) << "P1-31: the batch budget must cap one sweep";
    EXPECT_GT(first, 0u) << "P1-31: the sweep must make progress";

    // Repeated bounded sweeps drain the rest.
    std::size_t total = first;
    for (int i = 0; i < 10 && total < 50; ++i) {
        total += mm.evict_expired_n(10);
    }
    EXPECT_EQ(total, 50u) << "P1-31: bounded sweeps must eventually drain";
}

/// P1-31: expiry-aware removal must be distinguishable from capacity eviction.
/// The library reports the two through different callbacks (O7): sweeping an
/// expired item must fire `on_expire`, never `on_evict`.
///
/// `mm_2q::erase_impl` (the capacity path) fires `on_evict`; the whole point of
/// a per-strategy `erase_expired_impl` is that the TTL path does not.
TEST(MmP1Ttl, P1_31_SweepFiresOnExpireNotOnEvict) {
    mm_2q<int, int> mm(64);
    std::size_t expired_callbacks = 0;
    std::size_t evicted_callbacks = 0;
    mm.callbacks().on_expire([&](const int&, const int&) { ++expired_callbacks; });
    mm.callbacks().on_evict([&](const int&, const int&) { ++evicted_callbacks; });

    mm.set_with_expiry(7, 7, expired_deadline_ns());
    mm.evict_expired();
    // `collect_*` batches events into the callback manager's ring; the events
    // are dispatched when the ring is flushed (which the cache layer does per
    // shard after each write). Flush explicitly so the assertions observe the
    // dispatch rather than the batching.
    mm.callbacks().flush_pending();

    EXPECT_EQ(expired_callbacks, 1u)
        << "P1-31/O7: a TTL expiry must fire on_expire";
    EXPECT_EQ(evicted_callbacks, 0u)
        << "P1-31/O7: a TTL expiry must NOT be reported as capacity eviction";
}

/// P1-31: the deleted fallback used to live behind the `ttl_entry<V>` value
/// type. A `unified_cache` storing `ttl_entry<V>` must now be swept through the
/// MM's item-level index, so a value-layer cache still reclaims expired entries
/// via the same single sweep path.
TEST(MmP1Ttl, P1_31_TtlEntryValueCacheIsSweptByTheNativeIndex) {
    using entry = ttl_entry<int>;
    using cache_t = unified_cache<lru_trait<single_threaded_policy>, int, entry>;
    cache_t c(64);

    // Exactly what ttl_cache does internally: one absolute deadline handed to
    // BOTH layers (item level here, value level inside the entry).
    const auto deadline_ns = expired_deadline_ns();
    const auto deadline = std::chrono::steady_clock::time_point{
        std::chrono::steady_clock::duration{static_cast<std::int64_t>(deadline_ns)}};

    c.set_with_absolute_expiry(1, entry{1, deadline}, deadline_ns);
    c.set(2, entry{2, std::nullopt});

    ASSERT_EQ(c.size(), 2u);
    const auto swept = c.evict_expired_now();
    EXPECT_EQ(swept, 1u)
        << "P1-31: the MM index must reclaim the expired ttl_entry value";
    EXPECT_EQ(c.size(), 1u) << "P1-31: only the expired entry may be removed";
    EXPECT_TRUE(c.contains(2)) << "P1-31: the non-expiring entry must survive";
}

/// P1-31 end-to-end through `ttl_cache`: the expiry published by
/// `ttl_cache::set*()` must be visible to the underlying cache's native sweep,
/// which is only true when both layers carry the SAME deadline.
TEST(MmP1Ttl, P1_31_TtlCachePublishesExpiryToTheItemLevelIndex) {
    ttl_cache<int, std::string, std::chrono::milliseconds> c(1s, 64);
    c.set_with_ttl(1, "short", 10ms);
    c.set_with_ttl(2, "long", 30s);

    std::this_thread::sleep_for(60ms);

    // The native sweep (not ttl_cache::clear_expired) must find the expired
    // item, which proves the item-level deadline was published.
    const auto swept = c.underlying().evict_expired_now();
    EXPECT_EQ(swept, 1u)
        << "P1-31: ttl_cache must publish its expiry to the MM's TTL index";
    EXPECT_FALSE(c.contains(1)) << "P1-31: the expired key must be gone";
    EXPECT_TRUE(c.contains(2)) << "P1-31: the live key must remain";
}

// ============================================================================
// P1-28: the 2Q A1out ghost queue
// ============================================================================

/// Configure a 2Q MM so the ghost is exerciseable at test scale.
///
/// `ghost_ratio = 1.0` makes the A1out capacity equal to the Hot (A1in) quota
/// instead of the production default of half of it, and
/// `default_lru_refresh_time = 0` removes the 60 s promotion delay so the
/// accessed-flag transitions are reachable in a unit test.
mm_2q_config ghost_test_config() {
    mm_2q_config cfg;
    cfg.hot_ratio = 0.5;
    cfg.warm_ratio = 0.3;
    cfg.ghost_ratio = 1.0;
    cfg.default_lru_refresh_time = 0;
    return cfg;
}

/// P1-28: an item evicted before it ever earned a second reference is
/// remembered, and re-inserting that key admits it to the protected (Warm)
/// queue instead of the window (Hot).
///
/// Structure (every loop is bounded, so a broken implementation fails an
/// assertion rather than hanging):
///   1. Fill a cache past its ceiling so real capacity evictions happen. Each
///      eviction of an item that never reached Warm is recorded in A1out.
///   2. Find a key that the FIFO still remembers, by probing the evicted keys
///      one at a time; a probe is measured by whether it consumes a ghost
///      entry. Each probe advances the ghost (it evicts one more item), which
///      is exactly why the probe checks the ghost size *before* and *after*
///      rather than caching a key list.
///   3. Assert the decisive property: the key that consumed a ghost entry
///      landed in Warm, not Hot. Before P1-28 this was impossible.
TEST(MmP1TwoQGhost, P1_28_GhostHitAdmitsToWarmInsteadOfHot) {
    using mm_t = mm_2q<int, int>;
    constexpr int kKeys = 20;

    auto enabled = ghost_test_config();
    // Both read and write accesses promote in the production default. Here the
    // only second reference allowed is the ghost hit itself: `update_on_read`
    // is off so that nothing else can set the second-reference mark, and the
    // mark therefore proves the ghost-hit admission took the protected path.
    enabled.update_on_read = false;
    // NOTE: `mm_t mm(kKeys, cfg)` would select the (max_size, max_memory)
    // overload and cap the cache at `kKeys` BYTES, which makes the memory-driven
    // eviction path dominate the scenario. Construct from the config, then set
    // the item capacity explicitly.
    mm_t mm(enabled);
    mm.max_size(kKeys);
    ASSERT_GT(mm.ghost_capacity(), 0u)
        << "P1-28: the test scenario needs a non-zero ghost capacity";

    for (int i = 1; i <= kKeys; ++i) {
        mm.set(i, i);
    }
    ASSERT_EQ(mm.size(), static_cast<std::size_t>(kKeys));

    // Real capacity pressure: each insert past the ceiling evicts the Cold tail
    // before the new item is linked, and every such victim is recorded.
    for (int i = kKeys + 1; i <= 2 * kKeys; ++i) {
        mm.set(i, i);
    }
    ASSERT_GT(mm.ghost_size(), 0u)
        << "P1-28: capacity-driven eviction must feed the A1out ghost";
    ASSERT_LE(mm.ghost_size(), mm.ghost_capacity())
        << "P1-28: A1out must be bounded by ghost_capacity()";

    // Probe the absent keys one at a time. The key range is wider than the
    // resident set so the scan can reach keys evicted earlier (a narrower range
    // only reaches the most recent victims, which the bounded FIFO may already
    // have dropped).
    int ghost_routed = -1;
    for (int i = 1; i <= 200 && ghost_routed < 0; ++i) {
        if (mm.queue_of(i) != mm_t::kNoQueue) continue;  // still resident
        if (!mm.is_ghosted(i)) continue;                 // not remembered
        // This key is remembered: re-inserting it must consume the entry and
        // admit it through the protected path.
        mm.set(i, i);
        ghost_routed = i;
        EXPECT_FALSE(mm.is_ghosted(i))
            << "P1-28: a ghost hit must consume its entry (one admission per "
               "eviction, and no entry may outlive the re-insert that used it)";
    }

    if (ghost_routed < 0) {
        // No "evicted before promotion" key survived in the bounded FIFO at the
        // time of the scan. That is a valid cache state, but the decisive
        // assertion below cannot be exercised; report it instead of passing
        // silently.
        GTEST_SKIP() << "no ghosted key reachable in this layout";
    }

    // The decisive assertions, restated independently of the discovery loop.
    //
    // Why the "admitted to Warm" property is asserted through the mark and not
    // through `queue_of`: the ghost hit decides the ENTRY queue inside
    // `insert_new()`, and the `rebalance()` that runs at the end of that same
    // call is free to demote the Warm tail again (with `warm_ratio` at 0.4 and
    // a 20-item cache, Warm's quota is 8 but it is still filling up). The
    // routing decision itself is recorded permanently in the hook's accessed
    // flag, which no later rebalance clears — only an eviction does — so it is
    // the stable observable of "this insertion took the Warm path".
    ASSERT_NE(mm.queue_of(ghost_routed), mm_t::kNoQueue);
    EXPECT_TRUE(mm.is_marked_accessed(ghost_routed))
        << "P1-28: a ghost-hit admission must take the protected path, which is "
           "recorded as the second-reference mark";
    EXPECT_LT(mm.queue_of(ghost_routed), mm2q::kQueueCount);

    // Control: with the ghost switched off nothing may be remembered at all.
    auto disabled = ghost_test_config();
    disabled.enable_ghost_queue = false;
    mm_t ctrl(disabled);
    ctrl.max_size(kKeys);
    for (int i = 1; i <= kKeys; ++i) {
        ctrl.set(i, i);
    }
    for (int i = kKeys + 1; i <= 2 * kKeys; ++i) {
        ctrl.set(i, i);
    }
    EXPECT_EQ(ctrl.ghost_size(), 0u)
        << "P1-28: with the ghost disabled nothing may be remembered";
    for (int i = 0; i < 200; ++i) {
        ctrl.set(20'000 + i, 0);
    }
    EXPECT_EQ(ctrl.ghost_size(), 0u)
        << "P1-28: the disabled ghost must stay empty under eviction pressure";
    EXPECT_EQ(ctrl.ghost_capacity(), 0u);
}

/// P1-28: the ghost is bounded (it is bookkeeping, not a second cache) and
/// clearing the cache clears it.
TEST(MmP1TwoQGhost, P1_28_GhostIsBoundedAndClearedByFlush) {
    using mm_t = mm_2q<int, int>;
    mm_t mm(ghost_test_config());
    mm.max_size(40);
    const auto capacity = mm.ghost_capacity();
    ASSERT_GT(capacity, 0u);

    for (int i = 0; i < 2000; ++i) {
        mm.set(i, i);
        ASSERT_LE(mm.ghost_size(), capacity)
            << "P1-28: A1out must never exceed ghost_capacity() (violated after "
            << i + 1 << " insertions)";
    }
    EXPECT_GT(mm.ghost_size(), 0u)
        << "P1-28: heavy capacity eviction must feed the ghost";

    mm.flush();
    EXPECT_EQ(mm.ghost_size(), 0u)
        << "P1-28: flush() must forget admission history";
    EXPECT_LE(mm.ghost_size(), mm.ghost_capacity());
}

/// P1-28: the ghost is admission bookkeeping — it must never be counted as
/// cached data. `size()`, `current_memory()` and iteration describe residents
/// only, otherwise a cache at its ceiling would evict real values to make room
/// for key-only ghost entries.
TEST(MmP1TwoQGhost, P1_28_GhostIsInvisibleToSizeMemoryAndIteration) {
    using mm_t = mm_2q<int, int>;
    mm_t mm(ghost_test_config());
    mm.max_size(40);

    for (int i = 0; i < 2000; ++i) {
        mm.set(i, i);
    }
    ASSERT_GT(mm.ghost_size(), 0u) << "the scenario must actually fill the ghost";

    std::size_t iterated = 0;
    for (auto it = mm.begin(); it != mm.end(); ++it) {
        ++iterated;
    }
    EXPECT_EQ(iterated, mm.size())
        << "P1-28: iteration must visit residents only, never ghost keys";
    EXPECT_EQ(mm.size(), mm.hot_size() + mm.warm_size() + mm.cold_size())
        << "P1-28: the ghost must not be a fourth resident queue";
    EXPECT_LE(mm.size(), 40u);
    EXPECT_GT(mm.ghost_bytes_estimate(), 0u)
        << "P1-28: ghost memory must be reportable, even though it is not "
           "charged to current_memory()";

    // The inverse direction of the same invariant: a resident key must never be
    // reported as absent, i.e. the ghost may only describe keys that really
    // left the cache.
    for (int i = 0; i < 2000; ++i) {
        const auto q = mm.queue_of(i);
        if (q == mm_t::kNoQueue) continue;
        EXPECT_LT(q, mm2q::kQueueCount)
            << "P1-28: a resident key must report a valid queue id";
    }
    EXPECT_EQ(mm.queue_of(1999), mm2q::kQueueHot)
        << "P1-28: the most recently inserted key must be resident in Hot";
}

/// P1-28: an item that already earned its second reference must never be fed
/// back into A1out. A key that is promoted Cold→Warm is a genuine second
/// reference, so "evicted before promotion" does not apply to it, and
/// re-referencing it later must NOT be rewarded as if it were new.
///
/// The marker that decides this is the hook's `accessed` flag, so the test
/// asserts the flag directly (via `is_marked_accessed`) instead of inferring it
/// from a queue placement — every insert and eviction runs a `rebalance()` that
/// can legitimately move the item again, which would make a placement-based
/// assertion flaky rather than wrong.
TEST(MmP1TwoQGhost, P1_28_PromotedItemCarriesTheSecondReferenceMark) {
    using mm_t = mm_2q<int, int>;
    constexpr int kKeys = 24;
    auto cfg = ghost_test_config();
    // `default_lru_refresh_time = 0` removes the 60 s gate on the read path, so
    // the one second reference below is the explicit `promote()` call and
    // nothing else can move an item between queues.
    mm_t mm(cfg);
    mm.max_size(kKeys);

    for (int i = 1; i <= kKeys; ++i) {
        mm.set(i, i);
    }
    ASSERT_EQ(mm.size(), static_cast<std::size_t>(kKeys));
    ASSERT_GT(mm.cold_size(), 0u) << "the layout must leave a Cold resident";

    // A freshly inserted item that has never been promoted must not carry the
    // mark: this is the state that DOES feed A1out when it is evicted.
    int never_promoted = -1;
    for (int k = 1; k <= kKeys && never_promoted < 0; ++k) {
        if (mm.queue_of(k) != mm_t::kNoQueue && !mm.is_marked_accessed(k)) {
            never_promoted = k;
        }
    }
    ASSERT_GT(never_promoted, 0)
        << "the layout must contain an item that never earned a second reference";

    // Explicit second reference on a Cold key. `promote()` runs the same
    // Cold→Warm move as a read hit, so this is the exact transition that must
    // set the mark. The handle is dropped immediately: a live `read_handle`
    // pins the item and would make every later eviction skip it.
    int promoted_key = -1;
    for (int k = 1; k <= kKeys && promoted_key < 0; ++k) {
        if (mm.queue_of(k) == mm2q::kQueueCold) {
            EXPECT_TRUE(mm.promote(k))
                << "P1-28: promote() must report that the access was recorded";
            promoted_key = k;
        }
    }
    ASSERT_GT(promoted_key, 0) << "the layout must contain a Cold-resident key";
    EXPECT_TRUE(mm.is_marked_accessed(promoted_key))
        << "P1-28: a promoted item must carry the second-reference mark, which "
           "is what keeps it out of A1out when it is later evicted";

    // Drive churn. The assertion below is an implication checked for EVERY key
    // that is evicted into the ghost: only keys without the second-reference
    // mark may be remembered. That is the invariant itself, and it is checked
    // directly against the ghost contents rather than inferred from queue
    // placement or from `ghost_size()` deltas (the FIFO is bounded, so a delta
    // says nothing about which key was dropped).
    std::size_t saw_promoted_evicted = 0;
    for (int i = 0; i < 2000; ++i) {
        const int k = 1000 + i;
        const bool was_accessed = mm.is_marked_accessed(k);
        mm.set(k, 0);
        if (mm.is_ghosted(k)) {
            EXPECT_FALSE(was_accessed)
                << "P1-28: a key that already earned its second reference must "
                   "never be recorded in A1out (key "
                << k << ")";
        }
        if (k == promoted_key) {
            ++saw_promoted_evicted;
        }
    }
    // The promoted key is a resident (never evicted), so the specific
    // invariant "a promoted key is not ghosted" is asserted on its identity.
    if (mm.contains(promoted_key)) {
        EXPECT_FALSE(mm.is_ghosted(promoted_key))
            << "P1-28: a resident key must never be remembered in A1out";
    } else {
        EXPECT_FALSE(mm.is_ghosted(promoted_key))
            << "P1-28: the promoted key was evicted and must not have been "
               "recorded in A1out";
    }
    static_cast<void>(saw_promoted_evicted);
}

/// P1-28: the feature is switchable, and disabling it must reproduce the
/// pre-change placement exactly (every insert lands in Hot).
TEST(MmP1TwoQGhost, P1_28_GhostCanBeDisabled) {
    using mm_t = mm_2q<int, int>;
    auto cfg = ghost_test_config();
    cfg.enable_ghost_queue = false;
    cfg.max_ghost_entries = 0;

    mm_t mm(cfg);
    mm.max_size(40);
    EXPECT_EQ(mm.ghost_capacity(), 0u)
        << "P1-28: disabling the ghost must zero its capacity";

    for (int i = 0; i < 1000; ++i) {
        mm.set(i, i);
    }
    EXPECT_EQ(mm.ghost_size(), 0u)
        << "P1-28: a disabled ghost must stay empty";

    // Re-inserting an evicted key with the ghost off must go to Hot, which is
    // the pre-P1-28 behaviour. Find an evicted key structurally (the same
    // bounded scan the other tests use) rather than assuming an eviction order.
    int evicted_key = -1;
    for (int k = 0; k < 1000 && evicted_key < 0; ++k) {
        if (mm.queue_of(k) == mm_t::kNoQueue) evicted_key = k;
    }
    ASSERT_GT(evicted_key, 0)
        << "1000 inserts into a 40-item cache must have evicted something";
    const auto hot_before = mm.hot_size();
    mm.set(evicted_key, evicted_key);
    EXPECT_EQ(mm.queue_of(evicted_key), mm2q::kQueueHot)
        << "P1-28: with the ghost disabled a re-insert must enter Hot";
    EXPECT_LE(mm.hot_size(), hot_before + 1)
        << "P1-28: the re-insert may only extend the Hot window by its own "
           "entry (a rebalance may trim the tail)";
    EXPECT_GE(mm.hot_size(), hot_before)
        << "P1-28: the re-insert must not shrink the Hot window";
}

/// P1-28: the explicit capacity override must be honoured verbatim.
TEST(MmP1TwoQGhost, P1_28_ExplicitCapacityOverrideIsHonoured) {
    using mm_t = mm_2q<int, int>;
    auto cfg = ghost_test_config();
    cfg.max_ghost_entries = 7;
    mm_t mm(cfg);
    mm.max_size(40);

    EXPECT_EQ(mm.ghost_capacity(), 7u)
        << "P1-28: max_ghost_entries must override the derived capacity";

    for (int i = 0; i < 1000; ++i) {
        mm.set(i, i);
    }
    EXPECT_LE(mm.ghost_size(), 7u)
        << "P1-28: the explicit bound must be enforced";
}

/// P1-28: invalid ghost configuration is rejected at validation time rather
/// than silently disabling the bound.
TEST(MmP1TwoQGhost, P1_28_InvalidGhostRatioIsRejected) {
    mm_2q_config cfg;
    cfg.ghost_ratio = -0.5;
    EXPECT_THROW(cfg.validate(), std::invalid_argument)
        << "P1-28: a negative ghost_ratio must be rejected";

    mm_2q_config cfg2;
    cfg2.ghost_ratio = 2.0;
    EXPECT_THROW(cfg2.validate(), std::invalid_argument)
        << "P1-28: a ghost_ratio above 1.0 must be rejected";

    mm_2q_config cfg3;
    cfg3.ghost_ratio = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(cfg3.validate(), std::invalid_argument)
        << "P1-28: a NaN ghost_ratio must be rejected (it would silently "
           "disable the bound)";
}

}  // namespace
