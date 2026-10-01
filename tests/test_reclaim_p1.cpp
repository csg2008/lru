// P1 regression tests for the reclamation / locking / callback / worker fixes
// implemented in detail/hazptr.hpp, detail/epoch_reclamation.hpp,
// detail/concurrent_hash_table.hpp, detail/foundation.hpp and core.hpp:
//
//   P1-1  shared_spinlock::unlock() wakes only when a waiter exists
//   P1-4  protect_and_reload()/protect_and_reload_with() publish-then-reread
//   P1-7  seqlock::read_begin() never sleeps; read_retry() detects the writer
//   P1-9  the winning retire keeps its deleter (a losing retire cannot clobber it)
//   P1-10 the pending ledger tracks the real backlog instead of collapsing
//   P1-11 the default domains are process-lifetime (never destroyed)
//   P1-12 slot exhaustion uses one sentinel contract in both domains
//   P1-35 kReject dispatches every event exactly once (no loss, no duplicate)
//   P1-36 sharded counter reset() does not discard concurrent updates
//   P1-39 periodic_worker::set_interval() does not run the task spuriously
//
// These are deterministic and fast; they guard the *contracts* that the fixes
// established, which is what prevents the defects from silently returning.

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <string>
#include <thread>
#include <vector>

#include "../lru.hpp"

namespace {

using namespace std::chrono_literals;

// ---------------------------------------------------------------------------
// P1-9: only the winning retire may publish a deleter
// ---------------------------------------------------------------------------

struct DeleterProbe : lru::detail::hazptr_obj_base {
    static std::atomic<int> deleted_by_a;
    static std::atomic<int> deleted_by_b;
};
std::atomic<int> DeleterProbe::deleted_by_a{0};
std::atomic<int> DeleterProbe::deleted_by_b{0};

TEST(P1Reclaim, LosingRetireCannotClobberTheWinnersDeleter) {
    DeleterProbe::deleted_by_a.store(0);
    DeleterProbe::deleted_by_b.store(0);

    auto* obj = new DeleterProbe();
    auto& dom = lru::detail::hazptr_domain::default_domain();

    // First retire wins and installs deleter A.
    dom.retire_obj(obj, [](lru::detail::hazptr_obj_base* p) {
        DeleterProbe::deleted_by_a.fetch_add(1);
        delete static_cast<DeleterProbe*>(p);
    });
    // Second retire must be a no-op — in particular it must NOT overwrite the
    // deleter with B. Before P1-9 the caller wrote reclaim_ before the CAS, so
    // this would destroy the object through B's deleter.
    dom.retire_obj(obj, [](lru::detail::hazptr_obj_base* p) {
        DeleterProbe::deleted_by_b.fetch_add(1);
        delete static_cast<DeleterProbe*>(p);
    });

    dom.try_reclaim();
    EXPECT_EQ(DeleterProbe::deleted_by_a.load(), 1);
    EXPECT_EQ(DeleterProbe::deleted_by_b.load(), 0)
        << "the loser of the retire race overwrote the winner's deleter";
}

TEST(P1Reclaim, EbrDomainHasTheSameDeleterContract) {
    DeleterProbe::deleted_by_a.store(0);
    DeleterProbe::deleted_by_b.store(0);

    auto* obj = new DeleterProbe();
    auto& dom = lru::detail::epoch_domain::default_domain();
    dom.retire_obj(obj, [](lru::detail::hazptr_obj_base* p) {
        DeleterProbe::deleted_by_a.fetch_add(1);
        delete static_cast<DeleterProbe*>(p);
    });
    dom.retire_obj(obj, [](lru::detail::hazptr_obj_base* p) {
        DeleterProbe::deleted_by_b.fetch_add(1);
        delete static_cast<DeleterProbe*>(p);
    });

    dom.try_reclaim();
    EXPECT_EQ(DeleterProbe::deleted_by_a.load(), 1);
    EXPECT_EQ(DeleterProbe::deleted_by_b.load(), 0);
}

// ---------------------------------------------------------------------------
// P1-4: publish-then-reread
// ---------------------------------------------------------------------------

TEST(P1Reclaim, ProtectAndReloadConfirmsThePublishedValue) {
    std::atomic<DeleterProbe*> src{nullptr};
    DeleterProbe a;
    src.store(&a, std::memory_order_release);

    lru::detail::hazptr_holder h;
    ASSERT_TRUE(h.valid());
    EXPECT_EQ(h.protect_and_reload(src), &a);
    h.clear();
}

TEST(P1Reclaim, ProtectAndReloadRetriesUntilTheReloadAgrees) {
    DeleterProbe a;
    DeleterProbe b;
    std::atomic<int> calls{0};

    lru::detail::hazptr_holder h;
    ASSERT_TRUE(h.valid());
    // The reload reports a different value once, then agrees: the API must keep
    // publishing and re-reading rather than returning the stale pointer.
    DeleterProbe* got = h.protect_and_reload_with(&a, [&]() -> DeleterProbe* {
        return calls.fetch_add(1) == 0 ? &b : &a;
    });
    EXPECT_EQ(got, &a);
    // Sequence: publish(&a) -> reload #1 yields &b -> publish(&b) -> reload #2
    // yields &a -> publish(&a) -> reload #3 yields &a => accepted after 3 reloads.
    EXPECT_EQ(calls.load(), 3);
    h.clear();
}

// ---------------------------------------------------------------------------
// P1-7: the seqlock reader must not sleep waiting for a writer
// ---------------------------------------------------------------------------

TEST(P1Reclaim, SeqlockReadBeginDoesNotBlockOnAnActiveWriter) {
    lru::detail::seqlock sl;
    sl.write_lock();  // writer active: read_begin() must still return promptly

    const auto start = std::chrono::steady_clock::now();
    const std::uint32_t seq = sl.read_begin();
    const auto elapsed = std::chrono::steady_clock::now() - start;

    // Before P1-7 this spin+slept; 1ms is a very generous bound that still
    // fails if the microsecond sleep is reintroduced.
    EXPECT_LT(elapsed, std::chrono::milliseconds(1))
        << "read_begin() blocked/slept while a writer held the lock";

    // The sampled sequence must carry the writer claim so read_retry() detects
    // the concurrent write.
    EXPECT_TRUE((seq & lru::detail::seqlock::kWriterBit) != 0);

    sl.write_unlock();
    // After the writer released, the sequence has advanced: read_retry() on the
    // old sample must report "retry".
    EXPECT_TRUE(sl.read_retry(seq));
}

TEST(P1Reclaim, SeqlockSnapshotStaysConsistentUnderAConcurrentWriter) {
    lru::cache_stats stats;
    std::atomic<bool> stop{false};

    // Bounded writer: a fixed number of resets with a short pause. Hammering
    // reset_counters() in a tight loop makes every reader retry forever (each
    // retry re-copies the whole cache_stats), which is correct but takes
    // minutes — this test must stay fast while still exercising the concurrent
    // writer/reader path.
    std::thread writer([&] {
        for (int i = 0; i < 50; ++i) {
            stats.reset_counters();
            std::this_thread::sleep_for(1ms);
        }
        stop.store(true, std::memory_order_release);
    });

    // Readers must observe a self-consistent snapshot (the invariant the
    // seqlock exists to provide), must not hang, and must see values that are
    // in range.
    std::size_t snapshots = 0;
    while (!stop.load(std::memory_order_acquire)) {
        auto snap = stats.consistent_snapshot();
        EXPECT_LE(snap.current_size.load(std::memory_order_relaxed),
                  snap.max_size.load(std::memory_order_relaxed));
        ++snapshots;
    }
    writer.join();
    EXPECT_GT(snapshots, 0u);
}

// ---------------------------------------------------------------------------
// P1-10: the pending ledger tracks the real backlog
// ---------------------------------------------------------------------------

TEST(P1Reclaim, PendingLedgerTracksRetireAndReclaim) {
    auto& dom = lru::detail::hazptr_domain::default_domain();
    dom.try_reclaim();
    const std::size_t before = dom.pending_count();

    // Retire a batch: the ledger must grow (before P1-10 a concurrent push could
    // be discarded and the value could collapse).
    std::vector<DeleterProbe*> objs;
    for (int i = 0; i < 32; ++i) {
        objs.push_back(new DeleterProbe());
        dom.retire_obj(objs.back(), [](lru::detail::hazptr_obj_base* p) {
            delete static_cast<DeleterProbe*>(p);
        });
    }
    dom.flush_tls_buffer();
    EXPECT_GE(dom.pending_count(), before + 32)
        << "push_pending() did not reserve the retired objects";

    const std::size_t reclaimed = dom.try_reclaim();
    EXPECT_EQ(reclaimed, 32u);
    EXPECT_LE(dom.pending_count(), before)
        << "try_reclaim() did not release what it reclaimed";
}

// ---------------------------------------------------------------------------
// P1-11 / P1-12: process-lifetime domains and one sentinel contract
// ---------------------------------------------------------------------------

TEST(P1Reclaim, DomainsAreProcessLifetimeAndShareTheSentinel) {
    // P1-11: the accessor returns a stable reference that is never destroyed, so
    // repeated calls must yield the same object (a leaked, process-lifetime
    // domain rather than a function-local static that a late thread could see
    // being torn down).
    EXPECT_EQ(&lru::detail::hazptr_domain::default_domain(),
              &lru::detail::hazptr_domain::default_domain());
    EXPECT_EQ(&lru::detail::epoch_domain::default_domain(),
              &lru::detail::epoch_domain::default_domain());

    // P1-12: both domains expose the SAME sentinel contract, so a noexcept
    // caller has exactly one degradation path to handle.
    static_assert(lru::detail::hazptr_domain::npos ==
                  lru::detail::epoch_domain::npos);
    EXPECT_FALSE(lru::detail::epoch_domain::default_domain().is_slot_degraded());
}

// ---------------------------------------------------------------------------
// P1-36: reset() must not discard concurrent updates
// ---------------------------------------------------------------------------

TEST(P1Reclaim, ShardedCounterResetDoesNotLoseConcurrentUpdates) {
    lru::sharded_handle_counter counter;
    std::atomic<bool> stop{false};
    std::atomic<long long> live{0};

    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&] {
            while (!stop.load(std::memory_order_relaxed)) {
                counter.fetch_add(1);
                live.fetch_add(1);
                counter.fetch_sub(1);
                live.fetch_sub(1);
            }
        });
    }

    for (int i = 0; i < 500; ++i) {
        counter.reset();
    }

    stop.store(true, std::memory_order_relaxed);
    for (auto& th : threads) th.join();

    // After every thread stopped, the counter must drain back to 0 — the
    // property the teardown path depends on. `store(0)` could leave a stale
    // decrement behind (wrapping to a huge value and hanging the wait forever).
    counter.reset();
    EXPECT_EQ(counter.load(), 0u);
}

// ---------------------------------------------------------------------------
// P1-39: set_interval() must not run the task spuriously
// ---------------------------------------------------------------------------

TEST(P1Reclaim, PeriodicWorkerIntervalChangeDoesNotRunTheTaskEarly) {
    std::atomic<int> runs{0};
    lru::detail::periodic_worker worker([&runs] { runs.fetch_add(1); }, 10s);

    // Reconfiguring the interval must not be interpreted as "the interval
    // elapsed". Before P1-39 each set_interval() woke the wait and let the
    // condition variable's timeout appear satisfied, running the task once per
    // reconfiguration.
    for (int i = 0; i < 20; ++i) {
        worker.set_interval(10s);
    }
    std::this_thread::sleep_for(50ms);
    EXPECT_EQ(runs.load(), 0)
        << "set_interval() caused a spurious task execution";
    worker.stop();

    // Shortening the interval to (almost) zero must take effect promptly.
    std::atomic<int> fast_runs{0};
    lru::detail::periodic_worker fast([&fast_runs] { fast_runs.fetch_add(1); }, 10s);
    fast.set_interval(1ms);
    std::this_thread::sleep_for(150ms);
    fast.stop();
    EXPECT_GE(fast_runs.load(), 1)
        << "shortening the interval did not take effect";
}

// ---------------------------------------------------------------------------
// P1-35: kReject must deliver every event exactly once
// ---------------------------------------------------------------------------

TEST(P1Reclaim, AsyncRejectPolicyDispatchesEachEventExactlyOnce) {
    using mgr_t = lru::callback_manager<int, std::string>;
    mgr_t mgr;

    std::mutex seen_mtx;
    std::vector<int> seen_keys;
    std::vector<std::string> seen_values;
    mgr.on_hit([&](const int& k, const std::string& v) {
        std::lock_guard<std::mutex> lk(seen_mtx);
        seen_keys.push_back(k);
        seen_values.push_back(v);
    });

    // Async mode ON with a queue far too small for the batch => the kReject
    // policy must refuse the WHOLE batch up front and dispatch it synchronously.
    // (Because the early decision leaves the queue untouched, this path is
    // deterministic: nothing is enqueued, so the worker has nothing to do.)
    mgr.set_async_mode(true);
    mgr.set_async_queue_max_size(4);
    mgr.set_async_overflow_policy(mgr_t::overflow_policy::kReject);

    constexpr int kEvents = 64;
    for (int i = 0; i < kEvents; ++i) {
        mgr.collect_hit(i, "v" + std::to_string(i));
    }
    mgr.flush_pending();
    mgr.set_async_mode(false);  // stops the worker and drains synchronously

    std::lock_guard<std::mutex> lk(seen_mtx);
    // Before P1-35 the events already pushed into the queue were moved out of
    // the local batch as they were enqueued, then removed again by the rollback
    // — so their VALUES arrived empty even though the callback still fired. That
    // is why this test checks the payload, not just the invocation count.
    ASSERT_EQ(seen_values.size(), static_cast<std::size_t>(kEvents))
        << "kReject lost or duplicated hit events";
    for (std::size_t i = 0; i < seen_values.size(); ++i) {
        EXPECT_FALSE(seen_values[i].empty())
            << "event " << i << " arrived with a moved-from (empty) value";
    }
    EXPECT_EQ(seen_keys.size(), static_cast<std::size_t>(kEvents));
}

// P1-33 (fix.01 方案 A): a handle may outlive its cache; releasing it must be safe.
//
// These two tests are compiled ONLY when assertions are disabled. In an
// assert-enabled build `~unified_cache()` deliberately fails with
//   "unified_cache destroyed with outstanding read_handles — use
//    shutdown_and_wait() / force_wait_handles() first"
// (cache_trait.hpp:1265), i.e. the library already treats this as a programming
// error, and the test would abort the whole binary rather than report a failure.
//
// The UAF P1-33 fixes is therefore reachable only where that assertion is
// inactive — a release build (NDEBUG), where the 5s bounded wait and the CRITICAL
// message are the only remaining signals. Keeping the tests under `#ifdef NDEBUG`
// means a release CI run exercises the hardening, while the default debug suite
// stays runnable; the reasoning is recorded here so the conditional is not
// mistaken for an oversight.
#ifdef NDEBUG

TEST(P1Reclaim, HandleReleasedAfterCacheDestructionIsSafe) {
    using cache_t = lru::safe_cache<int, std::string>;

    auto* cache = new cache_t(64);
    cache->set(1, "one");
    auto handle = cache->get(1);
    ASSERT_TRUE(handle.has_value());

    // Destroy the cache while the handle is still alive: bounded wait, CRITICAL
    // warning, then destruction proceeds anyway.
    delete cache;

    // Release after the cache is gone — must be a safe no-op: the notifier is
    // process-lifetime, and `cache_alive` tells release() to stop touching
    // cache-owned state (per_cache_stats_).
    handle = lru::read_handle<std::string>{};
    SUCCEED();
}

TEST(P1Reclaim, HandleFromCopyAlsoSurvivesCacheDestruction) {
    using cache_t = lru::safe_cache<int, std::string>;
    auto* cache = new cache_t(64);
    cache->set(7, "seven");

    {
        auto a = cache->get(7);
        ASSERT_TRUE(a.has_value());
        auto b = a;  // copies the optional AND the handle inside it

        delete cache;

        // Both handles destruct HERE, after the cache is gone — that is the
        // release point this test pins.
    }
    SUCCEED();
}

#endif  // NDEBUG

}  // namespace
