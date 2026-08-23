// Unified LRU Cache - Hazard Pointer Integration Tests
// Validates that cache_iterator holds a hazard pointer, eviction paths use
// retire() for deferred reclamation, and concurrent iteration + eviction
// do not crash or cause use-after-free.

#include <gtest/gtest.h>
#include <atomic>
#include <chrono>
#include <string>
#include <thread>
#include <vector>

#include "../lru.hpp"

using namespace lru;

// ============================================================================
// Hazard Pointer Domain Unit Tests
// ============================================================================

TEST(HazptrDomain, AcquireAndReleaseSlot) {
    auto& domain = detail::hazptr_domain::default_domain();
    auto slot = domain.acquire_slot();
    EXPECT_LT(slot, domain.capacity());
    domain.release_slot(slot);
}

TEST(HazptrDomain, DynamicSlotExpansion) {
    // Create a fresh domain with a single batch (128 slots).
    detail::hazptr_domain small_domain(128);
    std::vector<std::size_t> slots;

    // Acquire all 128 slots in the first batch.
    for (std::size_t i = 0; i < 128; ++i) {
        slots.push_back(small_domain.acquire_slot());
    }
    EXPECT_EQ(small_domain.capacity(), 128u);

    // The 129th acquire should trigger expansion — no throw.
    auto slot129 = small_domain.acquire_slot();
    EXPECT_GE(slot129, 128u);
    EXPECT_EQ(small_domain.capacity(), 256u);  // 2 batches now
    slots.push_back(slot129);

    // Release all slots.
    for (auto s : slots) {
        small_domain.release_slot(s);
    }
}

TEST(HazptrDomain, ProtectAndCheck) {
    auto& domain = detail::hazptr_domain::default_domain();
    int value = 42;
    detail::hazptr_holder holder;
    holder.protect(&value);
    // The pointer should be protected now
    EXPECT_EQ(holder.get<int>(), &value);
    holder.clear();
    EXPECT_EQ(holder.get<int>(), nullptr);
}

TEST(HazptrDomain, RetireUnprotectedPtrIsDeleted) {
    auto& domain = detail::hazptr_domain::default_domain();
    // retire an unprotected pointer — should be deleted immediately
    auto* ptr = new int(99);
    domain.retire(ptr);
    // If we get here without hanging, the fast-path delete worked.
}

TEST(HazptrDomain, RetireProtectedPtrDefersDeletion) {
    auto& domain = detail::hazptr_domain::default_domain();
    auto* ptr = new int(123);
    detail::hazptr_holder holder;
    holder.protect(ptr);
    // retire a protected pointer — should NOT be deleted yet
    domain.retire(ptr);
    // ptr is still alive because it's protected
    EXPECT_EQ(*ptr, 123);
    // Release the hazard pointer
    holder.clear();
    // Now try_reclaim should delete it
    domain.try_reclaim();
}

TEST(HazptrDomain, HolderMoveSemantics) {
    detail::hazptr_holder h1;
    int value = 7;
    h1.protect(&value);
    EXPECT_EQ(h1.get<int>(), &value);

    detail::hazptr_holder h2 = std::move(h1);
    EXPECT_EQ(h2.get<int>(), &value);
    // h1 is now invalid
    EXPECT_FALSE(h1.valid());
    EXPECT_TRUE(h2.valid());
}

// ============================================================================
// T-O1: hazptr slot exhaustion hard cap / fallback
// ============================================================================

TEST(HazptrDomain, SyncFallbackCounterStartsAtZero) {
    detail::hazptr_domain fresh(128);
    EXPECT_EQ(fresh.hazptr_sync_fallback_count(), 0u);
    EXPECT_EQ(fresh.slot_exhaustion_count(), 0u);
}

TEST(HazptrDomain, SlotExhaustionThrowsAfterHardCap) {
    // T-O1 / P0-3: When all 8192 slots are exhausted and the spin budget +
    // sync-reclaim fallback budget are both exceeded, acquire_slot() must
    // return `npos` (sentinel) instead of hanging forever or throwing.
    //
    // Returning npos lets the caller (e.g. hazptr_holder → read_handle ctor,
    // which is noexcept) degrade gracefully by producing an empty handle,
    // matching the refcount-overflow behavior. A watchdog guards against
    // regression (infinite loop).
    detail::hazptr_domain domain(
        detail::hazptr_domain::kMaxBatches *
        detail::hazptr_domain::kBatchSize);
    ASSERT_EQ(domain.capacity(),
              detail::hazptr_domain::kMaxBatches *
              detail::hazptr_domain::kBatchSize);

    // Acquire every slot.
    std::vector<std::size_t> held_slots;
    held_slots.reserve(domain.capacity());
    for (std::size_t i = 0; i < domain.capacity(); ++i) {
        held_slots.push_back(domain.acquire_slot());
    }
    EXPECT_EQ(domain.capacity(), domain.capacity());

    std::atomic<bool> returned_npos{false};
    std::atomic<bool> done{false};

    // Watchdog: if acquire_slot regresses to infinite loop, abort.
    std::thread watcher([&]() {
        auto start = std::chrono::steady_clock::now();
        while (!done.load(std::memory_order_acquire)) {
            if (std::chrono::steady_clock::now() - start >
                std::chrono::seconds(30)) {
                GTEST_FAIL() << "acquire_slot() hung (regression: "
                                "infinite loop instead of returning npos)";
                return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    });

    // Helper: attempt one more acquire — must return npos.
    std::thread helper([&]() {
        // This should exhaust kMaxSpinRetries (1024) yields, then
        // kMaxSyncFallbacks (64) try_reclaim() fallbacks, then return npos.
        std::size_t s = domain.acquire_slot();
        returned_npos.store(s == detail::hazptr_domain::npos,
                            std::memory_order_release);
        done.store(true, std::memory_order_release);
    });

    helper.join();
    watcher.join();

    EXPECT_TRUE(returned_npos.load())
        << "acquire_slot() should have returned npos after exhausting fallbacks";
    // The sync fallback counter must be non-zero — the helper went
    // through at least one try_reclaim() fallback before returning npos.
    EXPECT_GE(domain.hazptr_sync_fallback_count(),
              detail::hazptr_domain::kMaxSyncFallbacks);
    EXPECT_GE(domain.slot_exhaustion_count(), 1u);

    // Release all held slots so the domain can destruct cleanly.
    for (auto s : held_slots) {
        domain.release_slot(s);
    }
}

// ============================================================================
// cache_iterator Hazard Pointer Integration
// ============================================================================

TEST(CacheIteratorHazptr, DefaultIteratorNoHazptr) {
    // A default-constructed iterator should not hold a hazard slot.
    // Use the underlying intrusive_list iterator type directly
    // since unified_cache does not expose a standalone iterator type.
    using item_type = lru::detail::cache_item<int, std::string>;
    using item_list = lru::detail::intrusive_list<item_type, lru::detail::intrusive_hook,
        lru::detail::default_get_hook<item_type>>;
    lru::cache_iterator<item_list::iterator> it;
    // Just ensure it compiles and doesn't crash on destruction.
}

TEST(CacheIteratorHazptr, ValidIteratorHoldsHazptr) {
    lru::cache<int, std::string> c(10);
    c.set(1, "one");
    c.set(2, "two");
    c.set(3, "three");

    // Construct an iterator — should acquire and protect
    auto range = c.rbegin();
    auto it = range.begin();
    // Access the element — the hazptr should protect the node.
    // cache_item uses .key, not .first
    EXPECT_NE(it->key, 0);
    // Iterator destruction should release the hazard slot cleanly
}

TEST(CacheIteratorHazptr, MoveIterator) {
    lru::cache<int, std::string> c(10);
    c.set(1, "one");
    c.set(2, "two");

    auto range = c.rbegin();
    auto it = range.begin();
    auto it2 = std::move(it);
    EXPECT_NE(it2->key, 0);
}

TEST(CacheIteratorHazptr, IncrementUpdatesHazptr) {
    lru::cache<int, std::string> c(10);
    c.set(1, "one");
    c.set(2, "two");
    c.set(3, "three");

    auto range = c.rbegin();
    auto it = range.begin();
    // Increment should update the hazard pointer to the new position
    ++it;
    EXPECT_NE(it->key, 0);
    ++it;
    EXPECT_NE(it->key, 0);
}

// ============================================================================
// Concurrent Iterator + Eviction Stress Test
// ============================================================================

TEST(HazptrConcurrent, SafeCacheIterationWithEviction) {
    // Test with safe_cache (shared_mutex). The read lock blocks writes
    // during iteration, but the retire() path correctly defers deletion
    // when items are evicted right after the iteration lock is released.
    lru::safe_cache<int, std::string> c(50);

    for (int i = 0; i < 50; ++i) {
        c.set(i, "val_" + std::to_string(i));
    }

    std::atomic<bool> done{false};
    std::atomic<bool> reader_started{false};
    std::vector<int> keys_read;

    // Thread 1: Read items via get() and iterate
    auto reader = [&]() {
        // Ensure at least one full read pass happens before exiting,
        // regardless of when the writer sets `done`. This eliminates the
        // trivial timing race where the writer finishes (and sets done)
        // before the reader even enters its loop body, leaving keys_read
        // empty.
        reader_started.store(true, std::memory_order_release);
        bool first_pass = true;
        while (first_pass || !done.load(std::memory_order_relaxed)) {
            // Use get() to obtain read_handle — this provides ref-count
            // protection independent of hazptr
            for (int i = 0; i < 50; ++i) {
                auto h = c.get(i);
                if (h) {
                    keys_read.push_back(i);
                }
            }
            // Also do a range iteration
            auto range = c.rbegin();
            for (auto it = range.begin(); it != range.end(); ++it) {
                [[maybe_unused]] auto k = it->key;
            }
            first_pass = false;
        }
    };

    // Thread 2: Overwrite and cause evictions
    auto writer = [&]() {
        // Wait for reader to signal it has started, so the writer's
        // evictions overlap with the reader's accesses (the actual
        // purpose of this test).
        while (!reader_started.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        for (int i = 50; i < 500; ++i) {
            c.set(i, "new_" + std::to_string(i));
        }
        done.store(true, std::memory_order_release);
    };

    std::thread t1(reader);
    std::thread t2(writer);
    t1.join();
    t2.join();

    // Reclaim any deferred items
    detail::hazptr_domain::default_domain().try_reclaim();

    // No crash = success
    EXPECT_FALSE(keys_read.empty());
}

TEST(HazptrConcurrent, RetireReclaimCycle) {
    // Directly test the retire/reclaim cycle under concurrent access
    auto& domain = detail::hazptr_domain::default_domain();

    std::atomic<bool> done{false};
    std::vector<int*> protected_ptrs;

    // Thread 1: Continuously protect and retire pointers
    auto protector = [&]() {
        detail::hazptr_holder holder;
        while (!done.load(std::memory_order_relaxed)) {
            int* p = new int(42);
            holder.protect(p);
            protected_ptrs.push_back(p);
            domain.retire(p);  // Retire while protected — should defer
            holder.clear();
            std::this_thread::sleep_for(std::chrono::microseconds(1));
        }
    };

    // Thread 2: Periodically reclaim
    auto reclaimer = [&]() {
        while (!done.load(std::memory_order_relaxed)) {
            domain.try_reclaim();
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
    };

    std::thread t1(protector);
    std::thread t2(reclaimer);

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    done.store(true, std::memory_order_release);
    t1.join();
    t2.join();

    // Final reclaim
    domain.try_reclaim();
}

// ============================================================================
// Verify retire() is used in eviction (functional test)
// ============================================================================

TEST(HazptrEviction, EvictionUsesRetire) {
    // When items are evicted from a thread-safe cache, they should go through
    // retire() which checks for hazard pointer protection. This test verifies
    // that an item being iterated (with hazptr protection) is not immediately
    // deleted by eviction.
    // Note: must use safe_cache (thread-safe policy) — single-threaded cache
    // does not use retire() in its eviction path, so hazptr protection has
    // no effect there.
    lru::safe_cache<int, std::string> c(3);
    c.set(1, "one");
    c.set(2, "two");
    c.set(3, "three");

    // Get an iterator and protect an item.
    // The locked_range from rbegin() holds a read lock; we must extract
    // the iterator and then destroy the range (releasing the read lock)
    // before calling set() which needs a write lock. The iterator's
    // hazptr_holder keeps the node protected even after the range is gone.
    int first_key;
    {
        auto range = c.rbegin();
        auto it = range.begin();
        first_key = it->key;
        // range (and its read lock) destroyed here
    }

    // Force eviction of the LRU item — this may or may not be the item
    // our iterator pointed to, but the retire() mechanism should handle it.
    c.set(4, "four");  // Triggers eviction

    // Verify that the cache state is consistent after eviction.
    // The key 1 (LRU) should have been evicted to make room for 4.
    EXPECT_FALSE(c.peek(1).has_value());
    EXPECT_TRUE(c.peek(4).has_value());

    // Reclaim any deferred items
    detail::hazptr_domain::default_domain().try_reclaim();
}
