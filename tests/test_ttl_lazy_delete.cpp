// SPDX-License-Identifier: MIT
// G22 regression test: G1 fix — TTL lazy deletion on get().
//
// Before the G1 fix, ttl_cache::get() returned std::nullopt for an expired
// entry but left the stale entry in the underlying cache. This caused the
// cache size to grow unbounded when entries were never actively removed
// (only lazily reported as "not found"). The fix (ttl.hpp get() lines
// 284-298) upgrades the read lock to a write lock and calls cache_.del(key)
// to physically remove the expired entry.
//
// This test verifies that after get() returns nullopt for an expired key:
//   1. peek(key) also returns nullopt (entry is gone, not just hidden)
//   2. size() == 0 (entry was physically removed, not merely reported expired)
//
// Without the fix, size() would remain 1 (entry still present in the cache).

#include <gtest/gtest.h>

#include <chrono>
#include <string>
#include <thread>

#include "../lru.hpp"

using namespace lru;
using namespace std::chrono_literals;

// ============================================================================
// G1: Expired entry is lazily deleted by get()
// ============================================================================
TEST(TtlLazyDelete, ExpiredEntryRemovedOnGet) {
    ttl_cache<int, std::string> cache;
    cache.set_with_ttl(1, "value", 100ms);

    // Immediately readable.
    ASSERT_TRUE(cache.get(1).has_value());
    EXPECT_EQ(cache.size(), 1u);

    // Wait for TTL to expire.
    std::this_thread::sleep_for(200ms);

    // get() must return nullopt (expired).
    EXPECT_FALSE(cache.get(1).has_value());

    // G1 key assertion: the expired entry must have been physically removed
    // by get()'s lazy-deletion path. Before the fix, the entry would remain
    // in the underlying cache and size() would still be 1.
    EXPECT_FALSE(cache.peek(1).has_value());
    EXPECT_EQ(cache.size(), 0u);
}

// ============================================================================
// G1: Lazy deletion does not affect non-expired entries
// ============================================================================
TEST(TtlLazyDelete, NonExpiredEntrySurvivesGet) {
    ttl_cache<int, std::string> cache;
    cache.set_with_ttl(1, "short", 100ms);
    cache.set_with_ttl(2, "long", 1s);

    ASSERT_TRUE(cache.get(1).has_value());
    ASSERT_TRUE(cache.get(2).has_value());
    EXPECT_EQ(cache.size(), 2u);

    std::this_thread::sleep_for(200ms);

    // Key 1 expired → get returns nullopt AND removes it.
    EXPECT_FALSE(cache.get(1).has_value());
    // Key 2 still alive → get returns value, entry stays.
    ASSERT_TRUE(cache.get(2).has_value());
    EXPECT_EQ(*cache.get(2), "long");

    EXPECT_FALSE(cache.peek(1).has_value());
    EXPECT_TRUE(cache.peek(2).has_value());
    EXPECT_EQ(cache.size(), 1u);
}

// ============================================================================
// G1: peek() does NOT lazily delete (const, read-only)
// ============================================================================
TEST(TtlLazyDelete, PeekDoesNotDelete) {
    ttl_cache<int, std::string> cache;
    cache.set_with_ttl(1, "value", 100ms);

    std::this_thread::sleep_for(200ms);

    // peek() reports the entry as expired (returns nullopt) but does NOT
    // remove it — only get() performs lazy deletion.
    EXPECT_FALSE(cache.peek(1).has_value());
    EXPECT_EQ(cache.size(), 1u);  // still present

    // Now get() lazily removes it.
    EXPECT_FALSE(cache.get(1).has_value());
    EXPECT_EQ(cache.size(), 0u);  // removed
}

// ============================================================================
// G1: Multiple expired entries are each lazily deleted
// ============================================================================
TEST(TtlLazyDelete, MultipleExpiredEntriesRemoved) {
    ttl_cache<int, std::string> cache;
    cache.set_with_ttl(1, "a", 100ms);
    cache.set_with_ttl(2, "b", 100ms);
    cache.set_with_ttl(3, "c", 100ms);

    EXPECT_EQ(cache.size(), 3u);

    std::this_thread::sleep_for(200ms);

    // Each get() on an expired key removes that key.
    EXPECT_FALSE(cache.get(1).has_value());
    EXPECT_EQ(cache.size(), 2u);
    EXPECT_FALSE(cache.get(2).has_value());
    EXPECT_EQ(cache.size(), 1u);
    EXPECT_FALSE(cache.get(3).has_value());
    EXPECT_EQ(cache.size(), 0u);
}
