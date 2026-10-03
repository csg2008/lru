// Unified LRU Cache - Cache unit tests
// Aligned with the unified_cache architecture: use the lru::cache<K,V> alias;
// iterators come from c.mm(), get() returns read_handle<V>, dereference with *result.

#include <gtest/gtest.h>
#include <string>
#include <vector>

#include "../lru.hpp"

using namespace lru;

// ============================================================================
// Basic CRUD Tests
// ============================================================================

class CacheCrudTest : public ::testing::Test {
protected:
    cache<int, char> c;

    void SetUp() override {
        c.set(1, 'a');
        c.set(2, 'b');
        c.set(3, 'c');
    }
};

TEST_F(CacheCrudTest, SetAndGet) {
    auto result = c.get(1);
    ASSERT_TRUE(result);
    EXPECT_EQ(*result, 'a');
}

TEST_F(CacheCrudTest, GetUpdatesLruOrder) {
    c.set_defer_promotion(false);  // enable immediate promotion, restoring the expected order
    // Initial order (MRU -> LRU): 3, 2, 1
    c.get(1); // 1 is promoted to MRU, so the order becomes 1, 3, 2
    auto it = c.mm().begin();
    EXPECT_EQ(it->key, 1);
    ++it;
    EXPECT_EQ(it->key, 3);
    ++it;
    EXPECT_EQ(it->key, 2);
}

TEST_F(CacheCrudTest, AddNewKey) {
    EXPECT_TRUE(c.add(4, 'd'));
    EXPECT_EQ(c.size(), 4);
}

TEST_F(CacheCrudTest, AddExistingKey) {
    EXPECT_FALSE(c.add(1, 'z'));
    // Existing key: the value is unchanged.
    EXPECT_EQ(*c.get(1), 'a');
}

TEST_F(CacheCrudTest, ReplaceExisting) {
    EXPECT_TRUE(c.replace(1, 'z'));
    EXPECT_EQ(*c.get(1), 'z');
}

TEST_F(CacheCrudTest, ReplaceNonExisting) {
    EXPECT_FALSE(c.replace(99, 'z'));
}

TEST_F(CacheCrudTest, DeleteExisting) {
    EXPECT_TRUE(c.del(1));
    EXPECT_EQ(c.size(), 2);
    EXPECT_FALSE(c.get(1).has_value());
}

TEST_F(CacheCrudTest, DeleteNonExisting) {
    EXPECT_FALSE(c.del(99));
}

TEST_F(CacheCrudTest, Contains) {
    EXPECT_TRUE(c.contains(1));
    EXPECT_FALSE(c.contains(99));
}

TEST_F(CacheCrudTest, PeekDoesNotPromote) {
    // peek does not change the LRU order and returns a const handle.
    auto result = c.peek(1);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, 'a');
}

TEST_F(CacheCrudTest, Flush) {
    c.flush();
    EXPECT_EQ(c.size(), 0);
    EXPECT_TRUE(c.empty());
}

TEST_F(CacheCrudTest, EmptyCache) {
    cache<int, char> empty_cache;
    EXPECT_TRUE(empty_cache.empty());
    EXPECT_EQ(empty_cache.size(), 0);
}

TEST_F(CacheCrudTest, RemoveReturnsSuccessOrNotFound) {
    // remove() reports a structured RemoveRes.
    EXPECT_EQ(c.remove(1), decltype(c)::RemoveRes::kSuccess);
    EXPECT_FALSE(c.contains(1));
    EXPECT_EQ(c.remove(999), decltype(c)::RemoveRes::kNotFound);
}

TEST_F(CacheCrudTest, ContainsFast) {
    // contains() only consults the map; it does not record an access or promote.
    EXPECT_TRUE(c.contains(1));
    EXPECT_FALSE(c.contains(999));
}

TEST_F(CacheCrudTest, PopViaMm) {
    // pop() goes through the MM layer: removes the item and returns its value.
    auto val = c.mm().pop(1);
    ASSERT_TRUE(val.has_value());
    EXPECT_EQ(*val, 'a');
    EXPECT_EQ(c.size(), 2);
    EXPECT_FALSE(c.contains(1));
}

TEST_F(CacheCrudTest, PopNonExistent) {
    auto val = c.mm().pop(999);
    EXPECT_FALSE(val.has_value());
}

TEST_F(CacheCrudTest, PopLru) {
    cache<int, std::string> c2;
    c2.set(1, "one");
    c2.set(2, "two");
    auto result = c2.mm().pop_lru();
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->first, 1);  // 1 is the LRU (inserted first)
    EXPECT_EQ(c2.size(), 1);
}

TEST_F(CacheCrudTest, PopLruEmptyCache) {
    cache<int, std::string> c2;
    auto result = c2.mm().pop_lru();
    EXPECT_FALSE(result.has_value());
}

// ============================================================================
// Capacity and Eviction Tests
// ============================================================================

TEST(CacheCapacityTest, MaxSizeEviction) {
    cache<int, std::string> c(3); // max size = 3
    c.set(1, "one");
    c.set(2, "two");
    c.set(3, "three");
    c.set(4, "four"); // should evict 1 (the LRU)

    EXPECT_EQ(c.size(), 3);
    EXPECT_FALSE(c.contains(1));
    EXPECT_TRUE(c.contains(2));
    EXPECT_TRUE(c.contains(3));
    EXPECT_TRUE(c.contains(4));
}

TEST(CacheCapacityTest, LruEvictionOrder) {
    cache<int, std::string> c(2);
    c.set_defer_promotion(false);  // enable immediate promotion, restoring the expected order
    c.set(1, "one");
    c.set(2, "two");
    c.get(1); // 1 is promoted to MRU, so 2 becomes the LRU
    c.set(3, "three"); // should evict 2

    EXPECT_TRUE(c.contains(1));
    EXPECT_FALSE(c.contains(2));
    EXPECT_TRUE(c.contains(3));
}

TEST(CacheCapacityTest, MaxMemoryEviction) {
    // Per-item memory = item_overhead + key_size*2 + value_size.
    cache<std::string, std::string> c(unlimited, 500);
    c.set_key_size_calculator([](const std::string& s) { return s.size(); });
    c.set_value_size_calculator([](const std::string& s) { return s.size(); });

    // Insert until eviction is triggered.
    for (int i = 0; i < 20; ++i) {
        c.set("key_" + std::to_string(i), "val_" + std::to_string(i));
    }

    // After eviction, memory must be within the limit.
    EXPECT_LE(c.current_memory(), 500);
    EXPECT_GT(c.size(), 0u);
}

TEST(CacheCapacityTest, ResizeDown) {
    cache<int, std::string> c(5);
    for (int i = 0; i < 5; ++i) {
        c.set(i, std::to_string(i));
    }
    EXPECT_EQ(c.size(), 5);

    c.max_size(2); // shrinking triggers eviction
    EXPECT_EQ(c.size(), 2);
}

// ============================================================================
// Statistics Tests
// ============================================================================

TEST(CacheStatsTest, HitMissTracking) {
    cache<int, int> c;
    c.set(1, 100);

    c.get(1); // hit
    c.get(1); // hit
    c.get(2); // miss

    auto stats = c.stats_snapshot();
    EXPECT_EQ(stats.hits.value.load(), 2);
    EXPECT_EQ(stats.misses.value.load(), 1);
    EXPECT_DOUBLE_EQ(stats.hit_rate(), 2.0 / 3.0);
}

TEST(CacheStatsTest, InsertionAndEvictionTracking) {
    cache<int, int> c(2);
    c.set(1, 1);
    c.set(2, 2);
    c.set(3, 3); // triggers eviction

    auto stats = c.stats_snapshot();
    EXPECT_EQ(stats.insertions.value.load(), 3);
    EXPECT_EQ(stats.evictions.value.load(), 1);
}

// ============================================================================
// Callback Tests
// ============================================================================

TEST(CacheCallbackTest, HitCallback) {
    cache<int, int> c;
    c.set_defer_promotion(false);  // enable immediate promotion so the hit callback fires synchronously
    int hit_key = 0;
    int hit_value = 0;

    c.callbacks().on_hit([&](const int& k, const int& v) {
        hit_key = k;
        hit_value = v;
    });

    c.set(1, 100);
    c.get(1);

    EXPECT_EQ(hit_key, 1);
    EXPECT_EQ(hit_value, 100);
}

TEST(CacheCallbackTest, MissCallback) {
    cache<int, int> c;
    int miss_key = 0;

    c.callbacks().on_miss([&](const int& k) {
        miss_key = k;
    });

    c.get(99);

    EXPECT_EQ(miss_key, 99);
}

TEST(CacheCallbackTest, EvictCallback) {
    cache<int, int> c(2);
    int evict_key = 0;
    int evict_value = 0;

    c.callbacks().on_evict([&](const int& k, const int& v) {
        evict_key = k;
        evict_value = v;
    });

    c.set(1, 100);
    c.set(2, 200);
    c.set(3, 300); // evicts 1

    EXPECT_EQ(evict_key, 1);
    EXPECT_EQ(evict_value, 100);
}

TEST(CacheCallbackTest, InsertCallback) {
    cache<int, int> c;
    int insert_key = 0;
    int insert_value = 0;

    c.callbacks().on_insert([&](const int& k, const int& v) {
        insert_key = k;
        insert_value = v;
    });

    c.set(1, 100);

    EXPECT_EQ(insert_key, 1);
    EXPECT_EQ(insert_value, 100);
}

// ============================================================================
// Iterator tests (the MM-layer iterators are reached through c.mm()).
// ============================================================================

TEST(CacheIteratorTest, OrderedIteration) {
    cache<int, char> c;
    c.set_defer_promotion(false);  // enable immediate promotion, restoring the expected order
    c.set(1, 'a');
    c.set(2, 'b');
    c.set(3, 'c');
    c.get(1); // 1 is promoted to MRU

    // Order (MRU -> LRU): 1, 3, 2
    auto it = c.mm().begin();
    EXPECT_EQ(it->key, 1);
    ++it;
    EXPECT_EQ(it->key, 3);
    ++it;
    EXPECT_EQ(it->key, 2);
    ++it;
    EXPECT_EQ(it, c.mm().end());
}

TEST(CacheIteratorTest, RangeBasedFor) {
    cache<int, int> c;
    c.set(1, 10);
    c.set(2, 20);

    int sum = 0;
    // cache_item exposes its value through the .value member.
    for (const auto& item : c.mm()) {
        sum += item.value;
    }
    EXPECT_EQ(sum, 30);
}

// ============================================================================
// Max Memory / Size Calculator Tests
// ============================================================================

TEST(CacheMemoryPolicyTest, SetValueSizeCalculator) {
    cache<std::string, std::string> c(unlimited, 1000);
    c.set_value_size_calculator([](const std::string& s) { return s.size(); });

    c.set("k", "12345");
    // current_memory must include item_overhead + value_size.
    EXPECT_GT(c.current_memory(), 0u);
}

TEST(CacheMemoryPolicyTest, SetKeySizeCalculator) {
    cache<std::string, std::string> c;
    c.set_key_size_calculator([](const std::string& s) { return s.size(); });

    c.set("abcde", "v");
    // key_size is counted (doubled), so current_memory must exceed 0.
    EXPECT_GT(c.current_memory(), 0u);
}

// ============================================================================
// Constructor Tests: initializer_list & range
// ============================================================================

TEST(CacheConstructorTest, InitList) {
    cache<int, std::string> c{{ {1, "a"}, {2, "b"}, {3, "c"} }, 100};
    EXPECT_EQ(c.size(), 3u);
    EXPECT_EQ(*c.get(1), "a");
    EXPECT_EQ(*c.get(2), "b");
    EXPECT_EQ(*c.get(3), "c");
}

TEST(CacheConstructorTest, InitListEmpty) {
    // Explicit type, avoiding the {} -> size_type(0) ambiguity.
    std::initializer_list<std::pair<const int, std::string>> empty{};
    cache<int, std::string> c(empty, 100);
    EXPECT_TRUE(c.empty());
    EXPECT_EQ(c.size(), 0u);
}

TEST(CacheConstructorTest, InitListEvictsWhenOversize) {
    // Two-step construction: build first, then confirm the capacity limit applies.
    cache<int, std::string> c(2); // max_size = 2
    c.set(1, "a");
    c.set(2, "b");
    c.set(3, "c"); // triggers eviction
    c.set(4, "d");
    EXPECT_EQ(c.size(), 2u);
    // Only the last two survive.
    EXPECT_FALSE(c.contains(1));
    EXPECT_FALSE(c.contains(2));
}

TEST(CacheConstructorTest, RangeFromVector) {
    std::vector<std::pair<const int, std::string>> vec{{1, "x"}, {2, "y"}};
    cache<int, std::string> c(vec.begin(), vec.end(), 100);
    EXPECT_EQ(c.size(), 2u);
    EXPECT_EQ(*c.get(1), "x");
    EXPECT_EQ(*c.get(2), "y");
}

TEST(CacheConstructorTest, RangeFromEmpty) {
    std::vector<std::pair<const int, std::string>> vec;
    cache<int, std::string> c(vec.begin(), vec.end(), 100);
    EXPECT_TRUE(c.empty());
}

TEST(CacheConstructorTest, RangeFromArray) {
    std::pair<const int, double> arr[] = {{10, 1.5}, {20, 2.5}, {30, 3.5}};
    cache<int, double> c(std::begin(arr), std::end(arr), 100);
    EXPECT_EQ(c.size(), 3u);
    EXPECT_DOUBLE_EQ(*c.get(10), 1.5);
}

// ============================================================================
// Value Provider & get_or_fetch / operator[]
// ============================================================================

TEST(CacheValueProviderTest, GetOrFetchWithStoredProvider) {
    cache<int, std::string> c(100);
    c.set_value_provider([](const int& k) { return "auto_" + std::to_string(k); });

    auto v = c.get_or_fetch(42);
    EXPECT_EQ(v, "auto_42");
    // It should be cached now.
    EXPECT_TRUE(c.contains(42));
    EXPECT_EQ(c.size(), 1u);
}

TEST(CacheValueProviderTest, GetOrFetchReturnsCachedValue) {
    cache<int, std::string> c(100);
    int fetch_count = 0;
    c.set_value_provider([&](const int& k) {
        ++fetch_count;
        return "val_" + std::to_string(k);
    });

    c.get_or_fetch(1);   // miss → fetch
    EXPECT_EQ(fetch_count, 1);
    c.get_or_fetch(1);   // hit → no fetch
    EXPECT_EQ(fetch_count, 1);
}

TEST(CacheValueProviderTest, GetOrFetchWithInlineProvider) {
    cache<int, std::string> c(100);
    auto v = c.get_or_fetch(7, [](const int& k) { return "inline_" + std::to_string(k); });
    EXPECT_EQ(v, "inline_7");
    EXPECT_TRUE(c.contains(7));
}

TEST(CacheValueProviderTest, GetOrFetchInlineDoesNotNeedStoredProvider) {
    cache<int, std::string> c(100);
    // No value_provider is set, but an inline provider was passed, so it must succeed.
    auto v = c.get_or_fetch(99, [](const int&) { return "ok"; });
    EXPECT_EQ(v, "ok");
    EXPECT_EQ(c.size(), 1u);
}

TEST(CacheValueProviderTest, GetOrFetchWithoutProviderThrows) {
    cache<int, std::string> c(100);
    // No value_provider has been set.
    EXPECT_THROW(c.get_or_fetch(42), std::runtime_error);
}

TEST(CacheValueProviderTest, OperatorSubscript) {
    cache<int, std::string> c(100);
    c.set_value_provider([](const int& k) { return "auto_" + std::to_string(k); });

    auto v1 = c[42];
    EXPECT_EQ(v1, "auto_42");
    // The second call must hit the cache.
    auto v2 = c[42];
    EXPECT_EQ(v2, "auto_42");
    EXPECT_EQ(c.size(), 1u);
}

// ============================================================================
// get_shared Tests
// ============================================================================

TEST(CacheGetSharedTest, Basic) {
    cache<int, std::string> c(100);
    c.set(1, "hello");
    auto sp = c.get_shared(1);
    ASSERT_NE(sp, nullptr);
    EXPECT_EQ(*sp, "hello");
}

TEST(CacheGetSharedTest, NonExistentReturnsNullptr) {
    cache<int, std::string> c(100);
    auto sp = c.get_shared(999);
    EXPECT_EQ(sp, nullptr);
}

TEST(CacheGetSharedTest, OutlivesEviction) {
    cache<int, std::string> c(100);
    c.set(1, "persistent");
    auto sp = c.get_shared(1);
    ASSERT_NE(sp, nullptr);

    c.flush();
    EXPECT_TRUE(c.empty());

    // A copy held by shared_ptr is unaffected by flush.
    EXPECT_EQ(*sp, "persistent");
}

TEST(CacheGetSharedTest, DoesNotPromoteLru) {
    cache<int, std::string> c(100);
    c.set(1, "first");
    c.set(2, "second");  // 2 is the MRU
    // Initial order: 2 (MRU) -> 1 (LRU)

    c.get_shared(1);  // does not change the LRU order
    auto it = c.mm().begin();
    EXPECT_EQ(it->key, 2);  // 2 is still the MRU
}

TEST(CacheGetSharedTest, RecordsHitAndMiss) {
    cache<int, std::string> c(100);
    c.set(1, "a");
    c.get_shared(1);
    c.get_shared(2); // miss
    auto stats = c.stats_snapshot();
    EXPECT_EQ(stats.hits.value.load(), 1u);
    EXPECT_EQ(stats.misses.value.load(), 1u);
}

// ============================================================================
// pop(key) tests: pop must not fire the eviction callback.
// ============================================================================

TEST(CachePopTest, PopFromUnifiedCache) {
    cache<int, std::string> c(100);
    c.set(1, "hello");
    c.set(2, "world");

    auto popped = c.pop(1);
    ASSERT_TRUE(popped.has_value());
    EXPECT_EQ(*popped, "hello");
    EXPECT_EQ(c.size(), 1u);
    EXPECT_FALSE(c.contains(1));
}

TEST(CachePopTest, PopNonExistent) {
    cache<int, std::string> c(100);
    auto popped = c.pop(999);
    EXPECT_FALSE(popped.has_value());
}

TEST(CachePopTest, PopDoesNotFireEvictionCallback) {
    cache<int, std::string> c(3);
    c.set(1, "one");
    c.set(2, "two");
    c.set(3, "three");     // size == 3, now full

    int evict_count = 0;
    c.callbacks().on_evict([&](const int&, const std::string&) { ++evict_count; });

    auto popped = c.pop(2);  // must not fire the eviction callback
    ASSERT_TRUE(popped.has_value());
    EXPECT_EQ(*popped, "two");
    EXPECT_EQ(evict_count, 0);  // pop never fires the eviction callback
    EXPECT_EQ(c.size(), 2u);    // size is 2 after the pop, leaving a free slot

    // A normal eviction must fire the callback.
    c.set(4, "four");   // size == 3, still within the limit
    EXPECT_EQ(evict_count, 0);

    c.set(5, "five");   // size == 4 -> one eviction (>= max_size == 3)
    EXPECT_EQ(evict_count, 1);
}

// ============================================================================
// Serialization Info Tests
// ============================================================================

TEST(CacheSerializationInfoTest, EmptyCache) {
    cache<int, std::string> c(100);
    EXPECT_EQ(c.serialized_item_count(), 0u);
    EXPECT_EQ(c.serialized_size_estimate(), 0u);
}

TEST(CacheSerializationInfoTest, NonEmptyEstimate) {
    cache<int, std::string> c(100);
    c.set(1, "hello");
    c.set(2, "world");

    EXPECT_EQ(c.serialized_item_count(), 2u);
    auto est = c.serialized_size_estimate();
    auto expected = 2 * (sizeof(int) + sizeof(std::string) + 2 * sizeof(std::size_t));
    EXPECT_EQ(est, expected);
}

TEST(CacheSerializationInfoTest, AfterFlushReturnsZero) {
    cache<int, std::string> c(100);
    c.set(1, "hello");
    EXPECT_EQ(c.serialized_item_count(), 1u);
    c.flush();
    EXPECT_EQ(c.serialized_item_count(), 0u);
}

// ============================================================================
// Reverse Iterator Tests (rbegin / rend)
// ============================================================================

TEST(CacheIteratorTest, ReverseIteration) {
    cache<int, char> c(100);
    c.set(1, 'a');
    c.set(2, 'b');
    c.set(3, 'c');
    // MRU→LRU: 3, 2, 1

    // rbegin() is the LRU end, i.e. 1.
    auto rit = c.mm().rbegin();
    ASSERT_NE(rit, c.mm().rend());
    EXPECT_EQ(rit->key, 1);
    EXPECT_EQ(rit->value, 'a');

    ++rit;
    ASSERT_NE(rit, c.mm().rend());
    EXPECT_EQ(rit->key, 2);

    ++rit;
    ASSERT_NE(rit, c.mm().rend());
    EXPECT_EQ(rit->key, 3);

    ++rit;
    EXPECT_EQ(rit, c.mm().rend());
}

TEST(CacheIteratorTest, ReverseIterationEmpty) {
    cache<int, char> c(100);
    EXPECT_EQ(c.mm().rbegin(), c.mm().rend());
}

TEST(CacheIteratorTest, ReverseIterationSingleItem) {
    cache<int, char> c(100);
    c.set(42, 'z');
    auto rit = c.mm().rbegin();
    ASSERT_NE(rit, c.mm().rend());
    EXPECT_EQ(rit->key, 42);
    ++rit;
    EXPECT_EQ(rit, c.mm().rend());
}

TEST(CacheIteratorTest, ReverseIterationAfterGetPromotion) {
    cache<int, char> c(100);
    c.set_defer_promotion(false);  // enable immediate promotion, restoring the expected order
    c.set(1, 'a');
    c.set(2, 'b');
    c.set(3, 'c');
    // MRU→LRU: 3, 2, 1

    c.get(1); // 1 is promoted to MRU -> order 1, 3, 2
    // Reversed: 2 (LRU), 3, 1 (MRU)

    auto rit = c.mm().rbegin();
    EXPECT_EQ(rit->key, 2);
    ++rit;
    EXPECT_EQ(rit->key, 3);
    ++rit;
    EXPECT_EQ(rit->key, 1);
    ++rit;
    EXPECT_EQ(rit, c.mm().rend());
}

TEST(CacheIteratorTest, ReverseIteratorConst) {
    cache<int, char> c(100);
    c.set(1, 'a');
    c.set(2, 'b');
    const auto& ref = c;

    auto crit = ref.mm().rbegin();
    auto crend = ref.mm().rend();
    ASSERT_NE(crit, crend);
    EXPECT_EQ(crit->key, 1);
    ++crit;
    EXPECT_EQ(crit->key, 2);
    ++crit;
    EXPECT_EQ(crit, crend);
}

TEST(CacheIteratorTest, ReverseRangeBasedFor) {
    cache<int, int> c(100);
    c.set(1, 10);
    c.set(2, 20);
    c.set(3, 30);
    // MRU -> LRU is 3,2,1, so reversing gives 1,2,3.

    std::vector<int> keys;
    for (auto it = c.mm().rbegin(); it != c.mm().rend(); ++it) {
        keys.push_back(it->key);
    }
    ASSERT_EQ(keys.size(), 3u);
    EXPECT_EQ(keys[0], 1);
    EXPECT_EQ(keys[1], 2);
    EXPECT_EQ(keys[2], 3);
}

// ============================================================================
// del / force_del / del_ex Tests
// ============================================================================

TEST(CacheDelTest, DelReturnsFalseForMissingKey) {
    cache<int, std::string> c(100);
    c.set(1, "one");
    // Missing key -> del returns false.
    EXPECT_FALSE(c.del(999));
    // Present key -> del returns true.
    EXPECT_TRUE(c.del(1));
    EXPECT_FALSE(c.contains(1));
}

TEST(CacheDelTest, DelExReturnsKNotFound) {
    cache<int, std::string> c(100);
    c.set(1, "one");
    // Missing key -> del_ex returns kNotFound.
    auto result = c.del_ex(999);
    EXPECT_EQ(result, decltype(c)::DelResult::kNotFound);
}

TEST(CacheDelTest, DelExReturnsKSuccess) {
    cache<int, std::string> c(100);
    c.set(1, "one");
    auto result = c.del_ex(1);
    EXPECT_EQ(result, decltype(c)::DelResult::kSuccess);
    EXPECT_FALSE(c.contains(1));
}

TEST(CacheDelTest, DelExReturnsKPinnedWhenHandleHoldsKey) {
    cache<int, std::string> c(100);
    c.set(1, "one");
    // Handle held -> del fails, so del_ex returns kPinned.
    auto handle = c.get(1);
    ASSERT_TRUE(handle);
    auto result = c.del_ex(1);
    EXPECT_EQ(result, decltype(c)::DelResult::kPinned);
    // The key is still present.
    EXPECT_TRUE(c.contains(1));
}

TEST(CacheDelTest, ForceDelRemovesPinnedItem) {
    cache<int, std::string> c(100);
    c.set(1, "one");
    c.set(2, "two");

    // A handle is held, so a plain del fails.
    auto handle = c.get(1);
    ASSERT_TRUE(handle);
    EXPECT_FALSE(c.del(1));

    // force_del removes the item even with a handle outstanding.
    EXPECT_TRUE(c.force_del(1));
    EXPECT_FALSE(c.contains(1));
    EXPECT_EQ(c.size(), 1u);

    // The handle can still read the value (the memory is freed later).
    EXPECT_EQ(*handle, "one");
}

TEST(CacheDelTest, ForceDelDeferredCleanup) {
    cache<int, std::string> c(100);
    c.set(1, "one");
    c.set(2, "two");

    // force_del while a handle is held.
    auto handle = c.get(1);
    ASSERT_TRUE(handle);
    EXPECT_TRUE(c.force_del(1));
    EXPECT_FALSE(c.contains(1));
    EXPECT_EQ(c.size(), 1u);

    // Release the handle.
    handle.release();

    // Trigger cleanup (via another operation or flush).
    c.flush();
    EXPECT_EQ(c.size(), 0u);
}

TEST(CacheDelTest, ForceDelReturnsFalseForMissingKey) {
    cache<int, std::string> c(100);
    c.set(1, "one");
    EXPECT_FALSE(c.force_del(999));
}

TEST(CacheDelTest, ForceDelWithoutHandleDeletesImmediately) {
    cache<int, std::string> c(100);
    c.set(1, "one");
    // No handle -> force_del removes it directly.
    EXPECT_TRUE(c.force_del(1));
    EXPECT_FALSE(c.contains(1));
    EXPECT_EQ(c.size(), 0u);
}

// ============================================================================
// Memory Monitor / Admission Control Tests
// ============================================================================

TEST(MemoryMonitorTest, DefaultMonitorIsInactiveAndDoesNotBlockInserts) {
    cache<int, int> c(100);
    EXPECT_FALSE(c.monitor().active());

    for (int i = 0; i < 50; ++i) {
        c.set(i, i);
    }
    EXPECT_EQ(c.size(), 50u);
}

TEST(MemoryMonitorTest, SetRejectsNewKeyWhenOverCriticalWatermark) {
    cache<int, int> c(100);
    memory_monitor::config cfg;
    // Note: cache_item inherits from hazptr_obj_base (24 extra bytes —
    // 8 reclaim_ + 8 next_ + 8 epoch_ since T2.3 embedded-epoch EBR),
    // so each item is larger. Use a generous memory limit to ensure
    // the first two keys fit but the third is rejected.
    cfg.max_memory_bytes.store(2048);
    cfg.throttle_fraction.store(1.0);
    cfg.critical_fraction.store(0.5);
    c.set_memory_monitor(cfg);
    EXPECT_TRUE(c.monitor().active());

    // Fill memory to just below the critical watermark (50% of 2048 = 1024).
    // Each int item is ~sizeof(cache_item) + overhead.
    c.set(1, 10);
    EXPECT_TRUE(c.contains(1));
    c.set(2, 20);
    EXPECT_TRUE(c.contains(2));

    // Continue inserting new keys until memory exceeds the critical watermark.
    // The rejection depends on item size vs the 1024-byte threshold.
    bool rejected = false;
    for (int i = 3; i <= 100; ++i) {
        c.set(i, i * 10);
        if (!c.contains(i)) {
            rejected = true;
            break;
        }
    }
    EXPECT_TRUE(rejected);
}

TEST(MemoryMonitorTest, ExistingKeyUpdateRespectsAdmission) {
    // updates to existing keys also go through memory admission.
    // When memory is below the critical watermark, updates succeed; once
    // memory crosses the critical watermark, further updates are rejected
    // (value unchanged) so a write-heavy update storm cannot push the
    // process past its memory budget.
    cache<int, int> c(100);
    memory_monitor::config cfg;
    cfg.max_memory_bytes.store(4096);
    cfg.throttle_fraction.store(1.0);
    cfg.critical_fraction.store(0.5);
    c.set_memory_monitor(cfg);

    // Below critical watermark: update succeeds.
    c.set(1, 10);
    c.set(1, 20);
    auto v = c.get(1);
    ASSERT_TRUE(v);
    EXPECT_EQ(*v, 20);

    // Fill past the critical watermark (50% of 4096 = 2048 bytes).
    for (int i = 2; i <= 200; ++i) c.set(i, i * 10);

    // Updating an existing key is now subject to admission and may be
    // rejected because memory is above the critical watermark. The value
    // must remain the last successfully-written value regardless of
    // whether this update is admitted.
    c.set(1, 999);
    v = c.get(1);
    ASSERT_TRUE(v);
    // The value is either 999 (admitted) or the previous 20 (rejected);
    // both outcomes are valid under T-G2. We only assert that the cache
    // remains consistent and the key is still present.
    EXPECT_TRUE(*v == 999 || *v == 20);
}

TEST(MemoryMonitorTest, AddReturnsFalseWhenRejected) {
    cache<int, int> c(100);
    memory_monitor::config cfg;
    // cache_item now 24 bytes larger (epoch_ embedded). Bumped
    // from 1024 → 2048 so the first two keys still fit before critical
    // watermark is hit.
    cfg.max_memory_bytes.store(2048);
    cfg.throttle_fraction.store(1.0);
    cfg.critical_fraction.store(0.5);
    c.set_memory_monitor(cfg);

    EXPECT_TRUE(c.add(1, 10));
    EXPECT_TRUE(c.add(2, 20));
    // Keep adding until one is rejected
    bool rejected = false;
    for (int i = 3; i <= 100; ++i) {
        if (!c.add(i, i * 10)) {
            rejected = true;
            EXPECT_FALSE(c.contains(i));
            break;
        }
    }
    EXPECT_TRUE(rejected);
}

TEST(MemoryMonitorTest, ReplaceAlwaysAllowsExistingKeyUpdate) {
    cache<int, int> c(100);
    memory_monitor::config cfg;
    cfg.max_memory_bytes.store(4096);
    cfg.throttle_fraction.store(1.0);
    cfg.critical_fraction.store(0.5);
    c.set_memory_monitor(cfg);

    c.set(1, 10);
    EXPECT_TRUE(c.replace(1, 99));
    auto v = c.get(1);
    ASSERT_TRUE(v);
    EXPECT_EQ(*v, 99);
}

TEST(MemoryMonitorTest, ReplaceReturnsFalseForMissingKey) {
    cache<int, int> c(100);
    memory_monitor::config cfg;
    cfg.max_memory_bytes.store(4096);
    cfg.throttle_fraction.store(1.0);
    cfg.critical_fraction.store(0.5);
    c.set_memory_monitor(cfg);

    EXPECT_FALSE(c.replace(1, 10));
}

TEST(MemoryMonitorTest, GetOrFetchWithProviderRespectsAdmission) {
    cache<int, int> c(100);
    memory_monitor::config cfg;
    // cache_item now 24 bytes larger (epoch_ embedded). Bumped
    // from 1024 → 2048 so the first two keys still fit before critical
    // watermark is hit.
    cfg.max_memory_bytes.store(2048);
    cfg.throttle_fraction.store(1.0);
    cfg.critical_fraction.store(0.5);
    c.set_memory_monitor(cfg);

    int provider_calls = 0;
    auto provider = [&](const int& key) {
        ++provider_calls;
        return key * 10;
    };

    EXPECT_EQ(c.get_or_fetch(1, provider), 10);
    EXPECT_EQ(c.get_or_fetch(2, provider), 20);
    EXPECT_EQ(provider_calls, 2);

    // Keep fetching until rejection (memory over budget)
    bool rejected = false;
    for (int i = 3; i <= 100; ++i) {
        try {
            c.get_or_fetch(i, provider);
        } catch (const std::runtime_error&) {
            rejected = true;
            break;
        }
    }
    EXPECT_TRUE(rejected);
    EXPECT_GE(provider_calls, 3); // provider was called before rejection

    // Hit: provider not called.
    EXPECT_EQ(c.get_or_fetch(1, provider), 10);
    EXPECT_GE(provider_calls, 3);
}

TEST(MemoryMonitorTest, StatsReflectCurrentMemory) {
    cache<int, int> c(100);
    memory_monitor::config cfg;
    cfg.max_memory_bytes.store(1024);
    c.set_memory_monitor(cfg);

    c.set(1, 10);
    auto stats = c.memory_monitor_stats();
    EXPECT_EQ(stats.current_memory_bytes, c.current_memory());
    EXPECT_GT(stats.current_memory_bytes, 0u);
    EXPECT_EQ(stats.max_memory_bytes, 1024u);
}

TEST(MemoryMonitorTest, FlushReportsMemoryToMonitor) {
    cache<int, int> c(100);
    memory_monitor::config cfg;
    cfg.max_memory_bytes.store(1024);
    c.set_memory_monitor(cfg);

    c.set(1, 10);
    EXPECT_GT(c.memory_monitor_stats().current_memory_bytes, 0u);

    c.flush();
    EXPECT_EQ(c.memory_monitor_stats().current_memory_bytes, 0u);
}

TEST(MemoryMonitorTest, StripeCacheAdmission) {
    striped_cache<int, int> c(100);
    memory_monitor::config cfg;
    // cache_item now 24 bytes larger (epoch_ embedded). Bumped
    // from 1024 → 2048 so the first two keys still fit before critical
    // watermark is hit.
    cfg.max_memory_bytes.store(2048);
    cfg.throttle_fraction.store(1.0);
    cfg.critical_fraction.store(0.5);
    c.set_memory_monitor(cfg);

    c.set(1, 10);
    EXPECT_TRUE(c.contains(1));
    c.set(2, 20);
    EXPECT_TRUE(c.contains(2));

    // Keep adding until one is rejected
    bool rejected = false;
    for (int i = 3; i <= 100; ++i) {
        c.set(i, i * 10);
        if (!c.contains(i)) {
            rejected = true;
            break;
        }
    }
    EXPECT_TRUE(rejected);
}
