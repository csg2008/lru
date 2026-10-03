// Unified LRU Cache - Thread-safe usage example
// Demonstrates production_cache, striped_cache, read_handle, get_shared_cached

#include <chrono>
#include <cmath>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "../lru.hpp"

int main() {
    using namespace lru;

    std::cout << "=== Thread-Safe LRU Cache Example ===\n\n";

    // ====================================================================
    // 1. production_cache: recommended for multi-threaded, high-concurrency,
    //    read-heavy production workloads. Features: segmented hash table + sharded
    //    LRU + 64-stripe distributed_shared_mutex.
    // ====================================================================
    std::cout << "--- production_cache (recommended for production) ---\n";
    production_cache<int, double> prod_cache{1000};

    // Fill the cache.
    for (int i = 0; i < 100; ++i) {
        prod_cache.set(i, std::sqrt(static_cast<double>(i)));
    }
    std::cout << "Initial size: " << prod_cache.size() << "\n";

    // Concurrent reads with a few writes.
    // thread indices are std::size_t so
    // that they can index the per-thread counter vectors without a
    // -Wsign-conversion diagnostic; key arithmetic stays int.
    using thread_index = std::size_t;
    std::vector<std::thread> threads;
    constexpr thread_index kReaders = 4;
    constexpr thread_index kWriters = 2;
    std::vector<int> read_hits(kReaders, 0);
    std::vector<int> write_count(kWriters, 0);

    for (thread_index t = 0; t < kReaders; ++t) {
        threads.emplace_back([&prod_cache, &read_hits, t]() {
            for (int i = 0; i < 2000; ++i) {
                int key = (i * 7 + static_cast<int>(t) * 13) % 100;
                // get() returns read_handle<V>: zero heap allocation, RAII unpins.
                auto handle = prod_cache.get(key);
                if (handle) {
                    ++read_hits[t];
                    // Read through the handle: *handle or handle.value().
                }
            }
        });
    }

    for (thread_index t = 0; t < kWriters; ++t) {
        threads.emplace_back([&prod_cache, &write_count, t]() {
            for (int i = 0; i < 100; ++i) {
                int key = 100 + static_cast<int>(t) * 100 + i;
                prod_cache.set(key, std::sqrt(static_cast<double>(key)));
                ++write_count[t];
            }
        });
    }

    for (auto& th : threads) th.join();

    int total_hits = 0;
    for (int h : read_hits) total_hits += h;
    int total_writes = 0;
    for (int w : write_count) total_writes += w;
    std::cout << "Read hits: " << total_hits << ", writes: " << total_writes << "\n";
    std::cout << "Final stats: " << prod_cache.stats_snapshot() << "\n\n";

    // ====================================================================
    // 2. read_handle: the recommended way to access a cached value.
    //    - zero heap allocation (unlike get_shared()'s std::make_shared)
    //    - RAII: decRefs on destruction, so an eviction cannot free a value in use
    //    - movable, so it can be passed across functions
    // ====================================================================
    std::cout << "--- read_handle usage ---\n";
    production_cache<int, std::string> str_cache{100};
    str_cache.set(42, "answer");

    if (auto h = str_cache.get(42)) {
        std::cout << "key 42: " << *h << "\n";            // operator* reads the value
        std::cout << "has_value: " << h.has_value() << "\n"; // explicit check
    }

    // A handle can be moved.
    auto make_handle = [&]() -> lru::read_handle<std::string> {
        return str_cache.get(42);
    };
    if (auto h = make_handle()) {
        std::cout << "handle after the move: " << *h << "\n";
    }
    std::cout << "\n";

    // ====================================================================
    // 3. get_shared_cached(): a TLS-cached shared_ptr (fewer heap allocations).
    //    Use it when you need shared_ptr semantics without allocating every call:
    //    repeated keys reuse the cached shared_ptr.
    // ====================================================================
    std::cout << "--- get_shared_cached() ---\n";
    if (auto sp = str_cache.get_shared_cached(42)) {
        std::cout << "shared_ptr value: " << *sp << " (use_count=" << sp.use_count() << ")\n";
    }
    std::cout << "\n";

    // ====================================================================
    // 4. safe_cache vs striped_cache vs production_cache.
    // ====================================================================
    std::cout << "--- thread-safe cache aliases ---\n";
    std::cout << "safe_cache:       one global lock; fine for low concurrency / tests\n";
    std::cout << "striped_cache:    64-stripe locking; fine for moderate concurrency\n";
    std::cout << "production_cache: segmented hash table + sharded LRU + 64 stripes; recommended\n\n";

    // striped_cache demo.
    striped_cache<int, std::string> scache{1000};
    std::cout << "striped_cache stripes: " << scache.num_stripes() << "\n";

    std::vector<std::thread> workers;
    for (int t = 0; t < 8; ++t) {
        workers.emplace_back([&scache, t]() {
            for (int i = 0; i < 100; ++i) {
                int key = t * 100 + i;
                scache.set(key, "val_" + std::to_string(key));
                if (auto v = scache.get(key)) { (void)v; }
            }
        });
    }
    for (auto& w : workers) w.join();
    std::cout << "striped_cache final size: " << scache.size() << "\n";

    // ====================================================================
    // 5. Bulk operations: get_multi / set_multi.
    //    Grouped by stripe to reduce the number of lock acquisitions.
    // ====================================================================
    std::cout << "\n--- Bulk operations ---\n";
    production_cache<int, std::string> batch_cache{1000};

    // Bulk write.
    std::vector<std::pair<int, std::string>> items = {
        {1, "one"}, {2, "two"}, {3, "three"}, {4, "four"}, {5, "five"}
    };
    batch_cache.set_multi(items);

    // Bulk read.
    std::vector<int> keys = {1, 2, 3, 4, 5, 99};
    auto results = batch_cache.get_multi(keys);
    for (size_t i = 0; i < keys.size(); ++i) {
        if (results[i]) {
            std::cout << "  key " << keys[i] << ": " << **results[i] << "\n";
        } else {
            std::cout << "  key " << keys[i] << ": (miss)\n";
        }
    }

    // ====================================================================
    // 6. peek: read-only access that leaves the LRU order alone.
    // ====================================================================
    std::cout << "\n--- peek (read-only access) ---\n";
    if (auto peeked = prod_cache.peek(50)) {
        std::cout << "key 50 peek: " << *peeked << " (no LRU promotion)\n";
    }

    return 0;
}
