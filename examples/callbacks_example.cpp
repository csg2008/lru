// Unified LRU Cache - Callbacks example
// Demonstrates sync callbacks + deferred async callback execution

#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "../lru.hpp"

int main() {
    using namespace lru;

    std::cout << "=== LRU Cache Callbacks Example ===\n\n";

    cache<int, std::string> c(5);

    // Track the events.
    std::vector<std::string> events;

    // Register a hit callback.
    c.callbacks().on_hit([&](const int& key, const std::string& value) {
        events.push_back("HIT: key=" + std::to_string(key) +
                        ", value=\"" + value + "\"");
    });

    // Register a miss callback.
    c.callbacks().on_miss([&](const int& key) {
        events.push_back("MISS: key=" + std::to_string(key));
    });

    // Register an insert callback.
    c.callbacks().on_insert([&](const int& key, const std::string& value) {
        events.push_back("INSERT: key=" + std::to_string(key) +
                        ", value=\"" + value + "\"");
    });

    // Register an eviction callback.
    c.callbacks().on_evict([&](const int& key, const std::string& value) {
        events.push_back("EVICT: key=" + std::to_string(key) +
                        ", value=\"" + value + "\"");
    });

    std::cout << "Inserting entries...\n";
    c.set(1, "apple");
    c.set(2, "banana");
    c.set(3, "cherry");
    c.set(4, "date");
    c.set(5, "elderberry");

    std::cout << "\nReading entries...\n";
    if (auto v = c.get(1)) (void)v; // hit
    if (auto v = c.get(2)) (void)v; // hit
    if (auto v = c.get(99)) (void)v; // miss

    std::cout << "\nTriggering evictions...\n";
    c.set(6, "fig");   // evicts 3 (LRU)
    c.set(7, "grape"); // evicts 4

    std::cout << "\n--- Synchronous callback log ---\n";
    for (const auto& event : events) {
        std::cout << event << "\n";
    }

    // Statistics.
    std::cout << "\n--- Statistics ---\n";
    std::cout << c.stats_snapshot() << "\n";

    // ====================================================================
    // Deferred callback demo.
    // ====================================================================
    std::cout << "\n=== Deferred callback demo ===\n";
    cache<int, std::string> c2(3);

    std::vector<std::string> deferred_events;
    c2.callbacks().on_hit([&](const int& k, const std::string& v) {
        deferred_events.push_back("(deferred) HIT: " + std::to_string(k) + "=" + v);
    });
    c2.callbacks().on_evict([&](const int& k, const std::string& v) {
        deferred_events.push_back("(deferred) EVICT: " + std::to_string(k) + "=" + v);
    });

    // Use the collect methods to record events inside the critical section.
    // Note: collect_hit / collect_insert store a value pointer (zero copy), so the
    // value passed in must stay alive until flush_pending().
    std::string one = "one";
    std::string two = "two";
    std::string three = "three";
    std::string one_copy = "one";
    c2.callbacks().collect_insert(1, one);
    c2.callbacks().collect_insert(2, two);
    c2.callbacks().collect_insert(3, three);
    c2.callbacks().collect_evict(1, std::move(one_copy)); // evict stores the moved value
    c2.callbacks().collect_hit(2, two);

    std::cout << "pending events before flush: " << c2.callbacks().pending_count() << "\n";

    // Flush outside the critical section so cache operations are not blocked.
    c2.callbacks().flush_pending();
    std::cout << "pending events after flush: " << c2.callbacks().pending_count() << "\n";

    std::cout << "\nDeferred event log:\n";
    for (const auto& e : deferred_events) {
        std::cout << "  " << e << "\n";
    }

    // ====================================================================
    // Custom eviction handler demo.
    // ====================================================================
    std::cout << "\n=== Custom eviction handling ===\n";
    std::vector<std::pair<int, std::string>> evicted_items;

    cache<int, std::string> c3(3);
    c3.callbacks().on_evict([&](const int& key, const std::string& value) {
        evicted_items.emplace_back(key, value);
    });

    c3.set(1, "first");
    c3.set(2, "second");
    c3.set(3, "third");
    c3.set(4, "fourth"); // evicts 1
    c3.set(5, "fifth");  // evicts 2

    std::cout << "Evicted entries:\n";
    for (const auto& [key, value] : evicted_items) {
        std::cout << "  key=" << key << ", value=\"" << value << "\"\n";
    }

    return 0;
}
