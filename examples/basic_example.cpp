// Unified LRU Cache - Basic usage example
// Demonstrates core API, read_handle, pop/pop_lru, remove

#include <iostream>
#include <string>

#include "../lru.hpp"

int main() {
    using namespace lru;

    std::cout << "=== Basic LRU Cache Example ===\n\n";

    // 1. Create a cache with a maximum capacity of 5.
    cache<int, std::string> c(5);

    // 2. Insert some entries.
    std::cout << "Inserting entries...\n";
    c.set(1, "one");
    c.set(2, "two");
    c.set(3, "three");
    c.set(4, "four");
    c.set(5, "five");

    std::cout << "Cache size: " << c.size() << "\n";

    // 3. get() returns a read_handle<V> (no heap allocation, RAII-safe).
    std::cout << "\nReading key 1...\n";
    auto handle = c.get(1);
    if (handle) {
        std::cout << "Value: " << *handle << "\n";
        // The read_handle decRefs on destruction, so an eviction cannot free a value in use.
    }

    // 4. Insert beyond capacity (evicts the LRU).
    std::cout << "\nInserting key 6 (should evict key 2)...\n";
    c.set(6, "six");

    std::cout << "contains(1): " << c.contains(1) << "\n";
    std::cout << "contains(2): " << c.contains(2) << " (evicted)\n";
    std::cout << "contains(6): " << c.contains(6) << "\n";

    // 5. Statistics.
    std::cout << "\nCache statistics:\n";
    std::cout << c.stats_snapshot() << "\n";

    // 6. Iterate (MRU to LRU) via c.mm().begin() / c.mm().end().
    std::cout << "\nCache contents (MRU to LRU):\n";
    for (auto it = c.mm().begin(); it != c.mm().end(); ++it) {
        std::cout << "  " << it->key << " -> " << it->value << "\n";
    }

    // 7. peek: read-only access that leaves the LRU order alone.
    std::cout << "\n=== Peek (read-only access) ===\n";
    auto peeked = c.peek(6);
    if (peeked) {
        std::cout << "Peek key 6: " << *peeked << "\n";
    }

    // 8. get_shared / get_shared_cached: pick one when you need a shared_ptr.
    std::cout << "\n=== get_shared / get_shared_cached ===\n";
    // get_shared(): heap-allocates and copies the value each call; use it when you
    // need independent ownership.
    auto sp = c.get_shared(6);
    if (sp) {
        std::cout << "get_shared(6): " << *sp << " (use_count=" << sp.use_count() << ")\n";
    }
    // get_shared_cached(): caches the shared_ptr in TLS, so repeated keys allocate nothing.
    auto sp2 = c.get_shared_cached(6);
    if (sp2) {
        std::cout << "get_shared_cached(6): " << *sp2 << " (use_count=" << sp2.use_count() << ")\n";
    }

    // 9. add / replace semantics.
    std::cout << "\n=== add / replace semantics ===\n";
    bool added = c.add(7, "seven");      // key absent, insert succeeds
    bool added_dup = c.add(6, "dup");    // key present, insert fails
    bool replaced = c.replace(7, "SEVEN");
    std::cout << "add(7) = " << added << ", add(6, dup) = " << added_dup
              << ", replace(7) = " << replaced << "\n";
    if (auto v = c.get(7)) {
        std::cout << "key 7 now holds: " << *v << "\n";
    }

    // 10. remove (CacheLib-aligned).
    std::cout << "\n=== remove ===\n";
    auto rem_res = c.remove(7);
    std::cout << "remove(7) = " << (rem_res == decltype(c)::RemoveRes::kSuccess ? "Success" : "NotFound") << "\n";
    std::cout << "contains(7) = " << c.contains(7) << " (removed)\n";

    // 11. pop and pop_lru, through the MM layer.
    std::cout << "\n=== pop / pop_lru semantics ===\n";
    c.set(8, "eight");
    c.set(9, "nine");
    auto popped = c.mm().pop(8);
    if (popped) {
        std::cout << "pop(8) = " << *popped << "\n";
    }
    auto popped_lru = c.mm().pop_lru();
    if (popped_lru) {
        std::cout << "pop_lru() = key:" << popped_lru->first << " val:" << popped_lru->second << "\n";
    }

    // 12. A memory-bounded cache.
    std::cout << "\n=== Memory-bounded cache ===\n";
    cache<std::string, std::string> mem_cache(unlimited, 100);
    mem_cache.set_key_size_calculator([](const std::string& s) { return s.size(); });
    mem_cache.set_value_size_calculator([](const std::string& s) { return s.size(); });

    mem_cache.set("short", "x");
    mem_cache.set("medium_key", "medium_val");
    mem_cache.set("very_long_key_here", "very_long_value_here"); // should trigger eviction

    std::cout << "Memory-bounded cache size: " << mem_cache.size() << "\n";
    std::cout << "Current memory: " << mem_cache.current_memory() << "\n";

    // 13. Switching strategy: a 2Q cache.
    std::cout << "\n=== 2Q cache (three queues) ===\n";
    two_q<int, std::string> twoq(4);
    twoq.set(1, "one");
    twoq.set(2, "two");
    twoq.set(3, "three");
    twoq.set(4, "four");
    std::cout << "Hot=" << twoq.mm().hot_size() << " Warm=" << twoq.mm().warm_size()
              << " Cold=" << twoq.mm().cold_size() << "\n";

    // 14. A W-TinyLFU example, showing frequency-aware admission.
    std::cout << "\n=== W-TinyLFU frequency admission ===\n";
    w_tiny_lfu<int, std::string> wlfu(4);
    // Access keys 1 and 2 repeatedly to raise their frequency.
    for (int i = 0; i < 5; ++i) {
        wlfu.set(1, "one");
        wlfu.set(2, "two");
        wlfu.get(1);
        wlfu.get(2);
    }
    wlfu.set(3, "three");
    wlfu.set(4, "four");
    wlfu.set(5, "five");

    std::cout << "W-TinyLFU size: " << wlfu.size() << "\n";
    std::cout << "contains key 1: " << wlfu.contains(1) << " (hot, should survive)\n";
    std::cout << "contains key 2: " << wlfu.contains(2) << " (hot, should survive)\n";
    std::cout << "contains key 5: " << wlfu.contains(5) << " (newcomer)\n";
    std::cout << "W-TinyLFU per-queue sizes: tiny=" << wlfu.mm().tiny_size()
              << " probation=" << wlfu.mm().probation_size()
              << " protection=" << wlfu.mm().protection_size() << "\n";
    std::cout << "W-TinyLFU stats: " << wlfu.stats_snapshot() << "\n";

    return 0;
}
