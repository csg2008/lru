// Unified LRU Cache - TTL example

#include <chrono>
#include <iostream>
#include <thread>

#include "../lru.hpp"

int main() {
    using namespace lru;
    using namespace std::chrono_literals;

    std::cout << "=== TTL Cache Example ===\n\n";

    // 1. Cache with 2-second default TTL
    ttl_cache<int, std::string> cache(2s, 100);

    std::cout << "Setting keys with default TTL (2s)...\n";
    cache.set(1, "one");
    cache.set(2, "two");
    cache.set(3, "three");

    std::cout << "Cache size: " << cache.size() << "\n";
    std::cout << "Key 1 exists: " << cache.contains(1) << "\n";

    // 2. Set with custom TTL
    std::cout << "\nSetting key 4 with 5-second TTL...\n";
    cache.set_with_ttl(4, "four", 5s);

    // 3. Set without TTL (never expires)
    std::cout << "Setting key 5 with no TTL...\n";
    cache.set_no_ttl(5, "permanent");

    // 4. Wait for default TTL to expire
    std::cout << "\nWaiting 2.5 seconds for keys 1-3 to expire...\n";
    std::this_thread::sleep_for(2500ms);

    std::cout << "After wait:\n";
    std::cout << "  Key 1 exists: " << cache.contains(1) << " (expired)\n";
    std::cout << "  Key 2 exists: " << cache.contains(2) << " (expired)\n";
    std::cout << "  Key 3 exists: " << cache.contains(3) << " (expired)\n";
    std::cout << "  Key 4 exists: " << cache.contains(4) << " (5s TTL, still alive)\n";
    std::cout << "  Key 5 exists: " << cache.contains(5) << " (no TTL, permanent)\n";
    std::cout << "  Cache size: " << cache.size() << "\n";

    // 5. Check remaining TTL
    auto remaining = cache.remaining_ttl(4);
    if (remaining) {
        std::cout << "  Key 4 remaining TTL: " << remaining->count() << "s\n";
    }

    // 6. Set with absolute expiry time
    std::cout << "\nSetting key 6 to expire in 1 second (absolute time)...\n";
    cache.set_until(6, "six", ttl_entry<std::string>::from_now(1s));

    // 7. clear_expired() eager cleanup
    std::cout << "\nEagerly clearing expired entries...\n";
    auto cleared = cache.clear_expired();
    std::cout << "Cleared " << cleared << " expired entries\n";
    std::cout << "Cache size after cleanup: " << cache.size() << "\n";

    // 8. Wait for key 4 and 6 to expire
    std::cout << "\nWaiting 3 more seconds...\n";
    std::this_thread::sleep_for(3000ms);

    std::cout << "Key 4 exists: " << cache.contains(4) << "\n";
    std::cout << "Key 5 exists: " << cache.contains(5) << " (still permanent)\n";
    std::cout << "Key 6 exists: " << cache.contains(6) << "\n";

    // 9. Statistics
    std::cout << "\n--- Statistics ---\n";
    std::cout << cache.stats() << "\n";

    return 0;
}
