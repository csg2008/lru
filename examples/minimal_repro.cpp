// Minimal repro for production_example destructor crash
#include <iostream>
#include <string>
#include "../lru.hpp"

int main() {
    std::fprintf(stderr, "=== minimal repro ===\n");
    {
        using namespace lru;
        production_cache<int, std::string> c{1000, 128};
        std::fprintf(stderr, "constructed\n");
        auto tracker = c.enable_event_tracking();
        std::fprintf(stderr, "tracking enabled\n");
        for (int i = 0; i < 100; ++i) {
            c.set(i, "v");
            (void)c.try_get(i);
        }
        std::fprintf(stderr, "workload done\n");
    }
    std::fprintf(stderr, "scope exit done\n");
    return 0;
}
