// Unified LRU Cache - Recommended production configuration example
//
// Targets multi-threaded, high-concurrency, read-heavy workloads.
// Each step below is annotated with its motivation and the production
// symptom it prevents. Encapsulate this in a factory function in your
// own codebase rather than scattering the calls across business logic.
//
// Build:
//   cmake -B build -G "MinGW Makefiles" \
//       -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang \
//       -DCMAKE_BUILD_TYPE=Release -DLRU_BUILD_EXAMPLES=ON
//   mingw32-make -C build -j2 production_example

#include <chrono>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "../lru.hpp"

namespace {
using namespace lru;
using namespace std::chrono_literals;

// Configure a production_cache in-place. The cache is non-copyable
// (its sharded_mm_lru member is non-copyable), so we configure it via
// a reference rather than returning by value.
template <typename K, typename V>
void configure_production_cache(production_cache<K, V>& c,
                                std::size_t max_size,
                                std::size_t num_stripes = 128) {
    (void)max_size;
    (void)num_stripes;

    // (2) Lock fairness. Default is writer_fair (no writer starvation).
    // Switch to reader_preferred only if the workload can tolerate
    // write latency spikes — typical read-heavy services can.
    c.set_fairness_mode(detail::fairness_mode::reader_preferred);

    // (3) Pre-reserve the hash table. Avoids synchronous rehash stalls
    // when the first big wave of inserts arrives.
    c.reserve(max_size);

    // (4) Memory watermarks + OOM handler. soft triggers aggressive
    // eviction; critical flips the cache to read-only so the process
    // survives memory pressure instead of being OOM-killed.
    c.set_memory_watermarks(/*soft=*/0.85, /*critical=*/0.95);
    c.set_oom_handler([](std::size_t cur, std::size_t max) {
        std::cerr << "[oom] cache memory " << cur << "/" << max << " bytes\n";
    });

    // (5) Background TTL cleaner: per-shard round-robin, no global lock.
    c.start_ttl_cleaner(1s);

    // (6) Event drain worker: bounds the time TLS access ring and
    // callback ring entries wait before being drained. Without this,
    // a quiet thread may hold deferred promotions indefinitely.
    c.start_event_drain(500ms);

    // (7) Async callbacks: on_evict may do IO (logging / metrics push),
    // so the drain thread must not block on user callbacks.
    c.set_async_callbacks(true);
    // NOTE: For this self-contained example we don't register an on_evict
    // callback — doing so safely requires the callback to outlive every
    // pending event in the async queue at destruction time. In production,
    // either (a) call set_async_callbacks(false) before destruction to drain
    // the queue, or (b) ensure the callback captures only state that
    // outlives the cache (e.g., a global logger). See "Graceful shutdown"
    // section of the README for the recommended ordering.
}

}  // namespace

int main() {
    using namespace lru;
    using K = int;
    using V = std::string;

    std::cout << "=== Production Cache Configuration Example ===\n\n";
    std::cout.flush();

    // (1) Select production_cache = segmented hash table + sharded LRU
    //     + striped locking. Incremental rehash is auto-enabled.
    production_cache<K, V> c{/*max_size=*/1'000'000, /*num_stripes=*/128};
    std::cout << "[step 1] cache constructed\n"; std::cout.flush();

    configure_production_cache(c, /*max_size=*/1'000'000, /*num_stripes=*/128);
    std::cout << "[step 2] cache configured\n"; std::cout.flush();

    // (8) Optional: event tracker for hot-key analysis. Records
    // insert/hit/evict into a TLS ring, drained by the event_drain worker.
    auto tracker = c.enable_event_tracking();
    std::cout << "[step 3] event tracking enabled\n"; std::cout.flush();

    // Warm up with some traffic so the example produces real numbers.
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&c, t]() {
            for (int i = 0; i < 5000; ++i) {
                K key = (i * 7 + t * 13) % 1000;
                if (i % 10 < 7) {
                    (void)c.try_get(key);  // 70% reads
                } else {
                    c.set(key, "v" + std::to_string(key));  // 30% writes
                }
            }
        });
    }
    for (auto& th : threads) th.join();
    std::cout << "[step 4] workload done\n"; std::cout.flush();

    // (9) Prometheus export — call from a monitor thread, NOT the read
    //     hot path. The text is meant to be served on a /metrics endpoint.
    std::cout << "[step 5] before prometheus_text\n"; std::cout.flush();
    std::string metrics = c.prometheus_text();
    std::cout << "[step 6] after prometheus_text, len=" << metrics.size() << "\n"; std::cout.flush();
    std::cout << "--- Prometheus metrics (truncated) ---\n";
    // Show first ~20 lines as a sanity check.
    std::size_t pos = 0;
    for (int line = 0; line < 20 && pos < metrics.size(); ++line) {
        auto nl = metrics.find('\n', pos);
        if (nl == std::string::npos) nl = metrics.size();
        std::cout << metrics.substr(pos, nl - pos) << "\n";
        pos = nl + 1;
    }
    std::cout << "...\n\n";
    std::cout.flush();

    // (10) Graceful shutdown: reject new ops, wait for outstanding
    //      read_handles to be released, then destroy.
    std::cout << "Shutting down...\n"; std::cout.flush();
    c.shutdown();
    std::cout << "[step 7] after shutdown, active=" << c.active_handle_count() << "\n"; std::cout.flush();
    while (c.active_handle_count() > 0) {
        std::this_thread::sleep_for(10ms);
    }
    // Drain async callback queue before destruction — required when
    // set_async_callbacks(true) was used. stop_ttl_cleaner /
    // stop_event_drain are called automatically by the destructor, but
    // async callback shutdown is the user's responsibility per the README.
    std::cout << "[step 8] before stop_ttl_cleaner\n"; std::cout.flush();
    c.stop_ttl_cleaner();
    std::cout << "[step 9] before set_async_callbacks(false)\n"; std::cout.flush();
    c.set_async_callbacks(false);
    std::cout << "[step 10] before destroy\n"; std::cout.flush();
    std::cout << "Shutdown complete. Safe to destroy.\n";
    return 0;
}
