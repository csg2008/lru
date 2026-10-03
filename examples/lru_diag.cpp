// SPDX-License-Identifier: MIT
// lru_diag — operator-facing diagnostic dump for a running cache.
//
// The library exposes diagnostics_text() / diagnostics_json() / prometheus_text()
// as library calls, but nothing showed how an operator is supposed to reach them
// from a live process. This program is that missing piece: it drives a cache,
// then dumps its state on demand, which is the pattern a deployment copies into
// a SIGUSR1 handler, an admin endpoint, or a periodic dump.
//
// Usage:
//   ./lru_diag [--interval-ms N] [--rounds K]
//
//   N  how often the simulated dump fires (default 250ms)
//   K  how many dumps to emit before exiting (default 3; 0 = run until Ctrl-C)
//
// Every dump includes the eviction breakdown by reason, which is the first
// thing to check when resident memory does not fall.

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>

#include "lru.hpp"

namespace {

std::atomic<bool> g_stop{false};

void handle_signal(int) {
    g_stop.store(true, std::memory_order_relaxed);
}

/// Parse `--flag <value>`; returns the default when the flag is absent.
long long arg_or(int argc, char** argv, const char* flag, long long fallback) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::string(argv[i]) == flag) {
            return std::strtoll(argv[i + 1], nullptr, 10);
        }
    }
    return fallback;
}

/// One dump. `force_refresh` matters: without it diagnostics_text() may return
/// a snapshot up to 500ms old, so two back-to-back dumps in a debugger would
/// look identical and read as "nothing changed".
void dump_once(const char* label, lru::production_cache<int, std::string>& cache) {
    std::printf("\n===== %s =====\n", label);
    std::printf("--- diagnostics_text ---\n%s\n", cache.diagnostics_text().c_str());

    const auto snap = cache.stats_snapshot();
    std::printf("--- eviction reasons ---\n");
    std::printf("  capacity : %llu\n",
                static_cast<unsigned long long>(
                    snap.evictions_capacity.value.load(std::memory_order_relaxed)));
    std::printf("  ttl      : %llu\n",
                static_cast<unsigned long long>(
                    snap.evictions_ttl.value.load(std::memory_order_relaxed)));
    std::printf("  explicit : %llu\n",
                static_cast<unsigned long long>(
                    snap.evictions_explicit.value.load(std::memory_order_relaxed)));
    std::printf("  failed   : %llu  (eviction attempts that freed nothing)\n",
                static_cast<unsigned long long>(
                    snap.eviction_failed.value.load(std::memory_order_relaxed)));
    std::printf("--- locks ---\n");
    std::printf("  write waits: %llu   read waits: %llu   try-fails: %llu\n",
                static_cast<unsigned long long>(
                    snap.write_lock_wait_count.value.load(std::memory_order_relaxed)),
                static_cast<unsigned long long>(
                    snap.read_lock_wait_count.value.load(std::memory_order_relaxed)),
                static_cast<unsigned long long>(
                    snap.try_lock_fail_count.value.load(std::memory_order_relaxed)));
    std::fflush(stdout);
}

}  // namespace

int main(int argc, char** argv) {
    std::signal(SIGINT, handle_signal);

    const auto interval =
        std::chrono::milliseconds(arg_or(argc, argv, "--interval-ms", 250));
    const auto rounds = arg_or(argc, argv, "--rounds", 3);

    // A small cache with a short TTL so all three eviction reasons show up in a
    // handful of seconds of traffic.
    lru::production_cache<int, std::string> cache(/*max_size=*/256);
    cache.set_ttl_jitter_pct(0.10);
    lru::config cfg;
    cfg.ttl_cleaner_interval = std::chrono::milliseconds(50);
    cfg.latency_tracking = true;  // so the percentile fields are populated
    (void)lru::apply_config(cache, cfg);

    std::printf("lru_diag: max_size=256, dump every %lldms, %lld round(s)\n",
                static_cast<long long>(interval.count()), rounds);

    std::thread traffic([&cache] {
        std::uint64_t i = 0;
        while (!g_stop.load(std::memory_order_relaxed)) {
            const int key = static_cast<int>(i % 512);
            cache.set(key, std::string(64, 'x'));
            // Read some keys (hits) and let others age out.
            (void)cache.try_get(key);
            if (i % 7 == 0) {
                (void)cache.remove(key);  // explicit removal
            }
            if (i % 1000 == 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            ++i;
        }
    });

    for (long long r = 0; rounds == 0 || r < rounds; ++r) {
        std::this_thread::sleep_for(interval);
        if (g_stop.load(std::memory_order_relaxed)) {
            std::printf("\nSIGINT — dumping final state\n");
            break;
        }
        const std::string label = "dump #" + std::to_string(r + 1);
        dump_once(label.c_str(), cache);
    }

    g_stop.store(true, std::memory_order_relaxed);
    traffic.join();
    dump_once("final", cache);
    std::printf("\nlru_diag: done\n");
    return 0;
}
