// SPDX-License-Identifier: MIT
// LRU Cache Library — Microbenchmarks
//
// Provides:
//   - Single-threaded insert / get / peek throughput
//   - Multi-threaded contention comparison (safe_cache vs striped_cache)
//   - Hit-rate comparison across eviction strategies
//   - TTL overhead measurement
//
// Build:
//   cmake -B build -DLRU_BUILD_BENCHMARKS=ON
//   mingw32-make -C build -j4 lru_cache_benchmark
//   ./build/benchmarks/lru_cache_benchmark

#include <algorithm>
#include <cstddef>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <benchmark/benchmark.h>

#include "lru.hpp"

// ============================================================================
// Helpers
// ============================================================================

/// Generate N random integer keys in range [0, max_key).
static std::vector<int> random_keys(std::size_t n, int max_key) {
    std::vector<int> keys(n);
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, max_key - 1);
    for (auto& k : keys) k = dist(rng);
    return keys;
}

/// Generate a sequential workload: set key i to value i.
static void sequential_insert(lru::cache<int, int>& c,
                               const std::vector<int>& keys) {
    for (auto k : keys) c.set(k, k);
}

// ============================================================================
// Single-threaded: LRU insert throughput
// ============================================================================

static void BM_LRU_Insert(benchmark::State& state) {
    const std::size_t capacity = state.range(0);
    const auto keys = random_keys(capacity * 2, static_cast<int>(capacity * 4));

    for (auto _ : state) {
        lru::cache<int, int> c(capacity);
        for (auto k : keys) {
            c.set(k, k);
        }
        benchmark::DoNotOptimize(c);
    }
}
BENCHMARK(BM_LRU_Insert)->Arg(1000)->Arg(10000)->Arg(100000);

// ============================================================================
// Single-threaded: LRU get (90% hit rate)
// ============================================================================

static void BM_LRU_Get(benchmark::State& state) {
    const std::size_t capacity = state.range(0);
    lru::cache<int, int> c(capacity);

    // Fill to capacity with known keys
    for (int i = 0; i < static_cast<int>(capacity); ++i) {
        c.set(i, i);
    }

    // Generate keys: 90% within range, 10% outside → ~90% hit rate
    std::vector<int> lookup_keys;
    {
        std::mt19937 rng(99);
        std::uniform_int_distribution<int> hot(0, static_cast<int>(capacity) - 1);
        std::uniform_int_distribution<int> cold(static_cast<int>(capacity),
                                                static_cast<int>(capacity) * 10);
        std::uniform_real_distribution<double> coin(0.0, 1.0);
        for (std::size_t i = 0; i < capacity * 2; ++i) {
            lookup_keys.push_back(coin(rng) < 0.9 ? hot(rng) : cold(rng));
        }
    }

    std::size_t idx = 0;
    for (auto _ : state) {
        auto result = c.get(lookup_keys[idx % lookup_keys.size()]);
        benchmark::DoNotOptimize(result);
        ++idx;
    }
}
BENCHMARK(BM_LRU_Get)->Arg(1000)->Arg(10000)->Arg(100000);

// ============================================================================
// Thread-safety: safe_cache vs striped_cache (read-heavy, 4 threads)
// ============================================================================

template <typename Cache>
static void BM_MultiThreaded_Reads(benchmark::State& state) {
    const std::size_t capacity = 10000;
    Cache c(capacity);

    // Pre-fill
    for (int i = 0; i < static_cast<int>(capacity); ++i) {
        c.set(i, i);
    }

    const int num_threads = static_cast<int>(state.range(0));
    const auto keys = random_keys(capacity * 4, static_cast<int>(capacity));

    for (auto _ : state) {
        state.PauseTiming();
        // Spin up threads
        std::vector<std::thread> threads;
        threads.reserve(num_threads);
        state.ResumeTiming();

        for (int t = 0; t < num_threads; ++t) {
            threads.emplace_back([&c, &keys, t]() {
                std::size_t idx = t * 1000;
                const auto end = idx + 10000;
                for (; idx < end; ++idx) {
                    auto k = keys[idx % keys.size()];
                    auto result = c.get(k);
                    benchmark::DoNotOptimize(result);
                }
            });
        }

        for (auto& t : threads) t.join();
    }
}

static void BM_SafeCache_Reads(benchmark::State& state) {
    BM_MultiThreaded_Reads<lru::safe_cache<int, int>>(state);
}
BENCHMARK(BM_SafeCache_Reads)->Arg(1)->Arg(2)->Arg(4);

static void BM_StripedCache_Reads(benchmark::State& state) {
    BM_MultiThreaded_Reads<lru::striped_cache<int, int>>(state);
}
BENCHMARK(BM_StripedCache_Reads)->Arg(1)->Arg(2)->Arg(4);

// ============================================================================
// Eviction strategy comparison
// ============================================================================

template <typename Cache>
static void BM_EvictionStrategy_Workload(benchmark::State& state,
                                          const std::vector<int>& keys,
                                          const std::vector<int>& lookup,
                                          const char* name) {
    const std::size_t capacity = state.range(0);
    for (auto _ : state) {
        Cache c(capacity);
        for (auto k : keys) c.set(k, k);
        std::size_t hits = 0;
        for (auto k : lookup) {
            if (c.get(k)) ++hits;
        }
        benchmark::DoNotOptimize(hits);
    }
    state.SetLabel(name);
}

static void BM_HitRate_Strategies(benchmark::State& state) {
    const std::size_t capacity = state.range(0);
    const auto keys = random_keys(capacity * 2, static_cast<int>(capacity * 10));
    const auto lookup = random_keys(capacity * 4, static_cast<int>(capacity * 10));

    if (state.range(1) == 0) {
        BM_EvictionStrategy_Workload<lru::cache<int, int>>(
            state, keys, lookup, "mm_lru");
    } else if (state.range(1) == 1) {
        BM_EvictionStrategy_Workload<lru::two_q<int, int>>(
            state, keys, lookup, "mm_2q");
    } else if (state.range(1) == 2) {
        BM_EvictionStrategy_Workload<lru::lfu_cache<int, int>>(
            state, keys, lookup, "mm_tiny_lfu");
    } else if (state.range(1) == 3) {
        BM_EvictionStrategy_Workload<lru::w_tiny_lfu<int, int>>(
            state, keys, lookup, "mm_wtiny_lfu");
    } else {
        BM_EvictionStrategy_Workload<lru::fifo_cache<int, int>>(
            state, keys, lookup, "mm_fifo");
    }
}
BENCHMARK(BM_HitRate_Strategies)
    ->Args({10000, 0})  // LRU
    ->Args({10000, 1})  // 2Q
    ->Args({10000, 2})  // TinyLFU
    ->Args({10000, 3})  // W-TinyLFU
    ->Args({10000, 4}); // FIFO

// ============================================================================
// TTL overhead
// ============================================================================

static void BM_TTL_Insert(benchmark::State& state) {
    const std::size_t capacity = state.range(0);
    const auto keys = random_keys(capacity * 2, static_cast<int>(capacity * 4));

    for (auto _ : state) {
        lru::ttl_cache<int, int> c(std::chrono::seconds(60), capacity);
        for (auto k : keys) {
            c.set(k, k);
        }
        benchmark::DoNotOptimize(c);
    }
}
BENCHMARK(BM_TTL_Insert)->Arg(1000)->Arg(10000);

// ============================================================================
// Peek vs Get overhead
// ============================================================================

static void BM_LRU_Peek(benchmark::State& state) {
    const std::size_t capacity = state.range(0);
    lru::cache<int, int> c(capacity);
    for (int i = 0; i < static_cast<int>(capacity); ++i) c.set(i, i);
    const auto keys = random_keys(capacity * 2, static_cast<int>(capacity));

    std::size_t idx = 0;
    for (auto _ : state) {
        auto result = c.peek(keys[idx % keys.size()]);
        benchmark::DoNotOptimize(result);
        ++idx;
    }
}
BENCHMARK(BM_LRU_Peek)->Arg(1000)->Arg(100000);

// ============================================================================
// Main
// ============================================================================

BENCHMARK_MAIN();
