// SPDX-License-Identifier: MIT
// LRU Cache Library — High-Concurrency Read-Heavy Benchmarks
//
// Comprehensive benchmarks for read-heavy concurrent workloads, covering:
//   1. Read-heavy workload (99% reads / 1% writes): safe_cache vs striped_cache
//   2. Callback overhead: with vs without callbacks, TLS ring zero-overhead
//   3. Batch read throughput: peek() (shared lock, no promotion) vs get() (exclusive lock, promotion)
//   4. Slab-style allocation vs new/delete for insert-heavy workload
//   5. Scalability: safe_cache and striped_cache with 1/2/4/8/16 threads
//
// Build:
//   cmake -B build -DLRU_BUILD_BENCHMARKS=ON
//   mingw32-make -C build -j4 lru_concurrent_read_benchmark
//   ./build/benchmarks/lru_concurrent_read_benchmark

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <random>
#include <shared_mutex>
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

/// Per-thread workload: performs ops_count operations on the cache.
/// read_ratio fraction are get(), the rest are set().
template <typename Cache>
static void read_heavy_worker(
    Cache& cache,
    const std::vector<int>& keys,
    std::size_t ops_count,
    double read_ratio,
    std::atomic<std::size_t>& total_hits)
{
    std::mt19937 rng(static_cast<std::uint32_t>(
        std::hash<std::thread::id>{}(std::this_thread::get_id())));
    std::uniform_real_distribution<double> coin(0.0, 1.0);
    std::uniform_int_distribution<int> key_dist(0, static_cast<int>(keys.size() - 1));
    // For writes: use keys outside the pre-filled range to trigger evictions
    std::uniform_int_distribution<int> write_key_dist(
        static_cast<int>(keys.size()), static_cast<int>(keys.size() * 2));

    std::size_t local_hits = 0;
    for (std::size_t i = 0; i < ops_count; ++i) {
        if (coin(rng) < read_ratio) {
            auto k = keys[key_dist(rng)];
            auto h = cache.get(k);
            if (h) ++local_hits;
        } else {
            auto k = write_key_dist(rng);
            cache.set(k, k);
        }
    }
    total_hits.fetch_add(local_hits, std::memory_order_relaxed);
}

// ============================================================================
// 1. Read-heavy workload (99% reads, 1% writes)
//    Multi-threaded benchmark with 1/2/4/8/16 threads,
//    comparing safe_cache vs striped_cache.
//    Pre-fill cache with 100K items, then each thread does 99% get() + 1% set().
// ============================================================================

template <typename Cache>
static void BM_ReadHeavy_Generic(benchmark::State& state) {
    const std::size_t capacity = 100000;
    const int num_threads = static_cast<int>(state.range(0));
    const std::size_t ops_per_thread = 50000;

    for (auto _ : state) {
        state.PauseTiming();
        Cache cache(capacity);

        // Pre-fill with 100K items
        for (int i = 0; i < static_cast<int>(capacity); ++i) {
            cache.set(i, i);
        }

        auto keys = random_keys(capacity, static_cast<int>(capacity));
        std::atomic<std::size_t> total_hits{0};
        state.ResumeTiming();

        std::vector<std::thread> threads;
        threads.reserve(num_threads);
        for (int t = 0; t < num_threads; ++t) {
            threads.emplace_back(read_heavy_worker<Cache>,
                std::ref(cache), std::cref(keys),
                ops_per_thread, 0.99, std::ref(total_hits));
        }
        for (auto& t : threads) t.join();

        benchmark::DoNotOptimize(total_hits.load());
    }

    state.SetItemsProcessed(state.iterations() * ops_per_thread * num_threads);
}

static void BM_SafeCache_ReadHeavy99(benchmark::State& state) {
    BM_ReadHeavy_Generic<lru::safe_cache<int, int>>(state);
}
BENCHMARK(BM_SafeCache_ReadHeavy99)
    ->Arg(1)->Arg(2)->Arg(4)->Arg(8)->Arg(16)
    ->Arg(32)->Arg(64)->Arg(128)
    ->UseRealTime();

static void BM_StripedCache_ReadHeavy99(benchmark::State& state) {
    BM_ReadHeavy_Generic<lru::striped_cache<int, int>>(state);
}
BENCHMARK(BM_StripedCache_ReadHeavy99)
    ->Arg(1)->Arg(2)->Arg(4)->Arg(8)->Arg(16)
    ->Arg(32)->Arg(64)->Arg(128)
    ->UseRealTime();

// ============================================================================
// 2. Callback overhead comparison
//    Register hit/miss callbacks and measure throughput with vs without.
//    Verifies that the TLS callback ring is zero-overhead when no callbacks
//    are registered (collect_* short-circuits on has_*_callbacks_ atomic flag).
// ============================================================================

/// Worker for callback benchmark: pure read workload.
template <typename Cache>
static void callback_read_worker(
    Cache& cache,
    const std::vector<int>& keys,
    std::size_t ops_count,
    std::atomic<std::size_t>& total_hits)
{
    std::mt19937 rng(static_cast<std::uint32_t>(
        std::hash<std::thread::id>{}(std::this_thread::get_id())));
    std::uniform_int_distribution<int> key_dist(0, static_cast<int>(keys.size() - 1));

    std::size_t local_hits = 0;
    for (std::size_t i = 0; i < ops_count; ++i) {
        auto k = keys[key_dist(rng)];
        auto h = cache.get(k);
        if (h) ++local_hits;
    }
    total_hits.fetch_add(local_hits, std::memory_order_relaxed);
}

/// No callbacks registered — should be zero overhead.
static void BM_Callback_NoCallbacks(benchmark::State& state) {
    const std::size_t capacity = 100000;
    const int num_threads = static_cast<int>(state.range(0));
    const std::size_t ops_per_thread = 50000;

    for (auto _ : state) {
        state.PauseTiming();
        lru::safe_cache<int, int> cache(capacity);
        for (int i = 0; i < static_cast<int>(capacity); ++i) {
            cache.set(i, i);
        }
        auto keys = random_keys(capacity, static_cast<int>(capacity));
        std::atomic<std::size_t> total_hits{0};
        state.ResumeTiming();

        std::vector<std::thread> threads;
        threads.reserve(num_threads);
        for (int t = 0; t < num_threads; ++t) {
            threads.emplace_back(callback_read_worker<lru::safe_cache<int, int>>,
                std::ref(cache), std::cref(keys),
                ops_per_thread, std::ref(total_hits));
        }
        for (auto& t : threads) t.join();

        benchmark::DoNotOptimize(total_hits.load());
    }

    state.SetItemsProcessed(state.iterations() * ops_per_thread * num_threads);
}
BENCHMARK(BM_Callback_NoCallbacks)
    ->Arg(4)->Arg(8)->Arg(32)
    ->UseRealTime();

/// With hit+miss callbacks registered — measures callback overhead.
static void BM_Callback_WithCallbacks(benchmark::State& state) {
    const std::size_t capacity = 100000;
    const int num_threads = static_cast<int>(state.range(0));
    const std::size_t ops_per_thread = 50000;

    for (auto _ : state) {
        state.PauseTiming();
        lru::safe_cache<int, int> cache(capacity);
        for (int i = 0; i < static_cast<int>(capacity); ++i) {
            cache.set(i, i);
        }

        // Register lightweight callbacks that do minimal work
        std::atomic<std::size_t> hit_count{0};
        std::atomic<std::size_t> miss_count{0};
        cache.on_hit([&hit_count](const int&, const int&) {
            hit_count.fetch_add(1, std::memory_order_relaxed);
        });
        cache.on_miss([&miss_count](const int&) {
            miss_count.fetch_add(1, std::memory_order_relaxed);
        });

        auto keys = random_keys(capacity, static_cast<int>(capacity));
        std::atomic<std::size_t> total_hits{0};
        state.ResumeTiming();

        std::vector<std::thread> threads;
        threads.reserve(num_threads);
        for (int t = 0; t < num_threads; ++t) {
            threads.emplace_back(callback_read_worker<lru::safe_cache<int, int>>,
                std::ref(cache), std::cref(keys),
                ops_per_thread, std::ref(total_hits));
        }
        for (auto& t : threads) t.join();

        benchmark::DoNotOptimize(total_hits.load());
        benchmark::DoNotOptimize(hit_count.load());
        benchmark::DoNotOptimize(miss_count.load());
    }

    state.SetItemsProcessed(state.iterations() * ops_per_thread * num_threads);
}
BENCHMARK(BM_Callback_WithCallbacks)
    ->Arg(4)->Arg(8)->Arg(32)
    ->UseRealTime();

/// Same test on striped_cache — callbacks should scale better with sharding.
static void BM_Callback_Striped_NoCallbacks(benchmark::State& state) {
    const std::size_t capacity = 100000;
    const int num_threads = static_cast<int>(state.range(0));
    const std::size_t ops_per_thread = 50000;

    for (auto _ : state) {
        state.PauseTiming();
        lru::striped_cache<int, int> cache(capacity);
        for (int i = 0; i < static_cast<int>(capacity); ++i) {
            cache.set(i, i);
        }
        auto keys = random_keys(capacity, static_cast<int>(capacity));
        std::atomic<std::size_t> total_hits{0};
        state.ResumeTiming();

        std::vector<std::thread> threads;
        threads.reserve(num_threads);
        for (int t = 0; t < num_threads; ++t) {
            threads.emplace_back(callback_read_worker<lru::striped_cache<int, int>>,
                std::ref(cache), std::cref(keys),
                ops_per_thread, std::ref(total_hits));
        }
        for (auto& t : threads) t.join();

        benchmark::DoNotOptimize(total_hits.load());
    }

    state.SetItemsProcessed(state.iterations() * ops_per_thread * num_threads);
}
BENCHMARK(BM_Callback_Striped_NoCallbacks)
    ->Arg(4)->Arg(8)->Arg(32)
    ->UseRealTime();

static void BM_Callback_Striped_WithCallbacks(benchmark::State& state) {
    const std::size_t capacity = 100000;
    const int num_threads = static_cast<int>(state.range(0));
    const std::size_t ops_per_thread = 50000;

    for (auto _ : state) {
        state.PauseTiming();
        lru::striped_cache<int, int> cache(capacity);
        for (int i = 0; i < static_cast<int>(capacity); ++i) {
            cache.set(i, i);
        }

        std::atomic<std::size_t> hit_count{0};
        std::atomic<std::size_t> miss_count{0};
        cache.on_hit([&hit_count](const int&, const int&) {
            hit_count.fetch_add(1, std::memory_order_relaxed);
        });
        cache.on_miss([&miss_count](const int&) {
            miss_count.fetch_add(1, std::memory_order_relaxed);
        });

        auto keys = random_keys(capacity, static_cast<int>(capacity));
        std::atomic<std::size_t> total_hits{0};
        state.ResumeTiming();

        std::vector<std::thread> threads;
        threads.reserve(num_threads);
        for (int t = 0; t < num_threads; ++t) {
            threads.emplace_back(callback_read_worker<lru::striped_cache<int, int>>,
                std::ref(cache), std::cref(keys),
                ops_per_thread, std::ref(total_hits));
        }
        for (auto& t : threads) t.join();

        benchmark::DoNotOptimize(total_hits.load());
        benchmark::DoNotOptimize(hit_count.load());
        benchmark::DoNotOptimize(miss_count.load());
    }

    state.SetItemsProcessed(state.iterations() * ops_per_thread * num_threads);
}
BENCHMARK(BM_Callback_Striped_WithCallbacks)
    ->Arg(4)->Arg(8)->Arg(32)
    ->UseRealTime();

// ============================================================================
// 3. Batch read throughput
//    Compare peek() (shared lock, no LRU promotion) vs get() (exclusive lock,
//    LRU promotion) vs get_shared() (shared lock, value copy, no LRU promotion)
//    on striped_cache with multi-threaded access.
//
//    peek() and get_shared() use shared locks — ideal for batch read scenarios
//    where LRU order updates are not needed. get() requires exclusive locks
//    for LRU promotion, making it the slowest for pure read throughput.
// ============================================================================

/// Worker for batch peek benchmark: uses peek() (read-only, shared lock).
static void batch_peek_worker(
    lru::striped_cache<int, int>& cache,
    const std::vector<int>& keys,
    std::size_t ops_count,
    std::atomic<std::size_t>& total_hits)
{
    std::mt19937 rng(static_cast<std::uint32_t>(
        std::hash<std::thread::id>{}(std::this_thread::get_id())));
    std::uniform_int_distribution<int> key_dist(0, static_cast<int>(keys.size() - 1));

    std::size_t local_hits = 0;
    for (std::size_t i = 0; i < ops_count; ++i) {
        auto k = keys[key_dist(rng)];
        auto h = cache.peek(k);
        if (h) ++local_hits;
    }
    total_hits.fetch_add(local_hits, std::memory_order_relaxed);
}

/// Worker for batch get benchmark: uses get() (write lock, LRU promotion).
static void batch_get_worker(
    lru::striped_cache<int, int>& cache,
    const std::vector<int>& keys,
    std::size_t ops_count,
    std::atomic<std::size_t>& total_hits)
{
    std::mt19937 rng(static_cast<std::uint32_t>(
        std::hash<std::thread::id>{}(std::this_thread::get_id())));
    std::uniform_int_distribution<int> key_dist(0, static_cast<int>(keys.size() - 1));

    std::size_t local_hits = 0;
    for (std::size_t i = 0; i < ops_count; ++i) {
        auto k = keys[key_dist(rng)];
        auto h = cache.get(k);
        if (h) ++local_hits;
    }
    total_hits.fetch_add(local_hits, std::memory_order_relaxed);
}

/// Worker for batch get_shared benchmark: uses get_shared() (shared lock, no promotion).
static void batch_get_shared_worker(
    lru::striped_cache<int, int>& cache,
    const std::vector<int>& keys,
    std::size_t ops_count,
    std::atomic<std::size_t>& total_hits)
{
    std::mt19937 rng(static_cast<std::uint32_t>(
        std::hash<std::thread::id>{}(std::this_thread::get_id())));
    std::uniform_int_distribution<int> key_dist(0, static_cast<int>(keys.size() - 1));

    std::size_t local_hits = 0;
    for (std::size_t i = 0; i < ops_count; ++i) {
        auto k = keys[key_dist(rng)];
        auto h = cache.get_shared(k);
        if (h) ++local_hits;
    }
    total_hits.fetch_add(local_hits, std::memory_order_relaxed);
}

/// Worker that does vectorized batch reads: accumulate keys then peek all.
/// Simulates get_multi() by batching N keys per lock acquisition via peek().
template <std::size_t BatchSize>
static void batch_vectorized_peek_worker(
    lru::striped_cache<int, int>& cache,
    const std::vector<int>& keys,
    std::size_t ops_count,
    std::atomic<std::size_t>& total_hits)
{
    std::mt19937 rng(static_cast<std::uint32_t>(
        std::hash<std::thread::id>{}(std::this_thread::get_id())));
    std::uniform_int_distribution<int> key_dist(0, static_cast<int>(keys.size() - 1));

    std::size_t local_hits = 0;
    std::array<int, BatchSize> batch_keys;

    std::size_t processed = 0;
    while (processed < ops_count) {
        // Fill batch
        std::size_t batch_count = std::min(BatchSize, ops_count - processed);
        for (std::size_t i = 0; i < batch_count; ++i) {
            batch_keys[i] = keys[key_dist(rng)];
        }
        // Process batch: sequential peek() calls benefit from cache locality
        // and shared lock reuse within the same stripe
        for (std::size_t i = 0; i < batch_count; ++i) {
            auto h = cache.peek(batch_keys[i]);
            if (h) ++local_hits;
        }
        processed += batch_count;
    }
    total_hits.fetch_add(local_hits, std::memory_order_relaxed);
}

static void BM_BatchRead_Get(benchmark::State& state) {
    const std::size_t capacity = 100000;
    const int num_threads = static_cast<int>(state.range(0));
    const std::size_t ops_per_thread = 50000;

    for (auto _ : state) {
        state.PauseTiming();
        lru::striped_cache<int, int> cache(capacity);
        for (int i = 0; i < static_cast<int>(capacity); ++i) {
            cache.set(i, i);
        }
        auto keys = random_keys(capacity, static_cast<int>(capacity));
        std::atomic<std::size_t> total_hits{0};
        state.ResumeTiming();

        std::vector<std::thread> threads;
        threads.reserve(num_threads);
        for (int t = 0; t < num_threads; ++t) {
            threads.emplace_back(batch_get_worker,
                std::ref(cache), std::cref(keys),
                ops_per_thread, std::ref(total_hits));
        }
        for (auto& t : threads) t.join();

        benchmark::DoNotOptimize(total_hits.load());
    }

    state.SetItemsProcessed(state.iterations() * ops_per_thread * num_threads);
}
BENCHMARK(BM_BatchRead_Get)
    ->Arg(4)->Arg(8)->Arg(32)->Arg(64)
    ->UseRealTime();

static void BM_BatchRead_Peek(benchmark::State& state) {
    const std::size_t capacity = 100000;
    const int num_threads = static_cast<int>(state.range(0));
    const std::size_t ops_per_thread = 50000;

    for (auto _ : state) {
        state.PauseTiming();
        lru::striped_cache<int, int> cache(capacity);
        for (int i = 0; i < static_cast<int>(capacity); ++i) {
            cache.set(i, i);
        }
        auto keys = random_keys(capacity, static_cast<int>(capacity));
        std::atomic<std::size_t> total_hits{0};
        state.ResumeTiming();

        std::vector<std::thread> threads;
        threads.reserve(num_threads);
        for (int t = 0; t < num_threads; ++t) {
            threads.emplace_back(batch_peek_worker,
                std::ref(cache), std::cref(keys),
                ops_per_thread, std::ref(total_hits));
        }
        for (auto& t : threads) t.join();

        benchmark::DoNotOptimize(total_hits.load());
    }

    state.SetItemsProcessed(state.iterations() * ops_per_thread * num_threads);
}
BENCHMARK(BM_BatchRead_Peek)
    ->Arg(4)->Arg(8)->Arg(32)->Arg(64)
    ->UseRealTime();

static void BM_BatchRead_GetShared(benchmark::State& state) {
    const std::size_t capacity = 100000;
    const int num_threads = static_cast<int>(state.range(0));
    const std::size_t ops_per_thread = 50000;

    for (auto _ : state) {
        state.PauseTiming();
        lru::striped_cache<int, int> cache(capacity);
        for (int i = 0; i < static_cast<int>(capacity); ++i) {
            cache.set(i, i);
        }
        auto keys = random_keys(capacity, static_cast<int>(capacity));
        std::atomic<std::size_t> total_hits{0};
        state.ResumeTiming();

        std::vector<std::thread> threads;
        threads.reserve(num_threads);
        for (int t = 0; t < num_threads; ++t) {
            threads.emplace_back(batch_get_shared_worker,
                std::ref(cache), std::cref(keys),
                ops_per_thread, std::ref(total_hits));
        }
        for (auto& t : threads) t.join();

        benchmark::DoNotOptimize(total_hits.load());
    }

    state.SetItemsProcessed(state.iterations() * ops_per_thread * num_threads);
}
BENCHMARK(BM_BatchRead_GetShared)
    ->Arg(4)->Arg(8)->Arg(32)->Arg(64)
    ->UseRealTime();

/// Vectorized batch peek with batch size 64 — simulates get_multi() behavior
/// where keys are pre-collected then processed in batches.
static void BM_BatchRead_VectorizedPeek64(benchmark::State& state) {
    const std::size_t capacity = 100000;
    const int num_threads = static_cast<int>(state.range(0));
    const std::size_t ops_per_thread = 50000;

    for (auto _ : state) {
        state.PauseTiming();
        lru::striped_cache<int, int> cache(capacity);
        for (int i = 0; i < static_cast<int>(capacity); ++i) {
            cache.set(i, i);
        }
        auto keys = random_keys(capacity, static_cast<int>(capacity));
        std::atomic<std::size_t> total_hits{0};
        state.ResumeTiming();

        std::vector<std::thread> threads;
        threads.reserve(num_threads);
        for (int t = 0; t < num_threads; ++t) {
            threads.emplace_back(batch_vectorized_peek_worker<64>,
                std::ref(cache), std::cref(keys),
                ops_per_thread, std::ref(total_hits));
        }
        for (auto& t : threads) t.join();

        benchmark::DoNotOptimize(total_hits.load());
    }

    state.SetItemsProcessed(state.iterations() * ops_per_thread * num_threads);
}
BENCHMARK(BM_BatchRead_VectorizedPeek64)
    ->Arg(4)->Arg(8)->Arg(32)->Arg(64)
    ->UseRealTime();

// ============================================================================
// 4. Slab-style allocation vs new/delete
//    Compares a simple slab/pool allocator against raw new/delete for
//    fixed-size block allocation, simulating the allocation pattern of
//    cache item insertion. The slab allocator pre-allocates large blocks
//    (slabs) and bumps a pointer within each slab, avoiding per-allocation
//    heap overhead. Deallocation is deferred to slab-level recycling.
// ============================================================================

/// Minimal slab allocator for fixed-size blocks.
/// Simulates the allocation pattern that CacheLib uses for item storage.
class slab_allocator {
public:
    struct config {
        std::size_t slab_size = 65536;      // bytes per slab
        std::size_t initial_slabs = 4;
        std::size_t max_slabs = 1024;
    };

    explicit slab_allocator(std::size_t block_size, std::size_t slab_sz = 65536,
                            std::size_t initial = 4, std::size_t max = 1024)
        : block_size_(block_size)
        , blocks_per_slab_(slab_sz / block_size)
        , max_slabs_(max)
    {
        for (std::size_t i = 0; i < initial; ++i) {
            add_slab();
        }
    }

    ~slab_allocator() {
        for (auto* slab : slabs_) {
            ::operator delete(slab);
        }
    }

    slab_allocator(const slab_allocator&) = delete;
    slab_allocator& operator=(const slab_allocator&) = delete;

    void* allocate() {
        std::lock_guard<std::mutex> lock(mutex_);
        // Try free list first
        if (free_head_) {
            void* block = free_head_;
            free_head_ = *static_cast<void**>(free_head_);
            return block;
        }
        // Bump allocate from current slab
        if (!slabs_.empty() && current_offset_ < blocks_per_slab_) {
            auto* slab = slabs_[current_slab_index_];
            void* block = static_cast<char*>(slab) + current_offset_ * block_size_;
            ++current_offset_;
            return block;
        }
        // Need a new slab
        add_slab_locked();
        {
            auto* slab = slabs_[current_slab_index_];
            void* block = static_cast<char*>(slab) + current_offset_ * block_size_;
            ++current_offset_;
            return block;
        }
    }

    void deallocate(void* block) {
        std::lock_guard<std::mutex> lock(mutex_);
        *static_cast<void**>(block) = free_head_;
        free_head_ = block;
    }

    std::size_t slab_count() const { return slabs_.size(); }

private:
    void add_slab() {
        auto* slab = ::operator new(block_size_ * blocks_per_slab_);
        slabs_.push_back(slab);
        current_slab_index_ = slabs_.size() - 1;
        current_offset_ = 0;
    }

    void add_slab_locked() {
        if (slabs_.size() >= max_slabs_) {
            // Recycle: scan slabs for a fully freed one (simplified: just add)
        }
        add_slab();
    }

    std::size_t block_size_;
    std::size_t blocks_per_slab_;
    std::size_t max_slabs_;
    std::vector<void*> slabs_;
    std::size_t current_slab_index_ = 0;
    std::size_t current_offset_ = 0;
    void* free_head_ = nullptr;
    std::mutex mutex_;
};

/// Benchmark: raw new/delete for fixed-size blocks (64 bytes each).
static void BM_Alloc_NewDelete(benchmark::State& state) {
    const std::size_t block_size = 64;
    const std::size_t ops_per_iter = state.range(0);

    for (auto _ : state) {
        std::vector<void*> blocks;
        blocks.reserve(ops_per_iter);

        // Allocate
        for (std::size_t i = 0; i < ops_per_iter; ++i) {
            void* p = ::operator new(block_size);
            std::memset(p, 0, block_size);
            blocks.push_back(p);
        }

        // Deallocate
        for (auto* p : blocks) {
            ::operator delete(p);
        }

        benchmark::DoNotOptimize(blocks.data());
    }

    state.SetItemsProcessed(state.iterations() * ops_per_iter * 2);  // alloc + dealloc
}
BENCHMARK(BM_Alloc_NewDelete)
    ->Arg(1000)->Arg(10000)->Arg(100000);

/// Benchmark: slab allocator for fixed-size blocks (64 bytes each).
static void BM_Alloc_Slab(benchmark::State& state) {
    const std::size_t block_size = 64;
    const std::size_t ops_per_iter = state.range(0);

    for (auto _ : state) {
        slab_allocator alloc(block_size, 65536, 4, 1024);

        std::vector<void*> blocks;
        blocks.reserve(ops_per_iter);

        // Allocate
        for (std::size_t i = 0; i < ops_per_iter; ++i) {
            void* p = alloc.allocate();
            std::memset(p, 0, block_size);
            blocks.push_back(p);
        }

        // Deallocate (return to slab free list)
        for (auto* p : blocks) {
            alloc.deallocate(p);
        }

        benchmark::DoNotOptimize(blocks.data());
    }

    state.SetItemsProcessed(state.iterations() * ops_per_iter * 2);
}
BENCHMARK(BM_Alloc_Slab)
    ->Arg(1000)->Arg(10000)->Arg(100000);

/// Multi-threaded slab allocation benchmark.
static void BM_Alloc_Slab_MultiThread(benchmark::State& state) {
    const std::size_t block_size = 64;
    const int num_threads = static_cast<int>(state.range(0));
    const std::size_t ops_per_thread = 10000;

    for (auto _ : state) {
        state.PauseTiming();
        slab_allocator alloc(block_size, 65536, 8, 1024);
        state.ResumeTiming();

        std::vector<std::thread> threads;
        threads.reserve(num_threads);
        for (int t = 0; t < num_threads; ++t) {
            threads.emplace_back([&alloc, ops_per_thread]() {
                std::vector<void*> blocks;
                blocks.reserve(ops_per_thread);
                for (std::size_t i = 0; i < ops_per_thread; ++i) {
                    void* p = alloc.allocate();
                    std::memset(p, 0, 64);
                    blocks.push_back(p);
                }
                for (auto* p : blocks) {
                    alloc.deallocate(p);
                }
            });
        }
        for (auto& t : threads) t.join();
    }

    state.SetItemsProcessed(state.iterations() * ops_per_thread * num_threads * 2);
}
BENCHMARK(BM_Alloc_Slab_MultiThread)
    ->Arg(1)->Arg(2)->Arg(4)->Arg(8)
    ->UseRealTime();

/// Multi-threaded new/delete benchmark for comparison.
static void BM_Alloc_NewDelete_MultiThread(benchmark::State& state) {
    const int num_threads = static_cast<int>(state.range(0));
    const std::size_t ops_per_thread = 10000;

    for (auto _ : state) {
        std::vector<std::thread> threads;
        threads.reserve(num_threads);
        for (int t = 0; t < num_threads; ++t) {
            threads.emplace_back([ops_per_thread]() {
                std::vector<void*> blocks;
                blocks.reserve(ops_per_thread);
                for (std::size_t i = 0; i < ops_per_thread; ++i) {
                    void* p = ::operator new(64);
                    std::memset(p, 0, 64);
                    blocks.push_back(p);
                }
                for (auto* p : blocks) {
                    ::operator delete(p);
                }
            });
        }
        for (auto& t : threads) t.join();
    }

    state.SetItemsProcessed(state.iterations() * ops_per_thread * num_threads * 2);
}
BENCHMARK(BM_Alloc_NewDelete_MultiThread)
    ->Arg(1)->Arg(2)->Arg(4)->Arg(8)
    ->UseRealTime();

// ============================================================================
// 5. Scalability chart
//    safe_cache and striped_cache with increasing thread counts (1,2,4,8,16)
//    under 99% read workload. These are the same as section 1 but with
//    explicit label for clarity and additional thread counts.
// ============================================================================

template <typename Cache>
static void BM_Scalability_Generic(benchmark::State& state) {
    const std::size_t capacity = 100000;
    const int num_threads = static_cast<int>(state.range(0));
    const std::size_t ops_per_thread = 100000;  // more ops for stable measurement

    for (auto _ : state) {
        state.PauseTiming();
        Cache cache(capacity);
        for (int i = 0; i < static_cast<int>(capacity); ++i) {
            cache.set(i, i);
        }
        auto keys = random_keys(capacity, static_cast<int>(capacity));
        std::atomic<std::size_t> total_hits{0};
        state.ResumeTiming();

        std::vector<std::thread> threads;
        threads.reserve(num_threads);
        for (int t = 0; t < num_threads; ++t) {
            threads.emplace_back(read_heavy_worker<Cache>,
                std::ref(cache), std::cref(keys),
                ops_per_thread, 0.99, std::ref(total_hits));
        }
        for (auto& t : threads) t.join();

        benchmark::DoNotOptimize(total_hits.load());
    }

    state.SetItemsProcessed(state.iterations() * ops_per_thread * num_threads);
}

static void BM_Scalability_SafeCache(benchmark::State& state) {
    BM_Scalability_Generic<lru::safe_cache<int, int>>(state);
}
BENCHMARK(BM_Scalability_SafeCache)
    ->Arg(1)->Arg(2)->Arg(4)->Arg(8)->Arg(16)
    ->Arg(32)->Arg(64)->Arg(128)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

static void BM_Scalability_StripedCache(benchmark::State& state) {
    BM_Scalability_Generic<lru::striped_cache<int, int>>(state);
}
BENCHMARK(BM_Scalability_StripedCache)
    ->Arg(1)->Arg(2)->Arg(4)->Arg(8)->Arg(16)
    ->Arg(32)->Arg(64)->Arg(128)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

// ============================================================================
// 6. P99 tail latency under contention (T-M3)
//    16 threads, 99% read workload. Every 1000 ops, a background thread
//    samples P50/P95/P99 from stats_snapshot().get_latency. At the end,
//    reports the latency curve as custom counters so CI can track
//    regressions vs baseline.
//
//    Covers: safe_cache, striped_cache, read_heavy_striped_cache.
//    This is the benchmark the spec (M-3) mandates for tail-latency
//    observability under read-heavy contention.
// ============================================================================

#include <chrono>

/// Per-thread worker that performs 99% read / 1% write operations and
/// periodically records the cache's get_latency percentiles. The percentiles
/// are read from stats_snapshot() (lock-free histogram reads) so sampling
/// does not perturb the workload.
template <typename Cache>
static void tail_latency_worker(
    Cache& cache,
    const std::vector<int>& keys,
    std::size_t ops_count,
    std::atomic<bool>& stop,
    std::vector<std::array<std::uint64_t, 3>>& samples,  // [p50, p95, p99] per sample
    std::atomic<std::size_t>& sample_idx,
    std::atomic<std::size_t>& total_hits)
{
    std::mt19937 rng(static_cast<std::uint32_t>(
        std::hash<std::thread::id>{}(std::this_thread::get_id())));
    std::uniform_real_distribution<double> coin(0.0, 1.0);
    std::uniform_int_distribution<int> key_dist(0, static_cast<int>(keys.size() - 1));
    std::uniform_int_distribution<int> write_key_dist(
        static_cast<int>(keys.size()), static_cast<int>(keys.size() * 2));

    std::size_t local_hits = 0;
    for (std::size_t i = 0; i < ops_count && !stop.load(std::memory_order_relaxed); ++i) {
        if (coin(rng) < 0.99) {
            auto k = keys[key_dist(rng)];
            auto h = cache.get(k);
            if (h) ++local_hits;
        } else {
            auto k = write_key_dist(rng);
            cache.set(k, k);
        }

        // Every 1000 ops, one thread (via atomic CAS on sample_idx) records
        // the current latency percentiles. This avoids per-thread sampling
        // overhead and ensures samples are spread across the timeline.
        if ((i & 0x3FFu) == 0) {
            std::size_t idx = sample_idx.fetch_add(1, std::memory_order_relaxed);
            if (idx < samples.size()) {
                auto snap = cache.stats_snapshot();
                samples[idx] = {
                    snap.get_latency.percentile(0.50),
                    snap.get_latency.percentile(0.95),
                    snap.get_latency.percentile(0.99)
                };
            }
        }
    }
    total_hits.fetch_add(local_hits, std::memory_order_relaxed);
}

/// Generic P99 tail latency benchmark. Runs `num_threads` workers for
/// `ops_per_thread` operations, collecting up to `max_samples` latency
/// snapshots. Reports P50/P95/P99 as custom counters (last sample +
/// max observed).
template <typename Cache>
static void BM_TailLatency_Generic(benchmark::State& state) {
    const std::size_t capacity = 100000;
    const int num_threads = static_cast<int>(state.range(0));
    const std::size_t ops_per_thread = 50000;
    constexpr std::size_t max_samples = 1024;

    for (auto _ : state) {
        state.PauseTiming();
        Cache cache(capacity);
        for (int i = 0; i < static_cast<int>(capacity); ++i) {
            cache.set(i, i);
        }
        auto keys = random_keys(capacity, static_cast<int>(capacity));

        std::vector<std::array<std::uint64_t, 3>> samples(max_samples, {0, 0, 0});
        std::atomic<std::size_t> sample_idx{0};
        std::atomic<bool> stop{false};
        std::atomic<std::size_t> total_hits{0};
        state.ResumeTiming();

        std::vector<std::thread> threads;
        threads.reserve(num_threads);
        for (int t = 0; t < num_threads; ++t) {
            threads.emplace_back(tail_latency_worker<Cache>,
                std::ref(cache), std::cref(keys),
                ops_per_thread, std::ref(stop),
                std::ref(samples), std::ref(sample_idx),
                std::ref(total_hits));
        }
        for (auto& t : threads) t.join();

        // Compute max P99 across all samples for regression detection.
        std::uint64_t max_p50 = 0, max_p95 = 0, max_p99 = 0;
        std::uint64_t last_p50 = 0, last_p95 = 0, last_p99 = 0;
        std::size_t actual_samples = std::min(sample_idx.load(), max_samples);
        for (std::size_t i = 0; i < actual_samples; ++i) {
            max_p50 = std::max(max_p50, samples[i][0]);
            max_p95 = std::max(max_p95, samples[i][1]);
            max_p99 = std::max(max_p99, samples[i][2]);
            last_p50 = samples[i][0];
            last_p95 = samples[i][1];
            last_p99 = samples[i][2];
        }

        // Report as custom counters for CI regression tracking.
        state.counters["p50_last_ns"] = static_cast<double>(last_p50);
        state.counters["p95_last_ns"] = static_cast<double>(last_p95);
        state.counters["p99_last_ns"] = static_cast<double>(last_p99);
        state.counters["p50_max_ns"] = static_cast<double>(max_p50);
        state.counters["p95_max_ns"] = static_cast<double>(max_p95);
        state.counters["p99_max_ns"] = static_cast<double>(max_p99);
        state.counters["samples"] = static_cast<double>(actual_samples);

        benchmark::DoNotOptimize(total_hits.load());
    }

    state.SetItemsProcessed(state.iterations() * ops_per_thread * num_threads);
}

static void BM_TailLatency_SafeCache(benchmark::State& state) {
    BM_TailLatency_Generic<lru::safe_cache<int, int>>(state);
}
BENCHMARK(BM_TailLatency_SafeCache)
    ->Arg(16)->Arg(32)->Arg(64)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

static void BM_TailLatency_StripedCache(benchmark::State& state) {
    BM_TailLatency_Generic<lru::striped_cache<int, int>>(state);
}
BENCHMARK(BM_TailLatency_StripedCache)
    ->Arg(16)->Arg(32)->Arg(64)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

static void BM_TailLatency_ReadHeavyStriped(benchmark::State& state) {
    BM_TailLatency_Generic<lru::read_heavy_striped_cache<int, int>>(state);
}
BENCHMARK(BM_TailLatency_ReadHeavyStriped)
    ->Arg(16)->Arg(32)->Arg(64)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

// ============================================================================
// 7. EBR vs hazptr comparison (P0-1.2)
//    Compares two striped_cache instances under 32/64 threads 99% read:
//      - hazptr mode (default)
//      - EBR mode    (set_ebr_domain(&default_domain()))
//    Reports throughput and reclaim_pending_count as custom counters so
//    CI can quantify the EBR fast-path advantage documented in AGENTS.md.
// ============================================================================
static void BM_EBR_vs_Hazptr_ReadHeavy32(benchmark::State& state) {
    const std::size_t capacity = 100000;
    const int num_threads = static_cast<int>(state.range(0));
    const std::size_t ops_per_thread = 50000;
    const bool use_ebr = (state.range(1) == 1);

    for (auto _ : state) {
        state.PauseTiming();
        lru::striped_cache<int, int> cache(capacity);
        if (use_ebr) {
            cache.set_ebr_domain(&lru::detail::epoch_domain::default_domain());
        }
        for (int i = 0; i < static_cast<int>(capacity); ++i) {
            cache.set(i, i);
        }
        auto keys = random_keys(capacity, static_cast<int>(capacity));
        std::atomic<std::size_t> total_hits{0};
        state.ResumeTiming();

        std::vector<std::thread> threads;
        threads.reserve(num_threads);
        for (int t = 0; t < num_threads; ++t) {
            threads.emplace_back(read_heavy_worker<lru::striped_cache<int, int>>,
                std::ref(cache), std::cref(keys),
                ops_per_thread, 0.99, std::ref(total_hits));
        }
        for (auto& t : threads) t.join();

        benchmark::DoNotOptimize(total_hits.load());
    }

    // Report reclamation mode for CI tracking.
    state.counters["mode_ebr"] = use_ebr ? 1.0 : 0.0;
    state.SetItemsProcessed(state.iterations() * ops_per_thread * num_threads);
}
BENCHMARK(BM_EBR_vs_Hazptr_ReadHeavy32)
    ->Args({32, 0})->Args({32, 1})
    ->Args({64, 0})->Args({64, 1})
    ->UseRealTime();

// ============================================================================
// 8. Production alias 32/64 thread benchmark (P0-1.3)
//    Quantifies throughput of the recommended production aliases under
//    32/64-thread 99% read workloads — the scenario the spec (P0-1)
//    identifies as the production target.
// ============================================================================
static void BM_ProductionCache_ReadHeavy32(benchmark::State& state) {
    BM_ReadHeavy_Generic<lru::production_cache<int, int>>(state);
}
BENCHMARK(BM_ProductionCache_ReadHeavy32)
    ->Arg(32)->Arg(64)
    ->UseRealTime();

static void BM_SegmentedCache_ReadHeavy32(benchmark::State& state) {
    BM_ReadHeavy_Generic<lru::segmented_cache<int, int>>(state);
}
BENCHMARK(BM_SegmentedCache_ReadHeavy32)
    ->Arg(32)->Arg(64)
    ->UseRealTime();

static void BM_F14ProductionCache_ReadHeavy32(benchmark::State& state) {
    BM_ReadHeavy_Generic<lru::f14_production_cache<int, int>>(state);
}
BENCHMARK(BM_F14ProductionCache_ReadHeavy32)
    ->Arg(32)->Arg(64)
    ->UseRealTime();

// ============================================================================
// Main
// ============================================================================

BENCHMARK_MAIN();
