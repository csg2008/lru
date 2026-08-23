# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Header-only C++20 LRU cache library (v4.0.0), inspired by Facebook's CacheLib architecture. It provides multiple eviction strategies (LRU, 2Q, TinyLFU, W-TinyLFU, FIFO), thread-safe variants, TTL support, serialization, memory monitoring, slab allocation, compact cache, compressed pointers, tiered storage, warm cache, shared memory backend, pooled cache, and related utilities.

## Build, test, and run (Windows MSYS2 Clang64)

The local development workflow assumes the MSYS2 Clang64 toolchain at `E:\app\msys64`. Run build commands inside `E:\app\msys64\usr\bin\bash.exe` with the Clang64 toolchain on `PATH`.

- ccache is available; set `CC`/`CXX` to use it.
- Prefer Debug builds for iteration speed.
- Build with `mingw32-make -j2` (two parallel jobs).
- Do not auto-install missing tools or libraries; output a prompt instead.
- **Clang64 is required** (not UCRT64): MSYS2 UCRT64 GCC lacks `libasan` runtime, so AddressSanitizer is unavailable. Clang64 provides both ASan and superior diagnostics.

### Configure and build tests

```bash
export PATH=/clang64/bin:/usr/bin:$PATH
export CC="ccache clang"
export CXX="ccache clang++"

cmake -B build -G "MinGW Makefiles" \
    -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang \
    -DCMAKE_BUILD_TYPE=Debug \
    -DLRU_BUILD_TESTS=ON \
    -DLRU_BUILD_EXAMPLES=OFF

mingw32-make -C build -j2
```

### Run all tests

```bash
ctest --test-dir build --output-on-failure
```

### Run a single test

With `ctest` (matches test names by regex):

```bash
ctest --test-dir build -R <test_name_pattern>
```

With the GoogleTest executable directly (most precise, supports `*` wildcards):

```bash
./build/tests/lru_cache_test.exe --gtest_filter=<TestSuite>.<TestName>
./build/tests/lru_refcount_test.exe --gtest_filter=<TestSuite>.<TestName>
```

### Build examples

```bash
cmake -B build -G "MinGW Makefiles" \
    -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang \
    -DCMAKE_BUILD_TYPE=Debug \
    -DLRU_BUILD_TESTS=OFF

mingw32-make -C build -j2
```

Built executables are in `build/examples/` (`basic_example.exe`, `threadsafe_example.exe`, `callbacks_example.exe`, `ttl_example.exe`).

### Build benchmarks

```bash
cmake -B build/bench -G "MinGW Makefiles" \
    -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang \
    -DCMAKE_BUILD_TYPE=Release \
    -DLRU_BUILD_TESTS=OFF \
    -DLRU_BUILD_EXAMPLES=OFF \
    -DLRU_BUILD_BENCHMARKS=ON

mingw32-make -C build/bench -j2
```

Benchmark executables are in `build/bench/benchmarks/` (`lru_cache_benchmark.exe`, `lru_concurrent_read_benchmark.exe`).

### Sanitizer builds

The project supports four sanitizers (ASan, TSan, UBSan, LSan) with CMake-enforced
composition rules. The `scripts/build.sh` and `scripts/test.sh` wrappers handle
environment setup, runtime options, and build directory conventions automatically.

**Composition matrix** (enforced by `CMakeLists.txt`):

| Profile | Flags | Platform | Notes |
|---|---|---|---|
| `asan` | `-fsanitize=address` | Clang64 / Linux | Includes LSan on Linux; MinGW LSan unavailable |
| `tsan` | `-fsanitize=thread` | Linux only | Mutually exclusive with ASan/LSan |
| `ubsan` | `-fsanitize=undefined -fno-sanitize-recover=undefined` | All | UB is fatal (abort on first UB) |
| `lsan` | `-fsanitize=leak` | Linux only | Standalone; redundant with ASan (auto-disabled) |
| `asan-ubsan` | `-fsanitize=address,undefined` | Clang64 / Linux | Combined; both fatal |

**Via scripts (recommended):**

```bash
# ASan (Clang64 only — UCRT64 GCC lacks libasan)
./scripts/build.sh asan && ./scripts/test.sh asan

# UBSan (undefined behavior; works on all platforms)
./scripts/build.sh ubsan && ./scripts/test.sh ubsan

# ASan + UBSan combined (catches both memory and UB errors)
./scripts/build.sh asan-ubsan && ./scripts/test.sh asan-ubsan

# TSan (Linux only — not available on MinGW)
./scripts/build.sh tsan && ./scripts/test.sh tsan

# LSan standalone (Linux only — on MinGW use ASan instead)
./scripts/build.sh lsan && ./scripts/test.sh lsan
```

**Via raw CMake (advanced):**

```bash
# AddressSanitizer
cmake -B build/asan -G "MinGW Makefiles" \
    -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang \
    -DCMAKE_BUILD_TYPE=Debug \
    -DLRU_BUILD_TESTS=ON -DLRU_BUILD_EXAMPLES=OFF -DLRU_ENABLE_ASAN=ON
mingw32-make -C build/asan -j2

# UBSan (undefined behavior sanitizer)
cmake -B build/ubsan -G "MinGW Makefiles" \
    -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang \
    -DCMAKE_BUILD_TYPE=Debug \
    -DLRU_BUILD_TESTS=ON -DLRU_BUILD_EXAMPLES=OFF -DLRU_ENABLE_UBSAN=ON
mingw32-make -C build/ubsan -j2

# ASan + UBSan combined
cmake -B build/asan-ubsan -G "MinGW Makefiles" \
    -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang \
    -DCMAKE_BUILD_TYPE=Debug \
    -DLRU_BUILD_TESTS=ON -DLRU_BUILD_EXAMPLES=OFF \
    -DLRU_ENABLE_ASAN=ON -DLRU_ENABLE_UBSAN=ON
mingw32-make -C build/asan-ubsan -j2

# ThreadSanitizer (not available on MinGW — requires Linux)
cmake -B build/tsan -G "MinGW Makefiles" \
    -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang \
    -DCMAKE_BUILD_TYPE=Debug \
    -DLRU_BUILD_TESTS=ON -DLRU_BUILD_EXAMPLES=OFF -DLRU_ENABLE_TSAN=ON
mingw32-make -C build/tsan -j2
```

**Runtime environment defaults** (set by `scripts/test.sh`, overridable via env):

| Profile | Variable | Default | Meaning |
|---|---|---|---|
| `asan` / `asan-ubsan` | `ASAN_OPTIONS` | `detect_leaks=0:abort_on_error=1:halt_on_error=0:print_stacktrace=1` | MinGW: LSan unavailable so `detect_leaks=0`; on Linux set `detect_leaks=1` |
| `tsan` | `TSAN_OPTIONS` | `halt_on_error=0:second_deadlock_stack=1:report_bugs=1` | Continue after first race to report all findings |
| `ubsan` / `asan-ubsan` | `UBSAN_OPTIONS` | `print_stacktrace=1:halt_on_error=0` | UB is already fatal via `-fno-sanitize-recover=undefined` compile flag |
| `lsan` | `LSAN_OPTIONS` | `exitcode=23:report_objects=1` | Exit 23 = leak (distinct from gtest's exit 1) |

**Sanitizer exit codes** (surfaced by `scripts/test.sh` on failure):

| Exit code | Sanitizer | Meaning |
|---|---|---|
| 1 | ASan / UBSan / asan-ubsan | Memory error or undefined behavior |
| 23 | LSan | Memory leak detected |
| 66 | TSan | Data race or deadlock |

**Cleaning sanitizer build directories:**

```bash
./scripts/clean.sh sanitizers   # Remove only build/asan, build/tsan, build/ubsan, build/lsan, build/asan-ubsan
./scripts/clean.sh              # Remove all build directories
```

### Debug instrumentation

```bash
# Enable per-T global handle counter (debug only, adds cache-line contention)
cmake -B build ... -DLRU_DEBUG=ON
```

### Optional C++23 features

```bash
cmake -B build ... -DLRU_ENABLE_CPP23=ON
```

## Dependencies

- **C++20** compiler (clang++/g++/MSVC). The local workflow is Clang via MSYS2 Clang64.
- **ankerl/unordered_dense**: CMake calls `find_package(unordered_dense)` first. If not found, it falls back to a sibling checkout at `../unordered_dense/include/ankerl/unordered_dense.h`.
- **GoogleTest** is fetched automatically via FetchContent when `-DLRU_BUILD_TESTS=ON`.
- **Google Benchmark** is fetched automatically when `-DLRU_BUILD_BENCHMARKS=ON`.

## Architecture

The library is header-only. The key headers and their roles:

- `lru.hpp` — Single include that pulls in everything and defines convenience aliases (`cache`, `safe_cache`, `striped_cache`, `production_cache`, `f14_production_cache`, `lfu_cache`, `w_tiny_lfu`, `two_q`, `fifo_cache`, read-heavy factories, etc.).
- `core.hpp` — Concepts, `callback_manager`, `cache_stats`, iterators, `read_handle`, default policies, C++23 feature detection.
- `cache_trait.hpp` — Compile-time composition of an MM (eviction) strategy and a lock policy into `unified_cache<Trait, K, V>`. Includes lock policies (`single_threaded_policy`, `thread_safe_policy`, `striped_thread_safe_policy`).
- `mm.hpp` — Eviction strategy implementations: `mm_lru`, `mm_2q`, `mm_tiny_lfu`, `mm_wtiny_lfu`, `mm_fifo`, and `sharded_mm_lru`. Includes slab allocator mixin, overflow policy, access mode.
- `detail/foundation.hpp` — Internal utilities: type traits helpers, string formatting, periodic worker, striped mutex, integer sequence helpers.
- `detail/refcount.hpp` — CAS-lockfree reference counting with embedded flags (64-bit atomic word: 5 flags + 3 admin_ref + 32 access_ref). Overflow is prevented via check-and-refuse: `incRef()` returns `kIncFailedOverflow` when `access_ref` reaches `kAccessRefMax`, letting callers (e.g. `read_handle` ctor) produce an empty handle instead of saturating silently.
- `detail/hazptr.hpp` — Lightweight hazard pointer mechanism for deferred reclamation. v4.2: lock-free retire path, TLS slot cache, hazptr_obj_base for zero-alloc retirement.
- `detail/epoch_reclamation.hpp` — Epoch-based reclamation (EBR), faster read-path than hazard pointers. Lock-free retire, TLS slot caching, compatible with hazptr_obj_base.
- `detail/latency_histogram.hpp` — Log-linear latency histogram with 512 lock-free atomic buckets. 16 sub-buckets per power-of-2 octave, ≤6.25% precision.
- `detail/intrusive_list.hpp` — Intrusive doubly-linked list used by all MM strategies; nodes embed prev/next/updateTime/refcount hooks. Supports compressed pointer hooks.
- `detail/count_min_sketch.hpp` — Count-Min Sketch frequency estimator for TinyLFU/W-TinyLFU.
- `detail/distributed_mutex.hpp` — Custom shared mutex used instead of `std::shared_mutex` on MinGW to avoid `pthread_rwlock_t` bugs. Default fairness mode is `writer_fair`; runtime-switchable to `reader_preferred`. Includes lock wait latency histograms, try-lock failure counters, and debug lock order validation (`LRU_DEBUG_LOCK_ORDER`).
- `detail/concurrent_hash_table.hpp` — Concurrent hash table backing the MM maps. Supports chain mode and F14 SIMD probing. Non-EmbeddedChain mode uses shared lock fallback (not lock-free reads) to prevent use-after-free. Optional incremental rehash (chain mode and F14 mode).
- `detail/native_wait_ops.hpp` — Native wait/wake wrappers (runtime-selected): Windows WaitOnAddress, Linux futex, macOS ulock, CV fallback.
- `detail/space_saving.hpp` — Space-Saving Top-K streaming heavy-hitter detection (Metwally et al., 2005). O(1) amortized update, O(K log K) query.
- `ttl.hpp`, `memory.hpp` (slab allocator + memory monitor), `admission.hpp`, `serialization.hpp`, `chained_item.hpp`, `compact_cache.hpp`, `compressed_ptr.hpp`, `event_tracker.hpp`, `event_types.hpp`, `pooled_cache.hpp`, `shared_memory_backend.hpp`, `tiered_storage.hpp`, `tls_ring.hpp`, `warm_cache.hpp` — Specialized subsystems.

### Strategy composition

`unified_cache` is parameterized by a `cache_trait<MMType, LockPolicy, ProbingStyle, Segmented>`:

- **MM types**: `mm_lru`, `mm_2q`, `mm_tiny_lfu`, `mm_wtiny_lfu`, `mm_fifo`, `sharded_mm_lru`.
- **Lock policies**:
  - `single_threaded_policy` — no-op locks, zero overhead.
  - `thread_safe_policy` — `distributed_shared_mutex` (shared reads, exclusive writes).
  - `striped_thread_safe_policy<N>` — N stripes (default 64) of `distributed_shared_mutex`, keyed by hash.
- **Probing styles**:
  - `chain_probing_tag` — traditional chaining with overflow list.
  - `f14_probing_tag` — F14 SIMD probing with 14-slot chunks and 8-bit hash tags (SSE2/NEON/scalar fallback).
- **Segmented**: when true, uses `segmented_concurrent_hash_table` (64 segments) for per-segment rehash with no global stall.

Convenience aliases wire common combinations:
- `lru::cache<K,V>` — single-threaded LRU (chain probing)
- `lru::safe_cache<K,V>` — thread-safe LRU (single global mutex)
- `lru::striped_cache<K,V>` — striped sharded LRU
- `lru::production_cache<K,V>` — **recommended for production**: segmented hash table + sharded MM + striped locking
- `lru::f14_cache<K,V>` — single-threaded LRU with F14 SIMD probing
- `lru::f14_striped_cache<K,V>` — striped sharded LRU with F14
- `lru::f14_production_cache<K,V>` — F14 sharded + striped locking
- `lru::segmented_cache<K,V>` / `segmented_striped_cache<K,V>` — segmented hash table variants
- `lru::read_heavy_cache<K,V>` / `read_heavy_striped_cache<K,V>` / `read_heavy_w_tiny_lfu<K,V>` — pre-configured for read-heavy workloads (defer_promotion=true + EBR on LRU variants, applied via `read_heavy_*_trait`)
- `lru::lfu_cache`, `w_tiny_lfu`, `two_q`, `fifo_cache` — single-threaded other eviction strategies
- `lru::safe_lfu_cache`, `safe_w_tiny_lfu`, `safe_two_q`, `safe_fifo_cache` — thread-safe variants (single global lock) of the above
- F14 and segmented variants exist for all eviction strategies.
- **Note (R-1 / R-4)**: The historical `make_safe_cache` / `make_striped_cache` / `make_read_heavy_*` factory functions have been removed; their behavior is now provided by the corresponding type aliases via trait-layer opt-in. Direct construction (e.g. `lru::safe_cache<K,V> c(n);`) is the only way to obtain these caches.

### Concurrency notes (hard constraints)

- **Atomic variables for cross-thread state**: Concurrent data structures must use atomic operations for thread-shared variables (e.g., `head_`/`tail_` in `tls_ring.hpp` must be `std::atomic`).
- **Non-EmbeddedChain shared lock fallback (R-3 / T-P1-3)**: `concurrent_hash_table.hpp` non-EmbeddedChain mode must use shared locks instead of lock-free reads to prevent use-after-free. **For read-heavy-write-light high-concurrency workloads (32+ threads, 99%+ reads), EmbeddedChain = true is a hard requirement** — non-EmbeddedChain mode degrades all lock-free read paths to shared_lock fallback, killing throughput under read contention. All production aliases (`production_cache`, `segmented_*`, `f14_*`, `safe_cache`, `striped_cache`) and all MM strategies in `mm.hpp` statically assert `map_type::uses_embedded_chain == true` at compile time. **DO NOT bypass this assertion via custom traits** — the resulting cache will appear to work but suffer 5-10x read throughput regression under 32+ thread read contention. Non-EmbeddedChain mode is acceptable only for write-heavy or low-concurrency scenarios where shared lock contention is negligible. The `diagnostics()` / `diagnostics_text()` API exposes `embedded_chain: 1|0` so operators can verify at runtime that the active cache is using lock-free reads.
- **Refcount overflow protection**: `refcount.hpp` `incRef()` uses check-and-refuse (returns `kIncFailedOverflow` when `access_ref` reaches `kAccessRefMax = 2^32 - 1`) instead of saturated addition. Callers must handle the failure (e.g. `read_handle` ctor produces an empty handle).
- Do not replace `detail::distributed_shared_mutex` with `std::shared_mutex` on Windows/MinGW; the MinGW `pthread_rwlock_t` implementation returns `EINVAL` under high mixed read/write contention.
- `distributed_shared_mutex` defaults to `writer_fair` fairness (prevents writer starvation). Use `set_fairness_mode()` on `unified_cache` to switch to `reader_preferred` if maximum read throughput is needed and write latency is acceptable.
- `striped_thread_safe_policy` is designed for `sharded_mm_lru`: different keys can hash to different shards and be accessed concurrently. The stripe count is runtime-configurable via the `(max_size, num_stripes)` constructor.
- **Shard hash distribution (R-9 / T-P2-5)**: `sharded_mm_lru::shard_for(key) = Hash{}(key) % num_shards_` deliberately does NOT apply additional hash mixing (fibonacci, splitmix, etc.) on top of the caller-supplied `Hash`. Rationale: when `max_size < num_shards`, `distribute_max_size()` grants only the first `max_size` shards a quota of 1 (the rest get 0); with identity hash (`std::hash<int>{}(i) == i`), keys 0..N-1 land exactly on the shards with capacity, while mixing would redistribute keys onto 0-capacity shards and silently evict/reject every insert. The shard layer must not second-guess the caller's hash function. **For workloads with poorly-distributed keys (e.g. sequential integers, monotonic IDs, low-entropy keys), callers MUST supply a well-mixed hash** — e.g. `ankerl::unordered_dense::hash`, `absl::Hash`, `std::hash` combined with a seed via `hash_combine`, or `splitmix64`. The default `std::hash<int>` is identity and will route sequential integer keys to consecutive shards, producing hot shards under skewed access. Verify even distribution via `hot_shards(n)` / `hot_shards_by_memory(n)`; if the top shard holds significantly more than `1/num_shards` of items, switch to a well-mixed hash.
- `single_threaded_policy` is truly zero-overhead.
- `defer_promotion` defaults to `true` in `mm_lru`/`sharded_mm_lru`: `get()` hits do not acquire the write lock for LRU promotion; accesses are batched in a TLS ring and drained later. This significantly reduces read-path lock pressure in read-heavy workloads.
- Incremental rehash (chain mode, F14 mode, and segmented mode): `set_incremental_rehash(true)` makes hash table expansion migrate buckets incrementally across multiple operations instead of blocking all writers during a single rehash. F14 mode implements incremental rehash via dual-array lookup (reads query both old and new arrays during migration; writes route by progress boundary). Segmented mode applies incremental rehash independently per segment (1/64 stall at any moment). T11.1: `segmented_*` and `production_*` aliases auto-enable incremental rehash on construction.
- Segmented hash table: 64 independent segments, each with its own bucket array, locks, and rehash state. Rehash only locks one segment (1/64 of the table), eliminating global stalls during hash table growth.
- T19.3 / T-P1-3: `diagnostics()` / `diagnostics_text()` report `f14_probing` / `segmented_hash_table` / `compressed_hook` / `embedded_chain` / `rehash_mode` so operators can identify the active hash table mode in a single dump.

### Production-grade APIs (read-heavy-write-light workloads)

`unified_cache` exposes the following APIs designed for production read-heavy scenarios:

- **Lock fairness**: `set_fairness_mode(mode)` / `get_fairness_mode()` — runtime switch between `writer_fair` (default) and `reader_preferred`.
- **Deferred promotion**: `set_defer_promotion(bool)` / `is_defer_promotion_enabled()` — toggle TLS-batched LRU promotion (default on).
- **Runtime stripe count**: `unified_cache(max_size, num_stripes)` constructor.
- **Incremental rehash**: `set_incremental_rehash(bool)` / `incremental_rehash_enabled()`.
- **EBR read path (R-6 / T-P1-4)**: `set_ebr_domain(&detail::epoch_domain::default_domain())` — switches the underlying `mm_lru`/`sharded_mm_lru` from hazptr-based deferred reclamation to Epoch-Based Reclamation. EBR's read path is faster than hazptr under sustained read contention (TLS epoch guard vs. per-node hazptr acquire/release), making it the recommended mode for read-heavy-write-light production workloads (32+ threads, 99%+ reads). EBR and hazptr coexist — evicted items are retired via the active domain (EBR when set, hazptr otherwise), and both are drained by the background `event_drain_worker`. The `read_heavy_cache` / `read_heavy_striped_cache` / `production_cache` / `f14_production_cache` aliases enable EBR by default (via their trait setting `auto_enable_ebr = true`); users of `safe_cache` / `striped_cache` aliases can opt in by calling `c.set_ebr_domain(&lru::detail::epoch_domain::default_domain())` after construction. `is_ebr_mode()` queries the active mode. Note: only `mm_lru` and `sharded_mm_lru` support EBR; calling `set_ebr_domain` on `mm_2q`/`mm_tiny_lfu`/`mm_wtiny_lfu`/`mm_fifo` is a guarded no-op.
- **Non-throwing reads**: `try_get(key)` / `try_get_or_fetch(key, provider)`.
- **Bulk reads**: `bulk_get(keys)` / `bulk_try_get(keys)` — batch multiple key lookups.
- **TTL-aware read**: `get_with_ttl(key)` returns handle + remaining TTL.
- **Atomic update**: `cas(key, expected, desired)` / `cas(key, predicate, desired)`.
- **Observability** (`stats_snapshot()`):
  - `get_latency` / `set_latency` histograms (P50/P95/P99 via `percentile()`)
  - `read_lock_wait_latency` / `write_lock_wait_latency` histograms
  - `read_lock_wait_total` / `write_lock_wait_total` counters
  - `read_trylock_failures` / `write_trylock_failures` counters
  - `eviction_search_steps` counter
  - `ttl_expired_count` / `ttl_checked_count` counters
  - `pinned_skip_count` counter
  - `active_handle_count`, `tls_ring_backlog`
- **Prometheus export**: `prometheus_text()` exports Prometheus exposition format with all metrics above.
- **Background TTL cleanup**: `start_ttl_cleaner(interval)` / `stop_ttl_cleaner()` — per-shard round-robin, no global lock.
- **Async callbacks**: `set_async_callbacks(bool)` — drain thread enqueues, dedicated worker dequeues; prevents user callbacks (IO/logging) from blocking the drain path.
- **Background drain worker**: Production deployments **MUST** call `start_event_drain()`. Thread-safe aliases (`safe_cache` / `striped_cache` / `production_cache` / `read_heavy_*`) auto-start it at construction with a default 1s interval. When `active_handle_count() > 0` the worker self-adjusts to 500ms interval to accelerate retired-object reclamation; with no live handles it reverts to 1s to save CPU. Failing to call this causes hazptr/EBR retired objects to accumulate, eventually causing OOM.
- **Graceful shutdown**: `shutdown()` / `is_shutdown()` / `active_handle_count()` — rejects new ops, keeps existing `read_handle` valid until released.
- **Per-shard serialization**: `save_per_shard()` / `load_per_shard()` — each shard locked independently, avoids global stall.
- **Debug lock order validation**: define `LRU_DEBUG_LOCK_ORDER` to enable lock order tracking and deadlock detection (Debug builds only).

### Callbacks

`callback_manager` supports `on_hit`, `on_miss`, `on_insert`, and `on_evict`. To reduce lock contention, callbacks can be collected inside the critical section via `collect_*()` and flushed outside via `flush_pending()`. For striped caches, flush per-shard. Async mode (`set_async_callbacks(true)`) moves callback execution to a dedicated worker thread so user callbacks performing IO do not block the drain path.

### `read_handle`

`get()` returns a `read_handle<Value>` that pins the item via refcount, preventing eviction while the handle is alive. `peek()` returns a read-only view without promotion. `get_shared()` returns a `std::shared_ptr<Value>` copy and does not change LRU order. `try_get()` is the non-throwing variant returning `std::optional<read_handle<Value>>`.

### Specialized subsystems

- **Compact Cache** (`compact_cache.hpp`): Dense fixed-size slot storage for small items (key+value ≤ 64 bytes). Eliminates heap allocator metadata overhead. Up to 50-70% memory savings for small items.
- **Compressed Pointers** (`compressed_ptr.hpp`): 32-bit offset-based pointers reducing intrusive hook size from 16 to 8 bytes. `compressed_intrusive_hook`, `compressed_region` allocator. Saves ~76 MB for 10M items + better cache locality.
- **Slab Allocator** (`memory.hpp`): Custom slab allocator for cache items and hash table nodes. Reduces allocation overhead and memory fragmentation. Configurable via `alloc_fn`/`dealloc_fn` in MM config or `set_hash_alloc_fns()`.
- **Pooled Cache** (`pooled_cache.hpp`): Multi-pool partitioning with independent MM strategies per pool. Pools share a global max_memory budget with weighted priority eviction. Multi-tenant isolation, priority tiers, dynamic resizing.
- **Tiered Storage** (`tiered_storage.hpp`): Primary in-memory cache + slower storage backend (SSD/database/remote). Read-through promotion, background warm worker, optional write-back on eviction. Backend I/O never blocks the primary cache lock.
- **Warm Cache** (`warm_cache.hpp`): `warm_cache_manager` for async snapshot loading with atomic swap. Incremental delta snapshots. Enables near-zero-downtime cache warm restarts.
- **Shared Memory Backend** (`shared_memory_backend.hpp`): Shared memory segment abstraction for warm restart. Windows (CreateFileMapping) / Linux (shm_open). Near-zero restart time by attaching to existing shared memory.
- **Event Tracker** (`event_tracker.hpp` + `event_types.hpp`): Lifecycle event tracking (insert/promote/demote/evict/hit). Uses Space-Saving Top-K for heavy-hitter detection.

## Common API patterns

- Insert/update: `set(key, value)`
- Read with promotion: `get(key)` — returns `std::optional<read_handle<Value>>`
- Non-throwing read: `try_get(key)` — returns `std::optional<read_handle<Value>>`, never throws
- Bulk read: `bulk_get(keys)` / `bulk_try_get(keys)`
- Read without promotion: `peek(key)` — returns `std::optional<reference_wrapper<const V>>`
- Get-or-fetch: `get_or_fetch(key, provider)` / `try_get_or_fetch(key, provider)`
- TTL-aware read: `get_with_ttl(key)` — returns handle + remaining TTL
- Atomic compare-and-swap: `cas(key, expected, desired)` / `cas(key, predicate, desired)`
- Insert only if missing: `add(key, value)`
- Replace only if present: `replace(key, value)`
- Delete with status: `remove(key)` → `RemoveRes::kSuccess` / `RemoveRes::kNotFound`
- Direct MM access: `c.mm()`; runtime configuration: `c.mm().set_config(...)`
- Pre-allocate: `c.reserve(expected_items)` — avoids runtime rehash stalls
- Graceful shutdown: `c.shutdown()` / `c.is_shutdown()` / `c.active_handle_count()`
- Metrics: `c.stats_snapshot()` / `c.prometheus_text()`
- Per-shard serialization: `c.save_per_shard()` / `c.load_per_shard()`
- Hash allocator: `c.set_hash_alloc_fns(alloc_fn, dealloc_fn)` — slab allocator for hash table nodes

## Engineering conventions

- Cache statistics (`cache_stats`) must include lock wait latency histograms (read/write), eviction search steps, and TTL metrics.
- Prometheus metrics for LRU cache must include lock wait totals, try-lock failures, pinned skips, TTL expired counts, and latency histograms.
- Debug builds must include lock order validation infrastructure when `LRU_DEBUG_LOCK_ORDER` is defined.

## Development requirements

1. 从资深 C++ 架构师角度思考如何让整体架构更优、性能更好、源码可读性更好
2. 如果一个问题有多个解决方案，需选择最佳实现而不是最简单实现
3. 尽量复用已有的工具类或函数实现，不要重复创造轮子
4. 问题标记完成时必须检查代码确认，如果未完成需要继续处理
5. 执行命令前需将 MSYS2 Clang64 工具链添加到 PATH 环境变量（`export PATH=/clang64/bin:/usr/bin:$PATH`）
6. 尽量以 Debug 模式编译测试，用 `mingw32-make -j2` 两进程并行编译
7. 编译时如果缺少工具或库不能自动安装，只能输出提示
8. 提交 git 时，消息前后不要添加无用的 @ 符号
9. 请用 bash 环境编译测试
