# LRU Cache Library v4.0.0

A high-performance, feature-rich LRU cache implementation for C++20, inspired by Facebook's CacheLib architecture.

## File Structure

```
lru/
  lru.hpp                        # 总入口（include 全部）
  core.hpp                       # concepts + traits + callbacks + statistics + iterator
  cache_trait.hpp                # 统一 trait 架构 + unified_cache 主模板
  mm.hpp                         # 全部 5 种淘汰策略 + sharded_mm_lru
  ttl.hpp                        # ttl_cache (P1-A: ttl_reaper removed, use detail::periodic_worker)
  memory.hpp                     # memory_monitor + slab_allocator + background_evictor
  admission.hpp                  # 准入策略
  chained_item.hpp               # 链式大值存储
  compressed_ptr.hpp             # 32 位压缩指针
  compact_cache.hpp              # 紧凑缓存（小对象优化）
  pooled_cache.hpp               # 多池分区缓存
  serialization.hpp              # saveState / restoreState
  event_tracker.hpp              # 生命周期事件追踪
  event_types.hpp                # 事件类型定义
  tiered_storage.hpp             # 分层存储（DRAM + SSD）
  tls_ring.hpp                   # 线程本地环形缓冲区（访问延迟提升 + 回调收集）
  warm_cache.hpp                 # 热缓存与快照恢复
  shared_memory_backend.hpp      # 共享内存后端（Windows/Linux）
  detail/
    foundation.hpp               # utils + periodic_worker + seqlock + striped_mutex
    intrusive_list.hpp           # 侵入式双链表 + refcount_flags
    count_min_sketch.hpp         # CountMinSketch 概率频率估计
    concurrent_hash_table.hpp    # 并发哈希表（乐观读 + hazptr + F14 SIMD + 分段 rehash）
    distributed_mutex.hpp        # writer_fair 默认 shared_mutex（WaitOnAddress/futex）
    epoch_reclamation.hpp        # Epoch-Based Reclamation（EBR）
    hazptr.hpp                   # Hazard Pointer 实现
    refcount.hpp                 # CAS 无锁引用计数（refcount_with_flags）
    latency_histogram.hpp        # 对数线性延迟直方图
    native_wait_ops.hpp          # 平台原生等待原语
    space_saving.hpp             # Space-Saving Top-K 重流检测
```

## Quick Start

```cpp
#include "lru.hpp"

int main() {
    using namespace lru;

    // 单线程 LRU 缓存
    cache<int, std::string> c(100);  // max 100 items
    c.set(1, "hello");
    auto val = c.get(1);            // read_handle<V>，零堆分配，隐式 bool 转换

    // 生产级线程安全缓存（推荐）
    production_cache<int, std::string> pc(1000000);
    // = segmented hash table + sharded LRU + 64-stripe distributed_shared_mutex

    // 线程安全缓存（按并发级别选择）
    safe_cache<int, std::string> sc(100);       // 单全局锁，低并发/测试
    striped_cache<int, std::string> stc(100);    // 64-stripe 锁，中等并发
    return 0;
}
```

## Cache Aliases

### 基础别名（标准哈希表）

| 别名 | 策略 | 线程安全 | 适用场景 |
|------|------|---------|---------|
| `lru::cache<K,V>` | LRU | 否 | 单线程 |
| `lru::safe_cache<K,V>` | LRU | 是（单全局锁） | 低并发/测试 |
| `lru::striped_cache<K,V>` | LRU | 是（64-stripe 锁） | 中等并发 |
| `lru::production_cache<K,V>` | LRU | 是（分段+分片+stripe 锁） | **生产环境推荐** |
| `lru::read_heavy_cache<K,V>` | LRU | 是（单全局锁，defer_promotion+EBR） | 读多写少预配置 |
| `lru::read_heavy_striped_cache<K,V>` | LRU | 是（64-stripe，defer_promotion+EBR） | 读多写少高并发 |
| `lru::read_heavy_w_tiny_lfu<K,V>` | W-TinyLFU | 是（defer_promotion） | 读多写少 W-TinyLFU |
| `lru::lfu_cache<K,V>` | TinyLFU | 否 | 频率感知准入 |
| `lru::safe_lfu_cache<K,V>` | TinyLFU | 是（单全局锁） | 频率感知准入（线程安全） |
| `lru::w_tiny_lfu<K,V>` | W-TinyLFU | 否 | 频率感知（三段式） |
| `lru::safe_w_tiny_lfu<K,V>` | W-TinyLFU | 是（单全局锁） | 三段式（线程安全） |
| `lru::two_q<K,V>` | 2Q | 否 | 冷热分离 |
| `lru::safe_two_q<K,V>` | 2Q | 是（单全局锁） | 冷热分离（线程安全） |
| `lru::fifo_cache<K,V>` | FIFO | 否 | 简单先进先出 |
| `lru::safe_fifo_cache<K,V>` | FIFO | 是（单全局锁） | 先进先出（线程安全） |

### F14 SIMD 加速别名

| 别名 | 策略 | 线程安全 | 说明 |
|------|------|---------|------|
| `lru::f14_cache<K,V>` | LRU | 否 | F14 SIMD 标签探测 |
| `lru::f14_safe_cache<K,V>` | LRU | 是（单全局锁） | F14 + 线程安全 |
| `lru::f14_striped_cache<K,V>` | LRU | 是（64-stripe 锁） | F14 + 高并发 |
| `lru::f14_lfu_cache<K,V>` | TinyLFU | 否 | F14 + 频率感知 |
| `lru::f14_w_tiny_lfu<K,V>` | W-TinyLFU | 否 | F14 + 三段式 |
| `lru::f14_two_q<K,V>` | 2Q | 否 | F14 + 冷热分离 |
| `lru::f14_fifo_cache<K,V>` | FIFO | 否 | F14 + FIFO |
| `lru::f14_production_cache<K,V>` | LRU | 是（F14 + 分片 + stripe 锁） | 生产环境 + SIMD 探测 |

### 分段哈希表别名（per-segment rehash，无全局停顿）

| 别名 | 策略 | 线程安全 | 说明 |
|------|------|---------|------|
| `lru::segmented_cache<K,V>` | LRU | 否 | 分段 rehash |
| `lru::segmented_safe_cache<K,V>` | LRU | 是（单全局锁） | 分段 rehash + 线程安全 |
| `lru::segmented_striped_cache<K,V>` | LRU | 是（64-stripe 锁） | 分段 rehash + 高并发 |
| `lru::segmented_two_q<K,V>` | 2Q | 否 | 分段 rehash + 冷热分离 |
| `lru::segmented_lfu_cache<K,V>` | TinyLFU | 否 | 分段 rehash + 频率感知 |
| `lru::segmented_w_tiny_lfu<K,V>` | W-TinyLFU | 否 | 分段 rehash + 三段式 |
| `lru::segmented_fifo_cache<K,V>` | FIFO | 否 | 分段 rehash + FIFO |

> **`production_cache`** = `segmented_striped_cache`，是生产环境推荐的别名，开箱即用。

## read_handle — 安全访问缓存值

`get()` 返回 `read_handle<V>`（可隐式转换为 `bool`），是访问缓存值的推荐方式：

```cpp
production_cache<int, std::string> cache(1000);
cache.set(1, "hello");

// 推荐：read_handle（零堆分配，RAII 安全）
if (auto h = cache.get(1)) {
    std::cout << *h << "\n";       // operator* 访问值
    // h 析构时自动 decRef，防止 eviction 释放正在使用的值
}

// 需要 shared_ptr 时：
auto sp = cache.get_shared(1);          // 每次堆分配 + 值拷贝
auto sp2 = cache.get_shared_cached(1);  // TLS 缓存，重复 key 零堆分配

// 只读访问（不提升 LRU）：
if (auto v = cache.peek(1)) {
    std::cout << *v << "\n";
}
```

| 方法 | 返回类型 | 堆分配 | LRU 提升 | 适用场景 |
|------|---------|--------|---------|---------|
| `get(key)` | `read_handle<V>` | 零 | 是 | **推荐** |
| `peek(key)` | `read_handle<const V>` | 零 | 否 | 只读不提升 |
| `get_shared(key)` | `shared_ptr<V>` | 每次 | 否 | 需要独立所有权 |
| `get_shared_cached(key)` | `shared_ptr<V>` | TLS 缓存 | 否 | 重复 key 零分配 |
| `get_view(key)` | `optional<reference_wrapper<V>>` | 零 | 否 | 轻量级只读视图 |

## Eviction Strategies

### mm_lru — Enhanced LRU（增强型 LRU）

```cpp
cache<int, std::string> c(1000);  // 默认
// 或通过 mm_lru_config 定制：
mm_lru_config config;
config.default_lru_refresh_time = 60;   // 60 秒内不重复提升
config.lru_refresh_ratio = 0.5;         // 自适应：tail 年龄的一半
config.lru_insertion_point_spec = 1;    // 在 1/2 处插入
config.update_on_read = true;
config.update_on_write = false;
unified_cache<lru_trait<single_threaded_policy>, int, std::string> c(1000, config);
```

- **延迟提升**（Delayed Promotion）：每个节点记录 `update_time`，`record_access()` 检查 `current_time - update_time >= lru_refresh_time`
- **自适应刷新时间**（Adaptive Refresh Time）：基于 tail 年龄动态调整刷新间隔
- **增量插入点**（Incremental Insertion Point）：O(1) 均摊维护插入位置
- **try_lock_update**：高争用场景跳过提升，不阻塞读路径

### mm_2q — Two Queues（三队列 2Q）

```cpp
two_q<int, std::string> c(1000);
// 访问三队列大小：
auto& mm = c.mm();
std::cout << "Hot=" << mm.hot_size() << " Warm=" << mm.warm_size()
          << " Cold=" << mm.cold_size() << "\n";
```

- 新元素进入 Hot 队列
- Hot 溢出 → Cold；Cold 被访问 → Warm
- Warm 溢出 → Cold；Cold 优先淘汰

### mm_tiny_lfu — TinyLFU（频率感知准入）

```cpp
lfu_cache<int, std::string> c(1000);
// 查看频率估计：
auto freq = c.mm().sketch().estimate(key);
```

- 小窗口队列（Tiny）+ 主队列（Main）
- CountMinSketch 概率频率估计
- 淘汰时比较 Tiny tail vs Main tail 频率

### mm_wtiny_lfu — W-TinyLFU（Window TinyLFU，三段式）

```cpp
w_tiny_lfu<int, std::string> c(1000);
// 查看三段队列大小：
auto& mm = c.mm();
std::cout << mm.tiny_size() << "/" << mm.probation_size()
          << "/" << mm.protection_size() << "\n";
```

- **Tiny**（窗口队列）→ **Probation**（试炼区）→ **Protection**（保护区）
- Protection 溢出时退化到 Probation **TAIL**（而非 HEAD）
- 三段频率感知准入/淘汰

### mm_fifo — FIFO（先进先出）

```cpp
fifo_cache<int, std::string> c(1000);
```

- 简单 FIFO 淘汰，适合不需要 LRU 排序的场景

## Core API (`unified_cache`)

| 方法 | 描述 |
|------|------|
| `set(key, value)` | 插入/更新 |
| `get(key)` | 读取（提升 LRU），返回 `read_handle<V>` |
| `try_get(key)` | 读取（不抛异常），返回 `optional<read_handle<V>>` |
| `peek(key)` | 只读（不提升），返回 `read_handle<const V>` |
| `add(key, value)` | 仅不存在时插入 |
| `replace(key, value)` | 仅存在时替换 |
| `get_or_fetch(key, provider)` | 命中返回，未命中调用 provider 并写入 |
| `try_get_or_fetch(key, provider)` | 同上但不抛异常 |
| `get_with_ttl(key)` | 读取并返回剩余 TTL |
| `cas(key, expected, desired)` | 比较交换（CAS），原子更新 |
| `del(key)` | 删除（触发 evict 回调） |
| `remove(key)` | 删除，返回 `RemoveRes::kSuccess / kNotFound` |
| `contains(key)` | 检查存在性（不触发访问/提升） |
| `get_multi(keys)` | 批量读取（`span<const Key>`），按 stripe 分组减少锁获取 |
| `set_multi(pairs)` | 批量写入（`span<const pair<K,V>>`），按 stripe 分组减少锁获取 |
| `bulk_get(first, last)` | 按 shard 分组批量读取，每 shard 仅获取一次读锁 |
| `flush()` | 清空所有条目 |
| `size()` | 当前条目数 |
| `max_size()` / `max_size(n)` | 容量上限 getter/setter |
| `current_memory()` / `max_memory(n)` | 内存用量 getter/setter |
| `stats_snapshot()` | 线程安全统计快照（含延迟分位 P50/P95/P99 与 rehash 统计） |
| `prometheus_text()` | Prometheus exposition format 指标导出 |
| `diagnostics()` | 运行时诊断快照（per-shard 争用、rehash 进度、TLS ring、handle 计数） |
| `diagnostics_text()` | 人类可读的诊断文本 |
| `hot_shards(n)` | 返回最热的 n 个分片（hits/misses/memory/rehash） |
| `empty()` | 是否为空 |
| `callbacks()` | 回调管理器引用 |
| `mm()` | MM 策略直接访问 |
| `reserve(n)` | 预分配哈希桶，避免运行时 rehash 停顿 |
| `shutdown()` | 优雅停机（拒绝新请求，等待活跃 handle 释放） |
| `is_shutdown()` | 查询停机状态 |
| `active_handle_count()` | 当前活跃 read_handle 数 |

### MM 层扩展 API（通过 `c.mm()` 访问）

| 方法 | 描述 |
|------|------|
| `pop(key)` | 移除并返回值 |
| `pop_lru()` | 移除并返回 LRU 端条目 |
| `begin()` / `end()` | MRU→LRU 迭代器 |
| `config()` / `set_config(...)` | 运行时策略配置 |

## Callbacks

```cpp
cache<int, std::string> c(5);
c.callbacks().on_hit([&](const int& k, const std::string& v) {});
c.callbacks().on_miss([&](const int& k) {});
c.callbacks().on_insert([&](const int& k, const std::string& v) {});
c.callbacks().on_evict([&](const int& k, const std::string& v) {});
```

**异步回调**（TLS 零分配，减少锁争用）：
```cpp
// 临界区内收集事件（TLS ring buffer，零锁）
c.callbacks().collect_hit(key, value);
c.callbacks().collect_evict(key, std::move(value));

// 临界区外 flush
c.callbacks().flush_pending();
```

## TTL Cache

```cpp
ttl_cache<int, std::string> c(5s, 100);  // 默认 5s TTL，max 100 项
c.set(1, "one");                          // 使用默认 TTL
c.set_with_ttl(2, "two", 10s);            // 自定义 TTL
c.set_no_ttl(3, "three");                 // 永不过期
c.set_until(4, "four", ttl_entry<std::string>::from_now(1min)); // 绝对时间

auto val = c.get(1);     // 惰性过期检查
c.clear_expired();       // 主动清理过期
c.remaining_ttl(1);      // 查看剩余 TTL
```

## Memory Monitor

```cpp
memory_monitor monitor(memory_monitor::config{
    .max_memory_bytes = 1 * 1024 * 1024,  // 1MB 预算
    .high_watermark_fraction = 0.90,
    .critical_watermark_fraction = 0.98,
});

if (monitor.should_admit(item_size)) {
    monitor.report_insert(item_size);
}
```

三态压力模型（对齐 CacheLib）：
- **Normal**（< throttle_fraction）：接受所有插入
- **Throttled**（throttle_fraction ~ critical_fraction）：接受插入，触发后台淘汰
- **Critical**（>= critical_fraction）：拒绝插入 + 激进后台淘汰

## Slab Allocator

```cpp
// 启用 slab 分配器（11 大小类：64 ~ 65536 字节）
cache.set_slab_allocator_options(slab_allocator_options{...});
cache.enable_slab_allocator();

// 后台再平衡
cache.start_slab_rebalancer();
```

- 每大小类独立 lock-free Treiber stack（ABA-safe tagged pointer）
- 后台 `slab_rebalancer` 按 hits-per-slab 策略动态调整大小类占比
- NUMA 感知分配（Linux `mbind()`）

## Concurrency Architecture

```
                         ┌──────────────────────────────┐
                         │    production_cache<K,V>      │
                         │ (segmented_striped_cache)     │
                         └──────────┬───────────────────┘
                                    │
             ┌──────────────────────┼──────────────────────┐
             │                      │                      │
    ┌────────▼────────┐  ┌─────────▼────────┐  ┌─────────▼────────┐
    │ Segmented Hash  │  │  Sharded MM LRU  │  │ 64-stripe Lock   │
    │ Table (64 seg)  │  │  (64 shards)     │  │ (dist_shared_mutex)│
    │ per-seg rehash  │  │  per-shard lock  │  │  writer_fair 默认 │
    └────────┬────────┘  └────────┬────────┘  └────────┬────────┘
             │                    │                     │
    ┌────────▼────────┐  ┌───────▼────────┐  ┌────────▼────────┐
    │ Optimistic Read │  │ TLS Access Ring │  │ WaitOnAddress/  │
    │ + hazptr + F14  │  │ Deferred Promo  │  │ futex fallback  │
    │ + seqlock       │  │ + try_lock      │  │ ~10ns uncontended│
    └─────────────────┘  └────────────────┘  └─────────────────┘
```

关键并发设计：
- **乐观读路径**：`find()` 使用 version-based 乐观读（无 RMW）→ hazptr/EBR 遍历 → bucket shared lock 三级 fallback。非 `EmbeddedChain` 模式下乐观读默认使用 shared-lock fallback，防止 UAF。
- **find_and_pin()**：原子化 find + incRef，消除 TOCTOU
- **TLS 延迟提升**：`get()` 在 TLS ring buffer 中记录访问，非阻塞 `drain_access_ring()` 批量提升
- **defer_promotion 默认开启**：`get()` 命中时不强制触发写锁，延迟到 TLS ring 批量提升，显著降低读路径锁压力
- **refcount_with_flags**：CAS 无锁引用计数（11-bit flags + 3-bit admin_ref + 18-bit access_ref）
- **distributed_shared_mutex**：默认 `writer_fair` 公平模式（防止写者饥饿），fast path 单 CAS ~10ns，WaitOnAddress/futex 慢路径；运行时可切换 `reader_preferred`；支持运行时锁顺序检查开关 `set_lock_order_checking()`
- **segmented rehash**：64 段独立 rehash，不阻塞其他段读写
- **增量 rehash**（chain / F14 / segmented 三种模式均支持）：`set_incremental_rehash(true)` 启用后，rehash 分批迁移桶，避免写路径一次性停顿。F14 模式按 14-slot chunk 迁移并使用 dual-array lookup；segmented 模式对 64 个段独立应用增量 rehash，任一时刻只阻塞 1/64 的桶。`segmented_*` / `production_*` 别名在构造时自动启用。
- **bulk_get 优化**：按 shard 分组，每 shard 仅获取一次读锁；`record_access` 移出锁范围以避免 drain 死锁
- **Pre-hashed API（T16）**：`get_prehashed(key, hash)` / `set_prehashed` / `try_get_prehashed` / `peek_prehashed` / `contains_prehashed`，调用者预计算哈希后复用于 shard dispatch + stripe lock + hash-table lookup，消除重复哈希计算。
- **可观测性**：`stats_snapshot()` 含延迟分位（P50/P95/P99）、rehash 统计、TLS ring 积压与 reclaim（pending/total/freed_bytes/invocations）指标；`prometheus_text()` 导出 Prometheus 格式指标（含 `lru_reclaim_*` 系列）；`diagnostics()` / `diagnostics_text()` 提供运行时诊断（含 hash table mode 段与 reclaim health 段）；`hot_shards(n)` 按访问量排序；`hot_shards_by_memory(n)`（T18）按内存维度排序；`try_reclaim_now()` 主动驱动 hazptr/EBR 回收。

## NUMA 部署建议

### 局限性

`distributed_shared_mutex` 内部使用单一 `std::atomic<uint32_t> state_` 表示读写状态，跨 socket
NUMA 访问时该原子会在 socket 间乒乓，导致：

- 读路径 fast path 的 CAS 在多 socket 系统上延迟可达 100-300ns（单 socket 仅 ~10ns）
- 高并发读多写少场景下，跨 socket 乒乓会显著抬高 P99/P999
- `state_` 上的 `fetch_add` / `fetch_sub` 在所有 CPU 间产生全局缓存一致性流量

### 推荐部署模式：每 NUMA node 一个 cache 实例 + 一致性哈希路由

```cpp
#include <lru.hpp>
#include <array>
#include <functional>

// 1. 获取 NUMA node 数量（Linux: numactrl --hardware; 程序内可解析 /sys/devices/system/node/online）
constexpr std::size_t kNumNumaNodes = 4;  // 示例：4 个 NUMA node

// 2. 每 NUMA node 一个独立 cache 实例
//    使用 numactl --cpunodebind=N --membind=N 绑定进程到对应 NUMA node 后启动多个进程，
//    或在单进程内使用 mbind() 将每个 cache 的内存固定到对应 NUMA node。
std::array<lru::production_cache<K, V>, kNumNumaNodes> per_node_caches{
    lru::production_cache<K, V>{1000000},
    lru::production_cache<K, V>{1000000},
    lru::production_cache<K, V>{1000000},
    lru::production_cache<K, V>{1000000},
};

// 3. 按 key 哈希路由到对应 NUMA node 的 cache
//    不同 key 落到不同 cache 实例，互不共享 distributed_shared_mutex::state_
auto& cache_for_key(const K& key) -> lru::production_cache<K, V>& {
    auto h = std::hash<K>{}(key);
    return per_node_caches[h % kNumNumaNodes];
}

// 4. 读写路径
Value get(const K& key) {
    auto h = cache_for_key(key).get(key);
    if (h) return *h;
    // miss fallback...
}
void set(const K& key, Value v) {
    cache_for_key(key).set(key, std::move(v));
}
```

### 配合 slab allocator 的 NUMA 内存策略

本库 slab allocator 已实现 Linux `mbind()` + `MPOL_BIND` 把每个大小类的内存绑定到当前 NUMA
node（详见「Slab Allocator」章节）。多 NUMA 部署建议：

1. **每进程绑定一个 NUMA node**（推荐）：用 `numactl --cpunodebind=N --membind=N ./app` 启动
   多个进程，每个进程内部一个 `production_cache`，由 slab allocator 的 `mbind()` 把
   cache 内存固定到本 node。跨 NUMA 访问交给客户端路由（如上例）。
2. **单进程多 NUMA**：在单进程内构造多个 cache 实例，每个实例的 slab allocator 用
   `mbind()` 绑定到不同 NUMA node（参见 [memory.hpp](memory.hpp) 中 `mbind()` 集成）。
   优点是单进程共享 fd/连接池；缺点是 cross-node 访问仍需通过 QPI/UPI，仅缓解
   `distributed_shared_mutex` 的乒乓问题，不消除跨 node 内存访问延迟。

### 何时不需要 NUMA 优化

- **单 socket 机器**（包括所有消费级台式机/笔记本）：`state_` 永远在本地 socket，无乒乓
- **UMA 多核服务器**：所有 CPU 访问内存延迟一致，无跨 socket 乒乓
- **并发度 < 物理 core 数**：单 socket 即可承载，无 NUMA 优化必要

## Production Features（读多写少生产场景）

以下 API 针对多线程高并发读多写少生产环境设计，降低尾延迟、提升可观测性与稳健性：

### 延迟提升与锁公平性

```cpp
striped_cache<int, std::string> c(100000);
// defer_promotion 默认开启：get() 命中时不触发写锁，延迟到 TLS ring 批量提升
c.set_defer_promotion(false);       // 显式关闭（每次 get 都 promote）
c.is_defer_promotion_enabled();     // 查询状态

// 锁公平模式：默认 writer_fair（防止写者饥饿），可切换 reader_preferred（最大读吞吐）
c.set_fairness_mode(lru::detail::fairness_mode::reader_preferred);
c.get_fairness_mode();

// 读多写少预配置别名（trait 层自动设置 defer_promotion=true + EBR + reader_preferred）
lru::read_heavy_cache<int, std::string> rh(100000);
lru::read_heavy_striped_cache<int, std::string> rhs(100000);
lru::read_heavy_w_tiny_lfu<int, std::string> rhw(100000);
```

### 运行时分片数与增量 rehash

```cpp
// 构造时指定分片数（高核机器推荐 128+）
striped_cache<int, std::string> c(1000000, /*num_stripes=*/128);

// 启用增量 rehash（chain 模式）：rehash 分批迁移桶，避免写路径停顿
c.set_incremental_rehash(true);
c.incremental_rehash_enabled();
```

### 内存水位线与 OOM 保护

```cpp
// 设置软/硬内存水位线（相对 max_memory 的比例）
c.set_memory_watermarks(0.85, 0.95);

// 超过 soft 水位：新插入触发激进淘汰
// 超过 critical 水位：cache 进入只读模式，set() 拒绝新项
if (c.is_memory_critical()) {
    // 当前处于 critical 模式
}

// OOM 处理器：首次进入 critical 模式时回调
c.set_oom_handler([](std::size_t current, std::size_t max) {
    std::cerr << "CRITICAL: " << current << "/" << max << "\n";
});
```

注意：这是 **cache 级** 水位线。`memory_monitor`（`memory.hpp`）提供的是系统级内存压力监控，两者可独立或组合使用。

### 生产常用 API

```cpp
// 不抛异常的 get
auto h = c.try_get(key);             // -> std::optional<read_handle<V>>

// 不抛异常的 get-or-fetch
auto v = c.try_get_or_fetch(key, [](const int& k) { return fetch(k); });

// 获取剩余 TTL
auto [handle, ttl] = c.get_with_ttl(key);

// 比较交换（CAS）
bool ok = c.cas(key, expected, desired);
bool ok2 = c.cas(key, [](const V& cur) { return cur == expected; }, desired);
```

### 可观测性

```cpp
auto stats = c.stats_snapshot();
// 延迟分位（纳秒）
stats.get_latency.percentile(0.99);  // P99 get 延迟
stats.set_latency.percentile(0.99);  // P99 set 延迟
// 活跃 handle 数、TLS ring 积压
stats.active_handle_count;
stats.tls_ring_backlog;
// rehash 统计
stats.rehash_count;
stats.rehash_total_time_ns;
stats.rehash_migrated_items;

// Prometheus 导出
std::string metrics = c.prometheus_text();
// 含 lru_hits_total / lru_misses_total / lru_size / lru_memory_bytes
//    lru_evictions_total / lru_lock_wait_total / lru_get_latency_p99_ms
//    lru_rehash_count / lru_rehash_total_time_ns / lru_memory_critical

// 运行时诊断（per-shard 争用、rehash 进度、TLS ring、handle 计数）
auto info = c.diagnostics();
std::cout << c.diagnostics_text();

// 热点分片识别
for (auto& hot : c.hot_shards(5)) {
    std::cout << "shard=" << hot.shard_index
              << " accesses=" << hot.total_accesses
              << " hit_rate=" << hot.hit_rate
              << " memory=" << hot.memory_bytes << "\n";
}
```

### TTL 主动清理后台线程

```cpp
c.start_ttl_cleaner(std::chrono::seconds(1));  // 每 1s 清理一个分片的过期 key
c.stop_ttl_cleaner();
c.is_ttl_cleaner_running();
c.evict_expired_now();  // 手动触发立即清理
```

### 异步回调

```cpp
// 异步模式：drain 线程只入队，独立 worker 出队执行，避免阻塞 drain
c.set_async_callbacks(true);
c.callbacks().on_evict([](const int& k, const std::string& v) {
    // 可执行 IO（写日志、上报监控）而不阻塞缓存 drain 线程
});
```

### 优雅停机

```cpp
c.shutdown();           // 拒绝新 get/set，等待活跃 read_handle 释放
c.is_shutdown();        // 查询停机状态
c.active_handle_count(); // 当前活跃 handle 数（降为 0 后才安全析构）
// shutdown() 后：get() 抛异常，try_get() 返回 nullopt
```

### 分片序列化

```cpp
// 分片序列化：每片独立锁，避免全局阻塞
lru::save_per_shard(c, "cache.dat");
lru::load_per_shard(c2, "cache.dat");
```

### 推荐生产配置（read-heavy 多线程高并发）

下面的代码片段是面向 **多线程高并发读多写少** 生产场景的推荐开箱即用配置。每一步都标注了
动机和实际收益。生产服务建议把这组配置封装成工厂函数，避免在业务代码里散落。

```cpp
#include "lru.hpp"
using namespace lru;
using namespace std::chrono_literals;

// 1) 选型：production_cache = segmented hash table (64 segments)
//    + sharded_mm_lru (64 shards) + striped_thread_safe_policy (64 stripes)。
//    自动启用 incremental rehash，扩容只阻塞 1/64 桶；defer_promotion 默认开启，
//    get() 命中走 TLS ring 批量提升，不抢写锁。
production_cache<int, std::string> c{
    /*max_size=*/1'000'000,
    /*num_stripes=*/128        // 高核机器（≥32 cores）推荐 128+，降低 stripe 冲突
};

// 2) 锁公平性：默认 writer_fair（防写者饥饿）。如果业务确认可接受写延迟，
//    切到 reader_preferred 可拿最大读吞吐（典型 +15-30% QPS）。仅在低写场景切换。
c.set_fairness_mode(lru::detail::fairness_mode::reader_preferred);

// 3) reserve：预热哈希表，避免首次写入大流量触发的同步 rehash 停顿。
//    估算最终容量后调用一次即可。
c.reserve(1'000'000);

// 4) 内存水位线：触发激进淘汰 / 只读模式，防止 OOM kill 整个进程。
//    soft=0.85：超过即开始激进淘汰；critical=0.95：set() 拒绝新项。
c.set_memory_watermarks(/*soft=*/0.85, /*critical=*/0.95);
c.set_oom_handler([](std::size_t cur, std::size_t max) {
    // 上报到监控系统 / 触发报警
});

// 5) 后台 TTL 清理：per-shard 轮转，避免全局锁。1s 足够覆盖大多数 TTL 场景。
c.start_ttl_cleaner(1s);

// 6) 事件 drain worker：周期性 drain TLS access ring + callback ring + hazptr/EBR。
//    未启动则 TLS ring 会在下一次 get()/set() 顺手 drain，但长连接 / 高 QPS 下
//    显式 worker 能保证 drain 时延有上界。
c.start_event_drain(500ms);

// 7) 异步回调（可选）：on_evict 如果做 IO（写日志、上报）必须开 async 模式，
//    避免 drain 线程被用户 callback 阻塞。
c.set_async_callbacks(true);
c.callbacks().on_evict([](const int& k, const std::string& v) {
    // 安全地执行 IO / 远程上报
});

// 8) 事件追踪（可选）：on_hit/on_insert/on_evict 进 event_tracker 的 TLS ring，
//    可在事后查 top-K 热点 key。drain worker 自动 drain tracker。
auto tracker = c.enable_event_tracking();
// tracker->top_keys(10) / tracker->generate_report() ...

// 9) Prometheus 导出：周期性（如 15s）调用并写到 /metrics 端点。
//    包含 hits/misses/size/memory/latency P50/P95/P99、lock_wait、rehash、reclaim_*。
std::string metrics = c.prometheus_text();

// 10) 优雅停机：进程退出前 shutdown()，等所有 read_handle 释放后再析构。
//     shutdown() 后 get() 抛异常、try_get() 返回 nullopt。
c.shutdown();
while (c.active_handle_count() > 0) {
    std::this_thread::sleep_for(10ms);
}
// 至此可安全析构 c
```

**避免的反模式**：

- ❌ 在高 QPS 读路径上调用 `stats_snapshot()` / `prometheus_text()` —— 它们读 bucket
  统计，O(num_shards + bucket_count)，应在监控采集线程里 15s 调用一次。
- ❌ `has_active_handles()` / `memory_monitor_stats()` 不要放在热路径上：前者会
  逐 shard 共享锁，后者读 atomic sum。运行时偶尔检查即可。
- ❌ 在 `on_evict` 里同步做磁盘 IO —— 必须开 `set_async_callbacks(true)`，否则
  drain 线程被阻塞，TLS ring 积压，写路径锁等待暴涨。
- ❌ 切换 `set_fairness_mode` 时未等待 quiescent —— 文档要求在无活跃锁时切换，
  否则可能让正在排队的 writer 永久等待（debug 构建有 assert）。
- ❌ 频繁 `set_incremental_rehash(true/false)` 运行时切换 —— 一次配置后保持。

## 2Q / TinyLFU / W-TinyLFU 缓存

```cpp
// 2Q
two_q<int, std::string> twoq(100);
twoq.callbacks().on_evict(...);

// TinyLFU
lfu_cache<int, std::string> lfu(100);
auto freq = lfu.mm().sketch().estimate(key);

// W-TinyLFU
w_tiny_lfu<int, std::string> wtlfu(100);
auto& mm = wtlfu.mm();
std::cout << mm.tiny_size() << "/" << mm.probation_size()
          << "/" << mm.protection_size() << "\n";
```

## Serialization

```cpp
// saveState（序列化为二进制）
auto data = serialize(c.mm());

// restoreState（从二进制恢复）
deserialize(c2.mm(), data);

// 支持 mm_lru / mm_2q / mm_tiny_lfu / mm_wtiny_lfu
```

## Scripts（推荐构建/测试入口）

`scripts/` 目录提供参数化的通用脚本，替代根目录散落的临时 `verify_*.sh` / `_*.sh`。所有脚本自动检测 MSYS2 Clang64 工具链，并默认启用 ccache。

### `scripts/build.sh` — 通用构建

```bash
./scripts/build.sh                # Debug 构建，tests on，examples off
./scripts/build.sh release        # Release 构建
./scripts/build.sh asan           # Debug + AddressSanitizer（Clang64 专用）
./scripts/build.sh tsan           # Debug + ThreadSanitizer（Linux 专用）
./scripts/build.sh ubsan          # Debug + UBSan（未定义行为；全平台可用）
./scripts/build.sh lsan           # Debug + LeakSanitizer（Linux 专用）
./scripts/build.sh asan-ubsan     # Debug + ASan + UBSan（组合）
./scripts/build.sh bench          # Release + benchmarks
./scripts/build.sh examples      # Debug + examples
./scripts/build.sh -j4            # 覆盖默认并行度（默认 -j2）
```

幂等 configure：若 `${BUILD_DIR}/Makefile` 已存在则跳过 cmake，直接 make。设置 `LRU_FORCE_CONFIGURE=1` 可强制重新 configure。

**Sanitizer 组合矩阵**（由 `CMakeLists.txt` 强制）：

| Profile | 标志 | 平台 | 说明 |
|---|---|---|---|
| `asan` | `-fsanitize=address` | Clang64 / Linux | Linux 下含 LSan；MinGW 下 LSan 不可用 |
| `tsan` | `-fsanitize=thread` | 仅 Linux | 与 ASan/LSan 互斥 |
| `ubsan` | `-fsanitize=undefined -fno-sanitize-recover=undefined` | 全平台 | UB 致命（首次 UB 即 abort） |
| `lsan` | `-fsanitize=leak` | 仅 Linux | 独立运行；与 ASan 冗余（自动禁用） |
| `asan-ubsan` | `-fsanitize=address,undefined` | Clang64 / Linux | 组合；两者均致命 |

### `scripts/test.sh` — 通用测试

```bash
./scripts/test.sh                                # 全量 ctest（build/）
./scripts/test.sh asan                           # 全量 ctest（build/asan/）
./scripts/test.sh ubsan                          # 全量 ctest（build/ubsan/）
./scripts/test.sh asan-ubsan                     # 全量 ctest（build/asan-ubsan/）
./scripts/test.sh -g "RefcountTest.*"            # gtest filter 直跑
./scripts/test.sh -g "DeadlockDetection.StripedCacheFlushUnderLoad" -r 20
                                                # 单测重复 20 次（flaky 验证）
./scripts/test.sh -R "RefcountTest"              # ctest regex
./scripts/test.sh -e lru_refcount_test           # 指定可执行文件
./scripts/test.sh --stress-duration 5            # 设 LRU_STRESS_DURATION_SECS=5
./scripts/test.sh -t 120                         # 超时秒数（默认 600）
```

模式选择：
- 默认走 `ctest`（适合全量回归）
- 使用 `-g`（gtest filter）或 `-e`（指定可执行文件）时切到 gtest 直跑，支持 `--gtest_repeat`
- Sanitizer profile 自动设置运行时环境变量：

| Profile | 变量 | 默认值 | 含义 |
|---|---|---|---|
| `asan` / `asan-ubsan` | `ASAN_OPTIONS` | `detect_leaks=0:abort_on_error=1:halt_on_error=0:print_stacktrace=1` | MinGW: LSan 不可用；Linux 下设 `detect_leaks=1` |
| `tsan` | `TSAN_OPTIONS` | `halt_on_error=0:second_deadlock_stack=1:report_bugs=1` | 首次竞争后继续以报告所有发现 |
| `ubsan` / `asan-ubsan` | `UBSAN_OPTIONS` | `print_stacktrace=1:halt_on_error=0` | UB 已通过编译标志设为致命 |
| `lsan` | `LSAN_OPTIONS` | `exitcode=23:report_objects=1` | 退出码 23 = 泄漏 |

失败时 `test.sh` 输出当前 sanitizer 与退出码含义：

| 退出码 | Sanitizer | 含义 |
|---|---|---|
| 1 | ASan / UBSan / asan-ubsan | 内存错误或未定义行为 |
| 23 | LSan | 检测到内存泄漏 |
| 66 | TSan | 数据竞争或死锁 |

### `scripts/gdb.sh` — GDB 批量调试

```bash
./scripts/gdb.sh "StressTest.MemoryPressureWithEviction"
./scripts/gdb.sh "DeadlockDetection.StripedCacheFlushUnderLoad" -t 30
```

在 GDB batch 模式下跑指定 gtest filter，捕获 `bt 30` + `info threads` + `thread apply all bt 8`，输出到 `gdb_output.log`。

### `scripts/clean.sh` — 清理

```bash
./scripts/clean.sh              # 默认：build/、build_*/、_deps/、CMake 生成文件
./scripts/clean.sh deep         # 同时删除 *.log
./scripts/clean.sh all          # 同时删除 compile_commands.json
./scripts/clean.sh sanitizers   # 仅删除 sanitizer 构建目录（asan/tsan/ubsan/lsan/asan-ubsan）
```

### 典型工作流

```bash
# 完整 TDD 周期
./scripts/build.sh && ./scripts/test.sh

# ASan 构建+测试
./scripts/build.sh asan && ./scripts/test.sh asan

# UBSan 构建+测试（未定义行为检测，全平台可用）
./scripts/build.sh ubsan && ./scripts/test.sh ubsan

# ASan + UBSan 组合（同时捕获内存错误和 UB）
./scripts/build.sh asan-ubsan && ./scripts/test.sh asan-ubsan

# 验证 flaky 测试不再失败（重复 20 次）
./scripts/test.sh -g "DeadlockDetection.StripedCacheFlushUnderLoad" -r 20

# 调试崩溃测试
./scripts/gdb.sh "StressTest.MemoryPressureWithEviction"

# 构建并运行 benchmark
./scripts/build.sh bench
./build/benchmarks/lru_concurrent_read_benchmark.exe

# 清理 sanitizer 构建目录（不影响主 build/）
./scripts/clean.sh sanitizers
```

### 环境变量覆盖

| 变量 | 作用 | 默认值 |
|------|------|--------|
| `BUILD_DIR` | 覆盖默认构建目录 | per-profile（build/、build/asan/、build/bench/...） |
| `CC` / `CXX` | 编译器 | `ccache clang` / `ccache clang++` |
| `ASAN_OPTIONS` | ASan 运行时选项（`asan` / `asan-ubsan`） | `detect_leaks=0:abort_on_error=1:halt_on_error=0:print_stacktrace=1` |
| `TSAN_OPTIONS` | TSan 运行时选项（`tsan`） | `halt_on_error=0:second_deadlock_stack=1:report_bugs=1` |
| `UBSAN_OPTIONS` | UBSan 运行时选项（`ubsan` / `asan-ubsan`） | `print_stacktrace=1:halt_on_error=0` |
| `LSAN_OPTIONS` | LSan 运行时选项（`lsan`） | `exitcode=23:report_objects=1` |
| `LRU_STRESS_DURATION_SECS` | 压测时长 | 各测试内嵌默认 |
| `LRU_FORCE_CONFIGURE` | `=1` 强制 build.sh 重跑 cmake configure | 未设置 |

> **说明**：上述 sanitizer `*_OPTIONS` 默认值由 `scripts/test.sh` 在对应 profile 下注入。
> 用户可通过环境变量覆盖（例：`ASAN_OPTIONS=detect_leaks=1 ./scripts/test.sh asan` 在 Linux 上启用 LSan）。
> UBSan 已通过编译标志 `-fno-sanitize-recover=undefined` 设为致命；`UBSAN_OPTIONS` 仅控制运行时打印行为。

## Build（底层 cmake 命令参考）

`scripts/build.sh` 内部即执行以下命令。如需直接调用 cmake，参考如下：

```bash
# 使用 MSYS2 Clang64 工具链
export PATH=/clang64/bin:/usr/bin:$PATH

cmake -B build -G "MinGW Makefiles" \
    -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang \
    -DCMAKE_BUILD_TYPE=Debug
mingw32-make -C build -j4

# 构建测试
cmake -B build -G "MinGW Makefiles" \
    -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang \
    -DCMAKE_BUILD_TYPE=Debug -DLRU_BUILD_TESTS=ON
mingw32-make -C build -j4
ctest --test-dir build

# 构建示例
cmake -B build -G "MinGW Makefiles" \
    -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang \
    -DCMAKE_BUILD_TYPE=Debug -DLRU_BUILD_EXAMPLES=ON
mingw32-make -C build -j4
# 示例可执行文件在 build/examples/ 下

# AddressSanitizer（需 Clang64，UCRT64 GCC 无 libasan）
cmake -B build/asan -G "MinGW Makefiles" \
    -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang \
    -DCMAKE_BUILD_TYPE=Debug -DLRU_BUILD_TESTS=ON -DLRU_ENABLE_ASAN=ON
mingw32-make -C build/asan -j4
ctest --test-dir build/asan

# 构建基准测试
cmake -B build/bench -G "MinGW Makefiles" \
    -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang \
    -DCMAKE_BUILD_TYPE=Release -DLRU_BUILD_BENCHMARKS=ON
mingw32-make -C build/bench -j4
```

Header-only library，仅需 C++20 编译器和 `ankerl/unordered_dense`（已包含）。

## Linux-Only TODO（需在 Linux 环境验证/完善）

以下项目已在代码中实现，但因依赖 Linux 特有机制，无法在 Windows MinGW 环境下验证或充分测试：

- [ ] **ThreadSanitizer 验证**：MinGW 不支持 TSan，需在 Linux GCC/Clang 上以 `-fsanitize=thread` 构建并运行全部测试（包括 `tests/test_chaos.cpp` 的 Zipf/并发 bulk_get/热点分片检测等混沌场景）
- [ ] **`std::shared_mutex` 基准对比**：MinGW 的 `pthread_rwlock_t` 在高并发下有 EINVAL bug，需在 Linux 上对比 `distributed_shared_mutex` vs `std::shared_mutex` 吞吐量
- [ ] **NUMA slab 分配运行时验证**：`mbind()` + `MPOL_BIND` 代码已实现，需在多 NUMA 节点 Linux 机器上验证跨节点访问延迟改善
- [ ] **futex 路径性能验证**：`distributed_shared_mutex` 和 `shared_spinlock` 的 futex 慢路径仅在 Linux 上可用
- [ ] **共享内存 warm restart 运行时验证**：`mmap(MAP_SHARED)` 路径已实现，需在 Linux 上验证进程重启后 mmap 复用
- [ ] **Linux CI pipeline 实际运行**：`.github/workflows/ci.yml` 已配置 ASan/TSan/benchmark 作业
- [ ] **`perf` cache line bouncing 测量**：需在 Linux 上用 `perf stat -e cache-misses` 测量乐观读与分布式锁优化前后的改善

## License

SPDX-License-Identifier: MIT
