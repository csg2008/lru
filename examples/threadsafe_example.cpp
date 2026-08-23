// Unified LRU Cache - Thread-safe usage example
// Demonstrates production_cache, striped_cache, read_handle, get_shared_cached

#include <chrono>
#include <cmath>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "../lru.hpp"

int main() {
    using namespace lru;

    std::cout << "=== Thread-Safe LRU Cache Example ===\n\n";

    // ====================================================================
    // 1. production_cache —— 推荐用于多线程高并发读多写少的生产环境
    //    特性：segmented hash table + sharded LRU + 64-stripe distributed_shared_mutex
    // ====================================================================
    std::cout << "--- production_cache (推荐生产环境) ---\n";
    production_cache<int, double> prod_cache{1000};

    // 填充缓存
    for (int i = 0; i < 100; ++i) {
        prod_cache.set(i, std::sqrt(static_cast<double>(i)));
    }
    std::cout << "初始大小: " << prod_cache.size() << "\n";

    // 多线程并发读 + 少量写
    std::vector<std::thread> threads;
    constexpr int kReaders = 4;
    constexpr int kWriters = 2;
    std::vector<int> read_hits(kReaders, 0);
    std::vector<int> write_count(kWriters, 0);

    for (int t = 0; t < kReaders; ++t) {
        threads.emplace_back([&prod_cache, &read_hits, t]() {
            for (int i = 0; i < 2000; ++i) {
                int key = (i * 7 + t * 13) % 100;
                // get() 返回 read_handle<V>，零堆分配，RAII 自动 unpin
                auto handle = prod_cache.get(key);
                if (handle) {
                    ++read_hits[t];
                    // 通过 handle 访问值：*handle 或 handle.value()
                }
            }
        });
    }

    for (int t = 0; t < kWriters; ++t) {
        threads.emplace_back([&prod_cache, &write_count, t]() {
            for (int i = 0; i < 100; ++i) {
                int key = 100 + t * 100 + i;
                prod_cache.set(key, std::sqrt(static_cast<double>(key)));
                ++write_count[t];
            }
        });
    }

    for (auto& th : threads) th.join();

    int total_hits = 0;
    for (int h : read_hits) total_hits += h;
    int total_writes = 0;
    for (int w : write_count) total_writes += w;
    std::cout << "读命中: " << total_hits << ", 写入: " << total_writes << "\n";
    std::cout << "最终统计: " << prod_cache.stats_snapshot() << "\n\n";

    // ====================================================================
    // 2. read_handle —— 安全访问缓存值的推荐方式
    //    - 零堆分配（相比 get_shared() 的 std::make_shared）
    //    - RAII：析构时自动 decRef，防止 eviction 释放正在使用的值
    //    - 支持移动语义，可跨函数传递
    // ====================================================================
    std::cout << "--- read_handle 用法 ---\n";
    production_cache<int, std::string> str_cache{100};
    str_cache.set(42, "answer");

    if (auto h = str_cache.get(42)) {
        std::cout << "key 42: " << *h << "\n";            // operator* 访问值
        std::cout << "has_value: " << h.has_value() << "\n"; // 显式检查
    }

    // handle 可以移动传递
    auto make_handle = [&]() -> lru::read_handle<std::string> {
        return str_cache.get(42);
    };
    if (auto h = make_handle()) {
        std::cout << "移动后的 handle: " << *h << "\n";
    }
    std::cout << "\n";

    // ====================================================================
    // 3. get_shared_cached() —— TLS 缓存的 shared_ptr（减少堆分配）
    //    适用于需要 shared_ptr 语义但不想每次堆分配的场景
    //    对重复 key 复用 TLS 缓存的 shared_ptr
    // ====================================================================
    std::cout << "--- get_shared_cached() ---\n";
    if (auto sp = str_cache.get_shared_cached(42)) {
        std::cout << "shared_ptr 值: " << *sp << " (use_count=" << sp.use_count() << ")\n";
    }
    std::cout << "\n";

    // ====================================================================
    // 4. safe_cache vs striped_cache vs production_cache 对比
    // ====================================================================
    std::cout << "--- 线程安全缓存别名对比 ---\n";
    std::cout << "safe_cache:   单全局锁，适合低并发/测试\n";
    std::cout << "striped_cache: 64-stripe 锁，适合中等并发\n";
    std::cout << "production_cache: 分段哈希表 + 分片 LRU + 64-stripe 锁，推荐生产环境\n\n";

    // striped_cache 演示
    striped_cache<int, std::string> scache{1000};
    std::cout << "striped_cache stripe 数量: " << scache.num_stripes() << "\n";

    std::vector<std::thread> workers;
    for (int t = 0; t < 8; ++t) {
        workers.emplace_back([&scache, t]() {
            for (int i = 0; i < 100; ++i) {
                int key = t * 100 + i;
                scache.set(key, "val_" + std::to_string(key));
                if (auto v = scache.get(key)) { (void)v; }
            }
        });
    }
    for (auto& w : workers) w.join();
    std::cout << "striped_cache 最终大小: " << scache.size() << "\n";

    // ====================================================================
    // 5. 批量操作 —— get_multi / set_multi
    //    按 stripe 分组，减少锁获取次数
    // ====================================================================
    std::cout << "\n--- 批量操作 ---\n";
    production_cache<int, std::string> batch_cache{1000};

    // 批量写入
    std::vector<std::pair<int, std::string>> items = {
        {1, "one"}, {2, "two"}, {3, "three"}, {4, "four"}, {5, "five"}
    };
    batch_cache.set_multi(items);

    // 批量读取
    std::vector<int> keys = {1, 2, 3, 4, 5, 99};
    auto results = batch_cache.get_multi(keys);
    for (size_t i = 0; i < keys.size(); ++i) {
        if (results[i]) {
            std::cout << "  key " << keys[i] << ": " << **results[i] << "\n";
        } else {
            std::cout << "  key " << keys[i] << ": (miss)\n";
        }
    }

    // ====================================================================
    // 6. peek —— 不改变 LRU 顺序的只读访问
    // ====================================================================
    std::cout << "\n--- peek（只读访问）---\n";
    if (auto peeked = prod_cache.peek(50)) {
        std::cout << "key 50 peek: " << *peeked << " (不提升 LRU)\n";
    }

    return 0;
}
