// Unified LRU Cache - Basic usage example
// Demonstrates core API, read_handle, pop/pop_lru, remove

#include <iostream>
#include <string>

#include "../lru.hpp"

int main() {
    using namespace lru;

    std::cout << "=== Basic LRU Cache Example ===\n\n";

    // 1. 创建一个最大容量为 5 的缓存
    cache<int, std::string> c(5);

    // 2. 插入条目
    std::cout << "插入条目...\n";
    c.set(1, "one");
    c.set(2, "two");
    c.set(3, "three");
    c.set(4, "four");
    c.set(5, "five");

    std::cout << "缓存大小: " << c.size() << "\n";

    // 3. get() 返回 read_handle<V>（零堆分配，RAII 安全）
    std::cout << "\n访问 key 1...\n";
    auto handle = c.get(1);
    if (handle) {
        std::cout << "值: " << *handle << "\n";
        // read_handle 在析构时自动 decRef，防止 eviction 释放正在使用的值
    }

    // 4. 超出容量插入（淘汰 LRU）
    std::cout << "\n插入 key 6（应该淘汰 key 2）...\n";
    c.set(6, "six");

    std::cout << "包含 1: " << c.contains(1) << "\n";
    std::cout << "包含 2: " << c.contains(2) << " (已淘汰)\n";
    std::cout << "包含 6: " << c.contains(6) << "\n";

    // 5. 统计信息
    std::cout << "\n缓存统计:\n";
    std::cout << c.stats_snapshot() << "\n";

    // 6. 迭代（MRU 到 LRU）—— 使用 c.mm().begin() / c.mm().end()
    std::cout << "\n缓存内容（MRU 到 LRU）:\n";
    for (auto it = c.mm().begin(); it != c.mm().end(); ++it) {
        std::cout << "  " << it->key << " -> " << it->value << "\n";
    }

    // 7. peek —— 不改变 LRU 顺序的只读访问
    std::cout << "\n=== Peek（只读访问）===\n";
    auto peeked = c.peek(6);
    if (peeked) {
        std::cout << "Peek key 6: " << *peeked << "\n";
    }

    // 8. get_shared / get_shared_cached —— 需要 shared_ptr 时的选择
    std::cout << "\n=== get_shared / get_shared_cached ===\n";
    // get_shared(): 每次堆分配 + 值拷贝，适合需要独立所有权的场景
    auto sp = c.get_shared(6);
    if (sp) {
        std::cout << "get_shared(6): " << *sp << " (use_count=" << sp.use_count() << ")\n";
    }
    // get_shared_cached(): TLS 缓存 shared_ptr，重复 key 零堆分配
    auto sp2 = c.get_shared_cached(6);
    if (sp2) {
        std::cout << "get_shared_cached(6): " << *sp2 << " (use_count=" << sp2.use_count() << ")\n";
    }

    // 9. add / replace 语义
    std::cout << "\n=== add / replace 语义 ===\n";
    bool added = c.add(7, "seven");      // key 不存在，插入成功
    bool added_dup = c.add(6, "dup");    // key 已存在，插入失败
    bool replaced = c.replace(7, "SEVEN");
    std::cout << "add(7) = " << added << ", add(6, dup) = " << added_dup
              << ", replace(7) = " << replaced << "\n";
    if (auto v = c.get(7)) {
        std::cout << "key 7 现在的值: " << *v << "\n";
    }

    // 10. remove（CacheLib 对齐）
    std::cout << "\n=== remove ===\n";
    auto rem_res = c.remove(7);
    std::cout << "remove(7) = " << (rem_res == decltype(c)::RemoveRes::kSuccess ? "Success" : "NotFound") << "\n";
    std::cout << "contains(7) = " << c.contains(7) << " (已删除)\n";

    // 11. pop 与 pop_lru（通过 MM 层）
    std::cout << "\n=== pop / pop_lru 语义 ===\n";
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

    // 12. 内存受限缓存
    std::cout << "\n=== 内存受限缓存 ===\n";
    cache<std::string, std::string> mem_cache(unlimited, 100);
    mem_cache.set_key_size_calculator([](const std::string& s) { return s.size(); });
    mem_cache.set_value_size_calculator([](const std::string& s) { return s.size(); });

    mem_cache.set("short", "x");
    mem_cache.set("medium_key", "medium_val");
    mem_cache.set("very_long_key_here", "very_long_value_here"); // 应触发淘汰

    std::cout << "内存缓存大小: " << mem_cache.size() << "\n";
    std::cout << "当前内存: " << mem_cache.current_memory() << "\n";

    // 13. 多策略切换 —— 2Q 缓存
    std::cout << "\n=== 2Q 缓存（三队列）===\n";
    two_q<int, std::string> twoq(4);
    twoq.set(1, "one");
    twoq.set(2, "two");
    twoq.set(3, "three");
    twoq.set(4, "four");
    std::cout << "Hot=" << twoq.mm().hot_size() << " Warm=" << twoq.mm().warm_size()
              << " Cold=" << twoq.mm().cold_size() << "\n";

    // 14. W-TinyLFU 示例 —— 展示频率感知准入
    std::cout << "\n=== W-TinyLFU 频率准入示例 ===\n";
    w_tiny_lfu<int, std::string> wlfu(4);
    // 反复访问 key 1 和 2，提高其频率
    for (int i = 0; i < 5; ++i) {
        wlfu.set(1, "one");
        wlfu.set(2, "two");
        wlfu.get(1);
        wlfu.get(2);
    }
    wlfu.set(3, "three");
    wlfu.set(4, "four");
    wlfu.set(5, "five");

    std::cout << "W-TinyLFU 大小: " << wlfu.size() << "\n";
    std::cout << "包含 key 1: " << wlfu.contains(1) << " (高频，应保留)\n";
    std::cout << "包含 key 2: " << wlfu.contains(2) << " (高频，应保留)\n";
    std::cout << "包含 key 5: " << wlfu.contains(5) << " (新进入者)\n";
    std::cout << "W-TinyLFU 分队列大小: tiny=" << wlfu.mm().tiny_size()
              << " probation=" << wlfu.mm().probation_size()
              << " protection=" << wlfu.mm().protection_size() << "\n";
    std::cout << "W-TinyLFU 统计: " << wlfu.stats_snapshot() << "\n";

    return 0;
}
