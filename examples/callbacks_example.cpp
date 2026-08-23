// Unified LRU Cache - Callbacks example
// Demonstrates sync callbacks + deferred async callback execution

#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "../lru.hpp"

int main() {
    using namespace lru;

    std::cout << "=== LRU Cache Callbacks Example ===\n\n";

    cache<int, std::string> c(5);

    // 跟踪事件
    std::vector<std::string> events;

    // 注册 hit 回调
    c.callbacks().on_hit([&](const int& key, const std::string& value) {
        events.push_back("HIT: key=" + std::to_string(key) +
                        ", value=\"" + value + "\"");
    });

    // 注册 miss 回调
    c.callbacks().on_miss([&](const int& key) {
        events.push_back("MISS: key=" + std::to_string(key));
    });

    // 注册 insert 回调
    c.callbacks().on_insert([&](const int& key, const std::string& value) {
        events.push_back("INSERT: key=" + std::to_string(key) +
                        ", value=\"" + value + "\"");
    });

    // 注册 eviction 回调
    c.callbacks().on_evict([&](const int& key, const std::string& value) {
        events.push_back("EVICT: key=" + std::to_string(key) +
                        ", value=\"" + value + "\"");
    });

    std::cout << "插入条目...\n";
    c.set(1, "apple");
    c.set(2, "banana");
    c.set(3, "cherry");
    c.set(4, "date");
    c.set(5, "elderberry");

    std::cout << "\n访问条目...\n";
    if (auto v = c.get(1)) (void)v; // hit
    if (auto v = c.get(2)) (void)v; // hit
    if (auto v = c.get(99)) (void)v; // miss

    std::cout << "\n触发淘汰...\n";
    c.set(6, "fig");   // 淘汰 3 (LRU)
    c.set(7, "grape"); // 淘汰 4

    std::cout << "\n--- 同步回调事件日志 ---\n";
    for (const auto& event : events) {
        std::cout << event << "\n";
    }

    // 统计信息
    std::cout << "\n--- 统计 ---\n";
    std::cout << c.stats_snapshot() << "\n";

    // ====================================================================
    // 异步（Deferred）回调演示
    // ====================================================================
    std::cout << "\n=== 异步（Deferred）回调演示 ===\n";
    cache<int, std::string> c2(3);

    std::vector<std::string> deferred_events;
    c2.callbacks().on_hit([&](const int& k, const std::string& v) {
        deferred_events.push_back("(deferred) HIT: " + std::to_string(k) + "=" + v);
    });
    c2.callbacks().on_evict([&](const int& k, const std::string& v) {
        deferred_events.push_back("(deferred) EVICT: " + std::to_string(k) + "=" + v);
    });

    // 使用 collect 方法在临界区内记录事件
    // 注意：collect_hit / collect_insert 存储 value 指针（零拷贝），
    // 传入的值必须在 flush_pending() 之前保持有效。
    std::string one = "one";
    std::string two = "two";
    std::string three = "three";
    std::string one_copy = "one";
    c2.callbacks().collect_insert(1, one);
    c2.callbacks().collect_insert(2, two);
    c2.callbacks().collect_insert(3, three);
    c2.callbacks().collect_evict(1, std::move(one_copy)); // evict 存储 move 后的值
    c2.callbacks().collect_hit(2, two);

    std::cout << "flush 前 pending 事件数: " << c2.callbacks().pending_count() << "\n";

    // 在临界区外 flush，不阻塞缓存操作
    c2.callbacks().flush_pending();
    std::cout << "flush 后 pending 事件数: " << c2.callbacks().pending_count() << "\n";

    std::cout << "\nDeferred 事件日志:\n";
    for (const auto& e : deferred_events) {
        std::cout << "  " << e << "\n";
    }

    // ====================================================================
    // 自定义淘汰处理演示
    // ====================================================================
    std::cout << "\n=== 自定义淘汰处理 ===\n";
    std::vector<std::pair<int, std::string>> evicted_items;

    cache<int, std::string> c3(3);
    c3.callbacks().on_evict([&](const int& key, const std::string& value) {
        evicted_items.emplace_back(key, value);
    });

    c3.set(1, "first");
    c3.set(2, "second");
    c3.set(3, "third");
    c3.set(4, "fourth"); // 淘汰 1
    c3.set(5, "fifth");  // 淘汰 2

    std::cout << "被淘汰的条目:\n";
    for (const auto& [key, value] : evicted_items) {
        std::cout << "  key=" << key << ", value=\"" << value << "\"\n";
    }

    return 0;
}
