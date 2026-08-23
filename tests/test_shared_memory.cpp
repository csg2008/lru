// Shared memory backend unit tests.
//
// Two layers of shared-memory warm restart are covered here:
//
//   Cache layer (WarmRestartCacheTest, WarmRestartCacheIntegrationTest):
//     warm_restart_cache::save() and attach(inserter) end-to-end:
//       - A second warm_restart_cache instance sharing the same segment
//         recovers >90% of items saved by the first instance.
//       - Trivially-copyable (int) and std::string keys/values both work.
//       - Corrupt data regions (truncated, oversized claimed count) are
//         detected and attach() returns gracefully without crashing.
//
//   Allocator layer (SlabAllocatorSharedMemTest):
//     slab_allocator's shared_memory_path config: fresh-start creates the
//     file, warm-restart detects and maps it, header validation catches
//     slab_size mismatches, corrupt files fall back to fresh start, and
//     slab counts are preserved across restarts. Migrated here from
//     test_slab_memory.cpp (2026-07-26) so all shared-memory warm-restart
//     coverage lives in one file.
//
// Note on persistence semantics: on Windows, a named file-mapping
// object is destroyed once ALL handles to it are closed. To simulate
// "process restart" within a single test we keep the original
// (writer) cache alive in the same scope while the second (reader)
// cache attaches to the same named segment. On POSIX the segment
// persists after close until shm_unlink(), so the same pattern works
// there too.

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(_WIN32)
#  if !defined(_WINDOWS_)
#    define WIN32_LEAN_AND_MEAN
#    define NOMINMAX
#  endif
#  include <windows.h>
#endif

#include "../lru.hpp"

using namespace lru;

namespace {

// Generate a unique segment name per test to avoid cross-test leakage.
// Windows file-mapping names are global per-session, so we use a
// process-unique prefix plus a counter.
std::string unique_name(const char* tag) {
    static std::atomic<std::size_t> counter{0};
    auto n = counter.fetch_add(1, std::memory_order_relaxed);
#if defined(_WIN32)
    auto pid = static_cast<std::size_t>(GetCurrentProcessId());
#else
    auto pid = static_cast<std::size_t>(getpid());
#endif
    return "lru_test_" + std::string(tag) + "_" + std::to_string(n) +
           "_" + std::to_string(pid);
}

}  // namespace

// ============================================================================
// Trivially-copyable key/value (int -> int)
// ============================================================================

TEST(WarmRestartCacheTest, SaveAndAttachRecoversTriviallyCopyableItems) {
    auto name = unique_name("int_int");

    constexpr std::size_t kItems = 1000;
    constexpr std::size_t kSegSize = sizeof(warm_restart_cache<int, int>::header) +
                                      kItems * (sizeof(uint32_t) * 2 + sizeof(int) * 2) +
                                      4096;  // slack

    // Phase 1: writer — create a fresh segment and save kItems items.
    // Kept alive so the named mapping persists for the reader below.
    std::map<int, int> saved;
    std::vector<std::pair<int, int>> items;
    items.reserve(kItems);
    for (int i = 0; i < static_cast<int>(kItems); ++i) {
        items.emplace_back(i, i * i);
        saved[i] = i * i;
    }

    shared_memory_config wcfg{name, kSegSize, /*create=*/true, /*read_only=*/false};
    warm_restart_cache<int, int> writer(wcfg);
    ASSERT_TRUE(writer.segment()) << "segment must be created";
    ASSERT_EQ(writer.save(items.begin(), items.end()), kItems);

    // Phase 2: reader — open the existing segment and recover.
    // create=true with same name returns ERROR_ALREADY_EXISTS, which
    // shared_memory_segment detects and sets is_newly_created()=false,
    // enabling attach().
    std::map<int, int> recovered;
    shared_memory_config rcfg{name, kSegSize, /*create=*/true, /*read_only=*/false};
    warm_restart_cache<int, int> reader(rcfg);
    ASSERT_TRUE(reader.segment()) << "reader segment must attach";
    ASSERT_TRUE(reader.has_previous_data())
        << "must detect previous run's data";

    auto n = reader.attach([&](const int& k, const int& v) {
        recovered[k] = v;
    });

    // P1-3 acceptance: >90% recovery.
    EXPECT_GE(n, kItems * 9 / 10)
        << "must recover >=90% of items; got " << n;
    EXPECT_EQ(n, kItems) << "all items should be recoverable";

    // Verify exact values.
    EXPECT_EQ(recovered.size(), kItems);
    for (const auto& [k, v] : saved) {
        auto it = recovered.find(k);
        ASSERT_NE(it, recovered.end()) << "missing key=" << k;
        EXPECT_EQ(it->second, v) << "wrong value for key=" << k;
    }
}

// ============================================================================
// std::string key and value
// ============================================================================

TEST(WarmRestartCacheTest, SaveAndAttachRecoversStringItems) {
    auto name = unique_name("str_str");

    constexpr std::size_t kItems = 500;
    constexpr std::size_t kSegSize = 1 << 20;  // 1 MiB plenty

    std::map<std::string, std::string> saved;
    std::vector<std::pair<std::string, std::string>> items;
    items.reserve(kItems);
    for (std::size_t i = 0; i < kItems; ++i) {
        std::string k = "key_" + std::to_string(i);
        std::string v = "value_" + std::to_string(i * 7);
        items.emplace_back(k, v);
        saved[k] = v;
    }

    shared_memory_config wcfg{name, kSegSize, true, false};
    warm_restart_cache<std::string, std::string> writer(wcfg);
    ASSERT_TRUE(writer.segment());
    ASSERT_EQ(writer.save(items.begin(), items.end()), kItems);

    std::map<std::string, std::string> recovered;
    shared_memory_config rcfg{name, kSegSize, true, false};
    warm_restart_cache<std::string, std::string> reader(rcfg);
    ASSERT_TRUE(reader.segment());
    ASSERT_TRUE(reader.has_previous_data());

    auto n = reader.attach([&](const std::string& k, const std::string& v) {
        recovered[k] = v;
    });
    EXPECT_EQ(n, kItems);

    EXPECT_EQ(recovered.size(), kItems);
    for (const auto& [k, v] : saved) {
        auto it = recovered.find(k);
        ASSERT_NE(it, recovered.end()) << "missing key=" << k;
        EXPECT_EQ(it->second, v) << "wrong value for key=" << k;
    }
}

// ============================================================================
// Mixed types: int key -> std::string value
// ============================================================================

TEST(WarmRestartCacheTest, SaveAndAttachHandlesMixedTypes) {
    auto name = unique_name("int_str");
    constexpr std::size_t kSegSize = 1 << 18;  // 256 KiB
    constexpr std::size_t kItems = 200;

    std::map<int, std::string> saved;
    std::vector<std::pair<int, std::string>> items;
    items.reserve(kItems);
    for (int i = 0; i < static_cast<int>(kItems); ++i) {
        std::string v = "v" + std::to_string(i) + "_payload";
        items.emplace_back(i, v);
        saved[i] = v;
    }

    shared_memory_config wcfg{name, kSegSize, true, false};
    warm_restart_cache<int, std::string> writer(wcfg);
    ASSERT_TRUE(writer.segment());
    ASSERT_EQ(writer.save(items.begin(), items.end()), kItems);

    std::map<int, std::string> recovered;
    shared_memory_config rcfg{name, kSegSize, true, false};
    warm_restart_cache<int, std::string> reader(rcfg);
    ASSERT_TRUE(reader.has_previous_data());

    auto n = reader.attach([&](const int& k, const std::string& v) {
        recovered[k] = v;
    });
    EXPECT_EQ(n, kItems);

    EXPECT_EQ(recovered.size(), kItems);
    for (const auto& [k, v] : saved) {
        EXPECT_EQ(recovered[k], v);
    }
}

// ============================================================================
// Empty cache round-trip
// ============================================================================

TEST(WarmRestartCacheTest, EmptySaveAttachesToZeroItems) {
    auto name = unique_name("empty");
    constexpr std::size_t kSegSize = 1 << 14;  // 16 KiB

    shared_memory_config wcfg{name, kSegSize, true, false};
    warm_restart_cache<int, int> writer(wcfg);
    ASSERT_TRUE(writer.segment());
    std::vector<std::pair<int, int>> empty;
    EXPECT_EQ(writer.save(empty.begin(), empty.end()), 0u);

    shared_memory_config rcfg{name, kSegSize, true, false};
    warm_restart_cache<int, int> reader(rcfg);
    ASSERT_TRUE(reader.has_previous_data());

    std::size_t recovered = 0;
    auto n = reader.attach([&](const int&, const int&) { ++recovered; });
    EXPECT_EQ(n, 0u);
    EXPECT_EQ(recovered, 0u);
}

// ============================================================================
// Overwriting an existing segment
// ============================================================================

TEST(WarmRestartCacheTest, OverwriteReplacesPreviousData) {
    auto name = unique_name("overwrite");
    constexpr std::size_t kSegSize = 1 << 16;  // 64 KiB

    // Writer: save v1.
    shared_memory_config wcfg{name, kSegSize, true, false};
    warm_restart_cache<int, std::string> writer(wcfg);
    std::vector<std::pair<int, std::string>> v1 = {{1, "old1"}, {2, "old2"}};
    ASSERT_EQ(writer.save(v1.begin(), v1.end()), 2u);

    // Reader: attach, verify v1, then overwrite with v2.
    shared_memory_config rcfg{name, kSegSize, true, false};
    warm_restart_cache<int, std::string> reader(rcfg);
    ASSERT_TRUE(reader.has_previous_data());

    std::map<int, std::string> got;
    auto n1 = reader.attach([&](const int& k, const std::string& v) { got[k] = v; });
    ASSERT_EQ(n1, 2u);
    EXPECT_EQ(got[1], "old1");

    // Overwrite with v2 (different content).
    std::vector<std::pair<int, std::string>> v2 = {{7, "new7"}, {8, "new8"}, {9, "new9"}};
    ASSERT_EQ(reader.save(v2.begin(), v2.end()), 3u);

    // A third instance attaches and sees v2 only (not v1).
    shared_memory_config r2cfg{name, kSegSize, true, false};
    warm_restart_cache<int, std::string> reader2(r2cfg);
    ASSERT_TRUE(reader2.has_previous_data());

    std::map<int, std::string> got2;
    auto n2 = reader2.attach([&](const int& k, const std::string& v) { got2[k] = v; });
    ASSERT_EQ(n2, 3u);
    EXPECT_EQ(got2[7], "new7");
    EXPECT_EQ(got2[8], "new8");
    EXPECT_EQ(got2[9], "new9");
    EXPECT_EQ(got2.count(1), 0u);
    EXPECT_EQ(got2.count(2), 0u);
}

// ============================================================================
// Data region truncation resilience
// ============================================================================

TEST(WarmRestartCacheTest, AttachSurvivesTruncatedDataRegion) {
    auto name = unique_name("trunc");
    constexpr std::size_t kSegSize = 1 << 16;

    // Writer: save 100 items.
    shared_memory_config wcfg{name, kSegSize, true, false};
    warm_restart_cache<int, int> writer(wcfg);
    std::vector<std::pair<int, int>> items;
    for (int i = 0; i < 100; ++i) items.emplace_back(i, i);
    ASSERT_EQ(writer.save(items.begin(), items.end()), 100u);

    // Reader: claim 110 items — 10 more than actually serialized.
    // attach() must recover the 100 real items and stop cleanly when
    // the data region ends (either by reading zeros or by hitting end).
    shared_memory_config rcfg{name, kSegSize, true, false};
    warm_restart_cache<int, int> reader(rcfg);
    ASSERT_TRUE(reader.has_previous_data());

    reader.set_item_count(110u);

    // Collect recovered keys into a set to dedup bogus zero-padded
    // tail records (which deserialize to (k=0, v=0)).
    std::set<int> recovered_keys;
    auto n = reader.attach([&](const int& k, const int& v) {
        if (k >= 0 && k < 100 && v == k) recovered_keys.insert(k);
    });
    // We should still recover all 100 real items; the rest of the
    // region ends and attach() stops cleanly.
    EXPECT_EQ(recovered_keys.size(), 100u);
    // attach() returns the count of successfully deserialized+inserted
    // records. With zero-padded tail, the loop may emit a few bogus
    // zero-key records (klen=vlen=0), so n may exceed 100 briefly;
    // but it must be bounded by the claimed count.
    EXPECT_LE(n, 110u);
}

// ============================================================================
// Integration: warm restart of an actual safe_cache
// ============================================================================

TEST(WarmRestartCacheIntegrationTest, SafeCacheRecoversFromSharedMemory) {
    auto name = unique_name("safe_integration");
    constexpr std::size_t kSegSize = 1 << 20;  // 1 MiB
    constexpr std::size_t kItems = 1000;

    // Writer: fill a safe_cache, snapshot to shared memory.
    safe_cache<int, std::string> primary{kItems * 2};
    for (int i = 0; i < static_cast<int>(kItems); ++i) {
        primary.set(i, "v" + std::to_string(i));
    }

    shared_memory_config wcfg{name, kSegSize, true, false};
    warm_restart_cache<int, std::string> writer(wcfg);
    ASSERT_TRUE(writer.segment());

    std::vector<std::pair<int, std::string>> items;
    items.reserve(kItems);
    for (int i = 0; i < static_cast<int>(kItems); ++i) {
        auto h = primary.get(i);
        ASSERT_TRUE(h.has_value());
        items.emplace_back(i, *h);
    }
    ASSERT_EQ(writer.save(items.begin(), items.end()), kItems);

    // Reader: attach and rebuild a fresh safe_cache.
    safe_cache<int, std::string> restored{kItems * 2};
    shared_memory_config rcfg{name, kSegSize, true, false};
    warm_restart_cache<int, std::string> reader(rcfg);
    ASSERT_TRUE(reader.has_previous_data());

    auto n = reader.attach([&](const int& k, const std::string& v) {
        restored.set(k, v);
    });
    EXPECT_GE(n, kItems * 9 / 10)
        << "P1-3 acceptance: >=90% recovery";
    EXPECT_EQ(n, kItems);

    // The restored cache must serve all keys.
    for (int i = 0; i < static_cast<int>(kItems); ++i) {
        auto h = restored.get(i);
        ASSERT_TRUE(h.has_value()) << "key=" << i;
        EXPECT_EQ(*h, "v" + std::to_string(i));
    }
}

// ============================================================================
// Allocator-layer shared memory: slab_allocator + shared_memory_path
// (migrated from test_slab_memory.cpp)
// ============================================================================

#include <cstdio>
#include <filesystem>

TEST(SlabAllocatorSharedMemTest, FreshStartCreatesFile) {
    // Create a temporary file path
    std::string path = std::filesystem::temp_directory_path().string() + "/lru_test_shm_fresh";

    // Clean up any leftover file
    std::filesystem::remove(path);

    {
        slab_allocator::config cfg;
        cfg.shared_memory_path = path;
        slab_allocator alloc(cfg);

        // Fresh start: not a warm restart
        EXPECT_FALSE(alloc.is_warm_restart());

        // Shared memory should be active
        EXPECT_NE(alloc.shared_memory_data(), nullptr);
        EXPECT_GT(alloc.shared_memory_data_size(), 0u);

        // Allocation should work
        void* ptr = alloc.allocate(128);
        EXPECT_NE(ptr, nullptr);
        alloc.deallocate(ptr, 128);
    }

    // File should exist after allocator is destroyed
    EXPECT_TRUE(std::filesystem::exists(path));

    // Clean up
    std::filesystem::remove(path);
}

TEST(SlabAllocatorSharedMemTest, WarmRestartFromExistingFile) {
    std::string path = std::filesystem::temp_directory_path().string() + "/lru_test_shm_warm";

    // Clean up any leftover file
    std::filesystem::remove(path);

    // Phase 1: Fresh start — create the shared memory file
    {
        slab_allocator::config cfg;
        cfg.shared_memory_path = path;
        slab_allocator alloc(cfg);

        EXPECT_FALSE(alloc.is_warm_restart());
        EXPECT_NE(alloc.shared_memory_data(), nullptr);

        // Allocate and write some data
        void* ptr = alloc.allocate(128);
        EXPECT_NE(ptr, nullptr);
        // Write a pattern to the allocated block
        std::memset(ptr, 0xAB, 128);
        alloc.deallocate(ptr, 128);
    }

    // Phase 2: Warm restart — map the existing file
    {
        slab_allocator::config cfg;
        cfg.shared_memory_path = path;
        slab_allocator alloc(cfg);

        // Should detect warm restart
        EXPECT_TRUE(alloc.is_warm_restart());
        EXPECT_NE(alloc.shared_memory_data(), nullptr);
        EXPECT_GT(alloc.shared_memory_data_size(), 0u);

        // Allocation should still work
        void* ptr = alloc.allocate(128);
        EXPECT_NE(ptr, nullptr);
        alloc.deallocate(ptr, 128);
    }

    // Clean up
    std::filesystem::remove(path);
}

TEST(SlabAllocatorSharedMemTest, WarmRestartValidatesHeader) {
    std::string path = std::filesystem::temp_directory_path().string() + "/lru_test_shm_header";

    // Clean up any leftover file
    std::filesystem::remove(path);

    // Create with slab_size=65536
    {
        slab_allocator::config cfg;
        cfg.slab_size = 65536;
        cfg.shared_memory_path = path;
        slab_allocator alloc(cfg);
        EXPECT_FALSE(alloc.is_warm_restart());
    }

    // Warm restart with matching slab_size — should succeed
    {
        slab_allocator::config cfg;
        cfg.slab_size = 65536;
        cfg.shared_memory_path = path;
        slab_allocator alloc(cfg);
        EXPECT_TRUE(alloc.is_warm_restart());
    }

    // Warm restart with mismatched slab_size — should fall back to fresh start
    {
        slab_allocator::config cfg;
        cfg.slab_size = 32768;  // Different from the file's 65536
        cfg.shared_memory_path = path;
        slab_allocator alloc(cfg);
        // Header validation fails (slab_size mismatch), so falls back to fresh
        EXPECT_FALSE(alloc.is_warm_restart());
    }

    // Clean up
    std::filesystem::remove(path);
}

TEST(SlabAllocatorSharedMemTest, AllocateFromSharedMemory) {
    std::string path = std::filesystem::temp_directory_path().string() + "/lru_test_shm_alloc";

    // Clean up any leftover file
    std::filesystem::remove(path);

    {
        slab_allocator::config cfg;
        cfg.shared_memory_path = path;
        cfg.initial_slabs_per_class = 2;
        slab_allocator alloc(cfg);

        EXPECT_FALSE(alloc.is_warm_restart());

        // Allocate from various size classes
        void* p64 = alloc.allocate(64);
        void* p128 = alloc.allocate(128);
        void* p256 = alloc.allocate(256);
        void* p512 = alloc.allocate(512);
        void* p1024 = alloc.allocate(1024);

        EXPECT_NE(p64, nullptr);
        EXPECT_NE(p128, nullptr);
        EXPECT_NE(p256, nullptr);
        EXPECT_NE(p512, nullptr);
        EXPECT_NE(p1024, nullptr);

        // Verify pointers are within the shared memory data region
        auto* data_start = static_cast<char*>(alloc.shared_memory_data());
        auto data_size = alloc.shared_memory_data_size();
        auto check_in_region = [&](void* p) {
            auto offset = static_cast<char*>(p) - data_start;
            EXPECT_GE(offset, 0);
            EXPECT_LT(static_cast<std::size_t>(offset), data_size);
        };
        check_in_region(p64);
        check_in_region(p128);
        check_in_region(p256);
        check_in_region(p512);
        check_in_region(p1024);

        alloc.deallocate(p64, 64);
        alloc.deallocate(p128, 128);
        alloc.deallocate(p256, 256);
        alloc.deallocate(p512, 512);
        alloc.deallocate(p1024, 1024);
    }

    // Clean up
    std::filesystem::remove(path);
}

TEST(SlabAllocatorSharedMemTest, NoSharedMemPathUsesRegularAllocation) {
    // Empty shared_memory_path — should use regular allocation
    slab_allocator::config cfg;
    cfg.shared_memory_path = "";  // default
    slab_allocator alloc(cfg);

    EXPECT_FALSE(alloc.is_warm_restart());
    EXPECT_EQ(alloc.shared_memory_data(), nullptr);
    EXPECT_EQ(alloc.shared_memory_data_size(), 0u);

    // Regular allocation should work
    void* ptr = alloc.allocate(128);
    EXPECT_NE(ptr, nullptr);
    alloc.deallocate(ptr, 128);
}

TEST(SlabAllocatorSharedMemTest, WarmRestartPreservesSlabCounts) {
    std::string path = std::filesystem::temp_directory_path().string() + "/lru_test_shm_counts";

    // Clean up any leftover file
    std::filesystem::remove(path);

    // Phase 1: Fresh start with 2 initial slabs per class
    {
        slab_allocator::config cfg;
        cfg.shared_memory_path = path;
        cfg.initial_slabs_per_class = 2;
        slab_allocator alloc(cfg);

        EXPECT_FALSE(alloc.is_warm_restart());
        auto stats = alloc.get_stats();

        // Each class should have 2 slabs
        for (const auto& s : stats) {
            EXPECT_EQ(s.num_slabs, 2u);
        }
    }

    // Phase 2: Warm restart — should have the same slab counts
    {
        slab_allocator::config cfg;
        cfg.shared_memory_path = path;
        cfg.initial_slabs_per_class = 2;
        slab_allocator alloc(cfg);

        EXPECT_TRUE(alloc.is_warm_restart());
        auto stats = alloc.get_stats();

        // Should still have 2 slabs per class from the previous run
        for (const auto& s : stats) {
            EXPECT_EQ(s.num_slabs, 2u);
        }
    }

    // Clean up
    std::filesystem::remove(path);
}

TEST(SlabAllocatorSharedMemTest, WarmRestartWithCorruptFileFallsBack) {
    std::string path = std::filesystem::temp_directory_path().string() + "/lru_test_shm_corrupt";

    // Clean up any leftover file
    std::filesystem::remove(path);

    // Create a file with invalid content (not a valid LRUS header)
    {
        std::ofstream out(path, std::ios::binary);
        // Write some garbage that's large enough to contain a header
        std::vector<char> garbage(256, '\xFF');
        out.write(garbage.data(), garbage.size());
    }

    // Try to use this file — should fall back to fresh start
    {
        slab_allocator::config cfg;
        cfg.shared_memory_path = path;
        slab_allocator alloc(cfg);

        // Not a warm restart because the header is invalid
        EXPECT_FALSE(alloc.is_warm_restart());

        // Should still be functional (fresh start with shared memory)
        EXPECT_NE(alloc.shared_memory_data(), nullptr);

        void* ptr = alloc.allocate(128);
        EXPECT_NE(ptr, nullptr);
        alloc.deallocate(ptr, 128);
    }

    // Clean up
    std::filesystem::remove(path);
}
