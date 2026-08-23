// Unified LRU Cache — Serde Serialization unit tests
// Covers: serde<T> primitives, serialize/deserialize round-trips for
// all MM strategies, serialization header (v5) inspection/corruption
// detection, and concurrent save/load correctness.
//
// Note: cache_stats/key_statistics_tracker/operator==/operator<< tests
// were split out into test_cache_stats.cpp on 2026-07-26.

#include <gtest/gtest.h>
#include <atomic>
#include <cstring>
#include <string>
#include <span>
#include <thread>
#include <vector>
#include <optional>
#include <utility>

#include "../lru.hpp"
#include "../serialization.hpp"

using namespace lru;

// ============================================================================
// Serde — 基本类型序列化
// ============================================================================

TEST(SerdeTest, TrivialTypeInt) {
    detail::binary_writer w;
    serde<int>::serialize(w, 42);
    detail::binary_reader r(w.data().data(), w.data().size());
    auto v = serde<int>::deserialize(r);
    EXPECT_EQ(v, 42);
}

TEST(SerdeTest, StringType) {
    detail::binary_writer w;
    serde<std::string>::serialize(w, "hello world");
    detail::binary_reader r(w.data().data(), w.data().size());
    auto v = serde<std::string>::deserialize(r);
    EXPECT_EQ(v, "hello world");
}

TEST(SerdeTest, StringEmpty) {
    detail::binary_writer w;
    serde<std::string>::serialize(w, "");
    detail::binary_reader r(w.data().data(), w.data().size());
    auto v = serde<std::string>::deserialize(r);
    EXPECT_TRUE(v.empty());
}

TEST(SerdeTest, PairIntString) {
    detail::binary_writer w;
    std::pair<int, std::string> src(42, "answer");
    serde<decltype(src)>::serialize(w, src);
    detail::binary_reader r(w.data().data(), w.data().size());
    auto v = serde<decltype(src)>::deserialize(r);
    EXPECT_EQ(v.first, 42);
    EXPECT_EQ(v.second, "answer");
}

TEST(SerdeTest, PairIntInt) {
    detail::binary_writer w;
    std::pair<int, int> src(10, 20);
    serde<decltype(src)>::serialize(w, src);
    detail::binary_reader r(w.data().data(), w.data().size());
    auto v = serde<decltype(src)>::deserialize(r);
    EXPECT_EQ(v.first, 10);
    EXPECT_EQ(v.second, 20);
}

TEST(SerdeTest, VectorUint8) {
    detail::binary_writer w;
    std::vector<uint8_t> src = {1, 2, 3, 255};
    serde<decltype(src)>::serialize(w, src);
    detail::binary_reader r(w.data().data(), w.data().size());
    auto v = serde<decltype(src)>::deserialize(r);
    ASSERT_EQ(v.size(), 4u);
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[3], 255);
}

TEST(SerdeTest, VectorString) {
    detail::binary_writer w;
    std::vector<std::string> src = {"alpha", "beta", "gamma"};
    serde<decltype(src)>::serialize(w, src);
    detail::binary_reader r(w.data().data(), w.data().size());
    auto v = serde<decltype(src)>::deserialize(r);
    ASSERT_EQ(v.size(), 3u);
    EXPECT_EQ(v[0], "alpha");
    EXPECT_EQ(v[1], "beta");
    EXPECT_EQ(v[2], "gamma");
}

TEST(SerdeTest, OptionalWithValue) {
    detail::binary_writer w;
    std::optional<std::string> src = "present";
    serde<decltype(src)>::serialize(w, src);
    detail::binary_reader r(w.data().data(), w.data().size());
    auto v = serde<decltype(src)>::deserialize(r);
    ASSERT_TRUE(v.has_value());
    EXPECT_EQ(*v, "present");
}

TEST(SerdeTest, OptionalEmpty) {
    detail::binary_writer w;
    std::optional<std::string> src = std::nullopt;
    serde<decltype(src)>::serialize(w, src);
    detail::binary_reader r(w.data().data(), w.data().size());
    auto v = serde<decltype(src)>::deserialize(r);
    EXPECT_FALSE(v.has_value());
}

TEST(SerdeTest, NestedPairVector) {
    detail::binary_writer w;
    std::pair<int, std::vector<std::string>> src = {1, {"a", "b"}};
    serde<decltype(src)>::serialize(w, src);
    detail::binary_reader r(w.data().data(), w.data().size());
    auto v = serde<decltype(src)>::deserialize(r);
    EXPECT_EQ(v.first, 1);
    ASSERT_EQ(v.second.size(), 2u);
    EXPECT_EQ(v.second[0], "a");
    EXPECT_EQ(v.second[1], "b");
}

// ============================================================================
// FIFO Serialization
// ============================================================================

TEST(FifoSerializationTest, RoundTrip) {
    fifo_cache<int, std::string> c(100);
    c.set(1, "one");
    c.set(2, "two");
    c.set(3, "three");

    auto data = serialize(c.mm());
    fifo_cache<int, std::string> c2(100);
    deserialize(c2.mm(), std::span<const uint8_t>(data.data(), data.size()));

    EXPECT_EQ(c2.size(), 3u);
    EXPECT_TRUE(c2.contains(1));
    EXPECT_TRUE(c2.contains(2));
    EXPECT_TRUE(c2.contains(3));
}

TEST(FifoSerializationTest, RoundTripStringKeyValue) {
    fifo_cache<std::string, std::string> c(100);
    c.set("k1", "value one");
    c.set("k2", "value two");

    auto data = serialize(c.mm());
    fifo_cache<std::string, std::string> c2(100);
    deserialize(c2.mm(), std::span<const uint8_t>(data.data(), data.size()));

    EXPECT_EQ(c2.size(), 2u);
    EXPECT_TRUE(c2.contains("k1"));
    EXPECT_TRUE(c2.contains("k2"));
}

TEST(FifoSerializationTest, RoundTripPairValue) {
    fifo_cache<int, std::pair<int, std::string>> c(100);
    c.set(1, std::pair<int, std::string>(100, "hundred"));

    auto data = serialize(c.mm());
    fifo_cache<int, std::pair<int, std::string>> c2(100);
    deserialize(c2.mm(), std::span<const uint8_t>(data.data(), data.size()));

    EXPECT_EQ(c2.size(), 1u);
    auto v = c2.get(1);
    ASSERT_TRUE(v.has_value());
    EXPECT_EQ(v->first, 100);
    EXPECT_EQ(v->second, "hundred");
}

// ============================================================================
// LRU Serialization (with complex types)
// ============================================================================

TEST(LruSerializationTest, PairValue) {
    cache<int, std::pair<int, std::string>> c(100);
    c.set(1, std::pair<int, std::string>(100, "hundred"));

    auto data = serialize(c.mm());
    cache<int, std::pair<int, std::string>> c2(100);
    deserialize(c2.mm(), std::span<const uint8_t>(data.data(), data.size()));

    EXPECT_EQ(c2.size(), 1u);
    EXPECT_TRUE(c2.contains(1));
}

TEST(LruSerializationTest, StringValue) {
    cache<int, std::string> c(100);
    c.set(42, "the answer");

    auto data = serialize(c.mm());
    cache<int, std::string> c2(100);
    deserialize(c2.mm(), std::span<const uint8_t>(data.data(), data.size()));

    EXPECT_EQ(c2.size(), 1u);
    auto v = c2.get(42);
    ASSERT_TRUE(v.has_value());
    EXPECT_EQ(*v, "the answer");
}

// ============================================================================
// TinyLFU Serialization
// ============================================================================

TEST(TinyLfuSerializationTest, RoundTrip) {
    tiny_lfu_cache<int, int> c(100);
    c.set(1, 10);
    c.set(2, 20);

    auto data = serialize(c);
    tiny_lfu_cache<int, int> c2(100);
    deserialize(c2, std::span<const uint8_t>(data.data(), data.size()));

    EXPECT_EQ(c2.size(), 2u);
    EXPECT_TRUE(c2.contains(1));
    EXPECT_TRUE(c2.contains(2));
}

TEST(TinyLfuSerializationTest, StringKey) {
    tiny_lfu_cache<std::string, int> c(100);
    c.set("alpha", 1);
    c.set("beta", 2);

    auto data = serialize(c);
    tiny_lfu_cache<std::string, int> c2(100);
    deserialize(c2, std::span<const uint8_t>(data.data(), data.size()));

    EXPECT_EQ(c2.size(), 2u);
    EXPECT_TRUE(c2.contains("alpha"));
    EXPECT_TRUE(c2.contains("beta"));
}

// ============================================================================
// W-TinyLFU Serialization
// ============================================================================

TEST(WTinyLfuSerializationTest, RoundTrip) {
    w_tiny_lfu_cache<int, int> c(100);
    c.set(1, 10);
    c.set(2, 20);

    auto data = serialize(c);
    w_tiny_lfu_cache<int, int> c2(100);
    deserialize(c2, std::span<const uint8_t>(data.data(), data.size()));

    EXPECT_EQ(c2.size(), 2u);
    EXPECT_TRUE(c2.contains(1));
    EXPECT_TRUE(c2.contains(2));
}

TEST(WTinyLfuSerializationTest, StringKey) {
    w_tiny_lfu_cache<std::string, int> c(100);
    c.set("x", 100);
    c.set("y", 200);

    auto data = serialize(c);
    w_tiny_lfu_cache<std::string, int> c2(100);
    deserialize(c2, std::span<const uint8_t>(data.data(), data.size()));

    EXPECT_EQ(c2.size(), 2u);
    EXPECT_TRUE(c2.contains("x"));
    EXPECT_TRUE(c2.contains("y"));
}

// ============================================================================
// Serialization header inspection and corruption detection (v5)
// ============================================================================

TEST(SerializationHeaderTest, InspectValidV5Header) {
    cache<int, std::string> c(100);
    c.set(1, "one");
    c.set(2, "two");

    auto data = serialize(c.mm());
    auto info = inspect_serialization_header(std::span<const uint8_t>(data.data(), data.size()));

    EXPECT_TRUE(info.magic_ok);
    EXPECT_TRUE(info.version_supported);
    EXPECT_TRUE(info.checksum_ok);
    EXPECT_EQ(info.magic, 0x5355524Cu);
    EXPECT_EQ(info.version, 5u);
    EXPECT_EQ(info.item_count, 2u);
    EXPECT_EQ(info.mm_type, mm_type_id::lru);
    EXPECT_EQ(info.header_size, 36u);
    EXPECT_EQ(info.payload_offset, 36u);
    EXPECT_EQ(info.feature_flags, static_cast<serialization_feature>(0));
}

TEST(SerializationHeaderTest, DetectCorruptedPayload) {
    cache<int, std::string> c(100);
    c.set(1, "one");

    auto data = serialize(c.mm());
    ASSERT_FALSE(data.empty());
    // Corrupt the last byte of payload (safely past the 36-byte header).
    data.back() ^= 0xFF;

    auto info = inspect_serialization_header(std::span<const uint8_t>(data.data(), data.size()));
    EXPECT_TRUE(info.magic_ok);
    EXPECT_TRUE(info.version_supported);
    EXPECT_FALSE(info.checksum_ok);

    cache<int, std::string> c2(100);
    EXPECT_THROW(
        deserialize(c2.mm(), std::span<const uint8_t>(data.data(), data.size())),
        std::runtime_error);
}

TEST(SerializationHeaderTest, DetectBadMagic) {
    std::vector<uint8_t> data(36, 0);
    data[0] = 'X'; data[1] = 'X'; data[2] = 'X'; data[3] = 'X';
    // version = 5
    data[4] = 5; data[5] = 0; data[6] = 0; data[7] = 0;

    auto info = inspect_serialization_header(std::span<const uint8_t>(data.data(), data.size()));
    EXPECT_FALSE(info.magic_ok);
    EXPECT_TRUE(info.version_supported);
}

TEST(SerializationHeaderTest, DetectUnsupportedVersion) {
    std::vector<uint8_t> data(36, 0);
    data[0] = 0x4C; data[1] = 0x52; data[2] = 0x55; data[3] = 0x53; // "LRUS"
    // version = 99
    data[4] = 99; data[5] = 0; data[6] = 0; data[7] = 0;

    auto info = inspect_serialization_header(std::span<const uint8_t>(data.data(), data.size()));
    EXPECT_TRUE(info.magic_ok);
    EXPECT_FALSE(info.version_supported);
}

// ============================================================================
// Version support — only v5 is accepted
// ============================================================================

TEST(SerializationHeaderTest, V4HeaderMarkedUnsupported) {
    // Build a minimal v4 header manually
    std::vector<uint8_t> data(28, 0);
    // magic = "LRUS"
    data[0] = 0x4C; data[1] = 0x52; data[2] = 0x55; data[3] = 0x53;
    // version = 4
    data[4] = 4; data[5] = 0; data[6] = 0; data[7] = 0;
    // item_count = 0
    // mm_type = 0 (LRU)
    // header_size = 28
    data[16] = 28; data[17] = 0; data[18] = 0; data[19] = 0;

    auto info = inspect_serialization_header(std::span<const uint8_t>(data.data(), data.size()));
    EXPECT_TRUE(info.magic_ok);
    EXPECT_FALSE(info.version_supported);
    EXPECT_EQ(info.version, 4u);
}

TEST(SerializationHeaderTest, LoadV4DataThrows) {
    // Create v4 data from a v5 blob by stripping feature_flags
    cache<int, std::string> c(100);
    c.set(10, "ten");
    c.set(20, "twenty");

    auto v5_data = serialize(c.mm());

    // Build v4 format: strip feature_flags (8 bytes) between flags and checksum
    std::vector<uint8_t> v4_data;
    v4_data.insert(v4_data.end(), v5_data.begin(), v5_data.begin() + 24);
    uint32_t v4 = 4;
    std::memcpy(v4_data.data() + 4, &v4, sizeof(v4));
    uint32_t hdr28 = 28;
    std::memcpy(v4_data.data() + 16, &hdr28, sizeof(hdr28));
    v4_data.insert(v4_data.end(), v5_data.begin() + 32, v5_data.begin() + 36);
    v4_data.insert(v4_data.end(), v5_data.begin() + 36, v5_data.end());

    // v4 data should be rejected by both inspect and deserialize
    auto info = inspect_serialization_header(std::span<const uint8_t>(v4_data.data(), v4_data.size()));
    EXPECT_TRUE(info.magic_ok);
    EXPECT_FALSE(info.version_supported);

    cache<int, std::string> c2(100);
    EXPECT_THROW(
        deserialize(c2.mm(), std::span<const uint8_t>(v4_data.data(), v4_data.size())),
        std::runtime_error);
}

TEST(SerializationHeaderTest, FeatureFlagsBitwiseOps) {
    auto flags = serialization_feature::kRefcountWithFlags | serialization_feature::kChainedItems;
    EXPECT_TRUE(has_feature(flags, serialization_feature::kRefcountWithFlags));
    EXPECT_TRUE(has_feature(flags, serialization_feature::kChainedItems));
    EXPECT_FALSE(has_feature(flags, serialization_feature::kTlsCallbackRing));
    EXPECT_FALSE(has_feature(flags, serialization_feature::kAllocationClass));
}

TEST(SerializationHeaderTest, RoundTripAfterCorruptionDetection) {
    cache<int, std::string> c(100);
    for (int i = 0; i < 50; ++i) {
        c.set(i, "val" + std::to_string(i));
    }

    auto data = serialize(c.mm());
    auto info = inspect_serialization_header(std::span<const uint8_t>(data.data(), data.size()));
    ASSERT_TRUE(info.checksum_ok);

    cache<int, std::string> c2(100);
    deserialize(c2.mm(), std::span<const uint8_t>(data.data(), data.size()));
    EXPECT_EQ(c2.size(), c.size());
    for (int i = 0; i < 50; ++i) {
        auto v = c2.get(i);
        ASSERT_TRUE(v.has_value());
        EXPECT_EQ(*v, "val" + std::to_string(i));
    }
}

// ============================================================================
// Thread-safe save()/load() — Concurrent Serialization Tests
// ============================================================================

TEST(ConcurrentSerializationTest, ConcurrentWriteAndSave) {
    // One thread continuously writes to the cache, another continuously
    // calls save() to serialize. Must not crash.
    safe_cache<int, int> c(500);
    std::atomic<bool> stop{false};
    std::atomic<int> save_count{0};

    // Writer thread
    std::thread writer([&]() {
        for (int i = 0; !stop.load(std::memory_order_relaxed); ++i) {
            c.set(i % 500, i);
        }
    });

    // Saver thread
    std::thread saver([&]() {
        while (!stop.load(std::memory_order_relaxed)) {
            auto data = c.save();
            EXPECT_FALSE(data.empty());
            ++save_count;
        }
    });

    // Let them run for a short burst
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    stop.store(true, std::memory_order_relaxed);

    writer.join();
    saver.join();

    // Should have done at least a few saves
    EXPECT_GT(save_count.load(), 0);
}

TEST(ConcurrentSerializationTest, SaveAndLoadRoundTrip) {
    // Populate cache, save, then load into a new cache — should match.
    safe_cache<int, std::string> c(100);
    for (int i = 0; i < 50; ++i) {
        c.set(i, "val" + std::to_string(i));
    }

    auto data = c.save();
    EXPECT_FALSE(data.empty());

    safe_cache<int, std::string> c2(100);
    c2.load(std::span<const uint8_t>(data.data(), data.size()));

    EXPECT_EQ(c2.size(), c.size());
    for (int i = 0; i < 50; ++i) {
        auto v = c2.get(i);
        ASSERT_TRUE(v.has_value());
        EXPECT_EQ(*v, "val" + std::to_string(i));
    }
}

TEST(ConcurrentSerializationTest, ConcurrentWriteSaveAndLoad) {
    // Writer thread, saver thread, and a loader thread that periodically
    // loads from saved snapshots and verifies data validity.
    safe_cache<int, int> c(200);
    std::atomic<bool> stop{false};
    std::atomic<int> load_ok{0};

    // Pre-populate
    for (int i = 0; i < 50; ++i) {
        c.set(i, i * 10);
    }

    // Writer thread — continuously mutates the cache
    std::thread writer([&]() {
        for (int i = 0; !stop.load(std::memory_order_relaxed); ++i) {
            c.set(i % 200, i);
        }
    });

    // Loader thread — takes a snapshot via save(), loads it into a fresh cache,
    // and validates that the loaded data is internally consistent.
    std::thread loader([&]() {
        while (!stop.load(std::memory_order_relaxed)) {
            auto data = c.save();
            if (data.empty()) continue;

            safe_cache<int, int> tmp(200);
            // load() should not throw on valid data
            tmp.load(std::span<const uint8_t>(data.data(), data.size()));
            // Loaded cache size must be within capacity
            EXPECT_LE(tmp.size(), 200u);
            // Collect keys under the read lock, then verify outside it
            // to avoid read→write deadlock (rbegin holds read lock, get needs write lock).
            std::vector<int> keys;
            for (const auto& item : tmp.rbegin()) {
                keys.push_back(item.key);
            }
            for (auto k : keys) {
                auto v = tmp.get(k);
                EXPECT_TRUE(v.has_value());
            }
            ++load_ok;
        }
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    stop.store(true, std::memory_order_relaxed);

    writer.join();
    loader.join();

    EXPECT_GT(load_ok.load(), 0);
}
