// Unified LRU Cache Library — Serialization (saveState / restoreState)
// SPDX-License-Identifier: MIT
// Inspired by Facebook CacheLib's serialization model
//
// Supports:
//   - Binary streaming serialization for all MM strategies
//   - Warm restart: restore cached state after process restart
//   - Crash recovery: periodic snapshots for nil-restart resilience
//   - Cross-process/machine cache migration
//
// Format (binary, little-endian):
//   ┌──────────────────────────┐
//   │ Header (36 bytes, v5)    │
//   │  - magic        (4 bytes)│  "LRUS" (0x5355524C)
//   │  - version      (4 bytes)│  5
//   │  - item_count   (4 bytes)│
//   │  - mm_type      (4 bytes)│  0=LRU, 1=2Q, 2=TinyLFU, 3=W-TinyLFU
//   │  - header_size  (4 bytes)│  total header bytes (36)
//   │  - flags        (4 bytes)│  reserved
//   │  - feature_flags (8 bytes)│ serialization_feature bitmask
//   │  - checksum     (4 bytes)│  CRC32 of payload (config+extra+items)
//   ├──────────────────────────┤
//   │ MM Config (variable)     │
//   │  (insertion point,       │
//   │   refresh time, etc.)    │
//   ├──────────────────────────┤
//   │ List State (20 bytes)    │
//   │  - ins_point_pos (4 B)   │  0=MRU, n=LRUtail, UINT32_MAX=unset
//   │  - tail_size    (4 B)    │
//   │  - reserved     (12 B)   │
//   ├──────────────────────────┤
//   │ Item Array               │
//   │  For each item in        │
//   │  MRU→LRU order:          │
//   │  - key_len  (4 B)        │
//   │  - val_len  (4 B)        │
//   │  - key_data (var)        │
//   │  - val_data (var)        │
//   │  - update_time (4 B)     │
//   │  - flags     (1 B)       │
//   │  - queue_id  (1 B)       │
//   └──────────────────────────┘
//
// inspect_serialization_header() can detect corruption without full deserialization.

#ifndef LRU_SERIALIZATION_HPP
#define LRU_SERIALIZATION_HPP

#include <algorithm>
#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "core.hpp"
#include "mm.hpp"

namespace lru {

// ============================================================================
// Endianness check
// ============================================================================
// This serialization format uses little-endian byte order.
// Big-endian architectures are not supported.
static_assert(std::endian::native == std::endian::little,
    "Serialization requires little-endian architecture. Big-endian is not supported.");

// ============================================================================
// Serialization Constants
// ============================================================================

inline constexpr uint32_t kSerializationMagic = 0x5355524C; // "LRUS" little-endian
inline constexpr uint32_t kSerializationVersion = 5;        // current format with feature_flags

inline constexpr uint32_t kV5HeaderSize = 36;               // header: magic+version+count+mm+hdrsize+flags+feature_flags+checksum

/// Feature flags indicating which optional data sections are present
/// in the serialized binary. Written as uint64_t in the header.
enum class serialization_feature : uint64_t {
    kRefcountWithFlags = 1ULL << 0,  // items include refcount_with_flags data
    kChainedItems      = 1ULL << 1,  // chained item associations present
    kTlsCallbackRing   = 1ULL << 2,  // TLS callback ring state
    kAllocationClass   = 1ULL << 3,  // slab allocator with allocation classes
    // ... more flags as needed
};

/// Bitwise OR operator for combining serialization_feature flags.
inline constexpr serialization_feature operator|(serialization_feature a, serialization_feature b) {
    return static_cast<serialization_feature>(static_cast<uint64_t>(a) | static_cast<uint64_t>(b));
}

/// Bitwise AND operator for testing serialization_feature flags.
inline constexpr serialization_feature operator&(serialization_feature a, serialization_feature b) {
    return static_cast<serialization_feature>(static_cast<uint64_t>(a) & static_cast<uint64_t>(b));
}

/// Test whether a specific feature flag is set.
inline constexpr bool has_feature(serialization_feature flags, serialization_feature test) {
    return (flags & test) == test;
}

enum class mm_type_id : uint32_t {
    lru = 0,
    two_q = 1,
    tiny_lfu = 2,
    w_tiny_lfu = 3,
    fifo = 4,
    sharded_lru = 5,
};

// ============================================================================
// Serialization header inspection
// ============================================================================

/// Result of inspecting a serialized cache header without fully deserializing.
struct serialization_header_info {
    uint32_t magic = 0;
    uint32_t version = 0;
    uint32_t item_count = 0;
    mm_type_id mm_type = mm_type_id::lru;
    uint32_t header_size = 0;   // Total header bytes (v5: 36); 0 for older/invalid formats.
    uint32_t flags = 0;         // Header flags.
    serialization_feature feature_flags = static_cast<serialization_feature>(0); // Feature flags.
    uint32_t checksum = 0;      // Stored checksum.
    uint32_t computed_checksum = 0; // CRC32 computed from payload.
    bool magic_ok = false;
    bool version_supported = false;
    bool checksum_ok = false;   // True if the computed CRC32 matches the stored checksum.
    bool migrated = false;      // True if the data was migrated from an older version (always false since v4 support removed).
    std::size_t payload_offset = 0; // Byte offset where the payload begins.
};

/// Inspect a serialized buffer's header. Defined after detail::binary_reader and
/// detail::crc32 are available.
serialization_header_info inspect_serialization_header(std::span<const uint8_t> data);

// ============================================================================
// Binary Writer / Reader (utility)
// ============================================================================

namespace detail {

// ============================================================================
// CRC32 checksum (IEEE 802.3 polynomial)
// ============================================================================

/// Compute CRC32 over a byte range. Initial value is 0 (suitable for fresh
/// computation). The result is bitwise-NOT'd at the end, matching common
/// CRC32 implementations (e.g., zlib, Ethernet FCS).
inline uint32_t crc32(const uint8_t* data, std::size_t size) {
    static const uint32_t table[] = {
        0x00000000, 0x77073096, 0xee0e612c, 0x990951ba, 0x076dc419, 0x706af48f,
        0xe963a535, 0x9e6495a3, 0x0edb8832, 0x79dcb8a4, 0xe0d5e91e, 0x97d2d988,
        0x09b64c2b, 0x7eb17cbd, 0xe7b82d07, 0x90bf1d91, 0x1db71064, 0x6ab020f2,
        0xf3b97148, 0x84be41de, 0x1adad47d, 0x6ddde4eb, 0xf4d4b551, 0x83d385c7,
        0x136c9856, 0x646ba8c0, 0xfd62f97a, 0x8a65c9ec, 0x14015c4f, 0x63066cd9,
        0xfa0f3d63, 0x8d080df5, 0x3b6e20c8, 0x4c69105e, 0xd56041e4, 0xa2677172,
        0x3c03e4d1, 0x4b04d447, 0xd20d85fd, 0xa50ab56b, 0x35b5a8fa, 0x42b2986c,
        0xdbbbc9d6, 0xacbcf940, 0x32d86ce3, 0x45df5c75, 0xdcd60dcf, 0xabd13d59,
        0x26d930ac, 0x51de003a, 0xc8d75180, 0xbfd06116, 0x21b4f4b5, 0x56b3c423,
        0xcfba9599, 0xb8bda50f, 0x2802b89e, 0x5f058808, 0xc60cd9b2, 0xb10be924,
        0x2f6f7c87, 0x58684c11, 0xc1611dab, 0xb6662d3d, 0x76dc4190, 0x01db7106,
        0x98d220bc, 0xefd5102a, 0x71b18589, 0x06b6b51f, 0x9fbfe4a5, 0xe8b8d433,
        0x7807c9a2, 0x0f00f934, 0x9609a88e, 0xe10e9818, 0x7f6a0dbb, 0x086d3d2d,
        0x91646c97, 0xe6635c01, 0x6b6b51f4, 0x1c6c6162, 0x856530d8, 0xf262004e,
        0x6c0695ed, 0x1b01a57b, 0x8208f4c1, 0xf50fc457, 0x65b0d9c6, 0x12b7e950,
        0x8bbeb8ea, 0xfcb9887c, 0x62dd1ddf, 0x15da2d49, 0x8cd37cf3, 0xfbd44c65,
        0x4db26158, 0x3ab551ce, 0xa3bc0074, 0xd4bb30e2, 0x4adfa541, 0x3dd895d7,
        0xa4d1c46d, 0xd3d6f4fb, 0x4369e96a, 0x346ed9fc, 0xad678846, 0xda60b8d0,
        0x44042d73, 0x33031de5, 0xaa0a4c5f, 0xdd0d7cc9, 0x5005713c, 0x270241aa,
        0xbe0b1010, 0xc90c2086, 0x5768b525, 0x206f85b3, 0xb966d409, 0xce61e49f,
        0x5edef90e, 0x29d9c998, 0xb0d09822, 0xc7d7a8b4, 0x59b33d17, 0x2eb40d81,
        0xb7bd5c3b, 0xc0ba6cad, 0xedb88320, 0x9abfb3b6, 0x03b6e20c, 0x74b1d29a,
        0xead54739, 0x9dd277af, 0x04db2615, 0x73dc1683, 0xe3630b12, 0x94643b84,
        0x0d6d6a3e, 0x7a6a5aa8, 0xe40ecf0b, 0x9309ff9d, 0x0a00ae27, 0x7d079eb1,
        0xf00f9344, 0x8708a3d2, 0x1e01f268, 0x6906c2fe, 0xf762575d, 0x806567cb,
        0x196c3671, 0x6e6b06e7, 0xfed41b76, 0x89d32be0, 0x10da7a5a, 0x67dd4acc,
        0xf9b9df6f, 0x8ebeeff9, 0x17b7be43, 0x60b08ed5, 0xd6d6a3e8, 0xa1d1937e,
        0x38d8c2c4, 0x4fdff252, 0xd1bb67f1, 0xa6bc5767, 0x3fb506dd, 0x48b2364b,
        0xd80d2bda, 0xaf0a1b4c, 0x36034af6, 0x41047a60, 0xdf60efc3, 0xa867df55,
        0x316e8eef, 0x4669be79, 0xcb61b38c, 0xbc66831a, 0x256fd2a0, 0x5268e236,
        0xcc0c7795, 0xbb0b4703, 0x220216b9, 0x5505262f, 0xc5ba3bbe, 0xb2bd0b28,
        0x2bb45a92, 0x5cb36a04, 0xc2d7ffa7, 0xb5d0cf31, 0x2cd99e8b, 0x5bdeae1d,
        0x9b64c2b0, 0xec63f226, 0x756aa39c, 0x026d930a, 0x9c0906a9, 0xeb0e363f,
        0x72076785, 0x05005713, 0x95bf4a82, 0xe2b87a14, 0x7bb12bae, 0x0cb61b38,
        0x92d28e9b, 0xe5d5be0d, 0x7cdcefb7, 0x0bdbdf21, 0x86d3d2d4, 0xf1d4e242,
        0x68ddb3f8, 0x1fda836e, 0x81be16cd, 0xf6b9265b, 0x6fb077e1, 0x18b74777,
        0x88085ae6, 0xff0f6a70, 0x66063bca, 0x11010b5c, 0x8f659eff, 0xf862ae69,
        0x616bffd3, 0x166ccf45, 0xa00ae278, 0xd70dd2ee, 0x4e048354, 0x3903b3c2,
        0xa7672661, 0xd06016f7, 0x4969474d, 0x3e6e77db, 0xaed16a4a, 0xd9d65adc,
        0x40df0b66, 0x37d83bf0, 0xa9bcae53, 0xdebb9ec5, 0x47b2cf7f, 0x30b5ffe9,
        0xbdbdf21c, 0xcabac28a, 0x53b39330, 0x24b4a3a6, 0xbad03605, 0xcdd70693,
        0x54de5729, 0x23d967bf, 0xb3667a2e, 0xc4614ab8, 0x5d681b02, 0x2a6f2b94,
        0xb40bbe37, 0xc30c8ea1, 0x5a05df1b, 0x2d02ef8d
    };
    uint32_t crc = 0xFFFFFFFFu;
    for (std::size_t i = 0; i < size; ++i) {
        crc = table[(crc ^ data[i]) & 0xFFu] ^ (crc >> 8);
    }
    return ~crc;
}

/// Simple append-only binary buffer for serialization.
class binary_writer {
public:
    binary_writer() = default;

    /// Reserve space upfront for better performance.
    void reserve(std::size_t capacity) {
        buffer_.reserve(capacity);
    }

    /// Write raw bytes.
    void write_bytes(const void* data, std::size_t size) {
        const auto* p = static_cast<const uint8_t*>(data);
        buffer_.insert(buffer_.end(), p, p + size);
    }

    /// Write a value of any trivial type in little-endian.
    template <typename T>
        requires std::is_trivially_copyable_v<T>
    void write(const T& value) {
        write_bytes(&value, sizeof(T));
    }

    /// Write a length-prefixed byte sequence.
    void write_span(std::span<const uint8_t> data) {
        write(static_cast<uint32_t>(data.size()));
        write_bytes(data.data(), data.size());
    }

    const std::vector<uint8_t>& data() const noexcept { return buffer_; }
    std::vector<uint8_t> release() noexcept { return std::move(buffer_); }
    std::size_t size() const noexcept { return buffer_.size(); }

    /// Patch a value at a previously written offset (in bytes).
    /// Used for checksum fields written before the payload is complete.
    template <typename T>
        requires std::is_trivially_copyable_v<T>
    void patch_at(std::size_t offset, const T& value) {
        if (offset + sizeof(T) > buffer_.size()) {
            throw std::runtime_error("binary_writer: patch offset out of bounds");
        }
        std::memcpy(buffer_.data() + offset, &value, sizeof(T));
    }

private:
    std::vector<uint8_t> buffer_;
};

/// Simple read-only binary parser for deserialization.
class binary_reader {
public:
    binary_reader(const uint8_t* data, std::size_t size)
        : data_(data), size_(size), pos_(0) {}

    explicit binary_reader(std::span<const uint8_t> data)
        : data_(data.data()), size_(data.size()), pos_(0) {}

    /// Read a value of any trivial type.
    template <typename T>
        requires std::is_trivially_copyable_v<T>
    T read() {
        if (pos_ + sizeof(T) > size_) {
            throw std::runtime_error("binary_reader: unexpected end of data");
        }
        T value;
        std::memcpy(&value, data_ + pos_, sizeof(T));
        pos_ += sizeof(T);
        return value;
    }

    /// Read raw bytes into a buffer.
    void read_bytes(void* dst, std::size_t count) {
        if (pos_ + count > size_) {
            throw std::runtime_error("binary_reader: unexpected end of data");
        }
        std::memcpy(dst, data_ + pos_, count);
        pos_ += count;
    }

    /// Read a length-prefixed span and return a view into the buffer.
    std::span<const uint8_t> read_span() {
        auto len = read<uint32_t>();
        if (pos_ + len > size_) {
            throw std::runtime_error("binary_reader: span exceeds data bounds");
        }
        auto result = std::span<const uint8_t>(data_ + pos_, len);
        pos_ += len;
        return result;
    }

    bool eof() const noexcept { return pos_ >= size_; }
    std::size_t remaining() const noexcept { return size_ > pos_ ? size_ - pos_ : 0; }

private:
    const uint8_t* data_;
    std::size_t size_;
    std::size_t pos_;
};

} // namespace detail

// ============================================================================
// Serialization header inspection implementation
// ============================================================================

inline serialization_header_info inspect_serialization_header(std::span<const uint8_t> data) {
    serialization_header_info info;
    if (data.size() < 8) return info;

    detail::binary_reader r(data);
    info.magic = r.read<uint32_t>();
    info.magic_ok = (info.magic == kSerializationMagic);
    info.version = r.read<uint32_t>();
    info.version_supported = (info.version == kSerializationVersion);
    if (!info.magic_ok) return info;

    if (info.version != kSerializationVersion) {
        info.version_supported = false;
        return info;
    }

    if (data.size() < kV5HeaderSize) return info;

    info.item_count = r.read<uint32_t>();
    if (info.item_count > 10'000'000) {
        info.version_supported = false;
        return info;
    }

    info.mm_type = static_cast<mm_type_id>(r.read<uint32_t>());
    info.header_size = r.read<uint32_t>();
    info.flags = r.read<uint32_t>();

    // v5 header: feature_flags before checksum
    info.feature_flags = static_cast<serialization_feature>(r.read<uint64_t>());

    info.checksum = r.read<uint32_t>();

    // Validate header_size
    if (info.header_size < kV5HeaderSize || info.header_size > data.size()) {
        info.version_supported = false;
        return info;
    }

    info.payload_offset = info.header_size;
    info.computed_checksum = detail::crc32(
        data.data() + info.header_size,
        data.size() - info.header_size);
    info.checksum_ok = (info.computed_checksum == info.checksum);

    return info;
}

/// 依赖型 always_false（用于 if constexpr 的 else 分支中的 static_assert）。
template <typename T> inline constexpr bool always_false_v = false;

namespace detail {

// ---- Serde 声明（实现在后面）----

void serde_write(binary_writer& w, const std::string& s);
std::string serde_read(binary_reader& r, const std::string*);

void serde_write(binary_writer& w, const std::vector<uint8_t>& v);
std::vector<uint8_t> serde_read(binary_reader& r, const std::vector<uint8_t>*);

template <typename A, typename B>
void serde_write(binary_writer& w, const std::pair<A, B>& p);
template <typename A, typename B>
std::pair<A, B> serde_read(binary_reader& r, const std::pair<A, B>*);

template <typename T>
void serde_write(binary_writer& w, const std::vector<T>& v);
template <typename T>
std::vector<T> serde_read(binary_reader& r, const std::vector<T>*);

template <typename T>
void serde_write(binary_writer& w, const std::optional<T>& opt);
template <typename T>
std::optional<T> serde_read(binary_reader& r, const std::optional<T>*);

// ---- 平凡可复制回退 ----
template <typename T>
std::enable_if_t<std::is_trivially_copyable_v<T>>
serde_write(binary_writer& w, const T& value) {
    w.write(static_cast<uint32_t>(sizeof(T)));
    w.write_bytes(&value, sizeof(T));
}

template <typename T>
std::enable_if_t<std::is_trivially_copyable_v<T>, T>
serde_read(binary_reader& r, const T*) {
    auto len = r.read<uint32_t>();
    if (len != sizeof(T))
        throw std::runtime_error("serde: size mismatch for trivially copyable type");
    T value; r.read_bytes(&value, sizeof(T));
    return value;
}

} // namespace detail

/// 序列化定制点（自由函数转发器）。
template <typename T>
struct serde {
    static void serialize(detail::binary_writer& w, const T& value) {
        detail::serde_write(w, value);
    }
    static T deserialize(detail::binary_reader& r) {
        return detail::serde_read(r, (const T*)nullptr);
    }
};

// ---- 自由函数实现 ----

namespace detail {

inline void serde_write(binary_writer& w, const std::string& s) {
    w.write(static_cast<uint32_t>(s.size()));
    w.write_bytes(s.data(), s.size());
}
inline std::string serde_read(binary_reader& r, const std::string*) {
    auto len = r.read<uint32_t>();
    if (len > 1'000'000) {
        throw std::runtime_error("deserialization: string length exceeds reasonable limit");
    }
    std::string s(len, '\0');
    if (len > 0) r.read_bytes(s.data(), len);
    return s;
}

inline void serde_write(binary_writer& w, const std::vector<uint8_t>& v) {
    w.write(static_cast<uint32_t>(v.size()));
    w.write_bytes(v.data(), v.size());
}
inline std::vector<uint8_t> serde_read(binary_reader& r, const std::vector<uint8_t>*) {
    auto len = r.read<uint32_t>();
    if (len > 100'000'000) {  // ~100MB upper bound for byte vectors
        throw std::runtime_error("deserialization: byte vector length exceeds reasonable limit");
    }
    std::vector<uint8_t> v(len);
    if (len > 0) r.read_bytes(v.data(), len);
    return v;
}

template <typename A, typename B>
void serde_write(binary_writer& w, const std::pair<A, B>& p) {
    serde_write(w, p.first);
    serde_write(w, p.second);
}
template <typename A, typename B>
std::pair<A, B> serde_read(binary_reader& r, const std::pair<A, B>*) {
    auto a = serde_read(r, (const A*)nullptr);
    auto b = serde_read(r, (const B*)nullptr);
    return std::pair<A, B>(std::move(a), std::move(b));
}

template <typename T>
void serde_write(binary_writer& w, const std::vector<T>& v) {
    w.write(static_cast<uint32_t>(v.size()));
    for (const auto& elem : v) serde_write(w, elem);
}
template <typename T>
std::vector<T> serde_read(binary_reader& r, const std::vector<T>*) {
    auto n = r.read<uint32_t>();
    if (n > 10'000'000) {  // ~40MB upper bound for uint32_t vectors
        throw std::runtime_error("deserialization: vector size exceeds reasonable limit");
    }
    std::vector<T> res; res.reserve(n);
    for (uint32_t i = 0; i < n; ++i) res.push_back(serde_read(r, (const T*)nullptr));
    return res;
}

template <typename T>
void serde_write(binary_writer& w, const std::optional<T>& opt) {
    w.write(static_cast<uint8_t>(opt.has_value() ? 1 : 0));
    if (opt.has_value()) serde_write(w, *opt);
}
template <typename T>
std::optional<T> serde_read(binary_reader& r, const std::optional<T>*) {
    auto has = r.read<uint8_t>();
    if (has) return std::optional<T>(serde_read(r, (const T*)nullptr));
    return std::nullopt;
}

} // namespace detail

// ============================================================================
// Serialized MM Config (shared across all MM types)
// ============================================================================

/// Config state that can be serialized/deserialized.
struct serialized_mm_config {
    uint32_t lru_refresh_time = 60;
    double lru_refresh_ratio = 0.0;
    bool update_on_write = false;
    bool update_on_read = true;
    bool try_lock_update = true;
    uint8_t lru_insertion_point_spec = 0;
    uint32_t mm_reconfigure_interval_secs = 0;

    // 2Q-specific
    double hot_ratio = 0.3;
    double warm_ratio = 0.4;
    bool rebalance_on_record_access = true;

    // TinyLFU / W-TinyLFU CountMinSketch error rate (v3+)
    double cms_error_rate = 0.01;

    void write(detail::binary_writer& w) const {
        w.write(lru_refresh_time);
        w.write(lru_refresh_ratio);
        w.write(static_cast<uint8_t>(update_on_write ? 1 : 0));
        w.write(static_cast<uint8_t>(update_on_read ? 1 : 0));
        w.write(static_cast<uint8_t>(try_lock_update ? 1 : 0));
        w.write(lru_insertion_point_spec);
        w.write(mm_reconfigure_interval_secs);
        w.write(hot_ratio);
        w.write(warm_ratio);
        w.write(static_cast<uint8_t>(rebalance_on_record_access ? 1 : 0));
        w.write(cms_error_rate);
    }

    void read(detail::binary_reader& r) {
        lru_refresh_time = r.read<uint32_t>();
        lru_refresh_ratio = r.read<double>();
        update_on_write = r.read<uint8_t>() != 0;
        update_on_read = r.read<uint8_t>() != 0;
        try_lock_update = r.read<uint8_t>() != 0;
        lru_insertion_point_spec = r.read<uint8_t>();
        mm_reconfigure_interval_secs = r.read<uint32_t>();
        hot_ratio = r.read<double>();
        warm_ratio = r.read<double>();
        rebalance_on_record_access = r.read<uint8_t>() != 0;
        cms_error_rate = r.read<double>();
    }
};

// ============================================================================
// Serialized Item
// ============================================================================

/// A single cache item in serialized form.
template <typename Key, typename Value>
struct serialized_item {
    Key key;
    Value value;
    uint32_t update_time = 0;
    uint8_t flags = 0;    // kTailFlag | kAccessedFlag
    uint8_t queue_id = 0; // for multi-queue strategies

    void write(detail::binary_writer& w) const {
        serde<Key>::serialize(w, key);
        serde<Value>::serialize(w, value);
        w.write(update_time);
        w.write(flags);
        w.write(queue_id);
    }

    void read(detail::binary_reader& r) {
        key   = serde<Key>::deserialize(r);
        value = serde<Value>::deserialize(r);
        update_time = r.read<uint32_t>();
        flags = r.read<uint8_t>();
        queue_id = r.read<uint8_t>();
    }
};

// ============================================================================
// Serialized List State (NEW in v2 — faithful restore)
// ============================================================================

/// S0: 链表状态，用于 faithful restore（保存/恢复插入点信息和 tail 段）。
struct serialized_list_state {
    /// 插入点位置（从 MRU head = 0 计）。max() = 未设置（spec=0）。
    uint32_t insertion_point_pos = std::numeric_limits<uint32_t>::max();
    /// tail 段大小（仅 spec>0 时有效）。
    uint32_t tail_size = 0;

    void write(detail::binary_writer& w) const {
        w.write(insertion_point_pos);
        w.write(tail_size);
        // 12 字节预留（结构对齐）
        uint32_t reserved[3] = {0, 0, 0};
        w.write(reserved[0]);
        w.write(reserved[1]);
        w.write(reserved[2]);
    }

    void read(detail::binary_reader& r) {
        insertion_point_pos = r.read<uint32_t>();
        tail_size = r.read<uint32_t>();
        // 跳过 12 字节预留
        r.read<uint32_t>();
        r.read<uint32_t>();
        r.read<uint32_t>();
    }
};

// ============================================================================
// Serialization API — Free Functions
// ============================================================================
//
// ⚠️  THREAD SAFETY WARNING
// These free functions are NOT thread-safe. They access the MM object's
// internal data without acquiring any lock. In a multi-threaded context,
// concurrent calls to set/del/get while serialize() or deserialize() is
// running can cause data corruption or crashes.
//
// For thread-safe serialization, use unified_cache::save() / load() instead,
// which automatically acquire the appropriate read/write locks.
// ============================================================================

// ============================================================================
// 通用 serialize / deserialize 实现模板（消除 ~300 行重复代码）
// ============================================================================

namespace detail {

// Types defined in parent lru:: namespace
using lru::serialized_mm_config;
using lru::serialized_item;
using lru::serialized_list_state;
using lru::mm_type_id;
using lru::kSerializationMagic;
using lru::kSerializationVersion;
using lru::kV5HeaderSize;
using lru::serialization_feature;
using lru::has_feature;
// kListStateSize is unused; kept for reference in serialized_list_state docs

/// 通用序列化实现：header + config + extra_state + items
/// v5 header adds feature_flags before checksum.
/// Checksum covers everything after the checksum field (config + extra + items).
template <typename MM, typename WriteConfigFn, typename WriteExtraFn>
std::vector<uint8_t> serialize_impl(
    const MM& cache,
    mm_type_id type_id,
    serialization_feature feature_flags,
    WriteConfigFn&& write_config,
    WriteExtraFn&& write_extra)
{
    using Key = typename MM::key_type;
    using Value = typename MM::mapped_type;

    binary_writer w;
    w.reserve(64 + cache.size() * (sizeof(Key) + sizeof(Value) + 64));

    // Header (v5: 36 bytes)
    w.write(kSerializationMagic);
    w.write(kSerializationVersion);
    w.write(static_cast<uint32_t>(cache.size()));
    w.write(static_cast<uint32_t>(type_id));
    w.write(kV5HeaderSize);                          // header_size
    w.write(static_cast<uint32_t>(0));               // flags
    w.write(static_cast<uint64_t>(feature_flags));   // feature_flags
    auto checksum_offset = w.size();
    w.write(static_cast<uint32_t>(0));               // checksum placeholder

    // MM Config
    serialized_mm_config cfg;
    write_config(cfg);
    cfg.write(w);

    // Extra state (per MM type: list_state, CMS, etc.)
    write_extra(w);

    // Items (in iteration order = MRU→LRU / multi-queue order)
    for (auto it = cache.begin(); it != cache.end(); ++it) {
        serialized_item<Key, Value> item;
        item.key = it->key;
        item.value = it->value;
        item.update_time = it->hook.update_time;
        item.flags = it->hook.flags;
        item.queue_id = it->queue_id;
        item.write(w);
    }

    // Compute and patch CRC32 over everything after the checksum field.
    uint32_t checksum = detail::crc32(
        w.data().data() + checksum_offset + sizeof(uint32_t),
        w.size() - checksum_offset - sizeof(uint32_t));
    w.patch_at(checksum_offset, checksum);

    return w.release();
}

/// 通用反序列化实现：header -> config -> restore_fn(reader, item_count, cfg)
/// restore_fn 负责读取 extra_state + items + 重建缓存。
/// 版本 != current_version 时抛出异常。
template <typename MM, typename RestoreFn>
void deserialize_impl(
    MM& cache,
    mm_type_id expected_type_id,
    std::span<const uint8_t> data,
    RestoreFn&& restore)
{
    using Key = typename MM::key_type;
    using Value = typename MM::mapped_type;

    binary_reader r(data);

    // Header
    auto magic = r.read<uint32_t>();
    if (magic != kSerializationMagic) {
        throw std::runtime_error("deserialize: invalid magic number");
    }
    auto version = r.read<uint32_t>();
    auto item_count = r.read<uint32_t>();
    if (item_count > 10'000'000) {  // ~400MB upper bound for cache items
        throw std::runtime_error("deserialize: item_count exceeds reasonable limit");
    }
    auto mm_type_raw = r.read<uint32_t>();
    if (static_cast<mm_type_id>(mm_type_raw) != expected_type_id) {
        throw std::runtime_error("deserialize: mm_type mismatch (expected " +
            std::to_string(static_cast<uint32_t>(expected_type_id)) +
            ", got " + std::to_string(mm_type_raw) + ")");
    }

    if (version != kSerializationVersion) {
        throw std::runtime_error("deserialize: unsupported version " + std::to_string(version) +
            " (only version " + std::to_string(kSerializationVersion) + " is supported)");
    }

    auto header_size = r.read<uint32_t>();
    if (header_size < kV5HeaderSize || header_size > data.size()) {
        throw std::runtime_error("deserialize: invalid header_size");
    }
    [[maybe_unused]] auto flags = r.read<uint32_t>();

    // Read feature_flags (v5 header)
    serialization_feature feature_flags = static_cast<serialization_feature>(r.read<uint64_t>());

    auto stored_checksum = r.read<uint32_t>();
    std::size_t payload_offset = header_size;
    uint32_t computed_checksum = detail::crc32(
        data.data() + payload_offset,
        data.size() - payload_offset);
    if (computed_checksum != stored_checksum) {
        throw std::runtime_error("deserialize: header checksum mismatch (data corrupted)");
    }

    // Clear existing contents
    cache.flush();

    // MM Config
    serialized_mm_config cfg;
    cfg.read(r);

    // MM-specific restore: reads extra_state + items + rebuilds
    restore(r, item_count, cfg);
}

// ============================================================================
// COW Snapshot Serialization
// ============================================================================
//
// These structures and functions enable copy-on-write serialization:
//   1. Collect a snapshot of the cache under a brief read lock
//   2. Release the lock
//   3. Serialize the snapshot without holding any lock
//
// This minimizes lock holding time during save(), which is critical for
// large caches where serialization may take seconds.
// ============================================================================

/// Snapshot data for serialization — holds all data needed to serialize a
/// cache without accessing the MM object. Collected under a brief read lock.
template <typename Key, typename Value>
struct cache_snapshot {
    mm_type_id type = mm_type_id::lru;
    serialization_feature feature_flags = static_cast<serialization_feature>(0);
    serialized_mm_config config;
    serialized_list_state list_state;            // LRU-specific extra state
    std::vector<uint32_t> cms_state;             // TinyLFU / W-TinyLFU extra state
    std::vector<serialized_item<Key, Value>> items;
};

/// Serialize a cache_snapshot into binary format.
/// This is the lock-free second phase of COW serialization.
template <typename Key, typename Value>
std::vector<uint8_t> serialize_from_snapshot(const cache_snapshot<Key, Value>& snap) {
    binary_writer w;
    w.reserve(64 + snap.items.size() * (sizeof(Key) + sizeof(Value) + 64));

    // Header (v5: 36 bytes)
    w.write(kSerializationMagic);
    w.write(kSerializationVersion);
    w.write(static_cast<uint32_t>(snap.items.size()));
    w.write(static_cast<uint32_t>(snap.type));
    w.write(kV5HeaderSize);                                    // header_size
    w.write(static_cast<uint32_t>(0));                         // flags
    w.write(static_cast<uint64_t>(snap.feature_flags));        // feature_flags
    auto checksum_offset = w.size();
    w.write(static_cast<uint32_t>(0));                         // checksum placeholder

    // MM Config
    snap.config.write(w);

    // Extra state (per MM type)
    if (snap.type == mm_type_id::lru || snap.type == mm_type_id::sharded_lru) {
        snap.list_state.write(w);
    } else if (snap.type == mm_type_id::tiny_lfu || snap.type == mm_type_id::w_tiny_lfu) {
        // CMS state
        auto cms_words = static_cast<uint32_t>(snap.cms_state.size());
        w.write(cms_words);
        for (auto cw : snap.cms_state) w.write(cw);
    }
    // 2Q and FIFO have no extra state

    // Items (in iteration order = MRU→LRU / multi-queue order)
    for (const auto& item : snap.items) {
        item.write(w);
    }

    // Compute and patch CRC32
    uint32_t checksum = detail::crc32(
        w.data().data() + checksum_offset + sizeof(uint32_t),
        w.size() - checksum_offset - sizeof(uint32_t));
    w.patch_at(checksum_offset, checksum);

    return w.release();
}

// ============================================================================
// Parsed Deserialization Data (two-phase load)
// ============================================================================
//
// Pre-parsed deserialization data for two-phase load():
//   Phase 1 (no lock): parse header + config + extra state + items
//   Phase 2 (write lock): flush + rebuild from parsed data
//
// This minimizes the write lock hold time during load().
// ============================================================================

/// Pre-parsed deserialization data, extracted from binary without modifying
/// any cache state. Used by load() to separate parsing from rebuilding.
template <typename Key, typename Value>
struct parsed_deserialization_data {
    serialization_feature feature_flags = static_cast<serialization_feature>(0);
    serialized_mm_config config;
    serialized_list_state list_state;            // LRU-specific extra state
    std::vector<uint32_t> cms_state;             // TinyLFU / W-TinyLFU extra state
    std::vector<serialized_item<Key, Value>> items;
};

/// Phase 1 of two-phase deserialization: parse binary data into a
/// parsed_deserialization_data without any lock or cache mutation.
/// Throws on format errors, but does NOT call cache.flush() or modify
/// the cache in any way.
/// Only v5 format is supported.
template <typename Key, typename Value>
parsed_deserialization_data<Key, Value> parse_serialized_data(
    mm_type_id expected_type_id,
    std::span<const uint8_t> data)
{
    parsed_deserialization_data<Key, Value> result;

    detail::binary_reader r(data);

    // Header
    auto magic = r.read<uint32_t>();
    if (magic != kSerializationMagic) {
        throw std::runtime_error("deserialize: invalid magic number");
    }
    auto version = r.read<uint32_t>();
    auto item_count = r.read<uint32_t>();
    if (item_count > 10'000'000) {
        throw std::runtime_error("deserialize: item_count exceeds reasonable limit");
    }
    auto mm_type_raw = r.read<uint32_t>();
    if (static_cast<mm_type_id>(mm_type_raw) != expected_type_id) {
        throw std::runtime_error("deserialize: mm_type mismatch (expected " +
            std::to_string(static_cast<uint32_t>(expected_type_id)) +
            ", got " + std::to_string(mm_type_raw) + ")");
    }
    if (version != kSerializationVersion) {
        throw std::runtime_error("deserialize: unsupported version " + std::to_string(version) +
            " (only version " + std::to_string(kSerializationVersion) + " is supported)");
    }

    auto header_size = r.read<uint32_t>();
    if (header_size < kV5HeaderSize || header_size > data.size()) {
        throw std::runtime_error("deserialize: invalid header_size");
    }
    [[maybe_unused]] auto flags = r.read<uint32_t>();

    // Read feature_flags (v5 header)
    result.feature_flags = static_cast<serialization_feature>(r.read<uint64_t>());

    auto stored_checksum = r.read<uint32_t>();
    std::size_t payload_offset = header_size;
    uint32_t computed_checksum = detail::crc32(
        data.data() + payload_offset,
        data.size() - payload_offset);
    if (computed_checksum != stored_checksum) {
        throw std::runtime_error("deserialize: header checksum mismatch (data corrupted)");
    }

    // MM Config
    result.config.read(r);

    // Extra state (per MM type)
    if (expected_type_id == mm_type_id::lru || expected_type_id == mm_type_id::sharded_lru) {
        result.list_state.read(r);
    } else if (expected_type_id == mm_type_id::tiny_lfu || expected_type_id == mm_type_id::w_tiny_lfu) {
        auto cms_words = r.read<uint32_t>();
        if (cms_words > 1'000'000) {
            throw std::runtime_error("deserialization: cms_words exceeds reasonable limit");
        }
        result.cms_state.resize(cms_words);
        for (uint32_t i = 0; i < cms_words; ++i) {
            result.cms_state[i] = r.read<uint32_t>();
        }
    }

    // Items
    result.items.reserve(item_count);
    for (uint32_t i = 0; i < item_count; ++i) {
        serialized_item<Key, Value> item;
        item.read(r);
        result.items.push_back(std::move(item));
    }

    return result;
}

} // namespace detail

// ============================================================================
// Serde — 序列化定制点（自由函数重载，彻底排除一切模板元问题）
// ============================================================================

// 所有已知类型的序列化实现通过重载自由函数完成。
// 对未知的平凡可复制类型用 enable_if 匹配。
// 用户可通过特化 serde<T> 覆盖默认行为。

/// Serialize an enhanced LRU cache.
template <typename Key, typename Value, typename Hash, typename KeyEqual>
std::vector<uint8_t> serialize(const mm_lru<Key, Value, Hash, KeyEqual>& cache) {
    return detail::serialize_impl(cache, mm_type_id::lru,
        static_cast<serialization_feature>(0),
        [&](serialized_mm_config& cfg) {
            const auto& c = cache.config();
            cfg.lru_refresh_time = cache.refresh_time();
            cfg.lru_insertion_point_spec = c.lru_insertion_point_spec;
            cfg.update_on_read = c.update_on_read;
            cfg.update_on_write = c.update_on_write;
            cfg.try_lock_update = c.try_lock_update;
            cfg.lru_refresh_ratio = c.lru_refresh_ratio;
            cfg.mm_reconfigure_interval_secs = c.mm_reconfigure_interval_secs;
        },
        [&](detail::binary_writer& w) {
            serialized_list_state lst;
            auto ins_pos = cache.insertion_point_position();
            if (ins_pos != mm_lru<Key, Value, Hash, KeyEqual>::npos) {
                lst.insertion_point_pos = static_cast<uint32_t>(ins_pos);
                lst.tail_size = static_cast<uint32_t>(cache.tail_size());
            }
            lst.write(w);
        });
}

/// Deserialize into an enhanced LRU cache (overwrites existing contents).
template <typename Key, typename Value, typename Hash, typename KeyEqual>
void deserialize(mm_lru<Key, Value, Hash, KeyEqual>& cache,
                 std::span<const uint8_t> data) {
    detail::deserialize_impl(cache, mm_type_id::lru, data,
        [&](detail::binary_reader& r, uint32_t item_count,
            serialized_mm_config& cfg) {
            using ser_item = serialized_item<Key, Value>;

            serialized_list_state lst;
            lst.read(r);
            uint32_t ins_pos = lst.insertion_point_pos;
            uint32_t tail_sz = lst.tail_size;

            // Config: apply from serialized data
            auto new_cfg = cache.config();
            new_cfg.lru_insertion_point_spec = cfg.lru_insertion_point_spec;
            new_cfg.update_on_read = cfg.update_on_read;
            new_cfg.update_on_write = cfg.update_on_write;
            new_cfg.try_lock_update = cfg.try_lock_update;
            new_cfg.lru_refresh_ratio = cfg.lru_refresh_ratio;
            new_cfg.mm_reconfigure_interval_secs = cfg.mm_reconfigure_interval_secs;
            new_cfg.default_lru_refresh_time = cfg.lru_refresh_time;

            // Read all items
            std::vector<ser_item> items;
            items.reserve(item_count);
            for (uint32_t i = 0; i < item_count; ++i) {
                ser_item item;
                item.read(r);
                items.push_back(std::move(item));
            }

            // S0: Faithful rebuild
            cache.rebuild_from_serialized(items.begin(), items.end(), ins_pos, tail_sz);

            // Apply config AFTER rebuild to set lru_refresh_time
            cache.set_config(new_cfg);
        });
}

// ============================================================================
// Serialization — 2Q
// ============================================================================

template <typename Key, typename Value, typename Hash, typename KeyEqual>
std::vector<uint8_t> serialize(const mm_2q<Key, Value, Hash, KeyEqual>& cache) {
    return detail::serialize_impl(cache, mm_type_id::two_q,
        static_cast<serialization_feature>(0),
        [&](serialized_mm_config& cfg) {
            const auto& c = cache.config();
            cfg.lru_refresh_time = cache.refresh_time();
            cfg.update_on_read = c.update_on_read;
            cfg.update_on_write = c.update_on_write;
            cfg.try_lock_update = c.try_lock_update;
            cfg.lru_refresh_ratio = c.lru_refresh_ratio;
            cfg.mm_reconfigure_interval_secs = c.mm_reconfigure_interval_secs;
            cfg.hot_ratio = c.hot_ratio;
            cfg.warm_ratio = c.warm_ratio;
            cfg.rebalance_on_record_access = c.rebalance_on_record_access;
        },
        [&](detail::binary_writer&) {}); // 2Q has no extra state
}

template <typename Key, typename Value, typename Hash, typename KeyEqual>
void deserialize(mm_2q<Key, Value, Hash, KeyEqual>& cache,
                 std::span<const uint8_t> data) {
    detail::deserialize_impl(cache, mm_type_id::two_q, data,
        [&](detail::binary_reader& r, uint32_t item_count,
            serialized_mm_config& cfg) {
            using ser_item = serialized_item<Key, Value>;

            auto new_cfg = cache.config();
            new_cfg.update_on_read = cfg.update_on_read;
            new_cfg.update_on_write = cfg.update_on_write;
            new_cfg.try_lock_update = cfg.try_lock_update;
            new_cfg.lru_refresh_ratio = cfg.lru_refresh_ratio;
            new_cfg.mm_reconfigure_interval_secs = cfg.mm_reconfigure_interval_secs;
            new_cfg.hot_ratio = cfg.hot_ratio;
            new_cfg.warm_ratio = cfg.warm_ratio;
            new_cfg.rebalance_on_record_access = cfg.rebalance_on_record_access;
            new_cfg.default_lru_refresh_time = cfg.lru_refresh_time;
            cache.set_config(new_cfg);

            std::vector<ser_item> items;
            items.reserve(item_count);
            for (uint32_t i = 0; i < item_count; ++i) {
                ser_item item;
                item.read(r);
                items.push_back(std::move(item));
            }
            cache.rebuild_from_serialized(items.begin(), items.end());
        });
}

// ============================================================================
// Serialization — FIFO
// ============================================================================

template <typename Key, typename Value, typename Hash, typename KeyEqual>
std::vector<uint8_t> serialize(const mm_fifo<Key, Value, Hash, KeyEqual>& cache) {
    return detail::serialize_impl(cache, mm_type_id::fifo,
        static_cast<serialization_feature>(0),
        [&](serialized_mm_config&) {}, // minimal config
        [&](detail::binary_writer&) {}); // FIFO has no extra state
}

template <typename Key, typename Value, typename Hash, typename KeyEqual>
void deserialize(mm_fifo<Key, Value, Hash, KeyEqual>& cache,
                 std::span<const uint8_t> data) {
    detail::deserialize_impl(cache, mm_type_id::fifo, data,
        [&](detail::binary_reader& r, uint32_t item_count,
            serialized_mm_config& /*cfg*/) {
            using ser_item = serialized_item<Key, Value>;
            for (uint32_t i = 0; i < item_count; ++i) {
                ser_item item;
                item.read(r);
                cache.set(item.key, std::move(item.value));
            }
        });
}

// ============================================================================
// Serialization — TinyLFU
// ============================================================================

template <typename Key, typename Value, typename Hash, typename KeyEqual>
std::vector<uint8_t> serialize(const mm_tiny_lfu<Key, Value, Hash, KeyEqual>& cache) {
    return detail::serialize_impl(cache, mm_type_id::tiny_lfu,
        static_cast<serialization_feature>(0),
        [&](serialized_mm_config& cfg) {
            const auto& c = cache.config();
            cfg.lru_refresh_time = cache.refresh_time();
            cfg.lru_refresh_ratio = c.lru_refresh_ratio;
            cfg.cms_error_rate = c.cms_error_rate;
            cfg.try_lock_update = c.try_lock_update;
        },
        [&](detail::binary_writer& w) {
            // S3: 序列化 CountMinSketch 状态
            uint32_t cms_words = static_cast<uint32_t>(cache.sketch().serialized_state_words());
            w.write(cms_words);
            std::vector<uint32_t> cms_buf(cms_words);
            cache.sketch().save_state(cms_buf.begin());
            for (auto cw : cms_buf) w.write(cw);
        });
}

template <typename Key, typename Value, typename Hash, typename KeyEqual>
void deserialize(mm_tiny_lfu<Key, Value, Hash, KeyEqual>& cache,
                 std::span<const uint8_t> data) {
    detail::deserialize_impl(cache, mm_type_id::tiny_lfu, data,
        [&](detail::binary_reader& r, uint32_t item_count,
            serialized_mm_config& cfg) {
            using ser_item = serialized_item<Key, Value>;

            // Restore config including cms_error_rate
            auto new_cfg = cache.config();
            new_cfg.lru_refresh_ratio = cfg.lru_refresh_ratio;
            new_cfg.cms_error_rate = cfg.cms_error_rate;
            new_cfg.try_lock_update = cfg.try_lock_update;
            new_cfg.default_lru_refresh_time = cfg.lru_refresh_time;
            cache.set_config(new_cfg);

            // S3: 恢复 CountMinSketch 状态
            {
                auto cms_words = r.read<uint32_t>();
                if (cms_words > 1'000'000) {  // ~4MB upper bound for CMS data
                    throw std::runtime_error("deserialization: cms_words exceeds reasonable limit");
                }
                std::vector<uint32_t> cms_buf(cms_words);
                for (uint32_t i = 0; i < cms_words; ++i) {
                    cms_buf[i] = r.read<uint32_t>();
                }
                auto it = cms_buf.begin();
                cache.sketch_mut().load_state(it);
            }

            std::vector<ser_item> items;
            items.reserve(item_count);
            for (uint32_t i = 0; i < item_count; ++i) {
                ser_item item;
                item.read(r);
                items.push_back(std::move(item));
            }
            cache.rebuild_from_serialized(items.begin(), items.end());
        });
}

// ============================================================================
// Serialization — W-TinyLFU
// ============================================================================

template <typename Key, typename Value, typename Hash, typename KeyEqual>
std::vector<uint8_t> serialize(const mm_wtiny_lfu<Key, Value, Hash, KeyEqual>& cache) {
    return detail::serialize_impl(cache, mm_type_id::w_tiny_lfu,
        static_cast<serialization_feature>(0),
        [&](serialized_mm_config& cfg) {
            const auto& c = cache.config();
            cfg.lru_refresh_time = cache.refresh_time();
            cfg.lru_refresh_ratio = c.lru_refresh_ratio;
            cfg.cms_error_rate = c.cms_error_rate;
            cfg.try_lock_update = c.try_lock_update;
        },
        [&](detail::binary_writer& w) {
            // S3: 序列化 CountMinSketch 状态
            uint32_t cms_words = static_cast<uint32_t>(cache.sketch().serialized_state_words());
            w.write(cms_words);
            std::vector<uint32_t> cms_buf(cms_words);
            cache.sketch().save_state(cms_buf.begin());
            for (auto cw : cms_buf) w.write(cw);
        });
}

template <typename Key, typename Value, typename Hash, typename KeyEqual>
void deserialize(mm_wtiny_lfu<Key, Value, Hash, KeyEqual>& cache,
                 std::span<const uint8_t> data) {
    detail::deserialize_impl(cache, mm_type_id::w_tiny_lfu, data,
        [&](detail::binary_reader& r, uint32_t item_count,
            serialized_mm_config& cfg) {
            using ser_item = serialized_item<Key, Value>;

            // Restore config including cms_error_rate
            auto new_cfg = cache.config();
            new_cfg.lru_refresh_ratio = cfg.lru_refresh_ratio;
            new_cfg.cms_error_rate = cfg.cms_error_rate;
            new_cfg.try_lock_update = cfg.try_lock_update;
            new_cfg.default_lru_refresh_time = cfg.lru_refresh_time;
            cache.set_config(new_cfg);

            // S3: 恢复 CountMinSketch 状态
            {
                auto cms_words = r.read<uint32_t>();
                if (cms_words > 1'000'000) {  // ~4MB upper bound for CMS data
                    throw std::runtime_error("deserialization: cms_words exceeds reasonable limit");
                }
                std::vector<uint32_t> cms_buf(cms_words);
                for (uint32_t i = 0; i < cms_words; ++i) {
                    cms_buf[i] = r.read<uint32_t>();
                }
                auto it = cms_buf.begin();
                cache.sketch_mut().load_state(it);
            }

            std::vector<ser_item> items;
            items.reserve(item_count);
            for (uint32_t i = 0; i < item_count; ++i) {
                ser_item item;
                item.read(r);
                items.push_back(std::move(item));
            }
            cache.rebuild_from_serialized(items.begin(), items.end());
        });
}

// ============================================================================
// Serialization — Sharded LRU
// ============================================================================

/// Serialize a sharded LRU cache. Format (v5):
///   Header:
///     - magic        (4 bytes)  kSerializationMagic
///     - version      (4 bytes)  kSerializationVersion
///     - num_shards   (4 bytes)
///     - header_size  (4 bytes)  total header bytes
///     - flags        (4 bytes)
///     - feature_flags(8 bytes)  serialization_feature bitmask
///     - checksum     (4 bytes)  CRC32 of payload (everything after checksum)
///   For each shard:
///     - shard_data_length (4 bytes)
///     - shard_data (variable, the serialized mm_lru for that shard)
template <typename Key, typename Value, typename Hash, typename KeyEqual>
std::vector<uint8_t> serialize(const sharded_mm_lru<Key, Value, Hash, KeyEqual>& cache) {
    // Serialize each shard individually, then concatenate
    std::vector<std::vector<uint8_t>> shard_data;
    shard_data.reserve(cache.num_shards());

    std::size_t total_size = kV5HeaderSize;
    for (std::size_t i = 0; i < cache.num_shards(); ++i) {
        auto data = serialize(static_cast<const mm_lru<Key, Value, Hash, KeyEqual>&>(cache.shard(i)));
        total_size += 4; // shard_data_length
        total_size += data.size();
        shard_data.push_back(std::move(data));
    }

    std::vector<uint8_t> result;
    result.reserve(total_size);

    detail::binary_writer w;
    w.write(kSerializationMagic);
    w.write(kSerializationVersion);
    w.write(static_cast<uint32_t>(cache.num_shards()));
    w.write(kV5HeaderSize);
    w.write(static_cast<uint32_t>(0)); // flags
    w.write(static_cast<uint64_t>(0)); // feature_flags
    auto checksum_offset = w.size();
    w.write(static_cast<uint32_t>(0)); // checksum placeholder
    for (auto& data : shard_data) {
        w.write(static_cast<uint32_t>(data.size()));
        w.write_bytes(data.data(), data.size());
    }

    uint32_t checksum = detail::crc32(
        w.data().data() + checksum_offset + sizeof(uint32_t),
        w.size() - checksum_offset - sizeof(uint32_t));
    w.patch_at(checksum_offset, checksum);

    return w.release();
}

/// Deserialize into a sharded LRU cache (overwrites existing contents).
/// Only v5 format is supported. Validates the checksum.
template <typename Key, typename Value, typename Hash, typename KeyEqual>
void deserialize(sharded_mm_lru<Key, Value, Hash, KeyEqual>& cache,
                 std::span<const uint8_t> data) {
    detail::binary_reader r(data.data(), data.size());

    auto magic = r.read<uint32_t>();
    if (magic != kSerializationMagic) {
        throw std::runtime_error("sharded_mm_lru deserialize: invalid magic number");
    }
    auto version = r.read<uint32_t>();
    if (version != kSerializationVersion) {
        throw std::runtime_error("sharded_mm_lru deserialize: unsupported version " +
            std::to_string(version) +
            " (only version " + std::to_string(kSerializationVersion) + " is supported)");
    }

    auto num_shards = r.read<uint32_t>();
    auto header_size = r.read<uint32_t>();
    if (header_size < kV5HeaderSize || header_size > data.size()) {
        throw std::runtime_error("sharded_mm_lru deserialize: invalid header_size");
    }
    [[maybe_unused]] auto flags = r.read<uint32_t>();

    // Read feature_flags (v5 header)
    [[maybe_unused]] auto feature_flags = r.read<uint64_t>();

    auto stored_checksum = r.read<uint32_t>();
    std::size_t payload_offset = header_size;
    uint32_t computed_checksum = detail::crc32(
        data.data() + payload_offset,
        data.size() - payload_offset);
    if (computed_checksum != stored_checksum) {
        throw std::runtime_error("sharded_mm_lru deserialize: header checksum mismatch");
    }

    if (num_shards != cache.num_shards()) {
        throw std::runtime_error("deserialization: num_shards mismatch");
    }

    for (uint32_t i = 0; i < num_shards; ++i) {
        auto shard_span = r.read_span();
        deserialize(cache.shard(i), shard_span);
    }
}

// ============================================================================
// Convenience — file I/O helpers
// ============================================================================

#ifdef LRU_SERIALIZATION_FILE_IO

#include <fstream>

/// Save serialized data to a file.
inline void save_to_file(const std::vector<uint8_t>& data, const std::string& path) {
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("save_to_file: cannot open " + path);
    }
    file.write(reinterpret_cast<const char*>(data.data()),
               static_cast<std::streamsize>(data.size()));
}

/// Load serialized data from a file.
inline std::vector<uint8_t> load_from_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("load_from_file: cannot open " + path);
    }
    auto size = file.tellg();
    if (size < 0) {
        throw std::runtime_error("load_from_file: failed to determine file size");
    }
    file.seekg(0);
    std::vector<uint8_t> data(static_cast<std::size_t>(size));
    file.read(reinterpret_cast<char*>(data.data()), size);
    return data;
}

#endif // LRU_SERIALIZATION_FILE_IO

} // namespace lru

#endif // LRU_SERIALIZATION_HPP
