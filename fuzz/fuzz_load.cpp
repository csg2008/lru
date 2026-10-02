// fix.01 §4.4: fuzz the warm-cache / snapshot loaders.
//
// warm_cache_manager::load_with_delta() and load_cache_from_file() parse a
// "<path>.full" snapshot followed by a "<path>.delta" replay. Both are
// untrusted-input boundaries that had no fuzz coverage; the delta replay in
// particular applies an arbitrary sequence of set/remove operations, so the
// invariant under test is simply "applying any byte sequence neither crashes
// nor corrupts the cache".
//
// Unlike fuzz_deserialize, this one needs filesystem input, so it writes the
// fuzz bytes to a temp file and drives the file-based loaders.
//
// Build (clang only):
//   cmake -B build-fuzz -DCMAKE_CXX_COMPILER=clang++ -DLRU_BUILD_FUZZERS=ON
//   cmake --build build-fuzz -j4
//   ./build-fuzz/fuzz/lru_fuzz_load -max_total_time=60

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "lru.hpp"

namespace {

/// Write `bytes` to `path`, best effort.
void write_file(const std::string& path, std::span<const std::uint8_t> bytes) {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return;
    if (!bytes.empty()) std::fwrite(bytes.data(), 1, bytes.size(), f);
    std::fclose(f);
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    if (size > (1u << 20)) return 0;

    // Split the input in two halves: the first becomes the ".full" snapshot,
    // the second the ".delta" replay.
    const std::size_t half = size / 2;
    const std::span<const std::uint8_t> full_bytes(data, half);
    const std::span<const std::uint8_t> delta_bytes(data + half, size - half);

    const std::string base = "lru_fuzz_load_tmp";
    write_file(base + ".full", full_bytes);
    write_file(base + ".delta", delta_bytes);

    // Drive both file-based loaders. The candidate cache type is the plain
    // thread-safe LRU (warm restart is documented against it).
    using cache_t = lru::safe_cache<int, std::string>;
    try {
        lru::warm_cache_manager<cache_t> mgr(64);
        (void)mgr.load_with_delta(base);
    } catch (const std::exception&) {
        // Rejected input is fine; the contract is "no crash, no corruption".
    }
    try {
        (void)lru::load_cache_from_file<cache_t>(base, 64);
    } catch (const std::exception&) {
    }

    std::remove((base + ".full").c_str());
    std::remove((base + ".delta").c_str());
    return 0;
}
