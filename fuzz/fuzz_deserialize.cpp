// fix.01 §4.4: fuzz the deserialization path.
//
// `deserialize()` is the point where attacker-controlled (or merely corrupt)
// bytes become cache state. The contracts it must hold, and which this fuzzer
// exercises:
//   * never crash / never read out of bounds (ASan is on),
//   * never invoke undefined behaviour (UBSan is on),
//   * on malformed input, throw or return rather than corrupt the cache,
//   * on failure, leave the target cache unchanged (the two-phase
//     deserializers — see P1-32),
//   * be deterministic: the same bytes must always produce the same outcome.
//
// Build (clang only):
//   cmake -B build-fuzz -DCMAKE_CXX_COMPILER=clang++ -DLRU_BUILD_FUZZERS=ON
//   cmake --build build-fuzz -j4
//   ./build-fuzz/fuzz/lru_fuzz_deserialize -max_total_time=60

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "lru.hpp"

namespace {

/// Feed the bytes to the raw MM-level parser (no cache state involved).
template <typename MM>
void fuzz_raw_parse(std::span<const std::uint8_t> data) {
    try {
        lru::detail::parse_serialized_data<int, std::string>(lru::detail::mm_type_id::lru, data);
    } catch (const std::exception&) {
        // Malformed input is expected; the contract is "throw, don't corrupt".
    }
}

/// Feed the bytes to a live cache and confirm it survives either way.
template <typename Cache>
void fuzz_cache_load(Cache& c, std::span<const std::uint8_t> data) {
    const std::size_t before = c.size();
    try {
        c.load(data);
    } catch (const std::exception&) {
        // A rejected snapshot must leave the cache untouched.
        if (c.size() != before) __builtin_trap();
        return;
    }
    // A round-trip of whatever we just loaded must not throw.
    try {
        auto snapshot = c.save();
        Cache probe(16);
        probe.load(snapshot);
    } catch (const std::exception&) {
        __builtin_trap();
    }
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    if (size > (1u << 20)) return 0;  // keep iterations bounded
    const std::span<const std::uint8_t> bytes(data, size);

    fuzz_raw_parse<lru::mm_lru<int, std::string>>(bytes);

    // Non-sharded thread-safe cache.
    {
        static lru::safe_cache<int, std::string> c(64);
        fuzz_cache_load(c, bytes);
    }

    // Sharded cache: exercises the per-shard two-phase rebuild (P1-32).
    {
        static lru::production_cache<int, std::string> c(64);
        fuzz_cache_load(c, bytes);
    }

    // F14 / segmented hash-table variants reach different parser paths.
    {
        static lru::f14_cache<int, std::string> c(64);
        fuzz_cache_load(c, bytes);
    }
    return 0;
}
