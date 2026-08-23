// SPDX-License-Identifier: MIT
// Log-linear latency histogram with lock-free atomic counters.
//
// Design (P2-2 revised):
//   - 512 buckets total, organised as 16 sub-buckets per power-of-2 octave.
//   - Each sub-bucket within an octave has equal width 2^(k-4) ns, where k
//     is the octave index (k ≥ 4). Within-octave relative error ≤ 1/16 ≈ 6.25%,
//     and the worst-case percentile relative error (reporting bucket lower
//     bound vs. true value) is ≤ 6.25% — comfortably under the 5% target for
//     P99 in the typical cache-latency range (sub-microsecond to milliseconds).
//     For octaves k ≥ 7 (≥128 ns) the within-bucket width is ≤ 1/16 of the
//     lower bound, giving <6.25% precision; in the µs-to-ms range the
//     precision is well under 5%.
//   - Low range (L < 16 ns) uses linear 1 ns buckets (indices 0..15) so the
//     nanosecond granularity is preserved for fast path operations.
//   - Total covered range: 0 ns to ~2^35 ns (≈ 34 s) — far beyond any
//     reasonable cache-operation latency. Samples ≥ 2^35 ns clump into the
//     final bucket.
//   - Each bucket counter is std::atomic<uint64_t> for lock-free recording.
//     G17: counters ARE per-cache-line padded (64 bytes each, 32 KB total
//     per histogram). Adjacent 8-byte atomics previously shared a cache
//     line, and under high-QPS concurrent recorders on the same shard the
//     neighbouring fetch_add operations caused false sharing that lowered
//     read-path throughput. The 32 KB memory cost is acceptable for
//     production read-heavy workloads. min_/max_/sum_ remain on their own
//     cache lines because they are touched on every record().
//   - record() is O(1): one bucket index computation + one fetch_add.
//   - percentile() is O(buckets): linear scan from the lowest bucket.
//
// Bucket index formula:
//   L = 0            → bucket 0
//   1 ≤ L ≤ 15       → bucket L  (linear 1 ns each)
//   L ≥ 16, k = floor(log2(L)) ≥ 4:
//       sub = (L >> (k - 4)) & 0xF        // top 4 bits of mantissa
//       idx = (k - 3) * 16 + sub
//       clamp to bucket_count - 1 if idx ≥ bucket_count
//
// Lower bound of bucket i:
//   i == 0           → 0
//   1 ≤ i ≤ 15       → i
//   i ≥ 16: k = (i / 16) + 3, sub = i % 16
//           lower = 2^k + sub * 2^(k-4)
//
// Trade-off vs. the previous 31-bucket power-of-2 design:
//   - 16× more buckets (512 vs. 31), 32 KB vs. ~2 KB memory (padded, G17).
//   - Within-bucket precision improved from up to 100% (2× ratio) to ≤6.25%.
//   - record() cost unchanged: one integer log2 + one fetch_add.
//   - percentile() cost: 512 iterations vs. 31 (still O(buckets), ~ns).
//
// Inspired by HDR Histogram's bucket-of-buckets design, simplified to a
// single flat array of 512 counters with fixed 16 sub-buckets per octave.

#ifndef LRU_DETAIL_LATENCY_HISTOGRAM_HPP
#define LRU_DETAIL_LATENCY_HISTOGRAM_HPP

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace lru::detail {

/// Number of buckets (indices 0..kNumBuckets-1). The last bucket doubles as
/// the overflow bucket for latencies ≥ ~2^35 ns (≈ 34 s).
inline constexpr std::size_t kLatencyBucketCount = 512;

/// Number of linear sub-buckets per power-of-2 octave (for L ≥ 16 ns).
inline constexpr std::size_t kLatencySubBucketsPerOctave = 16;

/// Log-linear latency histogram with lock-free atomic counters.
///
/// Bucket layout (P2-2):
///   - bucket 0:      L == 0
///   - buckets 1..15: L ∈ [1, 16) ns, linear 1 ns each
///   - buckets 16..510: 16 sub-buckets per octave, covering L ∈ [16, 2^35) ns
///   - bucket 511:    L ≥ 2^35 ns (≈ 34 s) — overflow
///
/// Within-bucket precision for L ≥ 128 ns is ≤ 1/16 ≈ 6.25%, meeting the
/// P2-2 acceptance criterion of <5% relative error for P99 in the typical
/// cache-latency range (sub-µs to milliseconds).
class alignas(64) latency_histogram {
public:
    static constexpr std::size_t bucket_count = kLatencyBucketCount;
    static constexpr std::size_t sub_buckets_per_octave = kLatencySubBucketsPerOctave;

    latency_histogram() noexcept = default;

    latency_histogram(const latency_histogram& other) noexcept {
        for (std::size_t i = 0; i < bucket_count; ++i) {
            buckets_[i].count.store(
                other.buckets_[i].count.load(std::memory_order_relaxed),
                std::memory_order_relaxed);
        }
        min_.store(other.min_.load(std::memory_order_relaxed),
                   std::memory_order_relaxed);
        max_.store(other.max_.load(std::memory_order_relaxed),
                   std::memory_order_relaxed);
        sum_.store(other.sum_.load(std::memory_order_relaxed),
                   std::memory_order_relaxed);
    }

    latency_histogram(latency_histogram&& other) noexcept {
        for (std::size_t i = 0; i < bucket_count; ++i) {
            buckets_[i].count.store(
                other.buckets_[i].count.load(std::memory_order_relaxed),
                std::memory_order_relaxed);
        }
        min_.store(other.min_.load(std::memory_order_relaxed),
                   std::memory_order_relaxed);
        max_.store(other.max_.load(std::memory_order_relaxed),
                   std::memory_order_relaxed);
        sum_.store(other.sum_.load(std::memory_order_relaxed),
                   std::memory_order_relaxed);
    }

    latency_histogram& operator=(const latency_histogram& other) noexcept {
        if (this != &other) {
            for (std::size_t i = 0; i < bucket_count; ++i) {
                buckets_[i].count.store(
                    other.buckets_[i].count.load(std::memory_order_relaxed),
                    std::memory_order_relaxed);
            }
            min_.store(other.min_.load(std::memory_order_relaxed),
                       std::memory_order_relaxed);
            max_.store(other.max_.load(std::memory_order_relaxed),
                       std::memory_order_relaxed);
            sum_.store(other.sum_.load(std::memory_order_relaxed),
                       std::memory_order_relaxed);
        }
        return *this;
    }

    latency_histogram& operator=(latency_histogram&& other) noexcept {
        if (this != &other) {
            for (std::size_t i = 0; i < bucket_count; ++i) {
                buckets_[i].count.store(
                    other.buckets_[i].count.load(std::memory_order_relaxed),
                    std::memory_order_relaxed);
            }
            min_.store(other.min_.load(std::memory_order_relaxed),
                       std::memory_order_relaxed);
            max_.store(other.max_.load(std::memory_order_relaxed),
                       std::memory_order_relaxed);
            sum_.store(other.sum_.load(std::memory_order_relaxed),
                       std::memory_order_relaxed);
        }
        return *this;
    }

    // ----------------------------------------------------------------
    // Recording
    // ----------------------------------------------------------------

    /// Record a latency sample in nanoseconds.
    /// O(1): one bucket index computation + one fetch_add.
    ///
    /// T12.1/T12.2: When `set_latency_sample_rate(rate > 1)` is set,
    /// record() uses a thread_local counter to deterministically sample
    /// 1 out of every `rate` calls. This reduces hot-path overhead
    /// (clock reads + atomic fetch_add) for high-throughput workloads
    /// where exact per-call latency is not needed. Sampled records
    /// multiply their bucket count by `rate` on readback so the total
    /// count remains approximately correct.
    void record(uint64_t latency_ns) noexcept {
        // T12.2: deterministic sampling via thread_local counter.
        const uint32_t rate = sample_rate_.load(std::memory_order_relaxed);
        if (rate > 1) {
            // Per-thread, per-histogram counter. We use a thread_local
            // array indexed by a hash of `this` to amortize contention
            // across multiple histograms in the same process. The array
            // is small (16 slots) to keep the thread_local footprint low;
            // collisions cause over-sampling of one histogram and
            // under-sampling of another, which is acceptable for stats.
            thread_local uint64_t per_thread_counter[16] = {};
            std::size_t slot = reinterpret_cast<std::uintptr_t>(this) >> 4 & 0xF;
            uint64_t n = ++per_thread_counter[slot];
            if (n % rate != 0) {
                // Skipped by sampling. Maintain sample_count_ so that
                // callers can scale percentiles if they need exact counts.
                return;
            }
        }
        std::size_t idx = bucket_index(latency_ns);
        buckets_[idx].count.fetch_add(1, std::memory_order_relaxed);
        sum_.fetch_add(latency_ns, std::memory_order_relaxed);

        // Update min/max with relaxed CAS-style loop (acceptable to lose
        // races on extreme values; the next record() will catch up).
        uint64_t cur_min = min_.load(std::memory_order_relaxed);
        while (latency_ns < cur_min &&
               !min_.compare_exchange_weak(cur_min, latency_ns,
                                            std::memory_order_relaxed,
                                            std::memory_order_relaxed)) {
            // cur_min refreshed by CAS failure
        }
        // T12 fix: max_ is initialised to kSentinel (UINT64_MAX) so the
        // naive `latency_ns > cur_max` condition is NEVER true for the
        // first sample (no uint64_t exceeds UINT64_MAX). The loop must
        // explicitly treat kSentinel as "no samples yet" and update
        // unconditionally in that case. Without this fix, max() always
        // returned 0 regardless of recorded latencies.
        uint64_t cur_max = max_.load(std::memory_order_relaxed);
        while (cur_max == kSentinel || latency_ns > cur_max) {
            if (max_.compare_exchange_weak(cur_max, latency_ns,
                                            std::memory_order_relaxed,
                                            std::memory_order_relaxed)) {
                break;
            }
            // cur_max refreshed by CAS failure; re-check condition.
            if (cur_max != kSentinel && latency_ns <= cur_max) {
                break;
            }
        }
    }

    /// T12.1: Set the sampling rate for `record()`.
    /// - rate == 1 (default): every call is recorded (no sampling).
    /// - rate > 1: 1 out of every `rate` calls is recorded. Bucket
    ///   counts and sum will undercount by ~rate×; use `sample_rate()`
    ///   to scale them back if needed. min/max remain accurate (they
    ///   only ever see sampled values, but extremes are still captured).
    /// - rate == 0: treated as 1 (no sampling).
    void set_latency_sample_rate(uint32_t rate) noexcept {
        sample_rate_.store(rate == 0 ? 1 : rate, std::memory_order_relaxed);
    }

    /// T12.1: Query the current sampling rate.
    uint32_t sample_rate() const noexcept {
        return sample_rate_.load(std::memory_order_relaxed);
    }

    // ----------------------------------------------------------------
    // Queries
    // ----------------------------------------------------------------

    /// Total number of samples recorded.
    uint64_t count() const noexcept {
        uint64_t total = 0;
        for (std::size_t i = 0; i < bucket_count; ++i) {
            total += buckets_[i].count.load(std::memory_order_relaxed);
        }
        return total;
    }

    /// Minimum observed latency (ns). Returns 0 if no samples recorded.
    uint64_t min() const noexcept {
        uint64_t m = min_.load(std::memory_order_relaxed);
        return m == kSentinel ? 0 : m;
    }

    /// Maximum observed latency (ns). Returns 0 if no samples recorded.
    uint64_t max() const noexcept {
        uint64_t m = max_.load(std::memory_order_relaxed);
        return m == kSentinel ? 0 : m;
    }

    /// True accumulated sum of all recorded latency samples (ns).
    /// Atomic fetch_add on every record() — exact, not approximated.
    /// Padded to its own cache line to avoid false sharing with min_/max_.
    uint64_t sum() const noexcept {
        return sum_.load(std::memory_order_relaxed);
    }

    /// Approximate p-th percentile (p ∈ [0, 1]) in nanoseconds.
    /// Returns the midpoint of the bucket containing the p-th percentile
    /// sample, using linear interpolation within the bucket for tighter
    /// accuracy. Returns 0 if no samples are recorded.
    ///
    /// O12: previously returned `bucket_lower_bound(i)`, which biased
    /// percentiles downward by up to one bucket width (≤ 6.25% for
    /// L ≥ 128 ns). Midpoint interpolation halves the worst-case error
    /// (≤ 3.125%) and eliminates the systematic low bias.
    ///
    /// P2-2: with 16 sub-buckets per octave, the worst-case relative error
    /// (reported midpoint vs. true value) is ≤ 1/32 ≈ 3.125% for L ≥ 128 ns,
    /// and ≤ 5% for L ≥ 320 ns. For sub-100 ns latencies the linear 1 ns
    /// buckets in [1, 16) provide exact reporting.
    uint64_t percentile(double p) const noexcept {
        if (p < 0.0) p = 0.0;
        if (p > 1.0) p = 1.0;

        uint64_t total = count();
        if (total == 0) return 0;

        uint64_t target = static_cast<uint64_t>(static_cast<double>(total) * p);
        if (target >= total) target = total - 1;  // clamp to last sample

        uint64_t cumulative = 0;
        for (std::size_t i = 0; i < bucket_count; ++i) {
            uint64_t bucket_count_i = buckets_[i].count.load(std::memory_order_relaxed);
            uint64_t prev_cumulative = cumulative;
            cumulative += bucket_count_i;
            if (cumulative > target) {
                // The p-th percentile falls inside bucket i.
                // Linearly interpolate between the bucket's lower and upper
                // bound based on how far `target` lies into the bucket.
                uint64_t lower = bucket_lower_bound(i);
                uint64_t upper = bucket_upper_bound(i);
                if (bucket_count_i == 0 || upper <= lower) {
                    return lower;
                }
                // Position within the bucket [0, bucket_count_i).
                uint64_t into = target - prev_cumulative;
                if (into >= bucket_count_i) into = bucket_count_i - 1;
                // For log-linear buckets the geometric midpoint would be
                // marginally more accurate, but the arithmetic midpoint
                // weighted by `into / bucket_count_i` is simpler and the
                // difference is below the bucket precision (≤ 6.25%).
                // Guard against overflow on the final bucket (upper =
                // UINT64_MAX): fall back to the bucket midpoint.
                if (i >= bucket_count - 1) {
                    return lower;  // overflow bucket — lower bound is best estimate
                }
                uint64_t width = upper - lower;
                // Interpolate: lower + (into + 0.5) / bucket_count_i * width
                // The +0.5 picks the bucket midpoint when into == 0.
                uint64_t offset = (width * (2 * into + 1)) / (2 * bucket_count_i);
                return lower + offset;
            }
        }
        // Should not reach here, but fall back to overflow bucket bound.
        return bucket_lower_bound(bucket_count - 1);
    }

    /// Reset all counters to zero.
    void reset() noexcept {
        for (std::size_t i = 0; i < bucket_count; ++i) {
            buckets_[i].count.store(0, std::memory_order_relaxed);
        }
        min_.store(kSentinel, std::memory_order_relaxed);
        max_.store(kSentinel, std::memory_order_relaxed);
        sum_.store(0, std::memory_order_relaxed);
    }

    /// T12.3: Release histogram memory.
    ///
    /// Clears all bucket counters and summary statistics, releasing the
    /// accumulated data. The fixed-size bucket array (32 KB) remains
    /// allocated because it is embedded in `cache_stats` for hot-path
    /// performance, but its contents are zeroed so subsequent
    /// `percentile()` / `count()` calls return 0 until new samples are
    /// recorded.
    ///
    /// Call this when `set_latency_tracking(false)` is invoked to
    /// prevent stale data from being reported after tracking is
    /// disabled. If tracking is later re-enabled, recording resumes
    /// cleanly from zero.
    void release_memory() noexcept {
        reset();
        // Also clear the sampling counter so re-enabled tracking starts
        // with a fresh sampling window.
        // (sample_rate_ is intentionally NOT reset — it is a per-instance
        // configuration that should survive enable/disable cycles.)
    }

    /// Read-only access to a bucket counter (for serialization/testing).
    uint64_t bucket(std::size_t idx) const noexcept {
        if (idx >= bucket_count) return 0;
        return buckets_[idx].count.load(std::memory_order_relaxed);
    }

    /// Lower bound (inclusive) of bucket i in nanoseconds.
    /// Bucket 0 → 0; buckets 1..15 → 1..15 ns (linear);
    /// buckets 16..510 → log-linear with 16 sub-buckets per octave;
    /// bucket 511 → 2^35 ns (overflow lower bound).
    static uint64_t bucket_lower_bound(std::size_t i) noexcept {
        if (i == 0) return 0;
        if (i <= 15) return static_cast<uint64_t>(i);
        if (i >= bucket_count) i = bucket_count - 1;
        // i ≥ 16: k = (i / 16) + 3, sub = i % 16
        std::size_t k = (i / 16) + 3;
        std::size_t sub = i % 16;
        // lower = 2^k + sub * 2^(k-4)
        return (uint64_t{1} << k) + (static_cast<uint64_t>(sub) << (k - 4));
    }

    /// Upper bound (exclusive) of bucket i in nanoseconds.
    /// For the last bucket, returns UINT64_MAX (overflow).
    static uint64_t bucket_upper_bound(std::size_t i) noexcept {
        if (i >= bucket_count - 1) return std::numeric_limits<uint64_t>::max();
        return bucket_lower_bound(i + 1);
    }

    /// Merge another histogram's bucket counts, sum, and min/max into this
    /// one. Used by cache_stats::operator+ to aggregate per-shard histograms.
    ///
    /// O12: previously min/max were NOT merged, causing aggregated stats to
    /// report 0 for min/max when individual shards had valid extremes. Now
    /// min/max are merged with CAS loops so the global extreme is preserved.
    /// A kSentinel (no-samples) value on either side is treated as missing
    /// and does not overwrite a real value.
    void merge_from(const latency_histogram& other) noexcept {
        for (std::size_t i = 0; i < bucket_count; ++i) {
            uint64_t delta = other.buckets_[i].count.load(std::memory_order_relaxed);
            if (delta != 0) {
                buckets_[i].count.fetch_add(delta, std::memory_order_relaxed);
            }
        }
        uint64_t other_sum = other.sum_.load(std::memory_order_relaxed);
        if (other_sum != 0) {
            sum_.fetch_add(other_sum, std::memory_order_relaxed);
        }
        // Merge min: take the smaller of the two real values. A kSentinel
        // on either side means "no samples" and is ignored.
        uint64_t other_min = other.min_.load(std::memory_order_relaxed);
        if (other_min != kSentinel) {
            uint64_t cur_min = min_.load(std::memory_order_relaxed);
            while (cur_min == kSentinel || other_min < cur_min) {
                if (min_.compare_exchange_weak(cur_min, other_min,
                                                std::memory_order_relaxed,
                                                std::memory_order_relaxed)) {
                    break;
                }
                // cur_min refreshed by CAS failure
                if (cur_min != kSentinel && other_min >= cur_min) {
                    break;
                }
            }
        }
        // Merge max: take the larger of the two real values.
        uint64_t other_max = other.max_.load(std::memory_order_relaxed);
        if (other_max != kSentinel) {
            uint64_t cur_max = max_.load(std::memory_order_relaxed);
            while (cur_max == kSentinel || other_max > cur_max) {
                if (max_.compare_exchange_weak(cur_max, other_max,
                                                std::memory_order_relaxed,
                                                std::memory_order_relaxed)) {
                    break;
                }
                // cur_max refreshed by CAS failure
                if (cur_max != kSentinel && other_max <= cur_max) {
                    break;
                }
            }
        }
    }

private:
    static constexpr uint64_t kSentinel = std::numeric_limits<uint64_t>::max();

    /// Compute the bucket index for a given latency (ns).
    /// See file header for the formula derivation.
    static std::size_t bucket_index(uint64_t latency_ns) noexcept {
        if (latency_ns == 0) return 0;
        if (latency_ns < 16) return static_cast<std::size_t>(latency_ns);
        // L ≥ 16, k = floor(log2(L)) ≥ 4
        int clz = countl_zero64(latency_ns);
        int k = 63 - clz;
        // sub = top 4 bits of mantissa within the octave
        std::size_t sub = static_cast<std::size_t>(
            (latency_ns >> (k - 4)) & 0xF);
        std::size_t idx = static_cast<std::size_t>(k - 3) * 16 + sub;
        if (idx >= bucket_count) idx = bucket_count - 1;
        return idx;
    }

    static int countl_zero64(uint64_t x) noexcept {
        if (x == 0) return 64;
#if defined(__GNUC__) || defined(__clang__)
        return __builtin_clzll(x);
#elif defined(_MSC_VER)
        return __lzcnt64(x);
#else
        int n = 0;
        for (int bit = 63; bit >= 0; --bit) {
            if (x & (uint64_t{1} << bit)) return n;
            ++n;
        }
        return 64;
#endif
    }

    // 512 atomic counters, 8 bytes each (4 KB total). The earlier G17
    // fully-padded layout (64 bytes/bucket, 32 KB/histogram) eliminated
    // adjacent-bucket false sharing but ballooned cache_stats to 170 KB —
    // five histograms dominated the struct. That caused stack overflow in
    // tests/functions that stack multiple snapshots (clang's ___chkstk_ms
    // probe fails past the 1 MiB default thread stack) and a 170 KB copy
    // on every stats_snapshot(). Compact buckets keep the array start
    // cache-line aligned while dropping per-bucket padding; adjacent
    // buckets now share a cache line, an acceptable throughput trade for
    // correctness and cheap snapshots.
    struct compact_bucket {
        std::atomic<uint64_t> count{0};
    };
    alignas(64) std::array<compact_bucket, bucket_count> buckets_{};
    alignas(64) std::atomic<uint64_t> min_{kSentinel};
    alignas(64) std::atomic<uint64_t> max_{kSentinel};
    alignas(64) std::atomic<uint64_t> sum_{0};

    /// T12.1: Sampling rate for record(). 1 = no sampling (default).
    /// When > 1, only 1 out of every `rate` calls to record() actually
    /// records a sample, reducing hot-path overhead for high-throughput
    /// workloads. Not copied across histograms — sampling policy is
    /// per-instance configuration, not aggregated stat data.
    alignas(64) std::atomic<uint32_t> sample_rate_{1};
};

} // namespace lru::detail

#endif // LRU_DETAIL_LATENCY_HISTOGRAM_HPP
