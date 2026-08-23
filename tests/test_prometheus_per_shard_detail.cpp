// P2-4: prometheus per-shard aggregate + detail toggle tests.
//
// Validates that:
//   - By default, sharded caches emit only aggregate load-factor gauges
//     (worst / avg / p95) and skip the per-shard labelled series.
//   - After `set_prometheus_per_shard_detail(true)`, the per-shard
//     labelled series is emitted.
//   - The flag is mutable at runtime and takes effect on the next
//     `prometheus_text()` call (also when metrics cache is enabled).
//   - Non-sharded caches are unaffected by the flag (only emit the
//     worst-shard gauge).

#include <gtest/gtest.h>
#include "../lru.hpp"

#include <string>

using namespace lru;

// ============================================================================
// Helper: count occurrences of a substring in a string
// ============================================================================
static std::size_t count_occurrences(const std::string& haystack,
                                     const std::string& needle) {
    std::size_t count = 0;
    std::size_t pos = 0;
    while ((pos = haystack.find(needle, pos)) != std::string::npos) {
        ++count;
        pos += needle.size();
    }
    return count;
}

// ============================================================================
// Default (detail off): only aggregate gauges emitted
// ============================================================================
TEST(PrometheusPerShardDetailTest, DefaultEmitsAggregateOnly) {
    striped_cache<int, std::string> c(1024, 8);
    for (int i = 0; i < 64; ++i) c.set(i, "v" + std::to_string(i));

    const std::string text = c.prometheus_text();

    // Aggregate gauges are always present for sharded caches.
    EXPECT_NE(text.find("lru_hash_load_factor_worst"), std::string::npos);
    EXPECT_NE(text.find("lru_hash_load_factor_avg"), std::string::npos);
    EXPECT_NE(text.find("lru_hash_load_factor_p95"), std::string::npos);

    // Per-shard labelled series is suppressed by default.
    EXPECT_EQ(text.find("lru_hash_load_factor_per_shard"), std::string::npos);

    // Sanity: default flag reports false.
    EXPECT_FALSE(c.prometheus_per_shard_detail());
}

// ============================================================================
// Detail on: per-shard labelled series emitted
// ============================================================================
TEST(PrometheusPerShardDetailTest, DetailOnEmitsLabelledSeries) {
    striped_cache<int, std::string> c(1024, 8);
    for (int i = 0; i < 64; ++i) c.set(i, "v" + std::to_string(i));

    c.set_prometheus_per_shard_detail(true);
    EXPECT_TRUE(c.prometheus_per_shard_detail());

    const std::string text = c.prometheus_text();

    // Aggregate gauges are still present.
    EXPECT_NE(text.find("lru_hash_load_factor_worst"), std::string::npos);
    EXPECT_NE(text.find("lru_hash_load_factor_avg"), std::string::npos);
    EXPECT_NE(text.find("lru_hash_load_factor_p95"), std::string::npos);

    // Per-shard labelled series is now present — one line per shard (8).
    // Count by the metric name; each line begins with the name (no HELP/TYPE
    // header is emitted for the labelled series).
    const std::size_t per_shard_lines =
        count_occurrences(text, "lru_hash_load_factor_per_shard{shard=\"");
    EXPECT_EQ(per_shard_lines, 8u);
}

// ============================================================================
// Toggle at runtime: flag flips output on next call
// ============================================================================
TEST(PrometheusPerShardDetailTest, ToggleAtRuntime) {
    striped_cache<int, std::string> c(256, 4);
    for (int i = 0; i < 16; ++i) c.set(i, "v" + std::to_string(i));

    // Initially off.
    std::string text_off = c.prometheus_text();
    EXPECT_EQ(text_off.find("lru_hash_load_factor_per_shard"), std::string::npos);

    // Flip on.
    c.set_prometheus_per_shard_detail(true);
    std::string text_on = c.prometheus_text();
    EXPECT_NE(text_on.find("lru_hash_load_factor_per_shard"), std::string::npos);

    // Flip off again.
    c.set_prometheus_per_shard_detail(false);
    std::string text_off_again = c.prometheus_text();
    EXPECT_EQ(text_off_again.find("lru_hash_load_factor_per_shard"),
              std::string::npos);
}

// ============================================================================
// Detail flag invalidates the metrics cache (when enabled)
// ============================================================================
TEST(PrometheusPerShardDetailTest, InvalidatesMetricsCache) {
    striped_cache<int, std::string> c(256, 4);
    for (int i = 0; i < 16; ++i) c.set(i, "v" + std::to_string(i));

    // Enable metrics caching so we can verify the cache is invalidated.
    c.set_metrics_cache_enabled(true);

    // Prime the cache with detail off.
    std::string text_off = c.prometheus_text();
    EXPECT_EQ(text_off.find("lru_hash_load_factor_per_shard"), std::string::npos);

    // Flip detail on — this must invalidate the cache so the next call
    // rebuilds with the labelled series included.
    c.set_prometheus_per_shard_detail(true);
    std::string text_on = c.prometheus_text();
    EXPECT_NE(text_on.find("lru_hash_load_factor_per_shard"), std::string::npos);

    c.set_metrics_cache_enabled(false);
}

// ============================================================================
// Non-sharded cache: detail flag has no effect
// ============================================================================
TEST(PrometheusPerShardDetailTest, NonShardedUnaffectedByFlag) {
    safe_cache<int, std::string> c(64);
    for (int i = 0; i < 16; ++i) c.set(i, "v" + std::to_string(i));

    // Non-sharded caches only emit the single `lru_hash_load_factor` gauge.
    // The aggregate gauges (worst/avg/p95) and per-shard labelled series
    // are skipped because there is only one shard.
    c.set_prometheus_per_shard_detail(false);
    std::string text_off = c.prometheus_text();
    EXPECT_NE(text_off.find("lru_hash_load_factor "), std::string::npos);
    EXPECT_EQ(text_off.find("lru_hash_load_factor_per_shard"), std::string::npos);
    EXPECT_EQ(text_off.find("lru_hash_load_factor_worst"), std::string::npos);

    // Flipping the flag does not change the output for non-sharded caches.
    c.set_prometheus_per_shard_detail(true);
    std::string text_on = c.prometheus_text();
    EXPECT_EQ(text_on.find("lru_hash_load_factor_per_shard"), std::string::npos);
    EXPECT_EQ(text_on.find("lru_hash_load_factor_worst"), std::string::npos);
}

// ============================================================================
// Aggregate gauges are well-formed (numeric values, not empty)
// ============================================================================
TEST(PrometheusPerShardDetailTest, AggregateGaugesAreNumeric) {
    striped_cache<int, std::string> c(1024, 8);
    for (int i = 0; i < 64; ++i) c.set(i, "v" + std::to_string(i));

    const std::string text = c.prometheus_text();

    // Each gauge line must end with a numeric value, not be empty.
    // Find the metric VALUE line (skip HELP/TYPE lines that start with '#').
    auto find_value = [&](const std::string& name) -> std::string {
        std::size_t pos = 0;
        while ((pos = text.find(name, pos)) != std::string::npos) {
            // Skip lines that start with '#' (HELP/TYPE annotations).
            std::size_t line_start = text.rfind('\n', pos);
            line_start = (line_start == std::string::npos) ? 0 : line_start + 1;
            if (line_start < text.size() && text[line_start] == '#') {
                pos += name.size();
                continue;
            }
            // This is the value line. Skip the metric name + space, then
            // read until end-of-line.
            pos += name.size();
            if (pos < text.size() && text[pos] == ' ') ++pos;
            auto end = text.find('\n', pos);
            return text.substr(pos, end - pos);
        }
        return "";
    };

    std::string worst = find_value("lru_hash_load_factor_worst");
    std::string avg = find_value("lru_hash_load_factor_avg");
    std::string p95 = find_value("lru_hash_load_factor_p95");

    ASSERT_FALSE(worst.empty());
    ASSERT_FALSE(avg.empty());
    ASSERT_FALSE(p95.empty());

    // Values must be parseable as float.
    EXPECT_NO_THROW({ (void)std::stof(worst); });
    EXPECT_NO_THROW({ (void)std::stof(avg); });
    EXPECT_NO_THROW({ (void)std::stof(p95); });

    // Worst >= p95 >= avg is not strictly required by definition, but
    // worst >= avg should hold (max of a set is >= mean of the set).
    EXPECT_GE(std::stof(worst), std::stof(avg));
}
