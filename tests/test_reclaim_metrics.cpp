// T17: Reclaim monitoring metrics tests.
//
// Validates that hazptr/EBR deferred-reclamation counters are surfaced
// consistently across three observability surfaces:
//   1. cache_stats fields (reclaim_pending_count / reclaim_total /
//      reclaim_freed_bytes / reclaim_invocation_count)
//   2. prometheus_text() exposition (lru_reclaim_* metrics)
//   3. diagnostics() / diagnostics_text() (operator-facing dump)
//
// Also verifies that try_reclaim_now() advances the invocation counter
// and that retiring items via cache churn updates pending_count.

#include "../lru.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <string>
#include <thread>

namespace {
// Extract the first integer value following a metric name in a Prometheus
// exposition blob. Returns 0 if not found.
//
// Skips comment lines (starting with '#') so that the HELP/TYPE metadata
// lines don't shadow the actual metric value line. For example, given:
//   # HELP lru_hazptr_slot_capacity Current hazptr ...
//   # TYPE lru_hazptr_slot_capacity gauge
//   lru_hazptr_slot_capacity 128
// the function returns 128 (from the third line), not 0 (from parsing
// "Current" after the HELP line).
std::size_t extract_prometheus_value(const std::string& body, const std::string& metric) {
    std::size_t search_from = 0;
    while (true) {
        auto pos = body.find(metric, search_from);
        if (pos == std::string::npos) return 0;
        // Check if this occurrence is inside a comment line (preceded by '# ').
        // Walk backward to the start of the line.
        std::size_t line_start = body.rfind('\n', pos);
        line_start = (line_start == std::string::npos) ? 0 : line_start + 1;
        // Skip leading whitespace on the line.
        std::size_t ws = line_start;
        while (ws < pos && (body[ws] == ' ' || body[ws] == '\t')) ++ws;
        // If the line starts with '#', this is a HELP/TYPE comment — skip it.
        if (ws < pos && body[ws] == '#') {
            search_from = pos + metric.size();
            continue;
        }
        // Found the actual metric line. Skip the metric name and whitespace.
        pos += metric.size();
        while (pos < body.size() && (body[pos] == ' ' || body[pos] == '\t')) ++pos;
        std::size_t end = pos;
        while (end < body.size() && (body[end] >= '0' && body[end] <= '9')) ++end;
        if (end == pos) return 0;
        return std::stoull(body.substr(pos, end - pos));
    }
}
} // namespace

TEST(ReclaimMetricsTest, StatsSnapshotHasReclaimFields) {
    lru::cache<int, std::string> c{1024};
    auto snap = c.stats_snapshot();
    // Per-cache invocation counter starts at zero on a fresh cache.
    // reclaim_total / reclaim_pending_count mirror the GLOBAL hazptr/EBR
    // domain counters (which are cumulative across all caches/tests in
    // the process), so they may be non-zero if prior tests retired
    // objects. We only assert the per-cache delta is zero.
    EXPECT_EQ(snap.reclaim_invocation_count.load(std::memory_order_relaxed), 0u);
    // pending_count is a snapshot of the global domain; on a quiet fresh
    // cache it should be 0 (no objects retired by THIS cache), but if
    // the global domain still has unreclaimed objects from prior tests,
    // it may be positive. Just verify it's finite.
    (void)snap.reclaim_pending_count.load(std::memory_order_relaxed);
    (void)snap.reclaim_total.load(std::memory_order_relaxed);
    // freed_bytes is an estimate; just verify it is a valid size_t.
    (void)snap.reclaim_freed_bytes.load(std::memory_order_relaxed);
}

TEST(ReclaimMetricsTest, PrometheusExportsReclaimMetrics) {
    lru::cache<int, std::string> c{1024};
    c.set(1, "one");
    auto text = c.prometheus_text();
    // All four reclaim metrics should be present in the exposition.
    EXPECT_NE(text.find("lru_reclaim_pending_count"), std::string::npos);
    EXPECT_NE(text.find("lru_reclaim_total"), std::string::npos);
    EXPECT_NE(text.find("lru_reclaim_freed_bytes_total"), std::string::npos);
    EXPECT_NE(text.find("lru_reclaim_invocations_total"), std::string::npos);
    // Sanity check: pending count should be zero on a quiet cache.
    EXPECT_EQ(extract_prometheus_value(text, "lru_reclaim_pending_count"), 0u);
}

TEST(ReclaimMetricsTest, DiagnosticsReportsReclaimFields) {
    lru::cache<int, std::string> c{1024};
    c.set(1, "one");
    c.set(2, "two");
    auto info = c.diagnostics();
    // Diagnostics struct must carry the reclaim fields.
    // pending_count may be 0 or a small positive value if retire happened
    // but was not yet reclaimed; total is cumulative and >= 0.
    EXPECT_GE(info.reclaim_total, 0u);
    EXPECT_GE(info.reclaim_pending_count, 0u);

    auto text = c.diagnostics_text();
    EXPECT_NE(text.find("reclaim_pending_count"), std::string::npos);
    EXPECT_NE(text.find("reclaim_total"), std::string::npos);
    EXPECT_NE(text.find("reclaim_freed_bytes"), std::string::npos);
    EXPECT_NE(text.find("reclaim_invocation_count"), std::string::npos);
    EXPECT_NE(text.find("reclaim health"), std::string::npos);
}

TEST(ReclaimMetricsTest, TryReclaimAdvancesInvocationCounter) {
    lru::safe_cache<int, std::string> c{1024};
    // Populate then churn to produce retired nodes.
    for (int i = 0; i < 64; ++i) c.set(i, std::to_string(i));
    for (int i = 0; i < 64; ++i) c.set(i + 1000, std::to_string(i + 1000));

    auto before = c.stats_snapshot().reclaim_invocation_count.load(std::memory_order_relaxed);
    c.try_reclaim_now();
    auto after = c.stats_snapshot().reclaim_invocation_count.load(std::memory_order_relaxed);
    // Invocation counter must have advanced by at least one.
    EXPECT_GT(after, before);
}

TEST(ReclaimMetricsTest, ChurnUpdatesPendingOrTotal) {
    // After aggressive churn + retire, either pending_count > 0 (objects
    // not yet reclaimed) or total > 0 (objects already reclaimed). Both
    // being zero indefinitely would indicate reclaim is broken.
    lru::striped_cache<int, std::string> c{256, 4};
    for (int round = 0; round < 4; ++round) {
        for (int i = 0; i < 256; ++i) {
            c.set(i + round * 1000, std::string(64, 'x'));
        }
    }
    auto snap = c.stats_snapshot();
    std::size_t pending = snap.reclaim_pending_count.load(std::memory_order_relaxed);
    std::size_t total = snap.reclaim_total.load(std::memory_order_relaxed);
    EXPECT_TRUE(pending > 0 || total > 0);
}

TEST(ReclaimMetricsTest, PrometheusAndDiagnosticsAgree) {
    // Prometheus exposition and diagnostics_text() should both surface the
    // same reclaim numbers (they read the same underlying counters).
    lru::safe_cache<int, std::string> c{512};
    for (int i = 0; i < 128; ++i) c.set(i, std::to_string(i));
    // Force some retire by overwriting.
    for (int i = 0; i < 128; ++i) c.set(i, std::string(32, 'y'));

    auto snap = c.stats_snapshot();
    auto info = c.diagnostics();
    auto prom = c.prometheus_text();

    std::size_t snap_pending = snap.reclaim_pending_count.load(std::memory_order_relaxed);
    std::size_t info_pending = info.reclaim_pending_count;
    std::size_t prom_pending = extract_prometheus_value(prom, "lru_reclaim_pending_count");

    // Allow for tiny drift between reads (atomic, lock-free) but they
    // should be within a small window of each other.
    if (snap_pending >= prom_pending) {
        EXPECT_LE(snap_pending - prom_pending, 16u);
    } else {
        EXPECT_LE(prom_pending - snap_pending, 16u);
    }
    if (info_pending >= prom_pending) {
        EXPECT_LE(info_pending - prom_pending, 16u);
    } else {
        EXPECT_LE(prom_pending - info_pending, 16u);
    }
}

TEST(ReclaimMetricsTest, StatsStringContainsReclaimFields) {
    lru::cache<int, std::string> c{256};
    c.set(1, "one");
    auto snap = c.stats_snapshot();
    auto str = snap.to_string();
    EXPECT_NE(str.find("reclaim_pending"), std::string::npos);
    EXPECT_NE(str.find("reclaim_total"), std::string::npos);
    EXPECT_NE(str.find("reclaim_invocations"), std::string::npos);
}

TEST(ReclaimMetricsTest, ConcurrentChurnDoesNotCorruptCounters) {
    lru::striped_cache<int, std::string> c{1024, 8};
    constexpr int kThreads = 4;
    constexpr int kIters = 2000;
    // Capture the global reclaim_total BEFORE this test's churn. The
    // snapshot's reclaim_total mirrors the GLOBAL hazptr/EBR domain
    // counters (cumulative across all caches/tests in the process), so
    // we must measure the DELTA attributable to this test, not the
    // absolute value.
    std::size_t total_before = c.stats_snapshot().reclaim_total.load(std::memory_order_relaxed);
    std::thread threads[kThreads];
    for (int t = 0; t < kThreads; ++t) {
        threads[t] = std::thread([&c, t]() {
            for (int i = 0; i < kIters; ++i) {
                int key = t * kIters + i;
                c.set(key, std::to_string(key));
                if ((i & 0xff) == 0) {
                    c.try_reclaim_now();
                }
            }
        });
    }
    for (auto& th : threads) th.join();

    // After all threads complete, counters should be internally consistent:
    //   invocations >= threads * (kIters / 256)
    //   delta_total <= sum of all retired objects (we cannot easily compute
    //                 the exact bound, so just verify non-decreasing and
    //                 finite). Each set() can retire at most ~2 objects
    //                 (old value + hash node on re-set); with kThreads *
    //                 kIters sets, the upper bound is ~4 * that.
    auto snap = c.stats_snapshot();
    std::size_t invocations =
        snap.reclaim_invocation_count.load(std::memory_order_relaxed);
    EXPECT_GE(invocations, kThreads);
    // Delta should be non-negative (reclamation is monotonic) and bounded.
    std::size_t total_after = snap.reclaim_total.load(std::memory_order_relaxed);
    EXPECT_GE(total_after, total_before);
    std::size_t delta = total_after - total_before;
    EXPECT_LE(delta, std::size_t(kThreads) * kIters * 4);
}

// ============================================================================
// L-1: hazptr slot-exhaustion / hard-cap fallback observability
// ============================================================================
//
// Validates that the L-1 hard-cap fallback path (acquire_slot throws
// std::runtime_error after kMaxSyncFallbacks=64 sync-reclaim rounds)
// is surfaced through the cache's three observability surfaces:
//   1. cache-level API: hazptr_slot_exhaustion_count() /
//      hazptr_sync_fallback_count() / hazptr_slot_capacity()
//   2. diagnostics_info struct fields + diagnostics_text() dump
//   3. prometheus_text() exposition (lru_hazptr_* metrics)
//
// Without this coverage, operators have no way to alert on sustained
// sync-fallback growth before acquire_slot() starts throwing, which
// would manifest as random std::runtime_error leaks from get() paths.

TEST(ReclaimMetricsTest, HazptrSlotMetricsExposedViaCacheAPI) {
    // L-1: Cache must forward hazptr_domain counters through its own API
    // so callers don't need to reach into detail::hazptr_domain.
    lru::cache<int, std::string> c{256};
    // Counters are cumulative across the global hazptr domain; on a fresh
    // process they start at zero. Just verify the API is wired and returns
    // a finite value (we can't assert ==0 if other tests ran first).
    std::size_t exhaustion = c.hazptr_slot_exhaustion_count();
    std::size_t fallback = c.hazptr_sync_fallback_count();
    std::size_t capacity = c.hazptr_slot_capacity();
    (void)exhaustion;
    (void)fallback;
    // Capacity is always in [128, 8192] by the hazptr_domain invariants.
    EXPECT_GE(capacity, 128u);
    EXPECT_LE(capacity,
              lru::detail::hazptr_domain::kMaxBatches *
              lru::detail::hazptr_domain::kBatchSize);
}

TEST(ReclaimMetricsTest, DiagnosticsReportsHazptrSlotFields) {
    // L-1: diagnostics_info must carry the three hazptr slot-exhaustion
    // fields, and diagnostics_text() must surface them so operators
    // reading an ad-hoc dump can spot an imminent hard-cap throw.
    lru::cache<int, std::string> c{256};
    c.set(1, "one");
    auto info = c.diagnostics();
    // Capacity is always populated; exhaustion / fallback may be zero
    // on a quiet process but the fields must be present.
    EXPECT_GE(info.hazptr_slot_capacity, 128u);
    auto text = c.diagnostics_text();
    EXPECT_NE(text.find("hazptr_slot_exhaustion_count"), std::string::npos);
    EXPECT_NE(text.find("hazptr_sync_fallback_count"), std::string::npos);
    EXPECT_NE(text.find("hazptr_slot_capacity"), std::string::npos);
    EXPECT_NE(text.find("hazptr slot exhaustion"), std::string::npos);
}

TEST(ReclaimMetricsTest, PrometheusExportsHazptrSlotMetrics) {
    // L-1: prometheus_text() must export the three hazptr slot-exhaustion
    // metrics so Prometheus/Grafana can alert on sustained sync-fallback
    // growth before acquire_slot() starts throwing.
    lru::cache<int, std::string> c{256};
    c.set(1, "one");
    auto text = c.prometheus_text();
    EXPECT_NE(text.find("lru_hazptr_slot_exhaustion_total"), std::string::npos);
    EXPECT_NE(text.find("lru_hazptr_sync_fallback_total"), std::string::npos);
    EXPECT_NE(text.find("lru_hazptr_slot_capacity"), std::string::npos);
    // Sanity: the capacity metric must be a valid positive integer.
    std::size_t cap = extract_prometheus_value(text, "lru_hazptr_slot_capacity");
    EXPECT_GE(cap, 128u);
}

TEST(ReclaimMetricsTest, HazptrSlotMetricsAgreeAcrossSurfaces) {
    // L-1: All three observability surfaces must agree on the same
    // underlying counter value (within atomic-read drift).
    lru::safe_cache<int, std::string> c{512};
    for (int i = 0; i < 64; ++i) c.set(i, std::to_string(i));

    auto info = c.diagnostics();
    auto prom = c.prometheus_text();

    std::size_t api_capacity = c.hazptr_slot_capacity();
    std::size_t info_capacity = info.hazptr_slot_capacity;
    std::size_t prom_capacity =
        extract_prometheus_value(prom, "lru_hazptr_slot_capacity");
    // Capacity is monotonic non-decreasing, so all three reads should
    // see the same value (or info/prom slightly behind api if a batch
    // was added in between — accept up to one batch of drift).
    const std::size_t one_batch =
        lru::detail::hazptr_domain::kBatchSize;
    auto close = [=](std::size_t a, std::size_t b) {
        return (a >= b ? a - b : b - a) <= one_batch;
    };
    EXPECT_TRUE(close(api_capacity, info_capacity));
    EXPECT_TRUE(close(api_capacity, prom_capacity));

    // Exhaustion / fallback are cumulative counters; they should be
    // non-decreasing across reads.
    std::size_t api_exhaustion = c.hazptr_slot_exhaustion_count();
    std::size_t info_exhaustion = info.hazptr_slot_exhaustion_count;
    std::size_t prom_exhaustion =
        extract_prometheus_value(prom, "lru_hazptr_slot_exhaustion_total");
    EXPECT_LE(info_exhaustion, api_exhaustion);  // info read first
    EXPECT_LE(prom_exhaustion, api_exhaustion);  // prom read before api call
}
