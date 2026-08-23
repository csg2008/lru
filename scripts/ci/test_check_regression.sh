#!/usr/bin/env bash
# Smoke-test the benchmark regression check script with synthetic JSON
# fixtures. Verifies the script correctly detects a regression, accepts
# an improvement, and handles a missing baseline gracefully.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CHECK="${REPO_ROOT}/scripts/ci/check_benchmark_regression.sh"

TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

# Synthetic Google Benchmark JSON with two tail-latency benchmarks.
# Current results: p99_max_ns is 600 (safe_cache) and 700 (striped_cache).
cat > "$TMPDIR/current.json" <<EOF
{
  "benchmarks": [
    {
      "name": "BM_TailLatency_SafeCache/16",
      "counters": {
        "p50_last_ns": {"value": 100.0},
        "p95_last_ns": {"value": 200.0},
        "p99_last_ns": {"value": 500.0},
        "p50_max_ns":  {"value": 150.0},
        "p95_max_ns":  {"value": 300.0},
        "p99_max_ns":  {"value": 600.0}
      }
    },
    {
      "name": "BM_TailLatency_StripedCache/16",
      "counters": {
        "p50_last_ns": {"value": 110.0},
        "p95_last_ns": {"value": 220.0},
        "p99_last_ns": {"value": 550.0},
        "p50_max_ns":  {"value": 160.0},
        "p95_max_ns":  {"value": 330.0},
        "p99_max_ns":  {"value": 700.0}
      }
    }
  ]
}
EOF

# Baseline: p99_max_ns was 500 (safe_cache) and 600 (striped_cache).
# SafeCache: 600 vs 500 = +20% — exactly at threshold (not a regression).
# StripedCache: 700 vs 600 = +16.67% — under threshold.
cat > "$TMPDIR/baseline_ok.json" <<EOF
{
  "benchmarks": [
    {
      "name": "BM_TailLatency_SafeCache/16",
      "counters": {"p99_max_ns": {"value": 500.0}}
    },
    {
      "name": "BM_TailLatency_StripedCache/16",
      "counters": {"p99_max_ns": {"value": 600.0}}
    }
  ]
}
EOF

# Baseline: p99_max_ns was 400 (safe_cache). 600 vs 400 = +50% — regression.
cat > "$TMPDIR/baseline_regress.json" <<EOF
{
  "benchmarks": [
    {
      "name": "BM_TailLatency_SafeCache/16",
      "counters": {"p99_max_ns": {"value": 400.0}}
    },
    {
      "name": "BM_TailLatency_StripedCache/16",
      "counters": {"p99_max_ns": {"value": 600.0}}
    }
  ]
}
EOF

echo "=== Test 1: no regression (expect PASS, exit 0) ==="
"$CHECK" --baseline "$TMPDIR/baseline_ok.json" --current "$TMPDIR/current.json"
echo "exit=$?"
echo

echo "=== Test 2: regression (expect FAIL, exit 1) ==="
"$CHECK" --baseline "$TMPDIR/baseline_regress.json" --current "$TMPDIR/current.json"
echo "exit=$?"
echo

echo "=== Test 3: missing baseline (expect PASS, exit 0) ==="
"$CHECK" --baseline "$TMPDIR/nonexistent.json" --current "$TMPDIR/current.json"
echo "exit=$?"
echo

echo "=== Test 4: improvement (expect PASS, exit 0) ==="
# Swap: baseline has higher p99 than current.
"$CHECK" --baseline "$TMPDIR/current.json" --current "$TMPDIR/baseline_ok.json"
echo "exit=$?"
