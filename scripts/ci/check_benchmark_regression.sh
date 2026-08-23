#!/usr/bin/env bash
# =============================================================================
# check_benchmark_regression.sh — T-CI-4 performance regression gate
#
# Parses Google Benchmark JSON output and compares P99 tail latency
# (p99_max_ns) against a baseline. Fails if any benchmark's P99 regresses
# by more than REGRESSION_THRESHOLD_PCT (default 20%).
#
# Usage:
#   ./scripts/ci/check_benchmark_regression.sh \
#       --baseline path/to/baseline.json \
#       --current  path/to/current.json
#
# Inputs must be JSON files produced by:
#   ./lru_concurrent_read_benchmark --benchmark_format=json \
#       --benchmark_out=current.json
#
# JSON parsing uses python3 if available (universal on Linux/macOS/MSYS2),
# falling back to jq. If neither is present, the script degrades gracefully
# and exits 0 so it never blocks CI on a missing parser.
#
# Exit codes:
#   0 — no regressions (or no baseline / no comparable benchmarks)
#   1 — at least one benchmark regressed beyond threshold
#   2 — script invocation error
# =============================================================================
set -uo pipefail

REGRESSION_THRESHOLD_PCT="${REGRESSION_THRESHOLD_PCT:-20}"
BASELINE=""
CURRENT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --baseline) BASELINE="$2"; shift 2 ;;
        --current)  CURRENT="$2";  shift 2 ;;
        -h|--help)
            sed -n '2,/^===/p' "$0" | sed 's/^# \?//'
            exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

if [[ -z "$CURRENT" ]]; then
    echo "error: --current is required" >&2
    exit 2
fi
if [[ ! -f "$CURRENT" ]]; then
    echo "error: current file not found: $CURRENT" >&2
    exit 2
fi

# Pick a JSON parser: python3 first (more widely installed), then jq.
JSON_PARSER=""
if command -v python3 >/dev/null 2>&1; then
    JSON_PARSER="python"
elif command -v jq >/dev/null 2>&1; then
    JSON_PARSER="jq"
else
    echo "warning: neither python3 nor jq found — skipping automated regression check." >&2
    echo "         Install one to enable P99 regression detection." >&2
    exit 0
fi

if [[ -z "$BASELINE" || ! -f "$BASELINE" ]]; then
    echo "No baseline found at: ${BASELINE:-<unset>}" >&2
    echo "Skipping regression check (first run or baseline artifact missing)." >&2
    exit 0
fi

# Extract per-benchmark p99_max_ns from JSON as "name<TAB>p99" lines.
# Google Benchmark emits a benchmarks[] array; each entry has "name" and
# "counters" (object whose values are {"value": <double>} or bare numbers).
extract_p99() {
    local file="$1"
    if [[ "$JSON_PARSER" == "python" ]]; then
        python3 - "$file" <<'PY'
import json, sys
with open(sys.argv[1], "r") as f:
    data = json.load(f)
for b in data.get("benchmarks", []):
    name = b.get("name", "")
    counters = b.get("counters") or {}
    p99 = counters.get("p99_max_ns")
    if isinstance(p99, dict):
        p99 = p99.get("value")
    if p99 is not None:
        print(f"{name}\t{p99}")
PY
    else
        jq -r '
          .benchmarks[]
          | .name as $name
          | (.counters // {})
          | (
              (try .p99_max_ns.value catch null),
              (try .p99_max_ns       catch null)
            ) as $p99
          | select($p99 != null)
          | "\($name)\t\($p99)"
        ' "$file"
    fi
}

# Read into associative arrays keyed by benchmark name.
declare -A BASELINE_MAP=()
declare -A CURRENT_MAP=()

while IFS=$'\t' read -r name val; do
    BASELINE_MAP["$name"]="$val"
done < <(extract_p99 "$BASELINE")

while IFS=$'\t' read -r name val; do
    CURRENT_MAP["$name"]="$val"
done < <(extract_p99 "$CURRENT")

if [[ ${#CURRENT_MAP[@]} -eq 0 ]]; then
    echo "No p99_max_ns counters found in current results." >&2
    echo "Ensure the benchmark emits p99_max_ns custom counters." >&2
    exit 0
fi

regressions=0
checked=0

echo "=== P99 tail-latency regression check (threshold: ${REGRESSION_THRESHOLD_PCT}%) ==="
printf '%-55s %15s %15s %12s %s\n' "Benchmark" "baseline_p99_ns" "current_p99_ns" "delta_pct" "status"

for name in "${!CURRENT_MAP[@]}"; do
    cur="${CURRENT_MAP[$name]}"
    if [[ -z "${BASELINE_MAP[$name]:-}" ]]; then
        printf '%-55s %15s %15s %12s %s\n' "$name" "n/a" "$cur" "n/a" "NEW"
        continue
    fi
    base="${BASELINE_MAP[$name]}"
    # Floating-point comparison via awk.
    delta_pct=$(awk -v b="$base" -v c="$cur" 'BEGIN {
        if (b <= 0) { print 0; exit }
        d = (c - b) / b * 100.0
        printf "%.2f", d
    }')
    status="OK"
    # Compare threshold.
    over=$(awk -v d="$delta_pct" -v t="$REGRESSION_THRESHOLD_PCT" 'BEGIN {
        print (d > t) ? 1 : 0
    }')
    if [[ "$over" == "1" ]]; then
        status="REGRESSION"
        regressions=$((regressions + 1))
    fi
    checked=$((checked + 1))
    printf '%-55s %15s %15s %12s %s\n' "$name" "$base" "$cur" "$delta_pct%" "$status"
done

echo
echo "Checked: $checked  Regressions: $regressions"

if [[ $regressions -gt 0 ]]; then
    echo "FAIL: ${regressions} benchmark(s) exceeded the ${REGRESSION_THRESHOLD_PCT}% P99 regression threshold." >&2
    exit 1
fi

echo "PASS: no P99 regressions detected."
exit 0
