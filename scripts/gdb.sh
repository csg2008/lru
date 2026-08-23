#!/bin/bash
# =============================================================================
# gdb.sh — Run a single gtest test under GDB (batch mode)
#
# Usage:
#   ./scripts/gdb.sh "StressTest.MemoryPressureWithEviction"
#   ./scripts/gdb.sh "DeadlockDetection.StripedCacheFlushUnderLoad" -t 30
#
# Captures full backtrace and thread info. Output is also mirrored to
# gdb_output.log in the repo root.
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -d /clang64/bin ]]; then
    export PATH=/clang64/bin:/usr/bin:$PATH
fi

cd "${REPO_ROOT}"

GTEST_FILTER="${1:-}"
TIMEOUT_SECS=120
if [[ "${2:-}" == "-t" ]]; then
    TIMEOUT_SECS="${3:-120}"
fi

if [[ -z "$GTEST_FILTER" ]]; then
    echo "Usage: $0 <gtest_filter> [-t <timeout_secs>]" >&2
    echo "  e.g. $0 'StressTest.MemoryPressureWithEviction'" >&2
    exit 2
fi

EXE="build/tests/lru_cache_test.exe"
if [[ ! -x "$EXE" ]]; then
    EXE="build/tests/lru_cache_test"
fi
if [[ ! -x "$EXE" ]]; then
    echo "[gdb.sh] ERROR: test executable not found (need to build first)" >&2
    exit 2
fi

GDB=/e/GCC/msys64/usr/bin/gdb.exe
if [[ ! -x "$GDB" ]]; then
    GDB=$(command -v gdb)
fi
if [[ -z "$GDB" ]]; then
    echo "[gdb.sh] ERROR: gdb not found in PATH and not at /e/GCC/msys64/usr/bin/gdb.exe" >&2
    exit 2
fi

LOG="${REPO_ROOT}/gdb_output.log"

{
    echo "=== [gdb.sh] filter='${GTEST_FILTER}' timeout=${TIMEOUT_SECS}s ==="
    echo "--- GDB: ${GDB} ---"
    cd "$(dirname "${EXE}")"

    timeout "${TIMEOUT_SECS}" "${GDB}" -batch \
        -ex "set pagination off" \
        -ex "run --gtest_filter=${GTEST_FILTER}" \
        -ex "bt 30" \
        -ex "info threads" \
        -ex "thread apply all bt 8" \
        ./$(basename "${EXE}")
} > "$LOG" 2>&1

echo "[gdb.sh] DONE — see ${LOG}"
echo ""
echo "--- Last 40 lines ---"
tail -40 "$LOG"
