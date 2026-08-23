#!/bin/bash
# =============================================================================
# test.sh — Generic test runner for the LRU cache library
#
# Usage:
#   ./scripts/test.sh                       # Run full ctest suite (build/)
#   ./scripts/test.sh asan                  # Run full ctest suite (build/asan/)
#   ./scripts/test.sh tsan                  # Run full ctest suite (build/tsan/)
#   ./scripts/test.sh ubsan                 # Run full ctest suite (build/ubsan/)
#   ./scripts/test.sh lsan                  # Run full ctest suite (build/lsan/)
#   ./scripts/test.sh asan-ubsan            # Run full ctest suite (build/asan-ubsan/)
#   ./scripts/test.sh -g "DeadlockDetection.*"     # Run tests matching gtest filter
#   ./scripts/test.sh -g "DeadlockDetection.StripedCacheFlushUnderLoad" -r 10
#   ./scripts/test.sh -R "RefcountTest"            # Run tests matching ctest regex
#   ./scripts/test.sh -e StressTest                # Run tests in executable StressTest
#   ./scripts/test.sh --stress-duration 5          # Set LRU_STRESS_DURATION_SECS=5
#
# Common combinations:
#   # Verify a flaky test no longer fails:
#   ./scripts/test.sh -g "DeadlockDetection.StripedCacheFlushUnderLoad" -r 20
#
#   # Run only the refcount tests:
#   ./scripts/test.sh -e RefcountTest
#
#   # ASan build + test in one go:
#   ./scripts/build.sh asan && ./scripts/test.sh asan
#
#   # ASan + UBSan combined (catches both memory and UB errors):
#   ./scripts/build.sh asan-ubsan && ./scripts/test.sh asan-ubsan
#
# Sanitizer runtime environment defaults (overridable via env):
#   ASAN_OPTIONS  — detect_leaks=0:abort_on_error=1:halt_on_error=0
#                   (MinGW: LSan unavailable, so detect_leaks=0; on Linux set
#                    detect_leaks=1 to enable leak checks.)
#   TSAN_OPTIONS  — halt_on_error=0:second_deadlock_stack=1
#   UBSAN_OPTIONS — print_stacktrace=1:halt_on_error=0
#                   (UB is compiled -fno-sanitize-recover=undefined, so halt_on_error
#                    here controls only the runtime print path; the abort already
#                    happens via the no-recover compile flag.)
#   LSAN_OPTIONS  — exitcode=23:report_objects=1
#
# Environment overrides:
#   BUILD_DIR     — output directory (overrides profile default)
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# --- MSYS2 Clang64 toolchain (Windows-only requirement) ---------------------
if [[ -d /clang64/bin ]]; then
    export PATH=/clang64/bin:/usr/bin:$PATH
fi

cd "${REPO_ROOT}"

# --- Defaults --------------------------------------------------------------
PROFILE="debug"
GTEST_FILTER=""
GTEST_REPEAT=1
CTEST_REGEX=""
EXE_NAME="lru_cache_test"
STRESS_DURATION=""
TIMEOUT_SECS=600
while [[ $# -gt 0 ]]; do
    case "$1" in
        debug|Debug)        PROFILE="debug" ;;
        release|Release)    PROFILE="release" ;;
        asan|ASan|ASAN)     PROFILE="asan" ;;
        tsan|TSan|TSAN)     PROFILE="tsan" ;;
        ubsan|UBSan|UBSAN)  PROFILE="ubsan" ;;
        lsan|LSan|LSAN)     PROFILE="lsan" ;;
        asan-ubsan|asan+ubsan) PROFILE="asan-ubsan" ;;
        -g|--gtest-filter)
            shift; GTEST_FILTER="$1" ;;
        -r|--repeat)
            shift; GTEST_REPEAT="$1" ;;
        -R|--ctest-regex)
            shift; CTEST_REGEX="$1" ;;
        -e|--executable)
            shift; EXE_NAME="$1" ;;
        --stress-duration)
            shift; STRESS_DURATION="$1" ;;
        -t|--timeout)
            shift; TIMEOUT_SECS="$1" ;;
        -h|--help)
            sed -n '3,40p' "$0"
            exit 0 ;;
        *)
            echo "[test.sh] Unknown argument: $1" >&2
            exit 2 ;;
    esac
    shift
done

# --- Resolve build directory ----------------------------------------------
BUILD_DIR="${BUILD_DIR:-}"
if [[ -z "${BUILD_DIR}" ]]; then
    case "$PROFILE" in
        debug)      BUILD_DIR="build" ;;
        release)    BUILD_DIR="build" ;;
        asan)       BUILD_DIR="build/asan" ;;
        tsan)       BUILD_DIR="build/tsan" ;;
        ubsan)      BUILD_DIR="build/ubsan" ;;
        lsan)       BUILD_DIR="build/lsan" ;;
        asan-ubsan) BUILD_DIR="build/asan-ubsan" ;;
    esac
fi

# --- Validate build exists -------------------------------------------------
if [[ ! -d "${BUILD_DIR}" ]]; then
    echo "[test.sh] ERROR: build directory '${BUILD_DIR}' does not exist." >&2
    echo "[test.sh]        Run: ./scripts/build.sh ${PROFILE}" >&2
    exit 2
fi

# --- Apply sanitizer runtime env defaults ---------------------------------
# Each *_OPTIONS variable is only set if not already present in the
# environment, so users can override per-invocation:
#   ASAN_OPTIONS=detect_leaks=1 ./scripts/test.sh asan
case "$PROFILE" in
    asan|asan-ubsan)
        # detect_leaks=0 because MinGW LSan is unavailable; on Linux flip to 1.
        # abort_on_error=1 so ASan violations fail the test process.
        # halt_on_error=0 lets the first violation print a full report
        # before aborting (better stack traces).
        export ASAN_OPTIONS="${ASAN_OPTIONS:-detect_leaks=0:abort_on_error=1:halt_on_error=0:print_stacktrace=1}"
        ;;
    tsan)
        # halt_on_error=0 so TSan continues and reports all races in a run.
        # second_deadlock_stack=1 reports both sides of a deadlock.
        export TSAN_OPTIONS="${TSAN_OPTIONS:-halt_on_error=0:second_deadlock_stack=1:report_bugs=1}"
        ;;
    ubsan)
        # UBSan is compiled with -fno-sanitize-recover=undefined, so the
        # first UB already aborts. print_stacktrace=1 adds the C++ stack
        # to the report. halt_on_error=0 is a no-op given no-recover but
        # kept for forward compatibility if we ever flip to recover mode.
        export UBSAN_OPTIONS="${UBSAN_OPTIONS:-print_stacktrace=1:halt_on_error=0}"
        ;;
    lsan)
        # exitcode=23 is the conventional LSan failure exit code (distinct
        # from gtest's 1). report_objects=1 lists each leaked object.
        export LSAN_OPTIONS="${LSAN_OPTIONS:-exitcode=23:report_objects=1}"
        ;;
esac

# --- Compose combined sanitizer env when profile = asan-ubsan -------------
# UBSan's runtime (libUBSan) is separate from ASan's, so both *_OPTIONS
# must be set. The order in the env does not matter; both runtimes read
# their respective variables at process startup.
if [[ "$PROFILE" == "asan-ubsan" ]]; then
    export UBSAN_OPTIONS="${UBSAN_OPTIONS:-print_stacktrace=1:halt_on_error=0}"
fi

if [[ -n "$STRESS_DURATION" ]]; then
    export LRU_STRESS_DURATION_SECS="$STRESS_DURATION"
fi

# --- Decide between gtest direct (filter or executable specified) vs ctest --
USE_GTEST_DIRECT=0
if [[ -n "$GTEST_FILTER" ]]; then
    USE_GTEST_DIRECT=1
elif [[ "$EXE_NAME" != "lru_cache_test" ]]; then
    # executable specified (e.g., lru_refcount_test)
    USE_GTEST_DIRECT=1
fi

if [[ $USE_GTEST_DIRECT -eq 1 ]]; then
    # Run via gtest executable directly (supports --gtest_repeat, gtest_filter)
    EXE_PATH="${BUILD_DIR}/tests/${EXE_NAME}.exe"
    if [[ ! -x "$EXE_PATH" ]]; then
        # Fallback for non-Windows: try without .exe
        EXE_PATH_NOEXT="${BUILD_DIR}/tests/${EXE_NAME}"
        if [[ -x "$EXE_PATH_NOEXT" ]]; then
            EXE_PATH="$EXE_PATH_NOEXT"
        else
            echo "[test.sh] ERROR: executable not found: ${EXE_PATH}" >&2
            exit 2
        fi
    fi

    ARGS=("--gtest_repeat=${GTEST_REPEAT}" "--gtest_break_on_failure=0" "--gtest_catch_exceptions=1")
    if [[ -n "$GTEST_FILTER" ]]; then
        ARGS+=("--gtest_filter=${GTEST_FILTER}")
    fi

    echo "=== [test.sh] ${EXE_PATH} ${ARGS[*]} ==="
    timeout "${TIMEOUT_SECS}" "$EXE_PATH" "${ARGS[@]}"
    rc=$?
else
    # Run via ctest (no --gtest_repeat, but supports -R regex)
    echo "=== [test.sh] ctest --test-dir ${BUILD_DIR} --output-on-failure ==="
    if [[ -n "$CTEST_REGEX" ]]; then
        timeout "${TIMEOUT_SECS}" ctest --test-dir "${BUILD_DIR}" --output-on-failure -R "$CTEST_REGEX"
    else
        timeout "${TIMEOUT_SECS}" ctest --test-dir "${BUILD_DIR}" --output-on-failure
    fi
    rc=$?
fi

if [[ $rc -ne 0 ]]; then
    echo "[test.sh] FAILED (exit $rc)" >&2
    # Surface which sanitizer (if any) was active to help interpret the exit code.
    case "$PROFILE" in
        asan)       echo "[test.sh] ASan was active; exit 1 = memory error." >&2 ;;
        tsan)       echo "[test.sh] TSan was active; exit 66 = data race / deadlock." >&2 ;;
        ubsan)      echo "[test.sh] UBSan was active; exit 1 = undefined behavior." >&2 ;;
        lsan)       echo "[test.sh] LSan was active; exit 23 = memory leak detected." >&2 ;;
        asan-ubsan) echo "[test.sh] ASan+UBSan was active; exit 1 = memory error or UB." >&2 ;;
    esac
    exit $rc
fi

echo "[test.sh] OK: all tests passed"
exit 0
