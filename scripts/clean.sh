#!/bin/bash
# =============================================================================
# clean.sh — Remove build directories and CMake-generated files
#
# Usage:
#   ./scripts/clean.sh            # Default: remove build/, build_*/, _deps/
#   ./scripts/clean.sh deep       # Also remove *.log and gdb_output.log
#   ./scripts/clean.sh all        # Remove EVERYTHING (incl. compile_commands.json)
#   ./scripts/clean.sh sanitizers # Remove only sanitizer build dirs (asan/tsan/...)
#
# The "sanitizers" mode is useful when iterating on sanitizer-specific
# issues without wiping the main build/ directory (which would force a
# full GoogleTest re-fetch + recompile).
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

MODE="${1:-default}"

echo "=== [clean.sh] mode=${MODE} ==="

# Sanitizer build directories created by build.sh. Each is independent
# (separate CMake cache, separate _deps) so they can be removed safely
# without affecting the main build/ directory.
SANITIZER_DIRS="build/asan build/tsan build/ubsan build/lsan build/asan-ubsan"

case "$MODE" in
    sanitizers)
        # Only remove sanitizer build directories; keep build/ and _deps/.
        rm -rf $SANITIZER_DIRS
        echo "[clean.sh] removed: $SANITIZER_DIRS"
        ;;
    default|deep|all)
        # Always remove build artifacts and generated CMake files.
        # The explicit sanitizer dirs are redundant with "rm -rf build"
        # but listed for clarity in case build/ was partially removed.
        rm -rf build build_* $SANITIZER_DIRS
        rm -rf _deps CMakeFiles CMakeCache.txt CTestTestfile.cmake cmake_install.cmake
        rm -rf Testing
        ;;
    *)
        echo "[clean.sh] Unknown mode: $MODE" >&2
        echo "[clean.sh] Valid modes: default | deep | all | sanitizers" >&2
        exit 2
        ;;
esac

if [[ "$MODE" == "deep" || "$MODE" == "all" ]]; then
    rm -f *.log gdb_output.log verify_output.log build_log.txt
fi

if [[ "$MODE" == "all" ]]; then
    rm -f compile_commands.json
fi

echo "[clean.sh] OK"
