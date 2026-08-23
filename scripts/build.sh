#!/bin/bash
# =============================================================================
# build.sh — Generic CMake build script for the LRU cache library
#
# Usage:
#   ./scripts/build.sh                # Debug build, tests on, examples off
#   ./scripts/build.sh release        # Release build, tests on
#   ./scripts/build.sh asan           # Debug + AddressSanitizer (Clang64 only)
#   ./scripts/build.sh tsan           # Debug + ThreadSanitizer (Linux only)
#   ./scripts/build.sh ubsan          # Debug + UBSan (undefined behavior)
#   ./scripts/build.sh lsan           # Debug + LeakSanitizer (Linux only)
#   ./scripts/build.sh asan-ubsan     # Debug + ASan + UBSan (combined)
#   ./scripts/build.sh bench         # Release build, benchmarks on
#   ./scripts/build.sh examples       # Debug build, examples on
#   ./scripts/build.sh -j4            # Override parallelism (default: -j2)
#
# Sanitizer composition matrix (enforced by CMakeLists.txt):
#   asan       — ASan (includes LSan on Linux; on MinGW LSan is unavailable)
#   ubsan      — UBSan only (UB is fatal: -fno-sanitize-recover=undefined)
#   lsan       — LSan standalone (Linux only; MinGW/Windows unsupported)
#   asan-ubsan — ASan + UBSan (Clang/GCC compose cleanly; both fatal)
#   tsan       — TSan (mutually exclusive with ASan/LSan; Linux only)
#
# Environment overrides:
#   BUILD_DIR     — output directory (default: build, build/asan, build/bench, ...)
#   CC, CXX       — compilers (default: ccache clang / ccache clang++)
#   LRU_FORCE_CONFIGURE=1 — force cmake re-configure even if Makefile exists
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# --- MSYS2 Clang64 toolchain (Windows-only requirement) ---------------------
if [[ -d /clang64/bin ]]; then
    export PATH=/clang64/bin:/usr/bin:$PATH
fi
export CC="${CC:-ccache clang}"
export CXX="${CXX:-ccache clang++}"

cd "${REPO_ROOT}"

# --- Parse arguments --------------------------------------------------------
PROFILE="debug"
PARALLEL="-j2"
for arg in "$@"; do
    case "$arg" in
        debug|Debug)         PROFILE="debug" ;;
        release|Release)     PROFILE="release" ;;
        asan|ASan|ASAN)      PROFILE="asan" ;;
        tsan|TSan|TSAN)      PROFILE="tsan" ;;
        ubsan|UBSan|UBSAN)   PROFILE="ubsan" ;;
        lsan|LSan|LSAN)      PROFILE="lsan" ;;
        asan-ubsan|asan+ubsan) PROFILE="asan-ubsan" ;;
        bench|benchmarks)    PROFILE="bench" ;;
        examples|example)    PROFILE="examples" ;;
        -j*)                 PARALLEL="$arg" ;;
        -h|--help)
            sed -n '3,30p' "$0"
            exit 0 ;;
        *)
            echo "[build.sh] Unknown argument: $arg" >&2
            exit 2 ;;
    esac
done

# --- Configure per-profile variables ---------------------------------------
BUILD_TYPE="Debug"
BUILD_DIR_DEFAULT="build"
LRU_BUILD_TESTS=ON
LRU_BUILD_EXAMPLES=OFF
LRU_BUILD_BENCHMARKS=OFF
LRU_ENABLE_ASAN=OFF
LRU_ENABLE_TSAN=OFF
LRU_ENABLE_UBSAN=OFF
LRU_ENABLE_LSAN=OFF
EXTRA_CMAKE_ARGS=()

case "$PROFILE" in
    release)
        BUILD_TYPE="Release"
        BUILD_DIR_DEFAULT="build"
        ;;
    asan)
        BUILD_TYPE="Debug"
        BUILD_DIR_DEFAULT="build/asan"
        LRU_ENABLE_ASAN=ON
        # MinGW: bypass ccache so ASan runtime symbols resolve correctly
        if [[ -d /clang64/bin ]]; then
            export CC="clang"
            export CXX="clang++"
        fi
        # Filter out conflicting conda library paths if present
        if [[ -d /e/Python/miniconda3/Library ]]; then
            EXTRA_CMAKE_ARGS+=(-DCMAKE_IGNORE_PATH="E:/Python/miniconda3/Library;E:/Python/miniconda3/Library/lib")
        fi
        ;;
    tsan)
        BUILD_TYPE="Debug"
        BUILD_DIR_DEFAULT="build/tsan"
        LRU_ENABLE_TSAN=ON
        # TSan is Linux-only (enforced by CMakeLists.txt); on MinGW the
        # configure step will FATAL_ERROR with a clear message.
        ;;
    ubsan)
        BUILD_TYPE="Debug"
        BUILD_DIR_DEFAULT="build/ubsan"
        LRU_ENABLE_UBSAN=ON
        # UBSan composes with ccache; no need to bypass it.
        ;;
    lsan)
        BUILD_TYPE="Debug"
        BUILD_DIR_DEFAULT="build/lsan"
        LRU_ENABLE_LSAN=ON
        # LSan standalone is Linux-only. On MinGW/Windows the linker will
        # fail with "undefined reference to __lsan_*"; surface this early.
        if [[ -d /clang64/bin ]]; then
            echo "[build.sh] WARNING: LSan standalone is not available on MinGW/Windows." >&2
            echo "[build.sh]          ASan (./scripts/build.sh asan) includes leak detection" >&2
            echo "[build.sh]          on Linux; on MinGW use ASan with detect_leaks=0." >&2
        fi
        ;;
    asan-ubsan)
        BUILD_TYPE="Debug"
        BUILD_DIR_DEFAULT="build/asan-ubsan"
        LRU_ENABLE_ASAN=ON
        LRU_ENABLE_UBSAN=ON
        if [[ -d /clang64/bin ]]; then
            export CC="clang"
            export CXX="clang++"
        fi
        if [[ -d /e/Python/miniconda3/Library ]]; then
            EXTRA_CMAKE_ARGS+=(-DCMAKE_IGNORE_PATH="E:/Python/miniconda3/Library;E:/Python/miniconda3/Library/lib")
        fi
        ;;
    bench)
        BUILD_TYPE="Release"
        BUILD_DIR_DEFAULT="build/bench"
        LRU_BUILD_TESTS=OFF
        LRU_BUILD_BENCHMARKS=ON
        ;;
    examples)
        BUILD_TYPE="Debug"
        BUILD_DIR_DEFAULT="build"
        LRU_BUILD_EXAMPLES=ON
        LRU_BUILD_TESTS=OFF
        ;;
esac

BUILD_DIR="${BUILD_DIR:-${BUILD_DIR_DEFAULT}}"

# Skip miniconda paths when present on Windows
if [[ -z "${CMAKE_IGNORE_PATH:-}" && -d /e/ProgramData/miniconda3/Library/mingw-w64 ]]; then
    EXTRA_CMAKE_ARGS+=(-DCMAKE_IGNORE_PATH="E:/ProgramData/miniconda3/Library/mingw-w64")
fi

# --- Configure (if needed) -------------------------------------------------
NEEDS_CONFIGURE=0
if [[ ! -f "${BUILD_DIR}/Makefile" ]]; then
    NEEDS_CONFIGURE=1
elif [[ "${LRU_FORCE_CONFIGURE:-0}" == "1" ]]; then
    NEEDS_CONFIGURE=1
fi

if [[ $NEEDS_CONFIGURE -eq 1 ]]; then
    echo "=== [build.sh] Configuring ${PROFILE} (BUILD_DIR=${BUILD_DIR}) ==="
    cmake -B "${BUILD_DIR}" -G "MinGW Makefiles" \
        -DCMAKE_CXX_COMPILER="${CXX##* }" \
        -DCMAKE_C_COMPILER="${CC##* }" \
        -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
        -DLRU_BUILD_TESTS=${LRU_BUILD_TESTS} \
        -DLRU_BUILD_EXAMPLES=${LRU_BUILD_EXAMPLES} \
        -DLRU_BUILD_BENCHMARKS=${LRU_BUILD_BENCHMARKS} \
        -DLRU_ENABLE_ASAN=${LRU_ENABLE_ASAN} \
        -DLRU_ENABLE_TSAN=${LRU_ENABLE_TSAN} \
        -DLRU_ENABLE_UBSAN=${LRU_ENABLE_UBSAN} \
        -DLRU_ENABLE_LSAN=${LRU_ENABLE_LSAN} \
        "${EXTRA_CMAKE_ARGS[@]}"
    rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "[build.sh] cmake configure failed (exit $rc)" >&2
        exit $rc
    fi
else
    echo "=== [build.sh] Using existing ${BUILD_DIR}/Makefile ==="
fi

# --- Build -----------------------------------------------------------------
echo "=== [build.sh] Building ${PARALLEL} ==="
mingw32-make -C "${BUILD_DIR}" "${PARALLEL}"
rc=$?
if [[ $rc -ne 0 ]]; then
    echo "[build.sh] build failed (exit $rc)" >&2
    exit $rc
fi

echo "[build.sh] OK: ${PROFILE} build complete in ${BUILD_DIR}"
echo "[build.sh] Next: ./scripts/test.sh ${PROFILE}"
exit 0
