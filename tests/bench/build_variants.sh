#!/usr/bin/env bash
# build_variants.sh — compile all pdf-raster feature variants into tests/bench/bins/.
#
# Variants produced:
#   pdf-raster-scalar        no SIMD, no rayon
#   pdf-raster-avx2          AVX2 blend/fill
#   pdf-raster-avx512        AVX2 + AVX-512 VPOPCNTDQ popcnt
#   pdf-raster-rayon         rayon parallelism, no SIMD
#   pdf-raster-avx2-rayon    AVX2 + rayon
#   pdf-raster-avx512-rayon  AVX2 + AVX-512 + rayon  (full)
#
# Usage:
#   build_variants.sh [OPTIONS]
#
# Options:
#   -d    Dry run: print cargo commands without building
#
# The AVX-512 variants require that the host CPU supports avx512vpopcntdq.
# Check with: grep avx512_vpopcntdq /proc/cpuinfo
# If unsupported they will be skipped with a warning.
#
# Requires: cargo

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BINS_DIR="${SCRIPT_DIR}/bins"
DRY_RUN=false

usage() {
    grep '^#' "$0" | grep -v '^#!/' | sed 's/^# \{0,2\}//'
    exit 0
}

while getopts ":dh" opt; do
    case $opt in
        d) DRY_RUN=true ;;
        h) usage ;;
        :) echo "Error: option -$OPTARG requires an argument." >&2; exit 1 ;;
       \?) echo "Error: unknown option -$OPTARG." >&2; exit 1 ;;
    esac
done

command -v cargo >/dev/null 2>&1 || { echo "Error: cargo not found in PATH" >&2; exit 1; }

# Detect AVX-512 VPOPCNTDQ support on this host.
has_avx512=false
if grep -q avx512_vpopcntdq /proc/cpuinfo 2>/dev/null; then
    has_avx512=true
fi

# AVX-512 RUSTFLAGS: enable the ISA extensions the code actually uses.
AVX512_RUSTFLAGS="-C target-feature=+avx512f,+avx512bw,+avx512vl,+avx512vpopcntdq"

# build <out_name> <features> [<rustflags>]
build() {
    local out_name="$1" features="$2" rustflags="${3:-}"
    local dest="${BINS_DIR}/${out_name}"

    local cargo_cmd=(
        cargo build --release -p pdf-raster
        --no-default-features
    )
    [[ -n "$features" ]] && cargo_cmd+=(--features "$features")

    if $DRY_RUN; then
        if [[ -n "$rustflags" ]]; then
            printf "  RUSTFLAGS='%s' %s\n" "$rustflags" "${cargo_cmd[*]}"
        else
            printf "  %s\n" "${cargo_cmd[*]}"
        fi
        printf "  → %s\n\n" "$dest"
        return
    fi

    echo "── Building ${out_name} ──────────────────────────────────────────────────"
    if [[ -n "$rustflags" ]]; then
        RUSTFLAGS="$rustflags" "${cargo_cmd[@]}"
    else
        "${cargo_cmd[@]}"
    fi
    mkdir -p "$BINS_DIR"
    cp "${REPO_ROOT}/target/release/pdf-raster" "$dest"
    echo "   → ${dest}"
    echo ""
}

if $DRY_RUN; then
    echo "DRY RUN — cargo commands that would be executed:"
    echo ""
fi

cd "$REPO_ROOT"

build "pdf-raster-scalar"       ""
build "pdf-raster-avx2"         "raster/simd-avx2"
build "pdf-raster-rayon"        "raster/rayon"
build "pdf-raster-avx2-rayon"   "raster/simd-avx2,raster/rayon"

if $has_avx512; then
    build "pdf-raster-avx512"         "raster/simd-avx512"          "$AVX512_RUSTFLAGS"
    build "pdf-raster-avx512-rayon"   "raster/simd-avx512,raster/rayon" "$AVX512_RUSTFLAGS"
else
    echo "Warning: CPU does not support avx512_vpopcntdq — skipping avx512 variants." >&2
fi

if ! $DRY_RUN; then
    echo "── Done ─────────────────────────────────────────────────────────────────"
    ls -lh "$BINS_DIR/"
fi
