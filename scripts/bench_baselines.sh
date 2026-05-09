#!/usr/bin/env bash
# External-rasterizer baselines (mutool draw + pdftoppm) across the corpus.
#
# Runs tests/bench_compare.sh once per reference tool and saves the captured
# output under bench/v07x-baselines/.  This is a one-shot baseline snapshot
# we don't repeat every release — these tools don't change much, and the
# goal is a reference point that's better calibrated than pdftoppm-alone.
#
# Usage:
#   scripts/bench_baselines.sh [--label local|testbench]
#
# Builds a CPU-only pdf-raster binary if one isn't already present.
set -euo pipefail

LABEL="local"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --label) LABEL="$2"; shift 2 ;;
    *) echo "ERROR: unknown argument: $1" >&2; exit 1 ;;
  esac
done

cd "$(dirname "$0")/.."
OUT_DIR="bench/v07x-baselines"
mkdir -p "$OUT_DIR"

# CPU-only release build (no GPU features).  Picks up target-cpu=native
# from .cargo/config.toml so AVX-512 paths and host auto-vectorisation are
# both active — same effective CPU codegen as mode A in bench_v070.sh.
# Cargo short-circuits if nothing has changed; the always-build is cheap
# and guarantees we pick up config/source changes.
BIN="target/release/pdf-raster"
echo "Ensuring CPU-only pdf-raster is up to date…"
cargo build --release -p pdf-raster --no-default-features --bin pdf-raster

GIT_SHA="$(git rev-parse --short HEAD)"
HOSTNAME_S="$(hostname)"
KERNEL="$(uname -r)"
CPU="$(grep -m1 '^model name' /proc/cpuinfo | sed 's/^[^:]*: *//')"

run_one() {
  local tool="$1"
  local out="$OUT_DIR/${LABEL}-${tool}.txt"
  echo "=== $tool baseline ($LABEL) -> $out ==="
  {
    printf "Host:     %s   Kernel: %s\n" "$HOSTNAME_S" "$KERNEL"
    printf "CPU:      %s\n" "$CPU"
    printf "Repo SHA: %s\n" "$GIT_SHA"
    bash tests/bench_compare.sh --ref-tool "$tool"
  } > "$out" 2>&1
  echo "  done"
}

run_one pdftoppm
run_one mutool

echo
echo "Baselines written to $OUT_DIR/${LABEL}-{pdftoppm,mutool}.txt"
