#!/usr/bin/env bash
# gpu-jpeg-huffman gate — 10-corpus × 3-mode wrapper.
#
# Modes:
#   A    CPU only (regression baseline — matches v0.7.0 mode A within noise).
#   D    Full GPU (nvjpeg + gpu-aa + gpu-icc), no cache.
#   H    GPU parallel-Huffman JPEG (gpu-jpeg-huffman + gpu-aa + gpu-icc).
#
# Gate passes when mode H ≤ 0.7× mode A on ≥ 1 corpus in 06-09.
#
# Usage:
#   scripts/bench_v080.sh [--force]
set -euo pipefail

OUT_DIR="bench/v080"
BENCH="tests/bench_corpus.sh"
MIN_FREE_GB=20

probe_path() {
  local var="$1"; shift
  if [[ -n "${!var:-}" ]]; then return; fi
  for cand in "$@"; do
    if [[ -e "$cand" ]]; then eval "$var=\"\$cand\""; return; fi
  done
}
probe_path CUDA_LIB \
  "/usr/local/cuda/lib64" \
  "/usr/local/cuda-13.2/lib64" \
  "/usr/local/cuda-13/lib64" \
  "/usr/local/cuda-12.8/lib64" \
  "/usr/lib/x86_64-linux-gnu"
CUDA_LIB="${CUDA_LIB:-/usr/local/cuda/lib64}"

MODES=(A D H)
declare -A MODE_BIN=([A]=cpu [D]=full-gpu [H]=huffman)
declare -A MODE_BACKEND=([A]=cpu [D]=cuda [H]=cuda)
declare -A MODE_LABEL=(
  [A]="CPU-only"
  [D]="Full GPU (nvjpeg, no cache)"
  [H]="GPU parallel-Huffman JPEG"
)

FORCE=0
if [[ ${1:-} == "--force" ]]; then FORCE=1; fi
SKIP_CUDA=0

log()  { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
fail() { printf '[%s] FAIL: %s\n' "$(date +%H:%M:%S)" "$*" >&2; exit 1; }
warn() { printf '[%s] WARN: %s\n' "$(date +%H:%M:%S)" "$*" >&2; }

preflight() {
  log "Pre-flight"
  local free_gb
  free_gb=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
  (( free_gb >= MIN_FREE_GB )) || fail "free disk ${free_gb}G < ${MIN_FREE_GB}G required"
  log "  disk: ${free_gb}G free"

  for cmd in cargo hyperfine mpstat iostat python3 awk; do
    command -v "$cmd" >/dev/null || fail "$cmd not in PATH"
  done
  [[ -x "$BENCH" ]] || fail "$BENCH not executable"
  log "  CUDA_ARCH: ${CUDA_ARCH:-sm_80 (build-script default)}"

  if [[ ! -e /dev/nvidia0 ]]; then
    warn "/dev/nvidia0 missing; modes D/H will be skipped"
    SKIP_CUDA=1
  elif [[ ! -e "$CUDA_LIB/libcudart.so" || ! -e "$CUDA_LIB/libnvjpeg.so" ]]; then
    warn "CUDA libs missing under $CUDA_LIB (need libcudart.so + libnvjpeg.so); modes D/H will be skipped"
    SKIP_CUDA=1
  else
    log "  nvidia: /dev/nvidia0 + cudart + nvjpeg present"
  fi

  mkdir -p "$OUT_DIR"
  log "Pre-flight OK"
}

build_binary() {
  local suffix="$1" features="$2"
  local out="target/release/pdf-raster-$suffix"
  if [[ -x "$out" && $FORCE -eq 0 ]]; then
    log "build[$suffix]: already present"; return 0
  fi
  log "build[$suffix]: features='$features'"
  RUSTFLAGS="-C target-cpu=native" \
    cargo build --release -p pdf-raster --features "$features"
  cp -f target/release/pdf-raster "$out"
  log "build[$suffix]: copied to $out"
}

builds() {
  log "Build phase"
  build_binary cpu ""
  if [[ $SKIP_CUDA -eq 0 ]]; then
    build_binary full-gpu "nvjpeg,gpu-aa,gpu-icc"
    build_binary huffman  "gpu-jpeg-huffman,gpu-aa,gpu-icc"
  else
    log "build[full-gpu]: SKIP (CUDA unavailable)"
    log "build[huffman]:  SKIP (CUDA unavailable)"
  fi
  log "Build phase OK"
}

bench_mode() {
  local mode="$1"
  local bin_suffix="${MODE_BIN[$mode]}"
  local backend="${MODE_BACKEND[$mode]}"
  local label="${MODE_LABEL[$mode]}"
  local bin="target/release/pdf-raster-$bin_suffix"
  local out="$OUT_DIR/$mode.txt"

  if [[ -f "$out" && $FORCE -eq 0 ]]; then
    log "bench[$mode/$label]: SKIP (already done — $out)"; return 0
  fi
  if [[ ! -x "$bin" ]]; then
    log "bench[$mode/$label]: SKIP (binary $bin missing)"; return 0
  fi
  if [[ "$backend" == "cuda" && $SKIP_CUDA -eq 1 ]]; then
    log "bench[$mode/$label]: SKIP (CUDA unavailable)"; return 0
  fi

  log "bench[$mode/$label]: starting"
  local ld_env=""
  if [[ "$backend" == "cuda" && -d "$CUDA_LIB" ]]; then
    ld_env="LD_LIBRARY_PATH=$CUDA_LIB:${LD_LIBRARY_PATH:-}"
  fi
  # shellcheck disable=SC2086
  env ${ld_env:+$ld_env} BIN="$bin" \
    "$BENCH" --backend "$backend" | tee "$out"
  log "bench[$mode/$label]: done — $out"
}

benches() {
  log "Bench phase"
  for mode in "${MODES[@]}"; do bench_mode "$mode"; done
  log "Bench phase OK"
}

aggregate() {
  log "Aggregating to $OUT_DIR/results.md"
  python3 scripts/aggregate_v080.py "$OUT_DIR" > "$OUT_DIR/results.md"
  log "Wrote $OUT_DIR/results.md"
}

preflight
builds
benches
aggregate
log "All done. Gate results: $OUT_DIR/results.md"
