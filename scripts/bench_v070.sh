#!/usr/bin/env bash
# v0.7.0 GPU + image-cache matrix — 10-corpus × 4-mode wrapper.
#
# Modes:
#   A    CPU only (regression check — should match v0.6.0 mode A within noise).
#   D    Full GPU (nvjpeg + gpu-aa + gpu-icc), no image cache (regression
#        check — should match v0.6.0 mode D within noise).
#   DC   Full GPU + image cache (cache feature on, prefetch off): cold
#        first-render with the cache populating as decoding happens.
#   DCP  Full GPU + image cache + prefetch (cache feature on, --prefetch):
#        OCR-pipeline pattern.  Prefetch primes the cache at session open
#        before render starts.
#
# Disk tier is wiped before every mode so each run starts cold.  Within a
# mode, mpstat/iostat/hyperfine averages over RUNS=5 repeats per corpus
# (cold-cache between runs, posix_fadvise(FADV_DONTNEED)).
#
# CUDA_ARCH is read from the environment at build time.  Default sm_80 (the
# build script's fallback).  Override per-machine: sm_75 for Turing, sm_120
# for Blackwell, etc.  See README.md for the full table.
#
# Usage:
#   scripts/bench_v070.sh [--force]
#     --force   Re-run modes even if their <mode>.txt already exists, and
#               rebuild binaries even if already present.
set -euo pipefail

# ─── Configuration ────────────────────────────────────────────────────────────
OUT_DIR="bench/v070"
BENCH="tests/bench_corpus.sh"
NVJPEG2K_LIB="/usr/lib/x86_64-linux-gnu/libnvjpeg2k/12"
CUDA_LIB="/usr/local/cuda-12.8/lib64"
MIN_FREE_GB=20

# Mode → (binary suffix, --backend value, extra args).
MODES=(A D DC DCP)
declare -A MODE_BIN=(
  [A]=nvjpeg [D]=full-nocache [DC]=full-cache [DCP]=full-cache
)
declare -A MODE_BACKEND=(
  [A]=cpu [D]=cuda [DC]=cuda [DCP]=cuda
)
declare -A MODE_EXTRA=(
  [A]="" [D]="" [DC]="" [DCP]="--prefetch"
)
declare -A MODE_LABEL=(
  [A]="CPU-only"
  [D]="Full GPU (no cache)"
  [DC]="Full GPU + cache"
  [DCP]="Full GPU + cache + prefetch"
)
declare -A BIN_FEATURES=(
  [nvjpeg]="nvjpeg,nvjpeg2k"
  [full-nocache]="nvjpeg,nvjpeg2k,gpu-aa,gpu-icc"
  [full-cache]="nvjpeg,nvjpeg2k,gpu-aa,gpu-icc,cache"
)

FORCE=0
if [[ ${1:-} == "--force" ]]; then FORCE=1; fi

SKIP_NVJPEG=0

# ─── Logging ──────────────────────────────────────────────────────────────────
log()  { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
fail() { printf '[%s] FAIL: %s\n' "$(date +%H:%M:%S)" "$*" >&2; exit 1; }
warn() { printf '[%s] WARN: %s\n' "$(date +%H:%M:%S)" "$*" >&2; }

# ─── Pre-flight ───────────────────────────────────────────────────────────────
preflight() {
  log "Pre-flight"

  local free_gb
  free_gb=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
  if (( free_gb < MIN_FREE_GB )); then
    fail "free disk ${free_gb}G < ${MIN_FREE_GB}G required"
  fi
  log "  disk: ${free_gb}G free"

  for cmd in cargo hyperfine mpstat iostat python3 awk; do
    command -v "$cmd" >/dev/null || fail "$cmd not in PATH"
  done
  log "  tools: cargo, hyperfine, mpstat, iostat, python3, awk all present"

  [[ -x "$BENCH" ]] || fail "$BENCH not executable"
  log "  bench script: $BENCH"

  log "  CUDA_ARCH: ${CUDA_ARCH:-sm_80 (build-script default)}"

  if [[ ! -e /dev/nvidia0 ]]; then
    warn "/dev/nvidia0 missing; modes D/DC/DCP will be skipped"
    SKIP_NVJPEG=1
  elif [[ ! -e "$CUDA_LIB/libcudart.so" || ! -e "$CUDA_LIB/libnvjpeg.so" ]]; then
    warn "CUDA libs missing under $CUDA_LIB; modes D/DC/DCP will be skipped"
    SKIP_NVJPEG=1
  elif [[ ! -e "$NVJPEG2K_LIB/libnvjpeg2k.so" ]]; then
    warn "libnvjpeg2k.so missing under $NVJPEG2K_LIB; modes D/DC/DCP will be skipped"
    SKIP_NVJPEG=1
  else
    log "  nvidia: /dev/nvidia0 + cudart + nvjpeg + nvjpeg2k present"
  fi

  mkdir -p "$OUT_DIR"
  log "Pre-flight OK"
}

# ─── Build ────────────────────────────────────────────────────────────────────
build_binary() {
  local suffix="$1"
  local features="${BIN_FEATURES[$suffix]}"
  local out="target/release/pdf-raster-$suffix"

  if [[ -x "$out" && $FORCE -eq 0 ]]; then
    log "build[$suffix]: $out already present"
    return 0
  fi

  log "build[$suffix]: features='$features'"
  RUSTFLAGS="-C target-cpu=native" cargo build --release -p pdf-raster --features "$features"
  cp -f target/release/pdf-raster "$out"
  log "build[$suffix]: copied to $out"
}

builds() {
  log "Build phase"
  if [[ $SKIP_NVJPEG -eq 1 ]]; then
    BIN_FEATURES[nvjpeg]=""
    warn "nvjpeg binary will be built without CUDA features (CUDA unavailable)"
  fi
  build_binary nvjpeg

  if [[ $SKIP_NVJPEG -eq 0 ]]; then
    build_binary full-nocache
    build_binary full-cache
  else
    log "build[full-nocache]: SKIP (CUDA unavailable)"
    log "build[full-cache]:   SKIP (CUDA unavailable)"
  fi
  log "Build phase OK"
}

# ─── Disk-cache wipe ──────────────────────────────────────────────────────────
# DC and DCP modes use the disk tier of the image cache.  Wipe before each
# mode so every run starts cold — measures the worst case and matches the
# v0.6.0 baseline conditions for mode A / D.
wipe_disk_cache() {
  local cache_root="${PDF_RASTER_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/pdf-raster}"
  if [[ -d "$cache_root" ]]; then
    rm -rf "$cache_root"
    log "  disk cache wiped: $cache_root"
  fi
}

# ─── Bench ────────────────────────────────────────────────────────────────────
bench_mode() {
  local mode="$1"
  local bin_suffix="${MODE_BIN[$mode]}"
  local backend="${MODE_BACKEND[$mode]}"
  local extra="${MODE_EXTRA[$mode]}"
  local label="${MODE_LABEL[$mode]}"
  local bin="target/release/pdf-raster-$bin_suffix"
  local out="$OUT_DIR/$mode.txt"

  if [[ -f "$out" && $FORCE -eq 0 ]]; then
    log "bench[$mode/$label]: SKIP (already done — $out)"
    return 0
  fi

  if [[ ! -x "$bin" ]]; then
    log "bench[$mode/$label]: SKIP (binary $bin missing)"
    return 0
  fi

  if [[ "$backend" == "cuda" && $SKIP_NVJPEG -eq 1 ]]; then
    log "bench[$mode/$label]: SKIP (CUDA unavailable)"
    return 0
  fi

  log "bench[$mode/$label]: starting (BIN=$bin --backend $backend $extra)"
  wipe_disk_cache

  local ld_path=""
  if [[ "$bin_suffix" =~ ^(nvjpeg|full-nocache|full-cache)$ && "$bin_suffix" != "" ]]; then
    if [[ -d "$NVJPEG2K_LIB" || -d "$CUDA_LIB" ]]; then
      ld_path="LD_LIBRARY_PATH=$NVJPEG2K_LIB:$CUDA_LIB:${LD_LIBRARY_PATH:-}"
    fi
  fi

  # Pass extra flags via PDF_RASTER_ARGS (read by tests/bench_corpus.sh's
  # internal hyperfine command — see how --backend is propagated).
  # The bench script accepts --backend natively; --prefetch is custom so
  # it goes through the env-var override that bench_corpus.sh inherits.
  # shellcheck disable=SC2086
  if [[ -n "$extra" ]]; then
    env $ld_path BIN="$bin" PDF_RASTER_EXTRA_ARGS="$extra" \
      "$BENCH" --backend "$backend" \
      | tee "$out"
  else
    # shellcheck disable=SC2086
    env $ld_path BIN="$bin" \
      "$BENCH" --backend "$backend" \
      | tee "$out"
  fi
  log "bench[$mode/$label]: done — $out"
}

benches() {
  log "Bench phase"
  for mode in "${MODES[@]}"; do
    bench_mode "$mode"
  done
  log "Bench phase OK"
}

# ─── Aggregate ────────────────────────────────────────────────────────────────
aggregate() {
  log "Aggregating to $OUT_DIR/results.md"
  python3 scripts/aggregate_v070.py "$OUT_DIR" > "$OUT_DIR/results.md"
  log "Wrote $OUT_DIR/results.md"
}

# ─── Main ─────────────────────────────────────────────────────────────────────
preflight
builds
benches
aggregate
log "All done. Matrix: $OUT_DIR/results.md"
