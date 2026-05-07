#!/usr/bin/env bash
# v0.6.0 GPU baseline matrix — 10-corpus × 4-backend wrapper.
#
# Builds two binaries (nvjpeg covers CPU/VA-API/CUDA-decode-only modes via
# runtime --backend; full adds GPU AA + GPU ICC kernels) then drives the
# existing tests/bench_corpus.sh four times to populate the matrix.
#
# Per-corpus measurement (hyperfine + mpstat + iostat + cache eviction)
# lives in tests/bench_corpus.sh and is reused unchanged.
#
# Usage: scripts/bench_v060.sh [--force]
#   --force   Re-run modes even if their <mode>.txt already exists, and
#             rebuild binaries even if already present.
set -euo pipefail

# ─── Configuration ────────────────────────────────────────────────────────────
OUT_DIR="bench/v060"
BENCH="tests/bench_corpus.sh"
NVJPEG2K_LIB="/usr/lib/x86_64-linux-gnu/libnvjpeg2k/12"
CUDA_LIB="/usr/local/cuda-12.8/lib64"
MIN_FREE_GB=20

# Mode → (binary suffix, --backend value, cargo features)
# The same nvjpeg binary is used for modes A, B, C; full is used for D.
MODES=(A B C D)
declare -A MODE_BIN=(
  [A]=nvjpeg [B]=nvjpeg [C]=nvjpeg [D]=full
)
declare -A MODE_BACKEND=(
  [A]=cpu [B]=vaapi [C]=cuda [D]=cuda
)
declare -A MODE_LABEL=(
  [A]="CPU-only" [B]="VA-API" [C]="nvJPEG only" [D]="Full GPU"
)
declare -A BIN_FEATURES=(
  [nvjpeg]="nvjpeg,nvjpeg2k,vaapi"
  [full]="nvjpeg,nvjpeg2k,vaapi,gpu-aa,gpu-icc"
)

FORCE=0
if [[ ${1:-} == "--force" ]]; then FORCE=1; fi

SKIP_NVJPEG=0
SKIP_VAAPI=0

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

  # GPU detection — file/device probes (nvidia-smi may not be installed).
  if [[ ! -e /dev/nvidia0 ]]; then
    warn "/dev/nvidia0 missing; modes C and D will be skipped"
    SKIP_NVJPEG=1
  elif [[ ! -e "$CUDA_LIB/libcudart.so" || ! -e "$CUDA_LIB/libnvjpeg.so" ]]; then
    warn "CUDA libs missing under $CUDA_LIB; modes C and D will be skipped"
    SKIP_NVJPEG=1
  elif [[ ! -e "$NVJPEG2K_LIB/libnvjpeg2k.so" ]]; then
    warn "libnvjpeg2k.so missing under $NVJPEG2K_LIB; modes C and D will be skipped"
    SKIP_NVJPEG=1
  else
    log "  nvidia: /dev/nvidia0 + cudart + nvjpeg + nvjpeg2k present"
  fi

  if [[ ! -e /dev/dri/renderD128 ]]; then
    warn "/dev/dri/renderD128 missing; mode B will be skipped"
    SKIP_VAAPI=1
  elif [[ ! -e /usr/lib/x86_64-linux-gnu/libva.so.2 ]]; then
    warn "libva.so.2 missing; mode B will be skipped"
    SKIP_VAAPI=1
  else
    log "  vaapi: /dev/dri/renderD128 + libva.so.2 present"
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
  # Always need nvjpeg binary — covers CPU/VA-API even when CUDA absent
  # because runtime --backend cpu and vaapi don't touch CUDA at runtime.
  if [[ $SKIP_NVJPEG -eq 1 ]]; then
    # Fall back to a vaapi-only binary if CUDA is unavailable. The features
    # in MODE_FEATURES include nvjpeg, which links nvjpeg.so — that would
    # fail at link time without CUDA. Strip it.
    BIN_FEATURES[nvjpeg]="vaapi"
    warn "nvjpeg binary will be built without CUDA features (CUDA unavailable)"
  fi
  build_binary nvjpeg

  if [[ $SKIP_NVJPEG -eq 0 ]]; then
    build_binary full
  else
    log "build[full]: SKIP (CUDA unavailable)"
  fi
  log "Build phase OK"
}

# ─── Bench ────────────────────────────────────────────────────────────────────
bench_mode() {
  local mode="$1"
  local bin_suffix="${MODE_BIN[$mode]}"
  local backend="${MODE_BACKEND[$mode]}"
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

  if [[ "$backend" == "vaapi" && $SKIP_VAAPI -eq 1 ]]; then
    log "bench[$mode/$label]: SKIP (VA-API unavailable)"
    return 0
  fi
  if [[ "$backend" == "cuda" && $SKIP_NVJPEG -eq 1 ]]; then
    log "bench[$mode/$label]: SKIP (CUDA unavailable)"
    return 0
  fi

  log "bench[$mode/$label]: starting (BIN=$bin --backend $backend)"
  # nvjpeg2k library is in a non-standard location; preload it for cuda mode.
  local ld_path=""
  if [[ "$backend" == "cuda" ]]; then
    ld_path="LD_LIBRARY_PATH=$NVJPEG2K_LIB:$CUDA_LIB:${LD_LIBRARY_PATH:-}"
  fi

  # Run the existing bench_corpus.sh; capture stdout to <mode>.txt.
  # ld_path is "LD_LIBRARY_PATH=..." or empty — env needs it as a separate
  # KEY=VAL token, so deliberately unquoted (shellcheck disable=SC2086).
  # shellcheck disable=SC2086
  env $ld_path BIN="$bin" "$BENCH" --backend "$backend" \
    | tee "$out"
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
# Each mode's .txt has rows like:
#   01-native-text-small             49ms ±    1ms       95%      99%        12 MB/s  [low-cpu...]
# We pivot into:
#   | corpus | A. CPU-only | B. VA-API | C. nvJPEG | D. Full GPU | flags |
aggregate() {
  log "Aggregating to $OUT_DIR/results.md"
  python3 scripts/aggregate_v060.py "$OUT_DIR" > "$OUT_DIR/results.md"
  log "Wrote $OUT_DIR/results.md"
}

# ─── Main ─────────────────────────────────────────────────────────────────────
preflight
builds
benches
aggregate
log "All done. Matrix: $OUT_DIR/results.md"
