#!/usr/bin/env bash
# v0.6.0 GPU baseline benchmark driver.
#
# Builds 4 binaries (CPU, VA-API, nvJPEG, full GPU) and runs hyperfine
# against the 10-corpus fixture set. Output: bench/v060/<corpus>-<mode>.json
# plus an aggregated bench/v060/results.md.
#
# Usage: scripts/bench_v060.sh [--force]
#   --force   Re-run cells even if their JSON already exists.
set -euo pipefail

# ─── Configuration ────────────────────────────────────────────────────────────
DPI=150
WARMUP=3
RUNS=8
OUT_DIR="bench/v060"
FIXTURES="tests/fixtures"
NVJPEG2K_LIB="/usr/lib/x86_64-linux-gnu/libnvjpeg2k/12"
CUDA_LIB="/usr/local/cuda-12.8/lib64"
MIN_FREE_GB=20

CORPORA=(
  corpus-01-native-text-small
  corpus-02-native-vector-text
  corpus-03-native-text-dense
  corpus-04-ebook-mixed
  corpus-05-academic-book
  corpus-06-modern-layout-dct
  corpus-07-journal-dct-heavy
  corpus-08-scan-dct-1927
  corpus-09-scan-dct-1836
  corpus-10-scan-jbig2-jpx
)

MODES=(cpu vaapi nvjpeg full)
declare -A MODE_FEATURES=(
  [cpu]=""
  [vaapi]="vaapi"
  [nvjpeg]="nvjpeg,nvjpeg2k"
  [full]="nvjpeg,nvjpeg2k,gpu-aa,gpu-icc"
)

FORCE=0
if [[ ${1:-} == "--force" ]]; then FORCE=1; fi

SKIP_NVJPEG=0
SKIP_VAAPI=0

# ─── Pre-flight ───────────────────────────────────────────────────────────────
log()  { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
fail() { printf '[%s] FAIL: %s\n' "$(date +%H:%M:%S)" "$*" >&2; exit 1; }
warn() { printf '[%s] WARN: %s\n' "$(date +%H:%M:%S)" "$*" >&2; }

preflight() {
  log "Pre-flight checks"

  # Disk
  local free_gb
  free_gb=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
  if (( free_gb < MIN_FREE_GB )); then
    fail "free disk ${free_gb}G < ${MIN_FREE_GB}G required"
  fi
  log "  disk: ${free_gb}G free"

  # Tools
  command -v hyperfine >/dev/null || fail "hyperfine not in PATH"
  command -v jq >/dev/null || fail "jq not in PATH"
  command -v cargo >/dev/null || fail "cargo not in PATH"
  log "  tools: hyperfine $(hyperfine --version | awk '{print $2}'), jq $(jq --version)"

  # Fixtures
  for c in "${CORPORA[@]}"; do
    [[ -f "$FIXTURES/$c.pdf" ]] || fail "missing fixture: $FIXTURES/$c.pdf"
  done
  log "  fixtures: 10/10 present"

  # GPU detection (warning, not fatal — the script falls back to CPU-only).
  # We probe the runtime requirements directly rather than relying on
  # nvidia-smi (which may not be installed). nvJPEG needs:
  #   - /dev/nvidia0 (kernel module + GPU device node)
  #   - libcudart.so + libnvjpeg.so under $CUDA_LIB
  #   - libnvjpeg2k.so under $NVJPEG2K_LIB
  if [[ ! -e /dev/nvidia0 ]]; then
    warn "/dev/nvidia0 missing; nvjpeg/full modes will be skipped"
    SKIP_NVJPEG=1
  elif [[ ! -e "$CUDA_LIB/libcudart.so" || ! -e "$CUDA_LIB/libnvjpeg.so" ]]; then
    warn "CUDA libs missing under $CUDA_LIB; nvjpeg/full modes will be skipped"
    SKIP_NVJPEG=1
  elif [[ ! -e "$NVJPEG2K_LIB/libnvjpeg2k.so" ]]; then
    warn "libnvjpeg2k.so missing under $NVJPEG2K_LIB; nvjpeg/full modes will be skipped"
    SKIP_NVJPEG=1
  else
    log "  nvidia: /dev/nvidia0 + cudart + nvjpeg + nvjpeg2k present"
  fi

  # VA-API detection (warning, not fatal). We probe the runtime requirements
  # directly rather than running vainfo (which is known to crash on some
  # driver/kernel combos even when libva itself is healthy).
  if [[ ! -e /dev/dri/renderD128 ]]; then
    warn "/dev/dri/renderD128 missing; vaapi mode will be skipped"
    SKIP_VAAPI=1
  elif [[ ! -e /usr/lib/x86_64-linux-gnu/libva.so.2 || ! -e /usr/lib/x86_64-linux-gnu/libva-drm.so.2 ]]; then
    warn "libva.so.2/libva-drm.so.2 missing; vaapi mode will be skipped"
    SKIP_VAAPI=1
  else
    log "  vaapi: /dev/dri/renderD128 + libva.so.2 + libva-drm.so.2 present"
  fi

  # sudo cache for drop_caches. We require the cache to be primed before
  # running the script (interactive prompting mid-run would stall the bench
  # unpredictably). Once primed, a background keepalive loop refreshes the
  # cache every 60 s for the duration of the run — needed because the full
  # matrix takes longer than the default 15-minute sudo cache window.
  if ! sudo -n true 2>/dev/null; then
    fail "sudo credential cache not primed; run 'sudo -v' first, then re-run this script"
  fi
  log "  sudo: cached"
  # Keepalive: refresh sudo every 60s in the background; killed via trap on exit.
  ( while true; do sudo -n true 2>/dev/null || exit; sleep 60; done ) &
  SUDO_KEEPALIVE_PID=$!
  trap 'kill "$SUDO_KEEPALIVE_PID" 2>/dev/null || true' EXIT
  log "  sudo: keepalive loop pid=$SUDO_KEEPALIVE_PID"

  mkdir -p "$OUT_DIR"
  log "Pre-flight OK"
}

# ─── Build phase ──────────────────────────────────────────────────────────────
# For each mode, build pdf-raster with that mode's feature set and copy the
# resulting binary to target/release/pdf-raster-<mode>. Skips a build if the
# named binary already exists and --force was not passed.
build_one() {
  local mode="$1"
  local features="${MODE_FEATURES[$mode]}"
  local out="target/release/pdf-raster-$mode"

  if [[ -x "$out" && $FORCE -eq 0 ]]; then
    log "build[$mode]: $out already present (use --force to rebuild)"
    return 0
  fi

  log "build[$mode]: features='$features'"
  local cargo_args=(build --release -p pdf-raster)
  if [[ -n "$features" ]]; then
    cargo_args+=(--features "$features")
  fi

  RUSTFLAGS="-C target-cpu=native" cargo "${cargo_args[@]}"
  cp -f target/release/pdf-raster "$out"
  log "build[$mode]: copied to $out"
}

builds() {
  log "Build phase"
  for mode in "${MODES[@]}"; do
    if [[ "$mode" == "vaapi" && $SKIP_VAAPI -eq 1 ]]; then
      log "build[$mode]: SKIP (VA-API unavailable)"
      continue
    fi
    if [[ ("$mode" == "nvjpeg" || "$mode" == "full") && $SKIP_NVJPEG -eq 1 ]]; then
      log "build[$mode]: SKIP (NVIDIA GPU unavailable)"
      continue
    fi
    build_one "$mode"
  done
  log "Build phase OK"
}

# ─── Bench phase ──────────────────────────────────────────────────────────────
# Drop kernel caches so the file read is genuinely cold. Hyperfine warmup
# runs go on top of this — they'll be hot, which is the same protocol as v0.5.1.
drop_caches() {
  sync
  sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
}

# Run hyperfine for one (corpus, mode) cell. Idempotent: skips if the JSON
# already exists and --force was not passed.
bench_one() {
  local corpus="$1"
  local mode="$2"
  local pdf="$FIXTURES/$corpus.pdf"
  local bin="target/release/pdf-raster-$mode"
  local json="$OUT_DIR/$corpus-$mode.json"

  if [[ -f "$json" && $FORCE -eq 0 ]]; then
    log "bench[$corpus/$mode]: SKIP (already done)"
    return 0
  fi

  if [[ ! -x "$bin" ]]; then
    log "bench[$corpus/$mode]: SKIP (binary missing)"
    return 0
  fi

  drop_caches
  log "bench[$corpus/$mode]: starting"

  # Bare-stem prefix routes pdf-raster output to /dev/shm (RAM) by default
  # in v0.6.0, which is the protocol we want to measure.
  local out_prefix="v060-out-$$-$corpus"

  local extra_env=""
  if [[ "$mode" == "nvjpeg" || "$mode" == "full" ]]; then
    extra_env="LD_LIBRARY_PATH=$NVJPEG2K_LIB:$CUDA_LIB:\${LD_LIBRARY_PATH:-}"
  fi

  hyperfine \
    --warmup "$WARMUP" \
    --runs "$RUNS" \
    --export-json "$json" \
    --shell=bash \
    "$extra_env $bin -r $DPI $pdf $out_prefix"

  # Clean up the per-cell output directory in /dev/shm to keep RAM tidy.
  # (pdf-raster routes bare-stem prefix to /dev/shm/pdf-raster-<pid>-<nanos>/.)
  rm -rf "/dev/shm/pdf-raster-"* 2>/dev/null || true

  log "bench[$corpus/$mode]: done — $(jq -r '.results[0].mean * 1000 | floor' "$json")ms mean"
}

benches() {
  log "Bench phase: 10 corpora × 4 modes = 40 cells"
  for corpus in "${CORPORA[@]}"; do
    for mode in "${MODES[@]}"; do
      # Mode-skip gates mirror the build phase so we never try to bench a
      # binary that wasn't built.
      if [[ "$mode" == "vaapi" && $SKIP_VAAPI -eq 1 ]]; then continue; fi
      if [[ ("$mode" == "nvjpeg" || "$mode" == "full") && $SKIP_NVJPEG -eq 1 ]]; then continue; fi
      bench_one "$corpus" "$mode"
    done
  done
  log "Bench phase OK"
}

# ─── Main ─────────────────────────────────────────────────────────────────────
preflight
builds
benches
log "Aggregating results"
./scripts/aggregate_v060.sh "$OUT_DIR" "$OUT_DIR/results.md"
log "All done. Results: $OUT_DIR/results.md"
