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

# ─── Main ─────────────────────────────────────────────────────────────────────
preflight
log "Pre-flight only mode (build/bench phases not yet implemented)"
