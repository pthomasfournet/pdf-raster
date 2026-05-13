#!/usr/bin/env bash
# Parallel-Huffman JPEG dispatch benchmark — five-mode corpus run.
#
# Modes:
#   A       CPU only (regression baseline).
#   C-std   CUDA standard pipeline (nvJPEG + gpu-aa + gpu-icc).
#   C-huff  CUDA parallel-Huffman JPEG (gpu-jpeg-huffman + gpu-aa + gpu-icc,
#           threshold=0 so every eligible JPEG goes through the GPU path).
#   V-std   Vulkan, Huffman dormant (threshold=u32::MAX; JPEGs via zune-jpeg CPU).
#   V-huff  Vulkan parallel-Huffman JPEG (threshold=0).
#
# V-std vs V-huff isolates Huffman dispatch cost on Vulkan.
# C-huff vs C-std isolates it on CUDA.
# V-huff vs C-huff compares Vulkan vs CUDA Huffman performance.
#
# Usage:
#   scripts/bench_v091.sh [--force]
set -euo pipefail

OUT_DIR="bench/v091"
BENCH="tests/bench_corpus.sh"
MIN_FREE_GB=20

# Threshold values for PDF_RASTER_HUFFMAN_THRESHOLD env var.
THRESHOLD_ALWAYS=0          # every eligible JPEG dispatched to GPU
THRESHOLD_NEVER=4294967295  # u32::MAX — Huffman path dormant

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

# Mode order: A first for load-free baseline; then GPU modes.
MODES=(A C-std C-huff V-std V-huff)

declare -A MODE_BIN=(
  [A]=cpu
  [C-std]=full-gpu
  [C-huff]=cuda-huffman
  [V-std]=vulkan-huffman
  [V-huff]=vulkan-huffman
)
declare -A MODE_BACKEND=(
  [A]=cpu
  [C-std]=cuda
  [C-huff]=cuda
  [V-std]=vulkan
  [V-huff]=vulkan
)
declare -A MODE_FEATURES=(
  [A]=""
  [C-std]="nvjpeg,gpu-aa,gpu-icc"
  [C-huff]="gpu-jpeg-huffman,gpu-aa,gpu-icc"
  [V-std]="vulkan"
  [V-huff]="vulkan"
)
declare -A MODE_HUFF_THRESHOLD=(
  [A]=""
  [C-std]=""
  [C-huff]="$THRESHOLD_ALWAYS"
  [V-std]="$THRESHOLD_NEVER"
  [V-huff]="$THRESHOLD_ALWAYS"
)
declare -A MODE_LABEL=(
  [A]="CPU-only"
  [C-std]="CUDA + nvJPEG"
  [C-huff]="CUDA parallel-Huffman"
  [V-std]="Vulkan (CPU JPEG)"
  [V-huff]="Vulkan parallel-Huffman"
)

FORCE=0
if [[ ${1:-} == "--force" ]]; then FORCE=1; fi
SKIP_CUDA=0
SKIP_VULKAN=0

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
    warn "/dev/nvidia0 missing; CUDA modes will be skipped"
    SKIP_CUDA=1; SKIP_VULKAN=1
  elif [[ ! -e "$CUDA_LIB/libcudart.so" || ! -e "$CUDA_LIB/libnvjpeg.so" ]]; then
    warn "CUDA libs missing under $CUDA_LIB; C-std mode will be skipped"
    SKIP_CUDA=1
  else
    log "  nvidia: /dev/nvidia0 + cudart + nvjpeg present"
  fi

  if [[ $SKIP_VULKAN -eq 0 ]]; then
    # Use grep -c to avoid SIGPIPE under pipefail (grep -q exits immediately,
    # causing vulkaninfo to get SIGPIPE; pipefail then sees exit 141 not 0).
    local vk_detected=0
    if command -v vulkaninfo >/dev/null 2>&1; then
      local vk_out
      vk_out=$(vulkaninfo 2>/dev/null) && \
        echo "$vk_out" | grep -c "Vulkan Instance" >/dev/null 2>&1 && \
        vk_detected=1 || true
    fi
    if [[ $vk_detected -eq 0 ]]; then
      warn "Vulkan ICD not found; V-std and V-huff modes will be skipped"
      SKIP_VULKAN=1
    else
      log "  vulkan: ICD present"
    fi
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
    cargo build --release -p pdf-raster ${features:+--features "$features"}
  cp -f target/release/pdf-raster "$out"
  log "build[$suffix]: copied to $out"
}

builds() {
  log "Build phase"
  build_binary cpu ""
  if [[ $SKIP_CUDA -eq 0 ]]; then
    build_binary full-gpu    "nvjpeg,gpu-aa,gpu-icc"
    build_binary cuda-huffman "gpu-jpeg-huffman,gpu-aa,gpu-icc"
  else
    log "build[full-gpu]: SKIP (CUDA unavailable)"
    log "build[cuda-huffman]: SKIP (CUDA unavailable)"
  fi
  if [[ $SKIP_VULKAN -eq 0 ]]; then
    # V-std and V-huff share the same binary; build once.
    build_binary vulkan-huffman "vulkan"
  else
    log "build[vulkan-huffman]: SKIP (Vulkan unavailable)"
  fi
  log "Build phase OK"
}

bench_mode() {
  local mode="$1"
  local bin_suffix="${MODE_BIN[$mode]}"
  local backend="${MODE_BACKEND[$mode]}"
  local label="${MODE_LABEL[$mode]}"
  local threshold="${MODE_HUFF_THRESHOLD[$mode]}"
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
  if [[ "$backend" == "vulkan" && $SKIP_VULKAN -eq 1 ]]; then
    log "bench[$mode/$label]: SKIP (Vulkan unavailable)"; return 0
  fi

  log "bench[$mode/$label]: starting (PDF_RASTER_HUFFMAN_THRESHOLD=${threshold:-<unset>})"

  local ld_env=""
  if [[ "$backend" == "cuda" && -d "$CUDA_LIB" ]]; then
    ld_env="LD_LIBRARY_PATH=$CUDA_LIB:${LD_LIBRARY_PATH:-}"
  fi

  local huff_env=""
  if [[ -n "$threshold" ]]; then
    huff_env="PDF_RASTER_HUFFMAN_THRESHOLD=$threshold"
  fi

  # shellcheck disable=SC2086
  env ${ld_env:+$ld_env} ${huff_env:+$huff_env} BIN="$bin" \
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
  python3 scripts/aggregate_v091.py "$OUT_DIR" > "$OUT_DIR/results.md"
  log "Wrote $OUT_DIR/results.md"
}

preflight
builds
benches
aggregate
log "All done. Results: $OUT_DIR/results.md"
