#!/usr/bin/env bash
# Phase 10 bench gate — A/D/V on vector-heavy corpora 01-05.
#
# Modes:
#   A   CPU-only baseline (backend=cpu, no GPU features).
#   D   CUDA full GPU (backend=cuda; nvjpeg + gpu-aa + gpu-icc).
#   V   Vulkan compute backend (backend=vulkan; vulkan + gpu-aa).
#
# Why corpora 01-05 only: the Vulkan binary doesn't include nvjpeg, so
# DCT-heavy corpora (06-10) compare Vulkan-CPU-decode vs CUDA-nvjpeg —
# not a fair comparison for the AA / tile-fill kernels that Phase 10
# actually migrated.  Corpora 01-05 are vector-/text-heavy where the
# fill kernels carry the work.
#
# Phase 10 bench-gate criteria (per ROADMAP.md):
#   1. CUDA path performance unchanged within ±5% vs bench/v070/D.txt.
#   2. Vulkan pixel-diff ≤ 1 LSB vs CUDA — verified separately.
#   3. Vulkan timing within 15% of CUDA on RTX 5070.
#
# Usage:
#   scripts/bench_v10.sh [--force]
#     --force   Re-run modes even if their <mode>.txt already exists,
#               and rebuild binaries even if already present.

set -euo pipefail

OUT_DIR="bench/v10"
BENCH="tests/bench_corpus.sh"
MIN_FREE_GB=20

# CUDA libs.  Mirrors bench_v070.sh.
probe_path() {
  local var="$1"; shift
  if [[ -n "${!var:-}" ]]; then return; fi
  for cand in "$@"; do
    if [[ -e "$cand" ]]; then
      eval "$var=\"\$cand\""
      return
    fi
  done
}
probe_path CUDA_LIB \
  "/usr/local/cuda/lib64" \
  "/usr/local/cuda-13.2/lib64" \
  "/usr/local/cuda-13/lib64" \
  "/usr/local/cuda-12.8/lib64" \
  "/usr/lib/x86_64-linux-gnu"

FORCE=0
[[ ${1:-} == "--force" ]] && FORCE=1

log()  { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
fail() { printf '[%s] FAIL: %s\n' "$(date +%H:%M:%S)" "$*" >&2; exit 1; }
warn() { printf '[%s] WARN: %s\n' "$(date +%H:%M:%S)" "$*" >&2; }

# ─── Pre-flight ───────────────────────────────────────────────────────────────
preflight() {
  log "Pre-flight"
  local free_gb
  free_gb=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
  (( free_gb >= MIN_FREE_GB )) || fail "free disk ${free_gb}G < ${MIN_FREE_GB}G"
  log "  disk: ${free_gb}G free"

  for cmd in cargo hyperfine mpstat iostat python3 awk; do
    command -v "$cmd" >/dev/null || fail "$cmd not in PATH"
  done

  [[ -x "$BENCH" ]] || fail "$BENCH not executable"
  log "  CUDA_ARCH: ${CUDA_ARCH:-sm_80 (build-script default)}"

  if [[ ! -e /dev/nvidia0 ]]; then
    warn "/dev/nvidia0 missing; mode D will be skipped"
    SKIP_CUDA=1
  else
    SKIP_CUDA=0
    log "  nvidia: /dev/nvidia0 present"
  fi

  mkdir -p "$OUT_DIR"
  log "Pre-flight OK"
}

# ─── Build the three binaries ─────────────────────────────────────────────────
build_one() {
  local suffix="$1" features="$2"
  local out="target/release/pdf-raster-$suffix"
  if [[ -x "$out" && $FORCE -eq 0 ]]; then
    log "build[$suffix]: $out already present"
    return 0
  fi
  log "build[$suffix]: features='$features'"
  RUSTFLAGS="-C target-cpu=native" \
    cargo build --release -p pdf-raster --features "$features"
  cp -f target/release/pdf-raster "$out"
  log "build[$suffix]: copied to $out"
}

builds() {
  log "Build phase"
  build_one cpu ""
  if [[ $SKIP_CUDA -eq 0 ]]; then
    build_one cuda  "nvjpeg,gpu-aa,gpu-icc"
  fi
  build_one vulkan "vulkan,gpu-aa"
  log "Build phase OK"
}

# ─── v0.7.0 baseline binary (live-capture) ───────────────────────────────────
# Builds the v0.7.0 binary in a worktree-local checkout so we can re-bench
# it under current driver/system state.  Otherwise the criterion-1
# comparison would conflate Phase 10 code drift with driver drift.
build_v070_baseline() {
  local out="target/release/pdf-raster-cuda-v070"
  if [[ -x "$out" && $FORCE -eq 0 ]]; then
    log "build[v070-baseline]: $out already present"
    return 0
  fi
  if [[ $SKIP_CUDA -eq 1 ]]; then
    log "build[v070-baseline]: SKIP (CUDA unavailable)"
    return 0
  fi
  if ! git rev-parse --verify --quiet v0.7.0 >/dev/null; then
    warn "v0.7.0 tag not found; criterion-1 baseline will be unavailable"
    return 0
  fi
  log "build[v070-baseline]: checking out v0.7.0 source"
  # Stash any pending changes, snap source to v0.7.0, build, restore.
  local stash_made=0
  if ! git diff --quiet HEAD -- crates/ Cargo.lock; then
    git stash push -q -m "bench_v10 v070 baseline" -- crates/ Cargo.lock || true
    stash_made=1
  fi
  git checkout -q v0.7.0 -- crates/ Cargo.lock
  RUSTFLAGS="-C target-cpu=native" \
    cargo build --release -p pdf-raster --features "nvjpeg,gpu-aa,gpu-icc" >/dev/null
  cp -f target/release/pdf-raster "$out"
  git checkout -q HEAD -- crates/ Cargo.lock
  if [[ $stash_made -eq 1 ]]; then
    git stash pop -q || warn "stash pop after v070 baseline build failed; check git status"
  fi
  log "build[v070-baseline]: copied to $out"
}

# ─── v0.7.0 baseline bench (live-capture) ────────────────────────────────────
bench_v070_baseline() {
  local bin="target/release/pdf-raster-cuda-v070"
  local out="$OUT_DIR/D-v070-baseline.txt"
  if [[ -f "$out" && $FORCE -eq 0 ]]; then
    log "bench[v070-baseline]: SKIP (already done — $out)"
    return 0
  fi
  if [[ ! -x "$bin" ]]; then
    log "bench[v070-baseline]: SKIP (binary $bin missing)"
    return 0
  fi
  local ld_path=""
  [[ -d "$CUDA_LIB" ]] && ld_path="LD_LIBRARY_PATH=$CUDA_LIB:${LD_LIBRARY_PATH:-}"
  log "bench[v070-baseline]: BIN=$bin --backend cuda"
  # shellcheck disable=SC2086
  env $ld_path BIN="$bin" \
    "$BENCH" --backend cuda --corpus-dir "$SUBSET_DIR" \
    | tee "$out"
  log "bench[v070-baseline]: done — $out"
}

# ─── Stage a corpus subset (01-05) as symlinks ────────────────────────────────
stage_subset() {
  SUBSET_DIR="$(mktemp -d)"
  for n in 01 02 03 04 05; do
    local src
    src=$(ls tests/fixtures/corpus-${n}-*.pdf 2>/dev/null | head -1)
    [[ -n "$src" ]] || fail "fixture corpus-${n}-*.pdf not found"
    ln -s "$(realpath "$src")" "$SUBSET_DIR/$(basename "$src")"
  done
  log "Subset staged at $SUBSET_DIR ($(ls "$SUBSET_DIR" | wc -l) files)"
}

# ─── Run one mode ─────────────────────────────────────────────────────────────
bench_mode() {
  local mode="$1" suffix="$2" backend="$3"
  local bin="target/release/pdf-raster-$suffix"
  local out="$OUT_DIR/$mode.txt"

  if [[ -f "$out" && $FORCE -eq 0 ]]; then
    log "bench[$mode]: SKIP (already done — $out)"
    return 0
  fi
  if [[ ! -x "$bin" ]]; then
    log "bench[$mode]: SKIP (binary $bin missing)"
    return 0
  fi
  if [[ "$backend" == "cuda" && $SKIP_CUDA -eq 1 ]]; then
    log "bench[$mode]: SKIP (CUDA unavailable)"
    return 0
  fi

  local ld_path=""
  [[ -d "$CUDA_LIB" ]] && ld_path="LD_LIBRARY_PATH=$CUDA_LIB:${LD_LIBRARY_PATH:-}"

  log "bench[$mode]: BIN=$bin --backend $backend (corpus=$SUBSET_DIR)"
  # shellcheck disable=SC2086
  env $ld_path BIN="$bin" \
    "$BENCH" --backend "$backend" --corpus-dir "$SUBSET_DIR" \
    | tee "$out"
  log "bench[$mode]: done — $out"
}

benches() {
  log "Bench phase"
  bench_v070_baseline
  bench_mode A cpu    cpu
  bench_mode D cuda   cuda
  bench_mode V vulkan vulkan
  log "Bench phase OK"
}

# ─── Aggregate to results.md ──────────────────────────────────────────────────
aggregate() {
  log "Aggregating to $OUT_DIR/results.md"
  python3 scripts/aggregate_v10.py "$OUT_DIR" > "$OUT_DIR/results.md"
  log "Wrote $OUT_DIR/results.md"
}

# ─── Main ─────────────────────────────────────────────────────────────────────
preflight
builds
build_v070_baseline
stage_subset
benches
aggregate
log "Phase 10 bench gate complete: $OUT_DIR/results.md"
# Cleanup temp subset dir.
rm -rf "$SUBSET_DIR"
