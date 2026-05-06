#!/bin/bash
# Compare pdf-raster against pdftoppm across all corpus PDFs.
#
# Usage:
#   bench_compare.sh [--backend cpu|vaapi|cuda] [--vaapi-device /dev/dri/renderD129]
#                    [--runs N] [--warmup N] [--corpus-dir <path>]
#
# Reports: pdf-raster mean, pdftoppm mean, speedup ratio.
# This is intentionally separate from bench_corpus.sh — pdftoppm runs are
# single-threaded and slow (minutes on scan corpora), so they should only
# be run when an explicit comparison is needed.
#
# Requirements: hyperfine, pdftoppm, python3

set -euo pipefail

BIN="${BIN:-$(dirname "$0")/../target/release/pdf-raster}"
FIXTURES="$(dirname "$0")/fixtures"
BACKEND="cpu"
VAAPI_DEVICE="/dev/dri/renderD129"
RUNS=3
WARMUP=1

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend)      BACKEND="$2";      shift 2 ;;
    --vaapi-device) VAAPI_DEVICE="$2"; shift 2 ;;
    --corpus-dir)   FIXTURES="$2";     shift 2 ;;
    --runs)         RUNS="$2";         shift 2 ;;
    --warmup)       WARMUP="$2";       shift 2 ;;
    *) echo "ERROR: unknown argument: $1" >&2; exit 1 ;;
  esac
done

# ── Dependency checks ─────────────────────────────────────────────────────────
[[ -x "$BIN" ]]         || { echo "ERROR: binary not found: $BIN" >&2; exit 1; }
command -v hyperfine    >/dev/null || { echo "ERROR: hyperfine not found — cargo install hyperfine" >&2; exit 1; }
command -v pdftoppm     >/dev/null || { echo "ERROR: pdftoppm not found" >&2; exit 1; }

# ── Pre-flight: system load gate ─────────────────────────────────────────────
NCORES=$(nproc)
LOAD1=$(cut -d' ' -f1 /proc/loadavg)
LOAD_INT=$(echo "$LOAD1" | awk '{print int($1)}')
if [[ "$LOAD_INT" -gt $((NCORES * 2)) ]]; then
  echo "ERROR: load ($LOAD1) too high — threshold $((NCORES*2)). Wait for system to settle." >&2
  exit 1
fi

# ── Cache eviction ────────────────────────────────────────────────────────────
evict_file() {
  python3 -c "
import ctypes, os, sys
fd = os.open(sys.argv[1], os.O_RDONLY)
ctypes.CDLL(None).posix_fadvise(fd, 0, os.fstat(fd).st_size, 4)
os.close(fd)
" "$1" 2>/dev/null || true
}

# ── Disk space check ──────────────────────────────────────────────────────────
check_disk() {
  local avail_kb
  avail_kb=$(df --output=avail -k /tmp 2>/dev/null | tail -1)
  if [[ -z "$avail_kb" || "$avail_kb" -lt 1048576 ]]; then
    echo "ERROR: less than 1 GB free on /tmp — aborting" >&2; exit 1
  fi
}

# ── Hyperfine mean ────────────────────────────────────────────────────────────
# hyperfine_mean PREPARE_CMD BENCH_CMD  →  prints mean_ms
hyperfine_mean() {
  local prepare_cmd="$1" bench_cmd="$2" json_f
  json_f=$(mktemp --suffix=.json)
  hyperfine --runs "$RUNS" --warmup "$WARMUP" \
    --prepare "$prepare_cmd" \
    --export-json "$json_f" \
    -- "$bench_cmd" > /dev/null 2>&1
  python3 - "$json_f" <<'EOF'
import json, sys
with open(sys.argv[1]) as f:
    r = json.load(f)['results'][0]
print(f"{r['mean']*1000:.0f}")
EOF
  rm -f "$json_f"
}

# ── Argument list for pdf-raster ─────────────────────────────────────────────
PDF_RASTER_ARGS=(--backend "$BACKEND" -r 150)
[[ "$BACKEND" == "vaapi" ]] && PDF_RASTER_ARGS+=(--vaapi-device "$VAAPI_DEVICE")

# ── Corpus list ───────────────────────────────────────────────────────────────
corpora=(
  "01-native-text-small"
  "02-native-vector-text"
  "03-native-text-dense"
  "04-ebook-mixed"
  "05-academic-book"
  "06-modern-layout-dct"
  "07-journal-dct-heavy"
  "08-scan-dct-1927"
  "09-scan-dct-1836"
  "10-scan-jbig2-jpx"
)

# ── Header ────────────────────────────────────────────────────────────────────
REF_LABEL="pdftoppm"
[[ "$BACKEND" == "vaapi" ]] && REF_LABEL="cpu-only"

printf "\nSystem: load=%-5s  cores=%d\n" "$LOAD1" "$NCORES"
printf "Binary: %s  backend=%s\n" "$BIN" "$BACKEND"
printf "Runs:   %d (warmup %d) — hyperfine mean, cold cache\n\n" "$RUNS" "$WARMUP"
printf "NOTE: pdftoppm is single-threaded. Scan corpora (08-09) take several minutes each.\n\n"

printf "%-32s  %12s  %12s  %10s\n" "corpus" "pdf-raster" "$REF_LABEL" "speedup"
printf "%-32s  %12s  %12s  %10s\n" "------" "----------" "----------" "-------"

# ── Per-corpus ────────────────────────────────────────────────────────────────
for name in "${corpora[@]}"; do
  pdf="$FIXTURES/corpus-${name}.pdf"
  if [[ ! -f "$pdf" ]]; then
    printf "SKIP  %-32s  (file not found)\n" "$name" >&2
    continue
  fi

  check_disk

  PREPARE="$(which python3) -c \"import ctypes,os,sys; fd=os.open('$pdf',os.O_RDONLY); ctypes.CDLL(None).posix_fadvise(fd,0,os.fstat(fd).st_size,4); os.close(fd)\""

  # pdf-raster
  outdir=$(mktemp -d -p /tmp)
  raster_ms=$(hyperfine_mean "$PREPARE" "$BIN ${PDF_RASTER_ARGS[*]} '$pdf' '$outdir/r'") \
    || { rm -rf "$outdir"; echo "ERROR: hyperfine failed for $name (pdf-raster)" >&2; exit 1; }
  rm -rf "$outdir"

  # reference
  outdir=$(mktemp -d -p /tmp)
  if [[ "$BACKEND" == "vaapi" ]]; then
    ref_ms=$(hyperfine_mean "$PREPARE" "$BIN --backend cpu -r 150 '$pdf' '$outdir/c'") \
      || { rm -rf "$outdir"; echo "ERROR: hyperfine failed for $name (cpu-only)" >&2; exit 1; }
  else
    ref_ms=$(hyperfine_mean "$PREPARE" "pdftoppm -r 150 '$pdf' '$outdir/p'") \
      || { rm -rf "$outdir"; echo "ERROR: hyperfine failed for $name (pdftoppm)" >&2; exit 1; }
  fi
  rm -rf "$outdir"

  speedup=$(awk "BEGIN{printf \"%.2fx\", ($raster_ms>0) ? $ref_ms/$raster_ms : 0}")
  printf "%-32s  %9dms    %9dms  %10s\n" "$name" "$raster_ms" "$ref_ms" "$speedup"
done
