#!/bin/bash
# Benchmark all corpus PDFs against pdftoppm at 150 DPI.
# Usage: bench_corpus.sh [--backend cpu|vaapi|cuda] [--vaapi-device /dev/dri/renderD129]
#
# Each PDF is timed independently; output is discarded.  Disk space on /tmp is
# checked before every run and the script aborts loudly if < 1 GB is free.

set -euo pipefail

BIN="${BIN:-$(dirname "$0")/../target/release/pdf-raster}"
FIXTURES="$(dirname "$0")/fixtures"
BACKEND="cpu"
VAAPI_DEVICE="/dev/dri/renderD129"

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend)      BACKEND="$2";      shift 2 ;;
    --vaapi-device) VAAPI_DEVICE="$2"; shift 2 ;;
    *) echo "ERROR: unknown argument: $1" >&2; exit 1 ;;
  esac
done

# ── Pre-flight checks ─────────────────────────────────────────────────────────
check_disk() {
  local dir="$1"
  local min_kb=1048576  # 1 GB
  local avail_kb
  avail_kb=$(df --output=avail -k "$dir" 2>/dev/null | tail -1)
  if [[ -z "$avail_kb" || "$avail_kb" -lt "$min_kb" ]]; then
    echo "ERROR: less than 1 GB free on $(df --output=target "$dir" | tail -1) — aborting to avoid silent zero-byte output" >&2
    exit 1
  fi
}

[[ -x "$BIN" ]] || { echo "ERROR: binary not found or not executable: $BIN" >&2; exit 1; }
command -v pdftoppm >/dev/null || { echo "ERROR: pdftoppm not found in PATH" >&2; exit 1; }

# ── Build pdf-raster argument list ───────────────────────────────────────────
PDF_RASTER_ARGS=(--backend "$BACKEND" -r 150)
if [[ "$BACKEND" == "vaapi" ]]; then
  PDF_RASTER_ARGS+=(--vaapi-device "$VAAPI_DEVICE")
fi

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
if [[ "$BACKEND" == "vaapi" ]]; then
  printf "%-45s %14s %14s %10s\n" "PDF" "vaapi (iGPU)" "cpu-only" "speedup"
else
  printf "%-45s %14s %14s %10s\n" "PDF" "pdf-raster" "pdftoppm" "speedup"
fi
printf "%-45s %14s %14s %10s\n" "---" "------------" "--------" "-------"

# ── Per-corpus timing ─────────────────────────────────────────────────────────
for name in "${corpora[@]}"; do
  pdf="$FIXTURES/corpus-${name}.pdf"
  if [[ ! -f "$pdf" ]]; then
    echo "SKIP: $pdf not found" >&2
    continue
  fi

  # ── pdf-raster ──
  check_disk /tmp
  TMPDIR_R=$(mktemp -d)
  t1=$( { time "$BIN" "${PDF_RASTER_ARGS[@]}" "$pdf" "$TMPDIR_R/r" >/dev/null 2>&1; } 2>&1 | awk '/^real/{print $2}')
  ms1=$(echo "$t1" | awk -F'[ms]' '{printf "%d", ($1*60+$2)*1000}')
  rm -rf "$TMPDIR_R"

  # ── reference (pdftoppm or cpu-only) ──
  check_disk /tmp
  TMPDIR_P=$(mktemp -d)
  if [[ "$BACKEND" == "vaapi" ]]; then
    t2=$( { time "$BIN" --backend cpu -r 150 "$pdf" "$TMPDIR_P/c" >/dev/null 2>&1; } 2>&1 | awk '/^real/{print $2}')
  else
    t2=$( { time pdftoppm -r 150 "$pdf" "$TMPDIR_P/p" >/dev/null 2>&1; } 2>&1 | awk '/^real/{print $2}')
  fi
  ms2=$(echo "$t2" | awk -F'[ms]' '{printf "%d", ($1*60+$2)*1000}')
  rm -rf "$TMPDIR_P"

  if [[ "$ms1" -le 0 || "$ms2" -le 0 ]]; then
    echo "ERROR: zero timing for $name (pdf-raster=${ms1}ms ref=${ms2}ms) — timing parse failed" >&2
    exit 1
  fi

  sp=$(awk "BEGIN{printf \"%.2fx\", $ms2/$ms1}")
  printf "%-45s %11dms %11dms %10s\n" "$name" "$ms1" "$ms2" "$sp"
done
