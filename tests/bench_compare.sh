#!/bin/bash
# Compare pdf-raster against an external rasterizer (pdftoppm or mutool draw)
# across all corpus PDFs.  Apples-to-apples: all three tools write PPM/PNG
# output to a real /tmp directory (no --ram tmpfs redirect).
#
# Usage:
#   bench_compare.sh [--ref-tool pdftoppm|mutool] [--runs N] [--warmup N]
#                    [--corpus-dir <path>]
#
# Reports: pdf-raster mean, reference-tool mean, speedup ratio.
# This is intentionally separate from bench_corpus.sh — the reference tools
# are single-threaded and slow (minutes on scan corpora), so this should
# only be run when an explicit comparison is needed.

set -euo pipefail

BIN="${BIN:-$(dirname "$0")/../target/release/pdf-raster}"
FIXTURES="$(dirname "$0")/fixtures"
REF_TOOL="pdftoppm"
RUNS=3
WARMUP=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ref-tool)   REF_TOOL="$2"; shift 2 ;;
    --corpus-dir) FIXTURES="$2"; shift 2 ;;
    --runs)       RUNS="$2";     shift 2 ;;
    --warmup)     WARMUP="$2";   shift 2 ;;
    *) echo "ERROR: unknown argument: $1" >&2; exit 1 ;;
  esac
done

case "$REF_TOOL" in
  pdftoppm|mutool) ;;
  *) echo "ERROR: --ref-tool must be pdftoppm or mutool (got: $REF_TOOL)" >&2; exit 1 ;;
esac

[[ -x "$BIN" ]]      || { echo "ERROR: binary not found: $BIN" >&2; exit 1; }
command -v hyperfine >/dev/null || { echo "ERROR: hyperfine not found — cargo install hyperfine" >&2; exit 1; }
command -v "$REF_TOOL" >/dev/null || { echo "ERROR: $REF_TOOL not found" >&2; exit 1; }

# ── Pre-flight: system load gate ─────────────────────────────────────────────
NCORES=$(nproc)
LOAD1=$(cut -d' ' -f1 /proc/loadavg)
LOAD_INT=$(echo "$LOAD1" | awk '{print int($1)}')
if [[ "$LOAD_INT" -gt $((NCORES * 2)) ]]; then
  echo "ERROR: load ($LOAD1) too high — threshold $((NCORES*2)). Wait for system to settle." >&2
  exit 1
fi

check_disk() {
  local avail_kb
  avail_kb=$(df --output=avail -k /tmp 2>/dev/null | tail -1)
  if [[ -z "$avail_kb" || "$avail_kb" -lt 1048576 ]]; then
    echo "ERROR: less than 1 GB free on /tmp — aborting" >&2; exit 1
  fi
}

# hyperfine_mean PREPARE_CMD BENCH_CMD  →  prints mean_ms.
# On failure, hyperfine's stderr is replayed before the script exits.
hyperfine_mean() {
  local prepare_cmd="$1" bench_cmd="$2" json_f err_f
  json_f=$(mktemp --suffix=.json)
  err_f=$(mktemp --suffix=.err)
  if ! hyperfine --runs "$RUNS" --warmup "$WARMUP" \
       --prepare "$prepare_cmd" --export-json "$json_f" \
       -- "$bench_cmd" >/dev/null 2>"$err_f"; then
    echo "ERROR: hyperfine failed:" >&2
    sed 's/^/  /' "$err_f" >&2
    rm -f "$json_f" "$err_f"
    return 1
  fi
  python3 - "$json_f" <<'EOF'
import json, sys
with open(sys.argv[1]) as f:
    r = json.load(f)['results'][0]
print(f"{r['mean']*1000:.0f}")
EOF
  rm -f "$json_f" "$err_f"
}

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
# mutool/pdftoppm both print version to stderr and exit non-zero with no
# subcommand, so swallow the exit status — `set -o pipefail` would kill us.
case "$REF_TOOL" in
  mutool)   REF_VER="$( { mutool --version 2>&1 || true; } | head -1)" ;;
  pdftoppm) REF_VER="$( { pdftoppm -v       2>&1 || true; } | head -1)" ;;
esac

printf "\nSystem: load=%-5s  cores=%d\n" "$LOAD1" "$NCORES"
printf "Binary: %s\n"            "$BIN"
printf "Reference: %s\n"         "$REF_VER"
printf "Runs:   %d (warmup %d) — hyperfine mean, cold cache\n\n" "$RUNS" "$WARMUP"
printf "NOTE: %s is single-threaded. Scan corpora (08-09) take several minutes each.\n\n" "$REF_TOOL"

printf "%-32s  %12s  %12s  %10s\n" "corpus" "pdf-raster" "$REF_TOOL" "speedup"
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
  raster_ms=$(hyperfine_mean "$PREPARE" "$BIN --backend cpu -r 150 '$pdf' '$outdir/r'") \
    || { rm -rf "$outdir"; exit 1; }
  rm -rf "$outdir"

  # reference
  outdir=$(mktemp -d -p /tmp)
  case "$REF_TOOL" in
    mutool)   ref_cmd="mutool draw -P -r 150 -o '$outdir/m-%d.png' '$pdf'" ;;
    pdftoppm) ref_cmd="pdftoppm -r 150 '$pdf' '$outdir/p'" ;;
  esac
  ref_ms=$(hyperfine_mean "$PREPARE" "$ref_cmd") \
    || { rm -rf "$outdir"; exit 1; }
  rm -rf "$outdir"

  speedup=$(awk "BEGIN{printf \"%.2fx\", ($raster_ms>0) ? $ref_ms/$raster_ms : 0}")
  printf "%-32s  %9dms    %9dms  %10s\n" "$name" "$raster_ms" "$ref_ms" "$speedup"
done
