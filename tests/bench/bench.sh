#!/usr/bin/env bash
# bench.sh — throughput benchmark: pdf-raster (+ optional pdftoppm and extras).
#
# For each fixture PDF and each DPI, runs hyperfine and reports wall-clock
# time + pages/sec.  pdftoppm is skipped by default; use -R to include it.
#
# Usage:
#   bench.sh [OPTIONS]
#
# Options:
#   -r DPIS         Comma-separated DPI values to test (default: 72,150,300)
#   -f DIR          Fixtures directory (default: tests/fixtures/)
#   -p PDF          Test only this PDF (basename, e.g. your-document.pdf)
#   -l PAGE         Last page rendered per run (default: 5, keeps runs short)
#   -w N            Hyperfine warmup runs per benchmark (default: 1)
#   -c N            Hyperfine measurement runs per benchmark (default: 5)
#   -o FILE         Write JSON results to FILE (default: bench/bench-results.json)
#   -R              Include pdftoppm as a reference (slow; skipped by default)
#   -e              Include extra rasterisers: Ghostscript (gs) and MuPDF (mutool)
#   -d              Dry run: print hyperfine commands without executing them
#
# Requires: hyperfine, pdf-raster (release build), python3
# With -R: also requires pdftoppm
# With -e: also requires gs (Ghostscript) and mutool (MuPDF)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RASTER_BIN="${REPO_ROOT}/target/release/pdf-raster"
DEFAULT_FIXTURES="${REPO_ROOT}/tests/fixtures"

# ── defaults ──────────────────────────────────────────────────────────────────
DPI_LIST="72,150,300"
FIXTURES_DIR="$DEFAULT_FIXTURES"
SINGLE_PDF=""
LAST_PAGE=5
WARMUP=1
RUNS=5
JSON_OUT="${SCRIPT_DIR}/bench-results.json"
INCLUDE_REF=false
EXTRAS=false
DRY_RUN=false

usage() {
    grep '^#' "$0" | grep -v '^#!/' | sed 's/^# \{0,2\}//'
    exit 0
}

die() { echo "Error: $*" >&2; exit 1; }
is_positive_int() { [[ "$1" =~ ^[1-9][0-9]*$ ]]; }

while getopts ":r:f:p:l:w:c:o:Redh" opt; do
    case $opt in
        r) DPI_LIST="$OPTARG" ;;
        f) FIXTURES_DIR="$OPTARG" ;;
        p) SINGLE_PDF="$OPTARG" ;;
        l) LAST_PAGE="$OPTARG" ;;
        w) WARMUP="$OPTARG" ;;
        c) RUNS="$OPTARG" ;;
        o) JSON_OUT="$OPTARG" ;;
        R) INCLUDE_REF=true ;;
        e) EXTRAS=true ;;
        d) DRY_RUN=true ;;
        h) usage ;;
        :) die "option -$OPTARG requires an argument." ;;
       \?) die "unknown option -$OPTARG." ;;
    esac
done

# ── input validation ──────────────────────────────────────────────────────────
is_positive_int "$LAST_PAGE" || die "-l must be a positive integer, got: $LAST_PAGE"
is_positive_int "$WARMUP"    || die "-w must be a positive integer, got: $WARMUP"
is_positive_int "$RUNS"      || die "-c must be a positive integer, got: $RUNS"

# ── dependency check ──────────────────────────────────────────────────────────
for cmd in hyperfine python3; do
    command -v "$cmd" >/dev/null 2>&1 || die "$cmd not found in PATH"
done
if $INCLUDE_REF; then
    command -v pdftoppm >/dev/null 2>&1 || die "-R requires pdftoppm but it was not found in PATH"
fi
if $EXTRAS; then
    for cmd in gs mutool; do
        command -v "$cmd" >/dev/null 2>&1 || die "-e requires $cmd but it was not found in PATH"
    done
fi
[[ -x "$RASTER_BIN" ]] || {
    echo "Error: pdf-raster not built at $RASTER_BIN" >&2
    echo "  Run: cargo build --release -p pdf-raster" >&2
    exit 1
}
[[ -d "$FIXTURES_DIR" ]] || die "fixtures dir not found: $FIXTURES_DIR"

# ── collect PDFs ──────────────────────────────────────────────────────────────
if [[ -n "$SINGLE_PDF" ]]; then
    pdf_files=("${FIXTURES_DIR}/${SINGLE_PDF}")
    [[ -f "${pdf_files[0]}" ]] || die "PDF not found: ${pdf_files[0]}"
else
    shopt -s nullglob
    pdf_files=("${FIXTURES_DIR}"/*.pdf)
    shopt -u nullglob
    [[ ${#pdf_files[@]} -gt 0 ]] || die "no PDFs found in $FIXTURES_DIR"
fi

# Parse DPI list once — used in both dry-run and live paths.
IFS=',' read -ra dpis <<< "$DPI_LIST"
[[ ${#dpis[@]} -gt 0 ]] || die "DPI list is empty"

# ── dry-run ───────────────────────────────────────────────────────────────────
if $DRY_RUN; then
    echo "DRY RUN — hyperfine commands that would be executed:"
    echo ""
    for pdf in "${pdf_files[@]}"; do
        pdf_name="$(basename "$pdf")"
        echo "  ── ${pdf_name} ──"
        for dpi in "${dpis[@]}"; do
            new_cmd="${RASTER_BIN} -r ${dpi} -f 1 -l ${LAST_PAGE} ${pdf} <TMP>/new/pg"
            frag="<TMP>/frag_${pdf_name//[^a-zA-Z0-9]/_}_${dpi}.json"
            echo "  hyperfine --warmup ${WARMUP} --runs ${RUNS} --export-json ${frag} \\"
            if $INCLUDE_REF; then
                ref_cmd="pdftoppm -r ${dpi} -f 1 -l ${LAST_PAGE} ${pdf} <TMP>/ref/pg"
                echo "    --command-name 'pdftoppm  @${dpi}dpi' '${ref_cmd}' \\"
            fi
            echo "    --command-name 'pdf-raster@${dpi}dpi' '${new_cmd}'"
            if $EXTRAS; then
                gs_cmd="gs -q -sDEVICE=ppmraw -r${dpi} -dNOPAUSE -dBATCH -dFirstPage=1 -dLastPage=${LAST_PAGE} -sOutputFile=<TMP>/gs/pg_%d.ppm ${pdf}"
                mut_cmd="mutool draw -q -r ${dpi} -F ppm -o <TMP>/mut/pg_%d.ppm ${pdf} 1-${LAST_PAGE}"
                echo "    --command-name 'gs        @${dpi}dpi' '${gs_cmd}' \\"
                echo "    --command-name 'mutool    @${dpi}dpi' '${mut_cmd}'"
            fi
            echo ""
        done
    done
    echo "  JSON output → ${JSON_OUT}"
    exit 0
fi

# ── temp dir for rendered pages (discarded between runs) ──────────────────────
OUT_DIR="$(mktemp -d)"
trap 'rm -rf "$OUT_DIR"' EXIT

json_fragments=()

ref_label=""
$INCLUDE_REF && ref_label=" vs pdftoppm"
extras_label=""
$EXTRAS && extras_label=" + gs + mutool"

echo "══════════════════════════════════════════════════════════════════════════"
printf " Benchmark: pdf-raster%s%s\n" "$ref_label" "$extras_label"
printf " Fixtures:  %s\n" "$FIXTURES_DIR"
printf " DPIs:      %s\n" "$DPI_LIST"
printf " Pages:     1-%s per run\n" "$LAST_PAGE"
printf " Runs:      %d (warmup %d)\n" "$RUNS" "$WARMUP"
echo "══════════════════════════════════════════════════════════════════════════"

for pdf in "${pdf_files[@]}"; do
    pdf_name="$(basename "$pdf")"
    echo ""
    echo "── ${pdf_name} ──────────────────────────────────────────────────────────"

    for dpi in "${dpis[@]}"; do
        echo ""
        printf "  DPI %s — pages 1-%s\n" "$dpi" "$LAST_PAGE"

        new_out="${OUT_DIR}/new_${dpi}"
        mkdir -p "$new_out"

        new_cmd="${RASTER_BIN} -r ${dpi} -f 1 -l ${LAST_PAGE} ${pdf} ${new_out}/pg"
        frag_file="${OUT_DIR}/frag_${pdf_name//[^a-zA-Z0-9]/_}_${dpi}.json"

        hyperfine_args=(
            --warmup "$WARMUP"
            --runs "$RUNS"
            --export-json "$frag_file"
            --command-name "pdf-raster@${dpi}dpi" "$new_cmd"
        )

        if $INCLUDE_REF; then
            ref_out="${OUT_DIR}/ref_${dpi}"
            mkdir -p "$ref_out"
            ref_cmd="pdftoppm -r ${dpi} -f 1 -l ${LAST_PAGE} ${pdf} ${ref_out}/pg"
            hyperfine_args+=(
                --command-name "pdftoppm  @${dpi}dpi" "$ref_cmd"
            )
        fi

        if $EXTRAS; then
            gs_out="${OUT_DIR}/gs_${dpi}"
            mut_out="${OUT_DIR}/mut_${dpi}"
            mkdir -p "$gs_out" "$mut_out"

            gs_cmd="gs -q -sDEVICE=ppmraw -r${dpi} -dNOPAUSE -dBATCH \
                -dFirstPage=1 -dLastPage=${LAST_PAGE} \
                -sOutputFile=${gs_out}/pg_%d.ppm ${pdf}"
            mut_cmd="mutool draw -q -r ${dpi} -F ppm \
                -o ${mut_out}/pg_%d.ppm ${pdf} 1-${LAST_PAGE}"

            hyperfine_args+=(
                --command-name "gs        @${dpi}dpi" "$gs_cmd"
                --command-name "mutool    @${dpi}dpi" "$mut_cmd"
            )
        fi

        hyperfine "${hyperfine_args[@]}"
        json_fragments+=("$frag_file")

        rm -rf "$new_out"
        $INCLUDE_REF && rm -rf "${OUT_DIR}/ref_${dpi}"
        $EXTRAS && rm -rf "${OUT_DIR}/gs_${dpi}" "${OUT_DIR}/mut_${dpi}"
    done
done

[[ ${#json_fragments[@]} -gt 0 ]] || die "no benchmark runs completed — nothing to write"

echo ""
echo "── Writing combined results → ${JSON_OUT} ───────────────────────────────"
python3 - "${json_fragments[@]}" "$JSON_OUT" <<'PYEOF'
import json, sys

frags = sys.argv[1:-1]
out_path = sys.argv[-1]

all_results = []
for f in frags:
    with open(f) as fh:
        data = json.load(fh)
    all_results.extend(data.get("results", []))

if not all_results:
    print("Warning: no benchmark results found in fragment files", file=sys.stderr)

with open(out_path, "w") as fh:
    json.dump({"results": all_results}, fh, indent=2)

print(f"  Written {len(all_results)} benchmark records to {out_path}")
PYEOF

echo ""
echo "── Pages/second summary ─────────────────────────────────────────────────"
python3 - "$JSON_OUT" "$LAST_PAGE" <<'PYEOF'
import json, sys
from collections import defaultdict

with open(sys.argv[1]) as fh:
    data = json.load(fh)

last_page = int(sys.argv[2])
results = data.get("results", [])

if not results:
    print("  (no results)", file=sys.stderr)
    sys.exit(0)

# Group by the DPI label (the "@NNNdpi" suffix) so the table is readable
# when multiple rasterisers and DPIs are in the same run.
def dpi_key(r):
    name = r["command"]
    return name.split("@")[1] if "@" in name else ""

groups = defaultdict(list)
for r in results:
    groups[dpi_key(r)].append(r)

# Build pdftoppm reference times per DPI key for the "vs ref" column.
ref_times = {
    dpi_key(r): r["mean"]
    for r in results
    if r["command"].strip().startswith("pdftoppm") and "@" in r["command"]
}

has_ref = bool(ref_times)
if has_ref:
    hdr = f"  {'Command':<32} {'Mean (s)':>9} {'Pages/s':>10} {'vs pdftoppm':>13}"
    sep = f"  {'-'*32} {'-'*9} {'-'*10} {'-'*13}"
else:
    hdr = f"  {'Command':<32} {'Mean (s)':>9} {'Pages/s':>10}"
    sep = f"  {'-'*32} {'-'*9} {'-'*10}"

for group_label in sorted(groups):
    print()
    print(f"  ── {group_label} {'─'*50}" if group_label else "  ──")
    print(hdr)
    print(sep)
    ref_t = ref_times.get(group_label, 0.0)
    for r in groups[group_label]:
        name = r["command"]
        mean = r["mean"]
        pps  = last_page / mean
        is_ref = name.strip().startswith("pdftoppm")
        if has_ref:
            if is_ref or ref_t <= 0:
                speedup = "  (ref)" if is_ref else ""
            else:
                speedup = f"{ref_t / mean:.2f}×"
            print(f"  {name:<32} {mean:>9.3f} {pps:>10.1f} {speedup:>13}")
        else:
            print(f"  {name:<32} {mean:>9.3f} {pps:>10.1f}")
PYEOF

echo ""
echo "Done."
