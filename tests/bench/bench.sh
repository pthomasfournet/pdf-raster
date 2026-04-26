#!/usr/bin/env bash
# bench.sh — throughput comparison: pdftoppm vs pdf-raster.
#
# For each fixture PDF and each DPI, runs hyperfine with both commands
# rendering the same page range and reports wall-clock time + pages/sec.
#
# Usage:
#   bench.sh [OPTIONS]
#
# Options:
#   -r DPIS         Comma-separated DPI values to test (default: 72,150,300)
#   -f DIR          Fixtures directory (default: tests/fixtures/)
#   -p PDF          Test only this PDF (basename, e.g. ritual-14th.pdf)
#   -l PAGE         Last page rendered per run (default: 5, keeps runs short)
#   -w N            Hyperfine warmup runs per benchmark (default: 1)
#   -c N            Hyperfine measurement runs per benchmark (default: 5)
#   -o FILE         Write JSON results to FILE (default: bench/bench-results.json)
#   -d              Dry run: print hyperfine commands without executing them
#
# Requires: hyperfine, pdftoppm, pdf-raster (release build), python3

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
DRY_RUN=false

usage() {
    grep '^#' "$0" | grep -v '^#!/' | sed 's/^# \{0,2\}//'
    exit 0
}

die() { echo "Error: $*" >&2; exit 1; }
is_positive_int() { [[ "$1" =~ ^[1-9][0-9]*$ ]]; }

while getopts ":r:f:p:l:w:c:o:dh" opt; do
    case $opt in
        r) DPI_LIST="$OPTARG" ;;
        f) FIXTURES_DIR="$OPTARG" ;;
        p) SINGLE_PDF="$OPTARG" ;;
        l) LAST_PAGE="$OPTARG" ;;
        w) WARMUP="$OPTARG" ;;
        c) RUNS="$OPTARG" ;;
        o) JSON_OUT="$OPTARG" ;;
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
for cmd in hyperfine pdftoppm python3; do
    command -v "$cmd" >/dev/null 2>&1 || die "$cmd not found in PATH"
done
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
            ref_cmd="pdftoppm -r ${dpi} -f 1 -l ${LAST_PAGE} ${pdf} <TMP>/ref/pg"
            new_cmd="${RASTER_BIN} -r ${dpi} -f 1 -l ${LAST_PAGE} ${pdf} <TMP>/new/pg"
            frag="<TMP>/frag_${pdf_name//[^a-zA-Z0-9]/_}_${dpi}.json"
            echo "  hyperfine --warmup ${WARMUP} --runs ${RUNS} --export-json ${frag} \\"
            echo "    --command-name 'pdftoppm  @${dpi}dpi' '${ref_cmd}' \\"
            echo "    --command-name 'pdf-raster@${dpi}dpi' '${new_cmd}'"
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

echo "══════════════════════════════════════════════════════════════════════════"
printf " Benchmark: pdftoppm vs pdf-raster\n"
printf " Fixtures:  %s\n" "$FIXTURES_DIR"
printf " DPIs:      %s\n" "$DPI_LIST"
printf " Pages:     1-%s per run\n" "$LAST_PAGE"
printf " Runs:      %d (warmup %d)\n" "$RUNS" "$WARMUP"
echo "══════════════════════════════════════════════════════════════════════════"

for pdf in "${pdf_files[@]}"; do
    pdf_name="$(basename "$pdf")"
    # Separate output dirs per DPI so stale pages from a prior DPI never linger.
    echo ""
    echo "── ${pdf_name} ──────────────────────────────────────────────────────────"

    for dpi in "${dpis[@]}"; do
        echo ""
        printf "  DPI %s — pages 1-%s\n" "$dpi" "$LAST_PAGE"

        ref_out="${OUT_DIR}/ref_${dpi}"
        new_out="${OUT_DIR}/new_${dpi}"
        mkdir -p "$ref_out" "$new_out"

        ref_cmd="pdftoppm -r ${dpi} -f 1 -l ${LAST_PAGE} ${pdf} ${ref_out}/pg"
        new_cmd="${RASTER_BIN} -r ${dpi} -f 1 -l ${LAST_PAGE} ${pdf} ${new_out}/pg"
        frag_file="${OUT_DIR}/frag_${pdf_name//[^a-zA-Z0-9]/_}_${dpi}.json"

        hyperfine \
            --warmup "$WARMUP" \
            --runs "$RUNS" \
            --export-json "$frag_file" \
            --command-name "pdftoppm  @${dpi}dpi" "$ref_cmd" \
            --command-name "pdf-raster@${dpi}dpi" "$new_cmd"

        json_fragments+=("$frag_file")

        # Discard rendered pages to keep disk usage bounded on large fixtures.
        rm -rf "$ref_out" "$new_out"
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

with open(sys.argv[1]) as fh:
    data = json.load(fh)

last_page = int(sys.argv[2])
results = data.get("results", [])

if not results:
    print("  (no results)", file=sys.stderr)
    sys.exit(0)

print(f"  {'Command':<30} {'Mean (s)':>9} {'Pages/s':>10} {'vs ref':>8}")
print(f"  {'-'*30} {'-'*9} {'-'*10} {'-'*8}")

# Build ref lookup before printing so ordering in the JSON doesn't matter.
ref_times = {
    r["command"].split("@")[1]: r["mean"]
    for r in results
    if r["command"].startswith("pdftoppm") and "@" in r["command"]
}

for r in results:
    name = r["command"]
    mean = r["mean"]
    pps  = last_page / mean
    label = name.split("@")[1] if "@" in name else ""
    is_ref = name.startswith("pdftoppm")

    speedup = ""
    if not is_ref and label in ref_times and ref_times[label] > 0:
        speedup = f"{ref_times[label] / mean:+.2f}×"

    print(f"  {name:<30} {mean:>9.3f} {pps:>10.1f} {speedup:>8}")
PYEOF

echo ""
echo "Done."
