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
#   -n N            Hyperfine measurement runs per benchmark (default: 5)
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

while getopts ":r:f:p:l:w:n:o:dh" opt; do
    case $opt in
        r) DPI_LIST="$OPTARG" ;;
        f) FIXTURES_DIR="$OPTARG" ;;
        p) SINGLE_PDF="$OPTARG" ;;
        l) LAST_PAGE="$OPTARG" ;;
        w) WARMUP="$OPTARG" ;;
        n) RUNS="$OPTARG" ;;
        o) JSON_OUT="$OPTARG" ;;
        d) DRY_RUN=true ;;
        h) usage ;;
        :) echo "Error: option -$OPTARG requires an argument." >&2; exit 1 ;;
       \?) echo "Error: unknown option -$OPTARG." >&2; exit 1 ;;
    esac
done

# ── dependency check ──────────────────────────────────────────────────────────
for cmd in hyperfine pdftoppm python3; do
    command -v "$cmd" >/dev/null 2>&1 || { echo "Error: $cmd not found in PATH" >&2; exit 1; }
done
[[ -x "$RASTER_BIN" ]] || {
    echo "Error: pdf-raster not built at $RASTER_BIN" >&2
    echo "  Run: cargo build --release -p pdf-raster" >&2
    exit 1
}

[[ -d "$FIXTURES_DIR" ]] || { echo "Error: fixtures dir not found: $FIXTURES_DIR" >&2; exit 1; }

# ── collect PDFs ──────────────────────────────────────────────────────────────
if [[ -n "$SINGLE_PDF" ]]; then
    pdf_files=("${FIXTURES_DIR}/${SINGLE_PDF}")
    [[ -f "${pdf_files[0]}" ]] || { echo "Error: PDF not found: ${pdf_files[0]}" >&2; exit 1; }
else
    shopt -s nullglob
    pdf_files=("${FIXTURES_DIR}"/*.pdf)
    shopt -u nullglob
    [[ ${#pdf_files[@]} -gt 0 ]] || { echo "Error: no PDFs in $FIXTURES_DIR" >&2; exit 1; }
fi

# ── dry-run ───────────────────────────────────────────────────────────────────
if $DRY_RUN; then
    echo "DRY RUN — hyperfine commands that would be executed:"
    echo ""
    IFS=',' read -ra dpis <<< "$DPI_LIST"
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

# ── aggregate JSON output ─────────────────────────────────────────────────────
json_fragments=()

echo "══════════════════════════════════════════════════════════════════════════"
printf " Benchmark: pdftoppm vs pdf-raster\n"
printf " Fixtures:  %s\n" "$FIXTURES_DIR"
printf " DPIs:      %s\n" "$DPI_LIST"
printf " Pages:     1-%s per run\n" "$LAST_PAGE"
printf " Runs:      %d (warmup %d)\n" "$RUNS" "$WARMUP"
echo "══════════════════════════════════════════════════════════════════════════"

IFS=',' read -ra dpis <<< "$DPI_LIST"

for pdf in "${pdf_files[@]}"; do
    pdf_name="$(basename "$pdf")"
    ref_out="${OUT_DIR}/ref"
    new_out="${OUT_DIR}/new"
    mkdir -p "$ref_out" "$new_out"

    echo ""
    echo "── ${pdf_name} ──────────────────────────────────────────────────────────"

    for dpi in "${dpis[@]}"; do
        echo ""
        printf "  DPI %s — pages 1-%s\n" "$dpi" "$LAST_PAGE"

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

        # Clean up rendered pages so disk doesn't fill up on large fixtures.
        rm -f "${ref_out}"/pg-*.ppm "${new_out}"/pg-*.ppm
    done
done

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

print(f"  {'Command':<30} {'Mean (s)':>9} {'Pages/s':>10} {'vs ref':>8}")
print(f"  {'-'*30} {'-'*9} {'-'*10} {'-'*8}")

ref_times = {}
for r in data["results"]:
    name = r["command"]
    mean = r["mean"]
    pages_per_sec = last_page / mean
    label = name.split("@")[1] if "@" in name else ""
    is_ref = name.startswith("pdftoppm")
    if is_ref:
        ref_times[label] = mean

    speedup = ""
    if not is_ref and label in ref_times:
        ratio = ref_times[label] / mean
        speedup = f"{ratio:+.2f}×"

    print(f"  {name:<30} {mean:>9.3f} {pages_per_sec:>10.1f} {speedup:>8}")
PYEOF

echo ""
echo "Done."
