#!/usr/bin/env bash
# bench_features.sh — benchmark every pdf-raster feature variant against each other
# and against pdftoppm.
#
# Variants tested:
#   pdftoppm             — reference C++ baseline (poppler 24.x)
#   pdf-raster-scalar    — no SIMD, single-threaded
#   pdf-raster-avx2      — AVX2 blend/fill, single-threaded
#   pdf-raster-avx512    — AVX2 + AVX-512 VPOPCNTDQ popcnt, single-threaded
#   pdf-raster-rayon     — no SIMD, rayon tile parallelism
#   pdf-raster-avx2-rayon    — AVX2 + rayon
#   pdf-raster-avx512-rayon  — AVX2 + AVX-512 + rayon  (full build)
#
# For each variant, --threads is swept: 1, 4, 12, 24 (where rayon applies).
# For single-threaded variants only --threads 1 is run.
#
# Usage:
#   bench_features.sh [OPTIONS]
#
# Options:
#   -r DPIS       Comma-separated DPI values (default: 72,150,300)
#   -p PDF        Test only this fixture basename (default: all fixtures)
#   -l PAGE       Last page per run (default: 5)
#   -w N          Hyperfine warmup runs (default: 1)
#   -n N          Hyperfine measurement runs (default: 5)
#   -o FILE       JSON output path (default: bench/bench-features-results.json)
#   -b DIR        Directory containing pre-built variant binaries
#                 (default: tests/bench/bins/ — run build_variants.sh first)
#   -d            Dry run: print commands without executing
#
# Requires: hyperfine, pdftoppm, python3
# Pre-requisite: run build_variants.sh to populate -b DIR.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ── defaults ──────────────────────────────────────────────────────────────────
DPI_LIST="72,150,300"
SINGLE_PDF=""
LAST_PAGE=5
WARMUP=1
RUNS=5
JSON_OUT="${SCRIPT_DIR}/bench-features-results.json"
BINS_DIR="${SCRIPT_DIR}/bins"
FIXTURES_DIR="${REPO_ROOT}/tests/fixtures"
DRY_RUN=false

# Thread counts to sweep for rayon-capable variants.
THREAD_COUNTS=(1 4 12 24)

usage() {
    grep '^#' "$0" | grep -v '^#!/' | sed 's/^# \{0,2\}//'
    exit 0
}

while getopts ":r:p:l:w:n:o:b:dh" opt; do
    case $opt in
        r) DPI_LIST="$OPTARG" ;;
        p) SINGLE_PDF="$OPTARG" ;;
        l) LAST_PAGE="$OPTARG" ;;
        w) WARMUP="$OPTARG" ;;
        n) RUNS="$OPTARG" ;;
        o) JSON_OUT="$OPTARG" ;;
        b) BINS_DIR="$OPTARG" ;;
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
[[ -d "$BINS_DIR" ]] || {
    echo "Error: bins directory not found: $BINS_DIR" >&2
    echo "  Run: bash ${SCRIPT_DIR}/build_variants.sh" >&2
    exit 1
}

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

IFS=',' read -ra dpis <<< "$DPI_LIST"

# ── variant table ─────────────────────────────────────────────────────────────
# Format: "bin_basename:has_rayon:label"
# has_rayon=1  → sweep THREAD_COUNTS; =0 → only run with --threads 1
declare -a VARIANTS=(
    "pdf-raster-scalar:0:scalar"
    "pdf-raster-avx2:0:avx2"
    "pdf-raster-avx512:0:avx512"
    "pdf-raster-rayon:1:rayon"
    "pdf-raster-avx2-rayon:1:avx2+rayon"
    "pdf-raster-avx512-rayon:1:avx512+rayon"
)

# ── build the flat command list for a given pdf+dpi ───────────────────────────
# Outputs tab-separated lines: "label<TAB>command"
# Tab is safe because neither labels nor filesystem paths contain tabs.
build_commands() {
    local pdf="$1" dpi="$2" out_dir="$3"

    # Reference: pdftoppm (single-threaded, no thread flag)
    local ref_out="${out_dir}/ref"
    mkdir -p "$ref_out"
    printf 'pdftoppm\tpdftoppm -r %s -f 1 -l %s %s %s/pg\n' \
        "$dpi" "$LAST_PAGE" "$pdf" "$ref_out"

    for entry in "${VARIANTS[@]}"; do
        local bin_name label has_rayon
        bin_name="$(cut -d: -f1 <<< "$entry")"
        has_rayon="$(cut -d: -f2 <<< "$entry")"
        label="$(cut -d: -f3 <<< "$entry")"
        local bin="${BINS_DIR}/${bin_name}"

        if [[ ! -x "$bin" ]]; then
            echo "Warning: $bin not found, skipping ${label}" >&2
            continue
        fi

        if [[ "$has_rayon" -eq 1 ]]; then
            for t in "${THREAD_COUNTS[@]}"; do
                local variant_out="${out_dir}/${bin_name}-t${t}"
                mkdir -p "$variant_out"
                printf '%s\t%s --threads %s -r %s -f 1 -l %s %s %s/pg\n' \
                    "${label}(t${t})" "$bin" "$t" "$dpi" "$LAST_PAGE" "$pdf" "$variant_out"
            done
        else
            local variant_out="${out_dir}/${bin_name}-t1"
            mkdir -p "$variant_out"
            printf '%s\t%s --threads 1 -r %s -f 1 -l %s %s %s/pg\n' \
                "${label}(t1)" "$bin" "$dpi" "$LAST_PAGE" "$pdf" "$variant_out"
        fi
    done
}

# ── dry run ───────────────────────────────────────────────────────────────────
if $DRY_RUN; then
    echo "DRY RUN — commands that would be passed to hyperfine:"
    echo ""
    for pdf in "${pdf_files[@]}"; do
        pdf_name="$(basename "$pdf")"
        for dpi in "${dpis[@]}"; do
            echo "  ── ${pdf_name}  @${dpi}dpi ──"
            while IFS=$'\t' read -r lbl cmd; do
                printf "    %-26s  %s\n" "[$lbl]" "$cmd"
            done < <(build_commands "$pdf" "$dpi" "<TMP>")
            echo ""
        done
    done
    echo "  JSON output → ${JSON_OUT}"
    exit 0
fi

# ── temp workspace ────────────────────────────────────────────────────────────
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

json_fragments=()

echo "══════════════════════════════════════════════════════════════════════════"
printf " Feature-matrix benchmark: pdftoppm vs pdf-raster variants\n"
printf " Bins:      %s\n" "$BINS_DIR"
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

        out_dir="${WORK_DIR}/${pdf_name//[^a-zA-Z0-9]/_}_${dpi}"
        mkdir -p "$out_dir"

        frag_file="${WORK_DIR}/frag_${pdf_name//[^a-zA-Z0-9]/_}_${dpi}.json"

        # Build parallel hyperfine --command-name / cmd pairs.
        hyperfine_args=(
            --warmup "$WARMUP"
            --runs "$RUNS"
            --export-json "$frag_file"
        )
        while IFS=$'\t' read -r lbl cmd; do
            hyperfine_args+=(--command-name "$lbl" "$cmd")
        done < <(build_commands "$pdf" "$dpi" "$out_dir")

        hyperfine "${hyperfine_args[@]}"
        json_fragments+=("$frag_file")

        # Discard rendered pages; keep fragment JSON.
        rm -rf "$out_dir"
    done
done

# ── merge JSON ────────────────────────────────────────────────────────────────
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

# ── summary table ─────────────────────────────────────────────────────────────
echo ""
echo "── Pages/second summary ─────────────────────────────────────────────────"
python3 - "$JSON_OUT" "$LAST_PAGE" <<'PYEOF'
import json, sys

with open(sys.argv[1]) as fh:
    data = json.load(fh)
last_page = int(sys.argv[2])

ref_mean = None
print(f"  {'Variant':<28} {'Mean (s)':>9} {'Pages/s':>10} {'vs pdftoppm':>13}")
print(f"  {'-'*28} {'-'*9} {'-'*10} {'-'*13}")
for r in data["results"]:
    name  = r["command"]
    mean  = r["mean"]
    pps   = last_page / mean
    if name == "pdftoppm":
        ref_mean = mean
    speedup = ""
    if ref_mean is not None and name != "pdftoppm":
        ratio = ref_mean / mean
        speedup = f"{ratio:+.2f}×"
    print(f"  {name:<28} {mean:>9.3f} {pps:>10.1f} {speedup:>13}")
PYEOF

echo ""
echo "Done."
