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
# For rayon-capable variants --threads is swept: 1, 4, 12, 24 (override with -T).
# For single-threaded variants only --threads 1 is used.
#
# Usage:
#   bench_features.sh [OPTIONS]
#
# Options:
#   -r DPIS       Comma-separated DPI values (default: 72,150,300)
#   -p PDF        Test only this fixture basename (default: all fixtures)
#   -l PAGE       Last page per run (default: 5)
#   -w N          Hyperfine warmup runs (default: 1)
#   -c N          Hyperfine measurement runs (default: 5)
#   -T COUNTS     Comma-separated thread counts to sweep for rayon variants
#                 (default: 1,4,12,24)
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
THREAD_LIST="1,4,12,24"
JSON_OUT="${SCRIPT_DIR}/bench-features-results.json"
BINS_DIR="${SCRIPT_DIR}/bins"
FIXTURES_DIR="${REPO_ROOT}/tests/fixtures"
DRY_RUN=false

usage() {
    grep '^#' "$0" | grep -v '^#!/' | sed 's/^# \{0,2\}//'
    exit 0
}

die() { echo "Error: $*" >&2; exit 1; }
is_positive_int() { [[ "$1" =~ ^[1-9][0-9]*$ ]]; }

while getopts ":r:p:l:w:c:T:o:b:dh" opt; do
    case $opt in
        r) DPI_LIST="$OPTARG" ;;
        p) SINGLE_PDF="$OPTARG" ;;
        l) LAST_PAGE="$OPTARG" ;;
        w) WARMUP="$OPTARG" ;;
        c) RUNS="$OPTARG" ;;
        T) THREAD_LIST="$OPTARG" ;;
        o) JSON_OUT="$OPTARG" ;;
        b) BINS_DIR="$OPTARG" ;;
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
[[ -d "$BINS_DIR" ]] || {
    echo "Error: bins directory not found: $BINS_DIR" >&2
    echo "  Run: bash ${SCRIPT_DIR}/build_variants.sh" >&2
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

# Parse lists once — shared by dry-run and live paths.
IFS=',' read -ra dpis         <<< "$DPI_LIST"
IFS=',' read -ra thread_counts <<< "$THREAD_LIST"
[[ ${#dpis[@]} -gt 0 ]]          || die "DPI list is empty"
[[ ${#thread_counts[@]} -gt 0 ]] || die "thread count list is empty"

# ── variant table ─────────────────────────────────────────────────────────────
# Each entry: "bin_basename TAB has_rayon TAB display_label"
# has_rayon=1 → sweep thread_counts; =0 → only run at --threads 1.
# Tab delimiter avoids colon conflicts in future labels.
declare -a VARIANTS=(
    "pdf-raster-scalar	0	scalar"
    "pdf-raster-avx2	0	avx2"
    "pdf-raster-avx512	0	avx512"
    "pdf-raster-rayon	1	rayon"
    "pdf-raster-avx2-rayon	1	avx2+rayon"
    "pdf-raster-avx512-rayon	1	avx512+rayon"
)

# ── build_commands <pdf> <dpi> <out_dir> ─────────────────────────────────────
# Prints tab-separated "label<TAB>command" lines for every (variant, thread).
# When out_dir is the dry-run sentinel "<TMP>", no real directories are created.
build_commands() {
    local pdf="$1" dpi="$2" out_dir="$3"
    local is_dry=false
    [[ "$out_dir" == "<TMP>" ]] && is_dry=true

    # Reference: pdftoppm, no thread flag.
    local ref_out="${out_dir}/ref"
    $is_dry || mkdir -p "$ref_out"
    printf 'pdftoppm\tpdftoppm -r %s -f 1 -l %s %s %s/pg\n' \
        "$dpi" "$LAST_PAGE" "$pdf" "$ref_out"

    for entry in "${VARIANTS[@]}"; do
        local bin_name has_rayon label
        # Split on tab — safe because bin names and labels contain no tabs.
        bin_name="$(  printf '%s' "$entry" | cut -f1)"
        has_rayon="$( printf '%s' "$entry" | cut -f2)"
        label="$(     printf '%s' "$entry" | cut -f3)"
        local bin="${BINS_DIR}/${bin_name}"

        if [[ ! -x "$bin" ]]; then
            echo "Warning: ${bin} not found — skipping ${label}" >&2
            continue
        fi

        if [[ "$has_rayon" -eq 1 ]]; then
            for t in "${thread_counts[@]}"; do
                local variant_out="${out_dir}/${bin_name}-t${t}"
                $is_dry || mkdir -p "$variant_out"
                printf '%s\t%s --threads %s -r %s -f 1 -l %s %s %s/pg\n' \
                    "${label}(t${t})" "$bin" "$t" "$dpi" "$LAST_PAGE" "$pdf" "$variant_out"
            done
        else
            local variant_out="${out_dir}/${bin_name}-t1"
            $is_dry || mkdir -p "$variant_out"
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
    echo "  Thread sweep: ${THREAD_LIST}"
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
printf " Threads:   %s (rayon variants)\n" "$THREAD_LIST"
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

        # Collect all (label, command) pairs for this pdf+dpi.
        hyperfine_args=(
            --warmup "$WARMUP"
            --runs   "$RUNS"
            --export-json "$frag_file"
        )
        cmd_count=0
        while IFS=$'\t' read -r lbl cmd; do
            hyperfine_args+=(--command-name "$lbl" "$cmd")
            cmd_count=$((cmd_count + 1))
        done < <(build_commands "$pdf" "$dpi" "$out_dir")

        if [[ $cmd_count -eq 0 ]]; then
            echo "  Warning: no runnable commands for ${pdf_name} @${dpi}dpi — skipping" >&2
            rm -rf "$out_dir"
            continue
        fi

        hyperfine "${hyperfine_args[@]}"
        json_fragments+=("$frag_file")

        # Rendered pages no longer needed; the WORK_DIR trap handles the rest.
        rm -rf "$out_dir"
    done
done

[[ ${#json_fragments[@]} -gt 0 ]] || die "no benchmark runs completed — nothing to write"

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

if not all_results:
    print("Warning: no benchmark results found in fragment files", file=sys.stderr)

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
results = data.get("results", [])

if not results:
    print("  (no results)", file=sys.stderr)
    sys.exit(0)

# Build pdftoppm reference times keyed by DPI label before printing,
# so ordering in the JSON does not affect the speedup column.
ref_times: dict[str, float] = {}
for r in results:
    name = r["command"]
    if name == "pdftoppm" or (name.startswith("pdftoppm") and "@" not in name):
        ref_times[""] = r["mean"]
    # pdftoppm has no @label in bench_features; treat first occurrence as global ref.

# Fallback: if there's exactly one pdftoppm entry, use it universally.
pdftoppm_mean: float | None = None
for r in results:
    if r["command"] == "pdftoppm":
        pdftoppm_mean = r["mean"]
        break

print(f"  {'Variant':<28} {'Mean (s)':>9} {'Pages/s':>10} {'vs pdftoppm':>13}")
print(f"  {'-'*28} {'-'*9} {'-'*10} {'-'*13}")
for r in results:
    name = r["command"]
    mean = r["mean"]
    pps  = last_page / mean
    is_ref = (name == "pdftoppm")

    speedup = ""
    if not is_ref and pdftoppm_mean is not None and pdftoppm_mean > 0:
        speedup = f"{pdftoppm_mean / mean:+.2f}×"

    print(f"  {name:<28} {mean:>9.3f} {pps:>10.1f} {speedup:>13}")
PYEOF

echo ""
echo "Done."
