#!/usr/bin/env bash
# compare.sh — pixel-accurate comparison of pdftoppm vs pdf-raster output.
#
# Usage:
#   compare.sh [OPTIONS] <PDF>
#
# Options:
#   -r DPI          Resolution (default: 150)
#   -f PAGE         First page (default: 1)
#   -l PAGE         Last page  (default: all)
#   -t THRESHOLD    Max allowed RMSE per page (0-255 scale, default: 2.0)
#   -v              Verbose: print per-pixel stats for passing pages too
#   -o DIR          Write diff images to DIR instead of discarding them
#
# Exit code: 0 if every page is within threshold, 1 otherwise.
#
# Requires: pdftoppm, pdf-raster, ImageMagick (compare/identify/convert), bc

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RASTER_BIN="${REPO_ROOT}/target/release/pdf-raster"

# ── defaults ──────────────────────────────────────────────────────────────────
DPI=150
FIRST=1
LAST=""
THRESHOLD=2.0
VERBOSE=false
OUT_DIR=""

usage() {
    sed -n 's/^# \{0,2\}//p' "$0" | sed '1d; /^!/d'
    exit 1
}

while getopts ":r:f:l:t:vo:h" opt; do
    case $opt in
        r) DPI="$OPTARG" ;;
        f) FIRST="$OPTARG" ;;
        l) LAST="$OPTARG" ;;
        t) THRESHOLD="$OPTARG" ;;
        v) VERBOSE=true ;;
        o) OUT_DIR="$OPTARG" ;;
        h) usage ;;
        :) echo "Option -$OPTARG requires an argument." >&2; exit 1 ;;
        \?) echo "Unknown option: -$OPTARG" >&2; exit 1 ;;
    esac
done
shift $((OPTIND - 1))

[[ $# -lt 1 ]] && { echo "Error: PDF argument required." >&2; usage; }
PDF="$1"
[[ -f "$PDF" ]] || { echo "Error: file not found: $PDF" >&2; exit 1; }

# ── dependency check ──────────────────────────────────────────────────────────
for cmd in pdftoppm compare identify convert bc; do
    command -v "$cmd" >/dev/null 2>&1 || { echo "Error: $cmd not found in PATH" >&2; exit 1; }
done
[[ -x "$RASTER_BIN" ]] || {
    echo "Error: pdf-raster not built at $RASTER_BIN" >&2
    echo "  Run: cargo build --release -p pdf-raster" >&2
    exit 1
}

# ── temp workspace ────────────────────────────────────────────────────────────
WORK_DIR="$(mktemp -d)"
cleanup() { rm -rf "$WORK_DIR"; }
trap cleanup EXIT

REF_DIR="${WORK_DIR}/ref"
NEW_DIR="${WORK_DIR}/new"
DIFF_DIR="${OUT_DIR:-${WORK_DIR}/diff}"
mkdir -p "$REF_DIR" "$NEW_DIR" "$DIFF_DIR"

# ── render: reference pdftoppm ────────────────────────────────────────────────
ref_args=(-r "$DPI" -f "$FIRST")
[[ -n "$LAST" ]] && ref_args+=(-l "$LAST")
pdftoppm "${ref_args[@]}" "$PDF" "${REF_DIR}/page" 2>/dev/null

# ── render: pdf-raster ────────────────────────────────────────────────────────
new_args=(-r "$DPI" -f "$FIRST")
[[ -n "$LAST" ]] && new_args+=(-l "$LAST")
"$RASTER_BIN" "${new_args[@]}" "$PDF" "${NEW_DIR}/page" 2>/dev/null

# ── compare page by page ──────────────────────────────────────────────────────
fail=0
pass=0
total_rmse="0"
max_rmse="0"
worst_page=""

shopt -s nullglob
ref_pages=("${REF_DIR}"/page-*.ppm)
shopt -u nullglob

if [[ ${#ref_pages[@]} -eq 0 ]]; then
    echo "Error: pdftoppm produced no output for ${PDF}" >&2
    exit 1
fi

for ref_file in "${ref_pages[@]}"; do
    page_tag="$(basename "$ref_file" .ppm)"  # e.g. "page-001"
    new_file="${NEW_DIR}/${page_tag}.ppm"
    diff_file="${DIFF_DIR}/${page_tag}-diff.png"

    if [[ ! -f "$new_file" ]]; then
        printf "MISSING  %s — pdf-raster produced no output\n" "$page_tag"
        fail=$((fail + 1))
        continue
    fi

    # Normalise dimensions: pdftoppm and pdf-raster may differ by ±1 pixel due
    # to rounding; resize the new image to match the reference before diffing.
    ref_dim="$(identify -format "%wx%h" "$ref_file" 2>/dev/null)"
    new_dim="$(identify -format "%wx%h" "$new_file" 2>/dev/null)"
    cmp_file="$new_file"
    if [[ "$ref_dim" != "$new_dim" ]]; then
        resized="${WORK_DIR}/resized.ppm"
        convert "$new_file" -resize "${ref_dim}!" -filter Point "$resized"
        cmp_file="$resized"
    fi

    # compare exits 1 when images differ; redirect stderr to capture metric line.
    rmse_line="$(compare -metric RMSE "$ref_file" "$cmp_file" "$diff_file" 2>&1 || true)"

    # Output format: "<absolute> (<normalized>)" e.g. "3.14 (0.0123)"
    rmse_norm="$(printf '%s' "$rmse_line" | grep -oP '(?<=\()[\d.e+-]+(?=\))' || echo "0")"

    # Scale normalised [0,1] RMSE to the more intuitive 0-255 range.
    rmse_255="$(bc -l <<< "scale=4; ${rmse_norm} * 255")"

    total_rmse="$(bc -l <<< "scale=4; ${total_rmse} + ${rmse_255}")"

    over_threshold="$(bc -l <<< "${rmse_255} > ${THRESHOLD}")"
    if [[ "$over_threshold" -eq 1 ]]; then
        printf "FAIL     %-12s  RMSE=%6s/255  (limit=%s)  diff→%s\n" \
            "$page_tag" "$rmse_255" "$THRESHOLD" "$diff_file"
        fail=$((fail + 1))
    else
        $VERBOSE && printf "OK       %-12s  RMSE=%6s/255\n" "$page_tag" "$rmse_255"
        pass=$((pass + 1))
    fi

    is_worse="$(bc -l <<< "${rmse_255} > ${max_rmse}")"
    if [[ "$is_worse" -eq 1 ]]; then
        max_rmse="$rmse_255"
        worst_page="$page_tag"
    fi
done

total=$((pass + fail))
avg_rmse="0"
[[ $total -gt 0 ]] && avg_rmse="$(bc -l <<< "scale=4; ${total_rmse} / ${total}")"

echo ""
echo "── Summary ──────────────────────────────────────────────────────────────"
printf "  PDF:       %s\n" "$(basename "$PDF")"
printf "  DPI:       %s\n" "$DPI"
printf "  Pages:     %d total, %d passed, %d failed\n" "$total" "$pass" "$fail"
printf "  Avg RMSE:  %.4f / 255\n" "$avg_rmse"
printf "  Max RMSE:  %.4f / 255  (%s)\n" "$max_rmse" "${worst_page:-none}"
printf "  Threshold: %.1f / 255\n" "$THRESHOLD"
[[ -n "$OUT_DIR" ]] && printf "  Diff dir:  %s\n" "$DIFF_DIR"
echo "─────────────────────────────────────────────────────────────────────────"

if [[ $fail -eq 0 ]]; then
    echo "PASSED"
else
    echo "FAILED"
    exit 1
fi
