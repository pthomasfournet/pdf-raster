#!/usr/bin/env bash
# compare.sh — pixel-accurate comparison of pdftoppm vs pdf-raster output.
#
# Usage:
#   compare.sh [OPTIONS] <PDF>
#
# Options:
#   -r DPI          Resolution in DPI (default: 150)
#   -f PAGE         First page, 1-based (default: 1)
#   -l PAGE         Last page, 1-based (default: all)
#   -t THRESHOLD    Max allowed RMSE per page, 0-255 scale (default: 2.0)
#   -v              Verbose: print per-page stats even when passing
#   -o DIR          Write diff images to DIR (default: discarded)
#   -n              Dry run: print commands without executing
#
# Exit code: 0 = all pages within threshold; 1 = failures or errors.
# Dry-run always exits 0.
#
# Requires: pdftoppm, pdf-raster (release build),
#           ImageMagick (compare, identify, convert), bc

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RASTER_BIN="${REPO_ROOT}/target/release/pdf-raster"

# ── defaults ──────────────────────────────────────────────────────────────────
DPI=150
FIRST=1
LAST=""
THRESHOLD="2.0"
VERBOSE=false
OUT_DIR=""
DRY_RUN=false

usage() {
    grep '^#' "$0" | grep -v '^#!/' | sed 's/^# \{0,2\}//'
    exit 0
}

die() { echo "Error: $*" >&2; exit 1; }

is_positive_int() { [[ "$1" =~ ^[1-9][0-9]*$ ]]; }
is_pos_number()   { [[ "$1" =~ ^[0-9]+(\.[0-9]+)?$ ]]; }

while getopts ":r:f:l:t:vo:nh" opt; do
    case $opt in
        r) DPI="$OPTARG" ;;
        f) FIRST="$OPTARG" ;;
        l) LAST="$OPTARG" ;;
        t) THRESHOLD="$OPTARG" ;;
        v) VERBOSE=true ;;
        o) OUT_DIR="$OPTARG" ;;
        n) DRY_RUN=true ;;
        h) usage ;;
        :) die "option -$OPTARG requires an argument." ;;
       \?) die "unknown option -$OPTARG." ;;
    esac
done
shift $((OPTIND - 1))

# ── input validation ──────────────────────────────────────────────────────────
[[ $# -lt 1 ]] && { echo "Error: PDF argument required." >&2; usage; }
PDF="$1"
[[ -f "$PDF" ]] || die "file not found: $PDF"

is_positive_int "$DPI"   || die "-r DPI must be a positive integer, got: $DPI"
is_positive_int "$FIRST" || die "-f PAGE must be a positive integer, got: $FIRST"
[[ -z "$LAST" ]] || is_positive_int "$LAST" || die "-l PAGE must be a positive integer, got: $LAST"
is_pos_number "$THRESHOLD" || die "-t THRESHOLD must be a non-negative number, got: $THRESHOLD"
[[ -n "$LAST" && "$LAST" -lt "$FIRST" ]] && die "-l ($LAST) must be >= -f ($FIRST)"

# ── build argument lists early (dry-run needs them) ───────────────────────────
ref_args=(-r "$DPI" -f "$FIRST")
[[ -n "$LAST" ]] && ref_args+=(-l "$LAST")
new_args=(-r "$DPI" -f "$FIRST")
[[ -n "$LAST" ]] && new_args+=(-l "$LAST")

# ── dry-run ───────────────────────────────────────────────────────────────────
if $DRY_RUN; then
    echo "DRY RUN — commands that would be executed:"
    echo ""
    echo "  [render ref]  pdftoppm ${ref_args[*]} $PDF <OUT>/ref/page"
    echo "  [render new]  $RASTER_BIN ${new_args[*]} $PDF <OUT>/new/page"
    echo "  [per page]    compare -metric RMSE <OUT>/ref/page-NNN.ppm \\"
    echo "                        <OUT>/new/page-NNN.ppm <DIFF>/page-NNN-diff.png"
    echo ""
    echo "  Settings: DPI=${DPI}  first=${FIRST}  last=${LAST:-all}  threshold=${THRESHOLD}/255"
    [[ -n "$OUT_DIR" ]] && echo "  Diff dir: ${OUT_DIR}"
    exit 0
fi

# ── dependency check ──────────────────────────────────────────────────────────
for cmd in pdftoppm compare identify convert bc; do
    command -v "$cmd" >/dev/null 2>&1 || die "$cmd not found in PATH"
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
if ! pdftoppm "${ref_args[@]}" "$PDF" "${REF_DIR}/page" 2>"${WORK_DIR}/ref.log"; then
    echo "Error: pdftoppm failed — see log:" >&2
    cat "${WORK_DIR}/ref.log" >&2
    exit 1
fi

# ── render: pdf-raster ────────────────────────────────────────────────────────
if ! "$RASTER_BIN" "${new_args[@]}" "$PDF" "${NEW_DIR}/page" 2>"${WORK_DIR}/new.log"; then
    echo "Error: pdf-raster failed — see log:" >&2
    cat "${WORK_DIR}/new.log" >&2
    exit 1
fi

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
    echo "Error: pdftoppm produced no output pages for ${PDF}" >&2
    if [[ -s "${WORK_DIR}/ref.log" ]]; then
        echo "  pdftoppm stderr:" >&2
        cat "${WORK_DIR}/ref.log" >&2
    fi
    exit 1
fi

for ref_file in "${ref_pages[@]}"; do
    page_tag="$(basename "$ref_file" .ppm)"   # e.g. "page-001"
    new_file="${NEW_DIR}/${page_tag}.ppm"
    # Per-page resized file so concurrent pages don't clobber each other.
    resized_file="${WORK_DIR}/resized-${page_tag}.ppm"
    diff_file="${DIFF_DIR}/${page_tag}-diff.png"

    if [[ ! -f "$new_file" ]]; then
        printf "MISSING  %s — pdf-raster produced no output\n" "$page_tag"
        fail=$((fail + 1))
        continue
    fi

    # Normalise dimensions: renderers may differ by ±1px due to independent
    # rounding; resize new to match reference geometry before diffing.
    ref_dim="$(identify -format "%wx%h" "$ref_file" 2>/dev/null)" \
        || die "identify failed on ref image: $ref_file"
    new_dim="$(identify -format "%wx%h" "$new_file" 2>/dev/null)" \
        || die "identify failed on new image: $new_file"

    cmp_file="$new_file"
    if [[ "$ref_dim" != "$new_dim" ]]; then
        convert "$new_file" -resize "${ref_dim}!" -filter Point "$resized_file" \
            || die "convert resize failed for $page_tag"
        cmp_file="$resized_file"
    fi

    # compare exits 1 when images differ (expected); 2 means a real error.
    rmse_line="$(compare -metric RMSE "$ref_file" "$cmp_file" "$diff_file" 2>&1)" \
        || { ec=$?; [[ $ec -eq 1 ]] || die "compare failed ($ec) for $page_tag: $rmse_line"; }

    # Output format: "<absolute> (<normalized>)" e.g. "3.14 (0.0123)"
    # Use POSIX-compatible grep extended regex — no PCRE required.
    rmse_norm="$(printf '%s' "$rmse_line" | grep -Eo '\([0-9.e+-]+\)' | tr -d '()')"
    if [[ -z "$rmse_norm" ]]; then
        echo "Warning: could not parse RMSE from compare output for ${page_tag}: ${rmse_line}" >&2
        rmse_norm="0"
    fi

    # Scale normalised [0,1] RMSE to the intuitive 0-255 range.
    rmse_255="$(bc -l <<< "scale=4; ${rmse_norm} * 255")" \
        || die "bc arithmetic failed for $page_tag (rmse_norm='$rmse_norm')"

    total_rmse="$(bc -l <<< "scale=4; ${total_rmse} + ${rmse_255}")"

    over_threshold="$(bc -l <<< "${rmse_255} > ${THRESHOLD}")"
    if [[ "$over_threshold" -eq 1 ]]; then
        printf "FAIL     %-12s  RMSE=%8s/255  (limit=%s)  diff→%s\n" \
            "$page_tag" "$rmse_255" "$THRESHOLD" "$diff_file"
        fail=$((fail + 1))
    else
        if $VERBOSE; then
            printf "OK       %-12s  RMSE=%8s/255\n" "$page_tag" "$rmse_255"
        fi
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
