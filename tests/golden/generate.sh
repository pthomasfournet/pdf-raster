#!/usr/bin/env bash
# generate.sh — regenerate golden reference images for cargo test --test golden.
#
# Renders each fixture PDF with the release pdf-raster binary and writes the
# output into tests/golden/ref/.  Commit the resulting PPMs so that the golden
# tests have stable references to compare against.
#
# Run this script whenever the rendering output legitimately changes (e.g.
# after a correctness fix) and inspect the diff before committing.
#
# Usage:
#   generate.sh [OPTIONS]
#
# Options:
#   -d    Dry run: print commands without executing
#   -h    Show this help text
#
# Requires: pdf-raster release binary (cargo build --release -p pdf-raster)
#
# IMPORTANT: The case table below MUST be kept in sync with the CASES array in
# crates/cli/tests/golden.rs.  Field order: pdf_basename ref_prefix dpi first last.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RASTER_BIN="${REPO_ROOT}/target/release/pdf-raster"
FIXTURES_DIR="${REPO_ROOT}/tests/fixtures"
REF_DIR="${SCRIPT_DIR}/ref"
DRY_RUN=false

usage() {
    grep '^#' "$0" | grep -v '^#!/' | sed 's/^# \{0,2\}//'
    exit 0
}

die() { echo "Error: $*" >&2; exit 1; }

while getopts ":dh" opt; do
    case $opt in
        d) DRY_RUN=true ;;
        h) usage ;;
        :) die "option -$OPTARG requires an argument." ;;
       \?) die "unknown option -$OPTARG." ;;
    esac
done

[[ -x "$RASTER_BIN" ]] || {
    echo "Error: pdf-raster not built at $RASTER_BIN" >&2
    echo "  Run: cargo build --release -p pdf-raster" >&2
    exit 1
}

# ── case table ────────────────────────────────────────────────────────────────
# Format: "pdf_basename  ref_prefix  dpi  first  last"
# MUST match the CASES array in crates/cli/tests/golden.rs.
#
# Fixture PDFs are gitignored — place your own PDFs in tests/fixtures/ and
# add corresponding entries here and in the CASES array in golden.rs.
#
# Example:
#   "my-document.pdf  my-document-72  72  1  3"
declare -a CASES=(
    # Add entries here matching the CASES array in crates/cli/tests/golden.rs.
)

if $DRY_RUN; then
    echo "DRY RUN — commands that would be executed:"
    echo ""
    for entry in "${CASES[@]}"; do
        read -r pdf prefix dpi first last <<< "$entry"
        pdf_path="${FIXTURES_DIR}/${pdf}"
        out_prefix="${REF_DIR}/${prefix}"
        echo "  $RASTER_BIN -r $dpi -f $first -l $last $pdf_path $out_prefix"
    done
    exit 0
fi

mkdir -p "$REF_DIR"

for entry in "${CASES[@]}"; do
    read -r pdf prefix dpi first last <<< "$entry"

    # Validate that all five fields were parsed — catches accidental whitespace
    # corruption of the case table.
    [[ -n "$pdf" && -n "$prefix" && -n "$dpi" && -n "$first" && -n "$last" ]] \
        || die "malformed case table entry: '$entry'"

    pdf_path="${FIXTURES_DIR}/${pdf}"
    out_prefix="${REF_DIR}/${prefix}"

    [[ -f "$pdf_path" ]] || die "fixture not found: $pdf_path"

    echo "Rendering ${pdf} pages ${first}-${last} @${dpi}dpi → ${out_prefix}-*.ppm"

    # Capture stderr; re-emit on failure so crashes are not silently swallowed.
    stderr_log="$(mktemp)"
    if ! "$RASTER_BIN" -r "$dpi" -f "$first" -l "$last" \
            "$pdf_path" "$out_prefix" 2>"$stderr_log"; then
        echo "Error: pdf-raster failed for ${pdf}:" >&2
        cat "$stderr_log" >&2
        rm -f "$stderr_log"
        exit 1
    fi
    rm -f "$stderr_log"

    # Verify that at least one output file was created.
    shopt -s nullglob
    generated=( "${out_prefix}"-*.ppm )
    shopt -u nullglob
    [[ ${#generated[@]} -gt 0 ]] \
        || die "pdf-raster produced no output for ${pdf} (prefix: ${out_prefix})"
done

echo ""
echo "Done. Files in ${REF_DIR}/:"
ls -lh "${REF_DIR}/"
echo ""
echo "Review changes with: git diff tests/golden/ref/"
echo "Then commit: git add tests/golden/ref/ && git commit -m 'golden: regenerate references'"
