#!/usr/bin/env bash
# Phase-11 contest re-bench, parameterised by build features + backend.
# Lets us compare CUDA-with-cache vs CUDA-without-cache vs Vulkan with
# the same archives and harness shape as run_contest_v11_gpu.sh.
#
# Usage:
#   scripts/run_contest_v11_variant.sh <label> <features> <backend>
#
# Examples:
#   scripts/run_contest_v11_variant.sh cuda-nocache "gpu-aa,gpu-icc" cuda
#   scripts/run_contest_v11_variant.sh vulkan      "vulkan,gpu-aa,gpu-icc" vulkan
#
#   label    — directory name under bench/ (results go to bench/v11-<label>/)
#   features — comma-list passed to --features on pdf_raster (no spaces)
#   backend  — value for CONTEST_BACKEND env var (cuda|vulkan|cpu|auto)

set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "usage: $0 <label> <features> <backend>" >&2
    exit 1
fi

LABEL="$1"
FEATURES="$2"
BACKEND="$3"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

ARCHIVE_DIR="/media/tom/Storage1/pdf-raster-bench/phase11"
ARCHIVE="$ARCHIVE_DIR/archive_10gb.pdf"
ARCHIVES_LIST="$ARCHIVE_DIR/archives.txt"
RESULTS_DIR="$REPO_ROOT/bench/v11-$LABEL"
LOG="$RESULTS_DIR/run.log"
CSV="$RESULTS_DIR/results.csv"
MD="$RESULTS_DIR/results.md"
E5_FIXTURE="$REPO_ROOT/tests/fixtures/corpus-08-scan-dct-1927.pdf"
E5_PAGE=100
E5_DPI=300

# Translate the bench feature set into per-crate feature args for both
# binaries.  pdf_raster expects the bare names (no namespace); the bench
# binary expects them prefixed with `pdf_raster/` because it depends on
# pdf_raster transitively.
PR_FEATURES="$FEATURES"
BENCH_FEATURES=$(echo "$FEATURES" | tr ',' '\n' | sed 's|^|pdf_raster/|' | paste -sd, -)

mkdir -p "$RESULTS_DIR"
exec > >(tee "$LOG") 2>&1

echo "==========================================================="
echo "Phase 11 variant '$LABEL' starting $(date -Iseconds)"
echo "Features:        $PR_FEATURES (pdf_raster), $BENCH_FEATURES (bench)"
echo "CONTEST_BACKEND: $BACKEND"
echo "==========================================================="

# --- Step 1: Build with the requested features ---
echo
echo "==> [1/5] Building binaries"
cargo build --release \
    -p bench --bin contest_v11 \
    --features "$BENCH_FEATURES" -q
cargo build --release \
    -p pdf-raster \
    --features "$PR_FEATURES" -q

CONTEST="./target/release/contest_v11"
RASTER="./target/release/pdf-raster"
ls -la "$CONTEST" "$RASTER"

# --- Step 2: Verify archives exist ---
echo
echo "==> [2/5] Verifying archive set"
if [[ ! -f "$ARCHIVE" ]]; then
    echo "error: $ARCHIVE missing — run scripts/run_contest_v11.sh first to build archives"
    exit 1
fi
if [[ ! -f "$ARCHIVES_LIST" ]]; then
    echo "error: $ARCHIVES_LIST missing"
    exit 1
fi
ARCHIVE_BYTES="$(stat -c %s "$ARCHIVE")"
ARCHIVE_GB="$(echo "scale=2; $ARCHIVE_BYTES / 1073741824" | bc)"
LIST_COUNT="$(wc -l < "$ARCHIVES_LIST")"
echo "    archive: $ARCHIVE_GB GB, cross-doc set: $LIST_COUNT"

# --- Step 3: Parity check (corpus-04 page 100, md5 must match) ---
# Note: parity check only runs in Auto/CUDA mode; the CLI's --backend flag
# defaults to Auto and we don't override it for the parity binary, so this
# tests the CPU/CUDA path regardless of CONTEST_BACKEND.  Skip on vulkan
# since the CLI's backend wiring would need a separate parity baseline.
echo
echo "==> [3/5] Parity check (corpus-04 p100 @ 150 DPI)"
if [[ "$BACKEND" == "vulkan" ]]; then
    echo "    skipped (vulkan path has separate parity expectations)"
else
    PARITY="/tmp/p11_${LABEL}_parity"
    "$RASTER" tests/fixtures/corpus-04-ebook-mixed.pdf "$PARITY" -f 100 -l 100 -r 150 >/dev/null
    EXPECTED="6c5703a00b2abd45b8c7ebbc31b54ba8"
    GOT="$(md5sum "${PARITY}-100.ppm" | awk '{print $1}')"
    rm -f "${PARITY}-100.ppm"
    if [[ "$GOT" != "$EXPECTED" ]]; then
        echo "error: parity check FAILED — got $GOT, expected $EXPECTED"
        exit 1
    fi
    echo "    parity OK ($GOT)"
fi

# --- Step 4: Run events ---
echo
echo "==> [4/5] Running events"
echo "event,kind,iter,elapsed_ms,competitor,competitor_ms" > "$CSV"

parse_ours()  { grep -oE 'ours    : [0-9.]+ ms' "$1" | head -1 | grep -oE '[0-9.]+'; }
parse_event() { grep -oE '^E[1-4]: [0-9.]+ ms' "$1" | head -1 | grep -oE '[0-9.]+ ms' | grep -oE '[0-9.]+'; }
parse_competitor() { grep -oE "$2 *: [0-9.]+ ms" "$1" | head -1 | grep -oE '[0-9.]+'; }

E1_PAGE=50000
E2_FIRST=50000

# E1
echo "    E1 (first-pixel, page $E1_PAGE)"
for kind in cold warm warm warm warm; do
    OUT="$RESULTS_DIR/.e1_${kind}_$RANDOM.log"
    CONTEST_BACKEND="$BACKEND" "$CONTEST" e1 "$ARCHIVE" "$E1_PAGE" > "$OUT" 2>&1
    OURS_MS="$(parse_ours "$OUT" || echo NA)"
    MU_MS="$(parse_competitor "$OUT" mutool || echo NA)"
    PP_MS="$(parse_competitor "$OUT" pdftoppm || echo NA)"
    echo "        $kind: ours=$OURS_MS  mutool=$MU_MS  pdftoppm=$PP_MS"
    echo "E1,$kind,1,$OURS_MS,mutool,$MU_MS"   >> "$CSV"
    echo "E1,$kind,1,$OURS_MS,pdftoppm,$PP_MS" >> "$CSV"
    rm -f "$OUT"
done

# E2
echo "    E2 (sustained, pages $E2_FIRST-$((E2_FIRST + 99)))"
for kind in cold warm warm warm warm; do
    OUT="$RESULTS_DIR/.e2_${kind}_$RANDOM.log"
    CONTEST_BACKEND="$BACKEND" "$CONTEST" e2 "$ARCHIVE" "$E2_FIRST" 100 > "$OUT" 2>&1
    OURS_MS="$(parse_event "$OUT" || echo NA)"
    echo "        $kind: $OURS_MS ms"
    echo "E2,$kind,1,$OURS_MS,," >> "$CSV"
    rm -f "$OUT"
done

# E3
echo "    E3 (cross-doc, $LIST_COUNT archives)"
for kind in cold warm warm warm warm; do
    OUT="$RESULTS_DIR/.e3_${kind}_$RANDOM.log"
    CONTEST_BACKEND="$BACKEND" "$CONTEST" e3 "$ARCHIVES_LIST" > "$OUT" 2>&1
    OURS_MS="$(parse_event "$OUT" || echo NA)"
    echo "        $kind: $OURS_MS ms"
    echo "E3,$kind,1,$OURS_MS,," >> "$CSV"
    rm -f "$OUT"
done

# E4
echo "    E4 (random-access, 1000 pages)"
for kind in cold warm warm warm warm; do
    OUT="$RESULTS_DIR/.e4_${kind}_$RANDOM.log"
    CONTEST_BACKEND="$BACKEND" "$CONTEST" e4 "$ARCHIVE" > "$OUT" 2>&1
    OURS_MS="$(parse_event "$OUT" || echo NA)"
    echo "        $kind: $OURS_MS ms"
    echo "E4,$kind,1,$OURS_MS,," >> "$CSV"
    rm -f "$OUT"
done

# E5 — wraps the standalone CLI; needs --backend on the CLI to honor it.
echo "    E5 (DCT-heavy single page, corpus-08 p$E5_PAGE @ ${E5_DPI} DPI)"
E5_OUR_TMPL="/tmp/p11_${LABEL}_e5_ours"
E5_MU_OUT="/tmp/p11_${LABEL}_e5_mu.ppm"
E5_PP_TMPL="/tmp/p11_${LABEL}_e5_pp"

# Map CONTEST_BACKEND value to the CLI's --backend flag value.
# Use a bash array — a bare string would need word-splitting at the use
# site, which exposes the args to glob expansion.
CLI_BACKEND_ARGS=()
case "$BACKEND" in
    cpu)    CLI_BACKEND_ARGS=(--backend cpu) ;;
    cuda)   CLI_BACKEND_ARGS=(--backend cuda) ;;
    vulkan) CLI_BACKEND_ARGS=(--backend vulkan) ;;
    auto|*) ;;  # empty array — let the CLI default kick in
esac

time_ms() {
    local t0 t1
    t0=$(date +%s%N)
    "$@" >/dev/null 2>&1
    t1=$(date +%s%N)
    echo "scale=1; ($t1 - $t0) / 1000000" | bc
}

for kind in cold warm warm warm warm; do
    rm -f "${E5_OUR_TMPL}-${E5_PAGE}.ppm" "$E5_MU_OUT" "${E5_PP_TMPL}-${E5_PAGE}.ppm"
    OURS_MS=$(time_ms "$RASTER" "${CLI_BACKEND_ARGS[@]}" "$E5_FIXTURE" "$E5_OUR_TMPL" -f $E5_PAGE -l $E5_PAGE -r $E5_DPI)
    MU_MS=$(time_ms mutool draw -q -P -N -r $E5_DPI -F ppm -o "$E5_MU_OUT" "$E5_FIXTURE" $E5_PAGE)
    PP_MS=$(time_ms pdftoppm -r $E5_DPI -f $E5_PAGE -l $E5_PAGE "$E5_FIXTURE" "$E5_PP_TMPL")
    echo "        $kind: ours=$OURS_MS  mutool=$MU_MS  pdftoppm=$PP_MS"
    echo "E5,$kind,1,$OURS_MS,mutool,$MU_MS"   >> "$CSV"
    echo "E5,$kind,1,$OURS_MS,pdftoppm,$PP_MS" >> "$CSV"
done
rm -f "${E5_OUR_TMPL}-${E5_PAGE}.ppm" "$E5_MU_OUT" "${E5_PP_TMPL}-${E5_PAGE}.ppm"

# --- Step 5: Generate results.md ---
echo
echo "==> [5/5] Writing results.md"

awk_median() {
    awk -F, -v ev="$1" '
        $1 == ev && $2 == "warm" && $4 != "NA" { print $4 }
    ' "$CSV" | sort -n | awk '
        { a[NR] = $1 }
        END { if (NR == 0) print "NA"; else if (NR % 2) print a[(NR+1)/2]; else printf "%.1f", (a[NR/2] + a[NR/2+1]) / 2 }
    '
}
awk_cold() {
    awk -F, -v ev="$1" '
        $1 == ev && $2 == "cold" && $4 != "NA" { print $4; exit }
    ' "$CSV"
}
awk_competitor_median() {
    awk -F, -v ev="$1" -v c="$2" '
        $1 == ev && $2 == "warm" && $5 == c && $6 != "NA" { print $6 }
    ' "$CSV" | sort -n | awk '
        { a[NR] = $1 }
        END { if (NR == 0) print "NA"; else if (NR % 2) print a[(NR+1)/2]; else printf "%.1f", (a[NR/2] + a[NR/2+1]) / 2 }
    '
}

E1_COLD="$(awk_cold E1)"; E1_WARM="$(awk_median E1)"
E1_MU="$(awk_competitor_median E1 mutool)"; E1_PP="$(awk_competitor_median E1 pdftoppm)"
E2_COLD="$(awk_cold E2)"; E2_WARM="$(awk_median E2)"
E3_COLD="$(awk_cold E3)"; E3_WARM="$(awk_median E3)"
E4_COLD="$(awk_cold E4)"; E4_WARM="$(awk_median E4)"
E5_COLD="$(awk_cold E5)"; E5_WARM="$(awk_median E5)"
E5_MU="$(awk_competitor_median E5 mutool)"; E5_PP="$(awk_competitor_median E5 pdftoppm)"

CPU_INFO="$(grep -m1 'model name' /proc/cpuinfo | sed 's/^.*: //')"
GPU_INFO="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "no NVIDIA GPU detected")"
KERNEL="$(uname -r)"

cat > "$MD" <<EOF
# Phase 11 contest results — variant '$LABEL'

**Run:** $(date -Iseconds)
**Hardware:** $CPU_INFO + $GPU_INFO, Linux $KERNEL
**Archive:** ${ARCHIVE_GB} GB synthetic PDF
**Cross-doc set:** $LIST_COUNT archives at ~300 MB each
**Methodology:** 1 cold run + 4 warm runs per event; warm median reported.
**Build features:** \`$PR_FEATURES\` on pdf_raster (no PGO)
**Backend:** \`CONTEST_BACKEND=$BACKEND\` on the bench, \`${CLI_BACKEND_ARGS[*]:---backend auto (default)}\` on the standalone CLI for E5

## E1 — first-pixel (page $E1_PAGE)

| Engine | Cold (ms) | Warm median (ms) |
|---|---|---|
| pdf-raster | $E1_COLD | **$E1_WARM** |
| mutool draw | — | $E1_MU |
| pdftoppm | — | $E1_PP |

## E2 — sustained (pages $E2_FIRST–$((E2_FIRST + 99)))

| Engine | Cold (ms) | Warm median (ms) | Per-page (ms) |
|---|---|---|---|
| pdf-raster | $E2_COLD | **$E2_WARM** | $(awk -v w="$E2_WARM" 'BEGIN{ printf "%.1f", w/100 }') |

## E3 — cross-doc ($LIST_COUNT archives, page 1 each)

| Engine | Cold (ms) | Warm median (ms) | Per-archive (ms) |
|---|---|---|---|
| pdf-raster | $E3_COLD | **$E3_WARM** | $(awk -v w="$E3_WARM" -v n="$LIST_COUNT" 'BEGIN{ printf "%.1f", w/n }') |

## E4 — random-access (1000 pages)

| Engine | Cold (ms) | Warm median (ms) | Per-page (ms) |
|---|---|---|---|
| pdf-raster | $E4_COLD | **$E4_WARM** | $(awk -v w="$E4_WARM" 'BEGIN{ printf "%.1f", w/1000 }') |

## E5 — single DCT page (corpus-08 p$E5_PAGE @ ${E5_DPI} DPI)

| Engine | Cold (ms) | Warm median (ms) |
|---|---|---|
| pdf-raster | $E5_COLD | **$E5_WARM** |
| mutool draw | — | $E5_MU |
| pdftoppm | — | $E5_PP |

---

Raw data: \`bench/v11-$LABEL/results.csv\`.  Full run log: \`bench/v11-$LABEL/run.log\`.
EOF

echo
echo "==========================================================="
echo "Variant '$LABEL' complete $(date -Iseconds)"
echo "Results: $MD"
echo "==========================================================="
