#!/usr/bin/env bash
# Phase-11 contest driver.  Runs the full four-event sweep against
# a 10 GB synthetic archive, with cold + warm timings per event.
#
# Output:
#   bench/v11/results.md   — narrative + tables
#   bench/v11/results.csv  — machine-readable
#   bench/v11/run.log      — full stdout/stderr stream

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

ARCHIVE_DIR="/media/tom/Storage1/pdf-raster-bench/phase11"
ARCHIVE="$ARCHIVE_DIR/archive_10gb.pdf"
CROSS_DOC_DIR="$ARCHIVE_DIR/cross_doc"
ARCHIVES_LIST="$ARCHIVE_DIR/archives.txt"
RESULTS_DIR="$REPO_ROOT/bench/v11"
LOG="$RESULTS_DIR/run.log"
CSV="$RESULTS_DIR/results.csv"
MD="$RESULTS_DIR/results.md"

mkdir -p "$ARCHIVE_DIR" "$CROSS_DOC_DIR" "$RESULTS_DIR"

# Pipe everything to the log file in addition to stdout.
exec > >(tee -a "$LOG") 2>&1

echo "==========================================================="
echo "Phase 11 contest run starting $(date -Iseconds)"
echo "==========================================================="

# --- Step 1: Build the contest binary (PGO-trained pdf_raster lib) ---
# We PGO-train on pdf-raster (CLI), then rebuild contest_v11 with
# -Cprofile-use= so the shared pdf_raster library code benefits.
echo
echo "==> [1/5] Building PGO-trained binaries"

PROFDIR="$(mktemp -d -t pdf_raster_pgo.XXXXXX)"
trap 'rm -rf "$PROFDIR"' EXIT

SYSROOT="$(rustc --print sysroot)"
HOST_TRIPLE="$(rustc -vV | sed -n 's|host: ||p')"
PROFDATA="$SYSROOT/lib/rustlib/$HOST_TRIPLE/bin/llvm-profdata"
if [[ ! -x "$PROFDATA" ]]; then
    if command -v llvm-profdata >/dev/null 2>&1; then
        PROFDATA="$(command -v llvm-profdata)"
    else
        echo "error: llvm-profdata not found." >&2
        echo "       rustup component add llvm-tools-preview" >&2
        exit 1
    fi
fi

# 1a. Instrument-build pdf-raster CLI.
echo "    [1a] PGO-instrument build of pdf-raster"
RUSTFLAGS="-Cprofile-generate=$PROFDIR" \
    cargo build --release -p pdf-raster -q

# 1b. Train: render a representative sample (10 pages of corpus-04).
echo "    [1b] Training profile on corpus-04 pages 1-10"
TRAIN_PFX="$(mktemp -d -t pgo_train.XXXXXX)"
./target/release/pdf-raster \
    tests/fixtures/corpus-04-ebook-mixed.pdf \
    "$TRAIN_PFX/page" -f 1 -l 10 >/dev/null
rm -rf "$TRAIN_PFX"

# 1c. Merge profile data.
echo "    [1c] Merging PGO profile"
"$PROFDATA" merge -o "$PROFDIR/merged.profdata" "$PROFDIR"
RAW_COUNT="$(find "$PROFDIR" -name '*.profraw' | wc -l)"
echo "         collected $RAW_COUNT .profraw file(s)"

# 1d. Rebuild both binaries with profile-use.
echo "    [1d] PGO-guided rebuild of pdf-raster + contest_v11"
RUSTFLAGS="-Cprofile-use=$PROFDIR/merged.profdata -Cllvm-args=-pgo-warn-missing-function" \
    cargo build --release -p pdf-raster -p bench -q

# Verify both binaries exist.
ls -la ./target/release/pdf-raster ./target/release/contest_v11 || {
    echo "error: build did not produce both binaries"
    exit 1
}

CONTEST="./target/release/contest_v11"

# --- Step 2: Build the main archive (~2.5 GB output) ---
echo
echo "==> [2/5] Building main archive"
# qpdf concat with renumbered objects per cycle now produces output
# matching input bytes; pass exact target.
TARGET_OUTPUT_BYTES=$(( (5 * 1024 * 1024 * 1024) / 2 ))  # 2.5 GiB

if [[ -f "$ARCHIVE" ]]; then
    SIZE_BYTES="$(stat -c %s "$ARCHIVE")"
    SIZE_GB="$(echo "scale=2; $SIZE_BYTES / 1073741824" | bc)"
    echo "    Archive already exists: $SIZE_GB GB — reusing"
else
    "$CONTEST" build-archive "$ARCHIVE" "$TARGET_OUTPUT_BYTES"
fi

ARCHIVE_BYTES="$(stat -c %s "$ARCHIVE")"
ARCHIVE_GB="$(echo "scale=2; $ARCHIVE_BYTES / 1073741824" | bc)"
echo "    Archive: $ARCHIVE_BYTES bytes ($ARCHIVE_GB GB)"

# --- Step 3: Build 100 cross-doc archives for E3 ---
echo
echo "==> [3/5] Building 100 cross-doc archives"
EXISTING="$(ls "$CROSS_DOC_DIR" 2>/dev/null | wc -l)"
if [[ "$EXISTING" -ge 100 ]]; then
    echo "    Cross-doc set already exists: $EXISTING files — reusing"
else
    # 100 archives × ~310 MB each (one cycle = 4 distinct fixtures).
    PER_OUT_BYTES=$((300 * 1024 * 1024))
    for i in $(seq -w 0 99); do
        OUT="$CROSS_DOC_DIR/archive_$i.pdf"
        if [[ ! -f "$OUT" ]]; then
            "$CONTEST" build-archive "$OUT" "$PER_OUT_BYTES" 2>&1 | tail -1
        fi
    done
fi

# Generate the archives.txt list.
find "$CROSS_DOC_DIR" -name 'archive_*.pdf' | sort > "$ARCHIVES_LIST"
LIST_COUNT="$(wc -l < "$ARCHIVES_LIST")"
echo "    archives.txt has $LIST_COUNT entries"

# Pick a mid-archive page index for E1/E2.  contest_v11 clamps to
# total_pages, so passing a large number is also safe.
E1_PAGE=50000
E2_FIRST=50000
echo "    using E1 page=$E1_PAGE (clamped to total_pages by harness), E2 first=$E2_FIRST"

# --- Step 4: Run events (cold + warm) ---
echo
echo "==> [4/5] Running events"

# CSV header.
echo "event,kind,iter,elapsed_ms,competitor,competitor_ms" > "$CSV"

# Helper: extract our timing from contest_v11 output.
parse_ours() {
    grep -oE 'ours    : [0-9.]+ ms' "$1" | head -1 | grep -oE '[0-9.]+'
}
parse_event() {
    # Matches "E1: 4.2 ms" / "E2: 21.0 ms (100 pages)" / etc.
    grep -oE '^E[1-4]: [0-9.]+ ms' "$1" | head -1 | grep -oE '[0-9.]+ ms' | grep -oE '[0-9.]+'
}
parse_competitor() {
    # First arg: log file. Second: name (mutool|pdftoppm).
    grep -oE "$2 *: [0-9.]+ ms" "$1" | head -1 | grep -oE '[0-9.]+'
}

# E1: cold then 4 warm.
echo "    E1 (first-pixel, page $E1_PAGE)"
for kind in cold warm warm warm warm; do
    OUT="$RESULTS_DIR/.e1_${kind}_$RANDOM.log"
    "$CONTEST" e1 "$ARCHIVE" "$E1_PAGE" > "$OUT" 2>&1
    OURS_MS="$(parse_ours "$OUT" || echo NA)"
    MU_MS="$(parse_competitor "$OUT" mutool || echo NA)"
    PP_MS="$(parse_competitor "$OUT" pdftoppm || echo NA)"
    echo "        $kind: ours=$OURS_MS  mutool=$MU_MS  pdftoppm=$PP_MS"
    echo "E1,$kind,1,$OURS_MS,mutool,$MU_MS" >> "$CSV"
    echo "E1,$kind,1,$OURS_MS,pdftoppm,$PP_MS" >> "$CSV"
    rm -f "$OUT"
done

# E2: cold then 4 warm.
echo "    E2 (sustained, pages $E2_FIRST-$((E2_FIRST + 99)))"
for kind in cold warm warm warm warm; do
    OUT="$RESULTS_DIR/.e2_${kind}_$RANDOM.log"
    "$CONTEST" e2 "$ARCHIVE" "$E2_FIRST" 100 > "$OUT" 2>&1
    OURS_MS="$(parse_event "$OUT" || echo NA)"
    echo "        $kind: $OURS_MS ms"
    echo "E2,$kind,1,$OURS_MS,," >> "$CSV"
    rm -f "$OUT"
done

# E3: cold then 4 warm.
echo "    E3 (cross-doc, $LIST_COUNT archives)"
for kind in cold warm warm warm warm; do
    OUT="$RESULTS_DIR/.e3_${kind}_$RANDOM.log"
    "$CONTEST" e3 "$ARCHIVES_LIST" > "$OUT" 2>&1
    OURS_MS="$(parse_event "$OUT" || echo NA)"
    echo "        $kind: $OURS_MS ms"
    echo "E3,$kind,1,$OURS_MS,," >> "$CSV"
    rm -f "$OUT"
done

# E4: cold then 4 warm.
echo "    E4 (random-access, 1000 pages)"
for kind in cold warm warm warm warm; do
    OUT="$RESULTS_DIR/.e4_${kind}_$RANDOM.log"
    "$CONTEST" e4 "$ARCHIVE" > "$OUT" 2>&1
    OURS_MS="$(parse_event "$OUT" || echo NA)"
    echo "        $kind: $OURS_MS ms"
    echo "E4,$kind,1,$OURS_MS,," >> "$CSV"
    rm -f "$OUT"
done

# --- Step 5: Generate results.md ---
echo
echo "==> [5/5] Writing results.md"

# Compute medians from the CSV.  awk: warm runs only, group by event.
awk_median() {
    awk -F, -v ev="$1" '
        $1 == ev && $2 == "warm" && $4 != "NA" {
            print $4
        }
    ' "$CSV" | sort -n | awk '
        { a[NR] = $1 }
        END { if (NR == 0) print "NA"; else if (NR % 2) print a[(NR+1)/2]; else printf "%.1f", (a[NR/2] + a[NR/2+1]) / 2 }
    '
}
awk_cold() {
    awk -F, -v ev="$1" '
        $1 == ev && $2 == "cold" && $4 != "NA" {
            print $4
            exit
        }
    ' "$CSV"
}
awk_competitor_median() {
    # arg1: event, arg2: competitor name
    awk -F, -v ev="$1" -v c="$2" '
        $1 == ev && $2 == "warm" && $5 == c && $6 != "NA" {
            print $6
        }
    ' "$CSV" | sort -n | awk '
        { a[NR] = $1 }
        END { if (NR == 0) print "NA"; else if (NR % 2) print a[(NR+1)/2]; else printf "%.1f", (a[NR/2] + a[NR/2+1]) / 2 }
    '
}

E1_COLD="$(awk_cold E1)"
E1_WARM="$(awk_median E1)"
E1_MU="$(awk_competitor_median E1 mutool)"
E1_PP="$(awk_competitor_median E1 pdftoppm)"
E2_COLD="$(awk_cold E2)"
E2_WARM="$(awk_median E2)"
E3_COLD="$(awk_cold E3)"
E3_WARM="$(awk_median E3)"
E4_COLD="$(awk_cold E4)"
E4_WARM="$(awk_median E4)"

CPU_INFO="$(grep -m1 'model name' /proc/cpuinfo | sed 's/^.*: //')"
GPU_INFO="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "no NVIDIA GPU detected")"
KERNEL="$(uname -r)"

cat > "$MD" <<EOF
# Phase 11 contest results

**Run:** $(date -Iseconds)
**Hardware:** $CPU_INFO + $GPU_INFO, Linux $KERNEL
**Archive:** ${ARCHIVE_GB} GB synthetic PDF (qpdf-concatenated corpus fixtures)
**Cross-doc set:** $LIST_COUNT archives at ~300 MB each (~$(echo "scale=1; $LIST_COUNT * 0.3" | bc) GB total)
**Methodology:** 1 cold run + 4 warm runs per event; warm median reported.
PGO-trained binary (rendering 10 pages of corpus-04 as the training workload).

## E1 — first-pixel (page $E1_PAGE)

| Engine | Cold (ms) | Warm median (ms) |
|---|---|---|
| pdf-raster | $E1_COLD | **$E1_WARM** |
| mutool draw | — | $E1_MU |
| pdftoppm | — | $E1_PP |

(Competitor cold runs not collected — they invoke as subprocesses with their own startup cost on every call, so cold/warm distinction is muddied.)

## E2 — sustained (pages $E2_FIRST–$((E2_FIRST + 99)))

| Engine | Cold (ms) | Warm median (ms) |
|---|---|---|
| pdf-raster | $E2_COLD | **$E2_WARM** |

## E3 — cross-doc ($LIST_COUNT archives, page 1 each)

| Engine | Cold (ms) | Warm median (ms) |
|---|---|---|
| pdf-raster | $E3_COLD | **$E3_WARM** |

## E4 — random-access (1000 pages)

| Engine | Cold (ms) | Warm median (ms) |
|---|---|---|
| pdf-raster | $E4_COLD | **$E4_WARM** |

---

Raw data: \`bench/v11/results.csv\`.  Full run log: \`bench/v11/run.log\`.
EOF

echo
echo "==========================================================="
echo "Phase 11 contest run complete $(date -Iseconds)"
echo "Results: $MD"
echo "==========================================================="
