#!/usr/bin/env bash
# PGO + BOLT release build for pdf-raster.
#
# Usage: ./scripts/release_pgo_build.sh
#
# Prereqs:
#   - rustup component add llvm-tools-preview      (for `llvm-profdata`)
#   - llvm-bolt on PATH for the optional BOLT step
#       apt install bolt          # Ubuntu 24.04+
#       or download a build from  https://github.com/llvm/llvm-project/releases
#
# Output: target/release/pdf-raster (PGO-optimised; BOLT applied on top if available).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BINARY="./target/release/pdf-raster"

# --- Locate llvm-profdata. -----------------------------------------------------
# rustup ships it under the toolchain sysroot when `llvm-tools-preview` is added.
SYSROOT="$(rustc --print sysroot)"
HOST_TRIPLE="$(rustc -vV | sed -n 's|host: ||p')"
PROFDATA="$SYSROOT/lib/rustlib/$HOST_TRIPLE/bin/llvm-profdata"
if [[ ! -x "$PROFDATA" ]]; then
    if command -v llvm-profdata >/dev/null 2>&1; then
        PROFDATA="$(command -v llvm-profdata)"
    else
        echo "error: llvm-profdata not found." >&2
        echo "       Install with: rustup component add llvm-tools-preview" >&2
        exit 1
    fi
fi

# --- Verify training fixture is present. --------------------------------------
TRAINING_PDF="tests/fixtures/corpus-04-ebook-mixed.pdf"
if [[ ! -f "$TRAINING_PDF" ]]; then
    echo "error: training fixture missing: $TRAINING_PDF" >&2
    exit 1
fi

# --- Scratch dirs cleaned on exit. --------------------------------------------
PROFDIR="$(mktemp -d -t pdf_raster_pgo.XXXXXX)"
TRAIN_OUTDIR="$(mktemp -d -t pdf_raster_train.XXXXXX)"
TRAIN_OUT_PREFIX="$TRAIN_OUTDIR/page"
cleanup() {
    rm -rf "$PROFDIR" "$TRAIN_OUTDIR"
}
trap cleanup EXIT

echo "==> [1/4] Instrumented build (profile-generate)"
echo "         PROFDIR=$PROFDIR"
RUSTFLAGS="-Cprofile-generate=$PROFDIR" cargo build --release -p pdf-raster

echo "==> [2/4] Profile training: rendering 10 pages of $TRAINING_PDF"
"$BINARY" "$TRAINING_PDF" "$TRAIN_OUT_PREFIX" -f 1 -l 10 >/dev/null
ls -1 "$TRAIN_OUTDIR" | head -3
RAW_PROFILES="$(find "$PROFDIR" -name '*.profraw' | wc -l)"
if [[ "$RAW_PROFILES" -eq 0 ]]; then
    echo "error: no .profraw files produced — training run did not write profile data." >&2
    exit 1
fi
echo "         collected $RAW_PROFILES .profraw file(s)"

echo "==> [3/4] Merging PGO profile data ($PROFDIR -> merged.profdata)"
"$PROFDATA" merge -o "$PROFDIR/merged.profdata" "$PROFDIR"

echo "==> [4/4] PGO-guided rebuild (profile-use)"
RUSTFLAGS="-Cprofile-use=$PROFDIR/merged.profdata -Cllvm-args=-pgo-warn-missing-function" \
    cargo build --release -p pdf-raster

if command -v llvm-bolt >/dev/null 2>&1; then
    echo "==> [bonus] BOLT optimisation"
    BOLT_BACKUP="$(mktemp -t pdf-raster.pre-bolt.XXXXXX)"
    cp "$BINARY" "$BOLT_BACKUP"
    if llvm-bolt "$BOLT_BACKUP" \
            -o "$BINARY" \
            -reorder-blocks=ext-tsp \
            -reorder-functions=hfsort \
            -split-functions \
            -split-all-cold \
            -dyno-stats; then
        echo "    BOLT applied."
    else
        echo "    BOLT failed; restoring PGO-only binary." >&2
        cp "$BOLT_BACKUP" "$BINARY"
    fi
    rm -f "$BOLT_BACKUP"
else
    echo "==> [bonus] BOLT not on PATH — skipping."
    echo "         Install: apt install bolt   (Ubuntu 24.04+)"
fi

echo
echo "==> Done. Binary at $BINARY"
ls -la "$BINARY"
