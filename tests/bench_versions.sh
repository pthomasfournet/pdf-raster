#!/bin/bash
# Benchmark one release binary per minor version against pdftoppm at 150 DPI.
#
# For each version tag (v0.1.0, v0.2.0, v0.3.0, v0.4.0, v0.5.1):
#   1. Creates a worktree at .worktrees/<tag> if absent
#   2. Builds a release binary in that worktree
#   3. Runs all 10 corpus PDFs with cold-cache eviction
#
# Output: markdown table with one column per version + pdftoppm reference column.
#
# Usage: tests/bench_versions.sh [--corpus-dir <path>]
#
# Requirements: cargo, pdftoppm, python3 (for cache eviction)

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
FIXTURES="$REPO/tests/fixtures"
WORKTREES="$REPO/.worktrees"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --corpus-dir) FIXTURES="$2"; shift 2 ;;
    *) echo "ERROR: unknown argument: $1" >&2; exit 1 ;;
  esac
done

# Versions: tag → has --backend flag (y/n)
declare -A TAGS
TAGS["v0.1.0"]="n"
TAGS["v0.2.0"]="n"
TAGS["v0.3.0"]="n"
TAGS["v0.4.0"]="y"
TAGS["v0.5.1"]="y"

VERSION_ORDER=(v0.1.0 v0.2.0 v0.3.0 v0.4.0 v0.5.1)

corpora=(
  "01-native-text-small"
  "02-native-vector-text"
  "03-native-text-dense"
  "04-ebook-mixed"
  "05-academic-book"
  "06-modern-layout-dct"
  "07-journal-dct-heavy"
  "08-scan-dct-1927"
  "09-scan-dct-1836"
  "10-scan-jbig2-jpx"
)

# ── Helpers ───────────────────────────────────────────────────────────────────

check_disk() {
  local dir="$1" min_kb=1048576
  local avail_kb
  avail_kb=$(df --output=avail -k "$dir" 2>/dev/null | tail -1)
  if [[ -z "$avail_kb" || "$avail_kb" -lt "$min_kb" ]]; then
    echo "ERROR: less than 1 GB free on $(df --output=target "$dir" | tail -1)" >&2
    exit 1
  fi
}

evict_file() {
  local path="$1"
  python3 -c "
import ctypes, os, sys
path = sys.argv[1]
fd = os.open(path, os.O_RDONLY)
size = os.fstat(fd).st_size
ctypes.CDLL(None).posix_fadvise(fd, 0, size, 4)
os.close(fd)
" "$path" 2>/dev/null || true
}

time_ms() {
  local t
  t=$( { time "$@" >/dev/null 2>&1; } 2>&1 | awk '/^real/{print $2}')
  echo "$t" | awk -F'[ms]' '{printf "%d", ($1*60+$2)*1000}'
}

# ── Step 1: prune dead worktrees ──────────────────────────────────────────────
echo "==> Pruning stale worktrees..." >&2
git -C "$REPO" worktree prune

# ── Step 2: create/verify worktrees and build binaries ───────────────────────
declare -A BINS

for tag in "${VERSION_ORDER[@]}"; do
  wt="$WORKTREES/$tag"
  bin="$wt/target/release/pdf-raster"

  if [[ ! -d "$wt" ]]; then
    echo "==> Creating worktree $tag at $wt..." >&2
    git -C "$REPO" worktree add --detach "$wt" "$tag"
  fi

  if [[ ! -x "$bin" ]]; then
    echo "==> Building $tag (release)..." >&2
    check_disk "$wt"
    RUSTFLAGS="-C target-cpu=native" cargo build --release \
      --manifest-path "$wt/Cargo.toml" -p pdf-raster 2>&1 | tail -3
    echo "    Built: $bin" >&2
  else
    echo "==> $tag binary already built, skipping." >&2
  fi

  BINS[$tag]="$bin"
done

# ── Step 3: collect pdftoppm timings (one cold-cache run per corpus) ──────────
echo "==> Timing pdftoppm reference..." >&2
declare -A PPTIMES

for name in "${corpora[@]}"; do
  pdf="$FIXTURES/corpus-${name}.pdf"
  if [[ ! -f "$pdf" ]]; then
    PPTIMES[$name]=-1
    continue
  fi
  evict_file "$pdf"
  check_disk /tmp
  TMPDIR_P=$(mktemp -d)
  PPTIMES[$name]=$(time_ms pdftoppm -r 150 "$pdf" "$TMPDIR_P/p")
  rm -rf "$TMPDIR_P"
  echo "    corpus-$name: ${PPTIMES[$name]}ms" >&2
done

# ── Step 4: collect per-version timings ──────────────────────────────────────
declare -A TIMES  # key: "tag:corpus"

for tag in "${VERSION_ORDER[@]}"; do
  bin="${BINS[$tag]}"
  has_backend="${TAGS[$tag]}"
  echo "==> Timing $tag..." >&2

  for name in "${corpora[@]}"; do
    pdf="$FIXTURES/corpus-${name}.pdf"
    if [[ ! -f "$pdf" ]]; then
      TIMES["$tag:$name"]=-1
      continue
    fi
    evict_file "$pdf"
    check_disk /tmp
    TMPDIR_R=$(mktemp -d)

    if [[ "$has_backend" == "y" ]]; then
      ms=$(time_ms "$bin" --backend cpu -r 150 "$pdf" "$TMPDIR_R/r")
    else
      ms=$(time_ms "$bin" -r 150 "$pdf" "$TMPDIR_R/r")
    fi

    rm -rf "$TMPDIR_R"
    TIMES["$tag:$name"]=$ms
    echo "    corpus-$name: ${ms}ms" >&2
  done
done

# ── Step 5: render markdown table ─────────────────────────────────────────────
echo ""
echo "## Version regression benchmark (CPU-only, 150 DPI, cold cache)"
echo ""

# Header
printf "| %-30s" "Corpus"
for tag in "${VERSION_ORDER[@]}"; do
  printf " | %10s" "$tag"
done
printf " | %10s" "pdftoppm"
printf " |\n"

# Separator
printf "| %s" "$(printf '%.0s-' {1..31})"
for tag in "${VERSION_ORDER[@]}"; do
  printf " | %s" "$(printf '%.0s-' {1..11})"
done
printf " | %s" "$(printf '%.0s-' {1..11})"
printf " |\n"

# Rows
for name in "${corpora[@]}"; do
  printf "| %-30s" "$name"
  for tag in "${VERSION_ORDER[@]}"; do
    ms="${TIMES[$tag:$name]}"
    if [[ "$ms" -lt 0 ]]; then
      printf " | %10s" "N/A"
    else
      printf " | %8dms" "$ms"
    fi
  done
  pp="${PPTIMES[$name]}"
  if [[ "$pp" -lt 0 ]]; then
    printf " | %10s" "N/A"
  else
    printf " | %8dms" "$pp"
  fi
  printf " |\n"
done

echo ""
echo "_Notes: v0.1.0–v0.3.0 use system allocator (glibc malloc); v0.4.0+ use mimalloc._"
echo "_Corpus 08/09 numbers for older versions will be inflated by allocator contention._"
