#!/bin/bash
# Benchmark all corpus PDFs against pdftoppm at 150 DPI.
#
# Usage:
#   bench_corpus.sh [--backend cpu|vaapi|cuda] [--vaapi-device /dev/dri/renderD129]
#                   [--runs N] [--warmup N] [--corpus-dir <path>]
#
# For each corpus PDF, runs pdf-raster and pdftoppm under hyperfine (statistical
# multi-run timing), captures CPU utilisation (mpstat) and disk throughput
# (iostat) during each run, and reports:
#
#   mean ± stddev  cpu_avg%  cpu_peak%  disk_read MB/s  speedup
#
# This catches contaminated runs: a high stddev or low cpu_avg% on a corpus
# that should saturate all threads means background activity is interfering.
#
# Requirements: hyperfine, mpstat (sysstat), iostat (sysstat), python3
# Recommended:  kernel.perf_event_paranoid=1 (set via /etc/sysctl.d/60-perf-profiling.conf)

set -euo pipefail

BIN="${BIN:-$(dirname "$0")/../target/release/pdf-raster}"
FIXTURES="$(dirname "$0")/fixtures"
BACKEND="cpu"
VAAPI_DEVICE="/dev/dri/renderD129"
RUNS=5
WARMUP=1

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend)      BACKEND="$2";      shift 2 ;;
    --vaapi-device) VAAPI_DEVICE="$2"; shift 2 ;;
    --corpus-dir)   FIXTURES="$2";     shift 2 ;;
    --runs)         RUNS="$2";         shift 2 ;;
    --warmup)       WARMUP="$2";       shift 2 ;;
    *) echo "ERROR: unknown argument: $1" >&2; exit 1 ;;
  esac
done

# ── Dependency checks ─────────────────────────────────────────────────────────
[[ -x "$BIN" ]] || { echo "ERROR: binary not found or not executable: $BIN" >&2; exit 1; }
command -v hyperfine >/dev/null || { echo "ERROR: hyperfine not found — install with: cargo install hyperfine" >&2; exit 1; }
command -v mpstat    >/dev/null || { echo "ERROR: mpstat not found — install with: sudo apt install sysstat" >&2; exit 1; }
command -v iostat    >/dev/null || { echo "ERROR: iostat not found — install with: sudo apt install sysstat" >&2; exit 1; }
if [[ "$BACKEND" != "vaapi" ]]; then
  command -v pdftoppm >/dev/null || { echo "ERROR: pdftoppm not found in PATH" >&2; exit 1; }
fi

# ── Pre-flight: disk space ────────────────────────────────────────────────────
check_disk() {
  local dir="$1" min_kb=1048576
  local avail_kb
  avail_kb=$(df --output=avail -k "$dir" 2>/dev/null | tail -1)
  if [[ -z "$avail_kb" || "$avail_kb" -lt "$min_kb" ]]; then
    echo "ERROR: less than 1 GB free on $(df --output=target "$dir" | tail -1) — aborting" >&2
    exit 1
  fi
}

# ── Pre-flight: warn if perf_event_paranoid blocks profiling ─────────────────
PARANOID=$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo "unknown")
if [[ "$PARANOID" != "1" && "$PARANOID" != "0" && "$PARANOID" != "-1" ]]; then
  echo "WARN: kernel.perf_event_paranoid=$PARANOID — perf/flamegraph unavailable." >&2
  echo "      Fix permanently: echo 'kernel.perf_event_paranoid = 1' | sudo tee /etc/sysctl.d/60-perf-profiling.conf && sudo sysctl -p /etc/sysctl.d/60-perf-profiling.conf" >&2
fi

# ── Pre-flight: system load gate ─────────────────────────────────────────────
# Refuse to run if 1-min load average exceeds 2× the number of physical cores.
# A contaminated system produces meaningless numbers and wastes 30+ minutes.
NCORES=$(nproc)
LOAD1=$(cut -d' ' -f1 /proc/loadavg)
LOAD_INT=$(echo "$LOAD1" | awk '{print int($1)}')
if [[ "$LOAD_INT" -gt $((NCORES * 2)) ]]; then
  echo "ERROR: system load ($LOAD1) is too high for reliable benchmarking (threshold: $((NCORES*2)))." >&2
  echo "       Wait for background activity to settle before running benchmarks." >&2
  exit 1
fi

# ── Cache eviction ────────────────────────────────────────────────────────────
# posix_fadvise(FADV_DONTNEED) evicts the file from the OS page cache without root.
evict_file() {
  python3 -c "
import ctypes, os, sys
fd = os.open(sys.argv[1], os.O_RDONLY)
ctypes.CDLL(None).posix_fadvise(fd, 0, os.fstat(fd).st_size, 4)
os.close(fd)
" "$1" 2>/dev/null || echo "WARN: cache eviction failed for $1 (non-fatal — run may be warm-cache)" >&2
}

# ── Monitored single run ──────────────────────────────────────────────────────
# Runs a command once while sampling mpstat + iostat in the background.
# Prints: wall_ms cpu_avg cpu_peak iowait_avg disk_read_mb
monitored_run() {
  local cmd=("$@")
  local stats_f iostat_f
  stats_f=$(mktemp)
  iostat_f=$(mktemp)

  # Detect the storage device backing /tmp (where output goes)
  local dev
  dev=$(df --output=source /tmp | tail -1 | sed 's|/dev/||')

  mpstat -P ALL 1 9999 > "$stats_f"  2>/dev/null &
  local mpstat_pid=$!
  iostat -d -m "$dev" 1 9999 > "$iostat_f" 2>/dev/null &
  local iostat_pid=$!

  sleep 0.3  # let samplers emit a baseline sample

  local t0 t1
  t0=$(date +%s%3N)
  "${cmd[@]}"
  t1=$(date +%s%3N)

  sleep 0.3  # capture trailing samples
  kill "$mpstat_pid" "$iostat_pid" 2>/dev/null
  wait "$mpstat_pid" "$iostat_pid" 2>/dev/null || true

  local wall=$(( t1 - t0 ))

  # mpstat: parse all-CPU rows (field 2 == "all"), compute util = 100 - %idle (last field)
  local cpu_avg cpu_peak
  read -r cpu_avg cpu_peak < <(awk '
    /^[0-9]/ && $2=="all" {
      util = 100 - $NF
      if (util > peak) peak = util
      sum += util; n++
    }
    END { if (n>0) printf "%.0f %.0f", sum/n, peak; else print "0 0" }
  ' "$stats_f")

  # iostat: average MB/s read on the detected device
  local disk_read
  disk_read=$(awk -v d="$dev" '$1==d {sum+=$3; n++} END {if(n>0) printf "%.0f", sum/n; else print "0"}' "$iostat_f")

  rm -f "$stats_f" "$iostat_f"
  echo "$wall $cpu_avg $cpu_peak $disk_read"
}

# ── Hyperfine timing (mean ± stddev) ─────────────────────────────────────────
# Returns "mean_ms stddev_ms" parsed from hyperfine JSON output.
hyperfine_ms() {
  local json
  json=$(hyperfine --runs "$RUNS" --warmup "$WARMUP" \
    --export-json /dev/stdout --output null \
    -- "$@" 2>/dev/null)
  echo "$json" | python3 -c "
import json, sys
r = json.load(sys.stdin)['results'][0]
mean_ms   = r['mean']   * 1000
stddev_ms = r['stddev'] * 1000
print(f'{mean_ms:.0f} {stddev_ms:.0f}')
"
}

# ── Build argument lists ──────────────────────────────────────────────────────
PDF_RASTER_ARGS=(--backend "$BACKEND" -r 150)
[[ "$BACKEND" == "vaapi" ]] && PDF_RASTER_ARGS+=(--vaapi-device "$VAAPI_DEVICE")

# ── Corpus list ───────────────────────────────────────────────────────────────
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

# ── Header ────────────────────────────────────────────────────────────────────
REF_LABEL="pdftoppm"
[[ "$BACKEND" == "vaapi" ]] && REF_LABEL="cpu-only"

printf "\nSystem: load=%-5s cores=%d  perf_paranoid=%s\n" "$LOAD1" "$NCORES" "$PARANOID"
printf "Binary: %s\n" "$BIN"
printf "Runs:   %d (warmup %d) via hyperfine + mpstat/iostat monitoring\n\n" "$RUNS" "$WARMUP"

printf "%-32s  %18s  %8s  %8s  %8s  %10s  %10s\n" \
  "PDF" "pdf-raster mean±σ" "cpu avg" "cpu peak" "disk MB/s" "$REF_LABEL" "speedup"
printf "%-32s  %18s  %8s  %8s  %8s  %10s  %10s\n" \
  "---" "-----------------" "-------" "--------" "---------" "----------" "-------"

# ── Per-corpus run ────────────────────────────────────────────────────────────
for name in "${corpora[@]}"; do
  pdf="$FIXTURES/corpus-${name}.pdf"
  if [[ ! -f "$pdf" ]]; then
    echo "SKIP: corpus-${name}.pdf not found" >&2
    continue
  fi

  check_disk /tmp

  # ── pdf-raster: cold-cache monitored run for CPU/disk metrics ──
  evict_file "$pdf"
  OUTDIR=$(mktemp -d -p /tmp)
  read -r mon_wall mon_cpu_avg mon_cpu_peak mon_disk < <(
    monitored_run "$BIN" "${PDF_RASTER_ARGS[@]}" "$pdf" "$OUTDIR/" > /dev/null 2>&1
  ) || { rm -rf "$OUTDIR"; echo "ERROR: monitored run failed for $name" >&2; exit 1; }
  rm -rf "$OUTDIR"

  # ── pdf-raster: hyperfine for stable mean ± stddev ──
  # Warmup runs serve as additional cache-eviction; we rely on hyperfine's
  # --warmup to prime the binary's startup path, not the file cache.
  evict_file "$pdf"
  HYPERFINE_PREPARE="$(which python3) -c \"
import ctypes,os,sys; fd=os.open('$pdf',os.O_RDONLY); ctypes.CDLL(None).posix_fadvise(fd,0,os.fstat(fd).st_size,4); os.close(fd)
\""
  OUTDIR=$(mktemp -d -p /tmp)
  read -r mean1 stddev1 < <(
    hyperfine --runs "$RUNS" --warmup "$WARMUP" \
      --prepare "$HYPERFINE_PREPARE" \
      --export-json /dev/stdout --output null \
      -- "$BIN ${PDF_RASTER_ARGS[*]} '$pdf' '$OUTDIR/r'" 2>/dev/null \
    | python3 -c "
import json,sys
r=json.load(sys.stdin)['results'][0]
print(f\"{r['mean']*1000:.0f} {r['stddev']*1000:.0f}\")
"
  ) || { rm -rf "$OUTDIR"; echo "ERROR: hyperfine failed for $name" >&2; exit 1; }
  rm -rf "$OUTDIR"

  # ── reference (pdftoppm or cpu-only baseline) ──
  evict_file "$pdf"
  OUTDIR=$(mktemp -d -p /tmp)
  if [[ "$BACKEND" == "vaapi" ]]; then
    read -r mean2 _ < <(
      hyperfine --runs "$RUNS" --warmup "$WARMUP" \
        --prepare "$HYPERFINE_PREPARE" \
        --export-json /dev/stdout --output null \
        -- "$BIN --backend cpu -r 150 '$pdf' '$OUTDIR/c'" 2>/dev/null \
      | python3 -c "
import json,sys
r=json.load(sys.stdin)['results'][0]
print(f\"{r['mean']*1000:.0f} 0\")
"
    ) || { rm -rf "$OUTDIR"; echo "ERROR: hyperfine ref failed for $name" >&2; exit 1; }
  else
    read -r mean2 _ < <(
      hyperfine --runs "$RUNS" --warmup "$WARMUP" \
        --prepare "$HYPERFINE_PREPARE" \
        --export-json /dev/stdout --output null \
        -- "pdftoppm -r 150 '$pdf' '$OUTDIR/p'" 2>/dev/null \
      | python3 -c "
import json,sys
r=json.load(sys.stdin)['results'][0]
print(f\"{r['mean']*1000:.0f} 0\")
"
    ) || { rm -rf "$OUTDIR"; echo "ERROR: hyperfine ref failed for $name" >&2; exit 1; }
  fi
  rm -rf "$OUTDIR"

  [[ "$mean1" -le 0 ]] && { echo "ERROR: zero mean for $name" >&2; exit 1; }

  speedup=$(awk "BEGIN{printf \"%.2fx\", $mean2/$mean1}")

  # Flag suspect runs: stddev > 15% of mean, or cpu_avg < 30% on a multi-page corpus
  flag=""
  stddev_pct=$(awk "BEGIN{printf \"%.0f\", 100*$stddev1/$mean1}")
  [[ "$stddev_pct" -gt 15 ]] && flag+="[high-σ] "
  # Short corpora (01,02) legitimately have low CPU% due to startup dominance
  pages=$(echo "$name" | cut -d'-' -f1)
  [[ "$mon_cpu_avg" -lt 30 && "$pages" -gt 2 ]] && flag+="[low-cpu] "

  # mon_wall is the single cold-cache wall time used to validate hyperfine mean
  # (if mon_wall >> mean1 the monitored run was contaminated by a warm-cache effect
  # on the hyperfine runs or vice versa; flag it so the reader can investigate)
  if [[ "$mon_wall" -gt $((mean1 * 3)) ]]; then
    flag+="[mon-wall=${mon_wall}ms>>mean] "
  fi

  printf "%-32s  %9dms ±%4dms  %7s%%  %7s%%  %7s MB/s  %9dms  %9s  %s\n" \
    "$name" "$mean1" "$stddev1" "$mon_cpu_avg" "$mon_cpu_peak" "$mon_disk" \
    "$mean2" "$speedup" "$flag"
done

echo ""
echo "Flags: [high-σ] = stddev > 15% of mean (background interference likely)"
echo "       [low-cpu] = avg CPU < 30% on multi-page corpus (allocator/IO contention?)"
