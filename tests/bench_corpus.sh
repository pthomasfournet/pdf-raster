#!/bin/bash
# Benchmark pdf-raster across all corpus PDFs.
#
# Usage:
#   bench_corpus.sh [--backend cpu|vaapi|cuda] [--vaapi-device /dev/dri/renderD129]
#                   [--runs N] [--warmup N] [--corpus-dir <path>]
#
# Reports per-corpus: hyperfine mean ± stddev, avg CPU%, peak CPU%, disk MB/s.
# Flags contaminated runs automatically (high stddev, low CPU utilisation).
#
# For comparison against pdftoppm, use tests/bench_compare.sh.
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
command -v hyperfine >/dev/null || { echo "ERROR: hyperfine not found — install: cargo install hyperfine" >&2; exit 1; }
command -v mpstat    >/dev/null || { echo "ERROR: mpstat not found — install: sudo apt install sysstat" >&2; exit 1; }
command -v iostat    >/dev/null || { echo "ERROR: iostat not found — install: sudo apt install sysstat" >&2; exit 1; }

# ── Pre-flight: perf_event_paranoid ──────────────────────────────────────────
PARANOID=$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo "unknown")
if [[ "$PARANOID" != "1" && "$PARANOID" != "0" && "$PARANOID" != "-1" ]]; then
  echo "WARN: kernel.perf_event_paranoid=$PARANOID — perf/flamegraph blocked." >&2
  echo "      Fix: echo 'kernel.perf_event_paranoid = 1' | sudo tee /etc/sysctl.d/60-perf-profiling.conf" >&2
fi

# ── Pre-flight: system load gate ─────────────────────────────────────────────
# Refuse to run if 1-min load > 2× core count — contaminated system wastes 30+ min.
NCORES=$(nproc)
LOAD1=$(cut -d' ' -f1 /proc/loadavg)
LOAD_INT=$(echo "$LOAD1" | awk '{print int($1)}')
if [[ "$LOAD_INT" -gt $((NCORES * 2)) ]]; then
  echo "ERROR: load ($LOAD1) too high — threshold $((NCORES*2)). Wait for system to settle." >&2
  exit 1
fi

# ── Pre-flight: disk space ────────────────────────────────────────────────────
check_disk() {
  local avail_kb
  avail_kb=$(df --output=avail -k /tmp 2>/dev/null | tail -1)
  if [[ -z "$avail_kb" || "$avail_kb" -lt 1048576 ]]; then
    echo "ERROR: less than 1 GB free on /tmp — aborting" >&2; exit 1
  fi
}

# ── Cache eviction ────────────────────────────────────────────────────────────
# posix_fadvise(FADV_DONTNEED=4) evicts without root.
evict_file() {
  python3 -c "
import ctypes, os, sys
fd = os.open(sys.argv[1], os.O_RDONLY)
ctypes.CDLL(None).posix_fadvise(fd, 0, os.fstat(fd).st_size, 4)
os.close(fd)
" "$1" 2>/dev/null || echo "WARN: cache eviction failed for $1 (run may be warm)" >&2
}

# ── Monitored single run ──────────────────────────────────────────────────────
# Runs the command once with mpstat + iostat sampling in the background.
# Stdout: wall_ms cpu_avg cpu_peak disk_read_mb
monitored_run() {
  local stats_f iostat_f dev
  stats_f=$(mktemp)
  iostat_f=$(mktemp)
  dev=$(df --output=source /tmp | tail -1 | sed 's|.*/||')

  mpstat -P ALL 1 9999 > "$stats_f" 2>/dev/null &
  local mpstat_pid=$!
  iostat -d -m "$dev" 1 9999 > "$iostat_f" 2>/dev/null &
  local iostat_pid=$!

  sleep 0.3  # let samplers emit one baseline interval

  local t0 t1
  t0=$(date +%s%3N)
  "$@" > /dev/null 2>&1
  t1=$(date +%s%3N)
  local wall=$(( t1 - t0 ))

  sleep 0.3  # capture trailing samples
  kill "$mpstat_pid" "$iostat_pid" 2>/dev/null
  wait "$mpstat_pid" "$iostat_pid" 2>/dev/null || true

  # mpstat output: "HH:MM:SS AM/PM  all  %usr %nice %sys %iowait ... %idle"
  # Match lines where the CPU column (field after timestamp) is "all".
  # Timestamp format varies (12h with AM/PM = 3 fields; 24h = 2 fields) so
  # match on the "all" token appearing anywhere before the numeric columns.
  local cpu_avg cpu_peak
  read -r cpu_avg cpu_peak < <(awk '
    /[[:space:]]all[[:space:]]/ {
      idle = $NF + 0
      if (idle > 0 && idle <= 100) {
        util = 100 - idle
        if (util > peak) peak = util
        sum += util; n++
      }
    }
    END { if (n > 0) printf "%.0f %.0f", sum/n, peak; else print "0 0" }
  ' "$stats_f")

  local disk_read
  disk_read=$(awk -v d="$dev" '$1==d {sum+=$3; n++} END {if(n>0) printf "%.0f", sum/n; else print "0"}' "$iostat_f")

  rm -f "$stats_f" "$iostat_f"
  echo "$wall $cpu_avg $cpu_peak $disk_read"
}

# ── Hyperfine mean ± stddev ───────────────────────────────────────────────────
# hyperfine_ms PREPARE_CMD BENCH_CMD  →  prints "mean_ms stddev_ms"
# Uses a temp JSON file; /dev/stdout is unreliable (hyperfine also writes there).
hyperfine_ms() {
  local prepare_cmd="$1" bench_cmd="$2" json_f
  json_f=$(mktemp --suffix=.json)
  hyperfine --runs "$RUNS" --warmup "$WARMUP" \
    --prepare "$prepare_cmd" \
    --export-json "$json_f" \
    -- "$bench_cmd" > /dev/null 2>&1
  python3 - "$json_f" <<'EOF'
import json, sys
with open(sys.argv[1]) as f:
    r = json.load(f)['results'][0]
print(f"{r['mean']*1000:.0f} {r['stddev']*1000:.0f}")
EOF
  rm -f "$json_f"
}

# ── Argument list for pdf-raster ─────────────────────────────────────────────
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
printf "\nSystem: load=%-5s  cores=%d  perf_paranoid=%s\n" "$LOAD1" "$NCORES" "$PARANOID"
printf "Binary: %s  backend=%s\n" "$BIN" "$BACKEND"
printf "Runs:   %d (warmup %d) — hyperfine mean±σ + mpstat/iostat\n\n" "$RUNS" "$WARMUP"

printf "%-32s  %18s  %8s  %8s  %9s  %s\n" \
  "corpus" "mean ± σ" "cpu avg" "cpu peak" "disk MB/s" "flags"
printf "%-32s  %18s  %8s  %8s  %9s\n" \
  "------" "--------" "-------" "--------" "---------"

# ── Per-corpus ────────────────────────────────────────────────────────────────
for name in "${corpora[@]}"; do
  pdf="$FIXTURES/corpus-${name}.pdf"
  if [[ ! -f "$pdf" ]]; then
    printf "SKIP  %-32s  (file not found)\n" "$name" >&2
    continue
  fi

  check_disk

  # Cold-cache monitored run — captures CPU% and disk MB/s for this corpus.
  evict_file "$pdf"
  mon_outdir=$(mktemp -d -p /tmp)
  read -r mon_wall mon_cpu_avg mon_cpu_peak mon_disk < <(
    monitored_run "$BIN" "${PDF_RASTER_ARGS[@]}" "$pdf" "$mon_outdir/"
  ) || { rm -rf "$mon_outdir"; echo "ERROR: monitored run failed for $name" >&2; exit 1; }
  rm -rf "$mon_outdir"

  # Hyperfine: cold-cache eviction before each run via --prepare.
  PREPARE="$(which python3) -c \"import ctypes,os,sys; fd=os.open('$pdf',os.O_RDONLY); ctypes.CDLL(None).posix_fadvise(fd,0,os.fstat(fd).st_size,4); os.close(fd)\""
  hf_outdir=$(mktemp -d -p /tmp)
  read -r mean_ms stddev_ms < <(
    hyperfine_ms "$PREPARE" "$BIN ${PDF_RASTER_ARGS[*]} '$pdf' '$hf_outdir/r'"
  ) || { rm -rf "$hf_outdir"; echo "ERROR: hyperfine failed for $name" >&2; exit 1; }
  rm -rf "$hf_outdir"

  # ── Quality flags ─────────────────────────────────────────────────────────
  flags=""
  stddev_pct=$(awk "BEGIN{printf \"%.0f\", ($mean_ms>0) ? 100*$stddev_ms/$mean_ms : 0}")
  [[ "$stddev_pct" -gt 15 ]] && flags+="[high-σ ${stddev_pct}%] "

  # Corpora 01-02 are short enough that startup overhead dominates CPU%;
  # only flag low utilisation on corpora 03+.
  corpus_num="${name:0:2}"
  if [[ "$mon_cpu_avg" -lt 30 && "$corpus_num" -gt "02" ]]; then
    flags+="[low-cpu ${mon_cpu_avg}%] "
  fi

  # Flag if the cold single-run is >3× slower than the hyperfine mean —
  # suggests the monitored run hit unusual I/O or scheduling pressure.
  if [[ "$mon_wall" -gt $((mean_ms * 3)) ]]; then
    flags+="[cold=${mon_wall}ms>>mean] "
  fi

  printf "%-32s  %9dms ±%5dms  %7s%%  %7s%%  %8s MB/s  %s\n" \
    "$name" "$mean_ms" "$stddev_ms" "$mon_cpu_avg" "$mon_cpu_peak" "$mon_disk" "$flags"
done

printf "\nFlags:\n"
printf "  [high-σ N%%]    stddev > 15%% of mean — background interference likely\n"
printf "  [low-cpu N%%]   avg CPU < 30%% on multi-page corpus — contention or I/O bound\n"
printf "  [cold=Nms>>mean] cold-cache single run >> hyperfine mean — scheduling outlier\n"
