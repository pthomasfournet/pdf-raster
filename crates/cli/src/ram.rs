//! `--ram` mode: redirect render output to a fresh tmpfs directory, with a
//! dynamic spill-to-disk fallback when free memory drops below a safety margin.
//!
//! When `--ram` is set, this module:
//! 1. Creates `/dev/shm/pdf-raster-<pid>-<nanos>/` (or honours `--ram-path`)
//!    and rewrites `args.output_prefix` to live inside it. The basename of
//!    the user's original prefix is preserved as the file stem so downstream
//!    tooling that consumes `out-NNN.ppm` keeps working unchanged.
//! 2. Returns a [`RamDirGuard`] that removes the directory on drop.
//! 3. Returns a [`SpillPolicy`] that the per-page writer queries to decide
//!    where each page goes — RAM by default, disk when memory tightens.
//!
//! The chosen path is printed:
//! - on stderr at startup (so a `tee` or human watcher can see it)
//! - on stdout at the end (so a follow-on tool can capture it via shell
//!   substitution: `out=$(pdf-raster --ram doc.pdf p)`)
//!
//! # Why dynamic instead of pre-flight?
//!
//! A static "this PDF will produce N MB of output, do you have room?" check
//! relies on (a) reading every page's `MediaBox` at startup (slow) and (b)
//! guessing PNG/JPEG compression ratios (impossible). The dynamic approach
//! lets the renderer start producing pages immediately and only switches to
//! disk if the kernel reports tight memory — at which point we've already
//! benefited from RAM-speed writes for the early pages.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::args::Args;

/// Minimum free RAM (bytes) we insist remains after the next write. Below
/// this, the writer spills to disk. 1 GiB is enough headroom for one more
/// in-flight page bitmap (worst-case ~50 MB at 600 DPI A3) plus normal
/// per-thread scratch and OS workings.
const RAM_SAFETY_MARGIN: u64 = 1024 * 1024 * 1024;

/// How long a memory-availability reading is cached before being re-probed.
/// Re-reading `/proc/meminfo` is ~10µs but happens N×M times per render
/// (N pages × M threads); 100ms is fast enough to react to a sudden memory
/// crunch and slow enough that the syscall cost is negligible.
const PROBE_TTL: Duration = Duration::from_millis(100);

/// RAII handle to a tmpfs directory created for `--ram` mode.
///
/// `Drop` removes the directory and everything inside it. If removal fails
/// (e.g. another process moved the directory) the error is logged and
/// swallowed — there is no actionable response at process exit.
pub struct RamDirGuard {
    /// `Some(path)` when this guard owns a directory; `None` when `--ram`
    /// was off and the guard is a no-op.
    dir: Option<PathBuf>,
}

impl Drop for RamDirGuard {
    fn drop(&mut self) {
        let Some(dir) = self.dir.take() else { return };
        if let Err(e) = fs::remove_dir_all(&dir) {
            // Stderr only; the user's stdout may already be committed to a
            // pipeline and a stray log line would corrupt downstream parsing.
            eprintln!(
                "pdf-raster: warning: could not remove ram dir {}: {e}",
                dir.display()
            );
        }
    }
}

/// Per-page write-target policy: RAM (the tmpfs dir) or disk (the user's
/// original `OUTPUT_PREFIX` location). When `--ram` is off, this still wraps
/// the user's prefix and is a no-op — `ram_prefix == disk_prefix` so every
/// page lands on disk regardless of memory conditions.
///
/// One instance is shared across all worker threads via a borrowed reference;
/// the inner state (atomics + mutex) is cheap to read concurrently.
pub struct SpillPolicy {
    /// `(ram_prefix, disk_prefix)`. When `--ram` is off both slots hold the
    /// user's original prefix and `next_prefix` short-circuits.
    targets: (String, String),
    /// Cached `MemAvailable` reading + when it was taken. Refreshed on TTL
    /// expiry; the lock is contended only when the cached value is older
    /// than [`PROBE_TTL`].
    probe: Mutex<MemoryProbe>,
    /// Latched once we spill the first page so the warning prints exactly
    /// once across all worker threads.
    spill_announced: AtomicBool,
}

impl SpillPolicy {
    /// Build a passthrough policy that always returns `disk_prefix` (used
    /// when `--ram` is off — keeps the renderer-side API uniform).
    fn passthrough(disk_prefix: String) -> Self {
        // Both slots hold the user's prefix so next_prefix() short-circuits.
        Self {
            targets: (disk_prefix.clone(), disk_prefix),
            probe: Mutex::new(MemoryProbe::new()),
            spill_announced: AtomicBool::new(false),
        }
    }

    /// Returns the prefix that should be used for the next page write.
    /// `--ram` off: always the original prefix. `--ram` on: `ram_prefix`
    /// while free RAM is comfortable, `disk_prefix` once it tightens.
    #[must_use]
    pub fn next_prefix(&self) -> &str {
        let (ram_prefix, disk_prefix) = &self.targets;

        // Passthrough fast path: --ram off means both prefixes are equal,
        // so skip the memory probe entirely.
        if ram_prefix == disk_prefix {
            return ram_prefix;
        }

        if self.ram_has_room() {
            ram_prefix
        } else {
            if !self.spill_announced.swap(true, Ordering::Relaxed) {
                eprintln!(
                    "pdf-raster: --ram: free memory dropped below safety margin; \
                     spilling subsequent pages to disk at {disk_prefix}-NNN.*"
                );
            }
            disk_prefix
        }
    }

    fn ram_has_room(&self) -> bool {
        let mut probe = self
            .probe
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        probe.refresh_if_stale();
        probe.last_available_bytes >= RAM_SAFETY_MARGIN
    }
}

/// Cached `/proc/meminfo` reading. Refreshed on TTL.
struct MemoryProbe {
    last_read: Instant,
    last_available_bytes: u64,
}

impl MemoryProbe {
    fn new() -> Self {
        // Force initial refresh: an Instant from before the TTL window means
        // the first call to refresh_if_stale always reads. checked_sub guards
        // the unlikely case where `Instant::now()` is too close to the
        // platform's monotonic-clock origin to subtract from (boot < 1s ago).
        let stale = Instant::now()
            .checked_sub(PROBE_TTL + Duration::from_secs(1))
            .unwrap_or_else(Instant::now);
        Self {
            last_read: stale,
            last_available_bytes: 0,
        }
    }

    fn refresh_if_stale(&mut self) {
        if self.last_read.elapsed() < PROBE_TTL {
            return;
        }
        // Avoid two threads both refreshing at the same instant: each just
        // does its own read; the cost is bounded (~10µs) and we don't want a
        // global condvar on the hot path.
        self.last_available_bytes = read_mem_available().unwrap_or(0);
        self.last_read = Instant::now();
    }
}

/// Read `MemAvailable` from `/proc/meminfo` (in bytes). Returns `None` if
/// the file is missing or malformed (non-Linux, container without /proc, …).
///
/// `MemAvailable` is the kernel's own estimate of how much memory can be
/// allocated by a new workload without swapping — strictly better than
/// `MemFree` for this kind of admission control.
fn read_mem_available() -> Option<u64> {
    let text = fs::read_to_string("/proc/meminfo").ok()?;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("MemAvailable:") {
            // Format: "MemAvailable:   12345678 kB"
            let kb = rest.split_whitespace().next()?.parse::<u64>().ok()?;
            return Some(kb.saturating_mul(1024));
        }
    }
    None
}

/// Inspect `args.ram` and, if set, create a tmpfs directory and rewrite
/// `args.output_prefix` to live inside it. Always returns a [`SpillPolicy`]
/// even when `--ram` is off (in which case the policy is a no-op).
///
/// # Errors
/// - directory creation failed (permissions, ENOSPC on /dev/shm at startup).
pub fn redirect_to_ram(args: &mut Args) -> std::io::Result<(RamDirGuard, SpillPolicy)> {
    let original_prefix = args.output_prefix.clone();

    if !args.ram {
        if args.ram_path.is_some() {
            eprintln!("pdf-raster: warning: --ram-path has no effect without --ram");
        }
        return Ok((
            RamDirGuard { dir: None },
            SpillPolicy::passthrough(original_prefix),
        ));
    }

    // Auto-generated paths use create_dir_all (parent dirs may need creation
    // on exotic /dev/shm setups). User-supplied --ram-path uses create_dir so
    // pre-existing dirs surface as a warning rather than silently mixing the
    // user's pages with leftover files from a previous crashed run.
    let dir = if let Some(p) = &args.ram_path {
        let dir = PathBuf::from(p);
        if let Err(e) = fs::create_dir(&dir) {
            if e.kind() == std::io::ErrorKind::AlreadyExists {
                eprintln!(
                    "pdf-raster: warning: --ram-path {} already exists; \
                     leftover files will be removed alongside this run's output",
                    dir.display()
                );
            } else {
                return Err(e);
            }
        }
        dir
    } else {
        let dir = default_ram_dir();
        fs::create_dir_all(&dir)?;
        dir
    };

    // Preserve the basename of the user's prefix as the file stem inside the
    // tmpfs dir so naming stays predictable. Falls back to "out" if the user
    // gave something pathological like "/" or empty.
    let stem = Path::new(&original_prefix)
        .file_name()
        .and_then(|s| s.to_str())
        .filter(|s| !s.is_empty())
        .unwrap_or("out");

    let ram_prefix_pb = dir.join(stem);
    let ram_prefix = ram_prefix_pb.to_string_lossy().into_owned();

    eprintln!(
        "pdf-raster: --ram → writing to {} (auto-removed on exit)",
        dir.display()
    );
    println!("{}", dir.display());

    // Renderer sees the RAM prefix as the "default" path. The SpillPolicy
    // overrides per-page when memory tightens.
    args.output_prefix.clone_from(&ram_prefix);

    Ok((
        RamDirGuard { dir: Some(dir) },
        SpillPolicy {
            targets: (ram_prefix, original_prefix),
            probe: Mutex::new(MemoryProbe::new()),
            spill_announced: AtomicBool::new(false),
        },
    ))
}

/// Build a default `/dev/shm/pdf-raster-<pid>-<nanos>` path.
///
/// `pid + nanos since epoch` is enough entropy for parallel CLI invocations
/// to never collide in practice — the only collision risk is two invocations
/// in the same nanosecond from the same pid, which is impossible.
fn default_ram_dir() -> PathBuf {
    let pid = std::process::id();
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    PathBuf::from(format!("/dev/shm/pdf-raster-{pid}-{nanos}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_dir_is_under_dev_shm() {
        let p = default_ram_dir();
        assert!(p.starts_with("/dev/shm"), "got {}", p.display());
    }

    #[test]
    fn read_mem_available_returns_plausible_value_on_linux() {
        // On any Linux host this should yield a positive value; on other
        // platforms it returns None, which is fine — the SpillPolicy treats
        // 0 as "tight" and falls back to disk, which is the safe behaviour.
        if let Some(bytes) = read_mem_available() {
            assert!(bytes > 0, "MemAvailable parsed as zero");
            assert!(bytes < 1024_u64.pow(5), "implausibly large MemAvailable");
        }
    }

    #[test]
    fn probe_caches_within_ttl() {
        let mut probe = MemoryProbe::new();
        probe.refresh_if_stale();
        let first = probe.last_read;
        // Immediately re-checking should NOT update last_read.
        probe.refresh_if_stale();
        assert_eq!(probe.last_read, first);
    }

    #[test]
    fn passthrough_policy_returns_original_prefix() {
        let p = SpillPolicy::passthrough("/foo/bar".into());
        assert_eq!(p.next_prefix(), "/foo/bar");
    }
}
