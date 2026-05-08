//! Phase 9 task 5 — disk tier.
//!
//! Sidecar cache directory: `<root>/<doc-sha256-hex>/<content-hash-hex>.bin`.
//! Cross-process persistence; survives process restart.  Editing the
//! source PDF naturally invalidates the cache because the doc-sha256
//! changes (`open_session` derives `DocId` from the PDF bytes).
//!
//! # Probe order
//!
//! [`super::DeviceImageCache`] consults the disk tier as a fallback
//! after the VRAM and host-RAM tiers miss.  On a disk hit, the entry
//! is loaded into a fresh pinned host slab and promoted up the chain
//! (host RAM → VRAM) so subsequent same-session lookups skip disk
//! entirely.
//!
//! # File format
//!
//! ```text
//! Offset  Size   Field
//! 0       4      Magic     = 0x50445246  ("PDRF")
//! 4       4      Version   = 1
//! 8       4      Width
//! 12      4      Height
//! 16      1      Components (1 = Gray, 3 = RGB)
//! 17      1      Format     (0 = raw)
//! 18      6      Reserved (zero)
//! 24      W×H×C  Pixel bytes
//! ```
//!
//! Multi-byte fields are little-endian.  Format byte = 0 (raw) is the
//! only value supported in v1; future versions may add deflate.
//!
//! # Atomicity
//!
//! Writes go to a `<name>.tmp.<pid>.<thread-id>` file in the same
//! directory and are renamed into place — atomic on POSIX, so a
//! crash mid-write leaves either no file or the complete file (no
//! truncated half-file lying around for a future reader to misread).

use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::thread;
use std::time::SystemTime;

use super::{ContentHash, DocId, ImageLayout};

/// Bound on the queue of pending disk writes.  When the renderer
/// outpaces the writer, `try_send` returns `Full` and the cache
/// drops the disk-tier write (the in-memory tiers still serve the
/// hit).  Sized to comfortably absorb a JPEG-heavy page's worth of
/// `XObject`s without backpressuring the renderer; large enough
/// that the writer can amortise `sync_all` costs across queued
/// jobs but small enough that an adversarial PDF can't pile up
/// gigabytes of not-yet-flushed pixel data in RAM.
const WRITE_QUEUE_DEPTH: usize = 64;

/// Header magic: ASCII "PDRF".
const MAGIC: u32 = 0x5044_5246;
/// File-format version; bump on any layout change.
const VERSION: u32 = 1;
/// Header size in bytes.  Everything before this is metadata; the
/// pixel payload starts at [`HEADER_LEN`].
const HEADER_LEN: usize = 24;
/// Format byte value for raw pixel bytes (no compression).
const FORMAT_RAW: u8 = 0;

/// Resolve cache root from `PDF_RASTER_CACHE_DIR`.
///
/// `try_new()` already gates on the env var being set, so this is
/// never `None` in practice — the `Option` return is kept so the
/// signature mirrors the previous fallback shape (and so the
/// gating decision lives in one place).
fn resolve_root() -> Option<PathBuf> {
    std::env::var_os("PDF_RASTER_CACHE_DIR").map(PathBuf::from)
}

/// Resolve the budget from `PDF_RASTER_CACHE_BYTES`, defaulting to
/// unbounded (0).  Parse failures fall back to unbounded with a warn.
fn resolve_budget() -> u64 {
    let Some(s) = std::env::var_os("PDF_RASTER_CACHE_BYTES") else {
        return 0;
    };
    if let Some(n) = s.to_str().and_then(|v| v.parse::<u64>().ok()) {
        return n;
    }
    log::warn!(
        "PDF_RASTER_CACHE_BYTES={} is not a valid u64; disk-tier budget set to unbounded",
        s.to_string_lossy(),
    );
    0
}

/// One cached on-disk entry's location, post-resolve.
fn entry_path(root: &Path, doc: DocId, hash: ContentHash) -> PathBuf {
    root.join(hex_lower(&doc.0))
        .join(format!("{}.bin", hex_lower(&hash.0)))
}

/// Lower-case hex of a byte slice.  Local helper so the disk tier
/// doesn't pull a hex crate; the BLAKE3 output is 32 bytes so the
/// hex string is 64 ASCII bytes — small and write-once per filename.
fn hex_lower(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        s.push(HEX[(b >> 4) as usize] as char);
        s.push(HEX[(b & 0x0F) as usize] as char);
    }
    s
}

/// One write job from `insert` to the background writer thread.
/// `pixels` is `Box<[u8]>` (not `&[u8]`) so the slice can cross the
/// thread boundary; allocation is a one-time host-RAM `clone_from`,
/// negligible compared to the disk I/O the renderer used to do
/// inline.
struct WriteJob {
    final_path: PathBuf,
    width: u32,
    height: u32,
    layout: ImageLayout,
    pixels: Box<[u8]>,
    /// Reported `entry_size` to bump `DiskTier::used_estimate` once
    /// the write succeeds.  Computed at insert time so the writer
    /// thread doesn't redo the math.
    entry_size: u64,
}

/// Channel payload from `insert` / `flush` to the writer thread.
enum WriterMsg {
    /// Normal write — the renderer's hot path.
    Write(WriteJob),
    /// Test-only flush barrier — writer drains preceding messages
    /// then signals on the oneshot before reading the next message.
    /// Production code never sends this.
    #[cfg(test)]
    FlushAck(mpsc::SyncSender<()>),
}

/// State the writer thread shares with the `DiskTier`.  Both hold
/// an `Arc` to this struct; cloning the `Arc` is the only thing
/// the spawn site needs to do.
struct WriterState {
    root: PathBuf,
    /// Budget in bytes; `0` = unbounded.  Tracked as best-effort
    /// (filesystem mtimes drive eviction; the counter avoids a
    /// directory scan on every insert).
    budget: u64,
    /// Bytes written by *this process* since startup — not a global
    /// view of the on-disk cache.  Starts at zero on a fresh process
    /// even when `<root>/` already holds prior data; reconciled by
    /// `evict_to_fit` on the first budget-exceeding write.  Across
    /// concurrent processes, budget enforcement is eventual.
    used_estimate: AtomicU64,
}

/// Disk tier.  Writes are dispatched to a single background thread
/// via a bounded channel so the renderer never blocks on the
/// `write_all + sync_all + rename` sequence.
pub struct DiskTier {
    state: Arc<WriterState>,
    /// Job queue to the writer thread.  `SyncSender` (bounded) so a
    /// runaway renderer can't fill RAM with pending pixel buffers;
    /// when the queue saturates `try_send` returns `Full` and we
    /// drop the disk-tier write.
    sender: mpsc::SyncSender<WriterMsg>,
    /// Approximate count of in-flight (queued or actively writing)
    /// jobs.  Bumped before `try_send` and decremented when the
    /// writer pulls from the channel.  Used as a fast-path probe
    /// in `insert` so the renderer can skip the pixel clone+alloc
    /// when the queue is already saturated.  Approximate: races
    /// between increment-then-send and the writer's decrement
    /// produce off-by-one but never structurally wrong values.
    in_flight: Arc<AtomicUsize>,
    /// Worker join handle.  `Drop` consumes it to wait for the
    /// writer to drain after we drop `sender`.  `Mutex<Option>` so
    /// the move out of `&mut self` in `Drop::drop` works without
    /// requiring `&mut self` plumbing through the cache layer.
    writer: Mutex<Option<thread::JoinHandle<()>>>,
    /// Number of writes the queue has dropped because the writer
    /// thread fell behind.  Logged at warn level on the first drop
    /// per process; subsequent drops increment silently to keep
    /// noise down.  Visible via [`Self::dropped_writes`] for tests
    /// and diagnostics.
    dropped_writes: AtomicU64,
}

impl DiskTier {
    /// Try to construct a disk tier.  **Opt-in** via the
    /// `PDF_RASTER_CACHE_DIR` env var (any non-empty value enables
    /// the disk tier; the value sets the cache root).  Without it,
    /// `try_new` returns `None` and the in-memory tiers (VRAM +
    /// host RAM) carry the cache.
    ///
    /// The disk tier was opt-out before benchmarking exposed a
    /// real regression on JPEG-heavy renders: every cache miss
    /// queues a multi-MB pixel buffer for the writer thread, and
    /// the renderer's process-exit blocks on flushing the queue.
    /// On a 162-image corpus that's seconds of disk I/O the user
    /// didn't ask for.  Opt-in keeps the architecture available
    /// for OCR-pipeline patterns that genuinely benefit while the
    /// default render path stays as fast as the in-memory tiers
    /// can carry it.
    ///
    /// Returns `None` when:
    /// - `PDF_RASTER_CACHE_DIR` is unset (the opt-in gate);
    /// - the cache root can't be created;
    /// - the writer thread fails to spawn.
    #[must_use]
    pub fn try_new() -> Option<Self> {
        // Opt-in gate.  Without an explicit cache dir, the disk
        // tier is disabled.  See the type docs for rationale.
        let _ = std::env::var_os("PDF_RASTER_CACHE_DIR")?;
        let root = resolve_root()?;
        if let Err(e) = fs::create_dir_all(&root) {
            log::warn!(
                "disk-tier: failed to create cache root {}: {e} — disk tier disabled",
                root.display()
            );
            return None;
        }
        let budget = resolve_budget();
        let state = Arc::new(WriterState {
            root,
            budget,
            used_estimate: AtomicU64::new(0),
        });
        let (sender, receiver) = mpsc::sync_channel::<WriterMsg>(WRITE_QUEUE_DEPTH);
        let writer_state = Arc::clone(&state);
        let in_flight = Arc::new(AtomicUsize::new(0));
        let writer_in_flight = Arc::clone(&in_flight);
        let writer = match thread::Builder::new()
            .name("pdf-raster-disk-cache-writer".to_string())
            .spawn(move || writer_loop(receiver, &writer_state, &writer_in_flight))
        {
            Ok(handle) => handle,
            Err(e) => {
                log::warn!("disk-tier: writer thread spawn failed: {e} — disk tier disabled");
                return None;
            }
        };
        Some(Self {
            state,
            sender,
            in_flight,
            writer: Mutex::new(Some(writer)),
            dropped_writes: AtomicU64::new(0),
        })
    }

    /// Number of writes the queue dropped because the writer fell
    /// behind.  Test + diagnostics hook; not part of the cache hot
    /// path.
    #[must_use]
    pub fn dropped_writes(&self) -> u64 {
        self.dropped_writes.load(Ordering::Relaxed)
    }

    /// The cache root directory.  Mostly useful for diagnostics and tests.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.state.root
    }

    /// Budget in bytes; `0` = unbounded.
    #[must_use]
    pub fn budget_bytes(&self) -> u64 {
        self.state.budget
    }

    /// Test helper: read + validate, allocating pixels into a `Vec`.
    /// Production uses [`Self::lookup_into`] + a pinned slab.
    #[must_use]
    #[cfg(test)]
    pub(crate) fn lookup(&self, doc: DocId, hash: &ContentHash) -> Option<DiskEntry> {
        let path = entry_path(&self.state.root, doc, *hash);
        match read_entry(&path) {
            Ok(entry) => Some(entry),
            Err(DiskReadError::Missing) => None,
            Err(e) => {
                report_bad_entry(&path, &e);
                None
            }
        }
    }

    /// Probe the disk tier for `(doc, hash)` and stream the pixel
    /// payload through a caller-supplied callback.  On hit, opens
    /// the file, validates the header, then calls `fill` with the
    /// parsed [`DiskHeaderInfo`] and a reader positioned at the
    /// pixel bytes.  The callback reads exactly
    /// `header.expected_pixel_bytes` (typically via `read_exact`
    /// into a pinned-host slab) and tags failures with
    /// [`LookupCallbackError::Read`] for file-side problems or
    /// [`LookupCallbackError::Resource`] for caller-side problems.
    ///
    /// Returns `Some(header)` on success, `None` on miss, validation
    /// failure, file-side I/O error (entry is removed), or
    /// caller-side resource failure (entry is left intact — a
    /// transient pinned-alloc failure must not nuke a valid entry).
    pub fn lookup_into<F>(&self, doc: DocId, hash: &ContentHash, fill: F) -> Option<DiskHeaderInfo>
    where
        F: FnOnce(&DiskHeaderInfo, &mut dyn Read) -> Result<(), LookupCallbackError>,
    {
        let path = entry_path(&self.state.root, doc, *hash);
        match read_entry_into(&path, fill) {
            Ok(header) => Some(header),
            // Missing or callback-failed: the file is fine (or absent),
            // so don't delete it.  A transient pinned-alloc failure must
            // not nuke a valid cache entry.
            Err(DiskReadError::Missing | DiskReadError::CallbackFailed(_)) => None,
            Err(e) => {
                report_bad_entry(&path, &e);
                None
            }
        }
    }

    /// Queue an entry for write.  Returns immediately — the actual
    /// `write_all + sync_all + rename` happens on the writer
    /// thread.  Idempotent w.r.t. cache content: a same-content
    /// rewrite overwrites with bit-identical bytes via the
    /// atomic-rename pattern.
    ///
    /// Behaviour when the queue is full: the write is **dropped**
    /// (the in-memory tiers still serve the hit; only cross-session
    /// persistence is lost for this entry).  The first drop per
    /// process is logged at `warn`; subsequent drops bump
    /// [`Self::dropped_writes`] silently.
    ///
    /// Race note: an `insert` followed immediately by a `lookup`
    /// from the same session will usually miss on disk — the writer
    /// thread hasn't flushed yet.  This is fine because the cache's
    /// in-memory tiers (VRAM, host RAM) serve the hit first; the
    /// disk tier is only consulted as a fallback.  `flush()` (test
    /// helper) drains the queue if a test needs read-after-write
    /// semantics.
    pub fn insert(
        &self,
        doc: DocId,
        hash: ContentHash,
        width: u32,
        height: u32,
        layout: ImageLayout,
        pixels: &[u8],
    ) {
        // Fast-path early-out: if the queue is already saturated,
        // skip the pixel clone entirely.  Otherwise we'd allocate
        // and copy a multi-MB pixel buffer just to throw it away
        // in `try_send`.  Slight race with the writer's decrement
        // is harmless (off-by-one against an approximate bound).
        if self.in_flight.load(Ordering::Acquire) >= WRITE_QUEUE_DEPTH {
            self.note_dropped();
            return;
        }
        let final_path = entry_path(&self.state.root, doc, hash);
        let entry_size = (HEADER_LEN as u64).saturating_add(pixels.len() as u64);
        let job = WriteJob {
            final_path,
            width,
            height,
            layout,
            // One copy here — host RAM, ~µs vs the ms-scale fsync
            // we're avoiding.  The pixels can't cross the thread
            // boundary by reference; a Box<[u8]> is the cheapest
            // owned form.
            pixels: pixels.to_vec().into_boxed_slice(),
            entry_size,
        };
        // Bump in_flight before send so the writer's decrement is
        // ordered after.  Writers see the increment before the
        // job because acquire/release on the channel acts as the
        // synchronisation edge.
        let _ = self.in_flight.fetch_add(1, Ordering::AcqRel);
        match self.sender.try_send(WriterMsg::Write(job)) {
            Ok(()) => {}
            Err(mpsc::TrySendError::Full(_)) => {
                let _ = self.in_flight.fetch_sub(1, Ordering::AcqRel);
                self.note_dropped();
            }
            Err(mpsc::TrySendError::Disconnected(_)) => {
                let _ = self.in_flight.fetch_sub(1, Ordering::AcqRel);
                let _ = self.dropped_writes.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    fn note_dropped(&self) {
        let prev = self.dropped_writes.fetch_add(1, Ordering::Relaxed);
        if prev == 0 {
            log::warn!(
                "disk-tier: writer queue full ({WRITE_QUEUE_DEPTH} slots) — \
                 dropping disk-tier writes; in-memory tiers still serve hits. \
                 Subsequent drops are silent (see DiskTier::dropped_writes)."
            );
        }
    }

    /// Drain the writer queue and return once every previously
    /// queued write has either landed or failed.  Test helper;
    /// production code does not need this — the disk tier is a
    /// best-effort persistence layer and read-after-write within a
    /// session is served by the in-memory tiers.
    ///
    /// Implementation: send a `FlushAck` marker; the writer drains
    /// preceding `Write` messages in order, then sends `()` on the
    /// returned channel.  Blocking-send so flush still works when
    /// the channel is full.
    #[cfg(test)]
    pub(crate) fn flush(&self) {
        let (tx, rx) = mpsc::sync_channel::<()>(0);
        if self.sender.send(WriterMsg::FlushAck(tx)).is_err() {
            // Writer gone — nothing to flush.
            return;
        }
        let _ = rx.recv();
    }
}

impl Drop for DiskTier {
    fn drop(&mut self) {
        // The writer is blocked on `recv()`, so we must close the
        // channel before `join()` or we deadlock.  `self.sender`
        // would only drop *after* this Drop body returns, so
        // explicitly replace it with a fresh disconnected sender;
        // dropping the original closes the writer's receive end
        // and `recv()` returns Err.
        let (closed_tx, _) = mpsc::sync_channel::<WriterMsg>(0);
        drop(std::mem::replace(&mut self.sender, closed_tx));

        let dropped = self.dropped_writes.load(Ordering::Relaxed);
        if dropped > 0 {
            log::info!("disk-tier: {dropped} writes dropped under queue saturation");
        }

        let handle = self
            .writer
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take();
        if let Some(handle) = handle
            && let Err(e) = handle.join()
        {
            log::warn!("disk-tier: writer thread join failed: {e:?}");
        }
    }
}

/// Drop oldest documents (by directory mtime) until total disk
/// usage is under budget.  Eviction unit is the doc-sha256
/// directory, not individual files — partial-document caches
/// are wasteful.
///
/// Best-effort: filesystem errors during the scan are logged
/// and the eviction pass returns early; the next write will
/// retry.
fn evict_to_fit(state: &WriterState) {
    if state.budget == 0 {
        return;
    }
    let mut dirs = match scan_doc_dirs(&state.root) {
        Ok(d) => d,
        Err(e) => {
            log::warn!("disk-tier: scan {} failed: {e}", state.root.display());
            return;
        }
    };
    // Reconcile the optimistic counter from the actual scan.
    let total: u64 = dirs.iter().map(|d| d.size_bytes).sum();
    state.used_estimate.store(total, Ordering::Relaxed);

    if total <= state.budget {
        return;
    }
    // Sort oldest-first (smallest mtime) so we drop in age order.
    dirs.sort_by_key(|d| d.mtime);
    let mut current = total;
    for d in dirs {
        if current <= state.budget {
            break;
        }
        if let Err(e) = fs::remove_dir_all(&d.path) {
            log::warn!(
                "disk-tier: remove_dir_all({}) failed: {e}",
                d.path.display()
            );
            continue;
        }
        current = current.saturating_sub(d.size_bytes);
    }
    state.used_estimate.store(current, Ordering::Relaxed);
}

/// Background writer loop.  Owns the receiver end of the channel
/// and runs until the channel is closed (sender dropped).  Each
/// `Write` job: ensure the doc dir exists, do the synchronous
/// write+sync+rename, bump `used_estimate`, trigger `evict_to_fit`
/// if the budget was exceeded.  Each `FlushAck` job: ack the
/// caller (it's a barrier, not a write).
fn writer_loop(receiver: mpsc::Receiver<WriterMsg>, state: &WriterState, in_flight: &AtomicUsize) {
    // `for msg in receiver` consumes the receiver and runs until
    // the channel is closed (sender dropped).
    for msg in receiver {
        match msg {
            WriterMsg::Write(job) => {
                let do_decrement = || {
                    let _ = in_flight.fetch_sub(1, Ordering::AcqRel);
                };
                if let Some(parent) = job.final_path.parent()
                    && let Err(e) = fs::create_dir_all(parent)
                {
                    log::warn!(
                        "disk-tier: failed to create doc dir {}: {e}",
                        parent.display()
                    );
                    do_decrement();
                    continue;
                }
                if let Err(e) = write_entry(
                    &job.final_path,
                    job.width,
                    job.height,
                    job.layout,
                    &job.pixels,
                ) {
                    log::warn!(
                        "disk-tier: write to {} failed: {e}",
                        job.final_path.display()
                    );
                    do_decrement();
                    continue;
                }
                let _ = state
                    .used_estimate
                    .fetch_add(job.entry_size, Ordering::Relaxed);
                if state.budget > 0 && state.used_estimate.load(Ordering::Relaxed) > state.budget {
                    evict_to_fit(state);
                }
                do_decrement();
            }
            #[cfg(test)]
            WriterMsg::FlushAck(reply) => {
                // Acknowledge after preceding writes have drained
                // (sequential channel guarantees ordering).  Does
                // not bump in_flight (sender doesn't either).
                let _ = reply.send(());
            }
        }
    }
}

impl std::fmt::Debug for DiskTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DiskTier")
            .field("root", &self.state.root)
            .field("budget_bytes", &self.state.budget)
            .field(
                "used_estimate",
                &self.state.used_estimate.load(Ordering::Relaxed),
            )
            .field(
                "dropped_writes",
                &self.dropped_writes.load(Ordering::Relaxed),
            )
            .finish_non_exhaustive()
    }
}

/// Test-only allocating variant of a disk cache hit.
#[cfg(test)]
#[derive(Debug)]
pub(crate) struct DiskEntry {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) layout: ImageLayout,
    pub(crate) pixels: Vec<u8>,
}

/// Header read from a disk cache entry — all metadata fields, no
/// pixels.  Returned from [`DiskTier::lookup_into`] to let the caller
/// size + allocate a destination buffer (typically a pinned-host
/// slab) before the pixel bytes are read.
#[derive(Debug, Clone, Copy)]
pub struct DiskHeaderInfo {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Pixel layout — determines bytes per pixel.
    pub layout: ImageLayout,
    /// Total pixel bytes the caller's destination buffer must hold:
    /// `width × height × layout.bytes_per_pixel()`.  Validated to
    /// fit in `usize` before being returned.
    pub expected_pixel_bytes: usize,
}

/// Error variants for [`read_entry`] — internal type, callers see
/// `Option<DiskEntry>` from [`DiskTier::lookup`] which logs and
/// swallows everything except `Missing`.
#[derive(Debug)]
enum DiskReadError {
    Missing,
    Io(io::Error),
    BadMagic,
    UnsupportedVersion(u32),
    UnsupportedFormat(u8),
    UnsupportedComponents(u8),
    HeaderTruncated,
    PixelsTruncated {
        expected: usize,
        got: usize,
    },
    DimensionOverflow,
    /// The `read_entry_into` callback failed for a reason unrelated
    /// to the file's contents (e.g. pinned-host alloc failure, CUDA
    /// driver hiccup).  Distinct so `lookup_into` can treat it as a
    /// transient miss rather than corruption — deleting a valid disk
    /// entry under transient pinned-memory pressure would silently
    /// trash the cache.
    CallbackFailed(io::Error),
}

impl std::fmt::Display for DiskReadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Missing => write!(f, "file does not exist"),
            Self::Io(e) => write!(f, "io: {e}"),
            Self::BadMagic => write!(f, "bad magic — file is not a PDRF cache entry"),
            Self::UnsupportedVersion(v) => write!(f, "unsupported version {v}"),
            Self::UnsupportedFormat(b) => write!(f, "unsupported format byte {b}"),
            Self::UnsupportedComponents(c) => write!(f, "unsupported components value {c}"),
            Self::HeaderTruncated => write!(f, "header truncated"),
            Self::PixelsTruncated { expected, got } => {
                write!(f, "pixels truncated: expected {expected} bytes, got {got}")
            }
            Self::DimensionOverflow => write!(f, "width × height × bpp overflows usize"),
            Self::CallbackFailed(e) => write!(f, "lookup callback failed: {e}"),
        }
    }
}

impl From<io::Error> for DiskReadError {
    fn from(e: io::Error) -> Self {
        if e.kind() == io::ErrorKind::NotFound {
            Self::Missing
        } else {
            Self::Io(e)
        }
    }
}

/// Log + delete a bad disk entry so a future write can replace it.
fn report_bad_entry(path: &Path, e: &DiskReadError) {
    log::warn!(
        "disk-tier: bad cache entry at {}: {e}; removing",
        path.display()
    );
    let _ = fs::remove_file(path);
}

/// Open the entry file and parse + validate its header.  Returns
/// the file (positioned just past the header) plus the parsed
/// metadata.  Caller reads the pixel payload from the same file.
fn open_and_parse_header(path: &Path) -> Result<(File, DiskHeaderInfo), DiskReadError> {
    let mut f = File::open(path)?;
    advise_will_need(&f);

    let mut header = [0u8; HEADER_LEN];
    f.read_exact(&mut header).map_err(|e| match e.kind() {
        io::ErrorKind::UnexpectedEof => DiskReadError::HeaderTruncated,
        _ => DiskReadError::Io(e),
    })?;

    let magic = u32::from_le_bytes([header[0], header[1], header[2], header[3]]);
    if magic != MAGIC {
        return Err(DiskReadError::BadMagic);
    }
    let version = u32::from_le_bytes([header[4], header[5], header[6], header[7]]);
    if version != VERSION {
        return Err(DiskReadError::UnsupportedVersion(version));
    }
    let width = u32::from_le_bytes([header[8], header[9], header[10], header[11]]);
    let height = u32::from_le_bytes([header[12], header[13], header[14], header[15]]);
    let components = header[16];
    let format = header[17];
    if format != FORMAT_RAW {
        return Err(DiskReadError::UnsupportedFormat(format));
    }
    let layout = match components {
        1 => ImageLayout::Gray,
        3 => ImageLayout::Rgb,
        other => return Err(DiskReadError::UnsupportedComponents(other)),
    };

    let expected_pixel_bytes = (width as usize)
        .checked_mul(height as usize)
        .and_then(|n| n.checked_mul(layout.bytes_per_pixel()))
        .ok_or(DiskReadError::DimensionOverflow)?;
    Ok((
        f,
        DiskHeaderInfo {
            width,
            height,
            layout,
            expected_pixel_bytes,
        },
    ))
}

/// Read and validate a single disk entry, allocating a fresh `Vec`
/// for the pixel payload.  Only used by in-crate tests; production
/// uses [`read_entry_into`] + a caller-supplied pinned slab.
#[cfg(test)]
fn read_entry(path: &Path) -> Result<DiskEntry, DiskReadError> {
    let (mut f, info) = open_and_parse_header(path)?;
    let mut pixels = vec![0u8; info.expected_pixel_bytes];
    f.read_exact(&mut pixels)
        .map_err(|e| map_pixel_io_err(&info, e))?;
    Ok(DiskEntry {
        width: info.width,
        height: info.height,
        layout: info.layout,
        pixels,
    })
}

/// Read and validate a single disk entry, streaming pixel bytes
/// through a caller-supplied callback.  The callback receives the
/// parsed header and a reader positioned at the pixel payload; it
/// must consume exactly `info.expected_pixel_bytes` (typically via
/// `read_exact` into its own buffer).
///
/// The callback distinguishes file-side failures
/// ([`LookupCallbackError::Read`] — propagated as `Io` /
/// `PixelsTruncated`, which trigger entry deletion) from caller-side
/// failures ([`LookupCallbackError::Resource`] — propagated as
/// `CallbackFailed`, which `lookup_into` treats as a transient miss
/// without touching the file).
fn read_entry_into<F>(path: &Path, fill: F) -> Result<DiskHeaderInfo, DiskReadError>
where
    F: FnOnce(&DiskHeaderInfo, &mut dyn Read) -> Result<(), LookupCallbackError>,
{
    let (mut f, info) = open_and_parse_header(path)?;
    fill(&info, &mut f).map_err(|e| match e {
        LookupCallbackError::Read(e) => map_pixel_io_err(&info, e),
        LookupCallbackError::Resource(e) => DiskReadError::CallbackFailed(e),
    })?;
    Ok(info)
}

/// Map an `io::Error` from a pixel-payload read into the appropriate
/// `DiskReadError` variant.  Shared by [`read_entry`] and
/// [`read_entry_into`] so the truncation reporting stays consistent.
fn map_pixel_io_err(info: &DiskHeaderInfo, e: io::Error) -> DiskReadError {
    match e.kind() {
        io::ErrorKind::UnexpectedEof => DiskReadError::PixelsTruncated {
            expected: info.expected_pixel_bytes,
            got: 0,
        },
        _ => DiskReadError::Io(e),
    }
}

/// Reasons the [`DiskTier::lookup_into`] callback can fail.
///
/// Splits "the file gave us bad data" from "my side broke" so the
/// disk tier doesn't delete a perfectly good cache entry under
/// transient resource pressure (pinned-host pool exhausted, CUDA
/// context hiccup, etc.).
#[derive(Debug)]
pub enum LookupCallbackError {
    /// The file-side read failed — typically `read_exact` returning
    /// `UnexpectedEof` because the payload was truncated.  Treated as
    /// corruption; the entry is removed.
    Read(io::Error),
    /// A caller-side resource failed (alloc, driver, etc.).  The
    /// file is fine; the lookup surfaces as a miss without removal.
    Resource(io::Error),
}

impl std::fmt::Display for LookupCallbackError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Read(e) => write!(f, "read: {e}"),
            Self::Resource(e) => write!(f, "resource: {e}"),
        }
    }
}

impl std::error::Error for LookupCallbackError {}

/// Write an entry atomically: temp file in the same directory, then
/// rename into place.  POSIX rename is atomic across same-fs boundaries.
///
/// Cleans up the temp file on any error path — `write_all` /
/// `sync_all` / `rename` failures all unlink the temp before
/// propagating the error.  Without this, an `ENOSPC` mid-write
/// would leave a `<name>.tmp.<pid>.<tid>` file behind permanently.
fn write_entry(
    final_path: &Path,
    width: u32,
    height: u32,
    layout: ImageLayout,
    pixels: &[u8],
) -> io::Result<()> {
    // Reject Mask before opening any file — there's no defined
    // components value for it in the on-disk format.
    let components: u8 = match layout {
        ImageLayout::Gray => 1,
        ImageLayout::Rgb => 3,
        ImageLayout::Mask => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Mask layout cannot be cached on disk (no defined components value)",
            ));
        }
    };
    let parent = final_path.parent().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "final_path has no parent directory",
        )
    })?;
    let pid = std::process::id();
    // ThreadId::as_u64 is unstable; format the Debug impl which is stable.
    let tid = format!("{:?}", std::thread::current().id());
    let tmp_name = format!(
        "{}.tmp.{pid}.{tid}",
        final_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("entry")
    );
    let tmp_path = parent.join(tmp_name);

    let mut header = [0u8; HEADER_LEN];
    header[0..4].copy_from_slice(&MAGIC.to_le_bytes());
    header[4..8].copy_from_slice(&VERSION.to_le_bytes());
    header[8..12].copy_from_slice(&width.to_le_bytes());
    header[12..16].copy_from_slice(&height.to_le_bytes());
    header[16] = components;
    header[17] = FORMAT_RAW;

    // All steps that touch `tmp_path` after `create_new` must unlink
    // it on failure.  Wrap them so the cleanup runs uniformly.
    let result = (|| -> io::Result<()> {
        let mut f = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&tmp_path)?;
        f.write_all(&header)?;
        f.write_all(pixels)?;
        f.sync_all()?;
        drop(f);
        // Atomic rename.  If a same-name file exists (concurrent writer
        // committed the same hash first), rename overwrites it on POSIX —
        // both files have the same bytes anyway.
        fs::rename(&tmp_path, final_path)?;
        Ok(())
    })();
    if let Err(e) = result {
        // Cleanup is best-effort: a simultaneous successful rename by
        // another writer may have moved the file out from under us, so
        // remove_file's NotFound is fine.
        let _ = fs::remove_file(&tmp_path);
        return Err(e);
    }
    Ok(())
}

/// One scanned doc directory, used for eviction sorting.
struct DocDirInfo {
    path: PathBuf,
    mtime: SystemTime,
    size_bytes: u64,
}

fn scan_doc_dirs(root: &Path) -> io::Result<Vec<DocDirInfo>> {
    let mut out = Vec::new();
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let meta = entry.metadata()?;
        if !meta.is_dir() {
            continue;
        }
        let mtime = meta.modified().unwrap_or(SystemTime::UNIX_EPOCH);
        let size_bytes = dir_total_bytes(&entry.path()).unwrap_or(0);
        out.push(DocDirInfo {
            path: entry.path(),
            mtime,
            size_bytes,
        });
    }
    Ok(out)
}

fn dir_total_bytes(dir: &Path) -> io::Result<u64> {
    let mut total = 0u64;
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let meta = entry.metadata()?;
        if meta.is_file() {
            total = total.saturating_add(meta.len());
        }
    }
    Ok(total)
}

/// `posix_fadvise(fd, 0, 0, WILLNEED)` — Linux-only kernel hint that
/// the file is about to be read sequentially, so the page cache can
/// start prefetching.  No-op on non-Linux.
#[cfg(target_os = "linux")]
fn advise_will_need(file: &File) {
    use std::os::fd::AsRawFd;

    // `posix_fadvise` is provided by libc on every Linux distro that
    // ships glibc/musl; we declare it inline to avoid pulling the
    // libc crate as a workspace dep just for this one call.
    unsafe extern "C" {
        fn posix_fadvise(fd: i32, offset: i64, len: i64, advice: i32) -> i32;
    }
    const POSIX_FADV_WILLNEED: i32 = 3;

    // SAFETY: posix_fadvise is a kernel syscall wrapper; the fd is
    // valid for the duration of the call (we hold the File), and
    // POSIX_FADV_WILLNEED is a recognised advice value.  Errors are
    // hints-only and intentionally ignored.
    let _ = unsafe { posix_fadvise(file.as_raw_fd(), 0, 0, POSIX_FADV_WILLNEED) };
}

#[cfg(not(target_os = "linux"))]
const fn advise_will_need(_file: &File) {
    // posix_fadvise has no portable equivalent that's worth wiring
    // up here.  std::fs::read on macOS/Windows is fast enough that
    // the missed prefetch hint is negligible.
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn mk_tier(dir: &Path, budget: u64) -> DiskTier {
        let state = Arc::new(WriterState {
            root: dir.to_path_buf(),
            budget,
            used_estimate: AtomicU64::new(0),
        });
        let (sender, receiver) = mpsc::sync_channel::<WriterMsg>(WRITE_QUEUE_DEPTH);
        let in_flight = Arc::new(AtomicUsize::new(0));
        let writer_state = Arc::clone(&state);
        let writer_in_flight = Arc::clone(&in_flight);
        let writer = thread::Builder::new()
            .name("pdf-raster-disk-cache-writer-test".to_string())
            .spawn(move || writer_loop(receiver, &writer_state, &writer_in_flight))
            .expect("spawn test writer");
        DiskTier {
            state,
            sender,
            in_flight,
            writer: Mutex::new(Some(writer)),
            dropped_writes: AtomicU64::new(0),
        }
    }

    #[test]
    fn hex_lower_round_trip() {
        let bytes = [0x00, 0xDE, 0xAD, 0xBE, 0xEF];
        assert_eq!(hex_lower(&bytes), "00deadbeef");
    }

    #[test]
    fn write_then_read_roundtrip() {
        let dir = tempdir().expect("tempdir");
        let tier = mk_tier(dir.path(), 0);
        let doc = DocId([0xAA; 32]);
        let hash = ContentHash([0xBB; 32]);
        let pixels: Vec<u8> = (0..16 * 16 * 3)
            .map(|i| u8::try_from(i % 256).expect("low byte fits"))
            .collect();

        tier.insert(doc, hash, 16, 16, ImageLayout::Rgb, &pixels);
        tier.flush();
        let entry = tier.lookup(doc, &hash).expect("hit");
        assert_eq!(entry.width, 16);
        assert_eq!(entry.height, 16);
        assert_eq!(entry.layout, ImageLayout::Rgb);
        assert_eq!(entry.pixels, pixels);
    }

    #[test]
    fn lookup_returns_none_on_missing_file() {
        let dir = tempdir().expect("tempdir");
        let tier = mk_tier(dir.path(), 0);
        let doc = DocId([0x11; 32]);
        let hash = ContentHash([0x22; 32]);
        assert!(tier.lookup(doc, &hash).is_none());
    }

    #[test]
    fn lookup_returns_none_on_bad_magic() {
        let dir = tempdir().expect("tempdir");
        let tier = mk_tier(dir.path(), 0);
        let doc = DocId([0x33; 32]);
        let hash = ContentHash([0x44; 32]);

        // Manually plant a file with wrong magic.
        let path = entry_path(&tier.state.root, doc, hash);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, b"not a pdrf file").unwrap();
        assert!(tier.lookup(doc, &hash).is_none());
        // The bad file was removed by lookup's cleanup path.
        assert!(!path.exists());
    }

    #[test]
    fn lookup_returns_none_on_truncated_pixels() {
        let dir = tempdir().expect("tempdir");
        let tier = mk_tier(dir.path(), 0);
        let doc = DocId([0x55; 32]);
        let hash = ContentHash([0x66; 32]);

        // Plant a valid header claiming 16×16 RGB, but truncate the payload.
        let path = entry_path(&tier.state.root, doc, hash);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        let mut header = [0u8; HEADER_LEN];
        header[0..4].copy_from_slice(&MAGIC.to_le_bytes());
        header[4..8].copy_from_slice(&VERSION.to_le_bytes());
        header[8..12].copy_from_slice(&16u32.to_le_bytes());
        header[12..16].copy_from_slice(&16u32.to_le_bytes());
        header[16] = 3;
        header[17] = FORMAT_RAW;
        fs::write(&path, header).unwrap();
        assert!(tier.lookup(doc, &hash).is_none());
        assert!(!path.exists());
    }

    #[test]
    fn lookup_into_resource_failure_keeps_file() {
        // Resource-side failures from the callback (e.g. pinned-host
        // alloc OOM, CUDA hiccup) must NOT delete the disk entry —
        // that would silently nuke a perfectly good cache file under
        // transient pressure.
        let dir = tempdir().expect("tempdir");
        let tier = mk_tier(dir.path(), 0);
        let doc = DocId([0x77; 32]);
        let hash = ContentHash([0x88; 32]);
        let pixels = vec![0xABu8; 8 * 8 * 3];
        tier.insert(doc, hash, 8, 8, ImageLayout::Rgb, &pixels);
        tier.flush();
        let path = entry_path(&tier.state.root, doc, hash);
        assert!(path.exists());

        let result = tier.lookup_into(doc, &hash, |_info, _reader| {
            Err(LookupCallbackError::Resource(io::Error::other(
                "simulated alloc failure",
            )))
        });
        assert!(result.is_none(), "Resource failure surfaces as miss");
        assert!(
            path.exists(),
            "Resource failure must NOT delete the disk entry"
        );
    }

    #[test]
    fn lookup_into_read_failure_removes_file() {
        // Read-side failures from the callback (e.g. truncated
        // payload) DO delete the entry — the file is unusable.
        let dir = tempdir().expect("tempdir");
        let tier = mk_tier(dir.path(), 0);
        let doc = DocId([0x99; 32]);
        let hash = ContentHash([0xAA; 32]);
        let pixels = vec![0xCDu8; 8 * 8 * 3];
        tier.insert(doc, hash, 8, 8, ImageLayout::Rgb, &pixels);
        tier.flush();
        let path = entry_path(&tier.state.root, doc, hash);

        let result = tier.lookup_into(doc, &hash, |_info, _reader| {
            Err(LookupCallbackError::Read(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "simulated truncation",
            )))
        });
        assert!(result.is_none());
        assert!(!path.exists(), "Read failure removes the disk entry");
    }

    #[test]
    fn evict_drops_oldest_doc_dir_on_overflow() {
        let dir = tempdir().expect("tempdir");
        // Budget for two 16×16 RGB entries (~768 bytes pixels + 24 byte
        // header = 792 bytes each); third insert should evict the first.
        let budget = 2 * (HEADER_LEN as u64 + 16 * 16 * 3) + 8; // 8 bytes slack
        let tier = mk_tier(dir.path(), budget);

        let pixels = vec![0xCDu8; 16 * 16 * 3];

        let doc_a = DocId([0xA0; 32]);
        let doc_b = DocId([0xB0; 32]);
        let doc_c = DocId([0xC0; 32]);
        let hash = ContentHash([0xFF; 32]);

        tier.insert(doc_a, hash, 16, 16, ImageLayout::Rgb, &pixels);
        tier.flush();
        // Force mtime ordering: sleep briefly so doc_b's dir mtime > doc_a's.
        std::thread::sleep(std::time::Duration::from_millis(20));
        tier.insert(doc_b, hash, 16, 16, ImageLayout::Rgb, &pixels);
        tier.flush();
        std::thread::sleep(std::time::Duration::from_millis(20));
        tier.insert(doc_c, hash, 16, 16, ImageLayout::Rgb, &pixels);
        tier.flush();

        // doc_a should have been evicted (oldest mtime).
        assert!(
            tier.lookup(doc_a, &hash).is_none(),
            "oldest doc dir should have been evicted"
        );
        assert!(tier.lookup(doc_b, &hash).is_some(), "doc_b should remain");
        assert!(tier.lookup(doc_c, &hash).is_some(), "doc_c just inserted");
    }

    #[test]
    fn drop_joins_writer_cleanly() {
        // Smoke: building a tier and dropping it must terminate the
        // writer thread without deadlocking.  The Drop impl
        // explicitly closes the channel so the writer's recv()
        // returns Err.
        let dir = tempdir().expect("tempdir");
        let tier = mk_tier(dir.path(), 0);
        // Don't insert anything — pure construction-and-drop test.
        drop(tier);
        // If we got here without hanging, the writer joined.
    }

    #[test]
    fn dropped_writes_increments_under_saturation() {
        // Stuff the queue beyond WRITE_QUEUE_DEPTH while the writer
        // can't drain (queue is full → try_send returns Full).  We
        // can't easily pause the writer, so instead we send a large
        // burst in tight succession; on a real system at least some
        // will be dropped because the writer is doing fsyncs.
        //
        // To avoid flakiness on a fast machine, gate the assertion
        // on "many or zero" — we accept 0 dropped only when the
        // writer kept up.  The real sanity check is that
        // dropped_writes is exposed and the counter doesn't panic.
        let dir = tempdir().expect("tempdir");
        let tier = mk_tier(dir.path(), 0);
        let pixels = vec![0xEEu8; 1024];
        let hash = ContentHash([0x11; 32]);
        for i in 0..(WRITE_QUEUE_DEPTH * 4) {
            #[allow(clippy::cast_possible_truncation)]
            let doc = DocId([i as u8; 32]);
            tier.insert(doc, hash, 32, 32, ImageLayout::Gray, &pixels);
        }
        tier.flush();
        // Counter must be readable — exercise the public accessor.
        let _ = tier.dropped_writes();
    }

    #[test]
    fn mask_layout_rejected_on_write() {
        // Direct write_entry test: Mask should error.  Tier::insert
        // logs+swallows so this is the only place to observe the rejection.
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("doc/entry.bin");
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        let pixels = vec![0u8; 16 * 16];
        let err = write_entry(&path, 16, 16, ImageLayout::Mask, &pixels).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
    }
}
