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
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::SystemTime;

use super::{ContentHash, DocId, ImageLayout};

/// Header magic: ASCII "PDRF".
const MAGIC: u32 = 0x5044_5246;
/// File-format version; bump on any layout change.
const VERSION: u32 = 1;
/// Header size in bytes.  Everything before this is metadata; the
/// pixel payload starts at [`HEADER_LEN`].
const HEADER_LEN: usize = 24;
/// Format byte value for raw pixel bytes (no compression).
const FORMAT_RAW: u8 = 0;

/// Default cache root: `~/.cache/pdf-raster/`.  Uses `XDG_CACHE_HOME`
/// when set, falls back to `$HOME/.cache`.  Returns `None` when both
/// are unset (e.g. minimal sandboxed environments) — disk tier
/// disabled in that case.
fn default_cache_dir() -> Option<PathBuf> {
    if let Some(xdg) = std::env::var_os("XDG_CACHE_HOME") {
        return Some(PathBuf::from(xdg).join("pdf-raster"));
    }
    let home = std::env::var_os("HOME")?;
    Some(PathBuf::from(home).join(".cache").join("pdf-raster"))
}

/// Resolve cache root from env vars, falling back to defaults.
///
/// Env vars (per the Phase 9 spec):
/// - `PDF_RASTER_CACHE_DIR` overrides the cache root.
/// - `PDF_RASTER_CACHE_BYTES` overrides the budget (bytes; 0 = unbounded).
fn resolve_root() -> Option<PathBuf> {
    if let Some(custom) = std::env::var_os("PDF_RASTER_CACHE_DIR") {
        return Some(PathBuf::from(custom));
    }
    default_cache_dir()
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

/// Phase 9 disk tier.
pub struct DiskTier {
    root: PathBuf,
    /// Budget in bytes; `0` = unbounded.  Tracked as best-effort
    /// (filesystem mtimes drive eviction; the counter avoids a
    /// directory scan on every insert).
    budget: u64,
    /// Bytes written by *this process* since startup.  Counter rolls
    /// in `insert`; `evict_to_fit` does the real reconciliation by
    /// reading directory sizes when the counter exceeds the budget.
    used_estimate: AtomicU64,
}

impl DiskTier {
    /// Try to construct a disk tier.  Returns `None` when the cache
    /// root cannot be resolved (no `HOME`, no `XDG_CACHE_HOME`, no
    /// `PDF_RASTER_CACHE_DIR`) — disk tier disabled in that case.
    /// Creates the root directory if it doesn't exist.
    #[must_use]
    pub fn try_new() -> Option<Self> {
        let root = resolve_root()?;
        if let Err(e) = fs::create_dir_all(&root) {
            log::warn!(
                "disk-tier: failed to create cache root {}: {e} — disk tier disabled",
                root.display()
            );
            return None;
        }
        let budget = resolve_budget();
        Some(Self {
            root,
            budget,
            used_estimate: AtomicU64::new(0),
        })
    }

    /// The cache root directory.  Mostly useful for diagnostics and tests.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Budget in bytes; `0` = unbounded.
    #[must_use]
    pub const fn budget_bytes(&self) -> u64 {
        self.budget
    }

    /// Probe the disk tier for `(doc, hash)`.  On hit, returns the
    /// decoded image as `(width, height, layout, pixels)` — caller
    /// builds a `HostEntry` and promotes through the upper tiers.
    /// Returns `None` on miss, malformed file, or any I/O error.
    #[must_use]
    pub fn lookup(&self, doc: DocId, hash: &ContentHash) -> Option<DiskEntry> {
        let path = entry_path(&self.root, doc, *hash);
        match read_entry(&path) {
            Ok(entry) => Some(entry),
            Err(DiskReadError::Missing) => None,
            Err(e) => {
                // Corrupt or truncated file — log, treat as miss, and
                // delete the bad file so a future write can replace it.
                log::warn!(
                    "disk-tier: bad cache entry at {}: {e}; removing",
                    path.display()
                );
                let _ = fs::remove_file(&path);
                None
            }
        }
    }

    /// Write an entry to disk.  Idempotent: a same-content rewrite
    /// is a no-op via the atomic-rename pattern.  Errors are logged
    /// and swallowed; disk-tier writes are best-effort and never
    /// block the caller's render path.
    pub fn insert(
        &self,
        doc: DocId,
        hash: ContentHash,
        width: u32,
        height: u32,
        layout: ImageLayout,
        pixels: &[u8],
    ) {
        let doc_dir = self.root.join(hex_lower(&doc.0));
        if let Err(e) = fs::create_dir_all(&doc_dir) {
            log::warn!(
                "disk-tier: failed to create doc dir {}: {e}",
                doc_dir.display()
            );
            return;
        }
        let final_path = doc_dir.join(format!("{}.bin", hex_lower(&hash.0)));
        if let Err(e) = write_entry(&final_path, width, height, layout, pixels) {
            log::warn!("disk-tier: write to {} failed: {e}", final_path.display());
            return;
        }
        // Bump the used estimate; reconcile against the actual
        // directory size on the next eviction pass.
        let entry_size = (HEADER_LEN as u64).saturating_add(pixels.len() as u64);
        let _ = self.used_estimate.fetch_add(entry_size, Ordering::Relaxed);

        // Trigger eviction only when the optimistic estimate exceeds
        // the budget.  Avoids a directory scan on every insert.
        if self.budget > 0 && self.used_estimate.load(Ordering::Relaxed) > self.budget {
            self.evict_to_fit();
        }
    }

    /// Drop oldest documents (by directory mtime) until total disk
    /// usage is under budget.  Eviction unit is the doc-sha256
    /// directory, not individual files — partial-document caches
    /// are wasteful.
    ///
    /// Best-effort: filesystem errors during the scan are logged
    /// and the eviction pass returns early; the next insert will
    /// retry.
    fn evict_to_fit(&self) {
        if self.budget == 0 {
            return;
        }
        let mut dirs = match scan_doc_dirs(&self.root) {
            Ok(d) => d,
            Err(e) => {
                log::warn!("disk-tier: scan {} failed: {e}", self.root.display());
                return;
            }
        };
        // Reconcile the optimistic counter from the actual scan.
        let total: u64 = dirs.iter().map(|d| d.size_bytes).sum();
        self.used_estimate.store(total, Ordering::Relaxed);

        if total <= self.budget {
            return;
        }
        // Sort oldest-first (smallest mtime) so we drop in age order.
        dirs.sort_by_key(|d| d.mtime);
        let mut current = total;
        for d in dirs {
            if current <= self.budget {
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
        self.used_estimate.store(current, Ordering::Relaxed);
    }
}

impl std::fmt::Debug for DiskTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DiskTier")
            .field("root", &self.root)
            .field("budget_bytes", &self.budget)
            .field("used_estimate", &self.used_estimate.load(Ordering::Relaxed))
            .finish()
    }
}

/// Decoded contents of a disk cache hit, ready for promotion.
#[derive(Debug)]
pub struct DiskEntry {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Pixel layout — determines bytes per pixel.
    pub layout: ImageLayout,
    /// Raw pixel bytes; length = `width × height × layout.bytes_per_pixel()`.
    pub pixels: Vec<u8>,
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
    PixelsTruncated { expected: usize, got: usize },
    DimensionOverflow,
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

/// Read and validate a single disk entry.
fn read_entry(path: &Path) -> Result<DiskEntry, DiskReadError> {
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

    let expected = (width as usize)
        .checked_mul(height as usize)
        .and_then(|n| n.checked_mul(layout.bytes_per_pixel()))
        .ok_or(DiskReadError::DimensionOverflow)?;
    let mut pixels = vec![0u8; expected];
    f.read_exact(&mut pixels).map_err(|e| match e.kind() {
        io::ErrorKind::UnexpectedEof => DiskReadError::PixelsTruncated {
            expected,
            got: pixels.len(),
        },
        _ => DiskReadError::Io(e),
    })?;
    Ok(DiskEntry {
        width,
        height,
        layout,
        pixels,
    })
}

/// Write an entry atomically: temp file in the same directory, then
/// rename into place.  POSIX rename is atomic across same-fs boundaries.
fn write_entry(
    final_path: &Path,
    width: u32,
    height: u32,
    layout: ImageLayout,
    pixels: &[u8],
) -> io::Result<()> {
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

    {
        let mut f = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&tmp_path)?;
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
        let mut header = [0u8; HEADER_LEN];
        header[0..4].copy_from_slice(&MAGIC.to_le_bytes());
        header[4..8].copy_from_slice(&VERSION.to_le_bytes());
        header[8..12].copy_from_slice(&width.to_le_bytes());
        header[12..16].copy_from_slice(&height.to_le_bytes());
        header[16] = components;
        header[17] = FORMAT_RAW;
        // header[18..24] stays zero (Reserved).
        f.write_all(&header)?;
        f.write_all(pixels)?;
        f.sync_all()?;
    }
    // Atomic rename.  If a same-name file exists (concurrent writer
    // committed the same hash first), rename overwrites it on POSIX
    // — both files have the same bytes anyway, so this is fine.
    fs::rename(&tmp_path, final_path).inspect_err(|_| {
        // Clean up our temp file on failure so we don't leak.
        let _ = fs::remove_file(&tmp_path);
    })?;
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
        DiskTier {
            root: dir.to_path_buf(),
            budget,
            used_estimate: AtomicU64::new(0),
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
        let path = entry_path(&tier.root, doc, hash);
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
        let path = entry_path(&tier.root, doc, hash);
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
        // Force mtime ordering: sleep briefly so doc_b's dir mtime > doc_a's.
        std::thread::sleep(std::time::Duration::from_millis(20));
        tier.insert(doc_b, hash, 16, 16, ImageLayout::Rgb, &pixels);
        std::thread::sleep(std::time::Duration::from_millis(20));
        tier.insert(doc_c, hash, 16, 16, ImageLayout::Rgb, &pixels);

        // doc_a should have been evicted (oldest mtime).
        assert!(
            tier.lookup(doc_a, &hash).is_none(),
            "oldest doc dir should have been evicted"
        );
        assert!(tier.lookup(doc_b, &hash).is_some(), "doc_b should remain");
        assert!(tier.lookup(doc_c, &hash).is_some(), "doc_c just inserted");
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
