//! E3-only batched xref-tail prefetch.
//!
//! Issues `posix_fadvise(POSIX_FADV_WILLNEED)` on the last 4 KB of each
//! archive in `paths`, where `startxref` lives.  This warms the kernel
//! page cache so the subsequent open+xref-parse for each archive doesn't
//! pay an I/O round-trip.  The plan originally proposed `io_uring`; we
//! use `posix_fadvise` instead because the kernel hint is the same and
//! no async abstraction is needed for our access pattern.
//!
//! Best-effort — failures are silently ignored.  No-op on non-Linux.

use std::path::PathBuf;

#[cfg(target_os = "linux")]
pub fn warm_xref_tails(paths: &[PathBuf]) {
    use rustix::fs::{Advice, fadvise};
    use std::num::NonZeroU64;

    const TAIL_BYTES: u64 = 4096;
    let nz_len = NonZeroU64::new(TAIL_BYTES).expect("4096 is non-zero");
    for p in paths {
        let Ok(file) = std::fs::File::open(p) else {
            continue;
        };
        let Ok(meta) = file.metadata() else {
            continue;
        };
        let len = meta.len();
        let offset = len.saturating_sub(TAIL_BYTES);
        let _ = fadvise(&file, offset, Some(nz_len), Advice::WillNeed);
    }
}

#[cfg(not(target_os = "linux"))]
pub fn warm_xref_tails(_: &[PathBuf]) {}
