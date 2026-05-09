//! Kernel-hint helpers for the `Document` mmap.
//!
//! All helpers are no-ops on platforms without `posix_fadvise`.  The `pdf`
//! crate enforces `unsafe_code = "deny"`; we use `rustix` for safe syscall
//! wrappers.
//!
//! The hints are advisory.  A wrong hint never causes a correctness issue;
//! the worst outcome is suboptimal kernel page-cache behaviour.

#[cfg(unix)]
use std::fs::File;

/// Tell the kernel "we will touch arbitrary 4 KB ranges; don't readahead."
///
/// Call once on document open, before any byte access.  Errors are
/// deliberately ignored — the hint is advisory and a failed `posix_fadvise`
/// just means we keep default readahead behaviour.
///
/// The `len` argument is `None`, which the kernel interprets as "to EOF".
#[cfg(unix)]
pub fn advise_random(file: &File) {
    use rustix::fs::{Advice, fadvise};
    let _ = fadvise(file, 0, None, Advice::Random);
}

/// Tell the kernel "we'll touch this byte range soon; please prefetch."
///
/// Useful after the page-tree descent has located a page's content stream
/// byte range, so kernel readahead overlaps with parse work.
///
/// A zero `len` is treated as a no-op (rustix requires `NonZero` for the
/// length argument; the kernel's "to EOF" semantics are deliberately not
/// exposed here, since callers always know the precise byte range).
#[cfg(unix)]
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "exposed for content-stream prefetch; not yet wired in lib builds"
    )
)]
pub fn advise_willneed(file: &File, offset: u64, len: u64) {
    use rustix::fs::{Advice, fadvise};
    let Some(nz_len) = std::num::NonZeroU64::new(len) else {
        return;
    };
    let _ = fadvise(file, offset, Some(nz_len), Advice::WillNeed);
}

#[cfg(not(unix))]
pub fn advise_random(_: &std::fs::File) {}
#[cfg(not(unix))]
pub fn advise_willneed(_: &std::fs::File, _: u64, _: u64) {}

#[cfg(test)]
#[cfg(unix)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn advise_random_does_not_panic() {
        let mut tmp = tempfile::NamedTempFile::new().expect("create tmp");
        tmp.write_all(b"hello world").expect("write");
        let file = tmp.reopen().expect("reopen");
        advise_random(&file);
    }

    #[test]
    fn advise_willneed_does_not_panic() {
        let mut tmp = tempfile::NamedTempFile::new().expect("create tmp");
        tmp.write_all(b"hello world").expect("write");
        let file = tmp.reopen().expect("reopen");
        advise_willneed(&file, 0, 11);
    }

    #[test]
    fn advise_willneed_zero_length_is_no_op() {
        let mut tmp = tempfile::NamedTempFile::new().expect("create tmp");
        tmp.write_all(b"hello").expect("write");
        let file = tmp.reopen().expect("reopen");
        advise_willneed(&file, 0, 0);
    }
}
