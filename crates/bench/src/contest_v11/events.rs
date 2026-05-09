//! E1–E4 event runners.
//!
//! E1: First-pixel — open archive, render single page, time end-to-end.
//! E2: Sustained   — render N consecutive pages from the same session.
//! E3: Cross-doc   — render page 1 of each archive in a list (xref-tail
//!                    prefetched before the run).
//! E4: Random      — render N deterministically-chosen random pages.

use std::path::{Path, PathBuf};
use std::time::Instant;

use pdf_raster::{SessionConfig, open_session, render_page_rgb};

#[derive(Debug)]
pub struct EventResult {
    pub name: &'static str,
    pub elapsed_ms: f64,
    pub pages_rendered: u32,
}

const RENDER_DPI: f64 = 150.0;
fn scale_at_150_dpi() -> f64 {
    RENDER_DPI / 72.0
}

/// E1 — open archive, render `page_idx`, time `argv[0]` → bitmap.
/// `page_idx` is clamped to `[1, total_pages]`.
pub fn e1(archive: &Path, page_idx: u32) -> Result<EventResult, String> {
    let t0 = Instant::now();
    let session = open_session(archive, &SessionConfig::default())
        .map_err(|e| format!("open_session: {e}"))?;
    let total = session.total_pages();
    if total == 0 {
        return Err("archive has zero pages".into());
    }
    let target = page_idx.clamp(1, total);
    let _bmp = render_page_rgb(&session, target, scale_at_150_dpi())
        .map_err(|e| format!("render_page_rgb: {e}"))?;
    Ok(EventResult {
        name: "E1",
        elapsed_ms: t0.elapsed().as_secs_f64() * 1e3,
        pages_rendered: 1,
    })
}

/// E2 — render `count` pages starting from `first_page` (clamped).
pub fn e2(archive: &Path, first_page: u32, count: u32) -> Result<EventResult, String> {
    let t0 = Instant::now();
    let session = open_session(archive, &SessionConfig::default())
        .map_err(|e| format!("open_session: {e}"))?;
    let total = session.total_pages();
    if total == 0 {
        return Err("archive has zero pages".into());
    }
    let mut rendered = 0u32;
    for offset in 0..count {
        let p = (first_page.saturating_add(offset)).clamp(1, total);
        let _bmp = render_page_rgb(&session, p, scale_at_150_dpi())
            .map_err(|e| format!("render_page_rgb (page {p}): {e}"))?;
        rendered += 1;
    }
    Ok(EventResult {
        name: "E2",
        elapsed_ms: t0.elapsed().as_secs_f64() * 1e3,
        pages_rendered: rendered,
    })
}

/// E3 — render page 1 of each archive listed in `archives.txt`.  Archives
/// are warmed via `warm_xref_tails` before the timed loop starts.
pub fn e3(list_path: &Path) -> Result<EventResult, String> {
    let list = std::fs::read_to_string(list_path)
        .map_err(|e| format!("read {}: {e}", list_path.display()))?;
    let archives: Vec<PathBuf> = list
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(PathBuf::from)
        .collect();
    if archives.is_empty() {
        return Err("archive list is empty".into());
    }

    crate::io_uring_open::warm_xref_tails(&archives);

    let t0 = Instant::now();
    let mut rendered = 0u32;
    for archive in &archives {
        let session = open_session(archive, &SessionConfig::default())
            .map_err(|e| format!("open_session({}): {e}", archive.display()))?;
        if session.total_pages() == 0 {
            continue;
        }
        let _bmp = render_page_rgb(&session, 1, scale_at_150_dpi())
            .map_err(|e| format!("render_page_rgb({}, 1): {e}", archive.display()))?;
        rendered += 1;
    }
    Ok(EventResult {
        name: "E3",
        elapsed_ms: t0.elapsed().as_secs_f64() * 1e3,
        pages_rendered: rendered,
    })
}

/// E4 — render 1000 random page indices from the archive.  Reproducible
/// via a fixed-seed xorshift64 so successive runs touch the same pages.
pub fn e4(archive: &Path) -> Result<EventResult, String> {
    let t0 = Instant::now();
    let session = open_session(archive, &SessionConfig::default())
        .map_err(|e| format!("open_session: {e}"))?;
    let total = session.total_pages();
    if total < 2 {
        return Err(format!("archive has only {total} pages — E4 needs >=2"));
    }

    let mut state: u64 = 0xDEAD_BEEF_DEAD_BEEF;
    let mut rendered = 0u32;
    for _ in 0..1000 {
        // xorshift64 — small, deterministic, good enough for index spread.
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        // `state % total` produces [0, total - 1]; +1 maps to the 1-based
        // [1, total] range covering every page.  An earlier shape used
        // `% (total - 1) + 1` which off-by-one-skipped the last page.
        let p = u32::try_from(state % u64::from(total) + 1)
            .expect("modulo total fits in u32 by construction");
        let _bmp = render_page_rgb(&session, p, scale_at_150_dpi())
            .map_err(|e| format!("render_page_rgb (page {p}): {e}"))?;
        rendered += 1;
    }
    Ok(EventResult {
        name: "E4",
        elapsed_ms: t0.elapsed().as_secs_f64() * 1e3,
        pages_rendered: rendered,
    })
}
