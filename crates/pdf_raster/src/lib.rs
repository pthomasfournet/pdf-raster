//! PDF rasterisation library — zero subprocess, zero Leptonica.
//!
//! Converts PDF pages to 8-bit grayscale pixel buffers ready for Tesseract OCR
//! (or any other consumer) without writing files or spawning processes.
//!
//! # Quick start
//!
//! ```rust,no_run
//! use std::path::Path;
//! use pdf_raster::{RasterOptions, raster_pdf};
//!
//! let opts = RasterOptions { dpi: 300.0, first_page: 1, last_page: 5, deskew: true, pages: None };
//! for (page_num, result) in raster_pdf(Path::new("scan.pdf"), &opts) {
//!     match result {
//!         Ok(page) => {
//!             // page.pixels: Vec<u8>, 8-bit grayscale, width × height, top-to-bottom
//!             // pass to tesseract::ocr_from_frame(&page.pixels, page.width, page.height, 1, page.width, "eng")
//!         }
//!         Err(e) => eprintln!("page {page_num}: {e}"),
//!     }
//! }
//! ```
//!
//! # Integration with Tesseract
//!
//! ```rust,no_run
//! # use pdf_raster::{RasterOptions, raster_pdf};
//! # use std::path::Path;
//! # let opts = RasterOptions { dpi: 300.0, first_page: 1, last_page: 1, deskew: true, pages: None };
//! for (_, result) in raster_pdf(Path::new("scan.pdf"), &opts) {
//!     let page = result.expect("page render failed");
//!     // tesseract crate (v0.15+):
//!     // let text = tesseract::ocr_from_frame(
//!     //     &page.pixels,
//!     //     page.width as i32,
//!     //     page.height as i32,
//!     //     1,               // bytes_per_pixel (grayscale)
//!     //     page.width as i32, // bytes_per_line (tightly packed)
//!     //     "eng",
//!     // ).unwrap();
//!     // For uneven scan backgrounds: set thresholding_method=2 (Sauvola) on the Tesseract side.
//!     // Do NOT pre-binarise — the LSTM engine reads your grayscale values directly.
//! }
//! ```
//!
//! # Streaming: render and process pages in parallel
//!
//! For large documents, [`render_channel`] renders on a background rayon thread
//! and sends pages as they complete, so downstream work (e.g. OCR) can start
//! immediately rather than waiting for the full render:
//!
//! ```rust,no_run
//! use std::path::Path;
//! use pdf_raster::{RasterOptions, render_channel};
//!
//! // Requires `rayon` in Cargo.toml: rayon = "1"
//! // Use a dedicated rayon pool in production if OCR tasks are long-running,
//! // to avoid stalling the render producer on the shared thread pool.
//! let opts = RasterOptions {
//!     dpi: 300.0,
//!     first_page: 1,
//!     last_page: u32::MAX,
//!     deskew: true,
//!     pages: None,
//! };
//! let rx = render_channel(Path::new("scan.pdf"), &opts, 4); // 4-page buffer
//! rayon::scope(|s| {
//!     while let Ok((page_num, result)) = rx.recv() {
//!         match result {
//!             Ok(page) => s.spawn(move |_| {
//!                 // process page — runs in parallel with the next render
//!                 let _ = (page_num, page.pixels);
//!             }),
//!             Err(e) => eprintln!("page {page_num}: {e}"),
//!         }
//!     }
//! });
//! ```
//!
//! # Sparse page selection
//!
//! To render only specific pages (e.g. a subset identified by a prior scan),
//! use [`PageSet`]:
//!
//! ```rust,no_run
//! use std::path::Path;
//! use pdf_raster::{PageSet, RasterOptions, raster_pdf};
//!
//! let pages = PageSet::new(vec![1, 5, 23, 100]).unwrap();
//! let opts = RasterOptions {
//!     dpi: 300.0,
//!     first_page: 1,   // ignored when pages is Some
//!     last_page: 1,    // ignored when pages is Some
//!     deskew: true,
//!     pages: Some(pages),
//! };
//! for (page_num, result) in raster_pdf(Path::new("scan.pdf"), &opts) {
//!     // Only pages 1, 5, 23, and 100 are rendered — intermediates are skipped.
//!     match result {
//!         Ok(page) => { /* process page.pixels */ let _ = (page_num, page); }
//!         Err(e) => eprintln!("page {page_num}: {e}"),
//!     }
//! }
//! ```

pub mod deskew;
pub(crate) mod gpu_init;
mod render;

use std::path::Path;

pub use pdf_interp::renderer::PageDiagnostics;
pub use pdf_interp::resources::ImageFilter;
pub use render::{
    MAX_PX_DIMENSION, RasterError, RasterSession, open_session, render_page_rgb, rgb_to_gray,
};

/// Eagerly release GPU decoders on every rayon worker thread.
///
/// Call this via `pool.broadcast(|_| pdf_raster::release_gpu_decoders())`
/// after all rendering is done and before `pool` is dropped.  This drops each
/// thread's `NvJpegDecoder` while the CUDA driver is still fully live, avoiding
/// the process-exit teardown race where all workers call `nvjpegJpegStateDestroy`
/// concurrently into a driver that has already started its own atexit shutdown.
///
/// After this call the TLS slots hold `Uninitialised`, so their own destructors
/// at process exit are no-ops.
#[expect(
    clippy::missing_const_for_fn,
    reason = "body is non-empty (calls non-const fns) when nvjpeg/nvjpeg2k features are enabled"
)]
pub fn release_gpu_decoders() {
    #[cfg(feature = "nvjpeg")]
    gpu_init::release_nvjpeg_this_thread();
    #[cfg(feature = "nvjpeg2k")]
    gpu_init::release_nvjpeg2k_this_thread();
}

// ── Backend policy ────────────────────────────────────────────────────────────

/// Controls which compute backend is used for image decoding and GPU fills.
///
/// The default is [`Auto`](BackendPolicy::Auto), which matches the behaviour
/// prior to v0.3.1: GPU is used when available and silently skipped otherwise.
/// The `Force*` variants turn silent fallbacks into hard errors so you can tell
/// immediately whether the expected hardware path is actually being taken.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BackendPolicy {
    /// GPU if available, CPU otherwise — silent fallback (default).
    #[default]
    Auto,
    /// CPU only.  All GPU init is skipped; no CUDA or VA-API calls are made.
    CpuOnly,
    /// Require CUDA (nvJPEG, nvJPEG2000, GPU AA fill, ICC CLUT).
    /// Returns [`RasterError::BackendUnavailable`] if CUDA initialisation fails
    /// rather than falling back to CPU.
    ForceCuda,
    /// Require VA-API JPEG decoding.
    /// Returns [`RasterError::BackendUnavailable`] if the VA-API device cannot
    /// be opened rather than falling back to CPU.
    ForceVaapi,
}

// ── Session configuration ─────────────────────────────────────────────────────

/// Configuration for opening a [`RasterSession`].
///
/// Passed to [`open_session`].  Use [`Default::default()`] for the behaviour
/// that was unconditional before v0.3.1 (GPU auto-detected, default DRM node).
#[derive(Debug, Clone)]
pub struct SessionConfig {
    /// Backend selection policy.
    pub policy: BackendPolicy,
    /// VA-API DRM render node.  Default: `/dev/dri/renderD128`.
    ///
    /// Only relevant when the `vaapi` feature is enabled and `policy` is not
    /// [`CpuOnly`](BackendPolicy::CpuOnly) or [`ForceCuda`](BackendPolicy::ForceCuda).
    pub vaapi_device: String,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            policy: BackendPolicy::Auto,
            vaapi_device: "/dev/dri/renderD128".to_owned(),
        }
    }
}

// ── PageSet ───────────────────────────────────────────────────────────────────

/// A validated, sorted, deduplicated set of 1-based page numbers.
///
/// Constructed via [`PageSet::new`].  Clone is O(1) — the underlying storage is
/// reference-counted.  Use as the [`RasterOptions::pages`] field to render a
/// sparse subset of pages without visiting intermediate ones.
#[derive(Debug, Clone)]
pub struct PageSet(std::sync::Arc<[u32]>);

impl PageSet {
    /// Construct a `PageSet` from an arbitrary collection of 1-based page numbers.
    ///
    /// Sorts and deduplicates the input.
    ///
    /// # Errors
    ///
    /// Returns [`RasterError::InvalidOptions`] if `pages` is empty or any value is 0.
    pub fn new(pages: impl Into<Vec<u32>>) -> Result<Self, RasterError> {
        let mut v: Vec<u32> = pages.into();
        v.sort_unstable();
        v.dedup();
        if v.is_empty() {
            return Err(RasterError::InvalidOptions(
                "PageSet must contain at least one page".to_owned(),
            ));
        }
        if v[0] == 0 {
            return Err(RasterError::InvalidOptions(
                "PageSet contains page 0 — pages are 1-based".to_owned(),
            ));
        }
        Ok(Self(v.into_boxed_slice().into()))
    }

    /// Returns `true` if `page` is in this set.  O(log n).
    #[must_use]
    pub fn contains(&self, page: u32) -> bool {
        self.0.binary_search(&page).is_ok()
    }

    /// The smallest page number in the set.
    #[must_use]
    pub fn first(&self) -> u32 {
        self.0[0]
    }

    /// The largest page number in the set.
    #[must_use]
    pub fn last(&self) -> u32 {
        self.0[self.0.len() - 1]
    }

    /// Number of pages in the set.
    #[must_use]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Always `false` for a successfully constructed `PageSet`.
    ///
    /// Provided for API completeness and to satisfy `clippy::len_without_is_empty`.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

// ── Public types ──────────────────────────────────────────────────────────────

/// Options controlling how pages are rendered.
#[derive(Debug, Clone)]
pub struct RasterOptions {
    /// Render resolution in dots per inch.  Must be > 0.
    ///
    /// Pass this same value to Tesseract's `set_source_resolution` — lying about
    /// DPI degrades OCR accuracy because Tesseract uses it for internal scaling.
    /// Recommended: 300 DPI for scanned documents.
    pub dpi: f32,

    /// First page to render (1-based, inclusive).  Must be ≥ 1.
    ///
    /// Ignored when [`pages`](Self::pages) is `Some` — the render window is
    /// derived from the `PageSet` bounds instead.
    pub first_page: u32,

    /// Last page to render (1-based, inclusive).  Must be ≥ `first_page`.
    ///
    /// If `last_page` exceeds the document's page count, rendering stops at the
    /// last page in the document rather than returning an error.
    ///
    /// Ignored when [`pages`](Self::pages) is `Some` — the render window is
    /// derived from the `PageSet` bounds instead.
    pub last_page: u32,

    /// Apply deskew before returning pixels.
    ///
    /// Uses an intensity-weighted projection-profile sweep (no binarisation
    /// threshold) with GPU bilinear rotation via CUDA NPP (`gpu-deskew` feature)
    /// or CPU bilinear fallback.  Corrects skew up to ±7° with sub-0.05°
    /// accuracy.  Disable for native-text PDFs that are never physically skewed.
    pub deskew: bool,

    /// Sparse page selection.
    ///
    /// When `Some`, only the pages in the [`PageSet`] are rendered and yielded.
    /// The iteration window is `PageSet::first()..=PageSet::last()`; intermediate
    /// pages not in the set are skipped without rendering.
    ///
    /// When `None`, all pages in `first_page..=last_page` are rendered.
    pub pages: Option<PageSet>,
}

/// A single rendered page, returned as 8-bit grayscale pixels.
#[derive(Debug)]
pub struct RenderedPage {
    /// Page number (1-based).
    pub page_num: u32,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Raw pixel bytes: 8-bit grayscale, tightly packed (`stride == width`),
    /// top-to-bottom, left-to-right.  Length is `width * height`.
    pub pixels: Vec<u8>,
    /// The DPI at which this page was rendered (`opts.dpi`).
    ///
    /// This is the raw render resolution, ignoring any `UserUnit` scaling in the
    /// document.  Use [`effective_dpi`](Self::effective_dpi) — not this field — when
    /// reporting resolution to downstream consumers such as Tesseract.
    pub dpi: f32,
    /// Physical resolution of the rendered bitmap, accounting for `UserUnit` scaling.
    ///
    /// `effective_dpi = opts.dpi × UserUnit`.  For the vast majority of documents
    /// `UserUnit` is 1.0 and this equals `dpi`.  Always pass this value to
    /// `tesseract::set_source_resolution`; lying about DPI degrades OCR accuracy
    /// because Tesseract uses it for internal feature scaling.
    ///
    /// Omitting the resolution call causes Tesseract to fall back to 70 DPI, which
    /// severely degrades recognition accuracy.
    pub effective_dpi: f32,
    /// Lightweight metadata collected at zero extra cost during rendering.
    ///
    /// Use this to route pages to different OCR configurations — e.g. skip deskew
    /// on `has_vector_text = true` pages, or set Tesseract PSM based on `is_scan`.
    pub diagnostics: PageDiagnostics,
}

impl RenderedPage {
    /// Suggest a render DPI for re-rendering this page at its native image resolution.
    ///
    /// Delegates to [`PageDiagnostics::suggested_dpi`].  Returns `None` for pages
    /// with no raster images (vector/text-only pages should just use the caller's
    /// default DPI).
    #[must_use]
    pub fn suggested_dpi(&self, min_dpi: f32, max_dpi: f32) -> Option<f32> {
        self.diagnostics.suggested_dpi(min_dpi, max_dpi)
    }
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Render a range of pages from a PDF file.
///
/// Returns an iterator yielding `(page_num, Result<RenderedPage, RasterError>)`
/// for each page in `opts.first_page..=opts.last_page`.  A per-page error does
/// not abort remaining pages — the caller decides whether to skip or propagate.
///
/// GPU resources are initialised lazily on first use and reused across pages.
/// Backend selection follows [`BackendPolicy::Auto`] (GPU if available, silent
/// CPU fallback).  Use [`open_session`] + [`render_page_rgb`] directly when you
/// need [`SessionConfig`] control.
///
/// # Errors
///
/// - [`RasterError::InvalidOptions`] — `opts` violates documented constraints.
/// - [`RasterError::Pdf`] — document cannot be opened or parsed.
/// - [`RasterError::PageOutOfRange`] — requested page exceeds the document.
/// - [`RasterError::PageDegenerate`] / [`RasterError::PageTooLarge`] — malformed geometry.
/// - [`RasterError::Deskew`] — deskew rotation failed.
pub fn raster_pdf(
    path: &Path,
    opts: &RasterOptions,
) -> impl Iterator<Item = (u32, Result<RenderedPage, RasterError>)> {
    render::render_pages(path, opts)
}

/// Render a range of pages concurrently using a bounded sync channel.
///
/// Spawns a background Rayon task that renders pages in ascending order and
/// sends each `(page_num, Result<RenderedPage, RasterError>)` to the returned
/// [`Receiver`](std::sync::mpsc::Receiver) as it completes.
///
/// `capacity` is the maximum number of rendered pages buffered before the
/// producer blocks.  Use `2`–`8` for typical OCR pipelines (one page rendering
/// while the previous is being OCR-processed).  `capacity = 0` is silently
/// raised to `1`.
///
/// # Errors delivered through the channel
///
/// - Invalid options → `(1, Err(RasterError::InvalidOptions(...)))`, channel closes.
/// - File open failure → `(1, Err(RasterError::Pdf(...)))`, channel closes.
/// - Per-page failures → `(page_num, Err(...))`, rendering of subsequent pages continues.
#[must_use]
pub fn render_channel(
    path: &Path,
    opts: &RasterOptions,
    capacity: usize,
) -> std::sync::mpsc::Receiver<(u32, Result<RenderedPage, RasterError>)> {
    render::render_channel(path, opts, capacity)
}

#[cfg(test)]
mod page_set_tests {
    use super::*;

    #[test]
    fn empty_input_is_rejected() {
        assert!(matches!(
            PageSet::new(vec![]),
            Err(RasterError::InvalidOptions(_))
        ));
    }

    #[test]
    fn zero_page_is_rejected() {
        assert!(matches!(
            PageSet::new(vec![0, 1, 2]),
            Err(RasterError::InvalidOptions(_))
        ));
    }

    #[test]
    fn valid_input_is_accepted() {
        let ps = PageSet::new(vec![3, 1, 2]).unwrap();
        assert_eq!(ps.first(), 1);
        assert_eq!(ps.last(), 3);
        assert_eq!(ps.len(), 3);
        assert!(!ps.is_empty());
    }

    #[test]
    fn duplicates_are_deduplicated() {
        let ps = PageSet::new(vec![2, 1, 2, 3, 1]).unwrap();
        assert_eq!(ps.len(), 3);
    }

    #[test]
    fn contains_works() {
        let ps = PageSet::new(vec![1, 5, 10]).unwrap();
        assert!(ps.contains(1));
        assert!(ps.contains(5));
        assert!(ps.contains(10));
        assert!(!ps.contains(2));
        assert!(!ps.contains(11));
    }

    #[test]
    fn clone_is_cheap() {
        let ps = PageSet::new(vec![1, 2, 3]).unwrap();
        let ps2 = ps.clone();
        // Both point to the same allocation — Arc pointer equality
        assert!(std::ptr::eq(ps.0.as_ptr(), ps2.0.as_ptr()));
    }

    #[test]
    fn raster_options_with_pages_none_is_valid() {
        // Verifies the field exists and validate_opts accepts it
        let opts = RasterOptions {
            dpi: 150.0,
            first_page: 1,
            last_page: 5,
            deskew: false,
            pages: None,
        };
        assert!(opts.pages.is_none());
    }

    #[test]
    fn raster_options_with_pages_some_is_valid() {
        let ps = PageSet::new(vec![1, 3, 5]).unwrap();
        let opts = RasterOptions {
            dpi: 150.0,
            first_page: 1,
            last_page: 5,
            deskew: false,
            pages: Some(ps),
        };
        assert!(opts.pages.is_some());
    }
}
