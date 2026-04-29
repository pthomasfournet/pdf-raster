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
//! let opts = RasterOptions { dpi: 300.0, first_page: 1, last_page: 5, deskew: true };
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
//! # let opts = RasterOptions { dpi: 300.0, first_page: 1, last_page: 1, deskew: true };
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

pub mod deskew;
mod render;

use std::path::Path;

pub use pdf_interp::renderer::PageDiagnostics;
pub use pdf_interp::resources::ImageFilter;
pub use render::{
    MAX_PX_DIMENSION, RasterError, RasterSession, open_session, render_page_rgb, rgb_to_gray,
};

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
    pub first_page: u32,

    /// Last page to render (1-based, inclusive).  Must be ≥ `first_page`.
    ///
    /// If `last_page` exceeds the document's page count, rendering stops at the
    /// last page in the document rather than returning an error.
    pub last_page: u32,

    /// Apply deskew before returning pixels.
    ///
    /// Uses an intensity-weighted projection-profile sweep (no binarisation
    /// threshold) with GPU bilinear rotation via CUDA NPP (`gpu-deskew` feature)
    /// or CPU bilinear fallback.  Corrects skew up to ±7° with sub-0.05°
    /// accuracy.  Disable for native-text PDFs that are never physically skewed.
    pub deskew: bool,
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
    ///
    /// Typical usage in an OCR pipeline:
    /// ```rust,ignore
    /// let opts = RasterOptions { dpi: 300.0, ..Default::default() };
    /// let page = render_one_page(path, page_num, &opts)?;
    /// if let Some(native_dpi) = page.suggested_dpi(150.0, 300.0) {
    ///     if (native_dpi - opts.dpi).abs() > 10.0 {
    ///         // Re-render at native resolution to avoid upsampling.
    ///     }
    /// }
    /// ```
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
/// Pages are rendered in ascending page-number order.  GPU resources are
/// initialised lazily on first use and reused across pages.
///
/// # Errors
///
/// - [`RasterError::InvalidOptions`] if `opts` violates documented constraints
///   (e.g. `dpi ≤ 0`, `first_page > last_page`).
/// - [`RasterError::Pdf`] if the document cannot be opened or parsed.
/// - [`RasterError::PageOutOfRange`] if a requested page exceeds the document.
/// - [`RasterError::PageDegenerate`] / [`RasterError::PageTooLarge`] for
///   malformed page geometry.
/// - [`RasterError::Deskew`] if deskew rotation fails (rare; falls back
///   gracefully when possible).
///
/// # Panics
///
/// Panics only on GPU driver bugs where a CUDA or nvJPEG function reports
/// success but returns a null handle — this cannot be triggered by valid input
/// or malformed PDFs, only by a faulty driver.
pub fn raster_pdf(
    path: &Path,
    opts: &RasterOptions,
) -> impl Iterator<Item = (u32, Result<RenderedPage, RasterError>)> {
    render::render_pages(path, opts)
}
