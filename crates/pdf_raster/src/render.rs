//! Core render pipeline: PDF page → pixel buffer.

#[cfg(any(feature = "nvjpeg", feature = "nvjpeg2k"))]
use std::cell::RefCell;
#[cfg(any(feature = "gpu-aa", feature = "gpu-icc"))]
use std::sync::Arc;

use color::{Gray8, Rgb8};
use raster::Bitmap;

use crate::{RasterOptions, RenderedPage};

// ── Safety limit ──────────────────────────────────────────────────────────────

/// Maximum pixel dimension (width or height) accepted from a PDF page.
///
/// Prevents absurdly large allocations from malformed or adversarial documents.
/// 32 768 px at 150 DPI corresponds to roughly 366 inches (~9.3 metres).
pub const MAX_PX_DIMENSION: u32 = 32_768;

// ── Per-thread GPU image decoders ─────────────────────────────────────────────
//
// Send but not Sync: each rayon worker thread owns exactly one instance, created
// lazily on first use.  DecoderInit<T> prevents retry-and-spam after a one-time
// init failure.  RefCell gives interior mutability; thread_local prevents races.

#[cfg(any(feature = "nvjpeg", feature = "nvjpeg2k"))]
#[derive(Default)]
enum DecoderInit<T> {
    #[default]
    Uninitialised,
    Ready(Option<T>),
    Failed,
}

#[cfg(feature = "nvjpeg")]
thread_local! {
    static NVJPEG_DEC: RefCell<DecoderInit<gpu::nvjpeg::NvJpegDecoder>> =
        const { RefCell::new(DecoderInit::Uninitialised) };
}

#[cfg(feature = "nvjpeg2k")]
thread_local! {
    static NVJPEG2K_DEC: RefCell<DecoderInit<gpu::nvjpeg2k::NvJpeg2kDecoder>> =
        const { RefCell::new(DecoderInit::Uninitialised) };
}

#[cfg(feature = "nvjpeg")]
fn init_nvjpeg() {
    NVJPEG_DEC.with(|cell| {
        let mut slot = cell.borrow_mut();
        if matches!(*slot, DecoderInit::Uninitialised) {
            match gpu::nvjpeg::NvJpegDecoder::new(0) {
                Ok(dec) => *slot = DecoderInit::Ready(Some(dec)),
                Err(e) => {
                    eprintln!(
                        "pdf_raster: nvJPEG unavailable ({e}); \
                         JPEG images will be decoded on CPU for this thread"
                    );
                    *slot = DecoderInit::Failed;
                }
            }
        }
    });
}

#[cfg(feature = "nvjpeg2k")]
fn init_nvjpeg2k() {
    NVJPEG2K_DEC.with(|cell| {
        let mut slot = cell.borrow_mut();
        if matches!(*slot, DecoderInit::Uninitialised) {
            match gpu::nvjpeg2k::NvJpeg2kDecoder::new(0) {
                Ok(dec) => *slot = DecoderInit::Ready(Some(dec)),
                Err(e) => {
                    eprintln!(
                        "pdf_raster: nvJPEG2000 unavailable ({e}); \
                         JPEG 2000 images will be decoded on CPU for this thread"
                    );
                    *slot = DecoderInit::Failed;
                }
            }
        }
    });
}

// ── Error type ────────────────────────────────────────────────────────────────

/// Errors returned by [`crate::raster_pdf`].
#[derive(Debug)]
pub enum RasterError {
    /// [`RasterOptions`](crate::RasterOptions) fields violate documented constraints
    /// (e.g. `dpi ≤ 0`, `first_page > last_page`).
    InvalidOptions(String),
    /// The PDF could not be opened or parsed.
    Pdf(pdf_interp::InterpError),
    /// The requested page number is outside the document.
    PageOutOfRange {
        /// The requested page (1-based).
        page: u32,
        /// Total number of pages in the document.
        total: u32,
    },
    /// The page has zero pixel width or height — malformed document.
    PageDegenerate {
        /// Width in pixels (0 when degenerate).
        width: u32,
        /// Height in pixels (0 when degenerate).
        height: u32,
    },
    /// The computed pixel dimensions exceed the safety limit.
    PageTooLarge {
        /// Width in pixels.
        width: u32,
        /// Height in pixels.
        height: u32,
    },
    /// Deskew rotation failed.
    Deskew(String),
}

impl std::fmt::Display for RasterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidOptions(msg) => write!(f, "invalid raster options: {msg}"),
            Self::Pdf(e) => write!(f, "PDF error: {e}"),
            Self::PageOutOfRange { page, total } => {
                write!(
                    f,
                    "page {page} is out of range (document has {total} pages)"
                )
            }
            Self::PageDegenerate { width, height } => write!(
                f,
                "page has degenerate pixel dimensions {width}×{height} — \
                 PDF MediaBox may be malformed"
            ),
            Self::PageTooLarge { width, height } => write!(
                f,
                "page pixel dimensions {width}×{height} exceed safety limit \
                 ({MAX_PX_DIMENSION}); lower the DPI or check the document"
            ),
            Self::Deskew(msg) => write!(f, "deskew failed: {msg}"),
        }
    }
}

impl std::error::Error for RasterError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Pdf(e) => Some(e),
            _ => None,
        }
    }
}

impl From<pdf_interp::InterpError> for RasterError {
    fn from(e: pdf_interp::InterpError) -> Self {
        match e {
            pdf_interp::InterpError::PageOutOfRange { page, total } => {
                Self::PageOutOfRange { page, total }
            }
            other => Self::Pdf(other),
        }
    }
}

// ── RasterSession ─────────────────────────────────────────────────────────────

/// An opened PDF document ready for per-page rendering.
///
/// Constructed via [`open_session`].  Provides both a sequential iterator
/// ([`raster_pdf`](crate::raster_pdf)) and a direct per-page call
/// ([`render_page_rgb`]) for parallel consumers such as the CLI.
///
/// `Sync` because the document is read-only after construction and GPU context
/// is wrapped in `Arc`.  Per-thread GPU image decoders are managed via
/// `thread_local!` inside `render_page_rgb`.
pub struct RasterSession {
    pub(crate) doc: lopdf::Document,
    /// Page-number → object-ID map, built once at construction.
    pub(crate) pages: std::collections::BTreeMap<u32, lopdf::ObjectId>,
    pub(crate) total_pages: u32,
    #[cfg(any(feature = "gpu-aa", feature = "gpu-icc"))]
    pub(crate) gpu_ctx: Option<Arc<gpu::GpuCtx>>,
}

impl RasterSession {
    /// Total number of pages in the document.
    pub fn total_pages(&self) -> u32 {
        self.total_pages
    }
}

/// Open a PDF and create a [`RasterSession`] for rendering.
///
/// Eagerly builds the page-ID map so repeated per-page calls are O(1) lookups
/// rather than O(n) rebuilds.
///
/// # Errors
///
/// Returns [`RasterError::Pdf`] if the file cannot be opened or parsed.
pub fn open_session(path: &std::path::Path) -> Result<RasterSession, RasterError> {
    let doc = pdf_interp::open(path).map_err(RasterError::from)?;
    let total_pages = pdf_interp::page_count(&doc);
    let pages = doc.get_pages();

    #[cfg(any(feature = "gpu-aa", feature = "gpu-icc"))]
    let gpu_ctx = gpu::GpuCtx::new().ok().map(Arc::new);

    Ok(RasterSession {
        doc,
        pages,
        total_pages,
        #[cfg(any(feature = "gpu-aa", feature = "gpu-icc"))]
        gpu_ctx,
    })
}

/// Render one page to an RGB bitmap.
///
/// The caller is responsible for any colour conversion (e.g. `rgb_to_gray`)
/// and file output.  GPU image decoders are initialised lazily per calling
/// thread and reused across pages — safe to call from multiple rayon threads
/// concurrently.
///
/// `scale` is `dpi / 72.0` (both axes equal for square-pixel rendering).
///
/// # Errors
///
/// Returns [`RasterError`] if the page geometry is invalid or the PDF cannot
/// be interpreted.
pub fn render_page_rgb(
    session: &RasterSession,
    page_num: u32,
    scale: f64,
) -> Result<Bitmap<Rgb8>, RasterError> {
    let doc = &session.doc;

    let geom = pdf_interp::page_size_pts(doc, page_num)?;

    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "page dimensions × scale are always positive and ≪ u32::MAX"
    )]
    let (w_px, h_px) = (
        (geom.width_pts * scale).round() as u32,
        (geom.height_pts * scale).round() as u32,
    );

    if w_px == 0 || h_px == 0 {
        return Err(RasterError::PageDegenerate {
            width: w_px,
            height: h_px,
        });
    }
    if w_px > MAX_PX_DIMENSION || h_px > MAX_PX_DIMENSION {
        return Err(RasterError::PageTooLarge {
            width: w_px,
            height: h_px,
        });
    }

    let page_id = *session
        .pages
        .get(&page_num)
        .ok_or(RasterError::PageOutOfRange {
            page: page_num,
            total: session.total_pages,
        })?;

    let ops = pdf_interp::parse_page(doc, page_num)?;

    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "scale = dpi/72 is always positive; result ≪ f32::MAX"
    )]
    let scale_f32 = scale as f32;

    let mut renderer = pdf_interp::renderer::PageRenderer::new_scaled(
        w_px,
        h_px,
        scale_f32.into(),
        geom.rotate_cw,
        doc,
        page_id,
    );

    #[cfg(any(feature = "gpu-aa", feature = "gpu-icc"))]
    renderer.set_gpu_ctx(session.gpu_ctx.as_ref().map(Arc::clone));

    #[cfg(feature = "nvjpeg")]
    {
        init_nvjpeg();
        NVJPEG_DEC.with(|cell| {
            if let DecoderInit::Ready(slot) = &mut *cell.borrow_mut() {
                renderer.set_nvjpeg(slot.take());
            }
        });
    }
    #[cfg(feature = "nvjpeg2k")]
    {
        init_nvjpeg2k();
        NVJPEG2K_DEC.with(|cell| {
            if let DecoderInit::Ready(slot) = &mut *cell.borrow_mut() {
                renderer.set_nvjpeg2k(slot.take());
            }
        });
    }

    renderer.execute(&ops);
    renderer.render_annotations(page_id);

    #[cfg(feature = "nvjpeg")]
    NVJPEG_DEC.with(|cell| {
        if let DecoderInit::Ready(slot) = &mut *cell.borrow_mut() {
            *slot = renderer.take_nvjpeg();
        }
    });
    #[cfg(feature = "nvjpeg2k")]
    NVJPEG2K_DEC.with(|cell| {
        if let DecoderInit::Ready(slot) = &mut *cell.borrow_mut() {
            *slot = renderer.take_nvjpeg2k();
        }
    });

    Ok(renderer.finish())
}

// ── Sequential iterator ───────────────────────────────────────────────────────

/// State shared across all pages rendered in one `raster_pdf` call.
struct RenderState {
    session: RasterSession,
    opts: RasterOptions,
    current_page: u32,
}

pub(crate) fn render_pages(
    path: &std::path::Path,
    opts: &RasterOptions,
) -> impl Iterator<Item = (u32, Result<RenderedPage, RasterError>)> {
    // Validate options early so the caller gets a clear error rather than silent
    // empty output or a confusing downstream panic.
    if opts.dpi <= 0.0 || !opts.dpi.is_finite() {
        return PageIter {
            state: Some(Err(RasterError::InvalidOptions(format!(
                "dpi must be a positive finite number, got {}",
                opts.dpi
            )))),
        };
    }
    if opts.first_page == 0 {
        return PageIter {
            state: Some(Err(RasterError::InvalidOptions(
                "first_page must be ≥ 1 (pages are 1-based)".to_owned(),
            ))),
        };
    }
    if opts.first_page > opts.last_page {
        return PageIter {
            state: Some(Err(RasterError::InvalidOptions(format!(
                "first_page ({}) > last_page ({})",
                opts.first_page, opts.last_page
            )))),
        };
    }

    let state = open_session(path).map(|session| RenderState {
        current_page: opts.first_page,
        session,
        opts: opts.clone(),
    });

    PageIter {
        state: Some(state),
    }
}

struct PageIter {
    // `None` once the iterator is exhausted or a document-open error has been yielded.
    state: Option<Result<RenderState, RasterError>>,
}

impl Iterator for PageIter {
    type Item = (u32, Result<RenderedPage, RasterError>);

    fn next(&mut self) -> Option<Self::Item> {
        match self.state.as_mut()? {
            Err(_) => {
                // Document-open error: take it out, yield once, then stop (state → None).
                let err = self
                    .state
                    .take()
                    .expect("state was Some(_) one line above")
                    .map(|_| unreachable!());
                // first_page is not available when the document failed to open; use 1
                // as a placeholder — the caller only cares about the Err payload.
                Some((1, err))
            }
            Ok(state) => {
                if state.current_page > state.opts.last_page
                    || state.current_page > state.session.total_pages
                {
                    self.state = None;
                    return None;
                }

                let page_num = state.current_page;
                state.current_page += 1;

                let result = render_one(state, page_num);
                Some((page_num, result))
            }
        }
    }
}

// ── Single-page render (gray + deskew) ───────────────────────────────────────

fn render_one(state: &RenderState, page_num: u32) -> Result<RenderedPage, RasterError> {
    let dpi = state.opts.dpi;
    let scale = f64::from(dpi) / 72.0;

    let rgb = render_page_rgb(&state.session, page_num, scale)?;
    let mut gray = rgb_to_gray(&rgb);

    if state.opts.deskew {
        crate::deskew::apply(&mut gray).map_err(|e| RasterError::Deskew(e.to_string()))?;
    }

    let pixels = bitmap_to_vec(&gray);

    Ok(RenderedPage {
        page_num,
        width: gray.width,
        height: gray.height,
        pixels,
        dpi,
    })
}

// ── Pixel helpers ─────────────────────────────────────────────────────────────

/// Convert an RGB bitmap to grayscale using BT.709 luminance coefficients.
///
/// Output is 0 = black, 255 = white, matching the input convention.
pub fn rgb_to_gray(src: &Bitmap<Rgb8>) -> Bitmap<Gray8> {
    let mut dst = Bitmap::<Gray8>::new(src.width, src.height, 1, false);
    let w = src.width as usize;
    for y in 0..src.height {
        let src_row = &src.row_bytes(y)[..w * 3];
        let dst_row = &mut dst.row_bytes_mut(y)[..w];
        for (dst_px, rgb) in dst_row.iter_mut().zip(src_row.chunks_exact(3)) {
            // BT.709: Y = 0.2126·R + 0.7152·G + 0.0722·B
            // Integer: (2126·R + 7152·G + 722·B + 5000) / 10000
            // Max numerator = 10000·255 + 5000 = 2 555 000 < u32::MAX.
            let (r, g, b) = (u32::from(rgb[0]), u32::from(rgb[1]), u32::from(rgb[2]));
            #[expect(
                clippy::cast_possible_truncation,
                reason = "sum ≤ 255 by BT.709 coefficient identity"
            )]
            {
                *dst_px = ((2126 * r + 7152 * g + 722 * b + 5000) / 10000) as u8;
            }
        }
    }
    dst
}

/// Flatten a `Bitmap<Gray8>` into a tightly-packed `Vec<u8>`.
///
/// The bitmap may have internal row padding (stride > width); this strips it
/// so the returned buffer has exactly `width * height` bytes.
fn bitmap_to_vec(bmp: &Bitmap<Gray8>) -> Vec<u8> {
    let w = bmp.width as usize;
    let h = bmp.height as usize;
    let mut out = Vec::with_capacity(w * h);
    for y in 0..bmp.height {
        out.extend_from_slice(&bmp.row_bytes(y)[..w]);
    }
    out
}
