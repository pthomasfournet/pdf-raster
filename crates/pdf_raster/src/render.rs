//! Core render pipeline: PDF page → pixel buffer.

#[cfg(any(feature = "nvjpeg", feature = "nvjpeg2k", feature = "vaapi"))]
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

#[cfg(any(feature = "nvjpeg", feature = "nvjpeg2k", feature = "vaapi"))]
#[derive(Default)]
enum DecoderInit<T> {
    #[default]
    Uninitialised,
    Ready(Option<T>),
    Failed,
}

// nvjpegCreateEx is not safe to call concurrently from multiple threads —
// simultaneous calls race inside the library and can null-deref.  This mutex
// serialises all decoder construction calls across threads.
// VA-API init (vaInitialize) is also serialised here out of caution.
#[cfg(any(feature = "nvjpeg", feature = "nvjpeg2k", feature = "vaapi"))]
static DECODER_INIT_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

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

#[cfg(feature = "vaapi")]
thread_local! {
    static VAAPI_JPEG_DEC: RefCell<DecoderInit<gpu::vaapi::VapiJpegDecoder>> =
        const { RefCell::new(DecoderInit::Uninitialised) };
}

#[cfg(feature = "nvjpeg")]
fn init_nvjpeg() {
    NVJPEG_DEC.with(|cell| {
        if !matches!(*cell.borrow(), DecoderInit::Uninitialised) {
            return;
        }
        // Serialise construction: nvjpegCreateEx races if called concurrently
        // from multiple threads and can null-deref inside the library.
        let guard = DECODER_INIT_LOCK.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if !matches!(*cell.borrow(), DecoderInit::Uninitialised) {
            return;
        }
        let result = gpu::nvjpeg::NvJpegDecoder::new(0);
        drop(guard);
        match result {
            Ok(dec) => *cell.borrow_mut() = DecoderInit::Ready(Some(dec)),
            Err(e) => {
                eprintln!(
                    "pdf_raster: nvJPEG unavailable ({e}); \
                     JPEG images will be decoded on CPU for this thread"
                );
                *cell.borrow_mut() = DecoderInit::Failed;
            }
        }
    });
}

/// Drop this thread's nvJPEG decoder immediately.
///
/// Called via `rayon::broadcast` before the pool is dropped, so TLS
/// destructors at process exit see `Uninitialised` and become no-ops.
/// This avoids the process-exit teardown race where all worker threads call
/// `nvjpegJpegStateDestroy` concurrently into an already-shutting-down driver.
#[cfg(feature = "nvjpeg")]
pub(crate) fn release_nvjpeg_this_thread() {
    NVJPEG_DEC.with(|cell| {
        *cell.borrow_mut() = DecoderInit::Uninitialised;
    });
}

#[cfg(feature = "nvjpeg2k")]
fn init_nvjpeg2k() {
    NVJPEG2K_DEC.with(|cell| {
        if !matches!(*cell.borrow(), DecoderInit::Uninitialised) {
            return;
        }
        let guard = DECODER_INIT_LOCK.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if !matches!(*cell.borrow(), DecoderInit::Uninitialised) {
            return;
        }
        let result = gpu::nvjpeg2k::NvJpeg2kDecoder::new(0);
        drop(guard);
        match result {
            Ok(dec) => *cell.borrow_mut() = DecoderInit::Ready(Some(dec)),
            Err(e) => {
                eprintln!(
                    "pdf_raster: nvJPEG2000 unavailable ({e}); \
                     JPEG 2000 images will be decoded on CPU for this thread"
                );
                *cell.borrow_mut() = DecoderInit::Failed;
            }
        }
    });
}

/// Drop this thread's nvJPEG2000 decoder immediately (same rationale as above).
#[cfg(feature = "nvjpeg2k")]
pub(crate) fn release_nvjpeg2k_this_thread() {
    NVJPEG2K_DEC.with(|cell| {
        *cell.borrow_mut() = DecoderInit::Uninitialised;
    });
}

/// Default DRM render node used when the `vaapi` feature is active.
///
/// Override by rebuilding with a different path if your setup differs.
#[cfg(feature = "vaapi")]
const VAAPI_DRM_NODE: &str = "/dev/dri/renderD128";

#[cfg(feature = "vaapi")]
fn init_vaapi_jpeg() {
    VAAPI_JPEG_DEC.with(|cell| {
        if !matches!(*cell.borrow(), DecoderInit::Uninitialised) {
            return;
        }
        let guard = DECODER_INIT_LOCK.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if !matches!(*cell.borrow(), DecoderInit::Uninitialised) {
            return;
        }
        let result = gpu::vaapi::VapiJpegDecoder::new(VAAPI_DRM_NODE);
        drop(guard);
        match result {
            Ok(dec) => *cell.borrow_mut() = DecoderInit::Ready(Some(dec)),
            Err(e) => {
                log::info!(
                    "pdf_raster: VA-API JPEG unavailable ({e}); \
                     JPEG images will be decoded on CPU for this thread"
                );
                *cell.borrow_mut() = DecoderInit::Failed;
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
    /// A Page dictionary entry is structurally valid but outside permitted range
    /// (e.g. `UserUnit` outside `[0.1, 10.0]`).
    InvalidPageGeometry(String),
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
            Self::InvalidPageGeometry(msg) => write!(f, "invalid page geometry: {msg}"),
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
            pdf_interp::InterpError::InvalidPageGeometry(msg) => Self::InvalidPageGeometry(msg),
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
    #[must_use]
    pub const fn total_pages(&self) -> u32 {
        self.total_pages
    }
}

// Compile-time assertions: RasterSession must be Sync (shared across rayon threads) and
// Send (moved into the rayon::spawn closure in render_channel).
// If lopdf::Document ever becomes !Sync / !Send this will fail here rather than at a
// confusing downstream call site.
const _: fn() = || {
    const fn assert_sync<T: Sync>() {}
    const fn assert_send<T: Send>() {}
    assert_sync::<RasterSession>();
    assert_send::<RasterSession>();
};

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
    // Build the page map once; derive total_pages from it to avoid a second get_pages() call.
    let pages = doc.get_pages();
    let total_pages = u32::try_from(pages.len()).unwrap_or(u32::MAX);

    #[cfg(any(feature = "gpu-aa", feature = "gpu-icc"))]
    let gpu_ctx = match gpu::GpuCtx::init() {
        Ok(ctx) => Some(Arc::new(ctx)),
        Err(e) => {
            eprintln!(
                "pdf_raster: GPU initialisation failed ({e}); \
                 falling back to CPU. Run `nvidia-smi` to verify the driver is loaded."
            );
            None
        }
    };

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
/// `scale` is the pixel-per-point multiplier: `dpi / 72.0` for square-pixel
/// rendering, or `(x_dpi/72 · y_dpi/72).sqrt()` for the geometric-mean when
/// horizontal and vertical DPI differ.  Must be a positive finite number.
///
/// # Errors
///
/// - [`RasterError::InvalidOptions`] if `scale` is ≤ 0 or non-finite.
/// - [`RasterError::InvalidPageGeometry`] if the page has a malformed `UserUnit`.
/// - [`RasterError::PageDegenerate`] if the page `MediaBox` has zero area.
/// - [`RasterError::PageTooLarge`] if the computed pixel dimensions exceed
///   [`MAX_PX_DIMENSION`].
/// - [`RasterError::PageOutOfRange`] if `page_num` is not in the document.
/// - [`RasterError::Pdf`] if the page content stream cannot be parsed.
pub fn render_page_rgb(
    session: &RasterSession,
    page_num: u32,
    scale: f64,
) -> Result<Bitmap<Rgb8>, RasterError> {
    let geom = pdf_interp::page_size_pts(&session.doc, page_num)?;
    render_page_rgb_with_geom(session, page_num, scale, geom).map(|(bmp, _diag)| bmp)
}

/// Inner implementation shared by [`render_page_rgb`] and [`render_one`].
/// Accepts a pre-resolved [`PageGeometry`] to avoid a second dict lookup when
/// the caller already has it (e.g. to read `user_unit` for `effective_dpi`).
/// Returns the bitmap and the [`PageDiagnostics`] collected during rendering.
fn render_page_rgb_with_geom(
    session: &RasterSession,
    page_num: u32,
    scale: f64,
    geom: pdf_interp::PageGeometry,
) -> Result<(Bitmap<Rgb8>, pdf_interp::renderer::PageDiagnostics), RasterError> {
    // A non-positive or non-finite scale is a caller error; return InvalidOptions
    // so callers can distinguish this from a genuinely malformed page MediaBox.
    if !scale.is_finite() || scale <= 0.0 {
        return Err(RasterError::InvalidOptions(format!(
            "scale must be a positive finite number, got {scale}"
        )));
    }

    let doc = &session.doc;

    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "scale and dimensions are positive; f64-to-u32 saturates at u32::MAX for \
                  adversarial values, which the MAX_PX_DIMENSION check below catches"
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
        reason = "scale = dpi/72 is always positive and small; f64→f32 precision loss is \
                  negligible for sub-pixel rounding at any practical DPI"
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

    #[cfg(feature = "vaapi")]
    {
        init_vaapi_jpeg();
        VAAPI_JPEG_DEC.with(|cell| {
            if let DecoderInit::Ready(slot) = &mut *cell.borrow_mut() {
                renderer.set_vaapi_jpeg(slot.take());
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

    #[cfg(feature = "vaapi")]
    VAAPI_JPEG_DEC.with(|cell| {
        if let DecoderInit::Ready(slot) = &mut *cell.borrow_mut() {
            *slot = renderer.take_vaapi_jpeg();
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

/// Validate [`RasterOptions`], returning the first violated constraint as an error.
///
/// Returns `None` when all constraints are satisfied.  Used by both
/// [`render_pages`] and [`render_channel`] to share identical validation logic.
fn validate_opts(opts: &RasterOptions) -> Option<RasterError> {
    if opts.dpi <= 0.0 || !opts.dpi.is_finite() {
        return Some(RasterError::InvalidOptions(format!(
            "dpi must be a positive finite number, got {}",
            opts.dpi
        )));
    }
    if opts.first_page == 0 {
        return Some(RasterError::InvalidOptions(
            "first_page must be ≥ 1 (pages are 1-based)".to_owned(),
        ));
    }
    if opts.first_page > opts.last_page {
        return Some(RasterError::InvalidOptions(format!(
            "first_page ({}) > last_page ({})",
            opts.first_page, opts.last_page
        )));
    }
    None
}

pub fn render_pages(
    path: &std::path::Path,
    opts: &RasterOptions,
) -> impl Iterator<Item = (u32, Result<RenderedPage, RasterError>)> {
    if let Some(e) = validate_opts(opts) {
        return PageIter {
            state: Some(Err(e)),
        };
    }

    let state = open_session(path).map(|session| RenderState {
        current_page: opts.first_page,
        session,
        opts: opts.clone(),
    });

    PageIter { state: Some(state) }
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
                // Document-open error: take it out (sets state → None), yield once.
                // Use page 1 as a placeholder; the caller cares about the Err, not the number.
                let Err(e) = self.state.take()? else {
                    unreachable!("matched Err arm above")
                };
                Some((1, Err(e)))
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

// ── Channel-based render (bounded, backpressure, Rayon-spawned producer) ─────

/// Render pages in the background and stream them through a bounded sync channel.
///
/// Spawns one Rayon task that renders pages in ascending order and sends each
/// `(page_num, Result<RenderedPage, RasterError>)` to the returned
/// [`Receiver`](std::sync::mpsc::Receiver) as it completes.  The channel is
/// bounded to `capacity` items (minimum 1): if the consumer falls behind, the
/// producer blocks on [`SyncSender::send`](std::sync::mpsc::SyncSender::send),
/// providing natural backpressure.
///
/// # Error contract
///
/// - Options validation failure: sent as `(1, Err(e))` synchronously before
///   the Rayon task is spawned; channel closes immediately.
/// - Session-open failure: sent as `(1, Err(e))` from the producer; channel
///   closes immediately.
/// - Per-page render errors: sent as `(page_num, Err(e))`; subsequent pages
///   continue rendering (same contract as [`render_pages`]).
///
/// # Disconnect
///
/// If the [`Receiver`](std::sync::mpsc::Receiver) is dropped before the
/// producer finishes, the producer exits cleanly on its next `send`.
#[must_use]
pub fn render_channel(
    path: &std::path::Path,
    opts: &RasterOptions,
    capacity: usize,
) -> std::sync::mpsc::Receiver<(u32, Result<RenderedPage, RasterError>)> {
    use std::sync::mpsc;

    // sync_channel(0) is a rendezvous channel: the first send blocks until someone
    // calls recv().  Sending an error in the fast-path below (before the caller can
    // call recv()) would deadlock.  Clamp to 1 to guarantee at least one buffered slot.
    let capacity = capacity.max(1);
    let (tx, rx) = mpsc::sync_channel(capacity);

    // Validate synchronously on the calling thread — the error is available
    // immediately without waiting for Rayon scheduling.
    if let Some(e) = validate_opts(opts) {
        // Receiver has been returned to the caller but not yet polled.  The send
        // will always succeed (capacity ≥ 1).  We discard the Result because the
        // channel closing on `return` is itself the end-of-stream signal.
        let _sent = tx.send((1, Err(e)));
        return rx;
    }

    // rayon::spawn requires 'static closures, so both path and opts must be owned.
    let path_owned: std::path::PathBuf = path.to_owned();
    let opts_owned: RasterOptions = opts.clone();

    rayon::spawn(move || {
        let session = match open_session(&path_owned) {
            Ok(s) => s,
            Err(e) => {
                // Receiver may already be dropped (caller called drop(rx) before
                // the Rayon task started).  Discard SendError — channel closing is
                // the end-of-stream signal either way.
                let _sent = tx.send((1, Err(e)));
                return;
            }
        };

        let state = RenderState {
            current_page: opts_owned.first_page,
            session,
            opts: opts_owned,
        };

        // Clamp last_page to actual document length (mirrors the iterator).
        let last = state.opts.last_page.min(state.session.total_pages);

        for page_num in state.opts.first_page..=last {
            let result = render_one(&state, page_num);
            // SyncSender::send blocks when the channel is full (backpressure).
            // Returns Err when the Receiver has been dropped — exit cleanly.
            if tx.send((page_num, result)).is_err() {
                return;
            }
        }
        // tx drops here → channel closes → Receiver::recv returns Err(RecvError).
    });

    rx
}

// ── Single-page render (gray + deskew) ───────────────────────────────────────

fn render_one(state: &RenderState, page_num: u32) -> Result<RenderedPage, RasterError> {
    let dpi = state.opts.dpi;
    let scale = f64::from(dpi) / 72.0;

    // Resolve geometry first to obtain UserUnit before rendering.
    let geom = pdf_interp::page_size_pts(&state.session.doc, page_num)?;

    let (rgb, diagnostics) = render_page_rgb_with_geom(&state.session, page_num, scale, geom)?;
    let mut gray = rgb_to_gray(&rgb);

    if state.opts.deskew {
        crate::deskew::apply(&mut gray).map_err(|e| RasterError::Deskew(e.to_string()))?;
    }

    let pixels = bitmap_to_vec(&gray);

    // effective_dpi accounts for UserUnit: a UserUnit:2.0 page has user-space units
    // twice as large (each unit = 2/72 inch), so the rendered pixels cover twice the
    // physical area — effective DPI is doubled.  For the common case (UserUnit = 1.0)
    // effective_dpi == dpi.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "dpi is an f32 (≤ ~3400 in practice); user_unit is validated to [0.1, 10.0]; \
                  the product is at most ~34 000, well within f32 range"
    )]
    let effective_dpi = (f64::from(dpi) * geom.user_unit) as f32;

    Ok(RenderedPage {
        page_num,
        width: gray.width,
        height: gray.height,
        pixels,
        dpi,
        effective_dpi,
        diagnostics,
    })
}

// ── Pixel helpers ─────────────────────────────────────────────────────────────

/// Convert an RGB bitmap to grayscale using BT.709 luminance coefficients.
///
/// Output is 0 = black, 255 = white, matching the input convention.
#[must_use]
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
    let mut out = Vec::with_capacity(w * bmp.height as usize);
    for y in 0..bmp.height {
        out.extend_from_slice(&bmp.row_bytes(y)[..w]);
    }
    out
}

#[cfg(test)]
mod channel_tests {
    use std::path::Path;

    use super::*;

    fn valid_opts() -> RasterOptions {
        RasterOptions {
            dpi: 150.0,
            first_page: 1,
            last_page: 1,
            deskew: false,
        }
    }

    #[test]
    fn validation_error_delivered_and_channel_closes() {
        let bad = RasterOptions {
            dpi: 0.0,
            ..valid_opts()
        };
        let rx = render_channel(Path::new("/irrelevant"), &bad, 4);
        let (page, res) = rx.recv().expect("first item must arrive");
        assert_eq!(page, 1);
        assert!(
            matches!(res, Err(RasterError::InvalidOptions(_))),
            "expected InvalidOptions, got {res:?}"
        );
        assert!(
            rx.recv().is_err(),
            "channel must be closed after validation error"
        );
    }

    #[test]
    fn session_open_failure_delivered_and_channel_closes() {
        let rx = render_channel(Path::new("/no_such_file_xyz.pdf"), &valid_opts(), 4);
        let (page, res) = rx.recv().expect("first item must arrive");
        assert_eq!(page, 1);
        assert!(res.is_err(), "expected Err from session open, got Ok");
        assert!(
            rx.recv().is_err(),
            "channel must be closed after session error"
        );
    }

    #[test]
    fn receiver_drop_does_not_panic() {
        // Dropping the Receiver before the producer finishes must not panic.
        // The producer's send loop returns on SendError rather than unwrapping,
        // so the Rayon task exits cleanly.  No sleep: a Rayon thread panic would
        // propagate as a test failure on the next rayon barrier, not here — the
        // absence of an immediate panic plus the structural guarantee (is_err check
        // in the loop) is sufficient.
        let rx = render_channel(Path::new("/no_such_file_xyz.pdf"), &valid_opts(), 1);
        drop(rx);
    }

    #[test]
    fn capacity_zero_raised_to_one_no_deadlock() {
        // capacity=0 would be a rendezvous; the error fast-path must not deadlock.
        let bad = RasterOptions {
            dpi: 0.0,
            ..valid_opts()
        };
        let rx = render_channel(Path::new("/irrelevant"), &bad, 0);
        // Must receive the error item without blocking forever.
        assert!(
            rx.recv().is_ok(),
            "error item must be delivered even with capacity=0"
        );
        assert!(rx.recv().is_err(), "channel must be closed after error");
    }

    #[test]
    fn validation_errors_match_between_iterator_and_channel() {
        let bad = RasterOptions {
            dpi: -1.0,
            ..valid_opts()
        };
        let (_, iter_err) = render_pages(Path::new("/irrelevant"), &bad)
            .next()
            .expect("iterator must yield one item");
        let rx = render_channel(Path::new("/irrelevant"), &bad, 1);
        let (_, chan_err) = rx.recv().expect("channel must yield one item");
        let Err(RasterError::InvalidOptions(i)) = iter_err else {
            panic!("iterator returned wrong variant: {iter_err:?}")
        };
        let Err(RasterError::InvalidOptions(c)) = chan_err else {
            panic!("channel returned wrong variant: {chan_err:?}")
        };
        assert_eq!(i, c, "validation error messages must be identical");
    }
}
