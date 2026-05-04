//! Core render pipeline: PDF page → pixel buffer.

#[cfg(any(feature = "gpu-aa", feature = "gpu-icc", feature = "vaapi"))]
use std::sync::Arc;

use color::{Gray8, Rgb8};
use raster::Bitmap;

#[cfg(any(feature = "nvjpeg", feature = "nvjpeg2k"))]
use crate::gpu_init;
use crate::{BackendPolicy, RasterOptions, RenderedPage, SessionConfig};

// ── Safety limit ──────────────────────────────────────────────────────────────

/// Maximum pixel dimension (width or height) accepted from a PDF page.
///
/// Prevents absurdly large allocations from malformed or adversarial documents.
/// 32 768 px at 150 DPI corresponds to roughly 366 inches (~9.3 metres).
pub const MAX_PX_DIMENSION: u32 = 32_768;

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
    /// A backend was required via [`BackendPolicy`] but could not be initialised.
    BackendUnavailable(String),
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
            Self::BackendUnavailable(msg) => write!(f, "backend unavailable: {msg}"),
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
/// `Sync` because the document is read-only after construction, the GPU context
/// is `Arc`-wrapped, and the VA-API decode queue is `Arc`-wrapped (its inner
/// `mpsc::Sender` is `Send + Sync`).
pub struct RasterSession {
    pub(crate) doc: lopdf::Document,
    /// Page-number → object-ID map, built once at construction.
    pub(crate) pages: std::collections::BTreeMap<u32, lopdf::ObjectId>,
    pub(crate) total_pages: u32,
    pub(crate) policy: BackendPolicy,
    /// VA-API DRM render node path. Retained in non-vaapi builds for forward
    /// compatibility; in vaapi builds the path is consumed during `open_session`
    /// to construct `vaapi_queue` and is not needed afterward.
    #[cfg(not(feature = "vaapi"))]
    #[expect(
        dead_code,
        reason = "stored but not read back; retained for SessionConfig symmetry"
    )]
    pub(crate) vaapi_device: String,
    #[cfg(any(feature = "gpu-aa", feature = "gpu-icc"))]
    pub(crate) gpu_ctx: Option<Arc<gpu::GpuCtx>>,
    /// Single-threaded VA-API JPEG decode queue.  One worker thread owns the
    /// `VapiJpegDecoder`; all Rayon page-render threads share handles to it.
    /// `None` when the `vaapi` feature is disabled, policy is `CpuOnly` /
    /// `ForceCuda`, or VA-API initialisation failed (soft failure on `Auto`).
    #[cfg(feature = "vaapi")]
    pub(crate) vaapi_queue: Option<Arc<gpu::DecodeQueue<gpu::vaapi::VapiJpegDecoder>>>,
}

impl RasterSession {
    /// Total number of pages in the document.
    #[must_use]
    pub const fn total_pages(&self) -> u32 {
        self.total_pages
    }

    /// Borrow the underlying `lopdf::Document` for read-only operations such as
    /// the [`pdf_raster::prescan_page`] pre-scan pass.
    #[must_use]
    pub const fn doc(&self) -> &lopdf::Document {
        &self.doc
    }
}

// Compile-time assertions: RasterSession must be Sync (shared across rayon threads) and
// Send (moved into the rayon::spawn closure in render_channel).
const _: fn() = || {
    const fn assert_sync<T: Sync>() {}
    const fn assert_send<T: Send>() {}
    assert_sync::<RasterSession>();
    assert_send::<RasterSession>();
};

/// Open a PDF and create a [`RasterSession`] for rendering.
///
/// Eagerly builds the page-ID map so repeated per-page calls are O(1) lookups.
/// GPU context (AA/ICC) is initialised here; JPEG decoders are initialised
/// lazily per rayon worker thread on first page render.
///
/// # Errors
///
/// - [`RasterError::Pdf`] if the file cannot be opened or parsed.
/// - [`RasterError::BackendUnavailable`] if `config.policy` is `ForceCuda` or
///   `ForceVaapi` and the required GPU context fails to initialise.
pub fn open_session(
    path: &std::path::Path,
    config: &SessionConfig,
) -> Result<RasterSession, RasterError> {
    let doc = pdf_interp::open(path).map_err(RasterError::from)?;
    let pages = doc.get_pages();
    let total_pages = u32::try_from(pages.len()).map_err(|_| {
        RasterError::InvalidPageGeometry(format!(
            "document has {} pages which exceeds u32::MAX — this is not a valid PDF",
            pages.len()
        ))
    })?;

    #[cfg(any(feature = "gpu-aa", feature = "gpu-icc"))]
    let gpu_ctx = init_gpu_ctx(config.policy)?;

    let vaapi_device = config.vaapi_device.clone();

    #[cfg(feature = "vaapi")]
    let vaapi_queue = crate::decode_queue::build_vaapi_queue(&vaapi_device, config.policy)
        .map_err(RasterError::BackendUnavailable)?
        .map(Arc::new);

    Ok(RasterSession {
        doc,
        pages,
        total_pages,
        policy: config.policy,
        #[cfg(not(feature = "vaapi"))]
        vaapi_device,
        #[cfg(any(feature = "gpu-aa", feature = "gpu-icc"))]
        gpu_ctx,
        #[cfg(feature = "vaapi")]
        vaapi_queue,
    })
}

/// Initialise the CUDA GPU context for AA fill and ICC colour transforms.
///
/// Returns `None` on `CpuOnly`; errors loudly on `ForceCuda` if init fails;
/// logs a warning and returns `None` on `Auto`/`ForceVaapi` if init fails.
#[cfg(any(feature = "gpu-aa", feature = "gpu-icc"))]
fn init_gpu_ctx(policy: BackendPolicy) -> Result<Option<Arc<gpu::GpuCtx>>, RasterError> {
    if matches!(policy, BackendPolicy::CpuOnly) {
        return Ok(None);
    }
    match gpu::GpuCtx::init() {
        Ok(ctx) => Ok(Some(Arc::new(ctx))),
        Err(e) => {
            if matches!(policy, BackendPolicy::ForceCuda) {
                Err(RasterError::BackendUnavailable(format!(
                    "CUDA GPU context required but unavailable: {e}. \
                     Verify with `nvidia-smi` that the driver is loaded."
                )))
            } else {
                log::warn!(
                    "pdf_raster: GPU initialisation failed ({e}); \
                     falling back to CPU. Run `nvidia-smi` to verify the driver is loaded."
                );
                Ok(None)
            }
        }
    }
}

/// Render one page to an RGB bitmap.
///
/// GPU image decoders are initialised lazily per calling thread on first use
/// and reused across pages — safe to call from multiple rayon threads
/// concurrently.
///
/// `scale` is the pixel-per-point multiplier: `dpi / 72.0` for square-pixel
/// rendering, or `(x_dpi/72 · y_dpi/72).sqrt()` for the geometric mean when
/// horizontal and vertical DPI differ.  Must be a positive finite number.
///
/// # Errors
///
/// - [`RasterError::InvalidOptions`] if `scale` is ≤ 0 or non-finite.
/// - [`RasterError::BackendUnavailable`] if a forced backend fails to init on
///   this thread.
/// - [`RasterError::InvalidPageGeometry`] / [`RasterError::PageDegenerate`] /
///   [`RasterError::PageTooLarge`] / [`RasterError::PageOutOfRange`] /
///   [`RasterError::Pdf`] as documented on the error variants.
pub fn render_page_rgb(
    session: &RasterSession,
    page_num: u32,
    scale: f64,
) -> Result<Bitmap<Rgb8>, RasterError> {
    let geom = pdf_interp::page_size_pts(&session.doc, page_num)?;
    render_page_rgb_with_geom(session, page_num, scale, geom).map(|(bmp, _diag)| bmp)
}

/// Inner implementation shared by [`render_page_rgb`] and [`render_one`].
fn render_page_rgb_with_geom(
    session: &RasterSession,
    page_num: u32,
    scale: f64,
    geom: pdf_interp::PageGeometry,
) -> Result<(Bitmap<Rgb8>, pdf_interp::renderer::PageDiagnostics), RasterError> {
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
    )?;

    #[cfg(any(feature = "gpu-aa", feature = "gpu-icc"))]
    renderer.set_gpu_ctx(session.gpu_ctx.as_ref().map(Arc::clone));

    lend_decoders(session, &mut renderer)?;
    renderer.execute(&ops);
    renderer.render_annotations(page_id);
    reclaim_decoders(&mut renderer);

    Ok(renderer.finish())
}

/// Lend per-thread GPU JPEG decoders to the renderer for one page.
///
/// Decoders are initialised lazily on first call.  On `CpuOnly` this is a
/// no-op.  On `ForceCuda`/`ForceVaapi` init failure is returned as
/// `RasterError::BackendUnavailable` rather than silently falling back.
#[cfg_attr(
    not(any(feature = "nvjpeg", feature = "nvjpeg2k")),
    allow(
        clippy::unnecessary_wraps,
        clippy::missing_const_for_fn,
        reason = "return type and body vary with GPU feature flags"
    )
)]
fn lend_decoders(
    session: &RasterSession,
    renderer: &mut pdf_interp::renderer::PageRenderer,
) -> Result<(), RasterError> {
    let policy = session.policy;
    if matches!(policy, BackendPolicy::CpuOnly) {
        return Ok(());
    }
    // `renderer` is used inside `#[cfg]`-gated blocks below; suppress the
    // unused-variable warning in CPU-only / no-GPU-decoder builds.
    #[cfg(not(any(feature = "nvjpeg", feature = "nvjpeg2k", feature = "vaapi")))]
    let _ = renderer;

    #[cfg(feature = "nvjpeg")]
    if !matches!(policy, BackendPolicy::ForceVaapi) {
        gpu_init::ensure_nvjpeg(policy).map_err(RasterError::BackendUnavailable)?;
        gpu_init::NVJPEG_DEC.with(|cell| {
            if let gpu_init::DecoderInit::Ready(slot) = &mut *cell.borrow_mut() {
                renderer.set_nvjpeg(slot.take());
            }
        });
    }

    #[cfg(feature = "nvjpeg2k")]
    if !matches!(policy, BackendPolicy::ForceVaapi) {
        gpu_init::ensure_nvjpeg2k(policy).map_err(RasterError::BackendUnavailable)?;
        gpu_init::NVJPEG2K_DEC.with(|cell| {
            if let gpu_init::DecoderInit::Ready(slot) = &mut *cell.borrow_mut() {
                renderer.set_nvjpeg2k(slot.take());
            }
        });
    }

    #[cfg(feature = "vaapi")]
    if let Some(queue) = &session.vaapi_queue {
        renderer.set_vaapi_queue(queue.handle());
    }

    Ok(())
}

/// Return GPU JPEG decoders from the renderer back into TLS slots for reuse.
///
/// The VA-API path is omitted here: `JpegQueueHandle` is cheaply cloneable and
/// is simply dropped with the renderer — no reclaim step is needed.  The
/// `Arc<DecodeQueue>` in `RasterSession` keeps the worker alive across pages.
#[cfg_attr(
    not(any(feature = "nvjpeg", feature = "nvjpeg2k")),
    expect(
        clippy::missing_const_for_fn,
        reason = "non-const only in GPU-decoder builds"
    )
)]
fn reclaim_decoders(renderer: &mut pdf_interp::renderer::PageRenderer) {
    #[cfg(not(any(feature = "nvjpeg", feature = "nvjpeg2k")))]
    let _ = renderer;
    #[cfg(feature = "nvjpeg")]
    gpu_init::NVJPEG_DEC.with(|cell| {
        if let gpu_init::DecoderInit::Ready(slot) = &mut *cell.borrow_mut() {
            *slot = renderer.take_nvjpeg();
        }
    });
    #[cfg(feature = "nvjpeg2k")]
    gpu_init::NVJPEG2K_DEC.with(|cell| {
        if let gpu_init::DecoderInit::Ready(slot) = &mut *cell.borrow_mut() {
            *slot = renderer.take_nvjpeg2k();
        }
    });
}

// ── Sequential iterator ───────────────────────────────────────────────────────

struct RenderState {
    session: RasterSession,
    opts: RasterOptions,
    current_page: u32,
}

/// Returns the inclusive page range to iterate: `PageSet` bounds when
/// `opts.pages` is set, otherwise `first_page..=last_page`.
fn page_window(opts: &RasterOptions) -> std::ops::RangeInclusive<u32> {
    opts.pages
        .as_ref()
        .map_or(opts.first_page..=opts.last_page, |ps| {
            ps.first()..=ps.last()
        })
}

/// Returns `true` if `page_num` should be rendered given `opts`.
///
/// When `opts.pages` is `None` every page in the iteration window is rendered.
/// When `Some`, only pages present in the [`PageSet`] are rendered.
fn should_render(opts: &RasterOptions, page_num: u32) -> bool {
    opts.pages.as_ref().is_none_or(|ps| ps.contains(page_num))
}

fn validate_opts(opts: &RasterOptions) -> Option<RasterError> {
    if opts.dpi <= 0.0 || !opts.dpi.is_finite() {
        return Some(RasterError::InvalidOptions(format!(
            "dpi must be a positive finite number, got {}",
            opts.dpi
        )));
    }
    if opts.pages.is_none() {
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

    let window = page_window(opts);
    let state = open_session(path, &SessionConfig::default()).map(|session| RenderState {
        current_page: *window.start(),
        session,
        opts: opts.clone(),
    });

    PageIter { state: Some(state) }
}

struct PageIter {
    state: Option<Result<RenderState, RasterError>>,
}

impl Iterator for PageIter {
    type Item = (u32, Result<RenderedPage, RasterError>);

    fn next(&mut self) -> Option<Self::Item> {
        match self.state.as_mut()? {
            Err(_) => {
                let Err(e) = self.state.take()? else {
                    unreachable!("matched Err arm above")
                };
                Some((1, Err(e)))
            }
            Ok(state) => {
                let window = page_window(&state.opts);
                loop {
                    if state.current_page > *window.end()
                        || state.current_page > state.session.total_pages
                    {
                        self.state = None;
                        return None;
                    }

                    let page_num = state.current_page;
                    // Saturate rather than wrap: u32::MAX is a legal page number
                    // in a PageSet, so a plain `+= 1` would overflow to 0 and
                    // loop forever on the next iteration.
                    state.current_page = state.current_page.saturating_add(1);

                    if !should_render(&state.opts, page_num) {
                        continue;
                    }

                    return Some((page_num, render_one(state, page_num)));
                }
            }
        }
    }
}

// ── Channel-based render ──────────────────────────────────────────────────────

#[must_use]
pub fn render_channel(
    path: &std::path::Path,
    opts: &RasterOptions,
    capacity: usize,
) -> std::sync::mpsc::Receiver<(u32, Result<RenderedPage, RasterError>)> {
    use std::sync::mpsc;

    let capacity = capacity.max(1);
    let (tx, rx) = mpsc::sync_channel(capacity);

    if let Some(e) = validate_opts(opts) {
        let _sent = tx.send((1, Err(e)));
        return rx;
    }

    let path_owned: std::path::PathBuf = path.to_owned();
    let opts_owned: RasterOptions = opts.clone();

    rayon::spawn(move || {
        let session = match open_session(&path_owned, &SessionConfig::default()) {
            Ok(s) => s,
            Err(e) => {
                let _sent = tx.send((1, Err(e)));
                return;
            }
        };

        let window = page_window(&opts_owned);
        let last = *window.end().min(&session.total_pages);
        let state = RenderState {
            current_page: *window.start(),
            session,
            opts: opts_owned,
        };

        for page_num in *window.start()..=last {
            if !should_render(&state.opts, page_num) {
                continue;
            }
            let result = render_one(&state, page_num);
            if tx.send((page_num, result)).is_err() {
                return;
            }
        }
    });

    rx
}

// ── Single-page render (gray + deskew) ───────────────────────────────────────

fn render_one(state: &RenderState, page_num: u32) -> Result<RenderedPage, RasterError> {
    let dpi = state.opts.dpi;
    let scale = f64::from(dpi) / 72.0;

    let geom = pdf_interp::page_size_pts(&state.session.doc, page_num)?;

    let (rgb, diagnostics) = render_page_rgb_with_geom(&state.session, page_num, scale, geom)?;
    let mut gray = rgb_to_gray(&rgb);

    if state.opts.deskew {
        crate::deskew::apply(&mut gray).map_err(|e| RasterError::Deskew(e.to_string()))?;
    }

    let pixels = bitmap_to_vec(&gray);

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
#[must_use]
pub fn rgb_to_gray(src: &Bitmap<Rgb8>) -> Bitmap<Gray8> {
    let mut dst = Bitmap::<Gray8>::new(src.width, src.height, 1, false);
    let w = src.width as usize;
    for y in 0..src.height {
        let src_row = &src.row_bytes(y)[..w * 3];
        let dst_row = &mut dst.row_bytes_mut(y)[..w];
        for (dst_px, rgb) in dst_row.iter_mut().zip(src_row.chunks_exact(3)) {
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
            pages: None,
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
        let rx = render_channel(Path::new("/no_such_file_xyz.pdf"), &valid_opts(), 1);
        drop(rx);
    }

    #[test]
    fn capacity_zero_raised_to_one_no_deadlock() {
        let bad = RasterOptions {
            dpi: 0.0,
            ..valid_opts()
        };
        let rx = render_channel(Path::new("/irrelevant"), &bad, 0);
        assert!(
            rx.recv().is_ok(),
            "error item must be delivered even with capacity=0"
        );
        assert!(rx.recv().is_err(), "channel must be closed after error");
    }

    #[test]
    #[ignore = "requires tests/fixtures/corpus-01-native-text-small.pdf (not in repo)"]
    fn sparse_pages_only_yields_requested_pages() {
        // corpus-01 is a small native-text PDF — use pages 1 and 3 from it.
        // If the doc has fewer than 3 pages, the test still passes (page 3 won't arrive).
        let ps = crate::PageSet::new(vec![1, 3]).unwrap();
        let opts = RasterOptions {
            dpi: 72.0,
            first_page: 1,  // ignored when pages is Some
            last_page: 100, // ignored when pages is Some
            deskew: false,
            pages: Some(ps),
        };
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");
        let path = std::path::Path::new(&manifest_dir)
            .parent()
            .expect("parent dir should exist")
            .parent()
            .expect("grandparent dir should exist")
            .join("tests/fixtures/corpus-01-native-text-small.pdf");
        // Collect all yielded page numbers from the iterator
        let yielded: Vec<u32> = render_pages(path.as_path(), &opts)
            .filter_map(|(pn, r)| r.ok().map(|_| pn))
            .collect();
        assert!(
            !yielded.is_empty(),
            "expected at least one page to render successfully"
        );
        // Must only contain pages 1 and/or 3 — no 2, no 4+
        for pn in &yielded {
            assert!(
                *pn == 1 || *pn == 3,
                "unexpected page {pn} in sparse render output"
            );
        }
    }

    #[test]
    #[ignore = "requires tests/fixtures/corpus-01-native-text-small.pdf (not in repo)"]
    fn sparse_pages_channel_only_yields_requested_pages() {
        let ps = crate::PageSet::new(vec![1, 3]).unwrap();
        let opts = RasterOptions {
            dpi: 72.0,
            first_page: 1,
            last_page: 100,
            deskew: false,
            pages: Some(ps),
        };
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");
        let path = std::path::Path::new(&manifest_dir)
            .parent()
            .expect("parent dir should exist")
            .parent()
            .expect("grandparent dir should exist")
            .join("tests/fixtures/corpus-01-native-text-small.pdf");
        let rx = render_channel(path.as_path(), &opts, 4);
        let mut yielded = Vec::new();
        while let Ok((pn, r)) = rx.recv() {
            if r.is_ok() {
                yielded.push(pn);
            }
        }
        assert!(
            !yielded.is_empty(),
            "expected at least one page to render successfully"
        );
        for pn in &yielded {
            assert!(
                *pn == 1 || *pn == 3,
                "unexpected page {pn} in sparse channel output"
            );
        }
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

    #[test]
    fn pages_some_bypasses_first_last_page_validation() {
        // When pages=Some, first_page/last_page are documented as ignored —
        // validate_opts must not reject zero-value or inverted defaults.
        let ps = crate::PageSet::new(vec![3u32]).unwrap();
        let opts = RasterOptions {
            dpi: 72.0,
            first_page: 0, // would normally be rejected
            last_page: 0,  // would normally be rejected (< first_page after clamping)
            deskew: false,
            pages: Some(ps),
        };
        assert!(
            validate_opts(&opts).is_none(),
            "zero first/last_page must be accepted when pages=Some"
        );
    }
}
