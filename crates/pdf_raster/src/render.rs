//! Core render pipeline: PDF page → pixel buffer.

use std::sync::Arc;

use color::{Gray8, Rgb8};
use raster::Bitmap;

#[cfg(any(feature = "nvjpeg", feature = "nvjpeg2k"))]
use crate::gpu_init;
use crate::{BackendPolicy, PageSet, RasterOptions, RenderedPage, SessionConfig};

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

/// Direct conversion from `pdf::PdfError` so call sites can write
/// `e.into()` instead of the two-level `RasterError::from(InterpError::from(e))`.
///
/// `PdfError::PageOutOfRange` carries a **0-based** index per the descender's
/// contract; the surrounding API surface is **1-based**.  Translate to the
/// 1-based [`RasterError::PageOutOfRange`] variant directly so the chained
/// conversion does not silently leak 0-based numbering through
/// `RasterError::Pdf(InterpError::Pdf(...))`.
impl From<pdf::PdfError> for RasterError {
    fn from(e: pdf::PdfError) -> Self {
        match e {
            pdf::PdfError::PageOutOfRange { page, total } => Self::PageOutOfRange {
                page: page.saturating_add(1),
                total,
            },
            other => Self::from(pdf_interp::InterpError::from(other)),
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
    pub(crate) doc: Arc<pdf::Document>,
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
    #[cfg(any(feature = "gpu-aa", feature = "gpu-icc", feature = "cache"))]
    pub(crate) gpu_ctx: Option<Arc<gpu::GpuCtx>>,
    /// Vulkan compute backend.  `Some` when the `vulkan` feature is
    /// enabled, the policy resolves to `Auto` or `ForceVulkan`, and
    /// `VulkanBackend::new` succeeded.  Mutually exclusive with
    /// `gpu_ctx` in practice — under `Auto` the caller skips CUDA
    /// init when this is `Some`; under `ForceVulkan` `init_gpu_ctx`
    /// returns `None` regardless.  Shared across all pages.
    #[cfg(feature = "vulkan")]
    pub(crate) vk_backend: Option<Arc<gpu::backend::vulkan::VulkanBackend>>,
    /// Single-threaded VA-API JPEG decode queue.  One worker thread owns the
    /// `VapiJpegDecoder`; all Rayon page-render threads share handles to it.
    /// `None` when the `vaapi` feature is disabled, policy is `CpuOnly` /
    /// `ForceCuda`, or VA-API initialisation failed (soft failure on `Auto`).
    #[cfg(feature = "vaapi")]
    pub(crate) vaapi_queue: Option<Arc<gpu::DecodeQueue<gpu::vaapi::VapiJpegDecoder>>>,
    /// Device-resident image cache.  `Some` when the `cache`
    /// feature is enabled and a CUDA device is available; `None`
    /// otherwise (CPU-only fallback).  Shared across all pages so
    /// content-hash dedup spans the whole render.
    #[cfg(feature = "cache")]
    pub(crate) image_cache: Option<Arc<gpu::cache::DeviceImageCache>>,
    /// Stable document identifier used as the cache's secondary alias
    /// key.  Today derived from the file path or a per-session UUID;
    /// any 32-byte content-addressable identifier works.
    #[cfg(feature = "cache")]
    pub(crate) doc_id: gpu::cache::DocId,
    /// Image-cache prefetcher handle — kept alive for the session's
    /// lifetime so the discovery + worker threads can drain.
    /// `None` when the cache is disabled or the user opted out via
    /// [`crate::SessionConfig::prefetch`].  Drop semantics: cancels
    /// in-flight prefetch and joins workers; safe to drop mid-render.
    #[cfg(feature = "cache")]
    #[expect(
        dead_code,
        reason = "held for Drop side-effect (cancels + joins prefetcher); \
                  callers don't read it but session lifetime owns it"
    )]
    pub(crate) prefetch: Option<pdf_interp::cache::PrefetchHandle>,
}

impl RasterSession {
    /// Total number of pages in the document.
    #[must_use]
    pub const fn total_pages(&self) -> u32 {
        self.total_pages
    }

    /// Borrow the underlying [`pdf::Document`] for read-only operations such as
    /// the [`crate::prescan_page`] pre-scan pass.
    #[must_use]
    pub fn doc(&self) -> &pdf::Document {
        &self.doc
    }

    /// The backend policy this session was opened with.
    #[must_use]
    pub const fn policy(&self) -> BackendPolicy {
        self.policy
    }

    /// Resolve a 1-based page number to its [`pdf::ObjectId`].
    ///
    /// Each call performs one logarithmic page-tree descent.  Per-render
    /// callers should resolve once at the entry point and pass the id into
    /// `pdf_interp::page_size_pts_by_id` and `pdf_interp::parse_page_by_id`
    /// so the single descent serves all three uses; an earlier shape with a
    /// session-side `RwLock<HashMap>` cache turned out to be hot-write
    /// cold-read on every contest event (each page rendered once) and was
    /// dropped.
    ///
    /// # Errors
    /// [`RasterError::PageOutOfRange`] when `page_num` is `0` or exceeds
    /// `self.total_pages`; [`RasterError::Pdf`] when the underlying page-tree
    /// descent fails (malformed `/Pages` node).
    pub fn resolve_page(&self, page_num: u32) -> Result<pdf::ObjectId, RasterError> {
        // The upper-bound check is structurally duplicated by
        // `Document::get_page`, but doing it here keeps the user-facing
        // error 1-based (the descent returns 0-based `PageOutOfRange`).
        // The `page_num == 0` check is load-bearing: guards `page_num - 1`
        // from u32 underflow.
        if page_num == 0 || page_num > self.total_pages {
            return Err(RasterError::PageOutOfRange {
                page: page_num,
                total: self.total_pages,
            });
        }
        self.doc.get_page(page_num - 1).map_err(RasterError::from)
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
/// Reads `/Pages /Count` directly (O(1) on well-formed PDFs) and defers
/// per-page id resolution until the first render of each page — opening
/// a 100 000-page document and rendering one page no longer pays for a
/// full page-tree walk.  GPU context (AA/ICC) is initialised here; JPEG
/// decoders are initialised lazily per rayon worker thread on first page
/// render.
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
    let doc = Arc::new(pdf_interp::open(path).map_err(RasterError::from)?);
    let total_pages = doc.page_count_fast();

    // Reject `ForceVulkan` at the policy gate when the `vulkan` feature
    // wasn't compiled in; otherwise initialise the Vulkan backend up
    // front so failures surface here rather than mid-render.
    #[cfg(not(feature = "vulkan"))]
    if matches!(config.policy, BackendPolicy::ForceVulkan) {
        return Err(RasterError::BackendUnavailable(
            "ForceVulkan requires the `vulkan` Cargo feature; \
             rebuild with `--features vulkan` or pick another --backend."
                .to_owned(),
        ));
    }
    #[cfg(feature = "vulkan")]
    let vk_backend = init_vk_backend(config.policy)?;

    // Symmetric reject for `ForceCuda` when no CUDA features are
    // compiled in.  Without this, the cfg-gated `init_gpu_ctx` call
    // below silently disappears and `--backend cuda` becomes a CPU
    // render — exactly the silent-fallback behaviour the `Force*`
    // variants exist to prevent.
    #[cfg(not(any(feature = "gpu-aa", feature = "gpu-icc", feature = "cache")))]
    if matches!(config.policy, BackendPolicy::ForceCuda) {
        return Err(RasterError::BackendUnavailable(
            "ForceCuda requires at least one of the `gpu-aa`, `gpu-icc`, or \
             `cache` Cargo features; rebuild with the desired feature set or \
             pick another --backend."
                .to_owned(),
        ));
    }

    // Under `Auto` Vulkan wins; skip CUDA init when it produced a backend
    // so we don't pay its ~240 ms cost for a path we won't use.
    #[cfg(any(feature = "gpu-aa", feature = "gpu-icc", feature = "cache"))]
    let gpu_ctx = {
        #[cfg(feature = "vulkan")]
        let vk_won_auto = vk_backend.is_some() && matches!(config.policy, BackendPolicy::Auto);
        #[cfg(not(feature = "vulkan"))]
        let vk_won_auto = false;

        if vk_won_auto {
            None
        } else {
            init_gpu_ctx(config.policy)?
        }
    };

    let vaapi_device = config.vaapi_device.clone();

    #[cfg(feature = "vaapi")]
    let vaapi_queue = crate::decode_queue::build_vaapi_queue(&vaapi_device, config.policy)
        .map_err(RasterError::BackendUnavailable)?
        .map(Arc::new);

    // Image cache: one per session, shared across all pages.
    // Construction needs a CUDA stream from the GpuCtx, so the cache
    // is gated on gpu_ctx availability — a CpuOnly session or a
    // failed init produces `image_cache = None`.
    #[cfg(feature = "cache")]
    let image_cache = gpu_ctx.as_ref().map(|ctx| {
        let mut cache = gpu::cache::DeviceImageCache::new(
            std::sync::Arc::clone(ctx.stream()),
            // Auto-detect would need a running CUDA stream; we use
            // the spec defaults so a session always boots.  Future
            // work: expose a SessionConfig::cache_budget knob.
            gpu::cache::VramBudget::DEFAULT,
            gpu::cache::HostBudget::DEFAULT,
        );
        // Enable disk persistence when the cache root resolves
        // (HOME / XDG_CACHE_HOME / PDF_RASTER_CACHE_DIR set).  No
        // disk tier in sandboxed environments where none of those
        // env vars are present — the cache stays in-process only.
        if let Some(disk) = gpu::cache::DiskTier::try_new() {
            cache = cache.with_disk(disk);
        }
        Arc::new(cache)
    });

    // DocId: BLAKE3 of the PDF bytes.  Stable per content; editing
    // the PDF naturally invalidates the disk tier (Task 5) because
    // the hash changes.  Costs one full BLAKE3 hash at session open
    // (~250 MB/s; ~40ms for a 10MB PDF) — paid once per session, not
    // per page.  Borrows the bytes already mmapped by `pdf_interp::open`
    // so the hash adds zero IO.
    #[cfg(feature = "cache")]
    let doc_id = {
        let hash = gpu::cache::DeviceImageCache::hash_bytes(doc.bytes());
        gpu::cache::DocId(hash.0)
    };

    // Spawn the prefetcher last (after the cache + doc_id are
    // resolved).  Skipped when the user didn't opt in or when no
    // cache exists to prefetch into.
    #[cfg(feature = "cache")]
    let prefetch = if config.prefetch
        && let Some(cache) = image_cache.as_ref()
    {
        Some(pdf_interp::cache::spawn_prefetch(
            Arc::clone(&doc),
            Arc::clone(cache),
            doc_id,
            pdf_interp::cache::PrefetchConfig::default(),
        ))
    } else {
        None
    };

    Ok(RasterSession {
        doc,
        total_pages,
        policy: config.policy,
        #[cfg(not(feature = "vaapi"))]
        vaapi_device,
        #[cfg(any(feature = "gpu-aa", feature = "gpu-icc", feature = "cache"))]
        gpu_ctx,
        #[cfg(feature = "vulkan")]
        vk_backend,
        #[cfg(feature = "vaapi")]
        vaapi_queue,
        #[cfg(feature = "cache")]
        image_cache,
        #[cfg(feature = "cache")]
        doc_id,
        #[cfg(feature = "cache")]
        prefetch,
    })
}

/// Initialise the Vulkan compute backend.
///
/// Returns `None` on every policy except `ForceVulkan`; errors loudly on
/// `ForceVulkan` if init fails (no silent CPU fallback when the user
/// asked for Vulkan).
#[cfg(feature = "vulkan")]
static VK_BACKEND: std::sync::OnceLock<Result<Arc<gpu::backend::vulkan::VulkanBackend>, String>> =
    std::sync::OnceLock::new();

/// Initialise the Vulkan compute backend.
///
/// `ForceVulkan` returns the backend or a hard error.  `Auto` returns the
/// backend on success and `Ok(None)` on failure — the caller falls
/// through to the CUDA path (and ultimately the CPU path) when Vulkan is
/// unavailable.  All other policies return `Ok(None)` without attempting
/// init.
///
/// Like [`init_gpu_ctx`], the result is cached in a process-wide
/// `OnceLock` so successive sessions don't re-create the device + load
/// shaders.
#[cfg(feature = "vulkan")]
fn init_vk_backend(
    policy: BackendPolicy,
) -> Result<Option<Arc<gpu::backend::vulkan::VulkanBackend>>, RasterError> {
    if !matches!(policy, BackendPolicy::Auto | BackendPolicy::ForceVulkan) {
        return Ok(None);
    }
    let cached = VK_BACKEND.get_or_init(|| match gpu::backend::vulkan::VulkanBackend::new() {
        Ok(b) => Ok(Arc::new(b)),
        Err(e) => Err(e.to_string()),
    });
    match cached {
        Ok(b) => Ok(Some(Arc::clone(b))),
        Err(e) => {
            if matches!(policy, BackendPolicy::ForceVulkan) {
                Err(RasterError::BackendUnavailable(format!(
                    "Vulkan backend required but unavailable: {e}. \
                     Verify with `vulkaninfo` that a Vulkan 1.3+ device is present."
                )))
            } else {
                log::debug!("pdf_raster: Vulkan unavailable under Auto ({e}); trying CUDA next");
                Ok(None)
            }
        }
    }
}

/// Initialise the CUDA GPU context for AA fill and ICC colour transforms.
///
/// Returns `None` on `CpuOnly` and `ForceVulkan`.  Errors loudly on
/// `ForceCuda` if init fails; logs a warning and returns `None` on
/// `Auto` / `ForceVaapi` if init fails.
///
/// The caller is expected to short-circuit *before* calling this when
/// Vulkan already won under `Auto` (see `open_session`) — keeping that
/// dispatch decision at the call site lets this function stay
/// policy-pure.
///
/// The CUDA context, stream, and 7 PTX modules cost ~240 ms warm /
/// ~1100 ms cold to build, and are process-wide state — there is no
/// per-session work hidden inside.  We therefore cache the init result
/// in a process-wide `OnceLock` so workloads that open many short-lived
/// sessions (e.g. one page per archive across 100 archives) pay the
/// cost once instead of once per `open_session` call.
///
/// Failures are also cached: on `Auto`/`ForceVaapi` we retain the
/// fallback `None` so we don't re-attempt CUDA init every session and
/// log the same warning hundreds of times.  `ForceCuda` still surfaces
/// the cached error message verbatim because the caller asked us to
/// fail loud rather than fall back.
#[cfg(any(feature = "gpu-aa", feature = "gpu-icc", feature = "cache"))]
static GPU_CTX: std::sync::OnceLock<Result<Arc<gpu::GpuCtx>, String>> = std::sync::OnceLock::new();

#[cfg(any(feature = "gpu-aa", feature = "gpu-icc", feature = "cache"))]
fn init_gpu_ctx(policy: BackendPolicy) -> Result<Option<Arc<gpu::GpuCtx>>, RasterError> {
    if matches!(policy, BackendPolicy::CpuOnly | BackendPolicy::ForceVulkan) {
        return Ok(None);
    }

    let cached = GPU_CTX.get_or_init(|| match gpu::GpuCtx::init() {
        Ok(ctx) => Ok(Arc::new(ctx)),
        Err(e) => Err(e.to_string()),
    });

    match cached {
        Ok(ctx) => Ok(Some(Arc::clone(ctx))),
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
    let page_id = session.resolve_page(page_num)?;
    let geom = pdf_interp::page_size_pts_by_id(&session.doc, page_id)?;
    render_page_rgb_with_geom(session, page_num, page_id, scale, geom, session.policy)
        .map(|(bmp, _diag)| bmp)
}

/// Like [`render_page_rgb`] but with an affinity-dispatch policy override.
///
/// When `effective_policy` is [`BackendPolicy::CpuOnly`], GPU decoder init is
/// skipped entirely for this page even if the session policy would normally
/// allow it.  The session policy is used as-is for all other variants.
///
/// Use this when content-aware routing has classified the page as not needing
/// GPU decoding — e.g. a pure-vector page where the prescan diagnostics
/// indicate `CpuOnly` is the right effective policy.
///
/// # Errors
///
/// Same as [`render_page_rgb`].
pub fn render_page_rgb_hinted(
    session: &RasterSession,
    page_num: u32,
    scale: f64,
    effective_policy: BackendPolicy,
) -> Result<Bitmap<Rgb8>, RasterError> {
    let page_id = session.resolve_page(page_num)?;
    let geom = pdf_interp::page_size_pts_by_id(&session.doc, page_id)?;
    render_page_rgb_with_geom(session, page_num, page_id, scale, geom, effective_policy)
        .map(|(bmp, _diag)| bmp)
}

/// Inner implementation shared by [`render_page_rgb`], [`render_page_rgb_hinted`], and [`render_one`].
///
/// `effective_policy` overrides `session.policy` for GPU decoder selection only.
/// All other session state (GPU context for AA/ICC, VA-API queue) is unaffected.
fn render_page_rgb_with_geom(
    session: &RasterSession,
    page_num: u32,
    page_id: pdf::ObjectId,
    scale: f64,
    geom: pdf_interp::PageGeometry,
    effective_policy: BackendPolicy,
) -> Result<(Bitmap<Rgb8>, pdf_interp::renderer::PageDiagnostics), RasterError> {
    let _ = page_num; // kept for diagnostic / future tracing; resolution happens once at the entry point
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

    let ops = pdf_interp::parse_page_by_id(doc, page_id)?;

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

    #[cfg(any(feature = "gpu-aa", feature = "gpu-icc", feature = "cache"))]
    renderer.set_gpu_ctx(session.gpu_ctx.as_ref().map(Arc::clone));

    #[cfg(feature = "vulkan")]
    renderer.set_vk_backend(session.vk_backend.as_ref().map(Arc::clone));

    #[cfg(feature = "cache")]
    renderer.set_image_cache(session.image_cache.as_ref().map(Arc::clone), session.doc_id);

    lend_decoders(session, &mut renderer, effective_policy)?;
    renderer.execute(&ops);
    renderer.render_annotations(page_id);
    reclaim_decoders(&mut renderer);

    Ok(renderer.finish())
}

/// Lend per-thread GPU JPEG decoders to the renderer for one page.
///
/// `effective_policy` is normally `session.policy` but callers may pass
/// [`BackendPolicy::CpuOnly`] to skip GPU decoder init for this page regardless
/// of the session-level policy — used by affinity dispatch for `CpuOnly` pages.
///
/// On `CpuOnly` this is a no-op.  On `ForceCuda`/`ForceVaapi` init failure is
/// returned as `RasterError::BackendUnavailable` rather than silently falling back.
#[cfg_attr(
    not(any(feature = "nvjpeg", feature = "nvjpeg2k")),
    expect(
        clippy::unnecessary_wraps,
        reason = "Result<()> only carries an error from the nvjpeg/nvjpeg2k init paths"
    )
)]
#[cfg_attr(
    not(any(feature = "nvjpeg", feature = "nvjpeg2k", feature = "vaapi")),
    expect(
        clippy::missing_const_for_fn,
        reason = "body collapses to a single match when no GPU-decoder feature is on"
    )
)]
fn lend_decoders(
    session: &RasterSession,
    renderer: &mut pdf_interp::renderer::PageRenderer,
    effective_policy: BackendPolicy,
) -> Result<(), RasterError> {
    if matches!(effective_policy, BackendPolicy::CpuOnly) {
        return Ok(());
    }
    // `session` and `renderer` are used inside `#[cfg]`-gated blocks below.
    // `session` is only read on the vaapi path; `renderer` only on the
    // nvjpeg/nvjpeg2k/vaapi paths. Suppress the unused-variable warning when
    // the relevant feature is off.
    #[cfg(not(feature = "vaapi"))]
    let _ = session;
    #[cfg(not(any(feature = "nvjpeg", feature = "nvjpeg2k", feature = "vaapi")))]
    let _ = renderer;

    #[cfg(feature = "nvjpeg")]
    if !matches!(effective_policy, BackendPolicy::ForceVaapi) {
        gpu_init::ensure_nvjpeg(effective_policy).map_err(RasterError::BackendUnavailable)?;
        gpu_init::NVJPEG_DEC.with(|cell| {
            if let gpu_init::DecoderInit::Ready(slot) = &mut *cell.borrow_mut() {
                renderer.set_nvjpeg(slot.take());
            }
        });
    }

    #[cfg(feature = "nvjpeg2k")]
    if !matches!(effective_policy, BackendPolicy::ForceVaapi) {
        gpu_init::ensure_nvjpeg2k(effective_policy).map_err(RasterError::BackendUnavailable)?;
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
    cursor: PageCursor,
}

/// Iterator over the pages to render.
///
/// `Range` walks every integer in `first..=last` (used when `RasterOptions::pages`
/// is `None`).  `Set` walks the explicit page numbers stored in a `PageSet` —
/// O(set length), not O(last − first), which matters when the set is sparse
/// across a wide range (e.g. `[1, u32::MAX]`).
enum PageCursor {
    Range { next: u32, end: u32 },
    Set { set: PageSet, idx: usize },
}

impl PageCursor {
    #[expect(
        clippy::option_if_let_else,
        reason = "match arms read more clearly than a 2-branch map_or here"
    )]
    fn new(opts: &RasterOptions) -> Self {
        match opts.pages.as_ref() {
            Some(ps) => Self::Set {
                set: ps.clone(),
                idx: 0,
            },
            None => Self::Range {
                next: opts.first_page,
                end: opts.last_page,
            },
        }
    }

    fn next_page(&mut self) -> Option<u32> {
        match self {
            Self::Range { next, end } => {
                if *next > *end {
                    return None;
                }
                let p = *next;
                // After yielding u32::MAX, bump `end` to 0 so the next call
                // returns None — saturating_add alone would leave next == end
                // == u32::MAX and yield forever.
                if p == u32::MAX {
                    *end = 0;
                } else {
                    *next = p + 1;
                }
                Some(p)
            }
            Self::Set { set, idx } => {
                let p = *set.as_slice().get(*idx)?;
                *idx += 1;
                Some(p)
            }
        }
    }
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

    let state = open_session(path, &SessionConfig::default()).map(|session| RenderState {
        cursor: PageCursor::new(opts),
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
        // Surface a deferred validation/open error exactly once, then close.
        if self.state.as_ref()?.is_err() {
            let err = self.state.take()?.err()?;
            return Some((1, Err(err)));
        }

        let state = self.state.as_mut()?.as_mut().ok()?;
        let page_num = state.cursor.next_page()?;
        if page_num > state.session.total_pages {
            self.state = None;
            return None;
        }
        Some((page_num, render_one(state, page_num)))
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

    let path_owned = path.to_owned();
    let opts_owned = opts.clone();

    rayon::spawn(move || {
        let session = match open_session(&path_owned, &SessionConfig::default()) {
            Ok(s) => s,
            Err(e) => {
                let _sent = tx.send((1, Err(e)));
                return;
            }
        };

        let mut state = RenderState {
            cursor: PageCursor::new(&opts_owned),
            session,
            opts: opts_owned,
        };

        while let Some(page_num) = state.cursor.next_page() {
            if page_num > state.session.total_pages {
                return;
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

    let page_id = state.session.resolve_page(page_num)?;
    let geom = pdf_interp::page_size_pts_by_id(&state.session.doc, page_id)?;

    let (rgb, diagnostics) = render_page_rgb_with_geom(
        &state.session,
        page_num,
        page_id,
        scale,
        geom,
        state.session.policy,
    )?;
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

    fn opts_with_pages(set: crate::PageSet) -> RasterOptions {
        RasterOptions {
            dpi: 72.0,
            first_page: 0,
            last_page: 0,
            deskew: false,
            pages: Some(set),
        }
    }

    #[test]
    fn cursor_range_yields_inclusive_window() {
        let opts = RasterOptions {
            dpi: 72.0,
            first_page: 3,
            last_page: 5,
            deskew: false,
            pages: None,
        };
        let mut c = PageCursor::new(&opts);
        assert_eq!(c.next_page(), Some(3));
        assert_eq!(c.next_page(), Some(4));
        assert_eq!(c.next_page(), Some(5));
        assert_eq!(c.next_page(), None);
        assert_eq!(c.next_page(), None);
    }

    #[test]
    fn cursor_range_saturates_at_u32_max() {
        // last_page=u32::MAX is the documented "render to end of document" idiom.
        // next.saturating_add(1) must keep `next > end` once we've yielded u32::MAX.
        let opts = RasterOptions {
            dpi: 72.0,
            first_page: u32::MAX - 1,
            last_page: u32::MAX,
            deskew: false,
            pages: None,
        };
        let mut c = PageCursor::new(&opts);
        assert_eq!(c.next_page(), Some(u32::MAX - 1));
        assert_eq!(c.next_page(), Some(u32::MAX));
        assert_eq!(c.next_page(), None);
    }

    #[test]
    fn cursor_set_yields_exactly_set_members() {
        let ps = crate::PageSet::new(vec![5, 1, 10, 1]).unwrap();
        let opts = opts_with_pages(ps);
        let mut c = PageCursor::new(&opts);
        assert_eq!(c.next_page(), Some(1));
        assert_eq!(c.next_page(), Some(5));
        assert_eq!(c.next_page(), Some(10));
        assert_eq!(c.next_page(), None);
    }

    #[test]
    fn cursor_set_walks_only_set_length_for_sparse_input() {
        // Regression: previously `PageIter` walked every integer in
        // first()..=last() and probed `PageSet::contains` on each — for
        // [1, u32::MAX] that meant ~4.3 billion iterations.  The cursor must
        // yield exactly two pages and terminate.
        let ps = crate::PageSet::new(vec![1, u32::MAX]).unwrap();
        let opts = opts_with_pages(ps);
        let mut c = PageCursor::new(&opts);
        let mut yielded = Vec::new();
        while let Some(p) = c.next_page() {
            yielded.push(p);
            assert!(
                yielded.len() <= 2,
                "cursor yielded a third page on a 2-element PageSet"
            );
        }
        assert_eq!(yielded, vec![1, u32::MAX]);
    }
}
