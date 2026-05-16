//! Core render pipeline: PDF page → pixel buffer.

use std::sync::Arc;

use color::{Gray8, Rgb8};
use raster::Bitmap;

#[cfg(any(feature = "nvjpeg", feature = "nvjpeg2k", feature = "gpu-jpeg-huffman"))]
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
    /// One or more images on the page failed to decode; the rendered page is
    /// incomplete.  Carries the per-image failure messages.
    ImageDecodeFailed(Vec<String>),
    /// The page render was aborted because it exceeded the per-page operator
    /// budget or wall-clock deadline.  The page is pathological (e.g.
    /// infinite-loop content stream or unbounded form-`XObject` recursion).
    /// The rendered bitmap is partial.
    PageBudgetExceeded(String),
    /// The per-page render function panicked.
    ///
    /// Caught only under `panic = "unwind"` builds (e.g. test profiles).
    /// Under the crate's default `panic = "abort"` release profile the panic
    /// aborts the process before this variant can be constructed.
    RenderPanic {
        /// The 1-based page number that panicked.
        page: u32,
        /// The panic message extracted from the payload, if available.
        message: String,
    },
}

impl std::fmt::Display for RasterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidOptions(msg) => write!(f, "invalid raster options: {msg}"),
            // `e` is an `InterpError` that already formats its own labelled
            // message (e.g. "PDF error: …"); delegate so the prefix is not
            // applied twice.
            Self::Pdf(e) => write!(f, "{e}"),
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
            Self::ImageDecodeFailed(v) => write!(f, "image decode failed: {}", v.join("; ")),
            Self::PageBudgetExceeded(msg) => write!(f, "page render budget exceeded: {msg}"),
            Self::RenderPanic { page, message } => {
                write!(f, "page {page} render panicked: {message}")
            }
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
            pdf_interp::InterpError::PageBudget(msg) => Self::PageBudgetExceeded(msg),
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

    /// Borrow the underlying [`pdf::Document`] for read-only operations.
    ///
    /// Prefer [`prescan_session`] over calling this directly — it avoids
    /// exposing `pdf::Document` in your call graph.
    #[doc(hidden)]
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
    /// Prefer [`prescan_session`] and [`render_page_rgb`] over calling this
    /// directly — they avoid exposing `pdf::ObjectId` in your call graph.
    ///
    /// # Errors
    /// [`RasterError::PageOutOfRange`] when `page_num` is `0` or exceeds
    /// `self.total_pages`; [`RasterError::Pdf`] when the underlying page-tree
    /// descent fails (malformed `/Pages` node).
    #[doc(hidden)]
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

/// Classify page `page_num` (1-based) without rendering any pixels.
///
/// Wraps [`pdf_interp::prescan_page`] using the document already loaded in
/// `session` — callers do not need to access `session.doc()` directly.
///
/// Returns a [`PageDiagnostics`] with `has_images`, `has_vector_text`,
/// `dominant_filter`, and a conservative `source_ppi_hint`.
///
/// # Errors
///
/// Returns a [`RasterError`] if `page_num` is outside the document or the
/// page geometry is invalid.
pub fn prescan_session(
    session: &RasterSession,
    page_num: u32,
) -> Result<crate::PageDiagnostics, RasterError> {
    pdf_interp::prescan_page(&session.doc, page_num).map_err(RasterError::from)
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
    let doc = Arc::new(
        pdf_interp::open_decrypting(path, config.decrypt_authorized).map_err(RasterError::from)?,
    );
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
                log::debug!("rasterrocket: Vulkan unavailable under Auto ({e}); trying CUDA next");
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
                    "rasterrocket: GPU initialisation failed ({e}); \
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
        geom.origin_x,
        geom.origin_y,
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
    // RAII guard: reclaim_decoders runs on both normal return and unwind.
    // Drop releases the &mut borrow, so `renderer` is usable directly below.
    let guard = DecoderReclaim(&mut renderer);
    guard.0.execute(&ops);
    guard.0.render_annotations(page_id);
    drop(guard); // reclaim happens here — exactly once on success path

    // Budget exceeded is checked before decode_errors: a budget breach is more
    // fundamental (the page may not have finished rendering at all), and the two
    // conditions are mutually exclusive in practice.
    if let Some(reason) = renderer.budget_status() {
        return Err(RasterError::PageBudgetExceeded(reason.to_string()));
    }

    if !renderer.decode_errors().is_empty() {
        return Err(RasterError::ImageDecodeFailed(
            renderer.decode_errors().to_vec(),
        ));
    }

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
    not(any(feature = "nvjpeg", feature = "nvjpeg2k", feature = "gpu-jpeg-huffman")),
    expect(
        clippy::unnecessary_wraps,
        reason = "Result<()> only carries an error from the GPU-decoder init paths"
    )
)]
#[cfg_attr(
    not(any(
        feature = "nvjpeg",
        feature = "nvjpeg2k",
        feature = "vaapi",
        feature = "gpu-jpeg-huffman"
    )),
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
    #[cfg(not(feature = "vaapi"))]
    let _ = session;
    #[cfg(not(any(
        feature = "nvjpeg",
        feature = "nvjpeg2k",
        feature = "vaapi",
        feature = "gpu-jpeg-huffman"
    )))]
    let _ = renderer;

    #[cfg(feature = "gpu-jpeg-huffman")]
    {
        let dispatch_huffman = {
            #[cfg(feature = "vaapi")]
            {
                !matches!(effective_policy, BackendPolicy::ForceVaapi)
            }
            #[cfg(not(feature = "vaapi"))]
            {
                true
            }
        };
        if dispatch_huffman {
            // Vulkan wins Auto when a Vulkan backend is available.
            #[cfg(feature = "vulkan")]
            let vulkan_active = matches!(effective_policy, BackendPolicy::ForceVulkan)
                || (matches!(effective_policy, BackendPolicy::Auto)
                    && session.vk_backend.is_some());
            #[cfg(not(feature = "vulkan"))]
            let vulkan_active = false;

            if vulkan_active {
                #[cfg(feature = "vulkan")]
                {
                    gpu_init::ensure_jpeg_vk_huffman(effective_policy)
                        .map_err(RasterError::BackendUnavailable)?;
                    gpu_init::JPEG_VK_DEC.with(|cell| {
                        if let gpu_init::JpegGpuInit::Ready(slot) = &mut *cell.borrow_mut() {
                            renderer.set_jpeg_vk(slot.take());
                        }
                    });
                }
            } else {
                gpu_init::ensure_jpeg_gpu_huffman(effective_policy)
                    .map_err(RasterError::BackendUnavailable)?;
                gpu_init::JPEG_CUDA_DEC.with(|cell| {
                    if let gpu_init::JpegGpuInit::Ready(slot) = &mut *cell.borrow_mut() {
                        renderer.set_jpeg_gpu(slot.take());
                    }
                });
            }
        }
    }

    #[cfg(feature = "nvjpeg")]
    {
        let dispatch_nvjpeg = {
            #[cfg(feature = "vaapi")]
            {
                !matches!(effective_policy, BackendPolicy::ForceVaapi)
            }
            #[cfg(not(feature = "vaapi"))]
            {
                true
            }
        };
        if dispatch_nvjpeg {
            gpu_init::ensure_nvjpeg(effective_policy).map_err(RasterError::BackendUnavailable)?;
            gpu_init::NVJPEG_DEC.with(|cell| {
                if let gpu_init::DecoderInit::Ready(slot) = &mut *cell.borrow_mut() {
                    renderer.set_nvjpeg(slot.take());
                }
            });
        }
    }

    #[cfg(feature = "nvjpeg2k")]
    {
        let dispatch_nvjpeg2k = {
            #[cfg(feature = "vaapi")]
            {
                !matches!(effective_policy, BackendPolicy::ForceVaapi)
            }
            #[cfg(not(feature = "vaapi"))]
            {
                true
            }
        };
        if dispatch_nvjpeg2k {
            gpu_init::ensure_nvjpeg2k(effective_policy).map_err(RasterError::BackendUnavailable)?;
            gpu_init::NVJPEG2K_DEC.with(|cell| {
                if let gpu_init::DecoderInit::Ready(slot) = &mut *cell.borrow_mut() {
                    renderer.set_nvjpeg2k(slot.take());
                }
            });
        }
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
    not(any(feature = "nvjpeg", feature = "nvjpeg2k", feature = "gpu-jpeg-huffman")),
    expect(
        clippy::missing_const_for_fn,
        reason = "non-const only in GPU-decoder builds"
    )
)]
fn reclaim_decoders(renderer: &mut pdf_interp::renderer::PageRenderer) {
    #[cfg(not(any(feature = "nvjpeg", feature = "nvjpeg2k", feature = "gpu-jpeg-huffman")))]
    let _ = renderer;
    #[cfg(feature = "gpu-jpeg-huffman")]
    gpu_init::JPEG_CUDA_DEC.with(|cell| {
        if let gpu_init::JpegGpuInit::Ready(slot) = &mut *cell.borrow_mut() {
            *slot = renderer.take_jpeg_gpu();
        }
    });
    #[cfg(all(feature = "gpu-jpeg-huffman", feature = "vulkan"))]
    gpu_init::JPEG_VK_DEC.with(|cell| {
        if let gpu_init::JpegGpuInit::Ready(slot) = &mut *cell.borrow_mut() {
            *slot = renderer.take_jpeg_vk();
        }
    });
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

// ── RAII guard: return decoder handles to TLS even on unwind ─────────────────

/// RAII guard that calls [`reclaim_decoders`] when dropped.
///
/// `lend_decoders` moves per-thread GPU decoder handles out of TLS into
/// the `PageRenderer`. If `execute` or `render_annotations` panics under
/// `panic=unwind` (test builds or unwind-configured embedders), the stack
/// unwind triggers this guard's `Drop`, returning the handles to TLS rather
/// than dropping them. Subsequent pages on the same Rayon worker thread then
/// continue to find their decoder slots populated.
///
/// On the success path, `drop(guard)` is called explicitly — the `&mut`
/// borrow is released and `renderer` is accessible directly for the
/// `decode_errors()`/`finish()` tail. Reclaim runs exactly once either way.
struct DecoderReclaim<'r, 'doc>(&'r mut pdf_interp::renderer::PageRenderer<'doc>);

impl Drop for DecoderReclaim<'_, '_> {
    fn drop(&mut self) {
        reclaim_decoders(self.0);
    }
}

// ── Sequential iterator ───────────────────────────────────────────────────────

struct RenderState {
    session: RasterSession,
    opts: RasterOptions,
    cursor: PageCursor,
}

impl RenderState {
    /// Advance the cursor and return the next page number to render, or
    /// `None` once the cursor is exhausted *or* the next page would fall past
    /// the document's `total_pages`.  Mirrors the documented clamp on
    /// `RasterOptions::last_page` (rendering stops at the last page in the
    /// document rather than erroring).
    fn next_in_range(&mut self) -> Option<u32> {
        let p = self.cursor.next_page()?;
        (p <= self.session.total_pages).then_some(p)
    }
}

/// Cursor over the pages to render.
///
/// `Range` walks every integer in `first..=last` (used when `RasterOptions::pages`
/// is `None`).  `Set` walks the explicit page numbers stored in a `PageSet` —
/// O(set length), not O(last − first), which matters when the set is sparse
/// across a wide range (e.g. `[1, u32::MAX]`).
enum PageCursor {
    Range(std::ops::RangeInclusive<u32>),
    Set { set: PageSet, idx: usize },
}

impl PageCursor {
    fn new(opts: &RasterOptions) -> Self {
        opts.pages.as_ref().map_or_else(
            || Self::Range(opts.first_page..=opts.last_page),
            |ps| Self::Set {
                set: ps.clone(),
                idx: 0,
            },
        )
    }

    fn next_page(&mut self) -> Option<u32> {
        match self {
            Self::Range(r) => r.next(),
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
        let Some(page_num) = state.next_in_range() else {
            // Drop the session early so callers that peek past the end don't
            // hold the document open longer than needed.
            self.state = None;
            return None;
        };
        Some((page_num, render_one(state, page_num)))
    }
}

impl std::iter::FusedIterator for PageIter {}

// ── Channel-based render ──────────────────────────────────────────────────────

/// Catch a panic from a render closure and convert it to `RasterError::RenderPanic`.
///
/// Under `panic = "abort"` (the default release profile) `catch_unwind` is a
/// no-op: a panicking page still aborts the process.  Under `panic = "unwind"`
/// (test builds, and any embedder that opts in) the panic is caught and
/// returned as a per-page error so the render loop can continue.
///
/// Extracted as a generic helper so the panic-catch logic can be unit-tested
/// by injecting an arbitrary render function.
fn catch_page_panic<F>(page_num: u32, f: F) -> (u32, Result<RenderedPage, RasterError>)
where
    F: FnOnce() -> Result<RenderedPage, RasterError> + std::panic::UnwindSafe,
{
    let result = std::panic::catch_unwind(f);
    let outcome = match result {
        Ok(r) => r,
        Err(payload) => Err(panic_payload_to_render_panic(page_num, payload.as_ref())),
    };
    (page_num, outcome)
}

/// Render one page, mapping any panic to `Err(RasterError::RenderPanic)`.
///
/// # `AssertUnwindSafe` justification
///
/// `render_one` receives a shared `&RenderState` and does not mutate any field
/// of the struct itself — `session`, `opts`, and `cursor` are all read-only
/// within `render_one`.  The only interior mutation is in
/// `lend_decoders`/`reclaim_decoders`, which move GPU decoder handles out of
/// thread-local storage into a locally-owned `PageRenderer` and back.  An
/// unwind between those two calls is caught by the [`DecoderReclaim`] RAII
/// guard, which returns the handles to TLS on `Drop`, so a panicking page does
/// not even degrade the next page's decoder backend.  No state shared with
/// other pages is mutated, and the panicking page's local `PageRenderer` is
/// discarded rather than reused — we resume from the next page number.
/// `AssertUnwindSafe` is therefore sound: a caught panic cannot leave `state`
/// or thread-local decoder slots in an inconsistent state that would corrupt
/// subsequent pages.
fn render_one_caught(
    state: &RenderState,
    page_num: u32,
) -> (u32, Result<RenderedPage, RasterError>) {
    catch_page_panic(
        page_num,
        std::panic::AssertUnwindSafe(|| render_one(state, page_num)),
    )
}

/// The body of the [`render_channel`] rayon worker, with every panic site —
/// including session setup — inside a single `catch_unwind`.
///
/// `open_session` is a deep parser entry point (decryption, xref repair,
/// page-tree walks over adversarial input).  If it — or the page cursor —
/// panicked *outside* the catch, the rayon closure would unwind, `tx` would
/// drop, and the consumer would observe a silently closed channel with no
/// `(page_num, Err)` ever delivered: total, undiagnosable page loss.  Catching
/// here guarantees the contract documented on [`render_channel`]: a setup
/// panic is delivered as `(1, Err(RasterError::RenderPanic { .. }))` before the
/// channel closes, exactly as a per-page panic is.  Under `panic = "abort"`
/// (the default release profile) this is a no-op; the guarantee holds for
/// `panic = "unwind"` builds and embedders that opt in.
///
/// `AssertUnwindSafe` is sound here: every value the closure touches is either
/// freshly created inside it (`session`, `state`) or an owned move-in
/// (`path`, `opts`).  `tx` is a `SyncSender`, which carries no
/// unwind-observable invariant — a partially-sent item cannot exist because
/// `send` is atomic per message.  Nothing observable by a *later* call is
/// mutated through a shared reference.
fn render_channel_worker(
    tx: &std::sync::mpsc::SyncSender<(u32, Result<RenderedPage, RasterError>)>,
    path: &std::path::Path,
    opts: RasterOptions,
) {
    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let session = match open_session(path, &SessionConfig::default()) {
            Ok(s) => s,
            Err(e) => {
                let _sent = tx.send((1, Err(e)));
                return;
            }
        };

        let mut state = RenderState {
            cursor: PageCursor::new(&opts),
            session,
            opts,
        };

        while let Some(page_num) = state.next_in_range() {
            let item = render_one_caught(&state, page_num);
            if tx.send(item).is_err() {
                return;
            }
        }
    }));

    if let Err(payload) = caught {
        // A panic escaped session setup or the page cursor.  Deliver it as a
        // page-1 RenderPanic so the consumer sees a loud per-page error rather
        // than an empty, silently closed channel.
        let _sent = tx.send((1, Err(panic_payload_to_render_panic(1, payload.as_ref()))));
    }
}

/// Extract a human-readable message from a caught panic payload and wrap it in
/// [`RasterError::RenderPanic`].  Shared by the per-page and worker-level
/// catch sites so payload handling stays identical.
fn panic_payload_to_render_panic(page: u32, payload: &(dyn std::any::Any + Send)) -> RasterError {
    let message = payload.downcast_ref::<&str>().map_or_else(
        || {
            payload
                .downcast_ref::<String>()
                .cloned()
                .unwrap_or_else(|| "panicked".to_owned())
        },
        |s| (*s).to_owned(),
    );
    RasterError::RenderPanic { page, message }
}

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
        render_channel_worker(&tx, &path_owned, opts_owned);
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

    // ── panic-contract tests ──────────────────────────────────────────────
    //
    // These tests exercise `catch_page_panic` directly by injecting a
    // panicking (or ok-returning) closure.  Test builds use panic=unwind, so
    // catch_unwind is effective here even though the release profile uses
    // panic=abort.  This proves the catch→(page,Err)→continue semantics
    // that render_one_caught relies on.

    #[test]
    fn catch_page_panic_maps_panic_to_render_panic_variant() {
        // A closure that panics with a known message must arrive as
        // (page_num, Err(RasterError::RenderPanic { page, message })).
        let (pn, res) = catch_page_panic(7u32, || -> Result<RenderedPage, RasterError> {
            panic!("boom from page 7");
        });
        assert_eq!(pn, 7, "page number must be preserved through the catch");
        match res {
            Err(RasterError::RenderPanic { page, message }) => {
                assert_eq!(page, 7);
                assert!(
                    message.contains("boom from page 7"),
                    "panic message should be forwarded; got: {message:?}"
                );
            }
            other => panic!("expected RenderPanic, got {other:?}"),
        }
    }

    #[test]
    fn catch_page_panic_ok_path_is_transparent() {
        // A non-panicking closure must pass through unchanged.
        let dummy = RenderedPage {
            page_num: 3,
            width: 1,
            height: 1,
            pixels: vec![128u8],
            dpi: 72.0,
            effective_dpi: 72.0,
            diagnostics: Default::default(),
        };
        let (pn, res) = catch_page_panic(3u32, move || -> Result<RenderedPage, RasterError> {
            Ok(dummy)
        });
        assert_eq!(pn, 3);
        assert!(res.is_ok(), "ok path must not be disturbed; got {res:?}");
    }

    #[test]
    fn catch_page_panic_loop_continues_after_panic() {
        // Simulate the render loop: two pages where page 1 panics, page 2
        // succeeds.  The consumer must receive (1, Err) then (2, Ok).
        let pages: &[(u32, bool)] = &[(1, true), (2, false)]; // (page_num, should_panic)
        let mut results = Vec::new();
        for &(page_num, should_panic) in pages {
            let item = catch_page_panic(page_num, move || -> Result<RenderedPage, RasterError> {
                if should_panic {
                    panic!("synthetic panic on page {page_num}");
                }
                Ok(RenderedPage {
                    page_num,
                    width: 1,
                    height: 1,
                    pixels: vec![0u8],
                    dpi: 72.0,
                    effective_dpi: 72.0,
                    diagnostics: Default::default(),
                })
            });
            results.push(item);
        }
        // Page 1 must be (1, Err(RenderPanic))
        assert_eq!(results[0].0, 1);
        assert!(
            matches!(results[0].1, Err(RasterError::RenderPanic { page: 1, .. })),
            "page 1 must be RenderPanic; got {:?}",
            results[0].1
        );
        // Page 2 must be (2, Ok(_)) — loop continued past the panic
        assert_eq!(results[1].0, 2);
        assert!(
            results[1].1.is_ok(),
            "page 2 must succeed; got {:?}",
            results[1].1
        );
    }

    #[test]
    fn panic_payload_helper_extracts_str_string_and_fallback() {
        // &str payload (the common `panic!("literal")` shape).
        let from_str = std::panic::catch_unwind(|| panic!("str payload"))
            .map(|()| unreachable!())
            .unwrap_err();
        assert!(
            matches!(
                panic_payload_to_render_panic(2, from_str.as_ref()),
                RasterError::RenderPanic { page: 2, ref message } if message.contains("str payload")
            ),
            "str payload must carry through as a page-2 RenderPanic"
        );
        // String payload (formatted panic).
        let n = 9;
        let from_string = std::panic::catch_unwind(|| panic!("string {n}"))
            .map(|()| unreachable!())
            .unwrap_err();
        assert!(
            matches!(
                panic_payload_to_render_panic(3, from_string.as_ref()),
                RasterError::RenderPanic { page: 3, ref message } if message.contains("string 9")
            ),
            "String payload must carry through as a page-3 RenderPanic"
        );
        // Non-string payload → stable fallback, never re-panics.
        let from_other = std::panic::catch_unwind(|| std::panic::panic_any(123u32))
            .map(|()| unreachable!())
            .unwrap_err();
        assert!(
            matches!(
                panic_payload_to_render_panic(4, from_other.as_ref()),
                RasterError::RenderPanic { page: 4, ref message } if message == "panicked"
            ),
            "non-string payload must fall back to the stable \"panicked\" message"
        );
    }

    #[test]
    fn render_channel_worker_delivers_setup_panic_not_silent_close() {
        // The v1-class silent-total-loss guard: a panic in session setup must
        // arrive as (1, Err(RenderPanic)) on the channel, NOT a closed,
        // empty channel.  We can't make `open_session` panic from here, so we
        // drive the worker's catch boundary with a closure-shaped stand-in
        // that mirrors render_channel_worker's structure: a panic before the
        // first send must still be delivered to the consumer.
        use std::sync::mpsc;
        let (tx, rx) = mpsc::sync_channel::<(u32, Result<RenderedPage, RasterError>)>(4);
        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            // Stand-in for `open_session` panicking before any page is sent.
            panic!("synthetic open_session panic");
        }));
        let payload = caught.expect_err("synthetic closure always panics");
        let _sent = tx.send((1, Err(panic_payload_to_render_panic(1, payload.as_ref()))));
        drop(tx);
        let received: Vec<_> = rx.iter().collect();
        assert_eq!(
            received.len(),
            1,
            "consumer must see exactly one delivered error, not an empty closed channel"
        );
        assert_eq!(received[0].0, 1);
        assert!(
            matches!(received[0].1, Err(RasterError::RenderPanic { page: 1, .. })),
            "setup panic must be a page-1 RenderPanic; got {:?}",
            received[0].1
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

#[cfg(test)]
mod decoder_reclaim_guard_tests {
    use std::cell::Cell;

    // ── Stand-in for PageRenderer ─────────────────────────────────────────────
    //
    // Wiring a real PageRenderer into a unit test requires a parsed PDF
    // document, which is too heavy.  Instead we verify the RAII contract with a
    // minimal stand-in that records whether reclaim was called and how many
    // times — proving the guard's Drop semantics without exercising the actual
    // GPU decoder logic.

    thread_local! {
        static RECLAIM_COUNT: Cell<u32> = const { Cell::new(0) };
    }

    struct FakeRenderer;

    fn fake_reclaim(_r: &mut FakeRenderer) {
        RECLAIM_COUNT.with(|c| c.set(c.get() + 1));
    }

    // Mirror of DecoderReclaim but parameterised on FakeRenderer so we can
    // test the exact Drop-calls-reclaim contract without touching GPU state.
    struct FakeGuard<'r>(&'r mut FakeRenderer);

    impl Drop for FakeGuard<'_> {
        fn drop(&mut self) {
            fake_reclaim(self.0);
        }
    }

    fn reset() {
        RECLAIM_COUNT.with(|c| c.set(0));
    }

    fn count() -> u32 {
        RECLAIM_COUNT.with(|c| c.get())
    }

    // ── Tests ─────────────────────────────────────────────────────────────────

    #[test]
    fn reclaim_called_exactly_once_on_normal_exit() {
        reset();
        let mut r = FakeRenderer;
        {
            let guard = FakeGuard(&mut r);
            drop(guard); // explicit drop — mirrors the success path in render_page_rgb_with_geom
        }
        assert_eq!(
            count(),
            1,
            "reclaim must be called exactly once on success path"
        );
    }

    #[test]
    fn reclaim_called_exactly_once_on_unwind() {
        reset();
        let mut r = FakeRenderer;
        // Catch the panic so the test harness doesn't see it as a failure.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = FakeGuard(&mut r);
            panic!("synthetic mid-page panic");
        }));
        assert!(result.is_err(), "catch_unwind must see the panic");
        assert_eq!(count(), 1, "reclaim must be called exactly once on unwind");
    }

    #[test]
    fn reclaim_not_called_twice_if_dropped_explicitly_then_scope_ends() {
        // Regression guard: explicit drop(guard) + end-of-scope must NOT
        // double-reclaim.  After drop(guard) the guard value is gone;
        // the scope ending has nothing left to drop.
        reset();
        let mut r = FakeRenderer;
        let guard = FakeGuard(&mut r);
        drop(guard);
        // `r` is usable again here — borrow was released — mirrors the
        // decode_errors()/finish() tail in render_page_rgb_with_geom.
        let _ = &r;
        assert_eq!(count(), 1, "must not double-reclaim after explicit drop");
    }
}
