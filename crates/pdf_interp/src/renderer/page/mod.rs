//! Per-page operator dispatcher.
//!
//! [`PageRenderer`] holds a target [`Bitmap`] and a [`GStateStack`], iterates
//! over a decoded operator sequence, and calls into the `raster` crate for
//! each painting operator.
//!
//! Sub-modules (all private; named for orientation only):
//! - `operators` — `execute_one` operator dispatch table.
//! - `text` — `show_text` and `show_text_type3` rendering.
//! - `patterns` — `resolve_fill_pattern` and `render_pattern_tile`.
//! - `text_ops` — `GlyphRecord` and `text_to_device` helpers.
//! - `annotations` — annotation appearance stream rendering.
//! - `gpu_ops` — GPU-accelerated fill dispatch.
//!
//! # What is implemented
//!
//! - Graphics state: `q Q cm w J j M d ri i gs`
//! - Path construction: `m l c v y h re`
//! - Path painting: `S s f F f* B B* b b* n`
//! - Clip paths: `W W*` (winding and even-odd; intersected into the current `Clip`)
//! - Colour: `g G rg RG k K sc scn SC SCN cs CS`
//! - Text objects + state: `BT ET Tf Tc Tw Tz TL Ts Tr Td TD Tm T*`
//! - Text showing: `Tj TJ ' "` (via `FreeType` through font crate)
//! - Font encoding `Differences` array — resolved via Adobe Glyph List
//!
//! - Type 0 / `CIDFont` composite fonts — `Encoding` `CMap` (including
//!   embedded stream `CMaps` and `Identity-H`/`V`), `CIDToGIDMap`, `DW`/`W`
//!   advance widths, and multi-byte character code iteration
//!
//! - Text render modes 4–7 — fill/stroke painting combined with text-as-clip
//!   (`glyph_path` outlines accumulated into the clip region per PDF §9.3.6)
//! - `PatternType` 1 tiling patterns via `scn`/`SCN`
//! - Type 3 paint-procedure fonts — `CharProc` content streams, `d0`/`d1` metrics
//!
//! - Optional Content Groups (OCG / layers) — `BDC /OC` respects `OCProperties/D`
//!   default state; inactive groups are skipped at operator level
//! - Annotation rendering — `AP/N` normal appearance streams blitted after page content
//! - Transparency groups — Form `XObjects` with `Group /S /Transparency` rendered into an
//!   intermediate bitmap via `raster::transparency` and composited back (PDF §11.6.6)
//!
//! # Not yet implemented
//!
//! - `ExtGState` blend mode (`BM`) — only `Normal` is currently mapped
//! - Type 0 named `CMaps` other than `Identity-H`/`V` (e.g. `/GB-EUC-H`)

mod annotations;
mod gpu_ops;
mod operators;
mod patterns;
mod text;
mod text_ops;
#[cfg(feature = "vulkan")]
mod vk_ops;

use pdf::{Document, ObjectId};

use color::Rgb8;
use font::{
    cache::GlyphCache,
    engine::{FontEngine, SharedEngine},
};
use raster::{
    Bitmap, eo_fill, fill,
    path::PathBuilder,
    pipe::{Pattern, PipeSrc, PipeState},
    shading::{gouraud::gouraud_triangle_fill, shaded_fill},
    state::TransferSet,
    stroke::{StrokeParams, stroke},
    transparency::{GroupParams, begin_group, paint_group},
    types::{BlendMode, LineCap, LineJoin},
    xpath::XPath,
};

use super::color::RasterColor;
use super::font_cache::FontCache;
use super::gstate::{GStateStack, ctm_multiply, ctm_transform};
use crate::InterpError;
use crate::content::Operator;
use crate::resources::{IMAGE_FILTER_COUNT, ImageColorSpace, ImageFilter, PageResources};
#[cfg(any(feature = "gpu-aa", feature = "gpu-icc", feature = "cache"))]
use gpu::GpuCtx;
#[cfg(feature = "vaapi")]
use gpu::JpegQueueHandle;
#[cfg(feature = "gpu-jpeg-huffman")]
use gpu::backend::cuda::CudaBackend;
#[cfg(feature = "vulkan")]
use gpu::backend::vulkan::VulkanBackend;
#[cfg(feature = "cache")]
use gpu::cache::{DeviceImageCache, DevicePageBuffer, DocId};
#[cfg(feature = "gpu-jpeg-huffman")]
use gpu::jpeg_decoder::JpegGpuDecoder;
#[cfg(feature = "nvjpeg")]
use gpu::nvjpeg::NvJpegDecoder;
#[cfg(feature = "nvjpeg2k")]
use gpu::nvjpeg2k::NvJpeg2kDecoder;
#[cfg(any(
    feature = "gpu-aa",
    feature = "gpu-icc",
    feature = "cache",
    feature = "vulkan"
))]
use std::sync::Arc;

const _: () = assert!(
    IMAGE_FILTER_COUNT == 6,
    "ImageFilter variant count changed — update filter_counts array size and FILTERS in finish()"
);

/// Lightweight page metadata collected at zero extra cost during rendering.
///
/// Returned alongside the finished bitmap from [`PageRenderer::finish`].
/// Callers can use this to route pages to different OCR configurations without
/// a separate post-hoc analysis pass.
#[derive(Debug, Clone, Default)]
pub struct PageDiagnostics {
    /// `true` when at least one image `XObject` or inline image was blitted.
    pub has_images: bool,
    /// `true` when at least one text-showing operator (`Tj`, `TJ`, `'`, `"`)
    /// was executed with a non-empty glyph sequence.  `false` on scan-only pages.
    pub has_vector_text: bool,
    /// The image filter seen most often on this page.
    ///
    /// `None` when no images were decoded (e.g. a pure-vector page).  When
    /// multiple filters are equally frequent the last one in [`ImageFilter`]
    /// discriminant order wins (Rust's `max_by_key` is last-wins on ties) —
    /// stable across identical pages.
    pub dominant_filter: Option<ImageFilter>,
    /// Estimated source pixels-per-inch of the dominant image, if any.
    ///
    /// Computed as `(image_width_px / page_width_pts) × 72`.  Useful as a hint
    /// for DPI auto-selection: rendering at the source PPI avoids upsampling a
    /// 67-PPI scan at 300 DPI (4.5× upscale with no information gain).
    ///
    /// `None` when no images were blitted or page geometry is unavailable.
    pub source_ppi_hint: Option<f32>,
}

impl PageDiagnostics {
    /// Suggest a render DPI based on the source image resolution embedded in the PDF.
    ///
    /// Returns the source PPI rounded up to the nearest standard DPI step
    /// (72 / 96 / 150 / 200 / 300 / 400 / 600), clamped to `[min_dpi, max_dpi]`.
    ///
    /// Returns `None` when [`source_ppi_hint`](Self::source_ppi_hint) is absent
    /// (no images on the page), in which case the caller should use their default DPI.
    ///
    /// ## Rationale
    ///
    /// Many scanned PDFs store images at 62–67 PPI.  Rendering at 300 DPI upsamples
    /// 4.5× with no additional information.  Passing the suggested DPI to
    /// the renderer (`pdf_raster::RasterOptions::dpi`) renders at the minimum resolution that avoids
    /// upsampling, bounded by the caller's quality floor (`min_dpi`) and ceiling
    /// (`max_dpi`).
    ///
    /// Typical call: `diag.suggested_dpi(150.0, 300.0)` — never render below
    /// 150 DPI (Tesseract minimum for acceptable accuracy), never above 300 DPI.
    #[must_use]
    pub fn suggested_dpi(&self, min_dpi: f32, max_dpi: f32) -> Option<f32> {
        // Standard DPI steps in ascending order.  Any PPI above the last step
        // falls back to the last step (600), which is then clamped by max_dpi.
        const STEPS: [f32; 7] = [72.0, 96.0, 150.0, 200.0, 300.0, 400.0, 600.0];
        const STEPS_MAX: f32 = STEPS[STEPS.len() - 1]; // 600.0 — infallible const index
        let ppi = self.source_ppi_hint?;
        let stepped = STEPS
            .iter()
            .copied()
            .find(|&s| s >= ppi)
            .unwrap_or(STEPS_MAX);
        Some(stepped.clamp(min_dpi, max_dpi))
    }
}

/// Identity CTM for passing to raster functions — coordinate transform is
/// already baked into the path points by `to_device`.
const DEVICE_MATRIX: [f64; 6] = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];

/// Maximum nesting depth for Form `XObjects`.
///
/// Prevents unbounded recursion from self-referencing or mutually-referencing
/// form streams in malformed documents.
const MAX_FORM_DEPTH: u32 = 32;

/// Maximum tiling pattern tile dimension in device pixels.
///
/// A malformed PDF could specify `XStep`/`YStep` values large enough to OOM
/// the process.  This cap limits the tile to a manageable size regardless of
/// the pattern parameters.
const MAX_TILE_PX: f64 = 4096.0;

/// Maximum nesting depth for tiling patterns.
///
/// `render_pattern_tile` spawns a fresh `PageRenderer` for each tile, so the
/// `form_depth` Form-XObject guard doesn't apply.  Without an independent
/// cap, a self-referential or cyclic pattern chain (`scn /A f` inside
/// pattern A's own content stream) would recurse until stack overflow.
/// Real-world patterns nest 0–1 deep; 4 is a conservative bound that still
/// allows hatch-of-hatch decorative uses without supporting attacker-crafted
/// cycles.
const MAX_PATTERN_DEPTH: u32 = 4;

/// Renders a decoded operator sequence onto a `Bitmap<Rgb8>`.
pub struct PageRenderer<'doc> {
    /// Target pixel buffer.
    pub(super) bitmap: Bitmap<Rgb8>,
    /// Page width in device pixels.
    width: u32,
    /// Page height in device pixels.
    height: u32,
    /// Graphics state save/restore stack.
    pub(super) gstate: GStateStack,
    /// Current path under construction, if any.
    path: Option<PathBuilder>,
    /// Pending clip rule set by `W`/`W*` (PDF §8.5.4).
    ///
    /// `W`/`W*` do NOT consume the path — they set this flag so the next
    /// painting or `n` operator intersects the current path into the clip
    /// region before (or instead of) painting.  `false` = non-zero winding,
    /// `true` = even-odd.  Cleared by every path-terminating operator.
    pending_clip: Option<bool>,
    /// Font face cache for this page.
    font_cache: FontCache,
    /// Resource accessor for the current page or innermost form.
    pub(super) resources: PageResources<'doc>,
    /// Current Form `XObject` nesting depth (0 = top-level page).
    form_depth: u32,
    /// Current tiling-pattern nesting depth (0 = top-level page).  Spawned
    /// tile renderers carry this incremented; capped at `MAX_PATTERN_DEPTH`
    /// to break self-referential pattern cycles (see `resolve_fill_pattern`).
    pub(super) pattern_depth: u32,
    /// Accumulated page diagnostics, populated during rendering.
    diag: PageDiagnostics,
    /// Per-filter blit counts used to compute `diag.dominant_filter`.
    filter_counts: [u32; IMAGE_FILTER_COUNT],
    /// Optional Content Group (OCG) visibility stack.
    ///
    /// Each `BDC /OC /Name` pushes a `bool` (`true` = visible); `EMC` pops it.
    /// Non-OCG `BDC`/`EMC` pairs do not touch this stack (they are `MarkedContent`
    /// no-ops).  Content is skipped whenever any entry is `false`.
    ocg_stack: Vec<bool>,
    /// GPU parallel-Huffman JPEG decoder via the CUDA backend.
    /// `None` means the path is inactive.
    #[cfg(feature = "gpu-jpeg-huffman")]
    jpeg_gpu: Option<JpegGpuDecoder<CudaBackend>>,
    /// GPU parallel-Huffman JPEG decoder via the Vulkan backend.
    /// Mutually exclusive with `jpeg_gpu` — only one is `Some` per render call.
    #[cfg(all(feature = "gpu-jpeg-huffman", feature = "vulkan"))]
    jpeg_vk: Option<JpegGpuDecoder<VulkanBackend>>,
    /// GPU-accelerated JPEG decoder, present when the `nvjpeg` feature is enabled
    /// and a CUDA device is available.  `None` means CPU-only JPEG decode.
    #[cfg(feature = "nvjpeg")]
    nvjpeg: Option<NvJpegDecoder>,
    /// VA-API JPEG decode queue handle (AMD/Intel iGPU), present when the
    /// `vaapi` feature is enabled and the DRM render node is accessible.
    ///
    /// A [`JpegQueueHandle`] is a cheaply-cloneable sender to a single OS
    /// thread that owns the `VapiJpegDecoder`.  All Rayon workers share the
    /// same worker thread via cloned handles, eliminating Mesa driver contention.
    #[cfg(feature = "vaapi")]
    vaapi_jpeg_queue: Option<JpegQueueHandle>,
    /// GPU-accelerated JPEG 2000 decoder, present when the `nvjpeg2k` feature is
    /// enabled and a CUDA device is available.  `None` means CPU-only JPX decode.
    #[cfg(feature = "nvjpeg2k")]
    nvjpeg2k: Option<NvJpeg2kDecoder>,
    /// Shared GPU context for AA fill dispatch and ICC CMYK→RGB colour conversion.
    /// Present when `gpu-aa` or `gpu-icc` features are enabled.
    #[cfg(any(feature = "gpu-aa", feature = "gpu-icc", feature = "cache"))]
    gpu_ctx: Option<Arc<GpuCtx>>,
    /// Vulkan compute backend, attached when the session was opened with
    /// [`BackendPolicy::ForceVulkan`].  When set, [`gpu_ops::try_gpu_aa_fill`]
    /// / `try_gpu_tile_fill` and the ICC CMYK→RGB path prefer the Vulkan
    /// trait surface (`alloc → upload → record_* → submit → wait → download`)
    /// over the CUDA `GpuCtx` path.  Mutually exclusive with `gpu_ctx` in
    /// practice — Phase 9 cache is CUDA-only, so opening a session under
    /// `ForceVulkan` skips CUDA init entirely (`pdf_raster::render::open_session`).
    /// The image-blit path (`DevicePageBuffer`) stays CUDA-only and is gated
    /// behind `gpu_ctx`, so under Vulkan blits silently land on the CPU
    /// fallback inside the renderer.
    #[cfg(feature = "vulkan")]
    vk_backend: Option<Arc<VulkanBackend>>,
    /// Per-page cache of baked ICC CMYK→RGB CLUT tables.  Keyed by a hash of the
    /// raw ICC profile bytes so that repeated images sharing the same profile (the
    /// common case in press PDFs) only pay the bake cost once per page.
    #[cfg(feature = "gpu-icc")]
    icc_clut_cache: crate::resources::image::IccClutCache,
    /// Phase 9 cache state.  `Some` only when the `cache` feature is
    /// on AND a cache was wired in via [`Self::set_image_cache`];
    /// `None` keeps the renderer fully CPU-resident.
    #[cfg(feature = "cache")]
    cache_state: Option<CacheState>,
}

/// Phase 9 renderer-side state bundled together so the "cache wired
/// in" condition is a single `Option<CacheState>` rather than three
/// implicit invariants on separate fields.
///
/// Without this bundling, `doc_id: DocId` had to default to
/// `DocId([0u8; 32])` — a valid hash that would collide with anyone
/// using the all-zeros doc id, a sharp foot-gun.
#[cfg(feature = "cache")]
struct CacheState {
    /// Phase 9 device-resident image cache.  Shared across pages so
    /// content-hash dedup spans the whole render session.
    cache: Arc<DeviceImageCache>,
    /// Stable identifier for the source PDF — combined with the
    /// per-image PDF object id, forms the cache's `(DocId, ObjId)`
    /// alias key for fast same-document lookups.
    doc_id: DocId,
    /// Lazy-allocated device-resident page buffer.  `None` until the
    /// first GPU image blit allocates it; non-None pages download +
    /// alpha-composite onto `bitmap` at `PageRenderer::finish`.
    page_buffer: Option<DevicePageBuffer>,
}

impl<'doc> PageRenderer<'doc> {
    /// Create a renderer for a blank white page of `width × height` pixels,
    /// where 1 user-space unit = 1 device pixel (72 dpi).
    ///
    /// `doc` and `page_id` are used to resolve font and resource dictionaries.
    ///
    /// # Errors
    ///
    /// Returns [`InterpError::FontInit`] if `FreeType` cannot be initialised.
    pub fn new(
        width: u32,
        height: u32,
        doc: &'doc Document,
        page_id: ObjectId,
    ) -> Result<Self, InterpError> {
        Self::new_scaled(width, height, 1.0, 0, doc, page_id)
    }

    /// Create a renderer with an initial uniform scale in the CTM.
    ///
    /// `scale = dpi / 72.0` maps PDF points to device pixels.  `rotate_cw` is
    /// the page `/Rotate` value (one of 0, 90, 180, 270); the bitmap dimensions
    /// must already be swapped for 90°/270° before calling this method.
    ///
    /// # Panics
    ///
    /// Panics if `scale` is not a positive finite number.
    ///
    /// # Errors
    ///
    /// Returns [`InterpError::FontInit`] if `FreeType` cannot be initialised.
    pub fn new_scaled(
        width: u32,
        height: u32,
        scale: f64,
        rotate_cw: u16,
        doc: &'doc Document,
        page_id: ObjectId,
    ) -> Result<Self, InterpError> {
        assert!(
            scale.is_finite() && scale > 0.0,
            "PageRenderer::new_scaled: scale must be a positive finite number, got {scale}"
        );

        let mut bitmap = Bitmap::<Rgb8>::new(width, height, 1, false);
        bitmap.data_mut().fill(255u8); // white background

        let mut gstate = GStateStack::new(width, height);
        // Build the initial CTM combining scale with the page rotation.
        // The `to_device` helper applies an additional Y-flip (`height - dy`) so
        // the formulas below account for that flip.  `w` and `h` are the output
        // bitmap dimensions in points (`width_px / scale`, `height_px / scale`).
        let w = f64::from(width) / scale;
        let h = f64::from(height) / scale;
        let ctm: [f64; 6] = match rotate_cw % 360 {
            // Rotate 0: standard scale + y-flip handled by to_device.
            0 => [scale, 0.0, 0.0, scale, 0.0, 0.0],
            // Rotate 90 CW: swap axes. Width of bitmap = original H, height = original W.
            90 => [0.0, -scale, -scale, 0.0, h * scale, w * scale],
            // Rotate 180: flip both axes.
            180 => [-scale, 0.0, 0.0, scale, w * scale, 0.0],
            // Rotate 270 CW (= 90 CCW): swap axes, opposite orientation.
            _ => [0.0, scale, scale, 0.0, 0.0, 0.0],
        };
        gstate.current_mut().ctm = ctm;

        let engine: SharedEngine = FontEngine::init(true, true, false)
            .map_err(|e| InterpError::FontInit(e.to_string()))?;
        let glyph_cache = GlyphCache::new();
        let font_cache = FontCache::new(engine, glyph_cache);
        let resources = PageResources::new(doc, page_id);

        Ok(Self {
            bitmap,
            width,
            height,
            gstate,
            path: None,
            pending_clip: None,
            font_cache,
            resources,
            form_depth: 0,
            pattern_depth: 0,
            diag: PageDiagnostics::default(),
            filter_counts: [0u32; IMAGE_FILTER_COUNT],
            ocg_stack: Vec::new(),
            #[cfg(feature = "gpu-jpeg-huffman")]
            jpeg_gpu: None,
            #[cfg(all(feature = "gpu-jpeg-huffman", feature = "vulkan"))]
            jpeg_vk: None,
            #[cfg(feature = "nvjpeg")]
            nvjpeg: None,
            #[cfg(feature = "vaapi")]
            vaapi_jpeg_queue: None,
            #[cfg(feature = "nvjpeg2k")]
            nvjpeg2k: None,
            #[cfg(any(feature = "gpu-aa", feature = "gpu-icc", feature = "cache"))]
            gpu_ctx: None,
            #[cfg(feature = "vulkan")]
            vk_backend: None,
            #[cfg(feature = "gpu-icc")]
            icc_clut_cache: crate::resources::image::IccClutCache::new(),
            #[cfg(feature = "cache")]
            cache_state: None,
        })
    }

    /// Attach a GPU parallel-Huffman JPEG decoder to this renderer.
    #[cfg(feature = "gpu-jpeg-huffman")]
    pub fn set_jpeg_gpu(&mut self, dec: Option<JpegGpuDecoder<CudaBackend>>) {
        self.jpeg_gpu = dec;
    }

    /// Detach and return the GPU parallel-Huffman JPEG decoder for reuse.
    #[cfg(feature = "gpu-jpeg-huffman")]
    pub const fn take_jpeg_gpu(&mut self) -> Option<JpegGpuDecoder<CudaBackend>> {
        self.jpeg_gpu.take()
    }

    /// Attach a Vulkan parallel-Huffman JPEG decoder to this renderer.
    ///
    /// Mutually exclusive with [`set_jpeg_gpu`] — call one or the other per
    /// render, never both.
    #[cfg(all(feature = "gpu-jpeg-huffman", feature = "vulkan"))]
    pub fn set_jpeg_vk(&mut self, dec: Option<JpegGpuDecoder<VulkanBackend>>) {
        self.jpeg_vk = dec;
    }

    /// Detach and return the Vulkan parallel-Huffman JPEG decoder for reuse.
    #[cfg(all(feature = "gpu-jpeg-huffman", feature = "vulkan"))]
    pub const fn take_jpeg_vk(&mut self) -> Option<JpegGpuDecoder<VulkanBackend>> {
        self.jpeg_vk.take()
    }

    /// [`crate::resources::image::GPU_JPEG_THRESHOLD_PX`] are decoded on the
    /// GPU via nvJPEG rather than the CPU JPEG decoder.
    ///
    /// Calling this with `None` detaches any existing decoder (reverts to CPU).
    #[cfg(feature = "nvjpeg")]
    pub fn set_nvjpeg(&mut self, dec: Option<NvJpegDecoder>) {
        self.nvjpeg = dec;
    }

    /// Attach a VA-API JPEG decode queue handle to this renderer.
    ///
    /// The handle is a cheaply-cloneable sender to a dedicated OS worker thread
    /// that owns the `VapiJpegDecoder`.  `DCTDecode` baseline JPEG streams with
    /// pixel area ≥ [`crate::resources::image::GPU_JPEG_THRESHOLD_PX`] are
    /// submitted to the queue instead of being decoded on the calling thread,
    /// eliminating Mesa driver contention across Rayon workers.
    ///
    /// The handle is dropped when the renderer drops — no explicit reclaim step
    /// is needed because the queue worker lives in `RasterSession`, which always
    /// outlives the renderer.
    #[cfg(feature = "vaapi")]
    pub fn set_vaapi_queue(&mut self, handle: JpegQueueHandle) {
        self.vaapi_jpeg_queue = Some(handle);
    }

    /// Attach a GPU JPEG 2000 decoder to this renderer.
    ///
    /// When set, `JPXDecode` image streams with pixel area ≥
    /// [`crate::resources::image::GPU_JPEG2K_THRESHOLD_PX`] are decoded on the
    /// GPU via nvJPEG2000 rather than the CPU JPEG 2000 decoder.
    ///
    /// Call with `None` to revert to CPU-only JPEG 2000 decode.
    #[cfg(feature = "nvjpeg2k")]
    pub fn set_nvjpeg2k(&mut self, dec: Option<NvJpeg2kDecoder>) {
        self.nvjpeg2k = dec;
    }

    /// Detach and return the nvJPEG decoder so the caller can reuse it.
    ///
    /// Returns `None` if no decoder was attached (e.g. GPU init failed).
    /// Used by the CLI to return the decoder to its thread-local slot
    /// after each page render so it survives across pages.
    #[cfg(feature = "nvjpeg")]
    pub const fn take_nvjpeg(&mut self) -> Option<NvJpegDecoder> {
        self.nvjpeg.take()
    }

    /// Detach and return the nvJPEG2000 decoder so the caller can reuse it.
    ///
    /// Returns `None` if no decoder was attached (e.g. GPU init failed).
    /// Used by the CLI to return the decoder to its thread-local slot
    /// after each page render so it survives across pages.
    #[cfg(feature = "nvjpeg2k")]
    pub const fn take_nvjpeg2k(&mut self) -> Option<NvJpeg2kDecoder> {
        self.nvjpeg2k.take()
    }

    /// Attach a GPU context for supersampled AA fill dispatch.
    ///
    /// When set, filled paths with a pixel bounding-box area above
    /// [`gpu::GPU_AA_FILL_THRESHOLD`] are rasterised on the GPU using a 64-sample
    /// jittered warp-ballot kernel instead of the CPU 4× scanline AA path.
    ///
    /// Call with `None` to revert to CPU-only AA.
    #[cfg(any(feature = "gpu-aa", feature = "gpu-icc", feature = "cache"))]
    pub fn set_gpu_ctx(&mut self, ctx: Option<Arc<GpuCtx>>) {
        self.gpu_ctx = ctx;
    }

    /// Attach a Vulkan compute backend for kernel dispatch.
    ///
    /// When set, GPU-eligible AA fill, tile fill, and ICC CMYK→RGB calls
    /// are routed through the [`gpu::backend::GpuBackend`] trait surface
    /// (`alloc → upload → record_* → submit → wait → download`) on the
    /// Vulkan recorder rather than the CUDA `GpuCtx`.  The CUDA path is
    /// not removed; it remains live for sessions opened with
    /// `BackendPolicy::Auto` / `BackendPolicy::ForceCuda` (defined in `pdf_raster`).
    ///
    /// The Phase 9 device-resident image cache (`DeviceImageCache`,
    /// `DevicePageBuffer`) is CUDA-only — under Vulkan the renderer
    /// runs uncached, matching Phase 9-pre-2026-05-07 behaviour.
    #[cfg(feature = "vulkan")]
    pub fn set_vk_backend(&mut self, backend: Option<Arc<VulkanBackend>>) {
        self.vk_backend = backend;
    }

    /// Attach a Phase 9 device-resident image cache.
    ///
    /// When set, JPEG image decode goes through the cache: a content-
    /// hash hit returns an `Arc<CachedDeviceImage>` with no decode
    /// work; a miss decodes on CPU, uploads to VRAM, caches, and
    /// returns the same handle.  The renderer dispatches `Gpu`-variant
    /// images to a CUDA blit kernel that writes into a per-page
    /// `DevicePageBuffer`; CPU-rasterised vector content stays on
    /// `bitmap`, and the two are alpha-composited at [`Self::finish`].
    ///
    /// `doc_id` should be a stable identifier for the source PDF
    /// (typically `BLAKE3(pdf_bytes)` or similar).  It's combined
    /// with the per-image PDF object number to form the cache's
    /// secondary alias key for fast same-document lookups.
    ///
    /// Call with `None` to revert to CPU-only image decode.
    ///
    /// **Caller contract:** call before any rendering operators
    /// execute on this renderer.  Wiring or unwiring a cache
    /// mid-page would discard any GPU-blit pixels written so far
    /// (the per-page `DevicePageBuffer` is dropped on reset);
    /// today no caller does this.
    #[cfg(feature = "cache")]
    pub fn set_image_cache(&mut self, cache: Option<Arc<DeviceImageCache>>, doc_id: DocId) {
        self.cache_state = cache.map(|cache| CacheState {
            cache,
            doc_id,
            page_buffer: None,
        });
    }

    /// Consume the renderer and return the finished bitmap and page diagnostics.
    #[must_use]
    pub fn finish(mut self) -> (Bitmap<Rgb8>, PageDiagnostics) {
        // Resolve dominant_filter from per-filter blit counts.
        // Order matches the ImageFilter discriminants (Dct=0, Jpx=1, CcittFax=2, Jbig2=3, Flate=4, Raw=5).
        const FILTERS: [ImageFilter; 6] = [
            ImageFilter::Dct,
            ImageFilter::Jpx,
            ImageFilter::CcittFax,
            ImageFilter::Jbig2,
            ImageFilter::Flate,
            ImageFilter::Raw,
        ];
        self.diag.dominant_filter = self
            .filter_counts
            .iter()
            .zip(FILTERS.iter())
            .filter(|(count, _)| **count > 0)
            .max_by_key(|(count, _)| *count)
            .map(|(_, filter)| *filter);

        // Phase 9: download the device-resident image-blit buffer (if
        // any GPU image-blit ran on this page) and source-over
        // composite it onto the host bitmap.  Pixels the kernel
        // didn't write read back as alpha=0 → no-op for the
        // composite, leaving the CPU-rasterised content untouched.
        #[cfg(feature = "cache")]
        if let Some(buf) = self
            .cache_state
            .as_mut()
            .and_then(|cs| cs.page_buffer.take())
        {
            self.composite_device_page_buffer(&buf);
        }

        (self.bitmap, self.diag)
    }

    /// Download the device page buffer and alpha-composite it onto
    /// `self.bitmap`.  Source-over: written GPU pixels (alpha=255)
    /// replace; unwritten pixels (alpha=0) leave the bitmap intact.
    /// This is the moment where the blit kernel's writes become
    /// visible on host.
    #[cfg(feature = "cache")]
    fn composite_device_page_buffer(&mut self, buf: &DevicePageBuffer) {
        use gpu::cache::RGBA_BPP;

        let host_rgba = match buf.download() {
            Ok(bytes) => bytes,
            Err(e) => {
                log::warn!(
                    "finish: device page buffer download failed: {e} — GPU images will be missing"
                );
                return;
            }
        };
        let dst_stride = self.width as usize * 3;
        let src_stride = self.width as usize * RGBA_BPP;
        // Sanity: degrade gracefully on a dimension mismatch rather
        // than panic-aborting on slice-out-of-bounds in release.
        // Today only `try_gpu_blit_image` constructs `buf` (with
        // `self.width` / `self.height`), so a mismatch indicates a
        // future bug — log loudly and skip the composite.
        let expected = src_stride * self.height as usize;
        if host_rgba.len() != expected {
            log::warn!(
                "finish: device page buffer size {} != expected {expected} ({}×{}×{RGBA_BPP}); skipping composite",
                host_rgba.len(),
                self.width,
                self.height,
            );
            return;
        }
        let dst = self.bitmap.data_mut();
        for y in 0..self.height as usize {
            let src_row = &host_rgba[y * src_stride..(y + 1) * src_stride];
            let dst_row = &mut dst[y * dst_stride..(y + 1) * dst_stride];
            for x in 0..self.width as usize {
                let s = &src_row[x * RGBA_BPP..x * RGBA_BPP + 4];
                if s[3] == 0 {
                    continue;
                }
                // Source-over with src.a == 255 in the kernel's
                // current implementation; opaque copy.  When/if
                // the kernel grows partial-alpha output, switch to
                // a full Porter-Duff compositor here.
                let d = &mut dst_row[x * 3..x * 3 + 3];
                d[0] = s[0];
                d[1] = s[1];
                d[2] = s[2];
            }
        }
    }

    /// Execute a slice of decoded operators in order.
    pub fn execute(&mut self, ops: &[Operator]) {
        for op in ops {
            self.execute_one(op);
        }
    }

    /// Render all annotations on the page (PDF §12.5).
    ///
    /// Each annotation with an `AP/N` (normal appearance) stream is rendered
    /// after the page content stream.  Only the `/N` entry is used; rollover
    /// (`/R`) and down (`/D`) appearances are ignored (this is a rasterizer,
    /// not an interactive viewer).
    ///
    /// Annotations with no appearance stream are silently skipped — many
    /// annotation types (e.g. `Link`) have no visible appearance by default.
    ///
    /// Call this after [`execute`](Self::execute) and before [`finish`](Self::finish).
    pub fn render_annotations(&mut self, page_id: ObjectId) {
        annotations::render_annotations(self, page_id);
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    fn set_fill(&mut self, c: RasterColor) {
        let gs = self.gstate.current_mut();
        gs.fill_color = c;
        gs.fill_pattern = None;
    }

    fn set_stroke(&mut self, c: RasterColor) {
        let gs = self.gstate.current_mut();
        gs.stroke_color = c;
        gs.stroke_pattern = None;
    }

    fn path_builder(&mut self) -> &mut PathBuilder {
        self.path.get_or_insert_with(PathBuilder::new)
    }

    /// Transform a user-space point to device space using the current CTM.
    /// Applies a y-flip because PDF uses bottom-left origin but bitmaps use top-left.
    fn to_device(&self, x: f64, y: f64) -> (f64, f64) {
        let ctm = &self.gstate.current().ctm;
        let (dx, dy) = ctm_transform(ctm, x, y);
        (dx, f64::from(self.height) - dy)
    }

    /// Apply a pending `W`/`W*` clip using the current path, then discard the path.
    ///
    /// Called by `EndPath` (`n`) when no painting follows.
    fn apply_pending_clip(&mut self) {
        if let Some(even_odd) = self.pending_clip.take() {
            if let Some(builder) = self.path.take() {
                self.clip_path_into_gstate(&builder.build(), even_odd);
            } else {
                log::debug!("pdf_interp: W/W* with no current path — clip unchanged");
            }
        }
    }

    /// Apply a pending `W`/`W*` clip using an already-built path.
    ///
    /// Called by paint helpers (`do_fill`, `do_stroke`, `do_fill_then_stroke`)
    /// when a `W`/`W*` preceded the painting operator.  The same path is used
    /// for both clipping and painting (PDF §8.5.4).
    fn apply_pending_clip_with(&mut self, path: &raster::path::Path) {
        if let Some(even_odd) = self.pending_clip.take() {
            self.clip_path_into_gstate(path, even_odd);
        }
    }

    /// Intersect `path` into the current graphics-state clip region.
    fn clip_path_into_gstate(&mut self, path: &raster::path::Path, even_odd: bool) {
        let flatness = self.gstate.current().flatness.max(0.1);
        let xpath = XPath::new(path, &DEVICE_MATRIX, flatness, true);
        self.gstate
            .current_mut()
            .clip
            .clip_to_path(&xpath, even_odd);
    }

    /// Build a [`PipeState`] with the given opacity and blend mode.
    fn make_pipe(a_input: u8, blend_mode: BlendMode) -> PipeState<'static> {
        PipeState {
            blend_mode,
            a_input,
            overprint_mask: 0xFFFF_FFFF,
            overprint_additive: false,
            transfer: TransferSet::identity_rgb(),
            soft_mask: None,
            alpha0: None,
            knockout: false,
            knockout_opacity: 255,
            non_isolated_group: false,
        }
    }

    fn do_fill(&mut self, even_odd: bool) {
        let Some(builder) = self.path.take() else {
            self.pending_clip = None;
            return;
        };
        let path = builder.build();
        self.apply_pending_clip_with(&path);
        self.fill_path(&path, even_odd);
    }

    fn do_stroke(&mut self) {
        let Some(builder) = self.path.take() else {
            self.pending_clip = None;
            return;
        };
        let path = builder.build();
        self.apply_pending_clip_with(&path);
        self.stroke_path(&path);
    }

    fn do_fill_then_stroke(&mut self, even_odd: bool) {
        let Some(builder) = self.path.take() else {
            self.pending_clip = None;
            return;
        };
        let path = builder.build();
        self.apply_pending_clip_with(&path);
        self.fill_path(&path, even_odd);
        self.stroke_path(&path);
    }

    /// Fill `path` using the current fill colour or pattern.  Shared by `do_fill`
    /// and `do_fill_then_stroke`.
    fn fill_path(&mut self, path: &raster::path::Path, even_odd: bool) {
        let gs = self.gstate.current();
        let clip = gs.clip.clone_shared();
        let flatness = gs.flatness.max(0.1);
        let pipe = Self::make_pipe(gs.fill_alpha, gs.blend_mode);

        // Resolve the fill source — pattern or solid colour.
        let pat_name = gs.fill_pattern.clone();
        let solid_color = gs.fill_color.as_slice().to_vec();
        let _ = gs; // end immutable borrow before the mutable resolve_fill_pattern call

        let tiled = pat_name
            .as_deref()
            .and_then(|name| self.resolve_fill_pattern(name));
        let src = tiled.as_ref().map_or(PipeSrc::Solid(&solid_color), |p| {
            PipeSrc::Pattern(p as &dyn Pattern)
        });

        // GPU fill dispatch — solid-colour fills only (patterns not yet
        // GPU-accelerated).  Cloning the `Arc` releases the immutable borrow
        // before the mutable `self` borrow inside the helpers.
        //
        // Dispatch order per backend: tile-parallel analytical (large fills)
        // → warp-ballot 64-sample AA (medium fills) → CPU scanline AA
        // (always-available fallback).  The Vulkan branch fires before the
        // CUDA branch when both are present, but in practice
        // `pdf_raster::render::open_session` wires only one of them based
        // on `BackendPolicy`, so the second branch only runs under
        // `Auto` / `ForceCuda`.
        #[cfg(feature = "vulkan")]
        if tiled.is_none()
            && let Some(vk) = self.vk_backend.clone()
        {
            if self.try_vk_tile_fill(path, even_odd, &pipe, &src, &vk) {
                return;
            }
            if self.try_vk_aa_fill(path, even_odd, &pipe, &src, &vk) {
                return;
            }
        }
        #[cfg(feature = "gpu-aa")]
        if tiled.is_none()
            && let Some(ctx) = self.gpu_ctx.clone()
        {
            if self.try_gpu_tile_fill(path, even_odd, &pipe, &src, &ctx) {
                return;
            }
            if self.try_gpu_aa_fill(path, even_odd, &pipe, &src, &ctx) {
                return;
            }
        }

        if even_odd {
            eo_fill(
                &mut self.bitmap,
                &clip,
                path,
                &pipe,
                &src,
                &DEVICE_MATRIX,
                flatness,
                true,
            );
        } else {
            fill(
                &mut self.bitmap,
                &clip,
                path,
                &pipe,
                &src,
                &DEVICE_MATRIX,
                flatness,
                true,
            );
        }
        drop(tiled);
    }

    /// Attempt to rasterise `path` with the GPU 64-sample AA kernel.
    ///
    /// Returns `true` if the GPU path was taken (caller skips the CPU fill).
    /// Returns `false` if the area is below the dispatch threshold, the segment
    /// list is empty, the bbox is non-finite, or the GPU call fails (warning
    /// logged; CPU fill used as fallback).
    #[cfg(feature = "gpu-aa")]
    fn try_gpu_aa_fill(
        &mut self,
        path: &raster::path::Path,
        even_odd: bool,
        pipe: &raster::pipe::PipeState<'_>,
        src: &raster::pipe::PipeSrc<'_>,
        ctx: &gpu::GpuCtx,
    ) -> bool {
        gpu_ops::try_gpu_aa_fill(self, path, even_odd, pipe, src, ctx)
    }

    /// Attempt to rasterise `path` with the GPU tile-parallel analytical fill kernel.
    ///
    /// Returns `true` if the GPU path was taken (caller skips the CPU fill).
    /// Returns `false` if the area is below the dispatch threshold, the segment
    /// list is empty, the bbox is non-finite, or the GPU call fails (warning
    /// logged; caller falls through to AA or CPU fill).
    #[cfg(feature = "gpu-aa")]
    fn try_gpu_tile_fill(
        &mut self,
        path: &raster::path::Path,
        even_odd: bool,
        pipe: &raster::pipe::PipeState<'_>,
        src: &raster::pipe::PipeSrc<'_>,
        ctx: &gpu::GpuCtx,
    ) -> bool {
        gpu_ops::try_gpu_tile_fill(self, path, even_odd, pipe, src, ctx)
    }

    /// Vulkan twin of [`Self::try_gpu_aa_fill`].  See [`vk_ops::try_vk_aa_fill`].
    #[cfg(feature = "vulkan")]
    fn try_vk_aa_fill(
        &mut self,
        path: &raster::path::Path,
        even_odd: bool,
        pipe: &raster::pipe::PipeState<'_>,
        src: &raster::pipe::PipeSrc<'_>,
        backend: &Arc<VulkanBackend>,
    ) -> bool {
        vk_ops::try_vk_aa_fill(self, backend, path, even_odd, pipe, src)
    }

    /// Vulkan twin of [`Self::try_gpu_tile_fill`].  See [`vk_ops::try_vk_tile_fill`].
    #[cfg(feature = "vulkan")]
    fn try_vk_tile_fill(
        &mut self,
        path: &raster::path::Path,
        even_odd: bool,
        pipe: &raster::pipe::PipeState<'_>,
        src: &raster::pipe::PipeSrc<'_>,
        backend: &Arc<VulkanBackend>,
    ) -> bool {
        vk_ops::try_vk_tile_fill(self, backend, path, even_odd, pipe, src)
    }

    fn stroke_path(&mut self, path: &raster::path::Path) {
        // Clone all graphics state data before any mutable borrow.
        let gs = self.gstate.current();
        let clip = gs.clip.clone_shared();
        let pipe = Self::make_pipe(gs.stroke_alpha, gs.blend_mode);
        let line_width = gs.line_width;
        let line_cap = gs.line_cap;
        let line_join = gs.line_join;
        let miter_limit = gs.miter_limit;
        let flatness = gs.flatness.max(0.1);
        let line_dash = gs.dash.0.clone();
        let line_dash_phase = gs.dash.1;
        let pat_name = gs.stroke_pattern.clone();
        let solid_color = gs.stroke_color.as_slice().to_vec();
        // gs immutable borrow ends here.

        let params = StrokeParams {
            line_width,
            line_cap,
            line_join,
            miter_limit,
            flatness,
            line_dash: &line_dash,
            line_dash_phase,
            stroke_adjust: false,
            vector_antialias: true,
        };

        // Resolve the stroke source — pattern or solid colour.
        let tiled = pat_name
            .as_deref()
            .and_then(|name| self.resolve_fill_pattern(name));
        let src = tiled.as_ref().map_or(PipeSrc::Solid(&solid_color), |p| {
            PipeSrc::Pattern(p as &dyn Pattern)
        });

        stroke(
            &mut self.bitmap,
            &clip,
            path,
            &pipe,
            &src,
            &DEVICE_MATRIX,
            &params,
        );
        drop(tiled);
    }

    // ── XObject rendering ─────────────────────────────────────────────────────

    /// Execute a `Do` operator: look up and render the named `XObject`.
    ///
    /// Image `XObjects` are blitted directly.  Form `XObjects` are executed
    /// recursively via [`do_form_xobject`].
    fn do_xobject(&mut self, name: &[u8]) {
        // Try Form first (cheap dict lookup, no pixel decoding).
        if let Some(form) = self.resources.form_xobject(name) {
            self.do_form_xobject(&form);
            return;
        }
        // Try Image.
        // Route to whichever GPU JPEG decoder is active (only one is Some per render).
        // Each branch is a separately monomorphised call; no decoder is wasted.
        #[cfg(all(feature = "gpu-jpeg-huffman", feature = "vulkan"))]
        if self.jpeg_vk.is_some() {
            let img = self.resources.image(
                name,
                #[cfg(feature = "nvjpeg")]
                self.nvjpeg.as_mut(),
                #[cfg(feature = "vaapi")]
                self.vaapi_jpeg_queue.as_ref(),
                #[cfg(feature = "nvjpeg2k")]
                self.nvjpeg2k.as_mut(),
                self.jpeg_vk.as_mut(),
                #[cfg(feature = "gpu-icc")]
                self.gpu_ctx.as_deref(),
                #[cfg(feature = "gpu-icc")]
                Some(&mut self.icc_clut_cache),
                #[cfg(feature = "cache")]
                self.cache_state.as_ref().map(|cs| &cs.cache),
                #[cfg(feature = "cache")]
                self.cache_state.as_ref().map(|cs| cs.doc_id),
            );
            if let Some(img) = img {
                self.blit_image(&img);
                return;
            }
            log::debug!(
                "pdf_interp: Do /{} skipped (unsupported filter or missing resource)",
                String::from_utf8_lossy(name)
            );
            return;
        }
        let img = self.resources.image(
            name,
            #[cfg(feature = "nvjpeg")]
            self.nvjpeg.as_mut(),
            #[cfg(feature = "vaapi")]
            self.vaapi_jpeg_queue.as_ref(),
            #[cfg(feature = "nvjpeg2k")]
            self.nvjpeg2k.as_mut(),
            #[cfg(feature = "gpu-jpeg-huffman")]
            self.jpeg_gpu.as_mut(),
            #[cfg(feature = "gpu-icc")]
            self.gpu_ctx.as_deref(),
            #[cfg(feature = "gpu-icc")]
            Some(&mut self.icc_clut_cache),
            #[cfg(feature = "cache")]
            self.cache_state.as_ref().map(|cs| &cs.cache),
            #[cfg(feature = "cache")]
            self.cache_state.as_ref().map(|cs| cs.doc_id),
        );
        if let Some(img) = img {
            self.blit_image(&img);
            return;
        }
        // Missing resource or unsupported filter.
        log::debug!(
            "pdf_interp: Do /{} skipped (unsupported filter or missing resource)",
            String::from_utf8_lossy(name)
        );
    }

    /// Execute an `sh` shading operator.
    ///
    /// Looks up the named shading resource and dispatches to either
    /// `shaded_fill` (for smooth gradient types 2–3) or `gouraud_triangle_fill`
    /// (for mesh types 4–5).
    fn do_shading(&mut self, name: &[u8]) {
        use crate::resources::shading::ShadingResult;

        let gs = self.gstate.current();
        let ctm = gs.ctm;
        let clip = gs.clip.clone_shared();
        let flatness = gs.flatness.max(0.1);
        let alpha = gs.fill_alpha;
        let blend_mode = gs.blend_mode;
        let page_h = f64::from(self.height);

        let Some(result) = self.resources.shading(name, &ctm, page_h) else {
            log::warn!(
                "pdf_interp: sh /{} — shading not available (unsupported type or missing resource)",
                String::from_utf8_lossy(name)
            );
            return;
        };

        let pipe = Self::make_pipe(alpha, blend_mode);

        match result {
            ShadingResult::Pattern(pattern, _bbox) => {
                // Build a path covering the full clip bounding box.
                let path = {
                    let mut pb = PathBuilder::new();
                    let _ = pb.move_to(clip.x_min, clip.y_min);
                    let _ = pb.line_to(clip.x_max, clip.y_min);
                    let _ = pb.line_to(clip.x_max, clip.y_max);
                    let _ = pb.line_to(clip.x_min, clip.y_max);
                    let _ = pb.close(true);
                    pb.build()
                };
                shaded_fill::<color::Rgb8>(
                    &mut self.bitmap,
                    &clip,
                    &path,
                    &pipe,
                    pattern.as_ref(),
                    &DEVICE_MATRIX,
                    flatness,
                    true,
                    false,
                );
            }
            ShadingResult::Mesh(triangles) => {
                for tri in triangles {
                    gouraud_triangle_fill::<color::Rgb8>(&mut self.bitmap, &clip, &pipe, tri);
                }
            }
        }
    }

    /// Execute a Form `XObject`'s content stream in the current graphics context.
    ///
    /// PDF §8.10.1: the form's `Matrix` is concatenated onto the current CTM,
    /// graphics state is saved/restored around execution, and the form's own
    /// `Resources` dict (if present) is used to resolve fonts and images inside
    /// the form.
    ///
    /// When the form carries a `Group /S /Transparency` entry, content is rendered
    /// into an intermediate group bitmap and composited back (PDF §11.6.6).
    fn do_form_xobject(&mut self, form: &crate::resources::FormXObject) {
        if self.form_depth >= MAX_FORM_DEPTH {
            log::warn!(
                "pdf_interp: Form XObject nesting depth {MAX_FORM_DEPTH} exceeded — skipping"
            );
            return;
        }

        // Capture the parent's compositing state before saving, so the group is
        // painted using the opacity/blend that was active when the form was invoked
        // (PDF §11.6.6: the group is composited using the surrounding context).
        let parent_fill_alpha = self.gstate.current().fill_alpha;
        let parent_blend_mode = self.gstate.current().blend_mode;

        // Save graphics state (equivalent to `q`).
        self.gstate.save();
        self.form_depth += 1;

        // Concatenate the form's Matrix onto the current CTM.
        let old_ctm = self.gstate.current().ctm;
        self.gstate.current_mut().ctm = ctm_multiply(&old_ctm, &form.matrix);

        // Switch to the form's resource context, keeping the parent for restore.
        let child_resources = self.resources.for_form(form);
        let parent_resources = std::mem::replace(&mut self.resources, child_resources);

        // Transparency group: allocate an intermediate bitmap, render into it,
        // then composite back onto the page.
        let mut group = form.transparency.and_then(|tg| {
            // Map BBox corners through the new CTM to get the device-space extent.
            let ctm = self.gstate.current().ctm;
            let page_h = f64::from(self.height);
            let [bx0, by0, bx1, by1] = form.bbox;
            let corners = [
                ctm_transform(&ctm, bx0, by0),
                ctm_transform(&ctm, bx1, by0),
                ctm_transform(&ctm, bx0, by1),
                ctm_transform(&ctm, bx1, by1),
            ];
            // A non-finite BBox or degenerate CTM produces NaN corners; the
            // subsequent as-i32 cast would be UB on some platforms.  Fall back
            // to rendering without a group (paint directly onto the page).
            if corners
                .iter()
                .any(|(x, y)| !x.is_finite() || !y.is_finite())
            {
                log::warn!(
                    "pdf_interp: transparency group BBox is non-finite — rendering without group"
                );
                return None;
            }
            let left = corners
                .iter()
                .map(|(x, _)| *x)
                .fold(f64::INFINITY, f64::min)
                .floor();
            let top = corners
                .iter()
                .map(|(_, y)| page_h - y)
                .fold(f64::INFINITY, f64::min)
                .floor();
            let right = corners
                .iter()
                .map(|(x, _)| *x)
                .fold(f64::NEG_INFINITY, f64::max)
                .ceil();
            let bottom = corners
                .iter()
                .map(|(_, y)| page_h - y)
                .fold(f64::NEG_INFINITY, f64::max)
                .ceil();

            #[expect(
                clippy::cast_possible_truncation,
                reason = "f64→i32 cast saturates in Rust ≥ 1.45 (no UB); \
                          begin_group clamps GroupParams to bitmap bounds"
            )]
            let params = GroupParams {
                x_min: left as i32,
                y_min: top as i32,
                x_max: right as i32,
                y_max: bottom as i32,
                isolated: tg.isolated,
                knockout: tg.knockout,
                soft_mask_type: raster::transparency::SoftMaskType::None,
            };
            let clip = self.gstate.current().clip.clone_shared();
            Some(begin_group(&self.bitmap, &clip, params))
        });

        // Swap in the group bitmap when present.
        if let Some(ref mut g) = group {
            std::mem::swap(&mut self.bitmap, &mut g.bitmap);
        }

        // Parse and execute the form's content stream.
        let ops = crate::content::parse(&form.content);
        self.execute(&ops);

        // Swap the group bitmap back and composite using the parent's opacity/blend
        // (captured before gstate.save() — the form's own state is irrelevant here).
        if let Some(mut g) = group {
            std::mem::swap(&mut self.bitmap, &mut g.bitmap);
            let pipe = Self::make_pipe(parent_fill_alpha, parent_blend_mode);
            // Residual Clip is unused — parent gstate is restored immediately below.
            let _ = paint_group(&mut self.bitmap, g, &pipe);
        }

        // Restore resources and graphics state.
        self.resources = parent_resources;
        self.form_depth -= 1;
        self.gstate.restore();
    }

    /// Blit a decoded image `XObject` onto the bitmap using the current CTM.
    ///
    /// The PDF convention is that the CTM maps the unit square `[0,1]×[0,1]`
    /// (image space) to the target rectangle on the page.  We sample each
    /// output pixel by inverse-mapping it back to image space.
    ///
    /// For axis-aligned transforms (the common case) this is just a scaled copy.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "device pixel coords are always in page bounds after clamping; safe casts"
    )]
    #[expect(
        clippy::many_single_char_names,
        reason = "PDF CTM components a–f are standard"
    )]
    #[expect(
        clippy::similar_names,
        reason = "dx_rel / dy_rel are the standard names for the per-axis deltas in the inverse CTM formula"
    )]
    #[expect(
        clippy::too_many_lines,
        reason = "axis-aligned fast path + full inverse-CTM path + diagnostic tracking; splitting would scatter the CTM logic across multiple helpers with shared mutable state"
    )]
    fn blit_image(&mut self, img: &crate::resources::ImageDescriptor) {
        // Degenerate image — nothing to blit.
        if img.width == 0 || img.height == 0 {
            return;
        }
        // PDF §8.9.6: stencil masks (/ImageMask true → ImageColorSpace::Mask)
        // do not carry an /SMask.  The Mask arms below intentionally skip
        // the smask gate that the Rgb / Gray arms apply; this assert pins
        // the spec invariant so a future decoder change that sets `smask`
        // on a Mask descriptor surfaces in debug builds.
        debug_assert!(
            img.smask.is_none() || !matches!(img.color_space, ImageColorSpace::Mask),
            "PDF §8.9.6: image masks must not carry SMask",
        );

        // Track diagnostics: image presence and per-filter counts.
        self.diag.has_images = true;
        let filter_idx = img.filter as usize;
        self.filter_counts[filter_idx] = self.filter_counts[filter_idx].saturating_add(1);

        // Update source_ppi_hint: take the widest image seen (best resolution proxy).
        // ppi = (image_px_width / page_pts_width) × 72.
        let page_pts_w = f64::from(self.width) / {
            // For unrotated pages CTM[0] carries the scale; for 90°/270° rotations
            // CTM[0] is 0 and CTM[1] carries it.  Take the larger of the two to
            // handle all axis-aligned cases.  Falls back to 1.0 for a degenerate CTM
            // (page_pts_w would be width in pixels — wrong, but finite).
            let ctm = self.gstate.current().ctm;
            let ctm_scale = ctm[0].abs().max(ctm[1].abs());
            if ctm_scale > 0.0 { ctm_scale } else { 1.0 }
        };
        #[expect(
            clippy::cast_possible_truncation,
            reason = "src_ppi is finite and positive, checked below; f64→f32 is safe for PPI values in practice (well under f32::MAX)"
        )]
        let src_ppi = ((f64::from(img.width) / page_pts_w) * 72.0) as f32;
        if src_ppi.is_finite() && src_ppi > 0.0 {
            self.diag.source_ppi_hint = Some(match self.diag.source_ppi_hint {
                Some(prev) if prev >= src_ppi => prev,
                _ => src_ppi,
            });
        }

        let ctm = self.gstate.current().ctm;
        // Copy fill colour as [u8; 3] — RasterColor::as_slice always returns 3 bytes.
        let fill_color = {
            let s = self.gstate.current().fill_color.as_slice();
            [s[0], s[1], s[2]]
        };
        let page_h = f64::from(self.height);

        // PDF CTM maps (0,0)→(1,0)→(1,1)→(0,1) (bottom-left origin).
        // y-flip converts PDF bottom-left to device top-left origin.
        let (x00, y00) = ctm_transform(&ctm, 0.0, 0.0);
        let (x10, y10) = ctm_transform(&ctm, 1.0, 0.0);
        let (x01, y01) = ctm_transform(&ctm, 0.0, 1.0);
        let (x11, y11) = ctm_transform(&ctm, 1.0, 1.0);

        // Guard against a non-finite CTM (malformed PDF).  A NaN or Inf corner
        // would produce i64::MIN/MAX after the floor/ceil cast, corrupting the
        // bounding box calculation.
        if ![x00, y00, x10, y10, x01, y01, x11, y11]
            .iter()
            .all(|v| v.is_finite())
        {
            log::warn!("pdf_interp: blit_image: non-finite CTM corner — skipping image");
            return;
        }

        // Bounding box of the 4 corners in device space (y-flipped).
        let dx0 = x00.min(x10).min(x01).min(x11).floor() as i64;
        let dx1 = x00.max(x10).max(x01).max(x11).ceil() as i64;
        let dy0 = (page_h - y00)
            .min(page_h - y10)
            .min(page_h - y01)
            .min(page_h - y11)
            .floor() as i64;
        let dy1 = (page_h - y00)
            .max(page_h - y10)
            .max(page_h - y01)
            .max(page_h - y11)
            .ceil() as i64;

        // Clamp to bitmap (all values fit u32 after clamping ≥ 0 and < dim).
        let bx0 = dx0.max(0) as u32;
        let bx1 = dx1.min(i64::from(self.width)) as u32;
        let by0 = dy0.max(0) as u32;
        let by1 = dy1.min(i64::from(self.height)) as u32;

        if bx0 >= bx1 || by0 >= by1 {
            return;
        }

        // Inverse CTM: for each device pixel (dx, dy_device) compute image-space
        // coordinates (u, v) ∈ [0,1]² via exact inverse affine mapping.
        //
        // CTM maps PDF user space (u, v) → device pixels (x, y_pdf):
        //   x      = a*u + c*v + e
        //   y_pdf  = b*u + d*v + f
        //
        // After y-flip (device_y = page_h − y_pdf) the inverse is:
        //   u = ( d*(dx − e) − c*(dy_pdf − f)) / det
        //   v = (−b*(dx − e) + a*(dy_pdf − f)) / det
        // where dy_pdf = page_h − dy_device.
        //
        // This is exact for any invertible CTM (axis-aligned or rotated/sheared).
        let [a, b, c, d, e, f] = ctm;
        let det = a.mul_add(d, -(b * c));
        // det is guaranteed non-zero: we already checked all corners are finite,
        // and a singular CTM would produce a degenerate bounding box caught above.
        let inv_det = if det.abs() < 1e-12 {
            log::debug!("pdf_interp: blit_image: near-singular CTM (det={det:.2e}) — skipping");
            return;
        } else {
            1.0 / det
        };

        let img_w = f64::from(img.width);
        let img_h = f64::from(img.height);
        let img_width_usize = img.width as usize;

        // Phase 9 GPU image-blit fast path: device-resident pixels +
        // CUDA kernel transform writing into a per-page DevicePageBuffer.
        // Returns true if the GPU path handled the image; false to fall
        // through to the CPU sampler below.
        #[cfg(feature = "cache")]
        if let crate::resources::image::ImageData::Gpu(cached) = &img.data {
            if self.try_gpu_blit_image(cached, &ctm, page_h) {
                return;
            }
            // Promotion or kernel dispatch failed; we don't have host
            // bytes either, so log and skip the image rather than CPU
            // sampling something that isn't there.
            log::warn!(
                "blit_image: GPU image-blit failed (no fallback bytes available — image will be missing from output)"
            );
            return;
        }

        // CPU path: needs host-resident pixels.  The graceful skip
        // below also covers the cache-feature-on, ImageData::Gpu
        // case if `try_gpu_blit_image` somehow falls through (it
        // currently doesn't, but the explicit None check is cheap).
        let Some(img_bytes) = img.data.as_cpu() else {
            log::warn!(
                "blit_image: image data not host-resident and GPU dispatch unavailable — skipping"
            );
            return;
        };
        // Decoder contract: bytes.len() == width × height × bpp.  Promote
        // the contract to a release-mode check so a truncated buffer (an
        // adversarial PDF reporting larger dims than the decoder produced)
        // skips the image cleanly instead of letting the safe slice index
        // below panic in the inner loop.  One CMP per image; well below
        // noise.
        let bpp = img.color_space.bytes_per_pixel();
        match check_image_bytes_len(img.width, img.height, bpp, img_bytes.len()) {
            Ok(_) => {}
            Err(ImageLenError::DimensionOverflow) => {
                log::warn!(
                    "blit_image: image dimensions overflow usize ({}×{}×{bpp} bpp) — skipping",
                    img.width,
                    img.height,
                );
                return;
            }
            Err(ImageLenError::ShortBuffer { expected }) => {
                log::warn!(
                    "blit_image: decoder produced short buffer ({} bytes, expected {expected}) — skipping",
                    img_bytes.len(),
                );
                return;
            }
        }

        let data = self.bitmap.data_mut();
        let stride = self.width as usize * 3; // Rgb8: 3 bytes per pixel

        // Axis-aligned fast path: when b ≈ 0 and c ≈ 0, the CTM has no rotation
        // or shear.  The inverse mapping simplifies to:
        //   u = (dx − e) / a        (constant step per column)
        //   v = (dy_pdf − f) / d    (constant per row)
        //
        // We convert both to fixed-point (Q32) so the inner loop is pure integer
        // arithmetic — no per-pixel f64 multiply, no clamp, no int→float conversion.
        //
        // Threshold: |b|, |c| < 0.5 device pixels across the full image extent.
        let is_axis_aligned = b.abs() * img_w < 0.5 && c.abs() * img_h < 0.5;

        if is_axis_aligned {
            // Fixed-point scale: 1 image pixel = FP_SCALE units.
            const FP_SCALE: i64 = 1 << 32;

            // ix step per output column: img_w / a pixels per device pixel.
            // a may be negative (x-flipped image).
            // FP_SCALE is 2^32, exactly representable in f64; cast is lossless.
            #[expect(
                clippy::cast_precision_loss,
                reason = "FP_SCALE = 1<<32 is exactly representable in f64; img_w/a fits easily"
            )]
            let ix_step_fp: i64 = ((img_w / a) * FP_SCALE as f64) as i64;

            // iy step per output row: -img_h / d  (v = (dy_pdf - f)/d, iy = (1-v)*img_h).
            // d is negative when PDF y-axis is flipped to device space.
            // Precomputed per-row below.

            // img dimensions are u32, so the usize values fit well within i64.
            #[expect(
                clippy::cast_possible_wrap,
                reason = "img dims are u32; usize→i64 cast is safe on 64-bit targets"
            )]
            let img_max_x = (img_width_usize - 1) as i64;

            let smask = img.smask.as_deref();

            for dy in by0..by1 {
                let dy_pdf = page_h - f64::from(dy);
                // v = (dy_pdf - f) / d; iy = (1 - v) * img_h, clamped.
                let v = ((dy_pdf - f) / d).clamp(0.0, 1.0);
                let iy = ((1.0 - v) * img_h).min(img_h - 1.0) as usize;
                let row_base = iy * img_width_usize;

                // ix at bx0: u = (bx0 - e) / a.
                let u0 = ((f64::from(bx0) - e) / a).clamp(0.0, 1.0);
                #[expect(
                    clippy::cast_precision_loss,
                    reason = "FP_SCALE = 1<<32 is exactly representable in f64"
                )]
                let mut ix_fp: i64 = (u0 * img_w * FP_SCALE as f64) as i64;

                let row_off = dy as usize * stride;

                match img.color_space {
                    ImageColorSpace::Rgb => {
                        for dx in bx0..bx1 {
                            let ix = (ix_fp >> 32).clamp(0, img_max_x) as usize;
                            ix_fp = ix_fp.wrapping_add(ix_step_fp);
                            let img_idx = row_base + ix;
                            if smask.is_some_and(|s| s.get(img_idx).copied().unwrap_or(0xFF) == 0) {
                                continue;
                            }
                            let src = img_idx * 3;
                            // Bounds checked: ix ∈ [0, img_width-1] and
                            // iy ∈ [0, img_height-1] by the clamp above; the
                            // length precheck guarantees src+3 ≤ img_bytes.len().
                            let rgb = &img_bytes[src..src + 3];
                            let pixel_off = row_off + dx as usize * 3;
                            data[pixel_off..pixel_off + 3].copy_from_slice(rgb);
                        }
                    }
                    ImageColorSpace::Gray => {
                        for dx in bx0..bx1 {
                            let ix = (ix_fp >> 32).clamp(0, img_max_x) as usize;
                            ix_fp = ix_fp.wrapping_add(ix_step_fp);
                            let img_idx = row_base + ix;
                            if smask.is_some_and(|s| s.get(img_idx).copied().unwrap_or(0xFF) == 0) {
                                continue;
                            }
                            // Same bounds rationale as RGB arm.
                            let v = img_bytes[img_idx];
                            let pixel_off = row_off + dx as usize * 3;
                            data[pixel_off] = v;
                            data[pixel_off + 1] = v;
                            data[pixel_off + 2] = v;
                        }
                    }
                    ImageColorSpace::Mask => {
                        // No smask gate here — PDF §8.9.6 (function-entry assert).
                        for dx in bx0..bx1 {
                            let ix = (ix_fp >> 32).clamp(0, img_max_x) as usize;
                            ix_fp = ix_fp.wrapping_add(ix_step_fp);
                            let img_idx = row_base + ix;
                            // Same bounds rationale as RGB arm.
                            if img_bytes[img_idx] == 0x00 {
                                let pixel_off = row_off + dx as usize * 3;
                                data[pixel_off..pixel_off + 3].copy_from_slice(&fill_color);
                            }
                        }
                    }
                }
            }
        } else {
            // General path: arbitrary affine CTM (rotation, shear, non-axis-aligned scale).
            for dy in by0..by1 {
                let dy_pdf = page_h - f64::from(dy);
                let dy_rel = dy_pdf - f;
                // Precompute the row-constant parts of the u/v formulas.
                let u_row = (-c * dy_rel) * inv_det;
                let v_row = (a * dy_rel) * inv_det;
                for dx in bx0..bx1 {
                    let dx_rel = f64::from(dx) - e;
                    // Image-space coordinates ∈ [0, 1]; clamp to guard edges.
                    let u = (d * dx_rel).mul_add(inv_det, u_row).clamp(0.0, 1.0);
                    let v = ((-b) * dx_rel).mul_add(inv_det, v_row).clamp(0.0, 1.0);

                    let ix = (u * img_w).min(img_w - 1.0) as usize;
                    let iy = ((1.0 - v) * img_h).min(img_h - 1.0) as usize;
                    let img_idx = iy * img_width_usize + ix;

                    if img
                        .smask
                        .as_deref()
                        .is_some_and(|s| s.get(img_idx).copied().unwrap_or(0xFF) == 0)
                    {
                        continue;
                    }

                    let pixel_off = dy as usize * stride + dx as usize * 3;

                    match img.color_space {
                        ImageColorSpace::Rgb => {
                            let src = img_idx * 3;
                            if let Some(rgb) = img_bytes.get(src..src + 3) {
                                data[pixel_off..pixel_off + 3].copy_from_slice(rgb);
                            }
                        }
                        ImageColorSpace::Gray => {
                            if let Some(&v) = img_bytes.get(img_idx) {
                                data[pixel_off] = v;
                                data[pixel_off + 1] = v;
                                data[pixel_off + 2] = v;
                            }
                        }
                        ImageColorSpace::Mask => {
                            // No smask gate here — PDF §8.9.6 (function-entry assert).
                            if img_bytes.get(img_idx) == Some(&0x00) {
                                data[pixel_off..pixel_off + 3].copy_from_slice(&fill_color);
                            }
                        }
                    }
                }
            }
        }
    }

    /// Phase 9 GPU image blit dispatcher.
    ///
    /// Lazy-allocates `device_page_buffer` on first call, builds the
    /// inverse-CTM coefficients, computes the destination AABB, and
    /// launches the CUDA blit kernel.  Returns `true` on success;
    /// `false` if any prerequisite is missing (no `GpuCtx`, singular
    /// CTM, page-buffer alloc fails, kernel dispatch errors).
    ///
    /// Designed to be cheap to call: per-call cost is one bbox
    /// computation + one ~10 µs kernel launch.  No host-side pixel
    /// touch.
    #[cfg(feature = "cache")]
    #[expect(
        clippy::similar_names,
        reason = "page_w_i / page_h_i are paired width/height constants — renaming would obscure the symmetry"
    )]
    #[expect(
        clippy::cast_possible_truncation,
        reason = "device pixel coords floor/ceil to i32 cleanly within page bounds (max ~64K at 600 DPI fits losslessly)"
    )]
    #[expect(
        clippy::cast_possible_wrap,
        reason = "self.width/height are PDF page dims well below i32::MAX"
    )]
    fn try_gpu_blit_image(
        &mut self,
        cached: &Arc<gpu::cache::CachedDeviceImage>,
        ctm: &[f64; 6],
        page_h: f64,
    ) -> bool {
        use gpu::blit::{BlitBbox, InverseCtm};

        let Some(gpu_ctx) = self.gpu_ctx.as_deref() else {
            log::debug!("blit_image: GPU image blit skipped — no GpuCtx attached");
            return false;
        };
        let Some(cache_state) = self.cache_state.as_mut() else {
            // Shouldn't happen: ImageData::Gpu only originates from a
            // wired-in cache.  Defensive check; cheap.
            return false;
        };
        let image_cache = &cache_state.cache;
        // Stream-identity invariant: the cache uploads `cached`'s
        // device memory on its own stream; the blit kernel below
        // launches on `gpu_ctx`'s stream; the page buffer's
        // `download()` syncs the page buffer's stream.  All three
        // must be the *same* stream — otherwise the kernel could
        // read pre-DMA bytes or `download()` could miss writes.
        // `open_session` constructs the cache from `gpu_ctx.stream()`
        // so this holds today; the assert future-proofs against a
        // refactor that introduces per-worker streams without
        // updating the cache + page buffer plumbing.
        debug_assert!(
            Arc::ptr_eq(gpu_ctx.stream(), image_cache.stream_arc()),
            "GpuCtx stream and DeviceImageCache stream diverged — cache upload, blit kernel, and page-buffer download must share one stream",
        );
        let Some(inv_ctm) = InverseCtm::from_ctm(*ctm) else {
            log::debug!("blit_image: GPU image blit skipped — singular CTM");
            return false;
        };

        // Compute the destination AABB (page pixels).  Must clamp at
        // launch time because rotated images can spill off the page;
        // the kernel itself also guards against negative coords for
        // defence-in-depth.
        let (x00, y00) = ctm_transform(ctm, 0.0, 0.0);
        let (x10, y10) = ctm_transform(ctm, 1.0, 0.0);
        let (x01, y01) = ctm_transform(ctm, 0.0, 1.0);
        let (x11, y11) = ctm_transform(ctm, 1.0, 1.0);
        if ![x00, y00, x10, y10, x01, y01, x11, y11]
            .iter()
            .all(|v| v.is_finite())
        {
            return false;
        }
        let dx0 = x00.min(x10).min(x01).min(x11).floor() as i32;
        let dx1 = x00.max(x10).max(x01).max(x11).ceil() as i32;
        let dy_pdf_min = (page_h - y00)
            .min(page_h - y10)
            .min(page_h - y01)
            .min(page_h - y11);
        let dy_pdf_max = (page_h - y00)
            .max(page_h - y10)
            .max(page_h - y01)
            .max(page_h - y11);
        let dy0 = dy_pdf_min.floor() as i32;
        let dy1 = dy_pdf_max.ceil() as i32;

        // Clamp to page bounds for kernel efficiency (the kernel
        // also guards each thread, but skipping out-of-page tiles is
        // cheaper than launching them).
        let page_w_i = self.width as i32;
        let page_h_i = self.height as i32;
        let bbox = BlitBbox {
            x0: dx0.max(0),
            y0: dy0.max(0),
            x1: dx1.min(page_w_i),
            y1: dy1.min(page_h_i),
        };

        // Lazy-allocate the per-page device buffer on first GPU
        // image; subsequent images on the same page reuse it.
        if cache_state.page_buffer.is_none() {
            let stream = Arc::clone(image_cache.stream_arc());
            match DevicePageBuffer::new(stream, self.width, self.height) {
                Ok(buf) => cache_state.page_buffer = Some(buf),
                Err(e) => {
                    log::warn!("blit_image: DevicePageBuffer alloc failed: {e}");
                    return false;
                }
            }
        }
        let Some(buf) = cache_state.page_buffer.as_mut() else {
            return false;
        };

        #[expect(
            clippy::cast_precision_loss,
            reason = "page_h is u32 page height; lossless f32 for u32 ≤ 2^24, vastly larger than any PDF page"
        )]
        let page_h_f = self.height as f32;
        if let Err(e) = gpu_ctx.blit_image_to_buffer(cached, buf, inv_ctm, bbox, page_h_f) {
            log::warn!("blit_image: GPU kernel dispatch failed: {e}");
            return false;
        }
        true
    }
}

// ── Component → colour helpers ────────────────────────────────────────────────

/// Map a raw component slice to a [`RasterColor`] by channel count.
pub(super) fn components_to_color(comps: &[f64]) -> RasterColor {
    match comps {
        [g] => RasterColor::gray(*g),
        [r, g, b] => RasterColor::rgb(*r, *g, *b),
        [c, m, y, k] => RasterColor::cmyk(*c, *m, *y, *k),
        _ => RasterColor::default(),
    }
}

/// Map a PDF line cap integer (0–2) to [`LineCap`]; defaults to `Butt`.
const fn int_to_cap(v: i32) -> LineCap {
    match v {
        1 => LineCap::Round,
        2 => LineCap::Projecting,
        _ => LineCap::Butt,
    }
}

/// Map a PDF line join integer (0–2) to [`LineJoin`]; defaults to `Miter`.
const fn int_to_join(v: i32) -> LineJoin {
    match v {
        1 => LineJoin::Round,
        2 => LineJoin::Bevel,
        _ => LineJoin::Miter,
    }
}

/// Reason `blit_image`'s decoder-contract check rejected an image.
#[derive(Debug, PartialEq, Eq)]
enum ImageLenError {
    /// `width × height × bpp` overflows `usize`.
    DimensionOverflow,
    /// Buffer is shorter than the declared dimensions require.
    ShortBuffer { expected: usize },
}

/// Validates the decoder contract `actual_len >= width × height × bpp` and
/// returns the expected length on success.  Pulled out of `blit_image` so the
/// two guard branches (overflow, short buffer) are unit-testable without
/// constructing a `PageRenderer`.
fn check_image_bytes_len(
    width: u32,
    height: u32,
    bpp: usize,
    actual_len: usize,
) -> Result<usize, ImageLenError> {
    let expected = (width as usize)
        .checked_mul(height as usize)
        .and_then(|n| n.checked_mul(bpp))
        .ok_or(ImageLenError::DimensionOverflow)?;
    if actual_len < expected {
        return Err(ImageLenError::ShortBuffer { expected });
    }
    Ok(expected)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn diag_with_ppi(ppi: f32) -> PageDiagnostics {
        PageDiagnostics {
            has_images: true,
            source_ppi_hint: Some(ppi),
            ..PageDiagnostics::default()
        }
    }

    #[test]
    fn suggested_dpi_rounds_up_to_nearest_step() {
        // 67 PPI → next step is 72.
        assert_eq!(diag_with_ppi(67.0).suggested_dpi(72.0, 600.0), Some(72.0));
        // 73 PPI → next step is 96.
        assert_eq!(diag_with_ppi(73.0).suggested_dpi(72.0, 600.0), Some(96.0));
        // 150 PPI exactly → stays at 150.
        assert_eq!(diag_with_ppi(150.0).suggested_dpi(72.0, 600.0), Some(150.0));
        // 280 PPI → next step is 300.
        assert_eq!(diag_with_ppi(280.0).suggested_dpi(72.0, 600.0), Some(300.0));
    }

    #[test]
    fn suggested_dpi_respects_min_clamp() {
        // 67 PPI → step is 72, but min=150 clamps it up.
        assert_eq!(diag_with_ppi(67.0).suggested_dpi(150.0, 300.0), Some(150.0));
    }

    #[test]
    fn suggested_dpi_respects_max_clamp() {
        // 700 PPI → above all steps; last step is 600, then clamped to max=300.
        assert_eq!(diag_with_ppi(700.0).suggested_dpi(72.0, 300.0), Some(300.0));
    }

    #[test]
    fn suggested_dpi_none_when_no_images() {
        let diag = PageDiagnostics::default(); // source_ppi_hint is None
        assert_eq!(diag.suggested_dpi(150.0, 300.0), None);
    }

    // ── check_image_bytes_len ──────────────────────────────────────────────

    #[test]
    fn check_image_bytes_len_accepts_exact() {
        // 2 × 2 × Rgb (bpp=3) → expected 12 bytes.
        assert_eq!(check_image_bytes_len(2, 2, 3, 12), Ok(12));
    }

    #[test]
    fn check_image_bytes_len_accepts_over_long_buffer() {
        // Over-long buffers are fine — only short buffers are rejected.
        assert_eq!(check_image_bytes_len(2, 2, 3, 15), Ok(12));
    }

    #[test]
    fn check_image_bytes_len_rejects_short_buffer() {
        assert_eq!(
            check_image_bytes_len(2, 2, 3, 6),
            Err(ImageLenError::ShortBuffer { expected: 12 }),
        );
    }

    #[test]
    fn check_image_bytes_len_rejects_dimension_overflow() {
        // u32::MAX × u32::MAX × 3 overflows usize on 64-bit targets.
        assert_eq!(
            check_image_bytes_len(u32::MAX, u32::MAX, 3, 0),
            Err(ImageLenError::DimensionOverflow),
        );
    }

    #[test]
    fn check_image_bytes_len_one_by_one_gray() {
        // 1 × 1 × Gray (bpp=1) → expected 1 byte.
        assert_eq!(check_image_bytes_len(1, 1, 1, 1), Ok(1));
    }
}
