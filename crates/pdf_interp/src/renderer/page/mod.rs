//! Per-page operator dispatcher.
//!
//! [`PageRenderer`] holds a target [`Bitmap`] and a [`GStateStack`], iterates
//! over a decoded operator sequence, and calls into the `raster` crate for
//! each painting operator.
//!
//! Sub-modules:
//! - [`operators`] — `execute_one` operator dispatch table.
//! - [`text`] — `show_text` and `show_text_type3` rendering.
//! - [`patterns`] — `resolve_fill_pattern` and `render_pattern_tile`.
//! - [`text_ops`] — `GlyphRecord` and `text_to_device` helpers.
//! - [`annotations`] — annotation appearance stream rendering.
//! - [`gpu_ops`] — GPU-accelerated fill dispatch.
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
#[cfg(any(feature = "gpu-aa", feature = "gpu-icc"))]
use gpu::GpuCtx;
#[cfg(feature = "vaapi")]
use gpu::JpegQueueHandle;
#[cfg(feature = "nvjpeg")]
use gpu::nvjpeg::NvJpegDecoder;
#[cfg(feature = "nvjpeg2k")]
use gpu::nvjpeg2k::NvJpeg2kDecoder;
#[cfg(any(feature = "gpu-aa", feature = "gpu-icc"))]
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
    /// [`RasterOptions::dpi`] renders at the minimum resolution that avoids
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
    #[cfg(any(feature = "gpu-aa", feature = "gpu-icc"))]
    gpu_ctx: Option<Arc<GpuCtx>>,
    /// Per-page cache of baked ICC CMYK→RGB CLUT tables.  Keyed by a hash of the
    /// raw ICC profile bytes so that repeated images sharing the same profile (the
    /// common case in press PDFs) only pay the bake cost once per page.
    #[cfg(feature = "gpu-icc")]
    icc_clut_cache: crate::resources::image::IccClutCache,
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
            diag: PageDiagnostics::default(),
            filter_counts: [0u32; IMAGE_FILTER_COUNT],
            ocg_stack: Vec::new(),
            #[cfg(feature = "nvjpeg")]
            nvjpeg: None,
            #[cfg(feature = "vaapi")]
            vaapi_jpeg_queue: None,
            #[cfg(feature = "nvjpeg2k")]
            nvjpeg2k: None,
            #[cfg(any(feature = "gpu-aa", feature = "gpu-icc"))]
            gpu_ctx: None,
            #[cfg(feature = "gpu-icc")]
            icc_clut_cache: crate::resources::image::IccClutCache::new(),
        })
    }

    /// Attach a GPU JPEG decoder to this renderer.
    ///
    /// When set, `DCTDecode` image streams with pixel area ≥
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
    pub fn take_nvjpeg(&mut self) -> Option<NvJpegDecoder> {
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
    #[cfg(any(feature = "gpu-aa", feature = "gpu-icc"))]
    pub fn set_gpu_ctx(&mut self, ctx: Option<Arc<GpuCtx>>) {
        self.gpu_ctx = ctx;
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
        (self.bitmap, self.diag)
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

        // GPU fill dispatch — only for solid-colour fills (patterns not yet
        // GPU-accelerated).  Arc::clone releases the borrow before the mutable
        // self borrow inside the try_* helpers.
        //
        // Dispatch order:
        //   1. Tile-parallel analytical fill (GPU_TILE_FILL_THRESHOLD) — exact
        //      analytical coverage, best for large solid fills.
        //   2. Warp-ballot 64-sample AA (GPU_AA_FILL_THRESHOLD) — stochastic
        //      but faster for medium-sized fills.
        //   3. CPU scanline AA (always available as final fallback).
        //
        // The nested ifs keep the pattern guard (tiled.is_none()) separate from
        // the GPU-availability guard to make each condition readable.
        #[cfg(feature = "gpu-aa")]
        #[expect(
            clippy::collapsible_if,
            reason = "outer guard (no tiling pattern) and inner guard (GPU context present) \
                      are logically distinct; collapsing would obscure the separation"
        )]
        if tiled.is_none() {
            if let Some(ctx) = self.gpu_ctx.clone() {
                if self.try_gpu_tile_fill(path, even_odd, &pipe, &src, &ctx) {
                    return;
                }
                if self.try_gpu_aa_fill(path, even_odd, &pipe, &src, &ctx) {
                    return;
                }
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
        let img = self.resources.image(
            name,
            #[cfg(feature = "nvjpeg")]
            self.nvjpeg.as_mut(),
            #[cfg(feature = "vaapi")]
            self.vaapi_jpeg_queue.as_ref(),
            #[cfg(feature = "nvjpeg2k")]
            self.nvjpeg2k.as_mut(),
            #[cfg(feature = "gpu-icc")]
            self.gpu_ctx.as_deref(),
            #[cfg(feature = "gpu-icc")]
            Some(&mut self.icc_clut_cache),
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
            paint_group(&mut self.bitmap, g, &pipe);
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
        let img_bytes: &[u8] = img.data.as_cpu().expect(
            "renderer blit requires CPU-resident image; GPU variant unreachable in legacy path",
        );

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
            #[expect(
                clippy::cast_possible_wrap,
                reason = "img dims are u32; usize→i64 cast is safe on 64-bit targets"
            )]
            let img_max_y = (img.height as usize - 1) as i64;

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
                            // SAFETY: ix ∈ [0, img_width-1] and iy ∈ [0, img_height-1]
                            // by the clamp above, so src+3 ≤ img.data.len().
                            #[expect(
                                unsafe_code,
                                reason = "bounds proven by clamp: ix < img_width, iy < img_height"
                            )]
                            let rgb = unsafe { img_bytes.get_unchecked(src..src + 3) };
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
                            // SAFETY: same bounds proof as RGB arm.
                            #[expect(
                                unsafe_code,
                                reason = "bounds proven by clamp: ix < img_width, iy < img_height"
                            )]
                            let v = unsafe { *img_bytes.get_unchecked(img_idx) };
                            let pixel_off = row_off + dx as usize * 3;
                            data[pixel_off] = v;
                            data[pixel_off + 1] = v;
                            data[pixel_off + 2] = v;
                        }
                    }
                    ImageColorSpace::Mask => {
                        for dx in bx0..bx1 {
                            let ix = (ix_fp >> 32).clamp(0, img_max_x) as usize;
                            ix_fp = ix_fp.wrapping_add(ix_step_fp);
                            let img_idx = row_base + ix;
                            // SAFETY: same bounds proof as RGB arm.
                            #[expect(
                                unsafe_code,
                                reason = "bounds proven by clamp: ix < img_width, iy < img_height"
                            )]
                            if unsafe { *img_bytes.get_unchecked(img_idx) } == 0x00 {
                                let pixel_off = row_off + dx as usize * 3;
                                data[pixel_off..pixel_off + 3].copy_from_slice(&fill_color);
                            }
                        }
                    }
                }

                // Suppress unused-variable warning in non-GPU builds.
                let _ = img_max_y;
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
                            if img_bytes.get(img_idx) == Some(&0x00) {
                                data[pixel_off..pixel_off + 3].copy_from_slice(&fill_color);
                            }
                        }
                    }
                }
            }
        }
    }
}

// ── Component → colour helpers ────────────────────────────────────────────────

/// Map a raw component slice to a [`RasterColor`] by channel count.
fn components_to_color(comps: &[f64]) -> RasterColor {
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
}
