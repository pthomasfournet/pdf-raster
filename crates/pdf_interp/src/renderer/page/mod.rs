//! Per-page operator dispatcher.
//!
//! [`PageRenderer`] holds a target [`Bitmap`] and a [`GStateStack`], iterates
//! over a decoded operator sequence, and calls into the `raster` crate for
//! each painting operator.
//!
//! Sub-modules:
//! - [`text_ops`] — `GlyphRecord` and `text_to_device`, used by `show_text`.
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

mod text_ops;

use self::text_ops::{GlyphRecord, text_to_device};

use lopdf::{Document, Object, ObjectId};

use color::Rgb8;
use font::{
    cache::GlyphCache,
    engine::{FontEngine, SharedEngine},
};
use raster::{
    Bitmap, TiledPattern, eo_fill, fill,
    glyph::{GlyphBitmap, fill_glyph},
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
use super::gstate::{GStateStack, ctm_multiply, ctm_transform, mat2x2_mul};
use super::text::TextState;
use crate::content::{Operator, TextArrayElement};
use crate::resources::{ImageColorSpace, PageResources, image::decode_inline_image};
#[cfg(feature = "nvjpeg")]
use gpu::nvjpeg::NvJpegDecoder;

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
    bitmap: Bitmap<Rgb8>,
    /// Page width in device pixels.
    width: u32,
    /// Page height in device pixels.
    height: u32,
    /// Graphics state save/restore stack.
    gstate: GStateStack,
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
    resources: PageResources<'doc>,
    /// Current Form `XObject` nesting depth (0 = top-level page).
    form_depth: u32,
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
}

impl<'doc> PageRenderer<'doc> {
    /// Create a renderer for a blank white page of `width × height` pixels,
    /// where 1 user-space unit = 1 device pixel (72 dpi).
    ///
    /// `doc` and `page_id` are used to resolve font and resource dictionaries.
    #[must_use]
    pub fn new(width: u32, height: u32, doc: &'doc Document, page_id: ObjectId) -> Self {
        Self::new_scaled(width, height, 1.0, doc, page_id)
    }

    /// Create a renderer with an initial uniform scale in the CTM.
    ///
    /// `scale = dpi / 72.0` maps PDF points to device pixels.
    ///
    /// # Panics
    ///
    /// Panics if `scale` is not a positive finite number, or if `FreeType`
    /// initialisation fails.
    #[must_use]
    pub fn new_scaled(
        width: u32,
        height: u32,
        scale: f64,
        doc: &'doc Document,
        page_id: ObjectId,
    ) -> Self {
        assert!(
            scale.is_finite() && scale > 0.0,
            "PageRenderer::new_scaled: scale must be a positive finite number, got {scale}"
        );

        let mut bitmap = Bitmap::<Rgb8>::new(width, height, 1, false);
        bitmap.data_mut().fill(255u8); // white background

        let mut gstate = GStateStack::new(width, height);
        if (scale - 1.0).abs() > f64::EPSILON {
            gstate.current_mut().ctm = [scale, 0.0, 0.0, scale, 0.0, 0.0];
        }

        let engine: SharedEngine =
            FontEngine::init(true, true, false).expect("FreeType initialisation failed");
        let glyph_cache = GlyphCache::new();
        let font_cache = FontCache::new(engine, glyph_cache);
        let resources = PageResources::new(doc, page_id);

        Self {
            bitmap,
            width,
            height,
            gstate,
            path: None,
            pending_clip: None,
            font_cache,
            resources,
            form_depth: 0,
            ocg_stack: Vec::new(),
            #[cfg(feature = "nvjpeg")]
            nvjpeg: None,
        }
    }

    /// Attach a GPU JPEG decoder to this renderer.
    ///
    /// When set, `DCTDecode` image streams with pixel area ≥
    /// [`crate::resources::image::GPU_JPEG_THRESHOLD_PX`] are decoded on the
    /// GPU via nvJPEG rather than `zune-jpeg`.
    ///
    /// Calling this with `None` detaches any existing decoder (reverts to CPU).
    #[cfg(feature = "nvjpeg")]
    pub fn set_nvjpeg(&mut self, dec: Option<NvJpegDecoder>) {
        self.nvjpeg = dec;
    }

    /// Consume the renderer and return the finished bitmap.
    #[must_use]
    pub fn finish(self) -> Bitmap<Rgb8> {
        self.bitmap
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
        let doc = self.resources.doc();
        let Ok(page_dict) = doc.get_dictionary(page_id) else { return };

        // Collect annotation refs to avoid borrow conflict.
        let annot_ids: Vec<ObjectId> = {
            let arr = match page_dict.get(b"Annots") {
                Ok(Object::Array(a)) => a.clone(),
                Ok(Object::Reference(id)) => {
                    match doc.get_object(*id).ok().and_then(|o| o.as_array().ok().cloned()) {
                        Some(a) => a,
                        None => return,
                    }
                }
                _ => return,
            };
            arr.iter()
                .filter_map(|o| if let Object::Reference(id) = o { Some(*id) } else { None })
                .collect()
        };

        for annot_id in annot_ids {
            self.render_one_annotation(annot_id);
        }
    }

    fn render_one_annotation(&mut self, annot_id: ObjectId) {
        let doc = self.resources.doc();

        let Ok(annot_dict) = doc.get_dictionary(annot_id) else { return };

        // Annotation rect in page user space: [llx, lly, urx, ury].
        let Some(rect) = read_rect(annot_dict) else { return };

        // Resolve AP/N appearance stream.
        let Some(ap_dict) = annot_dict.get(b"AP").ok().and_then(|o| match o {
            Object::Dictionary(d) => Some(d),
            Object::Reference(id) => doc.get_dictionary(*id).ok(),
            _ => None,
        }) else { return };

        // N can be a stream reference or a sub-dict (state-keyed appearances).
        let stream_id: ObjectId = {
            let Ok(n_obj) = ap_dict.get(b"N") else { return };
            match n_obj {
                Object::Reference(id) => *id,
                Object::Dictionary(_) => {
                    // State-keyed: look up the current appearance state (AS).
                    let state = annot_dict
                        .get(b"AS")
                        .ok()
                        .and_then(|o| o.as_name().ok());
                    let Some(state_key) = state else { return };
                    match n_obj.as_dict().ok().and_then(|d| d.get(state_key).ok()) {
                        Some(Object::Reference(id)) => *id,
                        _ => return,
                    }
                }
                _ => return,
            }
        };

        // Build the FormXObject from the appearance stream.
        let Some(mut form) = self.resources.form_from_stream_id(stream_id) else { return };

        // form.bbox was populated by form_from_stream_id; appearance streams carry BBox not Rect.
        let [llx, lly, urx, ury] = rect;
        let [bx0, by0, bx1, by1] = form.bbox;
        let bw = bx1 - bx0;
        let bh = by1 - by0;
        if bw.abs() < f64::EPSILON || bh.abs() < f64::EPSILON {
            return;
        }

        let sx = (urx - llx) / bw;
        let sy = (ury - lly) / bh;
        let tx = bx0.mul_add(-sx, llx);
        let ty = by0.mul_add(-sy, lly);
        let bbox_to_rect: [f64; 6] = [sx, 0.0, 0.0, sy, tx, ty];

        // Composed rendering matrix: stream.Matrix × bbox_to_rect.
        form.matrix = ctm_multiply(&form.matrix, &bbox_to_rect);

        self.do_form_xobject(&form);
    }

    #[expect(clippy::too_many_lines, reason = "operator dispatch table")]
    #[expect(
        clippy::match_same_arms,
        reason = "intentional stubs for unimplemented operators"
    )]
    #[expect(
        clippy::many_single_char_names,
        reason = "PDF matrix components and PDF spec variable names"
    )]
    fn execute_one(&mut self, op: &Operator) {
        // ── Optional Content Group (OCG) visibility ───────────────────────────
        // Always handle OCG stack operators regardless of visibility, then skip
        // all other rendering when inside an inactive group.
        match op {
            Operator::BeginOptionalContent(key) => {
                let visible = self.resources.ocg_is_visible(key);
                self.ocg_stack.push(visible);
                return;
            }
            Operator::EndOptionalContent => {
                self.ocg_stack.pop();
                return;
            }
            _ => {}
        }
        // Skip rendering operators when inside an inactive OCG.
        if self.ocg_stack.iter().any(|&v| !v) {
            return;
        }

        match op {
            // ── Graphics state ────────────────────────────────────────────────
            Operator::Save => self.gstate.save(),
            Operator::Restore => self.gstate.restore(),

            Operator::ConcatMatrix(m) => {
                let old = self.gstate.current().ctm;
                self.gstate.current_mut().ctm = ctm_multiply(&old, m);
            }

            Operator::SetLineWidth(w) => self.gstate.current_mut().line_width = *w,
            Operator::SetLineCap(c) => self.gstate.current_mut().line_cap = int_to_cap(*c),
            Operator::SetLineJoin(j) => self.gstate.current_mut().line_join = int_to_join(*j),
            Operator::SetMiterLimit(m) => self.gstate.current_mut().miter_limit = *m,
            Operator::SetFlatness(f) => self.gstate.current_mut().flatness = *f,
            Operator::SetDash { dashes, phase } => {
                self.gstate.current_mut().dash = (dashes.clone(), *phase);
            }
            // Rendering intent is an output-intent hint only; no change to rendering.
            Operator::SetRenderingIntent(_) => {}

            Operator::SetExtGState(name) => {
                if let Some(params) = self.resources.ext_gstate(name) {
                    let gs = self.gstate.current_mut();
                    if let Some(a) = params.fill_alpha {
                        gs.fill_alpha = a;
                    }
                    if let Some(a) = params.stroke_alpha {
                        gs.stroke_alpha = a;
                    }
                    if let Some(w) = params.line_width {
                        gs.line_width = w;
                    }
                    if let Some(c) = params.line_cap {
                        gs.line_cap = int_to_cap(c);
                    }
                    if let Some(j) = params.line_join {
                        gs.line_join = int_to_join(j);
                    }
                    if let Some(m) = params.miter_limit {
                        gs.miter_limit = m;
                    }
                    if let Some(f) = params.flatness {
                        gs.flatness = f;
                    }
                    if let Some(bm) = params.blend_mode {
                        gs.blend_mode = bm;
                    }
                }
            }

            // ── Colour ────────────────────────────────────────────────────────
            Operator::SetFillGray(g) => self.set_fill(RasterColor::gray(*g)),
            Operator::SetFillRgb(r, g, b) => self.set_fill(RasterColor::rgb(*r, *g, *b)),
            Operator::SetFillCmyk(c, m, y, k) => self.set_fill(RasterColor::cmyk(*c, *m, *y, *k)),
            Operator::SetFillColor(comps) => self.set_fill(components_to_color(comps)),
            Operator::SetFillColorSpace(_) => {
                // Switching colour space clears any active pattern.
                self.gstate.current_mut().fill_pattern = None;
            }
            Operator::SetFillPattern { name, components } => {
                let gs = self.gstate.current_mut();
                gs.fill_pattern = Some(name.clone());
                gs.fill_pattern_components.clone_from(components);
            }

            Operator::SetStrokeGray(g) => self.set_stroke(RasterColor::gray(*g)),
            Operator::SetStrokeRgb(r, g, b) => self.set_stroke(RasterColor::rgb(*r, *g, *b)),
            Operator::SetStrokeCmyk(c, m, y, k) => {
                self.set_stroke(RasterColor::cmyk(*c, *m, *y, *k));
            }
            Operator::SetStrokeColor(comps) => self.set_stroke(components_to_color(comps)),
            Operator::SetStrokeColorSpace(_) => {
                // Switching colour space clears any active pattern.
                self.gstate.current_mut().stroke_pattern = None;
            }
            Operator::SetStrokePattern { name, components } => {
                let gs = self.gstate.current_mut();
                gs.stroke_pattern = Some(name.clone());
                gs.stroke_pattern_components.clone_from(components);
            }

            // ── Path construction ─────────────────────────────────────────────
            Operator::MoveTo(x, y) => {
                let (dx, dy) = self.to_device(*x, *y);
                let _ = self.path_builder().move_to(dx, dy);
            }
            Operator::LineTo(x, y) => {
                let (dx, dy) = self.to_device(*x, *y);
                let _ = self.path_builder().line_to(dx, dy);
            }
            Operator::CurveTo(x1, y1, x2, y2, x3, y3) => {
                let (dx1, dy1) = self.to_device(*x1, *y1);
                let (dx2, dy2) = self.to_device(*x2, *y2);
                let (dx3, dy3) = self.to_device(*x3, *y3);
                let _ = self.path_builder().curve_to(dx1, dy1, dx2, dy2, dx3, dy3);
            }
            Operator::CurveToV(x2, y2, x3, y3) => {
                // `v`: first control point = current point.
                let Some(cp) = self.path_builder().cur_pt() else {
                    log::debug!("pdf_interp: CurveToV with no current point — operator ignored");
                    return;
                };
                let (dx2, dy2) = self.to_device(*x2, *y2);
                let (dx3, dy3) = self.to_device(*x3, *y3);
                let _ = self.path_builder().curve_to(cp.x, cp.y, dx2, dy2, dx3, dy3);
            }
            Operator::CurveToY(x1, y1, x3, y3) => {
                // `y`: second control point = endpoint.
                let (dx1, dy1) = self.to_device(*x1, *y1);
                let (dx3, dy3) = self.to_device(*x3, *y3);
                let _ = self.path_builder().curve_to(dx1, dy1, dx3, dy3, dx3, dy3);
            }
            Operator::ClosePath => {
                if let Some(b) = self.path.as_mut() {
                    let _ = b.close(false);
                }
            }
            Operator::Rectangle(x, y, w, h) => {
                let (x0, y0) = self.to_device(*x, *y);
                let (x1, y1) = self.to_device(*x + *w, *y + *h);
                let b = self.path_builder();
                let _ = b.move_to(x0, y0);
                let _ = b.line_to(x1, y0);
                let _ = b.line_to(x1, y1);
                let _ = b.line_to(x0, y1);
                let _ = b.close(true);
            }

            // ── Path painting ─────────────────────────────────────────────────
            Operator::Fill => self.do_fill(false),
            Operator::FillEvenOdd => self.do_fill(true),
            Operator::Stroke => self.do_stroke(),
            Operator::CloseStroke => {
                if let Some(b) = self.path.as_mut() {
                    let _ = b.close(false);
                }
                self.do_stroke();
            }
            Operator::CloseFillStroke => {
                if let Some(b) = self.path.as_mut() {
                    let _ = b.close(false);
                }
                self.do_fill_then_stroke(false);
            }
            Operator::CloseFillStrokeEvenOdd => {
                if let Some(b) = self.path.as_mut() {
                    let _ = b.close(false);
                }
                self.do_fill_then_stroke(true);
            }
            Operator::FillStroke => self.do_fill_then_stroke(false),
            Operator::FillStrokeEvenOdd => self.do_fill_then_stroke(true),
            Operator::EndPath => {
                self.apply_pending_clip();
                self.path = None;
            }

            // ── Clipping ──────────────────────────────────────────────────────
            // W/W* set a pending flag; the clip is applied by the *next*
            // path-terminating operator (PDF §8.5.4).  The path is NOT
            // consumed here — it may still be painted by e.g. "W f".
            Operator::Clip => self.pending_clip = Some(false),
            Operator::ClipEvenOdd => self.pending_clip = Some(true),

            // ── Text objects ──────────────────────────────────────────────────
            Operator::BeginText => self.gstate.current_mut().text.begin_text(),
            Operator::EndText => {}

            // ── Text state ────────────────────────────────────────────────────
            Operator::SetFont { name, size } => {
                let ts = &mut self.gstate.current_mut().text;
                ts.font_name.clone_from(name);
                ts.font_size = *size;
            }
            Operator::SetCharSpacing(v) => self.gstate.current_mut().text.char_spacing = *v,
            Operator::SetWordSpacing(v) => self.gstate.current_mut().text.word_spacing = *v,
            Operator::SetHorizScaling(v) => self.gstate.current_mut().text.horiz_scaling = *v,
            Operator::SetLeading(v) => self.gstate.current_mut().text.leading = *v,
            Operator::SetTextRise(v) => self.gstate.current_mut().text.rise = *v,
            Operator::SetTextRenderMode(v) => self.gstate.current_mut().text.render_mode = *v,

            // ── Text positioning ──────────────────────────────────────────────
            Operator::TextMove(tx, ty) => {
                self.gstate.current_mut().text.move_by(*tx, *ty);
            }
            Operator::TextMoveSetLeading(tx, ty) => {
                self.gstate
                    .current_mut()
                    .text
                    .move_by_and_set_leading(*tx, *ty);
            }
            Operator::SetTextMatrix(m) => self.gstate.current_mut().text.set_matrix(*m),
            Operator::NextLine => self.gstate.current_mut().text.next_line(),

            // ── Text showing ──────────────────────────────────────────────────
            Operator::ShowText(bytes) => {
                self.show_text(bytes);
            }
            Operator::ShowTextArray(elems) => {
                for elem in elems {
                    match elem {
                        TextArrayElement::Text(bytes) => self.show_text(bytes),
                        TextArrayElement::Offset(kern) => {
                            // Negative kern = move right; thousandths of text-space unit.
                            let ts = &mut self.gstate.current_mut().text;
                            let shift = -kern / 1000.0 * ts.font_size;
                            let [a, b, c, d, e, f] = ts.text_matrix;
                            // Apply horizontal shift along the text direction.
                            ts.text_matrix = [a, b, c, d, e + shift * a, f + shift * b];
                        }
                    }
                }
            }
            Operator::MoveNextLineShow(bytes) => {
                self.gstate.current_mut().text.next_line();
                self.show_text(bytes);
            }
            Operator::MoveNextLineShowSpaced { aw, ac, text, .. } => {
                let gs = self.gstate.current_mut();
                gs.text.word_spacing = *aw;
                gs.text.char_spacing = *ac;
                gs.text.next_line();
                self.show_text(text);
            }

            // ── XObjects / images / shading ────────────────────────────────────
            Operator::PaintXObject(name) => {
                self.do_xobject(name);
            }
            Operator::InlineImage { params, data } => {
                if let Some(img) = decode_inline_image(self.resources.doc(), params, data) {
                    self.blit_image(&img);
                } else {
                    log::debug!("pdf_interp: inline image decode failed — skipping");
                }
            }
            Operator::PaintShading(name) => {
                self.do_shading(name);
            }

            // ── Type 3 glyph metrics ──────────────────────────────────────────
            // `d0`/`d1` declare the glyph advance width and (for `d1`) a bounding
            // box.  Width is sourced from the font's `Widths` array at call time;
            // the BBox is informational.  Both operators are no-ops in the renderer
            // — the CharProc's subsequent path/colour operators do the actual work.
            Operator::Type3GlyphWidth(..) | Operator::Type3GlyphWidthBBox(..) => {}

            // ── No-ops ────────────────────────────────────────────────────────
            Operator::MarkedContent | Operator::CompatibilitySection => {}

            // Handled in the OCG pre-pass above; unreachable here.
            Operator::BeginOptionalContent(_) | Operator::EndOptionalContent => {}

            Operator::Unknown(kw) => {
                log::warn!(
                    "pdf_interp: unknown operator: {}",
                    String::from_utf8_lossy(kw)
                );
            }

            // Compile-time exhaustiveness guard.
            #[expect(
                unreachable_patterns,
                reason = "future Operator variants are caught here"
            )]
            _ => {}
        }
    }

    // ── Text rendering ────────────────────────────────────────────────────────

    /// Render a byte string using the current font, colour, and text matrix.
    ///
    /// Two-phase design to satisfy the borrow checker:
    /// 1. Rasterize all glyphs while holding the `font_cache` borrow (no bitmap access).
    /// 2. Blit the pre-rasterized bitmaps, using `bitmap` mutably (no `font_cache` access).
    #[expect(clippy::many_single_char_names, reason = "PDF matrix components")]
    #[expect(
        clippy::too_many_lines,
        reason = "two-phase glyph render (rasterize then blit) inherently interleaves font, matrix, and clip logic that cannot be cleanly split without adding costly intermediate allocations"
    )]
    fn show_text(&mut self, bytes: &[u8]) {
        if bytes.is_empty() {
            return;
        }

        // PDF render modes (PDF §9.3.6 Table 106):
        //   0 fill, 1 stroke, 2 fill+stroke, 3 invisible
        //   4 fill+clip, 5 stroke+clip, 6 fill+stroke+clip, 7 clip only
        //
        // For modes 4–6 we paint AND add glyph outlines to the clip path.
        // Mode 3 and 7 skip painting; mode 7 still adds outlines to the clip.
        let render_mode = self.gstate.current().text.render_mode;
        let do_paint = render_mode != 3 && render_mode != 7;
        let do_clip = render_mode >= 4;

        // Mode 3 (invisible) with no clip — nothing to do.
        if !do_paint && !do_clip {
            return;
        }

        // Snapshot all state we need before any mutable borrow of font_cache.
        let font_name = self.gstate.current().text.font_name.clone();
        let font_size = self.gstate.current().text.font_size;
        let char_spacing = self.gstate.current().text.char_spacing;
        let word_spacing = self.gstate.current().text.word_spacing;
        // PDF Tz is a percentage (default 100 %).  A value of 0 % is degenerate
        // (zero horizontal advance); we map it to 100 % to avoid invisible text.
        // This overrides an explicit `Tz 0` instruction — rare in practice.
        let raw_hz = self.gstate.current().text.horiz_scaling;
        let horiz_scaling = if raw_hz.abs() < f64::EPSILON {
            log::debug!("pdf_interp: Tz is 0 %, substituting 100 %");
            1.0
        } else {
            raw_hz / 100.0
        };
        let ctm = self.gstate.current().ctm;
        let rise = self.gstate.current().text.rise;
        let mut tm = self.gstate.current().text.text_matrix;

        // Resolve font descriptor (immutable borrow of resources).
        let Some(descriptor) = self.resources.font_dict(&font_name) else {
            log::debug!(
                "pdf_interp: no font dict for /{}",
                String::from_utf8_lossy(&font_name)
            );
            return;
        };

        // ── Type 3 fast path ──────────────────────────────────────────────────
        // Type 3 glyphs are PDF content streams executed at the current pen
        // position.  They paint directly into the page bitmap, so there is no
        // two-phase rasterise-then-blit loop — each CharProc runs inline.
        if let Some(t3) = descriptor.type3.clone() {
            self.show_text_type3(
                bytes,
                &t3,
                font_size,
                ctm,
                tm,
                char_spacing,
                word_spacing,
                horiz_scaling,
                rise,
                do_paint,
            );
            return;
        }

        // Phase 1: rasterize glyphs and optionally collect outlines for clip.
        //
        // Two-phase design: borrow `font_cache` here (read-only after load), then
        // release it before touching `self.bitmap` or `self.gstate` in phase 2.
        let mut records: Vec<GlyphRecord> = Vec::with_capacity(bytes.len());
        // Glyph outline paths in device space — populated only when `do_clip`.
        // One entry per character code; `None` when the glyph has no outline.
        let mut clip_paths: Vec<Option<raster::path::Path>> = if do_clip {
            Vec::with_capacity(bytes.len())
        } else {
            Vec::new()
        };

        {
            // Trm[2×2] = font_size × Tm[2×2] × CTM[2×2] — the text rendering
            // matrix in device pixels, encoding actual size and skew/rotation.
            // The 2×2 submatrix (indices 0–3) is stable across the glyph loop
            // because only the translation components (indices 4–5) change as
            // the pen advances — the orientation and size don't change mid-string.
            let tm2x2 = mat2x2_mul(&tm, &ctm);
            let trm = tm2x2.map(|v| v * font_size);

            // Load (or retrieve cached) FreeType face — mutable borrow of font_cache.
            let Some(face) = self.font_cache.get_or_load(&font_name, &descriptor, trm) else {
                return;
            };

            // Iterate character codes.  Simple fonts use one byte per character;
            // Type 0 composite fonts use 1–4 bytes determined by the Encoding CMap.
            let cid_enc = descriptor.cid_encoding.as_ref();

            // Shared helper: push a rasterized glyph into `records` (when painting)
            // and collect its outline path into `clip_paths` (when clipping).
            // `char_code_for_path` is the char-code argument for `glyph_path`
            // (same convention as `make_glyph` — the face resolves GID internally).
            let mut push_glyph = |char_code_for_path: u32, gid: u32, pen_x: i32, pen_y: i32| {
                if do_paint && let Some(bmp) = face.make_glyph(gid, 0) {
                    records.push(GlyphRecord {
                        pen_x,
                        pen_y,
                        x_off: bmp.x_off,
                        y_off: bmp.y_off,
                        width: bmp.width,
                        height: bmp.height,
                        aa: bmp.aa,
                        data: bmp.data,
                    });
                }
                if do_clip {
                    // Collect the glyph outline in device space.
                    // `glyph_path` returns coordinates in text space, where the
                    // FreeType transform (Trm) has already been applied.  The
                    // origin (0, 0) maps to the pen position; Y is positive-up
                    // (FreeType convention).  To get device space: translate by
                    // (pen_x, pen_y) and flip Y (device Y is positive-down).
                    let path = face.glyph_path(char_code_for_path).map(|mut p| {
                        let px = f64::from(pen_x);
                        let py = f64::from(pen_y);
                        for pt in &mut p.pts {
                            pt.x += px;
                            pt.y = py - pt.y;
                        }
                        p
                    });
                    clip_paths.push(path);
                }
            };

            if let Some(enc) = cid_enc {
                // Type 0 composite font: multi-byte character codes via Encoding CMap.
                // Identity-H/V (enc.encoding_cmap == None) uses 2-byte codes by convention
                // (PDF spec §9.7.4, Table 118); embedded CMaps specify their own width.
                let code_bytes = enc.encoding_cmap.as_ref().map_or(2, |cm| cm.code_bytes);

                let mut pos = 0usize;
                while pos + usize::from(code_bytes) <= bytes.len() {
                    let mut char_code = 0u32;
                    for &b in &bytes[pos..pos + usize::from(code_bytes)] {
                        char_code = (char_code << 8) | u32::from(b);
                    }
                    pos += usize::from(code_bytes);

                    // charcode → CID (identity if no CMap), then CID → GID.
                    let cid = enc
                        .encoding_cmap
                        .as_ref()
                        .and_then(|cm| cm.map.get(&char_code).copied())
                        .unwrap_or(char_code);
                    let gid = enc.code_to_gid(char_code);

                    let (pen_x, pen_y) = text_to_device(&ctm, &tm, 0.0, rise, self.height);
                    push_glyph(char_code, gid, pen_x, pen_y);

                    // CID width lookup uses the CID, not the raw char code.
                    // Advance formula: w/1000 * font_size (PDF §9.7.4.3).
                    // Word spacing does not apply to composite fonts (PDF §9.3.3).
                    let cid_width = enc.width_for_cid(cid);
                    let advance_glyph = f64::from(cid_width) / 1000.0;
                    let tx_adv = (advance_glyph * font_size + char_spacing) * horiz_scaling;
                    let [a, b_m, c, d, e, f] = tm;
                    tm = [a, b_m, c, d, e + tx_adv * a, f + tx_adv * b_m];
                }
            } else {
                // Simple font: one byte per character code.
                for &byte in bytes {
                    let (pen_x, pen_y) = text_to_device(&ctm, &tm, 0.0, rise, self.height);
                    push_glyph(u32::from(byte), u32::from(byte), pen_x, pen_y);

                    // Advance regardless of whether the glyph rendered — PDF §9.4.4
                    // requires the pen to advance even for missing/invisible glyphs.
                    let advance_glyph = face.glyph_advance(u32::from(byte)).max(0.0);
                    let extra = if byte == b' ' { word_spacing } else { 0.0 };
                    let tx_adv = (advance_glyph * font_size + char_spacing + extra) * horiz_scaling;
                    let [a, b_m, c, d, e, f] = tm;
                    tm = [a, b_m, c, d, e + tx_adv * a, f + tx_adv * b_m];
                }
            }
        } // font_cache borrow released here

        // Phase 2: blit rasterized glyphs (skipped for modes 3 and 7).
        if do_paint {
            let fill_bytes = self.gstate.current().fill_color.as_slice().to_vec();
            let clip = self.gstate.current().clip.clone_shared();
            let pipe = Self::make_pipe(
                self.gstate.current().fill_alpha,
                self.gstate.current().blend_mode,
            );
            let src = PipeSrc::Solid(&fill_bytes);

            for rec in &records {
                #[expect(
                    clippy::cast_possible_wrap,
                    reason = "glyph dimensions are always small (sub-pixel-sized); cannot exceed i32::MAX"
                )]
                let glyph = GlyphBitmap {
                    data: &rec.data,
                    x: rec.x_off,
                    y: rec.y_off,
                    w: rec.width as i32,
                    h: rec.height as i32,
                    aa: rec.aa,
                };
                fill_glyph::<Rgb8>(
                    &mut self.bitmap,
                    &clip,
                    &pipe,
                    &src,
                    rec.pen_x,
                    rec.pen_y,
                    &glyph,
                );
            }
        }

        // Phase 3: intersect glyph outlines into the clip path (modes 4–7).
        //
        // PDF §9.3.6: glyph outlines from a single text-showing operator are
        // unioned together first, then the union is intersected with the clip.
        // We build a single accumulated path from all glyphs and call
        // `clip_to_path` once with `even_odd = false` (non-zero winding), which
        // matches Acrobat's behaviour for composite glyph clip paths.
        if do_clip {
            let mut glyph_outline = raster::path::Path::new();
            for p in clip_paths.into_iter().flatten() {
                if !p.pts.is_empty() {
                    glyph_outline.append(&p);
                }
            }
            if !glyph_outline.pts.is_empty() {
                self.clip_path_into_gstate(&glyph_outline, false);
            }
        }

        // Write updated text matrix back.
        self.gstate.current_mut().text.text_matrix = tm;
    }

    // ── Type 3 font rendering ─────────────────────────────────────────────────

    /// Execute Type 3 `CharProcs` for all character codes in `bytes`.
    ///
    /// Each `CharProc` is a PDF content stream that paints the glyph using the
    /// full graphics pipeline.  This method sets up the CTM for each glyph
    /// position and executes the stream, then advances the text matrix.
    ///
    /// The `CharProc` coordinate system (glyph space) maps to device space via:
    /// `charproc_ctm = CTM × Tm_at_glyph × scale(font_size) × FontMatrix`
    #[expect(
        clippy::too_many_arguments,
        reason = "mirrors show_text() parameter set; all values are needed for glyph placement"
    )]
    #[expect(clippy::many_single_char_names, reason = "PDF matrix components a–f")]
    fn show_text_type3(
        &mut self,
        bytes: &[u8],
        t3: &crate::resources::font::Type3Data,
        font_size: f64,
        ctm: [f64; 6],
        mut tm: [f64; 6],
        char_spacing: f64,
        word_spacing: f64,
        horiz_scaling: f64,
        rise: f64,
        do_paint: bool,
    ) {
        for &byte in bytes {
            let Some(glyph) = t3.glyph(byte) else {
                // No CharProc for this code — advance by zero width, continue.
                let [a, b_m, c, d, e, f] = tm;
                let tx_adv = char_spacing * horiz_scaling;
                tm = [a, b_m, c, d, tx_adv.mul_add(a, e), tx_adv.mul_add(b_m, f)];
                continue;
            };

            if do_paint && self.form_depth < MAX_FORM_DEPTH {
                // Build the CharProc CTM:
                //   Tm_rise = Tm with rise shift applied (PDF §9.4.4)
                //   TRM     = [fs·fm[0], fs·fm[1], fs·fm[2], fs·fm[3], tx, ty]
                //             where [tx, ty] = Tm_rise × CTM translation
                //   charproc_ctm = TRM × CTM_current
                //
                // Concretely: we insert `(font_size × FontMatrix) × Tm` as the
                // new CTM component so that the charproc's user space maps
                // correctly to device pixels.
                let fm = t3.font_matrix;
                // font_size × FontMatrix (2×2 + translation)
                let fs_fm = [
                    font_size * fm[0],
                    font_size * fm[1],
                    font_size * fm[2],
                    font_size * fm[3],
                    fm[4],
                    fm[5],
                ];
                // Apply rise to text matrix (vertical shift in text space).
                let tm_rise = if rise.abs() > f64::EPSILON {
                    let [a, b_m, c, d, e, f] = tm;
                    [a, b_m, c, d, rise.mul_add(c, e), rise.mul_add(d, f)]
                } else {
                    tm
                };
                // TRM = fs_fm × tm_rise
                let trm = ctm_multiply(&fs_fm, &tm_rise);
                // charproc_ctm = trm × CTM
                let charproc_ctm = ctm_multiply(&trm, &ctm);

                self.gstate.save();
                self.form_depth += 1;
                self.gstate.current_mut().ctm = charproc_ctm;
                // Reset text state inside charproc (PDF §9.6.5 — charproc is a
                // fresh graphics state, not a text object).
                self.gstate.current_mut().text = TextState::default();

                let ops = crate::content::parse(&glyph.content);
                self.execute(&ops);

                self.form_depth -= 1;
                self.gstate.restore();
            }

            // Advance the text matrix by the glyph width (PDF §9.4.4).
            // width_units are in glyph space; scale by FontMatrix[0] × font_size.
            let glyph_advance = f64::from(glyph.width_units) * t3.font_matrix[0];
            let extra = if byte == b' ' { word_spacing } else { 0.0 };
            let tx_adv =
                (glyph_advance.mul_add(font_size, char_spacing) + extra) * horiz_scaling;
            let [a, b_m, c, d, e, f] = tm;
            tm = [a, b_m, c, d, tx_adv.mul_add(a, e), tx_adv.mul_add(b_m, f)];
        }

        // Write updated text matrix back.
        self.gstate.current_mut().text.text_matrix = tm;
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

    // ── Tiling pattern rasterisation ──────────────────────────────────────────

    /// Rasterise the named tiling pattern resource into a [`TiledPattern`] ready
    /// for use as a [`PipeSrc::Pattern`].
    ///
    /// The tile is rendered into a small bitmap by recursively invoking
    /// `PageRenderer` on the pattern's content stream.  The tile dimensions are
    /// derived from the pattern's `XStep` / `YStep` values scaled by the
    /// combined CTM.  The pattern origin (phase) is the CTM translation component
    /// of the pattern matrix composed with the current CTM.
    ///
    /// Returns `None` if the resource is missing, the tile would be degenerate
    /// (zero or negative pixel size), or rasterisation fails.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_possible_wrap,
        reason = "tile dimensions are capped at 4096 and clamped to positive; phase coords are page-bounded; safe casts"
    )]
    fn resolve_fill_pattern(&self, name: &[u8]) -> Option<TiledPattern> {
        let desc = self.resources.tiling_pattern(name)?;

        // Compose pattern matrix with current CTM to get pattern-space → device-space.
        let ctm = self.gstate.current().ctm;
        let pat_ctm = ctm_multiply(&desc.matrix, &ctm);

        // Tile size in device pixels — scale the XStep/YStep by the linear scale
        // of the combined transform.  We use the L2 norm of each column vector to
        // account for rotations.
        let sx = (pat_ctm[0].hypot(pat_ctm[1])).abs();
        let sy = (pat_ctm[2].hypot(pat_ctm[3])).abs();

        if sx < 0.5 || sy < 0.5 {
            log::debug!(
                "pdf_interp: Pattern /{} tile scale ({sx:.2}, {sy:.2}) too small — skipping",
                String::from_utf8_lossy(name)
            );
            return None;
        }

        // Cap the tile to MAX_TILE_PX to avoid huge allocations from malformed PDFs.
        let tile_w = (desc.x_step.abs() * sx).min(MAX_TILE_PX).ceil() as u32;
        let tile_h = (desc.y_step.abs() * sy).min(MAX_TILE_PX).ceil() as u32;

        if tile_w == 0 || tile_h == 0 {
            log::debug!(
                "pdf_interp: Pattern /{} produced zero-size tile ({tile_w}×{tile_h}) — skipping",
                String::from_utf8_lossy(name)
            );
            return None;
        }

        // Build the CTM for the tile renderer: pattern space → tile device space.
        // Translation is handled by phase below, so we zero it here.
        let tile_ctm = [pat_ctm[0], pat_ctm[1], pat_ctm[2], pat_ctm[3], 0.0, 0.0];

        // Rasterise the tile.
        let tile_bitmap = self.render_pattern_tile(&desc, tile_w, tile_h, &tile_ctm);

        // Phase = where the pattern origin lands in device space (after y-flip).
        let page_h = f64::from(self.height);
        let (ox, oy_pdf) = ctm_transform(&pat_ctm, 0.0, 0.0);
        let oy = page_h - oy_pdf;

        let phase_x = ox.round() as i32;
        let phase_y = oy.round() as i32;

        let pixels = tile_bitmap.data().to_vec();
        Some(TiledPattern::new(
            pixels,
            tile_w as i32,
            tile_h as i32,
            phase_x,
            phase_y,
        ))
    }

    /// Render a pattern content stream into a tile bitmap.
    ///
    /// Creates a child `PageRenderer` scoped to the pattern's resource context and
    /// executes the pattern's content stream.  The tile bitmap is `tile_w ×
    /// tile_h` pixels with a white background.
    fn render_pattern_tile(
        &self,
        desc: &crate::resources::tiling::TilingDescriptor,
        tile_w: u32,
        tile_h: u32,
        tile_ctm: &[f64; 6],
    ) -> Bitmap<Rgb8> {
        // Build a temporary child renderer sharing the same document / font engine.
        let doc = self.resources.doc();
        let mut tile_renderer = PageRenderer::new(tile_w, tile_h, doc, desc.stream_id);

        // Override the CTM to map pattern space → tile device space.
        tile_renderer.gstate.current_mut().ctm = *tile_ctm;

        // If the pattern has no own resources, point it at the parent context.
        if !desc.has_own_resources {
            tile_renderer.resources = PageResources::new(doc, self.resources.resource_context_id());
        }

        let ops = crate::content::parse(&desc.content);
        tile_renderer.execute(&ops);
        tile_renderer.finish()
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
            if corners.iter().any(|(x, y)| !x.is_finite() || !y.is_finite()) {
                log::warn!("pdf_interp: transparency group BBox is non-finite — rendering without group");
                return None;
            }
            let left   = corners.iter().map(|(x, _)| *x).fold(f64::INFINITY, f64::min).floor();
            let top    = corners.iter().map(|(_, y)| page_h - y).fold(f64::INFINITY, f64::min).floor();
            let right  = corners.iter().map(|(x, _)| *x).fold(f64::NEG_INFINITY, f64::max).ceil();
            let bottom = corners.iter().map(|(_, y)| page_h - y).fold(f64::NEG_INFINITY, f64::max).ceil();

            #[expect(clippy::cast_possible_truncation, reason = "device coords bounded by page dimensions")]
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
    #[expect(clippy::many_single_char_names, reason = "PDF CTM components a–f are standard")]
    #[expect(
        clippy::similar_names,
        reason = "dx_rel / dy_rel are the standard names for the per-axis deltas in the inverse CTM formula"
    )]
    fn blit_image(&mut self, img: &crate::resources::ImageDescriptor) {
        // Degenerate image — nothing to blit.
        if img.width == 0 || img.height == 0 {
            return;
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

        let data = self.bitmap.data_mut();
        let stride = self.width as usize * 3; // Rgb8: 3 bytes per pixel

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

                // Nearest-neighbour sample.  Clamp so ix < img.width, iy < img.height.
                // u maps to image column (left→right);
                // v maps to image row, flipped: v=0 is top, v=1 is bottom (PDF origin=bottom-left).
                let ix = (u * img_w).min(img_w - 1.0) as usize;
                let iy = ((1.0 - v) * img_h).min(img_h - 1.0) as usize;
                let img_idx = iy * img.width as usize + ix;

                // If a soft mask is present, skip fully-transparent pixels.
                // Out-of-bounds access (smask shorter than expected) defaults to
                // 0xFF (opaque) so a truncated mask never silently erases pixels.
                if img
                    .smask
                    .as_deref()
                    .is_some_and(|s| s.get(img_idx).copied().unwrap_or(0xFF) == 0)
                {
                    continue;
                }

                // Safety: bx0..bx1 and by0..by1 are clamped to bitmap bounds above,
                // so pixel_off is always in range for a valid Rgb8 bitmap.
                let pixel_off = dy as usize * stride + dx as usize * 3;

                match img.color_space {
                    ImageColorSpace::Rgb => {
                        let src = img_idx * 3;
                        // img.data length is validated in decode_raw.
                        if let Some(rgb) = img.data.get(src..src + 3) {
                            data[pixel_off..pixel_off + 3].copy_from_slice(rgb);
                        }
                    }
                    ImageColorSpace::Gray => {
                        if let Some(&v) = img.data.get(img_idx) {
                            data[pixel_off] = v;
                            data[pixel_off + 1] = v;
                            data[pixel_off + 2] = v;
                        }
                    }
                    ImageColorSpace::Mask => {
                        // 0x00 = paint with fill colour; any other value = transparent.
                        if img.data.get(img_idx) == Some(&0x00) {
                            data[pixel_off..pixel_off + 3].copy_from_slice(&fill_color);
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

/// Read a 4-element rect `[llx, lly, urx, ury]` from a PDF dictionary.
///
/// Returns `None` if the `Rect` key is absent or has fewer than 4 numeric entries.
fn read_rect(dict: &lopdf::Dictionary) -> Option<[f64; 4]> {
    let mut r = crate::resources::read_f64_n::<4>(dict, b"Rect")?;
    // Normalise so llx ≤ urx and lly ≤ ury.
    if r[0] > r[2] {
        r.swap(0, 2);
    }
    if r[1] > r[3] {
        r.swap(1, 3);
    }
    Some(r)
}
