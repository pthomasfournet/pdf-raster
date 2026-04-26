//! Per-page operator dispatcher.
//!
//! [`PageRenderer`] holds a target [`Bitmap`] and a [`GStateStack`], iterates
//! over a decoded operator sequence, and calls into the `raster` crate for
//! each painting operator.
//!
//! # What is implemented
//!
//! - Graphics state: `q Q cm w J j M d ri i`
//! - Path construction: `m l c v y h re`
//! - Path painting: `S s f F f* B B* b b* n`
//! - Colour: `g G rg RG k K sc scn SC SCN cs CS`
//! - Text objects + state: `BT ET Tf Tc Tw Tz TL Ts Tr Td TD Tm T*`
//! - Text showing: `Tj TJ ' "` (via `FreeType` through font crate)
//!
//! # Not yet implemented
//!
//! - Image `XObjects`: `DCTDecode` (JPEG), `JBIG2Decode`, `JPXDecode`
//! - Form `XObjects` — recursive content streams
//! - Shading (`sh`) — requires shading resource lookup
//! - Extended graphics state (`gs`) — requires `ExtGState` dict lookup
//! - Clip paths (W W*) — stub; page-rect clip used as fallback
//! - Char-to-glyph Differences encoding (phase 2)
//! - Type 0 / `CIDFont` composite fonts (phase 2)
//! - Type 3 paint-procedure fonts (phase 2)

use lopdf::{Document, ObjectId};

use color::Rgb8;
use font::{
    cache::GlyphCache,
    engine::{FontEngine, SharedEngine},
};
use raster::{
    Bitmap, Clip, eo_fill, fill,
    glyph::{GlyphBitmap, fill_glyph},
    path::PathBuilder,
    pipe::{PipeSrc, PipeState},
    state::TransferSet,
    stroke::{StrokeParams, stroke},
    types::{BlendMode, LineCap, LineJoin},
};

use super::color::RasterColor;
use super::font_cache::FontCache;
use super::gstate::{GStateStack, InterpGState, ctm_multiply, ctm_transform, mat2x2_mul};
use crate::content::{Operator, TextArrayElement};
use crate::resources::{ImageColorSpace, PageResources};

/// Identity CTM for passing to raster functions — coordinate transform is
/// already baked into the path points by `to_device`.
const DEVICE_MATRIX: [f64; 6] = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];

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
    /// Font face cache for this page.
    font_cache: FontCache,
    /// Resource accessor for the current page.
    resources: PageResources<'doc>,
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

        let mut gstate = GStateStack::new();
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
            font_cache,
            resources,
        }
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
            // Rendering intent and ExtGState require resource dict access —
            // deferred until the resource resolver is wired in.
            Operator::SetRenderingIntent(_) | Operator::SetExtGState(_) => {}

            // ── Colour ────────────────────────────────────────────────────────
            Operator::SetFillGray(g) => self.set_fill(RasterColor::gray(*g)),
            Operator::SetFillRgb(r, g, b) => self.set_fill(RasterColor::rgb(*r, *g, *b)),
            Operator::SetFillCmyk(c, m, y, k) => self.set_fill(RasterColor::cmyk(*c, *m, *y, *k)),
            Operator::SetFillColor(comps) => self.set_fill(components_to_color(comps)),
            Operator::SetFillColorSpace(_) => {}

            Operator::SetStrokeGray(g) => self.set_stroke(RasterColor::gray(*g)),
            Operator::SetStrokeRgb(r, g, b) => self.set_stroke(RasterColor::rgb(*r, *g, *b)),
            Operator::SetStrokeCmyk(c, m, y, k) => {
                self.set_stroke(RasterColor::cmyk(*c, *m, *y, *k));
            }
            Operator::SetStrokeColor(comps) => self.set_stroke(components_to_color(comps)),
            Operator::SetStrokeColorSpace(_) => {}

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
                let cp = self.path_builder().cur_pt().map(|p| (p.x, p.y));
                if cp.is_none() {
                    log::debug!("pdf_interp: CurveToV with no current point — operator ignored");
                    return;
                }
                let (cpx, cpy) = cp.expect("checked above");
                let (dx2, dy2) = self.to_device(*x2, *y2);
                let (dx3, dy3) = self.to_device(*x3, *y3);
                let _ = self.path_builder().curve_to(cpx, cpy, dx2, dy2, dx3, dy3);
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
                self.path = None;
            }

            // ── Clipping (stub — page-rect clip used until path-clip is wired) ─
            Operator::Clip | Operator::ClipEvenOdd => {
                // TODO(phase2): build a path-based clip from self.path.
                self.path = None;
            }

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
                let bytes = bytes.clone();
                self.show_text(&bytes);
            }
            Operator::MoveNextLineShowSpaced { aw, ac, text, .. } => {
                let gs = self.gstate.current_mut();
                gs.text.word_spacing = *aw;
                gs.text.char_spacing = *ac;
                gs.text.next_line();
                let text = text.clone();
                self.show_text(&text);
            }

            // ── XObjects / images / shading ────────────────────────────────────
            Operator::PaintXObject(name) => {
                self.do_xobject(name);
            }
            Operator::InlineImage { .. } => {
                log::debug!("pdf_interp: inline image not yet implemented");
            }
            Operator::PaintShading(name) => {
                log::debug!(
                    "pdf_interp: sh /{} not yet implemented",
                    String::from_utf8_lossy(name)
                );
            }

            // ── No-ops ────────────────────────────────────────────────────────
            Operator::MarkedContent | Operator::CompatibilitySection => {}

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
    fn show_text(&mut self, bytes: &[u8]) {
        if bytes.is_empty() {
            return;
        }

        // Render modes 3+ are invisible (clip or no-op for our purposes).
        if self.gstate.current().text.render_mode >= 3 {
            return;
        }

        // Snapshot all state we need before any mutable borrow of font_cache.
        let font_name = self.gstate.current().text.font_name.clone();
        let font_size = self.gstate.current().text.font_size;
        let char_spacing = self.gstate.current().text.char_spacing;
        let word_spacing = self.gstate.current().text.word_spacing;
        // PDF Tz is a percentage; 0 % means zero-advance (degenerate — treat as 100 %).
        let raw_hz = self.gstate.current().text.horiz_scaling;
        let horiz_scaling = if raw_hz.abs() < f64::EPSILON {
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

        // Phase 1: rasterize all glyphs.
        // We collect (pen_x, pen_y, GlyphBitmap data) as owned Vecs so that the
        // `font_cache` borrow is released before we touch `self.bitmap`.
        let mut records: Vec<GlyphRecord> = Vec::with_capacity(bytes.len());

        {
            // Trm[2×2] = font_size × Tm[2×2] × CTM[2×2] — the text rendering
            // matrix in device pixels, encoding actual size and skew/rotation.
            let tm2x2 = mat2x2_mul(&tm, &ctm);
            let trm = tm2x2.map(|v| v * font_size);

            // Load (or retrieve cached) FreeType face — mutable borrow of font_cache.
            let Some(face) = self.font_cache.get_or_load(&font_name, &descriptor, trm) else {
                return;
            };

            for &byte in bytes {
                let (pen_x, pen_y) = text_to_device(&ctm, &tm, 0.0, rise, self.height);

                if let Some(bmp) = face.make_glyph(u32::from(byte), 0) {
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

                // Advance text matrix.
                let advance_glyph = face.glyph_advance(u32::from(byte)).max(0.0);
                let extra = if byte == b' ' { word_spacing } else { 0.0 };
                let tx_adv = (advance_glyph * font_size + char_spacing + extra) * horiz_scaling;
                let [a, b_m, c, d, e, f] = tm;
                tm = [a, b_m, c, d, e + tx_adv * a, f + tx_adv * b_m];
            }
        } // font_cache borrow released here

        // Phase 2: blit rasterized glyphs — mutable borrow of bitmap only.
        let fill_bytes = self.gstate.current().fill_color.as_slice().to_vec();
        let clip = self.page_clip();
        let pipe = Self::make_pipe(self.gstate.current());
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

        // Write updated text matrix back.
        self.gstate.current_mut().text.text_matrix = tm;
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    fn set_fill(&mut self, c: RasterColor) {
        self.gstate.current_mut().fill_color = c;
    }

    fn set_stroke(&mut self, c: RasterColor) {
        self.gstate.current_mut().stroke_color = c;
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

    fn page_clip(&self) -> Clip {
        Clip::new(
            0.0,
            0.0,
            f64::from(self.width),
            f64::from(self.height),
            false,
        )
    }

    /// Build a [`PipeState`] for Normal-blend, fully-opaque, identity-transfer
    /// rendering.  Extended graphics state fields (opacity, soft mask, etc.) are
    /// Phase-2 work; the parameter is reserved for when we read `ExtGState` dicts.
    fn make_pipe(_gs: &InterpGState) -> PipeState<'static> {
        PipeState {
            blend_mode: BlendMode::Normal,
            a_input: 255,
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
            return;
        };
        self.fill_path(&builder.build(), even_odd);
    }

    fn do_stroke(&mut self) {
        let Some(builder) = self.path.take() else {
            return;
        };
        self.stroke_path(&builder.build());
    }

    fn do_fill_then_stroke(&mut self, even_odd: bool) {
        let Some(builder) = self.path.take() else {
            return;
        };
        let path = builder.build();
        self.fill_path(&path, even_odd);
        self.stroke_path(&path);
    }

    /// Fill `path` using the current fill colour.  Shared by `do_fill` and
    /// `do_fill_then_stroke`.
    fn fill_path(&mut self, path: &raster::path::Path, even_odd: bool) {
        let gs = self.gstate.current();
        let color = gs.fill_color.as_slice().to_vec();
        let clip = self.page_clip();
        let flatness = gs.flatness.max(0.1);
        let pipe = Self::make_pipe(gs);
        let src = PipeSrc::Solid(&color);
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
    }

    fn stroke_path(&mut self, path: &raster::path::Path) {
        let gs = self.gstate.current();
        let color = gs.stroke_color.as_slice().to_vec();
        let clip = self.page_clip();
        let pipe = Self::make_pipe(gs);
        let src = PipeSrc::Solid(&color);
        let params = StrokeParams {
            line_width: gs.line_width,
            line_cap: gs.line_cap,
            line_join: gs.line_join,
            miter_limit: gs.miter_limit,
            flatness: gs.flatness.max(0.1),
            line_dash: &gs.dash.0,
            line_dash_phase: gs.dash.1,
            stroke_adjust: false,
            vector_antialias: true,
        };
        stroke(
            &mut self.bitmap,
            &clip,
            path,
            &pipe,
            &src,
            &DEVICE_MATRIX,
            &params,
        );
    }

    // ── XObject rendering ─────────────────────────────────────────────────────

    /// Execute a `Do` operator: look up and render the named `XObject`.
    ///
    /// Only image `XObject`s are implemented; form `XObject`s are logged and
    /// skipped until recursive content-stream execution is added.
    fn do_xobject(&mut self, name: &[u8]) {
        let Some(img) = self.resources.image(name) else {
            // Possible causes: Form XObject (not yet implemented), unsupported
            // filter (DCTDecode / JBIG2 / JPX), or a missing resource entry.
            log::debug!(
                "pdf_interp: Do /{} skipped (Form XObject, unsupported filter, or missing resource)",
                String::from_utf8_lossy(name)
            );
            return;
        };
        self.blit_image(&img);
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
        clippy::cast_precision_loss,
        reason = "device pixel coords are always in page bounds after clamping; safe casts"
    )]
    fn blit_image(&mut self, img: &crate::resources::ImageDescriptor) {
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

        // For each output pixel, compute image-space coordinates by
        // inverse-mapping from bounding-box coordinates.  Exact for
        // axis-aligned transforms; approximate otherwise.
        let img_w = f64::from(img.width);
        let img_h = f64::from(img.height);
        let span_x = (dx1 - dx0).max(1) as f64;
        let span_y = (dy1 - dy0).max(1) as f64;
        let origin_x = dx0 as f64;
        let origin_y = dy0 as f64;

        let data = self.bitmap.data_mut();
        let stride = self.width as usize * 3; // Rgb8: 3 bytes per pixel

        for dy in by0..by1 {
            for dx in bx0..bx1 {
                // tx ∈ [0,1] left→right; ty ∈ [0,1] bottom→top (PDF convention).
                let tx = ((f64::from(dx) - origin_x) / span_x).clamp(0.0, 1.0);
                let ty = 1.0 - ((f64::from(dy) - origin_y) / span_y).clamp(0.0, 1.0);

                // Nearest-neighbour sample.  Clamp so ix < img.width, iy < img.height.
                let ix = (tx * img_w).min(img_w - 1.0).max(0.0) as usize;
                let iy = ((1.0 - ty) * img_h).min(img_h - 1.0).max(0.0) as usize;
                let img_idx = iy * img.width as usize + ix;

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

// ── Glyph record ─────────────────────────────────────────────────────────────

/// Holds rasterized glyph data for two-phase text rendering.
struct GlyphRecord {
    pen_x: i32,
    pen_y: i32,
    x_off: i32,
    y_off: i32,
    width: u32,
    height: u32,
    aa: bool,
    data: Vec<u8>,
}

// ── Coordinate helpers ────────────────────────────────────────────────────────

/// Map a text-space point through the text matrix and CTM to device-pixel
/// coordinates with y-flip (PDF origin = bottom-left; device origin = top-left).
///
/// `tm[6]` is `[a, b, c, d, e, f]` in PDF column-major form.
/// The full mapping is: `device = CTM × Tm × (tx, ty+rise)`.
#[expect(clippy::many_single_char_names, reason = "PDF matrix components")]
fn text_to_device(ctm: &[f64; 6], tm: &[f64; 6], tx: f64, ty: f64, page_h: u32) -> (i32, i32) {
    // Apply text matrix: user_space = Tm * (tx, ty).
    let [a, b, c, d, e, f] = *tm;
    let ux = a.mul_add(tx, c * ty) + e;
    let uy = b.mul_add(tx, d * ty) + f;

    // Apply CTM: device = CTM * (ux, uy).
    let (dx, dy) = ctm_transform(ctm, ux, uy);

    // y-flip.
    let dy_flipped = f64::from(page_h) - dy;

    #[expect(
        clippy::cast_possible_truncation,
        reason = "device pixels are always in [0, page_h|w]; round() output fits i32 for any real page"
    )]
    (dx.round() as i32, dy_flipped.round() as i32)
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
