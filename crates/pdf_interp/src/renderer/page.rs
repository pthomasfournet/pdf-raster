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
//! - Text showing: `Tj TJ ' "` (via FreeType through font crate)
//!
//! # Not yet implemented
//!
//! - XObjects / inline images — requires XObject resource resolver
//! - Shading (`sh`) — requires shading resource lookup
//! - Extended graphics state (`gs`) — requires ExtGState dict lookup
//! - Clip paths (W W*) — stub; page-rect clip used as fallback
//! - Char-to-glyph Differences encoding (phase 2)
//! - Type 0 / CIDFont composite fonts (phase 2)
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
use super::gstate::{GStateStack, InterpGState, ctm_multiply, ctm_transform};
use crate::content::{Operator, TextArrayElement};
use crate::resources::PageResources;

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
    #[must_use]
    pub fn new_scaled(
        width: u32,
        height: u32,
        scale: f64,
        doc: &'doc Document,
        page_id: ObjectId,
    ) -> Self {
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
                self.set_stroke(RasterColor::cmyk(*c, *m, *y, *k))
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
                let cp = self
                    .path_builder()
                    .cur_pt()
                    .map(|p| (p.x, p.y))
                    .unwrap_or((0.0, 0.0));
                let (dx2, dy2) = self.to_device(*x2, *y2);
                let (dx3, dy3) = self.to_device(*x3, *y3);
                let _ = self.path_builder().curve_to(cp.0, cp.1, dx2, dy2, dx3, dy3);
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
                ts.font_name = name.clone();
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

            // ── XObjects / images / shading (resource resolver pending) ───────
            Operator::PaintXObject(name) => {
                log::debug!(
                    "pdf_interp: Do /{} not yet implemented",
                    String::from_utf8_lossy(name)
                );
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
            #[allow(unreachable_patterns)]
            _ => {}
        }
    }

    // ── Text rendering ────────────────────────────────────────────────────────

    /// Render a byte string using the current font, colour, and text matrix.
    ///
    /// Two-phase design to satisfy the borrow checker:
    /// 1. Rasterize all glyphs while holding the `font_cache` borrow (no bitmap access).
    /// 2. Blit the pre-rasterized bitmaps, using `bitmap` mutably (no font_cache access).
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
        let horiz_scaling = self.gstate.current().text.horiz_scaling / 100.0;
        let ctm = self.gstate.current().ctm;
        let rise = self.gstate.current().text.rise;
        let mut tm = self.gstate.current().text.text_matrix;

        // Resolve font descriptor (immutable borrow of resources).
        let descriptor = match self.resources.font_dict(&font_name) {
            Some(d) => d,
            None => {
                log::debug!(
                    "pdf_interp: no font dict for /{}",
                    String::from_utf8_lossy(&font_name)
                );
                return;
            }
        };

        // Phase 1: rasterize all glyphs.
        // We collect (pen_x, pen_y, GlyphBitmap data) as owned Vecs so that the
        // `font_cache` borrow is released before we touch `self.bitmap`.
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
        let mut records: Vec<GlyphRecord> = Vec::with_capacity(bytes.len());

        {
            // Load (or retrieve cached) FreeType face — mutable borrow of font_cache.
            let face = match self
                .font_cache
                .get_or_load(&font_name, &descriptor, font_size)
            {
                Some(f) => f,
                None => return,
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
        let (_, pipe) = Self::make_pipe(self.gstate.current());
        let src = PipeSrc::Solid(&fill_bytes);

        for rec in &records {
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

    fn make_pipe(_gs: &InterpGState) -> (TransferSet<'static>, PipeState<'static>) {
        let transfer = TransferSet::identity_rgb();
        let pipe = PipeState {
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
        };
        (transfer, pipe)
    }

    fn do_fill(&mut self, even_odd: bool) {
        let Some(builder) = self.path.take() else {
            return;
        };
        let path = builder.build();
        let gs = self.gstate.current();
        let color = gs.fill_color.as_slice().to_vec();
        let clip = self.page_clip();
        let flatness = gs.flatness.max(0.1);
        let (_, pipe) = Self::make_pipe(gs);
        let src = PipeSrc::Solid(&color);
        if even_odd {
            eo_fill(
                &mut self.bitmap,
                &clip,
                &path,
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
                &path,
                &pipe,
                &src,
                &DEVICE_MATRIX,
                flatness,
                true,
            );
        }
    }

    fn do_stroke(&mut self) {
        let Some(builder) = self.path.take() else {
            return;
        };
        let path = builder.build();
        self.stroke_path(&path);
    }

    fn do_fill_then_stroke(&mut self, even_odd: bool) {
        let Some(builder) = self.path.take() else {
            return;
        };
        let path = builder.build();

        let gs = self.gstate.current();
        let fill_color = gs.fill_color.as_slice().to_vec();
        let clip = self.page_clip();
        let flatness = gs.flatness.max(0.1);
        let (_, pipe) = Self::make_pipe(gs);
        let src = PipeSrc::Solid(&fill_color);
        if even_odd {
            eo_fill(
                &mut self.bitmap,
                &clip,
                &path,
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
                &path,
                &pipe,
                &src,
                &DEVICE_MATRIX,
                flatness,
                true,
            );
        }

        self.stroke_path(&path);
    }

    fn stroke_path(&mut self, path: &raster::path::Path) {
        let gs = self.gstate.current();
        let color = gs.stroke_color.as_slice().to_vec();
        let clip = self.page_clip();
        let (_, pipe) = Self::make_pipe(gs);
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
}

// ── Coordinate helpers ────────────────────────────────────────────────────────

/// Map a text-space point through the text matrix and CTM to device-pixel
/// coordinates with y-flip (PDF origin = bottom-left; device origin = top-left).
///
/// `tm[6]` is `[a, b, c, d, e, f]` in PDF column-major form.
/// The full mapping is: `device = CTM × Tm × (tx, ty+rise)`.
fn text_to_device(ctm: &[f64; 6], tm: &[f64; 6], tx: f64, ty: f64, page_h: u32) -> (i32, i32) {
    // Apply text matrix: user_space = Tm * (tx, ty).
    let [a, b, c, d, e, f] = *tm;
    let ux = a * tx + c * ty + e;
    let uy = b * tx + d * ty + f;

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
