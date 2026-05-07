//! PDF operator dispatch — `execute_one`.

use super::super::color::RasterColor;
use super::super::gstate::ctm_multiply;
use super::{PageRenderer, components_to_color, int_to_cap, int_to_join};
use crate::content::{Operator, TextArrayElement};

impl PageRenderer<'_> {
    #[expect(clippy::too_many_lines, reason = "operator dispatch table")]
    #[expect(
        clippy::match_same_arms,
        reason = "intentional stubs for unimplemented operators"
    )]
    #[expect(
        clippy::many_single_char_names,
        reason = "PDF matrix components and PDF spec variable names"
    )]
    pub(super) fn execute_one(&mut self, op: &Operator) {
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
                // Unmatched EMC is tolerated to match real-world PDF readers;
                // the popped visibility flag is intentionally discarded.
                let _ = self.ocg_stack.pop();
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
                if let Err(e) = self.path_builder().move_to(dx, dy) {
                    log::debug!("pdf_interp: MoveTo failed: {e}");
                }
            }
            Operator::LineTo(x, y) => {
                let (dx, dy) = self.to_device(*x, *y);
                if let Err(e) = self.path_builder().line_to(dx, dy) {
                    log::debug!("pdf_interp: LineTo failed: {e}");
                }
            }
            Operator::CurveTo(x1, y1, x2, y2, x3, y3) => {
                let (dx1, dy1) = self.to_device(*x1, *y1);
                let (dx2, dy2) = self.to_device(*x2, *y2);
                let (dx3, dy3) = self.to_device(*x3, *y3);
                if let Err(e) = self.path_builder().curve_to(dx1, dy1, dx2, dy2, dx3, dy3) {
                    log::debug!("pdf_interp: CurveTo failed: {e}");
                }
            }
            Operator::CurveToV(x2, y2, x3, y3) => {
                // `v`: first control point = current point.
                let Some(cp) = self.path_builder().cur_pt() else {
                    log::debug!("pdf_interp: CurveToV with no current point — operator ignored");
                    return;
                };
                let (dx2, dy2) = self.to_device(*x2, *y2);
                let (dx3, dy3) = self.to_device(*x3, *y3);
                if let Err(e) = self.path_builder().curve_to(cp.x, cp.y, dx2, dy2, dx3, dy3) {
                    log::debug!("pdf_interp: CurveToV failed: {e}");
                }
            }
            Operator::CurveToY(x1, y1, x3, y3) => {
                // `y`: second control point = endpoint.
                let (dx1, dy1) = self.to_device(*x1, *y1);
                let (dx3, dy3) = self.to_device(*x3, *y3);
                if let Err(e) = self.path_builder().curve_to(dx1, dy1, dx3, dy3, dx3, dy3) {
                    log::debug!("pdf_interp: CurveToY failed: {e}");
                }
            }
            Operator::ClosePath => {
                if let Some(b) = self.path.as_mut()
                    && let Err(e) = b.close(false)
                {
                    log::debug!("pdf_interp: ClosePath failed: {e}");
                }
            }
            Operator::Rectangle(x, y, w, h) => {
                // PDF §8.5.2.1: `re` defines four corners in user space.
                // All four must be independently transformed — mixing device-space
                // coordinates from two corners produces the wrong shape under a
                // sheared CTM.
                let (x0, y0) = self.to_device(*x, *y);
                let (x1, y1) = self.to_device(*x + *w, *y);
                let (x2, y2) = self.to_device(*x + *w, *y + *h);
                let (x3, y3) = self.to_device(*x, *y + *h);
                let b = self.path_builder();
                // `re` is self-contained; individual segment failures are benign —
                // a partial rectangle is better than a panic on a malformed PDF.
                let _ = b.move_to(x0, y0);
                let _ = b.line_to(x1, y1);
                let _ = b.line_to(x2, y2);
                let _ = b.line_to(x3, y3);
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
                            // PDF §9.4.3: kern is in thousandths of a text-space unit.
                            // Displacement = -(w/1000) * Th * Tfs, where Th is horizontal
                            // scaling (Tz as a fraction, default 1.0) and Tfs is font size.
                            // Tz=0 is degenerate (invisible text); treat as 1.0 like show_text.
                            let ts = &mut self.gstate.current_mut().text;
                            let hz = if ts.horiz_scaling.abs() < f64::EPSILON {
                                1.0
                            } else {
                                ts.horiz_scaling / 100.0
                            };
                            let shift = -kern / 1000.0 * ts.font_size * hz;
                            let [a, b, c, d, e, f] = ts.text_matrix;
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
                if let Some(img) = crate::resources::image::decode_inline_image(
                    self.resources.doc(),
                    params,
                    data,
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
                ) {
                    self.blit_image(&img);
                } else {
                    log::warn!("pdf_interp: inline image decode failed — skipping");
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
}
