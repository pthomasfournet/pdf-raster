//! Text rendering — `show_text` and `show_text_type3`.

use super::super::gstate::{ctm_multiply, mat2x2_mul};
use super::super::text::TextState;
use super::MAX_FORM_DEPTH;
use super::PageRenderer;
use super::text_ops::{GlyphRecord, text_to_device};
use color::Rgb8;
use raster::{
    glyph::{GlyphBitmap, fill_glyph},
    pipe::PipeSrc,
};

impl PageRenderer<'_> {
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
    pub(super) fn show_text(&mut self, bytes: &[u8]) {
        if bytes.is_empty() {
            return;
        }
        self.diag.has_vector_text = true;

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
            // Tz=0 % is degenerate (zero-width advance = invisible text).
            // Substitute 100 % so text remains visible; warn because this
            // overrides explicit PDF authoring intent.
            log::warn!("pdf_interp: Tz is 0 %, substituting 100 % to avoid invisible text");
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
                do_clip,
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
            //
            // `gid` is the resolved glyph index passed to `make_glyph` directly.
            // `char_code_for_path` is the raw character code passed to `glyph_path`,
            // which calls `resolve_gid` internally — so it must be the encoding-level
            // code, not a pre-resolved GID.
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
                // PDF §9.2.4: the Widths array is the authoritative advance source
                // for simple fonts (within [FirstChar, LastChar]); MissingWidth
                // applies out-of-range.  FreeType's horiAdvance is consulted ONLY
                // when Widths is absent entirely (e.g. the 14 standard fonts
                // pre-PDF 1.5, where a substitute face supplies metrics).
                for &byte in bytes {
                    let (pen_x, pen_y) = text_to_device(&ctm, &tm, 0.0, rise, self.height);
                    push_glyph(u32::from(byte), u32::from(byte), pen_x, pen_y);

                    // PDF §9.4.4: pen advances even for missing/invisible glyphs.
                    let advance_glyph = descriptor.width_for_code(u32::from(byte)).map_or_else(
                        || face.glyph_advance(u32::from(byte)).unwrap_or(0.0),
                        |w| f64::from(w) / 1000.0,
                    );
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
                // The ClipResult return value is only relevant for accumulating
                // text-as-clip paths (text render modes 4–7), which Phase 3 below
                // handles separately via glyph_path; per-glyph fill output is unused.
                let _ = fill_glyph::<Rgb8>(
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
    ///
    /// `CharProcs` are re-rasterised on every call.  If Type 3 perf becomes a
    /// hot path, the right fix is to extend `font::GlyphCache` to key on Type
    /// 3 font instances and add an off-screen bitmap path here, not to
    /// reintroduce a separate Type-3-only cache.
    #[expect(
        clippy::too_many_arguments,
        reason = "mirrors show_text() parameter set; all values are needed for glyph placement"
    )]
    #[expect(clippy::many_single_char_names, reason = "PDF matrix components a–f")]
    pub(super) fn show_text_type3(
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
        do_clip: bool,
    ) {
        if do_clip {
            log::warn!(
                "pdf_interp: text-clip mode (render_mode ≥ 4) is not supported for Type 3 \
                 fonts — clip ignored"
            );
        }
        for &byte in bytes {
            let Some(glyph) = t3.glyph(byte) else {
                // No CharProc for this code — advance by zero width, continue.
                let [a, b_m, c, d, e, f] = tm;
                let tx_adv = char_spacing * horiz_scaling;
                tm = [a, b_m, c, d, tx_adv.mul_add(a, e), tx_adv.mul_add(b_m, f)];
                continue;
            };

            if do_paint && self.form_depth >= MAX_FORM_DEPTH {
                log::warn!(
                    "pdf_interp: Type 3 CharProc depth {MAX_FORM_DEPTH} exceeded — \
                     glyph 0x{byte:02X} not painted (text position still advances)"
                );
            } else if do_paint {
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
            let tx_adv = (glyph_advance.mul_add(font_size, char_spacing) + extra) * horiz_scaling;
            let [a, b_m, c, d, e, f] = tm;
            tm = [a, b_m, c, d, tx_adv.mul_add(a, e), tx_adv.mul_add(b_m, f)];
        }

        // Write updated text matrix back.
        self.gstate.current_mut().text.text_matrix = tm;
    }
}
