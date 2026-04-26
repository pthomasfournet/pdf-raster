//! Glyph bitmap blitting — replaces `Splash::fillGlyph2`.
//!
//! A glyph bitmap is either:
//! - **AA** (`aa = true`): one byte per pixel, containing a coverage value 0–255.
//! - **Mono** (`aa = false`): 1-bit-per-pixel, MSB-first packed, `ceil(w/8)` bytes per row.
//!
//! The blit clips to the destination bitmap bounds and optionally to the `Clip` region.
//!
//! # C++ equivalent
//! `Splash::fillGlyph2` (Splash.cc ~line 2643).

use crate::bitmap::Bitmap;
use crate::clip::{Clip, ClipResult};
use crate::pipe::{self, PipeSrc, PipeState};
use crate::simd;
use color::Pixel;

/// A rasterized glyph bitmap as produced by a font renderer.
///
/// The origin `(x, y)` is the pen-position offset in device pixels:
/// the top-left of the bitmap maps to `(pen_x - x, pen_y - y)`.
pub struct GlyphBitmap<'a> {
    /// Raw bitmap data. For AA glyphs: `w` bytes per row. For mono: `ceil(w/8)` bytes per row.
    pub data: &'a [u8],
    /// Horizontal offset from the pen position to the left edge of the bitmap.
    pub x: i32,
    /// Vertical offset from the pen position to the top edge of the bitmap.
    pub y: i32,
    /// Width in pixels.
    pub w: i32,
    /// Height in pixels.
    pub h: i32,
    /// `true` for anti-aliased (one u8 coverage byte per pixel);
    /// `false` for 1-bit MSB-first packed mono.
    pub aa: bool,
}

impl GlyphBitmap<'_> {
    /// Number of bytes per row in `data`.
    #[must_use]
    pub fn row_bytes(&self) -> usize {
        let w = self.w.max(0);
        if self.aa {
            #[expect(clippy::cast_sign_loss, reason = "w = self.w.max(0) is non-negative")]
            {
                w as usize
            }
        } else {
            // Use saturating_add so a pathological w near i32::MAX doesn't wrap.
            // In practice FreeType glyphs are always < 2^16 px wide.
            #[expect(clippy::cast_sign_loss, reason = "w = self.w.max(0) is non-negative")]
            {
                (w.saturating_add(7) / 8) as usize
            }
        }
    }
}

/// Blit a glyph at pen position `(pen_x, pen_y)` (device pixel coordinates).
///
/// `clip_all_inside` — if `true`, skip per-pixel clip tests (the caller has
/// already determined the entire glyph bbox is inside the clip region).
/// When `false`, the `clip` is tested per pixel.
///
/// The `pipe` and `src` describe the fill colour and compositing parameters.
///
/// # C++ equivalent
///
/// Matches `Splash::fillGlyph2` with `noClip = clip_all_inside`.
#[expect(
    clippy::too_many_arguments,
    reason = "mirrors Splash::fillGlyph2 API; all params necessary"
)]
pub fn blit_glyph<P: Pixel>(
    bitmap: &mut Bitmap<P>,
    clip: &Clip,
    clip_all_inside: bool,
    pipe: &PipeState<'_>,
    src: &PipeSrc<'_>,
    pen_x: i32,
    pen_y: i32,
    glyph: &GlyphBitmap<'_>,
) {
    // Compute the top-left destination pixel.
    let x_start_raw = pen_x - glyph.x;
    let y_start_raw = pen_y - glyph.y;

    // Clamp to bitmap bounds.
    #[expect(
        clippy::cast_possible_wrap,
        reason = "bitmap dims ≤ i32::MAX in practice"
    )]
    let bmp_w = bitmap.width as i32;
    #[expect(
        clippy::cast_possible_wrap,
        reason = "bitmap dims ≤ i32::MAX in practice"
    )]
    let bmp_h = bitmap.height as i32;

    let x_clip = x_start_raw.max(0);
    let y_clip = y_start_raw.max(0);
    let x_end = (x_start_raw + glyph.w).min(bmp_w);
    let y_end = (y_start_raw + glyph.h).min(bmp_h);

    if x_clip >= x_end || y_clip >= y_end {
        return;
    }

    // All four differences are non-negative because of the checks above.
    #[expect(
        clippy::cast_sign_loss,
        reason = "x_clip ≥ x_start_raw.max(0) so difference ≥ 0"
    )]
    let x_data_skip = (x_clip - x_start_raw) as usize;
    #[expect(
        clippy::cast_sign_loss,
        reason = "y_clip ≥ y_start_raw.max(0) so difference ≥ 0"
    )]
    let y_data_skip = (y_clip - y_start_raw) as usize;
    #[expect(clippy::cast_sign_loss, reason = "x_end > x_clip by guard above")]
    let xx_limit = (x_end - x_clip) as usize;
    #[expect(clippy::cast_sign_loss, reason = "y_end > y_clip by guard above")]
    let yy_limit = (y_end - y_clip) as usize;
    let row_bytes = glyph.row_bytes();

    if glyph.aa {
        blit_aa::<P>(
            bitmap,
            clip,
            clip_all_inside,
            pipe,
            src,
            glyph,
            x_clip,
            y_clip,
            x_data_skip,
            y_data_skip,
            xx_limit,
            yy_limit,
            row_bytes,
        );
    } else {
        blit_mono::<P>(
            bitmap,
            clip,
            clip_all_inside,
            pipe,
            src,
            glyph,
            x_clip,
            y_clip,
            x_data_skip,
            y_data_skip,
            xx_limit,
            yy_limit,
            row_bytes,
        );
    }
}

/// Blit an AA (per-byte coverage) glyph.
#[expect(
    clippy::too_many_arguments,
    reason = "internal helper; all params necessary for this blit path"
)]
fn blit_aa<P: Pixel>(
    bitmap: &mut Bitmap<P>,
    clip: &Clip,
    clip_all_inside: bool,
    pipe: &PipeState<'_>,
    src: &PipeSrc<'_>,
    glyph: &GlyphBitmap<'_>,
    x_start: i32,
    y_start: i32,
    x_data_skip: usize,
    y_data_skip: usize,
    xx_limit: usize,
    yy_limit: usize,
    row_bytes: usize,
) {
    let data = glyph.data;
    // Verify the glyph data buffer covers the visible region.  A mismatch
    // indicates a malformed GlyphBitmap (wrong w/h/row_bytes); we clamp via
    // get() in the inner loop, but the assert catches it in debug builds.
    debug_assert!(
        data.len() >= (y_data_skip + yy_limit) * row_bytes,
        "blit_aa: glyph data too short: len={} < (y_data_skip={y_data_skip} + yy_limit={yy_limit}) * row_bytes={row_bytes}",
        data.len(),
    );
    // Hoisted above the row loop to avoid per-row heap allocation.
    let mut run_shape: Vec<u8> = Vec::new();

    for yy in 0..yy_limit {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "yy < yy_limit ≤ bitmap.height ≤ i32::MAX"
        )]
        #[expect(clippy::cast_possible_wrap, reason = "yy < bitmap.height ≤ i32::MAX")]
        let y = y_start + yy as i32;
        let row_off = (y_data_skip + yy) * row_bytes + x_data_skip;

        let mut run_start: Option<i32> = None;
        run_shape.clear();

        for xx in 0..xx_limit {
            #[expect(
                clippy::cast_possible_truncation,
                reason = "xx < xx_limit ≤ bitmap.width ≤ i32::MAX"
            )]
            #[expect(clippy::cast_possible_wrap, reason = "xx < bitmap.width ≤ i32::MAX")]
            let x = x_start + xx as i32;
            let data_idx = row_off + xx;
            let alpha = data.get(data_idx).copied().unwrap_or(0);
            let inside_clip = clip_all_inside || clip.test(x, y);

            if inside_clip && alpha != 0 {
                if run_start.is_none() {
                    run_start = Some(x);
                    run_shape.clear();
                }
                run_shape.push(alpha);
            } else if let Some(rs) = run_start.take() {
                emit_aa_run::<P>(bitmap, pipe, src, rs, y, &run_shape);
            }
        }
        if let Some(rs) = run_start.take() {
            emit_aa_run::<P>(bitmap, pipe, src, rs, y, &run_shape);
        }
    }
}

/// Blit a mono (1-bit packed) glyph.
#[expect(
    clippy::too_many_arguments,
    reason = "internal helper; all params necessary"
)]
#[expect(
    clippy::too_many_lines,
    reason = "function handles both SIMD and scalar mono-unpack paths; splitting further would obscure the logic"
)]
fn blit_mono<P: Pixel>(
    bitmap: &mut Bitmap<P>,
    clip: &Clip,
    clip_all_inside: bool,
    pipe: &PipeState<'_>,
    src: &PipeSrc<'_>,
    glyph: &GlyphBitmap<'_>,
    x_start: i32,
    y_start: i32,
    x_data_skip: usize,
    y_data_skip: usize,
    xx_limit: usize,
    yy_limit: usize,
    row_bytes: usize,
) {
    let x_shift = x_data_skip % 8;
    let data = glyph.data;

    // Verify the glyph data buffer covers the visible region (debug builds only).
    debug_assert!(
        data.len() >= (y_data_skip + yy_limit) * row_bytes,
        "blit_mono: glyph data too short: len={} < (y_data_skip={y_data_skip} + yy_limit={yy_limit}) * row_bytes={row_bytes}",
        data.len(),
    );

    // Scratch buffer for SIMD-expanded bits: one byte per pixel, 0x00 or 0xFF.
    // Sized to the maximum row width; reused across rows.
    let mut expanded: Vec<u8> = vec![0u8; xx_limit];

    for yy in 0..yy_limit {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "yy < yy_limit ≤ bitmap.height ≤ i32::MAX"
        )]
        #[expect(clippy::cast_possible_wrap, reason = "yy < bitmap.height ≤ i32::MAX")]
        let y = y_start + yy as i32;
        let row_off = (y_data_skip + yy) * row_bytes + x_data_skip / 8;

        // Use SIMD unpack when x_shift == 0 (bits are byte-aligned).
        // When x_shift > 0 the pixels straddle byte boundaries, so we fall
        // back to the scalar bit-extraction path which handles that case.
        #[cfg(target_arch = "x86_64")]
        let use_simd_unpack = x_shift == 0;
        #[cfg(not(target_arch = "x86_64"))]
        let use_simd_unpack = false;

        if use_simd_unpack {
            let packed_row = &data[row_off..];
            simd::unpack_mono_row(packed_row, xx_limit, &mut expanded);

            let mut run_start: Option<i32> = None;
            for (xx, &cov) in expanded[..xx_limit].iter().enumerate() {
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "xx < xx_limit ≤ bitmap.width ≤ i32::MAX"
                )]
                #[expect(clippy::cast_possible_wrap, reason = "xx < bitmap.width ≤ i32::MAX")]
                let x = x_start + xx as i32;
                let set = cov != 0;
                let inside_clip = clip_all_inside || clip.test(x, y);

                if set && inside_clip {
                    if run_start.is_none() {
                        run_start = Some(x);
                    }
                } else if let Some(rs) = run_start.take() {
                    let rx1 = x - 1;
                    emit_solid_run::<P>(bitmap, pipe, src, rs, rx1, y);
                }
            }
            if let Some(rs) = run_start.take() {
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "xx_limit ≤ bitmap.width ≤ i32::MAX"
                )]
                #[expect(
                    clippy::cast_possible_wrap,
                    reason = "xx_limit < bitmap.width ≤ i32::MAX"
                )]
                let rx1 = x_start + xx_limit as i32 - 1;
                emit_solid_run::<P>(bitmap, pipe, src, rs, rx1, y);
            }
            continue;
        }

        let mut run_start: Option<i32> = None;
        let mut xx = 0usize;

        while xx < xx_limit {
            let byte_idx = row_off + xx / 8;

            // When x_shift > 0, straddle two source bytes to align the read window.
            let alpha0 = if x_shift > 0 && xx + 8 < xx_limit {
                let lo = data.get(byte_idx).copied().unwrap_or(0);
                let hi = data.get(byte_idx + 1).copied().unwrap_or(0);
                #[expect(clippy::cast_possible_truncation, reason = "shift result fits in u8")]
                {
                    (u16::from(lo) << x_shift | u16::from(hi) >> (8 - x_shift)) as u8
                }
            } else {
                data.get(byte_idx).copied().unwrap_or(0)
            };

            let bits_this_byte = (xx_limit - xx).min(8);
            for bit in 0..bits_this_byte {
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "xx + bit < xx_limit ≤ bitmap.width ≤ i32::MAX"
                )]
                #[expect(
                    clippy::cast_possible_wrap,
                    reason = "xx + bit < bitmap.width ≤ i32::MAX"
                )]
                let x = x_start + (xx + bit) as i32;
                let set = (alpha0 >> (7 - bit)) & 1 != 0;
                let inside_clip = clip_all_inside || clip.test(x, y);

                if set && inside_clip {
                    if run_start.is_none() {
                        run_start = Some(x);
                    }
                } else if let Some(rs) = run_start.take() {
                    let rx1 = x - 1;
                    emit_solid_run::<P>(bitmap, pipe, src, rs, rx1, y);
                }
            }
            xx += bits_this_byte;
        }
        if let Some(rs) = run_start.take() {
            #[expect(
                clippy::cast_possible_truncation,
                reason = "xx_limit ≤ bitmap.width ≤ i32::MAX"
            )]
            #[expect(
                clippy::cast_possible_wrap,
                reason = "xx_limit < bitmap.width ≤ i32::MAX"
            )]
            let rx1 = x_start + xx_limit as i32 - 1;
            emit_solid_run::<P>(bitmap, pipe, src, rs, rx1, y);
        }
    }
}

/// Emit one AA-shaped span via `render_span`.
fn emit_aa_run<P: Pixel>(
    bitmap: &mut Bitmap<P>,
    pipe: &PipeState<'_>,
    src: &PipeSrc<'_>,
    x0: i32,
    y: i32,
    shape: &[u8],
) {
    debug_assert!(!shape.is_empty());
    // shape.len() ≤ bitmap.width ≤ i32::MAX in practice.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "shape.len() ≤ bitmap.width ≤ i32::MAX"
    )]
    #[expect(
        clippy::cast_possible_wrap,
        reason = "shape.len() ≤ bitmap.width ≤ i32::MAX"
    )]
    let x1 = x0 + shape.len() as i32 - 1;
    #[expect(clippy::cast_sign_loss, reason = "y ≥ 0 by construction")]
    let y_u = y as u32;
    #[expect(clippy::cast_sign_loss, reason = "x0 ≥ 0")]
    let byte_off = x0 as usize * P::BYTES;
    #[expect(clippy::cast_sign_loss, reason = "x1 ≥ x0 ≥ 0")]
    let byte_end = (x1 as usize + 1) * P::BYTES;
    #[expect(clippy::cast_sign_loss, reason = "x0 ≥ 0, x1 ≥ x0")]
    let alpha_range = x0 as usize..=x1 as usize;

    let (row, alpha) = bitmap.row_and_alpha_mut(y_u);
    let dst_pixels = &mut row[byte_off..byte_end];
    let dst_alpha = alpha.map(|a| &mut a[alpha_range]);

    pipe::render_span::<P>(pipe, src, dst_pixels, dst_alpha, Some(shape), x0, x1, y);
}

/// Emit a fully opaque span (mono glyph pixels) via `render_span`.
fn emit_solid_run<P: Pixel>(
    bitmap: &mut Bitmap<P>,
    pipe: &PipeState<'_>,
    src: &PipeSrc<'_>,
    x0: i32,
    x1: i32,
    y: i32,
) {
    debug_assert!(x0 <= x1);
    #[expect(clippy::cast_sign_loss, reason = "y ≥ 0 by construction")]
    let y_u = y as u32;
    #[expect(clippy::cast_sign_loss, reason = "x0 ≥ 0")]
    let byte_off = x0 as usize * P::BYTES;
    #[expect(clippy::cast_sign_loss, reason = "x1 ≥ x0 ≥ 0")]
    let byte_end = (x1 as usize + 1) * P::BYTES;
    #[expect(clippy::cast_sign_loss, reason = "x0 ≥ 0, x1 ≥ x0")]
    let alpha_range = x0 as usize..=x1 as usize;

    let (row, alpha) = bitmap.row_and_alpha_mut(y_u);
    let dst_pixels = &mut row[byte_off..byte_end];
    let dst_alpha = alpha.map(|a| &mut a[alpha_range]);

    pipe::render_span::<P>(pipe, src, dst_pixels, dst_alpha, None, x0, x1, y);
}

/// Clip-test a glyph bbox and blit it.
///
/// Convenience wrapper that performs the `testRect` + `fillGlyph2` pattern from
/// `Splash::fillGlyph`. Returns the `ClipResult` for the caller to record as `opClipRes`.
pub fn fill_glyph<P: Pixel>(
    bitmap: &mut Bitmap<P>,
    clip: &Clip,
    pipe: &PipeState<'_>,
    src: &PipeSrc<'_>,
    pen_x: i32,
    pen_y: i32,
    glyph: &GlyphBitmap<'_>,
) -> ClipResult {
    let x0 = pen_x - glyph.x;
    let y0 = pen_y - glyph.y;
    let x1 = x0 + glyph.w - 1;
    let y1 = y0 + glyph.h - 1;

    let clip_res = clip.test_rect(x0, y0, x1, y1);
    if clip_res != ClipResult::AllOutside {
        blit_glyph::<P>(
            bitmap,
            clip,
            clip_res == ClipResult::AllInside,
            pipe,
            src,
            pen_x,
            pen_y,
            glyph,
        );
    }
    clip_res
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitmap::Bitmap;
    use crate::clip::Clip;
    use crate::pipe::{PipeSrc, PipeState};
    use crate::state::TransferSet;
    use crate::types::BlendMode;
    use color::Rgb8;

    fn simple_pipe() -> PipeState<'static> {
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

    fn make_clip(w: u32, h: u32) -> Clip {
        Clip::new(0.0, 0.0, w as f64 - 0.001, h as f64 - 0.001, false)
    }

    #[test]
    fn blit_aa_glyph_paints_pixels() {
        let mut bmp: Bitmap<Rgb8> = Bitmap::new(8, 8, 4, false);
        let clip = make_clip(8, 8);
        let pipe = simple_pipe();
        let color = [255u8, 0, 0];
        let src = PipeSrc::Solid(&color);

        // 4×4 AA glyph with full coverage (255) everywhere.
        let data = vec![255u8; 16];
        let glyph = GlyphBitmap {
            data: &data,
            x: 0,
            y: 0,
            w: 4,
            h: 4,
            aa: true,
        };

        blit_glyph::<Rgb8>(&mut bmp, &clip, true, &pipe, &src, 2, 2, &glyph);

        // Rows 2..5, cols 2..5 should be red.
        for y in 2..6u32 {
            let row = bmp.row(y);
            for x in 2..6usize {
                assert_eq!(row[x].r, 255, "y={y} x={x} R");
            }
        }
        assert_eq!(bmp.row(1)[1].r, 0);
    }

    #[test]
    fn blit_aa_glyph_zero_coverage_skips() {
        let mut bmp: Bitmap<Rgb8> = Bitmap::new(4, 4, 4, false);
        let clip = make_clip(4, 4);
        let pipe = simple_pipe();
        let color = [255u8, 0, 0];
        let src = PipeSrc::Solid(&color);

        let data = vec![0u8; 4];
        let glyph = GlyphBitmap {
            data: &data,
            x: 0,
            y: 0,
            w: 2,
            h: 2,
            aa: true,
        };

        blit_glyph::<Rgb8>(&mut bmp, &clip, true, &pipe, &src, 0, 0, &glyph);

        for y in 0..4u32 {
            let row = bmp.row(y);
            for x in 0..4usize {
                assert_eq!(row[x].r, 0, "should be unpainted");
            }
        }
    }

    #[test]
    fn blit_mono_glyph_paints_set_bits() {
        let mut bmp: Bitmap<Rgb8> = Bitmap::new(8, 4, 4, false);
        let clip = make_clip(8, 4);
        let pipe = simple_pipe();
        let color = [0u8, 255, 0];
        let src = PipeSrc::Solid(&color);

        // 8×2 mono glyph: row 0 all-set (0xFF), row 1 all-clear (0x00).
        let data = [0xFFu8, 0x00u8];
        let glyph = GlyphBitmap {
            data: &data,
            x: 0,
            y: 0,
            w: 8,
            h: 2,
            aa: false,
        };

        blit_glyph::<Rgb8>(&mut bmp, &clip, true, &pipe, &src, 0, 0, &glyph);

        let row0 = bmp.row(0);
        for x in 0..8usize {
            assert_eq!(row0[x].g, 255, "row 0 x={x} should be painted");
        }
        let row1 = bmp.row(1);
        for x in 0..8usize {
            assert_eq!(row1[x].g, 0, "row 1 x={x} should be clear");
        }
    }

    #[test]
    fn blit_glyph_clip_excludes_outside() {
        let mut bmp: Bitmap<Rgb8> = Bitmap::new(8, 8, 4, false);
        let clip = Clip::new(2.0, 2.0, 5.0, 5.0, false);
        let pipe = simple_pipe();
        let color = [255u8, 0, 0];
        let src = PipeSrc::Solid(&color);

        let data = vec![255u8; 16];
        let glyph = GlyphBitmap {
            data: &data,
            x: 0,
            y: 0,
            w: 4,
            h: 4,
            aa: true,
        };

        blit_glyph::<Rgb8>(&mut bmp, &clip, false, &pipe, &src, 0, 0, &glyph);

        assert_eq!(bmp.row(0)[0].r, 0, "row 0 should be clipped");
        assert_eq!(bmp.row(1)[0].r, 0, "row 1 should be clipped");
        assert_eq!(bmp.row(2)[2].r, 255, "(2,2) should be painted");
    }

    #[test]
    fn fill_glyph_returns_clip_result() {
        let mut bmp: Bitmap<Rgb8> = Bitmap::new(8, 8, 4, false);
        let clip = make_clip(8, 8);
        let pipe = simple_pipe();
        let color = [128u8, 128, 128];
        let src = PipeSrc::Solid(&color);

        let data = vec![128u8; 4];
        let glyph = GlyphBitmap {
            data: &data,
            x: 0,
            y: 0,
            w: 2,
            h: 2,
            aa: true,
        };

        let res = fill_glyph::<Rgb8>(&mut bmp, &clip, &pipe, &src, 1, 1, &glyph);
        assert_eq!(res, ClipResult::AllInside);
    }

    #[test]
    fn glyph_row_bytes_aa() {
        let data = [];
        let g = GlyphBitmap {
            data: &data,
            x: 0,
            y: 0,
            w: 7,
            h: 1,
            aa: true,
        };
        assert_eq!(g.row_bytes(), 7);
    }

    #[test]
    fn glyph_row_bytes_mono() {
        let data = [];
        let g = GlyphBitmap {
            data: &data,
            x: 0,
            y: 0,
            w: 7,
            h: 1,
            aa: false,
        };
        assert_eq!(g.row_bytes(), 1); // ceil(7/8) = 1
        let g9 = GlyphBitmap {
            data: &data,
            x: 0,
            y: 0,
            w: 9,
            h: 1,
            aa: false,
        };
        assert_eq!(g9.row_bytes(), 2); // ceil(9/8) = 2
    }
}
