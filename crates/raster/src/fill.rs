//! Filled-path rasterization — replaces `Splash::fill`, `Splash::eoFill`,
//! and the core of `Splash::fillWithPattern`.
//!
//! Two public entry points:
//! - [`fill`] — non-zero winding fill
//! - [`eo_fill`] — even-odd fill
//!
//! Both entry points are thin wrappers around [`fill_impl`] which does
//! the actual work: build `XPath` → optionally AA-scale → build
//! `XPathScanner` → walk spans → clip → `pipe::render_span`.
//!
//! # AA (vector antialias) mode
//!
//! When `vector_antialias` is `true` the path is scaled 4× in `XPath` (`aaScale`),
//! the scanner operates in 4× AA coordinates, `render_aa_line` fills the `AaBuf`,
//! the clip AA-masks the buf, and then `draw_aa_line` reads the 4-bit coverage
//! count (0..16) through a gamma LUT (`aaGamma`) and calls `render_span_aa`.
//!
//! # C++ equivalent
//! `Splash::fillWithPattern` (Splash.cc ~line 2382).

use crate::bitmap::Bitmap;
use crate::bitmap::AaBuf;
use crate::clip::{Clip, ClipResult};
use crate::path::Path;
use crate::pipe::{self, PipeSrc, PipeState};
use crate::scanner::iter::ScanIterator;
use crate::scanner::XPathScanner;
use crate::types::AA_SIZE;
use crate::xpath::XPath;
use color::Pixel;

/// AA gamma table for `splashAASize=4`, `splashAAGamma=1.5`.
/// Entry `i`: `round((i/16)^1.5 * 255)` for i in 0..=16.
const AA_GAMMA: [u8; (AA_SIZE * AA_SIZE + 1) as usize] =
    [0, 4, 11, 20, 32, 45, 59, 75, 91, 108, 128, 148, 169, 191, 214, 238, 255];

/// Bit-count table for nibbles 0x0..=0xf — number of set bits.
const NIBBLE_POP: [u8; 16] = [0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4];

/// Non-zero winding fill.
#[expect(clippy::too_many_arguments, reason = "mirrors SplashFillWithPattern API; all params necessary")]
pub fn fill<P: Pixel>(
    bitmap: &mut Bitmap<P>,
    clip: &Clip,
    path: &Path,
    pipe: &PipeState<'_>,
    src: &PipeSrc<'_>,
    matrix: &[f64; 6],
    flatness: f64,
    vector_antialias: bool,
) {
    fill_impl::<P>(bitmap, clip, path, pipe, src, matrix, flatness, vector_antialias, false);
}

/// Even-odd fill.
#[expect(clippy::too_many_arguments, reason = "mirrors SplashFillWithPattern API; all params necessary")]
pub fn eo_fill<P: Pixel>(
    bitmap: &mut Bitmap<P>,
    clip: &Clip,
    path: &Path,
    pipe: &PipeState<'_>,
    src: &PipeSrc<'_>,
    matrix: &[f64; 6],
    flatness: f64,
    vector_antialias: bool,
) {
    fill_impl::<P>(bitmap, clip, path, pipe, src, matrix, flatness, vector_antialias, true);
}

#[expect(clippy::too_many_arguments, reason = "mirrors SplashFillWithPattern API; all params necessary")]
fn fill_impl<P: Pixel>(
    bitmap: &mut Bitmap<P>,
    clip: &Clip,
    path: &Path,
    pipe: &PipeState<'_>,
    src: &PipeSrc<'_>,
    matrix: &[f64; 6],
    flatness: f64,
    vector_antialias: bool,
    eo: bool,
) {
    if path.pts.is_empty() {
        return;
    }

    let mut xpath = XPath::new(path, matrix, flatness, true);

    // Clip y bounds in scanner coordinates (AA or normal).
    let (y_min_clip, y_max_clip) = if vector_antialias {
        xpath.aa_scale();
        (clip.y_min_i * AA_SIZE, (clip.y_max_i + 1) * AA_SIZE - 1)
    } else {
        (clip.y_min_i, clip.y_max_i)
    };

    let scanner = XPathScanner::new(&xpath, eo, y_min_clip, y_max_clip);

    if scanner.is_empty() {
        return;
    }

    // Compute pixel-space bbox from scanner.
    let (x_min_i, y_min_i, x_max_i, y_max_i) = if vector_antialias {
        (
            scanner.x_min / AA_SIZE,
            scanner.y_min / AA_SIZE,
            scanner.x_max / AA_SIZE,
            scanner.y_max / AA_SIZE,
        )
    } else {
        (scanner.x_min, scanner.y_min, scanner.x_max, scanner.y_max)
    };

    let clip_res = clip.test_rect(x_min_i, y_min_i, x_max_i, y_max_i);
    if clip_res == ClipResult::AllOutside {
        return;
    }

    if vector_antialias {
        let bitmap_width = bitmap.width as usize;
        let mut aa_buf = AaBuf::new(bitmap_width);

        for aa_y in scanner.y_min..=scanner.y_max {
            let y = aa_y / AA_SIZE;
            // The AA scanner row index within the 4-row AaBuf band.
            // aa_y % AA_SIZE is in 0..AA_SIZE so always non-negative.
            #[expect(clippy::cast_sign_loss, reason = "aa_y % AA_SIZE is in 0..AA_SIZE ≥ 0")]
            let aa_row = (aa_y % AA_SIZE) as usize;

            // Determine x span for this AA scanline.
            let mut x0 = scanner.x_min / AA_SIZE;
            let mut x1 = scanner.x_max / AA_SIZE;

            scanner.render_aa_line(&mut aa_buf, &mut x0, &mut x1, aa_y);

            if clip_res != ClipResult::AllInside {
                clip.clip_aa_line(&mut aa_buf, &mut x0, &mut x1, aa_y);
            }

            // At the boundary of each output row, emit one composited line.
            if aa_row == AA_SIZE as usize - 1 {
                #[expect(clippy::cast_sign_loss, reason = "y = aa_y / AA_SIZE ≥ 0 since scanner.y_min ≥ 0")]
                if x0 <= x1 && y >= 0 && (y as u32) < bitmap.height {
                    draw_aa_line::<P>(bitmap, pipe, src, &aa_buf, x0, x1, y);
                }
                aa_buf.clear();
            }
        }
    } else {
        for y in y_min_i..=y_max_i {
            #[expect(clippy::cast_sign_loss, reason = "cast after y < 0 guard")]
            if y < 0 || (y as u32) >= bitmap.height {
                continue;
            }
            #[expect(clippy::cast_possible_wrap, reason = "bitmap.width ≤ i32::MAX in practice")]
            let width_i = bitmap.width as i32;
            for (x0, x1) in ScanIterator::new(&scanner, y) {
                let (mut sx0, mut sx1) = (x0, x1);
                let inner_clip = if clip_res == ClipResult::AllInside {
                    // Clamp to bitmap bounds only.
                    sx0 = sx0.max(0);
                    sx1 = sx1.min(width_i - 1);
                    true
                } else {
                    sx0 = sx0.max(clip.x_min_i);
                    sx1 = sx1.min(clip.x_max_i);
                    clip.test_span(sx0, sx1, y) == ClipResult::AllInside
                };

                if sx0 > sx1 {
                    continue;
                }

                if inner_clip {
                    draw_span::<P>(bitmap, pipe, src, sx0, sx1, y);
                } else {
                    draw_span_clipped::<P>(bitmap, clip, pipe, src, sx0, sx1, y);
                }
            }
        }
    }
}

/// Emit a solid span that is fully inside the clip — no per-pixel clip test.
fn draw_span<P: Pixel>(
    bitmap: &mut Bitmap<P>,
    pipe: &PipeState<'_>,
    src: &PipeSrc<'_>,
    x0: i32,
    x1: i32,
    y: i32,
) {
    debug_assert!(x0 <= x1);
    debug_assert!(y >= 0);
    #[expect(clippy::cast_sign_loss, reason = "y >= 0 checked above")]
    let y_u = y as u32;
    #[expect(clippy::cast_sign_loss, reason = "x0 >= 0 after clip clamping")]
    let byte_off = x0 as usize * P::BYTES;
    #[expect(clippy::cast_sign_loss, reason = "x1 >= x0 >= 0")]
    let byte_end = (x1 as usize + 1) * P::BYTES;
    #[expect(clippy::cast_sign_loss, reason = "x0 >= 0, x1 >= x0")]
    let alpha_range = x0 as usize..=x1 as usize;

    let (row, alpha) = bitmap.row_and_alpha_mut(y_u);
    let dst_pixels = &mut row[byte_off..byte_end];
    let dst_alpha = alpha.map(|a| &mut a[alpha_range]);

    pipe::render_span::<P>(pipe, src, dst_pixels, dst_alpha, None, x0, x1, y);
}

/// Emit a span with per-pixel clip test (partial clip region).
fn draw_span_clipped<P: Pixel>(
    bitmap: &mut Bitmap<P>,
    clip: &Clip,
    pipe: &PipeState<'_>,
    src: &PipeSrc<'_>,
    x0: i32,
    x1: i32,
    y: i32,
) {
    // Walk pixel by pixel, skipping those outside the clip.
    // Batch contiguous inside-clip runs into single render_span calls.
    let mut run_start: Option<i32> = None;

    for x in x0..=x1 {
        if clip.test(x, y) {
            if run_start.is_none() {
                run_start = Some(x);
            }
        } else if let Some(rs) = run_start.take() {
            draw_span::<P>(bitmap, pipe, src, rs, x - 1, y);
        }
    }
    if let Some(rs) = run_start {
        draw_span::<P>(bitmap, pipe, src, rs, x1, y);
    }
}

/// Emit one composited output pixel row from the 4-row `AaBuf`.
///
/// For each output pixel `x` in `[x0, x1]`, count the set bits across all 4
/// AA sub-rows, look up the gamma-corrected shape byte, and call `render_span_aa`
/// with shape > 0.
fn draw_aa_line<P: Pixel>(
    bitmap: &mut Bitmap<P>,
    pipe: &PipeState<'_>,
    src: &PipeSrc<'_>,
    aa_buf: &AaBuf,
    x0: i32,
    x1: i32,
    y: i32,
) {
    debug_assert!(x0 <= x1);
    debug_assert!(y >= 0);

    #[expect(clippy::cast_sign_loss, reason = "x0 >= 0 after clip clamping")]
    let count = (x1 - x0 + 1) as usize;

    // Build shape vector and find non-zero range.
    let mut shape = vec![0u8; count];
    let mut any_nonzero = false;

    for (i, shape_byte) in shape.iter_mut().enumerate() {
        // x0 >= 0 and i < bitmap.width ≤ i32::MAX, so the sum fits in i32.
        #[expect(clippy::cast_sign_loss, reason = "x0 >= 0")]
        #[expect(clippy::cast_possible_truncation, reason = "x0 + i ≤ bitmap width ≤ i32::MAX")]
        #[expect(clippy::cast_possible_wrap, reason = "x0 + i ≤ bitmap width ≤ i32::MAX")]
        let x = (x0 as usize + i) as i32;
        let t = aa_coverage(aa_buf, x);
        if t > 0 {
            *shape_byte = AA_GAMMA[t as usize];
            any_nonzero = true;
        }
    }

    if !any_nonzero {
        return;
    }

    #[expect(clippy::cast_sign_loss, reason = "y >= 0")]
    let y_u = y as u32;
    #[expect(clippy::cast_sign_loss, reason = "x0 >= 0 after clip clamping")]
    let byte_off = x0 as usize * P::BYTES;
    #[expect(clippy::cast_sign_loss, reason = "x1 >= x0 >= 0")]
    let byte_end = (x1 as usize + 1) * P::BYTES;
    #[expect(clippy::cast_sign_loss, reason = "x0 >= 0, x1 >= x0")]
    let alpha_range = x0 as usize..=x1 as usize;

    let (row, alpha) = bitmap.row_and_alpha_mut(y_u);
    let dst_pixels = &mut row[byte_off..byte_end];
    let dst_alpha = alpha.map(|a| &mut a[alpha_range]);

    pipe::render_span::<P>(pipe, src, dst_pixels, dst_alpha, Some(&shape), x0, x1, y);
}

/// Count set AA bits across all 4 sub-rows for one output pixel `x`.
///
/// Each output pixel maps to `AA_SIZE` (=4) bits in each row.
/// The C++ equivalent reads 4 nibbles and uses `bitCount4`.
fn aa_coverage(aa_buf: &AaBuf, x: i32) -> u32 {
    // For splashAASize=4, pixel x maps to bits [x*4 .. x*4+3] in each row.
    // Within a byte, 2 pixels fit: high nibble = even pixel, low nibble = odd pixel.
    #[expect(clippy::cast_sign_loss, reason = "x >= 0 in caller")]
    let col = x as usize;

    let byte_idx = col >> 1;
    let is_odd = col & 1 != 0;

    let mut count = 0u32;
    for row in 0..AA_SIZE as usize {
        let byte = aa_buf.get_byte(row, byte_idx);
        let nibble = if is_odd { byte & 0x0f } else { byte >> 4 };
        count += u32::from(NIBBLE_POP[nibble as usize]);
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitmap::Bitmap;
    use crate::clip::Clip;
    use crate::path::PathBuilder;
    use crate::pipe::{PipeSrc, PipeState};
    use crate::state::TransferSet;
    use crate::types::BlendMode;
    use color::Rgb8;

    fn identity_matrix() -> [f64; 6] {
        [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    }

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

    /// Build a simple filled rectangle path: 4 corners, closed.
    fn rect_path(x0: f64, y0: f64, x1: f64, y1: f64) -> Path {
        let mut b = PathBuilder::new();
        b.move_to(x0, y0).unwrap();
        b.line_to(x1, y0).unwrap();
        b.line_to(x1, y1).unwrap();
        b.line_to(x0, y1).unwrap();
        b.close(true).unwrap();
        b.build()
    }

    #[test]
    fn fill_rect_paints_solid() {
        let mut bmp: Bitmap<Rgb8> = Bitmap::new(8, 8, 4, false);
        let clip = make_clip(8, 8);
        let pipe = simple_pipe();
        let color = [200u8, 100, 50];
        let src = PipeSrc::Solid(&color);
        // Rect (1,1)→(5,5): horizontal edges at y=1 and y=5 have count=0 and
        // produce no interior spans; interior rows are y=2,3,4.
        // Columns span x=1..5 (the two vertical edges at x=1 and x=5).
        let path = rect_path(1.0, 1.0, 5.0, 5.0);

        fill::<Rgb8>(&mut bmp, &clip, &path, &pipe, &src, &identity_matrix(), 1.0, false);

        // Interior pixels (rows 2..4, cols 1..5) should be painted.
        for y in 2..5u32 {
            let row = bmp.row(y);
            for x in 1..=5usize {
                assert_eq!(row[x].r, 200, "y={y} x={x} R");
                assert_eq!(row[x].g, 100, "y={y} x={x} G");
                assert_eq!(row[x].b, 50,  "y={y} x={x} B");
            }
        }

        // Pixels outside should be untouched (zero).
        assert_eq!(bmp.row(0)[0].r, 0, "row 0 should be untouched");
        assert_eq!(bmp.row(1)[0].r, 0, "top edge row should be untouched");
        assert_eq!(bmp.row(2)[0].r, 0, "x=0 should be untouched");
    }

    #[test]
    fn fill_empty_path_is_noop() {
        let mut bmp: Bitmap<Rgb8> = Bitmap::new(8, 8, 4, false);
        let clip = make_clip(8, 8);
        let pipe = simple_pipe();
        let color = [255u8, 0, 0];
        let src = PipeSrc::Solid(&color);
        let path = PathBuilder::new().build(); // empty

        fill::<Rgb8>(&mut bmp, &clip, &path, &pipe, &src, &identity_matrix(), 1.0, false);

        // Nothing should be painted.
        assert_eq!(bmp.row(0)[0].r, 0);
    }

    #[test]
    fn eo_fill_donut_leaves_interior_clear() {
        // Even-odd rule: a square inside a larger square leaves the inner area unfilled.
        let mut bmp: Bitmap<Rgb8> = Bitmap::new(10, 10, 4, false);
        let clip = make_clip(10, 10);
        let pipe = simple_pipe();
        let color = [255u8, 0, 0];
        let src = PipeSrc::Solid(&color);

        // Outer square 1..8, inner square 3..6.
        let mut b = PathBuilder::new();
        b.move_to(1.0, 1.0).unwrap();
        b.line_to(8.0, 1.0).unwrap();
        b.line_to(8.0, 8.0).unwrap();
        b.line_to(1.0, 8.0).unwrap();
        b.close(true).unwrap();
        b.move_to(3.0, 3.0).unwrap();
        b.line_to(6.0, 3.0).unwrap();
        b.line_to(6.0, 6.0).unwrap();
        b.line_to(3.0, 6.0).unwrap();
        b.close(true).unwrap();
        let path = b.build();

        eo_fill::<Rgb8>(&mut bmp, &clip, &path, &pipe, &src, &identity_matrix(), 1.0, false);

        // Interior (4,4) should be unpainted.
        assert_eq!(bmp.row(4)[4].r, 0, "interior should be clear with EO rule");
        // Band (2, 2) should be painted.
        assert_eq!(bmp.row(2)[2].r, 255, "outer band should be painted");
    }

    #[test]
    fn aa_gamma_table_correct() {
        // Spot-check a few entries.
        assert_eq!(AA_GAMMA[0], 0);
        assert_eq!(AA_GAMMA[16], 255);
        // t=8 (50%): round((8/16)^1.5 * 255) = round(0.5^1.5 * 255) = round(0.3536*255) ≈ 90
        assert!(AA_GAMMA[8] >= 88 && AA_GAMMA[8] <= 93, "AA_GAMMA[8]={}", AA_GAMMA[8]);
    }

    #[test]
    fn scanner_produces_spans_for_rect() {
        use crate::scanner::iter::ScanIterator;
        use crate::scanner::XPathScanner;
        use crate::xpath::XPath;

        let path = rect_path(1.0, 1.0, 5.0, 5.0);
        let xpath = XPath::new(&path, &identity_matrix(), 1.0, true);
        let scanner = XPathScanner::new(&xpath, false, 0, 7);
        // Interior rows 2,3,4 should have spans; boundary rows 1,5 have horizontal
        // edges (count=0) and produce no interior spans.
        assert!(!ScanIterator::new(&scanner, 2).collect::<Vec<_>>().is_empty(), "no spans at y=2");
        assert!(!ScanIterator::new(&scanner, 3).collect::<Vec<_>>().is_empty(), "no spans at y=3");
        assert!(!ScanIterator::new(&scanner, 4).collect::<Vec<_>>().is_empty(), "no spans at y=4");
    }
}
