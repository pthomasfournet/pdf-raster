//! Filled-path rasterization — replaces `Splash::fill`, `Splash::eoFill`,
//! and the core of `Splash::fillWithPattern`.
//!
//! Two public entry points:
//! - [`fill`] — non-zero winding fill
//! - [`eo_fill`] — even-odd fill
//!
//! Both entry points are thin wrappers around the private `fill_impl` which
//! does the actual work: build `XPath` → optionally AA-scale → build
//! `XPathScanner` → walk spans → clip → `pipe::render_span`.
//!
//! # AA (vector antialias) mode
//!
//! When `vector_antialias` is `true` the path is scaled 4× in `XPath` (`aaScale`),
//! the scanner operates in 4× AA coordinates, `render_aa_line` fills the `AaBuf`,
//! the clip AA-masks the buf, and then `draw_aa_line` reads the 4-bit coverage
//! count (0..16) through a gamma LUT (`aaGamma`) and calls `render_span_aa`.
//!
//! # Parallel fill
//!
//! Rayon-parallel variants `fill_parallel` / `eo_fill_parallel` live in a
//! private `parallel` submodule and are re-exported here when the `rayon`
//! feature is enabled.
//!
//! # C++ equivalent
//! `Splash::fillWithPattern`.

#[cfg(feature = "rayon")]
mod parallel;
#[cfg(feature = "rayon")]
pub use parallel::{PARALLEL_FILL_MIN_HEIGHT, eo_fill_parallel, fill_parallel};

use crate::bitmap::{AaBuf, Bitmap, BitmapBand};
use crate::clip::{Clip, ClipResult};
use crate::path::Path;
use crate::pipe::{self, PipeSrc, PipeState};
use crate::scanner::XPathScanner;
use crate::scanner::iter::ScanIterator;
use crate::simd;
use crate::types::AA_SIZE;
use crate::xpath::XPath;
use color::Pixel;

/// AA gamma table for `splashAASize=4`, `splashAAGamma=1.5`.
/// Entry `i`: `round((i/16)^1.5 * 255)` for i in 0..=16.
pub(super) const AA_GAMMA: [u8; (AA_SIZE * AA_SIZE + 1) as usize] = [
    0, 4, 11, 21, 32, 45, 59, 74, 90, 108, 126, 145, 166, 187, 209, 231, 255,
];

/// Non-zero winding fill.
#[expect(
    clippy::too_many_arguments,
    reason = "mirrors SplashFillWithPattern API; all params necessary"
)]
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
    fill_impl::<P>(
        bitmap,
        clip,
        path,
        pipe,
        src,
        matrix,
        flatness,
        vector_antialias,
        false,
    );
}

/// Even-odd fill.
#[expect(
    clippy::too_many_arguments,
    reason = "mirrors SplashFillWithPattern API; all params necessary"
)]
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
    fill_impl::<P>(
        bitmap,
        clip,
        path,
        pipe,
        src,
        matrix,
        flatness,
        vector_antialias,
        true,
    );
}

#[expect(
    clippy::too_many_arguments,
    reason = "mirrors SplashFillWithPattern API; all params necessary"
)]
pub(super) fn fill_impl<P: Pixel>(
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
        let y_min = clip
            .y_min_i
            .checked_mul(AA_SIZE)
            .expect("AA y_lo overflows i32: clip.y_min_i is unreasonably large");
        let y_max = clip
            .y_max_i
            .checked_add(1)
            .and_then(|v| v.checked_mul(AA_SIZE))
            .map(|v| v - 1)
            .expect("AA y_hi overflows i32: clip.y_max_i is unreasonably large");
        (y_min, y_max)
    } else {
        (clip.y_min_i, clip.y_max_i)
    };

    let scanner = XPathScanner::new(&xpath, eo, y_min_clip, y_max_clip);

    if scanner.is_empty() {
        return;
    }

    if bitmap.width == 0 {
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
                #[expect(
                    clippy::cast_sign_loss,
                    reason = "y = aa_y / AA_SIZE ≥ 0 since scanner.y_min ≥ 0"
                )]
                if x0 <= x1 && y >= 0 && (y as u32) < bitmap.height {
                    draw_aa_line::<P>(bitmap, pipe, src, &aa_buf, x0, x1, y);
                }
                aa_buf.clear();
            }
        }
    } else {
        #[expect(
            clippy::cast_possible_wrap,
            reason = "bitmap.width ≤ i32::MAX in practice; zero checked above scanner.is_empty()"
        )]
        let width_i = bitmap.width as i32;

        // Iterate only over scanlines that have at least one intersection —
        // skips empty rows in the bounding box without touching the fill loop.
        for y in scanner.nonempty_rows() {
            #[expect(clippy::cast_sign_loss, reason = "cast after y < 0 guard")]
            if y < 0 || (y as u32) >= bitmap.height {
                continue;
            }
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
                    draw_span::<P, _>(bitmap, pipe, src, sx0, sx1, y);
                } else {
                    draw_span_clipped::<P, _>(bitmap, clip, pipe, src, sx0, sx1, y);
                }
            }
        }
    }
}

// ── RowSink — shared abstraction over Bitmap and BitmapBand ──────────────────

/// A target that can vend a mutable pixel row and an optional alpha row.
///
/// Implemented by [`Bitmap`] and [`BitmapBand`], letting `draw_span` and
/// `draw_span_clipped` work without duplication across the sequential and
/// parallel fill paths.
pub(super) trait RowSink<P: Pixel> {
    /// Return mutable access to the raw pixel bytes and the optional alpha
    /// plane for absolute row `y`.
    ///
    /// # Panics
    ///
    /// Panics if `y` is out of range for the sink (matches the behaviour of
    /// `Bitmap::row_and_alpha_mut` and `BitmapBand::row_and_alpha_mut`).
    fn row_and_alpha_mut(&mut self, y: u32) -> (&mut [u8], Option<&mut [u8]>);
}

impl<P: Pixel> RowSink<P> for Bitmap<P> {
    #[inline]
    fn row_and_alpha_mut(&mut self, y: u32) -> (&mut [u8], Option<&mut [u8]>) {
        self.row_and_alpha_mut(y)
    }
}

impl<P: Pixel> RowSink<P> for BitmapBand<'_, P> {
    #[inline]
    fn row_and_alpha_mut(&mut self, y: u32) -> (&mut [u8], Option<&mut [u8]>) {
        self.row_and_alpha_mut(y)
    }
}

// ── Span drawing helpers ──────────────────────────────────────────────────────

/// Emit a solid span into `sink` that is fully inside the clip — no per-pixel test.
pub(super) fn draw_span<P: Pixel, S: RowSink<P>>(
    sink: &mut S,
    pipe: &PipeState<'_>,
    src: &PipeSrc<'_>,
    x0: i32,
    x1: i32,
    y: i32,
) {
    debug_assert!(x0 <= x1, "draw_span: x0={x0} > x1={x1}");
    debug_assert!(
        x0 >= 0,
        "draw_span: x0={x0} is negative (caller must clamp before calling)"
    );
    debug_assert!(y >= 0, "draw_span: y={y} is negative");
    #[expect(clippy::cast_sign_loss, reason = "y >= 0 asserted above")]
    let y_u = y as u32;
    #[expect(clippy::cast_sign_loss, reason = "x0 >= 0 asserted above")]
    let byte_off = x0 as usize * P::BYTES;
    #[expect(clippy::cast_sign_loss, reason = "x1 >= x0 >= 0 asserted above")]
    let byte_end = (x1 as usize + 1) * P::BYTES;
    #[expect(clippy::cast_sign_loss, reason = "x0 >= 0, x1 >= x0 asserted above")]
    let alpha_range = x0 as usize..=x1 as usize;

    let (row, alpha) = sink.row_and_alpha_mut(y_u);
    let dst_pixels = &mut row[byte_off..byte_end];
    let dst_alpha = alpha.map(|a| &mut a[alpha_range]);

    pipe::render_span::<P>(pipe, src, dst_pixels, dst_alpha, None, x0, x1, y);
}

/// Emit a span with per-pixel clip test (partial clip region).
pub(super) fn draw_span_clipped<P: Pixel, S: RowSink<P>>(
    sink: &mut S,
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
            draw_span(sink, pipe, src, rs, x - 1, y);
        }
    }
    if let Some(rs) = run_start {
        draw_span(sink, pipe, src, rs, x1, y);
    }
}

/// Emit one composited output pixel row from the 4-row `AaBuf`.
///
/// For each output pixel `x` in `[x0, x1]`, count the set bits across all 4
/// AA sub-rows via `simd::aa_coverage_span` (SIMD-accelerated), look up the
/// gamma-corrected shape byte, and call `render_span_aa` with shape > 0.
fn draw_aa_line<P: Pixel>(
    bitmap: &mut Bitmap<P>,
    pipe: &PipeState<'_>,
    src: &PipeSrc<'_>,
    aa_buf: &AaBuf,
    x0: i32,
    x1: i32,
    y: i32,
) {
    debug_assert!(x0 >= 0, "draw_aa_line: x0={x0} is negative");
    debug_assert!(x0 <= x1, "draw_aa_line: x0={x0} > x1={x1}");
    debug_assert!(y >= 0, "draw_aa_line: y={y} is negative");

    #[expect(clippy::cast_sign_loss, reason = "x0 >= 0: asserted above")]
    let x0_usize = x0 as usize;
    #[expect(clippy::cast_sign_loss, reason = "x1 >= x0 >= 0")]
    let count = (x1 - x0 + 1) as usize;

    // Gather raw coverage counts (0..=16) into shape[], then gamma-map in place.
    // Single allocation: aa_coverage_span writes raw counts; we overwrite with
    // AA_GAMMA[t] in the same buffer (0 stays 0; non-zero gets the LUT value).
    let rows = [
        aa_buf.row_slice(0),
        aa_buf.row_slice(1),
        aa_buf.row_slice(2),
        aa_buf.row_slice(3),
    ];
    let mut shape = vec![0u8; count];
    simd::aa_coverage_span(rows, x0_usize, &mut shape);

    // Gamma-map in place: 0 → 0 (skip), 1..=16 → AA_GAMMA[t].
    let mut any_nonzero = false;
    for s in &mut shape {
        let t = *s as usize;
        if t > 0 {
            *s = AA_GAMMA[t];
            any_nonzero = true;
        }
    }

    if !any_nonzero {
        return;
    }

    #[expect(clippy::cast_sign_loss, reason = "y >= 0")]
    let y_u = y as u32;
    let byte_off = x0_usize * P::BYTES;
    #[expect(clippy::cast_sign_loss, reason = "x1 >= x0 >= 0")]
    let byte_end = (x1 as usize + 1) * P::BYTES;
    #[expect(clippy::cast_sign_loss, reason = "x1 >= x0 >= 0")]
    let alpha_range = x0_usize..=x1 as usize;

    let (row, alpha) = bitmap.row_and_alpha_mut(y_u);
    let dst_pixels = &mut row[byte_off..byte_end];
    let dst_alpha = alpha.map(|a| &mut a[alpha_range]);

    pipe::render_span::<P>(pipe, src, dst_pixels, dst_alpha, Some(&shape), x0, x1, y);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitmap::Bitmap;
    use crate::path::PathBuilder;
    use crate::pipe::PipeSrc;
    use crate::testutil::{identity_matrix, make_clip, rect_path, simple_pipe};
    use color::Rgb8;

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

        fill::<Rgb8>(
            &mut bmp,
            &clip,
            &path,
            &pipe,
            &src,
            &identity_matrix(),
            1.0,
            false,
        );

        // Interior pixels (rows 2..4, cols 1..5) should be painted.
        for y in 2..5u32 {
            let row = bmp.row(y);
            for x in 1..=5usize {
                assert_eq!(row[x].r, 200, "y={y} x={x} R");
                assert_eq!(row[x].g, 100, "y={y} x={x} G");
                assert_eq!(row[x].b, 50, "y={y} x={x} B");
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

        fill::<Rgb8>(
            &mut bmp,
            &clip,
            &path,
            &pipe,
            &src,
            &identity_matrix(),
            1.0,
            false,
        );

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

        eo_fill::<Rgb8>(
            &mut bmp,
            &clip,
            &path,
            &pipe,
            &src,
            &identity_matrix(),
            1.0,
            false,
        );

        // Interior (4,4) should be unpainted.
        assert_eq!(bmp.row(4)[4].r, 0, "interior should be clear with EO rule");
        // Band (2, 2) should be painted.
        assert_eq!(bmp.row(2)[2].r, 255, "outer band should be painted");
    }

    #[test]
    fn aa_gamma_table_correct() {
        // Validate every entry against the defining formula: round((i/16)^1.5 * 255).
        for i in 0u8..=16 {
            let expected = ((f64::from(i) / 16.0).powf(1.5) * 255.0).round() as u8;
            assert_eq!(
                AA_GAMMA[usize::from(i)],
                expected,
                "AA_GAMMA[{i}]: expected {expected}"
            );
        }
    }

    #[test]
    fn scanner_produces_spans_for_rect() {
        use crate::scanner::XPathScanner;
        use crate::scanner::iter::ScanIterator;
        use crate::xpath::XPath;

        let path = rect_path(1.0, 1.0, 5.0, 5.0);
        let xpath = XPath::new(&path, &identity_matrix(), 1.0, true);
        let scanner = XPathScanner::new(&xpath, false, 0, 7);
        // Interior rows 2,3,4 should have spans; boundary rows 1,5 have horizontal
        // edges (count=0) and produce no interior spans.
        assert!(
            !ScanIterator::new(&scanner, 2)
                .collect::<Vec<_>>()
                .is_empty(),
            "no spans at y=2"
        );
        assert!(
            !ScanIterator::new(&scanner, 3)
                .collect::<Vec<_>>()
                .is_empty(),
            "no spans at y=3"
        );
        assert!(
            !ScanIterator::new(&scanner, 4)
                .collect::<Vec<_>>()
                .is_empty(),
            "no spans at y=4"
        );
    }

    // ── Parallel fill tests ───────────────────────────────────────────────────

    #[cfg(feature = "rayon")]
    mod parallel {
        use super::*;
        use crate::fill::{eo_fill_parallel, fill_parallel};

        /// Parallel fill (n_bands=4) must produce identical pixel output to
        /// sequential fill for a large rectangle.
        #[test]
        fn fill_parallel_matches_sequential() {
            const W: u32 = 64;
            const H: u32 = 512;

            let mut seq: Bitmap<Rgb8> = Bitmap::new(W, H, 1, false);
            let mut par: Bitmap<Rgb8> = Bitmap::new(W, H, 1, false);

            let clip = make_clip(W, H);
            let pipe = simple_pipe();
            let color = [77u8, 155, 211];
            let src = PipeSrc::Solid(&color);
            let path = rect_path(4.0, 4.0, 60.0, 508.0);
            let matrix = identity_matrix();

            fill::<Rgb8>(&mut seq, &clip, &path, &pipe, &src, &matrix, 1.0, false);
            fill_parallel::<Rgb8>(&mut par, &clip, &path, &pipe, &src, &matrix, 1.0, false, 4);

            assert_eq!(
                seq.data(),
                par.data(),
                "parallel fill output differs from sequential"
            );
        }

        /// A single-band parallel fill must produce identical output to sequential fill.
        #[test]
        fn fill_parallel_single_band_is_sequential() {
            const W: u32 = 32;
            const H: u32 = 512;

            let mut seq: Bitmap<Rgb8> = Bitmap::new(W, H, 1, false);
            let mut par: Bitmap<Rgb8> = Bitmap::new(W, H, 1, false);

            let clip = make_clip(W, H);
            let pipe = simple_pipe();
            let color = [33u8, 66, 99];
            let src = PipeSrc::Solid(&color);
            let path = rect_path(2.0, 2.0, 30.0, 510.0);
            let matrix = identity_matrix();

            fill::<Rgb8>(&mut seq, &clip, &path, &pipe, &src, &matrix, 1.0, false);
            // n_bands=1 must behave identically to sequential.
            fill_parallel::<Rgb8>(&mut par, &clip, &path, &pipe, &src, &matrix, 1.0, false, 1);

            assert_eq!(
                seq.data(),
                par.data(),
                "single-band parallel fill output differs from sequential"
            );
        }

        /// Parallel even-odd fill (n_bands=4) must match sequential eo_fill.
        #[test]
        fn eo_fill_parallel_matches_sequential() {
            const W: u32 = 64;
            const H: u32 = 512;

            let mut seq: Bitmap<Rgb8> = Bitmap::new(W, H, 1, false);
            let mut par: Bitmap<Rgb8> = Bitmap::new(W, H, 1, false);

            let clip = make_clip(W, H);
            let pipe = simple_pipe();
            let color = [200u8, 100, 50];
            let src = PipeSrc::Solid(&color);

            // Donut: outer 4→60, inner 16→48 → EO fills only the ring.
            let mut b = PathBuilder::new();
            b.move_to(4.0, 4.0).unwrap();
            b.line_to(60.0, 4.0).unwrap();
            b.line_to(60.0, 508.0).unwrap();
            b.line_to(4.0, 508.0).unwrap();
            b.close(true).unwrap();
            b.move_to(16.0, 16.0).unwrap();
            b.line_to(48.0, 16.0).unwrap();
            b.line_to(48.0, 496.0).unwrap();
            b.line_to(16.0, 496.0).unwrap();
            b.close(true).unwrap();
            let path = b.build();
            let matrix = identity_matrix();

            eo_fill::<Rgb8>(&mut seq, &clip, &path, &pipe, &src, &matrix, 1.0, false);
            eo_fill_parallel::<Rgb8>(&mut par, &clip, &path, &pipe, &src, &matrix, 1.0, false, 4);

            assert_eq!(
                seq.data(),
                par.data(),
                "parallel eo_fill output differs from sequential"
            );
        }
    }
}
