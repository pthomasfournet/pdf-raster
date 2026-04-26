//! Rayon-parallel fill helpers.
//!
//! Everything in this module is gated on the `rayon` feature.  The public
//! entry points ([`fill_parallel`], [`eo_fill_parallel`]) are re-exported
//! from the parent module so the crate's public API is unchanged.

use crate::bitmap::{Bitmap, BitmapBand};
use crate::clip::{Clip, ClipResult};
use crate::path::Path;
use crate::pipe::{self, PipeSrc, PipeState};
use crate::scanner::XPathScanner;
use crate::scanner::iter::ScanIterator;
use crate::types::AA_SIZE;
use crate::xpath::XPath;
use color::Pixel;

use super::fill_impl;

/// Minimum fill height (in output pixel rows) for which the parallel path is
/// activated.  Below this threshold the thread-spawn overhead dominates and
/// sequential rendering is faster.
pub const PARALLEL_FILL_MIN_HEIGHT: u32 = 256;

/// Non-zero winding fill, parallelized across horizontal bands using rayon.
///
/// Only activated when `n_bands > 1` and the fill Y range spans at least
/// [`PARALLEL_FILL_MIN_HEIGHT`] rows.  Falls back to sequential [`fill`] otherwise.
#[expect(
    clippy::too_many_arguments,
    reason = "mirrors fill API; all params necessary"
)]
pub fn fill_parallel<P: Pixel + Send>(
    bitmap: &mut Bitmap<P>,
    clip: &Clip,
    path: &Path,
    pipe: &PipeState<'_>,
    src: &PipeSrc<'_>,
    matrix: &[f64; 6],
    flatness: f64,
    vector_antialias: bool,
    n_bands: usize,
) {
    fill_impl_parallel::<P>(
        bitmap,
        clip,
        path,
        pipe,
        src,
        matrix,
        flatness,
        vector_antialias,
        false,
        n_bands,
    );
}

/// Even-odd fill, parallelized across horizontal bands using rayon.
///
/// Only activated when `n_bands > 1` and the fill Y range spans at least
/// [`PARALLEL_FILL_MIN_HEIGHT`] rows.  Falls back to sequential [`eo_fill`] otherwise.
#[expect(
    clippy::too_many_arguments,
    reason = "mirrors eo_fill API; all params necessary"
)]
pub fn eo_fill_parallel<P: Pixel + Send>(
    bitmap: &mut Bitmap<P>,
    clip: &Clip,
    path: &Path,
    pipe: &PipeState<'_>,
    src: &PipeSrc<'_>,
    matrix: &[f64; 6],
    flatness: f64,
    vector_antialias: bool,
    n_bands: usize,
) {
    fill_impl_parallel::<P>(
        bitmap,
        clip,
        path,
        pipe,
        src,
        matrix,
        flatness,
        vector_antialias,
        true,
        n_bands,
    );
}

/// Inner implementation for parallel fill.
///
/// Falls back to [`fill_impl`] when the Y range is too small or `n_bands == 1`.
#[expect(
    clippy::too_many_arguments,
    reason = "mirrors fill_impl API; all params necessary"
)]
fn fill_impl_parallel<P: Pixel + Send>(
    bitmap: &mut Bitmap<P>,
    clip: &Clip,
    path: &Path,
    pipe: &PipeState<'_>,
    src: &PipeSrc<'_>,
    matrix: &[f64; 6],
    flatness: f64,
    vector_antialias: bool,
    eo: bool,
    n_bands: usize,
) {
    use rayon::iter::{IntoParallelIterator, ParallelIterator};

    if path.pts.is_empty() {
        return;
    }

    // Fall back to sequential for trivially small fills or single-band requests.
    if n_bands <= 1 || bitmap.height < PARALLEL_FILL_MIN_HEIGHT {
        fill_impl::<P>(
            bitmap,
            clip,
            path,
            pipe,
            src,
            matrix,
            flatness,
            vector_antialias,
            eo,
        );
        return;
    }

    // Build XPath + optional AA scaling once; shared read-only across all bands.
    let mut xpath = XPath::new(path, matrix, flatness, true);

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

    // Pixel-space bbox.
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

    // Only the non-AA path is parallelized; the AA path requires per-row shared
    // mutable state (AaBuf), which doesn't decompose cleanly into disjoint bands.
    // Fall back to sequential for the AA case.
    if vector_antialias {
        fill_impl::<P>(
            bitmap,
            clip,
            path,
            pipe,
            src,
            matrix,
            flatness,
            vector_antialias,
            eo,
        );
        return;
    }

    // Split the bitmap into disjoint horizontal bands; render each in parallel.
    // Each band borrows a disjoint slice of the bitmap data, so there are no
    // aliasing hazards. The scanner and pipe are shared read-only across bands.
    let bands = bitmap.bands_mut(n_bands);

    bands.into_par_iter().for_each(|mut band| {
        fill_band::<P>(&mut band, clip, pipe, src, &scanner, clip_res);
    });
}

/// Render the non-AA fill loop restricted to a single horizontal band.
///
/// `band.y_start` and `band.height` determine the y-range processed.
/// `y` coordinates passed to span routines are absolute (matching the parent bitmap).
/// Only called when `vector_antialias = false`; the AA path falls back to sequential.
fn fill_band<P: Pixel>(
    band: &mut BitmapBand<'_, P>,
    clip: &Clip,
    pipe: &PipeState<'_>,
    src: &PipeSrc<'_>,
    scanner: &XPathScanner,
    clip_res: ClipResult,
) {
    #[expect(
        clippy::cast_possible_wrap,
        reason = "y_start fits in i32 for valid PDF coordinates"
    )]
    let y_band_min = band.y_start as i32;
    #[expect(
        clippy::cast_possible_wrap,
        reason = "y_start + height fits in i32 for valid PDF coordinates"
    )]
    let y_band_max = (band.y_start + band.height - 1) as i32;

    // Clamp the scanner's y range to this band.
    let y_start = scanner.y_min.max(y_band_min);
    let y_end = scanner.y_max.min(y_band_max);

    if y_start > y_end {
        return;
    }

    #[expect(
        clippy::cast_possible_wrap,
        reason = "band.width ≤ i32::MAX in practice"
    )]
    let width_i = band.width as i32;

    // y_start >= y_band_min >= 0 (band.y_start is u32), so y is always non-negative.
    for y in y_start..=y_end {
        for (x0, x1) in ScanIterator::new(scanner, y) {
            let (mut sx0, mut sx1) = (x0, x1);
            let inner_clip = if clip_res == ClipResult::AllInside {
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
                draw_span_band::<P>(band, pipe, src, sx0, sx1, y);
            } else {
                draw_span_band_clipped::<P>(band, clip, pipe, src, sx0, sx1, y);
            }
        }
    }
}

/// Emit a solid span into a band row that is fully inside the clip.
fn draw_span_band<P: Pixel>(
    band: &mut BitmapBand<'_, P>,
    pipe: &PipeState<'_>,
    src: &PipeSrc<'_>,
    x0: i32,
    x1: i32,
    y: i32,
) {
    debug_assert!(x0 <= x1);
    debug_assert!(y >= 0);
    #[expect(clippy::cast_sign_loss, reason = "y >= 0")]
    let y_u = y as u32;
    #[expect(clippy::cast_sign_loss, reason = "x0 >= 0 after clip clamping")]
    let byte_off = x0 as usize * P::BYTES;
    #[expect(clippy::cast_sign_loss, reason = "x1 >= x0 >= 0")]
    let byte_end = (x1 as usize + 1) * P::BYTES;
    #[expect(clippy::cast_sign_loss, reason = "x0 >= 0, x1 >= x0")]
    let alpha_range = x0 as usize..=x1 as usize;

    let (row, alpha) = band.row_and_alpha_mut(y_u);
    let dst_pixels = &mut row[byte_off..byte_end];
    let dst_alpha = alpha.map(|a| &mut a[alpha_range]);

    pipe::render_span::<P>(pipe, src, dst_pixels, dst_alpha, None, x0, x1, y);
}

/// Emit a span with per-pixel clip test into a band row.
fn draw_span_band_clipped<P: Pixel>(
    band: &mut BitmapBand<'_, P>,
    clip: &Clip,
    pipe: &PipeState<'_>,
    src: &PipeSrc<'_>,
    x0: i32,
    x1: i32,
    y: i32,
) {
    let mut run_start: Option<i32> = None;
    for x in x0..=x1 {
        if clip.test(x, y) {
            if run_start.is_none() {
                run_start = Some(x);
            }
        } else if let Some(rs) = run_start.take() {
            draw_span_band::<P>(band, pipe, src, rs, x - 1, y);
        }
    }
    if let Some(rs) = run_start {
        draw_span_band::<P>(band, pipe, src, rs, x1, y);
    }
}
