//! Rayon-parallel fill helpers.
//!
//! Everything in this module is gated on the `rayon` feature.  The public
//! entry points ([`fill_parallel`], [`eo_fill_parallel`]) are re-exported
//! from the parent module so the crate's public API is unchanged.

use crate::bitmap::{Bitmap, BitmapBand};
use crate::clip::{Clip, ClipResult};
use crate::path::Path;
use crate::pipe::{PipeSrc, PipeState};
use crate::scanner::XPathScanner;
use crate::scanner::iter::ScanIterator;
use crate::types::AA_SIZE;
use crate::xpath::XPath;
use color::Pixel;

use super::{draw_span, draw_span_clipped, fill_impl};

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

    // Bitmap height must fit in i32 so band y-coordinates can be compared with
    // scanner coordinates (which are i32). Any real PDF page is far below 2^31 px.
    assert!(
        i32::try_from(bitmap.height).is_ok(),
        "bitmap height {} exceeds i32::MAX; cannot compute band y-coordinates",
        bitmap.height,
    );

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
    // bitmap.height ≤ i32::MAX is asserted in fill_impl_parallel before bands_mut;
    // band.y_start and band.y_start + band.height - 1 are both ≤ bitmap.height.
    #[expect(
        clippy::cast_possible_wrap,
        reason = "band.y_start ≤ bitmap.height ≤ i32::MAX; asserted in fill_impl_parallel"
    )]
    let y_band_min = band.y_start as i32;
    #[expect(
        clippy::cast_possible_wrap,
        reason = "band.y_start + band.height - 1 ≤ bitmap.height - 1 ≤ i32::MAX; asserted in fill_impl_parallel"
    )]
    let y_band_max = (band.y_start + band.height - 1) as i32;

    // Clamp the scanner's y range to this band.
    let y_start = scanner.y_min.max(y_band_min);
    let y_end = scanner.y_max.min(y_band_max);

    if y_start > y_end {
        return;
    }

    if band.width == 0 {
        return;
    }
    #[expect(
        clippy::cast_possible_wrap,
        reason = "band.width ≤ bitmap.width ≤ i32::MAX; zero checked above"
    )]
    let width_i = band.width as i32;

    // Iterate only over scanlines in this band that have at least one intersection.
    for y in scanner
        .nonempty_rows()
        .filter(|&y| y >= y_start && y <= y_end)
    {
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
                draw_span::<P, _>(band, pipe, src, sx0, sx1, y);
            } else {
                draw_span_clipped::<P, _>(band, clip, pipe, src, sx0, sx1, y);
            }
        }
    }
}
