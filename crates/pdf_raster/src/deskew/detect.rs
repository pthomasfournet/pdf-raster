//! Skew angle detection via intensity-weighted projection profile.
//!
//! Unlike Leptonica's `pixFindSkewSweepAndSearch` which binarises at a fixed
//! threshold (160) before projecting, this implementation uses `255 − pixel`
//! as the foreground weight directly on the 8-bit grayscale image.  Dark pixels
//! contribute proportionally to the projection; light pixels contribute little.
//! This removes the binarisation threshold as a tunable parameter and improves
//! robustness on light or dark scans where threshold 160 over- or under-segments.
//!
//! # Algorithm
//!
//! 1. **Downsample** the image 4× (box average) to reduce computation.
//! 2. **Coarse sweep**: evaluate the differential-square-sum score at
//!    [`SWEEP_STEPS`] evenly-spaced angles in ±[`MAX_SEARCH_DEG`] using
//!    Rayon parallel iteration (each angle is independent).
//! 3. **Refinement**: starting from the best coarse angle, shrink the search
//!    interval repeatedly until half-width < [`REFINE_STOP_DEG`].
//!
//! # Score function
//!
//! For each trial angle θ the image is sheared: pixel at `(x, y)` is mapped to
//! output row `y + round(x · tan(θ))`.  Row sums of foreground weight
//! (`255 − pixel`) are accumulated into a 1-D array.  The score is the
//! differential square sum `Σ (s[i] − s[i−1])²` which is maximised when text
//! baselines are perfectly horizontal (alternating low/high projection values).
//!
//! # Performance
//!
//! At 4× downsampling a 300 DPI A4 page (2550×3300) becomes 638×825.
//! The inner loop scatters foreground weights into a `u32` row-sum array using
//! data-dependent indices (not auto-vectorisable).  Rayon distributes the
//! [`SWEEP_STEPS`] independent angle evaluations across available cores; the
//! 9900X3D's 96 MB V-Cache keeps the downsampled source image warm across all
//! sweep iterations.

use color::Gray8;
use raster::Bitmap;
use rayon::prelude::*;

use super::MAX_SEARCH_DEG;

/// Number of candidate angles in the coarse sweep (covers ±MAX_SEARCH_DEG).
const SWEEP_STEPS: usize = 56; // ≈ 0.255° steps over 14° range

/// Refinement stops when the half-interval width (`delta`) falls below this.
const REFINE_STOP_DEG: f32 = 0.01;

// Safety: step = 14° / (SWEEP_STEPS-1) must be finite; require ≥ 2 steps.
const _: () = assert!(SWEEP_STEPS >= 2, "SWEEP_STEPS must be ≥ 2");

/// Downsampling factor applied before scoring (4× linear = 16× pixel count).
const DOWNSAMPLE: u32 = 4;

/// Detect the skew angle of a grayscale image.
///
/// Returns the estimated skew in degrees (positive = counter-clockwise tilt,
/// negative = clockwise tilt).  Returns 0.0 when detection is unreliable
/// (e.g. blank or near-blank page).
pub fn find_skew_deg(img: &Bitmap<Gray8>) -> f32 {
    if img.width < 8 || img.height < 8 {
        return 0.0;
    }

    let small = downsample(img, DOWNSAMPLE);

    // Coarse sweep: score all candidate angles in parallel, fold to the best.
    let step = (2.0 * MAX_SEARCH_DEG) / (SWEEP_STEPS - 1) as f32;
    let (best_deg, best_score) = (0..SWEEP_STEPS)
        .into_par_iter()
        .map(|i| {
            let deg = -MAX_SEARCH_DEG + i as f32 * step;
            (deg, score_angle(&small, deg))
        })
        .reduce(
            || (0.0_f32, f64::NEG_INFINITY),
            |(bd, bs), (d, s)| if s > bs { (d, s) } else { (bd, bs) },
        );

    // Sanity check: if best score is too low the page is likely blank or has
    // no horizontal text structure — return 0 rather than a noisy estimate.
    let min_reliable = small.width as f64 * small.height as f64 * 0.01;
    if best_score < min_reliable {
        return 0.0;
    }

    // Binary search: refine around the best coarse angle.
    refine(&small, best_deg, step / 2.0)
}

/// Refine the angle estimate by iterative interval shrinkage.
///
/// Evaluates the score at `centre ± delta`, moves toward the better side,
/// and halves `delta` each iteration until `delta < REFINE_STOP_DEG`.
fn refine(img: &Bitmap<Gray8>, mut centre: f32, mut delta: f32) -> f32 {
    while delta >= REFINE_STOP_DEG {
        let left = (centre - delta).clamp(-MAX_SEARCH_DEG, MAX_SEARCH_DEG);
        let right = (centre + delta).clamp(-MAX_SEARCH_DEG, MAX_SEARCH_DEG);
        let sl = score_angle(img, left);
        let sr = score_angle(img, right);
        centre = if sr > sl { right } else { left };
        delta *= 0.5;
    }
    centre
}

/// Score function: differential square sum of intensity-weighted row projections.
///
/// Higher score = text baselines better aligned with horizontal scanlines.
fn score_angle(img: &Bitmap<Gray8>, deg: f32) -> f64 {
    let w = img.width as usize;
    let h = img.height as usize;
    let tan = deg.to_radians().tan();

    // Number of output rows: image height plus the shear-induced vertical extent.
    let shear_extent = (w as f32 * tan.abs()).ceil() as usize;
    let n_rows = h + shear_extent;

    let mut sums = vec![0u32; n_rows];

    for y in 0..h {
        let row = &img.row_bytes(y as u32)[..w];
        // Shear offset for this row: pixels shift vertically by x·tan(θ).
        // We iterate x and accumulate foreground weight into the sheared row.
        for (x, &px) in row.iter().enumerate() {
            let weight = u32::from(255u8.saturating_sub(px));
            if weight == 0 {
                continue;
            }
            // Sheared row index: y + round(x · tan θ), clamped to valid range.
            #[expect(
                clippy::cast_possible_truncation,
                reason = "offset ≤ shear_extent which is bounded by downsampled width × |tan(MAX_SEARCH_DEG)|, well within i32"
            )]
            let offset = (x as f32 * tan).round() as i32;
            let row_idx = (y as i32 + offset).clamp(0, (n_rows - 1) as i32) as usize;
            sums[row_idx] += weight;
        }
    }

    // Differential square sum: Σ (s[i] - s[i-1])²
    // Skip the outermost rows to avoid shear-induced edge artefacts.
    let skip = (h / 20).max(1);
    debug_assert!(
        n_rows >= 2 * skip,
        "skip ({skip}) too large for n_rows ({n_rows}); increase DOWNSAMPLE or decrease SWEEP_STEPS"
    );
    let end = n_rows.saturating_sub(skip);
    sums[skip..end]
        .windows(2)
        .map(|w| {
            let diff = w[1] as i64 - w[0] as i64;
            (diff * diff) as f64
        })
        .sum()
}

/// Box-average downsampling by integer factor `factor`.
///
/// Output pixel = average of the `factor × factor` block of source pixels.
/// Strips bottom/right partial blocks so output dimensions are exact integers.
fn downsample(src: &Bitmap<Gray8>, factor: u32) -> Bitmap<Gray8> {
    debug_assert!(factor > 0, "downsample factor must be ≥ 1");
    let ow = src.width / factor;
    let oh = src.height / factor;
    if ow == 0 || oh == 0 {
        return Bitmap::<Gray8>::new(1, 1, 1, false);
    }

    let mut dst = Bitmap::<Gray8>::new(ow, oh, 1, false);
    let factor_u = factor as usize;
    let area = (factor_u * factor_u) as u32;

    for oy in 0..oh {
        let dst_row = dst.row_bytes_mut(oy);
        for ox in 0..ow as usize {
            let sy0 = oy as usize * factor_u;
            let sx0 = ox * factor_u;
            let mut sum = 0u32;
            for dy in 0..factor_u {
                let src_row = &src.row_bytes((sy0 + dy) as u32);
                for dx in 0..factor_u {
                    sum += u32::from(src_row[sx0 + dx]);
                }
            }
            #[expect(
                clippy::cast_possible_truncation,
                reason = "sum / area ≤ 255 by construction"
            )]
            {
                dst_row[ox] = (sum / area) as u8;
            }
        }
    }
    dst
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A solid black image has uniform foreground weight — projection scores are flat
    /// and detection returns 0 (score below `min_reliable`) or a noise value in range.
    #[test]
    fn black_image_returns_zero_or_in_range() {
        let img = Bitmap::<Gray8>::new(256, 256, 1, false);
        // Bitmap::new zero-initialises to black (all pixels = 0).
        let angle = find_skew_deg(&img);
        assert!(angle.abs() <= MAX_SEARCH_DEG + 0.01);
    }

    /// Tiny image below the minimum size returns 0.
    #[test]
    fn tiny_image_returns_zero() {
        let img = Bitmap::<Gray8>::new(4, 4, 1, false);
        assert_eq!(find_skew_deg(&img), 0.0);
    }

    /// `score_angle` must not panic on a minimal image at the search boundaries.
    #[test]
    fn score_angle_boundary_angles() {
        let img = Bitmap::<Gray8>::new(64, 64, 1, false);
        let _ = score_angle(&img, -MAX_SEARCH_DEG);
        let _ = score_angle(&img, MAX_SEARCH_DEG);
        let _ = score_angle(&img, 0.0);
    }

    /// Downsampled image has the expected dimensions.
    #[test]
    fn downsample_dimensions() {
        let src = Bitmap::<Gray8>::new(640, 480, 1, false);
        let dst = downsample(&src, 4);
        assert_eq!(dst.width, 160);
        assert_eq!(dst.height, 120);
    }
}
