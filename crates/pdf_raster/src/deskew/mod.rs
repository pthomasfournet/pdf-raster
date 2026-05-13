//! Document deskew: detect skew angle and rotate to correct it.
//!
//! # Algorithm
//!
//! Two-phase pipeline:
//!
//! 1. **Detection** (`detect` module) — intensity-weighted projection-profile
//!    sweep.  Dark pixels are weighted by `255 − intensity` rather than binary
//!    foreground/background, so no binarisation threshold is needed.  AVX-512
//!    `VPSADBW` row sums, Rayon-parallel angle sweep, binary-search refinement
//!    to 0.01° convergence.
//!
//! 2. **Rotation** (`rotate` module) — bilinear rotation by the detected angle.
//!    GPU path (CUDA NPP `nppiRotate`) when the `gpu-deskew` feature is enabled
//!    and a CUDA device is available; CPU bilinear fallback otherwise.
//!    Rotation is clockwise-positive: a detected CCW tilt of +θ° is corrected
//!    by passing +θ° to `rotate_inplace` (CW rotation by θ°).
//!
//! # Usage
//!
//! ```rust,no_run
//! use color::Gray8;
//! use raster::Bitmap;
//! use pdf_raster::deskew;
//!
//! let mut page: Bitmap<Gray8> = todo!();
//! deskew::apply(&mut page).expect("deskew failed");
//! ```

pub(crate) mod detect;
pub(crate) mod rotate;

use color::Gray8;
use raster::Bitmap;

/// Minimum skew magnitude that triggers rotation.
///
/// Angles smaller than this are considered measurement noise and are not
/// corrected.  Matches Tesseract's own skip threshold for consistency.
const MIN_CORRECT_DEG: f32 = 0.05;

/// Maximum skew magnitude searched.  Scanner jams rarely exceed ±10°; PDFs
/// with larger rotation use `/Rotate` which is handled at the page-geometry
/// level, not here.
pub(crate) const MAX_SEARCH_DEG: f32 = 7.0;

/// Error type for deskew operations.
#[derive(Debug)]
pub struct DeskewError(pub(crate) String);

impl std::fmt::Display for DeskewError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for DeskewError {}

/// Detect the skew angle of `img` and rotate it in-place to correct it.
///
/// Does nothing if the detected angle is below `MIN_CORRECT_DEG`.
///
/// # Errors
///
/// Returns [`DeskewError`] if rotation fails (e.g. GPU allocation error).
/// Detection failure is treated as zero skew (no rotation applied) rather
/// than an error, since a missing correction is less harmful than a crash.
pub fn apply(img: &mut Bitmap<Gray8>) -> Result<(), DeskewError> {
    let angle_deg = detect::find_skew_deg(img);

    if angle_deg.abs() < MIN_CORRECT_DEG {
        return Ok(());
    }

    rotate::rotate_inplace(img, angle_deg)
}
