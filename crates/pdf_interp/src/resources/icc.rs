/// ICC CMYK→RGB CLUT baking via `moxcms`.
///
/// Evaluates a CMYK ICC profile on a regular `grid_n^4` lattice and returns the
/// result as a flat `u8` table indexed by `(k*G³+c*G²+m*G+y)*3`, matching the
/// layout expected by `GpuCtx::icc_cmyk_to_rgb(..., Some((table, grid_n)))` and
/// the CPU quadrilinear fallback in `gpu::icc_cmyk_to_rgb_cpu`.
///
/// # Grid size choice
///
/// 17 nodes per axis (the ICC minimum) is sufficient for perceptual rendering
/// and produces a 17^4 × 3 = 250 563-byte table. 33 nodes give higher accuracy
/// at 33^4 × 3 ≈ 3.4 MB — too large for L2 residency on most GPUs.
/// Default is 17, matching the ICC/PDF standard minimum.
use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash as _, Hasher as _};
use std::sync::Arc;

use moxcms::{CmsError, ColorProfile, Layout, TransformOptions};

pub const DEFAULT_GRID_N: u32 = 17;

/// Per-page cache of baked CMYK CLUT tables, keyed by a hash of the raw ICC profile bytes.
///
/// Most press PDFs embed the same profile in every image, so the first image pays the bake cost
/// (a few ms) and all subsequent images get an `Arc` clone.
pub type IccClutCache = HashMap<u64, Arc<[u8]>>;

/// Error returned by [`bake_cmyk_clut`].
#[derive(Debug)]
pub enum BakeError {
    /// `grid_n` is outside the valid range `[2, 255]`.
    InvalidGridSize(u32),
    /// The ICC profile is malformed, uses the wrong colour space, or the
    /// transform could not be built.
    Cms(CmsError),
}

impl std::fmt::Display for BakeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidGridSize(n) => {
                write!(f, "grid_n={n} is outside the valid range [2, 255]")
            }
            Self::Cms(e) => write!(f, "ICC CMS error: {e}"),
        }
    }
}

impl std::error::Error for BakeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Cms(e) => Some(e),
            Self::InvalidGridSize(_) => None,
        }
    }
}

impl From<CmsError> for BakeError {
    fn from(e: CmsError) -> Self {
        Self::Cms(e)
    }
}

/// Map a grid index in `[0, grid_n-1]` to an 8-bit channel value in `[0, 255]`.
///
/// `i` must satisfy `i ≤ 254`; `step` = `255.0 / (grid_n - 1)`.
/// The multiply-then-round ensures the endpoint `i = grid_n-1` lands exactly on 255.
#[inline]
fn grid_to_u8(i: usize, step: f32) -> u8 {
    // i ≤ 254, step ≤ 255.0 → product ≤ 64_770.0, well within f32 precision.
    // round() avoids accumulated floating-point drift near endpoints.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss,
        reason = "i ≤ 254 (exact in f32); i*step ∈ [0.0, 255.0] after round(), cast is lossless"
    )]
    {
        (i as f32 * step).round() as u8
    }
}

/// Bake a CMYK ICC profile into a quadrilinear CLUT table.
///
/// `icc_bytes` — raw ICC profile data (the stream body of an `ICCBased` PDF
/// colour space object with N=4).
///
/// `grid_n` — nodes per axis; must be in `[2, 255]`. Use [`DEFAULT_GRID_N`] (17)
/// for standard accuracy.
///
/// Returns a `Vec<u8>` of length `grid_n^4 * 3` with the sRGB output for each
/// CMYK lattice point, stored in `(k·G³+c·G²+m·G+y)*3` order.
///
/// # Errors
///
/// Returns [`BakeError::InvalidGridSize`] if `grid_n` is outside `[2, 255]`.
/// Returns [`BakeError::Cms`] if the ICC data is malformed or uses the wrong colour space.
#[expect(
    clippy::many_single_char_names,
    reason = "g/c/m/y/k are conventional CMYK grid-iteration names — clarity trumps length here"
)]
pub fn bake_cmyk_clut(icc_bytes: &[u8], grid_n: u32) -> Result<Vec<u8>, BakeError> {
    if !(2..=255).contains(&grid_n) {
        return Err(BakeError::InvalidGridSize(grid_n));
    }

    let src = ColorProfile::new_from_slice(icc_bytes)?;
    let dst = ColorProfile::new_srgb();

    // CMYK profiles require `Layout::Rgba` (4 bytes/pixel: C M Y K).
    // The destination is sRGB `Layout::Rgb` (3 bytes/pixel: R G B).
    let xform =
        src.create_transform_8bit(Layout::Rgba, &dst, Layout::Rgb, TransformOptions::default())?;

    let g = grid_n as usize; // grid_n ≤ 255, fits usize trivially
    let g2 = g * g;
    let g3 = g2 * g;
    // grid_n ≤ 255 → g^4 ≤ 255^4 ≈ 4.2 × 10^9; fits usize on 64-bit (our only target).
    // The checked_mul guards against any future 32-bit build attempt.
    let total = g3
        .checked_mul(g)
        .and_then(|n| n.checked_mul(3))
        .expect("grid_n^4*3 overflows usize — unreachable for grid_n ≤ 255 on 64-bit");

    // step = 255 / (grid_n - 1) maps grid index i ∈ [0, grid_n-1] to channel value [0, 255].
    // Using (grid_n - 1) as divisor ensures the last node (i = grid_n-1) maps exactly to 255,
    // not 255*(1 - 1/(grid_n-1)) ≈ 254.  grid_n ≥ 2 so denominator ≥ 1: no division by zero.
    #[expect(
        clippy::cast_precision_loss,
        reason = "grid_n ≤ 255, fits exactly in f32 (needs only 8 mantissa bits)"
    )]
    let step = 255.0_f32 / (grid_n - 1) as f32;

    // Process one K-slice (G^3 pixels) at a time to amortise per-call transform overhead.
    let mut src_buf = vec![0u8; g3 * 4];
    let mut dst_buf = vec![0u8; g3 * 3];
    let mut table = vec![0u8; total];

    for ki in 0..g {
        let k = grid_to_u8(ki, step);

        // Fill the K-slice input buffer: CMYK quads in (c, m, y) inner order.
        let mut off = 0;
        for ci in 0..g {
            let c = grid_to_u8(ci, step);
            for mi in 0..g {
                let m = grid_to_u8(mi, step);
                for yi in 0..g {
                    let y = grid_to_u8(yi, step);
                    src_buf[off] = c;
                    src_buf[off + 1] = m;
                    src_buf[off + 2] = y;
                    src_buf[off + 3] = k;
                    off += 4;
                }
            }
        }

        xform.transform(&src_buf[..g3 * 4], &mut dst_buf[..g3 * 3])?;

        // Scatter RGB triples from dst_buf into table in (k·G³+c·G²+m·G+y)*3 order.
        // The transform wrote pixels in the same (c, m, y) inner order we filled, so
        // src_off advances monotonically while the table index is computed by position.
        let mut src_off = 0;
        for ci in 0..g {
            for mi in 0..g {
                for yi in 0..g {
                    let idx = (ki * g3 + ci * g2 + mi * g + yi) * 3;
                    table[idx] = dst_buf[src_off];
                    table[idx + 1] = dst_buf[src_off + 1];
                    table[idx + 2] = dst_buf[src_off + 2];
                    src_off += 3;
                }
            }
        }
    }

    Ok(table)
}

/// Hash `icc_bytes` to a `u64` key for use in [`IccClutCache`].
fn hash_icc(icc_bytes: &[u8]) -> u64 {
    let mut h = DefaultHasher::new();
    icc_bytes.hash(&mut h);
    h.finish()
}

/// Bake a CMYK ICC profile CLUT, returning a cached `Arc` when the same profile
/// has already been baked during this page render.
///
/// On a cache hit the bake is skipped entirely.  On a miss the table is baked,
/// stored in `cache`, and returned as an `Arc`.
///
/// # Errors
///
/// Propagates [`BakeError`] from [`bake_cmyk_clut`] on a cache miss.
pub fn bake_cmyk_clut_cached(
    icc_bytes: &[u8],
    grid_n: u32,
    cache: &mut IccClutCache,
) -> Result<Arc<[u8]>, BakeError> {
    let key = hash_icc(icc_bytes);
    if let Some(arc) = cache.get(&key) {
        return Ok(Arc::clone(arc));
    }
    let table: Arc<[u8]> = bake_cmyk_clut(icc_bytes, grid_n)?.into();
    cache.insert(key, Arc::clone(&table));
    Ok(table)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_garbage_icc() {
        let result = bake_cmyk_clut(b"not an icc profile", DEFAULT_GRID_N);
        assert!(
            matches!(result, Err(BakeError::Cms(_))),
            "expected Cms error for garbage ICC bytes"
        );
    }

    #[test]
    fn rejects_grid_zero() {
        assert!(matches!(
            bake_cmyk_clut(b"", 0),
            Err(BakeError::InvalidGridSize(0))
        ));
    }

    #[test]
    fn rejects_grid_one() {
        assert!(matches!(
            bake_cmyk_clut(b"", 1),
            Err(BakeError::InvalidGridSize(1))
        ));
    }

    #[test]
    fn rejects_grid_too_large() {
        assert!(matches!(
            bake_cmyk_clut(b"", 256),
            Err(BakeError::InvalidGridSize(256))
        ));
    }
}
