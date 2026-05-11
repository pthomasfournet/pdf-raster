//! Image rotation for deskew correction.
//!
//! Rotates an 8-bit grayscale bitmap by a given angle (in degrees, CW positive)
//! using bilinear interpolation.  Background pixels introduced by the rotation
//! are set to 255 (white — scanner background convention).
//!
//! # Implementation
//!
//! **CPU path** (always available): per-output-pixel inverse mapping with
//! bilinear interpolation.  The inner loop is written in the per-arch dispatch
//! pattern used throughout the codebase:
//!
//! - **aarch64**: `rotate_row_neon` processes 4 output pixels per iteration
//!   using `vmlaq_f32` for coordinate stepping and `vcvtmq_s32_f32` for floor.
//!   Any pixel whose source coordinates are out-of-bounds is handled by the
//!   scalar fallback within the same loop.
//! - **x86-64 / other**: scalar per-pixel loop; LLVM auto-vectorises to
//!   AVX-512 on `-C target-cpu=native` (Ryzen).
//!
//! **GPU path** (`gpu-deskew` feature, optional): CUDA NPP `nppiRotate_8u_C1R_Ctx`
//! with hardware bilinear via texture units.  ~0.3–0.5 ms for a 2550×3300 image
//! on RTX 5070.  Falls back to CPU silently when no CUDA device is available.

use color::Gray8;
use raster::Bitmap;

use super::DeskewError;

/// Rotate `img` in-place by `angle_deg` degrees (clockwise positive).
///
/// The image is replaced with a new bitmap of the same dimensions containing
/// the rotated content.  Out-of-bounds source pixels are filled with 255 (white).
///
/// When the `gpu-deskew` feature is enabled, attempts GPU rotation first and
/// falls back to the CPU bilinear path on any GPU error.
///
/// # Errors
///
/// Currently always returns `Ok(())` — the CPU path cannot fail beyond OOM
/// (which panics).  The return type is `Result` to accommodate future GPU-only
/// configurations where CPU fallback may not be available.
pub fn rotate_inplace(img: &mut Bitmap<Gray8>, angle_deg: f32) -> Result<(), DeskewError> {
    #[cfg(feature = "gpu-deskew")]
    match rotate_gpu(img, angle_deg) {
        Ok(rotated) => {
            *img = rotated;
            return Ok(());
        }
        Err(e) => {
            log::warn!("rotate_inplace: GPU rotation failed ({e}); falling back to CPU");
        }
    }

    *img = rotate_cpu(img, angle_deg);
    Ok(())
}

// ── Shared helpers ────────────────────────────────────────────────────────────

/// Parameters shared between the scalar and NEON inner loops.
///
/// Passed by value to avoid threading individual scalars through dispatch helpers.
struct RowParams {
    cos_a: f32,
    sin_a: f32,
    sx_base: f32,
    sy_base: f32,
    sx_max: i32,
    sy_max: i32,
}

/// Bilinear interpolation at source `(sx, sy)` given pre-floored origin `(x0, y0)`.
///
/// `x0` and `y0` must already be bounds-checked (`0 ≤ x0 ≤ sx_max`, `0 ≤ y0 ≤ sy_max`).
#[inline]
fn bilinear_sample(src: &Bitmap<Gray8>, x0: usize, y0: usize, fx: f32, fy: f32) -> u8 {
    // y0 ≤ sy_max ≤ h-2 and y0+1 ≤ h-1 ≤ 32767 — safe casts.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "y0 ≤ h-2 ≤ 32766 and y0+1 ≤ h-1 ≤ 32767; both fit in u32"
    )]
    let (r0, r1) = (src.row_bytes(y0 as u32), src.row_bytes((y0 + 1) as u32));

    let p00 = f32::from(r0[x0]);
    let p10 = f32::from(r0[x0 + 1]);
    let p01 = f32::from(r1[x0]);
    let p11 = f32::from(r1[x0 + 1]);

    let top = (p10 - p00).mul_add(fx, p00);
    let bot = (p11 - p01).mul_add(fx, p01);
    let val = (bot - top).mul_add(fy, top);

    // Bilinear of values in [0, 255] stays in [0, 255].
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "bilinear of [0,255] inputs stays in [0,255]; rounding keeps it non-negative"
    )]
    {
        val.round() as u8
    }
}

/// Compute a single output pixel at column `ox`.
///
/// Returns 255 (white) when the source coordinates are out of the valid
/// bilinear-sampling range.
#[inline]
fn rotate_pixel_scalar(src: &Bitmap<Gray8>, ox: f32, p: &RowParams) -> u8 {
    let sx = p.cos_a.mul_add(ox, p.sx_base);
    let sy = p.sin_a.mul_add(ox, p.sy_base);

    // floor() → i32: coordinates bounded by ≤ 32768·√2 ≈ 46341 — fits in i32.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "image coordinates bounded by rotated extent ≤ 32768 * sqrt(2) ≈ 46341; fits in i32"
    )]
    let x0 = sx.floor() as i32;
    #[expect(
        clippy::cast_possible_truncation,
        reason = "image coordinates bounded by rotated extent ≤ 32768 * sqrt(2) ≈ 46341; fits in i32"
    )]
    let y0 = sy.floor() as i32;

    if x0 < 0 || y0 < 0 || x0 > p.sx_max || y0 > p.sy_max {
        return 255;
    }

    // x0 ≥ 0, y0 ≥ 0 verified above.
    #[expect(
        clippy::cast_sign_loss,
        reason = "x0 ≥ 0 and y0 ≥ 0 verified by the guard above"
    )]
    let (x0u, y0u) = (x0 as usize, y0 as usize);
    // x0 ≤ 46341 ≤ 2^23 — exactly representable in f32; subtraction is exact.
    #[expect(
        clippy::cast_precision_loss,
        reason = "x0 ≤ 46341 ≤ 2^23 — exactly representable in f32"
    )]
    let fx = sx - x0 as f32;
    #[expect(
        clippy::cast_precision_loss,
        reason = "y0 ≤ 46341 ≤ 2^23 — exactly representable in f32"
    )]
    let fy = sy - y0 as f32;

    bilinear_sample(src, x0u, y0u, fx, fy)
}

// ── Per-arch row implementations ──────────────────────────────────────────────

/// NEON inner-loop: process one output row using `vmlaq_f32` coordinate stepping.
///
/// Handles 4 output pixels per iteration.  For any group of 4 where at least one
/// pixel's source coordinates are out-of-bounds, the entire group is processed
/// pixel-by-pixel via `rotate_pixel_scalar` (which fills OOB pixels with white).
/// A scalar tail handles any remaining pixels at the end of the row.
///
/// # Safety
///
/// NEON is mandatory on all ARMv8-A targets; no runtime check is needed.
/// `dst_row.len() == w` must hold (caller guarantees this via `row_bytes_mut`).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn rotate_row_neon(dst_row: &mut [u8], src: &Bitmap<Gray8>, p: &RowParams, w: usize) {
    use std::arch::aarch64::{
        vandq_u32, vcgeq_f32, vcleq_f32, vcvtmq_s32_f32, vdupq_n_f32, vgetq_lane_f32,
        vgetq_lane_s32, vmaxvq_u32, vmlaq_f32, vmvnq_u32,
    };

    // Lane offsets [0.0, 1.0, 2.0, 3.0]: dx[i] = ox + i for i in 0..4.
    let v_offsets = unsafe { std::arch::aarch64::vld1q_f32([0.0f32, 1.0, 2.0, 3.0].as_ptr()) };

    let v_cos = vdupq_n_f32(p.cos_a);
    let v_sin = vdupq_n_f32(p.sin_a);
    let v_sx_base = vdupq_n_f32(p.sx_base);
    let v_sy_base = vdupq_n_f32(p.sy_base);
    let v_zero = vdupq_n_f32(0.0);
    // sx_max / sy_max ≤ 32766 ≤ 2^23 — exactly representable in f32.
    #[expect(
        clippy::cast_precision_loss,
        reason = "sx_max / sy_max ≤ 32766 ≤ 2^23 — exactly representable in f32"
    )]
    let v_sx_max = vdupq_n_f32(p.sx_max as f32);
    #[expect(
        clippy::cast_precision_loss,
        reason = "sy_max ≤ 32766 ≤ 2^23 — exactly representable in f32"
    )]
    let v_sy_max = vdupq_n_f32(p.sy_max as f32);

    let mut ox = 0usize;

    while ox + 4 <= w {
        // dx = [ox, ox+1, ox+2, ox+3] as f32.
        // ox ≤ 32768; exact in f32.
        #[expect(
            clippy::cast_precision_loss,
            reason = "ox ≤ MAX_PX_DIMENSION = 32768; exact in f32"
        )]
        let v_dx = unsafe { std::arch::aarch64::vaddq_f32(vdupq_n_f32(ox as f32), v_offsets) };

        // sx[i] = sx_base + cos_a * dx[i]
        // sy[i] = sy_base + sin_a * dx[i]
        let v_sx = unsafe { vmlaq_f32(v_sx_base, v_cos, v_dx) };
        let v_sy = unsafe { vmlaq_f32(v_sy_base, v_sin, v_dx) };

        // in_bounds[i] = sx[i] ∈ [0, sx_max] && sy[i] ∈ [0, sy_max].
        // vcgeq_f32 / vcleq_f32: lane = 0xFFFF_FFFF if true, 0 if false.
        let in_bounds = unsafe {
            vandq_u32(
                vandq_u32(vcgeq_f32(v_sx, v_zero), vcleq_f32(v_sx, v_sx_max)),
                vandq_u32(vcgeq_f32(v_sy, v_zero), vcleq_f32(v_sy, v_sy_max)),
            )
        };

        // all_in: vmvnq_u32 flips: OOB lane → 0xFFFF_FFFF, IB lane → 0.
        // vmaxvq_u32 == 0 iff every lane was IB (flipped to 0).
        let all_in = unsafe { vmaxvq_u32(vmvnq_u32(in_bounds)) == 0 };

        if all_in {
            // All 4 source coords are in-bounds: SIMD floor + scalar gather + bilinear.
            // vcvtmq_s32_f32: floor f32 → i32 (round toward −∞).
            let v_x0 = unsafe { vcvtmq_s32_f32(v_sx) };
            let v_y0 = unsafe { vcvtmq_s32_f32(v_sy) };

            // vgetq_lane_* requires a const lane index — extract all 8 values explicitly.
            let x0_0 = unsafe { vgetq_lane_s32(v_x0, 0) };
            let x0_1 = unsafe { vgetq_lane_s32(v_x0, 1) };
            let x0_2 = unsafe { vgetq_lane_s32(v_x0, 2) };
            let x0_3 = unsafe { vgetq_lane_s32(v_x0, 3) };
            let y0_0 = unsafe { vgetq_lane_s32(v_y0, 0) };
            let y0_1 = unsafe { vgetq_lane_s32(v_y0, 1) };
            let y0_2 = unsafe { vgetq_lane_s32(v_y0, 2) };
            let y0_3 = unsafe { vgetq_lane_s32(v_y0, 3) };

            let sx_0 = unsafe { vgetq_lane_f32(v_sx, 0) };
            let sx_1 = unsafe { vgetq_lane_f32(v_sx, 1) };
            let sx_2 = unsafe { vgetq_lane_f32(v_sx, 2) };
            let sx_3 = unsafe { vgetq_lane_f32(v_sx, 3) };
            let sy_0 = unsafe { vgetq_lane_f32(v_sy, 0) };
            let sy_1 = unsafe { vgetq_lane_f32(v_sy, 1) };
            let sy_2 = unsafe { vgetq_lane_f32(v_sy, 2) };
            let sy_3 = unsafe { vgetq_lane_f32(v_sy, 3) };

            // x0 ≥ 0 (all_in), x0 ≤ sx_max ≤ 32766 ≤ 46341 ≤ 2^23: safe sign-loss and precision casts.
            #[expect(
                clippy::cast_sign_loss,
                reason = "x0 ≥ 0 and y0 ≥ 0 verified by the all_in OOB check"
            )]
            #[expect(
                clippy::cast_precision_loss,
                reason = "x0 / y0 ≤ 46341 ≤ 2^23 — exactly representable in f32"
            )]
            {
                dst_row[ox] = bilinear_sample(
                    src,
                    x0_0 as usize,
                    y0_0 as usize,
                    sx_0 - x0_0 as f32,
                    sy_0 - y0_0 as f32,
                );
                dst_row[ox + 1] = bilinear_sample(
                    src,
                    x0_1 as usize,
                    y0_1 as usize,
                    sx_1 - x0_1 as f32,
                    sy_1 - y0_1 as f32,
                );
                dst_row[ox + 2] = bilinear_sample(
                    src,
                    x0_2 as usize,
                    y0_2 as usize,
                    sx_2 - x0_2 as f32,
                    sy_2 - y0_2 as f32,
                );
                dst_row[ox + 3] = bilinear_sample(
                    src,
                    x0_3 as usize,
                    y0_3 as usize,
                    sx_3 - x0_3 as f32,
                    sy_3 - y0_3 as f32,
                );
            }
        } else {
            // At least one OOB lane — fall back to per-pixel scalar for all 4.
            // rotate_pixel_scalar handles OOB with white fill.
            #[expect(
                clippy::cast_precision_loss,
                reason = "ox ≤ MAX_PX_DIMENSION = 32768; exact in f32"
            )]
            for i in 0..4usize {
                dst_row[ox + i] = rotate_pixel_scalar(src, (ox + i) as f32, p);
            }
        }

        ox += 4;
    }

    // Scalar tail: remaining pixels where ox + 4 > w.
    #[expect(
        clippy::cast_precision_loss,
        reason = "i ≤ MAX_PX_DIMENSION = 32768; exact in f32"
    )]
    for i in ox..w {
        dst_row[i] = rotate_pixel_scalar(src, i as f32, p);
    }
}

// ── Per-arch dispatch ─────────────────────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
#[inline]
fn dispatch_rotate_row(dst_row: &mut [u8], src: &Bitmap<Gray8>, p: &RowParams) {
    let w = dst_row.len();
    // SAFETY: NEON mandatory on all ARMv8-A targets.
    unsafe { rotate_row_neon(dst_row, src, p, w) }
}

#[cfg(not(target_arch = "aarch64"))]
#[inline]
fn dispatch_rotate_row(dst_row: &mut [u8], src: &Bitmap<Gray8>, p: &RowParams) {
    for (ox, dst_px) in dst_row.iter_mut().enumerate() {
        #[expect(
            clippy::cast_precision_loss,
            reason = "ox ≤ MAX_PX_DIMENSION = 32768; exact in f32"
        )]
        {
            *dst_px = rotate_pixel_scalar(src, ox as f32, p);
        }
    }
}

// ── Public CPU path ───────────────────────────────────────────────────────────

/// CPU bilinear rotation (clockwise-positive).
///
/// Uses inverse mapping: for each output pixel `(ox, oy)`, compute the
/// corresponding source coordinates `(sx, sy)` by applying a CW rotation
/// matrix centred on the image centre, then bilinear-interpolate from the
/// four surrounding source pixels.
///
/// Out-of-bounds source pixels are filled with 255 (white).  The rightmost
/// and bottom source columns/rows are treated as out-of-bounds because bilinear
/// sampling requires a 2×2 neighbourhood — the last valid origin is (w-2, h-2).
#[expect(
    clippy::similar_names,
    reason = "sx_*/sy_* are paired coordinate variables; renaming would obscure the symmetry"
)]
pub(crate) fn rotate_cpu(src: &Bitmap<Gray8>, angle_deg: f32) -> Bitmap<Gray8> {
    let w = src.width as usize;
    let h = src.height as usize;

    let mut dst = Bitmap::<Gray8>::new(src.width, src.height, 1, false);

    let rad = angle_deg.to_radians();
    let cos_a = rad.cos();
    let sin_a = rad.sin();

    // Rotation centre: image centre (may be fractional).
    // w, h ≤ MAX_PX_DIMENSION = 32768; exact in f32 (24-bit mantissa covers integers to 16 M).
    #[expect(
        clippy::cast_precision_loss,
        reason = "image dimensions ≤ MAX_PX_DIMENSION = 32768; exact in f32"
    )]
    let cx = (w as f32 - 1.0) * 0.5;
    #[expect(
        clippy::cast_precision_loss,
        reason = "image dimensions ≤ MAX_PX_DIMENSION = 32768; exact in f32"
    )]
    let cy = (h as f32 - 1.0) * 0.5;

    // Valid source coordinate range for bilinear sampling: x ∈ [0, w-2], y ∈ [0, h-2].
    // w, h ≤ 32768 — safe to cast to i32 (fits in i32 range).
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        reason = "image dimensions ≤ MAX_PX_DIMENSION = 32768; fits in i32"
    )]
    let sx_max = (w as i32) - 2;
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        reason = "image dimensions ≤ MAX_PX_DIMENSION = 32768; fits in i32"
    )]
    let sy_max = (h as i32) - 2;

    for oy in 0..h {
        // oy < h ≤ 32768; safe to cast to u32.
        #[expect(
            clippy::cast_possible_truncation,
            reason = "oy < h ≤ MAX_PX_DIMENSION = 32768; fits in u32"
        )]
        let dst_row = dst.row_bytes_mut(oy as u32);

        // Pre-compute y-dependent terms outside the x loop.
        // oy ≤ 32768; exact in f32.
        #[expect(
            clippy::cast_precision_loss,
            reason = "oy ≤ MAX_PX_DIMENSION = 32768; exact in f32"
        )]
        let dy = oy as f32 - cy;
        let sx_base = cos_a.mul_add(-cx, -(sin_a * dy)) + cx;
        let sy_base = sin_a.mul_add(-cx, cos_a * dy) + cy;

        let params = RowParams {
            cos_a,
            sin_a,
            sx_base,
            sy_base,
            sx_max,
            sy_max,
        };
        dispatch_rotate_row(dst_row, src, &params);
    }

    dst
}

/// GPU rotation via CUDA NPP (`nppiRotate_8u_C1R_Ctx`).
#[cfg(feature = "gpu-deskew")]
fn rotate_gpu(src: &Bitmap<Gray8>, angle_deg: f32) -> Result<Bitmap<Gray8>, DeskewError> {
    let pixels =
        gpu::npp_rotate::rotate_gray8(src.data(), src.stride, src.width, src.height, angle_deg)
            .map_err(|e| DeskewError(e.to_string()))?;

    // Reconstruct a tightly-packed Bitmap from the returned pixel Vec.
    // rotate_gray8 returns width*height bytes with no stride padding.
    let mut dst = Bitmap::<Gray8>::new(src.width, src.height, 1, false);
    dst.data_mut().copy_from_slice(&pixels);
    Ok(dst)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Rotating by 0° is a no-op (pixels should be unchanged).
    #[test]
    fn rotate_zero_is_identity() {
        let mut src = Bitmap::<Gray8>::new(64, 64, 1, false);
        // Write a checkerboard pattern.
        for y in 0..64u32 {
            let row = src.row_bytes_mut(y);
            for (x, px) in row.iter_mut().enumerate() {
                *px = if (x + y as usize).is_multiple_of(2) {
                    0
                } else {
                    128
                };
            }
        }
        let rotated = rotate_cpu(&src, 0.0);
        // Centre pixels should be identical; edges may differ (white fill vs original).
        let r_src = &src.row_bytes(32)[1..63];
        let r_dst = &rotated.row_bytes(32)[1..63];
        assert_eq!(r_src, r_dst, "centre row should be unchanged at 0°");

        // Pin the OOB-guard strict-inequality boundaries: at 0° the inverse
        // map is identity, so output `(0, y)` samples source `x0 = 0` and
        // output `(sx_max, y)` samples `x0 = sx_max = w - 2 = 62`.  If the
        // guard `<` → `<=`, x0=0 wrongly becomes OOB → 255.  If `>` → `>=`,
        // x0=62 wrongly becomes OOB → 255.  The checkerboard has 0 at both
        // boundaries on row 32, so a white-fill mutation is unmissable.
        assert_eq!(
            rotated.row_bytes(32)[0],
            src.row_bytes(32)[0],
            "x0 = 0 must be in-bounds (guards `x0 < 0`, not `x0 <= 0`)"
        );
        assert_eq!(
            rotated.row_bytes(32)[62],
            src.row_bytes(32)[62],
            "x0 = sx_max must be in-bounds (guards `x0 > sx_max`, not `x0 >= sx_max`)"
        );
        // And the y-axis boundaries: row 0 and row 62.
        assert_eq!(
            rotated.row_bytes(0)[32],
            src.row_bytes(0)[32],
            "y0 = 0 must be in-bounds (guards `y0 < 0`, not `y0 <= 0`)"
        );
        assert_eq!(
            rotated.row_bytes(62)[32],
            src.row_bytes(62)[32],
            "y0 = sy_max must be in-bounds (guards `y0 > sy_max`, not `y0 >= sy_max`)"
        );
    }

    /// `rotate_inplace` at 0° does not panic.
    #[test]
    fn rotate_inplace_zero_no_panic() {
        let mut img = Bitmap::<Gray8>::new(32, 32, 1, false);
        rotate_inplace(&mut img, 0.0).unwrap();
    }

    /// `rotate_inplace` must actually mutate `*img`, not return Ok and leave
    /// the input untouched.  Pins the `rotate_inplace → Ok(())` mutant by
    /// asserting visible pixel content changes after a non-trivial rotation.
    #[test]
    fn rotate_inplace_modifies_pixels() {
        // 32×32 image with a single white-stripe column on the left half so
        // any non-no-op rotation visibly redistributes the white.
        let mut img = Bitmap::<Gray8>::new(32, 32, 1, false);
        for y in 0..32u32 {
            let row = img.row_bytes_mut(y);
            row[5] = 255; // vertical white line at x=5
        }
        let before_col5 = (0..32u32).map(|y| img.row_bytes(y)[5]).collect::<Vec<_>>();
        // 45° rotation breaks the vertical line into diagonal pieces.
        rotate_inplace(&mut img, 45.0).unwrap();
        let after_col5 = (0..32u32).map(|y| img.row_bytes(y)[5]).collect::<Vec<_>>();
        assert_ne!(
            before_col5, after_col5,
            "rotate_inplace at 45° must mutate the bitmap, not no-op"
        );
    }

    /// The centre pixel is the rotation pivot — at any angle it maps to
    /// itself, so the rotated image's centre must equal the source's centre
    /// (within bilinear-rounding tolerance).  This pins the centre-pivot
    /// shift formulae (`cx`, `cy` and the offset terms in `sx_base`/
    /// `sy_base`): any operator swap that mis-locates the pivot would shift
    /// the centre pixel away from its source value.
    ///
    /// Uses a centre-asymmetric pattern (single bright dot at the centre,
    /// black elsewhere) so any pivot drift onto a neighbouring black pixel
    /// changes the centre value dramatically.  A smooth gradient would let
    /// pivot-off-by-one mutations slip through because the bilinear samples
    /// of an off-by-one pivot still approximate the same gradient value.
    ///
    /// Tests three angles (12°, 31.7°, 89°) so a mutation that happens to
    /// preserve the pivot at one specific angle still fails at another.
    #[test]
    fn rotate_cpu_centre_pixel_is_rotation_fixed_point() {
        // Mid-grey dot at the centre (not 255) so the OOB-white fallback
        // (255) on a coordinate corruption is visibly different from a true
        // centre sample — catches mutations that produce NaN source
        // coordinates and silently route through the OOB guard.
        const DOT: u8 = 128;
        // Odd dimensions so cx, cy land on integer pixel centres (16.0).
        let mut src = Bitmap::<Gray8>::new(33, 33, 1, false);
        src.row_bytes_mut(16)[16] = DOT;

        for angle in [12.0_f32, 31.7, 89.0] {
            let rotated = rotate_cpu(&src, angle);
            let dst_centre = rotated.row_bytes(16)[16];
            // True bilinear sample at the rotated centre averages the DOT
            // pixel with some-fraction-of-zero neighbours, landing in
            // [~80, 128] depending on angle.  An off-by-one pivot would
            // sample 4 zero pixels (~0); a NaN-coordinate mutation would
            // OOB-fallback to 255.  Bracket both failure modes.
            assert!(
                (50..=200).contains(&dst_centre),
                "centre pixel must be a true bilinear sample of the centre dot at {angle}°: \
                 got dst={dst_centre} (expected ~50..=200); \
                 ~0 = pivot drift, 255 = NaN/OOB"
            );
        }
    }

    /// `rotate_cpu(rotate_cpu(S, 180), 180)` must equal S on the interior,
    /// modulo small bilinear-rounding error.  Catches any inner-loop
    /// arithmetic mutation that doesn't self-invert (e.g. `*` → `+` would
    /// break the double-rotation identity).
    #[test]
    fn rotate_cpu_180_then_180_round_trips() {
        let mut src = Bitmap::<Gray8>::new(32, 32, 1, false);
        for y in 0..32u32 {
            let row = src.row_bytes_mut(y);
            for (x, px) in row.iter_mut().enumerate() {
                // (x * 7 + y * 11) % 256 keeps the cast in [0, 255].
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "modulo 256 keeps the value in [0, u8::MAX]"
                )]
                {
                    *px = ((x * 7 + y as usize * 11) % 256) as u8;
                }
            }
        }

        let once = rotate_cpu(&src, 180.0);
        let twice = rotate_cpu(&once, 180.0);

        // Interior pixels only — borders accumulate two white-fill rounds.
        // Bilinear sampling at 180° hits half-pixel coordinates (cx/cy are
        // .5 in odd-sized images, 31.5 here), so a ±2 tolerance absorbs the
        // double-round error.
        let mut max_diff = 0i32;
        for y in 4..28u32 {
            let sr = &src.row_bytes(y)[4..28];
            let tr = &twice.row_bytes(y)[4..28];
            for (&a, &b) in sr.iter().zip(tr.iter()) {
                max_diff = max_diff.max((i32::from(a) - i32::from(b)).abs());
            }
        }
        assert!(
            max_diff <= 2,
            "180°×2 round-trip diverges from identity: max diff {max_diff} > 2"
        );
    }

    /// A 1-pixel-wide source rotated by a small angle must have white-fill
    /// (255) at the diagonal corners where the inverse mapping falls
    /// outside the source.  Pins the OOB-guard bounds checks
    /// (`x0 < 0 || y0 < 0 || x0 > sx_max || y0 > sy_max`) by exercising
    /// pixels that depend on the strict `>` / `<` comparison — if any
    /// boundary flips to `>=` / `<=`, an in-range pixel turns white or
    /// vice versa.
    #[test]
    fn rotate_cpu_oob_guard_fills_diagonal_with_white() {
        // 8×8 source, fully black (all zeros).  After a 5° rotation, the
        // four corner regions of the output map to OOB source coordinates
        // and get filled with white.
        let src = Bitmap::<Gray8>::new(8, 8, 1, false);
        let rotated = rotate_cpu(&src, 5.0);

        // Corner pixels of the rotated 8×8: their inverse-mapped source
        // coordinates fall outside [0, 6] × [0, 6] (sx_max = sy_max = 6).
        // At 5° with cx = cy = 3.5, the (0,0) output corner maps to roughly
        // sx = 3.5 - 3.5*cos5 - (-3.5)*sin5 ≈ 0.32, sy = -3.5*sin5 + (-3.5)*cos5 + 3.5 ≈ -0.18.
        // sy < 0 ⇒ OOB ⇒ white.  Source is all zeros, so any non-255 corner
        // would indicate the OOB guard is mis-firing.
        assert_eq!(
            rotated.row_bytes(0)[0],
            255,
            "(0,0) corner must be white-filled (inverse maps OOB at 5°)"
        );
        assert_eq!(
            rotated.row_bytes(0)[7],
            255,
            "(7,0) corner must be white-filled"
        );
        assert_eq!(
            rotated.row_bytes(7)[0],
            255,
            "(0,7) corner must be white-filled"
        );
        assert_eq!(
            rotated.row_bytes(7)[7],
            255,
            "(7,7) corner must be white-filled"
        );

        // Centre pixel: maps to (3.5, 3.5) which IS in-bounds; sample of
        // black source = 0.  If the bounds-check inverted to <= / >=, the
        // centre would also be white.
        assert_eq!(
            rotated.row_bytes(4)[4],
            0,
            "centre pixel must sample in-bounds black source, not OOB white"
        );
    }

    /// Rotating 360° should return approximately the original image.
    #[test]
    fn rotate_full_circle_near_identity() {
        let mut src = Bitmap::<Gray8>::new(64, 64, 1, false);
        for y in 0..64u32 {
            let row = src.row_bytes_mut(y);
            for (x, px) in row.iter_mut().enumerate() {
                // x ≤ 63 so x * 4 ≤ 252; well below u8::MAX, no truncation.
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "x ≤ 63 so x * 4 ≤ 252 < 256; fits u8 trivially"
                )]
                {
                    *px = (x * 4) as u8;
                }
            }
        }
        let rotated = rotate_cpu(&src, 360.0);
        // Bilinear + floating-point: allow ≤2 absolute difference at any pixel.
        let mut max_diff = 0i32;
        for y in 4..60u32 {
            let sr = &src.row_bytes(y)[4..60];
            let dr = &rotated.row_bytes(y)[4..60];
            for (&a, &b) in sr.iter().zip(dr.iter()) {
                max_diff = max_diff.max((i32::from(a) - i32::from(b)).abs());
            }
        }
        assert!(max_diff <= 2, "360° rotation max diff {max_diff} > 2");
    }

    /// GPU and CPU bilinear rotation agree to within 2 grey levels at any
    /// interior pixel for a typical deskew angle (2°).
    ///
    /// Gated on `gpu-validation` so it only runs when a CUDA device is
    /// available (same gate as the other GPU-vs-CPU parity tests).
    #[test]
    #[cfg(all(feature = "gpu-deskew", feature = "gpu-validation"))]
    fn gpu_vs_cpu_rotation_parity() {
        let mut src = Bitmap::<Gray8>::new(128, 128, 1, false);
        for y in 0..128u32 {
            let row = src.row_bytes_mut(y);
            for x in 0..128usize {
                row[x] = ((x * 2 + y as usize * 3) % 256) as u8;
            }
        }

        let angle = 2.0_f32;
        let cpu = rotate_cpu(&src, angle);
        let gpu = rotate_gpu(&src, angle).expect("GPU rotate failed");

        let mut max_diff = 0i32;
        // Skip a 4-pixel border where bilinear vs NPP edge treatment may differ.
        for y in 4..124u32 {
            let cr = &cpu.row_bytes(y)[4..124];
            let gr = &gpu.row_bytes(y)[4..124];
            for (&a, &b) in cr.iter().zip(gr.iter()) {
                max_diff = max_diff.max((a as i32 - b as i32).abs());
            }
        }
        assert!(
            max_diff <= 2,
            "GPU vs CPU rotation max diff {max_diff} > 2 at 2°"
        );
    }

    #[test]
    #[cfg(all(feature = "gpu-deskew", feature = "gpu-validation"))]
    fn gpu_rotate_zero_is_near_identity() {
        let mut src = Bitmap::<Gray8>::new(64, 64, 1, false);
        for y in 0..64u32 {
            let row = src.row_bytes_mut(y);
            for x in 0..64usize {
                row[x] = ((x * 4 + y as usize * 2) % 256) as u8;
            }
        }
        let gpu = rotate_gpu(&src, 0.0).expect("GPU rotate 0° failed");
        let mut max_diff = 0i32;
        for y in 4..60u32 {
            let sr = &src.row_bytes(y)[4..60];
            let gr = &gpu.row_bytes(y)[4..60];
            for (&a, &b) in sr.iter().zip(gr.iter()) {
                max_diff = max_diff.max((a as i32 - b as i32).abs());
            }
        }
        assert!(max_diff <= 2, "GPU 0° rotation max diff {max_diff} > 2");
    }

    /// Structural shape test for `rotate_gpu` — runs without a CUDA device.
    ///
    /// On a no-GPU machine, `rotate_gpu` returns `Err`; on a GPU machine it
    /// returns `Ok` with a bitmap matching the source dimensions.  This test
    /// accepts both outcomes but verifies the dimensions when the call
    /// succeeds.
    ///
    /// Pins the four whole-function-body replacement mutants
    /// (`Ok(Bitmap::new())`, `Ok(Bitmap::from_iter(...))`, etc.) that
    /// otherwise survive: each substitutes an empty / default Bitmap with
    /// dimensions ≠ source, which the size assertion catches even without
    /// a CUDA device. Pixel-content parity is covered by
    /// `gpu_vs_cpu_rotation_parity` under the `gpu-validation` gate.
    #[test]
    #[cfg(feature = "gpu-deskew")]
    fn rotate_gpu_preserves_dimensions_or_errors() {
        let mut src = Bitmap::<Gray8>::new(64, 64, 1, false);
        for y in 0..64u32 {
            let row = src.row_bytes_mut(y);
            for (x, px) in row.iter_mut().enumerate() {
                // x + y < 128, fits u8 trivially; bytemuck-equivalent cast.
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "x ≤ 63 and y ≤ 63 so x + y ≤ 126; well below u8::MAX"
                )]
                {
                    *px = (x + y as usize) as u8;
                }
            }
        }
        // No CUDA device → Err is the expected outcome; only the Ok arm needs
        // assertions (and is where cargo-mutants exercises rotate_gpu).
        if let Ok(out) = rotate_gpu(&src, 1.0) {
            assert_eq!(
                out.width, src.width,
                "rotate_gpu must return a bitmap with the source width"
            );
            assert_eq!(
                out.height, src.height,
                "rotate_gpu must return a bitmap with the source height"
            );
            assert_eq!(
                out.data().len(),
                (src.width as usize) * (src.height as usize),
                "rotate_gpu output buffer must have width*height bytes (tightly packed)"
            );
        }
    }

    /// Timing smoke-test: `rotate_cpu` on an 8.4 MP image (2900×2900) should
    /// complete in < 5 ms on modern hardware.  The test always passes — the time
    /// is printed for manual inspection.  Run with `--nocapture` to see it.
    ///
    /// A4 at 300 DPI ≈ 2480×3508 (8.7 MP); 2900×2900 is a round-number proxy.
    /// Baseline (Apr 2026, Ryzen 9 9900X3D, release build): ~0.6 ms.
    /// Target for Intel i7-8700K (Coffee Lake): < 5 ms.
    #[test]
    #[ignore = "manual perf smoke test; run with `--ignored --nocapture` for timing"]
    fn rotate_cpu_8mp_timing() {
        let w = 2900u32;
        let h = 2900u32;
        let mut src = Bitmap::<Gray8>::new(w, h, 1, false);
        for y in 0..h {
            let row = src.row_bytes_mut(y);
            for (x, px) in row.iter_mut().enumerate() {
                // % 256 keeps the value in [0, 255]; cast is exact.
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "modulo 256 keeps the value in [0, u8::MAX]; cast is exact"
                )]
                {
                    *px = ((x + y as usize * 3) % 256) as u8;
                }
            }
        }

        let t0 = std::time::Instant::now();
        let _ = rotate_cpu(&src, 1.5);
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;

        println!(
            "rotate_cpu 2900×2900 ({:.1} MP): {elapsed_ms:.2} ms",
            f64::from(w) * f64::from(h) / 1_000_000.0,
        );
        // Soft threshold: warn rather than fail so CI on slow machines doesn't flake.
        if elapsed_ms > 5.0 {
            eprintln!(
                "WARN: rotate_cpu exceeded 5 ms target ({elapsed_ms:.2} ms) — \
                 check SIMD auto-vectorisation on this machine"
            );
        }
    }
}
