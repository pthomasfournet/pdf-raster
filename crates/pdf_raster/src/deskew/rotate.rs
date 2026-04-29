//! Image rotation for deskew correction.
//!
//! Rotates an 8-bit grayscale bitmap by a given angle (in degrees, CW positive)
//! using bilinear interpolation.  Background pixels introduced by the rotation
//! are set to 255 (white — scanner background convention).
//!
//! # Implementation
//!
//! **CPU path** (always available): per-output-pixel inverse mapping with
//! bilinear interpolation.  Inner loop is written for auto-vectorisation:
//! simple scalar arithmetic that LLVM lowers to AVX-512 on `-C target-cpu=native`.
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
/// # Errors
///
/// Returns [`DeskewError`] if GPU allocation fails and CPU fallback is also
/// unavailable (extremely unlikely — the CPU path has no allocation that can fail
/// beyond `Vec` OOM, which panics).
pub fn rotate_inplace(img: &mut Bitmap<Gray8>, angle_deg: f32) -> Result<(), DeskewError> {
    #[cfg(feature = "gpu-deskew")]
    if let Ok(rotated) = rotate_gpu(img, angle_deg) {
        *img = rotated;
        return Ok(());
    }

    *img = rotate_cpu(img, angle_deg);
    Ok(())
}

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
pub(crate) fn rotate_cpu(src: &Bitmap<Gray8>, angle_deg: f32) -> Bitmap<Gray8> {
    let w = src.width as usize;
    let h = src.height as usize;

    let mut dst = Bitmap::<Gray8>::new(src.width, src.height, 1, false);

    let rad = angle_deg.to_radians();
    let cos_a = rad.cos();
    let sin_a = rad.sin();

    // Rotation centre: image centre (may be fractional).
    let cx = (w as f32 - 1.0) * 0.5;
    let cy = (h as f32 - 1.0) * 0.5;

    // Valid source coordinate range for bilinear sampling: x ∈ [0, w-2], y ∈ [0, h-2].
    let sx_max = (w as i32) - 2;
    let sy_max = (h as i32) - 2;

    for oy in 0..h {
        let dst_row = dst.row_bytes_mut(oy as u32);
        // Pre-compute the y-dependent terms outside the x loop.
        let dy = oy as f32 - cy;
        let sx_base = cos_a * (-cx) - sin_a * dy + cx;
        let sy_base = sin_a * (-cx) + cos_a * dy + cy;

        for ox in 0..w {
            let dx = ox as f32;
            let sx = sx_base + cos_a * dx;
            let sy = sy_base + sin_a * dx;

            let x0 = sx.floor() as i32;
            let y0 = sy.floor() as i32;

            if x0 < 0 || y0 < 0 || x0 > sx_max || y0 > sy_max {
                dst_row[ox] = 255;
                continue;
            }

            let x0u = x0 as usize;
            let y0u = y0 as usize;
            let fx = sx - x0 as f32;
            let fy = sy - y0 as f32;

            let r0 = src.row_bytes(y0u as u32);
            let r1 = src.row_bytes((y0u + 1) as u32);

            let p00 = r0[x0u] as f32;
            let p10 = r0[x0u + 1] as f32;
            let p01 = r1[x0u] as f32;
            let p11 = r1[x0u + 1] as f32;

            let top = p00 + fx * (p10 - p00);
            let bot = p01 + fx * (p11 - p01);
            let val = top + fy * (bot - top);

            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "bilinear of values in [0,255] stays in [0,255]"
            )]
            {
                dst_row[ox] = val.round() as u8;
            }
        }
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
            for x in 0..64usize {
                row[x] = if (x + y as usize) % 2 == 0 { 0 } else { 128 };
            }
        }
        let rotated = rotate_cpu(&src, 0.0);
        // Centre pixels should be identical; edges may differ (white fill vs original).
        let r_src = &src.row_bytes(32)[1..63];
        let r_dst = &rotated.row_bytes(32)[1..63];
        assert_eq!(r_src, r_dst, "centre row should be unchanged at 0°");
    }

    /// `rotate_inplace` at 0° does not panic.
    #[test]
    fn rotate_inplace_zero_no_panic() {
        let mut img = Bitmap::<Gray8>::new(32, 32, 1, false);
        rotate_inplace(&mut img, 0.0).unwrap();
    }

    /// Rotating 360° should return approximately the original image.
    #[test]
    fn rotate_full_circle_near_identity() {
        let mut src = Bitmap::<Gray8>::new(64, 64, 1, false);
        for y in 0..64u32 {
            let row = src.row_bytes_mut(y);
            for x in 0..64usize {
                row[x] = ((x * 4) % 256) as u8;
            }
        }
        let rotated = rotate_cpu(&src, 360.0);
        // Bilinear + floating-point: allow ≤2 absolute difference at any pixel.
        let mut max_diff = 0i32;
        for y in 4..60u32 {
            let sr = &src.row_bytes(y)[4..60];
            let dr = &rotated.row_bytes(y)[4..60];
            for (&a, &b) in sr.iter().zip(dr.iter()) {
                max_diff = max_diff.max((a as i32 - b as i32).abs());
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
}
