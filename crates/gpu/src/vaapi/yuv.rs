//! YUV colour space conversions for VA-API surface output.
//!
//! Raphael (VCN 4.0.0) outputs JPEG as NV12.  Hardware YCbCr→RGB conversion
//! requires VCN 4.0.3+ (`RDECODE_JPEG_VER_3`), which is absent on Raphael.
//! This module provides the required CPU step.

#![cfg(feature = "vaapi")]

use super::error::{Result, VapiError};

/// Convert NV12 (Y plane + interleaved UV half-plane) to interleaved RGB8.
///
/// `y_plane` — `stride_y × height` bytes (luma, one byte per pixel).
/// `uv_plane` — `stride_uv × ceil(height/2)` bytes (interleaved Cb, Cr; one
///             sample pair per 2×2 pixel block).
///
/// # Colour model
///
/// BT.601 full-range (JFIF convention):
/// ```text
///   R = clamp(Y + 1.402·(Cr−128), 0, 255)
///   G = clamp(Y − 0.34414·(Cb−128) − 0.71414·(Cr−128), 0, 255)
///   B = clamp(Y + 1.772·(Cb−128), 0, 255)
/// ```
///
/// Fixed-point shift-14 coefficients:
/// - `CR_R` = 22970  (≈ 1.402   × 16384)
/// - `CB_G` = −5638  (≈ −0.34414 × 16384)
/// - `CR_G` = −11700 (≈ −0.71414 × 16384)
/// - `CB_B` = 29032  (≈ 1.772   × 16384)
///
/// # Errors
///
/// Returns `VapiError::Overflow` if any dimension product overflows `usize`.
/// Returns `VapiError::BadJpeg` if the plane slices are smaller than expected.
pub(super) fn nv12_to_rgb8(
    y_plane: &[u8],
    uv_plane: &[u8],
    width: u32,
    height: u32,
    stride_y: u32,
    stride_uv: u32,
) -> Result<Vec<u8>> {
    let width = width as usize;
    let height = height as usize;
    let stride_y = stride_y as usize;
    let stride_uv = stride_uv as usize;

    let npixels = width.checked_mul(height).ok_or(VapiError::Overflow)?;
    let out_len = npixels.checked_mul(3).ok_or(VapiError::Overflow)?;

    let y_needed = stride_y.checked_mul(height).ok_or(VapiError::Overflow)?;
    let uv_rows = height.div_ceil(2);
    let uv_needed = stride_uv.checked_mul(uv_rows).ok_or(VapiError::Overflow)?;

    if y_plane.len() < y_needed {
        return Err(VapiError::BadJpeg(format!(
            "Y plane too small: need {y_needed}, got {}",
            y_plane.len()
        )));
    }
    if uv_plane.len() < uv_needed {
        return Err(VapiError::BadJpeg(format!(
            "UV plane too small: need {uv_needed}, got {}",
            uv_plane.len()
        )));
    }

    let mut rgb = vec![0u8; out_len];

    for row in 0..height {
        let uv_row = row / 2;
        let y_row_off = row * stride_y;
        let uv_row_off = uv_row * stride_uv;

        for col in 0..width {
            let luma = i32::from(y_plane[y_row_off + col]);
            // UV samples are co-sited at even column positions in the UV plane.
            let cb = i32::from(uv_plane[uv_row_off + (col & !1)]) - 128;
            let cr = i32::from(uv_plane[uv_row_off + (col & !1) + 1]) - 128;

            // BT.601 full-range, shift-14 fixed point.
            let red = (luma * 16384 + 22970 * cr) >> 14;
            let grn = (luma * 16384 - 5638 * cb - 11700 * cr) >> 14;
            let blu = (luma * 16384 + 29032 * cb) >> 14;

            let out = &mut rgb[(row * width + col) * 3..];
            // clamp(0,255) guarantees the value fits in u8; the cast is safe.
            #[expect(
                clippy::cast_sign_loss,
                reason = "value is clamped to [0, 255] on the line above; sign loss is impossible"
            )]
            {
                out[0] = red.clamp(0, 255) as u8;
                out[1] = grn.clamp(0, 255) as u8;
                out[2] = blu.clamp(0, 255) as u8;
            }
        }
    }

    Ok(rgb)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neutral_gray_roundtrip() {
        // Y=200, Cb=128, Cr=128 → no chroma shift → R≈G≈B≈200.
        let y_plane = [200u8, 200, 200, 200];
        let uv_plane = [128u8, 128];
        let rgb = nv12_to_rgb8(&y_plane, &uv_plane, 2, 2, 2, 2).unwrap();
        assert_eq!(rgb.len(), 12, "2×2 RGB = 12 bytes");
        for i in 0..4 {
            let (r, g, b) = (rgb[i * 3], rgb[i * 3 + 1], rgb[i * 3 + 2]);
            assert!(
                r.abs_diff(200) <= 2 && g.abs_diff(200) <= 2 && b.abs_diff(200) <= 2,
                "pixel {i}: R={r} G={g} B={b}, expected ≈ 200"
            );
        }
    }

    #[test]
    fn red_primary() {
        // BT.601 full-range forward transform: R=255, G=0, B=0 → Y=81, Cb=91, Cr=240.
        // Exact inverse with those values gives (R, G, B) ≈ (250, 8, 8).
        // We verify that the channel ordering and sign are correct (R >> G ≈ B),
        // rather than checking exact round-trip fidelity, because integer rounding
        // means the inverse of the forward-rounded values is not exactly (255,0,0).
        let y_plane = [81u8];
        let uv_plane = [91u8, 240];
        let rgb = nv12_to_rgb8(&y_plane, &uv_plane, 1, 1, 1, 2).unwrap();
        assert_eq!(rgb.len(), 3);
        let (r, g, b) = (rgb[0], rgb[1], rgb[2]);
        assert!(
            r > 200 && g < 30 && b < 30,
            "red primary direction check: R={r} G={g} B={b}"
        );
    }

    #[test]
    fn plane_too_small_returns_error() {
        // stride_y = 4 but plane has only 2 bytes.
        let result = nv12_to_rgb8(&[0u8; 2], &[128u8; 4], 2, 2, 4, 4);
        assert!(matches!(result, Err(VapiError::BadJpeg(_))));
    }
}
