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
/// and produces a 17^4 × 3 = 250 563-byte table.  33 nodes give higher accuracy
/// at 33^4 × 3 ≈ 3.4 MB — too large for L2 residency on most GPUs.
/// Default is 17, matching the ICC/PDF standard minimum.

use moxcms::{ColorProfile, CmsError, Layout, TransformOptions};

pub(crate) const DEFAULT_GRID_N: u32 = 17;

/// Bake a CMYK ICC profile into a quadrilinear CLUT table.
///
/// `icc_bytes` — raw ICC profile data (the stream body of an `ICCBased` PDF
/// colour space object).
///
/// `grid_n` — nodes per axis; must be ≥ 2 and ≤ 255. Use `DEFAULT_GRID_N` (17)
/// for standard accuracy.
///
/// Returns a `Vec<u8>` of length `grid_n^4 * 3` with the RGB output for each
/// CMYK lattice point, stored in `(k*G³+c*G²+m*G+y)*3` order.
///
/// Returns `Err` if the ICC data is malformed or has the wrong colour space.
pub(crate) fn bake_cmyk_clut(icc_bytes: &[u8], grid_n: u32) -> Result<Vec<u8>, CmsError> {
    assert!(grid_n >= 2 && grid_n <= 255, "grid_n must be in [2, 255]");

    let src = ColorProfile::new_from_slice(icc_bytes)?;
    let dst = ColorProfile::new_srgb();

    // CMYK profiles require Layout::Rgba (4 bytes/pixel: C M Y K).
    let xform = src.create_transform_8bit(
        Layout::Rgba,
        &dst,
        Layout::Rgb,
        TransformOptions::default(),
    )?;

    let g = grid_n as usize;
    let g3 = g * g * g;
    let total = g3.checked_mul(g).and_then(|n| n.checked_mul(3)).expect("grid too large");

    let step = 255.0_f32 / (grid_n - 1) as f32;

    // Allocate input buffer for a full K-slice at a time to amortise transform
    // call overhead.  One K-slice has G^3 CMYK quads = 4*G^3 bytes.
    let slice_px = g3;
    let mut src_buf = vec![0u8; slice_px * 4];
    let mut dst_buf = vec![0u8; slice_px * 3];
    let mut table = vec![0u8; total];

    for ki in 0..g {
        let k = (ki as f32 * step).round() as u8;

        // Fill the K-slice: iterate c, m, y in inner loops.
        let mut off = 0;
        for ci in 0..g {
            let c = (ci as f32 * step).round() as u8;
            for mi in 0..g {
                let m = (mi as f32 * step).round() as u8;
                for yi in 0..g {
                    let y = (yi as f32 * step).round() as u8;
                    src_buf[off]     = c;
                    src_buf[off + 1] = m;
                    src_buf[off + 2] = y;
                    src_buf[off + 3] = k;
                    off += 4;
                }
            }
        }

        xform.transform(&src_buf[..slice_px * 4], &mut dst_buf[..slice_px * 3])?;

        // Write output into table in (k*G³+c*G²+m*G+y)*3 order.
        let mut src_off = 0;
        for ci in 0..g {
            for mi in 0..g {
                for yi in 0..g {
                    let idx = (ki * g3 + ci * g * g + mi * g + yi) * 3;
                    table[idx]     = dst_buf[src_off];
                    table[idx + 1] = dst_buf[src_off + 1];
                    table[idx + 2] = dst_buf[src_off + 2];
                    src_off += 3;
                }
            }
        }
    }

    Ok(table)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke test: a synthetic 2-node CLUT baked from a real Fogra39 ICC profile
    /// would require an actual ICC file; instead we verify that passing garbage
    /// bytes returns an error (not a panic).
    #[test]
    fn bake_cmyk_clut_rejects_garbage() {
        let result = bake_cmyk_clut(b"not an icc profile", DEFAULT_GRID_N);
        assert!(result.is_err(), "expected error for garbage ICC bytes");
    }

    #[test]
    fn bake_cmyk_clut_rejects_small_grid() {
        std::panic::catch_unwind(|| bake_cmyk_clut(b"", 1)).expect_err(
            "grid_n=1 should panic",
        );
    }
}
