//! AA compositing scalar implementation.
//!
//! `composite_aa_rgb8` performs per-pixel Porter-Duff source-over compositing
//! for RGB pixels with a separate alpha plane and a per-pixel shape (coverage)
//! byte.  This is the same logic as `pipe/aa.rs`; full SIMD vectorisation of
//! the division-by-`a_result` step is deferred to Phase 4.

use color::convert::div255;

/// Composite `src` colour over the existing RGB pixels in `dst` / `dst_alpha`.
///
/// For each pixel `i`:
/// - `a_src = div255(a_input * shape[i])`
/// - If `a_src == 255`: write `src` directly, set `dst_alpha[i] = 255`.
/// - If `a_src == 0` and `dst_alpha[i] == 0`: zero pixel and alpha.
/// - Otherwise: Porter-Duff over blend.
pub fn composite_aa_rgb8(
    dst: &mut [u8],
    dst_alpha: &mut [u8],
    src: [u8; 3],
    a_input: u8,
    shape: &[u8],
) {
    let count = shape.len();
    debug_assert_eq!(dst.len(), count * 3);
    debug_assert_eq!(dst_alpha.len(), count);

    let a_in = u32::from(a_input);
    let [sr, sg, sb] = src;

    for i in 0..count {
        let shape_v = u32::from(shape[i]);
        let a_src = u32::from(div255(a_in * shape_v));
        let a_dst = u32::from(dst_alpha[i]);

        let base = i * 3;

        if a_src == 255 {
            dst[base] = sr;
            dst[base + 1] = sg;
            dst[base + 2] = sb;
            dst_alpha[i] = 255;
        } else if a_src == 0 && a_dst == 0 {
            dst[base] = 0;
            dst[base + 1] = 0;
            dst[base + 2] = 0;
            dst_alpha[i] = 0;
        } else {
            let a_result = a_src + a_dst - u32::from(div255(a_src * a_dst));

            let weight_src = a_src;
            let weight_dst = a_result - a_src;

            for (j, &cs) in [sr, sg, sb].iter().enumerate() {
                let c_dst = u32::from(dst[base + j]);
                let c_src = u32::from(cs);
                let blended = (weight_dst * c_dst + weight_src * c_src) / a_result;
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "blended = weighted_avg / a_result ≤ 255"
                )]
                {
                    dst[base + j] = blended as u8;
                }
            }

            #[expect(
                clippy::cast_possible_truncation,
                reason = "a_result is Porter-Duff clamped; ≤ 255"
            )]
            {
                dst_alpha[i] = a_result as u8;
            }
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_coverage_writes_src() {
        let src = [200u8, 100, 50];
        let shape = [255u8, 255];
        let mut dst = vec![50u8; 6];
        let mut alpha = vec![128u8; 2];

        composite_aa_rgb8(&mut dst, &mut alpha, src, 255, &shape);

        assert_eq!(&dst[0..3], &[200, 100, 50]);
        assert_eq!(&dst[3..6], &[200, 100, 50]);
        assert_eq!(alpha[0], 255);
        assert_eq!(alpha[1], 255);
    }

    #[test]
    fn zero_coverage_over_transparent_zeroes() {
        let src = [255u8, 255, 255];
        let shape = [0u8];
        let mut dst = vec![0u8; 3];
        let mut alpha = vec![0u8; 1];

        composite_aa_rgb8(&mut dst, &mut alpha, src, 255, &shape);

        assert_eq!(dst[0], 0);
        assert_eq!(alpha[0], 0);
    }

    #[test]
    fn half_coverage_over_opaque_black() {
        let src = [255u8, 255, 255];
        let shape = [128u8];
        let mut dst = vec![0u8; 3];
        let mut alpha = vec![255u8; 1];

        composite_aa_rgb8(&mut dst, &mut alpha, src, 255, &shape);

        // a_src ≈ 128; a_dst = 255; a_result = 255; c ≈ 128.
        let v = dst[0];
        assert!(v >= 125 && v <= 131, "expected ~128, got {v}");
        assert_eq!(alpha[0], 255);
    }

    #[test]
    fn empty_shape_is_noop() {
        let mut dst: Vec<u8> = vec![];
        let mut alpha: Vec<u8> = vec![];
        composite_aa_rgb8(&mut dst, &mut alpha, [1, 2, 3], 255, &[]);
        // No panic, no modification.
    }
}
