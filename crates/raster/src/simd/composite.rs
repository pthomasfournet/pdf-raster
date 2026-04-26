//! AA compositing: Porter-Duff source-over with per-pixel coverage.
//!
//! Two public functions:
//!
//! - [`composite_aa_rgb8`] — RGB pixels with a **separate alpha plane**.  Used by
//!   transparency-group collapse and the general AA path when `dst_alpha` is
//!   `Some`.  Handles the three-way per-pixel branch (full/zero/partial coverage).
//!
//! - [`composite_aa_rgb8_opaque`] — RGB pixels with **no alpha plane** (opaque
//!   destination).  The compositing formula simplifies to
//!   `c = div255((255-a_src)*c_dst + a_src*c_src)` per channel.  Expressed as
//!   `[u16; LANE]` lanes so that LLVM auto-vectorizes into AVX2/AVX-512 when the
//!   binary is compiled with `-C target-cpu=native`.
//!
//! # Why `[u16; LANE]` instead of explicit intrinsics
//!
//! tiny-skia's lowp pipeline uses the same technique (see `src/wide/u16x16_t.rs`
//! and its comment: "No need for explicit AVX2 SIMD; `-C target-cpu=native` will
//! autovectorize better than us").  A plain `[u16; 16]` array with straight
//! arithmetic on `u16` gives LLVM the freedom to choose the best instruction
//! width (128/256/512-bit) for the target, without us hard-coding a specific ISA.
//!
//! # div255 approximation
//!
//! `(v + 255) >> 8` approximates `v / 255` with at most ±1 LSB of error.
//! This matches the tiny-skia lowp `div255` and is cheaper to auto-vectorize
//! than the higher-precision `(v + (v>>8) + 0x80) >> 8` form.

// Number of pixels per SIMD-style lane chunk.  16 × u16 = 256 bits — one AVX2
// vector per colour channel.  LLVM will widen to 512-bit (AVX-512) automatically
// when the target supports it.
const LANE: usize = 16;

/// Approximate `v / 255` for `v` in `[0, 255²]`.  Maximum error: ±1 LSB.
#[inline]
const fn div255_u16(v: u16) -> u16 {
    (v + 255) >> 8
}

/// Composite `src` colour over the existing RGB pixels in `dst` / `dst_alpha`.
///
/// For each pixel `i`:
/// - `a_src = div255(a_input × shape[i])`
/// - If `a_src == 255`: write `src` directly, set `dst_alpha[i] = 255`.
/// - If `a_src == 0` and `dst_alpha[i] == 0`: zero pixel and alpha.
/// - Otherwise: Porter-Duff over blend.
///
/// # Panics (debug only)
///
/// Panics if `dst.len() != shape.len() * 3` or `dst_alpha.len() != shape.len()`.
pub fn composite_aa_rgb8(
    dst: &mut [u8],
    dst_alpha: &mut [u8],
    src: [u8; 3],
    a_input: u8,
    shape: &[u8],
) {
    let count = shape.len();
    debug_assert_eq!(dst.len(), count * 3, "composite_aa_rgb8: dst length mismatch");
    debug_assert_eq!(dst_alpha.len(), count, "composite_aa_rgb8: dst_alpha length mismatch");

    let a_in = u32::from(a_input);
    let [sr, sg, sb] = src;

    for i in 0..count {
        let shape_v = u32::from(shape[i]);
        // color::convert::div255 formula: (x + (x>>8) + 0x80) >> 8
        let prod = a_in * shape_v;
        let a_src = ((prod + (prod >> 8) + 0x80) >> 8).min(255);
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
            let prod_aa = a_src * a_dst;
            let a_result =
                (a_src + a_dst - ((prod_aa + (prod_aa >> 8) + 0x80) >> 8)).min(255);

            let weight_src = a_src;
            let weight_dst = a_result - a_src;

            for (j, &cs) in [sr, sg, sb].iter().enumerate() {
                let c_dst = u32::from(dst[base + j]);
                let c_src = u32::from(cs);
                // Numerator ≤ 255 × 255 = 65025 < u32::MAX; result ≤ 255.
                let blended = (weight_dst * c_dst + weight_src * c_src) / a_result;
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "blended = weighted average / a_result, both ≤ 255"
                )]
                {
                    dst[base + j] = blended as u8;
                }
            }

            dst_alpha[i] = a_result as u8;
        }
    }
}

/// Composite a solid RGB source over an opaque destination (no alpha plane).
///
/// The destination is assumed fully opaque (`a_dst = 255`).  The compositing
/// formula simplifies to:
/// ```text
/// c_result = div255((255 - a_src) * c_dst + a_src * c_src)
/// ```
///
/// Processes pixels in chunks of `LANE` (16) using `[u16; LANE]` lanes.
/// LLVM auto-vectorizes into 256/512-bit SIMD when compiled with
/// `-C target-cpu=native`.
///
/// # Arguments
///
/// - `dst`: packed RGB bytes, length must be `shape.len() * 3`.
/// - `src`: constant source colour applied to every pixel.
/// - `a_input`: source opacity (0 = transparent, 255 = opaque).
/// - `shape`: per-pixel AA coverage, one byte per pixel.
///
/// # Panics (debug only)
///
/// Panics if `dst.len() != shape.len() * 3`.
pub fn composite_aa_rgb8_opaque(dst: &mut [u8], src: [u8; 3], a_input: u8, shape: &[u8]) {
    let count = shape.len();
    debug_assert_eq!(
        dst.len(),
        count * 3,
        "composite_aa_rgb8_opaque: dst length mismatch (got {}, expected {})",
        dst.len(),
        count * 3,
    );

    let a_in = u16::from(a_input);
    let [sr, sg, sb] = [u16::from(src[0]), u16::from(src[1]), u16::from(src[2])];

    let full_chunks = count / LANE;
    let remainder = count % LANE;

    // ── Full LANE-wide chunks ──────────────────────────────────────────────────
    //
    // Both loops are structured so LLVM sees LANE independent iterations over
    // arrays of length LANE — the pattern that triggers 256/512-bit vectorization.
    for chunk in 0..full_chunks {
        let px_base = chunk * LANE;
        let byte_base = px_base * 3;

        // Step 1: compute a_src for each pixel in the chunk.
        let mut a_src_lane = [0u16; LANE];
        for (k, a) in a_src_lane.iter_mut().enumerate() {
            *a = div255_u16(a_in * u16::from(shape[px_base + k]));
        }

        // Step 2: composite each pixel using its a_src.
        for (k, &a_src) in a_src_lane.iter().enumerate() {
            let inv = 255 - a_src;
            let b = byte_base + k * 3;
            // Result of div255_u16 is ≤ 255, so truncation to u8 is safe.
            #[expect(clippy::cast_possible_truncation, reason = "div255_u16 result ≤ 255")]
            {
                dst[b] = div255_u16(inv * u16::from(dst[b]) + a_src * sr) as u8;
                dst[b + 1] = div255_u16(inv * u16::from(dst[b + 1]) + a_src * sg) as u8;
                dst[b + 2] = div255_u16(inv * u16::from(dst[b + 2]) + a_src * sb) as u8;
            }
        }
    }

    // ── Scalar tail ───────────────────────────────────────────────────────────
    let tail_px = full_chunks * LANE;
    let tail_byte = tail_px * 3;
    for k in 0..remainder {
        let a_src = div255_u16(a_in * u16::from(shape[tail_px + k]));
        let inv = 255 - a_src;
        let b = tail_byte + k * 3;
        #[expect(clippy::cast_possible_truncation, reason = "div255_u16 result ≤ 255")]
        {
            dst[b] = div255_u16(inv * u16::from(dst[b]) + a_src * sr) as u8;
            dst[b + 1] = div255_u16(inv * u16::from(dst[b + 1]) + a_src * sg) as u8;
            dst[b + 2] = div255_u16(inv * u16::from(dst[b + 2]) + a_src * sb) as u8;
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── composite_aa_rgb8 ────────────────────────────────────────────────────

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
    }

    // ── composite_aa_rgb8_opaque ─────────────────────────────────────────────

    #[test]
    fn opaque_full_coverage_writes_src() {
        let src = [200u8, 100, 50];
        let shape = [255u8; 4];
        let mut dst = vec![10u8; 12]; // 4 pixels

        composite_aa_rgb8_opaque(&mut dst, src, 255, &shape);

        for i in 0..4 {
            assert_eq!(&dst[i * 3..i * 3 + 3], &[200, 100, 50], "pixel {i}");
        }
    }

    #[test]
    fn opaque_zero_coverage_leaves_dst() {
        let src = [200u8, 100, 50];
        let shape = [0u8; 4];
        let original: Vec<u8> = (0..12).map(|i: u8| i * 10).collect();
        let mut dst = original.clone();

        composite_aa_rgb8_opaque(&mut dst, src, 255, &shape);

        assert_eq!(dst, original);
    }

    #[test]
    fn opaque_half_coverage_blends() {
        let src = [255u8, 255, 255];
        let shape = [128u8];
        let mut dst = vec![0u8; 3]; // black dst

        composite_aa_rgb8_opaque(&mut dst, src, 255, &shape);

        // div255_u16(128 * 255) ≈ 128; blend: div255_u16((255-128)*0 + 128*255) ≈ 128.
        let v = dst[0];
        assert!(v >= 125 && v <= 131, "expected ~128, got {v}");
    }

    #[test]
    fn opaque_matches_scalar_for_large_span() {
        // Verify LANE-chunked path matches a pixel-by-pixel reference.
        let src = [100u8, 150, 200];
        let a_input = 200u8;
        let count = 37usize; // crosses chunk boundary: 2 full chunks + 5 tail
        let shape: Vec<u8> = (0..count).map(|i| (i * 7 % 256) as u8).collect();
        let initial: Vec<u8> = (0..count * 3).map(|i| (i * 3 % 256) as u8).collect();

        // Scalar reference
        let mut ref_dst = initial.clone();
        let a_in = u16::from(a_input);
        let [sr, sg, sb] = [u16::from(src[0]), u16::from(src[1]), u16::from(src[2])];
        for i in 0..count {
            let a_src = div255_u16(a_in * u16::from(shape[i]));
            let inv = 255 - a_src;
            let b = i * 3;
            #[expect(clippy::cast_possible_truncation, reason = "div255_u16 result ≤ 255")]
            {
                ref_dst[b] = div255_u16(inv * u16::from(ref_dst[b]) + a_src * sr) as u8;
                ref_dst[b + 1] =
                    div255_u16(inv * u16::from(ref_dst[b + 1]) + a_src * sg) as u8;
                ref_dst[b + 2] =
                    div255_u16(inv * u16::from(ref_dst[b + 2]) + a_src * sb) as u8;
            }
        }

        let mut got = initial;
        composite_aa_rgb8_opaque(&mut got, src, a_input, &shape);

        assert_eq!(got, ref_dst, "chunked path mismatch vs scalar reference");
    }

    #[test]
    fn opaque_empty_is_noop() {
        let mut dst: Vec<u8> = vec![];
        composite_aa_rgb8_opaque(&mut dst, [1, 2, 3], 255, &[]);
    }
}
