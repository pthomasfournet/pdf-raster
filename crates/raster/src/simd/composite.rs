//! AA compositing: Porter-Duff source-over with per-pixel coverage.
//!
//! Single public entry: [`composite_aa_rgb8_opaque`] — RGB pixels with no
//! alpha plane (opaque destination).  The compositing formula simplifies to
//! `c = div255((255-a_src)*c_dst + a_src*c_src)` per channel.  Expressed as
//! `[u16; LANE]` lanes so that LLVM auto-vectorizes into AVX2/AVX-512 when
//! the binary is compiled with `-C target-cpu=native`.
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
                ref_dst[b + 1] = div255_u16(inv * u16::from(ref_dst[b + 1]) + a_src * sg) as u8;
                ref_dst[b + 2] = div255_u16(inv * u16::from(ref_dst[b + 2]) + a_src * sb) as u8;
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
