//! SIMD-accelerated solid-colour fill paths.
//!
//! `blend_solid_rgb8` — fill RGB pixels with a constant colour.
//! `blend_solid_gray8` — fill grayscale pixels with a constant value.
//!
//! On x86-64 with AVX2 both functions use 256-bit stores; on all other
//! targets (or when AVX2 is absent at runtime) they fall back to scalar code.

#![allow(unsafe_code)]

/// Fill `count` RGB pixels in `dst` with `color` using a scalar loop.
#[inline]
pub(super) fn blend_solid_rgb8_scalar(dst: &mut [u8], color: [u8; 3], count: usize) {
    for chunk in dst[..count * 3].chunks_exact_mut(3) {
        chunk.copy_from_slice(&color);
    }
}

/// Fill `count` grayscale pixels in `dst` with `color`.
#[inline]
pub(super) fn blend_solid_gray8_scalar(dst: &mut [u8], color: u8, count: usize) {
    dst[..count].fill(color);
}

// ── AVX2 paths ────────────────────────────────────────────────────────────────

#[cfg(all(target_arch = "x86_64", feature = "simd-avx2"))]
/// Fill `count` RGB pixels in `dst` with `color` using 96-byte AVX2 chunks.
///
/// # Safety
///
/// Must only be called when AVX2 is available (`is_x86_feature_detected!("avx2")`).
#[target_feature(enable = "avx2")]
unsafe fn blend_solid_rgb8_avx2(dst: &mut [u8], color: [u8; 3], count: usize) {
    use std::arch::x86_64::{__m256i, _mm256_loadu_si256, _mm256_storeu_si256};

    let [r, g, b] = color;
    // Build a 96-byte tile (32 pixels × 3 bytes = LCM(3,32)) so that three
    // 32-byte stores cover exactly 32 pixels with no partial pixel at any
    // store boundary.
    let mut tile = [0u8; 96];
    for (i, t) in tile.iter_mut().enumerate() {
        *t = match i % 3 {
            0 => r,
            1 => g,
            _ => b,
        };
    }

    let dst_ptr = dst.as_mut_ptr();
    let tile_ptr = tile.as_ptr();

    // Load the three 32-byte vectors once.
    // SAFETY (entire block): tile is 96 bytes on the stack; pointers are in-bounds.
    // dst_ptr..dst_ptr + count*3 ≤ dst.len() by construction.
    let (v0, v1, v2): (__m256i, __m256i, __m256i) = unsafe {
        (
            _mm256_loadu_si256(tile_ptr.cast()),
            _mm256_loadu_si256(tile_ptr.add(32).cast()),
            _mm256_loadu_si256(tile_ptr.add(64).cast()),
        )
    };

    // Number of complete 96-byte (32-pixel) chunks.
    let chunks = count / 32;
    for i in 0..chunks {
        // SAFETY: i * 96 + 96 ≤ chunks * 96 ≤ count * 3 ≤ dst.len().
        unsafe {
            let p = dst_ptr.add(i * 96);
            _mm256_storeu_si256(p.cast(), v0);
            _mm256_storeu_si256(p.add(32).cast(), v1);
            _mm256_storeu_si256(p.add(64).cast(), v2);
        }
    }

    // Scalar tail for the remaining pixels.
    let done = chunks * 32;
    for px in done..count {
        let base = px * 3;
        dst[base] = r;
        dst[base + 1] = g;
        dst[base + 2] = b;
    }
}

#[cfg(all(target_arch = "x86_64", feature = "simd-avx2"))]
/// Fill `count` grayscale pixels in `dst` with `color` using 32-byte AVX2 stores.
///
/// # Safety
///
/// Must only be called when AVX2 is available (`is_x86_feature_detected!("avx2")`).
#[target_feature(enable = "avx2")]
unsafe fn blend_solid_gray8_avx2(dst: &mut [u8], color: u8, count: usize) {
    use std::arch::x86_64::{_mm256_set1_epi8, _mm256_storeu_si256};

    #[expect(
        clippy::cast_possible_wrap,
        reason = "reinterpreting byte as i8 for SIMD; bit pattern preserved"
    )]
    let vec = _mm256_set1_epi8(color as i8);
    let dst_ptr = dst.as_mut_ptr();

    let chunks = count / 32;
    for i in 0..chunks {
        // SAFETY: i * 32 + 32 ≤ chunks * 32 ≤ count ≤ dst.len().
        unsafe { _mm256_storeu_si256(dst_ptr.add(i * 32).cast(), vec) };
    }

    // Scalar tail.
    let done = chunks * 32;
    dst[done..count].fill(color);
}

// ── Public dispatch functions ─────────────────────────────────────────────────

/// Fill `count` RGB pixels in `dst` (starting at byte 0) with `color`.
///
/// Uses AVX2 when available at runtime, otherwise falls back to scalar.
pub fn blend_solid_rgb8(dst: &mut [u8], color: [u8; 3], count: usize) {
    #[cfg(target_arch = "x86_64")]
    #[cfg(feature = "simd-avx2")]
    if is_x86_feature_detected!("avx2") && count >= 32 {
        // SAFETY: we just confirmed AVX2 is available.
        unsafe { blend_solid_rgb8_avx2(dst, color, count) };
        return;
    }

    blend_solid_rgb8_scalar(dst, color, count);
}

/// Fill `count` grayscale pixels in `dst` (starting at byte 0) with `color`.
///
/// Uses AVX2 when available at runtime, otherwise falls back to scalar.
pub fn blend_solid_gray8(dst: &mut [u8], color: u8, count: usize) {
    #[cfg(target_arch = "x86_64")]
    #[cfg(feature = "simd-avx2")]
    if is_x86_feature_detected!("avx2") && count >= 32 {
        // SAFETY: we just confirmed AVX2 is available.
        unsafe { blend_solid_gray8_avx2(dst, color, count) };
        return;
    }

    blend_solid_gray8_scalar(dst, color, count);
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── scalar ────────────────────────────────────────────────────────────────

    #[test]
    fn scalar_rgb8_small() {
        let color = [10u8, 20, 30];
        let mut dst = vec![0u8; 9];
        blend_solid_rgb8_scalar(&mut dst, color, 3);
        assert_eq!(dst, [10, 20, 30, 10, 20, 30, 10, 20, 30]);
    }

    #[test]
    fn scalar_rgb8_zero_count() {
        let color = [1u8, 2, 3];
        let mut dst = vec![0u8; 3];
        blend_solid_rgb8_scalar(&mut dst, color, 0);
        assert_eq!(dst, [0, 0, 0]);
    }

    #[test]
    fn scalar_gray8() {
        let mut dst = vec![0u8; 5];
        blend_solid_gray8_scalar(&mut dst, 42, 5);
        assert!(dst.iter().all(|&b| b == 42));
    }

    // ── dispatch (tests both paths if AVX2 present) ───────────────────────────

    #[test]
    fn dispatch_rgb8_matches_scalar() {
        let color = [100u8, 150, 200];
        // Use count > 32 so AVX2 path is triggered.
        let count = 64usize;
        let mut expected = vec![0u8; count * 3];
        blend_solid_rgb8_scalar(&mut expected, color, count);

        let mut got = vec![0u8; count * 3];
        blend_solid_rgb8(&mut got, color, count);
        assert_eq!(got, expected, "dispatch_rgb8 mismatch");
    }

    #[test]
    fn dispatch_gray8_matches_scalar() {
        let count = 128usize;
        let mut expected = vec![0u8; count];
        blend_solid_gray8_scalar(&mut expected, 77, count);

        let mut got = vec![0u8; count];
        blend_solid_gray8(&mut got, 77, count);
        assert_eq!(got, expected, "dispatch_gray8 mismatch");
    }

    #[test]
    fn dispatch_rgb8_tail_handled() {
        // count = 35: 32-pixel AVX2 chunk + 3-pixel scalar tail.
        let color = [7u8, 8, 9];
        let count = 35usize;
        let mut expected = vec![0u8; count * 3];
        blend_solid_rgb8_scalar(&mut expected, color, count);

        let mut got = vec![0u8; count * 3];
        blend_solid_rgb8(&mut got, color, count);
        assert_eq!(got, expected, "tail mismatch");
    }

    #[test]
    fn dispatch_rgb8_exact_32_pixels() {
        let color = [255u8, 0, 128];
        let count = 32usize;
        let mut expected = vec![0u8; count * 3];
        blend_solid_rgb8_scalar(&mut expected, color, count);

        let mut got = vec![0u8; count * 3];
        blend_solid_rgb8(&mut got, color, count);
        assert_eq!(got, expected, "exact 32-pixel mismatch");
    }

    /// Verify AVX2 path directly (skipped when AVX2 is unavailable).
    #[cfg(all(target_arch = "x86_64", feature = "simd-avx2"))]
    #[test]
    fn avx2_rgb8_matches_scalar_direct() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let color = [11u8, 22, 33];
        let count = 96usize;
        let mut expected = vec![0u8; count * 3];
        blend_solid_rgb8_scalar(&mut expected, color, count);

        let mut got = vec![0u8; count * 3];
        // SAFETY: we just checked AVX2 is available.
        unsafe { blend_solid_rgb8_avx2(&mut got, color, count) };
        assert_eq!(got, expected, "AVX2 RGB path mismatch");
    }

    #[cfg(all(target_arch = "x86_64", feature = "simd-avx2"))]
    #[test]
    fn avx2_gray8_matches_scalar_direct() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let count = 128usize;
        let mut expected = vec![0u8; count];
        blend_solid_gray8_scalar(&mut expected, 200, count);

        let mut got = vec![0u8; count];
        // SAFETY: we just checked AVX2 is available.
        unsafe { blend_solid_gray8_avx2(&mut got, 200, count) };
        assert_eq!(got, expected, "AVX2 gray path mismatch");
    }
}
