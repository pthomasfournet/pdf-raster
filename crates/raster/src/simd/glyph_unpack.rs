//! Mono glyph bitmap unpacking — 1-bit-per-pixel MSB-first to one byte per pixel.
//!
//! `unpack_mono_row(packed, width, out)` expands `width` bits from `packed`
//! (MSB first, `ceil(width/8)` bytes) into `out` (`width` bytes), writing
//! `0xFF` for set bits and `0x00` for clear bits.
//!
//! # Acceleration
//!
//! - **SSE2** (x86-64): expand two source bytes → 16 output bytes per
//!   iteration using bit-isolating masks and `_mm_cmpeq_epi8`.
//! - **NEON** (aarch64): expand two source bytes → 16 output bytes per
//!   iteration using `vtstq_u8` (test-bits → 0xFF/0x00 in one instruction).
//! - **Scalar**: one byte at a time, 8 output bytes per iteration.

/// Scalar implementation: expand `width` MSB-first packed bits into `out`.
#[inline]
pub(super) fn unpack_mono_row_scalar(packed: &[u8], width: usize, out: &mut [u8]) {
    debug_assert!(out.len() >= width);
    debug_assert!(packed.len() >= width.div_ceil(8));
    for px in 0..width {
        let byte = packed[px / 8];
        let bit = 7 - (px % 8);
        out[px] = if (byte >> bit) & 1 != 0 { 0xFF } else { 0x00 };
    }
}

// ── SSE2 path ─────────────────────────────────────────────────────────────────
//
// All intrinsics used here (_mm_and_si128, _mm_cmpeq_epi8, _mm_set_epi8,
// _mm_set1_epi8, _mm_setzero_si128, _mm_storeu_si128, _mm_xor_si128) are
// SSE2, stable in std::arch since Rust 1.27.  No SSE4.1 instructions are
// actually emitted; the gate was overstated when _mm_blendv_epi8 was removed.

/// Expand two packed bytes into 16 output bytes via SSE2.
///
/// `b0` and `b1` are consecutive packed bytes (b0 first in the bit stream).
/// Writes 16 bytes into `out`.
///
/// # Safety
///
/// Caller must ensure SSE2 is available and `out` has room for ≥ 16 bytes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn expand_two_bytes_sse2(b0: u8, b1: u8, out: &mut [u8]) {
    use std::arch::x86_64::{
        __m128i, _mm_and_si128, _mm_cmpeq_epi8, _mm_set_epi8, _mm_set1_epi8, _mm_setzero_si128,
        _mm_storeu_si128, _mm_xor_si128,
    };
    debug_assert!(out.len() >= 16);

    // SAFETY (entire block): all SIMD intrinsics here are either pure register
    // operations (no memory access) or write to `out`, which has ≥ 16 bytes.
    // The `sse2` feature is guaranteed by the `#[target_feature]` attribute.
    unsafe {
        // Bit-isolation masks for the 8 bit positions, MSB first.
        // _mm_set_epi8 fills from lane 15 (arg 0) down to lane 0 (arg 15).
        // Lane 15 → bit 7 (0x80), lane 14 → bit 6 (0x40), …, lane 8 → bit 0 (0x01) for b0.
        // Lane 7  → bit 7 (0x80), …, lane 0 → bit 0 (0x01) for b1.
        #[expect(
            clippy::cast_possible_wrap,
            reason = "reinterpreting byte patterns as i8 for SIMD; bit patterns preserved"
        )]
        let mask: __m128i = _mm_set_epi8(
            0x01u8 as i8,
            0x02u8 as i8,
            0x04u8 as i8,
            0x08u8 as i8,
            0x10u8 as i8,
            0x20u8 as i8,
            0x40u8 as i8,
            0x80u8 as i8,
            0x01u8 as i8,
            0x02u8 as i8,
            0x04u8 as i8,
            0x08u8 as i8,
            0x10u8 as i8,
            0x20u8 as i8,
            0x40u8 as i8,
            0x80u8 as i8,
        );

        // Broadcast b0 into the lower 8 lanes, b1 into the upper 8 lanes.
        #[expect(
            clippy::cast_possible_wrap,
            reason = "reinterpreting byte as i8 for SIMD"
        )]
        let src: __m128i = _mm_set_epi8(
            b1 as i8, b1 as i8, b1 as i8, b1 as i8, b1 as i8, b1 as i8, b1 as i8, b1 as i8,
            b0 as i8, b0 as i8, b0 as i8, b0 as i8, b0 as i8, b0 as i8, b0 as i8, b0 as i8,
        );

        let zero = _mm_setzero_si128();
        let all_ones = _mm_set1_epi8(-1i8); // 0xFF in every lane

        // Isolate each bit.
        let isolated = _mm_and_si128(src, mask);

        // Where isolated == 0 the bit was clear; otherwise it was set.
        // cmpeq gives 0xFF where equal (bit clear), 0x00 where not equal (bit set).
        let eq_zero = _mm_cmpeq_epi8(isolated, zero);

        // XOR with 0xFF: flip so 0xFF = bit set, 0x00 = bit clear.
        let result = _mm_xor_si128(eq_zero, all_ones);

        // SAFETY: out.len() ≥ 16 (checked above); unaligned store is always valid.
        _mm_storeu_si128(out.as_mut_ptr().cast(), result);
    }
}

/// SSE2 row unpacker: processes pairs of packed bytes (16 pixels) at a time.
///
/// # Safety
///
/// Caller must ensure SSE2 is available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn unpack_mono_row_sse2(packed: &[u8], width: usize, out: &mut [u8]) {
    let mut px = 0usize;
    while px + 16 <= width {
        let b0 = packed[px / 8];
        let b1 = packed[px / 8 + 1];
        // SAFETY: px + 16 ≤ width ≤ out.len(); out[px..] has ≥ 16 bytes.
        unsafe { expand_two_bytes_sse2(b0, b1, &mut out[px..]) };
        px += 16;
    }
    // Scalar tail for the remaining pixels.
    for i in px..width {
        let byte = packed[i / 8];
        let bit = 7 - (i % 8);
        out[i] = if (byte >> bit) & 1 != 0 { 0xFF } else { 0x00 };
    }
}

// ── NEON path ─────────────────────────────────────────────────────────────────
//
// `vtstq_u8(a, b)` sets each lane to 0xFF if `(a & b) != 0`, else 0x00.
// This is exactly the 1-bit-test-to-mask semantics we need, in one instruction,
// compared to SSE2's three-instruction sequence (and → cmpeq → xor).
//
// For two source bytes → 16 output pixels:
//   - broadcast b0 into the low 8 lanes, b1 into the high 8 lanes (via vld1q_dup_u8)
//   - AND/test against the 8-position mask [0x80,0x40,...,0x01, 0x80,...,0x01]
//   - store via vst1q_u8.

/// Expand two packed bytes into 16 output bytes via NEON.
///
/// `b0` and `b1` are consecutive packed bytes (b0 first in the bit stream).
/// Writes 16 bytes into `out`.
///
/// # Safety
///
/// Caller must ensure NEON is available (mandatory on all ARMv8-A targets)
/// and `out` has room for ≥ 16 bytes.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn expand_two_bytes_neon(b0: u8, b1: u8, out: &mut [u8]) {
    use std::arch::aarch64::{uint8x16_t, vcombine_u8, vdup_n_u8, vld1q_u8, vst1q_u8, vtstq_u8};
    debug_assert!(out.len() >= 16);

    // Bit-position masks, MSB first: lane 0 → bit 7 (0x80) … lane 7 → bit 0 (0x01).
    // Repeated for b1 in lanes 8..15.
    let bit_mask: [u8; 16] = [
        0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01, 0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02,
        0x01,
    ];
    // SAFETY: bit_mask is 16 bytes, aligned to 1-byte boundary; vld1q_u8 requires
    // only 1-byte alignment.
    let mask: uint8x16_t = unsafe { vld1q_u8(bit_mask.as_ptr()) };

    // Broadcast b0 to low 8 lanes, b1 to high 8 lanes.
    let src: uint8x16_t = vcombine_u8(vdup_n_u8(b0), vdup_n_u8(b1));

    // vtstq_u8(src, mask): lane i = 0xFF if (src[i] & mask[i]) != 0, else 0x00.
    let result: uint8x16_t = vtstq_u8(src, mask);

    // SAFETY: out.len() ≥ 16 (checked above); pointer is valid; vst1q_u8 requires
    // only 1-byte alignment.
    unsafe { vst1q_u8(out.as_mut_ptr(), result) };
}

/// NEON row unpacker: processes pairs of packed bytes (16 pixels) at a time.
///
/// # Safety
///
/// Caller must ensure NEON is available (mandatory on all ARMv8-A targets).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn unpack_mono_row_neon(packed: &[u8], width: usize, out: &mut [u8]) {
    let mut px = 0usize;
    while px + 16 <= width {
        let b0 = packed[px / 8];
        let b1 = packed[px / 8 + 1];
        // SAFETY: px + 16 ≤ width ≤ out.len(); out[px..] has ≥ 16 bytes.
        unsafe { expand_two_bytes_neon(b0, b1, &mut out[px..]) };
        px += 16;
    }
    // Scalar tail for the remaining pixels.
    for i in px..width {
        let byte = packed[i / 8];
        let bit = 7 - (i % 8);
        out[i] = if (byte >> bit) & 1 != 0 { 0xFF } else { 0x00 };
    }
}

// ── Per-arch dispatch ─────────────────────────────────────────────────────────

/// x86-64: SSE2 is always available; runtime-detect anyway to follow the pattern.
#[cfg(target_arch = "x86_64")]
#[inline]
fn dispatch_unpack(packed: &[u8], width: usize, out: &mut [u8]) {
    if width >= 16 && is_x86_feature_detected!("sse2") {
        // SAFETY: SSE2 confirmed by is_x86_feature_detected! above.
        unsafe { unpack_mono_row_sse2(packed, width, out) };
    } else {
        unpack_mono_row_scalar(packed, width, out);
    }
}

/// aarch64: NEON is mandatory on all ARMv8-A targets.
#[cfg(target_arch = "aarch64")]
#[inline]
fn dispatch_unpack(packed: &[u8], width: usize, out: &mut [u8]) {
    if width >= 16 {
        // SAFETY: NEON is mandatory on aarch64.
        unsafe { unpack_mono_row_neon(packed, width, out) };
    } else {
        unpack_mono_row_scalar(packed, width, out);
    }
}

/// Generic fallback for targets without SIMD specialisation.
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
#[inline]
fn dispatch_unpack(packed: &[u8], width: usize, out: &mut [u8]) {
    unpack_mono_row_scalar(packed, width, out);
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Expand `width` MSB-first packed bits from `packed` into `out`.
///
/// Each output byte is `0xFF` if the corresponding bit was set, `0x00` otherwise.
/// Uses the best SIMD available for the target arch (SSE2 on x86-64, NEON on
/// aarch64), with a scalar fallback for other targets or widths < 16.
pub fn unpack_mono_row(packed: &[u8], width: usize, out: &mut [u8]) {
    debug_assert!(out.len() >= width);
    debug_assert!(packed.len() >= width.div_ceil(8));
    dispatch_unpack(packed, width, out);
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_packed(bits: &[u8]) -> Vec<u8> {
        // `bits` is an array of 0/1 values; pack MSB-first.
        let nbytes = bits.len().div_ceil(8);
        let mut packed = vec![0u8; nbytes];
        for (i, &b) in bits.iter().enumerate() {
            if b != 0 {
                packed[i / 8] |= 0x80 >> (i % 8);
            }
        }
        packed
    }

    #[test]
    fn scalar_all_zeros() {
        let bits = [0u8; 8];
        let packed = make_packed(&bits);
        let mut out = [1u8; 8];
        unpack_mono_row_scalar(&packed, 8, &mut out);
        assert!(out.iter().all(|&b| b == 0), "expected all zeros");
    }

    #[test]
    fn scalar_all_ones() {
        let bits = [1u8; 8];
        let packed = make_packed(&bits);
        let mut out = [0u8; 8];
        unpack_mono_row_scalar(&packed, 8, &mut out);
        assert!(out.iter().all(|&b| b == 0xFF), "expected all 0xFF");
    }

    #[test]
    fn scalar_alternating() {
        // 0b1010_1010 = 0xAA
        let packed = [0xAAu8];
        let mut out = [0u8; 8];
        unpack_mono_row_scalar(&packed, 8, &mut out);
        let expected = [0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00];
        assert_eq!(out, expected);
    }

    #[test]
    fn scalar_partial_byte() {
        // Only 5 bits: 0b1101_0... → pack: 0b1101_0000 = 0xD0
        let packed = [0xD0u8];
        let mut out = [0u8; 5];
        unpack_mono_row_scalar(&packed, 5, &mut out);
        assert_eq!(out, [0xFF, 0xFF, 0x00, 0xFF, 0x00]);
    }

    #[test]
    fn dispatch_matches_scalar_16_pixels() {
        let bits: Vec<u8> = (0..16).map(|i| (i % 3 == 0) as u8).collect();
        let packed = make_packed(&bits);
        let mut expected = vec![0u8; 16];
        unpack_mono_row_scalar(&packed, 16, &mut expected);
        let mut got = vec![0u8; 16];
        unpack_mono_row(&packed, 16, &mut got);
        assert_eq!(got, expected, "dispatch 16-pixel mismatch");
    }

    #[test]
    fn dispatch_matches_scalar_large() {
        let bits: Vec<u8> = (0u8..128).map(|i| if i % 5 == 0 { 1 } else { 0 }).collect();
        let packed = make_packed(&bits);
        let mut expected = vec![0u8; 128];
        unpack_mono_row_scalar(&packed, 128, &mut expected);
        let mut got = vec![0u8; 128];
        unpack_mono_row(&packed, 128, &mut got);
        assert_eq!(got, expected, "dispatch large mismatch");
    }

    /// Non-multiple-of-16 width: exercises the scalar tail path on all arches.
    #[test]
    fn dispatch_matches_scalar_non_multiple() {
        for width in [1usize, 7, 9, 15, 17, 23, 33, 63, 65] {
            let bits: Vec<u8> = (0..width)
                .map(|i| (i.wrapping_mul(7) % 3 == 0) as u8)
                .collect();
            let packed = make_packed(&bits);
            let mut expected = vec![0u8; width];
            unpack_mono_row_scalar(&packed, width, &mut expected);
            let mut got = vec![0u8; width];
            unpack_mono_row(&packed, width, &mut got);
            assert_eq!(got, expected, "dispatch mismatch at width={width}");
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn sse2_expand_two_bytes_known() {
        if !is_x86_feature_detected!("sse2") {
            return;
        }
        // b0 = 0xAA = 0b1010_1010 → pixels 0..8: FF,00,FF,00,FF,00,FF,00
        // b1 = 0x55 = 0b0101_0101 → pixels 8..16: 00,FF,00,FF,00,FF,00,FF
        let mut out = [0u8; 16];
        // SAFETY: we just confirmed SSE2 is available and out has 16 bytes.
        unsafe { expand_two_bytes_sse2(0xAA, 0x55, &mut out) };
        let expected = [
            0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF,
            0x00, 0xFF,
        ];
        assert_eq!(out, expected, "SSE2 two-byte expand mismatch");
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn sse2_matches_scalar_random() {
        if !is_x86_feature_detected!("sse2") {
            return;
        }
        let packed = [0b1001_1010u8, 0b0110_0101u8, 0b1111_0000u8, 0b0000_1111u8];
        let width = 32usize;
        let mut expected = vec![0u8; width];
        unpack_mono_row_scalar(&packed, width, &mut expected);
        let mut got = vec![0u8; width];
        // SAFETY: SSE2 is available.
        unsafe { unpack_mono_row_sse2(&packed, width, &mut got) };
        assert_eq!(got, expected, "SSE2 row unpack mismatch");
    }

    /// NEON: expand two known bytes and verify against scalar.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_expand_two_bytes_known() {
        // b0 = 0xAA = 0b1010_1010 → pixels 0..8: FF,00,FF,00,FF,00,FF,00
        // b1 = 0x55 = 0b0101_0101 → pixels 8..16: 00,FF,00,FF,00,FF,00,FF
        let mut out = [0u8; 16];
        // SAFETY: NEON is mandatory on aarch64; out has 16 bytes.
        unsafe { expand_two_bytes_neon(0xAA, 0x55, &mut out) };
        let expected = [
            0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF,
            0x00, 0xFF,
        ];
        assert_eq!(out, expected, "NEON two-byte expand mismatch");
    }

    /// NEON: full row matches scalar for a variety of packed byte patterns.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_matches_scalar_random() {
        let packed = [0b1001_1010u8, 0b0110_0101u8, 0b1111_0000u8, 0b0000_1111u8];
        let width = 32usize;
        let mut expected = vec![0u8; width];
        unpack_mono_row_scalar(&packed, width, &mut expected);
        let mut got = vec![0u8; width];
        // SAFETY: NEON is mandatory on aarch64.
        unsafe { unpack_mono_row_neon(&packed, width, &mut got) };
        assert_eq!(got, expected, "NEON row unpack mismatch");
    }

    /// NEON: all-zeros and all-ones boundary cases.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_boundary_bytes() {
        for (b0, b1, desc) in [(0x00u8, 0x00u8, "all-zero"), (0xFFu8, 0xFFu8, "all-one")] {
            let mut neon_out = [0u8; 16];
            let mut scalar_out = [0u8; 16];
            let packed = [b0, b1];
            // SAFETY: NEON mandatory on aarch64.
            unsafe { expand_two_bytes_neon(b0, b1, &mut neon_out) };
            unpack_mono_row_scalar(&packed, 16, &mut scalar_out);
            assert_eq!(neon_out, scalar_out, "NEON boundary mismatch ({desc})");
        }
    }
}
