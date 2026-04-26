//! Bit-population-count for `AaBuf` rows.
//!
//! `popcnt_aa_row(row)` counts the number of set bits in a packed byte slice —
//! the quantity used by the AA fill path to convert a supersampled 1-bit
//! coverage buffer into a per-pixel coverage byte.
//!
//! # Acceleration tiers (x86-64, most to least preferred)
//!
//! 1. **AVX-512 VPOPCNTDQ** (`avx512vpopcntdq` + `avx512bw`):
//!    process 64 bytes at a time with `_mm512_popcnt_epi8`.
//! 2. **`popcnt`**: process 8 bytes at a time with the hardware `popcnt` instruction.
//! 3. **Scalar**: `u8::count_ones` per byte.

#![allow(unsafe_code)]

/// Scalar fallback: count set bits in `row`.
#[inline]
pub(super) fn popcnt_aa_row_scalar(row: &[u8]) -> u32 {
    row.iter().map(|b| b.count_ones()).sum()
}

// ── x86-64 SIMD tiers ─────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
/// `popcnt` tier: 8 bytes per iteration using the hardware `popcnt` instruction.
///
/// # Safety
///
/// Caller must ensure the `popcnt` feature is available.
#[target_feature(enable = "popcnt")]
unsafe fn popcnt_aa_row_sse(row: &[u8]) -> u32 {
    use std::arch::x86_64::_popcnt64;

    let mut total = 0u32;
    let mut chunks = row.chunks_exact(8);
    for chunk in chunks.by_ref() {
        // SAFETY: chunk is exactly 8 bytes; read_unaligned handles alignment.
        // `popcnt` feature confirmed by the `#[target_feature]` attribute.
        unsafe {
            let word = chunk.as_ptr().cast::<u64>().read_unaligned();
            #[expect(
                clippy::cast_sign_loss,
                reason = "_popcnt64 returns a non-negative count; cast to u32 is safe"
            )]
            {
                total += _popcnt64(word.cast_signed()) as u32;
            }
        }
    }
    // Scalar remainder.
    for &b in chunks.remainder() {
        total += b.count_ones();
    }
    total
}

#[cfg(target_arch = "x86_64")]
/// AVX-512 VPOPCNTDQ tier: 64 bytes per iteration.
///
/// # Safety
///
/// Caller must ensure `avx512vpopcntdq` and `avx512bw` features are available.
#[target_feature(enable = "avx512vpopcntdq,avx512bw")]
unsafe fn popcnt_aa_row_avx512(row: &[u8]) -> u32 {
    use std::arch::x86_64::{_mm512_loadu_si512, _mm512_popcnt_epi8, _mm512_storeu_si512};

    let mut total = 0u32;
    let mut chunks = row.chunks_exact(64);
    for chunk in chunks.by_ref() {
        // SAFETY: chunk is exactly 64 bytes; unaligned load/store are always valid.
        // `avx512vpopcntdq` + `avx512bw` confirmed by the `#[target_feature]` attribute.
        unsafe {
            let v = _mm512_loadu_si512(chunk.as_ptr().cast());
            // Each lane gets the popcount of the corresponding byte (value in 0..=8).
            let pcnt = _mm512_popcnt_epi8(v);
            // Store the 64 byte-popcounts, then accumulate as u8 values.
            let mut buf = [0u8; 64];
            _mm512_storeu_si512(buf.as_mut_ptr().cast(), pcnt);
            for b in buf {
                total += u32::from(b);
            }
        }
    }
    // Scalar remainder (< 64 bytes).
    for &b in chunks.remainder() {
        total += b.count_ones();
    }
    total
}

// ── Public dispatch ───────────────────────────────────────────────────────────

/// Count the number of set bits in an `AaBuf` row.
///
/// Selects the best available SIMD tier at runtime.
#[must_use]
pub fn popcnt_aa_row(row: &[u8]) -> u32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512vpopcntdq") && is_x86_feature_detected!("avx512bw") {
            // SAFETY: we just confirmed both features are present.
            return unsafe { popcnt_aa_row_avx512(row) };
        }

        if is_x86_feature_detected!("popcnt") {
            // SAFETY: we just confirmed `popcnt` is present.
            return unsafe { popcnt_aa_row_sse(row) };
        }
    }

    popcnt_aa_row_scalar(row)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_all_zeros() {
        assert_eq!(popcnt_aa_row_scalar(&[0u8; 8]), 0);
    }

    #[test]
    fn scalar_all_ones() {
        assert_eq!(popcnt_aa_row_scalar(&[0xFFu8; 8]), 64);
    }

    #[test]
    fn scalar_known_pattern() {
        // 0b1010_1010 = 4 bits set
        let row = [0xAAu8; 4];
        assert_eq!(popcnt_aa_row_scalar(&row), 16);
    }

    #[test]
    fn scalar_empty() {
        assert_eq!(popcnt_aa_row_scalar(&[]), 0);
    }

    #[test]
    fn dispatch_matches_scalar() {
        let row: Vec<u8> = (0u8..=127).collect();
        let expected = popcnt_aa_row_scalar(&row);
        let got = popcnt_aa_row(&row);
        assert_eq!(got, expected, "dispatch mismatch");
    }

    #[test]
    fn dispatch_large_matches_scalar() {
        // 256 bytes — exercises 64-byte AVX-512 chunks and 8-byte popcnt chunks.
        let row: Vec<u8> = (0u8..=127).chain(0u8..=127).collect();
        assert_eq!(row.len(), 256);
        let expected = popcnt_aa_row_scalar(&row);
        let got = popcnt_aa_row(&row);
        assert_eq!(got, expected, "large dispatch mismatch");
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn sse_popcnt_matches_scalar() {
        if !is_x86_feature_detected!("popcnt") {
            return;
        }
        let row: Vec<u8> = (0u8..=127).collect();
        let expected = popcnt_aa_row_scalar(&row);
        // SAFETY: we just confirmed `popcnt` is present.
        let got = unsafe { popcnt_aa_row_sse(&row) };
        assert_eq!(got, expected, "SSE popcnt mismatch");
    }
}
