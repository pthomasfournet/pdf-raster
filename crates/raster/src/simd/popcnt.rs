//! Bit-population-count for `AaBuf` rows.
//!
//! Two public entry points:
//!
//! - `popcnt_aa_row(row)` — counts total set bits in a packed byte slice.
//!   Used by clip-path AA paths that need row-level totals.
//!
//! - `aa_coverage_span(rows, x0, shape)` — fills a `shape` buffer with
//!   per-pixel AA coverage counts for output pixels `x0 .. x0+shape.len()`.
//!   This is the hot path called from `draw_aa_line` in `fill/mod.rs`.
//!   Each output pixel maps to 4 bits (one nibble) in each of the 4 `AaBuf`
//!   rows; `aa_coverage_span` sums those nibbles across rows for every pixel
//!   in the span in one vectorised pass.
//!
//! # `AaBuf` nibble layout
//!
//! For `AA_SIZE = 4`, output pixel `x` occupies the nibble at byte `x/2` of
//! each row: the **high** nibble if `x` is even, the **low** nibble if `x` is
//! odd.  Each nibble holds 0–4 set bits (one per AA sub-sample).  Summing the
//! four rows gives a coverage count in 0..=16.
//!
//! # Acceleration tiers for `popcnt_aa_row` (x86-64, most to least preferred)
//!
//! 1. **AVX-512 VPOPCNTDQ** (`avx512vpopcntdq` + `avx512bw`):
//!    process 64 bytes at a time with `_mm512_popcnt_epi8`.
//! 2. **`popcnt`**: process 8 bytes at a time with the hardware `popcnt` instruction.
//! 3. **Scalar**: `u8::count_ones` per byte.
//!
//! # Acceleration tiers for `aa_coverage_span` (x86-64, most to least preferred)
//!
//! 1. **AVX-512 BITALG** (`avx512bitalg` + `avx512bw`):
//!    `_mm512_popcnt_epi8` on each row after nibble masking (high and low
//!    nibbles separately), then horizontal sum across the 4 rows.
//!    Processes 128 output pixels per 64-byte AVX-512 iteration.
//! 2. **Scalar**: byte-by-byte nibble popcount via the `NIBBLE_POP` table.

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

// ── aa_coverage_span scalar fallback ─────────────────────────────────────────

/// Nibble popcount table: `NIBBLE_POP[n]` = number of set bits in `n` (0..=4).
const NIBBLE_POP: [u8; 16] = [0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4];

/// Scalar fallback for `aa_coverage_span`.
///
/// `rows[r]` is the byte slice for AA sub-row `r` (length = `row_bytes`).
/// `x0` is the index of the first output pixel in `shape`.
/// `shape[i]` is set to the coverage count (0..=16) for output pixel `x0 + i`.
pub(super) fn aa_coverage_span_scalar(rows: [&[u8]; 4], x0: usize, shape: &mut [u8]) {
    for (i, out) in shape.iter_mut().enumerate() {
        let x = x0 + i;
        let byte_idx = x >> 1;
        let is_odd = (x & 1) != 0;
        let mut count = 0u8;
        for row in rows {
            debug_assert!(
                byte_idx < row.len(),
                "aa_coverage_span_scalar: byte_idx={byte_idx} out of bounds (row.len={})",
                row.len()
            );
            // SAFETY: byte_idx < row.len() asserted above.
            let byte = row[byte_idx];
            let nibble = if is_odd { byte & 0x0f } else { byte >> 4 };
            count += NIBBLE_POP[nibble as usize];
        }
        *out = count;
    }
}

// ── aa_coverage_span AVX-512 BITALG tier ─────────────────────────────────────

#[cfg(target_arch = "x86_64")]
/// AVX-512 BITALG tier for `aa_coverage_span`.
///
/// Processes 128 output pixels (= 64 bytes) per loop iteration by running
/// `_mm512_popcnt_epi8` separately on the high and low nibbles of each of the
/// 4 AA sub-rows, then accumulating across rows.
///
/// Each byte of a row encodes two consecutive output pixels as nibbles:
/// - bits 7–4 (high nibble): even pixel `2k`
/// - bits 3–0 (low nibble):  odd  pixel `2k+1`
///
/// We mask each nibble to its own byte lane before calling `popcnt_epi8` so
/// that the per-byte popcount equals the per-pixel coverage count (0..=4).
/// After summing across the 4 rows, each even-pixel count lives in the high
/// half of its byte and each odd-pixel count lives in the low half; we unpack
/// them separately and interleave into `shape`.
///
/// # Safety
///
/// Caller must ensure `avx512bitalg` and `avx512bw` features are available.
#[target_feature(enable = "avx512bitalg,avx512bw")]
unsafe fn aa_coverage_span_avx512(rows: [&[u8]; 4], x0: usize, shape: &mut [u8]) {
    use std::arch::x86_64::{
        _mm512_add_epi8, _mm512_and_si512, _mm512_loadu_si512, _mm512_popcnt_epi8,
        _mm512_set1_epi8, _mm512_srli_epi16, _mm512_storeu_si512,
    };

    // x0 must be even: each byte covers pixels [2k, 2k+1], so an odd x0 would
    // misalign even/odd nibble assignments.  Fall back to scalar in release builds.
    debug_assert!(x0 & 1 == 0, "aa_coverage_span_avx512: x0={x0} must be even");
    if x0 & 1 != 0 {
        aa_coverage_span_scalar(rows, x0, shape);
        return;
    }

    let byte_x0 = x0 >> 1;
    let n = shape.len();

    // 0x0F mask: isolates low nibble (and high nibble after >>4) before popcnt.
    let mask_lo = _mm512_set1_epi8(0x0F_u8.cast_signed());

    // Number of complete 64-byte chunks (= 128 output pixels each).
    let n_bytes = n.div_ceil(2);
    let n_chunks = n_bytes / 64;

    for chunk_idx in 0..n_chunks {
        let byte_off = byte_x0 + chunk_idx * 64;

        let mut acc_hi = _mm512_set1_epi8(0);
        let mut acc_lo = _mm512_set1_epi8(0);

        for row in rows {
            debug_assert!(
                byte_off + 64 <= row.len(),
                "aa_coverage_span_avx512: byte_off+64={} out of bounds (row.len={})",
                byte_off + 64,
                row.len()
            );
            // SAFETY: byte_off + 64 ≤ row.len() asserted above; unaligned load is always valid.
            let v = unsafe { _mm512_loadu_si512(row[byte_off..].as_ptr().cast()) };
            // High nibble → bits 3–0 via >>4, then mask to kill carry from adjacent lane.
            let hi_bits = _mm512_and_si512(_mm512_srli_epi16(v, 4), mask_lo);
            // Low nibble → bits 3–0 via mask.
            let lo_bits = _mm512_and_si512(v, mask_lo);
            acc_hi = _mm512_add_epi8(acc_hi, _mm512_popcnt_epi8(hi_bits));
            acc_lo = _mm512_add_epi8(acc_lo, _mm512_popcnt_epi8(lo_bits));
        }

        let mut hi_buf = [0u8; 64];
        let mut lo_buf = [0u8; 64];
        // SAFETY: hi_buf and lo_buf are exactly 64 bytes; unaligned store is always valid.
        unsafe {
            _mm512_storeu_si512(hi_buf.as_mut_ptr().cast(), acc_hi);
            _mm512_storeu_si512(lo_buf.as_mut_ptr().cast(), acc_lo);
        }

        // Interleave: even pixel k*2 ← hi_buf[k], odd pixel k*2+1 ← lo_buf[k].
        let out_base = chunk_idx * 128;
        for k in 0..64 {
            let even_px = out_base + k * 2;
            let odd_px = even_px + 1;
            if even_px < n {
                shape[even_px] = hi_buf[k];
            }
            if odd_px < n {
                shape[odd_px] = lo_buf[k];
            }
        }
    }

    // Scalar remainder (< 64 bytes = < 128 output pixels).
    let scalar_out_start = n_chunks * 128;
    if scalar_out_start < n {
        aa_coverage_span_scalar(rows, x0 + scalar_out_start, &mut shape[scalar_out_start..]);
    }
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

/// Fill `shape[i]` with the AA coverage count (0..=16) for output pixel `x0 + i`.
///
/// `rows` are the four AA sub-row byte slices (one per `AA_SIZE` row), each of
/// length `row_bytes = (bitmap_width * 4 + 7) / 8`.  `x0` must be non-negative
/// and `x0 + shape.len()` must not exceed `bitmap_width`.
///
/// Selects the best available SIMD tier at runtime.
pub fn aa_coverage_span(rows: [&[u8]; 4], x0: usize, shape: &mut [u8]) {
    if shape.is_empty() {
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512bitalg") && is_x86_feature_detected!("avx512bw") {
            // SAFETY: we just confirmed both features are present.
            unsafe { aa_coverage_span_avx512(rows, x0, shape) };
            return;
        }
    }

    aa_coverage_span_scalar(rows, x0, shape);
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

    // ── aa_coverage_span tests ────────────────────────────────────────────────

    fn make_rows(data: [[u8; 4]; 4]) -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>) {
        (
            data[0].to_vec(),
            data[1].to_vec(),
            data[2].to_vec(),
            data[3].to_vec(),
        )
    }

    #[test]
    fn coverage_span_all_zero() {
        let (r0, r1, r2, r3) = make_rows([[0u8; 4]; 4]);
        let mut shape = [0xFFu8; 8];
        aa_coverage_span([&r0, &r1, &r2, &r3], 0, &mut shape);
        assert_eq!(shape, [0u8; 8]);
    }

    #[test]
    fn coverage_span_all_ones() {
        // 0xFF = high nibble 0xF (4 bits) + low nibble 0xF (4 bits), 8 output pixels.
        let (r0, r1, r2, r3) = make_rows([[0xFFu8; 4]; 4]);
        let mut shape = [0u8; 8];
        aa_coverage_span([&r0, &r1, &r2, &r3], 0, &mut shape);
        // Each pixel has 4 rows × 4 bits = 16.
        assert_eq!(shape, [16u8; 8]);
    }

    #[test]
    fn coverage_span_single_pixel_even() {
        // Pixel 0 is the high nibble of byte 0.  Set only the high nibble in row 0.
        let rows = [
            vec![0xF0u8, 0, 0, 0], // row 0: pixel 0 = 0xF = 4 bits
            vec![0u8; 4],
            vec![0u8; 4],
            vec![0u8; 4],
        ];
        let mut shape = [0u8; 2]; // pixels 0 and 1
        aa_coverage_span([&rows[0], &rows[1], &rows[2], &rows[3]], 0, &mut shape);
        assert_eq!(shape[0], 4, "pixel 0 should have count 4");
        assert_eq!(shape[1], 0, "pixel 1 should be 0");
    }

    #[test]
    fn coverage_span_single_pixel_odd() {
        // Pixel 1 is the low nibble of byte 0.  Set only the low nibble in row 0.
        let rows = [
            vec![0x0Fu8, 0, 0, 0], // row 0: pixel 1 = 0xF = 4 bits
            vec![0u8; 4],
            vec![0u8; 4],
            vec![0u8; 4],
        ];
        let mut shape = [0u8; 2];
        aa_coverage_span([&rows[0], &rows[1], &rows[2], &rows[3]], 0, &mut shape);
        assert_eq!(shape[0], 0, "pixel 0 should be 0");
        assert_eq!(shape[1], 4, "pixel 1 should have count 4");
    }

    #[test]
    fn coverage_span_x0_offset() {
        // x0=2: byte_idx for pixel 2 is 1 (pixel 2 = high nibble of byte 1).
        let rows = [
            vec![0u8, 0xA0, 0, 0], // byte 1 high nibble = 0xA = 1010 = 2 bits
            vec![0u8, 0x50, 0, 0], // byte 1 high nibble = 0x5 = 0101 = 2 bits
            vec![0u8; 4],
            vec![0u8; 4],
        ];
        // shape[0] = pixel 2 coverage = rows 0+1 high nibbles of byte 1 = 2+2 = 4
        let mut shape = [0u8; 1];
        aa_coverage_span([&rows[0], &rows[1], &rows[2], &rows[3]], 2, &mut shape);
        assert_eq!(shape[0], 4);
    }

    #[test]
    fn coverage_span_dispatch_matches_scalar() {
        // 256 output pixels = 128 bytes per row; exercises AVX-512 path on capable machines.
        const N: usize = 256;
        let row_bytes = N / 2;
        let r0: Vec<u8> = (0..row_bytes as u8).collect();
        let r1: Vec<u8> = (0..row_bytes as u8).map(|b| b.wrapping_mul(3)).collect();
        let r2: Vec<u8> = (0..row_bytes as u8).map(|b| b.wrapping_mul(7)).collect();
        let r3: Vec<u8> = (0..row_bytes as u8).map(|b| !b).collect();

        let mut expected = vec![0u8; N];
        aa_coverage_span_scalar([&r0, &r1, &r2, &r3], 0, &mut expected);

        let mut got = vec![0u8; N];
        aa_coverage_span([&r0, &r1, &r2, &r3], 0, &mut got);

        assert_eq!(got, expected, "dispatch mismatch on large input");
    }

    #[test]
    fn coverage_span_empty_is_noop() {
        let row = vec![0xFFu8; 4];
        let mut shape: [u8; 0] = [];
        // Must not panic.
        aa_coverage_span([&row, &row, &row, &row], 0, &mut shape);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx512_matches_scalar() {
        if !is_x86_feature_detected!("avx512bitalg") || !is_x86_feature_detected!("avx512bw") {
            return;
        }
        const N: usize = 300; // odd count to exercise remainder path
        let row_bytes = N.div_ceil(2);
        let r0: Vec<u8> = (0..row_bytes).map(|i| (i * 37 + 11) as u8).collect();
        let r1: Vec<u8> = (0..row_bytes).map(|i| (i * 53 + 7) as u8).collect();
        let r2: Vec<u8> = (0..row_bytes).map(|i| (i * 17 + 3) as u8).collect();
        let r3: Vec<u8> = (0..row_bytes).map(|i| (!i) as u8).collect();

        let mut expected = vec![0u8; N];
        aa_coverage_span_scalar([&r0, &r1, &r2, &r3], 0, &mut expected);

        let mut got = vec![0u8; N];
        // SAFETY: we just confirmed both features are present.
        unsafe { aa_coverage_span_avx512([&r0, &r1, &r2, &r3], 0, &mut got) };

        assert_eq!(got, expected, "AVX-512 mismatch vs scalar");
    }
}
