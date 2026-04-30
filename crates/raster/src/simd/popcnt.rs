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
//! # Acceleration tiers for `popcnt_aa_row`
//!
//! ## x86-64 (most to least preferred)
//! 1. **AVX-512 VPOPCNTDQ** (`avx512vpopcntdq` + `avx512bw`): 64 B/iter via `_mm512_popcnt_epi8`.
//! 2. **`popcnt`**: 8 B/iter via the scalar hardware `POPCNT` instruction.
//! 3. **Scalar**: `u8::count_ones` per byte.
//!
//! ## aarch64
//! 1. **NEON `vcntq_u8`**: hardware byte popcount, 16 B/iter.
//!    Result narrowed with `vpaddlq_u8` (→ u16) then summed with `vaddvq_u16`.
//!    NEON is mandatory on all ARMv8-A cores; no runtime detection is needed.
//!
//! # Acceleration tiers for `aa_coverage_span`
//!
//! ## x86-64 (most to least preferred)
//! 1. **AVX-512 BITALG** (`avx512bitalg` + `avx512bw`): `_mm512_popcnt_epi8` on
//!    nibble-isolated bytes, 128 output pixels per 64-byte iteration.
//! 2. **Scalar**: byte-by-byte nibble lookup via `NIBBLE_POP` table.
//!
//! ## aarch64
//! 1. **NEON**: nibble-isolated `vcntq_u8`, 32 output pixels per 16-byte iteration.
//!    High/low nibbles extracted with `vshrq_n_u8` + `vandq_u8`; four-row
//!    accumulation in u8 (max 16 ≤ 255); interleaved into `shape` via `vst2q_u8`.
//!    NEON is mandatory on all ARMv8-A cores; no runtime detection is needed.

// ── Scalar helpers ────────────────────────────────────────────────────────────

/// Count set bits in `row` one byte at a time.
#[inline]
#[cfg_attr(
    target_arch = "aarch64",
    expect(dead_code, reason = "NEON dispatch replaces scalar on aarch64")
)]
pub(super) fn popcnt_aa_row_scalar(row: &[u8]) -> u32 {
    row.iter().map(|b| b.count_ones()).sum()
}

/// Nibble popcount table: `NIBBLE_POP[n]` = number of set bits in `n` (0..=4).
const NIBBLE_POP: [u8; 16] = [0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4];

/// Scalar fallback for `aa_coverage_span`.
///
/// Writes the coverage count (0..=16) for output pixel `x0 + i` into `shape[i]`
/// by looking up the two nibbles (high = even pixel, low = odd pixel) of each
/// packed row byte in `NIBBLE_POP` and summing across the four rows.
///
/// # Panics
///
/// Panics if any byte index derived from `x0 + i` is out of bounds for a row
/// slice — i.e. if the caller's precondition `x0 + shape.len() ≤ bitmap_width`
/// is violated.
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
            let byte = row[byte_idx];
            let nibble = if is_odd { byte & 0x0f } else { byte >> 4 };
            count += NIBBLE_POP[nibble as usize];
        }
        *out = count;
    }
}

// ── Shared chunk-parameter computation ───────────────────────────────────────

/// Compute the byte offset of the first row byte for pixel `x0`, the number of
/// output pixels `n`, and the number of complete SIMD chunks of `chunk_bytes`
/// row bytes (= `chunk_bytes * 2` output pixels) that fit in the span.
///
/// Both the NEON and AVX-512 coverage-span kernels use the same arithmetic;
/// this function centralises it to avoid drift between the two.
#[inline]
fn coverage_chunk_params(x0: usize, n: usize, chunk_bytes: usize) -> (usize, usize) {
    let byte_x0 = x0 >> 1;
    // n.div_ceil(2): number of row bytes touched by the span.
    // Integer-divide by chunk_bytes to get complete chunks only.
    let n_chunks = n.div_ceil(2) / chunk_bytes;
    (byte_x0, n_chunks)
}

// ── aarch64 NEON tiers ────────────────────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
/// NEON tier for `popcnt_aa_row`: 16 B/iter using `vcntq_u8`.
///
/// `vcntq_u8` is a single hardware instruction on all ARMv8-A cores
/// (M1–M4, Cortex-A72/A76).  NEON is mandatory on aarch64; no runtime
/// detection is needed.
///
/// # Safety
///
/// Must be compiled for `target_arch = "aarch64"`.  NEON is mandatory on
/// all ARMv8-A targets covered by this cfg, so the intrinsics are always valid.
#[target_feature(enable = "neon")]
unsafe fn popcnt_aa_row_neon(row: &[u8]) -> u32 {
    use std::arch::aarch64::{uint8x16_t, vaddvq_u16, vcntq_u8, vld1q_u8, vpaddlq_u8};

    let mut total = 0u32;
    let mut chunks = row.chunks_exact(16);
    for chunk in chunks.by_ref() {
        // SAFETY: chunk is exactly 16 bytes; unaligned loads are always valid on ARMv8.
        let v: uint8x16_t = unsafe { vld1q_u8(chunk.as_ptr()) };
        // vcntq_u8: hardware byte-level popcount (ARMv8-A CNT instruction).
        let pcnt: uint8x16_t = vcntq_u8(v);
        // vpaddlq_u8: pairwise widen u8→u16, preventing overflow
        // (max 8 bits/byte × 16 bytes = 128 ≤ u16::MAX).
        let wide = vpaddlq_u8(pcnt);
        // vaddvq_u16: horizontal sum of 8 u16 lanes → scalar.
        total += u32::from(vaddvq_u16(wide));
    }
    for &b in chunks.remainder() {
        total += b.count_ones();
    }
    total
}

#[cfg(target_arch = "aarch64")]
/// NEON tier for `aa_coverage_span`: 32 output pixels (16 row bytes) per iteration.
///
/// Each row byte encodes two consecutive output pixels as nibbles:
/// - bits 7–4 (high nibble): even pixel `2k`   — extracted via `vshrq_n_u8(v, 4) & 0x0F`
/// - bits 3–0 (low  nibble): odd  pixel `2k+1` — extracted via `v & 0x0F`
///
/// `vcntq_u8` counts set bits per nibble-byte; accumulation across the four rows
/// stays in u8 (max 4 rows × 4 bits/nibble = 16 ≤ 255).  `vst2q_u8` interleaves
/// the even/odd accumulators into `shape` in a single store.
///
/// Odd `x0` cannot be handled by this kernel (nibble boundaries are byte-aligned).
/// The caller falls back to scalar when `x0` is odd.
///
/// # Safety
///
/// Must be compiled for `target_arch = "aarch64"`.  NEON is mandatory on all
/// ARMv8-A targets covered by this cfg.  Caller must ensure:
/// - `x0` is even (enforced at call site; odd `x0` is redirected to scalar).
/// - `x0 + shape.len() ≤ bitmap_width` (precondition of `aa_coverage_span`).
#[target_feature(enable = "neon")]
unsafe fn aa_coverage_span_neon(rows: [&[u8]; 4], x0: usize, shape: &mut [u8]) {
    use std::arch::aarch64::{
        uint8x16x2_t, vaddq_u8, vandq_u8, vcntq_u8, vdupq_n_u8, vld1q_u8, vshrq_n_u8, vst2q_u8,
    };

    debug_assert!(x0 & 1 == 0, "aa_coverage_span_neon: x0={x0} must be even");

    let n = shape.len();
    let (byte_x0, n_chunks) = coverage_chunk_params(x0, n, 16);

    // SAFETY: NEON intrinsics valid on all aarch64 targets (cfg gate).
    // Row bounds are checked inside the loop before each load.
    unsafe {
        let mask_lo = vdupq_n_u8(0x0F);

        for chunk_idx in 0..n_chunks {
            let byte_off = byte_x0 + chunk_idx * 16;

            let mut acc_hi = vdupq_n_u8(0);
            let mut acc_lo = vdupq_n_u8(0);

            for row in rows {
                assert!(
                    byte_off + 16 <= row.len(),
                    "aa_coverage_span_neon: row too short: \
                     need {} bytes at offset {byte_off}, have {}",
                    byte_off + 16,
                    row.len(),
                );
                let v = vld1q_u8(row[byte_off..].as_ptr());
                // High nibble → bits 3–0 via arithmetic right-shift, then mask.
                let hi = vandq_u8(vshrq_n_u8(v, 4), mask_lo);
                // Low nibble → bits 3–0 directly.
                let lo = vandq_u8(v, mask_lo);
                acc_hi = vaddq_u8(acc_hi, vcntq_u8(hi));
                acc_lo = vaddq_u8(acc_lo, vcntq_u8(lo));
            }

            // vst2q_u8 interleaves the two 16-byte vectors as
            // [hi[0], lo[0], hi[1], lo[1], …] = [px0, px1, px2, px3, …].
            let out_base = chunk_idx * 32;
            let remaining = n - out_base;
            if remaining >= 32 {
                vst2q_u8(shape[out_base..].as_mut_ptr(), uint8x16x2_t(acc_hi, acc_lo));
            } else {
                // Partial last chunk: write through a staging buffer to avoid a
                // 32-byte store past the end of `shape`.
                let mut tmp = [0u8; 32];
                vst2q_u8(tmp.as_mut_ptr(), uint8x16x2_t(acc_hi, acc_lo));
                shape[out_base..].copy_from_slice(&tmp[..remaining]);
            }
        }
    }

    // Scalar remainder for any output pixels not covered by complete NEON chunks.
    let scalar_start = n_chunks * 32;
    if scalar_start < n {
        aa_coverage_span_scalar(rows, x0 + scalar_start, &mut shape[scalar_start..]);
    }
}

// ── x86-64 SIMD tiers ─────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
/// `popcnt` tier: 8 B/iter via the scalar hardware `POPCNT` instruction.
///
/// # Safety
///
/// Caller must ensure the `popcnt` CPU feature is available.
#[target_feature(enable = "popcnt")]
unsafe fn popcnt_aa_row_sse(row: &[u8]) -> u32 {
    use std::arch::x86_64::_popcnt64;

    let mut total = 0u32;
    let mut chunks = row.chunks_exact(8);
    for chunk in chunks.by_ref() {
        // SAFETY: chunk is exactly 8 bytes; read_unaligned handles any alignment.
        // The `popcnt` feature is guaranteed by `#[target_feature]`.
        // SAFETY: chunk is exactly 8 bytes; read_unaligned handles any alignment.
        // `popcnt` CPU feature guaranteed by `#[target_feature]`.
        let word = unsafe { chunk.as_ptr().cast::<u64>().read_unaligned() };
        // _popcnt64 is safe to call within a `#[target_feature(enable = "popcnt")]`
        // function — the feature guarantee makes it non-unsafe in this context.
        #[expect(clippy::cast_sign_loss, reason = "_popcnt64 returns a non-negative count in 0..=64")]
        {
            total += _popcnt64(word.cast_signed()) as u32;
        }
    }
    for &b in chunks.remainder() {
        total += b.count_ones();
    }
    total
}

#[cfg(target_arch = "x86_64")]
/// AVX-512 VPOPCNTDQ tier: 64 B/iter via `_mm512_popcnt_epi8`.
///
/// # Safety
///
/// Caller must ensure `avx512vpopcntdq` and `avx512bw` CPU features are available.
#[target_feature(enable = "avx512vpopcntdq,avx512bw")]
unsafe fn popcnt_aa_row_avx512(row: &[u8]) -> u32 {
    use std::arch::x86_64::{_mm512_loadu_si512, _mm512_popcnt_epi8, _mm512_storeu_si512};

    let mut total = 0u32;
    let mut chunks = row.chunks_exact(64);
    for chunk in chunks.by_ref() {
        // SAFETY: chunk is exactly 64 bytes; unaligned load/store are always valid.
        // `avx512vpopcntdq` + `avx512bw` guaranteed by `#[target_feature]`.
        unsafe {
            let v = _mm512_loadu_si512(chunk.as_ptr().cast());
            // Each byte lane receives the popcount of the corresponding source byte.
            let pcnt = _mm512_popcnt_epi8(v);
            let mut buf = [0u8; 64];
            _mm512_storeu_si512(buf.as_mut_ptr().cast(), pcnt);
            for b in buf {
                total += u32::from(b);
            }
        }
    }
    for &b in chunks.remainder() {
        total += b.count_ones();
    }
    total
}

// ── aa_coverage_span AVX-512 BITALG tier ─────────────────────────────────────

#[cfg(target_arch = "x86_64")]
/// AVX-512 BITALG tier for `aa_coverage_span`.
///
/// Processes 128 output pixels (= 64 row bytes) per loop iteration.
/// Each byte encodes two pixels as nibbles:
/// - bits 7–4 (high nibble): even pixel `2k`
/// - bits 3–0 (low  nibble): odd  pixel `2k+1`
///
/// Each nibble is isolated into its own byte lane before `_mm512_popcnt_epi8`,
/// so the per-lane result equals the per-pixel coverage count (0..=4).  After
/// accumulating across all four rows the two 64-lane buffers are interleaved
/// into `shape`.
///
/// Odd `x0` is redirected to scalar by the caller.
///
/// # Safety
///
/// Caller must ensure `avx512bitalg` and `avx512bw` CPU features are available.
#[target_feature(enable = "avx512bitalg,avx512bw")]
unsafe fn aa_coverage_span_avx512(rows: [&[u8]; 4], x0: usize, shape: &mut [u8]) {
    use std::arch::x86_64::{
        _mm512_add_epi8, _mm512_and_si512, _mm512_loadu_si512, _mm512_popcnt_epi8,
        _mm512_set1_epi8, _mm512_setzero_si512, _mm512_srli_epi16, _mm512_storeu_si512,
    };

    debug_assert!(x0 & 1 == 0, "aa_coverage_span_avx512: x0={x0} must be even");

    let n = shape.len();
    let (byte_x0, n_chunks) = coverage_chunk_params(x0, n, 64);

    // 0x0F mask: after shifting or masking, isolates the low 4 bits of each byte.
    // AVX-512 intrinsics are safe to call within a `#[target_feature]` function —
    // the feature guarantee makes them non-unsafe in this context.
    let mask_lo = _mm512_set1_epi8(0x0F_u8.cast_signed());

    for chunk_idx in 0..n_chunks {
        let byte_off = byte_x0 + chunk_idx * 64;

        let (mut acc_hi, mut acc_lo) = (_mm512_setzero_si512(), _mm512_setzero_si512());

        for row in rows {
            assert!(
                byte_off + 64 <= row.len(),
                "aa_coverage_span_avx512: row too short: \
                 need {} bytes at offset {byte_off}, have {}",
                byte_off + 64,
                row.len(),
            );
            // SAFETY: byte_off + 64 ≤ row.len() asserted above.
            unsafe {
                let v = _mm512_loadu_si512(row[byte_off..].as_ptr().cast());
                // High nibble: arithmetic right-shift by 4, then mask off upper bits.
                let hi = _mm512_and_si512(_mm512_srli_epi16(v, 4), mask_lo);
                // Low nibble: mask directly.
                let lo = _mm512_and_si512(v, mask_lo);
                acc_hi = _mm512_add_epi8(acc_hi, _mm512_popcnt_epi8(hi));
                acc_lo = _mm512_add_epi8(acc_lo, _mm512_popcnt_epi8(lo));
            }
        }

        let mut hi_buf = [0u8; 64];
        let mut lo_buf = [0u8; 64];
        // SAFETY: buffers are exactly 64 bytes; unaligned stores are always valid.
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

    // Scalar remainder (< 64 row bytes = < 128 output pixels).
    let scalar_start = n_chunks * 128;
    if scalar_start < n {
        aa_coverage_span_scalar(rows, x0 + scalar_start, &mut shape[scalar_start..]);
    }
}

// ── Public dispatch ───────────────────────────────────────────────────────────

/// Count the number of set bits in an `AaBuf` row.
///
/// Selects the fastest available tier at compile/runtime:
/// - x86-64: AVX-512 VPOPCNTDQ → hardware `popcnt` → scalar
/// - aarch64: NEON `vcntq_u8` (always available on ARMv8-A)
/// - other architectures: scalar
#[must_use]
pub fn popcnt_aa_row(row: &[u8]) -> u32 {
    dispatch_popcnt(row)
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn dispatch_popcnt(row: &[u8]) -> u32 {
    if is_x86_feature_detected!("avx512vpopcntdq") && is_x86_feature_detected!("avx512bw") {
        // SAFETY: both features confirmed present.
        return unsafe { popcnt_aa_row_avx512(row) };
    }
    if is_x86_feature_detected!("popcnt") {
        // SAFETY: `popcnt` feature confirmed present.
        return unsafe { popcnt_aa_row_sse(row) };
    }
    popcnt_aa_row_scalar(row)
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn dispatch_popcnt(row: &[u8]) -> u32 {
    // NEON is mandatory on all ARMv8-A targets (no runtime detection needed).
    // SAFETY: aarch64 always has NEON.
    unsafe { popcnt_aa_row_neon(row) }
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
#[inline]
fn dispatch_popcnt(row: &[u8]) -> u32 {
    popcnt_aa_row_scalar(row)
}

/// Fill `shape[i]` with the AA coverage count (0..=16) for output pixel `x0 + i`.
///
/// `rows` are the four `AaBuf` sub-row byte slices, each of length
/// `(bitmap_width * AA_SIZE + 7) / 8`.
///
/// # Preconditions
///
/// - `x0 + shape.len() ≤ bitmap_width` — span must not exceed the bitmap row.
/// - `x0` should be **even** for the SIMD tiers to be used.  An odd `x0` is
///   silently handled by the scalar tier (correct output, slower).
///
/// # Panics
///
/// Panics if any row slice is shorter than required by the span — i.e. if the
/// `x0 + shape.len() ≤ bitmap_width` precondition is violated.
pub fn aa_coverage_span(rows: [&[u8]; 4], x0: usize, shape: &mut [u8]) {
    if shape.is_empty() {
        return;
    }
    dispatch_coverage(rows, x0, shape);
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn dispatch_coverage(rows: [&[u8]; 4], x0: usize, shape: &mut [u8]) {
    if x0 & 1 != 0 {
        // Odd x0: nibble boundaries are byte-aligned; SIMD paths require even x0.
        aa_coverage_span_scalar(rows, x0, shape);
        return;
    }
    if is_x86_feature_detected!("avx512bitalg") && is_x86_feature_detected!("avx512bw") {
        // SAFETY: both features confirmed present.
        unsafe { aa_coverage_span_avx512(rows, x0, shape) };
    } else {
        aa_coverage_span_scalar(rows, x0, shape);
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn dispatch_coverage(rows: [&[u8]; 4], x0: usize, shape: &mut [u8]) {
    if x0 & 1 != 0 {
        // Odd x0: fall back to scalar (NEON kernel requires even x0).
        aa_coverage_span_scalar(rows, x0, shape);
        return;
    }
    // NEON is mandatory on all ARMv8-A targets.
    // SAFETY: aarch64 always has NEON.
    unsafe { aa_coverage_span_neon(rows, x0, shape) };
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
#[inline]
fn dispatch_coverage(rows: [&[u8]; 4], x0: usize, shape: &mut [u8]) {
    aa_coverage_span_scalar(rows, x0, shape);
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── popcnt_aa_row ─────────────────────────────────────────────────────────

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
        // 0b1010_1010 = 4 bits set per byte
        assert_eq!(popcnt_aa_row_scalar(&[0xAAu8; 4]), 16);
    }

    #[test]
    fn scalar_empty() {
        assert_eq!(popcnt_aa_row_scalar(&[]), 0);
    }

    #[test]
    fn dispatch_matches_scalar() {
        let row: Vec<u8> = (0u8..=127).collect();
        assert_eq!(popcnt_aa_row(&row), popcnt_aa_row_scalar(&row));
    }

    #[test]
    fn dispatch_non_power_of_two_length() {
        // 270 bytes — exercises both the full-chunk path and the scalar remainder
        // on every tier (64-byte AVX-512 chunks leave 14 bytes; 16-byte NEON
        // chunks leave 14 bytes; 8-byte popcnt chunks leave 6 bytes).
        let row: Vec<u8> = (0u8..=255).chain(0u8..14).collect();
        assert_eq!(row.len(), 270);
        assert_eq!(popcnt_aa_row(&row), popcnt_aa_row_scalar(&row));
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn sse_popcnt_matches_scalar() {
        if !is_x86_feature_detected!("popcnt") {
            return;
        }
        let row: Vec<u8> = (0u8..=127).collect();
        // SAFETY: `popcnt` feature confirmed present.
        let got = unsafe { popcnt_aa_row_sse(&row) };
        assert_eq!(got, popcnt_aa_row_scalar(&row));
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_popcnt_matches_scalar() {
        // 270 bytes — exercises full 16-byte chunks and a 14-byte scalar remainder.
        let row: Vec<u8> = (0u8..=255).chain(0u8..14).collect();
        assert_eq!(row.len(), 270);
        // SAFETY: aarch64 always has NEON.
        let got = unsafe { popcnt_aa_row_neon(&row) };
        assert_eq!(got, popcnt_aa_row_scalar(&row));
    }

    // ── aa_coverage_span ──────────────────────────────────────────────────────

    /// Build four row buffers from a `[[u8; N]; 4]` literal.
    fn make_rows<const N: usize>(data: [[u8; N]; 4]) -> [Vec<u8>; 4] {
        data.map(|r| r.to_vec())
    }

    #[test]
    fn coverage_span_all_zero() {
        let rows = make_rows([[0u8; 4]; 4]);
        let mut shape = [0xFFu8; 8];
        aa_coverage_span([&rows[0], &rows[1], &rows[2], &rows[3]], 0, &mut shape);
        assert_eq!(shape, [0u8; 8]);
    }

    #[test]
    fn coverage_span_all_ones() {
        // 0xFF: high nibble 0xF (4 bits) + low nibble 0xF (4 bits) → 8 output pixels.
        // 4 rows × 4 bits/nibble = 16 per pixel.
        let rows = make_rows([[0xFFu8; 4]; 4]);
        let mut shape = [0u8; 8];
        aa_coverage_span([&rows[0], &rows[1], &rows[2], &rows[3]], 0, &mut shape);
        assert_eq!(shape, [16u8; 8]);
    }

    #[test]
    fn coverage_span_single_pixel_even() {
        // Pixel 0 = high nibble of byte 0.  Set only that nibble in row 0.
        let rows = [
            vec![0xF0u8, 0, 0, 0],
            vec![0u8; 4],
            vec![0u8; 4],
            vec![0u8; 4],
        ];
        let mut shape = [0u8; 2];
        aa_coverage_span([&rows[0], &rows[1], &rows[2], &rows[3]], 0, &mut shape);
        assert_eq!(shape, [4, 0]);
    }

    #[test]
    fn coverage_span_single_pixel_odd() {
        // Pixel 1 = low nibble of byte 0.  Set only that nibble in row 0.
        let rows = [
            vec![0x0Fu8, 0, 0, 0],
            vec![0u8; 4],
            vec![0u8; 4],
            vec![0u8; 4],
        ];
        let mut shape = [0u8; 2];
        aa_coverage_span([&rows[0], &rows[1], &rows[2], &rows[3]], 0, &mut shape);
        assert_eq!(shape, [0, 4]);
    }

    #[test]
    fn coverage_span_x0_offset() {
        // x0=2: pixel 2 = high nibble of byte 1.
        // Row 0 byte 1 high nibble = 0xA = 0b1010 = 2 bits.
        // Row 1 byte 1 high nibble = 0x5 = 0b0101 = 2 bits.  Total = 4.
        let rows = [
            vec![0u8, 0xA0, 0, 0],
            vec![0u8, 0x50, 0, 0],
            vec![0u8; 4],
            vec![0u8; 4],
        ];
        let mut shape = [0u8; 1];
        aa_coverage_span([&rows[0], &rows[1], &rows[2], &rows[3]], 2, &mut shape);
        assert_eq!(shape[0], 4);
    }

    #[test]
    fn coverage_span_odd_x0_matches_scalar() {
        // Odd x0=1: SIMD tiers must fall back to scalar; result must still be correct.
        const N: usize = 10;
        let row_bytes = (1 + N).div_ceil(2); // bytes needed for pixels 1..=10
        let r0: Vec<u8> = (0..row_bytes).map(|i| (i as u8).wrapping_mul(0x37)).collect();
        let r1: Vec<u8> = (0..row_bytes).map(|i| (i as u8).wrapping_mul(0x53)).collect();
        let r2: Vec<u8> = (0..row_bytes).map(|i| (i as u8).wrapping_mul(0x17)).collect();
        let r3: Vec<u8> = (0..row_bytes).map(|i| !(i as u8)).collect();

        let mut expected = vec![0u8; N];
        aa_coverage_span_scalar([&r0, &r1, &r2, &r3], 1, &mut expected);

        let mut got = vec![0u8; N];
        aa_coverage_span([&r0, &r1, &r2, &r3], 1, &mut got);

        assert_eq!(got, expected, "odd x0 result mismatch");
    }

    #[test]
    fn coverage_span_dispatch_matches_scalar() {
        // 300 output pixels = 150 row bytes.
        // Exercises: AVX-512 (2 × 64-byte chunks + 22-byte scalar remainder),
        //            NEON    (9 × 16-byte chunks + 6-byte scalar remainder),
        //            scalar  (full path).
        const N: usize = 300;
        let row_bytes = N.div_ceil(2);
        let r0: Vec<u8> = (0..row_bytes).map(|i| (i as u8).wrapping_mul(0x37)).collect();
        let r1: Vec<u8> = (0..row_bytes).map(|i| (i as u8).wrapping_mul(0x53)).collect();
        let r2: Vec<u8> = (0..row_bytes).map(|i| (i as u8).wrapping_mul(0x17)).collect();
        let r3: Vec<u8> = (0..row_bytes).map(|i| !(i as u8)).collect();

        let mut expected = vec![0u8; N];
        aa_coverage_span_scalar([&r0, &r1, &r2, &r3], 0, &mut expected);

        let mut got = vec![0u8; N];
        aa_coverage_span([&r0, &r1, &r2, &r3], 0, &mut got);

        assert_eq!(got, expected, "dispatch mismatch on N={N}");
    }

    #[test]
    fn coverage_span_empty_is_noop() {
        let row = vec![0xFFu8; 4];
        let mut shape: [u8; 0] = [];
        aa_coverage_span([&row, &row, &row, &row], 0, &mut shape); // must not panic
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx512_coverage_matches_scalar() {
        if !is_x86_feature_detected!("avx512bitalg") || !is_x86_feature_detected!("avx512bw") {
            return;
        }
        const N: usize = 300; // 2 full 64-byte chunks + 22-byte remainder
        let row_bytes = N.div_ceil(2);
        let r0: Vec<u8> = (0..row_bytes).map(|i| (i as u8).wrapping_mul(37).wrapping_add(11)).collect();
        let r1: Vec<u8> = (0..row_bytes).map(|i| (i as u8).wrapping_mul(53).wrapping_add(7)).collect();
        let r2: Vec<u8> = (0..row_bytes).map(|i| (i as u8).wrapping_mul(17).wrapping_add(3)).collect();
        let r3: Vec<u8> = (0..row_bytes).map(|i| !(i as u8)).collect();

        let mut expected = vec![0u8; N];
        aa_coverage_span_scalar([&r0, &r1, &r2, &r3], 0, &mut expected);

        let mut got = vec![0u8; N];
        // SAFETY: both features confirmed present above.
        unsafe { aa_coverage_span_avx512([&r0, &r1, &r2, &r3], 0, &mut got) };

        assert_eq!(got, expected, "AVX-512 coverage mismatch vs scalar");
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_coverage_matches_scalar() {
        // 300 output pixels — 9 full 16-byte NEON chunks + 6-byte scalar remainder.
        const N: usize = 300;
        let row_bytes = N.div_ceil(2);
        let r0: Vec<u8> = (0..row_bytes).map(|i| (i as u8).wrapping_mul(37).wrapping_add(11)).collect();
        let r1: Vec<u8> = (0..row_bytes).map(|i| (i as u8).wrapping_mul(53).wrapping_add(7)).collect();
        let r2: Vec<u8> = (0..row_bytes).map(|i| (i as u8).wrapping_mul(17).wrapping_add(3)).collect();
        let r3: Vec<u8> = (0..row_bytes).map(|i| !(i as u8)).collect();

        let mut expected = vec![0u8; N];
        aa_coverage_span_scalar([&r0, &r1, &r2, &r3], 0, &mut expected);

        let mut got = vec![0u8; N];
        // SAFETY: aarch64 always has NEON.
        unsafe { aa_coverage_span_neon([&r0, &r1, &r2, &r3], 0, &mut got) };

        assert_eq!(got, expected, "NEON coverage mismatch vs scalar");
    }
}
