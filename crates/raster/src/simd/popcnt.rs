//! Per-pixel AA coverage counts for `AaBuf` rows.
//!
//! Single public entry point:
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
//! # Acceleration tiers for `aa_coverage_span`
//!
//! ## x86-64 (most to least preferred)
//! 1. **AVX-512 BITALG** (`avx512bitalg` + `avx512bw`): `_mm512_popcnt_epi8` on
//!    nibble-isolated bytes, 128 output pixels per 64-byte iteration.
//! 2. **AVX2** (`avx2`): VPSHUFB nibble lookup, 64 output pixels per 32-byte iteration.
//! 3. **Scalar**: byte-by-byte nibble lookup via `NIBBLE_POP` table.
//!
//! ## aarch64 (most to least preferred)
//! 1. **SVE2** (`nightly-sve2` feature + `sve2` target feature): nibble-isolated `svcnt_u8_z`,
//!    `svcntb()*2` output pixels per iteration. Requires nightly Rust and `sve2` CPU feature.
//! 2. **NEON**: nibble-isolated `vcntq_u8`, 32 output pixels per 16-byte iteration.
//!    High/low nibbles extracted with `vshrq_n_u8` + `vandq_u8`; four-row
//!    accumulation in u8 (max 16 ≤ 255); interleaved into `shape` via `vst2q_u8`.
//!    NEON is mandatory on all ARMv8-A cores; no runtime detection is needed.

// ── Scalar helpers ────────────────────────────────────────────────────────────

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

/// Compute the byte offset of the first row byte for pixel `x0` and the number
/// of complete SIMD chunks of `chunk_bytes` row bytes (= `chunk_bytes * 2`
/// output pixels) that fit in the span of `n` output pixels.
///
/// Used by all SIMD coverage-span kernels to centralise this arithmetic and
/// avoid drift between implementations.
///
/// # Panics
///
/// Panics in debug builds if `chunk_bytes` is zero.
#[inline]
fn coverage_chunk_params(x0: usize, n: usize, chunk_bytes: usize) -> (usize, usize) {
    debug_assert!(
        chunk_bytes > 0,
        "coverage_chunk_params: chunk_bytes must be > 0"
    );
    let byte_x0 = x0 >> 1;
    // n.div_ceil(2): number of row bytes touched by the span.
    // Integer-divide by chunk_bytes to get complete chunks only.
    let n_chunks = n.div_ceil(2) / chunk_bytes;
    (byte_x0, n_chunks)
}

// ── aarch64 NEON tier ─────────────────────────────────────────────────────────

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

// ── aarch64 SVE2 tiers ───────────────────────────────────────────────────────
//
// Requires:
//   - Cargo feature `nightly-sve2`
//   - Rust nightly (stdarch_aarch64_sve is not yet stable)
//   - CPU `sve2` target feature at runtime
//
// On fixed-128-bit SVE2 (Apple M4, Graviton4 at 128b mode) svcntb() == 16,
// giving the same throughput as NEON. On wide-SVE2 server chips (Graviton4 at
// full width, Neoverse V2) svcntb() may be 32–64, giving 2–4× NEON throughput.

#[cfg(all(target_arch = "aarch64", feature = "nightly-sve2"))]
/// SVE2 tier for `aa_coverage_span`.
///
/// Each SVE2 vector of `vl` row bytes encodes `vl * 2` output pixels as
/// nibble pairs. High nibbles (even pixels) and low nibbles (odd pixels) are
/// extracted and popcounted independently with `svcnt_u8_z`. The two result
/// vectors are then scattered into `shape` by interleaving.
///
/// Odd `x0` is redirected to scalar by the caller (nibble boundaries are
/// byte-aligned).
///
/// # Safety
///
/// Caller must ensure the `sve2` CPU feature is available.
#[target_feature(enable = "sve2")]
unsafe fn aa_coverage_span_sve2(rows: [&[u8]; 4], x0: usize, shape: &mut [u8]) {
    use std::arch::aarch64::{
        svadd_u8_z, svand_u8_z, svcnt_u8_z, svcntb, svdup_n_u8, svld1_u8, svlsr_u8_z, svptrue_b8,
        svst1_u8,
    };

    debug_assert!(x0 & 1 == 0, "aa_coverage_span_sve2: x0={x0} must be even");

    #[expect(
        clippy::cast_possible_truncation,
        reason = "aarch64 is 64-bit; svcntb() ≤ 256 fits in usize"
    )]
    let vl = svcntb() as usize;
    let pg = svptrue_b8();
    let mask_lo = svdup_n_u8(0x0F);
    let shift4 = svdup_n_u8(4u8);

    let n = shape.len();
    let (byte_x0, n_chunks) = coverage_chunk_params(x0, n, vl);

    // Staging buffers for svst1_u8 output. Allocated once outside the chunk loop
    // to avoid per-iteration heap allocation. vl is at most 256 bytes.
    let mut hi_buf = vec![0u8; vl];
    let mut lo_buf = vec![0u8; vl];

    for chunk_idx in 0..n_chunks {
        let byte_off = byte_x0 + chunk_idx * vl;

        let mut acc_hi = svdup_n_u8(0u8);
        let mut acc_lo = svdup_n_u8(0u8);

        for row in rows {
            assert!(
                byte_off + vl <= row.len(),
                "aa_coverage_span_sve2: row too short: \
                 need {} bytes at offset {byte_off}, have {}",
                byte_off + vl,
                row.len(),
            );
            // SAFETY: caller guarantees sve2 is available; ptr is in-bounds (assert above).
            let v = unsafe { svld1_u8(pg, row.as_ptr().add(byte_off)) };
            // High nibble: shift right 4, mask to low 4 bits.
            let hi = svand_u8_z(pg, svlsr_u8_z(pg, v, shift4), mask_lo);
            // Low nibble: mask directly.
            let lo = svand_u8_z(pg, v, mask_lo);
            // Accumulate per-nibble popcount across rows (max 4 rows × 4 bits = 16 ≤ u8::MAX).
            acc_hi = svadd_u8_z(pg, acc_hi, svcnt_u8_z(pg, hi));
            acc_lo = svadd_u8_z(pg, acc_lo, svcnt_u8_z(pg, lo));
        }

        // Store to staging buffers and interleave into shape:
        // shape[out_base + k*2] = hi_buf[k] (even pixel), shape[out_base + k*2+1] = lo_buf[k] (odd).
        // SAFETY: caller guarantees sve2 is available; bufs are vl bytes, matching pg width.
        unsafe {
            svst1_u8(pg, hi_buf.as_mut_ptr(), acc_hi);
            svst1_u8(pg, lo_buf.as_mut_ptr(), acc_lo);
        }

        let out_base = chunk_idx * vl * 2;
        for k in 0..vl {
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

    // Scalar remainder for pixels not covered by complete SVE2 chunks.
    let scalar_start = n_chunks * vl * 2;
    if scalar_start < n {
        aa_coverage_span_scalar(rows, x0 + scalar_start, &mut shape[scalar_start..]);
    }
}

// ── x86-64 AVX2 tier ──────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
/// AVX2 tier for `aa_coverage_span`: 64 output pixels (32 row bytes) per iteration.
///
/// Each row byte encodes two consecutive pixels as nibbles.  VPSHUFB is used
/// as a 16-entry parallel lookup table for nibble popcounts. Accumulation
/// across the four rows stays in u8 (max 4 rows × 4 bits/nibble = 16 ≤ 255).
///
/// After the four-row accumulation, the 32-element `hi` (even pixels) and
/// `lo` (odd pixels) u8 vectors are interleaved into `shape` by extracting
/// 128-bit lanes and storing via a staging buffer.
///
/// Odd `x0` is redirected to scalar by the caller.
///
/// # Safety
///
/// Caller must ensure the `avx2` CPU feature is available.
#[target_feature(enable = "avx2")]
unsafe fn aa_coverage_span_avx2(rows: [&[u8]; 4], x0: usize, shape: &mut [u8]) {
    use std::arch::x86_64::{
        _mm256_add_epi8, _mm256_and_si256, _mm256_loadu_si256, _mm256_set_epi8, _mm256_set1_epi8,
        _mm256_setzero_si256, _mm256_shuffle_epi8, _mm256_srli_epi16, _mm256_storeu_si256,
    };

    debug_assert!(x0 & 1 == 0, "aa_coverage_span_avx2: x0={x0} must be even");

    let n = shape.len();
    let (byte_x0, n_chunks) = coverage_chunk_params(x0, n, 32);

    // Nibble popcount LUT broadcast to both 128-bit lanes.
    let lut = _mm256_set_epi8(
        4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0, 4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1,
        1, 0,
    );
    let mask_lo = _mm256_set1_epi8(0x0F_u8.cast_signed());

    for chunk_idx in 0..n_chunks {
        let byte_off = byte_x0 + chunk_idx * 32;

        let (mut acc_hi, mut acc_lo) = (_mm256_setzero_si256(), _mm256_setzero_si256());

        for row in rows {
            assert!(
                byte_off + 32 <= row.len(),
                "aa_coverage_span_avx2: row too short: \
                 need {} bytes at offset {byte_off}, have {}",
                byte_off + 32,
                row.len(),
            );
            // SAFETY: byte_off + 32 ≤ row.len() asserted above.
            let v = unsafe { _mm256_loadu_si256(row[byte_off..].as_ptr().cast()) };
            // High nibble → bits 3:0 via right-shift, then mask.
            let hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), mask_lo);
            // Low nibble → bits 3:0 directly.
            let lo = _mm256_and_si256(v, mask_lo);
            // VPSHUFB lookup: nibble → popcount(nibble).
            acc_hi = _mm256_add_epi8(acc_hi, _mm256_shuffle_epi8(lut, hi));
            acc_lo = _mm256_add_epi8(acc_lo, _mm256_shuffle_epi8(lut, lo));
        }

        // Write the 32-element hi and lo vectors to a staging buffer, then
        // interleave into shape: shape[out_base + 2k] = hi[k], shape[out_base + 2k+1] = lo[k].
        let mut hi_buf = [0u8; 32];
        let mut lo_buf = [0u8; 32];
        // SAFETY: hi_buf / lo_buf are exactly 32 bytes.
        unsafe {
            _mm256_storeu_si256(hi_buf.as_mut_ptr().cast(), acc_hi);
            _mm256_storeu_si256(lo_buf.as_mut_ptr().cast(), acc_lo);
        }

        let out_base = chunk_idx * 64;
        for k in 0..32 {
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

    // Scalar remainder for any output pixels not covered by complete 32-byte chunks.
    let scalar_start = n_chunks * 64;
    if scalar_start < n {
        aa_coverage_span_scalar(rows, x0 + scalar_start, &mut shape[scalar_start..]);
    }
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

/// Fill `shape[i]` with the AA coverage count (0..=16) for output pixel `x0 + i`.
///
/// `rows` are the four `AaBuf` sub-row byte slices, each of length
/// `(bitmap_width * AA_SIZE + 7) / 8`.
///
/// # Preconditions
///
/// - `x0 + shape.len() ≤ bitmap_width` — span must not exceed the bitmap row.
/// - `x0` should be **even** for the SIMD tiers to be used.  An odd `x0` is
///   correctly handled by the scalar tier (full precision, lower throughput).
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
    } else if is_x86_feature_detected!("avx2") {
        // SAFETY: avx2 confirmed present.
        unsafe { aa_coverage_span_avx2(rows, x0, shape) };
    } else {
        aa_coverage_span_scalar(rows, x0, shape);
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn dispatch_coverage(rows: [&[u8]; 4], x0: usize, shape: &mut [u8]) {
    if x0 & 1 != 0 {
        // Odd x0: nibble boundaries are byte-aligned; SIMD paths require even x0.
        aa_coverage_span_scalar(rows, x0, shape);
        return;
    }
    #[cfg(feature = "nightly-sve2")]
    if std::arch::is_aarch64_feature_detected!("sve2") {
        // SAFETY: sve2 feature confirmed present.
        unsafe { aa_coverage_span_sve2(rows, x0, shape) };
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
        let r0: Vec<u8> = (0..row_bytes)
            .map(|i| (i as u8).wrapping_mul(0x37))
            .collect();
        let r1: Vec<u8> = (0..row_bytes)
            .map(|i| (i as u8).wrapping_mul(0x53))
            .collect();
        let r2: Vec<u8> = (0..row_bytes)
            .map(|i| (i as u8).wrapping_mul(0x17))
            .collect();
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
        let r0: Vec<u8> = (0..row_bytes)
            .map(|i| (i as u8).wrapping_mul(0x37))
            .collect();
        let r1: Vec<u8> = (0..row_bytes)
            .map(|i| (i as u8).wrapping_mul(0x53))
            .collect();
        let r2: Vec<u8> = (0..row_bytes)
            .map(|i| (i as u8).wrapping_mul(0x17))
            .collect();
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
        let r0: Vec<u8> = (0..row_bytes)
            .map(|i| (i as u8).wrapping_mul(37).wrapping_add(11))
            .collect();
        let r1: Vec<u8> = (0..row_bytes)
            .map(|i| (i as u8).wrapping_mul(53).wrapping_add(7))
            .collect();
        let r2: Vec<u8> = (0..row_bytes)
            .map(|i| (i as u8).wrapping_mul(17).wrapping_add(3))
            .collect();
        let r3: Vec<u8> = (0..row_bytes).map(|i| !(i as u8)).collect();

        let mut expected = vec![0u8; N];
        aa_coverage_span_scalar([&r0, &r1, &r2, &r3], 0, &mut expected);

        let mut got = vec![0u8; N];
        // SAFETY: both features confirmed present above.
        unsafe { aa_coverage_span_avx512([&r0, &r1, &r2, &r3], 0, &mut got) };

        assert_eq!(got, expected, "AVX-512 coverage mismatch vs scalar");
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_coverage_matches_scalar() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        // 300 output pixels: 4 full 32-byte chunks (128 px each) + 44-byte scalar remainder
        // ... actually 300 / 64 = 4 full chunks (256 px) + 44-px scalar remainder.
        const N: usize = 300;
        let row_bytes = N.div_ceil(2);
        let r0: Vec<u8> = (0..row_bytes)
            .map(|i| (i as u8).wrapping_mul(37).wrapping_add(11))
            .collect();
        let r1: Vec<u8> = (0..row_bytes)
            .map(|i| (i as u8).wrapping_mul(53).wrapping_add(7))
            .collect();
        let r2: Vec<u8> = (0..row_bytes)
            .map(|i| (i as u8).wrapping_mul(17).wrapping_add(3))
            .collect();
        let r3: Vec<u8> = (0..row_bytes).map(|i| !(i as u8)).collect();

        let mut expected = vec![0u8; N];
        aa_coverage_span_scalar([&r0, &r1, &r2, &r3], 0, &mut expected);

        let mut got = vec![0u8; N];
        // SAFETY: avx2 confirmed present above.
        unsafe { aa_coverage_span_avx2([&r0, &r1, &r2, &r3], 0, &mut got) };

        assert_eq!(got, expected, "AVX2 coverage mismatch vs scalar");
    }

    #[cfg(all(target_arch = "aarch64", feature = "nightly-sve2"))]
    #[test]
    fn sve2_coverage_matches_scalar() {
        if !std::arch::is_aarch64_feature_detected!("sve2") {
            return;
        }
        const N: usize = 300;
        let row_bytes = N.div_ceil(2);
        let r0: Vec<u8> = (0..row_bytes)
            .map(|i| (i as u8).wrapping_mul(37).wrapping_add(11))
            .collect();
        let r1: Vec<u8> = (0..row_bytes)
            .map(|i| (i as u8).wrapping_mul(53).wrapping_add(7))
            .collect();
        let r2: Vec<u8> = (0..row_bytes)
            .map(|i| (i as u8).wrapping_mul(17).wrapping_add(3))
            .collect();
        let r3: Vec<u8> = (0..row_bytes).map(|i| !(i as u8)).collect();

        let mut expected = vec![0u8; N];
        aa_coverage_span_scalar([&r0, &r1, &r2, &r3], 0, &mut expected);

        let mut got = vec![0u8; N];
        // SAFETY: sve2 confirmed present above.
        unsafe { aa_coverage_span_sve2([&r0, &r1, &r2, &r3], 0, &mut got) };

        assert_eq!(got, expected, "SVE2 coverage mismatch vs scalar");
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_coverage_matches_scalar() {
        // 300 output pixels — 9 full 16-byte NEON chunks + 6-byte scalar remainder.
        const N: usize = 300;
        let row_bytes = N.div_ceil(2);
        let r0: Vec<u8> = (0..row_bytes)
            .map(|i| (i as u8).wrapping_mul(37).wrapping_add(11))
            .collect();
        let r1: Vec<u8> = (0..row_bytes)
            .map(|i| (i as u8).wrapping_mul(53).wrapping_add(7))
            .collect();
        let r2: Vec<u8> = (0..row_bytes)
            .map(|i| (i as u8).wrapping_mul(17).wrapping_add(3))
            .collect();
        let r3: Vec<u8> = (0..row_bytes).map(|i| !(i as u8)).collect();

        let mut expected = vec![0u8; N];
        aa_coverage_span_scalar([&r0, &r1, &r2, &r3], 0, &mut expected);

        let mut got = vec![0u8; N];
        // SAFETY: aarch64 always has NEON.
        unsafe { aa_coverage_span_neon([&r0, &r1, &r2, &r3], 0, &mut got) };

        assert_eq!(got, expected, "NEON coverage mismatch vs scalar");
    }
}
