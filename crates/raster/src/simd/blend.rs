//! SIMD-accelerated solid-colour fill paths.
//!
//! `blend_solid_rgb8` — fill RGB pixels with a constant colour.
//! `blend_solid_gray8` — fill grayscale pixels with a constant value.
//!
//! # Acceleration tiers (x86-64, most to least preferred for large spans)
//!
//! 1. **`movdir64b`** (> 256 px): non-temporal 64-byte atomic stores that bypass
//!    all cache levels.  Used for large write-only fills so the L3 V-Cache is not
//!    evicted of the edge table that the scanner keeps hot between page renders.
//! 2. **AVX2** (≥ 32 px): 256-bit stores; fast for medium spans where the data
//!    will be read back shortly after writing.
//! 3. **Scalar**: `copy_from_slice` / `fill` per pixel.
//!
//! The `movdir64b` path requires a 64-byte-aligned destination address.  A scalar
//! preamble advances the write pointer to the next alignment boundary; a scalar
//! tail handles any remaining bytes.  Because `movdir64b` is not yet exposed in
//! `std::arch::x86_64`, runtime detection uses a `std::sync::OnceLock` that
//! queries CPUID leaf 7 subleaf 0 ECX bit 28 exactly once per process.

/// Fill `count` RGB pixels in `dst` with `color` using a scalar loop.
#[inline]
pub(super) fn blend_solid_rgb8_scalar(dst: &mut [u8], color: [u8; 3], count: usize) {
    debug_assert!(
        dst.len() >= count * 3,
        "dst too short: {} < {}",
        dst.len(),
        count * 3
    );
    for chunk in dst[..count * 3].chunks_exact_mut(3) {
        chunk.copy_from_slice(&color);
    }
}

/// Fill `count` grayscale pixels in `dst` with `color`.
#[inline]
pub(super) fn blend_solid_gray8_scalar(dst: &mut [u8], color: u8, count: usize) {
    debug_assert!(
        dst.len() >= count,
        "dst too short: {} < {}",
        dst.len(),
        count
    );
    dst[..count].fill(color);
}

// ── AVX2 paths ────────────────────────────────────────────────────────────────

// AVX2 functions use unsafe SIMD intrinsics — required, not lazy.
#[cfg(all(target_arch = "x86_64", feature = "simd-avx2"))]
/// Fill `count` RGB pixels in `dst` with `color` using 96-byte AVX2 chunks.
///
/// # Safety
///
/// Must only be called when AVX2 is available (`is_x86_feature_detected!("avx2")`).
/// `dst.len() >= count * 3` must hold.
#[target_feature(enable = "avx2")]
unsafe fn blend_solid_rgb8_avx2(dst: &mut [u8], color: [u8; 3], count: usize) {
    use std::arch::x86_64::{__m256i, _mm256_loadu_si256, _mm256_storeu_si256};
    debug_assert!(
        dst.len() >= count * 3,
        "dst too short for AVX2 RGB fill: {} < {}",
        dst.len(),
        count * 3
    );

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
    // SAFETY: tile is 96 bytes on the stack; all three pointers are in-bounds.
    // dst_ptr..dst_ptr + count*3 ≤ dst.len() by the debug_assert above.
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
/// `dst.len() >= count` must hold.
#[target_feature(enable = "avx2")]
unsafe fn blend_solid_gray8_avx2(dst: &mut [u8], color: u8, count: usize) {
    use std::arch::x86_64::{_mm256_set1_epi8, _mm256_storeu_si256};
    debug_assert!(
        dst.len() >= count,
        "dst too short for AVX2 gray fill: {} < {}",
        dst.len(),
        count
    );

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

// ── movdir64b non-temporal fill ───────────────────────────────────────────────
//
// `movdir64b` is not yet exposed in `std::arch::x86_64`; we use inline asm.
// Detection uses CPUID leaf 7, subleaf 0, ECX bit 28, cached in a OnceLock.

#[cfg(target_arch = "x86_64")]
/// Pixel threshold above which `movdir64b` is preferred over AVX2 for solid fills.
///
/// Below this threshold the destination span is likely to be read back soon
/// (e.g. compositing, AA blending) and keeping it in L3 is beneficial.
/// Above it the output is write-only until the next page, so a non-temporal
/// store preserves the edge table's V-Cache residency.
const MOVDIR64B_THRESHOLD_PX: usize = 256;

#[cfg(target_arch = "x86_64")]
/// Query CPUID leaf 7, subleaf 0, ECX bit 28 to detect `movdir64b`.
///
/// Uses `std::arch::x86_64::__cpuid_count` (stable since 1.27) which handles
/// the rbx save/restore internally on all supported platforms.
/// Result is cached in a `OnceLock` so CPUID is executed at most once per process.
fn has_movdir64b() -> bool {
    use std::sync::OnceLock;
    static CACHE: OnceLock<bool> = OnceLock::new();
    *CACHE.get_or_init(|| {
        // leaf 7 / subleaf 0: on CPUs that don't support leaf 7 this returns
        // all-zeros, which correctly produces `false`.
        let result = std::arch::x86_64::__cpuid_count(7, 0);
        // ECX bit 28 = MOVDIR64B (Intel SDM vol. 2A, CPUID.07H.00H:ECX[28])
        (result.ecx >> 28) & 1 != 0
    })
}

/// Compute how many leading bytes are needed to bring `ptr` to `align`-byte
/// alignment, capped at `limit`.
///
/// `ptr::align_offset` returns `usize::MAX` for null pointers or when
/// alignment is provably impossible; in both cases we return `limit` so the
/// caller falls back to its scalar path for the entire span.
#[cfg(target_arch = "x86_64")]
#[inline]
fn preamble_len(ptr: *const u8, limit: usize, align: usize) -> usize {
    let off = ptr.align_offset(align);
    if off == usize::MAX { limit } else { off.min(limit) }
}

#[cfg(target_arch = "x86_64")]
/// Fill `count` RGB pixels in `dst` using `movdir64b` non-temporal 64-byte stores.
///
/// The tile is 192 bytes (`LCM(3, 64) = 192`): three consecutive `movdir64b`
/// stores advance the destination by 192 bytes (64 pixels) while reading from
/// three disjoint 64-byte sections of the tile.  This ensures every store
/// writes exactly one full RGB pixel at every byte boundary within the tile.
///
/// A scalar preamble aligns the destination pointer to 64 bytes; a scalar tail
/// handles any remaining bytes after the last aligned block.
///
/// # Safety
///
/// - `movdir64b` must be available (CPUID.07H.00H:ECX[28] = 1).
/// - `dst.len() >= count * 3` must hold.
unsafe fn blend_solid_rgb8_movdir64b(dst: &mut [u8], color: [u8; 3], count: usize) {
    // 192 = LCM(3, 64): three 64-byte stores cover exactly 64 RGB pixels
    // with no partial pixel at any 64-byte boundary.
    #[repr(align(64))]
    struct Tile([u8; 192]);

    debug_assert!(
        dst.len() >= count * 3,
        "dst too short for movdir64b RGB fill: {} < {}",
        dst.len(),
        count * 3,
    );

    let byte_count = count * 3;
    let dst_ptr = dst.as_mut_ptr();

    // ── Preamble ──────────────────────────────────────────────────────────────
    // Fill bytes [0, preamble) so that dst_ptr + preamble is 64-byte aligned.
    // Each byte is indexed by its absolute position mod 3 so the RGB pattern
    // remains continuous across the preamble / block boundary.
    let preamble = preamble_len(dst_ptr.cast_const(), byte_count, 64);
    for i in 0..preamble {
        dst[i] = color[i % 3];
    }

    // ── Aligned blocks ────────────────────────────────────────────────────────
    // Build a 192-byte tile phase-shifted by `preamble % 3` so that
    // tile[k] == color[(preamble + k) % 3], i.e. the correct channel for
    // absolute destination byte (preamble + k).
    let phase = preamble % 3;
    let mut tile = Tile([0u8; 192]);
    for (k, t) in tile.0.iter_mut().enumerate() {
        *t = color[(phase + k) % 3];
    }

    let blocks_start = preamble;
    debug_assert!(blocks_start <= byte_count, "preamble_len exceeded byte_count");
    let remaining = byte_count - blocks_start;
    let blocks = remaining / 192;

    for blk in 0..blocks {
        // SAFETY:
        // - blocks_start + blk*192 + 192 ≤ blocks_start + remaining ≤ byte_count ≤ dst.len().
        // - dst_ptr + blocks_start is 64-byte aligned: preamble_len guarantees this
        //   (or preamble == byte_count making blocks == 0, so the loop never runs).
        // - blocks_start + blk*192 is also 64-byte aligned (192 is a multiple of 64).
        // - tile.0 is 64-byte aligned by #[repr(align(64))]; sub-tiles at +64 and
        //   +128 are therefore also 64-byte aligned.
        // - MOVDIR64B destination operand is a register holding the aligned address;
        //   source operand is a memory reference [reg].
        unsafe {
            let dst_base = dst_ptr.add(blocks_start + blk * 192);
            let src0 = tile.0.as_ptr();
            let src1 = src0.add(64);
            let src2 = src0.add(128);
            std::arch::asm!(
                "movdir64b {d0}, [{s0}]",
                "movdir64b {d1}, [{s1}]",
                "movdir64b {d2}, [{s2}]",
                d0 = in(reg) dst_base,
                d1 = in(reg) dst_base.add(64),
                d2 = in(reg) dst_base.add(128),
                s0 = in(reg) src0,
                s1 = in(reg) src1,
                s2 = in(reg) src2,
                options(nostack, preserves_flags),
            );
        }
    }

    // ── Tail ──────────────────────────────────────────────────────────────────
    // Bytes after the last complete 192-byte block.  `off - blocks_start` is
    // the offset from the first block byte, so adding `phase` gives the correct
    // channel index consistent with the tile.
    let tail_start = blocks_start + blocks * 192;
    for off in tail_start..byte_count {
        dst[off] = color[(phase + (off - blocks_start)) % 3];
    }
}

#[cfg(target_arch = "x86_64")]
/// Fill `count` grayscale pixels in `dst` using `movdir64b` non-temporal stores.
///
/// Each `movdir64b` writes 64 bytes = 64 pixels.  A scalar preamble aligns the
/// destination to 64 bytes; a scalar tail handles the remainder.
///
/// # Safety
///
/// - `movdir64b` must be available (CPUID.07H.00H:ECX[28] = 1).
/// - `dst.len() >= count` must hold.
unsafe fn blend_solid_gray8_movdir64b(dst: &mut [u8], color: u8, count: usize) {
    #[repr(align(64))]
    struct Tile([u8; 64]);

    debug_assert!(
        dst.len() >= count,
        "dst too short for movdir64b gray fill: {} < {}",
        dst.len(),
        count,
    );

    let tile = Tile([color; 64]);
    let dst_ptr = dst.as_mut_ptr();

    // ── Preamble ──────────────────────────────────────────────────────────────
    let preamble = preamble_len(dst_ptr.cast_const(), count, 64);
    dst[..preamble].fill(color);

    // ── Aligned blocks ────────────────────────────────────────────────────────
    debug_assert!(preamble <= count, "preamble_len exceeded count");
    let blocks = (count - preamble) / 64;
    for blk in 0..blocks {
        // SAFETY:
        // - preamble + blk*64 + 64 ≤ preamble + (count - preamble) = count ≤ dst.len().
        // - dst_ptr + preamble is 64-byte aligned (preamble_len guarantees this,
        //   or preamble == count making blocks == 0).
        // - tile.0 is 64-byte aligned by #[repr(align(64))].
        // - MOVDIR64B destination operand is a register holding the aligned address.
        unsafe {
            let dst_blk = dst_ptr.add(preamble + blk * 64);
            let src = tile.0.as_ptr();
            std::arch::asm!(
                "movdir64b {dst}, [{src}]",
                dst = in(reg) dst_blk,
                src = in(reg) src,
                options(nostack, preserves_flags),
            );
        }
    }

    // ── Tail ──────────────────────────────────────────────────────────────────
    let tail_start = preamble + blocks * 64;
    dst[tail_start..count].fill(color);
}

// ── Public dispatch functions ─────────────────────────────────────────────────

/// Fill `count` RGB pixels in `dst` (starting at byte 0) with `color`.
///
/// # Panics
///
/// Panics if `dst.len() < count * 3`.
///
/// # Dispatch order (x86-64)
///
/// 1. `movdir64b` non-temporal stores when `count > 256` and the CPU supports it —
///    preserves L3 V-Cache residency for the scanner edge table.
/// 2. AVX2 256-bit stores when `count >= 32`.
/// 3. Scalar `copy_from_slice` loop.
pub fn blend_solid_rgb8(dst: &mut [u8], color: [u8; 3], count: usize) {
    assert!(
        dst.len() >= count * 3,
        "blend_solid_rgb8: dst too short ({} < {})",
        dst.len(),
        count * 3,
    );

    #[cfg(target_arch = "x86_64")]
    if count > MOVDIR64B_THRESHOLD_PX && has_movdir64b() {
        // SAFETY: movdir64b confirmed by has_movdir64b(); dst.len() >= count*3 asserted above.
        unsafe { blend_solid_rgb8_movdir64b(dst, color, count) };
        return;
    }

    #[cfg(all(target_arch = "x86_64", feature = "simd-avx2"))]
    if is_x86_feature_detected!("avx2") && count >= 32 {
        // SAFETY: AVX2 confirmed by is_x86_feature_detected!; dst.len() >= count*3 asserted above.
        unsafe { blend_solid_rgb8_avx2(dst, color, count) };
        return;
    }

    blend_solid_rgb8_scalar(dst, color, count);
}

/// Fill `count` grayscale pixels in `dst` (starting at byte 0) with `color`.
///
/// # Panics
///
/// Panics if `dst.len() < count`.
///
/// # Dispatch order (x86-64)
///
/// 1. `movdir64b` non-temporal stores when `count > 256` and the CPU supports it.
/// 2. AVX2 256-bit stores when `count >= 32`.
/// 3. Scalar `fill`.
pub fn blend_solid_gray8(dst: &mut [u8], color: u8, count: usize) {
    assert!(
        dst.len() >= count,
        "blend_solid_gray8: dst too short ({} < {})",
        dst.len(),
        count,
    );

    #[cfg(target_arch = "x86_64")]
    if count > MOVDIR64B_THRESHOLD_PX && has_movdir64b() {
        // SAFETY: movdir64b confirmed by has_movdir64b(); dst.len() >= count asserted above.
        unsafe { blend_solid_gray8_movdir64b(dst, color, count) };
        return;
    }

    #[cfg(all(target_arch = "x86_64", feature = "simd-avx2"))]
    if is_x86_feature_detected!("avx2") && count >= 32 {
        // SAFETY: AVX2 confirmed by is_x86_feature_detected!; dst.len() >= count asserted above.
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
        // SAFETY: is_x86_feature_detected! confirmed AVX2 is available.
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
        // SAFETY: is_x86_feature_detected! confirmed AVX2 is available.
        unsafe { blend_solid_gray8_avx2(&mut got, 200, count) };
        assert_eq!(got, expected, "AVX2 gray path mismatch");
    }

    // ── movdir64b tests ───────────────────────────────────────────────────────

    /// `dispatch_rgb8_large` exercises count > MOVDIR64B_THRESHOLD_PX so the
    /// movdir64b path (or AVX2 on machines without movdir64b) is selected.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn dispatch_rgb8_large_matches_scalar() {
        let color = [77u8, 133, 211];
        // 384 pixels: well above the 256-px threshold; exercises multiple 192-byte blocks.
        let count = 384usize;
        let mut expected = vec![0u8; count * 3];
        blend_solid_rgb8_scalar(&mut expected, color, count);

        let mut got = vec![0u8; count * 3];
        blend_solid_rgb8(&mut got, color, count);
        assert_eq!(got, expected, "large RGB dispatch mismatch");
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn dispatch_gray8_large_matches_scalar() {
        let count = 512usize;
        let mut expected = vec![0u8; count];
        blend_solid_gray8_scalar(&mut expected, 99, count);

        let mut got = vec![0u8; count];
        blend_solid_gray8(&mut got, 99, count);
        assert_eq!(got, expected, "large gray dispatch mismatch");
    }

    /// Exercise the movdir64b RGB path directly on capable machines.
    /// Uses a misaligned allocation (vec::as_mut_ptr is only 8-byte aligned by
    /// default) to exercise the preamble path as well.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn movdir64b_rgb8_matches_scalar() {
        if !has_movdir64b() {
            return;
        }
        let color = [11u8, 22, 33];
        // 512 pixels — exercises multiple 192-byte blocks plus preamble and tail.
        let count = 512usize;
        let mut expected = vec![0u8; count * 3];
        blend_solid_rgb8_scalar(&mut expected, color, count);

        let mut got = vec![0u8; count * 3];
        // SAFETY: has_movdir64b() confirmed CPUID.07H.00H:ECX[28] = 1.
        unsafe { blend_solid_rgb8_movdir64b(&mut got, color, count) };
        assert_eq!(got, expected, "movdir64b RGB mismatch");
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn movdir64b_gray8_matches_scalar() {
        if !has_movdir64b() {
            return;
        }
        let count = 512usize;
        let mut expected = vec![0u8; count];
        blend_solid_gray8_scalar(&mut expected, 200, count);

        let mut got = vec![0u8; count];
        // SAFETY: has_movdir64b() confirmed CPUID.07H.00H:ECX[28] = 1.
        unsafe { blend_solid_gray8_movdir64b(&mut got, 200, count) };
        assert_eq!(got, expected, "movdir64b gray mismatch");
    }

    /// Odd count exercises the tail path that handles non-block-multiple pixel counts.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn movdir64b_rgb8_odd_count() {
        if !has_movdir64b() {
            return;
        }
        let color = [3u8, 7, 11];
        let count = 257usize; // 257 px: one block of 64px + preamble & tail
        let mut expected = vec![0u8; count * 3];
        blend_solid_rgb8_scalar(&mut expected, color, count);

        let mut got = vec![0u8; count * 3];
        // SAFETY: has_movdir64b() confirmed CPUID.07H.00H:ECX[28] = 1.
        unsafe { blend_solid_rgb8_movdir64b(&mut got, color, count) };
        assert_eq!(got, expected, "movdir64b RGB odd-count mismatch");
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn movdir64b_gray8_odd_count() {
        if !has_movdir64b() {
            return;
        }
        let count = 259usize;
        let mut expected = vec![0u8; count];
        blend_solid_gray8_scalar(&mut expected, 17, count);

        let mut got = vec![0u8; count];
        // SAFETY: has_movdir64b() confirmed CPUID.07H.00H:ECX[28] = 1.
        unsafe { blend_solid_gray8_movdir64b(&mut got, 17, count) };
        assert_eq!(got, expected, "movdir64b gray odd-count mismatch");
    }

    // ── public API boundary checks ────────────────────────────────────────────

    #[test]
    #[should_panic(expected = "blend_solid_rgb8: dst too short")]
    fn rgb8_panics_on_short_dst() {
        let mut dst = vec![0u8; 5];
        blend_solid_rgb8(&mut dst, [1, 2, 3], 10);
    }

    #[test]
    #[should_panic(expected = "blend_solid_gray8: dst too short")]
    fn gray8_panics_on_short_dst() {
        let mut dst = vec![0u8; 5];
        blend_solid_gray8(&mut dst, 42, 10);
    }
}
