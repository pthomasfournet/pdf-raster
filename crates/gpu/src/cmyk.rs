//! CMYK→RGB CPU conversion paths (AVX-512, NEON, and scalar fallback).

use color::convert::cmyk_to_rgb_reflectance;

// ── AVX-512 CMYK→RGB ──────────────────────────────────────────────────────────
//
// Vectorised subtractive complement: R=(255−C)*(255−K)/255 (rounded).
// Processes 16 pixels per call using u16 arithmetic throughout.
//
// AoS→SoA: shuffle_epi8 gathers one channel from each 4-pixel 128-bit lane
// to bytes 0..3 of that lane (zeros elsewhere).  permute4x64(x, 0x88) selects
// epi64-lanes 0 and 2, giving [ch0..3 0000 ch4..7 0000] in 128 bits.  A final
// shuffle_epi8 compacts to [ch0..7 0×8]; unpacklo_epi64 joins two such halves
// (pixels 0..7 and 8..15) into 16 contiguous u8; cvtepu8_epi16 widens to u16.
//
// Division: exact ⌊(x+127)/255⌋ = (n + (n>>8) + 1) >> 8, n = x+127.
// Valid for n ∈ [0, 65152] (max n = 255²+127 = 65152 < 65280 = 255×256).

/// Convert 16 CMYK pixels to RGB using AVX-512 u16 arithmetic.
///
/// `cmyk` must be exactly 64 bytes (16 pixels × 4 channels).
/// `rgb` must be at least 48 bytes (16 pixels × 3 channels).
///
/// # Safety
///
/// Caller must ensure `avx512f` and `avx512bw` are available at runtime.
/// `cmyk.len() == 64` and `rgb.len() >= 48` must hold.
#[cfg(target_arch = "x86_64")]
#[expect(
    clippy::too_many_lines,
    reason = "SIMD shuffle/arithmetic pipeline — splitting would obscure the data flow"
)]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn cmyk_to_rgb_avx512(cmyk: &[u8; 64], rgb: &mut [u8]) {
    use std::arch::x86_64::{
        __m256i, __m512i, _mm_loadu_si128, _mm_shuffle_epi8, _mm_storeu_si128, _mm_unpacklo_epi64,
        _mm256_add_epi16, _mm256_castsi256_si128, _mm256_cvtepu8_epi16, _mm256_mullo_epi16,
        _mm256_packus_epi16, _mm256_permute4x64_epi64, _mm256_set1_epi16, _mm256_srli_epi16,
        _mm256_sub_epi16, _mm512_castsi512_si256, _mm512_extracti64x4_epi64, _mm512_loadu_si512,
        _mm512_shuffle_epi8,
    };

    debug_assert!(rgb.len() >= 48);

    unsafe {
        // Load all 16 CMYK pixels (64 bytes) into one 512-bit register.
        let raw: __m512i = _mm512_loadu_si512(cmyk.as_ptr().cast());

        // AoS→SoA via shuffle_epi8: permutes bytes within each 128-bit lane.
        // Each lane holds 4 CMYK pixels (16 bytes). The mask gathers one channel
        // to bytes 0..3 of the lane (zeros elsewhere), giving 4 lanes × 4 bytes =
        // 16 channel values spread across the 512-bit register.
        #[rustfmt::skip]
        let mask_c: [u8; 64] = [
            0, 4, 8,12, 0x80,0x80,0x80,0x80, 0x80,0x80,0x80,0x80, 0x80,0x80,0x80,0x80,
            0, 4, 8,12, 0x80,0x80,0x80,0x80, 0x80,0x80,0x80,0x80, 0x80,0x80,0x80,0x80,
            0, 4, 8,12, 0x80,0x80,0x80,0x80, 0x80,0x80,0x80,0x80, 0x80,0x80,0x80,0x80,
            0, 4, 8,12, 0x80,0x80,0x80,0x80, 0x80,0x80,0x80,0x80, 0x80,0x80,0x80,0x80,
        ];
        #[rustfmt::skip]
        let mask_m: [u8; 64] = [
            1, 5, 9,13, 0x80,0x80,0x80,0x80, 0x80,0x80,0x80,0x80, 0x80,0x80,0x80,0x80,
            1, 5, 9,13, 0x80,0x80,0x80,0x80, 0x80,0x80,0x80,0x80, 0x80,0x80,0x80,0x80,
            1, 5, 9,13, 0x80,0x80,0x80,0x80, 0x80,0x80,0x80,0x80, 0x80,0x80,0x80,0x80,
            1, 5, 9,13, 0x80,0x80,0x80,0x80, 0x80,0x80,0x80,0x80, 0x80,0x80,0x80,0x80,
        ];
        #[rustfmt::skip]
        let mask_y: [u8; 64] = [
            2, 6,10,14, 0x80,0x80,0x80,0x80, 0x80,0x80,0x80,0x80, 0x80,0x80,0x80,0x80,
            2, 6,10,14, 0x80,0x80,0x80,0x80, 0x80,0x80,0x80,0x80, 0x80,0x80,0x80,0x80,
            2, 6,10,14, 0x80,0x80,0x80,0x80, 0x80,0x80,0x80,0x80, 0x80,0x80,0x80,0x80,
            2, 6,10,14, 0x80,0x80,0x80,0x80, 0x80,0x80,0x80,0x80, 0x80,0x80,0x80,0x80,
        ];
        #[rustfmt::skip]
        let mask_k: [u8; 64] = [
            3, 7,11,15, 0x80,0x80,0x80,0x80, 0x80,0x80,0x80,0x80, 0x80,0x80,0x80,0x80,
            3, 7,11,15, 0x80,0x80,0x80,0x80, 0x80,0x80,0x80,0x80, 0x80,0x80,0x80,0x80,
            3, 7,11,15, 0x80,0x80,0x80,0x80, 0x80,0x80,0x80,0x80, 0x80,0x80,0x80,0x80,
            3, 7,11,15, 0x80,0x80,0x80,0x80, 0x80,0x80,0x80,0x80, 0x80,0x80,0x80,0x80,
        ];
        let shuf_c: __m512i = _mm512_loadu_si512(mask_c.as_ptr().cast());
        let shuf_m: __m512i = _mm512_loadu_si512(mask_m.as_ptr().cast());
        let shuf_y: __m512i = _mm512_loadu_si512(mask_y.as_ptr().cast());
        let shuf_k: __m512i = _mm512_loadu_si512(mask_k.as_ptr().cast());

        let c_bytes: __m512i = _mm512_shuffle_epi8(raw, shuf_c);
        let m_bytes: __m512i = _mm512_shuffle_epi8(raw, shuf_m);
        let y_bytes: __m512i = _mm512_shuffle_epi8(raw, shuf_y);
        let k_bytes: __m512i = _mm512_shuffle_epi8(raw, shuf_k);

        // After shuffle: each 128-bit lane has [ch0 ch1 ch2 ch3  0×12].
        // permute4x64(x, 0x88) selects epi64-lanes 0 and 2:
        //   low 128 = [ch0..3  0×4  ch4..7  0×4]  (two 4-byte groups, gaps between)
        // compact2 shuffle closes the gaps: bytes 0..3 and 8..11 → bytes 0..7.
        // unpacklo_epi64 joins the lo and hi halves → 16 contiguous u8.
        // cvtepu8_epi16 zero-extends to 16 u16.
        #[rustfmt::skip]
        let compact2: [u8; 16] = [
            0,1,2,3, 8,9,10,11, 0x80,0x80,0x80,0x80, 0x80,0x80,0x80,0x80,
        ];
        let compact2_128 = _mm_loadu_si128(compact2.as_ptr().cast());

        macro_rules! compact_to_u16 {
            ($v512:expr) => {{
                let lo128 = _mm_shuffle_epi8(
                    _mm256_castsi256_si128(_mm256_permute4x64_epi64(
                        _mm512_castsi512_si256($v512),
                        0x88_i32,
                    )),
                    compact2_128,
                );
                let hi128 = _mm_shuffle_epi8(
                    _mm256_castsi256_si128(_mm256_permute4x64_epi64(
                        _mm512_extracti64x4_epi64($v512, 1),
                        0x88_i32,
                    )),
                    compact2_128,
                );
                _mm256_cvtepu8_epi16(_mm_unpacklo_epi64(lo128, hi128))
            }};
        }

        let c_u16: __m256i = compact_to_u16!(c_bytes);
        let m_u16: __m256i = compact_to_u16!(m_bytes);
        let y_u16: __m256i = compact_to_u16!(y_bytes);
        let k_u16: __m256i = compact_to_u16!(k_bytes);

        // inv_ch = 255 − ch  (u16, max 255, no overflow)
        let v255 = _mm256_set1_epi16(255_i16);
        let inv_c = _mm256_sub_epi16(v255, c_u16);
        let inv_m = _mm256_sub_epi16(v255, m_u16);
        let inv_y = _mm256_sub_epi16(v255, y_u16);
        let inv_k = _mm256_sub_epi16(v255, k_u16);

        // prod = inv_ch * inv_k  (u16, max 255*255 = 65025 < 65536, no truncation)
        let prod_r = _mm256_mullo_epi16(inv_c, inv_k);
        let prod_g = _mm256_mullo_epi16(inv_m, inv_k);
        let prod_b = _mm256_mullo_epi16(inv_y, inv_k);

        // Exact ⌊(x + 127) / 255⌋ matching the scalar formula (255-c)*(255-k)+127)/255.
        // For n = x + 127 ∈ [0, 65152]: ⌊n/255⌋ = (n + (n>>8) + 1) >> 8.
        // Proof: let n = 255*q + r, 0 ≤ r < 255.
        //   n>>8 = ⌊n/256⌋ = q - ⌊(256q-n)/256⌋ = q - ⌊(256r - r·1)/256⌋ ≈ q (error < 1).
        //   More precisely (n + (n>>8) + 1)>>8 = ⌊(n + ⌊n/256⌋ + 1)/256⌋.
        //   For n < 65280 (= 255*256) this equals q = ⌊n/255⌋ exactly.
        //   max n = 65152 < 65280 ✓.
        let v127 = _mm256_set1_epi16(127_i16);
        let v1 = _mm256_set1_epi16(1_i16);
        macro_rules! div255 {
            ($x:expr) => {{
                let n = _mm256_add_epi16($x, v127); // n = x + 127
                _mm256_srli_epi16(
                    _mm256_add_epi16(_mm256_add_epi16(n, _mm256_srli_epi16(n, 8)), v1),
                    8,
                )
            }};
        }
        let r_u16 = div255!(prod_r);
        let g_u16 = div255!(prod_g);
        let b_u16 = div255!(prod_b);

        // Narrow u16 → u8 (values ≤ 255; saturation never triggers).
        // packus_epi16(v, v) → [v0..v7 v0..v7 | v8..v15 v8..v15] per 256-bit.
        // permute4x64(x, 0x88) packs the two unique halves to the low 128 bits.
        let r8 = _mm256_castsi256_si128(_mm256_permute4x64_epi64(
            _mm256_packus_epi16(r_u16, r_u16),
            0x88_i32,
        ));
        let g8 = _mm256_castsi256_si128(_mm256_permute4x64_epi64(
            _mm256_packus_epi16(g_u16, g_u16),
            0x88_i32,
        ));
        let b8 = _mm256_castsi256_si128(_mm256_permute4x64_epi64(
            _mm256_packus_epi16(b_u16, b_u16),
            0x88_i32,
        ));

        // Scatter R, G, B to interleaved RGB output via three scalar stores.
        // The vectorised multiply+divide above provides the bulk of the speedup;
        // this store loop is cheap (16 iterations, compiler unrolls it).
        let mut r_arr = [0u8; 16];
        let mut g_arr = [0u8; 16];
        let mut b_arr = [0u8; 16];
        _mm_storeu_si128(r_arr.as_mut_ptr().cast(), r8);
        _mm_storeu_si128(g_arr.as_mut_ptr().cast(), g8);
        _mm_storeu_si128(b_arr.as_mut_ptr().cast(), b8);
        for i in 0..16 {
            rgb[i * 3] = r_arr[i];
            rgb[i * 3 + 1] = g_arr[i];
            rgb[i * 3 + 2] = b_arr[i];
        }
    }
}

// ── ARM NEON CMYK→RGB ─────────────────────────────────────────────────────────
//
// Vectorised subtractive complement: R=(255−C)*(255−K)/255 (rounded).
// Processes 8 pixels per call using u16 arithmetic throughout.
//
// vmull_u8: u8×u8 → u16, widens each lane pair. Takes inv_ch (uint8x8_t) and
// inv_k (uint8x8_t) and produces uint16x8_t of products.
// vshrn_n_u16: narrow-right-shift by 8; combined with the +127 rounding bias
// gives ⌊(prod+127)/255⌋ ≈ ⌊prod/255⌋ rounded. We use the two-step identity:
//   n = prod + 127
//   result = (n + (n >> 8) + 1) >> 8
// which is exact for n ≤ 65152. The u16 additions cannot overflow because
// prod ≤ 255² = 65025, n ≤ 65152, and the intermediate sums are ≤ 65407 < 65536.
//
// Store: vst3q_u8 interleaves three uint8x8_t registers into 24 bytes of RGB.

/// Convert 8 CMYK pixels to RGB using NEON u16 widening arithmetic.
///
/// `cmyk` must be exactly 32 bytes (8 pixels × 4 channels).
/// `rgb` must be at least 24 bytes (8 pixels × 3 channels).
///
/// # Safety
///
/// Caller must ensure `neon` is available (mandatory on all ARMv8-A targets).
/// `cmyk.len() == 32` and `rgb.len() >= 24` must hold.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn cmyk_to_rgb_neon(cmyk: &[u8; 32], rgb: &mut [u8]) {
    use std::arch::aarch64::{
        uint8x8_t, uint8x8x3_t, uint16x8_t, vaddq_u16, vdup_n_u8, vld1_u8, vmull_u8, vshrn_n_u16,
        vsraq_n_u16, vst3_u8, vsub_u8,
    };

    debug_assert!(rgb.len() >= 24);

    // Load 8 CMYK pixels (32 bytes) and deinterleave channels via scalar gather,
    // then load each 8-element array into a NEON register. The bottleneck is
    // multiply, not load; scalar gather at 8px/iter is negligible overhead.
    let mut c_arr = [0u8; 8];
    let mut m_arr = [0u8; 8];
    let mut y_arr = [0u8; 8];
    let mut k_arr = [0u8; 8];
    for i in 0..8 {
        c_arr[i] = cmyk[i * 4];
        m_arr[i] = cmyk[i * 4 + 1];
        y_arr[i] = cmyk[i * 4 + 2];
        k_arr[i] = cmyk[i * 4 + 3];
    }

    // vld1_u8 is the 64-bit (8-byte) load; vld1q_u8 would read 16 bytes from
    // an 8-byte array, which is UB.  vdup_n_u8 avoids the load entirely for
    // the constant 255 vector.
    // SAFETY: each array is exactly 8 bytes; vld1_u8 requires only 1-byte alignment.
    let v255: uint8x8_t = vdup_n_u8(255);
    let c_v: uint8x8_t = unsafe { vld1_u8(c_arr.as_ptr()) };
    let m_v: uint8x8_t = unsafe { vld1_u8(m_arr.as_ptr()) };
    let y_v: uint8x8_t = unsafe { vld1_u8(y_arr.as_ptr()) };
    let k_v: uint8x8_t = unsafe { vld1_u8(k_arr.as_ptr()) };

    // inv_ch = 255 − ch. vsub_u8 wraps on underflow; ch ≤ 255 so no underflow.
    let inv_c: uint8x8_t = vsub_u8(v255, c_v);
    let inv_m: uint8x8_t = vsub_u8(v255, m_v);
    let inv_y: uint8x8_t = vsub_u8(v255, y_v);
    let inv_k: uint8x8_t = vsub_u8(v255, k_v);

    // prod = inv_ch * inv_k, widened to u16. max = 255*255 = 65025 < 65536 ✓.
    let prod_r: uint16x8_t = vmull_u8(inv_c, inv_k);
    let prod_g: uint16x8_t = vmull_u8(inv_m, inv_k);
    let prod_b: uint16x8_t = vmull_u8(inv_y, inv_k);

    // Exact ⌊(x + 127) / 255⌋ = (n + (n>>8) + 1) >> 8, n = x + 127.
    // vsraq_n_u16(a, b, n) = a + (b >> n); gives n + (n>>8) in one instruction.
    let v127: uint16x8_t = unsafe {
        let a = [127u16; 8];
        std::arch::aarch64::vld1q_u16(a.as_ptr())
    };
    let v1: uint16x8_t = unsafe {
        let a = [1u16; 8];
        std::arch::aarch64::vld1q_u16(a.as_ptr())
    };

    macro_rules! div255 {
        ($prod:expr) => {{
            let n: uint16x8_t = vaddq_u16($prod, v127);
            let shifted: uint16x8_t = vsraq_n_u16(n, n, 8);
            let rounded: uint16x8_t = vaddq_u16(shifted, v1);
            vshrn_n_u16(rounded, 8)
        }};
    }

    let r8: uint8x8_t = div255!(prod_r);
    let g8: uint8x8_t = div255!(prod_g);
    let b8: uint8x8_t = div255!(prod_b);

    // vst3_u8 interleaves three uint8x8_t into 24 bytes of packed RGB.
    let out = uint8x8x3_t(r8, g8, b8);
    // SAFETY: rgb.len() >= 24 (debug_assert above); pointer is valid.
    unsafe { vst3_u8(rgb.as_mut_ptr(), out) };
}

// ── Scalar CMYK→RGB ───────────────────────────────────────────────────────────

#[inline(always)]
fn cmyk_to_rgb_scalar(cmyk: &[u8], rgb: &mut [u8]) {
    for (src, dst) in cmyk.chunks_exact(4).zip(rgb.chunks_exact_mut(3)) {
        let (r, g, b) = cmyk_to_rgb_reflectance(src[0], src[1], src[2], src[3]);
        dst[0] = r;
        dst[1] = g;
        dst[2] = b;
    }
}

// ── Per-arch dispatch ─────────────────────────────────────────────────────────

/// x86-64: runtime-detect AVX-512; scalar fallback when unavailable.
///
/// Uses `is_x86_feature_detected!` so the same binary works on both
/// AVX-512 machines (Ryzen 9000, Sapphire Rapids) and consumer Intel without
/// AVX-512.  A compile-time `#[cfg(target_feature)]` gate would compile out
/// the scalar path on native builds, causing SIGILL on non-AVX-512 hosts.
#[cfg(target_arch = "x86_64")]
#[inline]
fn dispatch_cmyk_matrix(cmyk: &[u8], rgb: &mut [u8]) {
    if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
        let mut chunks = cmyk.chunks_exact(64);
        let mut out_off = 0usize;
        for chunk in chunks.by_ref() {
            // SAFETY: avx512f+avx512bw confirmed by is_x86_feature_detected! above;
            // chunk is exactly 64 bytes; rgb[out_off..] has ≥ 48 bytes remaining
            // because rgb was sized to n*3 and we advance by 48 per 64-byte input.
            unsafe {
                cmyk_to_rgb_avx512(
                    chunk.try_into().expect("chunk is exactly 64 bytes"),
                    &mut rgb[out_off..],
                );
            }
            out_off += 48;
        }
        // Scalar tail for remaining pixels (< 16).
        cmyk_to_rgb_scalar(chunks.remainder(), &mut rgb[out_off..]);
    } else {
        cmyk_to_rgb_scalar(cmyk, rgb);
    }
}

/// aarch64: NEON is mandatory on all ARMv8-A targets; no runtime detection needed.
#[cfg(target_arch = "aarch64")]
#[inline]
fn dispatch_cmyk_matrix(cmyk: &[u8], rgb: &mut [u8]) {
    let mut chunks = cmyk.chunks_exact(32);
    let mut out_off = 0usize;
    for chunk in chunks.by_ref() {
        // SAFETY: NEON is mandatory on aarch64; chunk is exactly 32 bytes;
        // rgb[out_off..] has ≥ 24 bytes remaining.
        unsafe {
            cmyk_to_rgb_neon(
                chunk.try_into().expect("chunk is exactly 32 bytes"),
                &mut rgb[out_off..],
            );
        }
        out_off += 24;
    }
    // Scalar tail for remaining pixels (< 8).
    cmyk_to_rgb_scalar(chunks.remainder(), &mut rgb[out_off..]);
}

/// Generic fallback for targets without SIMD specialisation.
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
#[inline]
fn dispatch_cmyk_matrix(cmyk: &[u8], rgb: &mut [u8]) {
    cmyk_to_rgb_scalar(cmyk, rgb);
}

// ── Public API ────────────────────────────────────────────────────────────────

/// CPU fallback for [`crate::GpuCtx::icc_cmyk_to_rgb`].
///
/// When `clut` is `None`, applies the subtractive complement formula:
///   `R = (255−C)*(255−K)/255` (rounded), same for G/M and B/Y.
///
/// When `clut` is `Some((table, grid_n))`, evaluates the 4D CLUT using
/// quadrilinear interpolation — the same algorithm as the GPU kernel.
///
/// The `clut = None` path uses the best SIMD available at runtime:
/// - x86-64: AVX-512 (16 px/iter) when avx512f+avx512bw are present, else scalar
/// - aarch64: NEON (8 px/iter) — mandatory on all ARMv8-A targets
/// - other: scalar per-pixel loop
///
/// # Panics
///
/// Panics if `clut` is `Some((_, grid_n))` and `grid_n < 2`.  A CLUT with
/// fewer than 2 nodes per axis is degenerate and unusable for interpolation.
#[must_use]
#[expect(
    clippy::too_many_lines,
    reason = "CLUT quadrilinear interpolation — cohesion outweighs length"
)]
pub fn icc_cmyk_to_rgb_cpu(cmyk: &[u8], clut: Option<(&[u8], u32)>) -> Vec<u8> {
    let n = cmyk.len() / 4;
    let mut rgb = vec![0u8; n * 3];

    match clut {
        None => {
            dispatch_cmyk_matrix(cmyk, &mut rgb);
        }
        Some((table, grid_n)) => {
            // grid_n = 0 would cause underflow in `grid_n - 1` and panic via
            // `g - 1` index arithmetic below.  Public function: guard explicitly.
            assert!(
                grid_n >= 2,
                "icc_cmyk_to_rgb_cpu: grid_n must be ≥ 2, got {grid_n}"
            );
            let g = grid_n as usize; // grid_n ≤ 255 from caller validation
            let g2 = g * g;
            let g3 = g2 * g;
            // grid_n ≤ 255 → (grid_n - 1) ≤ 254, exact in f32 (needs ≤ 8 mantissa bits).
            #[expect(
                clippy::cast_precision_loss,
                reason = "grid_n ≤ 255, fits exactly in f32 (8 bits < 23-bit mantissa)"
            )]
            let g1 = (grid_n - 1) as f32;
            let scale = g1 / 255.0;
            for (src, dst) in cmyk.chunks_exact(4).zip(rgb.chunks_exact_mut(3)) {
                let fc = f32::from(src[0]) * scale;
                let fm = f32::from(src[1]) * scale;
                let fy = f32::from(src[2]) * scale;
                let fk = f32::from(src[3]) * scale;

                // fc ∈ [0.0, g1] ⊂ [0.0, 254.0]; floor is non-negative and ≤ 254.
                // The sign-loss and truncation lints fire because `as usize` is UB
                // for negative or >usize::MAX floats; here neither can happen.
                #[expect(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "fc/fm/fy/fk are in [0.0, 254.0]; floor is exact and non-negative"
                )]
                let (ic0, im0, iy0, ik0) = (
                    (fc as usize).min(g - 1),
                    (fm as usize).min(g - 1),
                    (fy as usize).min(g - 1),
                    (fk as usize).min(g - 1),
                );
                let ic1 = (ic0 + 1).min(g - 1);
                let im1 = (im0 + 1).min(g - 1);
                let iy1 = (iy0 + 1).min(g - 1);
                let ik1 = (ik0 + 1).min(g - 1);

                // Fractional weights: difference between float position and floored index.
                // ic0 ≤ 254 → ic0 as f32 is exact (fits in 8 mantissa bits).
                #[expect(
                    clippy::cast_precision_loss,
                    reason = "ic0/im0/iy0/ik0 ≤ 254, exact in f32"
                )]
                let (wc, wm, wy, wk) = (
                    fc - ic0 as f32,
                    fm - im0 as f32,
                    fy - iy0 as f32,
                    fk - ik0 as f32,
                );

                let node = |ci: usize, mi: usize, yi: usize, ki: usize| -> [f32; 3] {
                    let idx = (ki * g3 + ci * g2 + mi * g + yi) * 3;
                    [
                        f32::from(table[idx]),
                        f32::from(table[idx + 1]),
                        f32::from(table[idx + 2]),
                    ]
                };

                let lerp = |a: f32, b: f32, t: f32| t.mul_add(b - a, a);
                let lerp3 = |a: [f32; 3], b: [f32; 3], t: f32| -> [f32; 3] {
                    [
                        lerp(a[0], b[0], t),
                        lerp(a[1], b[1], t),
                        lerp(a[2], b[2], t),
                    ]
                };

                // K=0 face: lerp Y → M → C
                let c0m0k0 = lerp3(node(ic0, im0, iy0, ik0), node(ic0, im0, iy1, ik0), wy);
                let c0m1k0 = lerp3(node(ic0, im1, iy0, ik0), node(ic0, im1, iy1, ik0), wy);
                let c1m0k0 = lerp3(node(ic1, im0, iy0, ik0), node(ic1, im0, iy1, ik0), wy);
                let c1m1k0 = lerp3(node(ic1, im1, iy0, ik0), node(ic1, im1, iy1, ik0), wy);
                let rk0 = lerp3(lerp3(c0m0k0, c0m1k0, wm), lerp3(c1m0k0, c1m1k0, wm), wc);

                // K=1 face: lerp Y → M → C
                let c0m0k1 = lerp3(node(ic0, im0, iy0, ik1), node(ic0, im0, iy1, ik1), wy);
                let c0m1k1 = lerp3(node(ic0, im1, iy0, ik1), node(ic0, im1, iy1, ik1), wy);
                let c1m0k1 = lerp3(node(ic1, im0, iy0, ik1), node(ic1, im0, iy1, ik1), wy);
                let c1m1k1 = lerp3(node(ic1, im1, iy0, ik1), node(ic1, im1, iy1, ik1), wy);
                let rk1 = lerp3(lerp3(c0m0k1, c0m1k1, wm), lerp3(c1m0k1, c1m1k1, wm), wc);

                let out = lerp3(rk0, rk1, wk);
                #[expect(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "clamped to [0,255] before cast"
                )]
                {
                    dst[0] = out[0].clamp(0.0, 255.0).round() as u8;
                    dst[1] = out[1].clamp(0.0, 255.0).round() as u8;
                    dst[2] = out[2].clamp(0.0, 255.0).round() as u8;
                }
            }
        }
    }

    rgb
}

#[cfg(test)]
mod tests {
    use super::icc_cmyk_to_rgb_cpu;

    #[test]
    fn icc_cmyk_matrix_white() {
        let cmyk = [0u8, 0, 0, 0];
        let rgb = icc_cmyk_to_rgb_cpu(&cmyk, None);
        assert_eq!(rgb, [255, 255, 255]);
    }

    #[test]
    fn icc_cmyk_matrix_black() {
        let cmyk = [0u8, 0, 0, 255];
        let rgb = icc_cmyk_to_rgb_cpu(&cmyk, None);
        assert_eq!(rgb, [0, 0, 0]);
    }

    #[test]
    fn icc_cmyk_matrix_pure_cyan() {
        // C=255, M=Y=K=0 → R=0, G=255, B=255
        let cmyk = [255u8, 0, 0, 0];
        let rgb = icc_cmyk_to_rgb_cpu(&cmyk, None);
        assert_eq!(rgb, [0, 255, 255]);
    }

    #[test]
    fn icc_cmyk_matrix_multi_pixel() {
        let cmyk = [0u8, 0, 0, 0, 0, 0, 0, 255];
        let rgb = icc_cmyk_to_rgb_cpu(&cmyk, None);
        assert_eq!(&rgb[0..3], &[255, 255, 255]);
        assert_eq!(&rgb[3..6], &[0, 0, 0]);
    }

    /// Parity: the active SIMD path must match `cmyk_to_rgb_reflectance` byte-for-byte.
    ///
    /// On x86-64 with `-C target-cpu=native` (AVX-512 machine) this exercises the
    /// AVX-512 branch. On aarch64 it exercises the NEON branch. On other targets
    /// both sides go scalar and the test is a tautology (still passes).
    #[test]
    fn icc_cmyk_matrix_simd_vs_scalar() {
        #[rustfmt::skip]
        let cmyk: Vec<u8> = vec![
            // white, black, cyan, magenta
              0,   0,   0,   0,
              0,   0,   0, 255,
            255,   0,   0,   0,
              0, 255,   0,   0,
            // yellow, key-only mid, all-max, all-mid
              0,   0, 255,   0,
              0,   0,   0, 128,
            255, 255, 255, 255,
            128, 128, 128, 128,
            // mid-range sweep (fills a full 16-pixel AVX-512 chunk and 8-pixel NEON chunk)
             64,  32,  16,   8,
            200, 100,  50,  25,
             10,  20,  30,  40,
             50,  60,  70,  80,
             90, 100, 110, 120,
            130, 140, 150, 160,
            170, 180, 190, 200,
            210, 220, 230, 240,
        ];
        assert_eq!(cmyk.len(), 64, "test vector must be exactly 16 pixels");

        let mut scalar_rgb = vec![0u8; 48];
        for (src, dst) in cmyk.chunks_exact(4).zip(scalar_rgb.chunks_exact_mut(3)) {
            let (r, g, b) = color::convert::cmyk_to_rgb_reflectance(src[0], src[1], src[2], src[3]);
            dst[0] = r;
            dst[1] = g;
            dst[2] = b;
        }

        let simd_rgb = icc_cmyk_to_rgb_cpu(&cmyk, None);

        for (i, (s, a)) in scalar_rgb.iter().zip(simd_rgb.iter()).enumerate() {
            assert_eq!(
                s,
                a,
                "RGB byte {i} (pixel {}, channel {}): scalar={s} simd={a}",
                i / 3,
                i % 3,
            );
        }
    }

    /// Non-multiple-of-chunk-size: exercises the scalar tail path on all SIMD tiers.
    #[test]
    fn icc_cmyk_matrix_tail_pixels() {
        // 5 pixels: AVX-512 processes 0 full chunks (< 16), NEON processes 0 (< 8).
        // All 5 pixels fall into the scalar tail on every arch.
        let cmyk: Vec<u8> = (0..5)
            .flat_map(|i| {
                let v = (i * 50) as u8;
                [
                    v,
                    v.wrapping_add(10),
                    v.wrapping_add(20),
                    v.wrapping_add(30),
                ]
            })
            .collect();
        let mut scalar_rgb = vec![0u8; 15];
        for (src, dst) in cmyk.chunks_exact(4).zip(scalar_rgb.chunks_exact_mut(3)) {
            let (r, g, b) = color::convert::cmyk_to_rgb_reflectance(src[0], src[1], src[2], src[3]);
            dst[0] = r;
            dst[1] = g;
            dst[2] = b;
        }
        assert_eq!(icc_cmyk_to_rgb_cpu(&cmyk, None), scalar_rgb);
    }

    /// NEON-specific: 8 pixels exactly — one full NEON chunk, zero AVX-512 chunks.
    #[test]
    fn icc_cmyk_matrix_eight_pixels() {
        let cmyk: Vec<u8> = (0u8..8)
            .flat_map(|i| {
                [
                    i.wrapping_mul(30),
                    i.wrapping_mul(20),
                    i.wrapping_mul(10),
                    i.wrapping_mul(5),
                ]
            })
            .collect();
        assert_eq!(cmyk.len(), 32);
        let mut scalar_rgb = vec![0u8; 24];
        for (src, dst) in cmyk.chunks_exact(4).zip(scalar_rgb.chunks_exact_mut(3)) {
            let (r, g, b) = color::convert::cmyk_to_rgb_reflectance(src[0], src[1], src[2], src[3]);
            dst[0] = r;
            dst[1] = g;
            dst[2] = b;
        }
        assert_eq!(icc_cmyk_to_rgb_cpu(&cmyk, None), scalar_rgb);
    }

    #[test]
    fn icc_cmyk_clut_identity_corners() {
        // 2^4 = 16-node CLUT where output = matrix formula at corners.
        let g: usize = 2;
        let mut table = vec![0u8; g * g * g * g * 3];
        for ki in 0..g {
            for ci in 0..g {
                for mi in 0..g {
                    for yi in 0..g {
                        let idx = (ki * g * g * g + ci * g * g + mi * g + yi) * 3;
                        let c = (ci * 255) as u8;
                        let m = (mi * 255) as u8;
                        let y = (yi * 255) as u8;
                        let k = (ki * 255) as u8;
                        let inv_k = u32::from(255 - k);
                        table[idx] = ((u32::from(255 - c) * inv_k) / 255) as u8;
                        table[idx + 1] = ((u32::from(255 - m) * inv_k) / 255) as u8;
                        table[idx + 2] = ((u32::from(255 - y) * inv_k) / 255) as u8;
                    }
                }
            }
        }
        let rgb = icc_cmyk_to_rgb_cpu(&[0u8, 0, 0, 0], Some((&table, 2)));
        assert_eq!(rgb, [255, 255, 255], "white corner");
        let rgb = icc_cmyk_to_rgb_cpu(&[0u8, 0, 0, 255], Some((&table, 2)));
        assert_eq!(rgb, [0, 0, 0], "black corner");
    }
}
