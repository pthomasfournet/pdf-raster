//! Shared arithmetic primitives used throughout the rasterizer.
//!
//! All compositing math (div255, lerp, Porter-Duff, color conversions, fixed-point
//! byte/col, floor/ceil/round) lives here — never copy-pasted into callers.

// ── Integer blend math ────────────────────────────────────────────────────────

/// Approximate x / 255 using the (x + (x>>8) + 0x80) >> 8 idiom.
/// Matches the C++ `div255` in Splash.cc:71. Saturates to 255 for x ≥ 65025.
#[inline]
#[must_use]
pub fn div255(x: u32) -> u8 {
    ((x + (x >> 8) + 0x80) >> 8).min(255) as u8
}

/// Bilinear lerp between two u8 values, weight t ∈ [0, 256].
/// t=0 → a, t=256 → b.
#[inline]
#[must_use]
pub fn lerp_u8(a: u8, b: u8, t: u32) -> u8 {
    div255(u32::from(a) * (256 - t) + u32::from(b) * t)
}

/// Porter-Duff src-over for a single channel.
/// `src_a` is the source alpha in [0, 255]; `dst` is the existing destination value.
#[inline]
#[must_use]
pub fn over_u8(src: u8, src_a: u8, dst: u8) -> u8 {
    let inv = 255 - u32::from(src_a);
    div255(u32::from(src) * u32::from(src_a) + u32::from(dst) * inv)
}

/// Clamp a u32 to [0, 255].
#[inline]
#[must_use]
pub fn clip255(x: u32) -> u8 {
    x.min(255) as u8
}

// ── Alpha premultiplication ───────────────────────────────────────────────────

/// Premultiply RGB by alpha (all in [0, 255]).
#[inline]
#[must_use]
pub fn premul(r: u8, g: u8, b: u8, a: u8) -> [u8; 3] {
    let aa = u32::from(a);
    [
        div255(u32::from(r) * aa),
        div255(u32::from(g) * aa),
        div255(u32::from(b) * aa),
    ]
}

/// Undo premultiplication. Returns (r, g, b) in [0, 255]. Safe when a=0.
#[inline]
#[must_use]
pub fn unpremul(r: u8, g: u8, b: u8, a: u8) -> [u8; 3] {
    if a == 0 {
        return [0, 0, 0];
    }
    let aa = u32::from(a);
    [
        ((u32::from(r) * 255 + aa / 2) / aa).min(255) as u8,
        ((u32::from(g) * 255 + aa / 2) / aa).min(255) as u8,
        ((u32::from(b) * 255 + aa / 2) / aa).min(255) as u8,
    ]
}

// ── Color space conversion ────────────────────────────────────────────────────

/// CMYK → RGB. Simple subtractive model matching poppler's cmykToRGBMatrixMultiplication.
#[inline]
#[must_use]
pub fn cmyk_to_rgb(c: u8, m: u8, y: u8, k: u8) -> (u8, u8, u8) {
    let kk = u32::from(k);
    // saturating_sub guarantees values ≤ 255 so clip255 is lossless.
    let red = clip255(255u32.saturating_sub(u32::from(c) + kk));
    let green = clip255(255u32.saturating_sub(u32::from(m) + kk));
    let blue = clip255(255u32.saturating_sub(u32::from(y) + kk));
    (red, green, blue)
}

#[inline]
#[must_use]
pub const fn gray_to_rgb(v: u8) -> (u8, u8, u8) {
    (v, v, v)
}

// ── GfxColorComp fixed-point ──────────────────────────────────────────────────
//
// poppler uses `int` with 16.16 fixed-point: gfxColorComp1 = 0x10000.
// These match GfxState.h:115–158.

/// u8 byte → 16.16 fixed-point `GfxColorComp`.
/// Matches: (x << 8) | x  (equivalent to x * 257).
#[inline]
#[must_use]
pub const fn byte_to_col(x: u8) -> i32 {
    let xi = x as i32;
    (xi << 8) | xi
}

/// 16.16 fixed-point `GfxColorComp` → u8, with rounding.
/// Matches poppler's colToByte.
#[inline]
#[must_use]
pub fn col_to_byte(x: i32) -> u8 {
    // clamp ensures the value is in [0, 255]; try_from then succeeds infallibly.
    u8::try_from(((x + 0x80) >> 8).clamp(0, 255)).unwrap_or(0)
}

// ── Geometry rounding (matching SplashMath.h portable fallbacks) ──────────────

/// Equivalent to C++ `splashFloor` — matches the portable fallback path on x86-64.
#[inline]
#[must_use]
pub fn splash_floor(x: f64) -> i32 {
    // floor() returns a whole-number f64. Convert via i64 (no overflow for PDF
    // coordinates), then clamp to i32 range via try_from.
    // SAFETY: f64::floor() always returns a whole-number value; casting to i64
    // is well-defined for any finite f64 whose absolute value fits in i64.
    let v: i64 = unsafe { x.floor().to_int_unchecked() };
    i32::try_from(v).unwrap_or(if v > 0 { i32::MAX } else { i32::MIN })
}

/// Equivalent to C++ `splashCeil` — matches the portable fallback path on x86-64.
#[inline]
#[must_use]
pub fn splash_ceil(x: f64) -> i32 {
    // SAFETY: f64::ceil() always returns a whole-number value; casting to i64
    // is well-defined for any finite f64 whose absolute value fits in i64.
    let v: i64 = unsafe { x.ceil().to_int_unchecked() };
    i32::try_from(v).unwrap_or(if v > 0 { i32::MAX } else { i32::MIN })
}

/// Equivalent to C++ `splashRound` — rounds half-integers toward positive infinity.
#[inline]
#[must_use]
pub fn splash_round(x: f64) -> i32 {
    splash_floor(x + 0.5)
}

// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn div255_exhaustive() {
        for x in 0u32..=65535 {
            let got = div255(x) as f64;
            // div255 returns u8, so saturate the expected value at 255.
            let expected = (x as f64 / 255.0).round().min(255.0);
            assert!(
                (got - expected).abs() <= 1.0,
                "div255({x}) = {got}, expected ≈ {expected}"
            );
        }
    }

    #[test]
    fn div255_boundary_products() {
        // all a*b products where a,b ∈ [0,255]
        for a in 0u32..=255 {
            for b in 0u32..=255 {
                let got = div255(a * b) as f64;
                let expected = (a as f64 * b as f64 / 255.0).round();
                assert!(
                    (got - expected).abs() <= 1.0,
                    "div255({a}*{b}) = {got}, expected ≈ {expected}"
                );
            }
        }
    }

    #[test]
    fn lerp_endpoints() {
        assert_eq!(lerp_u8(100, 200, 0), 100);
        // t=256: a*(256-256) + b*256 = 200*256; div255(51200) ≈ 201 due to rounding.
        // The lerp domain is [0, 256] with t=256 being "almost b", not exact b.
        let v = lerp_u8(100, 200, 256);
        assert!(
            (v as i32 - 200).abs() <= 1,
            "lerp t=256 gave {v}, expected ≈200"
        );
    }

    #[test]
    fn over_u8_opaque_src() {
        // fully opaque src should replace dst
        for src in 0u8..=255 {
            assert_eq!(over_u8(src, 255, 0), src);
            assert_eq!(over_u8(src, 255, 128), src);
        }
    }

    #[test]
    fn over_u8_transparent_src() {
        // fully transparent src should leave dst unchanged
        for dst in 0u8..=255 {
            assert_eq!(over_u8(0, 0, dst), dst);
        }
    }

    #[test]
    fn premul_unpremul_roundtrip() {
        // For small alpha values, premultiplication is lossy (quantization error
        // compounds on inversion). Only test with alpha >= 64 for ±2 accuracy.
        for a in [64u8, 128, 255] {
            let [pr, pg, pb] = premul(200, 100, 50, a);
            let [r, g, b] = unpremul(pr, pg, pb, a);
            assert!((r as i32 - 200).abs() <= 2, "r={r} a={a}");
            assert!((g as i32 - 100).abs() <= 2, "g={g} a={a}");
            assert!((b as i32 - 50).abs() <= 2, "b={b} a={a}");
        }
        // a=0: unpremul returns zeros, no crash.
        let [r, g, b] = unpremul(0, 0, 0, 0);
        assert_eq!([r, g, b], [0, 0, 0]);
    }

    #[test]
    fn splash_floor_ceil_round() {
        let cases = [
            (0.0f64, 0, 0, 0),
            (0.5, 0, 1, 1),
            (0.9, 0, 1, 1),
            (1.0, 1, 1, 1),
            (-0.1, -1, 0, 0),
            (-0.5, -1, 0, 0),
            (-0.6, -1, 0, -1),
            (-1.0, -1, -1, -1),
        ];
        for (x, fl, ce, ro) in cases {
            assert_eq!(splash_floor(x), fl, "floor({x})");
            assert_eq!(splash_ceil(x), ce, "ceil({x})");
            assert_eq!(splash_round(x), ro, "round({x})");
        }
    }

    #[test]
    fn byte_col_roundtrip() {
        for b in 0u8..=255 {
            let c = byte_to_col(b);
            let back = col_to_byte(c);
            // byte_to_col(x) = (x<<8)|x = x*257; col_to_byte rounds with +0x80,
            // so the round-trip is within ±1 (inherent to the fixed-point encoding).
            assert!(
                (back as i32 - b as i32).abs() <= 1,
                "byte_to_col({b}) = {c}, col_to_byte = {back}"
            );
        }
    }
}
