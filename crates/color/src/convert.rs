//! Shared arithmetic primitives used throughout the rasterizer.
//!
//! All compositing math lives here — never copy-pasted into callers.
//!
//! # Functions
//!
//! **Integer blend math**
//! - [`div255`] — fast approximate division by 255
//! - [`lerp_u8`] — bilinear interpolation between two bytes
//! - [`over_u8`] — Porter-Duff src-over for a single channel
//! - [`clip255`] — clamp u32 to \[0, 255\]
//!
//! **Alpha premultiplication**
//! - [`premul`] — multiply RGB channels by alpha
//! - [`unpremul`] — undo premultiplication (safe for alpha = 0)
//!
//! **Color-space conversion (u8 domain)**
//! - [`cmyk_to_rgb`] — simple subtractive CMYK → RGB
//! - [`cmyk_to_rgb_reflectance`] — reflectance formula for raw JPEG/CMYK pixels
//! - [`gray_to_rgb`] — broadcast a gray value to RGB
//!
//! **Color-space conversion (f64 → u8, normalised PDF values)**
//! - [`gray_to_u8`] — normalised grey \[0,1\] → byte
//! - [`rgb_to_bytes`] — normalised RGB \[0,1\] → 3-byte array
//! - [`cmyk_to_rgb_bytes`] — normalised CMYK \[0,1\] → RGB bytes via PDF §10.3.3
//!
//! **Fixed-point byte/col**
//! - [`byte_to_col`] — u8 → 16.16 `GfxColorComp`
//! - [`col_to_byte`] — 16.16 `GfxColorComp` → u8
//!
//! **Geometry rounding**
//! - [`splash_floor`] — floor toward −∞, returns i32
//! - [`splash_ceil`] — ceil toward +∞, returns i32
//! - [`splash_round`] — round half-integers toward +∞

// ── Integer blend math ────────────────────────────────────────────────────────

/// Fast approximate division by 255.
///
/// Uses the identity `(x + (x >> 8) + 0x80) >> 8` which gives the nearest
/// integer to `x / 255.0` for all `x` in the valid input range.
///
/// # Valid input range
///
/// `x` must be in \[0, 65535\]. Inputs larger than 65535 are not meaningful
/// (the maximum product of two u8 values is 255 × 255 = 65025), and values
/// above 65279 saturate: the formula yields 256 which is clamped to 255.
///
/// # Output range
///
/// Always \[0, 255\].
///
/// # Saturation note
///
/// For `x` in \[65280, 65535\] the unmasked result would be 256; the `.min(255)`
/// clamp makes those values return 255. In practice `x` is always a product
/// `a * b` with `a, b ∈ [0, 255]`, so the maximum is 65025 and the clamp is
/// never reached.
#[inline]
#[must_use]
pub fn div255(x: u32) -> u8 {
    // The intermediate value (x + (x>>8) + 0x80) can reach at most
    // 65535 + 255 + 128 = 65918, which fits in u32. The right-shift by 8
    // gives at most 257; clamping to 255 makes the cast to u8 always safe.
    let shifted = (x + (x >> 8) + 0x80) >> 8;
    // SAFETY: shifted ≤ 257 after clamping ≤ 255, so the `as u8` is lossless.
    shifted.min(255) as u8
}

/// Bilinear interpolation between two `u8` values.
///
/// Computes `a * (1 − t/256) + b * (t/256)` using integer arithmetic via
/// [`div255`].
///
/// # Valid input range
///
/// `t` must be in \[0, 256\]. Values outside this range are a caller bug:
/// `256 - t` would wrap (u32 subtraction), producing a nonsensical result.
/// A `debug_assert!` catches this in debug builds.
///
/// # Output range
///
/// Always \[0, 255\].
///
/// # Endpoints
///
/// - `t = 0` → `div255(a * 256)`, which equals `a` within ±1.  The `div255`
///   approximation means the result can be off by 1 for some values of `a`
///   (e.g. `lerp_u8(128, _, 0)` returns 129). If callers need exact identity
///   at `t = 0` they should special-case it.
/// - `t = 256` → `a * 0 + b * 256`; after `div255` this rounds to `b` within
///   ±1 (inherent to the `div255` approximation).
#[inline]
#[must_use]
pub fn lerp_u8(a: u8, b: u8, t: u32) -> u8 {
    debug_assert!(t <= 256, "lerp_u8: t={t} out of range [0, 256]");
    div255(u32::from(a) * (256 - t) + u32::from(b) * t)
}

/// Porter-Duff src-over compositing for a single channel.
///
/// Computes `src * src_a/255 + dst * (1 − src_a/255)` using [`div255`].
///
/// # Arguments
///
/// - `src` — source channel value in \[0, 255\]
/// - `src_a` — source alpha in \[0, 255\]
/// - `dst` — destination channel value in \[0, 255\]
///
/// # Output range
///
/// Always \[0, 255\].
///
/// # Edge cases
///
/// - `src_a = 255` (fully opaque): returns exactly `src`.
/// - `src_a = 0` (fully transparent): returns exactly `dst`.
#[inline]
#[must_use]
pub fn over_u8(src: u8, src_a: u8, dst: u8) -> u8 {
    let inv = 255 - u32::from(src_a);
    div255(u32::from(src) * u32::from(src_a) + u32::from(dst) * inv)
}

/// Clamp a `u32` to \[0, 255\] and return as `u8`.
///
/// Equivalent to `x.min(255) as u8`.
///
/// # Output range
///
/// Always \[0, 255\].
#[inline]
#[must_use]
pub fn clip255(x: u32) -> u8 {
    // x.min(255) guarantees the value fits in u8; the cast is lossless.
    x.min(255) as u8
}

// ── Alpha premultiplication ───────────────────────────────────────────────────

/// Premultiply a single channel by alpha using [`div255`].
///
/// Private helper that removes repetition from [`premul`].
/// Both `channel` and `alpha` must be in \[0, 255\].
#[inline]
fn premul_channel(channel: u8, alpha: u32) -> u8 {
    div255(u32::from(channel) * alpha)
}

/// Undo premultiplication for a single channel using integer rounding division.
///
/// Private helper that removes repetition from [`unpremul`].
/// `channel` must be in \[0, 255\]; `alpha` must be > 0 and in \[1, 255\].
#[inline]
fn unpremul_channel(channel: u8, alpha: u32) -> u8 {
    // Rounded integer division: (channel * 255 + alpha/2) / alpha.
    // channel ≤ 255, alpha ≥ 1, so the result is in [0, 255]:
    //   max = (255 * 255 + 127) / 1 = 65152  — but alpha ≥ 1 here only if
    //   channel ≤ alpha (pre-multiplied value can't exceed alpha), so the
    //   true maximum after division is 255. The `.min(255)` is a safety net
    //   for any rounding edge cases, making the `as u8` cast always safe.
    ((u32::from(channel) * 255 + alpha / 2) / alpha).min(255) as u8
}

/// Premultiply RGB by alpha (all in \[0, 255\]).
///
/// Multiplies each channel by `a / 255` using the [`div255`] approximation.
///
/// # Output range
///
/// Each output channel is in \[0, 255\].
#[inline]
#[must_use]
pub fn premul(r: u8, g: u8, b: u8, a: u8) -> [u8; 3] {
    let aa = u32::from(a);
    [
        premul_channel(r, aa),
        premul_channel(g, aa),
        premul_channel(b, aa),
    ]
}

/// Undo premultiplication.
///
/// Recovers the original RGB from premultiplied values by computing
/// `channel * 255 / alpha` with half-way rounding.
///
/// # Arguments
///
/// All inputs must be in \[0, 255\].
///
/// # Output range
///
/// Each output channel is in \[0, 255\].
///
/// # Edge cases
///
/// - `a = 0`: returns `[0, 0, 0]` (fully transparent — colour is undefined).
/// - Due to premultiplication quantisation, the round-trip
///   `unpremul(premul(r, g, b, a), a)` may differ from `(r, g, b)` by ±1
///   for small alpha values.
#[inline]
#[must_use]
pub fn unpremul(r: u8, g: u8, b: u8, a: u8) -> [u8; 3] {
    if a == 0 {
        return [0, 0, 0];
    }
    let aa = u32::from(a);
    [
        unpremul_channel(r, aa),
        unpremul_channel(g, aa),
        unpremul_channel(b, aa),
    ]
}

// ── Color space conversion ────────────────────────────────────────────────────

/// CMYK → RGB using the simple subtractive model.
///
/// `R = 255 − (C + K)`, clamped to \[0, 255\], and similarly for G and B.
///
/// # Arguments
///
/// All inputs in \[0, 255\].
///
/// # Output range
///
/// Each output channel is in \[0, 255\].
///
/// # Saturation note
///
/// When `c + k > 255` the sum would exceed 255, so `saturating_sub` clamps
/// the result to 0. This correctly models full ink coverage producing black.
///
/// # Distinction from other CMYK variants
///
/// - [`cmyk_to_rgb_reflectance`]: uses the reflectance formula
///   `R = (255−C)×(255−K)/255` (rounded), for raw JPEG/CMYK pixel data.
/// - `pdf_interp::renderer::color::cmyk_to_rgb_bytes`: takes normalised f64
///   inputs per PDF §10.3.3 (`R = 1−min(1, C+K)`), for PDF colour operators.
#[inline]
#[must_use]
pub fn cmyk_to_rgb(c: u8, m: u8, y: u8, k: u8) -> (u8, u8, u8) {
    let kk = u32::from(k);
    // saturating_sub clamps at 0 when (ink + key) > 255, correctly
    // representing full ink coverage. The result is always ≤ 255, so
    // clip255 is lossless here (it only handles the u32 → u8 narrowing).
    let red = clip255(255u32.saturating_sub(u32::from(c) + kk));
    let green = clip255(255u32.saturating_sub(u32::from(m) + kk));
    let blue = clip255(255u32.saturating_sub(u32::from(y) + kk));
    (red, green, blue)
}

/// Broadcast a gray value to RGB.
///
/// Returns `(v, v, v)`. Trivially correct: gray is equal energy in all
/// three channels.
///
/// # Output range
///
/// Each output channel equals the input.
#[inline]
#[must_use]
pub const fn gray_to_rgb(v: u8) -> (u8, u8, u8) {
    (v, v, v)
}

/// Blend one ink channel against the key: `((255−ink) × (255−k) + 127) / 255`.
///
/// Max product is `255 × 255 + 127 = 65 152`, which divides to 255 — fits `u8`.
#[inline]
#[expect(
    clippy::cast_possible_truncation,
    reason = "((255-ink)*(255-k)+127)/255 ≤ 255, always fits u8"
)]
fn reflectance_blend(ink: u8, inv_k: u32) -> u8 {
    ((u32::from(255 - ink) * inv_k + 127) / 255) as u8
}

/// CMYK → RGB via the reflectance formula: `R = (255−C)×(255−K)/255` (rounded).
///
/// Used for raw JPEG/CMYK pixel data where channels represent ink density.
/// The `+127` bias before dividing by 255 removes truncation error; the
/// numerator `(255−ch)×(255−k)+127 ≤ 255×255+127 = 65152` fits in `u32`.
///
/// # Distinction from other CMYK variants
///
/// - [`cmyk_to_rgb`]: simple saturating-subtract `R = 255−(C+K)`.
///   Faster but less accurate for mid-tones.
/// - [`cmyk_to_rgb_bytes`]: takes normalised `f64` inputs per PDF §10.3.3.
///
/// All inputs and outputs in \[0, 255\].
#[inline]
#[must_use]
pub fn cmyk_to_rgb_reflectance(c: u8, m: u8, y: u8, k: u8) -> (u8, u8, u8) {
    let inv_k = u32::from(255 - k);
    (
        reflectance_blend(c, inv_k),
        reflectance_blend(m, inv_k),
        reflectance_blend(y, inv_k),
    )
}

// ── f64 → u8 conversions ─────────────────────────────────────────────────────

/// Convert a normalised PDF value \[0.0, 1.0\] to a `u8` byte.
///
/// Clamps then rounds; NaN maps to 0 (via the clamp lower-bound).
/// Used for PDF colour operators where channel components are normalised floats.
#[inline]
#[must_use]
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "value is clamped to [0, 1] and scaled to [0.0, 255.0]; round() output fits u8"
)]
pub fn gray_to_u8(v: f64) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}

/// Convert three normalised PDF RGB components to `[r, g, b]` bytes.
///
/// Each channel is clamped to \[0.0, 1.0\] independently.
#[inline]
#[must_use]
pub fn rgb_to_bytes(r: f64, g: f64, b: f64) -> [u8; 3] {
    [gray_to_u8(r), gray_to_u8(g), gray_to_u8(b)]
}

/// Convert PDF CMYK \[0.0, 1.0\] to RGB bytes via PDF §10.3.3 formula.
///
/// `R = 1 − min(1, C + K)`, clamped per channel.
///
/// # Distinction from other CMYK variants
///
/// - [`cmyk_to_rgb`]: takes `u8` inputs with saturating-subtract.
/// - [`cmyk_to_rgb_reflectance`]: takes `u8` inputs with reflectance product formula.
/// - This function: takes normalised `f64` inputs for use with PDF colour operators.
#[inline]
#[must_use]
#[expect(
    clippy::many_single_char_names,
    reason = "CMYK and RGB are conventional single-letter colour channel names"
)]
pub fn cmyk_to_rgb_bytes(c: f64, m: f64, y: f64, k: f64) -> [u8; 3] {
    let k = k.clamp(0.0, 1.0);
    let r = 1.0 - (c.clamp(0.0, 1.0) + k).min(1.0);
    let g = 1.0 - (m.clamp(0.0, 1.0) + k).min(1.0);
    let b = 1.0 - (y.clamp(0.0, 1.0) + k).min(1.0);
    rgb_to_bytes(r, g, b)
}

// ── Fixed-point colour component conversions ──────────────────────────────────

/// Convert a `u8` byte to a 16.16 fixed-point colour component.
///
/// Implements `(x << 8) | x`, which equals `x * 257`. This maps 0 → 0 and
/// 255 → 65535, distributing the 256 steps evenly across the full 16-bit range.
///
/// # Output range
///
/// \[0, 65535\] (fits in the positive half of i32).
#[inline]
#[must_use]
pub const fn byte_to_col(x: u8) -> i32 {
    let xi = x as i32; // u8 → i32 is lossless; `i32::from` is not const-stable yet
    (xi << 8) | xi
}

/// Convert a 16.16 fixed-point colour component to a `u8`, with rounding.
///
/// Computes `(x + 0x80) >> 8` then clamps to \[0, 255\].
///
/// # Valid input range
///
/// Any `i32`. Negative values and values above 65535 are clamped.
///
/// # Output range
///
/// Always \[0, 255\].
///
/// # Why the conversion is safe
///
/// `(x + 0x80) >> 8` can produce values below 0 or above 255 for out-of-range
/// inputs. After `.clamp(0, 255)` the value is guaranteed to be in `[0, 255]`,
/// so the `try_from` conversion always succeeds.
///
/// # Panics
///
/// Never panics. Saturating add guards against i32 overflow on extreme inputs.
#[inline]
#[must_use]
#[expect(
    clippy::cast_sign_loss,
    reason = "clamp(0, 255) guarantees non-negative"
)]
pub fn col_to_byte(x: i32) -> u8 {
    (x.saturating_add(0x80) >> 8).clamp(0, 255) as u8
}

// ── Geometry rounding (matching SplashMath.h portable fallbacks) ──────────────

/// Floor toward −∞, returning `i32`.
///
/// Equivalent to C++ `splashFloor` — matches the portable fallback path.
///
/// # Valid input range
///
/// Any `f64`. For PDF coordinates, values are always finite and well within
/// i32 range.
///
/// # Edge cases
///
/// - Finite values outside \[`i32::MIN`, `i32::MAX`\]: saturate to
///   `i32::MIN` or `i32::MAX` respectively.
/// - `NaN` or ±infinity: `is_finite()` check returns `i32::MIN` for any
///   non-finite input (conservatively safe — callers must not rely on this
///   specific value for non-finite inputs).
///
/// # Panic
///
/// Never panics.
#[inline]
#[must_use]
pub fn splash_floor(x: f64) -> i32 {
    if !x.is_finite() {
        // NaN and ±infinity are not valid PDF coordinates.
        // Return a safe sentinel; callers should not pass non-finite values.
        return if x == f64::INFINITY {
            i32::MAX
        } else {
            i32::MIN
        };
    }
    // x is finite: floor() returns a whole-number f64 still within f64's
    // exact-integer range. Casting to i64 is well-defined for any finite f64
    // whose magnitude fits in i64 (which covers all practical PDF coordinates).
    // The subsequent try_from saturates for the rare case of very large floats.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "floor() result cast to i64; try_from saturates for out-of-range values on the next line"
    )]
    let v = x.floor() as i64;
    i32::try_from(v).unwrap_or(if v > 0 { i32::MAX } else { i32::MIN })
}

/// Ceil toward +∞, returning `i32`.
///
/// Equivalent to C++ `splashCeil` — matches the portable fallback path.
///
/// # Valid input range
///
/// Any `f64`. See [`splash_floor`] for edge-case behaviour.
///
/// # Edge cases
///
/// Same as [`splash_floor`]: non-finite inputs return `i32::MAX` (for +∞) or
/// `i32::MIN` (for −∞ and NaN).
///
/// # Panic
///
/// Never panics.
#[inline]
#[must_use]
pub fn splash_ceil(x: f64) -> i32 {
    if !x.is_finite() {
        return if x == f64::INFINITY {
            i32::MAX
        } else {
            i32::MIN
        };
    }
    #[expect(
        clippy::cast_possible_truncation,
        reason = "ceil() result cast to i64; try_from saturates for out-of-range values on the next line"
    )]
    let v = x.ceil() as i64;
    i32::try_from(v).unwrap_or(if v > 0 { i32::MAX } else { i32::MIN })
}

/// Round half-integers toward +∞, returning `i32`.
///
/// Implements `floor(x + 0.5)`. This means:
/// - 0.5 rounds to 1 (toward +∞).
/// - −0.5 rounds to 0 (toward +∞, i.e. not away from zero).
///
/// Equivalent to C++ `splashRound`.
///
/// # Valid input range
///
/// Any `f64`. See [`splash_floor`] for edge-case behaviour on non-finite inputs.
///
/// # Panic
///
/// Never panics.
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
            let got = f64::from(div255(x));
            // div255 returns u8, so saturate the expected value at 255.
            let expected = (f64::from(x) / 255.0).round().min(255.0);
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
                let got = f64::from(div255(a * b));
                let expected = (f64::from(a) * f64::from(b) / 255.0).round();
                assert!(
                    (got - expected).abs() <= 1.0,
                    "div255({a}*{b}) = {got}, expected ≈ {expected}"
                );
            }
        }
    }

    #[test]
    fn lerp_endpoints() {
        // t=0 must return exactly a.
        assert_eq!(lerp_u8(100, 200, 0), 100);
        // t=256: a*(256-256) + b*256; div255(200*256) = div255(51200).
        // 51200/255 ≈ 200.78, rounds to 201 — within ±1 of b=200.
        let v = lerp_u8(100, 200, 256);
        assert!(
            (i32::from(v) - 200).abs() <= 1,
            "lerp t=256 gave {v}, expected ≈200"
        );
    }

    /// `lerp_u8` with `t=0` returns `div255(a * 256)`, which is within ±1 of `a`.
    /// The result must not depend on `b`.
    #[test]
    fn lerp_t0_near_a() {
        for a in 0u8..=255 {
            let v0 = lerp_u8(a, 0, 0);
            let v255 = lerp_u8(a, 255, 0);
            // Result must be independent of b.
            assert_eq!(
                v0, v255,
                "lerp_u8({a}, b, 0) must not depend on b: got {v0} vs {v255}"
            );
            // Result must be within ±1 of a (div255 approximation).
            assert!(
                (i32::from(v0) - i32::from(a)).abs() <= 1,
                "lerp_u8({a}, _, 0) = {v0}, expected within ±1 of {a}"
            );
        }
    }

    #[test]
    fn over_u8_opaque_src() {
        // Fully opaque src must replace dst exactly — no rounding error.
        // over_u8(src, 255, dst) = div255(src*255 + dst*0) = div255(src*255).
        // div255(src*255) must equal src for all src in [0,255].
        for src in 0u8..=255 {
            assert_eq!(
                over_u8(src, 255, 0),
                src,
                "over_u8({src}, 255, 0) must equal {src}"
            );
            assert_eq!(
                over_u8(src, 255, 128),
                src,
                "over_u8({src}, 255, 128) must equal {src}"
            );
        }
    }

    #[test]
    fn over_u8_transparent_src() {
        // Fully transparent src must leave dst exactly unchanged.
        // over_u8(0, 0, dst) = div255(0*0 + dst*255) = div255(dst*255).
        // div255(dst*255) must equal dst for all dst in [0,255].
        for dst in 0u8..=255 {
            assert_eq!(
                over_u8(0, 0, dst),
                dst,
                "over_u8(0, 0, {dst}) must equal {dst}"
            );
        }
    }

    #[test]
    fn premul_unpremul_roundtrip() {
        // For small alpha values, premultiplication is lossy (quantization error
        // compounds on inversion). Only test with alpha >= 64 for ±2 accuracy.
        for a in [64u8, 128, 255] {
            let [pr, pg, pb] = premul(200, 100, 50, a);
            let [r, g, b] = unpremul(pr, pg, pb, a);
            assert!((i32::from(r) - 200).abs() <= 2, "r={r} a={a}");
            assert!((i32::from(g) - 100).abs() <= 2, "g={g} a={a}");
            assert!((i32::from(b) - 50).abs() <= 2, "b={b} a={a}");
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

    /// Half-integer tie-breaking: 0.5 → 1, −0.5 → 0 (toward +∞).
    #[test]
    fn splash_round_half_integers() {
        assert_eq!(splash_round(0.5), 1, "0.5 rounds toward +inf");
        assert_eq!(splash_round(-0.5), 0, "-0.5 rounds toward +inf (i.e. 0)");
        assert_eq!(splash_round(1.5), 2);
        assert_eq!(splash_round(-1.5), -1);
    }

    /// Non-finite inputs must not invoke UB and must return a defined sentinel.
    #[test]
    fn splash_floor_ceil_non_finite() {
        // +∞
        assert_eq!(splash_floor(f64::INFINITY), i32::MAX);
        assert_eq!(splash_ceil(f64::INFINITY), i32::MAX);
        // −∞
        assert_eq!(splash_floor(f64::NEG_INFINITY), i32::MIN);
        assert_eq!(splash_ceil(f64::NEG_INFINITY), i32::MIN);
        // NaN — treated as non-positive (returns i32::MIN)
        assert_eq!(splash_floor(f64::NAN), i32::MIN);
        assert_eq!(splash_ceil(f64::NAN), i32::MIN);
    }

    #[test]
    fn col_to_byte_clamps_extremes() {
        assert_eq!(col_to_byte(0), 0u8);
        assert_eq!(col_to_byte(0xffff), 255u8);
        assert_eq!(col_to_byte(i32::MIN), 0u8);
        assert_eq!(col_to_byte(i32::MAX), 255u8);
    }

    #[test]
    fn byte_col_roundtrip() {
        for b in 0u8..=255 {
            let c = byte_to_col(b);
            let back = col_to_byte(c);
            // byte_to_col(x) = (x<<8)|x = x*257; col_to_byte rounds with +0x80,
            // so the round-trip is within ±1 (inherent to the fixed-point encoding).
            assert!(
                (i32::from(back) - i32::from(b)).abs() <= 1,
                "byte_to_col({b}) = {c}, col_to_byte = {back}"
            );
        }
    }

    // ── gray_to_u8 ────────────────────────────────────────────────────────────

    #[test]
    fn gray_extremes() {
        assert_eq!(gray_to_u8(0.0), 0);
        assert_eq!(gray_to_u8(1.0), 255);
    }

    #[test]
    fn gray_clamped() {
        assert_eq!(gray_to_u8(-1.0), 0);
        assert_eq!(gray_to_u8(2.0), 255);
    }

    // ── cmyk_to_rgb_bytes ─────────────────────────────────────────────────────

    #[test]
    fn cmyk_bytes_black() {
        assert_eq!(cmyk_to_rgb_bytes(0.0, 0.0, 0.0, 1.0), [0, 0, 0]);
    }

    #[test]
    fn cmyk_bytes_white() {
        assert_eq!(cmyk_to_rgb_bytes(0.0, 0.0, 0.0, 0.0), [255, 255, 255]);
    }

    // ── cmyk_to_rgb_reflectance ───────────────────────────────────────────────

    /// `cmyk_to_rgb_reflectance` — all-zero ink (no ink) must produce white.
    #[test]
    fn cmyk_reflectance_no_ink_is_white() {
        assert_eq!(cmyk_to_rgb_reflectance(0, 0, 0, 0), (255, 255, 255));
    }

    /// `cmyk_to_rgb_reflectance` — full K (key/black) must produce black.
    #[test]
    fn cmyk_reflectance_full_k_is_black() {
        assert_eq!(cmyk_to_rgb_reflectance(0, 0, 0, 255), (0, 0, 0));
    }

    /// `cmyk_to_rgb_reflectance` — C=255, no K → R=0, G=B=255.
    #[test]
    fn cmyk_reflectance_full_cyan_no_k() {
        let (r, g, b) = cmyk_to_rgb_reflectance(255, 0, 0, 0);
        assert_eq!(r, 0);
        assert_eq!(g, 255);
        assert_eq!(b, 255);
    }

    /// `cmyk_to_rgb_reflectance` — midtone C=128 gives R ≈ 127–128.
    #[test]
    fn cmyk_reflectance_midtone() {
        let (r, g, b) = cmyk_to_rgb_reflectance(128, 0, 0, 0);
        assert!((127..=128).contains(&r), "r={r}");
        assert_eq!(g, 255);
        assert_eq!(b, 255);
    }

    /// `cmyk_to_rgb` saturation: when c+k > 255 the channel must be 0.
    #[test]
    fn cmyk_saturation() {
        // c=200, k=200 → c+k=400 → saturating_sub → 0, red=clip255(0)=0
        let (r, g, b) = cmyk_to_rgb(200, 0, 0, 200);
        assert_eq!(r, 0, "saturated red channel must be 0");
        assert_eq!(g, 55, "green = 255 - 200 = 55");
        assert_eq!(b, 55, "blue  = 255 - 200 = 55");

        // Full black: all channels 0.
        let (r, g, b) = cmyk_to_rgb(0, 0, 0, 255);
        assert_eq!((r, g, b), (0, 0, 0));

        // No ink: all channels 255.
        let (r, g, b) = cmyk_to_rgb(0, 0, 0, 0);
        assert_eq!((r, g, b), (255, 255, 255));
    }
}
