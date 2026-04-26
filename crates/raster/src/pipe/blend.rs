//! PDF blend modes (PDF spec §11.3.5).
//!
//! All separable modes operate per-channel with `src` and `dst` in `[0, 255]` and
//! return a blended value in `[0, 255]`.  For CMYK/DeviceN the caller inverts the
//! channel values before calling (`255 - v`) and inverts again after — the
//! subtractive complement trick used by the C++ `splashOutBlend*` functions.
//!
//! Non-separable modes (`Hue`, `Saturation`, `Color`, `Luminosity`) operate on
//! full RGB triples (always in additive space); CMYK callers convert first.

use crate::types::BlendMode;
use color::convert::div255;

// ── Separable blend primitives ────────────────────────────────────────────────

/// Normal: result is the source (Porter-Duff over handled in the pipe, not here).
#[must_use]
pub const fn blend_normal(src: u8, _dst: u8) -> u8 {
    src
}

/// Multiply: `div255(src * dst)`.
#[must_use]
pub fn blend_multiply(src: u8, dst: u8) -> u8 {
    div255(u32::from(src) * u32::from(dst))
}

/// Screen: `src + dst - div255(src * dst)`.
#[must_use]
pub fn blend_screen(src: u8, dst: u8) -> u8 {
    let s = u32::from(src);
    let d = u32::from(dst);
    #[expect(
        clippy::cast_possible_truncation,
        reason = "s + d - div255(s*d) ≤ 255: Screen result is always in [0,255]"
    )]
    let v = (s + d - u32::from(div255(s * d))) as u8;
    v
}

/// Overlay: hard-light of dst over src.
#[must_use]
pub fn blend_overlay(src: u8, dst: u8) -> u8 {
    if dst < 0x80 {
        div255(u32::from(src) * 2 * u32::from(dst))
    } else {
        let s = u32::from(255 - src);
        let d = u32::from(255 - dst);
        255 - div255(2 * s * d)
    }
}

/// Darken: `min(src, dst)`.
#[must_use]
pub fn blend_darken(src: u8, dst: u8) -> u8 {
    src.min(dst)
}

/// Lighten: `max(src, dst)`.
#[must_use]
pub fn blend_lighten(src: u8, dst: u8) -> u8 {
    src.max(dst)
}

/// Color dodge: brighten dst to reflect src.
#[must_use]
pub fn blend_color_dodge(src: u8, dst: u8) -> u8 {
    if src == 255 {
        255
    } else {
        ((u32::from(dst) * 255) / u32::from(255 - src)).min(255) as u8
    }
}

/// Color burn: darken dst to reflect src.
#[must_use]
pub fn blend_color_burn(src: u8, dst: u8) -> u8 {
    if src == 0 {
        0
    } else {
        let x = u32::from(255 - dst) * 255 / u32::from(src);
        #[expect(clippy::cast_possible_truncation, reason = "x < 255 is checked above, so 255 - x ≤ 254 which fits u8")]
        if x >= 255 { 0 } else { (255 - x) as u8 }
    }
}

/// Hard light: `Overlay(dst, src)` — the src/dst roles are swapped vs Overlay.
#[must_use]
pub fn blend_hard_light(src: u8, dst: u8) -> u8 {
    // hard_light(s, d) == overlay(d, s)
    blend_overlay(dst, src)
}

/// Soft light (PDF spec §11.3.5.3).
#[must_use]
pub fn blend_soft_light(src: u8, dst: u8) -> u8 {
    let s = i32::from(src);
    let d = i32::from(dst);
    let result = if s < 0x80 {
        d - (255 - 2 * s) * d * (255 - d) / (255 * 255)
    } else {
        let x = if d < 0x40 {
            (((16 * d - 12 * 255) * d / 255 + 4 * 255) * d) / 255
        } else {
            // Integer sqrt: sqrt(255 * d), scaled to [0,255].
            #[expect(clippy::cast_possible_truncation, reason = "sqrt of non-negative f64; result is in [0,255] before clamp")]
            { (f64::from(d) * 255.0).sqrt() as i32 }
        };
        d + (2 * s - 255) * (x - d) / 255
    };
    #[expect(clippy::cast_sign_loss, reason = "value is clamped to [0, 255] above")]
    { result.clamp(0, 255) as u8 }
}

/// Difference: `|src - dst|`.
#[must_use]
pub const fn blend_difference(src: u8, dst: u8) -> u8 {
    src.abs_diff(dst)
}

/// Exclusion: `src + dst - 2 × div255(src × dst)`.
#[must_use]
pub fn blend_exclusion(src: u8, dst: u8) -> u8 {
    let s = u32::from(src);
    let d = u32::from(dst);
    (s + d).saturating_sub(u32::from(div255(2 * s * d))).min(255) as u8
}

// ── Non-separable helpers (RGB additive space) ────────────────────────────────

/// Luminance of an RGB triple using the PDF/BT.709 weights (integer rounding).
#[must_use]
const fn get_lum(r: i32, g: i32, b: i32) -> i32 {
    (r * 77 + g * 151 + b * 28 + 0x80) >> 8
}

/// Saturation: max channel − min channel.
#[must_use]
fn get_sat(r: i32, g: i32, b: i32) -> i32 {
    r.max(g).max(b) - r.min(g).min(b)
}

/// Clip an out-of-range (r, g, b) triple back into [0, 255] while preserving luminance.
fn clip_color(r_in: i32, g_in: i32, b_in: i32) -> (i32, i32, i32) {
    let lum = get_lum(r_in, g_in, b_in);
    let rgb_min = r_in.min(g_in).min(b_in);
    let rgb_max = r_in.max(g_in).max(b_in);
    if rgb_min < 0 {
        let d = lum - rgb_min;
        (
            (lum + (r_in - lum) * lum / d).clamp(0, 255),
            (lum + (g_in - lum) * lum / d).clamp(0, 255),
            (lum + (b_in - lum) * lum / d).clamp(0, 255),
        )
    } else if rgb_max > 255 {
        let d = rgb_max - lum;
        (
            (lum + (r_in - lum) * (255 - lum) / d).clamp(0, 255),
            (lum + (g_in - lum) * (255 - lum) / d).clamp(0, 255),
            (lum + (b_in - lum) * (255 - lum) / d).clamp(0, 255),
        )
    } else {
        (r_in, g_in, b_in)
    }
}

/// Shift the luminance of (r, g, b) to `lum` while preserving hue/saturation.
fn set_lum(r: i32, g: i32, b: i32, lum: i32) -> (i32, i32, i32) {
    let d = lum - get_lum(r, g, b);
    clip_color(r + d, g + d, b + d)
}

/// Scale (r, g, b) so its saturation equals `sat`, while preserving hue.
fn set_sat(r_in: i32, g_in: i32, b_in: i32, sat: i32) -> (i32, i32, i32) {
    // Sort channels into (min, mid, max) keeping track of which index is which.
    let mut channels = [(r_in, 0usize), (g_in, 1), (b_in, 2)];
    channels.sort_unstable_by_key(|&(v, _)| v);
    let (ch_lo, slot_lo) = channels[0];
    let (ch_md, slot_md) = channels[1];
    let (ch_hi, slot_hi) = channels[2];

    let mut out = [0i32; 3];
    if ch_hi > ch_lo {
        out[slot_md] = ((ch_md - ch_lo) * sat / (ch_hi - ch_lo)).clamp(0, 255);
        out[slot_hi] = sat.clamp(0, 255);
    }
    out[slot_lo] = 0;
    #[expect(clippy::tuple_array_conversions, reason = "caller API returns a triple; no Into impl for non-Copy i32")]
    { let [a, b, c] = out; (a, b, c) }
}

/// Non-separable Hue blend: hue of src, saturation and luminosity of dst.
#[must_use]
pub fn blend_hue_rgb(src: [u8; 3], dst: [u8; 3]) -> [u8; 3] {
    let [sr, sg, sb] = src.map(i32::from);
    let [dr, dg, db] = dst.map(i32::from);
    let (r0, g0, b0) = set_sat(sr, sg, sb, get_sat(dr, dg, db));
    let (r1, g1, b1) = set_lum(r0, g0, b0, get_lum(dr, dg, db));
    #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "clip_color clamps all channels to [0, 255]")]
    [r1 as u8, g1 as u8, b1 as u8]
}

/// Non-separable Saturation blend: saturation of src, hue and luminosity of dst.
#[must_use]
pub fn blend_saturation_rgb(src: [u8; 3], dst: [u8; 3]) -> [u8; 3] {
    let [sr, sg, sb] = src.map(i32::from);
    let [dr, dg, db] = dst.map(i32::from);
    let (r0, g0, b0) = set_sat(dr, dg, db, get_sat(sr, sg, sb));
    let (r1, g1, b1) = set_lum(r0, g0, b0, get_lum(dr, dg, db));
    #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "clip_color clamps all channels to [0, 255]")]
    [r1 as u8, g1 as u8, b1 as u8]
}

/// Non-separable Color blend: hue and saturation of src, luminosity of dst.
#[must_use]
pub fn blend_color_rgb(src: [u8; 3], dst: [u8; 3]) -> [u8; 3] {
    let [sr, sg, sb] = src.map(i32::from);
    let [dr, dg, db] = dst.map(i32::from);
    let (r, g, b) = set_lum(sr, sg, sb, get_lum(dr, dg, db));
    #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "clip_color clamps all channels to [0, 255]")]
    [r as u8, g as u8, b as u8]
}

/// Non-separable Luminosity blend: luminosity of src, hue and saturation of dst.
#[must_use]
pub fn blend_luminosity_rgb(src: [u8; 3], dst: [u8; 3]) -> [u8; 3] {
    let [sr, sg, sb] = src.map(i32::from);
    let [dr, dg, db] = dst.map(i32::from);
    let (r, g, b) = set_lum(dr, dg, db, get_lum(sr, sg, sb));
    #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "clip_color clamps all channels to [0, 255]")]
    [r as u8, g as u8, b as u8]
}

// ── Dispatch ──────────────────────────────────────────────────────────────────

/// Apply a separable blend mode to each corresponding byte pair from `src` and `dst`.
///
/// For CMYK/DeviceN modes the caller must invert before and after (subtractive complement).
/// `src.len()`, `dst.len()`, and `out.len()` must all be equal.
///
/// # Panics
///
/// Panics if `mode` is a non-separable variant (`Hue`, `Saturation`, `Color`,
/// `Luminosity`).  Use [`apply_nonseparable_rgb`] for those.
pub fn apply_separable(mode: BlendMode, src: &[u8], dst: &[u8], out: &mut [u8]) {
    debug_assert_eq!(src.len(), dst.len());
    debug_assert_eq!(src.len(), out.len());
    let f: fn(u8, u8) -> u8 = match mode {
        BlendMode::Normal => blend_normal,
        BlendMode::Multiply => blend_multiply,
        BlendMode::Screen => blend_screen,
        BlendMode::Overlay => blend_overlay,
        BlendMode::Darken => blend_darken,
        BlendMode::Lighten => blend_lighten,
        BlendMode::ColorDodge => blend_color_dodge,
        BlendMode::ColorBurn => blend_color_burn,
        BlendMode::HardLight => blend_hard_light,
        BlendMode::SoftLight => blend_soft_light,
        BlendMode::Difference => blend_difference,
        BlendMode::Exclusion => blend_exclusion,
        BlendMode::Hue | BlendMode::Saturation | BlendMode::Color | BlendMode::Luminosity => {
            panic!("apply_separable called with non-separable mode {mode:?}")
        }
    };
    for ((s, d), o) in src.iter().zip(dst).zip(out.iter_mut()) {
        *o = f(*s, *d);
    }
}

/// Apply a non-separable blend mode (`Hue`/`Saturation`/`Color`/`Luminosity`) to an RGB triple.
///
/// `src` and `dst` are in additive space — CMYK callers convert first
/// (`255 - channel`).  Only the three components are blended; the caller
/// handles channel [3] (K/alpha) separately.
///
/// # Panics
///
/// Panics if `mode` is a separable variant.  Use [`apply_separable`] for those.
#[must_use]
pub fn apply_nonseparable_rgb(mode: BlendMode, src: [u8; 3], dst: [u8; 3]) -> [u8; 3] {
    match mode {
        BlendMode::Hue => blend_hue_rgb(src, dst),
        BlendMode::Saturation => blend_saturation_rgb(src, dst),
        BlendMode::Color => blend_color_rgb(src, dst),
        BlendMode::Luminosity => blend_luminosity_rgb(src, dst),
        _ => panic!("apply_nonseparable_rgb called with separable mode {mode:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multiply_identity() {
        assert_eq!(blend_multiply(255, 200), 200);
        assert_eq!(blend_multiply(200, 255), 200);
        assert_eq!(blend_multiply(0, 200), 0);
        assert_eq!(blend_multiply(200, 0), 0);
    }

    #[test]
    fn screen_identity() {
        assert_eq!(blend_screen(0, 200), 200);
        assert_eq!(blend_screen(200, 0), 200);
        assert_eq!(blend_screen(255, 100), 255);
    }

    #[test]
    fn difference_is_abs_diff() {
        for a in (0u8..=255).step_by(13) {
            for b in (0u8..=255).step_by(7) {
                assert_eq!(blend_difference(a, b), a.abs_diff(b));
            }
        }
    }

    #[test]
    fn color_dodge_saturates_at_src_255() {
        assert_eq!(blend_color_dodge(255, 100), 255);
    }

    #[test]
    fn color_burn_zeroes_at_src_0() {
        assert_eq!(blend_color_burn(0, 100), 0);
    }

    #[test]
    fn hard_light_matches_overlay_swapped() {
        for s in (0u8..=255).step_by(17) {
            for d in (0u8..=255).step_by(11) {
                assert_eq!(blend_hard_light(s, d), blend_overlay(d, s),
                    "hard_light({s},{d}) should equal overlay({d},{s})");
            }
        }
    }

    #[test]
    fn get_lum_white_is_255() {
        assert_eq!(get_lum(255, 255, 255), 255);
    }

    #[test]
    fn get_lum_black_is_0() {
        assert_eq!(get_lum(0, 0, 0), 0);
    }

    #[test]
    fn get_sat_grey_is_0() {
        assert_eq!(get_sat(128, 128, 128), 0);
    }

    #[test]
    fn luminosity_grey_dst_stays_grey() {
        // Luminosity of red over grey: result should be grey (all channels equal).
        let src = [200u8, 50, 50];
        let dst = [128u8, 128, 128];
        let out = blend_luminosity_rgb(src, dst);
        assert_eq!(out[0], out[1]);
        assert_eq!(out[1], out[2]);
    }

    #[test]
    fn apply_separable_multiply_all_channels() {
        let src = [100u8, 200, 50];
        let dst = [255u8, 128, 0];
        let mut out = [0u8; 3];
        apply_separable(BlendMode::Multiply, &src, &dst, &mut out);
        assert_eq!(out[0], blend_multiply(100, 255));
        assert_eq!(out[1], blend_multiply(200, 128));
        assert_eq!(out[2], blend_multiply(50, 0));
    }

    #[test]
    #[should_panic(expected = "apply_separable called with non-separable mode")]
    fn apply_separable_panics_on_nonseparable() {
        let mut out = [0u8; 3];
        apply_separable(BlendMode::Hue, &[0; 3], &[0; 3], &mut out);
    }

    #[test]
    fn soft_light_midpoint() {
        // For src=128, dst=128: result should be close to 128.
        let r = blend_soft_light(128, 128);
        assert!((r as i32 - 128).abs() <= 2, "soft_light(128,128)={r}, expected ~128");
    }

    #[test]
    fn exclusion_is_screen_minus_extra_mul() {
        // Exclusion(s,d) = s + d - 2×div255(s×d)
        // Screen(s,d)    = s + d - div255(s×d)
        // So exclusion ≤ screen for all inputs.
        for s in (0u8..=255).step_by(23) {
            for d in (0u8..=255).step_by(19) {
                let ex = blend_exclusion(s, d);
                let sc = blend_screen(s, d);
                assert!(ex <= sc, "exclusion({s},{d})={ex} > screen({s},{d})={sc}");
            }
        }
    }

    #[test]
    fn screen_clamp_never_exceeds_255() {
        for s in 0u8..=255 {
            for d in 0u8..=255 {
                let result = blend_screen(s, d);
                assert!(result <= 255);
            }
        }
    }
}
