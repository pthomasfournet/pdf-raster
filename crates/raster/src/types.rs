//! Raster-local enums and workspace-wide constants.
//!
//! Re-exports all [`color`] public types so that downstream modules within
//! this crate only need `use crate::types::*`.

use std::borrow::Cow;

/// Re-export every public type from the [`color`] crate, including all
/// arithmetic helpers from [`color::convert`].
///
/// Downstream modules within this crate can import everything they need with
/// a single `use crate::types::*`.
pub use color::{
    AnyColor, Cmyk8, DeviceN8, Gray8, NCOMPS, Pixel, PixelMode, Rgb8, Rgba8, TransferLut,
    convert::*,
};

// ── Rasterizer constants ──────────────────────────────────────────────────────

/// Supersampling factor for anti-aliasing.
///
/// The AA buffer is `bitmap_width × AA_SIZE` pixels wide and `AA_SIZE` rows
/// tall. Changing this value requires corresponding changes to all AA-buffer
/// allocation and compositing logic.
///
/// Unit: pixels (linear).
pub const AA_SIZE: i32 = 4;

/// Maximum number of De Casteljau subdivisions when flattening a Bézier curve.
///
/// A value of 1024 gives sub-pixel accuracy for all practical PDF coordinate
/// ranges. Increasing this value raises stack and array allocation costs
/// quadratically; decreasing it degrades curve quality.
///
/// Unit: dimensionless iteration count.
pub const MAX_CURVE_SPLITS: i32 = 1024;

/// Control-point ratio for approximating a quarter-circle with a cubic Bézier.
///
/// Derivation: `4 * (√2 − 1) / 3 ≈ 0.552_284_75`. Four cubic Bézier segments
/// with this control-point ratio approximate a unit circle with a maximum
/// radial error below 0.03 %.
///
/// Unit: dimensionless ratio (fraction of the radius).
pub const BEZIER_CIRCLE: f64 = 0.552_284_75;

/// Number of spot color channels in a [`DeviceN8`] pixel.
///
/// A `DeviceN8` pixel is laid out as `[C, M, Y, K, S0, S1, S2, S3]`, giving
/// 4 process + 4 spot = 8 bytes per pixel total.
pub const SPOT_NCOMPS: usize = 4;

// ── Enums ─────────────────────────────────────────────────────────────────────

/// Thin-line rendering treatment.
///
/// Controls how strokes whose device-space width is less than one pixel are
/// drawn. The default preserves the hairline width via shape anti-aliasing
/// when stroke adjustment (`SA`) is on.
///
/// # Extension policy
///
/// This enum is **not** `#[non_exhaustive]` because exhaustive matches on it
/// are used throughout the rasterizer to guarantee every mode is handled; the
/// compiler enforces this at every call site. Adding a new variant is an
/// intentional breaking change that forces all match sites to be updated.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
pub enum ThinLineMode {
    /// Preserve the hairline width.
    ///
    /// When stroke adjustment is on and the device-space line width is less
    /// than half a pixel, the line is drawn as a shaped (anti-aliased) stroke;
    /// otherwise it is drawn solid.
    #[default]
    Default,
    /// Render as a solid opaque 1-pixel line regardless of width.
    Solid,
    /// Always draw with shape anti-aliasing even for sub-pixel widths.
    Shape,
}

/// Stroke line-cap style.
///
/// Determines how open sub-path endpoints are capped. Corresponds directly to
/// the PDF `lineCap` graphics-state parameter (PDF 32000-1:2008, §8.4.3.3).
///
/// # Extension policy
///
/// Not `#[non_exhaustive]`: PDF specifies exactly three cap styles; exhaustive
/// matching is a compile-time guarantee that all are handled.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
pub enum LineCap {
    /// Square end flush with the endpoint (PDF cap style 0).
    #[default]
    Butt,
    /// Semicircle centred on the endpoint, radius = half line-width (PDF cap style 1).
    Round,
    /// Square extending half the line-width beyond the endpoint (PDF cap style 2).
    Projecting,
}

/// Stroke line-join style.
///
/// Determines how two stroke segments meet at a shared vertex. Corresponds to
/// the PDF `lineJoin` graphics-state parameter (PDF 32000-1:2008, §8.4.3.4).
///
/// # Extension policy
///
/// Not `#[non_exhaustive]`: PDF specifies exactly three join styles.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
pub enum LineJoin {
    /// Sharp mitered corner, clipped at the miter limit (PDF join style 0).
    #[default]
    Miter,
    /// Rounded join — a filled arc centred on the vertex (PDF join style 1).
    Round,
    /// Flat bevel — the outside corner is cut off (PDF join style 2).
    Bevel,
}

/// Halftone screen type used by [`ScreenParams`].
///
/// # Extension policy
///
/// Not `#[non_exhaustive]`: the screen-building code uses exhaustive match; the
/// compiler ensures new screen types are handled everywhere.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
pub enum ScreenType {
    /// Bayer-style dispersed-dot ordered dither.
    ///
    /// Chosen automatically below 300 dpi (4 × 4 matrix). The default.
    #[default]
    Dispersed,
    /// Clustered-dot halftone screen.
    Clustered,
    /// Stochastic clustered-dot screen (64 × 64 matrix, used ≥ 300 dpi).
    StochasticClustered,
}

/// Parameters for constructing a halftone screen.
///
/// # Valid values
///
/// - [`kind`](ScreenParams::kind): any [`ScreenType`] variant.
/// - [`size`](ScreenParams::size): must be a **power of two** and **≥ 2**
///   (e.g. 2, 4, 8, 16 …). The screen matrix is `size × size` cells. Values
///   that are not powers of two, or values less than 2, produce an undefined
///   screen pattern. Call [`validate`](ScreenParams::validate) after construction
///   to enforce these constraints.
/// - [`dot_radius`](ScreenParams::dot_radius): meaningful only for
///   [`ScreenType::StochasticClustered`]; must be ≥ 1. Ignored for other screen
///   types but must still be positive.
///
/// # Default
///
/// `{ Dispersed, size: 2, dot_radius: 2 }`.
#[derive(Copy, Clone, Debug)]
pub struct ScreenParams {
    /// The halftone algorithm to use.
    pub kind: ScreenType,

    /// Screen matrix dimension in cells.
    ///
    /// Must be a power of two and ≥ 2. Typical values: 2, 4, 8, 16, 32, 64.
    pub size: i32,

    /// Dot radius for [`ScreenType::StochasticClustered`] screens.
    ///
    /// Must be ≥ 1. Ignored (but still validated) for other screen types.
    pub dot_radius: i32,
}

impl Default for ScreenParams {
    /// Returns the default screen parameters: `{ Dispersed, size: 2, dot_radius: 2 }`.
    fn default() -> Self {
        Self {
            kind: ScreenType::Dispersed,
            size: 2,
            dot_radius: 2,
        }
    }
}

impl ScreenParams {
    /// Validates that the parameter values are within their documented ranges.
    ///
    /// # Constraints checked
    ///
    /// - `size` must be ≥ 2.
    /// - `size` must be a power of two.
    /// - `dot_radius` must be ≥ 1.
    ///
    /// # Errors
    ///
    /// Returns `Err` with a human-readable [`Cow<'static, str>`] message
    /// describing the first constraint violated.  All current error messages
    /// are static string literals (`Cow::Borrowed`), so no allocation occurs.
    /// Future callers that need dynamic messages (e.g. including the offending
    /// field value) can return `Cow::Owned(format!(...))` without a breaking
    /// API change.
    ///
    /// # Examples
    ///
    /// ```
    /// # use raster::types::{ScreenParams, ScreenType};
    /// assert!(ScreenParams::default().validate().is_ok());
    ///
    /// let bad = ScreenParams { size: 3, ..ScreenParams::default() };
    /// assert!(bad.validate().is_err());
    /// ```
    pub const fn validate(&self) -> Result<(), Cow<'static, str>> {
        if self.size < 2 {
            return Err(Cow::Borrowed("ScreenParams::size must be >= 2"));
        }
        // Casting to u32 is safe: we already checked size >= 2 > 0, so the
        // sign bit is clear and no bits are lost.
        #[expect(
            clippy::cast_sign_loss,
            reason = "size >= 2 has been asserted above; the value is positive, \
                      so the cast to u32 loses no bits"
        )]
        let size_u32 = self.size as u32;
        if !size_u32.is_power_of_two() {
            return Err(Cow::Borrowed("ScreenParams::size must be a power of two"));
        }
        if self.dot_radius < 1 {
            return Err(Cow::Borrowed("ScreenParams::dot_radius must be >= 1"));
        }
        Ok(())
    }
}

// ── BlendMode ─────────────────────────────────────────────────────────────────

/// PDF compositing blend mode (PDF 32000-1:2008, §11.3.5).
///
/// Blend mode is a typed enum; dispatch happens in `pipe::blend`.
///
/// # Separable vs. non-separable
///
/// The first twelve variants (`Normal` through `Exclusion`) are *separable*:
/// the blend function operates independently on each colour channel.
/// The last four (`Hue` through `Luminosity`) are *non-separable*: they operate
/// on the full RGB triple (with CMYK converted to additive space first).
///
/// `Normal` is by far the most common; the pipeline fast-paths for it.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
pub enum BlendMode {
    /// Standard Porter-Duff over (gfxBlendNormal).
    #[default]
    Normal,
    /// `Cs × Cd` (gfxBlendMultiply).
    Multiply,
    /// `Cs + Cd - Cs × Cd` (gfxBlendScreen).
    Screen,
    /// Hard-light of Cd over Cs (gfxBlendOverlay).
    Overlay,
    /// `min(Cs, Cd)` (gfxBlendDarken).
    Darken,
    /// `max(Cs, Cd)` (gfxBlendLighten).
    Lighten,
    /// Brighten Cd to reflect Cs (gfxBlendColorDodge).
    ColorDodge,
    /// Darken Cd to reflect Cs (gfxBlendColorBurn).
    ColorBurn,
    /// Multiply or screen depending on Cs < 0.5 (gfxBlendHardLight).
    HardLight,
    /// Soft version of `HardLight` (gfxBlendSoftLight).
    SoftLight,
    /// `|Cd - Cs|` (gfxBlendDifference).
    Difference,
    /// `Cs + Cd - 2 × Cs × Cd` (gfxBlendExclusion).
    Exclusion,
    /// Hue of Cs, saturation and luminosity of Cd (non-separable, gfxBlendHue).
    Hue,
    /// Saturation of Cs, hue and luminosity of Cd (non-separable, gfxBlendSaturation).
    Saturation,
    /// Hue and saturation of Cs, luminosity of Cd (non-separable, gfxBlendColor).
    Color,
    /// Luminosity of Cs, hue and saturation of Cd (non-separable, gfxBlendLuminosity).
    Luminosity,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn screen_params_default_matches_cpp() {
        let p = ScreenParams::default();
        assert_eq!(p.kind, ScreenType::Dispersed);
        assert_eq!(p.size, 2);
        assert_eq!(p.dot_radius, 2);
        assert!(p.validate().is_ok());
    }

    #[test]
    fn screen_params_validate_rejects_non_power_of_two() {
        let p = ScreenParams {
            size: 3,
            ..ScreenParams::default()
        };
        assert!(p.validate().is_err());
    }

    #[test]
    fn screen_params_validate_rejects_size_less_than_2() {
        let p = ScreenParams {
            size: 1,
            ..ScreenParams::default()
        };
        assert!(p.validate().is_err());
        let p = ScreenParams {
            size: 0,
            ..ScreenParams::default()
        };
        assert!(p.validate().is_err());
        let p = ScreenParams {
            size: -1,
            ..ScreenParams::default()
        };
        assert!(p.validate().is_err());
    }

    #[test]
    fn screen_params_validate_rejects_zero_dot_radius() {
        let p = ScreenParams {
            dot_radius: 0,
            ..ScreenParams::default()
        };
        assert!(p.validate().is_err());
    }

    /// `validate` errors are `Cow::Borrowed` (static string literals) — no
    /// heap allocation occurs for any of the three failure paths.
    #[test]
    fn screen_params_validate_errors_are_borrowed() {
        use std::borrow::Cow;

        let size_small = ScreenParams {
            size: 1,
            ..ScreenParams::default()
        };
        assert!(
            matches!(size_small.validate(), Err(Cow::Borrowed(_))),
            "size < 2 error should be Cow::Borrowed"
        );

        let size_non_pow2 = ScreenParams {
            size: 3,
            ..ScreenParams::default()
        };
        assert!(
            matches!(size_non_pow2.validate(), Err(Cow::Borrowed(_))),
            "non-power-of-two error should be Cow::Borrowed"
        );

        let bad_radius = ScreenParams {
            dot_radius: 0,
            ..ScreenParams::default()
        };
        assert!(
            matches!(bad_radius.validate(), Err(Cow::Borrowed(_))),
            "dot_radius < 1 error should be Cow::Borrowed"
        );
    }

    /// `validate` error messages contain the field name so they are human-readable.
    #[test]
    fn screen_params_validate_error_messages_mention_field() {
        let size_small = ScreenParams {
            size: 0,
            ..ScreenParams::default()
        };
        let msg = size_small.validate().unwrap_err();
        assert!(
            msg.contains("size"),
            "error for small size should mention 'size', got: {msg}"
        );

        let size_non_pow2 = ScreenParams {
            size: 3,
            ..ScreenParams::default()
        };
        let msg = size_non_pow2.validate().unwrap_err();
        assert!(
            msg.contains("size"),
            "error for non-power-of-two should mention 'size', got: {msg}"
        );

        let bad_radius = ScreenParams {
            dot_radius: 0,
            ..ScreenParams::default()
        };
        let msg = bad_radius.validate().unwrap_err();
        assert!(
            msg.contains("dot_radius"),
            "error for bad radius should mention 'dot_radius', got: {msg}"
        );
    }

    #[test]
    fn screen_params_validate_accepts_valid_power_of_two_sizes() {
        for exp in 1u32..=6 {
            let size = 1i32 << exp; // 2, 4, 8, 16, 32, 64
            let p = ScreenParams {
                size,
                dot_radius: 1,
                ..ScreenParams::default()
            };
            assert!(p.validate().is_ok(), "size={size} should be valid");
        }
    }

    #[test]
    fn spot_ncomps_matches_cpp() {
        // SPOT_NCOMPS = 4, matching #define SPOT_NCOMPS 4 in SplashTypes.h.
        assert_eq!(SPOT_NCOMPS, 4);
    }

    #[test]
    fn aa_size_matches_cpp() {
        // splashAASize = 4 in SplashTypes.h.
        assert_eq!(AA_SIZE, 4);
    }

    #[test]
    fn max_curve_splits_matches_cpp() {
        // splashMaxCurveSplits = 1 << 10 = 1024 in SplashXPath.h.
        assert_eq!(MAX_CURVE_SPLITS, 1 << 10);
    }

    #[test]
    fn bezier_circle_matches_cpp() {
        // bezierCircle = 0.55228475 defined in Splash.cc.
        // Check to 7 significant figures.
        assert!((BEZIER_CIRCLE - 0.552_284_75_f64).abs() < 1e-9);
    }
}
