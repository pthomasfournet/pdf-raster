//! Raster-local enums and workspace-wide constants.
//!
//! Re-exports all [`color`] public types so that downstream modules within
//! this crate only need `use crate::types::*`.

pub use color::{
    convert::*, AnyColor, Cmyk8, DeviceN8, Gray8, Pixel, PixelMode, Rgb8, Rgba8, TransferLut,
    NCOMPS,
};

// ── Rasterizer constants (from SplashTypes.h) ─────────────────────────────────

/// Supersampling factor for anti-aliasing (splashAASize = 4).
/// The AA buffer is `bitmap_width × AA_SIZE` pixels wide and `AA_SIZE` rows tall.
pub const AA_SIZE: i32 = 4;

/// Maximum number of De Casteljau subdivisions for Bezier flattening
/// (splashMaxCurveSplits = 1024).
pub const MAX_CURVE_SPLITS: i32 = 1024;

/// Control-point ratio for approximating a quarter-circle with a cubic Bezier.
/// Value: 4 * (√2 − 1) / 3 ≈ 0.55228475  (bezierCircle in Splash.cc).
pub const BEZIER_CIRCLE: f64 = 0.55228475;

/// Number of spot color channels in DeviceN8 (SPOT_NCOMPS = 4).
pub const SPOT_NCOMPS: usize = 4;

// ── Enums matching SplashTypes.h ─────────────────────────────────────────────

/// Thin-line rendering treatment (SplashThinLineMode).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum ThinLineMode {
    #[default]
    Default, // preserve hairline width
    Solid, // render as solid 1-pixel line
    Shape, // use shape anti-aliasing
}

/// Stroke line-cap style (SplashLineCap).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum LineCap {
    #[default]
    Butt, // square end at endpoint
    Round,      // semicircle beyond endpoint
    Projecting, // square extending half line-width beyond endpoint
}

/// Stroke line-join style (SplashLineJoin).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum LineJoin {
    #[default]
    Miter, // sharp corner up to miter limit
    Round, // rounded join
    Bevel, // cut-off corner
}

/// Halftone screen type (SplashScreenType).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum ScreenType {
    #[default]
    Dispersed, // Bayer-style dispersed dot (size 4×4 below 300 dpi)
    Clustered,           // clustered dot screen
    StochasticClustered, // stochastic clustered (64×64 at ≥ 300 dpi)
}

/// Parameters for constructing a [`HalftoneScreen`](crate::screen::HalftoneScreen).
#[derive(Copy, Clone, Debug)]
pub struct ScreenParams {
    pub kind: ScreenType,
    pub size: i32,       // matrix dimension (power of 2, ≥ 2)
    pub dot_radius: i32, // for StochasticClustered only
}

impl Default for ScreenParams {
    fn default() -> Self {
        // matches SplashScreen::defaultParams: { Dispersed, 2, 2 }
        Self {
            kind: ScreenType::Dispersed,
            size: 2,
            dot_radius: 2,
        }
    }
}
