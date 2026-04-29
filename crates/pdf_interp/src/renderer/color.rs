//! PDF colour space and colour value conversions.
//!
//! Maps PDF colour operator values (f64 in [0,1]) to raw u8 pixel components
//! understood by the raster crate.
//!
//! Pure conversion functions (`gray_to_u8`, `rgb_to_bytes`, `cmyk_to_rgb_bytes`)
//! live in `color::convert` and are re-exported from there.

pub use color::convert::{cmyk_to_rgb_bytes, gray_to_u8, rgb_to_bytes};

/// Current fill or stroke colour, in a form the raster pipeline accepts.
///
/// We target `Bitmap<Rgb8>` throughout, so all colours are stored as 3-byte
/// RGB values.  `PipeSrc::Solid` takes `&[u8]` and asserts its length equals
/// the pixel component count, so passing a 1-byte gray value would panic.
///
/// `gray()` expands to equal R=G=B channels; `cmyk()` converts via naive
/// PDF §10.3.3.  Full ICC-profile conversion is a later phase.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RasterColor {
    bytes: [u8; 3],
}

impl RasterColor {
    /// Grey [0, 1] — expanded to equal RGB components.
    #[must_use]
    pub fn gray(g: f64) -> Self {
        let v = gray_to_u8(g);
        Self { bytes: [v, v, v] }
    }

    /// RGB [0, 1] each.
    #[must_use]
    pub fn rgb(r: f64, g: f64, b: f64) -> Self {
        Self {
            bytes: rgb_to_bytes(r, g, b),
        }
    }

    /// CMYK [0, 1] each — converted to RGB via naive PDF §10.3.3 (no ICC).
    #[must_use]
    pub fn cmyk(c: f64, m: f64, y: f64, k: f64) -> Self {
        Self {
            bytes: cmyk_to_rgb_bytes(c, m, y, k),
        }
    }

    /// Raw bytes slice for `PipeSrc::Solid`.
    ///
    /// Always 3 bytes (RGB), matching `Rgb8::BYTES`.
    #[must_use]
    pub const fn as_slice(&self) -> &[u8] {
        &self.bytes
    }
}

impl Default for RasterColor {
    /// Default fill/stroke colour is black (PDF §8.4.4 initial value).
    fn default() -> Self {
        Self::gray(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raster_color_rgb_slice() {
        let c = RasterColor::rgb(1.0, 0.0, 0.0);
        assert_eq!(c.as_slice(), &[255, 0, 0]);
    }

    #[test]
    fn raster_color_gray_expands_to_rgb() {
        // Gray must always produce a 3-byte slice for Rgb8 pipeline compatibility.
        let black = RasterColor::gray(0.0);
        assert_eq!(black.as_slice(), &[0, 0, 0]);
        let white = RasterColor::gray(1.0);
        assert_eq!(white.as_slice(), &[255, 255, 255]);
    }

    #[test]
    fn raster_color_default_is_black() {
        let c = RasterColor::default();
        assert_eq!(c.as_slice(), &[0, 0, 0]);
    }
}
