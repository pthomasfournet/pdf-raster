//! Tiling pattern source for the compositing pipeline.
//!
//! [`TiledPattern`] holds a pre-rasterised tile bitmap and implements
//! [`pipe::Pattern`] by repeating the tile across device space using modular
//! arithmetic.  One tile is computed once and then referenced immutably during
//! the fill/stroke operation.

use crate::pipe::Pattern;

/// A pre-rasterised tiling pattern.
///
/// The tile is a row-major RGB8 pixel buffer of `width × height` pixels,
/// anchored at `(phase_x, phase_y)` in device space.  `fill_span` tiles the
/// buffer by wrapping coordinates with `rem_euclid`.
///
/// # Invariants
///
/// - `width > 0` and `height > 0` (asserted in [`TiledPattern::new`])
/// - `pixels.len() == width * height * 3`
pub struct TiledPattern {
    /// Rasterised tile pixels (RGB8, row-major).
    pixels: Vec<u8>,
    /// Tile width in device pixels.
    width: i32,
    /// Tile height in device pixels.
    height: i32,
    /// X offset of the pattern origin in device space.
    ///
    /// Used to phase the tiling so that the pattern is anchored at the correct
    /// position as specified by the pattern matrix translation.
    phase_x: i32,
    /// Y offset of the pattern origin in device space.
    phase_y: i32,
}

impl TiledPattern {
    /// Construct a new [`TiledPattern`].
    ///
    /// # Panics
    ///
    /// Panics if `width` or `height` is ≤ 0, or if `pixels.len()` does not
    /// equal `width * height * 3`.
    #[must_use]
    #[expect(
        clippy::cast_sign_loss,
        reason = "width/height are asserted positive just above the cast"
    )]
    pub fn new(pixels: Vec<u8>, width: i32, height: i32, phase_x: i32, phase_y: i32) -> Self {
        assert!(
            width > 0,
            "TiledPattern: width must be positive, got {width}"
        );
        assert!(
            height > 0,
            "TiledPattern: height must be positive, got {height}"
        );
        let expected = width as usize * height as usize * 3;
        assert!(
            pixels.len() == expected,
            "TiledPattern: pixels.len() {} does not match width({width}) * height({height}) * 3 = {expected}",
            pixels.len()
        );
        Self {
            pixels,
            width,
            height,
            phase_x,
            phase_y,
        }
    }
}

impl Pattern for TiledPattern {
    #[expect(
        clippy::cast_sign_loss,
        reason = "rem_euclid guarantees non-negative results; safe cast to usize"
    )]
    fn fill_span(&self, y: i32, x0: i32, x1: i32, out: &mut [u8]) {
        let ty = (y - self.phase_y).rem_euclid(self.height) as usize;
        let w = self.width as usize;
        let row_start = ty * w * 3;
        let row = &self.pixels[row_start..row_start + w * 3];

        let mut out_pos = 0usize;
        for x in x0..=x1 {
            let tx = (x - self.phase_x).rem_euclid(self.width) as usize;
            out[out_pos] = row[tx * 3];
            out[out_pos + 1] = row[tx * 3 + 1];
            out[out_pos + 2] = row[tx * 3 + 2];
            out_pos += 3;
        }
    }
}
