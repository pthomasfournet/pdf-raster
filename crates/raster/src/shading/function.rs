//! Function-based shading pattern.
//!
//! A PDF Type 1 shading applies an arbitrary function `f(x, y) → colour`.
//! The function is supplied as a trait object from `pdf_bridge`; this module
//! wraps it in the [`Pattern`] interface.
//!
//! The function receives pixel centres in device space and returns an RGB
//! triple already in device colour space.  Any colour-space conversion is the
//! caller's responsibility — this type only forwards the closure's output.
//!
//! # Panics
//!
//! The closure itself must not panic.  Panics inside the closure propagate
//! through `fill_span` to the rasterizer caller.

use crate::pipe::Pattern;

/// A shading pattern driven by an arbitrary `f(x, y) → [R, G, B]` closure.
///
/// The closure is boxed so it can capture ICC profiles, LUTs, or any other
/// per-page state from `pdf_bridge`.  It must be `Send + Sync` because
/// [`Pattern`] is used across rayon threads.
pub struct FunctionPattern {
    func: Box<dyn Fn(f64, f64) -> [u8; 3] + Send + Sync>,
}

impl FunctionPattern {
    /// Create a function-based pattern from a closure `f(x, y) → [R, G, B]`.
    ///
    /// `x` and `y` are the pixel-centre coordinates in device space.
    #[must_use]
    pub fn new<F>(func: F) -> Self
    where
        F: Fn(f64, f64) -> [u8; 3] + Send + Sync + 'static,
    {
        Self { func: Box::new(func) }
    }
}

impl Pattern for FunctionPattern {
    fn fill_span(&self, y: i32, x0: i32, x1: i32, out: &mut [u8]) {
        // out.len() == (x1-x0+1)*3 is an invariant guaranteed by render_span.
        let mut off = 0usize;
        for x in x0..=x1 {
            let rgb = (self.func)(f64::from(x), f64::from(y));
            out[off]     = rgb[0];
            out[off + 1] = rgb[1];
            out[off + 2] = rgb[2];
            off += 3;
        }
    }

    fn is_static_color(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encodes_x_coordinate_in_output() {
        // Each pixel's R/G/B equals its x coordinate (mod 256).
        let p = FunctionPattern::new(|x, _y| {
            let v = x as u8;
            [v, v, v]
        });
        let mut out = vec![0u8; 4 * 3];
        p.fill_span(0, 10, 13, &mut out);
        for i in 0..4usize {
            let expected = (10 + i) as u8;
            assert_eq!(out[i * 3],     expected, "pixel {i} R");
            assert_eq!(out[i * 3 + 1], expected, "pixel {i} G");
            assert_eq!(out[i * 3 + 2], expected, "pixel {i} B");
        }
    }

    #[test]
    fn constant_function_fills_uniform_color() {
        let p = FunctionPattern::new(|_, _| [128, 64, 32]);
        let mut out = vec![0u8; 3 * 3];
        p.fill_span(5, 0, 2, &mut out);
        for i in 0..3usize {
            assert_eq!(out[i * 3],     128, "pixel {i} R");
            assert_eq!(out[i * 3 + 1], 64,  "pixel {i} G");
            assert_eq!(out[i * 3 + 2], 32,  "pixel {i} B");
        }
    }

    #[test]
    fn encodes_y_coordinate_in_output() {
        let p = FunctionPattern::new(|_x, y| [0, 0, y as u8]);
        let mut out = [0u8; 3];
        p.fill_span(42, 0, 0, &mut out);
        assert_eq!(out[2], 42, "blue channel should encode y");
    }

    #[test]
    fn single_pixel_span_writes_three_bytes() {
        let p = FunctionPattern::new(|_, _| [1, 2, 3]);
        let mut out = [0u8; 3];
        p.fill_span(0, 7, 7, &mut out);
        assert_eq!(out, [1, 2, 3]);
    }
}
