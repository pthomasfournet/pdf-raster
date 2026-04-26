//! Axial (linear) gradient pattern — `SplashAxialPattern::getColor`.
//!
//! Colour at pixel (x, y) is determined by projecting onto the gradient axis
//! and linearly interpolating between `color0` and `color1`.

use crate::pipe::Pattern;
use color::convert::lerp_u8;

/// Linear gradient between two points in device space.
///
/// Matches `SplashAxialPattern::getColor`:
/// `t = dot(p - p0, axis) / |axis|²`, clamped to [t0, t1] (or extended).
pub struct AxialPattern {
    color0:       [u8; 3],
    color1:       [u8; 3],
    ax:           f64,
    ay:           f64,
    p0x:          f64,
    p0y:          f64,
    inv_len_sq:   f64,
    t0:           f64,
    t1:           f64,
    extend_start: bool,
    extend_end:   bool,
}

impl AxialPattern {
    /// Create an axial gradient.
    ///
    /// - `color0`, `color1`: RGB endpoints.
    /// - `(p0x, p0y)`, `(p1x, p1y)`: axis endpoints in device pixels.
    /// - `t0`, `t1`: parameter range (typically 0.0 and 1.0).
    /// - `extend_start` / `extend_end`: extend colour beyond axis endpoints.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        color0:       [u8; 3],
        color1:       [u8; 3],
        p0x: f64, p0y: f64,
        p1x: f64, p1y: f64,
        t0: f64,  t1: f64,
        extend_start: bool,
        extend_end:   bool,
    ) -> Self {
        let ax = p1x - p0x;
        let ay = p1y - p0y;
        let len_sq = ax.mul_add(ax, ay * ay);
        let inv_len_sq = if len_sq > 0.0 { 1.0 / len_sq } else { 0.0 };
        Self { color0, color1, ax, ay, p0x, p0y, inv_len_sq, t0, t1, extend_start, extend_end }
    }

    /// Compute the gradient parameter `t` for pixel centre `(x, y)`.
    fn t_for(&self, x: i32, y: i32) -> Option<f64> {
        if self.inv_len_sq == 0.0 {
            return None;
        }
        let dx = f64::from(x) - self.p0x;
        let dy = f64::from(y) - self.p0y;
        let t_raw = dx.mul_add(self.ax, dy * self.ay) * self.inv_len_sq;

        let t_range = self.t1 - self.t0;
        let t = t_raw.mul_add(t_range, self.t0);

        if t < self.t0 {
            if self.extend_start { Some(self.t0) } else { None }
        } else if t > self.t1 {
            if self.extend_end { Some(self.t1) } else { None }
        } else {
            Some(t)
        }
    }
}

impl Pattern for AxialPattern {
    fn fill_span(&self, y: i32, x0: i32, x1: i32, out: &mut [u8]) {
        // out.len() == (x1-x0+1)*3 is an invariant guaranteed by render_span.
        let mut off = 0usize;
        for x in x0..=x1 {
            if let Some(t) = self.t_for(x, y) {
                let t_norm = if (self.t1 - self.t0).abs() < f64::EPSILON {
                    0.0
                } else {
                    ((t - self.t0) / (self.t1 - self.t0)).clamp(0.0, 1.0)
                };
                #[expect(clippy::cast_sign_loss, reason = "t_norm ∈ [0,1]")]
                #[expect(clippy::cast_possible_truncation, reason = "t_norm * 256 ≤ 256")]
                let frac = (t_norm * 256.0) as u32;
                out[off]     = lerp_u8(self.color0[0], self.color1[0], frac);
                out[off + 1] = lerp_u8(self.color0[1], self.color1[1], frac);
                out[off + 2] = lerp_u8(self.color0[2], self.color1[2], frac);
            } else {
                out[off]     = 0;
                out[off + 1] = 0;
                out[off + 2] = 0;
            }
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

    fn make_axial(extend: bool) -> AxialPattern {
        AxialPattern::new(
            [0, 0, 0], [255, 255, 255],
            0.0, 0.0,
            8.0, 0.0,
            0.0, 1.0,
            extend, extend,
        )
    }

    #[test]
    fn midpoint_is_grey() {
        let p = make_axial(true);
        let mut out = [0u8; 3];
        p.fill_span(0, 4, 4, &mut out);
        assert!(out[0] >= 126 && out[0] <= 129, "mid R={}", out[0]);
    }

    #[test]
    fn start_is_black() {
        let p = make_axial(true);
        let mut out = [0u8; 3];
        p.fill_span(0, 0, 0, &mut out);
        assert_eq!(out[0], 0, "start should be black");
    }

    #[test]
    fn end_is_white() {
        let p = make_axial(true);
        let mut out = [0u8; 3];
        p.fill_span(0, 8, 8, &mut out);
        assert_eq!(out[0], 255, "end should be white");
    }

    #[test]
    fn outside_no_extend_is_zero() {
        let p = make_axial(false);
        let mut out = [0u8; 3];
        p.fill_span(0, -1, -1, &mut out);
        assert_eq!(out[0], 0, "before start with no-extend should be zero");
        p.fill_span(0, 9, 9, &mut out);
        assert_eq!(out[0], 0, "after end with no-extend should be zero");
    }

    #[test]
    fn outside_extend_clamps_to_endpoints() {
        let p = make_axial(true);
        let mut out = [0u8; 3];
        p.fill_span(0, -5, -5, &mut out);
        assert_eq!(out[0], 0, "before start with extend should be color0 (black)");
        p.fill_span(0, 100, 100, &mut out);
        assert_eq!(out[0], 255, "after end with extend should be color1 (white)");
    }

    #[test]
    fn degenerate_axis_returns_transparent() {
        let p = AxialPattern::new(
            [255, 0, 0], [0, 255, 0],
            5.0, 5.0, 5.0, 5.0,
            0.0, 1.0, false, false,
        );
        let mut out = [42u8; 3];
        p.fill_span(5, 5, 5, &mut out);
        assert_eq!(out, [0, 0, 0], "degenerate axis should write zeros");
    }

    #[test]
    fn span_length_matches_output() {
        let p = make_axial(true);
        let mut out = vec![0u8; 5 * 3];
        p.fill_span(0, 0, 4, &mut out);
        assert!(out[0] < out[3], "gradient should increase");
        assert!(out[3] < out[6], "gradient should increase");
    }
}
