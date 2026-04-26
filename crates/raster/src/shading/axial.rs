//! Axial (linear) gradient pattern — `SplashAxialPattern::getColor`.
//!
//! Colour at pixel (x, y) is determined by projecting onto the gradient axis
//! and linearly interpolating between `color0` and `color1`.
//!
//! # Out-of-range pixels
//!
//! When `extend_start` / `extend_end` is `false` and the pixel projects outside
//! the `[t0, t1]` range, `fill_span` writes black (`[0,0,0]`).  This is only
//! correct when the caller clips the shading to its bounding path — which
//! `shaded_fill` guarantees.  Do not use this pattern without a bounding clip.

use super::lerp_color;
use crate::pipe::Pattern;

/// Linear gradient between two points in device space.
///
/// `t = dot(p - p0, axis) / |axis|²`, clamped to `[t0, t1]`
/// (or extended to the nearest endpoint when `extend_start` / `extend_end` is set).
pub struct AxialPattern {
    color0:       [u8; 3],
    color1:       [u8; 3],
    ax:           f64,
    ay:           f64,
    p0x:          f64,
    p0y:          f64,
    /// `1 / (ax² + ay²)`; zero when the axis is degenerate (zero-length).
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
    /// - `t0`, `t1`: parameter range mapping to `color0` / `color1`.
    ///   May be inverted (`t0 > t1`).
    /// - `extend_start` / `extend_end`: extend colour beyond axis endpoints.
    ///
    /// # Degenerate case
    ///
    /// When `p0 == p1` (zero-length axis) every pixel returns `None` from
    /// `t_for`, so `fill_span` writes zeros.  This matches poppler's behaviour.
    #[must_use]
    #[expect(clippy::too_many_arguments, reason = "mirrors PDF shading dict: 2 colors + 2 points + t range + 2 extend flags")]
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

    /// Compute the gradient parameter `t ∈ [t0, t1]` for pixel `(x, y)`.
    ///
    /// Returns `None` when the axis is degenerate or the pixel is outside the
    /// gradient range and extension is disabled.
    fn t_for(&self, x: i32, y: i32) -> Option<f64> {
        if self.inv_len_sq == 0.0 {
            return None;
        }
        let dx = f64::from(x) - self.p0x;
        let dy = f64::from(y) - self.p0y;
        // Project onto axis and map to [t0, t1].
        let t_raw = dx.mul_add(self.ax, dy * self.ay) * self.inv_len_sq;
        let t = t_raw.mul_add(self.t1 - self.t0, self.t0);

        // Handle both t0 < t1 and t0 > t1 (inverted gradient).
        let (lo, hi) = if self.t0 <= self.t1 { (self.t0, self.t1) } else { (self.t1, self.t0) };
        if t < lo {
            if self.extend_start { Some(self.t0) } else { None }
        } else if t > hi {
            if self.extend_end { Some(self.t1) } else { None }
        } else {
            Some(t)
        }
    }
}

impl Pattern for AxialPattern {
    fn fill_span(&self, y: i32, x0: i32, x1: i32, out: &mut [u8]) {
        // out.len() == (x1-x0+1)*3 is an invariant guaranteed by render_span.
        let t_span = self.t1 - self.t0;
        let mut off = 0usize;
        for x in x0..=x1 {
            if let Some(t) = self.t_for(x, y) {
                // Normalise t to [0,1] regardless of t0/t1 ordering.
                let frac = if t_span.abs() < f64::EPSILON {
                    0_u32
                } else {
                    #[expect(clippy::cast_sign_loss, reason = "value clamped to 0.0..=256.0")]
                    #[expect(clippy::cast_possible_truncation, reason = "value ≤ 256")]
                    { (((t - self.t0) / t_span).clamp(0.0, 1.0) * 256.0) as u32 }
                };
                lerp_color(self.color0, self.color1, frac, &mut out[off..off + 3]);
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
    fn outside_no_extend_writes_zero() {
        let p = make_axial(false);
        let mut out = [0u8; 3];
        p.fill_span(0, -1, -1, &mut out);
        assert_eq!(out, [0, 0, 0], "before start with no-extend should write zero");
        p.fill_span(0, 9, 9, &mut out);
        assert_eq!(out, [0, 0, 0], "after end with no-extend should write zero");
    }

    #[test]
    fn outside_extend_clamps_to_endpoints() {
        let p = make_axial(true);
        let mut out = [0u8; 3];
        p.fill_span(0, -5, -5, &mut out);
        assert_eq!(out[0], 0, "before start with extend should clamp to color0");
        p.fill_span(0, 100, 100, &mut out);
        assert_eq!(out[0], 255, "after end with extend should clamp to color1");
    }

    #[test]
    fn degenerate_axis_writes_zeros() {
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
    fn gradient_increases_left_to_right() {
        let p = make_axial(true);
        let mut out = vec![0u8; 5 * 3];
        p.fill_span(0, 0, 4, &mut out);
        assert!(out[0] < out[3], "gradient R should increase left-to-right");
        assert!(out[3] < out[6], "gradient R should increase left-to-right");
    }

    #[test]
    fn inverted_t_range_reverses_gradient() {
        // t0=1 t1=0: color0 at right end, color1 at left end.
        let p = AxialPattern::new(
            [255, 0, 0], [0, 0, 255],
            0.0, 0.0, 4.0, 0.0,
            1.0, 0.0,   // inverted
            true, true,
        );
        let mut start = [0u8; 3];
        let mut end   = [0u8; 3];
        p.fill_span(0, 0, 0, &mut start);
        p.fill_span(0, 4, 4, &mut end);
        // t=1.0 at x=0 → color0=red; t=0.0 at x=4 → color1=blue.
        assert!(start[0] > 200, "x=0 should be near red (color0)");
        assert!(end[2]   > 200, "x=4 should be near blue (color1)");
    }
}
