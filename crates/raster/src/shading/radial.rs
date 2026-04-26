//! Radial gradient pattern — `SplashRadialPattern::getColor`.
//!
//! Solves the quadratic `|p - (c0 + t·(c1-c0))|² = (r0 + t·(r1-r0))²`
//! and takes the largest root in [t0, t1] (or the clamped endpoint when
//! `extend_start` / `extend_end` is set).  Matches poppler's `SplashRadialPattern`.

use crate::pipe::Pattern;
use color::convert::lerp_u8;

/// Gradient between two circles (centre `c0`, radius `r0`) and (`c1`, `r1`).
pub struct RadialPattern {
    color0:       [u8; 3],
    color1:       [u8; 3],
    c0x:          f64,
    c0y:          f64,
    // Δc = c1 - c0.
    dcx:          f64,
    dcy:          f64,
    r0:           f64,
    // Δr = r1 - r0.
    dr:           f64,
    t0:           f64,
    t1:           f64,
    extend_start: bool,
    extend_end:   bool,
    // `|Δc|² - Δr²` — the `t²` coefficient of the quadratic.
    a:            f64,
}

impl RadialPattern {
    /// Create a radial gradient.
    ///
    /// - `color0` / `color1`: RGB at inner / outer circle.
    /// - `(c0x, c0y, r0)`: inner circle centre and radius.
    /// - `(c1x, c1y, r1)`: outer circle centre and radius.
    /// - `t0`, `t1`: parameter range (typically 0.0 and 1.0).
    /// - `extend_start` / `extend_end`: extend colour beyond the gradient circles.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        color0: [u8; 3],
        color1: [u8; 3],
        c0x: f64, c0y: f64, r0: f64,
        c1x: f64, c1y: f64, r1: f64,
        t0: f64, t1: f64,
        extend_start: bool, extend_end: bool,
    ) -> Self {
        let dcx = c1x - c0x;
        let dcy = c1y - c0y;
        let dr  = r1 - r0;
        // a = |Δc|² - Δr²
        let a = dr.mul_add(-dr, dcx.mul_add(dcx, dcy * dcy));
        Self { color0, color1, c0x, c0y, dcx, dcy, r0, dr, t0, t1, extend_start, extend_end, a }
    }

    /// Solve for `t` at pixel `(px, py)`.
    ///
    /// Quadratic coefficients:
    /// - `a·t² + 2·b·t + c = 0`
    /// - `b = (p - c0)·Δc - r0·Δr`
    /// - `c = |p - c0|² - r0²`
    fn t_for(&self, xi: i32, yi: i32) -> Option<f64> {
        let rel_x = f64::from(xi) - self.c0x;
        let rel_y = f64::from(yi) - self.c0y;

        let b = self.r0.mul_add(-self.dr, rel_x.mul_add(self.dcx, rel_y * self.dcy));
        let c = self.r0.mul_add(-self.r0, rel_x.mul_add(rel_x, rel_y * rel_y));

        let t = if self.a.abs() < 1e-12 {
            // Linear: 2·b·t + c = 0
            if b.abs() < 1e-12 { return None; }
            -c / (2.0 * b)
        } else {
            let disc = b.mul_add(b, -(self.a * c));
            if disc < 0.0 { return None; }
            let sq = disc.sqrt();
            // Prefer larger root (poppler convention — picks the "outside" intersection).
            f64::max((-b + sq) / self.a, (-b - sq) / self.a)
        };

        if t < self.t0 {
            if self.extend_start { Some(self.t0) } else { None }
        } else if t > self.t1 {
            if self.extend_end { Some(self.t1) } else { None }
        } else {
            Some(t)
        }
    }
}

impl Pattern for RadialPattern {
    fn fill_span(&self, y: i32, x0: i32, x1: i32, out: &mut [u8]) {
        // out.len() == (x1-x0+1)*3 is an invariant guaranteed by render_span.
        let mut off = 0usize;
        for xi in x0..=x1 {
            if let Some(t) = self.t_for(xi, y) {
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

    fn make_concentric() -> RadialPattern {
        RadialPattern::new(
            [0, 0, 0], [255, 255, 255],
            4.0, 4.0, 0.0,
            4.0, 4.0, 4.0,
            0.0, 1.0,
            true, true,
        )
    }

    #[test]
    fn centre_is_color0() {
        let p = make_concentric();
        let mut out = [42u8; 3];
        p.fill_span(4, 4, 4, &mut out);
        assert!(out[0] < 10, "centre should be near-black, got {}", out[0]);
    }

    #[test]
    fn outer_ring_is_color1() {
        let p = make_concentric();
        let mut out = [0u8; 3];
        p.fill_span(4, 8, 8, &mut out);
        assert!(out[0] > 240, "outer ring should be near-white, got {}", out[0]);
    }

    #[test]
    fn pixel_on_inner_circle_is_color0() {
        let p = RadialPattern::new(
            [255, 0, 0], [0, 0, 255],
            4.0, 4.0, 2.0,
            4.0, 4.0, 6.0,
            0.0, 1.0,
            false, false,
        );
        let mut out = [0u8; 3];
        p.fill_span(4, 6, 6, &mut out);
        assert!(out[0] > 240, "inner circle should be near color0 red");
        assert!(out[2] < 20,  "inner circle should have near-zero blue");
    }

    #[test]
    fn no_real_intersection_outside_sphere_writes_zero() {
        let p = RadialPattern::new(
            [255, 0, 0], [0, 255, 0],
            0.0, 0.0, 1.0,
            10.0, 0.0, 1.0,
            0.0, 1.0,
            false, false,
        );
        let mut out = [42u8; 3];
        p.fill_span(100, 100, 100, &mut out);
        assert_eq!(out, [0, 0, 0], "no intersection should write zeros");
    }
}
