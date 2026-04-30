//! Radial gradient pattern (PDF §8.7.4.5, type 3).
//!
//! Solves the quadratic `|p - (c0 + t·(c1-c0))|² = (r0 + t·(r1-r0))²`
//! and takes the largest real root (the "outside" intersection, per PDF §8.7.4.5).
//! The root is then clamped to `[t0, t1]` or rejected when extension is off.
//!
//! # Out-of-range pixels
//!
//! When the quadratic has no real roots, or the solution falls outside
//! `[t0, t1]` and extension is disabled, `fill_span` writes `[0, 0, 0]`.
//! As with [`axial`], this is only correct when the caller clips to the
//! shading bounding path.
//!
//! [`axial`]: super::axial

use super::lerp_color;
use crate::pipe::Pattern;

/// Gradient between two circles (centre `c0`, radius `r0`) and (`c1`, `r1`).
pub struct RadialPattern {
    color0: [u8; 3],
    color1: [u8; 3],
    c0x: f64,
    c0y: f64,
    /// Δc = c1 - c0.
    dcx: f64,
    dcy: f64,
    r0: f64,
    /// Δr = r1 - r0.
    dr: f64,
    t0: f64,
    t1: f64,
    extend_start: bool,
    extend_end: bool,
    /// `|Δc|² - Δr²` — the `t²` coefficient of the quadratic.
    /// Zero when the gradient is linear (conic section degenerate).
    a: f64,
}

impl RadialPattern {
    /// Create a radial gradient.
    ///
    /// - `color0` / `color1`: RGB at inner / outer circle.
    /// - `(c0x, c0y, r0)`: inner circle centre and radius (≥ 0).
    /// - `(c1x, c1y, r1)`: outer circle centre and radius (≥ 0).
    /// - `t0`, `t1`: parameter range mapping to `color0` / `color1`.
    ///   May be inverted (`t0 > t1`).
    /// - `extend_start` / `extend_end`: extend colour beyond the gradient circles.
    #[must_use]
    #[expect(
        clippy::too_many_arguments,
        reason = "mirrors PDF shading dict: 2 colors + 2 circles (cx,cy,r each) + t range + 2 extend flags"
    )]
    pub fn new(
        color0: [u8; 3],
        color1: [u8; 3],
        c0x: f64,
        c0y: f64,
        r0: f64,
        c1x: f64,
        c1y: f64,
        r1: f64,
        t0: f64,
        t1: f64,
        extend_start: bool,
        extend_end: bool,
    ) -> Self {
        let dcx = c1x - c0x;
        let dcy = c1y - c0y;
        let dr = r1 - r0;
        // a = |Δc|² - Δr²
        let a = dr.mul_add(-dr, dcx.mul_add(dcx, dcy * dcy));
        Self {
            color0,
            color1,
            c0x,
            c0y,
            dcx,
            dcy,
            r0,
            dr,
            t0,
            t1,
            extend_start,
            extend_end,
            a,
        }
    }

    /// Solve for `t` at pixel `(xi, yi)`.
    ///
    /// Quadratic `a·t² + 2·b·t + c = 0` where:
    /// - `b = (p - c0)·Δc - r0·Δr`
    /// - `c = |p - c0|² - r0²`
    ///
    /// Returns `None` when no real solution exists in the gradient range.
    fn t_for(&self, xi: i32, yi: i32) -> Option<f64> {
        let rel_x = f64::from(xi) - self.c0x;
        let rel_y = f64::from(yi) - self.c0y;

        let b = self
            .r0
            .mul_add(-self.dr, rel_x.mul_add(self.dcx, rel_y * self.dcy));
        let c = self
            .r0
            .mul_add(-self.r0, rel_x.mul_add(rel_x, rel_y * rel_y));

        let t = if self.a.abs() < 1e-12 {
            // Linear equation: 2·b·t + c = 0.
            if b.abs() < 1e-12 {
                return None; // fully degenerate — no solution
            }
            -c / (b + b)
        } else {
            let disc = b.mul_add(b, -(self.a * c));
            if disc < 0.0 {
                return None; // no real intersection
            }
            let sq = disc.sqrt();
            // Take the larger root — the "outside" intersection (PDF §8.7.4.5).
            f64::max((-b + sq) / self.a, (-b - sq) / self.a)
        };

        // Handle both t0 < t1 and t0 > t1 (inverted gradient).
        let (lo, hi) = if self.t0 <= self.t1 {
            (self.t0, self.t1)
        } else {
            (self.t1, self.t0)
        };
        if t < lo {
            if self.extend_start {
                Some(self.t0)
            } else {
                None
            }
        } else if t > hi {
            if self.extend_end { Some(self.t1) } else { None }
        } else {
            Some(t)
        }
    }
}

impl Pattern for RadialPattern {
    fn fill_span(&self, y: i32, x0: i32, x1: i32, out: &mut [u8]) {
        // out.len() == (x1-x0+1)*3 is an invariant guaranteed by render_span.
        let t_span = self.t1 - self.t0;
        let mut off = 0usize;
        for xi in x0..=x1 {
            if let Some(t) = self.t_for(xi, y) {
                let frac = if t_span.abs() < f64::EPSILON {
                    0_u32
                } else {
                    #[expect(clippy::cast_sign_loss, reason = "value clamped to 0.0..=256.0")]
                    #[expect(clippy::cast_possible_truncation, reason = "value ≤ 256")]
                    {
                        (((t - self.t0) / t_span).clamp(0.0, 1.0) * 256.0) as u32
                    }
                };
                lerp_color(self.color0, self.color1, frac, &mut out[off..off + 3]);
            } else {
                out[off] = 0;
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

    /// Concentric radial: c0=c1=(4,4), r0=0, r1=4.
    fn make_concentric() -> RadialPattern {
        RadialPattern::new(
            [0, 0, 0],
            [255, 255, 255],
            4.0,
            4.0,
            0.0,
            4.0,
            4.0,
            4.0,
            0.0,
            1.0,
            true,
            true,
        )
    }

    #[test]
    fn centre_is_color0() {
        let p = make_concentric();
        let mut out = [42u8; 3];
        p.fill_span(4, 4, 4, &mut out);
        // Distance 0 from centre → t → 0 → color0 = black.
        assert!(out[0] < 10, "centre should be near-black, got {}", out[0]);
    }

    #[test]
    fn outer_ring_is_color1() {
        let p = make_concentric();
        let mut out = [0u8; 3];
        p.fill_span(4, 8, 8, &mut out); // distance 4 = r1 → t = 1 → color1 = white
        assert!(
            out[0] > 240,
            "outer ring should be near-white, got {}",
            out[0]
        );
    }

    #[test]
    fn pixel_on_inner_circle_maps_to_color0() {
        // r0=2: pixel at distance 2 from centre → t=0 → color0.
        let p = RadialPattern::new(
            [255, 0, 0],
            [0, 0, 255],
            4.0,
            4.0,
            2.0,
            4.0,
            4.0,
            6.0,
            0.0,
            1.0,
            false,
            false,
        );
        let mut out = [0u8; 3];
        // (6, 4) is exactly 2 units from (4,4) — on the inner circle.
        p.fill_span(4, 6, 6, &mut out);
        assert!(out[0] > 240, "inner circle should be near color0 (red)");
        assert!(out[2] < 20, "inner circle should have near-zero blue");
    }

    #[test]
    fn no_real_intersection_writes_zeros() {
        // Eccentric gradient where a pixel far off-axis has no real solution.
        let p = RadialPattern::new(
            [255, 0, 0],
            [0, 255, 0],
            0.0,
            0.0,
            1.0,
            10.0,
            0.0,
            1.0,
            0.0,
            1.0,
            false,
            false,
        );
        let mut out = [42u8; 3]; // non-zero sentinel
        p.fill_span(100, 100, 100, &mut out);
        assert_eq!(out, [0, 0, 0], "no intersection should write zeros");
    }

    #[test]
    fn degenerate_a_linear_fallback() {
        // When |Δc|² == Δr², a=0 and we use the linear path.
        // c0=(0,0,r0=0), c1=(3,4,r1=5): |Δc|²=25, Δr=5, Δr²=25 → a=0.
        let p = RadialPattern::new(
            [0, 0, 0],
            [255, 255, 255],
            0.0,
            0.0,
            0.0,
            3.0,
            4.0,
            5.0,
            0.0,
            1.0,
            true,
            true,
        );
        let mut out = [0u8; 15]; // 5 pixels × 3 bytes
        // Must not panic.
        p.fill_span(0, 0, 4, &mut out);
    }

    #[test]
    fn extend_clamps_outside_to_endpoints() {
        let p = make_concentric();
        let mut out = [42u8; 3];
        // Distance 10 > r1=4; with extend_end=true should clamp to color1 (white).
        p.fill_span(4, 14, 14, &mut out);
        assert!(
            out[0] > 240,
            "beyond outer with extend should clamp to color1 (white)"
        );
    }
}
