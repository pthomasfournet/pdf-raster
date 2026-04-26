//! Stroke-adjust hint application.
//!
//! Mirrors `SplashXPathAdjust` and `SplashXPath::strokeAdjust()` from
//! `splash/SplashXPath.cc`.
//!
//! ## What stroke adjustment does
//!
//! PDF stroke adjustment snaps near-axis-aligned path segments to integer
//! pixel boundaries so that adjacent stroked rectangles share a pixel edge
//! rather than leaving a half-pixel gap. This is controlled by the PDF
//! `strokeAdjust` graphics state parameter.
//!
//! Each [`XPathAdjust`] describes three "snap windows" (around x0, xm, x1)
//! with ±0.01 tolerance. Any transformed coordinate that falls inside a window
//! is replaced by the corresponding snapped target.

use crate::types::splash_round;

/// An axis-aligned stroke-adjust hint, derived from a [`StrokeAdjustHint`]
/// after path transformation.
///
/// Matches `SplashXPathAdjust` in `splash/SplashXPath.cc`.
#[derive(Clone, Debug)]
pub struct XPathAdjust {
    /// Range of path-point indices to adjust.
    pub first_pt: usize,
    pub last_pt: usize,
    /// `true` → adjust the x-coordinate; `false` → adjust the y-coordinate.
    pub vert: bool,
    // Snap windows: a coordinate v is snapped to target if v ∈ (xa, xb).
    pub x0a: f64,
    pub x0b: f64,
    pub x0: f64,
    pub xma: f64,
    pub xmb: f64,
    pub xm: f64,
    pub x1a: f64,
    pub x1b: f64,
    pub x1: f64,
}

impl XPathAdjust {
    /// Construct an adjust record for two endpoint values `adj0 ≤ adj1`.
    ///
    /// `adjust_lines` and `line_pos_i` implement the same special-case logic as
    /// in `SplashXPath` constructor: when the two rounded endpoints coincide and
    /// `adjust_lines` is true, the span is expanded to [line_pos_i, line_pos_i+1].
    pub fn new(
        first_pt: usize,
        last_pt: usize,
        vert: bool,
        adj0: f64,
        adj1: f64,
        adjust_lines: bool,
        line_pos_i: i32,
    ) -> Self {
        let mid = (adj0 + adj1) * 0.5;
        let mut r0 = splash_round(adj0) as f64;
        let mut r1 = splash_round(adj1) as f64;
        if r0 == r1 {
            if adjust_lines {
                r0 = line_pos_i as f64;
                r1 = line_pos_i as f64 + 1.0;
            } else {
                r1 += 1.0;
            }
        }
        Self {
            first_pt,
            last_pt,
            vert,
            x0a: adj0 - 0.01,
            x0b: adj0 + 0.01,
            x0: r0,
            xma: mid - 0.01,
            xmb: mid + 0.01,
            xm: (r0 + r1) * 0.5,
            x1a: adj1 - 0.01,
            x1b: adj1 + 0.01,
            x1: r1 - 0.01,
        }
    }
}

/// Apply a stroke-adjust hint to a single point `(x, y)`.
///
/// If `adj.vert` is true, the x-coordinate is examined; otherwise the
/// y-coordinate. If the relevant coordinate falls within any of the three snap
/// windows, it is replaced by the corresponding snapped target.
///
/// Matches `SplashXPath::strokeAdjust()` in `SplashXPath.cc`.
#[inline]
pub fn stroke_adjust(adj: &XPathAdjust, x: &mut f64, y: &mut f64) {
    let v = if adj.vert { x } else { y };
    if *v > adj.x0a && *v < adj.x0b {
        *v = adj.x0;
    } else if *v > adj.xma && *v < adj.xmb {
        *v = adj.xm;
    } else if *v > adj.x1a && *v < adj.x1b {
        *v = adj.x1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snaps_to_x0() {
        let adj = XPathAdjust::new(0, 1, true, 1.0, 3.0, false, 0);
        let mut x = 1.005;
        let mut y = 0.0;
        stroke_adjust(&adj, &mut x, &mut y);
        assert!((x - 1.0).abs() < 1e-10, "x={x}");
    }

    #[test]
    fn snaps_to_xm() {
        let adj = XPathAdjust::new(0, 1, true, 1.0, 3.0, false, 0);
        let mut x = 2.0; // midpoint of [1,3]
        let mut y = 0.0;
        stroke_adjust(&adj, &mut x, &mut y);
        assert!((x - 2.0).abs() < 1e-10, "x={x}"); // xm = (1+3)/2 = 2.0
    }

    #[test]
    fn snaps_to_x1() {
        let adj = XPathAdjust::new(0, 1, true, 1.0, 3.0, false, 0);
        let mut x = 3.005;
        let mut y = 0.0;
        stroke_adjust(&adj, &mut x, &mut y);
        // x1 = 3.0 - 0.01 = 2.99
        assert!((x - 2.99).abs() < 1e-10, "x={x}");
    }

    #[test]
    fn no_snap_outside_windows() {
        let adj = XPathAdjust::new(0, 1, true, 1.0, 3.0, false, 0);
        let mut x = 5.0;
        let mut y = 0.0;
        stroke_adjust(&adj, &mut x, &mut y);
        assert!((x - 5.0).abs() < 1e-10);
    }

    #[test]
    fn horizontal_adjusts_y() {
        let adj = XPathAdjust::new(0, 1, false, 2.0, 4.0, false, 0);
        let mut x = 0.0;
        let mut y = 2.005;
        stroke_adjust(&adj, &mut x, &mut y);
        assert!((y - 2.0).abs() < 1e-10, "y={y}");
        assert!((x - 0.0).abs() < 1e-10); // x unchanged
    }
}
