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
//! with ±0.01 tolerance. Any transformed coordinate that falls **strictly
//! inside** a window is replaced by the corresponding snapped target.
//!
//! ## Open-interval semantics
//!
//! The snap windows are **open** intervals: a coordinate exactly at a boundary
//! value (e.g. `v == adj0 - 0.01`) is **not** snapped. See [`stroke_adjust`].

use crate::types::splash_round;

/// An axis-aligned stroke-adjust hint, derived from a `StrokeAdjustHint`
/// (see `crate::path::stroke`) after path transformation.
///
/// Matches `SplashXPathAdjust` in `splash/SplashXPath.cc`.
///
/// ## Snap-window fields
///
/// The three snap windows are open intervals; a coordinate `v` is replaced by
/// the target only when `v ∈ (xa, xb)` (strictly inside):
///
/// | Fields | Meaning |
/// |--------|---------|
/// | `x0a`, `x0b`, `x0` | Left snap window: coordinate in `(x0a, x0b)` snaps to `x0`. |
/// | `xma`, `xmb`, `xm` | Mid snap window: coordinate in `(xma, xmb)` snaps to `xm`. |
/// | `x1a`, `x1b`, `x1` | Right snap window: coordinate in `(x1a, x1b)` snaps to `x1`. |
#[derive(Clone, Debug)]
pub struct XPathAdjust {
    /// First path-point index (inclusive) in the range to adjust.
    pub first_pt: usize,
    /// Last path-point index (inclusive) in the range to adjust.
    pub last_pt: usize,
    /// `true` → adjust the x-coordinate; `false` → adjust the y-coordinate.
    pub vert: bool,
    /// Left snap window lower bound (`adj0 - 0.01`).
    pub x0a: f64,
    /// Left snap window upper bound (`adj0 + 0.01`).
    pub x0b: f64,
    /// Left snap target (rounded `adj0`).
    pub x0: f64,
    /// Mid snap window lower bound (`mid - 0.01`).
    pub xma: f64,
    /// Mid snap window upper bound (`mid + 0.01`).
    pub xmb: f64,
    /// Mid snap target (midpoint of rounded endpoints).
    pub xm: f64,
    /// Right snap window lower bound (`adj1 - 0.01`).
    pub x1a: f64,
    /// Right snap window upper bound (`adj1 + 0.01`).
    pub x1b: f64,
    /// Right snap target (`rounded_adj1 - 0.01`).
    pub x1: f64,
}

impl XPathAdjust {
    /// Construct an adjust record for two endpoint values `adj0 ≤ adj1`.
    ///
    /// # Precondition
    ///
    /// `adj0 <= adj1`. The caller in `build_adjusts` in `xpath.rs` guarantees
    /// this by passing `min`/`max`-reduced values.
    ///
    /// `adjust_lines` and `line_pos_i` implement the same special-case logic as
    /// in `SplashXPath` constructor: when the two rounded endpoints coincide and
    /// `adjust_lines` is true, the span is expanded to
    /// `[line_pos_i, line_pos_i + 1]`.
    #[must_use]
    pub fn new(
        first_pt: usize,
        last_pt: usize,
        vert: bool,
        adj0: f64,
        adj1: f64,
        adjust_lines: bool,
        line_pos_i: i32,
    ) -> Self {
        debug_assert!(adj0 <= adj1, "adj0 must be <= adj1");

        let mid = adj0.midpoint(adj1);
        let mut r0 = f64::from(splash_round(adj0));
        let mut r1 = f64::from(splash_round(adj1));
        // Both `r0` and `r1` are exact f64 representations of i32 values
        // (from `splash_round`, which returns i32). Every i32 is representable
        // exactly as f64, so bit-exact comparison correctly detects when both
        // endpoints round to the same integer — no floating-point imprecision
        // can produce a false positive or false negative here.
        if r0.to_bits() == r1.to_bits() {
            if adjust_lines {
                r0 = f64::from(line_pos_i);
                r1 = f64::from(line_pos_i) + 1.0;
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
            xm: r0.midpoint(r1),
            x1a: adj1 - 0.01,
            x1b: adj1 + 0.01,
            x1: r1 - 0.01,
        }
    }
}

/// Apply a stroke-adjust hint to a single point `(x, y)`.
///
/// If `adj.vert` is true, the x-coordinate is examined; otherwise the
/// y-coordinate. If the relevant coordinate falls **strictly inside** any of
/// the three snap windows (open intervals), it is replaced by the corresponding
/// snapped target.
///
/// ## Open-interval semantics
///
/// Windows are open intervals; boundary values are **not** snapped. For
/// example, a coordinate equal to exactly `adj.x0a` (= `adj0 - 0.01`) is
/// outside the window `(x0a, x0b)` and will not be snapped.
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

    #[test]
    fn snaps_only_within_open_interval() {
        // A coordinate exactly on the window boundary must NOT be snapped,
        // because the windows are open intervals: v ∈ (x0a, x0b), not [x0a, x0b].
        let adj = XPathAdjust::new(0, 1, true, 1.0, 3.0, false, 0);
        // x0a = adj0 - 0.01 = 0.99 — exactly on the boundary, must not snap.
        let boundary = adj.x0a;
        let mut x = boundary;
        let mut y = 0.0;
        stroke_adjust(&adj, &mut x, &mut y);
        assert!(
            (x - boundary).abs() < 1e-15,
            "coordinate exactly at x0a boundary must not be snapped, got x={x}"
        );

        // Similarly for x0b = adj0 + 0.01 = 1.01.
        let mut x = adj.x0b;
        let boundary_b = adj.x0b;
        stroke_adjust(&adj, &mut x, &mut y);
        assert!(
            (x - boundary_b).abs() < 1e-15,
            "coordinate exactly at x0b boundary must not be snapped, got x={x}"
        );
    }
}
