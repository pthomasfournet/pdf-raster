//! Flattened, matrix-transformed path edge table.
//!
//! [`XPath`] is the Rust equivalent of `SplashXPath` from `splash/SplashXPath.h/.cc`.
//! It converts a [`Path`] (in user space) into a sorted sequence of line segments
//! in device space, ready for scan conversion by [`XPathScanner`].
//!
//! ## Key invariants (established by [`add_segment`])
//!
//! - For every non-horizontal segment, `y0 ≤ y1` after construction (swapped
//!   if necessary; [`XPathFlags::FLIPPED`] is set when a swap occurred).
//! - [`XPathFlags::HORIZ`] is set when `y0 == y1` (despite the misleading
//!   "vertical" comment in the original C++ header — trust the code).
//! - [`XPathFlags::VERT`] is set when `x0 == x1`.
//! - `dxdy = (x1-x0)/(y1-y0)` for sloped segments; 0.0 for horizontal/vertical.
//!
//! ## Affine transform convention
//!
//! ```text
//! x_out = x_in * m[0] + y_in * m[2] + m[4]
//! y_out = x_in * m[1] + y_in * m[3] + m[5]
//! ```
//! (column-vector convention matching `SplashXPath::transform`.)

use crate::path::adjust::{XPathAdjust, stroke_adjust};
use crate::path::flatten::{CurveData, flatten_curve};
use crate::path::{Path, PathFlags, PathPoint, StrokeAdjustHint};
use crate::types::AA_SIZE;
use bitflags::bitflags;

// ── XPathFlags ────────────────────────────────────────────────────────────────

bitflags! {
    /// Per-segment flags for [`XPathSeg`].
    #[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
    pub struct XPathFlags: u32 {
        /// Horizontal segment: y0 == y1. (`splashXPathHoriz`)
        /// NOTE: the C++ header comment says "vertical" — this is wrong.
        const HORIZ   = 0x01;
        /// Vertical segment: x0 == x1. (`splashXPathVert`)
        const VERT    = 0x02;
        /// Segment was flipped (original y0 > y1) to enforce y0 ≤ y1.
        const FLIPPED = 0x04;
    }
}

// ── XPathSeg ──────────────────────────────────────────────────────────────────

/// A single line segment in the edge table, in device space.
///
/// Matches `SplashXPathSeg` from `splash/SplashXPath.h`.
#[derive(Clone, Debug)]
pub struct XPathSeg {
    pub x0: f64,
    pub y0: f64,
    pub x1: f64,
    pub y1: f64,
    /// Slope (x1-x0)/(y1-y0). Undefined (0.0) for horizontal/vertical segments.
    pub dxdy: f64,
    pub flags: XPathFlags,
}

// ── XPath ─────────────────────────────────────────────────────────────────────

/// Matrix-transformed, flattened edge table derived from a [`Path`].
pub struct XPath {
    pub segs: Vec<XPathSeg>,
    /// Lazily allocated (~25 KB) scratch for Bezier subdivision.
    curve_data: Option<Box<CurveData>>,
}

impl XPath {
    /// Create an empty `XPath` (for tests and internal use).
    #[cfg(test)]
    pub(crate) const fn empty() -> Self {
        Self {
            segs: Vec::new(),
            curve_data: None,
        }
    }

    /// Build an `XPath` from a [`Path`] by applying `matrix` and flattening curves.
    ///
    /// - `flatness`: maximum chord deviation (in device pixels) for Bezier subdivision.
    /// - `close_subpaths`: if true, an implicit closing segment is added from the last
    ///   point of each subpath back to its first point (matching `SplashXPath` ctor).
    #[must_use]
    pub fn new(path: &Path, matrix: &[f64; 6], flatness: f64, close_subpaths: bool) -> Self {
        let flatness_sq = flatness * flatness;
        let mut xpath = Self {
            segs: Vec::new(),
            curve_data: None,
        };

        // Transform every path point into device space.
        let tpts: Vec<PathPoint> = path
            .pts
            .iter()
            .map(|p| transform(matrix, p.x, p.y))
            .collect();

        // Build stroke-adjust records from the path hints (if any).
        let adjusts = build_adjusts(&path.hints, &tpts, false, 0);

        // Apply stroke adjustments to the transformed points.
        let mut tpts = tpts;
        for adj in &adjusts {
            for pt in &mut tpts[adj.first_pt..=adj.last_pt] {
                let (x, y) = (&mut pt.x, &mut pt.y);
                stroke_adjust(adj, x, y);
            }
        }

        // Walk the path and emit segments.
        let n = path.pts.len();
        let mut i = 0usize;
        while i < n {
            if path.flags[i].contains(PathFlags::FIRST) {
                // Start of a new subpath.
                let sp_x = tpts[i].x;
                let sp_y = tpts[i].y;
                let mut cur_x = sp_x;
                let mut cur_y = sp_y;
                i += 1;
                while i < n {
                    if path.flags[i].contains(PathFlags::CURVE) {
                        // Cubic Bezier: consume 3 points (2 control + 1 endpoint).
                        if i + 2 >= n {
                            break;
                        }
                        let p0 = PathPoint::new(cur_x, cur_y);
                        let p1 = tpts[i];
                        let p2 = tpts[i + 1];
                        let p3 = tpts[i + 2];
                        let mut flat_pts = Vec::new();
                        flatten_curve(
                            p0,
                            p1,
                            p2,
                            p3,
                            flatness_sq,
                            &mut flat_pts,
                            &mut xpath.curve_data,
                        );
                        for fp in &flat_pts {
                            xpath.add_segment(cur_x, cur_y, fp.x, fp.y);
                            cur_x = fp.x;
                            cur_y = fp.y;
                        }
                        i += 3;
                    } else {
                        // Line segment.
                        let nx = tpts[i].x;
                        let ny = tpts[i].y;
                        xpath.add_segment(cur_x, cur_y, nx, ny);
                        cur_x = nx;
                        cur_y = ny;
                        let is_last = path.flags[i].contains(PathFlags::LAST);
                        i += 1;
                        if is_last {
                            break;
                        }
                    }
                }
                // Closing segment if requested and the subpath is not already closed.
                if close_subpaths && ((cur_x - sp_x).abs() > 1e-10 || (cur_y - sp_y).abs() > 1e-10)
                {
                    xpath.add_segment(cur_x, cur_y, sp_x, sp_y);
                }
            } else {
                i += 1;
            }
        }

        xpath
    }

    /// Multiply all segment coordinates by [`AA_SIZE`] (=4) for supersampled AA.
    ///
    /// `dxdy` (the slope) is invariant under uniform scaling and is not modified.
    /// Matches `SplashXPath::aaScale()`.
    pub fn aa_scale(&mut self) {
        let s = f64::from(AA_SIZE);
        for seg in &mut self.segs {
            seg.x0 *= s;
            seg.y0 *= s;
            seg.x1 *= s;
            seg.y1 *= s;
        }
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /// Append one line segment to the edge table, enforcing `y0 ≤ y1` for
    /// non-horizontal segments and computing `dxdy`.
    ///
    /// Matches `SplashXPath::addSegment` in `SplashXPath.cc`.
    fn add_segment(&mut self, mut x0: f64, mut y0: f64, mut x1: f64, mut y1: f64) {
        let mut flags = XPathFlags::empty();

        // Exact bit-equality is intentional: checking for axis-aligned segments
        // that were constructed with the same coordinate value.
        if y0.to_bits() == y1.to_bits() {
            // Horizontal segment.
            flags.insert(XPathFlags::HORIZ);
            if x0.to_bits() == x1.to_bits() {
                flags.insert(XPathFlags::VERT);
            }
            self.segs.push(XPathSeg {
                x0,
                y0,
                x1,
                y1,
                dxdy: 0.0,
                flags,
            });
            return; // Horizontal segments are NOT flipped.
        }
        if x0.to_bits() == x1.to_bits() {
            flags.insert(XPathFlags::VERT);
        }
        let dxdy = if flags.contains(XPathFlags::VERT) {
            0.0
        } else {
            (x1 - x0) / (y1 - y0)
        };
        if y0 > y1 {
            std::mem::swap(&mut x0, &mut x1);
            std::mem::swap(&mut y0, &mut y1);
            flags.insert(XPathFlags::FLIPPED);
        }
        self.segs.push(XPathSeg {
            x0,
            y0,
            x1,
            y1,
            dxdy,
            flags,
        });
    }
}

// ── Affine transform ──────────────────────────────────────────────────────────

/// Apply a 2-D affine matrix to point `(xi, yi)`.
///
/// Column-vector convention matching `SplashXPath::transform`:
/// ```text
/// x_out = xi*m[0] + yi*m[2] + m[4]
/// y_out = xi*m[1] + yi*m[3] + m[5]
/// ```
#[inline]
#[must_use]
pub const fn transform(m: &[f64; 6], xi: f64, yi: f64) -> PathPoint {
    PathPoint::new(
        xi.mul_add(m[0], yi.mul_add(m[2], m[4])),
        xi.mul_add(m[1], yi.mul_add(m[3], m[5])),
    )
}

// ── Stroke adjust record construction ────────────────────────────────────────

/// Build `XPathAdjust` records from path hints and transformed points.
/// Mirrors the hint-processing loop in `SplashXPath` constructor.
fn build_adjusts(
    hints: &[StrokeAdjustHint],
    tpts: &[PathPoint],
    adjust_lines: bool,
    line_pos_i: i32,
) -> Vec<XPathAdjust> {
    let mut adjusts = Vec::with_capacity(hints.len());
    for h in hints {
        // Validate indices.
        if h.ctrl0 + 1 >= tpts.len() || h.ctrl1 + 1 >= tpts.len() {
            continue;
        }
        let p00 = tpts[h.ctrl0];
        let p01 = tpts[h.ctrl0 + 1];
        let p10 = tpts[h.ctrl1];
        let p11 = tpts[h.ctrl1 + 1];
        // Determine orientation using bit-exact comparison (axis-aligned check).
        let vert = (p00.x.to_bits() == p01.x.to_bits()) && (p10.x.to_bits() == p11.x.to_bits());
        let horiz = (p00.y.to_bits() == p01.y.to_bits()) && (p10.y.to_bits() == p11.y.to_bits());
        if !vert && !horiz {
            continue;
        }
        // The two coordinates to snap.
        let (a0, a1) = if vert {
            (p00.x.min(p10.x), p00.x.max(p10.x))
        } else {
            (p00.y.min(p10.y), p00.y.max(p10.y))
        };
        adjusts.push(XPathAdjust::new(
            h.first_pt,
            h.last_pt,
            vert,
            a0,
            a1,
            adjust_lines,
            line_pos_i,
        ));
    }
    adjusts
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::path::PathBuilder;

    fn identity() -> [f64; 6] {
        [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    }

    #[test]
    fn horizontal_segment_not_flipped() {
        let mut xpath = XPath {
            segs: Vec::new(),
            curve_data: None,
        };
        xpath.add_segment(0.0, 5.0, 10.0, 5.0);
        let s = &xpath.segs[0];
        assert!(s.flags.contains(XPathFlags::HORIZ));
        assert!(!s.flags.contains(XPathFlags::FLIPPED));
        assert!((s.y0 - 5.0).abs() < f64::EPSILON);
        assert!((s.y1 - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn downward_segment_flipped() {
        let mut xpath = XPath {
            segs: Vec::new(),
            curve_data: None,
        };
        xpath.add_segment(0.0, 10.0, 0.0, 0.0); // y0 > y1 → flip
        let s = &xpath.segs[0];
        assert!(s.flags.contains(XPathFlags::FLIPPED));
        assert!(s.y0 <= s.y1, "y0={} y1={}", s.y0, s.y1);
    }

    #[test]
    fn aa_scale_multiplies_coords() {
        let mut xpath = XPath {
            segs: Vec::new(),
            curve_data: None,
        };
        xpath.add_segment(1.0, 0.0, 3.0, 2.0);
        let orig_dxdy = xpath.segs[0].dxdy;
        xpath.aa_scale();
        let s = &xpath.segs[0];
        assert!((s.x0 - 4.0).abs() < 1e-10);
        assert!((s.y0 - 0.0).abs() < 1e-10);
        assert!((s.x1 - 12.0).abs() < 1e-10);
        assert!((s.y1 - 8.0).abs() < 1e-10);
        assert!(
            (s.dxdy - orig_dxdy).abs() < 1e-10,
            "dxdy should be unchanged"
        );
    }

    #[test]
    fn vertical_segment_dxdy_zero() {
        let mut xpath = XPath {
            segs: Vec::new(),
            curve_data: None,
        };
        xpath.add_segment(5.0, 0.0, 5.0, 10.0);
        let s = &xpath.segs[0];
        assert!(s.flags.contains(XPathFlags::VERT));
        assert!(s.dxdy.abs() < f64::EPSILON);
    }

    #[test]
    fn triangle_from_path() {
        let mut b = PathBuilder::new();
        b.move_to(0.0, 0.0).unwrap();
        b.line_to(4.0, 0.0).unwrap();
        b.line_to(2.0, 4.0).unwrap();
        b.close(false).unwrap();
        let path = b.build();
        let xpath = XPath::new(&path, &identity(), 1.0, false);
        // 3 explicit segments + 1 closing → but PathBuilder's close() already
        // adds the closing lineTo, so 3 segments total.
        assert_eq!(xpath.segs.len(), 3);
    }
}
