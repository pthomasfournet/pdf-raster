//! Flattened, matrix-transformed path edge table.
//!
//! [`XPath`] is the Rust equivalent of `SplashXPath` from `splash/SplashXPath.h/.cc`.
//! It converts a [`Path`] (in user space) into a sorted sequence of line segments
//! in device space, ready for scan conversion by [`XPathScanner`].
//!
//! ## Typical usage pipeline
//!
//! ```text
//! XPath::new(path, matrix, flatness, close_subpaths)   ← construction
//!     │  internally calls add_segment for every flattened edge
//!     ▼
//! XPath                                                 ← device-space edge table
//!     │  optionally:
//!     ▼
//! xpath.aa_scale()                                      ← scale coords × AA_SIZE
//!     │  must be called AT MOST ONCE, before handing to XPathScanner
//!     ▼
//! XPathScanner                                          ← scan conversion
//! ```
//!
//! ## Key invariants (established by [`add_segment`](XPath::add_segment))
//!
//! - For every non-horizontal segment, `y0 ≤ y1` after construction (swapped
//!   if necessary; [`XPathFlags::FLIPPED`] is set when a swap occurred).
//! - [`XPathFlags::HORIZ`] is set when `y0 == y1` (despite the misleading
//!   "vertical" comment in the original C++ header — trust the code).
//! - [`XPathFlags::VERT`] is set when `x0 == x1`.
//! - `dxdy = (x1-x0)/(y1-y0)` for sloped segments; 0.0 for horizontal/vertical.
//!   Division is safe because the HORIZ early-return guarantees `y0 ≠ y1` for
//!   any segment that reaches the slope computation.
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
///
/// # Invariant
///
/// After construction by [`XPath::add_segment`], every non-horizontal segment
/// satisfies `y0 ≤ y1`. Horizontal segments (`HORIZ` flag set) are exempt.
#[derive(Clone, Debug)]
pub struct XPathSeg {
    /// X coordinate of the segment start point, in device space.
    pub x0: f64,
    /// Y coordinate of the segment start point, in device space.
    ///
    /// **Invariant**: `y0 ≤ y1` for all non-horizontal segments after
    /// construction. [`XPathFlags::FLIPPED`] is set when the original input had
    /// `y0 > y1` and the endpoints were swapped to enforce this invariant.
    pub y0: f64,
    /// X coordinate of the segment end point, in device space.
    pub x1: f64,
    /// Y coordinate of the segment end point, in device space.
    ///
    /// Always `y0 ≤ y1` for non-horizontal segments after construction.
    pub y1: f64,
    /// Slope `(x1-x0)/(y1-y0)`.
    ///
    /// Set to `0.0` for horizontal segments (where the denominator is zero) and
    /// for vertical segments (where the numerator is zero). For all other
    /// segments the denominator is strictly non-zero because the HORIZ early-
    /// return in [`XPath::add_segment`] guarantees `y0 ≠ y1`.
    pub dxdy: f64,
    /// Orientation and flip flags; see [`XPathFlags`] for the set of valid bits.
    pub flags: XPathFlags,
}

// ── XPath ─────────────────────────────────────────────────────────────────────

/// Matrix-transformed, flattened edge table derived from a [`Path`].
///
/// # Construction pipeline
///
/// 1. Call [`XPath::new`] to build the edge table from a [`Path`].
/// 2. Optionally call [`XPath::aa_scale`] **once** to scale coordinates by
///    [`AA_SIZE`] for supersampled anti-aliasing.
/// 3. Hand the resulting `XPath` to `XPathScanner` for scan conversion.
///
/// Calling `aa_scale` more than once will multiply coordinates by `AA_SIZE`
/// again, producing incorrect results. This is not checked at runtime.
pub struct XPath {
    /// The flattened, transformed edge segments making up this path, in insertion order.
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
    /// # Arguments
    ///
    /// - `path`: the source path in user (pre-transform) space.
    /// - `matrix`: a 2-D affine transform `[a, b, c, d, e, f]` mapping user
    ///   space to device space (column-vector convention; see module docs).
    /// - `flatness`: maximum chord deviation (in device pixels) for Bezier
    ///   subdivision. Smaller values produce more accurate curves but more
    ///   segments. Typical range: `0.1`–`1.0`.
    /// - `close_subpaths`: if `true`, an implicit closing segment is added from
    ///   the last point of each subpath back to its first point when they do not
    ///   already coincide (matches `SplashXPath` constructor behaviour).
    ///
    /// # Ordering constraints
    ///
    /// After this call, `segs` is in insertion order (one entry per flattened
    /// edge). No further sorting is performed here; callers that need a sorted
    /// edge table must sort `segs` themselves or use `XPathScanner`.
    ///
    /// Calling [`XPath::aa_scale`] after this method scales all coordinates by
    /// [`AA_SIZE`]. It must be called at most once.
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

        // Phase 1 always uses adjust_lines=false, line_pos_i=0 (thin-line adjustment is Phase 2).
        let adjusts = build_adjusts(&path.hints, &tpts, false, 0);

        // Apply stroke adjustments to the transformed points.
        let mut tpts = tpts;
        for adj in &adjusts {
            // Safety: build_adjusts validates that first_pt and last_pt are
            // within bounds before constructing the XPathAdjust records, so
            // this slice index cannot panic.
            debug_assert!(
                adj.last_pt < tpts.len(),
                "adj.last_pt ({}) out of bounds (tpts.len() = {})",
                adj.last_pt,
                tpts.len()
            );
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
                if close_subpaths
                    && ((cur_x - sp_x).abs() > 1e-10 || (cur_y - sp_y).abs() > 1e-10)
                {
                    xpath.add_segment(cur_x, cur_y, sp_x, sp_y);
                }
            } else {
                i += 1;
            }
        }

        xpath
    }

    /// Scale all segment coordinates by [`AA_SIZE`] for supersampled anti-aliasing.
    ///
    /// `dxdy` (the slope) is invariant under uniform scaling and is **not** modified —
    /// a uniform scale cancels out in `(x1-x0)/(y1-y0)`.
    ///
    /// Matches `SplashXPath::aaScale()`.
    ///
    /// # Ordering constraint
    ///
    /// This method must be called **after** [`XPath::new`] and **at most once**.
    /// Calling it a second time multiplies coordinates by [`AA_SIZE`] again,
    /// which will produce incorrect scan-conversion results. There is no runtime
    /// guard against double-scaling.
    ///
    /// # Panics
    ///
    /// Does not panic in practice. However, if any coordinate is so large that
    /// multiplying by [`AA_SIZE`] would overflow to `f64::INFINITY`, subsequent
    /// scan-conversion arithmetic will silently produce wrong results. A
    /// `debug_assert!` fires in debug builds if any coordinate is non-finite
    /// before scaling.
    pub fn aa_scale(&mut self) {
        let s = f64::from(AA_SIZE);
        for seg in &mut self.segs {
            debug_assert!(
                seg.x0.is_finite() && seg.y0.is_finite()
                    && seg.x1.is_finite() && seg.y1.is_finite(),
                "aa_scale: segment coordinates must be finite before scaling \
                 (x0={}, y0={}, x1={}, y1={})",
                seg.x0, seg.y0, seg.x1, seg.y1,
            );
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
    /// # y0 ≤ y1 invariant
    ///
    /// If the supplied `y0 > y1`, the endpoints are swapped and
    /// [`XPathFlags::FLIPPED`] is set. Horizontal segments (`y0 == y1`) are
    /// never flipped.
    ///
    /// # Division safety
    ///
    /// `dxdy = (x1-x0)/(y1-y0)` is computed only for segments that are neither
    /// horizontal nor vertical. The HORIZ early-return guarantees `y0 ≠ y1` for
    /// any segment that reaches this computation, so division by zero cannot
    /// occur. A `debug_assert!` enforces this contract in debug builds.
    ///
    /// Matches `SplashXPath::addSegment` in `SplashXPath.cc`.
    fn add_segment(&mut self, mut x0: f64, mut y0: f64, mut x1: f64, mut y1: f64) {
        let mut flags = XPathFlags::empty();

        // Exact bit-equality is intentional: checking for axis-aligned segments
        // that were constructed with the same coordinate value.
        if y0.to_bits() == y1.to_bits() {
            // Horizontal segment: y0 == y1, dxdy is undefined; set to 0.0.
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

        // Non-horizontal: y0 ≠ y1 is guaranteed by the early return above.
        if x0.to_bits() == x1.to_bits() {
            flags.insert(XPathFlags::VERT);
        }

        // Compute slope before the potential swap so that the sign is
        // consistent with the *original* orientation. After the swap below,
        // the stored dxdy is the slope in the y0-ascending direction.
        let dxdy = if flags.contains(XPathFlags::VERT) {
            0.0
        } else {
            // Division is safe: HORIZ guard above guarantees y1 - y0 ≠ 0.0.
            debug_assert_ne!(
                y1.to_bits(),
                y0.to_bits(),
                "add_segment: y0 == y1 must be caught by the HORIZ branch"
            );
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

/// Apply a 2-D affine matrix to point `(xi, yi)`, returning the transformed
/// [`PathPoint`] in device space.
///
/// Column-vector convention matching `SplashXPath::transform`:
///
/// ```text
/// x_out = xi*m[0] + yi*m[2] + m[4]
/// y_out = xi*m[1] + yi*m[3] + m[5]
/// ```
///
/// Uses `f64::mul_add` for fused multiply-add, giving one rounding error per
/// term rather than two.
#[inline]
#[must_use]
pub const fn transform(m: &[f64; 6], xi: f64, yi: f64) -> PathPoint {
    PathPoint::new(
        xi.mul_add(m[0], yi.mul_add(m[2], m[4])),
        xi.mul_add(m[1], yi.mul_add(m[3], m[5])),
    )
}

// ── Stroke adjust record construction ────────────────────────────────────────

/// Build [`XPathAdjust`] records from path hints and transformed points.
///
/// Mirrors the hint-processing loop in the `SplashXPath` constructor.
///
/// Only axis-aligned hint pairs (both edges strictly horizontal or both
/// strictly vertical after transformation) are converted to adjust records;
/// skewed pairs are silently dropped, matching the C++ behaviour.
///
/// Hints whose control-point indices are out of range for `tpts` are also
/// silently dropped rather than panicking, so that malformed PDF content
/// cannot cause crashes.
///
/// # Parameters `adjust_lines` and `line_pos_i`
///
/// These parameters are **reserved for Phase 2 thin-line stroke adjustment**
/// and are currently always called with `(false, 0)`. When `adjust_lines` is
/// `false` the `line_pos_i` value is unused and the `XPathAdjust::new` branch
/// that computes adjusted `r0`/`r1` from `line_pos_i` is never reached.
///
/// Do **not** remove these parameters; they will be wired up in Phase 2.
fn build_adjusts(
    hints: &[StrokeAdjustHint],
    tpts: &[PathPoint],
    adjust_lines: bool,
    line_pos_i: i32,
) -> Vec<XPathAdjust> {
    // Document the current calling convention; Phase 2 will relax this.
    debug_assert!(
        !adjust_lines && line_pos_i == 0,
        "build_adjusts: adjust_lines={adjust_lines} line_pos_i={line_pos_i}; \
         Phase 2 thin-line adjustment not yet implemented"
    );
    let mut adjusts = Vec::with_capacity(hints.len());
    for h in hints {
        // Validate indices: each hint references two consecutive point pairs.
        // ctrl0+1 and ctrl1+1 must both be valid indices into tpts.
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
        // The two coordinates to snap: take min/max so adj0 ≤ adj1 always.
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

    /// A degenerate point segment (x0==x1, y0==y1) should set both HORIZ and
    /// VERT flags and not panic.
    #[test]
    fn degenerate_point_segment() {
        let mut xpath = XPath {
            segs: Vec::new(),
            curve_data: None,
        };
        xpath.add_segment(3.0, 7.0, 3.0, 7.0);
        let s = &xpath.segs[0];
        assert!(s.flags.contains(XPathFlags::HORIZ));
        assert!(s.flags.contains(XPathFlags::VERT));
        assert!(!s.flags.contains(XPathFlags::FLIPPED));
        assert_eq!(s.dxdy.to_bits(), 0.0_f64.to_bits());
    }

    /// `dxdy` must equal `(x1-x0)/(y1-y0)` for a sloped segment.
    #[test]
    fn sloped_segment_dxdy() {
        let mut xpath = XPath {
            segs: Vec::new(),
            curve_data: None,
        };
        // y0 < y1, not vertical → dxdy = (6-2)/(5-1) = 1.0
        xpath.add_segment(2.0, 1.0, 6.0, 5.0);
        let s = &xpath.segs[0];
        assert!(!s.flags.contains(XPathFlags::HORIZ));
        assert!(!s.flags.contains(XPathFlags::VERT));
        assert!(!s.flags.contains(XPathFlags::FLIPPED));
        assert!((s.dxdy - 1.0).abs() < f64::EPSILON, "dxdy={}", s.dxdy);
    }

    /// A flipped sloped segment must have the same absolute dxdy value as its
    /// unflipped counterpart, and must satisfy y0 ≤ y1 after construction.
    #[test]
    fn flipped_sloped_segment_dxdy_consistent() {
        let mut xpath = XPath {
            segs: Vec::new(),
            curve_data: None,
        };
        // Supply (x0=6, y0=5) → (x1=2, y1=1): y0 > y1, so it will be flipped.
        xpath.add_segment(6.0, 5.0, 2.0, 1.0);
        let s = &xpath.segs[0];
        assert!(s.flags.contains(XPathFlags::FLIPPED));
        assert!(s.y0 <= s.y1, "y0 ≤ y1 invariant violated: y0={} y1={}", s.y0, s.y1);
        // dxdy = (x1_orig - x0_orig)/(y1_orig - y0_orig) = (2-6)/(1-5) = 1.0
        assert!((s.dxdy - 1.0).abs() < f64::EPSILON, "dxdy={}", s.dxdy);
    }
}
