//! PDF path geometry: points, flags, subpath state machine, and builder API.
//!
//! Mirrors `SplashPath` from `splash/SplashPath.h` and `splash/SplashPath.cc`.
//!
//! ## Subpath state machine
//!
//! A [`Path`] is always in one of three states (matching the C++ comments in
//! `SplashPath.cc`):
//!
//! | State | Condition | Meaning |
//! |---|---|---|
//! | No current point | `cur_subpath == pts.len()` | Fresh or just closed |
//! | One-point subpath | `cur_subpath == pts.len() - 1` | After moveTo, before lineTo |
//! | Open subpath | `cur_subpath < pts.len() - 1` | Active path with ≥ 2 points |

pub mod adjust;
pub mod flatten;

use bitflags::bitflags;

// ── Types ─────────────────────────────────────────────────────────────────────

/// A 2-D point in path space (f64, matching `SplashPathPoint`).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct PathPoint {
    pub x: f64,
    pub y: f64,
}

impl PathPoint {
    #[must_use]
    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
}

bitflags! {
    /// Per-point flags stored in the parallel `flags` array of a [`Path`].
    /// Matches the `splashPathFirst/Last/Closed/Curve` constants in `SplashPath.h`.
    #[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
    pub struct PathFlags: u8 {
        /// First point of a subpath.
        const FIRST  = 0x01;
        /// Last point of a subpath.
        const LAST   = 0x02;
        /// Subpath is closed (set on both the first and last point).
        const CLOSED = 0x04;
        /// This is a Bezier control point (not an on-curve endpoint).
        const CURVE  = 0x08;
    }
}

/// Stroke-adjust hint: a pair of path segments that should be snapped to
/// integer pixel boundaries to avoid seams between adjacent filled rectangles.
///
/// Matches `SplashPathHint` in `SplashPath.h`.
#[derive(Copy, Clone, Debug)]
pub struct StrokeAdjustHint {
    /// Index of the first control segment.
    pub ctrl0: usize,
    /// Index of the second control segment.
    pub ctrl1: usize,
    /// Range of points to adjust: `[first_pt, last_pt]`.
    pub first_pt: usize,
    pub last_pt: usize,
}

/// Error type for [`PathBuilder`] operations.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PathError {
    /// `lineTo` / `curveTo` called with no current point.
    NoCurPt,
    /// `moveTo` called while a one-point subpath is active.
    BogusPath,
}

// ── Path ──────────────────────────────────────────────────────────────────────

/// A PDF graphics path: an ordered sequence of subpaths built from lines and
/// cubic Bezier curves.
///
/// Invariants (matching `SplashPath.cc`):
/// - `pts.len() == flags.len()` always.
/// - `cur_subpath` points to the first point of the current (open) subpath, or
///   equals `pts.len()` when there is no current point.
/// - Control points (flag `CURVE`) always appear in groups of three between
///   two on-curve points.
#[derive(Clone, Debug, Default)]
pub struct Path {
    pub pts: Vec<PathPoint>,
    pub flags: Vec<PathFlags>,
    pub hints: Vec<StrokeAdjustHint>,
    /// Index of the first point of the currently open subpath.
    /// Equals `pts.len()` when there is no current point.
    pub cur_subpath: usize,
}

impl Path {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    // ── State predicates ──────────────────────────────────────────────────────

    /// True when there is no current point (fresh path or after close).
    #[inline]
    #[must_use]
    pub const fn no_current_point(&self) -> bool {
        self.cur_subpath == self.pts.len()
    }

    /// True after exactly one `moveTo` with no subsequent `lineTo`/`curveTo`.
    #[inline]
    #[must_use]
    pub const fn one_point_subpath(&self) -> bool {
        !self.pts.is_empty() && self.cur_subpath == self.pts.len() - 1
    }

    /// True when the current subpath has at least two points.
    #[inline]
    #[must_use]
    pub const fn open_subpath(&self) -> bool {
        !self.pts.is_empty() && self.cur_subpath < self.pts.len() - 1
    }

    /// Current point (last point appended), if any.
    #[must_use]
    pub fn current_point(&self) -> Option<PathPoint> {
        if self.no_current_point() {
            None
        } else {
            self.pts.last().copied()
        }
    }

    // ── Geometry queries ──────────────────────────────────────────────────────

    /// Translate all points by (dx, dy).
    pub fn offset(&mut self, dx: f64, dy: f64) {
        for p in &mut self.pts {
            p.x += dx;
            p.y += dy;
        }
    }

    /// Append all points and hints from `other` into `self`.
    ///
    /// `cur_subpath` is updated to `self.pts.len() + other.cur_subpath` before
    /// appending, matching `SplashPath::append`.
    pub fn append(&mut self, other: &Self) {
        let base = self.pts.len();
        self.cur_subpath = base + other.cur_subpath;
        self.pts.extend_from_slice(&other.pts);
        self.flags.extend_from_slice(&other.flags);
        for h in &other.hints {
            self.hints.push(StrokeAdjustHint {
                ctrl0: h.ctrl0 + base,
                ctrl1: h.ctrl1 + base,
                first_pt: h.first_pt + base,
                last_pt: h.last_pt + base,
            });
        }
    }
}

// ── PathBuilder ───────────────────────────────────────────────────────────────

/// Ergonomic builder for [`Path`], implementing the PDF path construction
/// operators (m, l, c, h) with the same state-machine semantics as
/// `SplashPath::moveTo/lineTo/curveTo/close`.
pub struct PathBuilder {
    path: Path,
}

impl PathBuilder {
    #[must_use]
    pub fn new() -> Self {
        Self { path: Path::new() }
    }

    /// Begin a new subpath at (x, y). Equivalent to the PDF `m` operator.
    ///
    /// # Errors
    ///
    /// Returns `Err(BogusPath)` if called while a one-point subpath is active
    /// (matching `SplashPath::moveTo` error semantics).
    pub fn move_to(&mut self, x: f64, y: f64) -> Result<(), PathError> {
        if self.path.one_point_subpath() {
            return Err(PathError::BogusPath);
        }
        let len = self.path.pts.len();
        self.path.pts.push(PathPoint::new(x, y));
        self.path.flags.push(PathFlags::FIRST | PathFlags::LAST);
        self.path.cur_subpath = len;
        Ok(())
    }

    /// Add a line segment from the current point to (x, y). PDF `l` operator.
    ///
    /// # Errors
    ///
    /// Returns `Err(NoCurPt)` if there is no current point.
    ///
    /// # Panics
    ///
    /// Panics if the flags vector is empty despite having a current point
    /// (invariant violation).
    pub fn line_to(&mut self, x: f64, y: f64) -> Result<(), PathError> {
        if self.path.no_current_point() {
            return Err(PathError::NoCurPt);
        }
        // Clear LAST on the previous point.
        let last = self.path.flags.last_mut().unwrap();
        last.remove(PathFlags::LAST);
        self.path.pts.push(PathPoint::new(x, y));
        self.path.flags.push(PathFlags::LAST);
        Ok(())
    }

    /// Add a cubic Bezier curve. PDF `c` operator.
    ///
    /// `(x1,y1)` and `(x2,y2)` are control points; `(x3,y3)` is the endpoint.
    ///
    /// # Errors
    ///
    /// Returns `Err(NoCurPt)` if there is no current point.
    ///
    /// # Panics
    ///
    /// Panics if the flags vector is empty despite having a current point
    /// (invariant violation).
    pub fn curve_to(
        &mut self,
        x1: f64,
        y1: f64,
        x2: f64,
        y2: f64,
        x3: f64,
        y3: f64,
    ) -> Result<(), PathError> {
        if self.path.no_current_point() {
            return Err(PathError::NoCurPt);
        }
        let last = self.path.flags.last_mut().unwrap();
        last.remove(PathFlags::LAST);
        // Two control points (flagged CURVE) + one on-curve endpoint (LAST).
        self.path.pts.push(PathPoint::new(x1, y1));
        self.path.flags.push(PathFlags::CURVE);
        self.path.pts.push(PathPoint::new(x2, y2));
        self.path.flags.push(PathFlags::CURVE);
        self.path.pts.push(PathPoint::new(x3, y3));
        self.path.flags.push(PathFlags::LAST);
        Ok(())
    }

    /// Close the current subpath. PDF `h` operator.
    ///
    /// If `force` is true a closing `lineTo` is always added; otherwise it is
    /// skipped only when the subpath is already closed (last == first point).
    ///
    /// # Errors
    ///
    /// Returns `Err(NoCurPt)` if there is no current point.
    pub fn close(&mut self, force: bool) -> Result<(), PathError> {
        if self.path.no_current_point() {
            return Err(PathError::NoCurPt);
        }
        let sp = self.path.cur_subpath;
        let last_idx = self.path.pts.len() - 1;
        let first = self.path.pts[sp];
        let last = self.path.pts[last_idx];
        // Add a closing lineTo unless first == last and we're not forced.
        if force || sp == last_idx || first != last {
            self.line_to(first.x, first.y)?;
        }
        // Set CLOSED on the first and last point of the subpath.
        let new_last = self.path.pts.len() - 1;
        self.path.flags[sp].insert(PathFlags::CLOSED);
        self.path.flags[new_last].insert(PathFlags::CLOSED);
        // Advance cur_subpath past this subpath.
        self.path.cur_subpath = self.path.pts.len();
        Ok(())
    }

    /// Add a stroke-adjust hint.
    pub fn add_stroke_adjust_hint(
        &mut self,
        ctrl0: usize,
        ctrl1: usize,
        first_pt: usize,
        last_pt: usize,
    ) {
        self.path.hints.push(StrokeAdjustHint {
            ctrl0,
            ctrl1,
            first_pt,
            last_pt,
        });
    }

    /// Translate all points by (dx, dy).
    pub fn offset(&mut self, dx: f64, dy: f64) {
        self.path.offset(dx, dy);
    }

    /// Consume the builder and return the completed [`Path`].
    #[must_use]
    pub fn build(self) -> Path {
        self.path
    }
}

impl Default for PathBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_state() {
        let p = Path::new();
        assert!(p.no_current_point());
        assert!(!p.one_point_subpath());
        assert!(!p.open_subpath());
    }

    #[test]
    fn move_to_gives_one_point() {
        let mut b = PathBuilder::new();
        b.move_to(1.0, 2.0).unwrap();
        assert!(b.path.one_point_subpath());
        assert!(!b.path.open_subpath());
    }

    #[test]
    fn line_to_opens_subpath() {
        let mut b = PathBuilder::new();
        b.move_to(0.0, 0.0).unwrap();
        b.line_to(10.0, 0.0).unwrap();
        assert!(b.path.open_subpath());
        assert_eq!(b.path.pts.len(), 2);
    }

    #[test]
    fn curve_to_adds_three_points() {
        let mut b = PathBuilder::new();
        b.move_to(0.0, 0.0).unwrap();
        b.curve_to(1.0, 2.0, 3.0, 4.0, 5.0, 0.0).unwrap();
        assert_eq!(b.path.pts.len(), 4); // moveTo + 3 curve points
        assert!(b.path.flags[1].contains(PathFlags::CURVE));
        assert!(b.path.flags[2].contains(PathFlags::CURVE));
        assert!(b.path.flags[3].contains(PathFlags::LAST));
    }

    #[test]
    fn close_sets_closed_flag() {
        let mut b = PathBuilder::new();
        b.move_to(0.0, 0.0).unwrap();
        b.line_to(10.0, 0.0).unwrap();
        b.line_to(5.0, 5.0).unwrap();
        b.close(false).unwrap();
        // CLOSED flag on first and last point of subpath
        assert!(b.path.flags[0].contains(PathFlags::CLOSED));
        assert!(b.path.flags.last().unwrap().contains(PathFlags::CLOSED));
        assert!(b.path.no_current_point());
    }

    #[test]
    fn no_cur_pt_errors() {
        let mut b = PathBuilder::new();
        assert_eq!(b.line_to(1.0, 1.0), Err(PathError::NoCurPt));
        assert_eq!(
            b.curve_to(1.0, 1.0, 2.0, 2.0, 3.0, 3.0),
            Err(PathError::NoCurPt)
        );
        assert_eq!(b.close(false), Err(PathError::NoCurPt));
    }

    #[test]
    fn bogus_path_on_double_moveto() {
        let mut b = PathBuilder::new();
        b.move_to(0.0, 0.0).unwrap();
        assert_eq!(b.move_to(1.0, 1.0), Err(PathError::BogusPath));
    }

    #[test]
    fn append_adjusts_hints() {
        let mut a = PathBuilder::new();
        a.move_to(0.0, 0.0).unwrap();
        a.line_to(1.0, 0.0).unwrap();
        let pa = a.build();

        let mut b_path = pa.clone();
        let mut other = pa;
        other.hints.push(StrokeAdjustHint {
            ctrl0: 0,
            ctrl1: 1,
            first_pt: 0,
            last_pt: 1,
        });
        b_path.append(&other);
        // Hint indices should be offset by original pa.pts.len() = 2
        assert_eq!(b_path.hints[0].ctrl0, 2);
        assert_eq!(b_path.hints[0].first_pt, 2);
    }
}
