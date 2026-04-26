//! PDF path geometry: points, flags, subpath state machine, and builder API.
//!
//! Mirrors `SplashPath` from `splash/SplashPath.h` and `splash/SplashPath.cc`.
//!
//! ## Subpath state machine
//!
//! A [`Path`] is always in one of three states (matching the C++ comments in
//! `SplashPath.cc`):
//!
//! | State            | Condition                        | Meaning                          |
//! |------------------|----------------------------------|----------------------------------|
//! | No current point | `cur_subpath == pts.len()`       | Fresh path or just after close   |
//! | One-point subpath| `cur_subpath == pts.len() - 1`   | After moveTo, before lineTo      |
//! | Open subpath     | `cur_subpath < pts.len() - 1`    | Active path with ≥ 2 points      |
//!
//! The `one_point_subpath` and `open_subpath` predicates both guard against
//! `pts.is_empty()` before comparing with `pts.len() - 1`, so neither can
//! underflow on an empty vector.

pub mod adjust;
pub mod flatten;

use bitflags::bitflags;

// ── Types ─────────────────────────────────────────────────────────────────────

/// A 2-D point in path space (f64 coordinates, matching `SplashPathPoint`).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct PathPoint {
    /// Horizontal coordinate.
    pub x: f64,
    /// Vertical coordinate.
    pub y: f64,
}

impl PathPoint {
    /// Construct a new point.
    #[must_use]
    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
}

impl From<(f64, f64)> for PathPoint {
    /// Convert a `(x, y)` tuple into a [`PathPoint`].
    fn from((x, y): (f64, f64)) -> Self {
        Self { x, y }
    }
}

impl From<PathPoint> for (f64, f64) {
    /// Destructure a [`PathPoint`] into a `(x, y)` tuple.
    fn from(p: PathPoint) -> Self {
        (p.x, p.y)
    }
}

bitflags! {
    /// Per-point flags stored in the parallel `flags` array of a [`Path`].
    ///
    /// Matches the `splashPathFirst/Last/Closed/Curve` constants in
    /// `SplashPath.h`.
    #[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
    pub struct PathFlags: u8 {
        /// First point of a subpath (set on the `moveTo` point).
        const FIRST  = 0x01;
        /// Last point of a subpath (set on every newly appended endpoint).
        const LAST   = 0x02;
        /// Subpath is closed (set on both the first **and** last point).
        const CLOSED = 0x04;
        /// This point is a cubic Bezier control point, not an on-curve endpoint.
        const CURVE  = 0x08;
    }
}

impl PathFlags {
    /// Returns `true` if this point is the first of its subpath.
    #[must_use]
    pub const fn is_first(self) -> bool {
        self.contains(Self::FIRST)
    }

    /// Returns `true` if this point is the last of its subpath.
    #[must_use]
    pub const fn is_last(self) -> bool {
        self.contains(Self::LAST)
    }

    /// Returns `true` if the subpath containing this point is closed.
    #[must_use]
    pub const fn is_closed(self) -> bool {
        self.contains(Self::CLOSED)
    }

    /// Returns `true` if this point is a Bezier control point (off-curve).
    #[must_use]
    pub const fn is_curve(self) -> bool {
        self.contains(Self::CURVE)
    }
}

/// Stroke-adjust hint: a pair of path segments that should be snapped to
/// integer pixel boundaries to avoid seams between adjacent filled rectangles.
///
/// Matches `SplashPathHint` in `SplashPath.h`.
#[derive(Copy, Clone, Debug)]
pub struct StrokeAdjustHint {
    /// Index (into [`Path::pts`]) of the first control segment.
    pub ctrl0: usize,
    /// Index (into [`Path::pts`]) of the second control segment.
    pub ctrl1: usize,
    /// First point of the range to adjust (inclusive, index into [`Path::pts`]).
    pub first_pt: usize,
    /// Last point of the range to adjust (inclusive, index into [`Path::pts`]).
    pub last_pt: usize,
}

/// Errors returned by [`PathBuilder`] construction methods.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PathError {
    /// `lineTo`, `curveTo`, or `close` was called when there is no current
    /// point (the path is fresh or was just closed).
    ///
    /// Callers should ensure a preceding `move_to` succeeded before calling
    /// drawing operators.
    NoCurPt,
    /// `moveTo` was called while a one-point subpath is active (a `moveTo`
    /// was immediately followed by another `moveTo` with no drawing operator
    /// in between).
    ///
    /// Callers should either draw at least one segment or close the current
    /// subpath before beginning a new one.
    BogusPath,
}

impl std::fmt::Display for PathError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoCurPt => f.write_str(
                "path error: no current point \
                 (call move_to before line_to, curve_to, or close)",
            ),
            Self::BogusPath => f.write_str(
                "path error: consecutive moveTo without a drawing operator \
                 (a one-point subpath is already active)",
            ),
        }
    }
}

impl std::error::Error for PathError {}

// ── Path ──────────────────────────────────────────────────────────────────────

/// A PDF graphics path: an ordered sequence of subpaths built from lines and
/// cubic Bezier curves.
///
/// # Invariants
///
/// These match `SplashPath.cc`:
///
/// - `pts.len() == flags.len()` at all times.
/// - `cur_subpath` is the index of the first point of the currently open
///   subpath, or equals `pts.len()` when there is no current point.
/// - Control points (flag [`PathFlags::CURVE`]) always appear in groups of
///   three between two on-curve points.
#[derive(Clone, Debug, Default)]
pub struct Path {
    /// Ordered sequence of path points.
    pub pts: Vec<PathPoint>,
    /// Per-point flags, parallel to [`Self::pts`].
    pub flags: Vec<PathFlags>,
    /// Optional stroke-adjust hints.
    pub hints: Vec<StrokeAdjustHint>,
    /// Index of the first point of the currently open subpath.
    ///
    /// Equals `pts.len()` when there is no current point (fresh path or
    /// immediately after a `close`).
    pub cur_subpath: usize,
}

impl Path {
    /// Create an empty path with no current point.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    // ── State predicates ──────────────────────────────────────────────────────

    /// Returns `true` when there is no current point.
    ///
    /// This is the case for a freshly created path and immediately after
    /// [`PathBuilder::close`] completes.
    #[inline]
    #[must_use]
    pub const fn no_current_point(&self) -> bool {
        self.cur_subpath == self.pts.len()
    }

    /// Returns `true` after exactly one [`PathBuilder::move_to`] with no
    /// subsequent [`PathBuilder::line_to`] or [`PathBuilder::curve_to`].
    ///
    /// # Underflow safety
    ///
    /// The `!self.pts.is_empty()` guard ensures `pts.len() - 1` is evaluated
    /// only when `pts` has at least one element, so no wrapping subtraction
    /// can occur.
    #[inline]
    #[must_use]
    pub const fn one_point_subpath(&self) -> bool {
        !self.pts.is_empty() && self.cur_subpath == self.pts.len() - 1
    }

    /// Returns `true` when the current subpath has at least two points.
    #[inline]
    #[must_use]
    pub const fn open_subpath(&self) -> bool {
        !self.pts.is_empty() && self.cur_subpath < self.pts.len() - 1
    }

    /// Returns the current point (last appended endpoint), if any.
    ///
    /// Returns `None` when [`Self::no_current_point`] is true — i.e. on a
    /// fresh path **and** immediately after a `close` (because `close` sets
    /// `cur_subpath` to `pts.len()`).
    #[must_use]
    pub fn current_point(&self) -> Option<PathPoint> {
        if self.no_current_point() {
            None
        } else {
            self.pts.last().copied()
        }
    }

    // ── Geometry ──────────────────────────────────────────────────────────────

    /// Translate every point in the path by `(dx, dy)`.
    pub fn offset(&mut self, dx: f64, dy: f64) {
        for p in &mut self.pts {
            p.x += dx;
            p.y += dy;
        }
    }

    /// Append all points and hints from `other` into `self`.
    ///
    /// `cur_subpath` is set to `self.pts.len() + other.cur_subpath` **before**
    /// appending, matching `SplashPath::append`.
    ///
    /// # Edge cases
    ///
    /// If `other` is empty (`other.pts.is_empty()`), then
    /// `other.cur_subpath == 0` (the default) and
    /// `self.cur_subpath` is set to `self.pts.len()` — the "no current point"
    /// sentinel — which is correct: appending an empty path does not create a
    /// current point.
    pub fn append(&mut self, other: &Self) {
        debug_assert!(
            other.cur_subpath <= other.pts.len(),
            "append: other.cur_subpath ({}) exceeds other.pts.len() ({}); invariant broken",
            other.cur_subpath,
            other.pts.len()
        );
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

/// Ergonomic builder for [`Path`] implementing the PDF path construction
/// operators (`m`, `l`, `c`, `h`) with the same state-machine semantics as
/// `SplashPath::moveTo` / `lineTo` / `curveTo` / `close`.
pub struct PathBuilder {
    path: Path,
}

impl PathBuilder {
    /// Create a new, empty builder with no current point.
    #[must_use]
    pub fn new() -> Self {
        Self { path: Path::new() }
    }

    /// Begin a new subpath at `(x, y)`. Equivalent to the PDF `m` operator.
    ///
    /// # Errors
    ///
    /// Returns [`PathError::BogusPath`] if a one-point subpath is already
    /// active (i.e. the previous operation was also a `move_to` with no
    /// drawing operator in between). Callers must not silently ignore this
    /// error: it indicates a malformed path construction sequence.
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

    /// Add a line segment from the current point to `(x, y)`.
    ///
    /// Equivalent to the PDF `l` operator.
    ///
    /// # Errors
    ///
    /// Returns [`PathError::NoCurPt`] if there is no current point. Callers
    /// must ensure a successful [`Self::move_to`] precedes this call.
    ///
    /// # Panics
    ///
    /// Panics if `flags` is empty despite a current point existing, which
    /// would indicate a broken `Path` invariant (`pts.len() == flags.len()`).
    pub fn line_to(&mut self, x: f64, y: f64) -> Result<(), PathError> {
        if self.path.no_current_point() {
            return Err(PathError::NoCurPt);
        }
        // Clear LAST on the previous endpoint before appending the new one.
        let last = self.path.flags.last_mut().unwrap();
        last.remove(PathFlags::LAST);
        self.path.pts.push(PathPoint::new(x, y));
        self.path.flags.push(PathFlags::LAST);
        Ok(())
    }

    /// Add a cubic Bezier curve. Equivalent to the PDF `c` operator.
    ///
    /// `(x1, y1)` and `(x2, y2)` are the two off-curve control points;
    /// `(x3, y3)` is the on-curve endpoint. Three points are always appended:
    /// the control points receive [`PathFlags::CURVE`] and the endpoint
    /// receives [`PathFlags::LAST`].
    ///
    /// # Errors
    ///
    /// Returns [`PathError::NoCurPt`] if there is no current point. Callers
    /// must ensure a successful [`Self::move_to`] precedes this call.
    ///
    /// # Panics
    ///
    /// Panics if `flags` is empty despite a current point existing, which
    /// would indicate a broken `Path` invariant (`pts.len() == flags.len()`).
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
        // Clear LAST on the previous endpoint.
        let last = self.path.flags.last_mut().unwrap();
        last.remove(PathFlags::LAST);
        // Two off-curve control points tagged CURVE, then the on-curve endpoint.
        self.path.pts.push(PathPoint::new(x1, y1));
        self.path.flags.push(PathFlags::CURVE);
        self.path.pts.push(PathPoint::new(x2, y2));
        self.path.flags.push(PathFlags::CURVE);
        self.path.pts.push(PathPoint::new(x3, y3));
        self.path.flags.push(PathFlags::LAST);
        Ok(())
    }

    /// Close the current subpath. Equivalent to the PDF `h` operator.
    ///
    /// Behaviour:
    ///
    /// - If `force` is `true`, a closing `lineTo(first)` is **always**
    ///   appended.
    /// - If `sp == last_idx` the subpath consists of exactly one point (the
    ///   `moveTo` with no drawing operators).  In this degenerate case the
    ///   subpath is trivially "closed" (first == last by identity), so no
    ///   extra `lineTo` is needed — the single point has [`PathFlags::CLOSED`]
    ///   set on itself.  This matches the C++ `SplashPath::close` behaviour.
    /// - Otherwise, a closing `lineTo(first)` is appended only when
    ///   `first != last` (the path is not already closed geometrically).
    ///
    /// After closing, `cur_subpath` is advanced to `pts.len()` (the
    /// "no current point" sentinel), so [`Path::current_point`] returns `None`
    /// until the next `move_to`.
    ///
    /// # Errors
    ///
    /// Returns [`PathError::NoCurPt`] if there is no current point. The `?`
    /// inside this method propagates any error from the internal `line_to`
    /// call; since `line_to` only errors on `NoCurPt` and we have already
    /// verified a current point exists at entry, that propagation path is only
    /// reachable if an invariant is broken.
    pub fn close(&mut self, force: bool) -> Result<(), PathError> {
        if self.path.no_current_point() {
            return Err(PathError::NoCurPt);
        }
        let sp = self.path.cur_subpath;
        let last_idx = self.path.pts.len() - 1;
        let first = self.path.pts[sp];
        let last = self.path.pts[last_idx];

        // Add a closing lineTo(first) when:
        //   • `force` is set, OR
        //   • the subpath has exactly one point (sp == last_idx) — no lineTo
        //     is needed but we still fall through to stamp CLOSED, OR
        //   • first != last (not yet geometrically closed).
        //
        // The `sp == last_idx` branch skips the `line_to` call because the
        // condition is placed *before* the `first != last` check.  The `?`
        // propagation is correct: we hold a current point so `line_to` can
        // only fail if the invariant `pts.len() == flags.len()` is broken.
        if force || (sp != last_idx && first != last) {
            self.line_to(first.x, first.y)?;
        }
        debug_assert_eq!(
            self.path.pts.len(),
            self.path.flags.len(),
            "close: pts/flags length invariant violated"
        );

        // Stamp CLOSED on the first and last point of the subpath.
        let new_last = self.path.pts.len() - 1;
        self.path.flags[sp].insert(PathFlags::CLOSED);
        self.path.flags[new_last].insert(PathFlags::CLOSED);

        // Advance past this subpath → "no current point" state.
        self.path.cur_subpath = self.path.pts.len();
        Ok(())
    }

    /// Add a stroke-adjust hint referencing existing point indices.
    ///
    /// Indices refer to positions in [`Path::pts`] at build time.
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

    /// Returns the number of points accumulated in the builder so far.
    ///
    /// This is a read-only view used by callers that need to record point
    /// indices for stroke-adjustment hints (e.g. `raster::stroke::make_stroke_path`).
    #[must_use]
    pub const fn pts_len(&self) -> usize {
        self.path.pts.len()
    }

    /// Translate every point accumulated so far by `(dx, dy)`.
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

    // ── State-machine basics ───────────────────────────────────────────────────

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

    // ── curve_to ──────────────────────────────────────────────────────────────

    #[test]
    fn curve_to_adds_three_points() {
        let mut b = PathBuilder::new();
        b.move_to(0.0, 0.0).unwrap();
        b.curve_to(1.0, 2.0, 3.0, 4.0, 5.0, 0.0).unwrap();
        // moveTo + 2 control points + 1 endpoint = 4
        assert_eq!(b.path.pts.len(), 4);
        assert!(b.path.flags[1].is_curve());
        assert!(b.path.flags[2].is_curve());
        assert!(b.path.flags[3].is_last());
        assert!(!b.path.flags[3].is_curve());
    }

    // ── close ─────────────────────────────────────────────────────────────────

    #[test]
    fn close_sets_closed_flag() {
        let mut b = PathBuilder::new();
        b.move_to(0.0, 0.0).unwrap();
        b.line_to(10.0, 0.0).unwrap();
        b.line_to(5.0, 5.0).unwrap();
        b.close(false).unwrap();
        assert!(b.path.flags[0].is_closed());
        assert!(b.path.flags.last().unwrap().is_closed());
        assert!(b.path.no_current_point());
    }

    /// After a close, `current_point()` must return `None` because `close`
    /// advances `cur_subpath` to `pts.len()`.
    #[test]
    fn after_close_current_point_is_none() {
        let mut b = PathBuilder::new();
        b.move_to(0.0, 0.0).unwrap();
        b.line_to(1.0, 0.0).unwrap();
        b.close(false).unwrap();
        assert_eq!(b.path.current_point(), None);
    }

    /// A one-point subpath (`moveTo` with no drawing operators) should have
    /// [`PathFlags::CLOSED`] set on that single point when `close` is called.
    /// The single point acts as both first and last, so both writes target
    /// `pts[sp]` — which is correct per `SplashPath::close`.
    #[test]
    fn close_one_point_subpath_sets_closed_flag() {
        let mut b = PathBuilder::new();
        b.move_to(3.0, 4.0).unwrap();
        assert!(b.path.one_point_subpath());
        b.close(false).unwrap();
        // The single point must carry CLOSED.
        assert!(b.path.flags[0].is_closed());
        // After close there is no current point.
        assert!(b.path.no_current_point());
        // No extra point should have been appended.
        assert_eq!(b.path.pts.len(), 1);
    }

    // ── Error paths ───────────────────────────────────────────────────────────

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

    // ── PathError::Display ────────────────────────────────────────────────────

    #[test]
    fn path_error_display() {
        let no_pt = PathError::NoCurPt.to_string();
        assert!(
            no_pt.contains("no current point"),
            "NoCurPt display should mention 'no current point', got: {no_pt}"
        );

        let bogus = PathError::BogusPath.to_string();
        assert!(
            bogus.contains("consecutive moveTo"),
            "BogusPath display should mention 'consecutive moveTo', got: {bogus}"
        );
    }

    // ── From<(f64, f64)> / From<PathPoint> ───────────────────────────────────

    #[test]
    #[expect(
        clippy::float_cmp,
        reason = "testing exact round-trip identity through From impls, not approximate equality"
    )]
    fn from_tuple_pathpoint() {
        let p: PathPoint = (1.5_f64, 2.5_f64).into();
        assert_eq!(p.x, 1.5);
        assert_eq!(p.y, 2.5);

        let t: (f64, f64) = p.into();
        assert_eq!(t, (1.5, 2.5));
    }

    // ── PathFlags helpers ─────────────────────────────────────────────────────

    #[test]
    fn path_flags_helpers() {
        let f = PathFlags::FIRST | PathFlags::LAST | PathFlags::CLOSED | PathFlags::CURVE;
        assert!(f.is_first());
        assert!(f.is_last());
        assert!(f.is_closed());
        assert!(f.is_curve());

        let empty = PathFlags::empty();
        assert!(!empty.is_first());
        assert!(!empty.is_last());
        assert!(!empty.is_closed());
        assert!(!empty.is_curve());
    }

    // ── Path::append ──────────────────────────────────────────────────────────

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
        // Hint indices should be offset by original pa.pts.len() = 2.
        assert_eq!(b_path.hints[0].ctrl0, 2);
        assert_eq!(b_path.hints[0].first_pt, 2);
    }

    /// Appending an empty path must leave `self` in the no-current-point state
    /// and must not panic.
    #[test]
    fn append_empty_other_is_safe() {
        let mut base = PathBuilder::new();
        base.move_to(0.0, 0.0).unwrap();
        base.line_to(1.0, 0.0).unwrap();
        let mut p = base.build();
        let original_len = p.pts.len();

        p.append(&Path::new());

        assert_eq!(p.pts.len(), original_len);
        // cur_subpath == pts.len() → no current point.
        assert!(p.no_current_point());
    }
}
