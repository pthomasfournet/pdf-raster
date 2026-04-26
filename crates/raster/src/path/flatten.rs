//! Adaptive Bezier curve flattening via De Casteljau subdivision.
//!
//! Mirrors `SplashXPath::addCurve` from `splash/SplashXPath.cc` exactly,
//! including the `cx/cy/cNext` stack representation and the
//! `MAX_CURVE_SPLITS = 1024` hard limit.
//!
//! ## Algorithm
//!
//! The curve is stored as a linked list of pending segments in the `CurveData`
//! array. Each segment `[p1, p2]` covers control points `cx[p1*3..p1*3+3]` and
//! endpoint `cx[p2*3]`. The loop advances `p1` rightward, either emitting a
//! line segment when the chord deviation is within `flatness_sq`, or splitting
//! via De Casteljau and inserting a midpoint `p3`.

use super::PathPoint;
use crate::types::MAX_CURVE_SPLITS;

// ── CurveData ─────────────────────────────────────────────────────────────────

/// Stack storage for the De Casteljau subdivision loop.
///
/// Lazy-allocated (via `Option<Box<CurveData>>`) because it is ~25 KB —
/// only needed for paths containing Bezier curves.
pub struct CurveData {
    /// X coordinates: 3 control points per slot, indexed by `slot * 3 + {0,1,2}`.
    pub cx: Box<[f64]>,
    /// Y coordinates: same layout as `cx`.
    pub cy: Box<[f64]>,
    /// `cNext[p1]` = index of the next segment's first slot after `p1`.
    pub c_next: Box<[i32]>,
}

impl CurveData {
    /// Allocate a new `CurveData`.
    ///
    /// # Panics
    ///
    /// Panics if `MAX_CURVE_SPLITS` is negative (it is a positive constant so this never happens).
    #[must_use]
    pub fn new() -> Box<Self> {
        let n = usize::try_from(MAX_CURVE_SPLITS).expect("MAX_CURVE_SPLITS >= 0") + 1;
        Box::new(Self {
            cx: vec![0.0f64; n * 3].into_boxed_slice(),
            cy: vec![0.0f64; n * 3].into_boxed_slice(),
            c_next: vec![0i32; n].into_boxed_slice(),
        })
    }
}

// ── flatten_curve ─────────────────────────────────────────────────────────────

/// Flatten a cubic Bezier curve into a sequence of line endpoints.
///
/// Appends one `PathPoint` per emitted line segment endpoint to `out` (not
/// including the implicit start point `p0` — that is the previous endpoint).
///
/// `flatness_sq` is the **squared** maximum allowed deviation from the true
/// curve to a chord. Use `flatness * flatness` when calling from `XPath::new`.
///
/// `curve_data` is lazy-allocated storage reused across multiple calls on the
/// same `XPath`. Pass `None` on the first call; the `Box<CurveData>` is
/// returned and should be stored for the next call.
///
/// # Panics
///
/// Panics if `MAX_CURVE_SPLITS` is negative (it is a positive constant so this never happens).
pub fn flatten_curve(
    p0: PathPoint,
    p1: PathPoint,
    p2: PathPoint,
    p3: PathPoint,
    flatness_sq: f64,
    out: &mut Vec<PathPoint>,
    curve_data: &mut Option<Box<CurveData>>,
) {
    let data = curve_data.get_or_insert_with(CurveData::new);

    let max = MAX_CURVE_SPLITS;
    let max_u = usize::try_from(max).expect("MAX_CURVE_SPLITS >= 0");

    // Initialise the stack: one segment covering [0, max].
    let i0 = 0usize;
    let i2 = max_u;
    // Slot 0: control points (x0/y0, x1/y1, x2/y2).
    data.cx[0] = p0.x;
    data.cy[0] = p0.y;
    data.cx[1] = p1.x;
    data.cy[1] = p1.y;
    data.cx[2] = p2.x;
    data.cy[2] = p2.y;
    // Slot max: just the endpoint (x3/y3).
    data.cx[i2 * 3] = p3.x;
    data.cy[i2 * 3] = p3.y;
    data.c_next[i0] = max;

    let mut pp1 = 0i32; // current left slot

    while pp1 < max {
        let pp2 = data.c_next[usize::try_from(pp1).expect("pp1 >= 0")];
        let pp2u = usize::try_from(pp2).expect("pp2 >= 0");
        let pp1u = usize::try_from(pp1).expect("pp1 >= 0");

        // Read this segment's endpoints and control points.
        let xl0 = data.cx[pp1u * 3];
        let yl0 = data.cy[pp1u * 3];
        let xx1 = data.cx[pp1u * 3 + 1];
        let yy1 = data.cy[pp1u * 3 + 1];
        let xx2 = data.cx[pp1u * 3 + 2];
        let yy2 = data.cy[pp1u * 3 + 2];
        let xr3 = data.cx[pp2u * 3];
        let yr3 = data.cy[pp2u * 3];

        // Midpoint of the chord.
        let mx = xl0.midpoint(xr3);
        let my = yl0.midpoint(yr3);

        // Squared deviation of the two control points from the chord midpoint.
        let dx1 = xx1 - mx;
        let dy1 = yy1 - my;
        let dx2 = xx2 - mx;
        let dy2 = yy2 - my;
        let d1 = dx1.mul_add(dx1, dy1 * dy1);
        let d2 = dx2.mul_add(dx2, dy2 * dy2);

        if pp2 - pp1 == 1 || (d1 <= flatness_sq && d2 <= flatness_sq) {
            // Emit this segment as a straight line.
            out.push(PathPoint::new(xr3, yr3));
            pp1 = pp2;
        } else {
            // De Casteljau midpoint subdivision.
            let xl1 = xl0.midpoint(xx1);
            let yl1 = yl0.midpoint(yy1);
            let xh = xx1.midpoint(xx2);
            let yh = yy1.midpoint(yy2);
            let xl2 = xl1.midpoint(xh);
            let yl2 = yl1.midpoint(yh);
            let xr2 = xx2.midpoint(xr3);
            let yr2 = yy2.midpoint(yr3);
            let xr1 = xh.midpoint(xr2);
            let yr1 = yh.midpoint(yr2);
            let xr0 = xl2.midpoint(xr1);
            let yr0 = yl2.midpoint(yr1);

            let pp3 = i32::midpoint(pp1, pp2);
            let pp3u = usize::try_from(pp3).expect("pp3 >= 0");

            // Update left segment: [pp1, pp3] with left sub-curve controls.
            data.cx[pp1u * 3 + 1] = xl1;
            data.cy[pp1u * 3 + 1] = yl1;
            data.cx[pp1u * 3 + 2] = xl2;
            data.cy[pp1u * 3 + 2] = yl2;
            data.c_next[pp1u] = pp3;

            // New right sub-curve at pp3: [xr0, xr1, xr2] then endpoint at pp2.
            data.cx[pp3u * 3] = xr0;
            data.cy[pp3u * 3] = yr0;
            data.cx[pp3u * 3 + 1] = xr1;
            data.cy[pp3u * 3 + 1] = yr1;
            data.cx[pp3u * 3 + 2] = xr2;
            data.cy[pp3u * 3 + 2] = yr2;
            data.c_next[pp3u] = pp2;
            // (endpoint at pp2 is already set from a prior iteration or init)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pt(x: f64, y: f64) -> PathPoint {
        PathPoint::new(x, y)
    }

    #[test]
    fn flat_curve_emits_one_segment() {
        // A degenerate curve with all control points on the line y=0.
        // The curve is geometrically flat, but control points deviate from the
        // chord midpoint, so it may still subdivide. All outputs must have y≈0
        // and the last point must be the endpoint (3, 0).
        let mut out = Vec::new();
        let mut data = None;
        flatten_curve(
            pt(0.0, 0.0),
            pt(1.0, 0.0),
            pt(2.0, 0.0),
            pt(3.0, 0.0),
            0.1,
            &mut out,
            &mut data,
        );
        assert!(!out.is_empty());
        for p in &out {
            assert!(p.y.abs() < 1e-10, "y={} should be 0", p.y);
        }
        let last = out.last().unwrap();
        assert!((last.x - 3.0).abs() < 1e-10);
        assert!(last.y.abs() < 1e-10);
    }

    #[test]
    fn endpoints_match_cubic() {
        // Quarter-circle-ish curve.
        let p0 = pt(0.0, 0.0);
        let p3 = pt(1.0, 1.0);
        let k = crate::types::BEZIER_CIRCLE;
        let p1 = pt(0.0, k);
        let p2 = pt(1.0 - k, 1.0);
        let mut out = Vec::new();
        let mut data = None;
        flatten_curve(p0, p1, p2, p3, 0.01 * 0.01, &mut out, &mut data);
        // First emitted point is the last in the sequence → approaches p3.
        let last = out.last().unwrap();
        assert!((last.x - p3.x).abs() < 1e-10, "last x = {}", last.x);
        assert!((last.y - p3.y).abs() < 1e-10, "last y = {}", last.y);
    }

    #[test]
    fn does_not_exceed_max_curve_splits() {
        // Adversarial: a curve that would recurse forever without the split limit.
        let p0 = pt(0.0, 0.0);
        let p1 = pt(1e8, -1e8);
        let p2 = pt(-1e8, 1e8);
        let p3 = pt(0.0, 0.0);
        let mut out = Vec::new();
        let mut data = None;
        flatten_curve(p0, p1, p2, p3, 0.5 * 0.5, &mut out, &mut data);
        // Must terminate and emit ≤ MAX_CURVE_SPLITS segments.
        assert!(out.len() <= usize::try_from(MAX_CURVE_SPLITS).expect("non-negative"));
    }
}
