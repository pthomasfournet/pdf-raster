//! `FreeType` outline → `raster::Path` decomposition.
//!
//! Mirrors the four `glyphPath*` callbacks in `SplashFTFont.cc`.
//!
//! `FreeType` outlines may contain quadratic (conic) Bézier segments.
//! PDF paths use only cubics, so each conic is promoted to a cubic using
//! the exact degree-elevation formula:
//!
//! ```text
//! p1 = (p0 + 2·pc) / 3
//! p2 = (2·pc + p3) / 3
//! ```
//!
//! where `pc` is the conic control point.  This matches `glyphPathConicTo`
//! in `SplashFTFont.cc` and is mathematically exact.

use freetype::Vector;
use freetype::outline::{Curve, Outline};
use raster::path::{Path, PathBuilder, PathPoint};

/// Decompose a `FreeType` `Outline` into a [`Path`].
///
/// `text_scale` converts from `FreeType`'s 26.6 fixed-point units to the
/// text-space coordinate system: `coord = ft_units / 64.0 * text_scale`.
///
/// Each `FreeType` contour becomes one closed subpath.  The implicit
/// close-back is applied matching `SplashFTFont.cc`'s `path.needClose`
/// pattern: a close is emitted only when at least one drawing operator
/// was issued after the contour's move-to.
///
/// # Return value
///
/// Returns `None` if `text_scale == 0.0` (degenerate transform — the
/// outline cannot be scaled to text space).  Non-zero `text_scale` values
/// with a valid `FreeType` outline always succeed.
#[must_use]
pub fn decompose_outline(outline: &Outline<'_>, text_scale: f64) -> Option<Path> {
    if text_scale == 0.0 {
        return None;
    }

    let mut builder = PathBuilder::new();

    for contour in outline.contours_iter() {
        let start = contour.start();
        // Each contour begins with an implicit move-to its first on-curve point.
        // The one-point-subpath error cannot occur here: we never emit two
        // consecutive move_to calls because there is always at least one
        // drawing operator between contours.
        let _ = builder.move_to(ft_x(*start, text_scale), ft_y(*start, text_scale));

        let mut need_close = false;

        for curve in contour {
            match curve {
                Curve::Line(pt) => {
                    // NoCurPt is impossible after the move_to above.
                    let _ = builder.line_to(ft_x(pt, text_scale), ft_y(pt, text_scale));
                    need_close = true;
                }
                Curve::Bezier2(ctrl, pt) => {
                    // Conic → cubic degree elevation requires the current point p0.
                    if let Some(p0) = builder.cur_pt() {
                        let xc = ft_x(ctrl, text_scale);
                        let yc = ft_y(ctrl, text_scale);
                        let x3 = ft_x(pt, text_scale);
                        let y3 = ft_y(pt, text_scale);
                        let (p1, p2) =
                            conic_to_cubic(p0, PathPoint::new(xc, yc), PathPoint::new(x3, y3));
                        let _ = builder.curve_to(p1.x, p1.y, p2.x, p2.y, x3, y3);
                        need_close = true;
                    }
                    // No current point: drop silently (matches C++ getCurPt → return 0).
                }
                Curve::Bezier3(c1, c2, pt) => {
                    let _ = builder.curve_to(
                        ft_x(c1, text_scale),
                        ft_y(c1, text_scale),
                        ft_x(c2, text_scale),
                        ft_y(c2, text_scale),
                        ft_x(pt, text_scale),
                        ft_y(pt, text_scale),
                    );
                    need_close = true;
                }
            }
        }

        if need_close {
            // force=false: close only if first ≠ last (C++ `SplashPath::close` semantics).
            let _ = builder.close(false);
        }
    }

    Some(builder.build())
}

/// Elevate a quadratic (conic) Bézier to a cubic.
///
/// This is the formula used by `glyphPathConicTo` in `SplashFTFont.cc`.
/// The conversion is mathematically exact — no approximation is introduced.
///
/// Returns `(p1, p2)` — the two cubic off-curve control points.
#[inline]
#[must_use]
pub fn conic_to_cubic(p0: PathPoint, pc: PathPoint, p3: PathPoint) -> (PathPoint, PathPoint) {
    let x1 = p0.x.mul_add(1.0 / 3.0, pc.x * (2.0 / 3.0));
    let y1 = p0.y.mul_add(1.0 / 3.0, pc.y * (2.0 / 3.0));
    let x2 = pc.x.mul_add(2.0 / 3.0, p3.x * (1.0 / 3.0));
    let y2 = pc.y.mul_add(2.0 / 3.0, p3.y * (1.0 / 3.0));
    (PathPoint::new(x1, y1), PathPoint::new(x2, y2))
}

/// Convert `FreeType` 26.6 fixed-point x coordinate to f64 in text space.
#[inline]
#[expect(
    clippy::cast_precision_loss,
    reason = "FT_Pos is i64; typical glyph coordinates are <2^20, well within f64 mantissa precision"
)]
fn ft_x(pt: Vector, scale: f64) -> f64 {
    pt.x as f64 * scale / 64.0
}

/// Convert `FreeType` 26.6 fixed-point y coordinate to f64 in text space.
#[inline]
#[expect(
    clippy::cast_precision_loss,
    reason = "FT_Pos is i64; typical glyph coordinates are <2^20, well within f64 mantissa precision"
)]
fn ft_y(pt: Vector, scale: f64) -> f64 {
    pt.y as f64 * scale / 64.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use raster::path::PathPoint;

    fn pt(x: f64, y: f64) -> PathPoint {
        PathPoint::new(x, y)
    }

    #[test]
    fn conic_to_cubic_control_points() {
        // p0=(0,0), pc=(1,1), p3=(2,0):
        //   p1 = (0 + 2)/3 = 2/3
        //   p2 = (2 + 2)/3 = 4/3
        let (p1, p2) = conic_to_cubic(pt(0.0, 0.0), pt(1.0, 1.0), pt(2.0, 0.0));
        assert!((p1.x - 2.0 / 3.0).abs() < 1e-12, "p1.x = {}", p1.x);
        assert!((p1.y - 2.0 / 3.0).abs() < 1e-12, "p1.y = {}", p1.y);
        assert!((p2.x - 4.0 / 3.0).abs() < 1e-12, "p2.x = {}", p2.x);
        assert!((p2.y - 2.0 / 3.0).abs() < 1e-12, "p2.y = {}", p2.y);
    }

    #[test]
    fn conic_to_cubic_collinear_stays_collinear() {
        // Collinear control point → the cubic approximation is a straight line.
        let p0 = pt(0.0, 0.0);
        let p3 = pt(3.0, 0.0);
        let pc = pt(1.5, 0.0);
        let (p1, p2) = conic_to_cubic(p0, pc, p3);
        assert!(p1.y.abs() < 1e-12, "collinear conic: p1.y must be zero");
        assert!(p2.y.abs() < 1e-12, "collinear conic: p2.y must be zero");
        assert!((p1.x - 1.0).abs() < 1e-12, "p1.x = {}", p1.x);
        assert!((p2.x - 2.0).abs() < 1e-12, "p2.x = {}", p2.x);
    }

    #[test]
    fn ft_x_converts_26_6_to_f64() {
        use freetype::Vector;
        let v = Vector { x: 128, y: 0 }; // 128 / 64 = 2.0
        assert!((ft_x(v, 1.0) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn ft_x_applies_text_scale() {
        use freetype::Vector;
        let v = Vector { x: 64, y: 0 }; // 64 / 64 = 1.0 → × scale = 3.0
        assert!((ft_x(v, 3.0) - 3.0).abs() < 1e-12);
    }
}
