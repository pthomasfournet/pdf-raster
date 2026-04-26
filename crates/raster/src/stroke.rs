//! Stroke rasterization — replaces `Splash::stroke`, `Splash::strokeNarrow`,
//! `Splash::strokeWide`, `Splash::flattenPath`, `Splash::makeDashedPath`, and
//! `Splash::makeStrokePath` from `splash/Splash.cc`.
//!
//! # Entry point
//!
//! [`stroke`] is the top-level function. It flattens curves, applies dashing if
//! needed, then either draws hairlines with [`stroke_narrow`] (zero-width) or
//! expands the stroke outline with [`make_stroke_path`] and fills it
//! ([`stroke_wide`]).
//!
//! # C++ correspondence
//!
//! | Rust function          | C++ function            | Approx. line |
//! |------------------------|-------------------------|-------------|
//! | [`stroke`]             | `Splash::stroke`        | ~1950        |
//! | [`stroke_narrow`]      | `Splash::strokeNarrow`  | ~2005        |
//! | [`stroke_wide`]        | `Splash::strokeWide`    | ~2094        |
//! | [`flatten_path`]       | `Splash::flattenPath`   | ~2100        |
//! | [`make_dashed_path`]   | `Splash::makeDashedPath`| ~2221        |
//! | [`make_stroke_path`]   | `Splash::makeStrokePath`| ~6091        |

use std::f64::consts::PI;

use crate::bitmap::Bitmap;
use crate::clip::{Clip, ClipResult};
use crate::fill::fill;
use crate::path::{Path, PathBuilder};
use crate::pipe::{PipeSrc, PipeState};
use crate::types::{BEZIER_CIRCLE, LineCap, LineJoin, splash_floor};
use crate::xpath::XPath;
use color::Pixel;

// ── Local constants ───────────────────────────────────────────────────────────

/// Half of `BEZIER_CIRCLE`, used for round-cap / round-join arc approximation.
/// C++ origin: `#define bezierCircle2 (0.5 * 0.55228475)` in `Splash.cc`.
const BEZIER_CIRCLE2: f64 = 0.5 * BEZIER_CIRCLE;

// ── Inline geometry helper ────────────────────────────────────────────────────

/// Euclidean distance between two points.
///
/// Equivalent to `splashDist(x0, y0, x1, y1)` in the C++ source.
#[inline]
fn splash_dist(x0: f64, y0: f64, x1: f64, y1: f64) -> f64 {
    let dx = x1 - x0;
    let dy = y1 - y0;
    dx.hypot(dy)
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Parameters governing how a stroke is rendered.
///
/// All fields mirror the correspondingly named fields on the C++ `SplashState`.
pub struct StrokeParams<'a> {
    /// Stroke line width in user space.
    pub line_width: f64,
    /// Line cap style (Butt / Round / Projecting).
    pub line_cap: LineCap,
    /// Line join style (Miter / Round / Bevel).
    pub line_join: LineJoin,
    /// Miter limit (dimensionless ratio).
    pub miter_limit: f64,
    /// Flatness for curve flattening (maximum chord deviation, device pixels).
    pub flatness: f64,
    /// Whether stroke adjustment is enabled.
    pub stroke_adjust: bool,
    /// Dash array; empty slice means solid line.
    pub line_dash: &'a [f64],
    /// Phase offset into the dash array.
    pub line_dash_phase: f64,
    /// Whether vector anti-aliasing is requested.
    pub vector_antialias: bool,
}

/// Top-level stroke entry point.
///
/// Mirrors `Splash::stroke` (~line 1950 of `Splash.cc`).
///
/// Steps:
/// 1. Flatten curves via [`flatten_path`].
/// 2. Apply dashing via [`make_dashed_path`] if a dash array is set.
/// 3. Choose hairline ([`stroke_narrow`]) or wide ([`stroke_wide`]) rendering.
pub fn stroke<P: Pixel>(
    bitmap: &mut Bitmap<P>,
    clip: &Clip,
    path: &Path,
    pipe: &PipeState<'_>,
    src: &PipeSrc<'_>,
    matrix: &[f64; 6],
    params: &StrokeParams<'_>,
) {
    if path.pts.is_empty() {
        return;
    }

    let mut path2 = flatten_path(path, matrix, params.flatness);

    if !params.line_dash.is_empty() {
        path2 = make_dashed_path(&path2, params.line_dash, params.line_dash_phase);
        if path2.pts.is_empty() {
            return;
        }
    }

    // Compute the approximate device-space line width by transforming a unit
    // square and taking half the maximum diagonal length. This mirrors the C++
    // computation in `Splash::stroke` that decides narrow vs wide.
    // (We skip the minLineWidth check as instructed.)
    if params.line_width == 0.0 {
        stroke_narrow::<P>(bitmap, clip, &path2, pipe, src, matrix, params.flatness);
    } else {
        stroke_wide::<P>(bitmap, clip, &path2, pipe, src, matrix, params);
    }
}

/// Draw zero-width (hairline) strokes: one pixel per scanline intersection.
///
/// Mirrors `Splash::strokeNarrow` (~line 2005 of `Splash.cc`).
///
/// Each segment of the flattened [`XPath`] is walked scanline-by-scanline,
/// drawing a single span (possibly one pixel wide) per row.
pub fn stroke_narrow<P: Pixel>(
    bitmap: &mut Bitmap<P>,
    clip: &Clip,
    path: &Path,
    pipe: &PipeState<'_>,
    src: &PipeSrc<'_>,
    matrix: &[f64; 6],
    flatness: f64,
) {
    // Build a non-closing XPath (no implicit close for hairlines).
    let xpath = XPath::new(path, matrix, flatness, false);

    for seg in &xpath.segs {
        // Orient so that y0 <= y1.
        let (sx0, sy0, sx1, sy1) = if seg.y0 <= seg.y1 {
            (seg.x0, seg.y0, seg.x1, seg.y1)
        } else {
            (seg.x1, seg.y1, seg.x0, seg.y0)
        };

        let y0 = splash_floor(sy0);
        let y1 = splash_floor(sy1);
        let x0 = splash_floor(sx0);
        let x1 = splash_floor(sx1);

        let (xl, xr) = if x0 <= x1 { (x0, x1) } else { (x1, x0) };
        let clip_res = clip.test_rect(xl, y0, xr, y1);
        if clip_res == ClipResult::AllOutside {
            continue;
        }

        if y0 == y1 {
            // Horizontal or near-horizontal: draw one span.
            let (span_x0, span_x1) = if x0 <= x1 { (x0, x1) } else { (x1, x0) };
            draw_narrow_span::<P>(bitmap, clip, pipe, src, span_x0, span_x1, y0, clip_res);
        } else {
            // Sloped segment: walk scanlines.
            let dxdy = seg.dxdy;

            // Clip the y range to the clip rectangle.
            let (mut cy0, mut cx0) = (y0, x0);
            let (mut cy1, mut cx1) = (y1, x1);

            if cy0 < clip.y_min_i {
                cy0 = clip.y_min_i;
                cx0 = splash_floor((clip.y_min - sy0).mul_add(dxdy, sx0));
            }
            if cy1 > clip.y_max_i {
                cy1 = clip.y_max_i;
                cx1 = splash_floor((clip.y_max - sy0).mul_add(dxdy, sx0));
            }

            // Hoist the shared initialisation out of both branches.
            let mut xa = cx0;
            let left_to_right = cx0 <= cx1;
            for y in cy0..=cy1 {
                let xb = if y < cy1 {
                    splash_floor((f64::from(y) + 1.0 - sy0).mul_add(dxdy, sx0))
                } else if left_to_right {
                    cx1 + 1
                } else {
                    cx1 - 1
                };
                let (span_x0, span_x1) = if left_to_right {
                    if xa == xb { (xa, xa) } else { (xa, xb - 1) }
                } else if xa == xb {
                    (xa, xa)
                } else {
                    (xb + 1, xa)
                };
                draw_narrow_span::<P>(bitmap, clip, pipe, src, span_x0, span_x1, y, clip_res);
                xa = xb;
            }
        }
    }
}

/// Draw a span (possibly a single pixel), clamped to the bitmap and clipped.
///
/// Used by [`stroke_narrow`] to draw each hairline pixel-run.
#[expect(
    clippy::too_many_arguments,
    reason = "all parameters are required; splitting would add indirection"
)]
fn draw_narrow_span<P: Pixel>(
    bitmap: &mut Bitmap<P>,
    clip: &Clip,
    pipe: &PipeState<'_>,
    src: &PipeSrc<'_>,
    x0: i32,
    x1: i32,
    y: i32,
    clip_res: ClipResult,
) {
    if y < 0 {
        return;
    }
    #[expect(clippy::cast_sign_loss, reason = "y >= 0 checked above")]
    if (y as u32) >= bitmap.height {
        return;
    }
    // Clamp x to bitmap.
    #[expect(
        clippy::cast_possible_wrap,
        reason = "bitmap width fits in i32 in practice"
    )]
    let width_i = bitmap.width as i32;

    let (sx0, sx1) = if clip_res == ClipResult::AllInside {
        (x0.max(0), x1.min(width_i - 1))
    } else {
        (x0.max(clip.x_min_i), x1.min(clip.x_max_i))
    };

    if sx0 > sx1 {
        return;
    }

    if clip_res == ClipResult::AllInside {
        draw_span_unchecked::<P>(bitmap, pipe, src, sx0, sx1, y);
    } else {
        // Per-pixel clip test for partial regions.
        let mut run_start: Option<i32> = None;
        for x in sx0..=sx1 {
            if clip.test(x, y) {
                if run_start.is_none() {
                    run_start = Some(x);
                }
            } else if let Some(rs) = run_start.take() {
                draw_span_unchecked::<P>(bitmap, pipe, src, rs, x - 1, y);
            }
        }
        if let Some(rs) = run_start {
            draw_span_unchecked::<P>(bitmap, pipe, src, rs, sx1, y);
        }
    }
}

/// Write one span of pixels directly to the bitmap (no clip check).
fn draw_span_unchecked<P: Pixel>(
    bitmap: &mut Bitmap<P>,
    pipe: &PipeState<'_>,
    src: &PipeSrc<'_>,
    x0: i32,
    x1: i32,
    y: i32,
) {
    debug_assert!(x0 <= x1);
    debug_assert!(y >= 0);
    #[expect(clippy::cast_sign_loss, reason = "y >= 0")]
    let y_u = y as u32;
    #[expect(clippy::cast_sign_loss, reason = "x0 >= 0 after clamping")]
    let byte_off = x0 as usize * P::BYTES;
    #[expect(clippy::cast_sign_loss, reason = "x1 >= x0 >= 0")]
    let byte_end = (x1 as usize + 1) * P::BYTES;
    #[expect(clippy::cast_sign_loss, reason = "x0 >= 0, x1 >= x0")]
    let alpha_range = x0 as usize..=x1 as usize;

    let (row, alpha) = bitmap.row_and_alpha_mut(y_u);
    let dst_pixels = &mut row[byte_off..byte_end];
    let dst_alpha = alpha.map(|a| &mut a[alpha_range]);

    crate::pipe::render_span::<P>(pipe, src, dst_pixels, dst_alpha, None, x0, x1, y);
}

/// Wide stroke: expand the path into a filled outline and fill it.
///
/// Mirrors `Splash::strokeWide` (~line 2094 of `Splash.cc`).
pub fn stroke_wide<P: Pixel>(
    bitmap: &mut Bitmap<P>,
    clip: &Clip,
    path: &Path,
    pipe: &PipeState<'_>,
    src: &PipeSrc<'_>,
    matrix: &[f64; 6],
    params: &StrokeParams<'_>,
) {
    let outline = make_stroke_path(path, params.line_width, params);
    fill::<P>(
        bitmap,
        clip,
        &outline,
        pipe,
        src,
        matrix,
        params.flatness,
        params.vector_antialias,
    );
}

/// Flatten all Bezier curves in `path` to straight-line segments.
///
/// Mirrors `Splash::flattenPath` (~line 2100 of `Splash.cc`).
///
/// Control-point triples (flagged [`PathFlags::CURVE`]) are replaced by a
/// sequence of straight-line endpoints computed by adaptive De Casteljau
/// subdivision (implemented in [`crate::path::flatten::flatten_curve`]).
///
/// The resulting path contains only `FIRST`, `LAST`, and `CLOSED` flags —
/// no `CURVE` points remain.
///
/// Note: the `matrix` and `flatness` parameters are used by [`XPath`] internally
/// for device-space deviation; however, `flattenPath` in Splash does its
/// subdivision in **user space** against `flatness²` without a matrix transform.
/// We match that behaviour: the flatness is passed through unchanged.
#[must_use]
pub fn flatten_path(path: &Path, _matrix: &[f64; 6], flatness: f64) -> Path {
    use crate::path::flatten::{CurveData, flatten_curve};

    let flatness_sq = flatness * flatness;
    let mut builder = PathBuilder::new();
    let mut curve_data: Option<Box<CurveData>> = None;

    let mut i = 0usize;
    while i < path.pts.len() {
        let flag = path.flags[i];
        if flag.is_first() {
            // moveTo — start a new subpath.
            let _ = builder.move_to(path.pts[i].x, path.pts[i].y);
            i += 1;
        } else if flag.is_curve() {
            // curveTo: pts[i], pts[i+1] are off-curve; pts[i+2] is on-curve.
            // The implicit start is pts[i-1] (the previous current point).
            let p0 = path.pts[i - 1];
            let p1 = path.pts[i];
            let p2 = path.pts[i + 1];
            let p3 = path.pts[i + 2];

            let mut flat_pts: Vec<crate::path::PathPoint> = Vec::new();
            flatten_curve(p0, p1, p2, p3, flatness_sq, &mut flat_pts, &mut curve_data);

            for pt in &flat_pts {
                let _ = builder.line_to(pt.x, pt.y);
            }
            i += 3;

            // Check whether the last point in the curve had the CLOSED flag.
            if path.flags[i - 1].is_closed() {
                let _ = builder.close(false);
            }
        } else {
            // lineTo.
            let _ = builder.line_to(path.pts[i].x, path.pts[i].y);
            i += 1;

            if path.flags[i - 1].is_closed() {
                let _ = builder.close(false);
            }
        }
    }

    builder.build()
}

/// Convert a solid-line path into a dashed path.
///
/// Mirrors `Splash::makeDashedPath` (~line 2221 of `Splash.cc`).
///
/// Each subpath is broken at dash boundaries. A dash phase offset
/// (`line_dash_phase`) shifts the start of the dash pattern.
///
/// If the entire dashed path would be empty but all input points are
/// coincident, a degenerate (zero-length) subpath is emitted to allow end-caps
/// to be drawn, matching Acrobat behaviour.
#[must_use]
#[expect(
    clippy::suboptimal_flops,
    reason = "a + b*c expressions match the C++ source; mul_add would obscure the 1:1 correspondence"
)]
#[expect(
    clippy::while_float,
    reason = "direct port of Splash::makeDashedPath; seg_len is decremented toward 0 each iteration"
)]
pub fn make_dashed_path(path: &Path, line_dash: &[f64], line_dash_phase: f64) -> Path {
    // Sum the dash array.
    let line_dash_total: f64 = line_dash.iter().sum();
    if line_dash_total == 0.0 {
        return Path::new();
    }

    // Normalise phase to [0, total).
    let mut phase = line_dash_phase;
    {
        let periods = splash_floor(phase / line_dash_total);
        phase -= f64::from(periods) * line_dash_total;
    }

    // Find which dash entry we start in and how far into it.
    let mut dash_start_on = true;
    let mut dash_start_idx: usize = 0;
    if phase > 0.0 {
        while dash_start_idx < line_dash.len() && phase >= line_dash[dash_start_idx] {
            dash_start_on = !dash_start_on;
            phase -= line_dash[dash_start_idx];
            dash_start_idx += 1;
        }
        if dash_start_idx == line_dash.len() {
            return Path::new();
        }
    }

    let mut builder = PathBuilder::new();

    // Walk each subpath.
    let mut i = 0usize;
    while i < path.pts.len() {
        // Find the end of this subpath (the index of the LAST-flagged point).
        let mut j = i;
        while j < path.pts.len().saturating_sub(1) && !path.flags[j].is_last() {
            j += 1;
        }

        // Initialise dash state for this subpath.
        let mut dash_on = dash_start_on;
        let mut dash_idx = dash_start_idx;
        let mut dash_dist = line_dash[dash_idx] - phase;
        let mut new_path = true;

        // Walk segment-by-segment within the subpath.
        for k in i..j {
            let x0 = path.pts[k].x;
            let y0 = path.pts[k].y;
            let x1 = path.pts[k + 1].x;
            let y1 = path.pts[k + 1].y;
            let mut seg_len = splash_dist(x0, y0, x1, y1);
            let mut xa = x0;
            let mut ya = y0;

            while seg_len > 0.0 {
                if dash_dist >= seg_len {
                    if dash_on {
                        if new_path {
                            let _ = builder.move_to(xa, ya);
                            new_path = false;
                        }
                        let _ = builder.line_to(x1, y1);
                    }
                    dash_dist -= seg_len;
                    seg_len = 0.0;
                } else {
                    let xb = xa + (dash_dist / seg_len) * (x1 - xa);
                    let yb = ya + (dash_dist / seg_len) * (y1 - ya);
                    if dash_on {
                        if new_path {
                            let _ = builder.move_to(xa, ya);
                            new_path = false;
                        }
                        let _ = builder.line_to(xb, yb);
                    }
                    xa = xb;
                    ya = yb;
                    seg_len -= dash_dist;
                    dash_dist = 0.0;
                }

                if dash_dist <= 0.0 {
                    dash_on = !dash_on;
                    dash_idx += 1;
                    if dash_idx == line_dash.len() {
                        dash_idx = 0;
                    }
                    dash_dist = line_dash[dash_idx];
                    new_path = true;
                }
            }
        }

        i = j + 1;
    }

    // If nothing was drawn but all path points are coincident, emit a
    // degenerate subpath so end-caps can be rendered (Acrobat behaviour).
    let result = builder.build();
    if result.pts.is_empty() && !path.pts.is_empty() {
        let all_same = path.pts.windows(2).all(|w| w[0] == w[1]);
        if all_same {
            let mut b2 = PathBuilder::new();
            let _ = b2.move_to(path.pts[0].x, path.pts[0].y);
            let _ = b2.line_to(path.pts[0].x, path.pts[0].y);
            return b2.build();
        }
    }

    result
}

/// Expand a stroked path into a filled outline path.
///
/// Mirrors `Splash::makeStrokePath` (~line 6091 of `Splash.cc`).
///
/// For each segment in the (already-flattened) input path, this function builds
/// a rectangular stroke outline with the correct cap and join geometry:
///
/// - **Butt cap**: a flat perpendicular line at the endpoint.
/// - **Round cap**: a Bezier-approximated semicircle at the endpoint.
/// - **Projecting cap**: a square extending `w/2` beyond the endpoint.
/// - **Miter join**: a pointed corner, clipped at `miter_limit`.
/// - **Round join**: a Bezier-approximated circular arc.
/// - **Bevel join**: a flat triangle closing the outside corner.
///
/// Stroke-adjust hints are emitted when `params.stroke_adjust` is `true`.
#[must_use]
#[expect(
    clippy::too_many_lines,
    reason = "direct port of Splash::makeStrokePath; splitting would obscure the 1:1 correspondence"
)]
#[expect(
    clippy::similar_names,
    reason = "geometry variables dx/dy/wdx/wdy share a common prefix by convention; renaming harms readability"
)]
#[expect(
    clippy::suboptimal_flops,
    reason = "a + b*c expressions match the C++ source exactly; mul_add would obscure the 1:1 correspondence"
)]
pub fn make_stroke_path(path: &Path, w: f64, params: &StrokeParams<'_>) -> Path {
    if path.pts.is_empty() {
        return Path::new();
    }

    let mut out = PathBuilder::new();

    // State variables (mirroring the C++ locals).
    let mut subpath_start0: usize = 0;
    let mut subpath_start1: usize = 0;
    let mut seg: usize = 0;
    let mut closed = false;
    let mut left0: usize = 0;
    let mut left1: usize = 0;
    let mut right0: usize = 0;
    let mut right1: usize = 0;
    let mut join0: usize = 0;
    let mut join1: usize = 0;
    let mut left_first: usize = 0;
    let mut right_first: usize = 0;
    let mut first_pt: usize = 0;

    // i0: start of this logical segment (may skip degenerate coincident points).
    // i1: actual start point (last of the run of coincident points starting at i0).
    let mut i0: usize = 0;
    let mut i1 = advance_past_coincident(path, i0);

    while i1 < path.pts.len() {
        let first = path.flags[i0].is_first();
        if first {
            subpath_start0 = i0;
            subpath_start1 = i1;
            seg = 0;
            closed = path.flags[i0].is_closed();
        }

        // j0: the next point after i1 (the far end of the current segment).
        let j0 = i1 + 1;
        let j1 = if j0 < path.pts.len() {
            advance_past_coincident(path, j0)
        } else {
            j0
        };

        // If i1 is the last point of its subpath, handle degenerate (zero-length)
        // subpath: only round caps generate output.
        if path.flags[i1].is_last() {
            if first && params.line_cap == LineCap::Round {
                // Zero-length subpath with round caps → draw a full circle.
                let cx = path.pts[i0].x;
                let cy = path.pts[i0].y;
                let r = 0.5 * w;
                let bc2w = BEZIER_CIRCLE2 * w;
                let _ = out.move_to(cx + r, cy);
                let _ = out.curve_to(cx + r, cy + bc2w, cx + bc2w, cy + r, cx, cy + r);
                let _ = out.curve_to(cx - bc2w, cy + r, cx - r, cy + bc2w, cx - r, cy);
                let _ = out.curve_to(cx - r, cy - bc2w, cx - bc2w, cy - r, cx, cy - r);
                let _ = out.curve_to(cx + bc2w, cy - r, cx + r, cy - bc2w, cx + r, cy);
                let _ = out.close(false);
            }
            i0 = j0;
            i1 = j1;
            continue;
        }

        let last = path.flags[j1].is_last();

        // k0: the start of the segment *after* (j1, next), used for join computation.
        let k0 = if last { subpath_start1 + 1 } else { j1 + 1 };
        let _k1 = advance_past_coincident(path, k0);

        // ── Compute the unit tangent for segment (i1 → j0) ───────────────────
        let d = 1.0
            / splash_dist(
                path.pts[i1].x,
                path.pts[i1].y,
                path.pts[j0].x,
                path.pts[j0].y,
            );
        let dx = d * (path.pts[j0].x - path.pts[i1].x);
        let dy = d * (path.pts[j0].y - path.pts[i1].y);
        let wdx = 0.5 * w * dx;
        let wdy = 0.5 * w * dy;

        // ── Draw the start cap ────────────────────────────────────────────────
        // moveTo left side of segment start.
        if out
            .move_to(path.pts[i0].x - wdy, path.pts[i0].y + wdx)
            .is_err()
        {
            break;
        }
        if i0 == subpath_start0 {
            first_pt = out.pts_len() - 1;
        }

        if first && !closed {
            // Open start of subpath → draw start cap.
            match params.line_cap {
                LineCap::Butt => {
                    let _ = out.line_to(path.pts[i0].x + wdy, path.pts[i0].y - wdx);
                }
                LineCap::Round => {
                    let _ = out.curve_to(
                        path.pts[i0].x - wdy - BEZIER_CIRCLE * wdx,
                        path.pts[i0].y + wdx - BEZIER_CIRCLE * wdy,
                        path.pts[i0].x - wdx - BEZIER_CIRCLE * wdy,
                        path.pts[i0].y - wdy + BEZIER_CIRCLE * wdx,
                        path.pts[i0].x - wdx,
                        path.pts[i0].y - wdy,
                    );
                    let _ = out.curve_to(
                        path.pts[i0].x - wdx + BEZIER_CIRCLE * wdy,
                        path.pts[i0].y - wdy - BEZIER_CIRCLE * wdx,
                        path.pts[i0].x + wdy - BEZIER_CIRCLE * wdx,
                        path.pts[i0].y - wdx - BEZIER_CIRCLE * wdy,
                        path.pts[i0].x + wdy,
                        path.pts[i0].y - wdx,
                    );
                }
                LineCap::Projecting => {
                    let _ = out.line_to(path.pts[i0].x - wdx - wdy, path.pts[i0].y + wdx - wdy);
                    let _ = out.line_to(path.pts[i0].x - wdx + wdy, path.pts[i0].y - wdx - wdy);
                    let _ = out.line_to(path.pts[i0].x + wdy, path.pts[i0].y - wdx);
                }
            }
        } else {
            // Continuation: just close off the left side.
            let _ = out.line_to(path.pts[i0].x + wdy, path.pts[i0].y - wdx);
        }

        // ── Draw the left side of the stroke rectangle ────────────────────────
        let left2 = out.pts_len() - 1;
        let _ = out.line_to(path.pts[j0].x + wdy, path.pts[j0].y - wdx);

        // ── Draw the end cap ──────────────────────────────────────────────────
        if last && !closed {
            match params.line_cap {
                LineCap::Butt => {
                    let _ = out.line_to(path.pts[j0].x - wdy, path.pts[j0].y + wdx);
                }
                LineCap::Round => {
                    let _ = out.curve_to(
                        path.pts[j0].x + wdy + BEZIER_CIRCLE * wdx,
                        path.pts[j0].y - wdx + BEZIER_CIRCLE * wdy,
                        path.pts[j0].x + wdx + BEZIER_CIRCLE * wdy,
                        path.pts[j0].y + wdy - BEZIER_CIRCLE * wdx,
                        path.pts[j0].x + wdx,
                        path.pts[j0].y + wdy,
                    );
                    let _ = out.curve_to(
                        path.pts[j0].x + wdx - BEZIER_CIRCLE * wdy,
                        path.pts[j0].y + wdy + BEZIER_CIRCLE * wdx,
                        path.pts[j0].x - wdy + BEZIER_CIRCLE * wdx,
                        path.pts[j0].y + wdx + BEZIER_CIRCLE * wdy,
                        path.pts[j0].x - wdy,
                        path.pts[j0].y + wdx,
                    );
                }
                LineCap::Projecting => {
                    let _ = out.line_to(path.pts[j0].x + wdy + wdx, path.pts[j0].y - wdx + wdy);
                    let _ = out.line_to(path.pts[j0].x - wdy + wdx, path.pts[j0].y + wdx + wdy);
                    let _ = out.line_to(path.pts[j0].x - wdy, path.pts[j0].y + wdx);
                }
            }
        } else {
            let _ = out.line_to(path.pts[j0].x - wdy, path.pts[j0].y + wdx);
        }

        // ── Close the segment rectangle ───────────────────────────────────────
        let right2 = out.pts_len() - 1;
        let _ = out.close(params.stroke_adjust);

        // ── Draw the join ─────────────────────────────────────────────────────
        let join2 = out.pts_len();
        if !last || closed {
            // Compute tangent for the *next* segment (j1 → k0).
            let dn = 1.0
                / splash_dist(
                    path.pts[j1].x,
                    path.pts[j1].y,
                    path.pts[k0].x,
                    path.pts[k0].y,
                );
            let dx_next = dn * (path.pts[k0].x - path.pts[j1].x);
            let dy_next = dn * (path.pts[k0].y - path.pts[j1].y);
            let wdx_next = 0.5 * w * dx_next;
            let wdy_next = 0.5 * w * dy_next;

            let crossprod = dx * dy_next - dy * dx_next;
            let dotprod = -(dx * dx_next + dy * dy_next);
            let has_angle = crossprod != 0.0 || dx * dx_next < 0.0 || dy * dy_next < 0.0;

            let (miter, m) = if dotprod > 0.999_9 {
                // Avoid divide-by-zero: set miter > miter_limit² so the miter
                // test always fails, and m is never used in that case.
                ((params.miter_limit + 1.0) * (params.miter_limit + 1.0), 0.0)
            } else {
                let mi = (2.0 / (1.0 - dotprod)).max(1.0);
                let mv = (mi - 1.0).sqrt();
                (mi, mv)
            };

            if has_angle && params.line_join == LineJoin::Round {
                // ── Round join ────────────────────────────────────────────────
                if crossprod < 0.0 {
                    // Join angle < 180° (inside corner on the left).
                    let angle = f64::atan2(dx, -dy);
                    let angle_next = f64::atan2(dx_next, -dy_next);
                    let angle = if angle < angle_next {
                        angle + 2.0 * PI
                    } else {
                        angle
                    };
                    let d_angle = (angle - angle_next) / PI;

                    if d_angle < 0.501 {
                        // Single arc (≤ 90°).
                        let kappa = d_angle * BEZIER_CIRCLE * w;
                        let cx1 = path.pts[j0].x - wdy + kappa * dx;
                        let cy1 = path.pts[j0].y + wdx + kappa * dy;
                        let cx2 = path.pts[j0].x - wdy_next - kappa * dx_next;
                        let cy2 = path.pts[j0].y + wdx_next - kappa * dy_next;
                        let _ = out.move_to(path.pts[j0].x, path.pts[j0].y);
                        let _ = out.line_to(path.pts[j0].x - wdy_next, path.pts[j0].y + wdx_next);
                        let _ = out.curve_to(
                            cx2,
                            cy2,
                            cx1,
                            cy1,
                            path.pts[j0].x - wdy,
                            path.pts[j0].y + wdx,
                        );
                    } else {
                        // Two arcs (> 90°).
                        let d_join = splash_dist(-wdy, wdx, -wdy_next, wdx_next);
                        if d_join > 0.0 {
                            let dx_join = (-wdy_next + wdy) / d_join;
                            let dy_join = (wdx_next - wdx) / d_join;
                            let xc =
                                path.pts[j0].x + 0.5 * w * f64::cos(0.5 * (angle + angle_next));
                            let yc =
                                path.pts[j0].y + 0.5 * w * f64::sin(0.5 * (angle + angle_next));
                            let kappa = d_angle * BEZIER_CIRCLE2 * w;
                            let cx1 = path.pts[j0].x - wdy + kappa * dx;
                            let cy1 = path.pts[j0].y + wdx + kappa * dy;
                            let cx2 = xc - kappa * dx_join;
                            let cy2 = yc - kappa * dy_join;
                            let cx3 = xc + kappa * dx_join;
                            let cy3 = yc + kappa * dy_join;
                            let cx4 = path.pts[j0].x - wdy_next - kappa * dx_next;
                            let cy4 = path.pts[j0].y + wdx_next - kappa * dy_next;
                            let _ = out.move_to(path.pts[j0].x, path.pts[j0].y);
                            let _ =
                                out.line_to(path.pts[j0].x - wdy_next, path.pts[j0].y + wdx_next);
                            let _ = out.curve_to(cx4, cy4, cx3, cy3, xc, yc);
                            let _ = out.curve_to(
                                cx2,
                                cy2,
                                cx1,
                                cy1,
                                path.pts[j0].x - wdy,
                                path.pts[j0].y + wdx,
                            );
                        }
                    }
                } else {
                    // Join angle ≥ 180° (inside corner on the right).
                    let angle = f64::atan2(-dx, dy);
                    let angle_next = f64::atan2(-dx_next, dy_next);
                    let angle_next = if angle_next < angle {
                        angle_next + 2.0 * PI
                    } else {
                        angle_next
                    };
                    let d_angle = (angle_next - angle) / PI;

                    if d_angle < 0.501 {
                        // Single arc.
                        let kappa = d_angle * BEZIER_CIRCLE * w;
                        let cx1 = path.pts[j0].x + wdy + kappa * dx;
                        let cy1 = path.pts[j0].y - wdx + kappa * dy;
                        let cx2 = path.pts[j0].x + wdy_next - kappa * dx_next;
                        let cy2 = path.pts[j0].y - wdx_next - kappa * dy_next;
                        let _ = out.move_to(path.pts[j0].x, path.pts[j0].y);
                        let _ = out.line_to(path.pts[j0].x + wdy, path.pts[j0].y - wdx);
                        let _ = out.curve_to(
                            cx1,
                            cy1,
                            cx2,
                            cy2,
                            path.pts[j0].x + wdy_next,
                            path.pts[j0].y - wdx_next,
                        );
                    } else {
                        // Two arcs.
                        let d_join = splash_dist(wdy, -wdx, wdy_next, -wdx_next);
                        if d_join > 0.0 {
                            let dx_join = (wdy_next - wdy) / d_join;
                            let dy_join = (-wdx_next + wdx) / d_join;
                            let xc =
                                path.pts[j0].x + 0.5 * w * f64::cos(0.5 * (angle + angle_next));
                            let yc =
                                path.pts[j0].y + 0.5 * w * f64::sin(0.5 * (angle + angle_next));
                            let kappa = d_angle * BEZIER_CIRCLE2 * w;
                            let cx1 = path.pts[j0].x + wdy + kappa * dx;
                            let cy1 = path.pts[j0].y - wdx + kappa * dy;
                            let cx2 = xc - kappa * dx_join;
                            let cy2 = yc - kappa * dy_join;
                            let cx3 = xc + kappa * dx_join;
                            let cy3 = yc + kappa * dy_join;
                            let cx4 = path.pts[j0].x + wdy_next - kappa * dx_next;
                            let cy4 = path.pts[j0].y - wdx_next - kappa * dy_next;
                            let _ = out.move_to(path.pts[j0].x, path.pts[j0].y);
                            let _ = out.line_to(path.pts[j0].x + wdy, path.pts[j0].y - wdx);
                            let _ = out.curve_to(cx1, cy1, cx2, cy2, xc, yc);
                            let _ = out.curve_to(
                                cx3,
                                cy3,
                                cx4,
                                cy4,
                                path.pts[j0].x + wdy_next,
                                path.pts[j0].y - wdx_next,
                            );
                        }
                    }
                }
            } else if has_angle {
                // ── Miter / Bevel join ────────────────────────────────────────
                let _ = out.move_to(path.pts[j0].x, path.pts[j0].y);
                if crossprod < 0.0 {
                    // Angle < 180° — outside corner is on the left.
                    let _ = out.line_to(path.pts[j0].x - wdy_next, path.pts[j0].y + wdx_next);
                    // Miter join: add the apex point when within the miter limit.
                    if params.line_join == LineJoin::Miter && miter.sqrt() <= params.miter_limit {
                        let _ = out.line_to(
                            path.pts[j0].x - wdy + wdx * m,
                            path.pts[j0].y + wdx + wdy * m,
                        );
                    }
                    // Both miter and bevel end at the left side of the rect.
                    let _ = out.line_to(path.pts[j0].x - wdy, path.pts[j0].y + wdx);
                } else {
                    // Angle ≥ 180° — outside corner is on the right.
                    let _ = out.line_to(path.pts[j0].x + wdy, path.pts[j0].y - wdx);
                    // Miter join: add the apex point when within the miter limit.
                    if params.line_join == LineJoin::Miter && miter.sqrt() <= params.miter_limit {
                        let _ = out.line_to(
                            path.pts[j0].x + wdy + wdx * m,
                            path.pts[j0].y - wdx + wdy * m,
                        );
                    }
                    // Both miter and bevel end at the right side of the rect.
                    let _ = out.line_to(path.pts[j0].x + wdy_next, path.pts[j0].y - wdx_next);
                }
            }

            let _ = out.close(false);
        }

        // ── Stroke adjustment hints ───────────────────────────────────────────
        if params.stroke_adjust {
            if seg == 0 && !closed {
                match params.line_cap {
                    LineCap::Butt => {
                        out.add_stroke_adjust_hint(first_pt, left2 + 1, first_pt, first_pt + 1);
                        if last {
                            out.add_stroke_adjust_hint(first_pt, left2 + 1, left2 + 1, left2 + 2);
                        }
                    }
                    LineCap::Projecting => {
                        if last {
                            out.add_stroke_adjust_hint(
                                first_pt + 1,
                                left2 + 2,
                                first_pt + 1,
                                first_pt + 2,
                            );
                            out.add_stroke_adjust_hint(
                                first_pt + 1,
                                left2 + 2,
                                left2 + 2,
                                left2 + 3,
                            );
                        } else {
                            out.add_stroke_adjust_hint(
                                first_pt + 1,
                                left2 + 1,
                                first_pt + 1,
                                first_pt + 2,
                            );
                        }
                    }
                    LineCap::Round => {}
                }
            }
            if seg >= 1 {
                if seg >= 2 {
                    out.add_stroke_adjust_hint(left1, right1, left0 + 1, right0);
                    out.add_stroke_adjust_hint(left1, right1, join0, left2);
                } else {
                    out.add_stroke_adjust_hint(left1, right1, first_pt, left2);
                }
                out.add_stroke_adjust_hint(left1, right1, right2 + 1, right2 + 1);
            }
            left0 = left1;
            left1 = left2;
            right0 = right1;
            right1 = right2;
            join0 = join1;
            join1 = join2;
            if seg == 0 {
                left_first = left2;
                right_first = right2;
            }
            if last {
                if seg >= 2 {
                    out.add_stroke_adjust_hint(left1, right1, left0 + 1, right0);
                    out.add_stroke_adjust_hint(left1, right1, join0, out.pts_len() - 1);
                } else {
                    out.add_stroke_adjust_hint(left1, right1, first_pt, out.pts_len() - 1);
                }
                if closed {
                    out.add_stroke_adjust_hint(left1, right1, first_pt, left_first);
                    out.add_stroke_adjust_hint(left1, right1, right_first + 1, right_first + 1);
                    out.add_stroke_adjust_hint(left_first, right_first, left1 + 1, right1);
                    out.add_stroke_adjust_hint(left_first, right_first, join1, out.pts_len() - 1);
                }
                if !closed && seg > 0 {
                    match params.line_cap {
                        LineCap::Butt => {
                            out.add_stroke_adjust_hint(left1 - 1, left1 + 1, left1 + 1, left1 + 2);
                        }
                        LineCap::Projecting => {
                            out.add_stroke_adjust_hint(left1 - 1, left1 + 2, left1 + 2, left1 + 3);
                        }
                        LineCap::Round => {}
                    }
                }
            }
        }

        i0 = j0;
        i1 = j1;
        seg += 1;
    }

    out.build()
}

// ── Private helpers ───────────────────────────────────────────────────────────

/// Advance `start` past any run of coincident points that share the same
/// coordinates as `path.pts[start]`, up to (but not including) the first
/// LAST-flagged index.
///
/// Returns the index of the last point in the run (i.e. the "real" start of the
/// geometric segment). Mirrors the inner `for (i1 = i0; ...)` loops in
/// `Splash::makeStrokePath`.
#[expect(
    clippy::float_cmp,
    reason = "exact equality is correct here: the C++ source also uses == to detect coincident \
              points (degenerate zero-length segments), and floating-point inequality is \
              intentional — even a tiny numerical difference should not be collapsed"
)]
fn advance_past_coincident(path: &Path, start: usize) -> usize {
    let mut i = start;
    while !path.flags[i].is_last()
        && i + 1 < path.pts.len()
        && path.pts[i + 1].x == path.pts[i].x
        && path.pts[i + 1].y == path.pts[i].y
    {
        i += 1;
    }
    i
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitmap::Bitmap;
    use crate::clip::Clip;
    use crate::path::PathBuilder;
    use crate::pipe::{PipeSrc, PipeState};
    use crate::state::TransferSet;
    use crate::types::BlendMode;
    use color::Rgb8;

    fn identity_matrix() -> [f64; 6] {
        [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    }

    fn simple_pipe() -> PipeState<'static> {
        PipeState {
            blend_mode: BlendMode::Normal,
            a_input: 255,
            overprint_mask: 0xFFFF_FFFF,
            overprint_additive: false,
            transfer: TransferSet::identity_rgb(),
            soft_mask: None,
            alpha0: None,
            knockout: false,
            knockout_opacity: 255,
            non_isolated_group: false,
        }
    }

    fn make_clip(w: u32, h: u32) -> Clip {
        Clip::new(0.0, 0.0, w as f64 - 0.001, h as f64 - 0.001, false)
    }

    fn default_params<'a>() -> StrokeParams<'a> {
        StrokeParams {
            line_width: 0.0,
            line_cap: LineCap::Butt,
            line_join: LineJoin::Miter,
            miter_limit: 10.0,
            flatness: 1.0,
            stroke_adjust: false,
            line_dash: &[],
            line_dash_phase: 0.0,
            vector_antialias: false,
        }
    }

    /// `stroke_narrow` should paint pixels along a diagonal line segment.
    #[test]
    fn stroke_narrow_draws_diagonal() {
        let mut bmp: Bitmap<Rgb8> = Bitmap::new(16, 16, 4, false);
        let clip = make_clip(16, 16);
        let pipe = simple_pipe();
        let color = [255u8, 0, 0];
        let src = PipeSrc::Solid(&color);

        // Build a diagonal path from (1,1) to (8,8).
        let mut b = PathBuilder::new();
        b.move_to(1.0, 1.0).unwrap();
        b.line_to(8.0, 8.0).unwrap();
        let path = b.build();

        // Flatten (it's already flat) and draw hairline.
        let flat = flatten_path(&path, &identity_matrix(), 1.0);
        stroke_narrow::<Rgb8>(&mut bmp, &clip, &flat, &pipe, &src, &identity_matrix(), 1.0);

        // At least one pixel on the diagonal should be painted.
        let mut any_painted = false;
        for i in 1u32..9 {
            if bmp.row(i)[i as usize].r == 255 {
                any_painted = true;
                break;
            }
        }
        assert!(
            any_painted,
            "stroke_narrow should paint at least one diagonal pixel"
        );
    }

    /// `make_stroke_path` must return a non-empty outline for a non-degenerate
    /// segment with positive line width.
    #[test]
    fn make_stroke_path_non_degenerate() {
        // A simple horizontal segment.
        let mut b = PathBuilder::new();
        b.move_to(0.0, 0.0).unwrap();
        b.line_to(10.0, 0.0).unwrap();
        let path = b.build();

        let params = StrokeParams {
            line_width: 2.0,
            ..default_params()
        };
        let outline = make_stroke_path(&path, 2.0, &params);
        assert!(
            !outline.pts.is_empty(),
            "make_stroke_path should return a non-empty path for a non-degenerate segment"
        );
        // The outline should have at least 4 points (a stroke rectangle).
        assert!(
            outline.pts.len() >= 4,
            "stroke outline should have at least 4 points, got {}",
            outline.pts.len()
        );
    }

    /// `make_dashed_path` must respect the dash array: segments alternate on/off.
    #[test]
    fn make_dashed_path_respects_dash_array() {
        // A horizontal line of length 20, dash array [4, 2] → on 4, off 2, ...
        let mut b = PathBuilder::new();
        b.move_to(0.0, 0.0).unwrap();
        b.line_to(20.0, 0.0).unwrap();
        let path = b.build();

        let dash = [4.0_f64, 2.0];
        let dashed = make_dashed_path(&path, &dash, 0.0);

        // Result should be non-empty.
        assert!(
            !dashed.pts.is_empty(),
            "dashed path should not be empty for a long segment"
        );

        // All point x coordinates must be in [0, 20].
        for pt in &dashed.pts {
            assert!(
                pt.x >= -1e-9 && pt.x <= 20.0 + 1e-9,
                "dashed point x={} out of [0, 20]",
                pt.x
            );
        }

        // Should have multiple subpaths (at least 2 FIRST-flagged points).
        let first_count = dashed.flags.iter().filter(|f| f.is_first()).count();
        assert!(
            first_count >= 2,
            "dashed path should have at least 2 subpaths (on segments), got {first_count}"
        );
    }

    /// Zero dash array should return an empty path (Acrobat behaviour).
    #[test]
    fn make_dashed_path_zero_dash_is_empty() {
        let mut b = PathBuilder::new();
        b.move_to(0.0, 0.0).unwrap();
        b.line_to(10.0, 0.0).unwrap();
        let path = b.build();

        let dash = [0.0_f64];
        let dashed = make_dashed_path(&path, &dash, 0.0);
        assert!(
            dashed.pts.is_empty(),
            "zero dash array should produce empty path"
        );
    }

    /// `flatten_path` must convert a curve to only straight segments (no CURVE flags).
    #[test]
    fn flatten_path_removes_curves() {
        let mut b = PathBuilder::new();
        b.move_to(0.0, 0.0).unwrap();
        b.curve_to(1.0, 2.0, 3.0, 4.0, 4.0, 0.0).unwrap();
        let path = b.build();

        // The original path has CURVE flags.
        assert!(path.flags.iter().any(|f| f.is_curve()));

        let flat = flatten_path(&path, &identity_matrix(), 1.0);
        // The flattened path must have no CURVE flags.
        assert!(
            flat.flags.iter().all(|f| !f.is_curve()),
            "flatten_path must remove all CURVE flags"
        );
        // The flattened path should have more than 2 points (the curve was subdivided).
        assert!(
            flat.pts.len() >= 2,
            "flattened curve should have at least 2 points"
        );
    }
}
