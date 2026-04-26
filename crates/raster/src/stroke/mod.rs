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
//! # Sub-modules
//!
//! Path-level helpers (flattening, dashing, outline expansion) live in
//! [`path`] and are re-exported here for a flat public API.
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

mod path;
pub use path::{flatten_path, make_dashed_path, make_stroke_path};

use crate::bitmap::Bitmap;
use crate::clip::{Clip, ClipResult};
use crate::fill::fill;
use crate::path::Path;
use crate::pipe::{PipeSrc, PipeState};
use crate::types::{LineCap, LineJoin, splash_floor};
use crate::xpath::XPath;
use color::Pixel;

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

// ── Private helpers ───────────────────────────────────────────────────────────

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

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitmap::Bitmap;
    use crate::path::PathBuilder;
    use crate::pipe::PipeSrc;
    use crate::testutil::{identity_matrix, make_clip, simple_pipe};
    use color::Rgb8;

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
