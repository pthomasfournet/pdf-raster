//! Gouraud-shaded triangle fill — `Splash::gouraudTriangleShadedFill`.
//!
//! Rasterises a triangle with per-vertex colours by scanline sweep.  Each
//! scanline linearly interpolates colours from the left edge to the right edge,
//! then calls the compositing pipeline for the span.
//!
//! # Algorithm
//!
//! 1. Sort the three vertices by Y (insertion sort).
//! 2. Build `EdgeMap`s: `x` and per-channel colour as linear functions of Y.
//! 3. Sweep scanlines Y ∈ [y0, y2]; switch the active short edge at `y == y1`.
//! 4. Interpolate colour across each scanline and call `render_span`.
//!
//! # Colour space constraint
//!
//! Only RGB (`P::BYTES == 3`) is supported.  For CMYK/Gray/DeviceN the caller
//! must perform colour-space conversion before invoking this function.
//! Passing a non-RGB pixel type is a **logic error**; it triggers `debug_assert!`
//! in debug builds and returns without painting in release builds.

use crate::bitmap::Bitmap;
use crate::clip::{Clip, ClipResult};
use crate::pipe::{self, Pattern, PipeSrc, PipeState};
use color::Pixel;
use color::convert::lerp_u8;

/// One vertex of a Gouraud-shaded triangle.
#[derive(Copy, Clone, Debug)]
pub struct GouraudVertex {
    /// Device-space X coordinate (sub-pixel; rounded to nearest pixel).
    pub x: f64,
    /// Device-space Y coordinate.
    pub y: f64,
    /// RGB colour in device space.
    pub color: [u8; 3],
}

/// Linear interpolation of a triangle edge as a function of scanline Y.
///
/// Both x and per-channel colour are expressed as `slope * y + offset`.
struct EdgeMap {
    x_slope: f64,
    x_off:   f64,
    c_slope: [f64; 3],
    c_off:   [f64; 3],
}

impl EdgeMap {
    fn from_vertices(va: &GouraudVertex, vb: &GouraudVertex) -> Self {
        let ya = f64::from(round_y(va));
        let yb = f64::from(round_y(vb));
        let dy = yb - ya;
        if dy.abs() < 1e-9 {
            // Horizontal edge — slope undefined; hold start-vertex values.
            return Self {
                x_slope: 0.0,
                x_off:   va.x,
                c_slope: [0.0; 3],
                c_off:   [f64::from(va.color[0]), f64::from(va.color[1]), f64::from(va.color[2])],
            };
        }
        let x_slope = (vb.x - va.x) / dy;
        let x_off   = ya.mul_add(-x_slope, va.x);
        let c_slope: [f64; 3] = std::array::from_fn(|ch| {
            (f64::from(vb.color[ch]) - f64::from(va.color[ch])) / dy
        });
        let c_off: [f64; 3] = std::array::from_fn(|ch| {
            ya.mul_add(-c_slope[ch], f64::from(va.color[ch]))
        });
        Self { x_slope, x_off, c_slope, c_off }
    }

    /// Evaluate the edge at scanline `y` → (pixel x, interpolated RGB).
    fn eval(&self, y: f64) -> (i32, [u8; 3]) {
        #[expect(clippy::cast_possible_truncation, reason = "pixel coordinate rounding (+ 0.5 before truncation)")]
        let x = (self.x_slope.mul_add(y, self.x_off) + 0.5) as i32;
        let c: [u8; 3] = std::array::from_fn(|ch| {
            #[expect(clippy::cast_possible_truncation, reason = "value clamped to 0..=255")]
            #[expect(clippy::cast_sign_loss, reason = "value clamped to 0.0..=255.0")]
            { self.c_slope[ch].mul_add(y, self.c_off[ch]).clamp(0.0, 255.0) as u8 }
        });
        (x, c)
    }
}

/// Round a vertex Y coordinate to the nearest pixel (matches C++ `splashRound`).
#[inline]
fn round_y(v: &GouraudVertex) -> i32 {
    #[expect(clippy::cast_possible_truncation, reason = "splashRound: floor(y + 0.5)")]
    { (v.y + 0.5) as i32 }
}

/// Round a vertex X coordinate to the nearest pixel.
#[inline]
fn round_x(v: &GouraudVertex) -> i32 {
    #[expect(clippy::cast_possible_truncation, reason = "splashRound: floor(x + 0.5)")]
    { (v.x + 0.5) as i32 }
}

/// Pre-computed row of colour bytes used as a `Pattern` source.
///
/// `data` covers exactly the span `[run_x0, run_x1]` for the current scanline.
/// `fill_span` asserts that `out.len() == data.len()` — a mismatch indicates
/// a `P::BYTES != 3` violation in the caller.
struct RowSrc<'a> {
    data: &'a [u8],
}

impl Pattern for RowSrc<'_> {
    fn fill_span(&self, _y: i32, x0: i32, x1: i32, out: &mut [u8]) {
        debug_assert_eq!(
            out.len(), self.data.len(),
            "RowSrc::fill_span: out.len()={} data.len()={} (x0={x0} x1={x1}) \
             — gouraud_triangle_fill called with P::BYTES != 3",
            out.len(), self.data.len(),
        );
        out.copy_from_slice(self.data);
    }
}

/// Emit one pre-coloured span `[run_x0, run_x1]` from `span_color` into `bitmap`.
///
/// `buf_x0` is the leftmost x whose colour is stored at `span_color[0]`.
/// `run_x0 >= buf_x0` must hold; violated only by a logic error in the caller.
fn emit_run<P: Pixel>(
    bitmap: &mut Bitmap<P>,
    pipe: &PipeState<'_>,
    span_color: &[u8],
    buf_x0: i32,
    run_x0: i32,
    run_x1: i32,
    y: i32,
) {
    debug_assert!(run_x0 >= buf_x0, "emit_run: run_x0={run_x0} < buf_x0={buf_x0}");
    debug_assert!(run_x0 <= run_x1, "emit_run: empty run x0={run_x0} x1={run_x1}");
    debug_assert!(y >= 0,           "emit_run: negative y={y}");

    #[expect(clippy::cast_sign_loss, reason = "run_x0 >= buf_x0 >= 0 (both clamped to 0 upstream)")]
    let buf_off = (run_x0 - buf_x0) as usize * 3;
    #[expect(clippy::cast_sign_loss, reason = "run_x1 >= run_x0")]
    let buf_end = buf_off + (run_x1 - run_x0 + 1) as usize * 3;
    let src = RowSrc { data: &span_color[buf_off..buf_end] };

    #[expect(clippy::cast_sign_loss, reason = "y >= 0")]
    let y_u = y as u32;
    #[expect(clippy::cast_sign_loss, reason = "run_x0 >= 0")]
    let byte_off = run_x0 as usize * P::BYTES;
    #[expect(clippy::cast_sign_loss, reason = "run_x1 >= run_x0 >= 0")]
    let byte_end = (run_x1 as usize + 1) * P::BYTES;
    #[expect(clippy::cast_sign_loss, reason = "run_x0 >= 0")]
    let alpha_range = run_x0 as usize..=run_x1 as usize;

    let (row, alpha) = bitmap.row_and_alpha_mut(y_u);
    pipe::render_span::<P>(
        pipe,
        &PipeSrc::Pattern(&src),
        &mut row[byte_off..byte_end],
        alpha.map(|a| &mut a[alpha_range]),
        None, run_x0, run_x1, y,
    );
}

/// Fill a single Gouraud-shaded triangle into `bitmap`.
///
/// Pixels are clipped to `clip` and composited via `pipe`.
///
/// # Colour space
///
/// Only `P::BYTES == 3` (RGB) is supported.  Callers with CMYK/Gray/DeviceN
/// bitmaps **must** convert vertex colours to device RGB before calling this
/// function.  Violating this constraint triggers `debug_assert!` in debug
/// builds; release builds return silently without painting.
pub fn gouraud_triangle_fill<P: Pixel>(
    bitmap: &mut Bitmap<P>,
    clip: &Clip,
    pipe: &PipeState<'_>,
    mut v: [GouraudVertex; 3],
) {
    debug_assert_eq!(
        P::BYTES, 3,
        "gouraud_triangle_fill: P::BYTES={} but only RGB (3) is supported",
        P::BYTES,
    );
    if P::BYTES != 3 { return; }

    // --- Sort vertices by Y (insertion sort, 3 elements) ---
    if v[0].y > v[1].y { v.swap(0, 1); }
    if v[1].y > v[2].y {
        v.swap(1, 2);
        if v[0].y > v[1].y { v.swap(0, 1); }
    }
    // v[0].y ≤ v[1].y ≤ v[2].y.

    let y0 = round_y(&v[0]);
    let y1 = round_y(&v[1]);
    let y2 = round_y(&v[2]);

    if y0 == y2 { return; } // all three vertices on the same scanline — degenerate

    let x_min = round_x(&v[0]).min(round_x(&v[1])).min(round_x(&v[2]));
    let x_max = round_x(&v[0]).max(round_x(&v[1])).max(round_x(&v[2]));
    // Single test_rect call; result used for both early-exit and per-scanline clip decisions.
    let clip_bbox = clip.test_rect(x_min, y0, x_max, y2);
    if clip_bbox == ClipResult::AllOutside { return; }

    let long_edge  = EdgeMap::from_vertices(&v[0], &v[2]);
    let upper_edge = EdgeMap::from_vertices(&v[0], &v[1]);
    let lower_edge = EdgeMap::from_vertices(&v[1], &v[2]);

    // Determine which side the long edge occupies.
    // When y0 == y1 (flat top), upper_edge has zero length — sample lower_edge
    // at y1+1 instead.  Guard against y1 == i32::MAX (pathological input).
    let long_is_left = if y0 == y1 {
        let sample_y = y1.saturating_add(1);
        let (xl, _) = long_edge.eval(f64::from(sample_y));
        let (xs, _) = lower_edge.eval(f64::from(sample_y));
        xl <= xs
    } else {
        let (xl, _) = long_edge.eval(f64::from(y1));
        let (xs, _) = upper_edge.eval(f64::from(y1));
        xl <= xs
    };

    #[expect(clippy::cast_possible_wrap, reason = "bitmap dims are bounded by platform address space")]
    let bmp_h = bitmap.height as i32;
    #[expect(clippy::cast_possible_wrap, reason = "bitmap dims are bounded by platform address space")]
    let bmp_w = bitmap.width as i32;
    let scan_y0 = y0.max(clip.y_min_i).max(0);
    let scan_y2 = y2.min(clip.y_max_i).min(bmp_h - 1);

    // Span colour buffer — allocated once to the widest possible scanline
    // (clip width + 1 px for rounding slack).  Never grows inside the loop.
    #[expect(clippy::cast_sign_loss, reason = "max(0) ensures non-negative before cast")]
    let max_span = (clip.x_max_i - clip.x_min_i + 2).max(0) as usize;
    let mut span_color = vec![0u8; max_span * 3];

    for y in scan_y0..=scan_y2 {
        let yf = f64::from(y);

        let short_edge = if y <= y1 { &upper_edge } else { &lower_edge };
        let (xl, cl) = long_edge.eval(yf);
        let (xs, cs) = short_edge.eval(yf);

        let (x_left, c_left, x_right, c_right) = if long_is_left {
            (xl, cl, xs, cs)
        } else {
            (xs, cs, xl, cl)
        };

        let sx0 = x_left.max(clip.x_min_i).max(0);
        let sx1 = x_right.min(clip.x_max_i).min(bmp_w - 1);
        if sx0 > sx1 { continue; }

        #[expect(clippy::cast_sign_loss, reason = "sx1 >= sx0 (guarded above)")]
        let span_len = (sx1 - sx0 + 1) as usize;
        // span_len ≤ max_span: sx1 - sx0 ≤ clip.x_max_i - clip.x_min_i < max_span.
        debug_assert!(span_len <= max_span,
            "span_len {span_len} exceeds pre-allocated max_span {max_span}");
        let buf = &mut span_color[..span_len * 3];

        let span_width = f64::from((x_right - x_left).max(1));
        for (i, px) in (sx0..=sx1).enumerate() {
            let t = (f64::from(px - x_left) / span_width).clamp(0.0, 1.0);
            #[expect(clippy::cast_sign_loss, reason = "t ∈ [0,1]")]
            #[expect(clippy::cast_possible_truncation, reason = "t * 256 ≤ 256")]
            let frac = (t * 256.0) as u32;
            buf[i * 3]     = lerp_u8(c_left[0], c_right[0], frac);
            buf[i * 3 + 1] = lerp_u8(c_left[1], c_right[1], frac);
            buf[i * 3 + 2] = lerp_u8(c_left[2], c_right[2], frac);
        }

        // Use per-pixel clip test only when the bbox test was inconclusive AND
        // the current span is not fully inside the clip.
        let needs_per_pixel_clip = clip_bbox != ClipResult::AllInside
            && clip.test_span(sx0, sx1, y) != ClipResult::AllInside;

        if needs_per_pixel_clip {
            // Walk pixel-by-pixel; batch contiguous inside-clip pixels into runs.
            let mut run_start: Option<i32> = None;
            for px in sx0..=sx1 {
                if clip.test(px, y) {
                    if run_start.is_none() { run_start = Some(px); }
                } else if let Some(rs) = run_start.take() {
                    emit_run::<P>(bitmap, pipe, &span_color, sx0, rs, px - 1, y);
                }
            }
            if let Some(rs) = run_start {
                emit_run::<P>(bitmap, pipe, &span_color, sx0, rs, sx1, y);
            }
        } else {
            emit_run::<P>(bitmap, pipe, &span_color, sx0, sx0, sx1, y);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitmap::Bitmap;
    use crate::clip::Clip;
    use crate::pipe::PipeState;
    use crate::state::TransferSet;
    use crate::types::BlendMode;
    use color::Rgb8;

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
        Clip::new(0.0, 0.0, f64::from(w) - 0.001, f64::from(h) - 0.001, false)
    }

    /// All-white vertices: every painted pixel must be white.
    #[test]
    fn flat_white_triangle_paints_white() {
        let mut bmp: Bitmap<Rgb8> = Bitmap::new(10, 10, 4, true);
        let clip = make_clip(10, 10);
        let pipe = simple_pipe();

        let tri = [
            GouraudVertex { x: 2.0, y: 2.0, color: [255, 255, 255] },
            GouraudVertex { x: 8.0, y: 2.0, color: [255, 255, 255] },
            GouraudVertex { x: 5.0, y: 7.0, color: [255, 255, 255] },
        ];
        gouraud_triangle_fill::<Rgb8>(&mut bmp, &clip, &pipe, tri);

        let row = bmp.row(3);
        assert_eq!(row[5].r, 255, "centroid R");
        assert_eq!(row[5].g, 255, "centroid G");
        assert_eq!(row[5].b, 255, "centroid B");
    }

    /// Centroid of a red-green-blue triangle ≈ average of vertex colours (~85 each).
    #[test]
    fn centroid_approximates_average_of_vertex_colors() {
        let mut bmp: Bitmap<Rgb8> = Bitmap::new(12, 12, 4, true);
        let clip = make_clip(12, 12);
        let pipe = simple_pipe();

        let tri = [
            GouraudVertex { x: 1.0,  y: 1.0, color: [255, 0,   0  ] },
            GouraudVertex { x: 10.0, y: 1.0, color: [0,   255, 0  ] },
            GouraudVertex { x: 5.5,  y: 9.0, color: [0,   0,   255] },
        ];
        gouraud_triangle_fill::<Rgb8>(&mut bmp, &clip, &pipe, tri);

        let px = bmp.row(3)[5];
        assert!(px.r > 50 && px.r < 150, "centroid R={} expected ~85", px.r);
        assert!(px.g > 50 && px.g < 150, "centroid G={} expected ~85", px.g);
        assert!(px.b > 50 && px.b < 150, "centroid B={} expected ~85", px.b);
    }

    /// All vertices on the same scanline — degenerate, must not paint.
    #[test]
    fn degenerate_triangle_is_noop() {
        let mut bmp: Bitmap<Rgb8> = Bitmap::new(8, 8, 4, false);
        let clip = make_clip(8, 8);
        let pipe = simple_pipe();

        let tri = [
            GouraudVertex { x: 1.0, y: 4.0, color: [255, 0, 0] },
            GouraudVertex { x: 5.0, y: 4.0, color: [0, 255, 0] },
            GouraudVertex { x: 3.0, y: 4.0, color: [0, 0, 255] },
        ];
        gouraud_triangle_fill::<Rgb8>(&mut bmp, &clip, &pipe, tri);

        for y in 0..8u32 {
            for x in 0..8usize {
                assert_eq!(bmp.row(y)[x].r, 0, "y={y} x={x} should be zero");
            }
        }
    }

    /// Triangle entirely outside clip must not paint.
    #[test]
    fn triangle_outside_clip_is_noop() {
        let mut bmp: Bitmap<Rgb8> = Bitmap::new(8, 8, 4, false);
        let clip = Clip::new(0.0, 0.0, 3.999, 3.999, false); // top-left 4×4 only
        let pipe = simple_pipe();

        let tri = [
            GouraudVertex { x: 5.0, y: 5.0, color: [255, 0, 0] },
            GouraudVertex { x: 7.0, y: 5.0, color: [0, 255, 0] },
            GouraudVertex { x: 6.0, y: 7.0, color: [0, 0, 255] },
        ];
        gouraud_triangle_fill::<Rgb8>(&mut bmp, &clip, &pipe, tri);

        for y in 0..8u32 {
            for x in 0..8usize {
                assert_eq!(bmp.row(y)[x].r, 0, "y={y} x={x} should be zero");
            }
        }
    }

    /// Both vertex orderings must produce a painted interior.
    #[test]
    fn vertex_order_both_paint_interior() {
        let tri_abc = [
            GouraudVertex { x: 2.0, y: 2.0, color: [255, 0, 0] },
            GouraudVertex { x: 8.0, y: 2.0, color: [0, 255, 0] },
            GouraudVertex { x: 5.0, y: 7.0, color: [0, 0, 255] },
        ];
        let tri_cba = [tri_abc[2], tri_abc[1], tri_abc[0]];

        let mut bmp1: Bitmap<Rgb8> = Bitmap::new(12, 10, 4, true);
        let mut bmp2: Bitmap<Rgb8> = Bitmap::new(12, 10, 4, true);
        let clip = make_clip(12, 10);
        let pipe = simple_pipe();

        gouraud_triangle_fill::<Rgb8>(&mut bmp1, &clip, &pipe, tri_abc);
        gouraud_triangle_fill::<Rgb8>(&mut bmp2, &clip, &pipe, tri_cba);

        let interior_painted = |bmp: &Bitmap<Rgb8>| {
            (3..7u32).any(|y| (3..8usize).any(|x| {
                let p = bmp.row(y)[x];
                p.r > 0 || p.g > 0 || p.b > 0
            }))
        };
        assert!(interior_painted(&bmp1), "ABC order: interior not painted");
        assert!(interior_painted(&bmp2), "CBA order: interior not painted");
    }

    /// Flat-bottom triangle (y1 == y2) must paint correctly.
    #[test]
    fn flat_bottom_triangle_paints() {
        let mut bmp: Bitmap<Rgb8> = Bitmap::new(10, 10, 4, true);
        let clip = make_clip(10, 10);
        let pipe = simple_pipe();

        // Apex at top, flat base at bottom.
        let tri = [
            GouraudVertex { x: 5.0, y: 1.0, color: [255, 0, 0] },
            GouraudVertex { x: 2.0, y: 7.0, color: [0, 255, 0] },
            GouraudVertex { x: 8.0, y: 7.0, color: [0, 0, 255] },
        ];
        gouraud_triangle_fill::<Rgb8>(&mut bmp, &clip, &pipe, tri);

        // Midpoint row should be painted.
        let mid = bmp.row(4);
        let any_painted = (2..8usize).any(|x| mid[x].r > 0 || mid[x].g > 0 || mid[x].b > 0);
        assert!(any_painted, "mid-row of flat-bottom triangle should be painted");
    }
}
