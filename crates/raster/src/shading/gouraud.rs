//! Gouraud-shaded triangle fill — `Splash::gouraudTriangleShadedFill`.
//!
//! Rasterises a triangle with per-vertex colours by scanline sweep.  Each
//! scanline linearly interpolates colours from the left edge to the right edge,
//! then calls the compositing pipeline for the span.
//!
//! # Algorithm
//!
//! 1. Sort the three vertices by Y (insertion sort).
//! 2. Set up left/right edge maps: `x = slope × y + intercept`.
//! 3. Sweep scanlines Y ∈ [y0, y2]; at `y == y1` switch the active short edge.
//! 4. Interpolate colour across each scanline and call `render_span`.

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
            // Horizontal — clamp to start vertex values.
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

    fn eval(&self, y: f64) -> (i32, [u8; 3]) {
        #[expect(clippy::cast_possible_truncation, reason = "pixel coordinate rounding")]
        let x = (self.x_slope.mul_add(y, self.x_off) + 0.5) as i32;
        let c: [u8; 3] = std::array::from_fn(|ch| {
            #[expect(clippy::cast_possible_truncation, reason = "value clamped to 0..=255")]
            #[expect(clippy::cast_sign_loss, reason = "value clamped to 0.0..=255.0")]
            { self.c_slope[ch].mul_add(y, self.c_off[ch]).clamp(0.0, 255.0) as u8 }
        });
        (x, c)
    }
}

#[inline]
fn round_y(v: &GouraudVertex) -> i32 {
    #[expect(clippy::cast_possible_truncation, reason = "coordinate rounding matches C++ splashRound")]
    { (v.y + 0.5) as i32 }
}

#[inline]
fn round_x(v: &GouraudVertex) -> i32 {
    #[expect(clippy::cast_possible_truncation, reason = "coordinate rounding matches C++ splashRound")]
    { (v.x + 0.5) as i32 }
}

/// Pre-computed row of colour bytes used as a `Pattern` source.
///
/// The data slice covers exactly the span `[x0, x1]` for the current scanline.
struct RowSrc<'a> {
    data: &'a [u8],
}

impl Pattern for RowSrc<'_> {
    fn fill_span(&self, _y: i32, x0: i32, x1: i32, out: &mut [u8]) {
        debug_assert_eq!(
            out.len(), self.data.len(),
            "RowSrc::fill_span: out.len()={} data.len()={} x0={x0} x1={x1} — ncomps/P::BYTES mismatch?",
            out.len(), self.data.len(),
        );
        out.copy_from_slice(self.data);
    }
}

/// Emit one pre-coloured span via `render_span`.
fn emit_run<P: Pixel>(
    bitmap: &mut Bitmap<P>,
    pipe: &PipeState<'_>,
    span_color: &[u8],
    buf_x0: i32,
    run_x0: i32,
    run_x1: i32,
    y: i32,
) {
    debug_assert!(run_x0 >= buf_x0 && run_x0 <= run_x1 && y >= 0);
    #[expect(clippy::cast_sign_loss, reason = "run_x0 >= buf_x0 >= 0")]
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
/// Only RGB (`P::BYTES == 3`) is supported — for other colour spaces the
/// caller must perform colour-space conversion before calling this function.
/// Non-RGB pixel types return without painting.
pub fn gouraud_triangle_fill<P: Pixel>(
    bitmap: &mut Bitmap<P>,
    clip: &Clip,
    pipe: &PipeState<'_>,
    mut v: [GouraudVertex; 3],
) {
    if P::BYTES != 3 { return; }

    // --- Sort vertices by Y (insertion sort) ---
    if v[0].y > v[1].y { v.swap(0, 1); }
    if v[1].y > v[2].y {
        v.swap(1, 2);
        if v[0].y > v[1].y { v.swap(0, 1); }
    }

    let y0 = round_y(&v[0]);
    let y1 = round_y(&v[1]);
    let y2 = round_y(&v[2]);

    if y0 == y2 { return; }

    let x_min = round_x(&v[0]).min(round_x(&v[1])).min(round_x(&v[2]));
    let x_max = round_x(&v[0]).max(round_x(&v[1])).max(round_x(&v[2]));
    if clip.test_rect(x_min, y0, x_max, y2) == ClipResult::AllOutside { return; }
    let clip_bbox = clip.test_rect(x_min, y0, x_max, y2);

    let long_edge  = EdgeMap::from_vertices(&v[0], &v[2]);
    let upper_edge = EdgeMap::from_vertices(&v[0], &v[1]);
    let lower_edge = EdgeMap::from_vertices(&v[1], &v[2]);

    // Which side is the long edge on?  Sample at a y with real triangle width.
    let long_is_left = if y0 == y1 {
        let (xl, _) = long_edge.eval(f64::from(y1 + 1));
        let (xs, _) = lower_edge.eval(f64::from(y1 + 1));
        xl <= xs
    } else {
        let (xl, _) = long_edge.eval(f64::from(y1));
        let (xs, _) = upper_edge.eval(f64::from(y1));
        xl <= xs
    };

    #[expect(clippy::cast_possible_wrap, reason = "bitmap dims fit in i32")]
    let bmp_h = bitmap.height as i32;
    #[expect(clippy::cast_possible_wrap, reason = "bitmap dims fit in i32")]
    let bmp_w = bitmap.width as i32;
    let scan_y0 = y0.max(clip.y_min_i).max(0);
    let scan_y2 = y2.min(clip.y_max_i).min(bmp_h - 1);

    #[expect(clippy::cast_sign_loss, reason = "max(0) ensures non-negative")]
    let max_width = (clip.x_max_i - clip.x_min_i + 2).max(0) as usize;
    let mut span_color = vec![0u8; max_width * 3];

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

        #[expect(clippy::cast_sign_loss, reason = "sx1 >= sx0 after the continue above")]
        let span_len = (sx1 - sx0 + 1) as usize;
        if span_color.len() < span_len * 3 {
            span_color.resize(span_len * 3, 0);
        }

        let span_width = f64::from((x_right - x_left).max(1));
        for (i, px) in (sx0..=sx1).enumerate() {
            let t = (f64::from(px - x_left) / span_width).clamp(0.0, 1.0);
            #[expect(clippy::cast_sign_loss, reason = "t ∈ [0,1]")]
            #[expect(clippy::cast_possible_truncation, reason = "t * 256 ≤ 256")]
            let frac = (t * 256.0) as u32;
            span_color[i * 3]     = lerp_u8(c_left[0], c_right[0], frac);
            span_color[i * 3 + 1] = lerp_u8(c_left[1], c_right[1], frac);
            span_color[i * 3 + 2] = lerp_u8(c_left[2], c_right[2], frac);
        }

        let needs_per_pixel_clip = clip_bbox != ClipResult::AllInside
            && clip.test_span(sx0, sx1, y) != ClipResult::AllInside;

        if needs_per_pixel_clip {
            // Walk pixel-by-pixel, batch contiguous inside-clip runs.
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

        let row = bmp.row(3);
        let px = row[5];
        assert!(px.r > 50 && px.r < 150, "centroid R={} expected ~85", px.r);
        assert!(px.g > 50 && px.g < 150, "centroid G={} expected ~85", px.g);
        assert!(px.b > 50 && px.b < 150, "centroid B={} expected ~85", px.b);
    }

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

    #[test]
    fn triangle_outside_clip_is_noop() {
        let mut bmp: Bitmap<Rgb8> = Bitmap::new(8, 8, 4, false);
        let clip = Clip::new(0.0, 0.0, 3.999, 3.999, false);
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

        let centroid_painted = |bmp: &Bitmap<Rgb8>| {
            (3..7u32).any(|y| (3..8usize).any(|x| {
                let p = bmp.row(y)[x];
                p.r > 0 || p.g > 0 || p.b > 0
            }))
        };
        assert!(centroid_painted(&bmp1), "ABC order: interior not painted");
        assert!(centroid_painted(&bmp2), "CBA order: interior not painted");
    }
}
