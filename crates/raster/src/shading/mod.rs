//! Shaded fills — gradient patterns and Gouraud-shaded triangles.
//!
//! Replaces `Splash::shadedFill` and `Splash::gouraudTriangleShadedFill`.
//!
//! # Module layout
//! - [`axial`]    — [`AxialPattern`]: linear gradient along an axis vector
//! - [`radial`]   — [`RadialPattern`]: gradient between two circles
//! - [`function`] — [`FunctionPattern`]: PDF function sampled per pixel
//! - [`gouraud`]  — [`gouraud_triangle_fill`]: vertex-coloured triangle scan
//!
//! # Entry point
//!
//! [`shaded_fill`] is a thin wrapper around [`fill::fill`] / [`fill::eo_fill`]
//! that passes the pattern as a [`PipeSrc::Pattern`].  The gradient patterns
//! implement [`Pattern`] and are injected by the caller (typically `pdf_bridge`).

pub mod axial;
pub mod function;
pub mod gouraud;
pub mod radial;

use crate::bitmap::Bitmap;
use crate::clip::Clip;
use crate::fill;
use crate::path::Path;
use crate::pipe::{Pattern, PipeSrc, PipeState};
use color::Pixel;
use color::convert::lerp_u8;

/// Linearly interpolate an RGB triple from `a` to `b` with `frac ∈ [0, 256]`.
///
/// Shared by [`axial`] and [`radial`] to avoid duplicating the per-channel lerp.
/// `frac = 0` → `a`; `frac = 256` → `b`.
#[inline]
pub(super) fn lerp_color(a: [u8; 3], b: [u8; 3], frac: u32, out: &mut [u8]) {
    debug_assert_eq!(out.len(), 3, "lerp_color: out must be exactly 3 bytes");
    out[0] = lerp_u8(a[0], b[0], frac);
    out[1] = lerp_u8(a[1], b[1], frac);
    out[2] = lerp_u8(a[2], b[2], frac);
}

/// Fill `path` using a shading pattern as the colour source.
///
/// Equivalent to `Splash::shadedFill`.  The path defines the shading's bounding
/// shape; `pattern` supplies per-pixel colour via the [`Pattern`] trait.
/// `eo` selects even-odd vs. non-zero winding rule.
/// The caller sets `pipe.a_input` to control fill/stroke opacity.
#[expect(
    clippy::too_many_arguments,
    reason = "mirrors shadedFill signature: bitmap+clip+path+pipe+pattern+matrix+flatness+aa+eo"
)]
pub fn shaded_fill<P: Pixel>(
    bitmap: &mut Bitmap<P>,
    clip: &Clip,
    path: &Path,
    pipe: &PipeState<'_>,
    pattern: &dyn Pattern,
    matrix: &[f64; 6],
    flatness: f64,
    vector_antialias: bool,
    eo: bool,
) {
    let src = PipeSrc::Pattern(pattern);
    if eo {
        fill::eo_fill::<P>(
            bitmap,
            clip,
            path,
            pipe,
            &src,
            matrix,
            flatness,
            vector_antialias,
        );
    } else {
        fill::fill::<P>(
            bitmap,
            clip,
            path,
            pipe,
            &src,
            matrix,
            flatness,
            vector_antialias,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitmap::Bitmap;
    use crate::clip::Clip;
    use crate::path::PathBuilder;
    use crate::pipe::PipeState;
    use crate::shading::axial::AxialPattern;
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

    fn rect_path(x0: f64, y0: f64, x1: f64, y1: f64) -> Path {
        let mut b = PathBuilder::new();
        b.move_to(x0, y0).unwrap();
        b.line_to(x1, y0).unwrap();
        b.line_to(x1, y1).unwrap();
        b.line_to(x0, y1).unwrap();
        b.close(true).unwrap();
        b.build()
    }

    #[test]
    fn shaded_fill_axial_paints_interior() {
        // 8×8 bitmap; fill (1,1)→(6,6) with a left→right gradient black→white.
        let mut bmp: Bitmap<Rgb8> = Bitmap::new(8, 8, 4, false);
        let clip = Clip::new(0.0, 0.0, 7.999, 7.999, false);
        let pipe = simple_pipe();
        let path = rect_path(1.0, 1.0, 6.0, 6.0);

        let pattern = AxialPattern::new(
            [0u8, 0, 0],
            [255u8, 255, 255],
            1.0,
            3.5,
            6.0,
            3.5,
            0.0,
            1.0,
            false,
            false,
        );

        shaded_fill::<Rgb8>(
            &mut bmp,
            &clip,
            &path,
            &pipe,
            &pattern,
            &identity_matrix(),
            1.0,
            false,
            false,
        );

        let r3 = bmp.row(3);
        assert!(r3[1].r < 60, "x=1 should be near-black (got {})", r3[1].r);
        assert!(r3[5].r > 180, "x=5 should be near-white (got {})", r3[5].r);
    }

    #[test]
    fn shaded_fill_eo_and_nonzero_both_work() {
        // Same rect path, eo=true vs eo=false — single contour, both should fill.
        let mut bmp_nz: Bitmap<Rgb8> = Bitmap::new(8, 8, 4, false);
        let mut bmp_eo: Bitmap<Rgb8> = Bitmap::new(8, 8, 4, false);
        let clip = Clip::new(0.0, 0.0, 7.999, 7.999, false);
        let pipe = simple_pipe();
        let path = rect_path(1.0, 1.0, 6.0, 6.0);
        let pattern = AxialPattern::new(
            [200u8, 0, 0],
            [200u8, 0, 0],
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            1.0,
            true,
            true,
        );
        shaded_fill::<Rgb8>(
            &mut bmp_nz,
            &clip,
            &path,
            &pipe,
            &pattern,
            &identity_matrix(),
            1.0,
            false,
            false,
        );
        shaded_fill::<Rgb8>(
            &mut bmp_eo,
            &clip,
            &path,
            &pipe,
            &pattern,
            &identity_matrix(),
            1.0,
            false,
            true,
        );
        assert_eq!(
            bmp_nz.row(3)[3].r,
            bmp_eo.row(3)[3].r,
            "non-zero and eo must agree for a simple convex path"
        );
    }

    #[test]
    fn shaded_fill_empty_path_is_noop() {
        let mut bmp: Bitmap<Rgb8> = Bitmap::new(8, 8, 4, false);
        let clip = Clip::new(0.0, 0.0, 7.999, 7.999, false);
        let pipe = simple_pipe();
        let path = PathBuilder::new().build();
        let pattern = AxialPattern::new(
            [255u8, 0, 0],
            [0u8, 255, 0],
            0.0,
            0.0,
            8.0,
            0.0,
            0.0,
            1.0,
            false,
            false,
        );
        shaded_fill::<Rgb8>(
            &mut bmp,
            &clip,
            &path,
            &pipe,
            &pattern,
            &identity_matrix(),
            1.0,
            false,
            false,
        );
        assert_eq!(bmp.row(4)[4].r, 0, "empty path must not paint");
    }
}
