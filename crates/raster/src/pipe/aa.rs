//! AA pipe: shape byte present, `BlendMode::Normal`, no soft mask, isolated group.
//!
//! Equivalent to `Splash::pipeRunAA{Mono8,RGB8,XBGR8,BGR8,CMYK8,DeviceN8}`.
//!
//! For each pixel:
//! 1. `a_src = div255(a_input * shape)` — scale source alpha by AA coverage.
//! 2. If `a_src == 255`: direct write (no read-back needed).
//! 3. If `a_src == 0` and `a_dst == 0`: write zeros.
//! 4. Otherwise: `a_result = a_src + a_dst - div255(a_src * a_dst)`.
//!    `c_result = ((a_result - a_src) * c_dst + a_src * c_src) / a_result`.
//!    Then apply transfer LUT.

use std::cell::RefCell;

use crate::pipe::{self, PipeSrc, PipeState};
use crate::types::BlendMode;
use color::Pixel;
use color::convert::div255;

// Per-thread scratch buffer for pattern spans — grow-never-shrink, zero per-span alloc.
thread_local! {
    static PAT_BUF: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

/// Composite a span with per-pixel shape (AA coverage) bytes.
///
/// `shape[i]` is the coverage for pixel `x0 + i`.  Length must equal
/// `x1 - x0 + 1`.
///
/// # Preconditions (checked in `render_span`)
///
/// - `pipe.use_aa_path()` — no soft mask, `BlendMode::Normal`, no group correction.
/// - `dst_pixels.len() == count * P::BYTES`.
/// - `shape.len() == count`.
/// - `P::BYTES > 0`.
#[expect(
    clippy::too_many_arguments,
    reason = "mirrors C++ SplashPipe API; all parameters are necessary"
)]
pub(crate) fn render_span_aa<P: Pixel>(
    pipe: &PipeState<'_>,
    src: &PipeSrc<'_>,
    dst_pixels: &mut [u8],
    dst_alpha: Option<&mut [u8]>,
    shape: &[u8],
    x0: i32,
    x1: i32,
    y: i32,
) {
    debug_assert_eq!(pipe.blend_mode, BlendMode::Normal);
    debug_assert!(pipe.soft_mask.is_none());

    #[expect(
        clippy::cast_sign_loss,
        reason = "x1 >= x0 is a precondition, so x1 - x0 + 1 >= 1 > 0"
    )]
    let count = (x1 - x0 + 1) as usize;
    let ncomps = P::BYTES;

    debug_assert_eq!(shape.len(), count, "shape length must equal pixel count");
    debug_assert_eq!(dst_pixels.len(), count * ncomps);

    let a_input = u32::from(pipe.a_input);

    match src {
        PipeSrc::Solid(color) => {
            debug_assert_eq!(color.len(), ncomps);
            // Read the solid colour directly — no allocation.
            render_span_aa_solid(pipe, color, dst_pixels, dst_alpha, shape, count, ncomps, a_input);
        }
        PipeSrc::Pattern(pat) => {
            // Fill the thread-local scratch buffer with pattern pixels — one allocation
            // ever per thread, grown as needed, never shrunk.
            PAT_BUF.with(|cell| {
                let mut buf = cell.borrow_mut();
                buf.resize(count * ncomps, 0);
                pat.fill_span(y, x0, x1, &mut buf);
                render_span_aa_pixels(pipe, &buf, dst_pixels, dst_alpha, shape, count, ncomps, a_input);
            });
        }
    }
}

/// Inner loop for the solid-source AA path.
///
/// Reads source colour directly from the fixed `color` slice — no intermediate buffer.
#[inline]
#[expect(clippy::too_many_arguments, reason = "all params are necessary; extracted to avoid duplication with pattern path")]
fn render_span_aa_solid(
    pipe: &PipeState<'_>,
    color: &[u8],
    dst_pixels: &mut [u8],
    dst_alpha: Option<&mut [u8]>,
    shape: &[u8],
    count: usize,
    ncomps: usize,
    a_input: u32,
) {
    match dst_alpha {
        Some(dst_alpha) => {
            for i in 0..count {
                let shape_v = u32::from(shape[i]);
                let a_src = u32::from(div255(a_input * shape_v));
                let a_dst = u32::from(dst_alpha[i]);

                let (a_result, fully_opaque_src) = if a_src == 255 {
                    (255u32, true)
                } else if a_src == 0 && a_dst == 0 {
                    let base = i * ncomps;
                    dst_pixels[base..base + ncomps].fill(0);
                    dst_alpha[i] = 0;
                    continue;
                } else {
                    let ar = a_src + a_dst - u32::from(div255(a_src * a_dst));
                    (ar, false)
                };

                let base = i * ncomps;
                let dst_px = &mut dst_pixels[base..base + ncomps];

                if fully_opaque_src {
                    pipe::apply_transfer_pixel(pipe, color, dst_px);
                } else {
                    for j in 0..ncomps {
                        let c_src = u32::from(color[j]);
                        let c_dst = u32::from(dst_px[j]);
                        let blended = ((a_result - a_src) * c_dst + a_src * c_src) / a_result;
                        #[expect(
                            clippy::cast_possible_truncation,
                            reason = "blended = (weighted sum) / a_result ≤ 255"
                        )]
                        {
                            dst_px[j] = blended as u8;
                        }
                    }
                    pipe::apply_transfer_in_place(pipe, dst_px);
                }
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "a_result is clamped via Porter-Duff; ≤ 255"
                )]
                {
                    dst_alpha[i] = a_result as u8;
                }
            }
        }
        None => {
            for (i, &sh) in shape.iter().enumerate().take(count) {
                let shape_v = u32::from(sh);
                let a_src = u32::from(div255(a_input * shape_v));
                let base = i * ncomps;
                let dst_px = &mut dst_pixels[base..base + ncomps];
                for j in 0..ncomps {
                    let blended =
                        div255((255 - a_src) * u32::from(dst_px[j]) + a_src * u32::from(color[j]));
                    dst_px[j] = blended;
                }
                pipe::apply_transfer_in_place(pipe, dst_px);
            }
        }
    }
}

/// Inner loop for the pattern-source AA path — `src_pixels` is the pre-filled
/// pattern buffer (reused thread-local scratch, length = count * ncomps).
#[inline]
#[expect(clippy::too_many_arguments, reason = "all params are necessary; extracted to avoid duplication with solid path")]
fn render_span_aa_pixels(
    pipe: &PipeState<'_>,
    src_pixels: &[u8],
    dst_pixels: &mut [u8],
    dst_alpha: Option<&mut [u8]>,
    shape: &[u8],
    count: usize,
    ncomps: usize,
    a_input: u32,
) {
    match dst_alpha {
        Some(dst_alpha) => {
            for i in 0..count {
                let shape_v = u32::from(shape[i]);
                let a_src = u32::from(div255(a_input * shape_v));
                let a_dst = u32::from(dst_alpha[i]);

                let (a_result, fully_opaque_src) = if a_src == 255 {
                    (255u32, true)
                } else if a_src == 0 && a_dst == 0 {
                    let base = i * ncomps;
                    dst_pixels[base..base + ncomps].fill(0);
                    dst_alpha[i] = 0;
                    continue;
                } else {
                    let ar = a_src + a_dst - u32::from(div255(a_src * a_dst));
                    (ar, false)
                };

                let base = i * ncomps;
                let src_px = &src_pixels[base..base + ncomps];
                let dst_px = &mut dst_pixels[base..base + ncomps];

                if fully_opaque_src {
                    pipe::apply_transfer_pixel(pipe, src_px, dst_px);
                } else {
                    for j in 0..ncomps {
                        let c_src = u32::from(src_px[j]);
                        let c_dst = u32::from(dst_px[j]);
                        let blended = ((a_result - a_src) * c_dst + a_src * c_src) / a_result;
                        #[expect(
                            clippy::cast_possible_truncation,
                            reason = "blended = (weighted sum) / a_result ≤ 255"
                        )]
                        {
                            dst_px[j] = blended as u8;
                        }
                    }
                    pipe::apply_transfer_in_place(pipe, dst_px);
                }
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "a_result is clamped via Porter-Duff; ≤ 255"
                )]
                {
                    dst_alpha[i] = a_result as u8;
                }
            }
        }
        None => {
            for (i, &sh) in shape.iter().enumerate().take(count) {
                let shape_v = u32::from(sh);
                let a_src = u32::from(div255(a_input * shape_v));
                let base = i * ncomps;
                let src_px = &src_pixels[base..base + ncomps];
                let dst_px = &mut dst_pixels[base..base + ncomps];
                for j in 0..ncomps {
                    let blended =
                        div255((255 - a_src) * u32::from(dst_px[j]) + a_src * u32::from(src_px[j]));
                    dst_px[j] = blended;
                }
                pipe::apply_transfer_in_place(pipe, dst_px);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipe::PipeSrc;
    use crate::state::TransferSet;
    use color::Rgb8;

    fn aa_pipe() -> PipeState<'static> {
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

    #[test]
    fn full_coverage_writes_src() {
        let pipe = aa_pipe();
        let color = [200u8, 100, 50];
        let src = PipeSrc::Solid(&color);
        let shape = [255u8, 255];

        let mut dst = vec![50u8; 6]; // two pixels, initially different from src
        let mut alpha = vec![128u8; 2];

        render_span_aa::<Rgb8>(&pipe, &src, &mut dst, Some(&mut alpha), &shape, 0, 1, 0);

        assert_eq!(&dst[0..3], &[200, 100, 50]);
        assert_eq!(&dst[3..6], &[200, 100, 50]);
        assert_eq!(alpha[0], 255);
        assert_eq!(alpha[1], 255);
    }

    #[test]
    fn zero_coverage_over_transparent_zeroes_output() {
        let pipe = aa_pipe();
        let color = [255u8, 255, 255];
        let src = PipeSrc::Solid(&color);
        let shape = [0u8];

        let mut dst = vec![0u8; 3];
        let mut alpha = vec![0u8; 1]; // dest also transparent

        render_span_aa::<Rgb8>(&pipe, &src, &mut dst, Some(&mut alpha), &shape, 0, 0, 0);

        assert_eq!(dst[0], 0);
        assert_eq!(alpha[0], 0);
    }

    #[test]
    fn half_coverage_blends_correctly() {
        let pipe = aa_pipe();
        // src = white (255,255,255), dst = black (0,0,0), shape ≈ 128 ≈ 50%.
        let color = [255u8, 255, 255];
        let src = PipeSrc::Solid(&color);
        let shape = [128u8];

        let mut dst = vec![0u8; 3];
        let mut alpha = vec![255u8; 1]; // fully opaque destination

        render_span_aa::<Rgb8>(&pipe, &src, &mut dst, Some(&mut alpha), &shape, 0, 0, 0);

        // a_src = div255(255 * 128) ≈ 128.
        // a_result = 128 + 255 - div255(128 * 255) ≈ 255.
        // c = ((255 - 128) * 0 + 128 * 255) / 255 ≈ 128.
        let v = dst[0];
        assert!(v >= 125 && v <= 131, "expected ~128, got {v}");
        assert_eq!(alpha[0], 255);
    }

    #[test]
    fn no_alpha_plane_uses_opaque_dst() {
        let pipe = aa_pipe();
        let color = [200u8, 100, 50];
        let src = PipeSrc::Solid(&color);
        let shape = [128u8];

        let mut dst = vec![0u8; 3]; // black dst

        render_span_aa::<Rgb8>(&pipe, &src, &mut dst, None, &shape, 0, 0, 0);

        // With implicit aDst=255: result should be a blend.
        let v = dst[0];
        // Expected: div255((255 - 128) * 0 + 128 * 200) ≈ 100.
        assert!(v >= 95 && v <= 105, "expected ~100, got {v}");
    }
}
