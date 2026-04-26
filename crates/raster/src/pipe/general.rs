//! General pipe: soft mask, non-normal blend modes, non-isolated/knockout groups.
//!
//! This is the "everything else" path, matching `Splash::pipeRun` (the general
//! case, not one of the specialised `pipeRunSimple*` or `pipeRunAA*` variants).
//!
//! # Compositing formula (PDF spec §11.3)
//!
//! Given source alpha `a_src`, destination alpha `a_dst`, and source/dest colours:
//!
//! ```text
//! a_result = a_src + a_dst - div255(a_src * a_dst)           (isolated, non-knockout)
//! c_result = ((a_result - a_src) * c_dst + a_src * blend(c_src, c_dst)) / a_result
//! ```
//!
//! For blend mode `Normal`, `blend(c_src, c_dst) = c_src` so the formula reduces
//! to the standard Porter-Duff over.

use crate::pipe::{PipeSrc, PipeState, blend};
use crate::types::BlendMode;
use color::Pixel;
use color::convert::div255;

const MAX_COMPS: usize = 8; // DeviceN8: 4 CMYK + 4 spot = 8 bytes

/// General-purpose compositing span function.
///
/// Handles soft mask, blend modes, non-isolated groups, knockout, and overprint.
/// Slower than `simple` or `aa` but covers every case.
#[expect(
    clippy::too_many_arguments,
    reason = "mirrors C++ SplashPipe API; all parameters are necessary"
)]
#[expect(
    clippy::too_many_lines,
    reason = "compositing formula has many branches that cannot be meaningfully split"
)]
pub(crate) fn render_span_general<P: Pixel>(
    pipe: &PipeState<'_>,
    src: &PipeSrc<'_>,
    dst_pixels: &mut [u8],
    dst_alpha: Option<&mut [u8]>,
    shape: Option<&[u8]>,
    x0: i32,
    x1: i32,
    y: i32,
) {
    #[expect(
        clippy::cast_sign_loss,
        reason = "x1 >= x0 is a precondition, so x1 - x0 + 1 >= 1 > 0"
    )]
    let count = (x1 - x0 + 1) as usize;
    let ncomps = P::BYTES;

    debug_assert_eq!(dst_pixels.len(), count * ncomps);
    if let Some(sh) = shape {
        debug_assert_eq!(sh.len(), count);
    }

    // Resolve source pixels into a buffer.
    let src_buf: Vec<u8> = match src {
        PipeSrc::Solid(color) => {
            debug_assert_eq!(color.len(), ncomps);
            color.iter().cycle().take(count * ncomps).copied().collect()
        }
        PipeSrc::Pattern(pat) => {
            let mut buf = vec![0u8; count * ncomps];
            pat.fill_span(y, x0, x1, &mut buf);
            buf
        }
    };

    let a_input = u32::from(pipe.a_input);
    let is_nonseparable = matches!(
        pipe.blend_mode,
        BlendMode::Hue | BlendMode::Saturation | BlendMode::Color | BlendMode::Luminosity
    );
    let is_cmyk_like = ncomps == 4 || ncomps == 8;

    // Convenience: get optional slice at index i.
    let shape_at = |i: usize| shape.map_or(0xFFu8, |s| s[i]);
    let soft_mask_at = |i: usize| pipe.soft_mask.map_or(0xFFu8, |s| s[i]);
    let alpha0_at = |i: usize| pipe.alpha0.map(|a| a[i]);

    match dst_alpha {
        Some(dst_alpha) => {
            debug_assert_eq!(dst_alpha.len(), count);
            for i in 0..count {
                let src_px = &src_buf[i * ncomps..(i + 1) * ncomps];
                let dst_px = &mut dst_pixels[i * ncomps..(i + 1) * ncomps];
                let a_dst = u32::from(dst_alpha[i]);
                let shape_v = u32::from(shape_at(i));
                let soft_v = u32::from(soft_mask_at(i));

                // Source alpha (PDF spec §11.3.6 eq 11.1).
                let a_src = if pipe.soft_mask.is_some() {
                    if shape.is_some() {
                        u32::from(div255(u32::from(div255(a_input * soft_v)) * shape_v))
                    } else {
                        u32::from(div255(a_input * soft_v))
                    }
                } else if shape.is_some() {
                    u32::from(div255(a_input * shape_v))
                } else {
                    a_input
                };

                // Non-isolated group colour correction (PDF spec §11.4.8).
                // c_src_corrected = c_src + (c_src - c_dst) * (a_dst * 255 / shape - a_dst) / 255.
                let mut c_src_corr: [u8; MAX_COMPS] = [0; MAX_COMPS];
                let c_src: &[u8] = if pipe.non_isolated_group && shape_v != 0 {
                    let t = (a_dst * 255) / shape_v - a_dst;
                    #[expect(
                        clippy::cast_possible_wrap,
                        reason = "t is a u32 alpha-domain value; wrapping is not expected in practice"
                    )]
                    let t_i = t as i32;
                    for j in 0..ncomps {
                        let v = i32::from(src_px[j])
                            + (i32::from(src_px[j]) - i32::from(dst_px[j])) * t_i / 255;
                        #[expect(
                            clippy::cast_sign_loss,
                            reason = "value is clamped to [0, 255] above"
                        )]
                        {
                            c_src_corr[j] = v.clamp(0, 255) as u8;
                        }
                    }
                    // Knockout: if shape >= knockout_opacity, clear the destination alpha.
                    if pipe.knockout && shape_v >= u32::from(pipe.knockout_opacity) {
                        dst_alpha[i] = 0;
                        // a_dst effectively becomes 0; recompute below if needed.
                    }
                    &c_src_corr[..ncomps]
                } else {
                    src_px
                };

                // Blend function.
                let mut c_blend: [u8; MAX_COMPS] = [0; MAX_COMPS];
                if pipe.blend_mode != BlendMode::Normal {
                    apply_blend_fn(
                        pipe.blend_mode,
                        c_src,
                        dst_px,
                        &mut c_blend[..ncomps],
                        is_cmyk_like,
                        is_nonseparable,
                    );
                }

                // Result alpha.
                let a_dst_eff = u32::from(dst_alpha[i]); // may have been cleared by knockout.
                let (a_result, alpha_i, alpha_im1) =
                    compute_alphas(a_src, a_dst_eff, shape_v, alpha0_at(i), pipe.knockout);

                // Result colour.
                if a_result == 0 {
                    dst_px.fill(0);
                } else {
                    for j in 0..ncomps {
                        let c_src_j = u32::from(c_src[j]);
                        let c_dst_j = u32::from(dst_px[j]);
                        let c_b_j = u32::from(c_blend[j]);

                        let c = if pipe.blend_mode == BlendMode::Normal {
                            // No blend function: standard Porter-Duff.
                            ((alpha_i - a_src) * c_dst_j + a_src * c_src_j) / alpha_i
                        } else {
                            // With blend function.
                            ((alpha_i - a_src) * c_dst_j
                                + a_src * ((255 - alpha_im1) * c_src_j + alpha_im1 * c_b_j) / 255)
                                / alpha_i
                        };
                        #[expect(
                            clippy::cast_possible_truncation,
                            reason = "c is a weighted average of values ≤ 255, so c ≤ 255"
                        )]
                        {
                            dst_px[j] = apply_transfer_channel(pipe, j, c as u8);
                        }
                    }
                }

                // Overprint: restore channels not in mask (additive or replace).
                if pipe.overprint_mask != 0xFFFF_FFFF {
                    apply_overprint(pipe, dst_px, src_px, ncomps);
                }

                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "a_result is clamped to ≤ 255 in compute_alphas"
                )]
                {
                    dst_alpha[i] = a_result as u8;
                }
            }
        }
        None => {
            // No separate alpha plane: aDest = 0xFF implicitly.
            // Simplifies: aResult = aSrc + 255 - div255(aSrc * 255) = 255, alpha_i = 255.
            for i in 0..count {
                let src_px = &src_buf[i * ncomps..(i + 1) * ncomps];
                let dst_px = &mut dst_pixels[i * ncomps..(i + 1) * ncomps];
                let shape_v = u32::from(shape_at(i));
                let soft_v = u32::from(soft_mask_at(i));

                let a_src = if pipe.soft_mask.is_some() {
                    if shape.is_some() {
                        u32::from(div255(u32::from(div255(a_input * soft_v)) * shape_v))
                    } else {
                        u32::from(div255(a_input * soft_v))
                    }
                } else if shape.is_some() {
                    u32::from(div255(a_input * shape_v))
                } else {
                    a_input
                };

                let mut c_blend: [u8; MAX_COMPS] = [0; MAX_COMPS];
                if pipe.blend_mode != BlendMode::Normal {
                    apply_blend_fn(
                        pipe.blend_mode,
                        src_px,
                        dst_px,
                        &mut c_blend[..ncomps],
                        is_cmyk_like,
                        is_nonseparable,
                    );
                }

                for j in 0..ncomps {
                    let c_src_j = u32::from(src_px[j]);
                    let c_dst_j = u32::from(dst_px[j]);
                    let c_b_j = u32::from(c_blend[j]);

                    // With implicit a_dst=255: alpha_i=255, alpha_im1=255.
                    // General formula simplifies to div255((255-a_src)*c_dst + a_src*c_b).
                    let c = if pipe.blend_mode == BlendMode::Normal {
                        u32::from(div255((255 - a_src) * c_dst_j + a_src * c_src_j))
                    } else {
                        u32::from(div255((255 - a_src) * c_dst_j + a_src * c_b_j))
                    };
                    #[expect(
                        clippy::cast_possible_truncation,
                        reason = "c is a weighted average of values ≤ 255, so c ≤ 255"
                    )]
                    {
                        dst_px[j] = apply_transfer_channel(pipe, j, c as u8);
                    }
                }

                if pipe.overprint_mask != 0xFFFF_FFFF {
                    apply_overprint(pipe, dst_px, src_px, ncomps);
                }
            }
        }
    }
}

/// Compute result alpha and the two intermediate alphas used in the colour formula.
///
/// Returns `(a_result, alpha_i, alpha_im1)`.
///
/// Matches the C++ `pipeRun` alpha logic for isolated/non-isolated, knockout/non-knockout.
#[expect(
    clippy::option_if_let_else,
    reason = "if-let form is clearer than map_or_else for this multi-branch alpha computation"
)]
fn compute_alphas(
    a_src: u32,
    a_dst: u32,
    shape: u32,
    alpha0: Option<u8>,
    knockout: bool,
) -> (u32, u32, u32) {
    if let Some(a0) = alpha0 {
        let a0 = u32::from(a0);
        if knockout {
            // Non-isolated, knockout.
            let a_result = a_src + u32::from(div255(a_dst * (255 - shape)));
            let alpha_i = a_result + a0 - u32::from(div255(a_result * a0));
            (a_result.min(255), alpha_i.min(255), a0)
        } else {
            // Non-isolated, non-knockout.
            let a_result = a_src + a_dst - u32::from(div255(a_src * a_dst));
            let alpha_i = a_result + a0 - u32::from(div255(a_result * a0));
            let alpha_im1 = a0 + a_dst - u32::from(div255(a0 * a_dst));
            (a_result.min(255), alpha_i.min(255), alpha_im1.min(255))
        }
    } else if knockout {
        // Isolated, knockout.
        let a_result = a_src + u32::from(div255(a_dst * (255 - shape)));
        (a_result.min(255), a_result.min(255), 0)
    } else {
        // Isolated, non-knockout (most common).
        let a_result = a_src + a_dst - u32::from(div255(a_src * a_dst));
        (a_result.min(255), a_result.min(255), a_dst)
    }
}

/// Apply the blend function and write into `c_blend`.
fn apply_blend_fn(
    mode: BlendMode,
    src: &[u8],
    dst: &[u8],
    c_blend: &mut [u8],
    is_cmyk_like: bool,
    is_nonseparable: bool,
) {
    debug_assert_eq!(src.len(), dst.len());
    debug_assert_eq!(src.len(), c_blend.len());
    let ncomps = src.len();

    if is_cmyk_like {
        // Subtractive complement: invert channels, blend in additive space, re-invert.
        let mut src2 = [0u8; MAX_COMPS];
        let mut dst2 = [0u8; MAX_COMPS];
        for j in 0..ncomps.min(4) {
            src2[j] = 255 - src[j];
            dst2[j] = 255 - dst[j];
        }

        if is_nonseparable {
            let s3 = [src2[0], src2[1], src2[2]];
            let d3 = [dst2[0], dst2[1], dst2[2]];
            let r3 = blend::apply_nonseparable_rgb(mode, s3, d3);
            c_blend[0] = 255 - r3[0];
            c_blend[1] = 255 - r3[1];
            c_blend[2] = 255 - r3[2];
            // K/spot channel: for Luminosity, use src K; for others use dst K (see C++).
            if ncomps >= 4 {
                c_blend[3] = 255
                    - (if mode == BlendMode::Luminosity {
                        src2[3]
                    } else {
                        dst2[3]
                    });
            }
            for j in 4..ncomps {
                c_blend[j] = 255 - dst2[j]; // spot channels pass through dst
            }
        } else {
            blend::apply_separable(
                mode,
                &src2[..ncomps.min(4)],
                &dst2[..ncomps.min(4)],
                &mut c_blend[..ncomps.min(4)],
            );
            for v in &mut c_blend[..ncomps.min(4)] {
                *v = 255 - *v;
            }
            c_blend[4..ncomps].copy_from_slice(&dst[4..ncomps]); // spot channels: pass dst unchanged
        }
    } else if is_nonseparable {
        // RGB/Gray additive space.
        let n = ncomps.min(3);
        let mut s3 = [0u8; 3];
        let mut d3 = [0u8; 3];
        s3[..n].copy_from_slice(&src[..n]);
        d3[..n].copy_from_slice(&dst[..n]);
        // Mono: replicate the single channel to all three.
        if ncomps == 1 {
            s3[1] = s3[0];
            s3[2] = s3[0];
            d3[1] = d3[0];
            d3[2] = d3[0];
        }
        let r3 = blend::apply_nonseparable_rgb(mode, s3, d3);
        c_blend[..n].copy_from_slice(&r3[..n]);
    } else {
        blend::apply_separable(mode, src, dst, c_blend);
    }
}

/// Apply the per-channel transfer LUT for one blended byte.
///
/// Channel mapping:
/// - 0 → RGB[0] (or gray / CMYK[0] for mono/subtractive modes — callers that need
///   full per-channel dispatch must call [`apply_transfer_pixel_general`] instead).
/// - 1 → RGB[1] (green / CMYK[1])
/// - 2 → RGB[2] (blue / CMYK[2])
/// - 3 → CMYK[3] (K / alpha)
/// - 4..7 → `device_n` spot channels
/// - other → identity (should not occur with supported pixel modes)
#[inline]
fn apply_transfer_channel(pipe: &PipeState<'_>, channel: usize, v: u8) -> u8 {
    let t = &pipe.transfer;
    match channel {
        0 => t.rgb[0][v as usize],
        1 => t.rgb[1][v as usize],
        2 => t.rgb[2][v as usize],
        3 => t.cmyk[3][v as usize],
        n @ 4..=7 => t.device_n[n][v as usize],
        _ => {
            debug_assert!(
                false,
                "apply_transfer_channel: unexpected channel={channel}"
            );
            v
        }
    }
}

/// Apply overprint: for channels where the bit in `overprint_mask` is 0,
/// restore the destination channel (do not paint it).
/// For additive overprint, blend additively; for replace, restore dst.
fn apply_overprint(pipe: &PipeState<'_>, dst_px: &mut [u8], src_px: &[u8], ncomps: usize) {
    if pipe.overprint_additive {
        for j in 0..ncomps {
            if pipe.overprint_mask & (1 << j) == 0 {
                // Additive: do not write this channel.
                // (The destination already has the value; we just didn't paint it.)
            } else {
                // Additive overprint: accumulate (clamped to 255, so the as u8 is safe).
                dst_px[j] = (u16::from(dst_px[j]) + u16::from(src_px[j])).min(255) as u8;
            }
        }
    } else {
        // Replace overprint: channels not in mask keep their original dst value.
        // Since we've already written dst_px with the blended result, we need to
        // restore the channels that should not be painted.  The original dst is lost
        // at this point (we don't have it separately).  This is a known limitation:
        // overprint in the general pipe requires the caller to pass the pre-blend dst.
        // For Phase 2 correctness, assert that the caller handles this edge case.
        debug_assert_eq!(
            pipe.overprint_mask, 0xFFFF_FFFF,
            "general pipe: replace overprint requires pre-blend dst preservation; not yet implemented"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipe::{PipeSrc, PipeState};
    use crate::state::TransferSet;
    use color::Rgb8;

    fn normal_pipe(a: u8) -> PipeState<'static> {
        PipeState {
            blend_mode: BlendMode::Normal,
            a_input: a,
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
    fn opaque_src_over_any_dst_gives_src() {
        let pipe = normal_pipe(255);
        let src_color = [200u8, 100, 50];
        let src = PipeSrc::Solid(&src_color);

        let mut dst = vec![10u8, 20, 30];
        let mut alpha = vec![128u8];

        render_span_general::<Rgb8>(&pipe, &src, &mut dst, Some(&mut alpha), None, 0, 0, 0);

        // a_src = 255, a_result = 255; c = (0 * c_dst + 255 * c_src) / 255 = c_src.
        assert_eq!(&dst, &[200, 100, 50]);
        assert_eq!(alpha[0], 255);
    }

    #[test]
    fn transparent_src_leaves_dst_unchanged() {
        let pipe = normal_pipe(0);
        let src = PipeSrc::Solid(&[255u8, 255, 255]);

        let mut dst = vec![10u8, 20, 30];
        let mut alpha = vec![200u8];

        render_span_general::<Rgb8>(&pipe, &src, &mut dst, Some(&mut alpha), None, 0, 0, 0);

        // a_src = 0; a_result = 0 + 200 - 0 = 200.
        // c = (200 * c_dst + 0) / 200 = c_dst.
        assert_eq!(&dst, &[10, 20, 30]);
        assert_eq!(alpha[0], 200);
    }

    #[test]
    fn blend_multiply_with_dst() {
        let mut pipe = normal_pipe(255);
        pipe.blend_mode = BlendMode::Multiply;

        // src = 128 (grey), dst = 200.
        let src = PipeSrc::Solid(&[128u8, 128, 128]);
        let mut dst = vec![200u8, 200, 200];
        let mut alpha = vec![255u8];

        render_span_general::<Rgb8>(&pipe, &src, &mut dst, Some(&mut alpha), None, 0, 0, 0);

        // With a_src=255, a_dst=255: a_result=255, alpha_i=255, alpha_im1=255.
        // c = ((255-255)*200 + 255*((255-255)*128 + 255*Multiply(128,200)/255)) / 255
        //   = Multiply(128, 200)
        //   = div255(128 * 200) ≈ 100.
        let v = dst[0];
        assert!((v as i32 - 100).abs() <= 1, "expected ~100, got {v}");
    }

    #[test]
    fn compute_alphas_isolated_non_knockout() {
        // a_src=128, a_dst=200.
        let (ar, ai, aim1) = compute_alphas(128, 200, 255, None, false);
        // a_result = 128 + 200 - div255(128 * 200) ≈ 228.
        assert!(ar >= 226 && ar <= 230, "a_result={ar}");
        assert_eq!(ai, ar, "isolated: alpha_i == a_result");
        assert_eq!(aim1, 200, "isolated non-knockout: alpha_im1 == a_dst");
    }

    #[test]
    fn soft_mask_modulates_alpha() {
        // soft_mask[0] = 128 → a_src = div255(255 * 128) ≈ 128.
        let soft = vec![128u8];
        let mut dst = vec![0u8; 3];
        let mut alpha = vec![0u8];

        // We need a pipe with a soft_mask reference.
        // Since PipeState has a 'bmp lifetime, we store soft_mask as a slice reference.
        let pipe = PipeState {
            blend_mode: BlendMode::Normal,
            a_input: 255,
            overprint_mask: 0xFFFF_FFFF,
            overprint_additive: false,
            transfer: TransferSet::identity_rgb(),
            soft_mask: Some(soft.as_slice()),
            alpha0: None,
            knockout: false,
            knockout_opacity: 255,
            non_isolated_group: false,
        };

        let src = PipeSrc::Solid(&[255u8, 255, 255]);
        render_span_general::<Rgb8>(&pipe, &src, &mut dst, Some(&mut alpha), None, 0, 0, 0);

        // a_src ≈ 128; a_dst = 0; a_result ≈ 128.
        // c = (0 * 0 + 128 * 255) / 128 = 255.
        assert_eq!(dst[0], 255);
        assert!((alpha[0] as i32 - 128).abs() <= 2, "alpha={}", alpha[0]);
    }
}
