//! Simple pipe: `a_input == 255`, `BlendMode::Normal`, no soft mask, no shape.
//!
//! Equivalent to `Splash::pipeRunSimple{Mono8,RGB8,XBGR8,BGR8,CMYK8,DeviceN8}`.
//! Mono1 is excluded: 1-bit packed bitmaps require bit-level addressing that
//! belongs in the fill/stroke caller, not a generic span function.

use std::cell::RefCell;

use crate::pipe::{self, PipeSrc, PipeState};
#[cfg(all(target_arch = "x86_64", feature = "simd-avx2"))]
use crate::simd;
use crate::types::BlendMode;
use color::Pixel;

// Per-thread scratch buffer for pattern spans — grow-never-shrink.
thread_local! {
    static PAT_BUF: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

/// Write `x1 - x0 + 1` pixels of solid source colour directly into the
/// destination row, applying the transfer function.  Destination alpha is
/// set to 255 (fully opaque).
///
/// # Preconditions (checked via `debug_assert!` in `render_span`)
///
/// - `pipe.no_transparency(false) && pipe.blend_mode == BlendMode::Normal`
/// - `dst_pixels.len() == count * P::BYTES`
/// - `P::BYTES > 0` (Mono1 handled by caller)
pub(crate) fn render_span_simple<P: Pixel>(
    pipe: &PipeState<'_>,
    src: &PipeSrc<'_>,
    dst_pixels: &mut [u8],
    dst_alpha: Option<&mut [u8]>,
    x0: i32,
    x1: i32,
    y: i32,
) {
    debug_assert_eq!(pipe.blend_mode, BlendMode::Normal);
    debug_assert_eq!(pipe.a_input, 255);

    #[expect(
        clippy::cast_sign_loss,
        reason = "x1 >= x0 is a precondition, so x1 - x0 + 1 >= 1 > 0"
    )]
    let count = (x1 - x0 + 1) as usize;
    let ncomps = P::BYTES;

    match src {
        PipeSrc::Solid(color) => {
            debug_assert_eq!(color.len(), ncomps, "solid color length must match ncomps");
            // Apply transfer to the source colour once, then fill.
            let mut applied = [0u8; 8]; // DeviceN8 is the largest: 8 bytes.
            debug_assert!(
                ncomps <= 8,
                "ncomps {ncomps} > 8; only modes up to DeviceN8 supported"
            );
            pipe::apply_transfer_pixel(pipe, color, &mut applied[..ncomps]);
            let applied = &applied[..ncomps];

            #[cfg(all(target_arch = "x86_64", feature = "simd-avx2"))]
            {
                match ncomps {
                    1 => simd::blend_solid_gray8(dst_pixels, applied[0], count),
                    3 => simd::blend_solid_rgb8(
                        dst_pixels,
                        [applied[0], applied[1], applied[2]],
                        count,
                    ),
                    _ => {
                        for chunk in dst_pixels.chunks_exact_mut(ncomps) {
                            chunk.copy_from_slice(applied);
                        }
                    }
                }
            }
            #[cfg(not(all(target_arch = "x86_64", feature = "simd-avx2")))]
            {
                for chunk in dst_pixels.chunks_exact_mut(ncomps) {
                    chunk.copy_from_slice(applied);
                }
            }
        }
        PipeSrc::Pattern(pat) => {
            PAT_BUF.with(|cell| {
                let mut buf = cell.borrow_mut();
                buf.resize(count * ncomps, 0);
                pat.fill_span(y, x0, x1, &mut buf);
                for chunk in buf.chunks_exact_mut(ncomps) {
                    pipe::apply_transfer_in_place(pipe, chunk);
                }
                dst_pixels.copy_from_slice(&buf);
            });
        }
    }

    // Overprint: mask determines which channels are written.
    if pipe.overprint_mask != 0xFFFF_FFFF {
        apply_overprint_simple(pipe, ncomps);
    }

    // Set destination alpha to fully opaque.
    if let Some(alpha) = dst_alpha {
        debug_assert_eq!(alpha.len(), count);
        for a in alpha.iter_mut() {
            *a = 255;
        }
    }
}

/// Guard: overprint in the simple path requires the general pipe.
///
/// The simple pipe overwrites `dst_pixels` before we can read the original
/// destination, so channel-selective restore is impossible here.  The caller
/// in `render_span` must route overprint operations through `render_span_general`.
/// This function exists only to produce a loud diagnostic if the invariant breaks.
fn apply_overprint_simple(pipe: &PipeState<'_>, ncomps: usize) {
    debug_assert_eq!(
        pipe.overprint_mask, 0xFFFF_FFFF,
        "simple pipe reached with overprint_mask {:#010x} and ncomps={ncomps}; \
         route overprint through render_span_general instead",
        pipe.overprint_mask,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::TransferSet;
    use color::{Rgb8, TransferLut};

    fn opaque_normal_pipe() -> PipeState<'static> {
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
    fn solid_fills_all_pixels() {
        let pipe = opaque_normal_pipe();
        let color = [100u8, 150, 200];
        let src = PipeSrc::Solid(&color);
        let count = 5usize;
        let mut dst = vec![0u8; count * 3];
        let mut alpha = vec![0u8; count];

        render_span_simple::<Rgb8>(&pipe, &src, &mut dst, Some(&mut alpha), 0, 4, 0);

        for i in 0..count {
            assert_eq!(&dst[i * 3..i * 3 + 3], &color, "pixel {i}");
            assert_eq!(alpha[i], 255, "alpha {i}");
        }
    }

    #[test]
    fn transfer_applied_to_solid() {
        // Use an inverting transfer: T[v] = 255 - v.
        let lut_data: [u8; 256] = core::array::from_fn(|i| 255 - i as u8);
        let lut = TransferLut(lut_data);

        // Build a TransferSet that uses the inverting LUT for all RGB channels.
        // We need a 'static reference, so we use a static binding.
        // For testing, allocate on heap and leak it.
        let r_ref: &'static [u8; 256] = Box::leak(Box::new(lut_data));
        let g_ref: &'static [u8; 256] = Box::leak(Box::new(lut_data));
        let b_ref: &'static [u8; 256] = Box::leak(Box::new(lut_data));

        // identity arrays for CMYK/gray/device_n
        static ID: [u8; 256] = {
            let mut a = [0u8; 256];
            let mut i = 0u8;
            loop {
                a[i as usize] = i;
                if i == 255 {
                    break;
                }
                i += 1;
            }
            a
        };
        static DN: [[u8; 256]; 8] = {
            let mut t = [[0u8; 256]; 8];
            let mut ch = 0;
            while ch < 8 {
                let mut i = 0u8;
                loop {
                    t[ch][i as usize] = i;
                    if i == 255 {
                        break;
                    }
                    i += 1;
                }
                ch += 1;
            }
            t
        };

        let transfer = TransferSet {
            rgb: [r_ref, g_ref, b_ref],
            gray: &ID,
            cmyk: [&ID, &ID, &ID, &ID],
            device_n: &DN,
        };
        let _ = lut; // suppress unused warning

        let pipe = PipeState {
            blend_mode: BlendMode::Normal,
            a_input: 255,
            overprint_mask: 0xFFFF_FFFF,
            overprint_additive: false,
            transfer,
            soft_mask: None,
            alpha0: None,
            knockout: false,
            knockout_opacity: 255,
            non_isolated_group: false,
        };

        let color = [100u8, 150, 200];
        let src = PipeSrc::Solid(&color);
        let mut dst = vec![0u8; 3];
        let mut alpha = vec![0u8; 1];
        render_span_simple::<Rgb8>(&pipe, &src, &mut dst, Some(&mut alpha), 0, 0, 0);

        assert_eq!(dst[0], 255 - 100, "R transfer inverted");
        assert_eq!(dst[1], 255 - 150, "G transfer inverted");
        assert_eq!(dst[2], 255 - 200, "B transfer inverted");
    }

    #[test]
    fn no_alpha_plane_works() {
        let pipe = opaque_normal_pipe();
        let color = [10u8, 20, 30];
        let src = PipeSrc::Solid(&color);
        let mut dst = vec![0u8; 3];
        render_span_simple::<Rgb8>(&pipe, &src, &mut dst, None, 0, 0, 0);
        assert_eq!(&dst, &[10, 20, 30]);
    }
}
