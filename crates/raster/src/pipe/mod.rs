//! Compositing pipeline — replaces `SplashPipe` and its `pipeRun*` family.
//!
//! The C++ design uses a struct with a function-pointer field (`pipe->run`) selected at
//! `pipeInit` time.  Here each paint operation calls a span-level function that is
//! monomorphized over `P: Pixel` at compile time, eliminating per-pixel vtable overhead.
//!
//! # Three pipeline variants
//!
//! | Variant | When selected | C++ equivalent |
//! |---------|--------------|----------------|
//! | `simple` | `a_input == 255`, no soft mask, `BlendMode::Normal` | `pipeRunSimple*` |
//! | `aa`     | shape byte present, no soft mask, `BlendMode::Normal`, no group correction | `pipeRunAA*` |
//! | `general`| everything else — soft mask, blend mode, non-isolated/knockout groups | `pipeRun` |

pub mod aa;
pub mod blend;
pub mod general;
pub mod simple;

use crate::state::TransferSet;
use crate::types::BlendMode;
use color::Pixel;

/// Source colour for a single paint operation.
///
/// The source is resolved once per span and may be a flat colour (most common) or
/// a dynamic [`Pattern`] queried per-pixel.
pub enum PipeSrc<'a> {
    /// Flat colour: `ncomps` bytes, already in device space.
    Solid(&'a [u8]),
    /// Dynamic pattern source — `fill_span` is called per output row.
    Pattern(&'a dyn Pattern),
}

/// A source of per-pixel colour for use with the compositing pipeline.
///
/// Implementors return raw device-space byte sequences into the provided buffer.
/// The buffer length is `(x1 - x0 + 1) * ncomps` where `ncomps` is the pixel size
/// for the target bitmap mode.
pub trait Pattern: Send + Sync {
    /// Fill `out` with colour bytes for pixels `x0..=x1` on scanline `y`.
    ///
    /// `out.len()` is guaranteed to be `(x1 - x0 + 1) * ncomps`.
    fn fill_span(&self, y: i32, x0: i32, x1: i32, out: &mut [u8]);

    /// Return `true` if this pattern yields the same colour at every coordinate.
    /// When `true`, `fill_span` will be called once and the result reused across
    /// the whole span (optimisation hint only — correctness is not affected).
    fn is_static_color(&self) -> bool {
        false
    }
}

/// Immutable parameters for one paint operation, built once per fill/stroke/glyph call.
///
/// `'bmp` borrows slices out of the destination bitmap's alpha plane and the
/// current graphics state's transfer tables.
#[derive(Copy, Clone, Debug)]
pub struct PipeState<'bmp> {
    /// Compositing blend mode.
    pub blend_mode: BlendMode,

    /// Source opacity, pre-scaled: `state.fill_alpha * 255.0` rounded to `u8`.
    /// For stroke operations the caller passes `stroke_alpha * 255.0`.
    pub a_input: u8,

    /// Overprint mask: bit `k` set means channel `k` is painted.
    /// `0xFFFF_FFFF` means all channels are painted (the default).
    pub overprint_mask: u32,

    /// If `true`, overprinted channels are additively blended (ink accumulation)
    /// rather than replaced.  Corresponds to `state.overprint_additive`.
    pub overprint_additive: bool,

    /// Transfer function tables for the current pixel mode, borrowed from
    /// `GraphicsState`.  Applied to the composited result before writing.
    pub transfer: TransferSet<'bmp>,

    /// Soft mask alpha plane for the current transparency group, if any.
    /// One byte per bitmap pixel; shares the bitmap's row stride.
    /// When `Some`, `a_input` is multiplied by the soft mask value per pixel.
    pub soft_mask: Option<&'bmp [u8]>,

    /// Group alpha0 plane (non-isolated transparency group).
    /// One byte per bitmap pixel; `None` for isolated groups and normal painting.
    pub alpha0: Option<&'bmp [u8]>,

    /// `true` if this is a knockout group: later objects replace earlier objects
    /// within the group rather than compositing on top of them.
    pub knockout: bool,

    /// For knockout compositing: the accumulated group opacity threshold.
    pub knockout_opacity: u8,

    /// `true` if we are inside a non-isolated group.
    pub non_isolated_group: bool,
}

impl PipeState<'_> {
    /// Returns `true` when the simple (no-transparency) fast path is applicable.
    ///
    /// Matches C++ `pipe->noTransparency`:
    /// `a_input == 255 && no soft_mask && no shape && !in_non_isolated_group && !in_knockout_group`
    ///
    /// Overprint must be excluded: the simple path overwrites `dst_pixels` before
    /// reading the original destination, making channel-selective restore impossible.
    #[must_use]
    pub const fn no_transparency(&self, uses_shape: bool) -> bool {
        self.a_input == 255
            && self.soft_mask.is_none()
            && !uses_shape
            && !self.non_isolated_group
            && !self.knockout
            && self.alpha0.is_none()
            && self.overprint_mask == 0xFFFF_FFFF
    }

    /// Returns `true` when the AA (shape-only) fast path is applicable.
    ///
    /// Matches C++ `pipeRunAA*` selection condition:
    /// no pattern, not noTransparency, no `soft_mask`, usesShape,
    /// no alpha0, `BlendMode::Normal`, not `non_isolated_group`.
    #[must_use]
    pub fn use_aa_path(&self) -> bool {
        self.soft_mask.is_none()
            && self.alpha0.is_none()
            && self.blend_mode == BlendMode::Normal
            && !self.non_isolated_group
    }
}

/// Select and run the appropriate pipeline variant for a horizontal span.
///
/// - `pipe`: compositing parameters for this paint operation.
/// - `src`: source colour (solid or pattern).
/// - `dst_pixels`: raw pixel bytes for the destination row, starting at `x0`.
/// - `dst_alpha`: alpha plane bytes for the destination row, starting at `x0`.
///   `None` means the destination has no separate alpha (Mono1, or solid-color bitmaps
///   without transparency).
/// - `soft_mask_row`: per-pixel soft-mask byte for this row, starting at `x0`.
/// - `alpha0_row`: per-pixel alpha0 byte for this row, starting at `x0`.
/// - `shape`: per-pixel AA shape byte, or `None` for the simple path.
/// - `screen`: optional halftone screen for Mono1 dithering.
/// - `x0`, `x1`: inclusive pixel coordinate range within the row.
/// - `y`: scanline y coordinate (used by pattern and halftone screen).
/// - `ncomps`: bytes per pixel (must match `P::BYTES`).
///
/// # Panics
///
/// Panics in debug mode if `x0 > x1` or `P::BYTES == 0` (Mono1 must be handled
/// by the caller).  Also panics if `dst_pixels.len()` does not equal
/// `(x1 - x0 + 1) * P::BYTES`.
#[expect(
    clippy::too_many_arguments,
    reason = "render_span mirrors the C++ SplashPipe API; all arguments are necessary"
)]
pub fn render_span<P: Pixel>(
    pipe: &PipeState<'_>,
    src: &PipeSrc<'_>,
    dst_pixels: &mut [u8],
    dst_alpha: Option<&mut [u8]>,
    shape: Option<&[u8]>,
    x0: i32,
    x1: i32,
    y: i32,
) {
    debug_assert!(x0 <= x1, "render_span: x0={x0} > x1={x1}");
    debug_assert!(P::BYTES > 0, "render_span: Mono1 must be handled by caller");

    #[expect(
        clippy::cast_sign_loss,
        reason = "x1 >= x0 is a precondition, so x1 - x0 + 1 >= 1 > 0"
    )]
    let count = (x1 - x0 + 1) as usize;
    debug_assert_eq!(
        dst_pixels.len(),
        count * P::BYTES,
        "render_span: dst_pixels length mismatch"
    );

    let uses_shape = shape.is_some();

    if pipe.no_transparency(uses_shape) && pipe.blend_mode == BlendMode::Normal {
        simple::render_span_simple::<P>(pipe, src, dst_pixels, dst_alpha, x0, x1, y);
    } else if uses_shape && pipe.use_aa_path() {
        aa::render_span_aa::<P>(
            pipe,
            src,
            dst_pixels,
            dst_alpha,
            shape.expect("use_aa_path requires shape"),
            x0,
            x1,
            y,
        );
    } else {
        general::render_span_general::<P>(pipe, src, dst_pixels, dst_alpha, shape, x0, x1, y);
    }
}

// ── Shared transfer helpers ───────────────────────────────────────────────────
//
// Identical transfer logic was duplicated across `aa`, `simple`, and `general`.
// These `pub(crate)` functions are the single canonical implementations.

/// Apply the per-channel transfer LUTs to `src` and write the result into `dst`.
///
/// Both slices must have the same length and match the pixel mode:
/// 1 byte → gray, 3 → RGB, 4 → CMYK/XBGR, 8 → `DeviceN`.
#[expect(
    clippy::inline_always,
    reason = "called per-pixel in the innermost compositing loop; 10-15 instructions, must inline across crate boundary"
)]
#[inline(always)]
pub(crate) fn apply_transfer_pixel(pipe: &PipeState<'_>, src: &[u8], dst: &mut [u8]) {
    debug_assert_eq!(
        src.len(),
        dst.len(),
        "apply_transfer_pixel: length mismatch"
    );
    let t = &pipe.transfer;
    match src.len() {
        1 => dst[0] = t.gray[src[0] as usize],
        3 => {
            dst[0] = t.rgb[0][src[0] as usize];
            dst[1] = t.rgb[1][src[1] as usize];
            dst[2] = t.rgb[2][src[2] as usize];
        }
        4 => {
            dst[0] = t.cmyk[0][src[0] as usize];
            dst[1] = t.cmyk[1][src[1] as usize];
            dst[2] = t.cmyk[2][src[2] as usize];
            dst[3] = t.cmyk[3][src[3] as usize];
        }
        8 => {
            for (i, (&s, d)) in src.iter().zip(dst.iter_mut()).enumerate() {
                *d = t.device_n[i][s as usize];
            }
        }
        n => {
            debug_assert!(false, "apply_transfer_pixel: unexpected ncomps={n}");
            dst.copy_from_slice(src);
        }
    }
}

/// Apply transfer LUTs in-place to a single pixel slice.
#[expect(
    clippy::inline_always,
    reason = "called per-pixel in the innermost compositing loop; delegates to apply_transfer_pixel, must inline"
)]
#[inline(always)]
pub(crate) fn apply_transfer_in_place(pipe: &PipeState<'_>, px: &mut [u8]) {
    // Avoid allocating: for ncomps ≤ 8 copy into a stack buffer, apply, copy back.
    let n = px.len();
    debug_assert!(n <= 8, "apply_transfer_in_place: ncomps={n} > 8");
    let mut tmp = [0u8; 8];
    tmp[..n].copy_from_slice(px);
    apply_transfer_pixel(pipe, &tmp[..n], px);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::make_pipe;
    use crate::types::BlendMode;
    use color::Rgb8;

    #[test]
    fn no_transparency_opaque_normal() {
        let pipe = make_pipe(255, BlendMode::Normal);
        assert!(pipe.no_transparency(false));
        assert!(!pipe.no_transparency(true)); // shape disables it
    }

    #[test]
    fn no_transparency_alpha_less_than_255() {
        let pipe = make_pipe(200, BlendMode::Normal);
        assert!(!pipe.no_transparency(false));
    }

    #[test]
    fn use_aa_path_requires_no_soft_mask() {
        let pipe = make_pipe(200, BlendMode::Normal);
        assert!(pipe.use_aa_path());
    }

    #[test]
    fn use_aa_path_false_for_blend_mode() {
        let pipe = make_pipe(200, BlendMode::Multiply);
        assert!(!pipe.use_aa_path());
    }

    #[test]
    fn render_span_simple_solid_opaque() {
        // Simple path: a_input=255, Normal, no shape.
        let pipe = make_pipe(255, BlendMode::Normal);
        let src_color = [200u8, 100, 50];
        let src = PipeSrc::Solid(&src_color);

        let mut dst = vec![0u8; 3 * 4]; // 4 pixels
        let mut alpha = vec![0u8; 4];
        render_span::<Rgb8>(&pipe, &src, &mut dst, Some(&mut alpha), None, 0, 3, 0);

        // All pixels should be set to src color.
        for i in 0..4 {
            assert_eq!(dst[i * 3], 200, "pixel {i} R");
            assert_eq!(dst[i * 3 + 1], 100, "pixel {i} G");
            assert_eq!(dst[i * 3 + 2], 50, "pixel {i} B");
            assert_eq!(alpha[i], 255, "pixel {i} alpha");
        }
    }

    #[test]
    fn render_span_aa_blends_with_dst() {
        // AA path: shape=128 (~50% coverage), src=255 white, dst=0 black (fully opaque).
        let pipe = make_pipe(255, BlendMode::Normal);
        let src_color = [255u8, 255, 255];
        let src = PipeSrc::Solid(&src_color);
        let shape = vec![128u8; 4];

        let mut dst = vec![0u8; 3 * 4];
        let mut alpha = vec![255u8; 4]; // fully opaque dst so blending is visible
        render_span::<Rgb8>(
            &pipe,
            &src,
            &mut dst,
            Some(&mut alpha),
            Some(&shape),
            0,
            3,
            0,
        );

        // a_src = div255(255*128) ≈ 128; a_dst=255; blends white over black at ~50%.
        for i in 0..4 {
            let v = dst[i * 3];
            assert!(v > 100 && v < 160, "pixel {i} R={v} expected ~128");
        }
    }

    #[test]
    fn no_transparency_false_with_overprint() {
        // Overprint must not use the simple path (dst channels already overwritten).
        let mut pipe = make_pipe(255, BlendMode::Normal);
        pipe.overprint_mask = 0x0000_0001; // only channel 0 painted
        assert!(
            !pipe.no_transparency(false),
            "overprint must route to general pipe"
        );
    }
}
