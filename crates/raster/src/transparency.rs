//! Transparency group compositing — replaces `Splash::beginTransparencyGroup`,
//! `Splash::endTransparencyGroup`, and `Splash::paintTransparencyGroup`.
//!
//! # PDF transparency model (§11.3–11.4)
//!
//! A **transparency group** is an intermediate compositing surface.  The caller:
//!
//! 1. Calls [`begin_group`] to allocate a fresh group bitmap and push it onto the
//!    stack.  All subsequent paint operations target that group.
//! 2. Renders into the group normally (fill, stroke, image, shading, glyph calls
//!    on the group bitmap).
//! 3. Calls [`end_group`] (or [`discard_group`] on error) to pop the group and
//!    composite it back into the underlying bitmap via [`paint_group`].
//!
//! # Isolated vs. non-isolated groups
//!
//! | Flag | Effect |
//! |------|--------|
//! | `isolated = true` | Group starts with a transparent background (alpha = 0). |
//! | `isolated = false` | Group is pre-initialised with the backdrop's colours. |
//!
//! Knockout groups clear the accumulated alpha on each object; non-knockout groups
//! accumulate.
//!
//! # Soft masks
//!
//! When `soft_mask_type != SoftMaskType::None`, the group is later used as a
//! luminosity or alpha soft mask rather than being composited directly.  In that
//! case the caller retrieves the group bitmap via [`GroupBitmap::into_bitmap`] and
//! stores it in [`GraphicsState::soft_mask`].
//!
//! # C++ equivalents
//!
//! - `Splash::beginTransparencyGroup` (~line 4954)
//! - `Splash::endTransparencyGroup`   (~line 5003)
//! - `Splash::paintTransparencyGroup` (~line 5041)

use std::sync::Arc;

use crate::bitmap::Bitmap;
use crate::clip::Clip;
use crate::pipe::{self, PipeSrc, PipeState};
use crate::types::BlendMode;
use color::Pixel;
use color::convert::div255;

// ── Public types ──────────────────────────────────────────────────────────────

/// Whether the group's soft-mask channel is alpha-based or luminosity-based.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SoftMaskType {
    /// Not a soft mask — group is composited normally.
    None,
    /// Soft mask based on the group's alpha channel.
    Alpha,
    /// Soft mask based on the perceived luminance of the group's RGB pixels.
    Luminosity,
}

/// Parameters for one transparency group, collected before `begin_group`.
#[derive(Clone, Debug)]
pub struct GroupParams {
    /// Left edge of the group bounding box in device pixels (inclusive).
    pub x_min: i32,
    /// Top edge of the group bounding box in device pixels (inclusive).
    pub y_min: i32,
    /// Right edge of the group bounding box in device pixels (inclusive).
    pub x_max: i32,
    /// Bottom edge of the group bounding box in device pixels (inclusive).
    pub y_max: i32,
    /// `true` → group starts transparent; `false` → backdrop is copied in.
    pub isolated: bool,
    /// `true` → each object within the group clears accumulated alpha first.
    pub knockout: bool,
    /// Role of this group's output.
    pub soft_mask_type: SoftMaskType,
    /// Compositing blend mode used when painting the group back.
    pub blend_mode: BlendMode,
    /// Opacity applied when painting the group back (`0` = transparent, `255` = opaque).
    pub opacity: u8,
}

/// A group bitmap together with its compositing metadata.
///
/// Returned by [`begin_group`]; passed to [`paint_group`].
pub struct GroupBitmap<P: Pixel> {
    /// The rendered group content.
    pub bitmap: Bitmap<P>,
    /// Clip region at the time the group was opened (restored on pop).
    pub saved_clip: Clip,
    /// Compositing parameters recorded at `begin_group` time.
    pub params: GroupParams,
    /// Pre-multiplied alpha plane (one byte per pixel, matching `bitmap`'s pixel
    /// count).  For an isolated group, this starts at zero; for a non-isolated
    /// group it is copied from the parent's alpha plane.
    pub alpha: Vec<u8>,
    /// For non-isolated groups: a copy of the parent alpha at the time the group
    /// was opened, used as `alpha0` during the compositing pass.
    pub alpha0: Option<Arc<Vec<u8>>>,
}

impl<P: Pixel> GroupBitmap<P> {
    /// Returns the width × height pixel dimensions of the group.
    #[must_use]
    pub const fn dims(&self) -> (u32, u32) {
        (self.bitmap.width, self.bitmap.height)
    }
}

// ── Group lifecycle ───────────────────────────────────────────────────────────

/// Open a new transparency group and return a group bitmap to render into.
///
/// - `parent` is the current destination bitmap; its alpha plane is read when
///   `!params.isolated` to initialise the group's alpha.
/// - The returned [`GroupBitmap`] becomes the new render target.  All
///   rendering until `end_group` must call into it.
///
/// # Panics
///
/// Panics (in debug mode) if the bounding box is empty (`x_min` > `x_max` or
/// `y_min` > `y_max`).
#[must_use]
pub fn begin_group<P: Pixel>(
    parent: &Bitmap<P>,
    clip: &Clip,
    params: GroupParams,
) -> GroupBitmap<P> {
    debug_assert!(
        params.x_min <= params.x_max,
        "begin_group: empty x range [{}, {}]",
        params.x_min,
        params.x_max
    );
    debug_assert!(
        params.y_min <= params.y_max,
        "begin_group: empty y range [{}, {}]",
        params.y_min,
        params.y_max
    );

    // Clamp to parent dimensions so the group never over-allocates.
    #[expect(clippy::cast_sign_loss, reason = "clamped to [0, parent.width)")]
    let gx0 = (params.x_min.max(0) as u32).min(parent.width.saturating_sub(1));
    #[expect(clippy::cast_sign_loss, reason = "clamped to [0, parent.height)")]
    let gy0 = (params.y_min.max(0) as u32).min(parent.height.saturating_sub(1));
    #[expect(clippy::cast_sign_loss, reason = "clamped to (0, parent.width]")]
    let gx1 = ((params.x_max + 1).max(0) as u32).min(parent.width);
    #[expect(clippy::cast_sign_loss, reason = "clamped to (0, parent.height]")]
    let gy1 = ((params.y_max + 1).max(0) as u32).min(parent.height);

    let gw = gx1.saturating_sub(gx0).max(1);
    let gh = gy1.saturating_sub(gy0).max(1);

    // Allocate the group bitmap with an alpha plane.
    let mut bitmap = Bitmap::<P>::new(gw, gh, 4, true);

    // For non-isolated groups, copy the parent backdrop into the group bitmap
    // and snapshot the parent alpha as alpha0.
    let alpha0 = if params.isolated {
        None
    } else {
        // Copy parent rows into group bitmap.
        for gy in 0..gh {
            let py = gy0 + gy;
            if py >= parent.height {
                break;
            }
            let src = parent.row_bytes(py);
            let ncomps = P::BYTES;
            // Copy only the group x-range from the parent row.
            let src_off = gx0 as usize * ncomps;
            let copy_len =
                (gw as usize).min((parent.width as usize).saturating_sub(gx0 as usize)) * ncomps;
            let dst = bitmap.row_bytes_mut(gy);
            dst[..copy_len].copy_from_slice(&src[src_off..src_off + copy_len]);
        }
        // Snapshot parent alpha as alpha0.
        let parent_alpha_size = (parent.width * parent.height) as usize;
        let mut snap = vec![0u8; parent_alpha_size];
        if let Some(pa) = parent.alpha_plane() {
            snap.copy_from_slice(&pa[..parent_alpha_size]);
        } else {
            snap.fill(255); // Opaque parent (no alpha plane).
        }
        Some(Arc::new(snap))
    };

    // Initialise the group's own alpha plane.
    let pixel_count = (gw * gh) as usize;
    let alpha = if params.isolated {
        vec![0u8; pixel_count] // transparent
    } else {
        // Copy parent alpha for the group region.
        let mut a = vec![255u8; pixel_count];
        if let Some(pa) = parent.alpha_plane() {
            for gy in 0..gh {
                let py = gy0 + gy;
                if py >= parent.height {
                    break;
                }
                let row_start = gy as usize * gw as usize;
                let px_start = (py * parent.width + gx0) as usize;
                let copy_w =
                    (gw as usize).min((parent.width as usize).saturating_sub(gx0 as usize));
                a[row_start..row_start + copy_w].copy_from_slice(&pa[px_start..px_start + copy_w]);
            }
        }
        a
    };

    GroupBitmap {
        bitmap,
        saved_clip: clip.clone_shared(),
        params,
        alpha,
        alpha0,
    }
}

/// Composite a finished group back into the parent bitmap and return the clip.
///
/// `group.params.x_min/y_min` specify the group's top-left corner in parent
/// device space.  `pipe` controls the compositing mode (blend mode, opacity).
///
/// After this call `group` is consumed — its bitmap is dropped.
pub fn paint_group<P: Pixel>(
    parent: &mut Bitmap<P>,
    group: GroupBitmap<P>,
    pipe: &PipeState<'_>,
) -> Clip {
    let gw = group.bitmap.width;
    let gh = group.bitmap.height;
    let alpha = &group.alpha;
    let parent_width = parent.width;
    let parent_height = parent.height;

    // x/y offset of the group in parent coordinates.
    #[expect(
        clippy::cast_sign_loss,
        reason = "x_min/y_min are clamped ≥ 0 in begin_group"
    )]
    let px0 = group.params.x_min.max(0) as u32;
    #[expect(
        clippy::cast_sign_loss,
        reason = "x_min/y_min are clamped ≥ 0 in begin_group"
    )]
    let py0 = group.params.y_min.max(0) as u32;

    for gy in 0..gh {
        let py = py0 + gy;
        if py >= parent_height {
            break;
        }

        // Group row source.
        let g_row_bytes = group.bitmap.row_bytes(gy);
        let alpha_row_off = (gy * gw) as usize;
        let g_alpha_row = &alpha[alpha_row_off..alpha_row_off + gw as usize];

        // Parent row destination.
        let (p_row, mut p_alpha) = parent.row_and_alpha_mut(py);
        let ncomps = P::BYTES;

        // Composite span by span (each pixel as a 1-pixel span).
        for gx in 0..gw {
            let px = px0 + gx;
            if px >= parent_width {
                break;
            }

            let g_src_a = g_alpha_row[gx as usize];
            if g_src_a == 0 {
                continue; // fully transparent group pixel — skip
            }

            let g_off = gx as usize * ncomps;
            let p_off = px as usize * ncomps;

            // Build a single-pixel src.
            let src_slice = &g_row_bytes[g_off..g_off + ncomps];

            // Effective source alpha = group_alpha × pipe.opacity.
            let eff_a = div255(u32::from(g_src_a) * u32::from(pipe.a_input));

            // Build a single-pixel PipeState with the effective alpha.
            let pixel_pipe = PipeState {
                a_input: eff_a,
                ..*pipe
            };
            let src = PipeSrc::Solid(src_slice);

            let dst_pix = &mut p_row[p_off..p_off + ncomps];
            // Index into the alpha row directly to avoid consuming the Option<&mut>.
            let pxi = px as usize;
            let dst_alpha: Option<&mut [u8]> = p_alpha.as_mut().map(|a| &mut a[pxi..=pxi]);

            #[expect(
                clippy::cast_possible_wrap,
                reason = "px/py are bounded by parent dimensions which fit in i32"
            )]
            pipe::render_span::<P>(
                &pixel_pipe,
                &src,
                dst_pix,
                dst_alpha,
                None,
                px as i32,
                px as i32,
                py as i32,
            );
        }
    }

    group.saved_clip
}

/// Discard a group without compositing it (used on error paths).
///
/// Returns the clip that was saved when the group was opened, allowing the
/// caller to restore graphics state cleanly.
#[must_use]
pub fn discard_group<P: Pixel>(group: GroupBitmap<P>) -> Clip {
    group.saved_clip
}

// ── Soft mask extraction ──────────────────────────────────────────────────────

/// Convert a finished group bitmap into a single-channel soft mask, one byte per pixel.
///
/// - [`SoftMaskType::Alpha`]: output is the group's alpha plane directly.
/// - [`SoftMaskType::Luminosity`]: output is `get_lum(r, g, b)` per pixel
///   (BT.709 weights rounded to [0, 255]).
/// - [`SoftMaskType::None`]: returns a fully-opaque mask (all 255).
///
/// The returned `Vec<u8>` has `width × height` entries, row-major, top-down.
/// It can be stored in [`GraphicsState::soft_mask`] after wrapping in `AnyBitmap`.
#[must_use]
pub fn extract_soft_mask<P: Pixel>(group: &GroupBitmap<P>) -> Vec<u8> {
    let GroupBitmap {
        bitmap,
        alpha,
        params,
        ..
    } = group;
    let pixel_count = (bitmap.width * bitmap.height) as usize;

    match params.soft_mask_type {
        SoftMaskType::None => vec![255u8; pixel_count],
        SoftMaskType::Alpha => alpha.clone(),
        SoftMaskType::Luminosity => {
            let ncomps = P::BYTES;
            // Only RGB (3-byte) and RGBA (4-byte) groups have meaningful luminance.
            // For mono/CMYK, fall back to the alpha plane.
            if ncomps < 3 {
                return alpha.clone();
            }
            let mut mask = Vec::with_capacity(pixel_count);
            for y in 0..bitmap.height {
                let row = bitmap.row_bytes(y);
                for x in 0..bitmap.width as usize {
                    let off = x * ncomps;
                    let r = i32::from(row[off]);
                    let g = i32::from(row[off + 1]);
                    let b = i32::from(row[off + 2]);
                    // BT.709 integer luma: (77*R + 151*G + 28*B + 128) >> 8
                    #[expect(
                        clippy::cast_possible_truncation,
                        clippy::cast_sign_loss,
                        reason = "luma = weighted sum of [0,255] channels; always in [0,255]"
                    )]
                    {
                        mask.push(((77 * r + 151 * g + 28 * b + 0x80) >> 8) as u8);
                    }
                }
            }
            mask
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitmap::Bitmap;
    use crate::clip::Clip;
    use crate::pipe::PipeState;
    use crate::state::TransferSet;
    use crate::types::BlendMode;
    use color::Rgb8;

    fn make_clip(w: u32, h: u32) -> Clip {
        Clip::new(0.0, 0.0, f64::from(w) - 0.001, f64::from(h) - 0.001, false)
    }

    fn opaque_pipe() -> PipeState<'static> {
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

    fn default_params(x_min: i32, y_min: i32, x_max: i32, y_max: i32) -> GroupParams {
        GroupParams {
            x_min,
            y_min,
            x_max,
            y_max,
            isolated: true,
            knockout: false,
            soft_mask_type: SoftMaskType::None,
            blend_mode: BlendMode::Normal,
            opacity: 255,
        }
    }

    /// An isolated group starts transparent.
    #[test]
    fn isolated_group_starts_transparent() {
        let parent: Bitmap<Rgb8> = Bitmap::new(8, 8, 4, true);
        let clip = make_clip(8, 8);
        let params = default_params(0, 0, 7, 7);
        let group = begin_group::<Rgb8>(&parent, &clip, params);
        // All alpha bytes should be 0 (transparent).
        assert!(
            group.alpha.iter().all(|&a| a == 0),
            "isolated group must start fully transparent"
        );
    }

    /// A non-isolated group copies the parent alpha.
    #[test]
    fn non_isolated_group_copies_parent_alpha() {
        let mut parent: Bitmap<Rgb8> = Bitmap::new(4, 4, 4, true);
        // Set parent alpha to 128 everywhere.
        if let Some(a) = parent.alpha_plane_mut() {
            a.fill(128);
        }
        let clip = make_clip(4, 4);
        let mut params = default_params(0, 0, 3, 3);
        params.isolated = false;
        let group = begin_group::<Rgb8>(&parent, &clip, params);
        assert!(
            group.alpha.iter().all(|&a| a == 128),
            "non-isolated group must copy parent alpha"
        );
    }

    /// Painting a solid-white opaque group over a black parent yields white.
    #[test]
    fn paint_group_opaque_white_over_black() {
        let mut parent: Bitmap<Rgb8> = Bitmap::new(4, 4, 4, true);
        let clip = make_clip(4, 4);
        let pipe = opaque_pipe();

        let params = default_params(1, 1, 2, 2); // 2×2 group at (1,1)
        let mut group = begin_group::<Rgb8>(&parent, &clip, params);

        // Fill the group with white pixels and full alpha.
        for y in 0..group.bitmap.height {
            let row = group.bitmap.row_bytes_mut(y);
            for chunk in row.chunks_exact_mut(3) {
                chunk.copy_from_slice(&[255, 255, 255]);
            }
        }
        group.alpha.fill(255);

        let _clip = paint_group::<Rgb8>(&mut parent, group, &pipe);

        // Parent pixels at (1,1)..(2,2) should be white.
        assert_eq!(parent.row(1)[1].r, 255, "pixel (1,1) R should be white");
        assert_eq!(parent.row(1)[2].r, 255, "pixel (1,2) R should be white");
    }

    /// Discarding a group returns the saved clip without painting.
    #[test]
    fn discard_group_does_not_paint() {
        let parent: Bitmap<Rgb8> = Bitmap::new(4, 4, 4, true);
        let clip = make_clip(4, 4);
        let params = default_params(0, 0, 3, 3);
        let mut group = begin_group::<Rgb8>(&parent, &clip, params);

        // Fill with red.
        for y in 0..group.bitmap.height {
            let row = group.bitmap.row_bytes_mut(y);
            for chunk in row.chunks_exact_mut(3) {
                chunk.copy_from_slice(&[255, 0, 0]);
            }
        }
        group.alpha.fill(255);

        let _saved = discard_group(group);

        // Parent should be unchanged (all zero).
        assert_eq!(parent.row(0)[0].r, 0, "discard must not paint");
    }

    /// extract_soft_mask with SoftMaskType::Alpha returns the group's alpha plane.
    #[test]
    fn extract_soft_mask_alpha_returns_alpha_plane() {
        let parent: Bitmap<Rgb8> = Bitmap::new(4, 4, 4, true);
        let clip = make_clip(4, 4);
        let mut params = default_params(0, 0, 3, 3);
        params.soft_mask_type = SoftMaskType::Alpha;
        let mut group = begin_group::<Rgb8>(&parent, &clip, params);
        group.alpha.fill(200);

        let mask = extract_soft_mask::<Rgb8>(&group);
        assert!(
            mask.iter().all(|&v| v == 200),
            "alpha soft mask must match alpha plane"
        );
    }

    /// extract_soft_mask with SoftMaskType::Luminosity computes luma from RGB.
    #[test]
    fn extract_soft_mask_luminosity_computes_luma() {
        let parent: Bitmap<Rgb8> = Bitmap::new(2, 1, 4, true);
        let clip = make_clip(2, 1);
        let mut params = default_params(0, 0, 1, 0);
        params.soft_mask_type = SoftMaskType::Luminosity;
        let mut group = begin_group::<Rgb8>(&parent, &clip, params);

        // Pixel 0: white (255,255,255) → luma = 255.
        // Pixel 1: black (0,0,0)      → luma = 0.
        let row = group.bitmap.row_bytes_mut(0);
        row[0] = 255;
        row[1] = 255;
        row[2] = 255;
        row[3] = 0;
        row[4] = 0;
        row[5] = 0;
        group.alpha.fill(255);

        let mask = extract_soft_mask::<Rgb8>(&group);
        assert_eq!(mask.len(), 2);
        assert_eq!(mask[0], 255, "white → luma=255");
        assert_eq!(mask[1], 0, "black → luma=0");
    }

    /// A transparent group pixel (alpha=0) leaves the parent unchanged.
    #[test]
    fn transparent_group_pixel_is_skipped() {
        let mut parent: Bitmap<Rgb8> = Bitmap::new(4, 4, 4, true);
        // Set parent pixel (0,0) to blue.
        parent.row_bytes_mut(0)[..3].copy_from_slice(&[0, 0, 255]);
        let clip = make_clip(4, 4);
        let pipe = opaque_pipe();
        let params = default_params(0, 0, 0, 0); // 1×1 group
        let group = begin_group::<Rgb8>(&parent, &clip, params);
        // group.alpha is all 0 (isolated, not painted into).
        let _saved = paint_group::<Rgb8>(&mut parent, group, &pipe);
        // Parent (0,0) should still be blue.
        assert_eq!(
            parent.row(0)[0].b,
            255,
            "transparent group pixel must not paint"
        );
    }
}
