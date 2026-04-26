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
//! 3. Calls [`paint_group`] (or [`discard_group`] on error) to pop the group and
//!    composite it back into the underlying bitmap.
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
//! luminosity or alpha soft mask rather than being composited directly.  Call
//! [`extract_soft_mask`] on the finished [`GroupBitmap`] to obtain the mask bytes,
//! then store them in [`GraphicsState::soft_mask`] after wrapping in `AnyBitmap`.
//!
//! # C++ equivalents
//!
//! - `Splash::beginTransparencyGroup`  (~line 4954)
//! - `Splash::endTransparencyGroup`    (~line 5003)
//! - `Splash::paintTransparencyGroup`  (~line 5041)

use std::sync::Arc;

use crate::bitmap::Bitmap;
use crate::clip::Clip;
use crate::pipe::{self, PipeSrc, PipeState};
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
    ///
    /// Only meaningful for RGB (3-byte) groups.  For all other pixel modes
    /// [`extract_soft_mask`] falls back to the alpha plane.
    Luminosity,
}

/// Parameters for one transparency group, collected before [`begin_group`].
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
    /// Role of this group's output — controls [`extract_soft_mask`] behaviour.
    pub soft_mask_type: SoftMaskType,
}

/// A group bitmap together with its compositing metadata.
///
/// Returned by [`begin_group`]; passed to [`paint_group`] or [`discard_group`].
pub struct GroupBitmap<P: Pixel> {
    /// The rendered group content.
    pub bitmap: Bitmap<P>,
    /// Clip region at the time the group was opened (restored on pop).
    pub saved_clip: Clip,
    /// Compositing parameters recorded at `begin_group` time.
    pub params: GroupParams,
    /// Per-pixel alpha plane (one byte per pixel, matching `bitmap`'s pixel
    /// count).  For an isolated group, this starts at zero; for a non-isolated
    /// group it is copied from the parent's alpha plane.
    pub alpha: Vec<u8>,
    /// For non-isolated groups: a snapshot of the parent alpha at the time the
    /// group was opened, used as `alpha0` during the compositing pass.
    pub alpha0: Option<Arc<[u8]>>,
}

impl<P: Pixel> GroupBitmap<P> {
    /// Returns the `(width, height)` of the group bitmap in pixels.
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
/// - The returned [`GroupBitmap`] becomes the new render target until
///   [`paint_group`] or [`discard_group`] is called.
///
/// The group bounding box is clamped to `parent` dimensions; a zero-size
/// bounding box (after clamping) is silently promoted to 1×1.
///
/// # Panics
///
/// Panics (in debug mode) if the bounding box is inverted (`x_min` > `x_max`
/// or `y_min` > `y_max`).
#[must_use]
pub fn begin_group<P: Pixel>(
    parent: &Bitmap<P>,
    clip: &Clip,
    params: GroupParams,
) -> GroupBitmap<P> {
    debug_assert!(
        params.x_min <= params.x_max,
        "begin_group: inverted x range [{}, {}]",
        params.x_min,
        params.x_max
    );
    debug_assert!(
        params.y_min <= params.y_max,
        "begin_group: inverted y range [{}, {}]",
        params.y_min,
        params.y_max
    );

    // Clamp bounding box to parent dimensions so the group never over-allocates.
    // saturating_add(1): x_max == i32::MAX must not wrap.
    #[expect(clippy::cast_sign_loss, reason = "clamped to [0, parent.width)")]
    let gx0 = (params.x_min.max(0) as u32).min(parent.width.saturating_sub(1));
    #[expect(clippy::cast_sign_loss, reason = "clamped to [0, parent.height)")]
    let gy0 = (params.y_min.max(0) as u32).min(parent.height.saturating_sub(1));
    #[expect(clippy::cast_sign_loss, reason = "clamped to (0, parent.width]")]
    let gx1 = (params.x_max.saturating_add(1).max(0) as u32).min(parent.width);
    #[expect(clippy::cast_sign_loss, reason = "clamped to (0, parent.height]")]
    let gy1 = (params.y_max.saturating_add(1).max(0) as u32).min(parent.height);

    let gw = gx1.saturating_sub(gx0).max(1);
    let gh = gy1.saturating_sub(gy0).max(1);
    // Safe: gw/gh are u32 derived from parent dims (≤ u32::MAX); usize widening
    // before multiplication prevents overflow on any realistic bitmap size.
    let pixel_count = gw as usize * gh as usize;
    let ncomps = P::BYTES;

    // Allocate the group bitmap with an alpha plane.
    let mut bitmap = Bitmap::<P>::new(gw, gh, 4, true);

    // For non-isolated groups, copy the parent backdrop into the group bitmap
    // and snapshot the full parent alpha as alpha0 (used during paint_group).
    let (alpha0, alpha) = if params.isolated {
        (None, vec![0u8; pixel_count])
    } else {
        // Fuse pixel-copy and alpha-copy into one row loop.
        let mut a = vec![255u8; pixel_count];

        for gy in 0..gh {
            let py = gy0 + gy;
            if py >= parent.height {
                break;
            }

            // Number of pixels actually available from the parent in this row.
            let copy_w = (gw as usize).min((parent.width as usize).saturating_sub(gx0 as usize));
            let group_row_off = gy as usize * gw as usize;

            // Copy pixel data: parent row [gx0, gx0+copy_w) → group row [0, copy_w).
            let src = parent.row_bytes(py);
            let src_off = gx0 as usize * ncomps;
            let dst = bitmap.row_bytes_mut(gy);
            dst[..copy_w * ncomps].copy_from_slice(&src[src_off..src_off + copy_w * ncomps]);

            // Copy alpha: same x-range.
            if let Some(pa) = parent.alpha_plane() {
                let px_start = py as usize * parent.width as usize + gx0 as usize;
                a[group_row_off..group_row_off + copy_w]
                    .copy_from_slice(&pa[px_start..px_start + copy_w]);
            }
            // else: parent has no alpha plane → treat as fully opaque (255, already filled).
        }

        // Snapshot the full parent alpha for use as alpha0 in paint_group.
        let snap: Arc<[u8]> = parent.alpha_plane().map_or_else(
            || vec![255u8; parent.width as usize * parent.height as usize].into(),
            std::convert::Into::into,
        );

        (Some(snap), a)
    };

    GroupBitmap {
        bitmap,
        saved_clip: clip.clone_shared(),
        params,
        alpha,
        alpha0,
    }
}

/// Composite a finished group back into the parent bitmap and return the saved clip.
///
/// `pipe` controls blend mode, opacity, and transfer for the compositing step.
/// The group's own alpha plane is folded into the effective source alpha as
/// `eff_a = div255(group_alpha × pipe.a_input)`.
///
/// Pixels with `group_alpha == 0` are skipped entirely (no-op fast path).
///
/// After this call `group` is consumed and its bitmap is dropped.
pub fn paint_group<P: Pixel>(
    parent: &mut Bitmap<P>,
    group: GroupBitmap<P>,
    pipe: &PipeState<'_>,
) -> Clip {
    let gw = group.bitmap.width;
    let gh = group.bitmap.height;
    let ncomps = P::BYTES;
    let alpha = &group.alpha;

    // Cache parent dimensions to avoid re-borrowing inside the inner loop.
    let parent_width = parent.width;
    let parent_height = parent.height;

    // Top-left of the group in parent coordinates (always ≥ 0 — clamped in begin_group).
    #[expect(
        clippy::cast_sign_loss,
        reason = "begin_group clamped x_min/y_min to ≥ 0 before allocating the group"
    )]
    let (px0, py0) = (
        group.params.x_min.max(0) as u32,
        group.params.y_min.max(0) as u32,
    );

    for gy in 0..gh {
        let py = py0 + gy;
        if py >= parent_height {
            break;
        }

        let g_row = group.bitmap.row_bytes(gy);
        let alpha_row_off = gy as usize * gw as usize;
        let g_alpha = &alpha[alpha_row_off..alpha_row_off + gw as usize];

        // Work out the x-extent that overlaps the parent this row.
        let x_count = (gw as usize).min((parent_width as usize).saturating_sub(px0 as usize));

        if x_count == 0 {
            continue;
        }

        let (p_row, mut p_alpha) = parent.row_and_alpha_mut(py);

        // Process each pixel in the overlap region.
        // gx drives px0+gx (parent x), g_off=gx*ncomps, and g_alpha[gx] — all
        // three use the same index, so an enumerate() iterator would not be cleaner.
        #[expect(
            clippy::needless_range_loop,
            reason = "gx indexes g_alpha, g_off, and parent x simultaneously; enumerate() adds noise"
        )]
        for gx in 0..x_count {
            let px = px0 as usize + gx;

            let g_src_a = g_alpha[gx];
            if g_src_a == 0 {
                continue; // fully transparent — leave parent unchanged
            }

            let g_off = gx * ncomps;
            let p_off = px * ncomps;

            // Effective source alpha = group alpha × overall pipe opacity.
            let eff_a = div255(u32::from(g_src_a) * u32::from(pipe.a_input));

            let pixel_pipe = PipeState {
                a_input: eff_a,
                ..*pipe
            };
            let src = PipeSrc::Solid(&g_row[g_off..g_off + ncomps]);
            let dst_pix = &mut p_row[p_off..p_off + ncomps];
            let dst_alpha: Option<&mut [u8]> = p_alpha.as_mut().map(|a| &mut a[px..=px]);

            // px is usize (derived from u32 parent coords); py is u32.
            // PDF page dimensions are bounded by 14400 pt ≈ 200K px at 1440 dpi,
            // well within i32::MAX, so these casts are always safe in practice.
            #[expect(
                clippy::cast_possible_wrap,
                clippy::cast_possible_truncation,
                reason = "px/py originate from u32 parent dimensions; \
                          PDF pages are always < 2^31 px in any real scenario"
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
/// Returns the clip that was saved when the group was opened so the caller can
/// restore graphics state cleanly.
#[must_use]
pub fn discard_group<P: Pixel>(group: GroupBitmap<P>) -> Clip {
    group.saved_clip
}

// ── Soft mask extraction ──────────────────────────────────────────────────────

/// Convert a finished group bitmap into a single-channel soft mask.
///
/// Returns one byte per pixel (`width × height` bytes, row-major, top-down):
///
/// | `soft_mask_type`         | Output                                        |
/// |--------------------------|-----------------------------------------------|
/// | [`SoftMaskType::None`]   | All 255 (fully opaque — no masking).          |
/// | [`SoftMaskType::Alpha`]  | Group alpha plane, verbatim.                  |
/// | [`SoftMaskType::Luminosity`] | BT.709 luma `(77R + 151G + 28B + 128) >> 8` per RGB pixel. Fallback to alpha for non-RGB groups. |
///
/// For [`SoftMaskType::Luminosity`], only 3-byte RGB groups compute a true luma
/// value.  All other pixel modes (gray, CMYK, `DeviceN`) fall back to the alpha
/// plane because their channel bytes do not map to R, G, B.
#[must_use]
pub fn extract_soft_mask<P: Pixel>(group: &GroupBitmap<P>) -> Vec<u8> {
    let GroupBitmap {
        bitmap,
        alpha,
        params,
        ..
    } = group;
    let pixel_count = bitmap.width as usize * bitmap.height as usize;

    match params.soft_mask_type {
        SoftMaskType::None => vec![255u8; pixel_count],

        SoftMaskType::Alpha => alpha.clone(),

        SoftMaskType::Luminosity => {
            // Only true RGB (exactly 3 bytes/pixel = Rgb8) carries R, G, B in
            // channels 0, 1, 2.  CMYK and DeviceN also have ≥ 3 bytes but their
            // channels are ink densities, not light intensities — computing luma
            // from them would be wrong.  Fall back to alpha for all non-RGB modes.
            if P::BYTES != 3 {
                return alpha.clone();
            }

            let mut mask = Vec::with_capacity(pixel_count);
            for y in 0..bitmap.height {
                let row = bitmap.row_bytes(y);
                for x in 0..bitmap.width as usize {
                    let off = x * 3;
                    let r = i32::from(row[off]);
                    let g = i32::from(row[off + 1]);
                    let b = i32::from(row[off + 2]);
                    // BT.709 integer luma; coefficients sum to 256, so the result
                    // is always in [0, 255] — the cast is exact.
                    // Max value: (256*255 + 0x80) >> 8 = (65280 + 128) >> 8 = 255.
                    #[expect(
                        clippy::cast_possible_truncation,
                        clippy::cast_sign_loss,
                        reason = "77+151+28 = 256; luma = weighted sum of [0,255] values; \
                                  max = (256*255+128)>>8 = 255, min = 0"
                    )]
                    mask.push(((77 * r + 151 * g + 28 * b + 0x80) >> 8) as u8);
                }
            }
            debug_assert_eq!(
                mask.len(),
                pixel_count,
                "extract_soft_mask: loop produced {} bytes, expected {} ({w}×{h})",
                mask.len(),
                pixel_count,
                w = bitmap.width,
                h = bitmap.height,
            );
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
        }
    }

    /// An isolated group starts transparent.
    #[test]
    fn isolated_group_starts_transparent() {
        let parent: Bitmap<Rgb8> = Bitmap::new(8, 8, 4, true);
        let clip = make_clip(8, 8);
        let params = default_params(0, 0, 7, 7);
        let group = begin_group::<Rgb8>(&parent, &clip, params);
        assert!(
            group.alpha.iter().all(|&a| a == 0),
            "isolated group must start fully transparent"
        );
    }

    /// A non-isolated group copies the parent alpha.
    #[test]
    fn non_isolated_group_copies_parent_alpha() {
        let mut parent: Bitmap<Rgb8> = Bitmap::new(4, 4, 4, true);
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

        // Fill group with red.
        for y in 0..group.bitmap.height {
            let row = group.bitmap.row_bytes_mut(y);
            for chunk in row.chunks_exact_mut(3) {
                chunk.copy_from_slice(&[255, 0, 0]);
            }
        }
        group.alpha.fill(255);

        let _saved = discard_group(group);

        assert_eq!(parent.row(0)[0].r, 0, "discard must not paint");
    }

    /// `extract_soft_mask` with `SoftMaskType::Alpha` returns the group's alpha plane.
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

    /// `extract_soft_mask` with `SoftMaskType::Luminosity` computes BT.709 luma from RGB.
    #[test]
    fn extract_soft_mask_luminosity_computes_luma() {
        let parent: Bitmap<Rgb8> = Bitmap::new(2, 1, 4, true);
        let clip = make_clip(2, 1);
        let mut params = default_params(0, 0, 1, 0);
        params.soft_mask_type = SoftMaskType::Luminosity;
        let mut group = begin_group::<Rgb8>(&parent, &clip, params);

        // Pixel 0: white (255, 255, 255) → luma = 255.
        // Pixel 1: black (0, 0, 0)      → luma = 0.
        let row = group.bitmap.row_bytes_mut(0);
        row[..3].copy_from_slice(&[255, 255, 255]);
        row[3..6].copy_from_slice(&[0, 0, 0]);
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
        assert_eq!(
            parent.row(0)[0].b,
            255,
            "transparent group pixel must not paint"
        );
    }

    /// `extract_soft_mask` luminosity falls back to alpha for non-RGB modes (CMYK).
    #[test]
    fn extract_soft_mask_luminosity_fallback_for_cmyk() {
        use color::Cmyk8;
        let parent: Bitmap<Cmyk8> = Bitmap::new(2, 1, 4, true);
        let clip = Clip::new(0.0, 0.0, 1.999, 0.999, false);
        let mut params = default_params(0, 0, 1, 0);
        params.soft_mask_type = SoftMaskType::Luminosity;
        let mut group = begin_group::<Cmyk8>(&parent, &clip, params);
        group.alpha.fill(77);

        let mask = extract_soft_mask::<Cmyk8>(&group);
        // For CMYK, luminosity falls back to alpha.
        assert!(
            mask.iter().all(|&v| v == 77),
            "CMYK luminosity soft mask must fall back to alpha"
        );
    }

    /// `x_max == i32::MAX` must not overflow during begin_group.
    #[test]
    fn begin_group_x_max_i32_max_does_not_overflow() {
        let parent: Bitmap<Rgb8> = Bitmap::new(4, 4, 4, true);
        let clip = make_clip(4, 4);
        // Very large bounding box; should be clamped silently to parent bounds.
        let params = default_params(0, 0, i32::MAX, i32::MAX);
        let group = begin_group::<Rgb8>(&parent, &clip, params);
        // Group is clamped to parent size: 4×4.
        assert_eq!(group.dims(), (4, 4));
    }
}
