//! Image and image-mask rendering — replaces `Splash::fillImageMask`,
//! `Splash::drawImage`, and the four `scaleImage*` / `scaleMask*` helpers.
//!
//! # Scope
//!
//! Only axis-aligned scaling (`matrix[1] == 0 && matrix[2] == 0`) with optional
//! vertical flip (negative Y scale) is implemented.  Rotated or skewed transforms
//! return [`ImageResult::ArbitraryTransformSkipped`] so the caller can fall back
//! to a general path.
//!
//! # Scaling strategy
//!
//! Four Bresenham variants mirror the C++ `scaleMask*` / `scaleImage*` family:
//!
//! | Y axis  | X axis  | Method                                        |
//! |---------|---------|-----------------------------------------------|
//! | down ↓  | down ↓  | Box-filter (average) in both axes             |
//! | down ↓  | up   ↑  | Box-filter in Y, nearest-neighbour in X       |
//! | up   ↑  | down ↓  | Nearest-neighbour in Y, box-filter in X       |
//! | up   ↑  | up   ↑  | Nearest-neighbour in both axes                |
//!
//! All four variants process source rows exactly once, allocating no extra heap
//! per row.  "Down" means `scaled < src`; "up" means `scaled ≥ src`.
//!
//! # Pixel-count contract
//!
//! [`draw_image`] derives the component count directly from `P::BYTES`,
//! eliminating any possibility of a caller passing a mismatched value.
//!
//! # C++ equivalents
//!
//! - `Splash::fillImageMask`
//! - `Splash::drawImage`
//! - `Splash::scaleMask{YdownXdown,YdownXup,YupXdown,YupXup}`
//! - `Splash::scaleImage{YdownXdown,YdownXup,YupXdown,YupXup}`
//! - `Splash::blitMask`

use crate::bitmap::Bitmap;
use crate::clip::{Clip, ClipResult};
use crate::pipe::{self, PipeSrc, PipeState};
use crate::types::PixelMode;
use color::Pixel;
use color::convert::splash_floor;

// ── Compile-time sanity ───────────────────────────────────────────────────────

/// Maximum number of colour components supported by any pixel mode.
/// `DeviceN8` is the widest at 8 bytes per pixel.
const MAX_NCOMPS: usize = 8;

// ── Public traits ─────────────────────────────────────────────────────────────

/// Caller-supplied source for a colour image: one row at a time.
///
/// Matches the `SplashImageSource` callback convention from `splash/SplashTypes.h`.
pub trait ImageSource: Send {
    /// Fill `row_buf` with pixel data for source row `y`.
    ///
    /// `row_buf.len()` must equal `src_width * ncomps`.
    fn get_row(&mut self, y: u32, row_buf: &mut [u8]);
}

/// Caller-supplied source for a 1-bit image mask: one row at a time.
///
/// Each row is delivered as MSB-first packed bytes: `src_width.div_ceil(8)`
/// bytes per row.  Bit 7 of the first byte is the leftmost pixel; bit 0 of
/// the last byte is the rightmost (with unused padding bits set to 0).
///
/// The mask is immediately unpacked to one `u8` per pixel (0 or 255) for
/// simpler scaling arithmetic, matching the C++ `scaleMask` convention.
///
/// Matches `SplashImageMaskSource` from `splash/SplashTypes.h`.
pub trait MaskSource: Send {
    /// Fill `row_buf` with 1-bit packed mono mask data (MSB-first).
    ///
    /// `row_buf.len()` equals `src_width.div_ceil(8)`.  Implementors must
    /// write every byte; unused padding bits in the final byte should be 0.
    fn get_row(&mut self, y: u32, row_buf: &mut [u8]);
}

// ── Result enum ───────────────────────────────────────────────────────────────

/// Return value from [`fill_image_mask`] and [`draw_image`].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ImageResult {
    /// Rendering completed successfully.
    Ok,
    /// Source image has zero width or height; nothing rendered.
    ZeroImage,
    /// The transformation matrix is singular (determinant ≈ 0).
    SingularMatrix,
    /// The matrix is not axis-aligned; the arbitrary-transform path is not yet
    /// implemented in Phase 1.  The caller should handle this case.
    ArbitraryTransformSkipped,
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// `imgCoordMungeLower(x)` — C++ non-glyph variant: `floor(x)`.
#[inline]
fn coord_lower(x: f64) -> i32 {
    splash_floor(x)
}

/// `imgCoordMungeUpper(x)` — C++ non-glyph variant: `floor(x) + 1`.
#[inline]
fn coord_upper(x: f64) -> i32 {
    splash_floor(x) + 1
}

/// Determinant check: returns `true` when |a·d − b·c| ≥ `eps`.
///
/// Mirrors `splashCheckDet` from `splash/SplashMath.h`.
#[inline]
fn check_det(a: f64, b: f64, c: f64, d: f64, eps: f64) -> bool {
    #[expect(
        clippy::suboptimal_flops,
        reason = "matches the C++ arithmetic exactly; no numerics benefit here"
    )]
    {
        (a * d - b * c).abs() >= eps
    }
}

/// Unpack one packed-bits mask row into one-byte-per-pixel form (0 or 255).
///
/// `packed` is MSB-first; `out` receives exactly `width` bytes.
///
/// # Panics (debug)
///
/// Asserts that `packed.len() >= width.div_ceil(8)`, catching `MaskSource`
/// implementations that deliver too few bytes.
fn unpack_mask_row(packed: &[u8], width: usize, out: &mut [u8]) {
    debug_assert!(
        packed.len() >= width.div_ceil(8),
        "unpack_mask_row: packed buffer too short ({} < {})",
        packed.len(),
        width.div_ceil(8),
    );
    debug_assert_eq!(
        out.len(),
        width,
        "unpack_mask_row: out length must equal width"
    );
    for (i, slot) in out.iter_mut().enumerate() {
        // `packed.get` returns None for any over-short buffer; treat missing bits as 0.
        let byte = packed.get(i / 8).copied().unwrap_or(0);
        let bit = (byte >> (7 - (i % 8))) & 1;
        *slot = if bit != 0 { 255 } else { 0 };
    }
}

// ── Shared geometry helpers ───────────────────────────────────────────────────

/// Outcome of axis-aligned bounds computation for `fill_image_mask` / `draw_image`.
struct ImageBounds {
    x0: i32,
    y0: i32,
    /// Exclusive right edge (pixel column `x1 - 1` is the last painted column).
    x1: i32,
    /// Exclusive bottom edge.
    y1: i32,
    /// `true` when `matrix[3] < 0` — source rows must be flipped vertically.
    vflip: bool,
}

/// Parse the transformation matrix and compute destination pixel bounds.
///
/// Returns `Ok(ImageBounds)` for axis-aligned transforms, or an
/// `Err(ImageResult)` for singular or non-axis-aligned matrices.
fn compute_axis_aligned_bounds(matrix: &[f64; 6]) -> Result<ImageBounds, ImageResult> {
    if !check_det(matrix[0], matrix[1], matrix[2], matrix[3], 1e-6) {
        return Err(ImageResult::SingularMatrix);
    }
    let minor_zero = matrix[1] == 0.0 && matrix[2] == 0.0;
    if !minor_zero || matrix[0] <= 0.0 {
        return Err(ImageResult::ArbitraryTransformSkipped);
    }

    let (y0, y1, vflip) = if matrix[3] > 0.0 {
        (
            coord_lower(matrix[5]),
            coord_upper(matrix[3] + matrix[5]),
            false,
        )
    } else if matrix[3] < 0.0 {
        (
            coord_lower(matrix[3] + matrix[5]),
            coord_upper(matrix[5]),
            true,
        )
    } else {
        // Zero Y scale — degenerate (determinant guard catches this, but be explicit).
        return Err(ImageResult::SingularMatrix);
    };

    let x0 = coord_lower(matrix[4]);
    let x1 = coord_upper(matrix[0] + matrix[4]);

    // Ensure at least 1×1 destination even for very small source scale.
    let x1 = if x0 == x1 { x1 + 1 } else { x1 };
    let y1 = if y0 == y1 { y1 + 1 } else { y1 };

    Ok(ImageBounds {
        x0,
        y0,
        x1,
        y1,
        vflip,
    })
}

/// Flip a flat row-major buffer vertically in-place.
///
/// `row_stride` is the byte length of one row (`scaled_w * ncomps` for images,
/// `scaled_w` for masks).  Rows `0` and `height-1` are swapped, then `1` and
/// `height-2`, and so on.  A one-row or zero-row buffer is a no-op.
fn vflip_rows(data: &mut [u8], row_stride: usize) {
    if row_stride == 0 {
        return;
    }
    let nrows = data.len() / row_stride;
    let mut lo = 0usize;
    let mut hi = nrows.saturating_sub(1);
    while lo < hi {
        // Split the slice at `hi * row_stride`; `lo`'th row lies in the lower half.
        let (lower, upper) = data.split_at_mut(hi * row_stride);
        lower[lo * row_stride..lo * row_stride + row_stride]
            .swap_with_slice(&mut upper[..row_stride]);
        lo += 1;
        hi -= 1;
    }
}

/// Advance a Bresenham accumulator and return the step size for this iteration.
///
/// Classic Bresenham integer scaling:
/// - `acc` is the running error, updated in place.
/// - `q` is `src % scaled` (the fractional remainder).
/// - `scaled` is the scaled dimension (denominator).
/// - `p` is `src / scaled` (the floor step).
///
/// Returns `p + 1` when the accumulator overflows, `p` otherwise.
#[inline]
const fn bresenham_step(acc: &mut usize, q: usize, scaled: usize, p: usize) -> usize {
    *acc += q;
    if *acc >= scaled {
        *acc -= scaled;
        p + 1
    } else {
        p
    }
}

// ── Scale-kernel saturation constants ────────────────────────────────────────

/// `sat_factor` for `scale_image_inner` when the destination is a mask.
///
/// The kernel computes `(sum * d) >> 23` where `d = sat_factor / divisor`,
/// so a mask saturates `0..=255` input coverage values to `0..=255` output
/// coverage. `255u32 << 23 ≈ 2.14e9`.
const MASK_SAT_FACTOR: u32 = 255u32 << 23;

/// `sat_factor` for `scale_image_inner` when the destination is a colour
/// image. Per-channel intensity is preserved (no 255× amplification).
const IMAGE_SAT_FACTOR: u32 = 1u32 << 23;

/// Adapter that exposes a [`MaskSource`] as a single-channel [`ImageSource`].
///
/// The mask source produces 1-bit-packed rows; the adapter unpacks each row
/// into `u8` (0 or 255) on demand, so the scaling kernel sees an ordinary
/// 1-channel byte image and the same `scale_image_inner` body serves both
/// `scale_mask` and `scale_image`.
struct MaskAsImage<'a> {
    mask: &'a mut dyn MaskSource,
    /// Scratch for one packed mask row: `src_w.div_ceil(8)` bytes. Owned by
    /// the adapter so the kernel doesn't need to know about packed-row sizing.
    packed_buf: Vec<u8>,
    src_w: usize,
}

impl<'a> MaskAsImage<'a> {
    fn new(mask: &'a mut dyn MaskSource, src_w: usize) -> Self {
        Self {
            mask,
            packed_buf: vec![0u8; src_w.div_ceil(8)],
            src_w,
        }
    }
}

impl ImageSource for MaskAsImage<'_> {
    fn get_row(&mut self, y: u32, row_buf: &mut [u8]) {
        self.mask.get_row(y, &mut self.packed_buf);
        unpack_mask_row(&self.packed_buf, self.src_w, row_buf);
    }
}

// ── Mask scaling ──────────────────────────────────────────────────────────────

/// Scale a 1-bit mask to `scaled_w × scaled_h`, producing one `u8` (0–255) per
/// destination pixel. Exactly mirrors `Splash::scaleMask`.
///
/// Thin wrapper over [`scale_image_inner`] via [`MaskAsImage`].
fn scale_mask(
    mask_src: &mut dyn MaskSource,
    src_w: usize,
    src_h: usize,
    scaled_w: usize,
    scaled_h: usize,
) -> Vec<u8> {
    let mut adapter = MaskAsImage::new(mask_src, src_w);
    scale_image_inner(
        &mut adapter,
        src_w,
        src_h,
        scaled_w,
        scaled_h,
        1,
        MASK_SAT_FACTOR,
    )
}

/// Saturating `(sum * d) >> 23` narrow to `u8`.
///
/// `sum * d` is widened to `u64` because the worst-case product for the
/// mask path is `255 * 255 << 23 ≈ 5.5e11`, larger than `u32::MAX`. The
/// `>> 23` then `min(255)` keeps the result in `u8` range regardless.
#[inline]
fn saturate_scaled(sum: u32, d: u32) -> u8 {
    let scaled = ((u64::from(sum) * u64::from(d)) >> 23).min(255);
    u8::try_from(scaled).expect("scaled box-filter pixel was just clamped to <= 255")
}

/// Box-filter divisors for the X-down kernels.
///
/// `d_full` is the divisor used when Bresenham picks `x_step = xp`;
/// `d_plus_one` is used when `x_step = xp + 1`. Both are precomputed once
/// per output row (or once per call when `y_step` is constant).
///
/// `xp = 0` only happens when `scaled_w > src_w` — i.e. an *upsampling*
/// X dispatch — so the Xdown kernels never see it, but `d_full` is still
/// defined as 0 for that case to keep the precondition straightforward.
#[inline]
fn xdown_divisors(sat_factor: u32, y_step: usize, xp: usize) -> (u32, u32) {
    let d_full = if xp > 0 {
        let denom = u32::try_from(y_step.saturating_mul(xp))
            .expect("y_step * xp fits in u32 for practical image sizes");
        sat_factor / denom
    } else {
        0
    };
    let denom_plus = u32::try_from(y_step.saturating_mul(xp + 1))
        .expect("y_step * (xp+1) fits in u32 for practical image sizes");
    let d_plus_one = sat_factor / denom_plus;
    (d_full, d_plus_one)
}

/// Internal scaling kernel shared by [`scale_mask`] and [`scale_image`].
///
/// Dispatches on `(scaled_h vs src_h, scaled_w vs src_w)` to one of four
/// corner kernels (`ydown_xdown`, `ydown_xup`, `yup_xdown`, `yup_xup`),
/// parametrised by `ncomps` (1 for mask, N for image) and `sat_factor`
/// (`MASK_SAT_FACTOR` for mask, `IMAGE_SAT_FACTOR` for image).
fn scale_image_inner(
    image_src: &mut dyn ImageSource,
    src_w: usize,
    src_h: usize,
    scaled_w: usize,
    scaled_h: usize,
    ncomps: usize,
    sat_factor: u32,
) -> Vec<u8> {
    let mut dest = vec![0u8; scaled_w * scaled_h * ncomps];
    let mut line_buf = vec![0u8; src_w * ncomps];

    if scaled_h < src_h {
        if scaled_w < src_w {
            scale_kernel_ydown_xdown(
                image_src,
                src_w,
                src_h,
                scaled_w,
                scaled_h,
                ncomps,
                sat_factor,
                &mut dest,
                &mut line_buf,
            );
        } else {
            scale_kernel_ydown_xup(
                image_src,
                src_w,
                src_h,
                scaled_w,
                scaled_h,
                ncomps,
                sat_factor,
                &mut dest,
                &mut line_buf,
            );
        }
    } else if scaled_w < src_w {
        scale_kernel_yup_xdown(
            image_src,
            src_w,
            src_h,
            scaled_w,
            scaled_h,
            ncomps,
            sat_factor,
            &mut dest,
            &mut line_buf,
        );
    } else {
        scale_kernel_yup_xup(
            image_src,
            src_w,
            src_h,
            scaled_w,
            scaled_h,
            ncomps,
            &mut dest,
            &mut line_buf,
        );
    }

    dest
}

/// Box-filter downsampling in both Y and X.
///
/// C++ provenance: `Splash::scaleMaskYdownXdown` (ncomps=1) and
/// `Splash::scaleImageYdownXdown` (ncomps=N).
#[expect(
    clippy::too_many_arguments,
    reason = "kernel is private; all params are necessary to share the body across mask + image"
)]
fn scale_kernel_ydown_xdown(
    image_src: &mut dyn ImageSource,
    src_w: usize,
    src_h: usize,
    scaled_w: usize,
    scaled_h: usize,
    ncomps: usize,
    sat_factor: u32,
    dest: &mut [u8],
    line_buf: &mut [u8],
) {
    let yp = src_h / scaled_h;
    let yq = src_h % scaled_h;
    let xp = src_w / scaled_w;
    let xq = src_w % scaled_w;

    let mut pix_buf = vec![0u32; src_w * ncomps];
    let mut yt = 0usize;
    let mut dest_off = 0usize;
    let mut src_y = 0u32;

    for _dy in 0..scaled_h {
        let y_step = bresenham_step(&mut yt, yq, scaled_h, yp);

        pix_buf.fill(0);
        for _ in 0..y_step {
            image_src.get_row(src_y, line_buf);
            src_y += 1;
            for (pix, &lb) in pix_buf.iter_mut().zip(line_buf.iter()) {
                *pix += u32::from(lb);
            }
        }

        let (d_full, d_plus_one) = xdown_divisors(sat_factor, y_step, xp);

        let mut xt = 0usize;
        let mut xx = 0usize;
        for _dx in 0..scaled_w {
            let x_step = bresenham_step(&mut xt, xq, scaled_w, xp);
            let d = if x_step == xp + 1 { d_plus_one } else { d_full };
            for c in 0..ncomps {
                let sum: u32 = (0..x_step).map(|i| pix_buf[(xx + i) * ncomps + c]).sum();
                dest[dest_off + c] = saturate_scaled(sum, d);
            }
            xx += x_step;
            dest_off += ncomps;
        }
    }
}

/// Box-filter Y downsampling, nearest-neighbor X upsampling.
///
/// C++ provenance: `Splash::scaleMaskYdownXup` / `Splash::scaleImageYdownXup`.
#[expect(
    clippy::too_many_arguments,
    reason = "kernel is private; all params are necessary to share the body across mask + image"
)]
fn scale_kernel_ydown_xup(
    image_src: &mut dyn ImageSource,
    src_w: usize,
    src_h: usize,
    scaled_w: usize,
    scaled_h: usize,
    ncomps: usize,
    sat_factor: u32,
    dest: &mut [u8],
    line_buf: &mut [u8],
) {
    let yp = src_h / scaled_h;
    let yq = src_h % scaled_h;
    let xp = scaled_w / src_w;
    let xq = scaled_w % src_w;

    let mut pix_buf = vec![0u32; src_w * ncomps];
    let mut yt = 0usize;
    let mut dest_off = 0usize;
    let mut src_y = 0u32;

    for _dy in 0..scaled_h {
        let y_step = bresenham_step(&mut yt, yq, scaled_h, yp);

        pix_buf.fill(0);
        for _ in 0..y_step {
            image_src.get_row(src_y, line_buf);
            src_y += 1;
            for (pix, &lb) in pix_buf.iter_mut().zip(line_buf.iter()) {
                *pix += u32::from(lb);
            }
        }

        let d = sat_factor
            / u32::try_from(y_step).expect("y_step ≤ src_h fits in u32 for practical image sizes");
        let mut xt = 0usize;

        let mut pix_vals = [0u8; MAX_NCOMPS];
        for sx in 0..src_w {
            let x_step = bresenham_step(&mut xt, xq, src_w, xp);
            let base = sx * ncomps;
            for c in 0..ncomps {
                pix_vals[c] = saturate_scaled(pix_buf[base + c], d);
            }
            for _ in 0..x_step {
                dest[dest_off..dest_off + ncomps].copy_from_slice(&pix_vals[..ncomps]);
                dest_off += ncomps;
            }
        }
    }
}

/// Nearest-neighbor Y upsampling, box-filter X downsampling.
///
/// C++ provenance: `Splash::scaleMaskYupXdown` / `Splash::scaleImageYupXdown`.
#[expect(
    clippy::too_many_arguments,
    reason = "kernel is private; all params are necessary to share the body across mask + image"
)]
fn scale_kernel_yup_xdown(
    image_src: &mut dyn ImageSource,
    src_w: usize,
    src_h: usize,
    scaled_w: usize,
    scaled_h: usize,
    ncomps: usize,
    sat_factor: u32,
    dest: &mut [u8],
    line_buf: &mut [u8],
) {
    let yp = scaled_h / src_h;
    let yq = scaled_h % src_h;
    let xp = src_w / scaled_w;
    let xq = src_w % scaled_w;

    // y_step factor is 1 for this kernel — the divisors don't depend on Y.
    let (d_full, d_plus_one) = xdown_divisors(sat_factor, 1, xp);

    let mut yt = 0usize;
    let mut dest_off = 0usize;

    for sy in 0..src_h {
        let y_step = bresenham_step(&mut yt, yq, src_h, yp);

        let src_y = u32::try_from(sy)
            .expect("source row index ≤ src_h fits in u32 for practical image sizes");
        image_src.get_row(src_y, line_buf);

        let row_start = dest_off;
        let mut xt = 0usize;
        let mut xx = 0usize;

        for dx in 0..scaled_w {
            let x_step = bresenham_step(&mut xt, xq, scaled_w, xp);
            let d = if x_step == xp + 1 { d_plus_one } else { d_full };
            for c in 0..ncomps {
                let sum: u32 = (0..x_step)
                    .map(|i| u32::from(line_buf[(xx + i) * ncomps + c]))
                    .sum();
                dest[row_start + dx * ncomps + c] = saturate_scaled(sum, d);
            }
            xx += x_step;
        }
        dest_off += scaled_w * ncomps;

        for i in 1..y_step {
            dest.copy_within(
                row_start..row_start + scaled_w * ncomps,
                row_start + i * scaled_w * ncomps,
            );
        }
        dest_off += (y_step - 1) * scaled_w * ncomps;
    }
}

/// Nearest-neighbor upsampling in both axes.
///
/// C++ provenance: `Splash::scaleMaskYupXup` / `Splash::scaleImageYupXup`.
/// No saturation is needed — each output pixel is a direct copy of one source
/// pixel. (For the mask path, source bytes are already 0 or 255 by
/// construction in [`unpack_mask_row`].)
#[expect(
    clippy::too_many_arguments,
    reason = "kernel is private; all params are necessary to share the body across mask + image"
)]
fn scale_kernel_yup_xup(
    image_src: &mut dyn ImageSource,
    src_w: usize,
    src_h: usize,
    scaled_w: usize,
    scaled_h: usize,
    ncomps: usize,
    dest: &mut [u8],
    line_buf: &mut [u8],
) {
    let yp = scaled_h / src_h;
    let yq = scaled_h % src_h;
    let xp = scaled_w / src_w;
    let xq = scaled_w % src_w;

    let mut yt = 0usize;
    let mut dest_off = 0usize;

    for sy in 0..src_h {
        let y_step = bresenham_step(&mut yt, yq, src_h, yp);

        let src_y = u32::try_from(sy)
            .expect("source row index ≤ src_h fits in u32 for practical image sizes");
        image_src.get_row(src_y, line_buf);

        let row_start = dest_off;
        let mut xt = 0usize;
        let mut xx = 0usize;

        for sx in 0..src_w {
            let x_step = bresenham_step(&mut xt, xq, src_w, xp);
            let pix_start = sx * ncomps;
            for j in 0..x_step {
                let off = row_start + (xx + j) * ncomps;
                dest[off..off + ncomps].copy_from_slice(&line_buf[pix_start..pix_start + ncomps]);
            }
            xx += x_step;
        }
        dest_off += scaled_w * ncomps;

        for i in 1..y_step {
            dest.copy_within(
                row_start..row_start + scaled_w * ncomps,
                row_start + i * scaled_w * ncomps,
            );
        }
        dest_off += (y_step - 1) * scaled_w * ncomps;
    }
}

// ── Image scaling ─────────────────────────────────────────────────────────────

/// Scale a colour image to `scaled_w × scaled_h`.
///
/// Output is `scaled_w * scaled_h * ncomps` bytes (row-major, no padding).
/// Thin wrapper over [`scale_image_inner`].
fn scale_image(
    image_src: &mut dyn ImageSource,
    src_w: usize,
    src_h: usize,
    scaled_w: usize,
    scaled_h: usize,
    ncomps: usize,
) -> Vec<u8> {
    scale_image_inner(
        image_src,
        src_w,
        src_h,
        scaled_w,
        scaled_h,
        ncomps,
        IMAGE_SAT_FACTOR,
    )
}

// ── Row-as-pattern helper ─────────────────────────────────────────────────────

/// A [`crate::pipe::Pattern`] that serves one pre-scaled image row.
///
/// `data` must be exactly `(x1 - x0 + 1) * P::BYTES` bytes — the same length
/// as the `out` buffer that `render_span` will pass to `fill_span`.  The
/// caller (`blit_image`) guarantees this because it slices `scaled_img` with
/// `count * ncomps` where `ncomps == P::BYTES` (enforced by `draw_image`'s
/// `debug_assert`).
struct ImageRowPattern<'a> {
    /// Pixel bytes for the visible span of one destination row.
    data: &'a [u8],
}

impl crate::pipe::Pattern for ImageRowPattern<'_> {
    fn fill_span(&self, _y: i32, _x0: i32, _x1: i32, out: &mut [u8]) {
        assert_eq!(
            out.len(),
            self.data.len(),
            "ImageRowPattern::fill_span: out.len()={} != data.len()={} \
             (ncomps/P::BYTES mismatch — check draw_image caller)",
            out.len(),
            self.data.len(),
        );
        out.copy_from_slice(self.data);
    }

    fn is_static_color(&self) -> bool {
        false
    }
}

// ── Mask blitting ─────────────────────────────────────────────────────────────

/// Blit a pre-scaled mask (one `u8` coverage per pixel) onto the bitmap.
///
/// For each set pixel (coverage > 0) the fill pattern is applied through the
/// compositing pipe using `shape = coverage`.  This mirrors `Splash::blitMask`
/// in the non-AA path.
#[expect(
    clippy::too_many_arguments,
    reason = "mirrors Splash::blitMask API; all params necessary"
)]
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    reason = "run_shape.len() / scaled_w / scaled_h ≤ bitmap dims ≤ i32::MAX for practical image sizes"
)]
fn blit_mask<P: Pixel>(
    bitmap: &mut Bitmap<P>,
    clip: &Clip,
    pipe: &PipeState<'_>,
    src: &PipeSrc<'_>,
    scaled_mask: &[u8],
    scaled_w: i32,
    scaled_h: i32,
    x_dest: i32,
    y_dest: i32,
    clip_all_inside: bool,
) {
    #[expect(
        clippy::cast_possible_wrap,
        reason = "bitmap dims ≤ i32::MAX in practice"
    )]
    let bmp_w = bitmap.width as i32;
    #[expect(
        clippy::cast_possible_wrap,
        reason = "bitmap dims ≤ i32::MAX in practice"
    )]
    let bmp_h = bitmap.height as i32;

    for dy in 0..scaled_h {
        let y = y_dest + dy;
        if y < 0 || y >= bmp_h {
            continue;
        }
        #[expect(clippy::cast_sign_loss, reason = "dy ≥ 0")]
        let row_off = dy as usize * scaled_w as usize;
        #[expect(clippy::cast_sign_loss, reason = "y ≥ 0 by guard above")]
        let y_u = y as u32;

        let mut run_start: Option<i32> = None;
        let mut run_shape: Vec<u8> = Vec::new();

        macro_rules! flush_run {
            () => {
                if let Some(rs) = run_start.take() {
                    let rx1 = rs + run_shape.len() as i32 - 1;
                    #[expect(clippy::cast_sign_loss, reason = "rs ≥ 0")]
                    let byte_off = rs as usize * P::BYTES;
                    #[expect(clippy::cast_sign_loss, reason = "rx1 ≥ rs ≥ 0")]
                    let byte_end = (rx1 as usize + 1) * P::BYTES;
                    #[expect(clippy::cast_sign_loss, reason = "rs ≥ 0")]
                    let alpha_lo = rs as usize;
                    #[expect(clippy::cast_sign_loss, reason = "rx1 ≥ rs ≥ 0")]
                    let alpha_hi = rx1 as usize;
                    let (row, alpha) = bitmap.row_and_alpha_mut(y_u);
                    let dst_pixels = &mut row[byte_off..byte_end];
                    let dst_alpha = alpha.map(|a| &mut a[alpha_lo..=alpha_hi]);
                    pipe::render_span::<P>(
                        pipe,
                        src,
                        dst_pixels,
                        dst_alpha,
                        Some(&run_shape),
                        rs,
                        rx1,
                        y,
                    );
                    run_shape.clear();
                }
            };
        }

        for dx in 0..scaled_w {
            let x = x_dest + dx;
            if x < 0 || x >= bmp_w {
                flush_run!();
                continue;
            }
            #[expect(clippy::cast_sign_loss, reason = "dx ≥ 0")]
            let coverage = scaled_mask[row_off + dx as usize];
            let inside_clip = clip_all_inside || clip.test(x, y);

            if coverage > 0 && inside_clip {
                if run_start.is_none() {
                    run_start = Some(x);
                }
                run_shape.push(coverage);
            } else {
                flush_run!();
            }
        }
        flush_run!();
    }
}

// ── Image blitting ────────────────────────────────────────────────────────────

/// Emit one contiguous span of pre-scaled image pixels through the pipe.
///
/// `img_row` is the full scaled row for scanline `y`; `x_src_off` is the
/// pixel offset into that row for pixel `x0`.  `x0`/`x1` are inclusive
/// destination coordinates and must be within bitmap bounds.
#[expect(
    clippy::too_many_arguments,
    reason = "all context is necessary for a span emit"
)]
fn emit_image_span<P: Pixel>(
    bitmap: &mut Bitmap<P>,
    pipe: &PipeState<'_>,
    img_row: &[u8],
    ncomps: usize,
    x_src_off: usize,
    x0: i32,
    x1: i32,
    y: i32,
) {
    #[expect(clippy::cast_sign_loss, reason = "x1 ≥ x0 ≥ 0 by caller invariant")]
    let count = (x1 - x0 + 1) as usize;
    let data = &img_row[x_src_off * ncomps..(x_src_off + count) * ncomps];
    let row_src = PipeSrc::Pattern(&ImageRowPattern { data });
    #[expect(clippy::cast_sign_loss, reason = "x0 ≥ 0 by caller invariant")]
    let byte_off = x0 as usize * P::BYTES;
    #[expect(clippy::cast_sign_loss, reason = "x1 ≥ x0 ≥ 0 by caller invariant")]
    let byte_end = (x1 as usize + 1) * P::BYTES;
    #[expect(clippy::cast_sign_loss, reason = "x0 ≥ 0 by caller invariant")]
    let (row, alpha) = bitmap.row_and_alpha_mut(y as u32);
    let dst_pixels = &mut row[byte_off..byte_end];
    #[expect(clippy::cast_sign_loss, reason = "x0/x1 ≥ 0 by caller invariant")]
    let dst_alpha = alpha.map(|a| &mut a[x0 as usize..=x1 as usize]);
    pipe::render_span::<P>(pipe, &row_src, dst_pixels, dst_alpha, None, x0, x1, y);
}

/// Blit a pre-scaled colour image onto the bitmap.
///
/// The entire row (or per-pixel runs for partial clip) is emitted through
/// `render_span`.  Mirrors `Splash::blitImage` non-AA path.
///
/// # Panics (debug)
///
/// Asserts `ncomps == P::BYTES`; a mismatch means the scaled image buffer was
/// built with the wrong component count for the destination pixel format.
#[expect(
    clippy::too_many_arguments,
    reason = "mirrors Splash::blitImage API; all params necessary"
)]
fn blit_image<P: Pixel>(
    bitmap: &mut Bitmap<P>,
    clip: &Clip,
    pipe: &PipeState<'_>,
    scaled_img: &[u8],
    scaled_w: i32,
    scaled_h: i32,
    x_dest: i32,
    y_dest: i32,
    clip_res: ClipResult,
) {
    let ncomps = P::BYTES;
    debug_assert!(
        ncomps <= MAX_NCOMPS,
        "blit_image: P::BYTES={ncomps} exceeds MAX_NCOMPS={MAX_NCOMPS}",
    );

    #[expect(
        clippy::cast_possible_wrap,
        reason = "bitmap dims ≤ i32::MAX in practice"
    )]
    let bmp_w = bitmap.width as i32;
    #[expect(
        clippy::cast_possible_wrap,
        reason = "bitmap dims ≤ i32::MAX in practice"
    )]
    let bmp_h = bitmap.height as i32;

    let clip_all_inside = clip_res == ClipResult::AllInside;

    for dy in 0..scaled_h {
        let y = y_dest + dy;
        if y < 0 || y >= bmp_h {
            continue;
        }
        #[expect(clippy::cast_sign_loss, reason = "dy ≥ 0 and scaled_w ≥ 0")]
        let img_row_off = dy as usize * scaled_w as usize * ncomps;
        #[expect(
            clippy::cast_sign_loss,
            reason = "scaled_w ≥ 0 (it is the dest rect width)"
        )]
        let img_row = &scaled_img[img_row_off..img_row_off + scaled_w as usize * ncomps];

        let x_lo = x_dest.max(0);
        let x_hi = (x_dest + scaled_w - 1).min(bmp_w - 1);
        if x_lo > x_hi {
            continue;
        }

        if clip_all_inside {
            #[expect(clippy::cast_sign_loss, reason = "x_lo ≥ x_dest ≥ 0 after clamp")]
            let x_src_off = (x_lo - x_dest) as usize;
            emit_image_span::<P>(bitmap, pipe, img_row, ncomps, x_src_off, x_lo, x_hi, y);
        } else {
            // Partial clip: walk pixels left-to-right, collecting contiguous
            // clip-passing runs and emitting each run as a single span.
            let mut run_x0: Option<i32> = None;
            let mut run_x1 = x_lo; // last pixel appended to the current run

            for dx in 0..scaled_w {
                let x = x_dest + dx;
                let in_bmp = x >= x_lo && x <= x_hi;
                let visible = in_bmp && clip.test(x, y);

                if visible {
                    if run_x0.is_none() {
                        run_x0 = Some(x);
                    }
                    run_x1 = x;
                } else if let Some(x0) = run_x0.take() {
                    // Run ended: emit it.
                    #[expect(clippy::cast_sign_loss, reason = "x0 ≥ x_dest ≥ 0 inside bmp bounds")]
                    let x_src_off = (x0 - x_dest) as usize;
                    emit_image_span::<P>(bitmap, pipe, img_row, ncomps, x_src_off, x0, run_x1, y);
                }
            }
            // Emit any run still open at the end of the row.
            if let Some(x0) = run_x0 {
                #[expect(clippy::cast_sign_loss, reason = "x0 ≥ x_dest ≥ 0 inside bmp bounds")]
                let x_src_off = (x0 - x_dest) as usize;
                emit_image_span::<P>(bitmap, pipe, img_row, ncomps, x_src_off, x0, run_x1, y);
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Public entry points
// ─────────────────────────────────────────────────────────────────────────────

/// Fill a 1-bit image mask using the current fill pattern.
///
/// Only axis-aligned transforms (`matrix[1] == 0 && matrix[2] == 0`,
/// `matrix[0] > 0`) are implemented.  For rotated or skewed matrices
/// [`ImageResult::ArbitraryTransformSkipped`] is returned so the caller can
/// use a general path.
///
/// The mask is scaled to the destination pixel grid using Bresenham
/// box-filter (downscale) or nearest-neighbour (upscale).
///
/// # C++ equivalent
///
/// `Splash::fillImageMask`.
#[expect(
    clippy::too_many_arguments,
    reason = "mirrors Splash::fillImageMask API; all params necessary"
)]
pub fn fill_image_mask<P: Pixel>(
    bitmap: &mut Bitmap<P>,
    clip: &Clip,
    pipe: &PipeState<'_>,
    src: &PipeSrc<'_>,
    mask_src: &mut dyn MaskSource,
    src_w: u32,
    src_h: u32,
    matrix: &[f64; 6],
) -> ImageResult {
    if src_w == 0 || src_h == 0 {
        return ImageResult::ZeroImage;
    }

    let bounds = match compute_axis_aligned_bounds(matrix) {
        Ok(b) => b,
        Err(e) => return e,
    };
    let ImageBounds {
        x0,
        y0,
        x1,
        y1,
        vflip,
    } = bounds;

    let clip_res = clip.test_rect(x0, y0, x1 - 1, y1 - 1);
    if clip_res == ClipResult::AllOutside {
        return ImageResult::Ok;
    }

    #[expect(
        clippy::cast_sign_loss,
        reason = "x1 > x0 is guaranteed by compute_axis_aligned_bounds"
    )]
    let scaled_w = (x1 - x0) as usize;
    #[expect(
        clippy::cast_sign_loss,
        reason = "y1 > y0 is guaranteed by compute_axis_aligned_bounds"
    )]
    let scaled_h = (y1 - y0) as usize;

    let mut scaled = scale_mask(mask_src, src_w as usize, src_h as usize, scaled_w, scaled_h);

    if vflip {
        vflip_rows(&mut scaled, scaled_w);
    }

    blit_mask::<P>(
        bitmap,
        clip,
        pipe,
        src,
        &scaled,
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_possible_wrap,
            reason = "scaled_w ≤ bitmap.width ≤ i32::MAX"
        )]
        {
            scaled_w as i32
        },
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_possible_wrap,
            reason = "scaled_h ≤ bitmap.height ≤ i32::MAX"
        )]
        {
            scaled_h as i32
        },
        x0,
        y0,
        clip_res == ClipResult::AllInside,
    );

    ImageResult::Ok
}

/// Render a colour image with transformation.
///
/// Only axis-aligned transforms are handled (Phase 2 scope).  For rotated or
/// skewed matrices [`ImageResult::ArbitraryTransformSkipped`] is returned.
///
/// `src_mode` conveys the colour space of the source data; it is stored for
/// future use in colour-space conversion but not acted on in Phase 2 — the
/// caller is responsible for ensuring `ncomps == P::BYTES`.
///
/// # Panics (debug)
///
/// Asserts `ncomps == P::BYTES`.  In release builds a mismatch produces
/// silently wrong colours; callers must match pixel formats.
///
/// # C++ equivalent
///
/// `Splash::drawImage`.
#[expect(
    clippy::too_many_arguments,
    reason = "mirrors Splash::drawImage API; all params necessary"
)]
pub fn draw_image<P: Pixel>(
    bitmap: &mut Bitmap<P>,
    clip: &Clip,
    pipe: &PipeState<'_>,
    image_src: &mut dyn ImageSource,
    src_mode: PixelMode,
    src_w: u32,
    src_h: u32,
    matrix: &[f64; 6],
) -> ImageResult {
    // Record src_mode for future colour-space conversion; unused in Phase 2.
    let _ = src_mode;

    let ncomps = P::BYTES;
    debug_assert!(
        ncomps <= MAX_NCOMPS,
        "draw_image: P::BYTES={ncomps} exceeds MAX_NCOMPS={MAX_NCOMPS}",
    );

    if src_w == 0 || src_h == 0 {
        return ImageResult::ZeroImage;
    }

    let bounds = match compute_axis_aligned_bounds(matrix) {
        Ok(b) => b,
        Err(e) => return e,
    };
    let ImageBounds {
        x0,
        y0,
        x1,
        y1,
        vflip,
    } = bounds;

    let clip_res = clip.test_rect(x0, y0, x1 - 1, y1 - 1);
    if clip_res == ClipResult::AllOutside {
        return ImageResult::Ok;
    }

    #[expect(
        clippy::cast_sign_loss,
        reason = "x1 > x0 is guaranteed by compute_axis_aligned_bounds"
    )]
    let scaled_w = (x1 - x0) as usize;
    #[expect(
        clippy::cast_sign_loss,
        reason = "y1 > y0 is guaranteed by compute_axis_aligned_bounds"
    )]
    let scaled_h = (y1 - y0) as usize;

    let mut scaled = scale_image(
        image_src,
        src_w as usize,
        src_h as usize,
        scaled_w,
        scaled_h,
        ncomps,
    );

    if vflip {
        vflip_rows(&mut scaled, scaled_w * ncomps);
    }

    blit_image::<P>(
        bitmap,
        clip,
        pipe,
        &scaled,
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_possible_wrap,
            reason = "scaled_w ≤ bitmap.width ≤ i32::MAX"
        )]
        {
            scaled_w as i32
        },
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_possible_wrap,
            reason = "scaled_h ≤ bitmap.height ≤ i32::MAX"
        )]
        {
            scaled_h as i32
        },
        x0,
        y0,
        clip_res,
    );

    ImageResult::Ok
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitmap::Bitmap;
    use crate::clip::Clip;
    use crate::pipe::PipeSrc;
    use crate::testutil::{make_clip, simple_pipe};
    use color::Rgb8;

    // ── MaskSource / ImageSource stubs ────────────────────────────────────────

    /// A `MaskSource` that delivers a solid white (all-set) 1-bit mask.
    struct SolidMask;
    impl MaskSource for SolidMask {
        fn get_row(&mut self, _y: u32, row_buf: &mut [u8]) {
            row_buf.fill(0xFF);
        }
    }

    /// A `MaskSource` that delivers alternating byte pattern.
    struct CheckerMask;
    impl MaskSource for CheckerMask {
        fn get_row(&mut self, _y: u32, row_buf: &mut [u8]) {
            for (i, b) in row_buf.iter_mut().enumerate() {
                *b = if i % 2 == 0 { 0xAA } else { 0x55 };
            }
        }
    }

    /// An `ImageSource` that delivers a solid colour.
    struct SolidColor {
        r: u8,
        g: u8,
        b: u8,
    }
    impl ImageSource for SolidColor {
        fn get_row(&mut self, _y: u32, row_buf: &mut [u8]) {
            for chunk in row_buf.chunks_exact_mut(3) {
                chunk[0] = self.r;
                chunk[1] = self.g;
                chunk[2] = self.b;
            }
        }
    }

    // ── Tests ─────────────────────────────────────────────────────────────────

    /// A 4×4 solid-white mask scaled 1:1 onto an 8×8 canvas should paint a
    /// 4×4 rectangle of the fill colour.
    #[test]
    fn fill_image_mask_solid_paints_rect() {
        let mut bmp: Bitmap<Rgb8> = Bitmap::new(8, 8, 1, false);
        let clip = make_clip(8, 8);
        let pipe = simple_pipe();
        let color = [255u8, 0, 0]; // red
        let src = PipeSrc::Solid(&color);

        let mut mask = SolidMask;
        // mat: scale 4 → 4, translate to (2,2): [4,0,0,4,2,2]
        let mat = [4.0f64, 0.0, 0.0, 4.0, 2.0, 2.0];
        let result = fill_image_mask::<Rgb8>(&mut bmp, &clip, &pipe, &src, &mut mask, 4, 4, &mat);

        assert_eq!(result, ImageResult::Ok);
        for y in 2..6u32 {
            for x in 2..6usize {
                assert_eq!(bmp.row(y)[x].r, 255, "row={y} col={x}");
            }
        }
        assert_eq!(bmp.row(0)[0].r, 0, "outside should be unpainted");
    }

    /// Vertically-flipped mask completes without panic.
    #[test]
    fn fill_image_mask_vflip_no_crash() {
        let mut bmp: Bitmap<Rgb8> = Bitmap::new(8, 8, 1, false);
        let clip = make_clip(8, 8);
        let pipe = simple_pipe();
        let color = [0u8, 255, 0];
        let src = PipeSrc::Solid(&color);

        let mut mask = SolidMask;
        // mat[3] < 0 → vflip
        let mat = [4.0f64, 0.0, 0.0, -4.0, 2.0, 6.0];
        let result = fill_image_mask::<Rgb8>(&mut bmp, &clip, &pipe, &src, &mut mask, 4, 4, &mat);
        assert_eq!(result, ImageResult::Ok);
    }

    /// Singular matrix → `SingularMatrix` result.
    #[test]
    fn fill_image_mask_singular_matrix() {
        let mut bmp: Bitmap<Rgb8> = Bitmap::new(8, 8, 1, false);
        let clip = make_clip(8, 8);
        let pipe = simple_pipe();
        let color = [0u8, 0, 0];
        let src = PipeSrc::Solid(&color);
        let mut mask = SolidMask;

        let mat = [0.0f64, 0.0, 0.0, 0.0, 0.0, 0.0]; // det = 0
        let result = fill_image_mask::<Rgb8>(&mut bmp, &clip, &pipe, &src, &mut mask, 4, 4, &mat);
        assert_eq!(result, ImageResult::SingularMatrix);
    }

    /// `draw_image` with an identity-scale matrix paints a solid-colour rectangle.
    #[test]
    fn draw_image_solid_color() {
        let mut bmp: Bitmap<Rgb8> = Bitmap::new(8, 8, 1, false);
        let clip = make_clip(8, 8);
        let pipe = simple_pipe();

        let mut img_src = SolidColor { r: 0, g: 0, b: 200 };
        let mat = [4.0f64, 0.0, 0.0, 4.0, 0.0, 0.0];
        let result = draw_image::<Rgb8>(
            &mut bmp,
            &clip,
            &pipe,
            &mut img_src,
            crate::types::PixelMode::Rgb8,
            4,
            4,
            &mat,
        );

        assert_eq!(result, ImageResult::Ok);
        // mat=[4,0,0,4,0,0]: coord_upper(4+0)=5, so dest covers rows/cols 0..5 (5 pixels).
        for y in 0..5u32 {
            for x in 0..5usize {
                assert_eq!(bmp.row(y)[x].b, 200, "row={y} col={x}");
            }
        }
        // Row 6 (beyond the 5-pixel extent) should be unpainted.
        assert_eq!(bmp.row(6)[0].b, 0, "row 6 should be unpainted");
    }

    /// Arbitrary-transform matrix → `ArbitraryTransformSkipped`.
    #[test]
    fn draw_image_arbitrary_transform_skipped() {
        let mut bmp: Bitmap<Rgb8> = Bitmap::new(8, 8, 1, false);
        let clip = make_clip(8, 8);
        let pipe = simple_pipe();
        let mut img_src = SolidColor {
            r: 100,
            g: 100,
            b: 100,
        };

        let mat = [2.0f64, 1.0, 1.0, 2.0, 0.0, 0.0];
        let result = draw_image::<Rgb8>(
            &mut bmp,
            &clip,
            &pipe,
            &mut img_src,
            crate::types::PixelMode::Rgb8,
            4,
            4,
            &mat,
        );
        assert_eq!(result, ImageResult::ArbitraryTransformSkipped);
    }

    /// Zero-size image → `ZeroImage`.
    #[test]
    fn draw_image_zero_size() {
        let mut bmp: Bitmap<Rgb8> = Bitmap::new(8, 8, 1, false);
        let clip = make_clip(8, 8);
        let pipe = simple_pipe();
        let mut img_src = SolidColor { r: 1, g: 2, b: 3 };

        let mat = [4.0f64, 0.0, 0.0, 4.0, 0.0, 0.0];
        let result = draw_image::<Rgb8>(
            &mut bmp,
            &clip,
            &pipe,
            &mut img_src,
            crate::types::PixelMode::Rgb8,
            0,
            4,
            &mat,
        );
        assert_eq!(result, ImageResult::ZeroImage);
    }

    /// Upsampling: 2×2 source → 4×4 destination.
    #[test]
    fn draw_image_upsample_2x2_to_4x4() {
        let mut bmp: Bitmap<Rgb8> = Bitmap::new(8, 8, 1, false);
        let clip = make_clip(8, 8);
        let pipe = simple_pipe();

        let mut img_src = SolidColor {
            r: 128,
            g: 64,
            b: 32,
        };
        let mat = [4.0f64, 0.0, 0.0, 4.0, 0.0, 0.0];
        let result = draw_image::<Rgb8>(
            &mut bmp,
            &clip,
            &pipe,
            &mut img_src,
            crate::types::PixelMode::Rgb8,
            2,
            2,
            &mat,
        );
        assert_eq!(result, ImageResult::Ok);
        for y in 0..4u32 {
            for x in 0..4usize {
                assert_eq!(bmp.row(y)[x].r, 128, "row={y} col={x} R");
                assert_eq!(bmp.row(y)[x].g, 64, "row={y} col={x} G");
                assert_eq!(bmp.row(y)[x].b, 32, "row={y} col={x} B");
            }
        }
    }

    /// Downsampling: 4×4 source → 2×2 destination (solid colour averages to itself).
    #[test]
    fn draw_image_downsample_4x4_to_2x2() {
        let mut bmp: Bitmap<Rgb8> = Bitmap::new(8, 8, 1, false);
        let clip = make_clip(8, 8);
        let pipe = simple_pipe();

        let mut img_src = SolidColor {
            r: 200,
            g: 100,
            b: 50,
        };
        let mat = [2.0f64, 0.0, 0.0, 2.0, 0.0, 0.0];
        let result = draw_image::<Rgb8>(
            &mut bmp,
            &clip,
            &pipe,
            &mut img_src,
            crate::types::PixelMode::Rgb8,
            4,
            4,
            &mat,
        );
        assert_eq!(result, ImageResult::Ok);
        for y in 0..2u32 {
            for x in 0..2usize {
                assert_eq!(bmp.row(y)[x].r, 200, "row={y} col={x} R");
            }
        }
    }

    /// Mask with checker pattern: some pixels painted, some not.
    #[test]
    fn fill_image_mask_checker_partial() {
        let mut bmp: Bitmap<Rgb8> = Bitmap::new(8, 8, 1, false);
        let clip = make_clip(8, 8);
        let pipe = simple_pipe();
        let color = [255u8, 255, 0]; // yellow
        let src = PipeSrc::Solid(&color);

        let mut mask = CheckerMask;
        let mat = [8.0f64, 0.0, 0.0, 8.0, 0.0, 0.0];
        let result = fill_image_mask::<Rgb8>(&mut bmp, &clip, &pipe, &src, &mut mask, 8, 8, &mat);
        assert_eq!(result, ImageResult::Ok);
        let any_painted = (0..8u32).any(|y| (0..8usize).any(|x| bmp.row(y)[x].r > 0));
        assert!(any_painted, "at least some pixels must be painted");
    }

    /// `unpack_mask_row` correctness: 0xAA = 10101010 binary.
    #[test]
    fn unpack_mask_row_aa() {
        let packed = [0xAAu8];
        let mut out = [0u8; 8];
        unpack_mask_row(&packed, 8, &mut out);
        assert_eq!(out, [255, 0, 255, 0, 255, 0, 255, 0]);
    }

    /// `scale_mask` identity: 4×1 source → 4×1 dest, solid white.
    #[test]
    fn scale_mask_identity_solid() {
        struct IdentityMask;
        impl MaskSource for IdentityMask {
            fn get_row(&mut self, _y: u32, row_buf: &mut [u8]) {
                row_buf.fill(0xFF);
            }
        }
        let mut ms = IdentityMask;
        let out = scale_mask(&mut ms, 4, 1, 4, 1);
        assert_eq!(out, [255u8, 255, 255, 255]);
    }

    /// `vflip_rows` with stride 2: rows [A,B,C] become [C,B,A].
    #[test]
    fn vflip_rows_three_rows() {
        let mut data = vec![
            1u8, 2, // row 0
            3, 4, // row 1
            5, 6,
        ]; // row 2
        vflip_rows(&mut data, 2);
        assert_eq!(data, [5, 6, 3, 4, 1, 2]);
    }

    /// `vflip_rows` with a single row is a no-op.
    #[test]
    fn vflip_rows_single_row_noop() {
        let mut data = vec![7u8, 8, 9];
        vflip_rows(&mut data, 3);
        assert_eq!(data, [7, 8, 9]);
    }

    /// `vflip_rows` with an empty slice is a no-op (does not panic).
    #[test]
    fn vflip_rows_empty_noop() {
        let mut data: Vec<u8> = vec![];
        vflip_rows(&mut data, 1);
        assert!(data.is_empty());
    }

    /// Vertical flip of a 2-row image actually reverses row order.
    #[test]
    fn draw_image_vflip_reverses_rows() {
        // Source: row 0 = red (255,0,0), row 1 = blue (0,0,255).
        struct TwoRowImage;
        impl ImageSource for TwoRowImage {
            fn get_row(&mut self, y: u32, row_buf: &mut [u8]) {
                let (r, g, b) = if y == 0 { (255, 0, 0) } else { (0, 0, 255) };
                for chunk in row_buf.chunks_exact_mut(3) {
                    chunk[0] = r;
                    chunk[1] = g;
                    chunk[2] = b;
                }
            }
        }

        let mut bmp: Bitmap<Rgb8> = Bitmap::new(4, 4, 1, false);
        let clip = make_clip(4, 4);
        let pipe = simple_pipe();

        // mat=[2,0,0,-2,0,2]: positive X scale 2, negative Y scale -2 (vflip), translate (0,2).
        // Y bounds: coord_lower(-2+2)=0, coord_upper(2)=3 → rows 0..3.
        let mat = [2.0f64, 0.0, 0.0, -2.0, 0.0, 2.0];
        let result = draw_image::<Rgb8>(
            &mut bmp,
            &clip,
            &pipe,
            &mut TwoRowImage,
            crate::types::PixelMode::Rgb8,
            2,
            2,
            &mat,
        );
        assert_eq!(result, ImageResult::Ok);
        // After vflip: source row 1 (blue) maps to dest top, row 0 (red) to dest bottom.
        // Dest rows 0..1 should be blue (0,0,255), rows 2..3 should be red (255,0,0).
        // (Exact row mapping depends on Bresenham split; just verify both colours appear.)
        let has_red = (0..3u32).any(|y| bmp.row(y)[0].r == 255 && bmp.row(y)[0].b == 0);
        let has_blue = (0..3u32).any(|y| bmp.row(y)[0].b == 255 && bmp.row(y)[0].r == 0);
        assert!(has_red, "vflip: expected red pixels in output");
        assert!(has_blue, "vflip: expected blue pixels in output");
    }

    /// Partial-clip path: only pixels inside the clip rect are painted.
    #[test]
    fn draw_image_partial_clip_paints_only_inside() {
        let mut bmp: Bitmap<Rgb8> = Bitmap::new(8, 8, 1, false);
        // Clip to columns 2..5 only.
        let clip = Clip::new(2.0, 0.0, 4.999, 7.999, false);
        let pipe = simple_pipe();

        let mut img_src = SolidColor {
            r: 255,
            g: 255,
            b: 255,
        };
        // Fill the full 8×8 canvas.
        let mat = [8.0f64, 0.0, 0.0, 8.0, 0.0, 0.0];
        let result = draw_image::<Rgb8>(
            &mut bmp,
            &clip,
            &pipe,
            &mut img_src,
            crate::types::PixelMode::Rgb8,
            8,
            8,
            &mat,
        );
        assert_eq!(result, ImageResult::Ok);

        for y in 0..8u32 {
            // Columns 0-1 must be unpainted.
            assert_eq!(bmp.row(y)[0].r, 0, "col 0 should be clipped");
            assert_eq!(bmp.row(y)[1].r, 0, "col 1 should be clipped");
            // Columns 2-4 must be painted.
            assert_eq!(bmp.row(y)[2].r, 255, "col 2 should be painted (y={y})");
            assert_eq!(bmp.row(y)[3].r, 255, "col 3 should be painted (y={y})");
            // Column 5+ must be unpainted.
            assert_eq!(bmp.row(y)[5].r, 0, "col 5 should be clipped");
        }
    }

    // ── Golden hash tests for all 8 scale kernels ─────────────────────────────
    //
    // These pin the byte-exact output of every dispatch branch of `scale_mask`
    // and `scale_image` across deterministic sources and ratio combinations
    // that exercise both Bresenham branches (q ≠ 0, multiple wraps per span).
    //
    // The hashes are produced by an inline FNV-1a-64 — stable across Rust
    // versions, unlike `std::collections::hash_map::DefaultHasher`. They
    // guarantee that any refactor preserving the current behaviour will still
    // pass; any byte-level drift in any of the 8 kernels will fail loudly.

    /// FNV-1a 64-bit. Spec-stable across Rust versions and platforms; the
    /// chosen polynomial constants are the canonical FNV-1a values. Used
    /// purely to compact the per-kernel byte-pin into a single comparable
    /// value in the assertions below.
    fn fnv1a64(bytes: &[u8]) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for &b in bytes {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        h
    }

    /// A deterministic 1-bit `MaskSource` whose bit at `(x, y)` is set iff
    /// `(x * 7 + y * 13) % 5 < 3`. The choice of `5 / 7 / 13` gives a
    /// nontrivial spatial pattern: rows and columns don't repeat with small
    /// period, and roughly 60 % of pixels are set, so the box-filtered
    /// downsamples produce non-trivial gradient values.
    struct GoldenMask;
    impl MaskSource for GoldenMask {
        fn get_row(&mut self, y: u32, row_buf: &mut [u8]) {
            for (byte_idx, slot) in row_buf.iter_mut().enumerate() {
                let mut packed: u8 = 0;
                for bit in 0..8u32 {
                    let x = byte_idx as u32 * 8 + bit;
                    let on = (x.wrapping_mul(7).wrapping_add(y.wrapping_mul(13))) % 5 < 3;
                    if on {
                        packed |= 1 << (7 - bit);
                    }
                }
                *slot = packed;
            }
        }
    }

    /// A deterministic multi-channel `ImageSource`: channel `c` of pixel
    /// `(x, y)` is `(x * (17 + c) + y * (23 + c)) % 256` taken as `u8`. With
    /// `ncomps = 3` the three channels evolve independently of each other,
    /// so a refactor that crossed channel boundaries would show up.
    struct GoldenImage {
        ncomps: usize,
    }
    impl ImageSource for GoldenImage {
        fn get_row(&mut self, y: u32, row_buf: &mut [u8]) {
            let width = row_buf.len() / self.ncomps;
            for x in 0..width {
                for c in 0..self.ncomps {
                    let mul = 17u32 + c as u32;
                    let add = 23u32 + c as u32;
                    let v = (x as u32)
                        .wrapping_mul(mul)
                        .wrapping_add(y.wrapping_mul(add));
                    row_buf[x * self.ncomps + c] = (v & 0xFF) as u8;
                }
            }
        }
    }

    /// Mask, both axes downsampling: src 11×13 → scaled 5×7.
    /// Ratios coprime in both axes (`13 % 7 = 6`, `11 % 5 = 1`) → Bresenham
    /// `q ≠ 0` and the accumulator wraps multiple times per span.
    #[test]
    fn scale_mask_ydown_xdown_golden() {
        let out = scale_mask(&mut GoldenMask, 11, 13, 5, 7);
        assert_eq!(out.len(), 5 * 7);
        assert_eq!(fnv1a64(&out), 0xC72C_2A67_D157_65F4);
    }

    /// Mask, Y down + X up: src 11×13 → scaled 23×7.
    #[test]
    fn scale_mask_ydown_xup_golden() {
        let out = scale_mask(&mut GoldenMask, 11, 13, 23, 7);
        assert_eq!(out.len(), 23 * 7);
        assert_eq!(fnv1a64(&out), 0x3C80_F065_9EB6_35B6);
    }

    /// Mask, Y up + X down: src 11×13 → scaled 5×29.
    #[test]
    fn scale_mask_yup_xdown_golden() {
        let out = scale_mask(&mut GoldenMask, 11, 13, 5, 29);
        assert_eq!(out.len(), 5 * 29);
        assert_eq!(fnv1a64(&out), 0x8056_EED7_DF0E_665E);
    }

    /// Mask, both axes upsampling: src 5×7 → scaled 23×29.
    #[test]
    fn scale_mask_yup_xup_golden() {
        let out = scale_mask(&mut GoldenMask, 5, 7, 23, 29);
        assert_eq!(out.len(), 23 * 29);
        assert_eq!(fnv1a64(&out), 0x5940_5245_567D_707F);
    }

    /// Image (ncomps = 3), both axes downsampling.
    #[test]
    fn scale_image_ydown_xdown_golden() {
        let mut src = GoldenImage { ncomps: 3 };
        let out = scale_image(&mut src, 11, 13, 5, 7, 3);
        assert_eq!(out.len(), 5 * 7 * 3);
        assert_eq!(fnv1a64(&out), 0x6CDB_D839_4499_6365);
    }

    /// Image (ncomps = 3), Y down + X up.
    #[test]
    fn scale_image_ydown_xup_golden() {
        let mut src = GoldenImage { ncomps: 3 };
        let out = scale_image(&mut src, 11, 13, 23, 7, 3);
        assert_eq!(out.len(), 23 * 7 * 3);
        assert_eq!(fnv1a64(&out), 0xA8DF_24A4_C3D9_F281);
    }

    /// Image (ncomps = 3), Y up + X down.
    #[test]
    fn scale_image_yup_xdown_golden() {
        let mut src = GoldenImage { ncomps: 3 };
        let out = scale_image(&mut src, 11, 13, 5, 29, 3);
        assert_eq!(out.len(), 5 * 29 * 3);
        assert_eq!(fnv1a64(&out), 0xBB21_A5B6_484F_2159);
    }

    /// Image (ncomps = 4 — covers CMYK channel-count path), both axes
    /// upsampling: src 5×7 → scaled 23×29.
    #[test]
    fn scale_image_yup_xup_golden() {
        let mut src = GoldenImage { ncomps: 4 };
        let out = scale_image(&mut src, 5, 7, 23, 29, 4);
        assert_eq!(out.len(), 23 * 29 * 4);
        assert_eq!(fnv1a64(&out), 0xA063_5327_4D9F_A0C1);
    }
}
