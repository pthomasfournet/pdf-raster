//! Image and image-mask rendering — replaces `Splash::fillImageMask`,
//! `Splash::drawImage`, and the four `scaleImage*` / `scaleMask*` helpers.
//!
//! # Phase 1 scope
//!
//! Only axis-aligned scaling (mat\[1\] == 0 && mat\[2\] == 0) with optional
//! vertical flip is implemented.  Arbitrary-transform paths return
//! [`ImageResult::ArbitraryTransformSkipped`] so that callers can fall back
//! to a different implementation or log the fact.
//!
//! # Scaling strategy
//!
//! Four cases exactly mirroring the C++ `scaleMask*` / `scaleImage*` family:
//!
//! | Y   | X   | Method            |
//! |-----|-----|-------------------|
//! | ↓   | ↓   | Bresenham box-filter in both axes (average) |
//! | ↓   | ↑   | Bresenham box-filter in Y, nearest in X     |
//! | ↑   | ↓   | Nearest in Y, Bresenham box-filter in X     |
//! | ↑   | ↑   | Nearest (nearest-neighbor) in both axes     |
//!
//! # C++ equivalents
//!
//! - `Splash::fillImageMask` (~line 2765)
//! - `Splash::drawImage` (~line 3528)
//! - `Splash::scaleMask{YdownXdown,YdownXup,YupXdown,YupXup}` (~lines 3128–3472)
//! - `Splash::scaleImage{YdownXdown,YdownXup,YupXdown,YupXup}` (~lines 4033–4782)
//! - `Splash::blitMask` (~line 3475)

use crate::bitmap::Bitmap;
use crate::clip::{Clip, ClipResult};
use crate::pipe::{self, PipeSrc, PipeState};
use crate::types::PixelMode;
use color::convert::splash_floor;
use color::Pixel;

// ── Public traits ─────────────────────────────────────────────────────────────

/// Caller-supplied source for a colour image: one row at a time.
///
/// Matches the `SplashImageSource` callback convention from `splash/SplashTypes.h`.
pub trait ImageSource: Send {
    /// Fill `row_buf` with pixel data for source row `y`.
    ///
    /// `row_buf.len()` must equal `src_width * ncomps`.
    fn get_row(&mut self, y: u32, row_buf: &mut [u8]);

    /// Optionally fill `alpha_buf` with per-pixel alpha for source row `y`.
    ///
    /// Returns `true` if alpha was written, `false` if the image is fully opaque.
    /// `alpha_buf.len()` must equal `src_width`.
    fn src_alpha(&mut self, _y: u32, _alpha_buf: &mut [u8]) -> bool {
        false
    }
}

/// Caller-supplied source for a 1-bit image mask: one row at a time.
///
/// Each row is delivered as MSB-first packed bytes: `ceil(src_width/8)` bytes
/// per row.  Bit 7 of the first byte is the leftmost pixel.
///
/// Internally the mask is immediately unpacked to one `u8` per pixel (0 or
/// 255) for simpler scaling arithmetic, matching the C++ `scaleMask` convention
/// where the callback produces one byte per pixel.
///
/// Matches `SplashImageMaskSource` from `splash/SplashTypes.h`.
pub trait MaskSource: Send {
    /// Fill `row_buf` with 1-bit packed mono mask data (MSB-first).
    ///
    /// `row_buf.len()` must equal `ceil(src_width / 8)`.
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
    #[expect(clippy::suboptimal_flops, reason = "matches the C++ arithmetic exactly; no numerics benefit here")]
    { (a * d - b * c).abs() >= eps }
}

/// Unpack one packed-bits mask row into one-byte-per-pixel form (0 or 255).
///
/// `packed` is MSB-first; `out` receives exactly `width` bytes.
fn unpack_mask_row(packed: &[u8], width: usize, out: &mut [u8]) {
    for (i, slot) in out.iter_mut().enumerate().take(width) {
        let byte = packed.get(i / 8).copied().unwrap_or(0);
        let bit = (byte >> (7 - (i % 8))) & 1;
        *slot = if bit != 0 { 255 } else { 0 };
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

// ── Mask scaling ──────────────────────────────────────────────────────────────

/// Scale a 1-bit mask to `scaled_w × scaled_h`, producing one `u8` (0–255) per
/// destination pixel.  Exactly mirrors `Splash::scaleMask`.
fn scale_mask(
    mask_src: &mut dyn MaskSource,
    src_w: usize,
    src_h: usize,
    scaled_w: usize,
    scaled_h: usize,
) -> Vec<u8> {
    let mut dest = vec![0u8; scaled_w * scaled_h];
    let packed_row_bytes = src_w.div_ceil(8);
    let mut packed_buf = vec![0u8; packed_row_bytes];
    let mut line_buf = vec![0u8; src_w];

    if scaled_h < src_h {
        if scaled_w < src_w {
            scale_mask_ydown_xdown(
                mask_src, src_w, src_h, scaled_w, scaled_h,
                &mut dest, &mut packed_buf, &mut line_buf,
            );
        } else {
            scale_mask_ydown_xup(
                mask_src, src_w, src_h, scaled_w, scaled_h,
                &mut dest, &mut packed_buf, &mut line_buf,
            );
        }
    } else if scaled_w < src_w {
        scale_mask_yup_xdown(
            mask_src, src_w, src_h, scaled_w, scaled_h,
            &mut dest, &mut packed_buf, &mut line_buf,
        );
    } else {
        scale_mask_yup_xup(
            mask_src, src_w, src_h, scaled_w, scaled_h,
            &mut dest, &mut packed_buf, &mut line_buf,
        );
    }

    dest
}

/// Vertical and horizontal box-filter downsampling.
///
/// C++: `Splash::scaleMaskYdownXdown`.
#[expect(clippy::too_many_arguments, reason = "mirrors C++ scaleMaskYdownXdown; all buffers are necessary")]
fn scale_mask_ydown_xdown(
    mask_src: &mut dyn MaskSource,
    src_w: usize,
    src_h: usize,
    scaled_w: usize,
    scaled_h: usize,
    dest: &mut [u8],
    packed_buf: &mut [u8],
    line_buf: &mut [u8],
) {
    let yp = src_h / scaled_h;
    let yq = src_h % scaled_h;
    let xp = src_w / scaled_w;
    let xq = src_w % scaled_w;

    let mut pix_buf = vec![0u32; src_w];
    let mut yt = 0usize;
    let mut dest_off = 0usize;
    let mut src_y = 0u32;

    for _dy in 0..scaled_h {
        let y_step = bresenham_step(&mut yt, yq, scaled_h, yp);

        pix_buf.fill(0);
        for _i in 0..y_step {
            mask_src.get_row(src_y, packed_buf);
            src_y += 1;
            unpack_mask_row(packed_buf, src_w, line_buf);
            for (pix, &lb) in pix_buf.iter_mut().zip(line_buf.iter()) {
                *pix += u32::from(lb);
            }
        }

        let d0 = if xp > 0 {
            #[expect(clippy::cast_possible_truncation, reason = "y_step*xp ≤ src_h*src_w; fits u32 for practical image sizes")]
            { (255u32 << 23) / (y_step * xp) as u32 }
        } else { 0 };
        #[expect(clippy::cast_possible_truncation, reason = "y_step*(xp+1) ≤ src_h*(src_w+1); fits u32 for practical image sizes")]
        let d1 = (255u32 << 23) / (y_step * (xp + 1)) as u32;

        let mut xt = 0usize;
        let mut xx = 0usize;
        for _dx in 0..scaled_w {
            let (x_step, d) = {
                let step = bresenham_step(&mut xt, xq, scaled_w, xp);
                if step == xp + 1 { (step, d1) } else { (step, d0) }
            };

            let pix = pix_buf[xx..xx + x_step].iter().sum::<u32>();
            xx += x_step;
            dest[dest_off] = ((pix * d) >> 23).min(255) as u8;
            dest_off += 1;
        }
    }
}

/// Box-filter Y downsampling, nearest-neighbor X upsampling.
///
/// C++: `Splash::scaleMaskYdownXup`.
#[expect(clippy::too_many_arguments, reason = "mirrors C++ scaleMaskYdownXup; all buffers are necessary")]
fn scale_mask_ydown_xup(
    mask_src: &mut dyn MaskSource,
    src_w: usize,
    src_h: usize,
    scaled_w: usize,
    scaled_h: usize,
    dest: &mut [u8],
    packed_buf: &mut [u8],
    line_buf: &mut [u8],
) {
    let yp = src_h / scaled_h;
    let yq = src_h % scaled_h;
    let xp = scaled_w / src_w;
    let xq = scaled_w % src_w;

    let mut pix_buf = vec![0u32; src_w];
    let mut yt = 0usize;
    let mut dest_off = 0usize;
    let mut src_y = 0u32;

    for _dy in 0..scaled_h {
        let y_step = bresenham_step(&mut yt, yq, scaled_h, yp);

        pix_buf.fill(0);
        for _i in 0..y_step {
            mask_src.get_row(src_y, packed_buf);
            src_y += 1;
            unpack_mask_row(packed_buf, src_w, line_buf);
            for (pix, &lb) in pix_buf.iter_mut().zip(line_buf.iter()) {
                *pix += u32::from(lb);
            }
        }

        #[expect(clippy::cast_possible_truncation, reason = "y_step ≤ src_h; fits u32 for practical image sizes")]
        let d = (255u32 << 23) / y_step as u32;
        let mut xt = 0usize;

        for pix_val in pix_buf.iter().take(src_w) {
            let x_step = bresenham_step(&mut xt, xq, src_w, xp);
            let pix = ((pix_val * d) >> 23).min(255) as u8;
            for slot in &mut dest[dest_off..dest_off + x_step] {
                *slot = pix;
            }
            dest_off += x_step;
        }
    }
}

/// Nearest-neighbor Y upsampling, box-filter X downsampling.
///
/// C++: `Splash::scaleMaskYupXdown`.
#[expect(clippy::too_many_arguments, reason = "mirrors C++ scaleMaskYupXdown; all buffers are necessary")]
fn scale_mask_yup_xdown(
    mask_src: &mut dyn MaskSource,
    src_w: usize,
    src_h: usize,
    scaled_w: usize,
    scaled_h: usize,
    dest: &mut [u8],
    packed_buf: &mut [u8],
    line_buf: &mut [u8],
) {
    let yp = scaled_h / src_h;
    let yq = scaled_h % src_h;
    let xp = src_w / scaled_w;
    let xq = src_w % scaled_w;

    let d0 = if xp > 0 {
        #[expect(clippy::cast_possible_truncation, reason = "xp ≤ src_w; fits u32 for practical image sizes")]
        { (255u32 << 23) / xp as u32 }
    } else { 0 };
    #[expect(clippy::cast_possible_truncation, reason = "xp+1 ≤ src_w+1; fits u32 for practical image sizes")]
    let d1 = (255u32 << 23) / (xp + 1) as u32;

    let mut yt = 0usize;
    let mut dest_off = 0usize;

    for sy in 0..src_h {
        let y_step = bresenham_step(&mut yt, yq, src_h, yp);

        #[expect(clippy::cast_possible_truncation, reason = "sy ≤ src_h ≤ u32::MAX for practical image sizes")]
        mask_src.get_row(sy as u32, packed_buf);
        unpack_mask_row(packed_buf, src_w, line_buf);

        let row_start = dest_off;
        let mut xt = 0usize;
        let mut xx = 0usize;

        for dx in 0..scaled_w {
            let (x_step, d) = {
                let step = bresenham_step(&mut xt, xq, scaled_w, xp);
                if step == xp + 1 { (step, d1) } else { (step, d0) }
            };

            let pix = line_buf[xx..xx + x_step].iter().map(|&b| u32::from(b)).sum::<u32>();
            xx += x_step;
            dest[row_start + dx] = ((pix * d) >> 23).min(255) as u8;
        }
        dest_off += scaled_w;

        for i in 1..y_step {
            dest.copy_within(row_start..row_start + scaled_w, row_start + i * scaled_w);
        }
        dest_off += (y_step - 1) * scaled_w;
    }
}

/// Nearest-neighbor upsampling in both axes.
///
/// C++: `Splash::scaleMaskYupXup`.
#[expect(clippy::too_many_arguments, reason = "mirrors C++ scaleMaskYupXup; all buffers are necessary")]
fn scale_mask_yup_xup(
    mask_src: &mut dyn MaskSource,
    src_w: usize,
    src_h: usize,
    scaled_w: usize,
    scaled_h: usize,
    dest: &mut [u8],
    packed_buf: &mut [u8],
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

        #[expect(clippy::cast_possible_truncation, reason = "sy ≤ src_h ≤ u32::MAX for practical image sizes")]
        mask_src.get_row(sy as u32, packed_buf);
        unpack_mask_row(packed_buf, src_w, line_buf);

        let row_start = dest_off;
        let mut xt = 0usize;
        let mut xx = 0usize;

        for &lb in line_buf.iter().take(src_w) {
            let x_step = bresenham_step(&mut xt, xq, src_w, xp);
            let pix = if lb != 0 { 255u8 } else { 0u8 };
            for slot in &mut dest[row_start + xx..row_start + xx + x_step] {
                *slot = pix;
            }
            xx += x_step;
        }
        dest_off += scaled_w;

        for i in 1..y_step {
            dest.copy_within(row_start..row_start + scaled_w, row_start + i * scaled_w);
        }
        dest_off += (y_step - 1) * scaled_w;
    }
}

// ── Image scaling ─────────────────────────────────────────────────────────────

/// Scale a colour image to `scaled_w × scaled_h`.
/// Output: `scaled_w * scaled_h * ncomps` bytes (row-major, no padding).
fn scale_image(
    image_src: &mut dyn ImageSource,
    src_w: usize,
    src_h: usize,
    scaled_w: usize,
    scaled_h: usize,
    ncomps: usize,
) -> Vec<u8> {
    let mut dest = vec![0u8; scaled_w * scaled_h * ncomps];
    let mut line_buf = vec![0u8; src_w * ncomps];

    if scaled_h < src_h {
        if scaled_w < src_w {
            scale_image_ydown_xdown(
                image_src, src_w, src_h, scaled_w, scaled_h, ncomps,
                &mut dest, &mut line_buf,
            );
        } else {
            scale_image_ydown_xup(
                image_src, src_w, src_h, scaled_w, scaled_h, ncomps,
                &mut dest, &mut line_buf,
            );
        }
    } else if scaled_w < src_w {
        scale_image_yup_xdown(
            image_src, src_w, src_h, scaled_w, scaled_h, ncomps,
            &mut dest, &mut line_buf,
        );
    } else {
        scale_image_yup_xup(
            image_src, src_w, src_h, scaled_w, scaled_h, ncomps,
            &mut dest, &mut line_buf,
        );
    }

    dest
}

/// Box-filter downsampling in both Y and X.
///
/// C++: `Splash::scaleImageYdownXdown`.
#[expect(clippy::too_many_arguments, reason = "mirrors C++ scaleImageYdownXdown; all buffers are necessary")]
fn scale_image_ydown_xdown(
    image_src: &mut dyn ImageSource,
    src_w: usize,
    src_h: usize,
    scaled_w: usize,
    scaled_h: usize,
    ncomps: usize,
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
        for _i in 0..y_step {
            image_src.get_row(src_y, line_buf);
            src_y += 1;
            for (pix, &lb) in pix_buf.iter_mut().zip(line_buf.iter()) {
                *pix += u32::from(lb);
            }
        }

        let d0 = if xp > 0 {
            #[expect(clippy::cast_possible_truncation, reason = "y_step*xp fits u32 for practical image sizes")]
            { (1u32 << 23) / (y_step * xp) as u32 }
        } else { 0 };
        #[expect(clippy::cast_possible_truncation, reason = "y_step*(xp+1) fits u32 for practical image sizes")]
        let d1 = (1u32 << 23) / (y_step * (xp + 1)) as u32;

        let mut xt = 0usize;
        let mut xx = 0usize;
        for _dx in 0..scaled_w {
            let (x_step, d) = {
                let step = bresenham_step(&mut xt, xq, scaled_w, xp);
                if step == xp + 1 { (step, d1) } else { (step, d0) }
            };

            for c in 0..ncomps {
                let pix: u32 = (0..x_step).map(|i| pix_buf[(xx + i) * ncomps + c]).sum();
                dest[dest_off + c] = ((pix * d) >> 23).min(255) as u8;
            }
            xx += x_step;
            dest_off += ncomps;
        }
    }
}

/// Box-filter Y downsampling, nearest-neighbor X upsampling.
///
/// C++: `Splash::scaleImageYdownXup`.
#[expect(clippy::too_many_arguments, reason = "mirrors C++ scaleImageYdownXup; all buffers are necessary")]
fn scale_image_ydown_xup(
    image_src: &mut dyn ImageSource,
    src_w: usize,
    src_h: usize,
    scaled_w: usize,
    scaled_h: usize,
    ncomps: usize,
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
        for _i in 0..y_step {
            image_src.get_row(src_y, line_buf);
            src_y += 1;
            for (pix, &lb) in pix_buf.iter_mut().zip(line_buf.iter()) {
                *pix += u32::from(lb);
            }
        }

        #[expect(clippy::cast_possible_truncation, reason = "y_step ≤ src_h; fits u32 for practical image sizes")]
        let d = (1u32 << 23) / y_step as u32;
        let mut xt = 0usize;

        for sx in 0..src_w {
            let x_step = bresenham_step(&mut xt, xq, src_w, xp);
            let base = sx * ncomps;
            // Max ncomps we expect is 8 (DeviceN8); use a fixed-size scratch.
            let mut pix_vals = [0u8; 8];
            for c in 0..ncomps.min(8) {
                pix_vals[c] = ((pix_buf[base + c] * d) >> 23).min(255) as u8;
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
/// C++: `Splash::scaleImageYupXdown`.
#[expect(clippy::too_many_arguments, reason = "mirrors C++ scaleImageYupXdown; all buffers are necessary")]
fn scale_image_yup_xdown(
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
    let xp = src_w / scaled_w;
    let xq = src_w % scaled_w;

    let d0 = if xp > 0 {
        #[expect(clippy::cast_possible_truncation, reason = "xp ≤ src_w; fits u32 for practical image sizes")]
        { (1u32 << 23) / xp as u32 }
    } else { 0 };
    #[expect(clippy::cast_possible_truncation, reason = "xp+1 ≤ src_w+1; fits u32 for practical image sizes")]
    let d1 = (1u32 << 23) / (xp + 1) as u32;

    let mut yt = 0usize;
    let mut dest_off = 0usize;

    for sy in 0..src_h {
        let y_step = bresenham_step(&mut yt, yq, src_h, yp);

        #[expect(clippy::cast_possible_truncation, reason = "sy ≤ src_h ≤ u32::MAX for practical image sizes")]
        image_src.get_row(sy as u32, line_buf);

        let row_start = dest_off;
        let mut xt = 0usize;
        let mut xx = 0usize;

        for dx in 0..scaled_w {
            let (x_step, d) = {
                let step = bresenham_step(&mut xt, xq, scaled_w, xp);
                if step == xp + 1 { (step, d1) } else { (step, d0) }
            };

            for c in 0..ncomps {
                let pix: u32 = (0..x_step).map(|i| u32::from(line_buf[(xx + i) * ncomps + c])).sum();
                dest[row_start + dx * ncomps + c] = ((pix * d) >> 23).min(255) as u8;
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
/// C++: `Splash::scaleImageYupXup`.
#[expect(clippy::too_many_arguments, reason = "mirrors C++ scaleImageYupXup; all buffers are necessary")]
fn scale_image_yup_xup(
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

        #[expect(clippy::cast_possible_truncation, reason = "sy ≤ src_h ≤ u32::MAX for practical image sizes")]
        image_src.get_row(sy as u32, line_buf);

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

// ── Row-as-pattern helper ─────────────────────────────────────────────────────

/// A `Pattern` that serves a pre-computed image row via `fill_span`.
///
/// The `data` slice must cover exactly `(x1 - x0 + 1) * ncomps` bytes as
/// passed from the outer blit loop.
struct RowPattern<'a> {
    data: &'a [u8],
}

impl crate::pipe::Pattern for RowPattern<'_> {
    fn fill_span(&self, _y: i32, _x0: i32, _x1: i32, out: &mut [u8]) {
        let len = out.len().min(self.data.len());
        out[..len].copy_from_slice(&self.data[..len]);
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
#[expect(clippy::too_many_arguments, reason = "mirrors Splash::blitMask API; all params necessary")]
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
    #[expect(clippy::cast_possible_wrap, reason = "bitmap dims ≤ i32::MAX in practice")]
    let bmp_w = bitmap.width as i32;
    #[expect(clippy::cast_possible_wrap, reason = "bitmap dims ≤ i32::MAX in practice")]
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
                    pipe::render_span::<P>(pipe, src, dst_pixels, dst_alpha, Some(&run_shape), rs, rx1, y);
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

/// Blit a pre-scaled colour image onto the bitmap.
///
/// The entire row (or per-pixel runs for partial clip) is emitted through
/// `render_span`.  Mirrors `Splash::blitImage` non-AA path.
#[expect(clippy::too_many_arguments, reason = "mirrors Splash::blitImage API; all params necessary")]
fn blit_image<P: Pixel>(
    bitmap: &mut Bitmap<P>,
    clip: &Clip,
    pipe: &PipeState<'_>,
    scaled_img: &[u8],
    scaled_w: i32,
    scaled_h: i32,
    x_dest: i32,
    y_dest: i32,
    ncomps: usize,
    clip_res: ClipResult,
) {
    #[expect(clippy::cast_possible_wrap, reason = "bitmap dims ≤ i32::MAX in practice")]
    let bmp_w = bitmap.width as i32;
    #[expect(clippy::cast_possible_wrap, reason = "bitmap dims ≤ i32::MAX in practice")]
    let bmp_h = bitmap.height as i32;

    let clip_all_inside = clip_res == ClipResult::AllInside;

    for dy in 0..scaled_h {
        let y = y_dest + dy;
        if y < 0 || y >= bmp_h {
            continue;
        }
        #[expect(clippy::cast_sign_loss, reason = "y ≥ 0 by guard above")]
        let y_u = y as u32;
        #[expect(clippy::cast_sign_loss, reason = "dy ≥ 0")]
        let img_row_off = dy as usize * scaled_w as usize * ncomps;

        let x_lo = x_dest.max(0);
        let x_hi = (x_dest + scaled_w - 1).min(bmp_w - 1);
        if x_lo > x_hi {
            continue;
        }

        if clip_all_inside {
            #[expect(clippy::cast_sign_loss, reason = "x_lo ≥ 0 (clamped)")]
            let img_x_off = (x_lo - x_dest) as usize * ncomps;
            #[expect(clippy::cast_sign_loss, reason = "x_hi - x_lo + 1 ≥ 1")]
            let count = (x_hi - x_lo + 1) as usize;
            let img_slice = &scaled_img[img_row_off + img_x_off..img_row_off + img_x_off + count * ncomps];
            let row_src = PipeSrc::Pattern(&RowPattern { data: img_slice });
            #[expect(clippy::cast_sign_loss, reason = "x_lo ≥ 0")]
            let byte_off = x_lo as usize * P::BYTES;
            #[expect(clippy::cast_sign_loss, reason = "x_hi ≥ x_lo ≥ 0")]
            let byte_end = (x_hi as usize + 1) * P::BYTES;
            #[expect(clippy::cast_sign_loss, reason = "x_lo ≥ 0")]
            let alpha_lo = x_lo as usize;
            #[expect(clippy::cast_sign_loss, reason = "x_hi ≥ x_lo ≥ 0")]
            let alpha_hi = x_hi as usize;
            let (row, alpha) = bitmap.row_and_alpha_mut(y_u);
            let dst_pixels = &mut row[byte_off..byte_end];
            let dst_alpha = alpha.map(|a| &mut a[alpha_lo..=alpha_hi]);
            pipe::render_span::<P>(pipe, &row_src, dst_pixels, dst_alpha, None, x_lo, x_hi, y);
        } else {
            // Partial clip: collect contiguous clip-passing spans.
            let mut run_start: Option<i32> = None;
            let mut run_end = 0i32;

            macro_rules! flush_img_run {
                () => {
                    if let Some(rs) = run_start.take() {
                        #[expect(clippy::cast_sign_loss, reason = "rs ≥ x_dest ≥ 0 inside bmp")]
                        let img_x_off = (rs - x_dest) as usize * ncomps;
                        #[expect(clippy::cast_sign_loss, reason = "run_end ≥ rs ≥ 0")]
                        let count = (run_end - rs + 1) as usize;
                        let img_slice = &scaled_img
                            [img_row_off + img_x_off..img_row_off + img_x_off + count * ncomps];
                        let row_src = PipeSrc::Pattern(&RowPattern { data: img_slice });
                        #[expect(clippy::cast_sign_loss, reason = "rs ≥ 0")]
                        let byte_off = rs as usize * P::BYTES;
                        #[expect(clippy::cast_sign_loss, reason = "run_end ≥ rs ≥ 0")]
                        let byte_end = (run_end as usize + 1) * P::BYTES;
                        #[expect(clippy::cast_sign_loss, reason = "rs ≥ 0")]
                        let alpha_lo = rs as usize;
                        #[expect(clippy::cast_sign_loss, reason = "run_end ≥ rs ≥ 0")]
                        let alpha_hi = run_end as usize;
                        let (row, alpha) = bitmap.row_and_alpha_mut(y_u);
                        let dst_pixels = &mut row[byte_off..byte_end];
                        let dst_alpha = alpha.map(|a| &mut a[alpha_lo..=alpha_hi]);
                        pipe::render_span::<P>(
                            pipe, &row_src, dst_pixels, dst_alpha, None, rs, run_end, y,
                        );
                    }
                };
            }

            for dx in 0..scaled_w {
                let x = x_dest + dx;
                if x < 0 || x >= bmp_w || !clip.test(x, y) {
                    flush_img_run!();
                    continue;
                }
                if run_start.is_none() {
                    run_start = Some(x);
                }
                run_end = x;
            }
            flush_img_run!();
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Public entry points
// ─────────────────────────────────────────────────────────────────────────────

/// Fill a 1-bit image mask using the current fill pattern.
///
/// Only the axis-aligned scaling cases (mat\[1\]==0 && mat\[2\]==0, mat\[0\]>0)
/// are implemented.  For all other matrices
/// [`ImageResult::ArbitraryTransformSkipped`] is returned.
///
/// # C++ equivalent
///
/// `Splash::fillImageMask` (~line 2765).
#[expect(clippy::too_many_arguments, reason = "mirrors Splash::fillImageMask API; all params necessary")]
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
    if !check_det(matrix[0], matrix[1], matrix[2], matrix[3], 1e-6) {
        return ImageResult::SingularMatrix;
    }

    let minor_axis_zero = matrix[1] == 0.0 && matrix[2] == 0.0;

    let (x0, y0, x1, y1, vflip) = if matrix[0] > 0.0 && minor_axis_zero && matrix[3] > 0.0 {
        let x0 = coord_lower(matrix[4]);
        let y0 = coord_lower(matrix[5]);
        let x1 = coord_upper(matrix[0] + matrix[4]);
        let y1 = coord_upper(matrix[3] + matrix[5]);
        (x0, y0, x1, y1, false)
    } else if matrix[0] > 0.0 && minor_axis_zero && matrix[3] < 0.0 {
        let x0 = coord_lower(matrix[4]);
        let y0 = coord_lower(matrix[3] + matrix[5]);
        let x1 = coord_upper(matrix[0] + matrix[4]);
        let y1 = coord_upper(matrix[5]);
        (x0, y0, x1, y1, true)
    } else {
        return ImageResult::ArbitraryTransformSkipped;
    };

    let x1 = if x0 == x1 { x1 + 1 } else { x1 };
    let y1 = if y0 == y1 { y1 + 1 } else { y1 };

    let clip_res = clip.test_rect(x0, y0, x1 - 1, y1 - 1);
    if clip_res == ClipResult::AllOutside {
        return ImageResult::Ok;
    }

    #[expect(clippy::cast_sign_loss, reason = "x1 > x0 enforced above")]
    let scaled_w = (x1 - x0) as usize;
    #[expect(clippy::cast_sign_loss, reason = "y1 > y0 enforced above")]
    let scaled_h = (y1 - y0) as usize;

    let mut scaled = scale_mask(mask_src, src_w as usize, src_h as usize, scaled_w, scaled_h);

    if vflip {
        let mut lo = 0usize;
        let mut hi = scaled_h.saturating_sub(1);
        while lo < hi {
            for c in 0..scaled_w {
                scaled.swap(lo * scaled_w + c, hi * scaled_w + c);
            }
            lo += 1;
            hi -= 1;
        }
    }

    blit_mask::<P>(
        bitmap,
        clip,
        pipe,
        src,
        &scaled,
        #[expect(clippy::cast_possible_truncation, clippy::cast_possible_wrap, reason = "scaled_w ≤ bitmap.width ≤ i32::MAX")]
        { scaled_w as i32 },
        #[expect(clippy::cast_possible_truncation, clippy::cast_possible_wrap, reason = "scaled_h ≤ bitmap.height ≤ i32::MAX")]
        { scaled_h as i32 },
        x0,
        y0,
        clip_res == ClipResult::AllInside,
    );

    ImageResult::Ok
}

/// Render a colour image with transformation.
///
/// Only axis-aligned scaling cases are implemented (Phase 1).  For all other
/// matrices [`ImageResult::ArbitraryTransformSkipped`] is returned.
///
/// `_src_mode` is accepted for API symmetry with the C++ `drawImage` but is
/// not used in Phase 1 (the caller controls `ncomps`).
///
/// # C++ equivalent
///
/// `Splash::drawImage` (~line 3528).
#[expect(clippy::too_many_arguments, reason = "mirrors Splash::drawImage API; all params necessary")]
pub fn draw_image<P: Pixel>(
    bitmap: &mut Bitmap<P>,
    clip: &Clip,
    pipe: &PipeState<'_>,
    image_src: &mut dyn ImageSource,
    _src_mode: PixelMode,
    src_w: u32,
    src_h: u32,
    matrix: &[f64; 6],
    ncomps: usize,
) -> ImageResult {
    if src_w == 0 || src_h == 0 {
        return ImageResult::ZeroImage;
    }
    if !check_det(matrix[0], matrix[1], matrix[2], matrix[3], 1e-6) {
        return ImageResult::SingularMatrix;
    }

    let minor_axis_zero = matrix[1] == 0.0 && matrix[2] == 0.0;

    let (x0, y0, x1, y1, vflip) = if matrix[0] > 0.0 && minor_axis_zero && matrix[3] > 0.0 {
        let x0 = coord_lower(matrix[4]);
        let y0 = coord_lower(matrix[5]);
        let x1 = coord_upper(matrix[0] + matrix[4]);
        let y1 = coord_upper(matrix[3] + matrix[5]);
        (x0, y0, x1, y1, false)
    } else if matrix[0] > 0.0 && minor_axis_zero && matrix[3] < 0.0 {
        let x0 = coord_lower(matrix[4]);
        let y0 = coord_lower(matrix[3] + matrix[5]);
        let x1 = coord_upper(matrix[0] + matrix[4]);
        let y1 = coord_upper(matrix[5]);
        (x0, y0, x1, y1, true)
    } else {
        return ImageResult::ArbitraryTransformSkipped;
    };

    let x1 = if x0 == x1 { x1 + 1 } else { x1 };
    let y1 = if y0 == y1 { y1 + 1 } else { y1 };

    let clip_res = clip.test_rect(x0, y0, x1 - 1, y1 - 1);
    if clip_res == ClipResult::AllOutside {
        return ImageResult::Ok;
    }

    #[expect(clippy::cast_sign_loss, reason = "x1 > x0 enforced above")]
    let scaled_w = (x1 - x0) as usize;
    #[expect(clippy::cast_sign_loss, reason = "y1 > y0 enforced above")]
    let scaled_h = (y1 - y0) as usize;

    let mut scaled = scale_image(image_src, src_w as usize, src_h as usize, scaled_w, scaled_h, ncomps);

    if vflip {
        let row_bytes = scaled_w * ncomps;
        let mut lo = 0usize;
        let mut hi = scaled_h.saturating_sub(1);
        while lo < hi {
            for c in 0..row_bytes {
                scaled.swap(lo * row_bytes + c, hi * row_bytes + c);
            }
            lo += 1;
            hi -= 1;
        }
    }

    blit_image::<P>(
        bitmap,
        clip,
        pipe,
        &scaled,
        #[expect(clippy::cast_possible_truncation, clippy::cast_possible_wrap, reason = "scaled_w ≤ bitmap.width ≤ i32::MAX")]
        { scaled_w as i32 },
        #[expect(clippy::cast_possible_truncation, clippy::cast_possible_wrap, reason = "scaled_h ≤ bitmap.height ≤ i32::MAX")]
        { scaled_h as i32 },
        x0,
        y0,
        ncomps,
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
    use crate::pipe::{PipeSrc, PipeState};
    use crate::state::TransferSet;
    use crate::types::BlendMode;
    use color::Rgb8;

    // ── Helpers ──────────────────────────────────────────────────────────────

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

    fn full_clip(w: u32, h: u32) -> Clip {
        Clip::new(0.0, 0.0, w as f64 - 0.001, h as f64 - 0.001, false)
    }

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
        let clip = full_clip(8, 8);
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
        let clip = full_clip(8, 8);
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
        let clip = full_clip(8, 8);
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
        let clip = full_clip(8, 8);
        let pipe = simple_pipe();

        let mut img_src = SolidColor { r: 0, g: 0, b: 200 };
        let mat = [4.0f64, 0.0, 0.0, 4.0, 0.0, 0.0];
        let result = draw_image::<Rgb8>(
            &mut bmp, &clip, &pipe, &mut img_src,
            crate::types::PixelMode::Rgb8, 4, 4, &mat, 3,
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
        let clip = full_clip(8, 8);
        let pipe = simple_pipe();
        let mut img_src = SolidColor { r: 100, g: 100, b: 100 };

        let mat = [2.0f64, 1.0, 1.0, 2.0, 0.0, 0.0];
        let result = draw_image::<Rgb8>(
            &mut bmp, &clip, &pipe, &mut img_src,
            crate::types::PixelMode::Rgb8, 4, 4, &mat, 3,
        );
        assert_eq!(result, ImageResult::ArbitraryTransformSkipped);
    }

    /// Zero-size image → `ZeroImage`.
    #[test]
    fn draw_image_zero_size() {
        let mut bmp: Bitmap<Rgb8> = Bitmap::new(8, 8, 1, false);
        let clip = full_clip(8, 8);
        let pipe = simple_pipe();
        let mut img_src = SolidColor { r: 1, g: 2, b: 3 };

        let mat = [4.0f64, 0.0, 0.0, 4.0, 0.0, 0.0];
        let result = draw_image::<Rgb8>(
            &mut bmp, &clip, &pipe, &mut img_src,
            crate::types::PixelMode::Rgb8, 0, 4, &mat, 3,
        );
        assert_eq!(result, ImageResult::ZeroImage);
    }

    /// Upsampling: 2×2 source → 4×4 destination.
    #[test]
    fn draw_image_upsample_2x2_to_4x4() {
        let mut bmp: Bitmap<Rgb8> = Bitmap::new(8, 8, 1, false);
        let clip = full_clip(8, 8);
        let pipe = simple_pipe();

        let mut img_src = SolidColor { r: 128, g: 64, b: 32 };
        let mat = [4.0f64, 0.0, 0.0, 4.0, 0.0, 0.0];
        let result = draw_image::<Rgb8>(
            &mut bmp, &clip, &pipe, &mut img_src,
            crate::types::PixelMode::Rgb8, 2, 2, &mat, 3,
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
        let clip = full_clip(8, 8);
        let pipe = simple_pipe();

        let mut img_src = SolidColor { r: 200, g: 100, b: 50 };
        let mat = [2.0f64, 0.0, 0.0, 2.0, 0.0, 0.0];
        let result = draw_image::<Rgb8>(
            &mut bmp, &clip, &pipe, &mut img_src,
            crate::types::PixelMode::Rgb8, 4, 4, &mat, 3,
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
        let clip = full_clip(8, 8);
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
}
