//! Netpbm P6 (binary PPM) encoder.
//!
//! PPM stores RGB 8-bit pixels.  Pixel modes that are not natively RGB are
//! converted on the fly:
//!
//! | Mode | Conversion |
//! |------|-----------|
//! | `Rgb8` | verbatim |
//! | `Bgr8` / `Xbgr8` | channel swap |
//! | `Cmyk8` | `R = 255−C−K`, `G = 255−M−K`, `B = 255−Y−K` (clamped) |
//! | `DeviceN8` | CMYK portion only (same formula) |
//! | `Gray8` / `Mono8` | not supported — use [`write_pgm`][crate::write_pgm] |
//! | `Mono1` | not supported |
//!
//! The output is a standard `P6` header followed by raw RGB bytes, matching
//! the format produced by poppler's `pdftoppm -r <dpi>`.

use std::io::{self, Write};

use color::{Cmyk8, DeviceN8, Pixel, PixelMode};
use raster::Bitmap;

use crate::EncodeError;

/// Write `bitmap` to `out` as a binary PPM (`P6`) image.
///
/// # Errors
///
/// Returns [`EncodeError::UnsupportedMode`] for grayscale or 1-bit modes —
/// use [`write_pgm`][crate::write_pgm] for those.
/// Returns [`EncodeError::Io`] on any I/O failure.
pub fn write_ppm<P: Pixel, W: Write>(bitmap: &Bitmap<P>, out: &mut W) -> Result<(), EncodeError> {
    match P::MODE {
        PixelMode::Mono1 | PixelMode::Mono8 => {
            return Err(EncodeError::UnsupportedMode(
                "grayscale/mono bitmap: use write_pgm instead",
            ));
        }
        PixelMode::Rgb8
        | PixelMode::Bgr8
        | PixelMode::Xbgr8
        | PixelMode::Cmyk8
        | PixelMode::DeviceN8 => {}
    }

    write_ppm_header(out, bitmap.width, bitmap.height)?;
    write_ppm_pixels::<P, W>(bitmap, out)?;
    out.flush()?;
    Ok(())
}

/// Write the P6 header.
fn write_ppm_header<W: Write>(out: &mut W, width: u32, height: u32) -> io::Result<()> {
    writeln!(out, "P6")?;
    writeln!(out, "{width} {height}")?;
    writeln!(out, "255")?;
    Ok(())
}

/// Write all pixel rows, converting to RGB in place.
fn write_ppm_pixels<P: Pixel, W: Write>(bitmap: &Bitmap<P>, out: &mut W) -> io::Result<()> {
    // Pre-allocate one row of output RGB bytes.
    let w = bitmap.width as usize;
    let mut rgb_row = vec![0u8; w * 3];

    for y in 0..bitmap.height {
        let src = bitmap.row_bytes(y);
        convert_row_to_rgb::<P>(src, &mut rgb_row, w);
        out.write_all(&rgb_row)?;
    }
    Ok(())
}

/// Convert one source row (any supported pixel mode) into RGB bytes.
#[inline]
fn convert_row_to_rgb<P: Pixel>(src: &[u8], dst: &mut [u8], width: usize) {
    match P::MODE {
        PixelMode::Rgb8 => {
            // Copy exactly 3 bytes per pixel — stride padding excluded via width.
            let n = width * 3;
            dst[..n].copy_from_slice(&src[..n]);
        }
        PixelMode::Bgr8 => {
            for (i, chunk) in src[..width * 3].chunks_exact(3).enumerate() {
                dst[i * 3] = chunk[2]; // R ← B-channel
                dst[i * 3 + 1] = chunk[1]; // G
                dst[i * 3 + 2] = chunk[0]; // B ← R-channel
            }
        }
        PixelMode::Xbgr8 => {
            // Layout: X B G R (little-endian 32-bit word).
            for (i, chunk) in src[..width * 4].chunks_exact(4).enumerate() {
                dst[i * 3] = chunk[3]; // R
                dst[i * 3 + 1] = chunk[2]; // G
                dst[i * 3 + 2] = chunk[1]; // B
            }
        }
        PixelMode::Cmyk8 => {
            for (i, chunk) in src[..width * 4].chunks_exact(4).enumerate() {
                let px = Cmyk8 {
                    c: chunk[0],
                    m: chunk[1],
                    y: chunk[2],
                    k: chunk[3],
                };
                let [r, g, b] = cmyk_to_rgb(px.c, px.m, px.y, px.k);
                dst[i * 3] = r;
                dst[i * 3 + 1] = g;
                dst[i * 3 + 2] = b;
            }
        }
        PixelMode::DeviceN8 => {
            for (i, chunk) in src[..width * 8].chunks_exact(8).enumerate() {
                // Use only the CMYK portion (bytes 0..4); spot channels are ignored.
                let px = DeviceN8 {
                    cmyk: Cmyk8 {
                        c: chunk[0],
                        m: chunk[1],
                        y: chunk[2],
                        k: chunk[3],
                    },
                    spots: [chunk[4], chunk[5], chunk[6], chunk[7]],
                };
                let [r, g, b] = cmyk_to_rgb(px.cmyk.c, px.cmyk.m, px.cmyk.y, px.cmyk.k);
                dst[i * 3] = r;
                dst[i * 3 + 1] = g;
                dst[i * 3 + 2] = b;
            }
        }
        PixelMode::Mono1 | PixelMode::Mono8 => {
            // Guarded in write_ppm; this branch is unreachable.
            debug_assert!(false, "convert_row_to_rgb: unsupported mono mode");
        }
    }
}

/// Simple CMYK → RGB: `R = 255 − C − K`, clamped to `[0, 255]`.
///
/// This is the naïve ink-density formula used by poppler's `pdftoppm` and
/// matches the Splash rasterizer's own CMYK-to-RGB path.
#[inline]
fn cmyk_to_rgb(cyan: u8, magenta: u8, yellow: u8, black: u8) -> [u8; 3] {
    let k = i32::from(black);
    #[expect(
        clippy::cast_sign_loss,
        reason = "clamped to [0, 255] before cast; value is always non-negative"
    )]
    let r = (255 - i32::from(cyan) - k).clamp(0, 255) as u8;
    #[expect(
        clippy::cast_sign_loss,
        reason = "clamped to [0, 255] before cast; value is always non-negative"
    )]
    let g = (255 - i32::from(magenta) - k).clamp(0, 255) as u8;
    #[expect(
        clippy::cast_sign_loss,
        reason = "clamped to [0, 255] before cast; value is always non-negative"
    )]
    let b = (255 - i32::from(yellow) - k).clamp(0, 255) as u8;
    [r, g, b]
}

#[cfg(test)]
mod tests {
    use super::*;
    use color::{Cmyk8, Rgb8};
    use raster::Bitmap;

    fn make_rgb_bitmap(w: u32, h: u32, fill: [u8; 3]) -> Bitmap<Rgb8> {
        let mut bmp = Bitmap::new(w, h, 1, false);
        for y in 0..h {
            let row = bmp.row_bytes_mut(y);
            for chunk in row.chunks_exact_mut(3) {
                chunk.copy_from_slice(&fill);
            }
        }
        bmp
    }

    #[test]
    fn rgb_ppm_header_and_pixels() {
        let bmp = make_rgb_bitmap(2, 1, [255, 128, 0]);
        let mut out = Vec::new();
        write_ppm::<Rgb8, _>(&bmp, &mut out).unwrap();

        // The header is ASCII text; pixels follow as raw binary.
        let expected_header = b"P6\n2 1\n255\n";
        assert!(
            out.starts_with(expected_header),
            "header mismatch: {:?}",
            &out[..expected_header.len().min(out.len())]
        );

        // Pixels: 2 × 3 bytes = 6 bytes after header.
        let pixels = &out[expected_header.len()..];
        assert_eq!(pixels.len(), 6);
        assert_eq!(&pixels[..3], &[255, 128, 0]);
        assert_eq!(&pixels[3..6], &[255, 128, 0]);
    }

    #[test]
    fn cmyk_black_converts_to_rgb_black() {
        let mut bmp: Bitmap<Cmyk8> = Bitmap::new(1, 1, 1, false);
        // CMYK (0, 0, 0, 255) = pure black.
        bmp.row_bytes_mut(0).copy_from_slice(&[0, 0, 0, 255]);
        let mut out = Vec::new();
        write_ppm::<Cmyk8, _>(&bmp, &mut out).unwrap();
        let header_len = "P6\n1 1\n255\n".len();
        let pixels = &out[header_len..];
        assert_eq!(pixels, &[0, 0, 0], "CMYK black should map to RGB (0,0,0)");
    }

    #[test]
    fn cmyk_white_converts_to_rgb_white() {
        let mut bmp: Bitmap<Cmyk8> = Bitmap::new(1, 1, 1, false);
        // CMYK (0, 0, 0, 0) = pure white.
        bmp.row_bytes_mut(0).copy_from_slice(&[0, 0, 0, 0]);
        let mut out = Vec::new();
        write_ppm::<Cmyk8, _>(&bmp, &mut out).unwrap();
        let header_len = "P6\n1 1\n255\n".len();
        let pixels = &out[header_len..];
        assert_eq!(pixels, &[255, 255, 255], "CMYK white → RGB white");
    }

    #[test]
    fn mono8_returns_unsupported_error() {
        use color::Gray8;
        let bmp: Bitmap<Gray8> = Bitmap::new(1, 1, 1, false);
        let mut out = Vec::new();
        let result = write_ppm::<Gray8, _>(&bmp, &mut out);
        assert!(
            matches!(result, Err(EncodeError::UnsupportedMode(_))),
            "Gray8 should return UnsupportedMode"
        );
    }

    #[test]
    fn cmyk_to_rgb_clamped() {
        // cyan=200, black=100 → 255-200-100 = -45 → clamped to 0.
        let [r, g, b] = cmyk_to_rgb(200, 0, 0, 100);
        assert_eq!(r, 0, "negative result must clamp to 0");
        assert_eq!(g, 155, "magenta=0, k=100: 255-0-100=155");
        assert_eq!(b, 155, "yellow=0, k=100: 255-0-100=155");
    }
}
