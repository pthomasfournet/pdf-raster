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

use color::{Pixel, PixelMode};
use raster::Bitmap;

use crate::EncodeError;

/// Write `bitmap` to `out` as a binary PPM (`P6`) image.
///
/// The sink `out` is consumed; wrap in `std::io::BufWriter` if buffering is
/// needed.
///
/// # Errors
///
/// Returns [`EncodeError::UnsupportedMode`] for grayscale or 1-bit modes —
/// use [`write_pgm`][crate::write_pgm] for those.
/// Returns [`EncodeError::Io`] on any I/O failure.
pub fn write_ppm<P: Pixel, W: Write>(bitmap: &Bitmap<P>, mut out: W) -> Result<(), EncodeError> {
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

    write_ppm_header(&mut out, bitmap.width, bitmap.height)?;
    write_ppm_pixels::<P, W>(bitmap, &mut out)?;
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
///
/// `dst` must be at least `width * 3` bytes long.
/// `src` must be at least `width * P::BYTES` bytes long.
#[inline]
fn convert_row_to_rgb<P: Pixel>(src: &[u8], dst: &mut [u8], width: usize) {
    match P::MODE {
        PixelMode::Rgb8 => {
            // Copy exactly 3 bytes per pixel — stride padding excluded via width.
            dst[..width * 3].copy_from_slice(&src[..width * 3]);
        }
        PixelMode::Bgr8 => {
            // Source: [B, G, R] → dest: [R, G, B].
            for (i, chunk) in src[..width * 3].chunks_exact(3).enumerate() {
                dst[i * 3] = chunk[2]; // R ← src[2]
                dst[i * 3 + 1] = chunk[1]; // G ← src[1]
                dst[i * 3 + 2] = chunk[0]; // B ← src[0]
            }
        }
        PixelMode::Xbgr8 => {
            // Source: [X, B, G, R] (little-endian 32-bit word) → dest: [R, G, B].
            for (i, chunk) in src[..width * 4].chunks_exact(4).enumerate() {
                dst[i * 3] = chunk[3]; // R ← src[3]
                dst[i * 3 + 1] = chunk[2]; // G ← src[2]
                dst[i * 3 + 2] = chunk[1]; // B ← src[1]
            }
        }
        PixelMode::Cmyk8 => {
            // Source: [C, M, Y, K] → dest: [R, G, B].
            for (i, chunk) in src[..width * 4].chunks_exact(4).enumerate() {
                let [r, g, b] = cmyk_to_rgb(chunk[0], chunk[1], chunk[2], chunk[3]);
                dst[i * 3] = r;
                dst[i * 3 + 1] = g;
                dst[i * 3 + 2] = b;
            }
        }
        PixelMode::DeviceN8 => {
            // Source: [C, M, Y, K, spot0..3] — use only CMYK (bytes 0..4).
            for (i, chunk) in src[..width * 8].chunks_exact(8).enumerate() {
                let [r, g, b] = cmyk_to_rgb(chunk[0], chunk[1], chunk[2], chunk[3]);
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
/// Matches the naïve ink-density formula used by poppler's `pdftoppm` and
/// the Splash rasterizer's own CMYK-to-RGB path.
#[inline]
#[expect(
    clippy::cast_sign_loss,
    reason = "each channel is clamped to [0, 255] before the cast; value is always non-negative"
)]
fn cmyk_to_rgb(cyan: u8, magenta: u8, yellow: u8, black: u8) -> [u8; 3] {
    let k = i32::from(black);
    let r = (255 - i32::from(cyan) - k).clamp(0, 255) as u8;
    let g = (255 - i32::from(magenta) - k).clamp(0, 255) as u8;
    let b = (255 - i32::from(yellow) - k).clamp(0, 255) as u8;
    [r, g, b]
}

#[cfg(test)]
mod tests {
    use super::*;
    use color::{Cmyk8, DeviceN8, Rgb8, Rgba8};
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

    /// Parse the P6 header and return the byte offset of the first pixel.
    fn header_len(out: &[u8]) -> usize {
        // Header format: "P6\n{w} {h}\n255\n" — find the third newline.
        let mut newlines = 0usize;
        for (i, &b) in out.iter().enumerate() {
            if b == b'\n' {
                newlines += 1;
                if newlines == 3 {
                    return i + 1;
                }
            }
        }
        panic!("malformed PPM header");
    }

    #[test]
    fn rgb_ppm_header_and_pixels() {
        let bmp = make_rgb_bitmap(2, 1, [255, 128, 0]);
        let mut out = Vec::new();
        write_ppm::<Rgb8, _>(&bmp, &mut out).unwrap();

        let expected_header = b"P6\n2 1\n255\n";
        assert!(
            out.starts_with(expected_header),
            "header mismatch: {:?}",
            &out[..expected_header.len().min(out.len())]
        );

        let pixels = &out[expected_header.len()..];
        assert_eq!(pixels.len(), 6, "2 pixels × 3 bytes");
        assert_eq!(&pixels[..3], &[255, 128, 0]);
        assert_eq!(&pixels[3..6], &[255, 128, 0]);
    }

    #[test]
    fn rgba8_xbgr_ppm_channel_swap() {
        // Rgba8 has MODE=Xbgr8; memory layout is [X/A, B, G, R].
        let mut bmp: Bitmap<Rgba8> = Bitmap::new(1, 1, 1, false);
        // [A=255, B=10, G=20, R=30]
        bmp.row_bytes_mut(0).copy_from_slice(&[255, 10, 20, 30]);
        let mut out = Vec::new();
        write_ppm::<Rgba8, _>(&bmp, &mut out).unwrap();
        let hlen = header_len(&out);
        assert_eq!(
            &out[hlen..],
            &[30, 20, 10],
            "Xbgr8 must become RGB (channels swapped)"
        );
    }

    #[test]
    fn cmyk_black_converts_to_rgb_black() {
        let mut bmp: Bitmap<Cmyk8> = Bitmap::new(1, 1, 1, false);
        // CMYK (0, 0, 0, 255) = pure black.
        bmp.row_bytes_mut(0).copy_from_slice(&[0, 0, 0, 255]);
        let mut out = Vec::new();
        write_ppm::<Cmyk8, _>(&bmp, &mut out).unwrap();
        let hlen = header_len(&out);
        assert_eq!(&out[hlen..], &[0, 0, 0], "CMYK black → RGB (0,0,0)");
    }

    #[test]
    fn cmyk_white_converts_to_rgb_white() {
        let mut bmp: Bitmap<Cmyk8> = Bitmap::new(1, 1, 1, false);
        // CMYK (0, 0, 0, 0) = pure white.
        bmp.row_bytes_mut(0).copy_from_slice(&[0, 0, 0, 0]);
        let mut out = Vec::new();
        write_ppm::<Cmyk8, _>(&bmp, &mut out).unwrap();
        let hlen = header_len(&out);
        assert_eq!(&out[hlen..], &[255, 255, 255], "CMYK white → RGB white");
    }

    #[test]
    fn devicen_uses_only_cmyk_portion() {
        let mut bmp: Bitmap<DeviceN8> = Bitmap::new(1, 1, 1, false);
        // DeviceN8: CMYK=(0,0,0,0) → white; spot channels ignored.
        bmp.row_bytes_mut(0)
            .copy_from_slice(&[0, 0, 0, 0, 99, 99, 99, 99]);
        let mut out = Vec::new();
        write_ppm::<DeviceN8, _>(&bmp, &mut out).unwrap();
        let hlen = header_len(&out);
        assert_eq!(
            &out[hlen..],
            &[255, 255, 255],
            "DeviceN spot channels must be ignored"
        );
    }

    #[test]
    fn stride_padding_not_written() {
        // width=1, pad=4 → stride=4 for Rgb8 (3 bytes/px rounded up to 4).
        let bmp: Bitmap<Rgb8> = Bitmap::new(1, 1, 4, false);
        let mut out = Vec::new();
        write_ppm::<Rgb8, _>(&bmp, &mut out).unwrap();
        let hlen = header_len(&out);
        // Must be exactly 3 pixel bytes, not 4 (stride).
        assert_eq!(
            out.len() - hlen,
            3,
            "stride padding must not appear in PPM output"
        );
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
