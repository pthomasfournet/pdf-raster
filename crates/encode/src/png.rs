//! PNG encoder (via the `png` crate).
//!
//! Supports `Rgb8`, `Gray8`, and `Rgba8` bitmaps directly.
//! Other modes return [`EncodeError::UnsupportedMode`] — convert to one
//! of the above before encoding.
//!
//! PNG is lossless and handles transparency (alpha plane) correctly.
//! Use this format in preference to PPM when the bitmap has an alpha plane.

use std::io::Write;

use color::{Pixel, PixelMode};
use raster::Bitmap;

use crate::EncodeError;

/// Write `bitmap` to `out` as a PNG image.
///
/// Supported pixel modes:
///
/// | Mode | PNG colour type |
/// |------|----------------|
/// | `Rgb8` | `RGB` (24-bit) |
/// | `Gray8` / `Mono8` | `Grayscale` (8-bit) |
/// | `Rgba8` | `RGBA` (32-bit) |
///
/// # Alpha plane
///
/// For `Rgb8` bitmaps: if the bitmap has an alpha plane, each RGB row is
/// interleaved with the corresponding alpha bytes and written as RGBA PNG.
///
/// For `Rgba8` bitmaps: the alpha stored within each pixel is used; any
/// separate alpha plane is ignored.
///
/// # Errors
///
/// Returns [`EncodeError::UnsupportedMode`] for CMYK, BGR, XBGR, `DeviceN`,
/// or `Mono1` bitmaps.
/// Returns [`EncodeError::Io`] or [`EncodeError::Png`] on failure.
pub fn write_png<P: Pixel, W: Write>(bitmap: &Bitmap<P>, out: W) -> Result<(), EncodeError> {
    match P::MODE {
        PixelMode::Rgb8 => write_png_rgb(bitmap, out),
        PixelMode::Mono8 => write_png_gray(bitmap, out),
        PixelMode::Xbgr8 => {
            // Rgba8 is stored as Xbgr8 in the Pixel trait implementation.
            write_png_rgba(bitmap, out)
        }
        PixelMode::Bgr8 | PixelMode::Cmyk8 | PixelMode::DeviceN8 | PixelMode::Mono1 => {
            Err(EncodeError::UnsupportedMode(
                "unsupported mode for PNG: convert to Rgb8/Gray8/Rgba8 first",
            ))
        }
    }
}

fn png_encoder<W: Write>(
    out: W,
    width: u32,
    height: u32,
    color: ::png::ColorType,
    depth: ::png::BitDepth,
) -> Result<::png::Writer<W>, EncodeError> {
    let mut encoder = ::png::Encoder::new(out, width, height);
    encoder.set_color(color);
    encoder.set_depth(depth);
    // Paeth filter gives good compression for photographic/gradient content.
    encoder.set_filter(::png::FilterType::Paeth);
    encoder.set_compression(::png::Compression::Default);
    Ok(encoder.write_header()?)
}

/// Write an `Rgb8` bitmap as PNG, promoting to RGBA if an alpha plane is present.
fn write_png_rgb<P: Pixel, W: Write>(bitmap: &Bitmap<P>, out: W) -> Result<(), EncodeError> {
    let w = bitmap.width as usize;
    let h = bitmap.height as usize;
    let has_alpha = bitmap.has_alpha();

    if has_alpha {
        // Promote to RGBA: interleave pixel RGB with alpha plane.
        let mut buf = vec![0u8; w * h * 4];
        for y in 0..bitmap.height {
            let rgb = bitmap.row_bytes(y);
            // alpha_row returns None only when has_alpha is false — checked above.
            let alpha = bitmap
                .alpha_row(y)
                .expect("has_alpha is true but alpha_row returned None");
            let row_off = y as usize * w * 4;
            for i in 0..w {
                buf[row_off + i * 4] = rgb[i * 3];
                buf[row_off + i * 4 + 1] = rgb[i * 3 + 1];
                buf[row_off + i * 4 + 2] = rgb[i * 3 + 2];
                buf[row_off + i * 4 + 3] = alpha[i];
            }
        }
        let mut writer = png_encoder(
            out,
            bitmap.width,
            bitmap.height,
            ::png::ColorType::Rgba,
            ::png::BitDepth::Eight,
        )?;
        writer.write_image_data(&buf)?;
    } else {
        // Pack rows contiguously (exclude stride padding).
        let mut buf = vec![0u8; w * h * 3];
        for y in 0..bitmap.height {
            let row = bitmap.row_bytes(y);
            let row_off = y as usize * w * 3;
            buf[row_off..row_off + w * 3].copy_from_slice(&row[..w * 3]);
        }
        let mut writer = png_encoder(
            out,
            bitmap.width,
            bitmap.height,
            ::png::ColorType::Rgb,
            ::png::BitDepth::Eight,
        )?;
        writer.write_image_data(&buf)?;
    }

    Ok(())
}

/// Write a `Gray8` bitmap as PNG.
fn write_png_gray<P: Pixel, W: Write>(bitmap: &Bitmap<P>, out: W) -> Result<(), EncodeError> {
    let w = bitmap.width as usize;
    let h = bitmap.height as usize;
    // Pack rows contiguously (exclude stride padding).
    let mut buf = vec![0u8; w * h];
    for y in 0..bitmap.height {
        let row = bitmap.row_bytes(y);
        let row_off = y as usize * w;
        buf[row_off..row_off + w].copy_from_slice(&row[..w]);
    }
    let mut writer = png_encoder(
        out,
        bitmap.width,
        bitmap.height,
        ::png::ColorType::Grayscale,
        ::png::BitDepth::Eight,
    )?;
    writer.write_image_data(&buf)?;
    Ok(())
}

/// Write an `Rgba8`-equivalent bitmap (stored as `Xbgr8`) as PNG RGBA.
///
/// `Rgba8` uses the `Xbgr8` pixel mode internally; the channel layout in memory
/// is X(=A), B, G, R (little-endian 32-bit), so we must swap to R, G, B, A.
fn write_png_rgba<P: Pixel, W: Write>(bitmap: &Bitmap<P>, out: W) -> Result<(), EncodeError> {
    let w = bitmap.width as usize;
    let h = bitmap.height as usize;
    let mut buf = vec![0u8; w * h * 4];
    for y in 0..bitmap.height {
        let row = bitmap.row_bytes(y);
        let row_off = y as usize * w * 4;
        // Memory layout: [X/A, B, G, R] per pixel (Xbgr8).
        for i in 0..w {
            let src = i * 4;
            let dst = row_off + i * 4;
            buf[dst] = row[src + 3]; // R
            buf[dst + 1] = row[src + 2]; // G
            buf[dst + 2] = row[src + 1]; // B
            buf[dst + 3] = row[src]; // A (was X)
        }
    }
    let mut writer = png_encoder(
        out,
        bitmap.width,
        bitmap.height,
        ::png::ColorType::Rgba,
        ::png::BitDepth::Eight,
    )?;
    writer.write_image_data(&buf)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use color::{Gray8, Rgb8};
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

    fn make_gray_bitmap(w: u32, h: u32, fill: u8) -> Bitmap<Gray8> {
        let mut bmp = Bitmap::new(w, h, 1, false);
        for y in 0..h {
            bmp.row_bytes_mut(y).fill(fill);
        }
        bmp
    }

    /// Decode a PNG from bytes and return (width, height, raw_pixels).
    fn decode_png(data: &[u8]) -> (u32, u32, Vec<u8>) {
        let decoder = ::png::Decoder::new(std::io::Cursor::new(data));
        let mut reader = decoder.read_info().expect("png decode header");
        let mut buf = vec![0u8; reader.output_buffer_size()];
        let frame = reader.next_frame(&mut buf).expect("png decode frame");
        let info = reader.info();
        (info.width, info.height, buf[..frame.buffer_size()].to_vec())
    }

    #[test]
    fn rgb_png_roundtrip() {
        let bmp = make_rgb_bitmap(4, 2, [100, 150, 200]);
        let mut out = Vec::new();
        write_png::<Rgb8, _>(&bmp, &mut out).unwrap();

        let (w, h, pixels) = decode_png(&out);
        assert_eq!((w, h), (4, 2));
        // 4×2 pixels × 3 bytes = 24 bytes.
        assert_eq!(pixels.len(), 24);
        for chunk in pixels.chunks_exact(3) {
            assert_eq!(chunk, &[100, 150, 200], "pixel mismatch");
        }
    }

    #[test]
    fn gray_png_roundtrip() {
        let bmp = make_gray_bitmap(3, 3, 77);
        let mut out = Vec::new();
        write_png::<Gray8, _>(&bmp, &mut out).unwrap();

        let (w, h, pixels) = decode_png(&out);
        assert_eq!((w, h), (3, 3));
        assert!(pixels.iter().all(|&v| v == 77), "grayscale pixel mismatch");
    }

    #[test]
    fn rgb_with_alpha_writes_rgba_png() {
        // Bitmap with alpha plane: every pixel [255,0,0] with alpha 128.
        let mut bmp: Bitmap<Rgb8> = Bitmap::new(2, 1, 1, true);
        let row = bmp.row_bytes_mut(0);
        row[..6].copy_from_slice(&[255, 0, 0, 255, 0, 0]);
        if let Some(a) = bmp.alpha_plane_mut() {
            a.fill(128);
        }

        let mut out = Vec::new();
        write_png::<Rgb8, _>(&bmp, &mut out).unwrap();

        let (w, h, pixels) = decode_png(&out);
        assert_eq!((w, h), (2, 1));
        // Should be RGBA: 2 pixels × 4 bytes.
        assert_eq!(pixels.len(), 8);
        assert_eq!(&pixels[..4], &[255, 0, 0, 128], "pixel 0 RGBA");
        assert_eq!(&pixels[4..8], &[255, 0, 0, 128], "pixel 1 RGBA");
    }

    #[test]
    fn stride_padding_not_included() {
        // width=3 with pad=4 → stride=4 for Gray8.
        let bmp: Bitmap<Gray8> = Bitmap::new(3, 1, 4, false);
        // data: [0,0,0, pad] per row — fill data with alternating pattern.
        // The pad byte should not appear in the PNG.
        let mut out = Vec::new();
        write_png::<Gray8, _>(&bmp, &mut out).unwrap();
        let (w, h, pixels) = decode_png(&out);
        assert_eq!((w, h), (3, 1));
        assert_eq!(pixels.len(), 3, "must not include stride padding in PNG");
    }

    #[test]
    fn cmyk_returns_unsupported_error() {
        use color::Cmyk8;
        let bmp: Bitmap<Cmyk8> = Bitmap::new(1, 1, 1, false);
        let mut out = Vec::new();
        let result = write_png::<Cmyk8, _>(&bmp, &mut out);
        assert!(
            matches!(result, Err(EncodeError::UnsupportedMode(_))),
            "Cmyk8 should return UnsupportedMode for PNG"
        );
    }
}
