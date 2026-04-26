//! Netpbm P5 (binary PGM) encoder.
//!
//! PGM stores 8-bit grayscale pixels.  Only `Gray8` / `Mono8` bitmaps are
//! accepted; all other modes return [`EncodeError::UnsupportedMode`].
//!
//! The output is a standard `P5` header followed by raw luminance bytes.

use std::io::{self, Write};

use color::{Pixel, PixelMode};
use raster::Bitmap;

use crate::EncodeError;

/// Write `bitmap` to `out` as a binary PGM (`P5`) image.
///
/// The sink `out` is consumed; wrap in `std::io::BufWriter` if buffering is
/// needed.
///
/// # Errors
///
/// Returns [`EncodeError::UnsupportedMode`] for non-grayscale modes —
/// use [`write_ppm`][crate::write_ppm] for colour bitmaps.
/// Returns [`EncodeError::Io`] on any I/O failure.
pub fn write_pgm<P: Pixel, W: Write>(bitmap: &Bitmap<P>, mut out: W) -> Result<(), EncodeError> {
    match P::MODE {
        PixelMode::Mono8 => {}
        PixelMode::Mono1
        | PixelMode::Rgb8
        | PixelMode::Bgr8
        | PixelMode::Xbgr8
        | PixelMode::Cmyk8
        | PixelMode::DeviceN8 => {
            return Err(EncodeError::UnsupportedMode(
                "non-grayscale bitmap: use write_ppm or write_png",
            ));
        }
    }

    write_pgm_header(&mut out, bitmap.width, bitmap.height)?;

    // Each row may be wider than `width` bytes (stride padding).
    // Write only the live pixel bytes.
    let w = bitmap.width as usize;
    for y in 0..bitmap.height {
        let row = bitmap.row_bytes(y);
        out.write_all(&row[..w])?;
    }

    out.flush()?;
    Ok(())
}

/// Write the P5 header.
fn write_pgm_header<W: Write>(out: &mut W, width: u32, height: u32) -> io::Result<()> {
    writeln!(out, "P5")?;
    writeln!(out, "{width} {height}")?;
    writeln!(out, "255")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use color::{Gray8, Rgb8};
    use raster::Bitmap;

    fn make_gray_bitmap(w: u32, h: u32, fill: u8) -> Bitmap<Gray8> {
        let mut bmp = Bitmap::new(w, h, 1, false);
        for y in 0..h {
            bmp.row_bytes_mut(y).fill(fill);
        }
        bmp
    }

    #[test]
    fn pgm_header_and_pixels() {
        let bmp = make_gray_bitmap(3, 2, 128);
        let mut out = Vec::new();
        write_pgm::<Gray8, _>(&bmp, &mut out).unwrap();

        let header = "P5\n3 2\n255\n";
        assert!(
            out.starts_with(header.as_bytes()),
            "header mismatch: {:?}",
            &out[..header.len().min(out.len())]
        );
        let pixels = &out[header.len()..];
        assert_eq!(pixels.len(), 6, "3×2 = 6 pixel bytes");
        assert!(pixels.iter().all(|&v| v == 128), "all pixels should be 128");
    }

    #[test]
    fn stride_padding_excluded() {
        // Bitmap with row_pad=4: stride for Gray8 (1 byte/px) at width=3 is 4.
        let bmp_padded: Bitmap<Gray8> = Bitmap::new(3, 1, 4, false);
        assert!(
            bmp_padded.stride >= 3,
            "padded stride must be at least width (sanity check on Bitmap::new)"
        );

        let mut out = Vec::new();
        write_pgm::<Gray8, _>(&bmp_padded, &mut out).unwrap();
        let header = "P5\n3 1\n255\n";
        let pixels = &out[header.len()..];
        // Must be exactly 3 bytes (width), not 4 (stride).
        assert_eq!(
            pixels.len(),
            3,
            "stride padding must not appear in PGM output"
        );
    }

    #[test]
    fn rgb8_returns_unsupported_error() {
        let bmp: Bitmap<Rgb8> = Bitmap::new(1, 1, 1, false);
        let mut out = Vec::new();
        let result = write_pgm::<Rgb8, _>(&bmp, &mut out);
        assert!(
            matches!(result, Err(EncodeError::UnsupportedMode(_))),
            "Rgb8 should return UnsupportedMode for PGM"
        );
    }
}
