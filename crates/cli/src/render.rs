//! Per-page rendering: call `pdf_raster` library → apply colour mode → write file.

use std::fs::{self, File};
use std::io::{BufWriter, Write as _};

use color::{Gray8, Rgb8};
use encode::{EncodeError, write_pbm, write_pgm, write_png, write_ppm};
use pdf_raster::{
    BackendPolicy, ImageFilter, RasterError, RasterSession, render_page_rgb_hinted, rgb_to_gray,
};
use raster::Bitmap;

use crate::args::{Args, OutputFormat};
use crate::naming::output_path;

// ── Error type ────────────────────────────────────────────────────────────────

/// Error returned by [`render_page`].
#[derive(Debug)]
pub enum RenderError {
    /// An I/O error writing the output file.
    Io(std::io::Error),
    /// The page could not be rasterised.
    Raster(RasterError),
    /// The encoder rejected the bitmap.
    Encode(EncodeError),
    /// The requested format + colour-mode combination is not supported.
    UnsupportedFormatCombination {
        /// The output format requested by the user.
        output: OutputFormat,
    },
}

impl std::fmt::Display for RenderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::Raster(e) => write!(f, "render error: {e}"),
            Self::Encode(e) => write!(f, "encode error: {e}"),
            Self::UnsupportedFormatCombination { output } => {
                write!(f, "output format {output} is not yet supported")
            }
        }
    }
}

impl std::error::Error for RenderError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::Raster(e) => Some(e),
            Self::Encode(e) => Some(e),
            Self::UnsupportedFormatCombination { .. } => None,
        }
    }
}

impl From<std::io::Error> for RenderError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}
impl From<RasterError> for RenderError {
    fn from(e: RasterError) -> Self {
        Self::Raster(e)
    }
}
impl From<EncodeError> for RenderError {
    fn from(e: EncodeError) -> Self {
        Self::Encode(e)
    }
}

// ── Per-page render + write ───────────────────────────────────────────────────

/// Render one page and write it to the appropriate output file.
///
/// `page_num` is 1-based.  Each call prescans the page to determine whether
/// it is text/vector-only, and if so passes `BackendPolicy::CpuOnly` to skip
/// GPU decoder initialisation on this thread.  The prescan runs on the same
/// worker thread as the render, so its cost is hidden inside parallel render
/// time rather than adding a serial phase before the pool starts.
///
/// JPEG and TIFF output are not yet implemented and return
/// [`RenderError::UnsupportedFormatCombination`].
pub fn render_page(
    session: &RasterSession,
    page_num: u32,
    total_pages: u32,
    args: &Args,
) -> Result<(), RenderError> {
    let format = args.output_format();

    if matches!(format, OutputFormat::Jpeg | OutputFormat::Tiff) {
        return Err(RenderError::UnsupportedFormatCombination { output: format });
    }

    // Prescan on the render thread — cost is hidden inside parallel render time.
    // No images → CpuOnly skips ensure_nvjpeg on this thread for this page.
    // Prescan errors fall back to the session policy (safe).
    let effective_policy = match pdf_raster::prescan_page(session.doc(), page_num).ok() {
        Some(ref diag) if !diag.has_images => BackendPolicy::CpuOnly,
        Some(ref diag) if matches!(diag.dominant_filter, Some(ImageFilter::Dct)) => {
            session.policy()
        }
        _ => session.policy(),
    };

    // Geometric-mean scale matches the pixel-box aspect ratio when x/y DPI differ.
    let x_dpi = args.x_dpi();
    let y_dpi = args.y_dpi();
    let scale = (x_dpi / 72.0 * (y_dpi / 72.0)).sqrt();

    let rgb: Bitmap<Rgb8> = render_page_rgb_hinted(session, page_num, scale, effective_policy)?;

    #[expect(
        clippy::cast_possible_wrap,
        reason = "total_pages ≤ i32::MAX; validated in main before calling this function"
    )]
    let out_path = output_path(args, page_num as i32, total_pages as i32, format);

    // Write to a sibling temp file then rename atomically so a failed encode
    // never leaves a partial output file at the final path.
    let tmp_path = format!("{out_path}.tmp");
    let encode_result = (|| -> Result<(), RenderError> {
        let mut out = BufWriter::new(File::create(&tmp_path)?);
        if args.mono {
            let mono = gray_to_mono(&rgb_to_gray(&rgb));
            match format {
                OutputFormat::Ppm => write_pbm::<Gray8, _>(&mono, &mut out)?,
                OutputFormat::Png => write_png::<Gray8, _>(&mono, &mut out)?,
                OutputFormat::Jpeg | OutputFormat::Tiff => unreachable!("rejected above"),
            }
        } else if args.gray {
            let gray = rgb_to_gray(&rgb);
            match format {
                OutputFormat::Ppm => write_pgm::<Gray8, _>(&gray, &mut out)?,
                OutputFormat::Png => write_png::<Gray8, _>(&gray, &mut out)?,
                OutputFormat::Jpeg | OutputFormat::Tiff => unreachable!("rejected above"),
            }
        } else {
            match format {
                OutputFormat::Ppm => write_ppm(&rgb, &mut out)?,
                OutputFormat::Png => write_png(&rgb, &mut out)?,
                OutputFormat::Jpeg | OutputFormat::Tiff => unreachable!("rejected above"),
            }
        }
        out.flush()?;
        Ok(())
    })();

    if encode_result.is_err() {
        let _ = fs::remove_file(&tmp_path);
        return encode_result;
    }

    fs::rename(&tmp_path, &out_path).map_err(|e| {
        let _ = fs::remove_file(&tmp_path);
        RenderError::Io(e)
    })?;
    Ok(())
}

// ── Pixel helpers ─────────────────────────────────────────────────────────────

/// Luma threshold separating dark ink from light paper (inclusive on the light side).
///
/// Values below this → 255 (black ink); values at or above → 0 (white paper).
const MONO_THRESHOLD: u8 = 128;

/// Threshold a grayscale bitmap to 2-level monochrome at the 50% midpoint.
///
/// Values < [`MONO_THRESHOLD`] (dark) → 255 (black ink);
/// values ≥ [`MONO_THRESHOLD`] (light) → 0 (white paper).
fn gray_to_mono(src: &Bitmap<Gray8>) -> Bitmap<Gray8> {
    let mut dst: Bitmap<Gray8> = Bitmap::new(src.width, src.height, 1, false);
    let w = src.width as usize;
    for y in 0..src.height {
        let src_row = &src.row_bytes(y)[..w];
        let dst_row = &mut dst.row_bytes_mut(y)[..w];
        for (dst_px, &luma) in dst_row.iter_mut().zip(src_row.iter()) {
            *dst_px = if luma < MONO_THRESHOLD { 255 } else { 0 };
        }
    }
    dst
}
