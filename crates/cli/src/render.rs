//! Per-page rendering: poppler → pixel buffer → encoded output file.

use std::fs::File;
use std::io::BufWriter;

use color::{Gray8, Rgb8};
use encode::{EncodeError, write_pgm, write_png, write_ppm};
use pdf_bridge::{Document, ImageFormat, RenderParams, RenderedPage};
use raster::Bitmap;

use crate::args::{Args, OutputFormat};
use crate::naming::output_path;

/// Error type for per-page rendering operations.
#[derive(Debug)]
pub enum RenderError {
    /// An I/O error writing the output file.
    Io(std::io::Error),
    /// Poppler failed to render the page.
    Bridge(pdf_bridge::Error),
    /// The encoder rejected the bitmap.
    Encode(EncodeError),
    /// The rendered pixel format was not one the CLI knows how to encode.
    UnexpectedFormat(Option<ImageFormat>),
    /// The requested output format is not supported for this pixel mode.
    UnsupportedFormatCombination {
        /// The output format requested by the user.
        output: OutputFormat,
        /// The pixel format that was rendered.
        pixel: Option<ImageFormat>,
    },
}

impl std::fmt::Display for RenderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::Bridge(e) => write!(f, "render error: {e}"),
            Self::Encode(e) => write!(f, "encode error: {e}"),
            Self::UnexpectedFormat(fmt) => write!(f, "unexpected pixel format: {fmt:?}"),
            Self::UnsupportedFormatCombination { output, pixel } => write!(
                f,
                "output format {output:?} is not supported for pixel format {pixel:?}"
            ),
        }
    }
}

impl std::error::Error for RenderError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::Bridge(e) => Some(e),
            Self::Encode(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for RenderError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}
impl From<pdf_bridge::Error> for RenderError {
    fn from(e: pdf_bridge::Error) -> Self {
        Self::Bridge(e)
    }
}
impl From<EncodeError> for RenderError {
    fn from(e: EncodeError) -> Self {
        Self::Encode(e)
    }
}

/// Render one page and write it to disk.
///
/// `page_num` is 1-based (matching PDF conventions and pdftoppm output).
pub fn render_page(
    doc: &Document,
    page_num: i32,
    total_pages: i32,
    args: &Args,
) -> Result<(), RenderError> {
    let page = doc.page(page_num - 1)?; // 0-based index

    let format = args.output_format();

    // JPEG, JPEG-CMYK, and TIFF are not yet implemented; report clearly rather
    // than silently producing a wrong output format or panicking.
    if matches!(format, OutputFormat::Jpeg | OutputFormat::Tiff) {
        return Err(RenderError::UnsupportedFormatCombination {
            output: format,
            pixel: None,
        });
    }

    let pop_format = if args.gray || args.mono {
        ImageFormat::Gray8
    } else {
        ImageFormat::Rgb24
    };

    let params = RenderParams {
        x_dpi: args.x_dpi(),
        y_dpi: args.y_dpi(),
        format: pop_format,
        antialias: args.antialias.is_on(),
        text_antialias: args.antialias.is_on(),
        text_hinting: false,
    };

    let img = page.render(&params)?;
    let out_path = output_path(args, page_num, total_pages, format);
    let out = BufWriter::new(File::create(&out_path)?);

    match img.format() {
        Some(ImageFormat::Rgb24) => write_page_rgb(&img, format, out)?,
        Some(ImageFormat::Gray8) => write_page_gray(&img, format, out)?,
        fmt => return Err(RenderError::UnexpectedFormat(fmt)),
    }

    Ok(())
}

fn write_page_rgb<W: std::io::Write>(
    img: &RenderedPage,
    format: OutputFormat,
    out: W,
) -> Result<(), RenderError> {
    let bitmap = rendered_to_bitmap::<Rgb8, 3>(img);
    match format {
        OutputFormat::Ppm => write_ppm(&bitmap, out)?,
        OutputFormat::Png => write_png(&bitmap, out)?,
        OutputFormat::Jpeg | OutputFormat::Tiff => {
            return Err(RenderError::UnsupportedFormatCombination {
                output: format,
                pixel: Some(ImageFormat::Rgb24),
            });
        }
    }
    Ok(())
}

fn write_page_gray<W: std::io::Write>(
    img: &RenderedPage,
    format: OutputFormat,
    out: W,
) -> Result<(), RenderError> {
    let bitmap = rendered_to_bitmap::<Gray8, 1>(img);
    match format {
        OutputFormat::Ppm => write_pgm(&bitmap, out)?,
        OutputFormat::Png => write_png(&bitmap, out)?,
        OutputFormat::Jpeg | OutputFormat::Tiff => {
            return Err(RenderError::UnsupportedFormatCombination {
                output: format,
                pixel: Some(ImageFormat::Gray8),
            });
        }
    }
    Ok(())
}

/// Copy a poppler image into a `Bitmap<P>`, stripping any stride padding.
///
/// `BPP` is the number of bytes per pixel (must match `P`'s in-memory layout).
fn rendered_to_bitmap<P, const BPP: usize>(img: &RenderedPage) -> Bitmap<P>
where
    P: color::Pixel,
{
    let w = img.width();
    let h = img.height();
    let bpr_src = img.bytes_per_row();
    let bpr_dst = w as usize * BPP;
    let src = img.data();

    let mut bitmap = Bitmap::<P>::new(w, h, 1, false);
    let dst = bitmap.data_mut();
    for row in 0..h as usize {
        dst[row * bpr_dst..(row + 1) * bpr_dst]
            .copy_from_slice(&src[row * bpr_src..row * bpr_src + bpr_dst]);
    }
    bitmap
}
