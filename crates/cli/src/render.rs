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
    /// The rendered pixel format was unexpected.
    UnexpectedFormat(Option<ImageFormat>),
}

impl std::fmt::Display for RenderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::Bridge(e) => write!(f, "render error: {e}"),
            Self::Encode(e) => write!(f, "encode error: {e}"),
            Self::UnexpectedFormat(fmt) => write!(f, "unexpected pixel format: {fmt:?}"),
        }
    }
}

impl std::error::Error for RenderError {}

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
    let bitmap = rendered_to_bitmap_rgb(img);
    match format {
        OutputFormat::Ppm => write_ppm(&bitmap, out)?,
        OutputFormat::Png => write_png(&bitmap, out)?,
        _ => unreachable!("caller ensures only PPM/PNG reach here for RGB"),
    }
    Ok(())
}

fn write_page_gray<W: std::io::Write>(
    img: &RenderedPage,
    format: OutputFormat,
    out: W,
) -> Result<(), RenderError> {
    let bitmap = rendered_to_bitmap_gray(img);
    match format {
        OutputFormat::Ppm => write_pgm(&bitmap, out)?,
        OutputFormat::Png => write_png(&bitmap, out)?,
        _ => unreachable!("caller ensures only PPM/PNG reach here for grey"),
    }
    Ok(())
}

/// Copy poppler RGB24 image into a `Bitmap<Rgb8>`, stripping stride padding.
fn rendered_to_bitmap_rgb(img: &RenderedPage) -> Bitmap<Rgb8> {
    let w = img.width();
    let h = img.height();
    let bpr_src = img.bytes_per_row();
    let bpr_dst = w as usize * 3;
    let src = img.data();

    let mut bitmap = Bitmap::<Rgb8>::new(w, h, 1, false);
    let dst = bitmap.data_mut();
    for row in 0..h as usize {
        dst[row * bpr_dst..(row + 1) * bpr_dst]
            .copy_from_slice(&src[row * bpr_src..row * bpr_src + bpr_dst]);
    }
    bitmap
}

/// Copy poppler Gray8 image into a `Bitmap<Gray8>`, stripping stride padding.
fn rendered_to_bitmap_gray(img: &RenderedPage) -> Bitmap<Gray8> {
    let w = img.width();
    let h = img.height();
    let bpr_src = img.bytes_per_row();
    let src = img.data();

    let mut bitmap = Bitmap::<Gray8>::new(w, h, 1, false);
    let dst = bitmap.data_mut();
    for row in 0..h as usize {
        dst[row * w as usize..(row + 1) * w as usize]
            .copy_from_slice(&src[row * bpr_src..row * bpr_src + w as usize]);
    }
    bitmap
}
