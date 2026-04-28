//! Per-page rendering: native Rust renderer → pixel buffer → output file.

use std::fs::File;
use std::io::{BufWriter, Write as _};
#[cfg(any(feature = "gpu-aa", feature = "nvjpeg", feature = "gpu-icc"))]
use std::sync::Arc;

use encode::{EncodeError, write_png, write_ppm};
use raster::Bitmap;

use crate::args::{Args, OutputFormat};
use crate::naming::output_path;

/// Maximum pixel dimension (width or height) accepted from a PDF page.
///
/// Prevents absurdly large allocations from malformed or adversarial documents.
/// 32 768 px at 150 DPI corresponds to roughly 366 inches (a 30-foot page).
const MAX_PX_DIMENSION: u32 = 32_768;

/// Error type for per-page rendering operations.
#[derive(Debug)]
pub enum RenderError {
    /// An I/O error writing the output file.
    Io(std::io::Error),
    /// The native renderer could not interpret the page.
    Native(pdf_interp::InterpError),
    /// The encoder rejected the bitmap.
    Encode(EncodeError),
    /// The requested output format is not supported.
    UnsupportedFormatCombination {
        /// The output format requested by the user.
        output: OutputFormat,
    },
    /// The computed pixel dimensions exceed the safety limit.
    PageTooLarge {
        /// Width in pixels.
        width: u32,
        /// Height in pixels.
        height: u32,
    },
}

impl std::fmt::Display for RenderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::Native(e) => write!(f, "render error: {e}"),
            Self::Encode(e) => write!(f, "encode error: {e}"),
            Self::UnsupportedFormatCombination { output } => {
                write!(
                    f,
                    "output format {output:?} is not yet supported by the native renderer"
                )
            }
            Self::PageTooLarge { width, height } => write!(
                f,
                "page pixel dimensions {width}×{height} exceed the safety limit \
                 ({MAX_PX_DIMENSION}); lower the DPI or check the document"
            ),
        }
    }
}

impl std::error::Error for RenderError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::Native(e) => Some(e),
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
impl From<pdf_interp::InterpError> for RenderError {
    fn from(e: pdf_interp::InterpError) -> Self {
        Self::Native(e)
    }
}
impl From<EncodeError> for RenderError {
    fn from(e: EncodeError) -> Self {
        Self::Encode(e)
    }
}

/// Render one page via the native Rust renderer and write it to disk.
///
/// `page_num` is 1-based.  PPM and PNG output are supported; JPEG and TIFF
/// are rejected with [`RenderError::UnsupportedFormatCombination`].
///
/// `--gray`/`--mono` are not yet implemented by the native renderer; a
/// warning is printed at startup and the bitmap is always RGB.
pub fn render_page_native(
    doc: &lopdf::Document,
    page_num: u32,
    total_pages: u32,
    args: &Args,
    #[cfg(any(feature = "gpu-aa", feature = "nvjpeg", feature = "gpu-icc"))]
    gpu_ctx: Option<&Arc<gpu::GpuCtx>>,
) -> Result<(), RenderError> {
    let format = args.output_format();

    if matches!(format, OutputFormat::Jpeg | OutputFormat::Tiff) {
        return Err(RenderError::UnsupportedFormatCombination { output: format });
    }

    let (w_pts, h_pts) = pdf_interp::page_size_pts(doc, page_num)?;
    let x_dpi = args.x_dpi();
    let y_dpi = args.y_dpi();

    // Geometric-mean scale so the square-pixel CTM matches the pixel box.
    let scale = (x_dpi / 72.0 * (y_dpi / 72.0)).sqrt();

    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "page dimensions in pts × scale are always positive and ≪ u32::MAX"
    )]
    let (w_px, h_px) = (
        (w_pts * scale).round() as u32,
        (h_pts * scale).round() as u32,
    );

    if w_px == 0 || h_px == 0 {
        return Err(RenderError::PageTooLarge {
            width: w_px,
            height: h_px,
        });
    }
    if w_px > MAX_PX_DIMENSION || h_px > MAX_PX_DIMENSION {
        return Err(RenderError::PageTooLarge {
            width: w_px,
            height: h_px,
        });
    }

    let pages = doc.get_pages();
    let page_id = *pages
        .get(&page_num)
        .ok_or_else(|| pdf_interp::InterpError::PageOutOfRange {
            page: page_num,
            total: pdf_interp::page_count(doc),
        })?;

    let ops = pdf_interp::parse_page(doc, page_num)?;
    let mut renderer =
        pdf_interp::renderer::PageRenderer::new_scaled(w_px, h_px, scale, doc, page_id);
    #[cfg(any(feature = "gpu-aa", feature = "nvjpeg", feature = "gpu-icc"))]
    renderer.set_gpu_ctx(gpu_ctx.map(Arc::clone));
    renderer.execute(&ops);
    renderer.render_annotations(page_id);
    let bitmap: Bitmap<color::Rgb8> = renderer.finish();

    #[expect(
        clippy::cast_possible_wrap,
        reason = "total_pages ≤ i32::MAX; validated in main before calling this function"
    )]
    let out_path = output_path(args, page_num as i32, total_pages as i32, format);
    let mut out = BufWriter::new(File::create(&out_path)?);

    match format {
        OutputFormat::Ppm => write_ppm(&bitmap, &mut out)?,
        OutputFormat::Png => write_png(&bitmap, &mut out)?,
        OutputFormat::Jpeg | OutputFormat::Tiff => {
            unreachable!("JPEG/TIFF rejected above; this arm cannot be reached")
        }
    }

    out.flush()?;
    Ok(())
}
