//! Per-page rendering: native Rust renderer → pixel buffer → output file.

use std::fs::File;
use std::io::{BufWriter, Write as _};
#[cfg(any(feature = "gpu-aa", feature = "nvjpeg", feature = "gpu-icc"))]
use std::sync::Arc;

use color::{Gray8, Rgb8};
use encode::{EncodeError, write_pbm, write_pgm, write_png, write_ppm};
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
    /// The page `MediaBox` resolved to zero width or height — malformed document.
    PageDegenerate {
        /// Width in pixels (0 when degenerate).
        width: u32,
        /// Height in pixels (0 when degenerate).
        height: u32,
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
            Self::PageDegenerate { width, height } => write!(
                f,
                "page has degenerate pixel dimensions {width}×{height} — \
                 the PDF MediaBox may be malformed"
            ),
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
            Self::UnsupportedFormatCombination { .. }
            | Self::PageDegenerate { .. }
            | Self::PageTooLarge { .. } => None,
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
/// When compiled with `gpu-aa`, `nvjpeg`, or `gpu-icc` features, `gpu_ctx`
/// is passed to the renderer.  `None` is safe — it reverts to the CPU path.
///
/// `--gray` converts the RGB bitmap to grayscale (BT.709) and writes PGM/gray PNG.
/// `--mono` additionally thresholds to 1-bit and writes PBM/gray PNG.
pub fn render_page_native(
    doc: &lopdf::Document,
    page_num: u32,
    total_pages: u32,
    args: &Args,
    #[cfg(any(feature = "gpu-aa", feature = "nvjpeg", feature = "gpu-icc"))] gpu_ctx: Option<
        &Arc<gpu::GpuCtx>,
    >,
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
        return Err(RenderError::PageDegenerate {
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
    let rgb: Bitmap<Rgb8> = renderer.finish();

    #[expect(
        clippy::cast_possible_wrap,
        reason = "total_pages ≤ i32::MAX; validated in main before calling this function"
    )]
    let out_path = output_path(args, page_num as i32, total_pages as i32, format);
    let mut out = BufWriter::new(File::create(&out_path)?);

    if args.mono {
        let gray = rgb_to_gray(&rgb);
        let mono = gray_to_mono(&gray);
        match format {
            OutputFormat::Ppm => write_pbm::<Gray8, _>(&mono, &mut out)?,
            OutputFormat::Png => write_png::<Gray8, _>(&mono, &mut out)?,
            OutputFormat::Jpeg | OutputFormat::Tiff => {
                unreachable!("JPEG/TIFF rejected above; this arm cannot be reached")
            }
        }
    } else if args.gray {
        let gray = rgb_to_gray(&rgb);
        match format {
            OutputFormat::Ppm => write_pgm::<Gray8, _>(&gray, &mut out)?,
            OutputFormat::Png => write_png::<Gray8, _>(&gray, &mut out)?,
            OutputFormat::Jpeg | OutputFormat::Tiff => {
                unreachable!("JPEG/TIFF rejected above; this arm cannot be reached")
            }
        }
    } else {
        match format {
            OutputFormat::Ppm => write_ppm(&rgb, &mut out)?,
            OutputFormat::Png => write_png(&rgb, &mut out)?,
            OutputFormat::Jpeg | OutputFormat::Tiff => {
                unreachable!("JPEG/TIFF rejected above; this arm cannot be reached")
            }
        }
    }

    out.flush()?;
    Ok(())
}

/// Convert an RGB bitmap to grayscale using BT.709 luminance coefficients.
fn rgb_to_gray(src: &Bitmap<Rgb8>) -> Bitmap<Gray8> {
    let mut dst: Bitmap<Gray8> = Bitmap::new(src.width, src.height, 1, false);
    for y in 0..src.height {
        let src_row = src.row_bytes(y);
        let dst_row = dst.row_bytes_mut(y);
        for x in 0..src.width as usize {
            // BT.709: Y = 0.2126·R + 0.7152·G + 0.0722·B, rounded to nearest.
            // Integer approximation: (2126·R + 7152·G + 722·B + 5000) / 10000.
            // Coefficients sum to 10000; max numerator = 10000·255 + 5000 = 2 555 000 < u32::MAX.
            let r = u32::from(src_row[x * 3]);
            let g = u32::from(src_row[x * 3 + 1]);
            let b = u32::from(src_row[x * 3 + 2]);
            #[expect(
                clippy::cast_possible_truncation,
                reason = "result ≤ 255 by BT.709 coefficient sum"
            )]
            {
                dst_row[x] = ((2126 * r + 7152 * g + 722 * b + 5000) / 10000) as u8;
            }
        }
    }
    dst
}

/// Threshold a grayscale bitmap to monochrome: < 128 → black (255), ≥ 128 → white (0).
///
/// Matches pdftoppm's `--mono` convention: black ink on white paper.
fn gray_to_mono(src: &Bitmap<Gray8>) -> Bitmap<Gray8> {
    let mut dst: Bitmap<Gray8> = Bitmap::new(src.width, src.height, 1, false);
    for y in 0..src.height {
        let src_row = src.row_bytes(y);
        let dst_row = dst.row_bytes_mut(y);
        for x in 0..src.width as usize {
            // < 128 is dark → black (255); ≥ 128 is light → white (0).
            dst_row[x] = if src_row[x] < 128 { 255 } else { 0 };
        }
    }
    dst
}
