//! Per-page rendering: native Rust renderer → pixel buffer → output file.

#[cfg(any(feature = "nvjpeg", feature = "nvjpeg2k"))]
use std::cell::RefCell;
use std::fs::File;
use std::io::{BufWriter, Write as _};
#[cfg(any(feature = "gpu-aa", feature = "gpu-icc"))]
use std::sync::Arc;

use color::{Gray8, Rgb8};
use encode::{EncodeError, write_pbm, write_pgm, write_png, write_ppm};
use raster::Bitmap;

use crate::args::{Args, OutputFormat};
use crate::naming::output_path;

// ── Per-thread GPU image decoders ─────────────────────────────────────────────
//
// NvJpegDecoder and NvJpeg2kDecoder are Send but not Sync: each rayon worker
// thread owns exactly one instance, created lazily on first use.  DecoderInit<T>
// tracks whether construction has been attempted so a one-time failure does not
// retry (and re-log) on every subsequent page.  RefCell gives interior mutability
// without a Mutex; the thread_local guarantee prevents concurrent access.

/// Initialisation state for a per-thread GPU decoder.
///
/// Tracks whether initialisation has been attempted so that a failed attempt
/// does not retry (and re-log) on every subsequent page render.
#[cfg(any(feature = "nvjpeg", feature = "nvjpeg2k"))]
#[derive(Default)]
enum DecoderInit<T> {
    /// `new()` has not been called yet on this thread.
    #[default]
    Uninitialised,
    /// `new()` succeeded; the decoder is ready (or temporarily moved out).
    Ready(Option<T>),
    /// `new()` failed; do not retry.
    Failed,
}

#[cfg(feature = "nvjpeg")]
thread_local! {
    static NVJPEG_DEC: RefCell<DecoderInit<gpu::nvjpeg::NvJpegDecoder>> =
        const { RefCell::new(DecoderInit::Uninitialised) };
}

#[cfg(feature = "nvjpeg2k")]
thread_local! {
    static NVJPEG2K_DEC: RefCell<DecoderInit<gpu::nvjpeg2k::NvJpeg2kDecoder>> =
        const { RefCell::new(DecoderInit::Uninitialised) };
}

/// Ensure the per-thread nvJPEG decoder is initialised.
///
/// On first call: attempts `NvJpegDecoder::new(0)`.  On success, stores the
/// decoder.  On failure, logs once to stderr and marks the slot `Failed` so
/// subsequent pages skip the attempt silently.
#[cfg(feature = "nvjpeg")]
fn init_nvjpeg() {
    NVJPEG_DEC.with(|cell| {
        let mut slot = cell.borrow_mut();
        if matches!(*slot, DecoderInit::Uninitialised) {
            match gpu::nvjpeg::NvJpegDecoder::new(0) {
                Ok(dec) => *slot = DecoderInit::Ready(Some(dec)),
                Err(e) => {
                    eprintln!(
                        "pdf-raster: nvJPEG unavailable ({e}); \
                         JPEG images will be decoded on CPU for this thread"
                    );
                    *slot = DecoderInit::Failed;
                }
            }
        }
    });
}

/// Ensure the per-thread nvJPEG2000 decoder is initialised.
///
/// On first call: attempts `NvJpeg2kDecoder::new(0)`.  On success, stores the
/// decoder.  On failure, logs once to stderr and marks the slot `Failed` so
/// subsequent pages skip the attempt silently.
#[cfg(feature = "nvjpeg2k")]
fn init_nvjpeg2k() {
    NVJPEG2K_DEC.with(|cell| {
        let mut slot = cell.borrow_mut();
        if matches!(*slot, DecoderInit::Uninitialised) {
            match gpu::nvjpeg2k::NvJpeg2kDecoder::new(0) {
                Ok(dec) => *slot = DecoderInit::Ready(Some(dec)),
                Err(e) => {
                    eprintln!(
                        "pdf-raster: nvJPEG2000 unavailable ({e}); \
                         JPEG 2000 images will be decoded on CPU for this thread"
                    );
                    *slot = DecoderInit::Failed;
                }
            }
        }
    });
}

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
/// GPU acceleration is enabled transparently when the relevant feature flags
/// are compiled in:
/// - `gpu-aa`, `gpu-icc`: shared [`gpu::GpuCtx`] passed via `gpu_ctx`
/// - `nvjpeg`: per-thread [`gpu::nvjpeg::NvJpegDecoder`] for `DCTDecode` images
/// - `nvjpeg2k`: per-thread [`gpu::nvjpeg2k::NvJpeg2kDecoder`] for `JPXDecode` images
///
/// All GPU paths degrade gracefully to CPU when the GPU is unavailable.
///
/// `--gray` converts the RGB bitmap to grayscale (BT.709) and writes PGM/gray PNG.
/// `--mono` additionally thresholds to 1-bit and writes PBM/gray PNG.
pub fn render_page_native(
    doc: &lopdf::Document,
    page_num: u32,
    total_pages: u32,
    args: &Args,
    #[cfg(any(feature = "gpu-aa", feature = "gpu-icc"))] gpu_ctx: Option<&Arc<gpu::GpuCtx>>,
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
    #[cfg(any(feature = "gpu-aa", feature = "gpu-icc"))]
    renderer.set_gpu_ctx(gpu_ctx.map(Arc::clone));

    // Attach per-thread GPU image decoders.  init_* attempts construction once
    // per thread; subsequent calls are no-ops.  borrow_mut() cannot panic: rayon
    // tasks are single-threaded — no concurrent borrow of the same thread-local
    // is possible within one task invocation.
    #[cfg(feature = "nvjpeg")]
    {
        init_nvjpeg();
        NVJPEG_DEC.with(|cell| {
            if let DecoderInit::Ready(slot) = &mut *cell.borrow_mut() {
                renderer.set_nvjpeg(slot.take());
            }
        });
    }
    #[cfg(feature = "nvjpeg2k")]
    {
        init_nvjpeg2k();
        NVJPEG2K_DEC.with(|cell| {
            if let DecoderInit::Ready(slot) = &mut *cell.borrow_mut() {
                renderer.set_nvjpeg2k(slot.take());
            }
        });
    }

    renderer.execute(&ops);
    renderer.render_annotations(page_id);

    // Return decoders to the thread-local slots so they survive across pages.
    // take_nvjpeg / take_nvjpeg2k return None if the decoder was never attached
    // (e.g. init failed); in that case the slot stays Failed and is not touched.
    #[cfg(feature = "nvjpeg")]
    NVJPEG_DEC.with(|cell| {
        if let DecoderInit::Ready(slot) = &mut *cell.borrow_mut() {
            *slot = renderer.take_nvjpeg();
        }
    });
    #[cfg(feature = "nvjpeg2k")]
    NVJPEG2K_DEC.with(|cell| {
        if let DecoderInit::Ready(slot) = &mut *cell.borrow_mut() {
            *slot = renderer.take_nvjpeg2k();
        }
    });

    let rgb: Bitmap<Rgb8> = renderer.finish();

    #[expect(
        clippy::cast_possible_wrap,
        reason = "total_pages ≤ i32::MAX; validated in main before calling this function"
    )]
    let out_path = output_path(args, page_num as i32, total_pages as i32, format);
    let mut out = BufWriter::new(File::create(&out_path)?);

    // JPEG/TIFF are rejected before reaching this point; the arms below are
    // exhaustive but the Jpeg/Tiff variants cannot be reached.
    if args.mono {
        let mono = gray_to_mono(&rgb_to_gray(&rgb));
        match format {
            OutputFormat::Ppm => write_pbm::<Gray8, _>(&mono, &mut out)?,
            OutputFormat::Png => write_png::<Gray8, _>(&mono, &mut out)?,
            OutputFormat::Jpeg | OutputFormat::Tiff => {
                unreachable!("JPEG/TIFF rejected above")
            }
        }
    } else if args.gray {
        let gray = rgb_to_gray(&rgb);
        match format {
            OutputFormat::Ppm => write_pgm::<Gray8, _>(&gray, &mut out)?,
            OutputFormat::Png => write_png::<Gray8, _>(&gray, &mut out)?,
            OutputFormat::Jpeg | OutputFormat::Tiff => {
                unreachable!("JPEG/TIFF rejected above")
            }
        }
    } else {
        match format {
            OutputFormat::Ppm => write_ppm(&rgb, &mut out)?,
            OutputFormat::Png => write_png(&rgb, &mut out)?,
            OutputFormat::Jpeg | OutputFormat::Tiff => {
                unreachable!("JPEG/TIFF rejected above")
            }
        }
    }

    out.flush()?;
    Ok(())
}

/// Convert an RGB bitmap to grayscale using BT.709 luminance coefficients.
///
/// Output uses the same 0 = black, 255 = white convention as the input.
fn rgb_to_gray(src: &Bitmap<Rgb8>) -> Bitmap<Gray8> {
    let mut dst: Bitmap<Gray8> = Bitmap::new(src.width, src.height, 1, false);
    let w = src.width as usize; // width ≤ MAX_PX_DIMENSION (32 768) — fits usize on all targets
    for y in 0..src.height {
        // row_bytes() returns the full stride; take only the live pixels (width × 3 bytes).
        let src_pixels = &src.row_bytes(y)[..w * 3];
        let dst_row = dst.row_bytes_mut(y);
        for (dst_px, rgb) in dst_row[..w].iter_mut().zip(src_pixels.chunks_exact(3)) {
            // BT.709: Y = 0.2126·R + 0.7152·G + 0.0722·B, rounded to nearest.
            // Integer: (2126·R + 7152·G + 722·B + 5000) / 10000.
            // Coefficients sum to 10000; max numerator = 10000·255 + 5000 = 2 555 000 < u32::MAX.
            let (r, g, b) = (u32::from(rgb[0]), u32::from(rgb[1]), u32::from(rgb[2]));
            #[expect(
                clippy::cast_possible_truncation,
                reason = "sum ≤ 255 by BT.709 coefficient identity"
            )]
            {
                *dst_px = ((2126 * r + 7152 * g + 722 * b + 5000) / 10000) as u8;
            }
        }
    }
    dst
}

/// Threshold a grayscale bitmap to 2-level monochrome at the 50% midpoint.
///
/// Values < 128 (dark) → 255 (black ink); values ≥ 128 (light) → 0 (white paper).
/// This matches pdftoppm's `--mono` output convention.
fn gray_to_mono(src: &Bitmap<Gray8>) -> Bitmap<Gray8> {
    let mut dst: Bitmap<Gray8> = Bitmap::new(src.width, src.height, 1, false);
    let w = src.width as usize; // width ≤ MAX_PX_DIMENSION — fits usize
    for y in 0..src.height {
        let src_row = &src.row_bytes(y)[..w];
        let dst_row = &mut dst.row_bytes_mut(y)[..w];
        for (dst_px, &luma) in dst_row.iter_mut().zip(src_row.iter()) {
            *dst_px = if luma < 128 { 255 } else { 0 };
        }
    }
    dst
}
