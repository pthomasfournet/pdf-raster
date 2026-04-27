//! Command-line argument definitions — mirrors all pdftoppm flags.

use clap::Parser;

/// Rust replacement for pdftoppm — renders PDF pages to images.
#[derive(Parser, Debug)]
#[command(name = "pdf-raster", about, long_about = None)]
#[allow(clippy::struct_excessive_bools)]
pub struct Args {
    /// Input PDF file ("-" for stdin).
    pub input: String,

    /// Output filename prefix (page number and extension are appended).
    pub output_prefix: String,

    // ── Page range ──────────────────────────────────────────────────────────
    /// First page to render (1-based, default: 1).
    #[arg(
        short = 'f',
        long = "first-page",
        default_value_t = 1,
        value_name = "N"
    )]
    pub first_page: i32,

    /// Last page to render (1-based, default: last page).
    #[arg(short = 'l', long = "last-page", value_name = "N")]
    pub last_page: Option<i32>,

    /// Render only odd pages.
    #[arg(short = 'o', long = "odd", default_value_t = false)]
    pub odd_only: bool,

    /// Render only even pages.
    #[arg(short = 'e', long = "even", default_value_t = false)]
    pub even_only: bool,

    /// Stop after rendering the first matching page (single-file output).
    #[arg(long = "singlefile", default_value_t = false)]
    pub single_file: bool,

    // ── Resolution / scaling ─────────────────────────────────────────────────
    /// Render resolution in DPI (both axes, default 150).
    #[arg(short = 'r', long = "resolution", value_name = "DPI")]
    pub resolution: Option<f64>,

    /// Horizontal resolution in DPI (overrides -r).
    #[arg(long = "rx", value_name = "DPI")]
    pub resolution_x: Option<f64>,

    /// Vertical resolution in DPI (overrides -r).
    #[arg(long = "ry", value_name = "DPI")]
    pub resolution_y: Option<f64>,

    /// Scale output so the longest edge is this many pixels (preserves aspect).
    #[arg(long = "scale-to", value_name = "PIXELS")]
    pub scale_to: Option<u32>,

    /// Scale output width to this many pixels (overrides --scale-to width).
    #[arg(long = "scale-to-x", value_name = "PIXELS")]
    pub scale_to_x: Option<u32>,

    /// Scale output height to this many pixels (overrides --scale-to height).
    #[arg(long = "scale-to-y", value_name = "PIXELS")]
    pub scale_to_y: Option<u32>,

    // ── Crop ─────────────────────────────────────────────────────────────────
    /// Crop x offset in pixels.
    #[arg(short = 'x', long, value_name = "PIXELS")]
    pub crop_x: Option<i32>,

    /// Crop y offset in pixels.
    #[arg(short = 'y', long, value_name = "PIXELS")]
    pub crop_y: Option<i32>,

    /// Crop width in pixels.
    #[arg(short = 'W', long = "crop-width", value_name = "PIXELS")]
    pub crop_w: Option<u32>,

    /// Crop height in pixels.
    #[arg(short = 'H', long = "crop-height", value_name = "PIXELS")]
    pub crop_h: Option<u32>,

    /// Use the PDF crop box instead of the media box.
    #[arg(long = "cropbox", default_value_t = false)]
    pub use_cropbox: bool,

    // ── Output format ────────────────────────────────────────────────────────
    /// Output PNG (default: PPM).
    #[arg(long, default_value_t = false)]
    pub png: bool,

    /// Output JPEG.
    #[arg(long, default_value_t = false)]
    pub jpeg: bool,

    /// Output JPEG in CMYK colour space.
    #[arg(long, default_value_t = false)]
    pub jpegcmyk: bool,

    /// Output TIFF.
    #[arg(long, default_value_t = false)]
    pub tiff: bool,

    /// Output grey-scale PPM/PNG.
    #[arg(long = "gray", default_value_t = false)]
    pub gray: bool,

    /// Output 1-bit mono PPM/PNG.
    #[arg(long = "mono", default_value_t = false)]
    pub mono: bool,

    /// JPEG quality 0-100 (default 75).
    #[arg(long = "jpegopt", value_name = "QUALITY", default_value_t = 75)]
    pub jpeg_quality: u8,

    // ── Rendering options ────────────────────────────────────────────────────
    /// Enable anti-aliasing (default true).
    #[arg(long = "aa", value_name = "yes|no", default_value = "yes")]
    pub antialias: AaFlag,

    /// Enable vector anti-aliasing (default true).
    #[arg(long = "aaVector", value_name = "yes|no", default_value = "yes")]
    pub vector_antialias: AaFlag,

    /// Hide PDF annotations.
    #[arg(long = "hide-annotations", default_value_t = false)]
    pub hide_annotations: bool,

    /// Enable overprint preview.
    #[arg(long = "overprint", default_value_t = false)]
    pub overprint: bool,

    /// Thin line rendering mode (default, solid, shape).
    #[arg(long = "thinlinemode", value_name = "MODE", default_value = "default")]
    pub thin_line_mode: ThinLineMode,

    // ── Passwords ────────────────────────────────────────────────────────────
    /// Owner password for encrypted PDFs.
    #[arg(long = "opw", value_name = "PASSWORD")]
    pub owner_password: Option<String>,

    /// User password for encrypted PDFs.
    #[arg(long = "upw", value_name = "PASSWORD")]
    pub user_password: Option<String>,

    // ── Parallelism ──────────────────────────────────────────────────────────
    /// Number of threads (0 = auto-detect).
    #[arg(long = "threads", value_name = "N", default_value_t = 0)]
    pub num_threads: usize,

    // ── Output naming ────────────────────────────────────────────────────────
    /// Separator character between prefix and page number (default "-").
    #[arg(long = "sep", value_name = "CHAR", default_value = "-")]
    pub separator: char,

    /// Zero-pad page numbers to at least N digits.
    #[arg(long = "forcenum", value_name = "DIGITS")]
    pub force_num_digits: Option<usize>,

    // ── Progress ─────────────────────────────────────────────────────────────
    /// Print progress to stderr: pages done, elapsed time, and ETA.
    #[arg(short = 'P', long = "progress", default_value_t = false)]
    pub progress: bool,
}

/// Yes/no flag for anti-aliasing options.
#[derive(Clone, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum AaFlag {
    /// Anti-aliasing enabled.
    Yes,
    /// Anti-aliasing disabled.
    No,
}

/// Thin-line rendering mode.
#[derive(Clone, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum ThinLineMode {
    /// Default thin-line mode (matches pdftoppm default).
    Default,
    /// Force solid thin lines.
    Solid,
    /// Use shape-based thin lines.
    Shape,
}

impl Args {
    /// Effective horizontal DPI.
    pub fn x_dpi(&self) -> f64 {
        self.resolution_x.or(self.resolution).unwrap_or(150.0)
    }

    /// Effective vertical DPI.
    pub fn y_dpi(&self) -> f64 {
        self.resolution_y.or(self.resolution).unwrap_or(150.0)
    }

    /// Resolved output format.
    pub const fn output_format(&self) -> OutputFormat {
        if self.png {
            OutputFormat::Png
        } else if self.jpeg || self.jpegcmyk {
            OutputFormat::Jpeg
        } else if self.tiff {
            OutputFormat::Tiff
        } else {
            OutputFormat::Ppm
        }
    }
}

/// Output image format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputFormat {
    /// Raw PPM/PGM (default).
    Ppm,
    /// PNG.
    Png,
    /// JPEG.
    Jpeg,
    /// TIFF.
    Tiff,
}

impl OutputFormat {
    /// File extension for this format.
    pub const fn extension(self) -> &'static str {
        match self {
            Self::Ppm => "ppm",
            Self::Png => "png",
            Self::Jpeg => "jpg",
            Self::Tiff => "tif",
        }
    }
}
