//! Command-line argument definitions — mirrors all pdftoppm flags.

use clap::Parser;

/// Parser that rejects non-positive or non-finite DPI values at the CLI boundary.
fn parse_positive_dpi(s: &str) -> Result<f64, String> {
    let v: f64 = s.parse().map_err(|_| format!("'{s}' is not a valid number"))?;
    if v.is_finite() && v > 0.0 {
        Ok(v)
    } else {
        Err(format!("DPI must be a positive finite number, got {s}"))
    }
}

/// Rust replacement for pdftoppm — renders PDF pages to images.
#[derive(Parser, Debug)]
#[command(name = "pdf-raster", about, long_about = None)]
#[expect(clippy::struct_excessive_bools, reason = "CLI mirrors all pdftoppm flags; each bool maps to a distinct flag")]
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
    #[arg(short = 'o', long = "odd")]
    pub odd_only: bool,

    /// Render only even pages.
    #[arg(short = 'e', long = "even")]
    pub even_only: bool,

    /// Stop after rendering the first matching page (single-file output).
    #[arg(long = "singlefile")]
    pub single_file: bool,

    // ── Resolution / scaling ─────────────────────────────────────────────────
    /// Render resolution in DPI (both axes, default 150).
    #[arg(
        short = 'r',
        long = "resolution",
        value_name = "DPI",
        value_parser = parse_positive_dpi
    )]
    pub resolution: Option<f64>,

    /// Horizontal resolution in DPI (overrides -r).
    #[arg(long = "rx", value_name = "DPI", value_parser = parse_positive_dpi)]
    pub resolution_x: Option<f64>,

    /// Vertical resolution in DPI (overrides -r).
    #[arg(long = "ry", value_name = "DPI", value_parser = parse_positive_dpi)]
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
    #[arg(long = "cropbox")]
    pub use_cropbox: bool,

    // ── Output format ────────────────────────────────────────────────────────
    /// Output PNG (default: PPM).
    #[arg(long)]
    pub png: bool,

    /// Output JPEG.
    #[arg(long)]
    pub jpeg: bool,

    /// Output JPEG in CMYK colour space.
    #[arg(long)]
    pub jpegcmyk: bool,

    /// Output TIFF.
    #[arg(long)]
    pub tiff: bool,

    /// Output grey-scale PPM/PNG.
    #[arg(long = "gray")]
    pub gray: bool,

    /// Output 1-bit mono PPM/PNG.
    #[arg(long = "mono")]
    pub mono: bool,

    /// JPEG quality 0-100 (default 75).
    #[arg(
        long = "jpegopt",
        value_name = "QUALITY",
        default_value_t = 75,
        value_parser = clap::value_parser!(u8).range(0..=100)
    )]
    pub jpeg_quality: u8,

    // ── Rendering options ────────────────────────────────────────────────────
    /// Enable anti-aliasing (default true).
    #[arg(long = "aa", value_name = "yes|no", default_value = "yes")]
    pub antialias: AaFlag,

    /// Enable vector anti-aliasing (default true).
    #[arg(long = "aaVector", value_name = "yes|no", default_value = "yes")]
    pub vector_antialias: AaFlag,

    /// Hide PDF annotations.
    #[arg(long = "hide-annotations")]
    pub hide_annotations: bool,

    /// Enable overprint preview.
    #[arg(long = "overprint")]
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
    #[arg(short = 'P', long = "progress")]
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

impl std::fmt::Display for OutputFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Ppm => f.write_str("PPM"),
            Self::Png => f.write_str("PNG"),
            Self::Jpeg => f.write_str("JPEG"),
            Self::Tiff => f.write_str("TIFF"),
        }
    }
}

impl OutputFormat {
    /// File extension for this format, taking `--gray` / `--mono` into account.
    ///
    /// - PPM + mono → `.pbm` (P4 binary Netpbm, 1-bit)
    /// - PPM + gray → `.pgm` (P5 grayscale Netpbm, 8-bit)
    /// - PNG + gray/mono → `.png` (grayscale PNG)
    /// - All other combinations use the format's natural extension.
    pub const fn extension_with_mode(self, gray: bool, mono: bool) -> &'static str {
        match (self, mono, gray) {
            (Self::Ppm, true, _) => "pbm",
            (Self::Ppm, false, true) => "pgm",
            (Self::Ppm, false, false) => "ppm",
            (Self::Png, _, _) => "png",
            (Self::Jpeg, _, _) => "jpg",
            (Self::Tiff, _, _) => "tif",
        }
    }
}
