//! Command-line argument definitions.

use clap::Parser;
use pdf_raster::{BackendPolicy, DEFAULT_VAAPI_DEVICE, SessionConfig};

/// Parser that rejects non-positive or non-finite DPI values at the CLI boundary.
fn parse_positive_dpi(s: &str) -> Result<f64, String> {
    let v: f64 = s
        .parse()
        .map_err(|_| format!("'{s}' is not a valid number"))?;
    if v.is_finite() && v > 0.0 {
        Ok(v)
    } else {
        Err(format!("DPI must be a positive finite number, got {s}"))
    }
}

/// Renders PDF pages to images.
#[derive(Parser, Debug)]
#[command(name = "rrocket", about, long_about = None)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "each bool maps to a distinct CLI flag"
)]
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

    /// Print per-page wall time and thread index to stderr for profiling.
    #[arg(long = "timings")]
    pub timings: bool,

    // ── RAM-backed output ─────────────────────────────────────────────────────
    /// Force RAM-backed output even when `OUTPUT_PREFIX` looks like a path.
    ///
    /// By default the program already redirects bare-stem prefixes (e.g.
    /// `rrocket doc.pdf out`) to a freshly-created tmpfs directory under
    /// `/dev/shm`, because writing to disk is 10–20× slower than RAM and
    /// dominates wall time. Path-like prefixes (anything with `/` or
    /// starting with `.`) opt out of that redirect and write to disk.
    ///
    /// Use `--ram` to override the heuristic: redirect to tmpfs even when
    /// the user-supplied prefix would otherwise be treated as a real path.
    /// Mutually exclusive with `--no-ram`.
    #[arg(long = "ram", conflicts_with = "no_ram")]
    pub ram: bool,

    /// Force on-disk output. Disables the default tmpfs redirect for bare
    /// stems and writes pages exactly at `OUTPUT_PREFIX`.
    /// Mutually exclusive with `--ram`.
    #[arg(long = "no-ram", conflicts_with = "ram")]
    pub no_ram: bool,

    /// Override the tmpfs directory used by RAM-backed output. Implies the
    /// RAM redirect. Default is a fresh `/dev/shm/rrocket-<pid>-<nanos>/`.
    #[arg(long = "ram-path", value_name = "PATH", conflicts_with = "no_ram")]
    pub ram_path: Option<String>,

    // ── Backend selection ─────────────────────────────────────────────────────
    /// Compute backend for image decoding and GPU fills.
    ///
    /// `auto`   — Vulkan if compiled in and present, else CUDA, else CPU
    ///            (silent fallback at every step).  This is the default.
    /// `cpu`    — CPU only; all GPU init is skipped.
    /// `cuda`   — Require CUDA (nvJPEG/AA fill/ICC); exit with error if unavailable.
    /// `vaapi`  — Require VA-API JPEG; exit with error if the DRM device cannot be opened.
    /// `vulkan` — Require the Vulkan compute backend.
    ///
    /// When `--backend` is omitted, the `PDF_RASTER_BACKEND` environment
    /// variable is consulted (same valid values).  If neither is set,
    /// the resolved policy is `auto`.
    #[arg(long = "backend", value_name = "BACKEND", verbatim_doc_comment)]
    pub backend: Option<BackendArg>,

    /// VA-API DRM render node (default: /dev/dri/renderD128).
    ///
    /// Only consulted when `--backend vaapi` or `--backend auto` with the
    /// `vaapi` feature compiled in and a VA-API device detected at runtime.
    #[arg(
        long = "vaapi-device",
        value_name = "PATH",
        default_value = DEFAULT_VAAPI_DEVICE
    )]
    pub vaapi_device: String,

    /// Prefetch image `XObject`s into the device-resident image cache at
    /// session open.  Only meaningful when the binary was built with the
    /// `cache` feature; ignored otherwise.  Off by default — the prefetch
    /// walk is wasted work for short single-page renders, but a clear win
    /// on multi-page or multi-pass workloads (OCR pipelines).
    #[cfg(feature = "cache")]
    #[arg(long = "prefetch")]
    pub prefetch: bool,
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
    /// Default thin-line mode.
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

    /// Validate mutually-exclusive output-format flags.
    ///
    /// Returns `Err` with a human-readable message if any incompatible
    /// combination is present.  Call this once at startup.
    pub fn validate_format_flags(&self) -> Result<(), String> {
        if self.jpeg && self.jpegcmyk {
            return Err("--jpeg and --jpegcmyk are mutually exclusive".to_owned());
        }
        if self.mono && self.gray {
            return Err("--mono and --gray are mutually exclusive".to_owned());
        }
        // Count exclusive format flags; at most one may be set.
        let format_count = usize::from(self.png)
            + usize::from(self.jpeg)
            + usize::from(self.jpegcmyk)
            + usize::from(self.tiff);
        if format_count > 1 {
            return Err(
                "at most one output format flag may be used (--png, --jpeg, --jpegcmyk, --tiff)"
                    .to_owned(),
            );
        }
        Ok(())
    }

    /// Resolved output format.
    ///
    /// Assumes [`validate_format_flags`](Self::validate_format_flags) has
    /// already been called — conflicting flags are not possible here.
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

// ── Backend selection ─────────────────────────────────────────────────────────

/// `--backend` argument value parsed by clap.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum BackendArg {
    /// Auto-select: Vulkan if compiled in and present, else CUDA, else CPU.
    /// Silent fallback at every step.  This is the default; when omitted,
    /// `PDF_RASTER_BACKEND` is also consulted.
    Auto,
    /// Skip all GPU init; use CPU paths only.
    Cpu,
    /// Require CUDA — error loudly if CUDA / nvJPEG is unavailable.
    Cuda,
    /// Require VA-API JPEG — error loudly if the DRM device cannot be opened.
    #[cfg(feature = "vaapi")]
    Vaapi,
    /// Require the Vulkan compute backend — error loudly if Vulkan
    /// initialisation fails.  The device-resident image cache is
    /// CUDA-only; under this mode the session runs without it.
    Vulkan,
}

impl Args {
    /// Build a [`SessionConfig`] from the `--backend` / `--vaapi-device` flags.
    ///
    /// Rejects `--vaapi-device` when combined with `--backend cpu` or
    /// `--backend cuda` — those modes never touch the DRM node, so allowing
    /// the combination would silently ignore a user expectation.
    ///
    /// Note: detection is heuristic — we compare against the default path
    /// (`/dev/dri/renderD128`).  A user who explicitly passes that default
    /// value alongside `--backend cpu` will not get the error; this is an
    /// acceptable trade-off given that clap does not expose an `is_present`
    /// flag for options with defaults.
    pub fn session_config(&self) -> Result<SessionConfig, String> {
        // Precedence: explicit `--backend <X>` wins; otherwise consult
        // `PDF_RASTER_BACKEND`; otherwise default to Auto.  Distinguishing
        // "user passed --backend auto" from "user passed nothing" requires
        // `Option<BackendArg>` rather than a clap `default_value` — with
        // a default value the two cases are indistinguishable and the env
        // var would never get a chance to override.
        let policy = match self.backend {
            Some(BackendArg::Auto) => BackendPolicy::Auto,
            Some(BackendArg::Cpu) => BackendPolicy::CpuOnly,
            Some(BackendArg::Cuda) => BackendPolicy::ForceCuda,
            #[cfg(feature = "vaapi")]
            Some(BackendArg::Vaapi) => BackendPolicy::ForceVaapi,
            Some(BackendArg::Vulkan) => BackendPolicy::ForceVulkan,
            None => BackendPolicy::from_env(),
        };

        if self.vaapi_device != DEFAULT_VAAPI_DEVICE
            && matches!(
                policy,
                BackendPolicy::CpuOnly | BackendPolicy::ForceCuda | BackendPolicy::ForceVulkan
            )
        {
            let backend_name = match policy {
                BackendPolicy::CpuOnly => "cpu",
                BackendPolicy::ForceCuda => "cuda",
                BackendPolicy::ForceVulkan => "vulkan",
                _ => unreachable!("matched CpuOnly | ForceCuda | ForceVulkan above"),
            };
            return Err(format!(
                "--vaapi-device has no effect with --backend {backend_name}.\n\
                 VA-API is only used with --backend auto or --backend vaapi."
            ));
        }

        let mut config = SessionConfig::with_policy(policy);
        config.vaapi_device = self.vaapi_device.clone();
        #[cfg(feature = "cache")]
        {
            config.prefetch = self.prefetch;
        }
        Ok(config)
    }

    /// Build the filtered, clamped list of 1-based page numbers to render.
    ///
    /// Returns `Err(message)` if no pages fall within the requested range so the
    /// caller can handle display and exit uniformly.  Clamping warnings are
    /// returned alongside the page list so the caller decides how to display them.
    ///
    /// Precondition: `total >= 1`, validated by the caller before this is invoked.
    pub fn build_page_list(&self, total: i32) -> Result<(Vec<i32>, Vec<String>), String> {
        let requested_first = self.first_page;
        let requested_last = self.last_page.unwrap_or(total);

        let first = requested_first.max(1);
        let last = requested_last.min(total);

        let mut warnings = Vec::new();
        if requested_first < 1 {
            warnings.push(format!("first page {requested_first} < 1; clamped to 1"));
        }
        if requested_last > total {
            warnings.push(format!(
                "last page {requested_last} exceeds document length ({total}); clamped to {total}"
            ));
        }

        if first > last {
            return Err(format!(
                "first page ({first}) is after last page ({last}); nothing to render"
            ));
        }

        let pages: Vec<i32> = (first..=last)
            .filter(|&p| {
                // odd_only and even_only are mutually exclusive; caller enforces this.
                if self.odd_only {
                    p % 2 == 1
                } else if self.even_only {
                    p % 2 == 0
                } else {
                    true
                }
            })
            .take(if self.single_file { 1 } else { usize::MAX })
            .collect();

        if pages.is_empty() {
            return Err("no pages selected by the current filter combination".to_owned());
        }

        Ok((pages, warnings))
    }
}

#[cfg(test)]
mod page_list_tests {
    use super::*;

    fn base_args() -> Args {
        Args::parse_from(["rrocket", "in.pdf", "out"])
    }

    #[test]
    fn all_pages_when_no_filter() {
        let args = base_args();
        let (pages, warnings) = args.build_page_list(3).unwrap();
        assert_eq!(pages, vec![1, 2, 3]);
        assert!(warnings.is_empty());
    }

    #[test]
    fn clamps_first_page_below_one() {
        let mut args = base_args();
        args.first_page = -5;
        let (pages, warnings) = args.build_page_list(3).unwrap();
        assert_eq!(pages[0], 1);
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains("clamped to 1"));
    }

    #[test]
    fn clamps_last_page_beyond_total() {
        let mut args = base_args();
        args.last_page = Some(999);
        let (pages, warnings) = args.build_page_list(3).unwrap();
        assert_eq!(*pages.last().unwrap(), 3);
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains("clamped to 3"));
    }

    #[test]
    fn odd_only_filter() {
        let mut args = base_args();
        args.odd_only = true;
        let (pages, _) = args.build_page_list(5).unwrap();
        assert_eq!(pages, vec![1, 3, 5]);
    }

    #[test]
    fn even_only_filter() {
        let mut args = base_args();
        args.even_only = true;
        let (pages, _) = args.build_page_list(5).unwrap();
        assert_eq!(pages, vec![2, 4]);
    }

    #[test]
    fn err_when_first_after_last() {
        let mut args = base_args();
        args.first_page = 5;
        args.last_page = Some(3);
        let result = args.build_page_list(10);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("nothing to render"));
    }

    #[test]
    fn err_when_no_pages_match_filter() {
        let mut args = base_args();
        args.first_page = 2;
        args.last_page = Some(2);
        args.odd_only = true;
        let result = args.build_page_list(5);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("no pages selected"));
    }

    #[test]
    fn single_file_takes_first_page_only() {
        let mut args = base_args();
        args.single_file = true;
        let (pages, _) = args.build_page_list(10).unwrap();
        assert_eq!(pages, vec![1]);
    }
}
