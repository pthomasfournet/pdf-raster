//! PDF rasterisation library — zero subprocess, zero Leptonica.
//!
//! Converts PDF pages to 8-bit grayscale pixel buffers ready for Tesseract OCR
//! (or any other consumer) without writing files or spawning processes.
//!
//! # Quick start
//!
//! ```rust,no_run
//! use std::path::Path;
//! use pdf_raster::{RasterOptions, raster_pdf};
//!
//! let opts = RasterOptions { dpi: 300.0, first_page: 1, last_page: 5, deskew: true, pages: None };
//! for (page_num, result) in raster_pdf(Path::new("scan.pdf"), &opts) {
//!     match result {
//!         Ok(page) => {
//!             // page.pixels: Vec<u8>, 8-bit grayscale, width × height, top-to-bottom
//!             // pass to tesseract::ocr_from_frame(&page.pixels, page.width, page.height, 1, page.width, "eng")
//!         }
//!         Err(e) => eprintln!("page {page_num}: {e}"),
//!     }
//! }
//! ```
//!
//! # Integration with Tesseract
//!
//! ```rust,no_run
//! # use pdf_raster::{RasterOptions, raster_pdf};
//! # use std::path::Path;
//! # let opts = RasterOptions { dpi: 300.0, first_page: 1, last_page: 1, deskew: true, pages: None };
//! for (_, result) in raster_pdf(Path::new("scan.pdf"), &opts) {
//!     let page = result.expect("page render failed");
//!     // tesseract crate (v0.15+):
//!     // let text = tesseract::ocr_from_frame(
//!     //     &page.pixels,
//!     //     page.width as i32,
//!     //     page.height as i32,
//!     //     1,               // bytes_per_pixel (grayscale)
//!     //     page.width as i32, // bytes_per_line (tightly packed)
//!     //     "eng",
//!     // ).unwrap();
//!     // For uneven scan backgrounds: set thresholding_method=2 (Sauvola) on the Tesseract side.
//!     // Do NOT pre-binarise — the LSTM engine reads your grayscale values directly.
//! }
//! ```
//!
//! # Streaming: render and process pages in parallel
//!
//! For large documents, [`render_channel`] renders on a background rayon thread
//! and sends pages as they complete, so downstream work (e.g. OCR) can start
//! immediately rather than waiting for the full render:
//!
//! ```rust,no_run
//! use std::path::Path;
//! use pdf_raster::{RasterOptions, render_channel};
//!
//! // `rayon` is a transitive dependency of `pdf_raster`; add it explicitly
//! // to your Cargo.toml only if you call rayon APIs directly: rayon = "1"
//! //
//! // The consumer `recv` loop MUST run on the calling thread (not inside a
//! // `rayon::scope` or a worker closure): `render_channel`'s producer also
//! // runs on rayon's global pool, so a recv loop inside a scope task could
//! // deadlock if the pool fills with scope tasks all blocked on `recv`.
//! //
//! // `rayon::spawn` *inside* the recv-arm is safe — those tasks return
//! // without blocking on recv themselves.  For non-CPU consumer work
//! // (Tesseract OCR, disk I/O), prefer `std::thread::spawn` to keep the
//! // rayon pool free for the renderer.
//! let opts = RasterOptions {
//!     dpi: 300.0,
//!     first_page: 1,
//!     last_page: u32::MAX,
//!     deskew: true,
//!     pages: None,
//! };
//! let rx = render_channel(Path::new("scan.pdf"), &opts, 4); // 4-page buffer
//! while let Ok((page_num, result)) = rx.recv() {
//!     match result {
//!         Ok(page) => rayon::spawn(move || {
//!             // process page — overlaps with the next render
//!             let _ = (page_num, page.pixels);
//!         }),
//!         Err(e) => eprintln!("page {page_num}: {e}"),
//!     }
//! }
//! // Note: rayon::spawn tasks may still be running here. Use a rayon scope
//! // or channel to join them if you need to wait for all work to complete.
//! ```
//!
//! # Sparse page selection
//!
//! To render only specific pages (e.g. a subset identified by a prior scan),
//! use [`PageSet`]:
//!
//! ```rust,no_run
//! use std::path::Path;
//! use pdf_raster::{PageSet, RasterOptions, raster_pdf};
//!
//! let pages = PageSet::new(vec![1, 5, 23, 100]).unwrap();
//! let opts = RasterOptions {
//!     dpi: 300.0,
//!     first_page: 1,        // ignored when pages is Some
//!     last_page: u32::MAX,  // ignored when pages is Some
//!     deskew: true,
//!     pages: Some(pages),
//! };
//! for (page_num, result) in raster_pdf(Path::new("scan.pdf"), &opts) {
//!     // Only pages 1, 5, 23, and 100 are rendered — intermediates are skipped.
//!     match result {
//!         Ok(page) => { /* process page.pixels */ let _ = (page_num, page); }
//!         Err(e) => eprintln!("page {page_num}: {e}"),
//!     }
//! }
//! ```

#[cfg(feature = "vaapi")]
pub(crate) mod decode_queue;
pub mod deskew;
pub(crate) mod gpu_init;
mod render;

use std::path::Path;

pub use pdf_interp::prescan_page;
pub use pdf_interp::renderer::PageDiagnostics;
pub use pdf_interp::resources::ImageFilter;
pub use render::{
    MAX_PX_DIMENSION, RasterError, RasterSession, open_session, render_page_rgb,
    render_page_rgb_hinted, rgb_to_gray,
};

/// Eagerly release GPU decoders on every rayon worker thread.
///
/// Call this via `pool.broadcast(|_| pdf_raster::release_gpu_decoders())`
/// after all rendering is done and before `pool` is dropped.  This drops each
/// thread's `NvJpegDecoder` while the CUDA driver is still fully live, avoiding
/// the process-exit teardown race where all workers call `nvjpegJpegStateDestroy`
/// concurrently into a driver that has already started its own atexit shutdown.
///
/// After this call the TLS slots hold `Uninitialised`, so their own destructors
/// at process exit are no-ops.
#[cfg_attr(
    not(any(
        feature = "nvjpeg",
        feature = "nvjpeg2k",
        all(feature = "gpu-jpeg-huffman", feature = "vulkan"),
    )),
    expect(
        clippy::missing_const_for_fn,
        reason = "body collapses to empty when no GPU-decoder feature is on"
    )
)]
pub fn release_gpu_decoders() {
    #[cfg(feature = "nvjpeg")]
    gpu_init::release_nvjpeg_this_thread();
    #[cfg(feature = "nvjpeg2k")]
    gpu_init::release_nvjpeg2k_this_thread();
    #[cfg(all(feature = "gpu-jpeg-huffman", feature = "vulkan"))]
    gpu_init::release_jpeg_vk_this_thread();
}

// ── Backend policy ────────────────────────────────────────────────────────────

/// Controls which compute backend is used for image decoding and GPU fills.
///
/// The default is [`Auto`](BackendPolicy::Auto), which prefers Vulkan when
/// compiled in, falls through to CUDA when Vulkan is unavailable, and finally
/// to the CPU paths.  The `Force*` variants turn silent fallbacks into hard
/// errors so you can tell immediately whether the expected hardware path is
/// actually being taken.
///
/// At runtime, [`SessionConfig::default()`] resolves the policy via
/// [`BackendPolicy::from_env`], so users can switch backends per process by
/// setting `PDF_RASTER_BACKEND={auto,cpu,cuda,vaapi,vulkan}` without
/// rebuilding.  Explicit construction (`SessionConfig { policy: ..., .. }`)
/// bypasses the env var.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BackendPolicy {
    /// Auto: prefer Vulkan when present, fall through to CUDA, then CPU.
    /// Silent fallback at every step.
    #[default]
    Auto,
    /// CPU only.  All GPU init is skipped; no CUDA or VA-API calls are made.
    CpuOnly,
    /// Require CUDA (nvJPEG, nvJPEG2000, GPU AA fill, ICC CLUT).
    /// Returns [`RasterError::BackendUnavailable`] if CUDA initialisation fails
    /// rather than falling back to CPU.
    ForceCuda,
    /// Require VA-API JPEG decoding.
    /// Returns [`RasterError::BackendUnavailable`] if the VA-API device cannot
    /// be opened rather than falling back to CPU.
    ForceVaapi,
    /// Require the Vulkan compute backend.  Returns
    /// [`RasterError::BackendUnavailable`] if Vulkan initialisation fails
    /// (or the binary was built without `--features vulkan`).
    ///
    /// AA fill and tile fill kernels dispatch through `VulkanBackend`;
    /// the device-resident image cache (`DeviceImageCache`,
    /// `DevicePageBuffer`) is CUDA-only, so under `ForceVulkan` JPEG
    /// images decode and composite via the CPU path, and ICC CMYK→RGB
    /// also stays on the CPU AVX-512 fallback.  This matches Phase 9-pre
    /// behaviour where the cache wasn't yet present.
    ForceVulkan,
}

impl BackendPolicy {
    /// Resolve a policy from the `PDF_RASTER_BACKEND` environment variable.
    ///
    /// Convenience wrapper for [`BackendPolicy::from_env_var`] with the
    /// canonical name.  Accepted values (case-insensitive): `auto`, `cpu`,
    /// `cuda`, `vaapi`, `vulkan`.  Unset, empty, or unrecognised values
    /// fall back to [`BackendPolicy::Auto`] — unrecognised values also
    /// emit a stderr warning so a typo doesn't silently mis-route.
    ///
    /// Precedence: an explicit `--backend` flag (CLI) wins over
    /// `PDF_RASTER_BACKEND`, which in turn wins over the compile-time
    /// default.  This lets a binary ship Vulkan-default while users
    /// override per-process without recompiling.
    #[must_use]
    pub fn from_env() -> Self {
        Self::from_env_var("PDF_RASTER_BACKEND")
    }

    /// Resolve a policy from a named environment variable.
    ///
    /// Used by tools (e.g. the contest harness) that want their own
    /// env-var namespace separate from `PDF_RASTER_BACKEND` so a
    /// developer's personal default doesn't leak into bench runs.
    /// Same value vocabulary and warning semantics as
    /// [`BackendPolicy::from_env`].
    ///
    /// Whitespace around the value is trimmed so `PDF_RASTER_BACKEND=" cuda "`
    /// (e.g. quoted in a shell config that didn't strip) still resolves.
    /// Matching is case-insensitive.  Unrecognised values emit a
    /// `log::warn!` (not stderr) so library embedders control the sink.
    #[must_use]
    pub fn from_env_var(name: &str) -> Self {
        // Unset / empty / unreadable env var → Auto with no allocation.
        let Some(raw) = std::env::var_os(name) else {
            return Self::Auto;
        };
        let Some(s) = raw.to_str() else {
            log::warn!("{name} contains non-UTF-8 bytes; using Auto");
            return Self::Auto;
        };
        s.parse().unwrap_or_else(|()| {
            log::warn!(
                "{name}={s:?} not recognised; using Auto. \
                 Valid values: auto, cpu, cuda, vaapi, vulkan."
            );
            Self::Auto
        })
    }
}

impl std::str::FromStr for BackendPolicy {
    type Err = ();

    /// Parse a backend name.  Whitespace is trimmed and matching is
    /// case-insensitive.  Empty input maps to `Auto`.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "" | "auto" => Ok(Self::Auto),
            "cpu" => Ok(Self::CpuOnly),
            "cuda" => Ok(Self::ForceCuda),
            "vaapi" => Ok(Self::ForceVaapi),
            "vulkan" => Ok(Self::ForceVulkan),
            _ => Err(()),
        }
    }
}

// ── Session configuration ─────────────────────────────────────────────────────

/// Default VA-API DRM render node path used by [`SessionConfig`] and the CLI.
pub const DEFAULT_VAAPI_DEVICE: &str = "/dev/dri/renderD128";

/// Configuration for opening a [`RasterSession`].
///
/// Passed to [`open_session`].  Use [`Default::default()`] for the standard
/// behaviour: backend resolved from `PDF_RASTER_BACKEND` env var (or `Auto`
/// if unset), default DRM render node, image-cache prefetch off.
#[derive(Debug, Clone)]
pub struct SessionConfig {
    /// Backend selection policy.
    pub policy: BackendPolicy,
    /// VA-API DRM render node.  Default: `/dev/dri/renderD128`.
    ///
    /// Only relevant when the `vaapi` feature is enabled and `policy` is not
    /// [`CpuOnly`](BackendPolicy::CpuOnly) or [`ForceCuda`](BackendPolicy::ForceCuda).
    pub vaapi_device: String,
    /// Whether to spawn the image-cache prefetcher at session open.
    /// Default `false` — opt-in because the prefetcher reads every
    /// page's resource dict eagerly, which is wasted work for short
    /// single-page renders.  Long renders, multi-pass renders (OCR
    /// pipelines), and re-renders of the same PDF benefit.
    ///
    /// No effect when the `cache` feature is disabled or when GPU
    /// initialisation fails (no cache → nowhere to prefetch into).
    #[cfg(feature = "cache")]
    pub prefetch: bool,
}

impl SessionConfig {
    /// Build a config with the given backend policy and otherwise-default
    /// fields, **without** consulting `PDF_RASTER_BACKEND`.
    ///
    /// Use this when you have already resolved the policy from a more
    /// specific source (a CLI flag, a different env var, an explicit
    /// argument) and don't want `Default::default()`'s env-var lookup
    /// firing — which would emit a spurious warning if the env var were
    /// set to a malformed value, even though that value is about to be
    /// overwritten anyway.
    #[must_use]
    pub fn with_policy(policy: BackendPolicy) -> Self {
        Self {
            policy,
            vaapi_device: DEFAULT_VAAPI_DEVICE.to_owned(),
            #[cfg(feature = "cache")]
            prefetch: false,
        }
    }
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self::with_policy(BackendPolicy::from_env())
    }
}

// ── PageSet ───────────────────────────────────────────────────────────────────

/// A validated, sorted, deduplicated set of 1-based page numbers.
///
/// Constructed via [`PageSet::new`].  Clone is O(1) — the underlying storage is
/// reference-counted.  Use as the [`RasterOptions::pages`] field to render a
/// sparse subset of pages without visiting intermediate ones.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PageSet(std::sync::Arc<[u32]>);

impl PageSet {
    /// Construct a `PageSet` from any collection of 1-based page numbers.
    ///
    /// Accepts `Vec<u32>`, arrays, slices (via `.to_vec()`), and any
    /// `IntoIterator<Item = u32>`.  The input is sorted and deduplicated
    /// before storage.
    ///
    /// # Errors
    ///
    /// Returns [`RasterError::InvalidOptions`] if the resulting set is empty
    /// (all-zeros input counts) or any value is 0.
    pub fn new(pages: impl IntoIterator<Item = u32>) -> Result<Self, RasterError> {
        let mut v: Vec<u32> = pages.into_iter().collect();
        v.sort_unstable();
        v.dedup();
        if v.is_empty() {
            return Err(RasterError::InvalidOptions(
                "PageSet must contain at least one page".to_owned(),
            ));
        }
        if v[0] == 0 {
            return Err(RasterError::InvalidOptions(
                "PageSet contains page 0 — pages are 1-based".to_owned(),
            ));
        }
        Ok(Self(v.into_boxed_slice().into()))
    }

    /// Returns `true` if `page` is in this set.  O(log n).
    #[must_use]
    pub fn contains(&self, page: u32) -> bool {
        self.0.binary_search(&page).is_ok()
    }

    /// The smallest page number in the set.
    ///
    /// # Panics
    ///
    /// Never panics on a correctly constructed `PageSet` — non-emptiness is a
    /// structural invariant enforced by [`PageSet::new`].
    #[must_use]
    pub fn first(&self) -> u32 {
        *self.0.first().expect("PageSet is non-empty by invariant")
    }

    /// The largest page number in the set.
    ///
    /// # Panics
    ///
    /// Never panics on a correctly constructed `PageSet` — non-emptiness is a
    /// structural invariant enforced by [`PageSet::new`].
    #[must_use]
    pub fn last(&self) -> u32 {
        *self.0.last().expect("PageSet is non-empty by invariant")
    }

    /// Number of pages in the set.
    #[must_use]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Iterate the pages in ascending order.
    pub fn iter(&self) -> std::iter::Copied<std::slice::Iter<'_, u32>> {
        self.0.iter().copied()
    }

    /// View the pages as a sorted, deduplicated slice.  O(1).
    #[must_use]
    pub fn as_slice(&self) -> &[u32] {
        &self.0
    }

    /// Always returns `false`.  A `PageSet` is guaranteed non-empty by
    /// construction (see [`PageSet::new`]); this method exists to satisfy the
    /// `clippy::len_without_is_empty` lint.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        false
    }
}

impl<'a> IntoIterator for &'a PageSet {
    type Item = u32;
    type IntoIter = std::iter::Copied<std::slice::Iter<'a, u32>>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

// ── Render options & output ───────────────────────────────────────────────────

/// Options controlling how pages are rendered.
#[derive(Debug, Clone)]
pub struct RasterOptions {
    /// Must be > 0.  Pass this same value to Tesseract's `set_source_resolution`
    /// — lying about DPI degrades OCR accuracy because Tesseract uses it for
    /// internal scaling.  Recommended: 300 DPI for scanned documents.
    pub dpi: f32,

    /// First page (1-based, inclusive).  Must be ≥ 1.  Ignored when
    /// [`pages`](Self::pages) is `Some` — the render window is then derived
    /// from the `PageSet` bounds.
    pub first_page: u32,

    /// Last page (1-based, inclusive).  Must be ≥ `first_page`.  If it exceeds
    /// the document's page count, rendering stops at the last page in the
    /// document rather than returning an error.  Ignored when
    /// [`pages`](Self::pages) is `Some`.
    pub last_page: u32,

    /// Uses an intensity-weighted projection-profile sweep (no binarisation
    /// threshold) with GPU bilinear rotation via CUDA NPP (`gpu-deskew` feature)
    /// or CPU bilinear fallback.  Corrects skew up to ±7° with sub-0.05°
    /// accuracy.  Disable for native-text PDFs that are never physically skewed.
    pub deskew: bool,

    /// Sparse page selection.  When `Some`, only the pages in the [`PageSet`]
    /// are rendered and yielded; intermediate pages are skipped without
    /// rendering.  When `None`, all pages in `first_page..=last_page` are
    /// rendered.
    pub pages: Option<PageSet>,
}

impl Default for RasterOptions {
    fn default() -> Self {
        Self {
            dpi: 300.0,
            first_page: 1,
            last_page: u32::MAX,
            deskew: false,
            pages: None,
        }
    }
}

/// A single rendered page, returned as 8-bit grayscale pixels.
#[derive(Debug)]
pub struct RenderedPage {
    /// Page number (1-based).
    pub page_num: u32,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Raw pixel bytes: 8-bit grayscale, tightly packed (`stride == width`),
    /// top-to-bottom, left-to-right.  Length is `width * height`.
    pub pixels: Vec<u8>,
    /// The DPI at which this page was rendered (`opts.dpi`).
    ///
    /// This is the raw render resolution, ignoring any `UserUnit` scaling in the
    /// document.  Use [`effective_dpi`](Self::effective_dpi) — not this field — when
    /// reporting resolution to downstream consumers such as Tesseract.
    pub dpi: f32,
    /// Physical resolution of the rendered bitmap, accounting for `UserUnit` scaling.
    ///
    /// `effective_dpi = opts.dpi × UserUnit`.  For the vast majority of documents
    /// `UserUnit` is 1.0 and this equals `dpi`.  Always pass this value to
    /// `tesseract::set_source_resolution`; lying about DPI degrades OCR accuracy
    /// because Tesseract uses it for internal feature scaling.
    ///
    /// Omitting the resolution call causes Tesseract to fall back to 70 DPI, which
    /// severely degrades recognition accuracy.
    pub effective_dpi: f32,
    /// Lightweight metadata collected at zero extra cost during rendering.
    ///
    /// Use this to route pages to different OCR configurations — e.g. skip deskew
    /// on `has_vector_text = true` pages, or set Tesseract PSM based on `is_scan`.
    pub diagnostics: PageDiagnostics,
}

// ── Top-level render entry points ─────────────────────────────────────────────

/// Render a range of pages from a PDF file.
///
/// Returns an iterator yielding `(page_num, Result<RenderedPage, RasterError>)`
/// for each page in `opts.first_page..=opts.last_page`.  A per-page error does
/// not abort remaining pages — the caller decides whether to skip or propagate.
///
/// GPU resources are initialised lazily on first use and reused across pages.
/// Backend selection follows [`SessionConfig::default`] — Vulkan when present,
/// CUDA next, CPU last; the `PDF_RASTER_BACKEND` env var overrides the
/// default.  Use [`open_session`] + [`render_page_rgb`] directly when you
/// need explicit [`SessionConfig`] control.
///
/// # Errors
///
/// - [`RasterError::InvalidOptions`] — `opts` violates documented constraints.
/// - [`RasterError::Pdf`] — document cannot be opened or parsed.
/// - [`RasterError::PageOutOfRange`] — requested page exceeds the document.
/// - [`RasterError::PageDegenerate`] / [`RasterError::PageTooLarge`] — malformed geometry.
/// - [`RasterError::Deskew`] — deskew rotation failed.
/// - [`RasterError::BackendUnavailable`] — `PDF_RASTER_BACKEND` requested a
///   forced backend (e.g. `cuda`, `vulkan`) and that backend's runtime
///   is unavailable.  Cannot occur if the env var is unset or `auto`.
pub fn raster_pdf(
    path: &Path,
    opts: &RasterOptions,
) -> impl Iterator<Item = (u32, Result<RenderedPage, RasterError>)> {
    render::render_pages(path, opts)
}

/// Render a range of pages concurrently using a bounded sync channel.
///
/// Spawns a background Rayon task that renders pages in ascending order and
/// sends each `(page_num, Result<RenderedPage, RasterError>)` to the returned
/// [`Receiver`](std::sync::mpsc::Receiver) as it completes.
///
/// `capacity` is the maximum number of rendered pages buffered before the
/// producer blocks.  Use `2`–`8` for typical OCR pipelines (one page rendering
/// while the previous is being OCR-processed).  `capacity = 0` is silently
/// raised to `1`.
///
/// # Errors delivered through the channel
///
/// - Invalid options → `(1, Err(RasterError::InvalidOptions(...)))`, channel closes.
/// - File open failure → `(1, Err(RasterError::Pdf(...)))`, channel closes.
/// - Forced backend unavailable (e.g. `PDF_RASTER_BACKEND=cuda` with no
///   CUDA driver) → `(1, Err(RasterError::BackendUnavailable(...)))`,
///   channel closes.
/// - Per-page failures → `(page_num, Err(...))`, rendering of subsequent pages continues.
#[must_use]
pub fn render_channel(
    path: &Path,
    opts: &RasterOptions,
    capacity: usize,
) -> std::sync::mpsc::Receiver<(u32, Result<RenderedPage, RasterError>)> {
    render::render_channel(path, opts, capacity)
}

#[cfg(test)]
mod page_set_tests {
    use super::*;

    #[test]
    fn empty_input_is_rejected() {
        assert!(matches!(
            PageSet::new(vec![]),
            Err(RasterError::InvalidOptions(_))
        ));
    }

    #[test]
    fn zero_page_is_rejected() {
        assert!(matches!(
            PageSet::new(vec![0, 1, 2]),
            Err(RasterError::InvalidOptions(_))
        ));
    }

    #[test]
    fn valid_input_is_accepted() {
        let ps = PageSet::new(vec![3, 1, 2]).unwrap();
        assert_eq!(ps.first(), 1);
        assert_eq!(ps.last(), 3);
        assert_eq!(ps.len(), 3);
        assert!(!ps.is_empty());
    }

    #[test]
    fn duplicates_are_deduplicated() {
        let ps = PageSet::new(vec![2, 1, 2, 3, 1]).unwrap();
        assert_eq!(ps.len(), 3);
    }

    #[test]
    fn contains_works() {
        let ps = PageSet::new(vec![1, 5, 10]).unwrap();
        assert!(ps.contains(1));
        assert!(ps.contains(5));
        assert!(ps.contains(10));
        assert!(!ps.contains(2));
        assert!(!ps.contains(11));
    }

    #[test]
    fn clone_is_cheap() {
        let ps = PageSet::new(vec![1, 2, 3]).unwrap();
        let ps2 = ps.clone();
        // Both point to the same allocation — Arc pointer equality
        assert!(std::ptr::eq(ps.0.as_ptr(), ps2.0.as_ptr()));
    }

    #[test]
    fn sole_zero_is_rejected() {
        assert!(matches!(
            PageSet::new(vec![0u32]),
            Err(RasterError::InvalidOptions(_))
        ));
    }

    #[test]
    fn single_max_page_is_accepted() {
        let ps = PageSet::new(vec![u32::MAX]).unwrap();
        assert_eq!(ps.first(), u32::MAX);
        assert_eq!(ps.last(), u32::MAX);
        assert_eq!(ps.len(), 1);
        assert!(!ps.is_empty());
        assert!(ps.contains(u32::MAX));
        assert!(!ps.contains(u32::MAX - 1));
    }

    #[test]
    fn accepts_iterator_input() {
        // IntoIterator covers slices, arrays, ranges, etc.
        let ps = PageSet::new([3u32, 1, 2]).unwrap();
        assert_eq!(ps.len(), 3);
        assert_eq!(ps.first(), 1);
    }

    #[test]
    fn equality_is_value_based() {
        let a = PageSet::new(vec![1, 2, 3]).unwrap();
        let b = PageSet::new([3u32, 1, 2]).unwrap(); // same content, different origin
        assert_eq!(a, b);
    }

    fn raster_options_with_pages_none_is_valid() {
        let opts = RasterOptions {
            dpi: 150.0,
            first_page: 1,
            last_page: 5,
            deskew: false,
            pages: None,
        };
        assert!(opts.pages.is_none());
        let default_opts = RasterOptions::default();
        assert_eq!(default_opts.dpi, 300.0);
        assert_eq!(default_opts.first_page, 1);
        assert_eq!(default_opts.last_page, u32::MAX);
        assert!(!default_opts.deskew);
        assert!(default_opts.pages.is_none());
    }

    #[test]
    fn raster_options_with_pages_some_is_valid() {
        let ps = PageSet::new(vec![1, 3, 5]).unwrap();
        let opts = RasterOptions {
            dpi: 150.0,
            first_page: 1,
            last_page: 5,
            deskew: false,
            pages: Some(ps),
        };
        assert!(opts.pages.is_some());
    }

    #[test]
    fn backend_policy_parses_known_names() {
        use std::str::FromStr;
        assert_eq!(BackendPolicy::from_str("auto"), Ok(BackendPolicy::Auto));
        assert_eq!(BackendPolicy::from_str("cpu"), Ok(BackendPolicy::CpuOnly));
        assert_eq!(
            BackendPolicy::from_str("cuda"),
            Ok(BackendPolicy::ForceCuda)
        );
        assert_eq!(
            BackendPolicy::from_str("vaapi"),
            Ok(BackendPolicy::ForceVaapi)
        );
        assert_eq!(
            BackendPolicy::from_str("vulkan"),
            Ok(BackendPolicy::ForceVulkan)
        );
    }

    #[test]
    fn backend_policy_empty_is_auto() {
        use std::str::FromStr;
        assert_eq!(BackendPolicy::from_str(""), Ok(BackendPolicy::Auto));
    }

    #[test]
    fn backend_policy_trims_and_ignores_case() {
        use std::str::FromStr;
        assert_eq!(
            BackendPolicy::from_str("  CUDA  "),
            Ok(BackendPolicy::ForceCuda)
        );
    }

    #[test]
    fn backend_policy_rejects_unknown() {
        use std::str::FromStr;
        assert!(BackendPolicy::from_str("metal").is_err());
    }
}
