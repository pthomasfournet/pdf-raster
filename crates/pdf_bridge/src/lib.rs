//! Safe Rust wrapper over `libpoppler-cpp`.
//!
//! The crate compiles a thin C++ shim (`cpp/poppler_shim.cc`) that exposes
//! `extern "C"` entry points, then builds a safe Rust layer on top.
//!
//! # Quick start
//!
//! ```no_run
//! use pdf_bridge::{Document, RenderParams};
//!
//! let doc = Document::from_file("/path/to/input.pdf", None, None)
//!     .expect("failed to open PDF");
//! let params = RenderParams::default(); // 150 DPI, RGB24, anti-aliasing on
//!
//! for i in 0..doc.page_count() {
//!     let page = doc.page(i).expect("page out of range");
//!     let img  = page.render(&params).expect("render failed");
//!     println!("page {}: {}×{} pixels", i + 1, img.width(), img.height());
//!     let _rgb: &[u8] = img.data();
//! }
//! ```

pub mod sys;

use std::ffi::{CStr, CString, c_char};
use sys::{
    OpaqueDocument, OpaqueImage, OpaquePage, poppler_shim_document_create_page,
    poppler_shim_document_free, poppler_shim_document_load_from_data,
    poppler_shim_document_load_from_file, poppler_shim_document_pages,
    poppler_shim_image_bytes_per_row, poppler_shim_image_data, poppler_shim_image_format,
    poppler_shim_image_free, poppler_shim_image_height, poppler_shim_image_width,
    poppler_shim_page_free, poppler_shim_page_height, poppler_shim_page_render,
    poppler_shim_page_rotation, poppler_shim_page_width, poppler_shim_set_log_callback,
    poppler_shim_version_major, poppler_shim_version_micro, poppler_shim_version_minor,
};

// ---------------------------------------------------------------------------
// Logging
// ---------------------------------------------------------------------------

/// Trampoline called by the C++ shim for every poppler diagnostic message.
///
/// Routes to [`log::debug!`] so output is suppressed unless the caller enables
/// `RUST_LOG=pdf_bridge=debug` (or coarser).  Real load failures are still
/// surfaced via `Err` returns from the public API; these messages are only
/// informational noise from quirky-but-readable PDFs.
///
/// # Safety
///
/// `msg` must be a valid, non-null, null-terminated byte string that remains
/// valid for the duration of this call.  Poppler guarantees this for all
/// messages it emits.
unsafe extern "C" fn log_trampoline(msg: *const c_char) {
    // SAFETY: caller guarantees msg is a valid null-terminated string.
    let s = unsafe { CStr::from_ptr(msg) }.to_string_lossy();
    log::debug!(target: "pdf_bridge::poppler", "{s}");
}

/// Redirect poppler's internal diagnostics to the [`log`] crate.
///
/// Call once at program startup, before opening any document.  Without this
/// call the C++ shim still suppresses stderr output — messages are silently
/// dropped.  After this call they become visible at `DEBUG` level under the
/// `pdf_bridge::poppler` target.
///
/// Calling this function multiple times is safe and simply overwrites the
/// previous callback atomically (the C++ side uses `std::atomic`).
pub fn install_log_callback() {
    // SAFETY: `log_trampoline` has the correct `extern "C"` signature, its
    // pointer is valid for the entire process lifetime, and the C++ shim
    // stores it in a `std::atomic` so concurrent reads from poppler's error
    // callback thread are safe.
    unsafe { poppler_shim_set_log_callback(Some(log_trampoline)) };
}

// ---------------------------------------------------------------------------
// Image format
// ---------------------------------------------------------------------------

/// Pixel format of a rendered [`RenderedPage`].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ImageFormat {
    /// 1-bit packed monochrome (MSB-first).
    Mono,
    /// 24-bit RGB, 3 bytes per pixel.
    Rgb24,
    /// 32-bit ARGB, 4 bytes per pixel (byte order: B G R A on little-endian).
    Argb32,
    /// 8-bit greyscale, 1 byte per pixel.
    Gray8,
    /// 24-bit BGR, 3 bytes per pixel.
    Bgr24,
}

impl ImageFormat {
    /// Bytes per pixel (0 for `Mono` — that format is bit-packed).
    #[must_use]
    pub const fn bytes_per_pixel(self) -> usize {
        match self {
            Self::Mono => 0,
            Self::Gray8 => 1,
            Self::Rgb24 | Self::Bgr24 => 3,
            Self::Argb32 => 4,
        }
    }

    pub(crate) const fn from_shim(v: i32) -> Option<Self> {
        match v {
            1 => Some(Self::Mono),
            2 => Some(Self::Rgb24),
            3 => Some(Self::Argb32),
            4 => Some(Self::Gray8),
            5 => Some(Self::Bgr24),
            _ => None,
        }
    }

    pub(crate) const fn to_shim(self) -> i32 {
        match self {
            Self::Mono => 1,
            Self::Rgb24 => 2,
            Self::Argb32 => 3,
            Self::Gray8 => 4,
            Self::Bgr24 => 5,
        }
    }
}

// ---------------------------------------------------------------------------
// Render parameters
// ---------------------------------------------------------------------------

/// Parameters for [`Page::render`].
#[derive(Clone, Debug)]
pub struct RenderParams {
    /// Horizontal resolution in DPI (default 150).
    pub x_dpi: f64,
    /// Vertical resolution in DPI (default 150).
    pub y_dpi: f64,
    /// Output pixel format (default [`ImageFormat::Rgb24`]).
    pub format: ImageFormat,
    /// Enable shape anti-aliasing (default true).
    pub antialias: bool,
    /// Enable text anti-aliasing (default true).
    pub text_antialias: bool,
    /// Enable `FreeType` hinting (default false).
    pub text_hinting: bool,
}

impl Default for RenderParams {
    fn default() -> Self {
        Self {
            x_dpi: 150.0,
            y_dpi: 150.0,
            format: ImageFormat::Rgb24,
            antialias: true,
            text_antialias: true,
            text_hinting: false,
        }
    }
}

impl RenderParams {
    const fn hint_flags(&self) -> u32 {
        let mut h = 0u32;
        if self.antialias {
            h |= 0x01;
        }
        if self.text_antialias {
            h |= 0x02;
        }
        if self.text_hinting {
            h |= 0x04;
        }
        h
    }
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors returned by `pdf_bridge` operations.
#[derive(Debug)]
pub enum Error {
    /// The PDF could not be opened (bad path, corrupted data, or wrong password).
    Open(String),
    /// Page index out of the document's valid range.
    PageOutOfRange {
        /// The requested 0-based index.
        index: i32,
        /// The total number of pages in the document.
        count: i32,
    },
    /// Rendering failed (poppler returned an invalid image).
    RenderFailed,
    /// An argument contains a null byte and cannot be passed to C.
    NulByte(std::ffi::NulError),
    /// The data buffer is too large to pass to poppler (exceeds `i32::MAX` bytes).
    DataTooLarge(usize),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Open(s) => write!(f, "failed to open PDF: {s}"),
            Self::PageOutOfRange { index, count } => {
                write!(
                    f,
                    "page index {index} out of range (document has {count} pages)"
                )
            }
            Self::RenderFailed => write!(f, "poppler render_page returned an invalid image"),
            Self::NulByte(e) => write!(f, "argument contains a null byte: {e}"),
            Self::DataTooLarge(len) => write!(
                f,
                "PDF data buffer ({len} bytes) exceeds the 2 GiB limit for in-memory loading"
            ),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::NulByte(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::ffi::NulError> for Error {
    fn from(e: std::ffi::NulError) -> Self {
        Self::NulByte(e)
    }
}

// ---------------------------------------------------------------------------
// Document
// ---------------------------------------------------------------------------

/// An open PDF document.
///
/// Dropping this value closes the document and frees all poppler resources.
pub struct Document {
    ptr: *mut OpaqueDocument,
}

// SAFETY: poppler::document is internally reference-counted and thread-safe
// for read operations (page rendering).  We never mutate it after open.
unsafe impl Send for Document {}
unsafe impl Sync for Document {}

impl Document {
    /// Open a PDF from a file path.
    ///
    /// `owner_password` and `user_password` are `None` for unencrypted files.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Open`] if the file cannot be opened, is corrupt,
    /// or the supplied password is wrong.
    pub fn from_file(
        path: &str,
        owner_password: Option<&str>,
        user_password: Option<&str>,
    ) -> Result<Self, Error> {
        let c_path = CString::new(path)?;
        let c_owner = CString::new(owner_password.unwrap_or(""))?;
        let c_user = CString::new(user_password.unwrap_or(""))?;

        let ptr = unsafe {
            poppler_shim_document_load_from_file(c_path.as_ptr(), c_owner.as_ptr(), c_user.as_ptr())
        };
        if ptr.is_null() {
            return Err(Error::Open(format!("could not open '{path}'")));
        }
        Ok(Self { ptr })
    }

    /// Open a PDF from an in-memory byte slice.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Open`] if the data is not a valid PDF or the password is wrong.
    /// Returns [`Error::DataTooLarge`] if the buffer exceeds `i32::MAX` bytes.
    pub fn from_bytes(
        data: &[u8],
        owner_password: Option<&str>,
        user_password: Option<&str>,
    ) -> Result<Self, Error> {
        let c_owner = CString::new(owner_password.unwrap_or(""))?;
        let c_user = CString::new(user_password.unwrap_or(""))?;

        let len = i32::try_from(data.len()).map_err(|_| Error::DataTooLarge(data.len()))?;
        let ptr = unsafe {
            poppler_shim_document_load_from_data(
                data.as_ptr().cast::<std::ffi::c_char>(),
                len,
                c_owner.as_ptr(),
                c_user.as_ptr(),
            )
        };
        if ptr.is_null() {
            return Err(Error::Open("could not parse PDF from bytes".to_owned()));
        }
        Ok(Self { ptr })
    }

    /// Total number of pages in the document.
    #[must_use]
    pub fn page_count(&self) -> i32 {
        unsafe { poppler_shim_document_pages(self.ptr) }
    }

    /// Return the page at `index` (0-based).
    ///
    /// # Errors
    ///
    /// Returns [`Error::PageOutOfRange`] if `index >= page_count()`.
    pub fn page(&self, index: i32) -> Result<Page<'_>, Error> {
        let count = self.page_count();
        if index < 0 || index >= count {
            return Err(Error::PageOutOfRange { index, count });
        }
        let ptr = unsafe { poppler_shim_document_create_page(self.ptr, index) };
        if ptr.is_null() {
            return Err(Error::PageOutOfRange { index, count });
        }
        Ok(Page {
            ptr,
            _doc: std::marker::PhantomData,
        })
    }
}

impl Drop for Document {
    fn drop(&mut self) {
        unsafe {
            poppler_shim_document_free(self.ptr);
        }
    }
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

/// A single page in a [`Document`].
///
/// The lifetime `'doc` ensures the page cannot outlive its owning document.
pub struct Page<'doc> {
    ptr: *mut OpaquePage,
    _doc: std::marker::PhantomData<&'doc Document>,
}

// SAFETY: page object is only read from (no interior mutation after creation).
unsafe impl Send for Page<'_> {}

impl Page<'_> {
    /// Page width in points (1/72 inch).
    #[must_use]
    pub fn width_pt(&self) -> f64 {
        unsafe { poppler_shim_page_width(self.ptr) }
    }

    /// Page height in points (1/72 inch).
    #[must_use]
    pub fn height_pt(&self) -> f64 {
        unsafe { poppler_shim_page_height(self.ptr) }
    }

    /// Page rotation stored in the PDF (0, 90, 180, or 270 degrees).
    #[must_use]
    pub fn rotation(&self) -> i32 {
        unsafe { poppler_shim_page_rotation(self.ptr) }
    }

    /// Width of the rendered image in pixels at the given DPI.
    ///
    /// Returns 0 if the page dimension is non-positive or the DPI is not finite.
    #[must_use]
    pub fn pixel_width(&self, x_dpi: f64) -> u32 {
        pts_to_pixels(self.width_pt(), x_dpi)
    }

    /// Height of the rendered image in pixels at the given DPI.
    ///
    /// Returns 0 if the page dimension is non-positive or the DPI is not finite.
    #[must_use]
    pub fn pixel_height(&self, y_dpi: f64) -> u32 {
        pts_to_pixels(self.height_pt(), y_dpi)
    }

    /// Render this page to a pixel buffer.
    ///
    /// # Errors
    ///
    /// Returns [`Error::RenderFailed`] if poppler cannot rasterize the page.
    pub fn render(&self, params: &RenderParams) -> Result<RenderedPage, Error> {
        let ptr = unsafe {
            poppler_shim_page_render(
                self.ptr,
                params.x_dpi,
                params.y_dpi,
                params.format.to_shim(),
                params.hint_flags(),
            )
        };
        if ptr.is_null() {
            return Err(Error::RenderFailed);
        }
        Ok(RenderedPage { ptr })
    }
}

impl Drop for Page<'_> {
    fn drop(&mut self) {
        unsafe {
            poppler_shim_page_free(self.ptr);
        }
    }
}

// ---------------------------------------------------------------------------
// RenderedPage
// ---------------------------------------------------------------------------

/// A rasterized page — owns the pixel buffer returned by poppler.
pub struct RenderedPage {
    ptr: *mut OpaqueImage,
}

// SAFETY: RenderedPage owns the image buffer; no shared mutable state.
unsafe impl Send for RenderedPage {}

impl RenderedPage {
    /// Image width in pixels.
    #[must_use]
    pub fn width(&self) -> u32 {
        nonneg_i32_to_u32(unsafe { poppler_shim_image_width(self.ptr) })
    }

    /// Image height in pixels.
    #[must_use]
    pub fn height(&self) -> u32 {
        nonneg_i32_to_u32(unsafe { poppler_shim_image_height(self.ptr) })
    }

    /// Bytes per row (may include stride padding).
    #[must_use]
    pub fn bytes_per_row(&self) -> usize {
        nonneg_i32_to_u32(unsafe { poppler_shim_image_bytes_per_row(self.ptr) }) as usize
    }

    /// Pixel format of the image.
    #[must_use]
    pub fn format(&self) -> Option<ImageFormat> {
        let fmt = unsafe { poppler_shim_image_format(self.ptr) };
        ImageFormat::from_shim(fmt)
    }

    /// Raw pixel data slice (length = `bytes_per_row() * height()`).
    ///
    /// The slice includes any stride padding; use [`RenderedPage::to_packed_vec`]
    /// or iterate with [`RenderedPage::row`] to strip it.
    #[must_use]
    pub fn data(&self) -> &[u8] {
        let bpr = self.bytes_per_row();
        let h = self.height() as usize;
        let len = bpr.saturating_mul(h);
        if len == 0 {
            return &[];
        }
        unsafe {
            let ptr = poppler_shim_image_data(self.ptr);
            std::slice::from_raw_parts(ptr.cast::<u8>(), len)
        }
    }

    /// Row `y` as a byte slice (0-indexed from the top), or `None` if out of range.
    ///
    /// The returned slice is `bytes_per_row()` long and may include stride padding.
    #[must_use]
    pub fn row(&self, y: u32) -> Option<&[u8]> {
        if y >= self.height() {
            return None;
        }
        let bpr = self.bytes_per_row();
        let start = (y as usize).saturating_mul(bpr);
        Some(&self.data()[start..start + bpr])
    }

    /// Copy the pixel data into a contiguous `Vec<u8>`, removing any stride
    /// padding.  Row stride is `width * bytes_per_pixel`.
    ///
    /// Returns `None` for [`ImageFormat::Mono`] (bit-packed; no simple
    /// bytes-per-pixel count) or if the image has an unrecognised format.
    #[must_use]
    pub fn to_packed_vec(&self) -> Option<Vec<u8>> {
        let fmt = self.format()?;
        let bpp = fmt.bytes_per_pixel();
        if bpp == 0 {
            // Mono is bit-packed — a simple bpp copy is not meaningful.
            return None;
        }
        let w = self.width() as usize;
        let h = self.height() as usize;
        let bpr_src = self.bytes_per_row();
        let bpr_dst = w * bpp;

        let mut out = Vec::with_capacity(bpr_dst * h);
        let src = self.data();
        for row in 0..h {
            out.extend_from_slice(&src[row * bpr_src..row * bpr_src + bpr_dst]);
        }
        Some(out)
    }
}

impl Drop for RenderedPage {
    fn drop(&mut self) {
        unsafe {
            poppler_shim_image_free(self.ptr);
        }
    }
}

// ---------------------------------------------------------------------------
// Version query
// ---------------------------------------------------------------------------

/// Poppler library version as `(major, minor, micro)`.
#[must_use]
pub fn poppler_version() -> (i32, i32, i32) {
    unsafe {
        (
            poppler_shim_version_major(),
            poppler_shim_version_minor(),
            poppler_shim_version_micro(),
        )
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Convert a non-negative i32 returned by poppler to u32.
///
/// Negative values (which indicate an error in the C API) are clamped to 0
/// so callers see an empty/zero-size image rather than a wrapped-around value.
#[inline]
fn nonneg_i32_to_u32(v: i32) -> u32 {
    v.max(0).cast_unsigned()
}

/// Convert a page dimension in points at a given DPI to a pixel count.
///
/// Returns 0 for non-positive dimensions or non-finite DPI values.
#[inline]
fn pts_to_pixels(pts: f64, dpi: f64) -> u32 {
    if pts <= 0.0 || !dpi.is_finite() || dpi <= 0.0 {
        return 0;
    }
    let px = (pts / 72.0 * dpi).round();
    // Clamp to u32::MAX rather than panicking on overflow for absurd DPI values.
    // The `px > u32::MAX` guard means the cast cannot truncate or lose sign.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "px is non-negative (pts and dpi are both positive) and bounded by u32::MAX above"
    )]
    if px > f64::from(u32::MAX) {
        u32::MAX
    } else {
        px as u32
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_is_reasonable() {
        let (major, _minor, _micro) = poppler_version();
        assert!(major >= 20, "expected poppler ≥ 20, got {major}");
    }

    #[test]
    fn image_format_round_trip() {
        for fmt in [
            ImageFormat::Mono,
            ImageFormat::Rgb24,
            ImageFormat::Argb32,
            ImageFormat::Gray8,
            ImageFormat::Bgr24,
        ] {
            assert_eq!(ImageFormat::from_shim(fmt.to_shim()), Some(fmt));
        }
    }

    #[test]
    fn render_params_hint_flags() {
        let p = RenderParams {
            antialias: true,
            text_antialias: false,
            text_hinting: true,
            ..Default::default()
        };
        assert_eq!(p.hint_flags(), 0x05);
    }

    #[test]
    fn render_params_default_flags() {
        assert_eq!(RenderParams::default().hint_flags(), 0x03);
    }

    #[test]
    fn pts_to_pixels_edge_cases() {
        assert_eq!(pts_to_pixels(0.0, 150.0), 0, "zero pts");
        assert_eq!(pts_to_pixels(-1.0, 150.0), 0, "negative pts");
        assert_eq!(pts_to_pixels(72.0, 0.0), 0, "zero dpi");
        assert_eq!(pts_to_pixels(72.0, f64::NAN), 0, "NaN dpi");
        assert_eq!(
            pts_to_pixels(72.0, f64::INFINITY),
            0,
            "infinite dpi is not valid"
        );
        assert_eq!(pts_to_pixels(72.0, 72.0), 72, "1 inch at 72 dpi");
        assert_eq!(pts_to_pixels(72.0, 150.0), 150, "1 inch at 150 dpi");
    }

    #[test]
    fn nonneg_i32_to_u32_clamps_negative() {
        assert_eq!(nonneg_i32_to_u32(-1), 0);
        assert_eq!(nonneg_i32_to_u32(0), 0);
        assert_eq!(nonneg_i32_to_u32(42), 42);
    }

    #[test]
    fn data_too_large_error_displayed() {
        let e = Error::DataTooLarge(3 * 1024 * 1024 * 1024);
        let s = e.to_string();
        assert!(s.contains("3221225472"), "should include byte count: {s}");
        assert!(s.contains("2 GiB"), "should mention limit: {s}");
    }

    /// Integration test: requires a real PDF at the path below.
    /// Run with: `POPPLER_TEST_PDF=/path/to/file.pdf cargo test -p pdf_bridge`
    #[test]
    fn open_and_render_first_page() {
        let Some(path) = std::env::var("POPPLER_TEST_PDF").ok() else {
            return; // skip if no test PDF provided
        };
        let doc = Document::from_file(&path, None, None).expect("failed to open PDF");
        assert!(doc.page_count() > 0);

        let page = doc.page(0).expect("page 0");
        assert!(page.width_pt() > 0.0);
        assert!(page.height_pt() > 0.0);

        let params = RenderParams {
            x_dpi: 72.0,
            y_dpi: 72.0,
            ..Default::default()
        };
        let img = page.render(&params).expect("render failed");
        assert!(img.width() > 0);
        assert!(img.height() > 0);
        assert_eq!(img.format(), Some(ImageFormat::Rgb24));
        assert!(!img.data().is_empty());
        assert!(img.row(0).is_some());
        assert!(img.row(img.height()).is_none());
    }
}
