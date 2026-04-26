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

use std::ffi::CString;
use sys::{
    OpaqueDocument, OpaqueImage, OpaquePage, poppler_shim_document_create_page,
    poppler_shim_document_free, poppler_shim_document_load_from_data,
    poppler_shim_document_load_from_file, poppler_shim_document_pages,
    poppler_shim_image_bytes_per_row, poppler_shim_image_data, poppler_shim_image_format,
    poppler_shim_image_free, poppler_shim_image_height, poppler_shim_image_width,
    poppler_shim_page_free, poppler_shim_page_height, poppler_shim_page_render,
    poppler_shim_page_rotation, poppler_shim_page_width, poppler_shim_version_major,
    poppler_shim_version_micro, poppler_shim_version_minor,
};

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
    /// Enable `FreeType` hinting (default false — matches pdftoppm default).
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
    pub fn from_bytes(
        data: &[u8],
        owner_password: Option<&str>,
        user_password: Option<&str>,
    ) -> Result<Self, Error> {
        let c_owner = CString::new(owner_password.unwrap_or(""))?;
        let c_user = CString::new(user_password.unwrap_or(""))?;

        let len = i32::try_from(data.len()).unwrap_or(i32::MAX);
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
    #[must_use]
    pub fn pixel_width(&self, x_dpi: f64) -> u32 {
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "page dimensions are non-negative; result fits in u32 for any sane DPI"
        )]
        {
            ((self.width_pt() / 72.0) * x_dpi).round() as u32
        }
    }

    /// Height of the rendered image in pixels at the given DPI.
    #[must_use]
    pub fn pixel_height(&self, y_dpi: f64) -> u32 {
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "page dimensions are non-negative; result fits in u32 for any sane DPI"
        )]
        {
            ((self.height_pt() / 72.0) * y_dpi).round() as u32
        }
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
        #[expect(
            clippy::cast_sign_loss,
            reason = "poppler never returns a negative width for valid images"
        )]
        let w = unsafe { poppler_shim_image_width(self.ptr) } as u32;
        w
    }

    /// Image height in pixels.
    #[must_use]
    pub fn height(&self) -> u32 {
        #[expect(
            clippy::cast_sign_loss,
            reason = "poppler never returns a negative height for valid images"
        )]
        let h = unsafe { poppler_shim_image_height(self.ptr) } as u32;
        h
    }

    /// Bytes per row (may include stride padding).
    #[must_use]
    pub fn bytes_per_row(&self) -> usize {
        #[expect(
            clippy::cast_sign_loss,
            reason = "poppler never returns a negative stride for valid images"
        )]
        let s = unsafe { poppler_shim_image_bytes_per_row(self.ptr) } as usize;
        s
    }

    /// Pixel format of the image.
    #[must_use]
    pub fn format(&self) -> Option<ImageFormat> {
        let fmt = unsafe { poppler_shim_image_format(self.ptr) };
        ImageFormat::from_shim(fmt)
    }

    /// Raw pixel data slice (length = `bytes_per_row() * height()`).
    #[must_use]
    pub fn data(&self) -> &[u8] {
        let len = self.bytes_per_row() * self.height() as usize;
        if len == 0 {
            return &[];
        }
        unsafe {
            let ptr = poppler_shim_image_data(self.ptr);
            std::slice::from_raw_parts(ptr.cast::<u8>(), len)
        }
    }

    /// Row `y` as a byte slice (0-indexed from the top).
    ///
    /// Useful for writing rows without first copying to a flat buffer.
    ///
    /// # Panics
    ///
    /// Panics if `y >= self.height()`.
    #[must_use]
    pub fn row(&self, y: u32) -> &[u8] {
        assert!(
            y < self.height(),
            "row {y} out of range (height {})",
            self.height()
        );
        let bpr = self.bytes_per_row();
        &self.data()[y as usize * bpr..(y as usize + 1) * bpr]
    }

    /// Copy the pixel data into a contiguous `Vec<u8>`, removing any stride
    /// padding.  Row stride is `width * bytes_per_pixel`.
    ///
    /// Returns `None` for `Mono` format (bit-packed; no simple byte-per-pixel
    /// count).
    #[must_use]
    pub fn to_packed_vec(&self) -> Option<Vec<u8>> {
        let fmt = self.format()?;
        let bpp = fmt.bytes_per_pixel();
        if bpp == 0 {
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
// Tests (require a PDF file at runtime — skipped in CI by default)
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
    }
}
