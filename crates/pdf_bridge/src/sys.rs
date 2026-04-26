//! Raw `extern "C"` declarations for `poppler_shim.cc`.
//!
//! These are the only unsafe entry points in the crate.  All public types
//! in `lib.rs` wrap them in a safe API.

// sys.rs is a raw FFI layer; blanket-suppress docs warnings here since every
// declaration is self-explanatory and documented at the safe-wrapper level.
#![allow(missing_docs, clippy::missing_safety_doc)]

use std::ffi::c_char;

/// Opaque handle for a `PopplerShimDocument` (C++ heap object).
#[repr(C)]
pub struct OpaqueDocument(());
/// Opaque handle for a `PopplerShimPage` (C++ heap object).
#[repr(C)]
pub struct OpaquePage(());
/// Opaque handle for a `PopplerShimImage` (C++ heap object).
#[repr(C)]
pub struct OpaqueImage(());

unsafe extern "C" {
    // Library init
    pub fn poppler_shim_set_data_dir(path: *const c_char);
    pub fn poppler_shim_set_log_callback(f: Option<unsafe extern "C" fn(*const c_char)>);

    // Document
    pub fn poppler_shim_document_load_from_file(
        filename: *const c_char,
        owner_password: *const c_char,
        user_password: *const c_char,
    ) -> *mut OpaqueDocument;

    pub fn poppler_shim_document_load_from_data(
        data: *const c_char,
        len: i32,
        owner_password: *const c_char,
        user_password: *const c_char,
    ) -> *mut OpaqueDocument;

    pub fn poppler_shim_document_free(d: *mut OpaqueDocument);
    pub fn poppler_shim_document_pages(d: *const OpaqueDocument) -> i32;

    // Page
    pub fn poppler_shim_document_create_page(
        d: *const OpaqueDocument,
        index: i32,
    ) -> *mut OpaquePage;
    pub fn poppler_shim_page_free(p: *mut OpaquePage);
    pub fn poppler_shim_page_width(p: *const OpaquePage) -> f64;
    pub fn poppler_shim_page_height(p: *const OpaquePage) -> f64;
    pub fn poppler_shim_page_rotation(p: *const OpaquePage) -> i32;

    // Render
    pub fn poppler_shim_page_render(
        page: *const OpaquePage,
        xres: f64,
        yres: f64,
        format: i32,
        hints: u32,
    ) -> *mut OpaqueImage;

    // Image
    pub fn poppler_shim_image_free(img: *mut OpaqueImage);
    pub fn poppler_shim_image_width(img: *const OpaqueImage) -> i32;
    pub fn poppler_shim_image_height(img: *const OpaqueImage) -> i32;
    pub fn poppler_shim_image_bytes_per_row(img: *const OpaqueImage) -> i32;
    pub fn poppler_shim_image_data(img: *const OpaqueImage) -> *const c_char;
    pub fn poppler_shim_image_format(img: *const OpaqueImage) -> i32;

    // Version
    pub fn poppler_shim_version_major() -> i32;
    pub fn poppler_shim_version_minor() -> i32;
    pub fn poppler_shim_version_micro() -> i32;
}
