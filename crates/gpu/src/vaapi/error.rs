//! Error type for the VA-API JPEG decoder.

#![cfg(feature = "vaapi")]

use std::ffi::CStr;

use super::ffi::{vaErrorStr, VAStatus, VA_STATUS_SUCCESS};

/// Errors returned by the VA-API JPEG decoder.
#[derive(Debug)]
pub enum VapiError {
    /// A VA-API API call returned a non-success status code.
    ///
    /// Common values:
    /// - 1 = `VA_STATUS_ERROR_OPERATION_FAILED`
    /// - 3 = `VA_STATUS_ERROR_INVALID_DISPLAY`
    /// - 7 = `VA_STATUS_ERROR_INVALID_PARAMETER`
    /// - 12 = `VA_STATUS_ERROR_UNSUPPORTED_PROFILE`
    /// - 13 = `VA_STATUS_ERROR_UNSUPPORTED_ENTRYPOINT`
    VaStatus(VAStatus, String),
    /// Failed to open the DRM render node.
    DrmOpen(String),
    /// `vaGetDisplayDRM` returned a null display pointer.
    NullDisplay,
    /// JPEG headers are missing or malformed.
    BadJpeg(String),
    /// Image has an unsupported component count (not 1 or 3).
    UnsupportedComponents(u8),
    /// Integer overflow in pixel buffer size computation.
    Overflow,
}

impl std::fmt::Display for VapiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::VaStatus(code, msg) => write!(f, "VA-API error {code}: {msg}"),
            Self::DrmOpen(msg) => write!(f, "DRM open failed: {msg}"),
            Self::NullDisplay => write!(f, "vaGetDisplayDRM returned null"),
            Self::BadJpeg(msg) => write!(f, "JPEG parse error: {msg}"),
            Self::UnsupportedComponents(n) => {
                write!(f, "unsupported JPEG component count {n}")
            }
            Self::Overflow => write!(f, "pixel buffer size overflow"),
        }
    }
}

impl std::error::Error for VapiError {}

pub type Result<T> = std::result::Result<T, VapiError>;

/// Convert a VA-API status code to `Err` with the human-readable error string.
pub(super) fn check(status: VAStatus, context: &str) -> Result<()> {
    if status == VA_STATUS_SUCCESS {
        return Ok(());
    }
    // SAFETY: vaErrorStr returns a static C string or NULL; we check for NULL.
    let msg = unsafe {
        let ptr = vaErrorStr(status);
        if ptr.is_null() {
            format!("{context} (status {status})")
        } else {
            format!("{context}: {}", CStr::from_ptr(ptr).to_string_lossy())
        }
    };
    Err(VapiError::VaStatus(status, msg))
}
