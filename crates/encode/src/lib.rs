//! Image encoding — write [`Bitmap<P>`] to PPM, PGM, and PNG.
//!
//! # Supported formats
//!
//! | Function | Format | Pixel modes |
//! |----------|--------|-------------|
//! | [`write_ppm`] | Netpbm P6 binary | `Rgb8`, `Bgr8`, `Xbgr8`, `Cmyk8`, `DeviceN8` |
//! | [`write_pgm`] | Netpbm P5 binary | `Gray8`, `Mono8` |
//! | [`write_png`] | PNG (via `png` crate) | `Rgb8`, `Gray8`, `Rgba8` |
//!
//! All functions write to any [`std::io::Write`] sink (file, `Vec<u8>`, …).
//!
//! # CMYK handling
//!
//! Neither PPM nor PNG natively supports CMYK.  [`write_ppm`] converts
//! CMYK/`DeviceN` to RGB via simple ink-density subtraction before writing.
//! Use [`write_png`] with a pre-converted `Bitmap<Rgb8>` for best quality.

pub mod pgm;
pub mod png;
pub mod ppm;

pub use pgm::write_pgm;
pub use png::write_png;
pub use ppm::write_ppm;

use std::io;

/// Errors that can occur during encoding.
#[derive(Debug)]
pub enum EncodeError {
    /// An I/O error writing to the output sink.
    Io(io::Error),
    /// The pixel mode is not supported by the chosen format.
    UnsupportedMode(&'static str),
    /// The `png` encoder returned an internal error.
    Png(::png::EncodingError),
}

impl std::fmt::Display for EncodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::UnsupportedMode(m) => write!(f, "pixel mode not supported: {m}"),
            Self::Png(e) => write!(f, "PNG encoding error: {e}"),
        }
    }
}

impl std::error::Error for EncodeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::Png(e) => Some(e),
            Self::UnsupportedMode(_) => None,
        }
    }
}

impl From<io::Error> for EncodeError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<::png::EncodingError> for EncodeError {
    fn from(e: ::png::EncodingError) -> Self {
        match e {
            ::png::EncodingError::IoError(io_err) => Self::Io(io_err),
            other => Self::Png(other),
        }
    }
}
