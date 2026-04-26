//! Image encoding — write [`Bitmap<P>`] to PPM, PGM, and PNG.
//!
//! # Supported formats
//!
//! | Function | Format | Accepted pixel types |
//! |----------|--------|--------------------|
//! | [`write_ppm`] | Netpbm P6 binary | `Rgb8`, `Bgr8`, `Rgba8` (`Xbgr8`), `Cmyk8`, `DeviceN8` |
//! | [`write_pgm`] | Netpbm P5 binary | `Gray8` |
//! | [`write_png`] | PNG | `Rgb8`, `Gray8`, `Rgba8` |
//!
//! All functions consume the output sink (`W: Write`).  Wrap in
//! [`std::io::BufWriter`] at the call site if buffering is needed.
//!
//! # CMYK handling
//!
//! Neither PPM nor PNG natively supports CMYK.  [`write_ppm`] converts
//! CMYK/`DeviceN` to RGB via the naïve ink-density formula
//! `R = 255 − C − K` (matching poppler's `pdftoppm` output).
//! For ICC-accurate colour, convert to `Rgb8` before encoding.

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
    /// The pixel mode is not supported by the chosen encoder.
    ///
    /// The message describes what the caller should do instead.
    UnsupportedMode(&'static str),
    /// An internal error from the `png` encoder (non-I/O).
    PngEncoder(::png::EncodingError),
}

impl std::fmt::Display for EncodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::UnsupportedMode(m) => write!(f, "pixel mode not supported: {m}"),
            Self::PngEncoder(e) => write!(f, "PNG encoder error: {e}"),
        }
    }
}

impl std::error::Error for EncodeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::PngEncoder(e) => Some(e),
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
        // Unwrap the I/O layer so EncodeError::Io is the canonical I/O path.
        match e {
            ::png::EncodingError::IoError(io_err) => Self::Io(io_err),
            other => Self::PngEncoder(other),
        }
    }
}
