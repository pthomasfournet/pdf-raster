//! Error taxonomy for the GPU JPEG Huffman decoder.
//!
//! Hand-rolled to match `crate::backend::BackendError`'s style (no
//! `thiserror` dep). Variants exist for conditions tests `matches!`
//! on or that benefit from typed log clarity; the catch-all
//! [`JpegGpuError::BackendError`] carries free-form messages.

use std::error::Error;
use std::fmt;

/// Errors surfaced by the GPU JPEG Huffman decoder.
#[derive(Debug)]
pub enum JpegGpuError {
    /// JPEG is progressive (SOF2); the v2 MVP only handles baseline (SOF0).
    Progressive,

    /// JPEG has a component layout the decoder does not handle. Valid
    /// values for the MVP are 1 (grayscale) and 3 (YCbCr); anything
    /// else (e.g. CMYK 4-component) is rejected here.
    UnsupportedComponents(u8),

    /// CPU pre-pass (header parse / SOS scan / unstuff) failed before
    /// the GPU phases ran. The carried string is a debug-formatted
    /// underlying error from the `jpeg` module.
    HeaderParse(String),

    /// Canonical Huffman table construction rejected the input.
    InvalidHuffmanTables(String),

    /// Phase 2 inter-sequence sync did not converge within the
    /// bounded retry count. The decoder returns this rather than
    /// hanging on an adversarial stream.
    SyncBoundExceeded,

    /// Phase 4 produced more coefficients than the IDCT stage can
    /// consume; the input is likely corrupt.
    CoefficientOverflow,

    /// The GPU backend ran out of memory while allocating decoder
    /// arenas.
    Oom,

    /// Catch-all for non-typed backend / dispatch errors.
    BackendError(String),
}

impl fmt::Display for JpegGpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Progressive => {
                write!(f, "JPEG is progressive (SOF2); not supported in v2 MVP")
            }
            Self::UnsupportedComponents(n) => write!(
                f,
                "JPEG has unsupported component layout: {n} components, expected 1 or 3"
            ),
            Self::HeaderParse(s) => write!(f, "CPU pre-pass header parse failed: {s}"),
            Self::InvalidHuffmanTables(s) => {
                write!(f, "CPU pre-pass produced invalid Huffman tables: {s}")
            }
            Self::SyncBoundExceeded => {
                write!(f, "Phase 2 sync did not converge within retry bound")
            }
            Self::CoefficientOverflow => write!(
                f,
                "Phase 4 coefficient-array overflow; input likely corrupt"
            ),
            Self::Oom => write!(f, "GPU backend out of memory at arena allocation"),
            Self::BackendError(s) => write!(f, "GPU backend error: {s}"),
        }
    }
}

impl Error for JpegGpuError {}
