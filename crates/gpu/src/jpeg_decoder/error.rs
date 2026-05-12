//! Error taxonomy for the GPU JPEG Huffman decoder.
//!
//! Hand-rolled to match `crate::backend::BackendError`'s style (no
//! `thiserror` dep). Typed variants exist for conditions tests
//! `matches!` on or that benefit from typed log clarity; the
//! [`JpegGpuError::Dispatch`] catch-all carries free-form messages
//! for non-typed dispatch / backend errors.

use std::error::Error;
use std::fmt;

/// Errors surfaced by the GPU JPEG Huffman decoder.
#[derive(Debug)]
pub enum JpegGpuError {
    /// JPEG uses progressive coding (SOF2 marker); the decoder only
    /// supports sequential / baseline (SOF0).
    Progressive,

    /// JPEG has a component layout the decoder does not handle. Valid
    /// values are 1 (grayscale) and 3 (YCbCr); anything else
    /// (e.g. CMYK 4-component) is rejected here.
    UnsupportedComponents(u8),

    /// CPU pre-pass (header parse / SOS scan / unstuff) failed before
    /// the GPU dispatch ran. The carried string is the underlying
    /// error formatted via `Display`.
    HeaderParse(String),

    /// Canonical Huffman table construction rejected the input
    /// (empty table, code-space overflow, or length mismatch).
    InvalidHuffmanTables(String),

    /// Inter-sequence sync (Wei §III-B) did not converge within the
    /// bounded retry count. The decoder returns this rather than
    /// hanging on an adversarial stream.
    SyncBoundExceeded,

    /// Re-decode produced more coefficients than the IDCT stage can
    /// consume; the input is likely corrupt.
    CoefficientOverflow,

    /// The GPU backend ran out of memory while allocating decoder
    /// arenas.
    Oom,

    /// Catch-all for non-typed dispatch / backend errors. Distinct
    /// from [`crate::backend::BackendError`] — this is the JPEG
    /// decoder's free-form message wrapper, not a re-export of the
    /// trait-level error type.
    Dispatch(String),
}

impl fmt::Display for JpegGpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Progressive => {
                write!(
                    f,
                    "JPEG is progressive (SOF2); decoder only supports baseline (SOF0)"
                )
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
                write!(f, "inter-sequence sync did not converge within retry bound")
            }
            Self::CoefficientOverflow => write!(
                f,
                "re-decode coefficient-array overflow; input likely corrupt"
            ),
            Self::Oom => write!(f, "GPU backend out of memory at arena allocation"),
            Self::Dispatch(s) => write!(f, "GPU JPEG dispatch error: {s}"),
        }
    }
}

impl Error for JpegGpuError {}
