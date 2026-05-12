//! JPEG support — shared parsing primitives consumed by VA-API and the
//! on-GPU decoder.
//!
//! | Module | Responsibility |
//! |---|---|
//! | [`headers`] | Walk a JPEG marker stream; extract DQT, DHT, SOF, SOS, DRI. |
//! | [`canonical`] | Build a canonical Huffman lookup table from the DHT-form `(num_codes, values)`. |
//! | [`unstuff`] | Strip `0xFF 0x00 → 0xFF` byte-stuffing; record RST marker positions. |
//! | [`dc_chain`] | Resolve per-block absolute DC values across the entropy stream. |
//! | [`prepass`] | Top-level orchestrator: bytes → [`CpuPrepassOutput`]. |

pub mod canonical;
pub mod dc_chain;
pub mod headers;
pub mod prepass;
pub mod unstuff;

#[cfg(test)]
pub(crate) mod test_fixtures;

pub use canonical::{
    CanonicalCodebook, CanonicalCodebookError, validate_canonical_table, visit_canonical_codes,
};
pub use dc_chain::{DcChainError, DcValues};
pub use headers::{
    DhtClass, JpegHeaderError, JpegHeaders, JpegHuffmanTable, JpegQuantTable, JpegScanComponent,
};
pub use prepass::{CpuPrepassError, CpuPrepassOutput, run_cpu_prepass};
pub use unstuff::{RstPosition, unstuff_into};
