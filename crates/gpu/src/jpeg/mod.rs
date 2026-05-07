//! JPEG support module — shared parsing primitives + the on-GPU decoder pipeline.
//!
//! The module is organised so that the JPEG-spec-level work (header parsing,
//! canonical Huffman table construction, byte-unstuffing, DC differential
//! resolution) is feature-flag-free and reusable by any caller in the crate.
//! VA-API and the on-GPU decoder both consume these primitives instead of
//! reimplementing them.
//!
//! ## Submodules
//!
//! | Module | Status | Responsibility |
//! |---|---|---|
//! | [`headers`] | always-on | Walk a JPEG marker stream; extract DQT, DHT, SOF, SOS, DRI. |
//! | [`canonical`] | always-on | Build a canonical Huffman lookup table from the DHT-form `(num_codes, values)`. |
//! | [`unstuff`] | always-on | Strip `0xFF 0x00 → 0xFF` byte-stuffing from the entropy-coded segment. |
//! | [`dc_chain`] | always-on | Walk the entropy-coded segment, resolve per-MCU absolute DC values. |
//! | [`prepass`] | always-on | Top-level orchestrator: bytes → [`CpuPrepassOutput`]. |
//!
//! All five submodules are CPU-only Rust. The companion GPU phases (parallel
//! Huffman, IDCT + colour convert) land behind the `nvjpeg-gpu` feature in a
//! later commit and consume [`CpuPrepassOutput`] as their input contract.

pub mod canonical;
pub mod dc_chain;
pub mod headers;
pub mod prepass;
pub mod unstuff;

#[cfg(test)]
pub(crate) mod test_fixtures;

pub use canonical::{CanonicalCodebook, CanonicalCodebookError};
pub use dc_chain::{DcChainError, DcValues};
pub use headers::{
    DhtClass, JpegHeaderError, JpegHeaders, JpegHuffmanTable, JpegQuantTable, JpegScanComponent,
};
pub use prepass::{CpuPrepassError, CpuPrepassOutput, run_cpu_prepass};
pub use unstuff::{RstPosition, unstuff_into};
