//! Parallel-Huffman JPEG decoder for the GPU compute path.
//!
//! Algorithm: Weißenberger & Schmidt, "A massively parallel Huffman
//! decoder" (TPDS 2021), §III–IV.
//!
//! This module is the orchestrator; GPU kernels (when they land)
//! live under `crates/gpu/kernels/jpeg/`. Today the surface is the
//! CPU-side primitives every backend will share: bitstream packing,
//! 2-tier codetable construction, a scalar reference decoder, and a
//! synthetic-stream oracle used by cross-backend bit-identity tests.

mod bitstream;
mod codetable;
mod cpu_prepass;
#[cfg(test)]
mod cpu_reference;
mod dispatch_util;
mod error;
mod jpeg_framing;
// huffman + phase1_oracle are test-only today (oracle-comparison
// tests against the GPU dispatcher); promote when production
// callers integrate the Phase 1 decode path. The huffman dispatcher
// needs a real CUDA device, hence the gpu-validation gate; the
// SubsequenceState type it owns is also re-imported by
// phase1_oracle, which is why that one is plain `#[cfg(test)]`.
#[cfg(all(test, feature = "gpu-validation"))]
mod huffman;
#[cfg(test)]
mod phase1_oracle;
#[cfg(test)]
mod phase2_oracle;
#[cfg(feature = "gpu-jpeg-huffman")]
pub(crate) mod quality;
mod scan;

pub use bitstream::{PackedBitstream, pack_be_words};
pub use codetable::{
    FullEntry, GpuCodetable, QUICK_CHECK_BITS, QUICK_TABLE_SIZE, QuickEntry, build_gpu_codetable,
};
pub use cpu_prepass::{JpegPreparedInput, build_mcu_schedule, prepare_jpeg};
pub use error::JpegGpuError;
pub use jpeg_framing::{JpegFramingError, decode_scan_symbols};
#[cfg(feature = "gpu-jpeg-huffman")]
pub use quality::pick_subsequence_size;
pub use scan::dispatch_blelloch_scan;

#[cfg(test)]
pub(crate) mod tests {
    pub mod fixtures;
    pub mod synthetic;
}
