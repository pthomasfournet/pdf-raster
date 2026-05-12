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
#[cfg(test)]
mod cpu_reference;
mod error;
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
mod scan;

pub use bitstream::{PackedBitstream, pack_be_words};
pub use codetable::{
    FullEntry, GpuCodetable, QUICK_CHECK_BITS, QUICK_TABLE_SIZE, QuickEntry, build_gpu_codetable,
};
pub use error::JpegGpuError;
pub use scan::dispatch_blelloch_scan;

#[cfg(test)]
pub(crate) mod tests {
    pub mod fixtures;
    pub mod synthetic;
}
