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
    pub mod synthetic;
}
