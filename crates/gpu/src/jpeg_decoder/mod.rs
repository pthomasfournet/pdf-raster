//! Parallel-Huffman JPEG decoder for the GPU compute path.
//!
//! Algorithm: Weißenberger & Schmidt 2021, "A massively parallel
//! Huffman decoder", §III–IV. Plan and design rationale live in
//! `docs/superpowers/plans/2026-05-11-gpu-jpeg-huffman-v2.md`.
//!
//! This module is the orchestrator; GPU kernels (when they land) live
//! under `crates/gpu/kernels/jpeg/`. Today the surface is the
//! CPU-side primitives every backend will share: bitstream packing,
//! 2-tier codetable construction, a scalar reference decoder, and a
//! synthetic-stream oracle used by cross-backend bit-identity tests.

mod bitstream;
mod codetable;
mod error;

pub use bitstream::{PackedBitstream, pack_be_words};
pub use codetable::{
    FullEntry, GpuCodetable, QUICK_CHECK_BITS, QUICK_TABLE_SIZE, QuickEntry, build_gpu_codetable,
};
pub use error::JpegGpuError;

#[cfg(test)]
pub(crate) mod tests {
    pub mod synthetic;
}
