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
#[cfg(feature = "gpu-jpeg-huffman")]
pub mod decoder;
#[cfg(feature = "gpu-jpeg-huffman")]
pub mod device_image;
mod dispatch_util;
mod error;
#[cfg(feature = "gpu-jpeg-huffman")]
pub(crate) mod huffman;
mod jpeg_framing;
pub(crate) mod phase1_oracle;
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
#[cfg(feature = "gpu-jpeg-huffman")]
pub use decoder::JpegGpuDecoder;
#[cfg(feature = "gpu-jpeg-huffman")]
pub use device_image::DeviceImage;
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
