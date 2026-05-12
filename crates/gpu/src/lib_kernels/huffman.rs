//! Parallel-Huffman JPEG decoder kernel dispatch.
//!
//! Per-phase launch helpers; the recorder's `record_huffman` picks
//! one based on `HuffmanParams.phase`.

use cudarc::driver::{CudaSlice, PushKernelArg};

use crate::GpuCtx;
use crate::backend::params::HUFFMAN_PHASE1_THREADS;

impl GpuCtx {
    /// Async launch of the Phase 1 intra-sequence-sync kernel.
    ///
    /// `s_info_out` is the per-subsequence `(p, n, c, z)` output:
    /// `num_subsequences * 16` bytes (uint4 per subsequence).
    ///
    /// # Errors
    /// Returns the underlying CUDA error if the kernel launch fails.
    #[expect(
        unused_results,
        reason = "cudarc LaunchArgs::arg returns &mut Self for chaining"
    )]
    #[expect(
        clippy::too_many_arguments,
        reason = "args mirror the .cu signature; grouping into a struct would just shuffle the bytes"
    )]
    pub(crate) fn launch_phase1_intra_sync_async(
        &self,
        bitstream: &CudaSlice<u8>,
        codebook: &CudaSlice<u8>,
        s_info_out: &CudaSlice<u8>,
        length_bits: u32,
        subsequence_bits: u32,
        num_subsequences: u32,
        num_components: u32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (num_subsequences.div_ceil(HUFFMAN_PHASE1_THREADS), 1, 1),
            block_dim: (HUFFMAN_PHASE1_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };

        let stream = &self.stream;
        let mut builder = stream.launch_builder(&self.kernels.phase1_intra_sync);
        builder
            .arg(bitstream)
            .arg(codebook)
            .arg(s_info_out)
            .arg(&length_bits)
            .arg(&subsequence_bits)
            .arg(&num_subsequences)
            .arg(&num_components);
        // SAFETY: arg count + types match the PTX entry's signature;
        // buffer capacities were validated by `HuffmanParams::validate`
        // before this call.
        unsafe { builder.launch(cfg) }?;
        Ok(())
    }
}
