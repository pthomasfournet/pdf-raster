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

    /// Async launch of one Phase 2 (inter-sequence sync) pass.
    ///
    /// Reads `s_info` + writes `s_info` (in-place advance) and
    /// `sync_flags` (one u32 per subseq; 1 = synced, 0 = unsynced
    /// and advanced this pass). Host loops the dispatch until all
    /// flags are 1 or the retry bound is exhausted.
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
    pub(crate) fn launch_phase2_inter_sync_async(
        &self,
        bitstream: &CudaSlice<u8>,
        codebook: &CudaSlice<u8>,
        s_info: &CudaSlice<u8>,
        sync_flags: &CudaSlice<u8>,
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
        let mut builder = stream.launch_builder(&self.kernels.phase2_inter_sync);
        builder
            .arg(bitstream)
            .arg(codebook)
            .arg(s_info)
            .arg(sync_flags)
            .arg(&length_bits)
            .arg(&subsequence_bits)
            .arg(&num_subsequences)
            .arg(&num_components);
        // SAFETY: arg count + types match the PTX phase2_inter_sync
        // signature; buffer capacities validated by HuffmanParams.
        unsafe { builder.launch(cfg) }?;
        Ok(())
    }

    /// Async launch of the Phase 4 re-decode + write kernel.
    ///
    /// One thread per subsequence. Re-walks the subseq's owned region
    /// `[prev.p, me.p)` and emits each decoded symbol to
    /// `symbols_out[offsets[seq_idx] + local_n]`.
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
    pub(crate) fn launch_phase4_redecode_async(
        &self,
        bitstream: &CudaSlice<u8>,
        codebook: &CudaSlice<u8>,
        s_info: &CudaSlice<u8>,
        offsets: &CudaSlice<u8>,
        symbols_out: &CudaSlice<u8>,
        decode_status: &CudaSlice<u8>,
        length_bits: u32,
        total_symbols: u32,
        num_subsequences: u32,
        num_components: u32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (num_subsequences.div_ceil(HUFFMAN_PHASE1_THREADS), 1, 1),
            block_dim: (HUFFMAN_PHASE1_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };

        let stream = &self.stream;
        let mut builder = stream.launch_builder(&self.kernels.phase4_redecode);
        builder
            .arg(bitstream)
            .arg(codebook)
            .arg(s_info)
            .arg(offsets)
            .arg(symbols_out)
            .arg(decode_status)
            .arg(&length_bits)
            .arg(&total_symbols)
            .arg(&num_subsequences)
            .arg(&num_components);
        // SAFETY: arg count + types match the PTX phase4_redecode
        // signature; buffer capacities validated by HuffmanParams.
        // Note: subsequence_bits is not in the CUDA Phase 4 arg list
        // because Phase 4 reads end_p from the snapshot's `me.x`; the
        // Slang side keeps it via the shared push cbuffer (other
        // phases use it).
        unsafe { builder.launch(cfg) }?;
        Ok(())
    }

    /// Async launch of one JPEG-framed Phase 2 (inter-sequence sync) pass.
    ///
    /// Reads `s_info` + writes `s_info` (in-place JPEG advance) and
    /// `sync_flags` (one u32 per subseq; 1 = synced, 0 = unsynced
    /// and advanced this pass). Host loops until all flags are 1 or
    /// the retry bound is exhausted.
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
    pub(crate) fn launch_jpeg_phase2_inter_sync_async(
        &self,
        bitstream: &CudaSlice<u8>,
        codebook: &CudaSlice<u8>,
        dc_codebook: &CudaSlice<u8>,
        mcu_schedule: &CudaSlice<u8>,
        s_info: &CudaSlice<u8>,
        sync_flags: &CudaSlice<u8>,
        length_bits: u32,
        subsequence_bits: u32,
        num_subsequences: u32,
        blocks_per_mcu: u32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (num_subsequences.div_ceil(HUFFMAN_PHASE1_THREADS), 1, 1),
            block_dim: (HUFFMAN_PHASE1_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };

        let stream = &self.stream;
        let mut builder = stream.launch_builder(&self.kernels.jpeg_phase2_inter_sync);
        builder
            .arg(bitstream)
            .arg(codebook)
            .arg(dc_codebook)
            .arg(mcu_schedule)
            .arg(s_info)
            .arg(sync_flags)
            .arg(&length_bits)
            .arg(&subsequence_bits)
            .arg(&num_subsequences)
            .arg(&blocks_per_mcu);
        // SAFETY: arg count + types match the PTX entry's signature;
        // buffer capacities were validated by `HuffmanParams::validate`
        // before this call.
        unsafe { builder.launch(cfg) }?;
        Ok(())
    }

    /// Async launch of the JPEG-framed Phase 4 re-decode + write kernel.
    ///
    /// One thread per subsequence. Inherits `(p, block_in_mcu, z_in_block)`
    /// from the predecessor's Phase-1 snapshot and re-walks the JPEG stream
    /// until `s_info[seq_idx].x`, emitting each decoded symbol to
    /// `symbols_out[offsets[seq_idx] + local_n]`.
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
    pub(crate) fn launch_jpeg_phase4_redecode_async(
        &self,
        bitstream: &CudaSlice<u8>,
        codebook: &CudaSlice<u8>,
        dc_codebook: &CudaSlice<u8>,
        mcu_schedule: &CudaSlice<u8>,
        s_info: &CudaSlice<u8>,
        offsets: &CudaSlice<u8>,
        symbols_out: &CudaSlice<u8>,
        decode_status: &CudaSlice<u8>,
        length_bits: u32,
        total_symbols: u32,
        num_subsequences: u32,
        blocks_per_mcu: u32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (num_subsequences.div_ceil(HUFFMAN_PHASE1_THREADS), 1, 1),
            block_dim: (HUFFMAN_PHASE1_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };

        let stream = &self.stream;
        let mut builder = stream.launch_builder(&self.kernels.jpeg_phase4_redecode);
        builder
            .arg(bitstream)
            .arg(codebook)
            .arg(dc_codebook)
            .arg(mcu_schedule)
            .arg(s_info)
            .arg(offsets)
            .arg(symbols_out)
            .arg(decode_status)
            .arg(&length_bits)
            .arg(&total_symbols)
            .arg(&num_subsequences)
            .arg(&blocks_per_mcu);
        // SAFETY: arg count + types match the PTX entry's signature;
        // buffer capacities were validated by `HuffmanParams::validate`
        // before this call.
        unsafe { builder.launch(cfg) }?;
        Ok(())
    }

    /// Async launch of the JPEG-framed Phase 1 intra-sequence-sync
    /// kernel.  Same dispatch grid as the synthetic
    /// [`Self::launch_phase1_intra_sync_async`]; the kernel reads
    /// `dc_codebook` for the DC slot of each block and `mcu_schedule`
    /// to pick the per-block `(dc_sel, ac_sel)` pair.
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
    pub(crate) fn launch_jpeg_phase1_intra_sync_async(
        &self,
        bitstream: &CudaSlice<u8>,
        codebook: &CudaSlice<u8>,
        dc_codebook: &CudaSlice<u8>,
        mcu_schedule: &CudaSlice<u8>,
        s_info_out: &CudaSlice<u8>,
        length_bits: u32,
        subsequence_bits: u32,
        num_subsequences: u32,
        blocks_per_mcu: u32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (num_subsequences.div_ceil(HUFFMAN_PHASE1_THREADS), 1, 1),
            block_dim: (HUFFMAN_PHASE1_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };

        let stream = &self.stream;
        let mut builder = stream.launch_builder(&self.kernels.jpeg_phase1_intra_sync);
        builder
            .arg(bitstream)
            .arg(codebook)
            .arg(dc_codebook)
            .arg(mcu_schedule)
            .arg(s_info_out)
            .arg(&length_bits)
            .arg(&subsequence_bits)
            .arg(&num_subsequences)
            .arg(&blocks_per_mcu);
        // SAFETY: arg count + types match the PTX entry's signature;
        // buffer capacities were validated by `HuffmanParams::validate`
        // before this call.
        unsafe { builder.launch(cfg) }?;
        Ok(())
    }
}
