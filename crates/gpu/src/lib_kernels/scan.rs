//! Blelloch exclusive scan kernel dispatch.
//!
//! Three CUDA entry points (`scan_per_workgroup`, `scan_block_sums`,
//! `scan_scatter`) cooperate to scan a flat u32 array. The host
//! invokes each phase by calling `launch_scan_async` with the
//! matching `ScanPhase`; the kernel side runs as one PTX module with
//! three `__global__` functions.

use cudarc::driver::{CudaSlice, PushKernelArg};

use crate::GpuCtx;
use crate::backend::params::{SCAN_WORKGROUP_SIZE, ScanPhase};

impl GpuCtx {
    /// Async launch of one Blelloch scan phase.
    ///
    /// Per-phase grid sizing:
    /// - `PerWorkgroup` / `ScatterBlockSums`: one workgroup per tile
    ///   of 1024 elements. Grid = `ceil(len_elems / 1024)` blocks.
    /// - `BlockSums`: exactly one workgroup over the (≤ 1024) block
    ///   sums. Grid = 1 block.
    ///
    /// All three phases take `(data, block_sums, len_elems)` in the
    /// same arg slot order so the host-side dispatcher rebinds
    /// nothing between phases. The `BlockSums` kernel ignores
    /// `data`; `Scatter` reads `block_sums[gid]` and adds it into
    /// its tile of `data`.
    ///
    /// # Errors
    /// Returns the underlying CUDA error if the kernel launch fails.
    #[expect(
        unused_results,
        reason = "cudarc LaunchArgs::arg returns &mut Self for chaining"
    )]
    pub(crate) fn launch_scan_async(
        &self,
        data: &CudaSlice<u8>,
        block_sums: &CudaSlice<u8>,
        len_elems: u32,
        phase: ScanPhase,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let tile = 2 * SCAN_WORKGROUP_SIZE;
        let block_count = len_elems.div_ceil(tile);
        let grid_dim = match phase {
            ScanPhase::PerWorkgroup | ScanPhase::ScatterBlockSums => (block_count, 1, 1),
            ScanPhase::BlockSums => (1, 1, 1),
        };
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim,
            block_dim: (SCAN_WORKGROUP_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };

        let stream = &self.stream;
        let kernel = match phase {
            ScanPhase::PerWorkgroup => &self.kernels.scan_per_workgroup,
            ScanPhase::BlockSums => &self.kernels.scan_block_sums,
            ScanPhase::ScatterBlockSums => &self.kernels.scan_scatter,
        };
        let mut builder = stream.launch_builder(kernel);
        // block_sums kernel takes (block_sums, len_elems); the other
        // two take (data, block_sums, len_elems). Arg order matches
        // the .cu signatures.
        match phase {
            ScanPhase::PerWorkgroup | ScanPhase::ScatterBlockSums => {
                builder.arg(data).arg(block_sums).arg(&len_elems);
            }
            ScanPhase::BlockSums => {
                builder.arg(block_sums).arg(&len_elems);
            }
        }
        // SAFETY: kernel arg counts/types match the PTX entry's
        // signature for this phase; `data` and `block_sums` capacities
        // were validated by `ScanParams::validate` before this call.
        unsafe { builder.launch(cfg) }?;
        Ok(())
    }
}
