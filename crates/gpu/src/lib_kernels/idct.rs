//! IDCT + dequant + colour-conversion kernel dispatch (Phase 5).

use cudarc::driver::{CudaSlice, PushKernelArg};

use crate::GpuCtx;

impl GpuCtx {
    /// Async launch of the IDCT + dequant + colour-conversion kernel.
    ///
    /// Dispatches one 8×8×3-thread workgroup per 8×8 MCU block.  The
    /// kernel performs zigzag inverse, dequantisation, 2-D LLM IDCT,
    /// YCbCr→RGB colour conversion, and RGBA8 packing.
    ///
    /// # Errors
    /// Returns the underlying CUDA error if the kernel launch fails.
    #[expect(
        unused_results,
        reason = "cudarc LaunchArgs::arg returns &mut Self for chaining"
    )]
    #[expect(
        clippy::too_many_arguments,
        reason = "args mirror the .cu signature; the kernel has 10 scalar push-constant args"
    )]
    pub(crate) fn launch_idct_dequant_colour_async(
        &self,
        coefficients: &CudaSlice<u8>,
        qtables: &CudaSlice<u8>,
        dc_values: &CudaSlice<u8>,
        pixels_rgba: &CudaSlice<u8>,
        width: u32,
        height: u32,
        num_components: u32,
        blocks_wide: u32,
        blocks_high: u32,
        num_qtables: u32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // One workgroup per (block_x, block_y); Z=1 because threads
        // cover all 3 components via the numthreads(8,8,3) block shape.
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (blocks_wide, blocks_high, 1),
            block_dim: (8, 8, 3),
            shared_mem_bytes: 0,
        };

        let stream = &self.stream;
        let mut builder = stream.launch_builder(&self.kernels.idct_dequant_colour);
        builder
            .arg(coefficients)
            .arg(qtables)
            .arg(dc_values)
            .arg(pixels_rgba)
            .arg(&width)
            .arg(&height)
            .arg(&num_components)
            .arg(&blocks_wide)
            .arg(&blocks_high)
            .arg(&num_qtables);
        // SAFETY: arg count + types match the PTX entry's signature;
        // buffer capacities were validated by `IdctParams::validate`
        // before this call.
        unsafe { builder.launch(cfg) }?;
        Ok(())
    }
}
