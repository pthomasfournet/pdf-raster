//! Soft-mask alpha multiplication on RGBA8 pixel buffers.

use cudarc::driver::{CudaSlice, PushKernelArg};

use crate::{GPU_SOFTMASK_THRESHOLD, GpuCtx, composite::apply_soft_mask_cpu, launch_cfg};

impl GpuCtx {
    /// Multiply each RGBA pixel's alpha channel by the corresponding soft-mask byte.
    ///
    /// `pixels` is RGBA8 interleaved; `mask` is one byte per pixel.
    ///
    /// # Errors
    ///
    /// Returns an error if GPU dispatch or data transfer fails.
    ///
    /// # Panics
    ///
    /// Panics if `pixels.len() != mask.len() * 4`.
    pub fn apply_soft_mask(
        &self,
        pixels: &mut [u8],
        mask: &[u8],
    ) -> Result<(), Box<dyn std::error::Error>> {
        assert_eq!(pixels.len(), mask.len() * 4);
        let n = mask.len();

        if n < GPU_SOFTMASK_THRESHOLD {
            apply_soft_mask_cpu(pixels, mask);
            return Ok(());
        }

        self.apply_soft_mask_gpu(pixels, mask)
    }

    /// Unconditional GPU dispatch for `apply_soft_mask` (skips threshold check).
    fn apply_soft_mask_gpu(
        &self,
        pixels: &mut [u8],
        mask: &[u8],
    ) -> Result<(), Box<dyn std::error::Error>> {
        let stream = &self.stream;
        let d_pixels = stream.clone_htod(pixels.as_ref())?;
        let d_mask = stream.clone_htod(mask)?;

        let n_u32 = u32::try_from(mask.len()).expect("pixel count exceeds u32::MAX");
        self.launch_soft_mask_async(&d_pixels, &d_mask, n_u32)?;

        stream.synchronize()?;
        stream.memcpy_dtoh(&d_pixels, pixels)?;
        Ok(())
    }

    /// Async kernel launch for soft-mask alpha multiplication.
    ///
    /// Caller is responsible for stream ordering and any final D→H
    /// download. This function does **not** call `synchronize` and does
    /// **not** touch host memory.
    ///
    /// `d_pixels` is the read-modify-write target; passed as
    /// `&CudaSlice<u8>` because cudarc's launch builder accepts both
    /// shared and exclusive borrows for kernel args (the raw device
    /// pointer is what gets pushed).  Single-stream serialisation
    /// makes the in-place mutation race-free for in-tree callers.
    ///
    /// `d_pixels` must contain `4 * n_pixels` bytes; `d_mask` must
    /// contain `n_pixels` bytes.
    ///
    /// # Errors
    ///
    /// Returns the underlying CUDA error if the launch fails.
    #[expect(
        unused_results,
        reason = "cudarc LaunchArgs::arg returns &mut Self for chaining; chain output is intentionally discarded"
    )]
    pub(crate) fn launch_soft_mask_async(
        &self,
        d_pixels: &CudaSlice<u8>,
        d_mask: &CudaSlice<u8>,
        n_pixels: u32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let cfg = launch_cfg(n_pixels as usize);
        let stream = &self.stream;
        let mut builder = stream.launch_builder(&self.kernels.apply_soft_mask);
        builder.arg(d_pixels).arg(d_mask).arg(&n_pixels);
        // SAFETY: 3 args match apply_soft_mask PTX signature; n_pixels is u32.
        unsafe { builder.launch(cfg) }?;
        Ok(())
    }
}
