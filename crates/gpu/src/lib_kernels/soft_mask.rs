//! Soft-mask alpha multiplication on RGBA8 pixel buffers.

use cudarc::driver::PushKernelArg;

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
        let n = mask.len();
        let n_u32 = u32::try_from(n).expect("pixel count exceeds u32::MAX");
        let stream = &self.stream;
        let mut d_pixels = stream.clone_htod(pixels.as_ref())?;
        let d_mask = stream.clone_htod(mask)?;

        let cfg = launch_cfg(n);
        let mut builder = stream.launch_builder(&self.kernels.apply_soft_mask);
        let _ = builder.arg(&mut d_pixels);
        let _ = builder.arg(&d_mask);
        let _ = builder.arg(&n_u32);
        // SAFETY: 3 args match apply_soft_mask PTX signature; n_u32 verified above.
        let _ = unsafe { builder.launch(cfg) }?;

        stream.synchronize()?;
        stream.memcpy_dtoh(&d_pixels, pixels)?;
        Ok(())
    }
}
