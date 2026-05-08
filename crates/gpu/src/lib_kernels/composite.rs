//! Porter-Duff source-over compositing on RGBA8 buffers.

use cudarc::driver::PushKernelArg;

use crate::{GPU_COMPOSITE_THRESHOLD, GpuCtx, composite::composite_rgba8_cpu, launch_cfg};

impl GpuCtx {
    /// Porter-Duff source-over compositing on RGBA8 pixel pairs.
    ///
    /// `src` and `dst` must have the same length (4 × `n_pixels` bytes).
    ///
    /// Falls back to CPU if the pixel count is below [`GPU_COMPOSITE_THRESHOLD`].
    ///
    /// # Errors
    ///
    /// Returns an error if GPU dispatch or data transfer fails.
    ///
    /// # Panics
    ///
    /// Panics if `src.len() != dst.len()` or `src.len()` is not a multiple of 4.
    pub fn composite_rgba8(
        &self,
        src: &[u8],
        dst: &mut [u8],
    ) -> Result<(), Box<dyn std::error::Error>> {
        assert_eq!(src.len(), dst.len());
        assert!(src.len().is_multiple_of(4));
        let n = src.len() / 4;

        if n < GPU_COMPOSITE_THRESHOLD {
            composite_rgba8_cpu(src, dst);
            return Ok(());
        }

        self.composite_rgba8_gpu(src, dst)
    }

    /// Unconditional GPU dispatch for `composite_rgba8` (skips threshold check).
    fn composite_rgba8_gpu(
        &self,
        src: &[u8],
        dst: &mut [u8],
    ) -> Result<(), Box<dyn std::error::Error>> {
        let n = src.len() / 4;
        let n_u32 = u32::try_from(n).expect("pixel count exceeds u32::MAX");
        let stream = &self.stream;
        let d_src = stream.clone_htod(src)?;
        let mut d_dst = stream.clone_htod(dst.as_ref())?;

        let cfg = launch_cfg(n);
        let mut builder = stream.launch_builder(&self.kernels.composite_rgba8);
        let _ = builder.arg(&d_src);
        let _ = builder.arg(&mut d_dst);
        let _ = builder.arg(&n_u32);
        // SAFETY: 3 args match composite_rgba8 PTX signature; n_u32 verified above.
        let _ = unsafe { builder.launch(cfg) }?;

        stream.synchronize()?;
        stream.memcpy_dtoh(&d_dst, dst)?;
        Ok(())
    }
}
