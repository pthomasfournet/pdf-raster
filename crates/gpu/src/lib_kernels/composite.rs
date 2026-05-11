//! Porter-Duff source-over compositing on RGBA8 buffers.

use cudarc::driver::{CudaSlice, PushKernelArg};

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
        let stream = &self.stream;
        let d_src = stream.clone_htod(src)?;
        let d_dst = stream.clone_htod(dst.as_ref())?;

        let n_u32 = u32::try_from(src.len() / 4).expect("pixel count exceeds u32::MAX");
        self.launch_composite_async(&d_src, &d_dst, n_u32)?;

        stream.synchronize()?;
        stream.memcpy_dtoh(&d_dst, dst)?;
        Ok(())
    }

    /// Async kernel launch for Porter-Duff source-over composite.
    ///
    /// Caller is responsible for stream ordering (the launch is recorded
    /// into the context's default stream) and for any final D→H download.
    /// This function does **not** call `synchronize` and does **not**
    /// touch host memory; that's what makes it usable inside the
    /// per-page recording flow in `backend::cuda::page_recorder`.
    ///
    /// Both buffers are passed as `&CudaSlice<u8>` (immutable borrow);
    /// at the cudarc layer the kernel arg is the raw `CUdeviceptr`, so
    /// the &-vs-&mut distinction has no effect on the launch — it's
    /// the caller's responsibility to ensure no concurrent reader is
    /// active on `d_dst` (the single-stream serialisation invariant
    /// guarantees this for in-tree callers).
    ///
    /// `d_src` and `d_dst` must each contain `4 * n_pixels` bytes.
    ///
    /// # Errors
    ///
    /// Returns the underlying CUDA error if the launch fails.
    #[expect(
        unused_results,
        reason = "cudarc LaunchArgs::arg returns &mut Self for chaining; chain output is intentionally discarded"
    )]
    pub(crate) fn launch_composite_async(
        &self,
        d_src: &CudaSlice<u8>,
        d_dst: &CudaSlice<u8>,
        n_pixels: u32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let cfg = launch_cfg(n_pixels as usize);
        let stream = &self.stream;
        let mut builder = stream.launch_builder(&self.kernels.composite_rgba8);
        builder.arg(d_src).arg(d_dst).arg(&n_pixels);
        // SAFETY: 3 args match composite_rgba8 PTX signature; n_pixels is u32.
        unsafe { builder.launch(cfg) }?;
        Ok(())
    }
}
