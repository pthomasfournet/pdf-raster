//! Supersampled AA fill kernel dispatch.

use cudarc::driver::{LaunchConfig, PushKernelArg};

use crate::{GPU_AA_FILL_THRESHOLD, GpuCtx, fill::aa_fill_cpu};

impl GpuCtx {
    /// Compute per-pixel AA coverage for a filled path using 64-sample jittered
    /// supersampling on the GPU.
    ///
    /// `segs` is a flat `[x0, y0, x1, y1]` f32 slice — 4 floats per segment.
    /// `x_min` / `y_min` are the device-pixel coordinates of the top-left corner of
    /// the output coverage rectangle. The output is `width * height` bytes, one byte
    /// per pixel (0 = fully outside, 255 = fully inside).
    ///
    /// Falls back to [`aa_fill_cpu`] when the pixel count is below
    /// [`GPU_AA_FILL_THRESHOLD`] or `segs` is empty.
    ///
    /// # Errors
    ///
    /// Returns an error if GPU dispatch or data transfer fails.
    ///
    /// # Panics
    ///
    /// Panics if `segs.len()` is not a multiple of 4 or if `width * height` overflows
    /// `u32::MAX`.
    pub fn aa_fill(
        &self,
        segs: &[f32],
        x_min: f32,
        y_min: f32,
        width: u32,
        height: u32,
        eo: bool,
    ) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        assert!(
            segs.len().is_multiple_of(4),
            "segs.len() must be a multiple of 4 (got {})",
            segs.len()
        );
        let n_pixels = (width as usize)
            .checked_mul(height as usize)
            .expect("width × height overflows usize");

        if segs.is_empty() || n_pixels < GPU_AA_FILL_THRESHOLD {
            return Ok(aa_fill_cpu(segs, x_min, y_min, width, height, eo));
        }

        self.aa_fill_gpu(segs, x_min, y_min, width, height, eo)
    }

    /// Unconditional GPU dispatch for `aa_fill` (skips threshold check).
    ///
    /// Use this when the caller has already decided GPU is appropriate
    /// (e.g. benchmarking or when the area is known to be large).
    ///
    /// # Errors
    ///
    /// Returns an error if GPU data transfer or kernel launch fails.
    ///
    /// # Panics
    ///
    /// Panics if `segs.len()` is not a multiple of 4 or if `width * height`
    /// overflows `u32::MAX`.
    pub fn aa_fill_gpu(
        &self,
        segs: &[f32],
        x_min: f32,
        y_min: f32,
        width: u32,
        height: u32,
        eo: bool,
    ) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        assert!(
            segs.len().is_multiple_of(4),
            "segs.len() must be a multiple of 4 (got {})",
            segs.len()
        );
        let n_pixels = (width as usize)
            .checked_mul(height as usize)
            .expect("width × height overflows usize");
        let n_segs = u32::try_from(segs.len() / 4).expect("segment count exceeds u32::MAX");
        let n_pixels_u32 = u32::try_from(n_pixels).expect("pixel count exceeds u32::MAX");

        let stream = &self.stream;

        // Upload segments and allocate coverage output on device.
        let d_segs = stream.clone_htod(segs)?;
        let d_coverage_init = vec![0u8; n_pixels];
        let mut d_coverage = stream.clone_htod(&d_coverage_init)?;

        // Launch: one block per output pixel, 64 threads per block.
        let cfg = LaunchConfig {
            grid_dim: (n_pixels_u32, 1, 1),
            block_dim: (64, 1, 1),
            shared_mem_bytes: 8, // two i32 warp_counts
        };

        let eo_int: i32 = i32::from(eo);
        let mut builder = stream.launch_builder(&self.kernels.aa_fill);
        let _ = builder.arg(&d_segs);
        let _ = builder.arg(&n_segs);
        let _ = builder.arg(&x_min);
        let _ = builder.arg(&y_min);
        let _ = builder.arg(&width);
        let _ = builder.arg(&height);
        let _ = builder.arg(&eo_int);
        let _ = builder.arg(&mut d_coverage);
        // SAFETY: kernel arguments match the PTX signature; bounds verified above.
        let _ = unsafe { builder.launch(cfg) }?;

        stream.synchronize()?;
        let mut coverage = vec![0u8; n_pixels];
        stream.memcpy_dtoh(&d_coverage, &mut coverage)?;
        Ok(coverage)
    }
}
