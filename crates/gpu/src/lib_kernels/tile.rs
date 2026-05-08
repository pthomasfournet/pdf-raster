//! Tile-parallel analytical fill kernel dispatch.

use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};

use crate::{GpuCtx, TILE_H, TILE_W, fill::TileRecord};

impl GpuCtx {
    /// Tile-parallel analytical fill rasterisation using signed-area integration.
    ///
    /// This is the GPU equivalent of the CPU scanline scanner but uses analytical
    /// per-pixel coverage (analytical trapezoid integrals) rather than sampling.
    ///
    /// All coordinates in `records` are already tile-local (produced by
    /// [`build_tile_records`]); no origin offset is applied in the kernel.
    ///
    /// # Arguments
    ///
    /// - `records` — tile records sorted by `(tile_y << 16 | tile_x)`, one per
    ///   (segment, tile-row) crossing.  Build with [`build_tile_records`].
    /// - `tile_starts` / `tile_counts` — prefix-sum index into `records` per flat
    ///   tile index `tile_y * grid_w + tile_x`.  Both have length `grid_w * grid_h`.
    /// - `grid_w` — number of tiles in the x direction (`width.div_ceil(TILE_W)`).
    /// - `width` / `height` — fill bbox size in device pixels (coverage buffer dims).
    /// - `eo` — `true` for even-odd fill rule, `false` for non-zero winding.
    ///
    /// # Returns
    ///
    /// A `Vec<u8>` of `width × height` coverage bytes (0 = outside, 255 = inside).
    ///
    /// # Errors
    ///
    /// Returns an error if GPU data transfer or kernel launch fails.
    ///
    /// # Panics
    ///
    /// Panics if `tile_starts.len() != tile_counts.len()`.
    #[expect(
        clippy::too_many_arguments,
        reason = "all 7 args are required: records + index arrays + grid/pixel dims + fill rule; no grouping is natural here"
    )]
    pub fn tile_fill(
        &self,
        records: &[TileRecord],
        tile_starts: &[u32],
        tile_counts: &[u32],
        grid_w: u32,
        width: u32,
        height: u32,
        eo: bool,
    ) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        assert_eq!(
            tile_starts.len(),
            tile_counts.len(),
            "tile_starts and tile_counts must have the same length"
        );
        let n_pixels = (width as usize)
            .checked_mul(height as usize)
            .expect("width × height overflows usize");
        if n_pixels == 0 {
            return Ok(Vec::new());
        }
        let stream = &self.stream;

        // Upload inputs as byte buffers so they match the trait's
        // `B::DeviceBuffer = CudaSlice<u8>` shape used by the async helper.
        let placeholder = [TileRecord::default()];
        let records_bytes: &[u8] = if records.is_empty() {
            // cudarc refuses zero-size allocations — use a one-record placeholder.
            bytemuck::cast_slice(&placeholder)
        } else {
            bytemuck::cast_slice(records)
        };
        let d_records = stream.clone_htod(records_bytes)?;
        let d_tile_starts = stream.clone_htod(bytemuck::cast_slice::<u32, u8>(tile_starts))?;
        let d_tile_counts = stream.clone_htod(bytemuck::cast_slice::<u32, u8>(tile_counts))?;
        let d_cov_init = vec![0u8; n_pixels];
        let d_coverage = stream.clone_htod(&d_cov_init)?;

        self.launch_tile_fill_async(
            &d_records,
            &d_tile_starts,
            &d_tile_counts,
            grid_w,
            width,
            height,
            eo,
            &d_coverage,
        )?;

        stream.synchronize()?;
        let mut coverage = vec![0u8; n_pixels];
        stream.memcpy_dtoh(&d_coverage, &mut coverage)?;
        Ok(coverage)
    }

    /// Async kernel launch for the tile-parallel analytical fill kernel.
    ///
    /// Caller is responsible for stream ordering and any final D→H
    /// download. This function does **not** call `synchronize` and does
    /// **not** touch host memory.
    ///
    /// All buffers are passed by reference; the caller owns them and is
    /// responsible for keeping them alive until the stream has executed
    /// the launch.
    ///
    /// # Errors
    ///
    /// Returns the underlying CUDA error if the launch fails.
    #[expect(
        clippy::too_many_arguments,
        reason = "all args mirror the PTX kernel signature; grouping would obscure the mapping"
    )]
    pub(crate) fn launch_tile_fill_async(
        &self,
        d_records: &CudaSlice<u8>,
        d_tile_starts: &CudaSlice<u8>,
        d_tile_counts: &CudaSlice<u8>,
        grid_w: u32,
        width: u32,
        height: u32,
        eo: bool,
        d_coverage: &CudaSlice<u8>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let grid_h = height.div_ceil(TILE_H);
        let cfg = LaunchConfig {
            grid_dim: (grid_w, grid_h, 1),
            block_dim: (TILE_W, TILE_H, 1),
            shared_mem_bytes: 0,
        };

        let eo_int: i32 = i32::from(eo);
        let stream = &self.stream;
        let mut builder = stream.launch_builder(&self.kernels.tile_fill);
        let _ = builder.arg(d_records);
        let _ = builder.arg(d_tile_starts);
        let _ = builder.arg(d_tile_counts);
        let _ = builder.arg(&grid_w);
        let _ = builder.arg(&width);
        let _ = builder.arg(&height);
        let _ = builder.arg(&eo_int);
        let _ = builder.arg(d_coverage);
        // SAFETY: kernel arguments match the PTX signature exactly (8 args, no
        // x_min/y_min — coords are tile-local from build_tile_records).
        let _ = unsafe { builder.launch(cfg) }?;
        Ok(())
    }
}
