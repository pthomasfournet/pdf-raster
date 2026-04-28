//! GPU-accelerated rasterizer helpers (CUDA via cudarc).
//!
//! Gate with `feature = "gpu"` in crates that depend on this.
//!
//! # Usage
//!
//! ```no_run
//! use gpu::GpuCtx;
//! let ctx = GpuCtx::init().expect("no CUDA device");
//! // … call ctx.composite_rgba8(src, dst) etc.
//! ```
//!
//! # Feature flags
//!
//! - `nvjpeg` — enables [`nvjpeg`] module for GPU-accelerated JPEG decoding
//!   via NVIDIA nvJPEG.  Requires `libnvjpeg.so` at link time.

#[cfg(feature = "nvjpeg")]
pub mod nvjpeg;

use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaStream, DeviceRepr, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::Ptx;

const PTX_COMPOSITE: &str = include_str!(concat!(env!("OUT_DIR"), "/composite_rgba8.ptx"));
const PTX_SOFT_MASK: &str = include_str!(concat!(env!("OUT_DIR"), "/apply_soft_mask.ptx"));
const PTX_AA_FILL: &str = include_str!(concat!(env!("OUT_DIR"), "/aa_fill.ptx"));
const PTX_TILE_FILL: &str = include_str!(concat!(env!("OUT_DIR"), "/tile_fill.ptx"));
const PTX_ICC_CLUT: &str = include_str!(concat!(env!("OUT_DIR"), "/icc_clut.ptx"));

/// Threshold in pixels below which CPU is faster than GPU dispatch overhead.
pub const GPU_COMPOSITE_THRESHOLD: usize = 500_000;
/// Threshold for soft-mask application.
pub const GPU_SOFTMASK_THRESHOLD: usize = 500_000;
/// Minimum fill area (pixels) for GPU supersampled AA to be faster than CPU.
///
/// Below this threshold the H2D/D2H transfer latency for the segment list and
/// coverage buffer dominates. Calibrated for RTX 5070 + `PCIe` 5.0 at ~150 DPI.
pub const GPU_AA_FILL_THRESHOLD: usize = 16_384;
/// Minimum fill area (pixels) for the tile-parallel analytical fill to be faster
/// than the GPU warp-ballot AA kernel.
///
/// Tile fill incurs sorting overhead; below this threshold the AA kernel is faster.
pub const GPU_TILE_FILL_THRESHOLD: usize = 65_536;
/// Tile width in pixels (must match `TILE_W` in `tile_fill.cu`).
pub const TILE_W: u32 = 16;
/// Tile height in pixels (must match `TILE_H` in `tile_fill.cu`).
pub const TILE_H: u32 = 16;
/// Minimum pixel count for GPU ICC CMYK→RGB transform to beat CPU + `PCIe` overhead.
///
/// Below this threshold H2D/D2H transfer latency dominates; use the CPU fallback.
/// Conservative default aligned with composite/softmask; calibrate against actual
/// `PCIe` 5.0 latency on the target machine once the native path is hot.
pub const GPU_ICC_CLUT_THRESHOLD: usize = 500_000;

/// One tile record per (segment, tile-row) crossing.
///
/// Layout must match `struct TileRecord` in `tile_fill.cu` exactly.
/// The struct is `repr(C)` and 32 bytes so that `bytemuck::cast_slice` can
/// transmit it to the GPU without additional copying.
#[derive(Clone, Copy, Default)]
#[repr(C)]
pub struct TileRecord {
    /// Sort key: `(tile_y << 16) | tile_x`.  Set by [`build_tile_records`].
    pub key: u32,
    /// Segment x at the top of the segment's y-extent within this tile (tile-local).
    pub x_enter: f32,
    /// Slope: dx/dy in device pixels.
    pub dxdy: f32,
    /// Segment start y within this tile row (0..`TILE_H`).
    pub y0_tile: f32,
    /// Segment end y within this tile row (0..`TILE_H`).
    pub y1_tile: f32,
    /// Sign: `+1.0` for an upward-crossing segment, `-1.0` for downward.
    pub sign: f32,
    /// Padding (must be 0).
    #[expect(
        clippy::pub_underscore_fields,
        reason = "padding field required for repr(C) alignment to match tile_fill.cu struct layout"
    )]
    pub _pad: u32,
    /// Padding (must be 0).
    #[expect(
        clippy::pub_underscore_fields,
        reason = "padding field required for repr(C) alignment to match tile_fill.cu struct layout"
    )]
    pub _pad2: u32,
}

// SAFETY: TileRecord is repr(C), all fields are primitive types (u32, f32), no
// uninitialised padding — bytemuck::Pod and cudarc::DeviceRepr are safe.
unsafe impl bytemuck::Pod for TileRecord {}
unsafe impl bytemuck::Zeroable for TileRecord {}
// SAFETY: TileRecord has no pointer types or other non-device-representable fields;
// all fields are plain u32/f32 with repr(C) alignment.
unsafe impl DeviceRepr for TileRecord {}

/// Build a sorted list of [`TileRecord`]s from a flat segment list, plus the
/// `tile_starts` / `tile_counts` index arrays required by [`GpuCtx::tile_fill`].
///
/// `segs` is packed `[x0, y0, x1, y1]` per segment in device pixels, same
/// format as [`GpuCtx::aa_fill`].  `x_min`, `y_min`, `width`, `height` define
/// the fill bounding box in device pixels.
///
/// Returns `(records, tile_starts, tile_counts, grid_w)`.
///
/// # Panics
///
/// Panics if `segs.len()` is not a multiple of 4, or if `width` or `height`
/// require more than 65535 tiles in either dimension (i.e. exceed `65535 × TILE_W`
/// or `65535 × TILE_H` pixels) — the sort key packs tile coordinates into 16 bits each.
#[must_use]
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    reason = "tile indices are bounds-checked to fit u16; f32 precision sufficient for tile geometry at realistic page sizes"
)]
pub fn build_tile_records(
    segs: &[f32],
    x_min: f32,
    y_min: f32,
    width: u32,
    height: u32,
) -> (Vec<TileRecord>, Vec<u32>, Vec<u32>, u32) {
    assert!(
        segs.len().is_multiple_of(4),
        "segs.len() must be a multiple of 4 (got {})",
        segs.len()
    );

    let grid_w = width.div_ceil(TILE_W);
    let grid_h = height.div_ceil(TILE_H);
    assert!(
        grid_w <= 0xFFFF && grid_h <= 0xFFFF,
        "raster too large: grid {grid_w}×{grid_h} tiles exceeds 16-bit tile key range",
    );
    let n_tiles = (grid_w * grid_h) as usize;

    let mut records: Vec<TileRecord> = Vec::new();

    for seg in segs.chunks_exact(4) {
        let (mut sx0, mut sy0, mut sx1, mut sy1) = (
            seg[0] - x_min,
            seg[1] - y_min,
            seg[2] - x_min,
            seg[3] - y_min,
        );

        // Skip horizontal segments (no winding contribution).
        if (sy1 - sy0).abs() < 1e-6 {
            continue;
        }

        // Enforce sy0 ≤ sy1; record crossing direction as sign.
        let sign = if sy0 > sy1 {
            std::mem::swap(&mut sx0, &mut sx1);
            std::mem::swap(&mut sy0, &mut sy1);
            -1.0f32
        } else {
            1.0f32
        };

        let dxdy = (sx1 - sx0) / (sy1 - sy0);

        // Clamp to output bounds.
        let ey0 = sy0.max(0.0);
        let ey1 = sy1.min(height as f32);
        if ey0 >= ey1 {
            continue;
        }

        // First and last tile rows the segment (after clamping) crosses.
        // Subtract a small epsilon from ey1 so a segment ending exactly on a
        // tile boundary doesn't bleed into the next tile row.
        let ty0 = (ey0 / TILE_H as f32).floor() as u32;
        let ty1 = ((ey1 - 1e-6).max(0.0) / TILE_H as f32).floor() as u32;

        for ty in ty0..=ty1.min(grid_h - 1) {
            let tile_top = (ty * TILE_H) as f32;
            let tile_bot = tile_top + TILE_H as f32;

            // Segment y-extent clipped to this tile row, in tile-local coords.
            let seg_y0_tile = ey0.max(tile_top) - tile_top;
            let seg_y1_tile = ey1.min(tile_bot) - tile_top;
            if seg_y0_tile >= seg_y1_tile {
                continue;
            }

            // Global x where the segment enters this tile row (at the clipped ey0).
            let x_enter_global = dxdy.mul_add(ey0.max(tile_top) - sy0, sx0);
            // Global x at the exit of this tile row.
            let x_at_exit = dxdy.mul_add(seg_y1_tile - seg_y0_tile, x_enter_global);

            // Tile columns spanned by this segment within the tile row.
            let xl = x_enter_global.min(x_at_exit);
            let xr = x_enter_global.max(x_at_exit);

            // Compute tx0/tx1 as i32 first to handle negative x safely, then
            // clamp to [0, grid_w-1].  A negative tx1 means the segment is
            // entirely left of the raster — skip the whole tile row.
            let tx0_i = (xl / TILE_W as f32).floor() as i32;
            let tx1_i = (xr / TILE_W as f32).floor() as i32;
            if tx1_i < 0 {
                continue;
            }
            let tx0 = tx0_i.max(0) as u32;
            let tx1 = (tx1_i as u32).min(grid_w - 1);

            for tx in tx0..=tx1 {
                records.push(TileRecord {
                    key: (ty << 16) | tx,
                    // tile-local x: subtract this tile column's left edge.
                    x_enter: x_enter_global - (tx * TILE_W) as f32,
                    dxdy,
                    y0_tile: seg_y0_tile,
                    y1_tile: seg_y1_tile,
                    sign,
                    _pad: 0,
                    _pad2: 0,
                });
            }
        }
    }

    // Sort records by (tile_y, tile_x) key — CPU sort is faster than CUB radix
    // sort for typical PDF segment counts (O(100–1000) records).
    records.sort_unstable_by_key(|r| r.key);

    // Build exclusive prefix-sum index: tile_starts[i] = first record index for tile i.
    let mut tile_starts = vec![0u32; n_tiles];
    let mut tile_counts = vec![0u32; n_tiles];

    for rec in &records {
        let tile_idx = ((rec.key >> 16) * grid_w + (rec.key & 0xFFFF)) as usize;
        // tile_idx is always < n_tiles: key components are clamped to [0, grid_w/h-1]
        // at record-build time above.
        debug_assert!(tile_idx < n_tiles, "record key out of tile grid range");
        if tile_idx < n_tiles {
            tile_counts[tile_idx] += 1;
        }
    }

    let mut running = 0u32;
    for (start, count) in tile_starts.iter_mut().zip(tile_counts.iter()) {
        *start = running;
        running = running
            .checked_add(*count)
            .expect("tile record count overflows u32");
    }

    (records, tile_starts, tile_counts, grid_w)
}

struct GpuKernels {
    composite_rgba8: CudaFunction,
    apply_soft_mask: CudaFunction,
    aa_fill: CudaFunction,
    tile_fill: CudaFunction,
    icc_cmyk_matrix: CudaFunction,
    icc_cmyk_clut: CudaFunction,
}

/// An initialised CUDA context and compiled kernel set.
///
/// Create once per process with [`GpuCtx::init`] and share across threads via `Arc`.
pub struct GpuCtx {
    stream: Arc<CudaStream>,
    kernels: GpuKernels,
}

impl GpuCtx {
    /// Initialise CUDA device 0 and compile the embedded kernels.
    ///
    /// # Errors
    ///
    /// Returns an error if no CUDA device is present or kernel load fails.
    pub fn init() -> Result<Self, Box<dyn std::error::Error>> {
        let ctx: Arc<CudaContext> = CudaContext::new(0)?;
        let stream = ctx.default_stream();

        let load =
            |ptx_src: &str, name: &str| -> Result<CudaFunction, Box<dyn std::error::Error>> {
                let ptx = Ptx::from_src(ptx_src);
                let module: Arc<CudaModule> = ctx.load_module(ptx)?;
                Ok(module.load_function(name)?)
            };

        Ok(Self {
            stream,
            kernels: GpuKernels {
                composite_rgba8: load(PTX_COMPOSITE, "composite_rgba8")?,
                apply_soft_mask: load(PTX_SOFT_MASK, "apply_soft_mask")?,
                aa_fill: load(PTX_AA_FILL, "aa_fill")?,
                tile_fill: load(PTX_TILE_FILL, "tile_fill")?,
                icc_cmyk_matrix: load(PTX_ICC_CLUT, "icc_cmyk_matrix")?,
                icc_cmyk_clut: load(PTX_ICC_CLUT, "icc_cmyk_clut")?,
            },
        })
    }

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
        let n_pixels = width as usize * height as usize;

        if segs.is_empty() || n_pixels < GPU_AA_FILL_THRESHOLD {
            return Ok(aa_fill_cpu(segs, x_min, y_min, width, height, eo));
        }

        self.aa_fill_gpu(segs, x_min, y_min, width, height, eo)
    }

    /// Unconditional GPU dispatch for `aa_fill` (skips threshold check).
    pub(crate) fn aa_fill_gpu(
        &self,
        segs: &[f32],
        x_min: f32,
        y_min: f32,
        width: u32,
        height: u32,
        eo: bool,
    ) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let n_pixels = width as usize * height as usize;
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
        // PushKernelArg::arg returns &mut Self; chain results are intentionally unused.
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

    /// Tile-parallel analytical fill rasterisation using signed-area integration.
    ///
    /// This is the GPU equivalent of the CPU scanline scanner but uses analytical
    /// per-pixel coverage (vello-style trapezoid integrals) rather than sampling.
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
    #[allow(clippy::too_many_arguments)]
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
        let n_pixels = width as usize * height as usize;
        let stream = &self.stream;

        // Upload inputs.
        let d_records = if records.is_empty() {
            // cudarc refuses zero-size allocations — use a dummy 1-element buffer.
            stream.clone_htod(&[TileRecord::default()])?
        } else {
            stream.clone_htod(records)?
        };
        let d_tile_starts = stream.clone_htod(tile_starts)?;
        let d_tile_counts = stream.clone_htod(tile_counts)?;
        let d_cov_init = vec![0u8; n_pixels];
        let mut d_coverage = stream.clone_htod(&d_cov_init)?;

        let grid_h = height.div_ceil(TILE_H);
        let cfg = LaunchConfig {
            grid_dim: (grid_w, grid_h, 1),
            block_dim: (TILE_W, TILE_H, 1),
            shared_mem_bytes: 0,
        };

        let eo_int: i32 = i32::from(eo);
        let mut builder = stream.launch_builder(&self.kernels.tile_fill);
        let _ = builder.arg(&d_records);
        let _ = builder.arg(&d_tile_starts);
        let _ = builder.arg(&d_tile_counts);
        let _ = builder.arg(&grid_w);
        let _ = builder.arg(&width);
        let _ = builder.arg(&height);
        let _ = builder.arg(&eo_int);
        let _ = builder.arg(&mut d_coverage);
        // SAFETY: kernel arguments match the PTX signature exactly (8 args, no
        // x_min/y_min — coords are tile-local from build_tile_records).
        let _ = unsafe { builder.launch(cfg) }?;

        stream.synchronize()?;
        let mut coverage = vec![0u8; n_pixels];
        stream.memcpy_dtoh(&d_coverage, &mut coverage)?;
        Ok(coverage)
    }

    /// Convert CMYK pixels to RGB using a GPU kernel.
    ///
    /// `cmyk` is interleaved CMYK, 4 bytes per pixel (PDF convention: 0 = no ink,
    /// 255 = full ink).  Returns interleaved RGB, 3 bytes per pixel.
    ///
    /// Two dispatch paths:
    /// - `clut` is `None` — uses the fast matrix kernel (subtractive complement
    ///   formula, identical to the CPU fallback).
    /// - `clut` is `Some((table, grid_n))` — uses the 4D quadrilinear CLUT kernel.
    ///   `table` must be `grid_n^4 * 3` bytes, ordered
    ///   `(k * G³ + c * G² + m * G + y) * 3` (RGB output values, u8).
    ///   `grid_n` is typically 17 (83 521 nodes) or 33 (1 185 921 nodes).
    ///
    /// Falls back to [`icc_cmyk_to_rgb_cpu`] when `n_pixels < GPU_ICC_CLUT_THRESHOLD`
    /// or `cmyk` is empty.
    ///
    /// # Errors
    ///
    /// Returns an error if GPU data transfer or kernel launch fails.
    ///
    /// # Panics
    ///
    /// # Panics
    ///
    /// Panics if `cmyk.len()` is not a multiple of 4, or if `clut` is `Some` and
    /// `table.len() != grid_n^4 * 3`.
    #[must_use = "the RGB pixel buffer is not written to the caller unless used"]
    pub fn icc_cmyk_to_rgb(
        &self,
        cmyk: &[u8],
        clut: Option<(&[u8], u32)>,
    ) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        assert!(
            cmyk.len().is_multiple_of(4),
            "cmyk.len() must be a multiple of 4 (got {})",
            cmyk.len()
        );
        if let Some((table, grid_n)) = clut {
            // grid_n ≤ 255 is enforced by the baking API; checked_pow guards future misuse.
            let expected = (grid_n as usize)
                .checked_pow(4)
                .and_then(|n| n.checked_mul(3))
                .unwrap_or_else(|| {
                    panic!("grid_n({grid_n})^4*3 overflows usize — grid_n must be ≤ 255")
                });
            assert_eq!(
                table.len(),
                expected,
                "CLUT table length {got} ≠ grid_n({grid_n})^4*3={expected}",
                got = table.len(),
            );
        }

        let n = cmyk.len() / 4;
        if n == 0 {
            return Ok(Vec::new());
        }
        if n < GPU_ICC_CLUT_THRESHOLD {
            return Ok(icc_cmyk_to_rgb_cpu(cmyk, clut));
        }

        self.icc_cmyk_to_rgb_gpu(cmyk, clut)
    }

    fn icc_cmyk_to_rgb_gpu(
        &self,
        cmyk: &[u8],
        clut: Option<(&[u8], u32)>,
    ) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let n = cmyk.len() / 4;
        let n_u32 = u32::try_from(n).expect("pixel count exceeds u32::MAX");
        let stream = &self.stream;

        let d_cmyk = stream.clone_htod(cmyk)?;
        let rgb_init = vec![0u8; n * 3];
        let mut d_rgb = stream.clone_htod(&rgb_init)?;

        let cfg = launch_cfg(n);

        match clut {
            None => {
                let mut builder = stream.launch_builder(&self.kernels.icc_cmyk_matrix);
                let _ = builder.arg(&d_cmyk);
                let _ = builder.arg(&mut d_rgb);
                let _ = builder.arg(&n_u32);
                // SAFETY: 3 args match icc_cmyk_matrix PTX signature exactly.
                let _ = unsafe { builder.launch(cfg) }?;
            }
            Some((table, grid_n)) => {
                let d_clut = stream.clone_htod(table)?;
                let mut builder = stream.launch_builder(&self.kernels.icc_cmyk_clut);
                let _ = builder.arg(&d_cmyk);
                let _ = builder.arg(&mut d_rgb);
                let _ = builder.arg(&d_clut);
                let _ = builder.arg(&grid_n);
                let _ = builder.arg(&n_u32);
                // SAFETY: 5 args match icc_cmyk_clut PTX signature exactly.
                let _ = unsafe { builder.launch(cfg) }?;
            }
        }

        stream.synchronize()?;
        let mut rgb = vec![0u8; n * 3];
        stream.memcpy_dtoh(&d_rgb, &mut rgb)?;
        Ok(rgb)
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
        // PushKernelArg::arg returns &mut Self (builder pattern); chain results are intentionally unused.
        let _ = builder.arg(&d_src);
        let _ = builder.arg(&mut d_dst);
        let _ = builder.arg(&n_u32);
        // SAFETY: kernel arguments match the PTX signature; n_u32 bounds are verified above.
        // launch returns Option<timing events> on success; we don't need timing, so discard it.
        let _ = unsafe { builder.launch(cfg) }?;

        stream.synchronize()?;
        stream.memcpy_dtoh(&d_dst, dst)?;
        Ok(())
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
        // PushKernelArg::arg returns &mut Self (builder pattern); chain results are intentionally unused.
        let _ = builder.arg(&mut d_pixels);
        let _ = builder.arg(&d_mask);
        let _ = builder.arg(&n_u32);
        // SAFETY: kernel arguments match the PTX signature; n_u32 bounds are verified above.
        // launch returns Option<timing events> on success; we don't need timing, so discard it.
        let _ = unsafe { builder.launch(cfg) }?;

        stream.synchronize()?;
        stream.memcpy_dtoh(&d_pixels, pixels)?;
        Ok(())
    }
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "n.div_ceil(256) ≤ u32::MAX for any practical pixel count (≤ 4B pixels)"
)]
const fn launch_cfg(n: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (n.div_ceil(256) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// CPU fallback for `composite_rgba8`.
pub fn composite_rgba8_cpu(src: &[u8], dst: &mut [u8]) {
    for (s, d) in src.chunks_exact(4).zip(dst.chunks_exact_mut(4)) {
        let a_src = u32::from(s[3]);
        if a_src == 0 {
            continue;
        }
        if a_src == 255 {
            d.copy_from_slice(s);
            continue;
        }
        let a_dst = u32::from(d[3]);
        let inv = 255 - a_src;
        let a_out = a_src + (a_dst * inv + 127) / 255;
        if a_out == 0 {
            continue;
        }
        for c in 0..3 {
            let blended =
                (u32::from(s[c]) * a_src + u32::from(d[c]) * a_dst * inv / 255 + a_out / 2) / a_out;
            d[c] = blended.min(255) as u8;
        }
        // a_out = a_src + (a_dst * inv + 127) / 255 ≤ 255 + 255 = 510, so min(255) is needed.
        d[3] = a_out.min(255) as u8;
    }
}

/// CPU fallback for `apply_soft_mask`.
pub fn apply_soft_mask_cpu(pixels: &mut [u8], mask: &[u8]) {
    for (p, &m) in pixels.chunks_exact_mut(4).zip(mask) {
        let a = u32::from(p[3]);
        let m = u32::from(m);
        // a*m is at most 255*255 = 65025; +127 = 65152 < u32::MAX; /255 ≤ 255: safe cast.
        #[expect(
            clippy::cast_possible_truncation,
            reason = "result ≤ 255, always fits u8"
        )]
        let scaled = ((a * m + 127) / 255) as u8;
        p[3] = scaled;
    }
}

/// Scalar per-pixel CMYK→RGB: `R=(255−C)*(255−K)/255` (rounded), same for G/M, B/Y.
///
/// `src` is a 4-byte CMYK slice; `dst` is a 3-byte RGB slice.
fn cmyk_to_rgb_pixel_scalar(src: &[u8], dst: &mut [u8]) {
    debug_assert_eq!(src.len(), 4);
    debug_assert_eq!(dst.len(), 3);
    // u16 arithmetic: max product = 255*255 = 65025, fits without overflow.
    // (255*255 + 127)/255 = 255 exactly — the as u8 cast cannot truncate.
    let inv_k = u16::from(255 - src[3]);
    #[expect(
        clippy::cast_possible_truncation,
        reason = "value ≤ 255 by construction"
    )]
    {
        dst[0] = ((u16::from(255 - src[0]) * inv_k + 127) / 255) as u8;
        dst[1] = ((u16::from(255 - src[1]) * inv_k + 127) / 255) as u8;
        dst[2] = ((u16::from(255 - src[2]) * inv_k + 127) / 255) as u8;
    }
}

// ── AVX-512 CMYK→RGB ──────────────────────────────────────────────────────────
//
// Vectorised subtractive complement: R=(255−C)*(255−K)/255 (rounded).
// Processes 16 pixels per call using u16 arithmetic throughout.
//
// AoS→SoA: shuffle_epi8 gathers one channel from each 4-pixel 128-bit lane
// to bytes 0..3 of that lane (zeros elsewhere).  permute4x64(x, 0x88) selects
// epi64-lanes 0 and 2, giving [ch0..3 0000 ch4..7 0000] in 128 bits.  A final
// shuffle_epi8 compacts to [ch0..7 0×8]; unpacklo_epi64 joins two such halves
// (pixels 0..7 and 8..15) into 16 contiguous u8; cvtepu8_epi16 widens to u16.
//
// Division: exact ⌊(x+127)/255⌋ = (n + (n>>8) + 1) >> 8, n = x+127.
// Valid for n ∈ [0, 65152] (max n = 255²+127 = 65152 < 65280 = 255×256).

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw"
))]
/// Convert 16 CMYK pixels to RGB using AVX-512 u16 arithmetic.
///
/// `cmyk` must be exactly 64 bytes (16 pixels × 4 channels).
/// `rgb` must be at least 48 bytes (16 pixels × 3 channels).
///
/// # Safety
///
/// Caller must ensure `avx512f` and `avx512bw` are available.
/// `cmyk.len() == 64` and `rgb.len() >= 48` must hold.
#[expect(
    clippy::too_many_lines,
    reason = "SIMD shuffle/arithmetic pipeline — splitting would obscure the data flow"
)]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn cmyk_to_rgb_avx512(cmyk: &[u8; 64], rgb: &mut [u8]) {
    use std::arch::x86_64::{
        __m256i, __m512i, _mm_loadu_si128, _mm_shuffle_epi8, _mm_storeu_si128, _mm_unpacklo_epi64,
        _mm256_add_epi16, _mm256_castsi256_si128, _mm256_cvtepu8_epi16, _mm256_mullo_epi16,
        _mm256_packus_epi16, _mm256_permute4x64_epi64, _mm256_set1_epi16, _mm256_srli_epi16,
        _mm256_sub_epi16, _mm512_castsi512_si256, _mm512_extracti64x4_epi64, _mm512_loadu_si512,
        _mm512_shuffle_epi8,
    };

    debug_assert!(rgb.len() >= 48);

    unsafe {
        // Load all 16 CMYK pixels (64 bytes) into one 512-bit register.
        let raw: __m512i = _mm512_loadu_si512(cmyk.as_ptr().cast());

        // AoS→SoA via shuffle_epi8: permutes bytes within each 128-bit lane.
        // Each lane holds 4 CMYK pixels (16 bytes). The mask gathers one channel
        // to bytes 0..3 of the lane (zeros elsewhere), giving 4 lanes × 4 bytes =
        // 16 channel values spread across the 512-bit register.
        #[rustfmt::skip]
    let mask_c: [u8; 64] = [
        0, 4, 8,12, 0x80,0x80,0x80,0x80, 0x80,0x80,0x80,0x80, 0x80,0x80,0x80,0x80,
        0, 4, 8,12, 0x80,0x80,0x80,0x80, 0x80,0x80,0x80,0x80, 0x80,0x80,0x80,0x80,
        0, 4, 8,12, 0x80,0x80,0x80,0x80, 0x80,0x80,0x80,0x80, 0x80,0x80,0x80,0x80,
        0, 4, 8,12, 0x80,0x80,0x80,0x80, 0x80,0x80,0x80,0x80, 0x80,0x80,0x80,0x80,
    ];
        #[rustfmt::skip]
    let mask_m: [u8; 64] = [
        1, 5, 9,13, 0x80,0x80,0x80,0x80, 0x80,0x80,0x80,0x80, 0x80,0x80,0x80,0x80,
        1, 5, 9,13, 0x80,0x80,0x80,0x80, 0x80,0x80,0x80,0x80, 0x80,0x80,0x80,0x80,
        1, 5, 9,13, 0x80,0x80,0x80,0x80, 0x80,0x80,0x80,0x80, 0x80,0x80,0x80,0x80,
        1, 5, 9,13, 0x80,0x80,0x80,0x80, 0x80,0x80,0x80,0x80, 0x80,0x80,0x80,0x80,
    ];
        #[rustfmt::skip]
    let mask_y: [u8; 64] = [
        2, 6,10,14, 0x80,0x80,0x80,0x80, 0x80,0x80,0x80,0x80, 0x80,0x80,0x80,0x80,
        2, 6,10,14, 0x80,0x80,0x80,0x80, 0x80,0x80,0x80,0x80, 0x80,0x80,0x80,0x80,
        2, 6,10,14, 0x80,0x80,0x80,0x80, 0x80,0x80,0x80,0x80, 0x80,0x80,0x80,0x80,
        2, 6,10,14, 0x80,0x80,0x80,0x80, 0x80,0x80,0x80,0x80, 0x80,0x80,0x80,0x80,
    ];
        #[rustfmt::skip]
    let mask_k: [u8; 64] = [
        3, 7,11,15, 0x80,0x80,0x80,0x80, 0x80,0x80,0x80,0x80, 0x80,0x80,0x80,0x80,
        3, 7,11,15, 0x80,0x80,0x80,0x80, 0x80,0x80,0x80,0x80, 0x80,0x80,0x80,0x80,
        3, 7,11,15, 0x80,0x80,0x80,0x80, 0x80,0x80,0x80,0x80, 0x80,0x80,0x80,0x80,
        3, 7,11,15, 0x80,0x80,0x80,0x80, 0x80,0x80,0x80,0x80, 0x80,0x80,0x80,0x80,
    ];
        let shuf_c: __m512i = _mm512_loadu_si512(mask_c.as_ptr().cast());
        let shuf_m: __m512i = _mm512_loadu_si512(mask_m.as_ptr().cast());
        let shuf_y: __m512i = _mm512_loadu_si512(mask_y.as_ptr().cast());
        let shuf_k: __m512i = _mm512_loadu_si512(mask_k.as_ptr().cast());

        let c_bytes: __m512i = _mm512_shuffle_epi8(raw, shuf_c);
        let m_bytes: __m512i = _mm512_shuffle_epi8(raw, shuf_m);
        let y_bytes: __m512i = _mm512_shuffle_epi8(raw, shuf_y);
        let k_bytes: __m512i = _mm512_shuffle_epi8(raw, shuf_k);

        // After shuffle: each 128-bit lane has [ch0 ch1 ch2 ch3  0×12].
        // permute4x64(x, 0x88) selects epi64-lanes 0 and 2:
        //   low 128 = [ch0..3  0×4  ch4..7  0×4]  (two 4-byte groups, gaps between)
        // compact2 shuffle closes the gaps: bytes 0..3 and 8..11 → bytes 0..7.
        // unpacklo_epi64 joins the lo and hi halves → 16 contiguous u8.
        // cvtepu8_epi16 zero-extends to 16 u16.
        #[rustfmt::skip]
        let compact2: [u8; 16] = [
            0,1,2,3, 8,9,10,11, 0x80,0x80,0x80,0x80, 0x80,0x80,0x80,0x80,
        ];
        let compact2_128 = _mm_loadu_si128(compact2.as_ptr().cast());

        macro_rules! compact_to_u16 {
            ($v512:expr) => {{
                let lo128 = _mm_shuffle_epi8(
                    _mm256_castsi256_si128(_mm256_permute4x64_epi64(
                        _mm512_castsi512_si256($v512),
                        0x88_i32,
                    )),
                    compact2_128,
                );
                let hi128 = _mm_shuffle_epi8(
                    _mm256_castsi256_si128(_mm256_permute4x64_epi64(
                        _mm512_extracti64x4_epi64($v512, 1),
                        0x88_i32,
                    )),
                    compact2_128,
                );
                _mm256_cvtepu8_epi16(_mm_unpacklo_epi64(lo128, hi128))
            }};
        }

        let c_u16: __m256i = compact_to_u16!(c_bytes);
        let m_u16: __m256i = compact_to_u16!(m_bytes);
        let y_u16: __m256i = compact_to_u16!(y_bytes);
        let k_u16: __m256i = compact_to_u16!(k_bytes);

        // inv_ch = 255 − ch  (u16, max 255, no overflow)
        let v255 = _mm256_set1_epi16(255_i16);
        let inv_c = _mm256_sub_epi16(v255, c_u16);
        let inv_m = _mm256_sub_epi16(v255, m_u16);
        let inv_y = _mm256_sub_epi16(v255, y_u16);
        let inv_k = _mm256_sub_epi16(v255, k_u16);

        // prod = inv_ch * inv_k  (u16, max 255*255 = 65025 < 65536, no truncation)
        let prod_r = _mm256_mullo_epi16(inv_c, inv_k);
        let prod_g = _mm256_mullo_epi16(inv_m, inv_k);
        let prod_b = _mm256_mullo_epi16(inv_y, inv_k);

        // Exact ⌊(x + 127) / 255⌋ matching the scalar formula (255-c)*(255-k)+127)/255.
        // For n = x + 127 ∈ [0, 65152]: ⌊n/255⌋ = (n + (n>>8) + 1) >> 8.
        // Proof: let n = 255*q + r, 0 ≤ r < 255.
        //   n>>8 = ⌊n/256⌋ = q - ⌊(256q-n)/256⌋ = q - ⌊(256r - r·1)/256⌋ ≈ q (error < 1).
        //   More precisely (n + (n>>8) + 1)>>8 = ⌊(n + ⌊n/256⌋ + 1)/256⌋.
        //   For n < 65280 (= 255*256) this equals q = ⌊n/255⌋ exactly.
        //   max n = 65152 < 65280 ✓.
        let v127 = _mm256_set1_epi16(127_i16);
        let v1 = _mm256_set1_epi16(1_i16);
        macro_rules! div255 {
            ($x:expr) => {{
                let n = _mm256_add_epi16($x, v127); // n = x + 127
                _mm256_srli_epi16(
                    _mm256_add_epi16(_mm256_add_epi16(n, _mm256_srli_epi16(n, 8)), v1),
                    8,
                )
            }};
        }
        let r_u16 = div255!(prod_r);
        let g_u16 = div255!(prod_g);
        let b_u16 = div255!(prod_b);

        // Narrow u16 → u8 (values ≤ 255; saturation never triggers).
        // packus_epi16(v, v) → [v0..v7 v0..v7 | v8..v15 v8..v15] per 256-bit.
        // permute4x64(x, 0x88) packs the two unique halves to the low 128 bits.
        let r8 = _mm256_castsi256_si128(_mm256_permute4x64_epi64(
            _mm256_packus_epi16(r_u16, r_u16),
            0x88_i32,
        ));
        let g8 = _mm256_castsi256_si128(_mm256_permute4x64_epi64(
            _mm256_packus_epi16(g_u16, g_u16),
            0x88_i32,
        ));
        let b8 = _mm256_castsi256_si128(_mm256_permute4x64_epi64(
            _mm256_packus_epi16(b_u16, b_u16),
            0x88_i32,
        ));

        // Scatter R, G, B to interleaved RGB output via three scalar stores.
        // The vectorised multiply+divide above provides the bulk of the speedup;
        // this store loop is cheap (16 iterations, compiler unrolls it).
        let mut r_arr = [0u8; 16];
        let mut g_arr = [0u8; 16];
        let mut b_arr = [0u8; 16];
        _mm_storeu_si128(r_arr.as_mut_ptr().cast(), r8);
        _mm_storeu_si128(g_arr.as_mut_ptr().cast(), g8);
        _mm_storeu_si128(b_arr.as_mut_ptr().cast(), b8);
        for i in 0..16 {
            rgb[i * 3] = r_arr[i];
            rgb[i * 3 + 1] = g_arr[i];
            rgb[i * 3 + 2] = b_arr[i];
        }
    }
}

/// CPU fallback for [`GpuCtx::icc_cmyk_to_rgb`].
///
/// When `clut` is `None`, applies the subtractive complement formula:
///   `R = (255−C)*(255−K)/255` (rounded), same for G/M and B/Y.
///
/// When `clut` is `Some((table, grid_n))`, evaluates the 4D CLUT using
/// quadrilinear interpolation — the same algorithm as the GPU kernel.
///
/// The `clut = None` path uses AVX-512 (avx512f + avx512bw) when available,
/// processing 16 pixels per iteration.  Falls back to scalar per-pixel loop.
#[must_use]
#[expect(
    clippy::too_many_lines,
    reason = "CLUT quadrilinear interpolation + AVX dispatch — cohesion outweighs length"
)]
#[expect(
    clippy::missing_panics_doc,
    reason = "the expect() guards an internal invariant (chunk size) that cannot fire"
)]
pub fn icc_cmyk_to_rgb_cpu(cmyk: &[u8], clut: Option<(&[u8], u32)>) -> Vec<u8> {
    let n = cmyk.len() / 4;
    let mut rgb = vec![0u8; n * 3];

    match clut {
        None => {
            #[cfg(all(
                target_arch = "x86_64",
                target_feature = "avx512f",
                target_feature = "avx512bw"
            ))]
            {
                // AVX-512 path: 16 pixels per iteration.
                let mut chunks = cmyk.chunks_exact(64);
                let mut out_off = 0usize;
                for chunk in chunks.by_ref() {
                    // SAFETY: avx512f+avx512bw confirmed by target_feature (compile-time on
                    // native builds; requires -C target-cpu=native or explicit target-feature).
                    // chunk is exactly 64 bytes; rgb[out_off..] has ≥ 48 bytes remaining.
                    unsafe {
                        cmyk_to_rgb_avx512(
                            chunk.try_into().expect("chunk is exactly 64 bytes"),
                            &mut rgb[out_off..],
                        );
                    }
                    out_off += 48;
                }
                // Scalar tail for remaining pixels (< 16).
                for (src, dst) in chunks
                    .remainder()
                    .chunks_exact(4)
                    .zip(rgb[out_off..].chunks_exact_mut(3))
                {
                    cmyk_to_rgb_pixel_scalar(src, dst);
                }
            }
            #[cfg(not(all(
                target_arch = "x86_64",
                target_feature = "avx512f",
                target_feature = "avx512bw"
            )))]
            {
                for (src, dst) in cmyk.chunks_exact(4).zip(rgb.chunks_exact_mut(3)) {
                    cmyk_to_rgb_pixel_scalar(src, dst);
                }
            }
        }
        Some((table, grid_n)) => {
            let g = grid_n as usize; // grid_n ≤ 255 from caller validation
            let g2 = g * g;
            let g3 = g2 * g;
            // grid_n ≤ 255 → (grid_n - 1) ≤ 254, exact in f32 (needs ≤ 8 mantissa bits).
            #[expect(
                clippy::cast_precision_loss,
                reason = "grid_n ≤ 255, fits exactly in f32 (8 bits < 23-bit mantissa)"
            )]
            let g1 = (grid_n - 1) as f32;
            let scale = g1 / 255.0;
            for (src, dst) in cmyk.chunks_exact(4).zip(rgb.chunks_exact_mut(3)) {
                let fc = f32::from(src[0]) * scale;
                let fm = f32::from(src[1]) * scale;
                let fy = f32::from(src[2]) * scale;
                let fk = f32::from(src[3]) * scale;

                // fc ∈ [0.0, g1] ⊂ [0.0, 254.0]; floor is non-negative and ≤ 254.
                // The sign-loss and truncation lints fire because `as usize` is UB
                // for negative or >usize::MAX floats; here neither can happen.
                #[expect(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "fc/fm/fy/fk are in [0.0, 254.0]; floor is exact and non-negative"
                )]
                let (ic0, im0, iy0, ik0) = (
                    (fc as usize).min(g - 1),
                    (fm as usize).min(g - 1),
                    (fy as usize).min(g - 1),
                    (fk as usize).min(g - 1),
                );
                let ic1 = (ic0 + 1).min(g - 1);
                let im1 = (im0 + 1).min(g - 1);
                let iy1 = (iy0 + 1).min(g - 1);
                let ik1 = (ik0 + 1).min(g - 1);

                // Fractional weights: difference between float position and floored index.
                // ic0 ≤ 254 → ic0 as f32 is exact (fits in 8 mantissa bits).
                #[expect(
                    clippy::cast_precision_loss,
                    reason = "ic0/im0/iy0/ik0 ≤ 254, exact in f32"
                )]
                let (wc, wm, wy, wk) = (
                    fc - ic0 as f32,
                    fm - im0 as f32,
                    fy - iy0 as f32,
                    fk - ik0 as f32,
                );

                let node = |ci: usize, mi: usize, yi: usize, ki: usize| -> [f32; 3] {
                    let idx = (ki * g3 + ci * g2 + mi * g + yi) * 3;
                    [
                        f32::from(table[idx]),
                        f32::from(table[idx + 1]),
                        f32::from(table[idx + 2]),
                    ]
                };

                let lerp = |a: f32, b: f32, t: f32| t.mul_add(b - a, a);
                let lerp3 = |a: [f32; 3], b: [f32; 3], t: f32| -> [f32; 3] {
                    [
                        lerp(a[0], b[0], t),
                        lerp(a[1], b[1], t),
                        lerp(a[2], b[2], t),
                    ]
                };

                // K=0 face: lerp Y → M → C
                let c0m0k0 = lerp3(node(ic0, im0, iy0, ik0), node(ic0, im0, iy1, ik0), wy);
                let c0m1k0 = lerp3(node(ic0, im1, iy0, ik0), node(ic0, im1, iy1, ik0), wy);
                let c1m0k0 = lerp3(node(ic1, im0, iy0, ik0), node(ic1, im0, iy1, ik0), wy);
                let c1m1k0 = lerp3(node(ic1, im1, iy0, ik0), node(ic1, im1, iy1, ik0), wy);
                let rk0 = lerp3(lerp3(c0m0k0, c0m1k0, wm), lerp3(c1m0k0, c1m1k0, wm), wc);

                // K=1 face: lerp Y → M → C
                let c0m0k1 = lerp3(node(ic0, im0, iy0, ik1), node(ic0, im0, iy1, ik1), wy);
                let c0m1k1 = lerp3(node(ic0, im1, iy0, ik1), node(ic0, im1, iy1, ik1), wy);
                let c1m0k1 = lerp3(node(ic1, im0, iy0, ik1), node(ic1, im0, iy1, ik1), wy);
                let c1m1k1 = lerp3(node(ic1, im1, iy0, ik1), node(ic1, im1, iy1, ik1), wy);
                let rk1 = lerp3(lerp3(c0m0k1, c0m1k1, wm), lerp3(c1m0k1, c1m1k1, wm), wc);

                let out = lerp3(rk0, rk1, wk);
                #[expect(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "clamped to [0,255] before cast"
                )]
                {
                    dst[0] = out[0].clamp(0.0, 255.0).round() as u8;
                    dst[1] = out[1].clamp(0.0, 255.0).round() as u8;
                    dst[2] = out[2].clamp(0.0, 255.0).round() as u8;
                }
            }
        }
    }

    rgb
}

/// Halton(2) jitter X offsets within [0,1) for 64-sample AA.
const HALTON2: [f32; 64] = [
    0.5,
    0.25,
    0.75,
    0.125,
    0.625,
    0.375,
    0.875,
    0.062_5,
    0.562_5,
    0.312_5,
    0.812_5,
    0.187_5,
    0.687_5,
    0.437_5,
    0.937_5,
    0.031_25,
    0.531_25,
    0.281_25,
    0.781_25,
    0.156_25,
    0.656_25,
    0.406_25,
    0.906_25,
    0.093_75,
    0.593_75,
    0.343_75,
    0.843_75,
    0.218_75,
    0.718_75,
    0.468_75,
    0.968_75,
    0.015_625,
    0.515_625,
    0.265_625,
    0.765_625,
    0.140_625,
    0.640_625,
    0.390_625,
    0.890_625,
    0.078_125,
    0.578_125,
    0.328_125,
    0.828_125,
    0.203_125,
    0.703_125,
    0.453_125,
    0.953_125,
    0.046_875,
    0.546_875,
    0.296_875,
    0.796_875,
    0.171_875,
    0.671_875,
    0.421_875,
    0.921_875,
    0.109_375,
    0.609_375,
    0.359_375,
    0.859_375,
    0.234_375,
    0.734_375,
    0.484_375,
    0.984_375,
    0.007_812_5,
];

/// Halton(3) jitter Y offsets within [0,1) for 64-sample AA.
const HALTON3: [f32; 64] = [
    0.333_333, 0.666_667, 0.111_111, 0.444_444, 0.777_778, 0.222_222, 0.555_556, 0.888_889,
    0.037_037, 0.370_370, 0.703_704, 0.148_148, 0.481_481, 0.814_815, 0.259_259, 0.592_593,
    0.925_926, 0.012_346, 0.345_679, 0.679_012, 0.123_457, 0.456_790, 0.790_123, 0.234_568,
    0.567_901, 0.901_235, 0.049_383, 0.382_716, 0.716_049, 0.160_494, 0.493_827, 0.827_160,
    0.271_605, 0.604_938, 0.938_272, 0.004_115, 0.337_449, 0.670_782, 0.115_226, 0.448_560,
    0.781_893, 0.226_337, 0.559_671, 0.893_004, 0.041_152, 0.374_486, 0.707_819, 0.152_263,
    0.485_597, 0.818_930, 0.263_374, 0.596_708, 0.930_041, 0.074_074, 0.407_407, 0.740_741,
    0.185_185, 0.518_519, 0.851_852, 0.296_296, 0.629_630, 0.962_963, 0.008_230, 0.341_564,
];

/// CPU fallback for `aa_fill` using 64-sample Halton jitter per pixel.
///
/// Matches the GPU kernel's coverage computation exactly: same Halton(2,3)
/// sample offsets, same winding-number / even-odd logic, same scale formula.
/// Used when `n_pixels < GPU_AA_FILL_THRESHOLD` or when no CUDA device is present.
#[must_use]
pub fn aa_fill_cpu(
    segs: &[f32],
    x_min: f32,
    y_min: f32,
    width: u32,
    height: u32,
    eo: bool,
) -> Vec<u8> {
    let n_pixels = width as usize * height as usize;
    let mut out = vec![0u8; n_pixels];

    for py in 0..height {
        for px in 0..width {
            #[expect(
                clippy::cast_precision_loss,
                reason = "px/py ≤ width/height ≤ u32::MAX; at typical DPIs (≤ 32768 px) \
                          the f32 precision loss is sub-pixel and irrelevant for AA coverage"
            )]
            let (cx, cy) = (x_min + px as f32 + 0.5, y_min + py as f32 + 0.5);
            let mut hits = 0u32;
            for s in 0..64usize {
                let sx = cx + HALTON2[s] - 0.5;
                let sy = cy + HALTON3[s] - 0.5;
                if aa_fill_cpu_sample(segs, sx, sy, eo) {
                    hits += 1;
                }
            }
            #[expect(
                clippy::cast_possible_truncation,
                reason = "hits ≤ 64; (64*255+32)>>6 = 255 — always fits u8"
            )]
            {
                out[py as usize * width as usize + px as usize] = ((hits * 255 + 32) >> 6) as u8;
            }
        }
    }
    out
}

fn aa_fill_cpu_sample(segs: &[f32], sx: f32, sy: f32, eo: bool) -> bool {
    let mut winding = 0i32;
    for seg in segs.chunks_exact(4) {
        let (x0, y0, x1, y1) = (seg[0], seg[1], seg[2], seg[3]);
        if y0 <= sy && sy < y1 {
            let t = (sy - y0) / (y1 - y0);
            let xi = t.mul_add(x1 - x0, x0);
            if xi >= sx {
                winding += 1;
            }
        } else if y1 <= sy && sy < y0 {
            let t = (sy - y1) / (y0 - y1);
            let xi = t.mul_add(x0 - x1, x1);
            if xi >= sx {
                winding -= 1;
            }
        }
    }
    if eo { (winding & 1) != 0 } else { winding != 0 }
}

#[cfg(test)]
mod tests {
    use super::{apply_soft_mask_cpu, composite_rgba8_cpu, icc_cmyk_to_rgb_cpu};

    #[test]
    fn composite_cpu_opaque_src() {
        let src = [200u8, 100, 50, 255];
        let mut dst = [10u8, 20, 30, 128];
        composite_rgba8_cpu(&src, &mut dst);
        assert_eq!(dst, [200, 100, 50, 255]);
    }

    #[test]
    fn composite_cpu_transparent_src() {
        let src = [200u8, 100, 50, 0];
        let mut dst = [10u8, 20, 30, 128];
        let expected = dst;
        composite_rgba8_cpu(&src, &mut dst);
        assert_eq!(dst, expected);
    }

    #[test]
    fn composite_cpu_half_alpha() {
        let src = [255u8, 255, 255, 128];
        let mut dst = [0u8, 0, 0, 255];
        composite_rgba8_cpu(&src, &mut dst);
        assert!(dst[0] >= 126 && dst[0] <= 130, "r={}", dst[0]);
        assert!(dst[1] >= 126 && dst[1] <= 130, "g={}", dst[1]);
        assert!(dst[2] >= 126 && dst[2] <= 130, "b={}", dst[2]);
        assert_eq!(dst[3], 255);
    }

    #[test]
    fn soft_mask_cpu_full() {
        let mut pixels = [100u8, 150, 200, 240];
        let mask = [255u8];
        apply_soft_mask_cpu(&mut pixels, &mask);
        assert_eq!(pixels[3], 240);
    }

    #[test]
    fn soft_mask_cpu_half() {
        let mut pixels = [100u8, 150, 200, 200];
        let mask = [128u8];
        apply_soft_mask_cpu(&mut pixels, &mask);
        assert_eq!(pixels[3], 100);
    }

    #[test]
    fn soft_mask_cpu_zero() {
        let mut pixels = [100u8, 150, 200, 255, 10, 20, 30, 128];
        let mask = [0u8, 0];
        apply_soft_mask_cpu(&mut pixels, &mask);
        assert_eq!(pixels[3], 0);
        assert_eq!(pixels[7], 0);
    }

    #[test]
    fn icc_cmyk_matrix_white() {
        let cmyk = [0u8, 0, 0, 0];
        let rgb = icc_cmyk_to_rgb_cpu(&cmyk, None);
        assert_eq!(rgb, [255, 255, 255]);
    }

    #[test]
    fn icc_cmyk_matrix_black() {
        let cmyk = [0u8, 0, 0, 255];
        let rgb = icc_cmyk_to_rgb_cpu(&cmyk, None);
        assert_eq!(rgb, [0, 0, 0]);
    }

    #[test]
    fn icc_cmyk_matrix_pure_cyan() {
        // C=255, M=Y=K=0 → R=0, G=255, B=255
        let cmyk = [255u8, 0, 0, 0];
        let rgb = icc_cmyk_to_rgb_cpu(&cmyk, None);
        assert_eq!(rgb, [0, 255, 255]);
    }

    #[test]
    fn icc_cmyk_matrix_multi_pixel() {
        let cmyk = [0u8, 0, 0, 0, 0, 0, 0, 255];
        let rgb = icc_cmyk_to_rgb_cpu(&cmyk, None);
        assert_eq!(&rgb[0..3], &[255, 255, 255]);
        assert_eq!(&rgb[3..6], &[0, 0, 0]);
    }

    /// Parity: AVX-512 path must match `cmyk_to_rgb_pixel_scalar` byte-for-byte.
    ///
    /// Covers axis extremes (all-0, all-255, pure K, pure C) and a mid-range
    /// sweep.  Requires `-C target-cpu=native` so the avx512f+avx512bw cfg
    /// gates activate at compile time — on non-AVX machines both paths go
    /// scalar and the test degenerates to a no-op tautology (still passes).
    #[test]
    fn icc_cmyk_matrix_avx_vs_scalar() {
        #[rustfmt::skip]
        let cmyk: Vec<u8> = vec![
            // white, black, cyan, magenta
              0,   0,   0,   0,
              0,   0,   0, 255,
            255,   0,   0,   0,
              0, 255,   0,   0,
            // yellow, key-only mid, all-max, all-mid
              0,   0, 255,   0,
              0,   0,   0, 128,
            255, 255, 255, 255,
            128, 128, 128, 128,
            // mid-range sweep
             64,  32,  16,   8,
            200, 100,  50,  25,
             10,  20,  30,  40,
             50,  60,  70,  80,
             90, 100, 110, 120,
            130, 140, 150, 160,
            170, 180, 190, 200,
            210, 220, 230, 240,
        ];
        assert_eq!(cmyk.len(), 64, "test vector must be exactly 16 pixels");

        // Reference via the extracted scalar helper — not an inline reimplementation.
        let mut scalar_rgb = vec![0u8; 48];
        for (src, dst) in cmyk.chunks_exact(4).zip(scalar_rgb.chunks_exact_mut(3)) {
            super::cmyk_to_rgb_pixel_scalar(src, dst);
        }

        let avx_rgb = icc_cmyk_to_rgb_cpu(&cmyk, None);

        for (i, (s, a)) in scalar_rgb.iter().zip(avx_rgb.iter()).enumerate() {
            assert_eq!(
                s,
                a,
                "RGB byte {i} (pixel {}, channel {}): scalar={s} avx={a}",
                i / 3,
                i % 3,
            );
        }
    }

    #[test]
    fn icc_cmyk_clut_identity_corners() {
        // 2^4 = 16-node CLUT where output = matrix formula at corners.
        let g: usize = 2;
        let mut table = vec![0u8; g * g * g * g * 3];
        for ki in 0..g {
            for ci in 0..g {
                for mi in 0..g {
                    for yi in 0..g {
                        let idx = (ki * g * g * g + ci * g * g + mi * g + yi) * 3;
                        let c = (ci * 255) as u8;
                        let m = (mi * 255) as u8;
                        let y = (yi * 255) as u8;
                        let k = (ki * 255) as u8;
                        let inv_k = u32::from(255 - k);
                        table[idx] = ((u32::from(255 - c) * inv_k) / 255) as u8;
                        table[idx + 1] = ((u32::from(255 - m) * inv_k) / 255) as u8;
                        table[idx + 2] = ((u32::from(255 - y) * inv_k) / 255) as u8;
                    }
                }
            }
        }
        let rgb = icc_cmyk_to_rgb_cpu(&[0u8, 0, 0, 0], Some((&table, 2)));
        assert_eq!(rgb, [255, 255, 255], "white corner");
        let rgb = icc_cmyk_to_rgb_cpu(&[0u8, 0, 0, 255], Some((&table, 2)));
        assert_eq!(rgb, [0, 0, 0], "black corner");
    }

    #[test]
    fn aa_fill_cpu_solid_rect_full_coverage() {
        let segs: Vec<f32> = vec![
            -100.0, -100.0, 100.0, -100.0, 100.0, -100.0, 100.0, 100.0, 100.0, 100.0, -100.0,
            100.0, -100.0, 100.0, -100.0, -100.0,
        ];
        let cov = super::aa_fill_cpu(&segs, 0.0, 0.0, 1, 1, false);
        assert_eq!(cov.len(), 1);
        assert_eq!(cov[0], 255, "fully covered pixel should be 255");
    }

    #[test]
    fn aa_fill_cpu_outside_rect_zero_coverage() {
        let segs: Vec<f32> = vec![
            0.0, 0.0, 10.0, 0.0, 10.0, 0.0, 10.0, 10.0, 10.0, 10.0, 0.0, 10.0, 0.0, 10.0, 0.0, 0.0,
        ];
        let cov = super::aa_fill_cpu(&segs, 200.0, 200.0, 1, 1, false);
        assert_eq!(cov[0], 0, "pixel outside rect should be 0");
    }

    #[test]
    fn aa_fill_cpu_empty_segs_zero_coverage() {
        let cov = super::aa_fill_cpu(&[], 0.0, 0.0, 4, 4, false);
        assert_eq!(cov.len(), 16);
        assert!(
            cov.iter().all(|&v| v == 0),
            "empty segs → all zero coverage"
        );
    }

    #[test]
    fn aa_fill_cpu_eo_donut_inner_zero() {
        let outer: [f32; 16] = [
            -10.0, -10.0, 10.0, -10.0, 10.0, -10.0, 10.0, 10.0, 10.0, 10.0, -10.0, 10.0, -10.0,
            10.0, -10.0, -10.0,
        ];
        let inner: [f32; 16] = [
            -5.0, -5.0, 5.0, -5.0, 5.0, -5.0, 5.0, 5.0, 5.0, 5.0, -5.0, 5.0, -5.0, 5.0, -5.0, -5.0,
        ];
        let segs: Vec<f32> = outer.iter().chain(inner.iter()).copied().collect();
        let cov = super::aa_fill_cpu(&segs, -0.5, -0.5, 1, 1, true);
        assert_eq!(cov[0], 0, "EO donut centre should be 0");
    }

    #[test]
    fn tile_records_empty_segs() {
        let (recs, starts, counts, grid_w) = super::build_tile_records(&[], 0.0, 0.0, 32, 32);
        assert!(recs.is_empty());
        assert_eq!(grid_w, 2);
        assert!(starts.iter().all(|&s| s == 0));
        assert!(counts.iter().all(|&c| c == 0));
    }

    #[test]
    fn tile_records_single_vertical_segment() {
        // Segment from (8,0) to (8,17): crosses tile rows 0 (y 0..16) and 1 (y 16..17).
        let segs = [8.0f32, 0.0, 8.0, 17.0];
        let (recs, _, _, _) = super::build_tile_records(&segs, 0.0, 0.0, 64, 64);
        assert_eq!(recs.len(), 2, "one record per tile row crossed");
    }

    #[test]
    fn tile_records_diagonal_segment() {
        let segs = [0.0f32, 0.0, 32.0, 32.0];
        let (recs, _, _, _) = super::build_tile_records(&segs, 0.0, 0.0, 128, 128);
        assert!(recs.len() >= 2, "diagonal must produce at least 2 records");
    }

    #[test]
    fn tile_records_sort_order() {
        let segs = [24.0f32, 0.0, 24.0, 8.0, 8.0f32, 16.0, 8.0, 24.0];
        let (recs, starts, counts, grid_w) = super::build_tile_records(&segs, 0.0, 0.0, 80, 80);
        assert_eq!(grid_w, 5);
        for w in recs.windows(2) {
            assert!(w[0].key <= w[1].key, "records must be sorted by key");
        }
        let tile_01 = 0 * grid_w as usize + 1;
        let tile_10 = 1 * grid_w as usize + 0;
        assert_eq!(counts[tile_01], 1);
        assert_eq!(counts[tile_10], 1);
        let _ = starts;
    }

    #[test]
    fn tile_records_prefix_sum_consistent() {
        let segs = [4.0f32, 0.0, 60.0, 63.0];
        let (recs, starts, counts, grid_w) = super::build_tile_records(&segs, 0.0, 0.0, 64, 64);
        let _ = grid_w;
        let total: u32 = counts.iter().sum();
        assert_eq!(total as usize, recs.len(), "sum of counts == total records");
        for i in 0..counts.len() - 1 {
            assert_eq!(
                starts[i] + counts[i],
                starts[i + 1],
                "prefix sum broken at {i}"
            );
        }
    }

    // ── GPU vs CPU AA fill parity ─────────────────────────────────────────────
    //
    // Assert byte-identical coverage output between aa_fill_cpu and the CUDA
    // aa_fill kernel.  Requires a CUDA device; gated on `gpu-validation` so
    // CI without a GPU skips them automatically.
    //
    //   cargo test -p gpu --lib --features gpu-validation -- gpu_vs_cpu

    #[cfg(feature = "gpu-validation")]
    mod gpu_vs_cpu {
        use std::sync::OnceLock;

        use super::super::{GpuCtx, aa_fill_cpu};

        // Initialise GpuCtx once for the whole test module.  Each test calls
        // `gpu()` rather than `GpuCtx::init()` directly so that:
        //   (a) the CUDA context is created only once, not once per test, and
        //   (b) a missing GPU produces a single clear message, not six separate
        //       panics with redundant stack traces.
        static GPU: OnceLock<GpuCtx> = OnceLock::new();

        fn gpu() -> &'static GpuCtx {
            GPU.get_or_init(|| {
                GpuCtx::init().expect(
                    "gpu-validation tests require a CUDA device — \
                     run with a machine that has an NVIDIA GPU",
                )
            })
        }

        // Closed CCW rectangle as four directed segments: top, right, bottom, left.
        // rustfmt::skip keeps the 4-floats-per-segment grouping readable.
        #[rustfmt::skip]
        fn rect_segs(x0: f32, y0: f32, x1: f32, y1: f32) -> Vec<f32> {
            vec![
                x0, y0,  x1, y0,   // top edge    (left → right)
                x1, y0,  x1, y1,   // right edge  (top  → bottom)
                x1, y1,  x0, y1,   // bottom edge (right → left)
                x0, y1,  x0, y0,   // left edge   (bottom → top)
            ]
        }

        // Run both CPU and GPU paths on `segs` over a 1×1 output at (`xmin`,`ymin`).
        // Returns `(cpu_byte, gpu_byte)`.  Panics loudly if the GPU call fails.
        fn both_1x1(segs: &[f32], x_min: f32, y_min: f32, eo: bool) -> (u8, u8) {
            let cpu = aa_fill_cpu(segs, x_min, y_min, 1, 1, eo);
            // aa_fill_gpu bypasses the dispatch threshold so we always hit the kernel,
            // even for single-pixel regions that would normally fall back to CPU.
            let gpu_cov = gpu()
                .aa_fill_gpu(segs, x_min, y_min, 1, 1, eo)
                .unwrap_or_else(|e| panic!("GPU aa_fill failed: {e}"));
            (cpu[0], gpu_cov[0])
        }

        #[test]
        fn fully_covered() {
            // Large rect encloses the 1×1 query window entirely.
            let segs = rect_segs(-100.0, -100.0, 100.0, 100.0);
            let (cpu, gpu) = both_1x1(&segs, 0.0, 0.0, false);
            assert_eq!(cpu, gpu, "CPU={cpu} GPU={gpu}");
            assert_eq!(cpu, 255, "fully enclosed pixel must be 255");
        }

        #[test]
        fn fully_outside() {
            // Rect at origin; query window is 200 px away.
            let segs = rect_segs(0.0, 0.0, 10.0, 10.0);
            let (cpu, gpu) = both_1x1(&segs, 200.0, 200.0, false);
            assert_eq!(cpu, gpu, "CPU={cpu} GPU={gpu}");
            assert_eq!(cpu, 0, "pixel outside path must be 0");
        }

        #[test]
        fn eo_donut_centre() {
            // EO donut: outer 20×20, inner 10×10, same winding direction.
            // Centre pixel has winding=2 → even → outside under EO rule.
            let segs: Vec<f32> = rect_segs(-10.0, -10.0, 10.0, 10.0)
                .into_iter()
                .chain(rect_segs(-5.0, -5.0, 5.0, 5.0))
                .collect();
            let (cpu, gpu) = both_1x1(&segs, -0.5, -0.5, true);
            assert_eq!(cpu, gpu, "CPU={cpu} GPU={gpu}");
            assert_eq!(cpu, 0, "EO donut centre must be 0");
        }

        #[test]
        fn nz_donut_centre() {
            // Same two-rect donut as above but NZ rule.
            // Winding=2 ≠ 0 → inside.
            let segs: Vec<f32> = rect_segs(-10.0, -10.0, 10.0, 10.0)
                .into_iter()
                .chain(rect_segs(-5.0, -5.0, 5.0, 5.0))
                .collect();
            let (cpu, gpu) = both_1x1(&segs, -0.5, -0.5, false);
            assert_eq!(cpu, gpu, "CPU={cpu} GPU={gpu}");
            assert_eq!(cpu, 255, "NZ donut centre (winding=2) must be 255");
        }

        #[test]
        fn partial_edge() {
            // Rect whose right edge bisects the pixel exactly at x=0.5.
            // Pixel centre is (0.5, 0.5); sub-pixel samples at cx + H2[s] - 0.5 = H2[s].
            // Samples with H2[s] < 0.5 are inside the rect; ~half the 64 Halton(2)
            // samples satisfy this, giving partial coverage strictly in (0, 255).
            let segs = rect_segs(-100.0, -100.0, 0.5, 100.0);
            let (cpu, gpu) = both_1x1(&segs, 0.0, 0.0, false);
            assert_eq!(cpu, gpu, "CPU={cpu} GPU={gpu}");
            assert!(
                cpu > 0 && cpu < 255,
                "half-covered pixel must be partial (got {cpu})"
            );
        }

        #[test]
        fn multi_pixel_region() {
            // Closed right-triangle: vertices (0,0), (8,0), (0,8).
            // Pixels in the upper-left corner should be covered; bottom-right corner outside.
            #[rustfmt::skip]
            let segs: Vec<f32> = vec![
                0.0, 0.0,  8.0, 0.0,   // hypotenuse top
                8.0, 0.0,  0.0, 8.0,   // hypotenuse diagonal
                0.0, 8.0,  0.0, 0.0,   // left edge back to origin
            ];
            let cpu = aa_fill_cpu(&segs, 0.0, 0.0, 8, 8, false);
            let gpu_cov = gpu()
                .aa_fill_gpu(&segs, 0.0, 0.0, 8, 8, false)
                .unwrap_or_else(|e| panic!("GPU aa_fill failed: {e}"));

            // Find the first mismatch position for a clear failure message.
            let first_diff = cpu
                .iter()
                .zip(gpu_cov.iter())
                .enumerate()
                .find(|(_, (a, b))| a != b)
                .map(|(i, (a, b))| format!("byte {i}: CPU={a} GPU={b}"));

            assert!(
                first_diff.is_none(),
                "multi-pixel region mismatch at {}",
                first_diff.as_deref().unwrap_or("(none)")
            );

            // Structural sanity: top-left pixel (0,0) must be inside the triangle.
            assert_eq!(cpu[0], 255, "pixel (0,0) must be fully inside triangle");
            // Bottom-right pixel (7,7) must be outside.
            assert_eq!(cpu[63], 0, "pixel (7,7) must be fully outside triangle");
        }
    }
}
