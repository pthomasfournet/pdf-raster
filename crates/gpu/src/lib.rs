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
//! - `nvjpeg2k` — enables [`nvjpeg2k`] module for GPU-accelerated JPEG 2000
//!   decoding via NVIDIA nvJPEG2000.  Requires `libnvjpeg2k.so` at link time.
//! - `gpu-deskew` — enables [`npp_rotate`] module for GPU rotation via CUDA NPP
//!   (`nppiRotate_8u_C1R_Ctx`).  Requires `libnppig.so`, `libnppc.so`, and
//!   `libcudart.so` at link time.
//! - `vaapi` — enables [`vaapi`] module for VA-API hardware JPEG decoding on AMD
//!   and Intel iGPU/discrete hardware.  Requires `libva.so.2` and `libva-drm.so.2`
//!   at link time.

mod cmyk;
mod composite;
pub(crate) mod cuda;

pub mod backend;
pub mod lib_kernels;

#[cfg(feature = "cache")]
pub mod blit;
#[cfg(feature = "cache")]
pub mod cache;

#[cfg(any(feature = "nvjpeg", feature = "vaapi"))]
pub mod decode_queue;
mod fill;
pub mod jpeg;
pub mod jpeg_sof;
pub mod traits;

pub use cmyk::icc_cmyk_to_rgb_cpu;
pub use composite::{apply_soft_mask_cpu, composite_rgba8_cpu};
#[cfg(any(feature = "nvjpeg", feature = "vaapi"))]
pub use decode_queue::{DecodeQueue, JpegQueueHandle};
pub use fill::{TileRecord, aa_fill_cpu, build_tile_records};
pub use jpeg_sof::{JpegSof, JpegVariant, jpeg_sof_info, jpeg_sof_type};
pub use traits::{DecodedImage, GpuCompute, GpuDecodeError, GpuJpeg2kDecoder, GpuJpegDecoder};

#[cfg(feature = "nvjpeg")]
pub mod nvjpeg;

#[cfg(feature = "nvjpeg2k")]
pub mod nvjpeg2k;

#[cfg(feature = "gpu-deskew")]
pub mod npp_rotate;

#[cfg(feature = "vaapi")]
pub mod vaapi;

use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::Ptx;

const PTX_COMPOSITE: &str = include_str!(concat!(env!("OUT_DIR"), "/composite_rgba8.ptx"));
const PTX_SOFT_MASK: &str = include_str!(concat!(env!("OUT_DIR"), "/apply_soft_mask.ptx"));
const PTX_AA_FILL: &str = include_str!(concat!(env!("OUT_DIR"), "/aa_fill.ptx"));
const PTX_TILE_FILL: &str = include_str!(concat!(env!("OUT_DIR"), "/tile_fill.ptx"));
const PTX_ICC_CLUT: &str = include_str!(concat!(env!("OUT_DIR"), "/icc_clut.ptx"));
#[cfg(feature = "cache")]
const PTX_BLIT_IMAGE: &str = include_str!(concat!(env!("OUT_DIR"), "/blit_image.ptx"));

/// Threshold in pixels below which CPU is faster than GPU dispatch overhead.
pub const GPU_COMPOSITE_THRESHOLD: usize = 500_000;
/// Threshold for soft-mask application.
pub const GPU_SOFTMASK_THRESHOLD: usize = 500_000;
/// Minimum fill area (pixels) for GPU supersampled AA to be faster than CPU.
///
/// Calibrated on RTX 5070 + `PCIe` 5.0 via `threshold_bench`: GPU wins at ≥ 256 px
/// (4.7× faster) and exceeds 95× at 16 384 px.  The old default of 16 384 left
/// ~90× speedup on the table for fills in the 256–16 383 px range.
///
/// The absolute floor is one full CUDA warp (32 threads = 2 warps × 16 pixels each),
/// so 256 is both the measured crossover and a natural hardware-aligned minimum.
pub const GPU_AA_FILL_THRESHOLD: usize = 256;
/// Minimum fill area (pixels) for the tile-parallel analytical fill to be faster
/// than the CPU 64-sample AA path.
///
/// Calibrated on RTX 5070 + `PCIe` 5.0 via `threshold_bench`: GPU wins at ≥ 256 px
/// (2.5× faster despite CPU-side sort overhead) and exceeds 100× at 16 384 px.
/// The tile fill is used in preference to AA fill above this threshold; both are
/// well below the `PCIe` saturation point for typical PDF page fills.
pub const GPU_TILE_FILL_THRESHOLD: usize = 256;
/// Tile width in pixels (must match `TILE_W` in `tile_fill.cu`).
pub const TILE_W: u32 = 16;
/// Tile height in pixels (must match `TILE_H` in `tile_fill.cu`).
pub const TILE_H: u32 = 16;
/// Minimum pixel count for GPU ICC CMYK→RGB CLUT transform to beat CPU + `PCIe` overhead.
///
/// Applies only when a full ICC CLUT is available (clut=Some).  The matrix path
/// (clut=None) always uses the CPU AVX-512 fallback — `threshold_bench` showed the GPU
/// matrix kernel never beats AVX-512 across 256–4M pixels on this machine (`PCIe`
/// round-trip cost exceeds the cheap per-pixel computation at all measured sizes).
///
/// The CLUT path is not yet benchmarked; 500 000 px is a conservative placeholder.
/// Run `threshold_bench` with a CLUT workload to calibrate once baking is in the hot path.
pub const GPU_ICC_CLUT_THRESHOLD: usize = 500_000;

pub(crate) struct GpuKernels {
    pub(crate) composite_rgba8: CudaFunction,
    pub(crate) apply_soft_mask: CudaFunction,
    pub(crate) aa_fill: CudaFunction,
    pub(crate) tile_fill: CudaFunction,
    pub(crate) icc_cmyk_matrix: CudaFunction,
    pub(crate) icc_cmyk_clut: CudaFunction,
    #[cfg(feature = "cache")]
    pub(crate) blit_image: CudaFunction,
}

/// An initialised CUDA context and compiled kernel set.
///
/// Create once per process with [`GpuCtx::init`] and share across threads via `Arc`.
///
/// # Concurrency
///
/// All methods take `&self` and are safe to call from multiple threads, but they
/// serialize on the single internal `CudaStream`: a call from thread B blocks until
/// thread A's `stream.synchronize()` returns.  This is correct for the current
/// architecture where GPU work is occasional (threshold-gated) and pages are the
/// unit of parallelism.  If image decode is ever parallelized within a page, replace
/// the shared stream with a `thread_local!` `Arc<CudaStream>` per rayon worker.
pub struct GpuCtx {
    pub(crate) stream: Arc<CudaStream>,
    pub(crate) kernels: GpuKernels,
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
                #[cfg(feature = "cache")]
                blit_image: load(PTX_BLIT_IMAGE, "blit_image")?,
            },
        })
    }

    /// The CUDA stream this context binds all kernel launches and
    /// device allocations to.
    ///
    /// Exposed so callers (e.g. the Phase 9 image cache) can share
    /// the same stream for upload + dispatch + download — single-
    /// stream serialisation eliminates the need for explicit
    /// cross-stream synchronisation.
    #[must_use]
    pub const fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
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
    /// Panics if `cmyk.len()` is not a multiple of 4, or if `clut` is `Some` and
    /// `table.len() != grid_n^4 * 3`.
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

        // Early-out before any CLUT validation: empty input always produces empty output.
        let n = cmyk.len() / 4;
        if n == 0 {
            return Ok(Vec::new());
        }

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
        // Matrix path (clut=None): CPU AVX-512 always beats GPU on this machine —
        // threshold_bench showed the PCIe round-trip cost exceeds the compute cost
        // at all measured sizes (256–4M pixels).  Always use the CPU path here.
        if clut.is_none() {
            return Ok(icc_cmyk_to_rgb_cpu(cmyk, None));
        }
        if n < GPU_ICC_CLUT_THRESHOLD {
            return Ok(icc_cmyk_to_rgb_cpu(cmyk, clut));
        }

        self.icc_cmyk_to_rgb_gpu(cmyk, clut)
    }

    /// Unconditional GPU dispatch for CMYK→RGB (skips threshold check).
    ///
    /// Use this when the caller has already decided GPU is appropriate
    /// (e.g. benchmarking or when the pixel count is known to be large).
    ///
    /// # Errors
    ///
    /// Returns an error if GPU data transfer or kernel launch fails.
    ///
    /// # Panics
    ///
    /// Panics if `cmyk.len()` is not a multiple of 4 or if the pixel count
    /// overflows `u32::MAX`.
    pub fn icc_cmyk_to_rgb_gpu(
        &self,
        cmyk: &[u8],
        clut: Option<(&[u8], u32)>,
    ) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        assert!(
            cmyk.len().is_multiple_of(4),
            "cmyk.len() must be a multiple of 4 (got {})",
            cmyk.len()
        );
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
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "callers validate n ≤ u32::MAX via u32::try_from before calling this; n.div_ceil(256) is therefore also ≤ u32::MAX"
)]
pub(crate) const fn launch_cfg(n: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (n.div_ceil(256) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    }
}

#[cfg(test)]
mod tests {
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
