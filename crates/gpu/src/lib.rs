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
    CudaContext, CudaFunction, CudaModule, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::Ptx;

const PTX_COMPOSITE: &str = include_str!(concat!(env!("OUT_DIR"), "/composite_rgba8.ptx"));
const PTX_SOFT_MASK: &str = include_str!(concat!(env!("OUT_DIR"), "/apply_soft_mask.ptx"));
const PTX_AA_FILL: &str = include_str!(concat!(env!("OUT_DIR"), "/aa_fill.ptx"));

/// Threshold in pixels below which CPU is faster than GPU dispatch overhead.
pub const GPU_COMPOSITE_THRESHOLD: usize = 500_000;
/// Threshold for soft-mask application.
pub const GPU_SOFTMASK_THRESHOLD: usize = 500_000;
/// Minimum fill area (pixels) for GPU supersampled AA to be faster than CPU.
///
/// Below this threshold the H2D/D2H transfer latency for the segment list and
/// coverage buffer dominates. Calibrated for RTX 5070 + [`PCIe`] 5.0 at ~150 DPI.
pub const GPU_AA_FILL_THRESHOLD: usize = 16_384;

struct GpuKernels {
    composite_rgba8: CudaFunction,
    apply_soft_mask: CudaFunction,
    aa_fill: CudaFunction,
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
    fn aa_fill_gpu(
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
            // Sub-pixel sample coordinates: pixel centre + [0,1) Halton offset − 0.5
            // so samples are uniformly distributed within the pixel square.
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
            // Scale 0..64 → 0..255 rounding to nearest (matches GPU: (hits*255+32)>>6).
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

/// Test whether sample point `(sx, sy)` is inside the filled path defined by
/// `segs` (packed `[x0,y0,x1,y1]` per segment, f32).
///
/// Returns `true` when the non-zero winding number is non-zero (or parity is
/// odd for even-odd rule).
fn aa_fill_cpu_sample(segs: &[f32], sx: f32, sy: f32, eo: bool) -> bool {
    let mut winding = 0i32;
    for seg in segs.chunks_exact(4) {
        let (x0, y0, x1, y1) = (seg[0], seg[1], seg[2], seg[3]);
        // Upward crossing: y0 <= sy < y1
        if y0 <= sy && sy < y1 {
            let t = (sy - y0) / (y1 - y0);
            let xi = t.mul_add(x1 - x0, x0);
            if xi >= sx {
                winding += 1;
            }
        // Downward crossing: y1 <= sy < y0
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
    use super::{GpuCtx, apply_soft_mask_cpu, composite_rgba8_cpu};

    // --- CPU tests ---

    #[test]
    fn composite_cpu_opaque_src() {
        let src = [200u8, 100, 50, 255]; // opaque src
        let mut dst = [10u8, 20, 30, 128];
        composite_rgba8_cpu(&src, &mut dst);
        assert_eq!(dst, [200, 100, 50, 255]);
    }

    #[test]
    fn composite_cpu_transparent_src() {
        let src = [200u8, 100, 50, 0]; // fully transparent src
        let mut dst = [10u8, 20, 30, 128];
        let expected = dst;
        composite_rgba8_cpu(&src, &mut dst);
        assert_eq!(dst, expected);
    }

    #[test]
    fn composite_cpu_half_alpha() {
        // white src at half alpha over opaque black dst
        // a_out = 128 + (255*127 + 127)/255 = 128 + 127 = 255
        // blended = (255*128 + 0*255*127/255 + 255/2) / 255 = (32640 + 127) / 255 ≈ 128
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
        // (240*255 + 127) / 255 = 240
        assert_eq!(pixels[3], 240);
    }

    #[test]
    fn soft_mask_cpu_half() {
        let mut pixels = [100u8, 150, 200, 200];
        let mask = [128u8];
        apply_soft_mask_cpu(&mut pixels, &mask);
        // (200*128 + 127) / 255 = 25727/255 = 100
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

    // --- aa_fill_cpu tests ---

    #[test]
    fn aa_fill_cpu_solid_rect_full_coverage() {
        // A large rectangle fully covering the pixel → all 64 samples inside → 255.
        // Segments: rectangle (−100,−100)→(100,100) counter-clockwise.
        let segs: Vec<f32> = vec![
            -100.0, -100.0, 100.0, -100.0, // bottom edge (horiz, ignored by winding)
            100.0, -100.0, 100.0, 100.0, // right edge upward
            100.0, 100.0, -100.0, 100.0, // top edge (horiz, ignored)
            -100.0, 100.0, -100.0, -100.0, // left edge downward
        ];
        let cov = super::aa_fill_cpu(&segs, 0.0, 0.0, 1, 1, false);
        assert_eq!(cov.len(), 1);
        assert_eq!(cov[0], 255, "fully covered pixel should be 255");
    }

    #[test]
    fn aa_fill_cpu_outside_rect_zero_coverage() {
        // Pixel at (200,200) is outside the rectangle (0..10)×(0..10) → 0.
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
        // Even-odd: outer rect (−10..10)² and inner rect (−5..5)².
        // Centre pixel at (0,0) has winding=2 for NZ → inside, but EO parity=0 → outside.
        let outer: [f32; 16] = [
            -10.0, -10.0, 10.0, -10.0, 10.0, -10.0, 10.0, 10.0, 10.0, 10.0, -10.0, 10.0, -10.0,
            10.0, -10.0, -10.0,
        ];
        let inner: [f32; 16] = [
            -5.0, -5.0, 5.0, -5.0, 5.0, -5.0, 5.0, 5.0, 5.0, 5.0, -5.0, 5.0, -5.0, 5.0, -5.0, -5.0,
        ];
        let segs: Vec<f32> = outer.iter().chain(inner.iter()).copied().collect();

        // NZ: (0,0) has winding 2 → inside → full coverage
        let cov_nz = super::aa_fill_cpu(&segs, -0.5, -0.5, 1, 1, false);
        assert!(
            cov_nz[0] > 200,
            "NZ centre should be mostly covered, got {}",
            cov_nz[0]
        );

        // EO: (0,0) has winding 2 → parity 0 → outside
        let cov_eo = super::aa_fill_cpu(&segs, -0.5, -0.5, 1, 1, true);
        assert_eq!(
            cov_eo[0], 0,
            "EO centre should be 0 (donut hole), got {}",
            cov_eo[0]
        );
    }

    // --- GPU tests ---
    // If GpuCtx::init() fails (no CUDA device), tests skip gracefully via early return.

    #[expect(clippy::cast_possible_truncation, reason = "i % 256 always fits u8")]
    fn make_composite_data(n_pixels: usize) -> (Vec<u8>, Vec<u8>) {
        let mut src = Vec::with_capacity(n_pixels * 4);
        let mut dst = Vec::with_capacity(n_pixels * 4);
        for i in 0..n_pixels {
            let b = (i % 256) as u8;
            src.extend_from_slice(&[b, b.wrapping_add(10), b.wrapping_add(20), 200]);
            dst.extend_from_slice(&[
                b.wrapping_add(5),
                b.wrapping_add(15),
                b.wrapping_add(25),
                180,
            ]);
        }
        (src, dst)
    }

    #[test]
    fn gpu_composite_matches_cpu() {
        let Ok(ctx) = GpuCtx::init() else { return }; // no GPU — skip gracefully

        let n = 1000;
        let (src, dst_orig) = make_composite_data(n);

        // CPU reference
        let mut expected = dst_orig.clone();
        composite_rgba8_cpu(&src, &mut expected);

        // GPU path (bypass threshold via internal method)
        let mut actual = dst_orig;
        ctx.composite_rgba8_gpu(&src, &mut actual)
            .expect("GPU composite_rgba8 failed");

        assert_eq!(
            expected, actual,
            "GPU and CPU composite_rgba8 results differ"
        );
    }

    #[expect(clippy::cast_possible_truncation, reason = "i % 256 always fits u8")]
    #[test]
    fn gpu_soft_mask_matches_cpu() {
        let Ok(ctx) = GpuCtx::init() else { return }; // no GPU — skip gracefully

        let n = 1000;
        let mut pixels_orig: Vec<u8> = (0..n)
            .flat_map(|i: usize| {
                let b = (i % 256) as u8;
                [b, b.wrapping_add(1), b.wrapping_add(2), 200u8]
            })
            .collect();
        let mask: Vec<u8> = (0..n).map(|i: usize| ((i * 37 + 50) % 256) as u8).collect();

        // CPU reference
        let mut pixels_cpu = pixels_orig.clone();
        apply_soft_mask_cpu(&mut pixels_cpu, &mask);

        // GPU path (bypass threshold via internal method)
        ctx.apply_soft_mask_gpu(&mut pixels_orig, &mask)
            .expect("GPU apply_soft_mask failed");

        assert_eq!(
            pixels_cpu, pixels_orig,
            "GPU and CPU apply_soft_mask results differ"
        );
    }

    #[test]
    fn gpu_aa_fill_matches_cpu() {
        let Ok(ctx) = GpuCtx::init() else { return }; // no GPU — skip gracefully

        // A rectangle (0..20)×(0..20) rendered at 8×8 pixel output.
        let segs: Vec<f32> = vec![
            0.0, 0.0, 20.0, 0.0, 20.0, 0.0, 20.0, 20.0, 20.0, 20.0, 0.0, 20.0, 0.0, 20.0, 0.0, 0.0,
        ];
        let (w, h) = (8u32, 8u32);

        let cpu = super::aa_fill_cpu(&segs, 0.0, 0.0, w, h, false);

        // Bypass threshold to force GPU path.
        let gpu = ctx
            .aa_fill_gpu(&segs, 0.0, 0.0, w, h, false)
            .expect("GPU aa_fill failed");

        assert_eq!(cpu.len(), gpu.len());
        for (i, (&c, &g)) in cpu.iter().zip(gpu.iter()).enumerate() {
            assert_eq!(c, g, "pixel {i}: CPU coverage {c} != GPU coverage {g}");
        }
    }
}
