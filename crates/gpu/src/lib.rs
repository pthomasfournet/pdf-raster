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

use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::Ptx;

const PTX_COMPOSITE: &str = include_str!(concat!(env!("OUT_DIR"), "/composite_rgba8.ptx"));
const PTX_SOFT_MASK: &str = include_str!(concat!(env!("OUT_DIR"), "/apply_soft_mask.ptx"));

/// Threshold in pixels below which CPU is faster than GPU dispatch overhead.
pub const GPU_COMPOSITE_THRESHOLD: usize = 500_000;
/// Threshold for soft-mask application.
pub const GPU_SOFTMASK_THRESHOLD: usize = 500_000;

struct GpuKernels {
    composite_rgba8: CudaFunction,
    apply_soft_mask: CudaFunction,
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
        // PushKernelArg::arg returns &mut Self (builder pattern); results intentionally dropped.
        #[allow(unused_results)]
        {
            builder.arg(&d_src);
            builder.arg(&mut d_dst);
            builder.arg(&n_u32);
            unsafe { builder.launch(cfg) }?;
        }

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
        // PushKernelArg::arg returns &mut Self (builder pattern); results intentionally dropped.
        #[allow(unused_results)]
        {
            builder.arg(&mut d_pixels);
            builder.arg(&d_mask);
            builder.arg(&n_u32);
            unsafe { builder.launch(cfg) }?;
        }

        stream.synchronize()?;
        stream.memcpy_dtoh(&d_pixels, pixels)?;
        Ok(())
    }
}

#[allow(clippy::cast_possible_truncation)]
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
        d[3] = a_out.min(255) as u8;
    }
}

/// CPU fallback for `apply_soft_mask`.
pub fn apply_soft_mask_cpu(pixels: &mut [u8], mask: &[u8]) {
    for (p, &m) in pixels.chunks_exact_mut(4).zip(mask) {
        let a = u32::from(p[3]);
        let m = u32::from(m);
        // a*m is at most 255*255 = 65025; +127 = 65152 < u32::MAX; /255 ≤ 255: safe cast.
        #[allow(clippy::cast_possible_truncation)]
        let scaled = ((a * m + 127) / 255) as u8;
        p[3] = scaled;
    }
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

    // --- GPU tests ---
    // If GpuCtx::init() fails (no CUDA device), tests skip gracefully via early return.

    #[allow(clippy::cast_possible_truncation)]
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
        let ctx = match GpuCtx::init() {
            Ok(c) => c,
            Err(_) => return, // no GPU — skip gracefully
        };

        let n = 1000;
        let (src, dst_orig) = make_composite_data(n);

        // CPU reference
        let mut dst_cpu = dst_orig.clone();
        composite_rgba8_cpu(&src, &mut dst_cpu);

        // GPU path (bypass threshold via internal method)
        let mut dst_gpu = dst_orig.clone();
        ctx.composite_rgba8_gpu(&src, &mut dst_gpu)
            .expect("GPU composite_rgba8 failed");

        assert_eq!(
            dst_cpu, dst_gpu,
            "GPU and CPU composite_rgba8 results differ"
        );
    }

    #[allow(clippy::cast_possible_truncation)]
    #[test]
    fn gpu_soft_mask_matches_cpu() {
        let ctx = match GpuCtx::init() {
            Ok(c) => c,
            Err(_) => return, // no GPU — skip gracefully
        };

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
}
