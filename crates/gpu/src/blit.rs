//! Phase 9 task 4 — image blit kernel dispatcher.
//!
//! Wraps `kernels/blit_image.cu`.  Affine-transforms a cached source
//! image into a [`DevicePageBuffer`] using the inverse CTM, with
//! nearest-neighbour sampling.  Output pixels in the bounding box that
//! fall outside the image's `[0, 1]²` extent are left untouched (the
//! page buffer is zero-initialised at allocation, so they composite
//! as fully transparent in the host-side overlay step).
//!
//! # Sampling parity
//!
//! The kernel matches the renderer's existing CPU sampler in
//! `crates/pdf_interp/src/renderer/page/mod.rs`:
//!
//! - Nearest-neighbour (no bilinear).
//! - `floor(u * src_w)` for the x index, with `min(.., src_w - 1)` to
//!   guard against `u == 1.0` rounding past the last column.
//! - `floor((1 - v) * src_h)` for y (PDF image-space y points up;
//!   bitmap y points down).
//!
//! This is required by the spec's "≤ 1 LSB per channel per pixel"
//! acceptance criterion — bilinear or different rounding would break
//! parity with the CPU baseline.

use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};

use crate::GpuCtx;
use crate::cache::{CachedDeviceImage, DevicePageBuffer, ImageLayout};

/// Inverse-CTM coefficients precomputed by the caller.
///
/// Given a 3x3 affine PDF CTM `[a, b, c, d, e, f]` applied as
/// `(x, y) → (a*x + c*y + e, b*x + d*y + f)`, this struct holds the
/// *coefficients used by the kernel's per-pixel formula*, not the
/// inverse matrix itself.  See `kernels/blit_image.cu` for the exact
/// equations; the host CPU path computes the same formulas.
#[derive(Debug, Clone, Copy)]
pub struct InverseCtm {
    /// `d * inv_det` — multiplies `dx_rel` to produce the source `u`.
    pub u_dx: f32,
    /// `-c * inv_det` — multiplies `dy_rel` to produce the source `u`.
    pub u_dy: f32,
    /// `-b * inv_det` — multiplies `dx_rel` to produce the source `v`.
    pub v_dx: f32,
    /// `a * inv_det` — multiplies `dy_rel` to produce the source `v`.
    pub v_dy: f32,
    /// CTM `e` (translation x).  `dx_rel = dx - tx`.
    pub tx: f32,
    /// CTM `f` (translation y).  `dy_rel = (page_h - dy) - ty`.
    pub ty: f32,
}

/// Narrow f64 to f32 for the kernel's per-pixel arithmetic.  The
/// kernel uses f32 for one-FMA-per-pixel performance; PDF-realistic
/// CTM scales are bounded enough that the truncation is sub-pixel.
#[expect(
    clippy::cast_possible_truncation,
    reason = "intentional f64→f32 narrowing — see fn doc"
)]
const fn narrow(x: f64) -> f32 {
    x as f32
}

impl InverseCtm {
    /// Build from a PDF CTM `[a, b, c, d, e, f]`.  Returns `None` when
    /// the matrix is singular (det ≈ 0); the caller should fall back
    /// to the CPU path or skip the image.
    #[must_use]
    #[expect(
        clippy::many_single_char_names,
        reason = "PDF CTM components a-f are spec terminology — renaming would obscure the math"
    )]
    pub fn from_ctm(ctm: [f64; 6]) -> Option<Self> {
        let [a, b, c, d, e, f] = ctm;
        let det = a.mul_add(d, -(b * c));
        if det.abs() < 1e-12 {
            return None;
        }
        let inv_det = 1.0 / det;
        Some(Self {
            u_dx: narrow(d * inv_det),
            u_dy: narrow(-c * inv_det),
            v_dx: narrow(-b * inv_det),
            v_dy: narrow(a * inv_det),
            tx: narrow(e),
            ty: narrow(f),
        })
    }

    /// Pack into the 6-float layout the kernel expects.
    const fn as_array(self) -> [f32; 6] {
        [self.u_dx, self.u_dy, self.v_dx, self.v_dy, self.tx, self.ty]
    }
}

/// Output bounding box for the blit kernel — only pixels within this
/// rectangle are dispatched.  Caller computes the conservative AABB
/// of the image's transformed extent on the page.
#[derive(Debug, Clone, Copy)]
pub struct BlitBbox {
    /// Inclusive left edge (page pixels).
    pub x0: i32,
    /// Inclusive top edge (page pixels).
    pub y0: i32,
    /// Exclusive right edge (page pixels).
    pub x1: i32,
    /// Exclusive bottom edge (page pixels).
    pub y1: i32,
}

impl BlitBbox {
    /// Width in pixels, saturating to 0 for an inverted or zero-area
    /// bbox.  `saturating_sub` rules out the i32-overflow case that
    /// `(x1 - x0).max(0)` could panic on for adversarial inputs.
    fn width(self) -> u32 {
        u32::try_from(self.x1.saturating_sub(self.x0)).unwrap_or(0)
    }
    fn height(self) -> u32 {
        u32::try_from(self.y1.saturating_sub(self.y0)).unwrap_or(0)
    }
}

impl GpuCtx {
    /// Dispatch the image-blit kernel: transform `src` into `dst`
    /// using `inv_ctm`, writing only pixels inside `bbox`.
    ///
    /// Mask-layout images aren't supported by this kernel — the caller
    /// must route them through the CPU path (the GPU dispatcher in the
    /// renderer is image-specific and falls back when this returns
    /// `Err(BlitError::UnsupportedLayout)`).
    ///
    /// # Errors
    /// - [`BlitError::UnsupportedLayout`] for `ImageLayout::Mask`.
    /// - [`BlitError::Cuda`] for any underlying CUDA failure (PTX
    ///   launch, argument upload, etc.).
    pub fn blit_image_to_buffer(
        &self,
        src: &CachedDeviceImage,
        dst: &mut DevicePageBuffer,
        inv_ctm: InverseCtm,
        bbox: BlitBbox,
        page_h: f32,
    ) -> Result<(), BlitError> {
        // Caller contract: bbox is non-inverted.  An inverted bbox
        // would silently render zero pixels (BlitBbox::width returns
        // 0 via saturating_sub), masking a programming error.  The
        // page_h must match dst.height for the kernel's y-flip math.
        debug_assert!(
            bbox.x0 <= bbox.x1 && bbox.y0 <= bbox.y1,
            "blit bbox is inverted: ({}..{}, {}..{})",
            bbox.x0,
            bbox.x1,
            bbox.y0,
            bbox.y1,
        );
        #[expect(
            clippy::cast_precision_loss,
            reason = "debug_assert: dst.height up to 2^24 fits losslessly in f32; PDF pages are far smaller"
        )]
        let height_f = dst.height as f32;
        // Exact equality: callers should pass `page_h = dst.height
        // as f32`.  Any non-zero discrepancy corrupts the kernel's
        // y-flip math (`page_h - dy`) on every sample.
        #[expect(
            clippy::float_cmp,
            reason = "page_h is f32-cast of dst.height (u32); equality is exact for u32 ≤ 2^24"
        )]
        let page_h_matches = page_h == height_f;
        debug_assert!(
            page_h_matches,
            "page_h ({page_h}) must equal dst.height as f32 ({height_f})",
        );

        let layout_code: i32 = match src.layout {
            ImageLayout::Rgb => 0,
            ImageLayout::Gray => 1,
            ImageLayout::Mask => return Err(BlitError::UnsupportedLayout),
        };
        if bbox.width() == 0 || bbox.height() == 0 {
            return Ok(());
        }

        self.launch_blit_image_async(
            &src.device_ptr,
            (src.width, src.height),
            layout_code,
            &dst.rgba,
            (dst.width, dst.height),
            bbox,
            &inv_ctm,
            page_h,
        )
    }

    /// Async kernel launch for the image-blit kernel.
    ///
    /// This is the trait-facing variant: it takes raw device byte
    /// buffers and dimensions, instead of the Phase 9 cache wrappers
    /// `CachedDeviceImage` / `DevicePageBuffer`.  It does **not**
    /// synchronise, does **not** touch host memory, and is the helper
    /// `CudaBackend::record_blit_image` calls.
    ///
    /// The caller is responsible for keeping `d_src`, `d_dst`, and the
    /// inverse-CTM upload alive until the stream has executed the
    /// launch (the helper allocates the inverse-CTM device buffer
    /// internally and returns once the kernel is queued).
    ///
    /// `layout_code` is the kernel's enum value: `0 = Rgb`, `1 = Gray`.
    /// Mask layout is rejected by the public wrapper before this
    /// helper is reached.
    ///
    /// # Errors
    /// - [`BlitError::DimensionsTooLarge`] if any dim doesn't fit i32.
    /// - [`BlitError::Cuda`] for any underlying cudarc failure.
    #[expect(
        clippy::too_many_arguments,
        reason = "kernel arg count is fixed by the PTX signature; grouping into a struct would just hide the mapping"
    )]
    pub(crate) fn launch_blit_image_async(
        &self,
        d_src: &CudaSlice<u8>,
        src_dims: (u32, u32),
        layout_code: i32,
        d_dst: &CudaSlice<u8>,
        dst_dims: (u32, u32),
        bbox: BlitBbox,
        inv_ctm: &InverseCtm,
        page_h: f32,
    ) -> Result<(), BlitError> {
        // 16×16 blocks: 256 threads / block, full-warp aligned, fits
        // in any modern SM's register budget for this kernel.
        const TILE: u32 = 16;

        let bw = bbox.width();
        let bh = bbox.height();
        if bw == 0 || bh == 0 {
            return Ok(());
        }

        let grid_x = bw.div_ceil(TILE);
        let grid_y = bh.div_ceil(TILE);
        let cfg = LaunchConfig {
            grid_dim: (grid_x, grid_y, 1),
            block_dim: (TILE, TILE, 1),
            shared_mem_bytes: 0,
        };

        let stream = &self.stream;
        let inv_ctm_arr = inv_ctm.as_array();
        let d_inv_ctm = stream.clone_htod(&inv_ctm_arr).map_err(BlitError::cuda)?;

        let src_w = i32::try_from(src_dims.0).map_err(|_| BlitError::DimensionsTooLarge)?;
        let src_h = i32::try_from(src_dims.1).map_err(|_| BlitError::DimensionsTooLarge)?;
        let dst_w = i32::try_from(dst_dims.0).map_err(|_| BlitError::DimensionsTooLarge)?;
        let dst_h = i32::try_from(dst_dims.1).map_err(|_| BlitError::DimensionsTooLarge)?;

        let mut builder = stream.launch_builder(&self.kernels.blit_image);
        let _ = builder.arg(d_src);
        let _ = builder.arg(&src_w);
        let _ = builder.arg(&src_h);
        let _ = builder.arg(&layout_code);
        let _ = builder.arg(d_dst);
        let _ = builder.arg(&dst_w);
        let _ = builder.arg(&dst_h);
        let _ = builder.arg(&bbox.x0);
        let _ = builder.arg(&bbox.y0);
        let _ = builder.arg(&bbox.x1);
        let _ = builder.arg(&bbox.y1);
        let _ = builder.arg(&d_inv_ctm);
        let _ = builder.arg(&page_h);

        // SAFETY: kernel signature in kernels/blit_image.cu matches
        // the argument list above; bounds are validated by the
        // checked casts on src_w/src_h/dst_w/dst_h.
        let _ = unsafe { builder.launch(cfg) }.map_err(BlitError::cuda)?;
        // Don't synchronise here — let the caller batch multiple
        // blits and download the buffer once at end-of-page.
        Ok(())
    }
}

/// Errors from [`GpuCtx::blit_image_to_buffer`].
#[derive(Debug)]
pub enum BlitError {
    /// `ImageLayout::Mask` — caller must route through the CPU path.
    UnsupportedLayout,
    /// Source or destination dimensions don't fit in `i32`.  Indicates
    /// a malformed image or page; should never happen at PDF-realistic
    /// resolutions (max u32 wide ≈ 4.3B px ≫ any real page).
    DimensionsTooLarge,
    /// Underlying cudarc driver error.
    Cuda(cudarc::driver::DriverError),
}

impl BlitError {
    const fn cuda(e: cudarc::driver::DriverError) -> Self {
        Self::Cuda(e)
    }
}

impl std::fmt::Display for BlitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedLayout => write!(f, "blit kernel does not handle Mask layout"),
            Self::DimensionsTooLarge => {
                write!(f, "image or page dimensions exceed i32::MAX")
            }
            Self::Cuda(e) => write!(f, "cuda: {e}"),
        }
    }
}

impl std::error::Error for BlitError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Cuda(e) => Some(e),
            Self::UnsupportedLayout | Self::DimensionsTooLarge => None,
        }
    }
}

#[cfg(test)]
#[expect(
    clippy::similar_names,
    reason = "PDF spec uses paired (dx, dy) / (dx_rel, dy_rel) terminology; renaming would obscure the math"
)]
#[expect(
    clippy::float_cmp,
    reason = "test asserts exact f32 representation of 0.0 / 1.0 / -0.25 — all exactly representable in f32"
)]
mod tests {
    use super::*;

    #[test]
    fn inverse_ctm_singular_returns_none() {
        // det = 1*1 - 1*1 = 0 → singular.
        let inv = InverseCtm::from_ctm([1.0, 1.0, 1.0, 1.0, 0.0, 0.0]);
        assert!(inv.is_none());
    }

    #[test]
    fn inverse_ctm_identity_round_trips() {
        let inv = InverseCtm::from_ctm([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]).expect("identity");
        // u = 1*dx_rel + 0*dy_rel = dx_rel
        // v = 0*dx_rel + 1*dy_rel = dy_rel
        assert_eq!(inv.u_dx, 1.0);
        assert_eq!(inv.u_dy, 0.0);
        assert_eq!(inv.v_dx, 0.0);
        assert_eq!(inv.v_dy, 1.0);
    }

    #[test]
    fn bbox_width_height_clamped_to_u32() {
        let b = BlitBbox {
            x0: 5,
            y0: 5,
            x1: 3,
            y1: 3,
        };
        assert_eq!(b.width(), 0);
        assert_eq!(b.height(), 0);
    }

    /// CPU reference: nearest-neighbour sampling matching the kernel's
    /// formula in `kernels/blit_image.cu`.  Returns the source pixel
    /// indices `(ix, iy)` for output `(dx, dy)`, or None if (u, v) lies
    /// outside `[0, 1]²`.
    ///
    /// All casts intentionally mirror the kernel's f32 arithmetic so
    /// the reference produces byte-identical results.  Cast-precision
    /// lints are suppressed; that's the point — losing precision in
    /// the same way the GPU does is what makes this a reference.
    #[expect(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::suboptimal_flops,
        reason = "test reference: arithmetic intentionally mirrors the kernel's f32 math byte-for-byte"
    )]
    fn cpu_reference_sample(
        dx: i32,
        dy: i32,
        inv: InverseCtm,
        page_h: f32,
        src_w: u32,
        src_h: u32,
    ) -> Option<(u32, u32)> {
        let dx_rel = dx as f32 - inv.tx;
        let dy_rel = (page_h - dy as f32) - inv.ty;
        let u = inv.u_dx * dx_rel + inv.u_dy * dy_rel;
        let v = inv.v_dx * dx_rel + inv.v_dy * dy_rel;
        if !(0.0..=1.0).contains(&u) || !(0.0..=1.0).contains(&v) {
            return None;
        }
        let ix = ((u * src_w as f32) as u32).min(src_w - 1);
        let iy = (((1.0 - v) * src_h as f32) as u32).min(src_h - 1);
        Some((ix, iy))
    }

    /// GPU integration test: blit a 4×4 RGB image into a 4×4 page
    /// buffer using a typical PDF CTM (scale + y-flip + translate).
    /// For every output pixel, compare the GPU readback against a
    /// CPU reference that uses the exact same nearest-neighbour
    /// formula — the test is testing the *kernel*, not our mental
    /// model of PDF coordinates.
    #[cfg(feature = "gpu-validation")]
    #[test]
    #[expect(
        clippy::cast_precision_loss,
        clippy::cast_sign_loss,
        reason = "test arithmetic on small constant dimensions; all values fit losslessly"
    )]
    fn blit_kernel_matches_cpu_reference_byte_for_byte() {
        use crate::cache::{DeviceImageCache, ImageLayout, InsertRequest};
        use crate::cache::{DocId, HostBudget, ObjId, VramBudget};
        use std::sync::Arc;

        const SRC: u32 = 4;
        const PAGE: u32 = 4;
        const PAGE_I: i32 = 4;

        let ctx = cudarc::driver::CudaContext::new(0).expect("CUDA device 0");
        let stream = ctx.default_stream();
        let gpu = GpuCtx::init().expect("gpu init");

        // Distinct bytes per RGB channel so a one-pixel mismatch is
        // visible.  Step 7 stays away from any obvious row/col stride.
        let pixels: Vec<u8> = (0..SRC * SRC * 3)
            .map(|i| u8::try_from(((i * 7) + 3) % 251).expect("fits"))
            .collect();
        let h = DeviceImageCache::hash_bytes(&pixels);

        let cache = DeviceImageCache::new(
            Arc::clone(&stream),
            VramBudget {
                vram_bytes: 1 << 16,
            },
            HostBudget { host_bytes: 0 },
        );
        let cached = cache
            .insert(InsertRequest {
                doc: DocId([0; 32]),
                obj: ObjId(1),
                hash: h,
                width: SRC,
                height: SRC,
                layout: ImageLayout::Rgb,
                pixels: &pixels,
            })
            .expect("insert");

        // Standard PDF image placement: scale unit square to 4×4 with
        // y-flip, translate so the image's top-left lands at device
        // (0, 0).
        let ctm = [4.0, 0.0, 0.0, -4.0, 0.0, 4.0];
        let inv = InverseCtm::from_ctm(ctm).expect("non-singular");
        let mut page = DevicePageBuffer::new(Arc::clone(&stream), PAGE, PAGE).expect("page");
        let bbox = BlitBbox {
            x0: 0,
            y0: 0,
            x1: PAGE_I,
            y1: PAGE_I,
        };
        let page_h = PAGE as f32;

        gpu.blit_image_to_buffer(&cached, &mut page, inv, bbox, page_h)
            .expect("blit");
        let host = page.download().expect("download");

        let mut compared = 0;
        for dy in 0..PAGE_I {
            for dx in 0..PAGE_I {
                let off = (dy as usize * PAGE as usize + dx as usize) * 4;
                let alpha = host[off + 3];
                match cpu_reference_sample(dx, dy, inv, page_h, SRC, SRC) {
                    Some((ix, iy)) => {
                        assert_eq!(alpha, 255, "alpha at ({dx},{dy}) — in-bounds sample");
                        let src_off = (iy as usize * SRC as usize + ix as usize) * 3;
                        assert_eq!(
                            host[off], pixels[src_off],
                            "R at ({dx},{dy}) (ix={ix}, iy={iy})",
                        );
                        assert_eq!(
                            host[off + 1],
                            pixels[src_off + 1],
                            "G at ({dx},{dy}) (ix={ix}, iy={iy})",
                        );
                        assert_eq!(
                            host[off + 2],
                            pixels[src_off + 2],
                            "B at ({dx},{dy}) (ix={ix}, iy={iy})",
                        );
                        compared += 1;
                    }
                    None => {
                        assert_eq!(alpha, 0, "alpha at ({dx},{dy}) — out-of-bounds sample");
                    }
                }
            }
        }
        // Sanity: at least some pixels must have been in-bounds.
        assert!(compared > 0, "test inputs degenerate; compared zero pixels");
    }
}
