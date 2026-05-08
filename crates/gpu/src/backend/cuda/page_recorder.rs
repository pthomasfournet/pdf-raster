//! CUDA page recorder — buffers `record_*` calls until `submit_page`.
//!
//! All `record_*` methods route through the per-kernel
//! `launch_<name>_async` helpers in `lib_kernels::*`, which queue the
//! kernel onto the context's default stream **without** synchronising or
//! downloading.  `submit_page` records a `CudaEvent`; `wait_page` blocks
//! the host on it.  Callers therefore stall once per page rather than
//! once per kernel.
//!
//! ## Concurrency note
//!
//! All buffer references in `params::*Params` are `&CudaSlice<u8>`
//! (immutable), even for read-modify-write operands like the
//! composite destination.  At the cudarc layer the kernel receives a
//! raw `CUdeviceptr`, so the &-vs-&mut distinction has no effect on
//! the launch.  In-place writes are race-free because the CUDA
//! backend serialises every launch on a single shared stream — by the
//! time the next launch starts, the previous one has finished writing.

#![expect(
    clippy::needless_pass_by_value,
    reason = "record_* methods take params by value to mirror the GpuBackend trait shape; the trait could take &Params, but every backend would still want to destructure, so the by-value form is canonical"
)]

use std::sync::Arc;

use cudarc::driver::CudaEvent;

use super::{StringError, be};
use crate::GpuCtx;
use crate::backend::{BackendError, Result, params};
#[cfg(feature = "cache")]
use crate::blit::{BlitBbox, InverseCtm};

pub(super) struct PageRecorder {
    pub(super) ctx: Arc<GpuCtx>,
}

/// A synchronisation token returned by [`PageRecorder::submit_page`].
///
/// Wraps a [`CudaEvent`] recorded on the backend's stream.  `wait_page`
/// calls `event.synchronize()`, which blocks the host until every prior
/// kernel queued on that stream has completed.
#[derive(Debug)]
pub struct PageFence(CudaEvent);

impl PageRecorder {
    pub(super) const fn new(ctx: Arc<GpuCtx>) -> Self {
        Self { ctx }
    }

    /// Begin a new page.
    ///
    /// The CUDA backend has nothing to do here today — there's a single
    /// shared stream and recording just means queuing into it.  The
    /// method exists to mirror the `GpuBackend` trait shape so the
    /// Vulkan backend (which **does** allocate a per-page command
    /// buffer) and the CUDA backend share the same call site in
    /// callers.
    #[expect(
        clippy::unused_self,
        clippy::unnecessary_wraps,
        clippy::missing_const_for_fn,
        reason = "shape-only impl; the Vulkan equivalent will allocate a command buffer here"
    )]
    pub(super) fn begin_page(&self) -> Result<()> {
        Ok(())
    }

    /// Record a stream event for the current page's queued work.
    ///
    /// All prior `record_*` launches on this stream are ordered before
    /// the event.  The host can then block on the event with
    /// `wait_page` instead of synchronising the whole stream.
    pub(super) fn submit_page(&self) -> Result<PageFence> {
        let event = self.ctx.stream().record_event(None).map_err(be)?;
        Ok(PageFence(event))
    }

    /// Block until the work covered by `fence` has completed.
    #[expect(
        clippy::unused_self,
        reason = "self-arg kept for symmetry with the trait method and forward-compat with future per-recorder fence pools"
    )]
    pub(super) fn wait_page(&self, fence: PageFence) -> Result<()> {
        fence.0.synchronize().map_err(be)
    }

    pub(super) fn record_blit_image(
        &self,
        p: params::BlitParams<'_, super::CudaBackend>,
    ) -> Result<()> {
        p.validate()?;
        Self::record_blit_image_inner(&self.ctx, p)
    }

    #[cfg(feature = "cache")]
    fn record_blit_image_inner(
        ctx: &Arc<GpuCtx>,
        p: params::BlitParams<'_, super::CudaBackend>,
    ) -> Result<()> {
        let inv_ctm = InverseCtm {
            u_dx: p.inv_ctm[0],
            u_dy: p.inv_ctm[1],
            v_dx: p.inv_ctm[2],
            v_dy: p.inv_ctm[3],
            tx: p.inv_ctm[4],
            ty: p.inv_ctm[5],
        };
        let bbox = BlitBbox {
            x0: p.bbox[0],
            y0: p.bbox[1],
            x1: p.bbox[2],
            y1: p.bbox[3],
        };
        // Trait passes layout as u32; kernel takes i32. validate() above
        // has already constrained the value to {0, 1}, so the cast is exact.
        // Keeping the explicit cast (rather than `as i32`) so a future
        // widening of valid layouts surfaces at this exact line.
        let layout_code =
            i32::try_from(p.src_layout).expect("validate() proved src_layout is 0 or 1, fits i32");

        ctx.launch_blit_image_async(
            p.src,
            (p.src_w, p.src_h),
            layout_code,
            p.dst,
            (p.dst_w, p.dst_h),
            bbox,
            &inv_ctm,
            p.page_h,
        )
        .map_err(be)
    }

    #[cfg(not(feature = "cache"))]
    fn record_blit_image_inner(
        _ctx: &Arc<GpuCtx>,
        _p: params::BlitParams<'_, super::CudaBackend>,
    ) -> Result<()> {
        Err(BackendError::new(StringError(
            "blit_image requires the gpu crate's `cache` feature".into(),
        )))
    }

    pub(super) fn record_aa_fill(
        &self,
        p: params::AaFillParams<'_, super::CudaBackend>,
    ) -> Result<()> {
        let eo = p.fill_rule != 0;
        // The trait API uses tile-local coordinates: callers compute
        // `x_min` / `y_min` into the segs they upload, so the kernel
        // origin is always (0, 0).  This matches `tile_fill` and
        // simplifies the trait surface; if a future renderer integration
        // needs a non-zero device-pixel offset, AaFillParams should grow
        // an explicit `origin` field rather than re-introducing implicit
        // x_min/y_min on this code path.
        let x_min = 0.0_f32;
        let y_min = 0.0_f32;
        self.ctx
            .launch_aa_fill_async(
                p.segs, p.n_segs, x_min, y_min, p.width, p.height, eo, p.coverage,
            )
            .map_err(be)
    }

    pub(super) fn record_icc_clut(
        &self,
        p: params::IccClutParams<'_, super::CudaBackend>,
    ) -> Result<()> {
        let clut_bytes = p.clut.len();
        let grid_n = grid_n_from_clut_len(clut_bytes).ok_or_else(|| {
            BackendError::new(StringError(format!(
                "ICC CLUT length {clut_bytes} is not a valid grid_n^4 * 3"
            )))
        })?;
        self.ctx
            .launch_icc_clut_async(p.cmyk, p.rgb, p.clut, grid_n, p.n_pixels)
            .map_err(be)
    }

    pub(super) fn record_tile_fill(
        &self,
        p: params::TileFillParams<'_, super::CudaBackend>,
    ) -> Result<()> {
        let eo = p.fill_rule != 0;
        let grid_w = p.width.div_ceil(crate::TILE_W);
        self.ctx
            .launch_tile_fill_async(
                p.records,
                p.tile_starts,
                p.tile_counts,
                grid_w,
                p.width,
                p.height,
                eo,
                p.coverage,
            )
            .map_err(be)
    }

    pub(super) fn record_composite(
        &self,
        p: params::CompositeParams<'_, super::CudaBackend>,
    ) -> Result<()> {
        self.ctx
            .launch_composite_async(p.src, p.dst, p.n_pixels)
            .map_err(be)
    }

    pub(super) fn record_apply_soft_mask(
        &self,
        p: params::SoftMaskParams<'_, super::CudaBackend>,
    ) -> Result<()> {
        self.ctx
            .launch_soft_mask_async(p.pixels, p.mask, p.n_pixels)
            .map_err(be)
    }
}

/// Given a CLUT byte length, recover `grid_n` such that
/// `len == grid_n^4 * 3`, or `None` if the length is invalid.
///
/// The CLUT layout is `(k * G³ + c * G² + m * G + y) * 3` bytes —
/// `grid_n^4` 3-byte RGB nodes.  Typical PDF profiles use `grid_n` = 17
/// or 33.
fn grid_n_from_clut_len(len: usize) -> Option<u32> {
    if !len.is_multiple_of(3) {
        return None;
    }
    let nodes = len / 3;
    // Integer 4th root by iteration: grid_n ≤ 255 in practice.
    for grid in 2u32..=255 {
        let g = grid as usize;
        let pow4 = g.checked_mul(g)?.checked_mul(g)?.checked_mul(g)?;
        if pow4 == nodes {
            return Some(grid);
        }
        if pow4 > nodes {
            return None;
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_n_round_trips() {
        assert_eq!(grid_n_from_clut_len(17 * 17 * 17 * 17 * 3), Some(17));
        assert_eq!(grid_n_from_clut_len(33 * 33 * 33 * 33 * 3), Some(33));
    }

    #[test]
    fn grid_n_rejects_non_multiple_of_3() {
        assert_eq!(grid_n_from_clut_len(83_521 * 3 + 1), None);
    }

    #[test]
    fn grid_n_rejects_non_4th_power() {
        assert_eq!(grid_n_from_clut_len(100 * 3), None);
    }
}
