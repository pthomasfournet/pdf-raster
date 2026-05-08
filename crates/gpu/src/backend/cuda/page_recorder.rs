//! CUDA page recorder — buffers `record_*` calls until `submit_page`.
//!
//! For task 1.8 this is a thin stub that accepts calls but does nothing.
//! Task 1.9 swaps the body to call through to the existing per-kernel
//! functions and remove per-kernel `synchronize` calls.

use std::sync::Arc;

use crate::GpuCtx;
use crate::backend::{Result, params};

pub(super) struct PageRecorder {
    #[expect(
        dead_code,
        reason = "used in task 1.9 when record_* methods call lib_kernels"
    )]
    pub(super) ctx: Arc<GpuCtx>,
}

/// A synchronisation token returned by [`PageRecorder::submit_page`].
///
/// For task 1.8 this is a zero-cost sentinel; task 1.9 will replace it
/// with a real stream-event handle.
#[derive(Debug)]
pub struct PageFence;

impl PageRecorder {
    pub(super) const fn new(ctx: Arc<GpuCtx>) -> Self {
        Self { ctx }
    }

    // begin_page / submit_page / wait_page are stubs whose signatures must
    // match the GpuBackend trait.  The `unused_self`, `unnecessary_wraps`, and
    // `missing_const_for_fn` lints fire because the body is trivial; suppress
    // them here until task 1.9 fills in the real impl.
    #[expect(
        clippy::unused_self,
        clippy::unnecessary_wraps,
        clippy::missing_const_for_fn,
        reason = "stub matching GpuBackend signature; real impl in task 1.9"
    )]
    pub(super) fn begin_page(&self) -> Result<()> {
        Ok(())
    }

    #[expect(
        clippy::unused_self,
        clippy::unnecessary_wraps,
        clippy::missing_const_for_fn,
        reason = "stub matching GpuBackend signature; real impl in task 1.9"
    )]
    pub(super) fn submit_page(&self) -> Result<PageFence> {
        Ok(PageFence)
    }

    #[expect(
        clippy::unused_self,
        clippy::unnecessary_wraps,
        clippy::missing_const_for_fn,
        reason = "stub matching GpuBackend signature; real impl in task 1.9"
    )]
    pub(super) fn wait_page(&self, _fence: PageFence) -> Result<()> {
        Ok(())
    }

    pub(super) fn record_blit_image(
        &self,
        _p: params::BlitParams<'_, super::CudaBackend>,
    ) -> Result<()> {
        // STUB: task 1.9 wires this to lib_kernels::blit
        unimplemented!("record_blit_image stubbed; real wiring in task 1.9")
    }

    pub(super) fn record_aa_fill(
        &self,
        _p: params::AaFillParams<'_, super::CudaBackend>,
    ) -> Result<()> {
        unimplemented!("record_aa_fill stubbed; real wiring in task 1.9")
    }

    pub(super) fn record_icc_clut(
        &self,
        _p: params::IccClutParams<'_, super::CudaBackend>,
    ) -> Result<()> {
        unimplemented!("record_icc_clut stubbed; real wiring in task 1.9")
    }

    pub(super) fn record_tile_fill(
        &self,
        _p: params::TileFillParams<'_, super::CudaBackend>,
    ) -> Result<()> {
        unimplemented!("record_tile_fill stubbed; real wiring in task 1.9")
    }

    pub(super) fn record_composite(
        &self,
        _p: params::CompositeParams<'_, super::CudaBackend>,
    ) -> Result<()> {
        unimplemented!("record_composite stubbed; real wiring in task 1.9")
    }

    pub(super) fn record_apply_soft_mask(
        &self,
        _p: params::SoftMaskParams<'_, super::CudaBackend>,
    ) -> Result<()> {
        unimplemented!("record_apply_soft_mask stubbed; real wiring in task 1.9")
    }
}
