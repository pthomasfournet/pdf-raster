//! CUDA backend implementing `GpuBackend`.
//!
//! Wraps the existing `GpuCtx` and per-kernel `lib_kernels::*` functions.
//! Per-page batching is recorded into a `PageRecorder`; see `page_recorder.rs`.

mod page_recorder;

use std::sync::Arc;

use cudarc::driver::{CudaSlice, PinnedHostSlice};

use crate::GpuCtx;
use crate::backend::{BackendError, GpuBackend, Result, VramBudget, params, reject_zero_size};

/// CUDA error adaptor.
///
/// `GpuCtx::init` returns `Box<dyn Error>` (not Send+Sync), so we wrap the
/// stringified message in a type that is `Send + Sync + Error`. Used only
/// inside [`be`]; non-driver-error callsites should use
/// [`BackendError::msg`] directly.
#[derive(Debug)]
struct StringError(String);

impl std::fmt::Display for StringError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for StringError {}

/// Convert any `Display` error to a `BackendError` via `StringError`.
pub(super) fn be(e: impl std::fmt::Display) -> BackendError {
    BackendError::new(StringError(e.to_string()))
}

/// A CUDA backend that wraps `GpuCtx`.
pub struct CudaBackend {
    ctx: Arc<GpuCtx>,
    recorder: page_recorder::PageRecorder,
}

impl CudaBackend {
    /// Initialise CUDA device 0 and compile the embedded kernels.
    ///
    /// # Errors
    /// Returns `BackendError` if no CUDA device is present or kernel load fails.
    pub fn new() -> Result<Self> {
        let ctx = Arc::new(GpuCtx::init().map_err(be)?);
        Ok(Self {
            recorder: page_recorder::PageRecorder::new(ctx.clone()),
            ctx,
        })
    }
}

impl GpuBackend for CudaBackend {
    type DeviceBuffer = CudaSlice<u8>;
    type HostBuffer = PinnedHostSlice<u8>;
    type PageFence = page_recorder::PageFence;

    fn alloc_device(&self, size: usize) -> Result<Self::DeviceBuffer> {
        reject_zero_size(size, "alloc_device")?;
        // cudarc's `alloc_zeros` always zero-fills.  We over-deliver
        // the trait contract here (which says `alloc_device` makes no
        // promise about contents) because cudarc doesn't expose a
        // cheaper unzeroed allocation path.  The contract is the
        // weaker one; callers that care must use `alloc_device_zeroed`.
        self.ctx.stream().alloc_zeros::<u8>(size).map_err(be)
    }

    fn alloc_device_zeroed(&self, size: usize) -> Result<Self::DeviceBuffer> {
        reject_zero_size(size, "alloc_device_zeroed")?;
        // cudarc syncs on the default stream; the buffer reads as zero
        // from the next host-visible op without needing wait_transfer.
        self.ctx.stream().alloc_zeros::<u8>(size).map_err(be)
    }

    fn device_buffer_len(&self, buf: &Self::DeviceBuffer) -> usize {
        buf.len()
    }

    fn free_device(&self, _buf: Self::DeviceBuffer) {
        // cudarc records a stream event in PinnedHostSlice's Drop so a
        // free during in-flight DMA is safe (the actual cuMemFree is
        // gated on the event signalling).  Nothing to do here.
    }

    fn alloc_host_pinned(&self, size: usize) -> Result<Self::HostBuffer> {
        reject_zero_size(size, "alloc_host_pinned")?;
        let cuda_ctx = self.ctx.stream().context();
        // Safety: the caller must initialise every byte before reading it back.
        // Mirrors the pattern in `cache::host_tier::HostTier::alloc_pinned`.
        unsafe { cuda_ctx.alloc_pinned::<u8>(size) }.map_err(be)
    }

    fn free_host_pinned(&self, _buf: Self::HostBuffer) {
        // cudarc frees the pinned allocation on Drop.
    }

    fn begin_page(&self) -> Result<()> {
        self.recorder.begin_page()
    }

    fn record_blit_image(&self, params: params::BlitParams<'_, Self>) -> Result<()> {
        self.recorder.record_blit_image(params)
    }

    fn record_aa_fill(&self, params: params::AaFillParams<'_, Self>) -> Result<()> {
        self.recorder.record_aa_fill(params)
    }

    fn record_icc_clut(&self, params: params::IccClutParams<'_, Self>) -> Result<()> {
        self.recorder.record_icc_clut(params)
    }

    fn record_tile_fill(&self, params: params::TileFillParams<'_, Self>) -> Result<()> {
        self.recorder.record_tile_fill(params)
    }

    fn record_composite(&self, params: params::CompositeParams<'_, Self>) -> Result<()> {
        self.recorder.record_composite(params)
    }

    fn record_apply_soft_mask(&self, params: params::SoftMaskParams<'_, Self>) -> Result<()> {
        self.recorder.record_apply_soft_mask(params)
    }

    fn submit_page(&self) -> Result<Self::PageFence> {
        self.recorder.submit_page()
    }

    fn wait_page(&self, fence: Self::PageFence) -> Result<()> {
        self.recorder.wait_page(fence)
    }

    fn upload_async(&self, _dst: &Self::DeviceBuffer, _src: &[u8]) -> Result<Self::PageFence> {
        // Stub: a dedicated copy-engine stream + event signalling will land
        // alongside the prefetcher integration. Until then, panic loudly so
        // accidental callers fail at the trait surface rather than silently
        // returning a bogus fence.
        unimplemented!("CudaBackend::upload_async is not implemented yet")
    }

    fn download_async<'a>(
        &self,
        _src: &'a Self::DeviceBuffer,
        _dst: &'a mut [u8],
    ) -> Result<crate::backend::DownloadHandle<'a, Self>> {
        // Stub: lands alongside upload_async.  The CUDA implementation
        // will issue cuMemcpyDtoHAsync directly into `dst` (no staging
        // buffer needed — host memory is naturally CPU-accessible) and
        // return a handle whose `finish` is a no-op event sync.
        unimplemented!("CudaBackend::download_async is not implemented yet")
    }

    fn wait_download(&self, _handle: crate::backend::DownloadHandle<'_, Self>) -> Result<()> {
        unimplemented!("CudaBackend::wait_download is not implemented yet")
    }

    fn submit_transfer(&self) -> Result<Self::PageFence> {
        // CUDA: the default stream serialises uploads/downloads with
        // compute already, so "submit transfer" is a no-op that just
        // records an event for callers that want a fence to wait on.
        page_recorder::PageRecorder::record_fence(&self.ctx)
    }

    fn wait_transfer(&self, fence: Self::PageFence) -> Result<()> {
        // Same primitive as wait_page on CUDA — both wait on a
        // CudaEvent recorded on the same stream.
        self.recorder.wait_page(fence)
    }

    fn detect_vram_budget(&self) -> Result<VramBudget> {
        let cuda_ctx = self.ctx.stream().context();
        let (free, total) = cuda_ctx.mem_get_info().map_err(be)?;
        // free and total are usize (bytes); `free * 3 / 4` keeps arithmetic in
        // integer space — no f64 cast, no precision loss, no sign ambiguity.
        let usable_bytes = (free as u64).saturating_mul(3) / 4;
        // VramBudget::new asserts usable <= total. cuMemGetInfo's free is always
        // <= total, and 3/4 of free is <= free, so the invariant holds.
        Ok(VramBudget::new(total as u64, usable_bytes))
    }
}
