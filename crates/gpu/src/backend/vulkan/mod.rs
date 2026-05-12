//! Vulkan compute backend implementing `GpuBackend`.
//!
//! Mirrors the CUDA backend's shape: instance → device → queue at init
//! time, then `begin_page → record_* → submit_page → wait_page` for each
//! rendered page.  All six kernel families dispatch through `record_*`
//! into a per-page command buffer; submission uses a timeline semaphore
//! so the host waits exactly once per page.
//!
//! Module split (all private; named here for orientation only):
//! - `device` — loader, physical-device pick, logical device + queue.
//! - `error` — `vk::Result → BackendError` adaptor.
//! - `memory` — slab sub-allocator, host-visible staging pool.
//! - `pipeline` — descriptor set layouts + pipeline cache (lazy SPIR-V compile).
//! - `recorder` — per-page command buffer + timeline-semaphore fence.
//! - `transfer` — `upload_async` helper (probes for dedicated transfer queue).

mod device;
mod error;
mod memory;
mod pipeline;
mod recorder;
mod transfer;

pub use memory::{DeviceBuffer, HostBuffer};
pub use recorder::PageFence;

use std::sync::Arc;

use crate::backend::{GpuBackend, Result, VramBudget, params, reject_zero_size};

/// Vulkan compute backend.
///
/// Wraps an Arc'd device context plus a slab allocator and pipeline cache.
/// The trait surface is identical to the CUDA backend; switching is a
/// one-line change at the renderer's `RasterSession` constructor.
pub struct VulkanBackend {
    // Field order = drop order: children that record into command buffers
    // / hold pool handles must drop before the device they were created on.
    transfer: transfer::TransferContext,
    recorder: recorder::PageRecorder,
    #[expect(
        dead_code,
        reason = "kept for explicit drop ordering; readers go via recorder's Arc clone"
    )]
    pipelines: Arc<pipeline::PipelineCache>,
    memory: memory::SlabAllocator,
    device: Arc<device::DeviceCtx>,
}

impl VulkanBackend {
    /// Initialise the Vulkan loader, instance, device, and kernel pipelines.
    ///
    /// # Errors
    /// Returns `BackendError` if no suitable Vulkan 1.3+ device is present,
    /// any required feature is missing, kernel SPIR-V fails to load, or any
    /// driver call returns a non-success status.
    pub fn new() -> Result<Self> {
        let device = device::init()?;
        let memory = memory::SlabAllocator::new(device.clone())?;
        let pipelines = pipeline::PipelineCache::new(device.clone())?;
        let recorder = recorder::PageRecorder::new(device.clone(), pipelines.clone())?;
        let transfer = transfer::TransferContext::new(device.clone())?;
        Ok(Self {
            transfer,
            recorder,
            pipelines,
            memory,
            device,
        })
    }

    /// Synchronously upload `src` into `dst[0..src.len()]`.
    ///
    /// Used by tests and short-running setup paths where blocking is OK.
    /// The async version (`upload_async`) is the trait method; it returns
    /// a fence the caller must wait on.
    ///
    /// # Errors
    /// Returns `BackendError` if `src.len()` exceeds `dst.size()` or if
    /// any underlying Vulkan call fails.
    pub fn upload_sync(&self, dst: &memory::DeviceBuffer, src: &[u8]) -> Result<()> {
        self.transfer.upload_sync(&self.memory, dst, src)
    }

    /// Synchronously download `src[0..dst.len()]` into `dst`.
    ///
    /// # Errors
    /// Returns `BackendError` if `dst.len()` exceeds `src.size()` or if
    /// any underlying Vulkan call fails.
    pub fn download_sync(&self, src: &memory::DeviceBuffer, dst: &mut [u8]) -> Result<()> {
        self.transfer.download_sync(&self.memory, src, dst)
    }
}

impl Drop for VulkanBackend {
    fn drop(&mut self) {
        // Wait for all in-flight work before any handle in any submodule
        // tears down; the per-module Drop impls assume nothing is using
        // their resources.  Failure here is logged and ignored — we're
        // already on the destruction path.
        // Safety: sole owner; called exactly once.
        unsafe {
            if let Err(e) = self.device.device.device_wait_idle() {
                log::warn!("vkDeviceWaitIdle failed during VulkanBackend::drop: {e:?}");
            }
        }
    }
}

impl GpuBackend for VulkanBackend {
    type DeviceBuffer = memory::DeviceBuffer;
    type HostBuffer = memory::HostBuffer;
    type PageFence = recorder::PageFence;

    fn alloc_device(&self, size: usize) -> Result<Self::DeviceBuffer> {
        reject_zero_size(size, "alloc_device")?;
        self.memory.alloc_device(size)
    }

    fn alloc_device_zeroed(&self, size: usize) -> Result<Self::DeviceBuffer> {
        reject_zero_size(size, "alloc_device_zeroed")?;
        let buf = self.memory.alloc_device(size)?;
        // vkCmdFillBuffer requires 4-byte-aligned size; every realistic
        // caller (RGBA8 pages, u32 indices) is naturally aligned.
        // Mis-aligned sizes fail loudly rather than silently downgrading
        // to a 33 MB host-vec staging path.
        //
        // Allocation-time zero-fill rides `run_one_shot`, which holds
        // both `TransferContext::cmd_pool` and `DeviceCtx::compute_queue`
        // for the full submit + `vkQueueWaitIdle` (~1–2 ms for a 4K RGBA8
        // page). Concurrent renderer threads serialise here.
        //
        // The cheaper path is [`GpuBackend::record_zero_buffer`], which
        // folds the fill into the page's own command buffer between
        // `begin_page` and `submit_page` — zero extra submits, no
        // transfer-queue lock. Callers that already hold an active page
        // recording should prefer it. This method exists for paths that
        // need the buffer zeroed *outside* a page (e.g., one-shot setup,
        // tests).
        self.transfer.fill_zero(&buf)?;
        Ok(buf)
    }

    fn device_buffer_len(&self, buf: &Self::DeviceBuffer) -> usize {
        // Vulkan stores sizes as u64 (vk::DeviceSize); the trait
        // returns usize.  pdf-raster targets 64-bit only, so the
        // conversion is total — failing loudly is correct because
        // a saturating fallback (e.g. usize::MAX) would silently
        // poison cache size accounting.
        usize::try_from(buf.size()).expect("DeviceBuffer size fits usize on 64-bit targets")
    }

    fn free_device(&self, buf: Self::DeviceBuffer) {
        self.memory.free_device(buf);
    }

    fn alloc_host_pinned(&self, size: usize) -> Result<Self::HostBuffer> {
        reject_zero_size(size, "alloc_host_pinned")?;
        self.memory.alloc_host(size)
    }

    fn free_host_pinned(&self, buf: Self::HostBuffer) {
        self.memory.free_host(buf);
    }

    fn begin_page(&self) -> Result<()> {
        self.recorder.begin_page()
    }

    fn record_blit_image(&self, params: params::BlitParams<'_, Self>) -> Result<()> {
        params.validate()?;
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

    fn record_scan(&self, _params: params::ScanParams<'_, Self>) -> Result<()> {
        // Kernel + recorder wiring lands with the JPEG decoder's
        // Blelloch-scan dispatcher. Trait method exists today so
        // every backend's surface stays in sync.
        Err(crate::backend::BackendError::msg(
            "VulkanBackend::record_scan: kernel not yet wired in",
        ))
    }

    fn record_zero_buffer(&self, buf: &Self::DeviceBuffer) -> Result<()> {
        self.recorder.record_zero_buffer(buf)
    }

    fn submit_page(&self) -> Result<Self::PageFence> {
        self.recorder.submit_page()
    }

    fn wait_page(&self, fence: Self::PageFence) -> Result<()> {
        self.recorder.wait_page(fence)
    }

    fn upload_async(&self, dst: &Self::DeviceBuffer, src: &[u8]) -> Result<Self::PageFence> {
        // Sync today; ROADMAP tracks the async transfer-queue follow-up.
        self.transfer.upload_sync(&self.memory, dst, src)?;
        Ok(PageFence::immediate())
    }

    fn download_async<'a>(
        &self,
        src: &'a Self::DeviceBuffer,
        dst: &'a mut [u8],
    ) -> Result<crate::backend::DownloadHandle<'a, Self>> {
        // Sync today; ROADMAP tracks the async transfer-queue follow-up.
        self.transfer.download_sync(&self.memory, src, dst)?;
        Ok(crate::backend::DownloadHandle {
            inner: Box::new(NoopDownloadInner),
            _borrow: std::marker::PhantomData,
            fence: PageFence::immediate(),
        })
    }

    fn wait_download(&self, mut handle: crate::backend::DownloadHandle<'_, Self>) -> Result<()> {
        // NoopDownloadInner::finish is idempotent (returns Ok(())), so
        // the post-finish Drop calling finish() again is a no-op.  When
        // the async impl lands and finish() is no longer idempotent,
        // disarm the handle here.
        handle.inner.finish()
    }

    fn submit_transfer(&self) -> Result<Self::PageFence> {
        // Nothing to flush while transfers are sync; `immediate` keeps
        // the trait contract (wait_transfer on it returns at once).
        // ROADMAP tracks the async transfer-queue follow-up.
        Ok(PageFence::immediate())
    }

    fn wait_transfer(&self, fence: Self::PageFence) -> Result<()> {
        // Transfers ride the same timeline-semaphore wait path as
        // page submissions today; future split-queue work may add a
        // second timeline.  Use the recorder's no-state-check variant:
        // `submit_transfer` doesn't transition the recorder out of
        // Idle, so going through `wait_page` would trip its state
        // assertion every time.
        self.recorder.wait_transfer_fence(fence)
    }

    fn detect_vram_budget(&self) -> Result<VramBudget> {
        let (used, budget) = device::query_memory_budget(&self.device)?;
        let usable = budget.saturating_sub(used).saturating_mul(3) / 4;
        let total = self.device.vram_total;
        // Spec invariant: usable <= total.  Both `total` (sum of
        // DEVICE_LOCAL heaps) and `budget` come from the same heap set,
        // so this holds; clamp anyway in case a driver reports an
        // inconsistent budget.
        Ok(VramBudget::new(total, usable.min(total)))
    }
}

/// No-op `DownloadInner` for the sync-download path — the
/// device→host copy already happened inside `download_sync`, so the
/// handle's `finish` has nothing to do.  Replaced by a real impl
/// (staging→dst memcpy under fence wait) when the async transfer
/// queue lands.
struct NoopDownloadInner;

impl crate::backend::DownloadInner for NoopDownloadInner {
    fn finish(&mut self) -> crate::backend::Result<()> {
        Ok(())
    }
}
