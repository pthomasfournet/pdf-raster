//! Vulkan compute backend implementing `GpuBackend`.
//!
//! Mirrors the CUDA backend's shape: instance → device → queue at init
//! time, then `begin_page → record_* → submit_page → wait_page` for each
//! rendered page.  All six kernel families dispatch through `record_*`
//! into a per-page command buffer; submission uses a timeline semaphore
//! so the host waits exactly once per page.
//!
//! Module split:
//! - [`device`] — loader, physical-device pick, logical device + queue.
//! - [`error`] — `vk::Result → BackendError` adaptor.
//! - [`memory`] — slab sub-allocator, host-visible staging pool.
//! - [`pipeline`] — descriptor set layouts + pipeline cache (lazy SPIR-V compile).
//! - [`recorder`] — per-page command buffer + timeline-semaphore fence.
//! - [`transfer`] — `upload_async` helper (probes for dedicated transfer queue).

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
    // Field declaration order matters: Rust drops struct fields in
    // declaration order (top to bottom).  We need recorder + transfer
    // dropped FIRST so their command pools / semaphores get destroyed
    // while the device is still alive; then pipelines (which destroys
    // VkPipeline + layouts); then memory (which calls vkDestroyBuffer
    // on the device); then finally the device.  The Arc<DeviceCtx>
    // ensures all the children keep the device alive even if we got
    // the drop order wrong, but explicit order is clearer.
    transfer: transfer::TransferContext,
    recorder: recorder::PageRecorder,
    /// Held as `Arc` so the recorder keeps a clone — pipelines must
    /// outlive every command buffer that bound them.
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
        // Order here mirrors the struct's drop order (declaration order):
        // transfer → recorder → pipelines → memory → device.  Both
        // initialisation and tear-down read top-down for clarity.
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

    fn submit_page(&self) -> Result<Self::PageFence> {
        self.recorder.submit_page()
    }

    fn wait_page(&self, fence: Self::PageFence) -> Result<()> {
        self.recorder.wait_page(fence)
    }

    fn upload_async(&self, dst: &Self::DeviceBuffer, src: &[u8]) -> Result<Self::PageFence> {
        // Today's path is sync (vkQueueWaitIdle inside `upload_sync`);
        // we hand back an already-signalled fence so callers writing
        // `upload_async(...).and_then(|f| wait_page(f))` keep working
        // unchanged once the dedicated transfer queue lands.
        self.transfer.upload_sync(&self.memory, dst, src)?;
        Ok(PageFence::immediate())
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
