//! GPU backend abstraction layer.
//!
//! The `GpuBackend` trait is the single seam between the renderer and the
//! GPU implementation. Two backends exist: CUDA (default on NVIDIA) and
//! Vulkan (cross-vendor; opt-in via the `vulkan` feature).
//!
//! Per-page semantics: callers use `begin_page` → `record_*` → `submit_page`
//! → `wait_page` to batch a page's GPU work into a single submission, so the
//! host blocks once per page instead of once per kernel.
//!
//! Pointer-passing ABI: device buffers are passed by reference into
//! `record_*` calls; the backend resolves them to native pointers (CUDA) or
//! buffer-device-addresses (Vulkan) at record time.

pub mod params;

use std::error::Error;
use std::fmt;

/// A backend-agnostic error type that wraps any `Error + Send + Sync`.
#[derive(Debug)]
pub struct BackendError(Box<dyn Error + Send + Sync + 'static>);

impl BackendError {
    /// Wrap an arbitrary error as a `BackendError`.
    pub fn new<E: Error + Send + Sync + 'static>(e: E) -> Self {
        Self(Box::new(e))
    }
}

impl fmt::Display for BackendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}

impl Error for BackendError {}

/// Convenience alias for `Result<T, BackendError>`.
pub type Result<T> = std::result::Result<T, BackendError>;

/// Live VRAM budget snapshot returned by `GpuBackend::detect_vram_budget`.
#[derive(Debug, Clone, Copy)]
pub struct VramBudget {
    /// Total VRAM on the device in bytes.
    pub total_bytes: u64,
    /// VRAM available for new allocations, accounting for driver overhead.
    pub usable_bytes: u64,
}

/// Per-trait declaration. Concrete backends live in `backend::cuda` and
/// `backend::vulkan` (the latter under the `vulkan` feature flag).
pub trait GpuBackend: Send + Sync {
    /// An opaque device-resident buffer handle.
    type DeviceBuffer: Send + Sync;
    /// An opaque host-pinned buffer handle.
    type HostBuffer: Send + Sync;
    /// A synchronisation primitive returned by `submit_page` / `upload_async`.
    type PageFence: Send + Sync;

    /// Allocate `size` bytes of device memory.
    ///
    /// # Errors
    /// Returns `BackendError` if the device allocation fails (OOM or driver error).
    fn alloc_device(&self, size: usize) -> Result<Self::DeviceBuffer>;
    /// Free a device buffer previously returned by `alloc_device`.
    fn free_device(&self, buf: Self::DeviceBuffer);

    /// Allocate `size` bytes of host-pinned (DMA-accessible) memory.
    ///
    /// # Errors
    /// Returns `BackendError` if the pinned allocation fails (OOM or driver error).
    fn alloc_host_pinned(&self, size: usize) -> Result<Self::HostBuffer>;
    /// Free a host-pinned buffer previously returned by `alloc_host_pinned`.
    fn free_host_pinned(&self, buf: Self::HostBuffer);

    /// Begin accumulating GPU work for a new page.
    ///
    /// # Errors
    /// Returns `BackendError` if the backend cannot initialise the per-page
    /// command state (e.g. stream creation failure).
    fn begin_page(&self) -> Result<()>;
    /// Record an image blit operation into the current page's command list.
    ///
    /// # Errors
    /// Returns `BackendError` if the operation cannot be recorded.
    fn record_blit_image(&self, params: params::BlitParams<'_, Self>) -> Result<()>;
    /// Record an antialiased fill operation into the current page's command list.
    ///
    /// # Errors
    /// Returns `BackendError` if the operation cannot be recorded.
    fn record_aa_fill(&self, params: params::AaFillParams<'_, Self>) -> Result<()>;
    /// Record an ICC CLUT colour-transform into the current page's command list.
    ///
    /// # Errors
    /// Returns `BackendError` if the operation cannot be recorded.
    fn record_icc_clut(&self, params: params::IccClutParams<'_, Self>) -> Result<()>;
    /// Record a tile-parallel analytical fill into the current page's command list.
    ///
    /// # Errors
    /// Returns `BackendError` if the operation cannot be recorded.
    fn record_tile_fill(&self, params: params::TileFillParams<'_, Self>) -> Result<()>;
    /// Record a Porter-Duff source-over composite into the current page's command list.
    ///
    /// # Errors
    /// Returns `BackendError` if the operation cannot be recorded.
    fn record_composite(&self, params: params::CompositeParams<'_, Self>) -> Result<()>;
    /// Record a soft-mask application into the current page's command list.
    ///
    /// # Errors
    /// Returns `BackendError` if the operation cannot be recorded.
    fn record_apply_soft_mask(&self, params: params::SoftMaskParams<'_, Self>) -> Result<()>;
    /// Submit all recorded work for the current page; returns a fence to wait on.
    ///
    /// # Errors
    /// Returns `BackendError` if the submission fails.
    fn submit_page(&self) -> Result<Self::PageFence>;
    /// Block the calling thread until the submitted page work is complete.
    ///
    /// # Errors
    /// Returns `BackendError` if the fence wait fails or the GPU raised an error.
    fn wait_page(&self, fence: Self::PageFence) -> Result<()>;

    /// Initiate an asynchronous host-to-device upload; returns a fence.
    ///
    /// # Errors
    /// Returns `BackendError` if the upload cannot be enqueued.
    fn upload_async(&self, dst: &Self::DeviceBuffer, src: &[u8]) -> Result<Self::PageFence>;

    /// Query the current VRAM budget from the driver.
    ///
    /// # Errors
    /// Returns `BackendError` if the driver query fails.
    fn detect_vram_budget(&self) -> Result<VramBudget>;
}
