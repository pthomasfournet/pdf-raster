//! GPU backend abstraction layer.
//!
//! The `GpuBackend` trait is the single seam between the renderer and any
//! concrete GPU implementation. Implementations are added as sibling modules
//! (`backend::cuda`, `backend::vulkan`).
//!
//! Per-page semantics: callers use `begin_page` → `record_*` → `submit_page`
//! → `wait_page` to batch a page's GPU work into a single submission, so the
//! host blocks once per page instead of once per kernel.
//!
//! Pointer-passing ABI: device buffers are passed by reference into
//! `record_*` calls; the backend resolves them to native pointers (CUDA) or
//! buffer-device-addresses (Vulkan) at record time.

pub mod cuda;
pub mod params;

use std::error::Error;
use std::fmt;

/// A backend-agnostic error type that wraps any `Error + Send + Sync`.
///
/// The inner `Box<dyn Error + Send + Sync>` carries the original backend-specific
/// error. [`Display`](fmt::Display) delegates to the inner error directly;
/// [`Error::source`] returns `None` because the inner message is already part of
/// the `Display` output — exposing it via `source` would cause `{:#}`
/// alternate-display to print the message twice. Mirrors the `GpuDecodeError`
/// pattern in `crate::traits`.
#[derive(Debug)]
pub struct BackendError(Box<dyn Error + Send + Sync + 'static>);

impl BackendError {
    /// Wrap an arbitrary error as a `BackendError`.
    pub fn new<E: Error + Send + Sync + 'static>(e: E) -> Self {
        Self(Box::new(e))
    }

    /// Build a `BackendError` from a free-form message.
    ///
    /// Convenience for callsites that don't have an underlying [`Error`]
    /// to wrap — typically backend invariants ('feature X not enabled',
    /// 'malformed input length'). `Display` for the resulting error
    /// prints the message verbatim.
    #[must_use]
    pub fn msg(message: impl Into<String>) -> Self {
        Self(Box::new(MsgError(message.into())))
    }
}

/// Internal carrier used by [`BackendError::msg`] so the inner type
/// satisfies `Error + Send + Sync + 'static`.
#[derive(Debug)]
struct MsgError(String);

impl fmt::Display for MsgError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for MsgError {}

impl fmt::Display for BackendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Delegate directly — no prefix — so the inner message is the full message.
        fmt::Display::fmt(&self.0, f)
    }
}

impl Error for BackendError {}

/// Convenience alias for `Result<T, BackendError>`.
pub type Result<T> = std::result::Result<T, BackendError>;

/// Reject zero-size allocation requests with a clear message.
///
/// Both Vulkan (`vkAllocateMemory` rejects 0) and CUDA (`cuMemAlloc(0)` returns
/// `CUDA_ERROR_INVALID_VALUE`) refuse zero-size allocations; pre-checking gives
/// a uniform, callable-side error rather than a driver-specific status.
pub(crate) fn reject_zero_size(size: usize, what: &'static str) -> Result<()> {
    if size == 0 {
        return Err(BackendError::new(ZeroSizeAlloc(what)));
    }
    Ok(())
}

#[derive(Debug)]
struct ZeroSizeAlloc(&'static str);

impl fmt::Display for ZeroSizeAlloc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} called with size = 0; backends require size > 0",
            self.0
        )
    }
}

impl Error for ZeroSizeAlloc {}

/// Live VRAM budget snapshot returned by `GpuBackend::detect_vram_budget`.
///
/// Construct via [`VramBudget::new`] to enforce the `usable_bytes <= total_bytes`
/// invariant; the public fields are kept for convenient pattern-matching but
/// hand-construction skips the assertion.
#[derive(Debug, Clone, Copy)]
pub struct VramBudget {
    /// Total VRAM on the device in bytes.
    pub total_bytes: u64,
    /// VRAM that allocations may safely consume in this process.
    ///
    /// Backends may apply a safety margin (e.g., a fraction of `free` reported
    /// by the driver) to account for fragmentation and other processes'
    /// allocations. Always `<= total_bytes`.
    pub usable_bytes: u64,
}

impl VramBudget {
    /// Construct a `VramBudget`, asserting `usable <= total`.
    ///
    /// # Panics
    /// Panics if `usable > total` — backends must apply their safety margin
    /// before calling this constructor, not let the caller derive a value that
    /// could exceed total VRAM.
    #[must_use]
    pub const fn new(total_bytes: u64, usable_bytes: u64) -> Self {
        assert!(
            usable_bytes <= total_bytes,
            "VramBudget invariant violated: usable_bytes must not exceed total_bytes"
        );
        Self {
            total_bytes,
            usable_bytes,
        }
    }
}

/// Abstraction over a concrete GPU implementation.
///
/// # Per-page state machine
///
/// `record_*` methods accumulate work into a per-backend command list. Callers
/// must follow the order:
///
/// ```text
/// begin_page() → [record_*()…] → submit_page() → wait_page(fence)
/// ```
///
/// Calling `record_*` outside a `begin_page` / `submit_page` pair, or calling
/// `submit_page` twice without an intervening `begin_page`, is implementation-
/// defined behaviour: backends may panic, return an error, or silently produce
/// incorrect results. Single-threaded usage is the supported pattern; cross-
/// thread interleaving requires external synchronisation even though the trait
/// is `Send + Sync`.
pub trait GpuBackend: Send + Sync {
    /// An opaque device-resident buffer handle.
    type DeviceBuffer: Send + Sync;
    /// An opaque host-pinned buffer handle.
    type HostBuffer: Send + Sync;
    /// A synchronisation primitive returned by `submit_page` / `upload_async`.
    type PageFence: Send + Sync;

    /// Allocate `size` bytes of device memory.
    ///
    /// `size` must be greater than zero; passing `0` returns a `BackendError`
    /// because both Vulkan (`vkAllocateMemory` rejects 0) and CUDA
    /// (`cuMemAlloc(0)` returns `CUDA_ERROR_INVALID_VALUE`) fail this call.
    /// Implementations should pre-check rather than fall through to the driver.
    ///
    /// # Errors
    /// Returns `BackendError` if `size == 0`, or if the device allocation fails
    /// (OOM or driver error).
    fn alloc_device(&self, size: usize) -> Result<Self::DeviceBuffer>;
    /// Free a device buffer previously returned by `alloc_device`.
    ///
    /// The caller must ensure no outstanding `PageFence` still references this
    /// buffer. Calling `wait_page` on every fence the buffer participated in
    /// before `free_device` is always safe; the backend may otherwise treat
    /// such a call as undefined behaviour or a panic.
    fn free_device(&self, buf: Self::DeviceBuffer);

    /// Allocate `size` bytes of host-pinned (DMA-accessible) memory.
    ///
    /// `size` must be greater than zero; the same zero-size constraints as
    /// `alloc_device` apply.
    ///
    /// # Errors
    /// Returns `BackendError` if `size == 0`, or if the pinned allocation fails
    /// (OOM or driver error).
    fn alloc_host_pinned(&self, size: usize) -> Result<Self::HostBuffer>;
    /// Free a host-pinned buffer previously returned by `alloc_host_pinned`.
    ///
    /// The caller must ensure no outstanding `PageFence` still references this
    /// buffer. Calling `wait_page` on every fence the buffer participated in
    /// before `free_host_pinned` is always safe; the backend may otherwise treat
    /// such a call as undefined behaviour or a panic.
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
    /// `src.len()` must not exceed the device-side capacity of `dst`.
    /// Implementations should reject the call rather than truncate.
    ///
    /// # Errors
    /// Returns `BackendError` if the upload cannot be enqueued, or if
    /// `src.len()` exceeds `dst`'s device capacity.
    fn upload_async(&self, dst: &Self::DeviceBuffer, src: &[u8]) -> Result<Self::PageFence>;

    /// Query the current VRAM budget from the driver.
    ///
    /// # Errors
    /// Returns `BackendError` if the driver query fails.
    fn detect_vram_budget(&self) -> Result<VramBudget>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vram_budget_new_accepts_usable_le_total() {
        let b = VramBudget::new(100, 75);
        assert_eq!(b.total_bytes, 100);
        assert_eq!(b.usable_bytes, 75);
    }

    #[test]
    fn vram_budget_new_accepts_equal() {
        let b = VramBudget::new(100, 100);
        assert_eq!(b.total_bytes, 100);
        assert_eq!(b.usable_bytes, 100);
    }

    #[test]
    fn vram_budget_new_accepts_zero() {
        // Degenerate but valid: a CPU-only backend may report no VRAM at all.
        let b = VramBudget::new(0, 0);
        assert_eq!(b.total_bytes, 0);
        assert_eq!(b.usable_bytes, 0);
    }

    #[test]
    #[should_panic(expected = "VramBudget invariant violated")]
    fn vram_budget_new_panics_when_usable_exceeds_total() {
        let _ = VramBudget::new(100, 101);
    }

    #[test]
    fn reject_zero_size_passes_nonzero() {
        assert!(reject_zero_size(1, "test").is_ok());
        assert!(reject_zero_size(usize::MAX, "test").is_ok());
    }

    #[test]
    fn reject_zero_size_returns_error_for_zero() {
        let err = reject_zero_size(0, "alloc_device").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("alloc_device"), "missing context: {msg}");
        assert!(msg.contains("size = 0"), "missing diagnostic: {msg}");
    }

    #[test]
    fn backend_error_msg_displays_verbatim() {
        let err = BackendError::msg("widget broke");
        assert_eq!(err.to_string(), "widget broke");
    }

    #[test]
    fn backend_error_msg_accepts_string_and_str() {
        let _from_str = BackendError::msg("static");
        let _from_string = BackendError::msg(String::from("owned"));
        let _from_format = BackendError::msg(format!("formatted {}", 42));
    }
}
