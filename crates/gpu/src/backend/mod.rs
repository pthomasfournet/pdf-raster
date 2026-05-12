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
#[cfg(feature = "vulkan")]
pub mod vulkan;

use std::error::Error;
use std::fmt;

/// A backend-agnostic error type.
///
/// Variants exist for conditions that tests assert on (`matches!`) or
/// that benefit from typed log clarity. The long tail of errors goes
/// through [`BackendError::Other`] — Vulkan FFI failures, allocator
/// errors, init failures, numeric overflow, and free-form messages
/// constructed via [`BackendError::msg`].
///
/// `Display` is hand-rolled per variant; `Error::source` returns
/// `Some(&inner)` only for `Other` (typed variants are themselves the
/// source).
#[derive(Debug)]
pub enum BackendError {
    /// `alloc_*` was called with `size = 0`. `what` names the allocator
    /// (`"alloc_device"`, `"alloc_host_pinned"`, `"alloc_device_zeroed"`).
    ZeroSizeAlloc {
        /// Name of the rejecting allocator entry point, for diagnostic context.
        what: &'static str,
    },
    /// `vkCmdFillBuffer` requires a 4-byte-aligned size; `alloc_device_zeroed`
    /// (and other fill paths) surface this loudly rather than silently
    /// downgrading and leaving 1–3 trailing bytes non-zero.
    UnalignedFill {
        /// The offending size in bytes.
        size: u64,
        /// The backend's required alignment (always `4` for `vkCmdFillBuffer`).
        required_alignment: u64,
    },
    /// Recorder cap on descriptor sets per page was reached. `max` is the
    /// production limit (`MAX_DESC_SETS_PER_PAGE`).
    DescriptorPoolExhausted {
        /// Number of descriptor sets already allocated this page.
        allocated: u32,
        /// The backend's per-page cap.
        max: u32,
    },
    /// A `*Params::validate` invariant failed. `kind` names which
    /// invariant family (`"BlitInvariantViolation"`, …); `detail` is
    /// the human-readable "why".
    InvariantViolation {
        /// Stable name of the invariant family for `matches!` ergonomics.
        kind: &'static str,
        /// Human-readable description of the specific violation.
        detail: &'static str,
    },
    /// Catch-all for any other backend error: Vulkan FFI failures,
    /// allocator errors, init failures, numeric overflow, free-form
    /// `BackendError::msg(...)`. `source()` returns the inner error.
    Other(Box<dyn Error + Send + Sync + 'static>),
}

impl BackendError {
    /// Wrap an arbitrary error as a `BackendError::Other`.
    pub fn new<E: Error + Send + Sync + 'static>(e: E) -> Self {
        Self::Other(Box::new(e))
    }

    /// Build a `BackendError::Other` from a free-form message.
    ///
    /// Convenience for callsites that don't have an underlying [`Error`]
    /// to wrap and don't fit any typed variant. Routes through `Other`,
    /// so `source()` returns the carrier.
    #[must_use]
    pub fn msg(message: impl Into<String>) -> Self {
        Self::Other(Box::new(MsgError(message.into())))
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
        match self {
            Self::ZeroSizeAlloc { what } => {
                write!(f, "{what} called with size = 0; backends require size > 0")
            }
            Self::UnalignedFill {
                size,
                required_alignment,
            } => write!(
                f,
                "fill size {size} is not a multiple of {required_alignment}"
            ),
            Self::DescriptorPoolExhausted { allocated, max } => write!(
                f,
                "descriptor pool exhausted: {allocated} sets allocated this page (max {max})"
            ),
            Self::InvariantViolation { kind, detail } => write!(f, "{kind}: {detail}"),
            Self::Other(e) => fmt::Display::fmt(e, f),
        }
    }
}

impl Error for BackendError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Other(e) => Some(&**e),
            _ => None,
        }
    }
}

/// Convenience alias for `Result<T, BackendError>`.
pub type Result<T> = std::result::Result<T, BackendError>;

/// Reject zero-size allocation requests with a clear message.
///
/// Both Vulkan (`vkAllocateMemory` rejects 0) and CUDA (`cuMemAlloc(0)` returns
/// `CUDA_ERROR_INVALID_VALUE`) refuse zero-size allocations; pre-checking gives
/// a uniform, callable-side error rather than a driver-specific status.
pub(crate) const fn reject_zero_size(size: usize, what: &'static str) -> Result<()> {
    if size == 0 {
        return Err(BackendError::ZeroSizeAlloc { what });
    }
    Ok(())
}

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
///
/// # Stability — closed implementation set
///
/// The trait is `pub` for callers (`pdf_raster`, `pdf_interp`) but the
/// implementation set is closed to in-tree backends (`CudaBackend`,
/// `VulkanBackend`).  The associated `DownloadHandle` carries
/// `pub(crate)` fields and an internal `DownloadInner` trait; external
/// crates cannot construct one and therefore cannot satisfy
/// `download_async` / `wait_download`.  This is deliberate — the
/// trait surface co-evolves with the in-tree backends and there is no
/// API stability commitment for external implementors.
pub trait GpuBackend: Send + Sync {
    /// An opaque device-resident buffer handle.
    type DeviceBuffer: Send + Sync;
    /// An opaque host-pinned buffer handle.
    type HostBuffer: Send + Sync;
    /// A synchronisation primitive returned by `submit_page` /
    /// `submit_transfer` / `upload_async`.
    ///
    /// `Clone` is required so callers can stash a fence inside a long-
    /// lived structure (e.g. `DeviceImageCache` keeps one per cached
    /// image to gate `free_device` until the last in-flight DMA
    /// completes) while a separate render path also waits on it.
    /// CUDA wraps `Arc<cudaEvent_t>`; Vulkan wraps `Arc<VkFence>`.
    type PageFence: Send + Sync + Clone;

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

    /// Allocate `size` bytes of device memory and zero-initialise them.
    ///
    /// The buffer is **eventually zero**: backends that fill on a
    /// transfer queue (Vulkan) make the zero-fill async, so callers
    /// that read the buffer must `wait_page` (or `wait_transfer`) on
    /// the next submission before assuming the contents are zero.
    /// Backends that fill synchronously (CUDA `cuMemsetD8`) over-
    /// deliver this contract.
    ///
    /// `DevicePageBuffer` relies on this: every page allocates a fresh
    /// zeroed RGBA8 buffer that the blit kernel only writes touched
    /// pixels into; reading `(0,0,0,0)` from un-touched pixels is the
    /// "transparent" signal for the host-side composite.
    ///
    /// # Errors
    /// Returns `BackendError` if `size == 0`, or if the allocation /
    /// zero-fill fails.
    fn alloc_device_zeroed(&self, size: usize) -> Result<Self::DeviceBuffer>;

    /// Byte length of a device buffer previously returned by
    /// `alloc_device` / `alloc_device_zeroed`.
    ///
    /// Exposed so generic callers (e.g. `DeviceImageCache`) can size
    /// transfers without holding a parallel length field per entry.
    fn device_buffer_len(&self, buf: &Self::DeviceBuffer) -> usize;

    /// Free a device buffer previously returned by `alloc_device` or
    /// `alloc_device_zeroed`.
    ///
    /// **Deferred-free contract:** the caller must ensure no
    /// outstanding `PageFence` (from `upload_async`, `download_async`,
    /// `submit_transfer`, or `submit_page`) still references this
    /// buffer. CUDA's stream-recorded events make a same-stream free
    /// always safe; Vulkan has no equivalent and `vkDestroyBuffer`
    /// while a copy is in-flight is undefined behaviour. Calling
    /// `wait_page` (or `wait_transfer`) on every fence the buffer
    /// participated in before `free_device` is always safe; the
    /// backend may otherwise treat such a call as undefined behaviour
    /// or a panic.
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
    /// Same deferred-free contract as `free_device`: the caller must
    /// ensure no outstanding `PageFence` (from `upload_async`,
    /// `download_async`, `submit_transfer`, or `submit_page`) still
    /// references this buffer.
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

    /// Record one phase of the Blelloch exclusive scan kernel.
    ///
    /// The scan is built from three dispatches (selected by
    /// `params.phase`); callers must invoke all three in order
    /// (`PerWorkgroup` → `BlockSums` → `ScatterBlockSums`) for the
    /// scan to complete. All three share `params.data` /
    /// `params.block_sums` / `params.len_elems`, so the caller builds
    /// one `ScanParams` and updates only `phase` between dispatches.
    ///
    /// The shared shape (one trait method, phase enum in the params)
    /// avoids multiplying `record_*` trait methods 1-per-kernel-phase.
    ///
    /// # Errors
    /// Returns `BackendError` if the operation cannot be recorded or
    /// if `params.validate(self)` fails.
    fn record_scan(&self, params: params::ScanParams<'_, Self>) -> Result<()>;

    /// Record one phase of the parallel-Huffman JPEG decoder.
    ///
    /// Today the only phase is `Phase1IntraSync` — one thread per
    /// subsequence walks the entropy-coded stream and writes a
    /// `(p, n, c, z)` tuple into `s_info[seq_idx]`. Phase 2 and 4
    /// land in follow-up commits; Phase 3 is the Blelloch scan (see
    /// [`Self::record_scan`]).
    ///
    /// # Errors
    /// Returns `BackendError` if the operation cannot be recorded or
    /// if `params.validate(self)` fails.
    fn record_huffman(&self, params: params::HuffmanParams<'_, Self>) -> Result<()>;
    /// Record a zero-fill of `buf` into the current page's command list.
    ///
    /// Folds what [`Self::alloc_device_zeroed`] does as a separate one-shot
    /// submission into the page's own command buffer / stream: zero extra
    /// submits, no transfer-queue lock acquisition, no `vkQueueWaitIdle`
    /// in the alloc path.  Callers that have an active `begin_page` (e.g.
    /// the per-page composite buffer) should prefer this over
    /// `alloc_device_zeroed` to avoid serialising concurrent renderer
    /// threads on the Vulkan transfer pool.
    ///
    /// The fill becomes visible by the time the page's `wait_page`
    /// returns — same "eventually zero" contract as `alloc_device_zeroed`.
    ///
    /// # Errors
    /// Returns `BackendError` if the operation cannot be recorded.
    /// Vulkan's `vkCmdFillBuffer` requires the buffer size to be a multiple
    /// of 4; non-aligned sizes return [`BackendError::UnalignedFill`].
    fn record_zero_buffer(&self, buf: &Self::DeviceBuffer) -> Result<()>;
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
    /// **Submission timing.** Vulkan accumulates the copy on the
    /// transfer queue and does NOT submit until `submit_transfer` is
    /// called; the returned fence still becomes valid at that point.
    /// CUDA enqueues on the default stream, equivalent to immediate
    /// submission. Either way, callers wait on the fence rather than
    /// caring which model is in play.
    ///
    /// # Errors
    /// Returns `BackendError` if the upload cannot be enqueued, or if
    /// `src.len()` exceeds `dst`'s device capacity.
    fn upload_async(&self, dst: &Self::DeviceBuffer, src: &[u8]) -> Result<Self::PageFence>;

    /// Initiate an asynchronous device-to-host download; returns a
    /// handle that ties the user-visible `dst` buffer's lifetime to the
    /// in-flight DMA.
    ///
    /// On Vulkan the copy goes via a staging buffer on the
    /// `TransferContext`'s host-visible heap; the device→staging copy
    /// is on the GPU and the staging→`dst` memcpy happens lazily
    /// inside [`Self::wait_download`]. CUDA performs `cuMemcpyDtoHAsync`
    /// directly into `dst` and signals when the stream catches up.
    ///
    /// `src` must outlive the returned handle (the borrow checker
    /// enforces this through the `'a` lifetime on `DownloadHandle`).
    ///
    /// # Errors
    /// Returns `BackendError` if the download cannot be enqueued, if
    /// `dst.len()` exceeds `src`'s device length, or if the staging
    /// buffer cannot be allocated.
    fn download_async<'a>(
        &self,
        src: &'a Self::DeviceBuffer,
        dst: &'a mut [u8],
    ) -> Result<DownloadHandle<'a, Self>>
    where
        Self: Sized;

    /// Block until a `DownloadHandle` completes; finalises the
    /// staging→`dst` copy on backends that buffer the download.
    ///
    /// After this returns `Ok(())`, the `dst` buffer originally passed
    /// to `download_async` contains the GPU bytes.
    ///
    /// # Errors
    /// Returns `BackendError` if the underlying fence wait fails.
    fn wait_download(&self, handle: DownloadHandle<'_, Self>) -> Result<()>
    where
        Self: Sized;

    /// Submit any work accumulated by `upload_async`,
    /// `download_async`, or `alloc_device_zeroed` on the transfer
    /// queue/stream. Returns a fence whose signal indicates that all
    /// transfers issued since the last `submit_transfer` have
    /// completed.
    ///
    /// CUDA implementations may treat this as a no-op (default-stream
    /// ordering already serialises the work) and return a freshly
    /// recorded event. Vulkan implementations end and submit the
    /// transfer command buffer here.
    ///
    /// Callers don't need to invoke this between `upload_async` and a
    /// later `submit_page` that reads the same buffer — the per-page
    /// recorder inserts the right cross-queue barriers. It exists for
    /// the cache's bulk-promotion paths that need to flush a batch of
    /// uploads/downloads without immediately consuming the result on
    /// the compute queue.
    ///
    /// # Errors
    /// Returns `BackendError` if the submission fails or no work is
    /// pending (Vulkan can no-op-and-succeed in that case).
    fn submit_transfer(&self) -> Result<Self::PageFence>;

    /// Block the calling thread until a transfer fence (from
    /// `upload_async`, `submit_transfer`, or `alloc_device_zeroed`)
    /// signals.
    ///
    /// Distinct from `wait_page` only because Vulkan tracks the
    /// transfer and graphics/compute queues independently — the same
    /// `PageFence` type covers both, but the underlying fence may
    /// belong to either queue's pool. CUDA implementations may treat
    /// this as identical to `wait_page`.
    ///
    /// # Errors
    /// Returns `BackendError` if the fence wait fails.
    fn wait_transfer(&self, fence: Self::PageFence) -> Result<()>;

    /// Query the current VRAM budget from the driver.
    ///
    /// # Errors
    /// Returns `BackendError` if the driver query fails.
    fn detect_vram_budget(&self) -> Result<VramBudget>;
}

/// Handle returned by [`GpuBackend::download_async`] that owns the
/// borrow on the user-visible destination slice for the duration of
/// the in-flight DMA.
///
/// Drop-without-wait is safe but wasteful: the staging buffer is
/// returned to the pool and the bytes never reach `dst`. Implementations
/// must NOT panic on `Drop` even if the underlying fence has not
/// signalled — instead, block-and-discard inside `Drop` so the staging
/// resource always returns to the pool. (Callers who want the bytes
/// must invoke [`GpuBackend::wait_download`] explicitly.)
pub struct DownloadHandle<'a, B: GpuBackend + ?Sized> {
    /// Backend-specific completion state. Boxed so `DownloadHandle`'s
    /// size doesn't depend on the backend's internal completion type.
    /// Read by both `wait_download` (explicit completion) and
    /// `Drop::drop` (block-and-discard), so it's never dead code even
    /// when the CUDA `download_async` stub is the only impl in scope.
    pub(crate) inner: Box<dyn DownloadInner + Send + Sync + 'a>,
    /// `&'a mut [u8]` borrow witness.  Two roles: (a) keeps the
    /// destination slice borrowed for the handle's lifetime so the
    /// caller can't free or reuse it while the DMA is in flight, and
    /// (b) pins variance — `Box<dyn Trait + 'a>` is covariant in `'a`,
    /// but `&mut [u8]` is invariant, and we need invariance so the
    /// borrow checker enforces unique-write semantics on `dst`.
    pub(crate) _borrow: std::marker::PhantomData<&'a mut [u8]>,
    /// The fence the caller can stash to gate other operations.
    /// Cloned out by `wait_download`; until then the handle owns a
    /// reference for cancellation safety.
    pub(crate) fence: B::PageFence,
}

impl<B: GpuBackend + ?Sized> DownloadHandle<'_, B> {
    /// Borrow the fence so callers can stash it in a refcount-pinned
    /// cache entry without moving the handle. The fence's `Clone` impl
    /// (from the trait bound) makes this cheap.
    #[must_use]
    pub const fn fence(&self) -> &B::PageFence {
        &self.fence
    }
}

/// Block-and-discard on drop: if the caller never invoked
/// `wait_download`, complete the transfer so the staging buffer
/// returns to the pool.  Errors are intentionally ignored — we're on
/// a destructor path and panicking would mask the original error path.
///
/// The doc-comment on `DownloadHandle` documents this contract
/// (implementations must not panic on Drop, must release the staging
/// resource).  Encoding it at the type level here means future
/// backends that hold real staging buffers + fences inside their
/// `DownloadInner` get the contract for free — they only have to
/// implement `finish()` correctly.
impl<B: GpuBackend + ?Sized> Drop for DownloadHandle<'_, B> {
    fn drop(&mut self) {
        let _ = self.inner.finish();
    }
}

/// Backend-internal trait for the staging→dst memcpy a Vulkan
/// download handle performs at completion. CUDA implementations can
/// supply a no-op impl since `cuMemcpyDtoHAsync` writes directly into
/// the user buffer.
///
/// `pub(crate)` because callers should never touch this — it exists to
/// give `DownloadHandle::Drop` and `wait_download` a uniform shape
/// across backends without exposing the staging buffer type.
pub(crate) trait DownloadInner {
    /// Block until the underlying transfer signals, then perform any
    /// staging→dst memcpy the backend deferred.
    ///
    /// Called either by `wait_download` (caller path) or by `Drop`
    /// (block-and-discard path).  `wait_download` swaps in a no-op
    /// `DownloadInner` after a successful call so `Drop` doesn't
    /// re-enter — implementations therefore only need to be correct
    /// for a single invocation per logical handle.
    fn finish(&mut self) -> Result<()>;
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
