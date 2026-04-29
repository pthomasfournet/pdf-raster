//! Shared CUDA driver API bindings and primary-context init helper.
//!
//! Used by both `npp_rotate` (`gpu-deskew` feature) and `nvjpeg2k` (`nvjpeg2k`
//! feature).  Extracted to avoid duplicating the eight-symbol FFI block and the
//! five-step init sequence in each consumer.

use std::ffi::c_void;
use std::ptr;

// ── CUDA driver API (libcuda.so) ──────────────────────────────────────────────

/// Opaque CUDA stream handle (`CUstream` / `cudaStream_t` — same ABI on Linux).
pub type CUstream = *mut c_void;

/// Opaque CUDA context handle (`CUcontext`).
pub type CUcontext = *mut c_void;

#[cfg_attr(
    not(any(feature = "gpu-deskew", feature = "nvjpeg2k")),
    allow(dead_code)
)]
unsafe extern "C" {
    pub fn cuInit(flags: u32) -> i32;
    pub fn cuDeviceGet(device: *mut i32, ordinal: i32) -> i32;
    pub fn cuDevicePrimaryCtxRetain(ctx: *mut CUcontext, device: i32) -> i32;
    pub fn cuDevicePrimaryCtxRelease(device: i32) -> i32;
    pub fn cuCtxSetCurrent(ctx: CUcontext) -> i32;
    pub fn cuStreamCreate(stream: *mut CUstream, flags: u32) -> i32;
    pub fn cuStreamDestroy(stream: CUstream) -> i32;
    pub fn cuStreamSynchronize(stream: CUstream) -> i32;
}

// ── CUDA runtime API (libcudart.so) ──────────────────────────────────────────

#[cfg_attr(
    not(any(feature = "gpu-deskew", feature = "nvjpeg2k")),
    allow(dead_code)
)]
unsafe extern "C" {
    fn cudaMalloc(dev_ptr: *mut *mut c_void, size: usize) -> i32;
    fn cudaFree(dev_ptr: *mut c_void) -> i32;
}

// ── DeviceBuf ─────────────────────────────────────────────────────────────────

/// RAII wrapper for a device-side `cudaMalloc` allocation.
///
/// The pointer is always non-null after successful construction (enforced by
/// `alloc`).  Freed on drop; failures are logged via `log::warn!` since they
/// cannot be propagated from `Drop`.
#[cfg_attr(
    not(any(feature = "gpu-deskew", feature = "nvjpeg2k")),
    allow(dead_code)
)]
pub struct DeviceBuf {
    pub ptr: *mut c_void,
}

#[cfg_attr(
    not(any(feature = "gpu-deskew", feature = "nvjpeg2k")),
    allow(dead_code)
)]
impl DeviceBuf {
    /// Allocate `size` bytes of device memory on the current CUDA device.
    ///
    /// # Errors
    ///
    /// Returns the raw `cudaMalloc` error code (non-zero) on failure.
    /// Callers map this to their own error type.
    pub fn alloc(size: usize) -> Result<Self, i32> {
        let mut ptr: *mut c_void = ptr::null_mut();
        // SAFETY: cudaMalloc writes a valid device pointer (or null) to `ptr`;
        // `size` is the requested allocation size.
        let code = unsafe { cudaMalloc(&raw mut ptr, size) };
        if code != 0 {
            return Err(code);
        }
        assert!(
            !ptr.is_null(),
            "cudaMalloc succeeded but returned null pointer"
        );
        Ok(Self { ptr })
    }
}

impl Drop for DeviceBuf {
    fn drop(&mut self) {
        // SAFETY: ptr came from cudaMalloc; no other reference exists at drop time.
        let code = unsafe { cudaFree(self.ptr) };
        if code != 0 {
            log::warn!("gpu: cudaFree failed: code {code}");
        }
    }
}

// SAFETY: DeviceBuf is an exclusively-owned device allocation; callers ensure
// it is only dropped from the thread that owns the CUDA context.
unsafe impl Send for DeviceBuf {}

// ── Init helper ───────────────────────────────────────────────────────────────

/// CUDA driver error: which init step failed and the raw driver error code.
#[cfg_attr(
    not(any(feature = "gpu-deskew", feature = "nvjpeg2k")),
    allow(dead_code)
)]
#[derive(Debug)]
pub struct CudaInitError {
    /// Name of the CUDA driver call that returned non-zero.
    ///
    /// Used by `npp_rotate` (`gpu-deskew` feature) to build a human-readable
    /// error string; unused when only `nvjpeg2k` is compiled.
    #[cfg_attr(not(feature = "gpu-deskew"), allow(dead_code))]
    pub step: &'static str,
    /// Raw CUDA driver error code (non-zero).
    pub code: i32,
}

/// Successful result of [`init_primary_ctx_and_stream`].
#[cfg_attr(
    not(any(feature = "gpu-deskew", feature = "nvjpeg2k")),
    allow(dead_code)
)]
pub struct CudaInit {
    /// Retained primary CUDA context for `device`.
    pub cu_ctx: CUcontext,
    /// CUDA device ordinal (i32 handle as returned by `cuDeviceGet`).
    pub device: i32,
    /// CUDA stream created in `cu_ctx`.
    pub stream: CUstream,
}

/// Run the five-step CUDA primary-context init sequence for `device_ordinal`.
///
/// Steps, in order:
/// 1. `cuInit(0)` — load and initialise the CUDA driver (idempotent).
/// 2. `cuDeviceGet` — resolve the integer ordinal to a driver device handle.
///    Negative values or out-of-range ordinals return `CUDA_ERROR_INVALID_VALUE`.
/// 3. `cuDevicePrimaryCtxRetain` — acquire (ref-count) the primary context.
/// 4. `cuCtxSetCurrent` — bind the primary context to the calling thread.
/// 5. `cuStreamCreate` — create a new stream within that context.
///
/// On failure at step 3, nothing is cleaned up (no retain was taken).
/// On failure at step 4 or 5, the retained primary context is released and the
/// calling thread's context binding is cleared (`cuCtxSetCurrent(null)`) to
/// avoid leaving a dangling context on the thread.
///
/// # Errors
///
/// Returns [`CudaInitError`] with the failing step name and error code if any
/// CUDA driver call returns non-zero.
///
/// # Panics
///
/// Panics if `cuDevicePrimaryCtxRetain` or `cuStreamCreate` reports success
/// but returns a null handle — this indicates a CUDA driver bug and cannot be
/// triggered by valid input or malformed PDFs.
#[cfg_attr(
    not(any(feature = "gpu-deskew", feature = "nvjpeg2k")),
    allow(dead_code)
)]
pub fn init_primary_ctx_and_stream(device_ordinal: i32) -> Result<CudaInit, CudaInitError> {
    // Step 1 — load the CUDA driver.  cuInit is idempotent: calling it on a
    // thread that already initialised CUDA is a no-op returning CUDA_SUCCESS.
    let r = unsafe { cuInit(0) };
    if r != 0 {
        return Err(CudaInitError {
            step: "cuInit",
            code: r,
        });
    }

    // Step 2 — resolve the integer ordinal to a device handle.
    // Negative ordinals and out-of-range values return CUDA_ERROR_INVALID_VALUE (1).
    let mut device: i32 = 0;
    let r = unsafe { cuDeviceGet(&raw mut device, device_ordinal) };
    if r != 0 {
        return Err(CudaInitError {
            step: "cuDeviceGet",
            code: r,
        });
    }

    // Step 3 — retain the primary context (ref-counted; safe to call multiple
    // times from multiple threads on the same device).
    let mut cu_ctx: CUcontext = ptr::null_mut();
    let r = unsafe { cuDevicePrimaryCtxRetain(&raw mut cu_ctx, device) };
    if r != 0 {
        return Err(CudaInitError {
            step: "cuDevicePrimaryCtxRetain",
            code: r,
        });
    }
    assert!(
        !cu_ctx.is_null(),
        "cuDevicePrimaryCtxRetain succeeded but returned null context"
    );

    // Step 4 — bind the primary context to the calling thread.
    let r = unsafe { cuCtxSetCurrent(cu_ctx) };
    if r != 0 {
        // Release the retain before returning.  Also clear the thread's context
        // binding: cuCtxSetCurrent may have partially committed before failing,
        // and leaving the thread pointing at a released context causes downstream
        // CUDA_ERROR_INVALID_CONTEXT on subsequent calls from this thread.
        unsafe {
            let _ = cuDevicePrimaryCtxRelease(device);
            let _ = cuCtxSetCurrent(ptr::null_mut());
        }
        return Err(CudaInitError {
            step: "cuCtxSetCurrent",
            code: r,
        });
    }

    // Step 5 — create a stream in the now-current context.
    let mut stream: CUstream = ptr::null_mut();
    let r = unsafe { cuStreamCreate(&raw mut stream, 0) };
    if r != 0 {
        // Release retain and clear thread context binding (same rationale as above).
        unsafe {
            let _ = cuDevicePrimaryCtxRelease(device);
            let _ = cuCtxSetCurrent(ptr::null_mut());
        }
        return Err(CudaInitError {
            step: "cuStreamCreate",
            code: r,
        });
    }
    assert!(
        !stream.is_null(),
        "cuStreamCreate succeeded but returned null stream"
    );

    Ok(CudaInit {
        cu_ctx,
        device,
        stream,
    })
}
