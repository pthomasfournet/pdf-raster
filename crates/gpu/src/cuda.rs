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
/// 3. `cuDevicePrimaryCtxRetain` — acquire (ref-count) the primary context.
/// 4. `cuCtxSetCurrent` — bind the primary context to the calling thread.
/// 5. `cuStreamCreate` — create a new stream within that context.
///
/// On failure at step 4 or 5, the retained primary context is released before
/// returning.
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
pub fn init_primary_ctx_and_stream(
    device_ordinal: i32,
) -> Result<CudaInit, CudaInitError> {
    // Step 1 — load the CUDA driver.  cuInit is idempotent: calling it on a
    // thread that already initialised CUDA is a no-op returning CUDA_SUCCESS.
    let r = unsafe { cuInit(0) };
    if r != 0 {
        return Err(CudaInitError { step: "cuInit", code: r });
    }

    // Step 2 — resolve the integer ordinal to a device handle.
    let mut device: i32 = 0;
    let r = unsafe { cuDeviceGet(&raw mut device, device_ordinal) };
    if r != 0 {
        return Err(CudaInitError { step: "cuDeviceGet", code: r });
    }

    // Step 3 — retain the primary context (ref-counted; safe to call multiple
    // times from multiple threads on the same device).
    let mut cu_ctx: CUcontext = ptr::null_mut();
    let r = unsafe { cuDevicePrimaryCtxRetain(&raw mut cu_ctx, device) };
    if r != 0 {
        return Err(CudaInitError { step: "cuDevicePrimaryCtxRetain", code: r });
    }
    assert!(
        !cu_ctx.is_null(),
        "cuDevicePrimaryCtxRetain succeeded but returned null context"
    );

    // Step 4 — bind the primary context to the calling thread.
    let r = unsafe { cuCtxSetCurrent(cu_ctx) };
    if r != 0 {
        let _ = unsafe { cuDevicePrimaryCtxRelease(device) };
        return Err(CudaInitError { step: "cuCtxSetCurrent", code: r });
    }

    // Step 5 — create a stream in the now-current context.
    let mut stream: CUstream = ptr::null_mut();
    let r = unsafe { cuStreamCreate(&raw mut stream, 0) };
    if r != 0 {
        let _ = unsafe { cuDevicePrimaryCtxRelease(device) };
        return Err(CudaInitError { step: "cuStreamCreate", code: r });
    }
    assert!(
        !stream.is_null(),
        "cuStreamCreate succeeded but returned null stream"
    );

    Ok(CudaInit { cu_ctx, device, stream })
}
