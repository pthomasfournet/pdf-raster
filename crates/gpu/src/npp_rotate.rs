//! GPU rotation via CUDA NPP `nppiRotate_8u_C1R_Ctx`.
//!
//! Rotates a single-channel 8-bit grayscale image by a given angle using
//! NVIDIA NPP hardware bilinear interpolation.  Expected throughput on
//! RTX 5070: ~0.3–0.5 ms for a 2550×3300 image (texture fill rate limited).
//!
//! # Angle convention
//!
//! [`rotate_gray8`] takes a **clockwise-positive** angle in degrees, matching
//! the deskew pipeline convention.  NPP also uses clockwise-positive in
//! image-space (Y axis points down), so the angle is passed through unchanged.
//! The post-rotation shift is computed to keep the image centred on its
//! original centre point.
//!
//! # Thread safety
//!
//! [`NppRotator`] is `Send` but not `Sync`.  Each thread gets its own
//! instance via a `thread_local!` inside [`rotate_gray8`] — lazy,
//! initialise-once, never retry on failure.

#![cfg(feature = "gpu-deskew")]

use std::cell::RefCell;
use std::ffi::c_void;
use std::ptr;

// ── CUDA driver API (libcuda.so) ──────────────────────────────────────────────

/// Opaque CUDA stream / context handle.
type CUstream = *mut c_void;

unsafe extern "C" {
    fn cuInit(flags: u32) -> i32;
    fn cuDeviceGet(device: *mut i32, ordinal: i32) -> i32;
    fn cuDevicePrimaryCtxRetain(ctx: *mut *mut c_void, device: i32) -> i32;
    fn cuDevicePrimaryCtxRelease(device: i32) -> i32;
    fn cuCtxSetCurrent(ctx: *mut c_void) -> i32;
    fn cuStreamCreate(stream: *mut CUstream, flags: u32) -> i32;
    fn cuStreamDestroy(stream: CUstream) -> i32;
    fn cuStreamSynchronize(stream: CUstream) -> i32;
}

// ── CUDA runtime API (libcudart.so) ───────────────────────────────────────────

const CUDA_MEMCPY_H2D: i32 = 1;
const CUDA_MEMCPY_D2H: i32 = 2;

unsafe extern "C" {
    fn cudaMalloc(dev_ptr: *mut *mut c_void, size: usize) -> i32;
    fn cudaFree(dev_ptr: *mut c_void) -> i32;
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: i32) -> i32;
}

// ── NPP core (libnppc.so) ─────────────────────────────────────────────────────

/// Opaque `cudaStream_t` handle as used by the CUDA runtime API.
type CudaStreamT = *mut c_void;

/// NPP status code: 0 = success, negative = error, positive = warning.
type NppStatus = i32;
const NPP_SUCCESS: i32 = 0;

/// NPP stream context; populated by [`nppGetStreamContext`] after
/// [`nppSetStream`] points NPP at the correct stream.
#[repr(C)]
#[derive(Default)]
struct NppStreamContext {
    h_stream: CudaStreamT,
    cuda_device_id: i32,
    multi_processor_count: i32,
    max_threads_per_multi_processor: i32,
    max_threads_per_block: i32,
    shared_mem_per_block: usize,
    compute_capability_major: i32,
    compute_capability_minor: i32,
    stream_flags: u32,
    reserved0: i32,
}

unsafe extern "C" {
    fn nppSetStream(stream: CudaStreamT) -> NppStatus;
    fn nppGetStreamContext(ctx: *mut NppStreamContext) -> NppStatus;
}

// ── NPP geometry (libnppig.so) ────────────────────────────────────────────────

/// 2-D integer size (width × height).
#[repr(C)]
struct NppiSize {
    width: i32,
    height: i32,
}

/// 2-D integer rectangle: top-left corner (x, y) and dimensions.
#[repr(C)]
struct NppiRect {
    x: i32,
    y: i32,
    width: i32,
    height: i32,
}

/// Bilinear interpolation mode (`NPPI_INTER_LINEAR = 2`).
const NPPI_INTER_LINEAR: i32 = 2;

unsafe extern "C" {
    #[allow(clippy::too_many_arguments)] // mirrors the C API exactly
    fn nppiRotate_8u_C1R_Ctx(
        p_src: *const u8,
        src_size: NppiSize,
        src_step: i32,
        src_roi: NppiRect,
        p_dst: *mut u8,
        dst_step: i32,
        dst_roi: NppiRect,
        angle: f64,
        shift_x: f64,
        shift_y: f64,
        interpolation: i32,
        npp_ctx: NppStreamContext,
    ) -> NppStatus;
}

// ── DeviceBuf RAII ────────────────────────────────────────────────────────────

/// RAII wrapper for a device-side `cudaMalloc` allocation.
struct DeviceBuf {
    ptr: *mut c_void,
}

impl DeviceBuf {
    /// Allocate `size` bytes on the current CUDA device.
    ///
    /// # Errors
    ///
    /// Returns [`NppRotateError`] if `cudaMalloc` fails.
    ///
    /// # Panics
    ///
    /// Panics if `cudaMalloc` reports success but returns a null pointer
    /// (should never happen in practice).
    fn alloc(size: usize) -> Result<Self> {
        let mut ptr: *mut c_void = ptr::null_mut();
        // SAFETY: cudaMalloc writes a valid device pointer (or null) to `ptr`.
        let code = unsafe { cudaMalloc(ptr::addr_of_mut!(ptr), size) };
        if code != 0 {
            return Err(NppRotateError(format!("cudaMalloc({size}): code {code}")));
        }
        assert!(!ptr.is_null(), "cudaMalloc succeeded but returned null");
        Ok(Self { ptr })
    }
}

impl Drop for DeviceBuf {
    fn drop(&mut self) {
        // SAFETY: ptr came from cudaMalloc; no other reference exists at drop time.
        let _ = unsafe { cudaFree(self.ptr) };
    }
}

// ── Error type ────────────────────────────────────────────────────────────────

/// Error returned by GPU rotation.
#[derive(Debug)]
pub struct NppRotateError(pub(crate) String);

impl std::fmt::Display for NppRotateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for NppRotateError {}

type Result<T> = std::result::Result<T, NppRotateError>;

// ── NppRotator ────────────────────────────────────────────────────────────────

/// Owns a CUDA primary context and stream for NPP rotation calls.
///
/// Reuse across pages — context and stream creation are expensive; the
/// per-rotate cost is two `cudaMalloc` + two `cudaMemcpy` + the NPP kernel.
pub struct NppRotator {
    cu_ctx: *mut c_void,
    device: i32,
    stream: CUstream,
}

// SAFETY: NppRotator is accessed exclusively through the per-thread
// `thread_local!` slot in `rotate_gray8`; it is never shared across threads.
unsafe impl Send for NppRotator {}

impl NppRotator {
    /// Initialise CUDA device 0 and create a stream for NPP use.
    ///
    /// # Errors
    ///
    /// Returns [`NppRotateError`] if any CUDA driver call fails (e.g. no GPU
    /// is present, or the driver is not loaded).
    pub fn new() -> Result<Self> {
        macro_rules! cu {
            ($call:expr, $label:literal) => {{
                let r = unsafe { $call };
                if r != 0 {
                    return Err(NppRotateError(format!("{}: code {}", $label, r)));
                }
            }};
        }

        cu!(cuInit(0), "cuInit");

        let mut device: i32 = 0;
        cu!(cuDeviceGet(ptr::addr_of_mut!(device), 0), "cuDeviceGet");

        let mut cu_ctx: *mut c_void = ptr::null_mut();
        cu!(
            cuDevicePrimaryCtxRetain(ptr::addr_of_mut!(cu_ctx), device),
            "cuDevicePrimaryCtxRetain"
        );

        let ctx_code = unsafe { cuCtxSetCurrent(cu_ctx) };
        if ctx_code != 0 {
            let _ = unsafe { cuDevicePrimaryCtxRelease(device) };
            return Err(NppRotateError(format!("cuCtxSetCurrent: code {ctx_code}")));
        }

        let mut stream: CUstream = ptr::null_mut();
        let stream_code = unsafe { cuStreamCreate(ptr::addr_of_mut!(stream), 0) };
        if stream_code != 0 {
            let _ = unsafe { cuDevicePrimaryCtxRelease(device) };
            return Err(NppRotateError(format!(
                "cuStreamCreate: code {stream_code}"
            )));
        }

        Ok(Self {
            cu_ctx,
            device,
            stream,
        })
    }

    /// Rotate a single-channel 8-bit grayscale image by `angle_deg` degrees
    /// (clockwise positive) via NPP bilinear.
    ///
    /// - `src`: pixel data, `src_stride * height` bytes (may have row padding).
    /// - `src_stride`: bytes per row in `src` (≥ `width`).
    /// - `width`, `height`: image dimensions in pixels.
    ///
    /// Returns tightly packed output pixels (`width * height` bytes, no padding).
    ///
    /// # Errors
    ///
    /// Returns [`NppRotateError`] if any CUDA or NPP call fails.
    pub fn rotate(
        &self,
        src: &[u8],
        src_stride: usize,
        width: u32,
        height: u32,
        angle_deg: f32,
    ) -> Result<Vec<u8>> {
        let w = width as usize;
        let h = height as usize;
        let dst_stride = w; // tightly packed output
        let dst_len = dst_stride * h;

        let (n_angle, shift_x, shift_y) = centre_pivot_shift(w, h, angle_deg);

        // Restore CUDA context (rayon workers may interleave).
        let ctx_code = unsafe { cuCtxSetCurrent(self.cu_ctx) };
        if ctx_code != 0 {
            return Err(NppRotateError(format!(
                "cuCtxSetCurrent (rotate): code {ctx_code}"
            )));
        }

        let d_src = DeviceBuf::alloc(src.len())?;
        let d_dst = DeviceBuf::alloc(dst_len)?;

        // H → D upload.
        // SAFETY: src is valid for `src.len()` bytes; d_src is a device alloc of the same size.
        let h2d = unsafe { cudaMemcpy(d_src.ptr, src.as_ptr().cast(), src.len(), CUDA_MEMCPY_H2D) };
        if h2d != 0 {
            return Err(NppRotateError(format!("cudaMemcpy H2D: code {h2d}")));
        }

        let npp_ctx = self.build_npp_ctx()?;
        Self::call_nppi_rotate(
            &d_src, src_stride, &d_dst, dst_stride, w, h, n_angle, shift_x, shift_y, npp_ctx,
        )?;

        // Stream sync then D → H download.
        let sync = unsafe { cuStreamSynchronize(self.stream) };
        if sync != 0 {
            return Err(NppRotateError(format!("cuStreamSynchronize: code {sync}")));
        }

        let mut dst_pixels = vec![0u8; dst_len];
        // SAFETY: d_dst was written by the NPP kernel and synced; dst_pixels has `dst_len` bytes.
        let d2h = unsafe {
            cudaMemcpy(
                dst_pixels.as_mut_ptr().cast(),
                d_dst.ptr,
                dst_len,
                CUDA_MEMCPY_D2H,
            )
        };
        if d2h != 0 {
            return Err(NppRotateError(format!("cudaMemcpy D2H: code {d2h}")));
        }

        Ok(dst_pixels)
    }

    /// Build the NPP stream context for the current stream.
    fn build_npp_ctx(&self) -> Result<NppStreamContext> {
        // nppSetStream + nppGetStreamContext fills in device properties,
        // avoiding manual cudaGetDeviceProperties calls.
        let set_code = unsafe { nppSetStream(self.stream.cast()) };
        if set_code != NPP_SUCCESS {
            return Err(NppRotateError(format!("nppSetStream: code {set_code}")));
        }
        let mut npp_ctx = NppStreamContext::default();
        let get_code = unsafe { nppGetStreamContext(ptr::addr_of_mut!(npp_ctx)) };
        if get_code != NPP_SUCCESS {
            return Err(NppRotateError(format!(
                "nppGetStreamContext: code {get_code}"
            )));
        }
        Ok(npp_ctx)
    }

    /// Launch `nppiRotate_8u_C1R_Ctx`.
    #[allow(clippy::too_many_arguments)]
    fn call_nppi_rotate(
        d_src: &DeviceBuf,
        src_stride: usize,
        d_dst: &DeviceBuf,
        dst_stride: usize,
        w: usize,
        h: usize,
        n_angle: f64,
        shift_x: f64,
        shift_y: f64,
        npp_ctx: NppStreamContext,
    ) -> Result<()> {
        // NPP image dimensions must fit in i32 (API contract).
        // Real PDF pages are well under 2^31 pixels per side.
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_possible_wrap,
            reason = "PDF page dimensions are always < 2^31; NPP API requires i32"
        )]
        let (wi, hi, src_step, dst_step) =
            (w as i32, h as i32, src_stride as i32, dst_stride as i32);

        let full_roi = NppiRect {
            x: 0,
            y: 0,
            width: wi,
            height: hi,
        };
        // SAFETY: d_src/d_dst are valid device allocations; size and step are consistent;
        //         npp_ctx was filled by nppGetStreamContext for the current device and stream.
        let status = unsafe {
            nppiRotate_8u_C1R_Ctx(
                d_src.ptr.cast::<u8>(),
                NppiSize {
                    width: wi,
                    height: hi,
                },
                src_step,
                NppiRect {
                    x: 0,
                    y: 0,
                    width: wi,
                    height: hi,
                },
                d_dst.ptr.cast::<u8>(),
                dst_step,
                full_roi,
                n_angle,
                shift_x,
                shift_y,
                NPPI_INTER_LINEAR,
                npp_ctx,
            )
        };
        // status >= 0: success or non-fatal warning (e.g. 29 = quad out of dst ROI).
        if status < 0 {
            return Err(NppRotateError(format!(
                "nppiRotate_8u_C1R_Ctx: status {status}"
            )));
        }
        Ok(())
    }
}

impl Drop for NppRotator {
    fn drop(&mut self) {
        // SAFETY: stream and cu_ctx were initialised in new(); this is the unique owner.
        unsafe {
            let _ = cuStreamDestroy(self.stream);
            let _ = cuDevicePrimaryCtxRelease(self.device);
        }
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Compute the NPP angle and post-rotation shift needed to rotate a `w × h`
/// image CW by `angle_deg` around its centre.
///
/// NPP uses a CW-positive angle convention (image-space: Y axis points down).
/// The post-rotation shift keeps the image centred on its original centre:
///   `nAngle  = +θ`   (CW-positive, same as our deskew convention)
///   `shiftX  = cx − cx·cos(θ) − cy·sin(θ)`
///   `shiftY  = cy + cx·sin(θ) − cy·cos(θ)`
fn centre_pivot_shift(w: usize, h: usize, angle_deg: f32) -> (f64, f64, f64) {
    let theta = f64::from(angle_deg).to_radians();
    let cos_t = theta.cos();
    let sin_t = theta.sin();
    // Image dimensions fit comfortably in f64 mantissa (≤ 32 768 px per side).
    #[expect(
        clippy::cast_precision_loss,
        reason = "image dimensions ≤ 32_768 px; f64 mantissa is 52 bits; no precision lost"
    )]
    let (cx, cy) = ((w as f64 - 1.0) * 0.5, (h as f64 - 1.0) * 0.5);
    let n_angle = f64::from(angle_deg); // NPP is CW-positive (image-space Y-down)
    let shift_x = cy.mul_add(-sin_t, cx.mul_add(-cos_t, cx));
    let shift_y = cx.mul_add(sin_t, cy.mul_add(-cos_t, cy));
    (n_angle, shift_x, shift_y)
}

// ── Thread-local entry point ──────────────────────────────────────────────────

thread_local! {
    /// Per-thread rotator: `None` = not yet initialised or previously failed.
    static ROTATOR: RefCell<Option<NppRotator>> = const { RefCell::new(None) };
}

/// Rotate a single-channel 8-bit grayscale image by `angle_deg` (clockwise
/// positive) on the GPU.
///
/// - `src`: pixel data, `src_stride * height` bytes (may have row padding).
/// - `src_stride`: bytes per row in `src`.
///
/// Returns tightly packed output pixels (`width * height` bytes).
///
/// On the first call per thread, initialises the CUDA context and stream.
/// Returns `Err` if GPU initialisation or the NPP kernel fails, so the
/// caller can fall back to the CPU path.
///
/// # Errors
///
/// Returns [`NppRotateError`] if GPU initialisation or any CUDA/NPP call fails.
///
/// # Panics
///
/// Panics if the internal `thread_local` slot is in an unexpected state (should
/// never happen; the slot is always `None` or `Some`).
pub fn rotate_gray8(
    src: &[u8],
    src_stride: usize,
    width: u32,
    height: u32,
    angle_deg: f32,
) -> std::result::Result<Vec<u8>, NppRotateError> {
    ROTATOR.with(|cell| {
        let mut slot = cell.borrow_mut();
        if slot.is_none() {
            match NppRotator::new() {
                Ok(r) => *slot = Some(r),
                Err(e) => {
                    log::warn!("npp_rotate: GPU init failed ({e}); using CPU fallback");
                    return Err(e);
                }
            }
        }
        slot.as_ref()
            .expect("just initialised above")
            .rotate(src, src_stride, width, height, angle_deg)
    })
}
