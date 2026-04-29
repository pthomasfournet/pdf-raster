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
//! [`NppRotator`] is `Send` but not `Sync`.  The recommended entry point is
//! [`rotate_gray8`], which manages a per-thread instance via `thread_local!`.
//! Constructing [`NppRotator`] directly is safe, but it must only be used from
//! the thread that called [`NppRotator::new`] — CUDA primary contexts are
//! thread-bound and sharing an instance across threads is unsound.

#![cfg(feature = "gpu-deskew")]

use std::cell::RefCell;
use std::ffi::c_void;

use crate::cuda::{
    CUstream as CudaHandle, DeviceBuf, cuCtxSetCurrent, cuDevicePrimaryCtxRelease,
    cuStreamDestroy, cuStreamSynchronize, init_primary_ctx_and_stream,
};

// ── CUDA runtime API (libcudart.so) ───────────────────────────────────────────

const CUDA_MEMCPY_H2D: i32 = 1;
const CUDA_MEMCPY_D2H: i32 = 2;

unsafe extern "C" {
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: i32) -> i32;
}

// ── NPP core (libnppc.so) ─────────────────────────────────────────────────────

/// NPP status code: 0 = success, negative = error, positive = warning.
type NppStatus = i32;
const NPP_SUCCESS: i32 = 0;
/// NPP warning: rotated source quad does not intersect the destination ROI;
/// **no pixels were written**.  Value confirmed in `nppdefs.h`.
const NPP_WRONG_INTERSECTION_QUAD_WARNING: i32 = 30;

/// NPP stream context; populated by [`nppGetStreamContext`] after
/// [`nppSetStream`] points NPP at the correct stream.
#[repr(C)]
#[derive(Default)]
struct NppStreamContext {
    h_stream: CudaHandle,
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
    // No warning codes are defined for these two calls; any non-zero is an error.
    fn nppSetStream(stream: CudaHandle) -> NppStatus;
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
#[derive(Clone, Copy)]
struct NppiRect {
    x: i32,
    y: i32,
    width: i32,
    height: i32,
}

/// Bilinear interpolation mode (`NPPI_INTER_LINEAR = 2`).
const NPPI_INTER_LINEAR: i32 = 2;

unsafe extern "C" {
    #[expect(clippy::too_many_arguments, reason = "mirrors the NPP C API exactly")]
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
///
/// # Safety
///
/// `NppRotator` must only be used from the thread that called [`NppRotator::new`].
/// CUDA primary contexts are thread-bound; using the same instance from a
/// different thread produces `CUDA_ERROR_INVALID_CONTEXT`.  The recommended
/// entry point is [`rotate_gray8`], which enforces this via `thread_local!`.
pub(crate) struct NppRotator {
    cu_ctx: *mut c_void,
    device: i32,
    stream: CudaHandle,
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
    ///
    /// # Panics
    ///
    /// Panics if a CUDA function reports success but returns a null handle —
    /// that indicates a driver bug.
    pub(crate) fn new() -> Result<Self> {
        let init = init_primary_ctx_and_stream(0)
            .map_err(|e| NppRotateError(format!("{}: code {}", e.step, e.code)))?;
        Ok(Self {
            cu_ctx: init.cu_ctx,
            device: init.device,
            stream: init.stream,
        })
    }

    /// Rotate a single-channel 8-bit grayscale image by `angle_deg` degrees
    /// (clockwise positive) via NPP bilinear.
    ///
    /// - `src`: pixel data; must be at least `src_stride * height` bytes.
    /// - `src_stride`: bytes per row in `src`; must be ≥ `width`.
    /// - `width`, `height`: image dimensions in pixels; both must be non-zero
    ///   and fit in `i32` (i.e. ≤ 2 147 483 647).
    /// - `angle_deg`: rotation angle in degrees (clockwise positive); must be
    ///   finite (not NaN or ±infinity).
    ///
    /// Returns tightly packed output pixels (`width * height` bytes, no padding).
    ///
    /// # Errors
    ///
    /// Returns [`NppRotateError`] if any precondition is violated or any CUDA
    /// or NPP call fails.
    pub(crate) fn rotate(
        &self,
        src: &[u8],
        src_stride: usize,
        width: u32,
        height: u32,
        angle_deg: f32,
    ) -> Result<Vec<u8>> {
        // ── Input validation ─────────────────────────────────────────────────

        if !angle_deg.is_finite() {
            return Err(NppRotateError(format!(
                "angle_deg is not finite: {angle_deg}"
            )));
        }

        let w = width as usize;
        let h = height as usize;

        if w == 0 || h == 0 {
            return Err(NppRotateError(format!(
                "degenerate image: {width}×{height}"
            )));
        }

        // NPP API uses i32 for dimensions and step; check before casting.
        // (PDF pages are well under this limit, but the public API must enforce it.)
        if w > i32::MAX as usize || h > i32::MAX as usize {
            return Err(NppRotateError(format!(
                "image {width}×{height} exceeds NPP i32 dimension limit"
            )));
        }
        if src_stride < w {
            return Err(NppRotateError(format!(
                "src_stride {src_stride} < width {w}"
            )));
        }
        if src_stride > i32::MAX as usize {
            return Err(NppRotateError(format!(
                "src_stride {src_stride} exceeds NPP i32 step limit"
            )));
        }

        let required_src = src_stride
            .checked_mul(h)
            .ok_or_else(|| NppRotateError("src_stride * height overflows usize".into()))?;
        if src.len() < required_src {
            return Err(NppRotateError(format!(
                "src too short: {} bytes for stride {src_stride} × {h} rows",
                src.len()
            )));
        }

        let dst_stride = w; // tightly packed output
        // w and h are both ≤ i32::MAX (2 147 483 647); checked_mul is defence-in-depth.
        let dst_len = w
            .checked_mul(h)
            .ok_or_else(|| NppRotateError("width * height overflows usize".into()))?;

        // ── Computation ──────────────────────────────────────────────────────

        let (n_angle, shift_x, shift_y) = centre_pivot_shift(w, h, angle_deg);

        // Restore CUDA context (rayon workers may interleave threads between calls).
        let r = unsafe { cuCtxSetCurrent(self.cu_ctx) };
        if r != 0 {
            return Err(NppRotateError(format!(
                "cuCtxSetCurrent (rotate): code {r}"
            )));
        }

        let d_src = DeviceBuf::alloc(src.len())
            .map_err(|code| NppRotateError(format!("cudaMalloc(src, {}): code {code}", src.len())))?;
        let d_dst = DeviceBuf::alloc(dst_len)
            .map_err(|code| NppRotateError(format!("cudaMalloc(dst, {dst_len}): code {code}")))?;

        // H → D upload.
        // SAFETY: src is valid for `src.len()` bytes; we validated src.len() >= src_stride*h
        // above, so NPP will not read past the device allocation.
        let r = unsafe { cudaMemcpy(d_src.ptr, src.as_ptr().cast(), src.len(), CUDA_MEMCPY_H2D) };
        if r != 0 {
            return Err(NppRotateError(format!("cudaMemcpy H2D: code {r}")));
        }

        let npp_ctx = self.build_npp_ctx()?;
        Self::call_nppi_rotate(
            &d_src, src_stride, &d_dst, dst_stride, w, h, n_angle, shift_x, shift_y, npp_ctx,
        )?;

        // Stream sync then D → H download.
        let r = unsafe { cuStreamSynchronize(self.stream) };
        if r != 0 {
            return Err(NppRotateError(format!("cuStreamSynchronize: code {r}")));
        }

        let mut dst_pixels = vec![0u8; dst_len];
        // SAFETY: d_dst was written by the NPP kernel and synced; dst_pixels has `dst_len` bytes.
        let r = unsafe {
            cudaMemcpy(
                dst_pixels.as_mut_ptr().cast(),
                d_dst.ptr,
                dst_len,
                CUDA_MEMCPY_D2H,
            )
        };
        if r != 0 {
            return Err(NppRotateError(format!("cudaMemcpy D2H: code {r}")));
        }

        Ok(dst_pixels)
    }

    /// Build the NPP stream context for the current stream.
    fn build_npp_ctx(&self) -> Result<NppStreamContext> {
        // nppSetStream + nppGetStreamContext fills in device properties,
        // avoiding manual cudaGetDeviceProperties calls.
        // No warning codes are defined for either call; any non-zero is a hard error.
        let r = unsafe { nppSetStream(self.stream) };
        if r != NPP_SUCCESS {
            return Err(NppRotateError(format!("nppSetStream: code {r}")));
        }
        let mut npp_ctx = NppStreamContext::default();
        let r = unsafe { nppGetStreamContext(&raw mut npp_ctx) };
        if r != NPP_SUCCESS {
            return Err(NppRotateError(format!("nppGetStreamContext: code {r}")));
        }
        Ok(npp_ctx)
    }

    /// Launch `nppiRotate_8u_C1R_Ctx`.
    ///
    /// Precondition: `w`, `h`, `src_stride`, and `dst_stride` all fit in `i32`
    /// (caller must validate before calling — see [`NppRotator::rotate`]).
    #[expect(
        clippy::too_many_arguments,
        reason = "internal helper mirroring the NPP C API"
    )]
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
        // All dimension/stride values were range-checked in rotate() before this call.
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_possible_wrap,
            reason = "bounds checked against i32::MAX in rotate() before this call"
        )]
        let (wi, hi, src_step, dst_step) =
            (w as i32, h as i32, src_stride as i32, dst_stride as i32);

        let full_roi = NppiRect {
            x: 0,
            y: 0,
            width: wi,
            height: hi,
        };
        // SAFETY: d_src/d_dst are valid device allocations sized for the pixel data;
        //         size and step are consistent with width/height; npp_ctx was filled
        //         by nppGetStreamContext for the current device and stream.
        let status = unsafe {
            nppiRotate_8u_C1R_Ctx(
                d_src.ptr.cast::<u8>(),
                NppiSize {
                    width: wi,
                    height: hi,
                },
                src_step,
                full_roi,
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

        if status == NPP_WRONG_INTERSECTION_QUAD_WARNING {
            // No pixels were written; the rotated quad fell entirely outside the
            // destination ROI.  This should not occur for small deskew angles
            // (≤ ±7°) on non-trivial images, but log it so it is diagnosable.
            log::warn!(
                "npp_rotate: nppiRotate returned NPP_WRONG_INTERSECTION_QUAD_WARNING (30); \
                 destination image will be blank"
            );
        } else if status < 0 {
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
            let r = cuStreamDestroy(self.stream);
            if r != 0 {
                log::warn!("npp_rotate: cuStreamDestroy failed: code {r}");
            }
            let r = cuDevicePrimaryCtxRelease(self.device);
            if r != 0 {
                log::warn!("npp_rotate: cuDevicePrimaryCtxRelease failed: code {r}");
            }
        }
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Compute the NPP angle and post-rotation shift needed to rotate a `w × h`
/// image CW by `angle_deg` around its centre.
///
/// NPP uses a CW-positive angle convention (image-space: Y axis points down).
/// The post-rotation shift keeps the image anchored to its original centre:
///
/// ```text
/// nAngle  = +θ                            (CW-positive, same as deskew convention)
/// shiftX  = cx − cx·cos(θ) − cy·sin(θ)
/// shiftY  = cy + cx·sin(θ) − cy·cos(θ)
/// ```
///
/// Derivation: NPP applies `dst = R_CW(θ)·src + shift`.  Setting the centre
/// point `(cx, cy)` to map to itself gives `shift = (I − R_CW(θ))·(cx, cy)`.
fn centre_pivot_shift(w: usize, h: usize, angle_deg: f32) -> (f64, f64, f64) {
    let theta = f64::from(angle_deg).to_radians();
    let cos_t = theta.cos();
    let sin_t = theta.sin();
    // w and h are ≤ i32::MAX (< 2^31); f64 mantissa is 52 bits; exact.
    #[expect(
        clippy::cast_precision_loss,
        reason = "dimensions validated ≤ i32::MAX < 2^31; f64 mantissa is 52 bits; exact"
    )]
    let (cx, cy) = ((w as f64 - 1.0) * 0.5, (h as f64 - 1.0) * 0.5);
    let n_angle = f64::from(angle_deg); // NPP is CW-positive (image-space Y-down)
    let shift_x = cy.mul_add(-sin_t, cx.mul_add(-cos_t, cx));
    let shift_y = cx.mul_add(sin_t, cy.mul_add(-cos_t, cy));
    (n_angle, shift_x, shift_y)
}

// ── Thread-local entry point ──────────────────────────────────────────────────

/// Per-thread rotator state.
enum RotatorSlot {
    /// Not yet initialised.
    Uninit,
    /// Initialisation failed permanently; do not retry.
    Failed,
    /// Ready to use.
    Ready(NppRotator),
}

thread_local! {
    static ROTATOR: RefCell<RotatorSlot> = const { RefCell::new(RotatorSlot::Uninit) };
}

/// Rotate a single-channel 8-bit grayscale image by `angle_deg` (clockwise
/// positive) on the GPU.
///
/// - `src`: pixel data; must be at least `src_stride * height` bytes.
/// - `src_stride`: bytes per row in `src`; must be ≥ `width`.
/// - `width`, `height`: image dimensions in pixels; both must be non-zero.
/// - `angle_deg`: rotation angle in degrees (clockwise positive); must be finite.
///
/// Returns tightly packed output pixels (`width * height` bytes).
///
/// On the first call per thread, initialises the CUDA context and stream.
/// If GPU initialisation fails, logs a warning and returns `Err` — the slot
/// is permanently marked as failed and no further init is attempted.
/// If the NPP kernel fails on a subsequent call, returns `Err` without
/// poisoning the slot (a transient error may be recoverable).
///
/// # Errors
///
/// Returns [`NppRotateError`] if GPU initialisation or any CUDA/NPP call fails.
pub fn rotate_gray8(
    src: &[u8],
    src_stride: usize,
    width: u32,
    height: u32,
    angle_deg: f32,
) -> std::result::Result<Vec<u8>, NppRotateError> {
    ROTATOR.with(|cell| {
        let mut slot = cell.borrow_mut();

        if matches!(*slot, RotatorSlot::Uninit) {
            match NppRotator::new() {
                Ok(r) => *slot = RotatorSlot::Ready(r),
                Err(e) => {
                    log::warn!("npp_rotate: GPU init failed ({e}); falling back to CPU for all pages on this thread");
                    *slot = RotatorSlot::Failed;
                    return Err(e);
                }
            }
        }

        match &*slot {
            RotatorSlot::Ready(r) => r.rotate(src, src_stride, width, height, angle_deg),
            RotatorSlot::Failed => Err(NppRotateError("GPU init previously failed".into())),
            RotatorSlot::Uninit => unreachable!("slot was just initialised"),
        }
    })
}
