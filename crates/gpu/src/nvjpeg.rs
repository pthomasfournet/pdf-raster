//! nvJPEG GPU-accelerated JPEG decoding.
//!
//! Wraps the NVIDIA nvJPEG library (CUDA 12, version 12.3.5 on this system)
//! to decode JPEG bitstreams on the GPU, writing finished pixel bytes directly
//! into a host `Vec<u8>` via an async CUDA stream.
//!
//! On the RTX 5070 (Blackwell, CC 12.0) this runs at ~10 GB/s, roughly 10–20×
//! faster than the CPU software JPEG path for large images.  We use
//! `NVJPEG_BACKEND_GPU_HYBRID` (GPU Huffman + CUDA SM IDCT/colour-convert), which
//! scales well with many concurrent Rayon threads each owning their own decoder.
//! `NVJPEG_BACKEND_HARDWARE` (fixed-function engine) is not used: it serialises
//! all callers through the engine and degrades multi-threaded throughput.
//!
//! # Memory model
//!
//! `nvjpegDecode` writes to a `PinnedBuf` — page-locked host memory allocated
//! via `cuMemAllocHost_v2`.  Pinned memory is required for GPU DMA; plain heap
//! allocations (`Vec<u8>`) are not DMA-accessible and segfault on some driver
//! versions.  After `cuStreamSynchronize` confirms all writes are complete, the
//! pinned buffer is copied into a `Vec<u8>` that the caller owns freely.
//!
//! # Usage
//!
//! ```no_run
//! use gpu::nvjpeg::NvJpegDecoder;
//! let mut dec = NvJpegDecoder::new(0).expect("nvJPEG unavailable");
//! let jpeg_bytes: &[u8] = &[]; // placeholder
//! let img = dec.decode_sync(jpeg_bytes).expect("decode failed");
//! // img.data is interleaved RGB (or luma) bytes, img.width × img.height pixels.
//! ```
//!
//! # Feature flag
//!
//! This module is only compiled when the `nvjpeg` feature is enabled on the
//! `gpu` crate.  Callers that want a CPU fallback should call
//! [`NvJpegDecoder::new`] at startup and fall back to the CPU JPEG decoder if it
//! returns `Err`.
//!
//! # Thread safety
//!
//! [`NvJpegDecoder`] is `Send` but not `Sync`: each thread must own its own
//! instance.  Creating one per thread is cheap — it retains the primary GPU
//! context (reference-counted by the driver) and creates one stream.

// All nvJPEG calls are FFI through raw pointers.
#![cfg(feature = "nvjpeg")]

use std::ptr;

// ── Raw FFI declarations ──────────────────────────────────────────────────────
//
// We declare only the nvJPEG surface we use.  This avoids the overhead of
// bindgen at build time while keeping the binding surface minimal and auditable.

/// nvJPEG status codes.  Only `NVJPEG_STATUS_SUCCESS = 0` is success.
#[expect(
    non_camel_case_types,
    reason = "C FFI type alias; must match the nvJPEG ABI name"
)]
type nvjpegStatus_t = i32;

/// Opaque nvJPEG library handle (wraps device allocators, backend state).
#[repr(C)]
struct NvjpegHandle_ {
    _opaque: [u8; 0],
}
#[expect(
    non_camel_case_types,
    reason = "C FFI type alias; must match the nvJPEG ABI name"
)]
type nvjpegHandle_t = *mut NvjpegHandle_;

/// Opaque per-decode state (Huffman tables, coefficient buffers, etc.).
#[repr(C)]
struct NvjpegJpegState_ {
    _opaque: [u8; 0],
}
#[expect(
    non_camel_case_types,
    reason = "C FFI type alias; must match the nvJPEG ABI name"
)]
type nvjpegJpegState_t = *mut NvjpegJpegState_;

/// Chroma subsampling type — required out-parameter for `nvjpegGetImageInfo`;
/// we don't inspect it (component count alone drives dispatch).
#[expect(
    non_camel_case_types,
    reason = "C FFI type alias; must match the nvJPEG ABI name"
)]
type nvjpegChromaSubsampling_t = i32;

/// Output format: we use `NVJPEG_OUTPUT_RGBI` (interleaved RGB, channel[0])
/// and `NVJPEG_OUTPUT_Y` (luma-only, channel[0]).
#[expect(
    non_camel_case_types,
    reason = "C FFI type alias; must match the nvJPEG ABI name"
)]
type nvjpegOutputFormat_t = i32;

/// `NVJPEG_OUTPUT_Y` — single-channel luma output to `channel[0]`.
const NVJPEG_OUTPUT_Y: nvjpegOutputFormat_t = 2;
/// `NVJPEG_OUTPUT_RGBI` — interleaved RGB output to `channel[0]`, 3 bytes per pixel.
const NVJPEG_OUTPUT_RGBI: nvjpegOutputFormat_t = 5;

/// Backend selector passed as the first argument to `nvjpegCreateEx`.
#[expect(
    non_camel_case_types,
    reason = "C FFI type alias; must match the nvJPEG ABI name"
)]
type nvjpegBackend_t = i32;

/// Auto-select the best available backend; opaque internal heuristic.
/// Used as a fallback when `GPU_HYBRID` is unavailable.
const NVJPEG_BACKEND_DEFAULT: nvjpegBackend_t = 0;
/// GPU Huffman decoding + CUDA SM IDCT/colour-convert.
/// Scales well under concurrent multi-thread load; our primary backend.
const NVJPEG_BACKEND_GPU_HYBRID: nvjpegBackend_t = 2;
/// Fixed-function hardware JPEG engine.  Supported on Ampere (A100/A30), Hopper,
/// Ada, Blackwell, and Jetson Thor — **not Turing** (despite marketing).  Serialises
/// all concurrent callers through the engine; documented as slower than
/// `GPU_HYBRID` for multi-threaded workloads.  Off by default; enable via
/// the `nvjpeg-hardware` cargo feature to measure on a specific machine.
#[cfg_attr(
    not(feature = "nvjpeg-hardware"),
    expect(
        dead_code,
        reason = "documents the HARDWARE backend value; only used when nvjpeg-hardware feature is on"
    )
)]
const NVJPEG_BACKEND_HARDWARE: nvjpegBackend_t = 3;

/// Maximum number of colour plane pointers in `nvjpegImage_t`.
const NVJPEG_MAX_COMPONENT: usize = 4;

/// Destination image buffer descriptor passed to `nvjpegDecode`.
///
/// For `NVJPEG_OUTPUT_RGBI` / `NVJPEG_OUTPUT_Y`, only `channel[0]` and
/// `pitch[0]` are used; the rest must be set to null / 0.
#[repr(C)]
#[derive(Debug)]
struct NvjpegImage {
    channel: [*mut u8; NVJPEG_MAX_COMPONENT],
    pitch: [usize; NVJPEG_MAX_COMPONENT],
}

impl NvjpegImage {
    const fn zeroed() -> Self {
        Self {
            channel: [ptr::null_mut(); NVJPEG_MAX_COMPONENT],
            pitch: [0; NVJPEG_MAX_COMPONENT],
        }
    }
}

/// A raw CUDA stream handle as an opaque pointer.  This is the C type
/// `CUstream` / `cudaStream_t`; we only pass it through to nvJPEG.
type CUstream = *mut std::ffi::c_void;

// CUDA driver API — context, stream, pinned host memory.
// These functions are in libcuda.so (the NVIDIA driver library).
unsafe extern "C" {
    /// Initialise the CUDA driver API.  Must be called once before any other API.
    /// `flags` must be 0.
    fn cuInit(flags: u32) -> i32;
    /// Get the CUDA device handle for device `ordinal`.
    fn cuDeviceGet(device: *mut i32, ordinal: i32) -> i32;
    /// Retain the primary context for `device`.
    fn cuDevicePrimaryCtxRetain(ctx: *mut *mut std::ffi::c_void, device: i32) -> i32;
    /// Release the primary context for `device`.
    fn cuDevicePrimaryCtxRelease(device: i32) -> i32;
    /// Set the context current on the calling thread.
    fn cuCtxSetCurrent(ctx: *mut std::ffi::c_void) -> i32;
    /// Create a CUDA stream in the current context.
    /// `flags` = 0 for default (non-blocking) stream.
    fn cuStreamCreate(stream: *mut CUstream, flags: u32) -> i32;
    /// Destroy a CUDA stream.
    fn cuStreamDestroy(stream: CUstream) -> i32;
    /// Block the calling thread until all work enqueued on `stream` completes.
    fn cuStreamSynchronize(stream: CUstream) -> i32;
    /// Probe a context for validity.  Returns non-zero if the context is not
    /// fully initialised (e.g. driver still recovering after a peer process crash).
    fn cuCtxGetApiVersion(ctx: *mut std::ffi::c_void, version: *mut u32) -> i32;
    /// Allocate page-locked (pinned) host memory.
    ///
    /// Pinned memory can be DMA'd by the GPU without an intermediate copy.
    /// `nvjpegDecode` writes to `channel[0]` via the GPU; the pointer must be
    /// pinned so the DMA transfer does not SIGSEGV on access.
    ///
    /// The CUDA header `cuda.h` defines `cuMemAllocHost` as a macro that
    /// redirects to `cuMemAllocHost_v2`; we declare the v2 symbol directly here
    /// so that the Rust FFI calls the same symbol as C code compiled with `cuda.h`.
    /// The old `cuMemAllocHost` symbol returns `CUDA_ERROR_INVALID_CONTEXT (201)`.
    #[link_name = "cuMemAllocHost_v2"]
    fn cuMemAllocHost(pp: *mut *mut std::ffi::c_void, bytesize: usize) -> i32;
    /// Free pinned host memory allocated by `cuMemAllocHost`.
    fn cuMemFreeHost(p: *mut std::ffi::c_void) -> i32;
}

// NVJPEGAPI on Linux is the default calling convention — no special attribute.
unsafe extern "C" {
    /// Create a library handle with explicit backend and custom allocator support.
    ///
    /// Prefer this over the deprecated `nvjpegCreate`.  Pass `NULL` for both
    /// allocator parameters to use the default CUDA allocators (`cudaMalloc` /
    /// `cudaFreeHost`).  `flags` must be 0.
    ///
    /// Signature (nvjpeg.h, CUDA 12.8):
    /// `nvjpegCreateEx(backend, dev_allocator, pinned_allocator, flags, handle)`
    fn nvjpegCreateEx(
        backend: nvjpegBackend_t,
        dev_allocator: *mut std::ffi::c_void,
        pinned_allocator: *mut std::ffi::c_void,
        flags: u32,
        handle: *mut nvjpegHandle_t,
    ) -> nvjpegStatus_t;
    fn nvjpegDestroy(handle: nvjpegHandle_t) -> nvjpegStatus_t;
    fn nvjpegJpegStateCreate(
        handle: nvjpegHandle_t,
        jpeg_handle: *mut nvjpegJpegState_t,
    ) -> nvjpegStatus_t;
    fn nvjpegJpegStateDestroy(jpeg_handle: nvjpegJpegState_t) -> nvjpegStatus_t;
    fn nvjpegGetImageInfo(
        handle: nvjpegHandle_t,
        data: *const u8,
        length: usize,
        n_components: *mut i32,
        subsampling: *mut nvjpegChromaSubsampling_t,
        widths: *mut i32,
        heights: *mut i32,
    ) -> nvjpegStatus_t;
    fn nvjpegDecode(
        handle: nvjpegHandle_t,
        jpeg_handle: nvjpegJpegState_t,
        data: *const u8,
        length: usize,
        output_format: nvjpegOutputFormat_t,
        destination: *mut NvjpegImage,
        stream: CUstream,
    ) -> nvjpegStatus_t;
}

const NVJPEG_STATUS_SUCCESS: nvjpegStatus_t = 0;
/// Bitstream truncated or incomplete.
const NVJPEG_STATUS_INCOMPLETE_BITSTREAM: nvjpegStatus_t = 10;

// ── Error type ────────────────────────────────────────────────────────────────

/// Errors returned by the nvJPEG wrapper.
#[derive(Debug)]
pub enum NvJpegError {
    /// nvJPEG API returned a non-zero status code.
    ///
    /// Common values:
    /// - 2: `NVJPEG_STATUS_INVALID_PARAMETER`
    /// - 3: `NVJPEG_STATUS_BAD_JPEG` — malformed bitstream
    /// - 4: `NVJPEG_STATUS_JPEG_NOT_SUPPORTED` — encoding type unsupported by backend
    /// - 6: `NVJPEG_STATUS_EXECUTION_FAILED` — GPU kernel error
    /// - 7: `NVJPEG_STATUS_ARCH_MISMATCH` — binary compiled for wrong GPU arch
    /// - 8: `NVJPEG_STATUS_INTERNAL_ERROR`
    /// - 9: `NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED` — backend not available
    /// - 10: `NVJPEG_STATUS_INCOMPLETE_BITSTREAM` — truncated or partial JPEG
    NvjpegStatus(i32),
    /// Image has an unsupported number of components (not 1 or 3; 4 = CMYK falls through to CPU).
    UnsupportedComponents(i32),
    /// The decoded image has a zero or negative dimension.
    ZeroDimension {
        /// Reported pixel width (≤ 0).
        width: i32,
        /// Reported pixel height (≤ 0).
        height: i32,
    },
    /// Arithmetic overflow computing pixel buffer size (image dimensions unreasonably large).
    Overflow,
    /// CUDA driver API error (`CUresult`).
    ///
    /// Returned from any raw CUDA call: `cuInit`, `cuDeviceGet`,
    /// `cuDevicePrimaryCtxRetain`, `cuCtxSetCurrent`, `cuStreamCreate`,
    /// `cuMemAllocHost`, or `cuStreamSynchronize`.
    /// The code is distinct from nvJPEG status codes; see `cuda.h` (`CUDA_ERROR_*`).
    CudaError(i32),
}

impl std::fmt::Display for NvJpegError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NvjpegStatus(code) => {
                let name = match *code {
                    2 => "NVJPEG_STATUS_INVALID_PARAMETER",
                    3 => "NVJPEG_STATUS_BAD_JPEG",
                    4 => "NVJPEG_STATUS_JPEG_NOT_SUPPORTED",
                    5 => "NVJPEG_STATUS_ALLOCATOR_FAILURE",
                    6 => "NVJPEG_STATUS_EXECUTION_FAILED",
                    7 => "NVJPEG_STATUS_ARCH_MISMATCH",
                    8 => "NVJPEG_STATUS_INTERNAL_ERROR",
                    9 => "NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED",
                    x if x == NVJPEG_STATUS_INCOMPLETE_BITSTREAM => {
                        "NVJPEG_STATUS_INCOMPLETE_BITSTREAM"
                    }
                    _ => "NVJPEG_STATUS_UNKNOWN",
                };
                write!(f, "nvJPEG error {code} ({name})")
            }
            Self::UnsupportedComponents(n) => write!(f, "unsupported JPEG component count {n}"),
            Self::ZeroDimension { width, height } => {
                write!(
                    f,
                    "JPEG reported zero or negative dimension {width}×{height}"
                )
            }
            Self::Overflow => write!(f, "pixel buffer size overflow (image too large)"),
            Self::CudaError(code) => {
                // Codes from cuda.h (CUDA 12.8); confirmed against installed headers.
                let name = match *code {
                    1 => "CUDA_ERROR_INVALID_VALUE",
                    2 => "CUDA_ERROR_OUT_OF_MEMORY",
                    3 => "CUDA_ERROR_NOT_INITIALIZED",
                    35 => "CUDA_ERROR_INSUFFICIENT_DRIVER",
                    100 => "CUDA_ERROR_NO_DEVICE",
                    101 => "CUDA_ERROR_INVALID_DEVICE",
                    200 => "CUDA_ERROR_INVALID_IMAGE",
                    201 => "CUDA_ERROR_INVALID_CONTEXT",
                    205 => "CUDA_ERROR_MAP_FAILED",
                    209 => "CUDA_ERROR_NO_BINARY_FOR_GPU",
                    218 => "CUDA_ERROR_INVALID_PTX",
                    400 => "CUDA_ERROR_INVALID_HANDLE",
                    600 => "CUDA_ERROR_NOT_READY",
                    700 => "CUDA_ERROR_ILLEGAL_ADDRESS",
                    _ => "CUDA_ERROR_UNKNOWN",
                };
                write!(f, "CUDA driver error {code} ({name})")
            }
        }
    }
}

impl std::error::Error for NvJpegError {}

/// Convenience alias.
pub type Result<T> = std::result::Result<T, NvJpegError>;

// ── Colour space ──────────────────────────────────────────────────────────────

/// Colour space of the pixels returned by [`NvJpeg::decode`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JpegColorSpace {
    /// One byte per pixel (luma / grayscale).
    Gray,
    /// Three bytes per pixel, interleaved R G B.
    Rgb,
}

// ── Pinned host memory ────────────────────────────────────────────────────────

/// RAII wrapper for a page-locked (pinned) host buffer allocated via `cuMemAllocHost`.
///
/// `nvjpegDecode` writes to `channel[0]` asynchronously via CUDA DMA.  The
/// destination pointer must be pinned host memory; regular heap memory (`Vec<u8>`)
/// is not DMA-accessible and will SIGSEGV on some CUDA driver versions.
struct PinnedBuf {
    ptr: *mut u8,
    len: usize,
}

impl PinnedBuf {
    /// Allocate `len` bytes of pinned host memory.
    fn alloc(len: usize) -> std::result::Result<Self, NvJpegError> {
        let mut raw: *mut std::ffi::c_void = std::ptr::null_mut();
        // SAFETY: cuMemAllocHost writes to `raw`; `len` is the allocation size.
        let result = unsafe { cuMemAllocHost(&raw mut raw, len) };
        if result != 0 {
            return Err(NvJpegError::CudaError(result));
        }
        assert!(
            !raw.is_null(),
            "cuMemAllocHost succeeded but returned null pointer"
        );
        Ok(Self {
            ptr: raw.cast::<u8>(),
            len,
        })
    }

    /// Copy the pinned buffer contents into a `Vec<u8>` for the caller.
    fn to_vec(&self) -> Vec<u8> {
        // SAFETY: ptr is valid and len bytes are written by the GPU before this is called.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len).to_vec() }
    }

    /// Return the raw mutable pointer for passing to `nvjpegImage_t::channel`.
    const fn dma_ptr(&self) -> *mut u8 {
        self.ptr
    }
}

impl Drop for PinnedBuf {
    fn drop(&mut self) {
        // alloc() asserts non-null before returning Ok, so ptr is always valid here.
        debug_assert!(
            !self.ptr.is_null(),
            "PinnedBuf::ptr is null — construction invariant violated"
        );
        // SAFETY: ptr came from cuMemAllocHost_v2; no other references exist at drop time.
        let _ = unsafe { cuMemFreeHost(self.ptr.cast()) };
    }
}

// SAFETY: PinnedBuf is an owned allocation; the raw pointer is not shared.
unsafe impl Send for PinnedBuf {}

// ── Decoded image ─────────────────────────────────────────────────────────────

/// Decoded JPEG image, host-resident.
#[derive(Debug)]
pub struct DecodedJpeg {
    /// Pixel bytes.  Layout matches [`color_space`](Self::color_space).
    pub data: Vec<u8>,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Colour space of the pixel bytes.
    pub color_space: JpegColorSpace,
}

// ── NvJpeg ────────────────────────────────────────────────────────────────────

/// nvJPEG decoder context.
///
/// Holds the library handle and a reusable per-decode state object.
/// `Send` but not `Sync`: each thread should own its own instance.
///
/// This type is an implementation detail of [`NvJpegDecoder`]; use that
/// public API instead.
pub(crate) struct NvJpeg {
    handle: nvjpegHandle_t,
    state: nvjpegJpegState_t,
}

// ── PendingDecode ─────────────────────────────────────────────────────────────

/// In-flight decode result: pinned buffer + metadata, before stream sync.
///
/// `decode_inner` returns this; `decode_sync` synchronises the stream and
/// then calls `into_decoded_jpeg` to copy pixels into a `Vec<u8>`.
struct PendingDecode {
    pinned: PinnedBuf,
    width: i32,
    height: i32,
    color_space: JpegColorSpace,
}

impl PendingDecode {
    /// Copy the finished pixel data into a `Vec<u8>`.
    ///
    /// Must only be called after the stream is synchronised; otherwise the GPU
    /// may still be writing and the copy will read partial data.
    fn into_decoded_jpeg(self) -> DecodedJpeg {
        // width and height are i32 and were validated > 0 in decode_inner;
        // positive i32 always fits in u32, so the cast is safe.
        debug_assert!(self.width > 0 && self.height > 0);
        #[expect(
            clippy::cast_sign_loss,
            reason = "validated > 0 in decode_inner before PendingDecode is constructed"
        )]
        DecodedJpeg {
            data: self.pinned.to_vec(),
            width: self.width as u32,
            height: self.height as u32,
            color_space: self.color_space,
        }
    }
}

// SAFETY: nvjpegHandle_t and nvjpegJpegState_t are opaque C pointers.
// nvJPEG guarantees that a handle used from one host thread at a time is safe;
// the internal CUDA stream serialises device-side work.
unsafe impl Send for NvJpeg {}

impl NvJpeg {
    /// Initialise nvJPEG with `GPU_HYBRID`, falling back to `DEFAULT`.
    ///
    /// `GPU_HYBRID` uses CUDA SMs for Huffman decode + IDCT/colour-convert and
    /// scales well when many Rayon worker threads decode concurrently.
    /// `DEFAULT` is an opaque library-internal selection used as a fallback when
    /// `GPU_HYBRID` is unavailable on older drivers.
    ///
    /// `HARDWARE` (fixed-function engine) is intentionally not used: it serialises
    /// all concurrent callers through a single engine and degrades multi-threaded
    /// throughput.  It is also not available on Turing GPUs despite marketing claims;
    /// the real availability baseline is Ampere (A100/A30), Hopper, Ada, Blackwell.
    ///
    /// # Errors
    ///
    /// Returns an error if both backends fail to initialise, or if per-decode state
    /// allocation fails.  Treat as "nvJPEG unavailable; fall back to CPU".
    pub(crate) fn new() -> Result<Self> {
        let handle = Self::create_handle()?;
        let mut state: nvjpegJpegState_t = ptr::null_mut();
        // SAFETY: handle is valid; nvjpegJpegStateCreate allocates per-decode buffers.
        let status = unsafe { nvjpegJpegStateCreate(handle, &raw mut state) };
        if status != NVJPEG_STATUS_SUCCESS {
            // Destroy the handle before propagating the error.
            // SAFETY: handle is valid; state was never initialised.
            let _ = unsafe { nvjpegDestroy(handle) };
            return Err(NvJpegError::NvjpegStatus(status));
        }
        assert!(
            !state.is_null(),
            "nvjpegJpegStateCreate succeeded but returned null state"
        );
        Ok(Self { handle, state })
    }

    /// Pick the nvJPEG backend.
    ///
    /// Default: `GPU_HYBRID` (CPU Huffman + GPU IDCT) with `DEFAULT` fallback.
    /// With the `nvjpeg-hardware` feature flag: try the fixed-function
    /// `HARDWARE` engine first, then `GPU_HYBRID`, then `DEFAULT`.  The flag
    /// exists for empirical comparison — datacenter parts (A100/H100/Ada)
    /// have multi-engine `HARDWARE` and should win on it; consumer Blackwell
    /// has a single engine and was assumed to lose.  Use the bench matrix to
    /// confirm rather than infer.
    fn create_handle() -> Result<nvjpegHandle_t> {
        #[cfg(feature = "nvjpeg-hardware")]
        {
            match Self::try_backend(NVJPEG_BACKEND_HARDWARE) {
                Ok(h) => {
                    log::debug!("nvJPEG: using HARDWARE backend (nvjpeg-hardware feature on)");
                    return Ok(h);
                }
                Err(e) => {
                    log::warn!(
                        "nvJPEG HARDWARE backend unavailable ({e}); falling back to GPU_HYBRID"
                    );
                }
            }
        }
        Self::try_backend(NVJPEG_BACKEND_GPU_HYBRID).or_else(|e| {
            log::debug!("nvJPEG GPU_HYBRID backend unavailable ({e}), falling back to DEFAULT");
            Self::try_backend(NVJPEG_BACKEND_DEFAULT)
        })
    }

    /// Attempt to open an nvJPEG library handle with the given `backend`.
    ///
    /// Returns the handle on success, or `NvjpegStatus` on failure.
    /// NULL allocator pointers request the default CUDA device/pinned allocators.
    /// `flags` must be 0 (reserved).
    fn try_backend(backend: nvjpegBackend_t) -> Result<nvjpegHandle_t> {
        let mut handle: nvjpegHandle_t = ptr::null_mut();
        // SAFETY: nvjpegCreateEx initialises a fresh library handle.
        // NULL allocator pointers request default CUDA allocators.
        // flags = 0 is the only valid value (reserved field).
        let status = unsafe {
            nvjpegCreateEx(
                backend,
                ptr::null_mut(),
                ptr::null_mut(),
                0,
                &raw mut handle,
            )
        };
        if status != NVJPEG_STATUS_SUCCESS {
            return Err(NvJpegError::NvjpegStatus(status));
        }
        assert!(
            !handle.is_null(),
            "nvjpegCreateEx succeeded but returned null handle"
        );
        Ok(handle)
    }

    /// Enqueue a JPEG decode on `stream`, returning a `PendingDecode`.
    ///
    /// The GPU is still writing when this returns; the caller must synchronise
    /// `stream` before calling `PendingDecode::into_decoded_jpeg`.
    fn decode(&mut self, data: &[u8], stream: CUstream) -> Result<PendingDecode> {
        // ── Phase 1: inspect headers ──────────────────────────────────────────
        let mut n_components: i32 = 0;
        // Required out-parameter for `nvjpegGetImageInfo`; value is not used —
        // component count alone drives format selection.
        let mut subsampling_out: nvjpegChromaSubsampling_t = 0;
        // nvjpegGetImageInfo fills widths[0]/heights[0]; remaining entries are 0.
        let mut widths = [0i32; NVJPEG_MAX_COMPONENT];
        let mut heights = [0i32; NVJPEG_MAX_COMPONENT];

        // SAFETY: all out-pointers are valid stack variables; data is a valid slice.
        let status = unsafe {
            nvjpegGetImageInfo(
                self.handle,
                data.as_ptr(),
                data.len(),
                &raw mut n_components,
                &raw mut subsampling_out,
                widths.as_mut_ptr(),
                heights.as_mut_ptr(),
            )
        };
        if status != NVJPEG_STATUS_SUCCESS {
            return Err(NvJpegError::NvjpegStatus(status));
        }

        let (out_fmt, bytes_per_px, color_space) = match n_components {
            1 => (NVJPEG_OUTPUT_Y, 1usize, JpegColorSpace::Gray),
            3 => (NVJPEG_OUTPUT_RGBI, 3usize, JpegColorSpace::Rgb),
            n => return Err(NvJpegError::UnsupportedComponents(n)),
        };

        let width = widths[0];
        let height = heights[0];
        if width <= 0 || height <= 0 {
            return Err(NvJpegError::ZeroDimension { width, height });
        }
        let width_u = usize::try_from(width).expect("width > 0 already checked");
        let height_u = usize::try_from(height).expect("height > 0 already checked");

        // ── Phase 2: allocate pinned host output buffer ───────────────────────
        // `nvjpegDecode` writes to `channel[0]` via CUDA DMA asynchronously on
        // `stream`.  The destination must be page-locked (pinned) host memory;
        // regular heap memory (`Vec<u8>`) is not DMA-accessible and causes
        // SIGSEGV on CUDA driver versions that don't implicitly pin it.
        //
        // Pitch = width × bytes_per_px (tightly packed, no row padding).
        let pitch = width_u
            .checked_mul(bytes_per_px)
            .ok_or(NvJpegError::Overflow)?;
        let buf_len = pitch.checked_mul(height_u).ok_or(NvJpegError::Overflow)?;

        let pinned = PinnedBuf::alloc(buf_len)?;

        let mut img = NvjpegImage::zeroed();
        img.channel[0] = pinned.dma_ptr();
        img.pitch[0] = pitch;

        // ── Phase 3: decode ───────────────────────────────────────────────────
        // SAFETY:
        // - self.handle and self.state are valid (created in `new` / `reinit_default_backend`).
        // - data slice is live for the duration of this call.
        // - img.channel[0] points into `pinned`, which is live for the call.
        // - stream is a valid CUstream owned by the caller's context.
        // nvjpegDecode enqueues GPU work on `stream` asynchronously; the caller
        // must synchronise the stream before reading `pinned`.
        let status = unsafe {
            nvjpegDecode(
                self.handle,
                self.state,
                data.as_ptr(),
                data.len(),
                out_fmt,
                &raw mut img,
                stream,
            )
        };
        if status != NVJPEG_STATUS_SUCCESS {
            // nvjpegDecode may have enqueued partial GPU work before returning the
            // error.  Synchronise the stream before dropping `pinned` so the driver
            // cannot write into freed memory after we return.
            // SAFETY: stream is valid; ignoring the sync error here is intentional —
            // we are already on an error path and can only do best-effort cleanup.
            let _ = unsafe { cuStreamSynchronize(stream) };
            return Err(NvJpegError::NvjpegStatus(status));
        }

        // Return the pinned buffer and metadata without copying yet.
        // The GPU is still writing; `decode_sync` synchronises the stream before
        // calling `into_decoded_jpeg` to copy the finished pixels into a Vec.
        Ok(PendingDecode {
            pinned,
            width,
            height,
            color_space,
        })
    }
}

impl Drop for NvJpeg {
    fn drop(&mut self) {
        // Ignore error codes on teardown — if the device is gone there is nothing
        // meaningful we can do, and panicking in Drop is unsound.
        //
        // SAFETY: handle and state are valid (new() asserts both non-null before
        // storing them); no other references exist at drop time.
        unsafe {
            let _ = nvjpegJpegStateDestroy(self.state);
            let _ = nvjpegDestroy(self.handle);
        }
    }
}

// ── NvJpegDecoder ─────────────────────────────────────────────────────────────

/// Safe, self-contained GPU JPEG decoder for use outside the `gpu` crate.
///
/// Manages its own CUDA primary context and stream via the raw CUDA driver API
/// (`libcuda.so`), completely independent of any higher-level CUDA wrapper such
/// as cudarc.  This is necessary because nvJPEG captures the CUDA context that
/// is current at `nvjpegCreateEx` time; mixing cudarc's context management with
/// nvJPEG's internal context causes `CUDA_ERROR_INVALID_CONTEXT (201)` on every
/// subsequent `cuStreamSynchronize`.
///
/// The verified initialisation sequence mirrors the C reference:
/// ```text
/// cuInit → cuDeviceGet → cuDevicePrimaryCtxRetain → cuCtxSetCurrent →
/// cuStreamCreate → NvJpeg::new() → (decode → cuStreamSynchronize)*
/// ```
///
/// `Send` but not `Sync`: each thread must own its own instance.
pub struct NvJpegDecoder {
    /// Wrapped in `ManuallyDrop` so we can explicitly drop it (calling
    /// nvjpegDestroy) before releasing the primary context in `Drop::drop`.
    dec: std::mem::ManuallyDrop<NvJpeg>,
    /// Raw CUDA device handle (i32 ordinal).
    device: i32,
    /// Retained primary context for `device`.  Owned by this struct; released in Drop.
    cu_ctx: *mut std::ffi::c_void,
    /// CUDA stream created in `cu_ctx`.  Owned; destroyed in Drop.
    stream: CUstream,
}

// SAFETY: NvJpeg is Send; the raw pointers are only accessed from the owning
// thread (decode_sync takes &mut self, which prevents concurrent access).
unsafe impl Send for NvJpegDecoder {}

impl NvJpegDecoder {
    /// Create an `NvJpegDecoder` on the given GPU device.
    ///
    /// `ordinal` is the GPU index (0 for the first GPU).  Follows the verified
    /// C initialisation sequence: `cuInit → cuDeviceGet →
    /// cuDevicePrimaryCtxRetain → cuCtxSetCurrent → cuStreamCreate → nvjpegCreateEx`.
    ///
    /// # Errors
    ///
    /// Returns an error if no CUDA device is available, or if any CUDA or
    /// nvJPEG initialisation step fails.
    ///
    /// # Panics
    ///
    /// Panics if `cuStreamCreate` or nvJPEG initialisation reports success but
    /// returns a null handle — this would indicate a driver bug.
    pub fn new(ordinal: usize) -> Result<Self> {
        // Step 1 — initialise the CUDA driver.  Safe to call multiple times.
        // SAFETY: flags must be 0.
        let r = unsafe { cuInit(0) };
        if r != 0 {
            return Err(NvJpegError::CudaError(r));
        }

        // Step 2 — get device handle.
        // cuDeviceGet takes a signed ordinal; real CUDA systems have at most a few
        // hundred GPUs, so anything outside i32 range is a caller error.
        // CUDA_ERROR_INVALID_DEVICE = 101.
        let ordinal_i32 = i32::try_from(ordinal).map_err(|_| NvJpegError::CudaError(101))?;
        let mut device: i32 = 0;
        let r = unsafe { cuDeviceGet(&raw mut device, ordinal_i32) };
        if r != 0 {
            return Err(NvJpegError::CudaError(r));
        }

        // Steps 3–4 — retain + bind the primary context.
        //
        // Retry up to 3 times with a short sleep: if a peer process crashed
        // without calling cuDevicePrimaryCtxRelease the driver marks the primary
        // context as "in recovery".  cuDevicePrimaryCtxRetain can return a
        // not-yet-fully-initialised context pointer in that window; probing it
        // with cuCtxGetApiVersion distinguishes a live context from a stale one.
        // After the driver finishes cleanup (typically < 200 ms) the next retain
        // succeeds cleanly.  Pattern confirmed by TensorFlow #51400 and Numba #2875.
        let mut cu_ctx: *mut std::ffi::c_void = ptr::null_mut();
        let mut ctx_ready = false;
        for attempt in 0..3_u32 {
            cu_ctx = ptr::null_mut();
            // SAFETY: cuDevicePrimaryCtxRetain writes to cu_ctx.
            let r = unsafe { cuDevicePrimaryCtxRetain(&raw mut cu_ctx, device) };
            if r != 0 {
                if attempt < 2 {
                    std::thread::sleep(std::time::Duration::from_millis(100));
                    continue;
                }
                return Err(NvJpegError::CudaError(r));
            }
            // Probe: a recovering context returns an error here even though Retain
            // succeeded and the pointer is non-null. We only inspect the return
            // code; the API-version out-param is discarded.
            let mut ver = std::mem::MaybeUninit::<u32>::uninit();
            // SAFETY: cu_ctx is non-null (Retain succeeded); cuCtxGetApiVersion is a
            // read-only probe that does not modify driver state. It writes a u32 to
            // the out-param when the call returns CUDA_SUCCESS, leaving it untouched
            // otherwise — we never read it either way.
            let probe = unsafe { cuCtxGetApiVersion(cu_ctx, ver.as_mut_ptr()) };
            if probe == 0 {
                ctx_ready = true;
                break;
            }
            log::debug!(
                "nvJPEG: primary context not ready (attempt {attempt}, probe={probe}), retrying"
            );
            // SAFETY: we just retained it; release before next attempt or final failure.
            let _ = unsafe { cuDevicePrimaryCtxRelease(device) };
            if attempt < 2 {
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
        }
        if !ctx_ready {
            // All three probe attempts failed; the context was never initialised.
            // CUDA_ERROR_NOT_INITIALIZED = 3.  The release already ran on the last iteration.
            return Err(NvJpegError::CudaError(3));
        }

        // Bind the now-verified primary context to the calling thread.
        let r = unsafe { cuCtxSetCurrent(cu_ctx) };
        if r != 0 {
            let _ = unsafe { cuDevicePrimaryCtxRelease(device) };
            return Err(NvJpegError::CudaError(r));
        }

        // Step 5 — create the stream IN THIS CONTEXT, before nvJPEG init.
        // Creating the stream first guarantees it belongs to the primary context
        // and not to any internal context that NvJpeg::new() might push.
        let mut stream: CUstream = ptr::null_mut();
        // SAFETY: cuStreamCreate writes to `stream`; flags = 0 is default.
        let r = unsafe { cuStreamCreate(&raw mut stream, 0) };
        if r != 0 {
            let _ = unsafe { cuDevicePrimaryCtxRelease(device) };
            return Err(NvJpegError::CudaError(r));
        }
        assert!(
            !stream.is_null(),
            "cuStreamCreate returned null despite success"
        );

        // Step 6 — initialise nvJPEG.  nvjpegCreateEx captures the current context
        // (set in step 4), so the handle and stream now share the same context.
        let dec = NvJpeg::new().inspect_err(|_| {
            // SAFETY: stream was successfully created; no work enqueued yet.
            unsafe {
                let _ = cuStreamDestroy(stream);
                let _ = cuDevicePrimaryCtxRelease(device);
            }
        })?;

        Ok(Self {
            dec: std::mem::ManuallyDrop::new(dec),
            device,
            cu_ctx,
            stream,
        })
    }

    /// Decode a JPEG bitstream synchronously, returning host-resident pixels.
    ///
    /// Rebinds the primary CUDA context to the calling thread, enqueues GPU
    /// work on the internal stream, blocks until the DMA into the pinned buffer
    /// is complete, then copies the finished pixels into a `Vec<u8>`.
    ///
    /// # Errors
    ///
    /// Returns an error if the JPEG is invalid, has unsupported dimensions,
    /// or if a CUDA API call fails.
    pub fn decode_sync(&mut self, data: &[u8]) -> Result<DecodedJpeg> {
        // Rebind the primary context — required when decode_sync is called from
        // a thread pool where threads change between calls.
        // SAFETY: cu_ctx is valid for the lifetime of this struct.
        let r = unsafe { cuCtxSetCurrent(self.cu_ctx) };
        if r != 0 {
            return Err(NvJpegError::CudaError(r));
        }

        let pending = self.dec.decode(data, self.stream)?;

        // Block until the GPU DMA into the pinned buffer is complete.
        // SAFETY: stream is valid; cu_ctx is current.
        let r = unsafe { cuStreamSynchronize(self.stream) };
        if r != 0 {
            return Err(NvJpegError::CudaError(r));
        }

        // Stream is fully synchronised — safe to copy pinned → Vec.
        Ok(pending.into_decoded_jpeg())
    }
}

impl Drop for NvJpegDecoder {
    fn drop(&mut self) {
        // Explicit teardown order matters: nvjpegDestroy must run while the CUDA
        // primary context is still current and our retain is still held.
        //
        // SAFETY: cu_ctx is valid; we hold the primary-context retain.
        unsafe {
            // 1. Bind the context so all CUDA calls below land in the right context.
            let _ = cuCtxSetCurrent(self.cu_ctx);

            // 2. Destroy the nvJPEG handle and state explicitly before releasing
            //    the context retain.  `dec` is ManuallyDrop so it does not run
            //    its own destructor automatically.
            std::mem::ManuallyDrop::drop(&mut self.dec);

            // 3. Destroy the stream (no work enqueued at drop time).
            if !self.stream.is_null() {
                let _ = cuStreamDestroy(self.stream);
            }

            // 4. Release our retain of the primary context.
            let _ = cuDevicePrimaryCtxRelease(self.device);
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// 16×16 grayscale JPEG (gray50, quality 80).
    /// Generated with: `convert -size 16x16 xc:gray50 -quality 80 /tmp/gray16x16.jpg && xxd -i`
    ///
    /// nvJPEG's GPU kernels require at least one full MCU block (8×8 pixels for
    /// baseline grayscale); a 1×1 JPEG triggers an internal segfault in the driver.
    const GRAY_16X16_JPEG: &[u8] = &[
        0xff, 0xd8, 0xff, 0xe0, 0x00, 0x10, 0x4a, 0x46, 0x49, 0x46, 0x00, 0x01, 0x01, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0xdb, 0x00, 0x43, 0x00, 0x06, 0x04, 0x05, 0x06, 0x05,
        0x04, 0x06, 0x06, 0x05, 0x06, 0x07, 0x07, 0x06, 0x08, 0x0a, 0x10, 0x0a, 0x0a, 0x09, 0x09,
        0x0a, 0x14, 0x0e, 0x0f, 0x0c, 0x10, 0x17, 0x14, 0x18, 0x18, 0x17, 0x14, 0x16, 0x16, 0x1a,
        0x1d, 0x25, 0x1f, 0x1a, 0x1b, 0x23, 0x1c, 0x16, 0x16, 0x20, 0x2c, 0x20, 0x23, 0x26, 0x27,
        0x29, 0x2a, 0x29, 0x19, 0x1f, 0x2d, 0x30, 0x2d, 0x28, 0x30, 0x25, 0x28, 0x29, 0x28, 0xff,
        0xc0, 0x00, 0x0b, 0x08, 0x00, 0x10, 0x00, 0x10, 0x01, 0x01, 0x11, 0x00, 0xff, 0xc4, 0x00,
        0x15, 0x00, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x01, 0xff, 0xc4, 0x00, 0x14, 0x10, 0x01, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0xda, 0x00,
        0x08, 0x01, 0x01, 0x00, 0x00, 0x3f, 0x00, 0x80, 0x3f, 0xff, 0xd9,
    ];

    /// `NvJpeg::new()` must not panic — it either succeeds or returns an error.
    /// Skipped gracefully on machines without a GPU.
    #[test]
    fn nvjpeg_new_does_not_panic() {
        let _ = NvJpeg::new(); // ignore Ok/Err
    }

    /// End-to-end decode of the embedded 16×16 gray JPEG via the public `NvJpegDecoder` API.
    /// Skipped if no GPU is available.
    #[test]
    fn decode_gray_16x16() {
        let mut dec = match NvJpegDecoder::new(0) {
            Ok(d) => d,
            Err(_) => return, // no GPU available
        };

        // decode_sync blocks until GPU work is complete.
        let img = dec
            .decode_sync(GRAY_16X16_JPEG)
            .expect("decode_sync failed");

        assert_eq!(img.width, 16);
        assert_eq!(img.height, 16);
        assert_eq!(img.color_space, JpegColorSpace::Gray);
        assert_eq!(img.data.len(), 16 * 16);
        // gray50 encodes to ~127.
        assert!(
            img.data.iter().all(|&p| p > 100 && p < 160),
            "unexpected luma values"
        );
    }

    /// `decode_sync` with an empty slice must return an error, not panic or UB.
    #[test]
    fn decode_empty_returns_error() {
        let mut dec = match NvJpegDecoder::new(0) {
            Ok(d) => d,
            Err(_) => return,
        };

        let result = dec.decode_sync(&[]);
        assert!(
            matches!(result, Err(NvJpegError::NvjpegStatus(_))),
            "expected NvjpegStatus error for empty input, got {result:?}",
        );
    }
}
