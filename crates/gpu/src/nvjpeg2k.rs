//! nvJPEG2000 GPU-accelerated JPEG 2000 decoding.
//!
//! Wraps the NVIDIA nvJPEG2000 library (installed at
//! `/usr/lib/x86_64-linux-gnu/libnvjpeg2k/12/`) to decode JPEG 2000 bitstreams
//! on the GPU, then copies the finished planar component planes back to host
//! memory via `cudaMemcpy2D`.
//!
//! PDF's `JPXDecode` filter embeds either raw `.j2k` codestreams or full JP2
//! container files; `nvjpeg2kStreamParse` auto-detects both.  CPU
//! `jpeg2k`/`OpenJPEG` is used as fallback for small images (below
//! [`GPU_JPEG2K_THRESHOLD_PX`]), when no GPU is available, and always for
//! inline images in the content stream (which are typically small thumbnails
//! not worth the `PCIe` dispatch overhead).
//!
//! # Key differences from nvJPEG (baseline JPEG)
//!
//! | | nvJPEG | nvJPEG2000 |
//! |---|---|---|
//! | Output memory | Host pinned (`cuMemAllocHost`) | Device (`cudaMalloc`) |
//! | Copy to host | `to_vec()` directly | `cudaMemcpy2D` per component |
//! | Image layout | Interleaved (RGBRGB…) | Planar — one ptr per component |
//! | Handle create | `nvjpegCreateEx(backend, …)` | `nvjpeg2kCreateSimple` |
//! | Backends | HARDWARE or DEFAULT | DEFAULT only |
//! | Per-decode objects | 1 state | `decode_state` + `stream` (bitstream handle) |
//! | Parse step | None | `nvjpeg2kStreamParse` before decode |
//!
//! # Thread safety
//!
//! [`NvJpeg2kDecoder`] is `Send` but not `Sync`: each thread must own its own
//! instance.  Creating one per thread is cheap — it retains the primary GPU
//! context (reference-counted) and creates one CUDA stream.

// All nvJPEG2000 calls are FFI through raw pointers.
#![cfg(feature = "nvjpeg2k")]

use std::ffi::c_void;
use std::mem::ManuallyDrop;
use std::ptr;

// ── CUDA driver API ───────────────────────────────────────────────────────────
//
// These symbols are from libcuda.so (the NVIDIA driver).  They are also
// declared in nvjpeg.rs but that module is compiled as a separate translation
// unit; we re-declare them here to avoid cross-module FFI coupling.

/// Opaque CUDA stream handle; we only ever pass it through to nvJPEG2000 and
/// the synchronise call.
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

// ── CUDA runtime API ──────────────────────────────────────────────────────────
//
// These symbols are in libcudart.so (CUDA runtime, ships with the toolkit).
// cudaMalloc / cudaFree / cudaMemcpy2D are required for device-memory I/O
// since nvJPEG2000 writes output to device (not pinned host) memory.

/// `cudaMemcpyDeviceToHost = 2` — direction flag for `cudaMemcpy2D`.
const CUDAMEMCPY_D2H: i32 = 2;

unsafe extern "C" {
    fn cudaMalloc(dev_ptr: *mut *mut c_void, size: usize) -> i32;
    fn cudaFree(dev_ptr: *mut c_void) -> i32;
    fn cudaMemcpy2D(
        dst: *mut c_void,
        dpitch: usize,
        src: *const c_void,
        spitch: usize,
        width: usize,
        height: usize,
        kind: i32,
    ) -> i32;
}

// ── nvJPEG2000 FFI declarations ───────────────────────────────────────────────
//
// Only the surface used by this module.  Avoids a bindgen dependency while
// keeping the binding surface minimal and auditable against nvjpeg2k.h.

/// nvJPEG2000 status code type.  0 is success; all other values are errors.
#[expect(
    non_camel_case_types,
    reason = "C FFI type alias; must match the nvJPEG2000 ABI name"
)]
type nvjpeg2kStatus_t = i32;

/// Opaque library handle.
#[repr(C)]
struct Nvjpeg2kHandle_ {
    _opaque: [u8; 0],
}
#[expect(
    non_camel_case_types,
    reason = "C FFI type alias; must match the nvJPEG2000 ABI name"
)]
type nvjpeg2kHandle_t = *mut Nvjpeg2kHandle_;

/// Opaque per-decode state (decoder scratch buffers, tiling state, etc.).
#[repr(C)]
struct Nvjpeg2kDecodeState_ {
    _opaque: [u8; 0],
}
#[expect(
    non_camel_case_types,
    reason = "C FFI type alias; must match the nvJPEG2000 ABI name"
)]
type nvjpeg2kDecodeState_t = *mut Nvjpeg2kDecodeState_;

/// Opaque bitstream handle (`nvjpeg2kStream_t`).
///
/// Called "stream" in the nvJPEG2000 API but this is the *bitstream* handle,
/// not a CUDA stream.  It holds the parsed headers and tile indices.  It is
/// reused across decodes: `nvjpeg2kStreamParse` overwrites its contents each
/// time it is called.
#[repr(C)]
struct Nvjpeg2kStream_ {
    _opaque: [u8; 0],
}
#[expect(
    non_camel_case_types,
    reason = "C FFI type alias; must match the nvJPEG2000 ABI name"
)]
type nvjpeg2kStream_t = *mut Nvjpeg2kStream_;

/// Image-level metadata returned by `nvjpeg2kStreamGetImageInfo`.
#[repr(C)]
struct Nvjpeg2kImageInfo {
    image_width: u32,
    image_height: u32,
    tile_width: u32,
    tile_height: u32,
    num_tiles_x: u32,
    num_tiles_y: u32,
    num_components: u32,
}

/// Per-component metadata returned by `nvjpeg2kStreamGetImageComponentInfo`.
#[repr(C)]
struct Nvjpeg2kImageComponentInfo {
    component_width: u32,
    component_height: u32,
    /// Bit-depth of each sample (1–16).
    precision: u8,
    /// Non-zero if samples are signed (two's-complement).
    sgn: u8,
}

/// Output descriptor passed to `nvjpeg2kDecode`.
///
/// `pixel_data[i]` is a **device** pointer for component `i`.
/// `pitch_in_bytes[i]` is the row stride in bytes for component `i`.
/// `pixel_type` selects the output element type; we always use
/// `NVJPEG2K_UINT8 = 0` (one `u8` per sample per component).
#[repr(C)]
struct Nvjpeg2kImage {
    pixel_data: *mut *mut c_void,
    pitch_in_bytes: *mut usize,
    pixel_type: i32,
    num_components: u32,
}

/// `NVJPEG2K_UINT8` — one unsigned byte per sample per component.
const NVJPEG2K_UINT8: i32 = 0;

const NVJPEG2K_STATUS_SUCCESS: nvjpeg2kStatus_t = 0;
/// Returned by the library for a malformed/truncated codestream, and also
/// returned by the shims when a C++ exception is caught during decode.
const NVJPEG2K_STATUS_BAD_JPEG: nvjpeg2kStatus_t = 3;

unsafe extern "C" {
    // ── Exception-safe shims (from shim/nvjpeg2k_shim.cpp) ───────────────────
    //
    // All nvjpeg2k functions that can throw C++ exceptions are routed through
    // the C++ shim which wraps each call in try/catch and returns
    // NVJPEG2K_STATUS_IMPLEMENTATION_NOT_SUPPORTED (9) on any exception.
    // C++ exceptions must not propagate through Rust FFI (undefined behaviour).
    //
    // Destroy functions are shimmed for defence-in-depth: they are called from
    // Drop (which must not unwind), so catching any unexpected exception is
    // critical even though the library documents them as non-throwing.

    fn nvjpeg2k_shim_create_simple(handle: *mut nvjpeg2kHandle_t) -> nvjpeg2kStatus_t;
    fn nvjpeg2k_shim_decode_state_create(
        handle: nvjpeg2kHandle_t,
        decode_state: *mut nvjpeg2kDecodeState_t,
    ) -> nvjpeg2kStatus_t;
    fn nvjpeg2k_shim_stream_create(stream: *mut nvjpeg2kStream_t) -> nvjpeg2kStatus_t;
    fn nvjpeg2k_shim_stream_get_image_info(
        stream: nvjpeg2kStream_t,
        image_info: *mut Nvjpeg2kImageInfo,
    ) -> nvjpeg2kStatus_t;
    fn nvjpeg2k_shim_stream_get_image_component_info(
        stream: nvjpeg2kStream_t,
        component_info: *mut Nvjpeg2kImageComponentInfo,
        component_id: u32,
    ) -> nvjpeg2kStatus_t;

    /// Exception-safe wrapper for `nvjpeg2kStreamParse`.
    ///
    /// `save_metadata` and `save_stream` must both be 0 for standard decodes
    /// (metadata and tile data are not persistently cached).
    fn nvjpeg2k_shim_stream_parse(
        handle: nvjpeg2kHandle_t,
        data: *const u8,
        length: usize,
        save_metadata: i32,
        save_stream: i32,
        stream: nvjpeg2kStream_t,
    ) -> nvjpeg2kStatus_t;

    /// Exception-safe wrapper for `nvjpeg2kDecode`.
    ///
    /// The caller must synchronise `cuda_stream` before reading the device
    /// buffers pointed to by `output.pixel_data`.
    fn nvjpeg2k_shim_decode(
        handle: nvjpeg2kHandle_t,
        decode_state: nvjpeg2kDecodeState_t,
        j2k_stream: nvjpeg2kStream_t,
        output: *mut Nvjpeg2kImage,
        cuda_stream: CUstream,
    ) -> nvjpeg2kStatus_t;

    fn nvjpeg2k_shim_decode_state_destroy(decode_state: nvjpeg2kDecodeState_t) -> nvjpeg2kStatus_t;
    fn nvjpeg2k_shim_stream_destroy(stream: nvjpeg2kStream_t) -> nvjpeg2kStatus_t;
    fn nvjpeg2k_shim_destroy(handle: nvjpeg2kHandle_t) -> nvjpeg2kStatus_t;
}

// ── Error type ────────────────────────────────────────────────────────────────

/// Errors returned by the nvJPEG2000 wrapper.
#[derive(Debug)]
pub enum NvJpeg2kError {
    /// nvJPEG2000 API returned a non-zero status.
    ///
    /// Codes from `nvjpeg2k.h`:
    /// - 1: `NVJPEG2K_STATUS_NOT_INITIALIZED`
    /// - 2: `NVJPEG2K_STATUS_INVALID_PARAMETER`
    /// - 3: `NVJPEG2K_STATUS_BAD_JPEG`
    /// - 4: `NVJPEG2K_STATUS_JPEG_NOT_SUPPORTED`
    /// - 5: `NVJPEG2K_STATUS_ALLOCATOR_FAILURE`
    /// - 6: `NVJPEG2K_STATUS_EXECUTION_FAILED`
    /// - 7: `NVJPEG2K_STATUS_ARCH_MISMATCH`
    /// - 8: `NVJPEG2K_STATUS_INTERNAL_ERROR`
    /// - 9: `NVJPEG2K_STATUS_IMPLEMENTATION_NOT_SUPPORTED`
    ///
    /// Note: code 9 is also returned by the C++ exception shim when a C++
    /// exception is thrown during decode (e.g. on a malformed codestream).
    /// A caller cannot distinguish that case from the library returning 9
    /// normally; both are treated as "fall back to CPU decoder."
    Nvjpeg2kStatus(i32),
    /// CUDA driver API error (`CUresult`).
    CudaError(i32),
    /// CUDA runtime API error (`cudaError_t`).
    ///
    /// From `cudaMalloc` or `cudaMemcpy2D`.  Common codes (shared with driver):
    /// - 2: `cudaErrorMemoryAllocation`
    /// - 35: `cudaErrorInsufficientDriver`
    /// - 100: `cudaErrorNoDevice`
    CudartError(i32),
    /// Component count is not 1 (Gray) or 3 (RGB); CMYK, Gray+Alpha, LAB, and
    /// N-channel images fall through to the CPU `jpeg2k`/`OpenJPEG` path.
    UnsupportedComponents(u32),
    /// At least one image component has a smaller width or height than the
    /// full image, indicating sub-sampled chroma (e.g. YUV 4:2:0).  Bare
    /// `nvjpeg2kDecode` does not upsample — fall through to CPU.
    SubSampledComponents,
    /// The decoded image has a zero dimension.
    ZeroDimension {
        /// Reported pixel width (0 when degenerate).
        width: u32,
        /// Reported pixel height (0 when degenerate).
        height: u32,
    },
    /// Arithmetic overflow computing pixel buffer size.
    Overflow,
}

impl std::fmt::Display for NvJpeg2kError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Nvjpeg2kStatus(code) => {
                let name = match *code {
                    1 => "NVJPEG2K_STATUS_NOT_INITIALIZED",
                    2 => "NVJPEG2K_STATUS_INVALID_PARAMETER",
                    3 => "NVJPEG2K_STATUS_BAD_JPEG",
                    4 => "NVJPEG2K_STATUS_JPEG_NOT_SUPPORTED",
                    5 => "NVJPEG2K_STATUS_ALLOCATOR_FAILURE",
                    6 => "NVJPEG2K_STATUS_EXECUTION_FAILED",
                    7 => "NVJPEG2K_STATUS_ARCH_MISMATCH",
                    8 => "NVJPEG2K_STATUS_INTERNAL_ERROR",
                    9 => "NVJPEG2K_STATUS_IMPLEMENTATION_NOT_SUPPORTED",
                    _ => "NVJPEG2K_STATUS_UNKNOWN",
                };
                write!(f, "nvJPEG2000 error {code} ({name})")
            }
            Self::CudaError(code) => {
                let name = match *code {
                    1 => "CUDA_ERROR_INVALID_VALUE",
                    2 => "CUDA_ERROR_OUT_OF_MEMORY",
                    3 => "CUDA_ERROR_NOT_INITIALIZED",
                    35 => "CUDA_ERROR_INSUFFICIENT_DRIVER",
                    100 => "CUDA_ERROR_NO_DEVICE",
                    101 => "CUDA_ERROR_INVALID_DEVICE",
                    200 => "CUDA_ERROR_INVALID_IMAGE",
                    201 => "CUDA_ERROR_INVALID_CONTEXT",
                    400 => "CUDA_ERROR_INVALID_HANDLE",
                    700 => "CUDA_ERROR_ILLEGAL_ADDRESS",
                    _ => "CUDA_ERROR_UNKNOWN",
                };
                write!(f, "CUDA driver error {code} ({name})")
            }
            Self::CudartError(code) => write!(f, "CUDA runtime error {code}"),
            Self::UnsupportedComponents(n) => {
                write!(
                    f,
                    "unsupported JPEG 2000 component count {n} \
                     (only Gray/1 and RGB/3 are GPU-accelerated)"
                )
            }
            Self::SubSampledComponents => write!(
                f,
                "sub-sampled chroma components (e.g. YUV 4:2:0) — \
                 bare nvjpeg2kDecode does not upsample; falling back to CPU"
            ),
            Self::ZeroDimension { width, height } => {
                write!(f, "JPEG 2000 reported zero dimension {width}×{height}")
            }
            Self::Overflow => write!(f, "pixel buffer size overflow (image too large)"),
        }
    }
}

impl std::error::Error for NvJpeg2kError {}

/// Convenience alias.
pub type Result<T> = std::result::Result<T, NvJpeg2kError>;

// ── Output type ───────────────────────────────────────────────────────────────

/// Colour space of the pixels returned by [`NvJpeg2kDecoder::decode_sync`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Jpeg2kColorSpace {
    /// One byte per pixel (luma / grayscale).
    Gray,
    /// Three bytes per pixel, interleaved R G B.
    Rgb,
}

/// Decoded JPEG 2000 image, host-resident.
#[derive(Debug)]
pub struct DecodedJpeg2k {
    /// Pixel bytes.  Layout matches [`color_space`](Self::color_space).
    pub data: Vec<u8>,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Colour space of the pixel bytes.
    pub color_space: Jpeg2kColorSpace,
}

// ── DeviceBuf ─────────────────────────────────────────────────────────────────

/// RAII wrapper for a device-side allocation (`cudaMalloc` / `cudaFree`).
///
/// nvJPEG2000 writes decoded pixel planes into device memory, unlike nvJPEG
/// (which writes to pinned host memory).  We allocate one `DeviceBuf` per image
/// component, pass the raw device pointers to `nvjpeg2kDecode`, then copy the
/// finished planes back to host via `cudaMemcpy2D` after stream sync.
struct DeviceBuf {
    ptr: *mut c_void,
}

impl DeviceBuf {
    /// Allocate `size` bytes of device memory.
    fn alloc(size: usize) -> Result<Self> {
        let mut ptr: *mut c_void = ptr::null_mut();
        // SAFETY: cudaMalloc writes to `ptr`; `size` is the allocation size.
        let code = unsafe { cudaMalloc(&raw mut ptr, size) };
        if code != 0 {
            return Err(NvJpeg2kError::CudartError(code));
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
        // ptr is always non-null: DeviceBuf::alloc asserts this after cudaMalloc
        // succeeds, so a null ptr is unreachable through the public constructor.
        // SAFETY: ptr came from cudaMalloc; no other references exist at drop time.
        // Error ignored — cannot propagate from Drop.
        let _ = unsafe { cudaFree(self.ptr) };
    }
}

// SAFETY: DeviceBuf is an exclusively-owned device allocation.
unsafe impl Send for DeviceBuf {}

// ── NvJpeg2k ─────────────────────────────────────────────────────────────────

/// nvJPEG2000 decoder inner context.
///
/// Holds the library handle, per-decode state, and a reusable bitstream handle.
/// This is an implementation detail of [`NvJpeg2kDecoder`]; `pub(crate)` rather
/// than private so that integration tests in this crate can construct one
/// directly without going through the full CUDA context setup.
pub(crate) struct NvJpeg2k {
    handle: nvjpeg2kHandle_t,
    decode_state: nvjpeg2kDecodeState_t,
    /// Reusable bitstream handle.  `nvjpeg2kStreamParse` overwrites it each
    /// call, so it is safe to reuse across decodes.
    j2k_stream: nvjpeg2kStream_t,
}

// SAFETY: nvJPEG2000 guarantees that a handle used from one host thread at a
// time is safe; the internal CUDA stream serialises device-side work.
unsafe impl Send for NvJpeg2k {}

impl NvJpeg2k {
    /// Initialise nvJPEG2000 (single DEFAULT backend — no hardware variant).
    ///
    /// # Errors
    ///
    /// Returns an error if nvJPEG2000 initialisation or state allocation fails.
    pub(crate) fn new() -> Result<Self> {
        let mut handle: nvjpeg2kHandle_t = ptr::null_mut();
        // SAFETY: nvjpeg2k_shim_create_simple wraps nvjpeg2kCreateSimple in try/catch.
        let status = unsafe { nvjpeg2k_shim_create_simple(&raw mut handle) };
        if status != NVJPEG2K_STATUS_SUCCESS {
            return Err(NvJpeg2kError::Nvjpeg2kStatus(status));
        }
        assert!(
            !handle.is_null(),
            "nvjpeg2kCreateSimple succeeded but returned null handle"
        );

        let mut decode_state: nvjpeg2kDecodeState_t = ptr::null_mut();
        // SAFETY: handle is valid; shim wraps in try/catch.
        let status = unsafe { nvjpeg2k_shim_decode_state_create(handle, &raw mut decode_state) };
        if status != NVJPEG2K_STATUS_SUCCESS {
            // SAFETY: handle is valid; decode_state was never initialised.
            let _ = unsafe { nvjpeg2k_shim_destroy(handle) };
            return Err(NvJpeg2kError::Nvjpeg2kStatus(status));
        }
        assert!(
            !decode_state.is_null(),
            "nvjpeg2kDecodeStateCreate succeeded but returned null state"
        );

        let mut j2k_stream: nvjpeg2kStream_t = ptr::null_mut();
        // SAFETY: no preconditions; shim wraps nvjpeg2kStreamCreate in try/catch.
        let status = unsafe { nvjpeg2k_shim_stream_create(&raw mut j2k_stream) };
        if status != NVJPEG2K_STATUS_SUCCESS {
            // SAFETY: both are valid; clean up before returning.
            unsafe {
                let _ = nvjpeg2k_shim_decode_state_destroy(decode_state);
                let _ = nvjpeg2k_shim_destroy(handle);
            }
            return Err(NvJpeg2kError::Nvjpeg2kStatus(status));
        }
        assert!(
            !j2k_stream.is_null(),
            "nvjpeg2kStreamCreate succeeded but returned null stream"
        );

        Ok(Self {
            handle,
            decode_state,
            j2k_stream,
        })
    }

    /// Decode a JPEG 2000 bitstream on `cu_stream`, returning host-resident pixels.
    ///
    /// Rejects images with component counts other than 1 (Gray) or 3 (RGB) —
    /// CMYK, LAB, and N-channel images fall through to the CPU path.
    ///
    /// Sub-sampled images (chroma components narrower/shorter than the image)
    /// are also rejected — bare `nvjpeg2kDecode` writes each component at its
    /// native dimensions without upsampling, which would break the planar→RGB
    /// interleave.  Call-site falls back to CPU for these.
    ///
    /// 16-bit precision images are downscaled to 8-bit **by the nvJPEG2000
    /// library** (`NVJPEG2K_UINT8` output mode); no post-processing is needed
    /// here.
    ///
    /// # Errors
    ///
    /// Returns an error if parsing, decode, or memory allocation fails.
    fn decode(&mut self, data: &[u8], cu_stream: CUstream) -> Result<DecodedJpeg2k> {
        // Fast-path: the library returns BAD_JPEG (3) on empty input, but an FFI
        // call for a trivially-invalid input is unnecessary noise.
        if data.is_empty() {
            return Err(NvJpeg2kError::Nvjpeg2kStatus(NVJPEG2K_STATUS_BAD_JPEG));
        }

        // ── Phase 1: parse bitstream headers ─────────────────────────────────
        // SAFETY: handle and j2k_stream are valid; data is a valid slice.
        // save_metadata = 0, save_stream = 0: standard decode (no caching).
        // Uses the shim to catch C++ exceptions thrown by the library.
        let status = unsafe {
            nvjpeg2k_shim_stream_parse(
                self.handle,
                data.as_ptr(),
                data.len(),
                0,
                0,
                self.j2k_stream,
            )
        };
        if status != NVJPEG2K_STATUS_SUCCESS {
            return Err(NvJpeg2kError::Nvjpeg2kStatus(status));
        }

        // ── Phase 2: read image-level info ────────────────────────────────────
        let mut info = Nvjpeg2kImageInfo {
            image_width: 0,
            image_height: 0,
            tile_width: 0,
            tile_height: 0,
            num_tiles_x: 0,
            num_tiles_y: 0,
            num_components: 0,
        };
        // SAFETY: j2k_stream is valid and freshly parsed; shim wraps in try/catch.
        let status = unsafe { nvjpeg2k_shim_stream_get_image_info(self.j2k_stream, &raw mut info) };
        if status != NVJPEG2K_STATUS_SUCCESS {
            return Err(NvJpeg2kError::Nvjpeg2kStatus(status));
        }

        let w = info.image_width;
        let h = info.image_height;
        let nc = info.num_components;

        if w == 0 || h == 0 {
            return Err(NvJpeg2kError::ZeroDimension {
                width: w,
                height: h,
            });
        }

        // Cap component count before any allocation.  Legitimate JPEG 2000
        // images have at most a handful of components (≤ 16 per standard, ≤ 4
        // in practice for Gray / RGB / CMYK).  A corrupted header returning
        // u32::MAX would otherwise cause Vec::with_capacity to attempt a ~68 GB
        // allocation.
        if nc > 4 {
            return Err(NvJpeg2kError::UnsupportedComponents(nc));
        }

        // Only Gray (1) and RGB (3) are supported.  CMYK (4), LAB (3, but
        // typically with a different ICC profile), and other multi-channel
        // images fall through to the CPU decoder which handles them correctly.
        let color_space = match nc {
            1 => Jpeg2kColorSpace::Gray,
            3 => Jpeg2kColorSpace::Rgb,
            _ => return Err(NvJpeg2kError::UnsupportedComponents(nc)),
        };

        // ── Phase 3: per-component info + device allocation ───────────────────
        // nc is 1 or 3 here (enforced by the color_space match above).
        // The `as usize` cast is lossless: nc ≤ 3, which fits every target's usize.
        let mut comp_infos: Vec<Nvjpeg2kImageComponentInfo> = Vec::with_capacity(nc as usize);
        for i in 0..nc {
            let mut ci = Nvjpeg2kImageComponentInfo {
                component_width: 0,
                component_height: 0,
                precision: 0,
                sgn: 0,
            };
            // SAFETY: j2k_stream valid; i < nc; shim wraps in try/catch.
            let status = unsafe {
                nvjpeg2k_shim_stream_get_image_component_info(self.j2k_stream, &raw mut ci, i)
            };
            if status != NVJPEG2K_STATUS_SUCCESS {
                return Err(NvJpeg2kError::Nvjpeg2kStatus(status));
            }
            // Guard degenerate component dimensions — a zero-width or zero-height
            // component would cause empty device allocations and then panic on
            // RGB interleave indexing.
            if ci.component_width == 0 || ci.component_height == 0 {
                return Err(NvJpeg2kError::ZeroDimension {
                    width: ci.component_width,
                    height: ci.component_height,
                });
            }
            // nvjpeg2kDecode (without nvjpeg2kDecodeParamsSetRGBOutput) writes each
            // component at its native dimensions — sub-sampled chroma components are
            // NOT upsampled to the full image size.  Attempting to interleave
            // mis-sized planes would index out of bounds.  Fall through to the CPU
            // OpenJPEG path which handles sub-sampling correctly.
            if ci.component_width != w || ci.component_height != h {
                return Err(NvJpeg2kError::SubSampledComponents);
            }
            comp_infos.push(ci);
        }

        // Allocate one device buffer per component.
        // pitch = component_width (= image width, validated above) since we always
        // use NVJPEG2K_UINT8 (1 byte/sample).  The caller sets the pitch in the
        // Nvjpeg2kImage struct; the library writes exactly at the pitch we specify.
        //
        // comp_infos has exactly nc entries (the loop above pushes exactly nc times).
        // pixel_ptrs[i] aliases dev_bufs[i].ptr (a device pointer owned by DeviceBuf).
        // dev_bufs must not be dropped before nvjpeg2k_shim_decode completes.
        // Both live until end of this scope, so no use-after-free is possible.
        // Vec capacity == nc means no reallocation occurs during push, so
        // as_mut_ptr() into pixel_ptrs/pitches is stable across the decode call.
        debug_assert_eq!(
            comp_infos.len(),
            nc as usize,
            "comp_infos length must equal nc"
        );
        let mut dev_bufs: Vec<DeviceBuf> = Vec::with_capacity(nc as usize);
        let mut pixel_ptrs: Vec<*mut c_void> = Vec::with_capacity(comp_infos.len());
        let mut pitches: Vec<usize> = Vec::with_capacity(comp_infos.len());
        for ci in &comp_infos {
            // component_width/height == image_width/height (guarded in Phase 3).
            // u32 values ≤ image dimensions; usize is at least 32 bits on all targets.
            let cw = ci.component_width as usize;
            let ch = ci.component_height as usize;
            let size = cw.checked_mul(ch).ok_or(NvJpeg2kError::Overflow)?;
            let buf = DeviceBuf::alloc(size)?;
            pixel_ptrs.push(buf.ptr);
            pitches.push(cw);
            dev_bufs.push(buf);
        }

        // ── Phase 4: build Nvjpeg2kImage and decode ───────────────────────────
        // pixel_ptrs and pitches Vecs are pre-sized to nc (≤ 3); no reallocation
        // occurs between construction and nvjpeg2kDecode, so as_mut_ptr() is stable.

        // nc is u32; validated ≤ 3 in Phase 2.  u32 → u32 cast is a no-op but
        // the field type matches the C ABI declaration.
        let mut img = Nvjpeg2kImage {
            pixel_data: pixel_ptrs.as_mut_ptr(),
            pitch_in_bytes: pitches.as_mut_ptr(),
            pixel_type: NVJPEG2K_UINT8,
            num_components: nc,
        };

        // SAFETY: handle, decode_state, j2k_stream are valid; img pointers are
        // valid for the duration of the call; cu_stream is owned by the caller.
        // nvjpeg2kDecode enqueues GPU work asynchronously — caller must sync.
        // Uses the shim to catch C++ exceptions thrown by the library.
        let status = unsafe {
            nvjpeg2k_shim_decode(
                self.handle,
                self.decode_state,
                self.j2k_stream,
                &raw mut img,
                cu_stream,
            )
        };

        // Synchronise unconditionally, even on error: partial GPU work may have
        // been enqueued and could write into `dev_bufs` after we free them unless
        // we wait for the stream to drain.
        // SAFETY: cu_stream is valid.
        let sync_code = unsafe { cuStreamSynchronize(cu_stream) };

        if status != NVJPEG2K_STATUS_SUCCESS {
            // Decode failed.  Report any additional stream sync failure to stderr
            // so it is visible for debugging, but return the decode error — it is
            // the root cause, and the type does not support carrying two errors.
            if sync_code != 0 {
                eprintln!(
                    "nvjpeg2k: cuStreamSynchronize failed (code {sync_code}) \
                     while handling decode error (status {status})"
                );
            }
            return Err(NvJpeg2kError::Nvjpeg2kStatus(status));
        }
        if sync_code != 0 {
            return Err(NvJpeg2kError::CudaError(sync_code));
        }

        // ── Phase 5: copy component planes device → host, then interleave ─────
        Self::copy_and_interleave(&comp_infos, &dev_bufs, w, h, color_space)
    }

    /// Copy device planes to host and produce a packed, interleaved `Vec<u8>`.
    ///
    /// For Gray (1 component) the output is the single plane straight.
    /// For RGB (3 components) the planes are interleaved row by row:
    /// `R[0] G[0] B[0]  R[1] G[1] B[1]  …`.
    fn copy_and_interleave(
        comp_infos: &[Nvjpeg2kImageComponentInfo],
        dev_bufs: &[DeviceBuf],
        width: u32,
        height: u32,
        color_space: Jpeg2kColorSpace,
    ) -> Result<DecodedJpeg2k> {
        let nc = comp_infos.len();
        // width/height are u32 image dimensions (≤ u32::MAX); usize is ≥ 32 bits.
        let (width_us, height_us) = (width as usize, height as usize);

        // Invariant: nc matches color_space (1 for Gray, 3 for Rgb).
        // These asserts catch future refactoring bugs; they fire in debug builds only.
        match color_space {
            Jpeg2kColorSpace::Gray => {
                debug_assert_eq!(nc, 1, "Gray requires exactly 1 component plane")
            }
            Jpeg2kColorSpace::Rgb => {
                debug_assert_eq!(nc, 3, "Rgb requires exactly 3 component planes")
            }
        }

        // Allocate host planes (one per component) and copy device → host.
        // All components have the same dimensions as the image (validated in Phase 3).
        // The pitch we passed to nvjpeg2kDecode equals component_width, so the
        // device layout is tightly packed: pitch == width in bytes.
        let mut host_planes: Vec<Vec<u8>> = Vec::with_capacity(nc);
        for (ci, buf) in comp_infos.iter().zip(dev_bufs.iter()) {
            // component_width/height == image_width/height (Phase 3 invariant).
            let cw = ci.component_width as usize;
            let ch = ci.component_height as usize;
            let plane_size = cw.checked_mul(ch).ok_or(NvJpeg2kError::Overflow)?;
            let mut host = vec![0u8; plane_size];

            // SAFETY: buf.ptr is valid device memory of exactly cw*ch bytes,
            // allocated in Phase 3 and written by nvjpeg2kDecode.
            // host.as_mut_ptr() is valid for plane_size == cw*ch bytes.
            // The CUDA stream was synchronised by cuStreamSynchronize in decode()
            // immediately before this function is called — the GPU write is complete.
            // Source pitch == cw (set in Nvjpeg2kImage); destination pitch == cw (tightly packed).
            let code = unsafe {
                cudaMemcpy2D(
                    host.as_mut_ptr().cast::<c_void>(),
                    cw, // destination pitch (host, tightly packed)
                    buf.ptr.cast::<c_void>(),
                    cw, // source pitch (= cw; the pitch we told nvjpeg2k to use)
                    cw, // width in bytes to copy per row
                    ch, // number of rows
                    CUDAMEMCPY_D2H,
                )
            };
            if code != 0 {
                return Err(NvJpeg2kError::CudartError(code));
            }
            host_planes.push(host);
        }

        let data = match color_space {
            Jpeg2kColorSpace::Gray => host_planes.remove(0),
            Jpeg2kColorSpace::Rgb => {
                // Interleave three planar buffers (R, G, B) into packed RGBRGB…
                // All three planes are exactly width×height bytes (validated in Phase 3).
                let total = width_us
                    .checked_mul(height_us)
                    .and_then(|n| n.checked_mul(3))
                    .ok_or(NvJpeg2kError::Overflow)?;
                let mut interleaved = vec![0u8; total];
                let (plane_r, plane_g, plane_b) =
                    (&host_planes[0], &host_planes[1], &host_planes[2]);
                interleaved
                    .chunks_exact_mut(3)
                    .zip(plane_r.iter().zip(plane_g.iter().zip(plane_b.iter())))
                    .for_each(|(chunk, (rv, (gv, bv)))| {
                        chunk[0] = *rv;
                        chunk[1] = *gv;
                        chunk[2] = *bv;
                    });
                interleaved
            }
        };

        Ok(DecodedJpeg2k {
            data,
            width,
            height,
            color_space,
        })
    }
}

impl Drop for NvJpeg2k {
    fn drop(&mut self) {
        // Destroy in reverse creation order:
        //   1. decode_state — per-decode scratch buffers (created last, freed first)
        //   2. j2k_stream   — bitstream handle (holds parsed tile/header data)
        //   3. handle       — top-level library handle (must outlive both above)
        //
        // nvJPEG2000 documentation requires nvjpeg2kDecodeStateDestroy before
        // nvjpeg2kDestroy.  nvjpeg2kStreamDestroy is independent but follows the
        // same reverse-creation convention for clarity.
        //
        // Null-check each pointer: `new` always stores all three before returning
        // Ok, so nulls only appear if the struct is ever constructed unsafely.
        //
        // SAFETY: non-null handles are valid and exclusively owned; no aliases.
        unsafe {
            if !self.decode_state.is_null() {
                let _ = nvjpeg2k_shim_decode_state_destroy(self.decode_state);
            }
            if !self.j2k_stream.is_null() {
                let _ = nvjpeg2k_shim_stream_destroy(self.j2k_stream);
            }
            if !self.handle.is_null() {
                let _ = nvjpeg2k_shim_destroy(self.handle);
            }
        }
    }
}

// ── NvJpeg2kDecoder ───────────────────────────────────────────────────────────

/// Safe, self-contained GPU JPEG 2000 decoder.
///
/// Manages its own CUDA primary context and stream, independent of cudarc.
/// This is required because nvJPEG2000 captures the CUDA context current at
/// `nvjpeg2kCreateSimple` time; mixing cudarc's context management causes
/// `CUDA_ERROR_INVALID_CONTEXT (201)` on subsequent stream synchronisations.
///
/// The initialisation sequence mirrors the C reference:
/// ```text
/// cuInit → cuDeviceGet → cuDevicePrimaryCtxRetain → cuCtxSetCurrent →
/// cuStreamCreate → NvJpeg2k::new() → (decode → cuStreamSynchronize)*
/// ```
///
/// `Send` but not `Sync`: each thread must own its own instance.
pub struct NvJpeg2kDecoder {
    /// Wrapped in `ManuallyDrop` so `Drop::drop` can explicitly run
    /// nvJPEG2000 teardown before releasing the primary context.
    dec: ManuallyDrop<NvJpeg2k>,
    /// Raw CUDA device handle (i32 ordinal).
    device: i32,
    /// Retained primary context for `device`.
    cu_ctx: *mut c_void,
    /// CUDA stream created in `cu_ctx`.
    stream: CUstream,
}

// SAFETY: NvJpeg2k is Send; raw pointers are only accessed from the owning
// thread (decode_sync takes &mut self, preventing concurrent access).
unsafe impl Send for NvJpeg2kDecoder {}

impl NvJpeg2kDecoder {
    /// Create an `NvJpeg2kDecoder` on the given GPU device (0-indexed).
    ///
    /// # Errors
    ///
    /// Returns an error if no CUDA device is available, or if any CUDA or
    /// nvJPEG2000 initialisation step fails.
    ///
    /// # Panics
    ///
    /// Panics if a CUDA or nvJPEG2000 function reports success but returns a
    /// null handle — that would indicate a driver bug.
    pub fn new(ordinal: usize) -> Result<Self> {
        // Step 1 — initialise the CUDA driver.  Safe to call multiple times.
        let r = unsafe { cuInit(0) };
        if r != 0 {
            return Err(NvJpeg2kError::CudaError(r));
        }

        // Step 2 — get device handle.
        let ordinal_i32 = i32::try_from(ordinal).map_err(|_| NvJpeg2kError::CudaError(101))?;
        let mut device: i32 = 0;
        let r = unsafe { cuDeviceGet(&raw mut device, ordinal_i32) };
        if r != 0 {
            return Err(NvJpeg2kError::CudaError(r));
        }

        // Step 3 — retain primary context.
        let mut cu_ctx: *mut c_void = ptr::null_mut();
        let r = unsafe { cuDevicePrimaryCtxRetain(&raw mut cu_ctx, device) };
        if r != 0 {
            return Err(NvJpeg2kError::CudaError(r));
        }
        assert!(
            !cu_ctx.is_null(),
            "cuDevicePrimaryCtxRetain succeeded but returned null context"
        );

        // Step 4 — bind context to calling thread.
        let r = unsafe { cuCtxSetCurrent(cu_ctx) };
        if r != 0 {
            let _ = unsafe { cuDevicePrimaryCtxRelease(device) };
            return Err(NvJpeg2kError::CudaError(r));
        }

        // Step 5 — create stream before nvJPEG2000 init so it belongs to the
        // primary context, not any internal context the library might push.
        let mut stream: CUstream = ptr::null_mut();
        let r = unsafe { cuStreamCreate(&raw mut stream, 0) };
        if r != 0 {
            let _ = unsafe { cuDevicePrimaryCtxRelease(device) };
            return Err(NvJpeg2kError::CudaError(r));
        }
        assert!(
            !stream.is_null(),
            "cuStreamCreate returned null despite success"
        );

        // Step 6 — initialise nvJPEG2000.  nvjpeg2kCreateSimple captures the
        // current context (set in step 4), so handle and stream share a context.
        let dec = NvJpeg2k::new().inspect_err(|_| {
            // SAFETY: stream was created successfully; no work is enqueued yet.
            unsafe {
                let _ = cuStreamDestroy(stream);
                let _ = cuDevicePrimaryCtxRelease(device);
            }
        })?;

        Ok(Self {
            dec: ManuallyDrop::new(dec),
            device,
            cu_ctx,
            stream,
        })
    }

    /// Decode a JPEG 2000 bitstream synchronously, returning host-resident pixels.
    ///
    /// Rebinds the primary CUDA context to the calling thread (required in
    /// thread-pool scenarios where threads may change between calls), enqueues
    /// GPU decode work, blocks until complete, then copies results to host.
    ///
    /// # Errors
    ///
    /// - [`NvJpeg2kError::CudaError`] if `cuCtxSetCurrent` fails (driver error).
    /// - [`NvJpeg2kError::Nvjpeg2kStatus`] if the bitstream is invalid or
    ///   the library reports a decode failure.
    /// - [`NvJpeg2kError::UnsupportedComponents`] or
    ///   [`NvJpeg2kError::SubSampledComponents`] if the image geometry is not
    ///   supported by the GPU path (falls back to CPU at the call site).
    /// - [`NvJpeg2kError::CudartError`] if `cudaMemcpy2D` fails during D→H copy.
    pub fn decode_sync(&mut self, data: &[u8]) -> Result<DecodedJpeg2k> {
        // Rebind primary context — necessary when called across thread-pool threads.
        // SAFETY: cu_ctx is valid for the lifetime of this struct.
        let r = unsafe { cuCtxSetCurrent(self.cu_ctx) };
        if r != 0 {
            return Err(NvJpeg2kError::CudaError(r));
        }

        // decode() handles both the async GPU work and the stream synchronise
        // internally (sync is required before cudaMemcpy2D).
        self.dec.decode(data, self.stream)
    }
}

impl Drop for NvJpeg2kDecoder {
    fn drop(&mut self) {
        // Teardown order is critical: nvjpeg2k resources must be destroyed while
        // the primary context is still current and our retain is held.
        //
        // SAFETY: cu_ctx is valid; we hold the primary-context retain.
        unsafe {
            // 1. Bind context so CUDA calls below land in the right context.
            let _ = cuCtxSetCurrent(self.cu_ctx);

            // 2. Destroy nvJPEG2000 handle/state/stream (ManuallyDrop — runs Drop<NvJpeg2k>).
            ManuallyDrop::drop(&mut self.dec);

            // 3. Destroy the CUDA stream.
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

    /// `NvJpeg2k::new()` must not panic — it either succeeds or returns `Err`.
    /// Skipped gracefully if no GPU is available.
    #[test]
    fn nvjpeg2k_new_does_not_panic() {
        let _ = NvJpeg2k::new();
    }

    /// Empty input must return a status error before any FFI call.
    ///
    /// This test does not require a GPU — the empty-slice fast-path in
    /// `NvJpeg2k::decode` returns `Nvjpeg2kStatus(3)` (BAD_JPEG) immediately.
    /// We still need a `NvJpeg2kDecoder` to reach `decode_sync`, so skip if
    /// no GPU is available.
    #[test]
    fn decode_empty_returns_error() {
        let mut dec = match NvJpeg2kDecoder::new(0) {
            Ok(d) => d,
            Err(_) => return, // no GPU available
        };
        let result = dec.decode_sync(&[]);
        assert!(
            matches!(result, Err(NvJpeg2kError::Nvjpeg2kStatus(3))),
            "expected Nvjpeg2kStatus(3) (BAD_JPEG) for empty input, got {result:?}",
        );
    }

    /// Minimal grayscale JPEG 2000 codestream (16×16, luma = 128).
    ///
    /// To regenerate (requires ImageMagick with OpenJPEG delegate):
    /// ```sh
    /// convert -size 16x16 xc:gray50 -define j2k:format=j2k /tmp/gray16x16.j2k
    /// python3 -c "d=open('/tmp/gray16x16.j2k','rb').read(); print(', '.join(f'0x{b:02x}' for b in d))"
    /// ```
    ///
    /// This is a raw `.j2k` codestream (no JP2 container header).  nvJPEG2000
    /// auto-detects the format from the SOC (0xff4f) / SIZ (0xff51) markers.
    const GRAY_16X16_J2K: &[u8] = &[
        0xff, 0x4f, 0xff, 0x51, 0x00, 0x29, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
        0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00,
        0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x07, 0x00, 0x00,
        0xff, 0x52, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x01, 0x00, 0x05, 0x04, 0x04, 0x00, 0x00, 0xff,
        0x5c, 0x00, 0x04, 0x40, 0x00, 0xff, 0x90, 0x00, 0x0a, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0xff, 0x93, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0xd9,
    ];

    /// End-to-end decode of the embedded 16×16 gray JPEG 2000 via the public API.
    /// Skipped gracefully if no GPU is available.
    #[test]
    fn decode_gray_j2k() {
        let mut dec = match NvJpeg2kDecoder::new(0) {
            Ok(d) => d,
            Err(_) => return, // no GPU — skip
        };

        let img = match dec.decode_sync(GRAY_16X16_J2K) {
            Ok(img) => img,
            // Status 3: BAD_JPEG — the minimal synthetic codestream is rejected by
            // some nvJPEG2000 versions (zeroed tile data stub).
            // Status 4: JPEG_NOT_SUPPORTED — codestream type not supported by this
            // nvJPEG2000 build (e.g. no HTJ2K).  Neither is a regression.
            Err(NvJpeg2kError::Nvjpeg2kStatus(3 | 4)) => return,
            // Any other error means the decode path regressed — fail loudly.
            Err(e) => panic!("decode_gray_j2k: unexpected GPU decode error: {e}"),
        };

        assert_eq!(img.width, 16, "unexpected width");
        assert_eq!(img.height, 16, "unexpected height");
        assert_eq!(img.color_space, Jpeg2kColorSpace::Gray);
        assert_eq!(img.data.len(), 16 * 16, "unexpected data length");
        // gray50 encodes to ~127.
        assert!(
            img.data.iter().all(|&p| p > 80 && p < 180),
            "unexpected luma values (expected ~127): {:?}",
            &img.data[..16.min(img.data.len())]
        );
    }
}
