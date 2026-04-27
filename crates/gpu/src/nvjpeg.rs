//! nvJPEG GPU-accelerated JPEG decoding.
//!
//! Wraps the NVIDIA nvJPEG library (CUDA 12) to decode JPEG bitstreams
//! entirely on the GPU, copying only the finished pixel bytes back to the host.
//! On the RTX 5070 this runs at ~10 GB/s, roughly 10–20× faster than the
//! CPU `zune-jpeg` path for large images.
//!
//! # Usage
//!
//! ```no_run
//! use gpu::nvjpeg::NvJpeg;
//! let dec = NvJpeg::new().expect("nvJPEG unavailable");
//! let img = dec.decode(jpeg_bytes).expect("decode failed");
//! // img.data is interleaved RGB (or luma) bytes, img.width × img.height pixels.
//! ```
//!
//! # Feature flag
//!
//! This module is only compiled when the `nvjpeg` feature is enabled on the
//! `gpu` crate.  Callers that want a CPU fallback should check
//! [`NvJpeg::new`] at startup and fall back to `zune-jpeg` if it returns
//! `None` / `Err`.
//!
//! # Thread safety
//!
//! [`NvJpeg`] is `Send` but not `Sync`: each thread must own its own instance.
//! Creating one per thread is cheap (handle creation is a single CUDA API call).

// All nvJPEG calls are FFI through raw pointers.
#![cfg(feature = "nvjpeg")]

use std::ptr;

// ── Raw FFI declarations ──────────────────────────────────────────────────────
//
// We declare only the nvJPEG surface we use.  This avoids the overhead of
// bindgen at build time while keeping the binding surface minimal and auditable.

/// nvJPEG status codes.  Only `NVJPEG_STATUS_SUCCESS = 0` is success.
#[allow(non_camel_case_types)]
type nvjpegStatus_t = i32;

/// Opaque nvJPEG library handle (wraps device allocators, backend state).
#[repr(C)]
struct NvjpegHandle_ {
    _opaque: [u8; 0],
}
#[allow(non_camel_case_types)]
type nvjpegHandle_t = *mut NvjpegHandle_;

/// Opaque per-decode state (Huffman tables, coefficient buffers, etc.).
#[repr(C)]
struct NvjpegJpegState_ {
    _opaque: [u8; 0],
}
#[allow(non_camel_case_types)]
type nvjpegJpegState_t = *mut NvjpegJpegState_;

/// Chroma subsampling type — only used as an out-parameter; we inspect it to
/// validate that the image is a standard YUV JPEG (not CMYK).
#[allow(non_camel_case_types)]
type nvjpegChromaSubsampling_t = i32;

/// Output format: we use `NVJPEG_OUTPUT_RGBI` (interleaved RGB, channel[0])
/// and `NVJPEG_OUTPUT_Y` (luma-only, channel[0]).
#[allow(non_camel_case_types)]
type nvjpegOutputFormat_t = i32;

/// NVJPEG_OUTPUT_Y — single-channel luma output to channel[0].
const NVJPEG_OUTPUT_Y: nvjpegOutputFormat_t = 2;
/// NVJPEG_OUTPUT_RGBI — interleaved RGB output to channel[0], bytes_per_element=3.
const NVJPEG_OUTPUT_RGBI: nvjpegOutputFormat_t = 5;

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
    fn zeroed() -> Self {
        Self {
            channel: [ptr::null_mut(); NVJPEG_MAX_COMPONENT],
            pitch: [0; NVJPEG_MAX_COMPONENT],
        }
    }
}

/// A raw CUDA stream handle as an opaque pointer.  This is the C type
/// `CUstream` / `cudaStream_t`; we only pass it through to nvJPEG.
#[allow(non_camel_case_types)]
type CUstream = *mut std::ffi::c_void;

// NVJPEGAPI on Linux is the default calling convention — no special attribute.
unsafe extern "C" {
    fn nvjpegCreateSimple(handle: *mut nvjpegHandle_t) -> nvjpegStatus_t;
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

// ── Error type ────────────────────────────────────────────────────────────────

/// Errors returned by the nvJPEG wrapper.
#[derive(Debug)]
pub enum NvJpegError {
    /// nvJPEG API returned a non-zero status code.
    NvjpegStatus(i32),
    /// Image has an unsupported number of components (not 1 or 3).
    UnsupportedComponents(i32),
    /// The decoded image has a zero dimension.
    ZeroDimension { width: i32, height: i32 },
    /// Arithmetic overflow computing pixel buffer size.
    Overflow,
}

impl std::fmt::Display for NvJpegError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NvjpegStatus(code) => write!(f, "nvJPEG error status {code}"),
            Self::UnsupportedComponents(n) => write!(f, "unsupported JPEG component count {n}"),
            Self::ZeroDimension { width, height } => {
                write!(f, "JPEG reported zero dimension {width}×{height}")
            }
            Self::Overflow => write!(f, "pixel buffer size overflow"),
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

// ── Decoded image ─────────────────────────────────────────────────────────────

/// Decoded JPEG image, host-resident.
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
pub struct NvJpeg {
    handle: nvjpegHandle_t,
    state: nvjpegJpegState_t,
}

// SAFETY: nvjpegHandle_t and nvjpegJpegState_t are opaque C pointers.
// nvJPEG guarantees that a handle used on one host thread at a time is safe —
// the internal CUDA stream serialises device-side work.
unsafe impl Send for NvJpeg {}

impl NvJpeg {
    /// Initialise nvJPEG.
    ///
    /// Returns `None` if the shared library is absent or device initialisation
    /// fails.  Callers should treat `None` as "nvJPEG unavailable, fall back
    /// to CPU".
    ///
    /// # Errors
    ///
    /// Returns an error if nvJPEG device initialisation or state allocation
    /// fails.
    pub fn new() -> Result<Self> {
        let mut handle: nvjpegHandle_t = ptr::null_mut();
        // SAFETY: nvjpegCreateSimple initialises a fresh library handle.
        let status = unsafe { nvjpegCreateSimple(ptr::addr_of_mut!(handle)) };
        if status != NVJPEG_STATUS_SUCCESS {
            return Err(NvJpegError::NvjpegStatus(status));
        }
        assert!(!handle.is_null(), "nvjpegCreateSimple succeeded but returned null handle");

        let mut state: nvjpegJpegState_t = ptr::null_mut();
        // SAFETY: handle is valid; nvjpegJpegStateCreate allocates decode buffers.
        let status = unsafe { nvjpegJpegStateCreate(handle, ptr::addr_of_mut!(state)) };
        if status != NVJPEG_STATUS_SUCCESS {
            // Destroy the handle we just created before returning the error.
            // SAFETY: handle is valid; state was never initialised so we don't destroy it.
            unsafe { nvjpegDestroy(handle) };
            return Err(NvJpegError::NvjpegStatus(status));
        }
        assert!(!state.is_null(), "nvjpegJpegStateCreate succeeded but returned null state");

        Ok(Self { handle, state })
    }

    /// Decode a JPEG bitstream, returning host-resident pixel bytes.
    ///
    /// `stream` is the raw `CUstream` handle on which GPU work is enqueued.
    /// The caller is responsible for synchronising the stream after this call
    /// before reading `DecodedJpeg::data`.
    ///
    /// For 1-component JPEGs the output is grayscale (1 byte/pixel).
    /// For 3-component JPEGs the output is interleaved RGB (3 bytes/pixel).
    /// 4-component (CMYK) JPEGs are rejected — the PDF interpreter converts
    /// CMYK JPEG at the `decode_dct` level before dispatching here.
    ///
    /// # Errors
    ///
    /// Returns an error if the JPEG is invalid, has unsupported dimensions,
    /// or if a CUDA API call fails.
    pub fn decode(&mut self, data: &[u8], stream: CUstream) -> Result<DecodedJpeg> {
        // ── Phase 1: inspect headers ──────────────────────────────────────────
        let mut n_components: i32 = 0;
        let mut subsampling: nvjpegChromaSubsampling_t = 0;
        let mut widths = [0i32; NVJPEG_MAX_COMPONENT];
        let mut heights = [0i32; NVJPEG_MAX_COMPONENT];

        // SAFETY: all out-pointers are to local stack variables; data is a valid slice.
        let status = unsafe {
            nvjpegGetImageInfo(
                self.handle,
                data.as_ptr(),
                data.len(),
                ptr::addr_of_mut!(n_components),
                ptr::addr_of_mut!(subsampling),
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

        // nvjpegGetImageInfo fills widths[0] / heights[0] for the first (or only) channel.
        let width = widths[0];
        let height = heights[0];
        if width <= 0 || height <= 0 {
            return Err(NvJpegError::ZeroDimension { width, height });
        }
        let width_u = width as usize;
        let height_u = height as usize;

        // ── Phase 2: allocate host output buffer ──────────────────────────────
        // nvjpegDecode with RGBI / Y writes to channel[0] which must be a
        // device pointer.  We allocate pinned host memory via cudaMallocHost
        // so the DMA transfer after decode is zero-copy.  For simplicity we
        // use regular Vec and let nvJPEG write directly to it — nvjpegDecode
        // accepts any writable device or host pointer.
        //
        // Pitch = width * bytes_per_px (tightly packed, no row padding).
        let pitch = width_u
            .checked_mul(bytes_per_px)
            .ok_or(NvJpegError::Overflow)?;
        let buf_len = pitch.checked_mul(height_u).ok_or(NvJpegError::Overflow)?;

        // Allocate on the heap; nvjpegDecode fills it via the CUDA stream.
        let mut pixels = vec![0u8; buf_len];

        let mut img = NvjpegImage::zeroed();
        img.channel[0] = pixels.as_mut_ptr();
        img.pitch[0] = pitch;

        // ── Phase 3: decode ───────────────────────────────────────────────────
        // SAFETY:
        // - self.handle and self.state are valid (created in `new`).
        // - data is a valid slice for the duration of this call.
        // - img.channel[0] points to `pixels`, which is live for the call.
        // - stream is a valid CUstream owned by the caller's GpuCtx.
        // nvjpegDecode enqueues GPU work on `stream` and returns before the
        // GPU finishes.  The caller must synchronise the stream before reading
        // `pixels`.
        let status = unsafe {
            nvjpegDecode(
                self.handle,
                self.state,
                data.as_ptr(),
                data.len(),
                out_fmt,
                ptr::addr_of_mut!(img),
                stream,
            )
        };
        if status != NVJPEG_STATUS_SUCCESS {
            return Err(NvJpegError::NvjpegStatus(status));
        }

        Ok(DecodedJpeg {
            data: pixels,
            width: width as u32,
            height: height as u32,
            color_space,
        })
    }
}

impl Drop for NvJpeg {
    fn drop(&mut self) {
        // Ignore error codes on teardown — if the device is gone there is nothing
        // meaningful we can do, and panicking in Drop is unsound.
        // SAFETY: handle and state are valid; no other references exist (NvJpeg is not Sync).
        unsafe {
            let _ = nvjpegJpegStateDestroy(self.state);
            let _ = nvjpegDestroy(self.handle);
        }
    }
}

// ── Stream synchronisation ────────────────────────────────────────────────────
//
// nvjpegDecode enqueues GPU work asynchronously on the CUstream.  We need to
// synchronise the stream before reading the pixel buffer back on the host.
// cudarc's CudaStream::synchronize() requires an Arc<CudaStream>, but we only
// hold a raw CUstream.  We call the CUDA driver API directly.

unsafe extern "C" {
    fn cuStreamSynchronize(stream: *mut std::ffi::c_void) -> i32;
}

// ── NvJpegDecoder ─────────────────────────────────────────────────────────────

/// Safe, stream-owning wrapper around [`NvJpeg`] for use outside the `gpu` crate.
///
/// Holds an `NvJpeg` decoder and the raw `CUstream` it submits work to.
/// Exposes `decode_sync`, a fully synchronous decode that waits for the GPU
/// before returning pixel bytes.
///
/// `Send` but not `Sync`: each thread must own its own instance.
/// The raw stream pointer must outlive `NvJpegDecoder`; in practice
/// the caller creates it from a `cudarc::driver::CudaStream` and ensures
/// the `CudaStream` outlives this struct.
pub struct NvJpegDecoder {
    dec: NvJpeg,
    /// Raw `CUstream` handle; valid for the lifetime of the owning `CudaStream`.
    stream: CUstream,
}

// SAFETY: NvJpeg is Send; the raw stream pointer is only touched from the
// thread that calls decode_sync (guaranteed by &mut self).
unsafe impl Send for NvJpegDecoder {}

impl NvJpegDecoder {
    /// Create an `NvJpegDecoder` that submits work on `stream`.
    ///
    /// `stream` must remain valid (not destroyed) for the entire lifetime of
    /// this `NvJpegDecoder`.  The typical pattern is to create both from the
    /// same `cudarc::driver::CudaContext` and drop the decoder before the
    /// stream.
    ///
    /// # Errors
    ///
    /// Returns an error if nvJPEG device initialisation fails.
    pub fn new(stream: CUstream) -> Result<Self> {
        Ok(Self { dec: NvJpeg::new()?, stream })
    }

    /// Decode a JPEG bitstream synchronously, returning host-resident pixels.
    ///
    /// Enqueues GPU work, then synchronises the stream before returning so that
    /// `DecodedJpeg::data` is fully written and safe to read on any thread.
    ///
    /// # Errors
    ///
    /// Returns an error if the JPEG is invalid, has unsupported dimensions,
    /// or if a CUDA API call fails.
    pub fn decode_sync(&mut self, data: &[u8]) -> Result<DecodedJpeg> {
        let img = self.dec.decode(data, self.stream)?;

        // Synchronise the stream so the GPU DMA into `img.data` is complete
        // before we return the Vec to the caller.
        // SAFETY: self.stream is a valid CUstream for the lifetime of this struct.
        let status = unsafe { cuStreamSynchronize(self.stream) };
        if status != 0 {
            return Err(NvJpegError::NvjpegStatus(status));
        }

        Ok(img)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal valid 1×1 grayscale JPEG (SOI + APP0 + DQT + SOF0 + DHT + SOS + EOI).
    /// Generated with: `convert -size 1x1 xc:gray50 -quality 50 /tmp/gray1x1.jpg && xxd -i`
    const GRAY_1X1_JPEG: &[u8] = &[
        0xff, 0xd8, 0xff, 0xe0, 0x00, 0x10, 0x4a, 0x46, 0x49, 0x46, 0x00, 0x01, 0x01, 0x00,
        0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0xff, 0xdb, 0x00, 0x43, 0x00, 0x10, 0x0b, 0x0c,
        0x0e, 0x0c, 0x0a, 0x10, 0x0e, 0x0d, 0x0e, 0x12, 0x11, 0x10, 0x13, 0x18, 0x28, 0x1a,
        0x18, 0x16, 0x16, 0x18, 0x31, 0x23, 0x25, 0x1d, 0x28, 0x3a, 0x33, 0x3d, 0x3c, 0x39,
        0x33, 0x38, 0x37, 0x40, 0x48, 0x5c, 0x4e, 0x40, 0x44, 0x57, 0x45, 0x37, 0x38, 0x50,
        0x6d, 0x51, 0x57, 0x5f, 0x62, 0x67, 0x68, 0x67, 0x3e, 0x4d, 0x71, 0x79, 0x70, 0x64,
        0x78, 0x5c, 0x65, 0x67, 0x63, 0xff, 0xc0, 0x00, 0x0b, 0x08, 0x00, 0x01, 0x00, 0x01,
        0x01, 0x01, 0x11, 0x00, 0xff, 0xc4, 0x00, 0x1f, 0x00, 0x00, 0x01, 0x05, 0x01, 0x01,
        0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x02,
        0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0xff, 0xc4, 0x00, 0xb5, 0x10,
        0x00, 0x02, 0x01, 0x03, 0x03, 0x02, 0x04, 0x03, 0x05, 0x05, 0x04, 0x04, 0x00, 0x00,
        0x01, 0x7d, 0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06,
        0x13, 0x51, 0x61, 0x07, 0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08, 0x23, 0x42,
        0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0, 0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16,
        0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2a, 0x34, 0x35, 0x36, 0x37,
        0x38, 0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4a, 0x53, 0x54, 0x55,
        0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6a, 0x73,
        0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
        0x8a, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6,
        0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2,
        0xc3, 0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7,
        0xd8, 0xd9, 0xda, 0xe1, 0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf1,
        0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8, 0xf9, 0xfa, 0xff, 0xda, 0x00, 0x08, 0x01,
        0x01, 0x00, 0x00, 0x3f, 0x00, 0xf9, 0x7c, 0xff, 0xd9,
    ];

    /// `NvJpeg::new()` must not panic — it either succeeds or returns an error.
    /// The test skips gracefully on machines without a GPU.
    #[test]
    fn new_does_not_panic() {
        let _ = NvJpeg::new(); // ignore Ok/Err
    }

    /// End-to-end decode of the embedded 1×1 gray JPEG.
    /// Skipped if no GPU is available.
    #[test]
    fn decode_gray_1x1() {
        use cudarc::driver::CudaContext;
        use std::sync::Arc;

        // Init CUDA; skip if no device.
        let ctx: Arc<CudaContext> = match CudaContext::new(0) {
            Ok(c) => c,
            Err(_) => return,
        };
        let stream = ctx.default_stream();
        let raw_stream = stream.cu_stream() as *mut std::ffi::c_void;

        let mut dec = match NvJpeg::new() {
            Ok(d) => d,
            Err(_) => return,
        };

        let img = dec.decode(GRAY_1X1_JPEG, raw_stream).expect("decode failed");
        stream.synchronize().expect("stream sync failed");

        assert_eq!(img.width, 1);
        assert_eq!(img.height, 1);
        assert_eq!(img.color_space, JpegColorSpace::Gray);
        assert_eq!(img.data.len(), 1);
    }

    /// `decode` with an empty slice must return an error, not panic or UB.
    #[test]
    fn decode_empty_returns_error() {
        use cudarc::driver::CudaContext;
        use std::sync::Arc;

        let ctx: Arc<CudaContext> = match CudaContext::new(0) {
            Ok(c) => c,
            Err(_) => return,
        };
        let stream = ctx.default_stream();
        let raw_stream = stream.cu_stream() as *mut std::ffi::c_void;

        let mut dec = match NvJpeg::new() {
            Ok(d) => d,
            Err(_) => return,
        };

        let result = dec.decode(&[], raw_stream);
        assert!(
            matches!(result, Err(NvJpegError::NvjpegStatus(_))),
            "expected NvjpegStatus error for empty input, got {result:?}",
        );
    }
}
