//! Phase 9 task 4 — device-resident page buffer.
//!
//! `DevicePageBuffer` is the GPU-side composition target for one page.
//! When the renderer dispatches an image blit through the cache, the
//! kernel writes transformed pixels into this buffer (with `alpha=255`
//! for written pixels, `0` for untouched).  At end-of-page,
//! [`Self::download`] DMAs the buffer to host RGBA bytes; the renderer
//! alpha-composites those over its CPU-rasterised vector content for
//! the final page bitmap.
//!
//! # Layout
//!
//! Row-major RGBA8 (4 bytes/pixel).  Width and height match the page
//! resolution.  Zero-initialised on creation so any pixel the kernel
//! doesn't write reads back as `(0, 0, 0, 0)` — i.e. fully transparent
//! and contributes nothing to the source-over composite.

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, DriverError};

/// Bytes per pixel for the RGBA layout.  Public because the blit
/// kernel and the host-side composite both need to agree.
pub const RGBA_BPP: usize = 4;

/// Device-side composition target for one rendered page.
///
/// Drop releases the device memory; the buffer doesn't need to outlive
/// the page render.
pub struct DevicePageBuffer {
    /// Width × height × 4 bytes, row-major RGBA8.
    pub rgba: CudaSlice<u8>,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// CUDA stream the buffer's allocations and writes are bound to.
    /// Stored so [`Self::download`] uses the same stream and so a
    /// caller can synchronise it before relying on completed writes.
    stream: Arc<CudaStream>,
}

impl DevicePageBuffer {
    /// Allocate and zero a fresh page buffer of `width × height` pixels.
    ///
    /// Zero-initialisation matters: the blit kernel doesn't write every
    /// pixel — only the ones that fall inside an image's transformed
    /// extent — and the host-side composite reads alpha from un-written
    /// pixels to leave the CPU-rasterised content alone.
    ///
    /// # Errors
    /// Returns the underlying [`DriverError`] if device allocation fails
    /// (typically VRAM exhaustion).
    pub fn new(stream: Arc<CudaStream>, width: u32, height: u32) -> Result<Self, DriverError> {
        let len = (width as usize)
            .checked_mul(height as usize)
            .and_then(|n| n.checked_mul(RGBA_BPP))
            .ok_or(DriverError(
                cudarc::driver::sys::CUresult::CUDA_ERROR_INVALID_VALUE,
            ))?;
        let rgba = stream.alloc_zeros::<u8>(len)?;
        Ok(Self {
            rgba,
            width,
            height,
            stream,
        })
    }

    /// Width × height × 4 bytes — the size of the device allocation
    /// and the host buffer [`Self::download`] writes into.
    #[must_use]
    pub const fn byte_len(&self) -> usize {
        (self.width as usize) * (self.height as usize) * RGBA_BPP
    }

    /// The CUDA stream this buffer was bound to.  Exposed so kernels
    /// dispatched on a different stream can synchronise before reading
    /// the buffer.
    #[must_use]
    pub fn stream(&self) -> &CudaStream {
        &self.stream
    }

    /// Copy the buffer to a host `Vec<u8>`.  Synchronises the stream
    /// so all pending writes (typically a sequence of blit-kernel
    /// launches) complete before the bytes are observed on host.
    ///
    /// # Errors
    /// Returns the underlying [`DriverError`] if the D→H copy or the
    /// stream sync fails.
    pub fn download(&self) -> Result<Vec<u8>, DriverError> {
        let mut host = vec![0u8; self.byte_len()];
        self.stream.memcpy_dtoh(&self.rgba, &mut host)?;
        self.stream.synchronize()?;
        Ok(host)
    }
}

impl std::fmt::Debug for DevicePageBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DevicePageBuffer")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("byte_len", &self.byte_len())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rgba_bpp_is_four() {
        assert_eq!(RGBA_BPP, 4);
    }

    #[cfg(feature = "gpu-validation")]
    #[test]
    fn alloc_and_download_zero_buffer() {
        let ctx = cudarc::driver::CudaContext::new(0).expect("CUDA device 0");
        let stream = ctx.default_stream();
        let buf = DevicePageBuffer::new(stream, 32, 16).expect("alloc");
        assert_eq!(buf.width, 32);
        assert_eq!(buf.height, 16);
        assert_eq!(buf.byte_len(), 32 * 16 * 4);
        let host = buf.download().expect("download");
        assert_eq!(host.len(), 32 * 16 * 4);
        assert!(host.iter().all(|&b| b == 0), "alloc_zeros must zero-fill");
    }

    #[cfg(feature = "gpu-validation")]
    #[test]
    fn alloc_rejects_overflow_dimensions() {
        let ctx = cudarc::driver::CudaContext::new(0).expect("CUDA device 0");
        let stream = ctx.default_stream();
        // u32::MAX × u32::MAX × 4 overflows usize.
        let err = DevicePageBuffer::new(stream, u32::MAX, u32::MAX).unwrap_err();
        // Don't lock to a specific CUresult variant; just verify we
        // surfaced an error rather than panicking on overflow.
        let _ = format!("{err:?}");
    }
}
