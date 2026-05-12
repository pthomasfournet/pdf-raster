//! Shared helpers for one-shot test/oracle dispatchers in this
//! module.
//!
//! The dispatchers (`scan::dispatch_blelloch_scan`,
//! `huffman::dispatch_phase1_intra_sync`, etc.) follow the same
//! shape: alloc several device buffers → upload → record → submit
//! → wait → download → free. The `?` operator on any intermediate
//! step skips the trailing `free_device` calls, leaking every
//! buffer allocated before the failing line.
//!
//! `DeviceBufferGuard` is a thin RAII wrapper that owns a
//! `B::DeviceBuffer` and frees it via the backend's `free_device`
//! on Drop. The happy path calls `.take()` before the explicit
//! free; the error path lets `Drop` clean up.

use crate::backend::{GpuBackend, Result};

/// RAII guard around an allocated `B::DeviceBuffer`. On Drop the
/// buffer is returned to the backend via `free_device` — unless
/// `take()` has already been called, in which case Drop is a
/// no-op and the caller owns the raw buffer.
///
/// The guard borrows the backend immutably so multiple guards can
/// coexist (each dispatcher allocates several buffers concurrently).
pub(super) struct DeviceBufferGuard<'a, B: GpuBackend + ?Sized> {
    backend: &'a B,
    buf: Option<B::DeviceBuffer>,
}

impl<'a, B: GpuBackend + ?Sized> DeviceBufferGuard<'a, B> {
    /// Allocate `size` bytes of (uninitialised) device memory and
    /// wrap the result in a Drop-frees guard.
    ///
    /// # Errors
    /// Propagates whatever `backend.alloc_device` returns.
    pub(super) fn alloc(backend: &'a B, size: usize) -> Result<Self> {
        Ok(Self {
            backend,
            buf: Some(backend.alloc_device(size)?),
        })
    }

    /// Allocate `size` bytes of zeroed device memory and wrap the
    /// result in a Drop-frees guard.
    ///
    /// # Errors
    /// Propagates whatever `backend.alloc_device_zeroed` returns.
    //
    // Only the test/oracle dispatchers under `huffman` need zeroed
    // memory (the bitstream's trailing peek16-headroom word + the
    // symbols_out buffer must start at zero). Gate matches `huffman`'s
    // own gating so this isn't dead-code-warned in non-test builds.
    #[cfg(all(test, feature = "gpu-validation"))]
    pub(super) fn alloc_zeroed(backend: &'a B, size: usize) -> Result<Self> {
        Ok(Self {
            backend,
            buf: Some(backend.alloc_device_zeroed(size)?),
        })
    }

    /// Borrow the underlying buffer for recording / uploads. Does
    /// not transfer ownership; the guard still frees on Drop.
    pub(super) const fn as_ref(&self) -> &B::DeviceBuffer {
        match &self.buf {
            Some(b) => b,
            None => panic!("DeviceBufferGuard::as_ref after take"),
        }
    }

    /// Transfer the buffer to the caller. Drop becomes a no-op;
    /// the caller is now responsible for freeing the buffer (e.g.,
    /// via `backend.free_device`).
    ///
    /// Used at the end of a dispatcher's happy path so the explicit
    /// `free_device` calls keep their existing single-source-of-
    /// ownership shape (rather than relying on Drop ordering).
    pub(super) fn take(mut self) -> B::DeviceBuffer {
        self.buf
            .take()
            .expect("DeviceBufferGuard::take called twice")
    }
}

impl<B: GpuBackend + ?Sized> Drop for DeviceBufferGuard<'_, B> {
    fn drop(&mut self) {
        if let Some(buf) = self.buf.take() {
            self.backend.free_device(buf);
        }
    }
}
