//! CUDA host↔device transfer helpers.
//!
//! `upload_async` / `download_async` are implemented here so the trait
//! impl in `mod.rs` stays a thin dispatch table. Both ride the default
//! stream alongside compute work — CUDA's stream-ordered model means a
//! later `record_*` kernel launch automatically observes the transfer
//! once the stream catches up, with no explicit barrier.
//!
//! Drops down to cudarc's raw `result::memcpy_*_async` entry points
//! rather than the safe `CudaStream::memcpy_htod` wrapper because the
//! safe wrapper takes `&mut CudaSlice<u8>` (Rust borrow semantics) but
//! our trait API gives us `&Self::DeviceBuffer` (shared reference,
//! stream-ordered exclusivity). Matches the in-tree pattern of
//! `PageRecorder::record_zero_buffer`, which does the same drop-down
//! for `cuMemsetD8Async`.

use std::sync::Arc;

use cudarc::driver::{CudaSlice, DevicePtr};

use crate::GpuCtx;
use crate::backend::{BackendError, DownloadInner, Result};

use super::be;
use super::page_recorder::{PageFence, PageRecorder};

/// Verify `src_len ≤ dst_capacity` and emit a uniform error message
/// with the operation label for diagnostic clarity.
///
/// Returns the validated length as `usize` for downstream use. The
/// length-mismatch case is the most common cause of an unsigned-int
/// underflow in transfer code, so this stays a hard check rather than
/// a debug-assert.
fn check_transfer_len(src_len: usize, dst_capacity: usize, op_label: &'static str) -> Result<()> {
    if src_len > dst_capacity {
        return Err(BackendError::msg(format!(
            "{op_label}: source length ({src_len} bytes) exceeds destination \
             capacity ({dst_capacity} bytes)"
        )));
    }
    Ok(())
}

/// Initiate an asynchronous host→device upload on the backend's
/// shared stream and return a fence that signals when the copy
/// completes.
///
/// Empty `src` is a no-op: cudarc rejects zero-byte transfers and the
/// trait contract is satisfied by returning an immediately-signalled
/// fence (one recorded after no preceding work).
pub(super) fn upload_async(
    ctx: &Arc<GpuCtx>,
    dst: &CudaSlice<u8>,
    src: &[u8],
) -> Result<PageFence> {
    check_transfer_len(src.len(), dst.len(), "upload_async")?;
    if src.is_empty() {
        return PageRecorder::record_fence(ctx);
    }
    let stream = ctx.stream();
    let (dptr, _sync) = dst.device_ptr(stream);
    // Safety: `dptr` came from cudarc for `dst` on `stream`;
    // src.as_ptr() is borrowed for the duration of this call (cudarc
    // copies the bytes asynchronously but the caller-supplied src is
    // valid through `src`'s lifetime in this function — the trait
    // contract requires callers to keep `src` alive until the fence
    // signals, but cudarc actually performs the copy eagerly into
    // pinned-host-driver-mapped memory if `src` is unpinned, so the
    // safety property reduces to "src is valid right now", which the
    // borrow checker guarantees.
    unsafe {
        cudarc::driver::result::memcpy_htod_async(dptr, src, stream.cu_stream()).map_err(be)?;
    }
    PageRecorder::record_fence(ctx)
}

/// Initiate an asynchronous device→host download on the backend's
/// shared stream. Returns a download inner that holds the fence; the
/// caller wraps it in a `DownloadHandle` and either waits on it via
/// `wait_download` or lets it block-and-discard on Drop.
///
/// The dtoh copy writes directly into `dst` (no staging buffer); the
/// caller-provided `&mut [u8]` lifetime is held by the
/// `DownloadHandle` for the duration of the in-flight DMA.
pub(super) fn download_async(
    ctx: &Arc<GpuCtx>,
    src: &CudaSlice<u8>,
    dst: &mut [u8],
) -> Result<(CudaDownloadInner, PageFence)> {
    check_transfer_len(dst.len(), src.len(), "download_async")?;
    // Empty `dst` path: skip the cudarc call (which would reject zero
    // bytes), but still record a fence so the trait's "callable, then
    // wait_download or wait on fence()" contract holds uniformly.
    if !dst.is_empty() {
        let stream = ctx.stream();
        let (dptr, _sync) = src.device_ptr(stream);
        // Safety: `dptr` from cudarc for `src` on `stream`; `dst` is
        // borrowed by the returned `DownloadHandle` for the in-flight
        // DMA's lifetime via the trait's PhantomData; cudarc's async
        // dtoh takes the host pointer at call time and the driver
        // retains the address (not the slice metadata) until the
        // stream catches up.
        unsafe {
            cudarc::driver::result::memcpy_dtoh_async(dst, dptr, stream.cu_stream()).map_err(be)?;
        }
    }
    let fence = PageRecorder::record_fence(ctx)?;
    Ok((CudaDownloadInner::new(fence.clone()), fence))
}

/// Backend-internal download-completion state for CUDA.
///
/// Wraps the post-copy `PageFence`. `finish()` blocks on the fence;
/// after a successful wait the inner becomes a no-op so `Drop` (which
/// also calls `finish()`) doesn't re-block.
pub(super) struct CudaDownloadInner {
    /// `None` after `finish()` returns Ok — disarms the Drop path.
    fence: Option<PageFence>,
}

impl CudaDownloadInner {
    const fn new(fence: PageFence) -> Self {
        Self { fence: Some(fence) }
    }
}

impl DownloadInner for CudaDownloadInner {
    fn finish(&mut self) -> Result<()> {
        if let Some(fence) = self.fence.take() {
            fence.synchronize()?;
        }
        Ok(())
    }
}

#[cfg(all(test, feature = "gpu-validation"))]
mod tests {
    use crate::backend::cuda::CudaBackend;
    use crate::backend::{DownloadHandle, GpuBackend};

    fn try_backend() -> Option<CudaBackend> {
        CudaBackend::new().ok()
    }

    #[test]
    fn upload_then_download_round_trip() {
        let Some(b) = try_backend() else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        let payload: Vec<u8> = (0u8..=255).collect();
        let dst = b.alloc_device(payload.len()).expect("alloc");
        let upload_fence = b.upload_async(&dst, &payload).expect("upload");
        b.wait_transfer(upload_fence).expect("wait upload");

        let mut readback = vec![0u8; payload.len()];
        let handle = b.download_async(&dst, &mut readback).expect("download");
        b.wait_download(handle).expect("wait download");
        assert_eq!(readback, payload);
    }

    #[test]
    fn upload_rejects_oversize_src() {
        let Some(b) = try_backend() else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        let dst = b.alloc_device(4).expect("alloc");
        let err = b
            .upload_async(&dst, &[0u8; 8])
            .expect_err("oversize must fail");
        let msg = err.to_string();
        assert!(
            msg.contains("upload_async") && msg.contains("exceeds"),
            "unexpected error: {msg}"
        );
    }

    #[test]
    fn download_rejects_oversize_dst() {
        let Some(b) = try_backend() else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        let src = b.alloc_device(4).expect("alloc");
        let mut dst = vec![0u8; 8];
        // DownloadHandle is not Debug; expect_err would need it. Match
        // manually instead.
        match b.download_async(&src, &mut dst) {
            Err(e) => {
                let msg = e.to_string();
                assert!(
                    msg.contains("download_async") && msg.contains("exceeds"),
                    "unexpected error: {msg}"
                );
            }
            Ok(_) => panic!("oversize download must fail"),
        }
    }

    #[test]
    fn empty_upload_returns_immediate_fence() {
        let Some(b) = try_backend() else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        // Can't alloc a zero-byte device buffer — but we can prove the
        // empty-src path returns Ok against any non-zero dst.
        let dst = b.alloc_device(4).expect("alloc");
        let fence = b.upload_async(&dst, &[]).expect("empty upload ok");
        b.wait_transfer(fence).expect("wait empty fence");
    }

    #[test]
    fn drop_without_wait_does_not_panic() {
        let Some(b) = try_backend() else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        let dst = b.alloc_device(64).expect("alloc");
        b.upload_async(&dst, &[0xAB; 64])
            .and_then(|f| b.wait_transfer(f))
            .expect("setup");
        let mut readback = vec![0u8; 64];
        let handle: DownloadHandle<'_, CudaBackend> =
            b.download_async(&dst, &mut readback).expect("download");
        // Drop without calling wait_download — CudaDownloadInner::finish
        // is idempotent (fence.take() arms it for exactly one
        // synchronize()), so Drop's block-and-discard runs cleanly.
        drop(handle);
    }
}
