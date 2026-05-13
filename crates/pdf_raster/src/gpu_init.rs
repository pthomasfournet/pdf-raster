//! Per-thread GPU decoder lifecycle: init, lend-to-renderer, reclaim, release.
//!
//! nvJPEG and nvJPEG2000 decoders are held in per-thread `thread_local!` slots
//! so each Rayon worker gets its own instance without locking on the hot path.
//!
//! VA-API JPEG decoding uses a different model: one `DecodeQueue<VapiJpegDecoder>`
//! per session (constructed in `open_session`, owned by `RasterSession`), which
//! routes all submissions through a single OS thread.  There is therefore no TLS
//! slot for VA-API — see `crate::decode_queue` instead.
//!
//! `DECODER_INIT_LOCK` serialises construction calls (`nvjpegCreateEx`,
//! `vaInitialize`) which are not safe to call concurrently.  It is also used by
//! `crate::decode_queue::build_vaapi_queue`.

#[cfg(any(feature = "nvjpeg", feature = "nvjpeg2k", feature = "gpu-jpeg-huffman"))]
use crate::BackendPolicy;
#[cfg(any(feature = "nvjpeg", feature = "nvjpeg2k", feature = "gpu-jpeg-huffman"))]
use std::cell::RefCell;

// ── Three-state decoder slot ──────────────────────────────────────────────────

#[cfg(any(feature = "nvjpeg", feature = "nvjpeg2k"))]
#[derive(Default)]
pub enum DecoderInit<T> {
    #[default]
    Uninitialised,
    Ready(Option<T>),
    Failed,
}

// nvjpegCreateEx races if called concurrently; vaInitialize is serialised through
// this lock too (used by crate::decode_queue::build_vaapi_queue).
// Only held during construction, never during render.
#[cfg(any(
    feature = "nvjpeg",
    feature = "nvjpeg2k",
    feature = "vaapi",
    feature = "gpu-jpeg-huffman",
))]
pub static DECODER_INIT_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

// ── Thread-local decoder slots ────────────────────────────────────────────────

#[cfg(feature = "nvjpeg")]
thread_local! {
    pub static NVJPEG_DEC: RefCell<DecoderInit<gpu::nvjpeg::NvJpegDecoder>> =
        const { RefCell::new(DecoderInit::Uninitialised) };
}

#[cfg(feature = "nvjpeg2k")]
thread_local! {
    pub static NVJPEG2K_DEC: RefCell<DecoderInit<gpu::nvjpeg2k::NvJpeg2kDecoder>> =
        const { RefCell::new(DecoderInit::Uninitialised) };
}

// ── Init helpers ──────────────────────────────────────────────────────────────

/// Try to initialise this thread's nvJPEG decoder.
///
/// Returns `Ok(())` if the decoder is (or becomes) ready, or `Err` with a
/// human-readable message if it fails and `policy` is `ForceCuda`.  On `Auto`
/// a failure is logged and silently skipped.
#[cfg(feature = "nvjpeg")]
pub fn ensure_nvjpeg(policy: BackendPolicy) -> Result<(), String> {
    NVJPEG_DEC.with(|cell| {
        // TLS is per-thread — only this thread writes this slot, so a single
        // check is sufficient before acquiring the construction lock.
        match *cell.borrow() {
            DecoderInit::Uninitialised => {}
            DecoderInit::Failed if matches!(policy, BackendPolicy::ForceCuda) => {
                return Err("nvJPEG decoder failed to initialise on a previous attempt".to_owned());
            }
            _ => return Ok(()),
        }

        // Serialise construction across threads — nvjpegCreateEx is not thread-safe.
        let guard = DECODER_INIT_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let result = gpu::nvjpeg::NvJpegDecoder::new(0);
        drop(guard);

        match result {
            Ok(dec) => {
                *cell.borrow_mut() = DecoderInit::Ready(Some(dec));
                Ok(())
            }
            Err(e) => {
                *cell.borrow_mut() = DecoderInit::Failed;
                if matches!(policy, BackendPolicy::ForceCuda) {
                    Err(format!("nvJPEG unavailable: {e}"))
                } else {
                    log::warn!(
                        "rasterrocket: nvJPEG unavailable ({e}); \
                         JPEG images will be decoded on CPU for this thread"
                    );
                    Ok(())
                }
            }
        }
    })
}

/// Try to initialise this thread's nvJPEG2000 decoder.
#[cfg(feature = "nvjpeg2k")]
pub fn ensure_nvjpeg2k(policy: BackendPolicy) -> Result<(), String> {
    NVJPEG2K_DEC.with(|cell| {
        match *cell.borrow() {
            DecoderInit::Uninitialised => {}
            DecoderInit::Failed if matches!(policy, BackendPolicy::ForceCuda) => {
                return Err(
                    "nvJPEG2000 decoder failed to initialise on a previous attempt".to_owned(),
                );
            }
            _ => return Ok(()),
        }

        let guard = DECODER_INIT_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let result = gpu::nvjpeg2k::NvJpeg2kDecoder::new(0);
        drop(guard);

        match result {
            Ok(dec) => {
                *cell.borrow_mut() = DecoderInit::Ready(Some(dec));
                Ok(())
            }
            Err(e) => {
                *cell.borrow_mut() = DecoderInit::Failed;
                if matches!(policy, BackendPolicy::ForceCuda) {
                    Err(format!("nvJPEG2000 unavailable: {e}"))
                } else {
                    log::warn!(
                        "rasterrocket: nvJPEG2000 unavailable ({e}); \
                         JPEG 2000 images will be decoded on CPU for this thread"
                    );
                    Ok(())
                }
            }
        }
    })
}

// ── Release helpers (called via rayon::broadcast before pool drop) ────────────

/// Drop this thread's nvJPEG decoder immediately so the TLS destructor at
/// process exit is a no-op — avoids the CUDA driver teardown race.
#[cfg(feature = "nvjpeg")]
pub fn release_nvjpeg_this_thread() {
    NVJPEG_DEC.with(|cell| {
        *cell.borrow_mut() = DecoderInit::Uninitialised;
    });
}

#[cfg(feature = "nvjpeg2k")]
pub fn release_nvjpeg2k_this_thread() {
    NVJPEG2K_DEC.with(|cell| {
        *cell.borrow_mut() = DecoderInit::Uninitialised;
    });
}

// ── GPU parallel-Huffman JPEG decoder ─────────────────────────────────────────

/// Three-state slot for a GPU parallel-Huffman JPEG decoder.
///
/// Mirrors [`DecoderInit<T>`] so a failed construction is remembered and not
/// retried on every subsequent page.
#[cfg(feature = "gpu-jpeg-huffman")]
pub enum JpegGpuInit<T> {
    Uninitialised,
    Ready(Option<T>),
    Failed,
}

#[cfg(feature = "gpu-jpeg-huffman")]
thread_local! {
    pub static JPEG_CUDA_DEC: RefCell<
        JpegGpuInit<gpu::jpeg_decoder::JpegGpuDecoder<gpu::backend::cuda::CudaBackend>>,
    > = const { RefCell::new(JpegGpuInit::Uninitialised) };
}

#[cfg(all(feature = "gpu-jpeg-huffman", feature = "vulkan"))]
thread_local! {
    pub static JPEG_VK_DEC: RefCell<
        JpegGpuInit<gpu::jpeg_decoder::JpegGpuDecoder<gpu::backend::vulkan::VulkanBackend>>,
    > = const { RefCell::new(JpegGpuInit::Uninitialised) };
}

/// Initialise this thread's CUDA parallel-Huffman JPEG decoder if not already done.
///
/// Errors are non-fatal when `policy` is `Auto`; the decoder slot moves to
/// `Failed` and the CPU path is used instead. A previous failure is remembered
/// and not retried on subsequent pages.
#[cfg(feature = "gpu-jpeg-huffman")]
pub fn ensure_jpeg_gpu_huffman(policy: BackendPolicy) -> Result<(), String> {
    JPEG_CUDA_DEC.with(|cell| {
        match *cell.borrow() {
            JpegGpuInit::Uninitialised => {}
            JpegGpuInit::Failed if matches!(policy, BackendPolicy::ForceCuda) => {
                return Err(
                    "GPU JPEG decoder failed to initialise on a previous attempt".to_owned(),
                );
            }
            _ => return Ok(()),
        }
        let guard = DECODER_INIT_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let result = gpu::backend::cuda::CudaBackend::new();
        drop(guard);
        match result {
            Ok(backend) => {
                *cell.borrow_mut() =
                    JpegGpuInit::Ready(Some(gpu::jpeg_decoder::JpegGpuDecoder::new(backend)));
                Ok(())
            }
            Err(e) => {
                *cell.borrow_mut() = JpegGpuInit::Failed;
                let msg = format!("rasterrocket: GPU JPEG decoder unavailable ({e})");
                if matches!(policy, BackendPolicy::ForceCuda) {
                    Err(msg)
                } else {
                    log::info!("{msg}; JPEG images will be decoded on CPU for this thread");
                    Ok(())
                }
            }
        }
    })
}

/// Initialise this thread's Vulkan parallel-Huffman JPEG decoder if not already done.
///
/// Each Rayon worker constructs its own `VulkanBackend` (independent Vulkan
/// logical device on the same physical device) so per-thread command-buffer
/// recording has no cross-thread contention. Construction is serialised via
/// `DECODER_INIT_LOCK` to avoid concurrent Vulkan driver init races.
#[cfg(all(feature = "gpu-jpeg-huffman", feature = "vulkan"))]
pub fn ensure_jpeg_vk_huffman(policy: BackendPolicy) -> Result<(), String> {
    JPEG_VK_DEC.with(|cell| {
        match *cell.borrow() {
            JpegGpuInit::Uninitialised => {}
            JpegGpuInit::Failed if matches!(policy, BackendPolicy::ForceVulkan) => {
                return Err(
                    "Vulkan JPEG decoder failed to initialise on a previous attempt".to_owned(),
                );
            }
            _ => return Ok(()),
        }
        let guard = DECODER_INIT_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let result = gpu::backend::vulkan::VulkanBackend::new();
        drop(guard);
        match result {
            Ok(backend) => {
                *cell.borrow_mut() =
                    JpegGpuInit::Ready(Some(gpu::jpeg_decoder::JpegGpuDecoder::new(backend)));
                Ok(())
            }
            Err(e) => {
                *cell.borrow_mut() = JpegGpuInit::Failed;
                let msg = format!("rasterrocket: Vulkan JPEG decoder unavailable ({e})");
                if matches!(policy, BackendPolicy::ForceVulkan) {
                    Err(msg)
                } else {
                    log::info!("{msg}; JPEG images will be decoded on CPU for this thread");
                    Ok(())
                }
            }
        }
    })
}

/// Drop this thread's Vulkan JPEG decoder immediately so the TLS destructor at
/// process exit is a no-op — avoids Vulkan device teardown races.
#[cfg(all(feature = "gpu-jpeg-huffman", feature = "vulkan"))]
pub fn release_jpeg_vk_this_thread() {
    JPEG_VK_DEC.with(|cell| {
        *cell.borrow_mut() = JpegGpuInit::Uninitialised;
    });
}
