//! Per-thread GPU decoder lifecycle: init, lend-to-renderer, reclaim, release.
//!
//! Each rayon worker thread owns one instance of each decoder type via
//! `thread_local!`.  `DecoderInit<T>` is a three-state machine that prevents
//! retry-spam after a one-time init failure without holding a lock on the hot
//! path.  The `DECODER_INIT_LOCK` mutex serialises the *construction* calls
//! (`nvjpegCreateEx`, `vaInitialize`) which are not safe to call concurrently.

#[cfg(any(feature = "nvjpeg", feature = "nvjpeg2k", feature = "vaapi"))]
use crate::BackendPolicy;
#[cfg(any(feature = "nvjpeg", feature = "nvjpeg2k", feature = "vaapi"))]
use std::cell::RefCell;

// ── Three-state decoder slot ──────────────────────────────────────────────────

#[cfg(any(feature = "nvjpeg", feature = "nvjpeg2k", feature = "vaapi"))]
#[derive(Default)]
pub(crate) enum DecoderInit<T> {
    #[default]
    Uninitialised,
    Ready(Option<T>),
    Failed,
}

// nvjpegCreateEx races if called concurrently; vaInitialize is serialised here
// out of caution too.  Only held during construction, never during render.
#[cfg(any(feature = "nvjpeg", feature = "nvjpeg2k", feature = "vaapi"))]
pub(crate) static DECODER_INIT_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

// ── Thread-local decoder slots ────────────────────────────────────────────────

#[cfg(feature = "nvjpeg")]
thread_local! {
    pub(crate) static NVJPEG_DEC: RefCell<DecoderInit<gpu::nvjpeg::NvJpegDecoder>> =
        const { RefCell::new(DecoderInit::Uninitialised) };
}

#[cfg(feature = "nvjpeg2k")]
thread_local! {
    pub(crate) static NVJPEG2K_DEC: RefCell<DecoderInit<gpu::nvjpeg2k::NvJpeg2kDecoder>> =
        const { RefCell::new(DecoderInit::Uninitialised) };
}

#[cfg(feature = "vaapi")]
thread_local! {
    pub(crate) static VAAPI_JPEG_DEC: RefCell<DecoderInit<gpu::vaapi::VapiJpegDecoder>> =
        const { RefCell::new(DecoderInit::Uninitialised) };
}

// ── Init helpers ──────────────────────────────────────────────────────────────

/// Try to initialise this thread's nvJPEG decoder.
///
/// Returns `Ok(())` if the decoder is (or becomes) ready, or `Err` with a
/// human-readable message if it fails and `policy` is `ForceCuda`.  On `Auto`
/// a failure is logged and silently skipped.
#[cfg(feature = "nvjpeg")]
pub(crate) fn ensure_nvjpeg(policy: BackendPolicy) -> Result<(), String> {
    NVJPEG_DEC.with(|cell| {
        if !matches!(*cell.borrow(), DecoderInit::Uninitialised) {
            return match &*cell.borrow() {
                DecoderInit::Failed if matches!(policy, BackendPolicy::ForceCuda) => {
                    Err("nvJPEG decoder failed to initialise on a previous attempt".to_owned())
                }
                _ => Ok(()),
            };
        }

        let _guard = DECODER_INIT_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // Re-check after acquiring the lock (another thread may have raced).
        if !matches!(*cell.borrow(), DecoderInit::Uninitialised) {
            drop(_guard);
            return match &*cell.borrow() {
                DecoderInit::Failed if matches!(policy, BackendPolicy::ForceCuda) => {
                    Err("nvJPEG decoder failed to initialise on a previous attempt".to_owned())
                }
                _ => Ok(()),
            };
        }

        let result = gpu::nvjpeg::NvJpegDecoder::new(0);
        drop(_guard);

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
                    eprintln!(
                        "pdf_raster: nvJPEG unavailable ({e}); \
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
pub(crate) fn ensure_nvjpeg2k(policy: BackendPolicy) -> Result<(), String> {
    NVJPEG2K_DEC.with(|cell| {
        if !matches!(*cell.borrow(), DecoderInit::Uninitialised) {
            return match &*cell.borrow() {
                DecoderInit::Failed if matches!(policy, BackendPolicy::ForceCuda) => {
                    Err("nvJPEG2000 decoder failed to initialise on a previous attempt".to_owned())
                }
                _ => Ok(()),
            };
        }

        let _guard = DECODER_INIT_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if !matches!(*cell.borrow(), DecoderInit::Uninitialised) {
            drop(_guard);
            return match &*cell.borrow() {
                DecoderInit::Failed if matches!(policy, BackendPolicy::ForceCuda) => {
                    Err("nvJPEG2000 decoder failed to initialise on a previous attempt".to_owned())
                }
                _ => Ok(()),
            };
        }

        let result = gpu::nvjpeg2k::NvJpeg2kDecoder::new(0);
        drop(_guard);

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
                    eprintln!(
                        "pdf_raster: nvJPEG2000 unavailable ({e}); \
                         JPEG 2000 images will be decoded on CPU for this thread"
                    );
                    Ok(())
                }
            }
        }
    })
}

/// Try to initialise this thread's VA-API JPEG decoder.
#[cfg(feature = "vaapi")]
pub(crate) fn ensure_vaapi(drm_node: &str, policy: BackendPolicy) -> Result<(), String> {
    VAAPI_JPEG_DEC.with(|cell| {
        if !matches!(*cell.borrow(), DecoderInit::Uninitialised) {
            return match &*cell.borrow() {
                DecoderInit::Failed if matches!(policy, BackendPolicy::ForceVaapi) => {
                    Err("VA-API JPEG decoder failed to initialise on a previous attempt".to_owned())
                }
                _ => Ok(()),
            };
        }

        let _guard = DECODER_INIT_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if !matches!(*cell.borrow(), DecoderInit::Uninitialised) {
            drop(_guard);
            return match &*cell.borrow() {
                DecoderInit::Failed if matches!(policy, BackendPolicy::ForceVaapi) => {
                    Err("VA-API JPEG decoder failed to initialise on a previous attempt".to_owned())
                }
                _ => Ok(()),
            };
        }

        let result = gpu::vaapi::VapiJpegDecoder::new(drm_node);
        drop(_guard);

        match result {
            Ok(dec) => {
                *cell.borrow_mut() = DecoderInit::Ready(Some(dec));
                Ok(())
            }
            Err(e) => {
                *cell.borrow_mut() = DecoderInit::Failed;
                if matches!(policy, BackendPolicy::ForceVaapi) {
                    Err(format!("VA-API unavailable on {drm_node}: {e}"))
                } else {
                    log::info!(
                        "pdf_raster: VA-API JPEG unavailable ({e}); \
                         JPEG images will be decoded on CPU for this thread"
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
pub(crate) fn release_nvjpeg_this_thread() {
    NVJPEG_DEC.with(|cell| {
        *cell.borrow_mut() = DecoderInit::Uninitialised;
    });
}

#[cfg(feature = "nvjpeg2k")]
pub(crate) fn release_nvjpeg2k_this_thread() {
    NVJPEG2K_DEC.with(|cell| {
        *cell.borrow_mut() = DecoderInit::Uninitialised;
    });
}
