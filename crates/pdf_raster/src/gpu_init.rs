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

#[cfg(any(feature = "nvjpeg", feature = "nvjpeg2k"))]
use crate::BackendPolicy;
#[cfg(any(feature = "nvjpeg", feature = "nvjpeg2k"))]
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
#[cfg(any(feature = "nvjpeg", feature = "nvjpeg2k", feature = "vaapi"))]
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
                        "pdf_raster: nvJPEG2000 unavailable ({e}); \
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
