// `pub(crate)` items inside a `pub(crate)` module trigger clippy::redundant_pub_crate
// when nursery lints are enabled, but the explicitness aids readability in a module
// that is only conditionally compiled and accessed from a single call site.
#![expect(
    clippy::redundant_pub_crate,
    reason = "explicitness aids readability for conditionally-compiled single-call-site module"
)]
//! Construction helpers that build [`gpu::DecodeQueue`] instances for each
//! supported hardware JPEG decoder backend.
//!
//! These functions follow the same `BackendPolicy` semantics as the TLS-based
//! `ensure_*` helpers in `gpu_init`: soft failures are logged and silently
//! ignored on `Auto`; hard failures on `Force*` policies are returned as `Err`.
//!
//! Construction is serialised through [`crate::gpu_init::DECODER_INIT_LOCK`]
//! because `nvjpegCreateEx` and `vaInitialize` are not safe to call concurrently.

use crate::{BackendPolicy, gpu_init::DECODER_INIT_LOCK};

/// Build a VA-API JPEG decode queue.
///
/// Opens a `VapiJpegDecoder` on `drm_node`, wraps it in a
/// [`gpu::DecodeQueue`], and returns it ready for use.
///
/// Returns `Ok(None)` (after logging) when the driver is unavailable and
/// `policy` is `Auto`.  Returns `Err(msg)` when `policy` is `ForceVaapi` and
/// initialisation fails, so the caller can surface it as
/// [`crate::RasterError::BackendUnavailable`].
#[cfg(feature = "vaapi")]
pub(crate) fn build_vaapi_queue(
    drm_node: &str,
    policy: BackendPolicy,
) -> Result<Option<gpu::DecodeQueue<gpu::vaapi::VapiJpegDecoder>>, String> {
    if matches!(policy, BackendPolicy::CpuOnly | BackendPolicy::ForceCuda) {
        return Ok(None);
    }

    let guard = DECODER_INIT_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let result = gpu::vaapi::VapiJpegDecoder::new(drm_node);
    drop(guard);

    match result {
        Ok(dec) => {
            let queue = gpu::DecodeQueue::new(dec, "vaapi-jpeg-worker");
            Ok(Some(queue))
        }
        Err(e) => {
            if matches!(policy, BackendPolicy::ForceVaapi) {
                Err(format!(
                    "VA-API JPEG decoder unavailable on {drm_node}: {e}"
                ))
            } else {
                log::warn!(
                    "pdf_raster: VA-API JPEG unavailable ({e}); \
                     JPEG images will be decoded on CPU"
                );
                Ok(None)
            }
        }
    }
}
