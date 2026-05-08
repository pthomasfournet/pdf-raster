//! VRAM budget for the image cache.
//!
//! This is the cache's *budget cap* — the maximum bytes the cache is allowed
//! to manage on the device.  Distinct from [`crate::backend::VramBudget`],
//! which is a *driver query result* (`{total_bytes, usable_bytes}`) returned
//! by `GpuBackend::detect_vram_budget`.  Don't conflate the two: this struct
//! is what the cache enforces; the backend's `VramBudget` is what the driver
//! reports the GPU can hold.

use cudarc::driver::CudaStream;

use super::CacheError;

/// VRAM budget for the cache.  See `CacheBudget` in the Phase 9 spec for
/// the full three-tier struct; this is the VRAM-only subset.
#[derive(Debug, Clone, Copy)]
pub struct VramBudget {
    /// Maximum bytes the cache is allowed to hold on the device.  When a
    /// new entry would exceed this, the cache evicts in LRU order until
    /// the new entry fits or there is nothing more to evict.
    pub vram_bytes: u64,
}

impl VramBudget {
    /// Spec-mandated VRAM hard cap: 6 GiB.  Caps `auto_detect` so we
    /// don't claim every byte on a workstation GPU shared with a
    /// desktop compositor, and serves as the value of [`Self::DEFAULT`]
    /// for tests and callers who can't probe the device.
    pub const HARD_CAP_BYTES: u64 = 6 * 1024 * 1024 * 1024;

    /// Conservative default for tests and non-GPU contexts.  Production
    /// code should prefer [`Self::auto_detect`] when a stream is on hand.
    pub const DEFAULT: Self = Self {
        vram_bytes: Self::HARD_CAP_BYTES,
    };

    /// Auto-tune from `cudaMemGetInfo`.  Returns `min(75% of free,
    /// HARD_CAP_BYTES)`, matching the spec's "default min(75% of free
    /// VRAM, 6 GB)" rule.
    ///
    /// # Errors
    /// Returns an error if `cudaMemGetInfo` fails (e.g. no CUDA device).
    pub fn auto_detect(stream: &CudaStream) -> Result<Self, CacheError> {
        let ctx = stream.context();
        let (free, _total) = ctx.mem_get_info().map_err(CacheError::cuda)?;
        // Saturating mul guards against absurd `free` values from a buggy driver.
        let three_quarters = (free as u64).saturating_mul(3) / 4;
        Ok(Self {
            vram_bytes: three_quarters.min(Self::HARD_CAP_BYTES),
        })
    }
}
