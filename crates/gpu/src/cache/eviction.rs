//! VRAM eviction logic for the image cache.
//!
//! Pulled out of `mod.rs` so the LRU scan, demote-to-host best-effort path,
//! and the `next_tick` LRU clock live in one focused module.  The
//! orchestration (`lookup_by_*`, `insert`) stays in `mod.rs`; this file
//! holds the mechanics.
//!
//! All methods are private to the cache and accessed through `&self` on
//! [`super::DeviceImageCache`].

use std::sync::Arc;
use std::sync::atomic::Ordering;

use dashmap::DashMap;

use super::host_tier::HostTier;
use super::{CacheError, CachedDeviceImage, ContentHash, DeviceImageCache};

impl DeviceImageCache {
    /// Copy the bytes of an evicted VRAM entry into pinned host memory
    /// and install in the host tier so a future lookup avoids re-decode.
    ///
    /// Skipped when the host tier already holds this hash (re-demoting
    /// would waste pinned RAM on identical bytes), or when the host
    /// budget is zero (host tier disabled), or when the D2H copy fails
    /// (logged; not propagated — demotion is a best-effort optimisation,
    /// not a correctness requirement).
    pub(super) fn demote_to_host(&self, hash: ContentHash, evicted: &Arc<CachedDeviceImage>) {
        if self.host.budget_bytes() == 0 {
            return;
        }
        if self.host.lookup(&hash, self.next_tick()).is_some() {
            return;
        }
        match HostTier::build_from_device(
            self.stream.context(),
            &self.stream,
            &evicted.device_ptr,
            evicted.width,
            evicted.height,
            evicted.layout,
        ) {
            Ok(host_entry) => {
                let _ = self.host.insert(hash, host_entry, self.next_tick());
            }
            Err(e) => {
                log::warn!(
                    "demotion D2H copy failed (hash={hash:?}, {}×{} {:?}): {e}",
                    evicted.width,
                    evicted.height,
                    evicted.layout,
                );
            }
        }
    }

    /// Evict in LRU order until `needed` bytes of free budget exist.
    ///
    /// Each evicted entry is demoted to the host tier (best effort)
    /// before its device memory is released, so a subsequent lookup
    /// can DMA back from pinned host RAM rather than re-decode the
    /// source bytes.
    ///
    /// Returns [`CacheError::OverBudget`] when nothing more is reclaimable
    /// (every entry is pinned by an outstanding `Arc`) or when an
    /// internal safety bound is hit — the latter is a livelock guard
    /// against pathological pin churn under heavy concurrent inserts.
    pub(super) fn evict_to_fit(&self, needed: u64) -> Result<(), CacheError> {
        // Bound the loop so a starvation pattern (every candidate gets
        // pinned the moment we read its strong_count) cannot spin
        // forever.  `2 × len() + 4` lets every entry be considered
        // twice plus a small constant; in practice most calls reclaim
        // in one or two iterations.
        let max_iters = self.primary.len().saturating_mul(2).saturating_add(4);
        for _ in 0..max_iters {
            let used = self.used_bytes.load(Ordering::Relaxed);
            if used
                .checked_add(needed)
                .is_some_and(|n| n <= self.budget.vram_bytes)
            {
                return Ok(());
            }

            let oldest = pick_oldest_unpinned(&self.primary);

            let Some((key, _)) = oldest else {
                return Err(CacheError::OverBudget {
                    needed,
                    used: self.used_bytes.load(Ordering::Relaxed),
                    budget: self.budget.vram_bytes,
                });
            };

            // A concurrent `lookup_by_*` may race in and clone the Arc
            // between the strong_count check above and `primary.remove`
            // below.  When that happens the slab is removed from the
            // index but kept alive by the lookup-holder's Arc — VRAM
            // stays occupied until they drop it.  We still subtract its
            // bytes from `used_bytes` because the cache budget tracks
            // "bytes managed by the cache", not "bytes physically live
            // in VRAM"; the lookup-holder's bytes are charged to them
            // until their Arc drops.
            if let Some((_, removed)) = self.primary.remove(&key) {
                let bytes = removed.vram_bytes();
                debug_assert!(
                    self.used_bytes.load(Ordering::Relaxed) >= bytes,
                    "used_bytes underflow: tracked {} B, evicting entry holds {bytes} B",
                    self.used_bytes.load(Ordering::Relaxed),
                );
                let _ = self.used_bytes.fetch_sub(bytes, Ordering::Relaxed);
                // Demote to host RAM if not already cached there.  The
                // copy runs on `self.stream` and synchronises before
                // returning, so when `removed` (and its `CudaSlice`)
                // drops at end of scope the bytes are stable on host.
                self.demote_to_host(key, &removed);
            }
            // Invite the OS scheduler if budget hasn't moved.  Cheap on x86.
            std::hint::spin_loop();
        }
        // Hit the safety bound — almost certainly because of churn.
        // Surface as OverBudget so the caller can retry or back off
        // rather than block indefinitely.
        Err(CacheError::OverBudget {
            needed,
            used: self.used_bytes.load(Ordering::Relaxed),
            budget: self.budget.vram_bytes,
        })
    }

    /// Monotonic LRU clock tick.  Relaxed atomic — ordering between
    /// ticks of different threads is not load-bearing for LRU
    /// correctness, only monotonicity per thread, which `fetch_add`
    /// guarantees.
    pub(super) fn next_tick(&self) -> u64 {
        self.tick.fetch_add(1, Ordering::Relaxed)
    }
}

/// Scan the primary index for the oldest unpinned entry.
///
/// "Unpinned" means `Arc::strong_count == 1` — the cache itself holds
/// the only strong reference, so dropping it is safe (no in-flight
/// kernel can read from the freed slab).  Returns `None` when every
/// entry is pinned by an outstanding `Arc`; the caller surfaces that
/// as [`CacheError::OverBudget`].
fn pick_oldest_unpinned(
    primary: &DashMap<ContentHash, Arc<CachedDeviceImage>>,
) -> Option<(ContentHash, u64)> {
    primary
        .iter()
        .filter_map(|e| {
            let arc = e.value();
            (Arc::strong_count(arc) == 1).then(|| (*e.key(), arc.last_used()))
        })
        .min_by_key(|&(_, ts)| ts)
}
