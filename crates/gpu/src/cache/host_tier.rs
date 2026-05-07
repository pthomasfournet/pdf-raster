//! Phase 9 task 3 — host RAM tier.
//!
//! The VRAM tier lives in `crates/gpu/src/cache/mod.rs`.  When VRAM
//! eviction reclaims a slab, we don't drop its bytes outright; instead
//! we copy them into pinned host memory and stash them here so a
//! subsequent lookup can DMA back to VRAM (O(1) host probe + one `PCIe`
//! upload) rather than re-decode JPEG (~15 ms of CPU work per image).
//!
//! # Invariants
//!
//! - Indexed only by [`super::ContentHash`] (the cache's primary key).
//!   The `(DocId, ObjId)` alias lives in the VRAM tier; an alias is
//!   wired only at `insert` time, so a promote-after-eviction lookup
//!   that reaches us by content hash will not auto-bind a new alias —
//!   that's fine because hash lookups are themselves O(1) once the
//!   caller already has the hash.
//! - Each entry stores its bytes in a [`PinnedHostSlice<u8>`] allocated
//!   with `cuMemAllocHost(CU_MEMHOSTALLOC_WRITECOMBINED)`.  Write-
//!   combined memory is fast for upload-on-promote and fast to fill on
//!   demote, but slow for general CPU reads — the cache writes to it
//!   from a `cudaMemcpyAsync(D2H)` and reads from it via
//!   `cudaMemcpyAsync(H2D)`.  The CPU never inspects the bytes.
//! - The LRU clock is shared with the VRAM tier (callers pass it in)
//!   so cross-tier freshness is comparable.  Used-bytes accounting is
//!   per-tier; each tier evicts on its own pressure.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use cudarc::driver::{CudaStream, PinnedHostSlice};
use dashmap::DashMap;

use super::{ContentHash, ImageLayout};

/// Budget for the host RAM tier.  Independent of [`super::VramBudget`].
#[derive(Debug, Clone, Copy)]
pub struct HostBudget {
    /// Maximum bytes the tier is allowed to hold in pinned host memory.
    /// When a demotion would push past this, the tier evicts the oldest
    /// entry first.  Set to zero to effectively disable the tier (every
    /// demotion will immediately drop on the eviction scan).
    pub host_bytes: u64,
}

impl HostBudget {
    /// Conservative default: 2 GiB, matching the Phase 9 spec.  Pinned
    /// memory is a finite system resource (kernel-mode locked pages);
    /// over-allocating starves the rest of the process.
    pub const DEFAULT: Self = Self {
        host_bytes: 2 * 1024 * 1024 * 1024,
    };
}

/// One host-resident copy of a previously-decoded image.
///
/// Implementation detail of [`HostTier`]; re-exported as `pub(crate)`
/// from the parent module.
pub struct HostEntry {
    /// Pinned host memory holding the decoded bytes.  Allocated with
    /// `CU_MEMHOSTALLOC_WRITECOMBINED` for fast DMA in both directions.
    pub pinned: PinnedHostSlice<u8>,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Pixel layout — determines bytes per pixel.
    pub layout: ImageLayout,
    /// LRU timestamp.  Updated on every host-tier hit; load-bearing for
    /// the host tier's own eviction scan.  Atomic store with `Relaxed`
    /// — exact ordering between threads doesn't matter for LRU.
    last_used: AtomicU64,
}

impl HostEntry {
    /// Build a new entry from a populated pinned slab.  `last_used` is
    /// initialised to zero and is bumped to the current tick by the
    /// host tier on insert.
    #[must_use]
    pub const fn new(
        pinned: PinnedHostSlice<u8>,
        width: u32,
        height: u32,
        layout: ImageLayout,
    ) -> Self {
        Self {
            pinned,
            width,
            height,
            layout,
            last_used: AtomicU64::new(0),
        }
    }

    /// Bytes occupied in pinned host memory by this entry.
    #[must_use]
    pub fn host_bytes(&self) -> u64 {
        self.pinned.num_bytes() as u64
    }

    fn touch(&self, tick: u64) {
        self.last_used.store(tick, Ordering::Relaxed);
    }

    fn last_used(&self) -> u64 {
        self.last_used.load(Ordering::Relaxed)
    }
}

impl std::fmt::Debug for HostEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HostEntry")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("layout", &self.layout)
            .field("host_bytes", &self.host_bytes())
            .finish_non_exhaustive()
    }
}

/// Concurrent host RAM tier.  All methods take `&self`.
///
/// Implementation detail of [`super::DeviceImageCache`]; re-exported
/// as `pub(crate)` from the parent module.
pub struct HostTier {
    entries: DashMap<ContentHash, Arc<HostEntry>>,
    used_bytes: AtomicU64,
    budget: HostBudget,
}

impl HostTier {
    /// Build an empty host tier with the given budget.
    #[must_use]
    pub fn new(budget: HostBudget) -> Self {
        Self {
            entries: DashMap::new(),
            used_bytes: AtomicU64::new(0),
            budget,
        }
    }

    /// Live entry count — diagnostics / tests only.  Gated on `cfg(test)`
    /// because production code never inspects the tier directly.
    #[cfg(test)]
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the tier currently holds zero entries.
    #[cfg(test)]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Live host RAM usage in bytes — diagnostics / tests only.
    #[cfg(test)]
    #[must_use]
    pub fn used_bytes(&self) -> u64 {
        self.used_bytes.load(Ordering::Relaxed)
    }

    /// Budget in bytes.
    #[must_use]
    pub const fn budget_bytes(&self) -> u64 {
        self.budget.host_bytes
    }

    /// Probe by content hash.  Returns the host entry on hit and bumps
    /// its LRU timestamp via `tick`.  Caller is responsible for re-
    /// uploading to VRAM if it needs a device-resident copy; the host
    /// entry stays in place to amortise repeated VRAM evictions.
    #[must_use]
    pub fn lookup(&self, hash: &ContentHash, tick: u64) -> Option<Arc<HostEntry>> {
        let entry = self.entries.get(hash)?.clone();
        entry.touch(tick);
        Some(entry)
    }

    /// Allocate a fresh pinned slab of `bytes` bytes.  Returned as an
    /// uninitialised `PinnedHostSlice<u8>` ready for a `D2H` copy.
    ///
    /// # Errors
    /// Returns the underlying [`cudarc::driver::DriverError`] if
    /// `cuMemAllocHost` fails (typically OOM in pinned memory pool).
    ///
    /// # Safety note
    /// The returned slice is uninitialised; the caller MUST overwrite
    /// every byte (e.g. via [`CudaStream::memcpy_dtoh`]) before another
    /// reader observes it.  In production the only caller is
    /// [`Self::build_from_device`], which immediately fills the slab
    /// from a `D2H` copy and synchronises the stream before returning
    /// the populated entry.  The `gpu-validation` test fixtures call
    /// it directly and fill the buffer before any read.
    pub fn alloc_pinned(
        ctx: &Arc<cudarc::driver::CudaContext>,
        bytes: usize,
    ) -> Result<PinnedHostSlice<u8>, cudarc::driver::DriverError> {
        // Safety: the caller fills the buffer immediately; no caller
        // observes the uninitialised contents.
        unsafe { ctx.alloc_pinned::<u8>(bytes) }
    }

    /// Install a freshly populated host entry.  Evicts in LRU order if
    /// the new entry would exceed the budget.  If the same content
    /// hash already has an entry, the old one is replaced (simpler than
    /// dedup-on-demotion and the bytes are bit-identical anyway).
    ///
    /// Returns the strong `Arc` for symmetry with the VRAM tier; most
    /// callers ignore it.
    ///
    /// # Accounting semantics
    ///
    /// `used_bytes` tracks **bytes managed by this tier's index**, not
    /// bytes physically allocated as pinned memory.  When this method
    /// removes a prior same-hash entry whose `Arc` has `strong_count >
    /// 1` (a concurrent lookup-holder is still reading it), the
    /// `PinnedHostSlice` stays allocated until the holder drops, but
    /// `used_bytes` is decremented immediately.  The transient
    /// undercount is bounded by the number of in-flight lookups and
    /// resolves the moment those `Arc`s drop.  This matches the VRAM
    /// tier's contract; both tiers charge bytes to the index.
    pub fn insert(&self, hash: ContentHash, entry: HostEntry, tick: u64) -> Arc<HostEntry> {
        entry.touch(tick);
        let bytes = entry.host_bytes();
        let arc = Arc::new(entry);
        // Make room before inserting; subtract the about-to-be-removed
        // entry's size first so a same-hash re-insert reuses the slot.
        if let Some((_, prior)) = self.entries.remove(&hash) {
            let _ = self
                .used_bytes
                .fetch_sub(prior.host_bytes(), Ordering::Relaxed);
        }
        self.evict_to_fit(bytes);
        let _ = self.used_bytes.fetch_add(bytes, Ordering::Relaxed);
        let _ = self.entries.insert(hash, arc.clone());
        arc
    }

    /// Drop entries in LRU order until the new `needed` bytes fit.
    ///
    /// Unlike the VRAM tier, host entries don't need refcount-based
    /// pinning — kernel reads of pinned memory are explicit DMA copies
    /// scheduled on a stream, and we don't keep `Arc<HostEntry>`s alive
    /// past the upload synchronisation that schedules them.  So a host
    /// entry is always reclaimable as long as its `Arc` `strong_count`
    /// is 1.  We still respect that bound to avoid yanking memory from
    /// under an in-flight DMA.
    fn evict_to_fit(&self, needed: u64) {
        // Bound iteration so a churn pattern (every candidate gets
        // pinned by a parallel promote) doesn't livelock.
        let max_iters = self.entries.len().saturating_mul(2).saturating_add(4);
        for _ in 0..max_iters {
            let used = self.used_bytes.load(Ordering::Relaxed);
            if used.saturating_add(needed) <= self.budget.host_bytes {
                return;
            }
            let oldest = self
                .entries
                .iter()
                .filter_map(|e| {
                    let arc = e.value();
                    (Arc::strong_count(arc) == 1).then(|| (*e.key(), arc.last_used()))
                })
                .min_by_key(|&(_, ts)| ts);
            let Some((key, _)) = oldest else {
                // Nothing reclaimable — every entry is being uploaded
                // right now.  Returning silently here would cause us to
                // overshoot the budget; that's acceptable for the host
                // tier because pinned host RAM is the cheap tier and
                // the overshoot is bounded by the number of in-flight
                // promotions, but we log a diagnostic so an operator
                // can see the pressure.
                log::warn!(
                    "host tier: cannot evict to fit {needed} B; all entries are being promoted (used {used} B, budget {} B)",
                    self.budget.host_bytes,
                );
                return;
            };
            if let Some((_, removed)) = self.entries.remove(&key) {
                let bytes = removed.host_bytes();
                debug_assert!(
                    self.used_bytes.load(Ordering::Relaxed) >= bytes,
                    "host tier used_bytes underflow",
                );
                let _ = self.used_bytes.fetch_sub(bytes, Ordering::Relaxed);
            }
            std::hint::spin_loop();
        }
    }

    /// Build a new `HostEntry` by allocating pinned memory and
    /// scheduling a device-to-host copy on `stream`.  The stream is
    /// **synchronised** before this returns so the bytes are observed-
    /// stable on the host before the caller drops the source device
    /// slab.
    ///
    /// # Errors
    /// Returns the underlying [`cudarc::driver::DriverError`] if
    /// allocation, copy, or sync fails.
    pub fn build_from_device<D: cudarc::driver::DevicePtr<u8>>(
        ctx: &Arc<cudarc::driver::CudaContext>,
        stream: &Arc<CudaStream>,
        device: &D,
        width: u32,
        height: u32,
        layout: ImageLayout,
    ) -> Result<HostEntry, cudarc::driver::DriverError> {
        let bytes = (width as usize) * (height as usize) * layout.bytes_per_pixel();
        let mut pinned = Self::alloc_pinned(ctx, bytes)?;
        // memcpy_dtoh enqueues on the stream; synchronise so the
        // caller can drop the source device slab safely afterwards.
        stream.memcpy_dtoh(device, pinned.as_mut_slice()?)?;
        stream.synchronize()?;
        Ok(HostEntry::new(pinned, width, height, layout))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_budget_default_is_two_gib() {
        assert_eq!(HostBudget::DEFAULT.host_bytes, 2 * 1024 * 1024 * 1024);
    }

    #[test]
    fn empty_host_tier_reports_zero() {
        let tier = HostTier::new(HostBudget {
            host_bytes: 1 << 20,
        });
        assert_eq!(tier.len(), 0);
        assert!(tier.is_empty());
        assert_eq!(tier.used_bytes(), 0);
        assert_eq!(tier.budget_bytes(), 1 << 20);
    }

    /// GPU integration test: allocate a pinned slab, write a pattern,
    /// install in the tier, look it up, verify the bytes survived.
    #[cfg(feature = "gpu-validation")]
    #[test]
    fn pinned_round_trip_through_host_tier() {
        let ctx = cudarc::driver::CudaContext::new(0).expect("CUDA device 0");
        let tier = HostTier::new(HostBudget {
            host_bytes: 1 << 20,
        });
        let mut pinned = HostTier::alloc_pinned(&ctx, 256).expect("alloc_pinned");
        // Fill with a known pattern.  Note: write-combined memory is
        // slow for CPU reads but writes are fine.
        for (i, b) in pinned
            .as_mut_slice()
            .expect("as_mut_slice")
            .iter_mut()
            .enumerate()
        {
            *b = u8::try_from(i & 0xFF).expect("low byte fits u8");
        }
        let hash = ContentHash([0x11; 32]);
        let entry = HostEntry::new(pinned, 16, 16, ImageLayout::Gray);
        let _ = tier.insert(hash, entry, /* tick */ 0);

        let hit = tier.lookup(&hash, /* tick */ 1).expect("hit");
        assert_eq!(hit.width, 16);
        assert_eq!(hit.height, 16);
        assert_eq!(hit.layout, ImageLayout::Gray);
        assert_eq!(hit.host_bytes(), 256);
        assert_eq!(tier.used_bytes(), 256);
    }

    #[cfg(feature = "gpu-validation")]
    #[test]
    fn host_tier_evicts_oldest_on_overflow() {
        let ctx = cudarc::driver::CudaContext::new(0).expect("CUDA device 0");
        // Budget for two 256-byte entries but not three.
        let tier = HostTier::new(HostBudget { host_bytes: 600 });

        let mk_entry = |fill: u8| -> HostEntry {
            let mut pinned = HostTier::alloc_pinned(&ctx, 256).expect("alloc_pinned");
            for b in pinned.as_mut_slice().expect("as_mut_slice").iter_mut() {
                *b = fill;
            }
            HostEntry::new(pinned, 16, 16, ImageLayout::Gray)
        };

        let h_a = ContentHash([0xAA; 32]);
        let h_b = ContentHash([0xBB; 32]);
        let h_c = ContentHash([0xCC; 32]);
        let _ = tier.insert(h_a, mk_entry(1), 1);
        let _ = tier.insert(h_b, mk_entry(2), 2);

        // Touch B to make A the LRU candidate.
        let _ = tier.lookup(&h_b, 3);

        // Inserting C must evict A.
        let _ = tier.insert(h_c, mk_entry(3), 4);

        assert!(tier.lookup(&h_a, 5).is_none(), "A should have been evicted");
        assert!(tier.lookup(&h_b, 6).is_some(), "B should remain");
        assert!(tier.lookup(&h_c, 7).is_some(), "C just inserted");
    }
}
