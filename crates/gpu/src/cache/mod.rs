//! Phase 9 device-resident image cache — VRAM tier.
//!
//! See `docs/superpowers/specs/2026-05-07-phase-9-device-resident-image-cache.md`
//! for the full architecture.  This module ships only the in-process VRAM
//! tier (Task 2); the host RAM and disk tiers land in tasks 3 and 5.
//!
//! # Concurrency model
//!
//! The cache is `&self` everywhere and safe to call from any number of
//! rayon worker threads.  Internal indices are [`dashmap`]-backed; LRU
//! timestamps are atomic; refcounts come for free from `Arc`.
//!
//! # Lifetime model
//!
//! [`CachedDeviceImage`] is wrapped in `Arc`.  The cache holds one strong
//! reference; every consumer (renderer, blit kernel) that takes a value
//! out of [`DeviceImageCache::lookup`] holds another.  Eviction drops the
//! cache's strong reference, leaving the slab alive until all in-flight
//! kernels finish.  This means a slab can stay live in VRAM after eviction
//! until the last consumer drops its `Arc` — that's the desired behaviour:
//! pulling memory from under an in-flight kernel would corrupt the page.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use cudarc::driver::{CudaSlice, CudaStream, DeviceRepr};
use dashmap::DashMap;

/// Stable identifier for a PDF document.
///
/// Used as part of the secondary cache key.  Today this is just the
/// SHA-256 of the original PDF bytes (or any other 32-byte
/// content-addressable identifier the caller chooses); the cache
/// treats it as opaque.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DocId(pub [u8; 32]);

/// Stable identifier for an image within a PDF document.  This is the
/// PDF object number; combined with [`DocId`] it forms the secondary key
/// for fast same-document lookup that bypasses content hashing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ObjId(pub u32);

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
    /// Auto-tune from `cudaMemGetInfo`.  Returns `min(75% of free, 6 GiB)`,
    /// matching the spec's "default min(75% of free VRAM, 6 GB)" rule.
    ///
    /// # Errors
    /// Returns an error if `cudaMemGetInfo` fails (e.g. no CUDA device).
    pub fn auto_detect(stream: &CudaStream) -> Result<Self, CacheError> {
        // Hard ceiling so we don't claim every byte on a workstation GPU
        // shared with a desktop compositor; matches the spec.
        const HARD_CAP: u64 = 6 * 1024 * 1024 * 1024;
        let ctx = stream.context();
        let (free, _total) = ctx.mem_get_info().map_err(CacheError::cuda)?;
        // Saturating mul guards against absurd `free` values from a buggy driver.
        let three_quarters = (free as u64).saturating_mul(3) / 4;
        Ok(Self {
            vram_bytes: three_quarters.min(HARD_CAP),
        })
    }
}

/// Component layout of a cached decoded image.  Matches `ImageColorSpace`
/// in `pdf_interp` but kept here to avoid a circular dependency.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageLayout {
    /// One byte per pixel: gray.
    Gray,
    /// Three bytes per pixel: R, G, B.
    Rgb,
    /// One byte per pixel: 0x00 = paint, 0xFF = transparent.  Byte-expanded
    /// from a logical 1-bit-per-pixel mask.
    Mask,
}

impl ImageLayout {
    /// Bytes per pixel for the byte-expanded layout.
    #[must_use]
    pub const fn bytes_per_pixel(self) -> usize {
        match self {
            Self::Gray | Self::Mask => 1,
            Self::Rgb => 3,
        }
    }
}

/// One cached decoded image living in VRAM.
///
/// `Arc<CachedDeviceImage>` is the only public handle.  Once obtained via
/// [`DeviceImageCache::lookup`] or [`DeviceImageCache::insert`], the slab
/// is guaranteed alive for the lifetime of the `Arc` — eviction may drop
/// the cache's reference, but the device memory is reclaimed only when
/// the last `Arc` drops.  This guarantee is what makes the cache safe to
/// use across multiple in-flight CUDA streams.
pub struct CachedDeviceImage {
    /// Owned device-side allocation.  Dropping `CachedDeviceImage` frees it.
    pub device_ptr: CudaSlice<u8>,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Pixel layout — determines bytes per pixel and renderer interpretation.
    pub layout: ImageLayout,
    /// LRU timestamp.  Updated on every cache hit via a single atomic store
    /// (no CAS needed; later writes always win and exact ordering doesn't
    /// matter for LRU correctness).
    last_used: AtomicU64,
}

impl CachedDeviceImage {
    /// Bytes in VRAM occupied by this entry.  Used for budget accounting.
    #[must_use]
    pub fn vram_bytes(&self) -> u64 {
        self.device_ptr.len() as u64
    }

    fn touch(&self, tick: u64) {
        // Relaxed is sufficient: LRU ordering is approximate and doesn't
        // synchronise any other state.
        self.last_used.store(tick, Ordering::Relaxed);
    }

    fn last_used(&self) -> u64 {
        self.last_used.load(Ordering::Relaxed)
    }
}

impl std::fmt::Debug for CachedDeviceImage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CachedDeviceImage")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("layout", &self.layout)
            .field("vram_bytes", &self.vram_bytes())
            .finish_non_exhaustive()
    }
}

/// Errors from cache operations.
#[derive(Debug)]
pub enum CacheError {
    /// A `cudarc` driver call failed (allocation, upload, sync, or budget probe).
    Cuda(cudarc::driver::DriverError),
    /// The cache cannot fit a new entry even after evicting every evictable
    /// entry — i.e. all live entries are pinned by outstanding `Arc`s and
    /// the new entry is too large.  Caller must back off (drop some `Arc`s)
    /// or grow the budget.
    OverBudget {
        /// Bytes the new entry needs.
        needed: u64,
        /// Bytes currently used by pinned entries.
        pinned: u64,
        /// Total budget in bytes.
        budget: u64,
    },
    /// A new entry alone exceeds the entire VRAM budget.  Either the
    /// budget is misconfigured or the image is unusably large.
    EntryExceedsBudget {
        /// Bytes the new entry needs.
        needed: u64,
        /// Total budget in bytes.
        budget: u64,
    },
}

impl CacheError {
    const fn cuda(e: cudarc::driver::DriverError) -> Self {
        Self::Cuda(e)
    }
}

impl std::fmt::Display for CacheError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cuda(e) => write!(f, "cuda: {e}"),
            Self::OverBudget {
                needed,
                pinned,
                budget,
            } => write!(
                f,
                "cache over budget: needed {needed} B, {pinned} B pinned by in-flight kernels, budget {budget} B",
            ),
            Self::EntryExceedsBudget { needed, budget } => write!(
                f,
                "single image {needed} B does not fit in cache budget of {budget} B",
            ),
        }
    }
}

impl std::error::Error for CacheError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Cuda(e) => Some(e),
            Self::OverBudget { .. } | Self::EntryExceedsBudget { .. } => None,
        }
    }
}

/// 32-byte BLAKE3 content hash used as the cache's primary key.
type ContentHash = [u8; 32];

/// One in-process VRAM tier.  Indexed by content hash (primary, for
/// cross-document dedup) and by `(DocId, ObjId)` (secondary, fast same-
/// document lookup).
///
/// All methods take `&self` and are safe to call from many threads
/// concurrently.  Eviction is read-then-CAS rather than locking the
/// whole tier, so a hot path that touches a present entry never
/// contends with eviction.
pub struct DeviceImageCache {
    primary: DashMap<ContentHash, Arc<CachedDeviceImage>>,
    /// Maps `(DocId, ObjId)` to the content hash that resolves it.  An
    /// alias entry is cheap (40 bytes) and lets us skip BLAKE3 hashing
    /// on every same-document re-render.
    by_doc_obj: DashMap<(DocId, ObjId), ContentHash>,
    budget: VramBudget,
    /// Monotonic LRU clock.  Incremented on every observable cache event.
    tick: AtomicU64,
    /// Sum of `vram_bytes()` over every entry currently held by the
    /// primary index.  Maintained on insert / evict.
    used_bytes: AtomicU64,
    /// Owned CUDA stream for upload work.  Held by `Arc` so per-call
    /// upload paths can clone cheaply.
    stream: Arc<CudaStream>,
}

/// Bundled arguments for [`DeviceImageCache::insert`].
///
/// Passed by struct rather than by parameter list because the call site
/// is a hot but readable construction; eight-positional `insert(doc,
/// obj, hash, w, h, layout, pixels)` would be easy to misread (which
/// dimension is which?), and adding a tier later (host RAM, disk) is a
/// non-breaking field addition rather than another parameter.
#[derive(Debug)]
pub struct InsertRequest<'a> {
    /// Document this image belongs to (for the secondary alias index).
    pub doc: DocId,
    /// PDF object id within the document (for the secondary alias index).
    pub obj: ObjId,
    /// Content hash (BLAKE3) of the encoded source bytes — primary key
    /// for cross-document deduplication.  Caller computes via
    /// [`DeviceImageCache::hash_bytes`].
    pub hash: ContentHash,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Pixel layout — determines bytes per pixel.
    pub layout: ImageLayout,
    /// Decoded host-side pixels.  Length must equal
    /// `width × height × layout.bytes_per_pixel()`; the cache trusts
    /// the caller and uploads the slice verbatim.
    pub pixels: &'a [u8],
}

impl DeviceImageCache {
    /// Build an empty cache bound to a CUDA stream and a VRAM budget.
    #[must_use]
    pub fn new(stream: Arc<CudaStream>, budget: VramBudget) -> Self {
        Self {
            primary: DashMap::new(),
            by_doc_obj: DashMap::new(),
            budget,
            tick: AtomicU64::new(0),
            used_bytes: AtomicU64::new(0),
            stream,
        }
    }

    /// Compute the BLAKE3 content hash of an encoded byte stream.  Exposed
    /// for callers that already have a hash and want to skip recomputation.
    #[must_use]
    pub fn hash_bytes(bytes: &[u8]) -> ContentHash {
        *blake3::hash(bytes).as_bytes()
    }

    /// Live entry count — for diagnostics, tests, and benches.
    #[must_use]
    pub fn len(&self) -> usize {
        self.primary.len()
    }

    /// Whether the cache currently holds zero entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.primary.is_empty()
    }

    /// Live VRAM usage in bytes.
    #[must_use]
    pub fn used_bytes(&self) -> u64 {
        self.used_bytes.load(Ordering::Relaxed)
    }

    /// Budget in bytes.
    #[must_use]
    pub const fn budget_bytes(&self) -> u64 {
        self.budget.vram_bytes
    }

    /// Probe the cache by `(DocId, ObjId)`.  Returns `None` if the alias
    /// is unknown or its content hash has been evicted from the primary.
    /// Updates LRU on hit.
    #[must_use]
    pub fn lookup_by_id(&self, doc: DocId, obj: ObjId) -> Option<Arc<CachedDeviceImage>> {
        let hash = *self.by_doc_obj.get(&(doc, obj))?;
        let entry = self.primary.get(&hash)?.clone();
        entry.touch(self.next_tick());
        Some(entry)
    }

    /// Probe the cache by content hash.  Returns `None` if the entry has
    /// been evicted.  Updates LRU on hit.
    #[must_use]
    pub fn lookup_by_hash(&self, hash: &ContentHash) -> Option<Arc<CachedDeviceImage>> {
        let entry = self.primary.get(hash)?.clone();
        entry.touch(self.next_tick());
        Some(entry)
    }

    /// Bind an existing primary entry to a `(DocId, ObjId)` alias.  Used
    /// after a content-hash hit to make subsequent same-document lookups
    /// O(1).
    pub fn alias(&self, doc: DocId, obj: ObjId, hash: ContentHash) {
        let _ = self.by_doc_obj.insert((doc, obj), hash);
    }

    /// Upload host pixels to VRAM, register them under both keys, and
    /// return the cached handle.  Evicts in LRU order if the new entry
    /// would exceed the budget.
    ///
    /// # Errors
    ///
    /// Returns [`CacheError::EntryExceedsBudget`] if the entry alone is
    /// larger than the budget; [`CacheError::OverBudget`] if no eviction
    /// can free enough space (every other entry is pinned by an
    /// outstanding `Arc`); [`CacheError::Cuda`] if the upload itself fails.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "by-value conveys 'caller hands over a fully built request'; every field is Copy so this is the same code as &-borrow"
    )]
    pub fn insert(&self, req: InsertRequest<'_>) -> Result<Arc<CachedDeviceImage>, CacheError> {
        let InsertRequest {
            doc,
            obj,
            hash,
            width,
            height,
            layout,
            pixels,
        } = req;
        let needed = pixels.len() as u64;
        if needed > self.budget.vram_bytes {
            return Err(CacheError::EntryExceedsBudget {
                needed,
                budget: self.budget.vram_bytes,
            });
        }
        // Make room before allocating; dropping device memory is async via
        // the stream so freeing happens before the new alloc completes.
        self.evict_to_fit(needed)?;
        let device_ptr = self.stream.clone_htod(pixels).map_err(CacheError::cuda)?;

        let entry = Arc::new(CachedDeviceImage {
            device_ptr,
            width,
            height,
            layout,
            last_used: AtomicU64::new(self.next_tick()),
        });

        // Insert primary; if a parallel insert already won, prefer the
        // existing entry and let our local upload drop on function exit.
        if let Some(existing) = self.primary.get(&hash) {
            let existing = existing.clone();
            // Wire the alias regardless — same-document lookups should
            // still resolve quickly even if we lost the dedup race.
            self.alias(doc, obj, hash);
            existing.touch(self.next_tick());
            return Ok(existing);
        }
        let _ = self
            .used_bytes
            .fetch_add(entry.vram_bytes(), Ordering::Relaxed);
        let _ = self.primary.insert(hash, entry.clone());
        self.alias(doc, obj, hash);
        Ok(entry)
    }

    /// Evict in LRU order until `needed` bytes of free budget exist.
    /// Returns `Ok` once enough has been freed; an [`CacheError::OverBudget`]
    /// is returned only when the cache cannot reach the target even after
    /// evicting every evictable entry (i.e. all live entries are pinned).
    fn evict_to_fit(&self, needed: u64) -> Result<(), CacheError> {
        loop {
            let used = self.used_bytes.load(Ordering::Relaxed);
            // We need (used + needed) ≤ budget.
            if used
                .checked_add(needed)
                .is_some_and(|n| n <= self.budget.vram_bytes)
            {
                return Ok(());
            }

            // Snapshot the current LRU candidate.  Iterating a DashMap
            // takes a per-shard read lock; we collect into a small vec
            // rather than holding the lock across the remove call.
            let oldest: Option<(ContentHash, u64, u64, usize)> = self
                .primary
                .iter()
                .map(|e| {
                    let arc = e.value();
                    (
                        *e.key(),
                        arc.last_used(),
                        arc.vram_bytes(),
                        Arc::strong_count(arc),
                    )
                })
                // Only entries with strong_count == 1 are reclaimable;
                // others are pinned by in-flight kernels.
                .filter(|&(_, _, _, refs)| refs == 1)
                .min_by_key(|&(_, ts, _, _)| ts);

            let Some((key, _, bytes, _)) = oldest else {
                // Nothing reclaimable.  Compute pinned size for the error.
                let pinned = self.used_bytes.load(Ordering::Relaxed);
                return Err(CacheError::OverBudget {
                    needed,
                    pinned,
                    budget: self.budget.vram_bytes,
                });
            };

            // Remove from primary; the slab drops when no consumer holds
            // an Arc, releasing VRAM.  We don't touch by_doc_obj here:
            // a stale alias becomes a content-hash miss on next lookup,
            // which is correct fallback behaviour.
            if self.primary.remove(&key).is_some() {
                let _ = self.used_bytes.fetch_sub(bytes, Ordering::Relaxed);
            }
            // Loop and re-check the budget; multiple evictions may be needed.
        }
    }

    fn next_tick(&self) -> u64 {
        // Relaxed is fine: ordering between ticks of different threads is
        // not load-bearing for LRU correctness, only monotonicity per
        // thread, which `fetch_add` guarantees.
        self.tick.fetch_add(1, Ordering::Relaxed)
    }
}

// Compile-time assertion: the cache and its handle are thread-safe.
// `CudaSlice<u8>` is `Send + Sync` (cudarc requires `T: DeviceRepr`,
// which `u8` satisfies); the rest of the state is atomics, `dashmap`,
// and `Arc<CudaStream>`.  The `DeviceRepr` bound on `u8` is referenced
// to keep the file compiling if cudarc ever loosens the `T` bound.
const fn _assert_thread_safety()
where
    DeviceImageCache: Send + Sync,
    Arc<CachedDeviceImage>: Send + Sync,
    u8: DeviceRepr,
{
}

#[cfg(test)]
mod tests {
    use super::*;

    fn doc(byte: u8) -> DocId {
        DocId([byte; 32])
    }

    fn hash(byte: u8) -> ContentHash {
        [byte; 32]
    }

    /// Smoke test: basic API shape sanity that does not require a CUDA
    /// device.  Verifies that newtype keys and the budget struct
    /// round-trip and have the documented properties.
    #[test]
    fn doc_id_and_obj_id_are_hashable() {
        use std::collections::HashSet;
        let mut set: HashSet<(DocId, ObjId)> = HashSet::new();
        let _ = set.insert((doc(0), ObjId(1)));
        let _ = set.insert((doc(0), ObjId(2)));
        let _ = set.insert((doc(0), ObjId(1))); // dup
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn image_layout_bytes_per_pixel() {
        assert_eq!(ImageLayout::Gray.bytes_per_pixel(), 1);
        assert_eq!(ImageLayout::Mask.bytes_per_pixel(), 1);
        assert_eq!(ImageLayout::Rgb.bytes_per_pixel(), 3);
    }

    #[test]
    fn blake3_hash_is_deterministic() {
        let bytes = b"hello phase 9";
        let h1 = DeviceImageCache::hash_bytes(bytes);
        let h2 = DeviceImageCache::hash_bytes(bytes);
        assert_eq!(h1, h2);
        // Different input → different hash with overwhelming probability.
        let h3 = DeviceImageCache::hash_bytes(b"hello phase 8");
        assert_ne!(h1, h3);
    }

    #[test]
    fn cache_error_is_std_error() {
        let e = CacheError::EntryExceedsBudget {
            needed: 1 << 30,
            budget: 1 << 20,
        };
        // Display contains the byte counts so an operator can read it.
        let msg = e.to_string();
        assert!(msg.contains("1073741824"));
        assert!(msg.contains("1048576"));
        // It implements Error and has no source for non-CUDA variants.
        assert!(std::error::Error::source(&e).is_none());

        // Compile-time: `impl Error` is satisfied.
        fn takes_error<E: std::error::Error>(_: &E) {}
        takes_error(&e);

        // Pinning hash is unused; reference it to verify the helper compiles.
        let _ = hash(7);
    }

    /// GPU integration test: round-trip a 16×16 RGB image through the
    /// cache.  Gated behind `gpu-validation` so workstations without
    /// CUDA can still run `cargo test -p gpu --lib`.
    #[cfg(feature = "gpu-validation")]
    #[test]
    fn round_trip_rgb_image() {
        let ctx = cudarc::driver::CudaContext::new(0).expect("CUDA device 0");
        let stream = ctx.default_stream();
        let budget = VramBudget {
            vram_bytes: 1 << 20,
        };
        let cache = DeviceImageCache::new(stream.clone(), budget);

        let pixels: Vec<u8> = (0..16 * 16 * 3).map(|i| (i % 256) as u8).collect();
        let h = DeviceImageCache::hash_bytes(&pixels);
        let entry = cache
            .insert(InsertRequest {
                doc: doc(1),
                obj: ObjId(42),
                hash: h,
                width: 16,
                height: 16,
                layout: ImageLayout::Rgb,
                pixels: &pixels,
            })
            .expect("insert");
        assert_eq!(entry.width, 16);
        assert_eq!(entry.height, 16);

        // Lookup by both keys hits.
        let by_id = cache.lookup_by_id(doc(1), ObjId(42)).expect("alias hit");
        let by_hash = cache.lookup_by_hash(&h).expect("primary hit");
        assert!(Arc::ptr_eq(&entry, &by_id));
        assert!(Arc::ptr_eq(&entry, &by_hash));

        // Read back to confirm the upload landed correctly.
        let mut readback = vec![0u8; pixels.len()];
        stream
            .memcpy_dtoh(&entry.device_ptr, &mut readback)
            .expect("dtoh");
        stream.synchronize().expect("sync");
        assert_eq!(readback, pixels);

        // Used bytes accounting reflects the entry.
        assert_eq!(cache.used_bytes(), pixels.len() as u64);
    }

    #[cfg(feature = "gpu-validation")]
    #[test]
    fn lru_evicts_oldest_unpinned_entry() {
        let ctx = cudarc::driver::CudaContext::new(0).expect("CUDA device 0");
        let stream = ctx.default_stream();
        // Budget large enough for two 256-byte entries but not three.
        let budget = VramBudget { vram_bytes: 600 };
        let cache = DeviceImageCache::new(stream.clone(), budget);

        let make_pixels = |fill: u8| -> Vec<u8> { vec![fill; 256] };

        let h_a = [0xAA; 32];
        let h_b = [0xBB; 32];
        let h_c = [0xCC; 32];

        let pixels_a = make_pixels(1);
        let pixels_b = make_pixels(2);
        let _entry_a = cache
            .insert(InsertRequest {
                doc: doc(1),
                obj: ObjId(1),
                hash: h_a,
                width: 16,
                height: 16,
                layout: ImageLayout::Gray,
                pixels: &pixels_a,
            })
            .expect("a");
        let _entry_b = cache
            .insert(InsertRequest {
                doc: doc(1),
                obj: ObjId(2),
                hash: h_b,
                width: 16,
                height: 16,
                layout: ImageLayout::Gray,
                pixels: &pixels_b,
            })
            .expect("b");
        // Drop the local Arcs so eviction can reclaim them.
        drop(_entry_a);
        drop(_entry_b);

        // Touch B so A is the oldest.
        let _ = cache.lookup_by_hash(&h_b);

        // Inserting C must evict A (oldest).
        let pixels_c = make_pixels(3);
        let _entry_c = cache
            .insert(InsertRequest {
                doc: doc(1),
                obj: ObjId(3),
                hash: h_c,
                width: 16,
                height: 16,
                layout: ImageLayout::Gray,
                pixels: &pixels_c,
            })
            .expect("c");

        assert!(
            cache.lookup_by_hash(&h_a).is_none(),
            "A should have been evicted"
        );
        assert!(cache.lookup_by_hash(&h_b).is_some(), "B should remain");
        assert!(cache.lookup_by_hash(&h_c).is_some(), "C just inserted");
    }

    #[cfg(feature = "gpu-validation")]
    #[test]
    fn pinned_entries_block_eviction() {
        let ctx = cudarc::driver::CudaContext::new(0).expect("CUDA device 0");
        let stream = ctx.default_stream();
        // Budget for one 256-byte entry.
        let budget = VramBudget { vram_bytes: 300 };
        let cache = DeviceImageCache::new(stream.clone(), budget);

        let h_a = [0xAA; 32];
        let h_b = [0xBB; 32];

        // Hold the Arc — A is now pinned.
        let pixels_a = vec![1u8; 256];
        let _pinned_a = cache
            .insert(InsertRequest {
                doc: doc(1),
                obj: ObjId(1),
                hash: h_a,
                width: 16,
                height: 16,
                layout: ImageLayout::Gray,
                pixels: &pixels_a,
            })
            .expect("a");

        // Inserting B must fail because A is pinned and there's no other
        // entry to evict.
        let pixels_b = vec![2u8; 256];
        let err = cache
            .insert(InsertRequest {
                doc: doc(1),
                obj: ObjId(2),
                hash: h_b,
                width: 16,
                height: 16,
                layout: ImageLayout::Gray,
                pixels: &pixels_b,
            })
            .unwrap_err();
        match err {
            CacheError::OverBudget {
                needed,
                pinned,
                budget,
            } => {
                assert_eq!(needed, 256);
                assert_eq!(pinned, 256);
                assert_eq!(budget, 300);
            }
            other => panic!("expected OverBudget, got {other:?}"),
        }
    }
}
