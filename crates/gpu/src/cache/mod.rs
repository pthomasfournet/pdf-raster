//! Phase 9 device-resident image cache — VRAM and host RAM tiers.
//!
//! See `docs/superpowers/specs/2026-05-07-phase-9-device-resident-image-cache.md`
//! for the full architecture.  This module ships the in-process VRAM
//! tier (Task 2) and the pinned host RAM demotion target (Task 3).
//! The disk persistence tier lands in Task 5.
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
//! out of `DeviceImageCache::lookup_by_*` holds another.  Eviction drops the
//! cache's strong reference, leaving the slab alive until all in-flight
//! kernels finish.  This means a slab can stay live in VRAM after eviction
//! until the last consumer drops its `Arc` — that's the desired behaviour:
//! pulling memory from under an in-flight kernel would corrupt the page.

mod host_tier;
mod page_buffer;

pub use host_tier::HostBudget;
pub(crate) use host_tier::HostTier;
pub use page_buffer::{DevicePageBuffer, RGBA_BPP};

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use cudarc::driver::{CudaSlice, CudaStream, DeviceRepr};
use dashmap::DashMap;
use dashmap::mapref::entry::Entry;

/// Stable identifier for a PDF document.
///
/// Used as part of the secondary cache key.  Today this is just the
/// SHA-256 of the original PDF bytes (or any other 32-byte
/// content-addressable identifier the caller chooses); the cache
/// treats it as opaque.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DocId(pub [u8; 32]);

/// Stable identifier for an image within a PDF document.
///
/// Combined with [`DocId`] it forms the secondary alias key for fast
/// same-document lookup that bypasses content hashing.
///
/// **Note:** PDF's authoritative object identifier is `(number, generation)`
/// (see `pdf::ObjectId`).  This newtype intentionally drops the generation
/// because image `XObjects` are written once and never bumped in any PDF
/// produced by tools we encounter.  If a workflow ever rewrites image
/// objects with new generations, callers must construct distinct `ObjId`s
/// (e.g. by hashing `(number, generation)` into the u32) to keep the alias
/// index from returning stale content.
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
/// [`DeviceImageCache::lookup_by_id`], [`DeviceImageCache::lookup_by_hash`],
/// or [`DeviceImageCache::insert`], the slab is guaranteed alive for the
/// lifetime of the `Arc` — eviction may drop the cache's reference, but
/// the device memory is reclaimed only when the last `Arc` drops.  This
/// guarantee is what makes the cache safe to use across multiple
/// in-flight CUDA streams.
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
        // CudaSlice::len is not yet const in cudarc 0.19; can't make
        // this const fn until it is.  Caller perf is fine — this is
        // only used in `evict_to_fit` and diagnostic accessors.
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
    /// The cache cannot fit a new entry because every reclaimable slab is
    /// pinned by an outstanding `Arc` (e.g. an in-flight CUDA stream).
    /// This condition is **transient**: once consumers drop their `Arc`s
    /// — typically after `cudaStreamSynchronize` on the kernel that read
    /// the slab — retry will succeed.
    OverBudget {
        /// Bytes the new entry needs.
        needed: u64,
        /// Total cache occupancy at the time of the failure.  This counts
        /// every entry held by `primary`, including ones whose only
        /// reference is the cache itself; the eviction scan tried and
        /// could not reclaim enough of them.
        used: u64,
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
    /// `insert` was called with a zero-length pixel buffer.  cudarc
    /// rejects zero-size allocations, so we surface this as a typed
    /// caller error rather than a `Cuda` driver error.
    EmptyPayload,
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
                used,
                budget,
            } => write!(
                f,
                "cache over budget (transient): needed {needed} B, {used} B held by entries pinned by in-flight kernels, budget {budget} B — retry once outstanding Arcs drop",
            ),
            Self::EntryExceedsBudget { needed, budget } => write!(
                f,
                "single image {needed} B does not fit in cache budget of {budget} B",
            ),
            Self::EmptyPayload => write!(f, "insert called with zero-length pixel buffer"),
        }
    }
}

impl std::error::Error for CacheError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Cuda(e) => Some(e),
            Self::OverBudget { .. } | Self::EntryExceedsBudget { .. } | Self::EmptyPayload => None,
        }
    }
}

/// 32-byte BLAKE3 content hash used as the cache's primary key.
///
/// Newtype so it can't be confused with [`DocId`] (also `[u8; 32]`)
/// at a call site — different roles, different keys.  Construct via
/// [`DeviceImageCache::hash_bytes`] or `ContentHash(bytes)` directly
/// when the caller already has the hash.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ContentHash(pub [u8; 32]);

/// Two-tier in-process image cache.
///
/// VRAM is the hot tier; pinned host RAM is the demotion target.
///
/// The VRAM tier is indexed by content hash (primary, for cross-
/// document dedup) and by `(DocId, ObjId)` (secondary, fast same-
/// document lookup).  The host tier is indexed only by content hash
/// and is consulted on a VRAM miss before re-decoding.
///
/// All methods take `&self` and are safe to call from many threads
/// concurrently.  Eviction is read-then-CAS rather than locking the
/// whole tier, so a hot path that touches a present entry never
/// contends with eviction.
///
/// LRU clock (`tick`) is shared across tiers so a host-tier hit's
/// "freshness" is comparable to a VRAM hit's; otherwise a freshly
/// promoted slab would look like the oldest VRAM entry to the next
/// eviction.
pub struct DeviceImageCache {
    primary: DashMap<ContentHash, Arc<CachedDeviceImage>>,
    /// Maps `(DocId, ObjId)` to the content hash that resolves it.  An
    /// alias entry is cheap (40 bytes) and lets us skip BLAKE3 hashing
    /// on every same-document re-render.
    by_doc_obj: DashMap<(DocId, ObjId), ContentHash>,
    /// Host RAM demotion target.  Populated when [`Self::evict_to_fit`]
    /// reclaims a VRAM slab; consulted by [`Self::lookup_by_hash`] on
    /// a primary miss.
    host: HostTier,
    budget: VramBudget,
    /// Monotonic LRU clock.  Incremented on every observable cache
    /// event in either tier.
    tick: AtomicU64,
    /// Sum of `vram_bytes()` over every entry currently held by the
    /// primary index.  Maintained on insert / evict.
    used_bytes: AtomicU64,
    /// Owned CUDA stream for upload work.  Held by `Arc` so per-call
    /// upload paths can clone cheaply, and so the host tier's D2H copy
    /// can synchronise on it before publishing a demoted entry.  The
    /// CUDA context is reachable as `self.stream.context()`.
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
    /// Build an empty cache bound to a CUDA stream, a VRAM budget, and
    /// a host-RAM demotion budget.
    #[must_use]
    pub fn new(stream: Arc<CudaStream>, vram: VramBudget, host: HostBudget) -> Self {
        Self {
            primary: DashMap::new(),
            by_doc_obj: DashMap::new(),
            host: HostTier::new(host),
            budget: vram,
            tick: AtomicU64::new(0),
            used_bytes: AtomicU64::new(0),
            stream,
        }
    }

    /// Compute the BLAKE3 content hash of an encoded byte stream.  Exposed
    /// for callers that already have a hash and want to skip recomputation.
    #[must_use]
    pub fn hash_bytes(bytes: &[u8]) -> ContentHash {
        ContentHash(*blake3::hash(bytes).as_bytes())
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

    /// Probe the cache by `(DocId, ObjId)`.  Returns `None` if the
    /// alias is unknown or its content hash is in neither tier; on a
    /// host-tier hit, transparently promotes to VRAM (one `PCIe`
    /// upload).  Updates LRU.
    #[must_use]
    pub fn lookup_by_id(&self, doc: DocId, obj: ObjId) -> Option<Arc<CachedDeviceImage>> {
        let hash = *self.by_doc_obj.get(&(doc, obj))?;
        self.lookup_by_hash(&hash)
    }

    /// Probe the cache by content hash.  On a VRAM hit, returns the
    /// device-resident handle and bumps LRU.  On a VRAM miss, falls
    /// through to the host tier and promotes back to VRAM (one `PCIe`
    /// upload).  Returns `None` only if the hash is in neither tier or
    /// the promotion upload fails.
    #[must_use]
    pub fn lookup_by_hash(&self, hash: &ContentHash) -> Option<Arc<CachedDeviceImage>> {
        if let Some(entry) = self.primary.get(hash) {
            let entry = entry.clone();
            entry.touch(self.next_tick());
            return Some(entry);
        }
        self.promote_from_host(hash)
    }

    /// Lift a host-tier entry back into VRAM and install it in the
    /// primary index.  Returns `None` if the host tier doesn't have it
    /// or the upload fails (treated as a cache miss; caller re-decodes).
    fn promote_from_host(&self, hash: &ContentHash) -> Option<Arc<CachedDeviceImage>> {
        let host_entry = self.host.lookup(hash, self.next_tick())?;
        let bytes = host_entry.pinned.num_bytes() as u64;
        // Make room in the VRAM tier.  If we can't (every entry pinned
        // by in-flight kernels), fall back to a cache miss; the host
        // entry stays in place for a future retry.
        if self.evict_to_fit(bytes).is_err() {
            return None;
        }
        // Pass the `PinnedHostSlice` directly so cudarc's `HostSlice`
        // impl records the H→D copy against the slice's internal event
        // — without that, dropping `host_entry` while the DMA is still
        // in flight would let `PinnedHostSlice::Drop` free the source
        // buffer mid-copy.  Calling `as_slice()` first would route
        // through the plain `[u8]` impl whose `SyncOnDrop` is a no-op.
        let device_ptr = self
            .stream
            .clone_htod(&host_entry.pinned)
            .map_err(|e| log::warn!("host-tier promotion upload failed: {e}"))
            .ok()?;
        let new_entry = Arc::new(CachedDeviceImage {
            device_ptr,
            width: host_entry.width,
            height: host_entry.height,
            layout: host_entry.layout,
            last_used: AtomicU64::new(self.next_tick()),
        });
        match self.primary.entry(*hash) {
            Entry::Occupied(occ) => {
                // Another thread won the same race — return their entry.
                Some(occ.get().clone())
            }
            Entry::Vacant(vac) => {
                let _ = self
                    .used_bytes
                    .fetch_add(new_entry.vram_bytes(), Ordering::Relaxed);
                let _ = vac.insert(new_entry.clone());
                Some(new_entry)
            }
        }
    }

    /// Bind an existing primary entry to a `(DocId, ObjId)` alias.  Used
    /// after a content-hash hit to make subsequent same-document lookups
    /// O(1).
    ///
    /// # Panics
    ///
    /// In debug builds only, panics if `(doc, obj)` is already bound to
    /// a *different* hash: in a correct PDF workflow object identity is
    /// stable, so a hash change for the same `(DocId, ObjId)` is either
    /// a caller bug (mis-hashed bytes) or a BLAKE3 collision (negligible
    /// probability).  Either way, silently overwriting would corrupt the
    /// alias index.  Release builds tolerate the overwrite to keep
    /// adversarial PDFs from aborting the renderer.
    pub fn alias(&self, doc: DocId, obj: ObjId, hash: ContentHash) {
        if cfg!(debug_assertions)
            && let Some(existing) = self.by_doc_obj.get(&(doc, obj))
            && *existing != hash
        {
            panic!(
                "alias({doc:?}, {obj:?}, …) tried to overwrite {:?} with {hash:?}; \
                 PDF object id should map to a stable content hash",
                *existing,
            );
        }
        let _ = self.by_doc_obj.insert((doc, obj), hash);
    }

    /// Upload host pixels to VRAM, register them under both keys, and
    /// return the cached handle.  Evicts in LRU order if the new entry
    /// would exceed the budget.
    ///
    /// # Stream contract
    ///
    /// The upload is enqueued on the cache's internal stream (see
    /// [`Self::stream`]).  Consumers that launch kernels against
    /// [`CachedDeviceImage::device_ptr`] on a *different* stream MUST
    /// synchronise the cache's stream first (or use cudaStreamWaitEvent),
    /// otherwise the kernel may read the slab before the H→D DMA
    /// completes.  Same-stream consumption is correct without an explicit
    /// sync because cudarc serialises operations on a single stream.
    ///
    /// # Errors
    ///
    /// - [`CacheError::EmptyPayload`] — `pixels.is_empty()` (cudarc
    ///   refuses zero-size allocations; this is a typed caller error).
    /// - [`CacheError::EntryExceedsBudget`] — the entry alone is larger
    ///   than the configured budget.
    /// - [`CacheError::OverBudget`] — no eviction can free enough space
    ///   (every other entry is pinned by an outstanding `Arc`).  The
    ///   condition is transient.
    /// - [`CacheError::Cuda`] — the upload itself failed.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "all fields are trivially movable (Copy newtypes plus a `&[u8]` slice header); by-value conveys ownership transfer at the call site without changing codegen vs `&InsertRequest`"
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
        if pixels.is_empty() {
            return Err(CacheError::EmptyPayload);
        }
        // Decoder contract: pixel buffer length matches the dimensions /
        // layout the caller declared.  A mismatched buffer would land
        // wrong-shaped bytes in VRAM and surface as a corrupt blit; loud
        // failure in debug, trust the caller in release.
        debug_assert_eq!(
            pixels.len(),
            (width as usize)
                .saturating_mul(height as usize)
                .saturating_mul(layout.bytes_per_pixel()),
            "pixel buffer length does not match width × height × bpp",
        );
        let needed = pixels.len() as u64;
        if needed > self.budget.vram_bytes {
            return Err(CacheError::EntryExceedsBudget {
                needed,
                budget: self.budget.vram_bytes,
            });
        }
        if let Some(existing) = self.lookup_by_hash(&hash) {
            self.alias(doc, obj, hash);
            return Ok(existing);
        }
        // Single Entry transition below so concurrent inserts of the same
        // hash can't both fetch_add `used_bytes` after their Vacant probe.
        self.evict_to_fit(needed)?;
        let device_ptr = self.stream.clone_htod(pixels).map_err(CacheError::cuda)?;
        let new_entry = Arc::new(CachedDeviceImage {
            device_ptr,
            width,
            height,
            layout,
            last_used: AtomicU64::new(self.next_tick()),
        });
        match self.primary.entry(hash) {
            Entry::Occupied(occ) => {
                // Lost the dedup race.  `new_entry` drops on function
                // exit, freeing its just-uploaded slab; we never
                // touched `used_bytes`, so accounting stays correct.
                let existing = occ.get().clone();
                drop(occ);
                self.alias(doc, obj, hash);
                existing.touch(self.next_tick());
                Ok(existing)
            }
            Entry::Vacant(vac) => {
                let _ = self
                    .used_bytes
                    .fetch_add(new_entry.vram_bytes(), Ordering::Relaxed);
                let _ = vac.insert(new_entry.clone());
                self.alias(doc, obj, hash);
                Ok(new_entry)
            }
        }
    }

    /// The CUDA stream on which the cache enqueues uploads.  Exposed so
    /// callers using a different stream can synchronise before reading
    /// from a freshly inserted [`CachedDeviceImage::device_ptr`] — call
    /// `cache.stream().synchronize()` (or wire a `cudaStreamWaitEvent`).
    #[must_use]
    pub fn stream(&self) -> &CudaStream {
        &self.stream
    }

    /// The cache's CUDA stream as an `Arc`, for callers that need to
    /// own a shared reference (e.g. constructing a [`DevicePageBuffer`]
    /// bound to the same stream so its `download()` synchronises
    /// against blits the cache enqueues).
    #[must_use]
    pub const fn stream_arc(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// Copy the bytes of an evicted VRAM entry into pinned host memory
    /// and install in the host tier so a future lookup avoids re-decode.
    ///
    /// Skipped when the host tier already holds this hash (re-demoting
    /// would waste pinned RAM on identical bytes), or when the host
    /// budget is zero (host tier disabled), or when the D2H copy fails
    /// (logged; not propagated — demotion is a best-effort optimisation,
    /// not a correctness requirement).
    fn demote_to_host(&self, hash: ContentHash, evicted: &Arc<CachedDeviceImage>) {
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
    fn evict_to_fit(&self, needed: u64) -> Result<(), CacheError> {
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

            let oldest: Option<(ContentHash, u64)> = self
                .primary
                .iter()
                .filter_map(|e| {
                    let arc = e.value();
                    // Only entries with strong_count == 1 are reclaimable;
                    // anything else is pinned by an in-flight kernel.
                    (Arc::strong_count(arc) == 1).then(|| (*e.key(), arc.last_used()))
                })
                .min_by_key(|&(_, ts)| ts);

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
        ContentHash([byte; 32])
    }

    /// Build a 16×16 [`InsertRequest`] for a tiny test image.  The
    /// `pixels` slice must outlive the returned request — call sites
    /// keep it in a local binding.
    #[cfg(feature = "gpu-validation")]
    fn req(obj: u32, h: ContentHash, layout: ImageLayout, pixels: &[u8]) -> InsertRequest<'_> {
        InsertRequest {
            doc: doc(1),
            obj: ObjId(obj),
            hash: h,
            width: 16,
            height: 16,
            layout,
            pixels,
        }
    }

    /// Build a fresh cache on CUDA device 0 with the given VRAM budget.
    /// Returns `(stream, cache)` so tests that need the stream after the
    /// cache is built (e.g. for a readback) can use it directly.  The
    /// host tier is configured large enough to never evict so each test
    /// can isolate the behaviour under inspection; tests that exercise
    /// the host tier explicitly use `mk_cache_with_host`.
    #[cfg(feature = "gpu-validation")]
    fn mk_cache(vram_bytes: u64) -> (Arc<CudaStream>, DeviceImageCache) {
        mk_cache_with_host(vram_bytes, /* host */ 1 << 24)
    }

    #[cfg(feature = "gpu-validation")]
    fn mk_cache_with_host(vram_bytes: u64, host_bytes: u64) -> (Arc<CudaStream>, DeviceImageCache) {
        let ctx = cudarc::driver::CudaContext::new(0).expect("CUDA device 0");
        let stream = ctx.default_stream();
        let cache = DeviceImageCache::new(
            Arc::clone(&stream),
            VramBudget { vram_bytes },
            HostBudget { host_bytes },
        );
        (stream, cache)
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
        // The Error::source call below is the load-bearing proof that
        // CacheError satisfies std::error::Error (a separate trait-bound
        // assertion would just be redundant ceremony).
        assert!(std::error::Error::source(&e).is_none());

        // Pinning hash is unused; reference it to verify the helper compiles.
        let _ = hash(7);
    }

    /// GPU integration test: round-trip a 16×16 RGB image through the
    /// cache.  Gated behind `gpu-validation` so workstations without
    /// CUDA can still run `cargo test -p gpu --lib`.
    #[cfg(feature = "gpu-validation")]
    #[test]
    fn round_trip_rgb_image() {
        let (stream, cache) = mk_cache(1 << 20);

        // 16×16×3 = 768; (i % 256) always fits u8, hence the unwrap.
        let pixels: Vec<u8> = (0_i32..16 * 16 * 3)
            .map(|i| u8::try_from(i % 256).expect("i % 256 fits u8"))
            .collect();
        let h = DeviceImageCache::hash_bytes(&pixels);
        let entry = cache
            .insert(req(42, h, ImageLayout::Rgb, &pixels))
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
        // Budget for two 256-byte entries but not three.  Host tier
        // disabled so an evicted entry truly disappears (otherwise the
        // demote-then-promote machinery would resurrect it on lookup).
        let (_stream, cache) = mk_cache_with_host(600, /* host */ 0);

        let make_pixels = |fill: u8| -> Vec<u8> { vec![fill; 256] };

        let h_a = ContentHash([0xAA; 32]);
        let h_b = ContentHash([0xBB; 32]);
        let h_c = ContentHash([0xCC; 32]);

        let pixels_a = make_pixels(1);
        let pixels_b = make_pixels(2);
        let entry_a = cache
            .insert(req(1, h_a, ImageLayout::Gray, &pixels_a))
            .expect("a");
        let entry_b = cache
            .insert(req(2, h_b, ImageLayout::Gray, &pixels_b))
            .expect("b");
        // Drop the local Arcs so eviction can reclaim them.
        drop(entry_a);
        drop(entry_b);

        // Touch B so A is the oldest.
        let _ = cache.lookup_by_hash(&h_b);

        // Inserting C must evict A (oldest).
        let pixels_c = make_pixels(3);
        let _entry_c = cache
            .insert(req(3, h_c, ImageLayout::Gray, &pixels_c))
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
        // Budget for one 256-byte entry.
        let (_stream, cache) = mk_cache(300);

        let h_a = ContentHash([0xAA; 32]);
        let h_b = ContentHash([0xBB; 32]);

        // Hold the Arc — A is now pinned.
        let pixels_a = vec![1u8; 256];
        let _pinned_a = cache
            .insert(req(1, h_a, ImageLayout::Gray, &pixels_a))
            .expect("a");

        // Inserting B must fail because A is pinned and there's no other
        // entry to evict.
        let pixels_b = vec![2u8; 256];
        let err = cache
            .insert(req(2, h_b, ImageLayout::Gray, &pixels_b))
            .unwrap_err();
        match err {
            CacheError::OverBudget {
                needed,
                used,
                budget,
            } => {
                assert_eq!(needed, 256);
                assert_eq!(used, 256);
                assert_eq!(budget, 300);
            }
            other => panic!("expected OverBudget, got {other:?}"),
        }
    }

    /// Bombard the cache with concurrent inserts of overlapping hashes
    /// from many threads.  After the dust settles, the cache should hold
    /// at most one entry per unique hash, and `used_bytes` should equal
    /// the sum of those unique entries' sizes — proving the dedup-race
    /// fix in `insert` (the `Entry::Vacant` atomic transition) accounts
    /// `used_bytes` exactly once per slab.
    #[cfg(feature = "gpu-validation")]
    #[test]
    fn concurrent_inserts_dedup_correctly() {
        const THREADS: u8 = 16;
        const PER_THREAD: u8 = 64;
        const UNIQUE_HASHES: u8 = 8;
        const PIXEL_BYTES: usize = 256;

        // Generous budget so eviction doesn't enter the picture; the
        // test is only about dedup accounting.
        let (_stream, cache) = mk_cache(u64::from(UNIQUE_HASHES) * (PIXEL_BYTES as u64) * 2);
        let cache = Arc::new(cache);

        // Each thread inserts PER_THREAD requests cycling through the
        // same UNIQUE_HASHES values, so threads heavily collide on each
        // hash and each request also legitimately overwrites the prior.
        std::thread::scope(|s| {
            for t in 0..THREADS {
                let cache = Arc::clone(&cache);
                let _ = s.spawn(move || {
                    for i in 0..PER_THREAD {
                        let hi = (t.wrapping_add(i)) % UNIQUE_HASHES;
                        let h = ContentHash([hi + 1; 32]);
                        // Pixel content keyed on hash so all threads
                        // racing on the same hash upload identical bytes.
                        let pixels = vec![hi + 1; PIXEL_BYTES];
                        let _ = cache.insert(InsertRequest {
                            doc: doc(1),
                            obj: ObjId(u32::from(t) * u32::from(PER_THREAD) + u32::from(i)),
                            hash: h,
                            width: 16,
                            height: 16,
                            layout: ImageLayout::Gray,
                            pixels: &pixels,
                        });
                    }
                });
            }
        });

        // Every unique hash should have produced exactly one cache
        // entry; `len()` counts primary entries.
        let len = cache.len();
        assert!(
            len <= usize::from(UNIQUE_HASHES),
            "cache holds {len} entries, expected ≤ {UNIQUE_HASHES}",
        );
        // `used_bytes` must equal exactly `len × PIXEL_BYTES` — one
        // fetch_add per surviving Vacant transition, no double-count.
        assert_eq!(
            cache.used_bytes(),
            (len as u64) * (PIXEL_BYTES as u64),
            "used_bytes drifted: {} B for {len} entries × {PIXEL_BYTES} B",
            cache.used_bytes(),
        );
    }

    /// End-to-end host tier round trip: insert entry A, evict it by
    /// inserting B, then look up A again — it must come back via the
    /// host tier's promotion path with bit-identical bytes.
    #[cfg(feature = "gpu-validation")]
    #[test]
    fn evicted_vram_entry_promotes_back_from_host() {
        // Budget for exactly one 768-byte (16×16×3) RGB entry.  Host
        // tier large enough to hold the demoted copy.
        let (stream, cache) = mk_cache_with_host(/* vram */ 1000, /* host */ 1 << 20);

        let pixels_a: Vec<u8> = (0_i32..16 * 16 * 3)
            .map(|i| u8::try_from((i * 7) % 256).expect("fits u8"))
            .collect();
        let pixels_b = vec![0xCDu8; 16 * 16 * 3];
        let h_a = DeviceImageCache::hash_bytes(&pixels_a);
        let h_b = DeviceImageCache::hash_bytes(&pixels_b);

        // Insert A and drop the local Arc so eviction can reclaim it.
        let entry_a = cache
            .insert(req(1, h_a, ImageLayout::Rgb, &pixels_a))
            .expect("insert a");
        drop(entry_a);

        // Insert B — must evict A and demote it to the host tier.
        let entry_b = cache
            .insert(req(2, h_b, ImageLayout::Rgb, &pixels_b))
            .expect("insert b");

        // Host tier should now have A.
        assert_eq!(
            cache.host.len(),
            1,
            "host tier should hold the demoted A; got {} entries",
            cache.host.len(),
        );

        // Drop B so the upcoming promote-of-A has room in VRAM (the
        // budget is one-entry-wide; promoting A while B is pinned would
        // legitimately fail with OverBudget).
        drop(entry_b);

        // Look up A — promotion from host should succeed and the
        // returned bytes must match the original pixels exactly.  The
        // bit-for-bit readback below is the load-bearing assertion;
        // checking Arc-pointer identity isn't (the heap may reuse a
        // slot under load and produce false negatives).
        let entry_a_promoted = cache.lookup_by_hash(&h_a).expect("promote hit");
        let mut readback = vec![0u8; pixels_a.len()];
        stream
            .memcpy_dtoh(&entry_a_promoted.device_ptr, &mut readback)
            .expect("dtoh");
        stream.synchronize().expect("sync");
        assert_eq!(
            readback, pixels_a,
            "promoted bytes must match the originally inserted pixels",
        );
    }
}
