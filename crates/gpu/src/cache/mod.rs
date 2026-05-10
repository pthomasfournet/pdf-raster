//! Phase 9 device-resident image cache — three tiers.
//!
//! See `docs/superpowers/specs/2026-05-07-phase-9-device-resident-image-cache.md`
//! for the full architecture.  This module ships:
//!
//! - VRAM tier (Task 2): primary `DashMap<ContentHash, Arc<...>>` +
//!   `(DocId, ObjId)` alias index, refcount-pinned LRU eviction.
//! - Host RAM tier (Task 3): pinned-memory demote-on-evict /
//!   promote-on-hit; in-process only.
//! - Disk tier (Task 5): `<root>/<doc>/<hash>.bin` sidecar files for
//!   cross-process persistence; opt-in via [`DeviceImageCache::with_disk`].
//!
//! # Module layout
//!
//! - [`budget`] — VRAM cap struct; auto-detect from `cudaMemGetInfo`.
//! - [`eviction`] — LRU scan, demote-to-host, monotonic LRU clock.
//! - [`promotion`] — host-tier and disk-tier promote-on-hit paths.
//! - [`host_tier`] — pinned-memory tier (`pub(crate)`).
//! - [`disk_tier`] — sidecar-file tier (`pub`).
//! - [`page_buffer`] — per-page composition target ([`DevicePageBuffer`]).
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

mod budget;
mod disk_tier;
mod eviction;
mod host_tier;
mod page_buffer;
mod promotion;

pub use crate::RGBA_BPP;
pub use budget::VramBudget;
pub use disk_tier::{DiskTier, LookupCallbackError};
pub use host_tier::HostBudget;
pub(crate) use host_tier::HostTier;
pub use page_buffer::DevicePageBuffer;

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
    pub(super) last_used: AtomicU64,
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

    pub(super) fn touch(&self, tick: u64) {
        // Relaxed is sufficient: LRU ordering is approximate and doesn't
        // synchronise any other state.
        self.last_used.store(tick, Ordering::Relaxed);
    }

    pub(super) fn last_used(&self) -> u64 {
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
    pub(super) const fn cuda(e: cudarc::driver::DriverError) -> Self {
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
    pub(super) primary: DashMap<ContentHash, Arc<CachedDeviceImage>>,
    /// Maps `(DocId, ObjId)` to the content hash that resolves it.  An
    /// alias entry is cheap (40 bytes) and lets us skip BLAKE3 hashing
    /// on every same-document re-render.
    pub(super) by_doc_obj: DashMap<(DocId, ObjId), ContentHash>,
    /// Host RAM demotion target.  Populated when the eviction path
    /// reclaims a VRAM slab; consulted by [`Self::lookup_by_hash`] on
    /// a primary miss.
    pub(super) host: HostTier,
    /// Phase 9 disk tier — sidecar cache directory keyed by
    /// `(DocId, ContentHash)`.  `None` disables disk persistence
    /// (fully in-process cache).  Wired in via [`Self::with_disk`].
    pub(super) disk: Option<DiskTier>,
    pub(super) budget: VramBudget,
    /// Monotonic LRU clock.  Incremented on every observable cache
    /// event in either tier.
    pub(super) tick: AtomicU64,
    /// Sum of `vram_bytes()` over every entry currently held by the
    /// primary index.  Maintained on insert / evict.
    pub(super) used_bytes: AtomicU64,
    /// Owned CUDA stream for upload work.  Held by `Arc` so per-call
    /// upload paths can clone cheaply, and so the host tier's D2H copy
    /// can synchronise on it before publishing a demoted entry.  The
    /// CUDA context is reachable as `self.stream.context()`.
    pub(super) stream: Arc<CudaStream>,
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
    /// a host-RAM demotion budget.  The disk tier is off by default;
    /// enable it via [`Self::with_disk`].
    #[must_use]
    pub fn new(stream: Arc<CudaStream>, vram: VramBudget, host: HostBudget) -> Self {
        Self {
            primary: DashMap::new(),
            by_doc_obj: DashMap::new(),
            host: HostTier::new(host),
            disk: None,
            budget: vram,
            tick: AtomicU64::new(0),
            used_bytes: AtomicU64::new(0),
            stream,
        }
    }

    /// Attach a disk tier for cross-process persistence.
    ///
    /// On a VRAM + host-RAM miss, a disk hit avoids the CPU re-decode
    /// pass: `NVMe` at roughly 1 GB/s sustained beats `zune-jpeg`'s
    /// per-image throughput because disk reads run in parallel with
    /// the CPU-busy decode workers.  Editing the source PDF
    /// invalidates the disk cache automatically because the doc-id
    /// is content-hashed (callers should derive `DocId` from PDF
    /// bytes, not the file path).
    #[must_use]
    pub fn with_disk(mut self, disk: DiskTier) -> Self {
        self.disk = Some(disk);
        self
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
        // Try the in-process tiers first; on miss, fall through to the
        // disk tier which is keyed by `(doc, hash)` (cross-document
        // dedup is in-process only — the disk format scopes entries
        // to the doc-id directory for clean eviction granularity).
        if let Some(entry) = self.lookup_by_hash(&hash) {
            return Some(entry);
        }
        self.promote_from_disk(doc, &hash)
    }

    /// Probe the cache by content hash.  On a VRAM hit, returns the
    /// device-resident handle and bumps LRU.  On a VRAM miss, falls
    /// through to the host tier and promotes back to VRAM (one `PCIe`
    /// upload).  Returns `None` only if the hash is in neither tier or
    /// the promotion upload fails.
    ///
    /// Does NOT probe the disk tier — that's keyed by `(doc, hash)`
    /// and only [`Self::lookup_by_id`] has the doc-id needed for the
    /// lookup.
    #[must_use]
    pub fn lookup_by_hash(&self, hash: &ContentHash) -> Option<Arc<CachedDeviceImage>> {
        if let Some(entry) = self.primary.get(hash) {
            let entry = entry.clone();
            entry.touch(self.next_tick());
            return Some(entry);
        }
        self.promote_from_host(hash)
    }

    /// Three-tier lookup keyed on a content hash + the doc-id used
    /// to scope disk-tier entries.  Consults VRAM, host RAM, and
    /// disk in that order.  Used by `decode_dct` after the
    /// `(doc, obj)` alias miss + BLAKE3 hash compute, so a cold
    /// process with a populated disk cache hits the disk tier
    /// instead of paying for a re-decode.
    ///
    /// Re-binds the alias on a hit so subsequent same-document
    /// lookups go through `lookup_by_id` and skip the hash compute.
    #[must_use]
    pub fn lookup_by_hash_for_doc(
        &self,
        doc: DocId,
        obj: ObjId,
        hash: &ContentHash,
    ) -> Option<Arc<CachedDeviceImage>> {
        if let Some(entry) = self.lookup_by_hash(hash) {
            self.alias(doc, obj, *hash);
            return Some(entry);
        }
        let promoted = self.promote_from_disk(doc, hash)?;
        self.alias(doc, obj, *hash);
        Some(promoted)
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
                // Persist to disk for cross-process / cross-session
                // reuse.  Best-effort: write errors are logged inside
                // `disk.insert` and don't fail the in-memory insert
                // — the renderer always gets a valid `Arc<...>`.
                if let Some(disk) = self.disk.as_ref() {
                    disk.insert(doc, hash, width, height, layout, pixels);
                }
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
