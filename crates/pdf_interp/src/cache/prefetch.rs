//! Background image-cache prefetcher.
//!
//! At session start the renderer typically has no idea which images
//! it'll need until it actually walks the content stream of each
//! page.  For multi-page or multi-pass workloads (OCR pipelines,
//! re-renders of the same PDF) priming the cache up front means the
//! first render hits the cache instead of paying the JPEG decode +
//! upload cost serially on the render hot path.
//!
//! This module:
//! 1. Walks `doc.get_pages()`; for each page reads `/Resources
//!    /XObject` and lists the named image references.
//! 2. Hands each `(page, name, stream_id)` tuple to a small worker
//!    pool that decodes + inserts into the cache.
//! 3. Dedupes by `ObjId` so an image referenced from N pages is
//!    decoded once, not N times.
//!
//! Only `/DCTDecode` (JPEG) images are prefetched — those are the
//! decode-expensive case and the one [`gpu::cache::DeviceImageCache`]
//! was built for.  CCITT, JBIG2, and JPEG-2000 images decode on the
//! renderer's first touch.
//!
//! Workers are plain `std::thread`s, not rayon, so prefetch never
//! steals capacity from the page-render pool.  The default worker
//! count is intentionally low (2) — the goal is to be done before
//! the renderer reaches the image, not to saturate the CPU.

use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::thread;

use gpu::cache::{DeviceImageCache, DocId, ObjId};
use pdf::{Document, Object, ObjectId};

use crate::resources::image::{ImageFilter, filter_name};
use crate::resources::resolve_image;

/// Upper bound on prefetcher worker threads.
///
/// Caps an arbitrary caller-supplied [`PrefetchConfig::workers`]
/// value so a misconfig (typo, overflow from `num_cpus * factor`)
/// can't spawn thousands of threads.  Beyond ~16 workers the
/// JPEG-decode pool would contend with the renderer's rayon pool
/// anyway.
pub const MAX_PREFETCH_WORKERS: usize = 16;

// Compile-time invariant: the `clamp(1, MAX_PREFETCH_WORKERS)` in
// `spawn_prefetch` would panic at runtime if MAX_PREFETCH_WORKERS
// were ever set to 0 (clamp requires min ≤ max).  Catch that at
// build time so the panic can't happen.
const _: () = assert!(MAX_PREFETCH_WORKERS >= 1);

/// Clamp a caller-supplied worker count to `[1, MAX_PREFETCH_WORKERS]`.
/// Extracted so the unit-tested clamp is the same code the spawn
/// site runs — a regression that drops the clamp in one place
/// alone would only escape detection if both copies were dropped.
const fn clamp_workers(n: usize) -> usize {
    if n < 1 {
        1
    } else if n > MAX_PREFETCH_WORKERS {
        MAX_PREFETCH_WORKERS
    } else {
        n
    }
}

/// Tunables for [`spawn_prefetch`].
#[derive(Debug, Clone, Copy)]
pub struct PrefetchConfig {
    /// Number of background worker threads.  Default 2; bumping this
    /// only helps if the renderer hasn't started yet (i.e. the
    /// session was opened well in advance of the first render call).
    /// Clamped to the range `1..=MAX_PREFETCH_WORKERS` at spawn time
    /// (see [`MAX_PREFETCH_WORKERS`]).
    pub workers: usize,
    /// Hard cap on the number of distinct images the prefetcher will
    /// decode.  Acts as a guardrail for adversarial PDFs that list
    /// tens of thousands of image `XObject`s; the renderer can still
    /// decode unprefetched images on first touch.  Default 4096.
    pub max_images: usize,
}

impl Default for PrefetchConfig {
    fn default() -> Self {
        Self {
            workers: 2,
            max_images: 4096,
        }
    }
}

/// Outcome counters reported by the prefetcher.  Useful for cache
/// hit-rate verification and log-level diagnostics.
#[derive(Debug, Default)]
pub struct PrefetchStats {
    /// Distinct `(doc_id, obj_id)` image references discovered while
    /// walking the page tree.  Includes images that were already in
    /// the cache (so [`Self::decoded`] + [`Self::already_cached`]
    /// + [`Self::errors`] = `discovered` once the run completes).
    pub discovered: AtomicU64,
    /// Images decoded + inserted into the cache by this run.
    pub decoded: AtomicU64,
    /// Images that were already in the cache (e.g. host or disk tier
    /// hit from a prior session).  Counted but not redecoded.
    pub already_cached: AtomicU64,
    /// Images the prefetcher attempted but `resolve_image` returned
    /// `None` for — typically a malformed stream or unsupported
    /// filter.  Logged at warn level by `resolve_image`.
    pub errors: AtomicU64,
}

/// Handle to a running prefetch job.  Drop the handle to cancel
/// in-flight prefetch and join the workers.
pub struct PrefetchHandle {
    cancel: Arc<AtomicBool>,
    workers: Mutex<Vec<thread::JoinHandle<()>>>,
}

impl Drop for PrefetchHandle {
    fn drop(&mut self) {
        // Signal cancellation; workers check this between images and
        // exit promptly when set.  The mpsc receiver also drops with
        // the sender so any worker blocked on `recv` wakes up too.
        self.cancel.store(true, Ordering::Release);
        // Best effort join — `wait` may have already drained.
        // Drain under the lock, then join unlocked.
        let drained: Vec<_> = {
            let mut guard = self
                .workers
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            guard.drain(..).collect()
        };
        for handle in drained {
            let _ = handle.join();
        }
    }
}

/// Spawn a prefetch job for `doc` against `cache` keyed by `doc_id`.
///
/// Returns immediately with a [`PrefetchHandle`]; the actual work
/// happens on background threads.  The renderer can start
/// immediately — cache hits become available the moment each
/// worker finishes a decode.
///
/// # Concurrency
///
/// `Document` is `Send + Sync` and the cache is internally
/// thread-safe, so handing `Arc`s to multiple workers is safe.
/// Discovery dedupes via a single-threaded `HashSet<ObjId>` local
/// to a private `discover_pages` helper; an image referenced from
/// many pages is only decoded once.
///
/// # Panics
///
/// Panics if `std::thread::Builder::spawn` fails (typically because
/// the OS cannot create a new thread).  Treat thread-creation
/// failure as a hard error rather than silently degrading: the
/// whole point of a prefetcher is to run threads.
#[must_use]
pub fn spawn_prefetch(
    doc: Arc<Document>,
    cache: Arc<DeviceImageCache>,
    doc_id: DocId,
    config: PrefetchConfig,
) -> PrefetchHandle {
    let state = Arc::new(PrefetchState {
        cache,
        doc,
        doc_id,
        stats: Arc::new(PrefetchStats::default()),
        cancel: Arc::new(AtomicBool::new(false)),
    });

    // Unbounded mpsc; the cap is `config.max_images`, enforced by
    // the discovery loop, so memory is bounded.
    let (tx, rx) = mpsc::channel::<PrefetchJob>();
    let rx = Arc::new(Mutex::new(rx));

    // Spawn workers first so they're ready when discovery begins.
    let workers = clamp_workers(config.workers);
    let mut handles = Vec::with_capacity(workers + 1);
    for worker_idx in 0..workers {
        let rx = Arc::clone(&rx);
        let state = Arc::clone(&state);
        handles.push(
            thread::Builder::new()
                .name(format!("pdf-prefetch-{worker_idx}"))
                .spawn(move || worker_loop(&rx, &state))
                .expect("spawn prefetch worker"),
        );
    }

    // Discovery thread.  Walks pages and pushes jobs into the
    // channel; closes the channel when done so workers exit.
    let discovery = {
        let state = Arc::clone(&state);
        let max_images = config.max_images;
        thread::Builder::new()
            .name("pdf-prefetch-walk".to_string())
            .spawn(move || {
                discover_pages(&state, max_images, &tx);
                // Dropping `tx` closes the channel and tells
                // workers to exit once their queues drain.
            })
            .expect("spawn prefetch discovery thread")
    };
    handles.push(discovery);

    PrefetchHandle {
        cancel: Arc::clone(&state.cancel),
        workers: Mutex::new(handles),
    }
}

/// Job payload passed from the discovery thread to the worker pool.
struct PrefetchJob {
    page_id: ObjectId,
    image_name: Vec<u8>,
}

/// Bundled state borrowed by both [`discover_pages`] and
/// [`worker_loop`] — keeps each function's arg list sane and means
/// the spawn site only owns one set of `Arc`s.
///
/// `seen` doesn't live here: discovery is single-threaded and is
/// the only writer, so the dedup set is a plain `HashSet` owned
/// inside [`discover_pages`].
struct PrefetchState {
    cache: Arc<DeviceImageCache>,
    doc: Arc<Document>,
    doc_id: DocId,
    stats: Arc<PrefetchStats>,
    cancel: Arc<AtomicBool>,
}

/// Walk every page's `/XObject` resource dict and emit a
/// [`PrefetchJob`] for each `/DCTDecode`-filtered image `XObject` we
/// haven't already seen.
///
/// Form `XObject`s are *not* recursed into; images nested inside
/// a Form's own resource dict will not be prefetched and the
/// renderer will decode them on first touch.  The renderer already
/// pays that cost only once per image (subsequent pages hit the
/// cache), so the gap is bounded.
fn discover_pages(state: &PrefetchState, max_images: usize, tx: &mpsc::Sender<PrefetchJob>) {
    // Discovery is single-threaded so the dedup set is a plain
    // local — no Mutex / Arc.  An image referenced from N pages
    // therefore generates one PrefetchJob, not N.  Pre-sized so
    // logo-heavy decks (thousands of XObjects) skip the first few
    // rehashes; capped at max_images so the prefetcher never
    // allocates more dedup space than it'd ever populate.
    let mut seen: HashSet<ObjId> = HashSet::with_capacity(max_images.min(256));
    let mut emitted = 0usize;
    'pages: for (_page_num, page_id) in state.doc.get_pages() {
        if state.cancel.load(Ordering::Acquire) {
            break;
        }
        // get_page_resource_dict already follows the page-tree
        // /Resources inheritance chain and resolves indirect refs;
        // when /XObject is absent we just get a NotFound error and
        // skip this page.
        let Ok(xobj_dict) = state.doc.get_page_resource_dict(page_id, b"XObject") else {
            continue;
        };
        for (name, value) in &xobj_dict {
            if state.cancel.load(Ordering::Acquire) {
                break 'pages;
            }
            if emitted >= max_images {
                log::debug!("prefetch: hit max_images={max_images}, stopping discovery early");
                break 'pages;
            }
            let Object::Reference(stream_id) = value else {
                continue;
            };
            let obj_id = ObjId(stream_id.0);
            if !seen.insert(obj_id) {
                continue;
            }
            if state.cache.lookup_by_id(state.doc_id, obj_id).is_some() {
                let _ = state.stats.discovered.fetch_add(1, Ordering::Relaxed);
                let _ = state.stats.already_cached.fetch_add(1, Ordering::Relaxed);
                continue;
            }
            // Quick filter check: only push DCT images.  We re-resolve
            // the stream object inside the worker (so the worker
            // owns the decode); here we just need to confirm
            // `/Subtype Image` and `/Filter DCTDecode` cheaply.
            if !is_dct_image_stream(&state.doc, *stream_id) {
                continue;
            }
            let _ = state.stats.discovered.fetch_add(1, Ordering::Relaxed);
            if tx
                .send(PrefetchJob {
                    page_id,
                    image_name: name.clone(),
                })
                .is_err()
            {
                // Receiver dropped — handle was cancelled.
                break 'pages;
            }
            emitted += 1;
        }
    }
}

/// Worker drain loop.  Pulls one job at a time off the shared
/// receiver, decodes, and counts the outcome in `state.stats`.
/// Returns when the channel is closed (discovery thread exited) or
/// cancellation flips.
fn worker_loop(rx: &Mutex<mpsc::Receiver<PrefetchJob>>, state: &PrefetchState) {
    loop {
        if state.cancel.load(Ordering::Acquire) {
            break;
        }
        // Each `recv` is a single-message critical section; multiple
        // workers contend on the same mpsc receiver here.  This is
        // fine for the prefetch hot path — JPEG decode (~ms per
        // image) dominates by orders of magnitude.
        let job = {
            let guard = rx.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            guard.recv()
        };
        let Ok(job) = job else { break };

        // Run the decode under `catch_unwind` so a single bad image
        // can't poison the whole prefetcher.  `resolve_image`
        // already swallows most errors and returns `None`, but
        // panic-on-arithmetic-overflow inside a third-party decoder
        // would otherwise crash the prefetch thread.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            decode_one(&state.doc, state.doc_id, &state.cache, &job)
        }));
        match result {
            Ok(true) => {
                let _ = state.stats.decoded.fetch_add(1, Ordering::Relaxed);
            }
            Ok(false) => {
                let _ = state.stats.errors.fetch_add(1, Ordering::Relaxed);
            }
            Err(_) => {
                let _ = state.stats.errors.fetch_add(1, Ordering::Relaxed);
                log::warn!(
                    "prefetch: decode panicked for image {:?} on page {:?}",
                    String::from_utf8_lossy(&job.image_name),
                    job.page_id,
                );
            }
        }
    }
}

/// Decode one image `XObject` into the cache via the shared
/// [`resolve_image`] entry point with all GPU back-ends disabled.
/// Returns `true` on success (cache populated), `false` on
/// `resolve_image` returning `None` (logged by `resolve_image` itself).
fn decode_one(
    doc: &Document,
    doc_id: DocId,
    cache: &Arc<DeviceImageCache>,
    job: &PrefetchJob,
) -> bool {
    // Use the leaf page dict (not the inheritance-aware
    // `get_page_resource_dict(.., b"XObject")` we used during
    // discovery): `resolve_image` walks `/Resources/XObject` from
    // its `page_dict` argument and won't follow page-tree
    // inheritance.  Inherited-resource cases will resolve `None`
    // here and the renderer decodes on first touch.
    let Ok(page_dict) = doc.get_dictionary(job.page_id) else {
        return false;
    };
    let result = resolve_image(
        doc,
        &page_dict,
        &job.image_name,
        #[cfg(feature = "nvjpeg")]
        None,
        #[cfg(feature = "vaapi")]
        None,
        #[cfg(feature = "nvjpeg2k")]
        None,
        #[cfg(feature = "gpu-icc")]
        None,
        #[cfg(feature = "gpu-icc")]
        None,
        Some(cache),
        Some(doc_id),
    );
    result.is_some()
}

/// Cheap pre-filter: does `stream_id` resolve to a stream with
/// `/Subtype Image` and `/Filter DCTDecode`?  Avoids spinning up a
/// worker for non-image references or non-DCT images.
///
/// Filter normalisation (single-name vs single-element array vs
/// chained filters vs empty array) is delegated to the shared
/// [`filter_name`] helper used by the image decode path so the
/// prefetcher and the renderer agree on what "is JPEG" means.
fn is_dct_image_stream(doc: &Document, stream_id: ObjectId) -> bool {
    let Ok(obj_arc) = doc.get_object(stream_id) else {
        return false;
    };
    let Some(stream) = obj_arc.as_stream() else {
        return false;
    };
    if !matches!(stream.dict.get(b"Subtype"), Some(Object::Name(n)) if n == b"Image") {
        return false;
    }
    let Some(filter_obj) = stream.dict.get(b"Filter") else {
        return false;
    };
    matches!(
        ImageFilter::from_filter_str(filter_name(filter_obj).as_deref()),
        ImageFilter::Dct
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal_pdf_no_images() -> Vec<u8> {
        b"%PDF-1.4\n\
1 0 obj\n<</Type /Catalog /Pages 2 0 R>>\nendobj\n\
2 0 obj\n<</Type /Pages /Kids [3 0 R] /Count 1>>\nendobj\n\
3 0 obj\n<</Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]>>\nendobj\n\
xref\n0 4\n\
0000000000 65535 f\r\n\
0000000009 00000 n\r\n\
0000000056 00000 n\r\n\
0000000111 00000 n\r\n\
trailer\n<</Size 4 /Root 1 0 R>>\n\
startxref\n180\n%%EOF"
            .to_vec()
    }

    #[test]
    fn prefetch_config_default_is_two_workers_4k_cap() {
        let c = PrefetchConfig::default();
        assert_eq!(c.workers, 2);
        assert_eq!(c.max_images, 4096);
    }

    #[test]
    fn clamp_workers_caps_high_and_lifts_zero() {
        // Pin the bound so an accidental bump shows up in review.
        assert_eq!(MAX_PREFETCH_WORKERS, 16);
        // Real call into the production helper — a regression that
        // drops the clamp in `spawn_prefetch` would still be caught
        // here so long as both this test and the helper aren't
        // deleted together.
        assert_eq!(clamp_workers(0), 1);
        assert_eq!(clamp_workers(1), 1);
        assert_eq!(clamp_workers(8), 8);
        assert_eq!(clamp_workers(MAX_PREFETCH_WORKERS), MAX_PREFETCH_WORKERS);
        assert_eq!(
            clamp_workers(MAX_PREFETCH_WORKERS + 1),
            MAX_PREFETCH_WORKERS
        );
        assert_eq!(clamp_workers(10_000), MAX_PREFETCH_WORKERS);
        assert_eq!(clamp_workers(usize::MAX), MAX_PREFETCH_WORKERS);
    }

    #[test]
    fn is_dct_image_stream_rejects_non_streams_and_missing_objects() {
        let doc = Document::from_bytes_owned(minimal_pdf_no_images()).expect("minimal PDF");
        // Object 1 is the catalog dict, not a stream.
        assert!(!is_dct_image_stream(&doc, (1, 0)));
        // Object 99 does not exist — must return false, not panic.
        assert!(!is_dct_image_stream(&doc, (99, 0)));
    }
}
