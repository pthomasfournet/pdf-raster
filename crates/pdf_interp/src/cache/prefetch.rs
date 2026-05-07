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

use crate::resources::resolve_image;

/// Tunables for [`spawn_prefetch`].
#[derive(Debug, Clone, Copy)]
pub struct PrefetchConfig {
    /// Number of background worker threads.  Default 2; bumping this
    /// only helps if the renderer hasn't started yet (i.e. the
    /// session was opened well in advance of the first render call).
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

impl PrefetchStats {
    fn snapshot(&self) -> PrefetchStatsSnapshot {
        PrefetchStatsSnapshot {
            discovered: self.discovered.load(Ordering::Relaxed),
            decoded: self.decoded.load(Ordering::Relaxed),
            already_cached: self.already_cached.load(Ordering::Relaxed),
            errors: self.errors.load(Ordering::Relaxed),
        }
    }
}

/// Plain-data snapshot of [`PrefetchStats`] — what
/// [`PrefetchHandle::wait`] returns once the prefetcher has drained.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct PrefetchStatsSnapshot {
    /// See [`PrefetchStats::discovered`].
    pub discovered: u64,
    /// See [`PrefetchStats::decoded`].
    pub decoded: u64,
    /// See [`PrefetchStats::already_cached`].
    pub already_cached: u64,
    /// See [`PrefetchStats::errors`].
    pub errors: u64,
}

/// Handle to a running prefetch job.  Drop the handle to cancel and
/// join the workers; or call [`Self::wait`] to block until every
/// queued image has been processed.
pub struct PrefetchHandle {
    cancel: Arc<AtomicBool>,
    workers: Mutex<Vec<thread::JoinHandle<()>>>,
    stats: Arc<PrefetchStats>,
}

impl PrefetchHandle {
    /// Block until every queued image has been processed and all
    /// workers have exited cleanly, then return the final counters.
    ///
    /// Idempotent: a second call returns the same stats and is a
    /// no-op (the worker handles have already been joined).
    pub fn wait(&self) -> PrefetchStatsSnapshot {
        // Drain handles out from under the lock so the rest of the
        // wait happens unlocked — `JoinHandle::join` blocks for the
        // worker's full runtime, which is too long to hold the
        // workers mutex.
        let drained: Vec<_> = {
            let mut guard = self
                .workers
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            guard.drain(..).collect()
        };
        for handle in drained {
            // Workers swallow panics by design (a bad image must not
            // poison the whole prefetch run).  `join` returning Err
            // here would mean a panic escaped — log and move on.
            if let Err(e) = handle.join() {
                log::warn!("prefetch worker panicked: {e:?}");
            }
        }
        self.stats.snapshot()
    }

    /// Read the current stats without joining.  Counters are
    /// monotonic so a partial read is still meaningful, e.g. when
    /// the renderer wants a progress bar.
    pub fn stats(&self) -> PrefetchStatsSnapshot {
        self.stats.snapshot()
    }
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
/// thread-safe, so handing `Arc`s to multiple workers is safe.  The
/// prefetcher does its own coarse dedup via a `Mutex<HashSet<ObjId>>`
/// so an image referenced from many pages is only decoded once.
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
        // Discovery is the only writer to `seen`, but the type
        // wraps it in a Mutex anyway so the call site stays a
        // one-line guard.
        seen: Arc::new(Mutex::new(HashSet::new())),
        stats: Arc::new(PrefetchStats::default()),
        cancel: Arc::new(AtomicBool::new(false)),
    });

    // Unbounded mpsc; the cap is `config.max_images`, enforced by
    // the discovery loop, so memory is bounded.
    let (tx, rx) = mpsc::channel::<PrefetchJob>();
    let rx = Arc::new(Mutex::new(rx));

    // Spawn workers first so they're ready when discovery begins.
    let workers = config.workers.max(1);
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
        stats: Arc::clone(&state.stats),
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
struct PrefetchState {
    cache: Arc<DeviceImageCache>,
    doc: Arc<Document>,
    doc_id: DocId,
    seen: Arc<Mutex<HashSet<ObjId>>>,
    stats: Arc<PrefetchStats>,
    cancel: Arc<AtomicBool>,
}

/// Walk every page's `/XObject` resource dict and emit a
/// [`PrefetchJob`] for each `/DCTDecode`-filtered image `XObject` we
/// haven't already seen.
fn discover_pages(state: &PrefetchState, max_images: usize, tx: &mpsc::Sender<PrefetchJob>) {
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
            // Already-cached images (alias hit) and already-queued
            // images both short-circuit here — the seen set is the
            // single dedup source for the whole run.
            if !state
                .seen
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .insert(obj_id)
            {
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
    // Filter may be a /Name or a [/Name ...] array; in either case
    // we accept if any element matches "DCTDecode".
    match filter_obj {
        Object::Name(n) => n == b"DCTDecode",
        Object::Array(arr) => arr
            .iter()
            .any(|o| matches!(o, Object::Name(n) if n == b"DCTDecode")),
        _ => false,
    }
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
    fn stats_default_starts_at_zero() {
        let snap = PrefetchStats::default().snapshot();
        assert_eq!(snap.discovered, 0);
        assert_eq!(snap.decoded, 0);
        assert_eq!(snap.already_cached, 0);
        assert_eq!(snap.errors, 0);
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
