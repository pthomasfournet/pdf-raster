//! Generic JPEG decode queue: one worker thread owns the decoder, Rayon workers
//! send jobs and block on a oneshot reply.
//!
//! # Why this exists
//!
//! VA-API (Mesa) serialises all `vaBeginPicture`/`vaRenderPicture`/`vaEndPicture`
//! submissions through one hardware VCN engine queue at the driver level.  When N
//! Rayon workers each hold a `VapiJpegDecoder` and call it concurrently, they
//! contend on an internal Mesa mutex — no real parallelism results.
//!
//! `DecodeQueue<D>` routes all submissions through a single OS thread that owns
//! the decoder.  Rayon workers send a `DecodeJob` and block on a oneshot reply
//! channel.  The hardware engine stays fully pipelined; Mesa mutex contention is
//! eliminated.
//!
//! The same generic type works for any `D: GpuJpegDecoder + Send + 'static`, so
//! it can serve both VA-API and (if needed in future) nvJPEG.

#[cfg(any(feature = "nvjpeg", feature = "vaapi"))]
use std::sync::mpsc::{self, Receiver, SyncSender};
#[cfg(any(feature = "nvjpeg", feature = "vaapi"))]
use std::{marker::PhantomData, thread::JoinHandle};

#[cfg(any(feature = "nvjpeg", feature = "vaapi"))]
use crate::traits::{DecodedImage, GpuDecodeError, GpuJpegDecoder};

// ── Job ───────────────────────────────────────────────────────────────────────

/// A JPEG decode request sent to the worker thread.
///
/// `data` is `Arc<[u8]>` so the JPEG bytes can cross the thread boundary without
/// copying *and* without unsafe lifetime extension.  One `memcpy` of the JPEG
/// payload (typically 50–500 KB) is negligible compared to VCN decode time.
#[cfg(any(feature = "nvjpeg", feature = "vaapi"))]
struct DecodeJob {
    data: std::sync::Arc<[u8]>,
    width: u32,
    height: u32,
    reply: SyncSender<Result<DecodedImage, GpuDecodeError>>,
}

// ── DecodeQueue ───────────────────────────────────────────────────────────────

/// A JPEG decode queue backed by a single dedicated OS worker thread.
///
/// The worker thread owns `D` exclusively and processes jobs serially, which is
/// exactly what hardware decode engines (VA-API VCN, nvJPEG HARDWARE backend)
/// prefer.  Rayon worker threads obtain a [`JpegQueueHandle`] via
/// [`DecodeQueue::handle`] and submit jobs that way.
///
/// Dropping `DecodeQueue` closes the job channel; the worker exits its loop and
/// drops `D` on its own thread — satisfying thread-affinity requirements for both
/// CUDA contexts and VA-API contexts.  `Drop` blocks until the worker finishes.
///
/// # Ownership invariant
///
/// All [`JpegQueueHandle`]s obtained from this queue (via [`DecodeQueue::handle`])
/// **must be dropped before the `DecodeQueue` itself**.  Handles hold a clone of the
/// job-channel sender; if any handle outlives the queue, `Drop::join` will deadlock
/// because the channel remains open.  In production, `DecodeQueue` lives in
/// `RasterSession`, which always outlives the per-page `PageRenderer` instances that
/// hold handles — the invariant is naturally satisfied.
#[cfg(any(feature = "nvjpeg", feature = "vaapi"))]
pub struct DecodeQueue<D: GpuJpegDecoder> {
    sender: mpsc::Sender<DecodeJob>,
    worker: Option<JoinHandle<()>>,
    _phantom: PhantomData<fn() -> D>,
}

#[cfg(any(feature = "nvjpeg", feature = "vaapi"))]
impl<D: GpuJpegDecoder + Send + 'static> DecodeQueue<D> {
    /// Construct a new queue, spawning a worker thread that owns `decoder`.
    ///
    /// # Panics
    ///
    /// Panics if the OS refuses to spawn the thread (extremely unlikely).
    pub fn new(decoder: D, thread_name: &str) -> Self {
        let (tx, rx) = mpsc::channel::<DecodeJob>();
        let handle = std::thread::Builder::new()
            .name(thread_name.to_owned())
            .stack_size(2 * 1024 * 1024)
            .spawn(move || worker_loop(decoder, rx))
            .expect("gpu: failed to spawn decode worker thread");
        Self {
            sender: tx,
            worker: Some(handle),
            _phantom: PhantomData,
        }
    }

    /// Return a cheaply-cloneable handle that Rayon workers can use to submit
    /// decode jobs without knowing the concrete decoder type.
    #[must_use]
    pub fn handle(&self) -> JpegQueueHandle {
        JpegQueueHandle {
            sender: self.sender.clone(),
        }
    }
}

#[cfg(any(feature = "nvjpeg", feature = "vaapi"))]
impl<D: GpuJpegDecoder> Drop for DecodeQueue<D> {
    fn drop(&mut self) {
        // We must close the job channel *before* joining the worker — otherwise
        // the worker blocks forever in `rx.recv()`.  We cannot move out of
        // `self`, so we swap in a dummy sender; `_real_sender` drops at the end
        // of the inner block, closing the original channel.
        //
        // This works only if all `JpegQueueHandle` clones have already been
        // dropped (each one holds a `Sender` clone to the same channel).  See
        // the "Ownership invariant" section in the struct doc.
        {
            let (dummy_tx, _dummy_rx) = mpsc::channel::<DecodeJob>();
            let _real_sender = std::mem::replace(&mut self.sender, dummy_tx);
            // _real_sender drops here → all senders on original channel gone
            // (provided all JpegQueueHandles are already dropped) → worker exits
        }

        if self.worker.take().and_then(|h| h.join().err()).is_some() {
            log::warn!("gpu: JPEG decode worker thread panicked during shutdown");
        }
    }
}

// ── Worker loop ───────────────────────────────────────────────────────────────

#[cfg(any(feature = "nvjpeg", feature = "vaapi"))]
// rx must be owned (moved in from the thread closure); clippy incorrectly
// suggests passing by reference since Receiver::recv takes &self.
#[expect(
    clippy::needless_pass_by_value,
    reason = "rx is consumed by the thread"
)]
fn worker_loop<D: GpuJpegDecoder>(mut decoder: D, rx: Receiver<DecodeJob>) {
    while let Ok(job) = rx.recv() {
        let result = decoder.decode_jpeg(&job.data, job.width, job.height);
        // Ignore send error: the Rayon worker may have timed out or been
        // cancelled; there is nothing meaningful we can do here.
        let _ = job.reply.send(result);
        // job.data Arc drops here → JPEG bytes freed if no other holder.
    }
    // `decoder` drops here, on the worker thread — correct for thread-affine
    // VA-API VAContext / CUDA primary context.
}

// ── JpegQueueHandle ───────────────────────────────────────────────────────────

/// A cheaply-cloneable send handle to a [`DecodeQueue`].
///
/// Rayon workers hold one `JpegQueueHandle` per page render.  Sending a job
/// blocks the calling thread until the worker replies — this is intentional:
/// it provides back-pressure and avoids unbounded job accumulation.
///
/// Dropping all handles does **not** shut down the queue; the owning
/// `DecodeQueue` must be dropped for that.
#[cfg(any(feature = "nvjpeg", feature = "vaapi"))]
#[derive(Clone)]
pub struct JpegQueueHandle {
    sender: mpsc::Sender<DecodeJob>,
}

#[cfg(any(feature = "nvjpeg", feature = "vaapi"))]
impl JpegQueueHandle {
    /// Submit `data` for decode and block until the worker replies.
    ///
    /// Returns `None` when:
    /// - The worker thread has panicked (channel disconnected).
    /// - The decoder returns a [`GpuDecodeError`] (logged as a warning).
    ///
    /// In both cases the caller should fall through to the CPU decode path.
    ///
    /// # Data copy note
    ///
    /// `data` is wrapped in `Arc<[u8]>` so it can cross the thread boundary
    /// safely.  This is one `memcpy` of the JPEG payload.  For images below
    /// `GPU_JPEG_THRESHOLD_PX` the caller already skips the queue, so only
    /// large images (≥ 512×512 px, typically 50–500 KB) pay this cost — which
    /// is negligible relative to VCN hardware decode time (1–5 ms/frame).
    #[must_use]
    pub fn decode(
        &self,
        data: std::sync::Arc<[u8]>,
        width: u32,
        height: u32,
    ) -> Option<DecodedImage> {
        let (reply_tx, reply_rx) = mpsc::sync_channel(1);
        self.sender
            .send(DecodeJob {
                data,
                width,
                height,
                reply: reply_tx,
            })
            .ok()?; // None if worker is gone
        match reply_rx.recv().ok()? {
            Ok(img) => Some(img),
            Err(e) => {
                log::warn!("gpu: JPEG queue decode failed: {e}");
                None
            }
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[cfg(any(feature = "nvjpeg", feature = "vaapi"))]
mod tests {
    use std::sync::Arc;

    use super::DecodeQueue;
    use crate::traits::{DecodedImage, GpuDecodeError, GpuJpegDecoder};

    // ── Mock decoders ─────────────────────────────────────────────────────────

    /// Always returns a 1-component (gray) image of the requested dimensions.
    struct AlwaysGray;
    impl GpuJpegDecoder for AlwaysGray {
        fn decode_jpeg(
            &mut self,
            _data: &[u8],
            width: u32,
            height: u32,
        ) -> Result<DecodedImage, GpuDecodeError> {
            let n = (width as usize) * (height as usize);
            Ok(DecodedImage {
                data: vec![128u8; n],
                width,
                height,
                components: 1,
            })
        }
    }

    /// Always returns a `GpuDecodeError`.
    struct AlwaysFail;
    impl GpuJpegDecoder for AlwaysFail {
        fn decode_jpeg(
            &mut self,
            _data: &[u8],
            _width: u32,
            _height: u32,
        ) -> Result<DecodedImage, GpuDecodeError> {
            Err(GpuDecodeError::new(std::io::Error::other("mock failure")))
        }
    }

    // ── Tests ─────────────────────────────────────────────────────────────────

    #[test]
    fn queue_routes_job_and_returns_decoded_image() {
        let q = DecodeQueue::new(AlwaysGray, "test-always-gray");
        let handle = q.handle();
        let data: Arc<[u8]> = Arc::from([0u8; 16].as_slice());
        let img = handle.decode(data, 4, 4).expect("should succeed");
        assert_eq!(img.width, 4);
        assert_eq!(img.height, 4);
        assert_eq!(img.components, 1);
        assert_eq!(img.data.len(), 16);
    }

    #[test]
    fn queue_falls_back_on_decoder_error() {
        let q = DecodeQueue::new(AlwaysFail, "test-always-fail");
        let handle = q.handle();
        let data: Arc<[u8]> = Arc::from([0u8; 4].as_slice());
        let result = handle.decode(data, 2, 2);
        assert!(result.is_none(), "AlwaysFail should produce None");
    }

    #[test]
    fn queue_shuts_down_cleanly_on_drop() {
        // Model the production invariant: DecodeQueue outlives all handles.
        // Handles must be dropped *before* the queue to avoid deadlock in Drop::join.
        let q = DecodeQueue::new(AlwaysGray, "test-shutdown");
        {
            let handle = q.handle();
            let data: Arc<[u8]> = Arc::from([0u8; 1].as_slice());
            let result = handle.decode(data, 1, 1);
            assert!(result.is_some(), "in-scope handle should work");
            // handle drops here — all senders except queue's own are gone
        }
        // Now drop the queue — must not deadlock or panic.
        drop(q);
    }

    #[test]
    fn multiple_concurrent_senders_serialize_through_worker() {
        use std::sync::Arc as StdArc;
        use std::sync::atomic::{AtomicU32, Ordering};

        let q = StdArc::new(DecodeQueue::new(AlwaysGray, "test-concurrent"));
        let counter = StdArc::new(AtomicU32::new(0));
        let n_threads = 8u32;

        let handles: Vec<_> = (0..n_threads)
            .map(|_| {
                let handle = q.handle();
                let counter = StdArc::clone(&counter);
                std::thread::spawn(move || {
                    let data: Arc<[u8]> = Arc::from([0u8; 9].as_slice());
                    let img = handle.decode(data, 3, 3).expect("should succeed");
                    assert_eq!(img.components, 1);
                    let _ = counter.fetch_add(1, Ordering::Relaxed);
                })
            })
            .collect();

        for h in handles {
            h.join().expect("thread should not panic");
        }
        // All handles have been dropped (moved into threads that are now joined).
        // Drop the queue explicitly to verify shutdown is clean, matching the
        // production invariant: handles gone before queue drops.
        drop(q);
        assert_eq!(counter.load(Ordering::Relaxed), n_threads);
    }

    #[test]
    fn handle_clone_is_independent() {
        let q = DecodeQueue::new(AlwaysGray, "test-clone");
        let h1 = q.handle();
        let h2 = h1.clone();
        // Drop h1 — h2 should still work because it cloned the Sender.
        drop(h1);
        let data: Arc<[u8]> = Arc::from([0u8; 4].as_slice());
        let img = h2.decode(data, 2, 2).expect("cloned handle should work");
        assert_eq!(img.width, 2);
    }
}
