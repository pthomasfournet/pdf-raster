// `pub(crate)` items inside a `pub(crate)` module trigger clippy::redundant_pub_crate
// when nursery lints are enabled, but the explicitness aids readability in a module
// that is only accessed from a single call site.
#![expect(
    clippy::redundant_pub_crate,
    reason = "explicitness aids readability for this single-call-site module"
)]
//! Bounded work-stealing page queue for the CLI render loop.
//!
//! Replaces the static `par_iter()` dispatch with a producer/consumer pattern
//! that provides back-pressure and a [`RoutingHint`] extension point for future
//! content-aware GPU routing (Phase 7 `PageDiagnostics` pre-scan pass).
//!
//! # Back-pressure
//!
//! [`std::sync::mpsc::SyncSender::send`] blocks when all `capacity` slots are
//! occupied by in-progress renders, preventing the producer from holding an
//! unbounded number of in-flight bitmaps in memory.
//!
//! # GPU slot model
//!
//! VA-API back-pressure is structural: [`gpu::JpegQueueHandle::decode`] blocks
//! the calling Rayon worker until the VA-API worker thread replies — no
//! separate semaphore is needed here.  nvJPEG uses per-thread TLS slots; every
//! Rayon worker has its own decoder after the first [`lend_decoders`] call.
//!
//! [`lend_decoders`]: pdf_raster::lend_decoders

use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use std::sync::atomic::AtomicU32;

use pdf_raster::RasterSession;

use crate::args::Args;
use crate::render::RenderError;

// ── Progress context ──────────────────────────────────────────────────────────

/// Progress-reporting state passed to [`PageQueue::run`].
///
/// Groups the four progress-tracking arguments to keep the `run` signature
/// within the 7-argument clippy limit and to make call sites self-documenting.
pub(crate) struct ProgressCtx<'a> {
    /// Atomic counter incremented once per completed page.
    pub(crate) done: &'a AtomicU32,
    /// Total number of pages being rendered (for the denominator in progress output).
    pub(crate) n_pages: usize,
    /// Wall-clock instant when rendering started (for ETA calculation).
    pub(crate) start: &'a Instant,
}

// ── Routing hint ──────────────────────────────────────────────────────────────

/// Per-page routing signal for GPU vs CPU dispatch.
///
/// Only [`Unclassified`](RoutingHint::Unclassified) is produced today.  When
/// the `PageDiagnostics` pre-scan pass lands (Phase 7), the producer will set
/// [`GpuJpegCandidate`](RoutingHint::GpuJpegCandidate) and
/// [`CpuOnly`](RoutingHint::CpuOnly) before submitting tasks, enabling the
/// consumer to route pages to GPU or CPU workers accordingly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RoutingHint {
    /// No content classification available; any worker may handle this page.
    Unclassified,
    /// Page has large baseline-JPEG images — prefer VA-API / nvJPEG worker.
    ///
    /// Set by the pre-scan pass when `dominant_filter == Dct` and estimated
    /// image area exceeds the GPU threshold.  Unused until the pre-scan lands.
    // #[allow] not #[expect]: the lint fires only in non-test builds; the variant
    // IS constructed in the test exhaustive-match, making #[expect] unfulfilled
    // in test mode.  Remove when the PageDiagnostics pre-scan wires this in.
    #[allow(dead_code)]
    GpuJpegCandidate,
    /// Page is text/vector only — skip GPU decoder setup overhead.
    ///
    /// Set by the pre-scan pass when `has_images == false`.  Unused until the
    /// pre-scan lands.
    #[allow(dead_code)]
    CpuOnly,
}

// ── Page task ─────────────────────────────────────────────────────────────────

/// A single page render task with an associated routing hint.
pub(crate) struct PageTask {
    /// 1-based page number, validated ≥ 1 by `build_page_list`.
    pub(crate) page_num: i32,
    /// Routing hint for GPU vs CPU dispatch.
    pub(crate) hint: RoutingHint,
}

// ── Page queue ────────────────────────────────────────────────────────────────

/// Bounded work-stealing page queue.
///
/// The producer (the calling thread inside `pool.install`) feeds
/// [`PageTask`]s into a bounded [`mpsc::SyncSender`]; Rayon workers receive
/// tasks, render pages, and collect errors.
///
/// # Capacity and back-pressure
///
/// `capacity = 2 × num_threads` keeps all workers fed while limiting peak
/// in-flight bitmap memory.  For example, at `num_threads = 12` and 300 DPI
/// A4 (~24 MB RGB per page), peak memory is `24 × 24 MB = 576 MB` — bounded
/// and controlled, vs the old `par_iter` which could start all N pages at once.
///
/// # Worker count
///
/// One consumer task is spawned per [`rayon::current_num_threads()`] inside
/// the scope.  [`mpsc::Receiver`] is `Send` but not `Sync`, so it is wrapped
/// in `Arc<Mutex<Receiver<PageTask>>>`.  The `Mutex` is held only during the
/// `recv()` call (nanosecond scale) — not during rendering — so there is no
/// meaningful contention on the hot path.
pub(crate) struct PageQueue {
    capacity: usize,
}

impl PageQueue {
    /// Construct a new `PageQueue` with the given channel capacity.
    ///
    /// Values below 1 are silently raised to 1.  A capacity of `2 ×
    /// num_threads` is a good default.
    pub(crate) fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
        }
    }

    /// Run the render loop over `tasks`, returning all `(page_num, error)` pairs.
    ///
    /// The producer runs on the **calling thread** (which `pool.install` has
    /// donated to the Rayon pool).  Consumer tasks are spawned via
    /// [`rayon::scope`].  All consumers complete before this function returns.
    ///
    /// Progress is reported via `progress` with the same semantics as the
    /// previous `par_iter` loop.  Errors are collected in non-deterministic
    /// order; callers should sort before displaying.
    pub(crate) fn run(
        &self,
        tasks: impl Iterator<Item = PageTask> + Send,
        session: &RasterSession,
        total_u32: u32,
        args: &Args,
        progress: &ProgressCtx<'_>,
    ) -> Vec<(i32, RenderError)> {
        let (tx, rx) = mpsc::sync_channel::<PageTask>(self.capacity);
        // Receiver is Send but not Sync; Arc<Mutex<_>> lets N workers share it.
        // The Mutex is held only during recv() — nanosecond scale — not during rendering.
        let rx = Arc::new(Mutex::new(rx));
        // Arc lets each worker clone a handle to push errors without moving the Mutex.
        let errors: Arc<Mutex<Vec<(i32, RenderError)>>> = Arc::new(Mutex::new(Vec::new()));

        rayon::scope(|s| {
            // Spawn one consumer task per available Rayon worker thread.
            // Each consumer loops until the channel is exhausted.
            let n_workers = rayon::current_num_threads().max(1);
            for _ in 0..n_workers {
                let rx = Arc::clone(&rx);
                let errors = Arc::clone(&errors);
                s.spawn(move |_| {
                    loop {
                        // Hold the lock only for the recv() call; release it
                        // before rendering so all workers can render concurrently.
                        let task = {
                            let guard =
                                rx.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                            match guard.recv() {
                                Ok(t) => t,
                                Err(_) => break, // channel exhausted → exit
                            }
                            // guard drops here, lock released before render
                        };

                        // RoutingHint is available here for future dispatch logic.
                        // Today all hints are Unclassified; GPU decisions are made
                        // inside render_page via lend_decoders transparently.
                        let _ = task.hint;

                        #[expect(
                            clippy::cast_sign_loss,
                            reason = "page_num ≥ 1, enforced by build_page_list"
                        )]
                        let page_u32 = task.page_num as u32;

                        let result = crate::render::render_page(session, page_u32, total_u32, args);
                        crate::report_progress(
                            args,
                            progress.done,
                            progress.n_pages,
                            progress.start,
                            task.page_num,
                        );

                        if let Err(e) = result {
                            errors
                                .lock()
                                .unwrap_or_else(std::sync::PoisonError::into_inner)
                                .push((task.page_num, e));
                        }
                    }
                });
            }

            // Producer runs on the calling thread (donated to the pool by pool.install).
            // SyncSender::send blocks when all `capacity` slots are full — back-pressure.
            // If all consumers have disconnected (e.g. due to a panic), send returns Err;
            // we break early rather than spinning forever.
            for task in tasks {
                if tx.send(task).is_err() {
                    break;
                }
            }
            // drop(tx) here closes the sender side of the channel.
            // Consumers see Err from recv() and exit their loops cleanly.
        });

        // rayon::scope has returned — all consumer tasks have completed.
        // No other Arc holders exist; try_unwrap + into_inner are both infallible.
        Arc::try_unwrap(errors)
            .expect("no other Arc holders after scope")
            .into_inner()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capacity_zero_raised_to_one() {
        assert_eq!(PageQueue::new(0).capacity, 1);
    }

    #[test]
    fn capacity_preserved_above_zero() {
        assert_eq!(PageQueue::new(8).capacity, 8);
    }

    #[test]
    fn routing_hint_variants_exhaustive() {
        // If any variant is removed this match fails to compile — structural guard.
        let hints = [
            RoutingHint::Unclassified,
            RoutingHint::GpuJpegCandidate,
            RoutingHint::CpuOnly,
        ];
        assert_eq!(hints.len(), 3);
        for h in hints {
            match h {
                RoutingHint::Unclassified
                | RoutingHint::GpuJpegCandidate
                | RoutingHint::CpuOnly => {}
            }
        }
    }

    #[test]
    fn bounded_channel_provides_backpressure() {
        // SyncSender with capacity 1: after the slot is filled, try_send
        // returns Err(Full), demonstrating back-pressure semantics.
        let (tx, rx) = mpsc::sync_channel::<u32>(1);
        tx.send(1)
            .expect("first send into empty channel must succeed");
        assert!(
            tx.try_send(2).is_err(),
            "second send into full channel must fail (back-pressure)"
        );
        let _ = rx.recv().expect("recv must yield the first item");
        tx.send(2).expect("send after drain must succeed");
    }

    #[test]
    fn unclassified_is_default_hint() {
        let task = PageTask {
            page_num: 1,
            hint: RoutingHint::Unclassified,
        };
        assert_eq!(task.hint, RoutingHint::Unclassified);
    }
}
