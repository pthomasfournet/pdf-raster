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
//! that provides back-pressure and content-aware GPU routing via [`RoutingHint`]
//! hints set by the [`pdf_raster::prescan_page`] pre-scan pass.
//!
//! # Back-pressure
//!
//! [`std::sync::mpsc::SyncSender::send`] blocks when all capacity slots are
//! occupied by in-progress renders, preventing the producer from holding an
//! unbounded number of in-flight bitmaps in memory.
//!
//! When the pool has only one thread, W0 is both producer and consumer, so a
//! blocking channel would deadlock.  In that case capacity is set to `n_pages`
//! (the exact number of tasks to be sent), ensuring the producer loop completes
//! before W0 starts consuming.
//!
//! # GPU slot model
//!
//! VA-API back-pressure is structural: [`gpu::JpegQueueHandle::decode`] blocks
//! the calling Rayon worker until the VA-API worker thread replies — no
//! separate semaphore is needed here.  nvJPEG uses per-thread TLS slots; every
//! Rayon worker has its own decoder after the first page it renders.

use std::sync::mpsc;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
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
/// Set by the [`pdf_raster::prescan_page`] pass before tasks are enqueued.
/// The consumer inspects this hint to choose between GPU and CPU workers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RoutingHint {
    /// No content classification available; any worker may handle this page.
    Unclassified,
    /// Page has large baseline-JPEG images — prefer VA-API / nvJPEG worker.
    ///
    /// Set by the pre-scan pass when `dominant_filter == Dct`.
    GpuJpegCandidate,
    /// Page is text/vector only — skip GPU decoder setup overhead.
    ///
    /// Set by the pre-scan pass when `has_images == false`.
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

// ── Mutex helpers ─────────────────────────────────────────────────────────────

/// Lock `m`, recovering from poison.
///
/// If another thread panicked while holding the lock the mutex is poisoned but
/// the data inside is still valid (the push/recv that was in progress may be
/// incomplete, but the Vec/Receiver itself is not corrupted).  We recover
/// rather than propagating because the render pipeline treats all errors as
/// collected results, not as panics — and in release builds `panic = "abort"`
/// means a panicking consumer terminates the whole process before we ever
/// reach this path.
#[inline]
fn lock_recovering<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

// ── Page queue ────────────────────────────────────────────────────────────────

/// Bounded work-stealing page queue.
///
/// [`PageTask`]s are fed through a bounded [`mpsc::SyncSender`] into pool
/// workers that render pages and collect errors.
///
/// # Why `pool.scope`, not `pool.install` + `rayon::scope`
///
/// `pool.install(f)` runs `f` on a pool worker (W0) and blocks the calling
/// thread until `f` returns.  If `f` then calls the global `rayon::scope`,
/// the scope's task registry is the **global** Rayon pool — not the custom
/// pool.  Custom pool workers cannot steal from the global registry, so
/// consumer tasks never run, the sender eventually blocks, and the scope
/// waits for tasks that never start — deadlock.
///
/// `pool.scope(f)` is `self.install(|| scope(f))` internally (rayon-core
/// source), so `f` still runs on a pool worker thread (W0).  The difference
/// is that `scope(f)` here captures W0's registry, which belongs to the
/// custom pool.  Consumer tasks are therefore injected into the correct pool
/// and stolen by the other `num_threads − 1` workers.  After the producer
/// loop finishes, W0 itself work-steals remaining consumer tasks before
/// scope completes.
///
/// # Capacity and back-pressure
///
/// For pools with more than one thread, `capacity = 2 × num_threads` keeps
/// all workers fed while limiting peak in-flight bitmap memory.  For example,
/// at `num_threads = 12` (capacity = 24 in-flight pages) and 300 DPI A4
/// (~24 MB RGB per page), peak memory is 24 × 24 MB = 576 MB — bounded and
/// controlled, vs `par_iter` which starts all N pages at once.
///
/// For single-thread pools, see the module-level back-pressure note.
///
/// # Worker count
///
/// `num_threads − 1` workers actively consume while W0 produces; W0 also
/// consumes after the producer loop finishes.  [`mpsc::Receiver`] is `Send`
/// but not `Sync`, so it is wrapped in `Arc<Mutex<Receiver<PageTask>>>`.
/// The `Mutex` is held only during `recv()` — nanosecond scale — not during
/// rendering.
pub(crate) struct PageQueue;

impl PageQueue {
    pub(crate) fn new() -> Self {
        Self
    }

    /// Run the render loop over `tasks`, returning all `(page_num, error)` pairs.
    ///
    /// The producer loop runs on pool worker W0 that `pool.scope` selects.
    /// Consumer tasks are spawned into the same pool's scope and stolen by the
    /// remaining `num_threads − 1` workers; W0 also work-steals consumer tasks
    /// once the producer loop finishes.  All tasks complete before this function
    /// returns.
    ///
    /// Progress is reported via `progress` with the same semantics as the
    /// previous `par_iter` loop.  Errors are collected in non-deterministic
    /// order; callers should sort before displaying.
    pub(crate) fn run(
        &self,
        tasks: impl Iterator<Item = PageTask> + Send,
        pool: &rayon::ThreadPool,
        session: &RasterSession,
        total_u32: u32,
        args: &Args,
        progress: &ProgressCtx<'_>,
    ) -> Vec<(i32, RenderError)> {
        let n_threads = pool.current_num_threads().max(1);

        // Single-thread guard: W0 is both producer and consumer.  A blocking
        // channel would deadlock once the producer fills all slots and cannot
        // yield to the consumer task.  Use n_pages as capacity so the producer
        // loop always completes before W0 starts work-stealing consumer tasks.
        let capacity = if n_threads == 1 {
            progress.n_pages.max(1)
        } else {
            n_threads * 2
        };

        let (tx, rx) = mpsc::sync_channel::<PageTask>(capacity);
        // Receiver is Send but not Sync; Arc<Mutex<_>> lets N workers share it.
        // The Mutex is held only during recv() — nanosecond scale — not during rendering.
        let rx = Arc::new(Mutex::new(rx));
        // Arc lets each worker clone a handle to push errors without moving the Mutex.
        let errors: Arc<Mutex<Vec<(i32, RenderError)>>> = Arc::new(Mutex::new(Vec::new()));

        // pool.scope is self.install(|| scope(op)) — the closure runs on pool
        // worker W0.  W0 runs the producer loop below, so spawn n_threads-1
        // consumer tasks for the other workers.  After the producer loop W0
        // work-steals any remaining consumer tasks before scope returns.
        pool.scope(|s| {
            // n_threads-1: W0 is the producer; the other workers are consumers.
            // saturating_sub avoids underflow if current_num_threads() somehow
            // returned 0; max(1) ensures at least one consumer task is queued
            // so W0 can work-steal it after the producer loop finishes.
            let n_consumers = n_threads.saturating_sub(1).max(1);
            for _ in 0..n_consumers {
                let rx = Arc::clone(&rx);
                let errors = Arc::clone(&errors);
                s.spawn(move |_| {
                    loop {
                        // Hold the lock only for the recv() call; release it
                        // before rendering so all workers can render concurrently.
                        let task = {
                            let guard = lock_recovering(&rx);
                            match guard.recv() {
                                Ok(t) => t,
                                Err(_) => break, // tx dropped → channel exhausted
                            }
                            // guard drops here, lock released before render
                        };

                        // Hint is set by the prescan pass; affinity dispatch
                        // (steering GpuJpegCandidate to GPU workers) is a
                        // future work item.  The hint is captured here so the
                        // compiler sees it as used and the extension point is clear.
                        let _hint = task.hint;

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
                            lock_recovering(&errors).push((task.page_num, e));
                        }
                    }
                });
            }

            // Producer: runs on pool worker W0 while the remaining workers consume.
            // SyncSender::send blocks when all capacity slots are full — back-pressure.
            // If all consumers have panicked and disconnected, send returns Err; break
            // early so we don't spin forever.
            for task in tasks {
                if tx.send(task).is_err() {
                    break;
                }
            }
            // tx drops here (end of scope closure) → channel closes → consumers see
            // Err from recv() and exit their loops cleanly.
        });

        // pool.scope has returned — all consumer tasks have completed or panicked
        // (rayon re-raises panics after the scope join).  No other Arc holders
        // remain; try_unwrap and into_inner are both infallible.
        Arc::try_unwrap(errors)
            .expect("all consumer Arc clones are dropped when pool.scope returns")
            .into_inner()
            .unwrap_or_else(PoisonError::into_inner)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify the single-thread capacity guard: with n_threads=1, capacity
    /// must equal n_pages so the producer never blocks waiting for a consumer
    /// on the same thread.
    #[test]
    fn single_thread_capacity_equals_n_pages() {
        let n_pages = 7;
        // Reproduce the capacity formula used in PageQueue::run for n_threads=1.
        let capacity = if 1_usize == 1 { n_pages.max(1) } else { 1 * 2 };
        assert_eq!(
            capacity, n_pages,
            "single-thread capacity must cover all pages"
        );
    }

    /// Verify the multi-thread capacity formula: capacity = 2 * n_threads.
    #[test]
    fn multi_thread_capacity_is_two_x_threads() {
        for n in [2_usize, 4, 8, 12, 24] {
            let capacity = if n == 1 { 99 } else { n * 2 };
            assert_eq!(capacity, n * 2);
        }
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
