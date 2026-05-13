//! Smoke test: `VulkanBackend` implements `GpuBackend` on whatever ICD
//! the system has.  Skipped if the `vulkan` feature isn't built.
//!
//! Run with:
//!   `cargo test -p gpu --features vulkan --test vulkan_backend_smoke`
//!
//! This test only requires *any* working Vulkan ICD — Mesa lavapipe (the
//! CPU software ICD) counts.  CI without a discrete GPU can run it via
//! `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json`.
//!
//! ## Backend lifecycle
//!
//! Most tests use [`shared_backend`], which returns the per-process
//! `Arc<VulkanBackend>` (lazy-initialised on first call) **paired with a
//! mutex guard** that serialises against any other shared-backend test.
//! The serialisation is required because:
//!   - `PageRecorder`'s state machine forbids two `begin_page` calls
//!     without an intervening `submit_page`; the recorder mutex would
//!     reject the racing call with `InvalidRecorderState` instead of
//!     interleaving the work, so parallel tests would flake.
//!   - The descriptor-pool exhaustion test allocates the entire per-page
//!     budget; a parallel `begin_page` would see a full pool and fail.
//!
//! Sharing the backend matches the *production* lifecycle (the renderer
//! holds one `VulkanBackend` for the whole process), so subtle leaks or
//! stale-state bugs that per-test reset would mask now surface here.
//!
//! [`vulkan_backend_initialises`] is the single test that keeps its own
//! `VulkanBackend::new()` — its job is to verify cold init.

#![cfg(feature = "vulkan")]

use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

use rasterrocket_gpu::backend::vulkan::VulkanBackend;
use rasterrocket_gpu::backend::{BackendError, GpuBackend};

/// Per-process `VulkanBackend`, lazy-initialised on first access.
///
/// Created via `OnceLock` so cold init (`vkCreateInstance` /
/// `vkCreateDevice` / pipeline-cache build) runs once for the whole test
/// binary instead of once per test.
static SHARED: OnceLock<Arc<VulkanBackend>> = OnceLock::new();

/// Mutex guarding shared-backend tests against concurrent execution.
///
/// Cargo's test runner is parallel by default; this lock makes
/// shared-backend tests sequential among themselves while still letting
/// them run in parallel with any test (e.g. [`vulkan_backend_initialises`])
/// that doesn't take the lock.
static SERIAL: Mutex<()> = Mutex::new(());

/// Acquire the shared `VulkanBackend` together with a serialisation
/// guard. Hold the returned tuple's second value (`_serial`) until the
/// test ends — dropping it releases the next shared-backend test.
fn shared_backend() -> (Arc<VulkanBackend>, MutexGuard<'static, ()>) {
    // Lock first, *then* lazily initialise: the OnceLock initialiser may
    // perform Vulkan FFI under the hood, and we want the same
    // serialisation discipline to cover initialisation as covers normal
    // use. Mutex poisoning would only happen if a previous test panicked
    // mid-critical-section; rethrow by panicking — the test infra will
    // report it on the next test that runs.
    let serial = SERIAL.lock().expect("SERIAL poisoned by a prior panic");
    let backend = SHARED
        .get_or_init(|| Arc::new(VulkanBackend::new().expect("VulkanBackend::new")))
        .clone();
    (backend, serial)
}

#[test]
fn vulkan_backend_initialises() {
    // Intentionally NOT shared_backend(): this test verifies cold init,
    // so it must mint its own VulkanBackend each run.
    let backend =
        VulkanBackend::new().expect("VulkanBackend::new failed — is a Vulkan ICD available?");
    let budget = backend
        .detect_vram_budget()
        .expect("detect_vram_budget failed — driver query returned an error");
    // total_bytes can legitimately be 0 on a software ICD that exposes
    // no DEVICE_LOCAL heap (lavapipe falls back to a HOST_VISIBLE heap).
    // Just check the invariant holds.
    assert!(
        budget.usable_bytes <= budget.total_bytes,
        "VramBudget invariant violated: usable {} > total {}",
        budget.usable_bytes,
        budget.total_bytes
    );
}

#[test]
fn vulkan_backend_alloc_free_round_trip() {
    let (backend, _serial) = shared_backend();
    let buf = backend.alloc_device(4096).expect("alloc_device(4096)");
    assert_eq!(buf.size(), 4096);
    backend.free_device(buf);
}

#[test]
fn vulkan_backend_alloc_zero_size_rejected() {
    let (backend, _serial) = shared_backend();
    let err = backend
        .alloc_device(0)
        .expect_err("alloc_device(0) must be rejected");
    assert!(
        matches!(
            err,
            BackendError::ZeroSizeAlloc {
                what: "alloc_device"
            }
        ),
        "expected ZeroSizeAlloc {{ what: \"alloc_device\" }}, got: {err:?}",
    );
}

/// Allocate a zeroed device buffer, download into a sentinel-filled
/// host vec, assert every byte is zero, free.  `tag` is included in
/// failure messages so concurrent / parameterised callers can
/// pinpoint which invocation broke.
fn assert_zeroed_round_trip(backend: &VulkanBackend, size: usize, tag: &str) {
    let buf = backend
        .alloc_device_zeroed(size)
        .unwrap_or_else(|e| panic!("{tag}: alloc_device_zeroed({size}) failed: {e}"));
    let mut readback = vec![0xAAu8; size];
    backend
        .download_sync(&buf, &mut readback)
        .unwrap_or_else(|e| panic!("{tag}: download_sync failed: {e}"));
    if let Some(bad) = readback.iter().position(|&b| b != 0) {
        panic!(
            "{tag}: alloc_device_zeroed({size}) leaked non-zero byte 0x{:02x} at index {bad}",
            readback[bad]
        );
    }
    backend.free_device(buf);
}

#[test]
fn vulkan_backend_alloc_device_zeroed_returns_zero_bytes() {
    let (backend, _serial) = shared_backend();
    assert_zeroed_round_trip(&backend, 4096, "single");
}

#[test]
fn vulkan_backend_alloc_device_zeroed_concurrent() {
    // Regression: VkCommandPool + VkQueue both require external sync
    // (Vulkan "Threading Behavior" table).  Without the locks on
    // DeviceCtx + TransferContext, concurrent callers race in
    // run_one_shot — either driver crash or non-zero readbacks.
    use std::thread;

    const THREADS: usize = 4;
    const ITERS_PER_THREAD: usize = 8;

    let (backend, _serial) = shared_backend();
    let mut handles = Vec::with_capacity(THREADS);
    for thread_idx in 0..THREADS {
        let backend = Arc::clone(&backend);
        handles.push(thread::spawn(move || {
            for iter in 0..ITERS_PER_THREAD {
                assert_zeroed_round_trip(
                    &backend,
                    4096,
                    &format!("thread {thread_idx} iter {iter}"),
                );
            }
        }));
    }
    for h in handles {
        h.join().expect("worker thread panicked");
    }
}

#[test]
fn vulkan_backend_descriptor_pool_exhausts_at_max_sets() {
    // Recorder caps descriptor sets per page at MAX_DESC_SETS_PER_PAGE (64).
    // Verify the boundary: 64 record_composite calls succeed, the 65th
    // returns Err with DescriptorPoolExhausted.  Pre-existing guard
    // had zero coverage; a refactor that reordered guard / allocate /
    // increment could silently regress.
    use rasterrocket_gpu::backend::params::CompositeParams;

    // Keep in sync with `MAX_DESC_SETS_PER_PAGE` in
    // crates/gpu/src/backend/vulkan/recorder.rs (private constant).
    // The let-else below destructures `BackendError::DescriptorPoolExhausted`
    // and asserts the production `max` field equals this — a production bump
    // without updating PER_PAGE_MAX fails loudly instead of silently
    // passing through the loop.
    const PER_PAGE_MAX: u32 = 64;
    const N_PIXELS: u32 = 4;
    const BUF_BYTES: usize = (N_PIXELS as usize) * rasterrocket_gpu::RGBA_BPP;

    let (backend, _serial) = shared_backend();
    let src = backend.alloc_device_zeroed(BUF_BYTES).expect("alloc src");
    let dst = backend.alloc_device_zeroed(BUF_BYTES).expect("alloc dst");

    backend.begin_page().expect("begin_page");
    for i in 0..PER_PAGE_MAX {
        backend
            .record_composite(CompositeParams {
                src: &src,
                dst: &dst,
                n_pixels: N_PIXELS,
            })
            .unwrap_or_else(|e| panic!("record_composite #{i} of {PER_PAGE_MAX} failed: {e}"));
    }
    let err = backend
        .record_composite(CompositeParams {
            src: &src,
            dst: &dst,
            n_pixels: N_PIXELS,
        })
        .expect_err("record_composite past the cap must be rejected");
    let BackendError::DescriptorPoolExhausted { allocated, max } = err else {
        panic!("expected DescriptorPoolExhausted, got: {err:?}");
    };
    assert_eq!(
        allocated, PER_PAGE_MAX,
        "exhaustion should report the cap as the count already in flight",
    );
    assert_eq!(
        max, PER_PAGE_MAX,
        "test PER_PAGE_MAX is out of sync with production MAX_DESC_SETS_PER_PAGE",
    );

    // Recorder state stayed Recording (the rejected call exited before
    // touching the cmd buffer); finish the page cleanly so VulkanBackend's
    // Drop doesn't see in-flight work on device_wait_idle.
    let fence = backend.submit_page().expect("submit_page");
    backend.wait_page(fence).expect("wait_page");

    backend.free_device(src);
    backend.free_device(dst);
}

#[test]
fn vulkan_backend_immediate_fence_round_trips() {
    // submit_transfer returns PageFence::immediate() (a structural None
    // sentinel); wait_transfer must return Ok without driving a Vulkan
    // FFI call.  Exercises the wait_timeline(None) short-circuit added
    // in the Option<NonZeroU64> rework.
    let (backend, _serial) = shared_backend();
    let fence = backend.submit_transfer().expect("submit_transfer");
    backend
        .wait_transfer(fence)
        .expect("wait_transfer on immediate fence should succeed");
}

#[test]
fn vulkan_backend_record_zero_buffer_zeros_dirty_alloc() {
    // alloc_device returns a buffer with undefined contents; upload a
    // sentinel pattern, then record_zero_buffer inside a page submission,
    // then verify the readback is all-zero.
    //
    // This is the production-shape path the audit asked for: zero-fill
    // folded into the per-page command buffer instead of riding its own
    // submit + vkQueueWaitIdle in alloc_device_zeroed.
    const SIZE: usize = 4096;
    let (backend, _serial) = shared_backend();
    let buf = backend.alloc_device(SIZE).expect("alloc_device");
    let sentinel = vec![0xCDu8; SIZE];
    backend
        .upload_sync(&buf, &sentinel)
        .expect("upload sentinel");

    backend.begin_page().expect("begin_page");
    backend
        .record_zero_buffer(&buf)
        .expect("record_zero_buffer");
    let fence = backend.submit_page().expect("submit_page");
    backend.wait_page(fence).expect("wait_page");

    let mut readback = vec![0xAAu8; SIZE];
    backend
        .download_sync(&buf, &mut readback)
        .expect("download");
    if let Some(bad) = readback.iter().position(|&b| b != 0) {
        panic!(
            "record_zero_buffer left non-zero byte 0x{:02x} at index {bad}",
            readback[bad]
        );
    }
    backend.free_device(buf);
}

#[test]
fn vulkan_backend_record_zero_buffer_rejects_outside_recording_state() {
    // record_zero_buffer is a state-machine op: legal only between
    // begin_page and submit_page.  Calling it from Idle must error,
    // not record into a stale command buffer.
    let (backend, _serial) = shared_backend();
    let buf = backend.alloc_device(64).expect("alloc_device");
    let err = backend
        .record_zero_buffer(&buf)
        .expect_err("record_zero_buffer outside Recording must be rejected");
    let msg = err.to_string();
    assert!(
        msg.contains("expected Recording"),
        "expected state-machine error, got: {msg}"
    );
    backend.free_device(buf);
}

#[test]
fn vulkan_backend_record_zero_buffer_rejects_unaligned_size() {
    // vkCmdFillBuffer requires 4-byte-aligned size; record_zero_buffer
    // surfaces this loudly via BackendError::UnalignedFill instead of
    // silently leaving 1–3 trailing bytes non-zero.
    let (backend, _serial) = shared_backend();
    let buf = backend.alloc_device(17).expect("alloc_device(17)");
    backend.begin_page().expect("begin_page");
    let err = backend
        .record_zero_buffer(&buf)
        .expect_err("record_zero_buffer on 17-byte buf must be rejected");
    assert!(
        matches!(
            err,
            BackendError::UnalignedFill {
                size: 17,
                required_alignment: 4,
            }
        ),
        "expected UnalignedFill{{17,4}}, got: {err:?}"
    );
    // Finish the page cleanly so VulkanBackend::Drop doesn't see
    // in-flight work — the recorder stays in Recording state after a
    // rejected call (neither the state nor the size check transitions
    // anything).  submit then wait drains the empty cmd buffer.
    let fence = backend.submit_page().expect("submit_page");
    backend.wait_page(fence).expect("wait_page");
    backend.free_device(buf);
}

#[test]
fn vulkan_backend_alloc_device_zeroed_rejects_unaligned_size() {
    // vkCmdFillBuffer requires 4-byte-aligned size; alloc_device_zeroed
    // must surface this loudly rather than silently downgrading.
    let (backend, _serial) = shared_backend();
    let err = backend
        .alloc_device_zeroed(17)
        .expect_err("alloc_device_zeroed(17) must be rejected (size not multiple of 4)");
    assert!(
        matches!(
            err,
            BackendError::UnalignedFill {
                size: 17,
                required_alignment: 4
            }
        ),
        "expected UnalignedFill {{ size: 17, required_alignment: 4 }}, got: {err:?}",
    );
}

#[test]
fn vulkan_backend_host_buffer_round_trip() {
    let (backend, _serial) = shared_backend();
    let mut buf = backend
        .alloc_host_pinned(256)
        .expect("alloc_host_pinned(256)");
    assert_eq!(buf.size(), 256);
    let slice = buf.as_mut_slice();
    for (i, b) in slice.iter_mut().enumerate() {
        *b = u8::try_from(i & 0xff).expect("masked to 0..=255");
    }
    let read_back = buf.as_slice();
    for (i, &b) in read_back.iter().enumerate() {
        assert_eq!(
            b,
            u8::try_from(i & 0xff).expect("masked to 0..=255"),
            "byte {i} mismatch"
        );
    }
    backend.free_host_pinned(buf);
}
