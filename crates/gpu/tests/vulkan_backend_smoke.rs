//! Smoke test: `VulkanBackend` implements `GpuBackend` on whatever ICD
//! the system has.  Skipped if the `vulkan` feature isn't built.
//!
//! Run with:
//!   `cargo test -p gpu --features vulkan --test vulkan_backend_smoke`
//!
//! This test only requires *any* working Vulkan ICD — Mesa lavapipe (the
//! CPU software ICD) counts.  CI without a discrete GPU can run it via
//! `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json`.

#![cfg(feature = "vulkan")]

use gpu::backend::GpuBackend;
use gpu::backend::vulkan::VulkanBackend;

#[test]
fn vulkan_backend_initialises() {
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
    let backend = VulkanBackend::new().expect("VulkanBackend::new");
    let buf = backend.alloc_device(4096).expect("alloc_device(4096)");
    assert_eq!(buf.size(), 4096);
    backend.free_device(buf);
}

#[test]
fn vulkan_backend_alloc_zero_size_rejected() {
    let backend = VulkanBackend::new().expect("VulkanBackend::new");
    let err = backend
        .alloc_device(0)
        .expect_err("alloc_device(0) must be rejected");
    let msg = err.to_string();
    assert!(msg.contains("size = 0"), "unexpected error message: {msg}");
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
    let backend = VulkanBackend::new().expect("VulkanBackend::new");
    assert_zeroed_round_trip(&backend, 4096, "single");
}

#[test]
fn vulkan_backend_alloc_device_zeroed_concurrent() {
    // Regression: VkCommandPool + VkQueue both require external sync
    // (Vulkan "Threading Behavior" table).  Without the locks on
    // DeviceCtx + TransferContext, concurrent callers race in
    // run_one_shot — either driver crash or non-zero readbacks.
    use std::sync::Arc;
    use std::thread;

    const THREADS: usize = 4;
    const ITERS_PER_THREAD: usize = 8;

    let backend = Arc::new(VulkanBackend::new().expect("VulkanBackend::new"));
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
    // returns Err with "descriptor pool exhausted".  Pre-existing guard
    // had zero coverage; a refactor that reordered guard / allocate /
    // increment could silently regress.
    use gpu::backend::params::CompositeParams;

    // Keep in sync with `MAX_DESC_SETS_PER_PAGE` in
    // crates/gpu/src/backend/vulkan/recorder.rs (private constant).
    // The test parses the production error message below and asserts
    // the embedded `max` value equals this — so a production bump
    // without updating PER_PAGE_MAX fails loudly instead of silently
    // passing through the loop.
    const PER_PAGE_MAX: usize = 64;
    const N_PIXELS: u32 = 4;
    const BUF_BYTES: usize = (N_PIXELS as usize) * gpu::RGBA_BPP;

    let backend = VulkanBackend::new().expect("VulkanBackend::new");
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
        .expect_err("65th record_composite must be rejected");
    let msg = err.to_string();
    assert!(
        msg.contains("descriptor pool exhausted"),
        "expected exhaustion error, got: {msg}"
    );
    // Self-check PER_PAGE_MAX against production's actual limit, parsed
    // from the error format:
    //   "descriptor pool exhausted: 64 sets allocated this page (max 64)"
    let production_max: usize = msg
        .rsplit_once("(max ")
        .and_then(|(_, tail)| tail.split_once(')'))
        .and_then(|(num, _)| num.parse().ok())
        .unwrap_or_else(|| panic!("could not parse production max from error: {msg}"));
    assert_eq!(
        production_max, PER_PAGE_MAX,
        "test PER_PAGE_MAX is out of sync with production MAX_DESC_SETS_PER_PAGE"
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
    let backend = VulkanBackend::new().expect("VulkanBackend::new");
    let fence = backend.submit_transfer().expect("submit_transfer");
    backend
        .wait_transfer(fence)
        .expect("wait_transfer on immediate fence should succeed");
}

#[test]
fn vulkan_backend_alloc_device_zeroed_rejects_unaligned_size() {
    // vkCmdFillBuffer requires 4-byte-aligned size; alloc_device_zeroed
    // must surface this loudly rather than silently downgrading.
    let backend = VulkanBackend::new().expect("VulkanBackend::new");
    let err = backend
        .alloc_device_zeroed(17)
        .expect_err("alloc_device_zeroed(17) must be rejected (size not multiple of 4)");
    let msg = err.to_string();
    assert!(
        msg.contains("multiple of 4"),
        "expected alignment error, got: {msg}"
    );
}

#[test]
fn vulkan_backend_host_buffer_round_trip() {
    let backend = VulkanBackend::new().expect("VulkanBackend::new");
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
