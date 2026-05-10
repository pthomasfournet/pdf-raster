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

#[test]
fn vulkan_backend_alloc_device_zeroed_returns_zero_bytes() {
    // 4 KB is comfortably above the alignment/page granularity and
    // small enough to round-trip back through download_sync without
    // grow-the-staging churn.  Multiple-of-4 size exercises the
    // vkCmdFillBuffer fast path in alloc_device_zeroed.
    let backend = VulkanBackend::new().expect("VulkanBackend::new");
    let buf = backend
        .alloc_device_zeroed(4096)
        .expect("alloc_device_zeroed(4096)");
    let mut readback = vec![0xAAu8; 4096];
    backend
        .download_sync(&buf, &mut readback)
        .expect("download_sync");
    assert!(
        readback.iter().all(|&b| b == 0),
        "alloc_device_zeroed returned non-zero bytes: first non-zero index = {:?}",
        readback.iter().position(|&b| b != 0)
    );
    backend.free_device(buf);
}

#[test]
fn vulkan_backend_alloc_device_zeroed_unaligned_size() {
    // Size 17 is not a multiple of 4; alloc_device_zeroed must still
    // return all-zero bytes via the host-vec fallback path.  Guards the
    // contract for callers that don't (or can't) round their alloc up.
    let backend = VulkanBackend::new().expect("VulkanBackend::new");
    let buf = backend
        .alloc_device_zeroed(17)
        .expect("alloc_device_zeroed(17)");
    let mut readback = vec![0xAAu8; 17];
    backend
        .download_sync(&buf, &mut readback)
        .expect("download_sync");
    assert!(
        readback.iter().all(|&b| b == 0),
        "alloc_device_zeroed(17) returned non-zero bytes: {readback:?}"
    );
    backend.free_device(buf);
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
