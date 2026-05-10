//! Host↔device transfer helpers.
//!
//! Synchronous upload/download via a reusable `HOST_VISIBLE` staging
//! buffer (grown to the high-water-mark of any transfer seen so far)
//! and a one-time-submit command buffer on the compute queue, with
//! `vkQueueWaitIdle` for completion.
//!
//! The dedicated-transfer-queue path (overlap DMA with rendering) is a
//! follow-up — see ROADMAP Phase 10 Task 3 follow-ups.

#![expect(
    clippy::significant_drop_tightening,
    reason = "the staging Mutex<Option<HostBuffer>> is held across Vulkan calls intentionally — releasing it between the host-side write and the GPU-side wait would let a second upload observe the same staging buffer mid-DMA"
)]

use std::sync::{Arc, Mutex};

use ash::vk;

use crate::backend::{BackendError, Result};

use super::device::DeviceCtx;
use super::error::vk_err;
use super::memory::{DeviceBuffer, HostBuffer, SlabAllocator};

/// Convert a host-side `usize` length to `u64` and verify it doesn't
/// exceed the device-side buffer capacity.  `op_label` and `cap_label`
/// only show up in the error message, e.g. `("upload_sync", "dst capacity")`.
fn check_len(len: usize, capacity: u64, op_label: &str, cap_label: &str) -> Result<u64> {
    let len_u64 = u64::try_from(len)
        .map_err(|_| BackendError::msg(format!("{op_label}: length ({len}) exceeds u64")))?;
    if len_u64 > capacity {
        return Err(BackendError::msg(format!(
            "{op_label}: length ({len_u64}) exceeds {cap_label} ({capacity})"
        )));
    }
    Ok(len_u64)
}

pub(super) struct TransferContext {
    device: Arc<DeviceCtx>,
    /// Reusable command pool + cached primary command buffer for
    /// transient one-shot transfers.  A separate pool from the
    /// recorder's so resets don't interleave; matches the canonical
    /// "one pool per logical stream of submissions" pattern.
    ///
    /// `VkCommandPool` requires external sync (spec "Threading
    /// Behavior" table; VUID-vkResetCommandPool-commandPool-00040
    /// forbids reset while any buffer from the pool is recording or
    /// pending) — the `Mutex` is the only access path.  The cached
    /// `vk::CommandBuffer` is reused across calls: pool reset
    /// transitions it back to Initial state without freeing it,
    /// avoiding a `vkAllocateCommandBuffers` round-trip per transfer.
    cmd_pool: Mutex<TransferPool>,
    /// Reusable staging buffer, grown to the high-water-mark of any
    /// upload/download seen so far.  `None` until the first transfer;
    /// reallocated only when a request exceeds the current capacity.
    /// The `Mutex` guards the staging payload: a concurrent
    /// `upload_sync` would otherwise observe the same staging buffer
    /// mid-DMA.  Command-pool / queue external-sync are NOT this
    /// lock's job — see `cmd_pool` and `DeviceCtx::with_queue`.
    staging: Mutex<Option<HostBuffer>>,
}

/// Pool + cached cmd buffer pair guarded together by `cmd_pool`'s mutex.
struct TransferPool {
    pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
}

impl TransferContext {
    pub(super) fn new(device: Arc<DeviceCtx>) -> Result<Self> {
        let pool_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::TRANSIENT)
            .queue_family_index(device.compute_queue_family);
        // Safety: device is live; create_command_pool returns Result on failure.
        let pool = unsafe { device.device.create_command_pool(&pool_info, None) }
            .map_err(vk_err("vkCreateCommandPool (transfer)"))?;

        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        // Safety: pool is live; allocate_command_buffers returns Result on failure.
        let buffers = unsafe { device.device.allocate_command_buffers(&alloc_info) }
            .map_err(vk_err("vkAllocateCommandBuffers (transfer init)"))
            .inspect_err(|_| unsafe { device.device.destroy_command_pool(pool, None) })?;
        let cmd = buffers[0];

        Ok(Self {
            device,
            cmd_pool: Mutex::new(TransferPool { pool, cmd }),
            staging: Mutex::new(None),
        })
    }

    /// Synchronous upload: copy `src` into `dst[0..src.len()]`, block
    /// until complete.  Used by the trait's `upload_async` (which hands
    /// back an already-signalled fence today) and by tests.
    pub(super) fn upload_sync(
        &self,
        allocator: &SlabAllocator,
        dst: &DeviceBuffer,
        src: &[u8],
    ) -> Result<()> {
        if src.is_empty() {
            return Ok(());
        }
        let len = check_len(src.len(), dst.size(), "upload_sync", "dst capacity")?;
        // Hold the staging slot for the entire operation: write the
        // payload, kick off the GPU copy, wait for completion.  Releasing
        // the lock between write and wait would let a second upload
        // observe the same staging buffer mid-DMA.
        let mut staging_slot = self
            .staging
            .lock()
            .expect("transfer staging mutex poisoned");
        let staging = ensure_staging(allocator, &mut staging_slot, src.len())?;
        staging.as_mut_slice()[..src.len()].copy_from_slice(src);
        let staging_handle = staging.handle();
        let dst_handle = dst.handle();
        self.copy_via_staging(staging_handle, dst_handle, len)
    }

    /// Synchronous download: copy `src[0..dst.len()]` into `dst`, block
    /// until complete.
    pub(super) fn download_sync(
        &self,
        allocator: &SlabAllocator,
        src: &DeviceBuffer,
        dst: &mut [u8],
    ) -> Result<()> {
        if dst.is_empty() {
            return Ok(());
        }
        let len = check_len(dst.len(), src.size(), "download_sync", "src size")?;
        let mut staging_slot = self
            .staging
            .lock()
            .expect("transfer staging mutex poisoned");
        let staging = ensure_staging(allocator, &mut staging_slot, dst.len())?;
        let staging_handle = staging.handle();
        let src_handle = src.handle();
        self.copy_via_staging(src_handle, staging_handle, len)?;
        // GPU has finished writing the staging buffer (queue_wait_idle
        // inside copy_via_staging).  HOST_COHERENT means the read sees
        // the latest data without an explicit invalidate.
        dst.copy_from_slice(&staging.as_slice()[..dst.len()]);
        Ok(())
    }

    /// Submit a single `vkCmdCopyBuffer(src→dst, len)` and wait for it
    /// to complete.  Caller chooses which side is the staging buffer.
    fn copy_via_staging(&self, src: vk::Buffer, dst: vk::Buffer, len: u64) -> Result<()> {
        self.run_one_shot(|cmd| {
            let region = [vk::BufferCopy::default()
                .src_offset(0)
                .dst_offset(0)
                .size(len)];
            // Safety: cmd is in Recording state per run_one_shot's contract.
            unsafe {
                self.device.device.cmd_copy_buffer(cmd, src, dst, &region);
            }
        })
    }

    /// Zero-fill `dst[0..dst.size()]` on the GPU via `vkCmdFillBuffer`.
    ///
    /// `vkCmdFillBuffer` requires `dst.size()` to be a multiple of 4 —
    /// non-multiple-of-4 sizes return an error rather than silently
    /// rounding down (which would leave the trailing 1–3 bytes
    /// non-zero).
    pub(super) fn fill_zero(&self, dst: &DeviceBuffer) -> Result<()> {
        let size = dst.size();
        if size == 0 {
            return Ok(());
        }
        if !size.is_multiple_of(4) {
            return Err(BackendError::UnalignedFill {
                size,
                required_alignment: 4,
            });
        }
        let dst_handle = dst.handle();
        self.run_one_shot(|cmd| {
            // Safety: run_one_shot has cmd in Recording state; dst_handle
            // was created with TRANSFER_DST usage (DEVICE_USAGE in
            // memory.rs); size is a checked multiple of 4.
            unsafe {
                self.device
                    .device
                    .cmd_fill_buffer(cmd, dst_handle, 0, size, 0);
            }
        })
    }

    /// Reset the pool, record `f` into the cached cmd buffer, submit,
    /// wait idle.  Holds the `cmd_pool` mutex for the whole body so
    /// the buffer can't be reused before `vkQueueWaitIdle` returns
    /// (cmd buffers in Pending state must not be reset).
    fn run_one_shot<F: FnOnce(vk::CommandBuffer)>(&self, f: F) -> Result<()> {
        let pool = self.cmd_pool.lock().expect("transfer cmd_pool poisoned");
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // Safety: pool mutex held throughout; cmd buffer was allocated
        // from this pool at TransferContext::new and is reused across
        // calls.  reset_command_pool transitions it back to Initial.
        unsafe {
            self.device
                .device
                .reset_command_pool(pool.pool, vk::CommandPoolResetFlags::empty())
                .map_err(vk_err("vkResetCommandPool (transfer)"))?;
            self.device
                .device
                .begin_command_buffer(pool.cmd, &begin_info)
                .map_err(vk_err("vkBeginCommandBuffer (transfer)"))?;
        }

        f(pool.cmd);

        // Safety: cmd is in Recording state; pool mutex still held.
        unsafe {
            self.device
                .device
                .end_command_buffer(pool.cmd)
                .map_err(vk_err("vkEndCommandBuffer (transfer)"))?;
        }

        let submit_buffers = [pool.cmd];
        let submit = vk::SubmitInfo::default().command_buffers(&submit_buffers);
        self.device.with_queue(|q| {
            // Safety: queue exclusively held via with_queue; submit
            // outlives the call.
            unsafe {
                self.device
                    .device
                    .queue_submit(q, &[submit], vk::Fence::null())
                    .map_err(vk_err("vkQueueSubmit (transfer)"))?;
                self.device
                    .device
                    .queue_wait_idle(q)
                    .map_err(vk_err("vkQueueWaitIdle (transfer)"))?;
            }
            Ok(())
        })
    }
}

/// Make sure the cached staging buffer is at least `min_size` bytes;
/// allocate or grow as needed.  Returns a mutable reference to the
/// (now-large-enough) buffer.
fn ensure_staging<'a>(
    allocator: &SlabAllocator,
    slot: &'a mut Option<HostBuffer>,
    min_size: usize,
) -> Result<&'a mut HostBuffer> {
    let current_capacity = slot.as_ref().map_or(0, HostBuffer::size);
    let needed = u64::try_from(min_size).expect("min_size fits u64");
    if needed > current_capacity {
        // Grow geometrically so a sequence of slowly-increasing sizes
        // doesn't reallocate every call.
        let grown = std::cmp::max(needed, current_capacity.saturating_mul(2));
        let grown_usize = usize::try_from(grown)
            .map_err(|_| BackendError::msg("staging grow size exceeds usize"))?;
        // Drop the old buffer first so its VkBuffer + allocation are
        // released before we ask for the new one.
        *slot = None;
        *slot = Some(allocator.alloc_host(grown_usize)?);
    }
    Ok(slot.as_mut().expect("ensure_staging always populates"))
}

impl Drop for TransferContext {
    fn drop(&mut self) {
        // VulkanBackend::drop calls device_wait_idle before module Drops
        // run, so the cached cmd buffer is not in flight.  Destroying
        // the pool also frees its allocated buffers.
        let pool = self.cmd_pool.lock().expect("transfer cmd_pool poisoned");
        // Safety: pool is owned by us; no in-flight references.
        unsafe {
            self.device.device.destroy_command_pool(pool.pool, None);
        }
    }
}
