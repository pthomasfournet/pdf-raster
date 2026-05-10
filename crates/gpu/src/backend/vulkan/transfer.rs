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
    /// Reusable command pool for transient one-shot transfer command
    /// buffers.  A separate pool from the recorder's so resets don't
    /// interleave; matches the canonical "one pool per logical stream
    /// of submissions" pattern.
    ///
    /// Vulkan requires `VkCommandPool` to be externally synchronized
    /// (spec "Threading Behavior" table; VUID-vkResetCommandPool-
    /// commandPool-00040 forbids reset while any buffer from the pool
    /// is recording or pending).  `cmd_pool_lock` covers `cmd_pool`'s
    /// reset / `vkAllocateCommandBuffers` / record / `vkEndCommandBuffer`
    /// window inside `run_one_shot`.
    cmd_pool: vk::CommandPool,
    /// Pool-level external-synchronization lock for `cmd_pool`.
    /// See the field doc on `cmd_pool` for why this is required.
    cmd_pool_lock: Mutex<()>,
    /// Reusable staging buffer, grown to the high-water-mark of any
    /// upload/download seen so far.  `None` until the first transfer;
    /// reallocated only when a request exceeds the current capacity.
    /// The `Mutex` guards the staging payload itself: a concurrent
    /// `upload_sync` would otherwise observe the same staging buffer
    /// mid-DMA.  Queue / command-pool external-sync are NOT this
    /// lock's job — see `cmd_pool_lock` and `DeviceCtx::submit_lock`.
    staging: Mutex<Option<HostBuffer>>,
}

impl TransferContext {
    pub(super) fn new(device: Arc<DeviceCtx>) -> Result<Self> {
        let pool_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::TRANSIENT)
            .queue_family_index(device.compute_queue_family);
        // Safety: device is live; create_command_pool returns Result on failure.
        let cmd_pool = unsafe { device.device.create_command_pool(&pool_info, None) }
            .map_err(vk_err("vkCreateCommandPool (transfer)"))?;
        Ok(Self {
            device,
            cmd_pool,
            cmd_pool_lock: Mutex::new(()),
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
    /// Skips the host vec + staging round-trip used by the
    /// `upload_sync(zeros)` shape: the GPU writes zeros to its own
    /// memory directly.  For an RGBA8 page buffer (≥ 8 MB at 1280×1664,
    /// up to ~33 MB at 4K) that eliminates both the host allocation and
    /// the `PCIe` transfer, leaving only a queue submit + wait.
    ///
    /// `vkCmdFillBuffer` requires `dst.size()` to be a multiple of 4
    /// (the spec rounds `VK_WHOLE_SIZE` *down* to a multiple of 4 from
    /// the buffer extent, which would leave the trailing 1–3 bytes
    /// non-zero and silently break the `alloc_device_zeroed` contract).
    /// Returns an error for non-multiple-of-4 sizes; every realistic
    /// caller (RGBA8 pages, `u32`-indexed scratch) is naturally aligned.
    pub(super) fn fill_zero(&self, dst: &DeviceBuffer) -> Result<()> {
        let size = dst.size();
        if size == 0 {
            return Ok(());
        }
        if !size.is_multiple_of(4) {
            return Err(BackendError::msg(format!(
                "fill_zero: buffer size ({size}) must be a multiple of 4 for vkCmdFillBuffer"
            )));
        }
        let dst_handle = dst.handle();
        self.run_one_shot(|cmd| {
            // Safety: cmd is in Recording state per run_one_shot's
            // contract; dst_handle was created with TRANSFER_DST usage
            // (see DEVICE_USAGE in memory.rs); size is a multiple of 4.
            unsafe {
                self.device
                    .device
                    .cmd_fill_buffer(cmd, dst_handle, 0, size, 0);
            }
        })
    }

    /// Allocate a one-shot command buffer, run `f` to record into it,
    /// submit it, wait for the queue to idle.  Slow but correct.
    ///
    /// Resets the pool at the start of each call so a prior failed run
    /// can't leak its command buffer; this is leak-safe in all error
    /// paths below.
    ///
    /// ## Synchronization
    /// Takes `cmd_pool_lock` for the entire body so concurrent callers
    /// can't race on `vkResetCommandPool` / `vkAllocateCommandBuffers`
    /// / record / `vkEndCommandBuffer` (Vulkan requires external sync
    /// on `VkCommandPool`).  Takes `device.submit_lock` for just the
    /// queue ops at the end so concurrent submitters (the per-page
    /// recorder, other `run_one_shot` calls) don't race on `VkQueue`.
    /// Two locks because `cmd_pool_lock` is per-`TransferContext` while
    /// `submit_lock` lives on the shared `DeviceCtx` (the recorder
    /// reaches for the same queue with its own command pool).
    fn run_one_shot<F: FnOnce(vk::CommandBuffer)>(&self, f: F) -> Result<()> {
        let _pool_guard = self
            .cmd_pool_lock
            .lock()
            .expect("transfer cmd_pool_lock poisoned");
        unsafe {
            self.device
                .device
                .reset_command_pool(self.cmd_pool, vk::CommandPoolResetFlags::empty())
                .map_err(vk_err("vkResetCommandPool (transfer)"))?;
        }

        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.cmd_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        // Safety: cmd_pool live; struct outlives call; cmd_pool_lock
        // held so no concurrent thread can reset or allocate from it.
        let buffers = unsafe { self.device.device.allocate_command_buffers(&alloc_info) }
            .map_err(vk_err("vkAllocateCommandBuffers (transfer)"))?;
        let cmd = buffers[0];

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // Safety: cmd is freshly allocated, in InitialState; cmd_pool_lock
        // held so no concurrent record can touch the same buffer.
        unsafe {
            self.device
                .device
                .begin_command_buffer(cmd, &begin_info)
                .map_err(vk_err("vkBeginCommandBuffer (transfer)"))?;
        }

        f(cmd);

        // Safety: cmd is in Recording state; cmd_pool_lock held.
        unsafe {
            self.device
                .device
                .end_command_buffer(cmd)
                .map_err(vk_err("vkEndCommandBuffer (transfer)"))?;
        }

        let submit_buffers = [cmd];
        let submit = vk::SubmitInfo::default().command_buffers(&submit_buffers);
        // VkQueue is externally synchronized — take submit_lock for the
        // submit + wait window only.  Holding cmd_pool_lock across the
        // wait is also intentional: the cmd buffer must not be reused
        // (next call's reset_command_pool) while still in pending state.
        {
            let _submit_guard = self
                .device
                .submit_lock
                .lock()
                .expect("device submit_lock poisoned");
            // Safety: queue + submit_info live for this call; submit_lock
            // held so no other thread can touch the queue.
            unsafe {
                self.device
                    .device
                    .queue_submit(self.device.compute_queue, &[submit], vk::Fence::null())
                    .map_err(vk_err("vkQueueSubmit (transfer)"))?;
                self.device
                    .device
                    .queue_wait_idle(self.device.compute_queue)
                    .map_err(vk_err("vkQueueWaitIdle (transfer)"))?;
            }
        }
        // No explicit free: the next call's reset_command_pool reclaims
        // the buffer.  This mirrors the recorder's pool-reset idiom.
        Ok(())
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
        // The cached staging buffer is dropped via the `Mutex<Option<...>>`
        // field's normal drop.  Order matters: the HostBuffer's Drop
        // calls into its Arc<Inner> (the SlabAllocator) and ultimately
        // vkDestroyBuffer on the device.  The device is kept alive by
        // our own Arc<DeviceCtx>; the SlabAllocator is kept alive by
        // the HostBuffer's parent Arc.  Field declaration order means
        // staging drops *before* cmd_pool, which is fine — they're
        // independent.
        // Safety: VulkanBackend::drop calls device_wait_idle before any
        // module Drop runs, so no command buffers are in flight.
        unsafe {
            self.device.device.destroy_command_pool(self.cmd_pool, None);
        }
    }
}
