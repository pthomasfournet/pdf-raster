//! Host↔device transfer helpers.
//!
//! Today's implementation is the simplest correct shape: synchronous
//! one-shot upload/download via a `HOST_VISIBLE` staging buffer and a
//! one-time-submit command buffer on the compute queue, with
//! `vkQueueWaitIdle` for completion.
//!
//! Spec follow-up: a dedicated transfer queue (`TRANSFER` family without
//! `GRAPHICS` or `COMPUTE`) plus a timeline-semaphore handoff so the
//! prefetcher can overlap uploads with rendering.  That's tracked in
//! the spec's "Vulkan dedicated transfer queue for the prefetcher"
//! section and gated on the prefetcher's measured critical path.

use std::sync::Arc;

use ash::vk;

use crate::backend::{BackendError, Result};

use super::device::DeviceCtx;
use super::error::vk_err;
use super::memory::{DeviceBuffer, SlabAllocator};

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
    cmd_pool: vk::CommandPool,
}

impl TransferContext {
    pub(super) fn new(device: Arc<DeviceCtx>) -> Result<Self> {
        let pool_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::TRANSIENT)
            .queue_family_index(device.compute_queue_family);
        // Safety: device is live; create_command_pool returns Result on failure.
        let cmd_pool = unsafe { device.device.create_command_pool(&pool_info, None) }
            .map_err(vk_err("vkCreateCommandPool (transfer)"))?;
        Ok(Self { device, cmd_pool })
    }

    /// Synchronous upload: copy `src` into `dst[0..src.len()]`, block
    /// until complete.
    ///
    /// Used by the trait's `upload_async` (which currently completes
    /// before returning, so the fence it hands back is already signalled)
    /// and directly by tests.  A real async path with a dedicated transfer
    /// queue is a spec follow-up.
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
        let mut staging = allocator.alloc_host(src.len())?;
        staging.as_mut_slice().copy_from_slice(src);
        self.copy_via_staging(staging.handle(), dst.handle(), len)
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
        let mut staging = allocator.alloc_host(dst.len())?;
        self.copy_via_staging(src.handle(), staging.handle(), len)?;
        dst.copy_from_slice(staging.as_mut_slice());
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

    /// Allocate a one-shot command buffer, run `f` to record into it,
    /// submit it, wait for the queue to idle.  Slow but correct.
    ///
    /// Resets the pool at the start of each call so a prior failed run
    /// can't leak its command buffer; this is leak-safe in all error
    /// paths below.
    fn run_one_shot<F: FnOnce(vk::CommandBuffer)>(&self, f: F) -> Result<()> {
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
        // Safety: cmd_pool live; struct outlives call.
        let buffers = unsafe { self.device.device.allocate_command_buffers(&alloc_info) }
            .map_err(vk_err("vkAllocateCommandBuffers (transfer)"))?;
        let cmd = buffers[0];

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // Safety: cmd is freshly allocated, in InitialState.
        unsafe {
            self.device
                .device
                .begin_command_buffer(cmd, &begin_info)
                .map_err(vk_err("vkBeginCommandBuffer (transfer)"))?;
        }

        f(cmd);

        // Safety: cmd is in Recording state.
        unsafe {
            self.device
                .device
                .end_command_buffer(cmd)
                .map_err(vk_err("vkEndCommandBuffer (transfer)"))?;
        }

        let submit_buffers = [cmd];
        let submit = vk::SubmitInfo::default().command_buffers(&submit_buffers);
        // Safety: queue + submit_info live for this call.
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
        // No explicit free: the next call's reset_command_pool reclaims
        // the buffer.  This mirrors the recorder's pool-reset idiom.
        Ok(())
    }
}

impl Drop for TransferContext {
    fn drop(&mut self) {
        // Safety: VulkanBackend::drop calls device_wait_idle before any
        // module Drop runs, so no command buffers are in flight.
        unsafe {
            self.device.device.destroy_command_pool(self.cmd_pool, None);
        }
    }
}
