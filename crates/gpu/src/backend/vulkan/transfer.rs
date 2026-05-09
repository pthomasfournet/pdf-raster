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
    /// until complete, return.
    ///
    /// Used by the trait's `upload_async` (which currently completes
    /// before returning, so the fence it hands back is already
    /// signalled) and directly by tests.  A real async path with a
    /// dedicated transfer queue is a spec follow-up.
    pub(super) fn upload_sync(
        &self,
        allocator: &SlabAllocator,
        dst: &DeviceBuffer,
        src: &[u8],
    ) -> Result<()> {
        if src.is_empty() {
            return Ok(());
        }
        let src_len = u64::try_from(src.len())
            .map_err(|_| BackendError::msg("upload_sync: src.len() exceeds u64"))?;
        if src_len > dst.size() {
            return Err(BackendError::msg(format!(
                "upload_sync: src.len() ({src_len}) exceeds dst capacity ({})",
                dst.size()
            )));
        }

        // Stage: HOST_VISIBLE buffer of exactly src.len() bytes.
        let mut staging = allocator.alloc_host(src.len())?;
        staging.as_mut_slice().copy_from_slice(src);

        self.run_one_shot(|cmd| unsafe {
            let region = [vk::BufferCopy::default()
                .src_offset(0)
                .dst_offset(0)
                .size(src_len)];
            self.device
                .device
                .cmd_copy_buffer(cmd, staging.handle(), dst.handle(), &region);
        })
    }

    /// Synchronous download: copy `src[0..len]` into `dst`, block until
    /// complete.
    pub(super) fn download_sync(
        &self,
        allocator: &SlabAllocator,
        src: &DeviceBuffer,
        dst: &mut [u8],
    ) -> Result<()> {
        if dst.is_empty() {
            return Ok(());
        }
        let dst_len = u64::try_from(dst.len())
            .map_err(|_| BackendError::msg("download_sync: dst.len() exceeds u64"))?;
        if dst_len > src.size() {
            return Err(BackendError::msg(format!(
                "download_sync: dst.len() ({dst_len}) exceeds src size ({})",
                src.size()
            )));
        }

        let mut staging = allocator.alloc_host(dst.len())?;

        self.run_one_shot(|cmd| unsafe {
            let region = [vk::BufferCopy::default()
                .src_offset(0)
                .dst_offset(0)
                .size(dst_len)];
            self.device
                .device
                .cmd_copy_buffer(cmd, src.handle(), staging.handle(), &region);
        })?;

        dst.copy_from_slice(staging.as_mut_slice());
        Ok(())
    }

    /// Allocate a one-shot command buffer, run `f` to record into it,
    /// submit it, wait for the queue to idle, then reset the pool.
    /// Slow but correct; only used for upload/download today.
    ///
    /// Pool reset (rather than `free_command_buffers`) is the canonical
    /// idiom for one-shot command buffers and is leak-safe even when an
    /// error path bails before the explicit free.  We reset *at the
    /// start* of each run so the previous transfer's buffer is reclaimed
    /// regardless of how that one ended.
    fn run_one_shot<F: FnOnce(vk::CommandBuffer)>(&self, f: F) -> Result<()> {
        // Reset upfront so a prior failed transfer can't leave the pool
        // dirty.  TRANSIENT lets us reset cheaply.  Safe to call even on
        // an empty pool.
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
