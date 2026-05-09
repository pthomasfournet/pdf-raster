//! Per-page command-buffer recorder.
//!
//! Lifecycle (matching the `GpuBackend` trait):
//!
//! - `begin_page`: reset the previous page's command pool, allocate a
//!   fresh primary command buffer, `vkBeginCommandBuffer`.
//! - `record_*`: bind the kernel's pipeline + a freshly allocated
//!   descriptor set, push scalars via push constants, dispatch, then
//!   emit a global compute→compute memory barrier so the next dispatch
//!   sees the writes.  Pessimistic but correct (mirrors the CUDA
//!   backend's single-stream serialisation).
//! - `submit_page`: end the command buffer, signal the timeline
//!   semaphore at value `next_value`, return a `PageFence` carrying the
//!   wait-value.
//! - `wait_page`: wait until the timeline semaphore reaches the fence's
//!   value.  After the wait succeeds, the descriptor pool used for the
//!   page is reset (it can be reused on the next page).
//!
//! ## Single-page-in-flight invariant
//!
//! For simplicity (mirroring the existing CUDA backend) the recorder
//! holds at most one page in flight.  `begin_page` immediately after
//! another `begin_page` without a `submit_page` in between is a
//! programmer error and produces a clear error.  Once the parity gate
//! is green, a small ring buffer (N=2-4) can replace the single slot
//! to allow CPU-record/GPU-execute overlap.

#![expect(
    clippy::needless_pass_by_value,
    reason = "record_* methods take params by value to mirror the GpuBackend trait shape; matches the CUDA backend's identical pragma in cuda/page_recorder.rs"
)]
#![expect(
    clippy::significant_drop_tightening,
    reason = "the inner Mutex<RecorderState> is held across Vulkan calls intentionally — single-page-in-flight design serialises every record_* through the same lock"
)]

use std::sync::Arc;
use std::sync::Mutex;

use ash::vk;

use crate::backend::params;
use crate::backend::{BackendError, Result};

use super::device::DeviceCtx;
use super::error::vk_err;
use super::pipeline::{KernelId, PipelineCache};

/// Synchronisation token returned by `submit_page`.  Holds the timeline
/// value the caller must wait on.
#[derive(Debug, Clone, Copy)]
pub struct PageFence {
    value: u64,
}

impl PageFence {
    /// A fence that has already signalled.  Used by upload paths that
    /// completed synchronously and have nothing for the caller to wait
    /// on; `wait_page(immediate)` returns immediately because the
    /// timeline semaphore's initial value is 0 ≥ 0.
    pub(super) const fn immediate() -> Self {
        Self { value: 0 }
    }
}

/// State machine for the recorder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    /// No page in flight; ready for `begin_page`.
    Idle,
    /// `begin_page` was called; `record_*` and `submit_page` are valid.
    Recording,
    /// `submit_page` was called; only `wait_page` is valid.
    Submitted,
}

/// Inner mutable state guarded by a single mutex.  Cheap to lock — the
/// recorder is single-page-in-flight by design.
struct RecorderState {
    state: State,
    /// Monotonically increasing timeline value; the next submission will
    /// signal `next_value` and the matching `wait_page` will wait on it.
    next_value: u64,
    /// Reusable command buffer (one allocation per submit; reset via
    /// pool reset between pages).
    cmd: vk::CommandBuffer,
    /// Descriptor pool used for sets allocated during `record_*` calls.
    /// Reset via `vkResetDescriptorPool` between pages.
    desc_pool: vk::DescriptorPool,
    /// Per-page scratch: descriptor sets allocated this page that need
    /// to outlive each `record_*` call but die at `wait_page`.  We don't
    /// track them individually because `vkResetDescriptorPool` blasts
    /// them all in one call.
    desc_sets_in_flight: u32,
}

pub(super) struct PageRecorder {
    device: Arc<DeviceCtx>,
    pipelines: Arc<PipelineCache>,
    timeline: vk::Semaphore,
    cmd_pool: vk::CommandPool,
    /// Mutex for the `RecorderState` above.
    inner: Mutex<RecorderState>,
}

/// Maximum descriptor sets allocated per page.  One per `record_*` call;
/// real renderers do tens, not hundreds — 64 is generous.
const MAX_DESC_SETS_PER_PAGE: u32 = 64;

impl PageRecorder {
    pub(super) fn new(device: Arc<DeviceCtx>, pipelines: Arc<PipelineCache>) -> Result<Self> {
        // Timeline semaphore.
        let mut tl_type = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(0);
        let tl_info = vk::SemaphoreCreateInfo::default().push_next(&mut tl_type);
        let timeline = unsafe { device.device.create_semaphore(&tl_info, None) }
            .map_err(vk_err("vkCreateSemaphore (timeline)"))?;

        // Command pool: TRANSIENT lets us reset between pages without
        // freeing/reallocating the buffer.
        let pool_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::TRANSIENT)
            .queue_family_index(device.compute_queue_family);
        let cmd_pool = unsafe { device.device.create_command_pool(&pool_info, None) }
            .map_err(vk_err("vkCreateCommandPool"))
            .inspect_err(|_| unsafe { device.device.destroy_semaphore(timeline, None) })?;

        // Allocate the single command buffer.
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(cmd_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cmd_buffers = unsafe { device.device.allocate_command_buffers(&alloc_info) }
            .map_err(vk_err("vkAllocateCommandBuffers"))
            .inspect_err(|_| unsafe {
                device.device.destroy_command_pool(cmd_pool, None);
                device.device.destroy_semaphore(timeline, None);
            })?;
        let cmd = cmd_buffers[0];

        // Descriptor pool: storage-buffer-only, sized for our worst case
        // kernel (tile_fill: 4 buffers per set) × MAX_DESC_SETS_PER_PAGE.
        let pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(MAX_DESC_SETS_PER_PAGE * 4)];
        let dp_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(MAX_DESC_SETS_PER_PAGE)
            .pool_sizes(&pool_sizes);
        let desc_pool = unsafe { device.device.create_descriptor_pool(&dp_info, None) }
            .map_err(vk_err("vkCreateDescriptorPool"))
            .inspect_err(|_| unsafe {
                device.device.destroy_command_pool(cmd_pool, None);
                device.device.destroy_semaphore(timeline, None);
            })?;

        Ok(Self {
            device,
            pipelines,
            timeline,
            cmd_pool,
            inner: Mutex::new(RecorderState {
                state: State::Idle,
                next_value: 1,
                cmd,
                desc_pool,
                desc_sets_in_flight: 0,
            }),
        })
    }

    pub(super) fn begin_page(&self) -> Result<()> {
        let mut s = self.inner.lock().expect("recorder mutex poisoned");
        if s.state != State::Idle {
            return Err(BackendError::msg(format!(
                "begin_page called from {:?} state; expected Idle",
                s.state
            )));
        }
        // Reset the command pool (frees/recycles the cmd buffer) and the
        // descriptor pool (frees all sets allocated last page).
        unsafe {
            self.device
                .device
                .reset_command_pool(self.cmd_pool, vk::CommandPoolResetFlags::empty())
                .map_err(vk_err("vkResetCommandPool"))?;
            self.device
                .device
                .reset_descriptor_pool(s.desc_pool, vk::DescriptorPoolResetFlags::empty())
                .map_err(vk_err("vkResetDescriptorPool"))?;
        }
        s.desc_sets_in_flight = 0;

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            self.device
                .device
                .begin_command_buffer(s.cmd, &begin_info)
                .map_err(vk_err("vkBeginCommandBuffer"))?;
        }
        s.state = State::Recording;
        Ok(())
    }

    pub(super) fn submit_page(&self) -> Result<PageFence> {
        let mut s = self.inner.lock().expect("recorder mutex poisoned");
        if s.state != State::Recording {
            return Err(BackendError::msg(format!(
                "submit_page called from {:?} state; expected Recording",
                s.state
            )));
        }
        unsafe {
            self.device
                .device
                .end_command_buffer(s.cmd)
                .map_err(vk_err("vkEndCommandBuffer"))?;
        }
        let signal_value = s.next_value;
        s.next_value = s
            .next_value
            .checked_add(1)
            .ok_or_else(|| BackendError::msg("timeline semaphore overflowed u64"))?;

        let mut tl_submit = vk::TimelineSemaphoreSubmitInfo::default()
            .signal_semaphore_values(std::slice::from_ref(&signal_value));
        let cmd_buffers = [s.cmd];
        let signal_semaphores = [self.timeline];
        let submit = vk::SubmitInfo::default()
            .command_buffers(&cmd_buffers)
            .signal_semaphores(&signal_semaphores)
            .push_next(&mut tl_submit);

        unsafe {
            self.device
                .device
                .queue_submit(self.device.compute_queue, &[submit], vk::Fence::null())
                .map_err(vk_err("vkQueueSubmit"))?;
        }
        s.state = State::Submitted;
        Ok(PageFence {
            value: signal_value,
        })
    }

    pub(super) fn wait_page(&self, fence: PageFence) -> Result<()> {
        // Hold the mutex across the wait so the Submitted → Idle
        // transition is atomic w.r.t. concurrent begin_page calls.
        let mut s = self.inner.lock().expect("recorder mutex poisoned");
        if s.state != State::Submitted {
            return Err(BackendError::msg(format!(
                "wait_page called from {:?} state; expected Submitted",
                s.state
            )));
        }
        let semaphores = [self.timeline];
        let values = [fence.value];
        let wait_info = vk::SemaphoreWaitInfo::default()
            .semaphores(&semaphores)
            .values(&values);
        // u64::MAX = wait forever.
        unsafe {
            self.device
                .device
                .wait_semaphores(&wait_info, u64::MAX)
                .map_err(vk_err("vkWaitSemaphores"))?;
        }
        s.state = State::Idle;
        Ok(())
    }

    // ── record_* ─────────────────────────────────────────────────────

    pub(super) fn record_composite(
        &self,
        p: params::CompositeParams<'_, super::VulkanBackend>,
    ) -> Result<()> {
        // Push: u32 n_pixels (4 bytes).  Workgroup size: 256.
        let n = p.n_pixels;
        let push: [u8; 4] = n.to_ne_bytes();
        self.dispatch_kernel(
            KernelId::Composite,
            &[p.src.handle(), p.dst.handle()],
            &[p.src.size(), p.dst.size()],
            &push,
            (n.div_ceil(256), 1, 1),
        )
    }

    pub(super) fn record_apply_soft_mask(
        &self,
        p: params::SoftMaskParams<'_, super::VulkanBackend>,
    ) -> Result<()> {
        let n = p.n_pixels;
        let push = n.to_ne_bytes();
        self.dispatch_kernel(
            KernelId::ApplySoftMask,
            &[p.pixels.handle(), p.mask.handle()],
            &[p.pixels.size(), p.mask.size()],
            &push,
            (n.div_ceil(256), 1, 1),
        )
    }

    pub(super) fn record_aa_fill(
        &self,
        p: params::AaFillParams<'_, super::VulkanBackend>,
    ) -> Result<()> {
        // Push: n_segs(u32), x_min(f32), y_min(f32), width(u32), height(u32), eo(i32).
        // x_min/y_min are 0 here for parity with the CUDA backend's choice.
        let mut push = [0u8; 24];
        push[0..4].copy_from_slice(&p.n_segs.to_ne_bytes());
        push[4..8].copy_from_slice(&0.0_f32.to_ne_bytes());
        push[8..12].copy_from_slice(&0.0_f32.to_ne_bytes());
        push[12..16].copy_from_slice(&p.width.to_ne_bytes());
        push[16..20].copy_from_slice(&p.height.to_ne_bytes());
        push[20..24].copy_from_slice(&i32::from(p.fill_rule != 0).to_ne_bytes());
        // Workgroup = 1 pixel (64 lanes). Grid = total pixel count.
        // The kernel uses a 1D dispatch over pixels which exceeds Vulkan's
        // guaranteed `maxComputeWorkGroupCount[0]` (65535) for any image
        // larger than ~256×256; `dispatch_kernel`'s `check_dispatch_size`
        // fails the call with a useful message before the driver does.
        // Spec follow-up: reparametrise as 2D (width × height) workgroups.
        let total_pixels = p
            .width
            .checked_mul(p.height)
            .ok_or_else(|| BackendError::msg("aa_fill: width * height overflowed u32"))?;
        self.dispatch_kernel(
            KernelId::AaFill,
            &[p.segs.handle(), p.coverage.handle()],
            &[p.segs.size(), p.coverage.size()],
            &push,
            (total_pixels, 1, 1),
        )
    }

    pub(super) fn record_tile_fill(
        &self,
        p: params::TileFillParams<'_, super::VulkanBackend>,
    ) -> Result<()> {
        // Push: grid_w(u32), out_w(u32), out_h(u32), eo(i32).
        let grid_w = p.width.div_ceil(crate::TILE_W);
        let mut push = [0u8; 16];
        push[0..4].copy_from_slice(&grid_w.to_ne_bytes());
        push[4..8].copy_from_slice(&p.width.to_ne_bytes());
        push[8..12].copy_from_slice(&p.height.to_ne_bytes());
        push[12..16].copy_from_slice(&i32::from(p.fill_rule != 0).to_ne_bytes());
        let grid_h = p.height.div_ceil(crate::TILE_H);
        self.dispatch_kernel(
            KernelId::TileFill,
            &[
                p.records.handle(),
                p.tile_starts.handle(),
                p.tile_counts.handle(),
                p.coverage.handle(),
            ],
            &[
                p.records.size(),
                p.tile_starts.size(),
                p.tile_counts.size(),
                p.coverage.size(),
            ],
            &push,
            (grid_w, grid_h, 1),
        )
    }

    pub(super) fn record_icc_clut(
        &self,
        p: params::IccClutParams<'_, super::VulkanBackend>,
    ) -> Result<()> {
        // Push: grid_n(u32), n(u32).  Recover grid_n from CLUT length
        // via the shared helper used by both backends.
        let clut_bytes = usize::try_from(p.clut.size())
            .map_err(|_| BackendError::msg("ICC CLUT size does not fit in usize"))?;
        let grid_n = params::grid_n_from_clut_len(clut_bytes).ok_or_else(|| {
            BackendError::msg(format!(
                "ICC CLUT length {clut_bytes} is not a valid grid_n^4 * 3"
            ))
        })?;
        let mut push = [0u8; 8];
        push[0..4].copy_from_slice(&grid_n.to_ne_bytes());
        push[4..8].copy_from_slice(&p.n_pixels.to_ne_bytes());
        self.dispatch_kernel(
            KernelId::IccClut,
            &[p.cmyk.handle(), p.rgb.handle(), p.clut.handle()],
            &[p.cmyk.size(), p.rgb.size(), p.clut.size()],
            &push,
            (p.n_pixels.div_ceil(256), 1, 1),
        )
    }

    #[expect(
        clippy::unused_self,
        reason = "shape-only stub until inv_ctm push-const re-port lands; method signature matches trait"
    )]
    pub(super) fn record_blit_image(
        &self,
        p: params::BlitParams<'_, super::VulkanBackend>,
    ) -> Result<()> {
        // The blit kernel takes inv_ctm as a StructuredBuffer<float>; we
        // need a small buffer holding the 6 floats.  Allocating one per
        // record_* call here would be wasteful; the preferred design is
        // to pass inv_ctm via push constants.  Spec follow-up: re-emit
        // blit_image.slang to read inv_ctm from push constants instead
        // of a buffer.  For now, error loudly so callers don't think
        // it's silently working.
        let _ = p;
        Err(BackendError::msg(
            "record_blit_image: Vulkan path needs inv_ctm push-const re-port (spec follow-up)",
        ))
    }

    /// Bail if `groups` exceeds the device's `maxComputeWorkGroupCount`
    /// per axis.  Without this, an oversized dispatch returns an opaque
    /// `ERROR_DEVICE_LOST` from the driver later — much harder to debug.
    fn check_dispatch_size(&self, groups: (u32, u32, u32), kernel: &str) -> Result<()> {
        let limits = self.device.max_workgroup_count;
        if groups.0 > limits[0] || groups.1 > limits[1] || groups.2 > limits[2] {
            return Err(BackendError::msg(format!(
                "{kernel}: dispatch ({},{},{}) exceeds device maxComputeWorkGroupCount ({},{},{})",
                groups.0, groups.1, groups.2, limits[0], limits[1], limits[2]
            )));
        }
        Ok(())
    }

    /// Allocate a descriptor set, bind pipeline + buffers + push constants,
    /// dispatch, emit a compute→compute memory barrier.  Holds the mutex
    /// for the whole body so two threads can't race on `s.cmd` (command
    /// buffers are not externally synchronised by default).
    fn dispatch_kernel(
        &self,
        id: KernelId,
        buffers: &[vk::Buffer],
        sizes: &[u64],
        push: &[u8],
        groups: (u32, u32, u32),
    ) -> Result<()> {
        debug_assert_eq!(buffers.len(), sizes.len());

        // Look up pipeline handles BEFORE locking — `pipelines.handles`
        // can take its own internal locks (PipelineCache's per-slot
        // OnceLock); locking ours first would risk an inversion.
        let handles = self.pipelines.handles(id)?;
        debug_assert_eq!(buffers.len(), handles.n_storage_buffers as usize);

        self.check_dispatch_size(groups, id.label())?;

        let mut s = self.inner.lock().expect("recorder mutex poisoned");
        if s.state != State::Recording {
            return Err(BackendError::msg(format!(
                "dispatch_kernel called from {:?} state; expected Recording",
                s.state
            )));
        }
        if s.desc_sets_in_flight >= MAX_DESC_SETS_PER_PAGE {
            return Err(BackendError::msg(format!(
                "descriptor pool exhausted: {} sets allocated this page (max {})",
                s.desc_sets_in_flight, MAX_DESC_SETS_PER_PAGE
            )));
        }

        // Allocate one descriptor set from the per-page pool.
        let layouts = [handles.descriptor_set_layout];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(s.desc_pool)
            .set_layouts(&layouts);
        let sets = unsafe { self.device.device.allocate_descriptor_sets(&alloc_info) }
            .map_err(vk_err("vkAllocateDescriptorSets"))?;
        let set = sets[0];

        // Build a write per buffer.  The DescriptorBufferInfo array must
        // outlive the write_descriptor_sets call, so own them locally.
        let buf_infos: Vec<vk::DescriptorBufferInfo> = buffers
            .iter()
            .zip(sizes.iter())
            .map(|(buf, sz)| {
                vk::DescriptorBufferInfo::default()
                    .buffer(*buf)
                    .offset(0)
                    .range(*sz)
            })
            .collect();
        // One write per binding.  We can't use a single multi-binding
        // write because each VkDescriptorBufferInfo has a different
        // .buffer field.
        let writes: Vec<vk::WriteDescriptorSet<'_>> = buf_infos
            .iter()
            .enumerate()
            .map(|(i, info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(u32::try_from(i).expect("binding fits u32"))
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(info))
            })
            .collect();

        // Safety: writes outlives the call; each write's buffer_info
        // points into buf_infos which is also live.
        unsafe {
            self.device.device.update_descriptor_sets(&writes, &[]);
        }

        // Safety: cmd is in recording state; pipeline & layout are live.
        unsafe {
            self.device.device.cmd_bind_pipeline(
                s.cmd,
                vk::PipelineBindPoint::COMPUTE,
                handles.pipeline,
            );
            self.device.device.cmd_bind_descriptor_sets(
                s.cmd,
                vk::PipelineBindPoint::COMPUTE,
                handles.layout,
                0,
                &[set],
                &[],
            );
            if !push.is_empty() {
                self.device.device.cmd_push_constants(
                    s.cmd,
                    handles.layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    push,
                );
            }
            self.device
                .device
                .cmd_dispatch(s.cmd, groups.0, groups.1, groups.2);

            // Global compute→compute memory barrier so the next dispatch
            // sees these writes.  Pessimistic but correct (the CUDA
            // backend serialises the same way).
            let mem_barrier = [vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_access_mask(
                    vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE,
                )];
            let dep_info = vk::DependencyInfo::default().memory_barriers(&mem_barrier);
            self.device.device.cmd_pipeline_barrier2(s.cmd, &dep_info);
        }

        // Increment under the same lock so two concurrent record_* calls
        // can't both observe an unincremented count and over-allocate.
        s.desc_sets_in_flight = s.desc_sets_in_flight.saturating_add(1);
        Ok(())
    }
}

impl Drop for PageRecorder {
    fn drop(&mut self) {
        // Recover from a poisoned mutex rather than panicking inside
        // Drop — a panic here is a double-panic which aborts the process
        // and can mask the original failure.  After a poison, the inner
        // state may be inconsistent, but the Vulkan handles themselves
        // are still owned by us and need to be destroyed.
        let s = match self.inner.lock() {
            Ok(g) => g,
            Err(poisoned) => {
                log::warn!("PageRecorder mutex was poisoned at drop; destroying handles anyway");
                poisoned.into_inner()
            }
        };
        // Safety: device_wait_idle was called by VulkanBackend::drop
        // before us, so all submissions are complete and no work
        // references these handles.
        unsafe {
            self.device
                .device
                .destroy_descriptor_pool(s.desc_pool, None);
            self.device.device.destroy_command_pool(self.cmd_pool, None);
            self.device.device.destroy_semaphore(self.timeline, None);
        }
    }
}
