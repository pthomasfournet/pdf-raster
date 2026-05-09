//! Pipeline cache: SPIR-V → `VkPipeline` per kernel.
//!
//! Each of the six kernels (`composite_rgba8`, `apply_soft_mask`, `aa_fill`,
//! `tile_fill`, `icc_clut`, `blit_image`) gets its own descriptor set layout +
//! pipeline layout + compute pipeline, lazy-created on first dispatch.
//! The SPIR-V blobs are baked into the binary via `include_bytes!` of the
//! build-script outputs.
//!
//! ## Descriptor model
//!
//! All kernels use a single descriptor set (set = 0) with N storage
//! buffers.  Scalar uniforms (`n_pixels`, width, height, eo, `inv_ctm`[6])
//! are passed via push constants (max 128 bytes — Vulkan's guaranteed
//! minimum is 128, every desktop driver supports at least that).
//!
//! ## Subgroup size for `aa_fill`
//!
//! `aa_fill` is the only kernel using subgroup ops.  Per the spec we
//! ship subgroup-size-agnostic Slang; the runtime determines the wave
//! width.  We don't pin a `requiredSubgroupSize` — the kernel adapts via
//! its `groupshared` cross-subgroup reduction.

use std::sync::Arc;

use ash::vk;

use crate::backend::{BackendError, Result};

use super::device::DeviceCtx;
use super::error::vk_err;

/// Identifier for one of the six compute kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[expect(
    dead_code,
    reason = "BlitImage variant is wired once the inv_ctm push-const re-port lands; spec follow-up in record_blit_image"
)]
pub(super) enum KernelId {
    Composite,
    ApplySoftMask,
    AaFill,
    TileFill,
    IccClut,
    BlitImage,
}

impl KernelId {
    /// SPIR-V blob for this kernel, baked into the binary at build time.
    const fn spirv(self) -> &'static [u8] {
        match self {
            Self::Composite => include_bytes!(concat!(env!("OUT_DIR"), "/composite_rgba8.spv")),
            Self::ApplySoftMask => {
                include_bytes!(concat!(env!("OUT_DIR"), "/apply_soft_mask.spv"))
            }
            Self::AaFill => include_bytes!(concat!(env!("OUT_DIR"), "/aa_fill.spv")),
            Self::TileFill => include_bytes!(concat!(env!("OUT_DIR"), "/tile_fill.spv")),
            Self::IccClut => include_bytes!(concat!(env!("OUT_DIR"), "/icc_clut.spv")),
            Self::BlitImage => include_bytes!(concat!(env!("OUT_DIR"), "/blit_image.spv")),
        }
    }

    /// SPIR-V entry-point name.  `slangc` renames every entry point to
    /// `"main"` regardless of the source function name (no flag suppresses
    /// this in the LunarG-bundled 2025.7.1 slangc), so all kernels share
    /// the same `"main"` name in the .spv binary.  The `-entry` arg in
    /// build.rs picks *which* function ends up as `"main"` for kernels
    /// that have multiple entry points (`icc_clut.slang`'s matrix vs CLUT).
    const fn entry_point(self) -> &'static str {
        // All slangc-emitted SPIR-V uses "main" — see comment above.
        let _ = self;
        "main"
    }

    /// Number of `STORAGE_BUFFER` descriptors in set 0.
    ///
    /// Order matches the Slang signature so the recorder binds in the
    /// same order it builds the descriptor write list.
    #[expect(
        clippy::match_same_arms,
        reason = "each variant gets its own arm + buffer-list comment so future kernel additions surface the right binding count next to the relevant kernel"
    )]
    const fn n_storage_buffers(self) -> u32 {
        match self {
            // (src, dst)
            Self::Composite => 2,
            // (pixels, mask)
            Self::ApplySoftMask => 2,
            // (segs, coverage)
            Self::AaFill => 2,
            // (records, tile_starts, tile_counts, coverage)
            Self::TileFill => 4,
            // (cmyk, rgb, clut)
            Self::IccClut => 3,
            // (src, dst_rgba, inv_ctm)
            Self::BlitImage => 3,
        }
    }
}

/// One compiled kernel: shader module + descriptor layout + pipeline layout + pipeline.
struct CompiledKernel {
    descriptor_set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    shader_module: vk::ShaderModule,
}

/// Lazy pipeline cache.  Uses `OnceLock<CompiledKernel>` per slot — one
/// dispatch builds the pipeline, the rest reuse it.
pub(super) struct PipelineCache {
    device: Arc<DeviceCtx>,
    /// Slots indexed by `KernelId as usize`.  `OnceLock` so the first
    /// caller initialises and the rest read; thread-safe by construction.
    slots: [std::sync::OnceLock<CompiledKernel>; 6],
}

impl PipelineCache {
    /// Construct an empty cache.  No pipelines are built until first
    /// dispatch.
    #[expect(
        clippy::unnecessary_wraps,
        reason = "Result kept for forward-compat: pipeline-cache file load (spec follow-up) can fail with IO/Vulkan errors"
    )]
    pub(super) fn new(device: Arc<DeviceCtx>) -> Result<Arc<Self>> {
        Ok(Arc::new(Self {
            device,
            slots: Default::default(),
        }))
    }

    /// Get (or build, if first call) the compiled kernel for `id`.
    fn get(&self, id: KernelId) -> Result<&CompiledKernel> {
        let slot = &self.slots[id as usize];
        // OnceLock::get_or_try_init isn't stable; use the explicit pattern.
        if let Some(c) = slot.get() {
            return Ok(c);
        }
        let compiled = self.compile(id)?;
        // Race: if another thread races us we discard ours; both produce
        // the same VkPipeline shape so functional correctness holds, but
        // the loser leaks a pipeline.  Tolerated for first-dispatch
        // contention in a backend that's almost always single-threaded
        // per page.  TODO: switch to OnceLock::get_or_try_init when stable.
        match slot.set(compiled) {
            Ok(()) => Ok(slot.get().expect("just set")),
            Err(extra) => {
                // Someone beat us; clean up our duplicate.
                self.destroy_one(&extra);
                Ok(slot.get().expect("set by the other thread"))
            }
        }
    }

    fn compile(&self, id: KernelId) -> Result<CompiledKernel> {
        let bindings: Vec<vk::DescriptorSetLayoutBinding<'_>> = (0..id.n_storage_buffers())
            .map(|i| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(i)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();
        let dsl_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        // Safety: bindings outlives this call.
        let descriptor_set_layout = unsafe {
            self.device
                .device
                .create_descriptor_set_layout(&dsl_info, None)
        }
        .map_err(vk_err("vkCreateDescriptorSetLayout"))?;

        let layouts = [descriptor_set_layout];
        let push_ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            // 128 bytes: Vulkan's guaranteed minimum max push-constant size.
            // Our largest kernel push struct is well under this (icc_clut
            // pushes 8 bytes; blit_image pushes 56).
            .size(128)];
        let pl_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&layouts)
            .push_constant_ranges(&push_ranges);
        let pipeline_layout = unsafe { self.device.device.create_pipeline_layout(&pl_info, None) }
            .map_err(vk_err("vkCreatePipelineLayout"))
            .inspect_err(|_| unsafe {
                self.device
                    .device
                    .destroy_descriptor_set_layout(descriptor_set_layout, None);
            })?;

        let spirv = id.spirv();
        // Safety: SPIR-V comes from `slangc -target spirv` at build time;
        // it's pre-validated by `spirv-val` and aligned to 4 bytes (any
        // SPIR-V file is, per spec).  We wrap in a Cursor to feed
        // ash::util::read_spv which expects a Read.
        let words = ash::util::read_spv(&mut std::io::Cursor::new(spirv))
            .map_err(|e| BackendError::msg(format!("read_spv({:?}): {e}", id.entry_point())))?;
        let sm_info = vk::ShaderModuleCreateInfo::default().code(&words);
        let shader_module = unsafe { self.device.device.create_shader_module(&sm_info, None) }
            .map_err(vk_err("vkCreateShaderModule"))
            .inspect_err(|_| unsafe {
                self.device
                    .device
                    .destroy_pipeline_layout(pipeline_layout, None);
                self.device
                    .device
                    .destroy_descriptor_set_layout(descriptor_set_layout, None);
            })?;

        let entry_point_c = std::ffi::CString::new(id.entry_point())
            .expect("entry-point name has no NULs by construction");
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(entry_point_c.as_c_str());
        let pipeline_info = [vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(pipeline_layout)];

        // Safety: pipeline_info outlives this call.
        let pipelines = unsafe {
            self.device.device.create_compute_pipelines(
                vk::PipelineCache::null(),
                &pipeline_info,
                None,
            )
        };
        let pipeline = match pipelines {
            Ok(mut v) => v.remove(0),
            Err((_partial, code)) => {
                unsafe {
                    self.device
                        .device
                        .destroy_shader_module(shader_module, None);
                    self.device
                        .device
                        .destroy_pipeline_layout(pipeline_layout, None);
                    self.device
                        .device
                        .destroy_descriptor_set_layout(descriptor_set_layout, None);
                }
                return Err(BackendError::msg(format!(
                    "vkCreateComputePipelines for {} failed: {code:?}",
                    id.entry_point()
                )));
            }
        };

        Ok(CompiledKernel {
            descriptor_set_layout,
            pipeline_layout,
            pipeline,
            shader_module,
        })
    }

    fn destroy_one(&self, c: &CompiledKernel) {
        // Safety: handles owned by us and not in use (race-loser path; we
        // never submitted a command buffer using these handles).
        unsafe {
            self.device.device.destroy_pipeline(c.pipeline, None);
            self.device
                .device
                .destroy_pipeline_layout(c.pipeline_layout, None);
            self.device
                .device
                .destroy_descriptor_set_layout(c.descriptor_set_layout, None);
            self.device
                .device
                .destroy_shader_module(c.shader_module, None);
        }
    }
}

impl Drop for PipelineCache {
    fn drop(&mut self) {
        // Take all kernels out of the OnceLocks first so we don't hold a
        // borrow into self.slots while calling self.destroy_one.
        let kernels: Vec<CompiledKernel> = self
            .slots
            .iter_mut()
            .filter_map(std::sync::OnceLock::take)
            .collect();
        for c in &kernels {
            self.destroy_one(c);
        }
    }
}

/// Public accessor returned to the recorder; carries enough handles for
/// it to bind and dispatch.  Borrowed for the lifetime of `&PipelineCache`.
pub(super) struct PipelineHandles<'a> {
    pub(super) pipeline: vk::Pipeline,
    pub(super) layout: vk::PipelineLayout,
    pub(super) descriptor_set_layout: vk::DescriptorSetLayout,
    pub(super) n_storage_buffers: u32,
    _life: std::marker::PhantomData<&'a ()>,
}

impl PipelineCache {
    /// Get pipeline handles for `id`, lazily compiling on first call.
    pub(super) fn handles(&self, id: KernelId) -> Result<PipelineHandles<'_>> {
        let c = self.get(id)?;
        Ok(PipelineHandles {
            pipeline: c.pipeline,
            layout: c.pipeline_layout,
            descriptor_set_layout: c.descriptor_set_layout,
            n_storage_buffers: id.n_storage_buffers(),
            _life: std::marker::PhantomData,
        })
    }
}
