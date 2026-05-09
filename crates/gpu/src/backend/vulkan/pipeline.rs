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

use std::ffi::CStr;
use std::sync::Arc;

use ash::vk;

use crate::backend::{BackendError, Result};

use super::device::DeviceCtx;
use super::error::vk_err;

/// Identifier for one of the six compute kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
    const ENTRY_POINT: &'static CStr = c"main";

    /// Human-readable kernel name for diagnostics.  Matches the source
    /// filename (and the function name in the .cu / .slang).
    pub(super) const fn label(self) -> &'static str {
        match self {
            Self::Composite => "composite_rgba8",
            Self::ApplySoftMask => "apply_soft_mask",
            Self::AaFill => "aa_fill",
            Self::TileFill => "tile_fill",
            Self::IccClut => "icc_cmyk_clut",
            Self::BlitImage => "blit_image",
        }
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
            // (src, dst_rgba) — inv_ctm and other scalars travel as push constants
            Self::BlitImage => 2,
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
/// dispatch builds the pipeline, the rest reuse it.  The host-side
/// `VkPipelineCache` (at `vk_cache`) accelerates compile time
/// across runs by persisting driver-internal pipeline state to a file.
pub(super) struct PipelineCache {
    device: Arc<DeviceCtx>,
    /// Slots indexed by `KernelId as usize`.  `OnceLock` so the first
    /// caller initialises and the rest read; thread-safe by construction.
    slots: [std::sync::OnceLock<CompiledKernel>; 6],
    /// Driver-side pipeline cache — populated from disk at startup if a
    /// matching file exists, written back at Drop.  Vulkan validates the
    /// cache header (driver UUID etc.) so a cache from a different driver
    /// version is silently ignored.  `vk::PipelineCache::null()` is a
    /// valid passthrough; we never expose this handle directly.
    vk_cache: vk::PipelineCache,
}

/// Filename of the on-disk pipeline-cache blob.  Lives under the user's
/// XDG cache root.  Vulkan's own header tags the device/driver, so a
/// shared filename is safe — mismatched caches are rejected at load.
const CACHE_FILENAME: &str = "vulkan_pipeline_cache.bin";

impl PipelineCache {
    /// Construct the cache, attempting to seed the driver-side
    /// `VkPipelineCache` from the on-disk blob.  A missing or unreadable
    /// file is logged at info level and treated as a cold start.
    pub(super) fn new(device: Arc<DeviceCtx>) -> Result<Arc<Self>> {
        let initial_data = match read_cache_file() {
            Ok(bytes) => bytes,
            Err(e) => {
                log::debug!("vulkan_pipeline_cache.bin not loaded: {e}");
                Vec::new()
            }
        };
        let mut info = vk::PipelineCacheCreateInfo::default();
        if !initial_data.is_empty() {
            info = info.initial_data(&initial_data);
        }
        // Safety: device is live; ash validates the create-info shape.
        let vk_cache = unsafe {
            device
                .device
                .create_pipeline_cache(&info, None)
                .map_err(vk_err("vkCreatePipelineCache"))?
        };
        Ok(Arc::new(Self {
            device,
            slots: Default::default(),
            vk_cache,
        }))
    }

    /// Get (or build, if first call) the compiled kernel for `id`.
    ///
    /// First-dispatch contention: a racing compile is correctness-safe;
    /// the loser's pipeline is destroyed before we return.
    /// TODO: switch to `OnceLock::get_or_try_init` once stable (rust-lang/rust#109737).
    fn get(&self, id: KernelId) -> Result<&CompiledKernel> {
        let slot = &self.slots[id as usize];
        if let Some(c) = slot.get() {
            return Ok(c);
        }
        let compiled = self.compile(id)?;
        match slot.set(compiled) {
            Ok(()) => Ok(slot.get().expect("just set")),
            Err(extra) => {
                self.destroy_one(&extra);
                Ok(slot.get().expect("set by the racing thread"))
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
        // ash::util::read_spv which expects a Read.  The Vec<u32> result
        // is dropped at the end of `compile`; the shader module owns its
        // own copy of the SPIR-V after vkCreateShaderModule succeeds.
        //
        // If read_spv fails (would only happen if our build-baked SPIR-V
        // bytes have a non-multiple-of-4 length, which the SPIR-V spec
        // forbids — so this branch is unreachable in practice), still
        // tear down the layouts we already created.
        let words = ash::util::read_spv(&mut std::io::Cursor::new(spirv)).map_err(|e| {
            // Safety: handles owned by us; nothing else holds them; we're
            // bailing out before they could be used.
            unsafe {
                self.device
                    .device
                    .destroy_pipeline_layout(pipeline_layout, None);
                self.device
                    .device
                    .destroy_descriptor_set_layout(descriptor_set_layout, None);
            }
            BackendError::msg(format!("read_spv({}): {e}", id.label()))
        })?;
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

        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(KernelId::ENTRY_POINT);
        let pipeline_info = [vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(pipeline_layout)];

        // Safety: pipeline_info outlives this call.  Pass our persistent
        // VkPipelineCache so the driver can warm-start compilation from
        // its on-disk blob.
        let pipelines = unsafe {
            self.device
                .device
                .create_compute_pipelines(self.vk_cache, &pipeline_info, None)
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
                    id.label()
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
        // Persist the driver-side pipeline cache to disk before tearing
        // down handles.  Failures are logged but not propagated — we're
        // already on the destruction path and the cache is opportunistic.
        // Safety: vk_cache is owned by us, populated above.
        let data = unsafe { self.device.device.get_pipeline_cache_data(self.vk_cache) };
        match data {
            Ok(bytes) if !bytes.is_empty() => {
                if let Err(e) = write_cache_file(&bytes) {
                    log::debug!("vulkan_pipeline_cache.bin not written: {e}");
                }
            }
            Ok(_) => {} // empty cache, nothing to persist
            Err(e) => log::warn!("vkGetPipelineCacheData failed: {e:?}"),
        }
        // Safety: created via vkCreatePipelineCache in Self::new.
        unsafe {
            self.device
                .device
                .destroy_pipeline_cache(self.vk_cache, None);
        }

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

/// Resolve the absolute path to the on-disk pipeline cache:
/// `$XDG_CACHE_HOME/pdf-raster/<file>` (falling back to `$HOME/.cache/...`).
/// Returns `None` if neither env var is set — pure-batch builds with no
/// home directory just skip the cache.
fn cache_path() -> Option<std::path::PathBuf> {
    let cache_root = std::env::var_os("XDG_CACHE_HOME")
        .map(std::path::PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| std::path::PathBuf::from(h).join(".cache")))?;
    Some(cache_root.join("pdf-raster").join(CACHE_FILENAME))
}

fn read_cache_file() -> std::io::Result<Vec<u8>> {
    let path = cache_path().ok_or_else(|| std::io::Error::other("no $XDG_CACHE_HOME or $HOME"))?;
    std::fs::read(&path)
}

fn write_cache_file(bytes: &[u8]) -> std::io::Result<()> {
    let path = cache_path().ok_or_else(|| std::io::Error::other("no $XDG_CACHE_HOME or $HOME"))?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&path, bytes)
}

/// Pipeline handles needed to record a dispatch.  The Vulkan handles are
/// `Copy` and remain valid for the lifetime of the owning `PipelineCache`,
/// which is tied to the `Arc<PipelineCache>` the recorder holds; no
/// separate borrow is needed here.
#[derive(Clone, Copy)]
pub(super) struct PipelineHandles {
    pub(super) pipeline: vk::Pipeline,
    pub(super) layout: vk::PipelineLayout,
    pub(super) descriptor_set_layout: vk::DescriptorSetLayout,
    pub(super) n_storage_buffers: u32,
}

impl PipelineCache {
    /// Get pipeline handles for `id`, lazily compiling on first call.
    pub(super) fn handles(&self, id: KernelId) -> Result<PipelineHandles> {
        let c = self.get(id)?;
        Ok(PipelineHandles {
            pipeline: c.pipeline,
            layout: c.pipeline_layout,
            descriptor_set_layout: c.descriptor_set_layout,
            n_storage_buffers: id.n_storage_buffers(),
        })
    }
}
