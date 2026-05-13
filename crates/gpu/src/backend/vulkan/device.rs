//! Vulkan instance, physical-device selection, and logical-device creation.
//!
//! Picks the first physical device that:
//! - Supports compute on at least one queue family.
//! - Reports `VK_API_VERSION_1_3` or higher (so we can rely on Vulkan 1.3
//!   core for `synchronization2`, `timelineSemaphore`, 8-bit storage,
//!   and buffer-device-address without enabling them as discrete extensions).
//!
//! No surface, no graphics queue — this is a headless compute backend.

use ash::ext::memory_budget;
use ash::vk;
use std::ffi::{CStr, c_char};
use std::sync::{Arc, Mutex};

use crate::backend::{BackendError, Result};

use super::error::vk_err;

/// Owned Vulkan handles.
///
/// Field declaration order matters: Rust drops fields in declaration
/// order, so the `Drop` impl below tears them down explicitly to enforce
/// `device → instance → entry`.
pub(super) struct DeviceCtx {
    pub(super) device: ash::Device,
    /// Index into the device's queue-family list — used at command pool create time.
    pub(super) compute_queue_family: u32,
    /// The compute queue handle, wrapped in a `Mutex` to enforce the
    /// Vulkan "Threading Behavior" external-synchronization requirement
    /// (`vkQueueSubmit` / `vkQueueWaitIdle` / present-class calls on
    /// the same queue must not run concurrently from multiple threads).
    /// The mutex *is* the access path — there is no way to reach the
    /// raw `vk::Queue` without holding the lock.  Both submitters (the
    /// per-page recorder and the transfer context) call
    /// [`Self::with_queue`] for the queue-touching FFI window only.
    compute_queue: Mutex<vk::Queue>,
    /// Cached physical-device handle for memory-property + budget queries.
    pub(super) phys: vk::PhysicalDevice,
    /// Cached memory properties (memoryTypeBits → `MemoryType` lookup).
    pub(super) mem_props: vk::PhysicalDeviceMemoryProperties,
    /// Per-axis maximum workgroup count for `vkCmdDispatch`.  Vulkan's
    /// guaranteed minimum is `(65535, 65535, 65535)`; checked at dispatch
    /// time so the kernel-launch helpers can fail with a useful message
    /// instead of letting the driver return `ERROR_DEVICE_LOST`.
    pub(super) max_workgroup_count: [u32; 3],
    /// Total VRAM advertised by the driver — fixed across the device's lifetime.
    pub(super) vram_total: u64,
    /// Whether `VK_EXT_memory_budget` was successfully enabled (gates
    /// runtime budget queries).
    pub(super) has_memory_budget: bool,
    /// Borrowed PFNs onto the live instance for runtime queries (e.g. memory budget).
    pub(super) instance: ash::Instance,
    /// Vulkan loader; must outlive instance.
    #[expect(
        dead_code,
        reason = "kept alive for the lifetime of `instance`/`device`; ash::Entry holds the dlopen'd libvulkan handle"
    )]
    pub(super) entry: ash::Entry,
}

impl Drop for DeviceCtx {
    fn drop(&mut self) {
        // Safety: every owned handle was created via the matching create_*
        // call above; matching destroy_* calls here, in reverse order, are
        // sound because we have exclusive ownership and the recorder calls
        // `vkDeviceWaitIdle` before the backend drops.
        unsafe {
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

/// Initialise the Vulkan loader, instance, and device.
///
/// Returns an `Arc<DeviceCtx>` so the slab allocator, pipeline cache, and
/// per-page recorder can hold their own clones without lifetime-threading
/// the `ash::Device` reference everywhere.
pub(super) fn init() -> Result<Arc<DeviceCtx>> {
    // Safety: `Entry::load` calls `dlopen("libvulkan.so")`; the returned
    // `Entry` owns the dylib handle and is `Send + Sync` by ash design.
    let entry = unsafe { ash::Entry::load() }
        .map_err(|e| BackendError::msg(format!("Vulkan loader failed: {e}")))?;

    let instance = create_instance(&entry)?;

    let (phys, compute_queue_family) = pick_physical_device(&instance)?;
    let mem_props = unsafe { instance.get_physical_device_memory_properties(phys) };

    // Pull the dispatch group-count limit once.  Spec guarantees ≥ 65535
    // per axis; some software ICDs (lavapipe) report u32::MAX which is
    // fine — we only check against this to fail loudly, not to clamp.
    let dev_props = unsafe { instance.get_physical_device_properties(phys) };
    let max_workgroup_count = dev_props.limits.max_compute_work_group_count;

    let (device, compute_queue, has_memory_budget) =
        create_device(&instance, phys, compute_queue_family)?;

    let vram_total = total_device_local_bytes(&mem_props);

    Ok(Arc::new(DeviceCtx {
        device,
        compute_queue_family,
        compute_queue: Mutex::new(compute_queue),
        phys,
        mem_props,
        max_workgroup_count,
        vram_total,
        has_memory_budget,
        instance,
        entry,
    }))
}

impl DeviceCtx {
    /// Run `f` with the externally-synchronized compute queue.
    ///
    /// The queue handle is only reachable through this method; callers
    /// can't bypass the mutex.  Hold the closure scope as small as
    /// possible — every other submitter (transfer context, recorder)
    /// is blocked while this call runs.
    pub(super) fn with_queue<R>(&self, f: impl FnOnce(vk::Queue) -> R) -> R {
        let queue = self.compute_queue.lock().expect("compute_queue poisoned");
        f(*queue)
    }
}

fn create_instance(entry: &ash::Entry) -> Result<ash::Instance> {
    let app_name = c"rasterrocket";
    let app_info = vk::ApplicationInfo::default()
        .application_name(app_name)
        .application_version(vk::make_api_version(0, 0, 7, 0))
        .engine_name(app_name)
        .engine_version(vk::make_api_version(0, 0, 7, 0))
        .api_version(vk::API_VERSION_1_3);

    // No surface/extension list at instance level — headless compute only.
    let create_info = vk::InstanceCreateInfo::default().application_info(&app_info);

    // Safety: `app_info` and `create_info` outlive this call; the pointers
    // captured by the ash::vk structs stay valid through the call.
    unsafe { entry.create_instance(&create_info, None) }.map_err(vk_err("vkCreateInstance"))
}

fn pick_physical_device(instance: &ash::Instance) -> Result<(vk::PhysicalDevice, u32)> {
    let physicals = unsafe { instance.enumerate_physical_devices() }
        .map_err(vk_err("vkEnumeratePhysicalDevices"))?;
    if physicals.is_empty() {
        return Err(BackendError::msg("no Vulkan physical devices found"));
    }

    // Prefer discrete GPUs over integrated/virtual/CPU ICDs.  Stable sort
    // by descending device-type rank so we walk the best candidate first
    // but still fall through to weaker devices (e.g. lavapipe) if no
    // discrete GPU meets the API-version requirement.
    let mut ranked: Vec<_> = physicals
        .into_iter()
        .map(|p| {
            let props = unsafe { instance.get_physical_device_properties(p) };
            (device_type_rank(props.device_type), p, props)
        })
        .collect();
    ranked.sort_by_key(|(r, _, _)| *r);

    for (_, phys, props) in ranked {
        // Require Vulkan 1.3 core; we depend on synchronization2,
        // timelineSemaphore, BDA, and 8-bit storage as core features.
        // `props.api_version` is the maximum API version this driver
        // supports; comparing against the packed VK_API_VERSION_1_3
        // constant gives a single integer comparison.
        if props.api_version < vk::API_VERSION_1_3 {
            continue;
        }

        let queue_families = unsafe { instance.get_physical_device_queue_family_properties(phys) };
        for (idx, qf) in queue_families.iter().enumerate() {
            if qf.queue_flags.contains(vk::QueueFlags::COMPUTE) {
                let idx_u32 = u32::try_from(idx)
                    .map_err(|_| BackendError::msg("queue family index does not fit in u32"))?;
                return Ok((phys, idx_u32));
            }
        }
    }

    Err(BackendError::msg(
        "no Vulkan 1.3+ device with a compute queue family was found",
    ))
}

fn create_device(
    instance: &ash::Instance,
    phys: vk::PhysicalDevice,
    queue_family: u32,
) -> Result<(ash::Device, vk::Queue, bool)> {
    let available_exts = unsafe { instance.enumerate_device_extension_properties(phys) }
        .map_err(vk_err("vkEnumerateDeviceExtensionProperties"))?;

    let has_memory_budget = available_exts.iter().any(|ext| {
        // Safety: extension_name is a NUL-terminated C string per spec.
        let name = unsafe { CStr::from_ptr(ext.extension_name.as_ptr()) };
        name == memory_budget::NAME
    });

    let mut enabled_exts: Vec<*const c_char> = Vec::new();
    if has_memory_budget {
        enabled_exts.push(memory_budget::NAME.as_ptr());
    }

    let priorities = [1.0_f32];
    let queue_info = [vk::DeviceQueueCreateInfo::default()
        .queue_family_index(queue_family)
        .queue_priorities(&priorities)];

    // Required core features — chained via push_next so the driver sees a
    // single VkPhysicalDeviceFeatures2 → 1.2 → 1.3 list.
    let mut f12 = vk::PhysicalDeviceVulkan12Features::default()
        .buffer_device_address(true)
        .timeline_semaphore(true)
        .storage_buffer8_bit_access(true)
        .uniform_and_storage_buffer8_bit_access(true)
        // The kernels declare `OpCapability Int8` because the Slang
        // sources use `uint8_t` for the byte-typed StructuredBuffers.
        // Without `shader_int8`, vkCreateShaderModule fails with a
        // missing-capability error from the validation layer.
        .shader_int8(true);
    let mut f13 = vk::PhysicalDeviceVulkan13Features::default().synchronization2(true);
    let mut features2 = vk::PhysicalDeviceFeatures2::default()
        .push_next(&mut f12)
        .push_next(&mut f13);

    let device_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(&queue_info)
        .enabled_extension_names(&enabled_exts)
        .push_next(&mut features2);

    let device = unsafe { instance.create_device(phys, &device_info, None) }
        .map_err(vk_err("vkCreateDevice"))?;

    let queue = unsafe { device.get_device_queue(queue_family, 0) };

    Ok((device, queue, has_memory_budget))
}

/// Rank physical-device types so we prefer discrete > integrated > virtual > CPU.
/// Lower number ⇒ higher preference.
const fn device_type_rank(t: vk::PhysicalDeviceType) -> u8 {
    match t {
        vk::PhysicalDeviceType::DISCRETE_GPU => 0,
        vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
        vk::PhysicalDeviceType::VIRTUAL_GPU => 2,
        vk::PhysicalDeviceType::CPU => 3,
        _ => 4,
    }
}

/// Sum of every `DEVICE_LOCAL` heap on the picked physical device.
fn total_device_local_bytes(props: &vk::PhysicalDeviceMemoryProperties) -> u64 {
    let mut total: u64 = 0;
    for i in 0..props.memory_heap_count as usize {
        let heap = props.memory_heaps[i];
        if heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL) {
            total = total.saturating_add(heap.size);
        }
    }
    total
}

/// Query the live `VK_EXT_memory_budget` snapshot if the extension is on,
/// otherwise return `(used, budget) = (0, vram_total)` — a safe fallback
/// that lets the caller derive a usable budget from total VRAM.
#[expect(
    clippy::unnecessary_wraps,
    reason = "Result kept for forward-compat: a future variant calls vkGetPhysicalDeviceMemoryProperties2 indirectly which can fail on lost devices"
)]
pub(super) fn query_memory_budget(ctx: &DeviceCtx) -> Result<(u64, u64)> {
    if !ctx.has_memory_budget {
        return Ok((0, ctx.vram_total));
    }

    let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
    let mut props2 = vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut budget);
    unsafe {
        ctx.instance
            .get_physical_device_memory_properties2(ctx.phys, &mut props2);
    }

    let mut used = 0_u64;
    let mut total_budget = 0_u64;
    for i in 0..ctx.mem_props.memory_heap_count as usize {
        let heap = ctx.mem_props.memory_heaps[i];
        if heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL) {
            used = used.saturating_add(budget.heap_usage[i]);
            total_budget = total_budget.saturating_add(budget.heap_budget[i]);
        }
    }
    Ok((used, total_budget))
}
