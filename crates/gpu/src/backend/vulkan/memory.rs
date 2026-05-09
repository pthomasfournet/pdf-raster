//! Vulkan memory allocator wrapping `gpu-allocator`.
//!
//! `gpu-allocator` (pure Rust, used by `wgpu`) handles slab sub-allocation,
//! memory-type selection, and dedicated-allocation cut-off internally.
//! This module wraps it in the buffer types the `GpuBackend` trait
//! expects: opaque `DeviceBuffer` / `HostBuffer` handles whose `Drop`
//! returns the underlying allocation to the slab.
//!
//! Buffer usage flags are uniform across kernels: every device buffer is
//! a generic storage buffer that can also be a transfer source/dest, and
//! has its device-address queried (used by the future BDA path —
//! today's descriptor-set path doesn't read it).
//!
//! ## Drop and fence interaction
//!
//! `DeviceBuffer::drop` calls `vkDestroyBuffer` and frees the gpu-allocator
//! `Allocation` immediately.  Callers must therefore ensure no submitted
//! command buffer still references the buffer — the trait's per-page
//! `wait_page` is the canonical safe point.  Between `submit_page` and
//! `wait_page`, the caller still holds the `DeviceBuffer` (we hand back
//! a fence, not a buffer), so this is a documentation issue, not a leak.

use std::sync::{Arc, Mutex};

use ash::vk;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{
    Allocation, AllocationCreateDesc, AllocationScheme, Allocator, AllocatorCreateDesc,
};

use crate::backend::{BackendError, Result};

use super::device::DeviceCtx;
use super::error::vk_err;

/// Buffer usage shared by every device buffer in this backend.
///
/// `STORAGE_BUFFER` covers every read/write the compute kernels do.
/// `TRANSFER_SRC` + `TRANSFER_DST` lets `upload_async` and any
/// host-readback path use the same buffer without a re-allocation.
/// `SHADER_DEVICE_ADDRESS` is required for `vkGetBufferDeviceAddress`.
const DEVICE_USAGE: vk::BufferUsageFlags = vk::BufferUsageFlags::from_raw(
    vk::BufferUsageFlags::STORAGE_BUFFER.as_raw()
        | vk::BufferUsageFlags::TRANSFER_SRC.as_raw()
        | vk::BufferUsageFlags::TRANSFER_DST.as_raw()
        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS.as_raw(),
);

/// Host-staging usage — host buffers are pure transfer sources/dests;
/// they aren't bound to descriptors, so no `STORAGE_BUFFER` and no BDA.
const HOST_USAGE: vk::BufferUsageFlags = vk::BufferUsageFlags::from_raw(
    vk::BufferUsageFlags::TRANSFER_SRC.as_raw() | vk::BufferUsageFlags::TRANSFER_DST.as_raw(),
);

/// A device-resident storage buffer.
///
/// Owns the `VkBuffer` and the gpu-allocator `Allocation` that backs it.
/// `Drop` frees both.
pub struct DeviceBuffer {
    pub(super) buffer: vk::Buffer,
    pub(super) size: u64,
    pub(super) device_address: vk::DeviceAddress,
    /// Held inside an `Option` so `Drop` can take ownership and hand the
    /// allocation back to the slab; gpu-allocator's `free` consumes by
    /// value.  Always `Some` between construction and drop.
    allocation: Option<Allocation>,
    parent: Arc<Inner>,
}

impl std::fmt::Debug for DeviceBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceBuffer")
            .field("buffer", &self.buffer)
            .field("size", &self.size)
            .field("device_address", &self.device_address)
            .finish_non_exhaustive()
    }
}

impl DeviceBuffer {
    /// Underlying `VkBuffer` handle.
    #[must_use]
    pub const fn handle(&self) -> vk::Buffer {
        self.buffer
    }
    /// Buffer size in bytes (as requested at allocation time, before
    /// alignment padding).
    #[must_use]
    pub const fn size(&self) -> u64 {
        self.size
    }
    /// Buffer device address — non-zero only when the device opted into
    /// `bufferDeviceAddress`.  Reserved for the future BDA push-constant
    /// path; today's descriptor-set bindings don't read it.
    #[must_use]
    pub const fn device_address(&self) -> vk::DeviceAddress {
        self.device_address
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        let alloc = self.allocation.take().expect("DeviceBuffer dropped twice");
        free_buffer(&self.parent, self.buffer, alloc, "DeviceBuffer");
    }
}

/// A host-visible (CPU-mappable) buffer.
///
/// `HOST_VISIBLE | HOST_COHERENT` so writes from the CPU are immediately
/// visible to the GPU without an explicit `vkFlushMappedMemoryRanges`.
pub struct HostBuffer {
    pub(super) buffer: vk::Buffer,
    pub(super) size: u64,
    /// Persistent map: gpu-allocator returns a stable `*mut u8` for the
    /// allocation's lifetime when we request a CpuToGpu/GpuToCpu mapping.
    mapped_ptr: *mut u8,
    allocation: Option<Allocation>,
    parent: Arc<Inner>,
}

// SAFETY: HostBuffer owns the mapping for its lifetime and gpu-allocator
// holds the underlying VkDeviceMemory; the raw pointer is just a cached
// copy of the persistent map.  Sending across threads is safe as long as
// only one thread accesses the slice at a time, which is enforced by the
// &self / &mut self borrow checker.
unsafe impl Send for HostBuffer {}
unsafe impl Sync for HostBuffer {}

impl std::fmt::Debug for HostBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HostBuffer")
            .field("buffer", &self.buffer)
            .field("size", &self.size)
            .field("mapped_ptr", &self.mapped_ptr)
            .finish_non_exhaustive()
    }
}

impl HostBuffer {
    /// Underlying `VkBuffer` handle.
    #[must_use]
    pub const fn handle(&self) -> vk::Buffer {
        self.buffer
    }
    /// Size in bytes.
    #[must_use]
    pub const fn size(&self) -> u64 {
        self.size
    }
    /// Read-only view of the persistently-mapped contents.
    #[must_use]
    #[expect(
        clippy::cast_possible_truncation,
        reason = "self.size came from `u64::try_from(usize)` at alloc time, so the round-trip back to usize is exact on 64-bit; on 32-bit it can only saturate at allocations > 4 GB which gpu-allocator wouldn't have made"
    )]
    pub const fn as_slice(&self) -> &[u8] {
        // Safety: gpu-allocator returns a valid `*mut u8` that stays
        // mapped for the allocation's lifetime; the slice lifetime is
        // tied to `&self` so the pointer is dereferenceable.
        let len = self.size as usize;
        unsafe { std::slice::from_raw_parts(self.mapped_ptr, len) }
    }
    /// Mutable view of the persistently-mapped contents.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "see as_slice — the cast is exact for any allocation gpu-allocator could have made on this host"
    )]
    pub const fn as_mut_slice(&mut self) -> &mut [u8] {
        let len = self.size as usize;
        // Safety: same as `as_slice`, with `&mut self` enforcing
        // exclusive access at the Rust-borrow level.
        unsafe { std::slice::from_raw_parts_mut(self.mapped_ptr, len) }
    }
}

impl Drop for HostBuffer {
    fn drop(&mut self) {
        let alloc = self.allocation.take().expect("HostBuffer dropped twice");
        free_buffer(&self.parent, self.buffer, alloc, "HostBuffer");
    }
}

/// Free a buffer + allocation pair.  Used by both `DeviceBuffer::drop`
/// and `HostBuffer::drop`; logs but does not propagate the gpu-allocator
/// free error since we're already on the destruction path.
fn free_buffer(parent: &Arc<Inner>, buffer: vk::Buffer, alloc: Allocation, kind: &str) {
    let mut allocator = parent.allocator.lock().expect("allocator mutex poisoned");
    if let Err(e) = allocator.free(alloc) {
        log::warn!("{kind} free returned: {e}");
    }
    drop(allocator);
    // Safety: `buffer` was created via `vkCreateBuffer` on `parent.device`;
    // we have exclusive ownership (the wrapping struct just took it via
    // `Drop::drop`); per the trait contract the caller has waited on any
    // PageFence that referenced it.
    unsafe {
        parent.device.device.destroy_buffer(buffer, None);
    }
}

/// Shared state held by every `DeviceBuffer` / `HostBuffer` so they can
/// reach back to the allocator and the device on Drop.
struct Inner {
    device: Arc<DeviceCtx>,
    /// gpu-allocator is `Send` but not `Sync`; wrap in a `Mutex` for the
    /// allocate/free hot path.  Contention is negligible — we make ~tens
    /// of allocations per page.
    allocator: Mutex<Allocator>,
}

/// Slab sub-allocator front-end.
pub struct SlabAllocator {
    inner: Arc<Inner>,
}

impl SlabAllocator {
    /// Build a `gpu-allocator` `Allocator` for the given device.
    ///
    /// # Errors
    /// Returns `BackendError` if `gpu-allocator` rejects the create
    /// description (typically only on a malformed instance/device pair,
    /// which we don't construct here, so this is unreachable in practice
    /// but propagated cleanly anyway).
    pub(super) fn new(device: Arc<DeviceCtx>) -> Result<Self> {
        // AllocatorDebugSettings is #[non_exhaustive] — build via the
        // `..Default::default()` style with a Default-constructed base
        // we then mutate.
        let mut debug_settings = gpu_allocator::AllocatorDebugSettings::default();
        debug_settings.log_memory_information = cfg!(feature = "gpu-validation");
        debug_settings.log_leaks_on_shutdown = cfg!(feature = "gpu-validation");

        let create_desc = AllocatorCreateDesc {
            instance: device.instance.clone(),
            device: device.device.clone(),
            physical_device: device.phys,
            debug_settings,
            buffer_device_address: true,
            allocation_sizes: gpu_allocator::AllocationSizes::default(),
        };
        let allocator = Allocator::new(&create_desc)
            .map_err(|e| BackendError::msg(format!("gpu_allocator::Allocator::new: {e}")))?;
        Ok(Self {
            inner: Arc::new(Inner {
                device,
                allocator: Mutex::new(allocator),
            }),
        })
    }

    /// Allocate a `DEVICE_LOCAL` storage buffer of `size` bytes.
    pub(super) fn alloc_device(&self, size: usize) -> Result<DeviceBuffer> {
        let (buffer, allocation) = self.create_and_bind(
            size,
            DEVICE_USAGE,
            MemoryLocation::GpuOnly,
            "pdf-raster device buffer",
        )?;
        let info = vk::BufferDeviceAddressInfo::default().buffer(buffer);
        // Safety: `buffer` is a freshly created handle bound to memory.
        let device_address = unsafe { self.inner.device.device.get_buffer_device_address(&info) };
        Ok(DeviceBuffer {
            buffer,
            size: u64::try_from(size).expect("size fits u64"),
            device_address,
            allocation: Some(allocation),
            parent: self.inner.clone(),
        })
    }

    /// Allocate a `HOST_VISIBLE` | `HOST_COHERENT` staging buffer of `size`
    /// bytes.  Persistently mapped — caller can read/write via
    /// [`HostBuffer::as_mut_slice`].
    pub(super) fn alloc_host(&self, size: usize) -> Result<HostBuffer> {
        let (buffer, allocation) = self.create_and_bind(
            size,
            HOST_USAGE,
            MemoryLocation::CpuToGpu,
            "pdf-raster host staging buffer",
        )?;
        // gpu-allocator persistently maps CpuToGpu allocations.  If the
        // pointer is missing, it's a gpu-allocator bug or an OOM that
        // wasn't surfaced — roll back cleanly and report.
        let mapped_ptr = if let Some(nn) = allocation.mapped_ptr() {
            nn.as_ptr().cast::<u8>()
        } else {
            self.rollback_alloc(buffer, allocation, "alloc_host: no mapped pointer");
            return Err(BackendError::msg(
                "host buffer allocation has no mapped pointer (gpu-allocator returned None)",
            ));
        };
        Ok(HostBuffer {
            buffer,
            size: u64::try_from(size).expect("size fits u64"),
            mapped_ptr,
            allocation: Some(allocation),
            parent: self.inner.clone(),
        })
    }

    /// Roll back a partially-constructed allocation: free the
    /// gpu-allocator block (logging if free returns a Result error;
    /// we're already on an error path) and destroy the unbound buffer.
    fn rollback_alloc(&self, buffer: vk::Buffer, allocation: Allocation, ctx: &str) {
        let mut allocator = self
            .inner
            .allocator
            .lock()
            .expect("allocator mutex poisoned");
        if let Err(e) = allocator.free(allocation) {
            log::warn!("rollback ({ctx}): gpu-allocator free returned: {e}");
        }
        drop(allocator);
        // Safety: caller guarantees the buffer was just created via
        // vkCreateBuffer and no other handle exists.
        unsafe {
            self.inner.device.device.destroy_buffer(buffer, None);
        }
    }

    /// Free a device buffer.  Drop runs the actual free; the explicit
    /// method exists to match the trait shape (`CudaBackend::free_device`
    /// is similarly empty).
    #[expect(
        clippy::unused_self,
        reason = "shape-only impl; mirrors CudaBackend::free_device"
    )]
    pub(super) fn free_device(&self, _buf: DeviceBuffer) {}

    /// Free a host buffer.  Drop runs the actual free.
    #[expect(
        clippy::unused_self,
        reason = "shape-only impl; mirrors CudaBackend::free_host_pinned"
    )]
    pub(super) fn free_host(&self, _buf: HostBuffer) {}

    /// Create a `VkBuffer`, allocate memory for it, and bind them
    /// together.  Common to both device and host alloc paths.  On error
    /// in any step, rolls back cleanly so neither resource leaks.
    fn create_and_bind(
        &self,
        size: usize,
        usage: vk::BufferUsageFlags,
        location: MemoryLocation,
        name: &'static str,
    ) -> Result<(vk::Buffer, Allocation)> {
        let create_info = vk::BufferCreateInfo::default()
            .size(u64::try_from(size).expect("size fits u64"))
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        // Safety: create_info outlives this call.
        let buffer = unsafe { self.inner.device.device.create_buffer(&create_info, None) }
            .map_err(vk_err("vkCreateBuffer"))?;

        // Safety: `buffer` is freshly created and unbound.
        let requirements = unsafe {
            self.inner
                .device
                .device
                .get_buffer_memory_requirements(buffer)
        };

        let alloc_desc = AllocationCreateDesc {
            name,
            requirements,
            location,
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        };

        let allocation = {
            let mut allocator = self
                .inner
                .allocator
                .lock()
                .expect("allocator mutex poisoned");
            match allocator.allocate(&alloc_desc) {
                Ok(a) => a,
                Err(e) => {
                    drop(allocator);
                    // Safety: buffer was freshly created above; no other handle exists.
                    unsafe {
                        self.inner.device.device.destroy_buffer(buffer, None);
                    }
                    return Err(BackendError::msg(format!("gpu_allocator::allocate: {e}")));
                }
            }
        };

        // Safety: buffer + allocation belong to the same device; the
        // allocation's memory + offset are valid per gpu-allocator's docs.
        let bind_result = unsafe {
            self.inner.device.device.bind_buffer_memory(
                buffer,
                allocation.memory(),
                allocation.offset(),
            )
        };
        if let Err(code) = bind_result {
            self.rollback_alloc(
                buffer,
                allocation,
                "create_and_bind: vkBindBufferMemory failed",
            );
            return Err(BackendError::msg(format!(
                "vkBindBufferMemory failed: {code:?}"
            )));
        }

        Ok((buffer, allocation))
    }
}
