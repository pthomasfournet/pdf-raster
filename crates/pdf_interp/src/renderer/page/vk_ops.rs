//! Vulkan-backed kernel dispatch helpers for [`PageRenderer`].
//!
//! Parallel branch to [`super::gpu_ops`] (which targets the CUDA `GpuCtx`).
//! Each helper follows the trait surface idiom:
//! `alloc → upload → begin_page → record_* → submit → wait → download`.
//!
//! Failure modes are reported as `bool`: returning `false` lets the caller
//! fall through to the CPU path, log-on-error keeps the renderer's
//! "GPU is opportunistic; CPU is the source of truth" contract.
//!
//! These helpers are gated on `feature = "vulkan"`; when the feature is off
//! the entire module compiles to nothing.

#![cfg(feature = "vulkan")]

use std::sync::Arc;

use gpu::backend::GpuBackend;
use gpu::backend::params::{AaFillParams, TileFillParams};
use gpu::backend::vulkan::VulkanBackend;

use super::PageRenderer;
use super::gpu_ops::{gpu_coverage_to_bitmap, gpu_fill_segs};

type DeviceBuffer = <VulkanBackend as GpuBackend>::DeviceBuffer;

/// Allocate `size` bytes on `backend`; on failure log a contextualised
/// warning at `warn` and return `None`.
///
/// `op_name` names the high-level operation (e.g. `"vulkan aa_fill"`)
/// and `what` names the buffer (`"segs"`, `"coverage"`, …).  Together
/// they produce a single, scannable log line at every call site.
fn alloc_or_warn(
    backend: &VulkanBackend,
    size: usize,
    op_name: &'static str,
    what: &'static str,
) -> Option<DeviceBuffer> {
    backend
        .alloc_device(size)
        .map_err(|e| log::warn!("{op_name}: alloc_device({what}) failed: {e}"))
        .ok()
}

/// Upload `bytes` into `dst`; on failure log a contextualised warning
/// and return `false`.  See [`alloc_or_warn`] for the naming contract.
fn upload_or_warn(
    backend: &VulkanBackend,
    dst: &DeviceBuffer,
    bytes: &[u8],
    op_name: &'static str,
    what: &'static str,
) -> bool {
    backend
        .upload_sync(dst, bytes)
        .map_err(|e| log::warn!("{op_name}: upload_sync({what}) failed: {e}"))
        .is_ok()
}

/// Map any `gpu::backend::Result<T>` to `Option<T>`, logging the
/// failure with the call's contextual labels.
fn warn_err<T>(r: gpu::backend::Result<T>, op_name: &'static str, step: &'static str) -> Option<T> {
    match r {
        Ok(v) => Some(v),
        Err(e) => {
            log::warn!("{op_name}: vulkan {step} failed: {e}");
            None
        }
    }
}

/// Run a closure with a freshly-recorded page on `backend`, downloading the
/// supplied `output` device buffer back to host on success.
///
/// The closure receives `&backend` and is expected to call any `record_*`
/// method exactly once.  After it returns successfully, this helper calls
/// `submit_page` / `wait_page` and downloads `output` into a freshly
/// allocated `Vec<u8>` of length `out_len`.
///
/// All error variants are logged at `warn` level (with the supplied
/// `op_name` prefix) and surfaced as `None` so callers can fall back to CPU.
fn run_one_kernel(
    backend: &VulkanBackend,
    output: &DeviceBuffer,
    out_len: usize,
    op_name: &'static str,
    record: impl FnOnce(&VulkanBackend) -> gpu::backend::Result<()>,
) -> Option<Vec<u8>> {
    warn_err(backend.begin_page(), op_name, "begin_page")?;
    warn_err(record(backend), op_name, "record_*")?;
    let fence = warn_err(backend.submit_page(), op_name, "submit_page")?;
    warn_err(backend.wait_page(fence), op_name, "wait_page")?;

    let mut out = vec![0u8; out_len];
    warn_err(
        backend.download_sync(output, &mut out),
        op_name,
        "download_sync",
    )?;
    Some(out)
}

/// `bbox.w as usize * bbox.h as usize` with overflow-check; on overflow
/// the helper returns `None` and the caller falls through to CPU.
///
/// Realistic page sizes can't approach `usize::MAX` on 64-bit, but
/// adversarial input or a malformed `/MediaBox` could; pre-checking
/// keeps the kernel arithmetic on a known-good footing.
const fn checked_pixel_count(w: u32, h: u32) -> Option<usize> {
    (w as usize).checked_mul(h as usize)
}

/// Attempt to rasterise `path` with the Vulkan AA fill kernel.
///
/// Mirrors [`super::gpu_ops::try_gpu_aa_fill`] but routes through the
/// Vulkan trait surface.  The Vulkan recorder's AA-fill path expects segs
/// already in coverage-buffer-local coordinates, so this helper subtracts
/// `bbox.x` / `bbox.y` from each segment endpoint before upload.
///
/// Returns `true` if the GPU path was taken (caller skips the CPU fill).
/// Returns `false` if the area is below the dispatch threshold, the
/// segment list is empty, the bbox is non-finite, the segment count
/// exceeds `u32::MAX`, or any Vulkan call fails (warning logged; CPU
/// fill used as fallback).
pub(super) fn try_vk_aa_fill(
    renderer: &mut PageRenderer<'_>,
    backend: &Arc<VulkanBackend>,
    path: &raster::path::Path,
    even_odd: bool,
    pipe: &raster::pipe::PipeState<'_>,
    src: &raster::pipe::PipeSrc<'_>,
) -> bool {
    use gpu::GPU_AA_FILL_THRESHOLD;

    const OP: &str = "vulkan aa_fill";

    let Some((mut segs_f32, bbox)) = gpu_fill_segs(renderer, path) else {
        return false;
    };
    let Some(n_pixels) = checked_pixel_count(bbox.w, bbox.h) else {
        return false;
    };
    if n_pixels < GPU_AA_FILL_THRESHOLD {
        return false;
    }

    // Pre-shift segments into coverage-buffer-local coordinates, in place.
    // The CUDA `aa_fill_gpu` uses kernel args `x_min`/`y_min` for the same
    // shift; the trait surface (`record_aa_fill`) does not expose those
    // params, so the caller bakes the offset in before upload.  See the
    // comment in `crates/gpu/src/backend/cuda/page_recorder.rs::record_aa_fill`.
    #[expect(
        clippy::cast_precision_loss,
        reason = "bbox.x/y are u32 bitmap coords; f32 precision is sub-pixel at realistic DPIs"
    )]
    let (bx, by) = (bbox.x as f32, bbox.y as f32);
    for s in segs_f32.chunks_exact_mut(4) {
        s[0] -= bx;
        s[1] -= by;
        s[2] -= bx;
        s[3] -= by;
    }
    let segs_bytes: &[u8] = bytemuck::cast_slice(&segs_f32);

    // Reject segment counts that don't fit the kernel's u32 parameter:
    // a saturating cast would silently feed the kernel the wrong count
    // and corrupt the coverage output.  Realistic PDF paths produce
    // ~10²–10⁴ segments after flattening; ≥ u32::MAX would only arise
    // from adversarial input or a parser bug.
    let Ok(n_segs) = u32::try_from(segs_f32.len() / 4) else {
        log::warn!(
            "{OP}: segment count {} exceeds u32::MAX; falling back to CPU",
            segs_f32.len() / 4
        );
        return false;
    };

    let Some(d_segs) = alloc_or_warn(backend, segs_bytes.len(), OP, "segs") else {
        return false;
    };
    let Some(d_cov) = alloc_or_warn(backend, n_pixels, OP, "coverage") else {
        return false;
    };
    if !upload_or_warn(backend, &d_segs, segs_bytes, OP, "segs") {
        return false;
    }

    let Some(coverage) = run_one_kernel(backend, &d_cov, n_pixels, OP, |be| {
        be.record_aa_fill(AaFillParams {
            segs: &d_segs,
            n_segs,
            coverage: &d_cov,
            width: bbox.w,
            height: bbox.h,
            fill_rule: u8::from(even_odd),
        })
    }) else {
        return false;
    };

    gpu_coverage_to_bitmap(renderer, &coverage, &bbox, pipe, src);
    true
}

/// Attempt to rasterise `path` with the Vulkan tile-parallel analytical fill kernel.
///
/// Mirrors [`super::gpu_ops::try_gpu_tile_fill`] but routes through the
/// Vulkan trait surface.  `build_tile_records` already produces tile-local
/// records when called with `bbox.x` / `bbox.y` as the offset, so no
/// further coordinate shift is needed.
///
/// Returns `true` on success, `false` on any failure (warning logged;
/// caller falls through to AA or CPU fill).
pub(super) fn try_vk_tile_fill(
    renderer: &mut PageRenderer<'_>,
    backend: &Arc<VulkanBackend>,
    path: &raster::path::Path,
    even_odd: bool,
    pipe: &raster::pipe::PipeState<'_>,
    src: &raster::pipe::PipeSrc<'_>,
) -> bool {
    use gpu::{GPU_TILE_FILL_THRESHOLD, build_tile_records};

    const OP: &str = "vulkan tile_fill";

    let Some((segs_f32, bbox)) = gpu_fill_segs(renderer, path) else {
        return false;
    };
    let Some(n_pixels) = checked_pixel_count(bbox.w, bbox.h) else {
        return false;
    };
    if n_pixels < GPU_TILE_FILL_THRESHOLD {
        return false;
    }

    #[expect(
        clippy::cast_precision_loss,
        reason = "bbox.x/y are u32 bitmap coords; f32 precision is sub-pixel at realistic DPIs"
    )]
    let (records, tile_starts, tile_counts, _grid_w) =
        build_tile_records(&segs_f32, bbox.x as f32, bbox.y as f32, bbox.w, bbox.h);

    // `build_tile_records` may legitimately produce zero records (e.g. all
    // segments horizontal); the Vulkan recorder rejects zero-size buffers
    // (`reject_zero_size`), so bail and let the caller fall through.
    // `tile_starts` and `tile_counts` are parallel arrays produced by the
    // same call, so checking either is sufficient.
    if records.is_empty() {
        return false;
    }

    let records_bytes: &[u8] = bytemuck::cast_slice(&records);
    let starts_bytes: &[u8] = bytemuck::cast_slice(&tile_starts);
    let counts_bytes: &[u8] = bytemuck::cast_slice(&tile_counts);

    let Some(d_records) = alloc_or_warn(backend, records_bytes.len(), OP, "records") else {
        return false;
    };
    let Some(d_starts) = alloc_or_warn(backend, starts_bytes.len(), OP, "starts") else {
        return false;
    };
    let Some(d_counts) = alloc_or_warn(backend, counts_bytes.len(), OP, "counts") else {
        return false;
    };
    let Some(d_cov) = alloc_or_warn(backend, n_pixels, OP, "coverage") else {
        return false;
    };
    if !upload_or_warn(backend, &d_records, records_bytes, OP, "records")
        || !upload_or_warn(backend, &d_starts, starts_bytes, OP, "starts")
        || !upload_or_warn(backend, &d_counts, counts_bytes, OP, "counts")
    {
        return false;
    }

    let Some(coverage) = run_one_kernel(backend, &d_cov, n_pixels, OP, |be| {
        be.record_tile_fill(TileFillParams {
            records: &d_records,
            tile_starts: &d_starts,
            tile_counts: &d_counts,
            coverage: &d_cov,
            width: bbox.w,
            height: bbox.h,
            fill_rule: u8::from(even_odd),
        })
    }) else {
        return false;
    };

    gpu_coverage_to_bitmap(renderer, &coverage, &bbox, pipe, src);
    true
}

// ICC CMYK→RGB on Vulkan is intentionally not wired here.  The CUDA dispatcher
// for the matrix path always short-circuits to CPU AVX-512 (PCIe round-trip
// dominates), and the CLUT path's renderer plumbing (`resolve_image` →
// `decode_dct` → `cmyk_raw_to_rgb`) currently threads only `Option<&GpuCtx>`.
// Adding a parallel `Option<&VulkanBackend>` parameter through that chain is
// scoped to a follow-up; under `BackendPolicy::ForceVulkan` `gpu_ctx` is
// `None`, so the CMYK path falls back to the CPU `cmyk_to_rgb_reflectance`
// matrix — a quality regression for CMYK images with embedded ICC profiles
// vs. CUDA, not a correctness bug.  Phase 9-pre-2026-05-07 had the same
// behaviour and it was acceptable then.
