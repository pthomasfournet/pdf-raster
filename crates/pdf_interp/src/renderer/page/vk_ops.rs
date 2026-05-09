//! Vulkan-backed kernel dispatch helpers for [`PageRenderer`].
//!
//! Parallel branch to [`super::gpu_ops`] (which targets the CUDA `GpuCtx`).
//! Each helper follows the trait surface idiom:
//! `alloc → upload → begin_page → record_* → submit → wait → download`.
//!
//! Failure modes are reported as `Result<bool, ()>`-shaped: returning `false`
//! lets the caller fall through to the CPU path, log-on-error keeps the
//! renderer's "GPU is opportunistic; CPU is the source of truth" contract.
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

/// Run a closure with a freshly-recorded page on `backend`, downloading the
/// supplied `output` device buffer back to host on success.
///
/// The closure receives `&backend` and is expected to call any `record_*`
/// method exactly once.  After it returns successfully, this helper calls
/// `submit_page` / `wait_page` and downloads `output` into a freshly
/// allocated `Vec<u8>` of length `out_len`.
///
/// All error variants are logged at `warn` level (with the supplied
/// `op_name` prefix) and surfaced as `Err(())` so callers can fall back
/// to CPU.
fn run_one_kernel(
    backend: &VulkanBackend,
    output: &<VulkanBackend as GpuBackend>::DeviceBuffer,
    out_len: usize,
    op_name: &'static str,
    record: impl FnOnce(&VulkanBackend) -> gpu::backend::Result<()>,
) -> Result<Vec<u8>, ()> {
    if let Err(e) = backend.begin_page() {
        log::warn!("{op_name}: vulkan begin_page failed: {e}");
        return Err(());
    }
    if let Err(e) = record(backend) {
        log::warn!("{op_name}: vulkan record_* failed: {e}");
        return Err(());
    }
    let fence = match backend.submit_page() {
        Ok(f) => f,
        Err(e) => {
            log::warn!("{op_name}: vulkan submit_page failed: {e}");
            return Err(());
        }
    };
    if let Err(e) = backend.wait_page(fence) {
        log::warn!("{op_name}: vulkan wait_page failed: {e}");
        return Err(());
    }

    let mut out = vec![0u8; out_len];
    if let Err(e) = backend.download_sync(output, &mut out) {
        log::warn!("{op_name}: vulkan download_sync failed: {e}");
        return Err(());
    }
    Ok(out)
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
/// segment list is empty, the bbox is non-finite, or any Vulkan call
/// fails (warning logged; CPU fill used as fallback).
pub(super) fn try_vk_aa_fill(
    renderer: &mut PageRenderer<'_>,
    backend: &Arc<VulkanBackend>,
    path: &raster::path::Path,
    even_odd: bool,
    pipe: &raster::pipe::PipeState<'_>,
    src: &raster::pipe::PipeSrc<'_>,
) -> bool {
    use gpu::GPU_AA_FILL_THRESHOLD;

    let Some((segs_f32, bbox)) = gpu_fill_segs(renderer, path) else {
        return false;
    };
    if (bbox.w as usize * bbox.h as usize) < GPU_AA_FILL_THRESHOLD {
        return false;
    }

    // Pre-shift segments into coverage-buffer-local coordinates.  The
    // CUDA `aa_fill_gpu` uses kernel args `x_min`/`y_min` for the same
    // shift; the trait surface (`record_aa_fill`) does not expose those
    // params, so the caller bakes the offset in before upload.  See the
    // comment on `crates/gpu/src/backend/cuda/page_recorder.rs` (around
    // `record_aa_fill`) for the rationale.
    #[expect(
        clippy::cast_precision_loss,
        reason = "bbox.x/y are u32 bitmap coords; f32 precision is sub-pixel at realistic DPIs"
    )]
    let (bx, by) = (bbox.x as f32, bbox.y as f32);
    let segs_local: Vec<f32> = segs_f32
        .chunks_exact(4)
        .flat_map(|s| [s[0] - bx, s[1] - by, s[2] - bx, s[3] - by])
        .collect();
    let segs_bytes: &[u8] = bytemuck::cast_slice(&segs_local);

    let n_pixels = bbox.w as usize * bbox.h as usize;
    let n_segs = u32::try_from(segs_local.len() / 4).unwrap_or(u32::MAX);

    let d_segs = match backend.alloc_device(segs_bytes.len()) {
        Ok(b) => b,
        Err(e) => {
            log::warn!("vulkan aa_fill: alloc_device(segs) failed: {e}");
            return false;
        }
    };
    let d_cov = match backend.alloc_device(n_pixels) {
        Ok(b) => b,
        Err(e) => {
            log::warn!("vulkan aa_fill: alloc_device(coverage) failed: {e}");
            return false;
        }
    };
    if let Err(e) = backend.upload_sync(&d_segs, segs_bytes) {
        log::warn!("vulkan aa_fill: upload_sync(segs) failed: {e}");
        return false;
    }

    let Ok(coverage) = run_one_kernel(backend, &d_cov, n_pixels, "vulkan aa_fill", |be| {
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

    let Some((segs_f32, bbox)) = gpu_fill_segs(renderer, path) else {
        return false;
    };
    if (bbox.w as usize * bbox.h as usize) < GPU_TILE_FILL_THRESHOLD {
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
    if records.is_empty() || tile_starts.is_empty() {
        return false;
    }

    let records_bytes: &[u8] = bytemuck::cast_slice(&records);
    let starts_bytes: &[u8] = bytemuck::cast_slice(&tile_starts);
    let counts_bytes: &[u8] = bytemuck::cast_slice(&tile_counts);
    let n_pixels = bbox.w as usize * bbox.h as usize;

    let d_records = match backend.alloc_device(records_bytes.len()) {
        Ok(b) => b,
        Err(e) => {
            log::warn!("vulkan tile_fill: alloc_device(records) failed: {e}");
            return false;
        }
    };
    let d_starts = match backend.alloc_device(starts_bytes.len()) {
        Ok(b) => b,
        Err(e) => {
            log::warn!("vulkan tile_fill: alloc_device(starts) failed: {e}");
            return false;
        }
    };
    let d_counts = match backend.alloc_device(counts_bytes.len()) {
        Ok(b) => b,
        Err(e) => {
            log::warn!("vulkan tile_fill: alloc_device(counts) failed: {e}");
            return false;
        }
    };
    let d_cov = match backend.alloc_device(n_pixels) {
        Ok(b) => b,
        Err(e) => {
            log::warn!("vulkan tile_fill: alloc_device(coverage) failed: {e}");
            return false;
        }
    };
    if let Err(e) = backend.upload_sync(&d_records, records_bytes) {
        log::warn!("vulkan tile_fill: upload_sync(records) failed: {e}");
        return false;
    }
    if let Err(e) = backend.upload_sync(&d_starts, starts_bytes) {
        log::warn!("vulkan tile_fill: upload_sync(starts) failed: {e}");
        return false;
    }
    if let Err(e) = backend.upload_sync(&d_counts, counts_bytes) {
        log::warn!("vulkan tile_fill: upload_sync(counts) failed: {e}");
        return false;
    }

    let Ok(coverage) = run_one_kernel(backend, &d_cov, n_pixels, "vulkan tile_fill", |be| {
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
