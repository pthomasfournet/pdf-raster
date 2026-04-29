//! GPU-accelerated fill helpers for [`PageRenderer`].
//!
//! All functions in this module are gated on `#[cfg(feature = "gpu-aa")]` and
//! take `renderer: &mut PageRenderer<'_>` as their first argument, delegating
//! from thin wrapper methods in `mod.rs`.

#[cfg(feature = "gpu-aa")]
use super::PageRenderer;

/// Clamped pixel-space bounding box for a GPU fill operation.
///
/// All fields are in device-pixel coordinates, clipped to the bitmap bounds.
/// `x` / `y` are the top-left corner; `w` / `h` are the extent.
#[cfg(feature = "gpu-aa")]
pub(super) struct GpuBbox {
    pub(super) x: u32,
    pub(super) y: u32,
    pub(super) w: u32,
    pub(super) h: u32,
}

/// Shared preamble for GPU fill paths: flatten path, compute clamped bbox,
/// convert segments to packed f32.
///
/// Returns `None` if the path is empty, produces no segments, or the bbox
/// is degenerate/non-finite.  Otherwise returns `(segs_f32, bbox)`.
#[cfg(feature = "gpu-aa")]
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "bbox is clamped to [0, bitmap.width/height] before cast; all values are \
              finite, non-negative, and ≤ u32::MAX"
)]
pub(super) fn gpu_fill_segs(
    renderer: &PageRenderer<'_>,
    path: &raster::path::Path,
) -> Option<(Vec<f32>, GpuBbox)> {
    use raster::xpath::XPath;

    use super::DEVICE_MATRIX;

    if path.pts.is_empty() {
        return None;
    }

    let xpath = XPath::new(path, &DEVICE_MATRIX, 0.1, true);
    if xpath.segs.is_empty() {
        return None;
    }

    // Pixel bbox from segment endpoints.
    let mut x_min_f = f64::INFINITY;
    let mut y_min_f = f64::INFINITY;
    let mut x_max_f = f64::NEG_INFINITY;
    let mut y_max_f = f64::NEG_INFINITY;
    for seg in &xpath.segs {
        x_min_f = x_min_f.min(seg.x0).min(seg.x1);
        y_min_f = y_min_f.min(seg.y0).min(seg.y1);
        x_max_f = x_max_f.max(seg.x0).max(seg.x1);
        y_max_f = y_max_f.max(seg.y0).max(seg.y1);
    }
    // Segment coords should always be finite; NaN/Inf would indicate a bug
    // in the path construction or CTM, so we log and bail rather than panic.
    if !x_min_f.is_finite() || !y_min_f.is_finite() || !x_max_f.is_finite() || !y_max_f.is_finite()
    {
        log::warn!(
            "gpu_fill_segs: non-finite segment bbox ({x_min_f}, {y_min_f}, {x_max_f}, {y_max_f}); skipping GPU path"
        );
        return None;
    }

    // Clamp to bitmap dimensions and quantise to integer pixels.
    let bmp_w = f64::from(renderer.bitmap.width);
    let bmp_h = f64::from(renderer.bitmap.height);
    x_min_f = x_min_f.max(0.0).floor();
    y_min_f = y_min_f.max(0.0).floor();
    x_max_f = x_max_f.min(bmp_w).ceil();
    y_max_f = y_max_f.min(bmp_h).ceil();
    if x_max_f <= x_min_f || y_max_f <= y_min_f {
        return None;
    }

    let bbox = GpuBbox {
        x: x_min_f as u32,
        y: y_min_f as u32,
        w: (x_max_f - x_min_f) as u32,
        h: (y_max_f - y_min_f) as u32,
    };

    // Pack segments as flat f32 [x0,y0,x1,y1].  The f64→f32 cast is
    // intentional: GPU kernels operate in f32, and precision loss is
    // sub-pixel at realistic PDF rasterisation DPIs.
    let segs_f32: Vec<f32> = xpath
        .segs
        .iter()
        .flat_map(|seg| [seg.x0 as f32, seg.y0 as f32, seg.x1 as f32, seg.y1 as f32])
        .collect();

    Some((segs_f32, bbox))
}

/// Paint a GPU-produced per-pixel coverage buffer into `renderer.bitmap`.
///
/// `coverage` must be exactly `bbox.w × bbox.h` bytes (one byte per pixel,
/// 0 = outside, 255 = inside), with `bbox` giving the top-left corner and
/// extent in device-pixel coordinates (already clamped to bitmap bounds).
///
/// Scans each row for contiguous non-zero spans, clips them to the active
/// clip region and bitmap bounds, then calls `pipe::render_span`.
#[cfg(feature = "gpu-aa")]
#[expect(
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "all coordinates bounded by bitmap dims which are derived from u32; \
              realistic pages are far below i32::MAX"
)]
pub(super) fn gpu_coverage_to_bitmap(
    renderer: &mut PageRenderer<'_>,
    coverage: &[u8],
    bbox: &GpuBbox,
    pipe: &raster::pipe::PipeState<'_>,
    src: &raster::pipe::PipeSrc<'_>,
) {
    use raster::Pixel;
    use raster::clip::ClipResult;
    use raster::pipe;

    let GpuBbox {
        x: bx,
        y: by,
        w: bw,
        h: bh,
    } = *bbox;

    // The caller guarantees coverage.len() == bw * bh.  A mismatch would
    // mean a bug in the GPU dispatch layer — catch it in debug builds.
    debug_assert_eq!(
        coverage.len(),
        bw as usize * bh as usize,
        "coverage buffer length mismatch: expected {}×{}={}, got {}",
        bw,
        bh,
        bw as usize * bh as usize,
        coverage.len()
    );
    // bbox was clamped to bitmap dims in gpu_fill_segs; bx+bw ≤ bitmap.width,
    // by+bh ≤ bitmap.height.  Assert in debug builds.
    debug_assert!(
        bx.saturating_add(bw) <= renderer.bitmap.width
            && by.saturating_add(bh) <= renderer.bitmap.height,
        "GPU bbox ({bx},{by} {bw}×{bh}) extends outside bitmap ({}×{})",
        renderer.bitmap.width,
        renderer.bitmap.height
    );

    let clip = renderer.gstate.current().clip.clone_shared();
    let bmp_w_i = renderer.bitmap.width as i32;

    for row in 0..bh {
        let y = by + row; // cannot exceed bitmap.height (clamped in gpu_fill_segs)
        let y_i = y as i32;

        let row_start = row as usize * bw as usize;
        let row_cov = &coverage[row_start..row_start + bw as usize];

        // Walk the row, collecting contiguous non-zero spans and emitting them.
        let mut span_start: Option<usize> = None;
        for col in 0..=bw as usize {
            let is_covered = col < bw as usize && row_cov[col] > 0;
            if is_covered {
                span_start.get_or_insert(col);
            } else if let Some(start) = span_start.take() {
                let x0 = bx as i32 + start as i32;
                let x1 = bx as i32 + col as i32 - 1;

                if clip.test_span(x0, x1, y_i) == ClipResult::AllOutside {
                    continue;
                }

                // Clamp to bitmap width and trim the coverage slice to match.
                let sx0 = x0.max(0);
                let sx1 = x1.min(bmp_w_i - 1);
                if sx0 > sx1 {
                    continue;
                }

                // trim_left = sx0 - x0 ≥ 0 (sx0 = x0.max(0)).
                // trim_right = x1 - sx1 ≥ 0 (sx1 = x1.min(bmp_w_i-1)).
                // trim_left + trim_right = (sx0-x0) + (x1-sx1) ≤ (x1-x0) = col-start-1
                // which is < shape_slice.len() = col-start.  So the subtraction is safe.
                let shape_slice = &row_cov[start..col];
                let trim_left = (sx0 - x0) as usize;
                let trim_right = (x1 - sx1) as usize;
                let trimmed_shape = &shape_slice[trim_left..shape_slice.len() - trim_right];
                if trimmed_shape.is_empty() {
                    continue;
                }

                let (row_pixels, alpha_row) = renderer.bitmap.row_and_alpha_mut(y);
                let byte_off = sx0 as usize * <color::Rgb8 as Pixel>::BYTES;
                let byte_end = (sx1 as usize + 1) * <color::Rgb8 as Pixel>::BYTES;
                let alpha_range = sx0 as usize..=sx1 as usize;
                let dst_pixels = &mut row_pixels[byte_off..byte_end];
                let dst_alpha = alpha_row.map(|a| &mut a[alpha_range]);

                pipe::render_span::<color::Rgb8>(
                    pipe,
                    src,
                    dst_pixels,
                    dst_alpha,
                    Some(trimmed_shape),
                    sx0,
                    sx1,
                    y_i,
                );
            }
        }
    }
}

/// Attempt to rasterise `path` with the GPU 64-sample AA kernel.
///
/// Returns `true` if the GPU path was taken (caller skips the CPU fill).
/// Returns `false` if the area is below the dispatch threshold, the segment
/// list is empty, the bbox is non-finite, or the GPU call fails (warning
/// logged; CPU fill used as fallback).
#[cfg(feature = "gpu-aa")]
pub(super) fn try_gpu_aa_fill(
    renderer: &mut PageRenderer<'_>,
    path: &raster::path::Path,
    even_odd: bool,
    pipe: &raster::pipe::PipeState<'_>,
    src: &raster::pipe::PipeSrc<'_>,
    ctx: &gpu::GpuCtx,
) -> bool {
    use gpu::GPU_AA_FILL_THRESHOLD;

    let Some((segs_f32, bbox)) = gpu_fill_segs(renderer, path) else {
        return false;
    };
    if (bbox.w as usize * bbox.h as usize) < GPU_AA_FILL_THRESHOLD {
        return false;
    }

    #[expect(
        clippy::cast_precision_loss,
        reason = "bbox.x/y are u32 bitmap coords; f32 precision is sub-pixel at realistic DPIs"
    )]
    let coverage = match ctx.aa_fill(
        &segs_f32,
        bbox.x as f32,
        bbox.y as f32,
        bbox.w,
        bbox.h,
        even_odd,
    ) {
        Ok(cov) => cov,
        Err(e) => {
            log::warn!("GPU AA fill failed, falling back to CPU: {e}");
            return false;
        }
    };

    gpu_coverage_to_bitmap(renderer, &coverage, &bbox, pipe, src);
    true
}

/// Attempt to rasterise `path` with the GPU tile-parallel analytical fill kernel.
///
/// Returns `true` if the GPU path was taken (caller skips the CPU fill).
/// Returns `false` if the area is below the dispatch threshold, the segment
/// list is empty, the bbox is non-finite, or the GPU call fails (warning
/// logged; caller falls through to AA or CPU fill).
#[cfg(feature = "gpu-aa")]
pub(super) fn try_gpu_tile_fill(
    renderer: &mut PageRenderer<'_>,
    path: &raster::path::Path,
    even_odd: bool,
    pipe: &raster::pipe::PipeState<'_>,
    src: &raster::pipe::PipeSrc<'_>,
    ctx: &gpu::GpuCtx,
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
    let (records, tile_starts, tile_counts, grid_w) =
        build_tile_records(&segs_f32, bbox.x as f32, bbox.y as f32, bbox.w, bbox.h);

    let coverage = match ctx.tile_fill(
        &records,
        &tile_starts,
        &tile_counts,
        grid_w,
        bbox.w,
        bbox.h,
        even_odd,
    ) {
        Ok(cov) => cov,
        Err(e) => {
            log::warn!("GPU tile fill failed, falling back to AA fill: {e}");
            return false;
        }
    };

    gpu_coverage_to_bitmap(renderer, &coverage, &bbox, pipe, src);
    true
}
