//! Parity tests: same kernel via CUDA (`.cu`-via-`nvcc`) and Vulkan
//! (`.slang`-via-`slangc`) must agree to ≤ 1 LSB per channel.
//!
//! Each test composes a small representative input, runs it through
//! both backends, and asserts byte-equal output (or ≤ 1 LSB delta for
//! floating-point kernels).
//!
//! Gated on the `vulkan` feature.  CUDA-side tests are additionally
//! gated on `gpu-validation` so machines without a CUDA device still
//! run the Vulkan side and validate against a CPU reference.
//!
//! Run with:
//!   `cargo test -p gpu --features vulkan --test cu_vs_slang_parity`
//!   `cargo test -p gpu --features 'vulkan,gpu-validation' --test cu_vs_slang_parity`

#![cfg(feature = "vulkan")]
#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::similar_names,
    clippy::too_many_arguments,
    clippy::unreadable_literal,
    reason = "test fixture: u32 LCG constants, paired (dx, dy) / (dx_rel, dy_rel) coordinates, \
              bounded `i & 0xff` masks, 8-10 arg CPU-reference helpers, and u32→f32 sample \
              indices intentionally mirror the kernel-side math byte-for-byte"
)]

use rasterrocket_gpu::backend::GpuBackend;
use rasterrocket_gpu::backend::params::{
    AaFillParams, BlitParams, CompositeParams, IccClutParams, SoftMaskParams,
};
use rasterrocket_gpu::backend::vulkan::VulkanBackend;

/// Run `composite_rgba8` on a representative input via `VulkanBackend`,
/// returning the result as a `Vec<u8>`.
fn run_composite_vulkan(src: &[u8], dst_in: &[u8]) -> Vec<u8> {
    assert_eq!(src.len(), dst_in.len());
    assert!(src.len().is_multiple_of(4));
    let n_pixels = u32::try_from(src.len() / 4).expect("n_pixels fits u32");

    let backend = VulkanBackend::new().expect("VulkanBackend::new");
    let d_src = backend.alloc_device(src.len()).expect("alloc src");
    let d_dst = backend.alloc_device(dst_in.len()).expect("alloc dst");
    backend.upload_sync(&d_src, src).expect("upload src");
    backend.upload_sync(&d_dst, dst_in).expect("upload dst");

    backend.begin_page().expect("begin_page");
    backend
        .record_composite(CompositeParams {
            src: &d_src,
            dst: &d_dst,
            n_pixels,
        })
        .expect("record_composite");
    let fence = backend.submit_page().expect("submit_page");
    backend.wait_page(fence).expect("wait_page");

    let mut out = vec![0u8; dst_in.len()];
    backend.download_sync(&d_dst, &mut out).expect("download");
    out
}

/// Reference: CPU implementation from `rasterrocket_gpu::composite::composite_rgba8_cpu`.
fn run_composite_cpu(src: &[u8], dst_in: &[u8]) -> Vec<u8> {
    let mut dst = dst_in.to_vec();
    rasterrocket_gpu::composite_rgba8_cpu(src, &mut dst);
    dst
}

#[test]
fn composite_solid_red_over_solid_white() {
    // 256 pixels of (255, 0, 0, 255) source over (255, 255, 255, 255) dst.
    // Result should be solid red.
    let n = 256;
    let mut src = vec![0u8; n * 4];
    let mut dst = vec![0u8; n * 4];
    for i in 0..n {
        src[i * 4] = 255;
        src[i * 4 + 3] = 255;
        dst[i * 4] = 255;
        dst[i * 4 + 1] = 255;
        dst[i * 4 + 2] = 255;
        dst[i * 4 + 3] = 255;
    }
    let cpu = run_composite_cpu(&src, &dst);
    let vk = run_composite_vulkan(&src, &dst);
    assert_eq!(cpu, vk, "Vulkan output diverges from CPU reference");
}

#[test]
fn composite_translucent_blue_over_red() {
    // Half-translucent blue over solid red — exercises the partial-alpha
    // path (a_src in [1, 254]).
    let n = 1024;
    let mut src = vec![0u8; n * 4];
    let mut dst = vec![0u8; n * 4];
    for i in 0..n {
        src[i * 4] = 0;
        src[i * 4 + 1] = 0;
        src[i * 4 + 2] = 255;
        src[i * 4 + 3] = 128;
        dst[i * 4] = 255;
        dst[i * 4 + 3] = 255;
    }
    let cpu = run_composite_cpu(&src, &dst);
    let vk = run_composite_vulkan(&src, &dst);
    assert_eq!(cpu, vk, "Vulkan output diverges from CPU reference");
}

#[test]
fn composite_zero_alpha_src_is_noop() {
    // a_src == 0 short-circuits in the kernel; dst must be untouched.
    let n = 64;
    let src = vec![0u8; n * 4]; // all zero (alpha 0)
    let mut dst = vec![0u8; n * 4];
    for i in 0..n {
        dst[i * 4] = (i & 0xff) as u8;
        dst[i * 4 + 1] = ((i >> 8) & 0xff) as u8;
        dst[i * 4 + 2] = 0xab;
        dst[i * 4 + 3] = 0xcd;
    }
    let dst_before = dst.clone();
    let vk = run_composite_vulkan(&src, &dst);
    assert_eq!(vk, dst_before, "alpha=0 src must be a no-op");
}

#[test]
fn composite_random_inputs() {
    // Pseudorandom bytes — exercises the full branch space.
    let n = 4096;
    let mut src = vec![0u8; n * 4];
    let mut dst = vec![0u8; n * 4];
    let mut rng = 0xdeadbeef_u32;
    for b in src.iter_mut().chain(dst.iter_mut()) {
        rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
        *b = (rng >> 24) as u8;
    }
    let cpu = run_composite_cpu(&src, &dst);
    let vk = run_composite_vulkan(&src, &dst);
    assert_eq!(
        cpu, vk,
        "Vulkan output diverges from CPU reference (random)"
    );
}

// ── apply_soft_mask ─────────────────────────────────────────────────

fn run_soft_mask_vulkan(pixels_in: &[u8], mask: &[u8]) -> Vec<u8> {
    assert!(pixels_in.len().is_multiple_of(4));
    assert_eq!(pixels_in.len() / 4, mask.len());
    let n_pixels = u32::try_from(mask.len()).expect("n_pixels fits u32");

    let backend = VulkanBackend::new().expect("VulkanBackend::new");
    let d_pixels = backend.alloc_device(pixels_in.len()).expect("alloc pixels");
    let d_mask = backend.alloc_device(mask.len()).expect("alloc mask");
    backend
        .upload_sync(&d_pixels, pixels_in)
        .expect("upload pixels");
    backend.upload_sync(&d_mask, mask).expect("upload mask");

    backend.begin_page().expect("begin_page");
    backend
        .record_apply_soft_mask(SoftMaskParams {
            pixels: &d_pixels,
            mask: &d_mask,
            n_pixels,
        })
        .expect("record_apply_soft_mask");
    let fence = backend.submit_page().expect("submit_page");
    backend.wait_page(fence).expect("wait_page");

    let mut out = vec![0u8; pixels_in.len()];
    backend
        .download_sync(&d_pixels, &mut out)
        .expect("download");
    out
}

fn run_soft_mask_cpu(pixels_in: &[u8], mask: &[u8]) -> Vec<u8> {
    let mut pixels = pixels_in.to_vec();
    rasterrocket_gpu::apply_soft_mask_cpu(&mut pixels, mask);
    pixels
}

#[test]
fn soft_mask_full_coverage_keeps_alpha() {
    // mask = 255 → alpha unchanged.
    let n = 256;
    let mut pixels = vec![0u8; n * 4];
    let mask = vec![255u8; n];
    for i in 0..n {
        pixels[i * 4 + 3] = ((i * 7) & 0xff) as u8; // varied alpha
    }
    let cpu = run_soft_mask_cpu(&pixels, &mask);
    let vk = run_soft_mask_vulkan(&pixels, &mask);
    assert_eq!(cpu, vk, "soft_mask diverges with mask=255");
}

#[test]
fn soft_mask_zero_zeroes_alpha() {
    // mask = 0 → alpha forced to 0 (mod the rounding).
    let n = 256;
    let mut pixels = vec![0u8; n * 4];
    let mask = vec![0u8; n];
    for i in 0..n {
        pixels[i * 4 + 3] = 255;
    }
    let cpu = run_soft_mask_cpu(&pixels, &mask);
    let vk = run_soft_mask_vulkan(&pixels, &mask);
    assert_eq!(cpu, vk, "soft_mask diverges with mask=0");
}

#[test]
fn soft_mask_random_inputs() {
    let n = 1024;
    let mut pixels = vec![0u8; n * 4];
    let mut mask = vec![0u8; n];
    let mut rng = 0xc0ffee_u32;
    for b in pixels.iter_mut().chain(mask.iter_mut()) {
        rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
        *b = (rng >> 24) as u8;
    }
    let cpu = run_soft_mask_cpu(&pixels, &mask);
    let vk = run_soft_mask_vulkan(&pixels, &mask);
    assert_eq!(cpu, vk, "soft_mask diverges with random inputs");
}

// ── icc_cmyk_clut ───────────────────────────────────────────────────

/// Build a synthetic identity CLUT for `grid_n` so the parity test has
/// a deterministic, hand-derivable input.  The kernel quadrilinear-
/// interpolates over the table; with values C and M and Y stored as the
/// node indices' RGB encoding, the result is a smooth ramp we can
/// compare against the CPU path's output byte-for-byte.
fn build_identity_clut(grid_n: u32) -> Vec<u8> {
    let g = grid_n as usize;
    let mut clut = vec![0u8; g * g * g * g * 3];
    // Layout: index = (k * G^3 + c * G^2 + m * G + y) * 3 (matches Rust baker
    // and CUDA kernel).
    for k in 0..g {
        for c in 0..g {
            for m in 0..g {
                for y in 0..g {
                    let idx = (k * g * g * g + c * g * g + m * g + y) * 3;
                    // R encodes C, G encodes M, B encodes Y; K modulates by 1-K/G.
                    let scale = (g - 1 - k) as u32 * 255 / (g - 1).max(1) as u32;
                    let r = (c as u32 * 255 / (g - 1).max(1) as u32) * scale / 255;
                    let gv = (m as u32 * 255 / (g - 1).max(1) as u32) * scale / 255;
                    let b = (y as u32 * 255 / (g - 1).max(1) as u32) * scale / 255;
                    clut[idx] = r as u8;
                    clut[idx + 1] = gv as u8;
                    clut[idx + 2] = b as u8;
                }
            }
        }
    }
    clut
}

fn run_icc_clut_vulkan(cmyk: &[u8], clut: &[u8]) -> Vec<u8> {
    assert!(cmyk.len().is_multiple_of(4));
    let n_pixels = u32::try_from(cmyk.len() / 4).expect("n_pixels fits u32");

    let backend = VulkanBackend::new().expect("VulkanBackend::new");
    let d_cmyk = backend.alloc_device(cmyk.len()).expect("alloc cmyk");
    let d_rgb = backend.alloc_device(cmyk.len() / 4 * 3).expect("alloc rgb");
    let d_clut = backend.alloc_device(clut.len()).expect("alloc clut");
    backend.upload_sync(&d_cmyk, cmyk).expect("upload cmyk");
    backend.upload_sync(&d_clut, clut).expect("upload clut");

    backend.begin_page().expect("begin_page");
    backend
        .record_icc_clut(IccClutParams {
            cmyk: &d_cmyk,
            rgb: &d_rgb,
            clut: &d_clut,
            n_pixels,
        })
        .expect("record_icc_clut");
    let fence = backend.submit_page().expect("submit_page");
    backend.wait_page(fence).expect("wait_page");

    let mut out = vec![0u8; cmyk.len() / 4 * 3];
    backend.download_sync(&d_rgb, &mut out).expect("download");
    out
}

fn run_icc_clut_cpu(cmyk: &[u8], clut: &[u8], grid_n: u32) -> Vec<u8> {
    rasterrocket_gpu::icc_cmyk_to_rgb_cpu(cmyk, Some((clut, grid_n)))
}

#[test]
fn icc_clut_corners_only() {
    // Pure C, M, Y, K corners — sample the 16 hypercube corners only.
    let grid_n = 17u32;
    let clut = build_identity_clut(grid_n);
    let mut cmyk = Vec::new();
    for c in [0u8, 255] {
        for m in [0u8, 255] {
            for y in [0u8, 255] {
                for k in [0u8, 255] {
                    cmyk.extend_from_slice(&[c, m, y, k]);
                }
            }
        }
    }
    let cpu = run_icc_clut_cpu(&cmyk, &clut, grid_n);
    let vk = run_icc_clut_vulkan(&cmyk, &clut);
    // Floating-point lerp: allow ≤ 1 LSB per channel.
    assert_within_1_lsb(&cpu, &vk, "icc_clut corners");
}

#[test]
fn icc_clut_random_inputs() {
    let grid_n = 17u32;
    let clut = build_identity_clut(grid_n);
    let n = 2048;
    let mut cmyk = vec![0u8; n * 4];
    let mut rng = 0x1234_5678_u32;
    for b in &mut cmyk {
        rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
        *b = (rng >> 24) as u8;
    }
    let cpu = run_icc_clut_cpu(&cmyk, &clut, grid_n);
    let vk = run_icc_clut_vulkan(&cmyk, &clut);
    assert_within_1_lsb(&cpu, &vk, "icc_clut random");
}

// ── aa_fill ─────────────────────────────────────────────────────────

fn run_aa_fill_vulkan(segs: &[f32], width: u32, height: u32, fill_rule: u8) -> Vec<u8> {
    let backend = VulkanBackend::new().expect("VulkanBackend::new");
    let segs_bytes: Vec<u8> = segs.iter().flat_map(|f| f.to_ne_bytes()).collect();
    let n_segs = u32::try_from(segs.len() / 4).expect("n_segs fits u32");
    let n_pixels = (width as usize) * (height as usize);

    let d_segs = backend.alloc_device(segs_bytes.len()).expect("alloc segs");
    let d_cov = backend.alloc_device(n_pixels).expect("alloc coverage");
    backend
        .upload_sync(&d_segs, &segs_bytes)
        .expect("upload segs");

    backend.begin_page().expect("begin_page");
    backend
        .record_aa_fill(AaFillParams {
            segs: &d_segs,
            n_segs,
            coverage: &d_cov,
            width,
            height,
            fill_rule,
        })
        .expect("record_aa_fill");
    let fence = backend.submit_page().expect("submit_page");
    backend.wait_page(fence).expect("wait_page");

    let mut out = vec![0u8; n_pixels];
    backend.download_sync(&d_cov, &mut out).expect("download");
    out
}

#[test]
fn aa_fill_axis_aligned_square() {
    // 8×8 square inside a 16×16 canvas; closed polygon, non-zero winding.
    // Quad from (4,4)→(12,4)→(12,12)→(4,12)→(4,4), expressed as 4 edges.
    // Each segment is [x0, y0, x1, y1].
    let segs: Vec<f32> = vec![
        4.0, 4.0, 12.0, 4.0, // top
        12.0, 4.0, 12.0, 12.0, // right
        12.0, 12.0, 4.0, 12.0, // bottom
        4.0, 12.0, 4.0, 4.0, // left
    ];
    let cpu = rasterrocket_gpu::aa_fill_cpu(&segs, 0.0, 0.0, 16, 16, false);
    let vk = run_aa_fill_vulkan(&segs, 16, 16, 0);
    assert_within_1_lsb(&cpu, &vk, "aa_fill axis-aligned square");
}

#[test]
fn aa_fill_triangle() {
    // Triangle (4,2)→(14,8)→(2,12)→(4,2), non-zero winding.
    let segs: Vec<f32> = vec![
        4.0, 2.0, 14.0, 8.0, 14.0, 8.0, 2.0, 12.0, 2.0, 12.0, 4.0, 2.0,
    ];
    let cpu = rasterrocket_gpu::aa_fill_cpu(&segs, 0.0, 0.0, 16, 16, false);
    let vk = run_aa_fill_vulkan(&segs, 16, 16, 0);
    assert_within_1_lsb(&cpu, &vk, "aa_fill triangle");
}

#[test]
fn aa_fill_degenerate_segment_zero_coverage() {
    // A single degenerate segment (start == end) covers nothing.  We
    // can't pass zero segments because the trait rejects zero-size
    // allocations — degenerate is the closest analogue.
    let segs: Vec<f32> = vec![0.0, 0.0, 0.0, 0.0];
    let vk = run_aa_fill_vulkan(&segs, 8, 8, 0);
    assert!(
        vk.iter().all(|&b| b == 0),
        "degenerate segment must produce zero coverage"
    );
}

#[test]
fn aa_fill_512x512_exceeds_old_1d_limit() {
    // The previous 1D dispatch capped out at maxComputeWorkGroupCount[0]
    // (≥ 65535 per Vulkan spec), so a 256×256 image (65536 pixels) was
    // already over the limit.  This test would have failed the old
    // check_dispatch_size guard; passes under the new 2D dispatch.
    let w = 512u32;
    let h = 512u32;
    let segs: Vec<f32> = vec![
        // Big square inside the canvas, axis-aligned.
        64.0, 64.0, 448.0, 64.0, // top
        448.0, 64.0, 448.0, 448.0, // right
        448.0, 448.0, 64.0, 448.0, // bottom
        64.0, 448.0, 64.0, 64.0, // left
    ];
    let cpu = rasterrocket_gpu::aa_fill_cpu(&segs, 0.0, 0.0, w, h, false);
    let vk = run_aa_fill_vulkan(&segs, w, h, 0);
    assert_within_1_lsb(&cpu, &vk, "aa_fill 512×512 square");
}

// ── blit_image ──────────────────────────────────────────────────────

/// Replicate the kernel's nearest-neighbour sample arithmetic on the CPU
/// (mirrors the test in `crates/gpu/src/blit.rs`).  Returns the sampled
/// pixel index or None for out-of-bounds.
#[expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::suboptimal_flops,
    reason = "test reference: arithmetic intentionally mirrors the kernel's f32 math byte-for-byte"
)]
fn blit_cpu_reference(
    page: &mut [u8],
    page_w: u32,
    page_h: u32,
    src: &[u8],
    src_w: u32,
    src_h: u32,
    src_layout: u32, // 0 = RGB, 1 = Gray
    bbox: [i32; 4],
    page_height_f: f32,
    inv_ctm: [f32; 6],
) {
    for dy in bbox[1].max(0)..bbox[3].min(page_h as i32) {
        for dx in bbox[0].max(0)..bbox[2].min(page_w as i32) {
            let dx_rel = dx as f32 - inv_ctm[4];
            let dy_rel = (page_height_f - dy as f32) - inv_ctm[5];
            let u = inv_ctm[0] * dx_rel + inv_ctm[1] * dy_rel;
            let v = inv_ctm[2] * dx_rel + inv_ctm[3] * dy_rel;
            if !(0.0..=1.0).contains(&u) || !(0.0..=1.0).contains(&v) {
                continue;
            }
            let ix = ((u * src_w as f32) as u32).min(src_w - 1);
            let iy = (((1.0 - v) * src_h as f32) as u32).min(src_h - 1);
            let dst_off = ((dy as u32) * page_w + dx as u32) as usize * 4;
            #[expect(
                clippy::branches_sharing_code,
                reason = "shared trailing alpha=255 mirrors the kernel branch shape; \
                          factoring out the assignment would obscure the per-branch RGB pulls"
            )]
            if src_layout == 0 {
                let src_off = (iy * src_w + ix) as usize * 3;
                page[dst_off] = src[src_off];
                page[dst_off + 1] = src[src_off + 1];
                page[dst_off + 2] = src[src_off + 2];
                page[dst_off + 3] = 255;
            } else {
                let src_off = (iy * src_w + ix) as usize;
                let g = src[src_off];
                page[dst_off] = g;
                page[dst_off + 1] = g;
                page[dst_off + 2] = g;
                page[dst_off + 3] = 255;
            }
        }
    }
}

fn run_blit_vulkan(
    src: &[u8],
    src_w: u32,
    src_h: u32,
    src_layout: u32,
    page_w: u32,
    page_h: u32,
    bbox: [i32; 4],
    inv_ctm: [f32; 6],
) -> Vec<u8> {
    let backend = VulkanBackend::new().expect("VulkanBackend::new");
    let dst_bytes = (page_w * page_h * 4) as usize;
    let d_src = backend.alloc_device(src.len()).expect("alloc src");
    let d_dst = backend.alloc_device(dst_bytes).expect("alloc dst");
    backend.upload_sync(&d_src, src).expect("upload src");
    let zeros = vec![0u8; dst_bytes];
    backend.upload_sync(&d_dst, &zeros).expect("zero dst");

    backend.begin_page().expect("begin_page");
    backend
        .record_blit_image(BlitParams {
            src: &d_src,
            dst: &d_dst,
            src_w,
            src_h,
            src_layout,
            dst_w: page_w,
            dst_h: page_h,
            bbox,
            page_h: page_h as f32,
            inv_ctm,
        })
        .expect("record_blit_image");
    let fence = backend.submit_page().expect("submit_page");
    backend.wait_page(fence).expect("wait_page");

    let mut out = vec![0u8; dst_bytes];
    backend.download_sync(&d_dst, &mut out).expect("download");
    out
}

#[test]
fn blit_identity_rgb() {
    // Identity-CTM blit of a 4×4 RGB image into a 4×4 page buffer.
    // inv_ctm = scale 4 (image coords are normalised to [0,1]) with PDF
    // y-flip baked in: dx_rel/4 = u, dy_rel/4 = v, so u_dx=0.25, v_dy=0.25.
    let src_w = 4u32;
    let src_h = 4u32;
    let page = 4u32;
    let pixels: Vec<u8> = (0..src_w * src_h * 3)
        .map(|i| ((i * 7 + 3) % 251) as u8)
        .collect();

    let inv_ctm = [
        0.25, 0.0, // u from dx
        0.0, 0.25, // v from dy_rel (= page_h - dy)
        0.0, 0.0, // tx, ty
    ];
    let bbox = [0, 0, page as i32, page as i32];

    let mut cpu = vec![0u8; (page * page * 4) as usize];
    blit_cpu_reference(
        &mut cpu,
        page,
        page,
        &pixels,
        src_w,
        src_h,
        0,
        bbox,
        page as f32,
        inv_ctm,
    );
    let vk = run_blit_vulkan(&pixels, src_w, src_h, 0, page, page, bbox, inv_ctm);
    assert_eq!(cpu, vk, "blit identity-CTM diverges");
}

#[test]
fn blit_gray_layout() {
    let src_w = 4u32;
    let src_h = 4u32;
    let page = 4u32;
    let pixels: Vec<u8> = (0..src_w * src_h).map(|i| (i * 17) as u8).collect();
    let inv_ctm = [0.25, 0.0, 0.0, 0.25, 0.0, 0.0];
    let bbox = [0, 0, page as i32, page as i32];

    let mut cpu = vec![0u8; (page * page * 4) as usize];
    blit_cpu_reference(
        &mut cpu,
        page,
        page,
        &pixels,
        src_w,
        src_h,
        1,
        bbox,
        page as f32,
        inv_ctm,
    );
    let vk = run_blit_vulkan(&pixels, src_w, src_h, 1, page, page, bbox, inv_ctm);
    assert_eq!(cpu, vk, "blit gray-layout diverges");
}

fn assert_within_1_lsb(a: &[u8], b: &[u8], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: length mismatch");
    let mut max_delta = 0u8;
    let mut delta_sites = 0;
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        let delta = x.abs_diff(y);
        assert!(
            delta <= 1,
            "{what}: byte {i} differs by {delta} ({x} vs {y}); requires ≤ 1 LSB"
        );
        if delta != 0 {
            delta_sites += 1;
            max_delta = max_delta.max(delta);
        }
    }
    if delta_sites != 0 {
        eprintln!(
            "{what}: {delta_sites}/{} bytes differ by ≤ 1 LSB (max {max_delta})",
            a.len()
        );
    }
}
