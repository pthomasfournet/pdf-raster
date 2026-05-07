// Phase 9 task 4 — image blit kernel.
//
// Transforms a cached source image into a destination RGBA8 page
// buffer using an affine CTM.  Each output pixel is one thread; the
// kernel computes the inverse-mapped source coordinate via the
// caller-supplied inverse CTM, samples nearest-neighbour from the
// source, and writes RGBA with alpha=255 for in-bounds samples.
// Out-of-bounds samples leave the destination untouched (the caller
// allocates the page buffer zero-initialised, so untouched pixels
// composite as fully-transparent in the host-side overlay step).
//
// Sampling matches the CPU path in
// crates/pdf_interp/src/renderer/page/mod.rs: nearest-neighbour with
// `floor(u * img_w)` / `floor((1 - v) * img_h)` indexing.  Bilinear
// filtering is intentionally NOT used — the spec acceptance is
// "≤ 1 LSB per channel per pixel" against the CPU baseline, and that
// requires byte-identical sampling.
//
// Inputs
// ------
//   src         : decoded image bytes, layout-dependent stride
//   src_w       : source width in pixels
//   src_h       : source height in pixels
//   src_layout  : 0 = RGB (3 bytes/pixel), 1 = Gray (1 byte/pixel)
//                 (Mask layout is handled on CPU; not dispatched here.)
//   dst_rgba    : destination RGBA8 buffer (4 bytes/pixel, row-major)
//   dst_w       : destination width in pixels (page width)
//   dst_h       : destination height in pixels (page height)
//   bx0, by0    : top-left of the bounding-box subregion to render
//                 (output coords; allows the caller to skip pixels
//                 known to fall outside the image's transformed AABB)
//   bx1, by1    : bottom-right (exclusive)
//   inv_ctm     : 6-float inverse of the user-space → device CTM,
//                 stored as [a, b, c, d, e, f]; given output (dx, dy),
//                 the corresponding image-space (u, v) is:
//                     dy_pdf = page_h - dy
//                     u = (a * (dx - e) + c * (-(dy_pdf - f))) ...
//                 In practice the CPU path computes:
//                     u = ( d * dx_rel - c * dy_rel) * inv_det
//                     v = (-b * dx_rel + a * dy_rel) * inv_det
//                 where det = a*d - b*c, dx_rel = dx - e,
//                 dy_rel = (page_h - dy) - f, and inv_det = 1/det.
//                 We pass the six pre-multiplied coefficients so the
//                 kernel's per-pixel work is two FMAs + one clamp.
//   page_h      : full page height (needed to flip dy to PDF coords)
//
// inv_ctm layout (matches the CPU formulas):
//   inv_ctm[0] =  d * inv_det
//   inv_ctm[1] = -c * inv_det
//   inv_ctm[2] = -b * inv_det
//   inv_ctm[3] =  a * inv_det
//   inv_ctm[4] =  e         (translation x; subtract from dx)
//   inv_ctm[5] =  f         (translation y; subtract from dy_pdf)
//
// So:
//   dx_rel = dx - inv_ctm[4]
//   dy_rel = (page_h - dy) - inv_ctm[5]
//   u = inv_ctm[0] * dx_rel + inv_ctm[1] * dy_rel
//   v = inv_ctm[2] * dx_rel + inv_ctm[3] * dy_rel

extern "C" __global__ void blit_image(
    const unsigned char* __restrict__ src,
    int src_w,
    int src_h,
    int src_layout,
    unsigned char* __restrict__ dst_rgba,
    int dst_w,
    int dst_h,
    int bx0, int by0,
    int bx1, int by1,
    const float* __restrict__ inv_ctm,
    float page_h
) {
    int dx = bx0 + (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int dy = by0 + (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (dx >= bx1 || dy >= by1 || dx >= dst_w || dy >= dst_h) {
        return;
    }

    // Inverse-map (dx, dy) in device space to (u, v) in [0, 1] image
    // space.  The (1 - v) flip below mirrors the CPU path: PDF's
    // image-space y axis points up, the page bitmap's y axis points
    // down.
    float dx_rel = (float)dx - inv_ctm[4];
    float dy_rel = (page_h - (float)dy) - inv_ctm[5];
    float u = inv_ctm[0] * dx_rel + inv_ctm[1] * dy_rel;
    float v = inv_ctm[2] * dx_rel + inv_ctm[3] * dy_rel;

    // Reject samples that fall outside the image's [0, 1]² extent.
    // The caller's bbox is conservative (axis-aligned over a
    // rotated/sheared transform), so a substantial fraction of bbox
    // pixels can land outside the image — leave dst_rgba untouched
    // for those (it stays at the alpha=0 zero-init).
    if (u < 0.0f || u > 1.0f || v < 0.0f || v > 1.0f) {
        return;
    }

    // Nearest-neighbour: floor(u * src_w), with the (1-v) flip for v.
    // `min(..., src_w - 1)` clamps the rare case where u rounds up
    // past the last pixel due to f32 imprecision near u==1.0.
    int ix = (int)(u * (float)src_w);
    if (ix >= src_w) ix = src_w - 1;
    int iy = (int)((1.0f - v) * (float)src_h);
    if (iy >= src_h) iy = src_h - 1;

    int dst_off = (dy * dst_w + dx) * 4;
    if (src_layout == 0) {
        // RGB: 3 bytes per source pixel
        int src_off = (iy * src_w + ix) * 3;
        dst_rgba[dst_off + 0] = src[src_off + 0];
        dst_rgba[dst_off + 1] = src[src_off + 1];
        dst_rgba[dst_off + 2] = src[src_off + 2];
        dst_rgba[dst_off + 3] = 255;
    } else {
        // Gray: 1 byte per source pixel; broadcast to RGB.
        int src_off = iy * src_w + ix;
        unsigned char g = src[src_off];
        dst_rgba[dst_off + 0] = g;
        dst_rgba[dst_off + 1] = g;
        dst_rgba[dst_off + 2] = g;
        dst_rgba[dst_off + 3] = 255;
    }
}
