// Image blit kernel — affine CTM source→dest with nearest-neighbour sampling.
// Slang port of blit_image.cu — semantics must match byte-for-byte (the spec
// gate is "≤ 1 LSB per channel per pixel" against the CPU baseline, which
// requires byte-identical sampling — no bilinear).
//
// inv_ctm[0..5] layout (matches CPU formulas in
// crates/pdf_interp/src/renderer/page/mod.rs):
//   inv_ctm[0] =  d * inv_det
//   inv_ctm[1] = -c * inv_det
//   inv_ctm[2] = -b * inv_det
//   inv_ctm[3] =  a * inv_det
//   inv_ctm[4] =  e   (translation x; subtract from dx)
//   inv_ctm[5] =  f   (translation y; subtract from page_h - dy)
//
// All scalar parameters travel via push constants — `inv_ctm` was a
// `StructuredBuffer<float>` in an earlier draft, which would force the
// caller to allocate a 24-byte device buffer per record_blit_image call
// just to pass 6 floats.  Push constants are the right tool here.

[shader("compute")]
[numthreads(16, 16, 1)]
void blit_image(
    uint3 gid : SV_DispatchThreadID,
    uniform StructuredBuffer<uint8_t> src,
    uniform RWStructuredBuffer<uint8_t> dst_rgba,
    uniform int src_w,
    uniform int src_h,
    uniform int src_layout,            // 0 = RGB, 1 = Gray
    uniform int dst_w,
    uniform int dst_h,
    uniform int bx0, uniform int by0,
    uniform int bx1, uniform int by1,
    uniform float page_h,
    uniform float inv_ctm0, uniform float inv_ctm1,
    uniform float inv_ctm2, uniform float inv_ctm3,
    uniform float inv_ctm4, uniform float inv_ctm5)
{
    // SV_DispatchThreadID is the global lane index (group * groupsize + lane).
    int dx = bx0 + int(gid.x);
    int dy = by0 + int(gid.y);

    // Reject pixels outside the bbox AND outside the page.  Page bounds are
    // load-bearing for memory safety — a rotated image partially off-page
    // can have bbox extents past 0 / dst_*; without these checks dst_off
    // would underflow/overflow into OOB memory.
    if (dx < 0 || dy < 0 || dx >= bx1 || dy >= by1 || dx >= dst_w || dy >= dst_h) {
        return;
    }

    // Inverse-map (dx, dy) device-space → (u, v) image-space in [0, 1].
    // The (1 - v) flip mirrors the CPU path: PDF image y-axis points up,
    // page bitmap y-axis points down.
    float dx_rel = float(dx) - inv_ctm4;
    float dy_rel = (page_h - float(dy)) - inv_ctm5;
    float u = inv_ctm0 * dx_rel + inv_ctm1 * dy_rel;
    float v = inv_ctm2 * dx_rel + inv_ctm3 * dy_rel;

    if (u < 0.0f || u > 1.0f || v < 0.0f || v > 1.0f) {
        return;
    }

    // Nearest-neighbour: floor(u * src_w), with (1 - v) flip on v.
    // Symmetric clamps remove the FMA-contraction-near-boundary failure mode.
    int ix = int(u * float(src_w));
    if (ix < 0) ix = 0;
    if (ix >= src_w) ix = src_w - 1;
    int iy = int((1.0f - v) * float(src_h));
    if (iy < 0) iy = 0;
    if (iy >= src_h) iy = src_h - 1;

    int dst_off = (dy * dst_w + dx) * 4;
    if (src_layout == 0) {
        int src_off = (iy * src_w + ix) * 3;
        dst_rgba[dst_off + 0] = src[src_off + 0];
        dst_rgba[dst_off + 1] = src[src_off + 1];
        dst_rgba[dst_off + 2] = src[src_off + 2];
        dst_rgba[dst_off + 3] = uint8_t(255);
    } else {
        int src_off = iy * src_w + ix;
        uint8_t g = src[src_off];
        dst_rgba[dst_off + 0] = g;
        dst_rgba[dst_off + 1] = g;
        dst_rgba[dst_off + 2] = g;
        dst_rgba[dst_off + 3] = uint8_t(255);
    }
}
