// Porter-Duff source-over compositing for RGBA8 pixels.
// Slang port of composite_rgba8.cu — semantics must match byte-for-byte.
//
// Each thread composites one pixel (4 bytes).  No subgroup ops, no shared memory.

[shader("compute")]
[numthreads(256, 1, 1)]
void composite_rgba8(
    uint3 tid : SV_DispatchThreadID,
    uniform StructuredBuffer<uint8_t> src_pixels,
    uniform RWStructuredBuffer<uint8_t> dst_pixels,
    uniform uint n_pixels)
{
    uint i = tid.x;
    if (i >= n_pixels) return;
    uint base = i * 4;

    uint a_src = uint(src_pixels[base + 3]);
    if (a_src == 0) return;

    if (a_src == 255) {
        dst_pixels[base + 0] = src_pixels[base + 0];
        dst_pixels[base + 1] = src_pixels[base + 1];
        dst_pixels[base + 2] = src_pixels[base + 2];
        dst_pixels[base + 3] = uint8_t(255);
        return;
    }

    uint a_dst = uint(dst_pixels[base + 3]);
    uint inv   = 255 - a_src;
    uint a_out = a_src + ((a_dst * inv + 127) / 255);
    if (a_out == 0) return;

    [unroll]
    for (int c = 0; c < 3; c++) {
        uint s = uint(src_pixels[base + c]);
        uint d = uint(dst_pixels[base + c]);
        uint v = (s * a_src + d * a_dst * inv / 255 + a_out / 2) / a_out;
        dst_pixels[base + c] = uint8_t(v);
    }
    dst_pixels[base + 3] = uint8_t(a_out);
}
