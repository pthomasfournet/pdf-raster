// Porter-Duff source-over compositing for RGBA8 pixels.
// src_pixels: RGBA interleaved (R,G,B,A per pixel)
// dst_pixels: RGBA interleaved (R,G,B,A per pixel)
// n_pixels: total pixel count
extern "C" __global__ void composite_rgba8(
    const unsigned char* __restrict__ src_pixels,
    unsigned char* __restrict__ dst_pixels,
    unsigned int n_pixels
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_pixels) return;
    unsigned int base = i * 4;
    unsigned int a_src = src_pixels[base + 3];
    if (a_src == 0) return;
    if (a_src == 255) {
        dst_pixels[base]     = src_pixels[base];
        dst_pixels[base + 1] = src_pixels[base + 1];
        dst_pixels[base + 2] = src_pixels[base + 2];
        dst_pixels[base + 3] = 255;
        return;
    }
    unsigned int a_dst  = dst_pixels[base + 3];
    unsigned int inv    = 255 - a_src;
    unsigned int a_out  = a_src + ((a_dst * inv + 127) / 255);
    if (a_out == 0) return;
    for (int c = 0; c < 3; c++) {
        unsigned int s = src_pixels[base + c];
        unsigned int d = dst_pixels[base + c];
        dst_pixels[base + c] = (unsigned char)(
            (s * a_src + d * a_dst * inv / 255 + a_out / 2) / a_out
        );
    }
    dst_pixels[base + 3] = (unsigned char)a_out;
}
