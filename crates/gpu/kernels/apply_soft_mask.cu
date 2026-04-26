// Multiply each RGBA pixel's alpha by the corresponding soft-mask byte.
// pixels: RGBA interleaved, alpha at byte offset 3
// mask: one byte per pixel [0..255]
// n_pixels: total pixel count
extern "C" __global__ void apply_soft_mask(
    unsigned char* __restrict__ pixels,
    const unsigned char* __restrict__ mask,
    unsigned int n_pixels
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_pixels) return;
    unsigned int a = pixels[i * 4 + 3];
    unsigned int m = mask[i];
    pixels[i * 4 + 3] = (unsigned char)((a * m + 127) / 255);
}
