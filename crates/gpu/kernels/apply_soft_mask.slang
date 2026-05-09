// Multiply each RGBA pixel's alpha by the corresponding soft-mask byte.
// Slang port of apply_soft_mask.cu — semantics must match byte-for-byte.

[shader("compute")]
[numthreads(256, 1, 1)]
void apply_soft_mask(
    uint3 tid : SV_DispatchThreadID,
    uniform RWStructuredBuffer<uint8_t> pixels,
    uniform StructuredBuffer<uint8_t> mask,
    uniform uint n_pixels)
{
    uint i = tid.x;
    if (i >= n_pixels) return;
    uint a = uint(pixels[i * 4 + 3]);
    uint m = uint(mask[i]);
    pixels[i * 4 + 3] = uint8_t((a * m + 127) / 255);
}
