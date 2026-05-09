// GPU supersampled anti-aliasing fill kernel.
// Slang port of aa_fill.cu.
//
// Each output pixel is handled by one workgroup of 64 threads.  Each thread
// evaluates the winding-number test at one jittered sub-pixel sample.
// Coverage is reduced via WaveActiveSum (one op per subgroup) then aggregated
// across subgroups via groupshared memory.  This is subgroup-size-agnostic:
// works for wave32 (NVIDIA / RDNA wave32 / Apple), wave64 (RDNA wave64), and
// Intel's variable SIMD widths.
//
// Inputs and outputs match the CUDA kernel byte-for-byte:
//   segs     : packed f32 pairs [x0,y0,x1,y1] per segment, n_segs segments
//   n_segs   : segment count
//   x_min    : leftmost output pixel X (inclusive, in device pixels)
//   y_min    : topmost output pixel Y (inclusive, in device pixels)
//   width    : output pixel width
//   height   : output pixel height
//   eo       : 1 = even-odd fill rule, 0 = non-zero winding
//   coverage : output array, one byte per pixel (row-major, width × height bytes)

// Quasi-random Halton(2, 3) jitter — 64 samples in [0,1)^2.
// Mirrors `JITTER_X` / `JITTER_Y` in aa_fill.cu and `HALTON3` in
// crates/gpu/src/fill.rs.  Earlier revisions had transcription errors at
// n=17..23 and n=41..47; keep these byte-aligned with the Rust source of truth.
static const float JITTER_X[64] = {
    0.5f, 0.25f, 0.75f, 0.125f, 0.625f, 0.375f, 0.875f, 0.0625f,
    0.5625f, 0.3125f, 0.8125f, 0.1875f, 0.6875f, 0.4375f, 0.9375f, 0.03125f,
    0.53125f, 0.28125f, 0.78125f, 0.15625f, 0.65625f, 0.40625f, 0.90625f, 0.09375f,
    0.59375f, 0.34375f, 0.84375f, 0.21875f, 0.71875f, 0.46875f, 0.96875f, 0.015625f,
    0.515625f, 0.265625f, 0.765625f, 0.140625f, 0.640625f, 0.390625f, 0.890625f, 0.078125f,
    0.578125f, 0.328125f, 0.828125f, 0.203125f, 0.703125f, 0.453125f, 0.953125f, 0.046875f,
    0.546875f, 0.296875f, 0.796875f, 0.171875f, 0.671875f, 0.421875f, 0.921875f, 0.109375f,
    0.609375f, 0.359375f, 0.859375f, 0.234375f, 0.734375f, 0.484375f, 0.984375f, 0.0078125f,
};

static const float JITTER_Y[64] = {
    0.333333f, 0.666667f, 0.111111f, 0.444444f, 0.777778f, 0.222222f, 0.555556f, 0.888889f,
    0.037037f, 0.370370f, 0.703704f, 0.148148f, 0.481481f, 0.814815f, 0.259259f, 0.592593f,
    0.925926f, 0.074074f, 0.407407f, 0.740741f, 0.185185f, 0.518519f, 0.851852f, 0.296296f,
    0.629630f, 0.962963f, 0.012346f, 0.345679f, 0.679012f, 0.123457f, 0.456790f, 0.790123f,
    0.234568f, 0.567901f, 0.901235f, 0.049383f, 0.382716f, 0.716049f, 0.160494f, 0.493827f,
    0.827160f, 0.271605f, 0.604938f, 0.938272f, 0.086420f, 0.419753f, 0.753086f, 0.197531f,
    0.530864f, 0.864198f, 0.308642f, 0.641975f, 0.975309f, 0.024691f, 0.358025f, 0.691358f,
    0.135802f, 0.469136f, 0.802469f, 0.246914f, 0.580247f, 0.913580f, 0.061728f, 0.395062f,
};

// Per-segment winding contribution at sample (px, py).
// Returns +1 (upward crossing right of sample), -1 (downward), or 0.
int winding_contrib(float x0, float y0, float x1, float y1, float px, float py)
{
    if (y0 <= py && py < y1) {
        float t = (py - y0) / (y1 - y0);
        float xi = x0 + t * (x1 - x0);
        return (xi >= px) ? 1 : 0;
    }
    if (y1 <= py && py < y0) {
        float t = (py - y1) / (y0 - y1);
        float xi = x1 + t * (x0 - x1);
        return (xi >= px) ? -1 : 0;
    }
    return 0;
}

// At most 64 / 1 = 64 subgroups per workgroup (smallest possible subgroup is
// 1 lane on hypothetical hardware; realistic minimum is 8 on Intel).  Sized
// generously; unused entries stay zero-init.
groupshared uint sub_sums[64];
groupshared uint sub_count;

[shader("compute")]
[numthreads(64, 1, 1)]
void aa_fill(
    uint3 gid : SV_DispatchThreadID,
    uint3 lid : SV_GroupThreadID,
    uint3 wid : SV_GroupID,
    uniform StructuredBuffer<float> segs,
    uniform uint n_segs,
    uniform float x_min,
    uniform float y_min,
    uniform uint width,
    uniform uint height,
    uniform int eo,
    uniform RWStructuredBuffer<uint8_t> coverage)
{
    uint px_idx = wid.x;
    if (px_idx >= width * height) return;

    uint px_x = px_idx % width;
    uint px_y = px_idx / width;

    // Device-space centre of this output pixel.
    float cx = x_min + float(px_x) + 0.5f;
    float cy = y_min + float(px_y) + 0.5f;

    uint tid = lid.x;
    float sx = cx + JITTER_X[tid] - 0.5f;
    float sy = cy + JITTER_Y[tid] - 0.5f;

    // Winding number for this thread's sample.
    int winding = 0;
    for (uint k = 0; k < n_segs; k++) {
        uint base = k * 4;
        float x0 = segs[base + 0];
        float y0 = segs[base + 1];
        float x1 = segs[base + 2];
        float y1 = segs[base + 3];
        winding += winding_contrib(x0, y0, x1, y1, sx, sy);
    }
    bool inside = (eo != 0) ? ((winding & 1) != 0) : (winding != 0);

    // Per-subgroup count (one op per subgroup, returns same value to every lane).
    uint sub_total = WaveActiveSum(uint(inside));

    // Lane 0 of each subgroup claims a slot and writes its sum.  Reset the
    // counter from thread 0 first.
    if (tid == 0) {
        sub_count = 0;
    }
    GroupMemoryBarrierWithGroupSync();

    if (WaveIsFirstLane()) {
        uint slot;
        InterlockedAdd(sub_count, 1u, slot);
        sub_sums[slot] = sub_total;
    }
    GroupMemoryBarrierWithGroupSync();

    // Thread 0 aggregates and writes the coverage byte.  Total in [0..64];
    // scale to [0..255] with rounding: (total * 255 + 32) >> 6.
    if (tid == 0) {
        uint total = 0;
        for (uint i = 0; i < sub_count; i++) {
            total += sub_sums[i];
        }
        coverage[px_idx] = uint8_t((total * 255u + 32u) >> 6u);
    }
}
