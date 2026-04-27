// GPU supersampled anti-aliasing fill kernel.
//
// Each output pixel is handled by one block of 64 threads (2 warps).
// Each thread evaluates the winding-number test at one jittered sub-pixel
// sample using the segment list for the filled path.
//
// Coverage is reduced via __ballot_sync + __popc (one 32-sample ballot per
// warp), summed across both warps (0..64), then scaled to 0..255 and written.
//
// Inputs
// ------
//   segs     : packed f32 pairs [x0,y0,x1,y1] per segment, n_segs segments
//   n_segs   : segment count
//   x_min    : leftmost output pixel X (inclusive, in device pixels)
//   y_min    : topmost output pixel Y (inclusive, in device pixels)
//   width    : output pixel width
//   height   : output pixel height
//   eo       : 1 = even-odd fill rule, 0 = non-zero winding
//   coverage : output array, one byte per pixel (row-major, width * height bytes)

// Quasi-random Halton(base2, base3) jitter table for 64 samples within [0,1)^2.
// Generated offline: Halton(2) for x, Halton(3) for y.
// Using a compile-time constant avoids a per-launch memory allocation.
__device__ __constant__ float JITTER_X[64] = {
    0.5f, 0.25f, 0.75f, 0.125f, 0.625f, 0.375f, 0.875f, 0.0625f,
    0.5625f, 0.3125f, 0.8125f, 0.1875f, 0.6875f, 0.4375f, 0.9375f, 0.03125f,
    0.53125f, 0.28125f, 0.78125f, 0.15625f, 0.65625f, 0.40625f, 0.90625f, 0.09375f,
    0.59375f, 0.34375f, 0.84375f, 0.21875f, 0.71875f, 0.46875f, 0.96875f, 0.015625f,
    0.515625f, 0.265625f, 0.765625f, 0.140625f, 0.640625f, 0.390625f, 0.890625f, 0.078125f,
    0.578125f, 0.328125f, 0.828125f, 0.203125f, 0.703125f, 0.453125f, 0.953125f, 0.046875f,
    0.546875f, 0.296875f, 0.796875f, 0.171875f, 0.671875f, 0.421875f, 0.921875f, 0.109375f,
    0.609375f, 0.359375f, 0.859375f, 0.234375f, 0.734375f, 0.484375f, 0.984375f, 0.0078125f,
};

__device__ __constant__ float JITTER_Y[64] = {
    0.333333f, 0.666667f, 0.111111f, 0.444444f, 0.777778f, 0.222222f, 0.555556f, 0.888889f,
    0.037037f, 0.370370f, 0.703704f, 0.148148f, 0.481481f, 0.814815f, 0.259259f, 0.592593f,
    0.925926f, 0.012346f, 0.345679f, 0.679012f, 0.123457f, 0.456790f, 0.790123f, 0.234568f,
    0.567901f, 0.901235f, 0.049383f, 0.382716f, 0.716049f, 0.160494f, 0.493827f, 0.827160f,
    0.271605f, 0.604938f, 0.938272f, 0.004115f, 0.337449f, 0.670782f, 0.115226f, 0.448560f,
    0.781893f, 0.226337f, 0.559671f, 0.893004f, 0.041152f, 0.374486f, 0.707819f, 0.152263f,
    0.485597f, 0.818930f, 0.263374f, 0.596708f, 0.930041f, 0.074074f, 0.407407f, 0.740741f,
    0.185185f, 0.518519f, 0.851852f, 0.296296f, 0.629630f, 0.962963f, 0.008230f, 0.341564f,
};

// Compute the non-zero winding number contribution of segment (x0,y0)→(x1,y1)
// for sample point (px, py).
//
// Uses the crossing-number approach adapted for winding: count +1 for
// upward-crossing and -1 for downward-crossing edges at the sample point.
// An "upward" crossing is one where y0 < py <= y1; a "downward" crossing
// is y1 < py <= y0. The x-intersection of the edge at y=py must be >= px
// (cast a ray in the +x direction).
//
// Returns the winding increment for this segment (+1, -1, or 0).
__device__ int winding_contrib(float x0, float y0, float x1, float y1,
                               float px, float py)
{
    // Upward crossing: y0 <= py < y1
    if (y0 <= py && py < y1) {
        // Compute x-intersection at py.
        float t = (py - y0) / (y1 - y0);
        float xi = x0 + t * (x1 - x0);
        return (xi >= px) ? 1 : 0;
    }
    // Downward crossing: y1 <= py < y0
    if (y1 <= py && py < y0) {
        float t = (py - y1) / (y0 - y1);
        float xi = x1 + t * (x0 - x1);
        return (xi >= px) ? -1 : 0;
    }
    return 0;
}

// Test whether sample point (px, py) is inside the filled path using the
// winding number (non-zero) or even-odd rule.
__device__ bool sample_inside(const float* __restrict__ segs, unsigned int n_segs,
                              float px, float py, int eo)
{
    int winding = 0;
    for (unsigned int k = 0; k < n_segs; k++) {
        unsigned int base = k * 4;
        float x0 = segs[base + 0];
        float y0 = segs[base + 1];
        float x1 = segs[base + 2];
        float y1 = segs[base + 3];
        winding += winding_contrib(x0, y0, x1, y1, px, py);
    }
    if (eo) {
        return (winding & 1) != 0;
    }
    return winding != 0;
}

extern "C" __global__ void aa_fill(
    const float* __restrict__ segs,
    unsigned int n_segs,
    float x_min,
    float y_min,
    unsigned int width,
    unsigned int height,
    int eo,
    unsigned char* __restrict__ coverage
) {
    // blockIdx.x = pixel index (row-major)
    unsigned int px_idx = blockIdx.x;
    if (px_idx >= width * height) return;

    unsigned int px_x = px_idx % width;
    unsigned int px_y = px_idx / width;

    // Device-space centre of this output pixel.
    float cx = x_min + (float)px_x + 0.5f;
    float cy = y_min + (float)px_y + 0.5f;

    // Each thread handles one of 64 jittered samples.
    unsigned int tid = threadIdx.x; // 0..63
    unsigned int warp = tid >> 5;   // 0 or 1
    unsigned int lane = tid & 31;   // 0..31

    // Sample sub-pixel offset from pixel centre: shift by [-0.5, +0.5).
    float sx = cx + JITTER_X[tid] - 0.5f;
    float sy = cy + JITTER_Y[tid] - 0.5f;

    bool inside = sample_inside(segs, n_segs, sx, sy, eo);

    // Warp-level ballot: each warp independently counts covered samples.
    unsigned int mask = __ballot_sync(0xFFFFFFFFu, inside);
    int count = __popc(mask);

    // One thread per warp writes its count into shared memory.
    __shared__ int warp_counts[2];
    if (lane == 0) {
        warp_counts[warp] = count;
    }
    __syncthreads();

    // Thread 0 aggregates and writes the final coverage byte.
    if (tid == 0) {
        int total = warp_counts[0] + warp_counts[1]; // 0..64
        // Scale 0..64 → 0..255, rounding to nearest.
        // total * 255 / 64 = total * 255 >> 6.
        coverage[px_idx] = (unsigned char)((total * 255 + 32) >> 6);
    }
}
