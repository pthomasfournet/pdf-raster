// Phase 5: zigzag inverse + dequant + 8×8 IDCT (LLM fixed-point) + JFIF YCbCr→RGB.
//
// Attribution: 1-D IDCT structure adapted from CESNET/GPUJPEG's gpujpeg_idct.cu
// (BSD-2-Clause, Copyright © 2011 CESNET z.s.p.o. and contributors).
// YCbCr→RGB uses T.871/JFIF full-range equations (round-half-away-from-zero).
//
// Dispatch: one block (8 × 8 × 3 threads) per 8×8 JPEG block.
// threadIdx.x = col (0..7), threadIdx.y = row (0..7), threadIdx.z = component.
// blockIdx.x = block column, blockIdx.y = block row.
//
// Mirror of idct_color.slang — keep the two in sync.

#include <stdint.h>

// ── Zigzag → natural index table ─────────────────────────────────────────────
__constant__ int zigzag_to_natural[64] = {
     0,  1,  8, 16,  9,  2,  3, 10,
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63
};

// ── LLM fixed-point constants (scaled by 2^13) ───────────────────────────────
#define FRAC_BITS 13
#define C1    11363   // cos(pi/16)  * sqrt(2) * 8192
#define C3     9633   // cos(3pi/16) * sqrt(2) * 8192
#define C6     4433   // cos(6pi/16) * sqrt(2) * 8192
#define SQRT2 11585   // sqrt(2)                * 8192

// Round-half-away-from-zero right shift (T.871).
__device__ __forceinline__ int rshift(int x, int s) {
    int bias = 1 << (s - 1);
    return (x + (x >= 0 ? bias : -bias)) >> s;
}

// Apply one 1-D 8-point LLM IDCT to v[0..8] in place.
__device__ void idct_1d(int v[8]) {
    int t0 = v[0]; int t1 = v[4];
    int t2 = v[2]; int t3 = v[6];
    int t4 = v[1]; int t5 = v[5];
    int t6 = v[3]; int t7 = v[7];

    // Even path
    int p0 = t0 + t1;
    int p1 = t0 - t1;
    int p2 = rshift(C6 * t2 - SQRT2 * t3, FRAC_BITS);
    int p3 = rshift(C6 * t3 + SQRT2 * t2, FRAC_BITS);

    int e0 = p0 + p3; int e1 = p1 + p2;
    int e2 = p1 - p2; int e3 = p0 - p3;

    // Odd path
    int q0 = rshift(C1 * t4 + C3 * t6, FRAC_BITS);
    int q1 = rshift(C3 * t4 - C1 * t6, FRAC_BITS);
    int q2 = rshift(C1 * t5 + C3 * t7, FRAC_BITS);
    int q3 = rshift(C3 * t5 - C1 * t7, FRAC_BITS);

    int o0 = q0 + q2; int o1 = q0 - q2;
    int o2 = rshift(SQRT2 * (q1 - q3), FRAC_BITS);
    int o3 = q1 + q3;

    v[0] = e0 + o3; v[1] = e1 + o2;
    v[2] = e2 + o1; v[3] = e3 + o0;
    v[4] = e3 - o0; v[5] = e2 - o1;
    v[6] = e1 - o2; v[7] = e0 - o3;
}

__device__ __forceinline__ int clamp_byte(int x) {
    return max(0, min(255, x));
}

// T.871 full-range YCbCr → RGB.
__device__ __forceinline__ void ycbcr_to_rgb(int Y, int Cb, int Cr,
                                              int *r, int *g, int *b) {
    int cb = Cb - 128; int cr = Cr - 128;
    *r = clamp_byte(rshift((Y << 8) + 359 * cr,            8));
    *g = clamp_byte(rshift((Y << 8) - 88 * cb - 183 * cr,  8));
    *b = clamp_byte(rshift((Y << 8) + 454 * cb,             8));
}

__device__ __forceinline__ uint32_t pack_rgba8(int r, int g, int b) {
    return (uint32_t)r | ((uint32_t)g << 8) | ((uint32_t)b << 16) | 0xFF000000u;
}

// Per-block groupshared scratch: scratch[component][row][col].
// Each block has 8×8×3 = 192 threads and 3×8×8 = 192 ints.
__shared__ int scratch[3][8][8];

extern "C" __global__ void idct_dequant_colour(
    const int * __restrict__ coefficients,   // zigzag-order DCT coefficients
    const int * __restrict__ qtables,        // quantisation tables, natural order
    const int * __restrict__ dc_values,      // absolute DC per block
          uint32_t * __restrict__ pixels_rgba, // RGBA8 output, row-major
    uint32_t width,
    uint32_t height,
    uint32_t num_components,
    uint32_t blocks_wide,
    uint32_t blocks_high,
    uint32_t num_qtables
) {
    const uint32_t col  = threadIdx.x;   // 0..7 within block
    const uint32_t row  = threadIdx.y;   // 0..7 within block
    const uint32_t comp = threadIdx.z;   // component index

    if (comp >= num_components) return;

    const uint32_t bx = blockIdx.x;  // block column
    const uint32_t by = blockIdx.y;  // block row

    if (bx >= blocks_wide || by >= blocks_high) return;

    // Flat block index within this component's grid.
    const uint32_t block_idx = comp * blocks_wide * blocks_high + by * blocks_wide + bx;

    // QT selector: 0 for luma, 1 (clamped) for chroma.
    uint32_t qt_sel = (comp == 0u) ? 0u : 1u;
    if (qt_sel >= num_qtables) qt_sel = num_qtables - 1u;
    const uint32_t qt_base   = qt_sel * 64u;
    const uint32_t coef_base = block_idx * 64u;

    // Step 1: dequantise + inverse-zigzag into shared scratch.
    const uint32_t zz_pos  = row * 8u + col;
    int coef = coefficients[coef_base + zz_pos];
    // DC override: use pre-resolved absolute DC from the host.
    if (zz_pos == 0u) coef = dc_values[block_idx];
    const int qval     = qtables[qt_base + (uint32_t)zigzag_to_natural[zz_pos]];
    const int dequanted = coef * qval;

    const uint32_t nat_pos = (uint32_t)zigzag_to_natural[zz_pos];
    scratch[comp][nat_pos / 8u][nat_pos & 7u] = dequanted;

    __syncthreads();

    // Step 2: row IDCT.
    {
        int v[8];
        for (int k = 0; k < 8; k++) v[k] = scratch[comp][row][k];
        idct_1d(v);
        for (int k = 0; k < 8; k++) scratch[comp][row][k] = v[k];
    }

    __syncthreads();

    // Step 3: column IDCT + level shift.
    {
        int v[8];
        for (int k = 0; k < 8; k++) v[k] = scratch[comp][k][col];
        idct_1d(v);
        // Two IDCT passes each contribute sqrt(8); combined >> 6 collapses to 1.
        for (int k = 0; k < 8; k++)
            scratch[comp][k][col] = clamp_byte(rshift(v[k], 6) + 128);
    }

    __syncthreads();

    // Step 4: colour conversion + output (comp 0 only).
    if (comp != 0u) return;

    const uint32_t px = bx * 8u + col;
    const uint32_t py = by * 8u + row;
    if (px >= width || py >= height) return;

    uint32_t rgba;
    if (num_components == 1u) {
        int luma = scratch[0][row][col];
        rgba = pack_rgba8(luma, luma, luma);
    } else {
        int Y = scratch[0][row][col];
        int Cb = scratch[1][row][col];
        int Cr = scratch[2][row][col];
        int r, g, b;
        ycbcr_to_rgb(Y, Cb, Cr, &r, &g, &b);
        rgba = pack_rgba8(r, g, b);
    }
    pixels_rgba[py * width + px] = rgba;
}
