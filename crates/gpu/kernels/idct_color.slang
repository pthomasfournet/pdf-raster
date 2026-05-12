// Phase 5: zigzag inverse + dequant + 8×8 IDCT (LLM fixed-point) + JFIF YCbCr→RGB.
//
// Attribution: 1-D IDCT structure adapted from CESNET/GPUJPEG's gpujpeg_idct.cu
// (BSD-2-Clause, Copyright © 2011 CESNET z.s.p.o. and contributors).
// YCbCr→RGB uses T.871/JFIF full-range equations (round-half-away-from-zero).
//
// Dispatch: one workgroup (8 threads × 8 threads) per 8×8 JPEG block.
// `gid.z` is the component index (0 = Y/luma, 1 = Cb, 2 = Cr for YCbCr;
// 0 = luma only for grayscale).
// Pixels are written to `pixels_rgba` as packed RGBA8 (one u32 per pixel),
// using 0xFF for the A channel.

[[vk::binding(0, 0)]] StructuredBuffer<int>    coefficients;  // Phase 4 output, zigzag order
[[vk::binding(1, 0)]] StructuredBuffer<int>    qtables;       // 64 × num_qtables, natural order
[[vk::binding(2, 0)]] StructuredBuffer<int>    dc_values;     // absolute DC per block, one entry/block
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> pixels_rgba;   // RGBA8 output, row-major

[[vk::push_constant]]
cbuffer PushParams {
    uint width;             // image width in pixels
    uint height;            // image height in pixels
    uint num_components;    // 1 (grayscale) or 3 (YCbCr)
    uint blocks_wide;       // ceil(width / 8)
    uint blocks_high;       // ceil(height / 8)
    uint num_qtables;       // number of quantisation tables (1–4)
};

// Zigzag index → natural (row-major 8×8) order.
// zigzag_to_natural[k] gives the (row * 8 + col) index in natural order
// for zigzag position k.
static const int zigzag_to_natural[64] = {
     0,  1,  8, 16,  9,  2,  3, 10,
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63
};

// LLM fixed-point constants (scaled by 2^13).
// Derived from Loeffler, Ligtenberg, Moschytz (1989).
static const int FRAC_BITS = 13;

// cos(k*pi/16) * sqrt(2) * 8192:
static const int C1    = 11363;  // cos(pi/16)  * sqrt(2) * 8192
static const int C3    = 9633;   // cos(3pi/16) * sqrt(2) * 8192
static const int C6    = 4433;   // cos(6pi/16) * sqrt(2) * 8192
static const int SQRT2 = 11585;  // sqrt(2)               * 8192

// Rounding right-shift preserving sign (T.871 round-half-away-from-zero).
int rshift(int x, int s) {
    int bias = 1 << (s - 1);
    return (x + (x >= 0 ? bias : -bias)) >> s;
}

// Apply one 1-D 8-point IDCT to `v[0..8]` in place (LLM algorithm).
// Output is unscaled integers; the caller cancels the 64× scale from
// both passes with >> 6 after the column pass.
void idct_1d(inout int v[8]) {
    // Stage 0: straight-butterfly on DC-row / Nyquist pairs.
    int t0  = v[0];
    int t1  = v[4];
    int t2  = v[2];
    int t3  = v[6];
    int t4  = v[1];
    int t5  = v[5];
    int t6  = v[3];
    int t7  = v[7];

    // Stage 1 (even path — indices 0, 4, 2, 6):
    int p0  = t0 + t1;
    int p1  = t0 - t1;
    int p2  = rshift(C6 * t2 - SQRT2 * t3, FRAC_BITS);
    int p3  = rshift(C6 * t3 + SQRT2 * t2, FRAC_BITS);

    int e0  = p0 + p3;
    int e1  = p1 + p2;
    int e2  = p1 - p2;
    int e3  = p0 - p3;

    // Stage 2 (odd path — indices 1, 5, 3, 7):
    int q0  = rshift(C1 * t4 + C3 * t6, FRAC_BITS);
    int q1  = rshift(C3 * t4 - C1 * t6, FRAC_BITS);
    int q2  = rshift(C1 * t5 + C3 * t7, FRAC_BITS);
    int q3  = rshift(C3 * t5 - C1 * t7, FRAC_BITS);

    int o0  = q0 + q2;
    int o1  = q0 - q2;
    int o2  = rshift(SQRT2 * (q1 - q3), FRAC_BITS);
    int o3  = q1 + q3;

    // Output combination:
    v[0] = e0 + o3;
    v[1] = e1 + o2;
    v[2] = e2 + o1;
    v[3] = e3 + o0;
    v[4] = e3 - o0;
    v[5] = e2 - o1;
    v[6] = e1 - o2;
    v[7] = e0 - o3;
}

// Clamp integer to [0, 255].
int clamp_byte(int x) {
    return clamp(x, 0, 255);
}

// T.871 full-range YCbCr → R, G, B.
// Uses fixed-point with FRAC_BITS=8 fractional bits internally.
// Coefficients from ITU-T T.871 §7:
//   R = Y + 1.402 * (Cr - 128)
//   G = Y - 0.34414 * (Cb - 128) - 0.71414 * (Cr - 128)
//   B = Y + 1.772 * (Cb - 128)
//
// Scaled approximations (×256):
//   1.402 ≈ 359/256     (error < 0.0001)
//   0.34414 ≈ 88/256    (error < 0.001)
//   0.71414 ≈ 183/256   (error < 0.0002)
//   1.772 ≈ 454/256     (error < 0.0001)
int3 ycbcr_to_rgb(int Y, int Cb, int Cr) {
    int cb = Cb - 128;
    int cr = Cr - 128;
    int r = rshift((Y << 8) + 359 * cr,          8);
    int g = rshift((Y << 8) - 88 * cb - 183 * cr, 8);
    int b = rshift((Y << 8) + 454 * cb,           8);
    return int3(clamp_byte(r), clamp_byte(g), clamp_byte(b));
}

// Pack R, G, B, A=255 into a little-endian RGBA8 u32.
uint pack_rgba8(int r, int g, int b) {
    return uint(r) | (uint(g) << 8u) | (uint(b) << 16u) | 0xFF000000u;
}

// Groupshared scratch for 3 component planes, each 8×8 pels.
// Layout: scratch[component][row][col].
groupshared int scratch[3][8][8];

// Main entry point: one workgroup = one 8×8 JPEG block.
// tid.x = col (0..7), tid.y = row (0..7), tid.z = component (0..num_components-1).
[shader("compute")][numthreads(8, 8, 3)]
void idct_dequant_colour(
    uint3 tid : SV_DispatchThreadID,
    uint3 lid : SV_GroupThreadID,
    uint3 gid : SV_GroupID
) {
    uint col       = lid.x;    // 0..7 within block
    uint row       = lid.y;    // 0..7 within block
    uint comp      = lid.z;    // component index

    // Only components 0..num_components-1 are active.
    if (comp >= num_components) {
        return;
    }
    // num_qtables == 0 would underflow in `min(qt_sel, num_qtables - 1u)`.
    // The host validates this before dispatch; guard here as defence-in-depth.
    if (num_qtables == 0u) {
        return;
    }

    // Block coordinate in the component plane.
    // For simplicity this kernel assumes 4:4:4 sampling (no subsampling):
    // all components have the same block grid. Subsampling upscaling
    // is deferred to a follow-on kernel or done by duplicating chroma blocks
    // before invoking this kernel.
    uint bx = gid.x;  // block column
    uint by = gid.y;  // block row

    if (bx >= blocks_wide || by >= blocks_high) {
        return;
    }

    // Flat block index within this component's block grid.
    // Widened to uint64_t: for a 3-component 65535×65535 JPEG, block_idx reaches
    // ~201 M which overflows u32; coef_base = block_idx * 64 reaches ~12.9 G.
    // The host rejects images exceeding the 2^26-block limit before dispatch;
    // the 64-bit arithmetic here is defence-in-depth for kernel correctness.
    uint64_t block_idx = uint64_t(comp) * uint64_t(blocks_wide) * uint64_t(blocks_high)
                       + uint64_t(by) * uint64_t(blocks_wide) + uint64_t(bx);

    // --- Step 1: dequantise and inverse zigzag into scratch[comp][row][col] ---
    // The qtable selector is 0 for Y (luma) and 1 for Cb/Cr (chroma) by JFIF convention;
    // clamp to num_qtables - 1 so a grayscale image with one table doesn't OOB.
    uint qt_sel = (comp == 0u) ? 0u : 1u;
    qt_sel = min(qt_sel, num_qtables - 1u);
    uint qt_base = qt_sel * 64u;
    uint64_t coef_base = block_idx * 64u;

    // Load all 64 zigzag coefficients for this thread's block.
    // We need the whole block to run the column IDCT, so each thread
    // loads one coefficient (8 threads × 8 threads = 64 total).
    uint zz_pos = row * 8u + col;
    int  coef   = coefficients[coef_base + zz_pos];
    int  qval   = qtables[qt_base + uint(zigzag_to_natural[zz_pos])];
    // DC absolute value override: Phase 4 symbols encode DC deltas;
    // `dc_values[block_idx]` is the running absolute DC reconstructed by
    // the host dispatcher. Overwrite slot 0 (zigzag DC position).
    if (zz_pos == 0u) {
        coef = dc_values[block_idx];
    }
    int dequanted = coef * qval;

    // Place into scratch at the natural (row,col) position.
    uint nat_pos = uint(zigzag_to_natural[zz_pos]);
    uint nat_row = nat_pos / 8u;
    uint nat_col = nat_pos & 7u;
    scratch[comp][nat_row][nat_col] = dequanted;

    GroupMemoryBarrierWithGroupSync();

    // --- Step 2: 1-D IDCT on rows (each thread handles one element of one row) ---
    // Thread (row, col) owns row `row` of component `comp`.
    {
        int v[8];
        for (uint k = 0u; k < 8u; k++) {
            v[k] = scratch[comp][row][k];
        }
        idct_1d(v);
        for (uint k = 0u; k < 8u; k++) {
            scratch[comp][row][k] = v[k];
        }
    }

    GroupMemoryBarrierWithGroupSync();

    // --- Step 3: 1-D IDCT on columns ---
    // Thread (row, col) owns column `col` of component `comp`.
    {
        int v[8];
        for (uint k = 0u; k < 8u; k++) {
            v[k] = scratch[comp][k][col];
        }
        idct_1d(v);
        // Scale down: two IDCT passes each introduce a factor of 8 (sqrt(64)),
        // so divide by 64 (>> 6) and add the JPEG level-shift of 128.
        for (uint k = 0u; k < 8u; k++) {
            scratch[comp][k][col] = clamp_byte(rshift(v[k], 6) + 128);
        }
    }

    GroupMemoryBarrierWithGroupSync();

    // --- Step 4: colour conversion + output ---
    // Thread (row, col, comp==0) writes the RGBA8 pixel for this (row, col).
    if (comp != 0u) {
        return;
    }

    uint px = bx * 8u + col;
    uint py = by * 8u + row;
    if (px >= width || py >= height) {
        return;
    }

    uint out_idx = py * width + px;
    uint rgba;
    if (num_components == 1u) {
        int luma = scratch[0][row][col];
        rgba = pack_rgba8(luma, luma, luma);
    } else {
        int Y  = scratch[0][row][col];
        int Cb = scratch[1][row][col];
        int Cr = scratch[2][row][col];
        int3 rgb = ycbcr_to_rgb(Y, Cb, Cr);
        rgba = pack_rgba8(rgb.x, rgb.y, rgb.z);
    }
    pixels_rgba[out_idx] = rgba;
}
