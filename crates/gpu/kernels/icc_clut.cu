// GPU ICC colour transform kernels.
//
// Two entry points:
//
//   icc_cmyk_matrix — fast path for profiles that reduce to a 3×4 linear matrix.
//     The formula is the standard subtractive CMYK complement:
//       R = (255 - C) * (255 - K) / 255
//       G = (255 - M) * (255 - K) / 255
//       B = (255 - Y) * (255 - K) / 255
//     This is identical to the CPU fallback in image.rs::cmyk_raw_to_rgb_triple.
//
//   icc_cmyk_clut — full 4D CLUT path for A2B0 profiles.
//     Evaluates the CLUT using manual quadrilinear interpolation with __ldg()
//     (routes through L1 texture cache on SM 6.0+).  16 node lookups per pixel,
//     no shared memory, no texture objects — the CLUT fits in L2 after the
//     first wave of blocks and remains resident for the page lifetime.
//
// Input layout:  `n_pixels` × 4 bytes, interleaved CMYK (PDF convention:
//                0 = no ink, 255 = full ink for DeviceCMYK).
// Output layout: `n_pixels` × 3 bytes, interleaved RGB.
//
// Thread/block layout: 256 threads/block, ceil(n_pixels/256) blocks —
// matches the existing composite_rgba8 pattern in composite_rgba8.cu.

// ── Matrix kernel ─────────────────────────────────────────────────────────────

// Convert one CMYK pixel to RGB using the subtractive complement formula.
// PDF DeviceCMYK convention: 0 = no ink, 255 = full ink.
//
// R = (255−C)*(255−K)/255,  G = (255−M)*(255−K)/255,  B = (255−Y)*(255−K)/255
//
// Parameters:
//   cmyk : interleaved input,  4 bytes per pixel
//   rgb  : interleaved output, 3 bytes per pixel
//   n    : total pixel count
extern "C" __global__ void icc_cmyk_matrix(
    const unsigned char* __restrict__ cmyk,
    unsigned char*       __restrict__ rgb,
    unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    unsigned int ci = i * 4;
    unsigned int ri = i * 3;

    unsigned int c = cmyk[ci];
    unsigned int m = cmyk[ci + 1];
    unsigned int y = cmyk[ci + 2];
    unsigned int k = cmyk[ci + 3];

    unsigned int inv_k = 255u - k;

    // Integer multiply-round-divide: (255−ch)*(255−k)/255, rounded to nearest.
    // Adding 127 before dividing by 255 gives unbiased rounding; without it,
    // truncation would systematically underestimate mid-range colours.
    // Maximum numerator: 255*255 + 127 = 65152, safe in u32.
    rgb[ri]     = (unsigned char)(((255u - c) * inv_k + 127u) / 255u);
    rgb[ri + 1] = (unsigned char)(((255u - m) * inv_k + 127u) / 255u);
    rgb[ri + 2] = (unsigned char)(((255u - y) * inv_k + 127u) / 255u);
}

// ── CLUT kernel ───────────────────────────────────────────────────────────────

// Fetch one node from the CLUT using __ldg (L1 texture cache path).
//
// `clut` is a flat u8 array of `grid_n^4 * 3` bytes, ordered:
//   index = (k_idx * G^3 + c_idx * G^2 + m_idx * G + y_idx) * 3
// where G = grid_n (typically 17).
//
// K is the outermost (major) axis to match the ICC/PDF CMYK LUT convention and
// the layout produced by the Rust baking code in pdf_interp/src/resources/icc.rs.
//
// Returns the (R, G, B) triple at (ci, mi, yi, ki).
__device__ __forceinline__ void clut_node(
    const unsigned char* __restrict__ clut,
    int ci, int mi, int yi, int ki,
    int G,
    float* r, float* g, float* b
) {
    // Clamp to [0, G-1] to prevent out-of-bounds CLUT access.
    // FP rounding on normalised inputs and extreme channel values can push floor
    // indices to G, so clamping is a mandatory safety guard, not a soft hint.
    ci = max(0, min(ci, G - 1));
    mi = max(0, min(mi, G - 1));
    yi = max(0, min(yi, G - 1));
    ki = max(0, min(ki, G - 1));

    int idx = (ki * G * G * G + ci * G * G + mi * G + yi) * 3;
    *r = (float)__ldg(&clut[idx]);
    *g = (float)__ldg(&clut[idx + 1]);
    *b = (float)__ldg(&clut[idx + 2]);
}

// Quadrilinear interpolation in a 4D CLUT.
//
// Each input channel is normalised to [0, G-1].  We split each into an integer
// floor index and a fractional weight, then lerp across the 16 corners of the
// 4D hypercube cell.
//
// cmyk  : interleaved input,  4 bytes per pixel (PDF convention)
// rgb   : interleaved output, 3 bytes per pixel
// clut  : flat u8 CLUT, grid_n^4 * 3 bytes
//         index = (k*G^3 + c*G^2 + m*G + y)*3
// grid_n: number of grid nodes per axis (typically 17)
// n     : total pixel count
extern "C" __global__ void icc_cmyk_clut(
    const unsigned char* __restrict__ cmyk,
    unsigned char*       __restrict__ rgb,
    const unsigned char* __restrict__ clut,
    unsigned int grid_n,
    unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    unsigned int ci_base = i * 4;
    unsigned int ri_base = i * 3;

    // Normalise each channel to [0, grid_n - 1].
    float G1 = (float)(grid_n - 1);
    float fc = __ldg(&cmyk[ci_base])     * (G1 / 255.0f);
    float fm = __ldg(&cmyk[ci_base + 1]) * (G1 / 255.0f);
    float fy = __ldg(&cmyk[ci_base + 2]) * (G1 / 255.0f);
    float fk = __ldg(&cmyk[ci_base + 3]) * (G1 / 255.0f);

    int G = (int)grid_n;

    // Floor indices.
    int ic0 = (int)fc;
    int im0 = (int)fm;
    int iy0 = (int)fy;
    int ik0 = (int)fk;

    // Ceil indices (clamped).
    int ic1 = min(ic0 + 1, G - 1);
    int im1 = min(im0 + 1, G - 1);
    int iy1 = min(iy0 + 1, G - 1);
    int ik1 = min(ik0 + 1, G - 1);

    // Fractional weights.
    float wc = fc - (float)ic0;
    float wm = fm - (float)im0;
    float wy = fy - (float)iy0;
    float wk = fk - (float)ik0;

    // Quadrilinear: lerp across 16 CLUT nodes.
    // Variable naming: r/g/b_<c><m><y><k> where each digit is 0 (floor) or 1 (ceil).
    float r00, g00, b00;
    float r01, g01, b01;
    float r10, g10, b10;
    float r11, g11, b11;

    // Interpolate along Y axis first (innermost), then M, then C, then K.

    // K=0 plane
    float rc000, gc000, bc000, rc001, gc001, bc001;
    float rc010, gc010, bc010, rc011, gc011, bc011;
    float rc100, gc100, bc100, rc101, gc101, bc101;
    float rc110, gc110, bc110, rc111, gc111, bc111;

    clut_node(clut, ic0, im0, iy0, ik0, G, &rc000, &gc000, &bc000);
    clut_node(clut, ic0, im0, iy1, ik0, G, &rc001, &gc001, &bc001);
    clut_node(clut, ic0, im1, iy0, ik0, G, &rc010, &gc010, &bc010);
    clut_node(clut, ic0, im1, iy1, ik0, G, &rc011, &gc011, &bc011);
    clut_node(clut, ic1, im0, iy0, ik0, G, &rc100, &gc100, &bc100);
    clut_node(clut, ic1, im0, iy1, ik0, G, &rc101, &gc101, &bc101);
    clut_node(clut, ic1, im1, iy0, ik0, G, &rc110, &gc110, &bc110);
    clut_node(clut, ic1, im1, iy1, ik0, G, &rc111, &gc111, &bc111);

    // Lerp Y, then M, then C → gives the K=0 face result.
    float r_c0m0 = __fmaf_rn(wy, rc001 - rc000, rc000);
    float r_c0m1 = __fmaf_rn(wy, rc011 - rc010, rc010);
    float r_c1m0 = __fmaf_rn(wy, rc101 - rc100, rc100);
    float r_c1m1 = __fmaf_rn(wy, rc111 - rc110, rc110);
    float g_c0m0 = __fmaf_rn(wy, gc001 - gc000, gc000);
    float g_c0m1 = __fmaf_rn(wy, gc011 - gc010, gc010);
    float g_c1m0 = __fmaf_rn(wy, gc101 - gc100, gc100);
    float g_c1m1 = __fmaf_rn(wy, gc111 - gc110, gc110);
    float b_c0m0 = __fmaf_rn(wy, bc001 - bc000, bc000);
    float b_c0m1 = __fmaf_rn(wy, bc011 - bc010, bc010);
    float b_c1m0 = __fmaf_rn(wy, bc101 - bc100, bc100);
    float b_c1m1 = __fmaf_rn(wy, bc111 - bc110, bc110);

    r00 = __fmaf_rn(wm, r_c0m1 - r_c0m0, r_c0m0);
    r10 = __fmaf_rn(wm, r_c1m1 - r_c1m0, r_c1m0);
    g00 = __fmaf_rn(wm, g_c0m1 - g_c0m0, g_c0m0);
    g10 = __fmaf_rn(wm, g_c1m1 - g_c1m0, g_c1m0);
    b00 = __fmaf_rn(wm, b_c0m1 - b_c0m0, b_c0m0);
    b10 = __fmaf_rn(wm, b_c1m1 - b_c1m0, b_c1m0);

    float rk0 = __fmaf_rn(wc, r10 - r00, r00);
    float gk0 = __fmaf_rn(wc, g10 - g00, g00);
    float bk0 = __fmaf_rn(wc, b10 - b00, b00);

    // K=1 plane
    float rc000k1, gc000k1, bc000k1, rc001k1, gc001k1, bc001k1;
    float rc010k1, gc010k1, bc010k1, rc011k1, gc011k1, bc011k1;
    float rc100k1, gc100k1, bc100k1, rc101k1, gc101k1, bc101k1;
    float rc110k1, gc110k1, bc110k1, rc111k1, gc111k1, bc111k1;

    clut_node(clut, ic0, im0, iy0, ik1, G, &rc000k1, &gc000k1, &bc000k1);
    clut_node(clut, ic0, im0, iy1, ik1, G, &rc001k1, &gc001k1, &bc001k1);
    clut_node(clut, ic0, im1, iy0, ik1, G, &rc010k1, &gc010k1, &bc010k1);
    clut_node(clut, ic0, im1, iy1, ik1, G, &rc011k1, &gc011k1, &bc011k1);
    clut_node(clut, ic1, im0, iy0, ik1, G, &rc100k1, &gc100k1, &bc100k1);
    clut_node(clut, ic1, im0, iy1, ik1, G, &rc101k1, &gc101k1, &bc101k1);
    clut_node(clut, ic1, im1, iy0, ik1, G, &rc110k1, &gc110k1, &bc110k1);
    clut_node(clut, ic1, im1, iy1, ik1, G, &rc111k1, &gc111k1, &bc111k1);

    float r_c0m0k1 = __fmaf_rn(wy, rc001k1 - rc000k1, rc000k1);
    float r_c0m1k1 = __fmaf_rn(wy, rc011k1 - rc010k1, rc010k1);
    float r_c1m0k1 = __fmaf_rn(wy, rc101k1 - rc100k1, rc100k1);
    float r_c1m1k1 = __fmaf_rn(wy, rc111k1 - rc110k1, rc110k1);
    float g_c0m0k1 = __fmaf_rn(wy, gc001k1 - gc000k1, gc000k1);
    float g_c0m1k1 = __fmaf_rn(wy, gc011k1 - gc010k1, gc010k1);
    float g_c1m0k1 = __fmaf_rn(wy, gc101k1 - gc100k1, gc100k1);
    float g_c1m1k1 = __fmaf_rn(wy, gc111k1 - gc110k1, gc110k1);
    float b_c0m0k1 = __fmaf_rn(wy, bc001k1 - bc000k1, bc000k1);
    float b_c0m1k1 = __fmaf_rn(wy, bc011k1 - bc010k1, bc010k1);
    float b_c1m0k1 = __fmaf_rn(wy, bc101k1 - bc100k1, bc100k1);
    float b_c1m1k1 = __fmaf_rn(wy, bc111k1 - bc110k1, bc110k1);

    r01 = __fmaf_rn(wm, r_c0m1k1 - r_c0m0k1, r_c0m0k1);
    r11 = __fmaf_rn(wm, r_c1m1k1 - r_c1m0k1, r_c1m0k1);
    g01 = __fmaf_rn(wm, g_c0m1k1 - g_c0m0k1, g_c0m0k1);
    g11 = __fmaf_rn(wm, g_c1m1k1 - g_c1m0k1, g_c1m0k1);
    b01 = __fmaf_rn(wm, b_c0m1k1 - b_c0m0k1, b_c0m0k1);
    b11 = __fmaf_rn(wm, b_c1m1k1 - b_c1m0k1, b_c1m0k1);

    float rk1 = __fmaf_rn(wc, r11 - r01, r01);
    float gk1 = __fmaf_rn(wc, g11 - g01, g01);
    float bk1 = __fmaf_rn(wc, b11 - b01, b01);

    // Final lerp along K.
    float out_r = __fmaf_rn(wk, rk1 - rk0, rk0);
    float out_g = __fmaf_rn(wk, gk1 - gk0, gk0);
    float out_b = __fmaf_rn(wk, bk1 - bk0, bk0);

    // Round to nearest and clamp to [0, 255].
    rgb[ri_base]     = (unsigned char)(int)fminf(fmaxf(out_r + 0.5f, 0.0f), 255.0f);
    rgb[ri_base + 1] = (unsigned char)(int)fminf(fmaxf(out_g + 0.5f, 0.0f), 255.0f);
    rgb[ri_base + 2] = (unsigned char)(int)fminf(fmaxf(out_b + 0.5f, 0.0f), 255.0f);
}
