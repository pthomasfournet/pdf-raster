// GPU ICC colour transform kernels — Slang port of icc_clut.cu.
//
// Two entry points:
//
//   icc_cmyk_matrix — fast path for profiles that reduce to a 3×4 linear matrix.
//     R = (255 − C) × (255 − K) / 255
//     G = (255 − M) × (255 − K) / 255
//     B = (255 − Y) × (255 − K) / 255
//
//   icc_cmyk_clut — full 4D CLUT path for A2B0 profiles.
//     Quadrilinear interpolation over a grid_n^4 CLUT, 16 corner lookups per pixel.
//
// The CUDA source uses `__fmaf_rn` for FMA-with-round-to-nearest semantics.
// Slang exposes the same via the `precise` keyword on the result variable
// (suppresses aggressive reassociation) — see SPIR-V `precise` decoration.
// CLUT access uses StructuredBuffer<uint8_t> (flat buffer); the Texture3D
// + sampler alternative is a perf-only follow-up (see ROADMAP Phase 10
// Task 3 follow-ups) — this layout matches the CUDA path byte-for-byte.

// ── Matrix entry point ──────────────────────────────────────────────────────

[shader("compute")]
[numthreads(256, 1, 1)]
void icc_cmyk_matrix(
    uint3 tid : SV_DispatchThreadID,
    uniform StructuredBuffer<uint8_t> cmyk,
    uniform RWStructuredBuffer<uint8_t> rgb,
    uniform uint n)
{
    uint i = tid.x;
    if (i >= n) return;

    uint ci = i * 4;
    uint ri = i * 3;

    uint c = uint(cmyk[ci + 0]);
    uint m = uint(cmyk[ci + 1]);
    uint y = uint(cmyk[ci + 2]);
    uint k = uint(cmyk[ci + 3]);

    uint inv_k = 255u - k;

    // Integer multiply-round-divide: (255−ch)*(255−k)/255 with unbiased rounding.
    rgb[ri + 0] = uint8_t(((255u - c) * inv_k + 127u) / 255u);
    rgb[ri + 1] = uint8_t(((255u - m) * inv_k + 127u) / 255u);
    rgb[ri + 2] = uint8_t(((255u - y) * inv_k + 127u) / 255u);
}

// ── CLUT entry point ────────────────────────────────────────────────────────

// Fetch one CLUT node.  Layout (matches Rust baker in
// crates/pdf_interp/src/resources/icc.rs):
//   index = (k_idx * G^3 + c_idx * G^2 + m_idx * G + y_idx) * 3
// K is the outermost (major) axis.
//
// Clamping is mandatory, not soft: FP rounding on normalised inputs at
// extreme channel values can push floor indices to G.
void clut_node(
    StructuredBuffer<uint8_t> clut,
    int ci, int mi, int yi, int ki,
    int G,
    out float r, out float g, out float b)
{
    ci = max(0, min(ci, G - 1));
    mi = max(0, min(mi, G - 1));
    yi = max(0, min(yi, G - 1));
    ki = max(0, min(ki, G - 1));

    int idx = (ki * G * G * G + ci * G * G + mi * G + yi) * 3;
    r = float(uint(clut[idx + 0]));
    g = float(uint(clut[idx + 1]));
    b = float(uint(clut[idx + 2]));
}

[shader("compute")]
[numthreads(256, 1, 1)]
void icc_cmyk_clut(
    uint3 tid : SV_DispatchThreadID,
    uniform StructuredBuffer<uint8_t> cmyk,
    uniform RWStructuredBuffer<uint8_t> rgb,
    uniform StructuredBuffer<uint8_t> clut,
    uniform uint grid_n,
    uniform uint n)
{
    uint i = tid.x;
    if (i >= n) return;

    uint ci_base = i * 4;
    uint ri_base = i * 3;

    float G1 = float(grid_n - 1u);
    float fc = float(uint(cmyk[ci_base + 0])) * (G1 / 255.0f);
    float fm = float(uint(cmyk[ci_base + 1])) * (G1 / 255.0f);
    float fy = float(uint(cmyk[ci_base + 2])) * (G1 / 255.0f);
    float fk = float(uint(cmyk[ci_base + 3])) * (G1 / 255.0f);

    int G = int(grid_n);

    int ic0 = int(fc);
    int im0 = int(fm);
    int iy0 = int(fy);
    int ik0 = int(fk);

    int ic1 = min(ic0 + 1, G - 1);
    int im1 = min(im0 + 1, G - 1);
    int iy1 = min(iy0 + 1, G - 1);
    int ik1 = min(ik0 + 1, G - 1);

    float wc = fc - float(ic0);
    float wm = fm - float(im0);
    float wy = fy - float(iy0);
    float wk = fk - float(ik0);

    // Quadrilinear: 16 corner lookups, lerp along Y → M → C → K.
    // `precise` on intermediates suppresses LLVM-style reassociation that
    // would change rounding vs the CUDA __fmaf_rn semantics.

    // K = 0 plane
    float rc000, gc000, bc000;  clut_node(clut, ic0, im0, iy0, ik0, G, rc000, gc000, bc000);
    float rc001, gc001, bc001;  clut_node(clut, ic0, im0, iy1, ik0, G, rc001, gc001, bc001);
    float rc010, gc010, bc010;  clut_node(clut, ic0, im1, iy0, ik0, G, rc010, gc010, bc010);
    float rc011, gc011, bc011;  clut_node(clut, ic0, im1, iy1, ik0, G, rc011, gc011, bc011);
    float rc100, gc100, bc100;  clut_node(clut, ic1, im0, iy0, ik0, G, rc100, gc100, bc100);
    float rc101, gc101, bc101;  clut_node(clut, ic1, im0, iy1, ik0, G, rc101, gc101, bc101);
    float rc110, gc110, bc110;  clut_node(clut, ic1, im1, iy0, ik0, G, rc110, gc110, bc110);
    float rc111, gc111, bc111;  clut_node(clut, ic1, im1, iy1, ik0, G, rc111, gc111, bc111);

    precise float r_c0m0 = wy * (rc001 - rc000) + rc000;
    precise float r_c0m1 = wy * (rc011 - rc010) + rc010;
    precise float r_c1m0 = wy * (rc101 - rc100) + rc100;
    precise float r_c1m1 = wy * (rc111 - rc110) + rc110;
    precise float g_c0m0 = wy * (gc001 - gc000) + gc000;
    precise float g_c0m1 = wy * (gc011 - gc010) + gc010;
    precise float g_c1m0 = wy * (gc101 - gc100) + gc100;
    precise float g_c1m1 = wy * (gc111 - gc110) + gc110;
    precise float b_c0m0 = wy * (bc001 - bc000) + bc000;
    precise float b_c0m1 = wy * (bc011 - bc010) + bc010;
    precise float b_c1m0 = wy * (bc101 - bc100) + bc100;
    precise float b_c1m1 = wy * (bc111 - bc110) + bc110;

    precise float r00 = wm * (r_c0m1 - r_c0m0) + r_c0m0;
    precise float r10 = wm * (r_c1m1 - r_c1m0) + r_c1m0;
    precise float g00 = wm * (g_c0m1 - g_c0m0) + g_c0m0;
    precise float g10 = wm * (g_c1m1 - g_c1m0) + g_c1m0;
    precise float b00 = wm * (b_c0m1 - b_c0m0) + b_c0m0;
    precise float b10 = wm * (b_c1m1 - b_c1m0) + b_c1m0;

    precise float rk0 = wc * (r10 - r00) + r00;
    precise float gk0 = wc * (g10 - g00) + g00;
    precise float bk0 = wc * (b10 - b00) + b00;

    // K = 1 plane
    float rc000k1, gc000k1, bc000k1;  clut_node(clut, ic0, im0, iy0, ik1, G, rc000k1, gc000k1, bc000k1);
    float rc001k1, gc001k1, bc001k1;  clut_node(clut, ic0, im0, iy1, ik1, G, rc001k1, gc001k1, bc001k1);
    float rc010k1, gc010k1, bc010k1;  clut_node(clut, ic0, im1, iy0, ik1, G, rc010k1, gc010k1, bc010k1);
    float rc011k1, gc011k1, bc011k1;  clut_node(clut, ic0, im1, iy1, ik1, G, rc011k1, gc011k1, bc011k1);
    float rc100k1, gc100k1, bc100k1;  clut_node(clut, ic1, im0, iy0, ik1, G, rc100k1, gc100k1, bc100k1);
    float rc101k1, gc101k1, bc101k1;  clut_node(clut, ic1, im0, iy1, ik1, G, rc101k1, gc101k1, bc101k1);
    float rc110k1, gc110k1, bc110k1;  clut_node(clut, ic1, im1, iy0, ik1, G, rc110k1, gc110k1, bc110k1);
    float rc111k1, gc111k1, bc111k1;  clut_node(clut, ic1, im1, iy1, ik1, G, rc111k1, gc111k1, bc111k1);

    precise float r_c0m0k1 = wy * (rc001k1 - rc000k1) + rc000k1;
    precise float r_c0m1k1 = wy * (rc011k1 - rc010k1) + rc010k1;
    precise float r_c1m0k1 = wy * (rc101k1 - rc100k1) + rc100k1;
    precise float r_c1m1k1 = wy * (rc111k1 - rc110k1) + rc110k1;
    precise float g_c0m0k1 = wy * (gc001k1 - gc000k1) + gc000k1;
    precise float g_c0m1k1 = wy * (gc011k1 - gc010k1) + gc010k1;
    precise float g_c1m0k1 = wy * (gc101k1 - gc100k1) + gc100k1;
    precise float g_c1m1k1 = wy * (gc111k1 - gc110k1) + gc110k1;
    precise float b_c0m0k1 = wy * (bc001k1 - bc000k1) + bc000k1;
    precise float b_c0m1k1 = wy * (bc011k1 - bc010k1) + bc010k1;
    precise float b_c1m0k1 = wy * (bc101k1 - bc100k1) + bc100k1;
    precise float b_c1m1k1 = wy * (bc111k1 - bc110k1) + bc110k1;

    precise float r01 = wm * (r_c0m1k1 - r_c0m0k1) + r_c0m0k1;
    precise float r11 = wm * (r_c1m1k1 - r_c1m0k1) + r_c1m0k1;
    precise float g01 = wm * (g_c0m1k1 - g_c0m0k1) + g_c0m0k1;
    precise float g11 = wm * (g_c1m1k1 - g_c1m0k1) + g_c1m0k1;
    precise float b01 = wm * (b_c0m1k1 - b_c0m0k1) + b_c0m0k1;
    precise float b11 = wm * (b_c1m1k1 - b_c1m0k1) + b_c1m0k1;

    precise float rk1 = wc * (r11 - r01) + r01;
    precise float gk1 = wc * (g11 - g01) + g01;
    precise float bk1 = wc * (b11 - b01) + b01;

    // Final lerp along K.
    precise float out_r = wk * (rk1 - rk0) + rk0;
    precise float out_g = wk * (gk1 - gk0) + gk0;
    precise float out_b = wk * (bk1 - bk0) + bk0;

    rgb[ri_base + 0] = uint8_t(int(min(max(out_r + 0.5f, 0.0f), 255.0f)));
    rgb[ri_base + 1] = uint8_t(int(min(max(out_g + 0.5f, 0.0f), 255.0f)));
    rgb[ri_base + 2] = uint8_t(int(min(max(out_b + 0.5f, 0.0f), 255.0f)));
}
