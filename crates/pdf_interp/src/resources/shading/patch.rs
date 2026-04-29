//! Bézier/Coons patch machinery for PDF shading types 6 and 7.
//!
//! Provides [`decode_type6_mesh`] and [`decode_type7_mesh`] which parse the
//! packed bit-stream and adaptively tessellate each patch into Gouraud triangles.

use lopdf::Dictionary;
use raster::shading::gouraud::GouraudVertex;

use super::{
    BitReader, cs_to_rgb, parse_bits_per_comp, parse_bits_per_coord, read_decode_array,
    transform_point,
};
use crate::resources::image::ImageColorSpace;

// ── Constants ─────────────────────────────────────────────────────────────────

/// Colour mixing threshold for subdivision: stop when all corner pairs differ
/// by less than this per component (matches poppler `patchColorDelta`).
pub(super) const COLOR_DELTA: f64 = 3.0 / 255.0;

/// Maximum adaptive subdivision depth (matches poppler `patchMaxDepth`).
pub(super) const MAX_PATCH_DEPTH: u8 = 6;

/// PDF-legal values for `BitsPerFlag` in patch mesh shadings (Table 86).
pub(super) const VALID_FLAG_BITS: &[u8] = &[2, 4, 8];

// ── Patch struct ──────────────────────────────────────────────────────────────

/// A bicubic patch: 4×4 control points in device space + 2×2 corner colours.
///
/// Grid convention (matches poppler `GfxPatch` / PDF §8.7.4.5.6):
/// - `xy[row][col]` where row 0 = "first" edge (u=0), row 3 = "last" edge (u=1).
/// - `color[u_corner][v_corner]` where 0 = min, 1 = max.
///
/// Coordinates are already in device space (CTM applied + y-flipped).
#[derive(Clone, Copy)]
pub(super) struct Patch {
    pub(super) xy: [[[f64; 2]; 4]; 4],
    pub(super) color: [[[u8; 3]; 2]; 2],
}

impl Patch {
    /// Create a zeroed patch (all points at origin, all colours black).
    pub(super) const fn zero() -> Self {
        Self {
            xy: [[[0.0; 2]; 4]; 4],
            color: [[[0u8; 3]; 2]; 2],
        }
    }
}

// ── Patch math helpers ────────────────────────────────────────────────────────

/// Componentwise midpoint of two device-space points — exact, never overflows.
#[inline]
pub(super) const fn mid2(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [f64::midpoint(a[0], b[0]), f64::midpoint(a[1], b[1])]
}

/// Componentwise midpoint of two sRGB corner colours.
#[inline]
pub(super) fn mid_color(a: [u8; 3], b: [u8; 3]) -> [u8; 3] {
    // lerp_u8(a, b, 128) = round((a + b) / 2) — matches poppler's bilinear interp.
    use color::convert::lerp_u8;
    [
        lerp_u8(a[0], b[0], 128),
        lerp_u8(a[1], b[1], 128),
        lerp_u8(a[2], b[2], 128),
    ]
}

/// Maximum per-component colour difference across all 4 corner pairs of the patch.
/// Used as the subdivision-stop criterion.
pub(super) fn color_delta(p: &Patch) -> f64 {
    let corners = [p.color[0][0], p.color[0][1], p.color[1][0], p.color[1][1]];
    let mut max_d = 0.0_f64;
    for i in 0..4 {
        for j in (i + 1)..4 {
            for (a, b) in corners[i].iter().zip(corners[j].iter()) {
                let d = f64::from(a.abs_diff(*b)) / 255.0;
                if d > max_d {
                    max_d = d;
                }
            }
        }
    }
    max_d
}

// ── Patch subdivision ─────────────────────────────────────────────────────────

/// Apply one step of de Casteljau to a row of 4 points, splitting at t=0.5.
///
/// Returns `(left, right)` where each is a 4-point row.
#[inline]
#[expect(
    clippy::similar_names,
    reason = "m01/m012/m0123 are standard de Casteljau intermediate-point names; renaming obscures the algorithm"
)]
pub(super) const fn casteljau_row(row: [[f64; 2]; 4]) -> ([[f64; 2]; 4], [[f64; 2]; 4]) {
    let m01 = mid2(row[0], row[1]);
    let m12 = mid2(row[1], row[2]);
    let m23 = mid2(row[2], row[3]);
    let m012 = mid2(m01, m12);
    let m123 = mid2(m12, m23);
    let m0123 = mid2(m012, m123);
    ([row[0], m01, m012, m0123], [m0123, m123, m23, row[3]])
}

/// Split `p` at u=0.5 — applies de Casteljau along columns (u direction).
///
/// Returns `(lo_u, hi_u)` where `lo_u` covers u∈[0,0.5] and `hi_u` u∈[0.5,1].
pub(super) fn split_patch_u(p: &Patch) -> (Patch, Patch) {
    let mut lo = Patch::zero();
    let mut hi = Patch::zero();
    for col in 0..4 {
        let col_pts = [p.xy[0][col], p.xy[1][col], p.xy[2][col], p.xy[3][col]];
        let (l, r) = casteljau_row(col_pts);
        for row in 0..4 {
            lo.xy[row][col] = l[row];
            hi.xy[row][col] = r[row];
        }
    }
    // Interpolate corner colours along u.
    let c_mid_v0 = mid_color(p.color[0][0], p.color[1][0]);
    let c_mid_v1 = mid_color(p.color[0][1], p.color[1][1]);
    lo.color[0] = p.color[0];
    lo.color[1] = [c_mid_v0, c_mid_v1];
    hi.color[0] = [c_mid_v0, c_mid_v1];
    hi.color[1] = p.color[1];
    (lo, hi)
}

/// Split `p` at v=0.5 — applies de Casteljau along rows (v direction).
///
/// Returns `(lo_v, hi_v)` where `lo_v` covers v∈[0,0.5] and `hi_v` v∈[0.5,1].
pub(super) fn split_patch_v(p: &Patch) -> (Patch, Patch) {
    let mut lo = Patch::zero();
    let mut hi = Patch::zero();
    for row in 0..4 {
        let (l, r) = casteljau_row(p.xy[row]);
        lo.xy[row] = l;
        hi.xy[row] = r;
    }
    // Interpolate corner colours along v.
    let c_mid_u0 = mid_color(p.color[0][0], p.color[0][1]);
    let c_mid_u1 = mid_color(p.color[1][0], p.color[1][1]);
    lo.color[0][0] = p.color[0][0];
    lo.color[0][1] = c_mid_u0;
    lo.color[1][0] = p.color[1][0];
    lo.color[1][1] = c_mid_u1;
    hi.color[0][0] = c_mid_u0;
    hi.color[0][1] = p.color[0][1];
    hi.color[1][0] = c_mid_u1;
    hi.color[1][1] = p.color[1][1];
    (lo, hi)
}

// ── Tessellation ──────────────────────────────────────────────────────────────

/// Extract a corner device-space vertex from a patch.
///
/// `u=0` → row 0; `u=1` → row 3. `v=0` → col 0; `v=1` → col 3.
#[inline]
pub(super) const fn corner_vertex(p: &Patch, u: usize, v: usize) -> GouraudVertex {
    let row = u * 3;
    let col = v * 3;
    GouraudVertex {
        x: p.xy[row][col][0],
        y: p.xy[row][col][1],
        color: p.color[u][v],
    }
}

/// Adaptively subdivide `patch` into Gouraud triangles and append them to `out`.
///
/// Subdivision stops when all corner colours are within [`COLOR_DELTA`] of each
/// other or [`MAX_PATCH_DEPTH`] is reached, matching poppler's strategy.
/// Uses an explicit work stack (no recursion) to avoid stack overflow.
#[expect(
    clippy::large_types_passed_by_value,
    reason = "Patch (272 B) is consumed on the work stack; reference would require a Clone per push"
)]
pub(super) fn tessellate_patch(patch: Patch, out: &mut Vec<[GouraudVertex; 3]>) {
    // Pre-allocate the work stack.  In a DFS traversal of a 4-ary tree of depth D,
    // at most 3*D+1 nodes are simultaneously live on the stack (3 siblings waiting
    // at each level plus the children just pushed).  For MAX_PATCH_DEPTH=6 that is 19;
    // 24 is a safe round-up.
    let mut stack: Vec<(Patch, u8)> = Vec::with_capacity(3 * MAX_PATCH_DEPTH as usize + 1);
    stack.push((patch, 0));

    while let Some((p, depth)) = stack.pop() {
        if depth >= MAX_PATCH_DEPTH || color_delta(&p) <= COLOR_DELTA {
            // Leaf: emit 2 triangles covering the patch quad from its 4 corners.
            let v00 = corner_vertex(&p, 0, 0);
            let v01 = corner_vertex(&p, 0, 1);
            let v10 = corner_vertex(&p, 1, 0);
            let v11 = corner_vertex(&p, 1, 1);
            out.push([v00, v01, v10]);
            out.push([v01, v11, v10]);
        } else {
            // Split into 4 sub-patches and push them.
            let (pu0, pu1) = split_patch_u(&p);
            let (p00, p01) = split_patch_v(&pu0);
            let (p10, p11) = split_patch_v(&pu1);
            let nd = depth + 1;
            stack.push((p00, nd));
            stack.push((p01, nd));
            stack.push((p10, nd));
            stack.push((p11, nd));
        }
    }
}

// ── Stream I/O helpers ────────────────────────────────────────────────────────

/// Read `n_pts` coordinate pairs from the bit stream and convert to device space.
///
/// Returns `None` on stream truncation (all previously decoded triangles are
/// returned by the caller rather than discarding the whole shading).
pub(super) fn read_patch_points(
    reader: &mut BitReader<'_>,
    n_pts: usize,
    bpc: u8,
    decode: &[f64],
    ctm: &[f64; 6],
    page_h: f64,
) -> Option<Vec<[f64; 2]>> {
    // bpc ∈ VALID_BITS ⊆ [1,32]; u64 fits the shift; cast to f64 is exact (≤ 2^32−1 < 2^53).
    #[expect(
        clippy::cast_precision_loss,
        reason = "max_coord ≤ 2^32−1; f64 mantissa is 52 bits"
    )]
    let max_coord = ((1u64 << bpc) - 1) as f64;
    let x_min = decode.first().copied().unwrap_or(0.0);
    let x_max = decode.get(1).copied().unwrap_or(1.0);
    let y_min = decode.get(2).copied().unwrap_or(0.0);
    let y_max = decode.get(3).copied().unwrap_or(1.0);

    let mut pts = Vec::with_capacity(n_pts);
    for _ in 0..n_pts {
        let raw_x = reader.read_bits(bpc)?;
        let raw_y = reader.read_bits(bpc)?;
        // u32 → f64 is always lossless (u32 < 2^32 ≤ 2^53 mantissa bits).
        let ux = (f64::from(raw_x) / max_coord).mul_add(x_max - x_min, x_min);
        let uy = (f64::from(raw_y) / max_coord).mul_add(y_max - y_min, y_min);
        pts.push(<[f64; 2]>::from(transform_point(ctm, ux, uy, page_h)));
    }
    Some(pts)
}

/// Read `n_colors` corner colours from the bit stream.
///
/// Returns `None` on stream truncation.
pub(super) fn read_patch_colors(
    reader: &mut BitReader<'_>,
    n_colors: usize,
    n_channels: usize,
    bpcomp: u8,
    decode: &[f64],
    cs: ImageColorSpace,
) -> Option<Vec<[u8; 3]>> {
    // bpcomp ∈ VALID_BITS ⊆ [1,32]; cast to f64 exact for u32-range values.
    #[expect(
        clippy::cast_precision_loss,
        reason = "max_comp ≤ 2^32−1; f64 mantissa is 52 bits"
    )]
    let max_comp = ((1u64 << bpcomp) - 1) as f64;
    let mut colors = Vec::with_capacity(n_colors);
    for _ in 0..n_colors {
        let mut raw = [0u32; 4];
        for ch in raw.iter_mut().take(n_channels) {
            *ch = reader.read_bits(bpcomp)?;
        }
        // u32 → f64 is always lossless; mul_add avoids a separate rounding step.
        let channels: Vec<f64> = raw[..n_channels]
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                let c_min = decode.get(4 + i * 2).copied().unwrap_or(0.0);
                let c_max = decode.get(4 + i * 2 + 1).copied().unwrap_or(1.0);
                (f64::from(v) / max_comp).mul_add(c_max - c_min, c_min)
            })
            .collect();
        colors.push(cs_to_rgb(cs, &channels));
    }
    Some(colors)
}

/// Validate and extract `BitsPerFlag` from a patch mesh shading dictionary.
pub(super) fn parse_bits_per_flag(sh: &Dictionary, tag: &str) -> Option<u8> {
    super::parse_bits_field(sh, b"BitsPerFlag", VALID_FLAG_BITS, "BitsPerFlag", tag)
}

// ── Patch assembly — Type 6 ───────────────────────────────────────────────────

/// Assign the 12 boundary stream points into the Type 6 4×4 grid for flag=0.
///
/// Stream order traces the boundary clockwise:
/// top row (left→right), right col (down), bottom row (right→left), left col (up).
/// Interior points `[1][1]`, `[1][2]`, `[2][1]`, `[2][2]` are computed from the boundary.
pub(super) fn build_patch_type6(pts: &[[f64; 2]], colors: &[[u8; 3]]) -> Patch {
    debug_assert_eq!(pts.len(), 12);
    debug_assert_eq!(colors.len(), 4);
    let mut p = Patch::zero();

    // Boundary point assignment (PDF §8.7.4.5.6 Table 87, flag=0).
    p.xy[0][0] = pts[0];
    p.xy[0][1] = pts[1];
    p.xy[0][2] = pts[2];
    p.xy[0][3] = pts[3];
    p.xy[1][3] = pts[4];
    p.xy[2][3] = pts[5];
    p.xy[3][3] = pts[6];
    p.xy[3][2] = pts[7];
    p.xy[3][1] = pts[8];
    p.xy[3][0] = pts[9];
    p.xy[2][0] = pts[10];
    p.xy[1][0] = pts[11];

    // Corner colours: [0][0]=c0, [0][1]=c1, [1][1]=c2, [1][0]=c3.
    p.color[0][0] = colors[0];
    p.color[0][1] = colors[1];
    p.color[1][1] = colors[2];
    p.color[1][0] = colors[3];

    fill_type6_interior(&mut p);
    p
}

/// Derive the 4 interior control points of a Coons patch from its boundary.
///
/// Formula from poppler `GfxState.cc:5374–5385`, which implements the standard
/// Coons patch interior derivation (same formula for x and y independently).
pub(super) fn fill_type6_interior(p: &mut Patch) {
    for k in 0..2usize {
        // Snapshot boundary values before writing interior — avoids borrow conflict.
        let [p00, p01, p02, p03] = [p.xy[0][0][k], p.xy[0][1][k], p.xy[0][2][k], p.xy[0][3][k]];
        let [p10, p13] = [p.xy[1][0][k], p.xy[1][3][k]];
        let [p20, p23] = [p.xy[2][0][k], p.xy[2][3][k]];
        let [p30, p31, p32, p33] = [p.xy[3][0][k], p.xy[3][1][k], p.xy[3][2][k], p.xy[3][3][k]];
        // Coons formula: (-4a + 6(b+c) - 2(d+e) + 3(f+g) - h) / 9
        // Inner mul_add calls use FMA for reduced rounding error.
        p.xy[1][1][k] = ((-4.0f64).mul_add(p00, 6.0 * (p01 + p10))
            + (-2.0f64).mul_add(p03 + p30, 3.0 * (p31 + p13))
            - p33)
            / 9.0;
        p.xy[1][2][k] = ((-4.0f64).mul_add(p03, 6.0 * (p02 + p13))
            + (-2.0f64).mul_add(p00 + p33, 3.0 * (p32 + p10))
            - p30)
            / 9.0;
        p.xy[2][1][k] = ((-4.0f64).mul_add(p30, 6.0 * (p31 + p20))
            + (-2.0f64).mul_add(p33 + p00, 3.0 * (p01 + p23))
            - p03)
            / 9.0;
        p.xy[2][2][k] = ((-4.0f64).mul_add(p33, 6.0 * (p32 + p23))
            + (-2.0f64).mul_add(p30 + p03, 3.0 * (p02 + p20))
            - p00)
            / 9.0;
    }
}

/// Apply shared-edge continuation for Type 6, flags 1–3.
///
/// Copies 4 control points and 2 corner colours from `prev`, then places the
/// 8 new boundary points from `pts` and 2 new colours from `colors`.
/// Interior points are derived after all boundary points are placed.
pub(super) fn apply_flag_type6(
    flag: u8,
    prev: &Patch,
    pts: &[[f64; 2]],
    colors: &[[u8; 3]],
) -> Patch {
    debug_assert_eq!(pts.len(), 8);
    debug_assert_eq!(colors.len(), 2);
    let mut p = Patch::zero();

    match flag {
        1 => {
            // Shared: prev right edge → new left edge (col 0 of new patch = col 3 of prev).
            for row in 0..4 {
                p.xy[row][0] = prev.xy[row][3];
            }
            p.color[0][0] = prev.color[0][1];
            p.color[1][0] = prev.color[1][1];
            // New boundary: right edge (col 3), then bottom (row 3) and top (row 0) partial.
            p.xy[0][1] = pts[0];
            p.xy[0][2] = pts[1];
            p.xy[0][3] = pts[2];
            p.xy[1][3] = pts[3];
            p.xy[2][3] = pts[4];
            p.xy[3][3] = pts[5];
            p.xy[3][2] = pts[6];
            p.xy[3][1] = pts[7];
            p.color[0][1] = colors[0];
            p.color[1][1] = colors[1];
        }
        2 => {
            // Shared: prev bottom edge → new top edge (row 0 of new = row 3 of prev).
            for col in 0..4 {
                p.xy[0][col] = prev.xy[3][col];
            }
            p.color[0][0] = prev.color[1][0];
            p.color[0][1] = prev.color[1][1];
            // New boundary.
            p.xy[1][3] = pts[0];
            p.xy[2][3] = pts[1];
            p.xy[3][3] = pts[2];
            p.xy[3][2] = pts[3];
            p.xy[3][1] = pts[4];
            p.xy[3][0] = pts[5];
            p.xy[2][0] = pts[6];
            p.xy[1][0] = pts[7];
            p.color[1][1] = colors[0];
            p.color[1][0] = colors[1];
        }
        3 => {
            // Shared: prev left edge → new right edge (col 3 of new = col 0 of prev).
            for row in 0..4 {
                p.xy[row][3] = prev.xy[row][0];
            }
            p.color[0][1] = prev.color[0][0];
            p.color[1][1] = prev.color[1][0];
            // New boundary: left edge (col 0), then top (row 0) and bottom (row 3) partial.
            p.xy[3][2] = pts[0];
            p.xy[3][1] = pts[1];
            p.xy[3][0] = pts[2];
            p.xy[2][0] = pts[3];
            p.xy[1][0] = pts[4];
            p.xy[0][0] = pts[5];
            p.xy[0][1] = pts[6];
            p.xy[0][2] = pts[7];
            p.color[0][0] = colors[0];
            p.color[1][0] = colors[1];
        }
        _ => unreachable!("apply_flag_type6 called with flag={flag}; only 1–3 are valid"),
    }

    fill_type6_interior(&mut p);
    p
}

// ── Patch assembly — Type 7 ───────────────────────────────────────────────────

/// Assign the 16 stream points into the Type 7 4×4 grid for flag=0.
///
/// Unlike Type 6, all 16 points (12 boundary + 4 interior) are read from the stream.
/// Stream order: top row, right col, bottom row (reversed), left col (reversed),
/// then 4 interior points in a specific order.
pub(super) fn build_patch_type7(pts: &[[f64; 2]], colors: &[[u8; 3]]) -> Patch {
    debug_assert_eq!(pts.len(), 16);
    debug_assert_eq!(colors.len(), 4);
    let mut p = Patch::zero();

    // Boundary (same layout as Type 6).
    p.xy[0][0] = pts[0];
    p.xy[0][1] = pts[1];
    p.xy[0][2] = pts[2];
    p.xy[0][3] = pts[3];
    p.xy[1][3] = pts[4];
    p.xy[2][3] = pts[5];
    p.xy[3][3] = pts[6];
    p.xy[3][2] = pts[7];
    p.xy[3][1] = pts[8];
    p.xy[3][0] = pts[9];
    p.xy[2][0] = pts[10];
    p.xy[1][0] = pts[11];

    // Interior points (PDF §8.7.4.5.7 Table 89, flag=0, pts 12–15).
    p.xy[1][1] = pts[12];
    p.xy[1][2] = pts[13];
    p.xy[2][2] = pts[14];
    p.xy[2][1] = pts[15];

    // Corner colours.
    p.color[0][0] = colors[0];
    p.color[0][1] = colors[1];
    p.color[1][1] = colors[2];
    p.color[1][0] = colors[3];
    p
}

/// Apply shared-edge continuation for Type 7, flags 1–3.
///
/// For Type 7, the 4 interior control points are always read from the stream
/// (as part of the 12 new points), never copied from the previous patch.
pub(super) fn apply_flag_type7(
    flag: u8,
    prev: &Patch,
    pts: &[[f64; 2]],
    colors: &[[u8; 3]],
) -> Patch {
    debug_assert_eq!(pts.len(), 12);
    debug_assert_eq!(colors.len(), 2);
    let mut p = Patch::zero();

    match flag {
        1 => {
            // Shared: prev right edge → new left edge.
            for row in 0..4 {
                p.xy[row][0] = prev.xy[row][3];
            }
            p.color[0][0] = prev.color[0][1];
            p.color[1][0] = prev.color[1][1];
            // 8 new boundary points + 4 interior.
            p.xy[0][1] = pts[0];
            p.xy[0][2] = pts[1];
            p.xy[0][3] = pts[2];
            p.xy[1][3] = pts[3];
            p.xy[2][3] = pts[4];
            p.xy[3][3] = pts[5];
            p.xy[3][2] = pts[6];
            p.xy[3][1] = pts[7];
            p.xy[1][1] = pts[8];
            p.xy[1][2] = pts[9];
            p.xy[2][2] = pts[10];
            p.xy[2][1] = pts[11];
            p.color[0][1] = colors[0];
            p.color[1][1] = colors[1];
        }
        2 => {
            // Shared: prev bottom edge → new top edge.
            for col in 0..4 {
                p.xy[0][col] = prev.xy[3][col];
            }
            p.color[0][0] = prev.color[1][0];
            p.color[0][1] = prev.color[1][1];
            p.xy[1][3] = pts[0];
            p.xy[2][3] = pts[1];
            p.xy[3][3] = pts[2];
            p.xy[3][2] = pts[3];
            p.xy[3][1] = pts[4];
            p.xy[3][0] = pts[5];
            p.xy[2][0] = pts[6];
            p.xy[1][0] = pts[7];
            p.xy[1][1] = pts[8];
            p.xy[1][2] = pts[9];
            p.xy[2][2] = pts[10];
            p.xy[2][1] = pts[11];
            p.color[1][1] = colors[0];
            p.color[1][0] = colors[1];
        }
        3 => {
            // Shared: prev left edge → new right edge.
            for row in 0..4 {
                p.xy[row][3] = prev.xy[row][0];
            }
            p.color[0][1] = prev.color[0][0];
            p.color[1][1] = prev.color[1][0];
            p.xy[3][2] = pts[0];
            p.xy[3][1] = pts[1];
            p.xy[3][0] = pts[2];
            p.xy[2][0] = pts[3];
            p.xy[1][0] = pts[4];
            p.xy[0][0] = pts[5];
            p.xy[0][1] = pts[6];
            p.xy[0][2] = pts[7];
            p.xy[1][1] = pts[8];
            p.xy[1][2] = pts[9];
            p.xy[2][2] = pts[10];
            p.xy[2][1] = pts[11];
            p.color[0][0] = colors[0];
            p.color[1][0] = colors[1];
        }
        _ => unreachable!("apply_flag_type7 called with flag={flag}; only 1–3 are valid"),
    }
    p
}

// ── Top-level mesh decoders ───────────────────────────────────────────────────

/// Decode a Type 6 (Coons patch mesh) shading stream.
///
/// Each record: `BitsPerFlag`-bit flag, then 12 (flag=0) or 8 (flag≥1) coordinate
/// pairs at `BitsPerCoordinate` bits each, then 4 (flag=0) or 2 (flag≥1) colours
/// at `BitsPerComponent` bits per channel. Interior control points are derived from
/// the boundary using the standard Coons formula. Each patch is tessellated into
/// Gouraud triangles via adaptive subdivision.
pub(super) fn decode_type6_mesh(
    sh: &Dictionary,
    data: &[u8],
    cs: ImageColorSpace,
    n_channels: usize,
    ctm: &[f64; 6],
    page_h: f64,
) -> Vec<[GouraudVertex; 3]> {
    let Some(bpc) = parse_bits_per_coord(sh, "type6") else {
        return vec![];
    };
    let Some(bpcomp) = parse_bits_per_comp(sh, "type6") else {
        return vec![];
    };
    let Some(bpf) = parse_bits_per_flag(sh, "type6") else {
        return vec![];
    };
    let decode = read_decode_array(sh, n_channels);

    let mut reader = BitReader::new(data);
    let mut triangles: Vec<[GouraudVertex; 3]> = Vec::new();
    let mut prev: Option<Patch> = None;

    #[expect(
        clippy::cast_possible_truncation,
        reason = "bpf ∈ {2,4,8}; value fits u8"
    )]
    while let Some(flag_raw) = reader.read_bits(bpf) {
        let flag = flag_raw as u8;

        let (n_pts, n_colors): (usize, usize) = if flag == 0 { (12, 4) } else { (8, 2) };

        let Some(pts) = read_patch_points(&mut reader, n_pts, bpc, &decode, ctm, page_h) else {
            break;
        };
        let Some(colors) =
            read_patch_colors(&mut reader, n_colors, n_channels, bpcomp, &decode, cs)
        else {
            break;
        };

        let patch = match (flag, &prev) {
            (0, _) => build_patch_type6(&pts, &colors),
            (1..=3, Some(p)) => apply_flag_type6(flag, p, &pts, &colors),
            (f, None) => {
                log::debug!("shading/type6: flag={f} with no previous patch — skipping record");
                continue;
            }
            (f, _) => {
                log::debug!("shading/type6: unknown flag {f} — skipping record");
                continue;
            }
        };

        tessellate_patch(patch, &mut triangles);
        prev = Some(patch);
    }

    triangles
}

/// Decode a Type 7 (tensor-product patch mesh) shading stream.
///
/// Like Type 6 but all 16 control points per patch are read directly from the
/// stream (4 interior points are not derived). Flag semantics and colour handling
/// are identical to Type 6.
pub(super) fn decode_type7_mesh(
    sh: &Dictionary,
    data: &[u8],
    cs: ImageColorSpace,
    n_channels: usize,
    ctm: &[f64; 6],
    page_h: f64,
) -> Vec<[GouraudVertex; 3]> {
    let Some(bpc) = parse_bits_per_coord(sh, "type7") else {
        return vec![];
    };
    let Some(bpcomp) = parse_bits_per_comp(sh, "type7") else {
        return vec![];
    };
    let Some(bpf) = parse_bits_per_flag(sh, "type7") else {
        return vec![];
    };
    let decode = read_decode_array(sh, n_channels);

    let mut reader = BitReader::new(data);
    let mut triangles: Vec<[GouraudVertex; 3]> = Vec::new();
    let mut prev: Option<Patch> = None;

    #[expect(
        clippy::cast_possible_truncation,
        reason = "bpf ∈ {2,4,8}; value fits u8"
    )]
    while let Some(flag_raw) = reader.read_bits(bpf) {
        let flag = flag_raw as u8;

        let (n_pts, n_colors): (usize, usize) = if flag == 0 { (16, 4) } else { (12, 2) };

        let Some(pts) = read_patch_points(&mut reader, n_pts, bpc, &decode, ctm, page_h) else {
            break;
        };
        let Some(colors) =
            read_patch_colors(&mut reader, n_colors, n_channels, bpcomp, &decode, cs)
        else {
            break;
        };

        let patch = match (flag, &prev) {
            (0, _) => build_patch_type7(&pts, &colors),
            (1..=3, Some(p)) => apply_flag_type7(flag, p, &pts, &colors),
            (f, None) => {
                log::debug!("shading/type7: flag={f} with no previous patch — skipping record");
                continue;
            }
            (f, _) => {
                log::debug!("shading/type7: unknown flag {f} — skipping record");
                continue;
            }
        };

        tessellate_patch(patch, &mut triangles);
        prev = Some(patch);
    }

    triangles
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_patch(
        x0: f64,
        y0: f64,
        x1: f64,
        y1: f64,
        c00: [u8; 3],
        c01: [u8; 3],
        c10: [u8; 3],
        c11: [u8; 3],
    ) -> Patch {
        // Bilinear interpolation of control points across the 4×4 grid.
        let mut p = Patch::zero();
        for r in 0..4usize {
            let tr = r as f64 / 3.0;
            for c in 0..4usize {
                let tc = c as f64 / 3.0;
                p.xy[r][c][0] = x0 + (x1 - x0) * tc;
                p.xy[r][c][1] = y0 + (y1 - y0) * tr;
            }
        }
        p.color[0][0] = c00;
        p.color[0][1] = c01;
        p.color[1][0] = c10;
        p.color[1][1] = c11;
        p
    }

    #[test]
    fn split_patch_u_flat_corners_unchanged() {
        let p = flat_patch(
            0.0,
            0.0,
            6.0,
            4.0,
            [0, 0, 0],
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
        );
        let (left, right) = split_patch_u(&p);
        // Top-left corner should be exactly preserved in both halves.
        assert_eq!(left.xy[0][0], p.xy[0][0]);
        assert_eq!(right.xy[3][0], p.xy[3][0]);
        // Split point on left side should be at midpoint in y.
        let mid_y = (p.xy[0][0][1] + p.xy[3][0][1]) / 2.0;
        assert!(
            (left.xy[3][0][1] - mid_y).abs() < 1e-9,
            "left bottom y={} expected={}",
            left.xy[3][0][1],
            mid_y
        );
        assert!(
            (right.xy[0][0][1] - mid_y).abs() < 1e-9,
            "right top y={} expected={}",
            right.xy[0][0][1],
            mid_y
        );
        // Colors: left's u=1 edge should be midpoint of original u edges.
        for ch in 0..3 {
            let expected = ((p.color[0][0][ch] as u16 + p.color[1][0][ch] as u16) / 2) as u8;
            assert!(
                (left.color[1][0][ch] as i16 - expected as i16).abs() <= 1,
                "color mismatch ch={ch}"
            );
        }
    }

    #[test]
    fn split_patch_v_flat_corners_unchanged() {
        let p = flat_patch(
            0.0,
            0.0,
            8.0,
            6.0,
            [10, 20, 30],
            [200, 100, 50],
            [30, 60, 90],
            [180, 90, 45],
        );
        let (left, right) = split_patch_v(&p);
        assert_eq!(left.xy[0][0], p.xy[0][0]);
        assert_eq!(right.xy[0][3], p.xy[0][3]);
        let mid_x = (p.xy[0][0][0] + p.xy[0][3][0]) / 2.0;
        assert!((left.xy[0][3][0] - mid_x).abs() < 1e-9);
        assert!((right.xy[0][0][0] - mid_x).abs() < 1e-9);
    }

    #[test]
    fn color_delta_uniform_is_zero() {
        let mut p = Patch::zero();
        let c = [128u8, 64, 32];
        p.color[0][0] = c;
        p.color[0][1] = c;
        p.color[1][0] = c;
        p.color[1][1] = c;
        assert_eq!(color_delta(&p), 0.0);
    }

    #[test]
    fn color_delta_max_diff_is_one() {
        let mut p = Patch::zero();
        p.color[0][0] = [0, 0, 0];
        p.color[0][1] = [255, 0, 0];
        p.color[1][0] = [0, 0, 0];
        p.color[1][1] = [0, 0, 0];
        let d = color_delta(&p);
        assert!((d - 1.0).abs() < 1e-9, "expected 1.0, got {d}");
    }

    #[test]
    fn tessellate_patch_uniform_emits_two_triangles() {
        // A patch where all corners have the same colour → depth=0 leaf immediately.
        let p = flat_patch(
            0.0,
            0.0,
            1.0,
            1.0,
            [128, 128, 128],
            [128, 128, 128],
            [128, 128, 128],
            [128, 128, 128],
        );
        let mut out = Vec::new();
        tessellate_patch(p, &mut out);
        assert_eq!(
            out.len(),
            2,
            "expected exactly 2 triangles for uniform-colour patch"
        );
    }

    #[test]
    fn build_patch_type6_interior_formula() {
        // Use a simple square boundary with known analytic interior.
        // p[0][0]=(0,0)  p[0][1]=(1,0) p[0][2]=(2,0) p[0][3]=(3,0)
        // p[1][3]=(3,1)  p[2][3]=(3,2)
        // p[3][3]=(3,3)  p[3][2]=(2,3) p[3][1]=(1,3) p[3][0]=(0,3)
        // p[2][0]=(0,2)  p[1][0]=(0,1)
        //
        // For a uniform-spaced rectilinear grid the Coons interior formula collapses to
        // simple bilinear: p[1][1]=(1,1), p[1][2]=(2,1), p[2][1]=(1,2), p[2][2]=(2,2).
        let pts: Vec<[f64; 2]> = vec![
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0], // p[0][0..3]
            [3.0, 1.0],
            [3.0, 2.0], // p[1][3], p[2][3]
            [3.0, 3.0],
            [2.0, 3.0],
            [1.0, 3.0],
            [0.0, 3.0], // p[3][3..0]
            [0.0, 2.0],
            [0.0, 1.0], // p[2][0], p[1][0]
        ];
        let colors: Vec<[u8; 3]> = vec![[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]];
        let p = build_patch_type6(&pts, &colors);
        for (r, c, ex, ey) in [
            (1, 1, 1.0, 1.0),
            (1, 2, 2.0, 1.0),
            (2, 1, 1.0, 2.0),
            (2, 2, 2.0, 2.0),
        ] {
            assert!(
                (p.xy[r][c][0] - ex).abs() < 1e-9 && (p.xy[r][c][1] - ey).abs() < 1e-9,
                "interior[{r}][{c}] = {:?} expected ({ex},{ey})",
                p.xy[r][c]
            );
        }
    }
}
