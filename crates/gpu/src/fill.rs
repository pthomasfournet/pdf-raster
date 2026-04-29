//! Tile records, Halton sampling, and CPU anti-aliased fill.

use cudarc::driver::DeviceRepr;

use super::{TILE_H, TILE_W};

/// One tile record per (segment, tile-row) crossing.
///
/// Layout must match `struct TileRecord` in `tile_fill.cu` exactly.
/// The struct is `repr(C)` and 32 bytes so that `bytemuck::cast_slice` can
/// transmit it to the GPU without additional copying.
#[derive(Clone, Copy, Default)]
#[repr(C)]
pub struct TileRecord {
    /// Sort key: `(tile_y << 16) | tile_x`.  Set by [`build_tile_records`].
    pub key: u32,
    /// Segment x at the top of the segment's y-extent within this tile (tile-local).
    pub x_enter: f32,
    /// Slope: dx/dy in device pixels.
    pub dxdy: f32,
    /// Segment start y within this tile row (0..`TILE_H`).
    pub y0_tile: f32,
    /// Segment end y within this tile row (0..`TILE_H`).
    pub y1_tile: f32,
    /// Sign: `+1.0` for an upward-crossing segment, `-1.0` for downward.
    pub sign: f32,
    /// Padding (must be 0).
    #[expect(
        clippy::pub_underscore_fields,
        reason = "padding field required for repr(C) alignment to match tile_fill.cu struct layout"
    )]
    pub _pad: u32,
    /// Padding (must be 0).
    #[expect(
        clippy::pub_underscore_fields,
        reason = "padding field required for repr(C) alignment to match tile_fill.cu struct layout"
    )]
    pub _pad2: u32,
}

// SAFETY: TileRecord is repr(C), all fields are primitive types (u32, f32), no
// uninitialised padding — bytemuck::Pod and cudarc::DeviceRepr are safe.
unsafe impl bytemuck::Pod for TileRecord {}
unsafe impl bytemuck::Zeroable for TileRecord {}
// SAFETY: TileRecord has no pointer types or other non-device-representable fields;
// all fields are plain u32/f32 with repr(C) alignment.
unsafe impl DeviceRepr for TileRecord {}

/// f32 aliases for tile dimensions — values are 16.0, exact in f32.
/// Avoids repeated `TILE_W/H as f32` casts inside `build_tile_records` that
/// would fire `cast_precision_loss` despite being trivially safe.
#[expect(
    clippy::cast_precision_loss,
    reason = "TILE_W/H = 16, exact in f32 (24-bit mantissa)"
)]
const TILE_W_F: f32 = TILE_W as f32;
#[expect(
    clippy::cast_precision_loss,
    reason = "TILE_W/H = 16, exact in f32 (24-bit mantissa)"
)]
const TILE_H_F: f32 = TILE_H as f32;

/// Build a sorted list of [`TileRecord`]s from a flat segment list, plus the
/// `tile_starts` / `tile_counts` index arrays required by [`crate::GpuCtx::tile_fill`].
///
/// `segs` is packed `[x0, y0, x1, y1]` per segment in device pixels, same
/// format as [`crate::GpuCtx::aa_fill`].  `x_min`, `y_min`, `width`, `height` define
/// the fill bounding box in device pixels.
///
/// Returns `(records, tile_starts, tile_counts, grid_w)`.
///
/// # Panics
///
/// Panics if `segs.len()` is not a multiple of 4, or if `width` or `height`
/// require more than 65535 tiles in either dimension (i.e. exceed `65535 × TILE_W`
/// or `65535 × TILE_H` pixels) — the sort key packs tile coordinates into 16 bits each.
#[must_use]
#[expect(
    clippy::too_many_lines,
    reason = "tile-record builder is a single coherent algorithm; splitting would obscure the data flow"
)]
pub fn build_tile_records(
    segs: &[f32],
    x_min: f32,
    y_min: f32,
    width: u32,
    height: u32,
) -> (Vec<TileRecord>, Vec<u32>, Vec<u32>, u32) {
    assert!(
        segs.len().is_multiple_of(4),
        "segs.len() must be a multiple of 4 (got {})",
        segs.len()
    );

    let grid_w = width.div_ceil(TILE_W);
    let grid_h = height.div_ceil(TILE_H);
    assert!(
        grid_w <= 0xFFFF && grid_h <= 0xFFFF,
        "raster too large: grid {grid_w}×{grid_h} tiles exceeds 16-bit tile key range",
    );
    // grid_w, grid_h ≤ 0xFFFF; product ≤ 65535² = 4_294_836_225 < u32::MAX — no overflow.
    let n_tiles = (grid_w * grid_h) as usize;

    let mut records: Vec<TileRecord> = Vec::new();

    for seg in segs.chunks_exact(4) {
        let (mut sx0, mut sy0, mut sx1, mut sy1) = (
            seg[0] - x_min,
            seg[1] - y_min,
            seg[2] - x_min,
            seg[3] - y_min,
        );

        // Skip horizontal segments (no winding contribution).
        if (sy1 - sy0).abs() < 1e-6 {
            continue;
        }

        // Enforce sy0 ≤ sy1; record crossing direction as sign.
        let sign = if sy0 > sy1 {
            std::mem::swap(&mut sx0, &mut sx1);
            std::mem::swap(&mut sy0, &mut sy1);
            -1.0f32
        } else {
            1.0f32
        };

        let dxdy = (sx1 - sx0) / (sy1 - sy0);

        // Clamp to output bounds.
        let ey0 = sy0.max(0.0);
        // height ≤ u32::MAX px; page heights are always ≤ 32768 at supported DPIs —
        // exact in f32 (which has a 24-bit mantissa, covering integers to 16M).
        #[expect(
            clippy::cast_precision_loss,
            reason = "height ≤ 32768 px in practice; exact in f32 (24-bit mantissa)"
        )]
        let ey1 = sy1.min(height as f32);
        if ey0 >= ey1 {
            continue;
        }

        // First and last tile rows the segment (after clamping) crosses.
        // Subtract a small epsilon from ey1 so a segment ending exactly on a
        // tile boundary doesn't bleed into the next tile row.
        // ey0/ey1 are non-negative and floored ≤ height/TILE_H_F ≤ grid_h ≤ 0xFFFF — fits u32.
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "floor(non-negative / TILE_H_F) is ≥ 0 and ≤ grid_h ≤ 0xFFFF — fits u32"
        )]
        let ty0 = (ey0 / TILE_H_F).floor() as u32;
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "floor(non-negative / TILE_H_F) is ≥ 0 and ≤ grid_h ≤ 0xFFFF — fits u32"
        )]
        let ty1 = ((ey1 - 1e-6).max(0.0) / TILE_H_F).floor() as u32;

        for ty in ty0..=ty1.min(grid_h - 1) {
            // ty ≤ 0xFFFF, TILE_H_F = 16.0; product ≤ ~1M — exact in f32 (24-bit mantissa).
            #[expect(
                clippy::cast_precision_loss,
                reason = "ty * TILE_H ≤ 0xFFFF*16 ≈ 1M; exact in f32"
            )]
            let tile_top = (ty * TILE_H) as f32;
            let tile_bot = tile_top + TILE_H_F;

            // Segment y-extent clipped to this tile row, in tile-local coords.
            let seg_y0_tile = ey0.max(tile_top) - tile_top;
            let seg_y1_tile = ey1.min(tile_bot) - tile_top;
            if seg_y0_tile >= seg_y1_tile {
                continue;
            }

            // Global x where the segment enters this tile row (at the clipped ey0).
            let x_enter_global = dxdy.mul_add(ey0.max(tile_top) - sy0, sx0);
            // Global x at the exit of this tile row.
            let x_at_exit = dxdy.mul_add(seg_y1_tile - seg_y0_tile, x_enter_global);

            // Tile columns spanned by this segment within the tile row.
            let xl = x_enter_global.min(x_at_exit);
            let xr = x_enter_global.max(x_at_exit);

            // Compute tx0/tx1 as i32 first to handle negative x safely, then
            // clamp to [0, grid_w-1].  A negative tx1 means the segment is
            // entirely left of the raster — skip the whole tile row.
            // xl/xr are f32 page coordinates; floor→i32 is safe for any realistic page width.
            #[expect(
                clippy::cast_possible_truncation,
                reason = "floor(f32) for page-coordinate x; realistic page widths fit comfortably in i32"
            )]
            let tx0_i = (xl / TILE_W_F).floor() as i32;
            #[expect(
                clippy::cast_possible_truncation,
                reason = "floor(f32) for page-coordinate x; realistic page widths fit comfortably in i32"
            )]
            let tx1_i = (xr / TILE_W_F).floor() as i32;
            if tx1_i < 0 {
                continue;
            }
            // tx0_i.max(0) is non-negative — safe to cast to u32.
            #[expect(clippy::cast_sign_loss, reason = "tx0_i.max(0) ≥ 0 by construction")]
            let tx0 = tx0_i.max(0) as u32;
            // tx1_i ≥ 0 checked above (continue if < 0) — safe to cast to u32.
            #[expect(
                clippy::cast_sign_loss,
                reason = "tx1_i ≥ 0 verified by the guard above"
            )]
            let tx1 = (tx1_i as u32).min(grid_w - 1);

            for tx in tx0..=tx1 {
                // tx ≤ grid_w-1 ≤ 0xFFFE, TILE_W_F = 16.0; product ≤ ~1M — exact in f32.
                #[expect(
                    clippy::cast_precision_loss,
                    reason = "tx * TILE_W ≤ 0xFFFE*16 ≈ 1M; exact in f32"
                )]
                records.push(TileRecord {
                    key: (ty << 16) | tx,
                    // tile-local x: subtract this tile column's left edge.
                    x_enter: x_enter_global - (tx * TILE_W) as f32,
                    dxdy,
                    y0_tile: seg_y0_tile,
                    y1_tile: seg_y1_tile,
                    sign,
                    _pad: 0,
                    _pad2: 0,
                });
            }
        }
    }

    // Sort records by (tile_y, tile_x) key — CPU sort is faster than CUB radix
    // sort for typical PDF segment counts (O(100–1000) records).
    records.sort_unstable_by_key(|r| r.key);

    // Build exclusive prefix-sum index: tile_starts[i] = first record index for tile i.
    let mut tile_starts = vec![0u32; n_tiles];
    let mut tile_counts = vec![0u32; n_tiles];

    for rec in &records {
        let tile_idx = ((rec.key >> 16) * grid_w + (rec.key & 0xFFFF)) as usize;
        // tile_idx is always < n_tiles: key components are clamped to [0, grid_w/h-1]
        // at record-build time above.
        debug_assert!(tile_idx < n_tiles, "record key out of tile grid range");
        if tile_idx < n_tiles {
            tile_counts[tile_idx] += 1;
        }
    }

    let mut running = 0u32;
    for (start, count) in tile_starts.iter_mut().zip(tile_counts.iter()) {
        *start = running;
        running = running
            .checked_add(*count)
            .expect("tile record count overflows u32");
    }

    (records, tile_starts, tile_counts, grid_w)
}

/// Halton(2) jitter X offsets within [0,1) for 64-sample AA.
const HALTON2: [f32; 64] = [
    0.5,
    0.25,
    0.75,
    0.125,
    0.625,
    0.375,
    0.875,
    0.062_5,
    0.562_5,
    0.312_5,
    0.812_5,
    0.187_5,
    0.687_5,
    0.437_5,
    0.937_5,
    0.031_25,
    0.531_25,
    0.281_25,
    0.781_25,
    0.156_25,
    0.656_25,
    0.406_25,
    0.906_25,
    0.093_75,
    0.593_75,
    0.343_75,
    0.843_75,
    0.218_75,
    0.718_75,
    0.468_75,
    0.968_75,
    0.015_625,
    0.515_625,
    0.265_625,
    0.765_625,
    0.140_625,
    0.640_625,
    0.390_625,
    0.890_625,
    0.078_125,
    0.578_125,
    0.328_125,
    0.828_125,
    0.203_125,
    0.703_125,
    0.453_125,
    0.953_125,
    0.046_875,
    0.546_875,
    0.296_875,
    0.796_875,
    0.171_875,
    0.671_875,
    0.421_875,
    0.921_875,
    0.109_375,
    0.609_375,
    0.359_375,
    0.859_375,
    0.234_375,
    0.734_375,
    0.484_375,
    0.984_375,
    0.007_812_5,
];

/// Halton(3) jitter Y offsets within [0,1) for 64-sample AA.
const HALTON3: [f32; 64] = [
    0.333_333, 0.666_667, 0.111_111, 0.444_444, 0.777_778, 0.222_222, 0.555_556, 0.888_889,
    0.037_037, 0.370_370, 0.703_704, 0.148_148, 0.481_481, 0.814_815, 0.259_259, 0.592_593,
    0.925_926, 0.012_346, 0.345_679, 0.679_012, 0.123_457, 0.456_790, 0.790_123, 0.234_568,
    0.567_901, 0.901_235, 0.049_383, 0.382_716, 0.716_049, 0.160_494, 0.493_827, 0.827_160,
    0.271_605, 0.604_938, 0.938_272, 0.004_115, 0.337_449, 0.670_782, 0.115_226, 0.448_560,
    0.781_893, 0.226_337, 0.559_671, 0.893_004, 0.041_152, 0.374_486, 0.707_819, 0.152_263,
    0.485_597, 0.818_930, 0.263_374, 0.596_708, 0.930_041, 0.074_074, 0.407_407, 0.740_741,
    0.185_185, 0.518_519, 0.851_852, 0.296_296, 0.629_630, 0.962_963, 0.008_230, 0.341_564,
];

/// CPU fallback for `aa_fill` using 64-sample Halton jitter per pixel.
///
/// Matches the GPU kernel's coverage computation exactly: same Halton(2,3)
/// sample offsets, same winding-number / even-odd logic, same scale formula.
/// Used when `n_pixels < GPU_AA_FILL_THRESHOLD` or when no CUDA device is present.
#[must_use]
pub fn aa_fill_cpu(
    segs: &[f32],
    x_min: f32,
    y_min: f32,
    width: u32,
    height: u32,
    eo: bool,
) -> Vec<u8> {
    let n_pixels = width as usize * height as usize;
    let mut out = vec![0u8; n_pixels];

    for py in 0..height {
        for px in 0..width {
            #[expect(
                clippy::cast_precision_loss,
                reason = "px/py ≤ width/height ≤ u32::MAX; at typical DPIs (≤ 32768 px) \
                          the f32 precision loss is sub-pixel and irrelevant for AA coverage"
            )]
            let (cx, cy) = (x_min + px as f32 + 0.5, y_min + py as f32 + 0.5);
            let mut hits = 0u32;
            for s in 0..64usize {
                let sx = cx + HALTON2[s] - 0.5;
                let sy = cy + HALTON3[s] - 0.5;
                if aa_fill_cpu_sample(segs, sx, sy, eo) {
                    hits += 1;
                }
            }
            #[expect(
                clippy::cast_possible_truncation,
                reason = "hits ≤ 64; (64*255+32)>>6 = 255 — always fits u8"
            )]
            {
                out[py as usize * width as usize + px as usize] = ((hits * 255 + 32) >> 6) as u8;
            }
        }
    }
    out
}

fn aa_fill_cpu_sample(segs: &[f32], sx: f32, sy: f32, eo: bool) -> bool {
    let mut winding = 0i32;
    for seg in segs.chunks_exact(4) {
        let (x0, y0, x1, y1) = (seg[0], seg[1], seg[2], seg[3]);
        if y0 <= sy && sy < y1 {
            let t = (sy - y0) / (y1 - y0);
            let xi = t.mul_add(x1 - x0, x0);
            if xi >= sx {
                winding += 1;
            }
        } else if y1 <= sy && sy < y0 {
            let t = (sy - y1) / (y0 - y1);
            let xi = t.mul_add(x0 - x1, x1);
            if xi >= sx {
                winding -= 1;
            }
        }
    }
    if eo { (winding & 1) != 0 } else { winding != 0 }
}

#[cfg(test)]
mod tests {
    use super::{aa_fill_cpu, build_tile_records};

    #[test]
    fn aa_fill_cpu_solid_rect_full_coverage() {
        let segs: Vec<f32> = vec![
            -100.0, -100.0, 100.0, -100.0, 100.0, -100.0, 100.0, 100.0, 100.0, 100.0, -100.0,
            100.0, -100.0, 100.0, -100.0, -100.0,
        ];
        let cov = aa_fill_cpu(&segs, 0.0, 0.0, 1, 1, false);
        assert_eq!(cov.len(), 1);
        assert_eq!(cov[0], 255, "fully covered pixel should be 255");
    }

    #[test]
    fn aa_fill_cpu_outside_rect_zero_coverage() {
        let segs: Vec<f32> = vec![
            0.0, 0.0, 10.0, 0.0, 10.0, 0.0, 10.0, 10.0, 10.0, 10.0, 0.0, 10.0, 0.0, 10.0, 0.0, 0.0,
        ];
        let cov = aa_fill_cpu(&segs, 200.0, 200.0, 1, 1, false);
        assert_eq!(cov[0], 0, "pixel outside rect should be 0");
    }

    #[test]
    fn aa_fill_cpu_empty_segs_zero_coverage() {
        let cov = aa_fill_cpu(&[], 0.0, 0.0, 4, 4, false);
        assert_eq!(cov.len(), 16);
        assert!(
            cov.iter().all(|&v| v == 0),
            "empty segs → all zero coverage"
        );
    }

    #[test]
    fn aa_fill_cpu_eo_donut_inner_zero() {
        let outer: [f32; 16] = [
            -10.0, -10.0, 10.0, -10.0, 10.0, -10.0, 10.0, 10.0, 10.0, 10.0, -10.0, 10.0, -10.0,
            10.0, -10.0, -10.0,
        ];
        let inner: [f32; 16] = [
            -5.0, -5.0, 5.0, -5.0, 5.0, -5.0, 5.0, 5.0, 5.0, 5.0, -5.0, 5.0, -5.0, 5.0, -5.0, -5.0,
        ];
        let segs: Vec<f32> = outer.iter().chain(inner.iter()).copied().collect();
        let cov = aa_fill_cpu(&segs, -0.5, -0.5, 1, 1, true);
        assert_eq!(cov[0], 0, "EO donut centre should be 0");
    }

    #[test]
    fn tile_records_empty_segs() {
        let (recs, starts, counts, grid_w) = build_tile_records(&[], 0.0, 0.0, 32, 32);
        assert!(recs.is_empty());
        assert_eq!(grid_w, 2);
        assert!(starts.iter().all(|&s| s == 0));
        assert!(counts.iter().all(|&c| c == 0));
    }

    #[test]
    fn tile_records_single_vertical_segment() {
        // Segment from (8,0) to (8,17): crosses tile rows 0 (y 0..16) and 1 (y 16..17).
        let segs = [8.0f32, 0.0, 8.0, 17.0];
        let (recs, _, _, _) = build_tile_records(&segs, 0.0, 0.0, 64, 64);
        assert_eq!(recs.len(), 2, "one record per tile row crossed");
    }

    #[test]
    fn tile_records_diagonal_segment() {
        let segs = [0.0f32, 0.0, 32.0, 32.0];
        let (recs, _, _, _) = build_tile_records(&segs, 0.0, 0.0, 128, 128);
        assert!(recs.len() >= 2, "diagonal must produce at least 2 records");
    }

    #[test]
    fn tile_records_sort_order() {
        let segs = [24.0f32, 0.0, 24.0, 8.0, 8.0f32, 16.0, 8.0, 24.0];
        let (recs, starts, counts, grid_w) = build_tile_records(&segs, 0.0, 0.0, 80, 80);
        assert_eq!(grid_w, 5);
        for w in recs.windows(2) {
            assert!(w[0].key <= w[1].key, "records must be sorted by key");
        }
        let tile_01 = 0 * grid_w as usize + 1;
        let tile_10 = 1 * grid_w as usize + 0;
        assert_eq!(counts[tile_01], 1);
        assert_eq!(counts[tile_10], 1);
        let _ = starts;
    }

    #[test]
    fn tile_records_prefix_sum_consistent() {
        let segs = [4.0f32, 0.0, 60.0, 63.0];
        let (recs, starts, counts, grid_w) = build_tile_records(&segs, 0.0, 0.0, 64, 64);
        let _ = grid_w;
        let total: u32 = counts.iter().sum();
        assert_eq!(total as usize, recs.len(), "sum of counts == total records");
        for i in 0..counts.len() - 1 {
            assert_eq!(
                starts[i] + counts[i],
                starts[i + 1],
                "prefix sum broken at {i}"
            );
        }
    }
}
