// GPU tile-parallel analytical fill rasterisation.
//
// Algorithm (vello-style signed-area integration):
//
//   Each path segment is "tiled" into one record per tile row it crosses.
//   Records are sorted by (tile_y, tile_x) by the host (CPU or CUB) before
//   passing to this kernel.  `tile_starts[i]` and `tile_counts[i]` tell the
//   fill kernel where to find records for tile i.
//
//   The fill kernel (one block per tile, TILE_W × TILE_H threads) computes
//   per-pixel signed area using the analytical formula:
//
//     For each record crossing the pixel row [py, py+1]:
//       - Clip segment to the pixel row.
//       - Compute the signed trapezoidal area swept across pixel column px.
//
//   Non-zero winding: coverage = min(|area|, 1) × 255.
//   Even-odd: coverage = (1 - |frac(area) - 0.5| × 2) × 255.
//
// Tile geometry:
//   TILE_W = 16 pixels wide,  TILE_H = 16 pixels tall.
//   Grid: ceil(out_w / TILE_W) × ceil(out_h / TILE_H).

#define TILE_W 16
#define TILE_H 16

// Tile record emitted by the host for each (segment, tile-row) pair.
// Must match the layout in `gpu/src/lib.rs :: TileRecord`.
struct TileRecord {
    unsigned int key;   // (tile_y << 16) | tile_x — sort key (host fills this)
    float x_enter;     // segment x at top of this tile's y-extent (in tile-local coords)
    float dxdy;        // slope dx/dy (device pixels)
    float y0_tile;     // segment start y within tile (0..TILE_H)
    float y1_tile;     // segment end y within tile   (0..TILE_H)
    float sign;        // +1 upward-crossing, -1 downward
    unsigned int _pad; // pad to 32 bytes
    unsigned int _pad2;
};

// Compute the signed area contribution of one segment to pixel column `px`
// for the pixel row [py_local, py_local + 1], where the segment is clipped to
// [iy0, iy1] (both in tile-local y coordinates).
//
// The segment enters at (x_enter, iy0) with slope dxdy (pixels per pixel).
//
// Returns sign × (area in [px, px+1] × [iy0, iy1]).
__device__ float segment_pixel_area(float x_enter, float dxdy,
                                    float iy0, float iy1,
                                    float sign, float px)
{
    float y_len = iy1 - iy0;
    if (y_len <= 0.0f) return 0.0f;

    float x0 = x_enter;
    float x1 = x_enter + dxdy * y_len;

    // Left/right x bounds of the segment in this pixel-row interval.
    float xl = fminf(x0, x1);
    float xr = fmaxf(x0, x1);

    // Pixel column spans [px, px + 1].
    // Area swept = y_len × (fraction of pixel column covered by the segment).
    // We compute the intersection of the linearly-varying x with [px, px+1].

    float cover;
    if (xr <= px) {
        // Segment entirely to the left: no area in this pixel column.
        cover = 0.0f;
    } else if (xl >= px + 1.0f) {
        // Segment entirely to the right: full y_len for winding.
        cover = y_len;
    } else {
        float dx = xr - xl;
        if (dx < 1e-6f) {
            // Near-vertical: step function at the midpoint.
            float xmid = 0.5f * (x0 + x1);
            cover = (xmid >= px + 0.5f) ? y_len : 0.0f;
        } else {
            // Clip [xl, xr] to [px, px+1].
            float left  = fmaxf(xl, px);
            float right = fminf(xr, px + 1.0f);

            // y fractions where x(y) = left and x(y) = right.
            // x(y) = xl + (xr - xl) * (y - y0_interval) / y_len
            // Solving: y_frac_left = (left - xl) / dx * y_len
            float yf_left  = (left  - xl) / dx * y_len;
            float yf_right = (right - xl) / dx * y_len;

            // Trapezoidal area: average height × width.
            float trap_area = 0.5f * (yf_left + yf_right) * (right - left);

            // Region where x > px+1 (fully to the right of the pixel): y_len - yf_right.
            float above = y_len - yf_right;
            // Total: trapezoid covering the partial pixel + full-pixel region.
            // Width is normalised to 1 (the pixel column width).
            cover = trap_area / 1.0f + above * 1.0f;
        }
    }

    return sign * cover;
}

// Tile fill kernel.
//
// Grid: (grid_w, grid_h, 1), Block: (TILE_W, TILE_H, 1).
//
// Parameters:
//   records     : tile records sorted by (tile_y << 16 | tile_x)
//   tile_starts : start index of records for each flat tile index
//   tile_counts : number of records for each flat tile index
//   grid_w      : number of tiles in x direction
//   x_min       : fill bbox left in device pixels
//   y_min       : fill bbox top in device pixels
//   out_w       : output coverage buffer width (pixels)
//   out_h       : output coverage buffer height (pixels)
//   eo          : 1 = even-odd, 0 = non-zero winding
//   coverage    : output, out_w × out_h bytes
extern "C" __global__ void tile_fill(
    const TileRecord* __restrict__ records,
    const unsigned int* __restrict__ tile_starts,
    const unsigned int* __restrict__ tile_counts,
    unsigned int grid_w,
    float x_min, float y_min,
    unsigned int out_w, unsigned int out_h,
    int eo,
    unsigned char* __restrict__ coverage
) {
    unsigned int tile_x = blockIdx.x;
    unsigned int tile_y = blockIdx.y;
    unsigned int px_local = threadIdx.x; // 0..TILE_W-1
    unsigned int py_local = threadIdx.y; // 0..TILE_H-1

    unsigned int px = tile_x * TILE_W + px_local;
    unsigned int py = tile_y * TILE_H + py_local;
    if (px >= out_w || py >= out_h) return;

    unsigned int tile_idx = tile_y * grid_w + tile_x;
    unsigned int rec_start = tile_starts[tile_idx];
    unsigned int rec_count = tile_counts[tile_idx];

    float area = 0.0f;
    float py_f = (float)py_local;

    for (unsigned int r = rec_start; r < rec_start + rec_count; r++) {
        TileRecord rec = records[r];

        // Clip segment to pixel row [py_local, py_local + 1].
        float iy0 = fmaxf(rec.y0_tile, py_f);
        float iy1 = fminf(rec.y1_tile, py_f + 1.0f);
        if (iy0 >= iy1) continue;

        // x of segment at iy0 (adjusted for how far into the tile we've clipped).
        float x_at_iy0 = rec.x_enter + rec.dxdy * (iy0 - rec.y0_tile);

        // Pixel column position: x_enter is relative to tile left.
        float px_f = (float)px_local;

        area += segment_pixel_area(x_at_iy0, rec.dxdy, iy0, iy1, rec.sign, px_f);
    }

    int cov;
    if (eo) {
        float a = fabsf(area);
        a = a - floorf(a);
        if (a > 0.5f) a = 1.0f - a;
        cov = (int)(a * 2.0f * 255.5f);
    } else {
        cov = (int)(fminf(fabsf(area), 1.0f) * 255.5f);
    }
    coverage[py * out_w + px] = (unsigned char)min(cov, 255);
}
