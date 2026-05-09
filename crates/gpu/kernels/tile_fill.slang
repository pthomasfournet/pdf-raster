// GPU tile-parallel analytical fill rasterisation.
// Slang port of tile_fill.cu.
//
// Algorithm: vello-style signed-area integration.  Each path segment is
// "tiled" into one record per tile row it crosses; records are sorted on
// the host by (tile_y, tile_x).  Each tile_starts[i] / tile_counts[i] pair
// tells the kernel where to find records for tile i.  One workgroup per
// tile, TILE_W × TILE_H threads each computing one pixel's coverage.
//
// Tile geometry: TILE_W = 16, TILE_H = 16.

static const uint TILE_W = 16;
static const uint TILE_H = 16;

// Tile record emitted by the host for each (segment, tile-row) pair.
// Layout must match `gpu::TileRecord` in crates/gpu/src/lib.rs (32 bytes).
// Slang struct fields default to natural alignment with std430-like packing
// when used in a StructuredBuffer.
struct TileRecord {
    uint  key;       // (tile_y << 16) | tile_x
    float x_enter;   // segment x at top of this tile's y-extent (tile-local x)
    float dxdy;      // slope dx/dy
    float y0_tile;   // segment start y within tile (0..TILE_H)
    float y1_tile;   // segment end y within tile
    float sign_;     // +1 upward, -1 downward  (`sign` is reserved in HLSL/Slang)
    uint  _pad;
    uint  _pad2;
};

// Signed y-span contribution of one segment to pixel column `px` for the
// pixel row [iy0, iy1] (both in tile-local y coordinates).  See aa_fill /
// tile_fill .cu commentary for the derivation.
float segment_pixel_area(float x_enter, float dxdy,
                         float iy0, float iy1,
                         float sign_arg, float px)
{
    float y_len = iy1 - iy0;
    if (y_len <= 0.0f) return 0.0f;

    float x0 = x_enter;
    float x1 = x_enter + dxdy * y_len;

    float xl = min(x0, x1);
    float xr = max(x0, x1);

    float cover;
    if (xr <= px) {
        cover = 0.0f;
    } else if (xl >= px + 1.0f) {
        cover = y_len;
    } else {
        float dx = xr - xl;
        if (dx < 1e-6f) {
            float xmid = 0.5f * (x0 + x1);
            cover = (xmid >= px + 0.5f) ? y_len : 0.0f;
        } else {
            float left_  = max(xl, px);
            float right_ = min(xr, px + 1.0f);
            float yf_left  = (left_  - xl) / dx * y_len;
            float yf_right = (right_ - xl) / dx * y_len;
            float above = y_len - yf_right;
            cover = (yf_right - yf_left) + above;
        }
    }

    return sign_arg * cover;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void tile_fill(
    uint3 lid : SV_GroupThreadID,
    uint3 wid : SV_GroupID,
    uniform StructuredBuffer<TileRecord> records,
    uniform StructuredBuffer<uint> tile_starts,
    uniform StructuredBuffer<uint> tile_counts,
    uniform uint grid_w,
    uniform uint out_w,
    uniform uint out_h,
    uniform int eo,
    uniform RWStructuredBuffer<uint8_t> coverage)
{
    uint tile_x = wid.x;
    uint tile_y = wid.y;
    uint px_local = lid.x;
    uint py_local = lid.y;

    uint px = tile_x * TILE_W + px_local;
    uint py = tile_y * TILE_H + py_local;
    if (px >= out_w || py >= out_h) return;

    uint tile_idx = tile_y * grid_w + tile_x;
    uint rec_start = tile_starts[tile_idx];
    uint rec_count = tile_counts[tile_idx];

    float area = 0.0f;
    float py_f = float(py_local);

    for (uint r = rec_start; r < rec_start + rec_count; r++) {
        TileRecord rec = records[r];

        float iy0 = max(rec.y0_tile, py_f);
        float iy1 = min(rec.y1_tile, py_f + 1.0f);
        if (iy0 >= iy1) continue;

        float x_at_iy0 = rec.x_enter + rec.dxdy * (iy0 - rec.y0_tile);
        float px_f = float(px_local);

        area += segment_pixel_area(x_at_iy0, rec.dxdy, iy0, iy1, rec.sign_, px_f);
    }

    int cov;
    if (eo != 0) {
        float a = abs(area);
        a = a - floor(a);
        if (a > 0.5f) a = 1.0f - a;
        cov = int(a * 2.0f * 255.5f);
    } else {
        cov = int(min(abs(area), 1.0f) * 255.5f);
    }
    coverage[py * out_w + px] = uint8_t(min(cov, 255));
}
