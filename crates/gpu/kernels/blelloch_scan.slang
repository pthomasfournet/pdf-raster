// Multi-workgroup Blelloch exclusive scan over u32 counts.
//
// Three entry points, dispatched separately by the host:
//   scan_per_workgroup  — each workgroup scans its 1024-element tile
//                         in shared memory, writes the tile sum to
//                         block_sums[workgroup_id].
//   scan_block_sums     — single workgroup scans the block_sums array
//                         in place (≤ SCAN_MAX_BLOCKS blocks).
//   scan_scatter        — each workgroup adds its now-scanned
//                         block_sums[workgroup_id] back into every
//                         element of its tile.
//
// Algorithm: Blelloch 1990, "Prefix Sums and Their Applications".
// SCAN_WORKGROUP_SIZE = 512 threads per workgroup; each thread loads
// + processes 2 elements, so the tile size is 1024.

// Binding 0: in/out data array (length len_elems).
[[vk::binding(0, 0)]] RWStructuredBuffer<uint> data;
// Binding 1: per-workgroup tile sums (length ceil(len_elems / 1024)).
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> block_sums;

[[vk::push_constant]]
cbuffer ScanPush {
    // Total element count; threads past this index are inactive.
    uint len_elems;
};

// Shared-memory working set for both upsweep and downsweep.
// Must match TILE_SIZE = 2 * SCAN_WORKGROUP_SIZE below.
groupshared uint shared_tile[1024];

// Per-workgroup tile size in elements (must equal 2 * numthreads.x).
static const uint TILE_SIZE = 1024;

// Load up to 2 elements per thread from `data` into `shared_tile`.
// Threads past `len_elems` load zero so the upsweep is identity.
void load_tile(uint tid, uint gid, uint base) {
    uint local0 = 2 * tid;
    uint local1 = 2 * tid + 1;
    uint global0 = base + local0;
    uint global1 = base + local1;
    shared_tile[local0] = (global0 < len_elems) ? data[global0] : 0u;
    shared_tile[local1] = (global1 < len_elems) ? data[global1] : 0u;
}

// Store the tile back to global `data`. Threads past `len_elems` skip
// so we don't write zeros over the tail of the previous tile.
void store_tile(uint tid, uint gid, uint base) {
    uint local0 = 2 * tid;
    uint local1 = 2 * tid + 1;
    uint global0 = base + local0;
    uint global1 = base + local1;
    if (global0 < len_elems) data[global0] = shared_tile[local0];
    if (global1 < len_elems) data[global1] = shared_tile[local1];
}

// In-place Blelloch exclusive scan over `shared_tile` (1024 elements,
// 512 threads). Optionally returns the tile sum (the value cleared
// from position TILE_SIZE-1 between upsweep and downsweep).
//
// `out_total` receives the tile sum only when called by thread 0 of
// the workgroup, via the caller's check.
void blelloch_in_shared(uint tid, out uint tile_total) {
    // Upsweep: balanced tree summation.
    // After stride s, thread t at index (s * (2t+1) - 1) holds the
    // partial sum of its left-child and right-child.
    uint stride = 1;
    [unroll(10)]
    for (uint d = TILE_SIZE >> 1; d > 0; d >>= 1) {
        GroupMemoryBarrierWithGroupSync();
        if (tid < d) {
            uint ai = stride * (2 * tid + 1) - 1;
            uint bi = stride * (2 * tid + 2) - 1;
            shared_tile[bi] = shared_tile[bi] + shared_tile[ai];
        }
        stride <<= 1;
    }

    // Save total + clear last slot (identity for exclusive scan).
    GroupMemoryBarrierWithGroupSync();
    if (tid == 0) {
        tile_total = shared_tile[TILE_SIZE - 1];
        shared_tile[TILE_SIZE - 1] = 0u;
    } else {
        tile_total = 0u;
    }

    // Downsweep: reverse of upsweep; swap (a, b) → (b, a+b).
    [unroll(10)]
    for (uint d = 1; d < TILE_SIZE; d <<= 1) {
        stride >>= 1;
        GroupMemoryBarrierWithGroupSync();
        if (tid < d) {
            uint ai = stride * (2 * tid + 1) - 1;
            uint bi = stride * (2 * tid + 2) - 1;
            uint t = shared_tile[ai];
            shared_tile[ai] = shared_tile[bi];
            shared_tile[bi] = shared_tile[bi] + t;
        }
    }
    GroupMemoryBarrierWithGroupSync();
}

[shader("compute")][numthreads(512, 1, 1)]
void scan_per_workgroup(
    uint3 tid : SV_GroupThreadID,
    uint3 gid : SV_GroupID
) {
    uint base = gid.x * TILE_SIZE;
    load_tile(tid.x, gid.x, base);
    GroupMemoryBarrierWithGroupSync();

    uint tile_total;
    blelloch_in_shared(tid.x, tile_total);

    if (tid.x == 0) {
        block_sums[gid.x] = tile_total;
    }
    store_tile(tid.x, gid.x, base);
}

// Single-workgroup scan over block_sums. Tile size matches the
// per-workgroup scan (1024 elements / 512 threads), which is the
// SCAN_MAX_BLOCKS limit the host validates.
[shader("compute")][numthreads(512, 1, 1)]
void scan_block_sums(
    uint3 tid : SV_GroupThreadID
) {
    uint local0 = 2 * tid.x;
    uint local1 = 2 * tid.x + 1;

    // Load block_sums into shared_tile. block_sums has at most
    // SCAN_MAX_BLOCKS = TILE_SIZE entries; missing tail is zero.
    uint n_blocks = (len_elems + TILE_SIZE - 1) / TILE_SIZE;
    shared_tile[local0] = (local0 < n_blocks) ? block_sums[local0] : 0u;
    shared_tile[local1] = (local1 < n_blocks) ? block_sums[local1] : 0u;
    GroupMemoryBarrierWithGroupSync();

    uint discarded_total;
    blelloch_in_shared(tid.x, discarded_total);

    if (local0 < n_blocks) block_sums[local0] = shared_tile[local0];
    if (local1 < n_blocks) block_sums[local1] = shared_tile[local1];
}

// Final phase: add scanned block_sums[gid] back to every element of
// the corresponding tile in data.
[shader("compute")][numthreads(512, 1, 1)]
void scan_scatter(
    uint3 tid : SV_GroupThreadID,
    uint3 gid : SV_GroupID
) {
    uint base = gid.x * TILE_SIZE;
    uint add = block_sums[gid.x];

    uint i0 = base + 2 * tid.x;
    uint i1 = base + 2 * tid.x + 1;
    if (i0 < len_elems) data[i0] = data[i0] + add;
    if (i1 < len_elems) data[i1] = data[i1] + add;
}
