// Multi-workgroup Blelloch exclusive scan over u32 counts.
// CUDA mirror of blelloch_scan.slang; algorithm: Blelloch 1990.
//
// Three __global__ entry points, dispatched separately by the host.
// All share a common __device__ helper that runs the in-shared-memory
// scan over a 1024-element tile (512 threads, 2 elements per thread).

#define TILE_SIZE 1024u
#define TILE_THREADS 512u  // numthreads.x; tile = 2 * TILE_THREADS

// In-place Blelloch exclusive scan over `tile` (TILE_SIZE elements).
// Returns the tile total via `out_total` (valid only for thread 0).
__device__ __forceinline__ void blelloch_in_shared(
    unsigned int tid,
    unsigned int* tile,
    unsigned int* out_total
) {
    // Upsweep: balanced tree summation.
    unsigned int stride = 1;
    #pragma unroll
    for (unsigned int d = TILE_SIZE >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            unsigned int ai = stride * (2 * tid + 1) - 1;
            unsigned int bi = stride * (2 * tid + 2) - 1;
            tile[bi] = tile[bi] + tile[ai];
        }
        stride <<= 1;
    }

    // Capture total + clear identity slot.
    __syncthreads();
    if (tid == 0) {
        *out_total = tile[TILE_SIZE - 1];
        tile[TILE_SIZE - 1] = 0u;
    }

    // Downsweep.
    #pragma unroll
    for (unsigned int d = 1; d < TILE_SIZE; d <<= 1) {
        stride >>= 1;
        __syncthreads();
        if (tid < d) {
            unsigned int ai = stride * (2 * tid + 1) - 1;
            unsigned int bi = stride * (2 * tid + 2) - 1;
            unsigned int t = tile[ai];
            tile[ai] = tile[bi];
            tile[bi] = tile[bi] + t;
        }
    }
    __syncthreads();
}

extern "C" __global__ void scan_per_workgroup(
    unsigned int* __restrict__ data,
    unsigned int* __restrict__ block_sums,
    unsigned int len_elems
) {
    __shared__ unsigned int tile[TILE_SIZE];

    unsigned int tid = threadIdx.x;
    unsigned int gid = blockIdx.x;
    unsigned int base = gid * TILE_SIZE;

    unsigned int g0 = base + 2 * tid;
    unsigned int g1 = base + 2 * tid + 1;
    tile[2 * tid]     = (g0 < len_elems) ? data[g0] : 0u;
    tile[2 * tid + 1] = (g1 < len_elems) ? data[g1] : 0u;
    __syncthreads();

    unsigned int tile_total = 0u;
    blelloch_in_shared(tid, tile, &tile_total);

    if (tid == 0) {
        block_sums[gid] = tile_total;
    }
    if (g0 < len_elems) data[g0] = tile[2 * tid];
    if (g1 < len_elems) data[g1] = tile[2 * tid + 1];
}

extern "C" __global__ void scan_block_sums(
    unsigned int* __restrict__ block_sums,
    unsigned int len_elems
) {
    __shared__ unsigned int tile[TILE_SIZE];

    unsigned int tid = threadIdx.x;
    unsigned int n_blocks = (len_elems + TILE_SIZE - 1u) / TILE_SIZE;

    unsigned int local0 = 2 * tid;
    unsigned int local1 = 2 * tid + 1;
    tile[local0] = (local0 < n_blocks) ? block_sums[local0] : 0u;
    tile[local1] = (local1 < n_blocks) ? block_sums[local1] : 0u;
    __syncthreads();

    unsigned int discarded = 0u;
    blelloch_in_shared(tid, tile, &discarded);

    if (local0 < n_blocks) block_sums[local0] = tile[local0];
    if (local1 < n_blocks) block_sums[local1] = tile[local1];
}

extern "C" __global__ void scan_scatter(
    unsigned int* __restrict__ data,
    const unsigned int* __restrict__ block_sums,
    unsigned int len_elems
) {
    unsigned int tid = threadIdx.x;
    unsigned int gid = blockIdx.x;
    unsigned int base = gid * TILE_SIZE;
    unsigned int add = block_sums[gid];

    unsigned int g0 = base + 2 * tid;
    unsigned int g1 = base + 2 * tid + 1;
    if (g0 < len_elems) data[g0] = data[g0] + add;
    if (g1 < len_elems) data[g1] = data[g1] + add;
}
