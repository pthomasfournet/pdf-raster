// Parallel-Huffman JPEG decoder kernels — CUDA mirror of
// parallel_huffman.slang. See the Slang for algorithm-level comments.

#define BLOCK_SIZE 256u
#define CODEBOOK_ENTRIES 65536u

// Peek 16 bits MSB-first starting at absolute `bit_pos`. Mirrors
// the host-side bitstream::peek16 exactly.
__device__ __forceinline__ unsigned int peek16_kernel(
    const unsigned int* __restrict__ bitstream,
    unsigned int bit_pos
) {
    unsigned int word_idx = bit_pos / 32u;
    unsigned int bit_in_word = bit_pos & 31u;

    unsigned long long hi = bitstream[word_idx];
    unsigned long long lo = bitstream[word_idx + 1];
    unsigned long long combined = (hi << 32) | lo;
    return (unsigned int)((combined >> (48u - bit_in_word)) & 0xFFFFu);
}

// One thread per subsequence; writes (p, n, c, z) to s_info_out[seq_idx].
// s_info_out is laid out as uint4 = (p, n, c, z) per subsequence.
extern "C" __global__ void phase1_intra_sync(
    const unsigned int* __restrict__ bitstream,
    const unsigned int* __restrict__ codebook,
    uint4* __restrict__ s_info_out,
    unsigned int length_bits,
    unsigned int subsequence_bits,
    unsigned int num_subsequences,
    unsigned int num_components
) {
    unsigned int seq_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (seq_idx >= num_subsequences) {
        return;
    }

    unsigned int start_bit = seq_idx * subsequence_bits;
    unsigned int sync_target = start_bit + 2u * subsequence_bits;
    unsigned int hard_limit = min(length_bits, sync_target);

    unsigned int p = start_bit;
    unsigned int n = 0u;
    unsigned int c = 0u;
    unsigned int z = 0u;

    unsigned int max_iters = (hard_limit - start_bit) + 1u;
    for (unsigned int iter = 0u; iter < max_iters; iter++) {
        if (p >= hard_limit) break;
        unsigned int peek = peek16_kernel(bitstream, p);
        unsigned int entry = codebook[c * CODEBOOK_ENTRIES + peek];
        unsigned int num_bits = (entry >> 8u) & 0xFFu;
        if (num_bits == 0u) break;
        unsigned int symbol = entry & 0xFFu;
        unsigned int value_bits = symbol & 0x0Fu;
        unsigned int advance = num_bits + value_bits;
        if (p + advance > length_bits) break;
        p += advance;
        n += 1u;
        // z is a 6-bit zig-zag index; mask is one PTX instruction
        // vs the % rem.u32 the compiler might or might not fold.
        z = (z + 1u) & 63u;
        if (z == 0u) {
            c = (c + 1u) % num_components;
        }
    }

    uint4 out;
    out.x = p;
    out.y = n;
    out.z = c;
    out.w = z;
    s_info_out[seq_idx] = out;
}
