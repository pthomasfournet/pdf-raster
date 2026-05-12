// Parallel-Huffman JPEG decoder kernels — CUDA mirror of
// parallel_huffman.slang. See the Slang for algorithm-level comments.
//
// Block size is the grid-dim argument supplied at launch by
// `backend::params::HUFFMAN_PHASE1_THREADS`; the kernel itself reads
// threadIdx via the standard CUDA intrinsics and doesn't pin the
// block size in source.

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

// Try to decode one Huffman symbol starting at `*p`, updating
// `(*p, *n, *c, *z)` in place. Returns 1 on success (symbol decoded
// + state advanced), 0 on either failure mode: prefix-miss (no
// codeword matches the 16-bit peek) or length-bits truncation
// (codeword would run past stream end). On failure, state is
// unchanged so callers can detect the miss without checkpoint
// arithmetic.
//
// Mirrors phase1_oracle::try_decode_one_symbol on the host side;
// shared by both phase1_intra_sync (loops until hard_limit) and
// phase2_inter_sync (calls once per unsynced subseq per pass).
__device__ __forceinline__ unsigned int try_decode_one_symbol_device(
    const unsigned int* __restrict__ bitstream,
    const unsigned int* __restrict__ codebook,
    unsigned int length_bits,
    unsigned int num_components,
    unsigned int* p,
    unsigned int* n,
    unsigned int* c,
    unsigned int* z
) {
    unsigned int peek = peek16_kernel(bitstream, *p);
    unsigned int entry = codebook[(*c) * CODEBOOK_ENTRIES + peek];
    unsigned int num_bits = (entry >> 8u) & 0xFFu;
    if (num_bits == 0u) return 0u;  // PrefixMiss
    unsigned int symbol = entry & 0xFFu;
    unsigned int value_bits = symbol & 0x0Fu;
    unsigned int advance = num_bits + value_bits;
    if (*p + advance > length_bits) return 0u;  // LengthBits

    *p += advance;
    *n += 1u;
    // z is a 6-bit zig-zag index; mask is one PTX instruction.
    *z = (*z + 1u) & 63u;
    if (*z == 0u) {
        *c = (*c + 1u) % num_components;
    }
    return 1u;
}

// Phase 1: one thread per subsequence; writes (p, n, c, z) to
// s_info_out[seq_idx]. s_info_out is laid out as uint4 = (p, n, c, z)
// per subsequence.
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
        if (try_decode_one_symbol_device(
                bitstream, codebook, length_bits, num_components,
                &p, &n, &c, &z) == 0u) {
            break;
        }
    }

    uint4 out;
    out.x = p;
    out.y = n;
    out.z = c;
    out.w = z;
    s_info_out[seq_idx] = out;
}

// Phase 2: one thread per subsequence. Checks whether `s_info[i]`
// is "synced" with `s_info[i+1]` (same c and z, and i's walk reached
// i+1's start). If synced, writes flags[i] = 1. Otherwise advances
// s_info[i] by one symbol and writes flags[i] = 0. The host loops
// the dispatch until every flag is 1 or the retry bound is exhausted.
//
// Note: the last subsequence (seq_idx + 1 == num_subsequences) has
// no right neighbour, so it's trivially synced — flags[i] = 1, no
// advance.
extern "C" __global__ void phase2_inter_sync(
    const unsigned int* __restrict__ bitstream,
    const unsigned int* __restrict__ codebook,
    uint4* __restrict__ s_info,
    unsigned int* __restrict__ flags,
    unsigned int length_bits,
    unsigned int subsequence_bits,
    unsigned int num_subsequences,
    unsigned int num_components
) {
    unsigned int seq_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (seq_idx >= num_subsequences) {
        return;
    }
    // Last subseq has no right neighbour — trivially synced.
    if (seq_idx + 1u == num_subsequences) {
        flags[seq_idx] = 1u;
        return;
    }

    // SubsequenceState is (p, n, c, z) in declaration order, mapped
    // to uint4's (x, y, z, w). The struct-field z (zig-zag index)
    // lands in uint4.w; the struct-field c (component index) lands
    // in uint4.z. Naming overlap aside: .x=p, .y=n, .z=c, .w=z.
    uint4 me = s_info[seq_idx];
    uint4 nxt = s_info[seq_idx + 1];
    unsigned int me_c = me.z;
    unsigned int me_z = me.w;
    unsigned int nxt_c = nxt.z;
    unsigned int nxt_z = nxt.w;
    unsigned int nxt_start_p = (seq_idx + 1u) * subsequence_bits;

    unsigned int in_range = (me.x >= nxt_start_p) ? 1u : 0u;
    unsigned int aligned = (me_c == nxt_c && me_z == nxt_z) ? 1u : 0u;

    if (in_range && aligned) {
        flags[seq_idx] = 1u;
        return;
    }

    // Not synced — advance me by one symbol, write back, mark unsynced.
    unsigned int p = me.x;
    unsigned int n = me.y;
    unsigned int c = me_c;
    unsigned int z = me_z;
    // Advance is permitted to fail (PrefixMiss / LengthBits); the
    // state stays put and we'll spin in this subseq until the bound.
    (void)try_decode_one_symbol_device(
        bitstream, codebook, length_bits, num_components,
        &p, &n, &c, &z);

    uint4 updated;
    updated.x = p;
    updated.y = n;
    updated.z = c;
    updated.w = z;
    s_info[seq_idx] = updated;
    flags[seq_idx] = 0u;
}
