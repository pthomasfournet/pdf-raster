// Parallel-Huffman JPEG decoder kernels.
//
// ## Phase 1 algorithm (Weißenberger & Schmidt 2021 §III-B)
//
// One thread per subsequence. Thread `i` starts at bit
// `i * subsequence_bits`, walks the entropy-coded stream symbol-by-
// symbol, advancing a 4-tuple state `(p, n, c, z)` until reaching a
// `hard_limit` (= `min(length_bits, (i + 2) * subsequence_bits)`).
// The kernel then writes the final state to `s_info_out[i]`.
//
// ## Codebook layout
//
// `codebook` is a `u32[num_components * 65 536]` flat buffer. Each
// entry packs `(num_bits << 8) | symbol`. Thread reads entry at
// `codebook[c * 65 536 + peek16]`. `num_bits == 0` = no match
// (the kernel breaks the loop; matches `CanonicalCodebook`'s sentinel).
//
// ## MVP simplification (matches phase1_oracle::phase1_walk)
//
// Real JPEG AC framing tracks (run, size) per symbol. The MVP advances
// `z += 1` per decoded symbol, rolling over `c` on the 64-symbol
// boundary. Production-grade AC framing is a follow-up.

[[vk::binding(0, 0)]] StructuredBuffer<uint> bitstream;
[[vk::binding(1, 0)]] StructuredBuffer<uint> codebook;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint4> s_info_out;

[[vk::push_constant]]
cbuffer PushParams {
    uint length_bits;
    uint subsequence_bits;
    uint num_subsequences;
    uint num_components;
};

/// Peek 16 bits MSB-first starting at absolute `bit_pos`.
/// Matches the host-side `bitstream::peek16` exactly — same shifts,
/// same zero-padding on past-end reads. Returns the 16-bit prefix
/// right-aligned in the low bits of the returned uint.
uint peek16_kernel(uint bit_pos) {
    uint word_idx = bit_pos / 32;
    uint bit_in_word = bit_pos & 31;

    uint hi = bitstream[word_idx];
    uint lo = bitstream[word_idx + 1];

    // Build (hi << 32) | lo in a uint64, then >> (48 - bit_in_word)
    // and mask to 16 bits. Slang exposes uint64_t for this — same
    // path the spec text uses, lifted into a helper.
    uint64_t combined = (uint64_t(hi) << 32) | uint64_t(lo);
    return uint((combined >> (48u - bit_in_word)) & 0xFFFFu);
}

[shader("compute")][numthreads(256, 1, 1)]
void phase1_intra_sync(uint3 tid : SV_DispatchThreadID) {
    uint seq_idx = tid.x;
    if (seq_idx >= num_subsequences) {
        return;
    }

    uint start_bit = seq_idx * subsequence_bits;
    uint sync_target = start_bit + 2u * subsequence_bits;
    uint hard_limit = min(length_bits, sync_target);

    uint p = start_bit;
    uint n = 0u;
    uint c = 0u;
    uint z = 0u;

    // Loop bound: at most one symbol every code_bits + value_bits bits.
    // code_bits >= 1, so at most `hard_limit - start_bit` iterations.
    // For safety in the face of a buggy codebook (entry => 0 advance),
    // cap iterations at `hard_limit - start_bit + 1` symbols — any
    // working codebook terminates well below this.
    uint max_iters = (hard_limit - start_bit) + 1u;
    for (uint iter = 0u; iter < max_iters; iter++) {
        if (p >= hard_limit) {
            break;
        }
        uint peek = peek16_kernel(p);
        uint entry = codebook[c * 65536u + peek];
        uint num_bits = (entry >> 8u) & 0xFFu;
        if (num_bits == 0u) {
            // PrefixMiss — no codeword matches this prefix.
            break;
        }
        uint symbol = entry & 0xFFu;
        uint value_bits = symbol & 0x0Fu;
        uint advance = num_bits + value_bits;

        // Guard against running past length_bits mid-codeword.
        if (p + advance > length_bits) {
            break;
        }

        p += advance;
        n += 1u;
        // z is a 6-bit zig-zag index; mask compiles to one op.
        z = (z + 1u) & 63u;
        if (z == 0u) {
            c = (c + 1u) % num_components;
        }
    }

    s_info_out[seq_idx] = uint4(p, n, c, z);
}
