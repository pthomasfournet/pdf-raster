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
// ## Phase 2 algorithm (Weißenberger & Schmidt 2021 §III-C)
//
// One thread per subsequence. Reads `s_info[i]` and `s_info[i+1]`,
// tests whether `(c, z)` agree and `s_info[i].p >= (i+1) *
// subsequence_bits`. If aligned: writes `flags[i] = 1`. Otherwise:
// advances `s_info[i]` by one symbol, writes `flags[i] = 0`. Host
// loops the dispatch until all flags are 1 or the retry bound is
// exhausted.
//
// ## Codebook layout
//
// `codebook` is a `u32[num_components * 65 536]` flat buffer. Each
// entry packs `(num_bits << 8) | symbol`. Thread reads entry at
// `codebook[c * 65 536 + peek16]`. `num_bits == 0` = no match.
//
// ## MVP simplification (matches phase1_oracle / phase2_oracle)
//
// Real JPEG AC framing tracks (run, size) per symbol. The MVP advances
// `z += 1` per decoded symbol, rolling over `c` on the 64-symbol
// boundary. Production-grade AC framing is a follow-up.

[[vk::binding(0, 0)]] StructuredBuffer<uint> bitstream;
[[vk::binding(1, 0)]] StructuredBuffer<uint> codebook;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint4> s_info_out;
// Phase 2 only — bound at slot 3 in the Phase 2 descriptor layout.
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> sync_flags;

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

    uint64_t combined = (uint64_t(hi) << 32) | uint64_t(lo);
    return uint((combined >> (48u - bit_in_word)) & 0xFFFFu);
}

/// Try to decode one Huffman symbol starting at `state.p`, returning
/// `true` if the state was advanced. On failure (prefix-miss or
/// length-bits truncation) `state` is left unchanged and the caller
/// can detect the miss without checkpoint arithmetic.
///
/// `count_to` controls per-region accounting: `state.y` (= n)
/// increments only when the symbol *starts* (state.x before advance)
/// below `count_to`. Symbols straddling the boundary belong to the
/// next subseq's region. Phase 1 passes `start_bit + subsequence_bits`;
/// Phase 2 passes the next subseq's `start_bit`.
///
/// Shared by Phase 1 (loops until hard_limit) and Phase 2 (calls
/// once per unsynced subseq per pass).
bool try_decode_one_symbol_kernel(inout uint4 state, uint count_to) {
    uint peek = peek16_kernel(state.x);
    uint entry = codebook[state.z * 65536u + peek];
    uint num_bits = (entry >> 8u) & 0xFFu;
    if (num_bits == 0u) {
        return false;
    }
    uint symbol = entry & 0xFFu;
    uint value_bits = symbol & 0x0Fu;
    uint advance = num_bits + value_bits;
    if (state.x + advance > length_bits) {
        return false;
    }
    uint count_this_one = (state.x < count_to) ? 1u : 0u;
    state.x += advance;
    state.y += count_this_one;
    // state.w is the zig-zag index (6-bit); state.z is the component.
    state.w = (state.w + 1u) & 63u;
    if (state.w == 0u) {
        state.z = (state.z + 1u) % num_components;
    }
    return true;
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
    // Symbols whose first bit lands at >= subseq_end_bit belong to
    // the next subseq's "owned" region; freeze `n` past that point.
    uint subseq_end_bit = min(length_bits, start_bit + subsequence_bits);

    // (p, n, c, z) packed into uint4 as (x, y, z, w).
    uint4 state = uint4(start_bit, 0u, 0u, 0u);

    // Loop bound: at most one symbol every (code_bits + value_bits)
    // bits, and code_bits >= 1 — so at most `hard_limit - start_bit`
    // symbols. The +1 caps a buggy codebook (entry => 0 advance);
    // working codebooks terminate well below this.
    uint max_iters = (hard_limit - start_bit) + 1u;
    for (uint iter = 0u; iter < max_iters; iter++) {
        if (state.x >= hard_limit) {
            break;
        }
        if (!try_decode_one_symbol_kernel(state, subseq_end_bit)) {
            break;
        }
    }

    s_info_out[seq_idx] = state;
}

[shader("compute")][numthreads(256, 1, 1)]
void phase2_inter_sync(uint3 tid : SV_DispatchThreadID) {
    uint seq_idx = tid.x;
    if (seq_idx >= num_subsequences) {
        return;
    }
    // Last subseq has no right neighbour — trivially synced.
    if (seq_idx + 1u == num_subsequences) {
        sync_flags[seq_idx] = 1u;
        return;
    }

    uint4 me = s_info_out[seq_idx];
    uint4 nxt = s_info_out[seq_idx + 1];
    uint nxt_start_p = (seq_idx + 1u) * subsequence_bits;

    // SubsequenceState.{p,n,c,z} maps to uint4.{x,y,z,w}.
    bool in_range = me.x >= nxt_start_p;
    bool aligned = me.z == nxt.z && me.w == nxt.w;

    if (in_range && aligned) {
        sync_flags[seq_idx] = 1u;
        return;
    }

    // Not synced — advance me by one symbol, write back, mark unsynced.
    // The advance is permitted to fail; on miss the state stays put
    // and we'll spin this subseq until the host's retry bound.
    // count_to = nxt_start_p: symbols starting before the next
    // subseq's start belong to my region.
    bool _ignored = try_decode_one_symbol_kernel(me, nxt_start_p);
    s_info_out[seq_idx] = me;
    sync_flags[seq_idx] = 0u;
}
