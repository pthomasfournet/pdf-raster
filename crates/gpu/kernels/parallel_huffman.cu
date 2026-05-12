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

// Status codes returned by `try_decode_one_symbol_device` /
// `try_decode_one_symbol_kernel`. Match the Rust-side
// `Phase4FailureKind` enum's discriminants exactly so the host can
// `transmute`-style interpret the per-subseq decode_status buffer
// without a translation layer.
//
// 0 reserved for Ok so a `decode_status` buffer freshly allocated
// with `alloc_device_zeroed` reads as Ok-everywhere by default.
#define DECODE_OK            0u
#define DECODE_PREFIX_MISS   1u
#define DECODE_LENGTH_BITS   2u
#define DECODE_INCOMPLETE    3u
// JPEG-framed kernels only — finer-grained failure modes.
#define DECODE_BAD_DC_CATEGORY 4u
#define DECODE_BAD_AC_SIZE     5u
#define DECODE_AC_OVERFLOW     6u

// Try to decode one Huffman symbol starting at `*p`, updating
// `(*p, *n, *c, *z)` in place and writing the decoded symbol byte
// to `*symbol_out` on success. Returns `DECODE_OK` on success,
// `DECODE_PREFIX_MISS` if no codeword matches the 16-bit peek
// (`num_bits == 0`), `DECODE_LENGTH_BITS` if the decoded codeword
// would run past stream end. On failure, state and `*symbol_out`
// are unchanged so callers can detect the miss without checkpoint
// arithmetic.
//
// `count_to` controls per-region accounting: `*n` increments only
// when the symbol *starts* (`*p` before advance) below `count_to`.
// Symbols straddling the boundary belong to the next subseq's
// region and must not inflate this subseq's count — that's what
// Phase 3's exclusive scan + Phase 4's per-thread write region rely
// on. Phase 1 passes `start_bit + subsequence_bits`; Phase 2 passes
// the next subseq's `start_bit`; Phase 4 passes `end_p` (every
// in-region symbol counts).
//
// Mirrors phase1_oracle::try_decode_one_symbol on the host side;
// shared by phase1_intra_sync (loops until hard_limit),
// phase2_inter_sync (calls once per unsynced subseq per pass),
// and phase4_redecode (loops until end_p, emits each decoded
// symbol via `*symbol_out`).
__device__ __forceinline__ unsigned int try_decode_one_symbol_device(
    const unsigned int* __restrict__ bitstream,
    const unsigned int* __restrict__ codebook,
    unsigned int length_bits,
    unsigned int num_components,
    unsigned int count_to,
    unsigned int* p,
    unsigned int* n,
    unsigned int* c,
    unsigned int* z,
    unsigned int* symbol_out
) {
    unsigned int peek = peek16_kernel(bitstream, *p);
    unsigned int entry = codebook[(*c) * CODEBOOK_ENTRIES + peek];
    unsigned int num_bits = (entry >> 8u) & 0xFFu;
    if (num_bits == 0u) return DECODE_PREFIX_MISS;
    unsigned int symbol = entry & 0xFFu;
    unsigned int value_bits = symbol & 0x0Fu;
    unsigned int advance = num_bits + value_bits;
    if (*p + advance > length_bits) return DECODE_LENGTH_BITS;

    unsigned int count_this_one = (*p < count_to) ? 1u : 0u;
    *p += advance;
    *n += count_this_one;
    // z is a 6-bit zig-zag index; mask is one PTX instruction.
    *z = (*z + 1u) & 63u;
    if (*z == 0u) {
        *c = (*c + 1u) % num_components;
    }
    *symbol_out = symbol;
    return DECODE_OK;
}

// Phase 1: one thread per subsequence; writes a "boundary snapshot"
// to s_info_out[seq_idx]. The snapshot is the (p, n, c, z) state at
// the first decode that crosses into the next subsequence's region
// (i.e., the first `p_before_advance < subseq_end_bit && p_after_advance >= subseq_end_bit`).
//
// Why a snapshot rather than the over-walk's end state: Phase 4 of
// subseq (seq_idx+1) inherits from `s_info_out[seq_idx]` to know
// where decoding resumes and what (c, z) it's resuming at. If
// `s_info_out[seq_idx]` stored the over-walk's END state (way past
// the boundary), subseq (seq_idx+1) would start decoding from too
// far in. The snapshot is the precise handoff point.
//
// Phase 2's sync predicate also operates on snapshots: subseq i
// is synced with subseq (i+1) when subseq i's snapshot equals subseq
// (i+1)'s START state. The original-Wei algorithm uses the over-walk
// end state for sync; we snapshot instead because it composes
// cleanly with Phase 4's needs. The sync predicate still works
// (snapshot's (c, z) equals subseq (i+1)'s walker's (c, z) at the
// same boundary when both agree on the codeword alignment).
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
    // Symbols whose first bit lands at >= subseq_end_bit belong to
    // the next subseq's "owned" region; freeze `n` past that point.
    unsigned int subseq_end_bit = min(length_bits, start_bit + subsequence_bits);

    unsigned int p = start_bit;
    unsigned int n = 0u;
    unsigned int c = 0u;
    unsigned int z = 0u;

    // Snapshot state — captured on the first decode whose
    // *post-advance* p crosses subseq_end_bit. If no such crossing
    // happens (stream ends before this subseq's boundary), the
    // snapshot is the final walk state.
    unsigned int snap_p = p;
    unsigned int snap_n = n;
    unsigned int snap_c = c;
    unsigned int snap_z = z;
    unsigned int snapshotted = 0u;

    // Defensive form: if a future dispatcher ever produces a
    // seq_idx where start_bit > hard_limit (e.g., off-by-one in
    // num_subsequences), the subtraction in the naive form would
    // underflow u32 and produce a huge iteration count. The loop
    // body's `p >= hard_limit` check would still terminate it
    // immediately, but the bound expression itself should not lie.
    unsigned int max_iters = ((hard_limit > start_bit) ? (hard_limit - start_bit) : 0u) + 1u;
    unsigned int symbol_sink;  // Phase 1 doesn't emit; the helper writes here and we ignore it.
    for (unsigned int iter = 0u; iter < max_iters; iter++) {
        if (p >= hard_limit) break;
        unsigned int p_before = p;
        if (try_decode_one_symbol_device(
                bitstream, codebook, length_bits, num_components, subseq_end_bit,
                &p, &n, &c, &z, &symbol_sink) != DECODE_OK) {
            break;
        }
        // First decode that crosses into the next subseq's region
        // (p_before was inside, p is now at or past). Capture the
        // post-advance state — that's where (seq_idx + 1) resumes.
        if (snapshotted == 0u && p_before < subseq_end_bit && p >= subseq_end_bit) {
            snap_p = p;
            snap_n = n;
            snap_c = c;
            snap_z = z;
            snapshotted = 1u;
        }
    }

    // No boundary crossing (stream ended inside this subseq's region
    // before subseq_end_bit). Use the walk's terminal state as the
    // snapshot — subseq (seq_idx + 1) won't exist in that case, so
    // the snapshot value only feeds Phase 2's last-subseq trivial-
    // sync test.
    if (snapshotted == 0u) {
        snap_p = p;
        snap_n = n;
        snap_c = c;
        snap_z = z;
    }

    uint4 out;
    out.x = snap_p;
    out.y = snap_n;
    out.z = snap_c;
    out.w = snap_z;
    s_info_out[seq_idx] = out;
}

// Phase 4: re-decode + write. One thread per subsequence. Re-walks
// the subseq's owned region using the predecessor's boundary
// snapshot as the starting (p, c, z) and the subseq's own snapshot
// as the end position. Writes each decoded symbol to
// `symbols_out[offsets[seq_idx] + local_n]`.
//
// `s_info[i]` is the boundary snapshot Phase 1 captured for subseq
// i — the (p, c, z) at the first decode crossing into subseq (i+1)'s
// region. Subseq i's owned region thus spans [s_info[i-1].p, s_info[i].p);
// subseq 0 starts fresh at (p=0, c=0, z=0). This decomposes the
// stream into disjoint per-thread regions that cover it exactly,
// which is what Phase 3's offset arithmetic + Phase 4's write
// arithmetic both rely on.
//
// The kernel also bounds-checks each write against `total_symbols`
// so adversarial / buggy upstream output can't corrupt host memory
// past symbols_out's end.
//
// Per-subseq `decode_status[seq_idx]` records the exit condition:
// DECODE_OK if the walk reached end_p cleanly, DECODE_PREFIX_MISS
// or DECODE_LENGTH_BITS if the decode helper failed mid-region,
// DECODE_INCOMPLETE if `max_iters` ran out without reaching end_p
// (degenerate codebook with zero-advance entries). Host reads this
// after dispatch to surface adversarial-input failures as typed
// errors rather than silently truncating the symbol stream.
extern "C" __global__ void phase4_redecode(
    const unsigned int* __restrict__ bitstream,
    const unsigned int* __restrict__ codebook,
    const uint4* __restrict__ s_info,
    const unsigned int* __restrict__ offsets,
    unsigned int* __restrict__ symbols_out,
    unsigned int* __restrict__ decode_status,
    unsigned int length_bits,
    unsigned int total_symbols,
    unsigned int num_subsequences,
    unsigned int num_components
) {
    unsigned int seq_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (seq_idx >= num_subsequences) {
        return;
    }

    uint4 me = s_info[seq_idx];
    unsigned int end_p = me.x;

    // (p, n, c, z) packed into uint4 (x, y, z, w).
    unsigned int p, n, c, z;
    if (seq_idx == 0u) {
        p = 0u; n = 0u; c = 0u; z = 0u;
    } else {
        uint4 prev = s_info[seq_idx - 1];
        p = prev.x; n = 0u; c = prev.z; z = prev.w;
    }

    unsigned int base = offsets[seq_idx];
    unsigned int status = DECODE_INCOMPLETE;  // overwritten on clean exit or typed failure

    unsigned int max_iters = ((end_p > p) ? (end_p - p) : 0u) + 1u;
    unsigned int symbol;
    for (unsigned int iter = 0u; iter < max_iters; iter++) {
        if (p >= end_p) {
            status = DECODE_OK;
            break;
        }
        // count_to = end_p: every in-region symbol counts (and only
        // in-region symbols can be reached, since we exit the loop
        // once p >= end_p).
        unsigned int step = try_decode_one_symbol_device(
            bitstream, codebook, length_bits, num_components, end_p,
            &p, &n, &c, &z, &symbol);
        if (step != DECODE_OK) {
            status = step;  // PrefixMiss or LengthBits
            break;
        }
        // Bounds-check the write: an adversarial / buggy upstream
        // could inflate this thread's emit count past its slot.
        // `n` was just incremented by the helper, so we write at
        // `base + n - 1` and require it to fit in symbols_out.
        unsigned int write_idx = base + (n - 1u);
        if (write_idx < total_symbols) {
            symbols_out[write_idx] = symbol;
        }
    }
    decode_status[seq_idx] = status;
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
    // count_to = nxt_start_p: symbols starting before the next
    // subseq's start belong to my region; symbols starting past
    // that don't (they belong to the next subseq).
    unsigned int symbol_sink;  // Phase 2 doesn't emit.
    (void)try_decode_one_symbol_device(
        bitstream, codebook, length_bits, num_components, nxt_start_p,
        &p, &n, &c, &z, &symbol_sink);

    uint4 updated;
    updated.x = p;
    updated.y = n;
    updated.z = c;
    updated.w = z;
    s_info[seq_idx] = updated;
    flags[seq_idx] = 0u;
}

// JPEG-framed helper: extract (dc_sel, ac_sel) from one packed
// mcu_schedule entry.  Layout matches the Slang side exactly:
//     bits 0..8   = component_idx  (unused on the kernel side)
//     bits 8..16  = dc_sel
//     bits 16..24 = ac_sel
__device__ __forceinline__ void unpack_schedule_device(
    const unsigned int* __restrict__ mcu_schedule,
    unsigned int block_in_mcu,
    unsigned int* dc_sel,
    unsigned int* ac_sel
) {
    unsigned int packed = mcu_schedule[block_in_mcu];
    *dc_sel = (packed >> 8u) & 0xFFu;
    *ac_sel = (packed >> 16u) & 0xFFu;
}

// JPEG-framed counterpart of try_decode_one_symbol_device.
//
// State semantics:
//   *p           = bit position in `bitstream`
//   *n           = symbols emitted by this thread so far
//   *block_in_mcu = 0 .. blocks_per_mcu - 1
//   *z_in_block  = 0 .. 64 (inclusive end of block)
//
// One symbol per call. DC slot (z_in_block == 0) looks up in
// `dc_codebook[dc_sel]`; AC slot looks up in `codebook[ac_sel]`.
// EOB (0x00) sets z := 64; ZRL (0xF0) advances z by 16; run+size
// advances z by run+1. Post-advance z > 64 is spec-illegal —
// returns DECODE_AC_OVERFLOW. DC category > 11 returns
// DECODE_BAD_DC_CATEGORY; AC size > 10 returns DECODE_BAD_AC_SIZE.
//
// When z_in_block reaches 64 the block rolls over modulo
// blocks_per_mcu and z resets to 0.
//
// Mirrors `try_decode_one_jpeg_symbol_kernel` in the Slang file
// byte-for-byte; a divergence between the two would surface as a
// CUDA-vs-Vulkan output mismatch in B2d's end-to-end test.
__device__ __forceinline__ unsigned int try_decode_one_jpeg_symbol_device(
    const unsigned int* __restrict__ bitstream,
    const unsigned int* __restrict__ codebook,
    const unsigned int* __restrict__ dc_codebook,
    const unsigned int* __restrict__ mcu_schedule,
    unsigned int length_bits,
    unsigned int blocks_per_mcu,
    unsigned int count_to,
    unsigned int* p,
    unsigned int* n,
    unsigned int* block_in_mcu,
    unsigned int* z_in_block,
    unsigned int* symbol_out
) {
    *symbol_out = 0u;
    unsigned int dc_sel;
    unsigned int ac_sel;
    unpack_schedule_device(mcu_schedule, *block_in_mcu, &dc_sel, &ac_sel);

    bool is_dc = (*z_in_block == 0u);
    unsigned int table_base = (is_dc ? dc_sel : ac_sel) * CODEBOOK_ENTRIES;
    unsigned int peek = peek16_kernel(bitstream, *p);
    unsigned int entry = (is_dc ? dc_codebook[table_base + peek]
                                : codebook[table_base + peek]);
    unsigned int num_bits = (entry >> 8u) & 0xFFu;
    if (num_bits == 0u) return DECODE_PREFIX_MISS;
    unsigned int symbol = entry & 0xFFu;

    // Spec caps on the magnitude size — mirror the Slang side.
    if (is_dc) {
        if (symbol > 11u) return DECODE_BAD_DC_CATEGORY;
    } else {
        unsigned int size_field = symbol & 0xFu;
        if (size_field > 10u) return DECODE_BAD_AC_SIZE;
    }
    unsigned int value_bits = symbol & 0x0Fu;
    unsigned int advance = num_bits + value_bits;
    if (*p + advance > length_bits) return DECODE_LENGTH_BITS;

    unsigned int count_this_one = (*p < count_to) ? 1u : 0u;
    *p += advance;
    *n += count_this_one;

    if (is_dc) {
        // DC: one symbol per block, transition to AC slot 1.
        *z_in_block = 1u;
    } else {
        if (symbol == 0u) {
            // EOB: rest of block implicitly zero.
            *z_in_block = 64u;
        } else if (symbol == 0xF0u) {
            unsigned int next_w = *z_in_block + 16u;
            if (next_w > 64u) return DECODE_AC_OVERFLOW;
            *z_in_block = next_w;
        } else {
            unsigned int run = (symbol >> 4u) & 0xFu;
            unsigned int next_w = *z_in_block + run + 1u;
            if (next_w > 64u) return DECODE_AC_OVERFLOW;
            *z_in_block = next_w;
        }
    }

    // End-of-block: rotate block_in_mcu, reset z.
    if (*z_in_block == 64u) {
        *block_in_mcu = (*block_in_mcu + 1u) % blocks_per_mcu;
        *z_in_block = 0u;
    }

    *symbol_out = symbol;
    return DECODE_OK;
}

// JPEG-framed Phase 2 (inter-sequence sync).  Same shape as the
// synthetic `phase2_inter_sync` but uses the JPEG state machine.
//
// State semantics mirror jpeg_phase1_intra_sync:
//   state.x = p              (bit position)
//   state.y = n              (per-thread symbol count)
//   state.z = block_in_mcu   (0 .. blocks_per_mcu - 1)
//   state.w = z_in_block     (0 .. 64)
//
// Sync predicate: subseq i is synced with subseq (i+1) when
//   i's snapshot has crossed (i+1)'s start bit (me.x >= nxt_start_p)
//   AND block_in_mcu and z_in_block agree (me.z == nxt.z, me.w == nxt.w).
// This is the JPEG analogue of the synthetic stream's (c, z) predicate.
extern "C" __global__ void jpeg_phase2_inter_sync(
    const unsigned int* __restrict__ bitstream,
    const unsigned int* __restrict__ codebook,
    const unsigned int* __restrict__ dc_codebook,
    const unsigned int* __restrict__ mcu_schedule,
    uint4* __restrict__ s_info,
    unsigned int* __restrict__ flags,
    unsigned int length_bits,
    unsigned int subsequence_bits,
    unsigned int num_subsequences,
    unsigned int blocks_per_mcu
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

    uint4 me = s_info[seq_idx];
    uint4 nxt = s_info[seq_idx + 1];
    // JPEG framing: block_in_mcu is in .z; z_in_block is in .w.
    unsigned int me_block = me.z;
    unsigned int me_z    = me.w;
    unsigned int nxt_block = nxt.z;
    unsigned int nxt_z     = nxt.w;
    unsigned int nxt_start_p = (seq_idx + 1u) * subsequence_bits;

    unsigned int in_range = (me.x >= nxt_start_p) ? 1u : 0u;
    unsigned int aligned  = (me_block == nxt_block && me_z == nxt_z) ? 1u : 0u;

    if (in_range && aligned) {
        flags[seq_idx] = 1u;
        return;
    }

    // Not synced — advance me by one JPEG symbol, write back.
    unsigned int p = me.x;
    unsigned int n = me.y;
    unsigned int block_in_mcu = me_block;
    unsigned int z_in_block   = me_z;
    unsigned int symbol_sink;
    (void)try_decode_one_jpeg_symbol_device(
        bitstream, codebook, dc_codebook, mcu_schedule,
        length_bits, blocks_per_mcu, nxt_start_p,
        &p, &n, &block_in_mcu, &z_in_block, &symbol_sink);

    uint4 updated;
    updated.x = p;
    updated.y = n;
    updated.z = block_in_mcu;
    updated.w = z_in_block;
    s_info[seq_idx] = updated;
    flags[seq_idx] = 0u;
}

// JPEG-framed Phase 1 (intra-sequence sync).  Same shape as
// `phase1_intra_sync` but uses the JPEG state machine.
//
// State semantics:
//   state.x = p              (bit position)
//   state.y = n              (per-thread symbol count)
//   state.z = block_in_mcu   (0 .. blocks_per_mcu - 1)
//   state.w = z_in_block     (0 .. 64)
extern "C" __global__ void jpeg_phase1_intra_sync(
    const unsigned int* __restrict__ bitstream,
    const unsigned int* __restrict__ codebook,
    const unsigned int* __restrict__ dc_codebook,
    const unsigned int* __restrict__ mcu_schedule,
    uint4* __restrict__ s_info_out,
    unsigned int length_bits,
    unsigned int subsequence_bits,
    unsigned int num_subsequences,
    unsigned int blocks_per_mcu
) {
    unsigned int seq_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (seq_idx >= num_subsequences) {
        return;
    }

    unsigned int start_bit = seq_idx * subsequence_bits;
    unsigned int sync_target = start_bit + 2u * subsequence_bits;
    unsigned int hard_limit = min(length_bits, sync_target);
    unsigned int subseq_end_bit = min(length_bits, start_bit + subsequence_bits);

    unsigned int p = start_bit;
    unsigned int n = 0u;
    unsigned int block_in_mcu = 0u;
    unsigned int z_in_block = 0u;

    unsigned int snap_p = p;
    unsigned int snap_n = n;
    unsigned int snap_block = block_in_mcu;
    unsigned int snap_z = z_in_block;
    unsigned int snapshotted = 0u;

    // Defensive max_iters mirrors the synthetic Phase 1.
    unsigned int max_iters = ((hard_limit > start_bit) ? (hard_limit - start_bit) : 0u) + 1u;
    unsigned int symbol_sink;
    for (unsigned int iter = 0u; iter < max_iters; iter++) {
        if (p >= hard_limit) break;
        unsigned int p_before = p;
        if (try_decode_one_jpeg_symbol_device(
                bitstream, codebook, dc_codebook, mcu_schedule,
                length_bits, blocks_per_mcu, subseq_end_bit,
                &p, &n, &block_in_mcu, &z_in_block, &symbol_sink) != DECODE_OK) {
            break;
        }
        if (snapshotted == 0u && p_before < subseq_end_bit && p >= subseq_end_bit) {
            snap_p = p;
            snap_n = n;
            snap_block = block_in_mcu;
            snap_z = z_in_block;
            snapshotted = 1u;
        }
    }

    if (snapshotted == 0u) {
        snap_p = p;
        snap_n = n;
        snap_block = block_in_mcu;
        snap_z = z_in_block;
    }

    uint4 out;
    out.x = snap_p;
    out.y = snap_n;
    out.z = snap_block;
    out.w = snap_z;
    s_info_out[seq_idx] = out;
}
