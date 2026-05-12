//! CPU oracle for the Phase 1 intra-sequence-sync kernel
//! (Weißenberger & Schmidt 2021 §III-B, Algorithm 3).
//!
//! Walks a single subsequence on the CPU, producing the same
//! `(p, n, c, z)` end-state the GPU kernel will produce for that
//! thread. Used as the cross-backend bit-identity oracle.
//!
//! ## Algorithm shape
//!
//! Each "subsequence" is a fixed-bit-length slice of the entropy-coded
//! stream. Thread `i` starts decoding at bit `i * subsequence_bits`
//! and continues past its subsequence boundary until it reaches a
//! `hard_limit` (capped at `length_bits`). The state advanced per
//! symbol is:
//!
//! - `p` — current absolute bit position.
//! - `n` — number of symbols decoded so far.
//! - `c` — current component (0..`num_components`).
//! - `z` — current zig-zag index within the 64-coefficient block.
//!
//! ## MVP simplification (matches the kernel)
//!
//! Real JPEG AC encoding is run-length: each AC symbol carries a
//! `(run, size)` pair where `run` is the number of zero coefficients
//! to skip before the next nonzero one. The MVP kernel ignores AC
//! framing and advances `z += 1` per decoded symbol, rolling over to
//! `(z=0, c=(c+1) % num_components)` on the 64-symbol boundary. This
//! oracle matches the kernel's simplification — it is NOT a real
//! JPEG decoder.

#![cfg(test)]

use crate::jpeg::CanonicalCodebook;
use crate::jpeg_decoder::PackedBitstream;
use crate::jpeg_decoder::bitstream::peek16;

/// Per-subsequence decoder state.
///
/// Mirrors the GPU kernel's `SInfo` struct one-for-one so test
/// assertions can compare host and device output without translation.
/// `repr(C)` + `bytemuck::Pod` lets the dispatcher download the
/// device buffer straight into a `Vec<SubsequenceState>`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub(super) struct SubsequenceState {
    /// Absolute bit position in `PackedBitstream`.
    pub p: u32,
    /// Symbols decoded since `start_bit`.
    pub n: u32,
    /// Current component index (0..`num_components`).
    pub c: u32,
    /// Current zig-zag index within the 64-coefficient block.
    pub z: u32,
}

// SAFETY: `repr(C)` over four `u32` fields; the layout is `[u32; 4]`
// in declaration order, so reinterpreting bytes-as-SubsequenceState
// matches the kernel's `uint4` write.
unsafe impl bytemuck::Zeroable for SubsequenceState {}
unsafe impl bytemuck::Pod for SubsequenceState {}

/// Reason the Phase 1 walk loop exited.
///
/// Useful for diagnostics in tests; matches what the GPU kernel
/// would observe when it falls out of its decode loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Phase1Stop {
    /// Reached `hard_limit` cleanly (the normal exit).
    HardLimit,
    /// `length_bits` reached mid-decode (final bit position past stream end).
    LengthBits,
    /// 16-bit peek matched no codeword in the active table — the
    /// kernel returns a miss sentinel and breaks.
    PrefixMiss,
}

/// Run the Phase 1 intra-sync decode for one subsequence.
///
/// The caller supplies `tables` indexed by component. `start_bit`
/// is the absolute bit position the thread begins decoding at;
/// `hard_limit` is the upper bound (typically
/// `(seq_idx + 2) * subsequence_bits` clamped to `length_bits`).
///
/// Returns the final `(p, n, c, z)` state plus the reason the loop
/// terminated. The state is what the GPU kernel writes into its
/// `s_info_out[seq_idx]` slot.
///
/// # Panics
/// Panics if `tables` is empty (caller bug — Phase 1 always has at
/// least one component-keyed codetable).
pub(super) fn phase1_walk(
    bitstream: &PackedBitstream,
    tables: &[CanonicalCodebook],
    start_bit: u32,
    hard_limit: u32,
) -> (SubsequenceState, Phase1Stop) {
    assert!(
        !tables.is_empty(),
        "phase1_walk requires at least one codetable"
    );
    // tables.len() comes from a `&[CanonicalCodebook]` whose length
    // is bounded by the number of JPEG components (at most 4) in
    // every realistic caller. The u32 fit is total, but we use
    // try_from to fail loud if a test ever passes 2^32 tables.
    let num_components =
        u32::try_from(tables.len()).expect("more codetables than fit in u32 — caller bug");
    let length_bits = bitstream.length_bits;

    let mut state = SubsequenceState {
        p: start_bit,
        n: 0,
        c: 0,
        z: 0,
    };

    loop {
        if state.p >= hard_limit {
            return (state, Phase1Stop::HardLimit);
        }
        // Peek 16 bits at the current position; the codetable is a
        // 65 536-entry LUT keyed by that prefix.
        let peek = peek16(bitstream, u64::from(state.p));
        let table = &tables[state.c as usize];
        let entry = table.lookup(peek);
        if entry.num_bits == 0 {
            // No codeword matches this prefix — kernel's miss path.
            return (state, Phase1Stop::PrefixMiss);
        }
        let code_bits = u32::from(entry.num_bits);
        // The kernel's MVP advance: code_bits + value_bits, where
        // value_bits is the low nibble of the symbol. Matches the
        // `decode_one_symbol` math in the plan's parallel_huffman.slang.
        let value_bits = u32::from(entry.symbol & 0x0F);
        let advance = code_bits + value_bits;

        if state.p + advance > length_bits {
            // Codeword + value bits would run past the end of the
            // stream. The kernel's outer `while (state.p < hard_limit)`
            // would still admit this iteration (because state.p < hard_limit
            // before the advance), but the resulting state.p would land
            // past length_bits with garbage in the value bits. Match the
            // kernel's pragmatic behaviour: stop without consuming.
            return (state, Phase1Stop::LengthBits);
        }

        state.p += advance;
        state.n = state.n.saturating_add(1);
        state.z = (state.z + 1) % 64;
        if state.z == 0 {
            state.c = (state.c + 1) % num_components;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jpeg::headers::{DhtClass, JpegHuffmanTable};
    use crate::jpeg_decoder::tests::fixtures::{book4_codebook, book4_stream};

    #[test]
    fn walks_one_symbol_advances_correctly() {
        // Stream = one symbol 0x00 (code "00", 2 bits).
        let stream = book4_stream(&[0x00]);
        let book = [book4_codebook()];
        let (state, stop) = phase1_walk(&stream, &book, 0, 2);
        assert_eq!(stop, Phase1Stop::HardLimit);
        assert_eq!(state.p, 2);
        assert_eq!(state.n, 1);
        assert_eq!(state.c, 0);
        assert_eq!(state.z, 1);
    }

    #[test]
    fn walks_64_symbols_rolls_over_zig_zag() {
        // 64 length-2 codewords (all symbol 0x00). z should wrap to 0
        // and c should advance to 1 if there's a second codetable.
        let symbols = vec![0x00; 64];
        let stream = book4_stream(&symbols);
        let book = [book4_codebook(), book4_codebook()];
        let (state, _) = phase1_walk(&stream, &book, 0, 128);
        assert_eq!(state.n, 64);
        assert_eq!(state.z, 0);
        assert_eq!(state.c, 1);
    }

    #[test]
    fn walks_64_symbols_single_component_rolls_back_to_0() {
        // Same as above but only one component — c stays at 0 because
        // (0 + 1) % 1 == 0.
        let symbols = vec![0x00; 64];
        let stream = book4_stream(&symbols);
        let book = [book4_codebook()];
        let (state, _) = phase1_walk(&stream, &book, 0, 128);
        assert_eq!(state.n, 64);
        assert_eq!(state.z, 0);
        assert_eq!(state.c, 0);
    }

    #[test]
    fn stops_at_hard_limit() {
        // 5 length-2 symbols = 10 bits. hard_limit = 6 bits.
        // Decoder should consume 3 symbols (6 bits) then exit.
        let symbols = vec![0x00; 5];
        let stream = book4_stream(&symbols);
        let book = [book4_codebook()];
        let (state, stop) = phase1_walk(&stream, &book, 0, 6);
        assert_eq!(stop, Phase1Stop::HardLimit);
        assert_eq!(state.p, 6);
        assert_eq!(state.n, 3);
    }

    #[test]
    fn start_bit_offsets_into_stream() {
        // 4 length-2 symbols = 8 bits. Start at bit 2; should
        // consume 3 symbols (bits 2..=7) then hit hard_limit at 8.
        let symbols = vec![0x00; 4];
        let stream = book4_stream(&symbols);
        let book = [book4_codebook()];
        let (state, _) = phase1_walk(&stream, &book, 2, 8);
        assert_eq!(state.p, 8);
        assert_eq!(state.n, 3);
    }

    #[test]
    fn empty_stream_returns_initial_state() {
        let stream = PackedBitstream {
            words: vec![],
            length_bits: 0,
        };
        let book = [book4_codebook()];
        let (state, stop) = phase1_walk(&stream, &book, 0, 0);
        // start_bit = 0 == hard_limit = 0 → immediate HardLimit exit.
        assert_eq!(stop, Phase1Stop::HardLimit);
        assert_eq!(state.p, 0);
        assert_eq!(state.n, 0);
    }

    #[test]
    fn stops_on_prefix_miss() {
        // Single length-1 codeword "0" → symbol 42. Stream has a
        // single bit "1" — prefix matches no codeword.
        let table = JpegHuffmanTable {
            class: DhtClass::Dc,
            table_id: 0,
            num_codes: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            values: vec![42],
        };
        let book = [CanonicalCodebook::build(&table).unwrap()];
        let stream = PackedBitstream {
            words: vec![0x8000_0000],
            length_bits: 1,
        };
        let (state, stop) = phase1_walk(&stream, &book, 0, 1);
        assert_eq!(stop, Phase1Stop::PrefixMiss);
        assert_eq!(state.n, 0);
        assert_eq!(state.p, 0);
    }

    #[test]
    fn stops_on_length_bits_mid_codeword() {
        // Stream is 1 bit long but the only codeword is 2 bits.
        // peek16 will read 0x0000 + zero-pad → match the 2-bit code
        // "00" claiming 2 bits, but state.p (0) + 2 > length_bits (1)
        // — return LengthBits.
        let stream = PackedBitstream {
            words: vec![0x0000_0000],
            length_bits: 1,
        };
        let book = [book4_codebook()];
        let (state, stop) = phase1_walk(&stream, &book, 0, 1);
        assert_eq!(stop, Phase1Stop::LengthBits);
        assert_eq!(state.n, 0);
    }

    #[test]
    fn ac_symbol_advances_by_code_plus_value_bits() {
        // AC-shaped symbol 0x52 (run=5, size=2 in JPEG's nibbling).
        // value_bits = 0x52 & 0x0F = 2. With a length-1 code "0", the
        // total advance per symbol is 1 + 2 = 3 bits. After 4 symbols
        // we've consumed 12 bits. The kernel reads but does not
        // *interpret* the value bits at this phase — we only check
        // that `p` advances by code_bits + value_bits.
        let table = JpegHuffmanTable {
            class: DhtClass::Ac,
            table_id: 0,
            num_codes: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            values: vec![0x52],
        };
        let book = [CanonicalCodebook::build(&table).unwrap()];
        // We need a stream that's at least 12 bits long. Construct it
        // by hand: every "0" bit is a codeword, each followed by 2
        // value bits. Use 12 zero bits as our stream.
        let stream = PackedBitstream {
            words: vec![0x0000_0000],
            length_bits: 12,
        };
        let (state, stop) = phase1_walk(&stream, &book, 0, 12);
        assert_eq!(stop, Phase1Stop::HardLimit);
        assert_eq!(state.p, 12, "p should advance by 4 * (1 + 2) = 12");
        assert_eq!(state.n, 4, "4 symbols decoded");
    }

    #[test]
    fn cross_subsequence_walk_matches_continued_decode() {
        // The kernel's intra-sync property: a thread starting at the
        // boundary of a previous subsequence should reach the same
        // state as a thread that decoded straight through. Build a
        // 200-bit stream, walk it 0..200 in one shot, then walk
        // 0..100 + 100..200 in two pieces, compare.
        let symbols = vec![0x00; 100]; // 100 length-2 codewords = 200 bits
        let stream = book4_stream(&symbols);
        let book = [book4_codebook()];
        let length_bits = stream.length_bits;
        assert_eq!(length_bits, 200);

        let (state_full, _) = phase1_walk(&stream, &book, 0, length_bits);
        let (state_first, _) = phase1_walk(&stream, &book, 0, 100);
        let (state_second, _) = phase1_walk(&stream, &book, state_first.p, length_bits);

        // n is local to each walk; the rest of the state should match
        // the full-walk's intermediate (p, c, z).
        assert_eq!(state_full.p, length_bits);
        assert_eq!(state_full.n, 100);
        assert_eq!(state_first.p, 100);
        assert_eq!(state_first.n, 50);
        assert_eq!(state_second.p, length_bits);
        assert_eq!(state_second.n, 50);
        // c rolls over every 64 symbols (single component → stays 0).
        assert_eq!(state_full.c, 0);
        assert_eq!(state_full.z, 100 % 64);
    }
}
