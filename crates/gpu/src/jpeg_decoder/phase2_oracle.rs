//! CPU oracle for the Phase 2 inter-sequence-sync kernel
//! (Weißenberger & Schmidt 2021 §III-C, Algorithm 4).
//!
//! Walks the per-subsequence `s_info` array produced by Phase 1,
//! checks each `(i, i+1)` pair for `(c, z)` agreement, advances any
//! unsynced subseq by one symbol per iteration, and repeats until
//! every pair agrees (`Converged`) or the retry bound is exhausted
//! (`SyncBoundExceeded`).
//!
//! ## Sync predicate (MVP — matches the plan's kernel pseudocode)
//!
//! Subseq `i` is "synced with its right neighbour" when:
//!
//! - `i + 1 == num_subsequences` (last subseq is trivially synced); OR
//! - `s_info[i].p >= (i + 1) * subsequence_bits` (subseq `i`'s walk
//!   reached at least the start of subseq `i+1`); AND
//! - `s_info[i].c == s_info[i+1].c`; AND
//! - `s_info[i].z == s_info[i+1].z`.
//!
//! Real JPEG framing would track the (run, size) AC structure to make
//! the sync check more robust against false positives. The MVP relies
//! on the `(c, z)` rollover happening identically across subsequences
//! when the codebook + bitstream are well-formed.
//!
//! ## Retry bound
//!
//! `2 * ceil(log2(num_subsequences))` per the plan. For 1 subseq this
//! is 0, so the function short-circuits to `Converged` immediately.

#![cfg(test)]

use crate::jpeg::CanonicalCodebook;
use crate::jpeg_decoder::PackedBitstream;
use crate::jpeg_decoder::phase1_oracle::{StepOutcome, SubsequenceState, try_decode_one_symbol};

/// Outcome of the Phase 2 retry loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Phase2Outcome {
    /// Every subseq is synced with its right neighbour. `iterations`
    /// is the 0-based index of the pass that found every pair
    /// aligned — so a pre-synced Phase 1 output returns `0`, and a
    /// stream that needed one advance pass returns `1`.
    Converged { iterations: u32 },
    /// Retry bound exhausted without convergence. Returned after
    /// `bound + 1` passes; the `s_info` state is left at whatever the
    /// last pass produced.
    SyncBoundExceeded { bound: u32 },
}

/// Maximum number of sync-and-advance passes before giving up.
///
/// `2 * ceil(log2(n))` per the plan. `next_power_of_two(0) = 1` and
/// `next_power_of_two(1) = 1`, both with `trailing_zeros() = 0`, so
/// the math returns `0` cleanly for `n ≤ 1` without a special case.
#[must_use]
pub(super) fn retry_bound(num_subsequences: usize) -> u32 {
    let pow2_exp = num_subsequences.next_power_of_two().trailing_zeros();
    2u32.saturating_mul(pow2_exp)
}

/// One sync-and-advance pass.
///
/// For each `i` in `0..s_info.len() - 1`:
/// - If `(c, z)` agree between `s_info[i]` and `s_info[i+1]` AND
///   `s_info[i].p >= (i+1) * subsequence_bits`, the pair is synced.
/// - Otherwise, advance `s_info[i]` by one symbol via
///   `try_decode_one_symbol`. If the advance step misses (no codeword
///   / past `length_bits`), `s_info[i]` stays unchanged for this pass.
///
/// Returns the number of unsynced subsequences after this pass.
fn phase2_sync_pass(
    s_info: &mut [SubsequenceState],
    bitstream: &PackedBitstream,
    tables: &[CanonicalCodebook],
    subsequence_bits: u32,
) -> u32 {
    let n = s_info.len();
    if n <= 1 {
        return 0;
    }
    let mut unsynced = 0u32;
    for i in 0..n - 1 {
        let me = s_info[i];
        let nxt = s_info[i + 1];
        let nxt_start_p = u32::try_from(i + 1)
            .expect("subseq idx fits u32 by HuffmanParams::validate")
            .saturating_mul(subsequence_bits);
        let in_range = me.p >= nxt_start_p;
        let aligned = me.c == nxt.c && me.z == nxt.z;
        if in_range && aligned {
            continue;
        }
        // Not synced — advance me by one symbol. A miss leaves the
        // state unchanged; this pass is reported as still unsynced.
        if let StepOutcome::Advanced(next) = try_decode_one_symbol(me, bitstream, tables) {
            s_info[i] = next;
        }
        unsynced += 1;
    }
    unsynced
}

/// Drive the Phase 2 sync loop to convergence.
///
/// Returns `Phase2Outcome::Converged { iterations }` once every pair
/// agrees, or `SyncBoundExceeded { bound }` after `retry_bound`
/// passes without convergence.
///
/// # Panics
/// Panics if `tables` is empty (caller bug — at least one codetable
/// is required for any decode work).
pub(super) fn phase2_run_to_sync(
    s_info: &mut [SubsequenceState],
    bitstream: &PackedBitstream,
    tables: &[CanonicalCodebook],
    subsequence_bits: u32,
) -> Phase2Outcome {
    assert!(
        !tables.is_empty(),
        "phase2_run_to_sync requires at least one codetable"
    );
    let bound = retry_bound(s_info.len());
    for iter in 0..=bound {
        let unsynced = phase2_sync_pass(s_info, bitstream, tables, subsequence_bits);
        if unsynced == 0 {
            return Phase2Outcome::Converged { iterations: iter };
        }
    }
    Phase2Outcome::SyncBoundExceeded { bound }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jpeg_decoder::phase1_oracle::phase1_walk;
    use crate::jpeg_decoder::tests::fixtures::{book4_codebook, book4_stream};

    /// Build a Phase 1 output by running `phase1_walk` per subsequence.
    /// Mirrors what the kernel's Phase 1 dispatch would produce.
    fn build_phase1_s_info(
        bitstream: &PackedBitstream,
        tables: &[CanonicalCodebook],
        subsequence_bits: u32,
    ) -> Vec<SubsequenceState> {
        let n = bitstream.length_bits.div_ceil(subsequence_bits);
        (0..n)
            .map(|i| {
                let start_bit = i * subsequence_bits;
                let hard_limit = bitstream.length_bits.min(start_bit + 2 * subsequence_bits);
                let (state, _stop) = phase1_walk(bitstream, tables, start_bit, hard_limit);
                state
            })
            .collect()
    }

    #[test]
    fn retry_bound_zero_for_zero_or_one_subseqs() {
        assert_eq!(retry_bound(0), 0);
        assert_eq!(retry_bound(1), 0);
    }

    #[test]
    fn retry_bound_matches_2_log2_ceil() {
        // 2 → ceil(log2(2)) = 1 → bound = 2
        assert_eq!(retry_bound(2), 2);
        // 3 → next_pow2 = 4 → exp 2 → bound = 4
        assert_eq!(retry_bound(3), 4);
        // 4 → next_pow2 = 4 → exp 2 → bound = 4
        assert_eq!(retry_bound(4), 4);
        // 1024 → exp 10 → bound = 20
        assert_eq!(retry_bound(1024), 20);
    }

    #[test]
    fn empty_or_singleton_converges_in_zero_iterations() {
        let stream = book4_stream(&[0x00; 100]);
        let book = [book4_codebook()];
        // 0 subseqs.
        let mut empty: Vec<SubsequenceState> = vec![];
        let r = phase2_run_to_sync(&mut empty, &stream, &book, 128);
        assert_eq!(r, Phase2Outcome::Converged { iterations: 0 });

        // 1 subseq (anything well-formed).
        let mut one = build_phase1_s_info(&stream, &book, 256);
        assert_eq!(one.len(), 1);
        let r = phase2_run_to_sync(&mut one, &stream, &book, 256);
        assert_eq!(r, Phase2Outcome::Converged { iterations: 0 });
    }

    #[test]
    fn well_formed_uniform_stream_converges_in_zero_iterations() {
        // 1024 length-2 codewords = 2048 bits. 16 subseqs of 128 bits
        // each, all the same width (length_bits is an exact multiple
        // of subsequence_bits). Every subseq decodes 128 symbols, so
        // end-state (c, z) matches across all pairs. Phase 2 sees a
        // pre-synced s_info and converges on the first sweep.
        //
        // Stream sizing matters: a stream where the final subseq is
        // shorter than the others can land its (c, z) anywhere
        // depending on how far phase1_walk gets, and the bound
        // `2 * log2(n)` is not enough advance steps to realign a
        // worst-case trailing subseq (needs O(subsequence_bits) in
        // the MVP). That's a different test (would need a larger
        // bound to exercise).
        let stream = book4_stream(&[0x00; 1024]);
        let book = [book4_codebook()];
        assert_eq!(stream.length_bits % 128, 0, "test setup invariant");
        let mut s_info = build_phase1_s_info(&stream, &book, 128);
        let r = phase2_run_to_sync(&mut s_info, &stream, &book, 128);
        assert_eq!(r, Phase2Outcome::Converged { iterations: 0 });
    }

    #[test]
    fn unsynced_input_advances_until_aligned() {
        // Build a pre-synced s_info on a stream sized to a clean
        // multiple of subsequence_bits, then deliberately perturb one
        // subseq's z so it doesn't match its neighbour. The retry
        // loop should advance it by one symbol until z realigns.
        //
        // For book4's length-2 codewords, advancing by one symbol
        // bumps z by 1 (mod 64). Perturbing s_info[0].z by -1
        // re-aligns after exactly one advance.
        let stream = book4_stream(&[0x00; 1024]);
        let book = [book4_codebook()];
        let mut s_info = build_phase1_s_info(&stream, &book, 128);
        s_info[0].z = (s_info[0].z + 63) % 64;
        let r = phase2_run_to_sync(&mut s_info, &stream, &book, 128);
        match r {
            Phase2Outcome::Converged { iterations } => {
                // First pass: subseq 0 unsynced → advance once.
                // Second pass: subseq 0 now aligned → all synced.
                assert!(
                    iterations <= 2,
                    "expected fast convergence, got {iterations} iterations"
                );
            }
            Phase2Outcome::SyncBoundExceeded { bound } => {
                panic!("unexpected SyncBoundExceeded with bound {bound}");
            }
        }
    }

    #[test]
    fn adversarial_input_returns_sync_bound_exceeded() {
        // Construct an s_info where (c, z) of every pair disagrees
        // permanently. We can't easily produce this via real decoding
        // (the codebook ensures alignment), so synthesise: every subseq
        // gets a unique z value. Advancing won't fix it within bound.
        let stream = book4_stream(&[0x00; 1000]);
        let book = [book4_codebook()];
        let mut s_info = build_phase1_s_info(&stream, &book, 128);
        // Force each subseq to a distinct z so no neighbour pair
        // aligns. After one advance, each subseq's z moves by 1, so
        // the relative phase stays constant. Bound for 16 subseqs is
        // 2 * ceil(log2(16)) = 8 passes.
        for (i, s) in s_info.iter_mut().enumerate() {
            s.z = u32::try_from(i).expect("test fixture: 16 fits u32") & 63;
        }
        let r = phase2_run_to_sync(&mut s_info, &stream, &book, 128);
        assert!(
            matches!(r, Phase2Outcome::SyncBoundExceeded { .. }),
            "expected SyncBoundExceeded, got {r:?}"
        );
    }
}
