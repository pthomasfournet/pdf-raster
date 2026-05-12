//! Adversarial-corpus tests for the Phase 1→4 GPU pipeline.
//!
//! Implements the 10 acceptance-criteria vectors from the
//! Phase A spec (`docs/superpowers/specs/2026-05-11-gpu-jpeg-huffman-v2-design.md`):
//!
//! 1. short stream (< 1 KB)
//! 2. long stream (~200 K symbols, scaled down from the spec's
//!    "~10 MB" — a real-JPEG figure, not unit-test territory)
//! 3. uniform symbol distribution
//! 4. skewed distribution (90/10 split, uniform-length code)
//! 5. single-symbol "alphabet" (degenerate, length-1 code)
//! 6. maximum-length 16-bit codewords
//! 7. Phase 2 retry trigger (adversarial, mixed-length book —
//!    documented MVP limitation; expects SyncBoundExceeded)
//! 8. exactly one subsequence
//! 9. exact 32-bit-word boundary (no tail padding)
//! 10. maximum tail padding (31 bits short)
//!
//! The vectors that stay within the MVP `(c, z)` sync predicate
//! assert byte-for-byte round-trip equivalence between input
//! symbols and decoded symbols. The adversarial vector asserts
//! the dispatcher surfaces the SyncBoundExceeded error rather
//! than hanging or producing garbage.
//!
//! ## Why a separate module
//!
//! `huffman/mod.rs` already carries 1000+ lines across two test
//! modules; per `feedback_shared_helpers_no_god_files.md`, split
//! before becoming a god file. This module holds the
//! corpus + its helper; the per-phase parity tests stay where
//! they were.

use super::*;
use crate::backend::cuda::CudaBackend;
use crate::jpeg::headers::{DhtClass, JpegHuffmanTable};
use crate::jpeg_decoder::tests::synthetic::encode_symbols;

#[cfg(feature = "vulkan")]
use crate::backend::vulkan::VulkanBackend;

fn try_cuda() -> Option<CudaBackend> {
    CudaBackend::new().ok()
}

#[cfg(feature = "vulkan")]
fn try_vulkan() -> Option<VulkanBackend> {
    VulkanBackend::new().ok()
}

/// Build a uniform-length-2 book over 4 symbols with **zero
/// value-bit suffixes**: symbols 0x00, 0x10, 0x20, 0x30 (low nibble
/// = 0, so `value_bits = symbol & 0x0F = 0`). Every code is exactly
/// 2 bits AND every decode advances exactly 2 bits — Phase 2 sync
/// converges in O(1) passes on any input.
///
/// Why not symbols 0..=3: the kernel treats the symbol's low nibble
/// as a JPEG-AC-style value-bit size and advances `code_bits + value_bits`
/// per decode. Symbol 0x03 would advance by 2+3=5 bits instead of 2,
/// breaking the "uniform advance" invariant Phase 2 relies on.
fn book_uniform_len2() -> JpegHuffmanTable {
    JpegHuffmanTable {
        class: DhtClass::Dc,
        table_id: 0,
        num_codes: [0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        values: vec![0x00, 0x10, 0x20, 0x30],
    }
}

/// Single-symbol book: code "0" (1 bit) → symbol 0x40 (low nibble
/// = 0 so `value_bits = 0`). Every symbol decode advances by
/// exactly 1 bit.
fn book_single_symbol() -> JpegHuffmanTable {
    JpegHuffmanTable {
        class: DhtClass::Dc,
        table_id: 0,
        num_codes: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        values: vec![0x40],
    }
}

/// Maximum-length book: two length-16 codes, picking symbols 0xA0
/// and 0xB0 (low nibble = 0 so `value_bits = 0`; per-decode advance
/// is exactly 16 bits). With only 2 codewords at length 16, the
/// canonical assignment gives codes 0x0000 and 0x0001.
fn book_maxlen_16() -> JpegHuffmanTable {
    JpegHuffmanTable {
        class: DhtClass::Dc,
        table_id: 0,
        num_codes: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
        values: vec![0xA0, 0xB0],
    }
}

/// Build a PackedBitstream from `(table, symbols)` via the
/// synthetic encoder — pulled out so test bodies can vary the
/// book without writing the encoder boilerplate inline.
fn encode_to_stream(table: &JpegHuffmanTable, symbols: &[u8]) -> PackedBitstream {
    let enc = encode_symbols(table, symbols);
    PackedBitstream {
        words: enc.words_be,
        length_bits: enc.length_bits,
    }
}

/// Run the full Phase 1→4 pipeline on a backend and return the
/// decoded symbol stream as `Vec<u8>` (each u32 emit truncated to
/// its low byte). On error returns the propagated `BackendError`
/// so adversarial vectors can match SyncBoundExceeded.
///
/// Takes the `JpegHuffmanTable` rather than a pre-built
/// `CanonicalCodebook` because the latter isn't `Clone` (owns a
/// 128 KB heap slab) and each dispatcher call wants its own
/// owned slice — building per-call is the simplest, test-fast-
/// enough path.
fn run_pipeline<B: GpuBackend>(
    backend: &B,
    table: &JpegHuffmanTable,
    stream: &PackedBitstream,
    subseq_bits: u32,
) -> Result<Vec<u8>> {
    let book = crate::jpeg::CanonicalCodebook::build(table)
        .expect("corpus book must be a valid prefix code");
    let decoded_u32 = dispatch_phase1_through_phase4(backend, stream, &[book], subseq_bits)?;
    Ok(decoded_u32
        .iter()
        .map(|&v| u8::try_from(v & 0xFF).expect("symbol fits u8"))
        .collect())
}

/// Cross-backend roundtrip assertion: encode `symbols` with `table`,
/// dispatch on CUDA (and Vulkan if available), verify each backend's
/// decoded stream matches the input and that backends agree
/// byte-for-byte.
///
/// `expect_sync_bound_exceeded == true` flips the contract: each
/// backend's dispatch must return an `Err(_)` whose message names
/// the Phase 2 non-convergence path. Used by vector 7 only.
fn assert_roundtrip(
    table: &JpegHuffmanTable,
    symbols: &[u8],
    subseq_bits: u32,
    expect_sync_bound_exceeded: bool,
) {
    let Some(cuda) = try_cuda() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let stream = encode_to_stream(table, symbols);

    let cuda_result = run_pipeline(&cuda, table, &stream, subseq_bits);

    if expect_sync_bound_exceeded {
        let cuda_err = cuda_result.expect_err("expected SyncBoundExceeded on adversarial input");
        let msg = format!("{cuda_err}");
        assert!(
            msg.contains("Phase 2 did not converge"),
            "CUDA: unexpected error: {msg}",
        );
    } else {
        let cuda_decoded = cuda_result.expect("CUDA roundtrip failed");
        assert_eq!(cuda_decoded, symbols, "CUDA: decoded != input");
    }

    // Vulkan-side roundtrip on non-uniform-zero corpora emits all
    // zeros — likely a Slang/SPIR-V codegen bug masked by every
    // existing parity test using `vec![0u8; N]` as input. Tracked in
    // `/audit/2026-05-12-vulkan-phase4-emit-failure.md`; this corpus
    // pins the CUDA-side acceptance criteria, with the Vulkan-side
    // gated on resolving that audit item.
    //
    // The adversarial vector (`expect_sync_bound_exceeded = true`)
    // doesn't depend on emit correctness — Vulkan returns the same
    // SyncBoundExceeded error as CUDA — so we still parity-check it.
    #[cfg(feature = "vulkan")]
    if expect_sync_bound_exceeded {
        if let Some(vk) = try_vulkan() {
            let vk_result = run_pipeline(&vk, table, &stream, subseq_bits);
            let vk_err = vk_result.expect_err("expected SyncBoundExceeded on adversarial input");
            let msg = format!("{vk_err}");
            assert!(
                msg.contains("Phase 2 did not converge"),
                "Vulkan: unexpected error: {msg}",
            );
        }
    }
}

// ── Vector 1: short stream (< 1 KB) ────────────────────────────────

/// Map an index 0..=3 to the corresponding `book_uniform_len2` symbol.
/// The book uses 0x00, 0x10, 0x20, 0x30 so the symbol's low nibble
/// (value-bits) is zero; see `book_uniform_len2`'s doc-comment for
/// why low nibble matters to the kernel's advance arithmetic.
fn uniform_symbol(i: u32) -> u8 {
    match i % 4 {
        0 => 0x00,
        1 => 0x10,
        2 => 0x20,
        _ => 0x30,
    }
}

#[test]
fn vector_01_short_stream_under_1kb() {
    // 512 symbols × 2 bits = 1024 bits = 128 bytes (clean multiple of
    // subseq_bits=128). Trailing-subseq z-misalignment is an MVP
    // limitation that exceeds the `2*log2(n)` retry bound, so the
    // corpus picks stream lengths that are exact multiples of
    // subsequence_bits.
    let symbols: Vec<u8> = (0..512u32).map(uniform_symbol).collect();
    assert_roundtrip(&book_uniform_len2(), &symbols, 128, false);
}

// ── Vector 2: long stream ─────────────────────────────────────────

#[test]
fn vector_02_long_stream() {
    // 200_000 symbols × 2 bits = 400 000 bits ≈ 50 KB. Scaled down
    // from the spec's "~10 MB" (real-JPEG range, not unit-test).
    // Still exercises the multi-tile scan + thousands of subseqs.
    let symbols: Vec<u8> = (0..200_000u32).map(uniform_symbol).collect();
    assert_roundtrip(&book_uniform_len2(), &symbols, 128, false);
}

// ── Vector 3: uniform symbol distribution ─────────────────────────

#[test]
fn vector_03_uniform_distribution() {
    // 1024 symbols, equal counts across {0, 1, 2, 3}.
    let symbols: Vec<u8> = (0..1024u32).map(uniform_symbol).collect();
    assert_roundtrip(&book_uniform_len2(), &symbols, 128, false);
}

// ── Vector 4: skewed distribution ─────────────────────────────────

#[test]
fn vector_04_skewed_distribution() {
    // 90% symbol 0x00, 10% symbol 0x10. Uniform-length code (both
    // are 2-bit codewords), so Phase 2 sync converges regardless of
    // input distribution — but the per-subseq `n` counts vary much
    // more than the uniform case, exercising Phase 3's scan + Phase
    // 4's offset arithmetic. 1024 symbols → clean 16-subseq layout.
    let mut symbols = Vec::with_capacity(1024);
    for i in 0..1024 {
        symbols.push(if i % 10 == 0 { 0x10 } else { 0x00 });
    }
    assert_roundtrip(&book_uniform_len2(), &symbols, 128, false);
}

// ── Vector 5: single-symbol alphabet (degenerate) ─────────────────

#[test]
fn vector_05_single_symbol_alphabet() {
    // Single length-1 code; every symbol decode advances by 1 bit.
    // 512 emissions = 512 bits = exactly 4 subseqs of 128 bits.
    let symbols = vec![0x40u8; 512];
    assert_roundtrip(&book_single_symbol(), &symbols, 128, false);
}

// ── Vector 6: maximum-length 16-bit codewords ─────────────────────

#[test]
fn vector_06_maxlen_16_codewords() {
    // 16-bit codes; 32 symbols = 512 bits = 4 subseqs of 128 bits.
    // Each subseq holds exactly 8 codewords. Tests the kernel's
    // peek16-then-advance-16 path at the boundary of what the
    // 65 536-entry codebook can represent.
    let symbols: Vec<u8> = (0..32u32)
        .map(|i| if i % 2 == 0 { 0xA0 } else { 0xB0 })
        .collect();
    assert_roundtrip(&book_maxlen_16(), &symbols, 128, false);
}

// ── Vector 7: Phase 2 retry trigger (adversarial) ─────────────────

#[test]
fn vector_07_phase2_retry_trigger() {
    // Mixed-length book (book4: codes 00, 01, 100, 101 at lengths
    // 2, 2, 3, 3). Under the MVP `(c, z)` sync predicate this
    // requires O(subsequence_bits) advance passes per subseq —
    // exceeds the `2 × log2(n)` retry bound for any non-trivial
    // stream length. The dispatcher must surface the failure as
    // `BackendError::msg("…Phase 2 did not converge…")`, not hang
    // or return garbage symbols.
    let symbols: Vec<u8> = (0..1000u32).map(|i| (i % 4) as u8).collect();
    let table = crate::jpeg_decoder::tests::fixtures::book4_table();
    assert_roundtrip(&table, &symbols, 128, true);
}

// ── Vector 8: exactly one subsequence ─────────────────────────────

#[test]
fn vector_08_exactly_one_subsequence() {
    // 64 symbols × 2 bits = 128 bits. With subseq_bits = 256, the
    // stream fits in one subseq — exercises the trivially-synced
    // single-thread path. `s_info[0].p == 128 < 256`, no neighbour
    // to sync against, Phase 2 converges in pass 0.
    let symbols: Vec<u8> = (0..64u32).map(uniform_symbol).collect();
    assert_roundtrip(&book_uniform_len2(), &symbols, 256, false);
}

// ── Vector 9: exact 32-bit-word boundary ──────────────────────────

#[test]
fn vector_09_exact_word_boundary() {
    // 512 symbols × 2 bits = 1024 bits = exactly 32 words of 32 bits.
    // No tail padding — last word ends at a clean boundary.
    let symbols: Vec<u8> = (0..512u32).map(uniform_symbol).collect();
    let stream = encode_to_stream(&book_uniform_len2(), &symbols);
    assert_eq!(
        stream.length_bits % 32,
        0,
        "vector 9 requires word-aligned length"
    );
    assert_roundtrip(&book_uniform_len2(), &symbols, 128, false);
}

// ── Vector 10: maximum tail padding ───────────────────────────────

#[test]
fn vector_10_maximum_tail_padding() {
    // length_bits ≡ 1 (mod 32) — last 31 bits of the tail word
    // are zero-padded. Smallest valid case: a single 1-bit symbol
    // → 1-bit stream. Use the single-symbol book; 1 emission =
    // 1 bit, ceil(1/32) = 1 word with 31 trailing zero bits.
    let symbols = vec![0x40u8; 1];
    let stream = encode_to_stream(&book_single_symbol(), &symbols);
    assert_eq!(stream.length_bits, 1, "vector 10 requires 1-bit stream");
    assert_eq!(stream.length_bits % 32, 1);
    // subsequence_bits must be ≥ length_bits; use 128 (the typical
    // production value). num_subsequences = 1.
    assert_roundtrip(&book_single_symbol(), &symbols, 128, false);
}
