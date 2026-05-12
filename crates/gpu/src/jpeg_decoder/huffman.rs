//! Host-side dispatch for the parallel-Huffman JPEG decoder kernels.
//!
//! Mirrors the layout of `jpeg_decoder::scan`: a one-shot
//! `dispatch_phase1_intra_sync` that owns the buffers it allocates,
//! plus a `build_gpu_codebook` helper that flattens a slice of
//! `CanonicalCodebook` into the device-side `u32[num_components *
//! 65536]` layout the kernel expects.
//!
//! Phase 1 today; Phase 2 / 4 dispatchers add similar entries when
//! their kernels land.

use crate::backend::params::{HUFFMAN_CODEBOOK_ENTRIES, HuffmanParams, HuffmanPhase};
use crate::backend::{BackendError, GpuBackend, Result};
use crate::jpeg::CanonicalCodebook;
use crate::jpeg_decoder::PackedBitstream;

/// Per-subsequence decoder state emitted by Phase 1.
///
/// Mirrors the kernel's `uint4 = (p, n, c, z)` output one-for-one
/// so test assertions and downstream phases can deserialize the
/// device buffer directly via [`bytemuck::cast_slice`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(C)]
pub struct SubsequenceState {
    /// Absolute bit position the thread reached.
    pub p: u32,
    /// Number of symbols decoded since `start_bit`.
    pub n: u32,
    /// Current component index.
    pub c: u32,
    /// Current zig-zag index within the 64-coefficient block.
    pub z: u32,
}

// SAFETY: `repr(C)` over four `u32` fields; the layout is `[u32; 4]`
// in declaration order, so reinterpreting bytes-as-SubsequenceState
// matches the kernel's `uint4` write.
unsafe impl bytemuck::Zeroable for SubsequenceState {}
unsafe impl bytemuck::Pod for SubsequenceState {}

/// Flatten a slice of canonical Huffman codebooks into the GPU layout.
///
/// Layout: `u32[num_components * 65536]`, where each entry packs
/// `(num_bits << 8) | symbol` (matching the kernel's `entry >> 8`
/// / `entry & 0xFF` decode).
///
/// The CPU-side `CanonicalCodebook` has a 65 536-entry LUT of
/// `(num_bits: u8, symbol: u8)` pairs; this is just a host-side
/// repack into u32 so the device side can read a single 32-bit
/// load per peek.
#[must_use]
pub fn build_gpu_codebook(codebooks: &[CanonicalCodebook]) -> Vec<u32> {
    let mut out = Vec::with_capacity(codebooks.len() * HUFFMAN_CODEBOOK_ENTRIES);
    for book in codebooks {
        for entry in book.table() {
            // entry.num_bits == 0 → kernel reads num_bits=0 and breaks
            // the decode loop (PrefixMiss). No special-casing needed.
            let packed = (u32::from(entry.num_bits) << 8) | u32::from(entry.symbol);
            out.push(packed);
        }
    }
    out
}

/// One-shot dispatch of the Phase 1 intra-sync kernel.
///
/// Allocates device buffers (bitstream + codebook + `s_info`),
/// uploads the inputs, runs the kernel, downloads the per-subsequence
/// state, and frees the buffers. For per-page reuse, callers should
/// keep their own buffers and use the trait methods directly; this
/// entry point is the test/oracle path.
///
/// # Errors
/// Returns the underlying `BackendError` if any allocation, upload,
/// dispatch, or download fails.
///
/// # Panics
/// Panics if the input bitstream has zero length — Phase 1 is
/// only meaningful for non-empty streams.
pub fn dispatch_phase1_intra_sync<B: GpuBackend>(
    backend: &B,
    stream: &PackedBitstream,
    codebooks: &[CanonicalCodebook],
    subsequence_bits: u32,
) -> Result<Vec<SubsequenceState>> {
    assert!(
        stream.length_bits > 0,
        "dispatch_phase1_intra_sync: empty bitstream"
    );
    assert!(
        subsequence_bits > 0,
        "dispatch_phase1_intra_sync: subsequence_bits must be > 0"
    );
    assert!(
        !codebooks.is_empty(),
        "dispatch_phase1_intra_sync: at least one codebook required"
    );

    let num_components = u32::try_from(codebooks.len()).map_err(|_| {
        BackendError::msg(format!(
            "more codebooks ({}) than fit in u32 — caller bug",
            codebooks.len()
        ))
    })?;
    let length_bits = stream.length_bits;
    let num_subsequences = length_bits.div_ceil(subsequence_bits);

    // Bitstream needs at least ceil(length_bits/32) words; the kernel's
    // peek16 also reads `word_idx + 1`, so we allocate one extra word
    // of zero-padded headroom.
    let bitstream_words = (length_bits.div_ceil(32) as usize) + 1;
    let bitstream_bytes = bitstream_words * 4;
    let mut bitstream_padded = vec![0u32; bitstream_words];
    bitstream_padded[..stream.words.len()].copy_from_slice(&stream.words);

    let codebook_flat = build_gpu_codebook(codebooks);
    let codebook_bytes = codebook_flat.len() * 4;
    let s_info_bytes = (num_subsequences as usize) * std::mem::size_of::<SubsequenceState>();

    let bitstream_buf = backend.alloc_device(bitstream_bytes)?;
    let codebook_buf = backend.alloc_device(codebook_bytes)?;
    let s_info_buf = backend.alloc_device(s_info_bytes)?;

    let up1 = backend.upload_async(&bitstream_buf, bytemuck::cast_slice(&bitstream_padded))?;
    backend.wait_transfer(up1)?;
    let up2 = backend.upload_async(&codebook_buf, bytemuck::cast_slice(&codebook_flat))?;
    backend.wait_transfer(up2)?;

    backend.begin_page()?;
    backend.record_huffman(HuffmanParams {
        bitstream: &bitstream_buf,
        codebook: &codebook_buf,
        s_info: &s_info_buf,
        length_bits,
        subsequence_bits,
        num_subsequences,
        num_components,
        phase: HuffmanPhase::Phase1IntraSync,
    })?;
    let fence = backend.submit_page()?;
    backend.wait_page(fence)?;

    let mut out_bytes = vec![0u8; s_info_bytes];
    let handle = backend.download_async(&s_info_buf, &mut out_bytes)?;
    backend.wait_download(handle)?;

    backend.free_device(bitstream_buf);
    backend.free_device(codebook_buf);
    backend.free_device(s_info_buf);

    Ok(bytemuck::cast_slice::<u8, SubsequenceState>(&out_bytes).to_vec())
}

#[cfg(all(test, feature = "gpu-validation"))]
mod tests {
    use super::*;
    use crate::backend::cuda::CudaBackend;
    use crate::jpeg::headers::{DhtClass, JpegHuffmanTable};
    use crate::jpeg_decoder::pack_be_words;
    use crate::jpeg_decoder::tests::synthetic::encode_symbols;

    fn book4_table() -> JpegHuffmanTable {
        JpegHuffmanTable {
            class: DhtClass::Dc,
            table_id: 0,
            num_codes: [0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            values: vec![0x00, 0x01, 0x02, 0x03],
        }
    }

    fn book4_codebook() -> CanonicalCodebook {
        CanonicalCodebook::build(&book4_table()).unwrap()
    }

    fn try_cuda() -> Option<CudaBackend> {
        CudaBackend::new().ok()
    }

    fn stream_from(symbols: &[u8]) -> PackedBitstream {
        let enc = encode_symbols(&book4_table(), symbols);
        pack_be_words(bytemuck::cast_slice(&enc.words_be), enc.length_bits)
    }

    /// CPU oracle that mirrors `phase1_oracle::phase1_walk` for one
    /// subsequence, returning the kernel-shape `SubsequenceState`.
    /// Duplicated here (one match against the oracle in
    /// `phase1_oracle.rs`) because the oracle is private to its own
    /// module — keeping the dispatcher test self-contained.
    fn cpu_phase1(
        stream: &PackedBitstream,
        tables: &[CanonicalCodebook],
        seq_idx: u32,
        subsequence_bits: u32,
    ) -> SubsequenceState {
        let start_bit = seq_idx * subsequence_bits;
        let sync_target = start_bit + 2 * subsequence_bits;
        let hard_limit = stream.length_bits.min(sync_target);
        let num_components = tables.len() as u32;
        let mut p = start_bit;
        let mut n: u32 = 0;
        let mut c: u32 = 0;
        let mut z: u32 = 0;
        loop {
            if p >= hard_limit {
                break;
            }
            let peek = crate::jpeg_decoder::bitstream::peek16(stream, u64::from(p));
            let entry = tables[c as usize].lookup(peek);
            if entry.num_bits == 0 {
                break;
            }
            let advance = u32::from(entry.num_bits) + u32::from(entry.symbol & 0x0F);
            if p + advance > stream.length_bits {
                break;
            }
            p += advance;
            n = n.saturating_add(1);
            z = (z + 1) % 64;
            if z == 0 {
                c = (c + 1) % num_components;
            }
        }
        SubsequenceState { p, n, c, z }
    }

    /// Run the kernel + run the CPU oracle for every subsequence;
    /// assert byte-identical state per subsequence.
    fn assert_phase1_parity(
        stream: &PackedBitstream,
        tables: &[CanonicalCodebook],
        subseq_bits: u32,
    ) {
        let Some(b) = try_cuda() else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        let gpu = dispatch_phase1_intra_sync(&b, stream, tables, subseq_bits).expect("dispatch ok");
        let num_subseq = stream.length_bits.div_ceil(subseq_bits);
        assert_eq!(gpu.len() as u32, num_subseq);
        for seq_idx in 0..num_subseq {
            let cpu = cpu_phase1(stream, tables, seq_idx, subseq_bits);
            assert_eq!(
                gpu[seq_idx as usize], cpu,
                "subsequence {seq_idx} GPU vs CPU state mismatch"
            );
        }
    }

    #[test]
    fn phase1_single_subsequence_matches_cpu() {
        // 100 length-2 codewords = 200 bits; one subsequence of 256 bits
        // (length_bits 200 < 256 → still one subsequence).
        let symbols = vec![0x00; 100];
        let stream = stream_from(&symbols);
        let book = [book4_codebook()];
        assert_phase1_parity(&stream, &book, 256);
    }

    #[test]
    fn phase1_two_subsequences_match_cpu() {
        // 128 length-2 codewords = 256 bits; two subsequences of 128 bits each.
        let symbols = vec![0x00; 128];
        let stream = stream_from(&symbols);
        let book = [book4_codebook()];
        assert_phase1_parity(&stream, &book, 128);
    }

    #[test]
    fn phase1_many_subsequences_match_cpu() {
        // 1000 length-2 codewords = 2000 bits; ~16 subsequences of 128 bits each.
        let symbols = vec![0x00; 1000];
        let stream = stream_from(&symbols);
        let book = [book4_codebook()];
        assert_phase1_parity(&stream, &book, 128);
    }

    #[test]
    fn phase1_mixed_symbols_match_cpu() {
        // 1000 symbols cycling through 0x00, 0x01, 0x02, 0x03; mixed
        // codeword lengths exercise both length-2 and length-3 codes.
        let symbols: Vec<u8> = (0..1000u32).map(|i| (i % 4) as u8).collect();
        let stream = stream_from(&symbols);
        let book = [book4_codebook()];
        assert_phase1_parity(&stream, &book, 128);
    }

    #[test]
    fn phase1_multi_component_match_cpu() {
        let symbols = vec![0x00; 200];
        let stream = stream_from(&symbols);
        // Three identical components — z rolls over every 64 symbols,
        // c walks 0 → 1 → 2 → 0.
        let book = [book4_codebook(), book4_codebook(), book4_codebook()];
        assert_phase1_parity(&stream, &book, 64);
    }

    #[test]
    fn build_gpu_codebook_packs_entries_correctly() {
        // 1-entry table: symbol 42 at length 1 (code "0"). Half the
        // 65536-entry LUT is filled (prefixes starting with 0).
        let table = JpegHuffmanTable {
            class: DhtClass::Dc,
            table_id: 0,
            num_codes: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            values: vec![42],
        };
        let book = CanonicalCodebook::build(&table).unwrap();
        let flat = build_gpu_codebook(&[book]);
        assert_eq!(flat.len(), HUFFMAN_CODEBOOK_ENTRIES);
        // Entry for "0..." (prefix < 0x8000): num_bits=1, symbol=42.
        // Packed = (1 << 8) | 42 = 256 + 42 = 298.
        assert_eq!(flat[0], 298);
        assert_eq!(flat[0x7FFF], 298);
        // Entry for "1..." (prefix >= 0x8000): no codeword, num_bits=0.
        assert_eq!(flat[0x8000], 0);
    }
}
