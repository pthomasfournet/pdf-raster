//! Host-side dispatch for the parallel-Huffman JPEG decoder kernels.
//!
//! `build_gpu_codebook` flattens a slice of `CanonicalCodebook` into
//! the device-side `u32[num_components * 65536]` layout the kernel
//! expects. `dispatch_phase1_intra_sync` is the one-shot test/oracle
//! entry point that owns its buffers; production callers should drive
//! the trait's `record_huffman` directly to keep buffers alive across
//! pages.

use crate::backend::params::{HUFFMAN_CODEBOOK_ENTRIES, HuffmanParams, HuffmanPhase};
use crate::backend::{BackendError, GpuBackend, Result};
use crate::jpeg::CanonicalCodebook;
use crate::jpeg_decoder::PackedBitstream;
use crate::jpeg_decoder::phase1_oracle::SubsequenceState;

/// Flatten a slice of canonical Huffman codebooks into the GPU layout:
/// `u32[num_components * 65536]`, each entry packing
/// `(num_bits << 8) | symbol`.
///
/// `num_bits == 0` (no codeword matches the prefix) is preserved
/// verbatim — the kernel uses that as its miss sentinel.
#[must_use]
fn build_gpu_codebook(codebooks: &[CanonicalCodebook]) -> Vec<u32> {
    let mut out = Vec::with_capacity(codebooks.len() * HUFFMAN_CODEBOOK_ENTRIES);
    for book in codebooks {
        for entry in book.table() {
            let packed = (u32::from(entry.num_bits) << 8) | u32::from(entry.symbol);
            out.push(packed);
        }
    }
    out
}

/// One-shot dispatch of the Phase 1 intra-sync kernel.
///
/// Allocates device buffers, uploads inputs, runs the kernel,
/// downloads per-subsequence state, frees buffers. Test/oracle path
/// only — per-page callers should keep buffers and drive the trait's
/// `record_huffman` directly.
///
/// Empty bitstream returns `Ok(Vec::new())` as a no-op; other invalid
/// inputs (zero `subsequence_bits`, no codebooks) return a typed
/// `BackendError` rather than panicking — matches the sibling
/// `scan::dispatch_blelloch_scan` style.
///
/// # Errors
/// Returns `BackendError` for invalid arguments, or the underlying
/// driver error if any allocation, upload, dispatch, or download
/// fails.
fn dispatch_phase1_intra_sync<B: GpuBackend>(
    backend: &B,
    stream: &PackedBitstream,
    codebooks: &[CanonicalCodebook],
    subsequence_bits: u32,
) -> Result<Vec<SubsequenceState>> {
    if stream.length_bits == 0 {
        return Ok(Vec::new());
    }
    if subsequence_bits == 0 {
        return Err(BackendError::msg(
            "dispatch_phase1_intra_sync: subsequence_bits must be > 0",
        ));
    }
    if codebooks.is_empty() {
        return Err(BackendError::msg(
            "dispatch_phase1_intra_sync: at least one codebook required",
        ));
    }

    let num_components = u32::try_from(codebooks.len()).map_err(|_| {
        BackendError::msg(format!(
            "more codebooks ({}) than fit in u32",
            codebooks.len()
        ))
    })?;
    let length_bits = stream.length_bits;
    let num_subsequences = length_bits.div_ceil(subsequence_bits);

    // peek16 reads `word_idx + 1` even at the stream tail, so the
    // bitstream buffer needs one extra zero-padded word of headroom
    // past `ceil(length_bits / 32)`.
    let bitstream_words = (length_bits.div_ceil(32) as usize) + 1;
    let bitstream_bytes = bitstream_words * 4;

    let codebook_flat = build_gpu_codebook(codebooks);
    let codebook_bytes = codebook_flat.len() * 4;
    let s_info_bytes = (num_subsequences as usize) * std::mem::size_of::<SubsequenceState>();

    // alloc_device_zeroed gives the trailing peek16-headroom word for
    // free — we then upload only the stream's actual words, leaving
    // the tail zero. Avoids a host-side Vec<u32> mirror of the full
    // bitstream.
    let bitstream_buf = backend.alloc_device_zeroed(bitstream_bytes)?;
    let codebook_buf = backend.alloc_device(codebook_bytes)?;
    let s_info_buf = backend.alloc_device(s_info_bytes)?;

    // Both uploads queue on the transfer queue; `submit_page` inserts
    // the cross-queue barrier before the kernel reads the buffers, so
    // explicit wait_transfer between uploads is unnecessary.
    let _up1 = backend.upload_async(&bitstream_buf, bytemuck::cast_slice(&stream.words))?;
    let _up2 = backend.upload_async(&codebook_buf, bytemuck::cast_slice(&codebook_flat))?;

    backend.begin_page()?;
    backend.record_huffman(HuffmanParams {
        bitstream: &bitstream_buf,
        codebook: &codebook_buf,
        s_info: &s_info_buf,
        length_bits,
        subsequence_bits,
        num_components,
        phase: HuffmanPhase::Phase1IntraSync,
    })?;
    let fence = backend.submit_page()?;
    backend.wait_page(fence)?;

    // Download directly into a Vec<SubsequenceState> view, skipping
    // the Vec<u8> → cast → to_vec double-copy. SubsequenceState is
    // Pod+Zeroable so `bytemuck::zeroed::<T>()` is the canonical
    // zero-value rather than a field-by-field literal.
    let mut out =
        vec![<SubsequenceState as bytemuck::Zeroable>::zeroed(); num_subsequences as usize];
    let dst = bytemuck::cast_slice_mut::<SubsequenceState, u8>(&mut out);
    let handle = backend.download_async(&s_info_buf, dst)?;
    backend.wait_download(handle)?;

    backend.free_device(bitstream_buf);
    backend.free_device(codebook_buf);
    backend.free_device(s_info_buf);

    Ok(out)
}

#[cfg(all(test, feature = "gpu-validation"))]
mod tests {
    use super::*;
    use crate::backend::cuda::CudaBackend;
    use crate::jpeg::headers::{DhtClass, JpegHuffmanTable};
    use crate::jpeg_decoder::phase1_oracle::phase1_walk;
    use crate::jpeg_decoder::tests::fixtures::{book4_codebook, book4_stream};

    fn try_cuda() -> Option<CudaBackend> {
        CudaBackend::new().ok()
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
        assert_eq!(
            u32::try_from(gpu.len()).expect("subseq count fits u32"),
            num_subseq
        );
        for seq_idx in 0..num_subseq {
            let start_bit = seq_idx * subseq_bits;
            let hard_limit = stream.length_bits.min(start_bit + 2 * subseq_bits);
            let (cpu, _stop) = phase1_walk(stream, tables, start_bit, hard_limit);
            assert_eq!(
                gpu[seq_idx as usize], cpu,
                "subsequence {seq_idx} GPU vs CPU state mismatch"
            );
        }
    }

    #[test]
    fn phase1_single_subsequence_matches_cpu() {
        // 100 length-2 codewords = 200 bits in one 256-bit subsequence.
        let symbols = vec![0x00; 100];
        let stream = book4_stream(&symbols);
        let book = [book4_codebook()];
        assert_phase1_parity(&stream, &book, 256);
    }

    #[test]
    fn phase1_two_subsequences_match_cpu() {
        let symbols = vec![0x00; 128];
        let stream = book4_stream(&symbols);
        let book = [book4_codebook()];
        assert_phase1_parity(&stream, &book, 128);
    }

    #[test]
    fn phase1_many_subsequences_match_cpu() {
        let symbols = vec![0x00; 1000];
        let stream = book4_stream(&symbols);
        let book = [book4_codebook()];
        assert_phase1_parity(&stream, &book, 128);
    }

    #[test]
    fn phase1_mixed_symbols_match_cpu() {
        // Mixed codeword lengths exercise both length-2 and length-3 codes.
        let symbols: Vec<u8> = (0..1000u32).map(|i| (i % 4) as u8).collect();
        let stream = book4_stream(&symbols);
        let book = [book4_codebook()];
        assert_phase1_parity(&stream, &book, 128);
    }

    #[test]
    fn phase1_multi_component_match_cpu() {
        let symbols = vec![0x00; 200];
        let stream = book4_stream(&symbols);
        // Three components — z rolls over every 64 symbols,
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
        // (1 << 8) | 42 = 298. Prefixes < 0x8000 → length-1 "0" match.
        assert_eq!(flat[0], 298);
        assert_eq!(flat[0x7FFF], 298);
        // Prefixes ≥ 0x8000 → no codeword, num_bits=0.
        assert_eq!(flat[0x8000], 0);
    }
}

#[cfg(all(test, feature = "vulkan", feature = "gpu-validation"))]
mod vulkan_tests {
    use super::*;
    use crate::backend::cuda::CudaBackend;
    use crate::backend::vulkan::VulkanBackend;
    use crate::jpeg_decoder::phase1_oracle::phase1_walk;
    use crate::jpeg_decoder::tests::fixtures::{book4_codebook, book4_stream};

    fn try_vulkan() -> Option<VulkanBackend> {
        VulkanBackend::new().ok()
    }

    fn try_cuda() -> Option<CudaBackend> {
        CudaBackend::new().ok()
    }

    /// Run the kernel on both backends + the CPU oracle for every
    /// subsequence; assert byte-identical state across all three.
    fn cross_backend_parity(
        stream: &PackedBitstream,
        tables: &[CanonicalCodebook],
        subseq_bits: u32,
    ) {
        let (Some(cuda), Some(vk)) = (try_cuda(), try_vulkan()) else {
            eprintln!("skipping: need both CUDA and Vulkan");
            return;
        };
        let gpu_cuda =
            dispatch_phase1_intra_sync(&cuda, stream, tables, subseq_bits).expect("cuda dispatch");
        let gpu_vk =
            dispatch_phase1_intra_sync(&vk, stream, tables, subseq_bits).expect("vulkan dispatch");
        let num_subseq = stream.length_bits.div_ceil(subseq_bits);
        for seq_idx in 0..num_subseq {
            let start_bit = seq_idx * subseq_bits;
            let hard_limit = stream.length_bits.min(start_bit + 2 * subseq_bits);
            let (cpu, _stop) = phase1_walk(stream, tables, start_bit, hard_limit);
            let i = seq_idx as usize;
            assert_eq!(gpu_cuda[i], cpu, "subseq {seq_idx}: CUDA vs CPU");
            assert_eq!(gpu_vk[i], cpu, "subseq {seq_idx}: Vulkan vs CPU");
            assert_eq!(gpu_cuda[i], gpu_vk[i], "subseq {seq_idx}: CUDA vs Vulkan");
        }
    }

    /// Vulkan-only parity check (skipped if Vulkan is unavailable but
    /// CUDA is — keeps the test useful on Vulkan-only hosts).
    fn vulkan_vs_cpu(stream: &PackedBitstream, tables: &[CanonicalCodebook], subseq_bits: u32) {
        let Some(vk) = try_vulkan() else {
            eprintln!("skipping: no Vulkan device");
            return;
        };
        let gpu = dispatch_phase1_intra_sync(&vk, stream, tables, subseq_bits).expect("vulkan");
        let num_subseq = stream.length_bits.div_ceil(subseq_bits);
        for seq_idx in 0..num_subseq {
            let start_bit = seq_idx * subseq_bits;
            let hard_limit = stream.length_bits.min(start_bit + 2 * subseq_bits);
            let (cpu, _stop) = phase1_walk(stream, tables, start_bit, hard_limit);
            assert_eq!(gpu[seq_idx as usize], cpu, "subseq {seq_idx} Vulkan vs CPU");
        }
    }

    #[test]
    fn phase1_single_subsequence_vulkan_vs_cpu() {
        let stream = book4_stream(&[0x00; 100]);
        let book = [book4_codebook()];
        vulkan_vs_cpu(&stream, &book, 256);
    }

    #[test]
    fn phase1_many_subsequences_vulkan_vs_cpu() {
        let stream = book4_stream(&[0x00; 1000]);
        let book = [book4_codebook()];
        vulkan_vs_cpu(&stream, &book, 128);
    }

    #[test]
    fn phase1_mixed_symbols_vulkan_vs_cpu() {
        let symbols: Vec<u8> = (0..1000u32).map(|i| (i % 4) as u8).collect();
        let stream = book4_stream(&symbols);
        let book = [book4_codebook()];
        vulkan_vs_cpu(&stream, &book, 128);
    }

    #[test]
    fn phase1_multi_component_vulkan_vs_cpu() {
        let stream = book4_stream(&[0x00; 200]);
        let book = [book4_codebook(), book4_codebook(), book4_codebook()];
        vulkan_vs_cpu(&stream, &book, 64);
    }

    #[test]
    fn cuda_vs_vulkan_single_subsequence() {
        let stream = book4_stream(&[0x00; 100]);
        let book = [book4_codebook()];
        cross_backend_parity(&stream, &book, 256);
    }

    #[test]
    fn cuda_vs_vulkan_many_subsequences() {
        let stream = book4_stream(&[0x00; 1000]);
        let book = [book4_codebook()];
        cross_backend_parity(&stream, &book, 128);
    }

    #[test]
    fn cuda_vs_vulkan_mixed_symbols() {
        let symbols: Vec<u8> = (0..1000u32).map(|i| (i % 4) as u8).collect();
        let stream = book4_stream(&symbols);
        let book = [book4_codebook()];
        cross_backend_parity(&stream, &book, 128);
    }

    #[test]
    fn cuda_vs_vulkan_multi_component() {
        let stream = book4_stream(&[0x00; 200]);
        let book = [book4_codebook(), book4_codebook(), book4_codebook()];
        cross_backend_parity(&stream, &book, 64);
    }
}
