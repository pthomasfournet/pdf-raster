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
#[cfg(feature = "gpu-validation")]
use crate::jpeg_decoder::phase2_oracle::{Phase2Outcome, retry_bound as phase2_retry_bound};

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
        sync_flags: None,
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

/// One-shot dispatch of Phase 1 + Phase 2 (bounded retry sync loop).
///
/// Allocates device buffers, uploads inputs, runs Phase 1, then loops
/// Phase 2 dispatches until all subseqs report `sync_flags[i] = 1` or
/// the retry bound is exhausted. Returns the final `s_info` either
/// way; the outcome tells the caller whether convergence happened.
///
/// Test/oracle path only — production callers should drive the trait
/// directly to keep buffers alive across pages.
///
/// # Errors
/// Returns `BackendError` if any alloc/upload/dispatch/download
/// fails. Bound-exceeded is reported via the outcome, not as an
/// error.
#[cfg(feature = "gpu-validation")]
fn dispatch_phase1_then_phase2<B: GpuBackend>(
    backend: &B,
    stream: &PackedBitstream,
    codebooks: &[CanonicalCodebook],
    subsequence_bits: u32,
) -> Result<(Vec<SubsequenceState>, Phase2Outcome)> {
    if stream.length_bits == 0 {
        return Ok((Vec::new(), Phase2Outcome::Converged { iterations: 0 }));
    }
    if subsequence_bits == 0 {
        return Err(BackendError::msg(
            "dispatch_phase1_then_phase2: subsequence_bits must be > 0",
        ));
    }
    if codebooks.is_empty() {
        return Err(BackendError::msg(
            "dispatch_phase1_then_phase2: at least one codebook required",
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

    let bitstream_words = (length_bits.div_ceil(32) as usize) + 1;
    let bitstream_bytes = bitstream_words * 4;
    let codebook_flat = build_gpu_codebook(codebooks);
    let codebook_bytes = codebook_flat.len() * 4;
    let s_info_bytes = (num_subsequences as usize) * std::mem::size_of::<SubsequenceState>();
    let flags_bytes = (num_subsequences as usize) * 4;

    let bitstream_buf = backend.alloc_device_zeroed(bitstream_bytes)?;
    let codebook_buf = backend.alloc_device(codebook_bytes)?;
    let s_info_buf = backend.alloc_device(s_info_bytes)?;
    let sync_flags_buf = backend.alloc_device(flags_bytes)?;

    let _up1 = backend.upload_async(&bitstream_buf, bytemuck::cast_slice(&stream.words))?;
    let _up2 = backend.upload_async(&codebook_buf, bytemuck::cast_slice(&codebook_flat))?;

    // Phase 1 — one pass.
    backend.begin_page()?;
    backend.record_huffman(HuffmanParams {
        bitstream: &bitstream_buf,
        codebook: &codebook_buf,
        s_info: &s_info_buf,
        sync_flags: None,
        length_bits,
        subsequence_bits,
        num_components,
        phase: HuffmanPhase::Phase1IntraSync,
    })?;
    let fence = backend.submit_page()?;
    backend.wait_page(fence)?;

    // Phase 2 — bounded retry loop.
    let bound = phase2_retry_bound(num_subsequences as usize);
    let mut flags_host = vec![0u32; num_subsequences as usize];
    let mut outcome = Phase2Outcome::SyncBoundExceeded { bound };
    for iter in 0..=bound {
        backend.begin_page()?;
        backend.record_huffman(HuffmanParams {
            bitstream: &bitstream_buf,
            codebook: &codebook_buf,
            s_info: &s_info_buf,
            sync_flags: Some(&sync_flags_buf),
            length_bits,
            subsequence_bits,
            num_components,
            phase: HuffmanPhase::Phase2InterSync,
        })?;
        let fence = backend.submit_page()?;
        backend.wait_page(fence)?;

        let dst = bytemuck::cast_slice_mut::<u32, u8>(&mut flags_host);
        let handle = backend.download_async(&sync_flags_buf, dst)?;
        backend.wait_download(handle)?;

        if flags_host.iter().all(|&f| f == 1) {
            outcome = Phase2Outcome::Converged { iterations: iter };
            break;
        }
    }

    let mut s_info_out =
        vec![<SubsequenceState as bytemuck::Zeroable>::zeroed(); num_subsequences as usize];
    let dst = bytemuck::cast_slice_mut::<SubsequenceState, u8>(&mut s_info_out);
    let handle = backend.download_async(&s_info_buf, dst)?;
    backend.wait_download(handle)?;

    backend.free_device(bitstream_buf);
    backend.free_device(codebook_buf);
    backend.free_device(s_info_buf);
    backend.free_device(sync_flags_buf);

    Ok((s_info_out, outcome))
}

/// Phase 3: exclusive prefix-sum of per-subsequence symbol counts.
///
/// Each subsequence's `.n` (symbols decoded between its `start_bit`
/// and the next subsequence's sync point) feeds an exclusive scan;
/// the result is the per-subsequence base offset into the final
/// coefficient stream — i.e., where Phase 4 should write the symbols
/// it re-decodes from that subsequence.
///
/// Thin wrapper over `scan::dispatch_blelloch_scan` — the heavy
/// lifting is already in place.
///
/// Empty input returns `Ok(Vec::new())`.
///
/// # Errors
/// Returns the underlying `BackendError` from
/// `dispatch_blelloch_scan` if any allocation, upload, dispatch, or
/// download fails.
#[cfg(feature = "gpu-validation")]
fn dispatch_phase3_offsets<B: GpuBackend>(
    backend: &B,
    s_info: &[SubsequenceState],
) -> Result<Vec<u32>> {
    let counts: Vec<u32> = s_info.iter().map(|s| s.n).collect();
    crate::jpeg_decoder::scan::dispatch_blelloch_scan(backend, &counts)
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
            let count_to = stream.length_bits.min(start_bit + subseq_bits);
            let (cpu, _stop) = phase1_walk(stream, tables, start_bit, hard_limit, count_to);
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

    /// Phase 2 dispatcher: well-formed stream (`length_bits` is an
    /// exact multiple of `subsequence_bits`) converges on the first
    /// pass. Phase 1's output is already pre-synced.
    #[test]
    fn phase2_well_formed_stream_converges_immediately_cuda() {
        let Some(b) = try_cuda() else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        // 1024 length-2 codewords = 2048 bits, 16 subseqs of 128 bits.
        let stream = book4_stream(&[0x00; 1024]);
        let book = [book4_codebook()];
        let (s_info, outcome) =
            dispatch_phase1_then_phase2(&b, &stream, &book, 128).expect("dispatch");
        // 16 subseqs.
        assert_eq!(s_info.len(), 16);
        // Pre-synced stream — first pass sees all flags = 1.
        assert_eq!(outcome, Phase2Outcome::Converged { iterations: 0 });
    }

    /// Phase 2 dispatcher matches the CPU oracle: run both on the
    /// same input, the final `s_info` should be byte-identical.
    #[test]
    fn phase2_matches_cpu_oracle_well_formed_cuda() {
        let Some(b) = try_cuda() else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        let stream = book4_stream(&[0x00; 1024]);
        let book = [book4_codebook()];

        // GPU: Phase 1 + Phase 2.
        let (gpu_s_info, _outcome) =
            dispatch_phase1_then_phase2(&b, &stream, &book, 128).expect("dispatch");

        // CPU: Phase 1 + Phase 2 in the same shape.
        let mut cpu_s_info: Vec<SubsequenceState> = (0u32..16)
            .map(|i| {
                let start_bit = i * 128;
                let hard_limit = stream.length_bits.min(start_bit + 2 * 128);
                let count_to = stream.length_bits.min(start_bit + 128);
                let (state, _stop) = phase1_walk(&stream, &book, start_bit, hard_limit, count_to);
                state
            })
            .collect();
        let cpu_outcome = crate::jpeg_decoder::phase2_oracle::phase2_run_to_sync(
            &mut cpu_s_info,
            &stream,
            &book,
            128,
        );
        assert!(matches!(
            cpu_outcome,
            crate::jpeg_decoder::phase2_oracle::Phase2Outcome::Converged { .. }
        ));

        assert_eq!(gpu_s_info, cpu_s_info);
    }

    /// Phase 3 dispatcher: exclusive-scan over `s_info[i].n` produces
    /// write offsets that match the CPU exclusive-scan reference.
    /// The pre-synced stream means every subseq's `n` equals its
    /// subsequence-bit / codeword-length ratio, but the test asserts
    /// the scan property, not the specific count values.
    #[test]
    fn phase3_offsets_match_exclusive_scan_cuda() {
        let Some(b) = try_cuda() else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        let stream = book4_stream(&[0x00; 1024]);
        let book = [book4_codebook()];
        let (s_info, _outcome) =
            dispatch_phase1_then_phase2(&b, &stream, &book, 128).expect("phase1+2");
        let offsets = dispatch_phase3_offsets(&b, &s_info).expect("phase3");

        assert_eq!(offsets.len(), s_info.len());
        assert_eq!(offsets[0], 0, "exclusive scan starts at 0");
        for i in 1..offsets.len() {
            assert_eq!(
                offsets[i],
                offsets[i - 1] + s_info[i - 1].n,
                "offset[{i}] = offset[{}] + n[{}]",
                i - 1,
                i - 1
            );
        }
    }

    /// Phase 3 with a single subsequence: scan returns `[0]`.
    #[test]
    fn phase3_single_subsequence_cuda() {
        let Some(b) = try_cuda() else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        // 100 symbols of length 2 = 200 bits in one 256-bit subseq.
        let stream = book4_stream(&[0x00; 100]);
        let book = [book4_codebook()];
        let (s_info, _outcome) =
            dispatch_phase1_then_phase2(&b, &stream, &book, 256).expect("phase1+2");
        assert_eq!(s_info.len(), 1);
        let offsets = dispatch_phase3_offsets(&b, &s_info).expect("phase3");
        assert_eq!(offsets, vec![0]);
    }

    /// Phase 3 with empty input is a no-op — Vec::new() round-trip.
    #[test]
    fn phase3_empty_input_cuda() {
        let Some(b) = try_cuda() else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        let offsets = dispatch_phase3_offsets(&b, &[]).expect("phase3 empty");
        assert!(offsets.is_empty());
    }

    /// Phase 3 over Phase 2's `s_info` matches a host-side scan of
    /// the same `.n` values. This is the contract Phase 3 must
    /// guarantee — converting the *meaning* of `.n` to "symbols in
    /// this subseq's own region" is Phase 4's job (re-decode pass).
    #[test]
    fn phase3_matches_cpu_exclusive_scan_mixed_cuda() {
        let Some(b) = try_cuda() else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        let symbols: Vec<u8> = (0..1000u32).map(|i| (i % 4) as u8).collect();
        let stream = book4_stream(&symbols);
        let book = [book4_codebook()];
        let (s_info, _outcome) =
            dispatch_phase1_then_phase2(&b, &stream, &book, 128).expect("phase1+2");

        let counts: Vec<u32> = s_info.iter().map(|s| s.n).collect();
        let cpu_off = crate::jpeg_decoder::scan::test_helpers::cpu_exclusive_scan(&counts);
        let gpu_off = dispatch_phase3_offsets(&b, &s_info).expect("phase3");
        assert_eq!(gpu_off, cpu_off);
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
            let count_to = stream.length_bits.min(start_bit + subseq_bits);
            let (cpu, _stop) = phase1_walk(stream, tables, start_bit, hard_limit, count_to);
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
            let count_to = stream.length_bits.min(start_bit + subseq_bits);
            let (cpu, _stop) = phase1_walk(stream, tables, start_bit, hard_limit, count_to);
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

    /// Phase 2 dispatcher: well-formed stream converges on both
    /// backends and produces byte-identical `s_info`.
    #[test]
    fn phase2_well_formed_stream_cuda_vs_vulkan() {
        let (Some(cuda), Some(vk)) = (try_cuda(), try_vulkan()) else {
            eprintln!("skipping: need both CUDA and Vulkan");
            return;
        };
        let stream = book4_stream(&[0x00; 1024]);
        let book = [book4_codebook()];
        let (cuda_s, cuda_o) =
            dispatch_phase1_then_phase2(&cuda, &stream, &book, 128).expect("cuda");
        let (vk_s, vk_o) = dispatch_phase1_then_phase2(&vk, &stream, &book, 128).expect("vulkan");
        assert_eq!(cuda_o, vk_o, "outcome mismatch");
        assert_eq!(cuda_s, vk_s, "s_info mismatch");
        assert_eq!(cuda_o, Phase2Outcome::Converged { iterations: 0 });
    }

    /// Phase 3 dispatcher: same input through both backends yields
    /// byte-identical offset vectors, and both match the CPU
    /// exclusive-scan reference.
    #[test]
    fn phase3_offsets_cuda_vs_vulkan() {
        let (Some(cuda), Some(vk)) = (try_cuda(), try_vulkan()) else {
            eprintln!("skipping: need both CUDA and Vulkan");
            return;
        };
        // Mixed symbols to get a non-trivial `n` distribution.
        let symbols: Vec<u8> = (0..1000u32).map(|i| (i % 4) as u8).collect();
        let stream = book4_stream(&symbols);
        let book = [book4_codebook()];
        let (cuda_s, _) =
            dispatch_phase1_then_phase2(&cuda, &stream, &book, 128).expect("cuda phase1+2");
        let (vk_s, _) =
            dispatch_phase1_then_phase2(&vk, &stream, &book, 128).expect("vulkan phase1+2");
        // Pin upstream agreement before pinning Phase 3 — otherwise a
        // Phase 1/2 divergence on the mixed corpus would surface as a
        // confusing "Vulkan diverges from CPU oracle" (the CPU oracle
        // is built from CUDA's counts).
        assert_eq!(
            cuda_s, vk_s,
            "Phase 1+2 must agree before Phase 3 can be compared"
        );
        let cuda_off = dispatch_phase3_offsets(&cuda, &cuda_s).expect("cuda phase3");
        let vk_off = dispatch_phase3_offsets(&vk, &vk_s).expect("vulkan phase3");

        let counts: Vec<u32> = cuda_s.iter().map(|s| s.n).collect();
        let cpu_off = crate::jpeg_decoder::scan::test_helpers::cpu_exclusive_scan(&counts);

        assert_eq!(cuda_off, cpu_off, "CUDA diverges from CPU oracle");
        assert_eq!(vk_off, cpu_off, "Vulkan diverges from CPU oracle");
        assert_eq!(cuda_off, vk_off, "CUDA and Vulkan disagree");
    }
}
