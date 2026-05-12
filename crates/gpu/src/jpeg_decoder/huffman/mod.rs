//! Host-side dispatch for the parallel-Huffman JPEG decoder kernels.
//!
//! `build_gpu_codebook` flattens a slice of `CanonicalCodebook` into
//! the device-side `u32[num_components * 65536]` layout the kernel
//! expects. `dispatch_phase1_intra_sync` is the one-shot test/oracle
//! entry point that owns its buffers; production callers should drive
//! the trait's `record_huffman` directly to keep buffers alive across
//! pages.

// Per-phase parity tests live in this file's `mod tests` /
// `mod vulkan_tests`; the 10-vector spec acceptance corpus lives in
// the sibling `corpus.rs`.
#[cfg(all(test, feature = "gpu-validation"))]
mod corpus;

use crate::backend::params::{HUFFMAN_CODEBOOK_ENTRIES, HuffmanParams, HuffmanPhase};
use crate::backend::{BackendError, GpuBackend, Result};
use crate::jpeg::CanonicalCodebook;
use crate::jpeg_decoder::PackedBitstream;
use crate::jpeg_decoder::dispatch_util::DeviceBufferGuard;
use crate::jpeg_decoder::phase1_oracle::SubsequenceState;
#[cfg(feature = "gpu-validation")]
use crate::jpeg_decoder::phase2_oracle::{Phase2Outcome, retry_bound as phase2_retry_bound};

/// Flatten an iterator of `&CanonicalCodebook` into the GPU layout:
/// `u32[num_components * 65536]`, each entry packing `(num_bits << 8) | symbol`.
///
/// `num_bits == 0` (no codeword matches the prefix) is preserved verbatim —
/// the kernel uses that as its miss sentinel.
fn flatten_codebooks<'a>(
    codebooks: impl ExactSizeIterator<Item = &'a CanonicalCodebook>,
) -> Vec<u32> {
    let mut out = Vec::with_capacity(codebooks.len() * HUFFMAN_CODEBOOK_ENTRIES);
    for book in codebooks {
        for entry in book.table() {
            out.push((u32::from(entry.num_bits) << 8) | u32::from(entry.symbol));
        }
    }
    out
}

#[must_use]
fn build_gpu_codebook(codebooks: &[CanonicalCodebook]) -> Vec<u32> {
    flatten_codebooks(codebooks.iter())
}

#[must_use]
fn build_gpu_codebook_refs(codebooks: &[&CanonicalCodebook]) -> Vec<u32> {
    flatten_codebooks(codebooks.iter().copied())
}

/// One-shot JPEG Phase 1 dispatch using real JPEG framing.
///
/// Builds the MCU schedule + flat codebook buffers from `prep`, uploads
/// them alongside the entropy bitstream, runs `JpegPhase1IntraSync`,
/// downloads the per-subsequence state, and frees all device buffers.
///
/// # Errors
/// Returns `BackendError` for any alloc / upload / dispatch / download
/// failure, or if `build_mcu_schedule` rejects an out-of-range selector.
fn dispatch_jpeg_phase1_intra_sync<B: GpuBackend>(
    backend: &B,
    prep: &crate::jpeg_decoder::JpegPreparedInput,
    subsequence_bits: u32,
) -> Result<Vec<SubsequenceState>> {
    use crate::jpeg_decoder::build_mcu_schedule;

    if prep.bitstream.length_bits == 0 {
        return Ok(Vec::new());
    }
    if subsequence_bits == 0 {
        return Err(BackendError::msg(
            "dispatch_jpeg_phase1_intra_sync: subsequence_bits must be > 0",
        ));
    }

    let (mcu_sched_host, blocks_per_mcu) =
        build_mcu_schedule(prep).map_err(|e| BackendError::msg(format!("{e}")))?;

    let ac_refs = prep.ac_codebooks_for_dispatch();
    let dc_refs = prep.dc_codebooks_for_dispatch();
    let ac_flat = build_gpu_codebook_refs(&ac_refs);
    let dc_flat = build_gpu_codebook_refs(&dc_refs);

    let num_components = u32::try_from(prep.components.len())
        .map_err(|_| BackendError::msg("components.len() does not fit in u32"))?;
    let length_bits = prep.bitstream.length_bits;
    let num_subsequences = length_bits.div_ceil(subsequence_bits);

    let bitstream_words = (length_bits.div_ceil(32) as usize) + 1;
    let bitstream_bytes = bitstream_words * 4;
    let ac_codebook_bytes = ac_flat.len() * 4;
    let dc_codebook_bytes = dc_flat.len() * 4;
    let mcu_sched_bytes = mcu_sched_host.len() * 4;
    let s_info_bytes = (num_subsequences as usize) * std::mem::size_of::<SubsequenceState>();

    let bitstream_buf = DeviceBufferGuard::alloc_zeroed(backend, bitstream_bytes)?;
    let codebook_buf = DeviceBufferGuard::alloc(backend, ac_codebook_bytes)?;
    let dc_codebook_buf = DeviceBufferGuard::alloc(backend, dc_codebook_bytes)?;
    let mcu_sched_buf = DeviceBufferGuard::alloc(backend, mcu_sched_bytes)?;
    let s_info_buf = DeviceBufferGuard::alloc(backend, s_info_bytes)?;

    let _up1 = backend.upload_async(
        bitstream_buf.as_ref(),
        bytemuck::cast_slice(&prep.bitstream.words),
    )?;
    let _up2 = backend.upload_async(codebook_buf.as_ref(), bytemuck::cast_slice(&ac_flat))?;
    let _up3 = backend.upload_async(dc_codebook_buf.as_ref(), bytemuck::cast_slice(&dc_flat))?;
    let _up4 = backend.upload_async(
        mcu_sched_buf.as_ref(),
        bytemuck::cast_slice(&mcu_sched_host),
    )?;

    backend.begin_page()?;
    backend.record_huffman(HuffmanParams {
        bitstream: bitstream_buf.as_ref(),
        codebook: codebook_buf.as_ref(),
        s_info: s_info_buf.as_ref(),
        sync_flags: None,
        offsets: None,
        symbols_out: None,
        decode_status: None,
        dc_codebook: Some(dc_codebook_buf.as_ref()),
        mcu_schedule: Some(mcu_sched_buf.as_ref()),
        length_bits,
        subsequence_bits,
        num_components,
        total_symbols: 0,
        blocks_per_mcu,
        phase: HuffmanPhase::JpegPhase1IntraSync,
    })?;
    let fence = backend.submit_page()?;
    backend.wait_page(fence)?;

    let mut out =
        vec![<SubsequenceState as bytemuck::Zeroable>::zeroed(); num_subsequences as usize];
    let dst = bytemuck::cast_slice_mut::<SubsequenceState, u8>(&mut out);
    let handle = backend.download_async(s_info_buf.as_ref(), dst)?;
    backend.wait_download(handle)?;

    backend.free_device(bitstream_buf.take());
    backend.free_device(codebook_buf.take());
    backend.free_device(dc_codebook_buf.take());
    backend.free_device(mcu_sched_buf.take());
    backend.free_device(s_info_buf.take());

    Ok(out)
}

/// One-shot JPEG Phase 1 + Phase 2 dispatch using real JPEG framing.
///
/// Runs JPEG Phase 1, then loops Phase 2 until all subseqs are synced
/// or the retry bound is exhausted. Returns the final `s_info` and the
/// `Phase2Outcome`. Mirrors `dispatch_phase1_then_phase2` but uses the
/// JPEG-framed kernel variants and JPEG-specific buffers
/// (`dc_codebook`, `mcu_schedule`).
///
/// Test/oracle path only — production callers should drive the trait
/// directly to keep buffers alive across pages.
///
/// # Errors
/// Returns `BackendError` if any alloc/upload/dispatch/download
/// fails. Bound-exceeded is reported via the outcome, not as an error.
#[cfg(feature = "gpu-validation")]
fn dispatch_jpeg_phase1_then_phase2<B: GpuBackend>(
    backend: &B,
    prep: &crate::jpeg_decoder::JpegPreparedInput,
    subsequence_bits: u32,
) -> Result<(Vec<SubsequenceState>, Phase2Outcome)> {
    use crate::jpeg_decoder::build_mcu_schedule;

    if prep.bitstream.length_bits == 0 {
        return Ok((Vec::new(), Phase2Outcome::Converged { iterations: 0 }));
    }
    if subsequence_bits == 0 {
        return Err(BackendError::msg(
            "dispatch_jpeg_phase1_then_phase2: subsequence_bits must be > 0",
        ));
    }

    let (mcu_sched_host, blocks_per_mcu) =
        build_mcu_schedule(prep).map_err(|e| BackendError::msg(format!("{e}")))?;

    let ac_refs = prep.ac_codebooks_for_dispatch();
    let dc_refs = prep.dc_codebooks_for_dispatch();
    let ac_flat = build_gpu_codebook_refs(&ac_refs);
    let dc_flat = build_gpu_codebook_refs(&dc_refs);

    let num_components = u32::try_from(prep.components.len())
        .map_err(|_| BackendError::msg("components.len() does not fit in u32"))?;
    let length_bits = prep.bitstream.length_bits;
    let num_subsequences = length_bits.div_ceil(subsequence_bits);

    let bitstream_words = (length_bits.div_ceil(32) as usize) + 1;
    let bitstream_bytes = bitstream_words * 4;
    let ac_codebook_bytes = ac_flat.len() * 4;
    let dc_codebook_bytes = dc_flat.len() * 4;
    let mcu_sched_bytes = mcu_sched_host.len() * 4;
    let s_info_bytes = (num_subsequences as usize) * std::mem::size_of::<SubsequenceState>();
    let flags_bytes = (num_subsequences as usize) * 4;

    let bitstream_buf = DeviceBufferGuard::alloc_zeroed(backend, bitstream_bytes)?;
    let codebook_buf = DeviceBufferGuard::alloc(backend, ac_codebook_bytes)?;
    let dc_codebook_buf = DeviceBufferGuard::alloc(backend, dc_codebook_bytes)?;
    let mcu_sched_buf = DeviceBufferGuard::alloc(backend, mcu_sched_bytes)?;
    let s_info_buf = DeviceBufferGuard::alloc(backend, s_info_bytes)?;
    let sync_flags_buf = DeviceBufferGuard::alloc(backend, flags_bytes)?;

    let _up1 = backend.upload_async(
        bitstream_buf.as_ref(),
        bytemuck::cast_slice(&prep.bitstream.words),
    )?;
    let _up2 = backend.upload_async(codebook_buf.as_ref(), bytemuck::cast_slice(&ac_flat))?;
    let _up3 = backend.upload_async(dc_codebook_buf.as_ref(), bytemuck::cast_slice(&dc_flat))?;
    let _up4 = backend.upload_async(
        mcu_sched_buf.as_ref(),
        bytemuck::cast_slice(&mcu_sched_host),
    )?;

    // Phase 1.
    backend.begin_page()?;
    backend.record_huffman(HuffmanParams {
        bitstream: bitstream_buf.as_ref(),
        codebook: codebook_buf.as_ref(),
        s_info: s_info_buf.as_ref(),
        sync_flags: None,
        offsets: None,
        symbols_out: None,
        decode_status: None,
        dc_codebook: Some(dc_codebook_buf.as_ref()),
        mcu_schedule: Some(mcu_sched_buf.as_ref()),
        length_bits,
        subsequence_bits,
        num_components,
        total_symbols: 0,
        blocks_per_mcu,
        phase: HuffmanPhase::JpegPhase1IntraSync,
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
            bitstream: bitstream_buf.as_ref(),
            codebook: codebook_buf.as_ref(),
            s_info: s_info_buf.as_ref(),
            sync_flags: Some(sync_flags_buf.as_ref()),
            offsets: None,
            symbols_out: None,
            decode_status: None,
            dc_codebook: Some(dc_codebook_buf.as_ref()),
            mcu_schedule: Some(mcu_sched_buf.as_ref()),
            length_bits,
            subsequence_bits,
            num_components,
            total_symbols: 0,
            blocks_per_mcu,
            phase: HuffmanPhase::JpegPhase2InterSync,
        })?;
        let fence = backend.submit_page()?;
        backend.wait_page(fence)?;

        let dst = bytemuck::cast_slice_mut::<u32, u8>(&mut flags_host);
        let handle = backend.download_async(sync_flags_buf.as_ref(), dst)?;
        backend.wait_download(handle)?;

        if flags_host.iter().all(|&f| f == 1) {
            outcome = Phase2Outcome::Converged { iterations: iter };
            break;
        }
    }

    let mut s_info_out =
        vec![<SubsequenceState as bytemuck::Zeroable>::zeroed(); num_subsequences as usize];
    let dst = bytemuck::cast_slice_mut::<SubsequenceState, u8>(&mut s_info_out);
    let handle = backend.download_async(s_info_buf.as_ref(), dst)?;
    backend.wait_download(handle)?;

    backend.free_device(bitstream_buf.take());
    backend.free_device(codebook_buf.take());
    backend.free_device(dc_codebook_buf.take());
    backend.free_device(mcu_sched_buf.take());
    backend.free_device(s_info_buf.take());
    backend.free_device(sync_flags_buf.take());

    Ok((s_info_out, outcome))
}

/// Run all four JPEG-framed phases and return the final decoded symbol
/// stream as a flat `Vec<u32>`.
///
/// Mirrors `dispatch_phase1_through_phase4` exactly but uses the
/// JPEG-framed kernels: `JpegPhase1IntraSync`, `JpegPhase2InterSync`,
/// CPU-side exclusive-scan for Phase 3, and `JpegPhase4Redecode`.
///
/// # Errors
/// Returns `BackendError` if any allocation, upload, kernel dispatch,
/// or download fails, or if Phase 2 does not converge within the retry
/// bound.
#[cfg(feature = "gpu-validation")]
#[expect(
    clippy::too_many_lines,
    reason = "linear 4-phase procedural script; same rationale as dispatch_phase1_through_phase4"
)]
fn dispatch_jpeg_phase1_through_phase4<B: GpuBackend>(
    backend: &B,
    prep: &crate::jpeg_decoder::JpegPreparedInput,
    subsequence_bits: u32,
) -> Result<Vec<u32>> {
    use crate::jpeg_decoder::build_mcu_schedule;

    if prep.bitstream.length_bits == 0 {
        return Ok(Vec::new());
    }
    if subsequence_bits == 0 {
        return Err(BackendError::msg(
            "dispatch_jpeg_phase1_through_phase4: subsequence_bits must be > 0",
        ));
    }

    let (mcu_sched_host, blocks_per_mcu) =
        build_mcu_schedule(prep).map_err(|e| BackendError::msg(format!("{e}")))?;

    let ac_refs = prep.ac_codebooks_for_dispatch();
    let dc_refs = prep.dc_codebooks_for_dispatch();
    let ac_flat = build_gpu_codebook_refs(&ac_refs);
    let dc_flat = build_gpu_codebook_refs(&dc_refs);

    let num_components = u32::try_from(prep.components.len())
        .map_err(|_| BackendError::msg("components.len() does not fit in u32"))?;
    let length_bits = prep.bitstream.length_bits;
    let num_subsequences = length_bits.div_ceil(subsequence_bits);

    let bitstream_words = (length_bits.div_ceil(32) as usize) + 1;
    let bitstream_bytes = bitstream_words * 4;
    let ac_codebook_bytes = ac_flat.len() * 4;
    let dc_codebook_bytes = dc_flat.len() * 4;
    let mcu_sched_bytes = mcu_sched_host.len() * 4;
    let s_info_bytes = (num_subsequences as usize) * std::mem::size_of::<SubsequenceState>();
    let flags_bytes = (num_subsequences as usize) * 4;
    let offsets_bytes = flags_bytes;

    let bitstream_buf = DeviceBufferGuard::alloc_zeroed(backend, bitstream_bytes)?;
    let codebook_buf = DeviceBufferGuard::alloc(backend, ac_codebook_bytes)?;
    let dc_codebook_buf = DeviceBufferGuard::alloc(backend, dc_codebook_bytes)?;
    let mcu_sched_buf = DeviceBufferGuard::alloc(backend, mcu_sched_bytes)?;
    let s_info_buf = DeviceBufferGuard::alloc(backend, s_info_bytes)?;
    let sync_flags_buf = DeviceBufferGuard::alloc(backend, flags_bytes)?;
    let offsets_buf = DeviceBufferGuard::alloc(backend, offsets_bytes)?;

    let _up1 = backend.upload_async(
        bitstream_buf.as_ref(),
        bytemuck::cast_slice(&prep.bitstream.words),
    )?;
    let _up2 = backend.upload_async(codebook_buf.as_ref(), bytemuck::cast_slice(&ac_flat))?;
    let _up3 = backend.upload_async(dc_codebook_buf.as_ref(), bytemuck::cast_slice(&dc_flat))?;
    let _up4 = backend.upload_async(
        mcu_sched_buf.as_ref(),
        bytemuck::cast_slice(&mcu_sched_host),
    )?;

    // Phase 1.
    backend.begin_page()?;
    backend.record_huffman(HuffmanParams {
        bitstream: bitstream_buf.as_ref(),
        codebook: codebook_buf.as_ref(),
        s_info: s_info_buf.as_ref(),
        sync_flags: None,
        offsets: None,
        symbols_out: None,
        decode_status: None,
        dc_codebook: Some(dc_codebook_buf.as_ref()),
        mcu_schedule: Some(mcu_sched_buf.as_ref()),
        length_bits,
        subsequence_bits,
        num_components,
        total_symbols: 0,
        blocks_per_mcu,
        phase: HuffmanPhase::JpegPhase1IntraSync,
    })?;
    let fence = backend.submit_page()?;
    backend.wait_page(fence)?;

    // Phase 2 — bounded retry loop.
    let bound = phase2_retry_bound(num_subsequences as usize);
    let mut flags_host = vec![0u32; num_subsequences as usize];
    let mut converged = false;
    for _iter in 0..=bound {
        backend.begin_page()?;
        backend.record_huffman(HuffmanParams {
            bitstream: bitstream_buf.as_ref(),
            codebook: codebook_buf.as_ref(),
            s_info: s_info_buf.as_ref(),
            sync_flags: Some(sync_flags_buf.as_ref()),
            offsets: None,
            symbols_out: None,
            decode_status: None,
            dc_codebook: Some(dc_codebook_buf.as_ref()),
            mcu_schedule: Some(mcu_sched_buf.as_ref()),
            length_bits,
            subsequence_bits,
            num_components,
            total_symbols: 0,
            blocks_per_mcu,
            phase: HuffmanPhase::JpegPhase2InterSync,
        })?;
        let fence = backend.submit_page()?;
        backend.wait_page(fence)?;

        let dst = bytemuck::cast_slice_mut::<u32, u8>(&mut flags_host);
        let handle = backend.download_async(sync_flags_buf.as_ref(), dst)?;
        backend.wait_download(handle)?;

        if flags_host.iter().all(|&f| f == 1) {
            converged = true;
            break;
        }
    }
    if !converged {
        return Err(BackendError::msg(
            "dispatch_jpeg_phase1_through_phase4: Phase 2 did not converge within retry bound",
        ));
    }

    // Download s_info for CPU-side Phase 3 (exclusive scan).
    let mut s_info_host =
        vec![<SubsequenceState as bytemuck::Zeroable>::zeroed(); num_subsequences as usize];
    let dst = bytemuck::cast_slice_mut::<SubsequenceState, u8>(&mut s_info_host);
    let handle = backend.download_async(s_info_buf.as_ref(), dst)?;
    backend.wait_download(handle)?;

    let counts: Vec<u32> = s_info_host.iter().map(|s| s.n).collect();
    let offsets_host = crate::jpeg_decoder::scan::test_helpers::cpu_exclusive_scan(&counts);
    let total_symbols = offsets_host
        .last()
        .copied()
        .unwrap_or(0)
        .checked_add(s_info_host.last().map_or(0, |s| s.n))
        .ok_or_else(|| {
            BackendError::msg(
                "dispatch_jpeg_phase1_through_phase4: total symbol count overflows u32",
            )
        })?;

    if total_symbols == 0 {
        return Ok(Vec::new());
    }

    let _up5 = backend.upload_async(offsets_buf.as_ref(), bytemuck::cast_slice(&offsets_host))?;

    let symbols_bytes = (total_symbols as usize) * 4;
    let symbols_buf = DeviceBufferGuard::alloc_zeroed(backend, symbols_bytes)?;
    let decode_status_buf = DeviceBufferGuard::alloc_zeroed(backend, flags_bytes)?;

    // Phase 4.
    backend.begin_page()?;
    backend.record_huffman(HuffmanParams {
        bitstream: bitstream_buf.as_ref(),
        codebook: codebook_buf.as_ref(),
        s_info: s_info_buf.as_ref(),
        sync_flags: None,
        offsets: Some(offsets_buf.as_ref()),
        symbols_out: Some(symbols_buf.as_ref()),
        decode_status: Some(decode_status_buf.as_ref()),
        dc_codebook: Some(dc_codebook_buf.as_ref()),
        mcu_schedule: Some(mcu_sched_buf.as_ref()),
        length_bits,
        subsequence_bits,
        num_components,
        total_symbols,
        blocks_per_mcu,
        phase: HuffmanPhase::JpegPhase4Redecode,
    })?;
    let fence = backend.submit_page()?;
    backend.wait_page(fence)?;

    // Inspect per-subseq decode status before downloading symbols.
    let mut status_host = vec![0u32; num_subsequences as usize];
    let dst = bytemuck::cast_slice_mut::<u32, u8>(&mut status_host);
    let handle = backend.download_async(decode_status_buf.as_ref(), dst)?;
    backend.wait_download(handle)?;
    if let Some((failing_idx, &raw)) = status_host.iter().enumerate().find(|&(_, &s)| s != 0) {
        let kind = crate::backend::params::Phase4FailureKind::from_u32(raw);
        return Err(BackendError::msg(format!(
            "dispatch_jpeg_phase1_through_phase4: Phase 4 decode failed on subseq {failing_idx}: {} (status code {raw})",
            kind.label(),
        )));
    }

    let mut symbols_host = vec![0u32; total_symbols as usize];
    let dst = bytemuck::cast_slice_mut::<u32, u8>(&mut symbols_host);
    let handle = backend.download_async(symbols_buf.as_ref(), dst)?;
    backend.wait_download(handle)?;

    backend.free_device(bitstream_buf.take());
    backend.free_device(codebook_buf.take());
    backend.free_device(dc_codebook_buf.take());
    backend.free_device(mcu_sched_buf.take());
    backend.free_device(s_info_buf.take());
    backend.free_device(sync_flags_buf.take());
    backend.free_device(offsets_buf.take());
    backend.free_device(symbols_buf.take());
    backend.free_device(decode_status_buf.take());

    Ok(symbols_host)
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

    // Guards free their buffers on Drop, so any `?` between here
    // and the explicit `take() + free_device` at the bottom doesn't
    // leak. alloc_zeroed gives the trailing peek16-headroom word
    // for free — we then upload only the stream's actual words,
    // leaving the tail zero.
    let bitstream_buf = DeviceBufferGuard::alloc_zeroed(backend, bitstream_bytes)?;
    let codebook_buf = DeviceBufferGuard::alloc(backend, codebook_bytes)?;
    let s_info_buf = DeviceBufferGuard::alloc(backend, s_info_bytes)?;

    // Both uploads queue on the transfer queue; `submit_page` inserts
    // the cross-queue barrier before the kernel reads the buffers, so
    // explicit wait_transfer between uploads is unnecessary.
    let _up1 = backend.upload_async(bitstream_buf.as_ref(), bytemuck::cast_slice(&stream.words))?;
    let _up2 = backend.upload_async(codebook_buf.as_ref(), bytemuck::cast_slice(&codebook_flat))?;

    backend.begin_page()?;
    backend.record_huffman(HuffmanParams {
        bitstream: bitstream_buf.as_ref(),
        codebook: codebook_buf.as_ref(),
        s_info: s_info_buf.as_ref(),
        sync_flags: None,
        offsets: None,
        symbols_out: None,
        decode_status: None,
        dc_codebook: None,
        mcu_schedule: None,
        length_bits,
        subsequence_bits,
        num_components,
        total_symbols: 0,
        blocks_per_mcu: 0,
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
    let handle = backend.download_async(s_info_buf.as_ref(), dst)?;
    backend.wait_download(handle)?;

    backend.free_device(bitstream_buf.take());
    backend.free_device(codebook_buf.take());
    backend.free_device(s_info_buf.take());

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

    // Drop-frees guards — see `dispatch_util` module doc-comment.
    let bitstream_buf = DeviceBufferGuard::alloc_zeroed(backend, bitstream_bytes)?;
    let codebook_buf = DeviceBufferGuard::alloc(backend, codebook_bytes)?;
    let s_info_buf = DeviceBufferGuard::alloc(backend, s_info_bytes)?;
    let sync_flags_buf = DeviceBufferGuard::alloc(backend, flags_bytes)?;

    let _up1 = backend.upload_async(bitstream_buf.as_ref(), bytemuck::cast_slice(&stream.words))?;
    let _up2 = backend.upload_async(codebook_buf.as_ref(), bytemuck::cast_slice(&codebook_flat))?;

    // Phase 1 — one pass.
    backend.begin_page()?;
    backend.record_huffman(HuffmanParams {
        bitstream: bitstream_buf.as_ref(),
        codebook: codebook_buf.as_ref(),
        s_info: s_info_buf.as_ref(),
        sync_flags: None,
        offsets: None,
        symbols_out: None,
        decode_status: None,
        dc_codebook: None,
        mcu_schedule: None,
        length_bits,
        subsequence_bits,
        num_components,
        total_symbols: 0,
        blocks_per_mcu: 0,
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
            bitstream: bitstream_buf.as_ref(),
            codebook: codebook_buf.as_ref(),
            s_info: s_info_buf.as_ref(),
            sync_flags: Some(sync_flags_buf.as_ref()),
            offsets: None,
            symbols_out: None,
            decode_status: None,
            dc_codebook: None,
            mcu_schedule: None,
            length_bits,
            subsequence_bits,
            num_components,
            total_symbols: 0,
            blocks_per_mcu: 0,
            phase: HuffmanPhase::Phase2InterSync,
        })?;
        let fence = backend.submit_page()?;
        backend.wait_page(fence)?;

        let dst = bytemuck::cast_slice_mut::<u32, u8>(&mut flags_host);
        let handle = backend.download_async(sync_flags_buf.as_ref(), dst)?;
        backend.wait_download(handle)?;

        if flags_host.iter().all(|&f| f == 1) {
            outcome = Phase2Outcome::Converged { iterations: iter };
            break;
        }
    }

    let mut s_info_out =
        vec![<SubsequenceState as bytemuck::Zeroable>::zeroed(); num_subsequences as usize];
    let dst = bytemuck::cast_slice_mut::<SubsequenceState, u8>(&mut s_info_out);
    let handle = backend.download_async(s_info_buf.as_ref(), dst)?;
    backend.wait_download(handle)?;

    backend.free_device(bitstream_buf.take());
    backend.free_device(codebook_buf.take());
    backend.free_device(s_info_buf.take());
    backend.free_device(sync_flags_buf.take());

    Ok((s_info_out, outcome))
}

/// One-shot dispatch of all four phases: 1 → 2 → 3 → 4.
///
/// Encodes the synthetic test path end-to-end. Returns the final
/// decoded symbol stream (u32 per symbol; the kernel writes a u8
/// value but the buffer is u32-strided to match the Slang/CUDA
/// signature).
///
/// Algorithm:
/// - Phase 1: per-subseq intra-walk → `s_info`.
/// - Phase 2: bounded sync loop → fixes up `s_info[i].p` and `.n`.
/// - Phase 3: exclusive scan over `s_info[i].n` → `offsets`. Done
///   host-side here (the values are tiny — one u32 per subseq);
///   production callers should use `dispatch_phase3_offsets` to
///   keep the scan on device.
/// - Phase 4: per-subseq re-decode from `[prev.p, me.p)`, writing
///   each symbol to `symbols_out[offsets[i] + local_n]`.
///
/// Test/oracle path only — production callers should drive the
/// trait directly to keep buffers alive across pages.
///
/// # Errors
/// Returns `BackendError` if any alloc/upload/dispatch/download
/// fails, or if Phase 2 fails to converge within the retry bound.
#[cfg(feature = "gpu-validation")]
#[expect(
    clippy::too_many_lines,
    reason = "linear 4-phase procedural script; splitting would obscure the phase-by-phase flow and force the per-phase param structs through tiny helper functions"
)]
fn dispatch_phase1_through_phase4<B: GpuBackend>(
    backend: &B,
    stream: &PackedBitstream,
    codebooks: &[CanonicalCodebook],
    subsequence_bits: u32,
) -> Result<Vec<u32>> {
    if stream.length_bits == 0 {
        return Ok(Vec::new());
    }
    if subsequence_bits == 0 {
        return Err(BackendError::msg(
            "dispatch_phase1_through_phase4: subsequence_bits must be > 0",
        ));
    }
    if codebooks.is_empty() {
        return Err(BackendError::msg(
            "dispatch_phase1_through_phase4: at least one codebook required",
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
    let offsets_bytes = flags_bytes;

    // Drop-frees guards — any `?` or early `return Err` between
    // here and the explicit `take() + free_device` at the bottom
    // releases the buffers automatically. The early `return Ok`
    // on `total_symbols == 0` works the same way.
    let bitstream_buf = DeviceBufferGuard::alloc_zeroed(backend, bitstream_bytes)?;
    let codebook_buf = DeviceBufferGuard::alloc(backend, codebook_bytes)?;
    let s_info_buf = DeviceBufferGuard::alloc(backend, s_info_bytes)?;
    let sync_flags_buf = DeviceBufferGuard::alloc(backend, flags_bytes)?;
    let offsets_buf = DeviceBufferGuard::alloc(backend, offsets_bytes)?;

    let _up1 = backend.upload_async(bitstream_buf.as_ref(), bytemuck::cast_slice(&stream.words))?;
    let _up2 = backend.upload_async(codebook_buf.as_ref(), bytemuck::cast_slice(&codebook_flat))?;

    // Phase 1.
    backend.begin_page()?;
    backend.record_huffman(HuffmanParams {
        bitstream: bitstream_buf.as_ref(),
        codebook: codebook_buf.as_ref(),
        s_info: s_info_buf.as_ref(),
        sync_flags: None,
        offsets: None,
        symbols_out: None,
        decode_status: None,
        dc_codebook: None,
        mcu_schedule: None,
        length_bits,
        subsequence_bits,
        num_components,
        total_symbols: 0,
        blocks_per_mcu: 0,
        phase: HuffmanPhase::Phase1IntraSync,
    })?;
    let fence = backend.submit_page()?;
    backend.wait_page(fence)?;

    // Phase 2.
    let bound = phase2_retry_bound(num_subsequences as usize);
    let mut flags_host = vec![0u32; num_subsequences as usize];
    let mut converged = false;
    for _iter in 0..=bound {
        backend.begin_page()?;
        backend.record_huffman(HuffmanParams {
            bitstream: bitstream_buf.as_ref(),
            codebook: codebook_buf.as_ref(),
            s_info: s_info_buf.as_ref(),
            sync_flags: Some(sync_flags_buf.as_ref()),
            offsets: None,
            symbols_out: None,
            decode_status: None,
            dc_codebook: None,
            mcu_schedule: None,
            length_bits,
            subsequence_bits,
            num_components,
            total_symbols: 0,
            blocks_per_mcu: 0,
            phase: HuffmanPhase::Phase2InterSync,
        })?;
        let fence = backend.submit_page()?;
        backend.wait_page(fence)?;

        let dst = bytemuck::cast_slice_mut::<u32, u8>(&mut flags_host);
        let handle = backend.download_async(sync_flags_buf.as_ref(), dst)?;
        backend.wait_download(handle)?;

        if flags_host.iter().all(|&f| f == 1) {
            converged = true;
            break;
        }
    }
    if !converged {
        return Err(BackendError::msg(
            "dispatch_phase1_through_phase4: Phase 2 did not converge within retry bound",
        ));
    }

    // Download s_info to compute Phase 3 offsets host-side and
    // determine the total symbol count for sizing symbols_out.
    let mut s_info_host =
        vec![<SubsequenceState as bytemuck::Zeroable>::zeroed(); num_subsequences as usize];
    let dst = bytemuck::cast_slice_mut::<SubsequenceState, u8>(&mut s_info_host);
    let handle = backend.download_async(s_info_buf.as_ref(), dst)?;
    backend.wait_download(handle)?;

    // Exclusive scan + total over s_info[*].n. checked_add surfaces
    // u32 overflow (silent wrapping would let Phase 4 write past the
    // allocated symbols_out). Shared with the scan parity tests.
    let counts: Vec<u32> = s_info_host.iter().map(|s| s.n).collect();
    let offsets_host = crate::jpeg_decoder::scan::test_helpers::cpu_exclusive_scan(&counts);
    let total_symbols = offsets_host
        .last()
        .copied()
        .unwrap_or(0)
        .checked_add(s_info_host.last().map_or(0, |s| s.n))
        .ok_or_else(|| {
            BackendError::msg("dispatch_phase1_through_phase4: total symbol count overflows u32")
        })?;

    if total_symbols == 0 {
        // Well-formed stream produced no decodable symbols. Skip
        // Phase 4 entirely (kernel + alloc would both reject 0-size).
        // Drop on the five guards frees their buffers.
        return Ok(Vec::new());
    }

    let _up3 = backend.upload_async(offsets_buf.as_ref(), bytemuck::cast_slice(&offsets_host))?;

    let symbols_bytes = (total_symbols as usize) * 4;
    let symbols_buf = DeviceBufferGuard::alloc_zeroed(backend, symbols_bytes)?;
    // decode_status: one u32 per subseq. Zero-init = Phase4FailureKind::Ok
    // so any subseq the kernel doesn't reach (degenerate dispatch grid)
    // stays Ok and the post-Phase-4 inspection sees clean data.
    let decode_status_buf = DeviceBufferGuard::alloc_zeroed(backend, flags_bytes)?;

    // Phase 4.
    backend.begin_page()?;
    backend.record_huffman(HuffmanParams {
        bitstream: bitstream_buf.as_ref(),
        codebook: codebook_buf.as_ref(),
        s_info: s_info_buf.as_ref(),
        sync_flags: None,
        offsets: Some(offsets_buf.as_ref()),
        symbols_out: Some(symbols_buf.as_ref()),
        decode_status: Some(decode_status_buf.as_ref()),
        dc_codebook: None,
        mcu_schedule: None,
        length_bits,
        subsequence_bits,
        num_components,
        total_symbols,
        blocks_per_mcu: 0,
        phase: HuffmanPhase::Phase4Redecode,
    })?;
    let fence = backend.submit_page()?;
    backend.wait_page(fence)?;

    // Inspect per-subseq decode status before downloading symbols.
    // Any non-Ok subseq surfaces as a typed error — adversarial inputs
    // that would have silently produced shorter symbol streams now
    // fail loudly.
    let mut status_host = vec![0u32; num_subsequences as usize];
    let dst = bytemuck::cast_slice_mut::<u32, u8>(&mut status_host);
    let handle = backend.download_async(decode_status_buf.as_ref(), dst)?;
    backend.wait_download(handle)?;
    if let Some((failing_idx, &raw)) = status_host.iter().enumerate().find(|&(_, &s)| s != 0) {
        let kind = crate::backend::params::Phase4FailureKind::from_u32(raw);
        return Err(BackendError::msg(format!(
            "dispatch_phase1_through_phase4: Phase 4 decode failed on subseq {failing_idx}: {} (status code {raw})",
            kind.label(),
        )));
    }

    let mut symbols_host = vec![0u32; total_symbols as usize];
    let dst = bytemuck::cast_slice_mut::<u32, u8>(&mut symbols_host);
    let handle = backend.download_async(symbols_buf.as_ref(), dst)?;
    backend.wait_download(handle)?;

    backend.free_device(bitstream_buf.take());
    backend.free_device(codebook_buf.take());
    backend.free_device(s_info_buf.take());
    backend.free_device(sync_flags_buf.take());
    backend.free_device(offsets_buf.take());
    backend.free_device(symbols_buf.take());
    backend.free_device(decode_status_buf.take());

    Ok(symbols_host)
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
    use crate::jpeg_decoder::build_mcu_schedule;
    use crate::jpeg_decoder::phase1_oracle::{phase1_jpeg_walk_snapshot, phase1_walk_snapshot};
    use crate::jpeg_decoder::prepare_jpeg;
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
            let (cpu, _stop) =
                phase1_walk_snapshot(stream, tables, start_bit, hard_limit, count_to);
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
                let (state, _stop) =
                    phase1_walk_snapshot(&stream, &book, start_bit, hard_limit, count_to);
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

    /// Phase 3 with empty input is a no-op — `Vec::new()` round-trip.
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

    /// End-to-end: encode synthetic symbols → run Phases 1..4 on GPU
    /// → verify the decoded symbol stream matches the input.
    ///
    /// Uniform symbol stream (all 0x00 → code "00"): every subseq's
    /// own region is a clean integer number of length-2 codewords,
    /// no straddling. This is the simplest possible end-to-end test.
    #[test]
    fn end_to_end_uniform_stream_cuda() {
        let Some(b) = try_cuda() else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        let symbols: Vec<u8> = vec![0x00; 1024];
        let stream = book4_stream(&symbols);
        let book = [book4_codebook()];
        let decoded = dispatch_phase1_through_phase4(&b, &stream, &book, 128).expect("e2e");

        assert_eq!(decoded.len(), symbols.len(), "symbol count mismatch");
        let got: Vec<u8> = decoded
            .iter()
            .map(|&v| u8::try_from(v & 0xFF).expect("symbol fits u8"))
            .collect();
        assert_eq!(got, symbols, "decoded stream diverges from input");
    }

    /// End-to-end on **mixed** codeword lengths (lengths 2 + 3).
    /// MVP Phase 2's `(c, z)` sync predicate can need `O(subseq_bits)`
    /// advance steps on this corpus, exceeding the `2 * log2(n)`
    /// retry bound. Documented limitation; the test expects either
    /// Convergence (and validates the round-trip) or the
    /// `SyncBoundExceeded` `BackendError`. Robust sync is a follow-up
    /// in the post-MVP corpus work.
    #[test]
    fn end_to_end_mixed_codeword_lengths_cuda_documented_limitation() {
        let Some(b) = try_cuda() else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        let symbols: Vec<u8> = (0..1000u32).map(|i| (i % 4) as u8).collect();
        let stream = book4_stream(&symbols);
        let book = [book4_codebook()];
        match dispatch_phase1_through_phase4(&b, &stream, &book, 128) {
            Ok(decoded) => {
                let got: Vec<u8> = decoded
                    .iter()
                    .map(|&v| u8::try_from(v & 0xFF).expect("symbol fits u8"))
                    .collect();
                assert_eq!(got, symbols, "decoded stream diverges from input");
            }
            Err(e) => {
                let msg = format!("{e}");
                assert!(
                    msg.contains("Phase 2 did not converge"),
                    "unexpected error: {msg}"
                );
                eprintln!("mixed-stream Phase 2 convergence is a documented MVP limitation: {msg}");
            }
        }
    }

    // ── B2d: JPEG Phase 1 parity tests ────────────────────────────────────

    /// Run `dispatch_jpeg_phase1_intra_sync` on CUDA and verify the
    /// per-subsequence state matches the CPU oracle for every subseq.
    #[test]
    fn jpeg_phase1_cuda_matches_cpu_oracle_grayscale() {
        use crate::jpeg::test_fixtures::GRAY_16X16_JPEG;
        let Some(b) = try_cuda() else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        let prep = prepare_jpeg(GRAY_16X16_JPEG).expect("baseline grayscale must prepare");
        let (mcu_sched, blocks_per_mcu) = build_mcu_schedule(&prep).expect("schedule");

        // Use a small subsequence so we get multiple subseqs on the 16×16 fixture.
        let subsequence_bits = 32u32;
        let gpu_states = dispatch_jpeg_phase1_intra_sync(&b, &prep, subsequence_bits)
            .expect("JPEG Phase 1 dispatch");

        let dc_refs = prep.dc_codebooks_for_dispatch();
        let ac_refs = prep.ac_codebooks_for_dispatch();
        let num_subseq = prep.bitstream.length_bits.div_ceil(subsequence_bits);
        assert_eq!(
            gpu_states.len() as u32,
            num_subseq,
            "GPU state count must equal num_subsequences"
        );

        for seq_idx in 0..num_subseq {
            let start_bit = seq_idx * subsequence_bits;
            let hard_limit = prep
                .bitstream
                .length_bits
                .min(start_bit + 2 * subsequence_bits);
            let count_to = prep.bitstream.length_bits.min(start_bit + subsequence_bits);

            let (cpu, _stop) = phase1_jpeg_walk_snapshot(
                &prep.bitstream,
                &dc_refs,
                &ac_refs,
                &mcu_sched,
                blocks_per_mcu,
                start_bit,
                hard_limit,
                count_to,
            );
            assert_eq!(
                gpu_states[seq_idx as usize], cpu,
                "subsequence {seq_idx}: GPU vs CPU JPEG Phase 1 state mismatch"
            );
        }
    }

    /// Validate that `build_mcu_schedule` integration with the dispatcher
    /// rejects an out-of-range selector before touching any device memory.
    #[test]
    fn jpeg_phase1_rejects_oob_selector_before_dispatch() {
        use crate::jpeg::test_fixtures::GRAY_16X16_JPEG;
        let Some(b) = try_cuda() else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        let mut prep = prepare_jpeg(GRAY_16X16_JPEG).unwrap();
        // Force the DC selector out of range for validation.
        prep.dc_selectors[0] = 99;
        let err = dispatch_jpeg_phase1_intra_sync(&b, &prep, 128)
            .expect_err("out-of-range selector must be rejected");
        let msg = format!("{err}");
        assert!(
            msg.contains("selector") || msg.contains("num_components"),
            "error should mention selector/num_components: {msg}"
        );
    }

    // ── B2e: JPEG Phase 2 tests ────────────────────────────────────────────

    /// JPEG Phase 1+2 on CUDA converges on the grayscale fixture (single
    /// subseq — trivially synced — and small multi-subseq with a stream
    /// whose length_bits is a clean multiple of subsequence_bits).
    #[test]
    fn jpeg_phase2_cuda_converges_on_grayscale() {
        use crate::jpeg::test_fixtures::GRAY_16X16_JPEG;
        let Some(b) = try_cuda() else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        let prep = prepare_jpeg(GRAY_16X16_JPEG).expect("baseline grayscale must prepare");
        // Use a small subsequence_bits to get multiple subseqs; the
        // grayscale fixture is small enough that all should converge
        // on the first Phase 2 pass.
        let (s_info, outcome) =
            dispatch_jpeg_phase1_then_phase2(&b, &prep, 32).expect("dispatch ok");
        let num_subseq = prep.bitstream.length_bits.div_ceil(32);
        assert_eq!(s_info.len() as u32, num_subseq);
        assert!(
            matches!(outcome, Phase2Outcome::Converged { .. }),
            "expected convergence on grayscale, got {outcome:?}"
        );
    }

    /// After JPEG Phase 1+2 on CUDA, every subseq's snapshot position
    /// must have advanced past its own start bit (the basic sync invariant).
    #[test]
    fn jpeg_phase2_cuda_snapshots_past_start_bit() {
        use crate::jpeg::test_fixtures::GRAY_16X16_JPEG;
        let Some(b) = try_cuda() else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        let prep = prepare_jpeg(GRAY_16X16_JPEG).expect("baseline grayscale must prepare");
        let subseq_bits = 32u32;
        let (s_info, _) =
            dispatch_jpeg_phase1_then_phase2(&b, &prep, subseq_bits).expect("dispatch ok");
        for (i, s) in s_info.iter().enumerate() {
            let start_p = i as u32 * subseq_bits;
            assert!(
                s.p >= start_p,
                "subseq {i}: snapshot p={} is before start_bit={start_p}",
                s.p
            );
        }
    }

    // ── B2f: JPEG Phase 4 CUDA tests ─────────────────────────────────────

    /// JPEG Phases 1–4 on CUDA decode GRAY_16X16_JPEG and the resulting
    /// symbol stream matches the CPU oracle (`decode_scan_symbols`).
    #[test]
    fn jpeg_phase4_cuda_matches_oracle_on_grayscale() {
        use crate::jpeg::test_fixtures::GRAY_16X16_JPEG;
        use crate::jpeg_decoder::decode_scan_symbols;
        let Some(b) = try_cuda() else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        let prep = prepare_jpeg(GRAY_16X16_JPEG).expect("baseline grayscale must prepare");
        let gpu_syms =
            dispatch_jpeg_phase1_through_phase4(&b, &prep, 32).expect("CUDA phase4 dispatch");
        let cpu_syms = decode_scan_symbols(&prep).expect("CPU oracle");
        assert_eq!(
            gpu_syms.len(),
            cpu_syms.len(),
            "CUDA vs CPU symbol count mismatch"
        );
        for (i, (&g, &c)) in gpu_syms.iter().zip(cpu_syms.iter()).enumerate() {
            assert_eq!(
                g, c,
                "CUDA vs CPU symbol[{i}] mismatch: GPU={g:#04x} CPU={c:#04x}"
            );
        }
    }

    /// JPEG Phases 1–4 returns the same symbol stream regardless of
    /// subsequence granularity (16 vs 32 vs 64 bits).
    #[test]
    fn jpeg_phase4_cuda_stable_across_subseq_sizes() {
        use crate::jpeg::test_fixtures::GRAY_16X16_JPEG;
        let Some(b) = try_cuda() else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        let prep = prepare_jpeg(GRAY_16X16_JPEG).expect("baseline grayscale must prepare");
        let syms16 =
            dispatch_jpeg_phase1_through_phase4(&b, &prep, 16).expect("subseq=16 dispatch");
        let syms32 =
            dispatch_jpeg_phase1_through_phase4(&b, &prep, 32).expect("subseq=32 dispatch");
        let syms64 =
            dispatch_jpeg_phase1_through_phase4(&b, &prep, 64).expect("subseq=64 dispatch");
        assert_eq!(syms16, syms32, "subseq 16 vs 32 symbol mismatch");
        assert_eq!(syms32, syms64, "subseq 32 vs 64 symbol mismatch");
    }

    /// JPEG Phases 1–4 on CUDA decode a 4:4:4 YCbCr image and the resulting
    /// symbol stream matches the CPU oracle.  This exercises blocks_per_mcu=3
    /// (one block per component per MCU), the MCU schedule rotation for
    /// z_in_block=0 → block_in_mcu advance, and multi-component AC framing.
    #[test]
    fn jpeg_phase4_cuda_matches_oracle_on_ycbcr_444() {
        use crate::jpeg_decoder::decode_scan_symbols;
        static COLOUR_32X32_444: &[u8] =
            include_bytes!("../../../../../../tests/fixtures/jpeg/colour_32x32_444.jpg");
        let Some(b) = try_cuda() else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        let prep = prepare_jpeg(COLOUR_32X32_444).expect("colour 4:4:4 JPEG must prepare");
        assert_eq!(
            prep.components.len(),
            3,
            "fixture must be 3-component YCbCr"
        );
        let gpu_syms =
            dispatch_jpeg_phase1_through_phase4(&b, &prep, 32).expect("CUDA phase4 dispatch");
        let cpu_syms = decode_scan_symbols(&prep).expect("CPU oracle");
        assert_eq!(
            gpu_syms.len(),
            cpu_syms.len(),
            "CUDA vs CPU symbol count mismatch on YCbCr fixture"
        );
        for (i, (&g, &c)) in gpu_syms.iter().zip(cpu_syms.iter()).enumerate() {
            assert_eq!(
                g, c,
                "CUDA vs CPU symbol[{i}] mismatch on YCbCr: GPU={g:#04x} CPU={c:#04x}"
            );
        }
    }

    /// JPEG Phases 1–4 returns a stable symbol stream for the 4:4:4 YCbCr
    /// fixture across subsequence sizes 16, 32, and 64 bits.  Validates that
    /// the MCU schedule rotation (block_in_mcu advance on z_in_block=64) is
    /// consistent regardless of where subsequence boundaries fall.
    #[test]
    fn jpeg_phase4_cuda_ycbcr_stable_across_subseq_sizes() {
        static COLOUR_32X32_444: &[u8] =
            include_bytes!("../../../../../../tests/fixtures/jpeg/colour_32x32_444.jpg");
        let Some(b) = try_cuda() else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        let prep = prepare_jpeg(COLOUR_32X32_444).expect("colour 4:4:4 JPEG must prepare");
        let syms16 =
            dispatch_jpeg_phase1_through_phase4(&b, &prep, 16).expect("subseq=16 dispatch");
        let syms32 =
            dispatch_jpeg_phase1_through_phase4(&b, &prep, 32).expect("subseq=32 dispatch");
        let syms64 =
            dispatch_jpeg_phase1_through_phase4(&b, &prep, 64).expect("subseq=64 dispatch");
        assert_eq!(syms16, syms32, "YCbCr subseq 16 vs 32 symbol mismatch");
        assert_eq!(syms32, syms64, "YCbCr subseq 32 vs 64 symbol mismatch");
    }
}

#[cfg(all(test, feature = "vulkan", feature = "gpu-validation"))]
mod vulkan_tests {
    use super::*;
    use crate::backend::cuda::CudaBackend;
    use crate::backend::vulkan::VulkanBackend;
    use crate::jpeg_decoder::phase1_oracle::phase1_walk_snapshot;
    use crate::jpeg_decoder::prepare_jpeg;
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
            let (cpu, _stop) =
                phase1_walk_snapshot(stream, tables, start_bit, hard_limit, count_to);
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
            let (cpu, _stop) =
                phase1_walk_snapshot(stream, tables, start_bit, hard_limit, count_to);
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

    /// End-to-end Phases 1..4 on both backends; the decoded symbol
    /// streams must be byte-identical and equal to the original
    /// input. Uniform-length corpus stays within Phase 2's MVP retry
    /// bound on both backends.
    #[test]
    fn end_to_end_uniform_cuda_vs_vulkan() {
        let (Some(cuda), Some(vk)) = (try_cuda(), try_vulkan()) else {
            eprintln!("skipping: need both CUDA and Vulkan");
            return;
        };
        let symbols: Vec<u8> = vec![0x00; 1024];
        let stream = book4_stream(&symbols);
        let book = [book4_codebook()];
        let cuda_decoded =
            dispatch_phase1_through_phase4(&cuda, &stream, &book, 128).expect("cuda e2e");
        let vk_decoded =
            dispatch_phase1_through_phase4(&vk, &stream, &book, 128).expect("vulkan e2e");

        assert_eq!(cuda_decoded, vk_decoded, "CUDA and Vulkan diverge");
        let got: Vec<u8> = cuda_decoded
            .iter()
            .map(|&v| u8::try_from(v & 0xFF).expect("symbol fits u8"))
            .collect();
        assert_eq!(got, symbols, "decoded stream diverges from input");
    }

    // ── B2d: JPEG Phase 1 cross-backend parity tests ──────────────────────

    use crate::jpeg_decoder::build_mcu_schedule;
    use crate::jpeg_decoder::phase1_oracle::phase1_jpeg_walk_snapshot;

    /// JPEG Phase 1 CUDA vs Vulkan: both backends must produce byte-
    /// identical per-subseq state on the grayscale fixture.
    #[test]
    fn jpeg_phase1_cuda_vs_vulkan_grayscale() {
        use crate::jpeg::test_fixtures::GRAY_16X16_JPEG;
        let (Some(cuda), Some(vk)) = (try_cuda(), try_vulkan()) else {
            eprintln!("skipping: need both CUDA and Vulkan");
            return;
        };
        let prep = prepare_jpeg(GRAY_16X16_JPEG).expect("baseline grayscale must prepare");
        let subsequence_bits = 32u32;
        let cuda_states = dispatch_jpeg_phase1_intra_sync(&cuda, &prep, subsequence_bits)
            .expect("CUDA JPEG Phase 1");
        let vk_states = dispatch_jpeg_phase1_intra_sync(&vk, &prep, subsequence_bits)
            .expect("Vulkan JPEG Phase 1");

        assert_eq!(
            cuda_states.len(),
            vk_states.len(),
            "state count mismatch between CUDA and Vulkan"
        );
        for (i, (cs, vs)) in cuda_states.iter().zip(vk_states.iter()).enumerate() {
            assert_eq!(
                cs, vs,
                "subsequence {i}: CUDA vs Vulkan JPEG Phase 1 mismatch"
            );
        }
    }

    /// JPEG Phase 1 Vulkan vs CPU oracle: each subseq's GPU state must
    /// match the host-side `phase1_jpeg_walk_snapshot` result.
    #[test]
    fn jpeg_phase1_vulkan_matches_cpu_oracle_grayscale() {
        use crate::jpeg::test_fixtures::GRAY_16X16_JPEG;
        let Some(vk) = try_vulkan() else {
            eprintln!("skipping: no Vulkan device");
            return;
        };
        let prep = prepare_jpeg(GRAY_16X16_JPEG).expect("baseline grayscale must prepare");
        let (mcu_sched, blocks_per_mcu) = build_mcu_schedule(&prep).expect("schedule");
        let subsequence_bits = 32u32;
        let gpu_states = dispatch_jpeg_phase1_intra_sync(&vk, &prep, subsequence_bits)
            .expect("Vulkan JPEG Phase 1");

        let dc_refs = prep.dc_codebooks_for_dispatch();
        let ac_refs = prep.ac_codebooks_for_dispatch();
        let num_subseq = prep.bitstream.length_bits.div_ceil(subsequence_bits);

        for seq_idx in 0..num_subseq {
            let start_bit = seq_idx * subsequence_bits;
            let hard_limit = prep
                .bitstream
                .length_bits
                .min(start_bit + 2 * subsequence_bits);
            let count_to = prep.bitstream.length_bits.min(start_bit + subsequence_bits);
            let (cpu, _) = phase1_jpeg_walk_snapshot(
                &prep.bitstream,
                &dc_refs,
                &ac_refs,
                &mcu_sched,
                blocks_per_mcu,
                start_bit,
                hard_limit,
                count_to,
            );
            assert_eq!(
                gpu_states[seq_idx as usize], cpu,
                "subsequence {seq_idx}: Vulkan vs CPU JPEG Phase 1 mismatch"
            );
        }
    }

    // ── B2e: JPEG Phase 2 cross-backend tests ─────────────────────────────

    /// JPEG Phase 1+2 on Vulkan converges on the grayscale fixture.
    #[test]
    fn jpeg_phase2_vulkan_converges_on_grayscale() {
        use crate::jpeg::test_fixtures::GRAY_16X16_JPEG;
        let Some(vk) = try_vulkan() else {
            eprintln!("skipping: no Vulkan device");
            return;
        };
        let prep = prepare_jpeg(GRAY_16X16_JPEG).expect("baseline grayscale must prepare");
        let (s_info, outcome) =
            dispatch_jpeg_phase1_then_phase2(&vk, &prep, 32).expect("dispatch ok");
        let num_subseq = prep.bitstream.length_bits.div_ceil(32);
        assert_eq!(s_info.len() as u32, num_subseq);
        assert!(
            matches!(outcome, Phase2Outcome::Converged { .. }),
            "expected Vulkan convergence on grayscale, got {outcome:?}"
        );
    }

    /// JPEG Phase 1+2 CUDA and Vulkan produce byte-identical `s_info`
    /// after convergence on the grayscale fixture.
    #[test]
    fn jpeg_phase2_cuda_vs_vulkan_grayscale() {
        use crate::jpeg::test_fixtures::GRAY_16X16_JPEG;
        let (Some(cuda), Some(vk)) = (try_cuda(), try_vulkan()) else {
            eprintln!("skipping: need both CUDA and Vulkan");
            return;
        };
        let prep = prepare_jpeg(GRAY_16X16_JPEG).expect("baseline grayscale must prepare");
        let (cuda_s, cuda_o) =
            dispatch_jpeg_phase1_then_phase2(&cuda, &prep, 32).expect("cuda dispatch");
        let (vk_s, vk_o) =
            dispatch_jpeg_phase1_then_phase2(&vk, &prep, 32).expect("vulkan dispatch");
        // Pin outcome agreement first so a backend divergence in
        // sync detection surfaces as a "outcome mismatch" before the
        // potentially-confusing byte-diff.
        assert_eq!(
            cuda_o, vk_o,
            "Phase 2 outcome mismatch: CUDA={cuda_o:?} Vulkan={vk_o:?}"
        );
        assert_eq!(
            cuda_s, vk_s,
            "Phase 2 s_info mismatch: CUDA and Vulkan disagree"
        );
    }

    // ── B2f: JPEG Phase 4 cross-backend tests ─────────────────────────────

    /// JPEG Phases 1–4 on Vulkan decodes GRAY_16X16_JPEG and the symbol
    /// stream matches the CPU oracle.
    #[test]
    fn jpeg_phase4_vulkan_matches_oracle_on_grayscale() {
        use crate::jpeg::test_fixtures::GRAY_16X16_JPEG;
        use crate::jpeg_decoder::decode_scan_symbols;
        let Some(vk) = try_vulkan() else {
            eprintln!("skipping: no Vulkan device");
            return;
        };
        let prep = prepare_jpeg(GRAY_16X16_JPEG).expect("baseline grayscale must prepare");
        let gpu_syms =
            dispatch_jpeg_phase1_through_phase4(&vk, &prep, 32).expect("Vulkan phase4 dispatch");
        let cpu_syms = decode_scan_symbols(&prep).expect("CPU oracle");
        assert_eq!(
            gpu_syms.len(),
            cpu_syms.len(),
            "Vulkan vs CPU symbol count mismatch"
        );
        for (i, (&g, &c)) in gpu_syms.iter().zip(cpu_syms.iter()).enumerate() {
            assert_eq!(
                g, c,
                "Vulkan vs CPU symbol[{i}] mismatch: GPU={g:#04x} CPU={c:#04x}"
            );
        }
    }

    /// JPEG Phases 1–4 CUDA and Vulkan produce byte-identical symbol
    /// streams on GRAY_16X16_JPEG.
    #[test]
    fn jpeg_phase4_cuda_vs_vulkan_grayscale() {
        use crate::jpeg::test_fixtures::GRAY_16X16_JPEG;
        let (Some(cuda), Some(vk)) = (try_cuda(), try_vulkan()) else {
            eprintln!("skipping: need both CUDA and Vulkan");
            return;
        };
        let prep = prepare_jpeg(GRAY_16X16_JPEG).expect("baseline grayscale must prepare");
        let cuda_syms = dispatch_jpeg_phase1_through_phase4(&cuda, &prep, 32).expect("cuda phase4");
        let vk_syms = dispatch_jpeg_phase1_through_phase4(&vk, &prep, 32).expect("vulkan phase4");
        assert_eq!(
            cuda_syms, vk_syms,
            "JPEG Phase 4 symbol stream: CUDA vs Vulkan mismatch"
        );
    }
}
