//! CPU pre-pass wrapper: lifts [`crate::jpeg::CpuPrepassOutput`] into the
//! shape the on-GPU decoder dispatch will consume.
//!
//! The CPU pre-pass (`crate::jpeg`) parses headers, byte-unstuffs the
//! entropy segment, builds canonical Huffman codebooks for the AC tables
//! the scan actually references, and resolves the DC chain on the CPU.
//! This module turns that output into [`JpegPreparedInput`]: the bitstream
//! packed into the kernel's big-endian 32-bit word layout, AC codebooks
//! kept selector-indexed plus a scan-order selector vector, and the rest
//! of the frame metadata held as-is for the IDCT + colour-convert kernel
//! to consume later.
//!
//! Non-baseline JPEGs (progressive / lossless / hierarchical) are detected
//! by [`crate::jpeg::run_cpu_prepass`] and surfaced here as the typed
//! [`JpegGpuError::Progressive`] so the caller can fall back to a CPU
//! decoder without parsing the error string.

use crate::jpeg::headers::{JpegFrameComponent, JpegScanHeader};
use crate::jpeg::{
    CanonicalCodebook, CpuPrepassError, CpuPrepassOutput, DcValues, JpegHeaderError,
    JpegQuantTable, run_cpu_prepass,
};
use crate::jpeg_decoder::{JpegGpuError, PackedBitstream, pack_be_words};

/// CPU-side input to the on-GPU JPEG decoder.
///
/// Owns everything the GPU pipeline needs so the original JPEG byte slice
/// can be released as soon as `prepare_jpeg` returns. Codebooks are kept
/// selector-indexed (matching the JPEG wire format) rather than
/// per-component, because YCbCr commonly shares one AC table across both
/// chroma channels. Use [`Self::ac_codebooks_for_dispatch`] to materialise
/// the per-component reference slice the parallel-Huffman dispatcher
/// accepts without copying the 128 KB tables.
#[derive(Debug)]
pub struct JpegPreparedInput {
    /// Entropy-coded segment packed for the kernel: big-endian 32-bit
    /// words, exact `length_bits` count.
    pub bitstream: PackedBitstream,
    /// AC Huffman codebooks indexed by table ID 0..=3.  Only slots the
    /// scan actually references are populated.
    pub ac_codebooks: [Option<CanonicalCodebook>; 4],
    /// AC selector for each active scan component, in scan order.  Length
    /// equals `components.len()`.  Each entry is an index into
    /// [`Self::ac_codebooks`].
    pub ac_selectors: Vec<u8>,
    /// Per-component absolute DC values, already resolved on the CPU.
    /// The Huffman kernels only decode AC; DC reconstruction lives here.
    pub dc_values: DcValues,
    /// Quantisation tables, indexed by table ID 0..=3.  Per-component
    /// dereference goes via `components[k].quant_selector`.
    pub quant_tables: [Option<JpegQuantTable>; 4],
    /// Image dimensions in pixels.
    pub width: u16,
    /// Image dimensions in pixels.
    pub height: u16,
    /// Active per-component frame metadata, length = scan component count
    /// (1, 3, or 4).  Carries sampling factors + quant-table selectors.
    pub components: Vec<JpegFrameComponent>,
    /// SOS scan header — preserves DC/AC selector mapping for downstream
    /// component-order checks.
    pub scan: JpegScanHeader,
    /// Restart interval in MCUs (0 = no restart markers in stream).
    pub restart_interval: u16,
}

impl JpegPreparedInput {
    /// Materialise the per-component AC codebook references the
    /// parallel-Huffman dispatcher accepts.  Shared selectors yield
    /// duplicate references; the slice is therefore always length
    /// `components.len()`.
    ///
    /// # Panics
    ///
    /// Panics if an `ac_selectors` entry points at an empty slot in
    /// [`Self::ac_codebooks`].  `prepare_jpeg` enforces population at
    /// construction time, so a trip here would mean the struct was
    /// hand-mutated after `prepare_jpeg` returned.
    #[must_use]
    pub fn ac_codebooks_for_dispatch(&self) -> Vec<&CanonicalCodebook> {
        self.ac_selectors
            .iter()
            .map(|&sel| {
                self.ac_codebooks[usize::from(sel)]
                    .as_ref()
                    .expect("AC selector populated at prepare_jpeg time")
            })
            .collect()
    }
}

/// Run the CPU pre-pass and lift its output into the on-GPU input shape.
///
/// # Errors
///
/// * [`JpegGpuError::Progressive`] — the JPEG is not baseline DCT (SOF2
///   progressive, SOF3 lossless, or other non-baseline variants).
/// * [`JpegGpuError::UnsupportedComponents`] — the SOF declared a component
///   count other than 1 (grayscale) or 3 (YCbCr).  4-component CMYK is
///   parsed by the CPU pre-pass but not yet wired through the GPU path.
/// * [`JpegGpuError::HeaderParse`] — any other pre-pass failure (malformed
///   headers, byte-unstuffing aborted on unexpected marker, Huffman table
///   overflow, DC chain corrupt).  The display of the underlying error is
///   preserved in the message so the caller can log it.
pub fn prepare_jpeg(jpeg_bytes: &[u8]) -> Result<JpegPreparedInput, JpegGpuError> {
    let prep = match run_cpu_prepass(jpeg_bytes) {
        Ok(out) => out,
        Err(CpuPrepassError::Header(JpegHeaderError::NotBaseline)) => {
            return Err(JpegGpuError::Progressive);
        }
        Err(e) => return Err(JpegGpuError::HeaderParse(e.to_string())),
    };

    if prep.components != 1 && prep.components != 3 {
        return Err(JpegGpuError::UnsupportedComponents(prep.components));
    }

    let ac_selectors = collect_ac_selectors(&prep)?;
    let bitstream = pack_unstuffed_bitstream(&prep.unstuffed);
    let components = prep.active_frame_components().to_vec();

    let CpuPrepassOutput {
        width,
        height,
        quant_tables,
        ac_codebooks,
        scan,
        restart_interval,
        dc_values,
        ..
    } = prep;

    Ok(JpegPreparedInput {
        bitstream,
        ac_codebooks,
        ac_selectors,
        dc_values,
        quant_tables,
        width,
        height,
        components,
        scan,
        restart_interval,
    })
}

/// Pack the unstuffed entropy segment into the kernel's BE-32 word layout.
///
/// Baseline JPEG pads the entropy-coded segment with 1-bits to a byte
/// boundary; the kernel stops decoding at per-MCU EOB before reaching the
/// tail, so the entire byte buffer is meaningful as far as the bitstream
/// container is concerned.
fn pack_unstuffed_bitstream(unstuffed: &[u8]) -> PackedBitstream {
    let length_bits = u32::try_from(unstuffed.len())
        .ok()
        .and_then(|n| n.checked_mul(8))
        .expect("unstuffed length in bits must fit in u32 (JPEG max scan size << 2^32)");
    pack_be_words(unstuffed, length_bits)
}

/// Collect the AC selector each scan component references, checking that
/// the prepass actually built a codebook for it.  The returned vector has
/// the same length as the active scan components.
///
/// The CPU pre-pass should never let an unreferenced selector slip into
/// the scan header (`CpuPrepassError::MissingHuffmanTable` is returned
/// earlier), so the typed error here is defensive — it surfaces an
/// invariant violation rather than panicking.
fn collect_ac_selectors(prep: &CpuPrepassOutput) -> Result<Vec<u8>, JpegGpuError> {
    let count = usize::from(prep.scan.component_count);
    let mut out = Vec::with_capacity(count);
    for sc in &prep.scan.components[..count] {
        if prep.ac_codebooks[usize::from(sc.ac_table)].is_none() {
            return Err(JpegGpuError::InvalidHuffmanTables(format!(
                "scan component id={} references AC selector {} but no codebook was built",
                sc.id, sc.ac_table,
            )));
        }
        out.push(sc.ac_table);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jpeg::test_fixtures::{GRAY_16X16_JPEG, PROGRESSIVE_MINIMAL};

    #[test]
    fn prepare_baseline_grayscale_produces_one_codebook() {
        let prep = prepare_jpeg(GRAY_16X16_JPEG).expect("baseline grayscale must prepare");

        assert_eq!(prep.width, 16);
        assert_eq!(prep.height, 16);
        assert_eq!(prep.components.len(), 1);
        // Grayscale scan references one AC selector; the dispatch slice is
        // length-1.
        assert_eq!(prep.ac_selectors.len(), 1);
        assert_eq!(prep.ac_codebooks_for_dispatch().len(), 1);
        // The referenced selector slot is populated; unused selectors are
        // None.  The fixture references selector 0 (confirmed by the
        // CPU pre-pass's own grayscale test).
        assert!(prep.ac_codebooks[usize::from(prep.ac_selectors[0])].is_some());
        // DC values for the single component are populated.
        assert_eq!(prep.dc_values.per_component.len(), 1);
        assert!(!prep.dc_values.per_component[0].is_empty());
        // No restart markers in the fixture.
        assert_eq!(prep.restart_interval, 0);
        // Bitstream is non-empty; length_bits = unstuffed byte count × 8
        // and `words` covers exactly `ceil(length_bits / 32)` words.
        assert!(prep.bitstream.length_bits > 0);
        assert_eq!(prep.bitstream.length_bits % 8, 0);
        assert_eq!(
            prep.bitstream.words.len(),
            (prep.bitstream.length_bits as usize).div_ceil(32),
        );
    }

    #[test]
    fn prepare_rejects_progressive_with_typed_error() {
        let err = prepare_jpeg(PROGRESSIVE_MINIMAL).expect_err("SOF2 must be rejected");
        assert!(
            matches!(err, JpegGpuError::Progressive),
            "expected Progressive, got: {err:?}"
        );
    }

    #[test]
    fn prepare_returns_header_parse_for_garbage() {
        let err = prepare_jpeg(&[0x00, 0x00]).expect_err("garbage must error");
        assert!(
            matches!(err, JpegGpuError::HeaderParse(_)),
            "expected HeaderParse, got: {err:?}"
        );
    }

    #[test]
    fn ac_selectors_match_scan_header_in_order() {
        // Selectors flatten in scan-component order; the dispatch slice
        // dereferences to the same logical codebook as the source array.
        let prep = prepare_jpeg(GRAY_16X16_JPEG).unwrap();
        let raw = run_cpu_prepass(GRAY_16X16_JPEG).unwrap();
        let scan_components = &raw.scan.components[..usize::from(raw.scan.component_count)];
        assert_eq!(prep.ac_selectors.len(), scan_components.len());
        let dispatch = prep.ac_codebooks_for_dispatch();
        for (i, sc) in scan_components.iter().enumerate() {
            assert_eq!(prep.ac_selectors[i], sc.ac_table);
            let expected = raw.ac_codebooks[usize::from(sc.ac_table)].as_ref().unwrap();
            // CanonicalCodebook has no PartialEq; compare packed tables —
            // a deterministic function of the DHT, so byte equality is
            // a sound oracle.
            assert_eq!(
                dispatch[i].table(),
                expected.table(),
                "dispatch slot {i} (id={}, ac_table={}) resolved to wrong codebook",
                sc.id,
                sc.ac_table,
            );
        }
    }
}
