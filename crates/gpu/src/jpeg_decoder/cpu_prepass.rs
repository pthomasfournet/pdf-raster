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
    CanonicalCodebook, CpuPrepassError, CpuPrepassOutput, DcValues, DhtClass, JpegHeaderError,
    JpegQuantTable, run_cpu_prepass,
};
use crate::jpeg_decoder::{JpegGpuError, PackedBitstream, pack_be_words};

/// CPU-side input to the on-GPU JPEG decoder.
///
/// Owns everything the GPU pipeline needs so the original JPEG byte slice
/// can be released as soon as `prepare_jpeg` returns. Codebooks are kept
/// selector-indexed (matching the JPEG wire format) rather than
/// per-component, because YCbCr commonly shares one AC or DC table across
/// both chroma channels. Use [`Self::ac_codebooks_for_dispatch`] (or the
/// DC counterpart) to materialise the per-component reference slice the
/// parallel-Huffman dispatcher accepts without copying the 128 KB tables.
#[derive(Debug)]
pub struct JpegPreparedInput {
    /// Entropy-coded segment packed for the kernel: big-endian 32-bit
    /// words, exact `length_bits` count.
    pub bitstream: PackedBitstream,
    /// DC Huffman codebooks indexed by table ID 0..=3.  Only slots the
    /// scan actually references are populated.  The CPU pre-pass already
    /// resolves the absolute DC values (see [`Self::dc_values`]); the GPU
    /// walker carries DC codebooks only so it can step *past* each
    /// block's DC symbol when consuming the wire bitstream.
    pub dc_codebooks: [Option<CanonicalCodebook>; 4],
    /// DC selector for each active scan component, in scan order.  Length
    /// equals `components.len()`.  Each entry is an index into
    /// [`Self::dc_codebooks`].
    pub dc_selectors: Vec<u8>,
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
        materialise_codebook_slice(&self.ac_codebooks, &self.ac_selectors, DhtClass::Ac)
    }

    /// Materialise the per-component DC codebook references for the
    /// kernel's DC-skip walker.  Same shape as
    /// [`Self::ac_codebooks_for_dispatch`].
    ///
    /// # Panics
    ///
    /// Panics if a `dc_selectors` entry points at an empty slot in
    /// [`Self::dc_codebooks`].  `prepare_jpeg` enforces population at
    /// construction time.
    #[must_use]
    pub fn dc_codebooks_for_dispatch(&self) -> Vec<&CanonicalCodebook> {
        materialise_codebook_slice(&self.dc_codebooks, &self.dc_selectors, DhtClass::Dc)
    }
}

fn materialise_codebook_slice<'a>(
    codebooks: &'a [Option<CanonicalCodebook>; 4],
    selectors: &[u8],
    class: DhtClass,
) -> Vec<&'a CanonicalCodebook> {
    selectors
        .iter()
        .map(|&sel| {
            codebooks[usize::from(sel)].as_ref().unwrap_or_else(|| {
                panic!(
                    "{} selector {sel} populated at prepare_jpeg time",
                    class.name()
                )
            })
        })
        .collect()
}

/// Run the CPU pre-pass and lift its output into the on-GPU input shape.
///
/// # Errors
///
/// * [`JpegGpuError::Progressive`] — the JPEG is not baseline DCT (SOF2
///   progressive, SOF3 lossless, or other non-baseline variants).
/// * [`JpegGpuError::UnsupportedComponents`] — the SOF declared a
///   component count outside `{1, 3}`.  Valid JPEG covers 1 (grayscale),
///   2 (rare two-component), 3 (YCbCr / RGB), and 4 (CMYK); the GPU path
///   only wires 1 and 3 today. The same variant also fires when the SOS
///   header's component count disagrees with the SOF — those would
///   silently produce wrong dispatch slices.
/// * [`JpegGpuError::InvalidHuffmanTables`] — the SOS references a DC
///   or AC selector for which no DHT was loaded, or the DHT itself was
///   structurally invalid (code-space overflow, length mismatch).
/// * [`JpegGpuError::HeaderParse`] — any other pre-pass failure
///   (malformed headers, byte-unstuffing aborted on unexpected marker,
///   DC chain corrupt, unstuffed segment too long to address in u32
///   bits). The display of the underlying error is preserved in the
///   message so the caller can log it.
pub fn prepare_jpeg(jpeg_bytes: &[u8]) -> Result<JpegPreparedInput, JpegGpuError> {
    let prep = match run_cpu_prepass(jpeg_bytes) {
        Ok(out) => out,
        Err(CpuPrepassError::Header(JpegHeaderError::NotBaseline)) => {
            return Err(JpegGpuError::Progressive);
        }
        Err(
            e @ (CpuPrepassError::Codebook { .. } | CpuPrepassError::MissingHuffmanTable { .. }),
        ) => {
            // Codebook / missing-table failures map to the typed Huffman
            // variant so callers don't have to substring-match.
            return Err(JpegGpuError::InvalidHuffmanTables(e.to_string()));
        }
        Err(e) => return Err(JpegGpuError::HeaderParse(e.to_string())),
    };

    if prep.components != 1 && prep.components != 3 {
        return Err(JpegGpuError::UnsupportedComponents(prep.components));
    }
    // SOS may declare a different number of scan components than SOF
    // (typically the parser already aligns them, but non-interleaved
    // baseline JPEGs would surface as a mismatch — the dispatcher would
    // then build wrong-length selector vectors). Refuse rather than
    // produce a garbage prep.
    if prep.scan.component_count != prep.components {
        return Err(JpegGpuError::UnsupportedComponents(
            prep.scan.component_count,
        ));
    }

    let dc_selectors = collect_scan_selectors(&prep, DhtClass::Dc)?;
    let ac_selectors = collect_scan_selectors(&prep, DhtClass::Ac)?;
    let bitstream = pack_unstuffed_bitstream(&prep.unstuffed)?;
    let components = prep.active_frame_components().to_vec();

    let CpuPrepassOutput {
        width,
        height,
        quant_tables,
        dc_codebooks,
        ac_codebooks,
        scan,
        restart_interval,
        dc_values,
        ..
    } = prep;

    Ok(JpegPreparedInput {
        bitstream,
        dc_codebooks,
        dc_selectors,
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
///
/// # Errors
///
/// Returns [`JpegGpuError::HeaderParse`] if `unstuffed.len() * 8` overflows
/// `u32` (would require a > 512 MB entropy segment — practically only a
/// hand-crafted adversarial input). The kernel's `length_bits` field is
/// `u32`; a `usize` overflow there would silently truncate and produce
/// garbage symbols, so we refuse rather than truncate.
fn pack_unstuffed_bitstream(unstuffed: &[u8]) -> Result<PackedBitstream, JpegGpuError> {
    let length_bits = u32::try_from(unstuffed.len())
        .ok()
        .and_then(|n| n.checked_mul(8))
        .ok_or_else(|| {
            JpegGpuError::HeaderParse(format!(
                "unstuffed entropy segment is {} bytes; length-bits would overflow u32",
                unstuffed.len(),
            ))
        })?;
    Ok(pack_be_words(unstuffed, length_bits))
}

/// Build the per-block MCU schedule the JPEG-framed GPU kernels consume.
///
/// The schedule is a flat `Vec<u32>` with one entry per 8×8 block in
/// one MCU (interleaved scan) or exactly one entry (non-interleaved).
/// Each entry packs `(ac_sel << 16) | (dc_sel << 8) | component_idx`.
///
/// # Validation
///
/// Both `dc_sel` and `ac_sel` must be `< num_components`; the kernel
/// uses them as a stride multiplier into the flat codebook buffer
/// (`table_base = sel * 65536`).  An out-of-range selector would
/// read past the buffer end — on CUDA that is undefined behaviour; on
/// Vulkan with `robustBufferAccess2` it returns 0 (a `DECODE_PREFIX_MISS`
/// sentinel), producing wrong symbols rather than crashing.
///
/// The validation is done here in the host wrapper so the kernel can
/// assume its inputs are valid.
///
/// # Errors
///
/// Returns [`JpegGpuError::InvalidHuffmanTables`] if any selector for
/// any block is ≥ `prep.components.len()`.
pub fn build_mcu_schedule(prep: &JpegPreparedInput) -> Result<(Vec<u32>, u32), JpegGpuError> {
    let num_components = prep.components.len();
    let num_comp_u32 = u32::try_from(num_components).expect("components.len() ≤ 4");
    let scan_components = num_components;

    let mut schedule: Vec<u32> = Vec::new();
    for (k, fc) in prep.components.iter().enumerate() {
        let k_u8 = u8::try_from(k).expect("component index < 4");
        let dc_sel = prep.dc_selectors[k];
        let ac_sel = prep.ac_selectors[k];

        if u32::from(dc_sel) >= num_comp_u32 || u32::from(ac_sel) >= num_comp_u32 {
            return Err(JpegGpuError::InvalidHuffmanTables(format!(
                "scan component {} references selectors (dc={dc_sel}, ac={ac_sel}) \
                 ≥ num_components={num_components}",
                k_u8,
            )));
        }

        let bpm: u8 = if scan_components == 1 {
            1
        } else {
            fc.h_sampling
                .checked_mul(fc.v_sampling)
                .expect("upstream BadSamplingFactor caps h, v ≤ 4")
        };
        let entry = (u32::from(ac_sel) << 16) | (u32::from(dc_sel) << 8) | u32::from(k_u8);
        for _ in 0..bpm {
            schedule.push(entry);
        }
    }

    let blocks_per_mcu = u32::try_from(schedule.len()).map_err(|_| {
        JpegGpuError::InvalidHuffmanTables("blocks_per_mcu overflows u32".to_string())
    })?;

    Ok((schedule, blocks_per_mcu))
}

/// Collect the DC or AC selector each scan component references, checking
/// that the prepass actually built a codebook for it.  The returned vector
/// has the same length as the active scan components.
///
/// The CPU pre-pass should never let an unreferenced selector slip into
/// the scan header (`CpuPrepassError::MissingHuffmanTable` is returned
/// earlier), so the typed error here is defensive — it surfaces an
/// invariant violation rather than panicking.
fn collect_scan_selectors(
    prep: &CpuPrepassOutput,
    class: DhtClass,
) -> Result<Vec<u8>, JpegGpuError> {
    let count = usize::from(prep.scan.component_count);
    let mut out = Vec::with_capacity(count);
    for sc in &prep.scan.components[..count] {
        let (selector, codebooks) = match class {
            DhtClass::Dc => (sc.dc_table, &prep.dc_codebooks),
            DhtClass::Ac => (sc.ac_table, &prep.ac_codebooks),
        };
        if codebooks[usize::from(selector)].is_none() {
            return Err(JpegGpuError::InvalidHuffmanTables(format!(
                "scan component id={} references {} selector {} but no codebook was built",
                sc.id,
                class.name(),
                selector,
            )));
        }
        out.push(selector);
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
        // Grayscale scan references one AC + one DC selector; each
        // dispatch slice is length-1.
        assert_eq!(prep.ac_selectors.len(), 1);
        assert_eq!(prep.dc_selectors.len(), 1);
        assert_eq!(prep.ac_codebooks_for_dispatch().len(), 1);
        assert_eq!(prep.dc_codebooks_for_dispatch().len(), 1);
        // Referenced selector slots are populated; unused selectors are
        // None.  The fixture references selector 0 on both classes
        // (confirmed by the CPU pre-pass's own grayscale test).
        assert!(prep.ac_codebooks[usize::from(prep.ac_selectors[0])].is_some());
        assert!(prep.dc_codebooks[usize::from(prep.dc_selectors[0])].is_some());
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
    fn prepare_routes_codebook_error_to_invalid_huffman_tables() {
        // Hand-rolled JPEG with a DHT that overflows the 1-bit code space
        // (DC table 0 declares 3 length-1 codes, only 2 fit). The CPU
        // pre-pass returns CpuPrepassError::Codebook; the wrapper must
        // surface that as the typed InvalidHuffmanTables — callers should
        // not have to substring-match the message.
        let mut data: Vec<u8> = vec![0xFF, 0xD8];
        // DQT, table 0, 64 zeros.
        data.extend_from_slice(&[0xFF, 0xDB, 0x00, 67, 0]);
        data.extend_from_slice(&[0u8; 64]);
        // SOF0, 1 component.
        data.extend_from_slice(&[
            0xFF, 0xC0, 0x00, 11, 8, 0x00, 0x10, 0x00, 0x10, 1, 1, 0x11, 0x00,
        ]);
        // DHT DC table 0: 3 length-1 codes (overflow).
        let mut dht = vec![0xFF, 0xC4, 0x00, 22, 0x00];
        dht.extend_from_slice(&[3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        dht.extend_from_slice(&[0, 1, 2]);
        data.extend_from_slice(&dht);
        // DHT AC table 0: one length-1 code, value 0.
        data.extend_from_slice(&[
            0xFF, 0xC4, 0x00, 20, 0x10, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ]);
        // SOS + one entropy byte + EOI.
        data.extend_from_slice(&[0xFF, 0xDA, 0x00, 8, 1, 1, 0x00, 0, 0x3F, 0x00, 0x80]);
        data.extend_from_slice(&[0xFF, 0xD9]);

        let err = prepare_jpeg(&data).expect_err("invalid DHT must fail");
        assert!(
            matches!(err, JpegGpuError::InvalidHuffmanTables(_)),
            "expected InvalidHuffmanTables, got: {err:?}"
        );
    }

    #[test]
    fn selectors_match_scan_header_in_order_for_both_classes() {
        // Selectors flatten in scan-component order; the dispatch slice
        // dereferences to the same logical codebook as the source array.
        // Verify for both DC and AC.
        let prep = prepare_jpeg(GRAY_16X16_JPEG).unwrap();
        let raw = run_cpu_prepass(GRAY_16X16_JPEG).unwrap();
        let scan_components = &raw.scan.components[..usize::from(raw.scan.component_count)];

        assert_eq!(prep.ac_selectors.len(), scan_components.len());
        assert_eq!(prep.dc_selectors.len(), scan_components.len());
        check_dispatch_slice(
            &prep.ac_selectors,
            &prep.ac_codebooks_for_dispatch(),
            &raw.ac_codebooks,
            scan_components,
            DhtClass::Ac,
        );
        check_dispatch_slice(
            &prep.dc_selectors,
            &prep.dc_codebooks_for_dispatch(),
            &raw.dc_codebooks,
            scan_components,
            DhtClass::Dc,
        );
    }

    fn check_dispatch_slice(
        prep_selectors: &[u8],
        dispatch: &[&CanonicalCodebook],
        source_codebooks: &[Option<CanonicalCodebook>; 4],
        scan_components: &[crate::jpeg::headers::JpegScanComponent],
        class: DhtClass,
    ) {
        for (i, sc) in scan_components.iter().enumerate() {
            let selector = match class {
                DhtClass::Dc => sc.dc_table,
                DhtClass::Ac => sc.ac_table,
            };
            assert_eq!(prep_selectors[i], selector);
            // CanonicalCodebook has no PartialEq; compare packed tables —
            // a deterministic function of the DHT, so byte equality is
            // a sound oracle.
            let expected = source_codebooks[usize::from(selector)].as_ref().unwrap();
            assert_eq!(
                dispatch[i].table(),
                expected.table(),
                "{} dispatch slot {i} (id={}, selector={selector}) wrong",
                class.name(),
                sc.id,
            );
        }
    }

    // ── build_mcu_schedule tests ──────────────────────────────────────────

    #[test]
    fn mcu_schedule_grayscale_one_block() {
        let prep = prepare_jpeg(GRAY_16X16_JPEG).unwrap();
        // Grayscale: 1 component, dc_selector=0, ac_selector=0 — one entry.
        let (sched, bpm) = build_mcu_schedule(&prep).expect("grayscale schedule must build");
        assert_eq!(bpm, 1, "grayscale has 1 block per MCU");
        assert_eq!(sched.len(), 1);
        // Component index = 0, dc_sel = 0, ac_sel = 0.
        // Packed: (0 << 16) | (0 << 8) | 0 = 0.
        assert_eq!(sched[0], 0, "grayscale entry should encode sel=0 / comp=0");
    }

    #[test]
    fn mcu_schedule_selector_bounds_validation() {
        // Manufacture a prep with selectors ≥ num_components to trigger
        // the validation error.  Grayscale has num_components = 1, so
        // dc_selector = 1 is out of bounds.
        let mut prep = prepare_jpeg(GRAY_16X16_JPEG).unwrap();
        prep.dc_selectors[0] = 1; // 1 ≥ 1 component → invalid
        let err = build_mcu_schedule(&prep).expect_err("out-of-range selector must be rejected");
        assert!(
            matches!(err, JpegGpuError::InvalidHuffmanTables(_)),
            "expected InvalidHuffmanTables, got: {err:?}",
        );

        let mut prep2 = prepare_jpeg(GRAY_16X16_JPEG).unwrap();
        prep2.ac_selectors[0] = 2; // 2 ≥ 1 component → invalid
        let err2 =
            build_mcu_schedule(&prep2).expect_err("out-of-range AC selector must be rejected");
        assert!(
            matches!(err2, JpegGpuError::InvalidHuffmanTables(_)),
            "expected InvalidHuffmanTables (AC), got: {err2:?}",
        );
    }

    #[test]
    fn mcu_schedule_entry_encodes_correct_fields() {
        // Verify the bit-field packing: component_idx in bits 0..8,
        // dc_sel in bits 8..16, ac_sel in bits 16..24.
        // For grayscale with selectors 0 the check above covers it.
        // Manufacture a 2-component scenario by hand (bypassing prepare_jpeg)
        // to test non-zero selectors.
        let prep = prepare_jpeg(GRAY_16X16_JPEG).unwrap();
        let (sched, bpm) = build_mcu_schedule(&prep).unwrap();
        for entry in &sched {
            let comp_idx = entry & 0xFF;
            let dc_sel = (entry >> 8) & 0xFF;
            let ac_sel = (entry >> 16) & 0xFF;
            assert!(
                (comp_idx as usize) < prep.components.len(),
                "component_idx oob"
            );
            assert_eq!(dc_sel, u32::from(prep.dc_selectors[comp_idx as usize]));
            assert_eq!(ac_sel, u32::from(prep.ac_selectors[comp_idx as usize]));
        }
        let _ = bpm;
    }
}
