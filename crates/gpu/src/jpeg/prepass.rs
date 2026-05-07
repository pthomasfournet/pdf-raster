//! Phase 0 orchestrator: turn a JPEG byte slice into [`CpuPrepassOutput`]
//! ready for upload to the on-GPU decoder pipeline.
//!
//! Composition:
//!
//! ```text
//! bytes ──▶ headers::JpegHeaders::parse ──▶ headers + scan_data borrow
//!                                                │
//!                                                ▼
//!                                  unstuff::unstuff_into ──▶ flat bitstream + RST positions
//!                                                │
//!                                                ▼
//!                                  build_scan_codebooks ──▶ DC + AC canonical lookup tables
//!                                                │
//!                                                ▼
//!                                  dc_chain::resolve_dc_chain ──▶ DcValues
//!                                                │
//!                                                ▼
//!                                  CpuPrepassOutput
//! ```
//!
//! The output is the input contract for Phase 1 (parallel Huffman) and Phase 5
//! (IDCT + colour convert).  The codebook construction is shared between the
//! DC-chain resolver here and Phase 1 — built once, consumed by both.

use super::canonical::{CanonicalCodebook, CanonicalCodebookError};
use super::dc_chain::{self, DcChainError, DcValues};
use super::headers::{DhtClass, JpegHeaderError, JpegHeaders, JpegScanComponent, JpegScanHeader};
use super::unstuff::{self, RstPosition, UnstuffError};

/// All CPU-side data the GPU pipeline needs to decode this JPEG.
///
/// Owned (not borrowed) so the original `&[u8]` can be released after Phase 0.
/// Lives until the GPU finishes decoding; typically one per in-flight image.
#[derive(Debug)]
pub struct CpuPrepassOutput {
    /// Image dimensions in pixels.
    pub width: u16,
    /// Image dimensions in pixels.
    pub height: u16,
    /// 1 = grayscale, 3 = YCbCr or RGB, 4 = CMYK.
    pub components: u8,
    /// Per-component frame metadata (only the first `components` entries valid).
    pub frame_components: [super::headers::JpegFrameComponent; 4],
    /// Quantisation tables, indexed by table ID 0..=3.  `None` slots were
    /// never loaded from a DQT segment.
    pub quant_tables: [Option<super::headers::JpegQuantTable>; 4],
    /// Canonical Huffman lookup tables for AC decoding.  Indexed by `table_id`
    /// (0..=1 in baseline JPEG); only entries referenced by the scan are present.
    pub ac_codebooks: [Option<CanonicalCodebook>; 4],
    /// Scan header (which components, which Huffman selectors).
    pub scan: super::headers::JpegScanHeader,
    /// Restart interval in MCUs (0 = none).  When > 0, [`Self::rst_positions`]
    /// is non-empty.
    pub restart_interval: u16,
    /// Flat entropy-coded bitstream, byte-stuffing already removed.
    pub unstuffed: Vec<u8>,
    /// Byte offsets within `unstuffed` where each restart marker landed.
    /// Empty when `restart_interval == 0`.
    pub rst_positions: Vec<RstPosition>,
    /// Per-component absolute DC values resolved on the CPU.
    pub dc_values: DcValues,
}

impl CpuPrepassOutput {
    /// Convenience accessor: number of pixels in the decoded image.
    #[must_use]
    pub fn pixel_count(&self) -> u64 {
        u64::from(self.width) * u64::from(self.height)
    }

    /// Active per-component frame metadata (slice of length `self.components`).
    #[must_use]
    pub fn active_frame_components(&self) -> &[super::headers::JpegFrameComponent] {
        &self.frame_components[..usize::from(self.components)]
    }

    /// Number of MCUs in the image.  Computed from `width`/`height`/sampling
    /// factors on demand — cheap and cache-line-friendly.
    #[must_use]
    pub fn num_mcus(&self) -> u32 {
        super::headers::mcu_count(self.width, self.height, self.active_frame_components())
    }
}

/// Errors emitted by [`run_cpu_prepass`].
///
/// Each variant wraps the underlying submodule error so the caller can
/// differentiate "header malformed" from "Huffman table illegal" from
/// "byte-unstuffing hit unexpected marker" from "DC chain corrupt", and route
/// to the right fallback.
#[derive(Debug)]
pub enum CpuPrepassError {
    /// Header parsing failed.
    Header(JpegHeaderError),
    /// Byte-unstuffing failed.
    Unstuff(UnstuffError),
    /// Building a canonical Huffman codebook failed.
    Codebook {
        /// DC or AC.
        class: DhtClass,
        /// Selector value (0..=3) the scan referenced.
        selector: u8,
        /// Underlying error from the codebook builder.
        source: CanonicalCodebookError,
    },
    /// Scan referenced a Huffman table that no DHT segment declared.
    MissingHuffmanTable {
        /// DC or AC.
        class: DhtClass,
        /// Selector value the scan referenced.
        selector: u8,
    },
    /// DC chain resolution failed.
    DcChain(DcChainError),
}

impl std::fmt::Display for CpuPrepassError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Header(e) => write!(f, "header parse: {e}"),
            Self::Unstuff(e) => write!(f, "byte unstuffing: {e}"),
            Self::Codebook {
                class,
                selector,
                source,
            } => write!(
                f,
                "build {class} Huffman codebook (selector {selector}): {source}",
            ),
            Self::MissingHuffmanTable { class, selector } => write!(
                f,
                "scan references {class} Huffman table {selector} but no DHT declared it",
            ),
            Self::DcChain(e) => write!(f, "DC chain pre-pass: {e}"),
        }
    }
}

impl std::error::Error for CpuPrepassError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Header(e) => Some(e),
            Self::Unstuff(e) => Some(e),
            Self::Codebook { source, .. } => Some(source),
            Self::MissingHuffmanTable { .. } => None,
            Self::DcChain(e) => Some(e),
        }
    }
}

impl From<JpegHeaderError> for CpuPrepassError {
    fn from(value: JpegHeaderError) -> Self {
        Self::Header(value)
    }
}

impl From<UnstuffError> for CpuPrepassError {
    fn from(value: UnstuffError) -> Self {
        Self::Unstuff(value)
    }
}

impl From<DcChainError> for CpuPrepassError {
    fn from(value: DcChainError) -> Self {
        Self::DcChain(value)
    }
}

/// Run Phase 0 end-to-end on a JPEG byte slice.
///
/// # Errors
///
/// Returns [`CpuPrepassError`] on any submodule failure.  The caller is
/// expected to fall back to a CPU JPEG decoder (zune-jpeg) on failure rather
/// than treating the image as fatal.
#[must_use = "discarding the prepass output discards all the decoded headers + DC values"]
pub fn run_cpu_prepass(data: &[u8]) -> Result<CpuPrepassOutput, CpuPrepassError> {
    let headers = JpegHeaders::parse(data)?;

    let mut unstuffed = Vec::with_capacity(headers.scan_data.len());
    let mut rst_positions = Vec::new();
    unstuff::unstuff_into(headers.scan_data, &mut unstuffed, &mut rst_positions)?;

    // Build codebooks once, hand both DC and AC sets to the DC-chain walker.
    // AC codebooks live in CpuPrepassOutput so the parallel-Huffman kernel
    // can reuse them without rebuilding.
    let (dc_codebooks, ac_codebooks) = build_scan_codebooks(&headers)?;

    let dc_values = dc_chain::resolve_dc_chain(
        &headers,
        &unstuffed,
        &rst_positions,
        &dc_codebooks,
        &ac_codebooks,
    )?;

    Ok(CpuPrepassOutput {
        width: headers.width,
        height: headers.height,
        components: headers.components,
        frame_components: headers.frame_components,
        quant_tables: headers.quant_tables,
        ac_codebooks,
        scan: headers.scan,
        restart_interval: headers.restart_interval,
        unstuffed,
        rst_positions,
        dc_values,
    })
}

/// Indexed-by-selector array of canonical Huffman codebooks.  Only the
/// `Some(...)` slots are populated; absent slots correspond to selectors the
/// scan never referenced.
pub(super) type CodebookSet = [Option<CanonicalCodebook>; 4];

/// Build canonical Huffman lookup tables for every (class, selector) the
/// scan header references.  Returns parallel [`CodebookSet`]s for DC and AC
/// codebooks; only the slots actually referenced by the scan are populated.
///
/// A scan must reference at least one DC selector and one AC selector; both
/// arrays are guaranteed to have at least one populated entry on success.
fn build_scan_codebooks(
    headers: &JpegHeaders<'_>,
) -> Result<(CodebookSet, CodebookSet), CpuPrepassError> {
    let mut dc_codebooks: CodebookSet = Default::default();
    let mut ac_codebooks: CodebookSet = Default::default();
    for sc in scan_components(&headers.scan) {
        if dc_codebooks[usize::from(sc.dc_table)].is_none() {
            dc_codebooks[usize::from(sc.dc_table)] =
                Some(load_codebook(headers, DhtClass::Dc, sc.dc_table)?);
        }
        if ac_codebooks[usize::from(sc.ac_table)].is_none() {
            ac_codebooks[usize::from(sc.ac_table)] =
                Some(load_codebook(headers, DhtClass::Ac, sc.ac_table)?);
        }
    }
    Ok((dc_codebooks, ac_codebooks))
}

fn load_codebook(
    headers: &JpegHeaders<'_>,
    class: DhtClass,
    selector: u8,
) -> Result<CanonicalCodebook, CpuPrepassError> {
    let table = headers
        .huffman(class, selector)
        .ok_or(CpuPrepassError::MissingHuffmanTable { class, selector })?;
    CanonicalCodebook::build(table).map_err(|source| CpuPrepassError::Codebook {
        class,
        selector,
        source,
    })
}

/// Iterator over the active scan components, hiding the
/// `&scan.components[..scan.component_count as usize]` boilerplate.
fn scan_components(scan: &JpegScanHeader) -> &[JpegScanComponent] {
    &scan.components[..usize::from(scan.component_count)]
}

#[cfg(test)]
mod tests {
    use super::super::test_fixtures::GRAY_16X16_JPEG;
    use super::*;

    #[test]
    fn prepass_succeeds_on_minimal_grayscale() {
        let out = run_cpu_prepass(GRAY_16X16_JPEG).expect("prepass failed");
        assert_eq!(out.width, 16);
        assert_eq!(out.height, 16);
        assert_eq!(out.components, 1);
        assert_eq!(out.num_mcus(), 4);
        // 4 MCUs × 1 block each = 4 DC values for the single component.
        assert_eq!(out.dc_values.per_component.len(), 1);
        assert_eq!(out.dc_values.per_component[0].len(), 4);
        // Unstuffed bitstream is non-empty.
        assert!(!out.unstuffed.is_empty());
        // No restart interval in the fixture, so no RST positions.
        assert_eq!(out.restart_interval, 0);
        assert!(out.rst_positions.is_empty());
        // AC codebook for selector 0 (the only one referenced by the scan)
        // is built; selectors 1..=3 are absent.
        assert!(out.ac_codebooks[0].is_some());
        assert!(out.ac_codebooks[1].is_none());
    }

    #[test]
    fn prepass_returns_header_error_on_garbage() {
        let err = run_cpu_prepass(&[0x00, 0x00, 0x00]).unwrap_err();
        assert!(matches!(err, CpuPrepassError::Header(_)));
    }

    #[test]
    fn prepass_pixel_count_matches_dimensions() {
        let out = run_cpu_prepass(GRAY_16X16_JPEG).unwrap();
        assert_eq!(out.pixel_count(), 16 * 16);
    }

    #[test]
    fn prepass_codebook_error_carries_class_and_selector() {
        // Build a JPEG with a DHT that declares 3 length-1 codes (overflows
        // the 1-bit code space of 2 entries).  This must fail with a
        // Codebook variant carrying the class/selector that referenced it.
        // Hand-rolled minimal SOF0+DHT+DHT+SOS + EOI:
        //   SOI / SOF0 (1c, 16×16) / DQT (table 0, 64 zeros) /
        //   DHT (DC class=0 id=0; num_codes [3,0,...]; values [0,1,2]) /
        //   DHT (AC class=1 id=0; num_codes [1,0,...]; values [0])  /
        //   SOS / 1 entropy byte / EOI
        let mut data: Vec<u8> = vec![0xFF, 0xD8];
        // DQT, table 0, 64 zeros
        data.extend_from_slice(&[0xFF, 0xDB, 0x00, 67, 0]);
        data.extend_from_slice(&[0u8; 64]);
        // SOF0, 1 component
        data.extend_from_slice(&[
            0xFF, 0xC0, 0x00, 11, 8, 0x00, 0x10, 0x00, 0x10, 1, 1, 0x11, 0x00,
        ]);
        // DHT DC table 0: class+id=0x00, num_codes=[3,0,...], values=[0,1,2]
        let mut dht = vec![0xFF, 0xC4, 0x00, 22, 0x00];
        dht.extend_from_slice(&[3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        dht.extend_from_slice(&[0, 1, 2]);
        data.extend_from_slice(&dht);
        // DHT AC table 0: class+id=0x10, one length-1 code, value=0
        data.extend_from_slice(&[
            0xFF, 0xC4, 0x00, 20, 0x10, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ]);
        // SOS, 1 component
        data.extend_from_slice(&[0xFF, 0xDA, 0x00, 8, 1, 1, 0x00, 0, 0x3F, 0x00, 0x80]);
        data.extend_from_slice(&[0xFF, 0xD9]);

        let err = run_cpu_prepass(&data).expect_err("invalid DC table must fail");
        match err {
            CpuPrepassError::Codebook {
                class, selector, ..
            } => {
                assert_eq!(class, DhtClass::Dc);
                assert_eq!(selector, 0);
            }
            other => panic!("expected Codebook error, got: {other:?}"),
        }
    }
}
