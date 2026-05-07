//! Phase 0 orchestrator: turn a JPEG byte slice into [`CpuPrepassOutput`]
//! ready for upload to the on-GPU decoder pipeline.
//!
//! Composition:
//!
//! ```text
//! bytes ──▶ headers::JpegHeaders::parse ──▶ headers + scan_data borrow
//!                                                │
//!                                                ▼
//!                                  unstuff::unstuff_into ──▶ flat bitstream
//!                                                │
//!                                                ▼
//!                                  dc_chain::resolve_dc_chain ──▶ DcValues
//!                                                │
//!                                                ▼
//!                                  CpuPrepassOutput
//! ```
//!
//! The output is the input contract for Phase 1 (parallel Huffman) and Phase 5
//! (IDCT + colour convert).

use super::canonical::{CanonicalCodebook, CanonicalCodebookError};
use super::dc_chain::{self, DcChainError, DcValues};
use super::headers::{DhtClass, JpegHeaderError, JpegHeaders};
use super::unstuff::{self, UnstuffError};

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
    /// Quantisation tables.  `quant_present[i]` says which slots are valid.
    pub quant_tables: [super::headers::JpegQuantTable; 4],
    /// Bitmap of valid quantiser slots.
    pub quant_present: [bool; 4],
    /// Canonical Huffman lookup tables for AC decoding.  Indexed by table_id
    /// (0..=1 in baseline JPEG); only entries referenced by the scan are present.
    pub ac_codebooks: [Option<CanonicalCodebook>; 4],
    /// Scan header (which components, which Huffman selectors).
    pub scan: super::headers::JpegScanHeader,
    /// Restart interval in MCUs (0 = none).
    pub restart_interval: u16,
    /// Flat entropy-coded bitstream, byte-stuffing already removed.
    pub unstuffed: Vec<u8>,
    /// Per-component absolute DC values resolved on the CPU.
    pub dc_values: DcValues,
    /// Total MCU count, redundantly cached so callers don't re-derive it.
    pub num_mcus: u32,
}

impl CpuPrepassOutput {
    /// Convenience accessor: number of pixels in the decoded image.
    #[must_use]
    pub fn pixel_count(&self) -> u64 {
        u64::from(self.width) * u64::from(self.height)
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
    Codebook(CanonicalCodebookError),
    /// DC chain resolution failed.
    DcChain(DcChainError),
}

impl std::fmt::Display for CpuPrepassError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Header(e) => write!(f, "header parse: {e}"),
            Self::Unstuff(e) => write!(f, "byte unstuffing: {e}"),
            Self::Codebook(e) => write!(f, "canonical Huffman codebook: {e}"),
            Self::DcChain(e) => write!(f, "DC chain pre-pass: {e}"),
        }
    }
}

impl std::error::Error for CpuPrepassError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Header(e) => Some(e),
            Self::Unstuff(e) => Some(e),
            Self::Codebook(e) => Some(e),
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

impl From<CanonicalCodebookError> for CpuPrepassError {
    fn from(value: CanonicalCodebookError) -> Self {
        Self::Codebook(value)
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
pub fn run_cpu_prepass(data: &[u8]) -> Result<CpuPrepassOutput, CpuPrepassError> {
    let headers = JpegHeaders::parse(data)?;

    // Strip 0xFF 0x00 byte-stuffing.
    let mut unstuffed = Vec::with_capacity(headers.scan_data.len());
    unstuff::unstuff_into(headers.scan_data, &mut unstuffed)?;

    // Build AC codebooks for every selector the scan references.  DC
    // codebooks are built inside resolve_dc_chain (and discarded after) since
    // Phase 1 only needs the AC ones — the DC chain is resolved on CPU.
    let mut ac_codebooks: [Option<CanonicalCodebook>; 4] = [None, None, None, None];
    let scan_components = headers.scan.component_count as usize;
    for sc in &headers.scan.components[..scan_components] {
        let sel = sc.ac_table as usize;
        if ac_codebooks[sel].is_none() {
            let table = headers.huffman(DhtClass::Ac, sc.ac_table).ok_or_else(|| {
                CpuPrepassError::DcChain(DcChainError::MissingHuffmanTable {
                    class: DhtClass::Ac,
                    selector: sc.ac_table,
                })
            })?;
            ac_codebooks[sel] = Some(CanonicalCodebook::build(table)?);
        }
    }

    let dc_values = dc_chain::resolve_dc_chain(&headers, &unstuffed)?;
    let num_mcus = headers.num_mcus();

    Ok(CpuPrepassOutput {
        width: headers.width,
        height: headers.height,
        components: headers.components,
        frame_components: headers.frame_components,
        quant_tables: headers.quant_tables,
        quant_present: headers.quant_present,
        ac_codebooks,
        scan: headers.scan,
        restart_interval: headers.restart_interval,
        unstuffed,
        dc_values,
        num_mcus,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Same 16×16 grayscale fixture as in `headers::tests`.
    const GRAY_16X16_JPEG: &[u8] = &[
        0xff, 0xd8, 0xff, 0xe0, 0x00, 0x10, 0x4a, 0x46, 0x49, 0x46, 0x00, 0x01, 0x01, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0xdb, 0x00, 0x43, 0x00, 0x06, 0x04, 0x05, 0x06, 0x05,
        0x04, 0x06, 0x06, 0x05, 0x06, 0x07, 0x07, 0x06, 0x08, 0x0a, 0x10, 0x0a, 0x0a, 0x09, 0x09,
        0x0a, 0x14, 0x0e, 0x0f, 0x0c, 0x10, 0x17, 0x14, 0x18, 0x18, 0x17, 0x14, 0x16, 0x16, 0x1a,
        0x1d, 0x25, 0x1f, 0x1a, 0x1b, 0x23, 0x1c, 0x16, 0x16, 0x20, 0x2c, 0x20, 0x23, 0x26, 0x27,
        0x29, 0x2a, 0x29, 0x19, 0x1f, 0x2d, 0x30, 0x2d, 0x28, 0x30, 0x25, 0x28, 0x29, 0x28, 0xff,
        0xc0, 0x00, 0x0b, 0x08, 0x00, 0x10, 0x00, 0x10, 0x01, 0x01, 0x11, 0x00, 0xff, 0xc4, 0x00,
        0x15, 0x00, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x01, 0xff, 0xc4, 0x00, 0x14, 0x10, 0x01, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0xda, 0x00,
        0x08, 0x01, 0x01, 0x00, 0x00, 0x3f, 0x00, 0x80, 0x3f, 0xff, 0xd9,
    ];

    #[test]
    fn prepass_succeeds_on_minimal_grayscale() {
        let out = run_cpu_prepass(GRAY_16X16_JPEG).expect("prepass failed");
        assert_eq!(out.width, 16);
        assert_eq!(out.height, 16);
        assert_eq!(out.components, 1);
        assert_eq!(out.num_mcus, 4);
        // 4 MCUs × 1 block each = 4 DC values for the single component.
        assert_eq!(out.dc_values.per_component.len(), 1);
        assert_eq!(out.dc_values.per_component[0].len(), 4);
        // Unstuffed bitstream is non-empty.
        assert!(!out.unstuffed.is_empty());
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
}
