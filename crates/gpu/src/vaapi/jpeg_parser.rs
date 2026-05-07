//! VA-API adapter over the shared JPEG header parser.
//!
//! The actual marker walker lives in [`crate::jpeg::headers`] and is feature-
//! flag-free.  This module re-exposes those headers in the shape VA-API
//! consumers expect (4-slot DC/AC `VaHuffmanEntry` layout, fixed-size arrays
//! sized for the FFI buffers).
//!
//! Wire-format parsing logic, including [`crate::jpeg::headers::JpegHeaders::num_mcus`],
//! is **not** duplicated — it lives in one place.

#![cfg(feature = "vaapi")]

use super::error::{Result, VapiError};
use super::ffi::VaHuffmanEntry;
use crate::jpeg::headers::{
    DhtClass, JpegHeaderError, JpegHeaders as SharedHeaders, JpegHuffmanTable,
};

/// Extracted JPEG headers in the shape VA-API parameter buffers expect.
///
/// This is an adapter around the shared [`SharedHeaders`] that flattens the
/// parsed data into the fixed-size layouts VA-API's FFI structures need.
/// All wire-format validation (segment lengths, table sizes, marker
/// correctness, sampling-factor ranges) happens in the shared parser before
/// we get here.
pub(super) struct JpegHeaders {
    pub(super) width: u16,
    pub(super) height: u16,
    pub(super) components: u8,
    pub(super) comp_ids: [u8; 4],
    pub(super) h_samp: [u8; 4],
    pub(super) v_samp: [u8; 4],
    pub(super) quant_sel: [u8; 4],
    /// Up to 4 quantisation tables (zigzag order, 64 bytes each).
    pub(super) quant_tables: [[u8; 64]; 4],
    pub(super) quant_present: [bool; 4],
    /// Huffman tables packed for VA-API: `[luma_DC=0, luma_AC=1, chroma_DC=2, chroma_AC=3]`.
    pub(super) huffman_entries: [Option<VaHuffmanEntry>; 4],
    /// SOS scan component selectors.
    pub(super) scan_comp_ids: [u8; 4],
    pub(super) scan_dc_table: [u8; 4],
    pub(super) scan_ac_table: [u8; 4],
    pub(super) scan_components: u8,
    pub(super) restart_interval: u16,
    /// Byte offset within the original JPEG data where the compressed scan starts.
    pub(super) scan_data_offset: usize,
    /// Length of the compressed scan data in bytes.
    pub(super) scan_data_size: usize,
    /// Pre-computed MCU count, delegated to the shared parser at parse time
    /// so the formula lives in exactly one place.
    num_mcus_cached: u32,
}

impl JpegHeaders {
    /// Parse a JPEG bitstream and translate into the VA-API-shaped layout.
    ///
    /// # Errors
    ///
    /// Returns [`VapiError::BadJpeg`] for any header parse failure, with the
    /// underlying [`JpegHeaderError`] message as the cause.
    #[expect(
        clippy::similar_names,
        reason = "scan_dc_table/scan_ac_table mirror JPEG spec field names; renaming would obscure the mapping"
    )]
    pub(super) fn parse(data: &[u8]) -> Result<Self> {
        let shared = SharedHeaders::parse(data).map_err(|e| jpeg_header_to_vapi(&e))?;
        let num_mcus_cached = shared.num_mcus();

        let mut comp_ids = [0u8; 4];
        let mut h_samp = [0u8; 4];
        let mut v_samp = [0u8; 4];
        let mut quant_sel = [0u8; 4];
        for (i, fc) in shared.frame_components.iter().enumerate() {
            comp_ids[i] = fc.id;
            h_samp[i] = fc.h_sampling;
            v_samp[i] = fc.v_sampling;
            quant_sel[i] = fc.quant_selector;
        }

        let mut quant_tables = [[0u8; 64]; 4];
        for (i, qt) in shared.quant_tables.iter().enumerate() {
            quant_tables[i] = qt.values;
        }

        let huffman_entries = pack_huffman_entries(&shared.huffman_tables);

        let mut scan_comp_ids = [0u8; 4];
        let mut scan_dc_table = [0u8; 4];
        let mut scan_ac_table = [0u8; 4];
        for (i, sc) in shared.scan.components.iter().enumerate() {
            scan_comp_ids[i] = sc.id;
            scan_dc_table[i] = sc.dc_table;
            scan_ac_table[i] = sc.ac_table;
        }

        Ok(Self {
            width: shared.width,
            height: shared.height,
            components: shared.components,
            comp_ids,
            h_samp,
            v_samp,
            quant_sel,
            quant_tables,
            quant_present: shared.quant_present,
            huffman_entries,
            scan_comp_ids,
            scan_dc_table,
            scan_ac_table,
            scan_components: shared.scan.component_count,
            restart_interval: shared.restart_interval,
            scan_data_offset: shared.scan_data_offset,
            scan_data_size: shared.scan_data.len(),
            num_mcus_cached,
        })
    }

    /// Number of MCUs (minimum coded units) in the image.  Cached at parse
    /// time using the shared `SharedHeaders::num_mcus()` formula.
    pub(super) const fn num_mcus(&self) -> u32 {
        self.num_mcus_cached
    }
}

/// Map shared parser error → VA-API error, preserving the message.
fn jpeg_header_to_vapi(e: &JpegHeaderError) -> VapiError {
    VapiError::BadJpeg(e.to_string())
}

/// Pack the shared parser's `Vec<JpegHuffmanTable>` into VA-API's 4-slot
/// `[luma_DC=0, luma_AC=1, chroma_DC=2, chroma_AC=3]` layout.
///
/// VA-API only accepts `table_id` 0 (luma) and 1 (chroma) in baseline JPEG; the
/// shared parser permits 0..=3, but a baseline-conforming stream will only
/// reference 0 and 1.  Out-of-range tables are silently dropped here — the
/// caller's SOS-table-selector validation already catches the spec violation.
fn pack_huffman_entries(tables: &[JpegHuffmanTable]) -> [Option<VaHuffmanEntry>; 4] {
    let mut entries: [Option<VaHuffmanEntry>; 4] = [None, None, None, None];
    for t in tables {
        if t.table_id >= 2 {
            continue;
        }
        let slot = (t.table_id as usize) * 2
            + match t.class {
                DhtClass::Dc => 0,
                DhtClass::Ac => 1,
            };
        let mut entry = entries[slot].take().unwrap_or(VaHuffmanEntry {
            num_dc_codes: [0; 16],
            dc_values: [0; 12],
            num_ac_codes: [0; 16],
            ac_values: [0; 162],
            _pad: [0; 2],
        });
        match t.class {
            DhtClass::Dc => {
                entry.num_dc_codes = t.num_codes;
                let n = t.values.len().min(12);
                entry.dc_values[..n].copy_from_slice(&t.values[..n]);
            }
            DhtClass::Ac => {
                entry.num_ac_codes = t.num_codes;
                let n = t.values.len().min(162);
                entry.ac_values[..n].copy_from_slice(&t.values[..n]);
            }
        }
        entries[slot] = Some(entry);
    }
    entries
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jpeg::test_fixtures::{GRAY_16X16_JPEG, PROGRESSIVE_MINIMAL};

    #[test]
    fn parse_gray_16x16_headers() {
        let h = JpegHeaders::parse(GRAY_16X16_JPEG).expect("header parse failed");
        assert_eq!(h.width, 16);
        assert_eq!(h.height, 16);
        assert_eq!(h.components, 1);
        assert!(h.quant_present[0], "quant table 0 must be present");
        assert!(h.scan_data_size > 0, "scan data must be non-empty");
        // Slot 0 = luma DC, slot 1 = luma AC: both present in this fixture.
        assert!(h.huffman_entries[0].is_some(), "luma DC entry missing");
        assert!(h.huffman_entries[1].is_some(), "luma AC entry missing");
        // num_mcus delegated to the shared formula: 16×16 grayscale → 4 MCUs.
        assert_eq!(h.num_mcus(), 4);
    }

    #[test]
    fn parse_empty_returns_error() {
        assert!(matches!(
            JpegHeaders::parse(&[]),
            Err(VapiError::BadJpeg(_))
        ));
    }

    #[test]
    fn parse_truncated_returns_error() {
        assert!(matches!(
            JpegHeaders::parse(&[0xFF, 0xD8]),
            Err(VapiError::BadJpeg(_))
        ));
    }

    #[test]
    fn parse_does_not_reject_sof2_directly() {
        // The shared parser fails non-SOF0 streams via MissingSof0/Truncated;
        // either error is fine, but the message must NOT mention "progressive"
        // — that's the upstream router's responsibility (jpeg_sof_type).
        match JpegHeaders::parse(PROGRESSIVE_MINIMAL) {
            Err(VapiError::BadJpeg(msg)) => assert!(
                !msg.contains("progressive"),
                "VA-API parser should not reject progressive JPEG itself; got: {msg}",
            ),
            Err(other) => panic!("unexpected error variant: {other:?}"),
            Ok(_) => panic!("SOF2-only stream must not parse successfully"),
        }
    }
}
