//! VA-API adapter over the shared JPEG header parser.
//!
//! The actual marker walker lives in [`crate::jpeg::headers`] and is feature-
//! flag-free.  This module re-exposes those headers in the shape VA-API
//! consumers expect (4-slot DC/AC `VaHuffmanEntry` layout, fixed-size arrays
//! sized for the FFI buffers).
//!
//! Wire-format parsing logic is **not** duplicated — it lives in one place.

#![cfg(feature = "vaapi")]

use super::error::{Result, VapiError};
use super::ffi::VaHuffmanEntry;
use crate::jpeg::headers::{
    DhtClass, JpegHeaderError, JpegHeaders as SharedHeaders, JpegHuffmanTable,
};

/// Extracted JPEG headers in the shape VA-API parameter buffers expect.
///
/// This is an adapter around [`SharedHeaders`] that flattens the parsed data
/// into the fixed-size layouts VA-API's FFI structures need.  All wire-format
/// validation (segment lengths, table sizes, marker correctness) happens in
/// the shared parser before we get here.
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
}

impl JpegHeaders {
    /// Parse a JPEG bitstream and translate into the VA-API-shaped layout.
    ///
    /// # Errors
    ///
    /// Returns [`VapiError::BadJpeg`] for any header parse failure, with the
    /// underlying [`JpegHeaderError`] message as the cause.
    pub(super) fn parse(data: &[u8]) -> Result<Self> {
        let shared = SharedHeaders::parse(data).map_err(jpeg_header_to_vapi)?;

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
        })
    }

    /// Number of MCUs (minimum coded units) in the image.
    pub(super) fn num_mcus(&self) -> u32 {
        if self.components == 0 {
            return 0;
        }
        let max_h = self.h_samp[..self.components as usize]
            .iter()
            .copied()
            .max()
            .unwrap_or(1)
            .max(1);
        let max_v = self.v_samp[..self.components as usize]
            .iter()
            .copied()
            .max()
            .unwrap_or(1)
            .max(1);
        let mcu_w = u32::from(max_h) * 8;
        let mcu_h = u32::from(max_v) * 8;
        let w = u32::from(self.width);
        let h = u32::from(self.height);
        w.div_ceil(mcu_w) * h.div_ceil(mcu_h)
    }
}

/// Map shared parser error → VA-API error, preserving the message.
fn jpeg_header_to_vapi(e: JpegHeaderError) -> VapiError {
    VapiError::BadJpeg(e.to_string())
}

/// Pack the shared parser's `Vec<JpegHuffmanTable>` into VA-API's 4-slot
/// `[luma_DC=0, luma_AC=1, chroma_DC=2, chroma_AC=3]` layout.
///
/// VA-API only accepts table_id 0 (luma) and 1 (chroma) in baseline JPEG; the
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

    /// Same 16×16 grayscale fixture used elsewhere in this crate's tests.
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
        let minimal_sof2: &[u8] = &[
            0xFF, 0xD8, 0xFF, 0xC2, 0x00, 0x11, 0x08, 0x00, 0x10, 0x00, 0x10, 0x03, 0x01, 0x11,
            0x00, 0x02, 0x11, 0x01, 0x03, 0x11, 0x01, 0xFF, 0xD9,
        ];
        match JpegHeaders::parse(minimal_sof2) {
            Err(VapiError::BadJpeg(msg)) => assert!(
                !msg.contains("progressive"),
                "VA-API parser should not reject progressive JPEG itself; got: {msg}",
            ),
            Err(other) => panic!("unexpected error variant: {other:?}"),
            Ok(_) => panic!("SOF2-only stream must not parse successfully"),
        }
    }
}
