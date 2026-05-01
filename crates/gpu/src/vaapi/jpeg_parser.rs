//! Minimal JPEG header parser for VA-API parameter buffer construction.
//!
//! Extracts exactly what the driver needs — SOF0 (dimensions/components),
//! DQT (quantisation tables), DHT (Huffman tables), DRI (restart interval),
//! and SOS (scan header + data offset).  No allocations are performed; all
//! output fields are fixed-size arrays.

#![cfg(feature = "vaapi")]

use super::error::{Result, VapiError};
use super::ffi::VaHuffmanEntry;

/// Extracted JPEG headers needed to fill VA-API parameter buffers.
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
    /// Huffman tables: [0]=luma DC, [1]=luma AC, [2]=chroma DC, [3]=chroma AC.
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
    /// Parse a JPEG bitstream, extracting all headers needed by VA-API.
    ///
    /// # Errors
    ///
    /// Returns [`VapiError::BadJpeg`] on any parse error: missing SOI, truncated
    /// segments, progressive JPEG (not supported by `VAEntrypointVLD`), unknown
    /// segment structure, or missing SOF0/SOS markers.
    #[expect(
        clippy::too_many_lines,
        reason = "single-function JPEG header parser — the state machine naturally spans many lines; splitting mid-loop would reduce clarity"
    )]
    #[expect(
        clippy::similar_names,
        reason = "scan_dc_table and scan_ac_table mirror the JPEG spec field names; renaming would obscure the mapping to the standard"
    )]
    pub(super) fn parse(data: &[u8]) -> Result<Self> {
        let err = |msg: &str| VapiError::BadJpeg(msg.to_string());

        let mut pos = 0usize;

        macro_rules! read_bytes {
            ($n:expr) => {{
                let end = pos.checked_add($n).ok_or_else(|| err("integer overflow"))?;
                if end > data.len() {
                    return Err(err("unexpected end of JPEG data"));
                }
                let slice = &data[pos..end];
                pos = end;
                slice
            }};
        }

        macro_rules! read_u16 {
            () => {{
                let b = read_bytes!(2);
                u16::from_be_bytes([b[0], b[1]])
            }};
        }

        // SOI check.
        if read_bytes!(2) != [0xFF, 0xD8] {
            return Err(err("missing JPEG SOI marker"));
        }

        let mut width: u16 = 0;
        let mut height: u16 = 0;
        let mut components: u8 = 0;
        let mut comp_ids = [0u8; 4];
        let mut h_samp = [0u8; 4];
        let mut v_samp = [0u8; 4];
        let mut quant_sel = [0u8; 4];
        let mut quant_tables = [[0u8; 64]; 4];
        let mut quant_present = [false; 4];
        let mut huffman_entries: [Option<VaHuffmanEntry>; 4] = [None, None, None, None];
        let mut scan_comp_ids = [0u8; 4];
        let mut scan_dc_table = [0u8; 4];
        let mut scan_ac_table = [0u8; 4];
        let mut scan_components: u8 = 0;
        let mut restart_interval: u16 = 0;
        let mut scan_data_offset: usize = 0;
        let mut scan_data_size: usize = 0;

        loop {
            if pos + 2 > data.len() {
                break;
            }
            if data[pos] != 0xFF {
                return Err(err("expected JPEG marker 0xFF"));
            }
            let marker = data[pos + 1];
            pos += 2;

            match marker {
                // SOI (re-encountered) or RST0–RST7 — no payload, continue.
                0xD0..=0xD8 => {}

                // SOF0 — baseline DCT frame header.
                0xC0 => {
                    let seg_len = read_u16!() as usize;
                    if seg_len < 2 {
                        return Err(err("SOF0 segment too short"));
                    }
                    let body = read_bytes!(seg_len - 2);
                    if body.len() < 6 {
                        return Err(err("SOF0 body too short"));
                    }
                    height = u16::from_be_bytes([body[1], body[2]]);
                    width = u16::from_be_bytes([body[3], body[4]]);
                    components = body[5];
                    if (components as usize) * 3 + 6 > body.len() {
                        return Err(err("SOF0 component table overruns segment"));
                    }
                    for i in 0..components.min(4) as usize {
                        let b = 6 + i * 3;
                        comp_ids[i] = body[b];
                        h_samp[i] = body[b + 1] >> 4;
                        v_samp[i] = body[b + 1] & 0x0F;
                        quant_sel[i] = body[b + 2];
                    }
                }

                // DQT — define quantisation table(s).
                0xDB => {
                    let seg_len = read_u16!() as usize;
                    if seg_len < 2 {
                        return Err(err("DQT segment too short"));
                    }
                    let body = read_bytes!(seg_len - 2);
                    let mut off = 0usize;
                    while off < body.len() {
                        if off + 1 > body.len() {
                            break;
                        }
                        let id_prec = body[off];
                        off += 1;
                        let prec = id_prec >> 4;
                        let table_id = (id_prec & 0x0F) as usize;
                        if table_id >= 4 {
                            return Err(err("DQT: table ID >= 4"));
                        }
                        if prec == 0 {
                            if off + 64 > body.len() {
                                return Err(err("DQT: 8-bit table overruns segment"));
                            }
                            quant_tables[table_id].copy_from_slice(&body[off..off + 64]);
                            off += 64;
                        } else {
                            // 16-bit table: take high byte only (VA-API uses u8).
                            if off + 128 > body.len() {
                                return Err(err("DQT: 16-bit table overruns segment"));
                            }
                            for k in 0..64 {
                                quant_tables[table_id][k] = body[off + k * 2];
                            }
                            off += 128;
                        }
                        quant_present[table_id] = true;
                    }
                }

                // DHT — define Huffman table.
                0xC4 => {
                    let seg_len = read_u16!() as usize;
                    if seg_len < 2 {
                        return Err(err("DHT segment too short"));
                    }
                    let body = read_bytes!(seg_len - 2);
                    let mut off = 0usize;
                    while off < body.len() {
                        if off + 17 > body.len() {
                            break;
                        }
                        let tc_th = body[off];
                        off += 1;
                        let tc = (tc_th >> 4) & 0x1; // 0=DC, 1=AC
                        let th = (tc_th & 0x0F) as usize; // 0 or 1
                        if th >= 2 {
                            return Err(err("DHT: table index >= 2"));
                        }

                        let mut num_codes = [0u8; 16];
                        num_codes.copy_from_slice(&body[off..off + 16]);
                        off += 16;
                        let total_codes: usize = num_codes.iter().map(|&n| n as usize).sum();

                        // Slot: DC uses 0/2, AC uses 1/3.
                        // huffman_entries layout: [luma_DC, luma_AC, chroma_DC, chroma_AC].
                        let slot = th * 2 + tc as usize;
                        if slot >= 4 {
                            return Err(err("DHT: computed slot index out of range"));
                        }

                        let mut entry = VaHuffmanEntry {
                            num_dc_codes: [0; 16],
                            dc_values: [0; 12],
                            num_ac_codes: [0; 16],
                            ac_values: [0; 162],
                            _pad: [0; 2],
                        };

                        if tc == 0 {
                            if total_codes > 12 {
                                return Err(err("DHT: DC table has > 12 codes"));
                            }
                            if off + total_codes > body.len() {
                                return Err(err("DHT: DC values overrun segment"));
                            }
                            entry.num_dc_codes.copy_from_slice(&num_codes);
                            entry.dc_values[..total_codes]
                                .copy_from_slice(&body[off..off + total_codes]);
                        } else {
                            if total_codes > 162 {
                                return Err(err("DHT: AC table has > 162 codes"));
                            }
                            if off + total_codes > body.len() {
                                return Err(err("DHT: AC values overrun segment"));
                            }
                            entry.num_ac_codes.copy_from_slice(&num_codes);
                            entry.ac_values[..total_codes]
                                .copy_from_slice(&body[off..off + total_codes]);
                        }
                        off += total_codes;
                        huffman_entries[slot] = Some(entry);
                    }
                }

                // DRI — define restart interval.
                0xDD => {
                    let seg_len = read_u16!() as usize;
                    if seg_len < 2 {
                        return Err(err("DRI segment too short"));
                    }
                    let body = read_bytes!(seg_len - 2);
                    if body.len() >= 2 {
                        restart_interval = u16::from_be_bytes([body[0], body[1]]);
                    }
                }

                // SOS — start of scan: last header we need.
                0xDA => {
                    let seg_len = read_u16!() as usize;
                    if seg_len < 2 {
                        return Err(err("SOS segment too short"));
                    }
                    let body = read_bytes!(seg_len - 2);
                    if body.is_empty() {
                        return Err(err("SOS header empty"));
                    }
                    scan_components = body[0];
                    if (scan_components as usize) * 2 + 4 > body.len() {
                        return Err(err("SOS component table overruns header"));
                    }
                    for i in 0..scan_components.min(4) as usize {
                        scan_comp_ids[i] = body[1 + i * 2];
                        scan_dc_table[i] = body[2 + i * 2] >> 4;
                        scan_ac_table[i] = body[2 + i * 2] & 0x0F;
                    }
                    scan_data_offset = pos;
                    // Scan data runs from here to the EOI marker (last 0xFF 0xD9).
                    let eoi = data
                        .windows(2)
                        .rposition(|w| w == [0xFF, 0xD9])
                        .unwrap_or(data.len());
                    scan_data_size = eoi.saturating_sub(pos);
                    break;
                }

                // SOF2 — progressive DCT; not supported by VAEntrypointVLD.
                0xC2 => {
                    return Err(VapiError::BadJpeg(
                        "progressive JPEG not supported by VA-API VLD entrypoint".into(),
                    ));
                }

                // APP0–APP15 (0xE0–0xEF), APP-style / COM (0xF0–0xFE) — length-prefixed; skip.
                0xE0..=0xFE => {
                    let seg_len = read_u16!() as usize;
                    if seg_len < 2 {
                        return Err(err("marker segment length < 2"));
                    }
                    let skip = seg_len - 2;
                    pos = pos
                        .checked_add(skip)
                        .ok_or_else(|| err("integer overflow in segment skip"))?;
                    if pos > data.len() {
                        return Err(err("segment body overruns JPEG data"));
                    }
                }

                // EOI or any unrecognised marker — stop parsing.
                _ => break,
            }
        }

        if width == 0 || height == 0 {
            return Err(err("JPEG SOF0 not found or zero dimensions"));
        }
        if components == 0 {
            return Err(err("JPEG has 0 components"));
        }
        if scan_data_size == 0 {
            return Err(err("JPEG SOS not found or empty scan data"));
        }

        Ok(Self {
            width,
            height,
            components,
            comp_ids,
            h_samp,
            v_samp,
            quant_sel,
            quant_tables,
            quant_present,
            huffman_entries,
            scan_comp_ids,
            scan_dc_table,
            scan_ac_table,
            scan_components,
            restart_interval,
            scan_data_offset,
            scan_data_size,
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

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// 16×16 grayscale JPEG — SOF0 should parse to w=16, h=16, components=1.
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
    }

    #[test]
    fn parse_empty_returns_error() {
        assert!(matches!(JpegHeaders::parse(&[]), Err(VapiError::BadJpeg(_))));
    }

    #[test]
    fn parse_truncated_returns_error() {
        assert!(matches!(
            JpegHeaders::parse(&[0xFF, 0xD8]),
            Err(VapiError::BadJpeg(_))
        ));
    }
}
