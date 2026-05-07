//! Shared JPEG header parser.
//!
//! Walks the JPEG marker stream and extracts every header that any downstream
//! consumer (VA-API parameter buffers, on-GPU decoder Phase 0, future paths)
//! needs.  No allocations on the hot path beyond a single `Vec<JpegHuffmanTable>`
//! (typically 4 entries).
//!
//! This parser handles only what we actually decode:
//!
//! - SOI / EOI framing.
//! - SOF0 (baseline DCT, 8-bit) — dimensions, components, sampling factors,
//!   quantiser-table selectors. Other SOF flavours (progressive SOF2, lossless
//!   SOF3, etc.) are detected via [`crate::jpeg_sof::jpeg_sof_type`] before
//!   calling [`JpegHeaders::parse`]; this parser deliberately fails when
//!   confronted with non-SOF0 data so the caller routes it elsewhere.
//! - DQT — 8-bit and 16-bit quantisation tables. 16-bit values are clamped to
//!   `u8` since every existing consumer (VA-API, our IDCT kernel) takes 8-bit
//!   quantisers; widening that contract is a separate change.
//! - DHT — DC and AC Huffman tables in JPEG wire form `(num_codes[16], values[])`.
//!   Canonical-code construction lives in [`super::canonical`].
//! - SOS — scan header + offset/length of the entropy-coded data.
//! - DRI — restart interval (0 if absent).
//! - APPn / COM (0xE0–0xFE) — skipped via length prefix.
//! - RSTn / SOI re-encounters (0xD0–0xD8) — silently consumed if encountered
//!   before SOS, treated as scan boundaries afterwards.

use std::error::Error;
use std::fmt;

/// A successfully parsed set of JPEG headers.
///
/// Lifetime is tied to the input slice: `scan_data` is a borrow of the
/// entropy-coded segment within the original JPEG bytes, *with byte-stuffing
/// still intact*. Use [`super::unstuff::unstuff_into`] to materialise a flat
/// bitstream.
#[derive(Debug, Clone)]
pub struct JpegHeaders<'a> {
    /// Image width in pixels, from SOF0.
    pub width: u16,
    /// Image height in pixels, from SOF0.
    pub height: u16,
    /// Number of components in the image (1 = grayscale, 3 = YCbCr/RGB, 4 = CMYK).
    pub components: u8,
    /// Per-component metadata (only the first `components` entries are valid).
    pub frame_components: [JpegFrameComponent; 4],
    /// Quantisation tables, indexed by table ID 0..=3. Only `quant_present` slots are valid.
    pub quant_tables: [JpegQuantTable; 4],
    /// Bitmap: `quant_present[i] == true` iff `quant_tables[i]` was loaded from a DQT segment.
    pub quant_present: [bool; 4],
    /// All Huffman tables seen in DHT segments. Index by `(class, table_id)` via [`Self::huffman`].
    pub huffman_tables: Vec<JpegHuffmanTable>,
    /// SOS scan header.
    pub scan: JpegScanHeader,
    /// Restart interval in MCUs (0 = no restart markers).
    pub restart_interval: u16,
    /// Borrow of the entropy-coded segment, byte-stuffing intact.
    pub scan_data: &'a [u8],
    /// Byte offset of `scan_data` within the original input.
    pub scan_data_offset: usize,
}

/// Per-component frame metadata from SOF0.
#[derive(Debug, Clone, Copy, Default)]
pub struct JpegFrameComponent {
    /// Component identifier (1=Y, 2=Cb, 3=Cr by JFIF convention; arbitrary otherwise).
    pub id: u8,
    /// Horizontal sampling factor (1..=4).
    pub h_sampling: u8,
    /// Vertical sampling factor (1..=4).
    pub v_sampling: u8,
    /// Quantiser table selector (0..=3).
    pub quant_selector: u8,
}

/// One quantisation table — 64 entries in **zigzag order**, dequantised values
/// applied directly to the entropy-decoded coefficients before IDCT.
#[derive(Debug, Clone, Copy)]
pub struct JpegQuantTable {
    /// 64 zigzag-ordered quantiser values. 16-bit table values are clamped to `u8`.
    pub values: [u8; 64],
}

impl Default for JpegQuantTable {
    fn default() -> Self {
        Self { values: [0; 64] }
    }
}

/// Class of a Huffman table — DC differentials or AC run/size pairs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DhtClass {
    /// DC differential codes (run length is implicit; `values` are size categories 0..=11).
    Dc,
    /// AC run/size codes (`values` encode `(run << 4) | size`, max 162 distinct codes).
    Ac,
}

/// One Huffman table parsed from a DHT segment, in JPEG wire form.
///
/// The `(num_codes, values)` pair is the spec-canonical representation:
/// `num_codes[i]` is the number of codewords with `i+1` bits.
#[derive(Debug, Clone)]
pub struct JpegHuffmanTable {
    /// DC or AC.
    pub class: DhtClass,
    /// Table identifier (0..=1 in JPEG baseline; we accept 0..=3 to match DCT-frame max).
    pub table_id: u8,
    /// `num_codes[i]` = number of `i+1`-bit codewords (i in 0..16).
    pub num_codes: [u8; 16],
    /// Symbol values in canonical order; length = sum(num_codes).
    pub values: Vec<u8>,
}

/// SOS (Start Of Scan) header — which components participate in this scan and
/// which Huffman tables they use.
#[derive(Debug, Clone, Copy, Default)]
pub struct JpegScanHeader {
    /// Number of components in the scan (1..=4).
    pub component_count: u8,
    /// Per-scan-component selectors (only the first `component_count` are valid).
    pub components: [JpegScanComponent; 4],
}

/// One component within an SOS header.
#[derive(Debug, Clone, Copy, Default)]
pub struct JpegScanComponent {
    /// Component identifier matching one of the SOF frame components.
    pub id: u8,
    /// DC Huffman table selector (0 or 1 in baseline JPEG).
    pub dc_table: u8,
    /// AC Huffman table selector (0 or 1 in baseline JPEG).
    pub ac_table: u8,
}

// ── Errors ────────────────────────────────────────────────────────────────────

/// Errors emitted by [`JpegHeaders::parse`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JpegHeaderError {
    /// Stream did not begin with `0xFF 0xD8`.
    MissingSoi,
    /// A marker prefix `0xFF` was expected but not found at the parser cursor.
    ExpectedMarker,
    /// A length-prefixed segment claimed a length that overruns the input.
    SegmentOverrun {
        /// Marker byte (e.g. 0xC0 for SOF0).
        marker: u8,
    },
    /// A segment's claimed length is < 2 (length field includes itself).
    SegmentTooShort {
        /// Marker byte.
        marker: u8,
    },
    /// SOF0 was not encountered before SOS, or `width`/`height`/`components` are zero.
    MissingSof0,
    /// SOF0 specified an unsupported precision (we only handle 8-bit baseline).
    UnsupportedPrecision {
        /// Precision in bits as declared in the SOF0 segment.
        precision: u8,
    },
    /// SOF0 declared > 4 components.
    TooManyComponents {
        /// Component count from the SOF0 segment.
        count: u8,
    },
    /// DQT or DHT used a table ID outside the valid range.
    BadTableId {
        /// Marker byte (0xDB or 0xC4).
        marker: u8,
        /// Out-of-range table identifier.
        table_id: u8,
    },
    /// DHT class byte had a class field outside `{0, 1}`.
    BadDhtClass {
        /// Class field value (high nibble of the class+id byte).
        class: u8,
    },
    /// DHT DC table claimed > 12 entries (JPEG baseline limit).
    DcTableTooLarge {
        /// Total entry count.
        total: usize,
    },
    /// DHT AC table claimed > 162 entries (JPEG baseline limit).
    AcTableTooLarge {
        /// Total entry count.
        total: usize,
    },
    /// DRI segment was not exactly 4 bytes.
    BadDriLength {
        /// Declared segment length.
        length: usize,
    },
    /// SOS header declared `component_count > 4` or did not fit in its segment.
    BadSosHeader,
    /// SOS specified a Huffman table selector outside `{0, 1}`.
    BadSosTableSelector {
        /// DC or AC selector.
        class: DhtClass,
        /// Out-of-range selector value.
        selector: u8,
    },
    /// Stream ended before SOS.
    Truncated,
    /// Integer arithmetic overflowed during parsing — input is malformed.
    Overflow,
}

impl fmt::Display for JpegHeaderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingSoi => write!(f, "JPEG SOI (0xFF 0xD8) not found at start of stream"),
            Self::ExpectedMarker => write!(f, "expected JPEG marker prefix 0xFF"),
            Self::SegmentOverrun { marker } => {
                write!(
                    f,
                    "segment 0x{marker:02X} declares length past end of input"
                )
            }
            Self::SegmentTooShort { marker } => {
                write!(f, "segment 0x{marker:02X} length < 2 bytes")
            }
            Self::MissingSof0 => write!(f, "SOF0 baseline frame header not found before SOS"),
            Self::UnsupportedPrecision { precision } => {
                write!(
                    f,
                    "SOF0 sample precision {precision} not supported (need 8)"
                )
            }
            Self::TooManyComponents { count } => {
                write!(f, "SOF0 declares {count} components (max 4)")
            }
            Self::BadTableId { marker, table_id } => {
                write!(f, "segment 0x{marker:02X} table_id {table_id} out of range",)
            }
            Self::BadDhtClass { class } => {
                write!(f, "DHT class {class} not in {{0=DC, 1=AC}}")
            }
            Self::DcTableTooLarge { total } => {
                write!(f, "DHT DC table has {total} entries (max 12)")
            }
            Self::AcTableTooLarge { total } => {
                write!(f, "DHT AC table has {total} entries (max 162)")
            }
            Self::BadDriLength { length } => {
                write!(f, "DRI segment length {length} != 4")
            }
            Self::BadSosHeader => write!(
                f,
                "SOS header malformed or component table overruns segment"
            ),
            Self::BadSosTableSelector { class, selector } => {
                let kind = match class {
                    DhtClass::Dc => "DC",
                    DhtClass::Ac => "AC",
                };
                write!(f, "SOS {kind} table selector {selector} not in {{0, 1}}")
            }
            Self::Truncated => write!(f, "JPEG stream truncated before SOS"),
            Self::Overflow => write!(f, "integer overflow during JPEG header parse"),
        }
    }
}

impl Error for JpegHeaderError {}

// ── Parser ────────────────────────────────────────────────────────────────────

impl<'a> JpegHeaders<'a> {
    /// Parse a JPEG bitstream and extract every header up to (and including) SOS.
    ///
    /// `scan_data` in the returned headers is a borrow of `data` covering the
    /// entropy-coded segment, byte-stuffing intact. Subsequent decoding stages
    /// must call [`super::unstuff::unstuff_into`] before bit-walking it.
    ///
    /// # Errors
    ///
    /// Returns a [`JpegHeaderError`] variant describing the first malformed
    /// or unsupported segment encountered. The parser is fail-fast: it does
    /// not attempt to recover from a bad segment.
    pub fn parse(data: &'a [u8]) -> Result<Self, JpegHeaderError> {
        let mut cursor = ByteCursor::new(data);

        // SOI.
        if cursor.read_bytes(2)? != [0xFF, 0xD8] {
            return Err(JpegHeaderError::MissingSoi);
        }

        let mut width: u16 = 0;
        let mut height: u16 = 0;
        let mut components: u8 = 0;
        let mut frame_components = [JpegFrameComponent::default(); 4];
        let mut quant_tables = [JpegQuantTable::default(); 4];
        let mut quant_present = [false; 4];
        let mut huffman_tables: Vec<JpegHuffmanTable> = Vec::with_capacity(4);
        let mut scan = JpegScanHeader::default();
        let mut restart_interval: u16 = 0;
        let mut scan_data_offset: usize = 0;
        let mut scan_data_size: usize = 0;
        let mut hit_sos = false;
        let mut truncated = false;

        loop {
            if cursor.remaining() < 2 {
                truncated = true;
                break;
            }
            if cursor.peek(0) != 0xFF {
                return Err(JpegHeaderError::ExpectedMarker);
            }
            let marker = cursor.peek(1);
            cursor.advance(2);

            match marker {
                // RST0–RST7 / SOI re-encounter: zero-payload markers.
                0xD0..=0xD8 => {}

                // SOF0 — baseline DCT 8-bit.
                0xC0 => {
                    let body = read_segment(&mut cursor, marker)?;
                    parse_sof0(
                        body,
                        &mut width,
                        &mut height,
                        &mut components,
                        &mut frame_components,
                    )?;
                }

                // DQT — quantisation tables.
                0xDB => {
                    let body = read_segment(&mut cursor, marker)?;
                    parse_dqt(body, &mut quant_tables, &mut quant_present)?;
                }

                // DHT — Huffman tables.
                0xC4 => {
                    let body = read_segment(&mut cursor, marker)?;
                    parse_dht(body, &mut huffman_tables)?;
                }

                // DRI — define restart interval.
                0xDD => {
                    let body = read_segment(&mut cursor, marker)?;
                    if body.len() != 2 {
                        return Err(JpegHeaderError::BadDriLength {
                            length: body.len() + 2,
                        });
                    }
                    restart_interval = u16::from_be_bytes([body[0], body[1]]);
                }

                // SOS — start of scan: last header we need.
                0xDA => {
                    let body = read_segment(&mut cursor, marker)?;
                    parse_sos(body, &mut scan)?;
                    scan_data_offset = cursor.position();
                    scan_data_size = locate_scan_end(data, scan_data_offset);
                    hit_sos = true;
                    break;
                }

                // APP0–APP15, COM, etc. (0xE0–0xFE except already-handled): length-prefixed; skip.
                0xE0..=0xFE => {
                    let body = read_segment(&mut cursor, marker)?;
                    let _ = body;
                }

                // EOI or any unrecognised marker — stop parsing.
                _ => break,
            }
        }

        if truncated || !hit_sos {
            return Err(JpegHeaderError::Truncated);
        }
        if width == 0 || height == 0 || components == 0 {
            return Err(JpegHeaderError::MissingSof0);
        }

        let scan_data = &data[scan_data_offset..scan_data_offset + scan_data_size];

        Ok(Self {
            width,
            height,
            components,
            frame_components,
            quant_tables,
            quant_present,
            huffman_tables,
            scan,
            restart_interval,
            scan_data,
            scan_data_offset,
        })
    }

    /// Look up a Huffman table by class and table ID. Returns `None` if no
    /// matching table was declared in any DHT segment.
    #[must_use]
    pub fn huffman(&self, class: DhtClass, table_id: u8) -> Option<&JpegHuffmanTable> {
        self.huffman_tables
            .iter()
            .find(|t| t.class == class && t.table_id == table_id)
    }

    /// Number of MCUs (minimum coded units) in this image, computed from
    /// `components`, `width`/`height`, and the per-component sampling factors.
    #[must_use]
    pub fn num_mcus(&self) -> u32 {
        if self.components == 0 {
            return 0;
        }
        let comps = &self.frame_components[..self.components as usize];
        let max_h = comps.iter().map(|c| c.h_sampling).max().unwrap_or(1).max(1);
        let max_v = comps.iter().map(|c| c.v_sampling).max().unwrap_or(1).max(1);
        let mcu_w = u32::from(max_h) * 8;
        let mcu_h = u32::from(max_v) * 8;
        let w = u32::from(self.width);
        let h = u32::from(self.height);
        w.div_ceil(mcu_w) * h.div_ceil(mcu_h)
    }
}

// ── Cursor + segment helpers ──────────────────────────────────────────────────

struct ByteCursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> ByteCursor<'a> {
    const fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    const fn remaining(&self) -> usize {
        self.data.len() - self.pos
    }

    const fn position(&self) -> usize {
        self.pos
    }

    fn peek(&self, offset: usize) -> u8 {
        self.data[self.pos + offset]
    }

    fn advance(&mut self, n: usize) {
        self.pos += n;
    }

    fn read_bytes(&mut self, n: usize) -> Result<&'a [u8], JpegHeaderError> {
        let end = self.pos.checked_add(n).ok_or(JpegHeaderError::Overflow)?;
        if end > self.data.len() {
            return Err(JpegHeaderError::Truncated);
        }
        let slice = &self.data[self.pos..end];
        self.pos = end;
        Ok(slice)
    }

    fn read_u16(&mut self) -> Result<u16, JpegHeaderError> {
        let b = self.read_bytes(2)?;
        Ok(u16::from_be_bytes([b[0], b[1]]))
    }
}

/// Read a length-prefixed segment body.  The length field is part of the
/// declared length, so the body is `seg_len - 2` bytes.
fn read_segment<'a>(cursor: &mut ByteCursor<'a>, marker: u8) -> Result<&'a [u8], JpegHeaderError> {
    let seg_len = cursor.read_u16()? as usize;
    if seg_len < 2 {
        return Err(JpegHeaderError::SegmentTooShort { marker });
    }
    let body_len = seg_len - 2;
    cursor.read_bytes(body_len).map_err(|e| match e {
        JpegHeaderError::Truncated => JpegHeaderError::SegmentOverrun { marker },
        other => other,
    })
}

fn parse_sof0(
    body: &[u8],
    width: &mut u16,
    height: &mut u16,
    components: &mut u8,
    frame_components: &mut [JpegFrameComponent; 4],
) -> Result<(), JpegHeaderError> {
    if body.len() < 6 {
        return Err(JpegHeaderError::SegmentTooShort { marker: 0xC0 });
    }
    if body[0] != 8 {
        return Err(JpegHeaderError::UnsupportedPrecision { precision: body[0] });
    }
    *height = u16::from_be_bytes([body[1], body[2]]);
    *width = u16::from_be_bytes([body[3], body[4]]);
    let nf = body[5];
    if nf > 4 {
        return Err(JpegHeaderError::TooManyComponents { count: nf });
    }
    if (nf as usize) * 3 + 6 > body.len() {
        return Err(JpegHeaderError::SegmentOverrun { marker: 0xC0 });
    }
    *components = nf;
    for i in 0..nf as usize {
        let off = 6 + i * 3;
        let quant_selector = body[off + 2];
        if quant_selector >= 4 {
            return Err(JpegHeaderError::BadTableId {
                marker: 0xC0,
                table_id: quant_selector,
            });
        }
        frame_components[i] = JpegFrameComponent {
            id: body[off],
            h_sampling: body[off + 1] >> 4,
            v_sampling: body[off + 1] & 0x0F,
            quant_selector,
        };
    }
    Ok(())
}

fn parse_dqt(
    body: &[u8],
    quant_tables: &mut [JpegQuantTable; 4],
    quant_present: &mut [bool; 4],
) -> Result<(), JpegHeaderError> {
    let mut off = 0usize;
    while off < body.len() {
        let id_prec = body[off];
        off += 1;
        let prec = id_prec >> 4;
        let table_id_u8 = id_prec & 0x0F;
        if table_id_u8 >= 4 {
            return Err(JpegHeaderError::BadTableId {
                marker: 0xDB,
                table_id: table_id_u8,
            });
        }
        let table_id = table_id_u8 as usize;
        if prec == 0 {
            if off + 64 > body.len() {
                return Err(JpegHeaderError::SegmentOverrun { marker: 0xDB });
            }
            quant_tables[table_id]
                .values
                .copy_from_slice(&body[off..off + 64]);
            off += 64;
        } else {
            // 16-bit quantisers — clamp each big-endian u16 to u8 for downstream
            // consumers that only support 8-bit. Documented widening contract.
            if off + 128 > body.len() {
                return Err(JpegHeaderError::SegmentOverrun { marker: 0xDB });
            }
            for k in 0..64 {
                let val = u16::from_be_bytes([body[off + k * 2], body[off + k * 2 + 1]]);
                // val.min(255) constrains to u8 range; cast is lossless.
                quant_tables[table_id].values[k] = val.min(255) as u8;
            }
            off += 128;
        }
        quant_present[table_id] = true;
    }
    Ok(())
}

fn parse_dht(
    body: &[u8],
    huffman_tables: &mut Vec<JpegHuffmanTable>,
) -> Result<(), JpegHeaderError> {
    let mut off = 0usize;
    while off < body.len() {
        if off + 17 > body.len() {
            return Err(JpegHeaderError::SegmentOverrun { marker: 0xC4 });
        }
        let tc_th = body[off];
        off += 1;
        let tc = (tc_th >> 4) & 0x0F;
        let th = tc_th & 0x0F;
        let class = match tc {
            0 => DhtClass::Dc,
            1 => DhtClass::Ac,
            other => return Err(JpegHeaderError::BadDhtClass { class: other }),
        };
        if th >= 4 {
            return Err(JpegHeaderError::BadTableId {
                marker: 0xC4,
                table_id: th,
            });
        }

        let mut num_codes = [0u8; 16];
        num_codes.copy_from_slice(&body[off..off + 16]);
        off += 16;
        let total: usize = num_codes.iter().map(|&n| n as usize).sum();

        match class {
            DhtClass::Dc if total > 12 => {
                return Err(JpegHeaderError::DcTableTooLarge { total });
            }
            DhtClass::Ac if total > 162 => {
                return Err(JpegHeaderError::AcTableTooLarge { total });
            }
            _ => {}
        }
        if off + total > body.len() {
            return Err(JpegHeaderError::SegmentOverrun { marker: 0xC4 });
        }

        let mut values = Vec::with_capacity(total);
        values.extend_from_slice(&body[off..off + total]);
        off += total;

        // De-duplicate: a JPEG may redefine a table; keep the last definition.
        if let Some(slot) = huffman_tables
            .iter_mut()
            .find(|t| t.class == class && t.table_id == th)
        {
            slot.num_codes = num_codes;
            slot.values = values;
        } else {
            huffman_tables.push(JpegHuffmanTable {
                class,
                table_id: th,
                num_codes,
                values,
            });
        }
    }
    Ok(())
}

fn parse_sos(body: &[u8], scan: &mut JpegScanHeader) -> Result<(), JpegHeaderError> {
    if body.is_empty() {
        return Err(JpegHeaderError::BadSosHeader);
    }
    let nc = body[0];
    if nc == 0 || nc > 4 {
        return Err(JpegHeaderError::BadSosHeader);
    }
    if (nc as usize) * 2 + 4 > body.len() {
        return Err(JpegHeaderError::BadSosHeader);
    }
    scan.component_count = nc;
    for i in 0..nc as usize {
        let cs = body[1 + i * 2];
        let td_ta = body[2 + i * 2];
        let dc_table = td_ta >> 4;
        let ac_table = td_ta & 0x0F;
        if dc_table > 1 {
            return Err(JpegHeaderError::BadSosTableSelector {
                class: DhtClass::Dc,
                selector: dc_table,
            });
        }
        if ac_table > 1 {
            return Err(JpegHeaderError::BadSosTableSelector {
                class: DhtClass::Ac,
                selector: ac_table,
            });
        }
        scan.components[i] = JpegScanComponent {
            id: cs,
            dc_table,
            ac_table,
        };
    }
    // The trailing 3 bytes (Ss, Se, Ah/Al) are spectral-selection / successive-
    // approximation parameters used only by progressive JPEGs.  Baseline scans
    // always carry `00 3F 00`; we do not validate them, since SOF0 detection
    // upstream guarantees baseline.
    Ok(())
}

/// Locate the byte just past the entropy-coded segment, given the offset where
/// it begins. The scan ends at the next non-RST marker (typically EOI =
/// `0xFF 0xD9`) or, defensively, at end-of-input.
fn locate_scan_end(data: &[u8], start: usize) -> usize {
    let tail = &data[start..];
    let mut i = 0;
    while i + 1 < tail.len() {
        if tail[i] == 0xFF {
            let nxt = tail[i + 1];
            // 0x00 = byte-stuffing, RST0..7 = inline restart markers, FF = fill byte.
            if nxt == 0x00 || (0xD0..=0xD7).contains(&nxt) || nxt == 0xFF {
                i += 2;
                continue;
            }
            // Any other marker terminates the scan.
            return i;
        }
        i += 1;
    }
    tail.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 16×16 grayscale baseline JPEG — matches the fixture used by the
    /// VA-API parser tests so we have a known-good input.
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
    fn parse_gray_16x16_basic() {
        let h = JpegHeaders::parse(GRAY_16X16_JPEG).expect("parse failed");
        assert_eq!(h.width, 16);
        assert_eq!(h.height, 16);
        assert_eq!(h.components, 1);
        assert!(h.quant_present[0]);
        assert_eq!(h.scan.component_count, 1);
        assert_eq!(h.scan.components[0].id, 1);
        assert!(!h.scan_data.is_empty());
        assert_eq!(h.restart_interval, 0);
    }

    #[test]
    fn parse_extracts_huffman_tables() {
        let h = JpegHeaders::parse(GRAY_16X16_JPEG).expect("parse failed");
        let dc = h.huffman(DhtClass::Dc, 0).expect("DC luma table missing");
        let ac = h.huffman(DhtClass::Ac, 0).expect("AC luma table missing");
        // From the fixture: DC num_codes [0,1,1,0,...] → 2 entries; AC has
        // num_codes [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] (no codes!) but the
        // body still has 1 trailing value byte 0x00, so total_codes = 0 and
        // values.len() = 0 — actual fixture has one DC and zero AC codes
        // depending on parse, but verifying both tables exist with values
        // matching their declared sums is enough.
        let dc_total: usize = dc.num_codes.iter().map(|&n| n as usize).sum();
        let ac_total: usize = ac.num_codes.iter().map(|&n| n as usize).sum();
        assert_eq!(dc.values.len(), dc_total);
        assert_eq!(ac.values.len(), ac_total);
    }

    #[test]
    fn parse_extracts_quant_table() {
        let h = JpegHeaders::parse(GRAY_16X16_JPEG).expect("parse failed");
        // First three zigzag values from the DQT in the fixture.
        assert_eq!(h.quant_tables[0].values[0], 0x06);
        assert_eq!(h.quant_tables[0].values[1], 0x04);
        assert_eq!(h.quant_tables[0].values[2], 0x05);
    }

    #[test]
    fn parse_empty_returns_truncated() {
        let err = JpegHeaders::parse(&[]).expect_err("must fail");
        assert_eq!(err, JpegHeaderError::Truncated);
    }

    #[test]
    fn parse_only_soi_returns_truncated() {
        let err = JpegHeaders::parse(&[0xFF, 0xD8]).expect_err("must fail");
        assert_eq!(err, JpegHeaderError::Truncated);
    }

    #[test]
    fn parse_missing_soi_returns_missing_soi() {
        let err = JpegHeaders::parse(&[0x00, 0x00, 0xFF, 0xD8]).expect_err("must fail");
        assert_eq!(err, JpegHeaderError::MissingSoi);
    }

    #[test]
    fn parse_progressive_minimal_returns_missing_sof0() {
        // SOF2 (progressive) instead of SOF0 — parser must not parse it as a
        // frame header. The whole-stream walk eventually hits truncation
        // because no SOS appears with valid framing; either result is
        // acceptable, but it must NOT silently succeed.
        let progressive: &[u8] = &[
            0xFF, 0xD8, // SOI
            0xFF, 0xC2, // SOF2 (progressive)
            0x00, 0x11, // length 17
            0x08, 0x00, 0x10, 0x00, 0x10, 0x03, // precision + dims + components
            0x01, 0x11, 0x00, // comp 1
            0x02, 0x11, 0x01, // comp 2
            0x03, 0x11, 0x01, // comp 3
            0xFF, 0xD9, // EOI
        ];
        let err = JpegHeaders::parse(progressive).expect_err("must fail");
        // Most likely outcome: SOF0 was never seen, so SOS-or-truncation path
        // hits MissingSof0 / Truncated.
        assert!(matches!(
            err,
            JpegHeaderError::MissingSof0 | JpegHeaderError::Truncated,
        ));
    }

    #[test]
    fn num_mcus_grayscale_16x16_is_4() {
        let h = JpegHeaders::parse(GRAY_16X16_JPEG).unwrap();
        // 16x16 with 1:1 sampling → 8x8 MCUs → 2×2 = 4 MCUs.
        assert_eq!(h.num_mcus(), 4);
    }
}
