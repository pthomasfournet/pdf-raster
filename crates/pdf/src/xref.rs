//! Cross-reference table parser.
//!
//! Reads the xref section(s) from the end of the file, follows `/Prev` chains
//! for incremental updates, and produces a flat `XrefTable` mapping object
//! numbers to their locations.  Handles all three formats:
//!
//! * Traditional ASCII xref tables (PDF ≤ 1.4)
//! * Binary cross-reference streams (PDF 1.5+, `/Type /XRef`)
//! * Hybrid files (ASCII table with `/XRefStm` pointer)

use std::collections::HashMap;

use crate::{
    dictionary::Dictionary,
    error::PdfError,
    lexer::{is_ws, parse_u64, skip_ws},
    object::{Object, parse_object},
};

/// Where to find an indirect object in the file.
#[derive(Debug, Clone, Copy)]
pub(crate) enum XrefEntry {
    /// Object lives at `offset` bytes from the start of the file.
    Direct {
        offset: u64,
        /// Generation number — stored but not used for rendering; reserved for
        /// future encryption / repair support.
        #[expect(dead_code, reason = "reserved for future encryption / repair support")]
        generation: u16,
    },
    /// Object is inside an object stream; `container` is the object number of
    /// the ObjStm, `index` is the 0-based position within it.
    InObjStm { container: u32, index: u32 },
}

/// Complete cross-reference table built from all xref sections in the file.
/// Later xref sections override earlier ones (incremental update semantics).
#[derive(Debug, Default)]
pub(crate) struct XrefTable {
    pub entries: HashMap<u32, XrefEntry>,
    /// The document trailer dictionary bytes.  We keep the raw bytes and parse
    /// on demand so this module does not depend on a full object graph.
    pub trailer: Dictionary,
}

impl XrefTable {
    pub(crate) fn get(&self, id: u32) -> Option<XrefEntry> {
        self.entries.get(&id).copied()
    }
}

/// Parse all xref sections starting from the `startxref` offset at the end of
/// the file.  Follows `/Prev` chains.
pub(crate) fn read_xref(data: &[u8]) -> Result<XrefTable, PdfError> {
    let start_offset = find_startxref(data)?;

    let mut table = XrefTable::default();
    // Track visited offsets to break infinite loops in corrupt /Prev chains.
    let mut visited = std::collections::HashSet::new();
    read_xref_at(data, start_offset, &mut table, &mut visited)?;
    Ok(table)
}

fn read_xref_at(
    data: &[u8],
    offset: u64,
    table: &mut XrefTable,
    visited: &mut std::collections::HashSet<u64>,
) -> Result<(), PdfError> {
    if !visited.insert(offset) {
        return Ok(()); // Cycle detected — stop.
    }
    let off = offset as usize;
    if off >= data.len() {
        return Err(PdfError::BadXref(format!(
            "offset {offset} beyond end of file"
        )));
    }

    // Detect format: "xref" keyword → traditional table; anything else must be
    // an indirect object (the xref stream).
    if data[off..].starts_with(b"xref") {
        read_traditional_xref(data, off, table, visited)
    } else {
        read_xref_stream(data, off, table, visited)
    }
}

// ── Traditional ASCII xref table ─────────────────────────────────────────────

fn read_traditional_xref(
    data: &[u8],
    start: usize,
    table: &mut XrefTable,
    visited: &mut std::collections::HashSet<u64>,
) -> Result<(), PdfError> {
    let mut pos = start + 4; // skip "xref"
    skip_ws(data, &mut pos);

    // Parse subsections: each is "<first_obj> <count>" followed by entries.
    while pos < data.len() {
        // Peek ahead: if we see "trailer" we're done with entries.
        if data[pos..].starts_with(b"trailer") {
            pos += 7;
            break;
        }

        let first_raw = parse_u64(data, &mut pos).ok_or_else(|| {
            PdfError::BadXref("expected object number in subsection header".into())
        })?;
        if first_raw > u64::from(u32::MAX) {
            return Err(PdfError::BadXref(format!(
                "subsection first object {first_raw} exceeds u32::MAX"
            )));
        }
        let first = first_raw as u32;
        skip_ws(data, &mut pos);
        let count = parse_u64(data, &mut pos)
            .ok_or_else(|| PdfError::BadXref("expected count in subsection header".into()))?;
        // Sanity cap: no real PDF has more than 10 M objects.
        if count > 10_000_000 {
            return Err(PdfError::BadXref(format!(
                "xref subsection count {count} exceeds sanity limit"
            )));
        }
        // Consume the line ending after the subsection header.
        while pos < data.len() && matches!(data[pos], b'\r' | b'\n') {
            pos += 1;
        }

        for i in 0..count {
            // Each entry is exactly 20 bytes: "nnnnnnnnnn ggggg n/f\r\n"
            // We are lenient and scan for the next token rather than fixed-stride.
            skip_ws(data, &mut pos);
            let offset = parse_u64(data, &mut pos)
                .ok_or_else(|| PdfError::BadXref(format!("bad entry offset at pos {pos}")))?;
            skip_ws(data, &mut pos);
            let generation = parse_u64(data, &mut pos)
                .ok_or_else(|| PdfError::BadXref(format!("bad generation at pos {pos}")))?
                as u16;
            skip_ws(data, &mut pos);
            let flag = data.get(pos).copied().unwrap_or(b'f');
            pos += 1;
            // Skip up to the end of this entry line (handles \r, \n, \r\n, space+\r etc.)
            while pos < data.len() && matches!(data[pos], b' ' | b'\r' | b'\n') {
                pos += 1;
            }

            let obj_num = first.checked_add(i as u32).ok_or_else(|| {
                PdfError::BadXref(format!("object number overflow: first={first} i={i}"))
            })?;
            // Only insert if not already present (newer xref sections win).
            if flag == b'n' && !table.entries.contains_key(&obj_num) {
                table
                    .entries
                    .insert(obj_num, XrefEntry::Direct { offset, generation });
            }
        }
    }

    // Parse the trailer dictionary.
    skip_ws(data, &mut pos);
    parse_trailer_dict(data, &mut pos, table, visited)
}

fn parse_trailer_dict(
    data: &[u8],
    pos: &mut usize,
    table: &mut XrefTable,
    visited: &mut std::collections::HashSet<u64>,
) -> Result<(), PdfError> {
    skip_ws(data, pos);
    // The trailer dict starts with "<<".
    if !data[*pos..].starts_with(b"<<") {
        return Err(PdfError::BadXref(format!(
            "expected '<<' for trailer dict at pos {pos}"
        )));
    }

    let obj = parse_object(data, pos)
        .ok_or_else(|| PdfError::BadXref("could not parse trailer dictionary".into()))?;
    let dict = match obj {
        Object::Dictionary(d) => d,
        other => {
            return Err(PdfError::BadXref(format!(
                "trailer is not a dict: {other:?}"
            )));
        }
    };

    merge_trailer_and_chain(data, &dict, table, visited)?;

    // Handle hybrid files: /XRefStm points to an additional xref stream.
    if let Some(Object::Integer(xrefstm)) = dict.get(b"XRefStm")
        && *xrefstm > 0
    {
        let stm_off = *xrefstm as u64;
        if !visited.contains(&stm_off) {
            read_xref_at(data, stm_off, table, visited)?;
        }
    }

    Ok(())
}

/// Merge `dict` into `table.trailer` (existing keys win — newer xref sections
/// already populated the table) and recurse into the `/Prev` chain if present
/// and positive.  Negative or zero `/Prev` values are silently ignored — they
/// indicate "no previous xref section" rather than an error.
fn merge_trailer_and_chain(
    data: &[u8],
    dict: &Dictionary,
    table: &mut XrefTable,
    visited: &mut std::collections::HashSet<u64>,
) -> Result<(), PdfError> {
    for (k, v) in dict {
        if !table.trailer.contains_key(k) {
            table.trailer.insert(k.clone(), v.clone());
        }
    }
    if let Some(Object::Integer(prev)) = dict.get(b"Prev")
        && *prev > 0
    {
        read_xref_at(data, *prev as u64, table, visited)?;
    }
    Ok(())
}

// ── Cross-reference stream (PDF 1.5+) ────────────────────────────────────────

fn read_xref_stream(
    data: &[u8],
    start: usize,
    table: &mut XrefTable,
    visited: &mut std::collections::HashSet<u64>,
) -> Result<(), PdfError> {
    // Parse the indirect object header: "<id> <gen> obj"
    let mut pos = start;
    skip_ws(data, &mut pos);
    parse_u64(data, &mut pos); // object number — discard
    skip_ws(data, &mut pos);
    parse_u64(data, &mut pos); // generation — discard
    skip_ws(data, &mut pos);
    if !data[pos..].starts_with(b"obj") {
        return Err(PdfError::BadXref(format!(
            "expected 'obj' keyword at offset {start}"
        )));
    }
    pos += 3;
    skip_ws(data, &mut pos);

    // parse_object returns Object::Stream when it finds the "stream" keyword.
    let dict_obj = parse_object(data, &mut pos)
        .ok_or_else(|| PdfError::BadXref("could not parse xref stream dict".into()))?;
    let (dict, raw_stream) = match dict_obj {
        Object::Stream(s) => (s.dict, s.content),
        _ => {
            return Err(PdfError::BadXref(
                "xref stream object is not a Stream".into(),
            ));
        }
    };
    let raw_stream = &raw_stream[..];

    // Decode the stream (almost always FlateDecode).
    let stream_bytes = decode_xref_stream(raw_stream, &dict)?;

    // Read /W field widths.
    let w = match dict.get(b"W") {
        Some(Object::Array(a)) => a.clone(),
        _ => return Err(PdfError::BadXref("xref stream: missing /W array".into())),
    };
    let w: Vec<usize> = w
        .iter()
        .map(|o| match o {
            Object::Integer(n) => *n as usize,
            _ => 0,
        })
        .collect();
    if w.len() < 3 {
        return Err(PdfError::BadXref("xref stream: /W needs 3 entries".into()));
    }
    let (w0, w1, w2) = (w[0], w[1], w[2]);
    let entry_len = w0 + w1 + w2;
    if entry_len == 0 {
        return Err(PdfError::BadXref("xref stream: zero-width entry".into()));
    }

    // Determine which object numbers this stream covers via /Index.
    let index_pairs: Vec<u32> = match dict.get(b"Index") {
        Some(Object::Array(a)) => a
            .iter()
            .map(|o| obj_to_u32(o, "/Index"))
            .collect::<Result<Vec<_>, _>>()?,
        _ => {
            // Default: one subsection starting at 0, covering /Size objects.
            let size = dict
                .get(b"Size")
                .ok_or_else(|| PdfError::BadXref("xref stream: missing /Size".into()))?;
            vec![0, obj_to_u32(size, "/Size")?]
        }
    };
    if !index_pairs.len().is_multiple_of(2) {
        return Err(PdfError::BadXref(
            "xref stream: /Index must have even number of entries".into(),
        ));
    }

    // Decode binary entries.
    let mut byte_pos = 0usize;
    let mut pair_idx = 0;
    while pair_idx + 1 < index_pairs.len() {
        let first_obj = index_pairs[pair_idx];
        let count = index_pairs[pair_idx + 1];
        pair_idx += 2;

        // Sanity cap to defeat malformed PDFs claiming billions of entries.
        if count > 10_000_000 {
            return Err(PdfError::BadXref(format!(
                "xref stream subsection count {count} exceeds sanity limit"
            )));
        }

        for i in 0..count {
            if byte_pos + entry_len > stream_bytes.len() {
                break;
            }

            let type_val = if w0 == 0 {
                1 // default type is 1 per spec
            } else {
                read_be_uint(&stream_bytes[byte_pos..byte_pos + w0])
            };
            let f1 = read_be_uint(&stream_bytes[byte_pos + w0..byte_pos + w0 + w1]);
            let f2 = if w2 == 0 {
                0
            } else {
                read_be_uint(&stream_bytes[byte_pos + w0 + w1..byte_pos + w0 + w1 + w2])
            };
            byte_pos += entry_len;

            let Some(obj_num) = first_obj.checked_add(i) else {
                break; // object number overflow — stop processing this subsection
            };
            if table.entries.contains_key(&obj_num) {
                continue; // newer xref section already has this
            }

            match type_val {
                0 => {} // free object — skip
                1 => {
                    table.entries.insert(
                        obj_num,
                        XrefEntry::Direct {
                            offset: f1,
                            generation: f2 as u16,
                        },
                    );
                }
                2 => {
                    table.entries.insert(
                        obj_num,
                        XrefEntry::InObjStm {
                            container: f1 as u32,
                            index: f2 as u32,
                        },
                    );
                }
                _ => {} // unknown type — ignore per spec
            }
        }
    }

    merge_trailer_and_chain(data, &dict, table, visited)
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Decode an xref stream's raw bytes.  Almost always FlateDecode; may be
/// uncompressed (no /Filter key).
fn decode_xref_stream(raw: &[u8], dict: &Dictionary) -> Result<Vec<u8>, PdfError> {
    use crate::stream::apply_flate;

    // /Filter is optional, a Name, or a single-element array of a Name.
    let filter: Option<&[u8]> = match dict.get(b"Filter") {
        None => None,
        Some(Object::Name(n)) => Some(n),
        Some(Object::Array(a)) if a.len() == 1 => a[0].as_name(),
        _ => None,
    };

    match filter {
        None => Ok(raw.to_vec()),
        Some(b"FlateDecode") => {
            let params = dict
                .get(b"DecodeParms")
                .map(|o| o as &dyn crate::stream::DictLookup);
            apply_flate(raw, params).map_err(|e| PdfError::DecodeFailed(e.to_string()))
        }
        Some(other) => Err(PdfError::DecodeFailed(format!(
            "unsupported xref stream filter: {}",
            String::from_utf8_lossy(other)
        ))),
    }
}

/// Convert an `Object::Integer` to a non-negative `u32`, rejecting negatives,
/// non-integers, and values that would overflow.  `label` is interpolated into
/// the error message to identify which dict key was being read.
fn obj_to_u32(obj: &Object, label: &str) -> Result<u32, PdfError> {
    match obj {
        Object::Integer(n) if *n >= 0 && *n <= i64::from(u32::MAX) => Ok(*n as u32),
        other => Err(PdfError::BadXref(format!(
            "xref stream {label}: bad integer {other:?}"
        ))),
    }
}

/// Read a big-endian unsigned integer from `bytes` (1–8 bytes).
fn read_be_uint(bytes: &[u8]) -> u64 {
    let mut val = 0u64;
    for &b in bytes {
        val = (val << 8) | u64::from(b);
    }
    val
}

/// Find the byte offset stored after the last `startxref` keyword in the file.
/// Scans the final 2048 bytes to handle files with trailing garbage.
fn find_startxref(data: &[u8]) -> Result<u64, PdfError> {
    let scan_start = data.len().saturating_sub(2048);
    let tail = &data[scan_start..];

    // Find the last occurrence of "startxref".
    let needle = b"startxref";
    let rel_pos = tail
        .windows(needle.len())
        .rposition(|w| w == needle)
        .ok_or_else(|| PdfError::BadXref("'startxref' not found in last 2048 bytes".into()))?;

    let mut pos = scan_start + rel_pos + needle.len();
    // Skip whitespace (should be a single newline).
    while pos < data.len() && is_ws(data[pos]) {
        pos += 1;
    }
    parse_u64(data, &mut pos)
        .ok_or_else(|| PdfError::BadXref("no integer after 'startxref'".into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn find_startxref_basic() {
        let data = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\nxref\n0 1\n0000000009 65535 f\r\ntrailer\n<<>>\nstartxref\n9\n%%EOF";
        let off = find_startxref(data).unwrap();
        assert_eq!(off, 9);
    }

    #[test]
    fn read_be_uint_cases() {
        assert_eq!(read_be_uint(&[0x00, 0x01, 0x23]), 0x0123);
        assert_eq!(read_be_uint(&[0xFF]), 0xFF);
        assert_eq!(read_be_uint(&[]), 0);
    }

    #[test]
    fn xref_subsection_count_capped() {
        // count = 99_999_999 exceeds the 10 M sanity limit and must error,
        // not allocate ~2 GB of HashMap entries.
        let data = b"\
%PDF-1.4
xref
0 99999999
0000000000 65535 f\r
trailer
<<>>
startxref
9
%%EOF";
        let err = read_xref(data).unwrap_err();
        match err {
            PdfError::BadXref(s) => assert!(s.contains("sanity limit"), "got: {s}"),
            other => panic!("expected BadXref, got {other:?}"),
        }
    }

    #[test]
    fn xref_negative_prev_does_not_recurse() {
        // /Prev = -1 used to be cast as u64 = 0xFFFF_FFFF_FFFF_FFFF and trigger
        // a "beyond end of file" error from a bogus offset; now it is skipped.
        let data = b"\
%PDF-1.4
xref
0 1
0000000000 65535 f\r
trailer
<</Size 1 /Prev -1>>
startxref
9
%%EOF";
        // Should succeed: /Prev=-1 is treated as no previous xref.
        let table = read_xref(data).expect("xref parse");
        assert_eq!(table.entries.len(), 0);
    }
}
