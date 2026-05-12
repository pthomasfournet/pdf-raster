//! `/Linearized` (Fast Web View) hint-table detection.
//!
//! Per PDF 1.7 § F, a linearized PDF places a `/Linearized` dict as the
//! first indirect object in the file.  This dict + hint streams together
//! let a reader jump to any page by index without walking the page tree.
//!
//! This module supports the detection layer: parse the dict and expose the
//! `/N`, `/O`, and `/H` (hint stream offsets) values.  The Page Offset
//! Hint Table contents (PDF § F.4.5, bit-packed) are NOT parsed yet — see
//! `ROADMAP.md` for the hint-stream parser follow-up.  Until that ships,
//! callers detect linearization purely for telemetry / future fast paths
//! and continue using the logarithmic page-tree descent for all lookups.

use crate::{document::Document, error::PdfError, object::Object};

/// Parsed `/Linearized` parameters.  All fields are public because the
/// future hint-stream parser will read `hint_stream_offset` /
/// `hint_stream_length` directly from the cached struct.
#[derive(Debug, Clone, Copy)]
pub struct LinearizationHints {
    /// `/N` — total number of pages.
    pub page_count: u32,
    /// `/O` — object number of the first page (page 0).
    pub first_page_obj: u32,
    /// `/H[0]` — byte offset of the Page Offset Hint Table stream.
    pub hint_stream_offset: u64,
    /// `/H[1]` — length of the Page Offset Hint Table stream.
    pub hint_stream_length: u64,
}

impl LinearizationHints {
    /// Try to load linearization hints from the document.  Returns
    /// `Ok(None)` for non-linearized PDFs or when required fields are
    /// missing / malformed.  Returns `Ok(Some(_))` only when a valid
    /// `/Linearized` dict is present with the minimum mandatory keys
    /// (`/N`, `/O`, `/H[0]`, `/H[1]`).
    ///
    /// # Errors
    /// Currently never errors — every malformed-or-absent case collapses
    /// to `Ok(None)` so callers can transparently fall back to the
    /// page-tree descent.  Returned as `Result` so the signature stays
    /// stable when the hint-stream parser ships and gains real failure
    /// modes (truncated stream, bad bit-widths, …).
    pub fn try_load(doc: &Document) -> Result<Option<Self>, PdfError> {
        let Some(lin_obj) = find_linearized_object(doc)? else {
            return Ok(None);
        };
        let Some(lin_dict) = lin_obj.as_dict() else {
            return Ok(None);
        };
        let hints = parse_dict(lin_dict);
        if hints.is_none() {
            log::debug!(
                "LinearizationHints: /Linearized dict present but required keys missing; \
                 falling back to descent",
            );
        }
        Ok(hints)
    }
}

/// Object 1 of a linearized PDF is the `/Linearized` dict.  Probe it; any
/// failure to read or destructure that object collapses to `Ok(None)` —
/// non-linearized PDFs reach this code path and we don't surface their
/// (perfectly valid) absence as an error.
fn find_linearized_object(doc: &Document) -> Result<Option<std::sync::Arc<Object>>, PdfError> {
    let Ok(obj) = doc.get_object((1, 0)) else {
        return Ok(None);
    };
    let Some(dict) = obj.as_dict() else {
        return Ok(None);
    };
    if dict.contains_key(b"Linearized") {
        Ok(Some(obj))
    } else {
        Ok(None)
    }
}

/// Pure parsing of a `/Linearized` dict into `LinearizationHints`.
/// Returns `None` if any required key is missing or malformed.
fn parse_dict(lin_dict: &crate::dictionary::Dictionary) -> Option<LinearizationHints> {
    let page_count = lin_dict.get(b"N").and_then(Object::as_u32)?;
    let first_page_obj = lin_dict.get(b"O").and_then(Object::as_u32)?;

    // /H is an array of [offset, length, ...].  First pair is mandatory
    // (Page Offset Hint Table); subsequent pairs are optional.  Borrow the
    // array — we only read [0] and [1], no need to clone the Vec.
    let h_array = lin_dict.get(b"H").and_then(Object::as_array)?;
    let hint_stream_offset = h_array.first().and_then(Object::as_u64)?;
    let hint_stream_length = h_array.get(1).and_then(Object::as_u64)?;

    Some(LinearizationHints {
        page_count,
        first_page_obj,
        hint_stream_offset,
        hint_stream_length,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::Document;

    /// Minimal non-linearized fixture (one empty page).  Object 1 is the
    /// catalog, not a `/Linearized` dict, so detection must return `None`.
    fn non_linearized_pdf() -> Vec<u8> {
        // Offsets verified by per-section byte counting:
        // %PDF-1.4\n = 9 bytes  → obj1 at 9
        // "1 0 obj\n<</Type /Catalog /Pages 2 0 R>>\nendobj\n" = 47 bytes → obj2 at 56
        // "2 0 obj\n<</Type /Pages /Kids [3 0 R] /Count 1>>\nendobj\n" = 55 bytes → obj3 at 111
        // "3 0 obj\n<</Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]>>\nendobj\n" = 69 bytes → xref at 180
        b"%PDF-1.4\n\
1 0 obj\n<</Type /Catalog /Pages 2 0 R>>\nendobj\n\
2 0 obj\n<</Type /Pages /Kids [3 0 R] /Count 1>>\nendobj\n\
3 0 obj\n<</Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]>>\nendobj\n\
xref\n0 4\n\
0000000000 65535 f\r\n\
0000000009 00000 n\r\n\
0000000056 00000 n\r\n\
0000000111 00000 n\r\n\
trailer\n<</Size 4 /Root 1 0 R>>\n\
startxref\n180\n%%EOF"
            .to_vec()
    }

    #[test]
    fn non_linearized_returns_none() {
        let doc = Document::from_bytes_owned(non_linearized_pdf()).unwrap();
        let hints = LinearizationHints::try_load(&doc).expect("ok");
        assert!(hints.is_none(), "non-linearized PDF should return None");
    }

    /// Object 1 in a linearized PDF is the linearization dict, not the
    /// catalog.  Construct a synthetic PDF where object 1 has /Linearized
    /// plus the minimum required keys, and verify try_load returns Some.
    /// The rest of the document need not be a "valid" linearized PDF;
    /// `try_load` only inspects object 1.
    #[test]
    fn linearized_dict_in_object_1_is_detected() {
        // Per-segment byte counts:
        // "%PDF-1.4\n" = 9 bytes  → obj1 at 9
        // "1 0 obj\n<</Linearized 1 /N 100 /O 5 /H [12345 678]>>\nendobj\n" = 60 bytes → obj2 at 69
        // "2 0 obj\n<</Type /Catalog /Pages 3 0 R>>\nendobj\n" = 47 bytes → obj3 at 116
        // "3 0 obj\n<</Type /Pages /Kids [] /Count 0>>\nendobj\n" = 50 bytes → xref at 166
        let bytes = b"%PDF-1.4\n\
1 0 obj\n<</Linearized 1 /N 100 /O 5 /H [12345 678]>>\nendobj\n\
2 0 obj\n<</Type /Catalog /Pages 3 0 R>>\nendobj\n\
3 0 obj\n<</Type /Pages /Kids [] /Count 0>>\nendobj\n\
xref\n0 4\n\
0000000000 65535 f\r\n\
0000000009 00000 n\r\n\
0000000069 00000 n\r\n\
0000000116 00000 n\r\n\
trailer\n<</Size 4 /Root 2 0 R>>\n\
startxref\n166\n%%EOF"
            .to_vec();
        let doc = Document::from_bytes_owned(bytes).unwrap();
        let hints = LinearizationHints::try_load(&doc)
            .expect("ok")
            .expect("Some(hints)");
        assert_eq!(hints.page_count, 100);
        assert_eq!(hints.first_page_obj, 5);
        assert_eq!(hints.hint_stream_offset, 12345);
        assert_eq!(hints.hint_stream_length, 678);
    }

    /// `/Linearized` dict with the marker key present but a required field
    /// missing.  Detection must collapse to `Ok(None)` rather than producing
    /// a `LinearizationHints` with garbage values — the contract says
    /// "minimum mandatory keys present" before returning `Some`.
    #[test]
    fn malformed_linearized_dict_returns_none() {
        // Object 1 has /Linearized 1 but no /N, /O, or /H.
        // "%PDF-1.4\n" = 9 bytes  → obj1 at 9
        // "1 0 obj\n<</Linearized 1>>\nendobj\n" = 33 bytes → obj2 at 42
        // "2 0 obj\n<</Type /Catalog /Pages 3 0 R>>\nendobj\n" = 47 bytes → obj3 at 89
        // "3 0 obj\n<</Type /Pages /Kids [] /Count 0>>\nendobj\n" = 50 bytes → xref at 139
        let bytes = b"%PDF-1.4\n\
1 0 obj\n<</Linearized 1>>\nendobj\n\
2 0 obj\n<</Type /Catalog /Pages 3 0 R>>\nendobj\n\
3 0 obj\n<</Type /Pages /Kids [] /Count 0>>\nendobj\n\
xref\n0 4\n\
0000000000 65535 f\r\n\
0000000009 00000 n\r\n\
0000000042 00000 n\r\n\
0000000089 00000 n\r\n\
trailer\n<</Size 4 /Root 2 0 R>>\n\
startxref\n139\n%%EOF"
            .to_vec();
        let doc = Document::from_bytes_owned(bytes).unwrap();
        let hints = LinearizationHints::try_load(&doc).expect("ok");
        assert!(
            hints.is_none(),
            "malformed /Linearized dict should return None"
        );
    }
}
