//! `/Linearized` (Fast Web View) hint-table detection.
//!
//! Per PDF 1.7 § F, a linearized PDF places a `/Linearized` dict as the
//! first indirect object in the file.  This dict + hint streams together
//! let a reader jump to any page by index without walking the page tree.
//!
//! This module supports the detection layer: parse the dict, expose the
//! `/N`, `/O`, and `/H` (hint stream offsets) values, and stub
//! `page_offset(idx) -> None` as a placeholder.  The Page Offset Hint
//! Table contents (PDF § F.4.5, bit-packed) are NOT parsed yet — see
//! ROADMAP.md for the hint-stream parser follow-up.
//!
//! When `page_offset(idx)` returns `None`, callers fall back to the
//! logarithmic page-tree descent (`descend_to_page_index`).

use crate::{dictionary::Dictionary, document::Document, error::PdfError, object::Object};

/// Parsed `/Linearized` parameters.  The Page Offset Hint Table is
/// detected (see `hint_stream_offset` / `hint_stream_length`) but
/// not yet decoded.
#[derive(Debug, Clone)]
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
    /// missing.  Returns `Ok(Some(_))` when a valid `/Linearized` dict
    /// is present with the minimum mandatory keys.
    ///
    /// # Errors
    /// Currently never errors — every malformed-or-absent case collapses
    /// to `Ok(None)` so callers can transparently fall back to the
    /// page-tree descent.  Returned as `Result` so the signature stays
    /// stable when the hint-stream parser ships and gains real failure
    /// modes (truncated stream, bad bit-widths, …).
    pub fn try_load(doc: &Document) -> Result<Option<Self>, PdfError> {
        let Some(lin_dict) = find_linearized_dict(doc)? else {
            return Ok(None);
        };

        let Some(page_count) = lin_dict
            .get(b"N")
            .and_then(Object::as_i64)
            .and_then(|n| u32::try_from(n).ok())
        else {
            return Ok(None);
        };
        let Some(first_page_obj) = lin_dict
            .get(b"O")
            .and_then(Object::as_i64)
            .and_then(|n| u32::try_from(n).ok())
        else {
            return Ok(None);
        };

        // /H is an array of [offset, length, ...].  First pair is mandatory
        // (Page Offset Hint Table).  Subsequent pairs are optional.
        let h_array = match lin_dict.get(b"H") {
            Some(Object::Array(a)) if a.len() >= 2 => a.clone(),
            _ => return Ok(None),
        };
        let Some(hint_stream_offset) = h_array
            .first()
            .and_then(Object::as_i64)
            .and_then(|n| u64::try_from(n).ok())
        else {
            return Ok(None);
        };
        let Some(hint_stream_length) = h_array
            .get(1)
            .and_then(Object::as_i64)
            .and_then(|n| u64::try_from(n).ok())
        else {
            return Ok(None);
        };

        Ok(Some(Self {
            page_count,
            first_page_obj,
            hint_stream_offset,
            hint_stream_length,
        }))
    }

    /// Byte offset of page `idx`'s page object.  Currently always returns
    /// `None` because the bit-packed Page Offset Hint Table parser is
    /// pending.  Callers MUST fall back to the logarithmic page-tree
    /// descent when this returns `None`.
    ///
    /// This is a deliberate stub: the spec for the hint-table layout
    /// (PDF 1.7 § F.4.5) packs fields at variable bit widths, and a
    /// half-correct parser would silently misdirect page lookups.  See
    /// ROADMAP.md for the follow-up that wires in a real implementation.
    #[must_use]
    pub fn page_offset(&self, _idx: u32) -> Option<u64> {
        // Hint-stream bit-parser pending; see ROADMAP.md.
        None
    }
}

/// Probe object 1; if its dict has a `/Linearized` key, return the dict.
/// Any failure to read or destructure object 1 collapses to `Ok(None)` —
/// non-linearized PDFs reach this code path and we don't want to surface
/// their (perfectly valid) absence as an error.
fn find_linearized_dict(doc: &Document) -> Result<Option<Dictionary>, PdfError> {
    let Ok(obj) = doc.get_object((1, 0)) else {
        return Ok(None);
    };
    let Some(dict) = obj.as_dict() else {
        return Ok(None);
    };
    if dict.contains_key(b"Linearized") {
        Ok(Some(dict.clone()))
    } else {
        Ok(None)
    }
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
    /// + the minimum required keys, and verify try_load returns Some.
    /// We don't need the rest of the document to be a "valid" linearized
    /// PDF — try_load only inspects object 1.
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
        // page_offset stub always returns None for now.
        assert_eq!(hints.page_offset(0), None);
    }
}
