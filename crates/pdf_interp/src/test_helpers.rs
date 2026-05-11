//! Test-only helpers shared across `pdf_interp`'s unit tests.

#![cfg(test)]

use pdf::Document;

/// Build a minimal valid PDF byte-stream with one empty page tree and parse
/// it.  `extra_catalog` is appended verbatim into the Catalog dictionary —
/// pass `""` for a bare empty document, or e.g. `" /OpenAction <</S
/// /JavaScript>>"` to inject behaviour a test needs to probe.
///
/// The byte offsets and xref table are computed from the actual section
/// lengths, so adding entries to `extra_catalog` keeps the xref correct
/// automatically.
pub fn make_doc(extra_catalog: &str) -> Document {
    let header = "%PDF-1.4\n";
    let obj1 = format!("1 0 obj\n<</Type /Catalog /Pages 2 0 R{extra_catalog}>>\nendobj\n");
    let obj2 = "2 0 obj\n<</Type /Pages /Kids [] /Count 0>>\nendobj\n";
    let off1 = header.len();
    let off2 = off1 + obj1.len();
    let xref_start = off2 + obj2.len();
    let xref =
        format!("xref\n0 3\n0000000000 65535 f\r\n{off1:010} 00000 n\r\n{off2:010} 00000 n\r\n");
    let trailer = format!("trailer\n<</Size 3 /Root 1 0 R>>\nstartxref\n{xref_start}\n%%EOF");
    let bytes = format!("{header}{obj1}{obj2}{xref}{trailer}").into_bytes();
    Document::from_bytes_owned(bytes).expect("test PDF parse")
}

/// `make_doc("")` shorthand — a bare empty document with no Catalog
/// extras.  Most tests want this form.
pub fn empty_doc() -> Document {
    make_doc("")
}
