//! Test-only helpers shared across `rasterrocket-interp`'s unit tests.

#![cfg(test)]

use std::fmt::Write as _;

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

/// Build a minimal valid PDF with one indirect stream object (id `(2, 0)`)
/// and parse it. The stream content is the raw `stream_bytes`; the stream
/// dictionary contains `/Length stream_bytes.len()` plus any caller-
/// supplied `dict_extras` (e.g. `" /N 3"` for an ICC profile stream).
///
/// Object layout:
/// - `1 0 obj` — Catalog with `/Pages 3 0 R`
/// - `2 0 obj` — the stream (caller's bytes)
/// - `3 0 obj` — empty Pages tree
///
/// Stream bytes are written verbatim (no `FlateDecode` wrapping);
/// callers embedding text content should ensure the bytes don't contain
/// `endstream` or other PDF tokens.
///
/// Used by tests that need to dereference an indirect stream — e.g. ICC
/// profile resolution.
pub fn make_doc_with_stream(stream_bytes: &[u8], dict_extras: &str) -> Document {
    let header = b"%PDF-1.4\n";
    let obj1 = b"1 0 obj\n<</Type /Catalog /Pages 3 0 R>>\nendobj\n";
    let obj2_dict = format!(
        "2 0 obj\n<</Length {}{dict_extras}>>\nstream\n",
        stream_bytes.len(),
    );
    let obj2_close = b"\nendstream\nendobj\n";
    let obj3 = b"3 0 obj\n<</Type /Pages /Kids [] /Count 0>>\nendobj\n";

    let off1 = header.len();
    let off2 = off1 + obj1.len();
    let obj2_len = obj2_dict.len() + stream_bytes.len() + obj2_close.len();
    let off3 = off2 + obj2_len;
    let xref_start = off3 + obj3.len();

    let xref = format!(
        "xref\n0 4\n0000000000 65535 f\r\n\
         {off1:010} 00000 n\r\n\
         {off2:010} 00000 n\r\n\
         {off3:010} 00000 n\r\n",
    );
    let trailer = format!("trailer\n<</Size 4 /Root 1 0 R>>\nstartxref\n{xref_start}\n%%EOF");

    let mut bytes = Vec::with_capacity(xref_start + xref.len() + trailer.len());
    bytes.extend_from_slice(header);
    bytes.extend_from_slice(obj1);
    bytes.extend_from_slice(obj2_dict.as_bytes());
    bytes.extend_from_slice(stream_bytes);
    bytes.extend_from_slice(obj2_close);
    bytes.extend_from_slice(obj3);
    bytes.extend_from_slice(xref.as_bytes());
    bytes.extend_from_slice(trailer.as_bytes());

    Document::from_bytes_owned(bytes).expect("test PDF with stream parse")
}

/// Build a minimal valid PDF whose object `(2, 0)` is the verbatim
/// `obj2_body` (e.g. an array literal `[ 3 3 250 ]`).  Returned alongside is
/// nothing — callers fetch `(2, 0)` via [`Document::get_object`] or build a
/// dictionary that references it (`Object::Reference((2, 0))`) and resolve it
/// through [`Document::resolve`].
///
/// Object layout:
/// - `1 0 obj` — Catalog with `/Pages 3 0 R`
/// - `2 0 obj` — the caller's body, written verbatim
/// - `3 0 obj` — empty Pages tree
pub fn make_doc_with_object(obj2_body: &str) -> Document {
    let header = "%PDF-1.4\n";
    let obj1 = "1 0 obj\n<</Type /Catalog /Pages 3 0 R>>\nendobj\n";
    let obj2 = format!("2 0 obj\n{obj2_body}\nendobj\n");
    let obj3 = "3 0 obj\n<</Type /Pages /Kids [] /Count 0>>\nendobj\n";
    let off1 = header.len();
    let off2 = off1 + obj1.len();
    let off3 = off2 + obj2.len();
    let xref_start = off3 + obj3.len();
    let xref = format!(
        "xref\n0 4\n0000000000 65535 f\r\n\
         {off1:010} 00000 n\r\n{off2:010} 00000 n\r\n{off3:010} 00000 n\r\n"
    );
    let trailer = format!("trailer\n<</Size 4 /Root 1 0 R>>\nstartxref\n{xref_start}\n%%EOF");
    let bytes = format!("{header}{obj1}{obj2}{obj3}{xref}{trailer}").into_bytes();
    Document::from_bytes_owned(bytes).expect("test PDF with object parse")
}

/// Build a minimal valid PDF containing arbitrary extra indirect objects.
///
/// `objects` is a slice of `(id, body)` pairs; each becomes
/// `{id} 0 obj\n{body}\nendobj\n`, written verbatim in the given order.  Ids
/// must be unique, all `> 3` (objects 1–3 are reserved for the Catalog and
/// page tree), and contiguous from 4 upward so the single-subsection xref
/// stays valid.  This supports tests that need indirect *inner* elements
/// (e.g. a `/W` array whose width entries are `9 0 R 10 0 R`) or deliberate
/// reference cycles (`5 0 obj 5 0 R endobj`) to prove cycle-bounded resolution.
///
/// Object layout:
/// - `1 0 obj` — Catalog with `/Pages 3 0 R`
/// - `2 0 obj` — empty (reserved; kept so ids line up with the other helpers)
/// - `3 0 obj` — empty Pages tree
/// - `4..` — caller's `objects`
pub fn make_doc_with_objects(objects: &[(u32, &str)]) -> Document {
    // The xref is a single `0 N` subsection, so the kth offset is bound to
    // object id k.  Ids must therefore be exactly 4, 5, 6, … in order; a gap
    // would silently bind the wrong object to a reference.  Assert loudly so a
    // mis-numbered test fails on construction, not on a confusing later value.
    for (expected, (id, _)) in (4u32..).zip(objects) {
        assert_eq!(
            *id, expected,
            "make_doc_with_objects requires contiguous ids from 4 upward"
        );
    }

    let header = "%PDF-1.4\n";
    let fixed = [
        "1 0 obj\n<</Type /Catalog /Pages 3 0 R>>\nendobj\n".to_string(),
        "2 0 obj\nnull\nendobj\n".to_string(),
        "3 0 obj\n<</Type /Pages /Kids [] /Count 0>>\nendobj\n".to_string(),
    ];
    let extra: Vec<String> = objects
        .iter()
        .map(|(id, body)| format!("{id} 0 obj\n{body}\nendobj\n"))
        .collect();

    // Byte offset of each object, in id order (1, 2, 3, then the extras).
    let mut offsets = Vec::with_capacity(3 + extra.len());
    let mut cursor = header.len();
    for section in fixed.iter().chain(extra.iter()) {
        offsets.push(cursor);
        cursor += section.len();
    }
    let xref_start = cursor;
    let count = offsets.len() + 1; // +1 for the free object 0.

    let mut xref = format!("xref\n0 {count}\n0000000000 65535 f\r\n");
    for off in &offsets {
        write!(xref, "{off:010} 00000 n\r\n").expect("writing to a String is infallible");
    }
    let trailer = format!("trailer\n<</Size {count} /Root 1 0 R>>\nstartxref\n{xref_start}\n%%EOF");

    let mut body = String::with_capacity(xref_start + xref.len() + trailer.len());
    body.push_str(header);
    for section in fixed.iter().chain(extra.iter()) {
        body.push_str(section);
    }
    body.push_str(&xref);
    body.push_str(&trailer);
    Document::from_bytes_owned(body.into_bytes()).expect("test PDF with objects parse")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn make_doc_with_stream_round_trips_bytes() {
        // Embed a small known byte sequence; dereference (2, 0) and verify
        // the bytes come back unchanged.  Catches xref-offset arithmetic
        // regressions in the helper itself.
        let content = b"Hello, ICC profile placeholder!";
        let doc = make_doc_with_stream(content, "");
        let obj = doc
            .get_object((2, 0))
            .expect("(2,0) must resolve to the stream object");
        let stream = obj.as_stream().expect("(2,0) must be a Stream");
        let decoded = stream
            .decompressed_content()
            .expect("no filter → raw bytes");
        assert_eq!(decoded, content);
    }

    #[test]
    fn make_doc_with_stream_carries_dict_extras() {
        // /N 3 in the stream dict — the resolver should see it on the dict.
        let doc = make_doc_with_stream(b"abc", " /N 3");
        let obj = doc.get_object((2, 0)).unwrap();
        let stream = obj.as_stream().unwrap();
        let n = stream.dict.get(b"N").and_then(pdf::Object::as_i64);
        assert_eq!(n, Some(3));
    }
}
