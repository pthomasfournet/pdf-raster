//! Test-only helpers shared across `rasterrocket-interp`'s unit tests.

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
