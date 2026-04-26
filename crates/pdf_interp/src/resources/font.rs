//! PDF font resource extraction.
//!
//! Given a PDF font dictionary (as returned by `lopdf::Document::get_page_fonts`),
//! [`resolve_font`] extracts everything the rendering pipeline needs to load a
//! `FreeType` face: the raw font bytes (from an embedded stream or a standard
//! substitute), the font kind, and the char-to-glyph-index map.
//!
//! # PDF font taxonomy
//!
//! ```text
//! Subtype
//!   Type1   → PostScript Type 1 / CFF
//!   MMType1 → Multiple Master (treated as Type 1)
//!   TrueType → TrueType / OpenType-TT
//!   Type0   → CIDFont composite (descendant font carries the actual bytes)
//!   Type3   → paint-procedure glyph (not yet supported)
//!   CIDFontType0 / CIDFontType2 → only appear as DescendantFonts entries
//! ```
//!
//! Embedded bytes live under `FontDescriptor` → `FontFile` (Type 1),
//! `FontFile2` (TrueType), or `FontFile3` (CFF/OpenType-CFF).
//!
//! When no bytes are embedded (standard 14, or unembedded third-party fonts)
//! we fall back to a hardcoded Helvetica substitute shipped with the `font` crate.

use lopdf::{Dictionary, Document, Object};

// ── Public types ──────────────────────────────────────────────────────────────

/// The kind of font outline — used to select FreeType hinting flags.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PdfFontKind {
    /// Type 1 / Type 1C (CFF-wrapped).
    Type1,
    /// TrueType / OpenType-TT.
    TrueType,
    /// Anything else (OpenType-CFF, CID, unknown).
    Other,
}

/// Everything needed to load a FreeType face for one PDF font resource.
#[derive(Debug)]
pub struct FontDescriptor {
    /// Kind of font outline — controls FreeType hinting strategy.
    pub kind: PdfFontKind,
    /// Raw font bytes.  `Some` if embedded in the PDF, `None` if we must fall
    /// back to a substitute.
    pub bytes: Option<Vec<u8>>,
    /// Per-character advance widths in thousandths of a text-space unit,
    /// indexed by `char_code - first_char`.  Empty for CID/Type0 fonts.
    pub widths: Vec<i32>,
    /// First char code covered by `widths`.
    pub first_char: u32,
    /// Glyph-index table: `code_to_gid[char_code]` → FT glyph index.
    /// Empty means the identity map (char_code == glyph index).
    pub code_to_gid: Vec<u32>,
}

// ── Entry point ───────────────────────────────────────────────────────────────

/// Extract a [`FontDescriptor`] from a PDF font dictionary.
///
/// This is infallible: if any optional field is missing we fill in a sensible
/// default so the renderer can always produce *something*, even if it is wrong.
#[must_use]
pub fn resolve_font(doc: &Document, dict: &Dictionary) -> FontDescriptor {
    let kind = classify_kind(dict);
    let bytes = extract_bytes(doc, dict);
    let (first_char, widths) = extract_widths(dict);
    let code_to_gid = extract_code_to_gid(doc, dict, &kind);

    FontDescriptor {
        kind,
        bytes,
        widths,
        first_char,
        code_to_gid,
    }
}

// ── Kind classification ───────────────────────────────────────────────────────

fn classify_kind(dict: &Dictionary) -> PdfFontKind {
    let subtype = dict
        .get(b"Subtype")
        .ok()
        .and_then(|o| o.as_name().ok())
        .unwrap_or(b"");

    match subtype {
        b"Type1" | b"MMType1" => PdfFontKind::Type1,
        b"TrueType" => PdfFontKind::TrueType,
        _ => PdfFontKind::Other,
    }
}

// ── Embedded byte extraction ──────────────────────────────────────────────────

/// Try to read embedded font bytes from the FontDescriptor sub-dict.
///
/// Priority: `FontFile2` (TT) → `FontFile3` (CFF/OTF) → `FontFile` (Type 1).
/// Returns `None` if nothing is embedded.
fn extract_bytes(doc: &Document, dict: &Dictionary) -> Option<Vec<u8>> {
    // Resolve FontDescriptor indirect reference.
    let fd = resolve_fd(doc, dict)?;

    for key in [
        b"FontFile2".as_ref(),
        b"FontFile3".as_ref(),
        b"FontFile".as_ref(),
    ] {
        if let Some(bytes) = read_stream(doc, fd, key) {
            return Some(bytes);
        }
    }
    None
}

/// Dereference the `FontDescriptor` entry, which is almost always an indirect
/// reference in practice.
fn resolve_fd<'a>(doc: &'a Document, dict: &'a Dictionary) -> Option<&'a Dictionary> {
    match dict.get(b"FontDescriptor").ok()? {
        Object::Reference(id) => doc.get_dictionary(*id).ok(),
        Object::Dictionary(d) => Some(d),
        _ => None,
    }
}

/// Read and decompress a stream at `dict[key]`, following an indirect ref if
/// necessary.  Returns `None` on any error.
fn read_stream(doc: &Document, dict: &Dictionary, key: &[u8]) -> Option<Vec<u8>> {
    let obj = match dict.get(key).ok()? {
        Object::Reference(id) => doc.get_object(*id).ok()?,
        other => other,
    };
    let stream = obj.as_stream().ok()?;
    stream.decompressed_content().ok()
}

// ── Advance-width extraction ──────────────────────────────────────────────────

/// Extract `(FirstChar, Widths[])` from the font dict.
///
/// Returns `(0, vec![])` if either key is absent.
fn extract_widths(dict: &Dictionary) -> (u32, Vec<i32>) {
    let first = dict
        .get(b"FirstChar")
        .ok()
        .and_then(|o| o.as_i64().ok())
        .filter(|&v| v >= 0)
        .map_or(0u32, |v| v as u32);

    let widths = dict
        .get(b"Widths")
        .ok()
        .and_then(|o| o.as_array().ok())
        .map(|arr| {
            arr.iter()
                .map(|o| o.as_i64().map_or(0, |v| v as i32))
                .collect()
        })
        .unwrap_or_default();

    (first, widths)
}

// ── Char → glyph-index mapping ────────────────────────────────────────────────

/// Build a char-to-glyph-index table from the font's `Encoding` entry.
///
/// Returns an empty `Vec` when the identity map is appropriate (most CID fonts,
/// or simple fonts with no explicit encoding).
fn extract_code_to_gid(doc: &Document, dict: &Dictionary, kind: &PdfFontKind) -> Vec<u32> {
    // Type 0 / CID composite fonts: the descendant font carries the GID map.
    // For now, return identity — full CMap support is phase 2.
    let subtype = dict
        .get(b"Subtype")
        .ok()
        .and_then(|o| o.as_name().ok())
        .unwrap_or(b"");
    if matches!(subtype, b"Type0") {
        return vec![];
    }

    let encoding = match dict.get(b"Encoding").ok() {
        Some(e) => e,
        None => return vec![],
    };

    match encoding {
        Object::Name(name) => standard_encoding_table(name.as_slice()),
        Object::Reference(id) => doc
            .get_dictionary(*id)
            .ok()
            .map(|d| dict_encoding_table(doc, d, kind))
            .unwrap_or_default(),
        Object::Dictionary(d) => dict_encoding_table(doc, d, kind),
        _ => vec![],
    }
}

/// Produce a code→glyph table for a standard named encoding.
///
/// Returns an empty `Vec` for encodings that use the identity map
/// (WinAnsiEncoding, MacRomanEncoding, StandardEncoding), since FreeType
/// can handle these itself via `FT_Get_Char_Index`.
///
/// `Symbol` and `ZapfDingbats` have non-standard mappings, but they are rare
/// and deferring full support is acceptable for now.
#[expect(
    clippy::missing_const_for_fn,
    reason = "vec![] is not const; clippy false-positive for this pattern"
)]
fn standard_encoding_table(name: &[u8]) -> Vec<u32> {
    // For all standard encodings, FreeType's charmap handles the mapping.
    // Return empty → identity map.
    let _ = name;
    vec![]
}

/// Parse a dictionary-form `Encoding` object.
///
/// Dictionary encodings have an optional `BaseEncoding` plus a `Differences`
/// array of the form `[base_code /GlyphName ...]`.  We map glyph names to
/// glyph indices using `FT_Get_Name_Index`.
///
/// For now we return empty (identity map) to keep phase-1 scope small;
/// full `Differences` parsing is phase 2.
#[expect(
    clippy::missing_const_for_fn,
    reason = "vec![] is not const; clippy false-positive for this pattern"
)]
fn dict_encoding_table(_doc: &Document, _dict: &Dictionary, _kind: &PdfFontKind) -> Vec<u32> {
    // TODO(phase2): parse BaseEncoding + Differences array and build
    // a proper 256-entry code_to_gid table via FT_Get_Name_Index.
    vec![]
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_dict(pairs: &[(&[u8], Object)]) -> Dictionary {
        let mut d = Dictionary::new();
        for (k, v) in pairs {
            d.set(*k, v.clone());
        }
        d
    }

    #[test]
    fn classify_type1() {
        let d = make_dict(&[(b"Subtype", Object::Name(b"Type1".to_vec()))]);
        assert_eq!(classify_kind(&d), PdfFontKind::Type1);
    }

    #[test]
    fn classify_truetype() {
        let d = make_dict(&[(b"Subtype", Object::Name(b"TrueType".to_vec()))]);
        assert_eq!(classify_kind(&d), PdfFontKind::TrueType);
    }

    #[test]
    fn classify_unknown_is_other() {
        let d = make_dict(&[(b"Subtype", Object::Name(b"Type0".to_vec()))]);
        assert_eq!(classify_kind(&d), PdfFontKind::Other);
    }

    #[test]
    fn widths_extracted() {
        let d = make_dict(&[
            (b"FirstChar", Object::Integer(32)),
            (
                b"Widths",
                Object::Array(vec![
                    Object::Integer(278),
                    Object::Integer(500),
                    Object::Integer(556),
                ]),
            ),
        ]);
        let (fc, ws) = extract_widths(&d);
        assert_eq!(fc, 32);
        assert_eq!(ws, vec![278, 500, 556]);
    }

    #[test]
    fn widths_absent_returns_empty() {
        let d = make_dict(&[]);
        let (fc, ws) = extract_widths(&d);
        assert_eq!(fc, 0);
        assert!(ws.is_empty());
    }

    #[test]
    fn no_embedded_bytes_returns_none() {
        // A minimal font dict with no FontDescriptor.
        let d = make_dict(&[(b"Subtype", Object::Name(b"Type1".to_vec()))]);
        // We need a Document to call extract_bytes, but for simple cases without
        // embedded streams the result is None.
        // Build a trivial document stub via lopdf.
        let doc = Document::new();
        assert!(extract_bytes(&doc, &d).is_none());
    }
}
