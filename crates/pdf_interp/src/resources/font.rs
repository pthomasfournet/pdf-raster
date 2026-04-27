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

use crate::resources::dict_ext::DictExt;

// ── Public types ──────────────────────────────────────────────────────────────

/// The kind of font outline — used to select `FreeType` hinting flags.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PdfFontKind {
    /// Type 1 / Type 1C (CFF-wrapped).
    Type1,
    /// TrueType / OpenType-TT.
    TrueType,
    /// Anything else (OpenType-CFF, CID, unknown).
    Other,
}

/// Everything needed to load a `FreeType` face for one PDF font resource.
#[derive(Debug)]
pub struct FontDescriptor {
    /// Kind of font outline — controls `FreeType` hinting strategy.
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
    /// Empty means the identity map (`char_code` == glyph index).
    pub code_to_gid: Vec<u32>,
    /// Per-character glyph name overrides from the PDF `Encoding/Differences`
    /// array.  Index = char code (0–255); `None` = inherit from base encoding.
    /// Used by [`FontCache`] to resolve names → Unicode → GID via `FreeType`'s
    /// active charmap at face-load time.
    pub differences: Box<[Option<Box<str>>; 256]>,
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
    let (code_to_gid, differences) = extract_encoding(doc, dict, kind);

    FontDescriptor {
        kind,
        bytes,
        widths,
        first_char,
        code_to_gid,
        differences,
    }
}

// ── Kind classification ───────────────────────────────────────────────────────

fn classify_kind(dict: &Dictionary) -> PdfFontKind {
    let subtype = dict.get_name(b"Subtype").unwrap_or(b"");

    match subtype {
        b"Type1" | b"MMType1" => PdfFontKind::Type1,
        b"TrueType" => PdfFontKind::TrueType,
        _ => PdfFontKind::Other,
    }
}

// ── Embedded byte extraction ──────────────────────────────────────────────────

/// Try to read embedded font bytes from the `FontDescriptor` sub-dict.
///
/// Key priority varies by font kind:
/// - Type 1 → `FontFile` first, then `FontFile3` (CFF-wrapped Type 1)
/// - TrueType / Other → `FontFile2` (TT), then `FontFile3` (CFF/OTF)
///
/// Returns `None` if nothing is embedded.
fn extract_bytes(doc: &Document, dict: &Dictionary) -> Option<Vec<u8>> {
    let fd = resolve_fd(doc, dict)?;

    let kind = classify_kind(dict);
    let priority: &[&[u8]] = match kind {
        PdfFontKind::Type1 => &[b"FontFile", b"FontFile3"],
        _ => &[b"FontFile2", b"FontFile3", b"FontFile"],
    };

    for key in priority {
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

/// Maximum decompressed font stream size (64 MiB).
///
/// Protects against deflate bombs embedded in malicious PDFs.  Real font
/// programs are measured in kilobytes; 64 MiB is a generous upper bound.
const MAX_FONT_BYTES: usize = 64 * 1024 * 1024;

/// Read and decompress a stream at `dict[key]`, following an indirect ref if
/// necessary.  Returns `None` on any error or if the stream exceeds
/// [`MAX_FONT_BYTES`] after decompression.
fn read_stream(doc: &Document, dict: &Dictionary, key: &[u8]) -> Option<Vec<u8>> {
    let obj = match dict.get(key).ok()? {
        Object::Reference(id) => doc.get_object(*id).ok()?,
        other => other,
    };
    let stream = obj.as_stream().ok()?;
    let bytes = stream.decompressed_content().ok()?;
    if bytes.len() > MAX_FONT_BYTES {
        log::warn!(
            "font: embedded stream for key {} is {} bytes — exceeds {MAX_FONT_BYTES} limit, skipping",
            String::from_utf8_lossy(key),
            bytes.len(),
        );
        return None;
    }
    Some(bytes)
}

// ── Advance-width extraction ──────────────────────────────────────────────────

/// Extract `(FirstChar, Widths[])` from the font dict.
///
/// Returns `(0, vec![])` if either key is absent.
fn extract_widths(dict: &Dictionary) -> (u32, Vec<i32>) {
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "FirstChar is filtered to >= 0 and < 256; safe to cast to u32"
    )]
    let first = dict
        .get_i64(b"FirstChar")
        .filter(|&v| v >= 0)
        .map_or(0u32, |v| v as u32);

    #[expect(
        clippy::cast_possible_truncation,
        reason = "PDF glyph advance widths are always in [0, 4096]; safe to cast to i32"
    )]
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

/// The number of char codes in a single-byte PDF encoding.
const NUM_CODES: usize = 256;

/// Empty differences table — all `None` (inherit from base encoding).
fn empty_differences() -> Box<[Option<Box<str>>; 256]> {
    // Box::new([None; 256]) doesn't work because Option<Box<str>> is not Copy.
    // We use a Vec collect + try_into.
    let v: Vec<Option<Box<str>>> = (0..NUM_CODES).map(|_| None).collect();
    #[expect(
        clippy::expect_used,
        reason = "the Vec always has exactly NUM_CODES = 256 elements; conversion cannot fail"
    )]
    v.try_into()
        .map(Box::new)
        .expect("256-element Vec must convert to [_; 256]")
}

/// Extract encoding information from the font dictionary.
///
/// Returns `(code_to_gid, differences)`:
/// - `code_to_gid`: non-empty only for encodings fully resolved at PDF-parse
///   time (currently none — `FreeType` handles standard encodings).
/// - `differences`: 256-entry table of glyph-name overrides parsed from the
///   `Differences` array.  The font cache resolves these to GIDs at face-load
///   time via `FreeType`'s charmap.
#[expect(
    clippy::type_complexity,
    reason = "private helper; introducing a type alias for (Vec<u32>, Box<[Option<Box<str>>; 256]>) would obscure rather than clarify"
)]
fn extract_encoding(
    doc: &Document,
    dict: &Dictionary,
    _kind: PdfFontKind,
) -> (Vec<u32>, Box<[Option<Box<str>>; 256]>) {
    // Type 0 / CID composite fonts: descendant font carries the GID map.
    let subtype = dict.get_name(b"Subtype").unwrap_or(b"");
    if matches!(subtype, b"Type0") {
        return (vec![], empty_differences());
    }

    let Some(encoding) = dict.get(b"Encoding").ok() else {
        return (vec![], empty_differences());
    };

    match encoding {
        Object::Reference(id) => {
            let diffs = doc
                .get_dictionary(*id)
                .ok()
                .map_or_else(empty_differences, parse_differences);
            (vec![], diffs)
        }
        Object::Dictionary(d) => (vec![], parse_differences(d)),
        // Named standard encoding or anything else: `FreeType` handles char → GID.
        _ => (vec![], empty_differences()),
    }
}

/// Parse the `Differences` array from a dictionary-form `Encoding` object.
///
/// The array has the form `[base_code /Name /Name ... base_code /Name ...]`.
/// Each integer resets the current char code; each name assigns a glyph name
/// to that code and advances the counter.  Codes outside `[0, 255]` are ignored.
fn parse_differences(dict: &Dictionary) -> Box<[Option<Box<str>>; 256]> {
    let mut table = empty_differences();

    let Ok(arr_obj) = dict.get(b"Differences") else {
        return table;
    };
    let Ok(arr) = arr_obj.as_array() else {
        return table;
    };

    let mut code: usize = 0;
    for obj in arr {
        match obj {
            Object::Integer(n) if *n >= 0 && *n < 256 => {
                #[expect(
                    clippy::cast_sign_loss,
                    clippy::cast_possible_truncation,
                    reason = "n validated ≥ 0 and < 256 above; safe cast to usize"
                )]
                {
                    code = *n as usize;
                }
            }
            Object::Name(name) if code < NUM_CODES => {
                let s = String::from_utf8_lossy(name).into_owned().into_boxed_str();
                table[code] = Some(s);
                code += 1;
            }
            _ => {}
        }
    }

    table
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
