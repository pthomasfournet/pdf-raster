//! PDF font resource extraction.
//!
//! Given a PDF font dictionary (as returned by `pdf::Document::get_page_fonts`),
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
//!   Type3   → paint-procedure glyph (content streams, no FreeType face)
//!   CIDFontType0 / CIDFontType2 → only appear as DescendantFonts entries
//! ```
//!
//! Embedded bytes live under `FontDescriptor` → `FontFile` (Type 1),
//! `FontFile2` (TrueType), or `FontFile3` (CFF/OpenType-CFF).
//!
//! When no bytes are embedded (standard 14, or unembedded third-party fonts)
//! we fall back to a hardcoded Helvetica substitute shipped with the `font` crate.

use std::sync::Arc;

use pdf::{Dictionary, Document, Object};

use crate::resources::cmap::{CMap, parse_cmap};
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

/// The base encoding a simple font's char codes resolve through before any
/// `Differences` overrides (PDF §9.6.6).  Determines the code→glyph-name
/// table the font cache uses when resolving glyphs by name.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BaseEncoding {
    /// `/StandardEncoding` — Adobe Standard.  Also the spec default for a
    /// non-symbolic simple font with no explicit base encoding.
    Standard,
    /// `/WinAnsiEncoding` — Windows Code Page 1252 superset.
    WinAnsi,
    /// `/MacRomanEncoding` — Mac OS Roman.
    MacRoman,
}

/// CID encoding information for Type 0 composite fonts.
///
/// Type 0 fonts encode character codes as 1–4 byte sequences.  The `Encoding`
/// `CMap` maps byte sequences to CIDs; the `CIDToGIDMap` (or identity) then maps
/// CIDs to `FreeType` GIDs.
#[derive(Debug)]
pub struct CidEncoding {
    /// Decoded `Encoding` `CMap` — maps char codes to CIDs.
    ///
    /// `None` when no `CMap` stream could be parsed (`Identity-H`/V or missing);
    /// in that case charcode == CID (identity mapping).
    pub encoding_cmap: Option<CMap>,

    /// Precomputed CID → GID lookup from `CIDToGIDMap`.
    ///
    /// `None` means Identity (CID == GID, the common case for OpenType-CFF and
    /// TrueType `CIDFonts`).
    pub cid_to_gid: Option<Vec<u32>>,

    /// Default advance width in thousandths of a text-space unit (`DW` key).
    pub default_width: i32,

    /// Explicit per-CID advance widths from the `W` array.
    ///
    /// Stored as `(first_cid, [widths…])` segments matching the PDF `W` format.
    pub widths: Vec<(u32, Vec<i32>)>,
}

impl CidEncoding {
    /// Return the advance width for `cid` in thousandths of a text-space unit.
    ///
    /// Searches the explicit `W` table first; falls back to `default_width`.
    #[must_use]
    pub fn width_for_cid(&self, cid: u32) -> i32 {
        for (first, ws) in &self.widths {
            if cid >= *first {
                let idx = (cid - first) as usize;
                if idx < ws.len() {
                    return ws[idx];
                }
            }
        }
        self.default_width
    }

    /// Map a character code to a GID, applying both `encoding_cmap` and
    /// `cid_to_gid`.
    ///
    /// Returns 0 (`.notdef`) when the code is absent from either map.
    #[must_use]
    pub fn code_to_gid(&self, char_code: u32) -> u32 {
        // Step 1: char code → CID via encoding CMap (identity if absent).
        let cid = self.encoding_cmap.as_ref().map_or(char_code, |cmap| {
            cmap.map.get(&char_code).copied().unwrap_or(0)
        });

        // Step 2: CID → GID via table (identity if absent).
        // CIDs are 16-bit values (PDF spec §9.7.4); usize is always ≥ 16 bits.
        self.cid_to_gid
            .as_ref()
            .map_or(cid, |table| table.get(cid as usize).copied().unwrap_or(0))
    }
}

/// Decoded `CharProc` entry for a single Type 3 glyph.
#[derive(Debug, Clone)]
pub struct Type3Glyph {
    /// Decompressed content stream bytes for this glyph's paint procedure.
    pub content: Vec<u8>,
    /// Advance width in thousandths of a text-space unit, from the `Widths` array.
    /// `None` when absent (fall back to 0).
    pub width_units: i32,
}

/// All data extracted from a Type 3 font dictionary.
///
/// Unlike FreeType-backed fonts, Type 3 glyphs are PDF content streams that
/// must be executed by a child renderer.  Each `CharProc` stream begins with a
/// `d0` (colorless) or `d1` (self-colored) operator that declares the glyph
/// advance width.
#[derive(Debug, Clone)]
pub struct Type3Data {
    /// Resolved glyph content keyed by char code (0–255).
    /// Missing entries mean `.notdef`; the caller renders nothing for them.
    pub glyphs: Vec<Option<Type3Glyph>>,
    /// 6-element `FontMatrix` (user-space → glyph-space transform).
    /// Default is `[0.001, 0, 0, 0.001, 0, 0]` per PDF spec §9.6.5.
    pub font_matrix: [f64; 6],
}

impl Type3Data {
    /// Return the content and nominal width for `char_code`, if any `CharProc` exists.
    #[must_use]
    pub fn glyph(&self, char_code: u8) -> Option<&Type3Glyph> {
        self.glyphs.get(usize::from(char_code))?.as_ref()
    }
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
    /// `MissingWidth` from the `FontDescriptor` sub-dict — the advance width
    /// (in thousandths of a text-space unit) used for char codes outside the
    /// `[first_char, first_char + widths.len())` range.  Default `0` per PDF
    /// §9.8.2.
    pub missing_width: i32,
    /// Glyph-index table: `code_to_gid[char_code]` → FT glyph index.
    /// Empty means the identity map (`char_code` == glyph index).
    pub code_to_gid: Vec<u32>,
    /// Per-character glyph name overrides from the PDF `Encoding/Differences`
    /// array.  Index = char code (0–255); `None` = inherit from base encoding.
    /// Used by `FontCache` (in `rasterrocket_interp::renderer::font_cache`) to resolve
    /// names → Unicode → GID via `FreeType`'s active charmap at face-load time.
    pub differences: Box<[Option<Box<str>>; 256]>,
    /// Base encoding for simple fonts: the code→glyph-name table that applies
    /// before `Differences` overrides.  `StandardEncoding` per PDF spec when
    /// no `/BaseEncoding` is named.  Unused for Type 0 / Type 3.
    pub base_encoding: BaseEncoding,
    /// CID encoding for Type 0 composite fonts.  `Some` when `Subtype` is
    /// `Type0`; `None` for all simple fonts (Type1, TrueType, `MMType1`).
    pub cid_encoding: Option<CidEncoding>,
    /// Type 3 glyph data.  `Some` only when `Subtype` is `Type3`; `None` for
    /// all FreeType-backed fonts.  When `Some`, `bytes` is `None` and `kind`
    /// is `Other` (Type 3 has no font program; glyphs are content streams).
    pub type3: Option<Type3Data>,
}

impl FontDescriptor {
    /// Return the simple-font advance width (in thousandths of a text-space
    /// unit) for `char_code`, or `None` if the `Widths` array is empty.
    ///
    /// PDF §9.2.4: the `Widths` array is the **authoritative** width source
    /// for simple fonts in the `[FirstChar, LastChar]` range.  Outside that
    /// range, `MissingWidth` applies.  The whole array is allowed to be
    /// absent only for the 14 standard fonts (Helvetica, Times-Roman, …),
    /// where the `FreeType` `horiAdvance` of the substitute face is the
    /// fallback — that's what `None` here signals to the caller.
    ///
    /// Not for Type 0 (use `CidEncoding::width_for_cid`) or Type 3
    /// (use `Type3Glyph::width_units`).
    #[must_use]
    pub fn width_for_code(&self, char_code: u32) -> Option<i32> {
        if self.widths.is_empty() {
            return None;
        }
        // Below FirstChar → out-of-range below; use MissingWidth.
        // At-or-above FirstChar: look up; out-of-range above also uses MissingWidth.
        let Some(idx) = (char_code as usize).checked_sub(self.first_char as usize) else {
            return Some(self.missing_width);
        };
        Some(self.widths.get(idx).copied().unwrap_or(self.missing_width))
    }
}

// ── Entry point ───────────────────────────────────────────────────────────────

/// Extract a [`FontDescriptor`] from a PDF font dictionary.
///
/// This is infallible: if any optional field is missing we fill in a sensible
/// default so the renderer can always produce *something*, even if it is wrong.
#[must_use]
pub fn resolve_font(doc: &Document, dict: &Dictionary) -> FontDescriptor {
    let subtype = dict.get_name(b"Subtype").unwrap_or(b"");

    if matches!(subtype, b"Type0") {
        return resolve_type0_font(doc, dict);
    }
    if matches!(subtype, b"Type3") {
        return resolve_type3_font(doc, dict);
    }

    let kind = classify_kind(dict);
    let bytes = extract_bytes(doc, dict);
    let (first_char, widths) = extract_widths(dict);
    let missing_width = extract_missing_width(doc, dict);
    let (code_to_gid, differences, base_encoding) = extract_encoding(doc, dict, kind);

    FontDescriptor {
        kind,
        bytes,
        widths,
        first_char,
        missing_width,
        code_to_gid,
        differences,
        base_encoding,
        cid_encoding: None,
        type3: None,
    }
}

/// Extract `MissingWidth` from the `FontDescriptor` sub-dict.  Returns `0`
/// when absent (PDF §9.8.2 default).  PDF stores `MissingWidth` inside the
/// `FontDescriptor` sub-dict, not the parent font dict — this helper handles
/// the indirect-reference resolution that `extract_widths` does not need.
fn extract_missing_width(doc: &Document, dict: &Dictionary) -> i32 {
    resolve_fd(doc, dict)
        .as_ref()
        .and_then(|fd| fd.get_i64(b"MissingWidth"))
        .map_or(0, pdf_width_to_i32)
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
    extract_bytes_with_kind(doc, dict, classify_kind(dict))
}

/// Dereference the `FontDescriptor` entry, which is almost always an indirect
/// reference in practice.
///
/// Returns an owned [`Dictionary`] because `doc.get_dictionary` hands back an
/// `Arc<Dictionary>` rather than a long-lived borrow.
fn resolve_fd(doc: &Document, dict: &Dictionary) -> Option<Dictionary> {
    match dict.get(b"FontDescriptor")? {
        Object::Reference(id) => doc.get_dictionary(*id).ok().map(|a| (*a).clone()),
        Object::Dictionary(d) => Some(d.clone()),
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
    // For inline objects we can read the stream directly; for references the
    // resolved Arc<Object> must outlive the &Stream borrow, so we keep it.
    let direct = dict.get(key)?;
    let resolved;
    let stream = match direct {
        Object::Reference(id) => {
            resolved = doc.get_object(*id).ok()?;
            resolved.as_stream()?
        }
        other => other.as_stream()?,
    };
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

/// Clamp a PDF integer width value to `i32`, saturating on overflow.
///
/// PDF advance-width values come from adversarial input and are not bound
/// by the spec to any particular range; the `Widths` array is `[i64]` at the
/// lopdf layer.  Real font widths sit in `[0, 4096]` (thousandths-of-em, font
/// sizes never exceed 1000em in practice), so any i64 that fits in i32 is the
/// real answer.  Out-of-range values are pathological and we clamp instead of
/// wrapping with `as i32` — a malformed `MissingWidth: i64::MIN` would
/// otherwise silently propagate to text-matrix arithmetic.
///
/// Used by every PDF-width parse site: `Widths` array elements, `MissingWidth`,
/// `DW` (Type 0 default width), and Type 0 `W` array entries.
#[must_use]
pub(crate) fn pdf_width_to_i32(v: i64) -> i32 {
    match i32::try_from(v) {
        Ok(w) => w,
        // Saturation direction depends on the sign of the overflow; v < 0 ⇒ underflow.
        Err(_) if v < 0 => i32::MIN,
        Err(_) => i32::MAX,
    }
}

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

    let widths = dict
        .get(b"Widths")
        .and_then(Object::as_array)
        .map(|arr| {
            arr.iter()
                .map(|o| o.as_i64().map_or(0, pdf_width_to_i32))
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

/// Extract encoding information from a simple (non-Type0) font dictionary.
///
/// Returns `(code_to_gid, differences, base_encoding)`:
/// - `code_to_gid`: non-empty only for encodings fully resolved at PDF-parse
///   time (currently none — resolution happens at face-load time).
/// - `differences`: 256-entry table of glyph-name overrides parsed from the
///   `Differences` array.  The font cache resolves these to GIDs at face-load
///   time.
/// - `base_encoding`: the code→glyph-name table to apply before `Differences`.
///   `/BaseEncoding` if named, else `StandardEncoding` (the PDF-spec default
///   for a simple font with no explicit base encoding).
#[expect(
    clippy::type_complexity,
    reason = "private helper; a type alias for the (code_to_gid, differences, base) triple would obscure rather than clarify"
)]
fn extract_encoding(
    doc: &Document,
    dict: &Dictionary,
    _kind: PdfFontKind,
) -> (Vec<u32>, Box<[Option<Box<str>>; 256]>, BaseEncoding) {
    let Some(encoding) = dict.get(b"Encoding") else {
        return (vec![], empty_differences(), BaseEncoding::Standard);
    };

    match encoding {
        Object::Reference(id) => doc.get_dictionary(*id).ok().map_or_else(
            || (vec![], empty_differences(), BaseEncoding::Standard),
            |d| {
                (
                    vec![],
                    parse_differences(&d),
                    parse_base_encoding(d.get_name(b"BaseEncoding")),
                )
            },
        ),
        Object::Dictionary(d) => (
            vec![],
            parse_differences(d),
            parse_base_encoding(d.get_name(b"BaseEncoding")),
        ),
        // A bare named encoding (`/WinAnsiEncoding`, `/MacRomanEncoding`, …):
        // no Differences, base encoding selected by the name.
        Object::Name(n) => (
            vec![],
            empty_differences(),
            parse_base_encoding(Some(n.as_slice())),
        ),
        _ => (vec![], empty_differences(), BaseEncoding::Standard),
    }
}

/// Map a PDF base-encoding name to [`BaseEncoding`].  Unknown or absent →
/// `StandardEncoding` (PDF-spec default for a simple font).
fn parse_base_encoding(name: Option<&[u8]>) -> BaseEncoding {
    match name {
        Some(b"WinAnsiEncoding") => BaseEncoding::WinAnsi,
        Some(b"MacRomanEncoding") => BaseEncoding::MacRoman,
        _ => BaseEncoding::Standard,
    }
}

// ── Type 0 / CIDFont extraction ───────────────────────────────────────────────

/// Resolve a `Type0` composite font dictionary into a [`FontDescriptor`].
///
/// # Structure
///
/// ```text
/// Type0 dict
///   Encoding: /Identity-H | /Identity-V | stream (CMap)
///   DescendantFonts: [ref]          ← array with exactly one entry
///     Subtype: CIDFontType0 | CIDFontType2
///     FontDescriptor: ref           ← embedded font bytes live here
///     DW: integer                   ← default advance width
///     W: array                      ← per-CID widths
///     CIDToGIDMap: /Identity | ref  ← CID → GID (CIDFontType2 only)
///   ToUnicode: stream               ← optional; not used for rendering
/// ```
fn resolve_type0_font(doc: &Document, dict: &Dictionary) -> FontDescriptor {
    // 1. Encoding CMap: maps char codes to CIDs.
    let encoding_cmap = extract_type0_encoding_cmap(doc, dict);

    // 2. DescendantFonts: the actual CIDFont dict with bytes and widths.
    let descendant = extract_descendant(doc, dict);

    // 3. Extract bytes from the CIDFont's FontDescriptor.
    let bytes = descendant
        .as_ref()
        .and_then(|d| extract_bytes_from_descendant(doc, d));

    // 4. CIDToGIDMap (CIDFontType2 only; CIDFontType0/CFF uses identity).
    let cid_to_gid = descendant.as_ref().and_then(|d| extract_cid_to_gid(doc, d));

    // 5. CIDFont metrics: DW and W.
    let (default_width, widths) = descendant
        .as_ref()
        .map_or((1000, vec![]), |d| extract_cid_widths(doc, d));

    // 6. Determine font kind from CIDFont Subtype.
    let kind = descendant
        .as_ref()
        .map_or(PdfFontKind::Other, classify_kind);

    FontDescriptor {
        kind,
        bytes,
        widths: vec![],
        first_char: 0,
        // Type 0 fonts use `CidEncoding::width_for_cid` exclusively;
        // `missing_width` is never consulted but must be initialised.
        missing_width: 0,
        code_to_gid: vec![],
        differences: empty_differences(),
        // Unused for Type 0 (CID path bypasses name resolution entirely).
        base_encoding: BaseEncoding::Standard,
        cid_encoding: Some(CidEncoding {
            encoding_cmap,
            cid_to_gid,
            default_width,
            widths,
        }),
        type3: None,
    }
}

// ── Type 3 font extraction ────────────────────────────────────────────────────

/// Maximum decompressed size for a single Type 3 `CharProc` stream (4 MiB).
///
/// Real Type 3 glyphs are tiny path/image sequences; 4 MiB is a generous cap.
const MAX_CHARPROC_BYTES: usize = 4 * 1024 * 1024;

/// Extract [`FontDescriptor`] for a Type 3 font.
///
/// Type 3 fonts have no embedded font program — each glyph is a PDF content
/// stream in the `CharProcs` dictionary.  The `Widths` array and `FirstChar`
/// key give nominal advance widths (supplemented by the `d0`/`d1` operators
/// inside each `CharProc`, which are authoritative).
fn resolve_type3_font(doc: &Document, dict: &Dictionary) -> FontDescriptor {
    let (first_char, widths) = extract_widths(dict);

    // FontMatrix transforms glyph space to text space.
    // PDF default is [0.001 0 0 0.001 0 0] (i.e. 1000 units = 1 text-space unit).
    let font_matrix =
        extract_matrix(dict, b"FontMatrix").unwrap_or([0.001, 0.0, 0.0, 0.001, 0.0, 0.0]);

    // Resolve CharProcs: maps glyph name → stream object.
    // We need an owned Dictionary because `doc.get_dictionary` returns Arc.
    let charprocs_obj = dict.get(b"CharProcs");
    let charprocs_dict: Option<Dictionary> = charprocs_obj.and_then(|o| match o {
        Object::Dictionary(d) => Some(d.clone()),
        Object::Reference(id) => doc.get_dictionary(*id).ok().map(|a| (*a).clone()),
        _ => None,
    });

    // Resolve Encoding/Differences to get char_code → glyph_name mapping.
    let differences = extract_encoding(doc, dict, PdfFontKind::Other).1;

    // Build the glyphs table: 256 entries, one per possible char code.
    let mut glyphs: Vec<Option<Type3Glyph>> = (0..256).map(|_| None).collect();

    if let Some(cp) = charprocs_dict {
        for code in 0u8..=255 {
            // Determine the glyph name for this char code.
            // Priority: Differences table overrides (if set), then treat the
            // char code as a single-byte Latin-1 name (rare fallback).
            let Some(glyph_name) = differences
                .get(usize::from(code))
                .and_then(|x| x.as_deref())
            else {
                continue; // No mapping → no glyph for this code.
            };
            let glyph_name = glyph_name.as_bytes();

            // Look up the CharProc stream.
            let Some(stream_obj) = cp.get(glyph_name) else {
                continue;
            };
            let resolved;
            let stream = match stream_obj {
                Object::Stream(s) => s,
                Object::Reference(id) => {
                    let Some(arc) = doc.get_object(*id).ok() else {
                        continue;
                    };
                    resolved = arc;
                    match resolved.as_ref() {
                        Object::Stream(s) => s,
                        _ => continue,
                    }
                }
                _ => continue,
            };

            let content = match stream.decompressed_content() {
                Ok(b) if b.len() <= MAX_CHARPROC_BYTES => b,
                Ok(b) => {
                    log::warn!(
                        "font/type3: CharProc for code {code} is {} bytes (limit {}), skipping",
                        b.len(),
                        MAX_CHARPROC_BYTES
                    );
                    continue;
                }
                Err(e) => {
                    log::warn!("font/type3: failed to decompress CharProc for code {code}: {e}");
                    continue;
                }
            };

            let width_idx = usize::from(code).saturating_sub(first_char as usize);
            let width_units = widths.get(width_idx).copied().unwrap_or(0);

            glyphs[usize::from(code)] = Some(Type3Glyph {
                content,
                width_units,
            });
        }
    } else {
        log::warn!("font/type3: CharProcs dict missing or unresolvable — no glyphs");
    }

    FontDescriptor {
        kind: PdfFontKind::Other,
        bytes: None,
        widths,
        first_char,
        // Type 3 glyphs carry their own widths via `Type3Glyph::width_units`;
        // the top-level Widths array is the same source the d0/d1 operators
        // store. `MissingWidth` doesn't apply (PDF spec §9.6 — Type 3 uses
        // Widths exclusively), so 0 is the correct never-consulted default.
        missing_width: 0,
        code_to_gid: vec![],
        differences,
        // Type 3 glyphs are content-stream procedures, not a font program;
        // name→GID resolution does not apply.
        base_encoding: BaseEncoding::Standard,
        cid_encoding: None,
        type3: Some(Type3Data {
            glyphs,
            font_matrix,
        }),
    }
}

/// Read a 6-element matrix from `dict[key]`.  Returns `None` if missing or malformed.
fn extract_matrix(dict: &Dictionary, key: &[u8]) -> Option<[f64; 6]> {
    let arr = dict.get(key)?.as_array()?;
    if arr.len() < 6 {
        return None;
    }
    let mut m = [0.0f64; 6];
    for (i, obj) in arr.iter().enumerate().take(6) {
        m[i] = match obj {
            Object::Real(r) => f64::from(*r),
            #[expect(
                clippy::cast_precision_loss,
                reason = "PDF matrix values are small integers"
            )]
            Object::Integer(n) => *n as f64,
            _ => return None,
        };
    }
    Some(m)
}

/// Parse the `Encoding` entry of a Type 0 font into a [`CMap`].
///
/// `Identity-H` and `Identity-V` are predefined `CMaps` where char code == CID.
/// We represent these as `None` (identity).  Any stream reference is parsed.
fn extract_type0_encoding_cmap(doc: &Document, dict: &Dictionary) -> Option<CMap> {
    let enc = dict.get(b"Encoding")?;
    match enc {
        // Named CMaps: Identity-H / Identity-V are identity (None = passthrough).
        Object::Name(n) if n == b"Identity-H" || n == b"Identity-V" => None,
        // Named non-identity CMaps (e.g. /GB-EUC-H) — we cannot parse these
        // without a CMap resource database.  Fall through to None and treat as
        // identity, which will produce wrong glyphs for CJK but not crash.
        Object::Name(n) => {
            log::warn!(
                "font: named CMap /{} is not Identity — text may render incorrectly",
                String::from_utf8_lossy(n)
            );
            None
        }
        // Stream reference — parse the CMap.
        Object::Reference(id) => {
            let obj = doc.get_object(*id).ok()?;
            let stream = obj.as_stream()?;
            let content = stream.decompressed_content().ok()?;
            let cmap = parse_cmap(&content);
            if cmap.is_none() {
                log::warn!("font: Type0 Encoding CMap stream could not be parsed");
            }
            cmap
        }
        _ => {
            log::warn!("font: Type0 Encoding is not a Name or Reference — treating as identity");
            None
        }
    }
}

/// Resolve the first entry of `DescendantFonts` to a dictionary.
///
/// The PDF spec requires exactly one descendant; we take the first and warn if
/// the array is absent or empty.
fn extract_descendant(doc: &Document, dict: &Dictionary) -> Option<Dictionary> {
    let Some(arr_obj) = dict.get(b"DescendantFonts") else {
        log::warn!("font: Type0 font has no DescendantFonts — cannot load face");
        return None;
    };
    match arr_obj {
        Object::Array(a) => {
            if a.is_empty() {
                log::warn!("font: Type0 DescendantFonts array is empty");
                return None;
            }
            super::resolve_dict(doc, &a[0])
        }
        Object::Reference(id) => {
            let obj = doc.get_object(*id).ok()?;
            let arr = obj.as_array()?;
            if arr.is_empty() {
                log::warn!("font: Type0 DescendantFonts array is empty");
                return None;
            }
            super::resolve_dict(doc, &arr[0])
        }
        _ => None,
    }
}

/// Extract embedded font bytes from the `FontDescriptor` of a `CIDFont` dict.
fn extract_bytes_from_descendant(doc: &Document, descendant: &Dictionary) -> Option<Vec<u8>> {
    // CIDFont kind determines file key priority.
    let kind = classify_kind(descendant);
    extract_bytes_with_kind(doc, descendant, kind)
}

/// Shared byte-extraction logic that works on any font dict (direct or descendant).
fn extract_bytes_with_kind(
    doc: &Document,
    dict: &Dictionary,
    kind: PdfFontKind,
) -> Option<Vec<u8>> {
    let fd = resolve_fd(doc, dict)?;
    let priority: &[&[u8]] = match kind {
        PdfFontKind::Type1 => &[b"FontFile", b"FontFile3"],
        _ => &[b"FontFile2", b"FontFile3", b"FontFile"],
    };
    for key in priority {
        if let Some(bytes) = read_stream(doc, &fd, key) {
            return Some(bytes);
        }
    }
    None
}

/// Parse the `CIDToGIDMap` stream from a `CIDFontType2` (TrueType) descendant.
///
/// The map is a byte array of 2-byte big-endian GIDs indexed by CID.
/// `/Identity` (or absent) means CID == GID; we represent that as `None`.
const MAX_CID_TO_GID_BYTES: usize = 65536 * 2;
fn extract_cid_to_gid(doc: &Document, descendant: &Dictionary) -> Option<Vec<u32>> {
    let obj = descendant.get(b"CIDToGIDMap")?;
    match obj {
        Object::Reference(id) => {
            let stream_obj = doc.get_object(*id).ok()?;
            let stream = stream_obj.as_stream()?;
            let bytes = stream.decompressed_content().ok()?;

            // A CIDToGIDMap covers CIDs 0..65535 at most → max 65536 × 2 = 131072 bytes.
            // Reject oversized maps to prevent memory exhaustion from malicious PDFs.
            if bytes.len() > MAX_CID_TO_GID_BYTES {
                log::warn!(
                    "font: CIDToGIDMap stream is {} bytes — exceeds {MAX_CID_TO_GID_BYTES} limit, ignoring",
                    bytes.len()
                );
                return None;
            }

            // CIDToGIDMap is pairs of bytes: GID[cid] = (bytes[2*cid] << 8) | bytes[2*cid+1].
            if bytes.len() % 2 != 0 {
                log::warn!(
                    "font: CIDToGIDMap stream length {} is odd — ignoring",
                    bytes.len()
                );
                return None;
            }
            let table: Vec<u32> = bytes
                .chunks_exact(2)
                .map(|pair| u32::from(pair[0]) << 8 | u32::from(pair[1]))
                .collect();
            Some(table)
        }
        _ => None,
    }
}

/// Hard cap on the number of CID→width entries materialised from one `/W`
/// array.  The range form `cfirst clast w` expands to `clast − cfirst + 1`
/// width slots; a hostile or corrupt PDF can chain many maximal ranges to
/// amplify a tiny `/W` array into gigabytes of `Vec<i32>`.  CIDs are 16-bit
/// (PDF §9.7.4.3 — at most 65 536 distinct CIDs), so a font can never
/// legitimately need more than this many slots; past it we stop appending
/// and warn rather than let the allocator OOM the process (fail loud-graceful,
/// not silent-hang).
const MAX_CID_WIDTH_ENTRIES: usize = 0x1_0000;

/// Resolve a single `/W` array element (possibly an indirect reference per
/// PDF §9.7.4.3 — *any* number in `[ c [w…] cfirst clast w … ]` may be
/// indirect) to its `Object`.  `doc.resolve` is a no-op pass-through for an
/// inline (non-reference) element and is depth/cycle-bounded, so a self- or
/// ring-referential `/W` element degrades to `None` (skip) rather than
/// hanging or panicking.
fn resolve_w_elem(doc: &Document, elem: &Object) -> Option<Arc<Object>> {
    doc.resolve(elem).ok()
}

/// Parse the `W` array body into `(first_cid, widths)` segments.
///
/// `w_arr` is the already-resolved top-level array.  Per PDF §9.7.4.3 the
/// grammar is `[ c [w1 w2 …]  cfirst clast w  … ]` and *every* element — the
/// leading CID, the nested width array and each of its elements, the range
/// `cfirst`/`clast`, and the range width — may itself be an indirect
/// reference.  Resolving only the top-level `/W` still drops widths for a PDF
/// that emits indirect *inner* elements, re-creating the "every glyph advances
/// a full em → scrambled body text" defect on a different input.
/// `resolve_w_elem` is a cycle-bounded no-op for inline elements, so inline-/W
/// PDFs are unaffected.
fn parse_w_segments(doc: &Document, w_arr: &[Object]) -> Vec<(u32, Vec<i32>)> {
    let mut segments: Vec<(u32, Vec<i32>)> = Vec::new();
    let mut materialised: usize = 0;
    let mut i = 0;

    while i < w_arr.len() {
        let Some(first_obj) = resolve_w_elem(doc, &w_arr[i]) else {
            i += 1;
            continue;
        };
        let Some(first_cid_raw) = first_obj.as_i64() else {
            i += 1;
            continue;
        };
        if first_cid_raw < 0 {
            log::warn!("font: W array has negative CID {first_cid_raw}, skipping entry");
            i += 1;
            continue;
        }
        #[expect(
            clippy::cast_sign_loss,
            clippy::cast_possible_truncation,
            reason = "first_cid_raw validated >= 0 and PDF CIDs fit in u32"
        )]
        let first_cid = first_cid_raw as u32;
        i += 1;

        if i >= w_arr.len() {
            break;
        }

        // The form discriminator (nested width array vs. range `last_cid`) may
        // itself be indirect, so resolve it before matching on its shape.
        let Some(second_obj) = resolve_w_elem(doc, &w_arr[i]) else {
            i += 1;
            continue;
        };
        match second_obj.as_ref() {
            // `first_cid [w0 w1 …]` form.  Each width may be indirect.
            Object::Array(ws) => {
                let widths: Vec<i32> = ws
                    .iter()
                    .map(|o| {
                        resolve_w_elem(doc, o)
                            .and_then(|r| r.as_i64())
                            .map_or(0, pdf_width_to_i32)
                    })
                    .collect();
                if materialised.saturating_add(widths.len()) > MAX_CID_WIDTH_ENTRIES {
                    log::warn!(
                        "font: W array exceeds {MAX_CID_WIDTH_ENTRIES} CID width slots \
                         (16-bit CID space) — truncating remaining segments"
                    );
                    break;
                }
                materialised += widths.len();
                segments.push((first_cid, widths));
                i += 1;
            }
            // `first_cid last_cid w` form.
            Object::Integer(last_cid_raw) => {
                let last_cid_raw = *last_cid_raw;
                if last_cid_raw < 0 {
                    log::warn!(
                        "font: W array has negative last_cid {last_cid_raw}, skipping entry"
                    );
                    i += 1;
                    continue;
                }
                #[expect(
                    clippy::cast_sign_loss,
                    clippy::cast_possible_truncation,
                    reason = "last_cid_raw validated >= 0 and PDF CIDs fit in u32"
                )]
                let last_cid = last_cid_raw as u32;
                i += 1;
                if i >= w_arr.len() {
                    break;
                }
                let w = resolve_w_elem(doc, &w_arr[i])
                    .and_then(|r| r.as_i64())
                    .map_or(0, pdf_width_to_i32);
                i += 1;

                if last_cid >= first_cid && last_cid - first_cid < 0x1_0000 {
                    let count = (last_cid - first_cid + 1) as usize;
                    if materialised.saturating_add(count) > MAX_CID_WIDTH_ENTRIES {
                        log::warn!(
                            "font: W array exceeds {MAX_CID_WIDTH_ENTRIES} CID width slots \
                             (16-bit CID space) — truncating remaining segments"
                        );
                        break;
                    }
                    materialised += count;
                    segments.push((first_cid, vec![w; count]));
                } else {
                    log::warn!(
                        "font: W array degenerate CID range {first_cid}–{last_cid}, skipping"
                    );
                }
            }
            _ => {
                i += 1;
            }
        }
    }

    segments
}

/// Parse `DW` and `W` from a `CIDFont` dictionary.
///
/// Returns `(default_width, segments)` where segments match the PDF `W` format:
/// `[first_cid, [w0, w1, …]]` (the `c [w...]` variant only; the `c1 c2 w`
/// range variant is also handled).
fn extract_cid_widths(doc: &Document, dict: &Dictionary) -> (i32, Vec<(u32, Vec<i32>)>) {
    // `DW`/`W` may be indirect references; MuPDF's `pdf_dict_get` resolves them
    // transparently (source_pdf/pdf-font.c load_cid_font), so we must too — an
    // unresolved `/W 119 0 R` would otherwise drop every per-CID width and make
    // all glyphs advance by `DW` (1000/em), spreading text far too wide.
    let dw = dict
        .get(b"DW")
        .and_then(|o| doc.resolve(o).ok())
        .and_then(|o| o.as_i64())
        .map_or(1000, pdf_width_to_i32);

    let Some(w_obj) = dict.get(b"W").and_then(|o| doc.resolve(o).ok()) else {
        return (dw, vec![]);
    };
    let Some(w_arr) = w_obj.as_array() else {
        return (dw, vec![]);
    };

    (dw, parse_w_segments(doc, w_arr))
}

/// Parse the `Differences` array from a dictionary-form `Encoding` object.
///
/// The array has the form `[base_code /Name /Name ... base_code /Name ...]`.
/// Each integer resets the current char code; each name assigns a glyph name
/// to that code and advances the counter.  Codes outside `[0, 255]` are ignored.
fn parse_differences(dict: &Dictionary) -> Box<[Option<Box<str>>; 256]> {
    let mut table = empty_differences();

    let Some(arr_obj) = dict.get(b"Differences") else {
        return table;
    };
    let Some(arr) = arr_obj.as_array() else {
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

    /// Build a `FontDescriptor` with explicit widths for testing the
    /// `width_for_code` lookup paths.  The fields not exercised by the
    /// lookup are filled with stubs.
    fn descriptor_with_widths(
        first_char: u32,
        widths: Vec<i32>,
        missing_width: i32,
    ) -> FontDescriptor {
        FontDescriptor {
            kind: PdfFontKind::Type1,
            bytes: None,
            widths,
            first_char,
            missing_width,
            code_to_gid: vec![],
            differences: empty_differences(),
            base_encoding: BaseEncoding::Standard,
            cid_encoding: None,
            type3: None,
        }
    }

    #[test]
    fn width_for_code_in_range() {
        let d = descriptor_with_widths(32, vec![278, 500, 556], 0);
        assert_eq!(d.width_for_code(32), Some(278));
        assert_eq!(d.width_for_code(33), Some(500));
        assert_eq!(d.width_for_code(34), Some(556));
    }

    #[test]
    fn width_for_code_above_range_returns_missing_width() {
        let d = descriptor_with_widths(32, vec![278, 500, 556], 250);
        // 35 is just past LastChar = first_char + len - 1 = 34.
        assert_eq!(d.width_for_code(35), Some(250));
        assert_eq!(d.width_for_code(255), Some(250));
    }

    #[test]
    fn width_for_code_below_range_returns_missing_width() {
        // PDF §9.2.4: below FirstChar is also "out of range" → MissingWidth.
        let d = descriptor_with_widths(32, vec![278, 500, 556], 250);
        assert_eq!(d.width_for_code(0), Some(250));
        assert_eq!(d.width_for_code(31), Some(250));
    }

    #[test]
    fn width_for_code_empty_widths_returns_none() {
        // Standard 14 fonts (Helvetica etc.) pre-PDF-1.5 can omit Widths;
        // None signals to the caller to fall back to the FreeType face.
        let d = descriptor_with_widths(0, vec![], 0);
        assert_eq!(d.width_for_code(65), None);
    }

    #[test]
    fn pdf_width_to_i32_in_range_pass_through() {
        assert_eq!(pdf_width_to_i32(0), 0);
        assert_eq!(pdf_width_to_i32(500), 500);
        assert_eq!(pdf_width_to_i32(-1000), -1000);
        assert_eq!(pdf_width_to_i32(i64::from(i32::MAX)), i32::MAX);
        assert_eq!(pdf_width_to_i32(i64::from(i32::MIN)), i32::MIN);
    }

    #[test]
    fn pdf_width_to_i32_saturates_on_overflow() {
        // Adversarial input must clamp, not wrap.  An i64::MAX width that
        // wraps with `as i32` would become -1, silently corrupting text.
        assert_eq!(pdf_width_to_i32(i64::MAX), i32::MAX);
        assert_eq!(pdf_width_to_i32(i64::MIN), i32::MIN);
        assert_eq!(pdf_width_to_i32(i64::from(i32::MAX) + 1), i32::MAX);
        assert_eq!(pdf_width_to_i32(i64::from(i32::MIN) - 1), i32::MIN);
    }

    #[test]
    fn extract_missing_width_inline_dict() {
        let fd = make_dict(&[(b"MissingWidth", Object::Integer(750))]);
        let parent = make_dict(&[(b"FontDescriptor", Object::Dictionary(fd))]);
        let doc = empty_doc();
        assert_eq!(extract_missing_width(&doc, &parent), 750);
    }

    #[test]
    fn extract_missing_width_absent_returns_zero() {
        // FontDescriptor sub-dict exists but no MissingWidth key.
        let fd = make_dict(&[]);
        let parent = make_dict(&[(b"FontDescriptor", Object::Dictionary(fd))]);
        let doc = empty_doc();
        assert_eq!(extract_missing_width(&doc, &parent), 0);
    }

    #[test]
    fn extract_missing_width_no_descriptor_returns_zero() {
        // Whole FontDescriptor key absent → default 0.
        let parent = make_dict(&[(b"Subtype", Object::Name(b"Type1".to_vec()))]);
        let doc = empty_doc();
        assert_eq!(extract_missing_width(&doc, &parent), 0);
    }

    #[test]
    fn width_for_code_zero_width_in_range_is_distinct_from_missing() {
        // A literal 0 in the Widths array means "glyph has no advance"
        // (zero-width joiner, combining marks). MUST NOT fall through to
        // MissingWidth — they're different signals.
        let d = descriptor_with_widths(32, vec![278, 0, 556], 999);
        assert_eq!(d.width_for_code(33), Some(0), "in-range 0 must be returned");
        assert_eq!(
            d.width_for_code(35),
            Some(999),
            "out-of-range must return MissingWidth, not 0"
        );
    }

    use crate::test_helpers::{empty_doc, make_doc_with_object};

    #[test]
    fn cid_widths_resolve_indirect_w_array() {
        // Real-world CIDFonts (Word/Acrobat output) emit `/W` as an indirect
        // reference (`/W 119 0 R`).  PDF §9.7.4.3: the value is a (possibly
        // indirect) array.  MuPDF reads it via `pdf_dict_get` which resolves
        // references transparently (source_pdf/pdf-font.c:1266).  Without
        // resolving, every per-CID width is dropped and all glyphs advance by
        // `DW` (1000/em), piling text into an unreadable scramble.
        let doc = make_doc_with_object("[ 3 3 250 5 [ 500 600 ] ]");
        let dict = make_dict(&[
            (b"DW", Object::Integer(1000)),
            (b"W", Object::Reference((2, 0))),
        ]);
        let (dw, segs) = extract_cid_widths(&doc, &dict);
        assert_eq!(dw, 1000);
        // `3 3 250` (range form) → CID 3 width 250; `5 [500 600]` → CIDs 5,6.
        assert_eq!(segs, vec![(3, vec![250]), (5, vec![500, 600])]);
    }

    #[test]
    fn cid_widths_resolve_indirect_dw() {
        // `/DW` may also be an indirect reference; it must resolve too, else
        // the default advance silently falls back to the 1000 hardcoded
        // default instead of the document's intended em width.
        let doc = make_doc_with_object("742");
        let dict = make_dict(&[(b"DW", Object::Reference((2, 0)))]);
        let (dw, segs) = extract_cid_widths(&doc, &dict);
        assert_eq!(dw, 742);
        assert!(segs.is_empty());
    }

    #[test]
    fn cid_widths_inline_w_array_unaffected() {
        // Inline (non-reference) `/W` arrays — the case for already-passing
        // CID PDFs — must keep working: `doc.resolve` is a transparent
        // pass-through for a non-reference Object, so this proves the indirect
        // fix does not regress the inline path.
        let doc = empty_doc();
        let dict = make_dict(&[
            (b"DW", Object::Integer(500)),
            (
                b"W",
                Object::Array(vec![
                    Object::Integer(10),
                    Object::Array(vec![Object::Integer(333), Object::Integer(444)]),
                ]),
            ),
        ]);
        let (dw, segs) = extract_cid_widths(&doc, &dict);
        assert_eq!(dw, 500);
        assert_eq!(segs, vec![(10, vec![333, 444])]);
    }

    #[test]
    fn cid_widths_resolve_indirect_inner_w_elements() {
        // PDF §9.7.4.3: *any* number in `/W` may be an indirect reference —
        // including the elements of the nested width array and the range
        // `last_cid`/`w`.  Resolving only the top-level `/W` and the leading
        // CID leaves an inner `5 0 R`/`6 0 R` width collapsing to 0-advance
        // (overlapping glyphs — the scramble defect on a different input).
        // This proves inner elements resolve too.
        //
        // obj 4 = `/W` array: `1 [ 5 0 R 6 0 R ]  20 7 0 R  8 0 R`
        //   → CID 1 widths [333, 444] (indirect), CIDs 20..21 width 555 from
        //     an indirect last_cid (obj 7 = 21) and indirect range width
        //     (obj 8 = 555).  Object ids MUST stay contiguous (4,5,6,7,8) so
        //     the single-subsection test xref stays valid.
        let doc = crate::test_helpers::make_doc_with_objects(&[
            (4, "[ 1 [ 5 0 R 6 0 R ] 20 7 0 R 8 0 R ]"),
            (5, "333"),
            (6, "444"),
            (7, "21"),
            (8, "555"),
        ]);
        let dict = make_dict(&[(b"W", Object::Reference((4, 0)))]);
        let (dw, segs) = extract_cid_widths(&doc, &dict);
        assert_eq!(dw, 1000, "DW absent → spec default");
        assert_eq!(
            segs,
            vec![(1, vec![333, 444]), (20, vec![555, 555])],
            "indirect inner width-array elements and indirect range last_cid/w \
             must resolve, not collapse to 0"
        );
    }

    #[test]
    fn cid_widths_cyclic_w_ref_is_loud_graceful() {
        // A self-referential `/W` (`/W 4 0 R; 4 0 obj 4 0 R`) is the
        // recursive-resolve DoS class.  `Document::resolve` is depth-bounded
        // (32) and returns an error on the cycle; `extract_cid_widths` maps
        // that to the empty-segment fallback (graceful — glyphs advance by DW,
        // never a panic, hang, or OOM).
        let doc = crate::test_helpers::make_doc_with_objects(&[(4, "4 0 R")]);
        let dict = make_dict(&[
            (b"DW", Object::Integer(600)),
            (b"W", Object::Reference((4, 0))),
        ]);
        let (dw, segs) = extract_cid_widths(&doc, &dict);
        assert_eq!(dw, 600, "DW still honoured when /W is a broken cycle");
        assert!(
            segs.is_empty(),
            "cyclic /W must degrade to no per-CID widths, not hang or panic"
        );
    }

    #[test]
    fn cid_widths_malformed_w_array_is_loud_graceful() {
        // Hostile/corrupt `/W`, parsed strictly left-to-right per the PDF
        // §9.7.4.3 grammar.  Sequence (with how the cursor consumes it):
        //   Name        → bad leading element, skip 1
        //   5 [ -7 9 ]  → well-formed width-array form: CID 5 widths [-7, 9]
        //                 (negative width saturates via pdf_width_to_i32, it is
        //                 NOT silently zeroed or panicked)
        //   50 40 200   → range form, cfirst(50) > clast(40) → degenerate,
        //                 logged + skipped, all 3 consumed
        //   99          → dangling leading CID with no value → loop breaks
        // No panic, no silent corruption of the one valid entry.
        let doc = empty_doc();
        let dict = make_dict(&[(
            b"W",
            Object::Array(vec![
                Object::Name(b"NotACid".to_vec()),
                Object::Integer(5),
                Object::Array(vec![Object::Integer(-7), Object::Integer(9)]),
                Object::Integer(50),
                Object::Integer(40),
                Object::Integer(200),
                Object::Integer(99),
            ]),
        )]);
        let (dw, segs) = extract_cid_widths(&doc, &dict);
        assert_eq!(dw, 1000);
        assert_eq!(segs, vec![(5, vec![-7, 9])]);
    }

    #[test]
    fn cid_widths_oversized_range_is_capped() {
        // A chain of maximal ranges must not amplify a tiny `/W` into
        // gigabytes of Vec<i32>.  Two back-to-back full-16-bit ranges request
        // 2 × 65 536 slots; the second crosses MAX_CID_WIDTH_ENTRIES and is
        // truncated (loud-graceful) rather than OOMing.
        let doc = empty_doc();
        let dict = make_dict(&[(
            b"W",
            Object::Array(vec![
                Object::Integer(0),
                Object::Integer(0xFFFF),
                Object::Integer(500),
                Object::Integer(0x1_0000),
                Object::Integer(0x1_FFFF),
                Object::Integer(600),
            ]),
        )]);
        let (_dw, segs) = extract_cid_widths(&doc, &dict);
        assert_eq!(segs.len(), 1, "second range exceeds the entry cap");
        assert_eq!(segs[0].0, 0);
        assert_eq!(segs[0].1.len(), 0x1_0000);
    }

    #[test]
    fn no_embedded_bytes_returns_none() {
        // A minimal font dict with no FontDescriptor.
        let d = make_dict(&[(b"Subtype", Object::Name(b"Type1".to_vec()))]);
        // We need a Document to call extract_bytes, but for simple cases without
        // embedded streams (no FontDescriptor key) the result is None regardless
        // of what's in the document.
        let doc = empty_doc();
        assert!(extract_bytes(&doc, &d).is_none());
    }
}
