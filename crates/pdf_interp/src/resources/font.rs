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
    /// Glyph-index table: `code_to_gid[char_code]` → FT glyph index.
    /// Empty means the identity map (`char_code` == glyph index).
    pub code_to_gid: Vec<u32>,
    /// Per-character glyph name overrides from the PDF `Encoding/Differences`
    /// array.  Index = char code (0–255); `None` = inherit from base encoding.
    /// Used by [`FontCache`] to resolve names → Unicode → GID via `FreeType`'s
    /// active charmap at face-load time.
    pub differences: Box<[Option<Box<str>>; 256]>,
    /// CID encoding for Type 0 composite fonts.  `Some` when `Subtype` is
    /// `Type0`; `None` for all simple fonts (Type1, TrueType, `MMType1`).
    pub cid_encoding: Option<CidEncoding>,
    /// Type 3 glyph data.  `Some` only when `Subtype` is `Type3`; `None` for
    /// all FreeType-backed fonts.  When `Some`, `bytes` is `None` and `kind`
    /// is `Other` (Type 3 has no font program; glyphs are content streams).
    pub type3: Option<Type3Data>,
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
    let (code_to_gid, differences) = extract_encoding(doc, dict, kind);

    FontDescriptor {
        kind,
        bytes,
        widths,
        first_char,
        code_to_gid,
        differences,
        cid_encoding: None,
        type3: None,
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
        .and_then(Object::as_array)
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

/// Extract encoding information from a simple (non-Type0) font dictionary.
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
    let Some(encoding) = dict.get(b"Encoding") else {
        return (vec![], empty_differences());
    };

    match encoding {
        Object::Reference(id) => {
            let diffs = doc
                .get_dictionary(*id)
                .ok()
                .map_or_else(empty_differences, |d| parse_differences(&d));
            (vec![], diffs)
        }
        Object::Dictionary(d) => (vec![], parse_differences(d)),
        // Named standard encoding or anything else: `FreeType` handles char → GID.
        _ => (vec![], empty_differences()),
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
        .map_or((1000, vec![]), extract_cid_widths);

    // 6. Determine font kind from CIDFont Subtype.
    let kind = descendant
        .as_ref()
        .map_or(PdfFontKind::Other, classify_kind);

    FontDescriptor {
        kind,
        bytes,
        widths: vec![],
        first_char: 0,
        code_to_gid: vec![],
        differences: empty_differences(),
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
        code_to_gid: vec![],
        differences,
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

/// Parse `DW` and `W` from a `CIDFont` dictionary.
///
/// Returns `(default_width, segments)` where segments match the PDF `W` format:
/// `[first_cid, [w0, w1, …]]` (the `c [w...]` variant only; the `c1 c2 w`
/// range variant is also handled).
fn extract_cid_widths(dict: &Dictionary) -> (i32, Vec<(u32, Vec<i32>)>) {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "DW is a glyph advance in design units, always small; safe to i32"
    )]
    let dw = dict.get_i64(b"DW").map_or(1000, |v| v as i32);

    let Some(w_arr) = dict.get(b"W").and_then(Object::as_array) else {
        return (dw, vec![]);
    };

    let mut segments: Vec<(u32, Vec<i32>)> = Vec::new();
    let mut i = 0;

    while i < w_arr.len() {
        // Each segment starts with a non-negative CID integer.
        let Some(first_cid_raw) = w_arr[i].as_i64() else {
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

        match &w_arr[i] {
            // `first_cid [w0 w1 …]` form.
            Object::Array(ws) => {
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "PDF glyph widths are in [0, 4096]"
                )]
                let widths: Vec<i32> = ws
                    .iter()
                    .map(|o| o.as_i64().map_or(0, |v| v as i32))
                    .collect();
                segments.push((first_cid, widths));
                i += 1;
            }
            // `first_cid last_cid w` form.
            Object::Integer(last_cid_raw) => {
                if *last_cid_raw < 0 {
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
                let last_cid = *last_cid_raw as u32;
                i += 1;
                if i >= w_arr.len() {
                    break;
                }
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "PDF glyph width in [0, 4096]"
                )]
                let w = w_arr[i].as_i64().map_or(0, |v| v as i32);
                i += 1;

                if last_cid >= first_cid && last_cid - first_cid < 0x1_0000 {
                    let count = (last_cid - first_cid + 1) as usize;
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

    (dw, segments)
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

    /// Build a minimal valid PDF byte-stream with one empty page tree and
    /// parse it via [`Document::from_bytes_owned`].  Used by tests that need a
    /// real `Document` reference but do not depend on its contents.
    fn empty_doc() -> Document {
        let header = "%PDF-1.4\n";
        let obj1 = "1 0 obj\n<</Type /Catalog /Pages 2 0 R>>\nendobj\n";
        let obj2 = "2 0 obj\n<</Type /Pages /Kids [] /Count 0>>\nendobj\n";
        let off1 = header.len();
        let off2 = off1 + obj1.len();
        let xref_start = off2 + obj2.len();
        let xref = format!(
            "xref\n0 3\n0000000000 65535 f\r\n{off1:010} 00000 n\r\n{off2:010} 00000 n\r\n",
        );
        let trailer = format!("trailer\n<</Size 3 /Root 1 0 R>>\nstartxref\n{xref_start}\n%%EOF",);
        let bytes = format!("{header}{obj1}{obj2}{xref}{trailer}").into_bytes();
        Document::from_bytes_owned(bytes).expect("test PDF parse")
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
