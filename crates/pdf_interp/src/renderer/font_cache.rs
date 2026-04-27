//! Per-page font cache.
//!
//! [`FontCache`] maps a PDF font resource name (e.g. `b"F1"`) to a loaded
//! [`FontFace`].  Faces are loaded on first use during rendering, then reused
//! for all subsequent glyphs in the same page.
//!
//! When a font has no embedded bytes the cache falls back to a system font
//! (Liberation Sans or Helvetica) located via a compile-time search path.
//!
//! # Thread safety
//!
//! [`FontCache`] is not `Send`.  One instance is created per [`PageRenderer`]
//! (one per thread when pages are rendered in parallel).  The shared
//! [`FontEngine`] is locked only during the face-load call.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::OnceLock;

use font::{
    cache::GlyphCache,
    engine::{FaceParams, SharedEngine},
    face::FontFace,
    hinting::FontKind,
};

use crate::resources::font::{FontDescriptor, PdfFontKind};

// ── Fallback font discovery ───────────────────────────────────────────────────

/// Candidate paths searched in order on Linux/macOS/Windows for a sans-serif
/// substitute when a PDF font is not embedded.
const FALLBACK_CANDIDATES: &[&str] = &[
    // Linux (fonts-liberation or ttf-liberation)
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/liberation/LiberationSans-Regular.ttf",
    // macOS
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial.ttf",
    // Windows (Wine / Proton)
    "C:/Windows/Fonts/arial.ttf",
];

/// Locate and cache the fallback font path.  Returns `None` if none of the
/// candidates exist.
fn fallback_font_path() -> Option<&'static str> {
    static CACHE: OnceLock<Option<&'static str>> = OnceLock::new();
    *CACHE.get_or_init(|| {
        FALLBACK_CANDIDATES
            .iter()
            .find(|&&p| PathBuf::from(p).exists())
            .copied()
    })
}

// ── Kind conversion ───────────────────────────────────────────────────────────

const fn to_font_kind(k: PdfFontKind) -> FontKind {
    match k {
        PdfFontKind::Type1 => FontKind::Type1,
        PdfFontKind::TrueType => FontKind::TrueType,
        PdfFontKind::Other => FontKind::Other,
    }
}

// ── Cache ─────────────────────────────────────────────────────────────────────

/// Per-page cache of loaded font faces.
pub struct FontCache {
    engine: SharedEngine,
    /// Shared process-wide glyph bitmap cache.
    glyph_cache: GlyphCache,
    /// Map from (PDF resource name, four f64-as-u64 bits of the Trm 2×2 matrix) to
    /// loaded face.  The full matrix is the key because the same font can appear at
    /// different sizes and skew angles within a single page.
    faces: HashMap<(Vec<u8>, [u64; 4]), Option<FontFace>>,
}

impl FontCache {
    /// Create a new cache backed by the given engine.
    #[must_use]
    pub fn new(engine: SharedEngine, glyph_cache: GlyphCache) -> Self {
        Self {
            engine,
            glyph_cache,
            faces: HashMap::new(),
        }
    }

    /// Return a mutable reference to the glyph bitmap cache.
    pub const fn glyph_cache_mut(&mut self) -> &mut GlyphCache {
        &mut self.glyph_cache
    }

    /// Return a reference to the [`FontFace`] for `name`, loading it from
    /// `descriptor` on first use.
    ///
    /// `trm` is the 2×2 submatrix of the **text rendering matrix** in device
    /// pixels: `font_size × Tm[2×2] × CTM[2×2]`.  It determines both the
    /// rendered pixel size and any shear/rotation.
    ///
    /// Returns `None` if the face cannot be loaded.  The caller should skip
    /// glyph rendering — missing glyphs are silently omitted.
    pub fn get_or_load(
        &mut self,
        name: &[u8],
        descriptor: &FontDescriptor,
        trm: [f64; 4],
    ) -> Option<&FontFace> {
        // Key encodes the full 2×2 Trm matrix as bit patterns: same font name at
        // different sizes or skews needs distinct FreeType faces.
        let mat_key = trm.map(f64::to_bits);
        let key = (name.to_vec(), mat_key);
        // Can't use the entry API here: `self.load_face` borrows `&self` while
        // `self.faces.entry(..)` holds a mutable borrow of `self.faces`.
        // Use contains_key + insert instead — the double lookup is one pointer
        // comparison per hit and acceptable given the per-page cache size.
        if !self.faces.contains_key(&key) {
            let face = self.load_face(descriptor, trm);
            if face.is_none() {
                log::warn!(
                    "font_cache: could not load face for /{} trm={trm:?}",
                    String::from_utf8_lossy(name)
                );
            }
            self.faces.insert(key.clone(), face);
        }
        self.faces.get(&key)?.as_ref()
    }

    fn load_face(&self, desc: &FontDescriptor, trm: [f64; 4]) -> Option<FontFace> {
        // Pre-flight: reject degenerate matrices before paying the mutex cost.
        // The y-column magnitude (hypot of indices 2 & 3) is the pixel size used by
        // FontFace::new; a value < 1.0 or non-finite produces an unusable face.
        if !trm_pixel_size_valid(trm) {
            log::debug!(
                "font_cache: degenerate trm (size={:.1}), skipping face load",
                f64::hypot(trm[2], trm[3])
            );
            return None;
        }

        // code_to_gid from the descriptor is populated for CID/Type0 fonts.
        // For simple fonts with Differences, we resolve after loading the face.
        let has_differences = desc.differences.iter().any(Option::is_some);

        let params = FaceParams {
            kind: to_font_kind(desc.kind),
            code_to_gid: desc.code_to_gid.clone(),
            mat: trm,
            // text_mat and mat are the same here: we use a single unified Trm.
            text_mat: trm,
        };

        let mut eng = self
            .engine
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);

        let mut face = if let Some(bytes) = &desc.bytes {
            eng.load_memory_face(bytes.clone(), 0, params)
                .inspect_err(|e| log::debug!("font_cache: memory face error: {e}"))
                .ok()?
        } else {
            let path = fallback_font_path()?;
            eng.load_file_face(path, 0, params)
                .inspect_err(|e| log::debug!("font_cache: file face error ({path}): {e}"))
                .ok()?
        };

        // If the Differences array supplied glyph names, resolve them to GIDs
        // using FreeType's active charmap via Unicode codepoint lookup.
        if has_differences {
            let code_to_gid = resolve_differences_to_gid(&face, &desc.differences);
            face.code_to_gid = code_to_gid;
        }

        Some(face)
    }
}

// ── Shared size-validation ────────────────────────────────────────────────────

/// Return `true` if the y-column magnitude of `trm` (used by `FontFace::new`
/// as the nominal pixel size) is a finite value ≥ 1.0.
///
/// This mirrors the validation inside `FontFace::new` and is used as an early
/// rejection to avoid the mutex lock cost for obviously degenerate matrices.
pub(crate) fn trm_pixel_size_valid(trm: [f64; 4]) -> bool {
    let size = f64::hypot(trm[2], trm[3]);
    size.is_finite() && size >= 1.0
}

// ── Differences → GID resolution ─────────────────────────────────────────────

/// Build a 256-entry `code_to_gid` table from a `Differences` glyph-name array.
///
/// For each char code that has a named override, look up the Adobe standard
/// glyph name → Unicode codepoint, then call `FreeType`'s `get_char_index` to
/// get the GID.  Codes without an override use `char_code` directly as the
/// char index (`FreeType`'s Unicode charmap fallback).
fn resolve_differences_to_gid(face: &FontFace, differences: &[Option<Box<str>>; 256]) -> Vec<u32> {
    let mut table: Vec<u32> = (0u32..256).collect();
    for (code, entry) in differences.iter().enumerate() {
        if let Some(name) = entry
            && let Some(unicode) = adobe_glyph_to_unicode(name)
        {
            table[code] = face.raw_get_char_index(unicode);
        }
        // Names absent from the Adobe list keep table[code] = code (identity).
    }
    table
}

/// Map an Adobe glyph name to its Unicode codepoint.
///
/// Covers the standard Latin-1 supplement, common punctuation, and symbols
/// from the Adobe Glyph List (AGL).
///
/// Implemented as a sorted static table + binary search rather than a `match`
/// expression.  A 345-arm `match` compiles to a LLVM decision tree that costs
/// hundreds of MiB of working memory per codegen unit in debug mode, which
/// caused the build to OOM on this machine.  A `static` array is pure data —
/// zero LLVM IR generated, O(log n) lookup.
#[expect(
    clippy::too_many_lines,
    reason = "static lookup table — 345 AGL entries, all data and no logic"
)]
fn adobe_glyph_to_unicode(name: &str) -> Option<u32> {
    // Fast path: names like "uni0041" or "u0041" encode the codepoint directly.
    if let Some(hex) = name.strip_prefix("uni").filter(|s| s.len() == 4) {
        return u32::from_str_radix(hex, 16).ok();
    }
    if let Some(hex) = name
        .strip_prefix('u')
        .filter(|s| (4..=6).contains(&s.len()) && s.chars().all(|c| c.is_ascii_hexdigit()))
    {
        return u32::from_str_radix(hex, 16).ok();
    }

    // Sorted by glyph name so binary_search_by_key works.
    #[expect(
        clippy::items_after_statements,
        reason = "static table placed after fast-path returns to keep it close to its only use"
    )]
    static AGL: &[(&str, u32)] = &[
        ("A", 0x0041),
        ("AE", 0x00C6),
        ("Aacute", 0x00C1),
        ("Abreve", 0x0102),
        ("Acircumflex", 0x00C2),
        ("Adieresis", 0x00C4),
        ("Agrave", 0x00C0),
        ("Alpha", 0x0391),
        ("Amacron", 0x0100),
        ("Aogonek", 0x0104),
        ("Aring", 0x00C5),
        ("Atilde", 0x00C3),
        ("B", 0x0042),
        ("Beta", 0x0392),
        ("C", 0x0043),
        ("Cacute", 0x0106),
        ("Ccaron", 0x010C),
        ("Ccedilla", 0x00C7),
        ("Ccircumflex", 0x0108),
        ("Cdotaccent", 0x010A),
        ("Chi", 0x03A7),
        ("D", 0x0044),
        ("Dcaron", 0x010E),
        ("Dcroat", 0x0110),
        ("Delta", 0x0394),
        ("E", 0x0045),
        ("Eacute", 0x00C9),
        ("Ebreve", 0x0114),
        ("Ecaron", 0x011A),
        ("Ecircumflex", 0x00CA),
        ("Edieresis", 0x00CB),
        ("Edotaccent", 0x0116),
        ("Egrave", 0x00C8),
        ("Emacron", 0x0112),
        ("Eogonek", 0x0118),
        ("Epsilon", 0x0395),
        ("Eta", 0x0397),
        ("Eth", 0x00D0),
        ("Euro", 0x20AC),
        ("F", 0x0046),
        ("G", 0x0047),
        ("Gamma", 0x0393),
        ("Gcircumflex", 0x011C),
        ("H", 0x0048),
        ("I", 0x0049),
        ("Iacute", 0x00CD),
        ("Icircumflex", 0x00CE),
        ("Idieresis", 0x00CF),
        ("Igrave", 0x00CC),
        ("Iota", 0x0399),
        ("J", 0x004A),
        ("K", 0x004B),
        ("Kappa", 0x039A),
        ("L", 0x004C),
        ("Lambda", 0x039B),
        ("Lslash", 0x0141),
        ("M", 0x004D),
        ("Mu", 0x039C),
        ("N", 0x004E),
        ("Nacute", 0x0143),
        ("Ncaron", 0x0147),
        ("Ntilde", 0x00D1),
        ("Nu", 0x039D),
        ("O", 0x004F),
        ("OE", 0x0152),
        ("Oacute", 0x00D3),
        ("Ocircumflex", 0x00D4),
        ("Odieresis", 0x00D6),
        ("Ograve", 0x00D2),
        ("Omega", 0x03A9),
        ("Omicron", 0x039F),
        ("Oslash", 0x00D8),
        ("Otilde", 0x00D5),
        ("P", 0x0050),
        ("Phi", 0x03A6),
        ("Pi", 0x03A0),
        ("Psi", 0x03A8),
        ("Q", 0x0051),
        ("R", 0x0052),
        ("Racute", 0x0154),
        ("Rcaron", 0x0158),
        ("Rho", 0x03A1),
        ("S", 0x0053),
        ("Sacute", 0x015A),
        ("Scaron", 0x0160),
        ("Scedilla", 0x015E),
        ("Scircumflex", 0x015C),
        ("Sigma", 0x03A3),
        ("T", 0x0054),
        ("Tau", 0x03A4),
        ("Tcaron", 0x0164),
        ("Theta", 0x0398),
        ("Thorn", 0x00DE),
        ("U", 0x0055),
        ("Uacute", 0x00DA),
        ("Ucircumflex", 0x00DB),
        ("Udieresis", 0x00DC),
        ("Ugrave", 0x00D9),
        ("Uhungarumlaut", 0x0170),
        ("Uogonek", 0x0172),
        ("Upsilon", 0x03A5),
        ("Uring", 0x016E),
        ("V", 0x0056),
        ("W", 0x0057),
        ("X", 0x0058),
        ("Xi", 0x039E),
        ("Y", 0x0059),
        ("Yacute", 0x00DD),
        ("Ydieresis", 0x0178),
        ("Z", 0x005A),
        ("Zacute", 0x0179),
        ("Zcaron", 0x017D),
        ("Zdotaccent", 0x017B),
        ("Zeta", 0x0396),
        ("a", 0x0061),
        ("aacute", 0x00E1),
        ("abreve", 0x0103),
        ("acircumflex", 0x00E2),
        ("acute", 0x00B4),
        ("adieresis", 0x00E4),
        ("ae", 0x00E6),
        ("agrave", 0x00E0),
        ("alpha", 0x03B1),
        ("amacron", 0x0101),
        ("ampersand", 0x0026),
        ("aogonek", 0x0105),
        ("approxequal", 0x2248),
        ("aring", 0x00E5),
        ("asciicircum", 0x005E),
        ("asciitilde", 0x007E),
        ("asterisk", 0x002A),
        ("at", 0x0040),
        ("atilde", 0x00E3),
        ("b", 0x0062),
        ("backslash", 0x005C),
        ("bar", 0x007C),
        ("beta", 0x03B2),
        ("braceleft", 0x007B),
        ("braceright", 0x007D),
        ("bracketleft", 0x005B),
        ("bracketright", 0x005D),
        ("breve", 0x02D8),
        ("brokenbar", 0x00A6),
        ("bullet", 0x2022),
        ("c", 0x0063),
        ("cacute", 0x0107),
        ("caron", 0x02C7),
        ("ccaron", 0x010D),
        ("ccedilla", 0x00E7),
        ("ccircumflex", 0x0109),
        ("cdotaccent", 0x010B),
        ("cedilla", 0x00B8),
        ("cent", 0x00A2),
        ("chi", 0x03C7),
        ("circumflex", 0x02C6),
        ("colon", 0x003A),
        ("comma", 0x002C),
        ("copyright", 0x00A9),
        ("currency", 0x00A4),
        ("d", 0x0064),
        ("dagger", 0x2020),
        ("daggerdbl", 0x2021),
        ("dcaron", 0x010F),
        ("dcroat", 0x0111),
        ("degree", 0x00B0),
        ("delta", 0x03B4),
        ("dieresis", 0x00A8),
        ("divide", 0x00F7),
        ("dollar", 0x0024),
        ("dotaccent", 0x02D9),
        ("e", 0x0065),
        ("eacute", 0x00E9),
        ("ebreve", 0x0115),
        ("ecaron", 0x011B),
        ("ecircumflex", 0x00EA),
        ("edieresis", 0x00EB),
        ("edotaccent", 0x0117),
        ("egrave", 0x00E8),
        ("eight", 0x0038),
        ("ellipsis", 0x2026),
        ("emacron", 0x0113),
        ("emdash", 0x2014),
        ("endash", 0x2013),
        ("eogonek", 0x0119),
        ("epsilon", 0x03B5),
        ("equal", 0x003D),
        ("eta", 0x03B7),
        ("eth", 0x00F0),
        ("exclam", 0x0021),
        ("exclamdown", 0x00A1),
        ("f", 0x0066),
        ("fi", 0xFB01),
        ("five", 0x0035),
        ("fl", 0xFB02),
        ("four", 0x0034),
        ("fraction", 0x2044),
        ("g", 0x0067),
        ("gamma", 0x03B3),
        ("gcircumflex", 0x011D),
        ("germandbls", 0x00DF),
        ("grave", 0x0060),
        ("greater", 0x003E),
        ("greaterequal", 0x2265),
        ("guillemotleft", 0x00AB),
        ("guillemotright", 0x00BB),
        ("guilsinglleft", 0x2039),
        ("guilsinglright", 0x203A),
        ("h", 0x0068),
        ("hungarumlaut", 0x02DD),
        ("hyphen", 0x002D),
        ("i", 0x0069),
        ("iacute", 0x00ED),
        ("icircumflex", 0x00EE),
        ("idieresis", 0x00EF),
        ("igrave", 0x00EC),
        ("infinity", 0x221E),
        ("integral", 0x222B),
        ("iota", 0x03B9),
        ("j", 0x006A),
        ("k", 0x006B),
        ("kappa", 0x03BA),
        ("l", 0x006C),
        ("lambda", 0x03BB),
        ("less", 0x003C),
        ("lessequal", 0x2264),
        ("logicalnot", 0x00AC),
        ("lozenge", 0x25CA),
        ("lslash", 0x0142),
        ("m", 0x006D),
        ("macron", 0x00AF),
        ("minus", 0x2212),
        ("mu", 0x00B5),
        ("multiply", 0x00D7),
        ("n", 0x006E),
        ("nacute", 0x0144),
        ("ncaron", 0x0148),
        ("notequal", 0x2260),
        ("nine", 0x0039),
        ("ntilde", 0x00F1),
        ("nu", 0x03BD),
        ("numbersign", 0x0023),
        ("o", 0x006F),
        ("oacute", 0x00F3),
        ("ocircumflex", 0x00F4),
        ("odieresis", 0x00F6),
        ("oe", 0x0153),
        ("one", 0x0031),
        ("ograve", 0x00F2),
        ("ogonek", 0x02DB),
        ("omicron", 0x03BF),
        ("onehalf", 0x00BD),
        ("onequarter", 0x00BC),
        ("onesuperior", 0x00B9),
        ("ordmasculine", 0x00BA),
        ("ordfeminine", 0x00AA),
        ("oslash", 0x00F8),
        ("otilde", 0x00F5),
        ("p", 0x0070),
        ("paragraph", 0x00B6),
        ("parenleft", 0x0028),
        ("parenright", 0x0029),
        ("partialdiff", 0x2202),
        ("percent", 0x0025),
        ("period", 0x002E),
        ("periodcentered", 0x00B7),
        ("perthousand", 0x2030),
        ("phi", 0x03C6),
        ("pi", 0x03C0),
        ("plus", 0x002B),
        ("plusminus", 0x00B1),
        ("product", 0x220F),
        ("psi", 0x03C8),
        ("q", 0x0071),
        ("question", 0x003F),
        ("questiondown", 0x00BF),
        ("quotedbl", 0x0022),
        ("quotedblbase", 0x201E),
        ("quotedblleft", 0x201C),
        ("quotedblright", 0x201D),
        ("quoteleft", 0x2018),
        ("quoteright", 0x2019),
        ("quotesinglbase", 0x201A),
        ("quotesingle", 0x0027),
        ("r", 0x0072),
        ("racute", 0x0155),
        ("radical", 0x221A),
        ("rcaron", 0x0159),
        ("registered", 0x00AE),
        ("rho", 0x03C1),
        ("ring", 0x02DA),
        ("s", 0x0073),
        ("sacute", 0x015B),
        ("scaron", 0x0161),
        ("scedilla", 0x015F),
        ("scircumflex", 0x015D),
        ("section", 0x00A7),
        ("semicolon", 0x003B),
        ("seven", 0x0037),
        ("six", 0x0036),
        ("sigma", 0x03C3),
        ("sigma1", 0x03C2),
        ("slash", 0x002F),
        ("space", 0x0020),
        ("sterling", 0x00A3),
        ("summation", 0x2211),
        ("t", 0x0074),
        ("tau", 0x03C4),
        ("tcaron", 0x0165),
        ("theta", 0x03B8),
        ("thorn", 0x00FE),
        ("three", 0x0033),
        ("threequarters", 0x00BE),
        ("threesuperior", 0x00B3),
        ("tilde", 0x02DC),
        ("two", 0x0032),
        ("twosuperior", 0x00B2),
        ("u", 0x0075),
        ("uacute", 0x00FA),
        ("ucircumflex", 0x00FB),
        ("udieresis", 0x00FC),
        ("ugrave", 0x00F9),
        ("uhungarumlaut", 0x0171),
        ("underscore", 0x005F),
        ("uogonek", 0x0173),
        ("upsilon", 0x03C5),
        ("uring", 0x016F),
        ("v", 0x0076),
        ("w", 0x0077),
        ("x", 0x0078),
        ("xi", 0x03BE),
        ("y", 0x0079),
        ("yacute", 0x00FD),
        ("ydieresis", 0x00FF),
        ("yen", 0x00A5),
        ("z", 0x007A),
        ("zacute", 0x017A),
        ("zcaron", 0x017E),
        ("zdotaccent", 0x017C),
        ("zero", 0x0030),
        ("zeta", 0x03B6),
    ];

    AGL.binary_search_by_key(&name, |&(k, _)| k)
        .ok()
        .map(|i| AGL[i].1)
}
