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
fn resolve_differences_to_gid(
    face: &FontFace,
    differences: &[Option<Box<str>>; 256],
) -> Vec<u32> {
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
/// from the Adobe Glyph List (AGL).  Full list at:
/// <https://github.com/adobe-type-tools/agl-aglfn/blob/master/aglfn.txt>
#[expect(clippy::too_many_lines, reason = "static glyph name lookup table — no sensible way to split this")]
fn adobe_glyph_to_unicode(name: &str) -> Option<u32> {
    // Fast path: names like "uni0041" or "u0041" encode the codepoint directly.
    if let Some(hex) = name.strip_prefix("uni").filter(|s| s.len() == 4) {
        return u32::from_str_radix(hex, 16).ok();
    }
    if let Some(hex) = name.strip_prefix('u').filter(|s| {
        (4..=6).contains(&s.len()) && s.chars().all(|c| c.is_ascii_hexdigit())
    }) {
        return u32::from_str_radix(hex, 16).ok();
    }

    // Static lookup table: Adobe glyph name → Unicode codepoint.
    // Sorted so a binary search could be used, but linear is fine for 256 max lookups.
    let cp: u32 = match name {
        // Basic Latin
        "space" => 0x0020, "exclam" => 0x0021, "quotedbl" => 0x0022,
        "numbersign" => 0x0023, "dollar" => 0x0024, "percent" => 0x0025,
        "ampersand" => 0x0026, "quoteright" => 0x2019, "quotesingle" => 0x0027,
        "parenleft" => 0x0028, "parenright" => 0x0029, "asterisk" => 0x002A,
        "plus" => 0x002B, "comma" => 0x002C, "hyphen" => 0x002D,
        "period" => 0x002E, "slash" => 0x002F,
        "zero" => 0x0030, "one" => 0x0031, "two" => 0x0032, "three" => 0x0033,
        "four" => 0x0034, "five" => 0x0035, "six" => 0x0036, "seven" => 0x0037,
        "eight" => 0x0038, "nine" => 0x0039,
        "colon" => 0x003A, "semicolon" => 0x003B, "less" => 0x003C,
        "equal" => 0x003D, "greater" => 0x003E, "question" => 0x003F,
        "at" => 0x0040,
        "A" => 0x0041, "B" => 0x0042, "C" => 0x0043, "D" => 0x0044,
        "E" => 0x0045, "F" => 0x0046, "G" => 0x0047, "H" => 0x0048,
        "I" => 0x0049, "J" => 0x004A, "K" => 0x004B, "L" => 0x004C,
        "M" => 0x004D, "N" => 0x004E, "O" => 0x004F, "P" => 0x0050,
        "Q" => 0x0051, "R" => 0x0052, "S" => 0x0053, "T" => 0x0054,
        "U" => 0x0055, "V" => 0x0056, "W" => 0x0057, "X" => 0x0058,
        "Y" => 0x0059, "Z" => 0x005A,
        "bracketleft" => 0x005B, "backslash" => 0x005C, "bracketright" => 0x005D,
        "asciicircum" => 0x005E, "underscore" => 0x005F, "grave" => 0x0060,
        "quoteleft" => 0x2018,
        "a" => 0x0061, "b" => 0x0062, "c" => 0x0063, "d" => 0x0064,
        "e" => 0x0065, "f" => 0x0066, "g" => 0x0067, "h" => 0x0068,
        "i" => 0x0069, "j" => 0x006A, "k" => 0x006B, "l" => 0x006C,
        "m" => 0x006D, "n" => 0x006E, "o" => 0x006F, "p" => 0x0070,
        "q" => 0x0071, "r" => 0x0072, "s" => 0x0073, "t" => 0x0074,
        "u" => 0x0075, "v" => 0x0076, "w" => 0x0077, "x" => 0x0078,
        "y" => 0x0079, "z" => 0x007A,
        "braceleft" => 0x007B, "bar" => 0x007C, "braceright" => 0x007D,
        "asciitilde" => 0x007E,
        // Latin-1 Supplement
        "exclamdown" => 0x00A1, "cent" => 0x00A2, "sterling" => 0x00A3,
        "currency" => 0x00A4, "yen" => 0x00A5, "brokenbar" => 0x00A6,
        "section" => 0x00A7, "dieresis" => 0x00A8, "copyright" => 0x00A9,
        "ordfeminine" => 0x00AA, "guillemotleft" => 0x00AB,
        "logicalnot" => 0x00AC, "registered" => 0x00AE, "macron" => 0x00AF,
        "degree" => 0x00B0, "plusminus" => 0x00B1, "twosuperior" => 0x00B2,
        "threesuperior" => 0x00B3, "acute" => 0x00B4, "mu" => 0x00B5,
        "paragraph" => 0x00B6, "periodcentered" => 0x00B7, "cedilla" => 0x00B8,
        "onesuperior" => 0x00B9, "ordmasculine" => 0x00BA,
        "guillemotright" => 0x00BB, "onequarter" => 0x00BC,
        "onehalf" => 0x00BD, "threequarters" => 0x00BE,
        "questiondown" => 0x00BF,
        "Agrave" => 0x00C0, "Aacute" => 0x00C1, "Acircumflex" => 0x00C2,
        "Atilde" => 0x00C3, "Adieresis" => 0x00C4, "Aring" => 0x00C5,
        "AE" => 0x00C6, "Ccedilla" => 0x00C7, "Egrave" => 0x00C8,
        "Eacute" => 0x00C9, "Ecircumflex" => 0x00CA, "Edieresis" => 0x00CB,
        "Igrave" => 0x00CC, "Iacute" => 0x00CD, "Icircumflex" => 0x00CE,
        "Idieresis" => 0x00CF, "Eth" => 0x00D0, "Ntilde" => 0x00D1,
        "Ograve" => 0x00D2, "Oacute" => 0x00D3, "Ocircumflex" => 0x00D4,
        "Otilde" => 0x00D5, "Odieresis" => 0x00D6, "multiply" => 0x00D7,
        "Oslash" => 0x00D8, "Ugrave" => 0x00D9, "Uacute" => 0x00DA,
        "Ucircumflex" => 0x00DB, "Udieresis" => 0x00DC, "Yacute" => 0x00DD,
        "Thorn" => 0x00DE, "germandbls" => 0x00DF,
        "agrave" => 0x00E0, "aacute" => 0x00E1, "acircumflex" => 0x00E2,
        "atilde" => 0x00E3, "adieresis" => 0x00E4, "aring" => 0x00E5,
        "ae" => 0x00E6, "ccedilla" => 0x00E7, "egrave" => 0x00E8,
        "eacute" => 0x00E9, "ecircumflex" => 0x00EA, "edieresis" => 0x00EB,
        "igrave" => 0x00EC, "iacute" => 0x00ED, "icircumflex" => 0x00EE,
        "idieresis" => 0x00EF, "eth" => 0x00F0, "ntilde" => 0x00F1,
        "ograve" => 0x00F2, "oacute" => 0x00F3, "ocircumflex" => 0x00F4,
        "otilde" => 0x00F5, "odieresis" => 0x00F6, "divide" => 0x00F7,
        "oslash" => 0x00F8, "ugrave" => 0x00F9, "uacute" => 0x00FA,
        "ucircumflex" => 0x00FB, "udieresis" => 0x00FC, "yacute" => 0x00FD,
        "thorn" => 0x00FE, "ydieresis" => 0x00FF,
        // Latin Extended-A
        "Amacron" => 0x0100, "amacron" => 0x0101, "Abreve" => 0x0102, "abreve" => 0x0103,
        "Aogonek" => 0x0104, "aogonek" => 0x0105, "Cacute" => 0x0106, "cacute" => 0x0107,
        "Ccircumflex" => 0x0108, "ccircumflex" => 0x0109, "Cdotaccent" => 0x010A,
        "cdotaccent" => 0x010B, "Ccaron" => 0x010C, "ccaron" => 0x010D,
        "Dcaron" => 0x010E, "dcaron" => 0x010F, "Dcroat" => 0x0110, "dcroat" => 0x0111,
        "Emacron" => 0x0112, "emacron" => 0x0113, "Ebreve" => 0x0114, "ebreve" => 0x0115,
        "Edotaccent" => 0x0116, "edotaccent" => 0x0117, "Eogonek" => 0x0118,
        "eogonek" => 0x0119, "Ecaron" => 0x011A, "ecaron" => 0x011B,
        "Gcircumflex" => 0x011C, "gcircumflex" => 0x011D,
        "Lslash" => 0x0141, "lslash" => 0x0142,
        "Nacute" => 0x0143, "nacute" => 0x0144,
        "Ncaron" => 0x0147, "ncaron" => 0x0148,
        "OE" => 0x0152, "oe" => 0x0153, "Racute" => 0x0154, "racute" => 0x0155,
        "Rcaron" => 0x0158, "rcaron" => 0x0159, "Sacute" => 0x015A, "sacute" => 0x015B,
        "Scircumflex" => 0x015C, "scircumflex" => 0x015D,
        "Scedilla" => 0x015E, "scedilla" => 0x015F, "Scaron" => 0x0160, "scaron" => 0x0161,
        "Tcaron" => 0x0164, "tcaron" => 0x0165,
        "Uring" => 0x016E, "uring" => 0x016F,
        "Uhungarumlaut" => 0x0170, "uhungarumlaut" => 0x0171,
        "Uogonek" => 0x0172, "uogonek" => 0x0173,
        "Ydieresis" => 0x0178,
        "Zacute" => 0x0179, "zacute" => 0x017A, "Zdotaccent" => 0x017B,
        "zdotaccent" => 0x017C, "Zcaron" => 0x017D, "zcaron" => 0x017E,
        // Punctuation / typographic
        "endash" => 0x2013, "emdash" => 0x2014,
        "quotedblleft" => 0x201C, "quotedblright" => 0x201D,
        "quotesinglbase" => 0x201A, "quotedblbase" => 0x201E,
        "dagger" => 0x2020, "daggerdbl" => 0x2021,
        "bullet" => 0x2022, "ellipsis" => 0x2026,
        "perthousand" => 0x2030, "guilsinglleft" => 0x2039, "guilsinglright" => 0x203A,
        "fraction" => 0x2044, "Euro" => 0x20AC,
        "fi" => 0xFB01, "fl" => 0xFB02,
        // Diacritics / combining marks (standalone)
        "ring" => 0x02DA, "caron" => 0x02C7, "breve" => 0x02D8,
        "dotaccent" => 0x02D9, "hungarumlaut" => 0x02DD, "ogonek" => 0x02DB,
        "tilde" => 0x02DC, "circumflex" => 0x02C6,
        // Greek (common)
        "Alpha" => 0x0391, "Beta" => 0x0392, "Gamma" => 0x0393, "Delta" => 0x0394,
        "Epsilon" => 0x0395, "Zeta" => 0x0396, "Eta" => 0x0397, "Theta" => 0x0398,
        "Iota" => 0x0399, "Kappa" => 0x039A, "Lambda" => 0x039B, "Mu" => 0x039C,
        "Nu" => 0x039D, "Xi" => 0x039E, "Omicron" => 0x039F, "Pi" => 0x03A0,
        "Rho" => 0x03A1, "Sigma" => 0x03A3, "Tau" => 0x03A4, "Upsilon" => 0x03A5,
        "Phi" => 0x03A6, "Chi" => 0x03A7, "Psi" => 0x03A8, "Omega" => 0x03A9,
        "alpha" => 0x03B1, "beta" => 0x03B2, "gamma" => 0x03B3, "delta" => 0x03B4,
        "epsilon" => 0x03B5, "zeta" => 0x03B6, "eta" => 0x03B7, "theta" => 0x03B8,
        "iota" => 0x03B9, "kappa" => 0x03BA, "lambda" => 0x03BB,
        "nu" => 0x03BD, "xi" => 0x03BE, "omicron" => 0x03BF, "pi" => 0x03C0,
        "rho" => 0x03C1, "sigma1" => 0x03C2, "sigma" => 0x03C3, "tau" => 0x03C4,
        "upsilon" => 0x03C5, "phi" => 0x03C6, "chi" => 0x03C7, "psi" => 0x03C8,
        "omega" => 0x03C9,
        // Math symbols
        "minus" => 0x2212, "infinity" => 0x221E, "radical" => 0x221A,
        "summation" => 0x2211, "product" => 0x220F,
        "integral" => 0x222B, "partialdiff" => 0x2202,
        "notequal" => 0x2260, "lessequal" => 0x2264, "greaterequal" => 0x2265,
        "approxequal" => 0x2248, "lozenge" => 0x25CA,
        _ => return None,
    };
    Some(cp)
}
