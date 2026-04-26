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
        // Validate that the matrix is not degenerate (NaN, Inf, or zero scale).
        // hypot(trm[2], trm[3]) is the y-column magnitude — used by FontFace::new
        // as the pixel size.  A zero or non-finite value produces a useless face.
        let size_f = f64::hypot(trm[2], trm[3]);
        if !size_f.is_finite() || size_f < 1.0 {
            log::debug!("font_cache: degenerate trm (size={size_f:.1}), skipping face load");
            return None;
        }

        let params = FaceParams {
            kind: to_font_kind(desc.kind),
            code_to_gid: desc.code_to_gid.clone(),
            mat: trm,
            text_mat: trm,
        };

        let mut eng = self.engine.lock().expect("FontEngine mutex poisoned");

        if let Some(bytes) = &desc.bytes {
            eng.load_memory_face(bytes.clone(), 0, params)
                .inspect_err(|e| log::debug!("font_cache: memory face error: {e}"))
                .ok()
        } else {
            // Fall back to system font.
            let path = fallback_font_path()?;
            eng.load_file_face(path, 0, params)
                .inspect_err(|e| log::debug!("font_cache: file face error ({path}): {e}"))
                .ok()
        }
    }
}
