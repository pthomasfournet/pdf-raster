//! Loaded font face — wraps a `freetype::Face` with rendering metadata.
//!
//! A [`FontFace`] corresponds to one `SplashFTFontFile` + one size matrix in
//! the C++ code.  It owns the `FT_Face` handle directly (not shared across
//! instances — each size/matrix combination gets its own `FT_Face`, as in
//! the original C++ model where each `SplashFTFont` has a private `FT_Size`
//! object attached to the shared `FT_Face`).
//!
//! Rendering is performed by [`FontFace::make_glyph`], which mirrors
//! `SplashFTFont::makeGlyph`.

use freetype::Matrix;
use freetype::Vector;

use crate::engine::FaceParams;
use crate::hinting::{FontKind, load_flags};
use crate::key::FaceId;
use crate::outline::decompose_outline;
use raster::path::Path;

/// The 2×2 transform matrix applied to the font (device-space, from the PDF
/// text/glyph matrix).  Stored as `[a, b, c, d]` matching `mat[0..4]` in
/// `SplashFTFont`.
pub type FontMatrix = [f64; 4];

/// A scaled font face, ready to rasterize glyphs.
///
/// `FontFace` is not `Send` because `freetype::Face` wraps a raw `FT_Face`
/// pointer which is not thread-safe.  Use one `FontFace` per thread or
/// protect with a `Mutex`.
#[expect(
    clippy::struct_excessive_bools,
    reason = "each bool is an independent FreeType render-state flag (aa, two \
              hinting sub-modes, symbol-PUA cmap); they are not a state enum"
)]
pub struct FontFace {
    /// Opaque identifier for this face (for cache keying).
    pub id: FaceId,
    /// The `FreeType` face handle, owned by this instance.
    face: freetype::Face,
    /// Glyph-index map: `code_to_gid[char_code]` → FT glyph index.
    /// An empty `Vec` means `char_code == glyph_id` (identity map).
    /// Set after face construction by the font cache when `Differences` entries
    /// are present; resolved via `FreeType`'s active charmap.
    pub code_to_gid: Vec<u32>,
    /// Font kind, for hinting-mode selection.
    pub kind: FontKind,
    /// Pixel size, computed as `splashRound(dist(mat[2], mat[3]))`.
    pub size_px: u16,
    /// Whether anti-aliasing is enabled for this face instance.
    pub aa: bool,
    /// `FreeType` 16.16 fixed-point transform matrix for `FT_Set_Transform`.
    ft_matrix: Matrix,
    /// `FreeType` 16.16 fixed-point text matrix (for outline decomposition).
    ft_text_matrix: Matrix,
    /// `textScale = dist(textMat[2], textMat[3]) / size_px`.
    ///
    /// Zero indicates a degenerate text matrix; glyph-path rendering is
    /// disabled when this is zero.
    pub text_scale: f64,
    /// Hinting mode: whether `FreeType` hinting is globally enabled.
    ft_hinting: bool,
    /// Slight-hinting sub-mode.
    slight_hinting: bool,
    /// The active charmap is a Microsoft-Symbol (3,0) cmap, whose entries live
    /// in the 0xF000–0xF0FF Private Use range.  Char-code lookups must add the
    /// 0xF000 offset before `FT_Get_Char_Index` (PDF §9.6.6.4).
    symbol_pua: bool,
}

impl FontFace {
    /// Construct a `FontFace` from a loaded `FreeType` face.
    ///
    /// `mat` is the 2×2 font/device matrix; `text_mat` is the 2×2 text
    /// matrix (both use the `[a, b, c, d]` layout from `SplashFTFont`).
    /// `code_to_gid` may be empty (identity map) or a per-character GID table.
    ///
    /// Returns `None` if the face cannot be set up — e.g. the size matrix
    /// is degenerate (zero scale or zero `units_per_EM`).  This matches
    /// `SplashFTFont`'s `isOk` guard.
    #[must_use]
    pub fn new(
        id: FaceId,
        face: freetype::Face,
        params: FaceParams,
        aa: bool,
        ft_hinting: bool,
        slight_hinting: bool,
    ) -> Option<Self> {
        let FaceParams {
            kind,
            code_to_gid,
            mat,
            text_mat,
        } = params;

        // Size is the magnitude of the y-column [c, d] of the font matrix —
        // this matches FreeType's nominal pixel-height for the face.
        let size_f = f64::hypot(mat[2], mat[3]).round();
        // Guard NaN/Inf (hypot returns Inf if either input is Inf; NaN if both are NaN)
        // and out-of-range values before the cast to u16.
        if !size_f.is_finite() || size_f < 1.0 || size_f > f64::from(u16::MAX) {
            return None;
        }
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "size_f is finite and clamped to [1, 65535] by the checks above"
        )]
        let size_px = size_f as u16;

        let text_scale = f64::hypot(text_mat[2], text_mat[3]) / f64::from(size_px);

        let units_per_em = face.raw().units_per_EM;
        // text_scale near-zero means the text matrix has a degenerate y-column;
        // use EPSILON rather than exact-zero to catch subnormal inputs.
        if text_scale < f64::EPSILON || units_per_em == 0 {
            return None;
        }

        let ft_matrix = to_ft_matrix_norm(&mat, f64::from(size_px));
        let ft_text_matrix = to_ft_matrix_norm(&text_mat, text_scale * f64::from(size_px));

        // Tell FreeType the nominal pixel size for this face.  Without this,
        // FreeType renders at an arbitrary internal default and transforms only
        // the outline shape, not its scale.  set_pixel_sizes(width=0, height)
        // lets FreeType derive the width from the height automatically.
        if face.set_pixel_sizes(0, u32::from(size_px)).is_err() {
            return None;
        }

        // Select the right cmap for symbolic embedded TrueType subsets.
        //
        // A producer that subsets a TrueType font for a symbolic PDF font
        // (Flags bit 3) commonly emits *no* /Encoding and ships only a (1,0)
        // Macintosh-Roman cmap — the real char-code→GID table — alongside a
        // (3,0) Microsoft-Symbol cmap whose entries are shifted into the
        // 0xF000–0xF0FF Private Use range.  FreeType's default charmap
        // priority activates the (3,0) Symbol cmap, so a raw `get_char_index`
        // on the small subset codes the content stream uses misses every
        // glyph and the whole page renders blank (PDF §9.6.6.4).
        //
        // When there is no Unicode cmap, prefer (1,0) Macintosh-Roman (direct
        // code lookup), else fall back to (3,0) Microsoft-Symbol with the
        // 0xF000 offset.  Unicode cmaps and non-TrueType fonts keep FreeType's
        // default (correct for WinAnsi/MacRoman/Standard encodings).
        let symbol_pua = select_truetype_cmap(&face, kind);

        Some(Self {
            id,
            face,
            code_to_gid,
            kind,
            size_px,
            aa,
            ft_matrix,
            ft_text_matrix,
            text_scale,
            ft_hinting,
            slight_hinting,
            symbol_pua,
        })
    }

    /// Rasterize glyph `char_code` and return a `GlyphBitmap` (defined in `crate::glyph`).
    ///
    /// Mirrors `SplashFTFont::makeGlyph`.
    ///
    /// Returns `None` on any `FreeType` failure (load, render, or zero-size
    /// output) — the caller should treat `None` as a blank/missing glyph.
    ///
    /// `x_frac` is the sub-pixel x offset in `[0, FRACTION)` units.
    #[must_use]
    pub fn make_glyph(&self, char_code: u32, x_frac: u8) -> Option<crate::bitmap::GlyphBitmap> {
        let glyph_id = self.resolve_gid(char_code);
        self.make_glyph_by_gid(glyph_id, x_frac)
    }

    /// Rasterize the glyph at `FreeType` glyph index `glyph_id` directly,
    /// bypassing the char-code → GID charmap resolution in [`Self::resolve_gid`].
    ///
    /// Used by the Type 0 / `CIDFont` text path, where the interpreter has
    /// already mapped the character code through the Encoding `CMap` and
    /// `CIDToGIDMap` to a final `FreeType` glyph index.  Routing that GID back
    /// through `resolve_gid` (a Unicode-charmap lookup) would mis-map it —
    /// for a CID-keyed CFF subset the charmap is absent or unrelated, so the
    /// lookup yields `.notdef` and every glyph renders blank.
    ///
    /// Returns `None` on any `FreeType` failure (load, render, or zero-size
    /// output) — the caller should treat `None` as a blank/missing glyph.
    #[must_use]
    pub fn make_glyph_by_gid(
        &self,
        glyph_id: u32,
        x_frac: u8,
    ) -> Option<crate::bitmap::GlyphBitmap> {
        // Sub-pixel offset: xFrac * fractionMul * 64 in 26.6 units.
        let frac_mul = 1.0 / f64::from(crate::key::FRACTION);
        let x_offset_f = f64::from(x_frac) * frac_mul * 64.0;
        #[expect(
            clippy::cast_possible_truncation,
            reason = "x_frac < FRACTION (4); x_offset_f ≤ 3 * 0.25 * 64 = 48; fits i64"
        )]
        let x_offset = x_offset_f.round() as i64;

        let mut offset = Vector { x: x_offset, y: 0 };
        let mut matrix = self.ft_matrix;
        self.face.set_transform(&mut matrix, &mut offset);

        let flags = load_flags(self.kind, self.aa, self.ft_hinting, self.slight_hinting);
        self.face.load_glyph(glyph_id, flags).ok()?;

        let slot = self.face.glyph();
        let render_mode = if self.aa {
            freetype::RenderMode::Normal
        } else {
            freetype::RenderMode::Mono
        };
        slot.render_glyph(render_mode).ok()?;

        let ft_bmp = slot.bitmap();
        // FreeType renders always produce non-negative dimensions; treat
        // negative (corrupt) values as empty glyphs.
        let w = u32::try_from(ft_bmp.width()).unwrap_or(0);
        let h = u32::try_from(ft_bmp.rows()).unwrap_or(0);
        if w == 0 || h == 0 {
            return None;
        }

        let row_bytes = if self.aa {
            w as usize
        } else {
            w.div_ceil(8) as usize
        };
        // FreeType pitch is negative for bottom-up bitmaps. We only handle
        // top-down (positive pitch) layouts; treat bottom-up as a render failure.
        let raw_pitch = ft_bmp.pitch();
        if raw_pitch < 0 {
            return None;
        }
        #[expect(clippy::cast_sign_loss, reason = "raw_pitch >= 0 checked above")]
        let pitch = raw_pitch as usize;
        // FreeType guarantees pitch ≥ row_bytes for valid bitmaps; guard anyway
        // so a corrupt/adversarial glyph doesn't cause a slice-bounds panic.
        if pitch < row_bytes {
            return None;
        }
        let raw = ft_bmp.buffer();

        let mut data = Vec::with_capacity(row_bytes * h as usize);
        for row in 0..h as usize {
            // Use checked_mul to guard against overflow on pathological glyph dims.
            let src = row.checked_mul(pitch)?;
            data.extend_from_slice(&raw[src..src + row_bytes]);
        }

        Some(crate::bitmap::GlyphBitmap {
            x_off: -slot.bitmap_left(),
            y_off: slot.bitmap_top(),
            width: w,
            height: h,
            aa: self.aa,
            data,
        })
    }

    /// Decompose the glyph outline for `char_code` into a [`Path`].
    ///
    /// Mirrors `SplashFTFont::getGlyphPath`.
    ///
    /// Returns `None` if the face has a degenerate text scale, if `FreeType`
    /// fails to load the glyph, or if the glyph has no outline (e.g. a
    /// bitmap-only font).
    #[must_use]
    pub fn glyph_path(&self, char_code: u32) -> Option<Path> {
        let glyph_id = self.resolve_gid(char_code);
        self.glyph_path_by_gid(glyph_id)
    }

    /// Decompose the outline at `FreeType` glyph index `glyph_id` directly,
    /// bypassing [`Self::resolve_gid`].  The Type 0 / `CIDFont` clip path uses
    /// this for the same reason [`Self::make_glyph_by_gid`] exists.
    #[must_use]
    pub fn glyph_path_by_gid(&self, glyph_id: u32) -> Option<Path> {
        // Mirror the near-zero check used in FontFace::new to catch subnormals.
        if self.text_scale < f64::EPSILON {
            return None;
        }

        let mut matrix = self.ft_text_matrix;
        let mut delta = Vector { x: 0, y: 0 };
        self.face.set_transform(&mut matrix, &mut delta);

        let flags = load_flags(self.kind, self.aa, self.ft_hinting, self.slight_hinting);
        self.face.load_glyph(glyph_id, flags).ok()?;

        let slot = self.face.glyph();
        let outline = slot.outline()?;
        decompose_outline(&outline, self.text_scale)
    }

    /// Return the advance width for `char_code` in font-size-normalised units
    /// (multiply by `font_size` to get device-space pixels).
    ///
    /// Returns `None` if `FreeType` fails to load the glyph.  For valid fonts
    /// the result is non-negative; adversarial fonts with negative
    /// `horiAdvance` metrics propagate through.
    #[must_use]
    pub fn glyph_advance(&self, char_code: u32) -> Option<f64> {
        let glyph_id = self.resolve_gid(char_code);
        self.glyph_advance_by_gid(glyph_id)
    }

    /// Advance width at `FreeType` glyph index `glyph_id`, bypassing
    /// [`Self::resolve_gid`].  Mirrors [`Self::make_glyph_by_gid`].
    #[must_use]
    pub fn glyph_advance_by_gid(&self, glyph_id: u32) -> Option<f64> {
        // Identity matrix + zero offset for advance measurement.
        let mut matrix = Matrix {
            xx: 65536,
            xy: 0,
            yx: 0,
            yy: 65536,
        };
        let mut delta = Vector { x: 0, y: 0 };
        self.face.set_transform(&mut matrix, &mut delta);

        let flags = load_flags(self.kind, self.aa, self.ft_hinting, self.slight_hinting);
        self.face.load_glyph(glyph_id, flags).ok()?;

        // `horiAdvance` is in 26.6 fixed-point; divide by 64 for pixels, then by size.
        #[expect(
            clippy::cast_precision_loss,
            reason = "horiAdvance is FT_Pos (i64); typical advance values fit in f64 mantissa"
        )]
        let advance = self.face.glyph().metrics().horiAdvance as f64;
        Some(advance / 64.0 / f64::from(self.size_px))
    }

    /// Look up a Unicode codepoint in `FreeType`'s active charmap and return
    /// the glyph index.  Returns 0 (`.notdef`) if the codepoint has no glyph.
    ///
    /// Used by the PDF interpreter's font cache to build a `code_to_gid` table
    /// from glyph names resolved via the Adobe Glyph List.
    #[must_use]
    pub fn raw_get_char_index(&self, unicode: u32) -> u32 {
        self.face.get_char_index(unicode as usize).unwrap_or(0)
    }

    /// Look up a glyph *name* in the font program's own charset / `post`
    /// table (`FreeType` `FT_Get_Name_Index`) and return the glyph index.
    /// Returns 0 (`.notdef`) when the font has no name table or the name is
    /// absent.
    ///
    /// This is the correct primitive for simple (non-CID) Type1/Type1C/CFF
    /// fonts: a subsetted embedded program packs glyphs in arbitrary order,
    /// so the char code is neither the GID nor (often) a usable Unicode cmap
    /// key — but the program's charset still names every glyph it contains.
    #[must_use]
    pub fn raw_get_name_index(&self, name: &str) -> u32 {
        self.face.get_name_index(name).unwrap_or(0)
    }

    /// Resolve a character code to a `FreeType` glyph index.
    ///
    /// When `code_to_gid` is populated (e.g. from a PDF `Differences` array),
    /// it is used directly.  Otherwise the char code is resolved through the
    /// face's active charmap, which [`select_truetype_cmap`] has already
    /// pointed at the correct table at construction time.  For standard
    /// encodings (`WinAnsi`, `MacRoman`, `Standard`) the byte value is a
    /// Unicode codepoint; for a symbolic embedded TrueType subset mapped via
    /// a (3,0) Microsoft-Symbol cmap the code is offset into the 0xF000 PUA
    /// (PDF §9.6.6.4) — `symbol_pua` records that case.
    fn resolve_gid(&self, char_code: u32) -> u32 {
        // Safe cast: PDF char codes are 0–255 (single-byte), so u32→usize is lossless
        // on all supported targets (usize ≥ 32 bits).
        let idx = char_code as usize;
        if let Some(&gid) = self.code_to_gid.get(idx) {
            return gid;
        }
        let gid = self.face.get_char_index(idx).unwrap_or(0);
        if gid != 0 || !self.symbol_pua {
            return gid;
        }
        // (3,0) Microsoft-Symbol cmap: entries live in the 0xF000–0xF0FF
        // Private Use range, so a raw byte code misses.  Retry with the
        // 0xF000 offset (PDF §9.6.6.4).  Returns 0 (.notdef) on a real miss —
        // the caller treats that as a blank/missing glyph.
        self.face.get_char_index(0xF000 | idx).unwrap_or(0)
    }
}

/// Point the face at the correct cmap for a symbolic embedded TrueType subset
/// and report whether that cmap is a (3,0) Microsoft-Symbol table (whose codes
/// need the 0xF000 PUA offset at lookup time).
///
/// Non-`TrueType` faces, and `TrueType` faces that expose a Microsoft-Unicode
/// (3,1)/(3,10) cmap, are left on `FreeType`'s default selection — that is
/// already correct for `WinAnsi`/`MacRoman`/`Standard` encodings and for
/// `Type 0` / `CIDFont` paths that bypass `resolve_gid` entirely.
///
/// Only when there is *no* Unicode cmap do we override: prefer the (1,0)
/// Macintosh-Roman cmap (the subset's real `code`→`GID` table, looked up by
/// raw code), else the (3,0) Microsoft-Symbol cmap (looked up with the 0xF000
/// offset).  Without this, `FreeType`'s default priority activates the (3,0)
/// Symbol cmap, every small subset code resolves to `.notdef`, and the entire
/// text layer of the page renders blank.
fn select_truetype_cmap(face: &freetype::Face, kind: FontKind) -> bool {
    if kind != FontKind::TrueType {
        return false;
    }
    let n = face.num_charmaps();
    if n <= 0 {
        return false;
    }
    let mut mac_roman: Option<isize> = None;
    let mut ms_symbol: Option<isize> = None;
    for i in 0..n as isize {
        let cm = face.get_charmap(i);
        match (cm.platform_id(), cm.encoding_id()) {
            // A Unicode cmap is present — FreeType's default is correct;
            // do not disturb it.
            (3, 1 | 10) | (0, _) => return false,
            (1, 0) if mac_roman.is_none() => mac_roman = Some(i),
            (3, 0) if ms_symbol.is_none() => ms_symbol = Some(i),
            _ => {}
        }
    }
    if let Some(i) = mac_roman {
        let cm = face.get_charmap(i);
        if face.set_charmap(&cm).is_ok() {
            return false;
        }
    }
    if let Some(i) = ms_symbol {
        let cm = face.get_charmap(i);
        if face.set_charmap(&cm).is_ok() {
            return true;
        }
    }
    false
}

/// Convert a 2×2 font matrix (normalised by `divisor`) to a `FreeType` 16.16
/// fixed-point matrix.
///
/// The `FreeType` matrix layout is:
/// ```text
/// [ xx  xy ]   [ mat[0]  mat[2] ]
/// [ yx  yy ] = [ mat[1]  mat[3] ]
/// ```
fn to_ft_matrix_norm(mat: &FontMatrix, divisor: f64) -> Matrix {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "values are (mat[i]/size) * 65536; for typical font sizes the result fits in i64 (FT_Fixed)"
    )]
    Matrix {
        xx: ((mat[0] / divisor) * 65536.0) as i64,
        yx: ((mat[1] / divisor) * 65536.0) as i64,
        xy: ((mat[2] / divisor) * 65536.0) as i64,
        yy: ((mat[3] / divisor) * 65536.0) as i64,
    }
}

#[cfg(test)]
mod tests {
    use crate::engine::{FaceParams, FontEngine};
    use crate::hinting::FontKind;

    /// A few candidate system fonts; the test is skipped if none exist so it
    /// stays green on minimal CI images.
    const CANDIDATE_FONTS: &[&str] = &[
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
    ];

    fn first_existing_font() -> Option<&'static str> {
        CANDIDATE_FONTS
            .iter()
            .copied()
            .find(|p| std::path::Path::new(p).exists())
    }

    /// Load an identity-mapped (`code_to_gid` empty) face at 40px from the
    /// shared engine.  The engine `Mutex` guard is scoped to this call so it
    /// is never held across the test's assertions.
    fn load_identity_face(eng: &std::sync::Mutex<FontEngine>, path: &str) -> super::FontFace {
        eng.lock()
            .expect("engine lock")
            .load_file_face(
                path,
                0,
                FaceParams {
                    kind: FontKind::TrueType,
                    code_to_gid: Vec::new(),
                    mat: [40.0, 0.0, 0.0, 40.0],
                    text_mat: [40.0, 0.0, 0.0, 40.0],
                },
            )
            .expect("load identity face")
    }

    /// Regression for the PDF-1.5 / `CIDFontType0C` blank-page bug.
    ///
    /// A `Type 0` / `CIDFont` caller resolves a character code through the
    /// Encoding `CMap` + `CIDToGIDMap` to a *final* `FreeType` glyph index,
    /// then must hand that GID to `FreeType` directly.  The pre-fix code
    /// routed it back through `resolve_gid` (a Unicode-charmap lookup), which
    /// for a CID-keyed CFF subset maps every glyph to `.notdef` → blank page.
    ///
    /// This asserts the two routing modes are genuinely distinct: a
    /// `code_to_gid` remap table that hijacks `make_glyph` must NOT touch
    /// `make_glyph_by_gid` (which indexes `FreeType` directly).
    #[test]
    fn make_glyph_by_gid_bypasses_charmap_resolution() {
        let Some(path) = first_existing_font() else {
            eprintln!("no system font available; skipping");
            return;
        };
        let eng_shared = FontEngine::init(true, false, false).expect("font engine");

        // Resolve two genuinely different glyph indices via the font's own
        // charmap, using a throwaway identity-mapped face.
        let probe = load_identity_face(&eng_shared, path);
        let gid_a = probe.raw_get_char_index('A' as u32);
        let gid_w = probe.raw_get_char_index('W' as u32);
        assert!(gid_a != 0 && gid_w != 0, "expected real glyph indices");
        assert!(gid_a != gid_w, "'A' and 'W' must have distinct glyphs");
        drop(probe);

        // A `code_to_gid` hijack table: char code 65 ('A') is forced to
        // resolve to 'W'`s glyph index.  Only the charmap path consults it.
        let mut hijack = vec![0u32; 256];
        hijack[65] = gid_w;

        let face = eng_shared
            .lock()
            .expect("engine lock")
            .load_file_face(
                path,
                0,
                FaceParams {
                    kind: FontKind::TrueType,
                    code_to_gid: hijack,
                    mat: [40.0, 0.0, 0.0, 40.0],
                    text_mat: [40.0, 0.0, 0.0, 40.0],
                },
            )
            .expect("load system face");

        // `make_glyph(65)` consults the hijack table → renders 'W'.
        let via_charmap = face
            .make_glyph(65, 0)
            .expect("charmap path rasterizes a glyph");
        // `make_glyph_by_gid(gid_a)` ignores the hijack entirely and indexes
        // FreeType directly → renders 'A'.  This is the path the CIDFont text
        // loop now uses; pre-fix it went through resolve_gid and broke.
        let via_gid = face
            .make_glyph_by_gid(gid_a, 0)
            .expect("by-gid path rasterizes the real glyph");
        assert!(
            via_gid.width > 0 && via_gid.height > 0,
            "by-gid glyph must be non-empty"
        );
        // 'A' and 'W' differ in shape → the two paths produced different
        // bitmaps, proving by-gid did NOT route through the hijack table.
        assert_ne!(
            (via_charmap.width, via_charmap.height, &via_charmap.data),
            (via_gid.width, via_gid.height, &via_gid.data),
            "by-gid path must bypass code_to_gid resolution"
        );

        // The outline variant must behave the same way.
        assert!(
            face.glyph_path_by_gid(gid_a).is_some(),
            "by-gid outline must resolve the real glyph"
        );
    }

    /// Hostile-input guard for the by-GID path.
    ///
    /// A `Type 0` PDF supplies untrusted character codes; the Encoding `CMap`
    /// plus `CIDToGIDMap` can map them to an arbitrary GID, including one far
    /// past the embedded subset's glyph count (a corrupt or adversarial
    /// `CIDToGIDMap` entry, or an Identity map fed an out-of-range CID).  That
    /// GID reaches `FreeType` *unmolested* through `make_glyph_by_gid` (the
    /// whole point of the bypass), so the bound check lives in `FreeType`'s
    /// `FT_Load_Glyph`.  This asserts the contract the text loop relies on:
    /// an out-of-range GID resolves to `None` (graceful .notdef / skipped
    /// glyph) and never panics, indexes out of bounds, or yields garbage.
    /// The outline and advance variants must behave the same way.
    #[test]
    fn by_gid_out_of_range_is_graceful_not_panic() {
        let Some(path) = first_existing_font() else {
            eprintln!("no system font available; skipping");
            return;
        };
        let eng_shared = FontEngine::init(true, false, false).expect("font engine");
        let face = load_identity_face(&eng_shared, path);

        // No real subset has 4 billion glyphs; this is the worst case a u32
        // CID/GID can produce.  u32::MAX and a merely-large value both probe
        // the `FreeType` bound, not just an off-by-one past the last glyph.
        for &bogus in &[u32::MAX, 1_000_000, 65_535] {
            assert!(
                face.make_glyph_by_gid(bogus, 0).is_none(),
                "out-of-range GID {bogus} must yield None, not a panic/garbage glyph"
            );
            assert!(
                face.glyph_path_by_gid(bogus).is_none(),
                "out-of-range GID {bogus} outline must yield None"
            );
            // Advance may be `Some(0.0)` (`FreeType` loads .notdef metrics) or
            // `None` depending on the driver; the contract is only that it
            // does not panic and returns a finite value when `Some`.
            if let Some(adv) = face.glyph_advance_by_gid(bogus) {
                assert!(
                    adv.is_finite(),
                    "out-of-range GID {bogus} advance must be finite, got {adv}"
                );
            }
        }
    }

    /// Dispatch invariant: the by-GID and by-char-code paths must reach the
    /// *same* glyph when no remap table is in play (identity `code_to_gid`,
    /// Unicode charmap).  This is the safety net for the campaign's recurring
    /// mis-dispatch class — if a future refactor wired the CID branch to the
    /// charmap path (or vice versa) for an identity-mapped face, this catches
    /// the silent divergence before it ships as a blank page.
    #[test]
    fn by_gid_and_charmap_agree_for_identity_face() {
        let Some(path) = first_existing_font() else {
            eprintln!("no system font available; skipping");
            return;
        };
        let eng_shared = FontEngine::init(true, false, false).expect("font engine");
        let face = load_identity_face(&eng_shared, path);

        // 'M' via the charmap (simple-font path) and the same glyph fetched
        // by its resolved GID (CIDFont path) must rasterize byte-identically.
        let gid_m = face.raw_get_char_index('M' as u32);
        assert!(gid_m != 0, "expected a real glyph for 'M'");
        let via_charmap = face.make_glyph('M' as u32, 0).expect("charmap 'M'");
        let via_gid = face.make_glyph_by_gid(gid_m, 0).expect("by-gid 'M'");
        assert_eq!(
            (via_charmap.width, via_charmap.height, &via_charmap.data),
            (via_gid.width, via_gid.height, &via_gid.data),
            "identity-face by-gid and charmap paths must reach the same glyph"
        );
    }

    // ── Symbolic-subset cmap regression ──────────────────────────────────────

    fn be16(v: u16) -> [u8; 2] {
        v.to_be_bytes()
    }
    fn be32(v: u32) -> [u8; 4] {
        v.to_be_bytes()
    }

    /// Build a minimal-but-valid TrueType font with exactly two glyphs
    /// (`.notdef` + one filled box at GID 1) and **only** a (1,0)
    /// Macintosh-Roman and a (3,0) Microsoft-Symbol cmap — deliberately *no*
    /// Unicode cmap.  Char code 7 maps to GID 1 in the (1,0) cmap (direct)
    /// and 0xF007 maps to GID 1 in the (3,0) cmap (PUA offset), mirroring the
    /// real scanned-book subset fonts in the NF-2 corpus.
    fn synth_symbolic_ttf() -> Vec<u8> {
        // glyf: GID 0 = empty; GID 1 = a simple 1-contour box.
        let mut box_glyph = Vec::new();
        box_glyph.extend_from_slice(&be16(1)); // numberOfContours
        box_glyph.extend_from_slice(&be16(0)); // xMin
        box_glyph.extend_from_slice(&be16(0)); // yMin
        box_glyph.extend_from_slice(&be16(500)); // xMax
        box_glyph.extend_from_slice(&be16(700)); // yMax
        box_glyph.extend_from_slice(&be16(3)); // endPtsOfContours[0] (4 pts)
        box_glyph.extend_from_slice(&be16(0)); // instructionLength
        box_glyph.extend_from_slice(&[0x01, 0x01, 0x01, 0x01]); // 4 on-curve flags, x&y short+pos
        // x deltas (short, positive): 0, +500, 0(-500 via neg bit not set→use words? keep simple)
        // Use the "repeat-free, all-short, all-positive-then-mirror" trick:
        // pts: (0,0)->(500,0)->(500,700)->(0,700) needs signed deltas, so
        // switch to NON-short coords (words).
        box_glyph.truncate(box_glyph.len() - 4);
        box_glyph.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]); // flags: all on-curve, word coords
        for d in [0i16, 500, 0, -500] {
            box_glyph.extend_from_slice(&d.to_be_bytes());
        }
        for d in [0i16, 0, 700, 0] {
            box_glyph.extend_from_slice(&d.to_be_bytes());
        }
        while box_glyph.len() % 2 != 0 {
            box_glyph.push(0);
        }
        let glyf = {
            let mut g = Vec::new();
            // GID 0: empty (zero-length)
            g.extend_from_slice(&box_glyph); // GID 1
            g
        };
        let loca: Vec<u8> = {
            let mut l = Vec::new();
            l.extend_from_slice(&be16(0)); // glyph 0 offset/2 = 0
            l.extend_from_slice(&be16(0)); // glyph 1 offset/2 = 0 (glyph 0 empty)
            l.extend_from_slice(&be16((glyf.len() / 2) as u16)); // end
            l
        };
        let head = {
            let mut h = Vec::new();
            h.extend_from_slice(&be32(0x0001_0000)); // version
            h.extend_from_slice(&be32(0x0001_0000)); // fontRevision
            h.extend_from_slice(&be32(0)); // checkSumAdjustment
            h.extend_from_slice(&be32(0x5F0F_3CF5)); // magic
            h.extend_from_slice(&be16(0)); // flags
            h.extend_from_slice(&be16(1000)); // unitsPerEm
            h.extend_from_slice(&be32(0));
            h.extend_from_slice(&be32(0)); // created
            h.extend_from_slice(&be32(0));
            h.extend_from_slice(&be32(0)); // modified
            h.extend_from_slice(&be16(0)); // xMin
            h.extend_from_slice(&be16(0)); // yMin
            h.extend_from_slice(&be16(500)); // xMax
            h.extend_from_slice(&be16(700)); // yMax
            h.extend_from_slice(&be16(0)); // macStyle
            h.extend_from_slice(&be16(8)); // lowestRecPPEM
            h.extend_from_slice(&be16(2)); // fontDirectionHint
            h.extend_from_slice(&be16(0)); // indexToLocFormat (0 = short)
            h.extend_from_slice(&be16(0)); // glyphDataFormat
            h
        };
        let maxp = {
            let mut m = Vec::new();
            m.extend_from_slice(&be32(0x0001_0000)); // version
            m.extend_from_slice(&be16(2)); // numGlyphs
            m.extend_from_slice(&[0u8; 26]); // remaining 1.0 fields
            m
        };
        let hhea = {
            let mut h = Vec::new();
            h.extend_from_slice(&be32(0x0001_0000));
            h.extend_from_slice(&be16(700)); // ascent
            h.extend_from_slice(&((-200i16).to_be_bytes())); // descent
            h.extend_from_slice(&be16(0)); // lineGap
            h.extend_from_slice(&be16(500)); // advanceWidthMax
            h.extend_from_slice(&be16(0));
            h.extend_from_slice(&be16(0));
            h.extend_from_slice(&be16(500));
            h.extend_from_slice(&be16(0));
            h.extend_from_slice(&be16(0));
            h.extend_from_slice(&be16(0));
            h.extend_from_slice(&[0u8; 8]);
            h.extend_from_slice(&be16(0)); // metricDataFormat
            h.extend_from_slice(&be16(2)); // numberOfHMetrics
            h
        };
        let hmtx = {
            let mut m = Vec::new();
            m.extend_from_slice(&be16(500)); // gid0 advance
            m.extend_from_slice(&be16(0));
            m.extend_from_slice(&be16(500)); // gid1 advance
            m.extend_from_slice(&be16(0));
            m
        };
        // cmap: (1,0) format-0 and (3,0) format-4.
        let cmap0 = {
            let mut c = Vec::new();
            c.extend_from_slice(&be16(0)); // format
            c.extend_from_slice(&be16(262)); // length
            c.extend_from_slice(&be16(0)); // language
            let mut g = [0u8; 256];
            g[7] = 1; // code 7 -> GID 1
            c.extend_from_slice(&g);
            c
        };
        let cmap4 = {
            // segments: [0xF007..0xF007 -> gid 1], [0xFFFF terminator]
            let mut c = Vec::new();
            c.extend_from_slice(&be16(4)); // format
            c.extend_from_slice(&be16(32)); // length
            c.extend_from_slice(&be16(0)); // language
            c.extend_from_slice(&be16(4)); // segCountX2 (2 segs)
            c.extend_from_slice(&be16(4)); // searchRange
            c.extend_from_slice(&be16(1)); // entrySelector
            c.extend_from_slice(&be16(0)); // rangeShift
            c.extend_from_slice(&be16(0xF007)); // endCode[0]
            c.extend_from_slice(&be16(0xFFFF)); // endCode[1]
            c.extend_from_slice(&be16(0)); // reservedPad
            c.extend_from_slice(&be16(0xF007)); // startCode[0]
            c.extend_from_slice(&be16(0xFFFF)); // startCode[1]
            // idDelta: gid = (code + delta) mod 65536. code 0xF007 -> 1
            // => delta = 1 - 0xF007 (mod 65536)
            c.extend_from_slice(&be16((1u32.wrapping_sub(0xF007) & 0xFFFF) as u16));
            c.extend_from_slice(&be16(1)); // idDelta[1] (0xFFFF -> 0)
            c.extend_from_slice(&be16(0)); // idRangeOffset[0]
            c.extend_from_slice(&be16(0)); // idRangeOffset[1]
            c
        };
        let cmap = {
            let mut c = Vec::new();
            c.extend_from_slice(&be16(0)); // version
            c.extend_from_slice(&be16(2)); // numTables
            let rec_end = 4 + 8 * 2;
            let off0 = rec_end as u32;
            let off1 = off0 + cmap0.len() as u32;
            c.extend_from_slice(&be16(1)); // platform 1 (Mac)
            c.extend_from_slice(&be16(0)); // encoding 0 (Roman)
            c.extend_from_slice(&be32(off0));
            c.extend_from_slice(&be16(3)); // platform 3 (MS)
            c.extend_from_slice(&be16(0)); // encoding 0 (Symbol)
            c.extend_from_slice(&be32(off1));
            c.extend_from_slice(&cmap0);
            c.extend_from_slice(&cmap4);
            c
        };
        let post = {
            let mut p = Vec::new();
            p.extend_from_slice(&be32(0x0003_0000)); // version 3.0
            p.extend_from_slice(&[0u8; 28]);
            p
        };
        let name = {
            let mut n = Vec::new();
            n.extend_from_slice(&be16(0)); // format
            n.extend_from_slice(&be16(0)); // count
            n.extend_from_slice(&be16(6)); // stringOffset
            n
        };

        let mut tables: Vec<(&[u8; 4], Vec<u8>)> = vec![
            (b"cmap", cmap),
            (b"glyf", glyf),
            (b"head", head),
            (b"hhea", hhea),
            (b"hmtx", hmtx),
            (b"loca", loca),
            (b"maxp", maxp),
            (b"name", name),
            (b"post", post),
        ];
        tables.sort_by(|a, b| a.0.cmp(b.0));

        let num = tables.len() as u16;
        let mut out = Vec::new();
        out.extend_from_slice(&be32(0x0001_0000)); // sfnt version
        out.extend_from_slice(&be16(num));
        let mut sr = 1u16;
        let mut es = 0u16;
        while sr * 2 <= num {
            sr *= 2;
            es += 1;
        }
        out.extend_from_slice(&be16(sr * 16));
        out.extend_from_slice(&be16(es));
        out.extend_from_slice(&be16(num * 16 - sr * 16));
        let mut offset = 12 + 16 * tables.len();
        let mut body = Vec::new();
        for (tag, data) in &tables {
            let cs: u32 = data
                .chunks(4)
                .map(|c| {
                    let mut w = [0u8; 4];
                    w[..c.len()].copy_from_slice(c);
                    u32::from_be_bytes(w)
                })
                .fold(0u32, u32::wrapping_add);
            out.extend_from_slice(*tag);
            out.extend_from_slice(&be32(cs));
            out.extend_from_slice(&be32(offset as u32));
            out.extend_from_slice(&be32(data.len() as u32));
            body.extend_from_slice(data);
            while body.len() % 4 != 0 {
                body.push(0);
            }
            offset = 12 + 16 * tables.len() + body.len();
        }
        out.extend_from_slice(&body);
        out
    }

    /// NF-2 Group A regression: a symbolic embedded TrueType subset whose only
    /// cmaps are (1,0) Macintosh-Roman and (3,0) Microsoft-Symbol must still
    /// resolve its small content-stream codes to real glyphs.
    ///
    /// Pre-fix, FreeType's default charmap priority activated the (3,0) Symbol
    /// cmap; `get_char_index(7)` returned `.notdef`, every glyph rendered
    /// blank, and text-only deep pages of scanned books came out pure white
    /// (lecouteux p194, etc.).  The fix points the face at the (1,0) cmap (or
    /// uses the 0xF000 PUA offset on the Symbol cmap).
    #[test]
    fn symbolic_truetype_subset_resolves_via_mac_or_symbol_cmap() {
        let ttf = synth_symbolic_ttf();
        let eng_shared = FontEngine::init(true, false, false).expect("font engine");
        let mut eng = eng_shared.lock().expect("engine lock");
        let face = eng
            .load_memory_face(
                ttf,
                0,
                FaceParams {
                    kind: FontKind::TrueType,
                    code_to_gid: Vec::new(),
                    mat: [40.0, 0.0, 0.0, 40.0],
                    text_mat: [40.0, 0.0, 0.0, 40.0],
                },
            )
            .expect("load synthetic symbolic TTF");

        // Code 7 is the glyph the (1,0)/(3,0) cmaps map to GID 1 (the box).
        // This is exactly the resolution that was broken (blank page).
        let g = face
            .make_glyph(7, 0)
            .expect("symbolic-subset code 7 must resolve to a real glyph, not .notdef");
        assert!(
            g.width > 0 && g.height > 0,
            "resolved glyph must be a non-empty bitmap"
        );
    }
}
