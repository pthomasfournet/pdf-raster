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
        // Mirror the near-zero check used in FontFace::new to catch subnormals.
        if self.text_scale < f64::EPSILON {
            return None;
        }

        let glyph_id = self.resolve_gid(char_code);
        let mut matrix = self.ft_text_matrix;
        let mut delta = Vector { x: 0, y: 0 };
        self.face.set_transform(&mut matrix, &mut delta);

        let flags = load_flags(self.kind, self.aa, self.ft_hinting, self.slight_hinting);
        self.face.load_glyph(glyph_id, flags).ok()?;

        let slot = self.face.glyph();
        let outline = slot.outline()?;
        decompose_outline(&outline, self.text_scale)
    }

    /// Return the advance width for `char_code`, normalised to `[0, ∞)`.
    ///
    /// Returns `-1.0` on failure (matching `SplashFTFont::getGlyphAdvance`).
    #[must_use]
    pub fn glyph_advance(&self, char_code: u32) -> f64 {
        let glyph_id = self.resolve_gid(char_code);

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
        if self.face.load_glyph(glyph_id, flags).is_err() {
            return -1.0;
        }

        // `horiAdvance` is in 26.6 fixed-point; divide by 64 for pixels, then by size.
        #[expect(
            clippy::cast_precision_loss,
            reason = "horiAdvance is FT_Pos (i64); typical advance values fit in f64 mantissa"
        )]
        let advance = self.face.glyph().metrics().horiAdvance as f64;
        advance / 64.0 / f64::from(self.size_px)
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

    /// Resolve a character code to a `FreeType` glyph index.
    ///
    /// When `code_to_gid` is populated (e.g. from a PDF `Differences` array),
    /// it is used directly.  When empty, we fall through to `FreeType`'s active
    /// charmap (`FT_Get_Char_Index`), treating the char code as a Unicode
    /// codepoint.  This is correct for standard encodings (`WinAnsi`, `MacRoman`,
    /// `Standard`) where byte values in the printable ASCII range are Unicode.
    fn resolve_gid(&self, char_code: u32) -> u32 {
        // Safe cast: PDF char codes are 0–255 (single-byte), so u32→usize is lossless
        // on all supported targets (usize ≥ 32 bits).
        let idx = char_code as usize;
        if let Some(&gid) = self.code_to_gid.get(idx) {
            return gid;
        }
        // Fall through to FreeType's active charmap.  Returns 0 (.notdef) on miss —
        // using char_code directly as a GID would produce garbage for Type1 fonts.
        self.face.get_char_index(idx).unwrap_or(0)
    }
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
