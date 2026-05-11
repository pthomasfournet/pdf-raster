//! PDF text state.
//!
//! Tracks the parameters that control text positioning and appearance within
//! a BT/ET text object. These are a subset of the PDF graphics state (§8.3).

/// PDF text matrix — a 6-element `[a b c d e f]` column-major matrix.
pub type TextMatrix = [f64; 6];

/// The identity text matrix.
pub const TEXT_MATRIX_IDENTITY: TextMatrix = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];

/// PDF text state (§8.3, Table 104).
///
/// Reset to default values at the start of each page; `q`/`Q` does *not*
/// save/restore the text matrix (it is local to the BT/ET text object).
#[derive(Debug, Clone)]
pub struct TextState {
    // ── Text state parameters (persist across BT/ET) ──────────────────────────
    /// Character spacing `Tc` (added after each glyph; 0 = none).
    pub char_spacing: f64,
    /// Word spacing `Tw` (added after ASCII SPACE; 0 = none).
    pub word_spacing: f64,
    /// Horizontal text scaling `Tz` (percentage; 100 = normal width).
    pub horiz_scaling: f64,
    /// Text leading `TL` (vertical advance for `T*`, `'`, `"`).
    pub leading: f64,
    /// Text rise `Ts` (baseline shift; 0 = no shift).
    pub rise: f64,
    /// Text rendering mode `Tr` (0 = fill, 1 = stroke, 2 = fill+stroke, …).
    pub render_mode: i32,

    // ── Font ─────────────────────────────────────────────────────────────────
    /// Font resource name from `Tf`.
    pub font_name: Vec<u8>,
    /// Font size from `Tf` (in text-space units).
    pub font_size: f64,

    // ── Text matrices (local to BT/ET block) ─────────────────────────────────
    /// Text matrix `Tm` (also updated by `Td`, `TD`, `T*`, `'`, `"`).
    pub text_matrix: TextMatrix,
    /// Text line matrix (updated only by `Td`, `TD`, `T*`, `'`, `"`).
    pub line_matrix: TextMatrix,
}

impl Default for TextState {
    fn default() -> Self {
        Self {
            char_spacing: 0.0,
            word_spacing: 0.0,
            horiz_scaling: 100.0,
            leading: 0.0,
            rise: 0.0,
            render_mode: 0,
            font_name: Vec::new(),
            font_size: 0.0,
            text_matrix: TEXT_MATRIX_IDENTITY,
            line_matrix: TEXT_MATRIX_IDENTITY,
        }
    }
}

impl TextState {
    /// `BT` — reset text and line matrices to identity.
    pub const fn begin_text(&mut self) {
        self.text_matrix = TEXT_MATRIX_IDENTITY;
        self.line_matrix = TEXT_MATRIX_IDENTITY;
    }

    /// `Td tx ty` — move text position by (tx, ty).
    ///
    /// Updates both the text matrix and the line matrix.
    pub fn move_by(&mut self, tx: f64, ty: f64) {
        // New line matrix = translate(tx, ty) × old line matrix
        self.line_matrix[4] += tx.mul_add(self.line_matrix[0], ty * self.line_matrix[2]);
        self.line_matrix[5] += tx.mul_add(self.line_matrix[1], ty * self.line_matrix[3]);
        self.text_matrix = self.line_matrix;
    }

    /// `TD tx ty` — move by (tx, ty) and set leading to −ty.
    pub fn move_by_and_set_leading(&mut self, tx: f64, ty: f64) {
        self.leading = -ty;
        self.move_by(tx, ty);
    }

    /// `T*` — move to start of next line (equivalent to `0 −TL Td`).
    pub fn next_line(&mut self) {
        self.move_by(0.0, -self.leading);
    }

    /// `Tm` — set both text matrix and line matrix directly.
    pub const fn set_matrix(&mut self, m: TextMatrix) {
        self.text_matrix = m;
        self.line_matrix = m;
    }

    /// Return the current text origin in user space (the `e` and `f` components
    /// of the text matrix, which encode the translation part).
    #[must_use]
    pub const fn origin(&self) -> (f64, f64) {
        (self.text_matrix[4], self.text_matrix[5])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[expect(
        clippy::float_cmp,
        reason = "begin_text must reset to the bit-exact identity matrix constant; a partial reset that drifted by 1 ULP from the literal would still be a bug"
    )]
    fn begin_text_resets_matrices() {
        let mut ts = TextState::default();
        ts.text_matrix[4] = 100.0;
        ts.begin_text();
        assert_eq!(ts.text_matrix, TEXT_MATRIX_IDENTITY);
        assert_eq!(ts.line_matrix, TEXT_MATRIX_IDENTITY);
    }

    #[test]
    fn move_by_updates_origin() {
        let mut ts = TextState::default();
        ts.move_by(50.0, -12.0);
        let (x, y) = ts.origin();
        assert!((x - 50.0).abs() < 1e-9, "x={x}");
        assert!((y - (-12.0)).abs() < 1e-9, "y={y}");
    }

    #[test]
    fn next_line_uses_leading() {
        let mut ts = TextState {
            leading: 14.0,
            ..TextState::default()
        };
        ts.move_by(0.0, 100.0); // move to some y first
        let (_, y0) = ts.origin();
        ts.next_line();
        let (_, y1) = ts.origin();
        assert!((y0 - y1 - 14.0).abs() < 1e-9, "y0={y0} y1={y1}");
    }

    #[test]
    fn td_sets_leading() {
        let mut ts = TextState::default();
        ts.move_by_and_set_leading(0.0, -12.0);
        assert!((ts.leading - 12.0).abs() < 1e-9);
    }
}
