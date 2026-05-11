//! PDF graphics state for the interpreter.
//!
//! [`InterpGState`] tracks the PDF-level state that sits *above* the raster
//! crate's [`raster::GraphicsState`]: current colour, text state, CTM, and
//! the path being constructed. The raster state is rebuilt from this when a
//! painting operator fires.
//!
//! The save/restore stack is a plain `Vec`; `q` pushes a clone, `Q` pops.

use raster::Clip;
use raster::types::{BlendMode, LineCap, LineJoin};

use super::color::RasterColor;
use super::text::TextState;
use crate::resources::colorspace::ColorSpace;

/// PDF current transformation matrix — a 6-element `[a b c d e f]` array.
pub type Ctm = [f64; 6];

/// The identity CTM.
pub const CTM_IDENTITY: Ctm = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];

/// Multiply two CTMs: result = `a * b` (both in column-major PDF order).
#[must_use]
pub fn ctm_multiply(a: &Ctm, b: &Ctm) -> Ctm {
    // PDF matrix multiplication (§8.3.4):
    // | a0 a1 0 |   | b0 b1 0 |
    // | a2 a3 0 | × | b2 b3 0 |
    // | a4 a5 1 |   | b4 b5 1 |
    [
        a[0].mul_add(b[0], a[1] * b[2]),
        a[0].mul_add(b[1], a[1] * b[3]),
        a[2].mul_add(b[0], a[3] * b[2]),
        a[2].mul_add(b[1], a[3] * b[3]),
        a[4].mul_add(b[0], a[5] * b[2]) + b[4],
        a[4].mul_add(b[1], a[5] * b[3]) + b[5],
    ]
}

/// Multiply the 2×2 linear parts of two CTMs (`a × b`), returning `[a,b,c,d]`.
///
/// A CTM is stored as `[a, b, c, d, e, f]` (PDF §8.3.4).  Indices 4–5 are the
/// translation vector and are dropped; only indices 0–3 (the linear submatrix)
/// participate in the product.  The result is a 4-element array in the same
/// `[a, b, c, d]` layout.
///
/// Used to compute `Trm[2×2] = font_size × Tm[2×2] × CTM[2×2]` — the size and
/// orientation of rendered glyphs in device space.
#[must_use]
pub fn mat2x2_mul(a: &Ctm, b: &Ctm) -> [f64; 4] {
    // Standard row-major 2×2 matrix product:
    // result[row][col] = Σ_k  a[row][k] × b[k][col]
    // With PDF [a,b,c,d] layout: row 0 = [a[0],a[1]], row 1 = [a[2],a[3]].
    [
        a[0].mul_add(b[0], a[1] * b[2]),
        a[0].mul_add(b[1], a[1] * b[3]),
        a[2].mul_add(b[0], a[3] * b[2]),
        a[2].mul_add(b[1], a[3] * b[3]),
    ]
}

/// Transform a point `(x, y)` by `ctm`, returning `(x', y')`.
#[must_use]
pub fn ctm_transform(ctm: &Ctm, x: f64, y: f64) -> (f64, f64) {
    (
        ctm[0].mul_add(x, ctm[2] * y) + ctm[4],
        ctm[1].mul_add(x, ctm[3] * y) + ctm[5],
    )
}

/// PDF graphics state at the interpreter level.
///
/// One instance lives per save/restore level; the stack is managed by
/// [`GStateStack`].
///
/// `Clone` is implemented manually so that `clip` uses `clone_shared`
/// (Arc-sharing scanners) rather than a deep copy.
pub struct InterpGState {
    /// Current transformation matrix.
    pub ctm: Ctm,
    /// Current clip region (page rect at init; narrowed by `W`/`W*` operators).
    ///
    /// Constructed with `antialias: true` so path-clip scanners are built in
    /// AA-scaled coordinates, matching the `vector_antialias = true` mode used
    /// by fill and stroke.
    pub clip: Clip,
    /// Current fill colour.
    pub fill_color: RasterColor,
    /// Current stroke colour.
    pub stroke_color: RasterColor,
    /// Fill opacity (PDF `ca`): 0 = transparent, 255 = opaque.
    pub fill_alpha: u8,
    /// Stroke opacity (PDF `CA`): 0 = transparent, 255 = opaque.
    pub stroke_alpha: u8,
    /// Compositing blend mode (PDF `BM`).  Defaults to `Normal`.
    pub blend_mode: BlendMode,
    /// Line width in user-space units.
    pub line_width: f64,
    /// Line cap style.
    pub line_cap: LineCap,
    /// Line join style.
    pub line_join: LineJoin,
    /// Miter limit.
    pub miter_limit: f64,
    /// Flatness tolerance (0 = device default).
    pub flatness: f64,
    /// Dash pattern: (lengths, phase).
    pub dash: (Vec<f64>, f64),
    /// Text state (font, size, spacing, matrices).
    pub text: TextState,
    /// Active fill pattern name (key into the page `Pattern` resource dict).
    ///
    /// Set by `scn /Name`; cleared when any other fill-colour operator fires.
    /// `None` means use `fill_color` as a solid colour.
    pub fill_pattern: Option<Vec<u8>>,
    /// Tint components for an uncoloured fill pattern (`PaintType` 2).
    pub fill_pattern_components: Vec<f64>,
    /// Active stroke pattern name (key into the page `Pattern` resource dict).
    ///
    /// Set by `SCN /Name`; cleared when any other stroke-colour operator fires.
    /// `None` means use `stroke_color` as a solid colour.
    pub stroke_pattern: Option<Vec<u8>>,
    /// Tint components for an uncoloured stroke pattern (`PaintType` 2).
    pub stroke_pattern_components: Vec<f64>,
    /// Declared colour space for fill operations (PDF §8.6).
    ///
    /// Set by `cs /Name`; defaults to `DeviceGray` per §8.6.4.1.  Used by the
    /// uncoloured-pattern tint dispatch to interpret `fill_pattern_components`
    /// in the spec-correct base space rather than guessing from component
    /// count.  No effect on direct `g`/`rg`/`k` operators (they bypass the
    /// colour-space slot per spec).
    pub fill_color_space: ColorSpace,
    /// Declared colour space for stroke operations (PDF §8.6).  Mirrors
    /// `fill_color_space`; set by `CS /Name`.
    pub stroke_color_space: ColorSpace,
}

impl Clone for InterpGState {
    fn clone(&self) -> Self {
        Self {
            ctm: self.ctm,
            clip: self.clip.clone_shared(),
            fill_color: self.fill_color.clone(),
            stroke_color: self.stroke_color.clone(),
            fill_alpha: self.fill_alpha,
            stroke_alpha: self.stroke_alpha,
            blend_mode: self.blend_mode,
            line_width: self.line_width,
            line_cap: self.line_cap,
            line_join: self.line_join,
            miter_limit: self.miter_limit,
            flatness: self.flatness,
            dash: self.dash.clone(),
            text: self.text.clone(),
            fill_pattern: self.fill_pattern.clone(),
            fill_pattern_components: self.fill_pattern_components.clone(),
            stroke_pattern: self.stroke_pattern.clone(),
            stroke_pattern_components: self.stroke_pattern_components.clone(),
            fill_color_space: self.fill_color_space.clone(),
            stroke_color_space: self.stroke_color_space.clone(),
        }
    }
}

impl InterpGState {
    /// Create the initial graphics state for a page of `width × height` device pixels.
    #[must_use]
    pub fn initial(width: u32, height: u32) -> Self {
        Self {
            ctm: CTM_IDENTITY,
            clip: Clip::new(0.0, 0.0, f64::from(width), f64::from(height), true),
            fill_color: RasterColor::default(),
            stroke_color: RasterColor::default(),
            fill_alpha: 255,
            stroke_alpha: 255,
            blend_mode: BlendMode::Normal,
            line_width: 1.0,
            line_cap: LineCap::Butt,
            line_join: LineJoin::Miter,
            miter_limit: 10.0,
            flatness: 0.0,
            dash: (Vec::new(), 0.0),
            text: TextState::default(),
            fill_pattern: None,
            fill_pattern_components: Vec::new(),
            stroke_pattern: None,
            stroke_pattern_components: Vec::new(),
            // PDF §8.6.4.1: DeviceGray is the initial colour space for both
            // stroking and non-stroking operations until the first `cs`/`CS`.
            fill_color_space: ColorSpace::DeviceGray,
            stroke_color_space: ColorSpace::DeviceGray,
        }
    }
}

/// Save/restore stack for [`InterpGState`].
///
/// `q` pushes a clone; `Q` pops. Unmatched `Q` operators are silently ignored
/// to be lenient with real-world PDFs that have mismatched save/restore counts.
pub struct GStateStack {
    /// Index 0 = bottom (page-level) state; last = current.
    stack: Vec<InterpGState>,
}

impl GStateStack {
    /// Create a new stack with the initial PDF graphics state for a page of
    /// `width × height` device pixels.
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            stack: vec![InterpGState::initial(width, height)],
        }
    }

    /// Return a shared reference to the current graphics state.
    ///
    /// # Panics
    ///
    /// Never panics in practice — the stack always contains at least the
    /// page-level state pushed by [`GStateStack::new`].
    #[must_use]
    pub fn current(&self) -> &InterpGState {
        // SAFETY: stack always has at least one element (the page-level state).
        self.stack.last().expect("GStateStack is always non-empty")
    }

    /// Return a mutable reference to the current graphics state.
    ///
    /// # Panics
    ///
    /// Never panics in practice — the stack always contains at least the
    /// page-level state pushed by [`GStateStack::new`].
    pub fn current_mut(&mut self) -> &mut InterpGState {
        self.stack
            .last_mut()
            .expect("GStateStack is always non-empty")
    }

    /// `q` — push a copy of the current state.
    pub fn save(&mut self) {
        let clone = self.current().clone();
        self.stack.push(clone);
    }

    /// `Q` — pop the current state. Silently ignores an unmatched `Q`.
    pub fn restore(&mut self) {
        // Never pop the bottom state; unmatched Q is lenient.
        if self.stack.len() > 1 {
            let _ = self.stack.pop();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[expect(
        clippy::float_cmp,
        reason = "identity × b must equal b bit-exactly: IEEE 754 guarantees 1.0*x == x and 0.0+x == x for finite x, so any drift here means ctm_multiply has a real bug (e.g. wrong index order)"
    )]
    fn ctm_identity_multiply() {
        let a = CTM_IDENTITY;
        let b = [2.0, 0.0, 0.0, 3.0, 10.0, 20.0];
        let r = ctm_multiply(&a, &b);
        assert_eq!(r, b);
    }

    #[test]
    fn ctm_transform_translation() {
        let ctm = [1.0, 0.0, 0.0, 1.0, 100.0, 200.0];
        let (x, y) = ctm_transform(&ctm, 0.0, 0.0);
        assert!((x - 100.0).abs() < 1e-9);
        assert!((y - 200.0).abs() < 1e-9);
    }

    #[test]
    fn save_restore_roundtrip() {
        let mut stack = GStateStack::new(100, 100);
        stack.current_mut().line_width = 5.0;
        stack.save();
        stack.current_mut().line_width = 10.0;
        assert!((stack.current().line_width - 10.0).abs() < 1e-9);
        stack.restore();
        assert!((stack.current().line_width - 5.0).abs() < 1e-9);
    }

    #[test]
    fn restore_unmatched_is_silent() {
        let mut stack = GStateStack::new(100, 100);
        stack.restore(); // should not panic
        stack.restore(); // still should not panic
    }

    #[test]
    fn mat2x2_mul_identity() {
        let id: Ctm = CTM_IDENTITY;
        let b: Ctm = [2.0, 3.0, 4.0, 5.0, 0.0, 0.0];
        let r = mat2x2_mul(&id, &b);
        // Identity × B should equal B's 2×2 part.
        assert!((r[0] - 2.0).abs() < 1e-12);
        assert!((r[1] - 3.0).abs() < 1e-12);
        assert!((r[2] - 4.0).abs() < 1e-12);
        assert!((r[3] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn mat2x2_mul_scaling() {
        // [2, 0, 0, 3, ·, ·] × [4, 0, 0, 5, ·, ·] = [8, 0, 0, 15, ·, ·]
        let a: Ctm = [2.0, 0.0, 0.0, 3.0, 99.0, 99.0]; // translations ignored
        let b: Ctm = [4.0, 0.0, 0.0, 5.0, 99.0, 99.0];
        let r = mat2x2_mul(&a, &b);
        assert!((r[0] - 8.0).abs() < 1e-12);
        assert!((r[1] - 0.0).abs() < 1e-12);
        assert!((r[2] - 0.0).abs() < 1e-12);
        assert!((r[3] - 15.0).abs() < 1e-12);
    }

    #[test]
    fn mat2x2_mul_rotation() {
        // 90° rotation matrix: [0,-1, 1, 0] (in PDF [a,b,c,d] = [0,-1,1,0])
        // Squaring it should give 180° = [-1, 0, 0, -1].
        let rot90: Ctm = [0.0, -1.0, 1.0, 0.0, 0.0, 0.0];
        let r = mat2x2_mul(&rot90, &rot90);
        assert!((r[0] - (-1.0)).abs() < 1e-12);
        assert!((r[1] - 0.0).abs() < 1e-12);
        assert!((r[2] - 0.0).abs() < 1e-12);
        assert!((r[3] - (-1.0)).abs() < 1e-12);
    }

    #[test]
    fn initial_clip_covers_page() {
        let stack = GStateStack::new(200, 100);
        let clip = &stack.current().clip;
        // Initial clip rect should span the full page in device pixels.
        assert!(clip.x_min < 1.0);
        assert!(clip.y_min < 1.0);
        assert!(clip.x_max > 199.0);
        assert!(clip.y_max > 99.0);
    }

    #[test]
    fn initial_clip_uses_antialias() {
        // Clip must be created with antialias=true so path-clip scanners are
        // built in AA-scaled coordinates, matching vector_antialias=true in fill/stroke.
        let stack = GStateStack::new(100, 100);
        assert!(stack.current().clip.antialias);
    }

    #[test]
    fn clip_narrows_on_save_restore() {
        let mut stack = GStateStack::new(100, 100);
        let original_xmax = stack.current().clip.x_max;

        // Narrow the clip in the current state.
        stack.current_mut().clip.clip_to_rect(0.0, 0.0, 50.0, 50.0);
        assert!(stack.current().clip.x_max <= 50.0);

        // Save, narrow further, then restore — should return to the 50-wide clip.
        stack.save();
        stack.current_mut().clip.clip_to_rect(0.0, 0.0, 10.0, 10.0);
        assert!(stack.current().clip.x_max <= 10.0);
        stack.restore();
        // After restore: should be back to the 50-wide clip, not the original page rect.
        assert!(stack.current().clip.x_max <= 50.0 + 1.0);
        assert!(stack.current().clip.x_max < original_xmax);
    }
}
