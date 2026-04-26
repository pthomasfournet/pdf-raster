//! PDF graphics state for the interpreter.
//!
//! [`InterpGState`] tracks the PDF-level state that sits *above* the raster
//! crate's [`raster::GraphicsState`]: current colour, text state, CTM, and
//! the path being constructed. The raster state is rebuilt from this when a
//! painting operator fires.
//!
//! The save/restore stack is a plain `Vec`; `q` pushes a clone, `Q` pops.

use raster::types::{LineCap, LineJoin};

use super::color::RasterColor;
use super::text::TextState;

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
        a[0] * b[0] + a[1] * b[2],
        a[0] * b[1] + a[1] * b[3],
        a[2] * b[0] + a[3] * b[2],
        a[2] * b[1] + a[3] * b[3],
        a[4] * b[0] + a[5] * b[2] + b[4],
        a[4] * b[1] + a[5] * b[3] + b[5],
    ]
}

/// Transform a point `(x, y)` by `ctm`, returning `(x', y')`.
#[must_use]
pub fn ctm_transform(ctm: &Ctm, x: f64, y: f64) -> (f64, f64) {
    (
        ctm[0] * x + ctm[2] * y + ctm[4],
        ctm[1] * x + ctm[3] * y + ctm[5],
    )
}

/// PDF graphics state at the interpreter level.
///
/// One instance lives per save/restore level; the stack is managed by
/// [`GStateStack`].
#[derive(Debug, Clone)]
pub struct InterpGState {
    /// Current transformation matrix.
    pub ctm: Ctm,
    /// Current fill colour.
    pub fill_color: RasterColor,
    /// Current stroke colour.
    pub stroke_color: RasterColor,
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
}

impl Default for InterpGState {
    /// PDF initial graphics state (§8.4.4 Table 52).
    fn default() -> Self {
        Self {
            ctm: CTM_IDENTITY,
            fill_color: RasterColor::default(),   // black
            stroke_color: RasterColor::default(), // black
            line_width: 1.0,
            line_cap: LineCap::Butt,
            line_join: LineJoin::Miter,
            miter_limit: 10.0,
            flatness: 0.0,
            dash: (Vec::new(), 0.0), // solid line
            text: TextState::default(),
        }
    }
}

/// Save/restore stack for [`InterpGState`].
///
/// `q` pushes a clone; `Q` pops. Unmatched `Q` operators are silently ignored
/// to be lenient with real-world PDFs that have mismatched save/restore counts.
#[derive(Debug, Default)]
pub struct GStateStack {
    /// Index 0 = bottom (page-level) state; last = current.
    stack: Vec<InterpGState>,
}

impl GStateStack {
    /// Create a new stack with the initial PDF graphics state at the bottom.
    #[must_use]
    pub fn new() -> Self {
        Self {
            stack: vec![InterpGState::default()],
        }
    }

    /// Return a shared reference to the current graphics state.
    #[must_use]
    pub fn current(&self) -> &InterpGState {
        // SAFETY: stack always has at least one element (the page-level state).
        self.stack.last().expect("GStateStack is always non-empty")
    }

    /// Return a mutable reference to the current graphics state.
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
            self.stack.pop();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
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
        let mut stack = GStateStack::new();
        stack.current_mut().line_width = 5.0;
        stack.save();
        stack.current_mut().line_width = 10.0;
        assert!((stack.current().line_width - 10.0).abs() < 1e-9);
        stack.restore();
        assert!((stack.current().line_width - 5.0).abs() < 1e-9);
    }

    #[test]
    fn restore_unmatched_is_silent() {
        let mut stack = GStateStack::new();
        stack.restore(); // should not panic
        stack.restore(); // still should not panic
    }
}
