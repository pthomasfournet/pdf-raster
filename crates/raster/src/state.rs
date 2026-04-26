//! Graphics state and save/restore stack.
//!
//! [`GraphicsState`] is the Rust equivalent of `SplashState` from
//! `splash/SplashState.h/.cc`, with the linked-list `*next` pointer replaced
//! by a [`StateStack`] that owns a `Vec<GraphicsState>`.
//!
//! ## Default values
//!
//! All defaults match the `SplashState` constructor exactly:
//! - CTM = identity `[1, 0, 0, 1, 0, 0]`
//! - Stroke/fill alpha = 1.0
//! - Line cap = Butt, line join = Miter, miter limit = 10.0, flatness = 1.0
//! - All transfer LUTs = identity
//! - Overprint mask = 0xFFFF_FFFF
//! - Clip rect = `[0, 0, width-0.001, height-0.001]` (intentional sub-pixel inset)
//!
//! ## `set_transfer` semantics
//!
//! Matches `SplashState::setTransfer`: the CMYK and DeviceN[0..4] LUTs are
//! derived from the **inverted** RGB/gray LUTs *before* the RGB/gray LUTs are
//! overwritten. See `SplashState.cc` for the detailed rationale.

use crate::bitmap::AnyBitmap;
use crate::clip::Clip;
use crate::types::{LineCap, LineJoin, ScreenParams, SPOT_NCOMPS};
use color::TransferLut;

// ── GraphicsState ─────────────────────────────────────────────────────────────

/// The complete graphics state for one rendering context.
///
/// Patterns are stubbed as `()` for Phase 1; Phase 2 replaces them with
/// `Box<dyn Pattern>`.
pub struct GraphicsState {
    // Current transformation matrix (column-vector 2-D affine).
    pub matrix: [f64; 6],

    // Patterns (Phase 2 placeholder).
    // pub stroke_pattern: Box<dyn Pattern>,
    // pub fill_pattern:   Box<dyn Pattern>,
    pub screen: ScreenParams,

    pub stroke_alpha: f64,
    pub fill_alpha: f64,
    pub multiply_pattern_alpha: bool,
    pub pattern_stroke_alpha: f64,
    pub pattern_fill_alpha: f64,

    pub line_width: f64,
    pub line_cap: LineCap,
    pub line_join: LineJoin,
    pub miter_limit: f64,
    pub flatness: f64,

    pub line_dash: Vec<f64>,
    pub line_dash_phase: f64,
    pub stroke_adjust: bool,

    pub clip: Clip,

    /// Soft mask bitmap (None if no soft mask is active).
    pub soft_mask: Option<Box<AnyBitmap>>,
    /// True when the state owns `soft_mask` and should drop it on restore.
    pub delete_soft_mask: bool,

    pub in_non_isolated_group: bool,
    pub in_knockout_group: bool,
    pub fill_overprint: bool,
    pub stroke_overprint: bool,
    pub overprint_mode: i32,

    /// Transfer LUTs — RGB channels (R=0, G=1, B=2).
    pub rgb_transfer: [TransferLut; 3],
    pub gray_transfer: TransferLut,
    /// Transfer LUTs — CMYK channels (C=0, M=1, Y=2, K=3).
    pub cmyk_transfer: [TransferLut; 4],
    /// Transfer LUTs — DeviceN channels (indices 0..SPOT_NCOMPS+4 = 8).
    pub device_n_transfer: Vec<[u8; 256]>,

    pub overprint_mask: u32,
    pub overprint_additive: bool,
}

impl GraphicsState {
    /// Construct the default state for a new page.
    ///
    /// `clip` is set to `[0, 0, width-0.001, height-0.001]` — the intentional
    /// 0.001 inset matches `SplashState` constructor and avoids edge-pixel issues.
    pub fn new(width: u32, height: u32, vector_antialias: bool) -> Self {
        let clip = Clip::new(
            0.0,
            0.0,
            width as f64 - 0.001,
            height as f64 - 0.001,
            vector_antialias,
        );
        Self {
            matrix: [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            screen: ScreenParams::default(),
            stroke_alpha: 1.0,
            fill_alpha: 1.0,
            multiply_pattern_alpha: false,
            pattern_stroke_alpha: 1.0,
            pattern_fill_alpha: 1.0,
            line_width: 1.0,
            line_cap: LineCap::Butt,
            line_join: LineJoin::Miter,
            miter_limit: 10.0,
            flatness: 1.0,
            line_dash: Vec::new(),
            line_dash_phase: 0.0,
            stroke_adjust: false,
            clip,
            soft_mask: None,
            delete_soft_mask: false,
            in_non_isolated_group: false,
            in_knockout_group: false,
            fill_overprint: false,
            stroke_overprint: false,
            overprint_mode: 0,
            rgb_transfer: [
                TransferLut::IDENTITY,
                TransferLut::IDENTITY,
                TransferLut::IDENTITY,
            ],
            gray_transfer: TransferLut::IDENTITY,
            cmyk_transfer: [
                TransferLut::IDENTITY,
                TransferLut::IDENTITY,
                TransferLut::IDENTITY,
                TransferLut::IDENTITY,
            ],
            device_n_transfer: (0..SPOT_NCOMPS + 4)
                .map(|_| {
                    let mut lut = [0u8; 256];
                    for (i, v) in lut.iter_mut().enumerate() {
                        *v = i as u8;
                    }
                    lut
                })
                .collect(),
            overprint_mask: 0xFFFF_FFFF,
            overprint_additive: false,
        }
    }

    /// Apply new RGB and gray transfer functions, deriving CMYK and DeviceN[0..4]
    /// from the **inverted** current RGB/gray values before overwriting them.
    ///
    /// Matches `SplashState::setTransfer` in `SplashState.cc`:
    /// ```text
    /// cmykTransferC[i] = 255 - rgbTransferR[255 - i]   (before overwrite)
    /// deviceNTransfer[0][i] = 255 - rgbTransferR[255 - i]
    /// …
    /// ```
    pub fn set_transfer(&mut self, r: &[u8; 256], g: &[u8; 256], b: &[u8; 256], gray: &[u8; 256]) {
        // Derive CMYK and DeviceN[0..4] from the CURRENT (pre-overwrite) LUTs.
        let mut cc = [0u8; 256];
        let mut cm = [0u8; 256];
        let mut cy = [0u8; 256];
        let mut ck = [0u8; 256];
        for i in 0usize..256 {
            cc[i] = 255 - self.rgb_transfer[0].0[255 - i];
            cm[i] = 255 - self.rgb_transfer[1].0[255 - i];
            cy[i] = 255 - self.rgb_transfer[2].0[255 - i];
            ck[i] = 255 - self.gray_transfer.0[255 - i];
        }
        self.cmyk_transfer = [
            TransferLut(cc),
            TransferLut(cm),
            TransferLut(cy),
            TransferLut(ck),
        ];
        for i in 0usize..256 {
            self.device_n_transfer[0][i] = 255 - self.rgb_transfer[0].0[255 - i];
            self.device_n_transfer[1][i] = 255 - self.rgb_transfer[1].0[255 - i];
            self.device_n_transfer[2][i] = 255 - self.rgb_transfer[2].0[255 - i];
            self.device_n_transfer[3][i] = 255 - self.gray_transfer.0[255 - i];
        }
        // Now overwrite RGB and gray.
        self.rgb_transfer[0] = TransferLut(*r);
        self.rgb_transfer[1] = TransferLut(*g);
        self.rgb_transfer[2] = TransferLut(*b);
        self.gray_transfer = TransferLut(*gray);
    }

    /// Clone this state for `save()`.
    ///
    /// Clip scanners are shared via `Arc` (matching C++ `shared_ptr` semantics).
    /// The soft mask is NOT inherited — the new state starts with `soft_mask = None`.
    pub fn save_clone(&self) -> Self {
        Self {
            matrix: self.matrix,
            screen: self.screen,
            stroke_alpha: self.stroke_alpha,
            fill_alpha: self.fill_alpha,
            multiply_pattern_alpha: self.multiply_pattern_alpha,
            pattern_stroke_alpha: self.pattern_stroke_alpha,
            pattern_fill_alpha: self.pattern_fill_alpha,
            line_width: self.line_width,
            line_cap: self.line_cap,
            line_join: self.line_join,
            miter_limit: self.miter_limit,
            flatness: self.flatness,
            line_dash: self.line_dash.clone(),
            line_dash_phase: self.line_dash_phase,
            stroke_adjust: self.stroke_adjust,
            clip: self.clip.clone_shared(),
            soft_mask: None, // new state does not own the parent's soft mask
            delete_soft_mask: false,
            in_non_isolated_group: self.in_non_isolated_group,
            in_knockout_group: self.in_knockout_group,
            fill_overprint: self.fill_overprint,
            stroke_overprint: self.stroke_overprint,
            overprint_mode: self.overprint_mode,
            rgb_transfer: self.rgb_transfer.clone(),
            gray_transfer: self.gray_transfer.clone(),
            cmyk_transfer: self.cmyk_transfer.clone(),
            device_n_transfer: self.device_n_transfer.clone(),
            overprint_mask: self.overprint_mask,
            overprint_additive: self.overprint_additive,
        }
    }
}

// ── StateStack ────────────────────────────────────────────────────────────────

/// A Vec-based save/restore stack of [`GraphicsState`] values.
///
/// Replaces the raw-pointer linked list (`SplashState *next`) in the C++
/// original. The stack always has at least one entry (the initial state).
pub struct StateStack {
    stack: Vec<GraphicsState>,
}

impl StateStack {
    /// Create a new stack with `initial` as the bottom state.
    pub fn new(initial: GraphicsState) -> Self {
        Self {
            stack: vec![initial],
        }
    }

    /// Borrow the current (top) state.
    pub fn current(&self) -> &GraphicsState {
        self.stack.last().expect("StateStack is empty")
    }

    /// Mutably borrow the current state.
    pub fn current_mut(&mut self) -> &mut GraphicsState {
        self.stack.last_mut().expect("StateStack is empty")
    }

    /// Save: push a clone of the current state.
    pub fn save(&mut self) {
        let cloned = self.stack.last().expect("StateStack is empty").save_clone();
        self.stack.push(cloned);
    }

    /// Restore: pop the top state. Returns `false` if the stack is at the bottom.
    pub fn restore(&mut self) -> bool {
        if self.stack.len() <= 1 {
            return false;
        }
        self.stack.pop();
        true
    }

    /// Current nesting depth (1 = only the initial state, no saves).
    pub fn depth(&self) -> usize {
        self.stack.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_matrix_is_identity() {
        let s = GraphicsState::new(100, 100, false);
        assert_eq!(s.matrix, [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn default_alpha_is_one() {
        let s = GraphicsState::new(100, 100, false);
        assert_eq!(s.stroke_alpha, 1.0);
        assert_eq!(s.fill_alpha, 1.0);
    }

    #[test]
    fn default_overprint_mask() {
        let s = GraphicsState::new(100, 100, false);
        assert_eq!(s.overprint_mask, 0xFFFF_FFFF);
    }

    #[test]
    fn default_transfer_is_identity() {
        let s = GraphicsState::new(100, 100, false);
        for i in 0u8..=255 {
            assert_eq!(s.rgb_transfer[0].apply(i), i);
            assert_eq!(s.gray_transfer.apply(i), i);
        }
    }

    #[test]
    fn clip_has_sub_pixel_inset() {
        let s = GraphicsState::new(200, 100, false);
        // x_max < 200.0, y_max < 100.0
        assert!(s.clip.x_max < 200.0);
        assert!(s.clip.y_max < 100.0);
    }

    #[test]
    fn save_restore_roundtrip() {
        let initial = GraphicsState::new(100, 100, false);
        let mut stack = StateStack::new(initial);
        assert_eq!(stack.depth(), 1);

        stack.save();
        assert_eq!(stack.depth(), 2);
        stack.current_mut().line_width = 5.0;

        assert!(stack.restore());
        assert_eq!(stack.depth(), 1);
        assert_eq!(stack.current().line_width, 1.0); // restored to default
    }

    #[test]
    fn restore_at_bottom_returns_false() {
        let initial = GraphicsState::new(100, 100, false);
        let mut stack = StateStack::new(initial);
        assert!(!stack.restore());
        assert_eq!(stack.depth(), 1);
    }

    #[test]
    fn set_transfer_derives_cmyk() {
        let mut s = GraphicsState::new(100, 100, false);
        // Inversion LUT: i → 255-i
        let inv: [u8; 256] = std::array::from_fn(|i| 255 - i as u8);
        s.set_transfer(&inv, &inv, &inv, &inv);
        // CMYK C = 255 - R[255-i] (before overwrite of R); with identity R:
        // R[255-i] = 255-i, so C[i] = 255-(255-i) = i → identity
        for i in 0u8..=255 {
            assert_eq!(s.cmyk_transfer[0].apply(i), i, "cmyk C[{i}]");
        }
    }
}
