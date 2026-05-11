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
//! - Overprint mask = `0xFFFF_FFFF`
//! - Clip rect = `[0, 0, width-0.001, height-0.001]` (intentional sub-pixel inset)
//!
//! ## `set_transfer` semantics
//!
//! Matches `SplashState::setTransfer`: the CMYK and `DeviceN`[0..4] LUTs are
//! derived from the **inverted** RGB/gray LUTs *before* the RGB/gray LUTs are
//! overwritten. See `SplashState.cc` for the detailed rationale.

use crate::bitmap::AnyBitmap;
use crate::clip::Clip;
use crate::types::{LineCap, LineJoin, SPOT_NCOMPS, ScreenParams};
use color::TransferLut;

// ── StateFlags ────────────────────────────────────────────────────────────────

/// Boolean flags packed into a single byte to avoid `struct_excessive_bools`.
///
/// Only `DELETE_SOFT_MASK` is currently used (cleared in `clone_for_xobject`).
/// Reintroduce named bits as concrete consumers appear.
#[derive(Copy, Clone, Debug, Default)]
pub struct StateFlags {
    bits: u8,
}

impl StateFlags {
    const DELETE_SOFT_MASK: u8 = 1 << 2;

    /// Sets whether the soft mask should be deleted on the next state restore.
    pub const fn set_delete_soft_mask(&mut self, v: bool) {
        if v {
            self.bits |= Self::DELETE_SOFT_MASK;
        } else {
            self.bits &= !Self::DELETE_SOFT_MASK;
        }
    }
}

// ── GraphicsState ─────────────────────────────────────────────────────────────

/// The complete graphics state for one rendering context.
///
/// Patterns are stubbed as `()` for Phase 1; Phase 2 replaces them with
/// `Box<dyn Pattern>`.
pub struct GraphicsState {
    /// Current transformation matrix in column-vector 2-D affine form: `[a, b, c, d, e, f]`.
    pub matrix: [f64; 6],

    // Patterns (Phase 2 placeholder).
    // pub stroke_pattern: Box<dyn Pattern>,
    // pub fill_pattern:   Box<dyn Pattern>,
    /// Halftone screen parameters controlling frequency, angle, and spot function.
    pub screen: ScreenParams,

    /// Opacity for stroking operations. `0.0` = fully transparent, `1.0` = fully opaque.
    pub stroke_alpha: f64,
    /// Opacity for fill operations. `0.0` = fully transparent, `1.0` = fully opaque.
    pub fill_alpha: f64,
    /// Effective stroke alpha after multiplying in pattern alpha (used when `multiply_pattern_alpha` is set).
    pub pattern_stroke_alpha: f64,
    /// Effective fill alpha after multiplying in pattern alpha (used when `multiply_pattern_alpha` is set).
    pub pattern_fill_alpha: f64,

    /// Stroke line width in user-space units; must be ≥ 0.
    pub line_width: f64,
    /// Style of line end caps (butt, round, or square).
    pub line_cap: LineCap,
    /// Style of line joins (miter, round, or bevel).
    pub line_join: LineJoin,
    /// Maximum ratio of miter length to line width before a miter join is beveled; default 10.0.
    pub miter_limit: f64,
    /// Maximum permitted distance between the path and the approximating line segments; default 1.0.
    pub flatness: f64,

    /// Dash pattern array: alternating on/off lengths in user-space units.
    pub line_dash: Vec<f64>,
    /// Offset into the dash pattern at which stroking begins.
    pub line_dash_phase: f64,

    /// Active clipping region; updated by clip operators.
    pub clip: Clip,

    /// Soft mask bitmap (None if no soft mask is active).
    pub soft_mask: Option<Box<AnyBitmap>>,

    /// PDF overprint mode: `0` = `CompatibleOverprint`, `1` = `IsolatePaint`.
    pub overprint_mode: i32,

    /// Transfer LUTs — RGB channels (R=0, G=1, B=2).
    pub rgb_transfer: [TransferLut; 3],
    /// Transfer LUT for the gray channel.
    pub gray_transfer: TransferLut,
    /// Transfer LUTs — CMYK channels (C=0, M=1, Y=2, K=3).
    pub cmyk_transfer: [TransferLut; 4],
    /// Transfer LUTs — `DeviceN` channels (indices `0..SPOT_NCOMPS+4` = 8).
    pub device_n_transfer: Vec<[u8; 256]>,

    /// Bitmask of color components that participate in overprinting; default `0xFFFF_FFFF` (all).
    pub overprint_mask: u32,

    /// Boolean flags (replaces individual bool fields to satisfy `struct_excessive_bools`).
    pub flags: StateFlags,
}

impl GraphicsState {
    /// Construct the default state for a new page.
    ///
    /// `clip` is set to `[0, 0, width-0.001, height-0.001]` — the intentional
    /// 0.001 inset matches `SplashState` constructor and avoids edge-pixel issues.
    ///
    /// # Panics
    ///
    /// Panics (in debug builds only) if `width` or `height` is zero.  A zero
    /// dimension would produce a negative clip bound (`0.0 - 0.001 = -0.001`),
    /// which is meaningless and almost certainly a caller bug.
    #[must_use]
    pub fn new(width: u32, height: u32, vector_antialias: bool) -> Self {
        debug_assert!(width > 0, "GraphicsState::new: width must be > 0");
        debug_assert!(height > 0, "GraphicsState::new: height must be > 0");
        let clip = Clip::new(
            0.0,
            0.0,
            f64::from(width) - 0.001,
            f64::from(height) - 0.001,
            vector_antialias,
        );
        Self {
            matrix: [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            screen: ScreenParams::default(),
            stroke_alpha: 1.0,
            fill_alpha: 1.0,
            pattern_stroke_alpha: 1.0,
            pattern_fill_alpha: 1.0,
            line_width: 1.0,
            line_cap: LineCap::Butt,
            line_join: LineJoin::Miter,
            miter_limit: 10.0,
            flatness: 1.0,
            line_dash: Vec::new(),
            line_dash_phase: 0.0,
            clip,
            soft_mask: None,
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
            device_n_transfer: vec![TransferLut::IDENTITY.0; SPOT_NCOMPS + 4],
            overprint_mask: 0xFFFF_FFFF,
            flags: StateFlags::default(),
        }
    }

    /// Apply new RGB and gray transfer functions, deriving CMYK and `DeviceN`[0..4]
    /// from the **inverted** current RGB/gray values before overwriting them.
    ///
    /// Matches `SplashState::setTransfer` in `SplashState.cc` exactly.
    ///
    /// ## Derivation
    ///
    /// CMYK uses a subtractive colour model: a CMYK value of 0 means "no ink"
    /// (full brightness) and 255 means "full ink" (zero brightness).  The
    /// relationship to the existing RGB transfer LUT is therefore:
    ///
    /// ```text
    /// Step 1 – invert the index:   look up rgb_transfer_R at position (255 - i)
    ///                               to obtain the complemented input value.
    /// Step 2 – invert the result:  subtract from 255 to flip from additive to
    ///                               subtractive space.
    ///
    /// cmykTransferC[i] = 255 - rgb_transfer_R[255 - i]   ← C mapped from R
    /// cmykTransferM[i] = 255 - rgb_transfer_G[255 - i]   ← M mapped from G
    /// cmykTransferY[i] = 255 - rgb_transfer_B[255 - i]   ← Y mapped from B
    /// cmykTransferK[i] = 255 - gray_transfer[255 - i]    ← K mapped from gray
    /// deviceNTransfer[0..=3][i]  = same as CMYK (C/M/Y/K), respectively
    /// ```
    ///
    /// Only after the CMYK/DeviceN tables have been built are `rgb_transfer` and
    /// `gray_transfer` overwritten with the new `r`, `g`, `b`, `gray` LUTs.
    pub fn set_transfer(&mut self, r: &[u8; 256], g: &[u8; 256], b: &[u8; 256], gray: &[u8; 256]) {
        // ── Step 1: Snapshot CMYK derivations from the CURRENT (pre-overwrite) LUTs ──
        //
        // Each entry uses invert_complement: cmyk[i] = 255 - rgb[255 - i].
        // This converts the additive RGB transfer into the subtractive CMYK
        // transfer in one pass, before the RGB/gray LUTs are replaced below.
        let mut cc = [0u8; 256];
        let mut cm = [0u8; 256];
        let mut cy = [0u8; 256];
        let mut ck = [0u8; 256];
        for i in 0usize..256 {
            cc[i] = 255 - self.rgb_transfer[0].0[255 - i]; // C ← invert_complement(R)
            cm[i] = 255 - self.rgb_transfer[1].0[255 - i]; // M ← invert_complement(G)
            cy[i] = 255 - self.rgb_transfer[2].0[255 - i]; // Y ← invert_complement(B)
            ck[i] = 255 - self.gray_transfer.0[255 - i]; // K ← invert_complement(gray)
        }
        // DeviceN channels 0–3 mirror CMYK C/M/Y/K exactly — copy before the
        // arrays are moved into the cmyk_transfer LUTs below.
        self.device_n_transfer[0] = cc;
        self.device_n_transfer[1] = cm;
        self.device_n_transfer[2] = cy;
        self.device_n_transfer[3] = ck;
        self.cmyk_transfer = [
            TransferLut(cc),
            TransferLut(cm),
            TransferLut(cy),
            TransferLut(ck),
        ];

        // ── Step 2: Overwrite RGB and gray with the caller-supplied LUTs ──
        self.rgb_transfer[0] = TransferLut(*r);
        self.rgb_transfer[1] = TransferLut(*g);
        self.rgb_transfer[2] = TransferLut(*b);
        self.gray_transfer = TransferLut(*gray);
    }

    /// Clone this state for `save()`.
    ///
    /// Clip scanners are shared via `Arc` (matching C++ `shared_ptr` semantics).
    /// The soft mask is NOT inherited — the new state starts with `soft_mask = None`.
    #[must_use]
    pub fn save_clone(&self) -> Self {
        Self {
            matrix: self.matrix,
            screen: self.screen,
            stroke_alpha: self.stroke_alpha,
            fill_alpha: self.fill_alpha,
            pattern_stroke_alpha: self.pattern_stroke_alpha,
            pattern_fill_alpha: self.pattern_fill_alpha,
            line_width: self.line_width,
            line_cap: self.line_cap,
            line_join: self.line_join,
            miter_limit: self.miter_limit,
            flatness: self.flatness,
            line_dash: self.line_dash.clone(),
            line_dash_phase: self.line_dash_phase,
            clip: self.clip.clone_shared(),
            soft_mask: None, // new state does not own the parent's soft mask
            overprint_mode: self.overprint_mode,
            rgb_transfer: self.rgb_transfer.clone(),
            gray_transfer: self.gray_transfer.clone(),
            cmyk_transfer: self.cmyk_transfer.clone(),
            device_n_transfer: self.device_n_transfer.clone(),
            overprint_mask: self.overprint_mask,
            flags: {
                let mut f = self.flags;
                // The new state should not inherit delete_soft_mask.
                f.set_delete_soft_mask(false);
                f
            },
        }
    }

    /// Borrow the transfer tables as a [`TransferSet`] for use in the pipe.
    ///
    /// The returned `TransferSet` borrows from `self` and is valid for the lifetime
    /// of this `GraphicsState`.
    #[must_use]
    pub fn transfer_set(&self) -> TransferSet<'_> {
        TransferSet {
            rgb: [
                self.rgb_transfer[0].as_array(),
                self.rgb_transfer[1].as_array(),
                self.rgb_transfer[2].as_array(),
            ],
            gray: self.gray_transfer.as_array(),
            cmyk: [
                self.cmyk_transfer[0].as_array(),
                self.cmyk_transfer[1].as_array(),
                self.cmyk_transfer[2].as_array(),
                self.cmyk_transfer[3].as_array(),
            ],
            device_n: &self.device_n_transfer,
        }
    }
}

// ── TransferSet ───────────────────────────────────────────────────────────────

/// Borrowed references to all transfer LUTs from a [`GraphicsState`].
///
/// Constructed via [`GraphicsState::transfer_set`] and stored in [`crate::PipeState`].
/// Avoids cloning the tables for each paint operation.
#[derive(Copy, Clone, Debug)]
pub struct TransferSet<'a> {
    /// RGB transfer tables: `[R, G, B]`, each 256 bytes.
    pub rgb: [&'a [u8; 256]; 3],
    /// Gray transfer table, 256 bytes.
    pub gray: &'a [u8; 256],
    /// CMYK transfer tables: `[C, M, Y, K]`, each 256 bytes.
    pub cmyk: [&'a [u8; 256]; 4],
    /// `DeviceN` transfer tables: `SPOT_NCOMPS + 4` tables of 256 bytes each, as a slice.
    pub device_n: &'a [[u8; 256]],
}

impl TransferSet<'_> {
    /// Return `true` if all three RGB transfer tables are the identity `v → v`.
    ///
    /// Used by the AA fast path to skip the transfer step entirely when it would
    /// be a no-op.  Checking pointer equality against the static identity table
    /// is O(1) and covers the common case; a full element-by-element comparison
    /// is the fallback for non-static tables that happen to be identity.
    #[must_use]
    pub fn is_identity_rgb(&self) -> bool {
        use color::TransferLut;
        let id = TransferLut::IDENTITY.as_array();
        self.rgb.iter().all(|lut| {
            // Fast path: same pointer (static identity table).
            std::ptr::eq(*lut, id) || *lut == id
        })
    }

    /// Return a `TransferSet` backed by identity (pass-through) arrays.
    ///
    /// Useful in tests and for the initial no-transfer state.
    /// The returned value borrows from static memory.
    #[must_use]
    pub fn identity_rgb() -> TransferSet<'static> {
        use color::TransferLut;
        // SAFETY: TransferLut::IDENTITY is a static constant; its inner [u8; 256]
        // reference is valid for 'static.
        TransferSet {
            rgb: [
                TransferLut::IDENTITY.as_array(),
                TransferLut::IDENTITY.as_array(),
                TransferLut::IDENTITY.as_array(),
            ],
            gray: TransferLut::IDENTITY.as_array(),
            cmyk: [
                TransferLut::IDENTITY.as_array(),
                TransferLut::IDENTITY.as_array(),
                TransferLut::IDENTITY.as_array(),
                TransferLut::IDENTITY.as_array(),
            ],
            device_n: {
                // A static identity table for all 8 DeviceN channels.
                static DN: [[u8; 256]; 8] = [TransferLut::IDENTITY.0; 8];
                &DN
            },
        }
    }
}

// ── StateStack ────────────────────────────────────────────────────────────────

/// A Vec-based save/restore stack of [`GraphicsState`] values.
///
/// Replaces the raw-pointer linked list (`SplashState *next`) in the C++
/// original.
///
/// ## Stack invariant
///
/// The stack **always** contains at least one entry — the initial state passed
/// to [`StateStack::new`].  This invariant is established by the constructor
/// and maintained by every method:
///
/// - [`save`](StateStack::save) only pushes (depth grows).
/// - [`restore`](StateStack::restore) refuses to pop the last entry and signals
///   this via its `bool` return value.
///
/// Because the invariant holds unconditionally, the `.last()` / `.last_mut()`
/// calls in [`current`](StateStack::current), [`current_mut`](StateStack::current_mut),
/// and [`save`](StateStack::save) can never return `None`.  The `.expect()`
/// calls there exist solely to surface a bug if the invariant is ever broken
/// during development.
pub struct StateStack {
    stack: Vec<GraphicsState>,
}

impl StateStack {
    /// Create a new stack with `initial` as the sole (bottom) state.
    ///
    /// After construction, [`depth`](StateStack::depth) returns `1`.
    #[must_use]
    pub fn new(initial: GraphicsState) -> Self {
        Self {
            stack: vec![initial],
        }
    }

    /// Borrow the current (top) state.
    ///
    /// # Panics
    ///
    /// Never panics in correct code — the stack invariant guarantees at least
    /// one entry at all times.  The `expect` is a development-time tripwire.
    #[must_use]
    pub fn current(&self) -> &GraphicsState {
        // SAFETY: stack invariant guarantees len >= 1; .last() cannot be None.
        self.stack
            .last()
            .expect("StateStack invariant violated: stack is empty")
    }

    /// Mutably borrow the current state.
    ///
    /// # Panics
    ///
    /// Never panics in correct code — the stack invariant guarantees at least
    /// one entry at all times.  The `expect` is a development-time tripwire.
    pub fn current_mut(&mut self) -> &mut GraphicsState {
        // SAFETY: stack invariant guarantees len >= 1; .last_mut() cannot be None.
        self.stack
            .last_mut()
            .expect("StateStack invariant violated: stack is empty")
    }

    /// Push a clone of the current state (PDF `q` operator).
    ///
    /// # Panics
    ///
    /// Never panics in correct code — the stack invariant guarantees at least
    /// one entry at all times.  The `expect` is a development-time tripwire.
    pub fn save(&mut self) {
        // SAFETY: stack invariant guarantees len >= 1; .last() cannot be None.
        let cloned = self
            .stack
            .last()
            .expect("StateStack invariant violated: stack is empty")
            .save_clone();
        self.stack.push(cloned);
    }

    /// Pop the top state (PDF `Q` operator).
    ///
    /// Returns `true` on success.
    ///
    /// Returns `false` — **without modifying the stack** — when the stack is at
    /// depth 1 (only the initial state remains).  The initial state is never
    /// popped; this preserves the stack invariant (`depth ≥ 1`).
    ///
    /// Callers that receive `false` should treat it as an unmatched `Q` operator
    /// and continue rendering with the current state unchanged.
    pub fn restore(&mut self) -> bool {
        if self.stack.len() <= 1 {
            // Invariant: never pop the last (initial) state.
            return false;
        }
        drop(self.stack.pop());
        true
    }

    /// Current nesting depth (`1` = only the initial state, no saves in progress).
    #[must_use]
    pub const fn depth(&self) -> usize {
        self.stack.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_matrix_is_identity() {
        let s = GraphicsState::new(100, 100, false);
        // These values are set by assignment from a literal, so exact equality is correct.
        assert!(
            s.matrix
                .iter()
                .zip([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
                .all(|(a, b)| (a - b).abs() < f64::EPSILON)
        );
    }

    #[test]
    fn default_alpha_is_one() {
        let s = GraphicsState::new(100, 100, false);
        assert!((s.stroke_alpha - 1.0).abs() < f64::EPSILON);
        assert!((s.fill_alpha - 1.0).abs() < f64::EPSILON);
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
        assert!((stack.current().line_width - 1.0).abs() < f64::EPSILON); // restored to default
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
        let inv: [u8; 256] = std::array::from_fn(|i| u8::try_from(255 - i).expect("i < 256"));
        s.set_transfer(&inv, &inv, &inv, &inv);
        // CMYK C = 255 - R[255-i] (before overwrite of R); with identity R:
        // R[255-i] = 255-i, so C[i] = 255-(255-i) = i → identity
        for i in 0u8..=255 {
            assert_eq!(s.cmyk_transfer[0].apply(i), i, "cmyk C[{i}]");
        }
    }

    /// In debug builds, constructing a GraphicsState with zero width must panic.
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "width must be > 0")]
    fn new_panics_on_zero_width() {
        let _ = GraphicsState::new(0, 100, false);
    }

    /// In debug builds, constructing a GraphicsState with zero height must panic.
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "height must be > 0")]
    fn new_panics_on_zero_height() {
        let _ = GraphicsState::new(100, 0, false);
    }

    /// Verify that multiple unmatched restores never pop below depth 1.
    #[test]
    fn restore_never_pops_below_one() {
        let initial = GraphicsState::new(100, 100, false);
        let mut stack = StateStack::new(initial);
        stack.save();
        stack.save();
        assert!(stack.restore());
        assert!(stack.restore());
        // At bottom now — further restores must return false without changing depth.
        assert!(!stack.restore());
        assert!(!stack.restore());
        assert_eq!(stack.depth(), 1);
    }
}
