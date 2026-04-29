//! Shared test utilities for the `raster` crate.
//!
//! This module is compiled only for tests (`#[cfg(test)]`).

use crate::clip::Clip;
use crate::path::{Path, PathBuilder};
use crate::pipe::PipeState;
use crate::state::TransferSet;
use crate::types::BlendMode;

/// Identity CTM `[a b c d e f] = [1 0 0 1 0 0]`.
pub(crate) fn identity_matrix() -> [f64; 6] {
    [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
}

/// Parametric pipe with configurable `a_input` and `blend_mode`.
///
/// All other fields are set to their neutral defaults (identity transfer,
/// no soft mask, no overprint, no knockout).
pub(crate) fn make_pipe(a_input: u8, blend_mode: BlendMode) -> PipeState<'static> {
    PipeState {
        blend_mode,
        a_input,
        overprint_mask: 0xFFFF_FFFF,
        overprint_additive: false,
        transfer: TransferSet::identity_rgb(),
        soft_mask: None,
        alpha0: None,
        knockout: false,
        knockout_opacity: 255,
        non_isolated_group: false,
    }
}

/// Opaque Normal-blend pipe — the simplest possible rendering state.
pub(crate) fn simple_pipe() -> PipeState<'static> {
    make_pipe(255, BlendMode::Normal)
}

/// Clip covering the entire `w × h` bitmap.
///
/// The upper bounds are inset by `0.001` to stay strictly inside the last pixel;
/// `Clip` uses exclusive-right semantics so a bound exactly at `w` would
/// exclude the rightmost column on some edge cases.
pub(crate) fn make_clip(w: u32, h: u32) -> Clip {
    Clip::new(0.0, 0.0, f64::from(w) - 0.001, f64::from(h) - 0.001, false)
}

pub(crate) fn rect_path(x0: f64, y0: f64, x1: f64, y1: f64) -> Path {
    let mut b = PathBuilder::new();
    b.move_to(x0, y0).expect("rect_path: move_to failed");
    b.line_to(x1, y0).expect("rect_path: line_to (top) failed");
    b.line_to(x1, y1)
        .expect("rect_path: line_to (right) failed");
    b.line_to(x0, y1)
        .expect("rect_path: line_to (bottom) failed");
    b.close(true).expect("rect_path: close failed");
    b.build()
}
