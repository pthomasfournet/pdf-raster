//! Shared test utilities for the `raster` crate.
//!
//! This module is compiled only for tests (`#[cfg(test)]`).

use crate::clip::Clip;
use crate::path::{Path, PathBuilder};
use crate::pipe::PipeState;
use crate::state::TransferSet;
use crate::types::BlendMode;

pub(crate) fn identity_matrix() -> [f64; 6] {
    [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
}

pub(crate) fn simple_pipe() -> PipeState<'static> {
    PipeState {
        blend_mode: BlendMode::Normal,
        a_input: 255,
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

pub(crate) fn make_clip(w: u32, h: u32) -> Clip {
    Clip::new(0.0, 0.0, f64::from(w) - 0.001, f64::from(h) - 0.001, false)
}

pub(crate) fn rect_path(x0: f64, y0: f64, x1: f64, y1: f64) -> Path {
    let mut b = PathBuilder::new();
    b.move_to(x0, y0).unwrap();
    b.line_to(x1, y0).unwrap();
    b.line_to(x1, y1).unwrap();
    b.line_to(x0, y1).unwrap();
    b.close(true).unwrap();
    b.build()
}
