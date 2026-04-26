//! Text-rendering helpers extracted from `page/mod.rs`.
//!
//! These types and functions are used exclusively by [`super::PageRenderer::show_text`].

use crate::renderer::gstate::ctm_transform;

// ── Glyph record ──────────────────────────────────────────────────────────────

/// Holds rasterized glyph data for two-phase text rendering.
pub(super) struct GlyphRecord {
    pub(super) pen_x: i32,
    pub(super) pen_y: i32,
    pub(super) x_off: i32,
    pub(super) y_off: i32,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) aa: bool,
    pub(super) data: Vec<u8>,
}

// ── Coordinate helpers ────────────────────────────────────────────────────────

/// Map a text-space point through the text matrix and CTM to device-pixel
/// coordinates with y-flip (PDF origin = bottom-left; device origin = top-left).
///
/// `tm[6]` is `[a, b, c, d, e, f]` in PDF column-major form.
/// The full mapping is: `device = CTM × Tm × (tx, ty+rise)`.
#[expect(clippy::many_single_char_names, reason = "PDF matrix components")]
pub(super) fn text_to_device(
    ctm: &[f64; 6],
    tm: &[f64; 6],
    tx: f64,
    ty: f64,
    page_h: u32,
) -> (i32, i32) {
    // Apply text matrix: user_space = Tm * (tx, ty).
    let [a, b, c, d, e, f] = *tm;
    let ux = a.mul_add(tx, c * ty) + e;
    let uy = b.mul_add(tx, d * ty) + f;

    // Apply CTM: device = CTM * (ux, uy).
    let (dx, dy) = ctm_transform(ctm, ux, uy);

    // y-flip.
    let dy_flipped = f64::from(page_h) - dy;

    #[expect(
        clippy::cast_possible_truncation,
        reason = "device pixels are always in [0, page_h|w]; round() output fits i32 for any real page"
    )]
    (dx.round() as i32, dy_flipped.round() as i32)
}
