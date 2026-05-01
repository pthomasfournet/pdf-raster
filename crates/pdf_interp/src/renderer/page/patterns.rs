//! Tiling pattern resolution and tile rendering.

use super::super::gstate::{ctm_multiply, ctm_transform};
use super::{MAX_TILE_PX, PageRenderer};
use crate::resources::PageResources;
use color::Rgb8;
use raster::{Bitmap, TiledPattern};

impl PageRenderer<'_> {
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_possible_wrap,
        reason = "tile dimensions are capped at 4096 and clamped to positive; phase coords are page-bounded; safe casts"
    )]
    pub(super) fn resolve_fill_pattern(&self, name: &[u8]) -> Option<TiledPattern> {
        let desc = self.resources.tiling_pattern(name)?;

        // Compose pattern matrix with current CTM to get pattern-space → device-space.
        let ctm = self.gstate.current().ctm;
        let pat_ctm = ctm_multiply(&desc.matrix, &ctm);

        // Tile size in device pixels — scale the XStep/YStep by the linear scale
        // of the combined transform.  We use the L2 norm of each column vector to
        // account for rotations.
        let sx = (pat_ctm[0].hypot(pat_ctm[1])).abs();
        let sy = (pat_ctm[2].hypot(pat_ctm[3])).abs();

        if !sx.is_finite() || !sy.is_finite() {
            log::warn!(
                "pdf_interp: Pattern /{} CTM scale is non-finite ({sx}, {sy}) — skipping",
                String::from_utf8_lossy(name)
            );
            return None;
        }

        if sx < 0.5 || sy < 0.5 {
            log::debug!(
                "pdf_interp: Pattern /{} tile scale ({sx:.2}, {sy:.2}) too small — skipping",
                String::from_utf8_lossy(name)
            );
            return None;
        }

        // Cap the tile to MAX_TILE_PX to avoid huge allocations from malformed PDFs.
        let tile_w = (desc.x_step.abs() * sx).min(MAX_TILE_PX).ceil() as u32;
        let tile_h = (desc.y_step.abs() * sy).min(MAX_TILE_PX).ceil() as u32;

        if tile_w == 0 || tile_h == 0 {
            log::debug!(
                "pdf_interp: Pattern /{} produced zero-size tile ({tile_w}×{tile_h}) — skipping",
                String::from_utf8_lossy(name)
            );
            return None;
        }

        // Build the CTM for the tile renderer: pattern space → tile device space.
        // Translation is handled by phase below, so we zero it here.
        let tile_ctm = [pat_ctm[0], pat_ctm[1], pat_ctm[2], pat_ctm[3], 0.0, 0.0];

        // Rasterise the tile.
        let tile_bitmap = self.render_pattern_tile(&desc, tile_w, tile_h, &tile_ctm);

        // Phase = where the pattern origin lands in device space (after y-flip).
        let page_h = f64::from(self.height);
        let (ox, oy_pdf) = ctm_transform(&pat_ctm, 0.0, 0.0);
        let oy = page_h - oy_pdf;

        if !ox.is_finite() || !oy.is_finite() {
            log::warn!(
                "pdf_interp: Pattern /{} origin is non-finite ({ox}, {oy}) — skipping",
                String::from_utf8_lossy(name)
            );
            return None;
        }

        let phase_x = ox.round() as i32;
        let phase_y = oy.round() as i32;

        let pixels = tile_bitmap.data().to_vec();
        Some(TiledPattern::new(
            pixels,
            tile_w as i32,
            tile_h as i32,
            phase_x,
            phase_y,
        ))
    }

    /// Render a pattern content stream into a tile bitmap.
    ///
    /// Creates a child `PageRenderer` scoped to the pattern's resource context and
    /// executes the pattern's content stream.  The tile bitmap is `tile_w ×
    /// tile_h` pixels with a white background.
    pub(super) fn render_pattern_tile(
        &self,
        desc: &crate::resources::tiling::TilingDescriptor,
        tile_w: u32,
        tile_h: u32,
        tile_ctm: &[f64; 6],
    ) -> Bitmap<Rgb8> {
        // Build a temporary child renderer sharing the same document / font engine.
        let doc = self.resources.doc();
        let mut tile_renderer = match PageRenderer::new(tile_w, tile_h, doc, desc.stream_id) {
            Ok(r) => r,
            Err(e) => {
                log::warn!("render_pattern_tile: {e}");
                return Bitmap::new(tile_w, tile_h, 1, false);
            }
        };

        // Override the CTM to map pattern space → tile device space.
        tile_renderer.gstate.current_mut().ctm = *tile_ctm;

        // If the pattern has no own resources, point it at the parent context.
        if !desc.has_own_resources {
            tile_renderer.resources = PageResources::new(doc, self.resources.resource_context_id());
        }

        let ops = crate::content::parse(&desc.content);
        tile_renderer.execute(&ops);
        tile_renderer.finish().0
    }
}
