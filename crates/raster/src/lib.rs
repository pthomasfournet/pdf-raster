//! # raster
//!
//! Pure software rasterizer for PDF page content — no I/O, no PDF parsing,
//! no SIMD (Phase 1).
//!
//! ## Phase 1 scope
//! Foundation types only: pixel buffers, path geometry, edge tables, clip
//! regions, halftone screens, and graphics state. Rendering (pipe, fill,
//! stroke, image, shading, glyph, transparency) is Phase 2.
//!
//! ## Module layout
//! - [`types`] — raster-local enums and constants (re-exports [`color`] types)
//! - [`bitmap`] — [`Bitmap<P>`] and [`AaBuf`]
//! - [`path`] — [`Path`], [`PathBuilder`], Bezier flattening, stroke adjustment
//! - [`xpath`] — [`XPath`] edge table (flattened, matrix-transformed segments)
//! - [`scanner`] — [`XPathScanner`] and [`ScanIterator`] (scanline span emission)
//! - [`clip`] — [`Clip`] (rect + arbitrary path clip stack)
//! - [`screen`] — [`HalftoneScreen`] (Bayer / stochastic threshold matrix)
//! - [`state`] — [`GraphicsState`] and [`StateStack`]
//! - [`pipe`] — compositing pipeline (simple, AA, general; blend modes; Pattern trait)

pub mod bitmap;
pub mod clip;
pub mod fill;
pub mod glyph;
pub mod image;
pub mod path;
pub mod pipe;
pub mod scanner;
pub mod screen;
pub mod shading;
pub mod simd;
pub mod state;
pub mod stroke;
pub mod tiling;
pub mod transparency;
pub mod types;
pub mod xpath;

#[cfg(test)]
pub(crate) mod testutil;

#[cfg(feature = "rayon")]
pub use bitmap::BitmapBand;
pub use bitmap::{AaBuf, Bitmap};
pub use clip::{Clip, ClipResult};
#[cfg(feature = "rayon")]
pub use fill::{PARALLEL_FILL_MIN_HEIGHT, eo_fill_parallel, fill_parallel};
pub use fill::{eo_fill, fill};
pub use glyph::{GlyphBitmap, blit_glyph, fill_glyph};
pub use image::{ImageResult, ImageSource, MaskSource, draw_image, fill_image_mask};
pub use path::{Path, PathBuilder, PathFlags, PathPoint, StrokeAdjustHint};
pub use pipe::{Pattern, PipeSrc, PipeState};
pub use scanner::iter::ScanIterator;
pub use scanner::{Intersect, XPathScanner};
pub use screen::HalftoneScreen;
pub use shading::axial::AxialPattern;
pub use shading::function::FunctionPattern;
pub use shading::gouraud::{GouraudVertex, gouraud_triangle_fill};
pub use shading::radial::RadialPattern;
pub use shading::shaded_fill;
pub use simd::{
    blend_solid_gray8, blend_solid_rgb8, composite_aa_rgb8, composite_aa_rgb8_opaque,
    popcnt_aa_row, unpack_mono_row,
};
pub use state::{GraphicsState, StateStack, TransferSet};
pub use stroke::{
    StrokeParams, flatten_path, make_dashed_path, make_stroke_path, stroke, stroke_narrow,
    stroke_wide,
};
pub use tiling::TiledPattern;
pub use types::*;
pub use xpath::{XPath, XPathFlags, XPathSeg};
