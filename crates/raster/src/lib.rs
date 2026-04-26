//! # raster
//!
//! Pure software rasterizer for PDF page content — no I/O, no PDF parsing,
//! no SIMD (Phase 1). This crate is the direct Rust replacement for poppler's
//! Splash backend (`splash/Splash.cc` et al.).
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
pub mod path;
pub mod pipe;
pub mod scanner;
pub mod screen;
pub mod state;
pub mod stroke;
pub mod types;
pub mod xpath;

pub use bitmap::{AaBuf, Bitmap};
pub use fill::{eo_fill, fill};
pub use glyph::{GlyphBitmap, blit_glyph, fill_glyph};
pub use clip::{Clip, ClipResult};
pub use path::{Path, PathBuilder, PathFlags, PathPoint, StrokeAdjustHint};
pub use pipe::{Pattern, PipeState, PipeSrc};
pub use scanner::iter::ScanIterator;
pub use scanner::{Intersect, XPathScanner};
pub use screen::HalftoneScreen;
pub use state::{GraphicsState, StateStack, TransferSet};
pub use stroke::{StrokeParams, flatten_path, make_dashed_path, make_stroke_path, stroke, stroke_narrow, stroke_wide};
pub use types::*;
pub use xpath::{XPath, XPathFlags, XPathSeg};
