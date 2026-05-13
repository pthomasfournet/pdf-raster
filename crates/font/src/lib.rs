//! `FreeType` bridge and glyph cache for the `rasterrocket` pipeline.
//!
//! # Structure
//!
//! | Module | Contents |
//! |--------|---------|
//! | [`key`] | [`key::FaceId`] and [`key::GlyphKey`] — cache key types |
//! | [`bitmap`] | [`bitmap::GlyphBitmap`] — rasterized glyph (owned data) |
//! | [`hinting`] | [`hinting::load_flags`] — `FreeType` load-flag policy |
//! | [`outline`] | [`outline::decompose_outline`] — FT outline → `raster::Path` |
//! | [`cache`] | [`cache::GlyphCache`] — global concurrent glyph cache |
//! | [`face`] | [`face::FontFace`] — scaled font face, renders glyphs |
//! | [`engine`] | [`engine::FontEngine`] — owns FT library, loads faces |
//!
//! # Usage
//!
//! ```no_run
//! use font::engine::{FontEngine, FaceParams};
//! use font::hinting::FontKind;
//! use font::cache::GlyphCache;
//! use font::key::GlyphKey;
//!
//! // One engine per process.
//! let engine = FontEngine::init(true, true, false).expect("FreeType init failed");
//!
//! // Shared across threads.
//! let cache = GlyphCache::new();
//!
//! // Load a face (called from the PDF rendering pipeline).
//! let identity = [1.0, 0.0, 0.0, 1.0];
//! let params = FaceParams {
//!     kind: FontKind::TrueType,
//!     code_to_gid: vec![],
//!     mat: identity,
//!     text_mat: identity,
//! };
//! let face = {
//!     let mut eng = engine.lock().unwrap();
//!     eng.load_file_face("/path/to/font.ttf", 0, params)
//!         .expect("face load failed")
//! };
//!
//! // Look up or render a glyph.
//! let key = GlyphKey::new(face.id, 65, face.size_px, 0, face.aa);
//! let bitmap = cache.get(&key).unwrap_or_else(|| {
//!     let bmp = face.make_glyph(65, 0).expect("glyph render failed");
//!     cache.insert(key, bmp);
//!     cache.get(&key).unwrap()
//! });
//! let _ = bitmap;
//! ```

pub mod bitmap;
pub mod cache;
pub mod engine;
pub mod face;
pub mod hinting;
pub mod key;
pub mod outline;
