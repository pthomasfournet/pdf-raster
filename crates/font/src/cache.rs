//! Global glyph bitmap cache.
//!
//! [`GlyphCache`] wraps a [`quick_cache::sync::Cache`] keyed by [`GlyphKey`].
//!
//! # Concurrency
//!
//! `quick_cache` uses sharded locking internally.  Unlike the previous
//! `DashMap<FaceId, lru::LruCache>` design, reads do not acquire a write lock
//! — `lru::LruCache::get` requires `&mut self` to update MRU order, which
//! forced every hit to take a write lock on the shard.  `quick_cache` updates
//! recency atomically without exclusive shard access, so concurrent reads from
//! multiple Rayon threads no longer serialise within a shard.
//!
//! # Capacity policy
//!
//! The cache holds at most [`GLOBAL_CAPACITY`] entries total.  Eviction is
//! LRU.  Entries are `Arc<GlyphBitmap>`, so eviction drops the `Arc` without
//! freeing the bitmap data if the renderer still holds a reference.
//!
//! # TODO: redesign when a second glyph generator lands
//!
//! Today this cache is `FreeType`-only.  When a second source appears
//! (Type 3 `CharProcs`, `COLRv1`, SVG `OpenType`), don't add a sibling cache —
//! the [`GlyphKey`] shape and the entry-count budget below are both
//! Splash-era artifacts and will start to leak:
//!
//! - [`GlyphKey`] hardcodes `(face_id, glyph_id, size_px, x_frac, aa)` —
//!   a Type 3 glyph is `(font instance, char_code, CTM)` and has no
//!   `size_px` or `FreeType` `face_id`.
//! - [`GLOBAL_CAPACITY`] = `256 × 4_096` is a port of the C++ two-level
//!   budget; with a single global LRU it's just an entry count rounded for
//!   nostalgia.  A bytes-budget weighter would treat 1024×1024 Type 3
//!   glyphs and 8×8 ASCII glyphs honestly.
//!
//! The right shape is a `GlyphSpec` trait keyed by hash, an opaque
//! `RenderedGlyph` value type, and a `quick_cache` weighter sized in bytes.
//! Don't speculate — design under load when the second consumer is real.

use std::sync::Arc;

use quick_cache::sync::Cache;

use crate::bitmap::GlyphBitmap;
use crate::key::{FaceId, GlyphKey};

/// Maximum number of glyphs cached globally.
///
/// Matches the old two-level budget of 256 faces × 4 096 glyphs per face.
pub const GLOBAL_CAPACITY: usize = 256 * 4_096;

/// Shared, thread-safe glyph bitmap cache.
///
/// Clone to share across threads — the inner `Arc<Cache>` is cheap to clone
/// and keeps the cache alive as long as any clone exists.
#[derive(Clone, Debug)]
pub struct GlyphCache {
    inner: Arc<Cache<GlyphKey, Arc<GlyphBitmap>>>,
}

impl GlyphCache {
    /// Create a new, empty cache.
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Cache::new(GLOBAL_CAPACITY)),
        }
    }

    /// Look up a glyph, promoting it to MRU on a hit.
    ///
    /// Returns a shared reference to the bitmap, or `None` on a miss.
    #[must_use]
    pub fn get(&self, key: &GlyphKey) -> Option<Arc<GlyphBitmap>> {
        self.inner.get(key)
    }

    /// Insert a rendered glyph.  Evicts the LRU entry when the cache is full.
    pub fn insert(&self, key: GlyphKey, bitmap: GlyphBitmap) {
        self.inner.insert(key, Arc::new(bitmap));
    }

    /// Remove all cached entries for the given face.
    ///
    /// Call when a [`crate::face::FontFace`] is unloaded to release its
    /// cached bitmaps eagerly rather than waiting for LRU eviction.
    pub fn evict_face(&self, face_id: FaceId) {
        self.inner.retain(|k, _| k.face_id != face_id);
    }

    /// Total number of cached glyphs.
    #[must_use]
    pub fn total_len(&self) -> usize {
        self.inner.len()
    }
}

impl Default for GlyphCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::key::{FaceId, GlyphKey};

    fn bmp(tag: u8) -> GlyphBitmap {
        GlyphBitmap {
            x_off: 0,
            y_off: 0,
            width: 1,
            height: 1,
            aa: true,
            data: vec![tag],
        }
    }

    fn key(face: u32, glyph: u32) -> GlyphKey {
        GlyphKey::new(FaceId(face), glyph, 12, 0, true)
    }

    #[test]
    fn miss_on_empty() {
        let cache = GlyphCache::new();
        assert!(cache.get(&key(0, 0)).is_none());
    }

    #[test]
    fn insert_then_get() {
        let cache = GlyphCache::new();
        cache.insert(key(0, 1), bmp(0xAA));
        let got = cache.get(&key(0, 1)).expect("should be present");
        assert_eq!(got.data[0], 0xAA);
    }

    #[test]
    fn different_face_ids_are_independent() {
        let cache = GlyphCache::new();
        cache.insert(key(0, 1), bmp(10));
        cache.insert(key(1, 1), bmp(20));
        assert_eq!(cache.get(&key(0, 1)).unwrap().data[0], 10);
        assert_eq!(cache.get(&key(1, 1)).unwrap().data[0], 20);
    }

    #[test]
    fn evict_face_removes_entries() {
        let cache = GlyphCache::new();
        cache.insert(key(5, 1), bmp(1));
        cache.insert(key(5, 2), bmp(2));
        cache.insert(key(6, 1), bmp(3));

        cache.evict_face(FaceId(5));

        assert!(
            cache.get(&key(5, 1)).is_none(),
            "evicted face should be gone"
        );
        assert!(
            cache.get(&key(5, 2)).is_none(),
            "evicted face should be gone"
        );
        assert!(cache.get(&key(6, 1)).is_some(), "other face must survive");
    }

    #[test]
    fn total_len_counts_all_faces() {
        let cache = GlyphCache::new();
        cache.insert(key(0, 1), bmp(1));
        cache.insert(key(0, 2), bmp(2));
        cache.insert(key(1, 1), bmp(3));
        assert_eq!(cache.total_len(), 3);
    }

    #[test]
    fn shared_clone_shares_state() {
        let a = GlyphCache::new();
        let b = a.clone();
        a.insert(key(0, 1), bmp(42));
        assert!(
            b.get(&key(0, 1)).is_some(),
            "clone must share the same backing store"
        );
    }

    #[test]
    fn insert_replaces_existing() {
        let cache = GlyphCache::new();
        cache.insert(key(0, 1), bmp(1));
        cache.insert(key(0, 1), bmp(2));
        let got = cache.get(&key(0, 1)).unwrap();
        assert_eq!(got.data[0], 2, "second insert must overwrite first");
    }
}
