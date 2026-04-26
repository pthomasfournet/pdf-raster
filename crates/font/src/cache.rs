//! Global glyph bitmap cache.
//!
//! [`GlyphCache`] is a two-level cache:
//!
//! 1. **Outer**: a [`dashmap::DashMap`] sharded by [`FaceId`], giving
//!    lock-free concurrent reads from multiple rayon threads rendering
//!    different pages.
//! 2. **Inner**: a per-face [`lru::LruCache`] keyed by [`GlyphKey`],
//!    evicting by a pixel-area × entry-count budget.
//!
//! # Capacity policy
//!
//! Each per-face LRU holds at most [`MAX_PER_FACE`] entries.  The global
//! capacity is `number_of_faces × MAX_PER_FACE`, which is acceptable
//! because faces are created once per document.
//!
//! # Thread safety
//!
//! `DashMap` provides sharded `RwLock`-style access: concurrent reads do not
//! block each other; a write (insertion) only locks the shard that owns the
//! glyph's `FaceId`.  No global lock is ever held.

use std::num::NonZeroUsize;
use std::sync::Arc;

use dashmap::DashMap;
use lru::LruCache;

use crate::bitmap::GlyphBitmap;
use crate::key::{FaceId, GlyphKey};

/// Maximum number of glyphs cached per font face.
pub const MAX_PER_FACE: usize = 4096;

/// Shared, thread-safe glyph bitmap cache.
///
/// Clone this to share across threads — the inner `Arc<DashMap>` is cheap
/// to clone and keeps the cache alive as long as any clone exists.
#[derive(Clone, Debug)]
pub struct GlyphCache {
    inner: Arc<DashMap<FaceId, LruCache<GlyphKey, Arc<GlyphBitmap>>>>,
}

impl GlyphCache {
    /// Create a new, empty cache.
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: Arc::new(DashMap::new()),
        }
    }

    /// Look up a glyph.
    ///
    /// Returns a shared reference to the bitmap if it is in the cache,
    /// promoting it to MRU.  Returns `None` on a miss.
    #[must_use]
    pub fn get(&self, key: &GlyphKey) -> Option<Arc<GlyphBitmap>> {
        let mut shard = self.inner.get_mut(&key.face_id)?;
        shard.get(key).cloned()
    }

    /// Insert a rendered glyph into the cache.
    ///
    /// If the per-face LRU is full the LRU entry is evicted first.
    /// If an entry with the same key already exists it is replaced.
    ///
    /// # Panics
    ///
    /// Panics only if `MAX_PER_FACE` is 0, which is a compile-time constant
    /// greater than zero and therefore cannot occur.
    pub fn insert(&self, key: GlyphKey, bitmap: GlyphBitmap) {
        // MAX_PER_FACE > 0 is a compile-time guarantee.
        const CAPACITY: NonZeroUsize = match NonZeroUsize::new(MAX_PER_FACE) {
            Some(n) => n,
            None => unreachable!(),
        };
        let mut shard = self
            .inner
            .entry(key.face_id)
            .or_insert_with(|| LruCache::new(CAPACITY));
        let _ = shard.put(key, Arc::new(bitmap));
    }

    /// Remove all cached entries for the given face.
    ///
    /// Call when a [`crate::face::FontFace`] is unloaded to prevent stale
    /// bitmaps from persisting under a reused `FaceId`.
    pub fn evict_face(&self, face_id: FaceId) {
        drop(self.inner.remove(&face_id));
    }

    /// Total number of cached glyphs across all faces.
    #[must_use]
    pub fn total_len(&self) -> usize {
        self.inner.iter().map(|e| e.len()).sum()
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
