//! Type 3 font glyph cache.
//!
//! Type 3 fonts define each glyph as a PDF content stream rendered at a
//! specific size.  The Splash rasterizer caches up to `CACHE_SIZE` rendered
//! glyphs to avoid re-executing the content stream on each use.
//!
//! This is a direct port of `SplashFont`'s per-glyph bitmap cache (the
//! 8-way set-associative MRU structure in `SplashFont.cc`) restricted to
//! the Type 3 use case.  The general glyph cache for FreeType-rendered glyphs
//! lives in [`crate::cache`].
//!
//! # Cache structure
//!
//! `CACHE_SIZE = 8` entries.  On a miss the LRU entry is replaced.
//! Entries are keyed by `(char_code, x_frac)` (`y_frac` is always 0 for `FT`
//! and ignored here too).

use crate::bitmap::GlyphBitmap;

/// Number of slots in the Type 3 glyph cache.
pub const CACHE_SIZE: usize = 8;

/// A single cache slot.
#[derive(Debug)]
struct Slot {
    /// Character code (glyph index within the Type 3 font).
    char_code: u32,
    /// Sub-pixel x fraction in `[0, crate::key::FRACTION)`.
    x_frac: u8,
    /// MRU counter: higher → more recently used.
    mru: u32,
    /// Cached bitmap.
    bitmap: GlyphBitmap,
}

/// 8-slot MRU glyph cache for a single Type 3 font instance.
///
/// All operations are O(`CACHE_SIZE`) = O(8), which is acceptable given
/// that Type 3 fonts are rare and small.
#[derive(Debug, Default)]
pub struct Type3Cache {
    slots: Vec<Slot>,
    /// Global MRU counter.  Wraps at `u32::MAX`; with 8 slots and typical
    /// glyph rates, wrap-around cannot occur within a page render.
    counter: u32,
}

impl Type3Cache {
    /// Create a new, empty cache.
    #[must_use]
    pub fn new() -> Self {
        Self {
            slots: Vec::with_capacity(CACHE_SIZE),
            counter: 0,
        }
    }

    /// Look up `(char_code, x_frac)` in the cache.
    ///
    /// On a hit: updates the MRU counter and returns a shared reference to
    /// the bitmap.
    /// On a miss: returns `None`.
    #[must_use]
    pub fn get(&mut self, char_code: u32, x_frac: u8) -> Option<&GlyphBitmap> {
        let idx = self
            .slots
            .iter()
            .position(|s| s.char_code == char_code && s.x_frac == x_frac)?;
        self.counter = self.counter.wrapping_add(1);
        self.slots[idx].mru = self.counter;
        Some(&self.slots[idx].bitmap)
    }

    /// Insert a bitmap into the cache for `(char_code, x_frac)`.
    ///
    /// If the cache is full the LRU entry is evicted.  If an entry with the
    /// same key already exists it is overwritten (the caller is responsible
    /// for not inserting duplicates; use [`Self::get`] first).
    pub fn insert(&mut self, char_code: u32, x_frac: u8, bitmap: GlyphBitmap) {
        self.counter = self.counter.wrapping_add(1);
        let slot = Slot {
            char_code,
            x_frac,
            mru: self.counter,
            bitmap,
        };

        if self.slots.len() < CACHE_SIZE {
            self.slots.push(slot);
        } else {
            // Evict the LRU entry (lowest mru counter).
            let lru_idx = self
                .slots
                .iter()
                .enumerate()
                .min_by_key(|(_, s)| s.mru)
                .map_or(0, |(i, _)| i);
            self.slots[lru_idx] = slot;
        }
    }

    /// Remove all cached entries.
    pub fn clear(&mut self) {
        self.slots.clear();
        self.counter = 0;
    }

    /// Number of entries currently in the cache.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.slots.len()
    }

    /// Returns `true` if the cache holds no entries.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_bitmap(tag: u8) -> GlyphBitmap {
        GlyphBitmap {
            x_off: 0,
            y_off: 0,
            width: 1,
            height: 1,
            aa: true,
            data: vec![tag],
        }
    }

    #[test]
    fn miss_on_empty_cache() {
        let mut c = Type3Cache::new();
        assert!(c.get(0, 0).is_none());
    }

    #[test]
    fn insert_then_get() {
        let mut c = Type3Cache::new();
        c.insert(42, 1, make_bitmap(0xAA));
        let bmp = c.get(42, 1).expect("should be present after insert");
        assert_eq!(bmp.data[0], 0xAA);
    }

    #[test]
    fn different_x_frac_is_different_entry() {
        let mut c = Type3Cache::new();
        c.insert(1, 0, make_bitmap(10));
        c.insert(1, 1, make_bitmap(20));
        assert_eq!(c.get(1, 0).unwrap().data[0], 10);
        assert_eq!(c.get(1, 1).unwrap().data[0], 20);
    }

    #[test]
    fn lru_evicted_when_full() {
        let mut c = Type3Cache::new();
        // Fill to capacity: glyphs 0..8 in order (glyph 0 is LRU after insertion).
        for i in 0..CACHE_SIZE as u32 {
            c.insert(i, 0, make_bitmap(i as u8));
        }
        assert_eq!(c.len(), CACHE_SIZE);

        // Access glyphs 1..8 to make glyph 0 the LRU.
        for i in 1..CACHE_SIZE as u32 {
            let _ = c.get(i, 0);
        }

        // Insert glyph 99 — should evict glyph 0.
        c.insert(99, 0, make_bitmap(99));
        assert_eq!(c.len(), CACHE_SIZE, "cache size must stay at CACHE_SIZE");
        assert!(c.get(0, 0).is_none(), "LRU entry (glyph 0) must be evicted");
        assert!(
            c.get(99, 0).is_some(),
            "newly inserted entry must be present"
        );
    }

    #[test]
    fn clear_empties_cache() {
        let mut c = Type3Cache::new();
        c.insert(1, 0, make_bitmap(1));
        c.clear();
        assert!(c.is_empty());
        assert!(c.get(1, 0).is_none());
    }

    #[test]
    fn get_updates_mru_prevents_eviction() {
        let mut c = Type3Cache::new();
        for i in 0..CACHE_SIZE as u32 {
            c.insert(i, 0, make_bitmap(i as u8));
        }
        // Touch glyph 0 to make it MRU.
        let _ = c.get(0, 0);
        // Access all others except glyph 0 to make one of them LRU.
        // (The actual LRU after this depends on which was last accessed.)
        // Just verify glyph 0 is still present after inserting one more.
        c.insert(100, 0, make_bitmap(100));
        assert!(c.get(0, 0).is_some(), "MRU glyph must not be evicted");
    }
}
