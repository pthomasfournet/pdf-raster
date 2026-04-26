//! Clip region: an axis-aligned rectangle intersected with zero or more
//! arbitrary path clip regions.
//!
//! Mirrors `SplashClip` from `splash/SplashClip.h/.cc`.
//!
//! ## Sharing semantics
//!
//! When a [`Clip`] is cloned (e.g. for `GraphicsState::save`), the path-clip
//! scanners are shared via [`Arc`] — matching the C++ `shared_ptr` behaviour.
//! [`XPathScanner`] instances are immutable after construction, so sharing
//! across threads and across `clone_shared` copies is safe: there is no
//! interior mutability in the shared objects.

use std::sync::Arc;

use crate::bitmap::AaBuf;
use crate::scanner::XPathScanner;
use crate::types::{AA_SIZE, splash_ceil, splash_floor};
use crate::xpath::XPath;

// ── ClipResult ────────────────────────────────────────────────────────────────

/// Result of a rectangular or span clip test. Matches `SplashClipResult`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ClipResult {
    /// The entire tested region lies within the clip boundary.
    AllInside,
    /// The entire tested region lies outside the clip boundary.
    AllOutside,
    /// The tested region straddles the clip boundary; per-pixel testing is required.
    Partial,
}

// ── Clip ──────────────────────────────────────────────────────────────────────

/// A clipping region combining an axis-aligned rectangle with an optional
/// stack of arbitrary path clips.
///
/// The effective clip is the intersection of the rectangle and all path clips.
pub struct Clip {
    /// Whether anti-aliasing (supersampling) is enabled; scales coordinates by
    /// [`AA_SIZE`] when testing path clips.
    pub antialias: bool,
    /// Left edge of the clip rectangle in floating-point device space (inclusive).
    pub x_min: f64,
    /// Top edge of the clip rectangle in floating-point device space (inclusive).
    pub y_min: f64,
    /// Right edge of the clip rectangle in floating-point device space (exclusive).
    pub x_max: f64,
    /// Bottom edge of the clip rectangle in floating-point device space (exclusive).
    pub y_max: f64,
    /// Integer pixel column of the left clip edge: `floor(x_min)`.
    pub x_min_i: i32,
    /// Integer pixel row of the top clip edge: `floor(y_min)`.
    pub y_min_i: i32,
    /// Integer pixel column of the right clip edge: `ceil(x_max) - 1`.
    pub x_max_i: i32,
    /// Integer pixel row of the bottom clip edge: `ceil(y_max) - 1`.
    pub y_max_i: i32,
    /// Arbitrary path-clip scanners.
    ///
    /// Shared across [`clone_shared`](Clip::clone_shared) copies via [`Arc`].
    /// [`XPathScanner`] is immutable after construction so no interior-mutability
    /// hazard exists.
    scanners: Vec<Arc<XPathScanner>>,
}

impl Clip {
    /// Create a new clip region from a rectangle.
    ///
    /// Matches `SplashClip(x0, y0, x1, y1, antialiasA)` in `SplashClip.cc`.
    #[must_use]
    pub fn new(x0: f64, y0: f64, x1: f64, y1: f64, antialias: bool) -> Self {
        let mut clip = Self {
            antialias,
            x_min: 0.0,
            y_min: 0.0,
            x_max: 0.0,
            y_max: 0.0,
            x_min_i: 0,
            y_min_i: 0,
            x_max_i: 0,
            y_max_i: 0,
            scanners: Vec::new(),
        };
        clip.set_rect(x0, y0, x1, y1);
        clip
    }

    /// Clone this `Clip`, sharing all path-clip scanners via [`Arc`].
    ///
    /// The cloned value and the original share the same [`XPathScanner`]
    /// instances. Because scanners are immutable after construction there is
    /// no interior-mutability hazard. This mirrors C++ `shared_ptr` copy
    /// semantics used in `GraphicsState::save`.
    #[must_use]
    pub fn clone_shared(&self) -> Self {
        Self {
            antialias: self.antialias,
            x_min: self.x_min,
            y_min: self.y_min,
            x_max: self.x_max,
            y_max: self.y_max,
            x_min_i: self.x_min_i,
            y_min_i: self.y_min_i,
            x_max_i: self.x_max_i,
            y_max_i: self.y_max_i,
            scanners: self.scanners.clone(), // Arc::clone per element
        }
    }

    /// Replace the clip rectangle and clear all path clips.
    pub fn reset_to_rect(&mut self, x0: f64, y0: f64, x1: f64, y1: f64) {
        self.set_rect(x0, y0, x1, y1);
        self.scanners.clear();
    }

    /// Intersect the clip rectangle with `[x0, y0, x1, y1]`.
    pub fn clip_to_rect(&mut self, x0: f64, y0: f64, x1: f64, y1: f64) {
        let (lx, rx) = (x0.min(x1), x0.max(x1));
        let (ly, ry) = (y0.min(y1), y0.max(y1));
        self.x_min = self.x_min.max(lx);
        self.x_max = self.x_max.min(rx);
        self.y_min = self.y_min.max(ly);
        self.y_max = self.y_max.min(ry);
        self.recompute_int_bounds();
    }

    /// Intersect with an arbitrary path clip.
    ///
    /// If the path resolves to a simple axis-aligned rectangle (4 segments,
    /// axis-aligned), it is reduced to `clip_to_rect`. Otherwise a new
    /// [`XPathScanner`] is pushed onto the scanner stack.
    ///
    /// An empty path forces the clip to be empty (nothing passes through).
    ///
    /// # Panics
    ///
    /// Panics in debug builds if the AA y-range arithmetic overflows `i32`.
    /// In practice `y_max_i` is bounded by the bitmap height (≪ `i32::MAX / AA_SIZE`).
    pub fn clip_to_path(&mut self, xpath: &XPath, eo: bool) {
        if xpath.segs.is_empty() {
            // Force empty: nothing passes.
            self.x_max = self.x_min - 1.0;
            self.y_max = self.y_min - 1.0;
            self.recompute_int_bounds();
            return;
        }
        // Detect axis-aligned rect (4 segments, 2 horiz + 2 vert, forming a closed box).
        if let Some((rx0, ry0, rx1, ry1)) = detect_rect(xpath) {
            self.clip_to_rect(rx0, ry0, rx1, ry1);
            return;
        }
        // General path clip: compute scanline range in (possibly scaled) space.
        let (y_lo, y_hi) = if self.antialias {
            // Invariant: y_max_i is a pixel coordinate bounded by bitmap height,
            // which is far below i32::MAX / AA_SIZE. The additions below cannot
            // realistically overflow, but we assert in debug builds.
            let lo = self
                .y_min_i
                .checked_mul(AA_SIZE)
                .expect("AA y_lo overflows i32: y_min_i is unreasonably large");
            let hi = self
                .y_max_i
                .checked_add(1)
                .and_then(|v| v.checked_mul(AA_SIZE))
                .map(|v| v - 1)
                .expect("AA y_hi overflows i32: y_max_i is unreasonably large");
            (lo, hi)
        } else {
            (self.y_min_i, self.y_max_i)
        };
        let scanner = XPathScanner::new(xpath, eo, y_lo, y_hi);
        self.scanners.push(Arc::new(scanner));
    }

    // ── Pixel-level tests ─────────────────────────────────────────────────────

    /// Test whether pixel `(x, y)` is inside the clip region.
    ///
    /// Returns `false` immediately if `(x, y)` is outside the axis-aligned
    /// rectangle; otherwise all path-clip scanners are consulted.
    #[inline]
    #[must_use]
    pub fn test(&self, x: i32, y: i32) -> bool {
        if x < self.x_min_i || x > self.x_max_i || y < self.y_min_i || y > self.y_max_i {
            return false;
        }
        self.test_clip_paths(x, y)
    }

    /// Test a pixel rectangle against the clip region.
    ///
    /// The rectangle is inclusive on both ends: `[left, right] × [top, bottom]`.
    #[must_use]
    pub fn test_rect(&self, left: i32, top: i32, right: i32, bottom: i32) -> ClipResult {
        // Half-open pixel rect: [left, right+1) × [top, bottom+1).
        // Clip rect: [x_min, x_max) × [y_min, y_max).
        if f64::from(right + 1) <= self.x_min
            || f64::from(left) >= self.x_max
            || f64::from(bottom + 1) <= self.y_min
            || f64::from(top) >= self.y_max
        {
            return ClipResult::AllOutside;
        }
        if f64::from(left) >= self.x_min
            && f64::from(right + 1) <= self.x_max
            && f64::from(top) >= self.y_min
            && f64::from(bottom + 1) <= self.y_max
            && self.scanners.is_empty()
        {
            return ClipResult::AllInside;
        }
        ClipResult::Partial
    }

    /// Test whether the span `[x0, x1]` on scanline `y` is fully inside the clip.
    ///
    /// Returns [`ClipResult::AllInside`] only when the span is inside both the
    /// bounding rectangle and every path-clip scanner. Returns
    /// [`ClipResult::AllOutside`] when the span is fully outside the rectangle.
    /// Otherwise returns [`ClipResult::Partial`].
    #[must_use]
    pub fn test_span(&self, x0: i32, x1: i32, y: i32) -> ClipResult {
        let result = self.test_rect(x0, y, x1, y);
        if result != ClipResult::AllInside {
            return result;
        }
        for scanner in &self.scanners {
            let (sx0, sx1, sy) = aa_coords(x0, x1, y, self.antialias);
            if !scanner.test_span(sx0, sx1, sy) {
                return ClipResult::Partial;
            }
        }
        ClipResult::AllInside
    }

    /// Clip an AA buffer row, zeroing bits outside the clip region.
    ///
    /// Matches `SplashClip::clipAALine` in `SplashClip.cc`. Each path-clip
    /// scanner is asked to render its coverage into `aa_buf`, and the output
    /// span `[*x0, *x1]` is clamped to the integer clip bounds.
    ///
    /// # Panics
    ///
    /// Panics if `AA_SIZE` cannot be represented as `usize` — `AA_SIZE` is the
    /// compile-time constant `4`, so this never occurs in practice.
    pub fn clip_aa_line(&self, aa_buf: &mut AaBuf, x0: &mut i32, x1: &mut i32, y: i32) {
        // AA_SIZE = 4 (positive compile-time constant); the expect never fires.
        let _aa = usize::try_from(AA_SIZE)
            .expect("AA_SIZE is a positive compile-time constant (4) and always fits in usize");

        // Apply path-clip scanners.
        for scanner in &self.scanners {
            scanner.render_aa_line(aa_buf, x0, x1, y);
        }
        // Clamp output range to the integer clip bounds.
        *x0 = (*x0).max(self.x_min_i);
        *x1 = (*x1).min(self.x_max_i);
    }

    // ── Private ───────────────────────────────────────────────────────────────

    fn set_rect(&mut self, x0: f64, y0: f64, x1: f64, y1: f64) {
        self.x_min = x0.min(x1);
        self.x_max = x0.max(x1);
        self.y_min = y0.min(y1);
        self.y_max = y0.max(y1);
        self.recompute_int_bounds();
    }

    fn recompute_int_bounds(&mut self) {
        self.x_min_i = splash_floor(self.x_min);
        self.y_min_i = splash_floor(self.y_min);
        self.x_max_i = splash_ceil(self.x_max) - 1;
        self.y_max_i = splash_ceil(self.y_max) - 1;
    }

    fn test_clip_paths(&self, x: i32, y: i32) -> bool {
        let (tx, ty, _) = aa_coords(x, x, y, self.antialias);
        self.scanners.iter().all(|s| s.test(tx, ty))
    }
}

// ── AA coordinate scaling ─────────────────────────────────────────────────────

/// Scale pixel coordinates to the supersampled AA grid when `antialias` is set.
///
/// Returns `(sx0, sx1, sy)` where:
/// - `sx0 = x0 * AA_SIZE` if AA, else `x0`
/// - `sx1 = x1 * AA_SIZE + (AA_SIZE - 1)` if AA, else `x1`
/// - `sy  = y  * AA_SIZE` if AA, else `y`
///
/// The expanded `sx1` covers all supersampled sub-pixels within device pixel `x1`.
///
/// # Panics
///
/// Panics in debug builds on overflow; in practice pixel coordinates are bounded
/// by the bitmap dimensions, which are far below `i32::MAX / AA_SIZE`.
#[inline]
fn aa_coords(x0: i32, x1: i32, y: i32, antialias: bool) -> (i32, i32, i32) {
    if antialias {
        let sx0 = x0
            .checked_mul(AA_SIZE)
            .expect("aa_coords: x0 * AA_SIZE overflows i32");
        let sx1 = x1
            .checked_mul(AA_SIZE)
            .and_then(|v| v.checked_add(AA_SIZE - 1))
            .expect("aa_coords: x1 * AA_SIZE + (AA_SIZE-1) overflows i32");
        let sy = y
            .checked_mul(AA_SIZE)
            .expect("aa_coords: y * AA_SIZE overflows i32");
        (sx0, sx1, sy)
    } else {
        (x0, x1, y)
    }
}

// ── Rectangle detection ───────────────────────────────────────────────────────

/// Detect whether an `XPath` is an axis-aligned rectangle.
///
/// Returns `Some((x0, y0, x1, y1))` giving the bounding box of the rectangle
/// if the path consists of exactly 4 axis-aligned segments (2 vertical + 2
/// horizontal). Returns `None` for any other path.
///
/// Matches the `SplashClip::isRect` logic in `SplashClip.cc`.
fn detect_rect(xpath: &XPath) -> Option<(f64, f64, f64, f64)> {
    use crate::xpath::XPathFlags;
    if xpath.segs.len() != 4 {
        return None;
    }
    let segs = &xpath.segs;
    // Need exactly 2 vertical + 2 horizontal segments.
    let verts = segs.iter().filter(|s| s.flags.contains(XPathFlags::VERT)).count();
    let horizs = segs.iter().filter(|s| s.flags.contains(XPathFlags::HORIZ)).count();
    if verts != 2 || horizs != 2 {
        return None;
    }
    // Extract x extents from vertical segments and y extents from horizontal segments.
    let vert_xs = segs
        .iter()
        .filter(|s| s.flags.contains(XPathFlags::VERT))
        .flat_map(|s| [s.x0, s.x1]);
    let horiz_ys = segs
        .iter()
        .filter(|s| s.flags.contains(XPathFlags::HORIZ))
        .flat_map(|s| [s.y0, s.y1]);

    let x0 = vert_xs.clone().fold(f64::INFINITY, f64::min);
    let x1 = vert_xs.fold(f64::NEG_INFINITY, f64::max);
    let y0 = horiz_ys.clone().fold(f64::INFINITY, f64::min);
    let y1 = horiz_ys.fold(f64::NEG_INFINITY, f64::max);

    Some((x0, y0, x1, y1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_clip_rect_bounds() {
        let c = Clip::new(1.5, 2.5, 10.5, 8.5, false);
        assert_eq!(c.x_min_i, 1); // floor(1.5) = 1
        assert_eq!(c.y_min_i, 2); // floor(2.5) = 2
        assert_eq!(c.x_max_i, 10); // ceil(10.5) - 1 = 11 - 1 = 10
        assert_eq!(c.y_max_i, 8); // ceil(8.5) - 1 = 9 - 1 = 8
    }

    #[test]
    fn test_inside() {
        let c = Clip::new(0.0, 0.0, 10.0, 10.0, false);
        assert!(c.test(5, 5));
    }

    #[test]
    fn test_outside() {
        let c = Clip::new(0.0, 0.0, 10.0, 10.0, false);
        assert!(!c.test(15, 5));
        assert!(!c.test(5, 15));
    }

    #[test]
    fn clip_to_rect_shrinks() {
        let mut c = Clip::new(0.0, 0.0, 10.0, 10.0, false);
        c.clip_to_rect(2.0, 3.0, 8.0, 7.0);
        assert_eq!(c.x_min_i, 2);
        assert_eq!(c.y_min_i, 3);
    }

    #[test]
    fn test_rect_all_inside() {
        let c = Clip::new(0.0, 0.0, 20.0, 20.0, false);
        assert_eq!(c.test_rect(1, 1, 5, 5), ClipResult::AllInside);
    }

    #[test]
    fn test_rect_all_outside() {
        let c = Clip::new(0.0, 0.0, 10.0, 10.0, false);
        assert_eq!(c.test_rect(15, 15, 20, 20), ClipResult::AllOutside);
    }

    #[test]
    fn clone_shares_scanners() {
        let c = Clip::new(0.0, 0.0, 10.0, 10.0, false);
        let c2 = c.clone_shared();
        assert_eq!(c2.x_min_i, c.x_min_i);
        // Both should have the same (empty) scanner list.
        assert_eq!(c.scanners.len(), c2.scanners.len());
    }

    #[test]
    fn aa_coords_non_aa_passthrough() {
        assert_eq!(aa_coords(3, 7, 5, false), (3, 7, 5));
    }

    #[test]
    fn aa_coords_aa_scales() {
        // AA_SIZE = 4
        // sx0 = 3 * 4 = 12
        // sx1 = 7 * 4 + 3 = 31
        // sy  = 5 * 4 = 20
        assert_eq!(aa_coords(3, 7, 5, true), (12, 31, 20));
    }
}
