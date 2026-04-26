//! Scanline intersection lists for path fill and clip.
//!
//! [`XPathScanner`] converts an [`XPath`] edge table into per-scanline
//! intersection spans, which are then consumed by [`ScanIterator`] to emit
//! `(x0, x1)` spans for compositing.
//!
//! ## Design vs. C++ original
//!
//! The C++ `SplashXPathScanner` stores `std::vector<SplashIntersect>` (or
//! `boost::container::small_vector<SplashIntersect, 4>`) per scanline — one
//! heap allocation per row. This replacement uses a **flat `SoA` layout**:
//!
//! - `row_start[i]` is the index into `intersects` of the first intersection
//!   on scanline `y_min + i`.
//! - `row_start[n_rows]` is the total number of intersections (sentinel).
//! - All intersections are stored in a single `Vec<Intersect>`, sorted by `x0`
//!   within each row.
//!
//! This eliminates per-row heap allocations and is friendlier to the CPU cache
//! and future SIMD work.

pub mod iter;

use crate::bitmap::AaBuf;
use crate::types::{AA_SIZE, splash_floor};
use crate::xpath::{XPath, XPathFlags};

// ── Intersect ─────────────────────────────────────────────────────────────────

/// One intersection entry for a scanline.
///
/// Matches `SplashIntersect` in `splash/SplashXPathScanner.h`.
#[derive(Copy, Clone, Debug)]
pub struct Intersect {
    pub x0: i32, // left pixel (inclusive)
    pub x1: i32, // right pixel (inclusive)
    /// Winding contribution: +1 or -1 for sloped/vertical segments, 0 for horizontal.
    /// Used by non-zero winding number fill rule; ignored by even-odd.
    pub count: i32,
}

// ── XPathScanner ──────────────────────────────────────────────────────────────

/// Per-scanline intersection table built from an [`XPath`].
///
/// After construction the scanner is read-only; it is cheaply shareable via
/// [`Arc`] (matching the C++ `shared_ptr<SplashXPathScanner>` used in `SplashClip`).
pub struct XPathScanner {
    pub eo: bool, // even-odd fill rule (false = non-zero winding)
    pub x_min: i32,
    pub x_max: i32,
    pub y_min: i32,
    pub y_max: i32,
    /// `row_start[i]` = start of row `y_min + i` in `intersects`.
    /// Length = `(y_max - y_min + 2)` (includes sentinel at the end).
    row_start: Vec<u32>,
    /// Flat sorted intersection list for all rows.
    intersects: Vec<Intersect>,
}

impl XPathScanner {
    /// Build a scanner from `xpath`, clipping to `[clip_y_min, clip_y_max]`.
    ///
    /// If the xpath is empty or the y-ranges don't overlap, returns an empty
    /// scanner (`is_empty()` returns true).
    ///
    /// # Panics
    ///
    /// Panics if `y_max < y_min` after clamping (cannot happen when clip range is valid).
    #[must_use]
    pub fn new(xpath: &XPath, eo: bool, clip_y_min: i32, clip_y_max: i32) -> Self {
        if xpath.segs.is_empty() || clip_y_min > clip_y_max {
            return Self::empty(eo);
        }

        // Compute floating-point bounding box over segments that intersect the
        // clip range, discarding NaN-bearing segments.
        let mut x_min_fp = f64::INFINITY;
        let mut x_max_fp = f64::NEG_INFINITY;
        let mut y_min_fp = f64::INFINITY;
        let mut y_max_fp = f64::NEG_INFINITY;

        for seg in &xpath.segs {
            if seg.x0.is_nan() || seg.y0.is_nan() || seg.x1.is_nan() || seg.y1.is_nan() {
                continue;
            }
            let sy_min = seg.y0.min(seg.y1);
            let sy_max = seg.y0.max(seg.y1);
            if sy_min >= f64::from(clip_y_max + 1) || sy_max < f64::from(clip_y_min) {
                continue;
            }
            x_min_fp = x_min_fp.min(seg.x0).min(seg.x1);
            x_max_fp = x_max_fp.max(seg.x0).max(seg.x1);
            y_min_fp = y_min_fp.min(sy_min);
            y_max_fp = y_max_fp.max(sy_max);
        }

        if x_min_fp > x_max_fp {
            return Self::empty(eo);
        }

        let x_min = splash_floor(x_min_fp);
        let x_max = splash_floor(x_max_fp);
        let y_min = splash_floor(y_min_fp).max(clip_y_min);
        let y_max = splash_floor(y_max_fp).min(clip_y_max);

        if y_min > y_max {
            return Self::empty(eo);
        }

        let n_rows = usize::try_from(y_max - y_min + 1).expect("y_max >= y_min");

        // Accumulate intersections into per-row buckets, then flatten.
        let mut buckets: Vec<Vec<Intersect>> = vec![Vec::new(); n_rows];
        fill_buckets(xpath, y_min, y_max, &mut buckets);

        // Sort each row's intersections by x0, then flatten into the SoA layout.
        let mut row_start = Vec::with_capacity(n_rows + 1);
        let mut intersects = Vec::new();
        for bucket in &mut buckets {
            row_start.push(u32::try_from(intersects.len()).unwrap_or(u32::MAX));
            bucket.sort_unstable_by_key(|i| i.x0);
            intersects.extend_from_slice(bucket);
        }
        row_start.push(u32::try_from(intersects.len()).unwrap_or(u32::MAX));

        Self {
            eo,
            x_min,
            x_max,
            y_min,
            y_max,
            row_start,
            intersects,
        }
    }

    const fn empty(eo: bool) -> Self {
        Self {
            eo,
            x_min: 1,
            x_max: 0, // x_min > x_max → is_empty()
            y_min: 0,
            y_max: -1,
            row_start: Vec::new(),
            intersects: Vec::new(),
        }
    }

    /// True when the scanner covers no scanlines.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.x_min > self.x_max
    }

    /// The intersections for scanline `y`, sorted by `x0`.
    ///
    /// # Panics
    ///
    /// Panics if `y < y_min` (cannot happen when `y` is within range).
    #[must_use]
    pub fn row(&self, y: i32) -> &[Intersect] {
        if y < self.y_min || y > self.y_max {
            return &[];
        }
        let i = usize::try_from(y - self.y_min).expect("y >= y_min");
        let s = self.row_start[i] as usize;
        let e = self.row_start[i + 1] as usize;
        &self.intersects[s..e]
    }

    /// Test whether pixel `(x, y)` is inside the path.
    #[must_use]
    pub fn test(&self, x: i32, y: i32) -> bool {
        let row = self.row(y);
        let eo_mask: i32 = if self.eo { 1 } else { !0 };
        let mut count = 0i32;
        for int in row {
            if int.x0 > x {
                break;
            }
            if x <= int.x1 {
                return true;
            }
            count = count.wrapping_add(int.count);
        }
        (count & eo_mask) != 0
    }

    /// Test whether the entire span `[x0, x1]` on scanline `y` is inside.
    #[must_use]
    pub fn test_span(&self, x0: i32, x1: i32, y: i32) -> bool {
        let row = self.row(y);
        let eo_mask: i32 = if self.eo { 1 } else { !0 };
        let mut count = 0i32;
        let mut i = 0;
        let mut xx1 = x0 - 1;
        while xx1 < x1 {
            if i >= row.len() {
                return false;
            }
            if row[i].x0 > xx1 + 1 && (count & eo_mask) == 0 {
                return false;
            }
            if row[i].x1 > xx1 {
                xx1 = row[i].x1;
            }
            count = count.wrapping_add(row[i].count);
            i += 1;
        }
        true
    }

    /// Render the AA supersampled row to `aa_buf` and update `[x0, x1]` output range.
    ///
    /// Matches `SplashXPathScanner::renderAALine`. `y` is in device-pixel space;
    /// the scanner must have been built from an `aa_scale()`'d `XPath`.
    ///
    /// # Panics
    ///
    /// Panics if any AA sub-row index `yy` is negative (impossible since `AA_SIZE` > 0).
    pub fn render_aa_line(&self, aa_buf: &mut AaBuf, x0: &mut i32, x1: &mut i32, y: i32) {
        aa_buf.clear();
        let aa_width_i32 = i32::try_from(aa_buf.width).unwrap_or(i32::MAX);
        let mut xx_min = aa_width_i32;
        let mut xx_max = -1i32;

        let eo_mask: i32 = if self.eo { 1 } else { !0 };
        let aa = AA_SIZE;

        for yy in 0..aa {
            let scan_y = y * aa + yy;
            let row = self.row(scan_y);
            let mut count = 0i32;
            let mut i = 0;
            let mut xx = 0i32;
            while i < row.len() || (count & eo_mask) != 0 {
                // Find next covered span start.
                let _span_x0 = if i < row.len() {
                    row[i].x0
                } else {
                    aa_width_i32
                };
                // Advance past entries that are before the current position.
                if (count & eo_mask) == 0 {
                    // Not inside → skip to next intersection.
                    if i >= row.len() {
                        break;
                    }
                    xx = row[i].x0;
                    count = count.wrapping_add(row[i].count);
                    i += 1;
                    continue;
                }
                // Currently inside a covered region — find end of this span.
                let mut span_x1;
                loop {
                    if i >= row.len() {
                        span_x1 = aa_width_i32;
                        break;
                    }
                    count = count.wrapping_add(row[i].count);
                    span_x1 = row[i].x1 + 1;
                    i += 1;
                    if (count & eo_mask) == 0 {
                        break;
                    }
                }
                let bx0 = usize::try_from(xx.max(0)).unwrap_or(0);
                let bx1 = usize::try_from(span_x1.min(aa_width_i32)).unwrap_or(0);
                if bx0 < bx1 {
                    aa_buf.set_span(usize::try_from(yy).expect("yy >= 0"), bx0, bx1);
                    let bx0_i32 = i32::try_from(bx0).unwrap_or(i32::MAX);
                    let bx1_i32 = i32::try_from(bx1).unwrap_or(i32::MAX);
                    if bx0_i32 < xx_min {
                        xx_min = bx0_i32;
                    }
                    if bx1_i32 > xx_max {
                        xx_max = bx1_i32;
                    }
                }
                xx = span_x1;
            }
        }

        if xx_min > xx_max {
            xx_min = xx_max;
        }
        *x0 = xx_min / aa;
        *x1 = (xx_max - 1) / aa;
    }
}

// ── Bucket filling ────────────────────────────────────────────────────────────

/// Distribute all path segments from `xpath` into `buckets` (one bucket per
/// scanline `y_min..=y_max`). Called only from `XPathScanner::new`.
fn fill_buckets(xpath: &XPath, y_min: i32, y_max: i32, buckets: &mut [Vec<Intersect>]) {
    for seg in &xpath.segs {
        if seg.x0.is_nan() || seg.y0.is_nan() {
            continue;
        }

        let seg_y_min = seg.y0.min(seg.y1);
        let seg_y_max = seg.y0.max(seg.y1);

        let row_y0 = splash_floor(seg_y_min).max(y_min);
        let row_y1 = splash_floor(seg_y_max).min(y_max);

        if seg.flags.contains(XPathFlags::HORIZ) {
            // Horizontal segments: count = 0 (no winding contribution).
            let row = splash_floor(seg_y_min);
            if row >= y_min && row <= y_max {
                let x0 = splash_floor(seg.x0.min(seg.x1));
                let x1 = splash_floor(seg.x0.max(seg.x1));
                let bucket_idx = usize::try_from(row - y_min).expect("row >= y_min");
                add_intersection(&mut buckets[bucket_idx], x0, x1, 0, true);
            }
        } else if seg.flags.contains(XPathFlags::VERT) {
            // Vertical segments.
            let count = if seg.flags.contains(XPathFlags::FLIPPED) {
                1
            } else {
                -1
            };
            let x = splash_floor(seg.x0);
            for row in row_y0..=row_y1 {
                // Count is 0 on the topmost row (seg_y_min < row), matching C++.
                let c = if seg_y_min < f64::from(row) { count } else { 0 };
                let bucket_idx = usize::try_from(row - y_min).expect("row >= y_min");
                add_intersection(&mut buckets[bucket_idx], x, x, c, false);
            }
        } else {
            // Sloped segment: interpolate x across the scanline range.
            let count = if seg.flags.contains(XPathFlags::FLIPPED) {
                1
            } else {
                -1
            };
            let x_base = seg.y0.mul_add(-seg.dxdy, seg.x0);
            let sloped_x_min = seg.x0.min(seg.x1);
            let sloped_x_max = seg.x0.max(seg.x1);

            let xx0 = if f64::from(row_y0) > seg.y0 {
                f64::from(row_y0).mul_add(seg.dxdy, x_base)
            } else {
                seg.x0
            };
            let xx0 = xx0.clamp(sloped_x_min, sloped_x_max);
            let mut px0 = splash_floor(xx0);

            for row in row_y0..=row_y1 {
                let xx1 = f64::from(row + 1)
                    .mul_add(seg.dxdy, x_base)
                    .clamp(sloped_x_min, sloped_x_max);
                let px1 = splash_floor(xx1);
                let c = if seg_y_min < f64::from(row) { count } else { 0 };
                let bucket_idx = usize::try_from(row - y_min).expect("row >= y_min");
                add_intersection(&mut buckets[bucket_idx], px0, px1, c, false);
                px0 = px1;
            }
        }
    }
}

// ── Intersection helper ───────────────────────────────────────────────────────

/// Add or merge one intersection entry into a row bucket.
///
/// Adjacent or overlapping entries are merged (count accumulated, x0/x1 expanded),
/// matching `SplashXPathScanner::addIntersection` in `SplashXPathScanner.cc`.
fn add_intersection(row: &mut Vec<Intersect>, x0: i32, x1: i32, count: i32, _is_horiz: bool) {
    let (lo, hi) = (x0.min(x1), x0.max(x1));
    let entry = Intersect {
        x0: lo,
        x1: hi,
        count,
    };

    if row.is_empty() {
        row.push(entry);
        return;
    }
    let last = row.last_mut().unwrap();
    if last.x1 + 1 < entry.x0 || last.x0 > entry.x1 + 1 {
        // Disjoint — push new entry.
        row.push(entry);
    } else {
        // Touch or overlap — merge.
        last.count = last.count.wrapping_add(entry.count);
        last.x0 = last.x0.min(entry.x0);
        last.x1 = last.x1.max(entry.x1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::path::PathBuilder;
    use crate::xpath::XPath;

    fn identity() -> [f64; 6] {
        [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    }

    fn triangle_xpath() -> XPath {
        let mut b = PathBuilder::new();
        b.move_to(0.0, 0.0).unwrap();
        b.line_to(4.0, 0.0).unwrap();
        b.line_to(2.0, 4.0).unwrap();
        b.close(false).unwrap();
        XPath::new(&b.build(), &identity(), 1.0, true)
    }

    #[test]
    fn empty_on_no_segs() {
        let xpath = XPath::empty();
        // Need to access private field; use new() on an empty XPath
        let scanner = XPathScanner::new(&xpath, false, 0, 10);
        assert!(scanner.is_empty());
    }

    #[test]
    fn triangle_has_rows() {
        let xpath = triangle_xpath();
        let scanner = XPathScanner::new(&xpath, false, 0, 4);
        assert!(!scanner.is_empty());
        assert!(scanner.y_min <= scanner.y_max);
    }

    #[test]
    fn horizontal_segment_count_zero() {
        // A purely horizontal segment should produce count=0.
        let _xpath = XPath::empty();
        // Directly test add_intersection
        let mut row = Vec::new();
        add_intersection(&mut row, 0, 5, 0, true);
        assert_eq!(row[0].count, 0);
    }

    #[test]
    fn test_inside_triangle() {
        let xpath = triangle_xpath();
        let scanner = XPathScanner::new(&xpath, false, 0, 4);
        // Centroid of the triangle is roughly (2, 1.33)
        assert!(scanner.test(2, 1));
    }

    #[test]
    fn test_outside_triangle() {
        let xpath = triangle_xpath();
        let scanner = XPathScanner::new(&xpath, false, 0, 4);
        assert!(!scanner.test(10, 2));
        assert!(!scanner.test(2, 5));
    }
}
