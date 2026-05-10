//! Scanline intersection lists for path fill and clip.
//!
//! [`XPathScanner`] converts an [`XPath`] edge table into per-scanline
//! intersection spans, which are then consumed by `ScanIterator` (a
//! private iterator type) to emit `(x0, x1)` spans for compositing.
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
//!
//! ## Two-pass construction (Tier 3)
//!
//! `XPathScanner::new` uses a two-pass counting-sort to avoid `n_rows`
//! individual `Vec` allocations:
//!
//! 1. **Count pass**: walk segments, increment a `counts[row]` entry for each
//!    intersection generated (identical edge-walking logic as the fill pass).
//! 2. **Prefix-sum**: convert counts into `row_start` offsets; allocate the
//!    flat `intersects` buffer in one shot.
//! 3. **Fill pass**: walk segments again, writing into the pre-allocated slots.
//! 4. **Sort**: sort each row's slice in-place by `x0`.
//!
//! This reduces construction from `O(n_rows)` heap allocs + flatten to a single
//! pair of allocations regardless of path height.

/// The coalescing span iterator for a scanner row.
pub mod iter;

use crate::bitmap::AaBuf;
use crate::types::{AA_SIZE, splash_floor};
use crate::xpath::{XPath, XPathFlags};

// ── Fixed-point helpers ───────────────────────────────────────────────────────

/// Convert an `f64` device-space coordinate to 16.16 fixed-point (`i32`).
///
/// Values outside the i32 range are saturated — in practice this only happens
/// for degenerate coordinates beyond ±32 768 px (impossible for a real page).
#[inline]
fn fp_from_f64(x: f64) -> i32 {
    let fp = x * 65536.0;
    if fp >= f64::from(i32::MAX) {
        i32::MAX
    } else if fp <= f64::from(i32::MIN) {
        i32::MIN
    } else {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "fp is bounds-checked to [i32::MIN, i32::MAX] by the branches above"
        )]
        {
            fp as i32
        }
    }
}

// ── Intersect ─────────────────────────────────────────────────────────────────

/// One intersection entry for a scanline.
///
/// Matches `SplashIntersect` in `splash/SplashXPathScanner.h`.
#[derive(Copy, Clone, Debug)]
pub struct Intersect {
    /// Left pixel of the intersection span, inclusive.
    pub x0: i32,
    /// Right pixel of the intersection span, inclusive. Always `x0 ≤ x1`.
    pub x1: i32,
    /// Winding contribution: +1 or -1 for sloped/vertical segments, 0 for horizontal.
    /// Used by non-zero winding number fill rule; ignored by even-odd.
    pub count: i32,
}

// ── XPathScanner ──────────────────────────────────────────────────────────────

/// Per-scanline intersection table built from an [`XPath`].
///
/// After construction the scanner is read-only; it is cheaply shareable via
/// `Arc` (matching the C++ `shared_ptr<SplashXPathScanner>` used in `SplashClip`).
///
/// ## `SoA` layout invariant
///
/// `row_start` has length `(y_max - y_min + 2)` when non-empty: one entry per
/// scanline plus a sentinel at index `n_rows`. For any valid row index `i`:
///
/// ```text
/// row_start[i] <= row_start[i + 1] <= intersects.len()
/// ```
///
/// Intersections within each row are sorted in ascending `x0` order.
pub struct XPathScanner {
    /// Fill rule: `true` = even-odd, `false` = non-zero winding number.
    pub eo: bool,
    /// Minimum device-pixel x-coordinate of any intersection in the table.
    pub x_min: i32,
    /// Maximum device-pixel x-coordinate of any intersection in the table.
    pub x_max: i32,
    /// First scanline (device-pixel y) covered by this scanner.
    pub y_min: i32,
    /// Last scanline (device-pixel y) covered by this scanner, inclusive.
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
    /// Panics if the number of rows after clamping overflows `usize` (cannot
    /// happen for valid PDF coordinate ranges, as `y_max - y_min + 1` is at
    /// most `i32::MAX`).
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
            if sy_min >= f64::from(clip_y_max) + 1.0 || sy_max < f64::from(clip_y_min) {
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

        let n_rows = usize::try_from(i64::from(y_max) - i64::from(y_min) + 1)
            .expect("n_rows overflows usize; page height is impossibly large");

        // ── Two-pass construction: count → prefix-sum → fill → finalise ──────
        //
        // Pass 1: count the maximum number of Intersect slots needed per row.
        // Adjacent-span merging may reduce the final count, so this is an upper bound.
        let mut counts = vec![0u32; n_rows];
        count_intersections(xpath, y_min, y_max, &mut counts);

        // Pass 2: prefix-sum to produce base offsets; allocate the flat buffer.
        let mut alloc_start = Vec::with_capacity(n_rows + 1);
        let mut total = 0u32;
        for &c in &counts {
            alloc_start.push(total);
            total = total
                .checked_add(c)
                .expect("intersection count exceeds u32::MAX; path has too many segments");
        }
        alloc_start.push(total);
        let total_n = total as usize;
        let mut intersects = vec![
            Intersect {
                x0: 0,
                x1: 0,
                count: 0
            };
            total_n
        ];

        // Pass 3: fill the flat buffer.  `cursors[i]` counts the actual number of
        // entries written into row i (≤ counts[i] after merging).  Reset counts
        // to zero and use them as the write cursors.
        counts.fill(0);
        fill_intersections(
            xpath,
            y_min,
            y_max,
            &alloc_start,
            &mut counts,
            &mut intersects,
        );
        // `counts` now holds the actual written count per row.

        // Pass 4: build the final row_start from actual counts, compact intersects,
        // and sort each row slice by x0.
        //
        // Compaction is needed because merging may have left unfilled slots between
        // rows (each row was pre-allocated for its upper-bound count, but the actual
        // fill may be smaller).  We re-pack in place using a single forward pass.
        let mut row_start = Vec::with_capacity(n_rows + 1);
        let mut write_pos = 0usize;
        for i in 0..n_rows {
            row_start.push(
                u32::try_from(write_pos).expect("compacted intersection count exceeds u32::MAX"),
            );
            let read_base = alloc_start[i] as usize;
            let n_written = counts[i] as usize;
            // Sort the actual written slice before copying.
            intersects[read_base..read_base + n_written].sort_unstable_by_key(|ix| ix.x0);
            // Copy into the compacted position (may overlap if no merging occurred).
            intersects.copy_within(read_base..read_base + n_written, write_pos);
            write_pos += n_written;
        }
        row_start
            .push(u32::try_from(write_pos).expect("compacted intersection count exceeds u32::MAX"));
        intersects.truncate(write_pos);

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
        self.x_min > self.x_max || self.y_min > self.y_max
    }

    /// Iterate over only the scanlines that have at least one intersection.
    ///
    /// Skips empty rows using the `row_start` sentinel array — an empty row has
    /// `row_start[i] == row_start[i+1]`.  For paths that are sparse over the
    /// bounding-box height (e.g., a thin diagonal across a tall page), this
    /// eliminates the per-row overhead for all-empty rows.
    ///
    /// Yields absolute device-pixel `y` values in ascending order.
    pub fn nonempty_rows(&self) -> impl Iterator<Item = i32> + '_ {
        // row_start has length n_rows + 1 (sentinel).  Row i is non-empty
        // iff row_start[i] < row_start[i + 1].
        self.row_start
            .windows(2)
            .enumerate()
            .filter(|(_, w)| w[0] < w[1])
            // i < n_rows ≤ i32::MAX (enforced by n_rows construction); saturating_add
            // is a no-cost guard against logic errors in debug builds.
            .map(|(i, _)| {
                self.y_min
                    .saturating_add(i32::try_from(i).unwrap_or(i32::MAX))
            })
    }

    /// The intersections for scanline `y`, sorted by `x0`.
    ///
    /// Returns an empty slice if `y` is outside `[y_min, y_max]`. Never panics.
    #[must_use]
    pub fn row(&self, y: i32) -> &[Intersect] {
        if y < self.y_min || y > self.y_max {
            return &[];
        }
        // y_min <= y <= y_max is guaranteed by the check above, so the
        // subtraction is non-negative and within [0, y_max - y_min].
        // That range fits in usize on any platform; cast is safe.
        #[expect(
            clippy::cast_sign_loss,
            reason = "y >= y_min is asserted by the bounds check above; \
                      the difference is non-negative"
        )]
        let i = (y - self.y_min) as usize;
        debug_assert!(i + 1 < self.row_start.len(), "row_start sentinel missing");
        // row_start entries are u32; on all supported (32/64-bit) platforms
        // usize >= u32, so `as usize` is a widening cast that cannot lose data.
        let s = self.row_start[i] as usize;
        let e = self.row_start[i + 1] as usize;
        debug_assert!(
            s <= e && e <= self.intersects.len(),
            "row_start invariant violated"
        );
        &self.intersects[s..e]
    }

    /// Accumulate winding crossings for all intersections in `row` up to (but
    /// not including) pixel `x`.
    ///
    /// Returns `(count, on_span)` where:
    /// - `count` is the sum of `Intersect::count` for every entry whose span
    ///   ends before `x` (i.e. `x1 < x`).
    /// - `on_span` is `true` if `x` falls within one of the intersection spans.
    ///
    /// The caller applies the fill-rule mask to `count` to decide whether the
    /// pixel is inside the path when `on_span` is `false`.
    fn count_crossings(row: &[Intersect], x: i32) -> (i32, bool) {
        let mut count = 0i32;
        for int in row {
            if int.x0 > x {
                break;
            }
            if x <= int.x1 {
                // `x` falls inside this intersection span.
                return (count, true);
            }
            count = count.wrapping_add(int.count);
        }
        (count, false)
    }

    /// Test whether pixel `(x, y)` is inside the path.
    #[must_use]
    pub fn test(&self, x: i32, y: i32) -> bool {
        let row = self.row(y);
        let eo_mask = if self.eo { 1i32 } else { !0i32 };
        let (count, on_span) = Self::count_crossings(row, x);
        on_span || (count & eo_mask) != 0
    }

    /// Test whether the entire span `[x0, x1]` on scanline `y` is inside the path.
    ///
    /// Returns `true` only if every pixel in `[x0, x1]` is covered by the fill.
    #[must_use]
    pub fn test_span(&self, x0: i32, x1: i32, y: i32) -> bool {
        let row = self.row(y);
        let eo_mask = if self.eo { 1i32 } else { !0i32 };
        let mut count = 0i32;
        let mut i = 0;
        let mut xx1 = x0.saturating_sub(1);
        while xx1 < x1 {
            if i >= row.len() {
                return false;
            }
            if row[i].x0 > xx1.saturating_add(1) && (count & eo_mask) == 0 {
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
    /// `x0` and `x1` are updated to the tightest bounding range of pixels
    /// written into `aa_buf`. If no pixels are written, `*x0` is set to `0`
    /// and `*x1` to `-1` (so `*x0 > *x1` reliably indicates an empty range).
    ///
    /// # Panics
    ///
    /// Does not panic in practice. `AA_SIZE` is a positive compile-time constant
    /// so all `try_from` conversions on loop indices succeed.
    pub fn render_aa_line(&self, aa_buf: &mut AaBuf, x0: &mut i32, x1: &mut i32, y: i32) {
        aa_buf.clear();
        let aa_width_i32 = i32::try_from(aa_buf.width).unwrap_or(i32::MAX);
        let mut xx_min = aa_width_i32;
        let mut xx_max = -1i32;

        let eo_mask = if self.eo { 1i32 } else { !0i32 };
        let aa = AA_SIZE;

        for yy in 0..aa {
            let scan_y = y * aa + yy;
            let row = self.row(scan_y);
            let mut count = 0i32;
            let mut i = 0usize;
            let mut xx = 0i32;

            // Walk through intersections, emitting covered spans into aa_buf.
            //
            // Winding rule: we are "inside" the path when
            //   (count & eo_mask) != 0
            // where count is the running sum of `Intersect::count` values seen so
            // far. For EO fill eo_mask=1 so the low bit tracks parity; for NZ
            // winding eo_mask=!0 so any non-zero count means inside.
            while i < row.len() || (count & eo_mask) != 0 {
                if (count & eo_mask) == 0 {
                    // Currently outside — skip to the next intersection entry.
                    if i >= row.len() {
                        break;
                    }
                    xx = row[i].x0;
                    count = count.wrapping_add(row[i].count);
                    i += 1;
                    continue;
                }

                // Currently inside a covered region — find the end of this span.
                let span_x1 = loop {
                    if i >= row.len() {
                        break aa_width_i32;
                    }
                    count = count.wrapping_add(row[i].count);
                    let end = row[i].x1 + 1;
                    i += 1;
                    if (count & eo_mask) == 0 {
                        break end;
                    }
                };

                let bx0 = usize::try_from(xx.max(0)).unwrap_or(0);
                let bx1 = usize::try_from(span_x1.min(aa_width_i32)).unwrap_or(0);
                if bx0 < bx1 {
                    let row_idx = usize::try_from(yy).expect("yy in 0..AA_SIZE, always >= 0");
                    aa_buf.set_span(row_idx, bx0, bx1);
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
            // Empty range signal: x0 > x1.
            *x0 = 0;
            *x1 = -1;
            return;
        }
        *x0 = xx_min / aa;
        *x1 = (xx_max - 1) / aa;
    }
}

// ── Two-pass intersection construction ───────────────────────────────────────

/// Pass 1: count how many `Intersect` entries each row will receive.
///
/// Walks the same segment logic as [`fill_intersections`] without writing
/// anything, so the two passes must stay in sync.
fn count_intersections(xpath: &XPath, y_min: i32, y_max: i32, counts: &mut [u32]) {
    for seg in &xpath.segs {
        if seg.x0.is_nan() || seg.y0.is_nan() {
            continue;
        }
        let seg_y_min = seg.y0.min(seg.y1);
        let seg_y_max = seg.y0.max(seg.y1);
        let row_y0 = splash_floor(seg_y_min).max(y_min);
        let row_y1 = splash_floor(seg_y_max).min(y_max);

        if seg.flags.contains(XPathFlags::HORIZ) {
            let row = splash_floor(seg_y_min);
            if row >= y_min && row <= y_max {
                let idx = usize::try_from(row - y_min).expect("row >= y_min");
                counts[idx] = counts[idx].saturating_add(1);
            }
        } else {
            for row in row_y0..=row_y1 {
                let idx = usize::try_from(row - y_min).expect("row >= y_min");
                counts[idx] = counts[idx].saturating_add(1);
            }
        }
    }
}

/// Pass 3: write `Intersect` entries into the pre-allocated flat buffer.
///
/// `row_start` holds the base offset for each row (prefix-summed from counts).
/// `cursors` starts at all-zero and is incremented per row as entries are written —
/// so `row_start[i] + cursors[i]` is the next free slot for row `i`.
///
/// Sloped segments use a 16.16 fixed-point integer accumulator (`xx1_fp += dxdy_fp`)
/// for the per-scanline step.  The accumulator is initialised from f64 for the
/// first row to preserve accuracy, then stepped by the pre-computed `i32` slope
/// — eliminating all f64 arithmetic from the inner loop.
fn fill_intersections(
    xpath: &XPath,
    y_min: i32,
    y_max: i32,
    row_start: &[u32],
    cursors: &mut [u32],
    intersects: &mut [Intersect],
) {
    for seg in &xpath.segs {
        if seg.x0.is_nan() || seg.y0.is_nan() {
            continue;
        }
        let seg_y_min = seg.y0.min(seg.y1);
        let seg_y_max = seg.y0.max(seg.y1);
        let row_y0 = splash_floor(seg_y_min).max(y_min);
        let row_y1 = splash_floor(seg_y_max).min(y_max);

        if seg.flags.contains(XPathFlags::HORIZ) {
            let row = splash_floor(seg_y_min);
            if row >= y_min && row <= y_max {
                let x0 = splash_floor(seg.x0.min(seg.x1));
                let x1 = splash_floor(seg.x0.max(seg.x1));
                write_intersect(row, y_min, row_start, cursors, intersects, x0, x1, 0);
            }
        } else if seg.flags.contains(XPathFlags::VERT) {
            let count = if seg.flags.contains(XPathFlags::FLIPPED) {
                1
            } else {
                -1
            };
            let x = splash_floor(seg.x0);
            for row in row_y0..=row_y1 {
                let c = if seg_y_min < f64::from(row) { count } else { 0 };
                write_intersect(row, y_min, row_start, cursors, intersects, x, x, c);
            }
        } else {
            let count = if seg.flags.contains(XPathFlags::FLIPPED) {
                1
            } else {
                -1
            };
            let x_base = seg.y0.mul_add(-seg.dxdy, seg.x0);
            let sloped_x_min = seg.x0.min(seg.x1);
            let sloped_x_max = seg.x0.max(seg.x1);

            // Clamp bounds in 16.16 fixed-point for the integer inner loop.
            // Shift left 16 bits; saturate to i32 range (overflows only for
            // geometrically impossible coordinates, >32k px).
            let fp_clamp_min = fp_from_f64(sloped_x_min);
            let fp_clamp_max = fp_from_f64(sloped_x_max);

            // Initial x at the top of the first row (or seg.x0 if row_y0 == seg.y0).
            let xx_init = if f64::from(row_y0) > seg.y0 {
                f64::from(row_y0).mul_add(seg.dxdy, x_base)
            } else {
                seg.x0
            };
            let mut px0 = splash_floor(xx_init.clamp(sloped_x_min, sloped_x_max));

            // 16.16 fixed-point accumulator for the right edge of each span.
            // Initialised from f64 to preserve accuracy at the first row, then
            // stepped by `dxdy_fp` (integer add) per scanline — no f64 arithmetic
            // inside the hot loop.
            let mut xx1_fp = fp_from_f64((f64::from(row_y0) + 1.0).mul_add(seg.dxdy, x_base));

            for row in row_y0..=row_y1 {
                let xx1_fp_clamped = xx1_fp.clamp(fp_clamp_min, fp_clamp_max);
                // Arithmetic right-shift gives floor() for both positive and negative.
                let px1 = xx1_fp_clamped >> 16;
                let c = if seg_y_min < f64::from(row) { count } else { 0 };
                write_intersect(row, y_min, row_start, cursors, intersects, px0, px1, c);
                px0 = px1;
                xx1_fp = xx1_fp.saturating_add(seg.dxdy_fp);
            }
        }
    }
}

/// Write one `Intersect` entry into its row slot, applying the same
/// adjacent-span merge logic as `add_intersection`.
///
/// `cursors[idx]` tracks how many entries have been written into row `idx` so far.
/// The slot is `row_start[idx] + cursors[idx] - 1` for merge, or `+ cursors[idx]`
/// for a new entry.
#[inline]
#[expect(
    clippy::too_many_arguments,
    reason = "private helper: all params are load-bearing"
)]
fn write_intersect(
    row: i32,
    y_min: i32,
    row_start: &[u32],
    cursors: &mut [u32],
    intersects: &mut [Intersect],
    x0: i32,
    x1: i32,
    count: i32,
) {
    let (lo, hi) = (x0.min(x1), x0.max(x1));
    let entry = Intersect {
        x0: lo,
        x1: hi,
        count,
    };
    let idx = usize::try_from(row - y_min).expect("row >= y_min");
    let base = row_start[idx] as usize;
    let cur = cursors[idx] as usize;

    if cur > 0 {
        // Check the last written entry in this row for adjacency/overlap.
        let last = &mut intersects[base + cur - 1];
        if last.x1.saturating_add(1) >= entry.x0 && last.x0 <= entry.x1.saturating_add(1) {
            // Merge: expand span and accumulate count.
            last.count = last.count.wrapping_add(entry.count);
            last.x0 = last.x0.min(entry.x0);
            last.x1 = last.x1.max(entry.x1);
            return;
        }
    }

    // New entry: write into the next free slot.
    // The count pass guarantees capacity; the debug_assert guards against
    // logic divergence between the two passes.
    debug_assert!(
        base + cur < intersects.len(),
        "write_intersect: slot out of range (row={row}, base={base}, cur={cur}, len={})",
        intersects.len()
    );
    intersects[base + cur] = entry;
    cursors[idx] += 1;
}

// ── Intersection helper ───────────────────────────────────────────────────────

/// Add or merge one intersection entry into a `Vec<Intersect>` row buffer.
///
/// Used only in unit tests to exercise the merge/disjoint logic independently.
/// Production code uses [`write_intersect`] with the pre-allocated flat buffer.
///
/// Two spans are merged when `last.x1 + 1 >= entry.x0` *and*
/// `last.x0 <= entry.x1 + 1` (i.e. they touch or overlap in either direction).
/// This matches `SplashXPathScanner::addIntersection` in `SplashXPathScanner.cc`.
#[cfg(test)]
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
    if last.x1.saturating_add(1) < entry.x0 || last.x0 > entry.x1.saturating_add(1) {
        row.push(entry);
    } else {
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

    #[test]
    fn row_out_of_range_returns_empty() {
        let xpath = triangle_xpath();
        let scanner = XPathScanner::new(&xpath, false, 0, 4);
        // y below y_min
        assert!(scanner.row(scanner.y_min - 1).is_empty());
        // y above y_max
        assert!(scanner.row(scanner.y_max + 1).is_empty());
        // large out-of-range values
        assert!(scanner.row(i32::MIN).is_empty());
        assert!(scanner.row(i32::MAX).is_empty());
    }

    #[test]
    fn add_intersection_merge_adjacent() {
        // [0,3] then [4,7]: last.x1+1 == entry.x0, so they should merge.
        let mut row = Vec::new();
        add_intersection(&mut row, 0, 3, -1, false);
        add_intersection(&mut row, 4, 7, 1, false);
        assert_eq!(row.len(), 1, "adjacent spans must merge");
        assert_eq!(row[0].x0, 0);
        assert_eq!(row[0].x1, 7);
        assert_eq!(row[0].count, 0); // -1 + 1 = 0
    }

    #[test]
    fn add_intersection_disjoint() {
        // [0,2] then [5,7]: gap of 2 pixels, must NOT merge.
        let mut row = Vec::new();
        add_intersection(&mut row, 0, 2, -1, false);
        add_intersection(&mut row, 5, 7, -1, false);
        assert_eq!(row.len(), 2, "disjoint spans must not merge");
    }

    #[test]
    fn count_crossings_inside_span() {
        // count_crossings with x inside an intersection span returns on_span=true.
        let row = vec![Intersect {
            x0: 2,
            x1: 5,
            count: -1,
        }];
        let (count, on_span) = XPathScanner::count_crossings(&row, 3);
        assert!(on_span);
        assert_eq!(count, 0); // not yet accumulated the entry's count
    }

    #[test]
    fn count_crossings_after_span() {
        // After span [2,5] count=-1: pixel 6 is outside, count reflects crossing.
        let row = vec![Intersect {
            x0: 2,
            x1: 5,
            count: -1,
        }];
        let (count, on_span) = XPathScanner::count_crossings(&row, 6);
        assert!(!on_span);
        assert_eq!(count, -1);
    }
}
