//! Coalescing span iterator over a scanner's intersection row.
//!
//! [`ScanIterator`] mirrors `SplashXPathScanIterator::getNextSpan` from
//! `splash/SplashXPathScanner.cc`. It merges adjacent and overlapping
//! intersections into contiguous `(x0, x1)` spans (inclusive) and handles
//! both even-odd and non-zero winding fill rules via the `eo_mask`.
//!
//! ## Winding-rule semantics
//!
//! The iterator maintains a running `count` — the sum of all
//! [`Intersect::count`] values consumed so far. Whether the current pixel
//! position is *inside* the path depends on the fill rule:
//!
//! - **Non-zero winding** (`eo = false`): inside when `count != 0`.
//!   `eo_mask = !0` so `count & eo_mask` is non-zero iff `count != 0`.
//! - **Even-odd** (`eo = true`): inside when the crossing count is odd.
//!   `eo_mask = 1` so `count & eo_mask` isolates the low bit (parity).
//!
//! ## Algorithm (`next`)
//!
//! ### Phase 1 — find the span start
//!
//! Before consuming each entry, check `count & eo_mask`:
//!
//! - Already non-zero → we are inside from a previous span's count residue;
//!   the current entry (`row[idx]`) is the span start. Record its `x0` and go
//!   to phase 2 *without* consuming it yet (phase 2 will consume it).
//! - Zero → consume the entry (add its `count`, advance `idx`). If now
//!   non-zero → this entry is the span start; record its `x0` (from
//!   `row[idx - 1]`) and go to phase 2 (the entry is already consumed).
//! - Still zero → continue to the next entry.
//!
//! ### Phase 2 — collect the span
//!
//! Starting from `row[phase2_start_idx]` (inclusive), consume entries, tracking
//! the maximum `x1` seen, until `count & eo_mask == 0`. Emit `(x0, x1)`.
//!
//! Each `Intersect` entry's `count` is accumulated **exactly once** across the
//! two phases. After `next` returns, `count` equals the winding number
//! immediately after the rightmost covered pixel `x1`.

use super::{Intersect, XPathScanner};

/// An iterator that yields `(x0, x1)` inclusive pixel spans for one scanline.
///
/// Spans are emitted in left-to-right order with no overlap.
///
/// Constructed via [`ScanIterator::new`]. Out-of-range `y` values produce an
/// immediately-exhausted iterator (matching the C++ empty-row behaviour).
pub struct ScanIterator<'a> {
    /// The sorted intersection list for this scanline.
    row: &'a [Intersect],
    /// Index of the next un-consumed entry in `row`.
    idx: usize,
    /// Running winding count: sum of `Intersect::count` for all consumed entries.
    ///
    /// Represents the winding number immediately *after* the rightmost pixel
    /// covered by the last consumed entry. Starts at 0 (outside the path).
    count: i32,
    /// Fill-rule mask.
    ///
    /// `!0` for non-zero winding (`count != 0` → inside); `1` for even-odd
    /// (`count & 1 != 0` → inside). See module-level doc for details.
    eo_mask: i32,
}

impl<'a> ScanIterator<'a> {
    /// Create an iterator for scanline `y` of `scanner`.
    ///
    /// If `y` is outside the scanner's range, the iterator is immediately
    /// exhausted (matching C++ behaviour where the empty `allIntersections[0]`
    /// row is returned with `interIdx = line.size()`).
    #[must_use]
    pub fn new(scanner: &'a XPathScanner, y: i32) -> Self {
        let row = scanner.row(y);
        let eo_mask = if scanner.eo { 1i32 } else { !0i32 };
        Self {
            row,
            idx: 0,
            count: 0,
            eo_mask,
        }
    }
}

impl Iterator for ScanIterator<'_> {
    /// `(x0, x1)` — both inclusive pixel coordinates.
    type Item = (i32, i32);

    /// Advance to the next filled span.
    ///
    /// See the [module-level algorithm description](self) for a precise
    /// account of the two-phase skip-then-collect approach and the
    /// winding-rule semantics governing each phase.
    fn next(&mut self) -> Option<(i32, i32)> {
        // ── Phase 1: find the entry that starts the next filled span ──────────
        //
        // We peek at `count & eo_mask` *before* consuming each entry:
        //   • Non-zero already → inside from previous call's count residue.
        //     `row[idx]` is the span start; leave idx unchanged for phase 2.
        //   • Zero → consume the entry; if count goes non-zero this entry is
        //     the span start (idx was already advanced).
        //   • Still zero → not inside; continue to the next entry.
        let (x0, phase2_start) = loop {
            if self.idx >= self.row.len() {
                return None;
            }
            if (self.count & self.eo_mask) != 0 {
                // Already inside (residue from previous span).
                // The span starts at row[idx]; do NOT consume it here —
                // phase 2 will consume it as the first collected entry.
                let x0 = self.row[self.idx].x0;
                break (x0, self.idx);
            }
            // Try consuming this entry.
            let new_count = self.count.wrapping_add(self.row[self.idx].count);
            self.idx += 1;
            self.count = new_count;
            if (self.count & self.eo_mask) != 0 {
                // This entry pushed us inside; it is the span start.
                let x0 = self.row[self.idx - 1].x0;
                break (x0, self.idx - 1);
            }
            // Still outside; continue.
        };

        // ── Phase 2: collect entries while inside the path ────────────────────
        //
        // `phase2_start` is the index of the entry that starts the span.
        //
        // "transition" branch: that entry was already consumed in phase 1
        //   (self.idx == phase2_start + 1); seed x1 from it, then continue.
        // "already inside" branch: that entry has NOT been consumed yet
        //   (self.idx == phase2_start); consume it now to seed x1.
        let mut x1 = self.row[phase2_start].x1;
        if self.idx == phase2_start {
            // "already inside": consume the span-start entry.
            self.count = self.count.wrapping_add(self.row[self.idx].count);
            self.idx += 1;
        }
        // "transition": span-start is already consumed; self.idx is correct.

        // Continue consuming while still inside.
        while self.idx < self.row.len() && (self.count & self.eo_mask) != 0 {
            x1 = x1.max(self.row[self.idx].x1);
            self.count = self.count.wrapping_add(self.row[self.idx].count);
            self.idx += 1;
        }

        Some((x0, x1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::path::PathBuilder;
    use crate::scanner::XPathScanner;
    use crate::xpath::XPath;

    fn identity() -> [f64; 6] {
        [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    }

    fn rect_xpath(x0: f64, y0: f64, x1: f64, y1: f64) -> XPath {
        let mut b = PathBuilder::new();
        b.move_to(x0, y0).unwrap();
        b.line_to(x1, y0).unwrap();
        b.line_to(x1, y1).unwrap();
        b.line_to(x0, y1).unwrap();
        b.close(false).unwrap();
        XPath::new(&b.build(), &identity(), 1.0, true)
    }

    #[test]
    fn rect_span_single_row() {
        let xpath = rect_xpath(2.0, 1.0, 6.0, 3.0);
        let scanner = XPathScanner::new(&xpath, false, 0, 10);
        let spans: Vec<_> = ScanIterator::new(&scanner, 2).collect();
        assert!(!spans.is_empty(), "expected at least one span on y=2");
        // The span should cover roughly [2, 5] (inclusive).
        let (x0, x1) = spans[0];
        assert!(x0 <= 2, "x0={x0}");
        assert!(x1 >= 4, "x1={x1}");
    }

    #[test]
    fn out_of_range_y_empty() {
        let xpath = rect_xpath(0.0, 0.0, 4.0, 4.0);
        let scanner = XPathScanner::new(&xpath, false, 0, 10);
        assert!(ScanIterator::new(&scanner, 100).next().is_none());
    }

    #[test]
    fn coalescing_adjacent_nz() {
        // NZ winding: [0,3] count=-1 then [4,7] count=+1.
        // count: 0 → (consume row[0]) -1 → inside; span starts at x0=0.
        // phase2: x1=3, count=-1. Consume row[1]: x1=max(3,7)=7, count=0. Outside.
        // Emit (0, 7). One span total.
        use crate::scanner::Intersect;
        let row = vec![
            Intersect { x0: 0, x1: 3, count: -1 },
            Intersect { x0: 4, x1: 7, count: 1 },
        ];
        let scanner_stub = XPathScanner {
            eo: false,
            x_min: 0,
            x_max: 7,
            y_min: 0,
            y_max: 0,
            row_start: vec![0, 2],
            intersects: row,
        };
        let spans: Vec<_> = ScanIterator::new(&scanner_stub, 0).collect();
        assert_eq!(spans, vec![(0, 7)]);
    }

    #[test]
    fn two_disjoint_spans_nz() {
        // NZ winding: two separate filled regions.
        // [0,2,-1] → inside; [3,5,+1] → count=0, outside → emit (0,5).
        // Wait, -1+1=0 so this is actually one span [0,5] that ends at row[1].
        // Use a different configuration for genuinely two spans:
        // [0,2,-1], [3,5,+1], [8,10,-1], [11,13,+1]
        // Call 1: consume [0,2,-1] → inside, x0=0. Phase2: x1=2, count=-1.
        //   Consume [3,5,+1]: x1=max(2,5)=5, count=0. Outside. Emit (0,5).
        // Call 2: count=0. Consume [8,10,-1] → inside, x0=8. Phase2: x1=10, count=-1.
        //   Consume [11,13,+1]: x1=max(10,13)=13, count=0. Outside. Emit (8,13).
        use crate::scanner::Intersect;
        let row = vec![
            Intersect { x0: 0,  x1: 2,  count: -1 },
            Intersect { x0: 3,  x1: 5,  count:  1 },
            Intersect { x0: 8,  x1: 10, count: -1 },
            Intersect { x0: 11, x1: 13, count:  1 },
        ];
        let scanner_stub = XPathScanner {
            eo: false,
            x_min: 0,
            x_max: 13,
            y_min: 0,
            y_max: 0,
            row_start: vec![0, 4],
            intersects: row,
        };
        let spans: Vec<_> = ScanIterator::new(&scanner_stub, 0).collect();
        assert_eq!(spans, vec![(0, 5), (8, 13)]);
    }

    #[test]
    fn eo_winding_two_spans() {
        // Even-odd: crossings toggle inside/outside at each entry.
        // All entries have count=1 (EO only looks at parity).
        // [0,2,1]: count 0→1 (odd=inside), x0=0.
        // [3,5,1]: count 1→2 (even=outside), x1=5. Emit (0,5). (spans don't
        //          cleanly separate here since x1 extends through the exit entry)
        //   Actually x1=max(2,5)=5 when consuming [3,5].
        // [8,10,1]: count 2→3 (odd=inside), x0=8.
        // [11,13,1]: count 3→4 (even=outside), x1=max(10,13)=13. Emit (8,13).
        use crate::scanner::Intersect;
        let row = vec![
            Intersect { x0: 0,  x1: 2,  count: 1 },
            Intersect { x0: 3,  x1: 5,  count: 1 },
            Intersect { x0: 8,  x1: 10, count: 1 },
            Intersect { x0: 11, x1: 13, count: 1 },
        ];
        let scanner_stub = XPathScanner {
            eo: true,
            x_min: 0,
            x_max: 13,
            y_min: 0,
            y_max: 0,
            row_start: vec![0, 4],
            intersects: row,
        };
        let spans: Vec<_> = ScanIterator::new(&scanner_stub, 0).collect();
        assert_eq!(spans, vec![(0, 5), (8, 13)]);
    }

    #[test]
    fn already_inside_residue() {
        // Simulate the "already inside" case: count=-1 at the start of next().
        // row has one entry [{5,7,+1}]. count should go to 0 after consuming it.
        // We test this by directly manipulating iterator state.
        use crate::scanner::Intersect;
        let row = vec![Intersect { x0: 5, x1: 7, count: 1 }];
        let scanner_stub = XPathScanner {
            eo: false,
            x_min: 5,
            x_max: 7,
            y_min: 0,
            y_max: 0,
            row_start: vec![0, 1],
            intersects: row,
        };
        let mut it = ScanIterator::new(&scanner_stub, 0);
        // Manually set count to -1 to simulate residue from a prior span.
        it.count = -1;
        // With count=-1 (inside), phase 1 immediately uses row[0] as the span
        // start without consuming it; phase 2 consumes it: x1=7, count=0.
        assert_eq!(it.next(), Some((5, 7)));
        assert_eq!(it.next(), None);
    }
}
