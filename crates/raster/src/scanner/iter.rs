//! Coalescing span iterator over a scanner's intersection row.
//!
//! [`ScanIterator`] mirrors `SplashXPathScanIterator::getNextSpan` from
//! `splash/SplashXPathScanner.cc`. It merges adjacent and overlapping
//! intersections into contiguous `(x0, x1)` spans (inclusive) and handles
//! both even-odd and non-zero winding fill rules via the `eo_mask`.

use super::{Intersect, XPathScanner};

/// An iterator that yields `(x0, x1)` inclusive pixel spans for one scanline.
///
/// Spans are emitted in left-to-right order with no overlap.
pub struct ScanIterator<'a> {
    row: &'a [Intersect],
    idx: usize,
    count: i32,
    /// `!0` for NZ winding (count ≠ 0 → inside); `1` for EO (count & 1 → inside).
    eo_mask: i32,
}

impl<'a> ScanIterator<'a> {
    /// Create an iterator for scanline `y` of `scanner`.
    ///
    /// If `y` is outside the scanner's range, the iterator is immediately
    /// exhausted (matching C++ behaviour where the empty `allIntersections[0]`
    /// row is returned with `interIdx = line.size()`).
    pub fn new(scanner: &'a XPathScanner, y: i32) -> Self {
        let row = scanner.row(y);
        let eo_mask = if scanner.eo { 1 } else { !0 };
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

    fn next(&mut self) -> Option<(i32, i32)> {
        // Advance past entries that leave us outside the path.
        loop {
            if self.idx >= self.row.len() {
                return None;
            }
            if (self.count & self.eo_mask) != 0 {
                break;
            }
            self.count = self.count.wrapping_add(self.row[self.idx].count);
            if (self.count & self.eo_mask) != 0 {
                break;
            }
            self.idx += 1;
        }
        if self.idx >= self.row.len() {
            return None;
        }

        // We are now at the start of a covered span.
        let x0 = self.row[self.idx].x0;
        let mut x1 = self.row[self.idx].x1;
        self.count = self.count.wrapping_add(self.row[self.idx].count);
        self.idx += 1;

        // Extend the span while still inside the path.
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
        let spans: Vec<_> = ScanIterator::new(&scanner, 100).collect();
        assert!(spans.is_empty());
    }

    #[test]
    fn coalescing_adjacent() {
        // Two intersections: [0,3] count=-1 then [4,7] count=+1.
        // Under NZ winding, count starts at 0 → transitions to -1 (inside) at
        // first intersection, then back to 0 (outside) at second. Both
        // intersections are traversed while inside, so the emitted span is [0, 7].
        use crate::scanner::Intersect;
        let row = vec![
            Intersect {
                x0: 0,
                x1: 3,
                count: -1,
            },
            Intersect {
                x0: 4,
                x1: 7,
                count: 1,
            },
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
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0], (0, 7));
    }
}
