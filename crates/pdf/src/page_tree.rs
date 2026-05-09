//! Logarithmic page-tree descent.
//!
//! Unlike `Document::get_pages()` (which walks the whole tree), this module
//! descends from the catalog's `/Pages` root to a single requested page index
//! using each interior node's `/Count` to pick the correct `/Kids` branch.
//! O(log pages) lookups for documents whose page-tree depth is shallow
//! (typically 4–5 levels for 100k-page docs).

use crate::{document::Document, error::PdfError, object::ObjectId};

/// Resolve a 0-based page index to its `ObjectId`, walking only the
/// root-to-leaf path.
///
/// Returns `PdfError::PageOutOfRange { page, total }` if `idx` is
/// outside `[0, total)` where `total` is the catalog's `/Pages /Count`.
///
/// Falls back to a linear scan via `Document::get_pages()` if any interior
/// node is missing `/Count` or has `/Kids` shorter than `/Count` claims —
/// real-world malformed PDFs do this, and silently rendering the wrong
/// page would be the worst possible failure mode.
pub fn descend_to_page_index(_doc: &Document, _idx: u32) -> Result<ObjectId, PdfError> {
    // Implementation pending; see ROADMAP.md.
    todo!("descend_to_page_index: logarithmic walker not yet implemented")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two-page PDF.  Verifies that descend_to_page_index returns the
    /// correct page object for both indices.  The eager `get_pages()` and
    /// the lazy `descend_to_page_index` must agree.
    fn two_page_pdf() -> Vec<u8> {
        // %PDF-1.4\n = 9 bytes  → obj1 at 9
        // "1 0 obj\n<</Type /Catalog /Pages 2 0 R>>\nendobj\n" = 47 bytes → obj2 at 56
        // "2 0 obj\n<</Type /Pages /Kids [3 0 R 4 0 R] /Count 2>>\nendobj\n" = 61 bytes → obj3 at 117
        // "3 0 obj\n<</Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]>>\nendobj\n" = 69 bytes → obj4 at 186
        // "4 0 obj\n<</Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]>>\nendobj\n" = 69 bytes → xref at 255
        b"%PDF-1.4\n\
1 0 obj\n<</Type /Catalog /Pages 2 0 R>>\nendobj\n\
2 0 obj\n<</Type /Pages /Kids [3 0 R 4 0 R] /Count 2>>\nendobj\n\
3 0 obj\n<</Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]>>\nendobj\n\
4 0 obj\n<</Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]>>\nendobj\n\
xref\n0 5\n\
0000000000 65535 f\r\n\
0000000009 00000 n\r\n\
0000000056 00000 n\r\n\
0000000117 00000 n\r\n\
0000000186 00000 n\r\n\
trailer\n<</Size 5 /Root 1 0 R>>\n\
startxref\n255\n%%EOF"
            .to_vec()
    }

    #[test]
    fn descend_finds_first_page() {
        let doc = Document::from_bytes_owned(two_page_pdf()).unwrap();
        let id = descend_to_page_index(&doc, 0).expect("page 0");
        assert_eq!(id.0, 3);
    }

    #[test]
    fn descend_finds_second_page() {
        let doc = Document::from_bytes_owned(two_page_pdf()).unwrap();
        let id = descend_to_page_index(&doc, 1).expect("page 1");
        assert_eq!(id.0, 4);
    }

    #[test]
    fn descend_rejects_out_of_range() {
        let doc = Document::from_bytes_owned(two_page_pdf()).unwrap();
        let err = descend_to_page_index(&doc, 2).expect_err("page 2 must error");
        match err {
            PdfError::PageOutOfRange { page, total } => {
                assert_eq!(page, 2);
                assert_eq!(total, 2);
            }
            other => panic!("expected PageOutOfRange, got {other:?}"),
        }
    }
}
