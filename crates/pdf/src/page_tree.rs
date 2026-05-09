//! Logarithmic page-tree descent.
//!
//! Unlike `Document::get_pages()` (which walks the whole tree), this module
//! descends from the catalog's `/Pages` root to a single requested page index
//! using each interior node's `/Count` to pick the correct `/Kids` branch.
//! O(log pages) lookups for documents whose page-tree depth is shallow
//! (typically 4–5 levels for 100k-page docs).

use std::collections::HashSet;

use crate::{
    dictionary::Dictionary, document::Document, error::PdfError, object::Object, object::ObjectId,
};

/// Maximum page-tree depth.  Bounds descent against pathological / cyclic PDFs.
const MAX_DEPTH: usize = 64;

/// Resolve a 0-based page index to its `ObjectId`, walking only the
/// root-to-leaf path.
///
/// Returns `PdfError::PageOutOfRange { page, total }` if `idx` is
/// outside `[0, total)` where `total` is the catalog's `/Pages /Count`
/// (or the eager page count if `/Count` is missing or invalid).
///
/// Cycle protection: tracks visited object numbers in a `HashSet<u32>`.
/// Depth protection: bounded to `MAX_DEPTH` levels.
pub fn descend_to_page_index(doc: &Document, idx: u32) -> Result<ObjectId, PdfError> {
    let catalog = doc.catalog()?;
    let pages_root = catalog
        .get(b"Pages")
        .and_then(Object::as_reference)
        .ok_or(PdfError::MissingKey("Pages"))?;

    let total = page_count_from_root(doc, pages_root)?;
    if idx >= total {
        return Err(PdfError::PageOutOfRange { page: idx, total });
    }

    let mut visited: HashSet<u32> = HashSet::new();
    let mut current = pages_root;
    let mut remaining = idx;

    for _ in 0..MAX_DEPTH {
        if !visited.insert(current.0) {
            return Err(PdfError::BadObject {
                id: current.0,
                detail: "cyclic /Kids reference during descent".into(),
            });
        }
        let dict = doc.get_dict(current)?;
        let node_type = dict.get(b"Type").and_then(Object::as_name).unwrap_or(b"");
        if node_type == b"Page" {
            return Ok(current);
        }

        // Interior /Pages node — pick the kid whose subtree contains `remaining`.
        let kids = resolve_kids(doc, current, &dict)?;
        let mut descended = false;
        for kid_ref in kids {
            let kid_dict = doc.get_dict(kid_ref)?;
            let kid_type = kid_dict
                .get(b"Type")
                .and_then(Object::as_name)
                .unwrap_or(b"");
            let kid_count: u32 = if kid_type == b"Page" {
                1
            } else {
                kid_dict
                    .get(b"Count")
                    .and_then(Object::as_i64)
                    .and_then(|n| u32::try_from(n).ok())
                    .ok_or_else(|| PdfError::BadObject {
                        id: kid_ref.0,
                        detail: "interior page-tree node missing /Count".into(),
                    })?
            };
            if remaining < kid_count {
                if kid_type == b"Page" {
                    return Ok(kid_ref);
                }
                current = kid_ref;
                descended = true;
                break;
            }
            remaining -= kid_count;
        }
        if !descended {
            return Err(PdfError::BadObject {
                id: current.0,
                detail: "ran out of /Kids before reaching requested page".into(),
            });
        }
    }
    Err(PdfError::BadObject {
        id: current.0,
        detail: "page-tree descent exceeded depth limit".into(),
    })
}

/// Read `/Pages /Count` from the root.  Falls back to a full eager walk
/// if the value is missing, negative, or out of range — malformed PDFs in
/// the wild ship like this and silently picking the wrong page would be
/// worse than the linear scan.
fn page_count_from_root(doc: &Document, pages_root: ObjectId) -> Result<u32, PdfError> {
    let dict = doc.get_dict(pages_root)?;
    if let Some(n) = dict.get(b"Count").and_then(Object::as_i64)
        && (0..=i64::from(u32::MAX)).contains(&n)
    {
        return Ok(n as u32);
    }
    log::warn!("page_count: /Pages /Count missing or invalid; falling back to eager count");
    Ok(doc.get_pages().count() as u32)
}

/// `/Kids` may be either an inline array or an indirect reference to one
/// (corpus-04 ships the second form).  Resolve both into a `Vec<ObjectId>`.
fn resolve_kids(
    doc: &Document,
    node_id: ObjectId,
    node_dict: &Dictionary,
) -> Result<Vec<ObjectId>, PdfError> {
    let kids_obj = node_dict.get(b"Kids").ok_or(PdfError::MissingKey("Kids"))?;
    let arr = match kids_obj {
        Object::Array(a) => a.clone(),
        Object::Reference(rid) => {
            let resolved = doc.get_object(*rid)?;
            match resolved.as_ref() {
                Object::Array(a) => a.clone(),
                _ => {
                    return Err(PdfError::BadObject {
                        id: rid.0,
                        detail: "/Kids reference does not resolve to an array".into(),
                    });
                }
            }
        }
        _ => {
            return Err(PdfError::BadObject {
                id: node_id.0,
                detail: "/Kids is neither an array nor a reference".into(),
            });
        }
    };
    let refs: Vec<ObjectId> = arr.iter().filter_map(Object::as_reference).collect();
    if refs.len() != arr.len() {
        log::warn!(
            "resolve_kids: {}/{} /Kids entries were not indirect references; non-refs skipped",
            arr.len() - refs.len(),
            arr.len()
        );
    }
    Ok(refs)
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

    /// Verify descend_to_page_index agrees with the eager get_pages() iterator
    /// across the same fixture.
    #[test]
    fn descend_agrees_with_eager_walk() {
        let doc = Document::from_bytes_owned(two_page_pdf()).unwrap();
        let eager: Vec<ObjectId> = doc.get_pages().map(|(_, id)| id).collect();
        for (idx, expected_id) in eager.iter().enumerate() {
            let got = descend_to_page_index(&doc, idx as u32).expect("descend");
            assert_eq!(got, *expected_id, "mismatch at idx {idx}");
        }
    }
}
