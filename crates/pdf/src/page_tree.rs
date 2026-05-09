//! Logarithmic page-tree descent.
//!
//! Unlike `Document::get_pages()` (which walks the whole tree), this module
//! descends from the catalog's `/Pages` root to a single requested page index
//! using each interior node's `/Count` to pick the correct `/Kids` branch.
//! O(log pages) lookups for documents whose page-tree depth is shallow
//! (typically 4–5 levels for 100k-page docs).

use std::collections::HashSet;

use crate::{
    dictionary::Dictionary,
    document::Document,
    error::PdfError,
    object::{Object, ObjectId},
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
//
// When the linearization Page Offset Hint Table parser ships, a fast
// path can short-circuit the descent here — `LinearizationHints::page_offset`
// will resolve idx → byte offset directly.  Today the parser stub returns
// None on every call, so wiring it in would just add overhead; the fast
// path lands when the real parser does.
pub fn descend_to_page_index(doc: &Document, idx: u32) -> Result<ObjectId, PdfError> {
    let pages_root = doc.pages_root_id()?;
    let total = doc.page_count_fast();
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
        let node_obj = doc.get_dict_arc(current)?;
        let node_dict = node_obj.as_dict().expect("get_dict_arc guarantees a dict");
        let node_type = node_dict
            .get(b"Type")
            .and_then(Object::as_name)
            .unwrap_or(b"");
        if node_type == b"Page" {
            return Ok(current);
        }

        let kids = resolve_kids(doc, current, node_dict)?;

        // Flat-tree fast path: when /Count exactly matches kids.len(), every
        // kid is implicitly a leaf with count 1.  Skip the per-kid /Type +
        // /Count probes and index directly.  This is the common case for
        // shallow PDFs (catalog → /Pages with all leaves directly attached).
        let parent_count = node_dict
            .get(b"Count")
            .and_then(Object::as_i64)
            .and_then(|n| u32::try_from(n).ok());
        if parent_count == Some(u32::try_from(kids.len()).unwrap_or(u32::MAX)) {
            // Bounds: idx < total ≤ Σ /Count along the descent path, and
            // here parent /Count == kids.len(), so `remaining` is a valid
            // index into kids[].
            let kid = kids
                .get(remaining as usize)
                .copied()
                .ok_or_else(|| PdfError::BadObject {
                    id: current.0,
                    detail: format!(
                        "flat /Pages: remaining={remaining} >= kids.len()={}",
                        kids.len()
                    ),
                })?;
            return Ok(kid);
        }

        // Mixed tree — sum /Count across siblings.
        let mut descended = false;
        for kid_ref in &kids {
            let kid_obj = doc.get_dict_arc(*kid_ref)?;
            let kid_dict = kid_obj.as_dict().expect("get_dict_arc guarantees a dict");
            let kid_count = kid_subtree_count(*kid_ref, kid_dict)?;
            if remaining < kid_count {
                // Found the kid containing the requested page.  If the kid
                // itself is a leaf, return immediately to avoid the
                // top-of-loop re-fetch on the next iteration.
                if kid_dict.get(b"Type").and_then(Object::as_name) == Some(b"Page") {
                    return Ok(*kid_ref);
                }
                current = *kid_ref;
                descended = true;
                break;
            }
            remaining -= kid_count;
        }
        if !descended {
            return Err(PdfError::BadObject {
                id: current.0,
                detail: format!(
                    "ran out of /Kids before reaching requested page (remaining={remaining})",
                ),
            });
        }
    }
    Err(PdfError::BadObject {
        id: current.0,
        detail: "page-tree descent exceeded depth limit".into(),
    })
}

/// Read a kid's subtree page count: 1 for leaves (Page), `/Count` for
/// interior /Pages nodes.  Returns `BadObject` if an interior node lacks
/// a valid `/Count`.
fn kid_subtree_count(kid_id: ObjectId, kid_dict: &Dictionary) -> Result<u32, PdfError> {
    let is_leaf = kid_dict.get(b"Type").and_then(Object::as_name) == Some(b"Page");
    if is_leaf {
        return Ok(1);
    }
    kid_dict
        .get(b"Count")
        .and_then(Object::as_i64)
        .and_then(|n| u32::try_from(n).ok())
        .ok_or_else(|| PdfError::BadObject {
            id: kid_id.0,
            detail: "interior page-tree node missing /Count".into(),
        })
}

/// `/Kids` may be either an inline array or an indirect reference to one
/// (corpus-04 ships the second form).  Resolve both into a `Vec<ObjectId>`.
///
/// Rejects malformed `/Kids` entries (non-references) hard rather than
/// silently filtering them out.  The descent indexes into the result by
/// page index, so a filtered list breaks the page-index invariant: a
/// `/Kids [3 0 R 4 0 R 99]` with `/Count 3` would silently misroute
/// page 2 to a `BadObject` error rather than the `99` slot the source
/// document intended.  Hard-fail surfaces the corruption to the caller.
fn resolve_kids(
    doc: &Document,
    node_id: ObjectId,
    node_dict: &Dictionary,
) -> Result<Vec<ObjectId>, PdfError> {
    let kids_obj = node_dict.get(b"Kids").ok_or(PdfError::MissingKey("Kids"))?;
    let arr_slice: &[Object] = match kids_obj {
        Object::Array(a) => a,
        Object::Reference(rid) => {
            let resolved = doc.get_object(*rid)?;
            return refs_from_array_or_err(resolved.as_ref(), node_id);
        }
        _ => {
            return Err(PdfError::BadObject {
                id: node_id.0,
                detail: "/Kids is neither an array nor a reference".into(),
            });
        }
    };
    refs_from_array(arr_slice, node_id)
}

fn refs_from_array_or_err(obj: &Object, node_id: ObjectId) -> Result<Vec<ObjectId>, PdfError> {
    match obj {
        Object::Array(a) => refs_from_array(a, node_id),
        _ => Err(PdfError::BadObject {
            id: node_id.0,
            detail: "/Kids reference does not resolve to an array".into(),
        }),
    }
}

fn refs_from_array(arr: &[Object], node_id: ObjectId) -> Result<Vec<ObjectId>, PdfError> {
    let refs: Vec<ObjectId> = arr.iter().filter_map(Object::as_reference).collect();
    if refs.len() != arr.len() {
        return Err(PdfError::BadObject {
            id: node_id.0,
            detail: format!(
                "/Kids contains {} non-reference entries (have {} of {} as refs)",
                arr.len() - refs.len(),
                refs.len(),
                arr.len()
            ),
        });
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

    /// `u32::MAX` is a representative pathological index — the function must
    /// reject it cleanly with `PageOutOfRange`, not panic on overflow during
    /// `remaining` arithmetic and not silently truncate.
    #[test]
    fn descend_rejects_u32_max_index() {
        let doc = Document::from_bytes_owned(two_page_pdf()).unwrap();
        let err = descend_to_page_index(&doc, u32::MAX).expect_err("u32::MAX must error");
        match err {
            PdfError::PageOutOfRange { page, total } => {
                assert_eq!(page, u32::MAX);
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

    /// Three-level page tree (catalog → root /Pages → 2 interior nodes →
    /// 4 leaves).  Forces the descent to actually iterate `kids` past
    /// index 0 and decrement `remaining` between siblings; the two-leaf
    /// flat tree above never exercises that branch.
    ///
    ///   1 = Catalog → /Pages 2 0 R
    ///   2 = /Pages /Kids [3 0 R 4 0 R] /Count 4
    ///   3 = /Pages /Kids [5 0 R 6 0 R] /Count 2 /Parent 2 0 R
    ///   4 = /Pages /Kids [7 0 R 8 0 R] /Count 2 /Parent 2 0 R
    ///   5–8 = /Page leaves
    fn three_level_pdf() -> Vec<u8> {
        // Byte offsets (computed exactly by summing segment lengths):
        //   header "%PDF-1.4\n"                                             9 bytes  → obj1 at 9
        //   obj1 (catalog)                                                 47 bytes  → obj2 at 56
        //   obj2 (root /Pages, /Count 4)                                   61 bytes  → obj3 at 117
        //   obj3 (interior /Pages, /Count 2, kids 5+6)                     75 bytes  → obj4 at 192
        //   obj4 (interior /Pages, /Count 2, kids 7+8)                     75 bytes  → obj5 at 267
        //   obj5..obj8 (leaves, Parent 3 / 3 / 4 / 4)                      69 bytes  → next at 336/405/474
        // xref at 543
        b"%PDF-1.4\n\
1 0 obj\n<</Type /Catalog /Pages 2 0 R>>\nendobj\n\
2 0 obj\n<</Type /Pages /Kids [3 0 R 4 0 R] /Count 4>>\nendobj\n\
3 0 obj\n<</Type /Pages /Kids [5 0 R 6 0 R] /Count 2 /Parent 2 0 R>>\nendobj\n\
4 0 obj\n<</Type /Pages /Kids [7 0 R 8 0 R] /Count 2 /Parent 2 0 R>>\nendobj\n\
5 0 obj\n<</Type /Page /Parent 3 0 R /MediaBox [0 0 612 792]>>\nendobj\n\
6 0 obj\n<</Type /Page /Parent 3 0 R /MediaBox [0 0 612 792]>>\nendobj\n\
7 0 obj\n<</Type /Page /Parent 4 0 R /MediaBox [0 0 612 792]>>\nendobj\n\
8 0 obj\n<</Type /Page /Parent 4 0 R /MediaBox [0 0 612 792]>>\nendobj\n\
xref\n0 9\n\
0000000000 65535 f\r\n\
0000000009 00000 n\r\n\
0000000056 00000 n\r\n\
0000000117 00000 n\r\n\
0000000192 00000 n\r\n\
0000000267 00000 n\r\n\
0000000336 00000 n\r\n\
0000000405 00000 n\r\n\
0000000474 00000 n\r\n\
trailer\n<</Size 9 /Root 1 0 R>>\n\
startxref\n543\n%%EOF"
            .to_vec()
    }

    /// Hand-built two-page PDF with a deliberately malformed /Kids array
    /// containing an inline integer (`99`) where a reference should be.
    /// `resolve_kids` must return `BadObject` rather than silently dropping
    /// the bad entry — silent filtering would misroute page 2 (the slot the
    /// integer occupies) to a misleading "ran out of /Kids" error.
    fn malformed_kids_pdf() -> Vec<u8> {
        // Byte offsets (computed exactly):
        //   header                                                              9 bytes  → obj1 at 9
        //   obj1 catalog                                                       47 bytes  → obj2 at 56
        //   obj2 /Pages /Kids [3 0 R 99 4 0 R] /Count 3                        64 bytes  → obj3 at 120
        //   obj3 leaf                                                          69 bytes  → obj4 at 189
        //   obj4 leaf                                                          69 bytes  → xref at 258
        b"%PDF-1.4\n\
1 0 obj\n<</Type /Catalog /Pages 2 0 R>>\nendobj\n\
2 0 obj\n<</Type /Pages /Kids [3 0 R 99 4 0 R] /Count 3>>\nendobj\n\
3 0 obj\n<</Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]>>\nendobj\n\
4 0 obj\n<</Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]>>\nendobj\n\
xref\n0 5\n\
0000000000 65535 f\r\n\
0000000009 00000 n\r\n\
0000000056 00000 n\r\n\
0000000120 00000 n\r\n\
0000000189 00000 n\r\n\
trailer\n<</Size 5 /Root 1 0 R>>\n\
startxref\n258\n%%EOF"
            .to_vec()
    }

    #[test]
    fn descend_rejects_non_reference_kids() {
        let doc = Document::from_bytes_owned(malformed_kids_pdf()).expect("open");
        let err = descend_to_page_index(&doc, 0).expect_err("malformed /Kids must error");
        match err {
            PdfError::BadObject { detail, .. } => {
                assert!(
                    detail.contains("non-reference"),
                    "expected non-reference error, got: {detail}"
                );
            }
            other => panic!("expected BadObject, got {other:?}"),
        }
    }

    #[test]
    fn descend_three_level_tree() {
        let doc = Document::from_bytes_owned(three_level_pdf()).expect("open");
        // page 0 → leaf object 5 (first kid of first interior)
        // page 1 → leaf object 6 (second kid of first interior — exercises
        //          the inner kid loop's increment past the first kid)
        // page 2 → leaf object 7 (first kid of second interior — exercises
        //          remaining decrement across an interior boundary)
        // page 3 → leaf object 8 (second kid of second interior)
        for (idx, expected_obj) in [(0u32, 5u32), (1, 6), (2, 7), (3, 8)] {
            let id = descend_to_page_index(&doc, idx)
                .unwrap_or_else(|e| panic!("descend idx={idx}: {e:?}"));
            assert_eq!(id.0, expected_obj, "idx={idx} mapped to wrong leaf");
        }
        // Also confirm /Pages /Count = 4 is read directly via fast path.
        assert_eq!(doc.page_count_fast(), 4);
    }
}
