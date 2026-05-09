//! Verify that opening a session does not eagerly walk the page tree.
//!
//! `open_session` reads `/Pages /Count` directly (memoised by the underlying
//! [`pdf::Document`]) and defers per-page `ObjectId` resolution to first
//! render via the [`pdf::Document::get_page`] descent.

use std::path::PathBuf;

#[test]
fn session_open_reports_correct_total_pages() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/corpus-05-academic-book.pdf");
    let session =
        pdf_raster::open_session(&path, &pdf_raster::SessionConfig::default()).expect("open");
    assert_eq!(session.total_pages(), 601);
}

#[test]
fn resolve_page_returns_consistent_object_ids() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/corpus-04-ebook-mixed.pdf");
    let session =
        pdf_raster::open_session(&path, &pdf_raster::SessionConfig::default()).expect("open");

    // Two calls for the same page must return the same object id.  The
    // underlying [`pdf::Document::get_page`] is a pure function of
    // `(doc, idx)` so the result is deterministic across calls.
    let id1 = session.resolve_page(1).expect("page 1 first");
    let id2 = session.resolve_page(1).expect("page 1 second");
    assert_eq!(id1, id2);

    // Out-of-range fails fast with a 1-based PageOutOfRange — the caller
    // does not have to translate from the descender's 0-based variant.
    let err = session
        .resolve_page(session.total_pages() + 1)
        .expect_err("must error past total_pages");
    match err {
        pdf_raster::RasterError::PageOutOfRange { page, total } => {
            assert_eq!(page, session.total_pages() + 1);
            assert_eq!(total, session.total_pages());
        }
        other => panic!("expected PageOutOfRange, got {other:?}"),
    }
}
