//! Verify that opening a session does not eagerly walk the entire page tree.
//!
//! `open_session` holds only the document handle and a `total_pages` count
//! immediately after construction.  Per-page `ObjectId` resolution is deferred
//! to first render via `RwLock<HashMap<u32, ObjectId>>`.

use std::path::PathBuf;

#[test]
fn session_open_does_not_eagerly_populate_page_cache() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/corpus-05-academic-book.pdf");
    let session =
        pdf_raster::open_session(&path, &pdf_raster::SessionConfig::default()).expect("open");
    assert_eq!(
        session.cached_page_count(),
        0,
        "session should not eagerly populate the page-id cache"
    );
    assert_eq!(session.total_pages(), 601);
}

#[test]
fn resolve_page_populates_cache_on_demand() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/corpus-04-ebook-mixed.pdf");
    let session =
        pdf_raster::open_session(&path, &pdf_raster::SessionConfig::default()).expect("open");
    assert_eq!(session.cached_page_count(), 0);

    // Render a single page; cache should hold exactly one entry afterwards.
    let _ = pdf_raster::render_page_rgb(&session, 1, 1.0).expect("render");
    assert_eq!(session.cached_page_count(), 1);
}
