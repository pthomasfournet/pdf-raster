//! PDF resource resolution — fonts, colour spaces, `XObject`s, shadings.
//!
//! The [`PageResources`] struct holds a reference to the lopdf [`Document`]
//! and the current page's [`ObjectId`], and provides lazy-loading helpers
//! for each resource category.
//!
//! # Design
//!
//! lopdf's `Document` is immutable after load; resource dicts are accessed
//! via `get_page_fonts()` and friends, which traverse the page tree and
//! dereference indirect objects.  All caching lives in the caller
//! ([`super::renderer::font_cache`]) rather than here, so this module
//! stays stateless and easy to test.

pub mod font;

use lopdf::{Document, ObjectId};

pub use font::{FontDescriptor, PdfFontKind, resolve_font};

/// Thin accessor wrapping a `(Document, page_id)` pair.
///
/// Passed by reference to rendering helpers that need to resolve resources.
pub struct PageResources<'doc> {
    /// The lopdf document being rendered.
    pub doc: &'doc Document,
    /// The page object ID within the document.
    pub page_id: ObjectId,
}

impl<'doc> PageResources<'doc> {
    /// Construct a [`PageResources`] for the given page.
    #[must_use]
    pub const fn new(doc: &'doc Document, page_id: ObjectId) -> Self {
        Self { doc, page_id }
    }

    /// Resolve the font dictionary for the named resource (e.g. `b"F1"`).
    ///
    /// Returns `None` if the resource name is not present in the page's Font
    /// resource dict.
    #[must_use]
    pub fn font_dict(&self, name: &[u8]) -> Option<font::FontDescriptor> {
        let fonts = self.doc.get_page_fonts(self.page_id).ok()?;
        let dict = fonts.get(name)?;
        Some(font::resolve_font(self.doc, dict))
    }
}
