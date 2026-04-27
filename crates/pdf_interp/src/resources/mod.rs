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

pub(crate) mod dict_ext;
pub mod font;
pub mod image;

use lopdf::{Document, Object, ObjectId};

pub use font::{FontDescriptor, PdfFontKind, resolve_font};
pub use image::{ImageColorSpace, ImageDescriptor, resolve_image};

/// Parsed Form `XObject` — content bytes and the CTM matrix to apply.
///
/// The `resources_id` is the `ObjectId` of the form stream itself; callers
/// construct a child [`PageResources`] from it via [`PageResources::for_form`].
pub struct FormXObject {
    /// Decoded content bytes of the form's content stream.
    pub content: Vec<u8>,
    /// Optional form Matrix `[a b c d e f]` (default: identity).
    pub matrix: [f64; 6],
    /// The object ID of the form stream, used to build a child `PageResources`.
    pub resources_id: ObjectId,
    /// True if the form has its own `Resources` dict; false means it inherits
    /// from the parent context (the caller should keep the parent resources).
    pub has_own_resources: bool,
}

/// Thin accessor wrapping a `(Document, resource_context_id)` pair.
///
/// `resource_context_id` is the `ObjectId` of the object (page or form stream)
/// whose `Resources` dict is used to resolve fonts and `XObjects`.  For a top-level
/// page render this is the page object; for a form `XObject` it is the form stream.
pub struct PageResources<'doc> {
    doc: &'doc Document,
    /// Object whose `Resources` key is consulted for font/XObject lookups.
    resource_context_id: ObjectId,
}

impl<'doc> PageResources<'doc> {
    /// Construct a [`PageResources`] for the given page.
    #[must_use]
    pub const fn new(doc: &'doc Document, page_id: ObjectId) -> Self {
        Self {
            doc,
            resource_context_id: page_id,
        }
    }

    /// Construct a child [`PageResources`] scoped to the given form's resources.
    ///
    /// If `form.has_own_resources` is false the form inherits the parent context;
    /// use `self` (the parent) directly instead of calling this.
    #[must_use]
    pub const fn for_form(&self, form: &FormXObject) -> Self {
        if form.has_own_resources {
            Self {
                doc: self.doc,
                resource_context_id: form.resources_id,
            }
        } else {
            Self {
                doc: self.doc,
                resource_context_id: self.resource_context_id,
            }
        }
    }

    /// The underlying `lopdf` document (read-only).
    #[must_use]
    pub const fn doc(&self) -> &'doc Document {
        self.doc
    }

    /// Resolve the font dictionary for the named resource (e.g. `b"F1"`).
    ///
    /// Returns `None` if the resource name is not present in the resource dict.
    #[must_use]
    pub fn font_dict(&self, name: &[u8]) -> Option<font::FontDescriptor> {
        let fonts = self.doc.get_page_fonts(self.resource_context_id).ok()?;
        let dict = fonts.get(name)?;
        Some(font::resolve_font(self.doc, dict))
    }

    /// Decode the named image `XObject` from the resource dictionary.
    ///
    /// Returns `None` if the name is absent, the object is not an image, or
    /// decoding fails.
    #[must_use]
    pub fn image(&self, name: &[u8]) -> Option<image::ImageDescriptor> {
        let page_dict = self.doc.get_dictionary(self.resource_context_id).ok()?;
        image::resolve_image(self.doc, page_dict, name)
    }

    /// Resolve the named `XObject` and return it as a [`FormXObject`] if its
    /// subtype is `Form`.  Returns `None` for Image `XObjects`, missing names,
    /// or unreadable streams.
    #[must_use]
    pub fn form_xobject(&self, name: &[u8]) -> Option<FormXObject> {
        let ctx_dict = self.doc.get_dictionary(self.resource_context_id).ok()?;
        let res = image::resolve_dict(self.doc, ctx_dict.get(b"Resources").ok()?)?;
        let xobj_dict = image::resolve_dict(self.doc, res.get(b"XObject").ok()?)?;
        let stream_id = match xobj_dict.get(name).ok()? {
            Object::Reference(id) => *id,
            _ => return None,
        };

        let obj = self.doc.get_object(stream_id).ok()?;
        let stream = obj.as_stream().ok()?;

        // Must be a Form subtype.
        if stream.dict.get(b"Subtype").ok()?.as_name().ok()? != b"Form" {
            return None;
        }

        let content = stream.decompressed_content().ok()?;

        // Optional Matrix — defaults to identity if absent or malformed.
        let matrix = read_matrix(&stream.dict).unwrap_or([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);

        let has_own_resources = stream.dict.get(b"Resources").is_ok();

        Some(FormXObject {
            content,
            matrix,
            resources_id: stream_id,
            has_own_resources,
        })
    }
}

/// Read a 6-element `Matrix` array from a dictionary, returning `None` if the
/// key is absent or has fewer than 6 numeric entries.
fn read_matrix(dict: &lopdf::Dictionary) -> Option<[f64; 6]> {
    let arr = dict.get(b"Matrix").ok()?.as_array().ok()?;
    if arr.len() < 6 {
        return None;
    }
    let mut m = [0.0f64; 6];
    for (i, obj) in arr.iter().take(6).enumerate() {
        m[i] = match obj {
            Object::Real(r) => f64::from(*r),
            #[expect(clippy::cast_precision_loss, reason = "PDF matrix values are small integers; precision loss is negligible")]
            Object::Integer(n) => *n as f64,
            _ => return None,
        };
    }
    Some(m)
}
