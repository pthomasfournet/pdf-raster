//! PDF content stream interpreter.
//!
//! # Architecture
//!
//! ```text
//! lopdf::Document  →  parse_page()  →  Vec<Operator>  →  (renderer — next phase)
//! ```
//!
//! The [`content`] module handles tokenization and operator decoding.
//! Rasterization will be wired in subsequent commits once the graphics-state
//! machine and resource resolver are in place.

pub mod content;
pub mod prescan;
pub mod renderer;
pub mod resources;

pub use prescan::prescan_page;

/// Entry points for fuzz targets — forwards to codec wrappers in `image::fuzz_entry`.
///
/// Only compiled when `--cfg fuzzing` is set (cargo-fuzz sets this automatically).
/// Not part of the public API.
#[cfg(fuzzing)]
#[doc(hidden)]
pub mod fuzz_helpers {
    pub use crate::resources::image::fuzz_entry::{decode_ccitt, decode_jbig2};
}

use std::path::Path;

use lopdf::Document;

/// Errors that can occur during PDF loading or content stream interpretation.
#[derive(Debug)]
pub enum InterpError {
    /// lopdf failed to load or parse the document.
    Pdf(lopdf::Error),
    /// The requested page number is outside the document's page range.
    PageOutOfRange {
        /// The requested page (1-based).
        page: u32,
        /// Total number of pages in the document.
        total: u32,
    },
    /// A required resource (font, image, colour space, …) could not be resolved.
    MissingResource(String),
    /// The document contains JavaScript.  We refuse to open it.
    ///
    /// The location field names the PDF construct that triggered the check
    /// (e.g. `"/OpenAction"`, `"/Names/JavaScript"`).
    JavaScript {
        /// Which entry point in the PDF triggered the rejection.
        location: &'static str,
    },
    /// A Page dictionary entry has a value that is structurally valid PDF but
    /// outside the range permitted by the spec or our safety limits.
    InvalidPageGeometry(String),
    /// `FreeType` library initialisation failed (library not available or internal error).
    FontInit(String),
}

impl std::fmt::Display for InterpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pdf(e) => write!(f, "PDF error: {e}"),
            Self::PageOutOfRange { page, total } => {
                write!(
                    f,
                    "page {page} is out of range (document has {total} pages)"
                )
            }
            Self::MissingResource(name) => write!(f, "missing PDF resource: {name}"),
            Self::JavaScript { location } => {
                write!(f, "document contains JavaScript ({location}) — refused")
            }
            Self::InvalidPageGeometry(msg) => write!(f, "invalid page geometry: {msg}"),
            Self::FontInit(msg) => write!(f, "FreeType initialisation failed: {msg}"),
        }
    }
}

impl std::error::Error for InterpError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Pdf(e) => Some(e),
            Self::FontInit(_)
            | Self::PageOutOfRange { .. }
            | Self::MissingResource(_)
            | Self::JavaScript { .. }
            | Self::InvalidPageGeometry(_) => None,
        }
    }
}

impl From<lopdf::Error> for InterpError {
    fn from(e: lopdf::Error) -> Self {
        Self::Pdf(e)
    }
}

/// Open a PDF document from a file path.
///
/// Returns [`InterpError::JavaScript`] immediately if the document contains
/// any JavaScript entry point — we treat JS in PDFs as an attack surface and
/// refuse to proceed rather than silently ignoring it.
///
/// Checked locations (PDF 2.0 §12.6):
/// - `/Catalog/OpenAction` with `/S /JavaScript`
/// - `/Catalog/AA` (catalog-level additional actions)
/// - `/Catalog/Names/JavaScript` (document JS name tree)
/// - `/Catalog/AcroForm/AA` (`AcroForm` additional actions)
///
/// # Errors
/// - [`InterpError::Pdf`] if the file cannot be read or is not a valid PDF.
/// - [`InterpError::JavaScript`] if any JavaScript entry point is detected.
pub fn open(path: impl AsRef<Path>) -> Result<Document, InterpError> {
    let doc = Document::load(path)?;
    reject_javascript(&doc)?;
    Ok(doc)
}

/// Scan the document catalog for JavaScript entry points and return
/// [`InterpError::JavaScript`] for the first one found.
///
/// No JS content is read, parsed, or evaluated — the check is purely
/// structural (dict key presence / subtype name).
fn reject_javascript(doc: &Document) -> Result<(), InterpError> {
    let Ok(catalog) = doc.catalog() else {
        return Ok(());
    };

    // 1. OpenAction with /S /JavaScript
    if let Ok(action) = catalog.get(b"OpenAction")
        && action_is_js(doc, action)
    {
        return Err(InterpError::JavaScript {
            location: "/OpenAction",
        });
    }

    // 2. Catalog-level additional actions (/AA)
    if catalog.get(b"AA").is_ok() {
        return Err(InterpError::JavaScript {
            location: "/AA (catalog additional actions)",
        });
    }

    // 3. Document JS name tree (/Names/JavaScript)
    if let Ok(names_obj) = catalog.get(b"Names")
        && let Some(names_dict) = resources::resolve_dict(doc, names_obj)
        && names_dict.get(b"JavaScript").is_ok()
    {
        return Err(InterpError::JavaScript {
            location: "/Names/JavaScript",
        });
    }

    // 4. AcroForm additional actions
    if let Ok(acroform_obj) = catalog.get(b"AcroForm")
        && let Some(acroform) = resources::resolve_dict(doc, acroform_obj)
        && acroform.get(b"AA").is_ok()
    {
        return Err(InterpError::JavaScript {
            location: "/AcroForm/AA",
        });
    }

    Ok(())
}

/// Return `true` if `obj` (or the dict it resolves to) has `/S /JavaScript`.
fn action_is_js(doc: &Document, obj: &lopdf::Object) -> bool {
    resources::resolve_dict(doc, obj)
        .and_then(|d| d.get(b"S").ok())
        .and_then(|s| s.as_name().ok())
        .is_some_and(|name| name == b"JavaScript")
}

/// Return the number of pages in `doc`.
///
/// Saturates at [`u32::MAX`] for pathological documents (>4 billion pages).
#[must_use]
pub fn page_count(doc: &Document) -> u32 {
    u32::try_from(doc.get_pages().len()).unwrap_or(u32::MAX)
}

/// Resolve a 1-based page number to its lopdf object ID.
///
/// Calls `doc.get_pages()` once and derives the total from the map, avoiding a
/// redundant second call when both the count and the ID are needed.
///
/// # Errors
/// [`InterpError::PageOutOfRange`] if `page_num` is 0 or exceeds the document.
pub(crate) fn resolve_page_id(
    doc: &Document,
    page_num: u32,
) -> Result<lopdf::ObjectId, InterpError> {
    let pages = doc.get_pages();
    let total = u32::try_from(pages.len()).unwrap_or(u32::MAX);
    if page_num == 0 || page_num > total {
        return Err(InterpError::PageOutOfRange {
            page: page_num,
            total,
        });
    }
    pages
        .get(&page_num)
        .copied()
        .ok_or(InterpError::PageOutOfRange {
            page: page_num,
            total,
        })
}

/// Page geometry: visible dimensions in PDF points, rotation, and user-space scale.
///
/// `width_pts` and `height_pts` are already adjusted for rotation — they describe the
/// output bitmap dimensions, not the raw `MediaBox`.  For example a landscape page with
/// `/Rotate 270` has `width_pts > height_pts` even though its `MediaBox` may be portrait.
///
/// Both dimensions are also already scaled by `user_unit` (PDF 1.6+ `UserUnit` key).
/// Multiply the render DPI by `user_unit` to get the true physical resolution of the
/// rendered bitmap (i.e. the value to pass to `tesseract::set_source_resolution`).
#[derive(Debug, Clone, Copy)]
pub struct PageGeometry {
    /// Width of the rendered output in PDF points (after rotation and `UserUnit` scaling).
    pub width_pts: f64,
    /// Height of the rendered output in PDF points (after rotation and `UserUnit` scaling).
    pub height_pts: f64,
    /// Clockwise rotation in degrees; always one of 0, 90, 180, 270.
    pub rotate_cw: u16,
    /// `UserUnit` scale factor from the Page dictionary (PDF 1.6+, ISO 32000-2 §14.11.2).
    ///
    /// Specifies the size of one default user-space unit in 1/72-inch increments.
    /// `1.0` for the vast majority of documents.  Validated to `[0.1, 10.0]`.
    pub user_unit: f64,
}

/// Return the geometry for page `page_num` (1-based).
///
/// Reads the page's `CropBox` (the display region, per ISO 32000-2 §14.11.2), falling back
/// to `MediaBox` when absent. Applies the `/Rotate` entry (multiples of 90° CW). Dimensions
/// in the returned struct are already swapped for 90°/270° rotations and scaled by `UserUnit`
/// so callers can use `width_pts × height_pts` directly as output point dimensions.
///
/// Falls back to US Letter (612 × 792 pt, no rotation, `UserUnit` 1.0) when neither box
/// can be read.
///
/// # Errors
///
/// - [`InterpError::PageOutOfRange`] if `page_num` is 0 or exceeds the document page count.
/// - [`InterpError::InvalidPageGeometry`] if `UserUnit` is present but outside `[0.1, 10.0]`.
pub fn page_size_pts(doc: &Document, page_num: u32) -> Result<PageGeometry, InterpError> {
    let page_id = resolve_page_id(doc, page_num)?;

    let fallback = PageGeometry {
        width_pts: 612.0,
        height_pts: 792.0,
        rotate_cw: 0,
        user_unit: 1.0,
    };

    let Ok(dict) = doc.get_dictionary(page_id) else {
        return Ok(fallback);
    };

    #[expect(
        clippy::cast_precision_loss,
        reason = "PDF page sizes are at most a few thousand points; i64 → f64 precision loss is negligible"
    )]
    let to_f64 = |o: &lopdf::Object| match o {
        lopdf::Object::Real(r) => f64::from(*r),
        lopdf::Object::Integer(i) => *i as f64,
        _ => 0.0,
    };

    let box_wh = |key: &[u8]| -> Option<(f64, f64)> {
        let arr = match dict.get(key) {
            Ok(lopdf::Object::Array(a)) if a.len() == 4 => a,
            _ => return None,
        };
        let (x0, y0, x1, y1) = (
            to_f64(&arr[0]),
            to_f64(&arr[1]),
            to_f64(&arr[2]),
            to_f64(&arr[3]),
        );
        let w = (x1 - x0).abs();
        let h = (y1 - y0).abs();
        if w > 0.0 && h > 0.0 {
            Some((w, h))
        } else {
            None
        }
    };

    // CropBox is the display box — the region actually shown to the viewer.
    // Falls back to MediaBox when CropBox is absent (spec: CropBox defaults to MediaBox).
    let Some((w_pts, h_pts)) = box_wh(b"CropBox").or_else(|| box_wh(b"MediaBox")) else {
        return Ok(fallback);
    };

    // /Rotate is a multiple of 90 (CW). Normalise to 0/90/180/270.
    // rem_euclid(360) is always 0..=359, which fits u16.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "rem_euclid(360) is always 0..=359, which fits u16"
    )]
    let rotate_cw: u16 = match dict.get(b"Rotate") {
        Ok(lopdf::Object::Integer(n)) => (*n).rem_euclid(360) as u16 / 90 * 90,
        _ => 0,
    };

    // UserUnit (PDF 1.6+, ISO 32000-2 §14.11.2): scales one default user-space unit
    // from 1/72 inch to UserUnit/72 inches.  Absent → 1.0.  Must be a finite positive
    // number.  We reject values outside [0.1, 10.0]: below 0.1 the page would be
    // unreasonably small at any practical DPI; above 10.0 the pixel dimensions would
    // exceed MAX_PX_DIMENSION at even modest DPI settings.  NaN/Inf in a Real object
    // are rejected explicitly because the range check (NaN comparisons are always false)
    // would otherwise silently pass them through.
    let user_unit: f64 = match dict.get(b"UserUnit") {
        Err(_) => 1.0, // absent — use default
        Ok(obj) => {
            match obj {
                lopdf::Object::Real(_) | lopdf::Object::Integer(_) => {}
                other => {
                    return Err(InterpError::InvalidPageGeometry(format!(
                        "UserUnit on page {page_num} is not a number (got {})",
                        other.enum_variant()
                    )));
                }
            }
            let v = to_f64(obj);
            if !v.is_finite() || !(0.1..=10.0).contains(&v) {
                return Err(InterpError::InvalidPageGeometry(format!(
                    "UserUnit {v} on page {page_num} is outside the valid range [0.1, 10.0]"
                )));
            }
            v
        }
    };

    // Swap dimensions for 90°/270° so the caller gets output-bitmap dimensions directly.
    // Scale both dimensions by UserUnit so downstream only needs to apply dpi/72.
    let (width_pts, height_pts) = if rotate_cw == 90 || rotate_cw == 270 {
        (h_pts * user_unit, w_pts * user_unit)
    } else {
        (w_pts * user_unit, h_pts * user_unit)
    };

    Ok(PageGeometry {
        width_pts,
        height_pts,
        rotate_cw,
        user_unit,
    })
}

/// Parse the content stream for page `page_num` (1-based) and return the
/// decoded operator sequence.
///
/// # Errors
/// Returns [`InterpError::PageOutOfRange`] if `page_num` is 0 or exceeds the
/// document page count, or [`InterpError::Pdf`] if the content stream cannot
/// be read.
pub fn parse_page(doc: &Document, page_num: u32) -> Result<Vec<content::Operator>, InterpError> {
    let page_id = resolve_page_id(doc, page_num)?;
    let content_bytes = doc.get_page_content(page_id)?;
    Ok(content::parse(&content_bytes))
}

#[cfg(test)]
mod js_guard_tests {
    use lopdf::{Dictionary, Document, Object};

    use super::{InterpError, reject_javascript};

    fn make_doc_with_open_action(subtype: &str) -> Document {
        let mut doc = Document::with_version("1.4");
        let action =
            Dictionary::from_iter([(b"S".to_vec(), Object::Name(subtype.as_bytes().to_vec()))]);
        let pages_id = doc.add_object(Dictionary::from_iter([
            (b"Type".to_vec(), Object::Name(b"Pages".to_vec())),
            (b"Kids".to_vec(), Object::Array(vec![])),
            (b"Count".to_vec(), Object::Integer(0)),
        ]));
        let catalog_id = doc.add_object(Dictionary::from_iter([
            (b"Type".to_vec(), Object::Name(b"Catalog".to_vec())),
            (b"Pages".to_vec(), Object::Reference(pages_id)),
            (b"OpenAction".to_vec(), Object::Dictionary(action)),
        ]));
        doc.trailer.set("Root", Object::Reference(catalog_id));
        doc
    }

    fn make_doc_with_names_js() -> Document {
        let mut doc = Document::with_version("1.4");
        let js_tree = Dictionary::from_iter([(b"Names".to_vec(), Object::Array(vec![]))]);
        let names = Dictionary::from_iter([(b"JavaScript".to_vec(), Object::Dictionary(js_tree))]);
        let pages_id = doc.add_object(Dictionary::from_iter([
            (b"Type".to_vec(), Object::Name(b"Pages".to_vec())),
            (b"Kids".to_vec(), Object::Array(vec![])),
            (b"Count".to_vec(), Object::Integer(0)),
        ]));
        let catalog_id = doc.add_object(Dictionary::from_iter([
            (b"Type".to_vec(), Object::Name(b"Catalog".to_vec())),
            (b"Pages".to_vec(), Object::Reference(pages_id)),
            (b"Names".to_vec(), Object::Dictionary(names)),
        ]));
        doc.trailer.set("Root", Object::Reference(catalog_id));
        doc
    }

    fn make_doc_clean() -> Document {
        let mut doc = Document::with_version("1.4");
        let pages_id = doc.add_object(Dictionary::from_iter([
            (b"Type".to_vec(), Object::Name(b"Pages".to_vec())),
            (b"Kids".to_vec(), Object::Array(vec![])),
            (b"Count".to_vec(), Object::Integer(0)),
        ]));
        let catalog_id = doc.add_object(Dictionary::from_iter([
            (b"Type".to_vec(), Object::Name(b"Catalog".to_vec())),
            (b"Pages".to_vec(), Object::Reference(pages_id)),
        ]));
        doc.trailer.set("Root", Object::Reference(catalog_id));
        doc
    }

    #[test]
    fn open_action_javascript_is_rejected() {
        let doc = make_doc_with_open_action("JavaScript");
        let err = reject_javascript(&doc).unwrap_err();
        assert!(
            matches!(err, InterpError::JavaScript { location } if location.contains("OpenAction")),
            "expected JavaScript error, got: {err}"
        );
    }

    #[test]
    fn open_action_goto_is_allowed() {
        let doc = make_doc_with_open_action("GoTo");
        assert!(reject_javascript(&doc).is_ok());
    }

    #[test]
    fn names_javascript_is_rejected() {
        let doc = make_doc_with_names_js();
        let err = reject_javascript(&doc).unwrap_err();
        assert!(
            matches!(err, InterpError::JavaScript { location } if location.contains("Names")),
            "expected JavaScript error, got: {err}"
        );
    }

    #[test]
    fn clean_document_is_allowed() {
        let doc = make_doc_clean();
        assert!(reject_javascript(&doc).is_ok());
    }
}
