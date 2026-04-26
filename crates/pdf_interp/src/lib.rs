//! PDF content stream interpreter — poppler-free render path.
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
pub mod renderer;
pub mod resources;

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
        }
    }
}

impl std::error::Error for InterpError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Pdf(e) => Some(e),
            _ => None,
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
/// # Errors
/// Returns [`InterpError::Pdf`] if the file cannot be read or is not a valid PDF.
pub fn open(path: impl AsRef<Path>) -> Result<Document, InterpError> {
    Ok(Document::load(path)?)
}

/// Return the number of pages in `doc`.
///
/// Saturates at [`u32::MAX`] for pathological documents (>4 billion pages).
#[must_use]
pub fn page_count(doc: &Document) -> u32 {
    u32::try_from(doc.get_pages().len()).unwrap_or(u32::MAX)
}

/// Return the `MediaBox` dimensions (`width_pts`, `height_pts`) for page `page_num` (1-based).
///
/// Falls back to US Letter (612 × 792 pt) if the `MediaBox` cannot be read.
///
/// # Errors
/// Returns [`InterpError::PageOutOfRange`] if the page number is invalid.
pub fn page_size_pts(doc: &Document, page_num: u32) -> Result<(f64, f64), InterpError> {
    let total = page_count(doc);
    if page_num == 0 || page_num > total {
        return Err(InterpError::PageOutOfRange {
            page: page_num,
            total,
        });
    }
    let pages = doc.get_pages();
    let page_id = *pages.get(&page_num).ok_or(InterpError::PageOutOfRange {
        page: page_num,
        total,
    })?;

    let fallback = (612.0, 792.0); // US Letter

    let Ok(dict) = doc.get_dictionary(page_id) else {
        return Ok(fallback);
    };

    let media_box = match dict.get(b"MediaBox") {
        Ok(lopdf::Object::Array(arr)) if arr.len() == 4 => arr,
        _ => return Ok(fallback),
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

    let x0 = to_f64(&media_box[0]);
    let y0 = to_f64(&media_box[1]);
    let x1 = to_f64(&media_box[2]);
    let y1 = to_f64(&media_box[3]);

    let w = (x1 - x0).abs();
    let h = (y1 - y0).abs();
    if w > 0.0 && h > 0.0 {
        Ok((w, h))
    } else {
        Ok(fallback)
    }
}

/// Parse the content stream for page `page_num` (1-based) and return the
/// decoded operator sequence.
///
/// # Errors
/// Returns [`InterpError::PageOutOfRange`] if `page_num` is 0 or exceeds the
/// document page count, or [`InterpError::Pdf`] if the content stream cannot
/// be read.
pub fn parse_page(doc: &Document, page_num: u32) -> Result<Vec<content::Operator>, InterpError> {
    let total = page_count(doc);
    if page_num == 0 || page_num > total {
        return Err(InterpError::PageOutOfRange {
            page: page_num,
            total,
        });
    }

    let pages = doc.get_pages();
    let page_id = *pages.get(&page_num).ok_or(InterpError::PageOutOfRange {
        page: page_num,
        total,
    })?;

    let content_bytes = doc.get_page_content(page_id)?;
    Ok(content::parse(&content_bytes))
}
