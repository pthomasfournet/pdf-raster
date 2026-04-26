//! PDF content stream interpreter — poppler-free render path.
//!
//! This crate bridges a parsed PDF document ([`lopdf`]) and the pixel-level
//! raster crate.  The entry point is [`render_page`].

pub mod content;

use lopdf::Document;

/// Errors that can occur during PDF interpretation.
#[derive(Debug)]
pub enum InterpError {
    /// lopdf failed to load or parse the document.
    Pdf(lopdf::Error),
    /// The requested page number is out of range.
    PageOutOfRange {
        /// The requested page (1-based).
        page: u32,
        /// Total pages in the document.
        total: u32,
    },
    /// A required resource (font, image, …) could not be resolved.
    MissingResource(String),
}

impl std::fmt::Display for InterpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pdf(e) => write!(f, "PDF error: {e}"),
            Self::PageOutOfRange { page, total } => {
                write!(f, "page {page} out of range (document has {total} pages)")
            }
            Self::MissingResource(name) => write!(f, "missing resource: {name}"),
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

/// Open a PDF from a file path and return the lopdf Document.
///
/// # Errors
/// Returns [`InterpError::Pdf`] if lopdf cannot load or parse the file.
pub fn open(path: &str) -> Result<Document, InterpError> {
    Ok(Document::load(path)?)
}

/// Return the number of pages in `doc`.
#[must_use]
pub fn page_count(doc: &Document) -> u32 {
    doc.get_pages().len() as u32
}

/// Parse the content stream for page `page_num` (1-based) and return the
/// decoded operator sequence.
///
/// This is the first step toward a full render; currently it only parses —
/// rasterization will be wired in subsequent commits.
///
/// # Errors
/// Returns [`InterpError`] if the page does not exist or the content stream
/// cannot be read.
pub fn parse_page(doc: &Document, page_num: u32) -> Result<Vec<content::Operator>, InterpError> {
    let total = page_count(doc);
    if page_num == 0 || page_num > total {
        return Err(InterpError::PageOutOfRange { page: page_num, total });
    }

    let pages = doc.get_pages();
    // get_pages() returns a BTreeMap<page_num, ObjectId>.
    let page_id = *pages
        .get(&page_num)
        .ok_or_else(|| InterpError::PageOutOfRange { page: page_num, total })?;

    let content_bytes = doc.get_page_content(page_id)?;
    Ok(content::parse(&content_bytes))
}
