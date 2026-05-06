use std::fmt;

#[derive(Debug)]
pub enum PdfError {
    Io(std::io::Error),
    /// xref table or stream could not be found or parsed.
    BadXref(String),
    /// An indirect object could not be parsed at its declared offset.
    BadObject {
        id: u32,
        detail: String,
    },
    /// A required dictionary key was missing or had the wrong type.
    MissingKey(&'static str),
    /// A stream's /Filter could not be decoded.
    DecodeFailed(String),
    /// The document has no pages or the page tree is malformed.
    NoPages,
    /// Requested page number is out of range.
    PageOutOfRange {
        page: u32,
        total: u32,
    },
}

impl fmt::Display for PdfError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::BadXref(msg) => write!(f, "xref error: {msg}"),
            Self::BadObject { id, detail } => write!(f, "object {id}: {detail}"),
            Self::MissingKey(k) => write!(f, "missing required key /{k}"),
            Self::DecodeFailed(msg) => write!(f, "stream decode failed: {msg}"),
            Self::NoPages => write!(f, "document has no pages"),
            Self::PageOutOfRange { page, total } => {
                write!(f, "page {page} out of range (document has {total} pages)")
            }
        }
    }
}

impl std::error::Error for PdfError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        if let Self::Io(e) = self {
            Some(e)
        } else {
            None
        }
    }
}

impl From<std::io::Error> for PdfError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}
