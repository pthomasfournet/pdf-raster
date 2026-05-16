use std::fmt;

#[derive(Debug)]
pub enum PdfError {
    Io(std::io::Error),
    /// xref table or stream could not be found or parsed.
    BadXref(String),
    /// The input file is zero bytes (failed download, sync-conflict
    /// placeholder, interrupted scan-to-disk).  Distinct from a generic
    /// xref failure so the operator is told the real problem.
    EmptyInput,
    /// No `%PDF-` signature was found near the start of the file, so the
    /// input is not a PDF at all (wrong file, HTML error page saved as
    /// `.pdf`, random binary).
    NotPdf,
    /// A `%PDF-` header is present but no usable cross-reference table or
    /// trailer could be found or reconstructed — the file is truncated or
    /// corrupt.  The string carries the underlying low-level reason.
    MissingXref(String),
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
    /// The document is encrypted (PDF Standard Security Handler) and could
    /// not be transparently decrypted.  The string carries an accurate,
    /// actionable explanation (qpdf missing, password-protected, or the
    /// CLI liability gate was declined) — never the misleading
    /// "document has no pages".
    EncryptedDocument(String),
}

impl fmt::Display for PdfError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::BadXref(msg) => write!(f, "xref error: {msg}"),
            Self::EmptyInput => write!(f, "input file is empty (0 bytes)"),
            Self::NotPdf => write!(f, "not a PDF file (missing %PDF- header)"),
            Self::MissingXref(msg) => write!(
                f,
                "truncated or corrupt PDF (no cross-reference table found: {msg})"
            ),
            Self::BadObject { id, detail } => write!(f, "object {id}: {detail}"),
            Self::MissingKey(k) => write!(f, "missing required key /{k}"),
            Self::DecodeFailed(msg) => write!(f, "stream decode failed: {msg}"),
            Self::NoPages => write!(f, "document has no pages"),
            Self::PageOutOfRange { page, total } => {
                write!(f, "page {page} out of range (document has {total} pages)")
            }
            Self::EncryptedDocument(msg) => write!(f, "{msg}"),
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
