//! Lazy, zero-copy PDF file parser for the pdf-raster render pipeline.
//!
//! # Design
//!
//! * [`Document::open`] parses only the xref table — no objects are loaded.
//! * Objects are resolved on first access and cached (`Arc`-wrapped) for reuse.
//! * All byte scanning uses hand-rolled primitives in [`lexer`]; no nom/nom_locate.
//! * Shared parsing primitives live in [`lexer`] — `pdf_interp`'s content
//!   tokenizer will import from here in a follow-up refactor.
//!
//! # Compatibility
//!
//! Public types mirror `lopdf` naming so `pdf_interp` can migrate
//! file-by-file with mostly mechanical changes.

mod document;
mod error;
mod lexer;
mod object;
mod objstm;
mod stream;
mod xref;

pub use document::Document;
pub use error::PdfError;
pub use object::{Object, ObjectId, Stream, StringFormat};

/// Convenience re-export of the `HashMap` type used for PDF dictionaries.
///
/// Saves callers from writing `std::collections::HashMap<Vec<u8>, pdf::Object>`.
pub type Dictionary = std::collections::HashMap<Vec<u8>, Object>;
