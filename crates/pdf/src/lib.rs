//! Lazy, zero-copy PDF file parser for the rasterrocket render pipeline.
//!
//! # Design
//!
//! * [`Document::open`] parses only the xref table — no objects are loaded.
//! * Objects are resolved on first access and cached (`Arc`-wrapped) for reuse.
//! * All byte scanning uses hand-rolled primitives in the private `lexer`
//!   submodule; no nom/nom_locate.
//! * Shared parsing primitives live in `lexer` — `rasterrocket-interp`'s content
//!   tokenizer will import from here in a follow-up refactor.
//!
//! # Compatibility
//!
//! Public types mirror `lopdf` naming so `rasterrocket-interp` can migrate
//! file-by-file with mostly mechanical changes.

mod dictionary;
mod document;
mod error;
mod lexer;
mod linearization;
mod madvise;
mod object;
mod objstm;
mod page_tree;
mod stream;
mod xref;

pub use dictionary::Dictionary;
pub use document::Document;
pub use error::PdfError;
pub use linearization::LinearizationHints;
pub use object::{Object, ObjectId, Stream, StringFormat};
pub use page_tree::descend_to_page_index;
