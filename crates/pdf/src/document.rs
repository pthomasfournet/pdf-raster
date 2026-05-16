//! The top-level [`Document`] type: lazy, memory-mapped PDF reader.
//!
//! Open a PDF with [`Document::open`].  Objects are resolved on demand from the
//! memory-mapped file — the constructor only parses the xref table and trailer.

use std::{
    collections::HashMap,
    path::Path,
    sync::{Arc, Mutex, OnceLock},
};

use memmap2::Mmap;

use crate::{
    decrypt::{self, DecryptGuard},
    dictionary::Dictionary,
    error::PdfError,
    linearization::LinearizationHints,
    object::{Object, ObjectId, parse_indirect_object},
    objstm::ObjStmCache,
    stream::decode_stream,
    xref::{XrefEntry, XrefTable, read_xref},
};

// ── Limits ───────────────────────────────────────────────────────────────────

/// Aggregate cap on a page's concatenated, decompressed content streams.
///
/// A `/Contents` array of many refs (each individually bounded by the
/// per-stream decompression limit in `stream.rs`) can otherwise sum without
/// bound. This cap mirrors the per-stream `MAX_DECOMPRESSED` discipline.
const MAX_PAGE_CONTENT: usize = 512 * 1024 * 1024; // 512 MiB

/// Collect the indirect references from a `/Contents` array.
///
/// Shared by the indirect-ref-to-array and direct-array arms of
/// [`Document::get_page_content`] so the extraction (and its diagnostics)
/// exist in exactly one place.  ISO 32000-2 §7.7.3.3 says every element of a
/// `/Contents` array is an indirect reference to a content stream, so any
/// element that is *not* a reference (an inline object, a nested array, a
/// number) cannot contribute renderable bytes.  Silently filtering such an
/// element is the v1 partial-blank failure mode, so each one is logged before
/// it is dropped; `page_id` is threaded purely so the message names the page.
fn collect_content_refs(arr: &[Object], page_id: ObjectId) -> Vec<ObjectId> {
    arr.iter()
        .filter_map(|el| {
            el.as_reference().or_else(|| {
                log::warn!(
                    "get_page_content: page {} /Contents array element is {}, \
                     not an indirect reference — skipping (page may be missing content)",
                    page_id.0,
                    el.enum_variant()
                );
                None
            })
        })
        .collect()
}

// ── Document ─────────────────────────────────────────────────────────────────

/// A lazily-parsed PDF document backed by a memory-mapped file.
pub struct Document {
    /// The memory-mapped file data.  `Arc` so callers can hold byte slices.
    data: Arc<MmapOrVec>,
    xref: XrefTable,
    /// Per-object parse cache.  `None` = not yet resolved.
    cache: Mutex<HashMap<u32, Arc<Object>>>,
    /// Object-stream decompression cache (shared across threads).
    objstm: ObjStmCache,
    /// Catalogue object ID, resolved once from the trailer.
    catalog_id: OnceLock<ObjectId>,
    /// `/Pages` root reference, resolved once from the catalog.  Cached
    /// because every page-tree descent reads it.
    pages_root_id: OnceLock<ObjectId>,
    /// `/Pages /Count`, memoised on first read.  Falls back to the eager
    /// walk on malformed catalogs.
    page_count_cache: OnceLock<u32>,
    /// Cached linearization hints, parsed lazily on first access.
    lin_hints: OnceLock<Option<LinearizationHints>>,
    /// Owns the qpdf-decrypted temporary file (if any) so it outlives the
    /// memory map taken over it.  A no-op guard for documents that were
    /// not encrypted.  Dropped with the `Document`, unlinking the temp
    /// plaintext — no leaked decrypted copies.
    #[expect(
        dead_code,
        reason = "lifetime anchor only; Drop unlinks the temp file when the Document is dropped"
    )]
    decrypt_guard: DecryptGuard,
}

/// Memory source: mmap for file paths, Vec<u8> for in-memory use (tests, fuzz).
enum MmapOrVec {
    Mapped(Mmap),
    Owned(Vec<u8>),
}

impl MmapOrVec {
    fn as_bytes(&self) -> &[u8] {
        match self {
            Self::Mapped(m) => m,
            Self::Owned(v) => v,
        }
    }
}

impl Document {
    /// Open a PDF file at `path`.
    ///
    /// Reads and parses only the xref table(s) and trailer.  Individual objects
    /// are parsed on first access and cached for subsequent calls.
    pub fn open(path: &Path) -> Result<Self, PdfError> {
        Self::open_at(path, DecryptGuard::none())
    }

    /// Open `path`, transparently qpdf-decrypting first when the document
    /// is encrypted and `authorized` is `true`.
    ///
    /// The decision to authorise decryption is made by the caller (the CLI
    /// gates it behind an interactive liability waiver / explicit operator
    /// bypass; the QA harness auto-authorises).  This constructor only
    /// performs the mechanical decrypt and anchors the resulting temp
    /// file's lifetime to the returned `Document`.
    ///
    /// Unencrypted documents take the original path with no qpdf spawn and
    /// no temp file — a pure structural trailer check.
    ///
    /// # Errors
    /// - [`PdfError::EncryptedDocument`] when the document is encrypted and
    ///   `authorized` is `false`, when `qpdf` is absent, or when qpdf
    ///   cannot decrypt (password-protected).  Never the misleading
    ///   "document has no pages".
    pub fn open_decrypting(path: &Path, authorized: bool) -> Result<Self, PdfError> {
        // Cheap probe: open the original to read the trailer's /Encrypt.
        // No objects are parsed, so this is the same cost as `open`.
        let probe = Self::open_at(path, DecryptGuard::none())?;
        let encrypted = probe.is_encrypted();
        drop(probe);

        let (source, guard) = decrypt::resolve_source(path, encrypted, authorized)?;
        Self::open_at(&source, guard)
    }

    /// Open `path` and take ownership of `guard` (the qpdf temp-file RAII
    /// anchor, or a no-op guard).
    fn open_at(path: &Path, guard: DecryptGuard) -> Result<Self, PdfError> {
        let file = std::fs::File::open(path)?;
        crate::madvise::advise_random(&file);
        // SAFETY: we do not mutate the file while it is mapped.
        let mmap = unsafe { Mmap::map(&file)? };
        let data = Arc::new(MmapOrVec::Mapped(mmap));
        Self::from_bytes_with_guard(data, guard)
    }

    /// Construct from an in-memory byte slice (used by tests).
    pub fn from_bytes_owned(bytes: Vec<u8>) -> Result<Self, PdfError> {
        Self::from_bytes_with_guard(Arc::new(MmapOrVec::Owned(bytes)), DecryptGuard::none())
    }

    fn from_bytes_with_guard(data: Arc<MmapOrVec>, guard: DecryptGuard) -> Result<Self, PdfError> {
        validate_pdf_input(data.as_bytes())?;
        let xref = read_xref(data.as_bytes()).map_err(|e| match e {
            // A `%PDF-` header is present (validate_pdf_input passed) but the
            // xref could neither be parsed nor reconstructed: the file is
            // truncated or corrupt.  Replace the low-level "startxref not
            // found" with an accurate, operator-actionable diagnosis.
            PdfError::BadXref(detail) => PdfError::MissingXref(detail),
            other => other,
        })?;
        Ok(Self {
            data,
            xref,
            cache: Mutex::new(HashMap::new()),
            objstm: ObjStmCache::new(),
            catalog_id: OnceLock::new(),
            pages_root_id: OnceLock::new(),
            page_count_cache: OnceLock::new(),
            lin_hints: OnceLock::new(),
            decrypt_guard: guard,
        })
    }

    // ── Object resolution ────────────────────────────────────────────────────

    /// Resolve an indirect reference and return a shared reference to its value.
    ///
    /// First call parses the object from the file; subsequent calls return the
    /// cached value.
    pub fn get_object(&self, id: ObjectId) -> Result<Arc<Object>, PdfError> {
        {
            // Poison recovery: a previous panic while holding the lock left the
            // mutex poisoned. The cache is just a memo of immutable parses, so
            // the inner state is safe to keep using.
            let guard = self.cache.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(obj) = guard.get(&id.0) {
                return Ok(Arc::clone(obj));
            }
        }

        let obj = self.parse_object_uncached(id)?;
        let arc = Arc::new(obj);
        self.cache
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(id.0, Arc::clone(&arc));
        Ok(arc)
    }

    /// Resolve a chain of indirect references until a non-Reference object is
    /// reached.  Prevents infinite loops with a depth counter.
    pub fn resolve(&self, obj: &Object) -> Result<Arc<Object>, PdfError> {
        let mut current = obj.clone();
        for _ in 0..32 {
            match current {
                Object::Reference(id) => current = (*self.get_object(id)?).clone(),
                other => return Ok(Arc::new(other)),
            }
        }
        Err(PdfError::BadObject {
            id: 0,
            detail: "infinite reference chain".into(),
        })
    }

    fn parse_object_uncached(&self, id: ObjectId) -> Result<Object, PdfError> {
        let entry = self.xref.get(id.0).ok_or_else(|| PdfError::BadObject {
            id: id.0,
            detail: "not in xref table".into(),
        })?;

        match entry {
            XrefEntry::Direct { offset, .. } => {
                let (_, obj) = parse_indirect_object(self.data.as_bytes(), offset as usize)
                    .ok_or_else(|| PdfError::BadObject {
                        id: id.0,
                        detail: format!("parse failed at offset {offset}"),
                    })?;
                Ok(obj)
            }
            XrefEntry::InObjStm { container, index } => {
                let container_obj = self.get_object((container, 0))?;
                let stream = match container_obj.as_ref() {
                    Object::Stream(s) => s,
                    _ => {
                        return Err(PdfError::BadObject {
                            id: container,
                            detail: "expected Stream for ObjStm container".into(),
                        });
                    }
                };
                self.objstm
                    .get_or_parse(container, index, &stream.content, &stream.dict)
            }
        }
    }

    // ── Typed dictionary accessors ────────────────────────────────────────────

    /// Resolve `id` and return it as a dictionary.
    ///
    /// Currently clones the dictionary out of the cached `Arc<Object>` because
    /// stable Rust has no zero-copy way to project an `Arc<Object>` to an
    /// `Arc<Dictionary>` (the dict lives inside the enum variant). Hot
    /// repeat-lookup callers (e.g. per-page resource resolution) should cache
    /// the result themselves; the dict is hashed and small (~tens of entries)
    /// so the clone cost is bounded.
    pub fn get_dict(&self, id: ObjectId) -> Result<Arc<Dictionary>, PdfError> {
        let obj = self.get_object(id)?;
        match obj.as_ref() {
            Object::Dictionary(d) => Ok(Arc::new(d.clone())),
            Object::Stream(s) => Ok(Arc::new(s.dict.clone())),
            _ => Err(PdfError::BadObject {
                id: id.0,
                detail: format!("expected Dictionary or Stream, got {}", obj.enum_variant()),
            }),
        }
    }

    /// Lopdf-compatible alias for [`Self::get_dict`].
    pub fn get_dictionary(&self, id: ObjectId) -> Result<Arc<Dictionary>, PdfError> {
        self.get_dict(id)
    }

    /// Resolve `id` and return the underlying `Arc<Object>` if it is a
    /// dictionary or stream-with-dict.
    ///
    /// Unlike [`Self::get_dict`], this does NOT clone the dictionary —
    /// the caller borrows directly from the cached `Arc<Object>`.  Use
    /// this on hot paths (e.g., page-tree descent) where every saved
    /// `HashMap<Vec<u8>, Object>` clone matters.  Callers must accept
    /// that the borrow lives only as long as the returned `Arc`.
    pub fn get_dict_arc(&self, id: ObjectId) -> Result<Arc<Object>, PdfError> {
        let obj = self.get_object(id)?;
        match obj.as_ref() {
            Object::Dictionary(_) | Object::Stream(_) => Ok(obj),
            _ => Err(PdfError::BadObject {
                id: id.0,
                detail: format!("expected Dictionary or Stream, got {}", obj.enum_variant()),
            }),
        }
    }

    // ── Document structure ────────────────────────────────────────────────────

    /// Return the document catalogue dictionary.
    pub fn catalog(&self) -> Result<Arc<Dictionary>, PdfError> {
        let id = self.catalog_id()?;
        self.get_dict(id)
    }

    fn catalog_id(&self) -> Result<ObjectId, PdfError> {
        if let Some(id) = self.catalog_id.get() {
            return Ok(*id);
        }
        let root = self
            .xref
            .trailer
            .get(b"Root")
            .ok_or(PdfError::MissingKey("Root"))?;
        let id = match root {
            Object::Reference(id) => *id,
            other => {
                // Some PDFs embed the catalog inline in the trailer.
                return Err(PdfError::BadObject {
                    id: 0,
                    detail: format!("trailer /Root is not a reference: {other:?}"),
                });
            }
        };
        let _ = self.catalog_id.set(id);
        Ok(id)
    }

    /// Return parsed linearization hints, or `None` if the document is
    /// not linearized.  Parsed once on first call and cached.
    ///
    /// `LinearizationHints` is `Copy` (24 bytes), so callers receive an
    /// owned value rather than a borrow that would tie up the document
    /// for its lifetime.
    ///
    /// # Errors
    /// Currently never errors — `try_load` collapses every failure mode
    /// (missing object 1, malformed dict, missing keys, invalid offsets)
    /// into `Ok(None)`.  The `Result` return type is reserved for the
    /// future hint-stream parser, which will gain real failure modes;
    /// the call site can switch to `OnceLock::get_or_try_init` when
    /// stable (rust-lang/rust#109737).
    pub fn linearization_hints(&self) -> Result<Option<LinearizationHints>, PdfError> {
        Ok(*self
            .lin_hints
            .get_or_init(|| LinearizationHints::try_load(self).ok().flatten()))
    }

    /// Raw byte view of the underlying file.  Used by hint-table parsing
    /// and by callers that want to hash the document content without
    /// re-reading the file.
    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        self.data.as_bytes()
    }

    /// True when the trailer carries an `/Encrypt` entry, i.e. the document
    /// uses the PDF Standard Security Handler.  This is a pure structural
    /// check — no decryption is attempted and no objects are parsed.
    ///
    /// Used to route encrypted documents through the qpdf-assisted decrypt
    /// preprocess instead of failing with the misleading "no pages".
    #[must_use]
    pub fn is_encrypted(&self) -> bool {
        self.xref.trailer.contains_key(b"Encrypt")
    }

    /// Return the number of pages in the document.
    pub fn page_count(&self) -> u32 {
        self.get_pages().count() as u32
    }

    /// Return the number of pages without walking the page tree.
    ///
    /// Reads `/Pages /Count` directly from the catalog and memoises it on
    /// first call.  Falls back to the eager [`Self::page_count`] (full tree
    /// walk) if the catalog is malformed.
    ///
    /// Cached, so repeated calls are a relaxed-load away from the answer.
    #[must_use]
    pub fn page_count_fast(&self) -> u32 {
        *self
            .page_count_cache
            .get_or_init(|| self.read_page_count_uncached())
    }

    /// Inner page-count read.  Used by `page_count_fast` (memoised) and
    /// the eager-walk fallback inside the page-tree descent.
    #[must_use]
    fn read_page_count_uncached(&self) -> u32 {
        // Linearized PDFs declare /N up front; trust it without descending
        // into the catalog dict.
        if let Ok(Some(hints)) = self.linearization_hints() {
            return hints.page_count;
        }
        let Ok(pages_root) = self.pages_root_id() else {
            return self.page_count();
        };
        let Ok(dict) = self.get_dict(pages_root) else {
            return self.page_count();
        };
        dict.get(b"Count")
            .and_then(Object::as_u32)
            .unwrap_or_else(|| self.page_count())
    }

    /// Return the catalog's `/Pages` reference, memoised on first call.
    ///
    /// `pages_root` is invariant for the document, so caching it avoids
    /// re-cloning the catalog dictionary on every page-tree descent.
    pub fn pages_root_id(&self) -> Result<ObjectId, PdfError> {
        if let Some(id) = self.pages_root_id.get() {
            return Ok(*id);
        }
        let catalog = self.catalog()?;
        let id = catalog
            .get(b"Pages")
            .and_then(Object::as_reference)
            .ok_or(PdfError::MissingKey("Pages"))?;
        let _ = self.pages_root_id.set(id);
        Ok(id)
    }

    /// Resolve a 0-based page index to its `ObjectId` via logarithmic descent.
    ///
    /// See [`crate::descend_to_page_index`] for the full descent contract.
    pub fn get_page(&self, idx: u32) -> Result<ObjectId, PdfError> {
        crate::page_tree::descend_to_page_index(self, idx)
    }

    /// Iterate over all pages in document order, yielding `(1-based page number, ObjectId)`.
    pub fn get_pages(&self) -> impl Iterator<Item = (u32, ObjectId)> + '_ {
        PageIter::new(self)
    }

    /// Return the decoded content stream bytes for a page.
    pub fn get_page_content(&self, page_id: ObjectId) -> Result<Vec<u8>, PdfError> {
        let page = self.get_object(page_id)?;
        let page_dict = page.as_dict().ok_or_else(|| PdfError::BadObject {
            id: page_id.0,
            detail: "page is not a dict".into(),
        })?;

        let contents = match page_dict.get(b"Contents") {
            None => return Ok(Vec::new()),
            Some(obj) => obj.clone(),
        };

        // /Contents (ISO 32000-2 §7.7.3.3) is either a single content stream
        // or an array of streams whose decoded bytes are concatenated.  Either
        // form may itself be reached through an indirect reference, so a
        // reference can resolve to a *stream* OR to an *array of stream
        // references* — modern producers (LibreOffice, many imposition tools)
        // emit `/Contents N 0 R` where object N is `[a 0 R b 0 R c 0 R]`.
        // The previous code treated such a reference as a single stream;
        // `get_object` returned the Array, the `Object::Stream` match failed,
        // and the page rendered blank.  Resolve one indirection level so a
        // ref-to-array expands into its element references.
        //
        // This expansion is intentionally a single, non-recursive pass: the
        // spec permits exactly one array of *direct stream references*, never
        // an array nested inside an array, so a flat collection both matches
        // the spec and makes a cyclic/self-referential `/Contents` ring
        // (an element ref pointing back at the array object) structurally
        // impossible to loop on — each `get_object` is visited at most once
        // and non-streams are diagnosed and skipped in the decode loop below.
        let refs: Vec<ObjectId> = match &contents {
            Object::Reference(id) => match self.get_object(*id)?.as_ref() {
                Object::Array(arr) => collect_content_refs(arr, page_id),
                // Stream (or anything else): keep the single ref; the decode
                // loop below filters and diagnoses non-streams.
                _ => vec![*id],
            },
            Object::Array(arr) => collect_content_refs(arr, page_id),
            _ => return Ok(Vec::new()),
        };

        let mut out = Vec::new();
        for r in refs {
            let stream_obj = self.get_object(r)?;
            let Object::Stream(s) = stream_obj.as_ref() else {
                // A `/Contents` array element that does not resolve to a
                // stream (a dict, null, number, or — non-conformant — a
                // nested array) is not renderable content.  Dropping it
                // silently is the v1 blank/partial-page failure: the page
                // would render with missing operators and EXIT 0.  Surface
                // it loudly and continue with the streams that *are* valid
                // so the page degrades gracefully instead of vanishing.
                log::warn!(
                    "get_page_content: page {} /Contents element obj {} is {}, \
                     not a stream — skipping (page may be missing content)",
                    page_id.0,
                    r.0,
                    stream_obj.enum_variant()
                );
                continue;
            };
            let decoded = decode_stream(&s.content, &s.dict).map_err(PdfError::DecodeFailed)?;
            // +1 accounts for the '\n' separator pushed below.
            if out.len().saturating_add(decoded.len()).saturating_add(1) > MAX_PAGE_CONTENT {
                return Err(PdfError::DecodeFailed(format!(
                    "page content exceeds {} MiB aggregate decompressed cap \
                     (possible decompression bomb)",
                    MAX_PAGE_CONTENT / (1024 * 1024)
                )));
            }
            // ISO 32000-2 §7.8.2: streams in a `/Contents` array are decoded
            // and *concatenated as if a single stream*, but a content stream
            // need not end at an operator/token boundary — the last token of
            // one stream can abut the first of the next (e.g. one stream ends
            // "...50 7", the next begins "00 cm").  A whitespace byte between
            // them is required so the lexer keeps the tokens distinct; a
            // single LF is the canonical, minimal separator.  Skip it only
            // when the accumulated buffer already ends in PDF whitespace
            // (LF/CR/space/tab/FF/NUL), which already provides the boundary.
            if out
                .last()
                .is_some_and(|b| !matches!(b, b'\n' | b'\r' | b' ' | b'\t' | 0x0C | 0x00))
            {
                out.push(b'\n');
            }
            out.extend_from_slice(&decoded);
        }
        Ok(out)
    }

    /// Return a merged font dictionary for the page (all /Font entries from
    /// /Resources, with parent-chain inheritance).
    pub fn get_page_fonts(&self, page_id: ObjectId) -> Result<Dictionary, PdfError> {
        self.get_page_resource_dict(page_id, b"Font")
    }

    /// Return one named sub-dictionary from the page's /Resources, following
    /// parent page-tree inheritance.
    pub fn get_page_resource_dict(
        &self,
        page_id: ObjectId,
        resource_name: &[u8],
    ) -> Result<Dictionary, PdfError> {
        // Walk up the page tree to find /Resources. Bound to 64 hops to defeat
        // cyclic /Parent references in malformed PDFs.
        let mut current_id = Some(page_id);
        let mut visited = std::collections::HashSet::new();
        for _ in 0..64 {
            let Some(id) = current_id else { break };
            if !visited.insert(id.0) {
                break;
            }
            let obj = self.get_object(id)?;
            let Some(dict) = obj.as_dict().cloned() else {
                break;
            };

            if let Some(res) = dict.get(b"Resources") {
                let res_dict = self.dict_from_object(res)?;
                if let Some(sub) = res_dict.get(resource_name) {
                    return self.dict_from_object(sub);
                }
            }

            // Walk to parent.
            current_id = dict.get(b"Parent").and_then(Object::as_reference);
        }
        Ok(Dictionary::new())
    }

    /// Return an inheritable page attribute value for a page object, following
    /// the `/Parent` chain per ISO 32000-2 §7.7.3.4.
    ///
    /// Checks the page leaf first, then walks up the `/Pages` ancestors until
    /// the key is found or the root is reached.  Returns `None` if the key is
    /// absent from every node in the chain.
    ///
    /// A hostile PDF can build a cyclic `/Parent` ring (page → pages → page) or
    /// a pathologically deep chain.  Both are defended: a visited-object set
    /// detects revisits on the first repeat, and the walk is hard-bounded to
    /// `PARENT_CHAIN_MAX_HOPS` (matching `page_tree.rs` `MAX_DEPTH`).  Either
    /// guard tripping is a malformed page tree, so it is logged loudly and the
    /// lookup fails closed (`None` → caller falls back to a default box) rather
    /// than spinning, recursing, or terminating silently.
    pub fn get_inherited_page_attr(&self, page_id: ObjectId, key: &[u8]) -> Option<Object> {
        // Matches `page_tree.rs` MAX_DEPTH so both /Parent walkers agree on the
        // legal page-tree depth bound.
        const PARENT_CHAIN_MAX_HOPS: usize = 64;

        let mut visited: std::collections::HashSet<u32> = std::collections::HashSet::new();
        let mut current_id = Some(page_id);
        for _ in 0..PARENT_CHAIN_MAX_HOPS {
            let id = current_id?;
            if !visited.insert(id.0) {
                log::warn!(
                    "cyclic /Parent reference at object {} while resolving inheritable \
                     attribute {:?}; treating as absent (malformed PDF, ISO 32000-2 §7.7.3.4)",
                    id.0,
                    String::from_utf8_lossy(key),
                );
                return None;
            }
            let obj = self.get_object(id).ok()?;
            let dict = match obj.as_ref() {
                Object::Dictionary(d) => d,
                Object::Stream(s) => &s.dict,
                _ => return None,
            };
            if let Some(val) = dict.get(key) {
                // Resolve one level of indirection so callers always get the
                // concrete value (e.g. an Array, not a Reference to an Array).
                let resolved = match val {
                    Object::Reference(rid) => {
                        self.get_object(*rid).ok().map(|o| o.as_ref().clone())
                    }
                    other => Some(other.clone()),
                };
                return resolved;
            }
            current_id = dict.get(b"Parent").and_then(Object::as_reference);
        }
        // Ran out of hops without finding the key or hitting the root.  A
        // conforming page tree is far shallower than the bound, so this is a
        // hostile/over-deep chain — surface it instead of silently defaulting.
        log::warn!(
            "/Parent chain exceeded {PARENT_CHAIN_MAX_HOPS} hops while resolving inheritable \
             attribute {:?}; treating as absent (malformed PDF, ISO 32000-2 §7.7.3.4)",
            String::from_utf8_lossy(key),
        );
        None
    }

    /// Take a dictionary value out of an `Object`, resolving one level of
    /// indirection if needed. Non-dict values yield an empty map (matches the
    /// behaviour of malformed-but-recoverable PDFs).
    fn dict_from_object(&self, obj: &Object) -> Result<Dictionary, PdfError> {
        match obj {
            Object::Dictionary(d) => Ok(d.clone()),
            Object::Reference(rid) => {
                let r = self.get_object(*rid)?;
                Ok(r.as_dict().cloned().unwrap_or_default())
            }
            _ => Ok(Dictionary::new()),
        }
    }
}

// ── Page iterator ─────────────────────────────────────────────────────────────

struct PageIter<'a> {
    doc: &'a Document,
    /// Stack of (node_id, next_child_index) for page-tree traversal.
    stack: Vec<(ObjectId, usize)>,
    /// Object numbers currently on the traversal stack — guards against cyclic
    /// /Kids references (malformed PDFs would otherwise loop forever).
    on_stack: std::collections::HashSet<u32>,
    page_num: u32,
}

impl<'a> PageIter<'a> {
    fn new(doc: &'a Document) -> Self {
        // Catalog errors collapse to an empty iterator — same outcome as a PDF
        // with no /Pages root. Logged so corruption doesn't go silent.
        let catalog = match doc.catalog() {
            Ok(c) => c,
            Err(e) => {
                log::warn!("PageIter: catalog unavailable, no pages will be yielded: {e:?}");
                Default::default()
            }
        };
        let pages_id = catalog.get(b"Pages").and_then(Object::as_reference);
        let mut on_stack = std::collections::HashSet::new();
        let stack = match pages_id {
            Some(id) => {
                on_stack.insert(id.0);
                vec![(id, 0)]
            }
            None => Vec::new(),
        };
        Self {
            doc,
            stack,
            on_stack,
            page_num: 0,
        }
    }

    fn pop(&mut self) {
        if let Some((id, _)) = self.stack.pop() {
            self.on_stack.remove(&id.0);
        }
    }
}

impl Iterator for PageIter<'_> {
    type Item = (u32, ObjectId);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let (node_id, child_idx) = self.stack.last_mut()?;
            let node_id = *node_id;

            let obj = match self.doc.get_object(node_id) {
                Ok(o) => o,
                Err(_) => {
                    self.pop();
                    continue;
                }
            };
            let dict = match obj.as_dict() {
                Some(d) => d.clone(),
                None => {
                    self.pop();
                    continue;
                }
            };

            let node_type = dict.get(b"Type").and_then(Object::as_name).unwrap_or(b"");

            if node_type == b"Page" {
                // Leaf page node.
                self.pop();
                self.page_num += 1;
                return Some((self.page_num, node_id));
            }

            // Pages node: iterate /Kids. The spec allows /Kids to be either
            // a direct array or an indirect reference to an array; resolve
            // the reference case so we don't silently treat valid PDFs as
            // empty (corpus-04 hit this).
            let kids = match dict.get(b"Kids") {
                Some(Object::Array(a)) => a.clone(),
                Some(Object::Reference(rid)) => match self.doc.get_object(*rid) {
                    Ok(o) => match o.as_ref() {
                        Object::Array(a) => a.clone(),
                        _ => {
                            self.pop();
                            continue;
                        }
                    },
                    Err(_) => {
                        self.pop();
                        continue;
                    }
                },
                _ => {
                    self.pop();
                    continue;
                }
            };

            let idx = *child_idx;
            if idx >= kids.len() {
                self.pop();
                continue;
            }
            *child_idx += 1;

            if let Some(kid_id) = kids[idx].as_reference() {
                if self.on_stack.insert(kid_id.0) {
                    self.stack.push((kid_id, 0));
                } else {
                    log::warn!(
                        "PageIter: cyclic /Kids reference to obj {} skipped",
                        kid_id.0
                    );
                }
            }
        }
    }
}

/// How far into the file the `%PDF-` header may legally appear.
///
/// The spec puts the header at byte 0, but real-world files prepend a few
/// junk bytes (mail transfer artefacts, UTF-8 BOM, shebang lines).  poppler
/// and mutool scan roughly the first kilobyte; we match that leniency.
const HEADER_SCAN_WINDOW: usize = 1024;

/// Reject obviously-unusable input *before* the xref parser runs so the
/// operator gets an accurate diagnosis ("file is empty" / "not a PDF")
/// instead of the low-level, misleading "'startxref' not found".
///
/// This is the single entry choke point shared by [`Document::open`] and
/// [`Document::from_bytes_owned`].
fn validate_pdf_input(data: &[u8]) -> Result<(), PdfError> {
    if data.is_empty() {
        return Err(PdfError::EmptyInput);
    }
    let window = &data[..data.len().min(HEADER_SCAN_WINDOW)];
    if !window.windows(5).any(|w| w == b"%PDF-") {
        return Err(PdfError::NotPdf);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal_pdf() -> Vec<u8> {
        // Minimal valid PDF with one empty page.
        // Offsets verified by byte-counting the sections below.
        // %PDF-1.4\n = 9 bytes  → obj1 at 9
        // "1 0 obj\n<</Type /Catalog /Pages 2 0 R>>\nendobj\n" = 47 bytes → obj2 at 56
        // "2 0 obj\n<</Type /Pages /Kids [3 0 R] /Count 1>>\nendobj\n" = 55 bytes → obj3 at 111
        // "3 0 obj\n<</Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]>>\nendobj\n" = 69 bytes → xref at 180
        b"%PDF-1.4\n\
1 0 obj\n<</Type /Catalog /Pages 2 0 R>>\nendobj\n\
2 0 obj\n<</Type /Pages /Kids [3 0 R] /Count 1>>\nendobj\n\
3 0 obj\n<</Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]>>\nendobj\n\
xref\n0 4\n\
0000000000 65535 f\r\n\
0000000009 00000 n\r\n\
0000000056 00000 n\r\n\
0000000111 00000 n\r\n\
trailer\n<</Size 4 /Root 1 0 R>>\n\
startxref\n180\n%%EOF"
            .to_vec()
    }

    #[test]
    fn open_minimal_pdf() {
        let doc = Document::from_bytes_owned(minimal_pdf()).unwrap();
        assert_eq!(doc.page_count(), 1);
    }

    #[test]
    fn page_count_fast_matches_eager() {
        let doc = Document::from_bytes_owned(minimal_pdf()).unwrap();
        assert_eq!(doc.page_count_fast(), 1);
        assert_eq!(doc.page_count_fast(), doc.page_count());
    }

    #[test]
    fn get_pages_returns_one_page() {
        let doc = Document::from_bytes_owned(minimal_pdf()).unwrap();
        let pages: Vec<_> = doc.get_pages().collect();
        assert_eq!(pages.len(), 1);
        assert_eq!(pages[0].0, 1);
    }

    /// Regression: a Pages node whose `/Kids` is an indirect reference to an
    /// array (rather than an inline array) used to bail and report 0 pages.
    /// Real PDFs in the wild ship this layout (e.g. corpus-04).
    #[test]
    fn get_pages_handles_kids_as_reference() {
        let header = "%PDF-1.4\n";
        let obj1 = "1 0 obj\n<</Type /Catalog /Pages 2 0 R>>\nendobj\n";
        // Kids points to obj 4, which is an array containing obj 3 (the page).
        let obj2 = "2 0 obj\n<</Type /Pages /Kids 4 0 R /Count 1>>\nendobj\n";
        let obj3 = "3 0 obj\n<</Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]>>\nendobj\n";
        let obj4 = "4 0 obj\n[3 0 R]\nendobj\n";
        let off1 = header.len();
        let off2 = off1 + obj1.len();
        let off3 = off2 + obj2.len();
        let off4 = off3 + obj3.len();
        let xref_start = off4 + obj4.len();
        let xref = format!(
            "xref\n0 5\n0000000000 65535 f\r\n\
             {off1:010} 00000 n\r\n{off2:010} 00000 n\r\n\
             {off3:010} 00000 n\r\n{off4:010} 00000 n\r\n"
        );
        let trailer = format!("trailer\n<</Size 5 /Root 1 0 R>>\nstartxref\n{xref_start}\n%%EOF");
        let bytes = format!("{header}{obj1}{obj2}{obj3}{obj4}{xref}{trailer}").into_bytes();
        let doc = Document::from_bytes_owned(bytes).expect("indirect-Kids PDF");
        assert_eq!(doc.page_count(), 1, "expected 1 page through indirect Kids");
    }

    /// Build a minimal PDF-1.5 whose cross-reference is a `/Type /XRef`
    /// stream (`/W [1 2 2]`) and whose Catalog / Pages / Page objects live
    /// inside a `/Type /ObjStm` compressed object stream.  This is the
    /// structure emitted by virtually every modern PDF toolchain (pdfTeX,
    /// LibreOffice, …); the streams here are left uncompressed (no `/Filter`)
    /// so the fixture needs no zlib round-trip.
    ///
    /// Regression for the PDF-1.5 blank-render class: the page object (obj 3)
    /// is reachable only via an xref-stream **type-2** entry → ObjStm
    /// extraction.  If `/W` decoding, type-2 `get_object` routing, or ObjStm
    /// directory parsing is wrong, the page object is unresolvable and the
    /// page renders blank.
    #[test]
    fn pdf15_xref_stream_with_objstm_resolves_in_objstm_page() {
        let header = b"%PDF-1.5\n";

        // Bodies for the three objects packed into the ObjStm.
        let b1 = b"<</Type/Catalog/Pages 2 0 R>>";
        let b2 = b"<</Type/Pages/Kids[3 0 R]/Count 1>>";
        let b3 = b"<</Type/Page/Parent 2 0 R/MediaBox[0 0 200 200]>>";
        let o1 = 0usize;
        let o2 = o1 + b1.len() + 1; // +1 for the '\n' body separator
        let o3 = o2 + b2.len() + 1;
        let directory = format!("1 {o1} 2 {o2} 3 {o3} ").into_bytes();
        let first = directory.len();
        let mut objstm_data = directory.clone();
        for body in [&b1[..], &b2[..], &b3[..]] {
            objstm_data.extend_from_slice(body);
            objstm_data.push(b'\n');
        }
        let mut objstm = format!(
            "4 0 obj\n<</Type/ObjStm/N 3/First {first}/Length {}>>\nstream\n",
            objstm_data.len()
        )
        .into_bytes();
        objstm.extend_from_slice(&objstm_data);
        objstm.extend_from_slice(b"\nendstream\nendobj\n");

        let off4 = header.len();
        let off5 = off4 + objstm.len();

        // /W [1 2 2] entries for object numbers 0..=5.
        let entry = |t: u8, f1: u32, f2: u32| -> [u8; 5] {
            let a = (f1 as u16).to_be_bytes();
            let b = (f2 as u16).to_be_bytes();
            [t, a[0], a[1], b[0], b[1]]
        };
        let mut entries = Vec::new();
        entries.extend_from_slice(&entry(0, 0, 65535)); // 0 free
        entries.extend_from_slice(&entry(2, 4, 0)); // 1 → ObjStm 4, idx 0
        entries.extend_from_slice(&entry(2, 4, 1)); // 2 → ObjStm 4, idx 1
        entries.extend_from_slice(&entry(2, 4, 2)); // 3 → ObjStm 4, idx 2 (PAGE)
        entries.extend_from_slice(&entry(1, off4 as u32, 0)); // 4 direct
        entries.extend_from_slice(&entry(1, off5 as u32, 0)); // 5 direct (self)

        let mut xref = format!(
            "5 0 obj\n<</Type/XRef/Size 6/W[1 2 2]/Root 1 0 R/Length {}>>\nstream\n",
            entries.len()
        )
        .into_bytes();
        xref.extend_from_slice(&entries);
        xref.extend_from_slice(b"\nendstream\nendobj\n");

        let mut bytes = Vec::new();
        bytes.extend_from_slice(header);
        bytes.extend_from_slice(&objstm);
        bytes.extend_from_slice(&xref);
        bytes.extend_from_slice(format!("startxref\n{off5}\n%%EOF").as_bytes());

        let doc = Document::from_bytes_owned(bytes).expect("PDF-1.5 xref-stream+ObjStm");

        // Catalog (obj 1) resolves out of the ObjStm.
        let catalog = doc.catalog().expect("catalog via ObjStm");
        assert!(catalog.get(b"Pages".as_slice()).is_some());

        // The page tree descends through ObjStm-resident Pages → Page.
        assert_eq!(doc.page_count(), 1, "one page via xref-stream type-2");
        let pages: Vec<_> = doc.get_pages().collect();
        assert_eq!(pages.len(), 1);

        // The page object itself (obj 3, an xref type-2 entry) resolves.
        let (_, page_id) = pages[0];
        assert_eq!(page_id.0, 3);
        let page = doc.get_object(page_id).expect("in-ObjStm page object");
        let pd = page.as_dict().expect("page is a dict");
        assert_eq!(
            pd.get(b"Type".as_slice()).and_then(Object::as_name),
            Some(&b"Page"[..])
        );
    }

    /// Direct unit test of the xref-stream `/W [1 2 2]` binary entry decoder
    /// for a **type-2** (in-object-stream) record.  Locks the big-endian
    /// field decode + the container/index assignment that the PDF-1.5
    /// resolution path depends on.
    #[test]
    fn xref_stream_w_type2_entry_decode() {
        // Single uncompressed xref stream, obj 1, /Size 2, covering objs 0..1.
        //   obj 0: type 0 (free)        → 00 0000 0000
        //   obj 1: type 2, container 7, index 3 → 02 0007 0003
        let entries: [u8; 10] = [0, 0, 0, 0, 0, 2, 0, 7, 0, 3];
        let body = format!(
            "1 0 obj\n<</Type/XRef/Size 2/W[1 2 2]/Root 9 0 R/Length {}>>\nstream\n",
            entries.len()
        );
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"%PDF-1.5\n");
        let startxref = bytes.len();
        bytes.extend_from_slice(body.as_bytes());
        bytes.extend_from_slice(&entries);
        bytes.extend_from_slice(b"\nendstream\nendobj\n");
        bytes.extend_from_slice(format!("startxref\n{startxref}\n%%EOF").as_bytes());

        let table = crate::xref::read_xref(&bytes).expect("xref stream parse");
        match table.get(1) {
            Some(crate::xref::XrefEntry::InObjStm { container, index }) => {
                assert_eq!(container, 7, "type-2 field2 = ObjStm container number");
                assert_eq!(index, 3, "type-2 field3 = index within ObjStm");
            }
            other => panic!("expected InObjStm entry for obj 1, got {other:?}"),
        }
    }

    /// Regression: a page whose `/Contents` is an *indirect reference to an
    /// array* of content-stream references (legal per ISO 32000-2 §7.7.3.3;
    /// emitted by LibreOffice and many imposition tools).  The pre-fix code
    /// dereferenced the single ref, found an `Object::Array` instead of an
    /// `Object::Stream`, skipped it, and returned an empty content buffer —
    /// every glyph-bearing page rendered pure white.  This asserts the array
    /// is expanded and all member streams are concatenated.
    #[test]
    fn get_page_content_contents_ref_to_array() {
        let header = "%PDF-1.4\n";
        let obj1 = "1 0 obj\n<</Type /Catalog /Pages 2 0 R>>\nendobj\n";
        let obj2 = "2 0 obj\n<</Type /Pages /Kids [3 0 R] /Count 1>>\nendobj\n";
        // /Contents is a single indirect reference (obj 5), NOT an inline array.
        let obj3 = "3 0 obj\n<</Type /Page /Parent 2 0 R /Contents 5 0 R \
                    /MediaBox [0 0 612 792]>>\nendobj\n";
        // obj 5 resolves to an ARRAY of two content-stream references.
        let obj5 = "5 0 obj\n[6 0 R 7 0 R]\nendobj\n";
        let s6 = "q\n";
        let obj6 = format!(
            "6 0 obj\n<</Length {}>>\nstream\n{s6}\nendstream\nendobj\n",
            s6.len()
        );
        let s7 = "BT ET\n";
        let obj7 = format!(
            "7 0 obj\n<</Length {}>>\nstream\n{s7}\nendstream\nendobj\n",
            s7.len()
        );
        let off1 = header.len();
        let off2 = off1 + obj1.len();
        let off3 = off2 + obj2.len();
        // obj 4 is unused; reserve a free slot so xref indices stay simple.
        let off5 = off3 + obj3.len();
        let off6 = off5 + obj5.len();
        let off7 = off6 + obj6.len();
        let xref_start = off7 + obj7.len();
        let xref = format!(
            "xref\n0 8\n0000000000 65535 f\r\n\
             {off1:010} 00000 n\r\n{off2:010} 00000 n\r\n\
             {off3:010} 00000 n\r\n0000000000 65535 f\r\n\
             {off5:010} 00000 n\r\n{off6:010} 00000 n\r\n{off7:010} 00000 n\r\n"
        );
        let trailer = format!("trailer\n<</Size 8 /Root 1 0 R>>\nstartxref\n{xref_start}\n%%EOF");
        let bytes =
            format!("{header}{obj1}{obj2}{obj3}{obj5}{obj6}{obj7}{xref}{trailer}").into_bytes();

        let doc = Document::from_bytes_owned(bytes).expect("contents-ref-to-array PDF");
        let page_id = doc.get_pages().next().expect("one page").1;
        let content = doc.get_page_content(page_id).expect("page content");
        // Both member streams must be present, concatenated with a separator.
        assert!(
            !content.is_empty(),
            "ref-to-array /Contents must not yield empty content (the blank-page bug)"
        );
        let text = String::from_utf8_lossy(&content);
        assert!(
            text.contains("q"),
            "first content stream (obj 6) missing: {text:?}"
        );
        assert!(
            text.contains("BT ET"),
            "second content stream (obj 7) missing: {text:?}"
        );
    }

    /// ISO 32000-2 §7.8.2: the streams of a `/Contents` array are decoded and
    /// concatenated *as if a single stream*, and a stream need not break at a
    /// token boundary.  This builds a page whose two content streams split a
    /// number mid-digits — obj 6 ends `"10 0 0 10 50 7"`, obj 7 begins
    /// `"00 cm ..."`.  Without the inter-stream separator the lexer reads
    /// `"700 cm"` (a corrupt CTM) instead of `"50 700"` then `"cm"`, so the
    /// page transforms wrongly and content lands off-page (silent-wrong, the
    /// v1 sin).  Asserts the LF separator sits exactly at the join.
    #[test]
    fn get_page_content_array_inserts_separator_at_midtoken_split() {
        let header = "%PDF-1.4\n";
        let obj1 = "1 0 obj\n<</Type /Catalog /Pages 2 0 R>>\nendobj\n";
        let obj2 = "2 0 obj\n<</Type /Pages /Kids [3 0 R] /Count 1>>\nendobj\n";
        let obj3 = "3 0 obj\n<</Type /Page /Parent 2 0 R /Contents 9 0 R \
                    /MediaBox [0 0 612 792]>>\nendobj\n";
        let obj9 = "9 0 obj\n[10 0 R 11 0 R]\nendobj\n";
        // Stream 10 ends in the middle of the literal "700": "...50 7".
        let s10 = "q 1 0 0 1 0 0 cm 10 0 0 10 50 7";
        let obj10 = format!(
            "10 0 obj\n<</Length {}>>\nstream\n{s10}\nendstream\nendobj\n",
            s10.len()
        );
        // Stream 11 resumes with "00 cm ...".  Joined raw this reads "700 cm".
        let s11 = "00 cm BT ET Q";
        let obj11 = format!(
            "11 0 obj\n<</Length {}>>\nstream\n{s11}\nendstream\nendobj\n",
            s11.len()
        );
        let off1 = header.len();
        let off2 = off1 + obj1.len();
        let off3 = off2 + obj2.len();
        let off9 = off3 + obj3.len();
        let off10 = off9 + obj9.len();
        let off11 = off10 + obj10.len();
        let xref_start = off11 + obj11.len();
        // Objects 4..8 are free placeholders so xref indices stay literal.
        let xref = format!(
            "xref\n0 12\n0000000000 65535 f\r\n\
             {off1:010} 00000 n\r\n{off2:010} 00000 n\r\n{off3:010} 00000 n\r\n\
             0000000000 65535 f\r\n0000000000 65535 f\r\n0000000000 65535 f\r\n\
             0000000000 65535 f\r\n0000000000 65535 f\r\n\
             {off9:010} 00000 n\r\n{off10:010} 00000 n\r\n{off11:010} 00000 n\r\n"
        );
        let trailer = format!("trailer\n<</Size 12 /Root 1 0 R>>\nstartxref\n{xref_start}\n%%EOF");
        let bytes =
            format!("{header}{obj1}{obj2}{obj3}{obj9}{obj10}{obj11}{xref}{trailer}").into_bytes();

        let doc = Document::from_bytes_owned(bytes).expect("midtoken-split PDF");
        let page_id = doc.get_pages().next().expect("one page").1;
        let content = doc.get_page_content(page_id).expect("page content");
        let text = String::from_utf8_lossy(&content);
        // The separator must keep the "7" and "00" tokens apart: the corrupt
        // glued form "700 cm" must NOT appear, and the correctly separated
        // "50 7\n00 cm" boundary must.
        assert!(
            !text.contains("700 cm"),
            "missing §7.8.2 separator glued mid-token numbers: {text:?}"
        );
        assert!(
            text.contains("50 7\n00 cm"),
            "expected LF separator exactly at the stream join: {text:?}"
        );
    }

    /// A cyclic / self-referential `/Contents`: obj 9 is `[10 0 R 9 0 R]` —
    /// the second element points back at the array object itself.  A naive
    /// recursive expander would loop forever (the campaign's DoS class).  The
    /// expander is a single non-recursive pass, so this must terminate
    /// promptly: obj 10's stream renders, and the back-reference (which
    /// resolves to an Array, not a Stream) is logged-and-skipped rather than
    /// recursed into or silently lost.
    #[test]
    fn get_page_content_cyclic_contents_terminates() {
        let header = "%PDF-1.4\n";
        let obj1 = "1 0 obj\n<</Type /Catalog /Pages 2 0 R>>\nendobj\n";
        let obj2 = "2 0 obj\n<</Type /Pages /Kids [3 0 R] /Count 1>>\nendobj\n";
        let obj3 = "3 0 obj\n<</Type /Page /Parent 2 0 R /Contents 9 0 R \
                    /MediaBox [0 0 612 792]>>\nendobj\n";
        // The array's 2nd element (9 0 R) is the array object itself.
        let obj9 = "9 0 obj\n[10 0 R 9 0 R]\nendobj\n";
        let s10 = "q BT ET Q";
        let obj10 = format!(
            "10 0 obj\n<</Length {}>>\nstream\n{s10}\nendstream\nendobj\n",
            s10.len()
        );
        let off1 = header.len();
        let off2 = off1 + obj1.len();
        let off3 = off2 + obj2.len();
        let off9 = off3 + obj3.len();
        let off10 = off9 + obj9.len();
        let xref_start = off10 + obj10.len();
        let xref = format!(
            "xref\n0 11\n0000000000 65535 f\r\n\
             {off1:010} 00000 n\r\n{off2:010} 00000 n\r\n{off3:010} 00000 n\r\n\
             0000000000 65535 f\r\n0000000000 65535 f\r\n0000000000 65535 f\r\n\
             0000000000 65535 f\r\n0000000000 65535 f\r\n\
             {off9:010} 00000 n\r\n{off10:010} 00000 n\r\n"
        );
        let trailer = format!("trailer\n<</Size 11 /Root 1 0 R>>\nstartxref\n{xref_start}\n%%EOF");
        let bytes = format!("{header}{obj1}{obj2}{obj3}{obj9}{obj10}{xref}{trailer}").into_bytes();

        let doc = Document::from_bytes_owned(bytes).expect("cyclic-/Contents PDF");
        let page_id = doc.get_pages().next().expect("one page").1;
        // Must return (not hang): the valid stream is kept, the self-ref is
        // skipped because it resolves to an Array rather than a Stream.
        let content = doc.get_page_content(page_id).expect("page content");
        let text = String::from_utf8_lossy(&content);
        assert!(
            text.contains("q BT ET Q"),
            "valid content stream must still render: {text:?}"
        );
    }

    // ── Input-boundary validation ────────────────────────────────────────

    /// `Document` is intentionally not `Debug`, so `Result::unwrap_err` is
    /// unavailable; pull the error out by hand.
    fn open_err(bytes: Vec<u8>) -> PdfError {
        match Document::from_bytes_owned(bytes) {
            Ok(_) => panic!("expected an error, document opened successfully"),
            Err(e) => e,
        }
    }

    #[test]
    fn empty_input_is_diagnosed_distinctly() {
        let err = open_err(Vec::new());
        assert!(
            matches!(err, PdfError::EmptyInput),
            "0-byte input must yield EmptyInput, got: {err}"
        );
        assert!(err.to_string().contains("empty"));
    }

    #[test]
    fn non_pdf_input_is_diagnosed_distinctly() {
        let err = open_err(b"just some text, not a pdf\n".to_vec());
        assert!(
            matches!(err, PdfError::NotPdf),
            "non-PDF input must yield NotPdf, got: {err}"
        );
        // Binary garbage with no %PDF- header takes the same path.
        let err2 = open_err(vec![0u8; 2048]);
        assert!(matches!(err2, PdfError::NotPdf), "2KB NUL: {err2}");
    }

    #[test]
    fn header_only_without_xref_is_truncated_not_xref_jargon() {
        // Has a valid %PDF- header but no startxref/trailer/objects: the
        // operator must hear "truncated or corrupt", never the raw
        // "'startxref' not found" jargon.
        let err = open_err(b"%PDF-1.7\n".to_vec());
        assert!(
            matches!(err, PdfError::MissingXref(_)),
            "header-only must yield MissingXref, got: {err}"
        );
        assert!(err.to_string().contains("truncated or corrupt"));
    }

    #[test]
    fn header_late_within_scan_window_is_accepted_as_pdf() {
        // The %PDF- signature need not be at byte 0; a few junk bytes are
        // tolerated (matches poppler/mutool). It still has no xref, so the
        // failure is the truncation diagnosis, *not* NotPdf.
        let mut bytes = b"\xEF\xBB\xBFleading junk\n".to_vec();
        bytes.extend_from_slice(b"%PDF-1.4\n");
        let err = open_err(bytes);
        assert!(
            matches!(err, PdfError::MissingXref(_)),
            "header-after-junk must pass the NotPdf gate, got: {err}"
        );
    }

    #[test]
    fn broken_xref_is_reconstructed_from_object_scan() {
        // A structurally complete body with NO xref table and NO startxref
        // at all. The strict parser cannot find it; reconstruction must
        // rebuild the table from the object headers and synthesise a
        // trailer pointing at the /Catalog so the document opens.
        let body = b"%PDF-1.4\n\
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n\
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n\
3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 200 200]>>endobj\n"
            .to_vec();
        let doc = Document::from_bytes_owned(body)
            .expect("reconstruction must recover a usable catalogue");
        assert_eq!(doc.page_count(), 1, "rebuilt page tree must yield 1 page");
    }

    #[test]
    fn bare_catalog_stub_without_pages_is_not_falsely_recovered() {
        // A truncated file whose only object is a Catalog with no /Pages is
        // genuinely unrecoverable; repair must decline and report the
        // truncation, not the misleading "document has no pages".
        let body = b"%PDF-1.7\n1 0 obj<</Type/Catalog>>endobj\n".to_vec();
        let err = open_err(body);
        assert!(
            matches!(err, PdfError::MissingXref(_)),
            "bare-Catalog stub must yield MissingXref, got: {err}"
        );
    }
}
