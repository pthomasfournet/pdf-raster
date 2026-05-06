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
    error::PdfError,
    object::{Object, ObjectId, parse_indirect_object},
    objstm::ObjStmCache,
    stream::decode_stream,
    xref::{XrefEntry, XrefTable, read_xref},
};

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
        let file = std::fs::File::open(path)?;
        // SAFETY: we do not mutate the file while it is mapped.
        let mmap = unsafe { Mmap::map(&file)? };
        let data = Arc::new(MmapOrVec::Mapped(mmap));
        Self::from_bytes(data)
    }

    /// Construct from an in-memory byte slice (used by tests).
    pub fn from_bytes_owned(bytes: Vec<u8>) -> Result<Self, PdfError> {
        Self::from_bytes(Arc::new(MmapOrVec::Owned(bytes)))
    }

    fn from_bytes(data: Arc<MmapOrVec>) -> Result<Self, PdfError> {
        let xref = read_xref(data.as_bytes())?;
        Ok(Self {
            data,
            xref,
            cache: Mutex::new(HashMap::new()),
            objstm: ObjStmCache::new(),
            catalog_id: OnceLock::new(),
        })
    }

    // ── Object resolution ────────────────────────────────────────────────────

    /// Resolve an indirect reference and return a shared reference to its value.
    ///
    /// First call parses the object from the file; subsequent calls return the
    /// cached value.
    pub fn get_object(&self, id: ObjectId) -> Result<Arc<Object>, PdfError> {
        {
            let guard = self.cache.lock().unwrap();
            if let Some(obj) = guard.get(&id.0) {
                return Ok(Arc::clone(obj));
            }
        }

        let obj = self.parse_object_uncached(id)?;
        let arc = Arc::new(obj);
        self.cache.lock().unwrap().insert(id.0, Arc::clone(&arc));
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
    pub fn get_dict(&self, id: ObjectId) -> Result<Arc<HashMap<Vec<u8>, Object>>, PdfError> {
        let obj = self.get_object(id)?;
        match obj.as_ref() {
            Object::Dictionary(_) | Object::Stream(_) => {
                // Return the dict without cloning by mapping the Arc.
                Ok(Arc::new(obj.as_dict().unwrap().clone()))
            }
            _ => Err(PdfError::BadObject {
                id: id.0,
                detail: "not a dictionary".into(),
            }),
        }
    }

    // ── Document structure ────────────────────────────────────────────────────

    /// Return the document catalogue dictionary.
    pub fn catalog(&self) -> Result<Arc<HashMap<Vec<u8>, Object>>, PdfError> {
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
            .get(b"Root".as_ref())
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

    /// Return the number of pages in the document.
    pub fn page_count(&self) -> u32 {
        self.get_pages().count() as u32
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

        let contents = match page_dict.get(b"Contents".as_ref()) {
            None => return Ok(Vec::new()),
            Some(obj) => obj.clone(),
        };

        // /Contents may be a single reference or an array of references.
        let refs: Vec<ObjectId> = match &contents {
            Object::Reference(id) => vec![*id],
            Object::Array(arr) => arr.iter().filter_map(Object::as_reference).collect(),
            _ => return Ok(Vec::new()),
        };

        let mut out = Vec::new();
        for r in refs {
            let stream_obj = self.get_object(r)?;
            if let Object::Stream(s) = stream_obj.as_ref() {
                let decoded = decode_stream(&s.content, &s.dict).map_err(PdfError::DecodeFailed)?;
                out.extend_from_slice(&decoded);
            }
        }
        Ok(out)
    }

    /// Return a merged font dictionary for the page (all /Font entries from
    /// /Resources, with parent-chain inheritance).
    pub fn get_page_fonts(&self, page_id: ObjectId) -> Result<HashMap<Vec<u8>, Object>, PdfError> {
        self.get_page_resource_dict(page_id, b"Font")
    }

    /// Return one named sub-dictionary from the page's /Resources, following
    /// parent page-tree inheritance.
    pub fn get_page_resource_dict(
        &self,
        page_id: ObjectId,
        resource_name: &[u8],
    ) -> Result<HashMap<Vec<u8>, Object>, PdfError> {
        // Walk up the page tree to find /Resources.
        let mut current_id = Some(page_id);
        while let Some(id) = current_id {
            let obj = self.get_object(id)?;
            let dict = match obj.as_dict() {
                Some(d) => d.clone(),
                None => break,
            };

            if let Some(res) = dict.get(b"Resources".as_ref()) {
                let res_dict = match res {
                    Object::Dictionary(d) => d.clone(),
                    Object::Reference(rid) => {
                        let r = self.get_object(*rid)?;
                        r.as_dict().cloned().unwrap_or_default()
                    }
                    _ => HashMap::new(),
                };
                if let Some(sub) = res_dict.get(resource_name) {
                    let sub_dict = match sub {
                        Object::Dictionary(d) => d.clone(),
                        Object::Reference(rid) => {
                            let r = self.get_object(*rid)?;
                            r.as_dict().cloned().unwrap_or_default()
                        }
                        _ => HashMap::new(),
                    };
                    return Ok(sub_dict);
                }
            }

            // Walk to parent.
            current_id = dict.get(b"Parent".as_ref()).and_then(Object::as_reference);
        }
        Ok(HashMap::new())
    }
}

// ── Page iterator ─────────────────────────────────────────────────────────────

struct PageIter<'a> {
    doc: &'a Document,
    /// Stack of (node_id, next_child_index) for page-tree traversal.
    stack: Vec<(ObjectId, usize)>,
    page_num: u32,
}

impl<'a> PageIter<'a> {
    fn new(doc: &'a Document) -> Self {
        let catalog = doc.catalog().unwrap_or_default();
        let pages_id = catalog
            .get(b"Pages".as_ref())
            .and_then(Object::as_reference);
        let stack = pages_id.map(|id| vec![(id, 0)]).unwrap_or_default();
        Self {
            doc,
            stack,
            page_num: 0,
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
                    self.stack.pop();
                    continue;
                }
            };
            let dict = match obj.as_dict() {
                Some(d) => d.clone(),
                None => {
                    self.stack.pop();
                    continue;
                }
            };

            let node_type = dict
                .get(b"Type".as_ref())
                .and_then(Object::as_name)
                .unwrap_or(b"");

            if node_type == b"Page" {
                // Leaf page node.
                self.stack.pop();
                self.page_num += 1;
                return Some((self.page_num, node_id));
            }

            // Pages node: iterate /Kids.
            let kids = match dict.get(b"Kids".as_ref()) {
                Some(Object::Array(a)) => a.clone(),
                _ => {
                    self.stack.pop();
                    continue;
                }
            };

            let idx = *child_idx;
            if idx >= kids.len() {
                self.stack.pop();
                continue;
            }
            *child_idx += 1;

            if let Some(kid_id) = kids[idx].as_reference() {
                self.stack.push((kid_id, 0));
            }
        }
    }
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
    fn get_pages_returns_one_page() {
        let doc = Document::from_bytes_owned(minimal_pdf()).unwrap();
        let pages: Vec<_> = doc.get_pages().collect();
        assert_eq!(pages.len(), 1);
        assert_eq!(pages[0].0, 1);
    }
}
