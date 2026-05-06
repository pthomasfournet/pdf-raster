//! Object-stream cache.
//!
//! PDF 1.5+ can pack multiple small objects into a single `/Type /ObjStm`
//! stream to save space.  A Type-2 xref entry says "object N is at index K
//! inside container object M."
//!
//! This module decompresses a container stream exactly once and caches all N
//! objects it contains.  Subsequent look-ups for different indices in the same
//! container are O(1) cache hits.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use crate::{
    error::PdfError,
    lexer::{parse_u64, skip_ws},
    object::{Object, parse_object},
    stream::decode_stream,
};

/// Cache of decompressed and parsed object streams, keyed by container object number.
#[derive(Default)]
pub(crate) struct ObjStmCache {
    cache: Mutex<HashMap<u32, Arc<Vec<Object>>>>,
}

impl ObjStmCache {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Return the object at `index` within the object stream whose compressed
    /// form is `stream_content` with dictionary `stream_dict`.
    pub(crate) fn get_or_parse(
        &self,
        container_id: u32,
        index: u32,
        stream_content: &[u8],
        stream_dict: &HashMap<Vec<u8>, Object>,
    ) -> Result<Object, PdfError> {
        // Poison recovery: cache stores immutable parses, safe to reuse.
        let mut guard = self.cache.lock().unwrap_or_else(|e| e.into_inner());

        if let Some(objects) = guard.get(&container_id) {
            let objects = Arc::clone(objects);
            drop(guard);
            return objects
                .get(index as usize)
                .cloned()
                .ok_or_else(|| PdfError::BadObject {
                    id: container_id,
                    detail: format!("index {index} out of range"),
                });
        }

        // Decompress the stream.
        let decoded =
            decode_stream(stream_content, stream_dict).map_err(|e| PdfError::BadObject {
                id: container_id,
                detail: e,
            })?;

        // Parse the object stream directory: N pairs of (obj_num, byte_offset).
        let n_raw = stream_dict
            .get(b"N".as_ref())
            .and_then(Object::as_i64)
            .unwrap_or(0);
        if !(0..=1_000_000).contains(&n_raw) {
            return Err(PdfError::BadObject {
                id: container_id,
                detail: format!("/N={n_raw} out of range"),
            });
        }
        let n = n_raw as usize;

        let first_raw = stream_dict
            .get(b"First".as_ref())
            .and_then(Object::as_i64)
            .unwrap_or(0);
        if first_raw < 0 {
            return Err(PdfError::BadObject {
                id: container_id,
                detail: format!("/First={first_raw} is negative"),
            });
        }
        let first = first_raw as usize;

        let objects =
            parse_objstm_objects(&decoded, n, first).map_err(|detail| PdfError::BadObject {
                id: container_id,
                detail,
            })?;

        let arc = Arc::new(objects);
        guard.insert(container_id, Arc::clone(&arc));
        drop(guard);

        arc.get(index as usize)
            .cloned()
            .ok_or_else(|| PdfError::BadObject {
                id: container_id,
                detail: format!("index {index} out of range (n={n})"),
            })
    }
}

/// Parse all N objects from a decompressed object stream.
fn parse_objstm_objects(data: &[u8], n: usize, first: usize) -> Result<Vec<Object>, String> {
    if first > data.len() {
        return Err(format!(
            "/First={first} extends past stream length {}",
            data.len()
        ));
    }

    // Read the directory: N pairs of (object_number, relative_byte_offset).
    let mut pos = 0usize;
    let mut offsets = Vec::with_capacity(n);
    for _ in 0..n {
        skip_ws(data, &mut pos);
        let _obj_num = parse_u64(data, &mut pos)
            .ok_or_else(|| format!("ObjStm directory: missing object number at pos {pos}"))?;
        skip_ws(data, &mut pos);
        let rel_off = parse_u64(data, &mut pos)
            .ok_or_else(|| format!("ObjStm directory: missing offset at pos {pos}"))?
            as usize;
        let abs_off = first
            .checked_add(rel_off)
            .ok_or_else(|| format!("ObjStm: offset overflow first={first} rel={rel_off}"))?;
        if abs_off > data.len() {
            return Err(format!(
                "ObjStm: object offset {abs_off} exceeds stream length {}",
                data.len()
            ));
        }
        offsets.push(abs_off);
    }

    // Parse each object at its computed offset.
    let mut objects = Vec::with_capacity(n);
    for offset in offsets {
        let mut obj_pos = offset;
        let obj = parse_object(data, &mut obj_pos)
            .ok_or_else(|| format!("ObjStm: failed to parse object at offset {offset}"))?;
        objects.push(obj);
    }
    Ok(objects)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_simple_objstm() {
        // Directory: two objects. First=10.
        // "1 0 2 5"  (obj 1 at offset 0, obj 2 at offset 5 relative to First=10)
        // Data at offset 10: "42 true"
        let mut data = b"1 0 2 5   ".to_vec(); // 10 bytes header, First=10
        data.extend_from_slice(b"42   "); // obj at first+0 = 10, length 5
        data.extend_from_slice(b"true "); // obj at first+5 = 15
        let objs = parse_objstm_objects(&data, 2, 10).unwrap();
        assert_eq!(objs[0], Object::Integer(42));
        assert_eq!(objs[1], Object::Boolean(true));
    }
}
