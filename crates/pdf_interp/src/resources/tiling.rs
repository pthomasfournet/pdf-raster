//! PDF tiling pattern resource resolution (`PatternType` 1).
//!
//! A tiling pattern is a content stream that is repeatedly stamped across the
//! painting area at fixed `XStep` × `YStep` intervals (PDF §8.7.3).
//!
//! # Supported paint types
//!
//! | `PaintType` | Name | Support |
//! |---|---|---|
//! | 1 | Coloured | yes — tile is rasterised at its native colours |
//! | 2 | Uncoloured | partial — tint components are resolved to a solid colour and
//!   applied as a fill override; full per-path tint application is not implemented |
//!
//! # Coordinate system
//!
//! The pattern's `Matrix` transforms from pattern space into user space; this is
//! then combined with the caller's CTM to reach device space.  The tile is
//! rasterised at the scale implied by `ctm * matrix` so that one pattern-space
//! unit maps to the correct number of device pixels.

use lopdf::{Dictionary, Document, Object, ObjectId};

use super::image::resolve_dict;

/// Parameters extracted from a PDF Type 1 (tiling) pattern stream.
///
/// Callers use these to rasterise a tile bitmap and assemble a
/// [`crate::renderer::page::TiledPattern`].
pub struct TilingDescriptor {
    /// Content stream of the pattern tile (already decompressed).
    pub content: Vec<u8>,
    /// Object ID of the pattern stream — used as the resource context for fonts
    /// and sub-objects referenced from the tile content.
    pub stream_id: ObjectId,
    /// True if the pattern stream carries its own `Resources` dict; false means
    /// the tile inherits from the parent context (rare but allowed by the spec).
    pub has_own_resources: bool,
    /// Pattern bounding box `[xmin ymin xmax ymax]` in pattern space.
    pub bbox: [f64; 4],
    /// X spacing between tile origins in pattern space.
    pub x_step: f64,
    /// Y spacing between tile origins in pattern space.
    pub y_step: f64,
    /// Pattern-to-user-space matrix (default: identity).
    pub matrix: [f64; 6],
    /// `PaintType`: 1 = coloured, 2 = uncoloured.
    pub paint_type: i64,
}

/// Look up a named `Pattern` resource and extract its tiling descriptor.
///
/// `ctx_dict` is the dictionary of the current resource context (page or form
/// stream object).  Returns `None` if the name is absent, the object is not a
/// tiling pattern, or any required key is missing.
#[must_use]
pub fn resolve_tiling(
    doc: &Document,
    ctx_dict: &Dictionary,
    name: &[u8],
) -> Option<TilingDescriptor> {
    // Navigate Resources → Pattern → name.
    let res = resolve_dict(doc, ctx_dict.get(b"Resources").ok()?)?;
    let pat_dict = resolve_dict(doc, res.get(b"Pattern").ok()?)?;
    let pat_obj = pat_dict.get(name).ok()?;

    let stream_id = if let Object::Reference(id) = pat_obj {
        *id
    } else {
        log::debug!(
            "pdf_interp: Pattern /{} is not a reference — skipping",
            String::from_utf8_lossy(name)
        );
        return None;
    };

    let stream = doc.get_object(stream_id).ok()?.as_stream().ok()?;

    // Must be `PatternType` 1 (tiling).
    let pattern_type = stream.dict.get(b"PatternType").ok()?.as_i64().ok()?;
    if pattern_type != 1 {
        log::debug!(
            "pdf_interp: Pattern /{} has PatternType {pattern_type} (only 1 supported for tiling)",
            String::from_utf8_lossy(name)
        );
        return None;
    }

    let paint_type = stream.dict.get(b"PaintType").ok()?.as_i64().ok()?;

    let bbox = read_bbox(&stream.dict)?;
    let x_step = read_real(&stream.dict, b"XStep")?;
    let y_step = read_real(&stream.dict, b"YStep")?;

    if x_step == 0.0 || y_step == 0.0 {
        log::warn!(
            "pdf_interp: Pattern /{} has zero XStep or YStep — skipping",
            String::from_utf8_lossy(name)
        );
        return None;
    }

    let matrix = read_matrix(&stream.dict).unwrap_or([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);

    let content = stream.decompressed_content().ok()?;
    let has_own_resources = stream.dict.get(b"Resources").is_ok();

    Some(TilingDescriptor {
        content,
        stream_id,
        has_own_resources,
        bbox,
        x_step,
        y_step,
        matrix,
        paint_type,
    })
}

// ── Private helpers ───────────────────────────────────────────────────────────

fn read_bbox(dict: &lopdf::Dictionary) -> Option<[f64; 4]> {
    let arr = dict.get(b"BBox").ok()?.as_array().ok()?;
    if arr.len() < 4 {
        return None;
    }
    let mut b = [0.0f64; 4];
    for (i, obj) in arr.iter().take(4).enumerate() {
        b[i] = obj_to_f64(obj)?;
    }
    Some(b)
}

fn read_matrix(dict: &lopdf::Dictionary) -> Option<[f64; 6]> {
    let arr = dict.get(b"Matrix").ok()?.as_array().ok()?;
    if arr.len() < 6 {
        return None;
    }
    let mut m = [0.0f64; 6];
    for (i, obj) in arr.iter().take(6).enumerate() {
        m[i] = obj_to_f64(obj)?;
    }
    Some(m)
}

fn read_real(dict: &lopdf::Dictionary, key: &[u8]) -> Option<f64> {
    obj_to_f64(dict.get(key).ok()?)
}

fn obj_to_f64(obj: &Object) -> Option<f64> {
    match obj {
        Object::Real(r) => Some(f64::from(*r)),
        #[expect(
            clippy::cast_precision_loss,
            reason = "pattern step and bbox values are small; precision loss is negligible"
        )]
        Object::Integer(n) => Some(*n as f64),
        _ => None,
    }
}
