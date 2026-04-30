//! Annotation rendering helpers for [`PageRenderer`].
//!
//! Free functions taking `renderer: &mut PageRenderer<'_>`, delegated from
//! thin wrapper methods in `mod.rs`.

use lopdf::{Object, ObjectId};

use super::super::gstate::ctm_multiply;
use super::PageRenderer;

/// Render all annotations on the page (PDF §12.5).
///
/// Each annotation with an `AP/N` (normal appearance) stream is rendered
/// after the page content stream.  Only the `/N` entry is used; rollover
/// (`/R`) and down (`/D`) appearances are ignored (this is a rasterizer,
/// not an interactive viewer).
///
/// Annotations with no appearance stream are silently skipped — many
/// annotation types (e.g. `Link`) have no visible appearance by default.
///
/// Call this after `execute` and before `finish`.
pub(super) fn render_annotations(renderer: &mut PageRenderer<'_>, page_id: ObjectId) {
    let doc = renderer.resources.doc();
    let Ok(page_dict) = doc.get_dictionary(page_id) else {
        log::warn!(
            "pdf_interp: render_annotations: page object ({}, {}) is not a dictionary — \
             skipping annotations",
            page_id.0,
            page_id.1
        );
        return;
    };

    // Collect annotation refs to avoid borrow conflict.
    let annot_ids: Vec<ObjectId> = {
        let arr = match page_dict.get(b"Annots") {
            Ok(Object::Array(a)) => a.clone(),
            Ok(Object::Reference(id)) => {
                if let Some(a) = doc
                    .get_object(*id)
                    .ok()
                    .and_then(|o| o.as_array().ok().cloned())
                {
                    a
                } else {
                    log::warn!(
                        "pdf_interp: render_annotations: /Annots reference ({}, {}) \
                         did not resolve to an array — skipping annotations",
                        id.0,
                        id.1
                    );
                    return;
                }
            }
            _ => return,
        };
        arr.iter()
            .filter_map(|o| {
                if let Object::Reference(id) = o {
                    Some(*id)
                } else {
                    None
                }
            })
            .collect()
    };

    for annot_id in annot_ids {
        render_one_annotation(renderer, annot_id);
    }
}

/// Render a single annotation object into `renderer`'s bitmap.
pub(super) fn render_one_annotation(renderer: &mut PageRenderer<'_>, annot_id: ObjectId) {
    let doc = renderer.resources.doc();

    let Ok(annot_dict) = doc.get_dictionary(annot_id) else {
        log::warn!(
            "pdf_interp: annotation ({}, {}) object missing or not a dictionary — skipping",
            annot_id.0,
            annot_id.1
        );
        return;
    };

    // Check for AP/N before Rect — annotations with no AP stream have no visible
    // appearance (e.g. Link) and are expected to be absent silently.
    let Some(ap_dict) = annot_dict.get(b"AP").ok().and_then(|o| match o {
        Object::Dictionary(d) => Some(d),
        Object::Reference(id) => doc.get_dictionary(*id).ok(),
        _ => None,
    }) else {
        return;
    };

    // Annotation rect in page user space: [llx, lly, urx, ury].
    // Warn when AP exists but Rect is absent/malformed — visible annotation dropped.
    let Some(rect) = read_rect(annot_dict) else {
        log::warn!(
            "pdf_interp: annotation ({}, {}) has AP/N stream but missing or malformed Rect — skipping",
            annot_id.0,
            annot_id.1
        );
        return;
    };

    // N can be a stream reference or a sub-dict (state-keyed appearances).
    let stream_id: ObjectId = {
        let Ok(n_obj) = ap_dict.get(b"N") else { return };
        match n_obj {
            Object::Reference(id) => *id,
            Object::Dictionary(_) => {
                // State-keyed: look up the current appearance state (AS).
                let state = annot_dict.get(b"AS").ok().and_then(|o| o.as_name().ok());
                let Some(state_key) = state else { return };
                match n_obj.as_dict().ok().and_then(|d| d.get(state_key).ok()) {
                    Some(Object::Reference(id)) => *id,
                    _ => return,
                }
            }
            _ => return,
        }
    };

    // Build the FormXObject from the appearance stream.
    let Some(mut form) = renderer.resources.form_from_stream_id(stream_id) else {
        return;
    };

    // form.bbox was populated by form_from_stream_id; appearance streams carry BBox not Rect.
    let [llx, lly, urx, ury] = rect;
    let [bx0, by0, bx1, by1] = form.bbox;
    let bw = bx1 - bx0;
    let bh = by1 - by0;
    if bw.abs() < f64::EPSILON || bh.abs() < f64::EPSILON {
        return;
    }

    let sx = (urx - llx) / bw;
    let sy = (ury - lly) / bh;
    let tx = bx0.mul_add(-sx, llx);
    let ty = by0.mul_add(-sy, lly);
    let bbox_to_rect: [f64; 6] = [sx, 0.0, 0.0, sy, tx, ty];

    // Composed rendering matrix: stream.Matrix × bbox_to_rect.
    form.matrix = ctm_multiply(&form.matrix, &bbox_to_rect);

    renderer.do_form_xobject(&form);
}

/// Read a 4-element rect `[llx, lly, urx, ury]` from a PDF dictionary.
///
/// Returns `None` if the `Rect` key is absent or has fewer than 4 numeric entries.
pub(super) fn read_rect(dict: &lopdf::Dictionary) -> Option<[f64; 4]> {
    let mut r = crate::resources::read_f64_n::<4>(dict, b"Rect")?;
    // Normalise so llx ≤ urx and lly ≤ ury.
    if r[0] > r[2] {
        r.swap(0, 2);
    }
    if r[1] > r[3] {
        r.swap(1, 3);
    }
    Some(r)
}
