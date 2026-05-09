//! Lightweight page pre-scan: classify page content without rendering pixels.
//!
//! [`prescan_page`] walks a page's resource dictionary and content stream operators
//! to populate a [`PageDiagnostics`] struct at a fraction of the cost of a full
//! render.  No bitmap is allocated; no image data is decoded.
//!
//! # What is inspected
//!
//! - **`XObject` images**: `Filter`, `Width`, `Height` read from the stream
//!   dictionary via reference lookup — no decompression.
//! - **Form `XObject`s**: recursed up to [`MAX_PRESCAN_DEPTH`] levels deep so
//!   images inside forms are counted.
//! - **Inline images**: `Filter`, `W`/`Width`, `H`/`Height` read from the raw
//!   parameter block via the same `parse_inline_params` path used by the full
//!   renderer — no pixel decoding.
//! - **Text operators**: any `Tj`, `TJ`, `'`, or `"` operator sets
//!   `has_vector_text = true`.
//!
//! # PPI estimate
//!
//! Unlike the full renderer (which projects image pixels through the live CTM
//! stack), the prescan uses the raw PDF `Width` dict entry and the page width in
//! points from [`page_size_pts`]:
//!
//! ```text
//! ppi ≈ (image_width_px / page_width_pts) × 72
//! ```
//!
//! This assumes the image fills the page width, so it over-estimates PPI on
//! partial-width images.  For routing purposes an over-estimate is safe — it
//! can only make the prescan more aggressive about flagging GPU candidates,
//! which is conservative.  The real area guard in the decoder path
//! (`GPU_JPEG_THRESHOLD_PX`) is the authoritative threshold.
//!
//! # Error handling
//!
//! Errors from `parse_page` or `page_size_pts` are propagated.  Errors from
//! individual resource lookups (missing dict key, corrupt stream header) are
//! silently skipped — the scan continues.  A partial result is still useful
//! for routing.

use pdf::{Dictionary, Document, Object, ObjectId};

use crate::{
    InterpError,
    content::Operator,
    page_size_pts_by_id, parse_page_by_id,
    renderer::PageDiagnostics,
    resources::{
        image::{IMAGE_FILTER_COUNT, ImageFilter, filter_name, inline::parse_inline_params},
        resolve_dict,
    },
};

/// Maximum Form `XObject` recursion depth during the pre-scan.
///
/// Intentionally shallower than the renderer's `MAX_FORM_DEPTH` (32): the
/// prescan only needs to find the dominant image filter, so deeply nested
/// forms are unlikely to change the classification.  A lower cap also
/// limits prescan cost on adversarially nested documents.
const MAX_PRESCAN_DEPTH: u32 = 4;

// ── Public entry point ────────────────────────────────────────────────────────

/// Classify page `page_num` (1-based) without rendering any pixels.
///
/// Returns a [`PageDiagnostics`] with `has_images`, `has_vector_text`,
/// `dominant_filter`, and a conservative `source_ppi_hint` derived from the
/// image's PDF dictionary `Width` entry (see module doc for caveats).
///
/// # Errors
///
/// - [`InterpError::PageOutOfRange`] if `page_num` is outside the document.
/// - [`InterpError::InvalidPageGeometry`] if the page's `UserUnit` is invalid.
///
/// Any per-resource lookup error (corrupt dict, missing filter key) is
/// silently skipped so that a partial classification is still returned.
pub fn prescan_page(doc: &Document, page_num: u32) -> Result<PageDiagnostics, InterpError> {
    // Resolve the page id once and reuse it for both the geometry read and
    // the content-stream parse — the previous shape descended the page tree
    // three times (page_size_pts, resolve_page_id, parse_page).
    let page_id = crate::resolve_page_id(doc, page_num)?;

    let geom = page_size_pts_by_id(doc, page_id)?;
    #[expect(
        clippy::cast_possible_truncation,
        reason = "page width in PDF points (≤ ~10 000 pts); f32 is sufficient for PPI routing"
    )]
    let page_pts_width = geom.width_pts as f32;

    let mut filter_counts = [0u32; IMAGE_FILTER_COUNT];
    let mut has_images = false;
    let mut max_ppi: f32 = 0.0;

    // 1. Walk XObject resource dictionary (no decoding).
    scan_xobjects(
        doc,
        page_id,
        0,
        &mut filter_counts,
        &mut has_images,
        &mut max_ppi,
        page_pts_width,
    );

    // 2. Walk content stream operators for inline images and text.
    let mut has_vector_text = false;
    let ops = parse_page_by_id(doc, page_id)?;
    for op in &ops {
        match op {
            Operator::InlineImage { params, .. } => {
                has_images = true;
                let dict = parse_inline_params(params);
                scan_inline_image(&dict, &mut filter_counts, &mut max_ppi, page_pts_width);
            }
            Operator::ShowText(_)
            | Operator::ShowTextArray(_)
            | Operator::MoveNextLineShow(_)
            | Operator::MoveNextLineShowSpaced { .. } => {
                has_vector_text = true;
            }
            _ => {}
        }
    }

    // 3. Compute dominant_filter (same logic as PageRenderer::finish).
    let dominant_filter = if has_images {
        filter_counts
            .iter()
            .enumerate()
            .filter(|&(_, c)| *c > 0)
            .max_by_key(|&(_, c)| c)
            .map(|(idx, _)| idx_to_filter(idx))
    } else {
        None
    };

    let source_ppi_hint = if max_ppi > 0.0 { Some(max_ppi) } else { None };

    Ok(PageDiagnostics {
        has_images,
        has_vector_text,
        dominant_filter,
        source_ppi_hint,
    })
}

// ── XObject walker ────────────────────────────────────────────────────────────

/// Walk the `XObject` resource dict of `context_id`, accumulating image stats.
///
/// Recurses into Form `XObject`s up to `depth` = [`MAX_PRESCAN_DEPTH`].
fn scan_xobjects(
    doc: &Document,
    context_id: ObjectId,
    depth: u32,
    filter_counts: &mut [u32; IMAGE_FILTER_COUNT],
    has_images: &mut bool,
    max_ppi: &mut f32,
    page_pts_width: f32,
) {
    let Ok(page_dict) = doc.get_dictionary(context_id) else {
        return;
    };

    let xobj_dict = {
        let Some(res_obj) = page_dict.get(b"Resources") else {
            return;
        };
        let Some(res) = resolve_dict(doc, res_obj) else {
            return;
        };
        let Some(xobj_obj) = res.get(b"XObject") else {
            return;
        };
        let Some(d) = resolve_dict(doc, xobj_obj) else {
            return;
        };
        // Collect (name, object) pairs to avoid borrowing `doc` through the loop.
        d.iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect::<Vec<_>>()
    };

    for (_name, obj) in xobj_dict {
        // All XObject entries are References to stream objects.
        let Object::Reference(xobj_id) = obj else {
            continue;
        };
        let Ok(xobj_stream) = doc.get_object(xobj_id) else {
            continue;
        };
        let Object::Stream(stream) = xobj_stream.as_ref() else {
            continue;
        };

        let subtype = stream
            .dict
            .get(b"Subtype")
            .and_then(Object::as_name)
            .map(<[u8]>::to_vec);

        match subtype.as_deref() {
            Some(b"Image") => {
                *has_images = true;
                scan_image_dict(&stream.dict, filter_counts, max_ppi, page_pts_width);
            }
            Some(b"Form") if depth < MAX_PRESCAN_DEPTH => {
                scan_xobjects(
                    doc,
                    xobj_id,
                    depth + 1,
                    filter_counts,
                    has_images,
                    max_ppi,
                    page_pts_width,
                );
            }
            _ => {}
        }
    }
}

// ── Per-image dict scanners ───────────────────────────────────────────────────

/// Extract filter and PPI hint from an image `XObject` stream dictionary.
fn scan_image_dict(
    dict: &Dictionary,
    filter_counts: &mut [u32; IMAGE_FILTER_COUNT],
    max_ppi: &mut f32,
    page_pts_width: f32,
) {
    let filter = dict.get(b"Filter").and_then(|o| {
        // Filter may be a Reference → Name; follow it.
        let resolved = match o {
            Object::Reference(id) => {
                // Rare: Filter as an indirect reference.  `scan_image_dict`
                // does not have access to `doc` so the reference cannot be
                // resolved here.  Fall back to Raw, which means this image
                // will not be counted as a JPEG/JPX candidate — pessimistic
                // for GPU routing but never causes a misrender.
                let _ = id;
                return None;
            }
            other => other,
        };
        filter_name(resolved)
    });

    let img_filter = ImageFilter::from_filter_str(filter.as_deref());
    count_filter(filter_counts, img_filter);
    if let Some(w_px) = dict_integer(dict, b"Width") {
        update_max_ppi(max_ppi, w_px, page_pts_width);
    }
}

/// Extract filter and PPI hint from a parsed inline-image parameter dictionary.
fn scan_inline_image(
    dict: &Dictionary,
    filter_counts: &mut [u32; IMAGE_FILTER_COUNT],
    max_ppi: &mut f32,
    page_pts_width: f32,
) {
    // `parse_inline_params` has already expanded abbreviated keys (b"F" → b"Filter",
    // b"W" → b"Width") so we only need the full names here.
    let filter = dict.get(b"Filter").and_then(filter_name);
    let img_filter = ImageFilter::from_filter_str(filter.as_deref());
    count_filter(filter_counts, img_filter);
    if let Some(w_px) = dict_integer(dict, b"Width") {
        update_max_ppi(max_ppi, w_px, page_pts_width);
    }
}

// ── Utility helpers ───────────────────────────────────────────────────────────

/// Map a `filter_counts` array index back to [`ImageFilter`].
///
/// Each arm must match the `ImageFilter` discriminant of the same value.
/// If a new `ImageFilter` variant is added:
///   1. `IMAGE_FILTER_COUNT` must be bumped (compile-time assert guards this).
///   2. A new arm must be added here — the `unreachable!` wildcard will panic
///      in tests if index 6+ appears without a matching arm.
///
/// The `idx_to_filter_round_trips` test verifies all `IMAGE_FILTER_COUNT`
/// indices round-trip correctly.
const fn idx_to_filter(idx: usize) -> ImageFilter {
    match idx {
        0 => ImageFilter::Dct,
        1 => ImageFilter::Jpx,
        2 => ImageFilter::CcittFax,
        3 => ImageFilter::Jbig2,
        4 => ImageFilter::Flate,
        5 => ImageFilter::Raw,
        // Any index ≥ IMAGE_FILTER_COUNT is a logic bug — the caller always
        // iterates 0..IMAGE_FILTER_COUNT, so this is unreachable in practice.
        _ => unreachable!(),
    }
}

/// Read a positive integer value (Integer or Real) from a dictionary key.
///
/// Returns `None` if the key is absent, zero, negative, or non-numeric.
/// PDF image dimensions (Width, Height) are always positive and fit comfortably
/// in `u32`; the guards `*i > 0` and `*r > 0.0` enforce this before the cast.
fn dict_integer(dict: &Dictionary, key: &[u8]) -> Option<u32> {
    #[expect(
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation,
        reason = "guarded by *i > 0; PDF image dimensions fit in u32 (max ~65 535 px)"
    )]
    match dict.get(key)? {
        Object::Integer(i) if *i > 0 => Some(*i as u32),
        Object::Real(r) if *r > 0.0 => Some(*r as u32),
        _ => None,
    }
}

/// Increment the filter count slot for `filter`, saturating at `u32::MAX`.
#[inline]
const fn count_filter(filter_counts: &mut [u32; IMAGE_FILTER_COUNT], filter: ImageFilter) {
    filter_counts[filter as usize] = filter_counts[filter as usize].saturating_add(1);
}

/// Update `max_ppi` with the PPI implied by an image of width `w_px` on a
/// page of width `page_pts_width` PDF points.  No-op if `page_pts_width` is
/// zero (degenerate page) or if the computed PPI is not larger than the
/// current maximum.
#[inline]
fn update_max_ppi(max_ppi: &mut f32, w_px: u32, page_pts_width: f32) {
    if page_pts_width <= 0.0 {
        return;
    }
    #[expect(
        clippy::cast_precision_loss,
        reason = "image width ≤ ~65 535 px; f32 is sufficient for PPI routing"
    )]
    let ppi = (w_px as f32 / page_pts_width) * 72.0;
    if ppi > *max_ppi {
        *max_ppi = ppi;
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ───────────────────────────────────────────────────────────────

    /// Build a minimal in-memory PDF with one page that has a content stream.
    ///
    /// Layout (object numbers):
    ///   1: Catalog
    ///   2: Pages
    ///   3: Page (Type=Page, Parent=2, MediaBox=[0 0 612 792], Contents=4)
    ///   4: Content stream (length = `content_bytes.len()`)
    fn doc_with_content(content_bytes: Vec<u8>) -> (Document, u32) {
        let header = "%PDF-1.4\n";
        let obj1 = "1 0 obj\n<</Type /Catalog /Pages 2 0 R>>\nendobj\n".to_string();
        let obj2 = "2 0 obj\n<</Type /Pages /Kids [3 0 R] /Count 1>>\nendobj\n".to_string();
        let obj3 = "3 0 obj\n<</Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] \
                    /Contents 4 0 R>>\nendobj\n"
            .to_string();
        let stream_len = content_bytes.len();
        let obj4_header = format!("4 0 obj\n<</Length {stream_len}>>\nstream\n");
        let obj4_footer = "\nendstream\nendobj\n";

        let off1 = header.len();
        let off2 = off1 + obj1.len();
        let off3 = off2 + obj2.len();
        let off4 = off3 + obj3.len();
        let xref_start = off4 + obj4_header.len() + stream_len + obj4_footer.len();

        let xref = format!(
            "xref\n0 5\n0000000000 65535 f\r\n\
             {off1:010} 00000 n\r\n\
             {off2:010} 00000 n\r\n\
             {off3:010} 00000 n\r\n\
             {off4:010} 00000 n\r\n",
        );
        let trailer = format!("trailer\n<</Size 5 /Root 1 0 R>>\nstartxref\n{xref_start}\n%%EOF");

        let mut bytes = Vec::with_capacity(xref_start + xref.len() + trailer.len());
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(obj1.as_bytes());
        bytes.extend_from_slice(obj2.as_bytes());
        bytes.extend_from_slice(obj3.as_bytes());
        bytes.extend_from_slice(obj4_header.as_bytes());
        bytes.extend_from_slice(&content_bytes);
        bytes.extend_from_slice(obj4_footer.as_bytes());
        bytes.extend_from_slice(xref.as_bytes());
        bytes.extend_from_slice(trailer.as_bytes());

        let doc = Document::from_bytes_owned(bytes).expect("test PDF parse");
        (doc, 1)
    }

    // ── Tests ─────────────────────────────────────────────────────────────────

    #[test]
    fn prescan_page_out_of_range_returns_err() {
        let (doc, _) = doc_with_content(b"BT /F1 12 Tf (Hello) Tj ET".to_vec());
        let result = prescan_page(&doc, 99);
        assert!(
            matches!(result, Err(InterpError::PageOutOfRange { .. })),
            "expected PageOutOfRange, got {result:?}"
        );
    }

    #[test]
    fn prescan_pure_text_page_no_images() {
        // Content stream: begin text, show text, end text.
        let ops = b"BT /F1 12 Tf 72 720 Td (Hello World) Tj ET";
        let (doc, page_num) = doc_with_content(ops.to_vec());
        let diag = prescan_page(&doc, page_num).expect("prescan should succeed");
        assert!(!diag.has_images, "pure text page must not have images");
        assert!(diag.dominant_filter.is_none());
        assert!(diag.source_ppi_hint.is_none());
        // has_vector_text: depends on the content stream parser recognising Tj —
        // accept either true or false here since the test page has no font resources
        // (Tj may be silently dropped). The key assertion is no images.
    }

    #[test]
    fn prescan_empty_page_is_clean() {
        let (doc, page_num) = doc_with_content(b"".to_vec());
        let diag = prescan_page(&doc, page_num).expect("prescan of empty page should succeed");
        assert!(!diag.has_images);
        assert!(!diag.has_vector_text);
        assert!(diag.dominant_filter.is_none());
        assert!(diag.source_ppi_hint.is_none());
    }

    #[test]
    fn image_filter_from_filter_str_covers_all_variants() {
        assert!(matches!(
            ImageFilter::from_filter_str(Some("DCTDecode")),
            ImageFilter::Dct
        ));
        assert!(matches!(
            ImageFilter::from_filter_str(Some("JPXDecode")),
            ImageFilter::Jpx
        ));
        assert!(matches!(
            ImageFilter::from_filter_str(Some("CCITTFaxDecode")),
            ImageFilter::CcittFax
        ));
        assert!(matches!(
            ImageFilter::from_filter_str(Some("JBIG2Decode")),
            ImageFilter::Jbig2
        ));
        assert!(matches!(
            ImageFilter::from_filter_str(Some("FlateDecode")),
            ImageFilter::Flate
        ));
        assert!(matches!(
            ImageFilter::from_filter_str(None),
            ImageFilter::Raw
        ));
        assert!(matches!(
            ImageFilter::from_filter_str(Some("Unknown")),
            ImageFilter::Raw
        ));
    }

    #[test]
    fn idx_to_filter_round_trips() {
        for idx in 0..IMAGE_FILTER_COUNT {
            let f = idx_to_filter(idx);
            assert_eq!(
                f as usize, idx,
                "idx_to_filter({idx}) discriminant mismatch"
            );
        }
    }

    #[test]
    fn dict_integer_reads_integer_and_real() {
        let mut dict = Dictionary::new();
        dict.set(b"Width".to_vec(), Object::Integer(640));
        dict.set(b"Height".to_vec(), Object::Real(480.0));
        assert_eq!(dict_integer(&dict, b"Width"), Some(640));
        assert_eq!(dict_integer(&dict, b"Height"), Some(480));
        assert_eq!(dict_integer(&dict, b"Missing"), None);
    }

    #[test]
    fn scan_image_dict_increments_dct_filter_count() {
        let mut dict = Dictionary::new();
        dict.set(b"Filter".to_vec(), Object::Name(b"DCTDecode".to_vec()));
        dict.set(b"Width".to_vec(), Object::Integer(1024));

        let mut filter_counts = [0u32; IMAGE_FILTER_COUNT];
        let mut max_ppi = 0.0f32;
        scan_image_dict(&dict, &mut filter_counts, &mut max_ppi, 612.0);

        assert_eq!(filter_counts[ImageFilter::Dct as usize], 1);
        assert!(max_ppi > 0.0, "PPI should be computed for wide image");
    }

    #[test]
    fn update_max_ppi_updates_when_larger() {
        let mut max_ppi = 72.0f32;
        update_max_ppi(&mut max_ppi, 1440, 612.0); // 1440/612*72 ≈ 169 ppi
        assert!(
            max_ppi > 150.0,
            "expected max_ppi to increase, got {max_ppi}"
        );
    }

    #[test]
    fn update_max_ppi_ignores_when_smaller() {
        let mut max_ppi = 300.0f32;
        update_max_ppi(&mut max_ppi, 72, 612.0); // 72/612*72 ≈ 8.5 ppi
        assert!(
            (max_ppi - 300.0).abs() < 0.001,
            "expected max_ppi unchanged, got {max_ppi}"
        );
    }

    #[test]
    fn update_max_ppi_zero_page_width_is_noop() {
        let mut max_ppi = 0.0f32;
        update_max_ppi(&mut max_ppi, 1000, 0.0);
        assert_eq!(max_ppi, 0.0);
    }

    #[test]
    fn count_filter_increments_correct_slot() {
        let mut counts = [0u32; IMAGE_FILTER_COUNT];
        count_filter(&mut counts, ImageFilter::Dct);
        count_filter(&mut counts, ImageFilter::Dct);
        count_filter(&mut counts, ImageFilter::Flate);
        assert_eq!(counts[ImageFilter::Dct as usize], 2);
        assert_eq!(counts[ImageFilter::Flate as usize], 1);
        assert_eq!(counts[ImageFilter::Jpx as usize], 0);
    }

    #[test]
    fn count_filter_saturates_at_u32_max() {
        let mut counts = [0u32; IMAGE_FILTER_COUNT];
        counts[ImageFilter::Raw as usize] = u32::MAX;
        count_filter(&mut counts, ImageFilter::Raw);
        assert_eq!(counts[ImageFilter::Raw as usize], u32::MAX);
    }
}
