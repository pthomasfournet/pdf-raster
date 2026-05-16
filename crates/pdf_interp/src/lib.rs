//! PDF content stream interpreter.
//!
//! # Architecture
//!
//! ```text
//! pdf::Document  →  parse_page()  →  Vec<Operator>  →  renderer::PageRenderer
//!                →  prescan_page() →  PageDiagnostics  (no pixels decoded)
//! ```
//!
//! The [`content`] module handles tokenisation and operator decoding.
//! The [`renderer`] module rasterises operators to a pixel bitmap.
//! The [`prescan`] module classifies pages cheaply before rendering.

#[cfg(feature = "cache")]
pub mod cache;
pub mod content;
pub mod prescan;
pub mod renderer;
pub mod resources;
#[cfg(test)]
mod test_helpers;

pub use prescan::prescan_page;

/// Entry points for fuzz targets — forwards to codec wrappers in `image::fuzz_entry`.
///
/// Only compiled when `--cfg fuzzing` is set (cargo-fuzz sets this automatically).
/// Not part of the public API.
#[cfg(fuzzing)]
#[doc(hidden)]
pub mod fuzz_helpers {
    pub use crate::resources::image::fuzz_entry::{decode_ccitt, decode_jbig2};
}

use std::path::Path;

use pdf::Document;

/// Errors that can occur during PDF loading or content stream interpretation.
#[derive(Debug)]
pub enum InterpError {
    /// The lazy PDF parser failed to load or parse the document.
    Pdf(pdf::PdfError),
    /// The requested page number is outside the document's page range.
    PageOutOfRange {
        /// The requested page (1-based).
        page: u32,
        /// Total number of pages in the document.
        total: u32,
    },
    /// A required resource (font, image, colour space, …) could not be resolved.
    MissingResource(String),
    /// A Page dictionary entry has a value that is structurally valid PDF but
    /// outside the range permitted by the spec or our safety limits.
    InvalidPageGeometry(String),
    /// `FreeType` library initialisation failed (library not available or internal error).
    FontInit(String),
    /// The page render was aborted because it exceeded the per-page operator
    /// budget or wall-clock deadline.  The page is pathological; the rendered
    /// bitmap is partial.
    PageBudget(String),
}

impl std::fmt::Display for InterpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pdf(e) => write!(f, "PDF error: {e}"),
            Self::PageOutOfRange { page, total } => {
                write!(
                    f,
                    "page {page} is out of range (document has {total} pages)"
                )
            }
            Self::MissingResource(name) => write!(f, "missing PDF resource: {name}"),
            Self::InvalidPageGeometry(msg) => write!(f, "invalid page geometry: {msg}"),
            Self::FontInit(msg) => write!(f, "FreeType initialisation failed: {msg}"),
            Self::PageBudget(msg) => write!(f, "page render budget exceeded: {msg}"),
        }
    }
}

impl std::error::Error for InterpError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        // pdf::PdfError doesn't implement std::error::Error yet; surface it
        // via Display only.
        None
    }
}

impl From<pdf::PdfError> for InterpError {
    fn from(e: pdf::PdfError) -> Self {
        Self::Pdf(e)
    }
}

/// Open a PDF document from a file path.
///
/// If the document contains any JavaScript entry point, a loud `WARN` is
/// emitted for each location found and the document still opens: rasterrocket
/// is a rasterizer with no JavaScript engine, so it never executes `/JS` —
/// the structural presence of a script has no effect on the static rendered
/// page, and refusing valid documents would be a false-negative total loss.
///
/// Checked locations (PDF 2.0 §12.6):
/// - `/Catalog/OpenAction` with `/S /JavaScript`
/// - `/Catalog/AA` (catalog-level additional actions)
/// - `/Catalog/Names/JavaScript` (document JS name tree)
/// - `/Catalog/AcroForm/AA` (`AcroForm` additional actions)
///
/// # Errors
/// - [`InterpError::Pdf`] if the file cannot be read or is not a valid PDF.
pub fn open(path: impl AsRef<Path>) -> Result<Document, InterpError> {
    open_decrypting(path, false)
}

/// Open a PDF, transparently qpdf-decrypting it first when it is encrypted
/// and `decrypt_authorized` is `true`.
///
/// `decrypt_authorized` MUST reflect a deliberate caller decision — the
/// CLI gates it behind an interactive private-copy / liability waiver or
/// an explicit operator bypass; the private QA harness auto-authorises.
/// When the document is encrypted and `decrypt_authorized` is `false`,
/// this returns [`InterpError::Pdf`] wrapping
/// [`pdf::PdfError::EncryptedDocument`] with an accurate, actionable
/// message — never the misleading "document has no pages".
///
/// Unencrypted documents take the fast path: no qpdf spawn, no temp file.
///
/// # Errors
/// - [`InterpError::Pdf`] if the file cannot be read, is not a valid PDF,
///   or is encrypted and could not be (or was not authorised to be)
///   decrypted.
pub fn open_decrypting(
    path: impl AsRef<Path>,
    decrypt_authorized: bool,
) -> Result<Document, InterpError> {
    let doc = Document::open_decrypting(path.as_ref(), decrypt_authorized)?;
    warn_if_javascript(&doc);
    Ok(doc)
}

/// Scan the document catalog for JavaScript entry points and emit a loud
/// `WARN` for every location present.  Never fails: rasterrocket has no
/// JavaScript engine and never executes `/JS`, so a script's structural
/// presence cannot change the static rendered page — the correct behaviour
/// is to render the document and tell the operator the JS was ignored.
///
/// No JS content is read, parsed, or evaluated — the check is purely
/// structural (dict key presence / subtype name).
fn warn_if_javascript(doc: &Document) {
    for location in javascript_locations(doc) {
        log::warn!(
            "document contains JavaScript ({location}); JavaScript is ignored — \
             rasterrocket does not execute scripts, rendering static content only"
        );
    }
}

/// Return every JavaScript entry point present in the document catalog, in
/// scan order.  Empty when the catalog is unreadable or carries no JS.
///
/// Split out from [`warn_if_javascript`] so the four detection branches are
/// unit-testable without capturing log output.
fn javascript_locations(doc: &Document) -> Vec<&'static str> {
    let mut found = Vec::new();
    let Ok(catalog) = doc.catalog() else {
        return found;
    };

    // 1. OpenAction with /S /JavaScript
    if let Some(action) = catalog.get(b"OpenAction")
        && action_is_js(doc, action)
    {
        found.push("/OpenAction");
    }

    // 2. Catalog-level additional actions (/AA)
    if catalog.get(b"AA").is_some() {
        found.push("/AA (catalog additional actions)");
    }

    // 3. Document JS name tree (/Names/JavaScript)
    if let Some(names_obj) = catalog.get(b"Names")
        && let Some(names_dict) = resources::resolve_dict(doc, names_obj)
        && names_dict.get(b"JavaScript").is_some()
    {
        found.push("/Names/JavaScript");
    }

    // 4. AcroForm additional actions
    if let Some(acroform_obj) = catalog.get(b"AcroForm")
        && let Some(acroform) = resources::resolve_dict(doc, acroform_obj)
        && acroform.get(b"AA").is_some()
    {
        found.push("/AcroForm/AA");
    }

    found
}

/// Return `true` if `obj` (or the dict it resolves to) has `/S /JavaScript`.
fn action_is_js(doc: &Document, obj: &pdf::Object) -> bool {
    let Some(d) = resources::resolve_dict(doc, obj) else {
        return false;
    };
    d.get(b"S")
        .and_then(pdf::Object::as_name)
        .is_some_and(|name| name == b"JavaScript")
}

/// Return the number of pages in `doc`.
///
/// Reads `/Pages /Count` directly (O(1) on well-formed PDFs); falls back to
/// the eager tree-walk count if the catalog entry is missing or malformed.
#[must_use]
pub fn page_count(doc: &Document) -> u32 {
    doc.page_count_fast()
}

/// Resolve a 1-based page number to its `pdf::ObjectId`.
///
/// Logarithmic in document page count: descends the `/Pages` tree using
/// each node's `/Count` to skip whole subtrees rather than walking every
/// leaf.
///
/// # Errors
/// [`InterpError::PageOutOfRange`] if `page_num` is 0 or exceeds the document;
/// [`InterpError::Pdf`] if the page tree itself is malformed.
pub(crate) fn resolve_page_id(doc: &Document, page_num: u32) -> Result<pdf::ObjectId, InterpError> {
    let total = doc.page_count_fast();
    if page_num == 0 || page_num > total {
        return Err(InterpError::PageOutOfRange {
            page: page_num,
            total,
        });
    }
    // 1-based → 0-based.  Underflow guarded by `page_num != 0` above.
    doc.get_page(page_num - 1).map_err(InterpError::Pdf)
}

/// Page geometry: visible dimensions in PDF points, rotation, and user-space scale.
///
/// `width_pts` and `height_pts` are already adjusted for rotation — they describe the
/// output bitmap dimensions, not the raw `MediaBox`.  For example a landscape page with
/// `/Rotate 270` has `width_pts > height_pts` even though its `MediaBox` may be portrait.
///
/// Both dimensions are also already scaled by `user_unit` (PDF 1.6+ `UserUnit` key).
/// Multiply the render DPI by `user_unit` to get the true physical resolution of the
/// rendered bitmap (i.e. the value to pass to `tesseract::set_source_resolution`).
#[derive(Debug, Clone, Copy)]
pub struct PageGeometry {
    /// Width of the rendered output in PDF points (after rotation and `UserUnit` scaling).
    pub width_pts: f64,
    /// Height of the rendered output in PDF points (after rotation and `UserUnit` scaling).
    pub height_pts: f64,
    /// Clockwise rotation in degrees; always one of 0, 90, 180, 270.
    pub rotate_cw: u16,
    /// `UserUnit` scale factor from the Page dictionary (PDF 1.6+, ISO 32000-2 §14.11.2).
    ///
    /// Specifies the size of one default user-space unit in 1/72-inch increments.
    /// `1.0` for the vast majority of documents.  Validated to `[0.1, 10.0]`.
    pub user_unit: f64,
    /// X-coordinate of the selected page box's lower-left corner in PDF user space (ISO 32000-2 §14.11.2).
    ///
    /// For the vast majority of documents this is `0.0`.  Non-zero for scanned PDFs
    /// whose `CropBox`/`MediaBox` does not start at the origin — e.g. a `CropBox`
    /// of `[36 36 576 756]` has `origin_x = 36`.  The initial CTM pre-translates by
    /// `(-origin_x, -origin_y)` so that the box's lower-left maps to device origin.
    pub origin_x: f64,
    /// Y-coordinate of the selected page box's lower-left corner in PDF user space.
    ///
    /// See `origin_x`.
    pub origin_y: f64,
}

/// Return the geometry for page `page_num` (1-based).
///
/// Reads the page's `CropBox` (the display region, per ISO 32000-2 §14.11.2), falling back
/// to `MediaBox` when absent. Applies the `/Rotate` entry (multiples of 90° CW). Dimensions
/// in the returned struct are already swapped for 90°/270° rotations and scaled by `UserUnit`
/// so callers can use `width_pts × height_pts` directly as output point dimensions.
///
/// Falls back to US Letter (612 × 792 pt, no rotation, `UserUnit` 1.0) when neither box
/// can be read.
///
/// # Errors
///
/// - [`InterpError::PageOutOfRange`] if `page_num` is 0 or exceeds the document page count.
/// - [`InterpError::InvalidPageGeometry`] if `UserUnit` is present but outside `[0.1, 10.0]`.
pub fn page_size_pts(doc: &Document, page_num: u32) -> Result<PageGeometry, InterpError> {
    let page_id = resolve_page_id(doc, page_num)?;
    page_size_pts_by_id(doc, page_id)
}

/// Like [`page_size_pts`] but takes a pre-resolved page object id.
///
/// Skips the page-tree descent.  Use when the caller already resolved the id —
/// the renderer resolves once at its entry point and reuses the id for both
/// `page_size_pts_by_id` and [`parse_page_by_id`].
///
/// # Errors
/// [`InterpError::InvalidPageGeometry`] if `UserUnit` is present but outside
/// `[0.1, 10.0]`.
#[expect(
    clippy::too_many_lines,
    reason = "all logic is tightly coupled to the same page-geometry resolution; splitting would require threading several local closures through helper boundaries"
)]
pub fn page_size_pts_by_id(
    doc: &Document,
    page_id: pdf::ObjectId,
) -> Result<PageGeometry, InterpError> {
    let fallback = PageGeometry {
        width_pts: 612.0,
        height_pts: 792.0,
        rotate_cw: 0,
        user_unit: 1.0,
        origin_x: 0.0,
        origin_y: 0.0,
    };

    let Ok(dict) = doc.get_dict(page_id) else {
        return Ok(fallback);
    };

    #[expect(
        clippy::cast_precision_loss,
        reason = "PDF page sizes are at most a few thousand points; i64 → f64 precision loss is negligible"
    )]
    let to_f64 = |o: &pdf::Object| match o {
        pdf::Object::Real(r) => f64::from(*r),
        pdf::Object::Integer(i) => *i as f64,
        _ => 0.0,
    };

    // Returns (x0, y0, w, h) — the lower-left origin and dimensions of the box.
    // The origin is needed to pre-translate the initial CTM so the box lower-left
    // maps to device origin (ISO 32000-2 §8.3.2: user space origin = box lower-left).
    //
    // Takes an Object (resolved via the inherited-attr lookup) rather than a dict
    // key, so the same parsing logic can be applied regardless of which node in the
    // /Parent chain supplied the value.
    let parse_box = |obj: &pdf::Object| -> Option<(f64, f64, f64, f64)> {
        let arr = match obj {
            pdf::Object::Array(a) if a.len() == 4 => a,
            _ => return None,
        };
        let (x0, y0, x1, y1) = (
            to_f64(&arr[0]),
            to_f64(&arr[1]),
            to_f64(&arr[2]),
            to_f64(&arr[3]),
        );
        let w = (x1 - x0).abs();
        let h = (y1 - y0).abs();
        if w > 0.0 && h > 0.0 {
            // Normalise: lower-left is the minimum corner regardless of array order.
            Some((x0.min(x1), y0.min(y1), w, h))
        } else {
            None
        }
    };

    // ISO 32000-2 §7.7.3.4: MediaBox and CropBox are inheritable.  Walk the
    // /Parent chain so a page that omits these keys inherits from the nearest
    // /Pages ancestor rather than falling back to the 612×792 default.
    let media_obj = doc.get_inherited_page_attr(page_id, b"MediaBox");
    let crop_obj = doc.get_inherited_page_attr(page_id, b"CropBox");

    let media_box = media_obj.as_ref().and_then(&parse_box);
    let crop_box = crop_obj.as_ref().and_then(&parse_box);

    // ISO 32000-2 §14.11.2: CropBox shall be intersected with MediaBox to
    // determine the visible region.  If CropBox is absent, use MediaBox
    // directly (spec: CropBox defaults to MediaBox).
    //
    // For the common case where the leaf has both boxes and CropBox ⊆ MediaBox
    // the intersection is the CropBox itself — byte-identical to the old path.
    let Some((box_left, box_bottom, w_pts, h_pts)) = (match (media_box, crop_box) {
        (None, None) => None,
        (Some(m), None) => Some(m),
        (None, Some(c)) => Some(c), // malformed but recover gracefully
        (Some(m), Some(c)) => {
            // Clamp CropBox to the MediaBox bounds (§14.11.2 intersection).
            let x0 = c.0.max(m.0);
            let y0 = c.1.max(m.1);
            let x1 = (c.0 + c.2).min(m.0 + m.2);
            let y1 = (c.1 + c.3).min(m.1 + m.3);
            let w = x1 - x0;
            let h = y1 - y0;
            if w > 0.0 && h > 0.0 {
                Some((x0, y0, w, h))
            } else {
                // Degenerate intersection (CropBox entirely outside MediaBox):
                // fall back to MediaBox so the page is not empty.
                Some(m)
            }
        }
    }) else {
        return Ok(fallback);
    };

    // ISO 32000-2 §7.7.3.4: /Rotate is inheritable.  Walk the /Parent chain
    // for the same reason as MediaBox/CropBox above.
    // rem_euclid(360) is always 0..=359, which fits u16.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "rem_euclid(360) is always 0..=359, which fits u16"
    )]
    let rotate_cw: u16 = match doc.get_inherited_page_attr(page_id, b"Rotate") {
        Some(pdf::Object::Integer(n)) => n.rem_euclid(360) as u16 / 90 * 90,
        _ => 0,
    };

    // UserUnit (PDF 1.6+, ISO 32000-2 §14.11.2): scales one default user-space unit
    // from 1/72 inch to UserUnit/72 inches.  Absent → 1.0.  Must be a finite positive
    // number.  We reject values outside [0.1, 10.0]: below 0.1 the page would be
    // unreasonably small at any practical DPI; above 10.0 the pixel dimensions would
    // exceed MAX_PX_DIMENSION at even modest DPI settings.  NaN/Inf in a Real object
    // are rejected explicitly because the range check (NaN comparisons are always false)
    // would otherwise silently pass them through.
    let user_unit: f64 = match dict.get(b"UserUnit") {
        None => 1.0, // absent — use default
        Some(obj) => {
            match obj {
                pdf::Object::Real(_) | pdf::Object::Integer(_) => {}
                other => {
                    return Err(InterpError::InvalidPageGeometry(format!(
                        "UserUnit on page object {} is not a number (got {})",
                        page_id.0,
                        other.enum_variant()
                    )));
                }
            }
            let v = to_f64(obj);
            if !v.is_finite() || !(0.1..=10.0).contains(&v) {
                return Err(InterpError::InvalidPageGeometry(format!(
                    "UserUnit {v} on page object {} is outside the valid range [0.1, 10.0]",
                    page_id.0,
                )));
            }
            v
        }
    };

    // Swap dimensions for 90°/270° so the caller gets output-bitmap dimensions directly.
    // Scale both dimensions by UserUnit so downstream only needs to apply dpi/72.
    let (width_pts, height_pts) = if rotate_cw == 90 || rotate_cw == 270 {
        (h_pts * user_unit, w_pts * user_unit)
    } else {
        (w_pts * user_unit, h_pts * user_unit)
    };

    // The origin is kept in unscaled PDF user-space coordinates (before UserUnit scaling).
    // The CTM in new_scaled applies (scale = dpi/72 * user_unit) already, so the
    // pre-translation by (-origin_x, -origin_y) in unscaled points is correct there.
    let (origin_x, origin_y) = (box_left, box_bottom);

    Ok(PageGeometry {
        width_pts,
        height_pts,
        rotate_cw,
        user_unit,
        origin_x,
        origin_y,
    })
}

/// Parse the content stream for page `page_num` (1-based) and return the
/// decoded operator sequence.
///
/// # Errors
/// Returns [`InterpError::PageOutOfRange`] if `page_num` is 0 or exceeds the
/// document page count, or [`InterpError::Pdf`] if the content stream cannot
/// be read.
pub fn parse_page(doc: &Document, page_num: u32) -> Result<Vec<content::Operator>, InterpError> {
    let page_id = resolve_page_id(doc, page_num)?;
    parse_page_by_id(doc, page_id)
}

/// Like [`parse_page`] but takes a pre-resolved page object id, skipping
/// the page-tree descent.
///
/// # Errors
/// [`InterpError::Pdf`] if the content stream cannot be read.
pub fn parse_page_by_id(
    doc: &Document,
    page_id: pdf::ObjectId,
) -> Result<Vec<content::Operator>, InterpError> {
    let content_bytes = doc.get_page_content(page_id)?;
    Ok(content::parse(&content_bytes))
}

#[cfg(test)]
mod js_guard_tests {
    use super::test_helpers::make_doc;
    use super::{javascript_locations, warn_if_javascript};

    #[test]
    fn open_action_javascript_is_warned_not_rejected() {
        let doc = make_doc(" /OpenAction <</S /JavaScript>>");
        // Detected, but the document is renderable: warn, never fail.
        assert_eq!(javascript_locations(&doc), vec!["/OpenAction"]);
        warn_if_javascript(&doc); // must not panic
    }

    #[test]
    fn open_action_goto_is_not_javascript() {
        let doc = make_doc(" /OpenAction <</S /GoTo>>");
        assert!(javascript_locations(&doc).is_empty());
    }

    #[test]
    fn names_javascript_is_warned_not_rejected() {
        let doc = make_doc(" /Names <</JavaScript <</Names []>>>>");
        assert_eq!(javascript_locations(&doc), vec!["/Names/JavaScript"]);
        warn_if_javascript(&doc); // must not panic
    }

    #[test]
    fn aa_and_acroform_aa_are_detected() {
        let doc =
            make_doc(" /AA <</WC <</S /JavaScript>>>> /AcroForm <</AA <</K <</S /JavaScript>>>>>>");
        assert_eq!(
            javascript_locations(&doc),
            vec!["/AA (catalog additional actions)", "/AcroForm/AA"]
        );
    }

    #[test]
    fn all_four_javascript_locations_detected() {
        let doc = make_doc(
            " /OpenAction <</S /JavaScript>> /AA <</WC <</S /JavaScript>>>> \
             /Names <</JavaScript <</Names []>>>> /AcroForm <</AA <</K <</S /JavaScript>>>>>>",
        );
        assert_eq!(
            javascript_locations(&doc),
            vec![
                "/OpenAction",
                "/AA (catalog additional actions)",
                "/Names/JavaScript",
                "/AcroForm/AA"
            ]
        );
        warn_if_javascript(&doc); // must not panic
    }

    #[test]
    fn clean_document_has_no_javascript() {
        let doc = make_doc("");
        assert!(javascript_locations(&doc).is_empty());
        warn_if_javascript(&doc); // no-op, must not panic
    }
}

// ── §7.7.3.4 inheritance + §14.11.2 clamp tests ──────────────────────────────

#[cfg(test)]
mod page_box_tests {
    use pdf::Document;

    use super::{page_size_pts, page_size_pts_by_id};

    /// Build a minimal one-page PDF where the MediaBox is on the /Pages node
    /// (inherited), not on the page leaf.  The leaf has only /Type /Page and a
    /// /Parent back-reference.
    ///
    /// Object layout:
    ///   1 0 — Catalog (/Pages 2 0 R)
    ///   2 0 — Pages root (/MediaBox [0 0 500 700] /Kids [3 0 R] /Count 1)
    ///   3 0 — Page leaf (/Type /Page /Parent 2 0 R — no MediaBox)
    fn make_inherited_mediabox_doc() -> Document {
        let header = "%PDF-1.4\n";
        let obj1 = "1 0 obj\n<</Type /Catalog /Pages 2 0 R>>\nendobj\n";
        let obj2 =
            "2 0 obj\n<</Type /Pages /MediaBox [0 0 500 700] /Kids [3 0 R] /Count 1>>\nendobj\n";
        let obj3 = "3 0 obj\n<</Type /Page /Parent 2 0 R>>\nendobj\n";

        let off1 = header.len();
        let off2 = off1 + obj1.len();
        let off3 = off2 + obj2.len();
        let xref_start = off3 + obj3.len();
        let xref = format!(
            "xref\n0 4\n\
             0000000000 65535 f\r\n\
             {off1:010} 00000 n\r\n\
             {off2:010} 00000 n\r\n\
             {off3:010} 00000 n\r\n"
        );
        let trailer = format!("trailer\n<</Size 4 /Root 1 0 R>>\nstartxref\n{xref_start}\n%%EOF");
        let bytes = format!("{header}{obj1}{obj2}{obj3}{xref}{trailer}").into_bytes();
        Document::from_bytes_owned(bytes).expect("inherited-mediabox test PDF parse")
    }

    /// Build a one-page PDF where the page leaf has its own MediaBox [0 0 600 800]
    /// and a CropBox [0 0 700 900] that extends beyond the MediaBox.  After the
    /// §14.11.2 clamp the effective box must be [0 0 600 800] (the intersection).
    fn make_cropbox_exceeds_mediabox_doc() -> Document {
        let header = "%PDF-1.4\n";
        let obj1 = "1 0 obj\n<</Type /Catalog /Pages 2 0 R>>\nendobj\n";
        let obj2 = "2 0 obj\n<</Type /Pages /Kids [3 0 R] /Count 1>>\nendobj\n";
        // CropBox [0 0 700 900] is larger than MediaBox [0 0 600 800] on all sides.
        let obj3 = "3 0 obj\n<</Type /Page /Parent 2 0 R \
                    /MediaBox [0 0 600 800] /CropBox [0 0 700 900]>>\nendobj\n";

        let off1 = header.len();
        let off2 = off1 + obj1.len();
        let off3 = off2 + obj2.len();
        let xref_start = off3 + obj3.len();
        let xref = format!(
            "xref\n0 4\n\
             0000000000 65535 f\r\n\
             {off1:010} 00000 n\r\n\
             {off2:010} 00000 n\r\n\
             {off3:010} 00000 n\r\n"
        );
        let trailer = format!("trailer\n<</Size 4 /Root 1 0 R>>\nstartxref\n{xref_start}\n%%EOF");
        let bytes = format!("{header}{obj1}{obj2}{obj3}{xref}{trailer}").into_bytes();
        Document::from_bytes_owned(bytes).expect("cropbox-exceeds-mediabox test PDF parse")
    }

    /// Build a one-page PDF where the page leaf has both MediaBox [0 0 612 792]
    /// and CropBox [72 72 540 720] (CropBox ⊆ MediaBox — the common born-digital
    /// case).  The result must match the CropBox exactly (pixel-neutral invariant).
    fn make_cropbox_inside_mediabox_doc() -> Document {
        let header = "%PDF-1.4\n";
        let obj1 = "1 0 obj\n<</Type /Catalog /Pages 2 0 R>>\nendobj\n";
        let obj2 = "2 0 obj\n<</Type /Pages /Kids [3 0 R] /Count 1>>\nendobj\n";
        let obj3 = "3 0 obj\n<</Type /Page /Parent 2 0 R \
                    /MediaBox [0 0 612 792] /CropBox [72 72 540 720]>>\nendobj\n";

        let off1 = header.len();
        let off2 = off1 + obj1.len();
        let off3 = off2 + obj2.len();
        let xref_start = off3 + obj3.len();
        let xref = format!(
            "xref\n0 4\n\
             0000000000 65535 f\r\n\
             {off1:010} 00000 n\r\n\
             {off2:010} 00000 n\r\n\
             {off3:010} 00000 n\r\n"
        );
        let trailer = format!("trailer\n<</Size 4 /Root 1 0 R>>\nstartxref\n{xref_start}\n%%EOF");
        let bytes = format!("{header}{obj1}{obj2}{obj3}{xref}{trailer}").into_bytes();
        Document::from_bytes_owned(bytes).expect("cropbox-inside-mediabox test PDF parse")
    }

    /// §7.7.3.4: a page leaf without its own MediaBox must inherit from the
    /// nearest ancestor /Pages node.  Before this fix the result would have
    /// been the hardcoded 612×792 fallback.
    #[test]
    fn inherited_mediabox_produces_correct_size() {
        let doc = make_inherited_mediabox_doc();
        let geom = page_size_pts(&doc, 1).expect("page_size_pts failed");
        assert_eq!(
            (geom.width_pts, geom.height_pts),
            (500.0, 700.0),
            "inherited MediaBox [0 0 500 700] must yield 500×700, got {}×{}",
            geom.width_pts,
            geom.height_pts
        );
    }

    /// §7.7.3.4 negative: before the fix, a leaf without its own MediaBox would
    /// have returned the 612×792 fallback instead of the inherited value.  This
    /// assertion confirms the old wrong behavior is gone.
    #[test]
    fn inherited_mediabox_is_not_fallback() {
        let doc = make_inherited_mediabox_doc();
        let geom = page_size_pts(&doc, 1).expect("page_size_pts failed");
        assert_ne!(
            (geom.width_pts, geom.height_pts),
            (612.0, 792.0),
            "page with inherited MediaBox must NOT return the 612×792 default fallback"
        );
    }

    /// §14.11.2: CropBox extending beyond MediaBox must be clamped to MediaBox.
    /// Effective box = intersection([0 0 700 900], [0 0 600 800]) = [0 0 600 800].
    #[test]
    fn cropbox_exceeding_mediabox_is_clamped() {
        let doc = make_cropbox_exceeds_mediabox_doc();
        let geom = page_size_pts(&doc, 1).expect("page_size_pts failed");
        assert_eq!(
            (geom.width_pts, geom.height_pts),
            (600.0, 800.0),
            "CropBox larger than MediaBox must be clamped to MediaBox; got {}×{}",
            geom.width_pts,
            geom.height_pts
        );
        // Origin must be the intersection lower-left (0,0 in this case).
        assert_eq!(
            (geom.origin_x, geom.origin_y),
            (0.0, 0.0),
            "clamped origin must be intersection lower-left"
        );
    }

    /// Pixel-neutrality invariant: CropBox ⊆ MediaBox → result equals the
    /// CropBox exactly (no change from pre-fix behavior for the common case).
    /// CropBox [72 72 540 720] inside MediaBox [0 0 612 792] → 468×648.
    #[test]
    fn cropbox_inside_mediabox_is_unchanged() {
        let doc = make_cropbox_inside_mediabox_doc();
        let geom = page_size_pts(&doc, 1).expect("page_size_pts failed");
        // w = 540-72 = 468, h = 720-72 = 648
        assert_eq!(
            (geom.width_pts, geom.height_pts),
            (468.0, 648.0),
            "CropBox inside MediaBox must be unchanged; got {}×{}",
            geom.width_pts,
            geom.height_pts
        );
        assert_eq!(
            (geom.origin_x, geom.origin_y),
            (72.0, 72.0),
            "origin must be CropBox lower-left when CropBox is inside MediaBox"
        );
    }

    /// Verify that `page_size_pts_by_id` with a pre-resolved id gives the same
    /// answer as `page_size_pts` (exercises the by-id entry point for inheritance).
    #[test]
    fn page_size_pts_by_id_agrees_with_page_size_pts() {
        let doc = make_inherited_mediabox_doc();
        let geom_num = page_size_pts(&doc, 1).expect("page_size_pts failed");
        let page_id = doc.get_page(0).expect("get_page(0) failed");
        let geom_id = page_size_pts_by_id(&doc, page_id).expect("page_size_pts_by_id failed");
        assert_eq!(
            (geom_num.width_pts, geom_num.height_pts),
            (geom_id.width_pts, geom_id.height_pts)
        );
    }
}
