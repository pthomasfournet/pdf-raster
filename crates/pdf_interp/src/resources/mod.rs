//! PDF resource resolution — fonts, colour spaces, `XObject`s, shadings.
//!
//! The [`PageResources`] struct holds a reference to the lopdf [`Document`]
//! and the current page's [`ObjectId`], and provides lazy-loading helpers
//! for each resource category.
//!
//! # Design
//!
//! lopdf's `Document` is immutable after load; resource dicts are accessed
//! via `get_page_fonts()` and friends, which traverse the page tree and
//! dereference indirect objects.  All caching lives in the caller
//! ([`super::renderer::font_cache`]) rather than here, so this module
//! stays stateless and easy to test.

pub(crate) mod dict_ext;
pub mod font;
pub mod image;
pub mod shading;

use lopdf::{Dictionary, Document, Object, ObjectId};
use raster::types::BlendMode;

pub use font::{FontDescriptor, PdfFontKind, resolve_font};
pub use image::{ImageColorSpace, ImageDescriptor, resolve_image};
#[cfg(feature = "nvjpeg")]
pub use gpu::nvjpeg::NvJpegDecoder;

/// Selected parameters extracted from a PDF `ExtGState` resource dictionary.
///
/// Only the subset needed by the current rasteriser is extracted; unknown keys
/// are silently ignored so that future additions do not break existing pages.
#[derive(Debug, Clone)]
pub struct ExtGStateParams {
    /// Non-stroking (fill) opacity `ca` in [0, 255].  `None` = unchanged.
    pub fill_alpha: Option<u8>,
    /// Stroking opacity `CA` in [0, 255].  `None` = unchanged.
    pub stroke_alpha: Option<u8>,
    /// Line width `LW` in user-space units.  `None` = unchanged.
    pub line_width: Option<f64>,
    /// Line cap style `LC` (0–2).  `None` = unchanged.
    pub line_cap: Option<i32>,
    /// Line join style `LJ` (0–2).  `None` = unchanged.
    pub line_join: Option<i32>,
    /// Miter limit `ML`.  `None` = unchanged.
    pub miter_limit: Option<f64>,
    /// Flatness `FL`.  `None` = unchanged.
    pub flatness: Option<f64>,
    /// Blend mode `BM`.  `None` = unchanged (keep current).
    pub blend_mode: Option<BlendMode>,
}

impl ExtGStateParams {
    fn from_dict(d: &Dictionary) -> Self {
        Self {
            fill_alpha: real_to_u8(d, b"ca"),
            stroke_alpha: real_to_u8(d, b"CA"),
            line_width: real_or_int(d, b"LW"),
            line_cap: int_val(d, b"LC"),
            line_join: int_val(d, b"LJ"),
            miter_limit: real_or_int(d, b"ML"),
            flatness: real_or_int(d, b"FL"),
            blend_mode: parse_blend_mode(d),
        }
    }
}

/// Parse the `BM` key from an `ExtGState` dictionary.
///
/// `BM` is either a single `Name` or an array of names (priority list; the
/// viewer uses the first one it supports).  Unknown names are silently ignored
/// and `None` is returned, leaving the current blend mode unchanged.
fn parse_blend_mode(d: &Dictionary) -> Option<BlendMode> {
    let obj = d.get(b"BM").ok()?;
    match obj {
        Object::Name(n) => bm_name_to_mode(n),
        // Array is a viewer-preference priority list; use the first recognised name.
        Object::Array(arr) => arr
            .iter()
            .find_map(|o| o.as_name().ok().and_then(bm_name_to_mode)),
        _ => None,
    }
}

/// Map a PDF blend mode name to [`BlendMode`].
///
/// Returns `None` for unrecognised names so the caller can keep the current mode.
const fn bm_name_to_mode(name: &[u8]) -> Option<BlendMode> {
    Some(match name {
        b"Normal" | b"Compatible" => BlendMode::Normal,
        b"Multiply" => BlendMode::Multiply,
        b"Screen" => BlendMode::Screen,
        b"Overlay" => BlendMode::Overlay,
        b"Darken" => BlendMode::Darken,
        b"Lighten" => BlendMode::Lighten,
        b"ColorDodge" => BlendMode::ColorDodge,
        b"ColorBurn" => BlendMode::ColorBurn,
        b"HardLight" => BlendMode::HardLight,
        b"SoftLight" => BlendMode::SoftLight,
        b"Difference" => BlendMode::Difference,
        b"Exclusion" => BlendMode::Exclusion,
        b"Hue" => BlendMode::Hue,
        b"Saturation" => BlendMode::Saturation,
        b"Color" => BlendMode::Color,
        b"Luminosity" => BlendMode::Luminosity,
        _ => return None,
    })
}

/// Read a real or integer key from a dictionary as `f64`.
fn real_or_int(d: &Dictionary, key: &[u8]) -> Option<f64> {
    match d.get(key).ok()? {
        Object::Real(r) => Some(f64::from(*r)),
        #[expect(
            clippy::cast_precision_loss,
            reason = "ExtGState numeric params are small integers"
        )]
        Object::Integer(n) => Some(*n as f64),
        _ => None,
    }
}

/// Read a real-valued opacity key and convert to u8 in [0, 255].
fn real_to_u8(d: &Dictionary, key: &[u8]) -> Option<u8> {
    let v = real_or_int(d, key)?.clamp(0.0, 1.0);
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "v clamped to [0,1], scaled to [0,255]; round() fits in u8"
    )]
    Some((v * 255.0).round() as u8)
}

/// Read an integer key.
fn int_val(d: &Dictionary, key: &[u8]) -> Option<i32> {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "line cap/join values are 0–2"
    )]
    d.get(key).ok()?.as_i64().ok().map(|n| n as i32)
}

/// Parsed Form `XObject` — content bytes and the CTM matrix to apply.
///
/// The `resources_id` is the `ObjectId` of the form stream itself; callers
/// construct a child [`PageResources`] from it via [`PageResources::for_form`].
pub struct FormXObject {
    /// Decoded content bytes of the form's content stream.
    pub content: Vec<u8>,
    /// Optional form Matrix `[a b c d e f]` (default: identity).
    pub matrix: [f64; 6],
    /// The object ID of the form stream, used to build a child `PageResources`.
    pub resources_id: ObjectId,
    /// True if the form has its own `Resources` dict; false means it inherits
    /// from the parent context (the caller should keep the parent resources).
    pub has_own_resources: bool,
}

/// Thin accessor wrapping a `(Document, resource_context_id)` pair.
///
/// `resource_context_id` is the `ObjectId` of the object (page or form stream)
/// whose `Resources` dict is used to resolve fonts and `XObjects`.  For a top-level
/// page render this is the page object; for a form `XObject` it is the form stream.
pub struct PageResources<'doc> {
    doc: &'doc Document,
    /// Object whose `Resources` key is consulted for font/XObject lookups.
    resource_context_id: ObjectId,
}

impl<'doc> PageResources<'doc> {
    /// Construct a [`PageResources`] for the given page.
    #[must_use]
    pub const fn new(doc: &'doc Document, page_id: ObjectId) -> Self {
        Self {
            doc,
            resource_context_id: page_id,
        }
    }

    /// Construct a child [`PageResources`] scoped to the given form's resources.
    ///
    /// If `form.has_own_resources` is false the form inherits the parent context;
    /// use `self` (the parent) directly instead of calling this.
    #[must_use]
    pub const fn for_form(&self, form: &FormXObject) -> Self {
        if form.has_own_resources {
            Self {
                doc: self.doc,
                resource_context_id: form.resources_id,
            }
        } else {
            Self {
                doc: self.doc,
                resource_context_id: self.resource_context_id,
            }
        }
    }

    /// The underlying `lopdf` document (read-only).
    #[must_use]
    pub const fn doc(&self) -> &'doc Document {
        self.doc
    }

    /// Resolve the font dictionary for the named resource (e.g. `b"F1"`).
    ///
    /// Returns `None` if the resource name is not present in the resource dict.
    #[must_use]
    pub fn font_dict(&self, name: &[u8]) -> Option<font::FontDescriptor> {
        let fonts = self.doc.get_page_fonts(self.resource_context_id).ok()?;
        let dict = fonts.get(name)?;
        Some(font::resolve_font(self.doc, dict))
    }

    /// Decode the named image `XObject` from the resource dictionary.
    ///
    /// Returns `None` if the name is absent, the object is not an image, or
    /// decoding fails.
    ///
    /// `gpu` enables GPU-accelerated JPEG decoding for large `DCTDecode` streams
    /// when the `nvjpeg` feature is active.  Pass `None` for CPU-only behaviour.
    #[must_use]
    pub fn image(
        &self,
        name: &[u8],
        #[cfg(feature = "nvjpeg")] gpu: Option<&mut gpu::nvjpeg::NvJpegDecoder>,
    ) -> Option<image::ImageDescriptor> {
        let page_dict = self.doc.get_dictionary(self.resource_context_id).ok()?;
        image::resolve_image(
            self.doc,
            page_dict,
            name,
            #[cfg(feature = "nvjpeg")]
            gpu,
        )
    }

    /// Look up a named `ExtGState` resource and return selected parameters.
    ///
    /// Returns `None` if the name is absent or the resource dict is unreadable.
    /// Unknown or unsupported keys in the dict are silently ignored.
    #[must_use]
    pub fn ext_gstate(&self, name: &[u8]) -> Option<ExtGStateParams> {
        let ctx_dict = self.doc.get_dictionary(self.resource_context_id).ok()?;
        let res = image::resolve_dict(self.doc, ctx_dict.get(b"Resources").ok()?)?;
        let eg_dict = image::resolve_dict(self.doc, res.get(b"ExtGState").ok()?)?;
        let gs_ref_or_dict = eg_dict.get(name).ok()?;
        let gs = image::resolve_dict(self.doc, gs_ref_or_dict)?;
        Some(ExtGStateParams::from_dict(gs))
    }

    /// Resolve the named `XObject` and return it as a [`FormXObject`] if its
    /// subtype is `Form`.  Returns `None` for Image `XObjects`, missing names,
    /// or unreadable streams.
    #[must_use]
    pub fn form_xobject(&self, name: &[u8]) -> Option<FormXObject> {
        let ctx_dict = self.doc.get_dictionary(self.resource_context_id).ok()?;
        let res = image::resolve_dict(self.doc, ctx_dict.get(b"Resources").ok()?)?;
        let xobj_dict = image::resolve_dict(self.doc, res.get(b"XObject").ok()?)?;
        let stream_id = match xobj_dict.get(name).ok()? {
            Object::Reference(id) => *id,
            _ => return None,
        };

        let obj = self.doc.get_object(stream_id).ok()?;
        let stream = obj.as_stream().ok()?;

        // Must be a Form subtype.
        if stream.dict.get(b"Subtype").ok()?.as_name().ok()? != b"Form" {
            return None;
        }

        let content = stream.decompressed_content().ok()?;

        // Optional Matrix — defaults to identity if absent or malformed.
        let matrix = read_matrix(&stream.dict).unwrap_or([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);

        let has_own_resources = stream.dict.get(b"Resources").is_ok();

        Some(FormXObject {
            content,
            matrix,
            resources_id: stream_id,
            has_own_resources,
        })
    }

    /// Resolve the named `Shading` resource and return a [`raster::pipe::Pattern`]
    /// plus an approximate bounding box in device space `[xmin, ymin, xmax, ymax]`.
    ///
    /// Returns `None` if the name is absent, the shading type is unsupported,
    /// or any required key is missing.
    #[must_use]
    pub fn shading(
        &self,
        name: &[u8],
        ctm: &[f64; 6],
        page_h: f64,
    ) -> Option<(Box<dyn raster::pipe::Pattern + Send + Sync>, [f64; 4])> {
        let ctx_dict = self.doc.get_dictionary(self.resource_context_id).ok()?;
        shading::resolve_shading(self.doc, ctx_dict, name, ctm, page_h)
    }
}

/// Read a 6-element `Matrix` array from a dictionary, returning `None` if the
/// key is absent or has fewer than 6 numeric entries.
fn read_matrix(dict: &lopdf::Dictionary) -> Option<[f64; 6]> {
    let arr = dict.get(b"Matrix").ok()?.as_array().ok()?;
    if arr.len() < 6 {
        return None;
    }
    let mut m = [0.0f64; 6];
    for (i, obj) in arr.iter().take(6).enumerate() {
        m[i] = match obj {
            Object::Real(r) => f64::from(*r),
            #[expect(
                clippy::cast_precision_loss,
                reason = "PDF matrix values are small integers; precision loss is negligible"
            )]
            Object::Integer(n) => *n as f64,
            _ => return None,
        };
    }
    Some(m)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bm_name_all_variants_round_trip() {
        let pairs: &[(&[u8], BlendMode)] = &[
            (b"Normal", BlendMode::Normal),
            (b"Compatible", BlendMode::Normal),
            (b"Multiply", BlendMode::Multiply),
            (b"Screen", BlendMode::Screen),
            (b"Overlay", BlendMode::Overlay),
            (b"Darken", BlendMode::Darken),
            (b"Lighten", BlendMode::Lighten),
            (b"ColorDodge", BlendMode::ColorDodge),
            (b"ColorBurn", BlendMode::ColorBurn),
            (b"HardLight", BlendMode::HardLight),
            (b"SoftLight", BlendMode::SoftLight),
            (b"Difference", BlendMode::Difference),
            (b"Exclusion", BlendMode::Exclusion),
            (b"Hue", BlendMode::Hue),
            (b"Saturation", BlendMode::Saturation),
            (b"Color", BlendMode::Color),
            (b"Luminosity", BlendMode::Luminosity),
        ];
        for &(name, expected) in pairs {
            assert_eq!(
                bm_name_to_mode(name),
                Some(expected),
                "bm_name_to_mode({:?})",
                String::from_utf8_lossy(name)
            );
        }
    }

    #[test]
    fn bm_name_unknown_returns_none() {
        assert!(bm_name_to_mode(b"Dissolve").is_none());
        assert!(bm_name_to_mode(b"").is_none());
    }

    #[test]
    fn parse_blend_mode_bare_name() {
        let mut dict = lopdf::Dictionary::new();
        dict.set("BM", lopdf::Object::Name(b"Multiply".to_vec()));
        assert_eq!(parse_blend_mode(&dict), Some(BlendMode::Multiply));
    }

    #[test]
    fn parse_blend_mode_array_picks_first_known() {
        let mut dict = lopdf::Dictionary::new();
        dict.set(
            "BM",
            lopdf::Object::Array(vec![
                lopdf::Object::Name(b"Unknown1".to_vec()),
                lopdf::Object::Name(b"Screen".to_vec()),
                lopdf::Object::Name(b"Normal".to_vec()),
            ]),
        );
        assert_eq!(parse_blend_mode(&dict), Some(BlendMode::Screen));
    }

    #[test]
    fn parse_blend_mode_absent_returns_none() {
        let dict = lopdf::Dictionary::new();
        assert!(parse_blend_mode(&dict).is_none());
    }

    #[test]
    fn parse_blend_mode_all_unknown_array_returns_none() {
        let mut dict = lopdf::Dictionary::new();
        dict.set(
            "BM",
            lopdf::Object::Array(vec![
                lopdf::Object::Name(b"Foo".to_vec()),
                lopdf::Object::Name(b"Bar".to_vec()),
            ]),
        );
        assert!(parse_blend_mode(&dict).is_none());
    }
}
