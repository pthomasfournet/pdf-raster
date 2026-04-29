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

pub mod cmap;
pub(crate) mod dict_ext;
pub mod font;
pub mod image;
pub mod shading;
pub mod tiling;

use lopdf::{Dictionary, Document, Object, ObjectId};
use raster::types::BlendMode;

pub use font::{FontDescriptor, PdfFontKind, resolve_font};
/// Re-exported from [`gpu::nvjpeg`] for callers that enable the `nvjpeg` feature.
/// Create via [`gpu::nvjpeg::NvJpegDecoder::new`] with a raw `CUstream` handle,
/// then pass to [`PageRenderer::set_nvjpeg`].
#[cfg(feature = "nvjpeg")]
pub use gpu::nvjpeg::NvJpegDecoder;
pub use image::{ImageColorSpace, ImageDescriptor, ImageFilter, resolve_image};

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
    /// Form bounding box in form user space `[llx, lly, urx, ury]`.
    ///
    /// Used when computing the device-space extent of a transparency group.
    /// Defaults to `[0.0, 0.0, 1.0, 1.0]` when absent.
    pub bbox: [f64; 4],
    /// The object ID of the form stream, used to build a child `PageResources`.
    pub resources_id: ObjectId,
    /// True if the form has its own `Resources` dict; false means it inherits
    /// from the parent context (the caller should keep the parent resources).
    pub has_own_resources: bool,
    /// Transparency group parameters, present when `Group /S /Transparency` is set.
    pub transparency: Option<TransparencyGroupParams>,
}

/// Parameters extracted from a Form `XObject`'s `Group` dictionary.
#[derive(Clone, Copy, Debug)]
pub struct TransparencyGroupParams {
    /// `I` flag: isolated group (default: false).
    pub isolated: bool,
    /// `K` flag: knockout group (default: false).
    pub knockout: bool,
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

    /// The object ID of the current resource context (page or form stream).
    ///
    /// Used by tiling-pattern rendering to inherit the parent context when the
    /// pattern stream carries no own `Resources` dict.
    #[must_use]
    pub const fn resource_context_id(&self) -> ObjectId {
        self.resource_context_id
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
    /// when the `nvjpeg` feature is active.  `gpu_j2k` enables GPU-accelerated
    /// JPEG 2000 decoding for large `JPXDecode` streams when the `nvjpeg2k`
    /// feature is active.  Pass `None` for CPU-only behaviour on either path.
    #[must_use]
    pub fn image(
        &self,
        name: &[u8],
        #[cfg(feature = "nvjpeg")] gpu: Option<&mut gpu::nvjpeg::NvJpegDecoder>,
        #[cfg(feature = "nvjpeg2k")] gpu_j2k: Option<&mut gpu::nvjpeg2k::NvJpeg2kDecoder>,
        #[cfg(feature = "gpu-icc")] gpu_ctx: Option<&gpu::GpuCtx>,
    ) -> Option<image::ImageDescriptor> {
        let page_dict = self.doc.get_dictionary(self.resource_context_id).ok()?;
        image::resolve_image(
            self.doc,
            page_dict,
            name,
            #[cfg(feature = "nvjpeg")]
            gpu,
            #[cfg(feature = "nvjpeg2k")]
            gpu_j2k,
            #[cfg(feature = "gpu-icc")]
            gpu_ctx,
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

        let bbox = read_bbox(&stream.dict).unwrap_or([0.0, 0.0, 1.0, 1.0]);

        let has_own_resources = stream.dict.get(b"Resources").is_ok();

        let transparency = read_transparency_group(self.doc, &stream.dict);

        Some(FormXObject {
            content,
            matrix,
            bbox,
            resources_id: stream_id,
            has_own_resources,
            transparency,
        })
    }

    /// Build a [`FormXObject`] directly from a stream object ID.
    ///
    /// Used for annotation appearance streams, which are referenced by object ID
    /// rather than through the page `Resources/XObject` dict.
    #[must_use]
    pub fn form_from_stream_id(&self, stream_id: ObjectId) -> Option<FormXObject> {
        let obj = self.doc.get_object(stream_id).ok()?;
        let stream = obj.as_stream().ok()?;
        let content = stream.decompressed_content().ok()?;
        let matrix = read_matrix(&stream.dict).unwrap_or([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        let bbox = read_bbox(&stream.dict).unwrap_or([0.0, 0.0, 1.0, 1.0]);
        let has_own_resources = stream.dict.get(b"Resources").is_ok();
        let transparency = read_transparency_group(self.doc, &stream.dict);
        Some(FormXObject {
            content,
            matrix,
            bbox,
            resources_id: stream_id,
            has_own_resources,
            transparency,
        })
    }

    /// Resolve the named `Pattern` resource as a tiling descriptor.
    ///
    /// Returns `None` if the name is absent, the pattern is not `PatternType 1`
    /// (tiling), or any required key is missing.  A debug message is logged for
    /// `PatternType` 2 (shading patterns referenced via `scn`).
    #[must_use]
    pub fn tiling_pattern(&self, name: &[u8]) -> Option<tiling::TilingDescriptor> {
        let ctx_dict = self.doc.get_dictionary(self.resource_context_id).ok()?;
        tiling::resolve_tiling(self.doc, ctx_dict, name)
    }

    /// Resolve the named `Shading` resource.
    ///
    /// Returns `None` if the name is absent, the type is unsupported, or any
    /// required key is missing.
    #[must_use]
    pub fn shading(
        &self,
        name: &[u8],
        ctm: &[f64; 6],
        page_h: f64,
    ) -> Option<shading::ShadingResult> {
        let ctx_dict = self.doc.get_dictionary(self.resource_context_id).ok()?;
        shading::resolve_shading(self.doc, ctx_dict, name, ctm, page_h)
    }

    /// Resolve a `Properties` resource entry to an OCG object ID.
    ///
    /// `props_key` is the name used in `BDC /OC /props_key` — it indexes the
    /// `Resources/Properties` sub-dictionary.  Returns `None` if the key is
    /// absent or the object has no object identity (inline dict OCGs are rare
    /// and not currently supported).
    #[must_use]
    pub fn ocg_object_id(&self, props_key: &[u8]) -> Option<lopdf::ObjectId> {
        let ctx_dict = self.doc.get_dictionary(self.resource_context_id).ok()?;
        let res = image::resolve_dict(self.doc, ctx_dict.get(b"Resources").ok()?)?;
        let props = image::resolve_dict(self.doc, res.get(b"Properties").ok()?)?;
        match props.get(props_key).ok()? {
            Object::Reference(id) => Some(*id),
            _ => None, // Inline-dict OCGs are not resolved (treat as visible).
        }
    }

    /// Check whether an Optional Content Group is visible in the document's
    /// default view configuration (`OCProperties/D`).
    ///
    /// Returns `true` (visible) when:
    /// - `props_key` cannot be resolved to an OCG object (fail-open),
    /// - the document has no `OCProperties` (no layers defined), or
    /// - the default config has no explicit `OFF` list or the group is not in it.
    ///
    /// Returns `false` only when the group appears in the default config `OFF`
    /// array (and that array does not also list the group in `ON`, where `ON`
    /// takes precedence per PDF §8.11.4.3).
    #[must_use]
    pub fn ocg_is_visible(&self, props_key: &[u8]) -> bool {
        let Some(ocg_id) = self.ocg_object_id(props_key) else {
            return true; // Cannot resolve → treat as visible.
        };

        // Locate the default view configuration in the document catalog.
        let Some(d_dict) = self
            .doc
            .catalog()
            .ok()
            .and_then(|cat| {
                cat.get(b"OCProperties").ok().and_then(|o| match o {
                    Object::Dictionary(d) => Some(d),
                    Object::Reference(id) => self.doc.get_dictionary(*id).ok(),
                    _ => None,
                })
            })
            .and_then(|ocp| {
                ocp.get(b"D").ok().and_then(|o| match o {
                    Object::Dictionary(d) => Some(d),
                    Object::Reference(id) => self.doc.get_dictionary(*id).ok(),
                    _ => None,
                })
            })
        else {
            return true; // No OCProperties → all groups visible.
        };

        // Check the OFF list.  Per PDF §8.11.4.3, if a group appears in ON it
        // overrides OFF; in practice most documents use only one of the two.
        let in_off = is_id_in_ref_array(self.doc, d_dict, b"OFF", ocg_id);
        let in_on = is_id_in_ref_array(self.doc, d_dict, b"ON", ocg_id);

        !in_off || in_on
    }
}

/// Return `true` when `target_id` appears in the named array of object
/// references inside `dict`.
fn is_id_in_ref_array(
    doc: &Document,
    dict: &lopdf::Dictionary,
    key: &[u8],
    target_id: lopdf::ObjectId,
) -> bool {
    let Ok(arr_obj) = dict.get(key) else {
        return false;
    };
    let arr = match arr_obj {
        Object::Array(a) => a,
        Object::Reference(id) => {
            if let Ok(o) = doc.get_object(*id)
                && let Ok(a) = o.as_array()
            {
                return a
                    .iter()
                    .any(|x| matches!(x, Object::Reference(r) if *r == target_id));
            }
            return false;
        }
        _ => return false,
    };
    arr.iter()
        .any(|x| matches!(x, Object::Reference(r) if *r == target_id))
}

/// Parse a fixed-length array of `f64` values from a PDF dictionary.
///
/// Returns `None` if the key is absent, the value is not an array, the array
/// has fewer than `N` elements, or any of the first `N` elements is not numeric.
pub(crate) fn read_f64_n<const N: usize>(dict: &lopdf::Dictionary, key: &[u8]) -> Option<[f64; N]> {
    let arr = dict.get(key).ok()?.as_array().ok()?;
    if arr.len() < N {
        return None;
    }
    let mut out = [0.0f64; N];
    for (i, obj) in arr.iter().take(N).enumerate() {
        out[i] = match obj {
            Object::Real(v) => f64::from(*v),
            #[expect(
                clippy::cast_precision_loss,
                reason = "PDF numeric values fit within f64 mantissa in all real-world uses"
            )]
            Object::Integer(v) => *v as f64,
            _ => return None,
        };
    }
    Some(out)
}

/// Read the `BBox` array `[llx, lly, urx, ury]` from a dictionary.
///
/// Returns `None` if absent or fewer than 4 numeric entries.
fn read_bbox(dict: &lopdf::Dictionary) -> Option<[f64; 4]> {
    let mut r = read_f64_n::<4>(dict, b"BBox")?;
    // Normalise so llx ≤ urx and lly ≤ ury — PDF allows inverted BBox.
    if r[0] > r[2] {
        r.swap(0, 2);
    }
    if r[1] > r[3] {
        r.swap(1, 3);
    }
    Some(r)
}

/// Extract transparency group parameters from a Form `XObject` stream dictionary.
///
/// Returns `Some` when `Group /S /Transparency` is present; `None` otherwise.
fn read_transparency_group(
    doc: &Document,
    dict: &lopdf::Dictionary,
) -> Option<TransparencyGroupParams> {
    let grp_obj = dict.get(b"Group").ok()?;
    let grp = match grp_obj {
        Object::Dictionary(d) => d,
        Object::Reference(id) => doc.get_dictionary(*id).ok()?,
        _ => return None,
    };
    // Must be /S /Transparency.
    if grp.get(b"S").ok()?.as_name().ok()? != b"Transparency" {
        return None;
    }
    let bool_flag = |key: &[u8]| {
        grp.get(key)
            .ok()
            .and_then(|o| o.as_bool().ok())
            .unwrap_or(false)
    };
    Some(TransparencyGroupParams {
        isolated: bool_flag(b"I"),
        knockout: bool_flag(b"K"),
    })
}

/// Read a 6-element `Matrix` array from a dictionary, returning `None` if the
/// key is absent or has fewer than 6 numeric entries.
fn read_matrix(dict: &lopdf::Dictionary) -> Option<[f64; 6]> {
    read_f64_n::<6>(dict, b"Matrix")
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
