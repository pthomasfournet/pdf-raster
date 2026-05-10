//! PDF resource resolution — fonts, colour spaces, `XObject`s, shadings.
//!
//! The [`PageResources`] struct holds a reference to the [`pdf::Document`]
//! and the current page's [`ObjectId`], and provides lazy-loading helpers
//! for each resource category.
//!
//! # Design
//!
//! `pdf::Document` is immutable after load; resource dicts are accessed
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

use std::sync::{Arc, OnceLock};

use pdf::{Dictionary, Document, Object, ObjectId};
use raster::types::BlendMode;

pub use font::{FontDescriptor, PdfFontKind, resolve_font};
/// Re-exported from [`gpu::nvjpeg`] for callers that enable the `nvjpeg` feature.
/// Create via [`gpu::nvjpeg::NvJpegDecoder::new`] with a raw `CUstream` handle,
/// then pass to the page renderer's `set_nvjpeg` configuration entry point.
#[cfg(feature = "nvjpeg")]
pub use gpu::nvjpeg::NvJpegDecoder;
pub use image::{IMAGE_FILTER_COUNT, ImageColorSpace, ImageDescriptor, ImageFilter, resolve_image};

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
    let obj = d.get(b"BM")?;
    match obj {
        Object::Name(n) => bm_name_to_mode(n),
        // Array is a viewer-preference priority list; use the first recognised name.
        Object::Array(arr) => arr
            .iter()
            .find_map(|o| o.as_name().and_then(bm_name_to_mode)),
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
    match d.get(key)? {
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
    d.get(key)?.as_i64().map(|n| n as i32)
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
    /// Cached context dictionary (page or form stream dict). `pdf::Document::get_dict`
    /// clones the underlying [`Dictionary`] on every call; resolving 5–6 resource
    /// types per page would otherwise pay that cost every time. Populated lazily on
    /// first access so error-path call sites that never look up resources don't pay.
    ctx_dict: OnceLock<Option<Arc<Dictionary>>>,
}

impl<'doc> PageResources<'doc> {
    /// Construct a [`PageResources`] for the given page.
    #[must_use]
    pub const fn new(doc: &'doc Document, page_id: ObjectId) -> Self {
        Self {
            doc,
            resource_context_id: page_id,
            ctx_dict: OnceLock::new(),
        }
    }

    /// Construct a child [`PageResources`] scoped to the given form's resources.
    ///
    /// If `form.has_own_resources` is false the form inherits the parent context;
    /// use `self` (the parent) directly instead of calling this.
    #[must_use]
    pub const fn for_form(&self, form: &FormXObject) -> Self {
        let resource_context_id = if form.has_own_resources {
            form.resources_id
        } else {
            self.resource_context_id
        };
        Self {
            doc: self.doc,
            resource_context_id,
            ctx_dict: OnceLock::new(),
        }
    }

    /// The underlying [`pdf::Document`] (read-only).
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

    /// Return the cached context dictionary, resolving it on first call.
    ///
    /// The dict is shared via `Arc` so repeated lookups across `image()`,
    /// `ext_gstate()`, `form_xobject()`, etc. all hit the same allocation
    /// instead of cloning the (potentially large) page dict 5–6 times per page.
    fn ctx_dict(&self) -> Option<Arc<Dictionary>> {
        self.ctx_dict
            .get_or_init(|| self.doc.get_dictionary(self.resource_context_id).ok())
            .clone()
    }

    /// Resolve the font dictionary for the named resource (e.g. `b"F1"`).
    ///
    /// Returns `None` if the resource name is not present in the resource dict.
    #[must_use]
    pub fn font_dict(&self, name: &[u8]) -> Option<font::FontDescriptor> {
        let fonts = self.doc.get_page_fonts(self.resource_context_id).ok()?;
        let entry = fonts.get(name)?;
        let dict = resolve_dict(self.doc, entry)?;
        Some(font::resolve_font(self.doc, &dict))
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
    #[cfg_attr(
        any(
            all(
                feature = "gpu-icc",
                feature = "cache",
                any(feature = "nvjpeg", feature = "vaapi", feature = "nvjpeg2k")
            ),
            all(
                feature = "nvjpeg",
                feature = "vaapi",
                feature = "nvjpeg2k",
                any(feature = "gpu-icc", feature = "cache")
            )
        ),
        expect(
            clippy::too_many_arguments,
            reason = "thin forwarder to resolve_image; argument count is feature-gated"
        )
    )]
    pub fn image(
        &self,
        name: &[u8],
        #[cfg(feature = "nvjpeg")] gpu: Option<&mut gpu::nvjpeg::NvJpegDecoder>,
        #[cfg(feature = "vaapi")] vaapi: Option<&gpu::JpegQueueHandle>,
        #[cfg(feature = "nvjpeg2k")] gpu_j2k: Option<&mut gpu::nvjpeg2k::NvJpeg2kDecoder>,
        #[cfg(feature = "gpu-icc")] gpu_ctx: Option<&gpu::GpuCtx>,
        #[cfg(feature = "gpu-icc")] clut_cache: Option<&mut image::IccClutCache>,
        #[cfg(feature = "cache")] image_cache: Option<
            &std::sync::Arc<gpu::cache::DeviceImageCache>,
        >,
        #[cfg(feature = "cache")] doc_id: Option<gpu::cache::DocId>,
    ) -> Option<image::ImageDescriptor> {
        let page_dict = self.ctx_dict()?;
        image::resolve_image(
            self.doc,
            &page_dict,
            name,
            #[cfg(feature = "nvjpeg")]
            gpu,
            #[cfg(feature = "vaapi")]
            vaapi,
            #[cfg(feature = "nvjpeg2k")]
            gpu_j2k,
            #[cfg(feature = "gpu-icc")]
            gpu_ctx,
            #[cfg(feature = "gpu-icc")]
            clut_cache,
            #[cfg(feature = "cache")]
            image_cache,
            #[cfg(feature = "cache")]
            doc_id,
        )
    }

    /// Look up a named `ExtGState` resource and return selected parameters.
    ///
    /// Returns `None` if the name is absent or the resource dict is unreadable.
    /// Unknown or unsupported keys in the dict are silently ignored.
    #[must_use]
    pub fn ext_gstate(&self, name: &[u8]) -> Option<ExtGStateParams> {
        let ctx_dict = self.ctx_dict()?;
        let res = resolve_dict(self.doc, ctx_dict.get(b"Resources")?)?;
        let eg_dict = resolve_dict(self.doc, res.get(b"ExtGState")?)?;
        let gs_ref_or_dict = eg_dict.get(name)?;
        let gs = resolve_dict(self.doc, gs_ref_or_dict)?;
        Some(ExtGStateParams::from_dict(&gs))
    }

    /// Resolve the named `XObject` and return it as a [`FormXObject`] if its
    /// subtype is `Form`.  Returns `None` for Image `XObjects`, missing names,
    /// or unreadable streams.
    #[must_use]
    pub fn form_xobject(&self, name: &[u8]) -> Option<FormXObject> {
        let ctx_dict = self.ctx_dict()?;
        let res = resolve_dict(self.doc, ctx_dict.get(b"Resources")?)?;
        let xobj_dict = resolve_dict(self.doc, res.get(b"XObject")?)?;
        let Object::Reference(id) = xobj_dict.get(name)? else {
            return None;
        };
        let stream_id = *id;

        let obj = self.doc.get_object(stream_id).ok()?;
        let stream = obj.as_stream()?;

        // Must be a Form subtype.
        if stream.dict.get(b"Subtype")?.as_name()? != b"Form" {
            return None;
        }

        let content = stream.decompressed_content().ok()?;

        // Optional Matrix — defaults to identity if absent or malformed.
        let matrix = read_matrix(&stream.dict).unwrap_or([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);

        let bbox = read_bbox(&stream.dict).unwrap_or([0.0, 0.0, 1.0, 1.0]);

        let has_own_resources = stream.dict.get(b"Resources").is_some();

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
        let stream = obj.as_stream()?;
        let content = stream.decompressed_content().ok()?;
        let matrix = read_matrix(&stream.dict).unwrap_or([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        let bbox = read_bbox(&stream.dict).unwrap_or([0.0, 0.0, 1.0, 1.0]);
        let has_own_resources = stream.dict.get(b"Resources").is_some();
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
        let ctx_dict = self.ctx_dict()?;
        tiling::resolve_tiling(self.doc, &ctx_dict, name)
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
        let ctx_dict = self.ctx_dict()?;
        shading::resolve_shading(self.doc, &ctx_dict, name, ctm, page_h)
    }

    /// Resolve a `Properties` resource entry to an OCG object ID.
    ///
    /// `props_key` is the name used in `BDC /OC /props_key` — it indexes the
    /// `Resources/Properties` sub-dictionary.  Returns `None` if the key is
    /// absent or the object has no object identity (inline dict OCGs are rare
    /// and not currently supported).
    #[must_use]
    pub fn ocg_object_id(&self, props_key: &[u8]) -> Option<ObjectId> {
        let ctx_dict = self.ctx_dict()?;
        let res = resolve_dict(self.doc, ctx_dict.get(b"Resources")?)?;
        let props = resolve_dict(self.doc, res.get(b"Properties")?)?;
        match props.get(props_key)? {
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
                cat.get(b"OCProperties").and_then(|o| match o {
                    Object::Dictionary(d) => Some(d.clone()),
                    Object::Reference(id) => {
                        self.doc.get_dictionary(*id).ok().map(|a| (*a).clone())
                    }
                    _ => None,
                })
            })
            .and_then(|ocp| {
                ocp.get(b"D").and_then(|o| match o {
                    Object::Dictionary(d) => Some(d.clone()),
                    Object::Reference(id) => {
                        self.doc.get_dictionary(*id).ok().map(|a| (*a).clone())
                    }
                    _ => None,
                })
            })
        else {
            return true; // No OCProperties → all groups visible.
        };

        // Check the OFF list.  Per PDF §8.11.4.3, if a group appears in ON it
        // overrides OFF; in practice most documents use only one of the two.
        let in_off = is_id_in_ref_array(self.doc, &d_dict, b"OFF", ocg_id);
        let in_on = is_id_in_ref_array(self.doc, &d_dict, b"ON", ocg_id);

        !in_off || in_on
    }
}

/// Return `true` when `target_id` appears in the named array of object
/// references inside `dict`.
fn is_id_in_ref_array(doc: &Document, dict: &Dictionary, key: &[u8], target_id: ObjectId) -> bool {
    let Some(arr_obj) = dict.get(key) else {
        return false;
    };
    let arr_owned: Vec<Object>;
    let arr: &[Object] = match arr_obj {
        Object::Array(a) => a,
        Object::Reference(id) => {
            if let Ok(o) = doc.get_object(*id)
                && let Some(a) = o.as_array()
            {
                arr_owned = a.to_vec();
                &arr_owned
            } else {
                return false;
            }
        }
        _ => return false,
    };
    arr.iter()
        .any(|x| matches!(x, Object::Reference(r) if *r == target_id))
}

/// Convert a [`pdf::Object`] (Real or Integer) to `f64`.
///
/// Returns `None` for any non-numeric object type.
pub(crate) fn obj_to_f64(obj: &Object) -> Option<f64> {
    match obj {
        Object::Real(r) => Some(f64::from(*r)),
        #[expect(
            clippy::cast_precision_loss,
            reason = "PDF numeric values fit within f64 mantissa in all real-world uses"
        )]
        Object::Integer(n) => Some(*n as f64),
        _ => None,
    }
}

/// Read a single numeric value from a dictionary key.
///
/// Returns `None` if the key is absent or the value is not a Real or Integer.
pub(crate) fn read_f64_1(dict: &Dictionary, key: &[u8]) -> Option<f64> {
    obj_to_f64(dict.get(key)?)
}

/// Parse a fixed-length array of `f64` values from a PDF dictionary.
///
/// Returns `None` if the key is absent, the value is not an array, the array
/// has fewer than `N` elements, or any of the first `N` elements is not numeric.
pub(crate) fn read_f64_n<const N: usize>(dict: &Dictionary, key: &[u8]) -> Option<[f64; N]> {
    let arr = dict.get(key)?.as_array()?;
    if arr.len() < N {
        return None;
    }
    let mut out = [0.0f64; N];
    for (i, obj) in arr.iter().take(N).enumerate() {
        out[i] = obj_to_f64(obj)?;
    }
    Some(out)
}

/// Read the `BBox` array `[llx, lly, urx, ury]` from a dictionary.
///
/// Returns `None` if absent or fewer than 4 numeric entries.
/// Normalises the result so that `llx ≤ urx` and `lly ≤ ury` — PDF allows
/// inverted `BBox` values.
pub(crate) fn read_bbox(dict: &Dictionary) -> Option<[f64; 4]> {
    let mut r = read_f64_n::<4>(dict, b"BBox")?;
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
fn read_transparency_group(doc: &Document, dict: &Dictionary) -> Option<TransparencyGroupParams> {
    let grp_obj = dict.get(b"Group")?;
    let grp = resolve_dict(doc, grp_obj)?;
    // Must be /S /Transparency.
    if grp.get(b"S")?.as_name()? != b"Transparency" {
        return None;
    }
    let bool_flag = |key: &[u8]| grp.get(key).and_then(Object::as_bool).unwrap_or(false);
    Some(TransparencyGroupParams {
        isolated: bool_flag(b"I"),
        knockout: bool_flag(b"K"),
    })
}

/// Read a 6-element `Matrix` array from a dictionary, returning `None` if the
/// key is absent or has fewer than 6 numeric entries.
pub(crate) fn read_matrix(dict: &Dictionary) -> Option<[f64; 6]> {
    read_f64_n::<6>(dict, b"Matrix")
}

/// Dereference a PDF `Object` to an owned `Dictionary`.
///
/// Accepts `Dictionary` (cloned) or `Reference` (dereferenced via `doc` and
/// cloned).  Returns owned rather than borrowed because the underlying lazy
/// document parser materialises objects on demand and hands back `Arc`s, not
/// long-lived borrows.  Dicts are small (a few hundred entries at most), so
/// the clone cost is negligible.
///
/// Use [`resolve_stream_dict`] when the referent may be a stream object.
pub(crate) fn resolve_dict(doc: &Document, obj: &Object) -> Option<Dictionary> {
    match obj {
        Object::Dictionary(d) => Some(d.clone()),
        Object::Reference(id) => doc.get_dictionary(*id).ok().map(|a| (*a).clone()),
        _ => None,
    }
}

/// Like [`resolve_dict`] but also handles `Reference → Stream → stream.dict`.
///
/// Needed for `ICCBased` colour spaces where the second array element is a
/// `Reference` to a stream whose dictionary carries the ICC metadata.
pub(crate) fn resolve_stream_dict(doc: &Document, obj: &Object) -> Option<Dictionary> {
    match obj {
        Object::Dictionary(d) => Some(d.clone()),
        Object::Reference(id) => {
            let referent = doc.get_object(*id).ok()?;
            match referent.as_ref() {
                Object::Stream(s) => Some(s.dict.clone()),
                // Reference to a plain dictionary: share the plain-dict path.
                other => resolve_dict(doc, other),
            }
        }
        _ => None,
    }
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
        let mut dict = Dictionary::new();
        dict.set("BM", Object::Name(b"Multiply".to_vec()));
        assert_eq!(parse_blend_mode(&dict), Some(BlendMode::Multiply));
    }

    #[test]
    fn parse_blend_mode_array_picks_first_known() {
        let mut dict = Dictionary::new();
        dict.set(
            "BM",
            Object::Array(vec![
                Object::Name(b"Unknown1".to_vec()),
                Object::Name(b"Screen".to_vec()),
                Object::Name(b"Normal".to_vec()),
            ]),
        );
        assert_eq!(parse_blend_mode(&dict), Some(BlendMode::Screen));
    }

    #[test]
    fn parse_blend_mode_absent_returns_none() {
        let dict = Dictionary::new();
        assert!(parse_blend_mode(&dict).is_none());
    }

    #[test]
    fn parse_blend_mode_all_unknown_array_returns_none() {
        let mut dict = Dictionary::new();
        dict.set(
            "BM",
            Object::Array(vec![
                Object::Name(b"Foo".to_vec()),
                Object::Name(b"Bar".to_vec()),
            ]),
        );
        assert!(parse_blend_mode(&dict).is_none());
    }
}
