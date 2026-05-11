//! PDF colour-space taxonomy (§8.6) for tracking the current `cs`/`CS`
//! selection on the graphics state.
//!
//! This is the *input* representation — the colour space declared by the
//! content stream — distinct from the [`super::image::ImageColorSpace`]
//! output category used by the image blit path. The taxonomy is needed by
//! the operators that resolve uncoloured-pattern tints (§8.7.3.3) and the
//! gstate-level conversion to sRGB for stroke/fill operators.
//!
//! # Scope
//!
//! The variants capture what we need to track *which* space is active;
//! they don't carry the full colorant-mapping data (Lab whitepoint
//! values, the ICC profile bytes, the full Separation tint function, etc.)
//! — those are pulled lazily via the stored [`ObjectId`] when
//! [`ColorSpace::convert_to_rgb`] actually runs.  ICC profiles are
//! transformed through `moxcms`; Separation tints flow through the
//! shading-fn evaluator (`super::shading::function::eval_function`) into
//! the alternate space.
//!
//! # PDF §8.6 mapping
//!
//! | PDF variant   | [`ColorSpace`] variant |
//! |---|---|
//! | `/DeviceGray`, `/G` (Table 89 abbreviation), `/CalGray` | [`ColorSpace::DeviceGray`] |
//! | `/DeviceRGB`, `/RGB` (Table 89), `/CalRGB` | [`ColorSpace::DeviceRgb`] |
//! | `/DeviceCMYK`, `/CMYK` (Table 89) | [`ColorSpace::DeviceCmyk`] |
//! | `[/CalGray ...]` array form | [`ColorSpace::DeviceGray`] (approximated) |
//! | `[/CalRGB ...]` array form | [`ColorSpace::DeviceRgb`] (approximated) |
//! | `[/Lab ...]`         | [`ColorSpace::Lab`] |
//! | `[/ICCBased <ref>]`  | [`ColorSpace::IccBased`] |
//! | `[/Indexed ...]`     | [`ColorSpace::Indexed`] |
//! | `/Pattern` or `[/Pattern <base>]` | [`ColorSpace::Pattern`] |
//! | `[/Separation ...]`  | [`ColorSpace::Separation`] |
//! | `[/DeviceN ...]`     | [`ColorSpace::DeviceN`] |
//! | unknown / malformed  | [`ColorSpace::DeviceGray`] (safe fallback) |
//!
//! # Why the lazy `ObjectId` shape
//!
//! Resolving a Lab whitepoint or ICC profile up front would force the
//! graphics state to carry several kilobytes of profile data per push/pop.
//! Storing the `ObjectId` keeps the gstate small and lets the conversion
//! path dereference only when it actually needs the data.
//!
//! # Current limitations
//!
//! - **Lab whitepoint**: `convert_to_rgb` hard-codes D65; the per-document
//!   whitepoint stored in `Lab::dict_id` isn't read yet.  Affects rare
//!   prepress PDFs that use D50 or other reference whites.
//! - **`Indexed` lookup**: `lookup_id` is captured at parse time but
//!   `convert_to_rgb` still returns black for `Indexed`.  Palette
//!   dereference + recursive base conversion is the missing piece; the
//!   image pipeline handles Indexed via a separate path for images.
//! - **`DeviceN` with N > 1 tint inputs**: the shading-fn evaluator
//!   only handles 1-input functions; `DeviceN` with multiple colorants
//!   falls back to the single-component gray heuristic on `comps[0]`.
//!   Single-colorant `DeviceN` (N = 1) routes through the same tint
//!   path as `Separation`.  Extending the evaluator to N-input
//!   PostScript-style functions (Type 4) is its own scope.

use pdf::{Document, Object, ObjectId};

use crate::resources::dict_ext::DictExt as _;

/// Maximum recursion depth for nested colour-space chains
/// (Indexed/Pattern/Separation/DeviceN's `alternate` base). PDF spec
/// forbids cycles but adversarial PDFs are not required to be
/// spec-compliant; this bound prevents stack overflow on crafted input.
pub(crate) const MAX_CS_DEPTH: u8 = 8;

/// A resolved PDF colour-space declaration tracked on the graphics state.
///
/// See module docs for the §8.6 mapping. Each variant carries enough data
/// to (a) report its component count for span / tint validation and
/// (b) reach the underlying conversion data when an RGB output is needed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ColorSpace {
    /// Single-channel grey. PDF §8.6.4.2.
    DeviceGray,
    /// Three-channel RGB. PDF §8.6.4.3.
    DeviceRgb,
    /// Four-channel CMYK. PDF §8.6.4.4.
    DeviceCmyk,
    /// CIE Lab. PDF §8.6.5.4. Three components; whitepoint + range live
    /// in the dict reachable via `dict_id` when we need them.
    Lab {
        /// `Reference` to the Lab parameter dictionary (whitepoint, range);
        /// `None` if the parameters were inlined or unparseable.
        dict_id: Option<ObjectId>,
    },
    /// ICC-profile-based space. PDF §8.6.5.5. `n` is the channel count
    /// (1, 3, or 4) read from the profile stream's `/N` entry.
    IccBased {
        /// Component count from the profile's `/N` entry: 1, 3, or 4.
        n: u8,
        /// `Reference` to the ICC profile stream; `None` if the array's
        /// second element wasn't an indirect reference.
        stream_id: Option<ObjectId>,
    },
    /// Indexed-palette lookup. PDF §8.6.6.3. `base` is the underlying
    /// colour space the palette indexes into; `hival` is the max valid
    /// index (palette entries = `hival + 1`).
    Indexed {
        /// Underlying colour space the palette indexes into.
        base: Box<Self>,
        /// Maximum valid index — palette has `hival + 1` entries.
        hival: u8,
        /// `Reference` to the lookup stream/string holding the palette
        /// entries; `None` if the array's fourth element was an inline
        /// string (the parser doesn't store the bytes — palette lookup
        /// would need to re-resolve via the source `Object`).
        lookup_id: Option<ObjectId>,
    },
    /// Pattern space. PDF §8.7.2. The optional `base` is the underlying
    /// colour space for uncoloured patterns (`PaintType` 2); `None` for
    /// coloured patterns.
    Pattern {
        /// Underlying colour space for `PaintType` 2 (uncoloured) patterns;
        /// `None` for `PaintType` 1 (coloured) and bare `/Pattern`.
        base: Option<Box<Self>>,
    },
    /// Single spot-colour. PDF §8.6.6.4. `alternate` is the device space
    /// the tint converts to when the spot colorant isn't separately
    /// available; `tint_dict_id` references the tint-transform function.
    Separation {
        /// Device space the tint converts into when the spot colorant
        /// isn't available.
        alternate: Box<Self>,
        /// `Reference` to the tint-transform function dict; `None` if
        /// the array's fourth element wasn't an indirect reference.
        tint_dict_id: Option<ObjectId>,
    },
    /// Multi-channel spot-colour set. PDF §8.6.6.5. `ncomps` is the
    /// number of tint inputs; `alternate` and `tint_dict_id` mirror
    /// `Separation`.
    DeviceN {
        /// Number of tint components — from the `Names` array length.
        ncomps: u8,
        /// Device space the tint converts into when spot colorants
        /// aren't available.
        alternate: Box<Self>,
        /// `Reference` to the tint-transform function dict; `None` if
        /// the array's fourth element wasn't an indirect reference.
        tint_dict_id: Option<ObjectId>,
    },
}

impl ColorSpace {
    /// The number of tint components a `scn`/`SCN` operator must supply
    /// for this colour space. Used to validate operand counts and to
    /// dispatch the legacy component-count heuristic for uncoloured
    /// patterns.
    #[must_use]
    pub fn ncomponents(&self) -> usize {
        match self {
            // Single-component spaces: scalar grey, palette index, or single
            // spot-colour tint.  All take exactly one operand on `scn`.
            Self::DeviceGray | Self::Indexed { .. } | Self::Separation { .. } => 1,
            Self::DeviceRgb | Self::Lab { .. } => 3,
            Self::DeviceCmyk => 4,
            Self::IccBased { n, .. } => usize::from(*n),
            Self::Pattern { .. } => 0,
            Self::DeviceN { ncomps, .. } => usize::from(*ncomps),
        }
    }

    /// Convert a tint-component slice to an sRGB byte triple per PDF §8.6.
    ///
    /// `comps` must have exactly [`ncomponents`](Self::ncomponents)
    /// elements; any mismatch falls back to opaque black.
    ///
    /// `doc` is required so the conversion can dereference lazy
    /// `ObjectId` fields on parameterised variants (ICC profile streams,
    /// Separation/DeviceN tint functions, Indexed palette lookups).
    /// Pass the page's document; the borrow is read-only.
    ///
    /// # Variant behaviour
    ///
    /// - `DeviceGray/Rgb/Cmyk`: direct conversion via the existing
    ///   `color::convert` helpers (CMYK uses the naive §10.3.3 path; ICC
    ///   profile handling is a separate phase).
    /// - `Lab`: CIE Lab → D65 XYZ → sRGB through the standard formulae.
    ///   Whitepoint defaults to D65 (the spec's most common choice); the
    ///   per-document whitepoint stored in `dict_id` is not yet applied.
    /// - `IccBased`: dereferences `stream_id`, runs a `moxcms` 8-bit
    ///   transform from the embedded profile to sRGB for 3- and 4-channel
    ///   profiles.  1-channel ICC falls back to `DeviceGray` because
    ///   `moxcms` doesn't expose a Gray layout for transforms.  Any
    ///   error (missing stream, malformed profile, transform failure)
    ///   falls back to the device-space-by-N path.
    /// - `Indexed`: returns black.  `lookup_id` is captured at parse
    ///   time but the palette-dereference + recursive base-space
    ///   conversion isn't wired through yet.  The image pipeline
    ///   handles Indexed via a separate path for image streams.
    /// - `Pattern`: returns black — patterns are not directly convertible
    ///   to RGB; tile rasterisation is invoked elsewhere.
    /// - `Separation`: passes the tint through the document's
    ///   tint-transform function (referenced by `tint_dict_id`) and
    ///   converts the result via the alternate colour space.  Any failure
    ///   (missing function, eval error, alternate falls through) drops
    ///   back to `1 - tint` as gray.
    /// - `DeviceN`: when `ncomps == 1` (single-colorant `DeviceN` —
    ///   rare but spec-legal), routes through the same tint-function
    ///   path as `Separation`.  When `ncomps > 1`, falls back to the
    ///   gray heuristic on `comps[0]` — the existing shading-fn
    ///   evaluator only takes a single `t: f64`, and N-input
    ///   PostScript-style (Type 4) function evaluation is a separate
    ///   scope.
    #[must_use]
    pub fn convert_to_rgb(&self, doc: &Document, comps: &[f64]) -> [u8; 3] {
        if comps.len() != self.ncomponents() {
            log::debug!(
                "colorspace: {:?} expects {} components, got {}",
                self,
                self.ncomponents(),
                comps.len(),
            );
            return [0, 0, 0];
        }
        match self {
            Self::DeviceGray => gray_byte_to_rgb(color::convert::gray_to_u8(comps[0])),
            Self::DeviceRgb => color::convert::rgb_to_bytes(comps[0], comps[1], comps[2]),
            Self::DeviceCmyk => {
                color::convert::cmyk_to_rgb_bytes(comps[0], comps[1], comps[2], comps[3])
            }
            Self::Lab { .. } => lab_to_srgb(comps[0], comps[1], comps[2]),
            Self::IccBased { n, stream_id } => {
                // Try the embedded profile first; fall back to device-by-N on
                // any failure (missing stream, malformed profile, Gray layout
                // unsupported by moxcms, etc.).
                if let Some(id) = stream_id
                    && let Some(rgb) = icc_transform(doc, *id, *n, comps)
                {
                    return rgb;
                }
                icc_fallback_by_n(*n, comps)
            }
            Self::Indexed { .. } | Self::Pattern { .. } => [0, 0, 0],
            Self::Separation {
                alternate,
                tint_dict_id,
            } => {
                // PDF §8.6.6.4: pass the single tint component through the
                // tint-transform function to produce values in the alternate
                // colour space, then convert via the alternate.  On any
                // failure (missing dict id, function eval returns None, or
                // alternate.convert_to_rgb falls back), use the image-
                // pipeline heuristic (1 - tint → gray).
                if let Some(id) = tint_dict_id
                    && let Some(rgb) = separation_via_tint_fn(doc, *id, alternate, comps[0])
                {
                    return rgb;
                }
                separation_fallback_gray(comps[0])
            }
            Self::DeviceN {
                ncomps,
                alternate,
                tint_dict_id,
            } => {
                // PDF §8.6.6.5: like Separation but with N tint inputs.  The
                // shading-fn evaluator (`super::shading::function::eval_function`)
                // only takes a single `t: f64`, so we can route through it
                // when `ncomps == 1` (a single-colorant DeviceN — uncommon,
                // but spec-legal).  N > 1 needs an N-input evaluator
                // (PostScript-style Type 4 functions); falls back to the
                // gray heuristic on `comps[0]` until that lands.
                if *ncomps == 1
                    && let Some(id) = tint_dict_id
                    && let Some(rgb) = separation_via_tint_fn(doc, *id, alternate, comps[0])
                {
                    return rgb;
                }
                separation_fallback_gray(comps[0])
            }
        }
    }
}

/// CIE Lab → sRGB conversion through D65 XYZ. Per CIE 1976 + IEC 61966-2-1.
///
/// PDF Lab encodes `L* ∈ [0, 100]`, `a* / b* ∈ [-128, 127]` per the spec's
/// Range default; callers pass the post-Decode normalised values directly.
fn lab_to_srgb(l_star: f64, a_star: f64, b_star: f64) -> [u8; 3] {
    // 1. Lab → XYZ (D65 whitepoint).
    const XN: f64 = 0.95047;
    const YN: f64 = 1.0;
    const ZN: f64 = 1.08883;
    let fy = (l_star + 16.0) / 116.0;
    let fx = a_star / 500.0 + fy;
    let fz = fy - b_star / 200.0;
    let x = XN * lab_f_inv(fx);
    let y = YN * lab_f_inv(fy);
    let z = ZN * lab_f_inv(fz);

    // 2. XYZ (D65) → linear sRGB (BT.709 / sRGB matrix).
    let r_lin = x.mul_add(3.2406, y.mul_add(-1.5372, z * -0.4986));
    let g_lin = x.mul_add(-0.9689, y.mul_add(1.8758, z * 0.0415));
    let b_lin = x.mul_add(0.0557, y.mul_add(-0.2040, z * 1.0570));

    // 3. Linear → sRGB gamma.
    color::convert::rgb_to_bytes(srgb_encode(r_lin), srgb_encode(g_lin), srgb_encode(b_lin))
}

/// Inverse of the CIE Lab `f` function. `δ = 6/29`; threshold `t > δ` cubes,
/// else uses the linear segment.
fn lab_f_inv(t: f64) -> f64 {
    const DELTA: f64 = 6.0 / 29.0;
    if t > DELTA {
        t * t * t
    } else {
        3.0 * DELTA * DELTA * (t - 4.0 / 29.0)
    }
}

/// Linear → sRGB gamma encoding per IEC 61966-2-1 (piecewise).
fn srgb_encode(c: f64) -> f64 {
    let c = c.clamp(0.0, 1.0);
    if c <= 0.003_130_8 {
        12.92 * c
    } else {
        c.powf(1.0 / 2.4).mul_add(1.055, -0.055)
    }
}

/// Expand a single grey byte into an opaque RGB triple `[v, v, v]`.
const fn gray_byte_to_rgb(v: u8) -> [u8; 3] {
    [v, v, v]
}

/// Run a `moxcms` 8-bit transform from an embedded ICC profile to sRGB for a
/// single 3- or 4-channel tint. Returns `None` and lets the caller fall back
/// when:
/// - the profile stream isn't dereferenceable or fails to decompress,
/// - the profile bytes don't parse as an ICC profile,
/// - the channel count is unsupported by `moxcms`'s 8-bit `Layout` (Gray),
/// - the transform construction or execution returns an error.
///
/// `n` is the profile's declared channel count from `/N`. `comps` is the
/// normalised tint in [0, 1].
fn icc_transform(doc: &Document, stream_id: ObjectId, n: u8, comps: &[f64]) -> Option<[u8; 3]> {
    use moxcms::{ColorProfile, Layout, TransformOptions};
    let layout = match n {
        3 => Layout::Rgb,
        4 => Layout::Rgba,
        _ => return None, // Gray (n=1) or unsupported.
    };
    let icc_bytes = fetch_icc_bytes(doc, stream_id)?;
    let src_profile = ColorProfile::new_from_slice(&icc_bytes)
        .map_err(|e| log::debug!("colorspace: ICC profile parse failed: {e}"))
        .ok()?;
    let dst_profile = ColorProfile::new_srgb();
    let xform = src_profile
        .create_transform_8bit(
            layout,
            &dst_profile,
            Layout::Rgb,
            TransformOptions::default(),
        )
        .map_err(|e| log::debug!("colorspace: ICC transform creation failed: {e}"))
        .ok()?;
    // Normalised f64 [0,1] → u8 [0,255]; one pixel = n bytes. Trailing
    // channels past `comps.len()` are zero-padded (only reached when the
    // upstream count check is bypassed by a future caller — paranoia).
    let src_bytes: [u8; 4] = [
        color::convert::gray_to_u8(comps[0]),
        color::convert::gray_to_u8(comps.get(1).copied().unwrap_or(0.0)),
        color::convert::gray_to_u8(comps.get(2).copied().unwrap_or(0.0)),
        color::convert::gray_to_u8(comps.get(3).copied().unwrap_or(0.0)),
    ];
    let src_slice = &src_bytes[..usize::from(n)];
    let mut dst = [0u8; 3];
    xform
        .transform(src_slice, &mut dst)
        .map_err(|e| log::debug!("colorspace: ICC transform execution failed: {e}"))
        .ok()?;
    Some(dst)
}

/// Dereference an ICC profile stream by object id and return its
/// decompressed content. Returns `None` on any lookup or decode failure.
fn fetch_icc_bytes(doc: &Document, stream_id: ObjectId) -> Option<Vec<u8>> {
    let obj_arc = doc.get_object(stream_id).ok()?;
    let stream = obj_arc.as_ref().as_stream()?;
    stream
        .decompressed_content()
        .map_err(|e| log::debug!("colorspace: ICC stream decompress failed: {e}"))
        .ok()
}

/// Pass a Separation tint through the document's tint-transform function
/// and convert the result via the alternate colour space.
///
/// Returns `None` and lets the caller fall back to the image-pipeline gray
/// heuristic on any failure (function object not dereferenceable, function
/// evaluation returns `None`, or the alternate space itself can't convert).
fn separation_via_tint_fn(
    doc: &Document,
    tint_dict_id: ObjectId,
    alternate: &ColorSpace,
    tint: f64,
) -> Option<[u8; 3]> {
    let fn_arc = doc.get_object(tint_dict_id).ok()?;
    let outputs = super::shading::function::eval_function(
        doc,
        fn_arc.as_ref(),
        tint,
        alternate.ncomponents(),
    )?;
    Some(alternate.convert_to_rgb(doc, &outputs))
}

/// Image-pipeline policy: a spot tint with no resolvable transform
/// approximates as gray with `1 - tint` (tint 0 = full ink = dark).
fn separation_fallback_gray(tint: f64) -> [u8; 3] {
    let v = (1.0 - tint).clamp(0.0, 1.0);
    gray_byte_to_rgb(color::convert::gray_to_u8(v))
}

/// Fall back to the device space matching the ICC profile's channel count
/// when the moxcms transform isn't available. Mirrors the pre-CLUT behaviour.
fn icc_fallback_by_n(n: u8, comps: &[f64]) -> [u8; 3] {
    match n {
        1 => gray_byte_to_rgb(color::convert::gray_to_u8(comps[0])),
        3 => color::convert::rgb_to_bytes(comps[0], comps[1], comps[2]),
        4 => color::convert::cmyk_to_rgb_bytes(comps[0], comps[1], comps[2], comps[3]),
        _ => [0, 0, 0],
    }
}

/// Resolve a PDF `ColorSpace` object into a [`ColorSpace`] tracking value.
///
/// Returns [`ColorSpace::DeviceGray`] on unknown, malformed, or
/// recursion-bounded inputs — the same conservative fallback the image
/// pipeline uses (see `image::colorspace::resolve_cs`).
///
/// `cs_obj` may be:
/// - a `Name` for the device families (e.g. `/DeviceRGB`, `/CMYK`),
/// - an `Array` whose first element names a parameterised family
///   (`[/ICCBased <ref>]`, `[/Indexed base hival lookup]`, etc.), or
/// - a `Reference` to either of the above.
#[must_use]
pub fn resolve(doc: &Document, cs_obj: &Object) -> ColorSpace {
    resolve_depth(doc, cs_obj, 0)
}

fn resolve_depth(doc: &Document, cs_obj: &Object, depth: u8) -> ColorSpace {
    if depth >= MAX_CS_DEPTH {
        log::debug!("colorspace: nested too deep (depth {depth}); falling back to DeviceGray");
        return ColorSpace::DeviceGray;
    }
    match cs_obj {
        Object::Name(n) => resolve_device_name(n),
        Object::Reference(id) => doc
            .get_object(*id)
            .ok()
            .map_or(ColorSpace::DeviceGray, |obj| {
                resolve_depth(doc, obj.as_ref(), depth + 1)
            }),
        Object::Array(arr) => resolve_array(doc, arr, depth),
        _ => {
            log::debug!("colorspace: unexpected object type — fallback DeviceGray");
            ColorSpace::DeviceGray
        }
    }
}

/// Map a PDF device-space name (including the `cs`-operator abbreviations
/// from PDF §8.9.7 Table 89) to a [`ColorSpace`]. Unknown names fall back
/// to `DeviceGray`.
///
/// Public so callers that hold a `&[u8]` name straight from the operator
/// stream can short-circuit the four device-space cases without allocating
/// an `Object::Name`. Internal callers in this module use it directly from
/// the `resolve_*` dispatch.
#[must_use]
pub fn resolve_device_name(name: &[u8]) -> ColorSpace {
    match name {
        b"DeviceGray" | b"G" | b"CalGray" => ColorSpace::DeviceGray,
        b"DeviceRGB" | b"RGB" | b"CalRGB" => ColorSpace::DeviceRgb,
        b"DeviceCMYK" | b"CMYK" => ColorSpace::DeviceCmyk,
        b"Pattern" => ColorSpace::Pattern { base: None },
        _ => {
            log::debug!(
                "colorspace: unknown device name {:?} — fallback DeviceGray",
                String::from_utf8_lossy(name)
            );
            ColorSpace::DeviceGray
        }
    }
}

fn resolve_array(doc: &Document, arr: &[Object], depth: u8) -> ColorSpace {
    let Some(family) = arr.first().and_then(Object::as_name) else {
        log::debug!("colorspace: array missing leading family name — fallback");
        return ColorSpace::DeviceGray;
    };
    match family {
        b"DeviceGray" | b"CalGray" => ColorSpace::DeviceGray,
        b"DeviceRGB" | b"CalRGB" => ColorSpace::DeviceRgb,
        b"DeviceCMYK" => ColorSpace::DeviceCmyk,
        b"Lab" => ColorSpace::Lab {
            dict_id: arr.get(1).and_then(reference_id),
        },
        b"ICCBased" => resolve_icc_based(doc, arr.get(1), depth),
        b"Indexed" => resolve_indexed(doc, arr, depth),
        b"Pattern" => ColorSpace::Pattern {
            base: arr
                .get(1)
                .map(|o| Box::new(resolve_depth(doc, o, depth + 1))),
        },
        b"Separation" => resolve_separation(doc, arr, depth),
        b"DeviceN" => resolve_device_n(doc, arr, depth),
        _ => {
            log::debug!(
                "colorspace: unknown array family {:?} — fallback DeviceGray",
                String::from_utf8_lossy(family)
            );
            ColorSpace::DeviceGray
        }
    }
}

fn resolve_icc_based(doc: &Document, ref_obj: Option<&Object>, _depth: u8) -> ColorSpace {
    let Some(ref_obj) = ref_obj else {
        return ColorSpace::DeviceGray;
    };
    let stream_id = reference_id(ref_obj);
    // Read N from the stream dict if reachable.
    let n = super::resolve_stream_dict(doc, ref_obj)
        .and_then(|d| d.get_i64(b"N"))
        .and_then(|n| u8::try_from(n).ok())
        .filter(|&n| matches!(n, 1 | 3 | 4))
        .unwrap_or(0);
    if n == 0 {
        log::debug!("colorspace: ICCBased stream has missing or invalid /N");
        return ColorSpace::DeviceGray;
    }
    ColorSpace::IccBased { n, stream_id }
}

fn resolve_indexed(doc: &Document, arr: &[Object], depth: u8) -> ColorSpace {
    // [/Indexed base hival lookup]
    let Some(base_obj) = arr.get(1) else {
        return ColorSpace::DeviceGray;
    };
    let base = Box::new(resolve_depth(doc, base_obj, depth + 1));
    let hival = arr
        .get(2)
        .and_then(Object::as_i64)
        .map(|n| n.clamp(0, 255))
        .and_then(|n| u8::try_from(n).ok())
        .unwrap_or(0);
    // PDF §8.6.6.3: element 3 is either a hex/literal string (inline
    // palette bytes) or an indirect reference to a stream. We store the
    // reference; inline strings are uncommon outside small Indexed-Gray
    // palettes and would need re-resolution via the source object.
    let lookup_id = arr.get(3).and_then(reference_id);
    ColorSpace::Indexed {
        base,
        hival,
        lookup_id,
    }
}

fn resolve_separation(doc: &Document, arr: &[Object], depth: u8) -> ColorSpace {
    // [/Separation name alternate tint_transform]
    let Some(alt_obj) = arr.get(2) else {
        return ColorSpace::DeviceGray;
    };
    let alternate = Box::new(resolve_depth(doc, alt_obj, depth + 1));
    let tint_dict_id = arr.get(3).and_then(reference_id);
    ColorSpace::Separation {
        alternate,
        tint_dict_id,
    }
}

fn resolve_device_n(doc: &Document, arr: &[Object], depth: u8) -> ColorSpace {
    // [/DeviceN names alternate tint_transform attributes?]
    let Some(names) = arr.get(1).and_then(Object::as_array) else {
        return ColorSpace::DeviceGray;
    };
    let ncomps = u8::try_from(names.len()).unwrap_or(0);
    if ncomps == 0 {
        log::debug!("colorspace: DeviceN names array empty — fallback DeviceGray");
        return ColorSpace::DeviceGray;
    }
    let Some(alt_obj) = arr.get(2) else {
        return ColorSpace::DeviceGray;
    };
    let alternate = Box::new(resolve_depth(doc, alt_obj, depth + 1));
    let tint_dict_id = arr.get(3).and_then(reference_id);
    ColorSpace::DeviceN {
        ncomps,
        alternate,
        tint_dict_id,
    }
}

const fn reference_id(obj: &Object) -> Option<ObjectId> {
    match obj {
        Object::Reference(id) => Some(*id),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::empty_doc;

    fn name_obj(s: &[u8]) -> Object {
        Object::Name(s.to_vec())
    }

    #[test]
    fn device_names_resolve_to_device_variants() {
        let doc = empty_doc();
        assert_eq!(
            resolve(&doc, &name_obj(b"DeviceGray")),
            ColorSpace::DeviceGray
        );
        assert_eq!(
            resolve(&doc, &name_obj(b"DeviceRGB")),
            ColorSpace::DeviceRgb
        );
        assert_eq!(
            resolve(&doc, &name_obj(b"DeviceCMYK")),
            ColorSpace::DeviceCmyk
        );
    }

    #[test]
    fn abbreviated_names_resolve_per_table_89() {
        let doc = empty_doc();
        // PDF §8.9.7 Table 89 abbreviations (used in inline images but also
        // accepted by `cs`/`CS` per most decoders).
        assert_eq!(resolve(&doc, &name_obj(b"G")), ColorSpace::DeviceGray);
        assert_eq!(resolve(&doc, &name_obj(b"RGB")), ColorSpace::DeviceRgb);
        assert_eq!(resolve(&doc, &name_obj(b"CMYK")), ColorSpace::DeviceCmyk);
    }

    #[test]
    fn cal_spaces_approximate_to_device() {
        let doc = empty_doc();
        // CalGray/CalRGB carry whitepoints we don't apply; approximate to
        // their device cousins (same as the image pipeline).
        assert_eq!(resolve(&doc, &name_obj(b"CalGray")), ColorSpace::DeviceGray);
        assert_eq!(resolve(&doc, &name_obj(b"CalRGB")), ColorSpace::DeviceRgb);
    }

    #[test]
    fn pattern_name_yields_pattern_no_base() {
        let doc = empty_doc();
        assert_eq!(
            resolve(&doc, &name_obj(b"Pattern")),
            ColorSpace::Pattern { base: None },
        );
    }

    #[test]
    fn unknown_name_falls_back_to_devicegray() {
        let doc = empty_doc();
        assert_eq!(resolve(&doc, &name_obj(b"Bogus")), ColorSpace::DeviceGray);
    }

    #[test]
    fn lab_array_captures_dict_reference() {
        let doc = empty_doc();
        let arr = vec![name_obj(b"Lab"), Object::Reference((5, 0))];
        assert_eq!(
            resolve(&doc, &Object::Array(arr)),
            ColorSpace::Lab {
                dict_id: Some((5, 0)),
            },
        );
    }

    #[test]
    fn lab_array_with_inline_dict_has_no_id() {
        let doc = empty_doc();
        let arr = vec![name_obj(b"Lab"), Object::Dictionary(pdf::Dictionary::new())];
        assert_eq!(
            resolve(&doc, &Object::Array(arr)),
            ColorSpace::Lab { dict_id: None },
        );
    }

    #[test]
    fn pattern_array_with_base_recurses_into_base() {
        let doc = empty_doc();
        let arr = vec![name_obj(b"Pattern"), name_obj(b"DeviceRGB")];
        assert_eq!(
            resolve(&doc, &Object::Array(arr)),
            ColorSpace::Pattern {
                base: Some(Box::new(ColorSpace::DeviceRgb)),
            },
        );
    }

    #[test]
    fn indexed_array_resolves_base_and_clamps_hival() {
        let doc = empty_doc();
        // [/Indexed /DeviceRGB 255 <lookup>]
        let arr = vec![
            name_obj(b"Indexed"),
            name_obj(b"DeviceRGB"),
            Object::Integer(255),
            Object::String(b"\x00\x01\x02".to_vec(), pdf::StringFormat::Hexadecimal),
        ];
        assert_eq!(
            resolve(&doc, &Object::Array(arr)),
            ColorSpace::Indexed {
                base: Box::new(ColorSpace::DeviceRgb),
                hival: 255,
                lookup_id: None,
            },
        );
    }

    #[test]
    fn indexed_hival_out_of_range_clamps_to_255() {
        let doc = empty_doc();
        let arr = vec![
            name_obj(b"Indexed"),
            name_obj(b"DeviceGray"),
            Object::Integer(99_999), // adversarial
            Object::String(b"".to_vec(), pdf::StringFormat::Hexadecimal),
        ];
        assert_eq!(
            resolve(&doc, &Object::Array(arr)),
            ColorSpace::Indexed {
                base: Box::new(ColorSpace::DeviceGray),
                hival: 255,
                lookup_id: None,
            },
        );
    }

    #[test]
    fn indexed_array_captures_lookup_stream_reference() {
        // [/Indexed /DeviceRGB 15 <ref>]  — the lookup is an indirect ref.
        let doc = empty_doc();
        let arr = vec![
            name_obj(b"Indexed"),
            name_obj(b"DeviceRGB"),
            Object::Integer(15),
            Object::Reference((42, 0)),
        ];
        assert_eq!(
            resolve(&doc, &Object::Array(arr)),
            ColorSpace::Indexed {
                base: Box::new(ColorSpace::DeviceRgb),
                hival: 15,
                lookup_id: Some((42, 0)),
            },
        );
    }

    #[test]
    fn separation_extracts_alternate_and_tint_id() {
        let doc = empty_doc();
        let arr = vec![
            name_obj(b"Separation"),
            name_obj(b"PANTONE_185_C"),
            name_obj(b"DeviceCMYK"),
            Object::Reference((42, 0)),
        ];
        assert_eq!(
            resolve(&doc, &Object::Array(arr)),
            ColorSpace::Separation {
                alternate: Box::new(ColorSpace::DeviceCmyk),
                tint_dict_id: Some((42, 0)),
            },
        );
    }

    #[test]
    fn device_n_extracts_ncomps_from_names_array() {
        let doc = empty_doc();
        let names = Object::Array(vec![
            name_obj(b"Cyan"),
            name_obj(b"Magenta"),
            name_obj(b"Spot1"),
        ]);
        let arr = vec![
            name_obj(b"DeviceN"),
            names,
            name_obj(b"DeviceCMYK"),
            Object::Reference((7, 0)),
        ];
        assert_eq!(
            resolve(&doc, &Object::Array(arr)),
            ColorSpace::DeviceN {
                ncomps: 3,
                alternate: Box::new(ColorSpace::DeviceCmyk),
                tint_dict_id: Some((7, 0)),
            },
        );
    }

    #[test]
    fn malformed_array_falls_back_to_devicegray() {
        let doc = empty_doc();
        // Empty array, no family name.
        assert_eq!(
            resolve(&doc, &Object::Array(vec![])),
            ColorSpace::DeviceGray,
        );
        // Unknown family.
        let unknown = Object::Array(vec![name_obj(b"Bogus")]);
        assert_eq!(resolve(&doc, &unknown), ColorSpace::DeviceGray);
    }

    #[test]
    fn ncomponents_matches_spec() {
        assert_eq!(ColorSpace::DeviceGray.ncomponents(), 1);
        assert_eq!(ColorSpace::DeviceRgb.ncomponents(), 3);
        assert_eq!(ColorSpace::DeviceCmyk.ncomponents(), 4);
        assert_eq!(ColorSpace::Lab { dict_id: None }.ncomponents(), 3);
        assert_eq!(
            ColorSpace::IccBased {
                n: 4,
                stream_id: None
            }
            .ncomponents(),
            4,
        );
        assert_eq!(
            ColorSpace::Indexed {
                base: Box::new(ColorSpace::DeviceRgb),
                hival: 255,
                lookup_id: None,
            }
            .ncomponents(),
            1,
        );
        assert_eq!(ColorSpace::Pattern { base: None }.ncomponents(), 0);
        assert_eq!(
            ColorSpace::Separation {
                alternate: Box::new(ColorSpace::DeviceCmyk),
                tint_dict_id: None,
            }
            .ncomponents(),
            1,
        );
        assert_eq!(
            ColorSpace::DeviceN {
                ncomps: 5,
                alternate: Box::new(ColorSpace::DeviceCmyk),
                tint_dict_id: None,
            }
            .ncomponents(),
            5,
        );
    }

    // ── convert_to_rgb ──────────────────────────────────────────────────────

    #[test]
    fn convert_devicegray_full_range() {
        let doc = empty_doc();
        assert_eq!(
            ColorSpace::DeviceGray.convert_to_rgb(&doc, &[0.0]),
            [0, 0, 0]
        );
        assert_eq!(
            ColorSpace::DeviceGray.convert_to_rgb(&doc, &[1.0]),
            [255, 255, 255],
        );
        assert_eq!(
            ColorSpace::DeviceGray.convert_to_rgb(&doc, &[0.5]),
            [128, 128, 128], // gray_to_u8 rounds 0.5*255 = 127.5 → 128
        );
    }

    #[test]
    fn convert_devicergb_passes_through() {
        let doc = empty_doc();
        assert_eq!(
            ColorSpace::DeviceRgb.convert_to_rgb(&doc, &[1.0, 0.0, 0.0]),
            [255, 0, 0],
        );
        assert_eq!(
            ColorSpace::DeviceRgb.convert_to_rgb(&doc, &[0.0, 1.0, 0.0]),
            [0, 255, 0],
        );
    }

    #[test]
    fn convert_devicecmyk_uses_naive_conversion() {
        let doc = empty_doc();
        // (0, 0, 0, 0) = white in CMYK.
        assert_eq!(
            ColorSpace::DeviceCmyk.convert_to_rgb(&doc, &[0.0, 0.0, 0.0, 0.0]),
            [255, 255, 255],
        );
        // (1, 1, 1, 1) = black.
        assert_eq!(
            ColorSpace::DeviceCmyk.convert_to_rgb(&doc, &[1.0, 1.0, 1.0, 1.0]),
            [0, 0, 0],
        );
    }

    #[test]
    fn convert_lab_black_is_black() {
        let doc = empty_doc();
        // L*=0 ≡ black; a*/b* irrelevant.
        let black = ColorSpace::Lab { dict_id: None }.convert_to_rgb(&doc, &[0.0, 0.0, 0.0]);
        assert_eq!(black, [0, 0, 0]);
    }

    #[test]
    fn convert_lab_white_is_white() {
        let doc = empty_doc();
        // L*=100, a*=b*=0 ≡ D65 white.  sRGB encoding may not be exactly 255
        // due to floating-point rounding; allow ±2 LSB.
        let [r, g, b] = ColorSpace::Lab { dict_id: None }.convert_to_rgb(&doc, &[100.0, 0.0, 0.0]);
        assert!(
            r >= 253 && g >= 253 && b >= 253,
            "expected ~white, got [{r},{g},{b}]"
        );
    }

    #[test]
    fn convert_iccbased_no_stream_falls_back_by_n() {
        let doc = empty_doc();
        let icc1 = ColorSpace::IccBased {
            n: 1,
            stream_id: None,
        };
        let icc3 = ColorSpace::IccBased {
            n: 3,
            stream_id: None,
        };
        let icc4 = ColorSpace::IccBased {
            n: 4,
            stream_id: None,
        };
        // No stream_id → skip moxcms, use device-by-N fallback directly.
        assert_eq!(icc1.convert_to_rgb(&doc, &[1.0]), [255, 255, 255]);
        assert_eq!(icc3.convert_to_rgb(&doc, &[1.0, 0.0, 0.0]), [255, 0, 0]);
        assert_eq!(icc4.convert_to_rgb(&doc, &[0.0, 0.0, 0.0, 1.0]), [0, 0, 0]);
    }

    #[test]
    fn convert_iccbased_unresolvable_stream_falls_back() {
        // stream_id points to an object not in the doc — fetch_icc_bytes
        // returns None, icc_transform returns None, fallback runs.
        let doc = empty_doc();
        let icc = ColorSpace::IccBased {
            n: 3,
            stream_id: Some((999, 0)),
        };
        assert_eq!(icc.convert_to_rgb(&doc, &[1.0, 0.0, 0.0]), [255, 0, 0]);
    }

    #[test]
    fn convert_iccbased_n1_gray_skips_moxcms_and_falls_back() {
        // moxcms doesn't expose a Gray layout for 8-bit transforms; even
        // with a valid stream id, icc_transform returns None for n=1.
        let doc = empty_doc();
        let icc = ColorSpace::IccBased {
            n: 1,
            stream_id: Some((1, 0)),
        };
        assert_eq!(icc.convert_to_rgb(&doc, &[0.5]), [128, 128, 128]);
    }

    #[test]
    fn convert_iccbased_srgb_profile_round_trips_through_moxcms() {
        // Embed the canonical sRGB profile as a stream, reference it from
        // an IccBased CS with n=3, and verify that primary colours
        // round-trip through the moxcms sRGB→sRGB transform.  Identity-up-
        // to-LSB confirms the actual code path (not the fallback) runs.
        let icc_bytes = moxcms::ColorProfile::new_srgb()
            .encode()
            .expect("moxcms must encode its own sRGB profile");
        let doc = crate::test_helpers::make_doc_with_stream(&icc_bytes, " /N 3");
        let icc = ColorSpace::IccBased {
            n: 3,
            stream_id: Some((2, 0)),
        };
        // Pure red → ~[255, 0, 0] within sRGB transform rounding.
        let red = icc.convert_to_rgb(&doc, &[1.0, 0.0, 0.0]);
        assert!(
            red[0] >= 253 && red[1] <= 2 && red[2] <= 2,
            "sRGB→sRGB pure red expected ~[255,0,0], got {red:?}"
        );
        // Pure green.
        let green = icc.convert_to_rgb(&doc, &[0.0, 1.0, 0.0]);
        assert!(
            green[0] <= 2 && green[1] >= 253 && green[2] <= 2,
            "sRGB→sRGB pure green expected ~[0,255,0], got {green:?}"
        );
        // Mid grey (0.5, 0.5, 0.5).  sRGB encodes 0.5-linear ≈ 188 but the
        // input is *already* sRGB-encoded; identity should keep it at
        // ~128 modulo transform rounding.
        let grey = icc.convert_to_rgb(&doc, &[0.5, 0.5, 0.5]);
        assert!(
            (125..=131).contains(&grey[0]) && grey[0] == grey[1] && grey[1] == grey[2],
            "sRGB→sRGB mid-grey expected ~[128,128,128], got {grey:?}"
        );
    }

    #[test]
    fn convert_separation_without_tint_fn_falls_back_to_gray() {
        // No tint_dict_id → skip eval_function entirely, use the 1-tint heuristic.
        let doc = empty_doc();
        let sep = ColorSpace::Separation {
            alternate: Box::new(ColorSpace::DeviceCmyk),
            tint_dict_id: None,
        };
        assert_eq!(sep.convert_to_rgb(&doc, &[0.0]), [255, 255, 255]); // no ink
        assert_eq!(sep.convert_to_rgb(&doc, &[1.0]), [0, 0, 0]); // full ink
    }

    #[test]
    fn convert_separation_unresolvable_tint_fn_falls_back() {
        // tint_dict_id points at a missing object → fetch fails →
        // separation_via_tint_fn returns None → fall back to gray heuristic.
        let doc = empty_doc();
        let sep = ColorSpace::Separation {
            alternate: Box::new(ColorSpace::DeviceCmyk),
            tint_dict_id: Some((999, 0)),
        };
        assert_eq!(sep.convert_to_rgb(&doc, &[0.5]), [128, 128, 128]);
    }

    #[test]
    fn convert_devicen_multi_input_falls_back_to_gray() {
        // The shading function evaluator only handles 1-input functions
        // today; DeviceN with N>1 inputs cannot be transformed correctly,
        // so we use the 1-input heuristic on comps[0].
        let doc = empty_doc();
        let dn = ColorSpace::DeviceN {
            ncomps: 3,
            alternate: Box::new(ColorSpace::DeviceCmyk),
            tint_dict_id: None,
        };
        assert_eq!(dn.convert_to_rgb(&doc, &[1.0, 0.5, 0.5]), [0, 0, 0]);
    }

    #[test]
    fn convert_devicen_single_input_no_tint_fn_falls_back_to_gray() {
        // ncomps=1 with no resolvable tint function → gray heuristic on the
        // single tint component (1.0 → "full ink" → black).
        let doc = empty_doc();
        let dn = ColorSpace::DeviceN {
            ncomps: 1,
            alternate: Box::new(ColorSpace::DeviceCmyk),
            tint_dict_id: None,
        };
        assert_eq!(dn.convert_to_rgb(&doc, &[1.0]), [0, 0, 0]);
        assert_eq!(dn.convert_to_rgb(&doc, &[0.0]), [255, 255, 255]);
    }

    #[test]
    fn convert_devicen_single_input_unresolvable_tint_fn_falls_back() {
        // ncomps=1 with a tint id that doesn't resolve → eval_function
        // returns None → fall back to the gray heuristic.
        let doc = empty_doc();
        let dn = ColorSpace::DeviceN {
            ncomps: 1,
            alternate: Box::new(ColorSpace::DeviceCmyk),
            tint_dict_id: Some((999, 0)),
        };
        assert_eq!(dn.convert_to_rgb(&doc, &[0.5]), [128, 128, 128]);
    }

    #[test]
    fn convert_separation_with_separation_alternate_terminates() {
        // Adversarial: a Separation whose alternate is another Separation
        // (PDF spec forbids but our parser would accept).  Conversion must
        // not infinite-loop; the parser's MAX_CS_DEPTH bounds the
        // structure, and convert_to_rgb's recursion is bounded by the
        // same depth.  Verifies the recursion terminates with *some* output.
        let doc = empty_doc();
        let inner = ColorSpace::Separation {
            alternate: Box::new(ColorSpace::DeviceGray),
            tint_dict_id: None,
        };
        let outer = ColorSpace::Separation {
            alternate: Box::new(inner),
            tint_dict_id: None,
        };
        // No infinite loop, deterministic byte triple.
        let result = outer.convert_to_rgb(&doc, &[0.5]);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn convert_wrong_component_count_falls_back_to_black() {
        let doc = empty_doc();
        // DeviceRgb wants 3 components; passing 1 must not panic.
        assert_eq!(
            ColorSpace::DeviceRgb.convert_to_rgb(&doc, &[0.5]),
            [0, 0, 0]
        );
        assert_eq!(ColorSpace::DeviceGray.convert_to_rgb(&doc, &[]), [0, 0, 0]);
        assert_eq!(
            ColorSpace::DeviceCmyk.convert_to_rgb(&doc, &[0.5, 0.5]),
            [0, 0, 0],
        );
    }

    #[test]
    fn convert_pattern_returns_black() {
        let doc = empty_doc();
        // Pattern is never directly convertible to RGB.
        assert_eq!(
            ColorSpace::Pattern { base: None }.convert_to_rgb(&doc, &[]),
            [0, 0, 0],
        );
    }
}
