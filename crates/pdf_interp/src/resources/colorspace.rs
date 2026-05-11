//! PDF colour-space taxonomy (§8.6) for tracking the current `cs`/`CS`
//! selection on the graphics state.
//!
//! This is the *input* representation — the colour space declared by the
//! content stream — distinct from the [`super::image::ImageColorSpace`]
//! output category used by the image blit path. The taxonomy is needed by
//! the operators that resolve uncoloured-pattern tints (§8.7.3.3) and any
//! future ICC-correct CMYK→RGB conversion on stroke/fill.
//!
//! # Scope
//!
//! The variants capture what we need to track *which* space is active;
//! they don't carry the full colorant-mapping data (Lab whitepoint
//! values, the ICC profile bytes, the full Separation tint function, etc.)
//! — those are pulled lazily via the stored [`ObjectId`] when the
//! conversion path actually runs.
//!
//! # PDF §8.6 mapping
//!
//! | PDF variant   | [`ColorSpace`] variant |
//! |---|---|
//! | `/DeviceGray`, `/G` | [`ColorSpace::DeviceGray`] |
//! | `/DeviceRGB`, `/RGB` | [`ColorSpace::DeviceRgb`] |
//! | `/DeviceCMYK`, `/CMYK` | [`ColorSpace::DeviceCmyk`] |
//! | `[/CalGray ...]`     | [`ColorSpace::DeviceGray`] (approximated) |
//! | `[/CalRGB ...]`      | [`ColorSpace::DeviceRgb`] (approximated) |
//! | `[/Lab ...]`         | [`ColorSpace::Lab`] |
//! | `[/ICCBased <ref>]`  | [`ColorSpace::IccBased`] |
//! | `[/Indexed ...]`     | [`ColorSpace::Indexed`] |
//! | `[/Pattern]` or `[/Pattern <base>]` | [`ColorSpace::Pattern`] |
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
    ColorSpace::Indexed { base, hival }
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
}
