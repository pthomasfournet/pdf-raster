//! PDF colour space resolution for image decoding.
//!
//! Maps the `ColorSpace` entry of an image stream dictionary to one of the
//! three device spaces the blit pipeline understands (`Gray`, `Rgb`, `Cmyk`),
//! and extracts palette data for `Indexed` spaces.
//!
//! The `Cmyk` variant is always converted to `Rgb` before the descriptor
//! leaves this module; callers never see raw CMYK pixels.

use pdf::{Dictionary, Document, Object};

use super::ImageColorSpace;
use crate::resources::dict_ext::DictExt as _;

// ── Public types ──────────────────────────────────────────────────────────────

/// Internal resolved colour space — what the decode path will actually produce.
///
/// `Cmyk` is a transient state: any function that returns an `ImageDescriptor`
/// converts CMYK bytes to RGB before the descriptor leaves the image module.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ResolvedCs {
    Gray,
    Rgb,
    /// Raw CMYK (4 bytes/pixel, 255 = full ink).  Converted to RGB before returning.
    Cmyk,
}

impl ResolvedCs {
    pub(super) const fn components(self) -> usize {
        match self {
            Self::Gray => 1,
            Self::Rgb => 3,
            Self::Cmyk => 4,
        }
    }

    pub(super) const fn to_image_cs(self) -> ImageColorSpace {
        match self {
            Self::Gray => ImageColorSpace::Gray,
            Self::Rgb | Self::Cmyk => ImageColorSpace::Rgb,
        }
    }
}

// ── Colour space resolution ───────────────────────────────────────────────────

/// Maximum recursion depth for `Indexed` base-space chains.
///
/// PDF spec §8.6.6.3 forbids nested `Indexed` spaces, but adversarial PDFs are
/// not required to be spec-compliant.  We stop after this many levels to prevent
/// stack overflow.
const MAX_CS_DEPTH: u8 = 8;

/// Resolve the `ColorSpace` entry of an image stream dictionary to one of the
/// three device spaces the blit path understands.
///
/// Handles:
/// - `DeviceGray` / `CalGray`        → Gray
/// - `DeviceRGB`  / `CalRGB`         → Rgb
/// - `DeviceCMYK`                    → Cmyk (caller converts to RGB)
/// - `ICCBased`   → inspect `N` in the ICC stream dict (1→Gray, 3→Rgb, 4→Cmyk)
/// - `Indexed`    → resolve the base space; Indexed expansion handled separately
/// - `Separation` / `DeviceN`        → approximate as Gray (tint 0 = full ink = dark)
/// - unknown / absent                → Gray (safe fallback)
pub(super) fn resolve_cs(doc: &Document, cs_obj: &Object) -> ResolvedCs {
    resolve_cs_depth(doc, cs_obj, 0)
}

fn resolve_cs_depth(doc: &Document, cs_obj: &Object, depth: u8) -> ResolvedCs {
    match cs_obj {
        Object::Name(n) => device_cs_name(n),
        Object::Reference(id) => {
            // Indirect colour space reference — dereference and resolve.
            doc.get_object(*id).ok().map_or(ResolvedCs::Gray, |obj| {
                resolve_cs_depth(doc, obj.as_ref(), depth)
            })
        }
        Object::Array(arr) => {
            let name = arr.first().and_then(Object::as_name).unwrap_or(b"");
            match name {
                b"DeviceRGB" | b"CalRGB" => ResolvedCs::Rgb,
                b"DeviceCMYK" => ResolvedCs::Cmyk,
                b"ICCBased" => {
                    // Second element is a reference to the ICC stream.
                    let stream_dict = arr
                        .get(1)
                        .and_then(|o| super::super::resolve_stream_dict(doc, o));
                    icc_based_cs(stream_dict.as_ref())
                }
                b"Indexed" => {
                    // [/Indexed base hival lookup] — base is element 1.
                    // Guard against adversarial nested Indexed chains.
                    if depth >= MAX_CS_DEPTH {
                        log::debug!(
                            "image: Indexed colour space nested too deeply (depth {depth}) — treating as Gray"
                        );
                        return ResolvedCs::Gray;
                    }
                    arr.get(1)
                        .map_or(ResolvedCs::Gray, |o| resolve_cs_depth(doc, o, depth + 1))
                }
                // DeviceGray, CalGray, Separation, DeviceN, and unknown → Gray.
                _ => {
                    if !matches!(
                        name,
                        b"DeviceGray" | b"CalGray" | b"Separation" | b"DeviceN"
                    ) {
                        log::debug!(
                            "image: unknown array colour space {:?} — treating as Gray",
                            String::from_utf8_lossy(name)
                        );
                    }
                    ResolvedCs::Gray
                }
            }
        }
        _ => ResolvedCs::Gray,
    }
}

/// Map a `ColorSpace` device name to a [`ResolvedCs`].
pub(super) fn device_cs_name(name: &[u8]) -> ResolvedCs {
    match name {
        b"DeviceRGB" | b"CalRGB" => ResolvedCs::Rgb,
        b"DeviceCMYK" => ResolvedCs::Cmyk,
        _ => ResolvedCs::Gray,
    }
}

/// Inspect the `N` key in an `ICCBased` stream dict to determine the colour space.
fn icc_based_cs(stream_dict: Option<&Dictionary>) -> ResolvedCs {
    match stream_dict.and_then(|d| d.get_i64(b"N")) {
        Some(1) => ResolvedCs::Gray,
        Some(3) => ResolvedCs::Rgb,
        Some(4) => ResolvedCs::Cmyk,
        Some(n) => {
            log::debug!("image: ICCBased with N={n}, treating as Gray");
            ResolvedCs::Gray
        }
        None => {
            log::debug!("image: ICCBased missing N, treating as Gray");
            ResolvedCs::Gray
        }
    }
}

/// Extract the raw ICC profile bytes from an `ICCBased` colour space object.
///
/// Returns `None` if the object is not `[/ICCBased <ref>]`, the stream cannot
/// be dereferenced, or decompression fails.  Only compiled under `gpu-icc`.
#[cfg(feature = "gpu-icc")]
pub(super) fn extract_icc_bytes(doc: &Document, cs_obj: &Object) -> Option<Vec<u8>> {
    let arr = cs_obj.as_array()?;
    if arr.first().and_then(Object::as_name) != Some(b"ICCBased") {
        return None;
    }
    let Object::Reference(id) = arr.get(1)? else {
        return None;
    };
    let obj_arc = doc.get_object(*id).ok()?;
    let stream = obj_arc.as_ref().as_stream()?;
    stream
        .decompressed_content()
        .map_err(|e| log::debug!("image: ICCBased stream decompression failed: {e}"))
        .ok()
}

// ── Indexed palette ───────────────────────────────────────────────────────────

/// Decode an `Indexed` colour space lookup table into a flat RGB/Gray palette.
///
/// `cs_arr` is the full `[/Indexed base hival lookup]` array.  Returns
/// `(palette, base_cs)` where `palette[i * stride .. i * stride + stride]`
/// is the colour for index `i` and `stride` is `base_cs.components()`.
///
/// `Cmyk` base spaces are converted to `Rgb` before return so callers always
/// receive either 1-byte (Gray) or 3-byte (Rgb) palette entries.
///
/// Returns `None` if the array is malformed or the lookup stream cannot be read.
pub(super) fn indexed_palette(doc: &Document, cs_arr: &[Object]) -> Option<(Vec<u8>, ResolvedCs)> {
    // [/Indexed base hival lookup]
    if cs_arr.len() < 4 {
        return None;
    }
    let base = resolve_cs(doc, &cs_arr[1]);

    // hival is clamped to [0, 255] before converting — cast is lossless.
    #[expect(
        clippy::cast_sign_loss,
        reason = "clamped to [0, 255] — never negative"
    )]
    let hival = cs_arr[2].as_i64()?.clamp(0, 255) as usize;
    let n_entries = hival + 1;

    // Lookup is either a string (literal bytes) or a reference to a stream.
    let lookup_bytes: Vec<u8> = match &cs_arr[3] {
        Object::String(bytes, _) => bytes.clone(),
        Object::Reference(id) => {
            let obj_arc = doc.get_object(*id).ok()?;
            match obj_arc.as_ref() {
                Object::Stream(s) => s.decompressed_content().ok()?,
                Object::String(bytes, _) => bytes.clone(),
                _ => return None,
            }
        }
        _ => return None,
    };

    let base_stride = base.components();
    let needed = n_entries.checked_mul(base_stride)?;
    if lookup_bytes.len() < needed {
        log::debug!(
            "image: Indexed palette too short ({} bytes, need {needed})",
            lookup_bytes.len()
        );
        return None;
    }

    // Build the output palette, converting CMYK entries to RGB inline.
    let palette: Vec<u8> = if base == ResolvedCs::Cmyk {
        let mut out = Vec::with_capacity(n_entries * 3);
        for chunk in lookup_bytes[..needed].chunks_exact(4) {
            let (r, g, b) =
                color::convert::cmyk_to_rgb_reflectance(chunk[0], chunk[1], chunk[2], chunk[3]);
            out.push(r);
            out.push(g);
            out.push(b);
        }
        out
    } else {
        lookup_bytes[..needed].to_vec()
    };

    // CMYK base is treated as Rgb after palette construction above.
    let out_cs = if base == ResolvedCs::Gray {
        ResolvedCs::Gray
    } else {
        ResolvedCs::Rgb
    };
    Some((palette, out_cs))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_cs_name_gray() {
        assert_eq!(device_cs_name(b"DeviceGray"), ResolvedCs::Gray);
        assert_eq!(device_cs_name(b"CalGray"), ResolvedCs::Gray);
    }

    #[test]
    fn device_cs_name_rgb() {
        assert_eq!(device_cs_name(b"DeviceRGB"), ResolvedCs::Rgb);
        assert_eq!(device_cs_name(b"CalRGB"), ResolvedCs::Rgb);
    }

    #[test]
    fn device_cs_name_cmyk() {
        assert_eq!(device_cs_name(b"DeviceCMYK"), ResolvedCs::Cmyk);
    }

    #[test]
    fn device_cs_name_unknown_is_gray() {
        assert_eq!(device_cs_name(b"Unknown"), ResolvedCs::Gray);
    }

    #[test]
    fn resolve_cs_indirect_reference_falls_back_to_gray() {
        // An unresolvable Reference should not panic — it falls back to Gray.
        // Reference (999, 0) is absent from the empty doc's xref so resolution
        // fails and we get the fallback.
        let doc = crate::test_helpers::empty_doc();
        let obj = Object::Reference((999, 0));
        assert_eq!(resolve_cs(&doc, &obj), ResolvedCs::Gray);
    }
}
