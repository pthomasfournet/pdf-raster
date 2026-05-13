//! Raw / `FlateDecode` image decoding.
//!
//! Entry point is [`decode_raw`], which dispatches on `BitsPerComponent` and
//! colour space.  Helper functions handle the common sub-cases: 8-bpp colour,
//! indexed palette expansion, and CMYK→RGB conversion.

use std::sync::Arc;

use pdf::{Dictionary, Document, Object};

use super::bitpack::{downsample_16bpp, expand_nbpp, expand_nbpp_indexed, unpack_packed_bits};
use super::colorspace::{ResolvedCs, indexed_palette, resolve_cs};
use super::{ImageColorSpace, ImageData, ImageDescriptor, ImageFilter};
use crate::resources::dict_ext::DictExt as _;

#[cfg(feature = "gpu-icc")]
use super::colorspace::extract_icc_bytes;
#[cfg(feature = "gpu-icc")]
use super::icc::IccClutCache;
#[cfg(feature = "gpu-icc")]
use gpu::GpuCtx;

// ── Public entry point ────────────────────────────────────────────────────────

/// Expand raw (already-decompressed) pixel bytes into a normalised 8-bpp form.
///
/// Supported `BitsPerComponent` values:
/// - 1  — packed MSB-first; 1 sample per component per pixel (PDF §8.9.3)
/// - 2  — 4 levels, scaled to 0x00/0x55/0xAA/0xFF
/// - 4  — 16 levels, scaled to 0x00…0xFF (value × 17)
/// - 8  — direct (common case)
/// - 16 — big-endian, high byte taken (linear 0–65535 → 0–255)
///
/// `Indexed` images with bpc 1/2/4 unpack the sub-byte palette indices first.
/// CMYK images (bpc 8/16) are converted to RGB inline.
#[cfg_attr(
    feature = "gpu-icc",
    expect(
        clippy::too_many_arguments,
        reason = "each argument is a distinct, non-groupable parameter; \
                  the extra gpu-icc args push the count over the limit"
    )
)]
pub(super) fn decode_raw(
    doc: &Document,
    data: &[u8],
    width: u32,
    height: u32,
    is_mask: bool,
    dict: &Dictionary,
    #[cfg(feature = "gpu-icc")] gpu_ctx: Option<&GpuCtx>,
    #[cfg(feature = "gpu-icc")] clut_cache: Option<&mut IccClutCache>,
) -> Option<ImageDescriptor> {
    let bpc = dict.get_i64(b"BitsPerComponent").unwrap_or(8);

    if is_mask {
        // Stencil mask — always 1 byte per pixel, no colour space conversion.
        // PDF §8.9.6.2: Decode=[1,0] inverts the stencil polarity so that
        // sample 1 = paint and sample 0 = transparent instead of the default.
        let invert = is_mask_inverted(dict);
        return decode_mask_raw(data, width, height, bpc, invert);
    }

    // Resolve the ColorSpace entry, following indirect references.  We need the
    // raw Object to detect Indexed (which uses a different decode path) as well
    // as for subsequent color-space resolution.
    let cs_obj_raw = dict.get(b"ColorSpace");
    // Follow a top-level Reference so Indexed detection below works correctly.
    // `Document::get_object` returns `Arc<Object>`, so we keep the Arc alive in
    // a local binding and borrow through it for the match below.
    let cs_obj_arc: Option<Arc<Object>> = if let Some(Object::Reference(id)) = cs_obj_raw {
        doc.get_object(*id).ok()
    } else {
        None
    };
    let cs_obj: Option<&Object> = if matches!(cs_obj_raw, Some(Object::Reference(_))) {
        cs_obj_arc.as_deref()
    } else {
        cs_obj_raw
    };

    // Check for Indexed colour space: [/Indexed base hival lookup].
    // These images carry palette indices rather than component samples, so they
    // need special handling before the general path.
    if let Some(Object::Array(arr)) = cs_obj
        && arr
            .first()
            .and_then(|o| o.as_name())
            .is_some_and(|n| n == b"Indexed")
    {
        return decode_raw_indexed(doc, data, width, height, bpc, arr);
    }

    // General path: resolve the colour space to Gray / Rgb / Cmyk.
    let resolved = cs_obj.map_or(ResolvedCs::Gray, |o| resolve_cs(doc, o));

    // For ICCBased CMYK, extract the raw ICC profile bytes for the GPU CLUT path.
    // Only done when the resolved space is actually CMYK (avoids unnecessary work
    // for Gray/RGB ICCBased images).
    #[cfg(feature = "gpu-icc")]
    let icc_bytes: Option<Vec<u8>> = if resolved == ResolvedCs::Cmyk {
        cs_obj.and_then(|o| extract_icc_bytes(doc, o))
    } else {
        None
    };

    // Parse the Decode array once; used by all bpc arms after expansion to 8-bpc.
    // PDF §8.9.5.2: Decode has 2 values per component: [Dmin Dmax ...].
    // Identity is [0 1] per component — no-op so we skip the allocation.
    let decode_map = parse_decode(dict, resolved.components());

    // Macro to apply the Decode remap (if any) to an already-8bpc buffer
    // and then call decode_raw_8bpp.  The `pixels_expr` is evaluated once.
    macro_rules! decode_and_bpp {
        ($pixels_expr:expr) => {{
            let pixels: std::borrow::Cow<[u8]> = $pixels_expr;
            let remapped = apply_decode(pixels, &decode_map);
            decode_raw_8bpp(
                &remapped,
                width,
                height,
                resolved,
                #[cfg(feature = "gpu-icc")]
                gpu_ctx,
                #[cfg(feature = "gpu-icc")]
                icc_bytes.as_deref(),
                #[cfg(feature = "gpu-icc")]
                clut_cache,
            )
        }};
    }

    match bpc {
        // bpc=1: PDF §8.9.3 packs one bit per sample, one sample per component.
        // Multi-component spaces (RGB: 3 bits/px, CMYK: 4 bits/px) must use
        // unpack_packed_bits with `components` samples per row — not expand_1bpp
        // which treats one bit as one pixel.
        1 => {
            let width_usize = usize::try_from(width).ok()?;
            let samples_per_row = width_usize.checked_mul(resolved.components())?;
            let pixels = unpack_packed_bits(data, 1, samples_per_row, height, |v| {
                if v == 0 { 0x00 } else { 0xFF }
            })?;
            decode_and_bpp!(std::borrow::Cow::Owned(pixels))
        }
        2 => decode_and_bpp!(std::borrow::Cow::Owned(expand_nbpp::<2>(
            data,
            width,
            height,
            resolved.components()
        )?)),
        4 => decode_and_bpp!(std::borrow::Cow::Owned(expand_nbpp::<4>(
            data,
            width,
            height,
            resolved.components()
        )?)),
        8 => decode_and_bpp!(std::borrow::Cow::Borrowed(data)),
        16 => {
            decode_and_bpp!(std::borrow::Cow::Owned(downsample_16bpp(
                data,
                width,
                height,
                resolved.components()
            )?))
        }
        other => {
            log::debug!("image: {other} bits-per-component not yet implemented");
            None
        }
    }
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Parse the `Decode` array into per-component `(dmin, dmax)` pairs.
///
/// Returns an empty `Vec` when the array is absent, has fewer than
/// `2 * components` entries (identity assumed by callers), or contains any
/// non-finite value (NaN/inf from a malformed PDF would silently blacken every
/// pixel via `NaN as u8 == 0`; warn and treat as identity instead).  Entries
/// that are already identity `(0.0, 1.0)` are preserved so that `apply_decode`
/// can detect and skip the no-op path without re-reading the dict.
fn parse_decode(dict: &Dictionary, components: usize) -> Vec<(f64, f64)> {
    let Some(Object::Array(arr)) = dict.get(b"Decode") else {
        return Vec::new();
    };
    let vals: Vec<f64> = arr.iter().filter_map(pdf::Object::as_f64).collect();
    if vals.len() < components * 2 {
        if !arr.is_empty() {
            log::debug!(
                "image: Decode array has {} entries, need {}×2={} — ignoring (identity assumed)",
                arr.len(),
                components,
                components * 2,
            );
        }
        return Vec::new();
    }
    // Reject non-finite values produced by a malformed PDF.  A NaN or infinite
    // dmin/dmax would make `is_identity` return false (NaN comparisons), then
    // `mul_add` would produce NaN, and the final `NaN as u8` cast would silently
    // produce 0 (black) for every pixel.  Warn and fall back to identity instead.
    let pairs: Vec<(f64, f64)> = vals
        .chunks_exact(2)
        .take(components)
        .map(|pair| (pair[0], pair[1]))
        .collect();
    if pairs
        .iter()
        .any(|&(d0, d1)| !d0.is_finite() || !d1.is_finite())
    {
        log::warn!(
            "image: Decode array contains non-finite value(s) — ignoring (identity assumed)"
        );
        return Vec::new();
    }
    pairs
}

/// Apply the per-component Decode remap to an already-8bpc pixel buffer.
///
/// Each sample `s ∈ [0, 255]` is mapped to
/// `round(dmin * 255 + s * (dmax - dmin))`, clamped to `[0, 255]`.
/// Identity pairs `(0.0, 1.0)` are short-circuited per component.
///
/// Takes a `Cow<[u8]>` so borrowed slices (e.g. the 8-bpc path when the Decode
/// array is identity) avoid a heap copy: the identity fast-path calls
/// `into_owned()` which only allocates when a borrow was passed; non-identity
/// paths call `into_owned()` first and mutate in place.
fn apply_decode(data: std::borrow::Cow<[u8]>, decode: &[(f64, f64)]) -> Vec<u8> {
    // Use an epsilon for float comparison: Decode values come from PDF real
    // objects (parsed f64) so exact 0.0/1.0 is normal, but an epsilon is safer.
    const EPS: f64 = 1e-9;
    let is_identity = |d0: f64, d1: f64| d0.abs() < EPS && (d1 - 1.0).abs() < EPS;
    if decode.is_empty() || decode.iter().all(|&(d0, d1)| is_identity(d0, d1)) {
        // Identity — return the buffer as-is.  If the caller passed a borrowed
        // slice (e.g. the 8-bpc path) this avoids a copy entirely.
        return data.into_owned();
    }
    let mut pixels = data.into_owned();
    let components = decode.len();
    for chunk in pixels.chunks_mut(components) {
        for (s, &(dmin, dmax)) in chunk.iter_mut().zip(decode.iter()) {
            if is_identity(dmin, dmax) {
                continue; // identity — leave sample unchanged
            }
            // `s` is in [0, 255]; output = dmin*255 + s*(dmax-dmin), giving
            // values in [dmin*255, dmax*255] (or swapped when dmax < dmin).
            // Clamp to [0.0, 255.0] before rounding and casting to u8.
            let remapped = dmin.mul_add(255.0, f64::from(*s) * (dmax - dmin));
            // clamp(0..=255) before cast: `as u8` after clamp is always in [0, 255].
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "remapped is clamped to [0.0, 255.0] before cast; round() output is non-negative"
            )]
            {
                *s = remapped.round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    pixels
}

/// Return `true` when the `ImageMask`'s `Decode` array is `[1, 0]` (inverted stencil).
fn is_mask_inverted(dict: &Dictionary) -> bool {
    let Some(Object::Array(arr)) = dict.get(b"Decode") else {
        return false;
    };
    let vals: Vec<f64> = arr.iter().filter_map(pdf::Object::as_f64).collect();
    matches!(vals.as_slice(), [v0, v1] if *v0 > 0.5 && *v1 < 0.5)
}

/// Decode a stencil mask (ImageMask=true) from raw data.
///
/// When `invert` is `true` (`Decode=[1,0]`), polarity is flipped: sample 1
/// maps to 0x00 (paint) and sample 0 maps to 0xFF (transparent).
fn decode_mask_raw(
    data: &[u8],
    width: u32,
    height: u32,
    bpc: i64,
    invert: bool,
) -> Option<ImageDescriptor> {
    match bpc {
        1 => {
            let width_usize = usize::try_from(width).ok()?;
            // Mask images are always 1-component: 1 bit per pixel.
            let pixels = unpack_packed_bits(data, 1, width_usize, height, |v| {
                let paint = if invert { v != 0 } else { v == 0 };
                if paint { 0x00 } else { 0xFF }
            })?;
            Some(ImageDescriptor {
                width,
                height,
                color_space: ImageColorSpace::Mask,
                data: ImageData::Cpu(pixels),
                smask: None,
                filter: ImageFilter::Raw,
            })
        }
        8 => {
            let expected = (width as usize).checked_mul(height as usize)?;
            if data.len() < expected {
                log::warn!(
                    "image: mask raw data too short ({} bytes, need {expected})",
                    data.len()
                );
                return None;
            }
            let pixels: Vec<u8> = if invert {
                data[..expected].iter().map(|&b| 255 - b).collect()
            } else {
                data[..expected].to_vec()
            };
            Some(ImageDescriptor {
                width,
                height,
                color_space: ImageColorSpace::Mask,
                data: ImageData::Cpu(pixels),
                smask: None,
                filter: ImageFilter::Raw,
            })
        }
        other => {
            log::debug!("image: mask {other} bpc not yet implemented");
            None
        }
    }
}

/// Decode an 8-bpp non-indexed image for `resolved` colour space.
///
/// CMYK is converted to RGB.  Gray and RGB are returned as-is.
fn decode_raw_8bpp(
    data: &[u8],
    width: u32,
    height: u32,
    resolved: ResolvedCs,
    #[cfg(feature = "gpu-icc")] gpu_ctx: Option<&GpuCtx>,
    #[cfg(feature = "gpu-icc")] icc_bytes: Option<&[u8]>,
    #[cfg(feature = "gpu-icc")] clut_cache: Option<&mut IccClutCache>,
) -> Option<ImageDescriptor> {
    use super::codecs::cmyk_raw_to_rgb;

    let components = resolved.components();
    let npixels = (width as usize).checked_mul(height as usize)?;
    let expected = npixels.checked_mul(components)?;

    if data.len() < expected {
        log::warn!(
            "image: raw data too short ({} bytes, need {expected} for {width}×{height}×{components})",
            data.len()
        );
        return None;
    }

    let raw = &data[..expected];

    if resolved == ResolvedCs::Cmyk {
        // Raw PDF CMYK: 255 = full ink, 0 = no ink (not inverted).
        let rgb = cmyk_raw_to_rgb(
            raw,
            false,
            #[cfg(feature = "gpu-icc")]
            gpu_ctx,
            #[cfg(feature = "gpu-icc")]
            icc_bytes,
            #[cfg(feature = "gpu-icc")]
            clut_cache,
        )?;
        Some(ImageDescriptor {
            width,
            height,
            color_space: ImageColorSpace::Rgb,
            data: ImageData::Cpu(rgb),
            smask: None,
            filter: ImageFilter::Raw,
        })
    } else {
        Some(ImageDescriptor {
            width,
            height,
            color_space: resolved.to_image_cs(),
            data: ImageData::Cpu(raw.to_vec()),
            smask: None,
            filter: ImageFilter::Raw,
        })
    }
}

/// Decode a raw `Indexed` colour space image.
///
/// For bpc 1, 2, 4: index values are packed MSB-first, N bits per index.
/// For bpc 8: one byte per index (common case).
/// PDF spec §8.6.6.3 allows bpc ∈ {1, 2, 4, 8} for Indexed images.
fn decode_raw_indexed(
    doc: &Document,
    data: &[u8],
    width: u32,
    height: u32,
    bpc: i64,
    cs_arr: &[Object],
) -> Option<ImageDescriptor> {
    let indices_8bit: Vec<u8>;
    let indices: &[u8] = match bpc {
        1 | 2 | 4 => {
            // bpc ∈ {1, 2, 4} — positive and fits u32.
            #[expect(
                clippy::cast_sign_loss,
                clippy::cast_possible_truncation,
                reason = "bpc ∈ {1, 2, 4} — positive and fits u32"
            )]
            let bits = bpc as u32;
            indices_8bit = expand_nbpp_indexed(data, width, height, bits)?;
            &indices_8bit
        }
        8 => data,
        other => {
            log::debug!("image: Indexed with {other} bpc not supported");
            return None;
        }
    };

    let (palette, out_cs) = indexed_palette(doc, cs_arr)?;
    let stride = out_cs.components();

    // `indexed_palette` guarantees stride ∈ {1, 3} and palette is a multiple of
    // stride with at least one entry, but we guard defensively.
    if stride == 0 || palette.is_empty() || palette.len() % stride != 0 {
        log::debug!(
            "image: Indexed palette invariant violated (len={}, stride={stride})",
            palette.len()
        );
        return None;
    }

    let npixels = (width as usize).checked_mul(height as usize)?;
    if indices.len() < npixels {
        log::warn!(
            "image: Indexed raw data too short ({} bytes, need {npixels})",
            indices.len()
        );
        return None;
    }

    let n_entries = palette.len() / stride; // ≥ 1 guaranteed by guards above
    let mut out = Vec::with_capacity(npixels.checked_mul(stride)?);
    for &idx in &indices[..npixels] {
        let i = usize::from(idx).min(n_entries - 1);
        out.extend_from_slice(&palette[i * stride..(i + 1) * stride]);
    }

    Some(ImageDescriptor {
        width,
        height,
        color_space: out_cs.to_image_cs(),
        data: ImageData::Cpu(out),
        smask: None,
        filter: ImageFilter::Raw,
    })
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    use crate::test_helpers::empty_doc as make_doc;

    #[test]
    fn decode_raw_gray_8bpp() {
        let doc = make_doc();
        let mut dict = Dictionary::new();
        dict.set("ColorSpace", Object::Name(b"DeviceGray".to_vec()));
        dict.set("BitsPerComponent", Object::Integer(8));
        let data = vec![0u8, 128, 255];
        let desc = decode_raw(
            &doc,
            &data,
            3,
            1,
            false,
            &dict,
            #[cfg(feature = "gpu-icc")]
            None,
            #[cfg(feature = "gpu-icc")]
            None,
        )
        .unwrap();
        assert_eq!(desc.color_space, ImageColorSpace::Gray);
        assert_eq!(desc.data.as_cpu().unwrap(), &[0, 128, 255]);
    }

    #[test]
    fn decode_raw_rgb_8bpp() {
        let doc = make_doc();
        let mut dict = Dictionary::new();
        dict.set("ColorSpace", Object::Name(b"DeviceRGB".to_vec()));
        dict.set("BitsPerComponent", Object::Integer(8));
        let data = vec![255u8, 0, 0, 0, 255, 0, 0, 0, 255];
        let desc = decode_raw(
            &doc,
            &data,
            3,
            1,
            false,
            &dict,
            #[cfg(feature = "gpu-icc")]
            None,
            #[cfg(feature = "gpu-icc")]
            None,
        )
        .unwrap();
        assert_eq!(desc.color_space, ImageColorSpace::Rgb);
        assert_eq!(desc.width, 3);
        assert_eq!(desc.data.len(), 9);
    }

    #[test]
    fn decode_raw_mask_ignores_cs() {
        let doc = make_doc();
        let mut dict = Dictionary::new();
        dict.set("BitsPerComponent", Object::Integer(8));
        let data = vec![0u8, 255];
        let desc = decode_raw(
            &doc,
            &data,
            2,
            1,
            true,
            &dict,
            #[cfg(feature = "gpu-icc")]
            None,
            #[cfg(feature = "gpu-icc")]
            None,
        )
        .unwrap();
        assert_eq!(desc.color_space, ImageColorSpace::Mask);
    }

    #[test]
    fn decode_raw_mask_decode_inverted() {
        // Decode=[1,0] flips stencil polarity: sample 0 → 0xFF (transparent),
        // sample 255 → 0x00 (paint).  Without the fix both would stay as-is.
        let doc = make_doc();
        let mut dict = Dictionary::new();
        dict.set("BitsPerComponent", Object::Integer(8));
        dict.set(
            "Decode",
            Object::Array(vec![Object::Real(1.0), Object::Real(0.0)]),
        );
        // pixel 0 → was "paint" in default polarity; with Decode=[1,0] it should
        // become 0xFF (transparent).  Pixel 255 should become 0x00 (paint).
        let data = vec![0u8, 255u8];
        let desc = decode_raw(
            &doc,
            &data,
            2,
            1,
            true,
            &dict,
            #[cfg(feature = "gpu-icc")]
            None,
            #[cfg(feature = "gpu-icc")]
            None,
        )
        .unwrap();
        let pixels = desc.data.as_cpu().unwrap();
        assert_eq!(
            pixels[0], 0xFF,
            "sample 0 with Decode=[1,0] must be transparent"
        );
        assert_eq!(
            pixels[1], 0x00,
            "sample 255 with Decode=[1,0] must be paint"
        );
    }

    #[test]
    fn decode_raw_cmyk_converts_to_rgb() {
        let doc = make_doc();
        let mut dict = Dictionary::new();
        dict.set("ColorSpace", Object::Name(b"DeviceCMYK".to_vec()));
        dict.set("BitsPerComponent", Object::Integer(8));
        // Single pixel: C=0, M=0, Y=0, K=0 → white (255, 255, 255).
        let data = vec![0u8, 0, 0, 0];
        let desc = decode_raw(
            &doc,
            &data,
            1,
            1,
            false,
            &dict,
            #[cfg(feature = "gpu-icc")]
            None,
            #[cfg(feature = "gpu-icc")]
            None,
        )
        .unwrap();
        assert_eq!(desc.color_space, ImageColorSpace::Rgb);
        assert_eq!(desc.data.as_cpu().unwrap(), &[255, 255, 255]);
    }

    #[test]
    fn decode_raw_too_short_returns_none() {
        let doc = make_doc();
        let mut dict = Dictionary::new();
        dict.set("ColorSpace", Object::Name(b"DeviceRGB".to_vec()));
        dict.set("BitsPerComponent", Object::Integer(8));
        // Claim 2×2 RGB but supply only 3 bytes (need 12).
        let data = vec![0u8, 0, 0];
        assert!(
            decode_raw(
                &doc,
                &data,
                2,
                2,
                false,
                &dict,
                #[cfg(feature = "gpu-icc")]
                None,
                #[cfg(feature = "gpu-icc")]
                None
            )
            .is_none()
        );
    }

    #[test]
    fn decode_raw_gray_1bpp() {
        // 2-pixel Gray image at bpc=1: byte 0b10_000000 → pixels [0xFF, 0x00].
        let doc = make_doc();
        let mut dict = Dictionary::new();
        dict.set("ColorSpace", Object::Name(b"DeviceGray".to_vec()));
        dict.set("BitsPerComponent", Object::Integer(1));
        let data = vec![0b1000_0000u8];
        let desc = decode_raw(
            &doc,
            &data,
            2,
            1,
            false,
            &dict,
            #[cfg(feature = "gpu-icc")]
            None,
            #[cfg(feature = "gpu-icc")]
            None,
        )
        .unwrap();
        assert_eq!(desc.color_space, ImageColorSpace::Gray);
        assert_eq!(desc.data.as_cpu().unwrap(), &[0xFF, 0x00]);
    }

    #[test]
    fn decode_raw_rgb_1bpp() {
        // 1-pixel RGB image at bpc=1: 3 bits packed MSB-first in one byte.
        // Byte 0b110_00000: R=1→0xFF, G=1→0xFF, B=0→0x00.
        let doc = make_doc();
        let mut dict = Dictionary::new();
        dict.set("ColorSpace", Object::Name(b"DeviceRGB".to_vec()));
        dict.set("BitsPerComponent", Object::Integer(1));
        let data = vec![0b1100_0000u8];
        let desc = decode_raw(
            &doc,
            &data,
            1,
            1,
            false,
            &dict,
            #[cfg(feature = "gpu-icc")]
            None,
            #[cfg(feature = "gpu-icc")]
            None,
        )
        .unwrap();
        assert_eq!(desc.color_space, ImageColorSpace::Rgb);
        assert_eq!(desc.data.as_cpu().unwrap(), &[0xFF, 0xFF, 0x00]);
    }

    #[test]
    fn decode_raw_gray_decode_inverted() {
        // A 2-pixel DeviceGray image with Decode=[1,0].
        // PDF §8.9.5.2: output = dmin*255 + s*(dmax-dmin) = 1*255 + s*(0-1) = 255 - s.
        // Sample 0 → output 255 (white); sample 255 → output 0 (black).
        let doc = make_doc();
        let mut dict = Dictionary::new();
        dict.set("ColorSpace", Object::Name(b"DeviceGray".to_vec()));
        dict.set("BitsPerComponent", Object::Integer(8));
        dict.set(
            "Decode",
            Object::Array(vec![Object::Real(1.0), Object::Real(0.0)]),
        );
        let data = vec![0u8, 255u8]; // sample 0 → output 255, sample 255 → output 0
        let desc = decode_raw(
            &doc,
            &data,
            2,
            1,
            false,
            &dict,
            #[cfg(feature = "gpu-icc")]
            None,
            #[cfg(feature = "gpu-icc")]
            None,
        )
        .unwrap();
        let pixels = desc.data.as_cpu().unwrap();
        assert_eq!(pixels[0], 255, "sample 0 with Decode=[1,0] must map to 255");
        assert_eq!(pixels[1], 0, "sample 255 with Decode=[1,0] must map to 0");
    }

    #[test]
    fn decode_raw_gray_decode_identity_unchanged() {
        // Explicit Decode=[0,1] is identity — output must equal input exactly.
        let doc = make_doc();
        let mut dict = Dictionary::new();
        dict.set("ColorSpace", Object::Name(b"DeviceGray".to_vec()));
        dict.set("BitsPerComponent", Object::Integer(8));
        dict.set(
            "Decode",
            Object::Array(vec![Object::Real(0.0), Object::Real(1.0)]),
        );
        let data = vec![128u8];
        let desc = decode_raw(
            &doc,
            &data,
            1,
            1,
            false,
            &dict,
            #[cfg(feature = "gpu-icc")]
            None,
            #[cfg(feature = "gpu-icc")]
            None,
        )
        .unwrap();
        assert_eq!(desc.data.as_cpu().unwrap()[0], 128);
    }

    #[test]
    fn decode_raw_gray_decode_integer_entries_accepted() {
        // Decode=[1, 0] written as Integer objects (not Real) must also invert.
        // Verifies that `parse_decode` / `Object::as_f64` handles Integer variants.
        let doc = make_doc();
        let mut dict = Dictionary::new();
        dict.set("ColorSpace", Object::Name(b"DeviceGray".to_vec()));
        dict.set("BitsPerComponent", Object::Integer(8));
        dict.set(
            "Decode",
            Object::Array(vec![Object::Integer(1), Object::Integer(0)]),
        );
        let data = vec![0u8, 255u8];
        let desc = decode_raw(
            &doc,
            &data,
            2,
            1,
            false,
            &dict,
            #[cfg(feature = "gpu-icc")]
            None,
            #[cfg(feature = "gpu-icc")]
            None,
        )
        .unwrap();
        let pixels = desc.data.as_cpu().unwrap();
        assert_eq!(
            pixels[0], 255,
            "integer Decode=[1,0]: sample 0 must map to 255"
        );
        assert_eq!(
            pixels[1], 0,
            "integer Decode=[1,0]: sample 255 must map to 0"
        );
    }

    #[test]
    fn decode_raw_gray_decode_nonfinite_falls_back_to_identity() {
        // A malformed Decode array with a non-finite value must fall back to
        // identity rather than silently blackening every pixel.
        // The test uses NaN; the code path is the same for +/-inf.
        let doc = make_doc();
        let mut dict = Dictionary::new();
        dict.set("ColorSpace", Object::Name(b"DeviceGray".to_vec()));
        dict.set("BitsPerComponent", Object::Integer(8));
        // f32::NAN stored in Object::Real; as_f64 converts through f32 so it stays NaN.
        dict.set(
            "Decode",
            Object::Array(vec![Object::Real(f32::NAN), Object::Real(1.0)]),
        );
        let data = vec![128u8];
        let desc = decode_raw(
            &doc,
            &data,
            1,
            1,
            false,
            &dict,
            #[cfg(feature = "gpu-icc")]
            None,
            #[cfg(feature = "gpu-icc")]
            None,
        )
        .unwrap();
        // Identity fallback: output must equal input.
        assert_eq!(
            desc.data.as_cpu().unwrap()[0],
            128,
            "non-finite Decode must fall back to identity"
        );
    }
}
