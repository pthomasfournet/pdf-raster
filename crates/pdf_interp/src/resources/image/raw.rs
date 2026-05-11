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
#[expect(
    clippy::too_many_lines,
    reason = "single-pass raw image decoder; cfg-gated arms preclude clean splits"
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
        return decode_mask_raw(data, width, height, bpc);
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
            decode_raw_8bpp(
                &pixels,
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
        }
        2 => decode_raw_8bpp(
            &expand_nbpp::<2>(data, width, height, resolved.components())?,
            width,
            height,
            resolved,
            #[cfg(feature = "gpu-icc")]
            gpu_ctx,
            #[cfg(feature = "gpu-icc")]
            icc_bytes.as_deref(),
            #[cfg(feature = "gpu-icc")]
            clut_cache,
        ),
        4 => decode_raw_8bpp(
            &expand_nbpp::<4>(data, width, height, resolved.components())?,
            width,
            height,
            resolved,
            #[cfg(feature = "gpu-icc")]
            gpu_ctx,
            #[cfg(feature = "gpu-icc")]
            icc_bytes.as_deref(),
            #[cfg(feature = "gpu-icc")]
            clut_cache,
        ),
        8 => decode_raw_8bpp(
            data,
            width,
            height,
            resolved,
            #[cfg(feature = "gpu-icc")]
            gpu_ctx,
            #[cfg(feature = "gpu-icc")]
            icc_bytes.as_deref(),
            #[cfg(feature = "gpu-icc")]
            clut_cache,
        ),
        16 => decode_raw_8bpp(
            &downsample_16bpp(data, width, height, resolved.components())?,
            width,
            height,
            resolved,
            #[cfg(feature = "gpu-icc")]
            gpu_ctx,
            #[cfg(feature = "gpu-icc")]
            icc_bytes.as_deref(),
            #[cfg(feature = "gpu-icc")]
            clut_cache,
        ),
        other => {
            log::debug!("image: {other} bits-per-component not yet implemented");
            None
        }
    }
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Decode a stencil mask (ImageMask=true) from raw data.
fn decode_mask_raw(data: &[u8], width: u32, height: u32, bpc: i64) -> Option<ImageDescriptor> {
    match bpc {
        1 => {
            let width_usize = usize::try_from(width).ok()?;
            // Mask images are always 1-component: 1 bit per pixel.
            let pixels = unpack_packed_bits(data, 1, width_usize, height, |v| {
                if v == 0 { 0x00 } else { 0xFF }
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
            Some(ImageDescriptor {
                width,
                height,
                color_space: ImageColorSpace::Mask,
                data: ImageData::Cpu(data[..expected].to_vec()),
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
}
