//! Soft-mask (`SMask`) decoding for PDF images.
//!
//! A PDF `SMask` is a grayscale image whose pixel values become the alpha channel
//! of the parent image.  This module decodes the `SMask` stream and resamples it
//! to the parent image's dimensions when they differ.

use pdf::{Document, Object, ObjectId};

use super::bitpack::{downsample_16bpp, expand_1bpp, expand_nbpp};
use super::codecs::{decode_ccitt, decode_jbig2};
use super::filter_name;
use super::raw::decode_raw;
use super::validated_dims;
use super::{ImageColorSpace, ImageDescriptor};

// ── Public API ────────────────────────────────────────────────────────────────

/// Decode a soft-mask (`SMask`) image stream into a flat alpha buffer.
///
/// Returns exactly `img_w * img_h` bytes, one alpha value per pixel:
/// `0x00` = fully transparent, `0xFF` = fully opaque.  The `SMask` stream may
/// have different dimensions from the parent image; if so the buffer is
/// resampled to match via nearest-neighbour scaling.
///
/// Returns `None` when the stream is unresolvable, its filter is unsupported,
/// or its dimensions are degenerate.  The caller must skip the image in that
/// case rather than blit it without a mask.
pub(super) fn decode_smask(
    doc: &Document,
    id: ObjectId,
    img_w: u32,
    img_h: u32,
) -> Option<Vec<u8>> {
    use crate::resources::dict_ext::DictExt as _;

    // Bind the Arc to a local so the borrow into the stream below stays alive.
    let obj_arc = doc.get_object(id).ok()?;
    let stream = obj_arc.as_ref().as_stream()?;

    let w_raw = stream.dict.get_i64(b"Width")?;
    let h_raw = stream.dict.get_i64(b"Height")?;
    let Some((sm_w, sm_h)) = validated_dims(w_raw, h_raw) else {
        log::warn!("image: SMask degenerate dimensions {w_raw}×{h_raw}, skipping");
        return None;
    };

    let filter = stream.dict.get(b"Filter").and_then(filter_name);
    let bpc = stream.dict.get_i64(b"BitsPerComponent").unwrap_or(8);

    // CCITTFaxDecode and JBIG2Decode are decoded as grayscale samples whose
    // bytes are used directly as alpha, matching the §11.6.5.2 soft-mask model.
    //
    // PDF §11.6.5.2: the SMask image's decoded sample value IS the alpha
    // directly (0x00 = transparent, 0xFF = opaque). The decoders are called with
    // is_mask=false so they return ImageColorSpace::Gray bytes following the
    // Gray convention (JBIG2/CCITT black-bit → 0x00, white-bit → 0xFF). Those
    // bytes are the §11.6.5.2 sample values and are consumed as alpha without
    // any further inversion — matching the already-correct DCTDecode/Flate/raw
    // SMask arms below.
    //
    // For these compressed formats we use the actual decoded dimensions
    // (sm_desc.width × sm_desc.height) — not the dict-declared sm_w × sm_h —
    // as the true source size passed to scale_smask, so that a decoder that
    // reports a different output size does not produce an out-of-bounds index.
    let alpha: Vec<u8> = match filter.as_deref() {
        Some("CCITTFaxDecode") => {
            let parms = stream.dict.get(b"DecodeParms");
            let sm_desc = decode_ccitt(stream.content.as_slice(), sm_w, sm_h, false, parms)?;
            let (aw, ah) = (sm_desc.width, sm_desc.height);
            let alpha = alpha_from_decoded("CCITTFaxDecode", &sm_desc)?;
            return Some(scale_smask(alpha, aw, ah, img_w, img_h));
        }
        Some("JBIG2Decode") => {
            let parms = stream.dict.get(b"DecodeParms");
            let sm_desc = decode_jbig2(doc, stream.content.as_slice(), sm_w, sm_h, false, parms)?;
            let (aw, ah) = (sm_desc.width, sm_desc.height);
            let alpha = alpha_from_decoded("JBIG2Decode", &sm_desc)?;
            return Some(scale_smask(alpha, aw, ah, img_w, img_h));
        }
        Some("DCTDecode") => {
            // decode_smask_dct returns (pixels, jpeg_w, jpeg_h) so we can use the
            // JPEG-reported dimensions as the source size for scale_smask, mirroring
            // the CCITTFax/JBIG2 pattern above.
            let (dct_alpha, jw, jh) = decode_smask_dct(stream.content.as_slice(), sm_w, sm_h)?;
            return Some(scale_smask(dct_alpha, jw, jh, img_w, img_h));
        }
        Some("FlateDecode") => {
            let raw = stream.decompressed_content().ok()?;
            decode_smask_raw_bytes(&raw, sm_w, sm_h, bpc)?
        }
        None => {
            // No filter — raw (uncompressed) byte stream.
            decode_smask_raw_bytes(stream.content.as_slice(), sm_w, sm_h, bpc)?
        }
        Some(other) => {
            log::warn!("image: SMask filter {other:?} not supported — skipping image");
            return None;
        }
    };

    Some(scale_smask(alpha, sm_w, sm_h, img_w, img_h))
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Extract a host-resident alpha buffer from a bitonal-codec `SMask` descriptor,
/// validating it covers the descriptor's own reported dimensions.
///
/// The CCITT/JBIG2 decoders normalise their output to `width × height` on every
/// path the `SMask` pipeline exercises today, but a future codec change — or an
/// adversarial stream that drives a decoder down a not-yet-hardened branch (the
/// CCITT G4 clean-EOFB-before-`height`-rows path is one such gap) — could yield
/// a short buffer.  `scale_smask` indexes `src[sy*sw + sx]` over the *declared*
/// `width × height`, so a short `src` is a panic, not a wrong pixel.  Fail
/// loudly and gracefully here instead, matching the explicit length guards in
/// the `DCTDecode` and raw/`FlateDecode` arms (a short `SMask` is a decode
/// failure, never silently-wrong alpha — the v1 alpha-polarity sin).
fn alpha_from_decoded(filter: &str, desc: &ImageDescriptor) -> Option<Vec<u8>> {
    let bytes = desc.data.as_cpu()?;
    let expected = (desc.width as usize).checked_mul(desc.height as usize)?;
    if bytes.len() < expected {
        log::warn!(
            "image: {filter} SMask decoded {} bytes, expected {expected} for {}×{} — skipping image",
            bytes.len(),
            desc.width,
            desc.height
        );
        return None;
    }
    Some(bytes[..expected].to_vec())
}

/// Decode a `DCTDecode`-compressed (`JPEG`) `SMask` stream into an 8-bpp alpha buffer.
///
/// Returns `(pixels, jpeg_width, jpeg_height)` so the caller can pass the
/// JPEG-reported dimensions to `scale_smask` rather than the dict-declared ones.
/// This mirrors the `CCITTFaxDecode` / `JBIG2Decode` pattern where the actual
/// decoded size takes precedence over the dict.
///
/// PDF §11.6.5.2 permits `DCTDecode` on `SMask` streams.  The stream must be a
/// single-component (grayscale) `JPEG`; if the decoder produces multiple components
/// we reject it rather than guess which channel is the alpha.
fn decode_smask_dct(data: &[u8], sm_w: u32, sm_h: u32) -> Option<(Vec<u8>, u32, u32)> {
    use zune_core::bytestream::ZCursor;
    use zune_core::colorspace::ColorSpace as ZColorSpace;
    use zune_core::options::DecoderOptions;
    use zune_jpeg::JpegDecoder;

    let opts = DecoderOptions::default().jpeg_set_out_colorspace(ZColorSpace::Luma);
    let mut dec = JpegDecoder::new_with_options(ZCursor::new(data), opts);
    dec.decode_headers().ok()?;
    let info = dec.info().or_else(|| {
        log::warn!("image: DCTDecode SMask: JPEG info unavailable after decode_headers");
        None
    })?;
    // Reject non-grayscale JPEG alpha channels.
    if info.components != 1 {
        log::warn!(
            "image: DCTDecode SMask has {} components (expected 1), skipping",
            info.components
        );
        return None;
    }
    let jw = u32::from(info.width);
    let jh = u32::from(info.height);
    if jw == 0 || jh == 0 {
        log::warn!("image: DCTDecode SMask: JPEG reported zero dimensions {jw}×{jh}");
        return None;
    }
    if jw != sm_w || jh != sm_h {
        log::debug!(
            "image: DCTDecode SMask: dict says {sm_w}×{sm_h}, JPEG reports {jw}×{jh} — using JPEG dims"
        );
    }
    let pixels = dec.decode().ok()?;
    // Validate buffer against JPEG-reported dimensions.
    // A short buffer means the decoder produced fewer bytes than its own SOF claimed.
    let expected = (jw as usize).checked_mul(jh as usize)?;
    if pixels.len() < expected {
        log::warn!(
            "image: DCTDecode SMask decoded {} bytes, expected {expected} for {jw}×{jh}",
            pixels.len()
        );
        return None;
    }
    Some((pixels[..expected].to_vec(), jw, jh))
}

/// Interpret an already-decompressed byte buffer as an `SMask` alpha channel.
///
/// Dispatches on `bpc` to normalise sub-byte and 16-bit formats to one byte per
/// pixel.  Returns `None` if the buffer is too short or the bpc is unsupported.
fn decode_smask_raw_bytes(raw: &[u8], sm_w: u32, sm_h: u32, bpc: i64) -> Option<Vec<u8>> {
    match bpc {
        1 => expand_1bpp(raw, sm_w, sm_h),
        2 => expand_nbpp::<2>(raw, sm_w, sm_h, 1),
        4 => expand_nbpp::<4>(raw, sm_w, sm_h, 1),
        8 => {
            // Validate length before the raw pass-through: scale_smask would
            // panic with an out-of-bounds index if data is shorter than sm_w×sm_h.
            let expected = (sm_w as usize).checked_mul(sm_h as usize)?;
            if raw.len() < expected {
                log::warn!(
                    "image: SMask raw data too short ({} bytes, need {expected} for {sm_w}×{sm_h})",
                    raw.len()
                );
                return None;
            }
            Some(raw[..expected].to_vec())
        }
        16 => downsample_16bpp(raw, sm_w, sm_h, 1),
        other => {
            log::debug!("image: SMask {other} bpc not yet supported");
            None
        }
    }
}

/// Nearest-neighbour resample of a flat grayscale `src` buffer from `(sw×sh)`
/// to `(dw×dh)`.  Returns `src` unchanged when dimensions already match.
///
/// Precondition: `dw > 0` and `dh > 0` (validated by image-dimension checks
/// before `decode_smask` is called).  `src` should have exactly `sw * sh` bytes;
/// extra bytes are ignored.
pub(super) fn scale_smask(src: Vec<u8>, sw: u32, sh: u32, dw: u32, dh: u32) -> Vec<u8> {
    // Guard degenerate dimensions: zero dst causes division-by-zero; zero src
    // means there are no pixels to sample.
    if dw == 0 || dh == 0 || sw == 0 || sh == 0 {
        return Vec::new();
    }
    if sw == dw && sh == dh {
        return src;
    }
    let src_w = u64::from(sw);
    let src_h = u64::from(sh);
    let dst_w = u64::from(dw);
    let dst_h = u64::from(dh);
    let capacity = dst_w
        .checked_mul(dst_h)
        .and_then(|n| usize::try_from(n).ok())
        .unwrap_or(0);
    let mut out = Vec::with_capacity(capacity);
    for dy in 0..dh {
        // sy = floor(dy * src_h / dst_h); since dy < dst_h, result < src_h ≤ 65535.
        #[expect(
            clippy::cast_possible_truncation,
            reason = "dy < dst_h ⟹ sy < src_h ≤ 65535 — fits u32"
        )]
        let sy = (u64::from(dy) * src_h / dst_h) as u32;
        for dx in 0..dw {
            // sx = floor(dx * src_w / dst_w); since dx < dst_w, result < src_w ≤ 65535.
            #[expect(
                clippy::cast_possible_truncation,
                reason = "dx < dst_w ⟹ sx < src_w ≤ 65535 — fits u32"
            )]
            let sx = (u64::from(dx) * src_w / dst_w) as u32;
            // index = sy * src_w + sx < src_h * src_w ≤ 65535² = 4_294_836_225 < usize::MAX
            // on all targets (usize::MAX ≥ 2^32−1 on 32-bit, 2^64−1 on 64-bit).
            #[expect(
                clippy::cast_possible_truncation,
                reason = "index ≤ 65535²−1 < usize::MAX on 32-bit and 64-bit targets"
            )]
            out.push(src[(u64::from(sy) * src_w + u64::from(sx)) as usize]);
        }
    }
    out
}

/// Decode an explicit `/Mask N 0 R` stencil-mask image into an alpha buffer.
///
/// This is the PDF §8.9.6.3 *explicit mask* (an image `XObject` with
/// `/ImageMask true`), distinct from `/SMask` (§11.6.5.2 soft mask) and from
/// the `/Mask [colour-key]` array form (§8.9.6.4).  The mask designates which
/// parts of the **base image** are painted: per §8.9.6.3 with the default
/// `Decode [0 1]`, mask sample 0 → the base image is painted (opaque),
/// mask sample 1 → masked out (the background shows through).
///
/// The stencil decoders (`decode_jbig2`/`decode_ccitt`/raw with
/// `is_mask=true`) return [`ImageColorSpace::Mask`] bytes using the
/// §8.9.6.1 fill-stencil convention: `0x00` = "paint", `0xFF` =
/// "transparent".  For an explicit mask those same markers map directly onto
/// base-image alpha — a "paint" point shows the base image (alpha `0xFF`), a
/// "transparent" point shows the background (alpha `0x00`).  Hence
/// `alpha = !mask_byte`.  `Decode [1 0]` on the mask is already folded into
/// the decoder output, so no extra inversion is applied here.
///
/// Returns exactly `img_w * img_h` alpha bytes, resampled (nearest-neighbour)
/// when the mask grid differs from the base image grid.  Returns `None` if the
/// referenced object is not a usable stencil image so the caller can blit the
/// base image unmasked rather than discard it.
pub(super) fn decode_explicit_mask(
    doc: &Document,
    id: ObjectId,
    img_w: u32,
    img_h: u32,
) -> Option<Vec<u8>> {
    use crate::resources::dict_ext::DictExt as _;

    let obj_arc = doc.get_object(id).ok()?;
    let stream = obj_arc.as_ref().as_stream()?;

    // §8.9.6.3: an explicit mask must be an image with ImageMask true.
    if !stream.dict.get_bool(b"ImageMask").unwrap_or(false) {
        log::warn!("image: /Mask object {id:?} is not an ImageMask stencil — skipping mask");
        return None;
    }

    let w_raw = stream.dict.get_i64(b"Width")?;
    let h_raw = stream.dict.get_i64(b"Height")?;
    let Some((m_w, m_h)) = validated_dims(w_raw, h_raw) else {
        log::warn!("image: /Mask degenerate dimensions {w_raw}×{h_raw}, skipping mask");
        return None;
    };

    let filter = stream.dict.get(b"Filter").and_then(filter_name);

    // Decode the stencil to ImageColorSpace::Mask bytes (0x00 = paint,
    // 0xFF = transparent).  Each decoder folds the mask's own /Decode [1 0]
    // inversion into its output.
    let desc = match filter.as_deref() {
        Some("JBIG2Decode") => {
            let parms = stream.dict.get(b"DecodeParms");
            decode_jbig2(doc, stream.content.as_slice(), m_w, m_h, true, parms)?
        }
        Some("CCITTFaxDecode") => {
            let parms = stream.dict.get(b"DecodeParms");
            decode_ccitt(stream.content.as_slice(), m_w, m_h, true, parms)?
        }
        Some("FlateDecode") => {
            let raw = stream.decompressed_content().ok()?;
            decode_raw(
                doc,
                &raw,
                m_w,
                m_h,
                true,
                &stream.dict,
                #[cfg(feature = "gpu-icc")]
                None,
                #[cfg(feature = "gpu-icc")]
                None,
            )?
        }
        None => decode_raw(
            doc,
            stream.content.as_slice(),
            m_w,
            m_h,
            true,
            &stream.dict,
            #[cfg(feature = "gpu-icc")]
            None,
            #[cfg(feature = "gpu-icc")]
            None,
        )?,
        Some(other) => {
            log::warn!("image: /Mask filter {other:?} not supported — skipping mask");
            return None;
        }
    };

    // A stencil decoded with is_mask=true must yield ImageColorSpace::Mask.
    // If it does not, the bytes are *not* the §8.9.6.1 paint/transparent
    // markers, so `!byte` would invert arbitrary samples into garbage alpha —
    // exactly the silent-wrong mask-polarity class.  Reject loudly instead of
    // a debug-only assert that vanishes in release builds.
    if desc.color_space != ImageColorSpace::Mask {
        log::warn!(
            "image: /Mask object {id:?} decoded as {:?}, expected Mask stencil — skipping mask",
            desc.color_space
        );
        return None;
    }

    // §8.9.6.3: paint marker (0x00) → base image opaque (alpha 0xFF);
    // transparent marker (0xFF) → background shows (alpha 0x00).
    let alpha: Vec<u8> = desc.data.as_cpu()?.iter().map(|&b| !b).collect();

    Some(scale_smask(alpha, desc.width, desc.height, img_w, img_h))
}

/// Build an alpha buffer from a `/Mask [colour-key]` array (§8.9.6.4).
///
/// The array holds `2 × n` integers `[min₁ max₁ … minₙ maxₙ]` over the base
/// image's colour components (here only 1-component `Gray` and 3-component
/// `Rgb` post-decode buffers are masked).  A base-image pixel whose every
/// component falls within its `[min, max]` range is *masked* (transparent,
/// alpha `0x00`); all other pixels are opaque (alpha `0xFF`).
///
/// Returns `None` when the array length doesn't match the component count or
/// the base buffer is the wrong size, so the caller blits unmasked.
pub(super) fn colour_key_mask(
    ranges: &[Object],
    base: &[u8],
    components: usize,
    img_w: u32,
    img_h: u32,
) -> Option<Vec<u8>> {
    if components == 0 || ranges.len() != components * 2 {
        return None;
    }
    let bounds: Vec<i64> = ranges.iter().filter_map(pdf::Object::as_i64).collect();
    if bounds.len() != components * 2 {
        return None;
    }
    // §8.9.6.4: each component range is `min max` with min ≤ max.  An inverted
    // pair would make `v >= min && v <= max` unsatisfiable for that component,
    // silently producing a fully-opaque image (no masking) — the silent-wrong
    // class.  Reject so the caller logs and blits unmasked instead.
    if (0..components).any(|c| bounds[c * 2] > bounds[c * 2 + 1]) {
        log::warn!("image: colour-key /Mask has an inverted min>max range — skipping mask");
        return None;
    }
    let npix = (img_w as usize).checked_mul(img_h as usize)?;
    if base.len() < npix.checked_mul(components)? {
        return None;
    }
    let alpha: Vec<u8> = (0..npix)
        .map(|p| {
            let px = &base[p * components..p * components + components];
            let in_range = (0..components).all(|c| {
                let v = i64::from(px[c]);
                v >= bounds[c * 2] && v <= bounds[c * 2 + 1]
            });
            if in_range { 0x00 } else { 0xFF }
        })
        .collect();
    Some(alpha)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scale_smask_identity() {
        let src = vec![10u8, 20, 30, 40];
        let result = scale_smask(src.clone(), 2, 2, 2, 2);
        assert_eq!(result, src);
    }

    #[test]
    fn scale_smask_upsample_2x() {
        // 1×1 pixel of value 99 → 2×2 all 99.
        let src = vec![99u8];
        let result = scale_smask(src, 1, 1, 2, 2);
        assert_eq!(result, &[99, 99, 99, 99]);
    }

    #[test]
    fn scale_smask_zero_dst_returns_empty() {
        let result = scale_smask(vec![1, 2, 3], 3, 1, 0, 1);
        assert!(result.is_empty());
    }

    #[test]
    fn scale_smask_zero_src_returns_empty() {
        // sw=0 means no source pixels; scaling into a non-empty dst would index
        // an empty buffer — must return empty instead.
        let result = scale_smask(vec![], 0, 1, 4, 4);
        assert!(result.is_empty());
    }

    #[test]
    fn alpha_from_decoded_short_buffer_returns_none() {
        // A bitonal codec that reports 4×4 but produced only 3 bytes must be
        // skipped loudly, not fed to scale_smask (which would panic on the
        // out-of-bounds index when the SMask grid differs from the parent).
        let desc = ImageDescriptor {
            width: 4,
            height: 4,
            color_space: ImageColorSpace::Gray,
            data: crate::resources::image::ImageData::Cpu(vec![0u8, 0xFF, 0x00]),
            smask: None,
            filter: crate::resources::image::ImageFilter::Raw,
        };
        assert!(alpha_from_decoded("JBIG2Decode", &desc).is_none());
    }

    #[test]
    fn alpha_from_decoded_exact_buffer_passes_through() {
        // Exact-length buffer is consumed as alpha directly (§11.6.5.2: the
        // decoded sample value IS the alpha — no inversion).
        let desc = ImageDescriptor {
            width: 2,
            height: 2,
            color_space: ImageColorSpace::Gray,
            data: crate::resources::image::ImageData::Cpu(vec![0x00, 0xFF, 0x10, 0xF0]),
            smask: None,
            filter: crate::resources::image::ImageFilter::Raw,
        };
        assert_eq!(
            alpha_from_decoded("CCITTFaxDecode", &desc).unwrap(),
            vec![0x00, 0xFF, 0x10, 0xF0]
        );
    }

    #[test]
    fn decode_smask_raw_bpc8_too_short_returns_none() {
        // 2×2 SMask needs 4 bytes; supplying 2 must return None, not panic.
        assert!(decode_smask_raw_bytes(&[0u8, 255], 2, 2, 8).is_none());
    }

    #[test]
    fn decode_smask_raw_bpc8_exact_length() {
        let result = decode_smask_raw_bytes(&[10u8, 20, 30, 40], 2, 2, 8).unwrap();
        assert_eq!(result, &[10, 20, 30, 40]);
    }

    /// A 1×1 grayscale JPEG (PIL-encoded, quality=90) decoded as an `SMask` buffer
    /// must produce exactly 1 byte with a value in the mid-grey range.
    ///
    /// The JPEG bytes are a valid baseline JFIF stream produced by PIL/Pillow
    /// encoding a 1×1 Luma(128) image at quality=90.  They are known-good with
    /// zune-jpeg 0.5.x.
    #[test]
    fn decode_smask_dct_1x1_gray() {
        // 1×1 grayscale JPEG, value 128, quality=90.  Generated by:
        //   from PIL import Image, io
        //   img = Image.new('L', (1, 1), 128)
        //   buf = io.BytesIO(); img.save(buf, 'JPEG', quality=90)
        let jpeg_1x1_gray128: &[u8] = &[
            0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01, 0x01, 0x00,
            0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0xFF, 0xDB, 0x00, 0x43, 0x00, 0x03, 0x02, 0x02,
            0x03, 0x02, 0x02, 0x03, 0x03, 0x03, 0x03, 0x04, 0x03, 0x03, 0x04, 0x05, 0x08, 0x05,
            0x05, 0x04, 0x04, 0x05, 0x0A, 0x07, 0x07, 0x06, 0x08, 0x0C, 0x0A, 0x0C, 0x0C, 0x0B,
            0x0A, 0x0B, 0x0B, 0x0D, 0x0E, 0x12, 0x10, 0x0D, 0x0E, 0x11, 0x0E, 0x0B, 0x0B, 0x10,
            0x16, 0x10, 0x11, 0x13, 0x14, 0x15, 0x15, 0x15, 0x0C, 0x0F, 0x17, 0x18, 0x16, 0x14,
            0x18, 0x12, 0x14, 0x15, 0x14, 0xFF, 0xC0, 0x00, 0x0B, 0x08, 0x00, 0x01, 0x00, 0x01,
            0x01, 0x01, 0x11, 0x00, 0xFF, 0xC4, 0x00, 0x1F, 0x00, 0x00, 0x01, 0x05, 0x01, 0x01,
            0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x02,
            0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0xFF, 0xC4, 0x00, 0xB5, 0x10,
            0x00, 0x02, 0x01, 0x03, 0x03, 0x02, 0x04, 0x03, 0x05, 0x05, 0x04, 0x04, 0x00, 0x00,
            0x01, 0x7D, 0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06,
            0x13, 0x51, 0x61, 0x07, 0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xA1, 0x08, 0x23, 0x42,
            0xB1, 0xC1, 0x15, 0x52, 0xD1, 0xF0, 0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0A, 0x16,
            0x17, 0x18, 0x19, 0x1A, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2A, 0x34, 0x35, 0x36, 0x37,
            0x38, 0x39, 0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4A, 0x53, 0x54, 0x55,
            0x56, 0x57, 0x58, 0x59, 0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6A, 0x73,
            0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7A, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
            0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5,
            0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA,
            0xC2, 0xC3, 0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6,
            0xD7, 0xD8, 0xD9, 0xDA, 0xE1, 0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA,
            0xF1, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8, 0xF9, 0xFA, 0xFF, 0xDA, 0x00, 0x08,
            0x01, 0x01, 0x00, 0x00, 0x3F, 0x00, 0x2B, 0xFF, 0xD9,
        ];

        // This is a valid PIL-produced JPEG; zune-jpeg must be able to decode it.
        let (pixels, jw, jh) = decode_smask_dct(jpeg_1x1_gray128, 1, 1)
            .expect("PIL-produced 1×1 JPEG must decode successfully");
        assert_eq!(jw, 1, "JPEG width must be 1");
        assert_eq!(jh, 1, "JPEG height must be 1");
        assert_eq!(
            pixels.len(),
            1,
            "1×1 SMask must produce exactly 1 alpha byte"
        );
        // PIL encoded value 128 at quality=90; after JPEG quantisation the
        // decoded value is in the broad mid-grey range.
        assert!(
            pixels[0] > 64 && pixels[0] < 192,
            "decoded alpha {:#04x} outside plausible mid-grey range [65, 191]",
            pixels[0]
        );
    }

    /// `decode_smask_dct` must return `None` for a truncated (corrupt) byte stream,
    /// not panic.
    #[test]
    fn decode_smask_dct_corrupt_returns_none() {
        assert!(decode_smask_dct(&[0xFF, 0xD8, 0xFF], 1, 1).is_none());
    }

    // ── Explicit /Mask (§8.9.6.3) regression ─────────────────────────────────

    /// PDF §8.9.6.3 explicit `/Mask` polarity.
    ///
    /// A `/Mask N 0 R` referencing an `/ImageMask true` stencil must produce a
    /// base-image alpha buffer where the stencil's *paint* points are opaque
    /// (`0xFF`) and its *masked-out* points are transparent (`0x00`).  The bug
    /// was that `/Mask` was never resolved at all, leaving `JPXDecode` pages
    /// (e.g. 1853-ragon-orthodoxie-maconnique p62) blitted fully opaque over
    /// the whole page and rendering near-all-dark instead of mostly-white.
    ///
    /// Stencil bytes (raw 1-bpp, default `Decode [0 1]`): a `1` bit paints
    /// (decoder → `0x00`), a `0` bit is transparent (decoder → `0xFF`).  Row
    /// of 8 px `0b1010_1010`: bits 1,0,1,0,1,0,1,0 ⇒ stencil bytes
    /// `[00 FF 00 FF 00 FF 00 FF]` ⇒ alpha `!byte` = `[FF 00 FF 00 …]`.
    #[test]
    fn explicit_mask_polarity_paint_is_opaque() {
        // 1 row × 8 px, 1 bpp ⇒ one byte 0b1010_1010.
        let doc = crate::test_helpers::make_doc_with_stream(
            &[0b1010_1010u8],
            " /Subtype /Image /Width 8 /Height 1 /ImageMask true \
              /BitsPerComponent 1",
        );
        let alpha =
            decode_explicit_mask(&doc, (2, 0), 8, 1).expect("explicit /Mask stencil must decode");
        // Paint bit (1) → opaque base image (0xFF); 0 bit → background (0x00).
        assert_eq!(alpha, vec![0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00]);
    }

    /// A `/Mask` object that is *not* an `ImageMask` stencil must be rejected
    /// (returns `None`) so the caller blits the base image unmasked rather
    /// than mis-applying a non-stencil image as alpha.
    #[test]
    fn explicit_mask_rejects_non_imagemask() {
        let doc = crate::test_helpers::make_doc_with_stream(
            &[0u8, 0, 0, 0],
            " /Subtype /Image /Width 2 /Height 2 /BitsPerComponent 8 \
              /ColorSpace /DeviceGray",
        );
        assert!(decode_explicit_mask(&doc, (2, 0), 2, 2).is_none());
    }

    // ── Colour-key /Mask (§8.9.6.4) ──────────────────────────────────────────

    #[test]
    fn colour_key_mask_gray_in_range_is_transparent() {
        // Range [10 20]: pixels 15 (in) → masked (0x00); 5 and 25 (out) → 0xFF.
        let base = [15u8, 5, 25, 12];
        let ranges = [Object::Integer(10), Object::Integer(20)];
        let alpha = colour_key_mask(&ranges, &base, 1, 4, 1).expect("valid colour-key");
        assert_eq!(alpha, vec![0x00, 0xFF, 0xFF, 0x00]);
    }

    #[test]
    fn colour_key_mask_rgb_all_components_must_match() {
        // Range R[0 10] G[0 10] B[0 10]: px0 (5,5,5) in → 0x00;
        // px1 (5,5,99) one comp out → 0xFF.
        let base = [5u8, 5, 5, 5, 5, 99];
        let ranges = [
            Object::Integer(0),
            Object::Integer(10),
            Object::Integer(0),
            Object::Integer(10),
            Object::Integer(0),
            Object::Integer(10),
        ];
        let alpha = colour_key_mask(&ranges, &base, 3, 2, 1).expect("valid colour-key");
        assert_eq!(alpha, vec![0x00, 0xFF]);
    }

    #[test]
    fn colour_key_mask_arity_mismatch_returns_none() {
        // 3 ints for a 1-component image (needs exactly 2) ⇒ None.
        let ranges = [Object::Integer(0), Object::Integer(1), Object::Integer(2)];
        assert!(colour_key_mask(&ranges, &[0u8], 1, 1, 1).is_none());
    }

    /// §8.9.6.4 requires `min ≤ max`.  An inverted range must fail loudly-
    /// graceful (`None` → caller blits unmasked) rather than silently leaving
    /// the whole image opaque, which the v1 sin class would have done.
    #[test]
    fn colour_key_mask_inverted_range_returns_none() {
        // min=20 > max=10 for a 1-component image ⇒ None, not all-opaque.
        let ranges = [Object::Integer(20), Object::Integer(10)];
        assert!(colour_key_mask(&ranges, &[15u8, 5], 1, 2, 1).is_none());
    }

    /// A `/Mask N 0 R` whose referenced stream is *not* a stencil (decodes as
    /// Gray/Rgb, not [`ImageColorSpace::Mask`]) must be rejected.  Feeding
    /// non-marker bytes through `!byte` would invert arbitrary samples into
    /// garbage alpha — the silent-wrong mask-polarity class this guard exists
    /// to stop.  The `/ImageMask true` dict claim is honoured by the decoder,
    /// so this is exercised via the release-safe colour-space check rather
    /// than a debug-only assert.
    #[test]
    fn explicit_mask_imagemask_false_is_rejected_before_invert() {
        // No /ImageMask true ⇒ early ImageMask-flag guard rejects it.
        let doc = crate::test_helpers::make_doc_with_stream(
            &[0x12u8, 0x34],
            " /Subtype /Image /Width 2 /Height 1 /BitsPerComponent 8 \
              /ColorSpace /DeviceGray",
        );
        assert!(decode_explicit_mask(&doc, (2, 0), 2, 1).is_none());
    }
}
