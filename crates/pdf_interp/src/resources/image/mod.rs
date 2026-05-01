//! PDF Image `XObject` resource extraction and decoding.
//!
//! [`resolve_image`] looks up a named `XObject` in the page resource dictionary
//! and, if it is an image, returns an [`ImageDescriptor`] with the decoded
//! pixel data ready for blitting.
//!
//! # Supported image types
//!
//! | Filter | `ImageMask` | Support |
//! |---|---|---|
//! | `CCITTFaxDecode` (`K<0`, Group 4 / T.6) | yes/no | yes |
//! | `CCITTFaxDecode` (`K=0`, Group 3 1D / T.4) | yes/no | yes |
//! | `FlateDecode` | yes/no | yes |
//! | none (raw) | yes/no | yes |
//! | `DCTDecode` (JPEG) | no | yes (CPU software or GPU nvJPEG) |
//! | `JPXDecode` (JPEG 2000) | no | yes (CPU software or GPU nvJPEG2000) |
//! | `JBIG2Decode` | yes/no | yes (via `hayro-jbig2`) |
//! | `CCITTFaxDecode` (`K>0`, Group 3 mixed 2D) | — | stub |
//!
//! # Pixel layout in `ImageDescriptor::data`
//!
//! | [`ImageColorSpace`] | Bytes per pixel | `0x00` meaning | `0xFF` meaning |
//! |---|---|---|---|
//! | `Gray` | 1 | black | white |
//! | `Rgb` | 3 | black (R=G=B=0) | white |
//! | `Mask` | 1 | paint with fill colour | transparent (leave background) |
//!
//! # nvJPEG acceleration (`nvjpeg` feature)
//!
//! When the crate is built with `--features nvjpeg`, `DCTDecode` streams with
//! pixel area ≥ [`GPU_JPEG_THRESHOLD_PX`] are decoded on the GPU via NVIDIA
//! nvJPEG instead of the CPU JPEG decoder.  Pass an [`NvJpegDecoder`] to
//! [`resolve_image`] to enable this path; pass `None` for CPU-only behaviour.
//!
//! # nvJPEG2000 acceleration (`nvjpeg2k` feature)
//!
//! When the crate is built with `--features nvjpeg2k`, `JPXDecode` streams
//! with pixel area ≥ [`GPU_JPEG2K_THRESHOLD_PX`] are decoded on the GPU via
//! NVIDIA nvJPEG2000 instead of the CPU JPEG 2000 decoder.  Pass an
//! [`NvJpeg2kDecoder`] to [`resolve_image`] to enable this path; pass `None`
//! for CPU-only behaviour.  Only 1- and 3-component images are accelerated;
//! CMYK and other multi-channel images always fall through to the CPU path.

use std::borrow::Cow;

use lopdf::{Dictionary, Document, Object, ObjectId};

use crate::resources::dict_ext::DictExt;

// ── GPU JPEG/JPEG2k acceleration ──────────────────────────────────────────────

#[cfg(feature = "nvjpeg")]
use gpu::nvjpeg::NvJpegDecoder;

#[cfg(feature = "nvjpeg2k")]
use gpu::nvjpeg2k::NvJpeg2kDecoder;

#[cfg(feature = "vaapi")]
use gpu::vaapi::VapiJpegDecoder;

// ── GPU ICC CMYK→RGB acceleration ─────────────────────────────────────────────

#[cfg(feature = "gpu-icc")]
use gpu::GpuCtx;

#[cfg(feature = "gpu-icc")]
#[path = "../icc.rs"]
pub(crate) mod icc;

/// Minimum pixel area (width × height) for GPU-accelerated `DCTDecode`.
///
/// Below this threshold transfer and decode setup overhead dominates and the
/// CPU decoder is faster.  512 × 512 = 262 144 pixels — empirically the
/// crossover between nvJPEG (~10 GB/s) and the CPU JPEG path (~1 GB/s) after
/// DMA latency, and a similarly effective threshold for VA-API.
#[cfg(any(feature = "nvjpeg", feature = "vaapi"))]
pub const GPU_JPEG_THRESHOLD_PX: u32 = 262_144;

/// Minimum pixel area (width × height) for GPU-accelerated `JPXDecode`.
///
/// Below this threshold `PCIe` transfer overhead dominates and the CPU decoder
/// is faster.  512 × 512 = 262 144 pixels — same crossover
/// as nvJPEG; JPEG 2000 decode is CPU-bound at similar pixel counts.
#[cfg(feature = "nvjpeg2k")]
pub const GPU_JPEG2K_THRESHOLD_PX: u32 = 262_144;

// ── Sub-modules ───────────────────────────────────────────────────────────────

pub(crate) mod codecs;
pub(crate) mod inline;

pub use inline::decode_inline_image;

/// Thin fuzz-only wrappers that forward to the `pub(super)` codec functions.
#[cfg(fuzzing)]
pub mod fuzz_entry {
    use lopdf::{Document, Object};

    use super::codecs;
    use super::ImageDescriptor;

    #[doc(hidden)]
    pub fn decode_ccitt(
        data: &[u8],
        width: u32,
        height: u32,
        is_mask: bool,
        parms: Option<&Object>,
    ) -> Option<ImageDescriptor> {
        codecs::decode_ccitt(data, width, height, is_mask, parms)
    }

    #[doc(hidden)]
    pub fn decode_jbig2(
        doc: &Document,
        data: &[u8],
        width: u32,
        height: u32,
        is_mask: bool,
        parms: Option<&Object>,
    ) -> Option<ImageDescriptor> {
        codecs::decode_jbig2(doc, data, width, height, is_mask, parms)
    }
}

// ── Public types ──────────────────────────────────────────────────────────────

/// Colour space of the decoded image pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageColorSpace {
    /// One byte per pixel, 0 = black, 255 = white.
    Gray,
    /// Three bytes per pixel: R, G, B.
    Rgb,
    /// 1-bit mask: pixel = 0 means "paint with current colour", 1 = transparent.
    Mask,
}

/// The compression filter used to store an image in the PDF stream.
///
/// Used by [`PageDiagnostics`](crate::renderer::page::PageDiagnostics) to
/// classify image content without re-reading the stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
pub enum ImageFilter {
    /// `DCTDecode` — JPEG baseline or progressive.
    Dct = 0,
    /// `JPXDecode` — JPEG 2000.
    Jpx = 1,
    /// `CCITTFaxDecode` — Group 3 (T.4) or Group 4 (T.6) fax compression.
    CcittFax = 2,
    /// `JBIG2Decode` — JBIG2 bilevel compression.
    Jbig2 = 3,
    /// `FlateDecode` — zlib/deflate.
    Flate = 4,
    /// No filter — raw uncompressed pixels.
    Raw = 5,
}

/// Number of [`ImageFilter`] variants — must match `filter_counts` array size in
/// `PageRenderer`.  The const assert in that module enforces this at compile time.
pub const IMAGE_FILTER_COUNT: usize = 6;

/// Decoded image data, ready for blitting onto the page bitmap.
///
/// See the module-level table for the exact byte layout per [`ImageColorSpace`].
#[derive(Debug)]
pub struct ImageDescriptor {
    /// Pixel width of the decoded image.
    pub width: u32,
    /// Pixel height of the decoded image.
    pub height: u32,
    /// Colour interpretation of `data`.
    pub color_space: ImageColorSpace,
    /// Raw pixel bytes — layout defined by `color_space` (see module doc).
    pub data: Vec<u8>,
    /// Optional soft-mask (`SMask`): one alpha byte per pixel, with exactly
    /// `width × height` entries matching the image grid.  `0x00` = fully
    /// transparent (pixel is skipped during blit); `0xFF` = fully opaque.
    /// `None` means every pixel is fully opaque.
    pub smask: Option<Vec<u8>>,
    /// The compression filter that was applied to this image in the PDF stream.
    pub filter: ImageFilter,
}

// ── Entry point ───────────────────────────────────────────────────────────────

/// Look up a named `XObject` in the page resource dictionary and, if it is an
/// image, decode and return its pixels.
///
/// Returns `None` if:
/// - the name is not present in the `XObject` resource dict,
/// - the object is not an image (`Subtype != Image`),
/// - the filter is unsupported (a warning is logged), or
/// - any decoding error occurs.
#[must_use]
pub fn resolve_image(
    doc: &Document,
    page_dict: &Dictionary,
    name: &[u8],
    #[cfg(feature = "nvjpeg")] gpu: Option<&mut NvJpegDecoder>,
    #[cfg(feature = "vaapi")] vaapi: Option<&mut VapiJpegDecoder>,
    #[cfg(feature = "nvjpeg2k")] gpu_j2k: Option<&mut NvJpeg2kDecoder>,
    #[cfg(feature = "gpu-icc")] gpu_ctx: Option<&GpuCtx>,
) -> Option<ImageDescriptor> {
    use codecs::{decode_ccitt, decode_dct, decode_jbig2, decode_jpx};

    let stream_id = xobject_id(doc, page_dict, name)?;
    let obj = doc.get_object(stream_id).ok()?;
    let stream = obj.as_stream().ok()?;

    // Must be an Image subtype.
    if stream.dict.get_name(b"Subtype")? != b"Image" {
        return None;
    }

    let w_raw = stream.dict.get_i64(b"Width")?;
    let h_raw = stream.dict.get_i64(b"Height")?;
    let Some((w, h)) = validated_dims(w_raw, h_raw) else {
        log::warn!("image: degenerate dimensions {w_raw}×{h_raw}, skipping");
        return None;
    };

    let is_mask = stream.dict.get_bool(b"ImageMask").unwrap_or(false);

    let filter = stream.dict.get(b"Filter").ok().and_then(filter_name);

    let img_filter = match filter.as_deref() {
        Some("DCTDecode") => ImageFilter::Dct,
        Some("JPXDecode") => ImageFilter::Jpx,
        Some("CCITTFaxDecode") => ImageFilter::CcittFax,
        Some("JBIG2Decode") => ImageFilter::Jbig2,
        Some("FlateDecode") => ImageFilter::Flate,
        _ => ImageFilter::Raw,
    };

    let mut img = match filter.as_deref() {
        None => decode_raw(
            doc,
            stream.content.as_slice(),
            w,
            h,
            is_mask,
            &stream.dict,
            #[cfg(feature = "gpu-icc")]
            gpu_ctx,
        ),
        Some("FlateDecode") => match stream.decompressed_content() {
            Ok(data) => decode_raw(
                doc,
                &data,
                w,
                h,
                is_mask,
                &stream.dict,
                #[cfg(feature = "gpu-icc")]
                gpu_ctx,
            ),
            Err(e) => {
                log::warn!("image: FlateDecode decompression failed: {e}");
                None
            }
        },
        Some("CCITTFaxDecode") => {
            let parms = stream.dict.get(b"DecodeParms").ok();
            decode_ccitt(stream.content.as_slice(), w, h, is_mask, parms)
        }
        Some("DCTDecode") => decode_dct(
            stream.content.as_slice(),
            w,
            h,
            #[cfg(feature = "nvjpeg")]
            gpu,
            #[cfg(feature = "vaapi")]
            vaapi,
            #[cfg(feature = "gpu-icc")]
            gpu_ctx,
        ),
        Some("JPXDecode") => {
            #[cfg(feature = "nvjpeg2k")]
            {
                decode_jpx(stream.content.as_slice(), w, h, gpu_j2k)
            }
            #[cfg(not(feature = "nvjpeg2k"))]
            {
                decode_jpx(stream.content.as_slice(), w, h)
            }
        }
        Some("JBIG2Decode") => {
            let parms = stream.dict.get(b"DecodeParms").ok();
            decode_jbig2(doc, stream.content.as_slice(), w, h, is_mask, parms)
        }
        Some(other) => {
            log::warn!("image: unknown filter {other:?}");
            None
        }
    }?;
    img.filter = img_filter;

    // Resolve and decode the soft mask (`SMask`), if present.
    if let Ok(Object::Reference(smask_id)) = stream.dict.get(b"SMask") {
        if let Some(alpha) = decode_smask(doc, *smask_id, img.width, img.height) {
            img.smask = Some(alpha);
        } else {
            // `SMask` is present but could not be decoded.  Blitting without a
            // mask would paint the image's colour over a large area it should
            // not cover (e.g. a solid-colour overlay that is transparent
            // everywhere the mask is zero).  Skip the image instead.
            log::warn!("image: skipping image — SMask (object {smask_id:?}) could not be decoded");
            return None;
        }
    }

    Some(img)
}

// ── SMask decoding ────────────────────────────────────────────────────────────

/// Decode a soft-mask (`SMask`) image stream into a flat alpha buffer.
///
/// Returns exactly `img_w * img_h` bytes, one alpha value per pixel:
/// `0x00` = fully transparent, `0xFF` = fully opaque.  The `SMask` stream may
/// have different dimensions from the parent image; if so, the buffer is
/// resampled to match via nearest-neighbour scaling.
///
/// Returns `None` when the stream is unresolvable, its filter is unsupported,
/// or its dimensions are degenerate.  The caller must skip the image in that
/// case rather than blit it without a mask.
fn decode_smask(doc: &Document, id: ObjectId, img_w: u32, img_h: u32) -> Option<Vec<u8>> {
    use codecs::{decode_ccitt, decode_jbig2};

    let obj = doc.get_object(id).ok()?;
    let stream = obj.as_stream().ok()?;

    let w_raw = stream.dict.get_i64(b"Width")?;
    let h_raw = stream.dict.get_i64(b"Height")?;
    let Some((sm_w, sm_h)) = validated_dims(w_raw, h_raw) else {
        log::debug!("image: SMask degenerate dimensions {w_raw}×{h_raw}, skipping");
        return None;
    };

    let filter = stream.dict.get(b"Filter").ok().and_then(filter_name);
    let bpc = stream.dict.get_i64(b"BitsPerComponent").unwrap_or(8);

    // Decode the compressed stream to a raw byte buffer, then interpret by bpc.
    // CCITTFaxDecode is handled separately because decode_ccitt returns Mask-space
    // (0x00 = paint, 0xFF = transparent), which must be inverted to alpha-space.
    let alpha: Vec<u8> = match filter.as_deref() {
        Some("CCITTFaxDecode") => {
            let parms = stream.dict.get(b"DecodeParms").ok();
            let sm_desc = decode_ccitt(stream.content.as_slice(), sm_w, sm_h, true, parms)?;
            // Mask-space polarity: 0x00 = fill (opaque in alpha) ↔ 0xFF = skip (transparent).
            sm_desc
                .data
                .iter()
                .map(|&v| if v == 0x00 { 0xFF } else { 0x00 })
                .collect()
        }
        Some("JBIG2Decode") => {
            let parms = stream.dict.get(b"DecodeParms").ok();
            let sm_desc = decode_jbig2(doc, stream.content.as_slice(), sm_w, sm_h, true, parms)?;
            // Mask-space polarity: 0x00 = fill (opaque in alpha) ↔ 0xFF = skip (transparent).
            sm_desc
                .data
                .iter()
                .map(|&v| if v == 0x00 { 0xFF } else { 0x00 })
                .collect()
        }
        Some(other) => {
            log::debug!("image: SMask filter {other:?} not yet supported");
            return None;
        }
        // Raw or FlateDecode: decode to bytes then interpret via bpc.
        filter_opt => {
            let raw: Vec<u8> = match filter_opt {
                None => stream.content.clone(),
                Some("FlateDecode") => stream.decompressed_content().ok()?,
                _ => unreachable!("matched above"),
            };
            match bpc {
                1 => expand_smask_1bpp(&raw, sm_w, sm_h)?,
                2 => expand_nbpp::<2>(&raw, sm_w, sm_h, 1)?,
                4 => expand_nbpp::<4>(&raw, sm_w, sm_h, 1)?,
                8 => raw,
                16 => downsample_16bpp(&raw, sm_w, sm_h, 1)?,
                other => {
                    log::debug!("image: SMask {other} bpc not yet supported");
                    return None;
                }
            }
        }
    };

    Some(scale_smask(alpha, sm_w, sm_h, img_w, img_h))
}

/// Expand a 1-bit-per-pixel packed `SMask` buffer to one byte per pixel.
///
/// Bit 1 (MSB-first) = opaque (0xFF); bit 0 = transparent (0x00).
/// Truncated rows are defensively treated as all-transparent (0-bit padding).
/// Returns `None` if `sm_w × sm_h` overflows `usize`.
fn expand_smask_1bpp(raw: &[u8], sm_w: u32, sm_h: u32) -> Option<Vec<u8>> {
    let cols = usize::try_from(sm_w).ok()?;
    let rows = usize::try_from(sm_h).ok()?;
    let total = cols.checked_mul(rows)?;
    let row_bytes = cols.div_ceil(8);
    let mut out = Vec::with_capacity(total);
    // Iterate exactly `rows` times regardless of how many bytes `raw` contains —
    // a FlateDecode stream that decompresses to more bytes than the declared
    // dimensions imply must not produce more output rows than `sm_h`.
    for row_idx in 0..rows {
        let row_start = row_idx * row_bytes;
        let row = if row_start < raw.len() {
            &raw[row_start..raw.len().min(row_start + row_bytes)]
        } else {
            &[]
        };
        for x in 0..cols {
            let byte = row.get(x / 8).copied().unwrap_or(0);
            let bit = (byte >> (7 - (x % 8))) & 1;
            out.push(if bit != 0 { 0xFF } else { 0x00 });
        }
    }
    Some(out)
}

/// Nearest-neighbour resample of a flat grayscale `src` buffer from `(sw×sh)`
/// to `(dw×dh)`.  Returns `src` unchanged when dimensions already match.
///
/// Precondition: `dw > 0` and `dh > 0` (callers must ensure this; both are
/// validated as part of image-dimension checks before `decode_smask` is called).
/// `src` should have exactly `sw * sh` bytes; extra bytes are ignored.
fn scale_smask(src: Vec<u8>, sw: u32, sh: u32, dw: u32, dh: u32) -> Vec<u8> {
    // Guard degenerate dimensions: zero dst causes division-by-zero; zero src
    // means there are no pixels to sample and the index calculation below would
    // access an empty buffer.
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
        // sy = floor(dy * src_h / dst_h); since dy < dst_h, result < src_h ≤ u32::MAX.
        #[expect(
            clippy::cast_possible_truncation,
            reason = "dy < dst_h ⟹ sy < src_h ≤ 65536"
        )]
        let sy = (u64::from(dy) * src_h / dst_h) as u32;
        for dx in 0..dw {
            // sx = floor(dx * src_w / dst_w); since dx < dst_w, result < src_w ≤ u32::MAX.
            #[expect(
                clippy::cast_possible_truncation,
                reason = "dx < dst_w ⟹ sx < src_w ≤ 65536"
            )]
            let sx = (u64::from(dx) * src_w / dst_w) as u32;
            // sy * src_w + sx < src_h * src_w ≤ 65536² = 2³², fits in usize on any target
            // (pdf_interp only runs on 32-bit+ systems; usize is at least 32 bits).
            #[expect(
                clippy::cast_possible_truncation,
                reason = "index < src_h*src_w ≤ 65536² = 4G; usize ≥ 32 bits"
            )]
            out.push(src[(u64::from(sy) * src_w + u64::from(sx)) as usize]);
        }
    }
    out
}

// ── XObject lookup ────────────────────────────────────────────────────────────

/// Validate raw `i64` image dimensions and cast them to `u32`.
///
/// Returns `None` (caller logs and propagates) if either dimension is ≤ 0 or
/// exceeds 65536.  The 65536 cap keeps `w × h` within `u32` and limits the
/// maximum allocation a single image can request to ≈ 16 GiB (before component
/// multiplication), which is checked separately by each decoder.
///
/// The cast is safe: after the range check, the value is in [1, 65536] which
/// fits in both `u32` and `usize`.
#[expect(
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    reason = "value range validated to [1, 65536] — fits in u32 without loss"
)]
pub(super) const fn validated_dims(w_raw: i64, h_raw: i64) -> Option<(u32, u32)> {
    if w_raw <= 0 || h_raw <= 0 || w_raw > 65536 || h_raw > 65536 {
        return None;
    }
    Some((w_raw as u32, h_raw as u32))
}

/// Resolve the `XObject` resource named `name` to its stream `ObjectId`.
fn xobject_id(doc: &Document, page_dict: &Dictionary, name: &[u8]) -> Option<ObjectId> {
    let res = super::resolve_dict(doc, page_dict.get(b"Resources").ok()?)?;
    let xobj = super::resolve_dict(doc, res.get(b"XObject").ok()?)?;
    match xobj.get(name).ok()? {
        Object::Reference(id) => Some(*id),
        _ => None,
    }
}

/// Resolve a PDF colour space object to an [`ImageColorSpace`].
///
/// Convenience wrapper around the internal `resolve_cs` for use in the shading module.
/// CMYK colour spaces are converted to RGB as in the image decode path.
pub(crate) fn cs_to_image_color_space(doc: &Document, cs_obj: &Object) -> ImageColorSpace {
    match resolve_cs(doc, cs_obj) {
        ResolvedCs::Gray => ImageColorSpace::Gray,
        // CMYK in shadings: treat as RGB (callers convert channels before blitting).
        ResolvedCs::Rgb | ResolvedCs::Cmyk => ImageColorSpace::Rgb,
    }
}

// ── Filter / colour-space name extraction ─────────────────────────────────────

/// Extract the filter name from a `Filter` entry.
///
/// Accepts either a bare `Name` or a single-element `Name` array.  PDF allows
/// chained filters as a multi-element array; chained filters are not supported
/// here — a warning is emitted and `None` is returned so the caller can skip
/// the image gracefully rather than trying to decode garbled data.
pub(super) fn filter_name(obj: &Object) -> Option<Cow<'_, str>> {
    match obj {
        Object::Name(n) => Some(String::from_utf8_lossy(n)),
        Object::Array(arr) => {
            if arr.len() > 1 {
                log::warn!(
                    "image: chained filters ({} filters in array) not supported — skipping image",
                    arr.len()
                );
                return None;
            }
            arr.first()
                .and_then(|o| o.as_name().ok())
                .map(String::from_utf8_lossy)
        }
        _ => None,
    }
}

// ── Raw / FlateDecode image decoding ─────────────────────────────────────────

/// Expand raw (already-decompressed) pixel bytes into a normalised 8-bpp form.
///
/// Supported `BitsPerComponent` values:
/// - 1 — packed MSB-first, expanded to 0x00/0xFF per pixel
/// - 2 — 4 levels, scaled to 0x00/0x55/0xAA/0xFF
/// - 4 — 16 levels, scaled to 0x00…0xFF (value × 17)
/// - 8 — direct (common case)
/// - 16 — big-endian, high byte taken (linear 0–65535 → 0–255)
///
/// `Indexed` images with bpc 1/2/4 unpack the sub-byte palette indices first.
/// CMYK images (bpc 8/16) are converted to RGB inline.
#[expect(
    clippy::too_many_lines,
    reason = "bpc dispatch table + cfg-gated GPU paths — splitting would obscure the decision tree"
)]
pub(super) fn decode_raw(
    doc: &Document,
    data: &[u8],
    width: u32,
    height: u32,
    is_mask: bool,
    dict: &Dictionary,
    #[cfg(feature = "gpu-icc")] gpu_ctx: Option<&GpuCtx>,
) -> Option<ImageDescriptor> {
    let bpc = dict.get_i64(b"BitsPerComponent").unwrap_or(8);

    if is_mask {
        // Stencil mask — always 1 byte per pixel, no colour space conversion.
        return match bpc {
            1 => {
                let pixels = expand_1bpp(data, width, height)?;
                Some(ImageDescriptor {
                    width,
                    height,
                    color_space: ImageColorSpace::Mask,
                    data: pixels,
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
                    data: data[..expected].to_vec(),
                    smask: None,
                    filter: ImageFilter::Raw,
                })
            }
            other => {
                log::debug!("image: mask {other} bpc not yet implemented");
                None
            }
        };
    }

    // Read raw ColorSpace object to detect Indexed before resolving.
    let cs_obj = dict.get(b"ColorSpace").ok();

    // Check for Indexed colour space: [/Indexed base hival lookup].
    // These images carry palette indices (bpc 1/2/4/8 bits per index) rather
    // than component samples, so they need special handling before the general path.
    if let Some(Object::Array(arr)) = cs_obj
        && arr
            .first()
            .and_then(|o| o.as_name().ok())
            .is_some_and(|n| n == b"Indexed")
    {
        return decode_raw_indexed(doc, data, width, height, bpc, arr);
    }

    // General path: resolve the colour space to Gray / Rgb / Cmyk.
    let resolved = cs_obj.map_or(ResolvedCs::Gray, |o| resolve_cs(doc, o));

    // For ICCBased CMYK, extract the raw ICC profile bytes so they can be baked
    // into a CLUT for the GPU path.  Only performed under the gpu-icc feature and
    // only when the resolved space is actually CMYK (avoids unnecessary work for
    // Gray/RGB ICCBased images).
    #[cfg(feature = "gpu-icc")]
    let icc_bytes: Option<Vec<u8>> = if resolved == ResolvedCs::Cmyk {
        cs_obj.and_then(|o| extract_icc_bytes(doc, o))
    } else {
        None
    };

    match bpc {
        1 => {
            let pixels = expand_1bpp(data, width, height)?;
            Some(ImageDescriptor {
                width,
                height,
                color_space: resolved.to_image_cs(),
                data: pixels,
                smask: None,
                filter: ImageFilter::Raw,
            })
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
        ),
        other => {
            log::debug!("image: {other} bits-per-component not yet implemented");
            None
        }
    }
}

/// Decode an 8-bpp non-indexed raw image for `resolved` colour space.
///
/// CMYK images are converted to RGB.  Gray and RGB are returned as-is.
fn decode_raw_8bpp(
    data: &[u8],
    width: u32,
    height: u32,
    resolved: ResolvedCs,
    #[cfg(feature = "gpu-icc")] gpu_ctx: Option<&GpuCtx>,
    #[cfg(feature = "gpu-icc")] icc_bytes: Option<&[u8]>,
) -> Option<ImageDescriptor> {
    use codecs::cmyk_raw_to_rgb;

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
        // Raw PDF CMYK: 255 = full ink, 0 = no ink.
        let rgb = cmyk_raw_to_rgb(
            raw,
            #[cfg(feature = "gpu-icc")]
            gpu_ctx,
            #[cfg(feature = "gpu-icc")]
            icc_bytes,
        )?;
        Some(ImageDescriptor {
            width,
            height,
            color_space: ImageColorSpace::Rgb,
            data: rgb,
            smask: None,
            filter: ImageFilter::Raw,
        })
    } else {
        Some(ImageDescriptor {
            width,
            height,
            color_space: resolved.to_image_cs(),
            data: raw.to_vec(),
            smask: None,
            filter: ImageFilter::Raw,
        })
    }
}

/// Decode a raw `Indexed` colour space image.
///
/// For bpc 1, 2, 4: index values are packed MSB-first, N bits per index.
/// For bpc 8: one byte per index (the common case).
/// PDF spec §8.6.6.3 allows bpc ∈ {1, 2, 4, 8} for Indexed images.
fn decode_raw_indexed(
    doc: &Document,
    data: &[u8],
    width: u32,
    height: u32,
    bpc: i64,
    cs_arr: &[Object],
) -> Option<ImageDescriptor> {
    // Expand sub-byte index streams to one byte per index before palette lookup.
    let indices_8bit: Vec<u8>;
    let indices: &[u8] = match bpc {
        1 | 2 | 4 => {
            // For Indexed images, "components" is always 1 (each sample is one palette index).
            // bpc ∈ {1, 2, 4} verified by the match arm — safe to cast to u32.
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

    // `indexed_palette` guarantees stride ∈ {1, 3} and palette.len() is a
    // multiple of stride with at least one entry, but we guard here defensively.
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
        data: out,
        smask: None,
        filter: ImageFilter::Raw,
    })
}

/// Expand 1-bit-per-pixel packed data (MSB first) to 1 byte per pixel.
///
/// Returns `None` if `width × height` overflows `usize`.
/// Output byte: `0x00` = black (bit=0 in PDF), `0xFF` = white (bit=1 in PDF).
fn expand_1bpp(data: &[u8], width: u32, height: u32) -> Option<Vec<u8>> {
    let row_bytes = (width as usize).div_ceil(8);
    let total = (width as usize).checked_mul(height as usize)?;
    let mut out = Vec::with_capacity(total);

    for row in 0..height as usize {
        let row_start = row * row_bytes;
        let row_data = if row_start < data.len() {
            &data[row_start..data.len().min(row_start + row_bytes)]
        } else {
            &[]
        };
        for col in 0..width as usize {
            let byte_idx = col / 8;
            let bit_idx = 7 - (col % 8);
            let bit = row_data.get(byte_idx).map_or(0u8, |b| (b >> bit_idx) & 1);
            // PDF convention: bit 0 = black (0x00), bit 1 = white (0xFF)
            out.push(if bit == 0 { 0x00 } else { 0xFF });
        }
    }
    Some(out)
}

/// Unpack MSB-first packed sub-byte data (PDF spec §7.4.1) to one byte per sample.
///
/// `bits` ∈ {1, 2, 4}; each row is padded to a whole-byte boundary.
/// `samples_per_row` is the number of samples to extract per row.
/// Each raw value [0, 2^bits − 1] is transformed by `map` before being pushed.
///
/// Returns `None` if `samples_per_row × height` overflows `usize`.
fn unpack_packed_bits(
    data: &[u8],
    bits: u32,
    samples_per_row: usize,
    height: u32,
    map: impl Fn(u8, u8) -> u8, // (raw_value, mask) → output byte
) -> Option<Vec<u8>> {
    // `bits` must be 1, 2, or 4 — enforced as a hard precondition rather than
    // debug_assert so that adversarial PDF data cannot trigger a divide-by-zero
    // in release builds.
    if !matches!(bits, 1 | 2 | 4) {
        return None;
    }
    let samples_per_byte = (8 / bits) as usize;
    let row_bytes = samples_per_row.div_ceil(samples_per_byte);
    let height_usize = usize::try_from(height).ok()?;
    let total = samples_per_row.checked_mul(height_usize)?;
    let mut out = Vec::with_capacity(total);
    let mask: u8 = (1u8 << bits) - 1; // bits ≤ 4, so 1u8 << bits ≤ 16 — no overflow

    for row in 0..height_usize {
        let row_start = row.checked_mul(row_bytes)?;
        let row_data = if row_start < data.len() {
            &data[row_start..data.len().min(row_start + row_bytes)]
        } else {
            &[]
        };
        // Walk samples MSB-first: within each byte, sample 0 is at the most-significant
        // `bits` bits. e.g. bits=4, byte=[ab cd]: s=0 → shift=4, s=1 → shift=0.
        for s in 0..samples_per_row {
            let byte_idx = s / samples_per_byte;
            let shift = bits as usize * (samples_per_byte - 1 - (s % samples_per_byte));
            let byte = row_data.get(byte_idx).copied().unwrap_or(0);
            let val = (byte >> shift) & mask;
            out.push(map(val, mask));
        }
    }
    Some(out)
}

/// Expand N-bits-per-sample packed data (MSB first) to 1 byte per sample, scaled to 0–255.
///
/// `BITS` must be 2 or 4 (enforced at compile time).  Samples are scaled to the full 0–255 range:
/// - bpc 2: 4 levels → 0x00, 0x55, 0xAA, 0xFF  (value × 85)
/// - bpc 4: 16 levels → 0x00, 0x11, 0x22, …, 0xFF  (value × 17)
///
/// `components` is the number of samples per pixel (e.g. 1 for Gray, 3 for RGB).
/// Each row is padded to a whole number of source bytes (PDF spec §7.4.1).
///
/// Returns `None` if `width × height × components` overflows `usize`.
fn expand_nbpp<const BITS: u32>(
    data: &[u8],
    width: u32,
    height: u32,
    components: usize,
) -> Option<Vec<u8>> {
    const { assert!(BITS == 2 || BITS == 4, "expand_nbpp: BITS must be 2 or 4") };
    // BITS ∈ {2, 4}, so 1u8 << BITS ≤ 16 and (1u8 << BITS) - 1 ≤ 15 — no overflow.
    let max_val = (1u8 << BITS) - 1;
    // Scale factor: maps max sample value to 255.
    // bpc 2: max = 3,  scale = 85  (3 × 85 = 255)
    // bpc 4: max = 15, scale = 17  (15 × 17 = 255)
    let scale = 255u16 / u16::from(max_val);
    let width_usize = usize::try_from(width).ok()?;
    let samples_per_row = width_usize.checked_mul(components)?;
    unpack_packed_bits(data, BITS, samples_per_row, height, |val, _mask| {
        // val ≤ max_val ≤ 15; val * scale ≤ 255 — fits u8.
        #[expect(
            clippy::cast_possible_truncation,
            reason = "val ≤ 15; val*scale ≤ 255 — fits u8"
        )]
        {
            (u16::from(val) * scale) as u8
        }
    })
}

/// Expand N-bits-per-index packed Indexed-image stream (MSB first) to 1 byte per index.
///
/// Unlike `expand_nbpp`, values are NOT scaled — they are raw palette indices in
/// [0, 2^bits − 1].  `bits` ∈ {1, 2, 4}.  Rows are padded to a byte boundary.
///
/// Returns `None` if `width × height` overflows `usize`.
fn expand_nbpp_indexed(data: &[u8], width: u32, height: u32, bits: u32) -> Option<Vec<u8>> {
    let width_usize = usize::try_from(width).ok()?;
    unpack_packed_bits(data, bits, width_usize, height, |val, _mask| val)
}

/// Downsample 16-bit-per-sample data (big-endian) to 8 bits per sample.
///
/// Takes the high byte of each 16-bit sample (i.e. `sample >> 8`, discarding
/// the low byte).  This is a linear truncation: a 16-bit value of 0xFFFF maps to
/// 0xFF, 0x0100 maps to 0x01, and 0x00FF maps to 0x00.  PDF sample values are
/// in the range [0, 2^bits − 1]; dividing by 256 preserves the relative scale
/// for the common case where source data spans the full 0–65535 range.
/// `components` is the number of samples per pixel.
///
/// Returns `None` if `width × height × components` overflows `usize`, or if
/// `data` is too short for the declared image dimensions.
fn downsample_16bpp(data: &[u8], width: u32, height: u32, components: usize) -> Option<Vec<u8>> {
    let width_usize = usize::try_from(width).ok()?;
    let height_usize = usize::try_from(height).ok()?;
    let npixels = width_usize.checked_mul(height_usize)?;
    let n_samples = npixels.checked_mul(components)?;
    let needed = n_samples.checked_mul(2)?;
    if data.len() < needed {
        log::debug!(
            "image: 16bpp data too short ({} bytes, need {needed} for {width}×{height}×{components}×2)",
            data.len()
        );
        return None;
    }
    // Each 16-bit sample is big-endian; take the high byte (bytes 0, 2, 4, …).
    let out: Vec<u8> = data[..needed].chunks_exact(2).map(|pair| pair[0]).collect();
    Some(out)
}

// ── Color space helpers ───────────────────────────────────────────────────────

/// Internal resolved colour space — what the decode path will actually produce.
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

/// Resolve the `ColorSpace` entry of an image stream dictionary to one of the
/// three device spaces the blit path understands.
///
/// Handles:
/// - `DeviceGray` / `CalGray`        → Gray
/// - `DeviceRGB`  / `CalRGB` / `sRGB` → Rgb
/// - `DeviceCMYK`                    → Cmyk (caller converts to RGB)
/// - `ICCBased`   → inspect `N` in the ICC stream dict (1→Gray, 3→Rgb, 4→Cmyk)
/// - `Indexed`    → resolve the base space; Indexed expansion happens separately
/// - `Separation` / `DeviceN`        → approximate as Gray (tint 0 = full ink = dark)
/// - unknown / absent                → Gray (safe fallback)
pub(super) fn resolve_cs<'a>(doc: &'a Document, cs_obj: &'a Object) -> ResolvedCs {
    match cs_obj {
        Object::Name(n) => device_cs_name(n),
        Object::Array(arr) => {
            let name = arr.first().and_then(|o| o.as_name().ok()).unwrap_or(b"");
            match name {
                b"DeviceRGB" | b"CalRGB" => ResolvedCs::Rgb,
                b"DeviceCMYK" => ResolvedCs::Cmyk,
                b"ICCBased" => {
                    // Second element is a reference to the ICC stream.
                    let stream_dict = arr.get(1).and_then(|o| super::resolve_stream_dict(doc, o));
                    icc_based_cs(stream_dict)
                }
                b"Indexed" => {
                    // [/Indexed base hival lookup] — base is element 1.
                    arr.get(1).map_or(ResolvedCs::Gray, |o| resolve_cs(doc, o))
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
/// Returns `None` if the object is not `[/ICCBased <ref>]`, the stream cannot be
/// dereferenced, or decompression fails.  Only called under the `gpu-icc` feature.
#[cfg(feature = "gpu-icc")]
pub(super) fn extract_icc_bytes(doc: &Document, cs_obj: &Object) -> Option<Vec<u8>> {
    let arr = cs_obj.as_array().ok()?;
    if arr.first().and_then(|o| o.as_name().ok()) != Some(b"ICCBased") {
        return None;
    }
    let stream_ref = arr.get(1)?;
    let Object::Reference(id) = stream_ref else {
        return None;
    };
    let id = *id;
    let stream = doc.get_object(id).ok()?.as_stream().ok()?;
    stream
        .decompressed_content()
        .map_err(|e| log::debug!("image: ICCBased stream decompression failed: {e}"))
        .ok()
}

/// Try to decode an `Indexed` colour space lookup table into a flat RGB/Gray
/// byte palette indexed by the 1-byte pixel values (0..=hival, inclusive).
///
/// Returns `(palette, base_cs)` where `palette[i * stride .. i * stride + stride]`
/// is the colour for index `i`, and `stride` is 1 for Gray or 3 for Rgb.
///
/// Returns `None` if the array is malformed or the lookup stream cannot be read.
fn indexed_palette<'a>(doc: &'a Document, cs_arr: &'a [Object]) -> Option<(Vec<u8>, ResolvedCs)> {
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
    let hival = cs_arr[2].as_i64().ok()?.clamp(0, 255) as usize;
    let n_entries = hival + 1;

    // Lookup is either a hex/literal string or a stream reference.
    let lookup_bytes: Vec<u8> = match &cs_arr[3] {
        Object::String(bytes, _) => bytes.clone(),
        Object::Reference(id) => {
            let obj = doc.get_object(*id).ok()?;
            match obj {
                Object::Stream(s) => s.decompressed_content().ok()?,
                Object::String(bytes, _) => bytes.clone(),
                _ => return None,
            }
        }
        _ => return None,
    };

    // Validate: need at least n_entries * base.components() bytes.
    let base_stride = base.components();
    let needed = n_entries.checked_mul(base_stride)?;
    if lookup_bytes.len() < needed {
        log::debug!(
            "image: Indexed palette too short ({} bytes, need {needed})",
            lookup_bytes.len()
        );
        return None;
    }

    // Build the output palette: convert CMYK entries to RGB if needed.
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

    // ── resolve_cs (Name variants) ────────────────────────────────────────────

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

    // ── expand_1bpp ───────────────────────────────────────────────────────────

    #[test]
    fn expand_1bpp_single_byte() {
        // 0b1010_0000 → pixels [white, black, white, black, black, black, black, black].
        let data = [0b1010_0000u8];
        let out = expand_1bpp(&data, 8, 1).unwrap();
        assert_eq!(out[0], 0xFF); // bit 7 = 1 → white
        assert_eq!(out[1], 0x00); // bit 6 = 0 → black
        assert_eq!(out[2], 0xFF); // bit 5 = 1 → white
        assert_eq!(out[3], 0x00); // bit 4 = 0 → black
    }

    #[test]
    fn expand_1bpp_partial_row() {
        // width=4, 1 byte: 0b1111_0000 → 4 white pixels (only top 4 bits used).
        let data = [0b1111_0000u8];
        let out = expand_1bpp(&data, 4, 1).unwrap();
        assert_eq!(out.len(), 4);
        assert!(out.iter().all(|&b| b == 0xFF));
    }

    // ── decode_raw (Gray, Rgb) ────────────────────────────────────────────────

    #[test]
    fn decode_raw_gray_8bpp() {
        let doc = lopdf::Document::new();
        let mut dict = lopdf::Dictionary::new();
        dict.set("ColorSpace", lopdf::Object::Name(b"DeviceGray".to_vec()));
        dict.set("BitsPerComponent", lopdf::Object::Integer(8));
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
        )
        .unwrap();
        assert_eq!(desc.color_space, ImageColorSpace::Gray);
        assert_eq!(desc.data, &[0, 128, 255]);
    }

    #[test]
    fn decode_raw_rgb_8bpp() {
        let doc = lopdf::Document::new();
        let mut dict = lopdf::Dictionary::new();
        dict.set("ColorSpace", lopdf::Object::Name(b"DeviceRGB".to_vec()));
        dict.set("BitsPerComponent", lopdf::Object::Integer(8));
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
        )
        .unwrap();
        assert_eq!(desc.color_space, ImageColorSpace::Rgb);
        assert_eq!(desc.width, 3);
        assert_eq!(desc.data.len(), 9);
    }

    #[test]
    fn decode_raw_mask_ignores_cs() {
        let doc = lopdf::Document::new();
        let mut dict = lopdf::Dictionary::new();
        dict.set("BitsPerComponent", lopdf::Object::Integer(8));
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
        )
        .unwrap();
        assert_eq!(desc.color_space, ImageColorSpace::Mask);
    }

    #[test]
    fn decode_raw_cmyk_converts_to_rgb() {
        let doc = lopdf::Document::new();
        let mut dict = lopdf::Dictionary::new();
        dict.set("ColorSpace", lopdf::Object::Name(b"DeviceCMYK".to_vec()));
        dict.set("BitsPerComponent", lopdf::Object::Integer(8));
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
        )
        .unwrap();
        assert_eq!(desc.color_space, ImageColorSpace::Rgb);
        assert_eq!(desc.data, &[255, 255, 255]);
    }

    #[test]
    fn decode_raw_too_short_returns_none() {
        let doc = lopdf::Document::new();
        let mut dict = lopdf::Dictionary::new();
        dict.set("ColorSpace", lopdf::Object::Name(b"DeviceRGB".to_vec()));
        dict.set("BitsPerComponent", lopdf::Object::Integer(8));
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
                None
            )
            .is_none()
        );
    }

    // ── scale_smask ───────────────────────────────────────────────────────────

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
        // sw=0 means no source pixels; scaling into a non-empty dst would panic on
        // src index access — must return empty instead.
        let result = scale_smask(vec![], 0, 1, 4, 4);
        assert!(result.is_empty());
    }

    #[test]
    fn validated_dims_rejects_zero() {
        assert!(validated_dims(0, 100).is_none());
        assert!(validated_dims(100, 0).is_none());
    }

    #[test]
    fn validated_dims_rejects_negative() {
        assert!(validated_dims(-1, 100).is_none());
        assert!(validated_dims(100, -1).is_none());
    }

    #[test]
    fn validated_dims_rejects_oversized() {
        assert!(validated_dims(65537, 1).is_none());
        assert!(validated_dims(1, 65537).is_none());
    }

    #[test]
    fn validated_dims_accepts_boundary() {
        assert_eq!(validated_dims(1, 1), Some((1, 1)));
        assert_eq!(validated_dims(65536, 65536), Some((65536, 65536)));
    }

    // ── expand_nbpp (bpc 2 and 4) ─────────────────────────────────────────────

    #[test]
    fn expand_2bpp_all_levels() {
        // One byte = 4 samples × 2 bits, MSB first.
        // 0b11_10_01_00 = 0xE4 → values [3, 2, 1, 0] → scaled [255, 170, 85, 0]
        let data = [0b1110_0100u8];
        let out = expand_nbpp::<2>(&data, 4, 1, 1).unwrap();
        assert_eq!(out, [255, 170, 85, 0]);
    }

    #[test]
    fn expand_2bpp_single_pixel() {
        // Value 0b01 at bits 7-6 of byte → value 1 → 85.
        let data = [0b0100_0000u8];
        let out = expand_nbpp::<2>(&data, 1, 1, 1).unwrap();
        assert_eq!(out, [85]);
    }

    #[test]
    fn expand_4bpp_all_levels_two_pixels() {
        // Byte 0xFA → upper nibble 0xF=15 → 255; lower nibble 0xA=10 → 170.
        let data = [0xFAu8];
        let out = expand_nbpp::<4>(&data, 2, 1, 1).unwrap();
        assert_eq!(out, [255, 170]);
    }

    #[test]
    fn expand_4bpp_row_boundary_padding() {
        // 3 pixels at bpc=4 → 1.5 bytes → padded to 2 bytes per row.
        // Byte 0: pixels 0=0xA(170), 1=0x5(85). Byte 1: pixel 2=0x0(0), padding nibble ignored.
        let data = [0xA5u8, 0x00u8];
        let out = expand_nbpp::<4>(&data, 3, 1, 1).unwrap();
        assert_eq!(out, [170, 85, 0]);
    }

    #[test]
    fn expand_2bpp_multi_row_padding() {
        // 3 pixels × 2 bpc = 6 bits → padded to 1 byte per row.
        // Row 0: byte 0b11_10_01_xx → values [3, 2, 1] → [255, 170, 85], 2 pad bits ignored.
        // Row 1: byte 0b00_01_10_xx → values [0, 1, 2] → [0, 85, 170].
        let data = [0b1110_0100u8, 0b0001_1000u8];
        let out = expand_nbpp::<2>(&data, 3, 2, 1).unwrap();
        assert_eq!(out, [255, 170, 85, 0, 85, 170]);
    }

    // ── expand_nbpp_indexed (bpc 1, 2, 4 for Indexed images) ─────────────────

    #[test]
    fn expand_nbpp_indexed_4bpp() {
        // Byte 0xAF → upper nibble = index 10, lower nibble = index 15.
        let out = expand_nbpp_indexed(&[0xAFu8], 2, 1, 4).unwrap();
        assert_eq!(out, [10, 15]);
    }

    #[test]
    fn expand_nbpp_indexed_2bpp() {
        // Byte 0b11_10_01_00 = 0xE4 → indices [3, 2, 1, 0].
        let out = expand_nbpp_indexed(&[0xE4u8], 4, 1, 2).unwrap();
        assert_eq!(out, [3, 2, 1, 0]);
    }

    #[test]
    fn expand_nbpp_indexed_1bpp() {
        // Byte 0b1010_0000 → indices [1, 0, 1, 0, 0, 0, 0, 0] (8 pixels).
        let out = expand_nbpp_indexed(&[0b1010_0000u8], 8, 1, 1).unwrap();
        assert_eq!(out, [1, 0, 1, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn expand_nbpp_indexed_short_input_zero_pads() {
        // Empty data → all zeros (palette index 0 = first entry).
        let out = expand_nbpp_indexed(&[], 4, 1, 2).unwrap();
        assert_eq!(out, [0, 0, 0, 0]);
    }

    // ── downsample_16bpp ──────────────────────────────────────────────────────

    #[test]
    fn downsample_16bpp_takes_high_byte() {
        // Two 16-bit big-endian samples: 0xABCD → 0xAB, 0x1234 → 0x12.
        let data = [0xABu8, 0xCD, 0x12, 0x34];
        let out = downsample_16bpp(&data, 2, 1, 1).unwrap();
        assert_eq!(out, [0xAB, 0x12]);
    }

    #[test]
    fn downsample_16bpp_max_is_255() {
        // 0xFFFF → high byte 0xFF; 0x0000 → 0x00.
        let data = [0xFFu8, 0xFF, 0x00, 0x00];
        let out = downsample_16bpp(&data, 2, 1, 1).unwrap();
        assert_eq!(out, [0xFF, 0x00]);
    }

    #[test]
    fn downsample_16bpp_short_input_returns_none() {
        // 1 pixel RGB 16bpp needs 6 bytes; 4 bytes is too short.
        assert!(downsample_16bpp(&[0u8; 4], 1, 1, 3).is_none());
    }
}
