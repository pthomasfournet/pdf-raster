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
//! | `DCTDecode` (JPEG) | no | yes (via `zune-jpeg`, or GPU nvJPEG) |
//! | `JPXDecode` (JPEG 2000) | no | yes (via `jpeg2k`/`OpenJPEG`, or GPU nvJPEG2000) |
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
//! nvJPEG instead of `zune-jpeg`.  Pass an [`NvJpegDecoder`] to
//! [`resolve_image`] to enable this path; pass `None` for CPU-only behaviour.
//!
//! # nvJPEG2000 acceleration (`nvjpeg2k` feature)
//!
//! When the crate is built with `--features nvjpeg2k`, `JPXDecode` streams
//! with pixel area ≥ [`GPU_JPEG2K_THRESHOLD_PX`] are decoded on the GPU via
//! NVIDIA nvJPEG2000 instead of `jpeg2k`/`OpenJPEG`.  Pass an
//! [`NvJpeg2kDecoder`] to [`resolve_image`] to enable this path; pass `None`
//! for CPU-only behaviour.  Only 1- and 3-component images are accelerated;
//! CMYK and other multi-channel images always fall through to `OpenJPEG`.

use std::borrow::Cow;

use hayro_jbig2::Decoder as Jbig2Decoder;
use jpeg2k::{Image as Jp2Image, ImageFormat, ImagePixelData};
use lopdf::{Dictionary, Document, Object, ObjectId};

use crate::resources::dict_ext::DictExt;
use zune_core::bytestream::ZCursor;
use zune_core::colorspace::ColorSpace as ZColorSpace;
use zune_core::options::DecoderOptions;
use zune_jpeg::JpegDecoder;

// ── nvJPEG GPU acceleration ───────────────────────────────────────────────────

#[cfg(feature = "nvjpeg")]
use gpu::nvjpeg::{JpegColorSpace as GpuCs, NvJpegDecoder};

#[cfg(feature = "nvjpeg2k")]
use gpu::nvjpeg2k::{Jpeg2kColorSpace as GpuJ2kCs, NvJpeg2kDecoder};

// ── GPU ICC CMYK→RGB acceleration ─────────────────────────────────────────────

#[cfg(feature = "gpu-icc")]
use gpu::GpuCtx;

#[cfg(feature = "gpu-icc")]
#[path = "icc.rs"]
mod icc;

/// Minimum pixel area (width × height) for GPU-accelerated `DCTDecode`.
///
/// Below this threshold `PCIe` transfer overhead dominates and CPU `zune-jpeg`
/// is faster.  512 × 512 = 262 144 pixels — empirically the crossover between
/// nvJPEG (~10 GB/s) and `zune-jpeg` (~1 GB/s) after `PCIe` DMA latency.
#[cfg(feature = "nvjpeg")]
pub const GPU_JPEG_THRESHOLD_PX: u32 = 262_144;

/// Minimum pixel area (width × height) for GPU-accelerated `JPXDecode`.
///
/// Below this threshold `PCIe` transfer overhead dominates and CPU
/// `jpeg2k`/`OpenJPEG` is faster.  512 × 512 = 262 144 pixels — same crossover
/// as nvJPEG; JPEG 2000 decode is CPU-bound at similar pixel counts.
#[cfg(feature = "nvjpeg2k")]
pub const GPU_JPEG2K_THRESHOLD_PX: u32 = 262_144;

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
    #[cfg(feature = "nvjpeg2k")] gpu_j2k: Option<&mut NvJpeg2kDecoder>,
    #[cfg(feature = "gpu-icc")] gpu_ctx: Option<&GpuCtx>,
) -> Option<ImageDescriptor> {
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
        Some("DCTDecode") => {
            #[cfg(all(feature = "nvjpeg", feature = "gpu-icc"))]
            {
                decode_dct(stream.content.as_slice(), w, h, gpu, gpu_ctx)
            }
            #[cfg(all(feature = "nvjpeg", not(feature = "gpu-icc")))]
            {
                decode_dct(stream.content.as_slice(), w, h, gpu)
            }
            #[cfg(all(not(feature = "nvjpeg"), feature = "gpu-icc"))]
            {
                decode_dct(stream.content.as_slice(), w, h, gpu_ctx)
            }
            #[cfg(all(not(feature = "nvjpeg"), not(feature = "gpu-icc")))]
            {
                decode_dct(stream.content.as_slice(), w, h)
            }
        }
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
            log::debug!("image: skipping image — SMask (object {smask_id:?}) could not be decoded");
            return None;
        }
    }

    Some(img)
}

// ── Inline image decoding (BI … ID … EI) ─────────────────────────────────────

/// Decode an inline image (`BI … ID … EI`) from raw parameter and data bytes.
///
/// `params` is the raw content between `BI` and `ID`; it contains key/value
/// pairs using either the full PDF names (`/Width`, `/ColorSpace`, …) or the
/// abbreviated forms defined in PDF §8.9.7 Table 89 (`/W`, `/CS`, …).
/// `data` is the raw content between `ID` and `EI` (compressed or raw pixels).
///
/// Inline images cannot carry an `SMask` stream (they have no object identity),
/// so the returned descriptor always has `smask: None`.
///
/// Returns `None` if the parameter block is unparseable, dimensions are
/// degenerate, or the filter is unsupported.
#[must_use]
#[expect(
    clippy::too_many_lines,
    reason = "one match arm per supported filter; each arm is small but the set is large"
)]
pub fn decode_inline_image(doc: &Document, params: &[u8], data: &[u8]) -> Option<ImageDescriptor> {
    let dict = parse_inline_params(params);

    let w_raw = dict.get_i64(b"Width")?;
    let h_raw = dict.get_i64(b"Height")?;
    let Some((w, h)) = validated_dims(w_raw, h_raw) else {
        log::warn!("inline image: degenerate dimensions {w_raw}×{h_raw}, skipping");
        return None;
    };

    let is_mask = dict.get_bool(b"ImageMask").unwrap_or(false);
    let filter = dict.get(b"Filter").ok().and_then(filter_name);

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
            data,
            w,
            h,
            is_mask,
            &dict,
            #[cfg(feature = "gpu-icc")]
            None,
        ),
        Some("FlateDecode") => {
            use lopdf::Stream;
            // Use lopdf to run Flate decompression on raw bytes.
            let stream = Stream::new(dict.clone(), data.to_vec());
            match stream.decompressed_content() {
                Ok(raw) => decode_raw(
                    doc,
                    &raw,
                    w,
                    h,
                    is_mask,
                    &dict,
                    #[cfg(feature = "gpu-icc")]
                    None,
                ),
                Err(e) => {
                    log::warn!("inline image: FlateDecode failed: {e}");
                    None
                }
            }
        }
        Some("CCITTFaxDecode") => {
            let decode_parms = dict.get(b"DecodeParms").ok();
            decode_ccitt(data, w, h, is_mask, decode_parms)
        }
        Some("DCTDecode") => {
            // Inline images are typically small (embedded in the content stream).
            // GPU dispatch for inline images is not worthwhile — use CPU always.
            #[cfg(all(feature = "nvjpeg", feature = "gpu-icc"))]
            {
                decode_dct(data, w, h, None, None)
            }
            #[cfg(all(feature = "nvjpeg", not(feature = "gpu-icc")))]
            {
                decode_dct(data, w, h, None)
            }
            #[cfg(all(not(feature = "nvjpeg"), feature = "gpu-icc"))]
            {
                decode_dct(data, w, h, None)
            }
            #[cfg(all(not(feature = "nvjpeg"), not(feature = "gpu-icc")))]
            {
                decode_dct(data, w, h)
            }
        }
        Some("JPXDecode") => {
            // Inline images are typically small — GPU dispatch is not worthwhile.
            #[cfg(feature = "nvjpeg2k")]
            {
                decode_jpx(data, w, h, None)
            }
            #[cfg(not(feature = "nvjpeg2k"))]
            {
                decode_jpx(data, w, h)
            }
        }
        Some("JBIG2Decode") => {
            // Inline images cannot reference a JBIG2Globals stream (no object identity).
            decode_jbig2(doc, data, w, h, is_mask, None)
        }
        Some("RunLengthDecode") => {
            let raw = decode_run_length(data);
            decode_raw(
                doc,
                &raw,
                w,
                h,
                is_mask,
                &dict,
                #[cfg(feature = "gpu-icc")]
                None,
            )
        }
        Some(other) => {
            log::warn!("inline image: unknown filter {other:?}");
            None
        }
    }?;
    img.filter = img_filter;
    Some(img)
}

/// Decode a `RunLengthDecode` stream (PDF §7.4.5).
///
/// The encoding is a simple run-length scheme:
/// - Byte `n` in [0, 127] → copy the next `n + 1` literal bytes.
/// - Byte `n` in [129, 255] → repeat the next byte `257 − n` times.
/// - Byte `128` → end-of-data marker.
///
/// Output is capped at 256 MiB to prevent adversarial OOM from a crafted stream.
fn decode_run_length(data: &[u8]) -> Vec<u8> {
    // 256 MiB: enough for a 65536×65536 single-channel image, hard limit against
    // adversarial run-length streams that encode billions of bytes from a few bytes.
    decode_run_length_capped(data, 256 * 1024 * 1024)
}

fn decode_run_length_capped(data: &[u8], max_output: usize) -> Vec<u8> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < data.len() {
        let run_byte = data[i];
        i += 1;
        match run_byte {
            128 => break, // EOD
            0..=127 => {
                let count = run_byte as usize + 1;
                let end = i.saturating_add(count).min(data.len());
                if out.len() + (end - i) > max_output {
                    log::warn!("inline image: RunLengthDecode output exceeds limit — truncating");
                    break;
                }
                out.extend_from_slice(&data[i..end]);
                i = end;
            }
            _ => {
                // 129..=255: repeat = 257 - run_byte
                let repeat = 257usize.saturating_sub(run_byte as usize);
                if let Some(&b) = data.get(i) {
                    if out.len() + repeat > max_output {
                        log::warn!(
                            "inline image: RunLengthDecode output exceeds limit — truncating"
                        );
                        break;
                    }
                    out.extend(std::iter::repeat_n(b, repeat));
                    i += 1;
                }
            }
        }
    }
    out
}

/// Parse the raw inline-image parameter block into a `lopdf::Dictionary`.
///
/// Keys and values are in PDF syntax but may use the abbreviated forms from
/// PDF §8.9.7 Table 89 (e.g., `/W` → `/Width`, `/Fl` → `/FlateDecode`).
/// Unknown keys are passed through unchanged; unrecognised values are passed
/// through unchanged — the caller's `DictExt` helpers return `None` for
/// malformed entries, so no validation is needed here.
fn parse_inline_params(params: &[u8]) -> Dictionary {
    // Tokenise the parameter block with the content-stream tokenizer and
    // collect alternating key/value pairs.
    use crate::content::tokenizer::{Token, Tokenizer};

    let mut dict = Dictionary::new();
    let tokens: Vec<Token<'_>> = Tokenizer::new(params).collect();
    let mut i = 0;

    while i + 1 < tokens.len() {
        let Token::Name(key_bytes) = &tokens[i] else {
            i += 1;
            continue;
        };
        // Expand abbreviated key name → canonical PDF key.
        let canonical_key: &[u8] = expand_inline_key(key_bytes);
        let canonical_key_vec = canonical_key.to_vec();

        let value_obj = inline_token_to_object(&tokens[i + 1]);
        // Expand abbreviated filter/cs names inside the value.
        let value_obj = expand_inline_value(value_obj);

        dict.set(canonical_key_vec, value_obj);
        i += 2;
    }

    dict
}

/// Map an abbreviated inline-image key to its canonical PDF name.
///
/// Abbreviations are defined in PDF §8.9.7 Table 89.  Unknown keys are
/// returned unchanged (as a byte slice borrow).
const fn expand_inline_key(key: &[u8]) -> &[u8] {
    match key {
        b"W" => b"Width",
        b"H" => b"Height",
        b"CS" => b"ColorSpace",
        b"BPC" => b"BitsPerComponent",
        b"F" => b"Filter",
        b"DP" => b"DecodeParms",
        b"IM" => b"ImageMask",
        b"D" => b"Decode",
        b"I" => b"Interpolate",
        b"Intent" => b"Intent",
        other => other,
    }
}

/// Convert a content-stream [`Token`] to a `lopdf::Object`.
fn inline_token_to_object(tok: &crate::content::tokenizer::Token<'_>) -> Object {
    use crate::content::tokenizer::Token;
    match tok {
        Token::Name(n) => Object::Name(n.to_vec()),
        Token::Number(f) => {
            // Store as Integer when the value is a whole number, Real otherwise.
            // Upper bound 1e15 safely fits in i64 without f64 precision issues.
            let rounded = f.round();
            if (f - rounded).abs() < f64::EPSILON && rounded.abs() < 1e15 {
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "checked: value is a whole number with abs < 1e15; fits in i64"
                )]
                Object::Integer(rounded as i64)
            } else {
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "f64 → f32: PDF numeric params are small; precision loss is acceptable"
                )]
                Object::Real(*f as f32)
            }
        }
        Token::Bool(b) => Object::Boolean(*b),
        Token::String(s) => Object::String(s.clone(), lopdf::StringFormat::Literal),
        Token::Array(items) => Object::Array(items.iter().map(inline_token_to_object).collect()),
        _ => Object::Null,
    }
}

/// Expand abbreviated filter or colour-space names inside an `Object`.
///
/// Affects `Name` objects and arrays of names (filter arrays).  The canonical
/// names are the full PDF spelling; abbreviations come from PDF §8.9.7 Table 89
/// (for filters) and the colour-space name list.
fn expand_inline_value(obj: Object) -> Object {
    match obj {
        Object::Name(ref n) => {
            let expanded = expand_inline_name(n);
            if expanded == n.as_slice() {
                obj
            } else {
                Object::Name(expanded.to_vec())
            }
        }
        Object::Array(arr) => Object::Array(
            arr.into_iter()
                .map(|o| match o {
                    Object::Name(n) => Object::Name(expand_inline_name(&n).to_vec()),
                    other => other,
                })
                .collect(),
        ),
        other => other,
    }
}

/// Expand an abbreviated inline-image name (filter or colour-space).
const fn expand_inline_name(name: &[u8]) -> &[u8] {
    match name {
        // Filter abbreviations (PDF §8.9.7 Table 89).
        b"AHx" => b"ASCIIHexDecode",
        b"A85" => b"ASCII85Decode",
        b"LZW" => b"LZWDecode",
        b"Fl" => b"FlateDecode",
        b"RL" => b"RunLengthDecode",
        b"CCF" => b"CCITTFaxDecode",
        b"DCT" => b"DCTDecode",
        // Colour-space abbreviations.
        b"G" => b"DeviceGray",
        b"RGB" => b"DeviceRGB",
        b"CMYK" => b"DeviceCMYK",
        b"I" => b"Indexed",
        other => other,
    }
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
    for row in raw.chunks(row_bytes) {
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
    let mut out = Vec::with_capacity(usize::try_from(dst_w * dst_h).unwrap_or(0));
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
const fn validated_dims(w_raw: i64, h_raw: i64) -> Option<(u32, u32)> {
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
fn filter_name(obj: &Object) -> Option<Cow<'_, str>> {
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
fn decode_raw(
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
    debug_assert!(
        bits == 1 || bits == 2 || bits == 4,
        "unpack_packed_bits: bits must be 1, 2, or 4"
    );
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
    debug_assert!(
        bits == 1 || bits == 2 || bits == 4,
        "expand_nbpp_indexed: bits must be 1, 2, or 4"
    );
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

// ── CCITTFaxDecode ─────────────────────────────────────────────────────────────

/// Decode a `CCITTFaxDecode` stream.
///
/// `K` in `DecodeParms` (PDF §7.4.6):
/// - `K < 0` → Group 4 (T.6, 2D) — fully supported.
/// - `K = 0` → Group 3 1D (T.4 1D) — supported.
/// - `K > 0` → Group 3 mixed 1D/2D (T.4 2D) — not yet implemented.
///
/// `Rows` (if present) caps the number of rows decoded; otherwise decodes
/// until the bitstream signals end-of-data.
fn decode_ccitt(
    data: &[u8],
    width: u32,
    height: u32,
    is_mask: bool,
    parms: Option<&Object>,
) -> Option<ImageDescriptor> {
    // Resolve DecodeParms once; all CCITT params live in the same dict.
    let parms_dict = parms.and_then(|o| o.as_dict().ok());

    let k = parms_dict.and_then(|d| d.get_i64(b"K")).unwrap_or(0);

    // BlackIs1: when true, 1-bits encode black (reversed from the T.4/T.6 default).
    let black_is_1 = parms_dict
        .and_then(|d| d.get_bool(b"BlackIs1"))
        .unwrap_or(false);

    // Rows: optional override for the number of rows in the stream.  When present
    // and smaller than height, we stop early; when absent we decode until EOD and
    // pad any missing rows with white.
    let rows_limit = parms_dict
        .and_then(|d| d.get_i64(b"Rows"))
        .and_then(|r| u32::try_from(r).ok())
        .unwrap_or(height);

    let w_u16 = u16::try_from(width).ok()?;
    let capacity = (width as usize).checked_mul(height as usize)?;
    let p = CcittParams {
        w_u16,
        capacity,
        width,
        height,
        is_mask,
        black_is_1,
    };

    match k.cmp(&0) {
        std::cmp::Ordering::Less => decode_ccitt_g4(data, &p),
        std::cmp::Ordering::Equal => decode_ccitt_g3_1d(data, &p, rows_limit),
        std::cmp::Ordering::Greater => {
            // K > 0: Group 3 mixed 1D/2D (T.4 2D, "MR").
            // The fax 0.2.x crate only decodes 1D; use hayro-ccitt for the mixed path.
            let k_u32 = u32::try_from(k).ok()?;
            decode_ccitt_g3_2d(data, &p, rows_limit, k_u32)
        }
    }
}

/// Shared parameters for the CCITT decode helpers.
struct CcittParams {
    w_u16: u16,
    capacity: usize,
    width: u32,
    height: u32,
    is_mask: bool,
    black_is_1: bool,
}

/// Decode a Group 4 (K<0, T.6) CCITT fax stream.
fn decode_ccitt_g4(data: &[u8], p: &CcittParams) -> Option<ImageDescriptor> {
    let h_u16 = u16::try_from(p.height).ok()?;
    let mut data_out: Vec<u8> = Vec::with_capacity(p.capacity);
    let mut rows_decoded: u32 = 0;

    let completed =
        fax::decoder::decode_g4(data.iter().copied(), p.w_u16, Some(h_u16), |transitions| {
            append_ccitt_row(&mut data_out, transitions, p.w_u16, p.width, p.black_is_1);
            rows_decoded += 1;
        });

    if completed.is_none() {
        log::warn!(
            "image: CCITTFaxDecode Group4 decode incomplete — got {rows_decoded}/{} rows",
            p.height
        );
        if rows_decoded == 0 {
            return None;
        }
        data_out.resize(p.capacity, 0xFF);
    }

    let cs = if p.is_mask {
        ImageColorSpace::Mask
    } else {
        ImageColorSpace::Gray
    };
    Some(ImageDescriptor {
        width: p.width,
        height: p.height,
        color_space: cs,
        data: data_out,
        smask: None,
        filter: ImageFilter::Raw,
    })
}

/// Decode a Group 3 1D (K=0, T.4) CCITT fax stream.
///
/// `rows_limit` is the maximum number of rows to emit (from `DecodeParms/Rows`
/// or `height` when absent).  Extra rows in the stream are discarded; missing
/// rows are padded with white.
fn decode_ccitt_g3_1d(data: &[u8], p: &CcittParams, rows_limit: u32) -> Option<ImageDescriptor> {
    let mut data_out: Vec<u8> = Vec::with_capacity(p.capacity);
    let mut rows_decoded: u32 = 0;

    // decode_g3 fires the callback once per decoded row (after each EOL).
    // It returns Some(()) on clean EOD, None on bitstream error.
    let result = fax::decoder::decode_g3(data.iter().copied(), |transitions| {
        if rows_decoded >= rows_limit {
            return; // discard extra rows beyond the declared height
        }
        append_ccitt_row(&mut data_out, transitions, p.w_u16, p.width, p.black_is_1);
        rows_decoded += 1;
    });

    if result.is_none() && rows_decoded == 0 {
        log::warn!("image: CCITTFaxDecode Group3 1D decode failed with no rows");
        return None;
    }
    if rows_decoded < p.height {
        log::debug!(
            "image: CCITTFaxDecode Group3 1D: got {rows_decoded}/{} rows — padding remainder",
            p.height
        );
        data_out.resize(p.capacity, 0xFF);
    }

    let cs = if p.is_mask {
        ImageColorSpace::Mask
    } else {
        ImageColorSpace::Gray
    };
    Some(ImageDescriptor {
        width: p.width,
        height: p.height,
        color_space: cs,
        data: data_out,
        smask: None,
        filter: ImageFilter::Raw,
    })
}

/// Decode a Group 3 mixed 1D/2D (K>0, T.4 2D / "MR") CCITT fax stream.
///
/// Uses `hayro-ccitt` which natively supports `EncodingMode::Group3_2D { k }`.
/// `rows_limit` caps the number of rows to emit (from `DecodeParms/Rows` or `height`).
fn decode_ccitt_g3_2d(
    data: &[u8],
    p: &CcittParams,
    rows_limit: u32,
    k: u32,
) -> Option<ImageDescriptor> {
    use hayro_ccitt::{DecodeSettings, DecoderContext, EncodingMode};

    // EncodingMode::Group3_1D/Group3_2D expect `end_of_line = true` for T.4.
    // `end_of_block` tells the decoder it may encounter a 6-EOL RTC marker.
    // `rows_are_byte_aligned` = false: T.4 mixed streams are NOT byte-aligned
    // between rows (only the EOL acts as a row separator).
    let settings = DecodeSettings {
        columns: p.width,
        rows: rows_limit,
        end_of_block: true,
        end_of_line: true,
        rows_are_byte_aligned: false,
        encoding: EncodingMode::Group3_2D { k },
        invert_black: p.black_is_1,
    };
    let mut ctx = DecoderContext::new(settings);
    let mut collector = HayroCcittCollector::new(p.capacity, p.width);

    match hayro_ccitt::decode(data, &mut collector, &mut ctx) {
        Ok(_) => {}
        Err(e) => {
            if collector.rows_decoded() == 0 {
                log::warn!("image: CCITTFaxDecode Group3 2D decode failed: {e}");
                return None;
            }
            log::debug!(
                "image: CCITTFaxDecode Group3 2D partial decode ({}/{} rows): {e}",
                collector.rows_decoded(),
                p.height
            );
        }
    }

    let rows_decoded = collector.rows_decoded();
    let mut data_out = collector.finish();
    // Truncate overlong output (malformed stream that emitted too many pixels)
    // then pad short output (truncated stream); both produce exactly p.capacity bytes.
    if data_out.len() > p.capacity {
        log::debug!(
            "image: CCITTFaxDecode Group3 2D: output too long ({} > {}), truncating",
            data_out.len(),
            p.capacity
        );
        data_out.truncate(p.capacity);
    } else if data_out.len() < p.capacity {
        log::debug!(
            "image: CCITTFaxDecode Group3 2D: got {rows_decoded}/{} rows — padding remainder",
            p.height
        );
        data_out.resize(p.capacity, 0xFF);
    }

    let cs = if p.is_mask {
        ImageColorSpace::Mask
    } else {
        ImageColorSpace::Gray
    };
    Some(ImageDescriptor {
        width: p.width,
        height: p.height,
        color_space: cs,
        data: data_out,
        smask: None,
        filter: ImageFilter::Raw,
    })
}

/// Accumulates `hayro-ccitt` decoded pixels into a `Vec<u8>` (one byte per pixel,
/// 0x00 = black, 0xFF = white).  Incomplete final rows are padded to `width`.
struct HayroCcittCollector {
    out: Vec<u8>,
    width: u32,
    col: u32,
    rows: u32,
}

impl HayroCcittCollector {
    fn new(capacity: usize, width: u32) -> Self {
        Self {
            out: Vec::with_capacity(capacity),
            width,
            col: 0,
            rows: 0,
        }
    }

    const fn rows_decoded(&self) -> u32 {
        self.rows
    }

    fn finish(mut self) -> Vec<u8> {
        // Pad any partial final row (malformed stream that ended mid-row).
        if self.col > 0 {
            // self.col < self.width is the invariant maintained by push_pixel /
            // push_pixel_chunk; subtraction is therefore safe.
            let remaining = usize::try_from(self.width - self.col).unwrap_or(0);
            self.out.extend(std::iter::repeat_n(0xFFu8, remaining));
        }
        self.out
    }
}

impl hayro_ccitt::Decoder for HayroCcittCollector {
    fn push_pixel(&mut self, white: bool) {
        // Guard against over-wide rows from malformed bitstreams.
        if self.col < self.width {
            self.out.push(if white { 0xFF } else { 0x00 });
            self.col += 1;
        }
    }

    fn push_pixel_chunk(&mut self, white: bool, chunk_count: u32) {
        // chunk_count × 8 pixels, all the same colour.
        let total = chunk_count.saturating_mul(8);
        let available = self.width.saturating_sub(self.col);
        let n = total.min(available);
        let byte = if white { 0xFFu8 } else { 0x00u8 };
        self.out.extend(std::iter::repeat_n(byte, n as usize));
        self.col += n;
    }

    fn next_line(&mut self) {
        // Pad any short row produced by a malformed or truncated bitstream.
        if self.col < self.width {
            let remaining = usize::try_from(self.width - self.col).unwrap_or(0);
            self.out.extend(std::iter::repeat_n(0xFFu8, remaining));
        }
        self.col = 0;
        self.rows += 1;
    }
}

/// Expand one row of CCITT transitions into bytes and append to `out`.
///
/// The row is padded/truncated to exactly `width` bytes.
/// `0x00` = black, `0xFF` = white (PDF image convention).
fn append_ccitt_row(
    out: &mut Vec<u8>,
    transitions: &[u16],
    w_u16: u16,
    width: u32,
    black_is_1: bool,
) {
    let row_start = out.len();
    out.extend(fax::decoder::pels(transitions, w_u16).map(|color| {
        let is_black = match color {
            fax::Color::Black => !black_is_1,
            fax::Color::White => black_is_1,
        };
        if is_black { 0x00u8 } else { 0xFFu8 }
    }));
    // Pad short rows (malformed data) with white.
    let row_end = row_start + width as usize;
    out.resize(row_end, 0xFF);
}

// ── DCTDecode (JPEG) ──────────────────────────────────────────────────────────

/// Decode a `DCTDecode` (JPEG) stream.
///
/// `pdf_w` and `pdf_h` from the PDF stream dict are used only for a size sanity
/// check; the actual dimensions come from the JPEG SOF marker and are
/// authoritative.
///
/// # CMYK handling
///
/// JPEG CMYK images in PDF store ink densities in *inverted* form: a byte value
/// of 0 means *full ink*, 255 means *no ink* (the complement of the usual
/// convention).  `zune-jpeg` returns raw bytes in this form.  We convert to RGB
/// using:
///
/// ```text
/// R = (255 - C) * (255 - K) / 255
/// ```
///
/// where `C`, `K` are the complemented (i.e. raw) CMYK byte values.  This is
/// equivalent to the standard CMY+K → RGB conversion applied to the inverted
/// ink densities.
fn decode_dct(
    data: &[u8],
    pdf_w: u32,
    pdf_h: u32,
    #[cfg(feature = "nvjpeg")] gpu: Option<&mut NvJpegDecoder>,
    #[cfg(feature = "gpu-icc")] gpu_ctx: Option<&GpuCtx>,
) -> Option<ImageDescriptor> {
    // ── GPU fast path (nvjpeg feature, large images, 1- or 3-component only) ──
    #[cfg(feature = "nvjpeg")]
    if let Some(dec) = gpu {
        // Only dispatch to GPU when the image area meets the threshold.
        // CMYK JPEG (4 components) is not supported by the nvJPEG RGBI/Y path;
        // decode_dct_gpu returns None for 4-component images, which causes
        // the fall-through to the CPU path below that handles CMYK→RGB.
        let area = pdf_w.saturating_mul(pdf_h);
        if area >= GPU_JPEG_THRESHOLD_PX {
            if let Some(img) = decode_dct_gpu(data, pdf_w, pdf_h, dec) {
                return Some(img);
            }
            // GPU decode failed (e.g. unsupported encoding) — fall through to CPU.
            log::debug!("image: DCTDecode: GPU path failed, retrying on CPU");
        }
    }

    // ── CPU path (zune-jpeg) ──────────────────────────────────────────────────
    // First pass: decode only the JPEG headers to learn the component count,
    // which determines the output colourspace we request on the second decode.
    // Two passes are necessary because zune-jpeg requires the output colourspace
    // to be set before `decode()` is called.
    let mut probe = JpegDecoder::new(ZCursor::new(data));
    probe.decode_headers().ok()?;
    let components = probe.info()?.components;

    // Choose the output colorspace zune-jpeg should produce.
    let out_cs = match components {
        1 => ZColorSpace::Luma,
        3 => ZColorSpace::RGB,
        // CMYK — request raw CMYK output (4 bytes/pixel), convert to RGB below.
        4 => ZColorSpace::CMYK,
        n => {
            log::warn!("image: DCTDecode: unexpected component count {n}");
            return None;
        }
    };

    let options = DecoderOptions::default().jpeg_set_out_colorspace(out_cs);
    let mut decoder = JpegDecoder::new_with_options(ZCursor::new(data), options);
    let pixels = decoder
        .decode()
        .map_err(|e| log::warn!("image: DCTDecode decode error: {e}"))
        .ok()?;

    let jpeg_info = decoder.info()?;
    let jw = u32::from(jpeg_info.width);
    let jh = u32::from(jpeg_info.height);

    if jw == 0 || jh == 0 {
        log::warn!("image: DCTDecode: JPEG reported zero dimensions {jw}×{jh}");
        return None;
    }

    if jw != pdf_w || jh != pdf_h {
        log::debug!(
            "image: DCTDecode: PDF dict says {pdf_w}×{pdf_h}, JPEG reports {jw}×{jh} — using JPEG dims"
        );
    }

    match out_cs {
        ZColorSpace::Luma => Some(ImageDescriptor {
            width: jw,
            height: jh,
            color_space: ImageColorSpace::Gray,
            data: pixels,
            smask: None,
            filter: ImageFilter::Raw,
        }),
        ZColorSpace::RGB => Some(ImageDescriptor {
            width: jw,
            height: jh,
            color_space: ImageColorSpace::Rgb,
            data: pixels,
            smask: None,
            filter: ImageFilter::Raw,
        }),
        ZColorSpace::CMYK => {
            // zune-jpeg returns JPEG CMYK with inverted convention (0=full ink, 255=no ink).
            // Complement to direct convention (255=full ink, 0=no ink) before dispatch.
            let direct: Vec<u8> = pixels.iter().map(|&b| 255 - b).collect();
            // JPEG streams embed their own colour profile; the PDF ICCBased stream
            // is not available here, so ICC CLUT baking is not performed for DCT.
            let rgb = cmyk_raw_to_rgb(
                &direct,
                #[cfg(feature = "gpu-icc")]
                gpu_ctx,
                #[cfg(feature = "gpu-icc")]
                None,
            )?;
            Some(ImageDescriptor {
                width: jw,
                height: jh,
                color_space: ImageColorSpace::Rgb,
                data: rgb,
                smask: None,
                filter: ImageFilter::Raw,
            })
        }
        // out_cs is always Luma, RGB, or CMYK — set from the components match above.
        _ => unreachable!("DCTDecode: unexpected out_cs variant"),
    }
}

/// GPU-accelerated DCT decode via nvJPEG.
///
/// Returns `None` if the component count is unsupported (e.g. CMYK, which
/// nvJPEG cannot output as RGBI) or if any CUDA API call fails.  The caller
/// must fall back to the CPU path when `None` is returned.
///
/// The stream is synchronised inside `NvJpegDecoder::decode_sync` before
/// pixel bytes are returned, so the result is safe to use immediately.
#[cfg(feature = "nvjpeg")]
fn decode_dct_gpu(
    data: &[u8],
    pdf_w: u32,
    pdf_h: u32,
    dec: &mut NvJpegDecoder,
) -> Option<ImageDescriptor> {
    let img = match dec.decode_sync(data) {
        Ok(img) => img,
        Err(gpu::nvjpeg::NvJpegError::UnsupportedComponents(_)) => {
            // CMYK or other unsupported component count — expected, fall back silently.
            return None;
        }
        Err(e) => {
            // Unexpected CUDA/nvJPEG failure — log at warn so it's visible.
            log::warn!("image: DCTDecode GPU: unexpected nvJPEG error: {e}");
            return None;
        }
    };

    if img.width != pdf_w || img.height != pdf_h {
        log::debug!(
            "image: DCTDecode GPU: PDF dict says {pdf_w}×{pdf_h}, nvJPEG reports {}×{} — using nvJPEG dims",
            img.width,
            img.height,
        );
    }

    let color_space = match img.color_space {
        GpuCs::Gray => ImageColorSpace::Gray,
        GpuCs::Rgb => ImageColorSpace::Rgb,
    };

    Some(ImageDescriptor {
        width: img.width,
        height: img.height,
        color_space,
        data: img.data,
        smask: None,
        filter: ImageFilter::Raw,
    })
}

/// Convert a raw CMYK pixel buffer to RGB, dispatching to GPU when available.
///
/// Input convention (raw images / `decode_raw_8bpp`): 0 = no ink, 255 = full ink.
/// This matches the `GpuCtx::icc_cmyk_to_rgb` matrix kernel directly.
///
/// `icc_bytes` — raw ICC profile bytes extracted from an `ICCBased` colour space.
/// When provided (and the `gpu-icc` feature is active), a CLUT is baked from the
/// profile and used for the colour transform instead of the fast matrix approximation.
///
/// Returns `None` only on arithmetic overflow (degenerate image size).
fn cmyk_raw_to_rgb(
    pixels: &[u8],
    #[cfg(feature = "gpu-icc")] gpu_ctx: Option<&GpuCtx>,
    #[cfg(feature = "gpu-icc")] icc_bytes: Option<&[u8]>,
) -> Option<Vec<u8>> {
    // GPU path: delegate to GpuCtx which handles the dispatch-threshold check and
    // CPU fallback internally.  When ICC bytes are present, bake a CLUT for
    // profile-accurate conversion; fall back to the fast matrix approximation if
    // baking fails (e.g. corrupt profile or wrong colour space).
    #[cfg(feature = "gpu-icc")]
    if let Some(ctx) = gpu_ctx {
        let clut_data: Option<Vec<u8>> = icc_bytes.and_then(|bytes| {
            icc::bake_cmyk_clut(bytes, icc::DEFAULT_GRID_N)
                .map_err(|e| log::warn!("image: ICC CLUT bake failed, using matrix fallback: {e}"))
                .ok()
        });

        match ctx.icc_cmyk_to_rgb(
            pixels,
            clut_data.as_deref().map(|t| (t, icc::DEFAULT_GRID_N)),
        ) {
            Ok(rgb) => return Some(rgb),
            Err(e) => log::warn!("image: GPU CMYK→RGB failed, falling back to CPU: {e}"),
        }
    }

    // CPU path (also used below threshold by GpuCtx itself, but we land here
    // when the feature is disabled or no GpuCtx is provided).
    let npixels = pixels.len() / 4;
    let mut rgb = Vec::with_capacity(npixels.checked_mul(3)?);
    for chunk in pixels.chunks_exact(4) {
        let (r, g, b) =
            color::convert::cmyk_to_rgb_reflectance(chunk[0], chunk[1], chunk[2], chunk[3]);
        rgb.push(r);
        rgb.push(g);
        rgb.push(b);
    }
    Some(rgb)
}

// ── JPXDecode (JPEG 2000) ─────────────────────────────────────────────────────

/// Decode a `JPXDecode` (JPEG 2000) stream.
///
/// When the `nvjpeg2k` feature is active and `gpu` is `Some`, large images
/// (pixel area ≥ [`GPU_JPEG2K_THRESHOLD_PX`]) are decoded on the GPU via
/// nvJPEG2000.  All other images, and any image for which the GPU path fails
/// (unsupported component count, CUDA error, etc.), fall through to the CPU
/// `jpeg2k`/`OpenJPEG` path.
///
/// PDF JPEG 2000 streams may be raw codestreams (`.j2k`) or full JP2 container
/// format (`.jp2`).  Both the GPU path (nvJPEG2000 via `nvjpeg2kStreamParse`)
/// and the CPU path (`jpeg2k`/`OpenJPEG`) auto-detect the format from the stream.
///
/// 16-bit component images are downscaled to 8-bit.  Alpha channels are dropped.
fn decode_jpx(
    data: &[u8],
    pdf_w: u32,
    pdf_h: u32,
    #[cfg(feature = "nvjpeg2k")] gpu: Option<&mut NvJpeg2kDecoder>,
) -> Option<ImageDescriptor> {
    // ── GPU fast path (nvjpeg2k feature, large images, 1- or 3-component only) ─
    #[cfg(feature = "nvjpeg2k")]
    if let Some(dec) = gpu {
        let area = pdf_w.saturating_mul(pdf_h);
        if area >= GPU_JPEG2K_THRESHOLD_PX {
            if let Some(img) = decode_jpx_gpu(data, pdf_w, pdf_h, dec) {
                return Some(img);
            }
            log::debug!("image: JPXDecode: GPU path failed, retrying on CPU");
        }
    }

    // ── CPU path (jpeg2k / OpenJPEG) ─────────────────────────────────────────
    let img = Jp2Image::from_bytes(data)
        .map_err(|e| log::warn!("image: JPXDecode open error: {e}"))
        .ok()?;

    let img_data = img
        .get_pixels(None)
        .map_err(|e| log::warn!("image: JPXDecode get_pixels error: {e}"))
        .ok()?;

    let jw = img_data.width;
    let jh = img_data.height;

    if jw == 0 || jh == 0 {
        log::warn!("image: JPXDecode: JP2 reported zero dimensions {jw}×{jh}");
        return None;
    }

    if jw != pdf_w || jh != pdf_h {
        log::debug!(
            "image: JPXDecode: PDF dict says {pdf_w}×{pdf_h}, JP2 reports {jw}×{jh} — using JP2 dims"
        );
    }

    // `jpeg2k` guarantees that `img_data.format` and `img_data.data` are always
    // consistent: an L8 format always carries L8 data, etc.  We use
    // `let … else { unreachable! }` inside each arm to surface any library
    // regression loudly rather than silently skipping images.
    match img_data.format {
        ImageFormat::L8 => {
            let ImagePixelData::L8(pixels) = img_data.data else {
                unreachable!("jpeg2k: L8 format paired with non-L8 data")
            };
            Some(jpx_gray(jw, jh, pixels))
        }
        ImageFormat::La8 => {
            let ImagePixelData::La8(pixels) = img_data.data else {
                unreachable!("jpeg2k: La8 format paired with non-La8 data")
            };
            // Drop the alpha channel — keep luma bytes (every other byte starting at 0).
            let gray = pixels.chunks_exact(2).map(|c| c[0]).collect();
            Some(jpx_gray(jw, jh, gray))
        }
        ImageFormat::Rgb8 => {
            let ImagePixelData::Rgb8(pixels) = img_data.data else {
                unreachable!("jpeg2k: Rgb8 format paired with non-Rgb8 data")
            };
            Some(jpx_rgb(jw, jh, pixels))
        }
        ImageFormat::Rgba8 => {
            let ImagePixelData::Rgba8(pixels) = img_data.data else {
                unreachable!("jpeg2k: Rgba8 format paired with non-Rgba8 data")
            };
            // Drop alpha channel.
            let rgb = pixels
                .chunks_exact(4)
                .flat_map(|c| [c[0], c[1], c[2]])
                .collect();
            Some(jpx_rgb(jw, jh, rgb))
        }
        ImageFormat::L16 => {
            let ImagePixelData::L16(pixels) = img_data.data else {
                unreachable!("jpeg2k: L16 format paired with non-L16 data")
            };
            // Downscale 16-bit → 8-bit: high byte (v >> 8 ≤ 255, cast is lossless).
            let gray = pixels.iter().map(|&v| (v >> 8) as u8).collect();
            Some(jpx_gray(jw, jh, gray))
        }
        ImageFormat::La16 => {
            let ImagePixelData::La16(pixels) = img_data.data else {
                unreachable!("jpeg2k: La16 format paired with non-La16 data")
            };
            // Drop alpha; downscale luma.
            let gray = pixels.chunks_exact(2).map(|c| (c[0] >> 8) as u8).collect();
            Some(jpx_gray(jw, jh, gray))
        }
        ImageFormat::Rgb16 => {
            let ImagePixelData::Rgb16(pixels) = img_data.data else {
                unreachable!("jpeg2k: Rgb16 format paired with non-Rgb16 data")
            };
            let rgb = pixels.iter().map(|&v| (v >> 8) as u8).collect();
            Some(jpx_rgb(jw, jh, rgb))
        }
        ImageFormat::Rgba16 => {
            let ImagePixelData::Rgba16(pixels) = img_data.data else {
                unreachable!("jpeg2k: Rgba16 format paired with non-Rgba16 data")
            };
            // Drop alpha; downscale RGB.
            let rgb = pixels
                .chunks_exact(4)
                .flat_map(|c| [(c[0] >> 8) as u8, (c[1] >> 8) as u8, (c[2] >> 8) as u8])
                .collect();
            Some(jpx_rgb(jw, jh, rgb))
        }
    }
}

/// GPU-accelerated JPEG 2000 decode via nvJPEG2000.
///
/// Returns `None` for unsupported component counts (CMYK, LAB, N-channel) or
/// when any CUDA API call fails; the caller falls back to the CPU path in
/// both cases.
#[cfg(feature = "nvjpeg2k")]
fn decode_jpx_gpu(
    data: &[u8],
    pdf_w: u32,
    pdf_h: u32,
    dec: &mut NvJpeg2kDecoder,
) -> Option<ImageDescriptor> {
    let img = match dec.decode_sync(data) {
        Ok(img) => img,
        // Unsupported component count (CMYK, Gray+Alpha, N-channel), sub-sampled
        // chroma, or a codestream type not supported by this nvJPEG2000 build
        // (HTJ2K, status 4) — all are expected; fall through to CPU silently.
        Err(
            gpu::nvjpeg2k::NvJpeg2kError::UnsupportedComponents(_)
            | gpu::nvjpeg2k::NvJpeg2kError::SubSampledComponents
            | gpu::nvjpeg2k::NvJpeg2kError::Nvjpeg2kStatus(4),
        ) => return None,
        Err(e) => {
            log::warn!("image: JPXDecode GPU: nvJPEG2000 error: {e}");
            return None;
        }
    };

    if img.width != pdf_w || img.height != pdf_h {
        log::debug!(
            "image: JPXDecode GPU: PDF dict says {pdf_w}×{pdf_h}, nvJPEG2000 reports {}×{} — using nvJPEG2000 dims",
            img.width,
            img.height,
        );
    }

    let color_space = match img.color_space {
        GpuJ2kCs::Gray => ImageColorSpace::Gray,
        GpuJ2kCs::Rgb => ImageColorSpace::Rgb,
    };

    Some(ImageDescriptor {
        width: img.width,
        height: img.height,
        color_space,
        data: img.data,
        smask: None,
        filter: ImageFilter::Raw,
    })
}

/// Wrap a decoded grayscale pixel buffer into an [`ImageDescriptor`].
#[inline]
const fn jpx_gray(width: u32, height: u32, data: Vec<u8>) -> ImageDescriptor {
    ImageDescriptor {
        width,
        height,
        color_space: ImageColorSpace::Gray,
        data,
        smask: None,
        filter: ImageFilter::Raw,
    }
}

/// Wrap a decoded RGB pixel buffer into an [`ImageDescriptor`].
#[inline]
const fn jpx_rgb(width: u32, height: u32, data: Vec<u8>) -> ImageDescriptor {
    ImageDescriptor {
        width,
        height,
        color_space: ImageColorSpace::Rgb,
        data,
        smask: None,
        filter: ImageFilter::Raw,
    }
}

// ── JBIG2Decode ──────────────────────────────────────────────────────────────

/// Decode a `JBIG2Decode` stream via `hayro-jbig2`.
///
/// PDF JBIG2 uses the "embedded organisation" (Annex D.3 of T.88): the stream
/// contains page segments only; global segments live in a separate
/// `JBIG2Globals` stream referenced from `DecodeParms`.
///
/// The decoder produces one byte per pixel: 0x00 = black, 0xFF = white (for
/// grayscale images) or 0x00 = paint, 0xFF = transparent (for `ImageMask`).
/// JBIG2 convention is 0 = white, 1 = black; we invert to match the rest of
/// the image pipeline.
fn decode_jbig2(
    doc: &Document,
    data: &[u8],
    width: u32,
    height: u32,
    is_mask: bool,
    parms: Option<&Object>,
) -> Option<ImageDescriptor> {
    // Resolve optional JBIG2Globals stream from DecodeParms.
    let globals_bytes: Option<Vec<u8>> = parms
        .and_then(|o| o.as_dict().ok())
        .and_then(|d| d.get(b"JBIG2Globals").ok())
        .and_then(|o| match o {
            Object::Reference(id) => {
                let g_obj = doc.get_object(*id).ok()?;
                let g_stream = g_obj.as_stream().ok()?;
                // JBIG2Globals streams are typically not compressed, but
                // decompressed_content handles both cases transparently.
                g_stream.decompressed_content().ok()
            }
            _ => None,
        });

    // Parse the embedded JBIG2 image (page segments + optional globals).
    let img = hayro_jbig2::Image::new_embedded(data, globals_bytes.as_deref())
        .map_err(|e| log::warn!("image: JBIG2Decode parse error: {e}"))
        .ok()?;

    let jw = img.width();
    let jh = img.height();

    // Validate decoded dimensions against the PDF image dict.
    if jw != width || jh != height {
        log::warn!(
            "image: JBIG2Decode dimension mismatch: image dict says {width}×{height}, JBIG2 stream says {jw}×{jh} — using stream dimensions"
        );
    }

    let n_pixels = (jw as usize).checked_mul(jh as usize)?;
    let mut collector = Jbig2Collector {
        data: Vec::with_capacity(n_pixels),
        is_mask,
    };

    img.decode(&mut collector)
        .map_err(|e| log::warn!("image: JBIG2Decode decode error: {e}"))
        .ok()?;

    if collector.data.len() != n_pixels {
        log::warn!(
            "image: JBIG2Decode produced {} pixels, expected {n_pixels} — skipping",
            collector.data.len()
        );
        return None;
    }

    let cs = if is_mask {
        ImageColorSpace::Mask
    } else {
        ImageColorSpace::Gray
    };
    Some(ImageDescriptor {
        width: jw,
        height: jh,
        color_space: cs,
        data: collector.data,
        smask: None,
        filter: ImageFilter::Raw,
    })
}

/// Pixel collector for `hayro_jbig2::Decoder`.
///
/// JBIG2 convention: 0 = white, 1 = black.
/// `ImageColorSpace::Gray` convention: 0x00 = black, 0xFF = white.
/// `ImageColorSpace::Mask` convention: 0x00 = paint (== JBIG2 black), 0xFF = transparent.
///
/// Both output conventions share the same polarity flip: JBIG2 black (1) → 0x00,
/// JBIG2 white (0) → 0xFF.
struct Jbig2Collector {
    data: Vec<u8>,
    is_mask: bool,
}

impl Jbig2Decoder for Jbig2Collector {
    fn push_pixel(&mut self, black: bool) {
        self.data.push(if black { 0x00 } else { 0xFF });
    }

    fn push_pixel_chunk(&mut self, black: bool, chunk_count: u32) {
        let byte = if black { 0x00 } else { 0xFF };
        let n = chunk_count as usize * 8;
        self.data.extend(std::iter::repeat_n(byte, n));
    }

    fn next_line(&mut self) {
        // Row boundary — nothing to do; pixels are already stored flat.
        let _ = self.is_mask; // used at construction; suppress dead-code lint
    }
}

// ── Color space helpers ───────────────────────────────────────────────────────

/// Internal resolved colour space — what the decode path will actually produce.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ResolvedCs {
    Gray,
    Rgb,
    /// Raw CMYK (4 bytes/pixel, 255 = full ink).  Converted to RGB before returning.
    Cmyk,
}

impl ResolvedCs {
    const fn components(self) -> usize {
        match self {
            Self::Gray => 1,
            Self::Rgb => 3,
            Self::Cmyk => 4,
        }
    }

    const fn to_image_cs(self) -> ImageColorSpace {
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
fn resolve_cs<'a>(doc: &'a Document, cs_obj: &'a Object) -> ResolvedCs {
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

fn device_cs_name(name: &[u8]) -> ResolvedCs {
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
fn extract_icc_bytes(doc: &Document, cs_obj: &Object) -> Option<Vec<u8>> {
    let arr = cs_obj.as_array().ok()?;
    if arr.first().and_then(|o| o.as_name().ok()) != Some(b"ICCBased") {
        return None;
    }
    let stream_ref = arr.get(1)?;
    let id = match stream_ref {
        Object::Reference(id) => *id,
        _ => return None,
    };
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

    // ── parse_inline_params / expand_inline_key ───────────────────────────────

    #[test]
    fn inline_key_expansion_canonical() {
        assert_eq!(expand_inline_key(b"W"), b"Width");
        assert_eq!(expand_inline_key(b"H"), b"Height");
        assert_eq!(expand_inline_key(b"CS"), b"ColorSpace");
        assert_eq!(expand_inline_key(b"BPC"), b"BitsPerComponent");
        assert_eq!(expand_inline_key(b"F"), b"Filter");
        assert_eq!(expand_inline_key(b"DP"), b"DecodeParms");
        assert_eq!(expand_inline_key(b"IM"), b"ImageMask");
    }

    #[test]
    fn inline_key_unknown_passthrough() {
        // Unrecognised keys pass through unchanged.
        assert_eq!(expand_inline_key(b"Width"), b"Width");
        assert_eq!(expand_inline_key(b"Blah"), b"Blah");
    }

    #[test]
    fn inline_name_filter_expansion() {
        assert_eq!(expand_inline_name(b"Fl"), b"FlateDecode");
        assert_eq!(expand_inline_name(b"DCT"), b"DCTDecode");
        assert_eq!(expand_inline_name(b"CCF"), b"CCITTFaxDecode");
        assert_eq!(expand_inline_name(b"RL"), b"RunLengthDecode");
        assert_eq!(expand_inline_name(b"AHx"), b"ASCIIHexDecode");
        assert_eq!(expand_inline_name(b"A85"), b"ASCII85Decode");
        assert_eq!(expand_inline_name(b"LZW"), b"LZWDecode");
    }

    #[test]
    fn inline_name_cs_expansion() {
        assert_eq!(expand_inline_name(b"G"), b"DeviceGray");
        assert_eq!(expand_inline_name(b"RGB"), b"DeviceRGB");
        assert_eq!(expand_inline_name(b"CMYK"), b"DeviceCMYK");
    }

    #[test]
    fn parse_inline_params_basic() {
        // /W 4 /H 2 /CS /G /BPC 8 (abbreviated keys + abbreviated CS name)
        let params = b"/W 4 /H 2 /CS /G /BPC 8";
        let dict = parse_inline_params(params);
        assert_eq!(dict.get_i64(b"Width"), Some(4));
        assert_eq!(dict.get_i64(b"Height"), Some(2));
        assert_eq!(dict.get_i64(b"BitsPerComponent"), Some(8));
        // CS /G should expand to /DeviceGray.
        assert_eq!(
            dict.get(b"ColorSpace").ok().and_then(|o| o.as_name().ok()),
            Some(b"DeviceGray".as_ref())
        );
    }

    #[test]
    fn parse_inline_params_full_names() {
        // Full (non-abbreviated) names should also parse correctly.
        let params = b"/Width 3 /Height 1 /ColorSpace /DeviceRGB /BitsPerComponent 8";
        let dict = parse_inline_params(params);
        assert_eq!(dict.get_i64(b"Width"), Some(3));
        assert_eq!(dict.get_i64(b"Height"), Some(1));
        assert_eq!(dict.get_i64(b"BitsPerComponent"), Some(8));
    }

    // ── decode_run_length ──────────────────────────────────────────────────────

    #[test]
    fn run_length_literal_run() {
        // Run byte 2 → copy next 3 bytes.
        let data = [2u8, 0xAA, 0xBB, 0xCC];
        assert_eq!(decode_run_length(&data), vec![0xAA, 0xBB, 0xCC]);
    }

    #[test]
    fn run_length_repeat_run() {
        // Run byte 254 → repeat = 257 - 254 = 3.
        let data = [254u8, 0x42];
        assert_eq!(decode_run_length(&data), vec![0x42, 0x42, 0x42]);
    }

    #[test]
    fn run_length_eod_terminates() {
        // Run byte 128 = EOD; bytes after are ignored.
        let data = [2u8, 0xAA, 0xBB, 0xCC, 128, 0xFF, 0xFF];
        assert_eq!(decode_run_length(&data), vec![0xAA, 0xBB, 0xCC]);
    }

    #[test]
    fn run_length_empty_input() {
        assert_eq!(decode_run_length(&[]), Vec::<u8>::new());
    }

    #[test]
    fn run_length_truncates_at_max_output() {
        // Use a tiny synthetic cap so the test allocates almost nothing.
        // Each record (2 bytes: 0x81, byte) expands to 128 copies.
        // Cap at 1000 bytes; send enough records to exceed it.
        const CAP: usize = 1000;
        let mut data = Vec::new();
        for _ in 0..20 {
            data.push(0x81_u8); // repeat = 257 - 129 = 128
            data.push(0xAB_u8);
        }
        // 20 × 128 = 2560 bytes would be decoded without cap.
        let out = decode_run_length_capped(&data, CAP);
        assert!(out.len() <= CAP, "output {} exceeded cap {CAP}", out.len());
        assert!(
            !out.is_empty(),
            "should have decoded something before truncation"
        );
        assert!(out.iter().all(|&b| b == 0xAB), "all bytes should be 0xAB");
    }

    // ── decode_inline_image ────────────────────────────────────────────────────

    #[test]
    fn inline_image_raw_gray() {
        // 2×2 raw DeviceGray image, 8 bpc, no filter.
        let params = b"/W 2 /H 2 /CS /G /BPC 8";
        let data = [0x00u8, 0x80, 0xFF, 0x40];
        let doc = lopdf::Document::new();
        let img = decode_inline_image(&doc, params, &data).expect("decode should succeed");
        assert_eq!(img.width, 2);
        assert_eq!(img.height, 2);
        assert_eq!(img.color_space, ImageColorSpace::Gray);
        assert_eq!(img.data, data.to_vec());
    }

    #[test]
    fn inline_image_mask() {
        // 2×1 mask image (ImageMask=true means Mask colour space).
        let params = b"/W 2 /H 1 /IM true /BPC 1";
        // 1-bpp mask: byte 0b10000000 → pixel 0 = 1 (transparent), pixel 1 = 0 (paint).
        let data = [0b1000_0000u8];
        let doc = lopdf::Document::new();
        let img = decode_inline_image(&doc, params, &data).expect("decode should succeed");
        assert_eq!(img.color_space, ImageColorSpace::Mask);
        // Expanded 1-bpp: bit 7 = 1 → 0xFF, bit 6 = 0 → 0x00
        assert_eq!(img.data[0], 0xFF); // first pixel: transparent
        assert_eq!(img.data[1], 0x00); // second pixel: paint
    }

    #[test]
    fn inline_image_degenerate_dims() {
        let params = b"/W 0 /H 1 /CS /G /BPC 8";
        let doc = lopdf::Document::new();
        assert!(decode_inline_image(&doc, params, &[]).is_none());
    }

    #[test]
    fn inline_image_missing_dims() {
        // No width/height → None.
        let params = b"/CS /G /BPC 8";
        let doc = lopdf::Document::new();
        assert!(decode_inline_image(&doc, params, &[0u8; 4]).is_none());
    }

    // ── JBIG2 collector unit tests ────────────────────────────────────────────

    #[test]
    fn jbig2_collector_push_pixel_grayscale() {
        // JBIG2: black=true → Gray 0x00, black=false → Gray 0xFF.
        let mut c = Jbig2Collector {
            data: Vec::new(),
            is_mask: false,
        };
        Jbig2Decoder::push_pixel(&mut c, true);
        Jbig2Decoder::push_pixel(&mut c, false);
        assert_eq!(c.data, [0x00, 0xFF]);
    }

    #[test]
    fn jbig2_collector_push_pixel_chunk() {
        let mut c = Jbig2Collector {
            data: Vec::new(),
            is_mask: false,
        };
        // chunk_count=2 → 16 pixels of white (0xFF each).
        Jbig2Decoder::push_pixel_chunk(&mut c, false, 2);
        assert_eq!(c.data.len(), 16);
        assert!(c.data.iter().all(|&b| b == 0xFF));
    }

    #[test]
    fn jbig2_decode_invalid_data_returns_none() {
        // Corrupt/empty JBIG2 data must not panic — it must return None.
        let doc = lopdf::Document::new();
        let result = decode_jbig2(&doc, b"\x00\x01\x02\x03", 4, 4, false, None);
        assert!(result.is_none());
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
