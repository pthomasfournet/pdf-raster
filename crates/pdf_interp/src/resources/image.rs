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
//! | `FlateDecode` | yes/no | yes |
//! | none (raw) | yes/no | yes |
//! | `DCTDecode` (JPEG) | no | yes (via `zune-jpeg`) |
//! | `JPXDecode` (JPEG 2000) | no | yes (via `jpeg2k`/`OpenJPEG`) |
//! | `JBIG2Decode` | — | stub |
//! | `CCITTFaxDecode` (`K≥0`, Group 3) | — | stub |
//!
//! # Pixel layout in `ImageDescriptor::data`
//!
//! | [`ImageColorSpace`] | Bytes per pixel | `0x00` meaning | `0xFF` meaning |
//! |---|---|---|---|
//! | `Gray` | 1 | black | white |
//! | `Rgb` | 3 | black (R=G=B=0) | white |
//! | `Mask` | 1 | paint with fill colour | transparent (leave background) |

use std::borrow::Cow;

use jpeg2k::{Image as Jp2Image, ImageFormat, ImagePixelData};
use lopdf::{Dictionary, Document, Object, ObjectId};

use crate::resources::dict_ext::DictExt;
use zune_core::bytestream::ZCursor;
use zune_core::colorspace::ColorSpace as ZColorSpace;
use zune_core::options::DecoderOptions;
use zune_jpeg::JpegDecoder;

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
    /// Optional soft-mask (SMask): one byte per pixel, same dimensions as
    /// `data`.  `0x00` = fully transparent (skip pixel); `0xFF` = fully
    /// opaque.  `None` means the image is fully opaque.
    pub smask: Option<Vec<u8>>,
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
    if w_raw <= 0 || h_raw <= 0 || w_raw > 65536 || h_raw > 65536 {
        log::warn!("image: degenerate dimensions {w_raw}×{h_raw}, skipping");
        return None;
    }
    #[expect(
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation,
        reason = "w_raw and h_raw are validated > 0 and ≤ 65536 above; safe to cast to u32"
    )]
    let (w, h) = (w_raw as u32, h_raw as u32);

    let is_mask = stream.dict.get_bool(b"ImageMask").unwrap_or(false);

    let filter = stream.dict.get(b"Filter").ok().and_then(filter_name);

    let mut img = match filter.as_deref() {
        None => decode_raw(stream.content.as_slice(), w, h, is_mask, &stream.dict),
        Some("FlateDecode") => match stream.decompressed_content() {
            Ok(data) => decode_raw(&data, w, h, is_mask, &stream.dict),
            Err(e) => {
                log::warn!("image: FlateDecode decompression failed: {e}");
                None
            }
        },
        Some("CCITTFaxDecode") => {
            let parms = stream.dict.get(b"DecodeParms").ok();
            decode_ccitt(stream.content.as_slice(), w, h, is_mask, parms)
        }
        Some("DCTDecode") => decode_dct(stream.content.as_slice(), w, h),
        Some("JPXDecode") => decode_jpx(stream.content.as_slice(), w, h),
        Some("JBIG2Decode") => {
            log::debug!("image: filter \"JBIG2Decode\" not yet implemented");
            None
        }
        Some(other) => {
            log::warn!("image: unknown filter {other:?}");
            None
        }
    }?;

    // Resolve and decode the soft mask (`SMask`), if present.
    if let Ok(Object::Reference(smask_id)) = stream.dict.get(b"SMask") {
        if let Some(alpha) = decode_smask(doc, *smask_id, img.width, img.height) {
            img.smask = Some(alpha);
        } else {
            // `SMask` is present but could not be decoded.  Blitting without a
            // mask would paint the image's colour over a large area it should
            // not cover (e.g. a solid-colour overlay that is transparent
            // everywhere the mask is zero).  Skip the image instead.
            log::debug!(
                "image: skipping image — SMask (object {smask_id:?}) could not be decoded"
            );
            return None;
        }
    }

    Some(img)
}

// ── SMask decoding ────────────────────────────────────────────────────────────

/// Decode a soft-mask (`SMask`) image stream into a flat `Vec<u8>` of grayscale
/// alpha values (0 = transparent, 255 = opaque), one byte per pixel.
///
/// Returns `None` if the `SMask` stream cannot be resolved or its filter is not
/// supported (caller should skip the image rather than blit without a mask).
fn decode_smask(doc: &Document, id: ObjectId, img_w: u32, img_h: u32) -> Option<Vec<u8>> {
    let obj = doc.get_object(id).ok()?;
    let stream = obj.as_stream().ok()?;

    #[expect(
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation,
        reason = "SMask Width/Height validated positive and ≤ image dims (already validated ≤ 65536)"
    )]
    let (sm_w, sm_h) = (
        stream.dict.get_i64(b"Width")? as u32,
        stream.dict.get_i64(b"Height")? as u32,
    );

    let filter = stream.dict.get(b"Filter").ok().and_then(filter_name);

    let raw: Vec<u8> = match filter.as_deref() {
        None => stream.content.clone(),
        Some("FlateDecode") => stream.decompressed_content().ok()?,
        Some("CCITTFaxDecode") => {
            let parms = stream.dict.get(b"DecodeParms").ok();
            let sm_desc = decode_ccitt(stream.content.as_slice(), sm_w, sm_h, true, parms)?;
            // decode_ccitt returns Mask-space; convert to alpha: 0x00 mask → opaque (0xFF alpha)
            let alpha = sm_desc
                .data
                .iter()
                .map(|&v| if v == 0x00 { 0xFF } else { 0x00 })
                .collect();
            // Resize to match image dimensions if needed (simple nearest-neighbour).
            return Some(scale_smask(alpha, sm_w, sm_h, img_w, img_h));
        }
        Some("JBIG2Decode") => {
            log::debug!("image: SMask filter \"JBIG2Decode\" not yet supported");
            return None;
        }
        Some(other) => {
            log::debug!("image: SMask filter {other:?} not yet supported");
            return None;
        }
    };

    let bpc = stream.dict.get_i64(b"BitsPerComponent").unwrap_or(8);

    // Decode raw bytes to one-byte-per-pixel alpha.
    let alpha: Vec<u8> = match bpc {
        1 => {
            // Expand 1bpp to 8bpp; 1-bit = opaque (0xFF), 0-bit = transparent (0x00).
            let cols = usize::try_from(sm_w).ok()?;
            let rows = usize::try_from(sm_h).ok()?;
            let row_bytes = cols.div_ceil(8);
            let mut out = Vec::with_capacity(cols * rows);
            for row in raw.chunks(row_bytes) {
                for x in 0..cols {
                    let byte = row.get(x / 8).copied().unwrap_or(0);
                    let bit = (byte >> (7 - (x % 8))) & 1;
                    out.push(if bit == 1 { 0xFF } else { 0x00 });
                }
            }
            out
        }
        8 => raw,
        other => {
            log::debug!("image: SMask {other} bpc not yet supported");
            return None;
        }
    };

    Some(scale_smask(alpha, sm_w, sm_h, img_w, img_h))
}

/// Nearest-neighbour resize of a flat grayscale buffer from `(sw, sh)` to `(dw, dh)`.
fn scale_smask(src: Vec<u8>, sw: u32, sh: u32, dw: u32, dh: u32) -> Vec<u8> {
    if sw == dw && sh == dh {
        return src;
    }
    let src_w = u64::from(sw);
    let src_h = u64::from(sh);
    let dst_w = u64::from(dw);
    let dst_h = u64::from(dh);
    let mut out = Vec::with_capacity(usize::try_from(dst_w * dst_h).unwrap_or(0));
    for dy in 0..dh {
        #[expect(clippy::cast_possible_truncation, reason = "nearest-neighbour: result ≤ sh-1")]
        let sy = (u64::from(dy) * src_h / dst_h) as u32;
        for dx in 0..dw {
            #[expect(clippy::cast_possible_truncation, reason = "nearest-neighbour: result ≤ sw-1")]
            let sx = (u64::from(dx) * src_w / dst_w) as u32;
            #[expect(clippy::cast_possible_truncation, reason = "sy*sw+sx ≤ sw*sh ≤ u32::MAX for validated dims")]
            out.push(src[(u64::from(sy) * src_w + u64::from(sx)) as usize]);
        }
    }
    out
}

// ── XObject lookup ────────────────────────────────────────────────────────────

/// Resolve the `XObject` resource named `name` to its stream `ObjectId`.
fn xobject_id(doc: &Document, page_dict: &Dictionary, name: &[u8]) -> Option<ObjectId> {
    let res = resolve_dict(doc, page_dict.get(b"Resources").ok()?)?;
    let xobj = resolve_dict(doc, res.get(b"XObject").ok()?)?;
    match xobj.get(name).ok()? {
        Object::Reference(id) => Some(*id),
        _ => None,
    }
}

/// Dereference a `Dictionary` or `Reference → Dictionary`.
fn resolve_dict<'a>(doc: &'a Document, obj: &'a Object) -> Option<&'a Dictionary> {
    match obj {
        Object::Dictionary(d) => Some(d),
        Object::Reference(id) => doc.get_dictionary(*id).ok(),
        _ => None,
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

/// Expand raw (already-decompressed) pixel bytes into a normalised form.
///
/// Handles 1-bit-per-sample images (packed MSB first) by expanding to 1 byte
/// per pixel.  All other bit depths must be 8.
fn decode_raw(
    data: &[u8],
    width: u32,
    height: u32,
    is_mask: bool,
    dict: &Dictionary,
) -> Option<ImageDescriptor> {
    let bpc = dict.get_i64(b"BitsPerComponent").unwrap_or(8);

    let cs = if is_mask {
        ImageColorSpace::Mask
    } else {
        color_space_from_dict(dict)
    };

    match bpc {
        1 => {
            // Packed 1-bit: expand to one byte per pixel.
            let pixels = expand_1bpp(data, width, height)?;
            Some(ImageDescriptor {
                width,
                height,
                color_space: cs,
                data: pixels,
                smask: None,
            })
        }
        8 => {
            let components: usize = match cs {
                ImageColorSpace::Gray | ImageColorSpace::Mask => 1,
                ImageColorSpace::Rgb => 3,
            };
            let expected = (width as usize)
                .checked_mul(height as usize)
                .and_then(|p| p.checked_mul(components));
            let Some(expected) = expected else {
                log::warn!("image: dimensions overflow usize ({width}×{height}×{components})");
                return None;
            };
            if data.len() < expected {
                log::warn!(
                    "image: raw data too short ({} bytes, need {expected} for {width}×{height}×{components})",
                    data.len()
                );
                return None;
            }
            Some(ImageDescriptor {
                width,
                height,
                color_space: cs,
                data: data[..expected].to_vec(),
                smask: None,
            })
        }
        other => {
            log::debug!("image: {other} bits-per-component not yet implemented");
            None
        }
    }
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

// ── CCITTFaxDecode ─────────────────────────────────────────────────────────────

/// Decode a `CCITTFaxDecode` stream.
///
/// `K` in `DecodeParms`:
/// - `K < 0` → Group 4 (T.6) — supported.
/// - `K = 0` → Group 3 1D — not yet implemented.
/// - `K > 0` → Group 3 2D — not yet implemented.
fn decode_ccitt(
    data: &[u8],
    width: u32,
    height: u32,
    is_mask: bool,
    parms: Option<&Object>,
) -> Option<ImageDescriptor> {
    // Resolve DecodeParms once; both K and BlackIs1 live in the same dict.
    let parms_dict = parms.and_then(|o| o.as_dict().ok());

    let k = parms_dict.and_then(|d| d.get_i64(b"K")).unwrap_or(0);

    // BlackIs1: when true, 1-bits encode black (the reverse of the default).
    let black_is_1 = parms_dict
        .and_then(|d| d.get_bool(b"BlackIs1"))
        .unwrap_or(false);

    if k >= 0 {
        log::debug!("image: CCITTFaxDecode K={k} (Group 3) not yet implemented");
        return None;
    }

    // Group 4 (K < 0).
    let w_u16 = u16::try_from(width).ok()?;
    let h_u16 = u16::try_from(height).ok()?;

    let capacity = (width as usize).checked_mul(height as usize)?;
    let mut data_out: Vec<u8> = Vec::with_capacity(capacity);
    let mut rows_decoded: u32 = 0;

    let completed =
        fax::decoder::decode_g4(data.iter().copied(), w_u16, Some(h_u16), |transitions| {
            let row_start = data_out.len();
            data_out.extend(fax::decoder::pels(transitions, w_u16).map(|color| {
                // Map fax Color to PDF image convention: 0x00 = black, 0xFF = white.
                let is_black = match color {
                    fax::Color::Black => !black_is_1,
                    fax::Color::White => black_is_1,
                };
                if is_black { 0x00u8 } else { 0xFFu8 }
            }));
            // Pad any short row (defensive against malformed data).
            let row_end = row_start + width as usize;
            data_out.resize(row_end, 0xFF);
            rows_decoded += 1;
        });

    if completed.is_none() {
        log::warn!(
            "image: CCITTFaxDecode Group4 decode incomplete — got {rows_decoded}/{height} rows"
        );
        if rows_decoded == 0 {
            return None;
        }
        // Partial decode: pad remaining rows with white so the image is not garbage.
        data_out.resize(capacity, 0xFF);
    }

    let cs = if is_mask {
        ImageColorSpace::Mask
    } else {
        ImageColorSpace::Gray
    };
    Some(ImageDescriptor {
        width,
        height,
        color_space: cs,
        data: data_out,
        smask: None,
    })
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
fn decode_dct(data: &[u8], pdf_w: u32, pdf_h: u32) -> Option<ImageDescriptor> {
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
        }),
        ZColorSpace::RGB => Some(ImageDescriptor {
            width: jw,
            height: jh,
            color_space: ImageColorSpace::Rgb,
            data: pixels,
            smask: None,
        }),
        ZColorSpace::CMYK => cmyk_to_rgb(&pixels, jw, jh),
        // out_cs is always Luma, RGB, or CMYK — set from the components match above.
        _ => unreachable!("DCTDecode: unexpected out_cs variant"),
    }
}

/// Convert a flat CMYK byte buffer (inverted ink densities, 4 bytes/pixel) to
/// an RGB byte buffer (3 bytes/pixel).
///
/// PDF JPEG CMYK stores inverted ink densities: 0 = full ink, 255 = no ink.
/// `zune-jpeg` returns bytes in this same inverted form.  The conversion is:
///
/// ```text
/// R = (255 - C) * (255 - K) / 255
/// ```
///
/// Each intermediate value `255u16.saturating_sub(x + k)` is ≤ 255,
/// making the `as u8` cast lossless.
#[expect(
    clippy::many_single_char_names,
    reason = "CMYK and RGB are conventional single-letter colour channel names"
)]
fn cmyk_to_rgb(pixels: &[u8], width: u32, height: u32) -> Option<ImageDescriptor> {
    let npixels = (width as usize)
        .checked_mul(height as usize)
        .expect("DCTDecode: CMYK pixel count overflows usize — dimensions were validated");

    if pixels.len() != npixels * 4 {
        log::warn!(
            "image: DCTDecode: CMYK buffer is {} bytes, expected {} for {width}×{height}",
            pixels.len(),
            npixels * 4
        );
        return None;
    }

    let rgb_cap = npixels
        .checked_mul(3)
        .expect("DCTDecode: RGB output size overflows usize — dimensions were validated");
    let mut rgb = Vec::with_capacity(rgb_cap);

    for chunk in pixels.chunks_exact(4) {
        // Inverted ink densities (0=full ink, 255=none) — complement to get ink density.
        let c = u16::from(255 - chunk[0]);
        let m = u16::from(255 - chunk[1]);
        let y = u16::from(255 - chunk[2]);
        let k = u16::from(255 - chunk[3]);
        // saturating_sub clamps to [0, 255]; the result fits in u8 without truncation.
        #[expect(
            clippy::cast_possible_truncation,
            reason = "saturating_sub result ≤ 255"
        )]
        let r = 255u16.saturating_sub(c + k) as u8;
        #[expect(
            clippy::cast_possible_truncation,
            reason = "saturating_sub result ≤ 255"
        )]
        let g = 255u16.saturating_sub(m + k) as u8;
        #[expect(
            clippy::cast_possible_truncation,
            reason = "saturating_sub result ≤ 255"
        )]
        let b = 255u16.saturating_sub(y + k) as u8;
        rgb.push(r);
        rgb.push(g);
        rgb.push(b);
    }

    Some(ImageDescriptor {
        width,
        height,
        color_space: ImageColorSpace::Rgb,
        data: rgb,
        smask: None,
    })
}

// ── JPXDecode (JPEG 2000) ─────────────────────────────────────────────────────

/// Decode a `JPXDecode` (JPEG 2000) stream via `jpeg2k` (`OpenJPEG` bindings).
///
/// PDF JPEG 2000 streams may be raw codestreams (`.j2k`) or full JP2 container
/// format (`.jp2`).  `jpeg2k::Image::from_bytes` auto-detects the format from
/// the first bytes of the stream.
///
/// 16-bit component images are downscaled to 8-bit by taking the high byte
/// (`v >> 8`), which is lossless for display purposes.  Alpha channels are
/// dropped — PDF image blitting does not use them.
fn decode_jpx(data: &[u8], pdf_w: u32, pdf_h: u32) -> Option<ImageDescriptor> {
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

/// Wrap a decoded grayscale pixel buffer into an [`ImageDescriptor`].
#[inline]
const fn jpx_gray(width: u32, height: u32, data: Vec<u8>) -> ImageDescriptor {
    ImageDescriptor {
        width,
        height,
        color_space: ImageColorSpace::Gray,
        data,
        smask: None,
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
    }
}

// ── Color space helpers ───────────────────────────────────────────────────────

fn color_space_from_dict(dict: &Dictionary) -> ImageColorSpace {
    let name = dict.get(b"ColorSpace").ok().and_then(cs_name);
    match name.as_deref() {
        Some("DeviceRGB" | "CalRGB" | "sRGB") => ImageColorSpace::Rgb,
        _ => ImageColorSpace::Gray, // DeviceGray, CalGray, ICCBased, unknown, or absent
    }
}

/// Extract the colour-space name from a `ColorSpace` value (either a `Name` or
/// the first element of an array).
fn cs_name(obj: &Object) -> Option<Cow<'_, str>> {
    match obj {
        Object::Name(n) => Some(String::from_utf8_lossy(n)),
        Object::Array(arr) => arr
            .first()
            .and_then(|o| o.as_name().ok())
            .map(String::from_utf8_lossy),
        _ => None,
    }
}
