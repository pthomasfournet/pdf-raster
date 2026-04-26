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
//! | `DCTDecode` (JPEG) | no | stub — logs and returns `None` |
//! | `JBIG2Decode` | — | stub |
//! | `JPXDecode` | — | stub |
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

use lopdf::{Dictionary, Document, Object, ObjectId};

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
    if stream.dict.get(b"Subtype").ok()?.as_name().ok()? != b"Image" {
        return None;
    }

    let w_raw = stream.dict.get(b"Width").ok()?.as_i64().ok()?;
    let h_raw = stream.dict.get(b"Height").ok()?.as_i64().ok()?;
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

    let is_mask = stream
        .dict
        .get(b"ImageMask")
        .ok()
        .and_then(|o| o.as_bool().ok())
        .unwrap_or(false);

    let filter = stream.dict.get(b"Filter").ok().and_then(filter_name);

    match filter.as_deref() {
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
        Some("DCTDecode" | "JBIG2Decode" | "JPXDecode") => {
            log::debug!(
                "image: filter {:?} not yet implemented",
                filter.as_deref().unwrap_or("(none)")
            );
            None
        }
        Some(other) => {
            log::warn!("image: unknown filter {other:?}");
            None
        }
    }
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

/// Extract the filter name from a `Filter` entry (either a `Name` or the first
/// element of a `Name` array).  Returns a borrowed `&str` when the bytes are
/// valid UTF-8, allocating only for non-ASCII names (which are invalid in PDF).
fn filter_name(obj: &Object) -> Option<Cow<'_, str>> {
    match obj {
        Object::Name(n) => Some(String::from_utf8_lossy(n)),
        Object::Array(arr) => arr
            .first()
            .and_then(|o| o.as_name().ok())
            .map(String::from_utf8_lossy),
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
    let bpc = dict
        .get(b"BitsPerComponent")
        .ok()
        .and_then(|o| o.as_i64().ok())
        .unwrap_or(8);

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

    let k = parms_dict
        .and_then(|d| d.get(b"K").ok())
        .and_then(|o| o.as_i64().ok())
        .unwrap_or(0);

    // BlackIs1: when true, 1-bits encode black (the reverse of the default).
    let black_is_1 = parms_dict
        .and_then(|d| d.get(b"BlackIs1").ok())
        .and_then(|o| o.as_bool().ok())
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
    })
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
