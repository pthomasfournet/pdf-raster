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
//! | `CCITTFaxDecode` (`K=-1`, Group 4) | yes/no | yes |
//! | `FlateDecode` | no | yes |
//! | none (raw) | no | yes |
//! | `DCTDecode` (JPEG) | no | stub — logs and returns `None` |
//! | `JBIG2Decode` | — | stub |
//! | `JPXDecode` | — | stub |

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
#[derive(Debug)]
pub struct ImageDescriptor {
    /// Pixel width of the decoded image.
    pub width: u32,
    /// Pixel height of the decoded image.
    pub height: u32,
    /// Colour interpretation of `data`.
    pub color_space: ImageColorSpace,
    /// Raw pixel bytes.  For `Mask`, 1 byte per pixel (0 = paint, 1 = transparent).
    /// For `Gray`, 1 byte per pixel.  For `Rgb`, 3 bytes per pixel.
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
pub fn resolve_image(doc: &Document, page_dict: &Dictionary, name: &[u8]) -> Option<ImageDescriptor> {
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
        Some("FlateDecode") => {
            match stream.decompressed_content() {
                Ok(data) => decode_raw(&data, w, h, is_mask, &stream.dict),
                Err(e) => {
                    log::warn!("image: FlateDecode failed: {e}");
                    None
                }
            }
        }
        Some("CCITTFaxDecode") => {
            let parms = stream.dict.get(b"DecodeParms").ok();
            decode_ccitt(stream.content.as_slice(), w, h, is_mask, parms)
        }
        Some(f @ ("DCTDecode" | "JBIG2Decode" | "JPXDecode")) => {
            log::debug!("image: {f} not yet implemented");
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

// ── Filter name extraction ─────────────────────────────────────────────────────

fn filter_name(obj: &Object) -> Option<String> {
    match obj {
        Object::Name(n) => Some(String::from_utf8_lossy(n).into_owned()),
        Object::Array(arr) => arr
            .first()
            .and_then(|o| o.as_name().ok())
            .map(|n| String::from_utf8_lossy(n).into_owned()),
        _ => None,
    }
}

// ── Raw / FlateDecode image decoding ─────────────────────────────────────────

/// Expand raw (already-decompressed) pixel bytes into a normalised form.
///
/// Handles 1-bit-per-sample images (packed MSB first) by expanding to 1 byte
/// per pixel.
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

    let components = match cs {
        ImageColorSpace::Gray | ImageColorSpace::Mask => 1u32,
        ImageColorSpace::Rgb => 3,
    };

    if bpc == 1 {
        // Packed 1-bit: expand to one byte per pixel (0 or 1).
        let pixels = expand_1bpp(data, width, height);
        Some(ImageDescriptor {
            width,
            height,
            color_space: cs,
            data: pixels,
        })
    } else if bpc == 8 {
        let expected = width as usize * height as usize * components as usize;
        if data.len() < expected {
            log::warn!("image: raw data too short ({} < {expected})", data.len());
            return None;
        }
        Some(ImageDescriptor {
            width,
            height,
            color_space: cs,
            data: data[..expected].to_vec(),
        })
    } else {
        log::debug!("image: {bpc} bits-per-component not yet implemented");
        None
    }
}

/// Expand 1-bit-per-pixel packed data (MSB first) to 1 byte per pixel.
/// Output byte = 0 for black, 255 for white.
fn expand_1bpp(data: &[u8], width: u32, height: u32) -> Vec<u8> {
    let row_bytes = (width as usize).div_ceil(8);
    let mut out = Vec::with_capacity(width as usize * height as usize);
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
            let bit = if byte_idx < row_data.len() {
                (row_data[byte_idx] >> bit_idx) & 1
            } else {
                0
            };
            // PDF convention: 0 = black, 1 = white
            out.push(if bit == 0 { 0 } else { 255 });
        }
    }
    out
}

// ── CCITTFaxDecode ─────────────────────────────────────────────────────────────

/// Decode a `CCITTFaxDecode` stream.
///
/// `K` in `DecodeParms`:
/// - `K < 0` → Group 4 (T.6), the only variant we support.
/// - `K = 0` → Group 3 1D — not yet implemented.
/// - `K > 0` → Group 3 2D — not yet implemented.
fn decode_ccitt(
    data: &[u8],
    width: u32,
    height: u32,
    is_mask: bool,
    parms: Option<&Object>,
) -> Option<ImageDescriptor> {
    let k = parms
        .and_then(|o| o.as_dict().ok())
        .and_then(|d| d.get(b"K").ok())
        .and_then(|o| o.as_i64().ok())
        .unwrap_or(0);

    // BlackIs1 flag flips polarity.
    let black_is_1 = parms
        .and_then(|o| o.as_dict().ok())
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

    let mut rows: Vec<Vec<u8>> = Vec::with_capacity(height as usize);

    fax::decoder::decode_g4(
        data.iter().copied(),
        w_u16,
        Some(h_u16),
        |transitions| {
            let row: Vec<u8> = fax::decoder::pels(transitions, w_u16)
                .map(|color| {
                    let is_black = match color {
                        fax::Color::Black => !black_is_1,
                        fax::Color::White => black_is_1,
                    };
                    if is_black { 0u8 } else { 255u8 }
                })
                .collect();
            rows.push(row);
        },
    );

    if rows.is_empty() {
        log::warn!("image: CCITTFaxDecode Group4 produced no rows");
        return None;
    }

    let mut data_out = Vec::with_capacity(width as usize * height as usize);
    for mut row in rows {
        // Pad short rows to the declared width (defensive; fax decoder should not produce them).
        row.resize(width as usize, 255u8);
        data_out.extend_from_slice(&row);
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
    let name = dict
        .get(b"ColorSpace")
        .ok()
        .and_then(cs_name)
        .unwrap_or_default();

    match name.as_str() {
        "DeviceRGB" | "CalRGB" | "sRGB" => ImageColorSpace::Rgb,
        _ => ImageColorSpace::Gray, // DeviceGray, CalGray, ICCBased, or unknown → grey
    }
}

fn cs_name(obj: &Object) -> Option<String> {
    match obj {
        Object::Name(n) => Some(String::from_utf8_lossy(n).into_owned()),
        Object::Array(arr) => arr
            .first()
            .and_then(|o| o.as_name().ok())
            .map(|n| String::from_utf8_lossy(n).into_owned()),
        _ => None,
    }
}
