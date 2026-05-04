//! Inline image decoding (`BI … ID … EI`) and parameter parsing.
#![expect(
    clippy::redundant_pub_crate,
    reason = "parse_inline_params is pub(crate) for prescan.rs; \
              redundant here because inline is a private module, but explicit visibility aids readability"
)]

use lopdf::{Dictionary, Document, Object};

use super::codecs::{decode_ccitt, decode_dct, decode_jbig2, decode_jpx};
use super::raw::decode_raw;
use super::{ImageDescriptor, ImageFilter, filter_name, validated_dims};
use crate::resources::dict_ext::DictExt;

#[cfg(feature = "nvjpeg")]
use gpu::nvjpeg::NvJpegDecoder;

#[cfg(feature = "nvjpeg2k")]
use gpu::nvjpeg2k::NvJpeg2kDecoder;

#[cfg(feature = "vaapi")]
use gpu::JpegQueueHandle;

#[cfg(feature = "gpu-icc")]
use gpu::GpuCtx;

#[cfg(feature = "gpu-icc")]
use super::icc::IccClutCache;

#[cfg(test)]
use super::ImageColorSpace;

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
/// GPU decoders are passed in for `DCTDecode` and `JPXDecode` streams.  The same
/// area threshold as `resolve_image` applies — images below
/// [`GPU_JPEG_THRESHOLD_PX`](super::GPU_JPEG_THRESHOLD_PX) use the CPU path
/// regardless.  Most inline images are small (thumbnails, icons) and will fall
/// through to the CPU path; the parameters are plumbed through so that the
/// dispatch logic in `decode_dct` / `decode_jpx` can make the threshold decision
/// rather than unconditionally bypassing GPU decode for all inline images.
///
/// Returns `None` if the parameter block is unparseable, dimensions are
/// degenerate, or the filter is unsupported.
#[must_use]
pub fn decode_inline_image(
    doc: &Document,
    params: &[u8],
    data: &[u8],
    #[cfg(feature = "nvjpeg")] gpu: Option<&mut NvJpegDecoder>,
    #[cfg(feature = "vaapi")] vaapi: Option<&JpegQueueHandle>,
    #[cfg(feature = "nvjpeg2k")] gpu_j2k: Option<&mut NvJpeg2kDecoder>,
    #[cfg(feature = "gpu-icc")] gpu_ctx: Option<&GpuCtx>,
    #[cfg(feature = "gpu-icc")] clut_cache: Option<&mut IccClutCache>,
) -> Option<ImageDescriptor> {
    let dict = parse_inline_params(params);

    let w_raw = dict.get_i64(b"Width")?;
    let h_raw = dict.get_i64(b"Height")?;
    let Some((w, h)) = validated_dims(w_raw, h_raw) else {
        log::warn!("inline image: degenerate dimensions {w_raw}×{h_raw}, skipping");
        return None;
    };

    let is_mask = dict.get_bool(b"ImageMask").unwrap_or(false);
    let filter = dict.get(b"Filter").ok().and_then(filter_name);

    let img_filter = ImageFilter::from_filter_str(filter.as_deref());

    let mut img = match filter.as_deref() {
        None => decode_raw(
            doc,
            data,
            w,
            h,
            is_mask,
            &dict,
            #[cfg(feature = "gpu-icc")]
            gpu_ctx,
            #[cfg(feature = "gpu-icc")]
            clut_cache,
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
                    gpu_ctx,
                    #[cfg(feature = "gpu-icc")]
                    clut_cache,
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
            // GPU dispatch is threshold-gated inside decode_dct (same threshold as
            // resolve_image). Most inline images are small and will use the CPU path.
            decode_dct(
                data,
                w,
                h,
                #[cfg(feature = "nvjpeg")]
                gpu,
                #[cfg(feature = "vaapi")]
                vaapi,
                #[cfg(feature = "gpu-icc")]
                gpu_ctx,
                #[cfg(feature = "gpu-icc")]
                clut_cache,
            )
        }
        Some("JPXDecode") => {
            // GPU dispatch is threshold-gated inside decode_jpx.
            #[cfg(feature = "nvjpeg2k")]
            {
                decode_jpx(data, w, h, gpu_j2k)
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
                gpu_ctx,
                #[cfg(feature = "gpu-icc")]
                clut_cache,
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

/// `RunLength` decode with an output cap; returns immediately when `max_output` bytes are reached.
pub(super) fn decode_run_length_capped(data: &[u8], max_output: usize) -> Vec<u8> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < data.len() {
        let run_byte = data[i];
        i += 1;
        match run_byte {
            128 => break, // EOD
            0..=127 => {
                let count = run_byte as usize + 1;
                let end = i.saturating_add(count);
                if end > data.len() {
                    log::warn!(
                        "inline image: RunLengthDecode literal run truncated at EOF \
                         (wanted {count} bytes, {} available)",
                        data.len() - i
                    );
                }
                let end = end.min(data.len());
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
                } else {
                    log::warn!(
                        "inline image: RunLengthDecode truncated — repeat byte missing at EOF"
                    );
                    break;
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
pub(crate) fn parse_inline_params(params: &[u8]) -> Dictionary {
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
pub(super) const fn expand_inline_key(key: &[u8]) -> &[u8] {
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
pub(super) fn inline_token_to_object(tok: &crate::content::tokenizer::Token<'_>) -> Object {
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
pub(super) fn expand_inline_value(obj: Object) -> Object {
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
pub(super) const fn expand_inline_name(name: &[u8]) -> &[u8] {
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

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

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
        use crate::resources::dict_ext::DictExt;
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
        use crate::resources::dict_ext::DictExt;
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
        let img = decode_inline_image(
            &doc,
            params,
            &data,
            #[cfg(feature = "nvjpeg")]
            None,
            #[cfg(feature = "vaapi")]
            None,
            #[cfg(feature = "nvjpeg2k")]
            None,
            #[cfg(feature = "gpu-icc")]
            None,
            #[cfg(feature = "gpu-icc")]
            None,
        )
        .expect("decode should succeed");
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
        let img = decode_inline_image(
            &doc,
            params,
            &data,
            #[cfg(feature = "nvjpeg")]
            None,
            #[cfg(feature = "vaapi")]
            None,
            #[cfg(feature = "nvjpeg2k")]
            None,
            #[cfg(feature = "gpu-icc")]
            None,
            #[cfg(feature = "gpu-icc")]
            None,
        )
        .expect("decode should succeed");
        assert_eq!(img.color_space, ImageColorSpace::Mask);
        // Expanded 1-bpp: bit 7 = 1 → 0xFF, bit 6 = 0 → 0x00
        assert_eq!(img.data[0], 0xFF); // first pixel: transparent
        assert_eq!(img.data[1], 0x00); // second pixel: paint
    }

    #[test]
    fn inline_image_degenerate_dims() {
        let params = b"/W 0 /H 1 /CS /G /BPC 8";
        let doc = lopdf::Document::new();
        assert!(
            decode_inline_image(
                &doc,
                params,
                &[],
                #[cfg(feature = "nvjpeg")]
                None,
                #[cfg(feature = "vaapi")]
                None,
                #[cfg(feature = "nvjpeg2k")]
                None,
                #[cfg(feature = "gpu-icc")]
                None,
                #[cfg(feature = "gpu-icc")]
                None,
            )
            .is_none()
        );
    }

    #[test]
    fn inline_image_missing_dims() {
        // No width/height → None.
        let params = b"/CS /G /BPC 8";
        let doc = lopdf::Document::new();
        assert!(
            decode_inline_image(
                &doc,
                params,
                &[0u8; 4],
                #[cfg(feature = "nvjpeg")]
                None,
                #[cfg(feature = "vaapi")]
                None,
                #[cfg(feature = "nvjpeg2k")]
                None,
                #[cfg(feature = "gpu-icc")]
                None,
                #[cfg(feature = "gpu-icc")]
                None,
            )
            .is_none()
        );
    }
}
