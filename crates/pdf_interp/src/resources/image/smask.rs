//! Soft-mask (`SMask`) decoding for PDF images.
//!
//! A PDF `SMask` is a grayscale image whose pixel values become the alpha channel
//! of the parent image.  This module decodes the `SMask` stream and resamples it
//! to the parent image's dimensions when they differ.

use pdf::{Document, ObjectId};

use super::ImageDescriptor;
use super::bitpack::{downsample_16bpp, expand_1bpp, expand_nbpp};
use super::codecs::{decode_ccitt, decode_jbig2};
use super::filter_name;
use super::validated_dims;

/// Invert a bitonal mask buffer to alpha space (`0x00` ↔ `0xFF`).
///
/// Used for `CCITTFaxDecode` / `JBIG2Decode` `SMask` streams whose decoders
/// emit Mask-space bytes (paint = 0x00, skip = 0xFF) that must be flipped to
/// alpha-space (transparent = 0x00, opaque = 0xFF) before composition.
///
/// Returns `None` when the descriptor's pixel storage is not host-resident —
/// today this never happens (CCITT and JBIG2 always decode to CPU), but the
/// graceful skip future-proofs against the Phase 9 GPU variant landing.
fn invert_bitonal_alpha(desc: &ImageDescriptor) -> Option<Vec<u8>> {
    let bytes = desc.data.as_cpu()?;
    Some(
        bytes
            .iter()
            .map(|&v| if v == 0x00 { 0xFF } else { 0x00 })
            .collect(),
    )
}

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

    // CCITTFaxDecode and JBIG2Decode return Mask-space (0x00 = paint, 0xFF = skip),
    // which must be inverted to alpha-space before use.
    //
    // For these compressed formats we use the actual decoded dimensions
    // (sm_desc.width × sm_desc.height) — not the dict-declared sm_w × sm_h —
    // as the true source size passed to scale_smask, so that a decoder that
    // reports a different output size does not produce an out-of-bounds index.
    let alpha: Vec<u8> = match filter.as_deref() {
        Some("CCITTFaxDecode") => {
            let parms = stream.dict.get(b"DecodeParms");
            let sm_desc = decode_ccitt(stream.content.as_slice(), sm_w, sm_h, true, parms)?;
            let actual_w = sm_desc.width;
            let actual_h = sm_desc.height;
            let inverted: Vec<u8> = invert_bitonal_alpha(&sm_desc)?;
            return Some(scale_smask(inverted, actual_w, actual_h, img_w, img_h));
        }
        Some("JBIG2Decode") => {
            let parms = stream.dict.get(b"DecodeParms");
            let sm_desc = decode_jbig2(doc, stream.content.as_slice(), sm_w, sm_h, true, parms)?;
            let actual_w = sm_desc.width;
            let actual_h = sm_desc.height;
            let inverted: Vec<u8> = invert_bitonal_alpha(&sm_desc)?;
            return Some(scale_smask(inverted, actual_w, actual_h, img_w, img_h));
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
    fn decode_smask_raw_bpc8_too_short_returns_none() {
        // 2×2 SMask needs 4 bytes; supplying 2 must return None, not panic.
        assert!(decode_smask_raw_bytes(&[0u8, 255], 2, 2, 8).is_none());
    }

    #[test]
    fn decode_smask_raw_bpc8_exact_length() {
        let result = decode_smask_raw_bytes(&[10u8, 20, 30, 40], 2, 2, 8).unwrap();
        assert_eq!(result, &[10, 20, 30, 40]);
    }
}
