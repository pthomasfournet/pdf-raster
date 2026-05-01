//! Soft-mask (`SMask`) decoding for PDF images.
//!
//! A PDF `SMask` is a grayscale image whose pixel values become the alpha channel
//! of the parent image.  This module decodes the `SMask` stream and resamples it
//! to the parent image's dimensions when they differ.

use lopdf::{Document, ObjectId};

use super::bitpack::{downsample_16bpp, expand_nbpp};
use super::codecs::{decode_ccitt, decode_jbig2};
use super::filter_name;
use super::validated_dims;

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

    // CCITTFaxDecode and JBIG2Decode return Mask-space (0x00 = paint, 0xFF = skip),
    // which must be inverted to alpha-space before use.
    let alpha: Vec<u8> = match filter.as_deref() {
        Some("CCITTFaxDecode") => {
            let parms = stream.dict.get(b"DecodeParms").ok();
            let sm_desc = decode_ccitt(stream.content.as_slice(), sm_w, sm_h, true, parms)?;
            sm_desc
                .data
                .iter()
                .map(|&v| if v == 0x00 { 0xFF } else { 0x00 })
                .collect()
        }
        Some("JBIG2Decode") => {
            let parms = stream.dict.get(b"DecodeParms").ok();
            let sm_desc = decode_jbig2(doc, stream.content.as_slice(), sm_w, sm_h, true, parms)?;
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

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Expand a 1-bit-per-pixel packed `SMask` buffer to one byte per pixel.
///
/// Bit 1 (MSB-first) = opaque (0xFF); bit 0 = transparent (0x00).
/// Truncated rows are defensively treated as all-transparent (0-bit padding).
///
/// Returns `None` if `sm_w × sm_h` overflows `usize`.
pub(super) fn expand_smask_1bpp(raw: &[u8], sm_w: u32, sm_h: u32) -> Option<Vec<u8>> {
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
            // sy * src_w + sx < src_h * src_w ≤ 65536² = 4G; usize ≥ 32 bits on any
            // target pdf_interp runs on.
            #[expect(
                clippy::cast_possible_truncation,
                reason = "index < src_h*src_w ≤ 65536² = 4G; usize ≥ 32 bits"
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
        // sw=0 means no source pixels; scaling into a non-empty dst would panic
        // on src index access — must return empty instead.
        let result = scale_smask(vec![], 0, 1, 4, 4);
        assert!(result.is_empty());
    }
}
