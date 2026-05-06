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
//! | `CCITTFaxDecode` (`K>0`, Group 3 mixed 2D) | yes/no | yes (via `hayro-ccitt`) |
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

use pdf::{Dictionary, Document, Object, ObjectId};

use crate::resources::dict_ext::DictExt;

// ── GPU JPEG/JPEG2k acceleration ──────────────────────────────────────────────

#[cfg(feature = "nvjpeg")]
use gpu::nvjpeg::NvJpegDecoder;

#[cfg(feature = "nvjpeg2k")]
use gpu::nvjpeg2k::NvJpeg2kDecoder;

#[cfg(feature = "vaapi")]
use gpu::JpegQueueHandle;

// ── GPU ICC CMYK→RGB acceleration ─────────────────────────────────────────────

#[cfg(feature = "gpu-icc")]
use gpu::GpuCtx;

#[cfg(feature = "gpu-icc")]
#[path = "../icc.rs"]
pub(crate) mod icc;

#[cfg(feature = "gpu-icc")]
pub use icc::IccClutCache;

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
/// is faster.  512 × 512 = 262 144 pixels — same crossover as nvJPEG; JPEG 2000
/// decode is CPU-bound at similar pixel counts.
#[cfg(feature = "nvjpeg2k")]
pub const GPU_JPEG2K_THRESHOLD_PX: u32 = 262_144;

// ── Sub-modules ───────────────────────────────────────────────────────────────

pub(crate) mod bitpack;
pub(crate) mod codecs;
pub(crate) mod colorspace;
pub(crate) mod inline;
pub(crate) mod raw;
pub(crate) mod smask;

pub use inline::decode_inline_image;

// Visible only when cargo-fuzz sets `--cfg fuzzing`.  Not part of the public API.
// `pub(super)` codec functions cannot be re-exported across the crate boundary,
// so these thin wrappers bridge the visibility gap without widening it in normal
// builds.
#[cfg(fuzzing)]
#[doc(hidden)]
pub mod fuzz_entry {
    use pdf::{Document, Object};

    use super::ImageDescriptor;
    use super::codecs;

    pub fn decode_ccitt(
        data: &[u8],
        width: u32,
        height: u32,
        is_mask: bool,
        parms: Option<&Object>,
    ) -> Option<ImageDescriptor> {
        codecs::decode_ccitt(data, width, height, is_mask, parms)
    }

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

impl ImageFilter {
    /// Map a PDF filter name string to the matching [`ImageFilter`] variant.
    ///
    /// Unrecognised or absent names map to [`ImageFilter::Raw`].  This is the
    /// single canonical mapping used by the renderer, the inline-image decoder,
    /// and the pre-scan pass.
    #[must_use]
    pub(crate) fn from_filter_str(name: Option<&str>) -> Self {
        match name {
            Some("DCTDecode") => Self::Dct,
            Some("JPXDecode") => Self::Jpx,
            Some("CCITTFaxDecode") => Self::CcittFax,
            Some("JBIG2Decode") => Self::Jbig2,
            Some("FlateDecode") => Self::Flate,
            _ => Self::Raw,
        }
    }
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
    #[cfg(feature = "vaapi")] vaapi: Option<&JpegQueueHandle>,
    #[cfg(feature = "nvjpeg2k")] gpu_j2k: Option<&mut NvJpeg2kDecoder>,
    #[cfg(feature = "gpu-icc")] gpu_ctx: Option<&GpuCtx>,
    #[cfg(feature = "gpu-icc")] clut_cache: Option<&mut IccClutCache>,
) -> Option<ImageDescriptor> {
    use codecs::{decode_ccitt, decode_dct, decode_jbig2, decode_jpx};
    use raw::decode_raw;
    use smask::decode_smask;

    let stream_id = xobject_id(doc, page_dict, name)?;
    // Bind the Arc to a local so the borrow into the stream below stays alive.
    let obj_arc = doc.get_object(stream_id).ok()?;
    let stream = obj_arc.as_ref().as_stream()?;

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

    let filter = stream.dict.get(b"Filter").and_then(filter_name);

    let img_filter = ImageFilter::from_filter_str(filter.as_deref());

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
            #[cfg(feature = "gpu-icc")]
            clut_cache,
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
                #[cfg(feature = "gpu-icc")]
                clut_cache,
            ),
            Err(e) => {
                log::warn!("image: FlateDecode decompression failed: {e}");
                None
            }
        },
        Some("CCITTFaxDecode") => {
            let parms = stream.dict.get(b"DecodeParms");
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
            #[cfg(feature = "gpu-icc")]
            clut_cache,
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
            let parms = stream.dict.get(b"DecodeParms");
            decode_jbig2(doc, stream.content.as_slice(), w, h, is_mask, parms)
        }
        Some(other) => {
            log::warn!("image: unknown filter {other:?}");
            None
        }
    }?;
    img.filter = img_filter;

    // Resolve and decode the soft mask (`SMask`), if present.
    if let Some(Object::Reference(smask_id)) = stream.dict.get(b"SMask") {
        if let Some(alpha) = decode_smask(doc, *smask_id, img.width, img.height) {
            img.smask = Some(alpha);
        } else {
            // `SMask` is present but could not be decoded.  Blitting without a
            // mask would paint the image's colour over a large area it should not
            // cover (e.g. a solid-colour overlay that is transparent everywhere
            // the mask is zero).  Skip the image instead.
            log::warn!("image: skipping image — SMask (object {smask_id:?}) could not be decoded");
            return None;
        }
    }

    Some(img)
}

// ── Colour space convenience wrapper ─────────────────────────────────────────

/// Resolve a PDF colour space object to an [`ImageColorSpace`].
///
/// Convenience wrapper around the internal `resolve_cs` for use in the shading
/// module.  CMYK colour spaces are converted to RGB as in the image decode path.
pub(crate) fn cs_to_image_color_space(doc: &Document, cs_obj: &Object) -> ImageColorSpace {
    use colorspace::{ResolvedCs, resolve_cs};
    match resolve_cs(doc, cs_obj) {
        ResolvedCs::Gray => ImageColorSpace::Gray,
        ResolvedCs::Rgb | ResolvedCs::Cmyk => ImageColorSpace::Rgb,
    }
}

// ── Shared helpers (used by multiple sub-modules) ─────────────────────────────

/// Validate raw `i64` image dimensions and cast them to `u32`.
///
/// Returns `None` (caller logs and propagates) if either dimension is ≤ 0 or
/// exceeds 65536.  The 65536 cap keeps `w × h` within `usize` on 64-bit targets
/// (65536² = 4 294 967 296, which fits in `u64` / 64-bit `usize` but not in
/// `u32`).  Each decoder is responsible for checking that the final allocation
/// does not exceed available memory.
///
/// The cast is safe: after the range check, each dimension is in [1, 65536]
/// which fits in `u32` without sign loss or truncation.
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

/// Extract the filter name from a `Filter` entry.
///
/// Accepts either a bare `Name` or a single-element `Name` array.  PDF allows
/// chained filters as a multi-element array; chained filters are not supported
/// here — a warning is emitted and `None` is returned so the caller can skip
/// the image gracefully rather than trying to decode garbled data.
///
/// An empty array (`Filter = []`) is treated as no filter (returns `None`)
/// and a debug message is emitted, matching the behaviour of an absent key.
pub(crate) fn filter_name(obj: &Object) -> Option<Cow<'_, str>> {
    match obj {
        Object::Name(n) => Some(String::from_utf8_lossy(n)),
        Object::Array(arr) => {
            if arr.is_empty() {
                log::debug!("image: Filter is an empty array — treating as no filter");
                return None;
            }
            if arr.len() > 1 {
                log::warn!(
                    "image: chained filters ({} filters in array) not supported — skipping image",
                    arr.len()
                );
                return None;
            }
            arr.first()
                .and_then(|o| o.as_name())
                .map(String::from_utf8_lossy)
        }
        _ => None,
    }
}

/// Resolve the `XObject` resource named `name` to its stream `ObjectId`.
fn xobject_id(doc: &Document, page_dict: &Dictionary, name: &[u8]) -> Option<ObjectId> {
    let res = super::resolve_dict(doc, page_dict.get(b"Resources")?)?;
    let xobj = super::resolve_dict(doc, res.get(b"XObject")?)?;
    match xobj.get(name)? {
        Object::Reference(id) => Some(*id),
        _ => None,
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

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
}
