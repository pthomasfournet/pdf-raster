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
//! pixel area ≥ `GPU_JPEG_THRESHOLD_PX` are decoded on the GPU via NVIDIA
//! nvJPEG instead of the CPU JPEG decoder.  Pass an `NvJpegDecoder` (from
//! `gpu::nvjpeg`) to [`resolve_image`] to enable this path; pass `None` for
//! CPU-only behaviour.
//!
//! # nvJPEG2000 acceleration (`nvjpeg2k` feature)
//!
//! When the crate is built with `--features nvjpeg2k`, `JPXDecode` streams
//! with pixel area ≥ `GPU_JPEG2K_THRESHOLD_PX` are decoded on the GPU via
//! NVIDIA nvJPEG2000 instead of the CPU JPEG 2000 decoder.  Pass an
//! `NvJpeg2kDecoder` (from `gpu::nvjpeg2k`) to [`resolve_image`] to enable this path; pass `None`
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

// ── GPU parallel-Huffman JPEG decoder ────────────────────────────────────────

#[cfg(feature = "gpu-jpeg-huffman")]
use gpu::jpeg_decoder::JpegGpuDecoder;

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
/// **Currently set to `u32::MAX` — nvJPEG dispatch is disabled on consumer
/// Blackwell.**  Verified on RTX 5070 (Blackwell, one NVJPG engine) running
/// the v0.6.0 baseline (10-corpus matrix, 2026-05-06): the GPU JPEG path was
/// 5–13× *slower* than 24-thread CPU zune-jpeg on every DCT-heavy corpus.
/// The mechanism is documented in the v0.6.0 baseline spec; the short
/// version is:
///
/// - `NVJPEG_BACKEND_HARDWARE` would serialise all 24 worker threads
///   through one on-die engine (consumer Blackwell has 1 NVJPG engine,
///   not multiple as on A100/H100/Ada datacenter parts).
/// - `NVJPEG_BACKEND_GPU_HYBRID` (currently selected; see
///   `crates/gpu/src/nvjpeg.rs`) only uses GPU for Huffman decoding when
///   batch size > 50–100; for single-image-per-call workloads it falls
///   back to CPU work behind a CUDA stream-sync wrapper, paying GPU-API
///   overhead for a CPU decode that is slower than calling zune-jpeg
///   directly.
/// - The batched nvJPEG API would fit, but requires collecting images
///   across pages, decoding them in batches of ≥ 100, then routing
///   results back to the per-page renderers — a major architectural
///   change not yet done.
///
/// On a datacenter GPU with multiple NVJPG engines (A100, H100, A30),
/// the `HARDWARE` backend should win at the original 512 × 512 threshold.
/// When that hardware becomes available the threshold should drop back
/// to `262_144`, the backend selector should be revisited, and the
/// matrix re-measured.
///
/// nvJPEG2000 (`GPU_JPEG2K_THRESHOLD_PX`) is *not* affected by this:
/// JPX decode on the same matrix was 1.7× faster on GPU than CPU on
/// corpus 10, validating the original threshold for JPEG 2000.
#[cfg(any(feature = "nvjpeg", feature = "vaapi"))]
pub const GPU_JPEG_THRESHOLD_PX: u32 = u32::MAX;

/// Minimum pixel area (width × height) for GPU parallel-Huffman JPEG decode.
///
/// Only 4:4:4 baseline JPEGs with 1 or 3 components are eligible.
/// Images below this threshold always use the CPU path.
///
/// Set to `u32::MAX` until a corpus benchmark confirms a crossover point —
/// mirrors the precedent set by `GPU_JPEG_THRESHOLD_PX` for nvJPEG.
///
/// Override at runtime for benchmarking: set the `PDF_RASTER_HUFFMAN_THRESHOLD`
/// environment variable to a decimal pixel-area value (e.g. `0` = always GPU,
/// `4294967295` = never GPU).  The env var is read once, lazily, on first use.
#[cfg(feature = "gpu-jpeg-huffman")]
pub const GPU_JPEG_HUFFMAN_THRESHOLD_PX: u32 = u32::MAX;

/// Returns the effective parallel-Huffman dispatch threshold: `GPU_JPEG_HUFFMAN_THRESHOLD_PX`
/// unless `PDF_RASTER_HUFFMAN_THRESHOLD` is set in the environment.
#[cfg(feature = "gpu-jpeg-huffman")]
pub(crate) fn huffman_threshold() -> u32 {
    use std::sync::OnceLock;
    static THRESHOLD: OnceLock<u32> = OnceLock::new();
    *THRESHOLD.get_or_init(|| {
        std::env::var("PDF_RASTER_HUFFMAN_THRESHOLD")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(GPU_JPEG_HUFFMAN_THRESHOLD_PX)
    })
}

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

// ── Resolution tri-state ─────────────────────────────────────────────────────

/// Outcome of [`resolve_image`]: distinguishes a genuine decode failure from a
/// legitimately-absent resource so callers can surface errors rather than silently
/// rendering a blank region.
pub enum ImageResolution {
    /// Image decoded successfully.
    Ok(ImageDescriptor),
    /// The codec returned `None` or the stream was structurally invalid.  The
    /// page is incomplete; the message should be surfaced to the caller.
    DecodeFailed(String),
    /// The named resource is not an image (wrong `Subtype`, not present in the
    /// `XObject` dict, etc.).  The renderer should skip silently — nothing to draw.
    Absent,
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

impl ImageColorSpace {
    /// Decoded bytes per pixel after the codec produces an [`ImageDescriptor`].
    /// `Mask` is byte-expanded by the decoders (one byte per pixel, not bit-packed).
    #[must_use]
    pub const fn bytes_per_pixel(self) -> usize {
        match self {
            Self::Gray | Self::Mask => 1,
            Self::Rgb => 3,
        }
    }
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

/// Backing storage for decoded image pixels.
///
/// Two variants depending on whether the `cache` feature is on:
///
/// - [`ImageData::Cpu`] always available; produced by every CPU
///   decode path (CCITT, JBIG2, raw, JPEG fallback).
/// - `ImageData::Gpu` (feature `cache`) — `Arc<CachedDeviceImage>`
///   handed back from the device image cache.  Produced by
///   `decode_dct` when a cache is wired in; consumed by the GPU
///   blit kernel which writes transformed pixels into the page's
///   `DevicePageBuffer` without a `PCIe` round-trip.
///
/// The enum is [`#[non_exhaustive]`] so future variants (e.g. an
/// `Arc<[u8]>` shared-host variant) can be added without breaking
/// downstream `match` consumers.
///
/// Hot-path access goes through [`Self::as_cpu`] which returns
/// `Option<&[u8]>` — `Some(&[..])` for the CPU variant, `None` for the
/// GPU variant.  GPU consumers must match on the variant explicitly to
/// reach the device pointer.
#[derive(Debug)]
#[non_exhaustive]
pub enum ImageData {
    /// Host-resident pixel bytes — layout defined by [`ImageColorSpace`].
    Cpu(Vec<u8>),
    /// Device-resident image, owned by the Phase 9 cache.  The `Arc`
    /// keeps the slab alive past in-flight kernel reads even if the
    /// cache evicts; see `gpu::cache::CachedDeviceImage` docs.
    #[cfg(feature = "cache")]
    Gpu(std::sync::Arc<gpu::cache::CachedDeviceImage>),
}

impl ImageData {
    /// Borrow the host-resident pixel bytes if this is a CPU-backed image.
    /// Returns `None` for variants whose pixels live in device memory.
    #[must_use]
    pub const fn as_cpu(&self) -> Option<&[u8]> {
        match self {
            Self::Cpu(bytes) => Some(bytes.as_slice()),
            #[cfg(feature = "cache")]
            Self::Gpu(_) => None,
        }
    }

    /// Length in bytes of the underlying pixel storage.  Cheap for either
    /// variant — no host memory access required for the GPU variant.
    #[must_use]
    #[cfg_attr(
        not(feature = "cache"),
        expect(
            clippy::missing_const_for_fn,
            reason = "the cache-feature path calls a non-const cudarc method; keep one signature"
        )
    )]
    pub fn len(&self) -> usize {
        match self {
            Self::Cpu(bytes) => bytes.len(),
            #[cfg(feature = "cache")]
            #[expect(
                clippy::cast_possible_truncation,
                reason = "vram_bytes returns u64 for forward-compat; 32-bit hosts will OOM long before this matters in practice"
            )]
            Self::Gpu(cached) => cached.vram_bytes() as usize,
        }
    }

    /// Whether the underlying pixel storage is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
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
    /// Pixel storage — host or (in future) device-resident.  Layout defined
    /// by `color_space` (see module doc).
    pub data: ImageData,
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
/// Returns:
/// - [`ImageResolution::Ok`] on success,
/// - [`ImageResolution::Absent`] if the name is not present or the object is not
///   an image (legitimately nothing to draw — skip silently),
/// - [`ImageResolution::DecodeFailed`] when the codec returns `None` or the stream
///   is structurally invalid (genuine failure — surface to the caller).
#[must_use]
#[expect(
    clippy::too_many_lines,
    reason = "filter dispatch table; splitting per-filter would scatter the dispatch logic"
)]
#[cfg_attr(
    any(
        all(
            feature = "gpu-icc",
            feature = "cache",
            any(feature = "nvjpeg", feature = "vaapi", feature = "nvjpeg2k")
        ),
        all(
            feature = "nvjpeg",
            feature = "vaapi",
            feature = "nvjpeg2k",
            any(feature = "gpu-icc", feature = "cache")
        )
    ),
    expect(
        clippy::too_many_arguments,
        reason = "decoder/cache handles are feature-gated cfg-args; bundling them \
                  into a struct would force every caller to construct a partially-\
                  populated context per call"
    )
)]
pub fn resolve_image<#[cfg(feature = "gpu-jpeg-huffman")] B: gpu::backend::GpuBackend>(
    doc: &Document,
    page_dict: &Dictionary,
    name: &[u8],
    #[cfg(feature = "nvjpeg")] gpu: Option<&mut NvJpegDecoder>,
    #[cfg(feature = "vaapi")] vaapi: Option<&JpegQueueHandle>,
    #[cfg(feature = "nvjpeg2k")] gpu_j2k: Option<&mut NvJpeg2kDecoder>,
    #[cfg(feature = "gpu-jpeg-huffman")] jpeg_gpu: Option<&mut JpegGpuDecoder<B>>,
    #[cfg(feature = "gpu-icc")] gpu_ctx: Option<&GpuCtx>,
    #[cfg(feature = "gpu-icc")] clut_cache: Option<&mut IccClutCache>,
    #[cfg(feature = "cache")] image_cache: Option<&std::sync::Arc<gpu::cache::DeviceImageCache>>,
    #[cfg(feature = "cache")] doc_id: Option<gpu::cache::DocId>,
) -> ImageResolution {
    use codecs::{decode_ccitt, decode_dct, decode_jbig2, decode_jpx};
    use raw::decode_raw;
    use smask::decode_smask;

    let Some(stream_id) = xobject_id(doc, page_dict, name) else {
        return ImageResolution::Absent;
    };
    // Bind the Arc to a local so the borrow into the stream below stays alive.
    let Some(obj_arc) = doc.get_object(stream_id).ok() else {
        return ImageResolution::Absent;
    };
    let Some(stream) = obj_arc.as_ref().as_stream() else {
        return ImageResolution::Absent;
    };

    // Must be an Image subtype.
    let Some(subtype) = stream.dict.get_name(b"Subtype") else {
        return ImageResolution::Absent;
    };
    if subtype != b"Image" {
        return ImageResolution::Absent;
    }

    let Some(w_raw) = stream.dict.get_i64(b"Width") else {
        return ImageResolution::Absent;
    };
    let Some(h_raw) = stream.dict.get_i64(b"Height") else {
        return ImageResolution::Absent;
    };
    let Some((w, h)) = validated_dims(w_raw, h_raw) else {
        log::warn!("image: degenerate dimensions {w_raw}×{h_raw}, skipping");
        return ImageResolution::DecodeFailed(format!(
            "image /{}: invalid dimensions {w_raw}×{h_raw}",
            String::from_utf8_lossy(name)
        ));
    };

    let is_mask = stream.dict.get_bool(b"ImageMask").unwrap_or(false);

    let filter_obj = stream.dict.get(b"Filter");
    let filter_chain = filter_obj.map(image_filter_chain).unwrap_or_default();
    let filter = filter_obj.and_then(filter_name);

    let img_filter = ImageFilter::from_filter_str(filter.as_deref());

    // Resolve the codec input bytes.  For a chained `/Filter` (e.g.
    // `[/ASCII85Decode /CCITTFaxDecode]`) every filter before the terminal one
    // is a transport/prefilter (ASCII85, Flate, …); `decompressed_content`
    // applies those left to right and passes the terminal image codec
    // (CCITTFax/DCT/JBIG2/JPX) through unchanged, so its output is exactly the
    // byte stream the terminal codec must decode.  A single transport filter
    // (plain `FlateDecode`) is also handled here.  A pure-codec single filter
    // (e.g. just `CCITTFaxDecode`) passes through, so `decompressed_content`
    // returns the raw content unchanged and behaviour is preserved.
    //
    // Fail loudly: a chain we cannot run (unsupported prefilter, corrupt
    // transport bytes) surfaces as a decode error — never a silent blank page.
    let codec_input: std::borrow::Cow<'_, [u8]> = if filter_chain.len() > 1 {
        match stream.decompressed_content() {
            Ok(data) => std::borrow::Cow::Owned(data),
            Err(e) => {
                log::warn!("image: chained-filter prefilter decode failed: {e}");
                return ImageResolution::DecodeFailed(format!("chained filter: {e}"));
            }
        }
    } else {
        std::borrow::Cow::Borrowed(stream.content.as_slice())
    };

    let terminal_parms = terminal_decode_parms(&stream.dict, filter_chain.len());

    let opt_img: Option<ImageDescriptor> = match filter.as_deref() {
        None => decode_raw(
            doc,
            &codec_input,
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
                return ImageResolution::DecodeFailed(format!("FlateDecode: {e}"));
            }
        },
        Some("CCITTFaxDecode") => decode_ccitt(&codec_input, w, h, is_mask, terminal_parms),
        Some("DCTDecode") => {
            #[cfg(feature = "cache")]
            let cache_ctx = codecs::DctCacheCtx::from_resources(image_cache, doc_id, stream_id);
            decode_dct(
                &codec_input,
                w,
                h,
                #[cfg(feature = "nvjpeg")]
                gpu,
                #[cfg(feature = "vaapi")]
                vaapi,
                #[cfg(feature = "gpu-jpeg-huffman")]
                jpeg_gpu,
                #[cfg(feature = "gpu-icc")]
                gpu_ctx,
                #[cfg(feature = "gpu-icc")]
                clut_cache,
                #[cfg(feature = "cache")]
                cache_ctx.as_ref(),
            )
        }
        Some("JPXDecode") => {
            #[cfg(feature = "nvjpeg2k")]
            {
                decode_jpx(&codec_input, w, h, gpu_j2k)
            }
            #[cfg(not(feature = "nvjpeg2k"))]
            {
                decode_jpx(&codec_input, w, h)
            }
        }
        Some("JBIG2Decode") => decode_jbig2(doc, &codec_input, w, h, is_mask, terminal_parms),
        Some(other) => {
            log::warn!("image: unknown filter {other:?}");
            return ImageResolution::DecodeFailed(format!("unsupported filter {other:?}"));
        }
    };

    let Some(mut img) = opt_img else {
        let codec = match filter.as_deref() {
            Some("CCITTFaxDecode") => "CCITTFaxDecode decode failed",
            Some("DCTDecode") => "DCTDecode decode failed",
            Some("JPXDecode") => "JPXDecode decode failed",
            Some("JBIG2Decode") => "JBIG2Decode decode failed",
            _ => "raw/unfiltered decode failed",
        };
        return ImageResolution::DecodeFailed(codec.to_owned());
    };
    img.filter = img_filter;

    // Resolve and decode the soft mask (`SMask`), if present.
    if let Some(Object::Reference(smask_id)) = stream.dict.get(b"SMask") {
        if let Some(alpha) = decode_smask(doc, *smask_id, img.width, img.height) {
            img.smask = Some(alpha);
        } else {
            // SMask decode failed.  Treat as a soft skip: the base image pixels
            // are valid so we return them without a mask rather than propagating
            // an error — an SMask failure should not discard an otherwise-correct
            // image from the page.  The renderer will blit the image fully opaque.
            log::warn!(
                "image: SMask (object {smask_id:?}) could not be decoded — blitting image without mask"
            );
        }
    } else if let Some(mask_obj) = stream.dict.get(b"Mask") {
        // PDF §8.9.6: an explicit /Mask, distinct from /SMask.  Two forms:
        //  - /Mask N 0 R           — §8.9.6.3 stencil mask image
        //  - /Mask [min max …]     — §8.9.6.4 colour-key masking
        // SMask takes precedence (handled above); /Mask is only consulted when
        // no /SMask is present.  A decode failure is a soft skip — the base
        // image is still valid, so it is blitted fully opaque.
        match mask_obj {
            Object::Reference(mask_id) => {
                if let Some(alpha) =
                    smask::decode_explicit_mask(doc, *mask_id, img.width, img.height)
                {
                    img.smask = Some(alpha);
                } else {
                    log::warn!(
                        "image: explicit /Mask (object {mask_id:?}) could not be decoded — blitting image without mask"
                    );
                }
            }
            Object::Array(ranges) => {
                if let Some(base) = img.data.as_cpu() {
                    if let Some(alpha) = smask::colour_key_mask(
                        ranges,
                        base,
                        img.color_space.bytes_per_pixel(),
                        img.width,
                        img.height,
                    ) {
                        img.smask = Some(alpha);
                    } else {
                        log::warn!(
                            "image: colour-key /Mask malformed or mismatched — blitting image without mask"
                        );
                    }
                }
            }
            _ => {
                log::warn!("image: /Mask is neither a reference nor an array — ignoring");
            }
        }
    }

    ImageResolution::Ok(img)
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

/// Collect every filter name from a `/Filter` entry, in application order.
///
/// Accepts a bare `Name` (one filter) or a `Name` array (a filter chain, PDF
/// §7.4: filters are listed in the order they were applied during encoding, so
/// decoding applies them left to right).  Non-name array entries are skipped.
/// An empty or absent array yields an empty vector (no filter).
pub(crate) fn image_filter_chain(obj: &Object) -> Vec<Cow<'_, str>> {
    match obj {
        Object::Name(n) => vec![String::from_utf8_lossy(n)],
        Object::Array(arr) => arr
            .iter()
            .filter_map(|o| o.as_name().map(String::from_utf8_lossy))
            .collect(),
        _ => Vec::new(),
    }
}

/// Extract the *terminal* image filter from a `/Filter` entry.
///
/// Accepts a bare `Name`, a single-element `Name` array, or a multi-element
/// filter chain (e.g. `[/ASCII85Decode /CCITTFaxDecode]`).  For a chain, the
/// returned name is the LAST filter — the terminal image codec.  Earlier
/// filters in the chain are transport/prefilters (ASCII85, Flate, …) whose
/// bytes are produced by [`pdf::Stream::decompressed_content`] before the
/// terminal codec runs; see the dispatch in `resolve_image`.
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
            arr.iter()
                .rev()
                .find_map(|o| o.as_name())
                .map(String::from_utf8_lossy)
        }
        _ => None,
    }
}

/// Resolve the `/DecodeParms` entry that belongs to the *terminal* filter.
///
/// `/DecodeParms` is index-aligned to `/Filter` (PDF §7.4.1): entry `i` is the
/// parameters for filter `i`; a `null` entry or an absent key means "no
/// parameters".  When `/Filter` is a chain, the terminal codec
/// (`CCITTFaxDecode`, `JBIG2Decode`, …) must receive ITS entry — the last one —
/// not the whole array, and not a prefilter's entry.
///
/// Forms handled:
/// - `/DecodeParms` is a dict, `/Filter` is a single name → that dict.
/// - `/DecodeParms` is an array → the last array element (if a dict / non-null).
/// - absent / `null` / mismatched → `None`.
fn terminal_decode_parms(dict: &Dictionary, filter_count: usize) -> Option<&Object> {
    match dict.get(b"DecodeParms")? {
        Object::Array(arr) => {
            // Prefer the entry index-aligned to the terminal (last) filter;
            // fall back to the last array element if lengths disagree.
            let idx = filter_count.checked_sub(1).filter(|&i| i < arr.len());
            let entry = idx.map_or_else(|| arr.last(), |i| arr.get(i))?;
            match entry {
                Object::Null => None,
                other => Some(other),
            }
        }
        Object::Null => None,
        other => Some(other),
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

    // ── Chained-filter dispatch ────────────────────────────────────────────────

    #[test]
    fn filter_name_single_name() {
        let obj = Object::Name(b"FlateDecode".to_vec());
        assert_eq!(filter_name(&obj).as_deref(), Some("FlateDecode"));
    }

    #[test]
    fn filter_name_single_element_array() {
        let obj = Object::Array(vec![Object::Name(b"CCITTFaxDecode".to_vec())]);
        assert_eq!(filter_name(&obj).as_deref(), Some("CCITTFaxDecode"));
    }

    #[test]
    fn filter_name_chain_returns_terminal_codec() {
        // The scanned-book corpus case: [/ASCII85Decode /CCITTFaxDecode].
        // The terminal (last) filter is the image codec and must be dispatched
        // — previously this returned None and the page rendered blank.
        let obj = Object::Array(vec![
            Object::Name(b"ASCII85Decode".to_vec()),
            Object::Name(b"CCITTFaxDecode".to_vec()),
        ]);
        assert_eq!(filter_name(&obj).as_deref(), Some("CCITTFaxDecode"));
    }

    #[test]
    fn filter_name_empty_array_is_none() {
        assert!(filter_name(&Object::Array(vec![])).is_none());
    }

    #[test]
    fn image_filter_chain_orders_left_to_right() {
        let obj = Object::Array(vec![
            Object::Name(b"ASCII85Decode".to_vec()),
            Object::Name(b"FlateDecode".to_vec()),
        ]);
        let chain = image_filter_chain(&obj);
        assert_eq!(chain, vec!["ASCII85Decode", "FlateDecode"]);
    }

    #[test]
    fn image_filter_chain_single_name() {
        let obj = Object::Name(b"DCTDecode".to_vec());
        assert_eq!(image_filter_chain(&obj), vec!["DCTDecode"]);
    }

    #[test]
    fn terminal_decode_parms_picks_last_array_entry() {
        // [/ASCII85Decode /CCITTFaxDecode] with DecodeParms [null <</K -1>>]:
        // the CCITT codec must receive its OWN params (index 1), not null and
        // not the whole array.
        let mut ccitt = Dictionary::new();
        ccitt.set(b"K".to_vec(), Object::Integer(-1));
        let mut dict = Dictionary::new();
        dict.set(
            b"DecodeParms".to_vec(),
            Object::Array(vec![Object::Null, Object::Dictionary(ccitt)]),
        );
        let parms = terminal_decode_parms(&dict, 2).expect("terminal parms present");
        assert_eq!(
            parms
                .as_dict()
                .and_then(|d| d.get(b"K"))
                .and_then(Object::as_i64),
            Some(-1)
        );
    }

    #[test]
    fn terminal_decode_parms_single_dict() {
        let mut ccitt = Dictionary::new();
        ccitt.set(b"Columns".to_vec(), Object::Integer(2480));
        let mut dict = Dictionary::new();
        dict.set(b"DecodeParms".to_vec(), Object::Dictionary(ccitt));
        let parms = terminal_decode_parms(&dict, 1).expect("single dict");
        assert_eq!(
            parms
                .as_dict()
                .and_then(|d| d.get(b"Columns"))
                .and_then(Object::as_i64),
            Some(2480)
        );
    }

    #[test]
    fn terminal_decode_parms_null_and_absent_are_none() {
        let dict = Dictionary::new();
        assert!(terminal_decode_parms(&dict, 1).is_none());

        let mut with_null = Dictionary::new();
        with_null.set(
            b"DecodeParms".to_vec(),
            Object::Array(vec![Object::Null, Object::Null]),
        );
        assert!(terminal_decode_parms(&with_null, 2).is_none());
    }

    #[test]
    fn chained_ascii85_then_flate_roundtrip() {
        // A `[/ASCII85Decode /FlateDecode]` stream the way a PDF stores it:
        // the payload is flate-compressed, then ASCII85-encoded.  The fixture
        // bytes below are `"chained filter round-trip payload \x00\x01\x02\xff"`
        // run through zlib then Adobe ASCII85 (terminated with `~>`).
        // `decompressed_content` must undo both filters left to right and
        // recover the original payload — this is exactly the prefilter step
        // the scanned-book corpus case relies on (only the terminal codec
        // differs: CCITTFax there, treated as raw here).
        let original: &[u8] = b"chained filter round-trip payload \x00\x01\x02\xff";
        let ascii85: &[u8] = b"GaqFP8Bf:N9i4I)bUH+8;CK[@btHG9.EX2<-qLG`a\\PT-?smOA%fdE-%FY~>";

        let mut dict = Dictionary::new();
        dict.set(
            b"Filter".to_vec(),
            Object::Array(vec![
                Object::Name(b"ASCII85Decode".to_vec()),
                Object::Name(b"FlateDecode".to_vec()),
            ]),
        );
        let stream = pdf::Stream::new(dict, ascii85.to_vec());
        let decoded = stream
            .decompressed_content()
            .expect("ASCII85 then Flate must round-trip");
        assert_eq!(decoded, original);
    }
}
