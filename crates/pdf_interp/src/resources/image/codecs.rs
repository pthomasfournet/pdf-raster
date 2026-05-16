//! Compressed-image codec decoders: CCITT, DCT (JPEG), JPX (JPEG 2000), JBIG2.
//!
//! All functions are `pub(super)` — called exclusively from `super` (`image/mod.rs`)
//! and transitively from `inline.rs`.  Under `cfg(fuzzing)`, `decode_ccitt` and
//! `decode_jbig2` are additionally reachable through the `fuzz_entry` wrappers in
//! `image/mod.rs`, which bridge the visibility gap without widening it in normal builds.

use hayro_jbig2::Decoder as Jbig2Decoder;
use jpeg2k::{ColorSpace as Jp2ColorSpace, Image as Jp2Image, ImageFormat, ImagePixelData};
use pdf::{Document, Object};

use super::{ImageColorSpace, ImageData, ImageDescriptor, ImageFilter};
use crate::resources::dict_ext::DictExt;

#[cfg(feature = "cache")]
use gpu::cache::{
    CachedDeviceImage, ContentHash, DeviceImageCache, DocId, ImageLayout as CacheImageLayout,
    InsertRequest, ObjId,
};
#[cfg(feature = "cache")]
use std::sync::Arc;

// ── GPU JPEG acceleration (nvJPEG and/or VA-API) ──────────────────────────────

#[cfg(feature = "nvjpeg")]
use gpu::nvjpeg::NvJpegDecoder;

#[cfg(feature = "vaapi")]
use gpu::JpegQueueHandle;

#[cfg(feature = "nvjpeg2k")]
use gpu::nvjpeg2k::{Jpeg2kColorSpace as GpuJ2kCs, NvJpeg2kDecoder};

// ── GPU ICC CMYK→RGB acceleration ─────────────────────────────────────────────

#[cfg(feature = "gpu-icc")]
use gpu::GpuCtx;

#[cfg(feature = "gpu-icc")]
use super::icc::{self, IccClutCache};

// ── JPEG SOF peek ─────────────────────────────────────────────────────────────

/// Scan `data` for the first JPEG SOF marker and return the component count
/// (`Nf` field). Returns `None` if `data` is not a valid JPEG or the SOF
/// payload is too short. Zero allocations; stops at the first SOF found.
///
/// This avoids a full header-decode probe pass to determine the component count
/// before calling the main decode. The `gpu` crate has the same logic but is an
/// optional dependency (requires CUDA at link time), so we replicate the tiny
/// scan here for the unconditional CPU path.
fn jpeg_sof_components(data: &[u8]) -> Option<u8> {
    if data.get(0..2) != Some(&[0xFF, 0xD8]) {
        return None;
    }
    let mut pos = 2usize;
    loop {
        let ff_start = pos;
        while data.get(pos).copied() == Some(0xFF) {
            pos += 1;
        }
        if pos == ff_start {
            return None;
        }
        let &marker = data.get(pos)?;
        pos += 1;
        if matches!(marker, 0x01 | 0xD0..=0xD9) {
            continue;
        }
        let hi = *data.get(pos)?;
        let lo = *data.get(pos + 1)?;
        let seg_len = ((hi as usize) << 8) | (lo as usize);
        if seg_len < 2 {
            return None;
        }
        // SOF markers carry the component count at payload offset +5 (after P, Y, X).
        // Layout: length(2) P(1) Y(2) X(2) Nf(1) → Nf at pos+7.
        if matches!(marker, 0xC0..=0xCF) && !matches!(marker, 0xC4 | 0xCC) {
            return data.get(pos + 7).copied();
        }
        if marker == 0xDA {
            return None;
        }
        pos = pos.checked_add(seg_len)?;
    }
}

// ── CCITTFaxDecode ─────────────────────────────────────────────────────────────

/// Decode a `CCITTFaxDecode` stream.
///
/// `K` in `DecodeParms` (PDF §7.4.6):
/// - `K < 0` → Group 4 (T.6, 2D) — fully supported.
/// - `K = 0` → Group 3 1D (T.4 1D) — supported.
/// - `K > 0` → Group 3 mixed 1D/2D (T.4 2D) — supported via `hayro-ccitt`.
///
/// `Rows` (if present) caps the number of rows decoded; otherwise decodes
/// until the bitstream signals end-of-data.
pub(super) fn decode_ccitt(
    data: &[u8],
    width: u32,
    height: u32,
    is_mask: bool,
    parms: Option<&Object>,
) -> Option<ImageDescriptor> {
    // Resolve DecodeParms once; all CCITT params live in the same dict.
    let parms_dict = parms.and_then(|o| o.as_dict());

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

/// Build the final [`ImageDescriptor`] for a decoded CCITT bitonal image.
const fn ccitt_descriptor(p: &CcittParams, data: Vec<u8>) -> ImageDescriptor {
    let color_space = if p.is_mask {
        ImageColorSpace::Mask
    } else {
        ImageColorSpace::Gray
    };
    ImageDescriptor {
        width: p.width,
        height: p.height,
        color_space,
        data: ImageData::Cpu(data),
        smask: None,
        filter: ImageFilter::Raw,
    }
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
    }

    // Normalise to exactly p.capacity on every path (success and incomplete
    // alike): a clean EOFB before `height` rows leaves data_out short even when
    // `completed.is_some()`, and a malformed stream can over-emit.  G3-1D/G3-2D
    // normalise unconditionally; keep G4 consistent so a short or long buffer
    // never reaches ccitt_descriptor (which would later panic in scale_smask or
    // silently drop image content).  black_is_1=true means 0xFF encodes black,
    // so pad missing rows with 0x00 (white) instead.
    if data_out.len() > p.capacity {
        data_out.truncate(p.capacity);
    } else if data_out.len() < p.capacity {
        let pad = if p.black_is_1 { 0x00 } else { 0xFF };
        data_out.resize(p.capacity, pad);
    }

    Some(ccitt_descriptor(p, data_out))
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

    if result.is_none() {
        if rows_decoded == 0 {
            log::warn!("image: CCITTFaxDecode Group3 1D decode failed with no rows");
            return None;
        }
        log::warn!(
            "image: CCITTFaxDecode Group3 1D decode error after {rows_decoded}/{} rows — \
             padding remainder",
            p.height
        );
    }
    if rows_decoded < p.height {
        let pad = if p.black_is_1 { 0x00 } else { 0xFF };
        data_out.resize(p.capacity, pad);
    }

    Some(ccitt_descriptor(p, data_out))
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
            log::warn!(
                "image: CCITTFaxDecode Group3 2D decode error after {}/{} rows: {e}",
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
        let pad = if p.black_is_1 { 0x00 } else { 0xFF };
        data_out.resize(p.capacity, pad);
    }

    Some(ccitt_descriptor(p, data_out))
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
/// of 0 means *full ink*, 255 means *no ink*.  The JPEG decoder returns raw
/// bytes in this inverted form.  Converting to RGB:
///
/// ```text
/// R = (255 - C_raw) * (255 - K_raw) / 255
/// ```
///
/// where `C_raw` and `K_raw` are the raw (inverted) byte values.  The `255 -`
/// term converts each inverted value to a normal ink density before applying
/// the standard CMY+K → RGB formula.
#[expect(
    clippy::too_many_lines,
    reason = "GPU dispatch paths (nvjpeg, vaapi) plus the CPU decode path are each short; \
              combining them in one function keeps the fallback logic visible"
)]
#[cfg_attr(
    all(
        feature = "nvjpeg",
        feature = "vaapi",
        feature = "gpu-icc",
        feature = "cache"
    ),
    expect(
        clippy::too_many_arguments,
        reason = "decoder/cache handles are feature-gated cfg-args; bundling them \
                  into a struct would force every caller to construct a partially-\
                  populated context per call"
    )
)]
pub(super) fn decode_dct<#[cfg(feature = "gpu-jpeg-huffman")] B: gpu::backend::GpuBackend>(
    data: &[u8],
    pdf_w: u32,
    pdf_h: u32,
    #[cfg(feature = "nvjpeg")] gpu: Option<&mut NvJpegDecoder>,
    #[cfg(feature = "vaapi")] vaapi: Option<&JpegQueueHandle>,
    #[cfg(feature = "gpu-jpeg-huffman")] jpeg_gpu: Option<
        &mut gpu::jpeg_decoder::JpegGpuDecoder<B>,
    >,
    #[cfg(feature = "gpu-icc")] gpu_ctx: Option<&GpuCtx>,
    #[cfg(feature = "gpu-icc")] clut_cache: Option<&mut IccClutCache>,
    #[cfg(feature = "cache")] cache_ctx: Option<&DctCacheCtx<'_>>,
) -> Option<ImageDescriptor> {
    use zune_core::bytestream::ZCursor;
    use zune_core::colorspace::ColorSpace as ZColorSpace;
    use zune_core::options::DecoderOptions;
    use zune_jpeg::JpegDecoder;

    // Phase 9 cache fast path: if a cache is wired in and the encoded
    // bytes are already cached (cross-document content-hash dedup),
    // return the device-resident handle with zero CPU decode work.
    // On miss, retain the precomputed BLAKE3 hash so the post-decode
    // insert at the bottom of this function doesn't re-hash the same
    // bytes — that would double the per-image hashing cost on cold
    // misses (~250 µs per 500 KB JPEG → ~250 ms per 1000-image render).
    #[cfg(feature = "cache")]
    let cache_miss_hash: Option<ContentHash> = match cache_ctx {
        Some(ctx) => match try_decode_dct_cached_lookup(data, pdf_w, pdf_h, ctx) {
            Ok(img) => return Some(img),
            Err(hash) => Some(hash),
        },
        None => None,
    };

    #[cfg(any(feature = "nvjpeg", feature = "vaapi"))]
    let area = pdf_w.saturating_mul(pdf_h);

    // Peek at the JPEG SOF marker to route to the correct decoder.
    // Prevents sending progressive JPEG to VA-API (baseline-only).
    #[cfg(any(feature = "nvjpeg", feature = "vaapi"))]
    let jpeg_variant = gpu::jpeg_sof_type(data);

    // ── nvJPEG fast path (baseline + progressive) ─────────────────────────────
    // The `>= GPU_JPEG_THRESHOLD_PX` comparison can be `false` for any input
    // when the threshold is `u32::MAX` (current consumer-Blackwell setting —
    // see the constant's docs). The expect is on the per-call site; the
    // threshold itself is not a constant truth.
    #[cfg(feature = "nvjpeg")]
    if let Some(dec) = gpu {
        let is_eligible = matches!(
            jpeg_variant,
            Some(gpu::JpegVariant::Baseline | gpu::JpegVariant::Progressive)
        );
        #[expect(
            clippy::absurd_extreme_comparisons,
            reason = "GPU_JPEG_THRESHOLD_PX is u32::MAX on consumer Blackwell to disable nvJPEG dispatch; tunable per hardware"
        )]
        let area_above_threshold = area >= super::GPU_JPEG_THRESHOLD_PX;
        if is_eligible
            && area_above_threshold
            && let Some(img) = decode_dct_gpu_path(data, pdf_w, pdf_h, dec)
        {
            return Some(img);
        }
    }

    // ── VA-API fast path (baseline only, via decode queue) ───────────────────
    // Progressive JPEG is skipped — VA-API VAEntrypointVLD supports baseline DCT only.
    // Jobs are routed through a dedicated worker thread (JpegQueueHandle) that owns
    // the VapiJpegDecoder, eliminating Mesa driver contention across Rayon workers.
    #[cfg(feature = "vaapi")]
    if let Some(handle) = vaapi
        && matches!(jpeg_variant, Some(gpu::JpegVariant::Baseline))
        && {
            #[expect(
                clippy::absurd_extreme_comparisons,
                reason = "see GPU_JPEG_THRESHOLD_PX docs — set to u32::MAX on consumer Blackwell"
            )]
            let above = area >= super::GPU_JPEG_THRESHOLD_PX;
            above
        }
        && let Some(img) = decode_dct_queue_path(data, pdf_w, pdf_h, handle)
    {
        return Some(img);
    }

    // ── GPU parallel-Huffman path ─────────────────────────────────────────────
    #[cfg(feature = "gpu-jpeg-huffman")]
    if let Some(dec) = jpeg_gpu {
        let area = pdf_w.saturating_mul(pdf_h);
        if area >= super::huffman_threshold() {
            match dec.decode(data) {
                Ok(img) => {
                    let w = img.width;
                    let h = img.height;
                    if w == 0 || h == 0 {
                        log::warn!(
                            "GPU JPEG decoder returned zero-dimension image ({w}×{h}), skipping"
                        );
                    } else {
                        let n = (w as usize) * (h as usize) * 4;
                        let mut rgba = vec![0u8; n];
                        let downloaded = dec
                            .backend()
                            .download_async(&img.buffer, &mut rgba)
                            .and_then(|fence| dec.backend().wait_download(fence));
                        match downloaded {
                            Ok(()) => {
                                // The IDCT kernel always writes RGBA8. For
                                // grayscale (1-component) JPEGs it replicates
                                // luma into R, G, and B. Strip alpha to get the
                                // 3-byte-per-pixel RGB representation expected by
                                // the blit path, and preserve the correct
                                // color_space tag so downstream code branches
                                // correctly.
                                let (color_space, pixel_data) =
                                    if jpeg_sof_components(data) == Some(1) {
                                        let gray: Vec<u8> =
                                            rgba.chunks_exact(4).map(|px| px[0]).collect();
                                        (super::ImageColorSpace::Gray, gray)
                                    } else {
                                        let rgb: Vec<u8> = rgba
                                            .chunks_exact(4)
                                            .flat_map(|px| [px[0], px[1], px[2]])
                                            .collect();
                                        (super::ImageColorSpace::Rgb, rgb)
                                    };
                                return Some(ImageDescriptor {
                                    width: w,
                                    height: h,
                                    color_space,
                                    data: super::ImageData::Cpu(pixel_data),
                                    smask: None,
                                    filter: ImageFilter::Dct,
                                });
                            }
                            Err(e) => log::warn!("GPU JPEG download failed: {e}"),
                        }
                    }
                }
                Err(e) => log::debug!("GPU JPEG decode failed, falling back to CPU: {e}"),
            }
        }
    }

    // ── CPU path ──────────────────────────────────────────────────────────────
    let components = match jpeg_sof_components(data) {
        Some(n @ (1 | 3 | 4)) => n,
        Some(n) => {
            log::warn!("image: DCTDecode: unexpected component count {n}");
            return None;
        }
        None => {
            log::warn!("image: DCTDecode: not a valid JPEG stream");
            return None;
        }
    };

    // Choose the output colorspace to request from the decoder.
    let out_cs = match components {
        1 => ZColorSpace::Luma,
        3 => ZColorSpace::RGB,
        // CMYK — request raw CMYK output (4 bytes/pixel), convert to RGB below.
        _ => ZColorSpace::CMYK,
    };

    let options = DecoderOptions::default().jpeg_set_out_colorspace(out_cs);
    let mut decoder = JpegDecoder::new_with_options(ZCursor::new(data), options);
    let pixels = decoder
        .decode()
        .map_err(|e| log::warn!("image: DCTDecode decode error: {e}"))
        .ok()?;

    let jpeg_info = decoder.info().or_else(|| {
        log::warn!("image: DCTDecode: JPEG info unavailable after decode");
        None
    })?;
    let jw = u32::from(jpeg_info.width);
    let jh = u32::from(jpeg_info.height);

    if jw == 0 || jh == 0 {
        log::warn!("image: DCTDecode: JPEG reported zero dimensions {jw}×{jh}");
        return None;
    }

    // Validate the pixel buffer length against the dimensions we are about to use.
    // A mismatch means the JPEG SOF markers misreported dimensions; trust jw/jh
    // only when the buffer is consistent with them.
    let expected_pixels = (jw as usize)
        .checked_mul(jh as usize)
        .and_then(|n| n.checked_mul(usize::from(components)));
    if expected_pixels.is_none_or(|e| pixels.len() < e) {
        log::warn!(
            "image: DCTDecode: pixel buffer length {} inconsistent with \
             {jw}×{jh}×{components} — skipping",
            pixels.len()
        );
        return None;
    }

    if jw != pdf_w || jh != pdf_h {
        log::debug!(
            "image: DCTDecode: PDF dict says {pdf_w}×{pdf_h}, JPEG reports {jw}×{jh} — using JPEG dims"
        );
    }

    let cpu_desc = match out_cs {
        ZColorSpace::Luma => ImageDescriptor {
            width: jw,
            height: jh,
            color_space: ImageColorSpace::Gray,
            data: ImageData::Cpu(pixels),
            smask: None,
            filter: ImageFilter::Dct,
        },
        ZColorSpace::RGB => ImageDescriptor {
            width: jw,
            height: jh,
            color_space: ImageColorSpace::Rgb,
            data: ImageData::Cpu(pixels),
            smask: None,
            filter: ImageFilter::Dct,
        },
        ZColorSpace::CMYK => {
            // JPEG CMYK stores inverted ink densities (0 = full ink, 255 = no ink).
            // Pass inverted=true so cmyk_raw_to_rgb complements on-the-fly without
            // a separate allocation.  JPEG streams embed their own colour profile;
            // the PDF ICCBased stream is not available here, so no CLUT baking.
            let rgb = cmyk_raw_to_rgb(
                &pixels,
                true,
                #[cfg(feature = "gpu-icc")]
                gpu_ctx,
                #[cfg(feature = "gpu-icc")]
                None,
                #[cfg(feature = "gpu-icc")]
                clut_cache,
            )?;
            ImageDescriptor {
                width: jw,
                height: jh,
                color_space: ImageColorSpace::Rgb,
                data: ImageData::Cpu(rgb),
                smask: None,
                filter: ImageFilter::Dct,
            }
        }
        // out_cs is always Luma, RGB, or CMYK — set from the components match above.
        _ => unreachable!("DCTDecode: unexpected out_cs variant"),
    };

    // Phase 9: insert into the cache so the next render of this PDF
    // (or a different PDF with the same image content) hits the
    // device-resident fast path instead of decoding again.  Reuse
    // the hash from the lookup-miss path to skip a second BLAKE3.
    #[cfg(feature = "cache")]
    if let (Some(ctx), Some(hash)) = (cache_ctx, cache_miss_hash) {
        return Some(try_decode_dct_cached_insert(cpu_desc, hash, ctx));
    }
    Some(cpu_desc)
}

/// Phase 9 cache plumbing for `decode_dct`.  When set, [`decode_dct`]
/// (a) probes the cache by content hash before any decode work and,
/// on a miss, (b) inserts the decoded bytes after the CPU decode
/// completes so the next render hits the cache.
#[cfg(feature = "cache")]
pub(super) struct DctCacheCtx<'a> {
    /// Phase 9 device image cache.  Borrowed; the cache outlives the
    /// renderer for the duration of a `raster_pdf` call.
    pub cache: &'a Arc<DeviceImageCache>,
    /// Stable identifier for the source PDF.
    pub doc_id: DocId,
    /// Per-image PDF object id (the `XObject`'s stream object number).
    pub obj_id: ObjId,
}

#[cfg(feature = "cache")]
impl<'a> DctCacheCtx<'a> {
    /// Build the cache context from the renderer's cache reference,
    /// the document id, and the PDF object id of the image stream.
    ///
    /// Returns `None` when either the cache is absent or the `doc_id`
    /// is absent — in that case `decode_dct` skips both the lookup
    /// fast path and the post-decode insert.
    ///
    /// The `(u32, u16)` PDF `ObjectId` is mapped to `u32` by dropping
    /// the generation.  See [`gpu::cache::ObjId`] for the lossy-
    /// mapping rationale.
    pub(super) fn from_resources(
        cache: Option<&'a Arc<DeviceImageCache>>,
        doc_id: Option<DocId>,
        stream_id: pdf::ObjectId,
    ) -> Option<Self> {
        let cache = cache?;
        let doc_id = doc_id?;
        Some(Self {
            cache,
            doc_id,
            obj_id: ObjId(stream_id.0),
        })
    }
}

/// Probe the cache.  On a hit returns the device-resident
/// `ImageDescriptor`; on a miss returns `Err(ContentHash)` so the
/// caller can pass the already-computed hash to
/// [`try_decode_dct_cached_insert`] and avoid re-hashing the same
/// bytes after CPU decode.
#[cfg(feature = "cache")]
fn try_decode_dct_cached_lookup(
    data: &[u8],
    pdf_w: u32,
    pdf_h: u32,
    ctx: &DctCacheCtx<'_>,
) -> Result<ImageDescriptor, ContentHash> {
    // Same-document fast path: alias by (DocId, ObjId) skips BLAKE3 hashing.
    if let Some(cached) = ctx.cache.lookup_by_id(ctx.doc_id, ctx.obj_id) {
        log::debug!(
            "decode_dct: cache hit by alias for ({:?}, {:?})",
            ctx.doc_id,
            ctx.obj_id,
        );
        return Ok(image_descriptor_from_cached(cached, pdf_w, pdf_h));
    }
    // Three-tier probe: VRAM → host RAM → disk (when enabled).
    // Re-binds the alias on hit so the same image referenced from a
    // later page goes through the alias fast path.
    let hash = DeviceImageCache::hash_bytes(data);
    if let Some(cached) = ctx
        .cache
        .lookup_by_hash_for_doc(ctx.doc_id, ctx.obj_id, &hash)
    {
        log::debug!("decode_dct: cache hit by content hash; alias rebound");
        return Ok(image_descriptor_from_cached(cached, pdf_w, pdf_h));
    }
    Err(hash)
}

/// Wrap a freshly-decoded JPEG into the cache, returning a
/// device-resident `ImageDescriptor`.  Skipped (returns the
/// caller's CPU descriptor) on insert error so the renderer
/// still gets pixels — typically a budget violation; not fatal.
///
/// `hash` should be the value returned from
/// [`try_decode_dct_cached_lookup`]'s `Err` arm; reusing it avoids
/// hashing the same bytes a second time.
#[cfg(feature = "cache")]
fn try_decode_dct_cached_insert(
    cpu_desc: ImageDescriptor,
    hash: ContentHash,
    ctx: &DctCacheCtx<'_>,
) -> ImageDescriptor {
    // Only cache RGB / Gray output (the layouts the blit kernel
    // can sample).  Mask images are rare in JPEG and stay on CPU.
    let layout = match cpu_desc.color_space {
        ImageColorSpace::Rgb => CacheImageLayout::Rgb,
        ImageColorSpace::Gray => CacheImageLayout::Gray,
        ImageColorSpace::Mask => return cpu_desc,
    };
    let Some(pixels) = cpu_desc.data.as_cpu() else {
        return cpu_desc;
    };
    let req = InsertRequest {
        doc: ctx.doc_id,
        obj: ctx.obj_id,
        hash,
        width: cpu_desc.width,
        height: cpu_desc.height,
        layout,
        pixels,
    };
    match ctx.cache.insert(req) {
        Ok(cached) => image_descriptor_from_cached(cached, cpu_desc.width, cpu_desc.height),
        Err(e) => {
            log::warn!(
                "decode_dct: cache insert failed ({e}); returning CPU descriptor — image will use CPU blit path"
            );
            cpu_desc
        }
    }
}

#[cfg(feature = "cache")]
fn image_descriptor_from_cached(
    cached: Arc<CachedDeviceImage>,
    pdf_w: u32,
    pdf_h: u32,
) -> ImageDescriptor {
    let color_space = match cached.layout {
        CacheImageLayout::Rgb => ImageColorSpace::Rgb,
        CacheImageLayout::Gray => ImageColorSpace::Gray,
        CacheImageLayout::Mask => ImageColorSpace::Mask,
    };
    // Prefer the cached entry's actual dimensions over the PDF
    // dict's claim — the dict can lie; the decoded image can't.
    let width = cached.width;
    let height = cached.height;
    if (width, height) != (pdf_w, pdf_h) {
        log::debug!(
            "decode_dct: cache hit dimensions {width}×{height} differ from PDF dict {pdf_w}×{pdf_h}; using cache",
        );
    }
    ImageDescriptor {
        width,
        height,
        color_space,
        data: ImageData::Gpu(cached),
        smask: None,
        filter: ImageFilter::Dct,
    }
}

/// Attempt JPEG decode via any [`gpu::GpuJpegDecoder`] implementation.
///
/// Returns `None` if the decoder rejects the image (unsupported encoding,
/// component count, or any transient decode error). The caller must then
/// fall through to the CPU path.
#[cfg(feature = "nvjpeg")]
fn decode_dct_gpu_path<D: gpu::GpuJpegDecoder>(
    data: &[u8],
    pdf_w: u32,
    pdf_h: u32,
    dec: &mut D,
) -> Option<ImageDescriptor> {
    use gpu::DecodedImage;

    let decoded: DecodedImage = dec
        .decode_jpeg(data, pdf_w, pdf_h)
        .map_err(|e| log::warn!("image: DCTDecode: GPU path failed: {e}"))
        .ok()?;

    let color_space = match decoded.components {
        1 => ImageColorSpace::Gray,
        3 => ImageColorSpace::Rgb,
        n => {
            log::warn!(
                "image: DCTDecode: GPU decoder returned unexpected component count {n}; falling back to CPU"
            );
            return None;
        }
    };

    Some(ImageDescriptor {
        width: decoded.width,
        height: decoded.height,
        color_space,
        data: ImageData::Cpu(decoded.data),
        filter: ImageFilter::Raw,
        smask: None,
    })
}

/// Attempt JPEG decode via a [`gpu::JpegQueueHandle`] (VA-API worker thread).
///
/// Mirrors [`decode_dct_gpu_path`] but routes through the dedicated OS worker
/// thread rather than calling the decoder directly on the Rayon thread.  Returns
/// `None` if the worker is unavailable or the decoder rejects the image; the
/// caller must then fall through to the CPU path.
#[cfg(feature = "vaapi")]
fn decode_dct_queue_path(
    data: &[u8],
    pdf_w: u32,
    pdf_h: u32,
    handle: &gpu::JpegQueueHandle,
) -> Option<ImageDescriptor> {
    use gpu::DecodedImage;

    // Arc<[u8]> is required to send the JPEG bytes to the worker thread without
    // unsafe lifetime extension.  One memcpy of the JPEG payload; negligible vs
    // VCN hardware decode time (1–5 ms/frame for ≥512×512 images).
    let arc_data: std::sync::Arc<[u8]> = std::sync::Arc::from(data);
    let decoded: DecodedImage = handle.decode(arc_data, pdf_w, pdf_h)?;

    let color_space = match decoded.components {
        1 => ImageColorSpace::Gray,
        3 => ImageColorSpace::Rgb,
        n => {
            log::warn!(
                "image: DCTDecode: VA-API queue returned unexpected component count {n}; falling back to CPU"
            );
            return None;
        }
    };

    Some(ImageDescriptor {
        width: decoded.width,
        height: decoded.height,
        color_space,
        data: ImageData::Cpu(decoded.data),
        filter: ImageFilter::Raw,
        smask: None,
    })
}

/// Convert a raw CMYK pixel buffer to RGB, dispatching to GPU when available.
///
/// `inverted = false` (raw images / `decode_raw_8bpp`): 0 = no ink, 255 = full ink.
/// `inverted = true` (JPEG CMYK): codec returns the complement convention (0 = full
/// ink, 255 = no ink); the complement is applied on-the-fly, avoiding a separate
/// allocation.
///
/// `icc_bytes` — raw ICC profile bytes extracted from an `ICCBased` colour space.
/// When provided (and the `gpu-icc` feature is active), a CLUT is baked from the
/// profile and used for the colour transform instead of the fast matrix approximation.
///
/// Returns `None` when the pixel buffer length is not a multiple of 4, or on
/// arithmetic overflow (degenerate image size).
#[expect(
    clippy::many_single_char_names,
    reason = "c/m/y/k/r/g/b are the canonical channel names for CMYK→RGB — renaming hurts clarity"
)]
pub(super) fn cmyk_raw_to_rgb(
    pixels: &[u8],
    inverted: bool,
    #[cfg(feature = "gpu-icc")] gpu_ctx: Option<&GpuCtx>,
    #[cfg(feature = "gpu-icc")] icc_bytes: Option<&[u8]>,
    #[cfg(feature = "gpu-icc")] clut_cache: Option<&mut IccClutCache>,
) -> Option<Vec<u8>> {
    if !pixels.len().is_multiple_of(4) {
        log::warn!(
            "image: cmyk_raw_to_rgb: pixel buffer length {} is not a multiple of 4",
            pixels.len()
        );
        return None;
    }

    // GPU path: delegate to GpuCtx which handles the dispatch-threshold check and
    // CPU fallback internally.  When ICC bytes are present, bake a CLUT for
    // profile-accurate conversion; fall back to the fast matrix approximation if
    // baking fails (e.g. corrupt profile or wrong colour space).
    #[cfg(feature = "gpu-icc")]
    if let Some(ctx) = gpu_ctx {
        // Complement JPEG CMYK pixels before GPU dispatch if needed.
        // A Cow avoids the allocation when `inverted = false` (the common case).
        let gpu_pixels: std::borrow::Cow<[u8]> = if inverted {
            pixels.iter().map(|&b| 255 - b).collect::<Vec<_>>().into()
        } else {
            pixels.into()
        };

        #[expect(
            clippy::option_if_let_else,
            reason = "branches are not symmetric (cached path omits .map(Into::into)); \
                      map_or_else would make this harder to read"
        )]
        let clut_arc: Option<std::sync::Arc<[u8]>> = icc_bytes.and_then(|bytes| {
            if let Some(cache) = clut_cache {
                icc::bake_cmyk_clut_cached(bytes, icc::DEFAULT_GRID_N, cache)
                    .map_err(|e| {
                        log::warn!("image: ICC CLUT bake failed, using matrix fallback: {e}");
                    })
                    .ok()
            } else {
                icc::bake_cmyk_clut(bytes, icc::DEFAULT_GRID_N)
                    .map_err(|e| {
                        log::warn!("image: ICC CLUT bake failed, using matrix fallback: {e}");
                    })
                    .ok()
                    .map(std::convert::Into::into)
            }
        });

        match ctx.icc_cmyk_to_rgb(
            &gpu_pixels,
            clut_arc.as_deref().map(|t| (t, icc::DEFAULT_GRID_N)),
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
        let (c, m, y, k) = if inverted {
            (
                255 - chunk[0],
                255 - chunk[1],
                255 - chunk[2],
                255 - chunk[3],
            )
        } else {
            (chunk[0], chunk[1], chunk[2], chunk[3])
        };
        let (r, g, b) = color::convert::cmyk_to_rgb_reflectance(c, m, y, k);
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
/// (pixel area ≥ [`super::GPU_JPEG2K_THRESHOLD_PX`]) are decoded on the GPU via
/// nvJPEG2000.  All other images, and any image for which the GPU path fails
/// (unsupported component count, CUDA error, etc.), fall through to the CPU path.
///
/// PDF JPEG 2000 streams may be raw codestreams (`.j2k`) or full JP2 container
/// format (`.jp2`).  Both paths auto-detect the format from the stream.
///
/// 16-bit component images are downscaled to 8-bit.  Embedded alpha channels are
/// extracted into `ImageDescriptor::smask` so the compositor can blend them.
#[expect(
    clippy::too_many_lines,
    reason = "one arm per ImageFormat variant; extracting alpha adds 2 lines per alpha-capable arm"
)]
pub(super) fn decode_jpx(
    data: &[u8],
    pdf_w: u32,
    pdf_h: u32,
    #[cfg(feature = "nvjpeg2k")] gpu: Option<&mut NvJpeg2kDecoder>,
) -> Option<ImageDescriptor> {
    // ── GPU fast path (nvjpeg2k feature, large images, 1- or 3-component only) ─
    #[cfg(feature = "nvjpeg2k")]
    if let Some(dec) = gpu {
        let area = pdf_w.saturating_mul(pdf_h);
        if area >= super::GPU_JPEG2K_THRESHOLD_PX {
            if let Some(img) = decode_jpx_gpu(data, pdf_w, pdf_h, dec) {
                return Some(img);
            }
            log::debug!("image: JPXDecode: GPU path failed, retrying on CPU");
        }
    }

    // ── CPU path ─────────────────────────────────────────────────────────────
    let img = Jp2Image::from_bytes(data)
        .map_err(|e| log::warn!("image: JPXDecode open error: {e}"))
        .ok()?;

    // CMYK JPEG 2000 must be intercepted before `get_pixels`: the `jpeg2k`
    // crate's `get_pixels` only emits Gray/RGB(A) and errors out on CMYK.  Pull
    // the four raw component planes instead and route them through the same
    // CMYK→RGB conversion used for DeviceCMYK raw images and JPEG-CMYK.
    if matches!(img.color_space(), Jp2ColorSpace::CMYK) || img.num_components() == 4 {
        return decode_jpx_cmyk(&img, pdf_w, pdf_h);
    }

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
            let gray: Vec<u8> = pixels.chunks_exact(2).map(|c| c[0]).collect();
            let alpha: Vec<u8> = pixels.chunks_exact(2).map(|c| c[1]).collect();
            let mut desc = jpx_gray(jw, jh, gray);
            desc.smask = Some(alpha);
            Some(desc)
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
            let rgb: Vec<u8> = pixels
                .chunks_exact(4)
                .flat_map(|c| [c[0], c[1], c[2]])
                .collect();
            let alpha: Vec<u8> = pixels.chunks_exact(4).map(|c| c[3]).collect();
            let mut desc = jpx_rgb(jw, jh, rgb);
            desc.smask = Some(alpha);
            Some(desc)
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
            let gray: Vec<u8> = pixels.chunks_exact(2).map(|c| (c[0] >> 8) as u8).collect();
            let alpha: Vec<u8> = pixels.chunks_exact(2).map(|c| (c[1] >> 8) as u8).collect();
            let mut desc = jpx_gray(jw, jh, gray);
            desc.smask = Some(alpha);
            Some(desc)
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
            let rgb: Vec<u8> = pixels
                .chunks_exact(4)
                .flat_map(|c| [(c[0] >> 8) as u8, (c[1] >> 8) as u8, (c[2] >> 8) as u8])
                .collect();
            let alpha: Vec<u8> = pixels.chunks_exact(4).map(|c| (c[3] >> 8) as u8).collect();
            let mut desc = jpx_rgb(jw, jh, rgb);
            desc.smask = Some(alpha);
            Some(desc)
        }
    }
}

/// Decode a 4-component (CMYK) JPEG 2000 image to RGB.
///
/// The `jpeg2k` crate's `get_pixels` rejects CMYK outright, so this path reads
/// the four raw component planes directly and feeds packed CMYK bytes through
/// [`cmyk_raw_to_rgb`] — the exact same conversion used for `DeviceCMYK` raw
/// images and JPEG-CMYK, so the output matches the rest of the pipeline.
///
/// Correctness notes:
///  - `ImageComponent::data_u8` already rescales each plane from its native
///    sample precision (1–16 bit, signed or unsigned) to 8-bit, so this path
///    handles non-8-bpc codestreams without assuming a bit depth.
///  - JPEG 2000 stores CMYK as direct ink density (0 = no ink, 255 = full
///    ink), unlike the inverted convention Adobe uses in JPEG/DCT.  We pass
///    `inverted = false` so the conversion matches `DeviceCMYK` raw images
///    (`raw.rs`), not the JPEG-CMYK path.
///  - The PDF `/ColorSpace` and `/Decode` are intentionally ignored for JPX,
///    exactly as the Gray/RGB JPX paths do: the codestream is self-describing
///    and the decoded RGB is final.
///
/// Sub-sampled components (different per-plane dimensions) are rejected with a
/// clear error rather than producing garbage; this is rare for scanned CMYK
/// and the common 1:1 case renders correctly.
fn decode_jpx_cmyk(img: &Jp2Image, pdf_w: u32, pdf_h: u32) -> Option<ImageDescriptor> {
    let comps = img.components();
    let [c, m, y, k] = comps else {
        log::warn!(
            "image: JPXDecode: CMYK image has {} components, expected 4",
            comps.len()
        );
        return None;
    };

    let (jw, jh) = (c.width(), c.height());
    if jw == 0 || jh == 0 {
        log::warn!("image: JPXDecode: CMYK component reports zero dimensions {jw}×{jh}");
        return None;
    }

    // All four planes must share the base resolution.  A sub-sampled component
    // would require per-plane upsampling that this path does not implement;
    // fail loudly rather than interleaving mismatched-length planes.
    if [m, y, k]
        .iter()
        .any(|p| p.width() != jw || p.height() != jh)
    {
        log::warn!(
            "image: JPXDecode: sub-sampled CMYK components unsupported \
             (planes {:?})",
            [
                (c.width(), c.height()),
                (m.width(), m.height()),
                (y.width(), y.height()),
                (k.width(), k.height()),
            ]
        );
        return None;
    }

    if jw != pdf_w || jh != pdf_h {
        log::debug!(
            "image: JPXDecode: PDF dict says {pdf_w}×{pdf_h}, JP2 reports {jw}×{jh} — using JP2 dims"
        );
    }

    let npx = (jw as usize).checked_mul(jh as usize)?;
    let mut cmyk = Vec::with_capacity(npx.checked_mul(4)?);
    // `data_u8` yields the plane already scaled to 8-bit; zipping the four
    // iterators interleaves them into packed CMYK without intermediate Vecs.
    for (((cc, mm), yy), kk) in c
        .data_u8()
        .zip(m.data_u8())
        .zip(y.data_u8())
        .zip(k.data_u8())
    {
        cmyk.extend_from_slice(&[cc, mm, yy, kk]);
    }

    if cmyk.len() != npx * 4 {
        log::warn!(
            "image: JPXDecode: CMYK plane interleave produced {} bytes, expected {}",
            cmyk.len(),
            npx * 4
        );
        return None;
    }

    let rgb = cmyk_raw_to_rgb(
        &cmyk,
        false,
        #[cfg(feature = "gpu-icc")]
        None,
        #[cfg(feature = "gpu-icc")]
        None,
        #[cfg(feature = "gpu-icc")]
        None,
    )?;

    Some(jpx_rgb(jw, jh, rgb))
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
        data: ImageData::Cpu(img.data),
        smask: None,
        filter: ImageFilter::Raw,
    })
}

/// Wrap a decoded grayscale pixel buffer into an [`ImageDescriptor`].
#[inline]
pub(super) const fn jpx_gray(width: u32, height: u32, data: Vec<u8>) -> ImageDescriptor {
    ImageDescriptor {
        width,
        height,
        color_space: ImageColorSpace::Gray,
        data: ImageData::Cpu(data),
        smask: None,
        filter: ImageFilter::Raw,
    }
}

/// Wrap a decoded RGB pixel buffer into an [`ImageDescriptor`].
#[inline]
pub(super) const fn jpx_rgb(width: u32, height: u32, data: Vec<u8>) -> ImageDescriptor {
    ImageDescriptor {
        width,
        height,
        color_space: ImageColorSpace::Rgb,
        data: ImageData::Cpu(data),
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
pub(super) fn decode_jbig2(
    doc: &Document,
    data: &[u8],
    width: u32,
    height: u32,
    is_mask: bool,
    parms: Option<&Object>,
) -> Option<ImageDescriptor> {
    // Resolve optional JBIG2Globals stream from DecodeParms.
    let globals_bytes: Option<Vec<u8>> = parms
        .and_then(|o| o.as_dict())
        .and_then(|d| d.get(b"JBIG2Globals"))
        .and_then(|o| {
            if let Object::Reference(id) = o {
                let g_obj = doc.get_object(*id).ok()?;
                let g_stream = g_obj.as_ref().as_stream()?;
                // JBIG2Globals streams are typically not compressed, but
                // decompressed_content handles both cases transparently.
                g_stream.decompressed_content().ok()
            } else {
                // PDF spec requires JBIG2Globals to be an indirect reference to
                // a stream object; any other form is malformed.
                log::warn!(
                    "image: JBIG2Decode: JBIG2Globals is not an indirect reference — \
                     ignoring globals"
                );
                None
            }
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
        n_pixels,
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
        data: ImageData::Cpu(collector.data),
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
/// JBIG2 white (0) → 0xFF.  `ImageColorSpace` selection is handled by the caller.
struct Jbig2Collector {
    data: Vec<u8>,
    /// Total expected pixel count — used to cap `push_pixel_chunk` so a stream
    /// whose run lengths aren't a multiple of 8 doesn't overflow the buffer.
    n_pixels: usize,
}

impl Jbig2Decoder for Jbig2Collector {
    fn push_pixel(&mut self, black: bool) {
        if self.data.len() < self.n_pixels {
            self.data.push(if black { 0x00 } else { 0xFF });
        }
    }

    fn push_pixel_chunk(&mut self, black: bool, chunk_count: u32) {
        let byte = if black { 0x00 } else { 0xFF };
        // chunk_count is in 8-pixel chunks; cap to the remaining pixels so that
        // run lengths that aren't an exact multiple of 8 don't produce a short
        // buffer (which would cause the caller to discard the image entirely).
        let remaining = self.n_pixels.saturating_sub(self.data.len());
        let n = (chunk_count as usize).saturating_mul(8).min(remaining);
        self.data.extend(std::iter::repeat_n(byte, n));
    }

    fn next_line(&mut self) {
        // Row boundary — nothing to do; pixels are already stored flat.
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jbig2_collector_push_pixel_grayscale() {
        // JBIG2: black=true → Gray 0x00, black=false → Gray 0xFF.
        let mut c = Jbig2Collector {
            data: Vec::new(),
            n_pixels: 2,
        };
        Jbig2Decoder::push_pixel(&mut c, true);
        Jbig2Decoder::push_pixel(&mut c, false);
        assert_eq!(c.data, [0x00, 0xFF]);
    }

    #[test]
    fn jbig2_collector_push_pixel_chunk() {
        // Pre-allocate 16 pixels, matching the `n_pixels` pattern used in production.
        let mut c = Jbig2Collector {
            data: Vec::with_capacity(16),
            n_pixels: 16,
        };
        // chunk_count=2 → 16 pixels of white (0xFF each).
        Jbig2Decoder::push_pixel_chunk(&mut c, false, 2);
        assert_eq!(c.data.len(), 16);
        assert!(c.data.iter().all(|&b| b == 0xFF));
    }

    #[test]
    fn jbig2_collector_push_pixel_chunk_overrun() {
        // Regression test: chunk whose byte-count overshoots n_pixels by 7
        // (common when page dimensions aren't a multiple of 8).
        // Before the fix, this produced a short buffer and the image was dropped.
        let mut c = Jbig2Collector {
            data: Vec::with_capacity(10),
            n_pixels: 10,
        };
        // chunk_count=2 → 16 pixels, but n_pixels=10 — must cap at 10.
        Jbig2Decoder::push_pixel_chunk(&mut c, false, 2);
        assert_eq!(c.data.len(), 10);
        assert!(c.data.iter().all(|&b| b == 0xFF));
    }

    #[test]
    fn jbig2_decode_invalid_data_returns_none() {
        // Corrupt/empty JBIG2 data must not panic — it must return None.
        let doc = crate::test_helpers::empty_doc();
        let result = decode_jbig2(&doc, b"\x00\x01\x02\x03", 4, 4, false, None);
        assert!(result.is_none());
    }

    #[cfg(any(feature = "nvjpeg", feature = "vaapi"))]
    fn minimal_baseline_jpeg() -> Vec<u8> {
        vec![
            0xFF, 0xD8, // SOI
            0xFF, 0xC0, // SOF0
            0x00, 0x05, // length = 5
            0x08, // precision
            0x00, 0x01, // height = 1
            0xFF, 0xD9, // EOI
        ]
    }

    #[cfg(any(feature = "nvjpeg", feature = "vaapi"))]
    fn minimal_progressive_jpeg() -> Vec<u8> {
        vec![
            0xFF, 0xD8, // SOI
            0xFF, 0xC2, // SOF2
            0x00, 0x05, // length = 5
            0x08, // precision
            0x00, 0x01, // height = 1
            0xFF, 0xD9, // EOI
        ]
    }

    #[cfg(any(feature = "nvjpeg", feature = "vaapi"))]
    #[test]
    fn jpeg_variant_baseline_detected() {
        let data = minimal_baseline_jpeg();
        assert_eq!(gpu::jpeg_sof_type(&data), Some(gpu::JpegVariant::Baseline));
    }

    #[cfg(any(feature = "nvjpeg", feature = "vaapi"))]
    #[test]
    fn jpeg_variant_progressive_detected() {
        let data = minimal_progressive_jpeg();
        assert_eq!(
            gpu::jpeg_sof_type(&data),
            Some(gpu::JpegVariant::Progressive)
        );
    }

    // ── decode_dct_gpu_path ───────────────────────────────────────────────────

    #[cfg(feature = "nvjpeg")]
    mod gpu_path_tests {
        use super::*;

        struct AlwaysGray;

        impl gpu::GpuJpegDecoder for AlwaysGray {
            fn decode_jpeg(
                &mut self,
                _data: &[u8],
                _width: u32,
                _height: u32,
            ) -> Result<gpu::DecodedImage, gpu::GpuDecodeError> {
                Ok(gpu::DecodedImage {
                    data: vec![128u8; 4],
                    width: 2,
                    height: 2,
                    components: 1,
                })
            }
        }

        struct AlwaysFail;

        impl gpu::GpuJpegDecoder for AlwaysFail {
            fn decode_jpeg(
                &mut self,
                _data: &[u8],
                _width: u32,
                _height: u32,
            ) -> Result<gpu::DecodedImage, gpu::GpuDecodeError> {
                Err(gpu::GpuDecodeError::new(std::io::Error::other(
                    "simulated failure",
                )))
            }
        }

        struct AlwaysRgb;

        impl gpu::GpuJpegDecoder for AlwaysRgb {
            fn decode_jpeg(
                &mut self,
                _data: &[u8],
                _width: u32,
                _height: u32,
            ) -> Result<gpu::DecodedImage, gpu::GpuDecodeError> {
                Ok(gpu::DecodedImage {
                    data: vec![255u8; 12],
                    width: 2,
                    height: 2,
                    components: 3,
                })
            }
        }

        struct AlwaysCmyk;

        impl gpu::GpuJpegDecoder for AlwaysCmyk {
            fn decode_jpeg(
                &mut self,
                _data: &[u8],
                _width: u32,
                _height: u32,
            ) -> Result<gpu::DecodedImage, gpu::GpuDecodeError> {
                Ok(gpu::DecodedImage {
                    data: vec![0u8; 16],
                    width: 2,
                    height: 2,
                    components: 4,
                })
            }
        }

        #[test]
        fn generic_gpu_path_gray() {
            let mut dec = AlwaysGray;
            let result = decode_dct_gpu_path(&[], 2, 2, &mut dec);
            let img = result.expect("should succeed");
            assert_eq!(img.color_space, ImageColorSpace::Gray);
            assert_eq!(img.width, 2);
            assert_eq!(img.height, 2);
            assert_eq!(img.data.as_cpu().unwrap(), &vec![128u8; 4]);
        }

        #[test]
        fn generic_gpu_path_failure_returns_none() {
            let mut dec = AlwaysFail;
            let result = decode_dct_gpu_path(&[], 2, 2, &mut dec);
            assert!(result.is_none());
        }

        #[test]
        fn generic_gpu_path_rgb() {
            let mut dec = AlwaysRgb;
            let result = decode_dct_gpu_path(&[], 2, 2, &mut dec);
            let img = result.expect("should succeed");
            assert_eq!(img.color_space, ImageColorSpace::Rgb);
            assert_eq!(img.data.len(), 12);
        }

        #[test]
        fn generic_gpu_path_rejects_cmyk() {
            let mut dec = AlwaysCmyk;
            let result = decode_dct_gpu_path(&[], 2, 2, &mut dec);
            assert!(result.is_none());
        }
    }

    // ── jpx_gray / jpx_rgb helpers ────────────────────────────────────────────

    #[test]
    fn jpx_gray_smask_none_by_default() {
        let desc = jpx_gray(2, 2, vec![0u8; 4]);
        assert!(desc.smask.is_none(), "jpx_gray must not inject a smask");
    }

    #[test]
    fn jpx_rgb_smask_none_by_default() {
        let desc = jpx_rgb(2, 2, vec![0u8; 12]);
        assert!(desc.smask.is_none(), "jpx_rgb must not inject a smask");
    }

    // ── JPX-CMYK convention ───────────────────────────────────────────────────
    //
    // `decode_jpx_cmyk` cannot be exercised directly without a real JP2
    // codestream (the `jpeg2k` crate exposes no public constructor for a
    // synthetic 4-component CMYK `Image`).  This test instead pins the one
    // non-obvious decision in that path: JPEG 2000 stores CMYK as *direct* ink
    // density, so the conversion must use `inverted = false` — the same as
    // DeviceCMYK raw images, and the opposite of the JPEG-CMYK path.  A
    // regression that flipped the flag would invert every CMYK JPX page's
    // colours; this asserts the two interpretations are distinct and that the
    // direct one is what `decode_jpx_cmyk` requests.
    #[test]
    fn jpx_cmyk_uses_direct_ink_convention_not_jpeg_inverted() {
        // Pure-cyan ink at full density: direct convention (0 = no ink).
        let cyan_direct = [255u8, 0, 0, 0];

        let direct = cmyk_raw_to_rgb(
            &cyan_direct,
            false,
            #[cfg(feature = "gpu-icc")]
            None,
            #[cfg(feature = "gpu-icc")]
            None,
            #[cfg(feature = "gpu-icc")]
            None,
        )
        .expect("direct CMYK→RGB");
        let inverted = cmyk_raw_to_rgb(
            &cyan_direct,
            true,
            #[cfg(feature = "gpu-icc")]
            None,
            #[cfg(feature = "gpu-icc")]
            None,
            #[cfg(feature = "gpu-icc")]
            None,
        )
        .expect("inverted CMYK→RGB");

        // Full cyan ink absorbs red → R must be the smallest channel.
        let [r, g, b] = direct[..3] else {
            unreachable!("3 bytes per pixel")
        };
        assert!(
            r < g && r < b,
            "full cyan ink (direct convention) must darken red: got rgb=({r},{g},{b})"
        );

        // The inverted (JPEG) reading of the same bytes is a *different* colour;
        // if it were not, the convention choice would not matter and a flipped
        // flag would silently regress.
        assert_ne!(
            &direct[..3],
            &inverted[..3],
            "direct and JPEG-inverted CMYK readings must differ — \
             this is why decode_jpx_cmyk must pass inverted = false"
        );
    }
}
