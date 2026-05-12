//! `JpegGpuDecoder` — end-to-end JPEG → RGBA8 device image.
//!
//! CPU pre-pass (`prepare_jpeg`) + CPU AC coefficient extraction +
//! GPU Phase 5 (IDCT + dequant + colour conversion).

use crate::backend::params::IdctParams;
use crate::backend::{BackendError as BackendErr, GpuBackend};
use crate::jpeg::bitreader::BitReader;
use crate::jpeg::headers::{JpegFrameComponent, mcu_count};
use crate::jpeg_decoder::cpu_prepass::{JpegPreparedInput, prepare_jpeg};
use crate::jpeg_decoder::device_image::DeviceImage;
use crate::jpeg_decoder::dispatch_util::DeviceBufferGuard;
use crate::jpeg_decoder::error::JpegGpuError;

/// End-to-end JPEG decoder that runs IDCT on the GPU.
///
/// Phases 1–4 (parallel Huffman decode) are available via
/// `dispatch_jpeg_phase1_through_phase4`; this struct uses a CPU
/// coefficient extraction pass and dispatches only Phase 5 to the GPU.
pub struct JpegGpuDecoder<B: GpuBackend> {
    backend: B,
}

impl<B: GpuBackend> JpegGpuDecoder<B> {
    /// Create a new decoder bound to `backend`.
    pub const fn new(backend: B) -> Self {
        Self { backend }
    }

    /// Borrow the underlying backend.
    pub const fn backend(&self) -> &B {
        &self.backend
    }

    /// Decode a baseline JFIF JPEG into a device-resident RGBA8 image.
    ///
    /// # Errors
    /// Returns `JpegGpuError` if the input is not a supported baseline JPEG,
    /// if coefficient extraction fails, or if any GPU operation fails.
    pub fn decode(&self, jpeg_bytes: &[u8]) -> std::result::Result<DeviceImage<B>, JpegGpuError> {
        let prep = prepare_jpeg(jpeg_bytes)?;

        // Reject subsampled input: the IDCT kernel assumes 4:4:4 (all components
        // share the same 8×8 block grid). Non-unity sampling factors produce the
        // wrong chroma block layout and corrupt output silently.
        for comp in &prep.components {
            if comp.h_sampling != 1 || comp.v_sampling != 1 {
                return Err(JpegGpuError::UnsupportedSubsampling {
                    component: comp.id,
                    h: comp.h_sampling,
                    v: comp.v_sampling,
                });
            }
        }

        let (coef_flat, dc_flat, qt_flat, num_qtables) =
            extract_coefficients(&prep).map_err(JpegGpuError::HeaderParse)?;

        let width = u32::from(prep.width);
        let height = u32::from(prep.height);
        let nc = prep.components.len();
        if nc != 1 && nc != 3 {
            return Err(JpegGpuError::UnsupportedComponents(
                u8::try_from(nc).unwrap_or(u8::MAX),
            ));
        }
        #[expect(
            clippy::cast_possible_truncation,
            reason = "nc is 1 or 3 (checked above); trivially fits in u32"
        )]
        let num_components = nc as u32;
        let blocks_wide = width.div_ceil(8);
        let blocks_high = height.div_ceil(8);

        self.dispatch_idct(
            &coef_flat,
            &qt_flat,
            &dc_flat,
            width,
            height,
            num_components,
            blocks_wide,
            blocks_high,
            num_qtables,
        )
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "10-scalar IDCT params; grouping them is IdctParams which requires a backend buffer ref"
    )]
    fn dispatch_idct(
        &self,
        coefficients: &[i32],
        qtables: &[i32],
        dc_values: &[i32],
        width: u32,
        height: u32,
        num_components: u32,
        blocks_wide: u32,
        blocks_high: u32,
        num_qtables: u32,
    ) -> std::result::Result<DeviceImage<B>, JpegGpuError> {
        let be = |e: BackendErr| JpegGpuError::Dispatch(e.to_string());

        let coef_bytes = std::mem::size_of_val(coefficients);
        let qt_bytes = std::mem::size_of_val(qtables);
        let dc_bytes = std::mem::size_of_val(dc_values);
        let px_bytes = (width as usize) * (height as usize) * 4;

        let coef_buf = DeviceBufferGuard::alloc(&self.backend, coef_bytes).map_err(be)?;
        let qt_buf = DeviceBufferGuard::alloc(&self.backend, qt_bytes).map_err(be)?;
        let dc_buf = DeviceBufferGuard::alloc(&self.backend, dc_bytes).map_err(be)?;
        let px_buf = DeviceBufferGuard::alloc_zeroed(&self.backend, px_bytes).map_err(be)?;

        let _u1 = self
            .backend
            .upload_async(coef_buf.as_ref(), bytemuck::cast_slice(coefficients))
            .map_err(be)?;
        let _u2 = self
            .backend
            .upload_async(qt_buf.as_ref(), bytemuck::cast_slice(qtables))
            .map_err(be)?;
        let _u3 = self
            .backend
            .upload_async(dc_buf.as_ref(), bytemuck::cast_slice(dc_values))
            .map_err(be)?;

        self.backend.begin_page().map_err(be)?;
        self.backend
            .record_idct(IdctParams {
                coefficients: coef_buf.as_ref(),
                qtables: qt_buf.as_ref(),
                dc_values: dc_buf.as_ref(),
                pixels_rgba: px_buf.as_ref(),
                width,
                height,
                num_components,
                blocks_wide,
                blocks_high,
                num_qtables,
            })
            .map_err(be)?;
        let fence = self.backend.submit_page().map_err(be)?;
        self.backend.wait_page(fence).map_err(be)?;

        self.backend.free_device(coef_buf.take());
        self.backend.free_device(qt_buf.take());
        self.backend.free_device(dc_buf.take());

        Ok(DeviceImage {
            buffer: px_buf.take(),
            width,
            height,
        })
    }
}

/// Reconstruct DCT coefficient arrays from `prep`.
///
/// Returns `(coefficients, dc_values, qtables, num_qtables)`:
/// - `coefficients`: `num_components × blocks_per_comp × 64` i32 in zigzag order
/// - `dc_values`: `num_components × blocks_per_comp` i32 (absolute DC from pre-pass)
/// - `qtables`: `num_qtables × 64` i32 in natural (row-major) order
/// - `num_qtables`: count of populated quantisation table slots
#[expect(
    clippy::type_complexity,
    reason = "4-tuple private fn return; a named struct is overkill here"
)]
#[expect(
    clippy::too_many_lines,
    reason = "AC walk + DC copy + QT copy; split would obscure the single-pass structure"
)]
fn extract_coefficients(
    prep: &JpegPreparedInput,
) -> std::result::Result<(Vec<i32>, Vec<i32>, Vec<i32>, u32), String> {
    let num_comp = prep.components.len();
    let blocks_wide = usize::from(prep.width.div_ceil(8));
    let blocks_high = usize::from(prep.height.div_ceil(8));
    let blocks_per_comp = blocks_wide * blocks_high;

    let mut coef_flat = vec![0i32; num_comp * blocks_per_comp * 64];

    // Recover raw bitstream bytes for BitReader:
    // PackedBitstream words use u32::from_be_bytes packing, so each word
    // unpacks to bytes via to_be_bytes().
    let raw_bytes: Vec<u8> = prep
        .bitstream
        .words
        .iter()
        .flat_map(|w| w.to_be_bytes())
        .collect();
    let mut bits = BitReader::new(&raw_bytes);

    let totalmcus = mcu_count(prep.width, prep.height, &prep.components);
    let mut block_counts = vec![0usize; num_comp];

    for mcu in 0..totalmcus {
        for (ci, comp) in prep.components.iter().enumerate() {
            let dc_sel = usize::from(
                *prep
                    .dc_selectors
                    .get(ci)
                    .ok_or_else(|| format!("no dc_selector for comp {ci}"))?,
            );
            let ac_sel = usize::from(
                *prep
                    .ac_selectors
                    .get(ci)
                    .ok_or_else(|| format!("no ac_selector for comp {ci}"))?,
            );
            let dc_cb = prep.dc_codebooks[dc_sel]
                .as_ref()
                .ok_or_else(|| format!("missing DC codebook {dc_sel}"))?;
            let ac_cb = prep.ac_codebooks[ac_sel]
                .as_ref()
                .ok_or_else(|| format!("missing AC codebook {ac_sel}"))?;

            let bpm = blocks_permcu_count(*comp);
            for _b in 0..bpm {
                let block_idx = block_counts[ci];
                let coef_base = (ci * blocks_per_comp + block_idx) * 64;
                block_counts[ci] += 1;

                // DC: peek + consume codeword, then read `category` magnitude bits.
                let peek = bits.peek_u16().ok_or_else(|| {
                    format!("bitstream empty at DC codeword (mcu={mcu} comp={ci})")
                })?;
                let dc_entry = dc_cb.lookup(peek);
                bits.consume(usize::from(dc_entry.num_bits));
                let category = dc_entry.symbol;
                if category > 0 {
                    let _ = bits.read_bits(usize::from(category)).ok_or_else(|| {
                        format!("DC magnitude truncated (mcu={mcu} comp={ci} cat={category})")
                    })?;
                }

                // AC: read each symbol + magnitude bits.
                let mut zz = 1usize;
                while zz < 64 {
                    let peek = bits.peek_u16().ok_or_else(|| {
                        format!("bitstream empty at AC (mcu={mcu} comp={ci} zz={zz})")
                    })?;
                    let ac_entry = ac_cb.lookup(peek);
                    bits.consume(usize::from(ac_entry.num_bits));
                    let sym_byte = ac_entry.symbol;

                    if sym_byte == 0x00 {
                        break; // EOB
                    }
                    if sym_byte == 0xF0 {
                        zz += 16; // ZRL
                        continue;
                    }
                    let run = (sym_byte >> 4) as usize;
                    let size = sym_byte & 0x0F;
                    zz += run;
                    if zz >= 64 {
                        break;
                    }
                    let ac_val = if size == 0 {
                        0i32
                    } else {
                        let raw = bits.read_bits(usize::from(size)).ok_or_else(|| {
                            format!("AC magnitude truncated (mcu={mcu} comp={ci} zz={zz})")
                        })?;
                        jpeg_extend(raw.cast_signed(), size)
                    };
                    if coef_base + zz >= coef_flat.len() {
                        return Err(format!(
                            "coefficient overflow: coef_base={coef_base} zz={zz} \
                             coef_flat.len()={} (mcu={mcu} comp={ci})",
                            coef_flat.len()
                        ));
                    }
                    coef_flat[coef_base + zz] = ac_val;
                    zz += 1;
                }
            }
        }
    }

    // DC values: pre-resolved absolute DC chain from the pre-pass.
    let mut dc_flat = vec![0i32; num_comp * blocks_per_comp];
    for (ci, dc_vec) in prep
        .dc_values
        .per_component
        .iter()
        .take(num_comp)
        .enumerate()
    {
        let base = ci * blocks_per_comp;
        for (bi, &dc) in dc_vec.iter().enumerate() {
            let idx = base + bi;
            if idx >= dc_flat.len() {
                return Err(format!(
                    "dc_values overflow: comp={ci} block={bi} idx={idx} \
                     dc_flat.len()={}",
                    dc_flat.len()
                ));
            }
            dc_flat[idx] = dc;
        }
    }

    // Quantisation tables: pack populated slots densely, preserving slot order.
    // The kernel selects qt_sel=0 for luma and qt_sel=1 (clamped) for chroma, so
    // tables must be dense with luma-table first.  JFIF always uses slot 0 (luma)
    // and slot 1 (chroma), so iterating in slot order and compacting gives the
    // correct dense layout.
    let mut qt_flat = Vec::<i32>::with_capacity(4 * 64);
    for (qt_idx, qt) in prep.quant_tables.iter().enumerate() {
        if let Some(qt) = qt {
            if qt.values.len() != 64 {
                return Err(format!(
                    "quant table {qt_idx} has {} entries (expected 64)",
                    qt.values.len()
                ));
            }
            for &v in &qt.values {
                qt_flat.push(i32::from(v));
            }
        }
    }
    if qt_flat.is_empty() {
        return Err("no quantisation tables in JPEG".to_owned());
    }
    // At most 4 slots in a JPEG, so qt_flat.len() / 64 ≤ 4, always fits in u32.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "qt_flat has at most 4*64 entries; quotient is at most 4, fits in u32"
    )]
    let num_qtables = (qt_flat.len() / 64) as u32;

    Ok((coef_flat, dc_flat, qt_flat, num_qtables))
}

/// Number of 8×8 blocks this component contributes per MCU.
/// Non-interleaved (1-component scan): always 1.
/// Interleaved: `h_sampling` × `v_sampling`.
const fn blocks_permcu_count(comp: JpegFrameComponent) -> usize {
    (comp.h_sampling as usize) * (comp.v_sampling as usize)
}

/// JPEG EXTEND: sign-extend an `nbits`-wide magnitude into a signed integer.
const fn jpeg_extend(value: i32, nbits: u8) -> i32 {
    if nbits == 0 {
        return 0;
    }
    let vt = 1i32 << (nbits - 1);
    if value < vt {
        value + (-1 << nbits) + 1
    } else {
        value
    }
}

/// Test-only re-export of the private `extract_coefficients` so sibling
/// test modules (e.g., `cpu_prepass` DRI tests) can call it without
/// duplicating the coefficient-extraction logic.
#[cfg(test)]
pub(crate) fn extract_coefficients_pub(
    prep: &crate::jpeg_decoder::cpu_prepass::JpegPreparedInput,
) -> std::result::Result<(Vec<i32>, Vec<i32>, Vec<i32>, u32), String> {
    extract_coefficients(prep)
}

#[cfg(all(test, feature = "gpu-validation"))]
mod tests {
    use super::*;
    use crate::backend::cuda::CudaBackend;

    #[test]
    fn decoder_decodes_grayscale_jpeg_on_cuda() {
        let backend = CudaBackend::new().expect("CUDA backend");
        let dec = JpegGpuDecoder::new(backend);
        let bytes = crate::jpeg::test_fixtures::GRAY_16X16_JPEG;
        let img = dec.decode(bytes).expect("decode");
        assert_eq!(img.width, 16);
        assert_eq!(img.height, 16);
    }

    /// Download device pixels and compare to a zune-jpeg reference decode.
    ///
    /// Tolerance: peak absolute error ≤ 1 LSB per channel, mean absolute
    /// error ≤ 0.02 LSB — matches the IEEE 1180-1990 quantisation error
    /// budget for the IDCT.
    fn assert_within_ieee1180(gpu_rgba: &[u8], zune_rgba: &[u8], label: &str) {
        assert_eq!(
            gpu_rgba.len(),
            zune_rgba.len(),
            "{label}: pixel buffer length mismatch"
        );
        let mut peak = 0u8;
        let mut sum: u64 = 0;
        for (&g, &r) in gpu_rgba.iter().zip(zune_rgba.iter()) {
            let diff = g.abs_diff(r);
            if diff > peak {
                peak = diff;
            }
            sum += u64::from(diff);
        }
        let mean = sum as f64 / gpu_rgba.len() as f64;
        assert!(peak <= 1, "{label}: peak error {peak} > 1 LSB (IEEE 1180)");
        assert!(
            mean <= 0.02,
            "{label}: mean error {mean:.4} > 0.02 LSB (IEEE 1180)"
        );
    }

    /// Decode `bytes` with zune-jpeg in RGBA8 and return the flat pixel buffer.
    fn zune_decode_rgba(bytes: &[u8]) -> Vec<u8> {
        use zune_jpeg::JpegDecoder;
        use zune_jpeg::zune_core::bytestream::ZCursor;
        use zune_jpeg::zune_core::colorspace::ColorSpace;
        use zune_jpeg::zune_core::options::DecoderOptions;
        let opts = DecoderOptions::default().jpeg_set_out_colorspace(ColorSpace::RGBA);
        let mut dec = JpegDecoder::new_with_options(ZCursor::new(bytes), opts);
        dec.decode().expect("zune-jpeg decode")
    }

    /// Download RGBA8 bytes from a `DeviceImage` via `download_async`.
    fn download_image(backend: &CudaBackend, img: &DeviceImage<CudaBackend>) -> Vec<u8> {
        let n = (img.width as usize) * (img.height as usize) * 4;
        let mut buf = vec![0u8; n];
        let handle = backend
            .download_async(&img.buffer, &mut buf)
            .expect("download_async");
        backend.wait_download(handle).expect("wait_download");
        buf
    }

    #[test]
    fn real_jpeg_q95_pixel_diff_within_ieee1180() {
        let bytes = include_bytes!("../../../../tests/fixtures/jpeg/q95_scan.jpg");
        let backend = CudaBackend::new().expect("CUDA backend");
        let dec = JpegGpuDecoder::new(backend);
        let img = dec.decode(bytes).expect("decode q95_scan.jpg");
        let gpu_rgba = download_image(dec.backend(), &img);
        let zune_rgba = zune_decode_rgba(bytes);
        assert_within_ieee1180(&gpu_rgba, &zune_rgba, "q95_scan.jpg");
    }

    #[test]
    fn real_jpeg_q20_pixel_diff_within_ieee1180() {
        let bytes = include_bytes!("../../../../tests/fixtures/jpeg/q20.jpg");
        let backend = CudaBackend::new().expect("CUDA backend");
        let dec = JpegGpuDecoder::new(backend);
        let img = dec.decode(bytes).expect("decode q20.jpg");
        let gpu_rgba = download_image(dec.backend(), &img);
        let zune_rgba = zune_decode_rgba(bytes);
        assert_within_ieee1180(&gpu_rgba, &zune_rgba, "q20.jpg");
    }
}
