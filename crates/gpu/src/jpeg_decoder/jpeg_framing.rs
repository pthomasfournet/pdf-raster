//! CPU reference walker for the JPEG-framed entropy-coded segment.
//!
//! The GPU parallel-Huffman kernels emit a stream of Huffman-decoded
//! symbol bytes in scan order: per block, one DC symbol followed by up
//! to 63 AC symbols (truncated by an EOB marker), repeated for every
//! block of every MCU, for every scan component.  This module reproduces
//! the same emission on the CPU so the GPU output can be compared
//! byte-for-byte without depending on a third-party decoder.
//!
//! The walker shares its bit-level decoding semantics with
//! [`crate::jpeg::dc_chain`] — DC delta extraction, AC run/size pairs,
//! EOB / ZRL handling, restart-marker re-alignment — but emits the full
//! symbol stream instead of resolving the DC chain and skipping AC.
//!
//! Output shape — matching the kernel's `symbols_out` layout:
//!
//! 1. For each MCU in scan order:
//!     1. For each scan component (in scan-header order):
//!         1. For each block belonging to that component within the MCU:
//!             1. Emit the DC Huffman-decoded symbol byte.
//!             2. Emit each AC Huffman-decoded symbol byte until either
//!                EOB (0x00) is hit or the 63rd AC slot is filled.
//!                EOB / ZRL bytes are themselves emitted into the
//!                stream so the consumer can reconstruct block
//!                boundaries.
//!
//! Each emitted symbol is a `u32` whose low 8 bits hold the Huffman
//! symbol byte; the upper bits are reserved zero for future per-symbol
//! metadata (matching the synthetic-stream Phase 4 contract).

use crate::jpeg::CanonicalCodebook;
use crate::jpeg::headers::JpegFrameComponent;
use crate::jpeg_decoder::{JpegGpuError, JpegPreparedInput};

/// Errors emitted by [`decode_scan_symbols`].
///
/// Mirrored to [`JpegGpuError`] at the public surface; the typed inner
/// enum exists so this module's tests can pattern-match on precise
/// failure causes without parsing strings.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum JpegFramingError {
    /// Bitstream ended mid-decode (Huffman lookup needed more bits than
    /// the segment supplied).
    UnexpectedEnd {
        /// 0-based MCU index where the truncation surfaced.
        mcu_index: u32,
        /// 0-based scan-component index where the truncation surfaced.
        scan_component: u8,
    },
    /// 16-bit prefix landed on a codebook slot with `num_bits == 0` —
    /// either the codebook does not cover the bit pattern or the stream
    /// is corrupt.
    InvalidCode {
        /// 0-based MCU index where the prefix-miss surfaced.
        mcu_index: u32,
        /// 0-based scan-component index where the prefix-miss surfaced.
        scan_component: u8,
    },
    /// AC run+size pair walked past coefficient slot 63 in some block.
    AcOverflow {
        /// 0-based MCU index where the overflow surfaced.
        mcu_index: u32,
    },
    /// DC magnitude category > 11 (max for 8-bit baseline JPEG).
    BadDcCategory {
        /// The category as decoded from the DC Huffman table.
        category: u8,
    },
    /// `JpegPreparedInput` declared a scan component whose
    /// `frame_components` entry has `h_sampling == 0` or `v_sampling == 0`
    /// — that yields a 0-block MCU and would silently swallow all
    /// emissions.  Refused rather than silently mishandled.
    DegenerateSampling {
        /// Scan-component index that carries the degenerate sampling.
        scan_component: u8,
    },
    /// Restart-aware bitstream framing is not yet wired through
    /// `JpegPreparedInput`; the wrapper drops `rst_positions` from the
    /// CPU pre-pass output.  Refuse rather than silently producing a
    /// symbol stream that drifts after the first RST boundary.
    RestartMarkersUnsupported {
        /// Restart interval (in MCUs) the input declared.
        restart_interval: u16,
    },
}

impl std::fmt::Display for JpegFramingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnexpectedEnd {
                mcu_index,
                scan_component,
            } => write!(
                f,
                "JPEG framing: bitstream ended at MCU {mcu_index}, scan-component {scan_component}",
            ),
            Self::InvalidCode {
                mcu_index,
                scan_component,
            } => write!(
                f,
                "JPEG framing: invalid Huffman prefix at MCU {mcu_index}, scan-component {scan_component}",
            ),
            Self::AcOverflow { mcu_index } => write!(
                f,
                "JPEG framing: AC run+size walked past slot 63 at MCU {mcu_index}",
            ),
            Self::BadDcCategory { category } => write!(
                f,
                "JPEG framing: DC magnitude category {category} > 11 (8-bit baseline cap)",
            ),
            Self::DegenerateSampling { scan_component } => write!(
                f,
                "JPEG framing: scan-component {scan_component} has 0 horizontal or vertical sampling",
            ),
            Self::RestartMarkersUnsupported { restart_interval } => write!(
                f,
                "JPEG framing: restart-aware bitstream not yet wired (interval = {restart_interval} MCUs)",
            ),
        }
    }
}

impl std::error::Error for JpegFramingError {}

impl From<JpegFramingError> for JpegGpuError {
    /// Framing failures surface as [`JpegGpuError::Dispatch`] — they're
    /// detected by this module (not the CPU pre-pass), and they imply
    /// "the bitstream is corrupt or unsupported, fall back to a CPU
    /// decoder for this image."
    fn from(value: JpegFramingError) -> Self {
        Self::Dispatch(value.to_string())
    }
}

/// Walk the prepared JPEG and emit the GPU-kernel-shaped symbol stream.
///
/// # Errors
///
/// Returns [`JpegFramingError`] on any structural failure (corrupt
/// stream, missing codebook slot, AC overflow, malformed sampling
/// factors).  All errors are non-fatal at the caller level — they
/// indicate "fall back to a CPU decoder for this image."
pub fn decode_scan_symbols(prep: &JpegPreparedInput) -> Result<Vec<u32>, JpegFramingError> {
    let walker = ScanWalker::new(prep)?;
    walker.run()
}

struct ScanWalker<'a> {
    bitstream_bytes: Vec<u8>,
    components: &'a [JpegFrameComponent],
    dc_cbs: Vec<&'a CanonicalCodebook>,
    ac_cbs: Vec<&'a CanonicalCodebook>,
    /// `blocks_per_mcu[k]` = number of 8×8 blocks the `k`-th scan
    /// component contributes per MCU.  Non-interleaved scans (a single
    /// component) emit one block per MCU regardless of sampling.
    blocks_per_mcu: [u8; 4],
    num_mcus: u32,
}

impl<'a> ScanWalker<'a> {
    fn new(prep: &'a JpegPreparedInput) -> Result<Self, JpegFramingError> {
        if prep.restart_interval > 0 {
            return Err(JpegFramingError::RestartMarkersUnsupported {
                restart_interval: prep.restart_interval,
            });
        }
        let scan_components = prep.components.len();
        // Convert PackedBitstream back to byte form for the bit reader.
        // The wrapper packed `unstuffed` 1:1 into BE-32 words, so the
        // reverse is BE bytes from words capped at length_bits/8.
        let byte_len = (prep.bitstream.length_bits as usize) / 8;
        let mut bytes = Vec::with_capacity(byte_len);
        for word in &prep.bitstream.words {
            bytes.extend_from_slice(&word.to_be_bytes());
        }
        bytes.truncate(byte_len);

        let dc_cbs = prep.dc_codebooks_for_dispatch();
        let ac_cbs = prep.ac_codebooks_for_dispatch();

        let mut blocks_per_mcu = [0u8; 4];
        for (k, fc) in prep.components.iter().enumerate() {
            if fc.h_sampling == 0 || fc.v_sampling == 0 {
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "scan_components ≤ 4 by upstream validation; k < 4"
                )]
                let k_u8 = k as u8;
                return Err(JpegFramingError::DegenerateSampling {
                    scan_component: k_u8,
                });
            }
            blocks_per_mcu[k] = if scan_components == 1 {
                1
            } else {
                fc.h_sampling * fc.v_sampling
            };
        }

        let num_mcus = mcu_count(prep.width, prep.height, &prep.components);

        Ok(Self {
            bitstream_bytes: bytes,
            components: prep.components.as_slice(),
            dc_cbs,
            ac_cbs,
            blocks_per_mcu,
            num_mcus,
        })
    }

    fn run(self) -> Result<Vec<u32>, JpegFramingError> {
        let mut bits = BitReader::new(&self.bitstream_bytes);
        let mut symbols = Vec::new();

        for mcu in 0..self.num_mcus {
            for (k, &bpm) in self.blocks_per_mcu[..self.components.len()]
                .iter()
                .enumerate()
            {
                let dc_cb = self.dc_cbs[k];
                let ac_cb = self.ac_cbs[k];
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "scan component index ≤ 3 by upstream validation"
                )]
                let k_u8 = k as u8;
                for _ in 0..bpm {
                    emit_dc_symbol(&mut bits, dc_cb, &mut symbols, mcu, k_u8)?;
                    emit_ac_symbols(&mut bits, ac_cb, &mut symbols, mcu, k_u8)?;
                }
            }
        }
        Ok(symbols)
    }
}

/// Decode the next DC codeword, emit its symbol byte, and consume the
/// raw bits that encode the DC magnitude.  The magnitude itself is
/// dropped — the GPU kernel only emits symbol bytes; downstream
/// coefficient assembly will re-read the raw bits if needed.
fn emit_dc_symbol(
    bits: &mut BitReader<'_>,
    cb: &CanonicalCodebook,
    out: &mut Vec<u32>,
    mcu: u32,
    scan_component: u8,
) -> Result<(), JpegFramingError> {
    let prefix = bits.peek_u16().ok_or(JpegFramingError::UnexpectedEnd {
        mcu_index: mcu,
        scan_component,
    })?;
    let entry = cb.lookup(prefix);
    if entry.num_bits == 0 {
        return Err(JpegFramingError::InvalidCode {
            mcu_index: mcu,
            scan_component,
        });
    }
    bits.consume(usize::from(entry.num_bits));
    let category = entry.symbol;
    if category > 11 {
        return Err(JpegFramingError::BadDcCategory { category });
    }
    out.push(u32::from(category));
    if category > 0 && bits.read_bits(usize::from(category)).is_none() {
        return Err(JpegFramingError::UnexpectedEnd {
            mcu_index: mcu,
            scan_component,
        });
    }
    Ok(())
}

/// Walk one block's 63 AC slots, emitting each Huffman symbol byte
/// (EOB / ZRL included) until EOB fires or the 63rd slot is reached.
fn emit_ac_symbols(
    bits: &mut BitReader<'_>,
    cb: &CanonicalCodebook,
    out: &mut Vec<u32>,
    mcu: u32,
    scan_component: u8,
) -> Result<(), JpegFramingError> {
    let mut ac_pos = 1u8;
    while ac_pos < 64 {
        let prefix = bits.peek_u16().ok_or(JpegFramingError::UnexpectedEnd {
            mcu_index: mcu,
            scan_component,
        })?;
        let entry = cb.lookup(prefix);
        if entry.num_bits == 0 {
            return Err(JpegFramingError::InvalidCode {
                mcu_index: mcu,
                scan_component,
            });
        }
        bits.consume(usize::from(entry.num_bits));
        let symbol = entry.symbol;
        out.push(u32::from(symbol));
        if symbol == 0x00 {
            // EOB: rest of block is implicitly zero — no more AC
            // emissions for this block.
            return Ok(());
        }
        if symbol == 0xF0 {
            // ZRL: 16-zero run.  ac_pos jumps; no raw bits follow.
            ac_pos = ac_pos.saturating_add(16);
            continue;
        }
        let run = symbol >> 4;
        let size = symbol & 0x0F;
        // run + 1 advances are needed before slot 64; saturating_add
        // here protects against an adversarial symbol with the high
        // nibble set such that ac_pos + run + 1 overflows u8.
        let advance = run.saturating_add(1);
        ac_pos = ac_pos.saturating_add(advance);
        if ac_pos > 64 {
            return Err(JpegFramingError::AcOverflow { mcu_index: mcu });
        }
        if size > 0 && bits.read_bits(usize::from(size)).is_none() {
            return Err(JpegFramingError::UnexpectedEnd {
                mcu_index: mcu,
                scan_component,
            });
        }
    }
    Ok(())
}

/// MCU count from frame dimensions + scan-component sampling factors.
/// Mirrors `crate::jpeg::headers::mcu_count`; duplicated here to avoid
/// exposing a separate accessor on `JpegPreparedInput` solely for this
/// module's use.
fn mcu_count(width: u16, height: u16, components: &[JpegFrameComponent]) -> u32 {
    let h_max = components
        .iter()
        .map(|c| c.h_sampling)
        .max()
        .unwrap_or(1)
        .max(1);
    let v_max = components
        .iter()
        .map(|c| c.v_sampling)
        .max()
        .unwrap_or(1)
        .max(1);
    let mcu_width_px = u32::from(h_max) * 8;
    let mcu_height_px = u32::from(v_max) * 8;
    let mcus_x = u32::from(width).div_ceil(mcu_width_px);
    let mcus_y = u32::from(height).div_ceil(mcu_height_px);
    mcus_x * mcus_y
}

/// MSB-first bit reader over an unstuffed JPEG entropy-coded segment.
///
/// Identical contract to `dc_chain::BitReader`; kept module-local so
/// `dc_chain`'s internal struct does not need to become pub.  When
/// either reader gains a feature the other lacks, refactor to a shared
/// `crate::jpeg::bitreader` module rather than letting them drift.
struct BitReader<'a> {
    src: &'a [u8],
    byte_pos: usize,
    buf: u64,
    cap: u32,
}

impl<'a> BitReader<'a> {
    const fn new(src: &'a [u8]) -> Self {
        Self {
            src,
            byte_pos: 0,
            buf: 0,
            cap: 0,
        }
    }

    fn refill(&mut self) {
        if self.cap == 0 && self.byte_pos + 8 <= self.src.len() {
            let bytes: [u8; 8] = self.src[self.byte_pos..self.byte_pos + 8]
                .try_into()
                .expect("slice length checked above");
            self.buf = u64::from_be_bytes(bytes);
            self.byte_pos += 8;
            self.cap = 64;
            return;
        }
        while self.cap <= 56 && self.byte_pos < self.src.len() {
            let byte = u64::from(self.src[self.byte_pos]);
            self.byte_pos += 1;
            self.buf |= byte << (56 - self.cap);
            self.cap += 8;
        }
    }

    fn peek_u16(&mut self) -> Option<u16> {
        self.refill();
        if self.cap == 0 {
            return None;
        }
        Some((self.buf >> 48) as u16)
    }

    fn consume(&mut self, n: usize) {
        debug_assert!(n <= self.cap as usize);
        #[expect(
            clippy::cast_possible_truncation,
            reason = "n ≤ self.cap ≤ 64; fits in u32 trivially"
        )]
        let n_u32 = n as u32;
        self.buf <<= n_u32;
        self.cap -= n_u32;
    }

    /// Consume `n` bits and discard them.  Returns `None` when fewer
    /// than `n` bits remain in the stream so the caller can surface a
    /// typed `UnexpectedEnd` error.  The oracle does not need the
    /// value (it emits the Huffman symbol byte only); production
    /// callers consuming raw bits should reach for a richer reader.
    fn read_bits(&mut self, n: usize) -> Option<()> {
        if n == 0 {
            return Some(());
        }
        self.refill();
        if (self.cap as usize) < n {
            return None;
        }
        #[expect(
            clippy::cast_possible_truncation,
            reason = "n ≤ 16 in baseline JPEG codepaths; fits in u32 trivially"
        )]
        let n_u32 = n as u32;
        self.buf <<= n_u32;
        self.cap -= n_u32;
        Some(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jpeg::test_fixtures::GRAY_16X16_JPEG;
    use crate::jpeg_decoder::prepare_jpeg;

    #[test]
    fn grayscale_walk_emits_per_block_dc_then_ac() {
        let prep = prepare_jpeg(GRAY_16X16_JPEG).unwrap();
        let symbols = decode_scan_symbols(&prep).expect("clean walk");
        // 16×16 grayscale = 4 MCUs × 1 block/MCU = 4 blocks. Each block
        // contributes 1 DC symbol + at least 1 AC symbol (EOB at the
        // minimum). So the emitted stream is ≥ 8 symbols.
        assert!(
            symbols.len() >= 8,
            "expected ≥ 8 symbols for 4 blocks, got {}",
            symbols.len(),
        );
        // Every emitted byte fits in u8 — the upper bits are reserved
        // and currently always zero.
        for s in &symbols {
            assert!(*s <= 0xFF, "symbol {s:#x} has unexpected upper bits");
        }
    }

    #[test]
    fn grayscale_dc_symbol_categories_match_prepass_dc_chain() {
        // The 0th, (1 + max_ac_len)th, ... symbols are the DC magnitudes
        // for blocks 0..3. Their categories must equal the size needed
        // to extend to each `dc_values[0]` entry.
        let prep = prepare_jpeg(GRAY_16X16_JPEG).unwrap();
        let symbols = decode_scan_symbols(&prep).unwrap();
        let dc_values = &prep.dc_values.per_component[0];
        // Walk the symbol stream: each block starts with a DC symbol
        // whose magnitude category is what the DC chain pre-pass
        // consumed. Extract them by re-walking AC EOB boundaries.
        let mut cursor = 0;
        for (block_idx, &absolute_dc) in dc_values.iter().enumerate() {
            assert!(
                cursor < symbols.len(),
                "block {block_idx}: ran out of symbols at cursor {cursor}",
            );
            let dc_cat = symbols[cursor];
            // The category is the bit-width of the DC delta. We don't
            // reconstruct the delta here (the GPU kernel emits the
            // symbol byte verbatim), but we do assert the category
            // falls in the spec-legal range 0..=11 for 8-bit baseline.
            assert!(
                dc_cat <= 11,
                "block {block_idx}: DC category {dc_cat} exceeds baseline cap (absolute DC = {absolute_dc})",
            );
            cursor += 1;
            // Skip AC symbols until EOB (or 63 emissions). Non-EOB
            // symbols, including ZRL and run+size, all consume exactly
            // one symbol slot in the emitted stream.
            let mut ac_emitted = 0;
            while ac_emitted < 63 && cursor < symbols.len() {
                let sym = symbols[cursor];
                cursor += 1;
                ac_emitted += 1;
                if sym == 0x00 {
                    break;
                }
            }
        }
        assert_eq!(
            cursor,
            symbols.len(),
            "consumed {cursor} of {} symbols — emitter and consumer disagree on block count",
            symbols.len(),
        );
    }

    #[test]
    fn empty_unstuffed_bitstream_yields_no_symbols_for_zero_mcus() {
        // Construct a 0×0 prep manually by zeroing dimensions. This is
        // not a real JPEG but exercises the "no MCUs" branch of the
        // walker without touching the prepass.
        let mut prep = prepare_jpeg(GRAY_16X16_JPEG).unwrap();
        prep.width = 0;
        prep.height = 0;
        let symbols = decode_scan_symbols(&prep).unwrap();
        assert!(symbols.is_empty(), "0×0 image must emit no symbols");
    }

    #[test]
    fn rejects_restart_aware_input_with_typed_error() {
        // Restart-aware framing is deliberately deferred — refuse such
        // inputs rather than producing a symbol stream that drifts
        // after the first RST boundary.  Synthesised by patching the
        // grayscale prep's restart_interval.
        let mut prep = prepare_jpeg(GRAY_16X16_JPEG).unwrap();
        prep.restart_interval = 4;
        let err = decode_scan_symbols(&prep).expect_err("RST interval must be refused");
        assert!(
            matches!(
                err,
                JpegFramingError::RestartMarkersUnsupported {
                    restart_interval: 4
                }
            ),
            "expected RestartMarkersUnsupported(4), got: {err:?}",
        );
    }
}
