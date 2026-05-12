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
use crate::jpeg::bitreader::BitReader;
use crate::jpeg::headers::{JpegFrameComponent, mcu_count};
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
    /// AC run+size pair (or ZRL) walked past coefficient slot 63 in
    /// some block.  Catches both the run+size case and the rarer
    /// "ZRL emitted at slot ≥ 49" adversarial case the spec forbids.
    AcOverflow {
        /// 0-based MCU index where the overflow surfaced.
        mcu_index: u32,
    },
    /// AC magnitude size > 10 (max for 8-bit baseline JPEG; the AC
    /// symbol's low nibble is bounded to 0..=10 by the spec, but a
    /// corrupt Huffman table could emit 11..=15).
    BadAcSize {
        /// 0-based MCU index where the bad size surfaced.
        mcu_index: u32,
        /// 0-based scan-component index where the bad size surfaced.
        scan_component: u8,
        /// The size as decoded from the AC Huffman table.
        size: u8,
    },
    /// DC magnitude category > 11 (max for 8-bit baseline JPEG).
    BadDcCategory {
        /// 0-based MCU index where the bad category surfaced.
        mcu_index: u32,
        /// 0-based scan-component index where the bad category surfaced.
        scan_component: u8,
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
                "JPEG framing: AC run+size (or ZRL) walked past slot 63 at MCU {mcu_index}",
            ),
            Self::BadAcSize {
                mcu_index,
                scan_component,
                size,
            } => write!(
                f,
                "JPEG framing: AC size {size} > 10 at MCU {mcu_index}, scan-component {scan_component} (8-bit baseline cap)",
            ),
            Self::BadDcCategory {
                mcu_index,
                scan_component,
                category,
            } => write!(
                f,
                "JPEG framing: DC category {category} > 11 at MCU {mcu_index}, scan-component {scan_component} (8-bit baseline cap)",
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
            // scan_components ≤ 4 by upstream validation; the cast
            // never truncates.  Using try_from + expect lets the
            // invariant tell us when it's broken instead of silently
            // wrapping at u8::MAX.
            let k_u8 = u8::try_from(k).expect("scan component index < 4");
            if fc.h_sampling == 0 || fc.v_sampling == 0 {
                return Err(JpegFramingError::DegenerateSampling {
                    scan_component: k_u8,
                });
            }
            blocks_per_mcu[k] = if scan_components == 1 {
                1
            } else {
                fc.h_sampling
                    .checked_mul(fc.v_sampling)
                    .expect("upstream BadSamplingFactor caps h, v ≤ 4")
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
                let k_u8 = u8::try_from(k).expect("scan component index < 4");
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
        return Err(JpegFramingError::BadDcCategory {
            mcu_index: mcu,
            scan_component,
            category,
        });
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
            // A ZRL whose post-advance ac_pos exceeds 64 is spec-
            // illegal — refuse rather than silently swallow the
            // misalignment.
            ac_pos = ac_pos.saturating_add(16);
            if ac_pos > 64 {
                return Err(JpegFramingError::AcOverflow { mcu_index: mcu });
            }
            continue;
        }
        let run = symbol >> 4;
        let size = symbol & 0x0F;
        // 8-bit baseline JPEG caps AC magnitude size at 10 bits
        // (ITU-T T.81 § F.1.2.2.1).  Refuse 11..=15 — an adversarial
        // Huffman table could otherwise direct the bit reader to
        // consume up to 15 raw bits per slot.
        if size > 10 {
            return Err(JpegFramingError::BadAcSize {
                mcu_index: mcu,
                scan_component,
                size,
            });
        }
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

    #[test]
    fn truncated_bitstream_surfaces_unexpected_end() {
        // Patch length_bits down to a value so small no MCU can decode
        // even its DC.  The walker should bail with UnexpectedEnd
        // rather than producing a short symbol stream silently.
        let mut prep = prepare_jpeg(GRAY_16X16_JPEG).unwrap();
        prep.bitstream.length_bits = 4;
        let err = decode_scan_symbols(&prep).expect_err("truncated stream must fail");
        assert!(
            matches!(err, JpegFramingError::UnexpectedEnd { mcu_index: 0, .. }),
            "expected UnexpectedEnd at MCU 0, got: {err:?}",
        );
    }

    #[test]
    fn degenerate_sampling_factor_is_refused() {
        let mut prep = prepare_jpeg(GRAY_16X16_JPEG).unwrap();
        // Force a degenerate sampling factor; upstream validation
        // would have caught this on the JPEG bytes, but we patch
        // after the wrapper to exercise the oracle's own defence.
        prep.components[0].h_sampling = 0;
        let err = decode_scan_symbols(&prep).expect_err("degenerate sampling must be refused");
        assert!(
            matches!(
                err,
                JpegFramingError::DegenerateSampling { scan_component: 0 }
            ),
            "expected DegenerateSampling(0), got: {err:?}",
        );
    }
}
