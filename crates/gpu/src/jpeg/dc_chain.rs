//! DC differential chain resolution.
//!
//! In a baseline JPEG scan, each 8×8 block's DC coefficient is encoded as a
//! difference from the previous block's DC value for the same component.
//! The encoder writes:
//!
//! ```text
//! DC delta for block i = DC[i] − DC[i−1]   (DC[−1] := 0; reset to 0 after RSTn)
//! ```
//!
//! This chain is **inherently serial** — block `i` cannot be decoded without
//! knowing block `i−1`'s absolute DC.  It's the one part of JPEG entropy
//! decoding that doesn't parallelise across the bitstream.
//!
//! Our on-GPU decoder pipeline sidesteps this by **resolving the DC chain on
//! the CPU during Phase 0** and shipping per-MCU absolute DC values to the
//! GPU alongside the (parallelisable) AC coefficient stream.  DC bits are a
//! tiny fraction of the entropy stream — typically 1–3% of the total bits
//! per MCU — so the CPU pre-pass cost is dominated by AC skip work, not DC
//! arithmetic.
//!
//! ## Wire format
//!
//! Per JPEG ISO/IEC 10918-1 § F.1.2.1.3 ("Encoding of DC coefficients"):
//!
//! 1. Read a Huffman code from the DC table → `T` (a 4-bit "size" 0..=11).
//! 2. Read `T` more raw bits → `bits`.
//! 3. The signed delta is `extend(bits, T)`:
//!    - if the high bit of `bits` is 1: delta = `bits`                 (positive)
//!    - else:                            delta = `bits − ((1 << T) − 1)` (negative).
//!    - if `T == 0`:                     delta = 0.
//!
//! AC coefficients use the same `extend` rule but with a different code
//! semantic (run/size pairs).  We **skip** AC during the DC pre-pass — we
//! only need to advance the bit position past the 63 AC coefficients of each
//! block.  This module does not produce AC outputs; that's Phase 1's job.

use super::canonical::{CanonicalCodebook, CanonicalEntry};
use super::headers::{DhtClass, JpegHeaders};

/// Per-component absolute DC values, one per MCU, in scan order.
///
/// `values[c]` is a `Vec<i32>` of length [`JpegHeaders::num_mcus`] for
/// component `c`.  Allocated up front so the GPU upload is a single
/// `cudaMemcpyAsync` per component.
#[derive(Debug, Clone)]
pub struct DcValues {
    /// Per-component scan-order DC values.  Outer index matches
    /// `JpegHeaders::frame_components`.
    pub per_component: Vec<Vec<i32>>,
}

/// Errors emitted by [`resolve_dc_chain`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DcChainError {
    /// Bitstream ended before all expected DC values were decoded.
    UnexpectedEnd,
    /// A Huffman lookup landed on an unassigned slot — bitstream is corrupt
    /// or the wrong table was used.
    InvalidCode {
        /// MCU index where the failure was observed.
        mcu_index: u32,
        /// Component index within the MCU.
        component_index: u8,
    },
    /// DC code declared a magnitude category > 11 (max for 8-bit baseline).
    BadDcCategory {
        /// Category claimed by the Huffman code.
        category: u8,
    },
    /// AC code declared run+size pair beyond the 63-coefficient block.
    AcOverflow {
        /// MCU index where AC ran past coefficient 63.
        mcu_index: u32,
    },
    /// SOS scan referenced a Huffman table that wasn't declared in any DHT.
    MissingHuffmanTable {
        /// Class (DC or AC).
        class: DhtClass,
        /// Selector value from SOS.
        selector: u8,
    },
    /// SOF declared a quantiser/sampling combination we don't support yet
    /// (e.g. interleaved scan with > 4 components, or v_sampling > 4).
    UnsupportedScanShape,
}

impl std::fmt::Display for DcChainError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnexpectedEnd => write!(f, "DC chain pre-pass: bitstream ended unexpectedly"),
            Self::InvalidCode {
                mcu_index,
                component_index,
            } => write!(
                f,
                "DC chain pre-pass: invalid Huffman code at MCU {mcu_index}, component {component_index}",
            ),
            Self::BadDcCategory { category } => {
                write!(
                    f,
                    "DC chain pre-pass: DC magnitude category {category} > 11"
                )
            }
            Self::AcOverflow { mcu_index } => write!(
                f,
                "DC chain pre-pass: AC run/size pair overflows block at MCU {mcu_index}",
            ),
            Self::MissingHuffmanTable { class, selector } => {
                let kind = match class {
                    DhtClass::Dc => "DC",
                    DhtClass::Ac => "AC",
                };
                write!(
                    f,
                    "DC chain pre-pass: scan references {kind} table {selector} but no DHT declared it",
                )
            }
            Self::UnsupportedScanShape => write!(f, "DC chain pre-pass: unsupported scan shape"),
        }
    }
}

impl std::error::Error for DcChainError {}

/// Walk the unstuffed entropy-coded segment, decoding only enough bits to
/// extract every DC coefficient's absolute value.  AC bits are skipped.
///
/// `unstuffed` is the entropy-coded data after [`super::unstuff::unstuff_into`]
/// has been applied.  `headers` provides the SOS scan metadata; this function
/// looks up the right DC and AC Huffman tables via `headers.huffman()`.
///
/// # Errors
///
/// Returns [`DcChainError`] on malformed or unsupported input.  All errors
/// are non-fatal at the caller level — the caller can fall back to a CPU
/// decoder for this image.
pub fn resolve_dc_chain(
    headers: &JpegHeaders<'_>,
    unstuffed: &[u8],
) -> Result<DcValues, DcChainError> {
    if headers.scan.component_count == 0 {
        return Err(DcChainError::UnsupportedScanShape);
    }
    let scan_components = headers.scan.component_count as usize;

    let (dc_codebooks, ac_codebooks) = build_scan_codebooks(headers)?;
    let scan_to_frame = map_scan_to_frame(headers)?;
    let n_mcus = headers.num_mcus();
    let frame_components = &headers.frame_components[..headers.components as usize];
    let mut per_component = allocate_dc_output(headers, n_mcus, &scan_to_frame);

    // Per-component running DC predictor; reset to 0 at start.  RST-marker
    // handling is the caller's responsibility — we resolve a contiguous
    // segment here.
    let mut dc_predictor = [0i32; 4];
    let mut bits = BitReader::new(unstuffed);

    for mcu_idx in 0..n_mcus {
        for (sc_idx, sc) in headers.scan.components[..scan_components]
            .iter()
            .enumerate()
        {
            let frame_idx = scan_to_frame[sc_idx];
            let fc = &frame_components[frame_idx];
            let blocks_per_mcu = if scan_components == 1 {
                1
            } else {
                usize::from(fc.h_sampling) * usize::from(fc.v_sampling)
            };

            let dc_cb = dc_codebooks[sc.dc_table as usize].as_ref().ok_or(
                DcChainError::MissingHuffmanTable {
                    class: DhtClass::Dc,
                    selector: sc.dc_table,
                },
            )?;
            let ac_cb = ac_codebooks[sc.ac_table as usize].as_ref().ok_or(
                DcChainError::MissingHuffmanTable {
                    class: DhtClass::Ac,
                    selector: sc.ac_table,
                },
            )?;

            let comp_index_u8 = u8::try_from(sc_idx).unwrap_or(u8::MAX);

            for _block in 0..blocks_per_mcu {
                let delta = decode_dc_delta(&mut bits, dc_cb, mcu_idx, comp_index_u8)?;
                dc_predictor[sc_idx] = dc_predictor[sc_idx].wrapping_add(delta);
                per_component[sc_idx].push(dc_predictor[sc_idx]);
                skip_ac_coefficients(&mut bits, ac_cb, mcu_idx, comp_index_u8)?;
            }
        }
    }

    Ok(DcValues { per_component })
}

/// Build canonical Huffman lookup tables for every (class, selector) the
/// scan header references.  Returns parallel arrays indexed by selector
/// (0..=3); only the slots referenced by the scan are populated.
fn build_scan_codebooks(
    headers: &JpegHeaders<'_>,
) -> Result<
    (
        [Option<CanonicalCodebook>; 4],
        [Option<CanonicalCodebook>; 4],
    ),
    DcChainError,
> {
    let scan_components = headers.scan.component_count as usize;
    let mut dc_codebooks: [Option<CanonicalCodebook>; 4] = [None, None, None, None];
    let mut ac_codebooks: [Option<CanonicalCodebook>; 4] = [None, None, None, None];

    for sc in &headers.scan.components[..scan_components] {
        if dc_codebooks[sc.dc_table as usize].is_none() {
            dc_codebooks[sc.dc_table as usize] =
                Some(load_codebook(headers, DhtClass::Dc, sc.dc_table)?);
        }
        if ac_codebooks[sc.ac_table as usize].is_none() {
            ac_codebooks[sc.ac_table as usize] =
                Some(load_codebook(headers, DhtClass::Ac, sc.ac_table)?);
        }
    }
    Ok((dc_codebooks, ac_codebooks))
}

fn load_codebook(
    headers: &JpegHeaders<'_>,
    class: DhtClass,
    selector: u8,
) -> Result<CanonicalCodebook, DcChainError> {
    let table = headers
        .huffman(class, selector)
        .ok_or(DcChainError::MissingHuffmanTable { class, selector })?;
    CanonicalCodebook::build(table).map_err(|_| DcChainError::InvalidCode {
        mcu_index: 0,
        component_index: 0,
    })
}

/// Map each scan component (by index) to its frame-component index.
fn map_scan_to_frame(headers: &JpegHeaders<'_>) -> Result<[usize; 4], DcChainError> {
    let scan_components = headers.scan.component_count as usize;
    let frame_components = &headers.frame_components[..headers.components as usize];
    let mut scan_to_frame = [0usize; 4];
    for (i, sc) in headers.scan.components[..scan_components]
        .iter()
        .enumerate()
    {
        scan_to_frame[i] = frame_components
            .iter()
            .position(|fc| fc.id == sc.id)
            .ok_or(DcChainError::UnsupportedScanShape)?;
    }
    Ok(scan_to_frame)
}

/// Pre-size the per-component output `Vec`s so the entropy walk does no
/// reallocation.
fn allocate_dc_output(
    headers: &JpegHeaders<'_>,
    n_mcus: u32,
    scan_to_frame: &[usize; 4],
) -> Vec<Vec<i32>> {
    let scan_components = headers.scan.component_count as usize;
    let frame_components = &headers.frame_components[..headers.components as usize];
    (0..scan_components)
        .map(|i| {
            let fc = &frame_components[scan_to_frame[i]];
            let blocks_per_mcu = usize::from(fc.h_sampling) * usize::from(fc.v_sampling);
            let count = if scan_components == 1 {
                n_mcus as usize
            } else {
                (n_mcus as usize) * blocks_per_mcu
            };
            Vec::with_capacity(count)
        })
        .collect()
}

/// Decode one block's DC delta and EXTEND to a signed value.
fn decode_dc_delta(
    bits: &mut BitReader<'_>,
    dc_cb: &CanonicalCodebook,
    mcu_idx: u32,
    comp_index: u8,
) -> Result<i32, DcChainError> {
    let dc_entry = decode_one(bits, dc_cb).ok_or(DcChainError::InvalidCode {
        mcu_index: mcu_idx,
        component_index: comp_index,
    })?;
    let category = dc_entry.symbol;
    if category > 11 {
        return Err(DcChainError::BadDcCategory { category });
    }
    let raw = if category == 0 {
        0i32
    } else {
        bits.read_bits(category as usize)
            .ok_or(DcChainError::UnexpectedEnd)?
    };
    Ok(extend(raw, category))
}

/// Walk the 63 AC coefficients of one block, advancing the bit reader past
/// them but discarding the values (Phase 0 only resolves the DC chain).
fn skip_ac_coefficients(
    bits: &mut BitReader<'_>,
    ac_cb: &CanonicalCodebook,
    mcu_idx: u32,
    comp_index: u8,
) -> Result<(), DcChainError> {
    let mut ac_pos = 1;
    while ac_pos < 64 {
        let ac_entry = decode_one(bits, ac_cb).ok_or(DcChainError::InvalidCode {
            mcu_index: mcu_idx,
            component_index: comp_index,
        })?;
        if ac_entry.symbol == 0x00 {
            // EOB: rest of block is zero.
            break;
        }
        if ac_entry.symbol == 0xF0 {
            // ZRL: 16 zero coefficients.
            ac_pos += 16;
            if ac_pos >= 64 {
                break;
            }
            continue;
        }
        let run = (ac_entry.symbol >> 4) as usize;
        let size = (ac_entry.symbol & 0x0F) as usize;
        ac_pos += run + 1;
        if ac_pos > 64 {
            return Err(DcChainError::AcOverflow { mcu_index: mcu_idx });
        }
        // Skip the value bits without interpreting them.
        let _ = bits.read_bits(size).ok_or(DcChainError::UnexpectedEnd)?;
    }
    Ok(())
}

/// JPEG `EXTEND` operation (spec § F.1.2.1.1, Figure F.12).  Sign-extends an
/// `nbits`-wide unsigned magnitude into a signed integer.
fn extend(value: i32, nbits: u8) -> i32 {
    if nbits == 0 {
        return 0;
    }
    let vt = 1i32 << (nbits - 1);
    if value < vt {
        value + (-1i32 << nbits) + 1
    } else {
        value
    }
}

/// Decode the next codeword.  Returns `None` if the bit pattern doesn't
/// resolve to a known symbol (stream corrupt or wrong table).
fn decode_one(bits: &mut BitReader<'_>, cb: &CanonicalCodebook) -> Option<CanonicalEntry> {
    let prefix = bits.peek_u16()?;
    let entry = cb.lookup(prefix);
    if entry.num_bits == 0 {
        return None;
    }
    bits.consume(entry.num_bits as usize);
    Some(entry)
}

/// Bit reader over an unstuffed JPEG entropy-coded stream.  MSB-first.
struct BitReader<'a> {
    src: &'a [u8],
    /// Byte position of the next byte to load into the buffer.
    byte_pos: usize,
    /// Buffered bits, MSB-aligned.  `cap` bits are valid in the high end;
    /// the low `64 - cap` bits are zero.
    buf: u64,
    /// Number of valid bits in `buf` (0..=64).
    cap: u32,
}

impl<'a> BitReader<'a> {
    fn new(src: &'a [u8]) -> Self {
        Self {
            src,
            byte_pos: 0,
            buf: 0,
            cap: 0,
        }
    }

    /// Refill `buf` with as many full bytes as fit, up to 8 bytes / 64 bits
    /// or until input runs out.
    fn refill(&mut self) {
        while self.cap <= 56 && self.byte_pos < self.src.len() {
            let byte = u64::from(self.src[self.byte_pos]);
            self.byte_pos += 1;
            self.buf |= byte << (56 - self.cap);
            self.cap += 8;
        }
    }

    /// Peek the next 16 bits MSB-first; pads with zeros if fewer remain.
    /// Returns `None` only if the buffer is completely empty.
    fn peek_u16(&mut self) -> Option<u16> {
        self.refill();
        if self.cap == 0 {
            return None;
        }
        // Top 16 bits of buf, with zero-pad if cap < 16 (high bits remain
        // as we left them; refill above appends from the high end).  The
        // shift makes the truncation provably lossless.
        Some((self.buf >> 48) as u16)
    }

    /// Consume `n` bits from the buffer.  Caller must have peeked at least
    /// `n` bits via `peek_u16` (which refills) before calling.
    ///
    /// `n` is bounded by `cap` (≤ 64) which itself is bounded by the
    /// `u64` buffer width, so the `u32` cast is lossless in practice.
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

    /// Read `n` bits MSB-first as an unsigned integer.  Returns `None` if
    /// fewer than `n` bits remain in the stream.
    ///
    /// JPEG codewords cap at 16 bits, so `n` is always ≤ 16 in practice; the
    /// cast and the high-16 extraction are both lossless under that constraint.
    fn read_bits(&mut self, n: usize) -> Option<i32> {
        if n == 0 {
            return Some(0);
        }
        self.refill();
        if (self.cap as usize) < n {
            return None;
        }
        debug_assert!(n <= 16, "JPEG codewords are at most 16 bits");
        #[expect(
            clippy::cast_possible_truncation,
            reason = "n ≤ 16 (debug-asserted); fits in u32 trivially"
        )]
        let n_u32 = n as u32;
        // Top `n` bits.
        #[expect(
            clippy::cast_possible_truncation,
            reason = "n ≤ 16 implies the right-shifted value fits in u32"
        )]
        let bits = (self.buf >> (64 - n_u32)) as u32;
        self.buf <<= n_u32;
        self.cap -= n_u32;
        #[expect(
            clippy::cast_possible_wrap,
            reason = "n ≤ 16 in JPEG; value fits in i32"
        )]
        Some(bits as i32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extend_zero_returns_zero() {
        assert_eq!(extend(0, 0), 0);
    }

    #[test]
    fn extend_positive_one_bit() {
        assert_eq!(extend(1, 1), 1);
    }

    #[test]
    fn extend_negative_one_bit() {
        assert_eq!(extend(0, 1), -1);
    }

    #[test]
    fn extend_three_bits_examples() {
        // From JPEG spec Table F.1: nbits=3 maps to range -7..=-4 ∪ 4..=7.
        assert_eq!(extend(4, 3), 4);
        assert_eq!(extend(7, 3), 7);
        assert_eq!(extend(0, 3), -7);
        assert_eq!(extend(3, 3), -4);
    }

    #[test]
    fn bitreader_reads_bytes_msb_first() {
        let src = [0b1010_1100, 0b1111_0000];
        let mut br = BitReader::new(&src);
        assert_eq!(br.read_bits(4), Some(0b1010));
        assert_eq!(br.read_bits(4), Some(0b1100));
        assert_eq!(br.read_bits(4), Some(0b1111));
        assert_eq!(br.read_bits(4), Some(0b0000));
        assert_eq!(br.read_bits(1), None);
    }

    #[test]
    fn bitreader_handles_mixed_widths() {
        let src = [0b1110_0101, 0b1100_0011];
        let mut br = BitReader::new(&src);
        assert_eq!(br.read_bits(3), Some(0b111));
        assert_eq!(br.read_bits(5), Some(0b00101));
        assert_eq!(br.read_bits(8), Some(0b1100_0011));
    }

    #[test]
    fn bitreader_peek_does_not_consume() {
        let src = [0xAB, 0xCD];
        let mut br = BitReader::new(&src);
        let p1 = br.peek_u16().unwrap();
        let p2 = br.peek_u16().unwrap();
        assert_eq!(p1, p2);
        assert_eq!(p1, 0xABCD);
    }

    #[test]
    fn bitreader_pads_short_input_with_zeros_in_peek() {
        let src = [0xAB];
        let mut br = BitReader::new(&src);
        assert_eq!(br.peek_u16(), Some(0xAB00));
    }
}
