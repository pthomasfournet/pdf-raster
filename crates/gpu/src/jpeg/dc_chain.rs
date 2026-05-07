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
//!
//! ## Restart markers
//!
//! When [`super::unstuff::unstuff_into`] strips a `0xFF 0xDn` marker it
//! records the resulting byte position in the unstuffed stream.  At each
//! such position the DC predictor is reset to zero for every component
//! (JPEG § F.1.1.5) and the bit reader's mid-byte phase is realigned to a
//! byte boundary.  Skipping these resets is a silent correctness bug — for
//! any JPEG with `restart_interval > 0` the DC values after the first
//! restart would accumulate the wrong predictor.

use super::canonical::{CanonicalCodebook, CanonicalEntry};
use super::headers::{DhtClass, JpegHeaders};
use super::unstuff::RstPosition;

/// Per-component absolute DC values, one per block, in scan order.
///
/// `per_component[c]` is a `Vec<i32>` of length `n_mcus × blocks_per_mcu` for
/// component `c`.  Allocated up front so the GPU upload is a single
/// `cudaMemcpyAsync` per component.
#[derive(Debug, Clone)]
pub struct DcValues {
    /// Per-component scan-order DC values.  Outer index matches scan order
    /// (`headers.scan.components[i]`), not frame order.
    pub per_component: Vec<Vec<i32>>,
}

/// Bounded number of MCUs we will accept from a single image.
///
/// 65535 px / 8 px-per-MCU ≈ 8192 MCUs per axis even at 1×1 sampling, so
/// 8192² caps a worst-case JPEG that the SOF0 width/height fields can express.
/// Anything beyond this means the SOF0 dimensions or sampling factors are
/// adversarial; we refuse rather than honour the resulting allocation.
pub(crate) const MAX_SAFE_MCUS: usize = 8192 * 8192;

/// Errors emitted by [`resolve_dc_chain`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DcChainError {
    /// Bitstream ended before all expected DC values were decoded.
    UnexpectedEnd {
        /// MCU index where the truncation was observed.
        mcu_index: u32,
        /// Component index within the MCU.
        component_index: u8,
    },
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
    /// SOS scan referenced a Huffman table that wasn't pre-built or wasn't
    /// declared in any DHT.
    MissingHuffmanTable {
        /// Class (DC or AC).
        class: DhtClass,
        /// Selector value from SOS.
        selector: u8,
    },
    /// SOF declared a quantiser/sampling combination we don't support yet
    /// (e.g. interleaved scan with > 4 components, or `v_sampling` > 4).
    UnsupportedScanShape,
    /// Image dimensions × sampling factors yield more MCUs than [`MAX_SAFE_MCUS`].
    /// Refuses the image rather than allocating an adversarial output buffer.
    TooManyMcus {
        /// MCU count requested.
        requested: u64,
    },
    /// A scan-component referenced a frame-component ID that does not appear
    /// in the SOF0 component list.
    UnknownScanComponent {
        /// The unknown component identifier.
        component_id: u8,
    },
}

impl std::fmt::Display for DcChainError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnexpectedEnd {
                mcu_index,
                component_index,
            } => write!(
                f,
                "DC chain pre-pass: bitstream ended unexpectedly at MCU {mcu_index}, component {component_index}",
            ),
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
            Self::MissingHuffmanTable { class, selector } => write!(
                f,
                "DC chain pre-pass: scan references {class} table {selector} but no codebook was provided",
            ),
            Self::UnsupportedScanShape => write!(f, "DC chain pre-pass: unsupported scan shape"),
            Self::TooManyMcus { requested } => write!(
                f,
                "DC chain pre-pass: image declares {requested} MCUs which exceeds the safety cap of {MAX_SAFE_MCUS}",
            ),
            Self::UnknownScanComponent { component_id } => write!(
                f,
                "DC chain pre-pass: scan references component id {component_id} not declared in SOF0",
            ),
        }
    }
}

impl std::error::Error for DcChainError {}

/// One block-emission step in the DC walk.  Carries enough metadata to
/// produce a precise error if anything fails.
#[derive(Debug, Clone, Copy)]
struct ScanStep<'a> {
    sc_idx: u8,
    sc_idx_usize: usize,
    blocks_per_mcu: usize,
    dc_cb: &'a CanonicalCodebook,
    ac_cb: &'a CanonicalCodebook,
}

/// Number of 8×8 blocks emitted per MCU per scan-component.
///
/// Per JPEG ISO/IEC 10918-1 § A.2.3: a non-interleaved scan (one component)
/// always emits one block per MCU regardless of that component's sampling
/// factors.  Interleaved scans emit `h_sampling × v_sampling` blocks per
/// component per MCU.
fn blocks_per_mcu(scan_components: usize, h_sampling: u8, v_sampling: u8) -> usize {
    if scan_components == 1 {
        1
    } else {
        usize::from(h_sampling) * usize::from(v_sampling)
    }
}

/// Walk the unstuffed entropy-coded segment, decoding only enough bits to
/// extract every DC coefficient's absolute value.  AC bits are skipped.
///
/// Caller must pre-build the canonical Huffman codebooks for every (class,
/// selector) the scan references and pass them in via `dc_codebooks` and
/// `ac_codebooks` indexed by selector (0..=3).  This avoids rebuilding the
/// 128 KB-per-table lookup arrays inside `resolve_dc_chain` when the caller
/// already has them — the on-GPU pipeline keeps the AC codebooks alive for
/// Phase 1 and would otherwise duplicate construction.
///
/// `rst_positions` carries every restart marker encountered by
/// [`super::unstuff::unstuff_into`].  At each marker the DC predictor is
/// reset to zero for every component and the bit reader is realigned to the
/// next byte boundary, per JPEG § F.1.1.5.
///
/// # Errors
///
/// Returns [`DcChainError`] on malformed or unsupported input.  All errors
/// are non-fatal at the caller level — the caller can fall back to a CPU
/// decoder for this image.
///
/// # Panics
///
/// Panics only on heap-allocation failure (the standard Rust OOM panic) when
/// pre-allocating the per-component DC value vectors.  The total allocation
/// is bounded by `MAX_SAFE_MCUS × 16 × sizeof(i32)` ≈ 4 GiB across all
/// components in the worst case, but is gated by the [`DcChainError::TooManyMcus`]
/// check before any allocation happens, so adversarial input gets a clean
/// error rather than a panic.
pub fn resolve_dc_chain(
    headers: &JpegHeaders<'_>,
    unstuffed: &[u8],
    rst_positions: &[RstPosition],
    dc_codebooks: &[Option<CanonicalCodebook>; 4],
    ac_codebooks: &[Option<CanonicalCodebook>; 4],
) -> Result<DcValues, DcChainError> {
    let scan_components = usize::from(headers.scan.component_count);
    if scan_components == 0 || scan_components > 4 {
        return Err(DcChainError::UnsupportedScanShape);
    }

    let scan_to_frame = map_scan_to_frame(headers)?;
    let n_mcus = u64::from(headers.num_mcus());
    let frame_components = &headers.frame_components[..usize::from(headers.components)];

    // Resolve the per-step metadata up front.  This both validates every
    // codebook reference and gives us a fixed-size array to index inside the
    // hot loop without re-doing scan-to-frame mapping or codebook lookup.
    let mut steps: [Option<ScanStep<'_>>; 4] = [None, None, None, None];
    let mut total_blocks_per_mcu: u64 = 0;
    for (sc_idx, sc) in headers.scan.components[..scan_components]
        .iter()
        .enumerate()
    {
        let fc = &frame_components[scan_to_frame[sc_idx]];
        let bpm = blocks_per_mcu(scan_components, fc.h_sampling, fc.v_sampling);
        if bpm == 0 {
            return Err(DcChainError::UnsupportedScanShape);
        }
        let dc_cb = dc_codebooks[usize::from(sc.dc_table)].as_ref().ok_or(
            DcChainError::MissingHuffmanTable {
                class: DhtClass::Dc,
                selector: sc.dc_table,
            },
        )?;
        let ac_cb = ac_codebooks[usize::from(sc.ac_table)].as_ref().ok_or(
            DcChainError::MissingHuffmanTable {
                class: DhtClass::Ac,
                selector: sc.ac_table,
            },
        )?;
        let sc_idx_u8 = u8::try_from(sc_idx).expect("scan_components ≤ 4 guarantees fit in u8");
        steps[sc_idx] = Some(ScanStep {
            sc_idx: sc_idx_u8,
            sc_idx_usize: sc_idx,
            blocks_per_mcu: bpm,
            dc_cb,
            ac_cb,
        });
        total_blocks_per_mcu = total_blocks_per_mcu.saturating_add(bpm as u64);
    }

    // Cap the output allocation before we honour an adversarial SOF0.
    // Compare in u64 throughout — MAX_SAFE_MCUS fits in usize on every
    // target we ship to, but on a 32-bit host casting `u64 as usize`
    // would silently truncate before the comparison.
    let total_blocks = n_mcus.saturating_mul(total_blocks_per_mcu);
    let max_safe_mcus_u64 = MAX_SAFE_MCUS as u64;
    let max_safe_blocks_u64 = max_safe_mcus_u64.saturating_mul(16);
    if total_blocks > max_safe_blocks_u64 {
        return Err(DcChainError::TooManyMcus {
            requested: total_blocks,
        });
    }
    if n_mcus > max_safe_mcus_u64 {
        return Err(DcChainError::TooManyMcus { requested: n_mcus });
    }

    let mut per_component = allocate_dc_output(scan_components, n_mcus, &steps);

    // Per-component running DC predictor; reset to 0 at start and at every
    // RST boundary.  RST positions are sorted in ascending byte order by
    // construction (unstuff_into emits them as it walks the input).
    let mut dc_predictor = [0i32; 4];
    let mut bits = BitReader::new(unstuffed);
    let mut rst_iter = rst_positions.iter().peekable();

    let n_mcus_u32 = u32::try_from(n_mcus).expect("MAX_SAFE_MCUS guarantees fit in u32");
    for mcu_idx in 0..n_mcus_u32 {
        // Consume every RST boundary the bit reader has crossed (or reached)
        // before decoding this MCU.  At each crossing the DC predictor resets
        // to zero and the bit reader realigns to a byte boundary, per
        // JPEG § F.1.1.5.
        while rst_iter
            .peek()
            .is_some_and(|&&r| bits.byte_position() >= r.byte_offset_in_dst)
        {
            let rst = rst_iter
                .next()
                .expect("peek returned Some immediately above");
            bits.realign_to_byte_at(rst.byte_offset_in_dst);
            dc_predictor = [0i32; 4];
        }

        for step in steps.iter().take(scan_components) {
            let step = step.expect("steps[..scan_components] populated above");
            for _block in 0..step.blocks_per_mcu {
                let delta = decode_dc_delta(&mut bits, step.dc_cb, mcu_idx, step.sc_idx)?;
                dc_predictor[step.sc_idx_usize] =
                    dc_predictor[step.sc_idx_usize].wrapping_add(delta);
                per_component[step.sc_idx_usize].push(dc_predictor[step.sc_idx_usize]);
                skip_ac_coefficients(&mut bits, step.ac_cb, mcu_idx, step.sc_idx)?;
            }
        }
    }

    Ok(DcValues { per_component })
}

/// Map each scan component (by index) to its frame-component index.
fn map_scan_to_frame(headers: &JpegHeaders<'_>) -> Result<[usize; 4], DcChainError> {
    let scan_components = usize::from(headers.scan.component_count);
    let frame_components = &headers.frame_components[..usize::from(headers.components)];
    let mut scan_to_frame = [0usize; 4];
    for (i, sc) in headers.scan.components[..scan_components]
        .iter()
        .enumerate()
    {
        scan_to_frame[i] = frame_components
            .iter()
            .position(|fc| fc.id == sc.id)
            .ok_or(DcChainError::UnknownScanComponent {
                component_id: sc.id,
            })?;
    }
    Ok(scan_to_frame)
}

/// Pre-size the per-component output `Vec`s so the entropy walk does no
/// reallocation.  Caller has already capped `n_mcus × bpm` to ≤
/// `MAX_SAFE_MCUS × 16`, which fits in `usize` on every supported target,
/// so the cast at the end is provably lossless.
fn allocate_dc_output(
    scan_components: usize,
    n_mcus: u64,
    steps: &[Option<ScanStep<'_>>; 4],
) -> Vec<Vec<i32>> {
    (0..scan_components)
        .map(|i| {
            let bpm = steps[i].as_ref().map_or(1u64, |s| s.blocks_per_mcu as u64);
            let total = n_mcus.saturating_mul(bpm);
            let count = usize::try_from(total).unwrap_or(usize::MAX);
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
    let dc_entry = decode_one(bits, dc_cb, mcu_idx, comp_index)?;
    let category = dc_entry.symbol;
    if category > 11 {
        return Err(DcChainError::BadDcCategory { category });
    }
    let raw = if category == 0 {
        0i32
    } else {
        bits.read_bits(category as usize)
            .ok_or(DcChainError::UnexpectedEnd {
                mcu_index: mcu_idx,
                component_index: comp_index,
            })?
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
        let ac_entry = decode_one(bits, ac_cb, mcu_idx, comp_index)?;
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
        let _ = bits.read_bits(size).ok_or(DcChainError::UnexpectedEnd {
            mcu_index: mcu_idx,
            component_index: comp_index,
        })?;
    }
    Ok(())
}

/// JPEG `EXTEND` operation (spec § F.1.2.1.1, Figure F.12).  Sign-extends an
/// `nbits`-wide unsigned magnitude into a signed integer.
///
/// `nbits` ≤ 11 in baseline JPEG (DC); the caller validates the cap before
/// calling.  The implementation tolerates `nbits == 0` (returns 0).
const fn extend(value: i32, nbits: u8) -> i32 {
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

/// Decode the next codeword.  Distinguishes end-of-stream from
/// invalid-code-pattern so callers can return a precise error variant.
fn decode_one(
    bits: &mut BitReader<'_>,
    cb: &CanonicalCodebook,
    mcu_idx: u32,
    comp_index: u8,
) -> Result<CanonicalEntry, DcChainError> {
    let prefix = bits.peek_u16().ok_or(DcChainError::UnexpectedEnd {
        mcu_index: mcu_idx,
        component_index: comp_index,
    })?;
    let entry = cb.lookup(prefix);
    if entry.num_bits == 0 {
        return Err(DcChainError::InvalidCode {
            mcu_index: mcu_idx,
            component_index: comp_index,
        });
    }
    bits.consume(usize::from(entry.num_bits));
    Ok(entry)
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
    const fn new(src: &'a [u8]) -> Self {
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

    /// Position of the next byte the reader will emit, measured in input
    /// bytes.  Used to detect when the reader has crossed an RST boundary.
    ///
    /// `byte_pos` is "next byte to load into the buffer".  Bytes still in
    /// `buf` are partly emitted: each fully-buffered byte (8 bits) hasn't
    /// been emitted yet, while a fractional byte (`cap % 8 != 0`) means the
    /// reader is mid-byte.  We use a ceiling division (`(cap + 7) / 8`) so
    /// a mid-byte read still reports the byte index the reader is **inside**
    /// rather than the next one.  This is the correct comparand for
    /// `RstPosition::byte_offset_in_dst`, which by construction falls on a
    /// byte boundary in the unstuffed stream.
    const fn byte_position(&self) -> usize {
        let bytes_unconsumed = (self.cap as usize).div_ceil(8);
        self.byte_pos.saturating_sub(bytes_unconsumed)
    }

    /// Discard the buffer and seek to `byte_offset` on a byte boundary.
    /// Used when crossing an RST marker — the entropy stream is byte-aligned
    /// at every RST per JPEG § F.1.1.5, so we drop any in-flight bits.
    fn realign_to_byte_at(&mut self, byte_offset: usize) {
        self.buf = 0;
        self.cap = 0;
        self.byte_pos = byte_offset.min(self.src.len());
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
    fn extend_max_baseline_dc_category() {
        // Baseline DC category ≤ 11.  Confirm endpoints don't overflow.
        // For nbits=11: range = -2047..=-1024 ∪ 1024..=2047.
        assert_eq!(extend(0, 11), -2047);
        assert_eq!(extend(1023, 11), -1024);
        assert_eq!(extend(1024, 11), 1024);
        assert_eq!(extend(2047, 11), 2047);
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

    #[test]
    fn bitreader_realign_clears_buffer_and_seeks() {
        let src = [0xAA, 0xBB, 0xCC, 0xDD];
        let mut br = BitReader::new(&src);
        // Consume part of one byte.
        let _ = br.read_bits(4);
        br.realign_to_byte_at(2);
        // Next read should now come from src[2] = 0xCC.
        assert_eq!(br.read_bits(8), Some(0xCC));
    }

    #[test]
    fn bitreader_realign_clamps_past_end() {
        let src = [0xAA];
        let mut br = BitReader::new(&src);
        br.realign_to_byte_at(99);
        assert_eq!(br.peek_u16(), None);
    }

    #[test]
    fn bitreader_byte_position_tracks_mid_byte() {
        // Pin down the byte_position contract: it must report the byte the
        // reader is currently *inside*, not the next one.  This is the
        // load-bearing invariant for RST-boundary detection.
        let src = [0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF, 0x00, 0x11, 0x22];
        let mut br = BitReader::new(&src);
        // Force a refill to a known state.
        let _ = br.peek_u16();
        assert_eq!(br.byte_position(), 0);
        // Consume one full byte: position advances to byte 1.
        let _ = br.read_bits(8);
        assert_eq!(br.byte_position(), 1);
        // Mid-byte: still inside byte 1.
        let _ = br.read_bits(4);
        assert_eq!(br.byte_position(), 1);
        // Cross into byte 2.
        let _ = br.read_bits(4);
        assert_eq!(br.byte_position(), 2);
    }

    #[test]
    fn blocks_per_mcu_non_interleaved_is_one() {
        // Single-component scan: always 1 block per MCU regardless of
        // sampling factors (JPEG § A.2.3).
        assert_eq!(blocks_per_mcu(1, 2, 2), 1);
        assert_eq!(blocks_per_mcu(1, 4, 4), 1);
    }

    #[test]
    fn blocks_per_mcu_interleaved_is_h_times_v() {
        assert_eq!(blocks_per_mcu(3, 1, 1), 1);
        assert_eq!(blocks_per_mcu(3, 2, 2), 4);
        assert_eq!(blocks_per_mcu(3, 2, 1), 2);
    }
}
