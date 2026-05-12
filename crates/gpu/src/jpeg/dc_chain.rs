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

use super::bitreader::BitReader;
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
    /// Image dimensions × sampling factors yield more MCUs than the
    /// crate's internal `MAX_SAFE_MCUS` cap.  Refuses the image rather
    /// than allocating an adversarial output buffer.
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
    /// Scan-component index (0..=3).  Stored as `usize` to match every
    /// hot-path use; the few error sites that need a `u8` cast at the
    /// boundary (the bound is enforced at construction).
    sc_idx: usize,
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
        steps[sc_idx] = Some(ScanStep {
            sc_idx,
            blocks_per_mcu: bpm,
            dc_cb,
            ac_cb,
        });
        total_blocks_per_mcu = total_blocks_per_mcu.saturating_add(bpm as u64);
    }

    // Compare in u64 throughout: on a 32-bit host casting `u64 as usize`
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
    // RST boundary.  RST i (the i-th `RstPosition`) lives at MCU index
    // `(i + 1) * restart_interval`, per JPEG ISO/IEC 10918-1 § F.1.1.5.
    let mut dc_predictor = [0i32; 4];
    let mut bits = BitReader::new(unstuffed);
    let mut rst_iter = rst_positions.iter();
    let restart_interval = u32::from(headers.restart_interval);

    let n_mcus_u32 = u32::try_from(n_mcus).expect("MAX_SAFE_MCUS guarantees fit in u32");
    for mcu_idx in 0..n_mcus_u32 {
        // Reset on MCU index, not on the bit reader's byte position.  A
        // truncated MCU could leave the cursor short of the marker and a
        // byte-position chase would silently skip the predictor reset; the
        // index-driven check is robust to that case.
        // If we run out of RST entries before MCUs end, decoding will
        // fail naturally on the next codeword; the resulting
        // `UnexpectedEnd` carries `mcu_idx`.
        if restart_interval > 0
            && mcu_idx > 0
            && mcu_idx.is_multiple_of(restart_interval)
            && let Some(rst) = rst_iter.next()
        {
            bits.realign_to_byte_at(rst.byte_offset_in_dst);
            dc_predictor = [0i32; 4];
        }

        for step in steps.iter().take(scan_components) {
            let step = step.expect("steps[..scan_components] populated above");
            // sc_idx is bounded by scan_components ≤ 4 (validated at the
            // top of `resolve_dc_chain`), so the cast is provably lossless.
            #[expect(
                clippy::cast_possible_truncation,
                reason = "sc_idx ≤ 3 by construction; scan_components is validated ≤ 4"
            )]
            let sc_idx_u8 = step.sc_idx as u8;
            for _block in 0..step.blocks_per_mcu {
                let delta = decode_dc_delta(&mut bits, step.dc_cb, mcu_idx, sc_idx_u8)?;
                dc_predictor[step.sc_idx] = dc_predictor[step.sc_idx].wrapping_add(delta);
                per_component[step.sc_idx].push(dc_predictor[step.sc_idx]);
                skip_ac_coefficients(&mut bits, step.ac_cb, mcu_idx, sc_idx_u8)?;
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
        0u32
    } else {
        bits.read_bits(usize::from(category))
            .ok_or(DcChainError::UnexpectedEnd {
                mcu_index: mcu_idx,
                component_index: comp_index,
            })?
    };
    // `raw` is bounded above by `(1 << category) - 1` with `category ≤ 11`,
    // so it fits in `i32` losslessly. The cast is the standard pattern
    // for "narrow unsigned magnitude" → "signed value about to be
    // sign-extended via EXTEND".
    #[expect(
        clippy::cast_possible_wrap,
        reason = "raw < 1 << 11 = 2048; well within i32"
    )]
    Ok(extend(raw as i32, category))
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
