//! JPEG byte-unstuffing.
//!
//! The JPEG entropy-coded segment uses `0xFF` as a marker prefix.  Any literal
//! `0xFF` byte produced by the Huffman encoder must therefore be followed by
//! `0x00` so the decoder can distinguish "literal 0xFF in compressed data"
//! from "marker prefix".  Before bit-walking the stream, the decoder strips
//! these inserted zero bytes.
//!
//! This module provides the strip operation as a standalone function so the
//! parallel-Huffman pipeline (which needs a flat bitstream upload to the GPU)
//! and any future host-side Huffman decoder can share it.
//!
//! The output is byte-aligned: stuffing only ever inserts a zero between two
//! bytes, never a single bit, so the bit positions of every Huffman codeword
//! shift by integer multiples of 8 after unstuffing.  Subsequent bit-walkers
//! can treat the output as a contiguous bitstream.
//!
//! ## Restart markers
//!
//! Restart markers (`0xFF 0xD0`–`0xFF 0xD7`) are stripped from the bitstream
//! but **must not be silently discarded**: they reset the DC differential
//! predictor for every component (JPEG ISO/IEC 10918-1 § F.1.1.5).  Callers
//! that maintain a DC chain (e.g. [`super::dc_chain`]) need to know the byte
//! offset within the unstuffed output where each RST landed so they can reset
//! their predictor state at the matching boundary.
//!
//! [`unstuff_into`] therefore records one [`RstPosition`] per RST marker
//! encountered, each pointing to the byte offset in `dst` *immediately after*
//! the stripped marker (i.e. the first byte of the next entropy segment) and
//! carrying the marker's index modulo 8.

/// One restart marker stripped from the entropy stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RstPosition {
    /// Byte offset in the unstuffed output where the next entropy segment begins
    /// (i.e. immediately after the stripped `0xFF 0xDn` pair).
    pub byte_offset_in_dst: usize,
    /// Marker index 0..=7 (`0xD0` → 0, …, `0xD7` → 7).  Encoders cycle through
    /// 0..7 so out-of-order markers can be detected.
    pub marker_index: u8,
}

/// Errors returned by [`unstuff_into`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnstuffError {
    /// Encountered a `0xFF` followed by something other than `0x00` or a
    /// fill-byte sequence — i.e. an unexpected marker mid-scan. Either the
    /// input was not the entropy-coded segment, or the JPEG is malformed.
    UnexpectedMarker {
        /// The byte after the `0xFF` prefix.
        following: u8,
        /// Byte offset in the input where the `0xFF` was seen.
        offset: usize,
    },
    /// Input ended with a dangling `0xFF` and no follow-up byte.
    TrailingFf,
}

impl std::fmt::Display for UnstuffError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnexpectedMarker { following, offset } => write!(
                f,
                "unexpected JPEG marker 0xFF 0x{following:02X} at offset {offset} during unstuffing"
            ),
            Self::TrailingFf => write!(f, "JPEG entropy segment ends with dangling 0xFF"),
        }
    }
}

impl std::error::Error for UnstuffError {}

/// Strip JPEG byte-stuffing (`0xFF 0x00 → 0xFF`) from `src` into `dst`,
/// recording every RST marker encountered into `rst_positions`.
///
/// `dst` and `rst_positions` are both cleared on entry.  Fill bytes
/// (`0xFF 0xFF`) are silently consumed; their position is **not** recorded
/// because they have no semantic meaning beyond padding.
///
/// # Errors
///
/// Returns [`UnstuffError`] on a `0xFF` followed by an unexpected byte or
/// truncated trailing `0xFF`.  The output is left in an indeterminate state
/// when an error is returned (caller should discard both buffers).
pub fn unstuff_into(
    src: &[u8],
    dst: &mut Vec<u8>,
    rst_positions: &mut Vec<RstPosition>,
) -> Result<(), UnstuffError> {
    dst.clear();
    dst.reserve(src.len());
    rst_positions.clear();

    let mut i = 0;
    while i < src.len() {
        let b = src[i];
        if b != 0xFF {
            dst.push(b);
            i += 1;
            continue;
        }
        // 0xFF prefix: peek at next byte to decide.
        let Some(&next) = src.get(i + 1) else {
            return Err(UnstuffError::TrailingFf);
        };
        match next {
            0x00 => {
                // Stuffed byte: emit literal 0xFF, skip the 0x00.
                dst.push(0xFF);
                i += 2;
            }
            0xFF => {
                // Fill byte: skip the leading 0xFF, leave the next 0xFF for
                // the next iteration to re-evaluate (it may be another fill,
                // a stuffed byte, or a marker).
                i += 1;
            }
            0xD0..=0xD7 => {
                // RST marker: scan-segment boundary.  Record the position so
                // the caller can reset the DC predictor at the next entropy
                // byte.  Marker index 0..=7 = next - 0xD0.
                rst_positions.push(RstPosition {
                    byte_offset_in_dst: dst.len(),
                    marker_index: next - 0xD0,
                });
                i += 2;
            }
            other => {
                return Err(UnstuffError::UnexpectedMarker {
                    following: other,
                    offset: i,
                });
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Convenience: drive [`unstuff_into`] against fresh buffers and return
    /// the unstuffed bytes plus any RST positions encountered.
    fn run(src: &[u8]) -> Result<(Vec<u8>, Vec<RstPosition>), UnstuffError> {
        let mut dst = Vec::new();
        let mut rsts = Vec::new();
        unstuff_into(src, &mut dst, &mut rsts)?;
        Ok((dst, rsts))
    }

    #[test]
    fn unstuff_no_ff_passes_through() {
        let (dst, rsts) = run(&[0x12, 0x34, 0x56, 0x78]).unwrap();
        assert_eq!(dst, [0x12, 0x34, 0x56, 0x78]);
        assert!(rsts.is_empty());
    }

    #[test]
    fn unstuff_ff_00_becomes_ff() {
        let (dst, rsts) = run(&[0xFF, 0x00, 0x12, 0xFF, 0x00]).unwrap();
        assert_eq!(dst, [0xFF, 0x12, 0xFF]);
        assert!(rsts.is_empty());
    }

    #[test]
    fn unstuff_records_rst_marker_position() {
        let (dst, rsts) = run(&[0x12, 0xFF, 0xD0, 0x34]).unwrap();
        // The marker bytes are stripped; bits before and after are concatenated.
        assert_eq!(dst, [0x12, 0x34]);
        // RST landed after one emitted byte, marker index 0.
        assert_eq!(
            rsts,
            vec![RstPosition {
                byte_offset_in_dst: 1,
                marker_index: 0,
            }],
        );
    }

    #[test]
    fn unstuff_records_all_rst_marker_indices() {
        for rst in 0xD0..=0xD7u8 {
            let (dst, rsts) = run(&[0xAA, 0xFF, rst, 0xBB]).unwrap();
            assert_eq!(dst, [0xAA, 0xBB], "RST 0x{rst:02X} not handled");
            assert_eq!(rsts.len(), 1);
            assert_eq!(rsts[0].marker_index, rst - 0xD0);
            assert_eq!(rsts[0].byte_offset_in_dst, 1);
        }
    }

    #[test]
    fn unstuff_records_multiple_rst_markers_in_order() {
        // Two RSTs separated by entropy data; index should cycle.
        let src = [0x11, 0xFF, 0xD0, 0x22, 0x33, 0xFF, 0xD1, 0x44];
        let (dst, rsts) = run(&src).unwrap();
        assert_eq!(dst, [0x11, 0x22, 0x33, 0x44]);
        assert_eq!(rsts.len(), 2);
        assert_eq!(rsts[0].byte_offset_in_dst, 1);
        assert_eq!(rsts[0].marker_index, 0);
        assert_eq!(rsts[1].byte_offset_in_dst, 3);
        assert_eq!(rsts[1].marker_index, 1);
    }

    #[test]
    fn unstuff_fill_bytes_consumed() {
        let (dst, rsts) = run(&[0x11, 0xFF, 0xFF, 0x00, 0x22]).unwrap();
        // 0xFF 0xFF → fill (skip leading); next 0xFF 0x00 → literal 0xFF.
        assert_eq!(dst, [0x11, 0xFF, 0x22]);
        assert!(rsts.is_empty());
    }

    #[test]
    fn unstuff_unexpected_marker_returns_error() {
        // 0xFF followed by 0xD9 (EOI) is unexpected mid-scan.
        let mut dst = Vec::new();
        let mut rsts = Vec::new();
        let err = unstuff_into(&[0x11, 0xFF, 0xD9, 0x22], &mut dst, &mut rsts).unwrap_err();
        assert!(matches!(
            err,
            UnstuffError::UnexpectedMarker {
                following: 0xD9,
                offset: 1
            }
        ));
    }

    #[test]
    fn unstuff_trailing_ff_returns_error() {
        let mut dst = Vec::new();
        let mut rsts = Vec::new();
        let err = unstuff_into(&[0x11, 0x22, 0xFF], &mut dst, &mut rsts).unwrap_err();
        assert!(matches!(err, UnstuffError::TrailingFf));
    }

    #[test]
    fn unstuff_clears_buffers_each_call() {
        let mut dst = vec![0xAA, 0xBB, 0xCC];
        let mut rsts = vec![RstPosition {
            byte_offset_in_dst: 99,
            marker_index: 7,
        }];
        unstuff_into(&[0x12, 0x34], &mut dst, &mut rsts).unwrap();
        assert_eq!(dst, [0x12, 0x34]);
        assert!(rsts.is_empty());
    }
}
