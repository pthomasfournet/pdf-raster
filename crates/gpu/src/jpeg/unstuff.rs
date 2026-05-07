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

/// Strip JPEG byte-stuffing (`0xFF 0x00 → 0xFF`) from `src` into `dst`.
///
/// `dst` is cleared and grown to fit the unstuffed output.  RST markers
/// (`0xFF 0xD0`..=`0xFF 0xD7`) are silently consumed: they delimit
/// independently-decodable scan segments but do not contribute bits to
/// the entropy-coded data.  Fill bytes (`0xFF 0xFF`) are also tolerated.
///
/// # Errors
///
/// Returns [`UnstuffError`] on a `0xFF` followed by an unexpected byte or
/// truncated trailing `0xFF`.  The output is left in an indeterminate state
/// when an error is returned (caller should discard it).
pub fn unstuff_into(src: &[u8], dst: &mut Vec<u8>) -> Result<(), UnstuffError> {
    dst.clear();
    dst.reserve(src.len());

    let mut i = 0;
    while i < src.len() {
        let b = src[i];
        if b != 0xFF {
            dst.push(b);
            i += 1;
            continue;
        }
        // 0xFF prefix: peek at next byte to decide.
        let next_idx = i + 1;
        if next_idx >= src.len() {
            return Err(UnstuffError::TrailingFf);
        }
        let next = src[next_idx];
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
                // RST marker: scan-segment boundary. Skip both bytes; the
                // entropy bitstream within an RST segment is independent.
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

    #[test]
    fn unstuff_no_ff_passes_through() {
        let src = [0x12, 0x34, 0x56, 0x78];
        let mut dst = Vec::new();
        unstuff_into(&src, &mut dst).unwrap();
        assert_eq!(dst, src);
    }

    #[test]
    fn unstuff_ff_00_becomes_ff() {
        let src = [0xFF, 0x00, 0x12, 0xFF, 0x00];
        let mut dst = Vec::new();
        unstuff_into(&src, &mut dst).unwrap();
        assert_eq!(dst, [0xFF, 0x12, 0xFF]);
    }

    #[test]
    fn unstuff_skips_rst_marker() {
        let src = [0x12, 0xFF, 0xD0, 0x34];
        let mut dst = Vec::new();
        unstuff_into(&src, &mut dst).unwrap();
        // RST is silently skipped; bits before and after are concatenated
        // (caller is responsible for handling the DC predictor reset that
        // semantically goes with an RST marker).
        assert_eq!(dst, [0x12, 0x34]);
    }

    #[test]
    fn unstuff_skips_all_rst_markers() {
        for rst in 0xD0..=0xD7u8 {
            let src = [0xAA, 0xFF, rst, 0xBB];
            let mut dst = Vec::new();
            unstuff_into(&src, &mut dst).unwrap();
            assert_eq!(dst, [0xAA, 0xBB], "RST 0x{rst:02X} not handled");
        }
    }

    #[test]
    fn unstuff_fill_bytes_consumed() {
        let src = [0x11, 0xFF, 0xFF, 0x00, 0x22];
        let mut dst = Vec::new();
        unstuff_into(&src, &mut dst).unwrap();
        // 0xFF 0xFF → fill (skip leading); next 0xFF 0x00 → literal 0xFF.
        assert_eq!(dst, [0x11, 0xFF, 0x22]);
    }

    #[test]
    fn unstuff_unexpected_marker_returns_error() {
        // 0xFF followed by 0xD9 (EOI) is unexpected mid-scan.
        let src = [0x11, 0xFF, 0xD9, 0x22];
        let mut dst = Vec::new();
        let err = unstuff_into(&src, &mut dst).unwrap_err();
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
        let src = [0x11, 0x22, 0xFF];
        let mut dst = Vec::new();
        let err = unstuff_into(&src, &mut dst).unwrap_err();
        assert!(matches!(err, UnstuffError::TrailingFf));
    }

    #[test]
    fn unstuff_clears_dst_each_call() {
        let mut dst = vec![0xAA, 0xBB, 0xCC];
        unstuff_into(&[0x12, 0x34], &mut dst).unwrap();
        assert_eq!(dst, [0x12, 0x34]);
    }
}
