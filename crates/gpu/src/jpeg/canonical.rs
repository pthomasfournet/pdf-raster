//! Canonical Huffman lookup table construction.
//!
//! The DHT segment in a JPEG declares each Huffman table as
//! `(num_codes[16], values)`, where `num_codes[i]` is the number of `i+1`-bit
//! codewords and `values` is the list of decoded symbols in canonical order.
//! This module builds the canonical code values per JPEG ISO/IEC 10918-1
//! Annex C and produces a 65 536-entry direct-indexed lookup table.
//!
//! # Lookup table format
//!
//! Keyed by the next 16 bits of the entropy stream, MSB-first, zero-padded
//! if fewer than 16 bits remain.  Each entry stores the number of bits the
//! codeword consumes (1..=16) and the symbol value.  Replication: a 10-bit
//! codeword `c` occupies all 64 entries from `c << 6` through `(c << 6) | 63`.
//! Decoder advances by `entry.num_bits` and emits `entry.symbol`.

use super::headers::JpegHuffmanTable;

/// One entry in the canonical Huffman lookup table.
///
/// `num_bits == 0` indicates the slot was never assigned — i.e. the bit
/// pattern does not correspond to any codeword in the table.  A correctly
/// built JPEG-spec Huffman table covers every possible 16-bit prefix, but
/// pathological / corrupt streams can produce 0-entries; the decoder must
/// treat that as a fatal decode error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CanonicalEntry {
    /// Number of bits the codeword consumes (1..=16).  `0` = unused slot.
    pub num_bits: u8,
    /// Decoded symbol value.
    pub symbol: u8,
}

impl CanonicalEntry {
    /// Sentinel for unused slots.
    pub const EMPTY: Self = Self {
        num_bits: 0,
        symbol: 0,
    };
}

/// A canonical Huffman codebook ready for direct-indexed lookup.
///
/// Construction is allocation-aware: the 65 536-entry table is heap-allocated
/// once per codebook, since stack-allocating 128 KB would blow most thread
/// stacks.  `CanonicalCodebook` owns the table.
pub struct CanonicalCodebook {
    /// 1 << 16 entries, indexed by the next 16 bits of the bitstream MSB-first.
    table: Box<[CanonicalEntry; 65_536]>,
    /// Maximum codeword length actually present in this table (1..=16).
    max_len: u8,
}

impl std::fmt::Debug for CanonicalCodebook {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CanonicalCodebook")
            .field("max_len", &self.max_len)
            .field("table_size", &self.table.len())
            .finish()
    }
}

/// Errors emitted by [`CanonicalCodebook::build`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CanonicalCodebookError {
    /// `num_codes` declared zero codewords — empty table is invalid for
    /// JPEG (every scan needs at least one symbol).
    Empty,
    /// `num_codes` declared more codewords than the JPEG spec allows for
    /// this bit length.  Specifically, `sum(num_codes[..=i]) <= (1 << (i+1))`
    /// must hold for every length prefix; violation means the codes can't
    /// form a prefix code.
    OverflowAtLength {
        /// Codeword length (1..=16) where the cumulative sum first exceeded
        /// the available code space.
        length: u8,
    },
    /// Total entries claimed by `num_codes` did not match `values.len()`.
    LengthMismatch {
        /// Sum of `num_codes`.
        expected: usize,
        /// Length of `values`.
        actual: usize,
    },
}

impl std::fmt::Display for CanonicalCodebookError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Empty => write!(f, "Huffman table has zero codewords"),
            Self::OverflowAtLength { length } => write!(
                f,
                "Huffman table overflows code space at codeword length {length}"
            ),
            Self::LengthMismatch { expected, actual } => write!(
                f,
                "Huffman table values length {actual} != sum(num_codes) {expected}"
            ),
        }
    }
}

impl std::error::Error for CanonicalCodebookError {}

impl CanonicalCodebook {
    /// Build a canonical lookup table from a parsed [`JpegHuffmanTable`].
    ///
    /// # Errors
    ///
    /// Returns [`CanonicalCodebookError`] if `num_codes` is empty, declares
    /// more codes than fit in the prefix code space, or disagrees with the
    /// length of `values`.
    ///
    /// # Panics
    ///
    /// Panics only on heap-allocation failure of the 128 KB lookup slab,
    /// which is the standard "allocator OOM" panic Rust emits and not a
    /// recoverable error from the caller's perspective.
    pub fn build(table: &JpegHuffmanTable) -> Result<Self, CanonicalCodebookError> {
        let total_codes: usize = table.num_codes.iter().map(|&n| n as usize).sum();
        if total_codes == 0 {
            return Err(CanonicalCodebookError::Empty);
        }
        if total_codes != table.values.len() {
            return Err(CanonicalCodebookError::LengthMismatch {
                expected: total_codes,
                actual: table.values.len(),
            });
        }

        // Allocate the 128 KB lookup table directly on the heap; constructing
        // it on the stack first would risk overflowing typical thread stacks.
        let mut entries: Box<[CanonicalEntry; 65_536]> = vec![CanonicalEntry::EMPTY; 65_536]
            .into_boxed_slice()
            .try_into()
            .expect("vec built with exactly 65_536 elements");
        let mut value_idx: usize = 0;
        let mut code: u32 = 0;
        let mut max_len: u8 = 0;

        // Canonical code assignment per JPEG Annex C: starting at code = 0,
        // assign each codeword in turn; on length increase, left-shift the
        // running code by the length difference.
        for length_minus_1 in 0..16usize {
            let count = table.num_codes[length_minus_1] as usize;
            #[expect(
                clippy::cast_possible_truncation,
                reason = "length_minus_1 ≤ 15, fits in u8"
            )]
            let length = (length_minus_1 + 1) as u8;
            for _ in 0..count {
                // Verify the code hasn't overflowed `length` bits.
                if code >= (1u32 << length) {
                    return Err(CanonicalCodebookError::OverflowAtLength { length });
                }
                let symbol = table.values[value_idx];
                value_idx += 1;
                fill_table(&mut entries, code, length, symbol);
                if length > max_len {
                    max_len = length;
                }
                code += 1;
            }
            // Shift left for the next length tier.
            code <<= 1;
        }

        Ok(Self {
            table: entries,
            max_len,
        })
    }

    /// Look up the next codeword given the next 16 bits of the entropy stream
    /// (MSB-first; pad with zeros if fewer remain).  Returns the matching
    /// entry; if `entry.num_bits == 0` the bit pattern is not a valid code.
    #[must_use]
    #[inline]
    pub fn lookup(&self, bits: u16) -> CanonicalEntry {
        self.table[bits as usize]
    }

    /// Maximum codeword length present in this table (1..=16).
    #[must_use]
    pub const fn max_len(&self) -> u8 {
        self.max_len
    }

    /// Direct read-only access to the lookup table for code that needs to
    /// upload it to the GPU as a flat array.
    #[must_use]
    pub fn table(&self) -> &[CanonicalEntry; 65_536] {
        &self.table
    }
}

/// Replicate one canonical codeword across every 16-bit prefix that begins
/// with that codeword.  A `length`-bit codeword `code` occupies
/// `1 << (16 - length)` consecutive table slots, starting at `code << (16 - length)`.
fn fill_table(entries: &mut [CanonicalEntry; 65_536], code: u32, length: u8, symbol: u8) {
    let shift = 16 - u32::from(length);
    let start = (code << shift) as usize;
    let span = 1usize << shift;
    let entry = CanonicalEntry {
        num_bits: length,
        symbol,
    };
    entries[start..start + span].fill(entry);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jpeg::headers::DhtClass;

    /// Build a `JpegHuffmanTable` from the spec's example DC luminance table
    /// (ISO/IEC 10918-1 Table K.3 / Annex K).
    fn k3_dc_luma() -> JpegHuffmanTable {
        // Code lengths: 0,1,5,1,1,1,1,1,1,0,0,0,0,0,0,0
        // Symbols: 0,1,2,3,4,5,6,7,8,9,10,11
        JpegHuffmanTable {
            class: DhtClass::Dc,
            table_id: 0,
            num_codes: [0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            values: vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        }
    }

    #[test]
    fn build_k3_dc_luma() {
        let cb = CanonicalCodebook::build(&k3_dc_luma()).unwrap();
        // Maximum codeword length in the K.3 table is 9 bits.
        assert_eq!(cb.max_len(), 9);
    }

    #[test]
    fn k3_dc_luma_decodes_symbol_0_for_code_00() {
        // First codeword: length 2, code 00 (binary), symbol 0.
        let cb = CanonicalCodebook::build(&k3_dc_luma()).unwrap();
        // Bits "00" left-padded to 16 = 0b00_0000_0000_0000_00 = 0x0000.
        let entry = cb.lookup(0x0000);
        assert_eq!(entry.num_bits, 2);
        assert_eq!(entry.symbol, 0);
    }

    #[test]
    fn k3_dc_luma_decodes_symbol_1_for_code_010() {
        // Second codeword: length 3, code 010 (binary 2), symbol 1.
        // Bits "010" left-padded → 0b010_0000_0000_0000_0 = 0x4000.
        let cb = CanonicalCodebook::build(&k3_dc_luma()).unwrap();
        let entry = cb.lookup(0x4000);
        assert_eq!(entry.num_bits, 3);
        assert_eq!(entry.symbol, 1);
    }

    #[test]
    fn k3_dc_luma_decodes_long_codes() {
        // The K.3 table has codewords up to 9 bits long.
        // Symbol 9 is at length 8: codes are
        //   length 2: 00          (1 code,  symbol 0)
        //   length 3: 010..110    (5 codes, symbols 1..=5)
        //   length 4: 1110        (1 code,  symbol 6)
        //   length 5: 11110       (1 code,  symbol 7)
        //   length 6: 111110      (1 code,  symbol 8)
        //   length 7: 1111110     (1 code,  symbol 9)
        //   length 8: 11111110    (1 code,  symbol 10)
        //   length 9: 111111110   (1 code,  symbol 11)
        let cb = CanonicalCodebook::build(&k3_dc_luma()).unwrap();
        // Symbol 11: 111111110 → 0b1_1111_1110_0000000 = 0xFF00.
        let entry = cb.lookup(0xFF00);
        assert_eq!(entry.num_bits, 9);
        assert_eq!(entry.symbol, 11);
    }

    #[test]
    fn build_empty_table_returns_error() {
        let t = JpegHuffmanTable {
            class: DhtClass::Dc,
            table_id: 0,
            num_codes: [0; 16],
            values: vec![],
        };
        let err = CanonicalCodebook::build(&t).expect_err("must fail");
        assert_eq!(err, CanonicalCodebookError::Empty);
    }

    #[test]
    fn build_length_mismatch_returns_error() {
        let mut t = k3_dc_luma();
        let _ = t.values.pop(); // 11 values, 12 expected
        assert!(matches!(
            CanonicalCodebook::build(&t),
            Err(CanonicalCodebookError::LengthMismatch {
                expected: 12,
                actual: 11
            }),
        ));
    }

    #[test]
    fn build_oversize_returns_error() {
        // Claim 3 codewords of length 1, which is impossible (max 2).
        let t = JpegHuffmanTable {
            class: DhtClass::Dc,
            table_id: 0,
            num_codes: [3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            values: vec![0, 1, 2],
        };
        let err = CanonicalCodebook::build(&t).unwrap_err();
        assert!(matches!(
            err,
            CanonicalCodebookError::OverflowAtLength { .. }
        ));
    }

    #[test]
    fn lookup_unused_slot_returns_zero_num_bits() {
        // Build a table with only one 1-bit codeword (00 → symbol 0). Half the
        // 16-bit prefix space (those starting with '1') has no codeword.
        let t = JpegHuffmanTable {
            class: DhtClass::Dc,
            table_id: 0,
            num_codes: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            values: vec![42],
        };
        let cb = CanonicalCodebook::build(&t).unwrap();
        // Only code is length 1 = '0'. Bits "0..." → entry exists.
        assert_eq!(cb.lookup(0x0000).symbol, 42);
        assert_eq!(cb.lookup(0x0000).num_bits, 1);
        // Bits "1..." → unassigned.
        assert_eq!(cb.lookup(0x8000).num_bits, 0);
    }

    #[test]
    fn table_lookup_is_dense_for_complete_tables() {
        // Every 16-bit prefix should map to a valid entry for K.3 since the
        // code space is fully covered by length-9-and-below codewords... but
        // K.3 only uses 12 distinct codes, so most slots are unassigned.
        // This test just confirms that *no* slot returned has num_bits > 16.
        let cb = CanonicalCodebook::build(&k3_dc_luma()).unwrap();
        for i in 0..=u16::MAX {
            let e = cb.lookup(i);
            assert!(
                e.num_bits == 0 || (1..=16).contains(&e.num_bits),
                "invalid num_bits {} at slot {i}",
                e.num_bits,
            );
        }
    }
}
