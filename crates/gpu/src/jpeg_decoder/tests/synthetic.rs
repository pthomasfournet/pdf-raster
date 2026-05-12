//! Test-only synthetic Huffman encoder.
//!
//! Reverse of `cpu_reference::decode_scalar`: given a JPEG-form
//! `(num_codes, values)` table and a list of symbols, produces a
//! packed bitstream that the GPU decoder must round-trip.
//!
//! Used as the correctness oracle for cross-backend bit-identity
//! tests.
//!
//! **Caller contract:** the encoder assumes `table` is a *valid*
//! canonical prefix code (every codeword length yields at most
//! `1 << length` codes; the total matches `values.len()`). It does
//! not re-validate. Call `build_gpu_codetable` first if your test
//! constructs a table dynamically; the canonical-walk dedup in
//! `/audit/2026-05-11-canonical-loop-dedup.md` will fold the
//! validation in once the shared visitor lands.

#![cfg(test)]

use crate::jpeg::JpegHuffmanTable;

/// `(symbol → (code, code_bits))` lookup, derived once from a DHT.
///
/// Sparse (symbol space is `u8`, 256 entries) so a flat `[Option<…>;
/// 256]` is cheaper than a `HashMap` and lets the encoder branch on
/// a single `None`.
#[derive(Debug, Clone)]
pub struct SymbolEncoder {
    table: [Option<(u16, u8)>; 256],
}

impl SymbolEncoder {
    /// Materialise the canonical assignment for `table` into a
    /// symbol-keyed lookup. Mirrors the canonical loop in
    /// `CanonicalCodebook::build` and `codetable::build_gpu_codetable`.
    #[must_use]
    pub fn from_table(table: &JpegHuffmanTable) -> Self {
        let mut out = Self { table: [None; 256] };
        let mut value_idx = 0usize;
        let mut code: u32 = 0;
        for length_minus_1 in 0..16u8 {
            let count = usize::from(table.num_codes[length_minus_1 as usize]);
            let length = length_minus_1 + 1;
            for _ in 0..count {
                let symbol = table.values[value_idx];
                value_idx += 1;
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "code < 1 << length, length ≤ 16, so code fits in u16"
                )]
                let code_u16 = code as u16;
                out.table[usize::from(symbol)] = Some((code_u16, length));
                code += 1;
            }
            code <<= 1;
        }
        out
    }

    /// Look up the canonical `(code, code_bits)` for `symbol`, or
    /// `None` if `symbol` is not in the DHT.
    #[must_use]
    pub fn lookup(&self, symbol: u8) -> Option<(u16, u8)> {
        self.table[usize::from(symbol)]
    }
}

/// Encoded synthetic stream + the original symbols for round-trip
/// assertions.
#[derive(Debug, Clone)]
pub struct SyntheticEncoded {
    /// Packed big-endian 32-bit words; first bit of the stream is bit
    /// 31 of `words_be[0]`.
    pub words_be: Vec<u32>,
    /// Exact bit count; the tail word's trailing bits are zero.
    pub length_bits: u32,
    /// Symbols fed in, for the round-trip assertion in the caller.
    pub symbols_in: Vec<u8>,
}

/// Pack `symbols` through the canonical encoding of `table`.
///
/// Panics if any symbol is not in the DHT — tests are expected to
/// build the symbol stream from the table's value list.
#[must_use]
pub fn encode_symbols(table: &JpegHuffmanTable, symbols: &[u8]) -> SyntheticEncoded {
    let enc = SymbolEncoder::from_table(table);
    let mut acc: u64 = 0;
    let mut acc_bits: u32 = 0;
    let mut out_words = Vec::new();
    let mut length_bits = 0u32;

    for &s in symbols {
        let (code, code_bits) = enc
            .lookup(s)
            .unwrap_or_else(|| panic!("symbol {s:#04x} not in DHT"));
        let cb = u32::from(code_bits);
        acc = (acc << cb) | u64::from(code);
        acc_bits += cb;
        length_bits += cb;
        while acc_bits >= 32 {
            let shift = acc_bits - 32;
            let word = ((acc >> shift) & 0xFFFF_FFFF) as u32;
            out_words.push(word);
            // shift ≤ 15 (acc_bits never exceeds 47 here), so the mask
            // `(1 << shift) - 1` is well-defined and never overflows.
            acc &= (1u64 << shift) - 1;
            acc_bits = shift;
        }
    }
    if acc_bits > 0 {
        let word = ((acc << (32 - acc_bits)) & 0xFFFF_FFFF) as u32;
        out_words.push(word);
    }
    SyntheticEncoded {
        words_be: out_words,
        length_bits,
        symbols_in: symbols.to_vec(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jpeg::headers::DhtClass;

    fn book4() -> JpegHuffmanTable {
        // 4-symbol book: codes 00, 01, 100, 101 (lengths 2, 2, 3, 3).
        JpegHuffmanTable {
            class: DhtClass::Dc,
            table_id: 0,
            num_codes: [0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            values: vec![0x00, 0x01, 0x02, 0x03],
        }
    }

    #[test]
    fn symbol_encoder_materialises_canonical_codes() {
        let enc = SymbolEncoder::from_table(&book4());
        assert_eq!(enc.lookup(0x00), Some((0b00, 2)));
        assert_eq!(enc.lookup(0x01), Some((0b01, 2)));
        assert_eq!(enc.lookup(0x02), Some((0b100, 3)));
        assert_eq!(enc.lookup(0x03), Some((0b101, 3)));
        assert_eq!(enc.lookup(0x04), None);
    }

    #[test]
    fn encoded_stream_retains_input_symbols_for_oracle_use() {
        // symbols_in is consumed by A5's round-trip test; assert here
        // that encode_symbols populates it verbatim.
        let symbols = vec![0x00, 0x02, 0x01, 0x03];
        let out = encode_symbols(&book4(), &symbols);
        assert_eq!(out.symbols_in, symbols);
    }

    #[test]
    fn encode_two_symbol_stream_packs_top_bits() {
        // Encode 0x00, 0x02 → 00 100 → high-aligned in word 0.
        let out = encode_symbols(&book4(), &[0x00, 0x02]);
        assert_eq!(out.length_bits, 5);
        assert_eq!(out.words_be.len(), 1);
        // 5 bits "00100" in the upper bits of a u32 = 0010_0_000...0.
        assert_eq!(out.words_be[0], 0x2000_0000);
    }

    #[test]
    fn encode_exactly_32_bits_fills_one_word() {
        // 16 length-2 codewords alternating symbol 0 (code 00) and
        // symbol 1 (code 01) → bit pattern "00 01 00 01 …".
        let symbols: Vec<u8> = (0u8..16).map(|i| i & 1).collect();
        let out = encode_symbols(&book4(), &symbols);
        assert_eq!(out.length_bits, 32);
        assert_eq!(out.words_be.len(), 1);
        // 8 repeats of "0001" = 0x11111111.
        assert_eq!(out.words_be[0], 0x1111_1111);
    }

    #[test]
    fn encode_just_over_32_bits_spills_to_second_word() {
        // 17 length-2 codewords = 34 bits. First 32 bits fill word 0,
        // last 2 bits sit in the top of word 1.
        let symbols: Vec<u8> = std::iter::repeat_n(0u8, 17).collect();
        let out = encode_symbols(&book4(), &symbols);
        assert_eq!(out.length_bits, 34);
        assert_eq!(out.words_be.len(), 2);
        assert_eq!(out.words_be[0], 0x0000_0000);
        assert_eq!(out.words_be[1], 0x0000_0000);
    }
}
