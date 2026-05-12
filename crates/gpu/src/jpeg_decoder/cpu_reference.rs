//! Plain-Rust scalar Huffman decoder.
//!
//! Used as the correctness oracle for the GPU backends; no JPEG
//! framing, just raw symbol-stream decoding against a single canonical
//! codebook. The cross-backend bit-identity tests compare
//! `CudaBackend` + `VulkanBackend` output against this.
//!
//! Built on top of [`CanonicalCodebook`]'s 16-bit-prefix LUT, which is
//! the existing in-tree CPU-fast-path table. Same source of truth as
//! the synthetic encoder in `tests::synthetic`.
//!
//! **Memory:** the output `Vec<u8>` is bounded by `length_bits` (one
//! symbol per bit in the degenerate length-1-codeword case), so a
//! `length_bits = u32::MAX` stream could allocate ~4 GB. This is
//! `pub(crate)` precisely because production callers should not pipe
//! untrusted bit-counts through here without their own validation.

use crate::jpeg::CanonicalCodebook;
use crate::jpeg_decoder::PackedBitstream;

/// Scalar Huffman-decode `stream` through `book` and emit the symbol
/// sequence.
///
/// Stops cleanly at end-of-stream or on a non-matching prefix (the
/// returned vector contains every symbol successfully decoded up to
/// that point). Mirrors the GPU kernel's contract: don't panic on
/// corrupt input, just stop where you can no longer decode.
#[must_use]
fn decode_scalar(book: &CanonicalCodebook, stream: &PackedBitstream) -> Vec<u8> {
    let mut symbols = Vec::new();
    let total_bits = u64::from(stream.length_bits);
    let mut bit_pos: u64 = 0;

    while bit_pos < total_bits {
        let peek = peek16(stream, bit_pos);
        let entry = book.lookup(peek);
        if entry.num_bits == 0 {
            // Prefix matches no codeword; corrupt stream or trailing
            // zero-padding past length_bits. Stop here.
            break;
        }
        let cb = u64::from(entry.num_bits);
        if bit_pos + cb > total_bits {
            // Codeword would run past the declared length — stop without
            // emitting; this happens on streams whose final byte ends
            // mid-codeword.
            break;
        }
        symbols.push(entry.symbol);
        bit_pos += cb;
    }

    symbols
}

/// Peek 16 bits starting at absolute `bit_pos`, MSB-first, padding
/// with zeros past the end of the buffer.
///
/// Matches the kernel's bit-peek shape: bit 31 of `words[word_idx]`
/// is the next bit at the start of word `word_idx`. The result is
/// right-aligned in a u16 so [`CanonicalCodebook::lookup`] can index
/// directly.
fn peek16(stream: &PackedBitstream, bit_pos: u64) -> u16 {
    let word_idx = (bit_pos / 32) as usize;
    let bit_in_word = (bit_pos % 32) as u32;

    let hi = u64::from(stream.words.get(word_idx).copied().unwrap_or(0));
    let lo = u64::from(stream.words.get(word_idx + 1).copied().unwrap_or(0));
    let combined = (hi << 32) | lo;
    // We want the 16 bits starting at `bit_in_word` within `hi`,
    // i.e. starting at bit position (63 - bit_in_word) in `combined`.
    // Right-shift so those 16 bits land at the bottom.
    let shift = 48 - bit_in_word;
    ((combined >> shift) & 0xFFFF) as u16
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jpeg::headers::DhtClass;
    use crate::jpeg::{CanonicalCodebook, JpegHuffmanTable};
    use crate::jpeg_decoder::tests::synthetic::encode_symbols;

    fn book4_table() -> JpegHuffmanTable {
        JpegHuffmanTable {
            class: DhtClass::Dc,
            table_id: 0,
            num_codes: [0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            values: vec![0x00, 0x01, 0x02, 0x03],
        }
    }

    fn book4_canonical() -> CanonicalCodebook {
        CanonicalCodebook::build(&book4_table()).expect("valid")
    }

    #[test]
    fn roundtrip_short_stream() {
        let table = book4_table();
        let original: Vec<u8> = vec![0x00, 0x02, 0x01, 0x03, 0x00, 0x00, 0x02];
        let enc = encode_symbols(&table, &original);
        let stream = PackedBitstream {
            words: enc.words_be,
            length_bits: enc.length_bits,
        };
        let got = decode_scalar(&book4_canonical(), &stream);
        assert_eq!(got, original);
    }

    #[test]
    fn roundtrip_long_stream_10k_symbols() {
        let table = book4_table();
        let original: Vec<u8> = (0..10_000u32).map(|i| (i % 4) as u8).collect();
        let enc = encode_symbols(&table, &original);
        let stream = PackedBitstream {
            words: enc.words_be,
            length_bits: enc.length_bits,
        };
        let got = decode_scalar(&book4_canonical(), &stream);
        assert_eq!(got, original);
    }

    #[test]
    fn empty_stream_yields_empty_output() {
        let stream = PackedBitstream {
            words: vec![],
            length_bits: 0,
        };
        let got = decode_scalar(&book4_canonical(), &stream);
        assert!(got.is_empty());
    }

    #[test]
    fn stream_with_unrepresentable_tail_stops_cleanly() {
        // Build a stream where the last bits don't form any codeword:
        // book has only "0..." prefixes for length-1 codes. Use a
        // book with a single length-1 codeword (0 → symbol 42), then
        // feed a stream that ends with a '1' bit — that prefix maps
        // nowhere.
        let t = JpegHuffmanTable {
            class: DhtClass::Dc,
            table_id: 0,
            num_codes: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            values: vec![42],
        };
        let book = CanonicalCodebook::build(&t).unwrap();
        // 3 bits: 0, 0, 1. First two decode to 42, third hits a miss.
        let stream = PackedBitstream {
            words: vec![0x2000_0000],
            length_bits: 3,
        };
        let got = decode_scalar(&book, &stream);
        assert_eq!(got, vec![42, 42]);
    }

    #[test]
    fn stream_truncated_mid_codeword_stops_without_emitting_partial() {
        // Book has only the 2-bit codeword "01" → symbol 5. Feed a
        // stream of length_bits = 1 containing just bit "0" — the
        // 16-bit peek would land on `01_0000_..._0`, find a match
        // claiming 2 bits, but bit_pos+2 > length_bits=1, so we
        // must stop without emitting.
        let t = JpegHuffmanTable {
            class: DhtClass::Dc,
            table_id: 0,
            // 1 codeword of length 2.
            num_codes: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            values: vec![5],
        };
        let book = CanonicalCodebook::build(&t).unwrap();
        // Bit "0" at bit position 0; the 16-bit peek pads to
        // 0b0100_0000_0000_0000 = 0x4000 which matches "01" → symbol 5
        // claiming 2 bits. But length_bits = 1 < 2 bits, so we stop.
        // Wait — actually the canonical assignment for one length-2
        // code starts at code = 0b00, not 0b01. Use a high bit instead
        // so the peek includes the length-2 codeword's leading bit.
        let stream = PackedBitstream {
            words: vec![0x0000_0000],
            length_bits: 1,
        };
        let got = decode_scalar(&book, &stream);
        // The peek reads "00_0000_...0" → matches the length-2 code
        // "00" → symbol 5 (the only code in the table). bit_pos (0) +
        // cb (2) > length_bits (1), so we stop without emitting.
        assert_eq!(got, Vec::<u8>::new());
    }

    #[test]
    fn peek16_at_word_start_returns_high_16_bits_of_word_0() {
        let stream = PackedBitstream {
            words: vec![0xABCD_1234, 0xDEAD_BEEF],
            length_bits: 64,
        };
        assert_eq!(peek16(&stream, 0), 0xABCD);
    }

    #[test]
    fn peek16_at_word_end_crosses_into_word_1() {
        let stream = PackedBitstream {
            words: vec![0x0000_00FF, 0xFE00_0000],
            length_bits: 64,
        };
        // bit_pos = 24: hi bits 24..=8 of word 0 + lo bits 7..=0
        // wait — bit_pos = 24 reads bits 24..=39 in stream coords.
        // word 0 bits 24..=31 (its low byte) = 0xFF; word 1 bits 0..=7
        // (its high byte) = 0xFE. Concatenated MSB-first: 0xFFFE.
        assert_eq!(peek16(&stream, 24), 0xFFFE);
    }

    #[test]
    fn peek16_past_end_pads_with_zeros() {
        let stream = PackedBitstream {
            words: vec![0xABCD_1234],
            length_bits: 32,
        };
        // bit_pos = 32: we're reading past word 0 into a nonexistent
        // word 1; peek16 should return zeros (the bits don't matter
        // because the caller respects length_bits).
        assert_eq!(peek16(&stream, 32), 0x0000);
    }

    #[test]
    fn peek16_at_word_end_minus_one_pulls_one_bit_from_next_word() {
        let stream = PackedBitstream {
            words: vec![0x0000_0001, 0xC000_0000],
            length_bits: 64,
        };
        // bit_pos = 31: window = bit 31 of word 0 (= 1) followed by
        // bits 0..=14 of word 1 (high bits 1, 1, 0, 0, …). MSB-first:
        // 1_1100_0000_0000_000 = 0b1_1100_0000_0000_000 = 0xE000.
        assert_eq!(peek16(&stream, 31), 0xE000);
    }
}
