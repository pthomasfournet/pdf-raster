//! Plain-Rust scalar Huffman decoder.
//!
//! Used as the correctness oracle for the GPU backends; no JPEG
//! framing, just raw symbol-stream decoding against a single canonical
//! codebook. The Phase-A cross-backend bit-identity tests will compare
//! `CudaBackend` + `VulkanBackend` output against this.
//!
//! Built on top of [`CanonicalCodebook`]'s 16-bit-prefix LUT, which is
//! the existing in-tree CPU-fast-path table. Same source of truth as
//! the synthetic encoder in `tests::synthetic`.

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
pub fn decode_scalar(book: &CanonicalCodebook, stream: &PackedBitstream) -> Vec<u8> {
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
}
