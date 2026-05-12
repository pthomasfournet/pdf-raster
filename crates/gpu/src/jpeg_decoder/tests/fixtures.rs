//! Shared test fixtures for the `jpeg_decoder` modules.
//!
//! `book4` is a 4-symbol Huffman book (codes 00, 01, 100, 101 at
//! lengths 2, 2, 3, 3) used by every parity test. `book4_stream`
//! encodes a symbol stream against it. Both lived in four sibling
//! modules until the b9cbb10 hardening pass folded them here.

use crate::jpeg::CanonicalCodebook;
use crate::jpeg::headers::{DhtClass, JpegHuffmanTable};
use crate::jpeg_decoder::tests::synthetic::encode_symbols;
use crate::jpeg_decoder::{PackedBitstream, pack_be_words};

/// 4-symbol Huffman book: codes 00, 01, 100, 101 (lengths 2, 2, 3, 3).
///
/// Symbols are 0x00 .. 0x03, so all have `value_bits` = 0 (low nibble
/// of the symbol is the size). The per-symbol bit advance equals
/// `code_bits` — easy to reason about in tests.
pub fn book4_table() -> JpegHuffmanTable {
    JpegHuffmanTable {
        class: DhtClass::Dc,
        table_id: 0,
        num_codes: [0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        values: vec![0x00, 0x01, 0x02, 0x03],
    }
}

/// Canonical lookup built from [`book4_table`].
pub fn book4_codebook() -> CanonicalCodebook {
    CanonicalCodebook::build(&book4_table()).expect("book4 is a valid canonical prefix code")
}

/// Encode `symbols` against `book4_table` and pack into a stream.
pub fn book4_stream(symbols: &[u8]) -> PackedBitstream {
    let enc = encode_symbols(&book4_table(), symbols);
    pack_be_words(bytemuck::cast_slice(&enc.words_be), enc.length_bits)
}
