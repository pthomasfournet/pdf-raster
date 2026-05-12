//! 32-bit big-endian word packing for the GPU bitstream input.
//!
//! The GPU kernels read the entropy-coded segment as a buffer of u32
//! words in big-endian bit order: bit 31 of word 0 is the first bit
//! of the stream. `length_bits` carries the exact bit count so the
//! tail word's padding zeros are never decoded.

/// Packed bitstream ready for upload to the GPU.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackedBitstream {
    /// Big-endian 32-bit words. The first bit of the stream is bit 31
    /// of `words[0]`.
    pub words: Vec<u32>,
    /// Exact bit count of the stream. Always `<= words.len() * 32`;
    /// the remainder of the final word is zero-padded.
    pub length_bits: u32,
}

/// Pack `bits` into big-endian 32-bit words (MSB-first within bytes).
///
/// `length_bits` is the meaningful prefix length; bytes beyond
/// `ceil(length_bits / 8)` are ignored, and the final word's trailing
/// bits are zero.
#[must_use]
pub fn pack_be_words(bits: &[u8], length_bits: u32) -> PackedBitstream {
    let word_count = length_bits.div_ceil(32) as usize;
    let mut words = Vec::with_capacity(word_count);
    for chunk_idx in 0..word_count {
        let byte_idx = chunk_idx * 4;
        let mut word = 0u32;
        for b in 0..4 {
            let i = byte_idx + b;
            if i < bits.len() {
                word |= u32::from(bits[i]) << (24 - b * 8);
            }
        }
        words.push(word);
    }
    PackedBitstream { words, length_bits }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_empty_is_empty() {
        let out = pack_be_words(&[], 0);
        assert_eq!(out.words, Vec::<u32>::new());
        assert_eq!(out.length_bits, 0);
    }

    #[test]
    fn pack_single_byte_pads_to_word() {
        let out = pack_be_words(&[0b1011_0100], 8);
        assert_eq!(out.words, vec![0xB400_0000]);
        assert_eq!(out.length_bits, 8);
    }

    #[test]
    fn pack_four_bytes_one_word() {
        let out = pack_be_words(&[0xDE, 0xAD, 0xBE, 0xEF], 32);
        assert_eq!(out.words, vec![0xDEAD_BEEF]);
        assert_eq!(out.length_bits, 32);
    }

    #[test]
    fn pack_five_bytes_two_words_tail_zero_padded() {
        let out = pack_be_words(&[0xDE, 0xAD, 0xBE, 0xEF, 0x42], 40);
        assert_eq!(out.words, vec![0xDEAD_BEEF, 0x4200_0000]);
        assert_eq!(out.length_bits, 40);
    }

    #[test]
    fn pack_length_shorter_than_byte_count_preserves_length() {
        // 9 bits of data in 2 bytes; length_bits is the source of truth
        // for the decoder, not the byte count.
        let out = pack_be_words(&[0xFF, 0x80], 9);
        assert_eq!(out.words, vec![0xFF80_0000]);
        assert_eq!(out.length_bits, 9);
    }
}
