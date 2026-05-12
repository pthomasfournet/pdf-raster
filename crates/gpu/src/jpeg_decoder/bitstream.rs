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
/// `length_bits` is the meaningful prefix length the GPU kernel will
/// honour; bits past `length_bits` in the final word are zero and not
/// part of the stream. Bytes in `bits` past `ceil(length_bits / 8)`
/// are ignored.
///
/// # Invariant
///
/// `length_bits <= bits.len() * 8`. Debug-asserted; a release build
/// silently zero-pads missing bytes, but the GPU kernel will read
/// those zeros and almost certainly produce garbage symbols, so the
/// debug-assert is the line of defence.
#[must_use]
pub fn pack_be_words(bits: &[u8], length_bits: u32) -> PackedBitstream {
    // Source-of-truth check: caller must provide at least
    // ceil(length_bits / 8) bytes. usize::from(length_bits) is safe
    // because length_bits is u32 and we only target 64-bit platforms
    // for the GPU path; the saturating_mul guards the assertion
    // message arithmetic regardless.
    debug_assert!(
        (length_bits as usize).div_ceil(8) <= bits.len(),
        "length_bits {} exceeds bits slice length {} bytes ({} bits available)",
        length_bits,
        bits.len(),
        bits.len().saturating_mul(8),
    );

    let word_count = length_bits.div_ceil(32) as usize;
    let words = (0..word_count)
        .map(|chunk_idx| {
            // Read up to 4 bytes starting at chunk_idx * 4; missing
            // tail bytes are zero. u32::from_be_bytes packs them into
            // big-endian order with the first byte at bit position 24.
            let start = chunk_idx * 4;
            let end = (start + 4).min(bits.len());
            let mut buf = [0u8; 4];
            if start < end {
                buf[..end - start].copy_from_slice(&bits[start..end]);
            }
            u32::from_be_bytes(buf)
        })
        .collect();
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
        // 9 meaningful bits; the final word still carries the full
        // byte (0x80) — bits past length_bits sit in the word but the
        // GPU kernel won't decode them.
        let out = pack_be_words(&[0xFF, 0x80], 9);
        assert_eq!(out.words, vec![0xFF80_0000]);
        assert_eq!(out.length_bits, 9);
    }

    #[test]
    fn pack_zero_length_with_nonempty_bits_emits_no_words() {
        // length_bits == 0 is the source of truth; the bits slice is
        // ignored entirely.
        let out = pack_be_words(&[0xFF, 0xFF, 0xFF, 0xFF], 0);
        assert!(out.words.is_empty());
        assert_eq!(out.length_bits, 0);
    }

    #[test]
    fn pack_input_bytes_past_length_bits_are_dropped() {
        // 5 bytes given but only 24 meaningful bits — packer reads
        // exactly the bytes needed for ceil(24/32) = 1 word (4 bytes).
        // Byte 4 (0x42) sits past length_bits and is silently dropped.
        let out = pack_be_words(&[0xDE, 0xAD, 0xBE, 0xEF, 0x42], 24);
        assert_eq!(out.words, vec![0xDEAD_BEEF]);
        assert_eq!(out.length_bits, 24);
    }

    #[test]
    fn pack_one_bit_high_only() {
        let out = pack_be_words(&[0x80], 1);
        assert_eq!(out.words, vec![0x8000_0000]);
        assert_eq!(out.length_bits, 1);
    }

    #[test]
    fn pack_33_bits_spills_to_second_word_with_one_bit_set() {
        let out = pack_be_words(&[0xFF, 0xFF, 0xFF, 0xFF, 0x80], 33);
        assert_eq!(out.words, vec![0xFFFF_FFFF, 0x8000_0000]);
        assert_eq!(out.length_bits, 33);
    }

    #[test]
    #[should_panic(expected = "exceeds bits slice length")]
    fn pack_more_bits_than_input_panics_in_debug() {
        // Asks for 16 bits but only provides 1 byte. Debug-only
        // contract violation; in release this would silently emit
        // 0xAB00_0000 (zero-padded), which is exactly the footgun the
        // assert exists to prevent.
        let _ = pack_be_words(&[0xAB], 16);
    }
}
