//! Bit-depth expansion helpers for PDF image decoding.
//!
//! PDF images can store samples at 1, 2, 4, 8, or 16 bits per component (bpc).
//! The functions here normalise sub-byte and 16-bit data to 8 bpc so the rest of
//! the pipeline only ever sees one byte per sample.
//!
//! All functions are row-aware: each image row is padded to a whole-byte boundary
//! as required by PDF spec §7.4.1.

// ── Packed-bit unpacking ──────────────────────────────────────────────────────

/// Unpack MSB-first packed sub-byte data (PDF spec §7.4.1) to one byte per sample.
///
/// `bits` ∈ {1, 2, 4}; each row is padded to a whole-byte boundary.
/// `samples_per_row` is the number of samples to extract per row.
/// Each raw value in [0, 2^bits − 1] is transformed by `map` before being pushed.
///
/// Returns `None` if `bits` is not in {1, 2, 4}, or if
/// `samples_per_row × height` overflows `usize`.
pub(super) fn unpack_packed_bits(
    data: &[u8],
    bits: u32,
    samples_per_row: usize,
    height: u32,
    map: impl Fn(u8) -> u8, // raw_value → output byte
) -> Option<Vec<u8>> {
    // Hard precondition — not a debug_assert so adversarial PDF data cannot
    // trigger a divide-by-zero in release builds.
    if !matches!(bits, 1 | 2 | 4) {
        return None;
    }
    let samples_per_byte = (8 / bits) as usize;
    let row_bytes = samples_per_row.div_ceil(samples_per_byte);
    let height_usize = usize::try_from(height).ok()?;
    let total = samples_per_row.checked_mul(height_usize)?;
    let mut out = Vec::with_capacity(total);
    let mask: u8 = (1u8 << bits) - 1; // bits ≤ 4 ⟹ 1u8<<bits ≤ 16 — no overflow

    for row in 0..height_usize {
        let row_start = row.checked_mul(row_bytes)?;
        let row_data = if row_start < data.len() {
            &data[row_start..data.len().min(row_start + row_bytes)]
        } else {
            &[]
        };
        // Sample 0 occupies the most-significant `bits` bits of byte 0.
        // e.g. bits=4, byte=[ab cd]: s=0 → shift=4, s=1 → shift=0.
        for s in 0..samples_per_row {
            let byte_idx = s / samples_per_byte;
            let shift = bits as usize * (samples_per_byte - 1 - (s % samples_per_byte));
            let byte = row_data.get(byte_idx).copied().unwrap_or(0);
            let val = (byte >> shift) & mask;
            out.push(map(val));
        }
    }
    Some(out)
}

// ── 1-bpp expansion ───────────────────────────────────────────────────────────

/// Expand 1-bit-per-pixel packed data (MSB first) to 1 byte per pixel.
///
/// Output byte: `0x00` = black (bit=0 in PDF), `0xFF` = white (bit=1 in PDF).
/// Truncated rows are treated as all-black (missing bits default to 0).
///
/// Returns `None` if `width × height` overflows `usize`.
pub(super) fn expand_1bpp(data: &[u8], width: u32, height: u32) -> Option<Vec<u8>> {
    let width_usize = usize::try_from(width).ok()?;
    unpack_packed_bits(data, 1, width_usize, height, |val| {
        if val == 0 { 0x00 } else { 0xFF }
    })
}

// ── N-bpp expansion ───────────────────────────────────────────────────────────

/// Expand N-bits-per-sample packed data (MSB first) to 1 byte per sample, scaled to 0–255.
///
/// `BITS` must be 2 or 4 (enforced at compile time).  Samples are scaled to the
/// full 0–255 range:
/// - bpc 2: 4 levels  → 0x00, 0x55, 0xAA, 0xFF  (value × 85)
/// - bpc 4: 16 levels → 0x00, 0x11, 0x22, …, 0xFF  (value × 17)
///
/// `components` is the number of samples per pixel (1 for Gray, 3 for RGB).
/// Each row is padded to a whole number of source bytes (PDF spec §7.4.1).
///
/// Returns `None` if `width × height × components` overflows `usize`.
pub(super) fn expand_nbpp<const BITS: u32>(
    data: &[u8],
    width: u32,
    height: u32,
    components: usize,
) -> Option<Vec<u8>> {
    const { assert!(BITS == 2 || BITS == 4, "expand_nbpp: BITS must be 2 or 4") };
    // BITS ∈ {2, 4} ⟹ 1u8<<BITS ≤ 16 and (1u8<<BITS)-1 ≤ 15 — no overflow.
    let max_val = (1u8 << BITS) - 1;
    // Scale factor: maps max sample value to 255.
    // bpc 2: max=3,  scale=85  (3×85=255)
    // bpc 4: max=15, scale=17  (15×17=255)
    let scale = 255u16 / u16::from(max_val);
    let width_usize = usize::try_from(width).ok()?;
    let samples_per_row = width_usize.checked_mul(components)?;
    unpack_packed_bits(data, BITS, samples_per_row, height, |val| {
        // val ≤ max_val ≤ 15; val*scale ≤ 255 — fits u8.
        #[expect(
            clippy::cast_possible_truncation,
            reason = "val ≤ 15; val*scale ≤ 255 — fits u8"
        )]
        {
            (u16::from(val) * scale) as u8
        }
    })
}

/// Expand N-bits-per-index packed Indexed-image stream (MSB first) to 1 byte per index.
///
/// Unlike [`expand_nbpp`], values are NOT scaled — they are raw palette indices in
/// [0, 2^bits − 1].  `bits` ∈ {1, 2, 4}.  Rows are padded to a byte boundary.
///
/// Returns `None` if `width × height` overflows `usize`.
pub(super) fn expand_nbpp_indexed(
    data: &[u8],
    width: u32,
    height: u32,
    bits: u32,
) -> Option<Vec<u8>> {
    let width_usize = usize::try_from(width).ok()?;
    unpack_packed_bits(data, bits, width_usize, height, |val| val)
}

// ── 16-bpp downsampling ───────────────────────────────────────────────────────

/// Downsample 16-bit-per-sample big-endian data to 8 bits per sample.
///
/// Takes the high byte of each 16-bit sample (`sample >> 8`), discarding the low
/// byte.  A value of 0xFFFF maps to 0xFF; 0x0100 maps to 0x01; 0x00FF maps to 0x00.
/// `components` is the number of samples per pixel.
///
/// Returns `None` if `width × height × components` overflows `usize`, or if
/// `data` is too short for the declared image dimensions.
pub(super) fn downsample_16bpp(
    data: &[u8],
    width: u32,
    height: u32,
    components: usize,
) -> Option<Vec<u8>> {
    let width_usize = usize::try_from(width).ok()?;
    let height_usize = usize::try_from(height).ok()?;
    let npixels = width_usize.checked_mul(height_usize)?;
    let n_samples = npixels.checked_mul(components)?;
    let needed = n_samples.checked_mul(2)?;
    if data.len() < needed {
        log::warn!(
            "image: 16bpp data too short ({} bytes, need {needed} for {width}×{height}×{components}×2)",
            data.len()
        );
        return None;
    }
    // Each 16-bit sample is big-endian; take the high byte (bytes 0, 2, 4, …).
    let out: Vec<u8> = data[..needed].chunks_exact(2).map(|pair| pair[0]).collect();
    Some(out)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── expand_1bpp ───────────────────────────────────────────────────────────

    #[test]
    fn expand_1bpp_single_byte() {
        // 0b1010_0000 → pixels [white, black, white, black, black, black, black, black].
        let data = [0b1010_0000u8];
        let out = expand_1bpp(&data, 8, 1).unwrap();
        assert_eq!(out[0], 0xFF); // bit 7 = 1 → white
        assert_eq!(out[1], 0x00); // bit 6 = 0 → black
        assert_eq!(out[2], 0xFF); // bit 5 = 1 → white
        assert_eq!(out[3], 0x00); // bit 4 = 0 → black
    }

    #[test]
    fn expand_1bpp_partial_row() {
        // width=4, 1 byte: 0b1111_0000 → 4 white pixels (only top 4 bits used).
        let data = [0b1111_0000u8];
        let out = expand_1bpp(&data, 4, 1).unwrap();
        assert_eq!(out.len(), 4);
        assert!(out.iter().all(|&b| b == 0xFF));
    }

    // ── expand_nbpp (bpc 2 and 4) ─────────────────────────────────────────────

    #[test]
    fn expand_2bpp_all_levels() {
        // One byte = 4 samples × 2 bits, MSB first.
        // 0b11_10_01_00 = 0xE4 → values [3, 2, 1, 0] → scaled [255, 170, 85, 0]
        let data = [0b1110_0100u8];
        let out = expand_nbpp::<2>(&data, 4, 1, 1).unwrap();
        assert_eq!(out, [255, 170, 85, 0]);
    }

    #[test]
    fn expand_2bpp_single_pixel() {
        // Value 0b01 at bits 7-6 of byte → value 1 → 85.
        let data = [0b0100_0000u8];
        let out = expand_nbpp::<2>(&data, 1, 1, 1).unwrap();
        assert_eq!(out, [85]);
    }

    #[test]
    fn expand_4bpp_all_levels_two_pixels() {
        // Byte 0xFA → upper nibble 0xF=15 → 255; lower nibble 0xA=10 → 170.
        let data = [0xFAu8];
        let out = expand_nbpp::<4>(&data, 2, 1, 1).unwrap();
        assert_eq!(out, [255, 170]);
    }

    #[test]
    fn expand_4bpp_row_boundary_padding() {
        // 3 pixels at bpc=4 → 1.5 bytes → padded to 2 bytes per row.
        // Byte 0: pixels 0=0xA(170), 1=0x5(85). Byte 1: pixel 2=0x0(0), padding nibble ignored.
        let data = [0xA5u8, 0x00u8];
        let out = expand_nbpp::<4>(&data, 3, 1, 1).unwrap();
        assert_eq!(out, [170, 85, 0]);
    }

    #[test]
    fn expand_2bpp_multi_row_padding() {
        // 3 pixels × 2 bpc = 6 bits → padded to 1 byte per row.
        // Row 0: byte 0b11_10_01_xx → values [3, 2, 1] → [255, 170, 85], 2 pad bits ignored.
        // Row 1: byte 0b00_01_10_xx → values [0, 1, 2] → [0, 85, 170].
        let data = [0b1110_0100u8, 0b0001_1000u8];
        let out = expand_nbpp::<2>(&data, 3, 2, 1).unwrap();
        assert_eq!(out, [255, 170, 85, 0, 85, 170]);
    }

    // ── expand_nbpp_indexed ───────────────────────────────────────────────────

    #[test]
    fn expand_nbpp_indexed_4bpp() {
        // Byte 0xAF → upper nibble = index 10, lower nibble = index 15.
        let out = expand_nbpp_indexed(&[0xAFu8], 2, 1, 4).unwrap();
        assert_eq!(out, [10, 15]);
    }

    #[test]
    fn expand_nbpp_indexed_2bpp() {
        // Byte 0b11_10_01_00 = 0xE4 → indices [3, 2, 1, 0].
        let out = expand_nbpp_indexed(&[0xE4u8], 4, 1, 2).unwrap();
        assert_eq!(out, [3, 2, 1, 0]);
    }

    #[test]
    fn expand_nbpp_indexed_1bpp() {
        // Byte 0b1010_0000 → indices [1, 0, 1, 0, 0, 0, 0, 0] (8 pixels).
        let out = expand_nbpp_indexed(&[0b1010_0000u8], 8, 1, 1).unwrap();
        assert_eq!(out, [1, 0, 1, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn expand_nbpp_indexed_short_input_zero_pads() {
        // Empty data → all zeros (palette index 0 = first entry).
        let out = expand_nbpp_indexed(&[], 4, 1, 2).unwrap();
        assert_eq!(out, [0, 0, 0, 0]);
    }

    // ── downsample_16bpp ──────────────────────────────────────────────────────

    #[test]
    fn downsample_16bpp_takes_high_byte() {
        // Two 16-bit big-endian samples: 0xABCD → 0xAB, 0x1234 → 0x12.
        let data = [0xABu8, 0xCD, 0x12, 0x34];
        let out = downsample_16bpp(&data, 2, 1, 1).unwrap();
        assert_eq!(out, [0xAB, 0x12]);
    }

    #[test]
    fn downsample_16bpp_max_is_255() {
        // 0xFFFF → high byte 0xFF; 0x0000 → 0x00.
        let data = [0xFFu8, 0xFF, 0x00, 0x00];
        let out = downsample_16bpp(&data, 2, 1, 1).unwrap();
        assert_eq!(out, [0xFF, 0x00]);
    }

    #[test]
    fn downsample_16bpp_short_input_returns_none() {
        // 1 pixel RGB 16bpp needs 6 bytes; 4 bytes is too short.
        assert!(downsample_16bpp(&[0u8; 4], 1, 1, 3).is_none());
    }
}
