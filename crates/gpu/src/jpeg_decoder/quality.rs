//! Pick subsequence size for the parallel-Huffman JPEG decoder based on a
//! quality estimate derived from the luma quantisation table.

use crate::jpeg_decoder::JpegPreparedInput;

/// Choose the subsequence bit-width for the parallel-Huffman JPEG dispatcher.
///
/// High-quality images have denser Huffman streams (symbols are longer,
/// fewer symbols per bit) so larger subsequences give each thread more
/// work and reduce Phase 2 convergence iterations. Low-quality images
/// produce shorter codewords; smaller subsequences keep threads balanced.
///
/// Heuristic: sum of the 64 luma quantisation-table values.
/// Low sum → high quality → large subsequence.
#[must_use]
pub fn pick_subsequence_size(prep: &JpegPreparedInput) -> u32 {
    let luma_sum: u32 = prep
        .quant_tables
        .iter()
        .find_map(|qt| qt.as_ref())
        .map_or(0, |qt| qt.values.iter().map(|&v| u32::from(v)).sum());

    match luma_sum {
        0..=400 => 1024,  // high quality (Q ≈ 80–95)
        401..=900 => 512, // mid (Q ≈ 50–80)
        _ => 128,         // low (Q < 50)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jpeg_decoder::cpu_prepass::prepare_jpeg;

    #[test]
    fn high_quality_picks_1024() {
        let bytes = include_bytes!("../../../../tests/fixtures/jpeg/q95_scan.jpg");
        let prep = prepare_jpeg(bytes).unwrap();
        assert_eq!(pick_subsequence_size(&prep), 1024);
    }

    #[test]
    fn low_quality_picks_128() {
        let bytes = include_bytes!("../../../../tests/fixtures/jpeg/q20.jpg");
        let prep = prepare_jpeg(bytes).unwrap();
        assert_eq!(pick_subsequence_size(&prep), 128);
    }
}
