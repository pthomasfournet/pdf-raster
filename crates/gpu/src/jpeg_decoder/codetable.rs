//! GPU-side 2-tier (quick + full) Huffman lookup table.
//!
//! Layout follows GPUJPEG's `gpujpeg_huffman_gpu_decoder.cu` quick-table
//! optimisation: the top [`QUICK_CHECK_BITS`] bits of the peek index a
//! direct table; on miss the kernel falls back to a linear scan of the
//! full canonical book (codewords longer than `QUICK_CHECK_BITS`).
//!
//! `QUICK_CHECK_BITS = 10` matches GPUJPEG's choice — JPEG's typical
//! AC quantiser distributions keep ≥ 95 % of codewords ≤ 10 bits, so
//! the slow path is rare.

use crate::jpeg::{JpegHuffmanTable, validate_canonical_table, visit_canonical_codes};
use crate::jpeg_decoder::JpegGpuError;

/// Prefix bits indexed by the quick table. Must match the kernel.
pub const QUICK_CHECK_BITS: u32 = 10;
/// Size of the quick table (`1 << QUICK_CHECK_BITS`).
pub const QUICK_TABLE_SIZE: usize = 1 << QUICK_CHECK_BITS;

/// One packed quick-table entry.
///
/// Bit layout (LSB → MSB):
/// - bits 0..4: value bit size (`symbol & 0x0F`, 0..=15).
/// - bits 4..9: code bit size (0..=16; 0 ⇒ miss, kernel takes slow path).
/// - bits 9..16: run-length count + 1 (1..=16 for AC; 1 for DC).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuickEntry(pub u16);

impl QuickEntry {
    /// Sentinel for "no codeword starts with this prefix; fall back to
    /// the full table".
    pub const MISS: Self = Self(0);

    /// Pack `(value_bits, code_bits, runlength_plus1)` into the wire
    /// format. Debug-asserts on out-of-range inputs.
    #[must_use]
    pub fn pack(value_bits: u8, code_bits: u8, runlength_plus1: u8) -> Self {
        debug_assert!(value_bits <= 15, "value_bits {value_bits} out of range");
        debug_assert!(code_bits <= 16, "code_bits {code_bits} out of range");
        debug_assert!(
            (1..=64).contains(&runlength_plus1),
            "runlength_plus1 {runlength_plus1} out of range"
        );
        let v = u16::from(value_bits) & 0xF;
        let c = (u16::from(code_bits) & 0x1F) << 4;
        let r = (u16::from(runlength_plus1) & 0x7F) << 9;
        Self(v | c | r)
    }

    /// `true` if this prefix does not match any codeword ≤
    /// `QUICK_CHECK_BITS` bits long.
    #[must_use]
    pub const fn is_miss(self) -> bool {
        self.code_bits() == 0
    }

    /// Decoded `code_bits` field (0 if miss).
    #[must_use]
    pub const fn code_bits(self) -> u8 {
        ((self.0 >> 4) & 0x1F) as u8
    }

    /// Decoded `value_bits` field (size category, 0..=15).
    #[must_use]
    pub const fn value_bits(self) -> u8 {
        (self.0 & 0xF) as u8
    }

    /// Decoded `runlength_plus1` field.
    #[must_use]
    pub const fn runlength_plus1(self) -> u8 {
        ((self.0 >> 9) & 0x7F) as u8
    }
}

/// One canonical `(code_bits, code, symbol)` triple in the full table.
///
/// Used by the kernel's slow-path linear search for codewords longer
/// than [`QUICK_CHECK_BITS`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FullEntry {
    /// Codeword length (1..=16).
    pub code_bits: u8,
    /// Canonical codeword value (right-aligned).
    pub code: u16,
    /// Decoded symbol value (size category for DC; `(run << 4) | size`
    /// for AC).
    pub symbol: u8,
}

/// GPU-side 2-tier Huffman lookup table.
#[derive(Debug, Clone)]
pub struct GpuCodetable {
    /// `QUICK_TABLE_SIZE` direct-indexed entries (top
    /// [`QUICK_CHECK_BITS`] bits of the 16-bit peek).
    pub quick: Vec<QuickEntry>,
    /// Slow-path linear-scan table for codewords longer than
    /// [`QUICK_CHECK_BITS`]. Ordered by increasing `code_bits` so the
    /// kernel can exit early.
    pub full: Vec<FullEntry>,
}

/// Build a GPU-side codetable from a parsed JPEG DHT segment.
///
/// Packs DC and AC tables identically — DC symbols have
/// `(symbol >> 4) == 0` so `runlength_plus1` lands at 1, which is the
/// same encoding AC symbols use for "no run". The kernel branches on
/// the table identity, not on a per-entry DC/AC flag.
///
/// # Errors
///
/// Returns [`JpegGpuError::InvalidHuffmanTables`] if the DHT does not
/// form a valid canonical prefix code (empty, code-space overflow at
/// some length, or `values.len()` mismatches `sum(num_codes)`).
pub fn build_gpu_codetable(table: &JpegHuffmanTable) -> Result<GpuCodetable, JpegGpuError> {
    validate_canonical_table(table)
        .map_err(|e| JpegGpuError::InvalidHuffmanTables(e.to_string()))?;

    let mut quick = vec![QuickEntry::MISS; QUICK_TABLE_SIZE];
    let mut full = Vec::new();

    // Shared canonical walker; per-codeword emit splits into the
    // QUICK_CHECK_BITS-cut quick LUT vs. the long-code FullEntry list.
    visit_canonical_codes(table, |length, code, symbol| {
        let value_bits = symbol & 0x0F;
        let runlength_plus1 = ((symbol >> 4) & 0x0F) + 1;

        if u32::from(length) <= QUICK_CHECK_BITS {
            let shift = QUICK_CHECK_BITS - u32::from(length);
            let base = (code << shift) as usize;
            let span = 1usize << shift;
            let packed = QuickEntry::pack(value_bits, length, runlength_plus1);
            quick[base..base + span].fill(packed);
        } else {
            #[expect(
                clippy::cast_possible_truncation,
                reason = "code < 1 << length, length ≤ 16, so code fits in u16"
            )]
            let code_u16 = code as u16;
            full.push(FullEntry {
                code_bits: length,
                code: code_u16,
                symbol,
            });
        }
    })
    .map_err(|e| JpegGpuError::InvalidHuffmanTables(e.to_string()))?;

    Ok(GpuCodetable { quick, full })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jpeg::headers::DhtClass;
    use crate::jpeg_decoder::tests::fixtures::book4_table as book4_dc;

    #[test]
    fn quick_entry_pack_roundtrip() {
        let e = QuickEntry::pack(7, 9, 3);
        assert!(!e.is_miss());
        assert_eq!(e.value_bits(), 7);
        assert_eq!(e.code_bits(), 9);
        assert_eq!(e.runlength_plus1(), 3);
    }

    #[test]
    fn quick_entry_miss_has_zero_code_bits() {
        assert!(QuickEntry::MISS.is_miss());
        assert_eq!(QuickEntry::MISS.code_bits(), 0);
    }

    #[test]
    fn build_short_dc_table_fills_quick_table() {
        let tbl = build_gpu_codetable(&book4_dc()).expect("valid");
        assert_eq!(tbl.quick.len(), QUICK_TABLE_SIZE);
        assert!(tbl.full.is_empty(), "all codes fit in the quick table");

        // Prefix 00xxxxxxxx → first codeword, length 2, symbol 0 (size 0).
        let e0 = tbl.quick[0];
        assert!(!e0.is_miss());
        assert_eq!(e0.code_bits(), 2);
        assert_eq!(e0.value_bits(), 0);

        // Prefix 100xxxxxxx → third codeword, length 3, symbol 0x02.
        let e_100 = tbl.quick[0b10_0000_0000];
        assert_eq!(e_100.code_bits(), 3);
        assert_eq!(e_100.value_bits(), 2);
    }

    #[test]
    fn build_with_long_codewords_populates_full_table() {
        // 1 codeword each at lengths 11 and 12. Each should land in `full`.
        let table = JpegHuffmanTable {
            class: DhtClass::Ac,
            table_id: 0,
            num_codes: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
            values: vec![0xAB, 0xCD],
        };
        let tbl = build_gpu_codetable(&table).expect("valid");
        assert_eq!(tbl.full.len(), 2);
        assert_eq!(tbl.full[0].code_bits, 11);
        assert_eq!(tbl.full[0].symbol, 0xAB);
        assert_eq!(tbl.full[1].code_bits, 12);
        assert_eq!(tbl.full[1].symbol, 0xCD);
        // The quick table sees no hits.
        assert!(tbl.quick.iter().all(|e| e.is_miss()));
    }

    #[test]
    fn build_rejects_empty() {
        let t = JpegHuffmanTable {
            class: DhtClass::Dc,
            table_id: 0,
            num_codes: [0; 16],
            values: vec![],
        };
        let err = build_gpu_codetable(&t).unwrap_err();
        assert!(matches!(err, JpegGpuError::InvalidHuffmanTables(_)));
    }

    #[test]
    fn build_rejects_length_mismatch() {
        let mut t = book4_dc();
        // The popped value is irrelevant — we only care that the
        // resulting `values` length no longer matches sum(num_codes).
        let _ = t.values.pop();
        let err = build_gpu_codetable(&t).unwrap_err();
        assert!(matches!(err, JpegGpuError::InvalidHuffmanTables(_)));
    }

    #[test]
    fn ac_symbol_packs_runlength_correctly() {
        // AC symbol 0x52 = (run=5, size=2). runlength_plus1 = 6, value_bits = 2.
        let table = JpegHuffmanTable {
            class: DhtClass::Ac,
            table_id: 0,
            num_codes: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            values: vec![0x52],
        };
        let tbl = build_gpu_codetable(&table).expect("valid");
        // Length-1 code "0" fills the lower half of the quick table.
        let e = tbl.quick[0];
        assert_eq!(e.value_bits(), 2);
        assert_eq!(e.code_bits(), 1);
        assert_eq!(e.runlength_plus1(), 6);
    }
}
