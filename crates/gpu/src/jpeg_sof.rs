//! Lightweight JPEG SOF-marker peek.

/// Coding mode of a JPEG stream, determined by the first SOF marker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JpegVariant {
    /// SOF0 — baseline DCT. Supported by VA-API `VAEntrypointVLD`.
    Baseline,
    /// SOF2 / SOF10 — progressive DCT. Supported by nvJPEG; not by VA-API.
    Progressive,
    /// Any other SOF marker (SOF1, SOF3, SOF9, etc.), or no SOF found.
    Other,
}

/// SOF marker information extracted in a single zero-allocation pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JpegSof {
    /// Coding variant determined by the SOF marker byte.
    pub variant: JpegVariant,
    /// Number of image components (1 = grayscale, 3 = YCbCr/RGB, 4 = CMYK).
    /// `0` means the SOF payload was too short to read the field.
    pub components: u8,
}

/// Peek into `data` and return SOF variant + component count.
///
/// Returns `None` only if `data` does not start with a valid JPEG SOI marker
/// (`0xFF 0xD8`). Returns `Some` with `variant = Other` for any malformed
/// content encountered after the SOI, or when no SOF marker is found.
/// Zero allocations; stops scanning at the first SOF marker found.
#[must_use]
pub fn jpeg_sof_info(data: &[u8]) -> Option<JpegSof> {
    if data.get(0..2) != Some(&[0xFF, 0xD8]) {
        return None;
    }

    let mut pos = 2usize;
    loop {
        // Skip padding 0xFF bytes, but require at least one 0xFF prefix.
        let ff_start = pos;
        while data.get(pos).copied() == Some(0xFF) {
            pos += 1;
        }
        if pos == ff_start {
            return Some(JpegSof {
                variant: JpegVariant::Other,
                components: 0,
            });
        }
        let Some(&marker) = data.get(pos) else {
            return Some(JpegSof {
                variant: JpegVariant::Other,
                components: 0,
            });
        };
        pos += 1;

        // Stand-alone markers (no length field): TEM (0x01), SOI, EOI, RST0–RST7.
        if matches!(marker, 0x01 | 0xD0..=0xD9) {
            continue;
        }

        // Read the 2-byte segment length (includes the length field itself).
        let Some(&hi) = data.get(pos) else {
            return Some(JpegSof {
                variant: JpegVariant::Other,
                components: 0,
            });
        };
        let Some(&lo) = data.get(pos + 1) else {
            return Some(JpegSof {
                variant: JpegVariant::Other,
                components: 0,
            });
        };
        let seg_len = ((hi as usize) << 8) | (lo as usize);
        if seg_len < 2 {
            return Some(JpegSof {
                variant: JpegVariant::Other,
                components: 0,
            });
        }

        // SOS signals end of the header section; no SOF will appear after this.
        if marker == 0xDA {
            return Some(JpegSof {
                variant: JpegVariant::Other,
                components: 0,
            });
        }

        // SOF markers: 0xC0–0xCF excluding DHT (0xC4) and DAC (0xCC).
        // SOF payload layout: P(1) Y(2) X(2) Nf(1) → component count at offset +5
        // from the start of the payload (i.e. data[pos + 2 + 5] = data[pos + 7]).
        let variant = match marker {
            0xC0 => Some(JpegVariant::Baseline),
            0xC2 | 0xCA => Some(JpegVariant::Progressive),
            0xC1 | 0xC3 | 0xC5..=0xC7 | 0xC9 | 0xCB | 0xCD..=0xCF => Some(JpegVariant::Other),
            _ => None,
        };

        if let Some(variant) = variant {
            // pos points at the length field; payload starts at pos+2.
            // SOF payload: P(1) Y(2) X(2) Nf(1) → Nf is at pos+2+5 = pos+7.
            let components = data.get(pos + 7).copied().unwrap_or(0);
            return Some(JpegSof {
                variant,
                components,
            });
        }

        // Skip over this segment's payload.
        let Some(new_pos) = pos.checked_add(seg_len) else {
            return Some(JpegSof {
                variant: JpegVariant::Other,
                components: 0,
            });
        };
        pos = new_pos;
    }
}

/// Peek into `data` and return the JPEG coding variant.
///
/// Thin wrapper around [`jpeg_sof_info`] for callers that only need the variant.
#[must_use]
pub fn jpeg_sof_type(data: &[u8]) -> Option<JpegVariant> {
    jpeg_sof_info(data).map(|s| s.variant)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_jpeg(marker: u8, payload: &[u8]) -> Vec<u8> {
        let len = (payload.len() as u16) + 2;
        let mut v = vec![0xFF, 0xD8];
        v.push(0xFF);
        v.push(marker);
        v.push((len >> 8) as u8);
        v.push((len & 0xFF) as u8);
        v.extend_from_slice(payload);
        v.push(0xFF);
        v.push(0xD9);
        v
    }

    #[test]
    fn detects_baseline() {
        let data = make_jpeg(0xC0, &[8, 0, 1]);
        assert_eq!(jpeg_sof_type(&data), Some(JpegVariant::Baseline));
    }

    #[test]
    fn detects_progressive_sof2() {
        let data = make_jpeg(0xC2, &[8, 0, 1]);
        assert_eq!(jpeg_sof_type(&data), Some(JpegVariant::Progressive));
    }

    #[test]
    fn detects_progressive_sof10() {
        let data = make_jpeg(0xCA, &[8, 0, 1]);
        assert_eq!(jpeg_sof_type(&data), Some(JpegVariant::Progressive));
    }

    #[test]
    fn detects_other_sof1() {
        let data = make_jpeg(0xC1, &[8, 0, 1]);
        assert_eq!(jpeg_sof_type(&data), Some(JpegVariant::Other));
    }

    #[test]
    fn not_jpeg_returns_none() {
        assert_eq!(jpeg_sof_type(b"not a jpeg"), None);
        assert_eq!(jpeg_sof_type(&[]), None);
        assert_eq!(jpeg_sof_type(&[0xFF, 0xD7]), None);
    }

    #[test]
    fn no_sof_returns_other() {
        let data = make_jpeg(0xE0, &[0u8; 14]);
        assert_eq!(jpeg_sof_type(&data), Some(JpegVariant::Other));
    }

    #[test]
    fn skips_app_segments_before_sof() {
        let app0_payload = [0u8; 14];
        let app0_len = (app0_payload.len() as u16) + 2;
        let sof0_payload = [8u8, 0, 1];
        let sof0_len = (sof0_payload.len() as u16) + 2;
        let mut data = vec![0xFF, 0xD8];
        data.extend_from_slice(&[0xFF, 0xE0, (app0_len >> 8) as u8, (app0_len & 0xFF) as u8]);
        data.extend_from_slice(&app0_payload);
        data.extend_from_slice(&[0xFF, 0xC0, (sof0_len >> 8) as u8, (sof0_len & 0xFF) as u8]);
        data.extend_from_slice(&sof0_payload);
        data.extend_from_slice(&[0xFF, 0xD9]);
        assert_eq!(jpeg_sof_type(&data), Some(JpegVariant::Baseline));
    }

    #[test]
    fn truncated_segment_does_not_panic() {
        // valid SOI + marker byte prefix, but length bytes missing → Some(Other)
        let data = vec![0xFF, 0xD8, 0xFF, 0xC0];
        let result = jpeg_sof_type(&data);
        assert_eq!(result, Some(JpegVariant::Other));
    }

    #[test]
    fn skips_dqt_and_dht_before_sof() {
        // DQT (0xDB) with 2-byte payload, then SOF0 — verifies non-APP segments are skipped.
        let dqt_payload = [0u8; 2];
        let dqt_len = (dqt_payload.len() as u16) + 2;
        let sof0_payload = [8u8, 0, 1];
        let sof0_len = (sof0_payload.len() as u16) + 2;
        let mut data = vec![0xFF, 0xD8];
        data.extend_from_slice(&[0xFF, 0xDB, (dqt_len >> 8) as u8, (dqt_len & 0xFF) as u8]);
        data.extend_from_slice(&dqt_payload);
        data.extend_from_slice(&[0xFF, 0xC0, (sof0_len >> 8) as u8, (sof0_len & 0xFF) as u8]);
        data.extend_from_slice(&sof0_payload);
        data.extend_from_slice(&[0xFF, 0xD9]);
        assert_eq!(jpeg_sof_type(&data), Some(JpegVariant::Baseline));
    }

    #[test]
    fn sos_before_sof_returns_other() {
        // Malformed: SOS (0xDA) before any SOF. Must return Other quickly, not scan entropy data.
        let sos_payload = [0x01u8, 0x01, 0x00, 0x00, 0x00]; // minimal scan header
        let sos_len = (sos_payload.len() as u16) + 2;
        let mut data = vec![0xFF, 0xD8];
        data.extend_from_slice(&[0xFF, 0xDA, (sos_len >> 8) as u8, (sos_len & 0xFF) as u8]);
        data.extend_from_slice(&sos_payload);
        data.extend_from_slice(&[0xFF, 0xD9]);
        assert_eq!(jpeg_sof_type(&data), Some(JpegVariant::Other));
    }

    #[test]
    fn minimum_seg_len_two_returns_other() {
        // seg_len == 2 means zero-byte payload. Malformed but should not panic.
        let mut data = vec![0xFF, 0xD8];
        data.extend_from_slice(&[0xFF, 0xE0, 0x00, 0x02]); // APP0, length=2 (zero payload)
        data.extend_from_slice(&[0xFF, 0xD9]);
        // No SOF found → Other
        assert_eq!(jpeg_sof_type(&data), Some(JpegVariant::Other));
    }

    #[test]
    fn jpeg_sof_info_returns_component_count() {
        // SOF0 payload: P(1)=8, Y(2)=0x00,0x10, X(2)=0x00,0x10, Nf(1)=3 → RGB
        let payload = [8u8, 0x00, 0x10, 0x00, 0x10, 3];
        let data = make_jpeg(0xC0, &payload);
        let sof = jpeg_sof_info(&data).unwrap();
        assert_eq!(sof.variant, JpegVariant::Baseline);
        assert_eq!(sof.components, 3);
    }

    #[test]
    fn jpeg_sof_info_grayscale() {
        let payload = [8u8, 0x00, 0x10, 0x00, 0x10, 1];
        let data = make_jpeg(0xC0, &payload);
        let sof = jpeg_sof_info(&data).unwrap();
        assert_eq!(sof.variant, JpegVariant::Baseline);
        assert_eq!(sof.components, 1);
    }

    #[test]
    fn jpeg_sof_info_stream_ends_before_nf_returns_zero_components() {
        // SOI + SOF0 length field + 4 payload bytes, then EOF (no EOI).
        // data[pos+7] is past end of slice → get() returns None → components=0.
        // Bytes: FF D8 FF C0 00 06 08 00 10 00  (10 bytes, truncated before Nf)
        let data = vec![0xFF, 0xD8, 0xFF, 0xC0, 0x00, 0x06, 0x08, 0x00, 0x10, 0x00];
        let sof = jpeg_sof_info(&data).unwrap();
        assert_eq!(sof.variant, JpegVariant::Baseline);
        assert_eq!(sof.components, 0);
    }
}
