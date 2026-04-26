//! Transfer function LUT — a `[u8; 256]` lookup table applied per channel at
//! compositing write time.
//!
//! Matches `SplashState`'s `rgbTransferR/G/B`, `grayTransfer`,
//! `cmykTransfer*`, and `deviceNTransfer` arrays.
//!
//! The identity LUT (`IDENTITY`) is the default: every value maps to itself.
//! Non-identity LUTs implement PDF transfer functions and halftone screen
//! calibration curves.

/// A per-channel lookup table: output[i] = lut[input[i]].
///
/// Always 256 entries (one per possible byte value). Stored as a newtype so
/// that callers cannot accidentally pass a raw `[u8; 256]` in the wrong order.
#[derive(Clone)]
pub struct TransferLut(pub [u8; 256]);

impl TransferLut {
    /// Identity mapping: every value i maps to i.
    pub const IDENTITY: Self = {
        let mut t = [0u8; 256];
        // Use a u8 loop variable so the index→byte cast is lossless by type.
        let mut i = 0u8;
        loop {
            t[i as usize] = i;
            if i == 255 {
                break;
            }
            i += 1;
        }
        Self(t)
    };

    /// Apply the LUT to a single byte.
    #[inline]
    #[must_use]
    pub const fn apply(&self, v: u8) -> u8 {
        self.0[v as usize]
    }

    /// Apply the LUT in-place to every byte in a row slice.
    ///
    /// The slice may span multiple channels; callers that want per-channel
    /// application must stride through the slice themselves.
    pub fn apply_row(&self, row: &mut [u8]) {
        for b in row.iter_mut() {
            *b = self.apply(*b);
        }
    }

    /// Invert: produce a new LUT where output[i] = 255 - self[255 - i].
    /// Used by `GraphicsState::set_transfer` to derive the CMYK LUTs from
    /// the RGB/gray LUTs (matching `SplashState::setTransfer` in SplashState.cc).
    #[must_use]
    pub fn invert_complement(&self) -> Self {
        let mut out = [0u8; 256];
        for (i, v) in out.iter_mut().enumerate() {
            *v = 255 - self.0[255 - i];
        }
        Self(out)
    }

    /// Return the raw table, e.g. for memcpy into a state block.
    #[must_use]
    pub const fn as_array(&self) -> &[u8; 256] {
        &self.0
    }
}

impl Default for TransferLut {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl std::fmt::Debug for TransferLut {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "TransferLut([{}, {}, ..., {}])",
            self.0[0], self.0[1], self.0[255]
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_is_identity() {
        let lut = TransferLut::IDENTITY;
        for i in 0u8..=255 {
            assert_eq!(lut.apply(i), i);
        }
    }

    #[test]
    fn apply_row_identity() {
        let lut = TransferLut::IDENTITY;
        let mut row: Vec<u8> = (0..=255).collect();
        let original = row.clone();
        lut.apply_row(&mut row);
        assert_eq!(row, original);
    }

    #[test]
    fn invert_complement_roundtrip() {
        let lut = TransferLut::IDENTITY;
        let inv = lut.invert_complement();
        // identity inverted complement = identity (255 - (255 - i) = i)
        for i in 0u8..=255 {
            assert_eq!(inv.apply(i), i);
        }
    }

    #[test]
    fn invert_complement_nontrivial() {
        // Build a LUT that maps i → 255 - i (inversion).
        let mut t = [0u8; 256];
        for (i, v) in t.iter_mut().enumerate() {
            *v = u8::try_from(255 - i).unwrap_or(0);
        }
        let lut = TransferLut(t);
        let inv = lut.invert_complement();
        // inv[i] = 255 - lut[255-i] = 255 - (255 - (255-i)) = 255 - i
        for i in 0u8..=255 {
            assert_eq!(inv.apply(i), 255 - i);
        }
    }
}
