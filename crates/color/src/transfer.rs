//! Transfer function lookup tables for PDF rendering.
//!
//! A *transfer function* in PDF maps a device-colour value in \[0, 1\] to an
//! adjusted output value, allowing calibration curves and halftone screen
//! corrections.  Inside this crate the function is pre-sampled to a 256-entry
//! `u8→u8` lookup table (`TransferLut`): entry `i` holds the output byte for
//! input byte `i`.
//!
//! ## Identity and non-identity LUTs
//!
//! The identity LUT (`TransferLut::IDENTITY`) maps every value to itself and is
//! the correct default when no PDF transfer function is in effect.  Non-identity
//! LUTs are built by sampling the PDF transfer function at the 256 points
//! `i/255.0` for `i` in `0..=255` and converting the floating-point result back
//! to a `u8`.
//!
//! ## PDF mapping
//!
//! PDF transfer functions work per-channel.  The graphics state therefore keeps
//! one `TransferLut` per channel (R, G, B, gray, and the four CMYK channels).
//! The CMYK LUTs are derived from the RGB/gray LUT via
//! [`TransferLut::invert_complement`], matching the logic in
//! `SplashState::setTransfer`.
//!
//! ## Safety / infallibility
//!
//! Every public method is infallible.  `apply` is a plain array index whose
//! index is always a `u8`, so it is always in `0..=255`, which is exactly the
//! length of the internal array.  There are no panicking paths in normal use.
//!
//! ## Public inner field
//!
//! `TransferLut(pub [u8; 256])` exposes the raw array deliberately: PDF allows
//! arbitrary 256-entry transfer tables, and code that samples a PDF function
//! needs direct write access.  See [`From<[u8; 256]>`](TransferLut#impl-From<[u8;+256]>-for-TransferLut)
//! for ergonomic construction.

/// A per-channel lookup table: `output[i] = lut[input[i]]`.
///
/// Always exactly 256 entries — one for every possible byte value.  The type is
/// a newtype so that callers cannot accidentally pass a raw `[u8; 256]` in the
/// wrong argument position.
///
/// The inner field is `pub` intentionally: PDF transfer functions are arbitrary
/// 256-entry tables, and the code that samples them needs direct write access.
/// Prefer [`From<[u8; 256]>`] for construction from an owned array.
#[derive(Clone, PartialEq, Eq)]
pub struct TransferLut(pub [u8; 256]);

// Compile-time assertion: the inner array must be exactly 256 bytes.
// This guards against any future refactor that might change the constant.
const _ASSERT_LUT_LEN: () = assert!(
    std::mem::size_of::<TransferLut>() == 256,
    "TransferLut must contain exactly 256 bytes"
);

impl TransferLut {
    /// The identity mapping: every value `i` maps to `i`.
    ///
    /// This is the correct default when no PDF transfer function is active.
    /// All 256 entries are initialised at compile time via a `const` loop using
    /// a `u8` counter so that the index-to-byte cast is lossless by type.
    pub const IDENTITY: Self = {
        let mut t = [0u8; 256];
        // A u8 loop variable guarantees the index value fits in a byte without
        // any arithmetic: `i as usize` is always in 0..=255.
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

    /// The per-byte inverting mapping: every value `i` maps to `255 - i`.
    ///
    /// Unlike [`invert_complement`](Self::invert_complement) (which composes an
    /// existing LUT with the complement), this is the standalone "negate" LUT —
    /// suitable for test fixtures that need a non-identity transfer where the
    /// output is easy to assert against. Same `u8` loop-counter rationale as
    /// [`IDENTITY`](Self::IDENTITY): index-to-byte cast is lossless by type.
    pub const INVERTED: Self = {
        let mut t = [0u8; 256];
        let mut i = 0u8;
        loop {
            t[i as usize] = 255 - i;
            if i == 255 {
                break;
            }
            i += 1;
        }
        Self(t)
    };

    /// Apply the LUT to a single byte.
    ///
    /// # Safety / infallibility
    ///
    /// `v` is a `u8`, so `v as usize` is always in `0..=255`.  The inner array
    /// is always 256 entries (enforced by the compile-time assertion
    /// `_ASSERT_LUT_LEN`), so the index is always in bounds.  This method
    /// cannot panic.
    #[inline]
    #[must_use]
    pub const fn apply(&self, v: u8) -> u8 {
        self.0[v as usize]
    }

    /// Produce a new LUT where `output[i] = 255 - self[255 - i]`.
    ///
    /// Used by `GraphicsState::set_transfer` to derive the CMYK LUTs from the
    /// RGB/gray LUTs, matching `SplashState::setTransfer` in SplashState.cc.
    ///
    /// # Index safety
    ///
    /// The closure index `i` comes from `0..256` (the length of the output
    /// array), so `255 - i` is always in `0..=255` — always a valid index into
    /// the 256-entry `self.0` array.
    #[must_use]
    pub fn invert_complement(&self) -> Self {
        Self(std::array::from_fn(|i| 255 - self.0[255 - i]))
    }

    /// Return a reference to the raw 256-entry array.
    ///
    /// Useful for `memcpy`-style copies into an external state block.
    #[must_use]
    pub const fn as_array(&self) -> &[u8; 256] {
        &self.0
    }

    /// Compose two LUTs: apply `self` first, then `other`.
    ///
    /// The returned LUT is equivalent to `other.apply(self.apply(v))` for every
    /// input byte `v`.  This is the natural piping / chaining operation for
    /// transfer functions.
    ///
    /// # Examples
    ///
    /// ```
    /// use color::TransferLut;
    ///
    /// // Build an inversion LUT: i → 255 - i.
    /// let mut inv_table = [0u8; 256];
    /// for i in 0usize..=255 {
    ///     inv_table[i] = (255 - i) as u8;
    /// }
    /// let inv = TransferLut::from(inv_table);
    ///
    /// // Composing inversion with itself gives the identity.
    /// let composed = inv.compose(&inv);
    /// assert_eq!(composed, TransferLut::IDENTITY);
    ///
    /// // Composing with identity leaves any LUT unchanged.
    /// let id = TransferLut::IDENTITY;
    /// assert_eq!(inv.compose(&id), inv);
    /// assert_eq!(id.compose(&inv), inv);
    /// ```
    #[must_use]
    pub fn compose(&self, other: &Self) -> Self {
        Self(std::array::from_fn(|i| other.apply(self.0[i])))
    }

    /// Compose a slice of LUTs left-to-right, starting from the identity.
    ///
    /// `compose_many(&[a, b, c])` is equivalent to
    /// `IDENTITY.compose(a).compose(b).compose(c)`.  An empty slice returns
    /// [`IDENTITY`](Self::IDENTITY).
    #[must_use]
    pub fn compose_many(luts: &[&Self]) -> Self {
        luts.iter()
            .fold(Self::IDENTITY, |acc, lut| acc.compose(lut))
    }
}

impl Default for TransferLut {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl std::fmt::Debug for TransferLut {
    /// Compact representation showing the first two and last entry only,
    /// avoiding a 256-element dump that would overwhelm debug output.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "TransferLut([{}, {}, ..., {}])",
            self.0[0], self.0[1], self.0[255]
        )
    }
}

impl From<[u8; 256]> for TransferLut {
    /// Construct a `TransferLut` from a raw 256-entry array.
    fn from(arr: [u8; 256]) -> Self {
        Self(arr)
    }
}

impl From<TransferLut> for [u8; 256] {
    /// Unwrap a `TransferLut` back into a raw array (consuming).
    fn from(lut: TransferLut) -> Self {
        lut.0
    }
}

impl AsRef<[u8]> for TransferLut {
    /// View the LUT as a byte slice (length 256).
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── IDENTITY ────────────────────────────────────────────────────────────

    #[test]
    fn identity_all_256_entries() {
        let lut = TransferLut::IDENTITY;
        for i in 0u8..=255 {
            assert_eq!(lut.apply(i), i, "IDENTITY should map {i} to {i}");
        }
    }

    // ── invert_complement ───────────────────────────────────────────────────

    #[test]
    fn invert_complement_of_identity_is_identity() {
        let lut = TransferLut::IDENTITY;
        let inv = lut.invert_complement();
        // 255 - (255 - i) = i for all i
        for i in 0u8..=255 {
            assert_eq!(
                inv.apply(i),
                i,
                "invert_complement(identity)[{i}] should be {i}"
            );
        }
    }

    #[test]
    fn invert_complement_nontrivial() {
        // Build a LUT that maps i → 255 - i (inversion).
        let lut = TransferLut(std::array::from_fn(|i| {
            u8::try_from(255 - i).expect("i < 256")
        }));
        let inv = lut.invert_complement();
        // inv[i] = 255 - lut[255 - i] = 255 - (255 - (255 - i)) = 255 - i
        for i in 0u8..=255 {
            assert_eq!(inv.apply(i), 255 - i);
        }
    }

    // ── compose ─────────────────────────────────────────────────────────────

    #[test]
    fn compose_identity_is_neutral() {
        let id = TransferLut::IDENTITY;
        let lut = TransferLut::from(std::array::from_fn::<u8, 256, _>(|i| {
            u8::try_from((i * 2) % 256).expect("(i*2)%256 < 256")
        }));
        assert_eq!(lut.compose(&id), lut, "lut ∘ id should equal lut");
        assert_eq!(id.compose(&lut), lut, "id ∘ lut should equal lut");
    }

    #[test]
    fn compose_inversion_twice_is_identity() {
        let inv = TransferLut::from(std::array::from_fn::<u8, 256, _>(|i| {
            u8::try_from(255 - i).expect("i < 256")
        }));
        let roundtrip = inv.compose(&inv);
        assert_eq!(roundtrip, TransferLut::IDENTITY);
    }

    #[test]
    fn compose_associativity() {
        let double = TransferLut::from(std::array::from_fn::<u8, 256, _>(|i| {
            u8::try_from((i * 2) % 256).expect("(i*2)%256 < 256")
        }));
        let inv = TransferLut::from(std::array::from_fn::<u8, 256, _>(|i| {
            u8::try_from(255 - i).expect("i < 256")
        }));
        let half = TransferLut::from(std::array::from_fn::<u8, 256, _>(|i| {
            u8::try_from(i / 2).expect("i/2 < 256")
        }));
        // (double ∘ inv) ∘ half  ==  double ∘ (inv ∘ half)
        let left = double.compose(&inv).compose(&half);
        let right = double.compose(&inv.compose(&half));
        assert_eq!(left, right);
    }

    // ── compose_many ────────────────────────────────────────────────────────

    #[test]
    fn compose_many_empty_is_identity() {
        let result = TransferLut::compose_many(&[]);
        assert_eq!(result, TransferLut::IDENTITY);
    }

    #[test]
    fn compose_many_single() {
        let inv = TransferLut::from(std::array::from_fn::<u8, 256, _>(|i| {
            u8::try_from(255 - i).expect("i < 256")
        }));
        let result = TransferLut::compose_many(&[&inv]);
        assert_eq!(result, inv);
    }

    #[test]
    fn compose_many_matches_sequential_compose() {
        let a = TransferLut::from(std::array::from_fn::<u8, 256, _>(|i| {
            u8::try_from((i * 3) % 256).expect("(i*3)%256 < 256")
        }));
        let b = TransferLut::from(std::array::from_fn::<u8, 256, _>(|i| {
            u8::try_from(255 - i).expect("i < 256")
        }));
        let c = TransferLut::from(std::array::from_fn::<u8, 256, _>(|i| {
            u8::try_from(i / 2).expect("i/2 < 256")
        }));
        let expected = a.compose(&b).compose(&c);
        let got = TransferLut::compose_many(&[&a, &b, &c]);
        assert_eq!(got, expected);
    }

    // ── From / AsRef conversions ─────────────────────────────────────────────

    #[test]
    fn from_array_roundtrip() {
        let arr =
            std::array::from_fn::<u8, 256, _>(|i| u8::try_from(i ^ 0xAA).expect("i^0xAA < 256"));
        let lut = TransferLut::from(arr);
        let arr2: [u8; 256] = lut.into();
        assert_eq!(arr, arr2);
    }

    #[test]
    fn as_ref_length_and_content() {
        let lut = TransferLut::IDENTITY;
        let slice: &[u8] = lut.as_ref();
        assert_eq!(slice.len(), 256);
        for (i, &b) in slice.iter().enumerate() {
            assert_eq!(b, u8::try_from(i).expect("IDENTITY has 256 entries"));
        }
    }

    // ── Debug ────────────────────────────────────────────────────────────────

    #[test]
    fn debug_format_does_not_dump_256_entries() {
        let s = format!("{:?}", TransferLut::IDENTITY);
        // Should be short — just a compact summary, not 256 comma-separated values.
        assert!(s.len() < 80, "Debug output unexpectedly long: {s}");
        assert!(s.contains("TransferLut"));
    }
}
