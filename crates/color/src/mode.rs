//! Pixel color modes, mirroring `SplashColorMode` from `SplashTypes.h`.
//!
//! # Overview
//!
//! [`PixelMode`] describes the per-pixel memory layout used by rasterizer
//! bitmaps. Each variant corresponds to a distinct color space and bit depth.
//!
//! ## Bytes per pixel and `NCOMPS`
//!
//! [`NCOMPS`] is a `pub const` array kept for C-interop and workspace-level
//! re-exports. [`PixelMode::bytes_per_pixel`] encodes the same information via
//! an exhaustive `match`, so the compiler enforces consistency at compile time.
//! A `#[cfg(debug_assertions)]` test in this module asserts that the two sources
//! agree for every variant.
//!
//! ## The `Mono1` special case
//!
//! `Mono1` is a **packed-bits** format: one bit per pixel, MSB-first, with
//! `(width + 7) / 8` bytes per row. `bytes_per_pixel` returns **0** for
//! `Mono1` because no whole-byte count is meaningful at the per-pixel level.
//! Callers must check [`PixelMode::is_packed_bits`] before using
//! `bytes_per_pixel` or [`PixelMode::pixel_count_to_bytes`].
//!
//! ## Why `#[non_exhaustive]` is not used
//!
//! Adding `#[non_exhaustive]` would prevent the `match` inside
//! `bytes_per_pixel` from being exhaustiveness-checked by the compiler, which
//! is the primary safety guarantee this module provides. New variants must be
//! added here, in `NCOMPS`, and in every `match` across the workspace — the
//! compiler will flag each missed arm.

/// Pixel color mode, describing the memory layout of one pixel in a bitmap row.
///
/// The discriminant values mirror `SplashColorMode` in `SplashTypes.h` so that
/// raw FFI conversions remain stable.
///
/// # Packed-bits exception
///
/// [`Mono1`](PixelMode::Mono1) stores **one bit per pixel** and cannot be
/// expressed as a whole-number byte count per pixel. See
/// [`PixelMode::is_packed_bits`] and [`PixelMode::bits_per_pixel`].
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
pub enum PixelMode {
    /// 1 bit/pixel, MSB-first packed; row stride is `(width + 7) / 8` bytes.
    ///
    /// `bytes_per_pixel()` returns **0** for this variant — guard with
    /// [`is_packed_bits`](PixelMode::is_packed_bits) before using it.
    #[default]
    Mono1 = 0,
    /// 1 byte/pixel grayscale (8 bits, luminance only).
    Mono8 = 1,
    /// 3 bytes/pixel, channel order: R G B.
    Rgb8 = 2,
    /// 3 bytes/pixel, channel order: B G R.
    Bgr8 = 3,
    /// 4 bytes/pixel, channel order: X B G R, where X = 255.
    /// Used by the Cairo and Qt/QImage backends.
    Xbgr8 = 4,
    /// 4 bytes/pixel: C M Y K.
    Cmyk8 = 5,
    /// 8 bytes/pixel: C M Y K + 4 spot channels.
    /// Corresponds to `SPOT_NCOMPS = 4` in `SplashTypes.h`.
    DeviceN8 = 6,
}

/// Bytes per pixel for each [`PixelMode`] variant, indexed by discriminant.
///
/// `NCOMPS[0]` (`Mono1`) is **0** because `Mono1` is a packed-bits format;
/// see the [module-level documentation](self) for details.
///
/// This table is kept as a `pub const` for C-interop compatibility and because
/// the `raster` crate re-exports it. New variants **must** be reflected here
/// *and* in the `match` inside [`PixelMode::bytes_per_pixel`]; a
/// `#[cfg(debug_assertions)]` test asserts they stay in sync.
pub const NCOMPS: [usize; 7] = [
    0, // Mono1  — packed bits, not a whole byte count
    1, // Mono8
    3, // Rgb8
    3, // Bgr8
    4, // Xbgr8
    4, // Cmyk8
    8, // DeviceN8
];

impl PixelMode {
    /// Returns the number of bytes occupied by a single pixel in this mode.
    ///
    /// # Packed-bits exception
    ///
    /// Returns **0** for [`Mono1`](PixelMode::Mono1). Callers must check
    /// [`is_packed_bits`](Self::is_packed_bits) before dividing or
    /// multiplying by this value, or use [`pixel_count_to_bytes`](Self::pixel_count_to_bytes)
    /// which handles the case safely.
    ///
    /// # Exhaustiveness
    ///
    /// The implementation uses an exhaustive `match`, so adding a new variant
    /// without updating this function is a **compile error**.
    #[inline]
    #[must_use]
    pub const fn bytes_per_pixel(self) -> usize {
        match self {
            Self::Mono1 => 0,
            Self::Mono8 => 1,
            Self::Rgb8 | Self::Bgr8 => 3,
            Self::Xbgr8 | Self::Cmyk8 => 4,
            Self::DeviceN8 => 8,
        }
    }

    /// Returns the number of bits per pixel for this mode.
    ///
    /// Unlike [`bytes_per_pixel`](Self::bytes_per_pixel), this method always
    /// returns a meaningful positive value, making it safe to use for
    /// [`Mono1`](PixelMode::Mono1) without a prior guard.
    ///
    /// | Mode       | Bits |
    /// |------------|------|
    /// | `Mono1`    |    1 |
    /// | `Mono8`    |    8 |
    /// | `Rgb8`     |   24 |
    /// | `Bgr8`     |   24 |
    /// | `Xbgr8`    |   32 |
    /// | `Cmyk8`    |   32 |
    /// | `DeviceN8` |   64 |
    #[inline]
    #[must_use]
    pub const fn bits_per_pixel(self) -> usize {
        match self {
            Self::Mono1 => 1,
            Self::Mono8 => 8,
            Self::Rgb8 | Self::Bgr8 => 24,
            Self::Xbgr8 | Self::Cmyk8 => 32,
            Self::DeviceN8 => 64,
        }
    }

    /// Returns `true` if pixels are packed at sub-byte granularity.
    ///
    /// Currently only [`Mono1`](PixelMode::Mono1) is packed. For packed modes,
    /// [`bytes_per_pixel`](Self::bytes_per_pixel) returns 0 and must not be
    /// used for per-pixel arithmetic; use the row-stride formula
    /// `(width + 7) / 8` instead.
    #[inline]
    #[must_use]
    pub const fn is_packed_bits(self) -> bool {
        matches!(self, Self::Mono1)
    }

    /// Converts a raw discriminant byte into a [`PixelMode`], or `None` if
    /// the value does not correspond to any variant.
    ///
    /// Prefer this over `unsafe` transmute or unchecked `as` casts when
    /// parsing mode values from untrusted sources (e.g. C FFI, file headers).
    ///
    /// ```
    /// # use color::mode::PixelMode;
    /// assert_eq!(PixelMode::from_u8(2), Some(PixelMode::Rgb8));
    /// assert_eq!(PixelMode::from_u8(99), None);
    /// ```
    #[inline]
    #[must_use]
    pub const fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Mono1),
            1 => Some(Self::Mono8),
            2 => Some(Self::Rgb8),
            3 => Some(Self::Bgr8),
            4 => Some(Self::Xbgr8),
            5 => Some(Self::Cmyk8),
            6 => Some(Self::DeviceN8),
            _ => None,
        }
    }

    /// Multiplies `count` pixels by [`bytes_per_pixel`](Self::bytes_per_pixel),
    /// returning `None` on overflow **or** when the mode is
    /// [`Mono1`](PixelMode::Mono1) (packed bits have no per-pixel byte count).
    ///
    /// Use this instead of `count * mode.bytes_per_pixel()` to avoid panics or
    /// silent overflow in release builds.
    ///
    /// ```
    /// # use color::mode::PixelMode;
    /// assert_eq!(PixelMode::Rgb8.pixel_count_to_bytes(10), Some(30));
    /// assert_eq!(PixelMode::Mono1.pixel_count_to_bytes(10), None);
    /// assert_eq!(PixelMode::Rgb8.pixel_count_to_bytes(usize::MAX), None);
    /// ```
    #[inline]
    #[must_use]
    pub const fn pixel_count_to_bytes(self, count: usize) -> Option<usize> {
        if self.is_packed_bits() {
            return None;
        }
        count.checked_mul(self.bytes_per_pixel())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// All valid discriminant values for iteration in tests.
    const ALL: &[PixelMode] = &[
        PixelMode::Mono1,
        PixelMode::Mono8,
        PixelMode::Rgb8,
        PixelMode::Bgr8,
        PixelMode::Xbgr8,
        PixelMode::Cmyk8,
        PixelMode::DeviceN8,
    ];

    /// `NCOMPS` and `bytes_per_pixel` must agree for every variant.
    #[cfg(debug_assertions)]
    #[test]
    fn ncomps_matches_bytes_per_pixel() {
        for &mode in ALL {
            let i = mode as usize;
            assert_eq!(
                NCOMPS[i],
                mode.bytes_per_pixel(),
                "NCOMPS[{i}] disagrees with bytes_per_pixel for {mode:?}",
            );
        }
    }

    #[test]
    fn from_u8_roundtrip() {
        for &mode in ALL {
            assert_eq!(PixelMode::from_u8(mode as u8), Some(mode));
        }
    }

    #[test]
    fn from_u8_invalid() {
        for v in 7u8..=255 {
            assert_eq!(PixelMode::from_u8(v), None);
        }
    }

    #[test]
    fn mono1_is_packed() {
        assert!(PixelMode::Mono1.is_packed_bits());
        for &mode in ALL.iter().skip(1) {
            assert!(!mode.is_packed_bits());
        }
    }

    #[test]
    fn bits_per_pixel_nonzero() {
        for &mode in ALL {
            assert!(
                mode.bits_per_pixel() > 0,
                "{mode:?} has zero bits_per_pixel"
            );
        }
    }

    #[test]
    fn bits_consistent_with_bytes() {
        for &mode in ALL {
            if !mode.is_packed_bits() {
                assert_eq!(
                    mode.bits_per_pixel(),
                    mode.bytes_per_pixel() * 8,
                    "{mode:?}: bits_per_pixel != bytes_per_pixel * 8",
                );
            }
        }
    }

    #[test]
    fn pixel_count_to_bytes_mono1_is_none() {
        assert_eq!(PixelMode::Mono1.pixel_count_to_bytes(0), None);
        assert_eq!(PixelMode::Mono1.pixel_count_to_bytes(100), None);
    }

    #[test]
    fn pixel_count_to_bytes_overflow_is_none() {
        assert_eq!(PixelMode::Rgb8.pixel_count_to_bytes(usize::MAX), None);
    }

    #[test]
    fn pixel_count_to_bytes_correct() {
        assert_eq!(PixelMode::Mono8.pixel_count_to_bytes(5), Some(5));
        assert_eq!(PixelMode::Rgb8.pixel_count_to_bytes(10), Some(30));
        assert_eq!(PixelMode::Xbgr8.pixel_count_to_bytes(4), Some(16));
        assert_eq!(PixelMode::Cmyk8.pixel_count_to_bytes(3), Some(12));
        assert_eq!(PixelMode::DeviceN8.pixel_count_to_bytes(2), Some(16));
    }

    #[test]
    fn pixel_count_to_bytes_zero_count() {
        for &mode in ALL.iter().skip(1) {
            assert_eq!(mode.pixel_count_to_bytes(0), Some(0));
        }
    }
}
