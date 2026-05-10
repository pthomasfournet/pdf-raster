//! Pixel types and the [`Pixel`] trait.
//!
//! # Design overview
//!
//! Each concrete type is a `#[repr(C)]` struct that also implements
//! [`bytemuck::Pod`] and [`bytemuck::Zeroable`], allowing zero-copy casts
//! between `&[u8]` row buffers and typed pixel slices via
//! `bytemuck::cast_slice`.
//!
//! ## The `Pixel` trait
//!
//! [`Pixel`] is the generic bound used by `Bitmap<P>` and the rasterizer
//! pipeline. It requires `Copy + Pod + Zeroable + Send + Sync + 'static` so
//! that pixel buffers can be shared across threads without additional
//! synchronisation. Every implementation must keep `BYTES` equal to
//! `std::mem::size_of::<Self>()` — the module-level compile-time assertions
//! enforce this.
//!
//! ## Monomorphization
//!
//! The `Pixel` bound is used as a generic parameter on `Bitmap<P>` and the hot
//! rasterizer loops. The compiler generates one specialised code path per pixel
//! format, eliminating runtime mode dispatch in the inner loop.
//!
//! ## `AnyColor` vs `Pixel`
//!
//! Use [`AnyColor`] when you need to carry a pixel value alongside its mode at
//! runtime (e.g. paper colour, graphics-state default colour). Use a concrete
//! `impl Pixel` type — or a `Pixel`-bounded generic — everywhere else.

use bytemuck::{Pod, Zeroable};

use crate::mode::PixelMode;

// ── Compile-time size assertions ──────────────────────────────────────────────
//
// Each assertion fires at compile time if `BYTES` disagrees with the actual
// struct size. This catches padding surprises that the runtime test would only
// catch after the binary is built.

const _: () = assert!(std::mem::size_of::<Rgb8>() == Rgb8::BYTES);
const _: () = assert!(std::mem::size_of::<Rgba8>() == Rgba8::BYTES);
const _: () = assert!(std::mem::size_of::<Gray8>() == Gray8::BYTES);
const _: () = assert!(std::mem::size_of::<Cmyk8>() == Cmyk8::BYTES);
const _: () = assert!(std::mem::size_of::<DeviceN8>() == DeviceN8::BYTES);

// ── Pixel trait ───────────────────────────────────────────────────────────────

/// A typed pixel value that can be stored in a `Bitmap<P>` row buffer.
///
/// All implementations are `Copy + Pod`, enabling zero-copy row access via
/// `bytemuck::cast_slice`. `BYTES` must match `std::mem::size_of::<Self>()`;
/// compile-time assertions in this module enforce the invariant.
pub trait Pixel: Copy + Pod + Zeroable + Send + Sync + 'static {
    /// The [`PixelMode`] variant that identifies this pixel format at runtime.
    const MODE: PixelMode;
    /// Byte width of one pixel. Must equal `std::mem::size_of::<Self>()`.
    const BYTES: usize;
}

// ── Concrete pixel types ──────────────────────────────────────────────────────

/// 8-bit RGB, 3 bytes/pixel, wire layout `[R, G, B]`.
///
/// The most common rasterizer output format, matching `SplashModRGB8` in the
/// C++ side.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default, Pod, Zeroable)]
pub struct Rgb8 {
    /// Red channel, `0` = minimum, `255` = full intensity.
    pub r: u8,
    /// Green channel, `0` = minimum, `255` = full intensity.
    pub g: u8,
    /// Blue channel, `0` = minimum, `255` = full intensity.
    pub b: u8,
}

impl Pixel for Rgb8 {
    const MODE: PixelMode = PixelMode::Rgb8;
    const BYTES: usize = 3;
}

/// 8-bit RGBA, 4 bytes/pixel, wire layout `[R, G, B, A]`.
///
/// This is the working format for transparency groups. The `MODE` constant is
/// set to [`PixelMode::Xbgr8`] as an intentional approximation: the rasterizer
/// internally uses this struct for transparency groups and the mode field is
/// only used for external dispatch (e.g. choosing a blitter). The actual byte
/// layout is `[R, G, B, A]`, **not** `[X, B, G, R]`; callers that perform
/// memory-layout-sensitive operations must use the struct fields directly
/// rather than relying on the `MODE` variant.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default, Pod, Zeroable)]
pub struct Rgba8 {
    /// Red channel, `0` = minimum, `255` = full intensity.
    pub r: u8,
    /// Green channel, `0` = minimum, `255` = full intensity.
    pub g: u8,
    /// Blue channel, `0` = minimum, `255` = full intensity.
    pub b: u8,
    /// Alpha channel, `0` = fully transparent, `255` = fully opaque.
    pub a: u8,
}

impl Pixel for Rgba8 {
    // Intentional approximation — see struct doc comment above.
    const MODE: PixelMode = PixelMode::Xbgr8;
    const BYTES: usize = 4;
}

/// 8-bit grayscale, 1 byte/pixel, wire layout `[Y]`.
///
/// Used for `-gray` output. RGB→luminance uses BT.709 coefficients.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default, Pod, Zeroable)]
pub struct Gray8 {
    /// Luminance, `0` = black, `255` = white.
    pub v: u8,
}

impl Pixel for Gray8 {
    const MODE: PixelMode = PixelMode::Mono8;
    const BYTES: usize = 1;
}

/// 8-bit CMYK, 4 bytes/pixel, wire layout `[C, M, Y, K]`.
///
/// Used for `-jpegcmyk` and overprint modes.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default, Pod, Zeroable)]
pub struct Cmyk8 {
    /// Cyan ink, `0` = no ink, `255` = full coverage.
    pub c: u8,
    /// Magenta ink, `0` = no ink, `255` = full coverage.
    pub m: u8,
    /// Yellow ink, `0` = no ink, `255` = full coverage.
    pub y: u8,
    /// Black (key) ink, `0` = no ink, `255` = full coverage.
    pub k: u8,
}

impl Pixel for Cmyk8 {
    const MODE: PixelMode = PixelMode::Cmyk8;
    const BYTES: usize = 4;
}

/// CMYK + 4 spot channels, 8 bytes/pixel, wire layout `[C, M, Y, K, S0, S1, S2, S3]`.
///
/// Used with `-overprint`. `SPOT_NCOMPS = 4` is fixed at compile time,
/// matching the C++ default.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default, Pod, Zeroable)]
pub struct DeviceN8 {
    /// Process CMYK channels.
    pub cmyk: Cmyk8,
    /// Spot channels `S0`–`S3`, `0` = no ink, `255` = full coverage.
    pub spots: [u8; 4],
}

impl Pixel for DeviceN8 {
    const MODE: PixelMode = PixelMode::DeviceN8;
    const BYTES: usize = 8;
}

// ── Erased pixel buffer ───────────────────────────────────────────────────────

/// A mode-erased pixel value carrying up to 8 bytes (matching `SplashColor`).
///
/// # When to use `AnyColor` vs a concrete `Pixel` type
///
/// Prefer a concrete `impl Pixel` type — or a `Pixel`-bounded generic — in
/// every performance-sensitive path; the monomorphized code paths avoid runtime
/// dispatch. Use `AnyColor` only in the small number of places that must handle
/// **all** modes at runtime without monomorphizing the entire call stack: paper
/// colour, graphics-state default colour, and similar configuration values.
#[derive(Copy, Clone, Debug, Default)]
pub struct AnyColor {
    /// Raw pixel bytes; only the first `mode.bytes_per_pixel()` entries are meaningful.
    pub bytes: [u8; 8],
    /// The pixel format that determines how `bytes` should be interpreted.
    pub mode: PixelMode,
}

impl AnyColor {
    /// Return the black (zero-ink / zero-intensity) colour for `mode`.
    #[must_use]
    pub const fn black(mode: PixelMode) -> Self {
        Self {
            bytes: [0; 8],
            mode,
        }
    }

    /// Return the white colour for `mode`.
    ///
    /// # Per-mode encoding
    ///
    /// | Mode | White encoding |
    /// |------|----------------|
    /// | `Mono1` | `bytes[0] = 0xFF` — all 8 bits set = all pixels white (MSB-first packed format) |
    /// | `Mono8` | `bytes[0] = 255` |
    /// | `Rgb8`, `Bgr8` | `bytes[0..3] = [255, 255, 255]` |
    /// | `Xbgr8` | `bytes[0..4] = [255, 255, 255, 255]` — `byte[3]` is the ignored X/padding byte, set to 255 for consistency so the full 4-byte value reads as opaque white in any RGBA interpretation |
    /// | `Cmyk8`, `DeviceN8` | all bytes zero — CMYK white is zero ink on all channels |
    #[must_use]
    pub const fn white(mode: PixelMode) -> Self {
        let mut bytes = [0u8; 8];
        match mode {
            // Mono1: 0xFF means all 8 packed bits = 1 = white (MSB-first format).
            // Mono8: 255 = maximum luminance = white.
            PixelMode::Mono1 | PixelMode::Mono8 => bytes[0] = 255,
            PixelMode::Rgb8 | PixelMode::Bgr8 => {
                bytes[0] = 255;
                bytes[1] = 255;
                bytes[2] = 255;
            }
            PixelMode::Xbgr8 => {
                bytes[0] = 255;
                bytes[1] = 255;
                bytes[2] = 255;
                // byte[3] is the X (ignored/padding) byte in XBGR. Setting it
                // to 255 ensures the 4-byte word reads as fully-opaque white
                // when interpreted as any RGBA variant, and avoids leaving
                // uninitialised-looking padding in the output buffer.
                bytes[3] = 255;
            }
            // CMYK/DeviceN white = no ink on any channel = all zeros.
            PixelMode::Cmyk8 | PixelMode::DeviceN8 => {}
        }
        Self { bytes, mode }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The compile-time assertions at the top of the module already enforce
    // BYTES == size_of, but the runtime test provides a readable failure message
    // during `cargo test` in case someone adds a new type and forgets the const.
    #[test]
    fn sizes_match_bytes_const() {
        assert_eq!(std::mem::size_of::<Rgb8>(), Rgb8::BYTES);
        assert_eq!(std::mem::size_of::<Rgba8>(), Rgba8::BYTES);
        assert_eq!(std::mem::size_of::<Gray8>(), Gray8::BYTES);
        assert_eq!(std::mem::size_of::<Cmyk8>(), Cmyk8::BYTES);
        assert_eq!(std::mem::size_of::<DeviceN8>(), DeviceN8::BYTES);
    }

    #[test]
    fn cmyk8_black() {
        let px = Cmyk8 {
            c: 0,
            m: 0,
            y: 0,
            k: 255,
        };
        let (r, g, b) = crate::convert::cmyk_to_rgb(px.c, px.m, px.y, px.k);
        assert_eq!((r, g, b), (0, 0, 0));
    }

    /// White for `Mono1` must be `0xFF` — all 8 packed bits set to 1.
    #[test]
    fn any_color_white_mono1_is_all_bits_set() {
        let w = AnyColor::white(PixelMode::Mono1);
        assert_eq!(w.bytes[0], 0xFF, "Mono1 white must be 0xFF (all bits = 1)");
    }

    /// White for `Xbgr8` must set all four bytes including the padding X byte.
    #[test]
    fn any_color_white_xbgr8_sets_padding_byte() {
        let w = AnyColor::white(PixelMode::Xbgr8);
        assert_eq!(w.bytes, [255, 255, 255, 255, 0, 0, 0, 0]);
    }

    /// CMYK/DeviceN white is zero ink on all channels.
    #[test]
    fn any_color_white_cmyk_is_zero() {
        let w = AnyColor::white(PixelMode::Cmyk8);
        assert_eq!(w.bytes, [0u8; 8]);
    }
}
