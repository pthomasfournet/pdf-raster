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

// ── Shared conversion helpers ─────────────────────────────────────────────────

/// Convert a linear `f32` channel value in `[0.0, 1.0]` to a rounded `u8`.
///
/// Pipeline:
/// 1. Clamp to `[0.0, 1.0]` — guards against small floating-point overshoot.
/// 2. Scale to `[0.0, 255.0]` and add a 0.5 rounding bias via `mul_add`.
/// 3. Clamp again to `[0.0, 255.0]` — eliminates any NaN (NaN comparisons
///    fail, so `f32::clamp(NaN, lo, hi) = NaN`; a second clamp with the same
///    bounds still returns NaN, but step 4 then maps it to 0).
/// 4. Cast `f32 → u32` (saturating since Rust 1.45: NaN → 0, out-of-range
///    → saturated). After step 3 the value is in `[0.0, 255.0]`, so the u32
///    result is in `[0, 255]`.
/// 5. Cast `u32 → u8` via `try_from` (infallible: value ≤ 255).
#[must_use]
#[inline]
const fn f32_to_u8(f: f32) -> u8 {
    // Clippy::cast_possible_truncation is a known false-positive for provably-
    // bounded float→int casts (github.com/rust-lang/rust-clippy/issues/7486).
    // The preceding clamp guarantees the value is in [0.5, 255.5]; truncation
    // to u8 is therefore safe. #[expect] (not #[allow]) errors if the lint
    // is ever fixed upstream, keeping this annotation honest.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "value is clamped to [0.0, 255.0] before cast; truncation and sign-loss are impossible"
    )]
    {
        f.clamp(0.0, 1.0).mul_add(255.0, 0.5).clamp(0.0, 255.0) as u8
    }
}

/// Compute one CMY channel from an RGB channel given the black key `k` and the
/// denominator `dk = 255 - k`.
///
/// Returns 0 when `dk == 0` (pure black, all CMY channels are zero) to avoid
/// an integer division by zero.
#[must_use]
#[inline]
fn rgb_to_cmyk_channel(channel: u8, black: u8, dk: u8) -> u8 {
    if dk == 0 {
        0
    } else {
        // After `.min(255)` the value is guaranteed ≤ 255, so `as u8` is exact.
        (255u32.saturating_sub(u32::from(channel) + u32::from(black)) * 255 / u32::from(dk))
            .min(255) as u8
    }
}

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

    /// Convert to linear RGBA f32, with alpha = 1.0 unless the type carries alpha.
    #[must_use]
    fn to_rgba_f32(self) -> [f32; 4];

    /// Convert from linear RGBA f32. Clamps to [0, 1] before quantising.
    #[must_use]
    fn from_rgba_f32(v: [f32; 4]) -> Self;
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

    fn to_rgba_f32(self) -> [f32; 4] {
        [
            f32::from(self.r) / 255.0,
            f32::from(self.g) / 255.0,
            f32::from(self.b) / 255.0,
            1.0,
        ]
    }

    fn from_rgba_f32(v: [f32; 4]) -> Self {
        Self {
            r: f32_to_u8(v[0]),
            g: f32_to_u8(v[1]),
            b: f32_to_u8(v[2]),
        }
    }
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

    fn to_rgba_f32(self) -> [f32; 4] {
        [
            f32::from(self.r) / 255.0,
            f32::from(self.g) / 255.0,
            f32::from(self.b) / 255.0,
            f32::from(self.a) / 255.0,
        ]
    }

    fn from_rgba_f32(v: [f32; 4]) -> Self {
        Self {
            r: f32_to_u8(v[0]),
            g: f32_to_u8(v[1]),
            b: f32_to_u8(v[2]),
            a: f32_to_u8(v[3]),
        }
    }
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

    fn to_rgba_f32(self) -> [f32; 4] {
        let f = f32::from(self.v) / 255.0;
        [f, f, f, 1.0]
    }

    fn from_rgba_f32(v: [f32; 4]) -> Self {
        // BT.709 luminance: Y = 0.2126·R + 0.7152·G + 0.0722·B.
        //
        // Written with `mul_add` for fused multiply-add precision:
        //   inner = 0.2126·R + 0.7152·G   (via mul_add)
        //   lum   = 0.0722·B + inner       (via mul_add)
        // which is mathematically identical to the standard formula.
        let lum = 0.0722_f32
            .mul_add(v[2], 0.2126_f32.mul_add(v[0], 0.7152 * v[1]))
            .clamp(0.0, 1.0);
        Self { v: f32_to_u8(lum) }
    }
}

/// 8-bit CMYK, 4 bytes/pixel, wire layout `[C, M, Y, K]`.
///
/// Used for `-jpegcmyk` and overprint modes.
///
/// # RGB→CMYK conversion
///
/// [`from_rgba_f32`](Pixel::from_rgba_f32) uses a simple **UCR (Under Colour
/// Removal)** model: `K = 255 − max(R, G, B)`. This is **not ICC-correct**;
/// a real ICC profile would apply a device-specific tone reproduction curve
/// and total ink limit. Use this type only for rasterizer-internal
/// intermediate storage or simple CMYK approximation, not for production
/// colour-managed output.
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

    fn to_rgba_f32(self) -> [f32; 4] {
        let (red, green, blue) = crate::convert::cmyk_to_rgb(self.c, self.m, self.y, self.k);
        [
            f32::from(red) / 255.0,
            f32::from(green) / 255.0,
            f32::from(blue) / 255.0,
            1.0,
        ]
    }

    fn from_rgba_f32(v: [f32; 4]) -> Self {
        let red = f32_to_u8(v[0]);
        let green = f32_to_u8(v[1]);
        let blue = f32_to_u8(v[2]);
        // Simple UCR black generation: K = 255 − max(R, G, B).
        // See struct-level doc comment for the limitations of this model.
        let black = 255u8.saturating_sub(red.max(green).max(blue));
        let dk = 255u8.saturating_sub(black);
        Self {
            c: rgb_to_cmyk_channel(red, black, dk),
            m: rgb_to_cmyk_channel(green, black, dk),
            y: rgb_to_cmyk_channel(blue, black, dk),
            k: black,
        }
    }
}

/// CMYK + 4 spot channels, 8 bytes/pixel, wire layout `[C, M, Y, K, S0, S1, S2, S3]`.
///
/// Used with `-overprint`. `SPOT_NCOMPS = 4` is fixed at compile time,
/// matching the C++ default.
///
/// # Delegation to `Cmyk8`
///
/// `to_rgba_f32` delegates directly to `Cmyk8::to_rgba_f32` on the `cmyk`
/// sub-field; spot channels have no representation in RGBA and are ignored.
///
/// `from_rgba_f32` delegates to `Cmyk8::from_rgba_f32` for the CMYK portion
/// and zero-initialises all four spot channels, because the RGBA working space
/// carries no spot-colour information.
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

    fn to_rgba_f32(self) -> [f32; 4] {
        // Delegate to Cmyk8; spot channels have no RGBA equivalent.
        self.cmyk.to_rgba_f32()
    }

    fn from_rgba_f32(v: [f32; 4]) -> Self {
        Self {
            cmyk: Cmyk8::from_rgba_f32(v),
            // Spot channels are zero-initialised: RGBA carries no spot
            // colour information.
            spots: [0; 4],
        }
    }
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
    fn rgb8_roundtrip() {
        let px = Rgb8 {
            r: 123,
            g: 45,
            b: 200,
        };
        let f = px.to_rgba_f32();
        let back = Rgb8::from_rgba_f32(f);
        assert_eq!(back, px);
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

    /// `f32_to_u8` must return 0 for NaN inputs (explicit NaN guard in the helper).
    #[test]
    fn f32_to_u8_nan_gives_zero() {
        assert_eq!(f32_to_u8(f32::NAN), 0);
    }

    /// `f32_to_u8` must clamp values above 1.0 to 255.
    #[test]
    fn f32_to_u8_overflow_clamps() {
        assert_eq!(f32_to_u8(2.0), 255);
    }

    /// `f32_to_u8` must clamp values below 0.0 to 0.
    #[test]
    fn f32_to_u8_underflow_clamps() {
        assert_eq!(f32_to_u8(-1.0), 0);
    }

    /// BT.709 luminance: pure green (0, 1, 0) → ~182 (= round(0.7152 × 255)).
    #[test]
    fn gray8_bt709_green() {
        let px = Gray8::from_rgba_f32([0.0, 1.0, 0.0, 1.0]);
        // 0.7152 × 255 + 0.5 = 182.676 → 182
        assert_eq!(px.v, 182);
    }

    /// `DeviceN8` spot channels are zeroed when converting from RGBA.
    #[test]
    fn device_n8_spots_are_zero_after_from_rgba() {
        let px = DeviceN8::from_rgba_f32([1.0, 0.0, 0.0, 1.0]);
        assert_eq!(px.spots, [0u8; 4]);
    }
}
