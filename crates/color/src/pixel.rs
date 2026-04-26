//! Pixel types and the `Pixel` trait.
//!
//! Each concrete type is a `#[repr(C)]` struct that implements `bytemuck::Pod`,
//! allowing zero-copy casts between `&[u8]` row buffers and typed pixel slices.
//!
//! The `Pixel` trait is the generic bound used by `Bitmap<P>` and the rasterizer
//! pipeline. Monomorphization over `P` is how we avoid runtime mode dispatch in
//! hot loops — the compiler generates one code path per pixel format.

use bytemuck::{Pod, Zeroable};

use crate::mode::PixelMode;

// ── Pixel trait ───────────────────────────────────────────────────────────────

/// A typed pixel value that can be stored in a `Bitmap<P>` row buffer.
///
/// All implementations are `Copy + Pod`, enabling zero-copy row access via
/// `bytemuck::cast_slice`. `BYTES` must match `std::mem::size_of::<Self>()`.
pub trait Pixel: Copy + Pod + Zeroable + Send + Sync + 'static {
    const MODE: PixelMode;
    const BYTES: usize;

    /// Convert to linear RGBA f32, with alpha = 1.0 unless the type carries alpha.
    fn to_rgba_f32(self) -> [f32; 4];

    /// Convert from linear RGBA f32. Clamps to [0, 1] before quantising.
    fn from_rgba_f32(v: [f32; 4]) -> Self;
}

// ── Concrete pixel types ──────────────────────────────────────────────────────

/// 8-bit RGB, 3 bytes/pixel. The most common rasterizer output format.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default, Pod, Zeroable)]
pub struct Rgb8 {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

impl Pixel for Rgb8 {
    const MODE: PixelMode = PixelMode::Rgb8;
    const BYTES: usize = 3;

    fn to_rgba_f32(self) -> [f32; 4] {
        [
            self.r as f32 / 255.0,
            self.g as f32 / 255.0,
            self.b as f32 / 255.0,
            1.0,
        ]
    }

    fn from_rgba_f32(v: [f32; 4]) -> Self {
        Self {
            r: (v[0].clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
            g: (v[1].clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
            b: (v[2].clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
        }
    }
}

/// 8-bit RGBA, 4 bytes/pixel. Working format for transparency groups.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default, Pod, Zeroable)]
pub struct Rgba8 {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl Pixel for Rgba8 {
    const MODE: PixelMode = PixelMode::Xbgr8; // closest mode; groups use this internally
    const BYTES: usize = 4;

    fn to_rgba_f32(self) -> [f32; 4] {
        [
            self.r as f32 / 255.0,
            self.g as f32 / 255.0,
            self.b as f32 / 255.0,
            self.a as f32 / 255.0,
        ]
    }

    fn from_rgba_f32(v: [f32; 4]) -> Self {
        Self {
            r: (v[0].clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
            g: (v[1].clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
            b: (v[2].clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
            a: (v[3].clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
        }
    }
}

/// 8-bit grayscale, 1 byte/pixel. Used for `-gray` output mode.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default, Pod, Zeroable)]
pub struct Gray8 {
    pub v: u8,
}

impl Pixel for Gray8 {
    const MODE: PixelMode = PixelMode::Mono8;
    const BYTES: usize = 1;

    fn to_rgba_f32(self) -> [f32; 4] {
        let f = self.v as f32 / 255.0;
        [f, f, f, 1.0]
    }

    fn from_rgba_f32(v: [f32; 4]) -> Self {
        let lum = (0.2126 * v[0] + 0.7152 * v[1] + 0.0722 * v[2]).clamp(0.0, 1.0);
        Self {
            v: (lum * 255.0 + 0.5) as u8,
        }
    }
}

/// 8-bit CMYK, 4 bytes/pixel. Used for `-jpegcmyk` and overprint modes.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default, Pod, Zeroable)]
pub struct Cmyk8 {
    pub c: u8,
    pub m: u8,
    pub y: u8,
    pub k: u8,
}

impl Pixel for Cmyk8 {
    const MODE: PixelMode = PixelMode::Cmyk8;
    const BYTES: usize = 4;

    fn to_rgba_f32(self) -> [f32; 4] {
        let (r, g, b) = crate::convert::cmyk_to_rgb(self.c, self.m, self.y, self.k);
        [r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0, 1.0]
    }

    fn from_rgba_f32(v: [f32; 4]) -> Self {
        let r = (v[0].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        let g = (v[1].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        let b = (v[2].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        let k = 255u8.saturating_sub(r.max(g).max(b));
        let dk = 255u8.saturating_sub(k);
        let c = if dk == 0 {
            0
        } else {
            (255u32.saturating_sub(r as u32 + k as u32) * 255 / dk as u32).min(255) as u8
        };
        let m = if dk == 0 {
            0
        } else {
            (255u32.saturating_sub(g as u32 + k as u32) * 255 / dk as u32).min(255) as u8
        };
        let y = if dk == 0 {
            0
        } else {
            (255u32.saturating_sub(b as u32 + k as u32) * 255 / dk as u32).min(255) as u8
        };
        Self { c, m, y, k }
    }
}

/// CMYK + 4 spot channels, 8 bytes/pixel. Used with `-overprint`.
///
/// `SPOT_NCOMPS = 4` is fixed at compile time (matching the C++ default).
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default, Pod, Zeroable)]
pub struct DeviceN8 {
    pub cmyk: Cmyk8,
    pub spots: [u8; 4],
}

impl Pixel for DeviceN8 {
    const MODE: PixelMode = PixelMode::DeviceN8;
    const BYTES: usize = 8;

    fn to_rgba_f32(self) -> [f32; 4] {
        self.cmyk.to_rgba_f32()
    }

    fn from_rgba_f32(v: [f32; 4]) -> Self {
        Self {
            cmyk: Cmyk8::from_rgba_f32(v),
            spots: [0; 4],
        }
    }
}

// ── Erased pixel buffer ───────────────────────────────────────────────────────

/// A mode-erased pixel value carrying up to 8 bytes (matching `SplashColor`).
///
/// Used in contexts that must handle all modes at runtime (e.g. paper colour,
/// graphics state default colour) without monomorphizing the entire call stack.
#[derive(Copy, Clone, Debug, Default)]
pub struct AnyColor {
    pub bytes: [u8; 8],
    pub mode: PixelMode,
}

impl AnyColor {
    pub fn black(mode: PixelMode) -> Self {
        Self {
            bytes: [0; 8],
            mode,
        }
    }

    pub fn white(mode: PixelMode) -> Self {
        let mut bytes = [0u8; 8];
        match mode {
            PixelMode::Mono8 => bytes[0] = 255,
            PixelMode::Rgb8 | PixelMode::Bgr8 => {
                bytes[0] = 255;
                bytes[1] = 255;
                bytes[2] = 255;
            }
            PixelMode::Xbgr8 => {
                bytes[0] = 255;
                bytes[1] = 255;
                bytes[2] = 255;
                bytes[3] = 255;
            }
            // CMYK white = no ink
            PixelMode::Cmyk8 | PixelMode::DeviceN8 => {}
            PixelMode::Mono1 => bytes[0] = 0xff,
        }
        Self { bytes, mode }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
