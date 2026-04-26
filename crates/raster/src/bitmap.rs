//! Pixel buffers and the anti-aliasing scratch buffer.
//!
//! [`Bitmap<P>`] replaces `SplashBitmap` with a clean, always-top-down,
//! typed-generic alternative. Key differences from the C++ original:
//! - `rowSize` is always positive (no bottom-up layout).
//! - Typed row access via [`bytemuck::cast_slice`] — zero-copy, no mode switch.
//! - Alpha is an `Option<Vec<u8>>` separate plane (matching C++ layout).
//!
//! [`AaBuf`] is the 1-bit supersampled scratch buffer used by the AA fill
//! path. It is 4 rows × (`bitmap_width` × 4) columns, MSB-first packed.

use std::marker::PhantomData;

use crate::types::{AA_SIZE, Pixel};

// ── internal helper ───────────────────────────────────────────────────────────

/// Convert a `u32` to `usize`.
///
/// On every platform where Rust runs, `usize` is at least 32 bits wide, so
/// this conversion is always exact. The compile-time `const` assertion below
/// enforces that invariant; any platform that violates it will fail to
/// compile rather than producing silently truncated values.
#[inline]
const fn u32_to_usize(v: u32) -> usize {
    const { assert!(u32::BITS <= usize::BITS, "platform has a narrower usize than u32") }
    v as usize
}

// ── Bitmap ────────────────────────────────────────────────────────────────────

/// A typed, top-down pixel buffer.
///
/// `P` determines the in-memory layout. The row stride is `width * P::BYTES`
/// rounded up to the next multiple of `row_pad` (default 4, matching the C++
/// `bitmapRowPad = 4` in `pdftoppm.cc`).
///
/// The optional alpha plane is always `width * height` bytes, one byte per pixel,
/// top-down. It is stored separately from the colour data (matching `SplashBitmap`).
pub struct Bitmap<P: Pixel> {
    /// Width of the bitmap in pixels.
    pub width: u32,
    /// Height of the bitmap in pixels.
    pub height: u32,
    /// Byte distance between the start of consecutive rows; always `≥ width * P::BYTES`
    /// and a multiple of `row_pad`.
    pub stride: usize,
    data: Vec<u8>,
    alpha: Option<Vec<u8>>,
    _marker: PhantomData<P>,
}

impl<P: Pixel> Bitmap<P> {
    /// Allocate a new zeroed bitmap.
    ///
    /// `row_pad` pads each row to a multiple of that many bytes (pass 1 for no padding).
    /// `with_alpha` allocates a separate alpha plane initialised to 0.
    ///
    /// # Panics
    ///
    /// Panics if `row_pad` is 0.
    #[must_use]
    pub fn new(width: u32, height: u32, row_pad: usize, with_alpha: bool) -> Self {
        assert!(row_pad >= 1, "row_pad must be ≥ 1");
        let raw_stride = u32_to_usize(width) * P::BYTES;
        let stride = if row_pad <= 1 {
            raw_stride
        } else {
            raw_stride.div_ceil(row_pad) * row_pad
        };
        let data = vec![0u8; stride * u32_to_usize(height)];
        let alpha = if with_alpha {
            Some(vec![0u8; u32_to_usize(width) * u32_to_usize(height)])
        } else {
            None
        };
        Self {
            width,
            height,
            stride,
            data,
            alpha,
            _marker: PhantomData,
        }
    }

    // ── Row access ────────────────────────────────────────────────────────────

    /// Typed read-only access to row `y`.
    ///
    /// Returns a slice of exactly `width` pixels.
    ///
    /// # Panics
    ///
    /// Panics if `y >= height`.
    #[must_use]
    pub fn row(&self, y: u32) -> &[P] {
        let bytes = self.row_bytes(y);
        bytemuck::cast_slice(&bytes[..u32_to_usize(self.width) * P::BYTES])
    }

    /// Typed mutable access to row `y`.
    ///
    /// # Panics
    ///
    /// Panics if `y >= height`.
    pub fn row_mut(&mut self, y: u32) -> &mut [P] {
        let w = u32_to_usize(self.width) * P::BYTES;
        let bytes = self.row_bytes_mut(y);
        bytemuck::cast_slice_mut(&mut bytes[..w])
    }

    /// Raw byte read-only access to the full stride of row `y` (including padding).
    ///
    /// # Panics
    ///
    /// Panics if `y >= height`.
    #[must_use]
    pub fn row_bytes(&self, y: u32) -> &[u8] {
        assert!(
            y < self.height,
            "row index {y} is out of bounds for bitmap height {}",
            self.height
        );
        let off = u32_to_usize(y) * self.stride;
        &self.data[off..off + self.stride]
    }

    /// Raw byte mutable access to the full stride of row `y`.
    ///
    /// # Panics
    ///
    /// Panics if `y >= height`.
    pub fn row_bytes_mut(&mut self, y: u32) -> &mut [u8] {
        assert!(
            y < self.height,
            "row index {y} is out of bounds for bitmap height {}",
            self.height
        );
        let off = u32_to_usize(y) * self.stride;
        &mut self.data[off..off + self.stride]
    }

    /// Raw pointer to the start of row `y`. For SIMD inner loops.
    ///
    /// # Panics
    ///
    /// Panics if `y >= height`.
    #[must_use]
    pub fn row_ptr(&self, y: u32) -> *const u8 {
        self.row_bytes(y).as_ptr()
    }

    /// Mutable raw pointer to the start of row `y`. For SIMD inner loops.
    ///
    /// # Panics
    ///
    /// Panics if `y >= height`.
    pub fn row_ptr_mut(&mut self, y: u32) -> *mut u8 {
        self.row_bytes_mut(y).as_mut_ptr()
    }

    // ── Alpha access ──────────────────────────────────────────────────────────

    /// Alpha plane row `y`, if the alpha plane was allocated.
    ///
    /// Returns `None` if this bitmap was created without an alpha plane.
    ///
    /// # Panics
    ///
    /// Panics if `y >= height`.
    #[must_use]
    pub fn alpha_row(&self, y: u32) -> Option<&[u8]> {
        assert!(
            y < self.height,
            "row index {y} is out of bounds for bitmap height {}",
            self.height
        );
        self.alpha.as_ref().map(|a| {
            let w = u32_to_usize(self.width);
            let off = u32_to_usize(y) * w;
            &a[off..off + w]
        })
    }

    /// Mutable alpha plane row `y`.
    ///
    /// Returns `None` if this bitmap was created without an alpha plane.
    ///
    /// # Panics
    ///
    /// Panics if `y >= height`.
    pub fn alpha_row_mut(&mut self, y: u32) -> Option<&mut [u8]> {
        assert!(
            y < self.height,
            "row index {y} is out of bounds for bitmap height {}",
            self.height
        );
        let w = u32_to_usize(self.width);
        self.alpha.as_mut().map(|a| {
            let off = u32_to_usize(y) * w;
            &mut a[off..off + w]
        })
    }

    /// Returns `true` if this bitmap has a separate alpha plane.
    #[must_use]
    pub const fn has_alpha(&self) -> bool {
        self.alpha.is_some()
    }

    // ── Bulk operations ───────────────────────────────────────────────────────

    /// Fill the entire bitmap with a solid colour and an alpha value.
    ///
    /// `alpha` is ignored if the bitmap has no alpha plane.
    pub fn clear(&mut self, color: P, alpha: u8) {
        let pixel_bytes: &[u8] = bytemuck::bytes_of(&color);
        if P::BYTES == 1 {
            // Optimise single-byte pixels: memset the entire data buffer.
            self.data.fill(pixel_bytes[0]);
        } else {
            // Write one pixel-sized chunk per pixel in each row, then zero the
            // stride padding. Using `chunks_exact_mut` avoids manual index arithmetic.
            let w = u32_to_usize(self.width);
            let n = w * P::BYTES;
            let stride = self.stride;
            for row in self.data.chunks_exact_mut(stride) {
                for dst in row[..n].chunks_exact_mut(P::BYTES) {
                    dst.copy_from_slice(pixel_bytes);
                }
                row[n..].fill(0);
            }
        }
        if let Some(ref mut a) = self.alpha {
            a.fill(alpha);
        }
    }

    /// Total byte size of the colour data buffer.
    #[must_use]
    pub const fn data_len(&self) -> usize {
        self.data.len()
    }

    /// Access the full colour data buffer as a flat byte slice.
    #[must_use]
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Mutable access to the full colour data buffer as a flat byte slice.
    pub fn data_mut(&mut self) -> &mut [u8] {
        &mut self.data
    }
}

// ── Type-erased bitmap ────────────────────────────────────────────────────────

/// A mode-erased bitmap, used when the transparency group stack must store
/// bitmaps of varying pixel types without monomorphizing the entire stack.
///
/// Phase 2 will add rendering methods; Phase 1 only needs storage + dimensions.
pub enum AnyBitmap {
    /// 24-bit RGB, 8 bits per channel.
    Rgb8(Bitmap<crate::types::Rgb8>),
    /// 32-bit RGBA, 8 bits per channel (alpha stored in a separate plane).
    Rgba8(Bitmap<crate::types::Rgba8>),
    /// 8-bit grayscale.
    Gray8(Bitmap<crate::types::Gray8>),
    /// 32-bit CMYK, 8 bits per channel.
    Cmyk8(Bitmap<crate::types::Cmyk8>),
    /// Up to 8 spot/device-N channels, 8 bits each.
    DeviceN8(Bitmap<crate::types::DeviceN8>),
}

impl AnyBitmap {
    /// Width of the contained bitmap in pixels.
    #[must_use]
    pub const fn width(&self) -> u32 {
        match self {
            Self::Rgb8(b) => b.width,
            Self::Rgba8(b) => b.width,
            Self::Gray8(b) => b.width,
            Self::Cmyk8(b) => b.width,
            Self::DeviceN8(b) => b.width,
        }
    }

    /// Height of the contained bitmap in pixels.
    #[must_use]
    pub const fn height(&self) -> u32 {
        match self {
            Self::Rgb8(b) => b.height,
            Self::Rgba8(b) => b.height,
            Self::Gray8(b) => b.height,
            Self::Cmyk8(b) => b.height,
            Self::DeviceN8(b) => b.height,
        }
    }

    /// The pixel mode (colour space + bit depth) of the contained bitmap.
    #[must_use]
    pub const fn mode(&self) -> crate::types::PixelMode {
        match self {
            Self::Rgb8(_) => crate::types::PixelMode::Rgb8,
            Self::Rgba8(_) => crate::types::PixelMode::Xbgr8,
            Self::Gray8(_) => crate::types::PixelMode::Mono8,
            Self::Cmyk8(_) => crate::types::PixelMode::Cmyk8,
            Self::DeviceN8(_) => crate::types::PixelMode::DeviceN8,
        }
    }
}

// ── AaBuf ─────────────────────────────────────────────────────────────────────

/// Anti-aliasing scratch buffer: 4 rows × (`bitmap_width` × 4) columns, 1 bit/pixel.
///
/// Bit packing is MSB-first (matching `SplashBitmap` `Mono1` layout and the
/// `renderAALine` / `clipAALine` code in `SplashXPathScanner.cc`):
/// - Bit 7 of byte 0 is pixel 0 of the row.
/// - Left-edge mask for pixel x: `0xff >> (x & 7)`.
/// - Right-edge mask for pixel x1 (exclusive): `(0xff00u16 >> (x1 & 7)) as u8`.
///
/// `height` is always [`AA_SIZE`] (4). `width` is `bitmap_width * AA_SIZE`.
pub struct AaBuf {
    /// Width in pixels (= `bitmap_width` × `AA_SIZE`).
    pub width: usize,
    /// Always [`AA_SIZE`] (4).
    pub height: usize,
    data: Vec<u8>, // row-major; rows are (width+7)/8 bytes each
}

/// `AA_SIZE` converted to `usize`. Computed once as a compile-time constant.
///
/// `AA_SIZE` is `i32 = 4`, which is always non-negative, so the conversion
/// can never fail; the `expect` ensures any future change to a negative value
/// is caught immediately.
const AA_SIZE_USIZE: usize = {
    // i32 → usize: can only fail if AA_SIZE is negative.
    assert!(AA_SIZE >= 0, "AA_SIZE must be non-negative");
    AA_SIZE as usize
};

impl AaBuf {
    /// Create a new zeroed AA buffer for a bitmap of the given pixel width.
    #[must_use]
    pub fn new(bitmap_width: usize) -> Self {
        let width = bitmap_width * AA_SIZE_USIZE;
        let height = AA_SIZE_USIZE;
        let row_bytes = width.div_ceil(8);
        Self {
            width,
            height,
            data: vec![0u8; row_bytes * height],
        }
    }

    /// Bytes per row (= `width.div_ceil(8)`).
    #[inline]
    #[must_use]
    pub const fn row_bytes(&self) -> usize {
        self.width.div_ceil(8)
    }

    /// Clear all bits to 0.
    #[inline]
    pub fn clear(&mut self) {
        self.data.fill(0);
    }

    /// Set pixels `[x0, x1)` to 1 in the given row.
    ///
    /// `x0` and `x1` are clamped to `[0, width]`. No-op if `x0 ≥ x1`.
    ///
    /// # Panics
    ///
    /// Panics if `row >= height`.
    pub fn set_span(&mut self, row: usize, x0: usize, x1: usize) {
        assert!(
            row < self.height,
            "row index {row} is out of bounds for AaBuf height {}",
            self.height
        );
        let x0 = x0.min(self.width);
        let x1 = x1.min(self.width);
        if x0 >= x1 {
            return;
        }
        let rb = self.row_bytes();
        let base = row * rb;
        let b0 = x0 >> 3;
        let b1 = (x1 - 1) >> 3;
        if b0 == b1 {
            // Span is entirely within one byte.
            // Left mask: top x0%8 bits clear, rest set.
            // Right mask: top x1%8 bits set (or all bits if byte-aligned).
            let left_mask = 0xff_u8 >> (x0 & 7);
            let right_shift = x1 & 7;
            let right_mask = if right_shift == 0 {
                0xff_u8
            } else {
                !(0xff_u8 >> right_shift)
            };
            self.data[base + b0] |= left_mask & right_mask;
        } else {
            // Left partial byte.
            self.data[base + b0] |= 0xff_u8 >> (x0 & 7);
            // Full bytes in the middle.
            for b in (b0 + 1)..b1 {
                self.data[base + b] = 0xff;
            }
            // Right partial byte: set top (x1 % 8) bits; 0xff if x1 is byte-aligned.
            let right_shift = x1 & 7;
            let right_mask = if right_shift == 0 {
                0xff_u8
            } else {
                !(0xff_u8 >> right_shift)
            };
            self.data[base + b1] |= right_mask;
        }
    }

    /// Read one raw byte from the given `row` at byte index `byte_idx`.
    ///
    /// # Panics
    ///
    /// Panics if `row >= height` or `byte_idx >= row_bytes()`.
    #[inline]
    #[must_use]
    pub fn get_byte(&self, row: usize, byte_idx: usize) -> u8 {
        let rb = self.row_bytes();
        assert!(
            row < self.height,
            "row index {row} is out of bounds for AaBuf height {}",
            self.height
        );
        assert!(
            byte_idx < rb,
            "byte_idx {byte_idx} is out of bounds for row_bytes {rb}"
        );
        self.data[row * rb + byte_idx]
    }

    /// Read-only access to a full row as a byte slice.
    ///
    /// # Panics
    ///
    /// Panics if `row >= height`.
    #[must_use]
    pub fn row_slice(&self, row: usize) -> &[u8] {
        assert!(
            row < self.height,
            "row index {row} is out of bounds for AaBuf height {}",
            self.height
        );
        let rb = self.row_bytes();
        &self.data[row * rb..(row + 1) * rb]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Rgb8;

    #[test]
    fn bitmap_stride_alignment() {
        let bm = Bitmap::<Rgb8>::new(7, 3, 4, false);
        // 7 * 3 = 21 bytes, next multiple of 4 = 24
        assert_eq!(bm.stride, 24);
    }

    #[test]
    fn bitmap_row_len() {
        let bm = Bitmap::<Rgb8>::new(5, 2, 1, false);
        assert_eq!(bm.row(0).len(), 5);
        assert_eq!(bm.row(1).len(), 5);
    }

    #[test]
    fn bitmap_clear() {
        let mut bm = Bitmap::<Rgb8>::new(4, 2, 1, true);
        bm.clear(
            Rgb8 {
                r: 255,
                g: 0,
                b: 128,
            },
            200,
        );
        for y in 0..2u32 {
            for px in bm.row(y) {
                assert_eq!(
                    *px,
                    Rgb8 {
                        r: 255,
                        g: 0,
                        b: 128
                    }
                );
            }
            for &a in bm.alpha_row(y).unwrap() {
                assert_eq!(a, 200);
            }
        }
    }

    #[test]
    fn aabuf_set_span_single_byte() {
        let mut buf = AaBuf::new(4); // width = 16, row_bytes = 2
        buf.set_span(0, 2, 6);
        // pixels 2..6 → bits 2,3,4,5 → byte 0: 0b00111100 = 0x3c
        assert_eq!(buf.get_byte(0, 0), 0x3c);
        assert_eq!(buf.get_byte(0, 1), 0x00);
    }

    #[test]
    fn aabuf_set_span_cross_byte() {
        let mut buf = AaBuf::new(4); // width = 16
        buf.set_span(0, 6, 10);
        // pixels 6..10: byte0 bits 6,7 = 0x03; byte1 bits 0,1 = 0xc0
        assert_eq!(buf.get_byte(0, 0), 0x03);
        assert_eq!(buf.get_byte(0, 1), 0xc0);
    }

    #[test]
    fn aabuf_clear() {
        let mut buf = AaBuf::new(4);
        buf.set_span(0, 0, 16);
        buf.clear();
        for i in 0..buf.row_bytes() {
            assert_eq!(buf.get_byte(0, i), 0);
        }
    }
}
