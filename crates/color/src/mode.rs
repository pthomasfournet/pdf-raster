//! Pixel color modes, mirroring SplashColorMode from SplashTypes.h.

#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
pub enum PixelMode {
    #[default]
    Mono1 = 0, // 1 bit/px, MSB-first packed; (width+7)/8 bytes/row
    Mono8 = 1,    // 1 byte/px
    Rgb8 = 2,     // R G B
    Bgr8 = 3,     // B G R
    Xbgr8 = 4,    // X B G R (X=255; used by Cairo/QImage backends)
    Cmyk8 = 5,    // C M Y K
    DeviceN8 = 6, // C M Y K + 4 spot channels = 8 bytes/px
}

/// Bytes per pixel for each mode.
/// Mono1 is 0 because it is sub-byte (handled separately in callers).
pub const NCOMPS: [usize; 7] = [0, 1, 3, 3, 4, 4, 8];

impl PixelMode {
    #[inline(always)]
    pub fn bytes_per_pixel(self) -> usize {
        NCOMPS[self as usize]
    }
}
