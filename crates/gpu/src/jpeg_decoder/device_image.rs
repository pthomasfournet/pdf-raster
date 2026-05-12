//! GPU device image — a decoded JPEG residing in device memory.

use crate::backend::GpuBackend;

/// A decoded JPEG image in device memory, ready for blit or cache insertion.
///
/// The buffer holds packed RGBA8 pixels in row-major order.  Backed by
/// a `B::DeviceBuffer` owned by this struct; freed when dropped if the
/// backend's `free_device` is called.
///
/// Colour space is always RGBA8 after IDCT: the kernel writes A=0xFF
/// unconditionally (JFIF images have no alpha channel).
pub struct DeviceImage<B: GpuBackend> {
    /// RGBA8 row-major pixel buffer on the device.
    pub buffer: B::DeviceBuffer,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
}
