//! Parameter structs for `GpuBackend::record_*` calls.
//!
//! Each struct holds references (not owned values) to caller-owned device
//! buffers and small scalar args. Backends resolve buffer references to
//! their native pointer/handle type at record time.

use super::GpuBackend;

/// Parameters for a GPU image blit (texture-mapped composite onto the page buffer).
pub struct BlitParams<'a, B: GpuBackend + ?Sized> {
    /// Source image in device memory.
    pub src: &'a B::DeviceBuffer,
    /// Destination page buffer in device memory.
    pub dst: &'a B::DeviceBuffer,
    /// Source image width in pixels.
    pub src_w: u32,
    /// Source image height in pixels.
    pub src_h: u32,
    /// Source image channel layout (backend-defined enum value).
    pub src_layout: u32,
    /// Destination buffer width in pixels.
    pub dst_w: u32,
    /// Destination buffer height in pixels.
    pub dst_h: u32,
    /// Destination bounding box `[x0, y0, x1, y1]` in page-space pixels.
    pub bbox: [i32; 4],
    /// Page height used for PDF → raster coordinate flip.
    pub page_h: i32,
    /// Inverse current transformation matrix (6 coefficients: a b c d e f).
    pub inv_ctm: [f32; 6],
}

/// Parameters for a GPU antialiased fill pass.
pub struct AaFillParams<'a, B: GpuBackend + ?Sized> {
    /// Packed edge segments in device memory.
    pub segs: &'a B::DeviceBuffer,
    /// Number of segments in `segs`.
    pub n_segs: u32,
    /// Output coverage buffer in device memory.
    pub coverage: &'a B::DeviceBuffer,
    /// Coverage buffer width in pixels.
    pub width: u32,
    /// Coverage buffer height in pixels.
    pub height: u32,
    /// Fill rule: 0 = non-zero winding, 1 = even-odd.
    pub fill_rule: u8,
}

/// Parameters for a GPU ICC CMYK→RGB colour transform.
pub struct IccClutParams<'a, B: GpuBackend + ?Sized> {
    /// Input CMYK pixels in device memory.
    pub cmyk: &'a B::DeviceBuffer,
    /// Output RGB pixels in device memory.
    pub rgb: &'a B::DeviceBuffer,
    /// CLUT table in device memory; `None` selects the matrix fast-path.
    pub clut: Option<&'a B::DeviceBuffer>,
    /// Number of pixels to transform.
    pub n_pixels: u32,
}

/// Parameters for a GPU tile-parallel analytical fill.
pub struct TileFillParams<'a, B: GpuBackend + ?Sized> {
    /// Packed tile-fill records in device memory.
    pub records: &'a B::DeviceBuffer,
    /// Per-tile start offsets into `records`.
    pub tile_starts: &'a B::DeviceBuffer,
    /// Per-tile record counts.
    pub tile_counts: &'a B::DeviceBuffer,
    /// Output coverage buffer in device memory.
    pub coverage: &'a B::DeviceBuffer,
    /// Coverage buffer width in pixels.
    pub width: u32,
    /// Coverage buffer height in pixels.
    pub height: u32,
    /// Fill rule: 0 = non-zero winding, 1 = even-odd.
    pub fill_rule: u8,
}

/// Parameters for a GPU Porter-Duff source-over composite.
pub struct CompositeParams<'a, B: GpuBackend + ?Sized> {
    /// Source RGBA pixels in device memory.
    pub src: &'a B::DeviceBuffer,
    /// Destination RGBA pixels in device memory (read-modify-write).
    pub dst: &'a B::DeviceBuffer,
    /// Number of pixels to composite.
    pub n_pixels: u32,
}

/// Parameters for a GPU soft-mask application.
pub struct SoftMaskParams<'a, B: GpuBackend + ?Sized> {
    /// RGBA pixel buffer to mask in device memory (read-modify-write).
    pub pixels: &'a B::DeviceBuffer,
    /// Greyscale mask buffer in device memory.
    pub mask: &'a B::DeviceBuffer,
    /// Number of pixels to process.
    pub n_pixels: u32,
}
