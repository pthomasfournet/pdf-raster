//! Parameter structs for `GpuBackend::record_*` calls.
//!
//! Each struct holds references (not owned values) to caller-owned device
//! buffers and small scalar args. Backends resolve buffer references to
//! their native pointer/handle type at record time.

use super::GpuBackend;

/// Parameters for a GPU image blit (texture-mapped composite onto the page buffer).
///
/// # Invariants enforced by `BlitParams::validate`
/// - `src_w > 0`, `src_h > 0`, `dst_w > 0`, `dst_h > 0`
/// - `src_layout` is `0` (RGB, 3 bytes/pixel) or `1` (Gray, 1 byte/pixel).
///   Mask images (layout `2`) are CPU-only and must not reach the GPU.
/// - `bbox` is `[x0, y0, x1, y1]` with `x0 <= x1` and `y0 <= y1`
/// - `page_h.is_finite()` and `page_h > 0.0`; conventionally `page_h == dst_h as f32`
/// - All six `inv_ctm` coefficients `is_finite()`
///
/// Backends should call `validate()` at record time. Violating these
/// invariants would propagate `NaN`/`Inf` into kernel arithmetic, route
/// Mask images through the Gray code path silently, or produce garbage
/// pixels (or panics in debug builds via `debug_assert`).
pub struct BlitParams<'a, B: GpuBackend + ?Sized> {
    /// Source image in device memory.
    pub src: &'a B::DeviceBuffer,
    /// Destination page buffer in device memory.
    pub dst: &'a B::DeviceBuffer,
    /// Source image width in pixels (must be > 0).
    pub src_w: u32,
    /// Source image height in pixels (must be > 0).
    pub src_h: u32,
    /// Source image channel layout: `0` = RGB, `1` = Gray. Other values
    /// are rejected by `validate()`.
    pub src_layout: u32,
    /// Destination buffer width in pixels (must be > 0).
    pub dst_w: u32,
    /// Destination buffer height in pixels (must be > 0).
    pub dst_h: u32,
    /// Destination bounding box `[x0, y0, x1, y1]` in page-space pixels (`x0 <= x1`, `y0 <= y1`).
    pub bbox: [i32; 4],
    /// Page height used for PDF → raster coordinate flip; finite and `> 0`.
    pub page_h: f32,
    /// Inverse current transformation matrix (6 coefficients: a b c d e f). Must all be finite.
    pub inv_ctm: [f32; 6],
}

impl<B: GpuBackend + ?Sized> BlitParams<'_, B> {
    /// Validate the invariants documented on `BlitParams`.
    ///
    /// Returns a [`super::BackendError`] describing the first violated
    /// invariant. Backends should call this at the top of `record_blit_image`
    /// so misuse fails loudly with a clear message rather than producing
    /// undefined kernel behaviour.
    ///
    /// # Errors
    /// Returns a `BackendError` if any documented invariant is violated.
    pub fn validate(&self) -> super::Result<()> {
        if self.src_w == 0 || self.src_h == 0 {
            return Err(super::BackendError::new(BlitInvariantViolation(
                "src dimensions must be > 0",
            )));
        }
        if self.dst_w == 0 || self.dst_h == 0 {
            return Err(super::BackendError::new(BlitInvariantViolation(
                "dst dimensions must be > 0",
            )));
        }
        let [x0, y0, x1, y1] = self.bbox;
        if x0 > x1 || y0 > y1 {
            return Err(super::BackendError::new(BlitInvariantViolation(
                "bbox must satisfy x0 <= x1 and y0 <= y1",
            )));
        }
        if !self.page_h.is_finite() || self.page_h <= 0.0 {
            return Err(super::BackendError::new(BlitInvariantViolation(
                "page_h must be finite and > 0",
            )));
        }
        if !self.inv_ctm.iter().all(|c| c.is_finite()) {
            return Err(super::BackendError::new(BlitInvariantViolation(
                "inv_ctm must contain only finite coefficients",
            )));
        }
        // src_layout: 0 = RGB, 1 = Gray. Anything else (especially 2 = Mask)
        // would fall through the kernel's `if (src_layout == 0) { RGB } else { Gray }`
        // dispatch and silently produce wrong pixels.
        if self.src_layout > 1 {
            return Err(super::BackendError::new(BlitInvariantViolation(
                "src_layout must be 0 (RGB) or 1 (Gray); Mask layout is CPU-only",
            )));
        }
        Ok(())
    }
}

#[derive(Debug)]
struct BlitInvariantViolation(&'static str);

impl std::fmt::Display for BlitInvariantViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "BlitParams invariant: {}", self.0)
    }
}

impl std::error::Error for BlitInvariantViolation {}

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

/// Parameters for a GPU ICC CMYK→RGB CLUT lookup.
///
/// The matrix fast-path is handled on the CPU (AVX-512); only the CLUT
/// kernel runs on the GPU, so `clut` is required.
pub struct IccClutParams<'a, B: GpuBackend + ?Sized> {
    /// Input CMYK pixels in device memory.
    pub cmyk: &'a B::DeviceBuffer,
    /// Output RGB pixels in device memory.
    pub rgb: &'a B::DeviceBuffer,
    /// CLUT table in device memory.
    pub clut: &'a B::DeviceBuffer,
    /// Number of pixels to transform.
    pub n_pixels: u32,
}

/// Recover `grid_n` such that `clut_byte_len == grid_n^4 * 3`, or `None`
/// if the length is not a valid CLUT layout.
///
/// The CLUT layout is `(k * G^3 + c * G^2 + m * G + y) * 3` bytes —
/// `grid_n^4` 3-byte RGB nodes.  Typical PDF profiles use `grid_n` of
/// 17 or 33.  Shared between the CUDA and Vulkan backends; both
/// recover `grid_n` from the buffer size at record time.
#[must_use]
pub fn grid_n_from_clut_len(len: usize) -> Option<u32> {
    if !len.is_multiple_of(3) {
        return None;
    }
    let nodes = len / 3;
    // Integer 4th root by iteration: grid_n ≤ 255 in practice (PDF
    // profiles rarely exceed 33; 255 is a generous upper bound).
    for grid in 2u32..=255 {
        let g = grid as usize;
        let pow4 = g.checked_mul(g)?.checked_mul(g)?.checked_mul(g)?;
        if pow4 == nodes {
            return Some(grid);
        }
        if pow4 > nodes {
            return None;
        }
    }
    None
}

#[cfg(test)]
mod clut_helpers_tests {
    use super::grid_n_from_clut_len;

    #[test]
    fn round_trips_typical_grids() {
        assert_eq!(grid_n_from_clut_len(17 * 17 * 17 * 17 * 3), Some(17));
        assert_eq!(grid_n_from_clut_len(33 * 33 * 33 * 33 * 3), Some(33));
    }

    #[test]
    fn rejects_non_multiple_of_3() {
        assert_eq!(grid_n_from_clut_len(83_521 * 3 + 1), None);
    }

    #[test]
    fn rejects_non_4th_power() {
        assert_eq!(grid_n_from_clut_len(100 * 3), None);
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Phantom backend used solely for validating param-struct invariants in
    /// pure-CPU tests (no cudarc required).
    enum FakeBackend {}

    impl GpuBackend for FakeBackend {
        type DeviceBuffer = ();
        type HostBuffer = ();
        type PageFence = ();

        fn alloc_device(&self, _size: usize) -> super::super::Result<Self::DeviceBuffer> {
            unreachable!("FakeBackend is only constructed via &(); never instantiated")
        }
        fn free_device(&self, _buf: Self::DeviceBuffer) {}
        fn alloc_host_pinned(&self, _size: usize) -> super::super::Result<Self::HostBuffer> {
            unreachable!()
        }
        fn free_host_pinned(&self, _buf: Self::HostBuffer) {}
        fn begin_page(&self) -> super::super::Result<()> {
            unreachable!()
        }
        fn record_blit_image(&self, _p: BlitParams<'_, Self>) -> super::super::Result<()> {
            unreachable!()
        }
        fn record_aa_fill(&self, _p: AaFillParams<'_, Self>) -> super::super::Result<()> {
            unreachable!()
        }
        fn record_icc_clut(&self, _p: IccClutParams<'_, Self>) -> super::super::Result<()> {
            unreachable!()
        }
        fn record_tile_fill(&self, _p: TileFillParams<'_, Self>) -> super::super::Result<()> {
            unreachable!()
        }
        fn record_composite(&self, _p: CompositeParams<'_, Self>) -> super::super::Result<()> {
            unreachable!()
        }
        fn record_apply_soft_mask(&self, _p: SoftMaskParams<'_, Self>) -> super::super::Result<()> {
            unreachable!()
        }
        fn submit_page(&self) -> super::super::Result<Self::PageFence> {
            unreachable!()
        }
        fn wait_page(&self, _fence: Self::PageFence) -> super::super::Result<()> {
            unreachable!()
        }
        fn upload_async(
            &self,
            _dst: &Self::DeviceBuffer,
            _src: &[u8],
        ) -> super::super::Result<Self::PageFence> {
            unreachable!()
        }
        fn alloc_device_zeroed(&self, _size: usize) -> super::super::Result<Self::DeviceBuffer> {
            unreachable!()
        }
        fn device_buffer_len(&self, _buf: &Self::DeviceBuffer) -> usize {
            unreachable!()
        }
        fn download_async<'a>(
            &self,
            _src: &'a Self::DeviceBuffer,
            _dst: &'a mut [u8],
        ) -> super::super::Result<super::super::DownloadHandle<'a, Self>> {
            unreachable!()
        }
        fn wait_download(
            &self,
            _handle: super::super::DownloadHandle<'_, Self>,
        ) -> super::super::Result<()> {
            unreachable!()
        }
        fn submit_transfer(&self) -> super::super::Result<Self::PageFence> {
            unreachable!()
        }
        fn wait_transfer(&self, _fence: Self::PageFence) -> super::super::Result<()> {
            unreachable!()
        }
        fn detect_vram_budget(&self) -> super::super::Result<super::super::VramBudget> {
            unreachable!()
        }
    }

    fn ok_blit() -> BlitParams<'static, FakeBackend> {
        static UNIT: () = ();
        BlitParams {
            src: &UNIT,
            dst: &UNIT,
            src_w: 100,
            src_h: 100,
            src_layout: 0,
            dst_w: 200,
            dst_h: 200,
            bbox: [0, 0, 100, 100],
            page_h: 200.0,
            inv_ctm: [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        }
    }

    #[test]
    fn validate_accepts_valid_blit_params() {
        ok_blit().validate().expect("valid params should pass");
    }

    #[test]
    fn validate_rejects_zero_src_dim() {
        let mut p = ok_blit();
        p.src_w = 0;
        let err = p.validate().unwrap_err().to_string();
        assert!(err.contains("src dimensions"), "{err}");
    }

    #[test]
    fn validate_rejects_zero_dst_dim() {
        let mut p = ok_blit();
        p.dst_h = 0;
        let err = p.validate().unwrap_err().to_string();
        assert!(err.contains("dst dimensions"), "{err}");
    }

    #[test]
    fn validate_rejects_inverted_bbox() {
        let mut p = ok_blit();
        p.bbox = [50, 0, 10, 100]; // x0 > x1
        let err = p.validate().unwrap_err().to_string();
        assert!(err.contains("bbox"), "{err}");
    }

    #[test]
    fn validate_rejects_nan_page_h() {
        let mut p = ok_blit();
        p.page_h = f32::NAN;
        let err = p.validate().unwrap_err().to_string();
        assert!(err.contains("page_h"), "{err}");
    }

    #[test]
    fn validate_rejects_negative_page_h() {
        let mut p = ok_blit();
        p.page_h = -1.0;
        let err = p.validate().unwrap_err().to_string();
        assert!(err.contains("page_h"), "{err}");
    }

    #[test]
    fn validate_rejects_zero_page_h() {
        let mut p = ok_blit();
        p.page_h = 0.0;
        let err = p.validate().unwrap_err().to_string();
        assert!(err.contains("page_h"), "{err}");
    }

    #[test]
    fn validate_rejects_inf_inv_ctm() {
        let mut p = ok_blit();
        p.inv_ctm[3] = f32::INFINITY;
        let err = p.validate().unwrap_err().to_string();
        assert!(err.contains("inv_ctm"), "{err}");
    }

    #[test]
    fn validate_rejects_nan_inv_ctm() {
        let mut p = ok_blit();
        p.inv_ctm[0] = f32::NAN;
        let err = p.validate().unwrap_err().to_string();
        assert!(err.contains("inv_ctm"), "{err}");
    }

    #[test]
    fn validate_accepts_layout_0_rgb() {
        let mut p = ok_blit();
        p.src_layout = 0;
        p.validate().expect("layout 0 (RGB) is valid");
    }

    #[test]
    fn validate_accepts_layout_1_gray() {
        let mut p = ok_blit();
        p.src_layout = 1;
        p.validate().expect("layout 1 (Gray) is valid");
    }

    #[test]
    fn validate_rejects_layout_2_mask() {
        let mut p = ok_blit();
        p.src_layout = 2;
        let err = p.validate().unwrap_err().to_string();
        assert!(err.contains("Mask"), "{err}");
        assert!(err.contains("CPU-only"), "{err}");
    }

    #[test]
    fn validate_rejects_unknown_layout() {
        let mut p = ok_blit();
        p.src_layout = 999;
        let err = p.validate().unwrap_err().to_string();
        assert!(err.contains("src_layout"), "{err}");
    }
}
