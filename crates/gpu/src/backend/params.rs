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
        const KIND: &str = "BlitInvariantViolation";
        let invariant =
            |detail: &'static str| super::BackendError::InvariantViolation { kind: KIND, detail };
        if self.src_w == 0 || self.src_h == 0 {
            return Err(invariant("src dimensions must be > 0"));
        }
        if self.dst_w == 0 || self.dst_h == 0 {
            return Err(invariant("dst dimensions must be > 0"));
        }
        let [x0, y0, x1, y1] = self.bbox;
        if x0 > x1 || y0 > y1 {
            return Err(invariant("bbox must satisfy x0 <= x1 and y0 <= y1"));
        }
        if !self.page_h.is_finite() || self.page_h <= 0.0 {
            return Err(invariant("page_h must be finite and > 0"));
        }
        if !self.inv_ctm.iter().all(|c| c.is_finite()) {
            return Err(invariant("inv_ctm must contain only finite coefficients"));
        }
        // src_layout: 0 = RGB, 1 = Gray. Anything else (especially 2 = Mask)
        // would fall through the kernel's `if (src_layout == 0) { RGB } else { Gray }`
        // dispatch and silently produce wrong pixels.
        if self.src_layout > 1 {
            return Err(invariant(
                "src_layout must be 0 (RGB) or 1 (Gray); Mask layout is CPU-only",
            ));
        }
        Ok(())
    }
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

/// Workgroup size used by every phase of the Blelloch scan kernel.
///
/// Public so dispatchers can size `block_sums` correctly:
/// `block_count = ceil(len_elems / SCAN_WORKGROUP_SIZE)`. Must match
/// the `numthreads(...)` declaration in `kernels/jpeg/blelloch_scan.slang`
/// (each thread handles 2 elements, so the per-workgroup tile is
/// `2 * SCAN_WORKGROUP_SIZE = 1024` elements).
pub const SCAN_WORKGROUP_SIZE: u32 = 512;

/// Maximum number of workgroups the single-tier `BlockSums` phase can handle.
///
/// The middle phase runs as a single workgroup whose tile covers
/// `2 * SCAN_WORKGROUP_SIZE = 1024` elements; arrays whose block
/// count exceeds this require a recursive scan (out of scope for
/// the v1 JPEG decoder).
pub const SCAN_MAX_BLOCKS: u32 = 1024;

/// Phase selector for [`record_scan`](super::GpuBackend::record_scan).
///
/// The three phases of the multi-workgroup Blelloch exclusive scan
/// (Blelloch 1990); see `kernels/jpeg/blelloch_scan.slang`.
///
/// All three phases share the same buffer set so callers don't have to
/// rebind between dispatches. The kernel reads `phase` as a flat u32
/// (encoded by `ScanPhase::as_kernel_arg`) and branches internally.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScanPhase {
    /// Per-workgroup local scan: each workgroup of
    /// `SCAN_WORKGROUP_SIZE` threads exclusively scans its 1024-element
    /// tile of `data` in shared memory, then writes the tile sum into
    /// `block_sums[workgroup_idx]`.
    PerWorkgroup,
    /// Single-workgroup scan over `block_sums`. Requires the block
    /// count to be ≤ [`SCAN_MAX_BLOCKS`].
    BlockSums,
    /// Adds the (now-scanned) `block_sums[workgroup_idx]` back into
    /// every element of workgroup `workgroup_idx`'s output slice.
    ScatterBlockSums,
}

impl ScanPhase {
    /// Kernel argument encoding. Stable across backends — the SPIR-V
    /// and PTX entry points read this and dispatch with a switch.
    #[must_use]
    pub const fn as_kernel_arg(self) -> u32 {
        match self {
            Self::PerWorkgroup => 0,
            Self::BlockSums => 1,
            Self::ScatterBlockSums => 2,
        }
    }
}

/// Parameters for one phase of the Blelloch exclusive scan kernel.
///
/// The same buffer set is reused across the three phases; only the
/// `phase` field changes between dispatches.
///
/// # Invariants enforced by `ScanParams::validate`
/// - `len_elems > 0`.
/// - `data` device capacity ≥ `len_elems * 4` bytes (u32 elements).
/// - `block_sums` device capacity ≥
///   `block_count * 4` bytes where
///   `block_count = ceil(len_elems / (2 * SCAN_WORKGROUP_SIZE))`.
/// - `block_count <= SCAN_MAX_BLOCKS` — the `BlockSums` middle phase
///   uses a single workgroup, so arrays whose block count exceeds the
///   workgroup tile size require a recursive scan (not implemented).
///
/// Validation is mandatory: violating these would silently produce
/// out-of-bounds device-pointer arithmetic in the kernel.
pub struct ScanParams<'a, B: GpuBackend + ?Sized> {
    /// Input/output u32 buffer scanned in place. Must hold at least
    /// `len_elems` u32s.
    pub data: &'a B::DeviceBuffer,
    /// Per-workgroup scratch holding tile sums between phases. Sized
    /// for the worst-case block count (must hold at least
    /// `ceil(len_elems / 1024)` u32s).
    pub block_sums: &'a B::DeviceBuffer,
    /// Number of u32 elements to scan. Backend computes the dispatch
    /// grid from this; the kernel reads it as a push-constant /
    /// kernel-arg.
    pub len_elems: u32,
    /// Which of the three Blelloch phases this dispatch executes.
    pub phase: ScanPhase,
}

impl<B: GpuBackend + ?Sized> ScanParams<'_, B> {
    /// Validate the invariants documented on `ScanParams`.
    ///
    /// Callers should run this once per scan (after constructing the
    /// params for `PerWorkgroup`, before any of the three dispatches)
    /// — the invariants don't change across phases since all three
    /// share buffers and `len_elems`.
    ///
    /// # Errors
    /// Returns a `BackendError` if any documented invariant fails.
    pub fn validate(&self, backend: &B) -> super::Result<()> {
        const KIND: &str = "ScanInvariantViolation";
        let invariant =
            |detail: &'static str| super::BackendError::InvariantViolation { kind: KIND, detail };

        if self.len_elems == 0 {
            return Err(invariant("len_elems must be > 0"));
        }

        // Each element is u32 = 4 bytes; check both buffers have room.
        let data_bytes_needed = (self.len_elems as usize)
            .checked_mul(4)
            .ok_or_else(|| invariant("len_elems * 4 overflows usize"))?;
        if backend.device_buffer_len(self.data) < data_bytes_needed {
            return Err(invariant("data buffer is smaller than len_elems * 4 bytes"));
        }

        let workgroup_tile = 2 * SCAN_WORKGROUP_SIZE;
        let block_count = self.len_elems.div_ceil(workgroup_tile);
        if block_count > SCAN_MAX_BLOCKS {
            return Err(invariant(
                "block_count exceeds SCAN_MAX_BLOCKS (1024); recursive scan not implemented",
            ));
        }
        let block_sums_bytes_needed = (block_count as usize)
            .checked_mul(4)
            .ok_or_else(|| invariant("block_count * 4 overflows usize"))?;
        if backend.device_buffer_len(self.block_sums) < block_sums_bytes_needed {
            return Err(invariant(
                "block_sums buffer is smaller than ceil(len_elems / 1024) * 4 bytes",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Phantom backend used solely for validating param-struct invariants in
    /// pure-CPU tests (no cudarc required).
    ///
    /// `DeviceBuffer = usize` — the buffer "is" its capacity in bytes,
    /// so `device_buffer_len` is a trivial identity. Most other trait
    /// methods are `unreachable!()` because validate-only tests don't
    /// route through them.
    struct FakeBackend;

    impl GpuBackend for FakeBackend {
        type DeviceBuffer = usize;
        type HostBuffer = ();
        type PageFence = ();

        fn alloc_device(&self, _size: usize) -> super::super::Result<Self::DeviceBuffer> {
            unreachable!("FakeBackend allocation paths are not exercised by validate-only tests")
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
        fn record_scan(&self, _p: ScanParams<'_, Self>) -> super::super::Result<()> {
            unreachable!()
        }
        fn record_zero_buffer(&self, _buf: &Self::DeviceBuffer) -> super::super::Result<()> {
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
        fn device_buffer_len(&self, buf: &Self::DeviceBuffer) -> usize {
            *buf
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

    /// Static stand-in for a device buffer of "large enough" capacity.
    /// The numeric value is only consulted by `device_buffer_len`, which
    /// `BlitParams::validate` doesn't call — but `ScanParams::validate` does.
    static FAKE_BUF: usize = usize::MAX;

    fn ok_blit() -> BlitParams<'static, FakeBackend> {
        BlitParams {
            src: &FAKE_BUF,
            dst: &FAKE_BUF,
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

    // --- ScanParams ---

    /// Build a `ScanParams` with both buffers sized to `data_bytes` /
    /// `block_sums_bytes` respectively (the test consults
    /// `device_buffer_len`, which for `FakeBackend` is identity over
    /// the `usize` buffer).
    fn scan_params<'a>(
        data_bytes: &'a usize,
        block_sums_bytes: &'a usize,
        len_elems: u32,
        phase: ScanPhase,
    ) -> ScanParams<'a, FakeBackend> {
        ScanParams {
            data: data_bytes,
            block_sums: block_sums_bytes,
            len_elems,
            phase,
        }
    }

    #[test]
    fn scan_phase_kernel_arg_is_stable() {
        // The kernel reads this; the encoding must not drift.
        assert_eq!(ScanPhase::PerWorkgroup.as_kernel_arg(), 0);
        assert_eq!(ScanPhase::BlockSums.as_kernel_arg(), 1);
        assert_eq!(ScanPhase::ScatterBlockSums.as_kernel_arg(), 2);
    }

    #[test]
    fn scan_validate_accepts_minimal_valid() {
        let data = 4096usize;
        let block_sums = 16usize;
        let p = scan_params(&data, &block_sums, 1024, ScanPhase::PerWorkgroup);
        p.validate(&FakeBackend)
            .expect("4 KiB data + 1024 elems is valid");
    }

    #[test]
    fn scan_validate_rejects_zero_len() {
        let data = 4096usize;
        let block_sums = 16usize;
        let p = scan_params(&data, &block_sums, 0, ScanPhase::PerWorkgroup);
        let err = p.validate(&FakeBackend).unwrap_err().to_string();
        assert!(err.contains("len_elems"), "{err}");
    }

    #[test]
    fn scan_validate_rejects_undersized_data_buffer() {
        // 1024 elems * 4 bytes = 4096 bytes needed; provide 4095.
        let data = 4095usize;
        let block_sums = 16usize;
        let p = scan_params(&data, &block_sums, 1024, ScanPhase::PerWorkgroup);
        let err = p.validate(&FakeBackend).unwrap_err().to_string();
        assert!(err.contains("data buffer"), "{err}");
    }

    #[test]
    fn scan_validate_rejects_undersized_block_sums() {
        // 2049 elems → block_count = ceil(2049/1024) = 3 → 12 bytes needed.
        let data = 2049 * 4usize;
        let block_sums = 4usize; // only room for 1 block
        let p = scan_params(&data, &block_sums, 2049, ScanPhase::PerWorkgroup);
        let err = p.validate(&FakeBackend).unwrap_err().to_string();
        assert!(err.contains("block_sums buffer"), "{err}");
    }

    #[test]
    fn scan_validate_rejects_too_many_blocks() {
        // len_elems = 1024 * 1024 + 1 → block_count = 1025 > SCAN_MAX_BLOCKS.
        let data = (1024 * 1024 + 1) * 4usize;
        let block_sums = 1025 * 4usize;
        let p = scan_params(&data, &block_sums, 1024 * 1024 + 1, ScanPhase::PerWorkgroup);
        let err = p.validate(&FakeBackend).unwrap_err().to_string();
        assert!(err.contains("block_count"), "{err}");
        assert!(err.contains("recursive scan"), "{err}");
    }

    #[test]
    fn scan_validate_accepts_at_max_blocks() {
        // Exactly SCAN_MAX_BLOCKS blocks (the upper limit).
        let max_elems = SCAN_MAX_BLOCKS * 2 * SCAN_WORKGROUP_SIZE;
        let data = (max_elems as usize) * 4;
        let block_sums = (SCAN_MAX_BLOCKS as usize) * 4;
        let p = scan_params(&data, &block_sums, max_elems, ScanPhase::PerWorkgroup);
        p.validate(&FakeBackend)
            .expect("exactly SCAN_MAX_BLOCKS is valid");
    }
}
