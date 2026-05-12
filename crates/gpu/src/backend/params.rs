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

    /// Workgroup grid dimensions for this phase given `len_elems`.
    ///
    /// - `PerWorkgroup` and `ScatterBlockSums` dispatch one workgroup
    ///   per `2 * SCAN_WORKGROUP_SIZE`-element tile (rounded up).
    /// - `BlockSums` dispatches exactly one workgroup over the (≤
    ///   `SCAN_MAX_BLOCKS`) tile-sums array.
    ///
    /// Shared by the CUDA and Vulkan recorders so the two backends
    /// can't drift on the grid shape — the kernel-side workgroup
    /// layout assumes this exact partitioning.
    #[must_use]
    pub const fn dispatch_grid(self, len_elems: u32) -> (u32, u32, u32) {
        match self {
            Self::PerWorkgroup | Self::ScatterBlockSums => (scan_block_count(len_elems), 1, 1),
            Self::BlockSums => (1, 1, 1),
        }
    }
}

/// Number of per-workgroup tiles required to cover `len_elems`
/// elements of the Blelloch scan. Each tile is
/// `2 * SCAN_WORKGROUP_SIZE = 1024` elements.
///
/// Used by `ScanPhase::dispatch_grid` and by host dispatchers that
/// need to size the `block_sums` buffer. Single source of truth for
/// the partition factor.
#[must_use]
pub const fn scan_block_count(len_elems: u32) -> u32 {
    let tile = 2 * SCAN_WORKGROUP_SIZE;
    len_elems.div_ceil(tile)
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

        let block_count = scan_block_count(self.len_elems);
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

/// Number of canonical-codebook entries the parallel-Huffman kernel reads per component.
///
/// The CPU-side `CanonicalCodebook` is a 65 536-entry 16-bit-prefix
/// LUT; the GPU codebook buffer is a flat
/// `u32[num_components * HUFFMAN_CODEBOOK_ENTRIES]`. Public so
/// dispatchers can size + populate the buffer correctly.
pub const HUFFMAN_CODEBOOK_ENTRIES: usize = 65_536;

/// Workgroup size for the parallel-Huffman Phase 1 kernel.
///
/// Must match the `numthreads(...)` in `parallel_huffman.slang` /
/// the block-dim arg the CUDA launcher (`launch_phase1_intra_sync_async`)
/// uses. Backends size their dispatch grid as
/// `ceil(num_subsequences / HUFFMAN_PHASE1_THREADS)`.
pub const HUFFMAN_PHASE1_THREADS: u32 = 256;

/// Phase selector for [`record_huffman`](super::GpuBackend::record_huffman).
///
/// Phase 3 of the Weißenberger algorithm is the Blelloch scan; see
/// [`ScanPhase`] for that one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HuffmanPhase {
    /// One thread per subsequence; walks the stream from
    /// `seq_idx * subsequence_bits` to `min(length_bits,
    /// (seq_idx + 2) * subsequence_bits)`, writing the final
    /// `(p, n, c, z)` state into `s_info[seq_idx]`.
    Phase1IntraSync,
    /// One thread per subsequence; checks whether `s_info[i]` is
    /// synced with `s_info[i+1]` (matching `c` and `z`, and `i`'s
    /// walk reached `i+1`'s start). Writes `sync_flags[i] = 1` on
    /// match, else advances `s_info[i]` by one symbol and writes
    /// `sync_flags[i] = 0`. The host loops this dispatch until all
    /// flags are 1 or the retry bound is exhausted.
    Phase2InterSync,
    /// One thread per subsequence; re-walks the subseq's owned
    /// region `[prev.p, me.p)` and writes each decoded symbol to
    /// `symbols_out[offsets[seq_idx] + local_n]`. Requires
    /// `offsets` and `symbols_out` to be `Some` (Phase 3's
    /// exclusive-scan output + a buffer sized to the total symbol
    /// count).
    Phase4Redecode,
    /// JPEG-framed counterpart of `Phase1IntraSync`.  Uses real
    /// JPEG decoding semantics — DC magnitude bits are skipped,
    /// AC run/size pairs advance the zig-zag cursor by `run + 1`,
    /// `EOB` (sym = 0x00) ends the block, `ZRL` (sym = 0xF0)
    /// advances by 16.  Requires `dc_codebook` and `mcu_schedule`
    /// to be `Some`.
    JpegPhase1IntraSync,
    /// JPEG-framed counterpart of `Phase2InterSync`.  Same
    /// requirements as `JpegPhase1IntraSync` plus the
    /// `sync_flags` buffer Phase2 needs.
    JpegPhase2InterSync,
    /// JPEG-framed counterpart of `Phase4Redecode`.  Emits the
    /// per-block Huffman symbol stream (DC magnitude category +
    /// AC run/size bytes).  Same requirements as
    /// `JpegPhase1IntraSync` plus the Phase4 `offsets` +
    /// `symbols_out` + `decode_status` buffers.
    JpegPhase4Redecode,
}

impl HuffmanPhase {
    /// True for the JPEG-framed phase variants
    /// (`JpegPhase{1,2,4}*`).  Used by validation + dispatch to
    /// pick the JPEG kernel path.
    #[must_use]
    pub const fn is_jpeg_framed(self) -> bool {
        matches!(
            self,
            Self::JpegPhase1IntraSync | Self::JpegPhase2InterSync | Self::JpegPhase4Redecode,
        )
    }
}

/// Parameters for one phase of the parallel-Huffman JPEG decoder.
///
/// # Invariants enforced by `HuffmanParams::validate`
/// - `length_bits > 0` and `subsequence_bits > 0`.
/// - `num_components ≥ 1`.
/// - `bitstream` device capacity ≥ `ceil(length_bits / 32) * 4` bytes,
///   plus one trailing word (the kernel's `peek16` reads two adjacent
///   words even at the stream end).
/// - `codebook` device capacity ≥
///   `num_components * HUFFMAN_CODEBOOK_ENTRIES * 4` bytes.
/// - `s_info` device capacity ≥ `num_subsequences * 16` bytes,
///   where `num_subsequences = ceil(length_bits / subsequence_bits)`
///   (one `uint4 = (p, n, c, z)` per subsequence).
/// - When `phase ∈ {Phase2InterSync, JpegPhase2InterSync}`:
///   `sync_flags` must be `Some` with device capacity ≥
///   `num_subsequences * 4` bytes (one u32 flag per subsequence). For
///   the Phase 1 / Phase 4 variants the field is ignored.
/// - When `phase ∈ {Phase4Redecode, JpegPhase4Redecode}`: `offsets`,
///   `symbols_out`, and `decode_status` must all be `Some`. `offsets`
///   ≥ `num_subsequences * 4` bytes; `symbols_out` ≥ `total_symbols *
///   4` bytes; `decode_status` ≥ `num_subsequences * 4` bytes (one
///   u32 per subseq encoding the kernel's exit condition — see
///   `Phase4FailureKind` for the discriminants).
/// - When `phase.is_jpeg_framed()`: `dc_codebook` and `mcu_schedule`
///   must both be `Some`, and `blocks_per_mcu ≥ 1`.  `dc_codebook`
///   has the same flat layout as `codebook`; `mcu_schedule` is
///   `u32[blocks_per_mcu]`, each entry packing
///   `(ac_sel << 16) | (dc_sel << 8) | component_idx`.
pub struct HuffmanParams<'a, B: GpuBackend + ?Sized> {
    /// Packed bitstream as big-endian u32 words.
    pub bitstream: &'a B::DeviceBuffer,
    /// Flat codebook: `u32[num_components * HUFFMAN_CODEBOOK_ENTRIES]`.
    /// Entry layout: bits 8..16 = `num_bits`, bits 0..8 = `symbol`.
    /// `num_bits == 0` = no codeword matches that prefix.
    pub codebook: &'a B::DeviceBuffer,
    /// Per-subsequence output state: `uint4 = (p, n, c, z)` each.
    pub s_info: &'a B::DeviceBuffer,
    /// Per-subseq sync flag (1 = synced with right neighbour,
    /// 0 = unsynced and advanced this pass). Required for
    /// `Phase2InterSync` and `JpegPhase2InterSync`; ignored for the
    /// Phase 1 / Phase 4 variants.
    pub sync_flags: Option<&'a B::DeviceBuffer>,
    /// Exclusive-scan output from Phase 3 (`u32[num_subsequences]`):
    /// `offsets[i]` is the base index into `symbols_out` where
    /// subseq `i`'s emitted symbols are written. Required for
    /// `Phase4Redecode` and `JpegPhase4Redecode`; ignored for the
    /// Phase 1 / Phase 2 variants.
    pub offsets: Option<&'a B::DeviceBuffer>,
    /// Final decoded symbol stream (`u32[total_symbols]`). Required
    /// for `Phase4Redecode` and `JpegPhase4Redecode`; ignored for
    /// the Phase 1 / Phase 2 variants. Caller sizes it from
    /// Phase 3's scan total + last subseq's `n`.
    pub symbols_out: Option<&'a B::DeviceBuffer>,
    /// Per-subseq exit-condition output (`u32[num_subsequences]`):
    /// `decode_status[i]` encodes the Phase 4 kernel's exit
    /// condition for subseq `i` — `Phase4FailureKind::Ok` on clean
    /// `end_p` exit, or one of the typed failure modes
    /// (`PrefixMiss`, `LengthBits`, `Incomplete`) when the helper
    /// bails mid-region. Allocate zero-filled so untouched slots
    /// default to Ok. Required for `Phase4Redecode` and
    /// `JpegPhase4Redecode`; ignored for the Phase 1 / Phase 2
    /// variants.
    pub decode_status: Option<&'a B::DeviceBuffer>,
    /// Flat DC-class codebook for the JPEG-framed phases.  Same
    /// layout as [`Self::codebook`] (one
    /// `HUFFMAN_CODEBOOK_ENTRIES`-wide block per selector).
    /// Required for any `phase.is_jpeg_framed()`; ignored for the
    /// synthetic-stream phases.
    pub dc_codebook: Option<&'a B::DeviceBuffer>,
    /// Per-block schedule within an MCU: `u32[blocks_per_mcu]`.
    /// Each entry encodes `(ac_sel << 16) | (dc_sel << 8) |
    /// component_idx`, where `component_idx`, `dc_sel`, and
    /// `ac_sel` are all 0..=3. Required for any
    /// `phase.is_jpeg_framed()`; ignored otherwise.
    pub mcu_schedule: Option<&'a B::DeviceBuffer>,
    /// Exact bit count of `bitstream`.
    pub length_bits: u32,
    /// Subsequence size (Wei §III); 128 / 512 / 1024 in practice.
    pub subsequence_bits: u32,
    /// Number of Huffman tables (one per component for the MVP).
    pub num_components: u32,
    /// Number of 8×8 blocks per MCU in the JPEG-framed phases.
    /// Equals the sum of `h_sampling * v_sampling` across active
    /// scan components for an interleaved scan, or 1 for a
    /// non-interleaved scan (JPEG § A.2.3). Must be ≥ 1 when
    /// `phase.is_jpeg_framed()`; ignored otherwise (set to 0).
    pub blocks_per_mcu: u32,
    /// Total symbol count = exclusive scan of `s_info[*].n` total +
    /// `s_info[num_subseq-1].n`. `Phase4Redecode` reads this in the
    /// kernel to bounds-check each `symbols_out` write — an
    /// adversarial / buggy upstream that inflates `n` past the
    /// thread's slot would otherwise corrupt host memory past
    /// `symbols_out`'s end. Set to 0 for other phases (kernel
    /// ignores it).
    pub total_symbols: u32,
    /// Which phase of the parallel-Huffman algorithm this dispatch
    /// executes.
    pub phase: HuffmanPhase,
}

/// Phase 4 per-subseq exit condition.
///
/// Encoded as `u32` in the kernel-side `decode_status` buffer.
/// Discriminants MUST match the kernel's `DECODE_*` constants
/// exactly (CUDA `#define`, Slang `static const uint`).
///
/// The Phase 4 dispatcher inspects `decode_status` after the kernel
/// returns and surfaces the first non-`Ok` subseq as a typed
/// `BackendError`. Adversarial inputs (truncated streams, codetable
/// mismatches) that would have silently produced shorter-than-
/// expected symbol streams now fail loudly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum Phase4FailureKind {
    /// Walker reached `end_p` cleanly (default-zero state).
    Ok = 0,
    /// 16-bit peek matched no codeword in the active table —
    /// codetable mismatch or corrupt bitstream.
    PrefixMiss = 1,
    /// Codeword would run past `length_bits` — stream truncated
    /// mid-codeword.
    LengthBits = 2,
    /// `max_iters` exhausted without reaching `end_p`. Should not
    /// happen with a valid codebook (every codeword advances `p`
    /// by ≥ 1 bit); kept as defense-in-depth for degenerate
    /// codebooks with zero-advance entries.
    Incomplete = 3,
    /// JPEG-framed phases only: DC magnitude category > 11.
    /// 8-bit baseline JPEG caps the DC bit-width at 11 per
    /// ITU-T T.81 § F.1.2.1.1; a corrupt DC codebook can emit
    /// 12..=255 and would direct the kernel to over-consume raw
    /// bits.
    BadDcCategory = 4,
    /// JPEG-framed phases only: AC magnitude size > 10.
    /// 8-bit baseline JPEG caps the AC bit-width at 10 per
    /// ITU-T T.81 § F.1.2.2.1; a corrupt AC codebook (low nibble
    /// of the symbol byte = 11..=15) would otherwise drive over-
    /// consumption.
    BadAcSize = 5,
    /// JPEG-framed phases only: AC run+size or ZRL would advance
    /// the per-block zig-zag cursor past slot 63 — spec-illegal,
    /// catches adversarial Huffman tables.
    AcOverflow = 6,
}

// Discriminants are wire values shared with the Slang `DECODE_*` and
// CUDA `#define DECODE_*` macros.  A drift here breaks the host's
// `decode_status` round-trip and is silent at runtime (the kernel
// writes a value the host can't recognise).  Pin the values at
// build time so any reorder of the enum surfaces as a compile
// error.
const _: () = assert!(Phase4FailureKind::Ok as u32 == 0);
const _: () = assert!(Phase4FailureKind::PrefixMiss as u32 == 1);
const _: () = assert!(Phase4FailureKind::LengthBits as u32 == 2);
const _: () = assert!(Phase4FailureKind::Incomplete as u32 == 3);
const _: () = assert!(Phase4FailureKind::BadDcCategory as u32 == 4);
const _: () = assert!(Phase4FailureKind::BadAcSize as u32 == 5);
const _: () = assert!(Phase4FailureKind::AcOverflow as u32 == 6);

impl Phase4FailureKind {
    /// Decode the kernel's u32 status code. Unknown values map to
    /// `Incomplete` since "kernel produced a value the host doesn't
    /// recognise" is most safely treated as a degenerate failure.
    #[must_use]
    pub const fn from_u32(v: u32) -> Self {
        // Unknown values fall through to `Incomplete` since "kernel
        // produced a value the host doesn't recognise" is most safely
        // treated as a degenerate failure rather than silently
        // mapping to Ok or panicking.
        match v {
            0 => Self::Ok,
            1 => Self::PrefixMiss,
            2 => Self::LengthBits,
            4 => Self::BadDcCategory,
            5 => Self::BadAcSize,
            6 => Self::AcOverflow,
            _ => Self::Incomplete,
        }
    }

    /// Human-readable label for error formatting.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Ok => "Ok",
            Self::PrefixMiss => "PrefixMiss",
            Self::LengthBits => "LengthBits",
            Self::Incomplete => "Incomplete",
            Self::BadDcCategory => "BadDcCategory",
            Self::BadAcSize => "BadAcSize",
            Self::AcOverflow => "AcOverflow",
        }
    }
}

/// Diagnostic kind shared by every `HuffmanParams::validate*` branch.
/// Hoisted out of the closure form so the helper methods can use it
/// directly and the kind constant lives in exactly one place.
const HUFFMAN_INVARIANT_KIND: &str = "HuffmanInvariantViolation";

/// Build a typed `InvariantViolation` for the Huffman params surface.
/// Replaces the per-method `let invariant = |detail| ...` closures so
/// the three validation helpers share one shape.
const fn huffman_invariant(detail: &'static str) -> super::BackendError {
    super::BackendError::InvariantViolation {
        kind: HUFFMAN_INVARIANT_KIND,
        detail,
    }
}

impl<B: GpuBackend + ?Sized> HuffmanParams<'_, B> {
    /// Number of subsequences = `ceil(length_bits / subsequence_bits)`.
    /// Backends use this to size the dispatch grid.
    #[must_use]
    pub const fn num_subsequences(&self) -> u32 {
        self.length_bits.div_ceil(self.subsequence_bits)
    }

    /// Validate the invariants documented on `HuffmanParams`.
    ///
    /// # Errors
    /// Returns a `BackendError::InvariantViolation` if any check fails.
    pub fn validate(&self, backend: &B) -> super::Result<()> {
        let invariant = huffman_invariant;
        if self.length_bits == 0 {
            return Err(invariant("length_bits must be > 0"));
        }
        if self.subsequence_bits == 0 {
            return Err(invariant("subsequence_bits must be > 0"));
        }
        if self.num_components == 0 {
            return Err(invariant("num_components must be ≥ 1"));
        }
        // Kernel computes `sync_target = start_bit + 2 * subsequence_bits`
        // in u32. With start_bit bounded by length_bits, the worst case
        // is `length_bits + 2 * subsequence_bits` (and that's only when
        // sync_target sits past the end of the stream, which it always
        // does for the last few subsequences). Reject inputs that would
        // wrap u32; the kernel's `min(length_bits, sync_target)` would
        // otherwise produce a garbage hard_limit.
        if self
            .length_bits
            .checked_add(2u32.saturating_mul(self.subsequence_bits))
            .is_none()
        {
            return Err(invariant(
                "length_bits + 2 * subsequence_bits overflows u32; kernel sync_target would wrap",
            ));
        }

        // bitstream capacity: words = ceil(length_bits/32), plus one
        // trailing word for peek16's two-word read.
        let stream_words = self.length_bits.div_ceil(32) as usize + 1;
        let stream_bytes = stream_words
            .checked_mul(4)
            .ok_or_else(|| invariant("bitstream byte count overflows usize"))?;
        if backend.device_buffer_len(self.bitstream) < stream_bytes {
            return Err(invariant(
                "bitstream buffer is smaller than ceil(length_bits/32)+1 words",
            ));
        }

        let codebook_bytes = (self.num_components as usize)
            .checked_mul(HUFFMAN_CODEBOOK_ENTRIES)
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| invariant("codebook byte count overflows usize"))?;
        if backend.device_buffer_len(self.codebook) < codebook_bytes {
            return Err(invariant(
                "codebook buffer is smaller than num_components * 65536 * 4 bytes",
            ));
        }

        // s_info: 4 u32 = 16 bytes per subsequence.
        let s_info_bytes = (self.num_subsequences() as usize)
            .checked_mul(16)
            .ok_or_else(|| invariant("s_info byte count overflows usize"))?;
        if backend.device_buffer_len(self.s_info) < s_info_bytes {
            return Err(invariant(
                "s_info buffer is smaller than num_subsequences * 16 bytes",
            ));
        }

        // Phase 2 sync_flags — required by the synthetic and
        // JPEG-framed counterparts alike.
        if matches!(
            self.phase,
            HuffmanPhase::Phase2InterSync | HuffmanPhase::JpegPhase2InterSync,
        ) {
            let Some(flags) = self.sync_flags else {
                return Err(invariant(
                    "Phase2InterSync (or JpegPhase2InterSync) requires sync_flags to be Some",
                ));
            };
            let flags_bytes = (self.num_subsequences() as usize)
                .checked_mul(4)
                .ok_or_else(|| invariant("sync_flags byte count overflows usize"))?;
            if backend.device_buffer_len(flags) < flags_bytes {
                return Err(invariant(
                    "sync_flags buffer is smaller than num_subsequences * 4 bytes",
                ));
            }
        }

        if matches!(
            self.phase,
            HuffmanPhase::Phase4Redecode | HuffmanPhase::JpegPhase4Redecode,
        ) {
            self.validate_phase4_buffers(backend)?;
        }

        if self.phase.is_jpeg_framed() {
            self.validate_jpeg_framed(backend)?;
        }

        Ok(())
    }

    /// Validate the Phase 4 (and `JpegPhase4Redecode`) buffer set:
    /// `offsets`, `symbols_out`, `decode_status`.  Pulled out of
    /// [`Self::validate`] so each branch stays under the clippy
    /// `too_many_lines` threshold and so the invariants are visible
    /// as a discrete block.
    fn validate_phase4_buffers(&self, backend: &B) -> super::Result<()> {
        let invariant = huffman_invariant;

        let Some(offsets) = self.offsets else {
            return Err(invariant(
                "Phase4Redecode (or JpegPhase4Redecode) requires offsets to be Some",
            ));
        };
        let Some(symbols_out) = self.symbols_out else {
            return Err(invariant(
                "Phase4Redecode (or JpegPhase4Redecode) requires symbols_out to be Some",
            ));
        };
        let Some(decode_status) = self.decode_status else {
            return Err(invariant(
                "Phase4Redecode (or JpegPhase4Redecode) requires decode_status to be Some",
            ));
        };
        // offsets + decode_status: 1 u32 per subsequence.
        let per_subseq_bytes = (self.num_subsequences() as usize)
            .checked_mul(4)
            .ok_or_else(|| invariant("per-subseq byte count overflows usize"))?;
        if backend.device_buffer_len(offsets) < per_subseq_bytes {
            return Err(invariant(
                "offsets buffer is smaller than num_subsequences * 4 bytes",
            ));
        }
        if backend.device_buffer_len(decode_status) < per_subseq_bytes {
            return Err(invariant(
                "decode_status buffer is smaller than num_subsequences * 4 bytes",
            ));
        }
        // symbols_out: 1 u32 per decoded symbol.
        let symbols_bytes = (self.total_symbols as usize)
            .checked_mul(4)
            .ok_or_else(|| invariant("symbols_out byte count overflows usize"))?;
        if backend.device_buffer_len(symbols_out) < symbols_bytes {
            return Err(invariant(
                "symbols_out buffer is smaller than total_symbols * 4 bytes",
            ));
        }
        Ok(())
    }

    /// Validate the JPEG-framed-only invariants: [`Self::dc_codebook`],
    /// [`Self::mcu_schedule`], [`Self::blocks_per_mcu`].  Separated
    /// from [`Self::validate`] to keep the main fn under the clippy
    /// `too_many_lines` threshold and to surface the JPEG-specific
    /// invariants as a discrete block.
    ///
    /// `codebook_bytes` is recomputed inside this helper so the caller
    /// does not have to pre-thread it; the cost is one multiply, and
    /// the alternative (passing it in) would let the caller silently
    /// hand the wrong size on a future refactor.
    fn validate_jpeg_framed(&self, backend: &B) -> super::Result<()> {
        let invariant = huffman_invariant;

        if self.blocks_per_mcu == 0 {
            return Err(invariant("JPEG-framed phases require blocks_per_mcu ≥ 1"));
        }
        let Some(dc) = self.dc_codebook else {
            return Err(invariant(
                "JPEG-framed phases require dc_codebook to be Some",
            ));
        };
        let codebook_bytes = (self.num_components as usize)
            .checked_mul(HUFFMAN_CODEBOOK_ENTRIES)
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| invariant("dc_codebook byte count overflows usize"))?;
        if backend.device_buffer_len(dc) < codebook_bytes {
            return Err(invariant(
                "dc_codebook buffer is smaller than num_components * 65536 * 4 bytes",
            ));
        }
        let Some(sched) = self.mcu_schedule else {
            return Err(invariant(
                "JPEG-framed phases require mcu_schedule to be Some",
            ));
        };
        let sched_bytes = (self.blocks_per_mcu as usize)
            .checked_mul(4)
            .ok_or_else(|| invariant("mcu_schedule byte count overflows usize"))?;
        if backend.device_buffer_len(sched) < sched_bytes {
            return Err(invariant(
                "mcu_schedule buffer is smaller than blocks_per_mcu * 4 bytes",
            ));
        }
        Ok(())
    }
}

/// Parameters for the IDCT + dequant + colour-conversion kernel (Phase 5).
///
/// The kernel runs one 8×8-thread workgroup per MCU block per component and
/// writes packed RGBA8 pixels to `pixels_rgba`.  The caller is responsible
/// for building `coefficients` from the symbol stream, running the DC
/// reconstruction, and packing quantisation tables in natural (row-major) order.
///
/// # Invariants
/// - `width > 0`, `height > 0`, `num_components ∈ {1, 3}`
/// - `blocks_wide == ceil(width / 8)`, `blocks_high == ceil(height / 8)`
/// - `num_qtables ∈ {1..=4}`
/// - `coefficients` capacity ≥ `num_components × blocks_wide × blocks_high × 64 × 4` bytes
/// - `qtables` capacity ≥ `num_qtables × 64 × 4` bytes
/// - `dc_values` capacity ≥ `num_components × blocks_wide × blocks_high × 4` bytes
/// - `pixels_rgba` capacity ≥ `width × height × 4` bytes
pub struct IdctParams<'a, B: GpuBackend + ?Sized> {
    /// Zigzag-ordered DCT coefficients from Phase 4 (one i32 per slot).
    /// Layout: `[comp][block_y][block_x][64]`.
    pub coefficients: &'a B::DeviceBuffer,
    /// Quantisation tables in natural (row-major) order.
    /// Layout: `[qt_index][64]` as i32.
    pub qtables: &'a B::DeviceBuffer,
    /// Absolute DC value per block (reconstructed by host before dispatch).
    /// Layout: `[comp][block_y][block_x]` as i32.
    pub dc_values: &'a B::DeviceBuffer,
    /// RGBA8 output buffer, row-major, `width × height` u32 pixels.
    pub pixels_rgba: &'a B::DeviceBuffer,
    /// Image width in pixels (> 0).
    pub width: u32,
    /// Image height in pixels (> 0).
    pub height: u32,
    /// Number of components: 1 (grayscale) or 3 (YCbCr).
    pub num_components: u32,
    /// `ceil(width / 8)`.
    pub blocks_wide: u32,
    /// `ceil(height / 8)`.
    pub blocks_high: u32,
    /// Number of quantisation tables present (1–4).
    pub num_qtables: u32,
}

impl<B: GpuBackend + ?Sized> IdctParams<'_, B> {
    /// Total blocks per component (`blocks_wide × blocks_high`).
    #[must_use]
    pub const fn blocks_per_component(&self) -> u32 {
        self.blocks_wide * self.blocks_high
    }

    /// Validate the invariants documented on [`IdctParams`].
    ///
    /// # Errors
    /// Returns a `BackendError::InvariantViolation` if any check fails.
    pub fn validate(&self, backend: &B) -> super::Result<()> {
        let inv = |detail: &'static str| super::BackendError::InvariantViolation {
            kind: "IdctInvariantViolation",
            detail,
        };
        if self.width == 0 {
            return Err(inv("width must be > 0"));
        }
        if self.height == 0 {
            return Err(inv("height must be > 0"));
        }
        if self.num_components != 1 && self.num_components != 3 {
            return Err(inv("num_components must be 1 (grayscale) or 3 (YCbCr)"));
        }
        if self.blocks_wide != self.width.div_ceil(8) {
            return Err(inv("blocks_wide must equal ceil(width / 8)"));
        }
        if self.blocks_high != self.height.div_ceil(8) {
            return Err(inv("blocks_high must equal ceil(height / 8)"));
        }
        if self.num_qtables == 0 || self.num_qtables > 4 {
            return Err(inv("num_qtables must be in 1..=4"));
        }
        let num_blocks =
            (self.num_components as usize).saturating_mul(self.blocks_per_component() as usize);
        let coef_bytes = num_blocks.saturating_mul(64 * 4);
        if backend.device_buffer_len(self.coefficients) < coef_bytes {
            return Err(inv(
                "coefficients buffer too small for num_components × blocks × 64 × 4",
            ));
        }
        let qt_bytes = (self.num_qtables as usize).saturating_mul(64 * 4);
        if backend.device_buffer_len(self.qtables) < qt_bytes {
            return Err(inv("qtables buffer too small for num_qtables × 64 × 4"));
        }
        let dc_bytes = num_blocks.saturating_mul(4);
        if backend.device_buffer_len(self.dc_values) < dc_bytes {
            return Err(inv(
                "dc_values buffer too small for num_components × blocks × 4",
            ));
        }
        let px_bytes = (self.width as usize)
            .saturating_mul(self.height as usize)
            .saturating_mul(4);
        if backend.device_buffer_len(self.pixels_rgba) < px_bytes {
            return Err(inv("pixels_rgba buffer too small for width × height × 4"));
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
        fn record_huffman(&self, _p: HuffmanParams<'_, Self>) -> super::super::Result<()> {
            unreachable!()
        }
        fn record_idct(&self, _p: IdctParams<'_, Self>) -> super::super::Result<()> {
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

    /// Discriminants must match the kernel-side DECODE_* constants
    /// exactly; the host downloads a `u32[num_subseq]` buffer from
    /// the device and round-trips each entry through `from_u32`. Any
    /// drift here would silently misclassify failures.
    #[test]
    fn phase4_failure_kind_from_u32_round_trips_known_codes() {
        assert_eq!(Phase4FailureKind::from_u32(0), Phase4FailureKind::Ok);
        assert_eq!(
            Phase4FailureKind::from_u32(1),
            Phase4FailureKind::PrefixMiss
        );
        assert_eq!(
            Phase4FailureKind::from_u32(2),
            Phase4FailureKind::LengthBits
        );
        assert_eq!(
            Phase4FailureKind::from_u32(3),
            Phase4FailureKind::Incomplete
        );
        assert_eq!(
            Phase4FailureKind::from_u32(4),
            Phase4FailureKind::BadDcCategory,
        );
        assert_eq!(Phase4FailureKind::from_u32(5), Phase4FailureKind::BadAcSize);
        assert_eq!(
            Phase4FailureKind::from_u32(6),
            Phase4FailureKind::AcOverflow,
        );
    }

    #[test]
    fn phase4_failure_kind_from_u32_maps_unknown_to_incomplete() {
        // Unknown values are most safely treated as a degenerate
        // failure rather than silently mapping to Ok or panicking.
        // Smallest unknown is one past the last defined variant.
        assert_eq!(
            Phase4FailureKind::from_u32(7),
            Phase4FailureKind::Incomplete
        );
        assert_eq!(
            Phase4FailureKind::from_u32(42),
            Phase4FailureKind::Incomplete
        );
        assert_eq!(
            Phase4FailureKind::from_u32(u32::MAX),
            Phase4FailureKind::Incomplete,
        );
    }

    #[test]
    fn phase4_failure_kind_labels_are_distinct() {
        let mut labels = vec![
            Phase4FailureKind::Ok.label(),
            Phase4FailureKind::PrefixMiss.label(),
            Phase4FailureKind::LengthBits.label(),
            Phase4FailureKind::Incomplete.label(),
            Phase4FailureKind::BadDcCategory.label(),
            Phase4FailureKind::BadAcSize.label(),
            Phase4FailureKind::AcOverflow.label(),
        ];
        let total = labels.len();
        labels.sort_unstable();
        labels.dedup();
        assert_eq!(
            labels.len(),
            total,
            "labels must be unique for error formatting",
        );
    }

    // --- HuffmanPhase + HuffmanParams validation tests ---------------

    #[test]
    fn huffman_phase_is_jpeg_framed_only_for_jpeg_variants() {
        assert!(!HuffmanPhase::Phase1IntraSync.is_jpeg_framed());
        assert!(!HuffmanPhase::Phase2InterSync.is_jpeg_framed());
        assert!(!HuffmanPhase::Phase4Redecode.is_jpeg_framed());
        assert!(HuffmanPhase::JpegPhase1IntraSync.is_jpeg_framed());
        assert!(HuffmanPhase::JpegPhase2InterSync.is_jpeg_framed());
        assert!(HuffmanPhase::JpegPhase4Redecode.is_jpeg_framed());
    }

    /// Minimal `HuffmanParams` suitable for synthetic-stream phases.
    /// Caller fills phase + per-phase Options.  Buffers are sized to
    /// the documented invariants for `length_bits = 256`,
    /// `subsequence_bits = 128`, `num_components = 1`.
    fn ok_huffman_synthetic<'a>(
        bitstream: &'a usize,
        codebook: &'a usize,
        s_info: &'a usize,
    ) -> HuffmanParams<'a, FakeBackend> {
        HuffmanParams {
            bitstream,
            codebook,
            s_info,
            sync_flags: None,
            offsets: None,
            symbols_out: None,
            decode_status: None,
            dc_codebook: None,
            mcu_schedule: None,
            length_bits: 256,
            subsequence_bits: 128,
            num_components: 1,
            total_symbols: 0,
            blocks_per_mcu: 0,
            phase: HuffmanPhase::Phase1IntraSync,
        }
    }

    #[test]
    fn huffman_validate_accepts_synthetic_phase1_with_no_jpeg_fields() {
        // 256 bits = 8 words, +1 trailing = 9 * 4 = 36 bytes.
        let bitstream = 36;
        // codebook: 1 component × 65536 × 4 = 262144 bytes.
        let codebook = 1 << 18;
        // s_info: ceil(256/128) = 2 subsequences × 16 bytes = 32.
        let s_info = 32;
        let params = ok_huffman_synthetic(&bitstream, &codebook, &s_info);
        params
            .validate(&FakeBackend)
            .expect("synthetic phase1 with sized buffers must validate");
    }

    #[test]
    fn huffman_validate_rejects_jpeg_phase_without_dc_codebook() {
        let bitstream = 36;
        let codebook = 1 << 18;
        let s_info = 32;
        let mut params = ok_huffman_synthetic(&bitstream, &codebook, &s_info);
        params.phase = HuffmanPhase::JpegPhase1IntraSync;
        params.blocks_per_mcu = 1;
        let mcu_schedule = 4;
        params.mcu_schedule = Some(&mcu_schedule);
        // dc_codebook intentionally None.
        let err = params
            .validate(&FakeBackend)
            .expect_err("JPEG-framed phase without dc_codebook must be rejected");
        match err {
            super::super::BackendError::InvariantViolation { detail, .. } => {
                assert!(
                    detail.contains("dc_codebook"),
                    "expected dc_codebook-related detail, got: {detail}"
                );
            }
            other => panic!("expected InvariantViolation, got: {other:?}"),
        }
    }

    #[test]
    fn huffman_validate_rejects_jpeg_phase_without_mcu_schedule() {
        let bitstream = 36;
        let codebook = 1 << 18;
        let s_info = 32;
        let dc_codebook = 1 << 18;
        let mut params = ok_huffman_synthetic(&bitstream, &codebook, &s_info);
        params.phase = HuffmanPhase::JpegPhase1IntraSync;
        params.blocks_per_mcu = 1;
        params.dc_codebook = Some(&dc_codebook);
        // mcu_schedule intentionally None.
        let err = params
            .validate(&FakeBackend)
            .expect_err("JPEG-framed phase without mcu_schedule must be rejected");
        match err {
            super::super::BackendError::InvariantViolation { detail, .. } => {
                assert!(
                    detail.contains("mcu_schedule"),
                    "expected mcu_schedule-related detail, got: {detail}"
                );
            }
            other => panic!("expected InvariantViolation, got: {other:?}"),
        }
    }

    #[test]
    fn huffman_validate_rejects_jpeg_phase_with_zero_blocks_per_mcu() {
        let bitstream = 36;
        let codebook = 1 << 18;
        let s_info = 32;
        let dc_codebook = 1 << 18;
        let mcu_schedule = 4;
        let mut params = ok_huffman_synthetic(&bitstream, &codebook, &s_info);
        params.phase = HuffmanPhase::JpegPhase1IntraSync;
        params.dc_codebook = Some(&dc_codebook);
        params.mcu_schedule = Some(&mcu_schedule);
        // blocks_per_mcu intentionally 0.
        let err = params
            .validate(&FakeBackend)
            .expect_err("blocks_per_mcu = 0 must be rejected on JPEG-framed phases");
        match err {
            super::super::BackendError::InvariantViolation { detail, .. } => {
                assert!(
                    detail.contains("blocks_per_mcu"),
                    "expected blocks_per_mcu-related detail, got: {detail}"
                );
            }
            other => panic!("expected InvariantViolation, got: {other:?}"),
        }
    }

    #[test]
    fn huffman_validate_accepts_jpeg_phase1_with_all_jpeg_fields() {
        let bitstream = 36;
        let codebook = 1 << 18;
        let s_info = 32;
        let dc_codebook = 1 << 18;
        // blocks_per_mcu = 1 × 4 = 4 bytes.
        let mcu_schedule = 4;
        let mut params = ok_huffman_synthetic(&bitstream, &codebook, &s_info);
        params.phase = HuffmanPhase::JpegPhase1IntraSync;
        params.dc_codebook = Some(&dc_codebook);
        params.mcu_schedule = Some(&mcu_schedule);
        params.blocks_per_mcu = 1;
        params
            .validate(&FakeBackend)
            .expect("JPEG-framed phase1 with all required fields must validate");
    }

    #[test]
    fn huffman_validate_rejects_undersized_dc_codebook() {
        let bitstream = 36;
        let codebook = 1 << 18;
        let s_info = 32;
        // dc_codebook one byte too small.
        let dc_codebook = (1 << 18) - 1;
        let mcu_schedule = 4;
        let mut params = ok_huffman_synthetic(&bitstream, &codebook, &s_info);
        params.phase = HuffmanPhase::JpegPhase1IntraSync;
        params.dc_codebook = Some(&dc_codebook);
        params.mcu_schedule = Some(&mcu_schedule);
        params.blocks_per_mcu = 1;
        let err = params
            .validate(&FakeBackend)
            .expect_err("under-sized dc_codebook must be rejected");
        match err {
            super::super::BackendError::InvariantViolation { detail, .. } => {
                assert!(
                    detail.contains("dc_codebook") && detail.contains("smaller"),
                    "expected dc_codebook size-check detail, got: {detail}"
                );
            }
            other => panic!("expected InvariantViolation, got: {other:?}"),
        }
    }

    #[test]
    fn huffman_validate_rejects_undersized_mcu_schedule() {
        let bitstream = 36;
        let codebook = 1 << 18;
        let s_info = 32;
        let dc_codebook = 1 << 18;
        // blocks_per_mcu = 4 needs 16 bytes; provide 12.
        let mcu_schedule = 12;
        let mut params = ok_huffman_synthetic(&bitstream, &codebook, &s_info);
        params.phase = HuffmanPhase::JpegPhase1IntraSync;
        params.dc_codebook = Some(&dc_codebook);
        params.mcu_schedule = Some(&mcu_schedule);
        params.blocks_per_mcu = 4;
        let err = params
            .validate(&FakeBackend)
            .expect_err("under-sized mcu_schedule must be rejected");
        match err {
            super::super::BackendError::InvariantViolation { detail, .. } => {
                assert!(
                    detail.contains("mcu_schedule") && detail.contains("smaller"),
                    "expected mcu_schedule size-check detail, got: {detail}"
                );
            }
            other => panic!("expected InvariantViolation, got: {other:?}"),
        }
    }
}
