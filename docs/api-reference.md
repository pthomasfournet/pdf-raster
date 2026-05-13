# API Reference

## `pdf_raster` crate — public API

The `pdf_raster` crate is the integration entry point. It wraps `pdf_interp` and `raster` behind a simple, stable API.

---

### `raster_pdf`

```rust
pub fn raster_pdf(
    path: &Path,
    opts: &RasterOptions,
) -> impl Iterator<Item = (u32, Result<RenderedPage, RasterError>)>
```

Renders a range of pages from a PDF file. Returns an iterator that yields `(page_num, result)` for each page in `opts.first_page..=opts.last_page`.

**Behaviour:**

- Pages are rendered in ascending order.
- A per-page error does not abort remaining pages. The caller decides whether to skip or propagate.
- If `opts.last_page` exceeds the document's page count, rendering stops at the last page — no error is returned for the overshoot.
- GPU resources are initialised lazily on first use and reused across pages.

**Errors (yielded as iterator items):**

- `RasterError::InvalidOptions` — `opts` violates constraints (e.g. `dpi ≤ 0`, `first_page > last_page`). Yielded as `(1, Err(...))`, iterator ends immediately.
- `RasterError::Pdf` — document cannot be opened or parsed. Yielded as `(1, Err(...))`, iterator ends immediately.
- `RasterError::PageOutOfRange` — page number exceeds document length.
- `RasterError::PageDegenerate` — page has zero pixel dimensions.
- `RasterError::PageTooLarge` — pixel dimensions exceed `MAX_PX_DIMENSION` (32 768).
- `RasterError::InvalidPageGeometry` — `UserUnit` outside `[0.1, 10.0]`.
- `RasterError::Deskew` — deskew rotation failed (rare; graceful fallback applied when possible).

---

### `render_channel`

```rust
pub fn render_channel(
    path: &Path,
    opts: &RasterOptions,
    capacity: usize,
) -> std::sync::mpsc::Receiver<(u32, Result<RenderedPage, RasterError>)>
```

Renders pages concurrently in a background Rayon task, streaming results through a bounded sync channel.

**`capacity`** — maximum number of rendered pages buffered before the producer blocks (natural backpressure). `capacity = 0` is silently raised to `1`. Use `2`–`8` for typical OCR pipelines.

**Error contract:**

- Options validation failure → `(1, Err(RasterError::InvalidOptions(...)))`, channel closes.
- File open failure → `(1, Err(RasterError::Pdf(...)))`, channel closes.
- Per-page failures → `(page_num, Err(...))`, remaining pages continue.

If the `Receiver` is dropped before the producer finishes, the producer exits cleanly on its next `send` — no panic.

---

### `pdf_raster::session` module

Lower-level API for explicit control over PDF opening and per-page rendering. All items are also re-exported at the crate root for backward compatibility.

```rust
pub mod session {
    pub use super::{
        open_session, prescan_session, render_page_rgb,
        render_page_rgb_hinted, rgb_to_gray,
    };
}
```

```rust
pub fn open_session(path: &Path, config: &SessionConfig) -> Result<RasterSession, RasterError>
```

Opens the PDF and builds an O(1) page-ID map. Also initialises the shared GPU context (for `gpu-aa` / `gpu-icc` features) according to `config.policy`. Errors with `RasterError::Pdf` if the file is unreadable, corrupt, or contains JavaScript. Errors with `RasterError::BackendUnavailable` if `config.policy` is `ForceCuda` or `ForceVaapi` and the required GPU cannot be initialised.

```rust
pub fn render_page_rgb(
    session: &RasterSession,
    page_num: u32,
    scale: f64,
) -> Result<Bitmap<Rgb8>, RasterError>
```

Renders one page to an RGB bitmap. `scale` is the pixel-per-point multiplier: `dpi / 72.0` for uniform DPI, or `sqrt((rx/72) × (ry/72))` for non-square pixels.

Safe to call from multiple Rayon threads concurrently. GPU image decoders are managed per-thread via `thread_local!`.

```rust
pub fn rgb_to_gray(src: &Bitmap<Rgb8>) -> Bitmap<Gray8>
```

BT.709 luminance conversion: `Y = 0.2126·R + 0.7152·G + 0.0722·B`.

```rust
pub fn prescan_session(
    session: &RasterSession,
    page_num: u32,
) -> Result<PageDiagnostics, RasterError>
```

Classifies page `page_num` without rendering any pixels. Returns `PageDiagnostics` with image/text presence, dominant filter, and a PPI hint. Use this before `render_page_rgb` to choose the backend policy (e.g. skip GPU init for vector-only pages).

Errors with `RasterError::PageOutOfRange` or `RasterError::InvalidPageGeometry` if the page is invalid.

---

### `RasterSession`

```rust
pub struct RasterSession { /* opaque */ }

impl RasterSession {
    pub const fn total_pages(&self) -> u32
    pub const fn policy(&self) -> BackendPolicy
    // doc() and resolve_page() are #[doc(hidden)] — use prescan_session instead
}
```

An opened, read-only document. `Sync + Send` — safe to share across Rayon threads.

---

### `RasterOptions`

```rust
#[derive(Debug, Clone)]
pub struct RasterOptions {
    pub dpi: f32,
    pub first_page: u32,
    pub last_page: u32,
    pub deskew: bool,
    pub pages: Option<PageSet>,
}
```

| Field | Constraints | Notes |
|---|---|---|
| `dpi` | `> 0`, finite | Render resolution. Pass `effective_dpi` (not `dpi`) to Tesseract. |
| `first_page` | `≥ 1` | 1-based inclusive. Ignored when `pages` is `Some`. |
| `last_page` | `≥ first_page` | 1-based inclusive. Clamped to document length silently. Ignored when `pages` is `Some`. |
| `deskew` | — | Applies intensity-weighted projection-profile deskew (±7°, sub-0.05° accuracy). Disable for native-text PDFs. |
| `pages` | — | When `Some`, only the pages in the `PageSet` are rendered; `first_page`/`last_page` are ignored. |

**Default:** `RasterOptions` implements `Default`:

```rust
// Short form (dpi only, everything else defaults)
let opts = RasterOptions { dpi: 150.0, ..RasterOptions::default() };
```

Defaults: `dpi = 300.0`, `first_page = 1`, `last_page = u32::MAX`, `deskew = false`, `pages = None`.

---

### `PageSet`

```rust
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PageSet(/* opaque Arc<[u32]> */);

impl PageSet {
    pub fn new(pages: impl IntoIterator<Item = u32>) -> Result<Self, RasterError>
    pub fn contains(&self, page: u32) -> bool  // O(log n)
    pub fn first(&self) -> u32
    pub fn last(&self) -> u32
    pub fn len(&self) -> usize
    pub fn is_empty(&self) -> bool  // always false by invariant
}
```

A validated, sorted, deduplicated set of 1-based page numbers. The internal storage is reference-counted (`Arc<[u32]>`); `clone()` is O(1). Used in `RasterOptions::pages` to render a sparse subset of pages without visiting intermediate ones.

**`PageSet::new`** accepts any `IntoIterator<Item = u32>` (Vec, array, slice, range). The input is sorted and deduplicated before storage. Returns `RasterError::InvalidOptions` if the resulting set is empty or contains a zero page number.

```rust
let pages = PageSet::new(vec![1, 5, 23, 100])?;
let opts = RasterOptions {
    dpi: 300.0,
    first_page: 1,       // ignored — PageSet controls the range
    last_page: u32::MAX, // ignored — PageSet controls the range
    deskew: true,
    pages: Some(pages),
};
```

---

### `RenderedPage`

```rust
pub struct RenderedPage {
    pub page_num: u32,
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
    pub dpi: f32,
    pub effective_dpi: f32,
    pub diagnostics: PageDiagnostics,
}
```

| Field | Notes |
|---|---|
| `page_num` | 1-based page number. |
| `width`, `height` | Output bitmap dimensions in pixels. |
| `pixels` | 8-bit grayscale, tightly packed (`stride == width`), top-to-bottom. Length is exactly `width * height`. |
| `dpi` | Raw render DPI (`opts.dpi`). |
| `effective_dpi` | `opts.dpi × UserUnit`. Pass this to `tesseract::set_source_resolution`. For the vast majority of documents `UserUnit = 1.0` and this equals `dpi`. |
| `diagnostics` | Lightweight rendering metadata — see `PageDiagnostics`. |

### `PageDiagnostics`

```rust
pub use pdf_interp::renderer::PageDiagnostics;
```

Collected at zero extra cost during rendering.

| Field | Type | Notes |
|---|---|---|
| `has_images` | `bool` | At least one image XObject or inline image was rendered. |
| `has_vector_text` | `bool` | At least one text-showing operator (`Tj`, `TJ`, `'`, `"`) was executed. `false` on scan-only pages. |
| `dominant_filter` | `Option<ImageFilter>` | Most common image decode filter on this page (`None` for pure-vector pages). |
| `source_ppi_hint` | `Option<f32>` | Estimated native pixels-per-inch of the dominant image. Computed as `(image_width_px / page_width_pts) × 72`. `None` when no images were blitted. |

#### `PageDiagnostics::suggested_dpi`

```rust
pub fn suggested_dpi(&self, min_dpi: f32, max_dpi: f32) -> Option<f32>
```

Suggests a re-render DPI based on `source_ppi_hint`. Returns `None` for vector/text-only pages. Snaps to the nearest standard step (72, 96, 150, 200, 300, 400, 600) and clamps to `[min_dpi, max_dpi]`.

```rust
if let Some(native_dpi) = page.diagnostics.suggested_dpi(150.0, 600.0) {
    if (native_dpi - opts.dpi).abs() > 10.0 {
        // Re-render at native resolution
    }
}
```

Use `diagnostics` to route pages to different OCR configurations:

```rust
// Page looks like a scan (has images, no vector text)
if page.diagnostics.has_images && !page.diagnostics.has_vector_text {
    // Sauvola handles uneven scan backgrounds better than Otsu
    tesseract.set_variable("thresholding_method", "2");
}

// Avoid deskew overhead on native-text pages
if !page.diagnostics.has_images && page.diagnostics.has_vector_text {
    // deskew: false for this page
}
```

---

### `BackendPolicy`

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BackendPolicy {
    Auto,        // GPU when available, silent CPU fallback (default)
    CpuOnly,     // Skip all GPU init entirely
    ForceCuda,   // Require CUDA; error if unavailable
    #[cfg(feature = "vaapi")]
    ForceVaapi,  // Require VA-API JPEG; error if unavailable — only when `vaapi` feature is enabled
    ForceVulkan, // Require the Vulkan compute backend; error if unavailable
                 // (or if the binary was built without `--features vulkan`)
}
```

Controls which compute backend is used. `Auto` matches pre-v0.4.0 behaviour. The `Force*` variants convert silent GPU fallbacks into hard `RasterError::BackendUnavailable` errors so you know immediately whether the expected hardware path is actually active. `ForceVaapi` is only present when the `vaapi` Cargo feature is enabled.

`ForceVulkan` runs the AA-fill, tile-fill, and parallel-Huffman JPEG decode kernels on the Vulkan compute backend (cross-vendor: NVIDIA, AMD, Intel, Apple via `MoltenVK`).  The device-resident image cache is CUDA-only, so under `ForceVulkan` the renderer runs uncached; ICC CMYK→RGB stays on the CPU AVX-512 fallback.  JPEG dispatch goes through the GPU parallel-Huffman path. The path is dormant by default (`GPU_JPEG_HUFFMAN_THRESHOLD_PX = u32::MAX`); enable it by setting `PDF_RASTER_HUFFMAN_THRESHOLD=0` at runtime.

---

### `SessionConfig`

```rust
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct SessionConfig {
    pub policy: BackendPolicy,
    pub vaapi_device: String,  // default: "/dev/dri/renderD128"
    // `prefetch: bool` exists only when `cache` feature is enabled
}

impl Default for SessionConfig { /* Auto policy, default DRM node */ }
```

Passed to `open_session`. `SessionConfig` is `#[non_exhaustive]` — construct it via `SessionConfig::default()` or `SessionConfig::with_policy(policy)`, not a struct literal. This is required because the `prefetch` field only exists when the `cache` Cargo feature is enabled. Set `vaapi_device` to override the VA-API DRM render node (useful when `/dev/dri/renderD128` is not the correct device on your system).

---

### `deskew` module

The `deskew` module exposes a limited public API:

- `deskew::apply` — applies a deskew angle to a bitmap.
- `deskew::DeskewError` — error type returned by `apply`.

The `deskew::detect` and `deskew::rotate` submodules are `pub(crate)` and are not part of the public API.

---

### `release_gpu_decoders`

```rust
pub fn release_gpu_decoders()
```

Eagerly drops GPU decoder state (nvJPEG, nvJPEG2000, Vulkan Huffman) on the calling rayon worker thread. Call via `pool.broadcast` before dropping the pool:

```rust
let _ = pool.broadcast(|_| pdf_raster::release_gpu_decoders());
drop(pool);
```

No-op when none of the GPU decoder features (`nvjpeg`, `nvjpeg2k`, `gpu-jpeg-huffman + vulkan`) are compiled in. Never panics.

---

### `RasterError`

```rust
pub enum RasterError {
    InvalidOptions(String),
    Pdf(pdf_interp::InterpError),
    PageOutOfRange { page: u32, total: u32 },
    PageDegenerate { width: u32, height: u32 },
    PageTooLarge { width: u32, height: u32 },
    Deskew(String),
    InvalidPageGeometry(String),
    BackendUnavailable(String),  // forced backend failed to init
}
```

Implements `std::error::Error` with a `source()` chain. `RasterError::Pdf(e)` has `e` as its source for chained error reporting. `BackendUnavailable` is only returned when `SessionConfig.policy` is `ForceCuda` or `ForceVaapi`.

---

### `MAX_PX_DIMENSION`

```rust
pub const MAX_PX_DIMENSION: u32 = 32_768;
```

Maximum accepted pixel dimension (width or height). `PageTooLarge` is returned if either dimension exceeds this. At 150 DPI this corresponds to ~366 inches (~9.3 metres).

---

### Re-exports

```rust
pub use pdf_interp::renderer::PageDiagnostics;
pub use pdf_interp::resources::ImageFilter;
```

`ImageFilter` identifies which decode filter was used for an embedded image (DCTDecode, JPXDecode, FlateDecode, etc.). Available through `PageDiagnostics` for routing decisions.

---

## `pdf_interp` crate — lower-level API

Direct use of `pdf_interp` is not required for most consumers. Use it when building a custom render loop or accessing document metadata without rendering.

### `open`

```rust
pub fn open(path: impl AsRef<Path>) -> Result<pdf::Document, InterpError>
```

Opens and validates a PDF. Returns `InterpError::JavaScript` immediately if any JS entry point is detected (checked locations: `/OpenAction`, `/AA`, `/Names/JavaScript`, `/AcroForm/AA`). No JS is parsed or evaluated.

The returned `pdf::Document` comes from the in-tree `pdf` crate (replaced lopdf 0.40 in v0.6.0). It is a lazy, mmap-backed reader: opening the file parses only the xref table and trailer; individual objects resolve on demand via byte-offset seek with a per-object `Arc` cache shared safely across worker threads. Object stream (`ObjStm`) decompression is cached once and reused. The API mirrors the lopdf method names previously used here.

### `page_count`

```rust
pub fn page_count(doc: &Document) -> u32
```

Total pages. Saturates at `u32::MAX` for pathological documents (> 4 billion pages).

### `page_size_pts`

```rust
pub fn page_size_pts(doc: &Document, page_num: u32) -> Result<PageGeometry, InterpError>
```

Returns geometry for page `page_num` (1-based). `width_pts` and `height_pts` are already adjusted for rotation and `UserUnit` scaling — use them directly as output bitmap dimensions.

Falls back to US Letter (612 × 792 pt) when neither `CropBox` nor `MediaBox` can be read.

### `parse_page`

```rust
pub fn parse_page(doc: &Document, page_num: u32) -> Result<Vec<Operator>, InterpError>
```

Parses the content stream for page `page_num` and returns the decoded operator sequence. Typically called internally by the renderer; exposed for tooling (e.g. `dump_ops` example).

### `PageGeometry`

```rust
pub struct PageGeometry {
    pub width_pts: f64,   // output width in PDF points (rotation + UserUnit applied)
    pub height_pts: f64,  // output height in PDF points (rotation + UserUnit applied)
    pub rotate_cw: u16,   // 0, 90, 180, or 270
    pub user_unit: f64,   // UserUnit scale factor, validated to [0.1, 10.0]
}
```

Dimensions are swapped for 90°/270° rotations so that `width_pts` always corresponds to the horizontal extent of the rendered bitmap.

To get pixel dimensions: `(width_pts × dpi / 72.0).round()`.

### `InterpError`

```rust
pub enum InterpError {
    Pdf(pdf::PdfError),
    PageOutOfRange { page: u32, total: u32 },
    MissingResource(String),
    JavaScript { location: &'static str },
    InvalidPageGeometry(String),
}
```

Implements `std::error::Error`. `InterpError::Pdf(e)` chains to `pdf::PdfError`.

---

## Hardware compatibility

### CPU

**Supported:** x86-64 (AMD and Intel) and `aarch64` (ARM / Apple Silicon).

**x86-64:**
- AVX2 SIMD blend and fill paths are runtime-detected with a scalar fallback.
- AVX-512 (`avx512f/bw/vl/dq/vnni/vpopcntdq` and related sub-extensions) is activated by building with `-C target-cpu=native` on a compatible CPU. Developed and benchmarked on an AMD Ryzen 9900X3D.
- All Intel consumer CPUs (Alder/Raptor/Arrow Lake) have AVX2; AVX-512 is disabled on them — Xeon only.

**aarch64:**
- NEON is used unconditionally (mandatory on all ARMv8-A). No runtime detection needed.
- SVE2 (`svcnt_u8_z` popcount tier) is available behind the `nightly-sve2` Cargo feature on nightly Rust. Gives up to 4× NEON throughput on wide-SVE2 server chips (Graviton4 full width).
- `cargo check --target aarch64-unknown-linux-gnu` is clean; no Apple Metal native backend yet (Vulkan via `MoltenVK` covers Apple).

### GPU

**NVIDIA (CUDA 12 or 13):**

| Feature flag | Minimum requirement | Notes |
|---|---|---|
| `nvjpeg` | CUDA-capable NVIDIA GPU | `libnvjpeg.so` ships with CUDA 12 or 13 toolkit |
| `nvjpeg2k` | CUDA-capable NVIDIA GPU | `libnvjpeg2k.so` is a separate download; build script probes `/13` then `/12` |
| `gpu-aa` | CUDA-capable NVIDIA GPU | CUDA runtime only |
| `gpu-icc` | CUDA-capable NVIDIA GPU | CUDA runtime only |
| `gpu-deskew` | CUDA-capable NVIDIA GPU | Requires CUDA NPP: `libnppig.so` + `libnppc.so` |
| `cache` | CUDA-capable NVIDIA GPU | Phase 9 3-tier image cache (CUDA-only) |

`cudarc` is pinned to the `cuda-12080` driver-API binding; the same source builds against both 12.x and 13.x drivers.

**Vulkan compute (cross-vendor — NVIDIA, AMD, Intel, Apple via `MoltenVK`):**

| Feature flag | Supported hardware | Libraries required |
|---|---|---|
| `vulkan` | Any Vulkan 1.3+ device | Vulkan ICD (e.g. `mesa-vulkan-drivers`, NVIDIA driver); `slangc` from the LunarG Vulkan SDK at *build* time. Implies `gpu-aa`. |

Vulkan covers the AA-fill, tile-fill, and parallel-Huffman JPEG decode kernels (`vulkan` implies `gpu-jpeg-huffman`).  `cache` and `nvjpeg` stay CUDA-only; under `--backend vulkan` the renderer runs uncached but JPEG decode routes through the GPU parallel-Huffman path.

**VA-API (Linux iGPU/dGPU — AMD VCN, Intel Quick Sync, Intel Arc):**

| Feature flag | Supported hardware | Libraries required |
|---|---|---|
| `vaapi` | AMD VCN (Raphael+), Intel UHD 630+, Intel Arc | `libva.so.2`, `libva-drm.so.2` |

VA-API provides hardware JPEG baseline decode. CMYK and progressive JPEGs fall through to CPU. When both `nvjpeg` and `vaapi` are active, nvJPEG takes priority; VA-API fires as fallback.

GPU initialisation failures at runtime print a warning to stderr and fall back to CPU — no error is returned, rendering continues normally.

### Platform support

| Platform | CPU SIMD | GPU acceleration | Status |
|---|---|---|---|
| x86-64 AMD (Ryzen) | AVX-512 + AVX2 | NVIDIA CUDA + AMD VA-API + Vulkan | **Supported** |
| x86-64 Intel (consumer) | AVX2 | NVIDIA CUDA + Intel VA-API + Vulkan | **Supported** |
| x86-64 Intel (Xeon) | AVX-512 + AVX2 | NVIDIA CUDA + Intel VA-API + Vulkan | **Supported** |
| aarch64 Linux (Graviton, RPi) | NEON + SVE2 † | Vulkan (Mesa) | CPU full, Vulkan untested on aarch64 |
| Apple Silicon (M1–M4) | NEON | Vulkan via `MoltenVK` (untested) | CPU full, Vulkan untested |
| AMD/Radeon ROCm | — | — | Not implemented (Vulkan covers Radeon) |
| Apple Metal (native) | — | — | Not implemented (Vulkan via `MoltenVK` is the path) |

† SVE2 requires `nightly-sve2` Cargo feature and nightly Rust.

## Feature flags

### `pdf_raster` features

| Feature | Requires | Effect |
|---|---|---|
| `nvjpeg` | CUDA 12 or 13, `libnvjpeg.so` | GPU JPEG decode (DCTDecode). Falls back to CPU zune-jpeg below 512×512 px. |
| `nvjpeg2k` | CUDA 12 or 13, `libnvjpeg2k.so` | GPU JPEG 2000 decode (JPXDecode). Falls back to CPU OpenJPEG below 512×512 px or for sub-sampled chroma. |
| `gpu-aa` | CUDA 12 or 13 | GPU supersampled AA fill (64-sample warp-ballot kernel). Falls back to CPU 4× scanline AA below 256 px. |
| `gpu-icc` | CUDA 12 or 13 | GPU ICC CMYK→RGB via 4D CLUT. Falls back to CPU AVX-512 matrix formula below 500 000 px. |
| `gpu-deskew` | CUDA 12 or 13, CUDA NPP | GPU bilinear rotation (nppiRotate). Falls back to CPU bilinear when disabled. |
| `cache` | CUDA 12 or 13 | Phase 9 device-resident image cache (3-tier VRAM/host/disk). Cross-document content-hash dedup. CUDA-only; no Vulkan support today. Disk-tier persistence is opt-in via `PDF_RASTER_CACHE_DIR`. |
| `vaapi` | `libva.so.2`, `libva-drm.so.2` | VA-API JPEG baseline decode on Linux iGPU/dGPU. Falls back to CPU on CMYK/progressive JPEG. When `nvjpeg` is also active, nvJPEG takes priority. |
| `vulkan` | Vulkan 1.3+ ICD; LunarG `slangc` at build time. Implies `gpu-aa` and `gpu-jpeg-huffman`. | Vulkan compute backend. AA-fill, tile-fill, and parallel-Huffman JPEG decode kernels run on any Vulkan 1.3+ device (NVIDIA, AMD, Intel, Apple via `MoltenVK`). No nvJPEG / `cache` support under this backend. |
| `gpu-validation` | CUDA device at test time | Enables GPU vs CPU parity tests (`cargo test -p gpu --features gpu-validation`). |

GPU initialisation failures print a warning to stderr and fall back to CPU — they do not return errors.  `cudarc` is pinned to the `cuda-12080` driver-API binding so the same source builds against both 12.x and 13.x drivers (forward-compatible per the CUDA driver-API ABI).

### GPU dispatch thresholds

| Path | Threshold | Constant |
|---|---|---|
| nvJPEG (DCTDecode) | ≥ 512×512 px | `GPU_JPEG_THRESHOLD_PX` |
| nvJPEG2000 (JPXDecode) | ≥ 512×512 px | `GPU_JPEG2K_THRESHOLD_PX` |
| GPU AA fill | ≥ 256 px (longest edge) | `GPU_AA_FILL_THRESHOLD` |
| GPU tile fill | ≥ 256 px (longest edge) | `GPU_TILE_FILL_THRESHOLD` |
| GPU ICC CLUT | ≥ 500 000 px (area) | `GPU_ICC_CLUT_THRESHOLD` |

Fill dispatch order: GPU tile fill → GPU AA fill → CPU scanline AA.
