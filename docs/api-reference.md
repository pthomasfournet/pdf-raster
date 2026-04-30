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

### `open_session` / `render_page_rgb`

Lower-level API for parallel consumers (e.g. the CLI uses this with Rayon).

```rust
pub fn open_session(path: &Path) -> Result<RasterSession, RasterError>
```

Opens the PDF and builds an O(1) page-ID map. Also initialises the shared GPU context (for `gpu-aa` / `gpu-icc` features). Errors with `RasterError::Pdf` if the file is unreadable, corrupt, or contains JavaScript.

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

---

### `RasterSession`

```rust
pub struct RasterSession { /* opaque */ }

impl RasterSession {
    pub const fn total_pages(&self) -> u32
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
}
```

| Field | Constraints | Notes |
|---|---|---|
| `dpi` | `> 0`, finite | Render resolution. Pass `effective_dpi` (not `dpi`) to Tesseract. |
| `first_page` | `≥ 1` | 1-based inclusive. |
| `last_page` | `≥ first_page` | 1-based inclusive. Clamped to document length silently. |
| `deskew` | — | Applies intensity-weighted projection-profile deskew (±7°, sub-0.05° accuracy). Disable for native-text PDFs. |

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

#### `RenderedPage::suggested_dpi`

```rust
pub fn suggested_dpi(&self, min_dpi: f32, max_dpi: f32) -> Option<f32>
```

Suggests a re-render DPI based on the native resolution of embedded raster images. Returns `None` for vector/text-only pages (use your default DPI). Clamped to `[min_dpi, max_dpi]`.

---

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
}
```

Implements `std::error::Error` with a `source()` chain. `RasterError::Pdf(e)` has `e` as its source for chained error reporting.

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
pub fn open(path: impl AsRef<Path>) -> Result<lopdf::Document, InterpError>
```

Opens and validates a PDF. Returns `InterpError::JavaScript` immediately if any JS entry point is detected (checked locations: `/OpenAction`, `/AA`, `/Names/JavaScript`, `/AcroForm/AA`). No JS is parsed or evaluated.

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
    Pdf(lopdf::Error),
    PageOutOfRange { page: u32, total: u32 },
    MissingResource(String),
    JavaScript { location: &'static str },
    InvalidPageGeometry(String),
}
```

Implements `std::error::Error`. `InterpError::Pdf(e)` chains to `lopdf::Error`.

---

## Feature flags

### `pdf_raster` features

| Feature | Requires | Effect |
|---|---|---|
| `nvjpeg` | CUDA 12, `libnvjpeg.so` | GPU JPEG decode (DCTDecode). Falls back to CPU zune-jpeg below 512×512 px. |
| `nvjpeg2k` | CUDA 12, `libnvjpeg2k.so` | GPU JPEG 2000 decode (JPXDecode). Falls back to CPU OpenJPEG below 512×512 px or for sub-sampled chroma. |
| `gpu-aa` | CUDA 12 | GPU supersampled AA fill (64-sample warp-ballot kernel). Falls back to CPU 4× scanline AA below 256 px. |
| `gpu-icc` | CUDA 12 | GPU ICC CMYK→RGB via 4D CLUT. Falls back to CPU AVX-512 matrix formula below 500 000 px. |
| `gpu-deskew` | CUDA 12, CUDA NPP | GPU bilinear rotation (nppiRotate). Falls back to CPU bilinear when disabled. |
| `gpu-validation` | CUDA device at test time | Enables GPU vs CPU parity tests (`cargo test -p gpu --features gpu-validation`). |

GPU initialisation failures print a warning to stderr and fall back to CPU — they do not return errors.

### GPU dispatch thresholds

| Path | Threshold | Constant |
|---|---|---|
| nvJPEG (DCTDecode) | ≥ 512×512 px | `GPU_JPEG_THRESHOLD_PX` |
| nvJPEG2000 (JPXDecode) | ≥ 512×512 px | `GPU_JPEG2K_THRESHOLD_PX` |
| GPU AA fill | ≥ 256 px (longest edge) | `GPU_AA_FILL_THRESHOLD` |
| GPU tile fill | ≥ 256 px (longest edge) | `GPU_TILE_FILL_THRESHOLD` |
| GPU ICC CLUT | ≥ 500 000 px (area) | `GPU_ICC_CLUT_THRESHOLD` |

Fill dispatch order: GPU tile fill → GPU AA fill → CPU scanline AA.
