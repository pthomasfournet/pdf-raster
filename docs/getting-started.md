# Getting Started

pdf-raster converts PDF pages to pixel buffers in pure Rust. No subprocesses, no Leptonica, no Poppler in the render path.

## What it does

Given a PDF file, pdf-raster renders each page to an 8-bit grayscale buffer in memory. The primary use case is feeding pages directly into Tesseract OCR without writing any intermediate files.

## Hardware requirements

### CPU

**Supported:** x86-64 processors with AMD or Intel silicon.

- AVX2 is used when available (runtime-detected, scalar fallback otherwise).
- AVX-512 (`avx512f/bw/vl/dq/vnni/vpopcntdq` and related extensions) is used when the binary is built with `-C target-cpu=native` on a compatible CPU. The project is developed and benchmarked on an AMD Ryzen 9900X3D.
- **ARM / Apple Silicon are not supported.** NEON SIMD paths have not been implemented. The code will not compile for `aarch64` targets. There is no Apple Metal backend.

### GPU (optional)

All GPU features require an **NVIDIA GPU with CUDA 12**.

| Feature | Minimum GPU | Library required |
|---|---|---|
| `nvjpeg` | Any CUDA 12-capable NVIDIA GPU | `libnvjpeg.so` (ships with CUDA 12) |
| `nvjpeg2k` | Any CUDA 12-capable NVIDIA GPU | `libnvjpeg2k.so` (separate download) |
| `gpu-aa` | Any CUDA 12-capable NVIDIA GPU | CUDA runtime |
| `gpu-icc` | Any CUDA 12-capable NVIDIA GPU | CUDA runtime |
| `gpu-deskew` | Any CUDA 12-capable NVIDIA GPU | CUDA NPP (`libnppig.so`, `libnppc.so`) |

**AMD/Radeon GPUs are not supported.** ROCm / HIP backends have not been implemented.

**Intel GPUs (Arc, Iris Xe) are not supported.** There is no oneAPI / Level Zero backend.

If no NVIDIA GPU is present, or if CUDA initialisation fails at runtime, all GPU features fall back to CPU automatically — a warning is printed to stderr but no error is returned. The CPU path is fully functional on its own.

## Installation

Add `pdf_raster` to your `Cargo.toml`:

```toml
[dependencies]
pdf_raster = { path = "../pdf_raster" }  # path dependency until published
```

For GPU acceleration (CUDA 12 required):

```toml
[dependencies]
pdf_raster = { path = "../pdf_raster", features = ["nvjpeg", "nvjpeg2k", "gpu-aa", "gpu-icc", "gpu-deskew"] }
```

## Quickstart

```rust
use std::path::Path;
use pdf_raster::{RasterOptions, raster_pdf};

let opts = RasterOptions {
    dpi: 300.0,
    first_page: 1,
    last_page: u32::MAX,  // render all pages
    deskew: true,
};

for (page_num, result) in raster_pdf(Path::new("document.pdf"), &opts) {
    match result {
        Ok(page) => {
            // page.pixels: Vec<u8>, 8-bit grayscale
            // length = page.width * page.height
            // layout: top-to-bottom, left-to-right, no padding (stride == width)
            println!("page {page_num}: {}×{} px", page.width, page.height);
        }
        Err(e) => eprintln!("page {page_num}: {e}"),
    }
}
```

## Tesseract integration

```rust
use pdf_raster::{RasterOptions, raster_pdf};

let opts = RasterOptions {
    dpi: 300.0,
    first_page: 1,
    last_page: u32::MAX,
    deskew: true,
};

for (page_num, result) in raster_pdf(Path::new("scan.pdf"), &opts) {
    let page = match result {
        Ok(p) => p,
        Err(e) => { eprintln!("page {page_num}: {e}"); continue; }
    };

    let text = tesseract::ocr_from_frame(
        &page.pixels,
        page.width as i32,
        page.height as i32,
        1,                  // bytes_per_pixel — always 1 (grayscale)
        page.width as i32,  // bytes_per_line — stride == width, no padding
        "eng",
    )?;

    // IMPORTANT: always pass effective_dpi, not dpi.
    // effective_dpi accounts for the PDF UserUnit field.
    // Passing the wrong value degrades Tesseract recognition accuracy.
    tesseract_handle.set_source_resolution(page.effective_dpi as i32);
}
```

**Do not binarise the pixels before passing to Tesseract.** The LSTM engine reads grayscale values directly for feature extraction; binarising discards information it would have used. For uneven scan backgrounds, configure Tesseract's Sauvola thresholding instead:

```rust
// has_images && !has_vector_text is a reliable heuristic for scanned pages
if page.diagnostics.has_images && !page.diagnostics.has_vector_text {
    // Sauvola handles uneven backgrounds better than Otsu
    tesseract_handle.set_variable("thresholding_method", "2");
}
```

## DPI guidelines

| Document type | Recommended DPI | Notes |
|---|---|---|
| Scanned documents | 300 | Standard for OCR |
| High-quality scans | 400–600 | When source scan is high-res |
| Native/vector PDFs | 150–200 | Text is resolution-independent |
| Thumbnails / preview | 72–96 | Not for OCR |

Use `page.suggested_dpi(min, max)` to re-render at the document's native image resolution:

```rust
let page = render_at_default_dpi()?;
if let Some(native_dpi) = page.suggested_dpi(150.0, 600.0) {
    if (native_dpi - opts.dpi).abs() > 10.0 {
        // Re-render at native resolution to avoid up/downsampling artefacts
    }
}
```

## Concurrent rendering

For multi-document pipelines, use `render_channel` to render pages in the background while OCR processes the previous one:

```rust
use pdf_raster::{RasterOptions, render_channel};

let opts = RasterOptions { dpi: 300.0, first_page: 1, last_page: 100, deskew: true };

// capacity=4: up to 4 rendered pages buffered before the producer blocks
let rx = render_channel(Path::new("scan.pdf"), &opts, 4);

for (page_num, result) in rx {
    match result {
        Ok(page) => { /* OCR page.pixels */ }
        Err(e)   => eprintln!("page {page_num}: {e}"),
    }
}
```

## Error handling

Per-page errors do not abort the remaining pages. Handle them per-iteration:

```rust
for (page_num, result) in raster_pdf(path, &opts) {
    match result {
        Ok(page) => process(page),
        Err(pdf_raster::RasterError::PageTooLarge { width, height }) => {
            eprintln!("page {page_num}: {width}×{height} exceeds limit — skipping");
        }
        Err(e) => return Err(e.into()),
    }
}
```

Document-open errors (bad path, corrupt PDF, JavaScript detected) are yielded as `(1, Err(...))` and the iterator ends immediately after.

## Security

pdf-raster refuses to open PDFs that contain JavaScript entry points and returns `RasterError::Pdf(InterpError::JavaScript { location })` immediately. No JavaScript is parsed or evaluated. This check covers:

- `/Catalog/OpenAction` with `/S /JavaScript`
- `/Catalog/AA` (catalog-level additional actions)
- `/Catalog/Names/JavaScript` (document JS name tree)
- `/Catalog/AcroForm/AA` (AcroForm additional actions)

## Building the CLI

```bash
# CPU-only (no CUDA dependency)
cargo build --release -p cli

# With all GPU features
CUDA_ARCH=sm_120 cargo build --release -p cli \
  --features "pdf_raster/nvjpeg,pdf_raster/nvjpeg2k,pdf_raster/gpu-aa,pdf_raster/gpu-icc,pdf_raster/gpu-deskew"
```

See [cli-reference.md](cli-reference.md) for CLI usage.
