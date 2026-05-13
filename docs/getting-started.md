# Getting Started

pdf-raster converts PDF pages to pixel buffers in pure Rust. No subprocesses, no Leptonica, no Poppler in the render path.

## What it does

Given a PDF file, pdf-raster renders each page to an 8-bit grayscale buffer in memory. The primary use case is feeding pages directly into Tesseract OCR without writing any intermediate files.

## Hardware requirements

### CPU

**Supported:** x86-64 (AMD and Intel) and `aarch64` (ARM / Apple Silicon).

**x86-64:**
- AVX2 SIMD paths are runtime-detected with a scalar fallback.
- AVX-512 (`avx512f/bw/vl/dq/vnni/vpopcntdq` and extensions) is active when built with `-C target-cpu=native` on a compatible CPU. Developed and benchmarked on an AMD Ryzen 9900X3D.
- All Intel consumer CPUs (Alder/Raptor/Arrow Lake) have AVX2; AVX-512 is disabled on them — Xeon only.

**aarch64:**
- NEON is used unconditionally (mandatory on all ARMv8-A).
- SVE2 (`svcnt_u8_z` popcount tier) is available behind the `nightly-sve2` Cargo feature on nightly Rust. Gives up to 4× NEON throughput on wide-SVE2 server chips (Graviton4 full width).
- `cargo check --target aarch64-unknown-linux-gnu` is clean; no Apple Metal GPU backend yet.

### GPU (optional)

Three GPU backends are available on Linux:

**NVIDIA (CUDA 12 or 13):**

| Feature | Minimum GPU | Library required |
|---|---|---|
| `nvjpeg` | Any CUDA-capable NVIDIA GPU | `libnvjpeg.so` (ships with CUDA 12 or 13 toolkit) |
| `nvjpeg2k` | Any CUDA-capable NVIDIA GPU | `libnvjpeg2k.so` (separate download; build script probes `/13` then `/12`) |
| `gpu-aa` | Any CUDA-capable NVIDIA GPU | CUDA runtime |
| `gpu-icc` | Any CUDA-capable NVIDIA GPU | CUDA runtime |
| `gpu-deskew` | Any CUDA-capable NVIDIA GPU | CUDA NPP (`libnppig.so`, `libnppc.so`) |
| `cache` | Any CUDA-capable NVIDIA GPU | CUDA runtime; opt-in disk-tier persistence via `PDF_RASTER_CACHE_DIR` |

`cudarc` is pinned to the `cuda-12080` driver-API binding so the same source builds against both 12.x and 13.x drivers (forward-compatible per the CUDA driver-API ABI).

**Vulkan compute (cross-vendor — NVIDIA, AMD, Intel, Apple via `MoltenVK`):**

| Feature | Supported hardware | Libraries required |
|---|---|---|
| `vulkan` | Any Vulkan 1.3+ device | Vulkan ICD (e.g. `mesa-vulkan-drivers`, `nvidia-driver`); `slangc` from the LunarG Vulkan SDK at *build* time |

Vulkan covers the AA-fill, tile-fill, and parallel-Huffman JPEG decode kernels (`vulkan` implies `gpu-jpeg-huffman`).  No `nvjpeg` / `cache` support today; under `--backend vulkan` the renderer runs uncached but JPEG decode goes through the GPU parallel-Huffman path (GPU_JPEG_HUFFMAN_THRESHOLD_PX = u32::MAX today — dormant until threshold tuning).  Verified on RTX 5070; cross-vendor smoke (AMD-RADV, Intel-ANV, Apple Metal-via-MoltenVK) pending hardware.

**VA-API (Linux iGPU/dGPU — AMD VCN, Intel Quick Sync, Intel Arc):**

| Feature | Supported hardware | Libraries required |
|---|---|---|
| `vaapi` | AMD VCN (Raphael+), Intel UHD 630+, Intel Arc | `libva.so.2`, `libva-drm.so.2` |

VA-API provides hardware JPEG baseline decode. CMYK and progressive JPEGs fall through to CPU. When both `nvjpeg` and `vaapi` are active, nvJPEG takes priority; VA-API fires as fallback.

All GPU features fall back to CPU automatically if initialisation fails — a message is logged but no error is returned.  Use `--backend cuda` / `--backend vulkan` / `--backend vaapi` to convert silent fallbacks into hard errors.

## Platform support

| Platform | CPU SIMD | GPU acceleration | Status |
|---|---|---|---|
| x86-64 AMD (Ryzen) | AVX-512 + AVX2 | NVIDIA CUDA + AMD VA-API + Vulkan | **Supported** |
| x86-64 Intel (consumer) | AVX2 | NVIDIA CUDA + Intel VA-API + Vulkan | **Supported** |
| x86-64 Intel (Xeon) | AVX-512 + AVX2 | NVIDIA CUDA + Intel VA-API + Vulkan | **Supported** |
| aarch64 Linux (Graviton, RPi) | NEON + SVE2† | Vulkan (Mesa) | CPU full, Vulkan untested on aarch64 |
| Apple Silicon (M1–M4) | NEON | Vulkan via `MoltenVK` (untested) | CPU full, Vulkan untested |
| AMD/Radeon ROCm | — | — | Not implemented (Vulkan covers Radeon) |
| Apple Metal (native) | — | — | Not implemented (Vulkan via `MoltenVK` is the path) |

† SVE2 requires `nightly-sve2` Cargo feature and nightly Rust.

## Installation

Add `pdf_raster` to your `Cargo.toml` as a git dependency:

```toml
[dependencies]
pdf_raster = { git = "https://github.com/pthomasfournet/pdf-raster", tag = "v0.8.0" }
```

For GPU acceleration — NVIDIA (CUDA 12 or 13) + Vulkan (cross-vendor) + VA-API (Linux iGPU/dGPU):

```toml
[dependencies]
# CUDA GPU features (NVIDIA only; full feature set including the Phase 9 cache):
pdf_raster = { git = "https://github.com/pthomasfournet/pdf-raster", tag = "v0.8.0", features = ["nvjpeg", "nvjpeg2k", "gpu-aa", "gpu-icc", "gpu-deskew", "cache"] }

# Vulkan compute (cross-vendor — NVIDIA, AMD, Intel, Apple via MoltenVK):
pdf_raster = { git = "https://github.com/pthomasfournet/pdf-raster", tag = "v0.8.0", features = ["vulkan"] }

# VA-API (AMD/Intel iGPU on Linux — libva required):
pdf_raster = { git = "https://github.com/pthomasfournet/pdf-raster", tag = "v0.8.0", features = ["vaapi"] }
```

To track the latest commit on `master` instead of a pinned tag:

```toml
[dependencies]
pdf_raster = { git = "https://github.com/pthomasfournet/pdf-raster", branch = "master" }
```

Pin to a tag for reproducible builds. Run `cargo update -p pdf_raster` to advance to a newer tag when one is released.

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
cargo build --release -p pdf-raster

# With all GPU features
# Default CUDA_ARCH is sm_80 (Ampere); override to match your card.
CUDA_ARCH=sm_120 cargo build --release -p pdf-raster \
  --features "pdf_raster/nvjpeg,pdf_raster/nvjpeg2k,pdf_raster/gpu-aa,pdf_raster/gpu-icc,pdf_raster/gpu-deskew,pdf_raster/cache"
```

`CUDA_ARCH` must match your GPU's Compute Capability (e.g. `sm_75` for RTX 20-series, `sm_86` for RTX 30-series, `sm_89` for RTX 40-series, `sm_120` for RTX 50-series). Look yours up at [developer.nvidia.com/cuda-gpus](https://developer.nvidia.com/cuda-gpus). See the [README's GPU matrix](../README.md#picking-cuda_arch-for-your-gpu) for the common-case table and the [feature-flag list](../README.md#feature-flags).

### A note on output destination (v0.6.0+)

The CLI defaults to writing rendered pages into a fresh `/dev/shm/pdf-raster-<pid>-<nanos>/` tmpfs directory whenever `OUTPUT_PREFIX` looks like a bare stem (e.g. `out`, `p`). Disk writes were dominating wall time on JPEG-heavy workloads — ext4 `auto_da_alloc` rename serialisation parked all 24 render threads in `do_renameat2`. Path-like prefixes (anything containing `/` or starting with `.`) opt out and write where you asked. Pass `--no-ram` to force on-disk output even for bare stems, or `--ram-path <DIR>` to override the tmpfs location. A built-in spill policy automatically falls through to disk when MemAvailable drops below 1 GiB.

See [cli-reference.md](cli-reference.md) for CLI usage.
