# pdf-raster

Pure Rust PDF → pixels pipeline. Zero Poppler, zero subprocesses, zero Leptonica in the render path.

Renders PDF pages to 8-bit grayscale pixel buffers for direct consumption by Tesseract OCR or any other downstream consumer. No intermediate files.

```toml
# Cargo.toml
pdf_raster = { git = "https://github.com/pthomasfournet/pdf-raster", tag = "v0.5.1" }
```

```rust
use pdf_raster::{RasterOptions, raster_pdf};

let opts = RasterOptions { dpi: 300.0, first_page: 1, last_page: u32::MAX, deskew: true };

for (page_num, result) in raster_pdf(Path::new("scan.pdf"), &opts) {
    let page = result?;
    // page.pixels: Vec<u8>, 8-bit grayscale, width × height, top-to-bottom
    let text = tesseract::ocr_from_frame(
        &page.pixels, page.width as i32, page.height as i32,
        1, page.width as i32, "eng",
    )?;
    tesseract_handle.set_source_resolution(page.effective_dpi as i32);
}
```

## Documentation

| Document | Contents |
|---|---|
| [Getting Started](docs/getting-started.md) | Installation, quickstart, Tesseract integration, DPI guidance, error handling, security |
| [API Reference](docs/api-reference.md) | Full signatures for `raster_pdf`, `render_channel`, `RasterOptions`, `RenderedPage`, `RasterError`, `PageDiagnostics`, feature flags, GPU dispatch thresholds |
| [CLI Reference](docs/cli-reference.md) | All `pdf-raster` command-line flags, output format matrix, examples, pixel-diff comparison |
| [Benchmarks](docs/benchmarks.md) | Methodology, 10-document corpus results, CPU-only AVX-512 vs AVX2, GPU-accelerated, reproduction steps |

## Crate map

| Crate | Role |
|---|---|
| `pdf_raster` | **Public API** — `raster_pdf`, `render_channel`, `RasterOptions`, `RenderedPage` |
| `pdf_interp` | Native PDF interpreter — content streams, fonts, images, shading, transparency |
| `raster` | Pixel-level fill/composite with AVX-512, AVX2, and NEON SIMD |
| `gpu` | CUDA kernels — nvJPEG, nvJPEG2000, AA fill, tile fill, ICC CLUT, deskew; VA-API JPEG decode |
| `font` | FreeType glyph cache and rendering |
| `color` | Pixel types, colour math |
| `encode` | PPM / PGM / PBM / PNG output |
| `cli` | `pdf-raster` binary |
| `pdf_bridge` | Poppler C++ wrapper — reference baseline only, not linked by CLI |

## Hardware compatibility

**CPU:** x86-64 (AMD and Intel) and `aarch64` (ARM). AVX2/AVX-512 on x86-64; NEON (and SVE2 on nightly) on aarch64. Build with `-C target-cpu=native` to enable AVX-512 or native NEON width.

**GPU (optional):** NVIDIA via CUDA 12, and Linux iGPU/dGPU via VA-API (AMD VCN, Intel Quick Sync, Intel Arc). AMD/Radeon ROCm and Apple Metal backends are not yet implemented. All GPU features fall back to CPU automatically when unavailable.

**Planned:** Apple Metal (macOS/Apple Silicon) → Vulkan compute. See [getting-started.md](docs/getting-started.md#platform-support) for details.

## Build

```bash
# CPU-only (no CUDA)
cargo build --release -p cli

# With all GPU features (CUDA 12, RTX/NVIDIA GPU required)
CUDA_ARCH=sm_120 cargo build --release -p pdf-raster \
  --features "pdf_raster/nvjpeg,pdf_raster/nvjpeg2k,pdf_raster/gpu-aa,pdf_raster/gpu-icc,pdf_raster/gpu-deskew"
```

## Testing

```bash
# Unit tests (always filter by module, never run unfiltered)
cargo test -p pdf_interp --lib -- resources
cargo test -p gpu --lib -- icc

# Pixel-diff comparison against pdftoppm (requires release build in PATH)
tests/compare/compare.sh -r 150 tests/fixtures/input.pdf
```

## Performance

Benchmarks vs Poppler's `pdftoppm` on a 10-document corpus at 150 DPI. Full methodology, hardware details, and AVX2 vs AVX-512 comparison in **[docs/benchmarks.md](docs/benchmarks.md)**.

**CPU-only (no GPU), Ryzen 9 9900X3D + AVX-512:**

| Document | Pages | pdf-raster | pdftoppm | Speedup |
|---|---|---|---|---|
| Native text, small | 16 | 48 ms | 154 ms | 3.2× |
| Native text, dense | 254 | 592 ms | 3 853 ms | **6.5×** |
| Journal, DCT-heavy | 162 | 813 ms | 5 273 ms | **6.5×** |
| 1927 scan, DCT | 390 | 19 s | 388 s | **20×** |
| 1836 scan, DCT | 490 | 58 s | 390 s | 6.7× |
| Scan, JBIG2+JPX | 576 | 22 s | 151 s | 7.0× |

**GPU-accelerated (nvJPEG + nvJPEG2000), same machine + RTX 5070:**

| Document | Pages | pdf-raster | pdftoppm | Speedup |
|---|---|---|---|---|
| Native text, dense | 254 | 4.3 s | 9.8 s | 2.3× |
| 1927 scan, DCT | 390 | 50 s | 279 s | **5.6×** |
| 1836 scan, DCT | 490 | 71 s | 356 s | **5.0×** |
| Scan, JBIG2+JPX | 576 | 19.6 s | 148.9 s | **7.6×** |

Largest gains on scan-heavy corpora where SIMD JPEG decoding (CPU) and nvJPEG/nvJPEG2000 (GPU) dominate. Short native-text PDFs are startup-bound and show modest gains. See [docs/benchmarks.md](docs/benchmarks.md) for the full table including an Intel i7-8700K (AVX2-only) comparison.
