# pdf-raster

Pure Rust PDF → pixels pipeline. Zero Poppler, zero subprocesses, zero Leptonica in the render path.

Renders PDF pages to 8-bit grayscale pixel buffers for direct consumption by Tesseract OCR or any other downstream consumer. No intermediate files.

```toml
# Cargo.toml
pdf_raster = { git = "https://github.com/pthomasfournet/pdf-raster", tag = "v0.2.0" }
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
CUDA_ARCH=sm_120 cargo build --release -p cli \
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

Benchmarks vs a reference renderer (Ryzen 9900X3D + RTX 5070, 150 DPI, GPU features enabled, 5 runs):

| Document | Size / Pages | Character | pdf-raster | Reference | Speedup |
|---|---|---|---|---|---|
| Native text, small | 84 KB / 16 pp | Light text | 217 ms | 252 ms | 1.2× |
| Native vector + text | 236 KB / 16 pp | Vector paths + text | 256 ms | 268 ms | 1.05× |
| Native text, dense | 2.1 MB / 254 pp | Dense text layout | 4.3 s | 9.8 s | 2.3× |
| Ebook, mixed | 16 MB / 358 pp | Mixed content | 7.8 s | 12.4 s | 1.6× |
| Academic book | 12 MB / 601 pp | Images + vector | 12.8 s | 16.7 s | 1.3× |
| Modern layout, DCT | 88 MB / 160 pp | JPEG-heavy layout | 11.7 s | 7.3 s | 0.6× |
| Journal, DCT-heavy | 168 MB / 162 pp | Dense JPEG pages | 3.8 s | 4.9 s | 1.3× |
| 1927 scan, DCT | 145 MB / 390 pp | Scanned JPEG | 50 s | 279 s | **5.6×** |
| 1836 scan, DCT | 148 MB / 490 pp | Scanned JPEG | 71 s | 356 s | **5.0×** |
| Scan, JBIG2+JPX | 50 MB / 576 pp | Scanned JBIG2/JPEG2K | 19.6 s | 148.9 s | **7.6×** |

Largest gains on scan-heavy corpora via nvJPEG + nvJPEG2000 GPU decoding. The modern-layout DCT corpus (row 6) is slower due to high parallelism in the reference renderer on that specific layout type.
