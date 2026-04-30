# pdf-raster

Pure Rust PDF → pixels pipeline. Zero Poppler, zero subprocesses, zero Leptonica in the render path.

Renders PDF pages to 8-bit grayscale pixel buffers for direct consumption by Tesseract OCR or any other downstream consumer. No intermediate files.

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
| `raster` | Pixel-level fill/composite with AVX-512 and AVX2 SIMD |
| `gpu` | CUDA kernels — nvJPEG, nvJPEG2000, AA fill, tile fill, ICC CLUT, deskew |
| `font` | FreeType glyph cache and rendering |
| `color` | Pixel types, colour math |
| `encode` | PPM / PGM / PBM / PNG output |
| `cli` | `pdf-raster` binary |
| `pdf_bridge` | Poppler C++ wrapper — reference baseline only, not linked by CLI |

## Hardware compatibility

**CPU:** x86-64 only (AMD and Intel). AVX2 used when available; AVX-512 enabled with `-C target-cpu=native`. ARM / Apple Silicon / NEON are not yet supported — there is no `aarch64` build and no Apple Metal backend.

**GPU:** NVIDIA only via CUDA 12. AMD/Radeon (ROCm/HIP) and Intel (oneAPI) GPU backends are not yet implemented. If no NVIDIA GPU is present, all GPU features fall back to CPU automatically.

**Planned:** ARM NEON + Apple Metal → Intel CPU/GPU → Vulkan + AMD/Radeon. See [getting-started.md](docs/getting-started.md#planned-platform-support) for the roadmap.

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
tests/compare/compare.sh -r 150 tests/fixtures/ritual-14th.pdf
```

## Performance

Benchmarks vs `pdftoppm` (Ryzen 9900X3D + RTX 5070, GPU features enabled):

| Document | Character | pdf-raster | pdftoppm | Speedup |
|---|---|---|---|---|
| ritual-14th.pdf (116 KB, 41 pp) | Light vector + text | 213 ms | 262 ms | 1.2× |
| cryptic-rite.pdf (281 KB, 7 pp) | Mixed vector + images | 109 ms | 291 ms | 2.7× |
| kt-r2000.pdf (2.1 MB, 34 pp) | Dense vector paths | 495 ms | 1 507 ms | 3.0× |
| xxxii-sr.pdf (11 MB) | Mixed, image-heavy | 5.2 s | 44.4 s | 8.5× |
| scotch-rite-illustrated.pdf (50 MB) | Scan-heavy JPEG/JPEG2K | 17.2 s | 155.7 s | 9.1× |

Largest gains on scan-heavy corpora via nvJPEG + nvJPEG2000 GPU decoding.
