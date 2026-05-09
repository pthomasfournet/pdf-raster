# pdf-raster

Pure Rust PDF → pixels pipeline. Zero Poppler, zero subprocesses, zero Leptonica in the render path.

Renders PDF pages to 8-bit grayscale pixel buffers for direct consumption by Tesseract OCR or any other downstream consumer. No intermediate files.

```toml
# Cargo.toml
pdf_raster = { git = "https://github.com/pthomasfournet/pdf-raster", tag = "v0.6.0" }
```

## What's new in v0.6.0

- **lopdf rip-out.** The lopdf 0.40 dependency is gone; replaced by an in-tree `pdf` crate (lazy mmap-based parser, threadsafe per-object cache, DOS-hardened limits). lopdf's `load_objects_raw` previously burned ~20% of corpus-07 cycles in `nom_locate` on the main thread before render workers could start, capping CPU utilisation at ~1.6/24 cores. With the new parser the same corpus is now cold-cache 689 ms.
- **RAM-backed output by default.** Bare-stem prefixes (e.g. `pdf-raster doc.pdf out`) now write pages to `/dev/shm/pdf-raster-<pid>-<nanos>/` rather than disk, avoiding ext4 `auto_da_alloc` rename serialisation that was parking 24 workers in `do_renameat2`. A `SpillPolicy` polls `/proc/meminfo` every 100 ms and spills subsequent pages to disk when MemAvailable drops below 1 GiB. New flags: `--ram`, `--no-ram`, `--ram-path`. Path-like prefixes (anything with `/` or starting with `.`) opt out and write where the user asked.

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

**GPU (optional):**
- **NVIDIA via CUDA 12 or 13** — full feature set (nvJPEG, nvJPEG2000, AA fill, ICC CLUT, ICC matrix, deskew, image cache).  `cudarc` is pinned to the `cuda-12080` driver-API binding so the same source builds against both 12.x and 13.x drivers (forward-compatible per the CUDA driver-API ABI).
- **Cross-vendor via Vulkan compute** — AA fill and tile fill kernels run on any Vulkan 1.3+ device (NVIDIA, AMD, Intel, Apple via `MoltenVK`). Verified on RTX 5070; cross-vendor smoke pending hardware.  No nvJPEG / cache support under Vulkan today.
- **Linux iGPU/dGPU via VA-API** — JPEG baseline decode on AMD VCN, Intel Quick Sync, Intel Arc.

All GPU features fall back to CPU automatically when unavailable.  AMD/Radeon ROCm and Apple Metal-native backends are not implemented (Vulkan covers Apple via `MoltenVK`).

## Build

```bash
# CPU-only (no CUDA)
cargo build --release -p cli

# With all GPU features (CUDA 12 or 13 toolkit, NVIDIA GPU required)
# Default CUDA_ARCH is sm_80 (Ampere); override for older or newer GPUs.
CUDA_ARCH=sm_120 cargo build --release -p pdf-raster \
  --features "pdf_raster/nvjpeg,pdf_raster/nvjpeg2k,pdf_raster/gpu-aa,pdf_raster/gpu-icc,pdf_raster/gpu-deskew,pdf_raster/cache"

# With Vulkan compute backend (cross-vendor; no NVIDIA dependency).
# Requires the LunarG Vulkan SDK on the build host (slangc compiles the
# .slang shaders to SPIR-V).  Vulkan 1.3+ ICD on the runtime host.
cargo build --release -p pdf-raster --features "pdf_raster/vulkan"
```

### Picking `CUDA_ARCH` for your GPU

The `CUDA_ARCH` environment variable controls which Compute Capability the PTX kernels target. Mismatched arch flags produce kernels the GPU can't load at runtime. Set it to your card's CC (e.g. `sm_75`, `sm_86`, `sm_120`).

| GPU generation | Architecture | `CUDA_ARCH` |
|---|---|---|
| GTX 10-series | Pascal | `sm_61` |
| RTX 20-series, Quadro RTX | Turing | `sm_75` |
| RTX 30-series, A100 | Ampere | `sm_80` / `sm_86` |
| RTX 40-series | Ada Lovelace | `sm_89` |
| H100 / Hopper | Hopper | `sm_90` |
| RTX 50-series | Blackwell | `sm_120` |

Look up your card's exact Compute Capability at [developer.nvidia.com/cuda-gpus](https://developer.nvidia.com/cuda-gpus). The build defaults to `sm_80` if `CUDA_ARCH` is unset; that's a reasonable fallback for any Ampere-or-later card thanks to PTX forward-compatibility, but matching your hardware exactly produces better-optimised code.

### Feature flags

| Flag | What it enables | Required runtime |
|---|---|---|
| `nvjpeg` | GPU JPEG decode for `DCTDecode` | `libnvjpeg.so` (ships with CUDA 12 or 13 toolkit) |
| `nvjpeg2k` | GPU JPEG-2000 decode for `JPXDecode` | `libnvjpeg2k.so` |
| `gpu-aa` | GPU supersampled anti-aliased fill | CUDA |
| `gpu-icc` | GPU CMYK→RGB ICC transform | CUDA |
| `gpu-deskew` | GPU deskew rotation via NPP | CUDA + NPP |
| `cache` | Device-resident image cache (3-tier VRAM/host/disk) | CUDA |
| `vaapi` | Linux iGPU/dGPU JPEG decode (AMD/Intel) | `libva.so.2` + DRM render node |
| `vulkan` | Vulkan compute backend for AA / tile fill (cross-vendor) | Vulkan 1.3+ ICD; pulls in `gpu-aa`. Slang shaders compiled to SPIR-V via `slangc` from the `LunarG` Vulkan SDK |

All GPU features fall back to CPU automatically when the runtime requirement is missing, except `--backend cuda` / `--backend vulkan` / `--backend vaapi` which fail loudly with a clear error.

### Backend selection

The runtime backend is chosen from three sources, in priority order:

1. The CLI `--backend {auto,cpu,cuda,vaapi,vulkan}` flag.
2. The `PDF_RASTER_BACKEND` environment variable (same valid values).
3. The compile-time default — `auto`.

Under `auto`, when both backends are compiled in, **Vulkan is preferred over CUDA**. Vulkan's per-process init is faster and the kernel dispatch is comparable on the workloads that matter; CUDA wins narrowly when the device-resident `cache` feature is firing and amortising across many pages from one session.  Both backends fall through to CPU when their runtime is unavailable; `--backend cuda` / `--backend vulkan` make the failure loud instead.

```bash
# Ship a Vulkan-default binary, override per-process when you need CUDA:
PDF_RASTER_BACKEND=cuda pdf-raster input.pdf out

# CLI flag always wins over the env var:
PDF_RASTER_BACKEND=cuda pdf-raster --backend cpu input.pdf out   # uses CPU
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

**CPU-only (no GPU), Ryzen 9 9900X3D + AVX-512, v0.6.0, RAM-backed output, cold cache, hyperfine 5 runs:**

| Document | Pages | pdf-raster |
|---|---|---|
| Native text, small | 16 | 51 ms ± 1 ms |
| Native text, dense | 254 | 254 ms ± 3 ms |
| Ebook, mixed | 358 | 382 ms ± 4 ms |
| Journal, DCT-heavy | 162 | 689 ms ± 7 ms |
| 1927 scan, DCT | 150 | 2.21 s ± 291 ms |
| 1836 scan, DCT | 490 | 2.54 s ± 5 ms |
| Scan, JBIG2+JPX | 576 | 18.22 s ± 55 ms |

Per-version regression history and the full pdftoppm comparison are in **[docs/benchmarks.md](docs/benchmarks.md)** — corpus-09 dropped from 36 s (v0.5.1) to 2.54 s (v0.6.0), a 14× win driven by the lopdf rip-out and RAM-backed output default.

**GPU-accelerated (nvJPEG + nvJPEG2000), same machine + RTX 5070:**

| Document | Pages | pdf-raster | pdftoppm | Speedup |
|---|---|---|---|---|
| Native text, dense | 254 | 4.3 s | 9.8 s | 2.3× |
| 1927 scan, DCT | 390 | 50 s | 279 s | **5.6×** |
| 1836 scan, DCT | 490 | 71 s | 356 s | **5.0×** |
| Scan, JBIG2+JPX | 576 | 19.6 s | 148.9 s | **7.6×** |

Largest gains on scan-heavy corpora where SIMD JPEG decoding (CPU) and nvJPEG/nvJPEG2000 (GPU) dominate. Short native-text PDFs are startup-bound and show modest gains. See [docs/benchmarks.md](docs/benchmarks.md) for the full table including an Intel i7-8700K (AVX2-only) comparison.
