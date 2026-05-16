# rasterrocket

Pure Rust PDF → pixels pipeline. Zero Poppler, zero subprocesses, zero Leptonica in the render path.

Renders PDF pages to 8-bit grayscale pixel buffers for direct consumption by Tesseract OCR or any other downstream consumer. No intermediate files.

```toml
# Cargo.toml — library
rasterrocket = "1.0"
```

```bash
# CLI — drop-in pdftoppm replacement
cargo install rasterrocket-cli
```

## What's new in v1.0.0

- **Spec-correct simple-font text.** The `Widths`-array lookup now prevails over FreeType metrics for all embedded and non-embedded simple fonts, per PDF §9.2.4. Academic and English-language PDFs show the largest improvement: avg RMSE vs pdftoppm dropped 17–19 points on corpus-05 and corpus-14.
- **Vulkan compute backend.** `--features vulkan` or `--backend vulkan` runs AA fill, tile fill, and parallel-Huffman JPEG decode on any Vulkan 1.3+ device — NVIDIA, AMD, Intel, or Apple via `MoltenVK`. Verified on RTX 5070; cross-vendor smoke pending hardware. Under `auto`, Vulkan is preferred over CUDA when both are compiled in (faster process init on single-session workloads).
- **Process-static GPU init.** The CUDA and Vulkan contexts are initialised once per process (not once per `open_session`). Short-lived multi-document pipelines no longer pay ~240 ms per document.
- **`PDF_RASTER_BACKEND` env var.** Switch backends at runtime without recompiling. Valid values: `auto`, `cpu`, `cuda`, `vaapi`, `vulkan`. The CLI `--backend` flag takes precedence.
- **Parallel-Huffman JPEG.** GPU-accelerated Huffman decode (`gpu-jpeg-huffman` feature, implied by `vulkan`) is wired into the production decode path. Dormant by default (threshold = `u32::MAX`); enable with `PDF_RASTER_HUFFMAN_THRESHOLD=0` for benchmarking.
- **`RasterOptions::default()`** — `dpi = 300`, `first_page = 1`, `last_page = u32::MAX`, `deskew = false`, `pages = None`. Use `..RasterOptions::default()` to fill unset fields.
- **`PageSet`** — render a sparse subset of pages without visiting intermediate ones.

```rust
use rasterrocket::{RasterOptions, raster_pdf};

let opts = RasterOptions { dpi: 300.0, ..RasterOptions::default() };

for (page_num, result) in raster_pdf(Path::new("scan.pdf"), &opts) {
    let page = result?;
    // page.pixels — Vec<u8>, 8-bit greyscale, width × height, top-to-bottom
    // page.effective_dpi — pass to your OCR engine (accounts for PDF UserUnit scaling)
    // See the OCR Integration wiki for Tesseract, ocrs, Google Cloud Vision, and GPT-5 patterns.
}
```

## Documentation

| Document | Contents |
|---|---|
| [Getting Started](docs/getting-started.md) | Installation, quickstart, Tesseract integration, DPI guidance, error handling, security |
| [API Reference](docs/api-reference.md) | Full signatures for `raster_pdf`, `render_channel`, `RasterOptions`, `RenderedPage`, `RasterError`, `PageDiagnostics`, feature flags, GPU dispatch thresholds |
| [CLI Reference](docs/cli-reference.md) | All `rrocket` command-line flags, output format matrix, examples, pixel-diff comparison |
| [Benchmarks](../../wiki/Benchmarks) | Methodology, 10-document corpus results, CPU-only AVX-512 vs AVX2, GPU-accelerated, reproduction steps |
| [OCR Integration](../../wiki/OCR-Integration) | Tesseract (`leptess`) and ocrs — instance reuse, zero-copy patterns, DPI wiring, multi-threaded examples |
| [LLM Vision OCR Integration](../../wiki/LLM-Vision-OCR-Integration) | Google Cloud Vision and GPT-5 — encoding helper, Rust + Python examples, cost and latency guidance |

## Crates

| Crate | What you get |
|---|---|
| [`rasterrocket`](https://crates.io/crates/rasterrocket) | Library — `raster_pdf`, `render_channel`, `RasterOptions`, `RenderedPage` |
| [`rasterrocket-cli`](https://crates.io/crates/rasterrocket-cli) | `rrocket` binary — drop-in `pdftoppm` replacement |

## Hardware compatibility

**CPU:** x86-64 (AMD and Intel) and `aarch64` (ARM). AVX2/AVX-512 on x86-64; NEON (and SVE2 on nightly) on aarch64. Build with `-C target-cpu=native` to enable AVX-512 or native NEON width.

**GPU (optional):**
- **NVIDIA via CUDA 12 or 13** — full feature set (nvJPEG, nvJPEG2000, AA fill, ICC CLUT, ICC matrix, deskew, image cache).  `cudarc` is pinned to the `cuda-12080` driver-API binding so the same source builds against both 12.x and 13.x drivers (forward-compatible per the CUDA driver-API ABI).
- **Cross-vendor via Vulkan compute** — AA fill, tile fill, and parallel-Huffman JPEG decode kernels run on any Vulkan 1.3+ device (NVIDIA, AMD, Intel, Apple via `MoltenVK`). Verified on RTX 5070; cross-vendor smoke pending hardware.  No nvJPEG / cache support under Vulkan today (JPEG decode goes through the GPU parallel-Huffman path, not nvJPEG).
- **Linux iGPU/dGPU via VA-API** — JPEG baseline decode on AMD VCN, Intel Quick Sync, Intel Arc.

All GPU features fall back to CPU automatically when unavailable.  AMD/Radeon ROCm and Apple Metal-native backends are not implemented (Vulkan covers Apple via `MoltenVK`).

## Build

```bash
# CPU-only (no CUDA)
cargo build --release -p rasterrocket-cli

# With all GPU features (CUDA 12 or 13 toolkit, NVIDIA GPU required)
# Default CUDA_ARCH is sm_80 (Ampere); override for older or newer GPUs.
CUDA_ARCH=sm_120 cargo build --release -p rasterrocket-cli \
  --features "rasterrocket/nvjpeg,rasterrocket/nvjpeg2k,rasterrocket/gpu-aa,rasterrocket/gpu-icc,rasterrocket/gpu-deskew,rasterrocket/cache"

# With Vulkan compute backend (cross-vendor; no NVIDIA dependency).
# Requires the LunarG Vulkan SDK on the build host (slangc compiles the
# .slang shaders to SPIR-V).  Vulkan 1.3+ ICD on the runtime host.
cargo build --release -p rasterrocket-cli --features "rasterrocket/vulkan"
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
| `vulkan` | Vulkan compute backend for AA fill, tile fill, and parallel-Huffman JPEG decode (cross-vendor) | Vulkan 1.3+ ICD; pulls in `gpu-aa` and `gpu-jpeg-huffman`. Slang shaders compiled to SPIR-V via `slangc` from the `LunarG` Vulkan SDK |

All GPU features fall back to CPU automatically when the runtime requirement is missing, except `--backend cuda` / `--backend vulkan` / `--backend vaapi` which fail loudly with a clear error.

### Backend selection

The runtime backend is chosen from three sources, in priority order:

1. The CLI `--backend {auto,cpu,cuda,vaapi,vulkan}` flag.
2. The `PDF_RASTER_BACKEND` environment variable (same valid values).
3. The compile-time default — `auto`.

Under `auto`, when both backends are compiled in, **Vulkan is preferred over CUDA**. Vulkan's per-process init is faster and the kernel dispatch is comparable on the workloads that matter; CUDA wins narrowly when the device-resident `cache` feature is firing and amortising across many pages from one session.  Both backends fall through to CPU when their runtime is unavailable; `--backend cuda` / `--backend vulkan` make the failure loud instead.

```bash
# Ship a Vulkan-default binary, override per-process when you need CUDA:
PDF_RASTER_BACKEND=cuda rrocket input.pdf out

# CLI flag always wins over the env var:
PDF_RASTER_BACKEND=cuda rrocket --backend cpu input.pdf out   # uses CPU
```

### Compile cache (`sccache`) — optional

`.cargo/config.toml` sets `SCCACHE_CACHE_MULTIARCH=1` so `-C target-cpu=native` builds *can* be cached, but the wrapper itself is opt-in: plain `cargo build` works without sccache.  To enable:

```bash
cargo install sccache       # one-time install
export RUSTC_WRAPPER=sccache # add to ~/.bashrc to make it permanent
```

Shared with any other Rust project on the same machine that opts in — cross-project cache keys don't collide (sccache hashes the full compiler args).  Verify with `sccache --show-stats` (a healthy hit-rate after the first build is ≥ 70%).  Do not `rsync ~/.cache/sccache` between machines with different CPUs — `target-cpu=native` resolves differently per arch.

## Testing

```bash
# Unit tests (always filter by module, never run unfiltered)
cargo test -p rasterrocket --lib -- deskew
cargo test -p rasterrocket-gpu --lib -- icc

# Pixel-diff comparison against pdftoppm (requires release build in PATH)
tests/compare/compare.sh -r 150 tests/fixtures/input.pdf
```

## Security

rasterrocket parses untrusted PDF input. Its hardening posture:

- **Memory-safe core.** The PDF parser, content interpreter, font/glyph
  resolution, and the JBIG2 / CCITT / JPEG / Flate / LZW decoders are pure
  Rust. Malformed, truncated, adversarial, or hostile input is converted to a
  clear per-page `Err` or a bounded skip — never a silent wrong render, an
  unbounded allocation, an infinite loop, or a process abort. The render
  pipeline enforces a per-page operator/wall-clock/form-depth watchdog, an
  aggregate content-size cap, a filter-chain length and decompression-bomb
  cap, and a total-raster-area cap; per-page panics are isolated so one bad
  page cannot abort a batch.
- **No script execution.** rasterrocket has no JavaScript engine. A PDF
  containing JavaScript (`/OpenAction`, `/AA`, `/Names/JavaScript`) is
  rendered for its static appearance and a warning is logged; no `/JS` is
  ever decoded or evaluated.
- **Native FFI trust boundary.** Two transitive dependencies wrap C
  libraries: glyph rasterization links the system **FreeType**
  (`libfreetype6`) and JPEG 2000 decoding links the system **OpenJPEG**
  (`libopenjp2`). These are the historically highest-CVE components of any
  PDF stack. rasterrocket links the *system* libraries (not vendored
  copies), so their CVE exposure is exactly your host's package patch level.
  **Deployments that process untrusted PDFs MUST keep `libfreetype6` and
  `libopenjp2` patched** (or sandbox the process). `cargo audit` is clean
  for the Rust dependency tree, but RustSec does not track upstream C-library
  CVEs — the host's patch cadence is the control there.
- **Encrypted input.** Owner-password-only encrypted PDFs are decrypted via
  an opt-in, default-deny liability gate (see Getting Started). Unencrypted
  input never spawns a subprocess and never writes a temp file.

## Performance

Benchmarks vs Poppler's `pdftoppm` on a 10-document corpus at 150 DPI. Full methodology, hardware details, and AVX2 vs AVX-512 comparison in **[the Benchmarks wiki page](../../wiki/Benchmarks)**.

**CPU-only (no GPU), Ryzen 9 9900X3D + AVX-512, v0.9.1, RAM-backed output, cold cache, hyperfine 5 runs:**

| Document | Pages | rasterrocket |
|---|---|---|
| Native text, small | 16 | 41 ms ± 1 ms |
| Native vector + text | 16 | 18 ms ± 1 ms |
| Native text, dense | 254 | 231 ms ± 2 ms |
| Ebook, mixed | 358 | 278 ms ± 3 ms |
| Academic book | 601 | 582 ms ± 12 ms |
| Modern layout, DCT | 160 | 1 450 ms ± 10 ms |
| Journal, DCT-heavy | 162 | 783 ms ± 5 ms |
| 1927 scan, DCT | 390 | 1 652 ms ± 89 ms |
| 1836 scan, DCT | 490 | 2 859 ms ± 658 ms |
| Scan, JBIG2+JPX | 576 | 17 616 ms ± 260 ms |

Per-version regression history and the full pdftoppm comparison are in **[the Benchmarks wiki page](../../wiki/Benchmarks)**.

**GPU-accelerated (CUDA: nvJPEG + nvJPEG2000), same machine + RTX 5070 (v0.9.1):**

| Document | Pages | rasterrocket | pdftoppm | Speedup |
|---|---|---|---|---|
| Native text, dense | 254 | 4.3 s | 9.8 s | 2.3× |
| 1927 scan, DCT | 390 | 50 s | 279 s | **5.6×** |
| 1836 scan, DCT | 490 | 71 s | 356 s | **5.0×** |
| Scan, JBIG2+JPX | 576 | 19.6 s | 148.9 s | **7.6×** |

Largest gains on scan-heavy corpora where SIMD JPEG decoding (CPU) and nvJPEG/nvJPEG2000 (GPU) dominate. Short native-text PDFs are startup-bound and show modest gains. See [the Benchmarks wiki page](../../wiki/Benchmarks) for the full table including an Intel i7-8700K (AVX2-only) comparison.
