# Benchmarks

## Methodology

All benchmarks render all pages of each corpus PDF at **150 DPI** (the default) to PPM output, measuring wall-clock time with millisecond precision. Each tool is run sequentially (one PDF at a time, one tool at a time) to avoid contention. Output files are written to a temporary directory and discarded.

**Tool versions:**

| Tool | Version |
|---|---|
| pdf-raster | v0.3.0, built with `-C target-cpu=native` |
| pdftoppm (Poppler) | 24.02.0 (Ryzen), 25.03.0 (Intel) |

**Corpus:** 10 real-world PDFs spanning the common workload categories. Files are not distributed with the repository (see `.gitignore`); the corpus used internally matches the 10 fixture files in `tests/fixtures/corpus-*.pdf`.

| # | Document | File size | Pages | Character |
|---|---|---|---|---|
| 01 | Native text, small | 84 KB | 16 | Light text-only |
| 02 | Native vector + text | 236 KB | 16 | Vector paths + text |
| 03 | Native text, dense | 2.1 MB | 254 | Dense typeset layout |
| 04 | Ebook, mixed | 16 MB | 358 | Mixed text + raster |
| 05 | Academic book | 12 MB | 601 | Images + vector diagrams |
| 06 | Modern layout, DCT | 88 MB | 160 | JPEG-heavy print layout |
| 07 | Journal, DCT-heavy | 168 MB | 162 | Dense JPEG page images |
| 08 | 1927 scan, DCT | 145 MB | 390 | Scanned pages, JPEG |
| 09 | 1836 scan, DCT | 148 MB | 490 | Scanned pages, JPEG |
| 10 | Scan, JBIG2+JPX | 50 MB | 576 | Scanned, JBIG2 + JPEG 2000 |

---

## CPU-only: Ryzen 9 9900X3D (AVX-512) vs Intel i7-8700K (AVX2)

Both binaries built **without GPU features** (`--backend cpu`). This isolates the SIMD and interpreter work from GPU acceleration, and shows how the AVX-512 specialisation in the `raster` crate translates to real-world throughput.

**Hardware:**

| Machine | CPU | ISA | Cores | RAM |
|---|---|---|---|---|
| Ryzen bench | AMD Ryzen 9 9900X3D @ 4.4 GHz | x86-64 + AVX-512 | 12C/24T | 32 GB DDR5 |
| Intel bench | Intel Core i7-8700K @ 3.7 GHz | x86-64 + AVX2 | 6C/12T | 32 GB DDR4 |

**Reference:** `pdftoppm` (Poppler), CPU only, same machine as each pdf-raster run.

### Results

| # | Document | Pages | pdf-raster (AVX-512) | pdftoppm | Speedup | pdf-raster (AVX2) | pdftoppm | Speedup |
|---|---|---|---|---|---|---|---|---|
| 01 | Native text, small | 16 | 48 ms | 154 ms | **3.2×** | 319 ms | 300 ms | 0.94× |
| 02 | Native vector + text | 16 | 129 ms | 148 ms | **1.1×** | 493 ms | 290 ms | 0.59× |
| 03 | Native text, dense | 254 | 592 ms | 3 853 ms | **6.5×** | 2 725 ms | 7 651 ms | **2.8×** |
| 04 | Ebook, mixed | 358 | 3 155 ms | 4 922 ms | **1.6×** | 8 156 ms | 8 098 ms | 1.0× |
| 05 | Academic book | 601 | 4 318 ms | 7 532 ms | **1.7×** | 3 423 ms | 13 158 ms | **3.8×** |
| 06 | Modern layout, DCT | 160 | 2 856 ms | 6 479 ms | **2.3×** | 12 109 ms | 11 515 ms | 0.95× |
| 07 | Journal, DCT-heavy | 162 | 813 ms | 5 273 ms | **6.5×** | 1 460 ms | 8 379 ms | **5.7×** |
| 08 | 1927 scan, DCT | 390 | 19 112 ms | 387 636 ms | **20.3×** | — | — | — |
| 09 | 1836 scan, DCT | 490 | 58 012 ms | 389 933 ms | **6.7×** | — | — | — |
| 10 | Scan, JBIG2+JPX | 576 | 21 573 ms | 151 170 ms | **7.0×** | — | — | — |

_Intel 08–10 pending (large scan PDFs, ~15–30 min each on 12 threads)._

### Notes

- **Short PDFs (01–02):** The AVX2 build is slower than pdftoppm on the Intel box. These are 16-page documents where startup overhead and font subsystem init dominate. pdf-raster's startup path is not optimised for sub-100ms workloads.
- **DCT-heavy (07–09):** The 6–20× gains come from SIMD-accelerated JPEG decoding in the interpreter hot path, not GPU acceleration — this is pure CPU work on both machines.
- **Corpus 09 (490 pages):** The apparent variance between the two measurements for corpus-09 on the AVX-512 machine (58 s vs pdftoppm's 390 s) is genuine — pdftoppm serialises its JPEG decode path differently.
- **AVX-512 vs AVX2 (corpus 01–07):** The Ryzen machine wins clearly on text-dense and DCT-heavy workloads where the AVX-512 fill and composite kernels in the `raster` crate engage. On very short documents the difference is masked by fixed startup costs.

---

## GPU-accelerated: Ryzen 9 9900X3D + RTX 5070

Built with all GPU features enabled (nvJPEG, nvJPEG2000, GPU AA fill, ICC CLUT). Compared against `pdftoppm` (CPU only) on the same machine.

| # | Document | Pages | pdf-raster (GPU) | pdftoppm | Speedup |
|---|---|---|---|---|---|
| 01 | Native text, small | 16 | 217 ms | 252 ms | 1.2× |
| 02 | Native vector + text | 16 | 256 ms | 268 ms | 1.05× |
| 03 | Native text, dense | 254 | 4.3 s | 9.8 s | 2.3× |
| 04 | Ebook, mixed | 358 | 7.8 s | 12.4 s | 1.6× |
| 05 | Academic book | 601 | 12.8 s | 16.7 s | 1.3× |
| 06 | Modern layout, DCT | 160 | 11.7 s | 7.3 s | 0.6× |
| 07 | Journal, DCT-heavy | 162 | 3.8 s | 4.9 s | 1.3× |
| 08 | 1927 scan, DCT | 390 | 50 s | 279 s | **5.6×** |
| 09 | 1836 scan, DCT | 490 | 71 s | 356 s | **5.0×** |
| 10 | Scan, JBIG2+JPX | 576 | 19.6 s | 148.9 s | **7.6×** |

GPU gains are largest on scan-heavy corpora where nvJPEG and nvJPEG2000 offload bulk JPEG/JPEG2000 decode from the CPU. Corpus 06 (modern layout DCT) is slower because the images are small and frequent — GPU dispatch overhead exceeds the decode savings at that image size.

---

## Reproducing

```bash
# Build CPU-only release (no GPU features needed)
RUSTFLAGS="-C target-cpu=native" cargo build --release -p pdf-raster

# Time a single PDF against pdftoppm
time ./target/release/pdf-raster --backend cpu -r 150 input.pdf /tmp/out
time pdftoppm -r 150 input.pdf /tmp/ref

# Full pixel-diff comparison (verifies correctness, not just speed)
tests/compare/compare.sh -r 150 input.pdf
```

To reproduce the GPU benchmarks, build with the full feature set:

```bash
CUDA_ARCH=sm_120 RUSTFLAGS="-C target-cpu=native" \
  cargo build --release -p pdf-raster \
  --features "pdf_raster/nvjpeg,pdf_raster/nvjpeg2k,pdf_raster/gpu-aa,pdf_raster/gpu-icc"
```
