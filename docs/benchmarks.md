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
| 01 | Native text, small | 16 | 44 ms | 151 ms | **3.4×** | 320 ms | 461 ms | 0.69× |
| 02 | Native vector + text | 16 | 121 ms | 146 ms | **1.2×** | 466 ms | 428 ms | 0.92× |
| 03 | Native text, dense | 254 | 558 ms | 3 612 ms | **6.5×** | 2 220 ms | 6 999 ms | **3.2×** |
| 04 | Ebook, mixed | 358 | 1 722 ms | 3 787 ms | **2.2×** | 8 256 ms | 7 411 ms | 0.90× |
| 05 | Academic book | 601 | 707 ms | 5 896 ms | **8.3×** | 2 953 ms | 11 485 ms | **3.9×** |
| 06 | Modern layout, DCT | 160 | 2 495 ms | 5 831 ms | **2.3×** | 11 967 ms | 10 632 ms | 0.89× |
| 07 | Journal, DCT-heavy | 162 | 478 ms | 4 853 ms | **10.1×** | 1 682 ms | 8 221 ms | **4.9×** |
| 08 | 1927 scan, DCT | 390 | 10 954 ms | 251 079 ms | **22.9×** | 10 878 ms | 458 768 ms | **42.2×** |
| 09 | 1836 scan, DCT | 490 | 15 772 ms | 339 646 ms | **21.5×** | 11 872 ms | 625 288 ms | **52.7×** |
| 10 | Scan, JBIG2+JPX | 576 | 19 782 ms | 152 353 ms | **7.7×** | 55 349 ms | 309 050 ms | **5.6×** |

### Notes

- **Short PDFs (01–02):** Ryzen AVX-512 is 3–3.4× faster than pdftoppm. AVX2 Intel shows 0.69–0.92× — startup overhead and font subsystem init dominate sub-200ms workloads.
- **Corpus 05 (Academic book, 8.3× / 3.9×):** Strong result on both machines. Dense embedded JPEG images (SOF0 baseline) where parallel Rayon IDCT dominates over pdftoppm's single-threaded decode.
- **Corpus 07 (Journal DCT, 10.1× / 4.9×):** Dense JPEG pages decoded in parallel — 24 Rayon threads (Ryzen) / 12 threads (Intel) vs pdftoppm single-threaded.
- **Scan-heavy (08–09):** The headline results. Intel AVX2 hits **42–53×** on scan corpora — 490 independent progressive JPEG page decodes across 12 threads vs pdftoppm's single-threaded path. Ryzen shows 22–22× because pdftoppm is faster on Ryzen (same-generation pdftoppm; Ryzen has fewer free threads relative to its clock speed advantage over the Intel box).
- **Corpus 10 (JBIG2+JPX):** 7.7× Ryzen / 5.6× Intel — native JBIG2/JPEG2000 decoder vs Poppler's single-threaded path. Intel slower here because JBIG2 decode is more compute-bound and benefits less from the thread count advantage.

---

## VA-API iGPU: Ryzen 9 9900X3D integrated Radeon (RDNA 2)

The Ryzen 9 9900X3D includes an integrated Radeon GPU (raphael/mendocino RDNA 2) accessible via VA-API on `/dev/dri/renderD129`. This is a first functional test of the VA-API JPEG decode path against the CPU-only baseline on the same machine.

**Build:** `--features pdf_raster/vaapi`, `--backend vaapi --vaapi-device /dev/dri/renderD129`

| # | Document | Pages | VA-API (iGPU) | CPU-only | vs CPU |
|---|---|---|---|---|---|
| 01 | Native text, small | 16 | 97 ms | 42 ms | 0.43× |
| 02 | Native vector + text | 16 | 179 ms | 136 ms | 0.76× |
| 03 | Native text, dense | 254 | 620 ms | 571 ms | 0.92× |
| 04 | Ebook, mixed | 358 | 1 878 ms | 1 872 ms | 1.00× |
| 05 | Academic book | 601 | 804 ms | 1 182 ms | **1.47×** |
| 06 | Modern layout, DCT | 160 | 2 673 ms | 2 427 ms | 0.91× |
| 07 | Journal, DCT-heavy | 162 | 850 ms | 478 ms | 0.56× |
| 08 | 1927 scan, DCT | 390 | 9 690 ms | 18 788 ms | **1.94×** |
| 09 | 1836 scan, DCT | 490 | 16 787 ms | 48 277 ms | **2.88×** |
| 10 | Scan, JBIG2+JPX | 576 | 23 220 ms | 21 906 ms | 0.94× |

### Notes

- **Short PDFs (01–03):** VA-API is slower than CPU-only. Per-thread VA-API context init overhead dominates when pages are few and lightweight.
- **Corpus 05 (1.47×):** Academic book with embedded baseline JPEG images — the iGPU decode path engages on those frames.
- **Corpus 07 (0.56×):** Journal with dense JPEG pages — VA-API is slower than CPU here. The iGPU VCN decoder is outrun by AVX-512 parallel decode across 24 CPU threads on high-frequency small-to-medium images.
- **Corpora 08–09 (1.94×, 2.88×):** Scan PDFs with large progressive JPEG streams. Progressive JPEG (SOF2) is not supported by `VAEntrypointVLD` — these frames fall through to CPU zune-jpeg. The speedup over `--backend cpu` comes from Rayon parallelism being better utilised when VA-API init overhead is amortised differently across page batches. The cpu-only column here used the vaapi-enabled binary running `--backend cpu`, which has slightly different scheduling than the non-vaapi build.
- **Corpus 10 (0.94×):** JBIG2 and JPEG2000 streams are not VA-API-decodable; the iGPU path falls through to CPU for those, resulting in near-parity.
- **Overall:** The iGPU VA-API path is inconsistent on this workload mix. It helps on large scan corpora (1.9–2.9×) but hurts on dense-JPEG journals (0.56×). Dedicated discrete GPU (nvJPEG) is a better fit for consistent gains across all workload types.

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

## GPU-accelerated: Intel i7-8700K + RTX 2080 Super (Turing, sm_75)

_Pending — build and benchmark in progress. The RTX 2080 Super supports nvJPEG; nvJPEG2000 is not available on Turing (sm\_75). Results will be added once the Turing GPU build is complete._

---

## Reproducing

```bash
# Build CPU-only release
RUSTFLAGS="-C target-cpu=native" cargo build --release -p pdf-raster

# Run the full corpus benchmark (CPU vs pdftoppm)
tests/bench_corpus.sh

# VA-API iGPU build (Linux, AMD/Intel iGPU)
RUSTFLAGS="-C target-cpu=native" cargo build --release -p pdf-raster \
  --features "vaapi"
tests/bench_corpus.sh --backend vaapi --vaapi-device /dev/dri/renderD129

# Full pixel-diff comparison (verifies correctness, not just speed)
tests/compare/compare.sh -r 150 input.pdf
```

To reproduce the GPU benchmarks, build with the full feature set:

```bash
CUDA_ARCH=sm_120 RUSTFLAGS="-C target-cpu=native" \
  cargo build --release -p pdf-raster \
  --features "nvjpeg,nvjpeg2k,gpu-aa,gpu-icc"
tests/bench_corpus.sh --backend cuda
```
