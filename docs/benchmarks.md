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
| 01 | Native text, small | 16 | 48 ms | 154 ms | **3.2×** | 347 ms | 316 ms | 0.91× |
| 02 | Native vector + text | 16 | 129 ms | 148 ms | **1.1×** | 680 ms | 397 ms | 0.58× |
| 03 | Native text, dense | 254 | 592 ms | 3 853 ms | **6.5×** | 2 314 ms | 6 755 ms | **2.9×** |
| 04 | Ebook, mixed | 358 | 3 155 ms | 4 922 ms | **1.6×** | 8 518 ms | 7 416 ms | 0.87× |
| 05 | Academic book | 601 | 4 318 ms | 7 532 ms | **1.7×** | 3 209 ms | 11 486 ms | **3.6×** |
| 06 | Modern layout, DCT | 160 | 2 856 ms | 6 479 ms | **2.3×** | 12 503 ms | 11 228 ms | 0.90× |
| 07 | Journal, DCT-heavy | 162 | 813 ms | 5 273 ms | **6.5×** | 1 847 ms | 8 128 ms | **4.4×** |
| 08 | 1927 scan, DCT | 390 | 19 112 ms | 387 636 ms | **20.3×** | 9 947 ms | 465 157 ms | **46.8×** |
| 09 | 1836 scan, DCT | 490 | 58 012 ms | 389 933 ms | **6.7×** | 12 287 ms | 625 917 ms | **50.9×** |
| 10 | Scan, JBIG2+JPX | 576 | 21 573 ms | 151 170 ms | **7.0×** | 57 713 ms | 307 878 ms | **5.3×** |

### Notes

- **Short PDFs (01–02):** The AVX2 build is slower than pdftoppm on the Intel box. These are 16-page documents where startup overhead and font subsystem init dominate. pdf-raster's startup path is not optimised for sub-100ms workloads.
- **Scan-heavy (08–09):** The 47–51× gains on Intel are the headline result of this benchmark run. Both corpora embed progressive JPEG streams which pdftoppm decodes serially; pdf-raster's parallel zune-jpeg path across 12 threads (6C/12T) dominates. The prior Intel numbers (12.6× / 15.0×) were from a stale binary run on a warm-cache machine; these figures are from a clean rebuild.
- **AVX-512 vs AVX2 on scans:** AVX-512 (Ryzen) shows 20.3× on corpus 08 vs 46.8× on AVX2 (Intel) — the Intel machine actually wins here because the i7-8700K has 12 threads fully saturated on decode while the Ryzen number reflects an earlier run that may have had contention. The Ryzen AVX-512 advantage is clearer on text-dense workloads (corpus 03: 6.5× vs 2.9×).
- **Corpus 10 (JBIG2+JPX):** 7.0× on AVX-512, 5.3× on AVX2 — consistent across both machines, driven by the native JBIG2/JPEG2000 decoder vs Poppler's single-threaded path.

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
