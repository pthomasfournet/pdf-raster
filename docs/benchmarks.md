# Benchmarks

## Methodology

All benchmarks render all pages of each corpus PDF at **150 DPI** (the default) to PPM output, measuring wall-clock time with millisecond precision. Each tool is run sequentially (one PDF at a time, one tool at a time) to avoid contention. Output files are written to a tmpfs temporary directory and discarded. The input PDF is evicted from the OS page cache before each timed run via `posix_fadvise(FADV_DONTNEED)`, ensuring cold-cache reads that reflect real-world single-run usage.

**Tool versions:**

| Tool | Version |
|---|---|
| pdf-raster | v0.5.1, built with `-C target-cpu=native` |
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

| Machine | CPU | ISA | Cores | RAM | Storage |
|---|---|---|---|---|---|
| Ryzen bench | AMD Ryzen 9 9900X3D @ 4.4 GHz | x86-64 + AVX-512 | 12C/24T | 32 GB DDR5 | SATA SSD |
| Intel bench | Intel Core i7-8700K @ 3.7 GHz | x86-64 + AVX2 | 6C/12T | 32 GB DDR4 | SATA SSD |

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
| 01 | Native text, small | 16 | 473 ms | 43 ms | 0.09× |
| 02 | Native vector + text | 16 | 170 ms | 122 ms | 0.72× |
| 03 | Native text, dense | 254 | 591 ms | 586 ms | 0.99× |
| 04 | Ebook, mixed | 358 | 1 832 ms | 1 788 ms | 0.98× |
| 05 | Academic book | 601 | 752 ms | 705 ms | 0.94× |
| 06 | Modern layout, DCT | 160 | 2 785 ms | 2 818 ms | 1.01× |
| 07 | Journal, DCT-heavy | 162 | 889 ms | 827 ms | 0.93× |
| 08 | 1927 scan, DCT | 390 | 9 979 ms | 10 312 ms | 1.03× |
| 09 | 1836 scan, DCT | 490 | 17 278 ms | 41 694 ms | **2.41×** |
| 10 | Scan, JBIG2+JPX | 576 | 20 899 ms | 21 235 ms | 1.02× |

_All corpora: fresh uncontested cold-cache runs._

### Notes

- **Short PDFs (01–02):** VA-API is significantly slower than CPU-only. Corpus-01 (473 ms vs 43 ms) shows the VA-API context init cost dominating completely — each page requires a fresh `VAContext` + `VASurface` allocation.
- **Corpora 03–05 (~0.94–0.99×):** Near-parity. VA-API engages on embedded SOF0 baseline JPEG images but context creation overhead cancels the decode speedup.
- **Corpora 06–08, 10 (~1.0×):** Essentially neutral. VA-API falls through to CPU for progressive JPEG (SOF2) and JBIG2/JPX streams; any marginal win from VA-API on SOF0 images is offset by init overhead.
- **Corpus 09 (2.41×):** Largest scan corpus (490 pages). The VA-API path amortises init cost better at this scale; the cpu-only comparison uses the vaapi-enabled binary on `--backend cpu` which has slightly different Rayon scheduling.
- **Overall:** iGPU VA-API is largely neutral on this workload mix. The only meaningful win is corpus-09 (2.41×), driven by scheduling differences rather than hardware JPEG decode (all frames are progressive SOF2, unsupported by `VAEntrypointVLD`). Dedicated GPU (nvJPEG) is the right tool for consistent gains.

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

Built with nvJPEG, GPU AA fill, and ICC CLUT features. nvJPEG2000 is not available on Turing (sm_75); JBIG2/JPEG2000 streams fall through to CPU. Compared against `pdftoppm` (CPU only) on the same machine.

**Build:** `CUDA_ARCH=sm_75 LIBZ_SYS_STATIC=1 RUSTFLAGS="-C target-cpu=native" cargo build --release -p pdf-raster --features "pdf_raster/nvjpeg,pdf_raster/gpu-aa,pdf_raster/gpu-icc"`

| # | Document | Pages | pdf-raster (GPU) | pdftoppm | Speedup |
|---|---|---|---|---|---|
| 01 | Native text, small | 16 | 658 ms | 582 ms | 0.88× |
| 02 | Native vector + text | 16 | 806 ms | 610 ms | 0.76× |
| 03 | Native text, dense | 254 | 2 671 ms | 7 256 ms | **2.7×** |
| 04 | Ebook, mixed | 358 | 8 556 ms | 7 554 ms | 0.88× |
| 05 | Academic book | 601 | 3 527 ms | 12 043 ms | **3.4×** |
| 06 | Modern layout, DCT | 160 | 12 219 ms | 11 241 ms | 0.92× |
| 07 | Journal, DCT-heavy | 162 | 14 222 ms | 8 319 ms | 0.58× |
| 08 | 1927 scan, DCT | 390 | 12 284 ms | 473 651 ms | **38.6×** |
| 09 | 1836 scan, DCT | 490 | 14 003 ms | 633 098 ms | **45.2×** |
| 10 | Scan, JBIG2+JPX | 576 | 57 850 ms | 311 040 ms | **5.4×** |

### Notes

- **Short PDFs (01–02):** GPU init overhead exceeds any decode savings at 16 pages. 0.76–0.88× expected.
- **Corpus 07 (0.58×):** Dense small JPEG pages — nvJPEG dispatch overhead per image exceeds decode savings. The RTX 2080 Super's nvJPEG throughput is outrun by the 12-thread CPU path for high-frequency small images.
- **Corpora 08–09 (38.6×, 45.2×):** Headline results. Large progressive JPEG scan pages offloaded to nvJPEG (Turing supports progressive JPEG unlike VA-API). pdftoppm takes 8–10 minutes single-threaded; GPU finishes in 12–14 seconds.
- **Corpus 10 (5.4×):** JBIG2+JPX — no nvJPEG2000 on Turing, falls through to CPU. Speedup is purely from Rayon parallelism vs pdftoppm single-threaded, same as the CPU-only result.
- **vs CPU-only on same machine:** GPU adds overhead on corpora 01–07 relative to the CPU-only run (GPU init cost). The win is exclusively on scan-heavy corpora 08–09 where nvJPEG offloads large JPEG decode from the CPU thread pool.

---

## Version regression history (CPU-only, cold cache)

All five minor versions benchmarked on the Ryzen 9 9900X3D, CPU-only backend, 150 DPI, cold cache. No pdftoppm reference — this table is purely for tracking regression and improvement across releases.

Built with `RUSTFLAGS="-C target-cpu=native"` at each tagged commit. v0.1.0–v0.3.0 use `glibc malloc`; v0.4.0+ use `mimalloc`.

| Corpus | v0.1.0 | v0.2.0 | v0.3.0 | v0.4.0 | v0.5.1 |
|---|---|---|---|---|---|
| 01 native text, small | 45 ms | 50 ms | 41 ms | 47 ms | 48 ms |
| 02 native vector + text | 129 ms | 138 ms | 126 ms | 138 ms | 145 ms |
| 03 native text, dense | 598 ms | 622 ms | 591 ms | 599 ms | 595 ms |
| 04 ebook, mixed | 1 784 ms | 1 944 ms | 1 824 ms | 1 836 ms | 1 916 ms |
| 05 academic book | 743 ms | 763 ms | 751 ms | 760 ms | 785 ms |
| 06 modern layout, DCT | 2 663 ms | 2 718 ms | 2 658 ms | 2 698 ms | 2 662 ms |
| 07 journal, DCT-heavy | 768 ms | 776 ms | 761 ms | 760 ms | 757 ms |
| 08 1927 scan, DCT | 11 431 ms | 12 882 ms | 12 616 ms | 13 711 ms | 12 601 ms |
| **09 1836 scan, DCT** | **35 502 ms** | **77 132 ms** | **58 232 ms** | **60 137 ms** | **36 661 ms** |
| 10 scan, JBIG2+JPX | 18 703 ms | 18 405 ms | 18 350 ms | 18 492 ms | 18 069 ms |

### What the numbers show

**Corpora 01–07 and 10 are essentially flat.** The native text, vector, JBIG2, and JPEG 2000 render paths have not regressed or improved in wall time across any release — variance is noise. This is both reassuring (no regressions) and honest (no CPU-path gains either).

**Corpus 08 (baseline JPEG, 390 pages)** shows mild drift: +1–2 s added over v0.1.0 baseline. The renderer adds overhead with each release — more dispatch layers, more feature flag branches, more plumbing through the call stack — and it accumulates. At ~12 s total the regression is ~10% over five versions. Not catastrophic but real.

**Corpus 09 (progressive JPEG, 490 pages) is the headline.** v0.1.0 ran in 35 s. v0.2.0 blew up to 77 s — a 2.2× regression. v0.3.0 partially recovered (58 s), v0.4.0 remained slow (60 s), and v0.5.1 recovered to 36 s, nearly back to v0.1.0.

The cause: v0.2.0 introduced VA-API JPEG dispatch (`feat(gpu/interp): GPU decoder traits + inline image GPU dispatch`). Even on a CPU-only build, every progressive JPEG frame attempted VA-API decode first, failed (VA-API is baseline-only), and fell through to zune-jpeg — paying parse overhead for every single frame in a 490-page progressive-JPEG corpus. v0.5.1 fixed this with SOF-aware routing (`feat(pdf_interp): content-aware JPEG dispatch`) that short-circuits progressive JPEG directly to zune-jpeg without touching VA-API.

### Lessons

The version history makes one thing clear: **every new dispatch layer has a cost that shows up in cold benchmarks even when the feature is not active.** Adding GPU/VA-API paths adds branching and initialization overhead that runs on every JPEG frame regardless of backend. This is the nature of a feature-rich pipeline — each capability added to the fast path is overhead added to all paths.

The practical implication: **re-run this table before every release.** A new feature that looks neutral in isolation may show a corpus-09-style blowup only when measured across a full cold-cache run. `tests/bench_versions.sh` automates the build and timing for all versions.

---

## Parallelism diagnostic traps

These are failure modes that have caused multi-day benchmark investigations. Check them in order before touching domain code.

### 1. Low CPU utilisation ≠ I/O bound

20–30% CPU across all threads on a JPEG-heavy corpus is not a sign of disk bottleneck. It can be allocator contention. glibc malloc serializes under concurrent large allocations via a futex — threads appear idle but are queued on a lock. This is invisible in `top`/`htop` (blocked threads show ~0% user and ~0% sys).

**First check:** run with `--timings`. If pages of similar size show wildly different wall times on different threads, something shared is serializing them.

```bash
./target/release/pdf-raster --timings corpus.pdf /tmp/out/ 2>&1 | grep timing
```

**Second check:** `perf record`, then look for allocator symbols.

```bash
perf record -g ./target/release/pdf-raster corpus.pdf /tmp/out/
perf report   # look for: malloc, free, __lll_lock_wait, _int_malloc
```

If allocator symbols are hot: the fix is already in place (mimalloc is the global allocator in the CLI binary). Any new binary or benchmark harness must also use mimalloc.

### 2. GPU backend serialization

`NVJPEG_BACKEND_HARDWARE` routes all decodes through a single hardware engine — effective single-threading regardless of Rayon pool size. The CLI uses `GPU_HYBRID` (per-thread software pipeline + hardware assist), which is correct for multi-threaded workloads. If a new GPU path shows unexpected serialization, verify the backend constant.

### 3. VA-API per-frame context overhead

VA-API creates/destroys `VAContext`+`VASurface` per decode unless context reuse is active (added in `perf(vaapi): reuse VAContext+VASurface`). This overhead eats the decode win for high-frequency small JPEG pages. If VA-API benchmarks show no gain over CPU, check whether context reuse is triggering (same dimensions across frames required).

### 4. Serial prescan on the hot path

A serial pre-scan pass over all pages before rendering starts blocks the first render thread until all pages are scanned. This was removed (see `refactor(cli): remove serial prescan`). If prescan-like logic is re-introduced, ensure it runs inline per render thread, not as a sequential preflight.

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
