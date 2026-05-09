# Phase 11 contest results — GPU re-bench

**Run:** 2026-05-09T14:26:25-04:00
**Hardware:** AMD Ryzen 9 9900X3D 12-Core Processor + NVIDIA GeForce RTX 5070, Linux 6.17.0-23-generic
**Archive:** 2.78 GB synthetic PDF (qpdf-concatenated corpus fixtures)
**Cross-doc set:** 100 archives at ~300 MB each
**Methodology:** 1 cold run + 4 warm runs per event; warm median reported.
**Build features:** `pdf_raster/cache,pdf_raster/gpu-aa,pdf_raster/gpu-icc` (no PGO).
**nvjpeg:** intentionally omitted — `GPU_JPEG_THRESHOLD_PX = u32::MAX` gates dispatch
at the call site, so the feature flag is a no-op for benchmarking.

## E1 — first-pixel (page 50000)

| Engine | Cold (ms) | Warm median (ms) |
|---|---|---|
| pdf-raster | 1449.9 | **1134.7** |
| mutool draw | — | 79.5 |
| pdftoppm | — | NA |

## E2 — sustained (pages 50000–50099)

| Engine | Cold (ms) | Warm median (ms) | Per-page (ms) |
|---|---|---|---|
| pdf-raster | 1741.9 | **1762.5** | 17.6 |

## E3 — cross-doc (100 archives, page 1 each)

| Engine | Cold (ms) | Warm median (ms) | Per-archive (ms) |
|---|---|---|---|
| pdf-raster | 24289.8 | **24091.2** | 240.9 |

## E4 — random-access (1000 pages)

| Engine | Cold (ms) | Warm median (ms) | Per-page (ms) |
|---|---|---|---|
| pdf-raster | 13719.8 | **13634.9** | 13.6 |

## E5 — single DCT page (corpus-08 p100 @ 300 DPI, end-to-end including write)

| Engine | Cold (ms) | Warm median (ms) |
|---|---|---|
| pdf-raster | 329.7 | **323.6** |
| mutool draw | — | 72.5 |
| pdftoppm | — | 928.2 |

---

Raw data: `bench/v11-gpu/results.csv`.  Full run log: `bench/v11-gpu/run.log`.
