# Phase 11 contest results — GPU re-bench

**Run:** 2026-05-09T14:33:55-04:00
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
| pdf-raster | 1108.6 | **1117.6** |
| mutool draw | — | 79.8 |
| pdftoppm | — | NA |

## E2 — sustained (pages 50000–50099)

| Engine | Cold (ms) | Warm median (ms) | Per-page (ms) |
|---|---|---|---|
| pdf-raster | 1736.9 | **1734.9** | 17.3 |

## E3 — cross-doc (100 archives, page 1 each)

| Engine | Cold (ms) | Warm median (ms) | Per-archive (ms) |
|---|---|---|---|
| pdf-raster | 14611.2 | **14435.9** | 144.4 |

## E4 — random-access (1000 pages)

| Engine | Cold (ms) | Warm median (ms) | Per-page (ms) |
|---|---|---|---|
| pdf-raster | 14237.0 | **13968.0** | 14.0 |

## E5 — single DCT page (corpus-08 p100 @ 300 DPI, end-to-end including write)

| Engine | Cold (ms) | Warm median (ms) |
|---|---|---|
| pdf-raster | 309.8 | **320.8** |
| mutool draw | — | 73.9 |
| pdftoppm | — | 942.2 |

---

Raw data: `bench/v11-gpu/results.csv`.  Full run log: `bench/v11-gpu/run.log`.
