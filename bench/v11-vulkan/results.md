# Phase 11 contest results — variant 'vulkan'

**Run:** 2026-05-09T14:38:39-04:00
**Hardware:** AMD Ryzen 9 9900X3D 12-Core Processor + NVIDIA GeForce RTX 5070, Linux 6.17.0-23-generic
**Archive:** 2.78 GB synthetic PDF
**Cross-doc set:** 100 archives at ~300 MB each
**Methodology:** 1 cold run + 4 warm runs per event; warm median reported.
**Build features:** `vulkan,gpu-aa,gpu-icc` on pdf_raster (no PGO)
**Backend:** `CONTEST_BACKEND=vulkan` on the bench, `--backend vulkan` on the standalone CLI for E5

## E1 — first-pixel (page 50000)

| Engine | Cold (ms) | Warm median (ms) |
|---|---|---|
| pdf-raster | 958.7 | **116.2** |
| mutool draw | — | 78.7 |
| pdftoppm | — | NA |

## E2 — sustained (pages 50000–50099)

| Engine | Cold (ms) | Warm median (ms) | Per-page (ms) |
|---|---|---|---|
| pdf-raster | 1947.9 | **1975.8** | 19.8 |

## E3 — cross-doc (100 archives, page 1 each)

| Engine | Cold (ms) | Warm median (ms) | Per-archive (ms) |
|---|---|---|---|
| pdf-raster | 343.0 | **342.5** | 3.4 |

## E4 — random-access (1000 pages)

| Engine | Cold (ms) | Warm median (ms) | Per-page (ms) |
|---|---|---|---|
| pdf-raster | 14129.3 | **14222.1** | 14.2 |

## E5 — single DCT page (corpus-08 p100 @ 300 DPI)

| Engine | Cold (ms) | Warm median (ms) |
|---|---|---|
| pdf-raster | 260.0 | **188.1** |
| mutool draw | — | 71.0 |
| pdftoppm | — | 925.8 |

---

Raw data: `bench/v11-vulkan/results.csv`.  Full run log: `bench/v11-vulkan/run.log`.
