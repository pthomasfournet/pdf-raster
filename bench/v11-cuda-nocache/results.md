# Phase 11 contest results — variant 'cuda-nocache'

**Run:** 2026-05-09T14:36:49-04:00
**Hardware:** AMD Ryzen 9 9900X3D 12-Core Processor + NVIDIA GeForce RTX 5070, Linux 6.17.0-23-generic
**Archive:** 2.78 GB synthetic PDF
**Cross-doc set:** 100 archives at ~300 MB each
**Methodology:** 1 cold run + 4 warm runs per event; warm median reported.
**Build features:** `gpu-aa,gpu-icc` on pdf_raster (no PGO)
**Backend:** `CONTEST_BACKEND=cuda` on the bench, `--backend cuda` on the standalone CLI for E5

## E1 — first-pixel (page 50000)

| Engine | Cold (ms) | Warm median (ms) |
|---|---|---|
| pdf-raster | 146.0 | **144.9** |
| mutool draw | — | 79.4 |
| pdftoppm | — | NA |

## E2 — sustained (pages 50000–50099)

| Engine | Cold (ms) | Warm median (ms) | Per-page (ms) |
|---|---|---|---|
| pdf-raster | 1943.3 | **1946.8** | 19.5 |

## E3 — cross-doc (100 archives, page 1 each)

| Engine | Cold (ms) | Warm median (ms) | Per-archive (ms) |
|---|---|---|---|
| pdf-raster | 373.7 | **369.6** | 3.7 |

## E4 — random-access (1000 pages)

| Engine | Cold (ms) | Warm median (ms) | Per-page (ms) |
|---|---|---|---|
| pdf-raster | 14265.1 | **14460.3** | 14.5 |

## E5 — single DCT page (corpus-08 p100 @ 300 DPI)

| Engine | Cold (ms) | Warm median (ms) |
|---|---|---|
| pdf-raster | 249.8 | **251.2** |
| mutool draw | — | 73.2 |
| pdftoppm | — | 937.3 |

---

Raw data: `bench/v11-cuda-nocache/results.csv`.  Full run log: `bench/v11-cuda-nocache/run.log`.
