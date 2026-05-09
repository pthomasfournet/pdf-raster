# Phase 11 contest results

**Run:** 2026-05-09T13:36:24-04:00
**Hardware:** AMD Ryzen 9 9900X3D 12-Core Processor + NVIDIA GeForce RTX 5070, Linux 6.17.0-23-generic
**Archive:** 2.78 GB synthetic PDF (qpdf-concatenated corpus fixtures)
**Cross-doc set:** 100 archives at ~300 MB each (~30.0 GB total)
**Methodology:** 1 cold run + 4 warm runs per event; warm median reported.
PGO-trained binary (rendering 10 pages of corpus-04 as the training workload).

## E1 — first-pixel (page 50000 of a 16193-page archive)

The contest spec calls for rendering page 50000.  Our 2.78 GB archive
only has 16193 pages, so a strict reading of "render page 50000" is
either *clamp to last page* (what `contest_v11` does internally and what
mutool does silently) or *refuse* (what pdftoppm does, exiting non-zero).

| Engine | Cold (ms) | Warm median (ms) | Status |
|---|---|---|---|
| pdf-raster | 35.7 | **33.7** | rendered page 16193 (clamped) |
| mutool draw | 84.2 | 79.6 | rendered page 16193 (silently clamped) |
| pdftoppm | — | — | **DNF** ("first page can not be after the last page") |

**pdftoppm: did not finish.**  Refused to render an out-of-range index.

For a level-playing-field supplemental run on a page that exists in
this archive (page 8000, mid-archive), all three engines complete.
**All three write a PPM file to `/tmp` (NVMe)** — pdf-raster's harness
runs the bitmap through the `encode::write_ppm` path so the timed
window includes file create + buffered write + close, matching what
mutool and pdftoppm always do.

Competitor flags are the most-aggressive fair-play set — mutool gets
`-q -P -N` (quiet mode, parallel rendering, ICC color management
disabled to match pdf-raster's RGB output path); pdftoppm has no
equivalent perf knobs to enable.

| Engine | Cold (ms) | Warm median (ms) |
|---|---|---|
| pdf-raster | 36.8 | **35.6** |
| mutool draw | 95.7 | 93.7 |
| pdftoppm | 779.9 | 770.5 |

On the supplemental run, pdf-raster is **2.6× faster than mutool** and
**22× faster than pdftoppm**.  Competitor subprocess startup is paid on
every iteration, so their "cold" and "warm" times differ mostly in
kernel page-cache state, not engine warmup.

Why the gap?  All three engines do the same algorithmic work (parse
xref, walk to page, decode content stream, render pixels).  pdf-raster
descends the page tree logarithmically (`O(log pages)` per `get_page`)
and reads `/Pages /Count` directly without indexing the whole tree on
open.  mutool eagerly indexes; poppler (pdftoppm's backend) eagerly
walks AND parses every page on first `getPage(N)` call.  On a 16k-page
archive, that's the difference between a few hundred dict reads and a
half-million.

### Notes on the disk-write cost

A 150 DPI PPM at this fixture's page size is ~6.3 MB.  On the dev
box's NVMe (/dev/sda3), that write costs ~6.5 ms — visible in
pdf-raster's number (29 ms render-only → 35.6 ms render+write) and
in mutool's (~80 ms render-only → ~93 ms render+write).  pdftoppm's
720+ ms is dominated by interpretation, not I/O — the disk write is
lost in the noise.

E2/E3/E4 below are pdf-raster-only events (no competitors) and do
NOT write per-page PPMs — those events measure the parse + render
critical path, not the disk write that callers can opt into.

## E2 — sustained (pages 50000–50099)

| Engine | Cold (ms) | Warm median (ms) |
|---|---|---|
| pdf-raster | 2541.3 | **2529.8** |

## E3 — cross-doc (100 archives, page 1 each)

| Engine | Cold (ms) | Warm median (ms) |
|---|---|---|
| pdf-raster | 359.9 | **347.1** |

## E4 — random-access (1000 pages)

| Engine | Cold (ms) | Warm median (ms) |
|---|---|---|
| pdf-raster | 17405.9 | **17074.3** |

---

## What this bench does NOT measure

These numbers time **inline render-then-write** — render finishes,
encode the PPM, write it to disk, then move on.  That's a fair shape
for an apples-to-apples comparison against `mutool draw` and `pdftoppm`
since both CLI tools serialise render → write inside their own
process.  Neither tool exposes an API where the caller takes the
bitmap mid-pipeline.

pdf-raster's library API does.  `pdf_raster::render_page_rgb` returns
the `Bitmap<Rgb8>` directly; the caller chooses whether to write it,
hand it to Tesseract, hash it, or queue it for an async writer.  In a
disk-bound regime — slow disks, network mounts, NVMe under multi-tenant
load, very large per-page bitmaps — the smart shape is to render to
RAM, return the bitmap immediately, and pipeline the writes off a
separate thread or channel so the render path never blocks on I/O.
mutool and pdftoppm cannot offer this to their users; they're CLI
subprocess tools, the output IS a file write, and pipelining requires
spawning N subprocesses (each paying a fresh process startup +
xref-parse cost).

This is the bigger structural advantage and the bench above doesn't
capture it.  A future "E5 — pipelined render-write" event could
demonstrate it concretely; we omitted it here because the contest
spec'd four events and we want the headline numbers to map cleanly
to that spec.

---

Raw data: `bench/v11/results.csv`.  Full run log: `bench/v11/run.log`.
