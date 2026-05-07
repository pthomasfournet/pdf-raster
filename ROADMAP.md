# pdf-raster Roadmap

Goal: full PDF → pixels pipeline in pure Rust. Zero poppler. Zero C++ in the render path.

The raster crate is complete at the pixel level. The `pdf_interp` crate is the native renderer and is now the only CLI path. The `pdf_bridge` / poppler crate is retained as a reference baseline but is no longer linked by the CLI binary.

**Integration target (Apr 2026):** pdf-raster replaces steps 3 (pdftoppm subprocess) and 4 (Leptonica preprocessing) in an OCR pipeline:

```
pdf_oxide → [quality check fails] → pdf-raster (rasterise + deskew) → Tesseract → (LLM correct)
```

The caller's Tesseract step becomes a single in-process call — no subprocess, no files, no Leptonica:

```rust
let page = pdf_raster::render_page(path, page_num, &opts)?;
// page.pixels is 8-bit grayscale, tightly packed, top-to-bottom
let text = tesseract::ocr_from_frame(
    &page.pixels, page.width as i32, page.height as i32,
    1, page.width as i32, "eng",
)?;
```

Phase 5 is complete. The API exists and is integrated.

---

## Release history

### v0.7.0 (May 2026)

**New since v0.6.0:**

- **Device-resident image cache (3-tier).** New `cache` feature on the `gpu` and `pdf_interp` crates wires a `DeviceImageCache` with three tiers: VRAM (primary, refcount-pinned LRU), pinned host RAM (demote-on-evict / promote-on-hit), and disk (`<root>/<doc-sha256>/<content-hash>.bin` sidecar files for cross-session persistence). Keys: BLAKE3 content hash (cross-document dedup) + `(DocId, ObjId)` alias (same-document fast path). Disk writes are atomic via temp+rename, gated on env vars `PDF_RASTER_CACHE_DIR` / `PDF_RASTER_CACHE_BYTES`, invalidated automatically when the source PDF changes (DocId is BLAKE3 of the bytes).
- **Device-resident page buffer + GPU image blit.** New `crates/gpu/kernels/blit_image.cu` 16×16-block kernel with f32 inverse-CTM nearest-neighbour sampling that matches the CPU path byte-for-byte (verified by an in-tree CPU-reference parity test). `DevicePageBuffer` is lazy-allocated on first GPU image; source-over composite onto the host bitmap happens in one `cudaMemcpyAsync` at `PageRenderer::finish`. `ImageData::Gpu(Arc<CachedDeviceImage>)` is the cached-decode product `decode_dct` returns when the cache is on.
- **Image-cache prefetcher.** `pdf_interp::cache::spawn_prefetch` walks every page's `/XObject` resource dict at session open, dedupes by `ObjId`, and decodes `/DCTDecode` images on a small `std::thread` worker pool (default 2, capped at `MAX_PREFETCH_WORKERS = 16`). Decoder panics caught per-image so one bad XObject doesn't kill the run. Opt-in via `SessionConfig::prefetch`; default off because eager resource-dict walks are wasted work for short single-page renders. `RasterSession.doc` upgraded to `Arc<Document>` so the prefetcher can hold its own clone without changing how the renderer borrows.
- **JPEG scaffolding correctness fixes (`crates/gpu/src/jpeg/`).** RST predictor reset is now driven by MCU index (`mcu_idx % restart_interval == 0`) instead of the bit reader's byte position — the byte-position chase worked by incidental ordering but a truncated MCU could leave the cursor short of the marker and silently skip the predictor reset. The `aa_fill.cu` `JITTER_Y` table had 8 wrong Halton(3) values at indices 17–23 and 44–47, found while bringing up gpu-validation tests; CPU `HALTON3` in `fill.rs` is now the source of truth.
- **JPEG scaffolding cleanup.** Collapsed the double SOF scan in `JpegHeaders::parse` (non-baseline detection inline in the marker loop, no separate `jpeg_sof_type` pre-pass). `BitReader::refill` grew an 8-byte `u64::from_be_bytes` fast path on the common cap-zero case (~2× Huffman codeword throughput per textbooks). `canonical::fill_table` switched to `slice::fill`. VA-API adapter no longer caches `num_mcus` — derives from a shared `mcu_count_from_max_sampling` helper.
- **Documentation.** README gains a "Picking CUDA_ARCH for your GPU" subsection mapping consumer GPU generations (Pascal → Blackwell, A100, H100) to the right `sm_NN` flag, plus a feature-flag table covering `nvjpeg`, `nvjpeg2k`, `gpu-aa`, `gpu-icc`, `gpu-deskew`, `cache`, `vaapi`. Build script default of `sm_80` is documented inline.

**Bench gate (pending):** the device-resident-image-cache spec defines five acceptance criteria (cache hit rate ≥ 95% on logo-heavy multi-page renders, second-render time ≤ 30% of first-render, mode A no regression, no OOM on corpus 09 with default budgets, mode D beats mode A on at least 3 of corpora 04–08). v0.7.0 ships the implementation; the bench gate has not yet been run, so the architectural payoff is structurally complete but not yet validated end-to-end.

### v0.6.0 (May 2026)

**New since v0.5.1:**

- **lopdf rip-out — in-tree `pdf` crate.** Replaced lopdf 0.40 with `crates/pdf/`: a lazy mmap-based parser that reads only the xref table and trailer at `Document::open` and resolves objects on demand via byte-offset seek. Per-object `Arc` cache + mutex; ObjStm decompression cached once across worker threads. API surface (`Object`, `Dictionary` newtype, `Stream`, `ObjectId`, `PdfError`) mirrors the lopdf names previously used so the migration was mechanical. DOS-hardened: caps on xref entries (10M), `/N` (1M), PNG predictor output (256 MiB); `checked_add` throughout. `pdf_interp` (17 files) and `pdf_raster` swapped over file-by-file; lopdf is gone from the entire workspace. Motivation: lopdf's `load_objects_raw` had been burning ~20% of corpus-07 cycles in `nom_locate`'s `memchr` on the main thread before render workers could start, capping CPU utilisation at ~1.6 of 24 cores. Cold-cache corpus-07 went from 757 ms → 689 ms.
- **RAM-backed output by default.** Disk I/O was hiding actual CPU work — the previous temp-file + atomic-rename pattern triggered ext4 `auto_da_alloc` on every page, parking 24 workers in `do_renameat2`. Two changes: dropped the temp-rename dance (write directly to the final path, delete on encode failure); defaulted per-page output to `/dev/shm/pdf-raster-<pid>-<nanos>/` for bare-stem prefixes. New CLI flags: `--ram`, `--no-ram`, `--ram-path <PATH>`. Heuristic: bare stem (`out`, `p`) → RAM; path-like (`./out`, `/tmp/p`) → disk literally. `SpillPolicy` queries `/proc/meminfo` MemAvailable every 100 ms; subsequent pages spill to disk automatically when free RAM drops below 1 GiB, with a one-shot stderr warning.
- **`PageIter` handles indirect `/Kids`.** The PDF spec allows `/Kids` to be either an inline array or an indirect Reference to one. `PageIter` only handled the inline case, silently reporting `page_count=0` for files using the reference form. Now resolves the reference one level. Regression test added in `pdf/src/document.rs`. Discovered while benchmarking corpus-04.

### v0.5.1 (May 2026)

**New since v0.5.0:**

- **Phase 7 — SOF-aware JPEG dispatch** — `gpu::jpeg_sof_type()` peeks the JPEG SOF marker byte (`0xC0` baseline / `0xC2` progressive / other); progressive JPEG is now routed directly to nvJPEG, bypassing VA-API (which supports baseline only); VA-API early-returns on SOF2 without a wasted parse attempt. `decode_dct_gpu` + `decode_dct_vaapi` collapsed into a single generic `decode_dct_gpu_path<D: GpuJpegDecoder>`. Hardening: `jpeg_sof.rs` — fixed None/Other contract, SOS guard, `0xFF` prefix check, TEM marker handling, 8 unit tests; `jpeg_parser.rs` — fixed 16-bit DQT/DHT truncation, range validation, SOS/EOI bounds.
- **Bug fixes** — `u32` overflow in `PageIter` fixed; `render_channel` streaming doc corrected (removed rayon::scope deadlock risk in example).
- **CI** — `actions/cache` v4 → v5, `actions/checkout` v4 → v6 (Node.js 24).

### v0.5.0 (May 2026)

**New since v0.4.0:**

- **`PageSet` sparse page selection** — `PageSet::new(pages)` creates a validated, sorted, deduplicated set of 1-based page numbers stored in an `Arc<[u32]>` (clone is O(1)). `RasterOptions::pages: Option<PageSet>` enables rendering a sparse subset of pages without visiting intermediates. `first_page`/`last_page` are ignored when `pages` is `Some`. Wired through `render_pages` and `render_channel`. 9 unit tests; sparse-page integration tests added (marked `#[ignore]` for CI).

### v0.4.0 (May 2026)

**New since v0.3.0:**

- **`--backend auto|cpu|cuda|vaapi` flag** — `BackendPolicy` enum (`Auto`, `CpuOnly`, `ForceCuda`, `ForceVaapi`) exposed on `SessionConfig`; `RasterError::BackendUnavailable` for forced-backend failures. CLI `--backend` and `--vaapi-device` flags wired through; `vaapi` feature exposed on the CLI crate.
- **Compositing correctness hardening** — 5 bugs in the general pipe, 4 safety assertions; AA gamma table values corrected with exhaustive test; `ncomps` parameter removed from `draw_image`/`blit_image` (derived from pixel type instead).
- **Bug fixes** — TJ kern ignores Tz correctly; FreeType init error propagated instead of panicking; `col_to_byte` uses saturating cast; PTX compilation now triggered correctly on `gpu-aa`/`gpu-icc` builds; PDF page cache evicted before each timed bench run.
- **Refactors** — `finish_pixel` helper extracted; `compute_a_src` helper extracted eliminating duplicated alpha logic; `page/mod.rs` split into focused sub-modules.
- **CLI shared-helper refactor** — `DEFAULT_VAAPI_DEVICE` const eliminates 3 independent string literals; `diagnostics.rs` module extracted from `main.rs` (4 error display functions); `build_page_list` moved into `Args::build_page_list(&self, total) -> Result<(Vec<i32>, Vec<String>), String>` (testable, side-effect-free); `routing_hint_from_diag` + `ProgressCtx::report` moved into `page_queue.rs` (eliminating cross-module call inversion); serial prescan loop removed (recovered 15-20% performance regression); `count_filter` + `update_max_ppi` helpers extracted in `prescan.rs` (eliminated duplicate PPI/filter-count blocks); `main.rs` reduced to ~100 lines of pure orchestration; 21 new unit tests.
- **Rayon pool hardening** — deadlock fix: `tx` now explicitly dropped inside `pool.scope` closure; single-thread pool deadlock guard (`capacity = n_pages` when `n_threads == 1`); ETA guard prevents `~0.0s remaining` on first page; `debug_assert!(n_pages >= 1)` makes invariant explicit; capacity tests now verify actual channel back-pressure behavior.

### v0.3.0 (May 2026)

Phases 5 and 6 are complete and integrated.  All core roadmap milestones done.

**New since v0.2.0:**

- **`pdf_raster` public library crate** — `raster_pdf`, `render_channel`, `open_session`, `RasterOptions`, `RenderedPage`, `PageDiagnostics`, `RasterError`.  Three review passes; full validation, GPU teardown, `render_channel` backpressure, atomic temp-file rename in CLI.
- **`UserUnit` support** — `page_size_pts` reads, validates, and propagates `UserUnit`; `RenderedPage.effective_dpi` is the correct value to pass to Tesseract.
- **`PageDiagnostics`** — `has_images`, `has_vector_text`, `dominant_filter`, `source_ppi_hint`, `suggested_dpi()` — zero-cost collection during render.
- **Pipelined render+OCR** — `render_channel(path, opts, capacity)` for bounded producer/consumer.
- **DPI auto-selection hint** — `suggested_dpi(min, max)` snaps to nearest standard DPI step.
- **GPU teardown** — explicit `release_gpu_decoders()` via `pool.broadcast()` before pool drop; eliminates CUDA atexit race.
- **Fuzz targets** — `crates/fuzz`: CCITTFaxDecode and JBIG2Decode coverage-guided fuzz targets.
- **Image module refactor** — 1 500-line `image/mod.rs` split into focused submodules.
- **Glyph cache** — `DashMap` + `lru` replaced with `quick_cache::sync::Cache` (sharded; reads no longer force write lock).
- **CLI hardening** — named rayon workers, 8 MiB stack, `MONO_THRESHOLD` const, atomic temp-file rename, `--odd`/`--even` mutual exclusion, DPI/JPEG quality validation.
- **Compositing correctness** — `apply_transfer_channel` removed; general pipe now calls `apply_transfer_in_place` with correct gray/CMYK LUT dispatch.  Overprint routing fixed: `no_transparency()` now excludes `overprint_mask != 0xFFFF_FFFF`.  Replace-overprint unimplemented path now panics loudly in release.
- **Performance** — `panic = "abort"` in release profile; `#[inline(always)]` on `apply_transfer_pixel` / `apply_transfer_in_place`; CMYK CLUT tables cached per page render; `Compression::Fast` for PNG output; `black_box` bench fencing.
- **Image decoding hardening** — 33 bugs fixed across image submodules and GPU/CLI paths over three hardening passes.
- **`#[expect]` throughout** — all `#[allow]` replaced with `#[expect(lint, reason = "...")]`.

### v0.2.0 (May 2026)

ARM/aarch64 platform: NEON acceleration for AA popcount, CMYK→RGB, glyph unpack, solid fill, and bilinear deskew.  SVE2 popcount tier behind `nightly-sve2` feature.  AVX2 AA popcount and CMYK→RGB tiers for Intel consumer CPUs.  VA-API JPEG decode (`vaapi` feature) for AMD/Intel iGPU on Linux.  CPU-only CI workflow.  Full 10-corpus benchmark results.

### v0.1.0 (Apr 2026)

Initial release.  Native PDF interpreter (Phases 1–4), GPU acceleration (nvJPEG, nvJPEG2000, GPU AA fill, tile fill, ICC CLUT), deskew, CLI (`pdftoppm` replacement).

---

## Phase 0 — Library API research ✓ COMPLETE (Apr 2026)

### Tesseract integration findings (researched Apr 2026)

**Tesseract 5.3.4 / Leptonica 1.82.0 on this machine.**

| Question | Answer |
|---|---|
| Raw pixel input without files? | Yes — `tesseract::ocr_from_frame(&[u8], w, h, bpp, stride, lang)` in the `tesseract` crate (v0.15.2). No file I/O on either side. |
| Best Rust crate? | `tesseract` 0.15.2 (April 2025, actively maintained). `leptess` is stale (last release Feb 2023). |
| Pre-binarise before passing? | **No.** LSTM engine reads grayscale directly for feature extraction; binarising first discards information it would have used. Feed 8-bit gray. |
| Background normalisation needed? | **No — drop it from our scope.** Tesseract does its own internal binarisation (Otsu / tiled Otsu / Sauvola, configurable). For uneven scanned backgrounds, caller sets `thresholding_method=2` (Sauvola) on the Tesseract side. |
| Does Tesseract deskew? | **No.** Tesseract can *detect* skew angle (PSM 0/1) but the caller must rotate the image. Deskew is the **one preprocessing step we still own**. |
| DPI handling? | Call `set_source_resolution(dpi)` explicitly after `set_frame`. Default fallback is 70 DPI which severely degrades accuracy. Pass the actual render DPI. |
| libopenjp2 on this machine? | Yes — Leptonica 1.82.0 links libopenjp2 2.5.0. JPEG 2000 works natively. |

### What exists in pdf-raster

- `render_page_native()` in `crates/cli/src/render.rs` — closest to a pipeline entry point, but CLI-entangled: takes `&Args`, writes to disk, returns `()`
- `rgb_to_gray()` in `crates/cli/src/render.rs` — BT.709 grayscale, unexported
- `pdf_interp::open()`, `page_count()`, `page_size_pts()`, `parse_page()` — clean public surface
- `raster::Bitmap<T>` — pixel buffer type, usable as a return type
- GPU decoder lifecycle (`DecoderInit<T>` thread-locals) — CLI-specific, needs encapsulation

### Remaining gaps for Phase 5

| Gap | Notes |
|---|---|
| Library crate with public API | No such crate; logic buried in CLI binary |
| In-memory grayscale output | `rgb_to_gray` unexported; nothing returns `Bitmap<Gray8>` |
| Deskew (±7°) | The one preprocessing step we own; algorithm decided — see Phase 5 |
| Per-page error handling | CLI fails fast; library should return `Result` per page |
| GPU decoder lifecycle for library callers | `DecoderInit` thread-locals are CLI-specific |

---

## Phase 1 — Native PDF interpreter ✓ COMPLETE

### Done

- [x] Content stream tokenizer + operator dispatcher (50+ operators)
- [x] Graphics state: `q Q cm w J j M d i ri gs`
- [x] Path construction: `m l c v y h re`
- [x] Path painting: `S s f F f* B B* b b* n`
- [x] Clip paths: `W W*` — intersected into live `Clip` with correct pending-flag semantics
- [x] Colour operators: `g G rg RG k K sc scn SC SCN cs CS`
- [x] Text objects + state: `BT ET Tf Tc Tw Tz TL Ts Tr Td TD Tm T*`
- [x] Text showing: `Tj TJ ' "` via FreeType
- [x] Font encoding `Differences` array → Adobe Glyph List → GID
- [x] `ExtGState` (`gs`): fill/stroke opacity, line width, cap, join, miter, flatness
- [x] Form XObjects: recursive execution, resource isolation, depth limit
- [x] Image XObjects: FlateDecode, DCTDecode (JPEG), JPXDecode (JPEG 2000), CCITTFaxDecode Group 3 (K=0, K>0) + Group 4, raw
- [x] Image colour spaces: DeviceRGB, DeviceGray, mask (stencil)
- [x] Soft mask (SMask) compositing on images
- [x] JavaScript rejection — hard fail on any JS entry point in the document
- [x] CLI `--native` flag wired to `pdf_interp` render path

### Blocking parity — must land before deleting pdf_bridge

Ordered by priority. Wire CLI by default is the finish line.

- [x] **ICCBased / Indexed / Separation colour spaces** — resolve_cs inspects ICC `N`, expands Indexed palettes, converts CMYK inline; Separation/DeviceN fall back to Gray
- [x] **ExtGState blend modes (`BM`)** — all 16 PDF modes parsed + threaded through make_pipe to raster compositor
- [x] **CCITTFaxDecode Group 3** — K=0 (1D T.4) via fax::decoder::decode_g3; K>0 (mixed 1D/2D "MR") via hayro-ccitt EncodingMode::Group3_2D
- [x] **Inline images (`BI ID EI`)** — decode_inline_image: abbreviated key/name expansion, FlateDecode/DCT/CCITT/RL/raw dispatch, wired to blit_image
- [x] **Shading (`sh`)** — Types 2 (axial) and 3 (radial) resolved; Function Types 2 (Exponential) and 3 (Stitching) evaluated; wired to shaded_fill
- [x] **Wire CLI by default** — `--native` flag removed; native is the only path; pdf_bridge dep removed from cli (crate retained for reference)

### Nice-to-have before default (won't block, but improve coverage)

- [x] **Text render modes 4–7** — text-as-clip via `glyph_path` outline collection; glyph paths unioned and intersected into clip per PDF §9.3.6
- [x] **Type 0 / CIDFont composite fonts** — CMap parsing, DescendantFonts, CIDToGIDMap, DW/W metrics, multi-byte charcode iteration
- [x] **Tiling patterns** — `scn`/`SCN` with Pattern colour space; `PatternType` 1 tiles rasterised via child `PageRenderer` and tiled with `rem_euclid`; PaintType 2 (uncoloured) falls through to solid fill

### Phase 1 parking lot (post-shipping coverage work)

- [x] Type 3 paint-procedure fonts
- [x] JBIG2Decode image filter
- [x] Optional content groups (layers / OCG)
- [x] Annotation rendering
- [x] Non-axis-aligned image transforms (currently bounding-box nearest-neighbour approximation)

### ~~Open: inline images never use GPU decoders~~ — RESOLVED

`decode_inline_image` now accepts the same `#[cfg]`-gated GPU decoder
parameters as `resolve_image`.  The `PageRenderer::InlineImage` arm passes
`self.nvjpeg.as_mut()` / `self.nvjpeg2k.as_mut()` / `self.gpu_ctx.as_deref()`
through.  The threshold-based dispatch inside `decode_dct` / `decode_jpx`
handles the actual gating — most inline images are small and take the CPU path,
but large inline JPEG/JPEG 2000 images (≥ 512×512) are now eligible for GPU
acceleration.

---

## Phase 2 — Raster performance ✓ COMPLETE

**Hardware context (Ryzen 9 9900X3D):** 128 MiB 3D V-Cache means edge tables and scanline sweep structures for most real-world documents fit in L3. The scanline sweep is therefore compute-bound, not memory-bound — algorithms that improve cache utilisation (sparse tiles) give less uplift here than on a normal CPU. AVX-512 extensions available: `avx512f/bw/vl/dq/cd/ifma/vbmi/vbmi2/vnni/bf16/bitalg/vpopcntdq/vp2intersect`. Target `-C target-cpu=native`.

- [x] **Eliminate per-span heap allocations** — `PipeSrc::Solid` and pattern scratch bufs use thread-local grow-never-shrink `PAT_BUF`; zero allocation per span
- [x] **u16×16 compositing inner loop** — `composite_aa_rgb8_opaque` processes 16 pixels/iter as `[u16; 16]`, `div255_u16 = (v+255)>>8`; LLVM auto-vectorizes to AVX2/AVX-512
- [x] **Fixed-point edge stepping (FDot16)** — `XPathSeg::dxdy_fp: i32` (16.16) added; scanner inner loop does `xx1_fp += dxdy_fp` (integer add) instead of `f64` accumulation
- [x] **Sparse nonempty-row iteration** — `XPathScanner::nonempty_rows()` uses the existing `row_start` sentinel array as a free sparsity index; fill loops skip empty rows with zero overhead

**Decision: CPU sparse tile rasterisation is deferred.** The original item (replace flat SoA with tile records sorted by (y,x)) was motivated by cache-miss reduction. On the 9900X3D the working set fits in L3, so the scanline sweep is already compute-bound and the win would be marginal. Tile records become high-value as the **GPU dispatch format** (Phase 4), where they map directly to warp-parallel execution. Implementing them twice — once for CPU, once for GPU — is redundant; Phase 4 will do it once, correctly, for the right target.

**AA quality note:** the current 4× scanline supersampling (`render_aa_line`) is an approximation. Analytical sub-pixel coverage (vello-style trapezoid integrals) is strictly better in quality and would be faster on the GPU. This is addressed in Phase 4.

---

## Phase 2.5 — CPU-side AVX-512 specialisation ✓ COMPLETE

Targeted use of AVX-512 extensions that LLVM does not auto-vectorize to. All paths use runtime detection (`is_x86_feature_detected!` / CPUID) with scalar fallbacks; binary runs on non-AVX-512 machines.

- [x] **`avx512bitalg` + `avx512bw` AA popcount** (`simd/popcnt.rs`) — `aa_coverage_span` uses `_mm512_popcnt_epi8` on nibble-masked AaBuf rows, processing 128 output pixels per 64-byte iteration. Falls back to `avx512vpopcntdq` + `avx512bw` (`popcnt_aa_row`), then scalar `NIBBLE_POP` table.

- [x] **`avx512vpopcntdq` + `avx512bw` row popcount** (`simd/popcnt.rs`) — `popcnt_aa_row` uses `_mm512_popcnt_epi8` on 64-byte chunks; falls back to hardware `popcnt` on 8-byte chunks, then scalar `u8::count_ones`.

- [x] **`movdir64b` non-temporal solid fill** (`simd/blend.rs`) — `blend_solid_rgb8` uses 192-byte tiles (LCM of 3 and 64) of inline-asm `movdir64b` stores for spans > 256 px; bypasses L3 for write-only solid fill data, preserving edge table residency in V-Cache. CPUID.07H.00H:ECX[28] detection via inline asm. Falls back to AVX2 32-px chunks, then scalar.

- [x] **`avx2` blend / glyph unpack** (`simd/blend.rs`, `simd/glyph_unpack.rs`) — `blend_solid_rgb8` and `blend_solid_gray8` use AVX2 for 32-px solid fill chunks; `unpack_mono_row` uses SSE4.1 `_mm_blendv_epi8` for 1-bpp → 8-bpp glyph expansion.

- [x] **`avx512bw` ICC CMYK→RGB matrix** (`gpu/src/lib.rs`, `cmyk_to_rgb_avx512`) — processes 16 pixels per call using `_mm256_mullo_epi16` u16 arithmetic. VNNI (`_mm512_dpbusds_epi32`) was ruled out: it requires one operand to be compile-time constant weights, but the subtractive formula `(255−C)*(255−K)/255` has both operands as runtime pixel data. AoS→SoA via `_mm512_shuffle_epi8` gather + `permute4x64` + `shuffle_epi8` compact; exact `⌊(x+127)/255⌋` divide via `(n+(n>>8)+1)>>8`. Scalar fallback for tail and non-AVX-512 targets.

- [x] **`cat_l3` / `cdp_l3` cache partitioning** — deployment note documented in `ROADMAP_INTEL.md` (Deployment notes section): `resctrl` on Xeon/EPYC; not available on Intel consumer CPUs; no code change required.

---

## Phase 3 — Coverage completeness ✓ COMPLETE

Track and close fidelity gaps against pdftoppm once the native path is default.

- [x] Coons patch / tensor mesh shading (Type 4–7)
- [x] Non-axis-aligned image transforms — exact inverse-CTM nearest-neighbour sampling for arbitrary rotated/sheared images; row-constant hoisting eliminates redundant multiplies per inner loop
- [~] Halftone screens for CMYK separation output — out of scope for a screen rasterizer; PDF viewers intentionally ignore `HT` and render continuous tone; only relevant to print RIPs
- [x] PDF transparency groups (isolated / non-isolated / knockout) at the page level

### Phase 3 follow-on (post-Phase-4 coverage work, Apr 2026)

- [x] **bpc 2, 4, 16 image decoding** — `expand_nbpp<const BITS>` (MSB-first, scaled to 0–255), `expand_nbpp_indexed` (raw palette indices, bpc 1/2/4), `downsample_16bpp` (high-byte truncation); shared `unpack_packed_bits` helper eliminates loop duplication; all three applied in `decode_raw`, SMask decoder, and `decode_raw_indexed`
- [x] **CCITTFaxDecode K>0 (Group 3 mixed 2D / T.4 MR)** — `decode_ccitt_g3_2d` via hayro-ccitt 0.3.0 `EncodingMode::Group3_2D { k }`; `HayroCcittCollector` implements the `Decoder` trait; per-row and final-row white padding for truncated/malformed streams
- [x] **`--gray` / `--mono` CLI flags** — post-render RGB→Gray8 conversion (BT.709 integer coefficients) and 50%-midpoint threshold; `--gray` writes PGM/gray PNG, `--mono` writes PBM (P4)/gray PNG; new `encode::write_pbm` (P4 encoder)

### Still open / lower priority

- [x] Function-based shading (Type 1) — pre-sampled 64×64 grid; bilinear interpolation in fill_span; BBox intersection; full CTM inversion
- [x] nvJPEG2000 for JPXDecode — GPU fast path via `nvjpeg2k` feature; planar→interleaved copy (`cudaMemcpy2D` D→H per component); sub-sampling guard + OOM cap + zero-dimension guard; CPU `jpeg2k`/OpenJPEG fallback; threshold-gated at 512×512 px (see Phase 4 item 1 for full audit)
- [ ] OptiX BVH (evaluate only if profiling shows complex paths as bottleneck)

---

## Phase 4 — GPU acceleration (cudarc)

Unblocked by Phase 1 completion (poppler must be gone first). **Phase 1 is complete — Phase 4 is now unblocked.**

**Hardware context (RTX 5070, CC 12.0 Blackwell, 12 GB GDDR7):** cudarc 0.19 is already wired in `crates/gpu` with two kernels (Porter-Duff composite, soft mask) and CPU fallbacks. Target `sm_120` PTX. The GPU dispatch threshold is currently 500k pixels — validate this against actual transfer latency on this machine once the native path is hot. Do **not** use wgpu/Vello's GPU backend — CUDA is strictly better for a batch server pipeline on NVIDIA hardware.

**Do not use DLSS, MSAA, CSAA, or TAA.** These are real-time game rendering features (temporal, triangle-mesh, depth-buffer dependent) and have no applicability to batch PDF rasterisation.

### Priority order

**1. nvJPEG image decoding — highest value, implement first** ✓ COMPLETE

For scan-heavy corpora (JPEG/JBIG2/CCITT image layers + thin OCR text overlay), image decoding dominates wall-clock time. nvJPEG decodes at ~10 GB/s on the RTX 5070; the CPU JPEG path (libjpeg via DCTDecode) is 10–20× slower. No rasterizer changes required — wire nvJPEG into the existing `blit_image` path behind a feature flag.

- [x] `gpu::nvjpeg` module: minimal raw FFI surface (no bindgen); `NvJpeg` (pub(crate)) + `NvJpegDecoder` (pub) safe wrapper; `decode_sync` blocks on `cuStreamSynchronize` after GPU DMA completes
- [x] DCTDecode dispatch: image area ≥ `GPU_JPEG_THRESHOLD_PX` (512×512) → nvJPEG; else CPU zune-jpeg; CMYK JPEG falls through to CPU
- [x] Feature flags: `gpu/nvjpeg` + `pdf_interp/nvjpeg`; zero-cost when disabled; pdf_interp maintains `unsafe_code = "deny"`
- [x] `NVJPEG_BACKEND_HARDWARE` (on-die engine, RTX 5070/Turing+) with automatic fallback to `NVJPEG_BACKEND_DEFAULT` on `NVJPEG_STATUS_JPEG_NOT_SUPPORTED` (progressive JPEGs); fallback is one-shot per decoder instance
- [x] Output buffer is `PinnedBuf` via `cuMemAllocHost_v2` — declare the `_v2` symbol explicitly via `#[link_name]`; calling the old `cuMemAllocHost` symbol returns `CUDA_ERROR_INVALID_CONTEXT=201`; plain `Vec<u8>` segfaults on DMA
- [x] Pure raw CUDA driver API in `NvJpegDecoder` (no cudarc at runtime): `cuInit → cuDeviceGet → cuDevicePrimaryCtxRetain → cuCtxSetCurrent → cuStreamCreate → nvjpegCreateEx`; mixing cudarc's primary context with nvJPEG's internal context causes `CUDA_ERROR_INVALID_CONTEXT=201` on every `cuStreamSynchronize`
- [x] `NvJpegDecoder::dec` is `ManuallyDrop<NvJpeg>` so Drop explicitly calls nvjpegDestroy *before* `cuDevicePrimaryCtxRelease`; Rust's field-drop order would otherwise release the context while nvJPEG handles are still live
- [x] `cuStreamSynchronize` called on error path from `nvjpegDecode` before dropping `PinnedBuf` — GPU may have enqueued partial work that would write into freed memory
- [x] Minimum JPEG size: nvJPEG GPU kernels require ≥ one full 8×8 MCU block; 1×1 JPEGs crash inside the driver (test fixture is 16×16)
- [x] API correctness audit (Apr 2026, CUDA 12.8 headers): `nvjpegCreate` deprecated → replaced with `nvjpegCreateEx(backend, dev_alloc, pinned_alloc, flags, handle)`; CUDA error code 209 corrected (NO_BINARY_FOR_GPU not MAP_FAILED=205); `is_x86_feature_detected!("movdir64b")` does not exist on stable — detection uses `__cpuid_count(7,0).ecx >> 28`; glyph unpack gate was SSE4.1 but all intrinsics are SSE2; `_mm512_popcnt_epi8` stable since Rust 1.89; `cuDevicePrimaryCtxRetain` is the NVIDIA-recommended pattern (not `cuCtxCreate`); `nvjpegDecode` not deprecated (batched pipeline API is optional); `cuStreamCreate(flags=0)` = CU_STREAM_DEFAULT still correct
- [x] **nvJPEG2000 for JPXDecode (JPEG 2000)** ✓ COMPLETE
  - `gpu::nvjpeg2k` module: `DeviceBuf` RAII (`cudaMalloc`/`cudaFree`); `NvJpeg2k` (pub(crate)) inner decoder; `NvJpeg2kDecoder` (pub) safe wrapper with `ManuallyDrop<NvJpeg2k>` for explicit drop order
  - Output memory is **device** (`cudaMalloc` inside library), not host-pinned; `cudaMemcpy2D` per component after stream sync to copy D→H
  - Image layout is **planar** (separate device ptr per component); Gray (1 comp) passthrough; RGB (3 comps) interleaved via `chunks_exact_mut(3).zip(r.iter().zip(g.iter().zip(b.iter())))`
  - Parse step: `nvjpeg2kStreamParse` before `nvjpeg2kDecode`; bitstream handle (`nvjpeg2kStream_t`) distinct from CUDA stream; reused across decodes
  - **Sub-sampling guard (CRITICAL)**: bare `nvjpeg2kDecode` writes components at their native (reduced) dimensions — it does NOT upsample sub-sampled chroma (unlike `nvjpeg2kDecodeParamsSetRGBOutput`); images where any `component_width/height` differs from `image_width/height` are rejected → CPU OpenJPEG fallback
  - **OOM guard (CRITICAL)**: corrupt header returning `u32::MAX` for `num_components` would cause `Vec::with_capacity(usize::MAX)` (~68 GB); capped at `nc > 4` → `UnsupportedComponents` error before any allocation
  - **Zero-dimension guard**: explicit `ZeroDimension { width, height }` error if any component dimension is 0
  - **Pitch ownership**: caller sets `pitch_in_bytes` in `Nvjpeg2kImage`; library writes at that exact pitch — no mismatch possible since we define it; documented explicitly
  - Drop order: `nvjpeg2kDecodeStateDestroy` → `nvjpeg2kStreamDestroy` → `nvjpeg2kDestroy` (API contract; reverse creation order); enforced via `ManuallyDrop` in `NvJpeg2kDecoder`
  - Pure raw CUDA driver API (same rationale as nvJPEG): `cuInit → cuDeviceGet → cuDevicePrimaryCtxRetain → cuCtxSetCurrent → cuStreamCreate → nvjpeg2kCreateSimple`; no cudarc at runtime
  - `cuStreamSynchronize` called on error path before returning — GPU may have enqueued partial work
  - Library path: `/usr/lib/x86_64-linux-gnu/libnvjpeg2k/12/` (non-standard; explicit `rustc-link-search` in `build.rs`); `cudart` linked for `cudaMalloc`/`cudaFree`/`cudaMemcpy2D`
  - Dispatch threshold: `GPU_JPEG2K_THRESHOLD_PX = 262_144` (512×512 px); CPU `jpeg2k`/OpenJPEG fallback for small images and unsupported sub-sampled streams
  - `NvJpeg2kError`: `Nvjpeg2kStatus`, `CudaError`, `CudartError`, `UnsupportedComponents`, `ZeroDimension`, `Overflow`
- [x] **CLI wiring for nvJPEG + nvJPEG2K** — `thread_local! DecoderInit<T>` state machine per rayon worker thread in `crates/cli/src/render.rs`; lazy construction on first page; decoder moved into renderer before `execute()` and returned to the slot after `render_annotations()`; `DecoderInit::Failed` prevents retry-and-spam after a one-time init failure; `PageRenderer::take_nvjpeg` / `take_nvjpeg2k` recover the decoder after each page so the CUDA context and stream survive across pages with zero re-init cost

**2. GPU supersampled AA — replaces CPU 4× scanline AA** ✓ COMPLETE

The current `render_aa_line` + nibble-popcount AA is the weakest part of the CPU pipeline. Replace it with a CUDA kernel doing **jittered supersampling** at 64 samples/pixel using warp-level ballot reduction:

```cuda
// One warp (32 threads) per output pixel
bool inside = winding_test(segs, n_segs, jittered_sample(px, py, threadIdx.x));
int coverage = __popc(__ballot_sync(0xFFFFFFFF, inside));
output[py * width + px] = (uint8_t)((coverage * 255) / 32);
```

`__ballot_sync` + `__popc` gives 32-sample coverage in a single warp cycle. With 2 warps/pixel: 64 samples. Quality far exceeds the CPU 4×4 grid; cost is lower because the 4352 CUDA cores run all pixels in parallel. The CPU AA path remains as fallback below the dispatch threshold.

- [x] CUDA kernel: jittered 64-sample winding test per pixel (`kernels/aa_fill.cu`; Halton(2,3) sample table; winding-number + EO rule; scales 0..64 → 0..255 via `(total*255+32)>>6`)
- [x] Warp-ballot reduction: `__ballot_sync` + `__popc` per warp (2 warps/pixel = 64 samples); warp counts aggregated via shared memory; thread 0 writes final byte
- [x] Wire into fill dispatch: `PageRenderer::try_gpu_aa_fill` (gated on `pdf_interp/gpu-aa` feature); CPU fallback below `GPU_AA_FILL_THRESHOLD`; pattern fills always CPU
- [x] Validate quality vs CPU AA on pixel-diff benchmark — pixel-identical (RMSE=0) across 41 pages / 98 GPU-dispatched fills at 600 DPI; CLI `gpu-aa` feature wires `GpuCtx` into renderer
- [x] **Dispatch threshold calibration** (`src/bin/threshold_bench.rs`): geometric sweep 256–4M px on RTX 5070 + `PCIe` 5.0; `GPU_AA_FILL_THRESHOLD` 16 384 → **256 px** (GPU wins immediately; 2.5× at 256 px, 100× at 16 384 px)

**3. Tile-parallel fill rasterisation — GPU path only** ✓ COMPLETE (kernel + Rust API; PageRenderer integration pending)

Tile records (sorted by (tile_y, tile_x)) are the natural GPU work unit. One 16×16 thread block per tile, independent analytical coverage accumulation per pixel, no inter-tile communication required.

CUB radix sort was evaluated and rejected for this use case: typical PDF pages have O(100–1000) segments, generating O(1000–10000) tile records. CPU `sort_unstable_by_key` is faster end-to-end than the CUB two-pass launch + temp-buffer allocation at these sizes. The sort stays on the CPU; the heavy per-pixel integration runs on the GPU.

- [x] Tile record format: `TileRecord` (32 bytes, `repr(C)`): `{key: u32, x_enter: f32, dxdy: f32, y0_tile: f32, y1_tile: f32, sign: f32, _pad: u32, _pad2: u32}`; 32-byte alignment matches CUDA global memory transaction size
- [x] CPU record builder: `build_tile_records(segs, x_min, y_min, width, height)` — one record per (segment, tile-row) crossing; sorted CPU-side by `key = (tile_y << 16) | tile_x`; prefix-sum `tile_starts`/`tile_counts` index built inline; `bytemuck::Pod` + `cudarc::DeviceRepr` for zero-copy upload
- [~] CUB radix sort: replaced with CPU `sort_unstable_by_key` (see rationale above; CUB left as a future micro-optimisation if segment counts exceed ~50k)
- [x] Fill kernel (`kernels/tile_fill.cu`): grid `(grid_w, grid_h, 1)`, block `(TILE_W=16, TILE_H=16, 1)`; each thread accumulates signed trapezoidal area for its pixel column across all segments crossing its tile row; NZ rule: `min(|area|, 1) × 255.5`; EO rule: folded-fraction formula
- [x] `GpuCtx::tile_fill()` Rust API: uploads records/starts/counts via `stream.clone_htod`, launches kernel, synchronises, copies coverage bytes back; threshold `GPU_TILE_FILL_THRESHOLD`
- [x] **Dispatch threshold calibration**: `GPU_TILE_FILL_THRESHOLD` 65 536 → **256 px** (same crossover as AA fill; tile records + CPU sort overhead is still faster than pure CPU AA at all sizes above 256 px)
- [x] Wire into `PageRenderer` fill dispatch: `try_gpu_tile_fill` (area ≥ `GPU_TILE_FILL_THRESHOLD`) tried first, then `try_gpu_aa_fill` (area ≥ `GPU_AA_FILL_THRESHOLD`), then CPU scanline AA; shared `gpu_fill_segs` + `gpu_coverage_to_bitmap` helpers eliminate duplication

**4. ICC colour transforms** ✓ COMPLETE (CPU AVX-512 + GPU CLUT kernel)

DeviceCMYK → DeviceRGB via two paths depending on whether a full ICC CLUT is available:

- [x] **CPU matrix path** (`icc_cmyk_to_rgb_cpu`, clut=None): subtractive formula `(255−ch)*(255−K)/255` vectorised with `avx512bw` + `avx2` — 16 pixels/call via `_mm256_mullo_epi16`. VNNI was evaluated and rejected: `_mm512_dpbusds_epi32` requires compile-time constant weights; both operands are runtime pixel data here. Exact `⌊(x+127)/255⌋` divide matches scalar to the bit. Scalar fallback for non-AVX-512 targets and tail pixels.
- [x] **GPU CLUT kernel** (`kernels/icc_clut.cu`): 4D quadrilinear interpolation over a baked `grid_n⁴ × 3` byte table; one thread per pixel; threshold `GPU_ICC_CLUT_THRESHOLD = 500 000 px` (conservative placeholder; CLUT path not yet in the hot path)
- [x] **ICC matrix dispatch fix**: `icc_cmyk_to_rgb` short-circuits to `icc_cmyk_to_rgb_cpu` before the threshold check when `clut=None` — `threshold_bench` showed GPU matrix kernel never beats AVX-512 across all measured sizes (256–4M px); `PCIe` round-trip cost exceeds the cheap per-pixel computation
- [x] `bake_cmyk_clut` (`pdf_interp/src/resources/icc.rs`): bakes a Little CMS ICC profile into a compact `u8` CLUT for upload; `BakeError` with `InvalidGridSize` and `Cms` variants; `DEFAULT_GRID_N = 17`
- [x] Rounding bias fix in CUDA kernel: `((255u - c) * inv_k + 127u) / 255u` (was missing the `+127` bias)
- [x] Parity tests: `icc_cmyk_matrix_avx_vs_scalar` asserts AVX-512 and scalar agree byte-for-byte across 16 representative pixels including axis extremes and mid-range sweep
- [x] nvJPEG2000 for JPXDecode — implemented (see Phase 4 item 1 above)

**5. OptiX BVH for complex paths — low priority, evaluate later**

RT cores on Blackwell provide hardware BVH traversal. For pages with thousands of path segments, an OptiX any-hit kernel computing winding numbers via ray casting would be faster than the tile rasteriser for very complex geometry. In practice, most PDF pages have O(100) path segments, not O(10000), so this is unlikely to be the bottleneck. Evaluate only after profiling shows complex path rasterisation in the flamegraph.

### GPU dispatch table

| Target | Value | Unblocked by |
|---|---|---|
| nvJPEG image decoding | **Highest** — scan-heavy corpora | Phase 1 image pipeline ✓ |
| GPU supersampled AA (warp ballot) | High — quality + speed | GPU segment upload |
| Tile-parallel fill rasterisation | High — sparse/complex paths | GPU segment upload |
| ICC colour transforms | Medium — CMYK docs | Phase 1 colour spaces ✓ | ✓ COMPLETE |
| OptiX BVH winding test | Low — only extreme geometry | Tile rasteriser |
| Blend / composite | Low — already fast on CPU | Phase 2 perf work ✓ |

FreeType text rendering is **not** a GPU target — hinting is sequential per glyph. A GPU text path requires a GPU-resident rasteriser (SDF atlas or Slug algorithm) and is a separate major project.

---

## Benchmarking

**Status: baseline benchmarks complete (Apr 2026).** All GPU features live. Machine: Ryzen 9 9900X3D + RTX 5070, 150 DPI, `--features nvjpeg,nvjpeg2k,gpu-aa,gpu-icc`, `RUSTFLAGS="-C target-cpu=native"`, `--warmup 3 --runs 8`.

### Results vs pdftoppm (poppler 24.x)

| Fixture | Size | Character | pdf-raster | pdftoppm | Speedup |
|---|---|---|---|---|---|
| light-vector.pdf | 116 KB | Light vector + text, 41 pp | 213 ms | 262 ms | **1.2×** |
| mixed-vector.pdf | 281 KB | Mixed vector + images, 7 pp | 109 ms | 291 ms | **2.7×** |
| dense-vector.pdf | 2.1 MB | Dense vector / complex paths, 34 pp | 495 ms | 1507 ms | **3.0×** |
| mixed-images.pdf | 11 MB | Mixed; image-heavy | 5.2 s | 44.4 s | **8.5×** |
| scan-heavy.pdf | 50 MB | Scan-heavy JPEG/JPEG2K | 17.2 s | 155.7 s | **9.1×** |

The scan-heavy corpus (JPEG/JPEG2K) shows the largest gap because nvJPEG + nvJPEG2K GPU decode replaces the CPU libjpeg/OpenJPEG path. The light-vector fixture shows the smallest gap — that workload is entirely CPU path-fill and text.

### Pixel diff vs poppler

`compare -metric AE` on 3 pages of a light-vector PDF at 150 DPI. Same page dimensions (700×1050 px). AE of 0.9–17% — entirely explained by sub-pixel anti-aliasing differences at glyph edges (amplified diff shows ghosted text, no structural content difference). This is expected for two independent renderers with different AA strategies.

### ~~Known gap: page rotation (`/Rotate`)~~ — RESOLVED (commit `82efbe5`)

`/Rotate` and `CropBox` are fully handled: `pdf_interp::page_size_pts` reads
`CropBox` (falling back to `MediaBox`), normalises `/Rotate` to 0/90/180/270,
and swaps dimensions for 90°/270° rotations.  `PageRenderer::new_scaled`
applies the matching CTM so all four rotation values produce correctly-oriented
output.  A landscape PDF with `/Rotate: 270` portrait MediaBox now renders as
landscape, matching poppler.

### ~~Known gap: `UserUnit` scaling~~ — RESOLVED (commits 4aa17b5 / ce10242)

`page_size_pts` now reads `UserUnit`, validates to `[0.1, 10.0]`, multiplies
`w_pts`/`h_pts` by it, and returns `user_unit` on `PageGeometry`.
`RenderedPage.effective_dpi` = `opts.dpi × UserUnit` is the correct value for
`tesseract::set_source_resolution`.  Non-numeric and NaN/Inf values are
rejected with `RasterError::InvalidPageGeometry`.

### Fixture inventory

Fixture PDFs are gitignored. Provide your own corpus covering these character classes:

| Character | Size range | Notes |
|---|---|---|
| Light vector + text | ~100 KB | Minimal render path; baseline for overhead measurement |
| Mixed vector + images | ~300 KB | Exercises JPEG decode + path fill together |
| Dense vector / complex paths | ~2 MB | Exercises scanline AA at scale |
| Mixed; image-heavy | ~10 MB | GPU ICC CLUT path |
| Scan-heavy JPEG/JPEG2K | ~50 MB | Primary nvJPEG + nvJPEG2K workload |

### Commands

```bash
# Build with all GPU features
RUSTFLAGS="-C target-cpu=native" cargo build --release \
  --manifest-path crates/cli/Cargo.toml \
  --features nvjpeg,nvjpeg2k,gpu-aa,gpu-icc

BIN=target/release/pdf-raster
LD_LIB=LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/libnvjpeg2k/12:/usr/local/cuda-12.8/lib64

# Throughput vs pdftoppm
env $LD_LIB hyperfine --warmup 3 --runs 8 \
  "$BIN -r 150 tests/fixtures/scan-heavy.pdf /tmp/out" \
  'pdftoppm -r 150 tests/fixtures/scan-heavy.pdf /tmp/ref'

# Pixel diff vs poppler reference (ImageMagick AE metric)
pdftoppm -r 150 tests/fixtures/light-vector.pdf /tmp/ref
env $LD_LIB $BIN -r 150 tests/fixtures/light-vector.pdf /tmp/out
for i in /tmp/ref-*.ppm; do
  n=$(basename $i .ppm | sed 's/ref-//')
  ae=$(compare -metric AE $i /tmp/out-${n}.ppm /dev/null 2>&1)
  echo "$(basename $i): AE=$ae"
done

# Flamegraph — find the new bottleneck after GPU image decode is wired
CARGO_PROFILE_RELEASE_DEBUG=true env $LD_LIB \
flamegraph -o /tmp/flame.svg -- \
  $BIN -r 150 tests/fixtures/scan-heavy.pdf /tmp/out

# Synthetic fill microbenchmark (raster crate path-fill vs vello_cpu)
RUSTFLAGS="-C target-cpu=native" cargo run -p bench --release -- --iters 30 --stars 200

# Threshold bench — recalibrate GPU dispatch crossovers after any kernel change
cargo run -p gpu --release --bin threshold_bench

# L3 occupancy monitoring (9900X3D — requires resctrl mount)
# mount -t resctrl resctrl /sys/fs/resctrl
# cat /sys/fs/resctrl/mon_data/mon_L3_XX/llc_occupancy
```

---

## Phase 5 — Public library API ✓ COMPLETE (Apr 2026)

Extract the render pipeline into a reusable library crate. The caller gets 8-bit grayscale pixels in memory and passes them directly to Tesseract — no subprocess, no files, no Leptonica.

### Crate: `crates/pdf_raster`

```rust
pub struct RasterOptions {
    pub dpi: f32,          // render DPI; pass same value to Tesseract set_source_resolution
    pub first_page: u32,   // 1-based
    pub last_page: u32,    // 1-based, inclusive
    pub deskew: bool,      // run deskew before returning pixels (scanned PDFs only)
}

pub struct RenderedPage {
    pub page_num: u32,
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,   // 8-bit grayscale, tightly packed, top-to-bottom
    pub dpi: f32,          // pass to Tesseract set_source_resolution — do not lie
}

/// Render pages from a PDF file, one result per page.
/// A per-page error does not abort remaining pages.
pub fn raster_pdf(
    path: &Path,
    opts: &RasterOptions,
) -> impl Iterator<Item = (u32, Result<RenderedPage, RasterError>)>;
```

**Caller's OCR step after integration:**

```rust
for (page_num, result) in pdf_raster::raster_pdf(path, &opts) {
    let page = result?;
    let text = tesseract::ocr_from_frame(
        &page.pixels, page.width as i32, page.height as i32,
        1, page.width as i32, "eng",   // bpp=1 (grayscale), stride=width
    )?;
    // for uneven-background scans, set thresholding_method=2 (Sauvola) on Tesseract side
}
```

### Preprocessing scope

| Step | Owner | Notes |
|---|---|---|
| Rasterise to grayscale | **pdf-raster** | BT.709 RGB→Gray; already in CLI, just needs exporting |
| Deskew | **pdf-raster** | See deskew design below |
| Background normalisation | **Tesseract** | Sauvola `thresholding_method=2` on the caller side |
| Binarisation | **Tesseract** | LSTM reads grayscale directly; do NOT pre-binarise |
| DPI metadata | **caller** | Pass `page.dpi` to `set_source_resolution`; default is 70 DPI (useless) |

### Deskew design (researched Apr 2026)

**Goal**: beat Leptonica's `pixDeskew` in both speed and accuracy.

**How Leptonica works (and where it fails):**
Hierarchical differential-projection-profile sweep: binarise at threshold 160 → 4x downsample → 14-angle coarse sweep (±7°, 1° steps) → quadratic interpolation → binary search to 0.01° convergence. Accuracy: ~0.03–0.05°. Failure modes: fixed threshold 160 fails on light/dark scans; skips angles < 0.1°; single-threaded; CPU-only rotation.

**Our approach — two-phase hybrid:**

**Phase A — Angle detection (CPU, intensity-weighted projection profile)**

Same algorithm family as Leptonica but without the binarisation threshold:
- Use `255 - pixel` as the foreground weight on raw 8-bit gray — dark pixels count as foreground proportionally, no hard threshold, no parameter to tune
- 4× downsample for coarse sweep (620×825 working set)
- 28-angle coarse sweep at 0.5° steps (±7°), scored by differential square sum of weighted row sums
- Binary search refinement to 0.01° convergence
- AVX-512 row summation via `VPSADBW` (64 pixels/cycle): each row of 2550px takes ~40 AVX-512 ops
- Parallelise sweep angles across Rayon threads (each angle is independent)
- **9900X3D V-Cache advantage**: 8.4MB image fits entirely in 96MB L3; stays warm through all sweep iterations — no DRAM traffic after first load
- Target: **1–3ms** for detection

Accuracy advantage over Leptonica: no binarisation threshold → correct on images where threshold 160 over- or under-segments; corrects angles < 0.1° that Leptonica skips.

**Phase B — Rotation (GPU, CUDA texture bilinear)**

- Bind source image as `cudaTextureObject_t` with `cudaFilterModeLinear` — hardware bilinear at no extra compute cost
- Use `nppiRotate_8u_C1R_Ctx` (NPP single-channel 8-bit rotate) or a custom kernel with `tex2D<float>()` per output pixel
- RTX 5070 texture fill rate: 482 GTexel/s → **0.3–0.5ms** for 8.4MP
- If image is already on GPU from nvJPEG/nvJPEG2K decode, PCIe upload cost is zero
- CPU fallback (12-core AVX-512 bilinear): ~1.5ms — used when GPU unavailable or image is CPU-only

**Steady-state pipeline (scan-heavy PDFs with GPU decode active):**
```
CPU: detect angle for page N+1  (~1ms, overlapped)
GPU stream A: rotate page N     (~0.4ms)
GPU stream B: D→H transfer N-1  (~0.3ms)
```
Net deskew cost per page at steady state: **~0.4ms** (rotation-bound; detection hidden).

**Single-page cold path (CPU RAM, no GPU decode):**
- Detection: ~2ms
- PCIe H2D (8.4MB @ 28GB/s): ~0.3ms
- GPU rotation: ~0.4ms
- PCIe D2H: ~0.3ms
- Total: **~3ms** — still faster than Leptonica's ~10–15ms

### Work items

- [x] New `crates/pdf_raster` library crate; add to `Cargo.toml` workspace members
- [x] Move `render_page_native` core (minus `&Args`, minus file I/O) into library
- [x] Export `rgb_to_gray` (BT.709) from library (currently private in CLI)
- [x] Encapsulate GPU decoder lifecycle (`DecoderInit<T>`) inside library — not caller-visible
- [x] `crates/pdf_raster/src/deskew/detect.rs` — intensity-weighted projection profile, AVX-512 row sums, Rayon sweep parallelism
- [x] `crates/pdf_raster/src/deskew/rotate.rs` — CPU bilinear fallback; GPU path via `nppiRotate_8u_C1R_Ctx`
- [x] Review pass: sentinel hack → `Option<Result>`, pages map O(n²) → O(n), `InvalidOptions` validation, `debug_assert` → `assert`, `NVJPEG2K_STATUS_IMPLEMENTATION_NOT_SUPPORTED` constant, `remove(0)` → `swap_remove(0)`, bilinear inlined into rotate loop, `downsample` factor=0 guard
- [x] Make CLI a thin wrapper over `crates/pdf_raster` (RasterSession, render_page_rgb, open_session)
- [x] Second review pass (Apr 2026): scale validation guard in `render_page_rgb`; GPU init failure `eprintln!`; `PageIter::next` Err arm cleaned; dead variable removed from `bitmap_to_vec`; `# Panics` doc corrected in lib.rs; `MONO_THRESHOLD` const extracted in CLI; atomic temp-file rename in CLI `render_page` (no partial files on encode failure)
- [x] Third review pass (Apr 2026): `open_session` double get_pages eliminated; bad `scale` returns `InvalidOptions` (not `PageDegenerate`); `PageIter::next` Err arm rewritten with explicit match; compile-time `Sync` assertion on `RasterSession`; `#[expect]` replaces `#[allow]` on Args; erroneous `cast_sign_loss` suppression removed from f64→f32 and f32→i32 casts; SWEEP_STEPS≥2 compile-time assert; `n_rows-skip` saturation guard; intermediate Vec allocations in coarse sweep eliminated (par reduce); `assert!→debug_assert!` in private `downsample`; rotation docs corrected (CW-positive throughout); GPU deskew stub noted in lib.rs; scatter loop AVX-512 auto-vec claim removed from doc; rename failure now also removes temp file; `--odd`+`--even` mutual exclusion check; open_session error walks source chain; DPI args validated ≥1 at CLI; jpeg_quality validated 0–100; `OutputFormat` implements Display; 13 redundant `default_value_t=false` removed
- [x] GPU rotation: `rotate_gpu` via `nppiRotate_8u_C1R_Ctx` — NPP CW-positive (Y-down); GPU/CPU parity ≤2 grey levels at 2°; thread-local `NppRotator`, CPU fallback retained; hardening pass (input validation, three-state slot, Drop logging, null asserts)
- [x] Integration tests: round-trip a fixture PDF, assert pixel dimensions and grayscale range; deskew unit tests with synthetic skewed images at known angles

---

## Phase 6 — Integration hardening and OCR pipeline fit ✓ COMPLETE (Apr 2026)

### Goal

Make pdf-raster the drop-in replacement for the pdftoppm + Leptonica preprocessing
stack in the mss OCR pipeline.  The rasterise + deskew path is feature-complete;
Phase 6 closes the remaining gaps before the first production integration.

### Open work items

- [x] **`UserUnit` support** — `page_size_pts` now reads `UserUnit`, validates
  it to `[0.1, 10.0]` (returning `RasterError::InvalidPageGeometry` for
  out-of-range values), multiplies `w_pts`/`h_pts` by `user_unit`, and exposes
  `PageGeometry.user_unit`.  `RenderedPage.effective_dpi` = `opts.dpi × UserUnit`
  is the correct value to pass to `tesseract::set_source_resolution`.
  Non-numeric or NaN/Inf `UserUnit` values are also rejected with a descriptive
  error.  The double `get_pages()` call in `page_size_pts` and `parse_page` was
  eliminated via a shared `resolve_page_id` helper (commits 4aa17b5 / ce10242 /
  cf3b3a7).

- [x] **`RenderDiagnostics` on `RenderedPage`** — `RenderedPage.diagnostics`
  (`PageDiagnostics`) exposes: `has_images`, `has_vector_text`,
  `dominant_filter` (most-used `ImageFilter` variant: `Dct / Jpx / CcittFax /
  Jbig2 / Flate / Raw`), and `source_ppi_hint` (estimated source PPI of the
  dominant image).  Collected at zero extra cost during rendering: `blit_image`
  increments per-filter counts; `show_text` sets `has_vector_text`; `finish()`
  resolves `dominant_filter` from counts.  `ImageFilter` and `PageDiagnostics`
  are re-exported from `pdf_raster` (commit 199d13a).

- [x] **Pipelined render + OCR** — `pdf_raster::render_channel(path, opts, capacity)`
  returns a `std::sync::mpsc::Receiver<(u32, Result<RenderedPage>)>`.  A
  Rayon-spawned producer renders pages in ascending order and sends them as they
  complete; the consumer (Tesseract) processes each page immediately.  The channel
  is bounded to `capacity` slots (min 1): producer blocks when consumer falls
  behind, capping peak memory at `capacity × page_size`.  Options validation runs
  synchronously before spawn; session-open and per-page errors are delivered
  through the channel (same non-fatal contract as the iterator).  Zero new
  dependencies — `rayon` was already present; `std::sync::mpsc` is stdlib.
  `validate_opts` extracted from `render_pages` so both paths share identical
  validation.  5 unit tests cover all error paths and backpressure.

- [x] **DPI auto-selection hint** — `PageDiagnostics::suggested_dpi(min, max)`
  rounds `source_ppi_hint` up to the nearest standard DPI step
  (72 / 96 / 150 / 200 / 300 / 400 / 600) and clamps to `[min, max]`.
  `RenderedPage::suggested_dpi` delegates to it.  Returns `None` for
  vector/text-only pages so callers fall back to their default DPI.
  No stored field — pure computed from existing `source_ppi_hint`.

- [x] **`npp_rotate` / `nvjpeg2k` shared CUDA init helper** — the duplicated
  five-step CUDA driver init sequence (`cuInit → cuDeviceGet →
  cuDevicePrimaryCtxRetain → cuCtxSetCurrent → cuStreamCreate`) and the eight
  `libcuda.so` FFI declarations are extracted into `crates/gpu/src/cuda.rs` as
  `gpu::cuda::init_primary_ctx_and_stream(device_ordinal: i32) → Result<CudaInit,
  CudaInitError>`.  `NppRotator::new` maps the error to `NppRotateError(format!)`
  and `NvJpeg2kDecoder::new` maps it to `NvJpeg2kError::CudaError(code)`.
  The module is unconditionally compiled; `#[cfg_attr]` guards suppress dead-code
  lints when neither GPU feature is active.

---

## Phase 7 — Heterogeneous dispatch hub

**Goal:** dynamic per-page work routing across CPU threads, GPU decoders, and iGPU (VA-API) based on page content type, JPEG variant, and image size — rather than the current static pixel-area threshold.

### Motivation

Benchmarking revealed that the current threshold-only dispatch leaves significant wins on the table:

- **Progressive JPEG (SOF2)** falls through VA-API entirely (VAEntrypointVLD supports baseline only) — but nvJPEG handles progressive JPEG natively. Detecting the JPEG type at parse time and routing to nvJPEG instead of CPU would recover the GPU decode win on scan corpora 08–09.
- **Corpus 09 (490 progressive JPEG pages):** our CPU path finishes in ~12s (24 threads) while Poppler takes 10+ minutes single-threaded. A GPU path (nvJPEG progressive) would reduce this further.
- **Mixed workloads** waste GPU dispatch overhead on small images. A smarter router that inspects content type before dispatch avoids the penalty on dense-image-small corpora (corpus 06 is currently 0.6× on GPU).

### Design

**Content-aware dispatch signals** (inspected at parse time, zero extra I/O):

| Signal | Source | Routing hint |
|---|---|---|
| JPEG variant (SOF0 vs SOF2) | JPEG SOF marker | SOF2 → nvJPEG (not VA-API); SOF0 → VA-API eligible |
| Image pixel area | PDF dict Width×Height | Below threshold → CPU always |
| Page image count | Accumulate during parse | Many small images → CPU batch; few large → GPU |
| Dominant filter | `PageDiagnostics.dominant_filter` | DCT-heavy → prefer GPU; Flate/CCITT → CPU |

**Work-stealing queue:**

Replace the current Rayon page-parallel split with a dynamic queue where each page is a task. GPU decoder slots (nvJPEG, nvJPEG2k, VA-API) are resources claimed per-task. CPU threads take tasks when GPU slots are full. This gives:
- GPU handles progressive/large JPEG pages
- CPU handles text/vector pages concurrently
- No idle time waiting for GPU if CPU work is available

**JPEG type detection in `decode_dct`:**

Read the SOF marker byte before dispatch:
- `0xC0` (SOF0, baseline) → VA-API eligible, nvJPEG eligible
- `0xC2` (SOF2, progressive) → nvJPEG only (VA-API skipped, no wasted parse attempt)
- `0xC1`, `0xC3` (extended/lossless) → CPU only

Currently every progressive JPEG incurs a full VA-API header parse + `BadJpeg` error + fallthrough. Detecting SOF type in ~3 bytes eliminates this overhead and routes correctly.

### Work items

- [x] Extract SOF marker detection into `gpu::jpeg_sof_type()` — `crates/gpu/src/jpeg_sof.rs`; `JpegVariant { Baseline, Progressive, Other }`; zero-allocation marker scan; `#[must_use]`; 8 unit tests; shared by VA-API and dispatch
- [x] Update `decode_dct` dispatch: `jpeg_variant = gpu::jpeg_sof_type(data)` before threshold check; nvJPEG accepts `Baseline | Progressive`; VA-API accepts `Baseline` only — progressive skipped entirely; `VapiJpegDecoder::decode_sync` also guards with early return; `decode_dct_gpu` + `decode_dct_vaapi` collapsed into generic `decode_dct_gpu_path<D: GpuJpegDecoder>`
- [x] Work-stealing page queue: bounded `mpsc::sync_channel` + `rayon::scope`; `RoutingHint` extension point; back-pressure at 2× thread count; `crates/cli/src/page_queue.rs`; deadlock fix + single-thread guard; `routing_hint_from_diag` + `ProgressCtx::report` live in `page_queue.rs`
- [x] `PageDiagnostics` pre-scan pass: `pdf_interp::prescan_page` walks XObject dict + content stream operators without decoding pixels; sets `GpuJpegCandidate`/`CpuOnly` hints before enqueueing; `crates/pdf_interp/src/prescan.rs`; `count_filter` + `update_max_ppi` helpers extracted
- [x] Serial prescan loop removed from CLI render path — all pages default to `RoutingHint::Unclassified`; `routing_hint_from_diag` retained as extension point for future affinity dispatch; recovered 15-20% throughput regression
- [x] Affinity dispatch: prescan all pages sequentially before pool start; `CpuOnly` hint → `BackendPolicy::CpuOnly` override in `render_page_rgb_hinted` → `lend_decoders` skips `ensure_nvjpeg` and `DECODER_INIT_LOCK` acquisition; `GpuJpegCandidate` uses session policy unchanged; single rayon pool (soft affinity)
- [x] Benchmark: full v0.6.0 matrix on 9900X3D + RTX 5070 (`bench/v060/results.md`) and i7-8700K + RTX 2080 SUPER testbench (`bench/v060/results-testbench-i7-8700K-rtx2080super.md`).  Target corpus 08/09 GPU speedup ≥ 5× **was not met** — nvJPEG-via-`GPU_HYBRID` is 5–13× *slower* than 24-thread zune-jpeg on every DCT-heavy corpus on both machines.  Root cause and fix scoped under Phase 8.
- [ ] Re-bench with `nvjpeg-hardware` feature flag enabled — fifth matrix column (mode E, `NVJPEG_BACKEND_HARDWARE`) added 2026-05-07 to measure rather than infer.  The original v0.6.0 matrix only ran `GPU_HYBRID`; the inference that `HARDWARE` would also lose on consumer Blackwell was never directly tested.  If `HARDWARE` wins, Phase 8 changes shape.

---

## Hard blocker: NVJPG silicon access on consumer NVIDIA

Documented here once, then referred to elsewhere.

The fixed-function NVJPG hardware engine is **closed to consumer GeForce SKUs** in three independent ways:

1. **The user-space library `libnvjpeg.so` rejects `NVJPEG_BACKEND_HARDWARE` at handle creation** on consumer cards (verified on RTX 5070; the `nvjpeg-hardware` cargo feature added 2026-05-07 confirms this empirically).
2. **The open kernel module (`open-gpu-kernel-modules`) exposes NVJPG class IDs** for current architectures (`NVCDD1`, `NVCFD1`) but **deliberately does not publish the command-buffer methods**, the PRI register definitions, or the firmware that drives the engine. ~220 lines of NVJPG-related kernel code in the open repo, all of it context lifecycle / capability-table reading; zero of it is the actual decode submission path.
3. **No Vulkan extension exposes JPEG decode.** `VK_KHR_video_decode_*` covers H.264/H.265/AV1/VP9 only; Khronos has not standardised JPEG video, NVIDIA has not proposed a vendor extension. Vulkan-on-NVIDIA on consumer Blackwell exposes H.264/H.265 decode operations on the queue family, nothing else.

Reverse-engineering desktop NVJPG would take a multi-month research project (cf. Asahi Linux's GPU work for analogous scope), wouldn't transfer across architecture generations, and would land in a legally murky zone (Falcon firmware signatures). No academic or community project is publicly working on this; the ROI doesn't exist for the open-source ecosystem either.

**What this means for pdf-raster:** the only open path to GPU acceleration of JPEG-related work is the SM array via custom CUDA or Vulkan compute shaders. The fixed-function engine is unreachable from any open code path. This is the load-bearing constraint behind Phase 9's design (CPU decode + device-resident pipeline) and Phase 8's deferral.

---

## Phase 8 — Custom on-GPU parallel Huffman decoder (DEFERRED)

**Status:** Phase 0 (CPU pre-pass) shipped 2026-05-06/07. Phases 1+ deferred indefinitely as a research project, not a performance work item.

**Why deferred:**

Phase 8 was originally scoped under the assumption that we needed a GPU-resident JPEG decoder to enable device-resident pixels. That assumption was wrong: CPU decode + one strategic upload achieves the same architectural property (Phase 9) with dramatically less work.

The v0.6.0 baseline matrices on both 9900X3D + RTX 5070 and 8700K + RTX 2080 SUPER, plus the consumer-Blackwell NVJPG investigation, established that:

- Multi-thread CPU JPEG decode (zune-jpeg, AVX-512 IDCT) at 24 threads delivers ~5 GB/s aggregate. This is the path Phase 9 keeps.
- A custom parallel-Huffman GPU decoder, even built ideally per Wei et al. 2024, would *match or marginally beat* per-image latency but **lose on aggregate throughput** to 24-thread CPU. The 51× speedup over libjpeg-turbo in the paper is on A100 datacenter hardware against a single-thread CPU baseline; consumer Blackwell + 24-thread CPU is a different comparison.
- The architectural payoff (device-resident pixels enabling GPU AA / ICC / tile fill / composite) is achievable via Phase 9 without Phase 8.

**What stays in the codebase:**

The Phase 0 work (`crates/gpu/src/jpeg/`) shipped clean across three commits and 84 passing unit tests. It stays as scaffolding for a future hobbyist port and as well-tested JPEG metadata utilities that any future work might want.

**Why Phase 8 stays in the roadmap at all:**

The Weißenberger 2018 self-synchronizing parallel Huffman algorithm is genuinely beautiful CUDA work. Implementing it produces an open, redistributable artifact that demonstrates a non-obvious algorithm. If pdf-raster ever wants to run on a workload where 24 CPU threads aren't available (embedded, single-core, etc.) the GPU decoder becomes relevant. The work has long-term value as a learning project; it's just not the path to faster pdf-raster.

See `docs/superpowers/specs/2026-05-06-gpu-jpeg-pipeline.md` for the full original spec, kept on the local-only `docs/superpowers/` side because it's a research artifact, not an active spec.

---

## Phase 9 — Device-resident image cache and GPU page buffer (COMPLETE — bench gate pending)

**Spec:** `docs/superpowers/specs/2026-05-07-phase-9-device-resident-image-cache.md`

**Goal:** decoded image pixels and the page being rendered both live in VRAM for the lifetime of a render session, so the rendering hot path performs zero PCIe round-trips per image and zero decode work on cache hits.

**Why this is the right phase now:**

Three load-bearing facts from the v0.6.0 baseline + consumer-Blackwell NVJPG investigation:

1. Multi-thread CPU JPEG decode is faster than any GPU JPEG path we can ship on consumer hardware.
2. The 4 already-shipped GPU kernels (AA fill, ICC CLUT, tile fill, composite) don't carry their weight today because every kernel pays a PCIe round-trip per invocation.
3. The OCR pipeline pattern renders the same PDF multiple times; today every pass re-decodes every JPEG.

Phase 9 addresses all three at the same time: the cache makes (3) free after the first render, the device-resident page buffer makes (2) finally pay off, and (1) is no longer a problem because we're not trying to beat CPU JPEG decode — we're keeping its output in VRAM.

**Architecture (one-liner):** three-tier cache (VRAM → host RAM → disk) with content-hash dedup (BLAKE3) keying across documents, plus a device-resident page bitmap that the existing GPU kernels read and write in place.

**Work items:**

- [x] **Task 1 — `ImageData` enum, `ImageData::Cpu` variant only** (commit `f0519ca`, hardened in `8a19c3a`/`48aeecb`). `Vec<u8>` → `ImageData::Cpu(Vec<u8>)` plumbing across the renderer; `#[non_exhaustive]` enum so `Gpu` variant is a non-breaking add. The Gpu variant itself is deferred to Task 4 wiring.
- [x] **Task 2 — VRAM tier in-process** (commit `e3709ee`, hardened in `e3acb21`/`69a1cd2`). `crates/gpu/src/cache/` module behind a new `cache` feature: `DeviceImageCache` with dual-key (BLAKE3 content hash + (DocId, ObjId) alias), DashMap-backed concurrent access, LRU + refcount-pinned eviction, `InsertRequest` builder, structured `CacheError`. 8 cache tests under `cache,gpu-validation`; concurrent-insert dedup test proves no double-counted `used_bytes`.
- [x] **Task 3 — Host RAM tier** (commit `0e197c3`, hardened in `52acfdf`/`e2f750d`). `crates/gpu/src/cache/host_tier.rs` with `PinnedHostSlice<u8>` slabs, independent budget + LRU, demote-on-evict from VRAM, promote-on-hit back to VRAM. Critical fix in the hardening pass: `clone_htod` must take `&PinnedHostSlice` directly (not `as_slice().ok()?`) so cudarc records the H→D event back to the slice's internal event — otherwise `PinnedHostSlice::Drop` could free pinned memory mid-DMA. 13 cache tests; end-to-end demote+promote round-trip verifies bit-identical pixels.
- [x] **Task 4 — Device page buffer + GPU blit kernel** (commits `6ee47de`/`738ba14`/`ef67045` GPU side, `8f01c3d` AA-fill fix, `a7859e4` renderer integration). GPU side: `kernels/blit_image.cu` (16×16-block CUDA kernel with f32 inverse-CTM nearest-neighbour sampling matching the CPU path byte-for-byte), `gpu::blit` module (`InverseCtm`, `BlitBbox`, `GpuCtx::blit_image_to_buffer`, structured `BlitError`), `cache::DevicePageBuffer` (zero-init RGBA8). Renderer integration: `ImageData::Gpu(Arc<CachedDeviceImage>)` feature-gated variant, `decode_dct → ImageData::Gpu` wiring with BLAKE3 content-hash dedup + `(DocId, ObjId)` alias, per-page `DevicePageBuffer` lazy-allocated on first GPU image, source-over composite of buffer onto host bitmap at `finish()`. New `cache` feature in `pdf_interp`, `pdf_raster`, and the CLI. AA fill / ICC / tile fill / composite still use the CPU bitmap; rewiring those to read/write the device buffer is deferred (the `coverage_scratch` field in the spec). Pre-existing AA-fill `JITTER_Y` corruption (8 wrong Halton(3) values) found and fixed in `8f01c3d`. **Bench gate pending**: needs an end-to-end run on corpus 06–08 to confirm mode D ≤ 0.7× mode A.
- [x] **Task 5 — Disk tier**. `crates/gpu/src/cache/disk_tier.rs` — `<root>/<doc-hex>/<hash-hex>.bin` sidecar files with PDRF magic + version + dimensions header.  Atomic write via temp+rename; `posix_fadvise(WILLNEED)` on Linux for read-ahead; document-mtime eviction.  Env-var overrides: `PDF_RASTER_CACHE_DIR`, `PDF_RASTER_CACHE_BYTES`.  `open_session` switched to BLAKE3-of-PDF-bytes for the `DocId` so editing a PDF naturally invalidates the disk cache.  7 unit tests under `cache` feature (no GPU needed).
- [x] **Task 6 — Pre-fetcher** (commits `013219b` + `a2e81d9` hardening pass). `crates/pdf_interp/src/cache/prefetch.rs` — `spawn_prefetch(doc, cache, doc_id, config)` walks every page's `/XObject` resource dict, dedupes by `ObjId`, decodes `/DCTDecode` images on a small `std::thread` worker pool (default 2, capped at `MAX_PREFETCH_WORKERS = 16`).  Discovery is single-threaded; `seen` is a plain local `HashSet<ObjId>`.  Decoder panics caught per-image so one bad XObject doesn't kill the run.  Opt-in via `SessionConfig::prefetch`; `RasterSession.doc` upgraded to `Arc<Document>` so the prefetcher can hold its own clone.  Form-XObject contents are not recursed into; renderer decodes them on first touch.  4 unit tests under the `cache` feature (no GPU needed).

**Bench gate (pending measurement):** Phase 9 ships end-to-end if and only if (1) cache hit rate ≥ 95% on logo-heavy multi-page renders; (2) second-render time ≤ 30% of first-render time; (3) mode A doesn't regress vs the v0.6.0 baseline; (4) no OOM on corpus 09 (~1900 pages) with default budgets; (5) **mode D beats mode A on at least 3 of corpora 04–08** — this is the architectural win the whole project has been chasing.  All six implementation tasks are complete; the bench has not yet been run, so the phase is structurally done but not yet validated against the gate.

**Total scope:** ~1850 LoC new Rust + ~150 LoC new CUDA + ~400 LoC modified existing. Tasks 1+2 ship in ~5-7 days; full pipeline ~3 weeks elapsed.

---

## Phase 10 — Vulkan compute backend (PLANNED)

**Spec:** `docs/superpowers/specs/2026-05-07-phase-10-vulkan-compute-backend.md`

**Goal:** replace the CUDA-specific kernel launch and device-memory layer with a backend-abstracted layer that has both CUDA and Vulkan compute implementations, so the same algorithmic kernels run on NVIDIA, AMD, Intel, and Apple GPUs from one source tree.

**Why now (and why not before):**

Vulkan compute on the dev machine (RTX 5070, driver 580.142) was confirmed 2026-05-07 to expose Vulkan 1.4.312, conformance 1.4.1.3, full subgroup operations (the equivalent of CUDA warp intrinsics), tensor cores via `VK_KHR_cooperative_matrix`, and the same SM array CUDA uses. Cross-vendor portability is the real reason — the same SPIR-V kernel runs on AMD (RADV), Intel (ANV), Apple (MoltenVK→Metal), and Mesa lavapipe (CPU debug).

What was missing before Phase 9 was *the abstraction layer to even consider a backend swap*. Phase 9 introduces backend-agnostic shapes (`ImageData::Gpu`, `CachedDeviceImage`, `DevicePageBuffer`); Phase 10 swaps the *implementation* behind those shapes from CUDA-specific to backend-trait-driven, with concrete CUDA and Vulkan backends.

**Approach:**

- Single kernel source-of-truth in **Slang** (Khronos-supported shading language). One `.slang` file per algorithm; `slangc` compiles to PTX for CUDA backend and SPIR-V for Vulkan backend.
- `GpuBackend` trait abstracting device-memory, kernel launch, host↔device transfer, synchronisation. Two implementations: `CudaBackend` (current, refactored) and `VulkanBackend` (new, via `ash`).
- `BackendPreference` enum on `RasterOptions`: `Auto` (CUDA on NVIDIA, Vulkan elsewhere), `ForceCuda`, `ForceVulkan`, `CpuOnly`.
- The Phase 9 cache and page buffer become generic over `B: GpuBackend`.

**Work items:**

- [ ] **Task 1 — Backend trait + CUDA refactor** (~600 new + ~400 modified LoC). Extract `CudaBackend` from existing code; refactor Phase 9 cache + page buffer to be generic. Bench parity required.
- [ ] **Task 2 — Slang port of all kernels** (~1000 LoC). Translate 5 existing `.cu` files + Phase 9's `blit_image.cu` to `.slang`. nvcc compiles Slang→PTX for CUDA; SPIR-V backend used in task 3. **Riskiest task in Phase 10**; rollback plan is to keep `.cu` files alongside `.slang` until bench parity confirmed.
- [ ] **Task 3 — Vulkan backend implementation** (~1500 LoC). `VulkanBackend` impl via `ash`; pipeline cache; descriptor set management; synchronisation; integrates behind the trait. Cross-vendor smoke test on at least one AMD or Intel GPU.

**Bench gate:** Phase 10 ships if (1) CUDA path performance unchanged within ±5%; (2) Vulkan path functional on RTX 5070 with pixel-diff ≤ 1 LSB vs CUDA; (3) Vulkan timing within 15% of CUDA on RTX 5070; (4) cross-vendor proof of life on AMD or Intel.

**Total scope:** ~3100 LoC new Rust + ~1000 LoC Slang + ~400 LoC modified. Estimated ~6-8 weeks elapsed (3-4 weeks tasks 1+2; 3-4 weeks task 3).

**Sequencing:** Phase 9 must ship first. Phase 10 task 1 generifies Phase 9's cache; doing them in parallel would mean fighting merge conflicts.

---

## Testing strategy

### proptest — property-based testing for geometric primitives

`proptest` is the right tool for algorithmic correctness in the raster and path layers.
Shrinking finds the minimal failing input automatically, which is valuable for geometric
edge cases that are hard to construct by hand.

**High-value targets:**

| Area | What to test | Why |
|---|---|---|
| Path flattening | Arbitrary Bézier control points including degenerate (coincident, collinear, zero-length) | Recursive subdivision blows the stack or produces NaN coordinates on degenerate input |
| Clipping | Random clip rect × path combinations; assert output is subset of input bbox | Clip intersection logic has winding-number edge cases |
| Transformation matrix composition | Arbitrary CTM chains; assert round-trip inverse within ε | Accumulated floating-point error in nested Form XObjects |
| `cmyk_to_rgb_reflectance` | All 256⁴ is too large; proptest over random (C,M,Y,K) tuples; assert output ∈ [0,255]³ | Overflow/underflow in the subtractive formula |
| `grid_to_u8` in icc.rs | `i ∈ [0, grid_n-1]`, assert endpoints map exactly to 0 and 255 | Off-by-one at boundary nodes corrupts CLUT edges |

**Where fuzzing beats proptest** (already covered by `crates/fuzz`):

- PDF stream parsing — coverage-guided fuzzing finds parser bugs that proptest's
  random generation misses; shrinking is less valuable when the bug is a specific
  byte sequence
- CCITTFaxDecode / JBIG2Decode — malformed bitstreams need coverage guidance, not
  algebraic shrinking

**To add proptest:** reinstate `proptest = { workspace = true }` in the relevant
crate's `[dev-dependencies]` when writing the tests. The workspace declaration was
removed (commit 4334283) because it was unused; add it back alongside the actual
test code.
