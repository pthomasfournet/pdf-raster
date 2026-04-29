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

### Open: inline images never use GPU decoders

`decode_inline_image` in `pdf_interp/src/resources/image.rs` does not accept GPU decoder parameters — all inline `DCTDecode` and `JPXDecode` images go through the CPU path regardless of image area or enabled features. Most inline images are small (thumbnails, icons) so the threshold would rarely trigger, but the gap means the GPU path is structurally incomplete for inline streams. Fix: add the same `#[cfg]`-gated decoder parameters to `decode_inline_image` as `resolve_image` has.

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

- [ ] **`cat_l3` / `cdp_l3` cache partitioning** — deployment note: Linux `resctrl` to reserve a fixed L3 partition for edge tables in batch/server context.

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
| `ritual-14th.pdf` | 116 KB | Light vector + text, 41 pp | 213 ms | 262 ms | **1.2×** |
| `cryptic-rite.pdf` | 281 KB | Mixed vector + images, 7 pp | 109 ms | 291 ms | **2.7×** |
| `kt-r2000.pdf` | 2.1 MB | Dense vector / complex paths, 34 pp | 495 ms | 1507 ms | **3.0×** |
| `xxxii-sr.pdf` | 11 MB | Mixed; image-heavy | 5.2 s | 44.4 s | **8.5×** |
| `scotch-rite-illustrated.pdf` | 50 MB | Scan-heavy JPEG/JPEG2K | 17.2 s | 155.7 s | **9.1×** |

The scan-heavy corpus (JPEG/JPEG2K) shows the largest gap because nvJPEG + nvJPEG2K GPU decode replaces the CPU libjpeg/OpenJPEG path. The vector-only fixture (ritual-14th) shows the smallest gap — that workload is entirely CPU path-fill and text.

### Pixel diff vs poppler

`compare -metric AE` on 3 pages of `ritual-14th` at 150 DPI. Same page dimensions (700×1050 px). AE of 0.9–17% — entirely explained by sub-pixel anti-aliasing differences at glyph edges (amplified diff shows ghosted text, no structural content difference). This is expected for two independent renderers with different AA strategies.

### ~~Known gap: page rotation (`/Rotate`)~~ — RESOLVED (commit `82efbe5`)

`/Rotate` and `CropBox` are fully handled: `pdf_interp::page_size_pts` reads
`CropBox` (falling back to `MediaBox`), normalises `/Rotate` to 0/90/180/270,
and swaps dimensions for 90°/270° rotations.  `PageRenderer::new_scaled`
applies the matching CTM so all four rotation values produce correctly-oriented
output.  `kt-r2000.pdf` page 1 (was `/Rotate: 270` portrait) now renders as
landscape, matching poppler.

### Known gap: `UserUnit` scaling (PDF 1.6+)

`UserUnit` is a Page dictionary key that scales the default user-space unit
from 1/72 inch to `UserUnit/72` inches.  `page_size_pts` does not read it; a
`UserUnit: 2.0` page renders at half the intended physical size and
`RenderedPage.dpi` is wrong (actual resolution = `opts.dpi × user_unit`).
Rare in practice — affects some large-format and engineering PDFs.  Fix:
multiply `w_pts`/`h_pts` by `UserUnit` in `page_size_pts`, expose
`effective_dpi` on `RenderedPage` or `PageGeometry`.  Return an error (not a
silent clamp) for `UserUnit` values outside the valid range defined by the spec.

### Fixture inventory

| File | Size | Character |
|---|---|---|
| `ritual-14th.pdf` | 116 KB | Light vector + text |
| `cryptic-rite.pdf` | 281 KB | Mixed vector + images |
| `kt-r2000.pdf` | 2.1 MB | Dense vector / complex paths |
| `xxxii-sr.pdf` | 11 MB | Mixed; image-heavy pages |
| `scotch-rite-illustrated.pdf` | 50 MB | Scan-heavy; primary JPEG/JPEG2K workload |

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
  "$BIN -r 150 tests/fixtures/scotch-rite-illustrated.pdf /tmp/out" \
  'pdftoppm -r 150 tests/fixtures/scotch-rite-illustrated.pdf /tmp/ref'

# Pixel diff vs poppler reference (ImageMagick AE metric)
pdftoppm -r 150 tests/fixtures/ritual-14th.pdf /tmp/ref
env $LD_LIB $BIN -r 150 tests/fixtures/ritual-14th.pdf /tmp/out
for i in /tmp/ref-*.ppm; do
  n=$(basename $i .ppm | sed 's/ref-//')
  ae=$(compare -metric AE $i /tmp/out-${n}.ppm /dev/null 2>&1)
  echo "$(basename $i): AE=$ae"
done

# Flamegraph — find the new bottleneck after GPU image decode is wired
CARGO_PROFILE_RELEASE_DEBUG=true env $LD_LIB \
flamegraph -o /tmp/flame.svg -- \
  $BIN -r 150 tests/fixtures/scotch-rite-illustrated.pdf /tmp/out

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

## Phase 6 — Integration hardening and OCR pipeline fit

### Goal

Make pdf-raster the drop-in replacement for the pdftoppm + Leptonica preprocessing
stack in the mss OCR pipeline.  The rasterise + deskew path is feature-complete;
Phase 6 closes the remaining gaps before the first production integration.

### Open work items

- [ ] **`UserUnit` support** — `page_size_pts` does not read the `UserUnit` Page
  dictionary key (PDF 1.6+, scales user-space from 1/72 in to `UserUnit/72` in).
  Fix: multiply `w_pts`/`h_pts` by `UserUnit`; expose `effective_dpi` on
  `RenderedPage` (= `opts.dpi × user_unit`) so callers pass the right value to
  `tesseract::set_source_resolution`.  Reject `UserUnit` outside [0.1, 10.0]
  with `RasterError::InvalidPageGeometry` rather than silently producing a
  wrong-scale bitmap.  Affects large-format and engineering PDFs; rare in the
  Gallica/GODF corpus but a latent correctness bug.

- [ ] **`RenderDiagnostics` on `RenderedPage`** — add a lightweight metadata
  struct exposing information the renderer already has at decode time:
  `{ is_scan: bool, dominant_filter: ImageFilter, has_vector_text: bool }`.
  `is_scan`: true when all image XObjects use `DCTDecode` or `CCITTFaxDecode`
  and no text operators are present.  `dominant_filter`: most-used image
  compression type on the page (`DCT`, `JBIG2`, `JPX`, `Raw`, `Mixed`).
  `has_vector_text`: any `Tj`/`TJ`/`'`/`"` operators executed.
  This lets the OCR caller make better routing decisions (force_ocr, PSM, DPI)
  without a separate post-hoc page analysis pass.

- [ ] **Pipelined render + OCR** — `raster_pdf` returns an iterator but mss-pdf
  collects it into `Vec` before OCR starts, keeping the sequential bottleneck.
  Add a `render_channel` API: `fn render_channel(path, opts, capacity) ->
  Receiver<(u32, Result<RenderedPage>)>` backed by a Rayon-spawned producer.
  The consumer (Tesseract) processes pages as they arrive; the producer renders
  ahead up to `capacity` pages.  Halves peak memory on large books; hides GPU
  decode latency behind Tesseract inference on the previous page.

- [ ] **DPI auto-selection hint** — expose a `suggested_dpi` on `RenderedPage`
  or `RenderDiagnostics` based on the source image PPI embedded in the PDF.
  Many Gallica scans encode images at 62–67 PPI; rendering at 300 DPI upsamples
  4× without adding information.  The hint lets the caller choose: render at
  source PPI for speed, or override for quality.  Read-only hint — no behaviour
  change, caller decides.

- [ ] **`npp_rotate` / `nvjpeg2k` shared CUDA init helper** — `NppRotator::new`
  and `NvJpeg2kDecoder::new` duplicate the five-step CUDA init sequence
  (`cuInit → cuDeviceGet → cuDevicePrimaryCtxRetain → cuCtxSetCurrent →
  cuStreamCreate`) with divergent null checks.  Extract into
  `gpu::cuda::init_stream(device_ordinal) -> Result<CudaStream>` shared by both.
