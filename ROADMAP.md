# pdf-raster Roadmap

Goal: full PDF → pixels pipeline in pure Rust. Zero poppler. Zero C++ in the render path.

The raster crate is complete at the pixel level. The `pdf_interp` crate is the native renderer and is now the only CLI path. The `pdf_bridge` / poppler crate is retained as a reference baseline but is no longer linked by the CLI binary.

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
- [x] Image XObjects: FlateDecode, DCTDecode (JPEG), JPXDecode (JPEG 2000), CCITTFaxDecode Group 4, raw
- [x] Image colour spaces: DeviceRGB, DeviceGray, mask (stencil)
- [x] Soft mask (SMask) compositing on images
- [x] JavaScript rejection — hard fail on any JS entry point in the document
- [x] CLI `--native` flag wired to `pdf_interp` render path

### Blocking parity — must land before deleting pdf_bridge

Ordered by priority. Wire CLI by default is the finish line.

- [x] **ICCBased / Indexed / Separation colour spaces** — resolve_cs inspects ICC `N`, expands Indexed palettes, converts CMYK inline; Separation/DeviceN fall back to Gray
- [x] **ExtGState blend modes (`BM`)** — all 16 PDF modes parsed + threaded through make_pipe to raster compositor
- [x] **CCITTFaxDecode Group 3 (K=0)** — 1D T.4 supported via fax::decoder::decode_g3; K>0 (mixed 2D) stub
- [x] **Inline images (`BI ID EI`)** — decode_inline_image: abbreviated key/name expansion, FlateDecode/DCT/CCITT/RL/raw dispatch, wired to blit_image
- [x] **Shading (`sh`)** — Types 2 (axial) and 3 (radial) resolved; Function Types 2 (Exponential) and 3 (Stitching) evaluated; wired to shaded_fill
- [x] **Wire CLI by default** — `--native` flag removed; native is the only path; pdf_bridge dep removed from cli (crate retained for reference)

### Nice-to-have before default (won't block, but improve coverage)

- [ ] **Text render modes 4–7** — text-as-clip (glyph outlines → XPath intersection); rare in practice but used in some graphics-heavy docs
- [ ] **Type 0 / CIDFont composite fonts** — needed for CJK and other multi-byte encodings
- [ ] **Tiling patterns** — `scn` with pattern colour space; used for hatching, textures

### Phase 1 parking lot (post-shipping coverage work)

- [ ] Type 3 paint-procedure fonts
- [ ] JBIG2Decode image filter
- [ ] Optional content groups (layers / OCG)
- [ ] Annotation rendering
- [ ] Non-axis-aligned image transforms (currently bounding-box nearest-neighbour approximation)

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

## Phase 2.5 — CPU-side AVX-512 specialisation

Targeted use of AVX-512 extensions that LLVM does not auto-vectorize to, but that have direct applicability to the hot paths identified above. All items gated on `#[cfg(target_feature = "avx512...")]` with scalar fallbacks; no unsafe required beyond the intrinsic calls themselves.

- [ ] **`avx512_bitalg` AA popcount** — replace the `NIBBLE_POP` lookup-table loop in `aa_coverage()` (`fill/mod.rs`) with `_mm_popcnt_epi8` / `vpshufbitqmb`. The current loop reads 4 bytes and does 4 table lookups per output pixel; the SIMD path collapses an entire AaBuf row (AA_SIZE=4 sub-rows, `bitmap_width/2` bytes) into a single `VPOPCNTB` + horizontal reduce. Expected 4–8× speedup on the AA fill path.

- [ ] **`avx512vnni` ICC colour matrix** — when the ICC CMYK→RGB transform lands (Phase 4 item 4), use `_mm512_dpbusds_epi32` (int8 dot product, 4-element accumulate per cycle) for the 3×4 matrix multiply per pixel. Saturating int8 arithmetic matches ICC profile precision requirements. Falls back to the scalar f32 path on non-AVX-512VNNI targets.

- [ ] **`movdir64b` non-temporal solid fill** — add a non-temporal store path to `draw_span` / `render_span` for large solid fills (span width above a cache-line threshold, e.g. > 256 px) where the output bitmap will not be read back immediately. `MOVDIR64B` writes 64 bytes atomically without polluting L3 with write-only data, preserving the edge table's residency in the 128 MiB V-Cache. Gate on span width; fall back to the existing AVX-512 store for small spans.

- [ ] **`cat_l3` / `cdp_l3` cache partitioning** — document (and optionally wire) Linux `resctrl` to reserve a fixed L3 partition for the edge table across page renders in a server/batch context. Not a code change — a deployment note in the benchmarking section.

---

## Phase 3 — Coverage completeness

Track and close fidelity gaps against pdftoppm once the native path is default.

- [ ] Coons patch / tensor mesh shading (Type 4–7)
- [ ] Non-axis-aligned image transforms (currently nearest-neighbour bounding-box approximation)
- [ ] Halftone screens for CMYK separation output
- [ ] PDF transparency groups (isolated / non-isolated / knockout) at the page level

---

## Phase 4 — GPU acceleration (cudarc)

Unblocked by Phase 1 completion (poppler must be gone first). **Phase 1 is complete — Phase 4 is now unblocked.**

**Hardware context (RTX 5070, CC 12.0 Blackwell, 12 GB GDDR7):** cudarc 0.19 is already wired in `crates/gpu` with two kernels (Porter-Duff composite, soft mask) and CPU fallbacks. Target `sm_120` PTX. The GPU dispatch threshold is currently 500k pixels — validate this against actual transfer latency on this machine once the native path is hot. Do **not** use wgpu/Vello's GPU backend — CUDA is strictly better for a batch server pipeline on NVIDIA hardware.

**Do not use DLSS, MSAA, CSAA, or TAA.** These are real-time game rendering features (temporal, triangle-mesh, depth-buffer dependent) and have no applicability to batch PDF rasterisation.

### Priority order

**1. nvJPEG image decoding — highest value, implement first** ✓ COMPLETE

For scan-heavy corpora (JPEG/JBIG2/CCITT image layers + thin OCR text overlay), image decoding dominates wall-clock time. nvJPEG decodes at ~10 GB/s on the RTX 5070; the CPU JPEG path (libjpeg via DCTDecode) is 10–20× slower. No rasterizer changes required — wire nvJPEG into the existing `blit_image` path behind a feature flag.

- [x] `gpu::nvjpeg` module: minimal raw FFI surface (no bindgen); NvJpeg context + NvJpegDecoder safe wrapper; decode_sync handles cuStreamSynchronize
- [x] DCTDecode dispatch: image area ≥ GPU_JPEG_THRESHOLD_PX (512×512) → nvJPEG; else CPU zune-jpeg; CMYK JPEG falls through to CPU
- [x] Feature flags: `gpu/nvjpeg` + `pdf_interp/nvjpeg`; zero-cost when disabled; pdf_interp maintains unsafe_code = "deny"
- [ ] nvJPEG2000 for JPXDecode (JPEG 2000); lower priority than baseline JPEG

**2. GPU supersampled AA — replaces CPU 4× scanline AA**

The current `render_aa_line` + nibble-popcount AA is the weakest part of the CPU pipeline. Replace it with a CUDA kernel doing **jittered supersampling** at 64 samples/pixel using warp-level ballot reduction:

```cuda
// One warp (32 threads) per output pixel
bool inside = winding_test(segs, n_segs, jittered_sample(px, py, threadIdx.x));
int coverage = __popc(__ballot_sync(0xFFFFFFFF, inside));
output[py * width + px] = (uint8_t)((coverage * 255) / 32);
```

`__ballot_sync` + `__popc` gives 32-sample coverage in a single warp cycle. With 2 warps/pixel: 64 samples. Quality far exceeds the CPU 4×4 grid; cost is lower because the 4352 CUDA cores run all pixels in parallel. The CPU AA path remains as fallback below the dispatch threshold.

- [ ] CUDA kernel: jittered 64-sample winding test per pixel
- [ ] Warp-ballot reduction: `__ballot_sync` + `__popc` for coverage count
- [ ] Wire into `render_aa_line` dispatch: if fill area > threshold → GPU kernel
- [ ] Validate quality vs CPU AA on pixel-diff benchmark

**3. Tile-parallel fill rasterisation — GPU path only**

Tile records (sorted by (y, x)) are the natural GPU work unit. One thread block per tile strip, independent coverage accumulation per tile, no warp divergence. The sort is done on the GPU via CUB radix sort (ships with CUDA toolkit).

This is the correct implementation of the ROADMAP's original "sparse tile rasterisation" item — done once, for the GPU, where it actually matters. The CPU scanline scanner is retained unchanged for fills below the dispatch threshold (large solid fills are already near-memset speed on the CPU via AVX-512 `render_span`).

- [ ] Tile record format: `{x: u16, y: u16, packed: u32}` (8 bytes, matches vello_common layout)
- [ ] GPU segment upload: XPath edge list → device buffer via cudarc
- [ ] CUB radix sort: sort tile records by (y << 16 | x) on device
- [ ] Fill kernel: one thread block per strip, analytical trapezoid coverage (vello algorithm)
- [ ] Winding kernel: accumulate integer winding across tile rows using prefix sum
- [ ] Integrate with `fill_impl_parallel` dispatch: if `vector_antialias && area > threshold` → GPU

**4. ICC colour transforms (cuBLAS / custom kernel)**

ICC profile evaluation is a per-pixel matrix multiply. For DeviceCMYK → DeviceRGB conversion on large images, a CUDA kernel with cuBLAS GEMM can saturate memory bandwidth. Medium priority — only visible on CMYK-heavy documents.

- [ ] Kernel: 4→3 channel matrix multiply per pixel, fused with image decode
- [ ] Unblocked by: Phase 1 colour spaces (complete)

**5. OptiX BVH for complex paths — low priority, evaluate later**

RT cores on Blackwell provide hardware BVH traversal. For pages with thousands of path segments, an OptiX any-hit kernel computing winding numbers via ray casting would be faster than the tile rasteriser for very complex geometry. In practice, most PDF pages have O(100) path segments, not O(10000), so this is unlikely to be the bottleneck. Evaluate only after profiling shows complex path rasterisation in the flamegraph.

### GPU dispatch table

| Target | Value | Unblocked by |
|---|---|---|
| nvJPEG image decoding | **Highest** — scan-heavy corpora | Phase 1 image pipeline ✓ |
| GPU supersampled AA (warp ballot) | High — quality + speed | GPU segment upload |
| Tile-parallel fill rasterisation | High — sparse/complex paths | GPU segment upload |
| ICC colour transforms | Medium — CMYK docs | Phase 1 colour spaces ✓ |
| OptiX BVH winding test | Low — only extreme geometry | Tile rasteriser |
| Blend / composite | Low — already fast on CPU | Phase 2 perf work ✓ |

FreeType text rendering is **not** a GPU target — hinting is sequential per glyph. A GPU text path requires a GPU-resident rasteriser (SDF atlas or Slug algorithm) and is a separate major project.

---

## Benchmarking

```bash
# Native vs pdftoppm (run after --native is the default)
hyperfine --warmup 3 \
  'target/release/pdf-raster -r 150 tests/fixtures/ritual-14th.pdf /tmp/out' \
  '/usr/bin/pdftoppm -r 150 tests/fixtures/ritual-14th.pdf /tmp/ref'

# Pixel diff vs reference
compare -metric AE /tmp/ref-01.ppm /tmp/out-01.ppm /dev/null

# Flamegraph
CARGO_PROFILE_RELEASE_DEBUG=true flamegraph -o /tmp/flame.svg \
  -- target/release/pdf-raster -r 150 tests/fixtures/cryptic-rite.pdf /tmp/out

# Fill microbenchmark (raster crate only)
RUSTFLAGS="-C target-cpu=native" cargo run -p bench --release -- --iters 30 --stars 200

# GPU AA benchmark (once Phase 4 item 2 lands)
RUSTFLAGS="-C target-cpu=native" cargo run -p bench --release -- --iters 30 --stars 200 --aa gpu

# L3 occupancy monitoring via hardware CQM counters (9900X3D specific)
# Requires kernel resctrl mount: mount -t resctrl resctrl /sys/fs/resctrl
# Reports per-process L3 occupancy in bytes — use to verify edge table stays
# resident in the 128 MiB V-Cache across sequential page renders.
# cat /sys/fs/resctrl/mon_data/mon_L3_XX/llc_occupancy
```

Current pixel diff vs poppler (--native, 150 dpi):
- `cryptic-rite` page 1: ~1.8 %
- `ritual-14th` page 1: ~1.2 %
