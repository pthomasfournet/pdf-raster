# pdf-raster Roadmap

Goal: full PDF → pixels pipeline in pure Rust. Zero poppler. Zero C++ in the render path.

The raster crate is complete at the pixel level. The `pdf_interp` crate is the in-progress native renderer. The `pdf_bridge` / poppler path exists only as a reference baseline until the native path reaches parity.

---

## Phase 1 — Native PDF interpreter (in progress)

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
- [ ] **Inline images (`BI ID EI`)** — stub; some generators use these heavily
- [ ] **Shading (`sh`)** — gradients visually obvious when missing; needs shading resource lookup + axial/radial rasterisation wired through
- [ ] **Wire CLI by default** — remove `--native` flag; make native the only path; delete `pdf_bridge`

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

## Phase 2 — Raster performance

Do not start until the native CLI path is the default (pdf_bridge deleted). The raster crate is not yet in the hot path — optimising it now is optimising the wrong thing.

**Hardware context (Ryzen 9 9900X3D):** 128 MiB 3D V-Cache means edge tables and scanline sweep structures for most real-world documents fit in L3. Algorithms that are cache-bound on a normal CPU are compute-bound here — this shifts the priority order vs. generic advice. Sparse tile rasterisation has an outsized benefit because it maximises L3 utilisation. AVX-512 (F/BW/VL/VNNI/BF16/VPOPCNTDQ) is fully available; target `avx512f,avx512bw,avx512vl` with `-C target-cpu=native`.

- [ ] **Eliminate per-span heap allocations** — `PipeSrc::Solid` and pattern scratch bufs allocate a `Vec` per span; replace with thread-local grow-never-shrink buffers
- [ ] **u16×16 compositing inner loop** — process 16 pixels/iter as `[u16; 16]`, replace `div255` with `(x + 255) >> 8`; target AVX-512BW for 32-pixel-wide SIMD under `-C target-cpu=native`
- [ ] **Fixed-point edge stepping (FDot16)** — add `x_cur: i32` + `dx: i32` (16.16) to `XPathSeg`; hot loop does `x_cur += dx` instead of `x0 + (y−y0)×dxdy` (eliminates f64 multiply per edge per scanline)
- [ ] **Sparse tile rasterisation** — replace flat SoA edge table + per-scanline sweep with tile records sorted by (y, x); only non-empty tiles touched; reference: vello_cpu sparse_strips/. Especially high value with 3D V-Cache.

---

## Phase 3 — Coverage completeness

Track and close fidelity gaps against pdftoppm once the native path is default.

- [ ] Coons patch / tensor mesh shading (Type 4–7)
- [ ] Non-axis-aligned image transforms (currently nearest-neighbour bounding-box approximation)
- [ ] Halftone screens for CMYK separation output
- [ ] PDF transparency groups (isolated / non-isolated / knockout) at the page level

---

## Phase 4 — GPU acceleration (cudarc)

Unblocked by Phase 1 completion (poppler must be gone first).

**Hardware context (RTX 5070, CC 12.0 Blackwell, 12 GB GDDR7):** cudarc 0.19 is already wired in `crates/gpu` with two kernels (Porter-Duff composite, soft mask) and CPU fallbacks. Target `sm_120` PTX. The GPU dispatch threshold is currently 500k pixels — validate this against actual transfer latency on this machine once the native path is hot. Do **not** use wgpu/Vello's GPU backend — CUDA is strictly better for a batch server pipeline on NVIDIA hardware.

**Corpus note:** if the workload is predominantly scanned pages (JPEG/JBIG2/CCITT image layers + thin OCR text overlay), image decoding throughput will dominate wall-clock time, not rasterisation. Profile first — nvJPEG may be the highest-value GPU target, not tile rasterisation.

| Target | Value | Unblocked by |
|---|---|---|
| Tile-parallel rasterisation | High | Phase 2 sparse tiles |
| Image decoding (nvJPEG / cuvid) | High if scan-heavy corpus | Phase 1 image pipeline |
| ICC colour transforms | Medium | Phase 1 colour spaces |
| Blend / composite | Low | Phase 2 perf work |

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
```

Current pixel diff vs poppler (--native, 150 dpi):
- `cryptic-rite` page 1: ~1.8 %
- `ritual-14th` page 1: ~1.2 %
