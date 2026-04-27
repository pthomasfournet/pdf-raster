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

### In progress / next

- [ ] **ICCBased / Indexed / Separation colour spaces** — `SetFillColorSpace` is currently a no-op; affects image fidelity and some drawn content
- [ ] **CCITTFaxDecode Group 3 (K≥0)** — Group 4 (K<0) done; Group 3 stub
- [ ] **ExtGState blend modes (`BM`)** — only `Normal` mapped; `Multiply`, `Screen`, `Overlay`, etc. pending
- [ ] **Inline images (`BI ID EI`)** — stub; log only
- [ ] **Shading (`sh`)** — requires shading resource lookup and gradient rasterisation
- [ ] **Text render modes 4–7** — text-as-clip (glyph outlines → XPath intersection)
- [ ] **Wire CLI by default** — remove `--native` flag; make native the only path; delete `pdf_bridge`

### Phase 1 parking lot (real-world coverage, not blocking parity)

- [ ] Type 0 / CIDFont composite fonts
- [ ] Type 3 paint-procedure fonts
- [ ] JBIG2Decode image filter
- [ ] Tiling patterns (`scn` with pattern colour space)
- [ ] Optional content groups (layers / OCG)
- [ ] Annotation rendering

---

## Phase 2 — Raster performance

Do not start until the native CLI path is the default (pdf_bridge deleted). The raster crate is not yet in the hot path — optimising it now is optimising the wrong thing.

- [ ] **Eliminate per-span heap allocations** — `PipeSrc::Solid` and pattern scratch bufs allocate a `Vec` per span; replace with thread-local grow-never-shrink buffers
- [ ] **u16×16 compositing inner loop** — process 16 pixels/iter as `[u16; 16]`, replace `div255` with `(x + 255) >> 8`, let LLVM auto-vectorise under `-C target-cpu=native`
- [ ] **Fixed-point edge stepping (FDot16)** — add `x_cur: i32` + `dx: i32` (16.16) to `XPathSeg`; hot loop does `x_cur += dx` instead of `x0 + (y−y0)×dxdy` (eliminates f64 multiply per edge per scanline)
- [ ] **Sparse tile rasterisation** — replace flat SoA edge table + per-scanline sweep with tile records sorted by (y, x); only non-empty tiles touched (large win for sparse paths, à la Vello)

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

| Target | Value | Unblocked by |
|---|---|---|
| Tile-parallel rasterisation | High | Phase 2 sparse tiles |
| Image decoding (nvJPEG / cuvid) | Medium | Phase 1 image pipeline |
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
